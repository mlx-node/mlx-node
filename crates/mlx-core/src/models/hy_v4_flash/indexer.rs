//! Hy4 k-pool indexer (llama.cpp PR #32517 `build_kpool_select`): picks the
//! cells an indexed layer attends.
//!
//! ```text
//! tok_k  = rms(attn_k·x)·k_norm                     [n, d] f32, not roped
//! group g = positions [kp·g, kp·g + kp), closed once all members are in
//! pooled = Σ_slot (Wv·tok_k)·softmax_slot(Wg·tok_k + ape)   per channel
//! key_g  = rope(rms(pooled)·compressor_norm, kp·g)  -> PooledBuf, f16, in place
//! iq     = rope(q_b·qr, t)  [n, H, d]               w = proj·x · (H·d)^-½
//! score[t, g] = Σ_h w[t, h]·relu(iq[t, h]·key_g)    g < visible(t)
//! cells(t) = kp cells of each of the top n_top groups + own tail kp·visible(t) ..= t
//! ```
//!
//! With `select_tail` a group is visible to `t` once it ends before `t`
//! (`visible = t / kp`), else once it ends at or before `t`. Rows with at most
//! `n_top` visible groups attend every earlier cell. Other rows pick on the
//! GPU: score descending, ties to the lowest group. Groups close before any
//! row of the call is scored, so a chunked call picks what one call picks.

use std::ops::Range;

use napi::{Error, Result};

use super::cache::{LatentBuf, WINDOW_SLACK};
use super::config::{Attn, HyV4FlashConfig};
use super::moe::{RowKind, per_row, qlinear};
use super::weights::IndexerWeights;
use crate::array::{DType, MxArray};
use crate::models::deepseek_v41::attention::{linear, rms_norm};
use crate::models::deepseek_v41::indexer::INDEX_CHUNK_BYTES;
use crate::models::deepseek_v41::ops::{apply_partial_rope, inv_freqs, rope_tables};
use crate::models::deepseek_v41::store::Dense;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 indexer: {msg}"))
}

/// The sizes and rules one layer's indexer runs with.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexerGeometry {
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub kpool: usize,
    /// History groups picked by score.
    pub n_top: usize,
    /// The query's own group is attended through its tail cells.
    pub select_tail: bool,
    pub eps: f32,
    pub weight_scale: f32,
    pub rope_theta: f32,
}

/// What one call `[start, start + n)` does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallPlan {
    /// Groups the call closes (pooled rows written).
    pub groups: Range<usize>,
    /// Open positions carried in from earlier calls.
    pub open_in: Range<usize>,
    /// Open positions carried out to the next call.
    pub tail_out: Range<usize>,
    /// Rows that attend every earlier cell.
    pub dense: Range<usize>,
    /// Rows that pick groups by score.
    pub sparse: Range<usize>,
    /// Index row width of the call.
    pub kmax: usize,
    /// Pooled keys scored (every group closed after the call).
    pub cols: usize,
}

impl IndexerGeometry {
    /// The indexer of layer `layer` of `cfg`; refuses a sliding layer.
    pub fn of(cfg: &HyV4FlashConfig, layer: usize) -> Result<Self> {
        let l = cfg
            .layers
            .get(layer)
            .ok_or_else(|| bad(format!("layer {layer} of {}", cfg.layers.len())))?;
        if l.attn != Attn::Indexed {
            return Err(bad(format!("layer {layer} is a sliding-window layer")));
        }
        let ix = &cfg.indexer;
        Ok(Self {
            n_heads: ix.n_heads,
            head_dim: ix.head_dim,
            rope_dim: cfg.mla.rope_dim,
            kpool: ix.kpool,
            n_top: ix.n_top_pool(),
            select_tail: ix.select_tail,
            eps: cfg.norm_eps,
            weight_scale: ix.weight_scale(),
            rope_theta: l.rope_theta,
        })
    }

    /// Groups query `t` may pick from.
    pub fn visible(&self, t: usize) -> usize {
        if self.select_tail {
            t / self.kpool
        } else {
            (t + 1) / self.kpool
        }
    }

    pub fn picked(&self, t: usize) -> usize {
        self.visible(t).min(self.n_top)
    }

    /// The own-group cells query `t` attends besides its picked groups.
    pub fn tail_cells(&self, t: usize) -> Range<usize> {
        self.kpool * self.visible(t)..t + 1
    }

    /// Cells query `t` attends.
    pub fn cells(&self, t: usize) -> usize {
        self.kpool * self.picked(t) + self.tail_cells(t).len()
    }

    /// Query `t` attends every earlier cell.
    pub fn is_dense(&self, t: usize) -> bool {
        self.visible(t) <= self.n_top
    }

    /// The first query that picks by score.
    pub fn first_sparse(&self) -> usize {
        self.kpool * (self.n_top + 1) - usize::from(!self.select_tail)
    }

    /// Index slots of the own-group tail.
    pub fn tail_slots(&self) -> usize {
        self.kpool - usize::from(!self.select_tail)
    }

    /// Index row width of a call with a scored row.
    pub fn kmax(&self) -> usize {
        self.kpool * self.n_top + self.tail_slots()
    }

    pub fn plan(&self, start: usize, n: usize) -> CallPlan {
        let kp = self.kpool;
        let end = start + n;
        let (before, after) = (start / kp, end / kp);
        let split = self.first_sparse().clamp(start, end);
        CallPlan {
            groups: before..after,
            open_in: kp * before..start,
            tail_out: kp * after..end,
            dense: start..split,
            sparse: split..end,
            kmax: if split < end { self.kmax() } else { end },
            cols: after,
        }
    }
}

/// Bytes of one score row over `cols` pooled keys (`[heads, cols]` f32).
pub fn score_row_bytes(heads: usize, cols: usize) -> usize {
    4 * heads * cols
}

/// Query rows per score sub-chunk within `budget`.
pub fn score_rows(heads: usize, cols: usize, budget: usize) -> usize {
    (budget / score_row_bytes(heads, cols).max(1)).max(1)
}

/// Index rows of the dense queries `start..start + n`: row `t` lists
/// `0..=t`, padded with `-1` to `width` slots.
pub fn dense_rows(start: usize, n: usize, width: usize) -> Result<Vec<i32>> {
    let mut idx = Vec::with_capacity(n * width);
    for t in start..start + n {
        if t >= width {
            return Err(bad(format!("dense row {t} does not fit {width} slots")));
        }
        idx.extend((0..=t).map(|k| k as i32));
        idx.extend(std::iter::repeat_n(-1, width - t - 1));
    }
    Ok(idx)
}

/// `[r, n_top]` i32 ids of the groups each score row picks, ascending, `-1`
/// where a row has fewer visible groups. Row `i` sees groups `<
/// visible[i]`; it takes the `n_top` best by score, ties to the lowest
/// group. `scores [r, cols]` f32, `cols >= n_top`.
pub fn pick_groups(scores: &MxArray, visible: &[i32], n_top: usize) -> Result<MxArray> {
    let d = scores.shape()?.as_ref().to_vec();
    let &[r, cols] = d.as_slice() else {
        return Err(bad(format!("scores must be [rows, groups], got {d:?}")));
    };
    let top = i64::try_from(n_top).map_err(|_| bad("n_top exceeds i64"))?;
    if r != visible.len() as i64 || top == 0 || cols < top {
        return Err(bad(format!(
            "scores {d:?} for {} rows and {n_top} picks",
            visible.len()
        )));
    }
    let i32_of = |v: i64| i32::try_from(v).map_err(|_| bad(format!("{v} exceeds i32")));
    let kth = i32_of(top - 1)?;
    let ids = MxArray::arange(0.0, cols as f64, None, Some(DType::Int32))?.reshape(&[1, cols])?;
    let vis = ids.less(&MxArray::from_int32(visible, &[r, 1])?)?;
    let inf = MxArray::from_float32(&[f32::INFINITY], &[1])?;
    // ascending order of `neg` is the pick order; invisible groups sort last
    let neg = vis.where_(&scores.astype(DType::Float32)?.negative()?, &inf)?;
    let th = neg
        .partition(kth, Some(-1))?
        .slice(&[0, top - 1], &[r, top])?;
    let better = neg.less(&th)?;
    let tie = neg.equal(&th)?.logical_and(&vis)?;
    let need = MxArray::from_int32(&[i32_of(top)?], &[1])?
        .sub(&better.astype(DType::Int32)?.sum(Some(&[-1]), Some(true))?)?;
    let rank = tie.astype(DType::Int32)?.cumsum(-1)?;
    let sel = better.logical_or(&tie.logical_and(&rank.less_equal(&need)?)?)?;
    let past = MxArray::from_int32(&[i32_of(cols)?], &[1])?;
    let key = sel.where_(&ids, &ids.add(&past)?)?;
    let got = key
        .partition(kth, Some(-1))?
        .slice(&[0, 0], &[r, top])?
        .sort(Some(-1))?;
    got.less(&past)?
        .where_(&got, &MxArray::from_int32(&[-1], &[1])?)
}

/// Group ids `[r, k]` (`-1` = none) -> their cells `[r, k·kpool]`.
pub fn group_cells(ids: &MxArray, kpool: usize) -> Result<MxArray> {
    let r = ids.shape_at(0)?;
    let k = ids.shape_at(1)?;
    let kp = kpool as i64;
    let ids = ids.reshape(&[r, k, 1])?;
    let base = ids.mul(&MxArray::from_int32(&[kpool as i32], &[1])?)?;
    let slot =
        MxArray::arange(0.0, kpool as f64, None, Some(DType::Int32))?.reshape(&[1, 1, kp])?;
    ids.greater_equal(&MxArray::from_int32(&[0], &[1])?)?
        .where_(&base.add(&slot)?, &MxArray::from_int32(&[-1], &[1])?)?
        .reshape(&[r, k * kp])
}

/// `Σ_h w[r, h]·relu(q[r, h]·key_g)`: `q [r, H, d]`, `w [r, H]`, `keys
/// [cols, d]` -> `[r, cols]` f32.
pub fn scores(q: &MxArray, w: &MxArray, keys: &MxArray) -> Result<MxArray> {
    let r = q.shape_at(0)?;
    let h = q.shape_at(1)?;
    let d = q.shape_at(2)?;
    let cols = keys.shape_at(0)?;
    let sc = q
        .astype(DType::Float32)?
        .reshape(&[r * h, d])?
        .matmul(&keys.astype(DType::Float32)?.transpose(Some(&[1, 0]))?)?
        .reshape(&[r, h, cols])?;
    sc.maximum(&MxArray::from_float32(&[0.0], &[1])?)?
        .mul(&w.astype(DType::Float32)?.reshape(&[r, h, 1])?)?
        .sum(Some(&[1]), Some(false))
}

/// `tok_k` rows of positions `len - held .. len` (the open group and up to
/// `WINDOW_SLACK` positions before it), f32, a fresh array.
#[derive(Clone)]
pub struct TokTail {
    kpool: usize,
    width: i64,
    keep: usize,
    rows: Option<MxArray>,
    held: usize,
    len: usize,
}

impl TokTail {
    fn new(g: &IndexerGeometry) -> Self {
        Self {
            kpool: g.kpool,
            width: g.head_dim as i64,
            keep: g.kpool - 1 + WINDOW_SLACK,
            rows: None,
            held: 0,
            len: 0,
        }
    }

    /// Positions seen.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `tok_k` rows held now.
    pub fn held(&self) -> usize {
        self.held
    }

    /// Most `tok_k` rows ever held.
    pub fn keep(&self) -> usize {
        self.keep
    }

    fn nbytes(&self) -> usize {
        self.rows.as_ref().map_or(0, MxArray::nbytes)
    }

    /// Positions of the open group when `groups` groups are pooled.
    fn open(&self, groups: usize) -> Range<usize> {
        self.kpool * groups..self.len
    }

    /// The last `n` held rows (`n <= held`).
    fn last_rows(&self, n: usize) -> Result<Option<MxArray>> {
        if n == 0 {
            return Ok(None);
        }
        let held = self.held;
        if n > held {
            return Err(bad(format!("{n} open rows needed, {held} held")));
        }
        let r = self
            .rows
            .as_ref()
            .ok_or_else(|| bad("tok_k rows missing"))?;
        Ok(Some(r.slice(
            &[(held - n) as i64, 0],
            &[held as i64, self.width],
        )?))
    }

    /// Appends `tok [n, d]` (positions `len..len + n`).
    fn push(&mut self, tok: &MxArray) -> Result<()> {
        let n = tok.shape_at(0)? as usize;
        let hold = (self.held + n).min(self.keep);
        let from_tok = hold.min(n);
        let new = tok.slice(&[(n - from_tok) as i64, 0], &[n as i64, self.width])?;
        let rows = match self.last_rows(hold - from_tok)? {
            Some(old) => MxArray::concatenate(&old, &new, 0)?,
            None => new,
        }
        .deep_copy()?;
        MxArray::eval_arrays_with_context(&[&rows], "hy_v4_flash indexer tail")?;
        self.rows = Some(rows);
        self.held = hold;
        self.len += n;
        Ok(())
    }

    /// This tail cut to `len` positions. Refused when the open rows of
    /// `len` are no longer held.
    fn truncated(&self, len: usize) -> Result<Self> {
        let now = self.len;
        if len > now {
            return Err(bad(format!("truncate to {len}, {now} positions held")));
        }
        let closed = len / self.kpool;
        let first = now - self.held;
        if len > closed * self.kpool && closed * self.kpool < first {
            return Err(bad(format!(
                "truncate {now} -> {len} needs tok_k from {}, held from {first}",
                closed * self.kpool
            )));
        }
        let keep = len.saturating_sub(first);
        let rows = match (&self.rows, keep) {
            (_, 0) => None,
            (Some(r), k) => {
                let r = r.slice(&[0, 0], &[k as i64, self.width])?.deep_copy()?;
                MxArray::eval_arrays_with_context(&[&r], "hy_v4_flash indexer tail")?;
                Some(r)
            }
            (None, _) => return Err(bad("tok_k rows missing")),
        };
        Ok(Self {
            rows,
            held: keep,
            len,
            ..*self
        })
    }
}

/// The lengths and open rows of an indexer state, to go back to later.
#[derive(Clone)]
pub struct IndexerMark {
    tail: TokTail,
}

impl IndexerMark {
    pub fn len(&self) -> usize {
        self.tail.len
    }

    pub fn is_empty(&self) -> bool {
        self.tail.len == 0
    }

    /// The groups pooled at the mark; refused past the `pooled` groups held.
    fn groups(&self, pooled: usize) -> Result<usize> {
        let closed = self.tail.len / self.tail.kpool;
        if closed > pooled {
            return Err(bad(format!(
                "restore to {} positions, {pooled} groups pooled",
                self.tail.len
            )));
        }
        Ok(closed)
    }
}

/// The pooled keys of the closed groups, in group order.
trait PooledKeys {
    fn groups(&self) -> usize;
    /// Writes `keys [k, d]` f32 as groups `first..first + k`; `first` must
    /// equal `groups()`.
    fn store(&mut self, keys: &MxArray, first: usize) -> Result<()>;
    /// Groups `0..k`, `[k, d]` f16.
    fn read(&self, k: usize) -> Result<MxArray>;
    /// Forgets the groups past `k` (`k <= groups()`).
    fn cut(&mut self, k: usize) -> Result<()>;
}

impl PooledKeys for LatentBuf {
    fn groups(&self) -> usize {
        self.len()
    }

    fn store(&mut self, keys: &MxArray, first: usize) -> Result<()> {
        self.write(keys, first)
    }

    fn read(&self, k: usize) -> Result<MxArray> {
        self.rows()?.slice(&[0, 0], &[k as i64, self.width()])
    }

    fn cut(&mut self, k: usize) -> Result<()> {
        self.truncate(k)
    }
}

/// A paged sequence's pooled keys: the index plane of pool layer `layer`
/// of the adapter's active request.
struct PlaneKeys<'a> {
    adapter: &'a mut PagedKVCacheAdapter,
    layer: u32,
    groups: &'a mut usize,
}

impl PooledKeys for PlaneKeys<'_> {
    fn groups(&self) -> usize {
        *self.groups
    }

    fn store(&mut self, keys: &MxArray, first: usize) -> Result<()> {
        if first != *self.groups {
            return Err(bad(format!(
                "index plane write at group {first}, {} pooled",
                self.groups
            )));
        }
        let k = keys.shape_at(0)? as usize;
        let at = u32::try_from(first).map_err(|_| bad("group id exceeds u32"))?;
        self.adapter
            .store_index_rows(self.layer, keys, at)
            .map_err(bad)?;
        *self.groups += k;
        Ok(())
    }

    fn read(&self, k: usize) -> Result<MxArray> {
        let k = u32::try_from(k).map_err(|_| bad("group count exceeds u32"))?;
        self.adapter.index_rows(self.layer, k).map_err(bad)
    }

    fn cut(&mut self, k: usize) -> Result<()> {
        if k > *self.groups {
            return Err(bad(format!(
                "index plane cut to {k} groups, {} pooled",
                self.groups
            )));
        }
        *self.groups = k;
        Ok(())
    }
}

/// Per-sequence indexer state of one layer: pooled keys of the closed
/// groups (one f16 `[cap / kpool, d]` buffer written in place) and the
/// `tok_k` tail.
pub struct IndexerState {
    pooled: LatentBuf,
    tail: TokTail,
}

impl IndexerState {
    /// Empty state for a context of `cap` positions.
    pub fn new(cap: usize, g: &IndexerGeometry) -> Self {
        Self {
            pooled: LatentBuf::new(cap / g.kpool, g.head_dim as i64),
            tail: TokTail::new(g),
        }
    }

    /// Positions seen.
    pub fn len(&self) -> usize {
        self.tail.len
    }

    pub fn is_empty(&self) -> bool {
        self.tail.is_empty()
    }

    /// Closed groups (pooled rows live).
    pub fn pooled_len(&self) -> usize {
        self.pooled.len()
    }

    /// The pooled key buffer; rows `0..pooled_len()` are live. A caller must
    /// not keep the handle past its call.
    pub fn pooled_rows(&self) -> Result<&MxArray> {
        self.pooled.rows()
    }

    /// Positions of the open group.
    pub fn open(&self) -> Range<usize> {
        self.tail.open(self.pooled.len())
    }

    /// `tok_k` rows held.
    pub fn held(&self) -> usize {
        self.tail.held
    }

    pub fn mark(&self) -> IndexerMark {
        IndexerMark {
            tail: self.tail.clone(),
        }
    }

    /// Goes back to `m`; the pooled rows below `m`'s closed groups must not
    /// have been rewritten since.
    pub fn restore(&mut self, m: IndexerMark) -> Result<()> {
        restore_in(&mut self.pooled, &mut self.tail, m)
    }

    /// Drops the positions past `len`. Refused when the open rows of `len`
    /// are no longer held.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        let tail = self.tail.truncated(len)?;
        self.pooled.truncate(len / tail.kpool)?;
        self.tail = tail;
        Ok(())
    }

    pub fn nbytes(&self) -> usize {
        self.pooled.nbytes() + self.tail.nbytes()
    }
}

/// Per-sequence indexer state of one layer of a paged sequence: the pooled
/// keys live in the index plane of pool layer `layer` (8 groups per 32-row
/// block), so only the count and the `tok_k` tail are held here.
pub struct PagedIndexerState {
    layer: u32,
    groups: usize,
    tail: TokTail,
}

impl PagedIndexerState {
    pub fn new(layer: u32, g: &IndexerGeometry) -> Self {
        Self {
            layer,
            groups: 0,
            tail: TokTail::new(g),
        }
    }

    /// The pool layer.
    pub fn layer(&self) -> u32 {
        self.layer
    }

    /// Positions seen.
    pub fn len(&self) -> usize {
        self.tail.len
    }

    pub fn is_empty(&self) -> bool {
        self.tail.is_empty()
    }

    /// Closed groups (index plane rows live).
    pub fn pooled_len(&self) -> usize {
        self.groups
    }

    /// Positions of the open group.
    pub fn open(&self) -> Range<usize> {
        self.tail.open(self.groups)
    }

    pub fn tail(&self) -> &TokTail {
        &self.tail
    }

    pub fn mark(&self) -> IndexerMark {
        IndexerMark {
            tail: self.tail.clone(),
        }
    }

    /// Goes back to `m`; the index plane rows below `m`'s closed groups must
    /// not have been rewritten since.
    pub fn restore(&mut self, m: IndexerMark) -> Result<()> {
        self.groups = m.groups(self.groups)?;
        self.tail = m.tail;
        Ok(())
    }

    /// Drops the positions past `len`. Refused when the open rows of `len`
    /// are no longer held.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        let tail = self.tail.truncated(len)?;
        self.groups = len / tail.kpool;
        self.tail = tail;
        Ok(())
    }

    /// Bytes held outside the pool.
    pub fn nbytes(&self) -> usize {
        self.tail.nbytes()
    }
}

fn restore_in<P: PooledKeys>(pooled: &mut P, tail: &mut TokTail, m: IndexerMark) -> Result<()> {
    let closed = m.groups(pooled.groups())?;
    pooled.cut(closed)?;
    *tail = m.tail;
    Ok(())
}

/// What an indexed layer needs besides its attention weights.
pub struct Indexing<'a> {
    pub indexer: &'a Indexer,
    pub weights: &'a IndexerWeights,
    pub state: &'a mut IndexerState,
    pub kind: RowKind,
}

/// `Indexing` of a paged sequence.
pub struct PagedIndexing<'a> {
    pub indexer: &'a Indexer,
    pub weights: &'a IndexerWeights,
    pub state: &'a mut PagedIndexerState,
    pub kind: RowKind,
}

/// The rows of one call at positions `start..start + n`: `x [n, dim]`
/// (`rms(h)·attn_norm`) and `qr [n, q_lora]`. Decode rows take the f32 head
/// weights one row per call, so a row's picks do not depend on the others.
#[derive(Clone, Copy)]
pub struct Call<'a> {
    pub x: &'a MxArray,
    pub qr: &'a MxArray,
    pub start: usize,
    pub kind: RowKind,
}

/// The indexer of one layer; the weights stay in `Resident`.
pub struct Indexer {
    g: IndexerGeometry,
    inv: Vec<f32>,
}

fn want(d: &Dense, axis: u32, n: usize, what: &str) -> Result<()> {
    let got = d.shape_at(axis)?;
    if got != n as i64 {
        return Err(bad(format!("{what} axis {axis} is {got}, need {n}")));
    }
    Ok(())
}

fn want_shape(a: &MxArray, shape: &[usize], what: &str) -> Result<()> {
    let got = a.shape()?.as_ref().to_vec();
    if !got.iter().map(|&x| x as usize).eq(shape.iter().copied()) {
        return Err(bad(format!("{what} is {got:?}, need {shape:?}")));
    }
    Ok(())
}

impl Indexer {
    /// Checks `w` against `g`.
    pub fn new(g: IndexerGeometry, w: &IndexerWeights) -> Result<Self> {
        if g.n_heads == 0
            || g.kpool < 2
            || g.n_top == 0
            || g.rope_dim == 0
            || !g.rope_dim.is_multiple_of(2)
            || g.rope_dim >= g.head_dim
        {
            return Err(bad(format!("degenerate geometry {g:?}")));
        }
        let d = g.head_dim;
        want(&w.attn_k, 0, d, "indexer.attn_k")?;
        want(&w.q_b, 0, g.n_heads * d, "indexer.attn_q_b")?;
        want(&w.proj, 0, g.n_heads, "indexer.proj")?;
        want(&w.proj, 1, w.attn_k.shape_at(1)? as usize, "indexer.proj")?;
        want_shape(&w.k_norm, &[d], "indexer.k_norm")?;
        want_shape(&w.compressor_value, &[d, d], "indexer_compressor_value")?;
        want_shape(&w.compressor_gate, &[d, d], "indexer_compressor_gate")?;
        want_shape(&w.compressor_ape, &[g.kpool, d], "indexer_compressor_ape")?;
        want_shape(&w.compressor_norm, &[d], "indexer_compressor_norm")?;
        let inv = inv_freqs(g.rope_dim, g.rope_theta, None);
        Ok(Self { g, inv })
    }

    pub fn geometry(&self) -> &IndexerGeometry {
        &self.g
    }

    /// RoPE on the last `rope_dim` dims of `x [r, .., d]` at `positions`.
    pub fn rope(&self, x: &MxArray, positions: &[u64]) -> Result<MxArray> {
        let (cos, sin) = rope_tables(&self.inv, positions)?;
        apply_partial_rope(x, self.g.rope_dim as i64, &cos, &sin, false)
    }

    /// `rms(attn_k·x)·k_norm` `[n, d]` f32.
    pub fn tok_k(&self, w: &IndexerWeights, x: &MxArray) -> Result<MxArray> {
        rms_norm(
            &qlinear(&x.astype(DType::Float32)?, &w.attn_k)?,
            &w.k_norm,
            self.g.eps,
        )
    }

    /// Pooled keys `[k, d]` f32 of groups `first..first + k` from their
    /// members `grp [k, kpool, d]`.
    pub fn pool(&self, w: &IndexerWeights, grp: &MxArray, first: usize) -> Result<MxArray> {
        let k = grp.shape_at(0)? as usize;
        let grp = grp.astype(DType::Float32)?;
        let v = linear(&grp, &w.compressor_value)?;
        let gt =
            linear(&grp, &w.compressor_gate)?.add(&w.compressor_ape.astype(DType::Float32)?)?;
        let m = gt.max(Some(&[1]), Some(true))?;
        let e = gt.sub(&m)?.exp()?;
        let p = e.div(&e.sum(Some(&[1]), Some(true))?)?;
        let pooled = v.mul(&p)?.sum(Some(&[1]), Some(false))?;
        let normed = rms_norm(&pooled, &w.compressor_norm, self.g.eps)?;
        let pos: Vec<u64> = (first..first + k)
            .map(|g| (g * self.g.kpool) as u64)
            .collect();
        self.rope(&normed, &pos)
    }

    /// Roped index queries `[r, H, d]` of `qr [r, q_lora]` at `positions`.
    pub fn queries(&self, w: &IndexerWeights, qr: &MxArray, positions: &[u64]) -> Result<MxArray> {
        let r = qr.shape_at(0)?;
        let q = qlinear(&qr.astype(DType::Float32)?, &w.q_b)?.reshape(&[
            r,
            self.g.n_heads as i64,
            self.g.head_dim as i64,
        ])?;
        self.rope(&q, positions)
    }

    /// Head weights `proj·x · (H·d)^-½` `[r, H]`; decode rows one per call.
    pub fn head_weights(&self, w: &IndexerWeights, x: &MxArray, kind: RowKind) -> Result<MxArray> {
        let one = |r: &MxArray| {
            linear(&r.astype(DType::Float32)?, &w.proj)?.mul_scalar(f64::from(self.g.weight_scale))
        };
        match kind {
            RowKind::Decode => per_row(x, one),
            RowKind::Prefill => one(x),
        }
    }

    /// Pools the groups the call `x [n, dim]` at `start` closes and carries
    /// its open rows; `start` must equal `st.len()`.
    pub fn update(
        &self,
        w: &IndexerWeights,
        x: &MxArray,
        start: usize,
        st: &mut IndexerState,
    ) -> Result<()> {
        self.update_in(w, x, start, &mut st.pooled, &mut st.tail)
    }

    fn update_in<P: PooledKeys>(
        &self,
        w: &IndexerWeights,
        x: &MxArray,
        start: usize,
        pooled: &mut P,
        tail: &mut TokTail,
    ) -> Result<()> {
        let n = x.shape_at(0)? as usize;
        if n == 0 || start != tail.len || tail.kpool != self.g.kpool {
            return Err(bad(format!(
                "update of {n} rows at {start}, {} positions held",
                tail.len
            )));
        }
        let kp = self.g.kpool;
        let plan = self.g.plan(start, n);
        let open = tail.open(pooled.groups());
        if pooled.groups() != plan.groups.start || open != plan.open_in {
            return Err(bad(format!(
                "state holds {} groups and open {open:?}, the call needs {:?} and {:?}",
                pooled.groups(),
                plan.groups,
                plan.open_in
            )));
        }
        let tok = self.tok_k(w, x)?;
        if !plan.groups.is_empty() {
            let d = self.g.head_dim as i64;
            let all = match tail.last_rows(plan.open_in.len())? {
                Some(open) => MxArray::concatenate(&open, &tok, 0)?,
                None => tok.clone(),
            };
            let k = plan.groups.len();
            let grp = all
                .slice(&[0, 0], &[(k * kp) as i64, d])?
                .reshape(&[k as i64, kp as i64, d])?;
            let keys = self.pool(w, &grp, plan.groups.start)?;
            pooled.store(&keys, plan.groups.start)?;
        }
        tail.push(&tok)
    }

    /// Index rows `[n, plan.kmax]` i32 (`-1` = no cell) of `call`, after
    /// `update` of the same call.
    pub fn cells(&self, w: &IndexerWeights, call: Call<'_>, st: &IndexerState) -> Result<MxArray> {
        self.cells_in(w, call, &st.pooled, &st.tail)
    }

    fn cells_in<P: PooledKeys>(
        &self,
        w: &IndexerWeights,
        call: Call<'_>,
        pooled: &P,
        tail: &TokTail,
    ) -> Result<MxArray> {
        let start = call.start;
        let n = call.x.shape_at(0)? as usize;
        let plan = self.g.plan(start, n);
        if tail.len != start + n || pooled.groups() != plan.cols {
            return Err(bad(format!(
                "cells of [{start}, {}) with {} positions and {} groups",
                start + n,
                tail.len,
                pooled.groups()
            )));
        }
        let width = plan.kmax;
        let mut parts = Vec::new();
        if !plan.dense.is_empty() {
            let rows = plan.dense.len();
            parts.push(MxArray::from_int32(
                &dense_rows(plan.dense.start, rows, width)?,
                &[rows as i64, width as i64],
            )?);
        }
        if !plan.sparse.is_empty() {
            let keys = pooled.read(plan.cols)?.astype(DType::Float32)?;
            parts.push(self.sparse_cells(w, call, &plan, &keys)?);
        }
        match parts.len() {
            1 => Ok(parts.swap_remove(0)),
            _ => MxArray::concatenate_many(parts.iter().collect(), Some(0)),
        }
    }

    /// The picks of the scored rows over `keys [plan.cols, d]` f32.
    fn sparse_cells(
        &self,
        w: &IndexerWeights,
        call: Call<'_>,
        plan: &CallPlan,
        keys: &MxArray,
    ) -> Result<MxArray> {
        let g = &self.g;
        let start = call.start;
        let rows = |a: &MxArray, r: &Range<usize>| -> Result<MxArray> {
            a.slice(
                &[(r.start - start) as i64, 0],
                &[(r.end - start) as i64, a.shape_at(1)?],
            )
        };
        let sp = &plan.sparse;
        let pos: Vec<u64> = sp.clone().map(|t| t as u64).collect();
        let q = self.queries(w, &rows(call.qr, sp)?, &pos)?;
        let hw = self.head_weights(w, &rows(call.x, sp)?, call.kind)?;
        let to_i32 = |v: usize| i32::try_from(v).map_err(|_| bad(format!("{v} exceeds i32")));
        let visible = sp
            .clone()
            .map(|t| to_i32(g.visible(t)))
            .collect::<Result<Vec<_>>>()?;
        let slots = g.tail_slots();
        let mut tails = Vec::with_capacity(sp.len() * slots);
        for t in sp.clone() {
            let cells = g.tail_cells(t);
            let k = cells.len();
            if k > slots {
                return Err(bad(format!("tail of {t} has {k} cells, {slots} slots")));
            }
            for c in cells {
                tails.push(to_i32(c)?);
            }
            tails.extend(std::iter::repeat_n(-1, slots - k));
        }
        let (h, d) = (g.n_heads as i64, g.head_dim as i64);
        let per = score_rows(g.n_heads, plan.cols, INDEX_CHUNK_BYTES);
        let total = sp.len();
        let mut outs = Vec::new();
        let mut lo = 0;
        while lo < total {
            let hi = (lo + per).min(total);
            let c = (hi - lo) as i64;
            let qc = q.slice(&[lo as i64, 0, 0], &[hi as i64, h, d])?;
            let wc = hw.slice(&[lo as i64, 0], &[hi as i64, h])?;
            let sc = scores(&qc, &wc, keys)?;
            let ids = pick_groups(&sc, &visible[lo..hi], g.n_top)?;
            let tail = MxArray::from_int32(&tails[lo * slots..hi * slots], &[c, slots as i64])?;
            let idx = MxArray::concatenate(&group_cells(&ids, g.kpool)?, &tail, 1)?;
            if per < total {
                MxArray::eval_arrays_with_context(&[&idx], "hy_v4_flash indexer pick")?;
            }
            outs.push(idx);
            lo = hi;
        }
        match outs.len() {
            1 => Ok(outs.swap_remove(0)),
            _ => MxArray::concatenate_many(outs.iter().collect(), Some(0)),
        }
    }

    /// `update` then `cells`; on error `st` holds what it held before.
    pub fn select(
        &self,
        w: &IndexerWeights,
        call: Call<'_>,
        st: &mut IndexerState,
    ) -> Result<MxArray> {
        self.select_in(w, call, &mut st.pooled, &mut st.tail)
    }

    /// `select` for a paged sequence: the pooled keys go to and come from
    /// the index plane of the adapter's active request.
    pub fn select_paged(
        &self,
        w: &IndexerWeights,
        call: Call<'_>,
        st: &mut PagedIndexerState,
        adapter: &mut PagedKVCacheAdapter,
    ) -> Result<MxArray> {
        let mut keys = PlaneKeys {
            adapter,
            layer: st.layer,
            groups: &mut st.groups,
        };
        self.select_in(w, call, &mut keys, &mut st.tail)
    }

    fn select_in<P: PooledKeys>(
        &self,
        w: &IndexerWeights,
        call: Call<'_>,
        pooled: &mut P,
        tail: &mut TokTail,
    ) -> Result<MxArray> {
        let mark = IndexerMark { tail: tail.clone() };
        let out = self
            .update_in(w, call.x, call.start, pooled, tail)
            .and_then(|()| self.cells_in(w, call, pooled, tail));
        if out.is_err() {
            restore_in(pooled, tail, mark)?;
        }
        out
    }
}

#[cfg(test)]
mod tests;
