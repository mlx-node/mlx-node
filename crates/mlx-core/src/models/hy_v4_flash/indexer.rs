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
use super::ops::{fast_partial_rope_tail, rms_norm, rope_inv_freqs};
use super::weights::{Dense, IndexerWeights};
use crate::array::{DType, MxArray};
use crate::models::moe_common::{INDEX_CHUNK_BYTES, inv_freqs, linear};
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

/// Extra slots reserved in [`TokTail`]'s circular ring beyond `keep` so
/// decode steps up to 4 rows write disjoint slots and roll back in O(1)
/// without cloning the ring buffer.
const TAIL_ROLLBACK_SLACK: usize = 4;

fn linear_tail_range(
    buf: &MxArray,
    cap: usize,
    width: i64,
    first_pos: usize,
    count: usize,
    detach: bool,
) -> Result<MxArray> {
    let lo = first_pos % cap;
    if lo + count <= cap {
        let view = buf.slice(&[lo as i64, 0], &[(lo + count) as i64, width])?;
        if detach { view.deep_copy() } else { Ok(view) }
    } else {
        let head = buf.slice(&[lo as i64, 0], &[cap as i64, width])?;
        let tail = buf.slice(&[0, 0], &[(lo + count - cap) as i64, width])?;
        MxArray::concatenate(&head, &tail, 0)
    }
}

/// `tok_k` rows of positions `len - held .. len` (the open group and up to
/// `WINDOW_SLACK` positions before it), f32. Live tails use a fixed circular
/// ring buffer (`ring_cap > 0`) updated in place via `Hy4RingWrite`; snapshots
/// and restored `of_rows` tails hold a linear `[held, width]` array (`ring_cap == 0`).
#[derive(Clone)]
pub struct TokTail {
    kpool: usize,
    width: i64,
    keep: usize,
    rows: Option<MxArray>,
    ring_cap: usize,
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
            ring_cap: 0,
            held: 0,
            len: 0,
        }
    }

    fn target_ring_cap(&self) -> usize {
        self.keep + TAIL_ROLLBACK_SLACK
    }

    /// Max rows a push can append without touching any of the `keep` held rows,
    /// enabling O(1) rollback with zero buffer clone.
    pub(crate) fn rollback_slack(&self) -> usize {
        self.ring_cap.saturating_sub(self.keep)
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

    /// A tail of `len` positions holding `rows [held, d]` (f32), the
    /// `tok_k` rows of its last `held` positions. Refused past `keep` rows,
    /// past `len`, or without the open group's rows.
    pub fn of_rows(g: &IndexerGeometry, rows: Option<MxArray>, len: usize) -> Result<Self> {
        let mut t = Self::new(g);
        let held = match &rows {
            None => 0,
            Some(r) => match r.shape()?.as_ref() {
                &[n, w] if w == t.width && n > 0 => n as usize,
                s => {
                    return Err(bad(format!("tail rows {s:?}, need [n > 0, {}]", t.width)));
                }
            },
        };
        let open = len % t.kpool;
        if held > t.keep || held > len || held < open {
            return Err(bad(format!(
                "a tail of {len} positions with {held} rows (keep {}, open {open})",
                t.keep
            )));
        }
        t.rows = rows.map(|r| r.astype(DType::Float32)).transpose()?;
        t.ring_cap = 0;
        t.held = held;
        t.len = len;
        Ok(t)
    }

    /// The held rows `[held, d]` f32 (for linear snapshots / `of_rows`).
    pub fn rows(&self) -> Option<&MxArray> {
        self.rows.as_ref()
    }

    /// Detached linear oldest-first `[held, d]` copy (unevaluated).
    pub(crate) fn snapshot_lazy(&self) -> Result<Self> {
        if self.held == 0 {
            return Ok(Self {
                kpool: self.kpool,
                width: self.width,
                keep: self.keep,
                rows: None,
                ring_cap: 0,
                held: 0,
                len: self.len,
            });
        }
        let buf = self
            .rows
            .as_ref()
            .ok_or_else(|| bad("tok_k rows missing"))?;
        let linear = if self.ring_cap == 0 {
            buf.clone()
        } else {
            let first_pos = self.len - self.held;
            linear_tail_range(buf, self.ring_cap, self.width, first_pos, self.held, true)?
        };
        Ok(Self {
            kpool: self.kpool,
            width: self.width,
            keep: self.keep,
            rows: Some(linear),
            ring_cap: 0,
            held: self.held,
            len: self.len,
        })
    }

    /// Evaluated detached linear oldest-first `[held, d]` snapshot so live ring
    /// buffers drop back to refcount 1.
    pub(crate) fn snapshot(&self) -> Result<Self> {
        let snap = self.snapshot_lazy()?;
        if let Some(rows) = snap.rows()
            && self.ring_cap > 0
        {
            MxArray::eval_arrays_with_context(&[rows], "hy_v4_flash indexer tail snapshot")?;
        }
        Ok(snap)
    }

    pub fn nbytes(&self) -> usize {
        self.rows.as_ref().map_or(0, MxArray::nbytes)
    }

    /// Positions of the open group when `groups` groups are pooled.
    fn open(&self, groups: usize) -> Range<usize> {
        self.kpool * groups..self.len
    }

    /// Backing buffer, start slot, and circular capacity for reading the last
    /// `n` held rows without slicing or concatenating.
    fn open_ring_args(&self, n: usize) -> Result<Option<(&MxArray, usize, usize)>> {
        if n == 0 {
            return Ok(None);
        }
        if n > self.held {
            return Err(bad(format!("{n} open rows needed, {} held", self.held)));
        }
        let buf = self
            .rows
            .as_ref()
            .ok_or_else(|| bad("tok_k rows missing"))?;
        if self.ring_cap > 0 {
            let first = self.len - n;
            Ok(Some((buf, first % self.ring_cap, self.ring_cap)))
        } else {
            Ok(Some((buf, self.held - n, self.held)))
        }
    }

    /// The last `n` held rows (`n <= held`) in oldest-first linear order.
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
        if self.ring_cap > 0 {
            let first = self.len - n;
            Ok(Some(linear_tail_range(
                r,
                self.ring_cap,
                self.width,
                first,
                n,
                false,
            )?))
        } else {
            Ok(Some(r.slice(
                &[(held - n) as i64, 0],
                &[held as i64, self.width],
            )?))
        }
    }

    /// Appends `tok [n, d]` (positions `len..len + n`) via in-place donated
    /// circular ring write.
    fn push(&mut self, tok: &MxArray) -> Result<()> {
        let n = tok.shape_at(0)? as usize;
        let cap = if self.ring_cap > 0 {
            self.ring_cap
        } else {
            self.target_ring_cap()
        };
        let cap_i64 = i64::try_from(cap).map_err(|_| bad("tail ring capacity exceeds i64"))?;

        let prev = if self.ring_cap > 0 {
            match self.rows.take() {
                Some(b) => b,
                None => MxArray::zeros(&[cap_i64, self.width], Some(DType::Float32))?,
            }
        } else {
            let zero = MxArray::zeros(&[cap_i64, self.width], Some(DType::Float32))?;
            if let Some(linear) = self.rows.take()
                && self.held > 0
            {
                let first_pos = self.len - self.held;
                let filled = super::ops::ring_write(&zero, &linear, first_pos % cap, cap)?;
                drop(zero);
                filled
            } else {
                zero
            }
        };

        let (src, write_start) = if n > cap {
            (
                tok.slice(&[(n - cap) as i64, 0], &[n as i64, self.width])?,
                self.len + n - cap,
            )
        } else {
            (tok.clone(), self.len)
        };
        let next = super::ops::ring_write(&prev, &src, write_start % cap, cap)?;
        drop(prev);
        self.rows = Some(next);
        self.ring_cap = cap;
        if let Some(buf) = self.rows.as_ref() {
            super::stats::defer(buf, "hy_v4_flash indexer tail")?;
        }
        self.held = (self.held + n).min(self.keep);
        self.len += n;
        Ok(())
    }

    /// Cuts this tail to `len` positions in place. O(1) metadata-only when
    /// backed by a circular ring; refused when open rows of `len` are gone.
    fn truncate_in_place(&mut self, len: usize) -> Result<()> {
        let now = self.len;
        if len > now {
            return Err(bad(format!("truncate to {len}, {now} positions held")));
        }
        if len == now {
            return Ok(());
        }
        let closed = len / self.kpool;
        let first = now - self.held;
        if len > closed * self.kpool && closed * self.kpool < first {
            return Err(bad(format!(
                "truncate {now} -> {len} needs tok_k from {}, held from {first}",
                closed * self.kpool
            )));
        }
        let rem = len.saturating_sub(first);
        if self.ring_cap > 0 {
            self.held = rem;
            self.len = len;
            return Ok(());
        }
        let rows = match (&self.rows, rem) {
            (_, 0) => None,
            (Some(r), k) => {
                let r = r.slice(&[0, 0], &[k as i64, self.width])?.deep_copy()?;
                super::stats::defer(&r, "hy_v4_flash indexer tail")?;
                Some(r)
            }
            (None, _) => return Err(bad("tok_k rows missing")),
        };
        self.rows = rows;
        self.held = rem;
        self.len = len;
        Ok(())
    }

    /// Reverts a circular-ring push of at most `rollback_slack()` rows in O(1),
    /// restoring the exact `prev_held` rows that were untouched in disjoint slots.
    fn rollback_push(&mut self, prev_len: usize, prev_held: usize) {
        self.len = prev_len;
        self.held = prev_held;
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

    pub fn tail(&self) -> &TokTail {
        &self.tail
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

/// Direct backing storage for fused indexer scoring without materializing
/// `[cols, d]` f32 keys.
enum ScoreKeys {
    Flat { pool: MxArray },
    Paged { pool: MxArray, pages: MxArray },
}

/// The pooled keys of the closed groups, in group order.
trait PooledKeys {
    fn groups(&self) -> usize;
    /// Storage dtype of pooled rows (`f16` in production, `f32` under test override).
    fn storage_dtype(&self) -> DType;
    /// Writes `keys [k, d]` as groups `first..first + k`; `first` must
    /// equal `groups()`.
    fn store(&mut self, keys: &MxArray, first: usize) -> Result<()>;
    /// Groups `0..k`, `[k, d]` f16.
    fn read(&self, k: usize) -> Result<MxArray>;
    /// Forgets the groups past `k` (`k <= groups()`).
    fn cut(&mut self, k: usize) -> Result<()>;
    /// Raw f16 pool view for fused scoring, when available.
    fn score_keys(&self) -> Result<Option<ScoreKeys>> {
        Ok(None)
    }
}

impl PooledKeys for LatentBuf {
    fn groups(&self) -> usize {
        self.len()
    }

    fn storage_dtype(&self) -> DType {
        super::cache::row_dtype()
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

    fn score_keys(&self) -> Result<Option<ScoreKeys>> {
        if self.is_empty() {
            return Ok(None);
        }
        Ok(Some(ScoreKeys::Flat {
            pool: self.rows()?.clone(),
        }))
    }
}

/// A paged sequence's pooled keys: the index plane of pool layer `layer`
/// of the adapter's active request.
struct PlaneKeys<'a> {
    adapter: &'a mut PagedKVCacheAdapter,
    layer: u32,
    groups: &'a mut usize,
    pages: Option<MxArray>,
}

impl PooledKeys for PlaneKeys<'_> {
    fn groups(&self) -> usize {
        *self.groups
    }

    fn storage_dtype(&self) -> DType {
        match self.adapter.layer_kv_pool().index_dtype() {
            mlx_paged_attn::metal::MetalDtype::Float16 => DType::Float16,
            mlx_paged_attn::metal::MetalDtype::BFloat16 => DType::BFloat16,
            _ => DType::Float32,
        }
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
        let written = self
            .adapter
            .store_index_rows(self.layer, keys, at)
            .map_err(bad)?;
        super::stats::defer(&written, "hy_v4_flash paged index store")?;
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

    fn score_keys(&self) -> Result<Option<ScoreKeys>> {
        let Some(pages) = self.pages.as_ref() else {
            return Ok(None);
        };
        let pool = self.adapter.index_pool_array(self.layer).map_err(bad)?;
        Ok(Some(ScoreKeys::Paged {
            pool,
            pages: pages.clone(),
        }))
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

    pub fn tail(&self) -> &TokTail {
        &self.tail
    }

    pub fn mark(&self) -> IndexerMark {
        IndexerMark {
            tail: self
                .tail
                .snapshot_lazy()
                .unwrap_or_else(|_| self.tail.clone()),
        }
    }

    /// Goes back to `m`; the pooled rows below `m`'s closed groups must not
    /// have been rewritten since.
    pub fn restore(&mut self, m: IndexerMark) -> Result<()> {
        restore_in(&mut self.pooled, &mut self.tail, m)
    }

    /// Reverts a ring push of at most `rollback_slack()` rows in O(1),
    /// restoring exact `prev_len`, `prev_held`, and closed group count.
    pub(crate) fn rollback_push(&mut self, prev_len: usize, prev_held: usize) -> Result<()> {
        self.pooled.cut(prev_len / self.tail.kpool)?;
        self.tail.rollback_push(prev_len, prev_held);
        Ok(())
    }

    /// Fills an empty state: the pooled keys `[k, d]` of groups `0..k`
    /// (none when `k = 0`) and `tail`, which must close exactly `k` groups.
    pub fn fill(&mut self, pooled: Option<&MxArray>, tail: TokTail) -> Result<()> {
        if !self.is_empty() || !self.pooled.is_empty() {
            return Err(bad("fill of an indexer that holds positions"));
        }
        if (tail.kpool, tail.width) != (self.tail.kpool, self.tail.width) {
            return Err(bad("fill with a tail of another indexer"));
        }
        let k = pooled.map(|p| p.shape_at(0)).transpose()?.unwrap_or(0) as usize;
        if tail.len / tail.kpool != k {
            return Err(bad(format!(
                "a tail of {} positions over {k} pooled groups",
                tail.len
            )));
        }
        if let Some(p) = pooled {
            self.pooled.write(p, 0)?;
        }
        self.tail = tail;
        Ok(())
    }

    /// Appends the pooled keys `[k, d]` of the next closed groups to a state
    /// being filled, which has seen no position yet.
    pub fn append_pooled(&mut self, pooled: &MxArray) -> Result<()> {
        if !self.is_empty() {
            return Err(bad("fill of an indexer that holds positions"));
        }
        if pooled.shape_at(0)? > 0 {
            let at = self.pooled.len();
            self.pooled.write(pooled, at)?;
        }
        Ok(())
    }

    /// Ends a fill by [`Self::append_pooled`]: `tail` must close exactly
    /// the groups appended.
    pub fn close_fill(&mut self, tail: TokTail) -> Result<()> {
        if !self.is_empty() {
            return Err(bad("fill of an indexer that holds positions"));
        }
        if (tail.kpool, tail.width) != (self.tail.kpool, self.tail.width) {
            return Err(bad("fill with a tail of another indexer"));
        }
        let k = self.pooled.len();
        if tail.len / tail.kpool != k {
            return Err(bad(format!(
                "a tail of {} positions over {k} pooled groups",
                tail.len
            )));
        }
        self.tail = tail;
        Ok(())
    }

    /// Drops the positions past `len`. Refused when the open rows of `len`
    /// are no longer held.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        let mut tail = self.tail.clone();
        tail.truncate_in_place(len)?;
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

    /// The state at `tail.len()` positions with every closed group in the
    /// index plane.
    pub fn at(layer: u32, tail: TokTail) -> Self {
        Self {
            layer,
            groups: tail.len / tail.kpool,
            tail,
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
            tail: self
                .tail
                .snapshot_lazy()
                .unwrap_or_else(|_| self.tail.clone()),
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
        self.tail.truncate_in_place(len)?;
        self.groups = len / self.tail.kpool;
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
    inv_freqs: MxArray,
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
        let inv_freqs = rope_inv_freqs(&inv_freqs(g.rope_dim, g.rope_theta, None))?;
        Ok(Self { g, inv_freqs })
    }

    pub fn geometry(&self) -> &IndexerGeometry {
        &self.g
    }

    /// RoPE on the last `rope_dim` dims of `x [r, .., d]` at `positions`.
    pub fn rope(&self, x: &MxArray, positions: &[u64]) -> Result<MxArray> {
        fast_partial_rope_tail(x, &self.inv_freqs, positions)
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
        if k > 0
            && self.g.head_dim == 128
            && self.g.kpool == 4
            && let Some(keys) = super::ops::try_kpool_compress(
                None,
                &grp.reshape(&[(k * 4) as i64, 128])?,
                &w.compressor_value,
                &w.compressor_gate,
                &w.compressor_ape,
                &w.compressor_norm,
                &self.inv_freqs,
                k,
                first,
                0,
                0,
                1,
                self.g.eps,
                DType::Float32,
            )?
        {
            return Ok(keys);
        }
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

    /// Head weights `proj·x · (H·d)^-½` `[r, H]`; single-dispatch row-invariant
    /// for both f32 and quantized `proj` at decode widths (`M <= 16`).
    pub fn head_weights(&self, w: &IndexerWeights, x: &MxArray, kind: RowKind) -> Result<MxArray> {
        if let Some(hw) = super::ops::try_f32_head_weights(x, &w.proj, self.g.weight_scale)? {
            return Ok(hw);
        }
        let one = |r: &MxArray| {
            qlinear(&r.astype(DType::Float32)?, &w.proj)?.mul_scalar(f64::from(self.g.weight_scale))
        };
        match kind {
            RowKind::Decode if !w.proj.is_qmm() || x.shape_at(0)? > 16 => per_row(x, one),
            _ => one(x),
        }
    }

    /// Pools the groups the call `x [n, dim]` at `start` closes and carries
    /// its open rows; `start` must equal `st.len()`. On error `st` holds
    /// what it held before.
    pub fn update(
        &self,
        w: &IndexerWeights,
        x: &MxArray,
        start: usize,
        st: &mut IndexerState,
    ) -> Result<()> {
        let n = x.shape_at(0)? as usize;
        let fast_rollback = (st.tail.len == 0 && n <= st.tail.target_ring_cap())
            || (st.tail.ring_cap > 0 && n <= st.tail.rollback_slack());
        if fast_rollback {
            let (prev_len, prev_held, prev_groups, had_buf) = (
                st.tail.len,
                st.tail.held,
                st.pooled.groups(),
                st.tail.rows.is_some(),
            );
            let out = self.update_in(w, x, start, &mut st.pooled, &mut st.tail);
            if out.is_err() {
                st.pooled.cut(prev_groups)?;
                if !had_buf {
                    st.tail.rows = None;
                    st.tail.ring_cap = 0;
                    st.tail.held = 0;
                    st.tail.len = 0;
                } else {
                    st.tail.rollback_push(prev_len, prev_held);
                }
            }
            return out;
        }
        let mark = st.mark();
        let out = self.update_in(w, x, start, &mut st.pooled, &mut st.tail);
        if out.is_err() {
            st.restore(mark)?;
        }
        out
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
            let k = plan.groups.len();
            let open_len = plan.open_in.len();
            let fused = if self.g.head_dim == 128 && kp == 4 {
                let (tail_buf, tail_start, tail_cap) = match tail.open_ring_args(open_len)? {
                    Some((buf, s, c)) => (Some(buf), s, c),
                    None => (None, 0, 1),
                };
                super::ops::try_kpool_compress(
                    tail_buf,
                    &tok,
                    &w.compressor_value,
                    &w.compressor_gate,
                    &w.compressor_ape,
                    &w.compressor_norm,
                    &self.inv_freqs,
                    k,
                    plan.groups.start,
                    open_len,
                    tail_start,
                    tail_cap,
                    self.g.eps,
                    pooled.storage_dtype(),
                )?
            } else {
                None
            };
            let keys = match fused {
                Some(keys) => keys,
                None => {
                    let d = self.g.head_dim as i64;
                    let all = match tail.last_rows(open_len)? {
                        Some(open) => MxArray::concatenate(&open, &tok, 0)?,
                        None => tok.clone(),
                    };
                    let grp = all
                        .slice(&[0, 0], &[(k * kp) as i64, d])?
                        .reshape(&[k as i64, kp as i64, d])?;
                    self.pool(w, &grp, plan.groups.start)?
                }
            };
            pooled.store(&keys, plan.groups.start)?;
        }
        tail.push(&tok)
    }

    /// Index rows `[n, plan.kmax]` i32 (`-1` = no cell) of `call`, after
    /// `update` of the same call.
    pub fn cells(&self, w: &IndexerWeights, call: Call<'_>, st: &IndexerState) -> Result<MxArray> {
        self.cells_in(w, call, &st.pooled, &st.tail, None)
    }

    fn cells_in<P: PooledKeys>(
        &self,
        w: &IndexerWeights,
        call: Call<'_>,
        pooled: &P,
        tail: &TokTail,
        map_pages: Option<&MxArray>,
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
            let dense = MxArray::from_int32(
                &dense_rows(plan.dense.start, rows, width)?,
                &[rows as i64, width as i64],
            )?;
            let dense = match map_pages {
                Some(pages) => super::paged::slot_ids(pages, &dense)?,
                None => dense,
            };
            parts.push(dense);
        }
        if !plan.sparse.is_empty() {
            parts.push(self.sparse_cells(w, call, &plan, pooled, map_pages)?);
        }
        match parts.len() {
            1 => Ok(parts.swap_remove(0)),
            _ => MxArray::concatenate_many(parts.iter().collect(), Some(0)),
        }
    }

    /// The picks of the scored rows over `pooled` (logical cells when
    /// `map_pages` is `None`, physical slots when `Some`).
    fn sparse_cells<P: PooledKeys>(
        &self,
        w: &IndexerWeights,
        call: Call<'_>,
        plan: &CallPlan,
        pooled: &P,
        map_pages: Option<&MxArray>,
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
        let slots = g.tail_slots();
        let mut visible = Vec::with_capacity(sp.len());
        let mut tail_starts = Vec::with_capacity(sp.len());
        let mut tail_lens = Vec::with_capacity(sp.len());
        for t in sp.clone() {
            visible.push(to_i32(g.visible(t))?);
            let cells = g.tail_cells(t);
            let k = cells.len();
            if k > slots {
                return Err(bad(format!("tail of {t} has {k} cells, {slots} slots")));
            }
            tail_starts.push(to_i32(cells.start)?);
            tail_lens.push(to_i32(k)?);
        }

        let score_keys = pooled.score_keys()?;
        let can_fused_score =
            score_keys.is_some() && g.head_dim == 128 && (1..=16).contains(&g.n_heads);
        let mut flat_keys_f32: Option<MxArray> = None;

        let (h, d) = (g.n_heads as i64, g.head_dim as i64);
        let per = if can_fused_score {
            (INDEX_CHUNK_BYTES / (4 * plan.cols.max(1))).max(1)
        } else {
            score_rows(g.n_heads, plan.cols, INDEX_CHUNK_BYTES)
        };
        let total = sp.len();
        let mut outs = Vec::new();
        let mut lo = 0;
        while lo < total {
            let hi = (lo + per).min(total);
            let c = (hi - lo) as i64;
            let (qc, wc) = if lo == 0 && hi == total {
                (q.clone(), hw.clone())
            } else {
                (
                    q.slice(&[lo as i64, 0, 0], &[hi as i64, h, d])?,
                    hw.slice(&[lo as i64, 0], &[hi as i64, h])?,
                )
            };
            let vis_arr = MxArray::from_int32(&visible[lo..hi], &[c])?;
            let sc = match score_keys.as_ref() {
                Some(ScoreKeys::Flat { pool }) => super::ops::try_indexer_score_causal(
                    &qc,
                    &wc,
                    pool,
                    None,
                    Some(&vis_arr),
                    plan.cols,
                    0,
                )?,
                Some(ScoreKeys::Paged { pool, pages }) => super::ops::try_indexer_score_causal(
                    &qc,
                    &wc,
                    pool,
                    Some(pages),
                    Some(&vis_arr),
                    plan.cols,
                    0,
                )?,
                None => None,
            };
            let sc = match sc {
                Some(sc) => sc,
                None => {
                    let keys = match flat_keys_f32.as_ref() {
                        Some(k) => k,
                        None => {
                            flat_keys_f32.insert(pooled.read(plan.cols)?.astype(DType::Float32)?)
                        }
                    };
                    scores(&qc, &wc, keys)?
                }
            };
            let ts_arr = MxArray::from_int32(&tail_starts[lo..hi], &[c])?;
            let tl_arr = MxArray::from_int32(&tail_lens[lo..hi], &[c])?;
            let idx = match super::ops::try_indexer_topk_cells(
                &sc,
                &vis_arr,
                &ts_arr,
                &tl_arr,
                map_pages,
                g.n_top,
                g.kpool,
                plan.kmax,
                super::paged::BLOCK as usize,
            )? {
                Some(idx) => idx,
                None => {
                    let ids = pick_groups(&sc, &visible[lo..hi], g.n_top)?;
                    let mut tails = Vec::with_capacity((hi - lo) * slots);
                    for idx_r in lo..hi {
                        let s0 = tail_starts[idx_r];
                        let tl = tail_lens[idx_r] as usize;
                        for off in 0..tl {
                            tails.push(s0 + off as i32);
                        }
                        tails.extend(std::iter::repeat_n(-1, slots - tl));
                    }
                    let tail = MxArray::from_int32(&tails, &[c, slots as i64])?;
                    let logical = MxArray::concatenate(&group_cells(&ids, g.kpool)?, &tail, 1)?;
                    match map_pages {
                        Some(pages) => super::paged::slot_ids(pages, &logical)?,
                        None => logical,
                    }
                }
            };
            if per < total {
                super::stats::eval(&[&idx], "hy_v4_flash indexer pick")?;
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
        self.select_in(w, call, &mut st.pooled, &mut st.tail, None)
    }

    /// Single-row read-only selection without mutating `st` when `select_tail`
    /// guarantees the new row never scores its own group (`visible(t) == t / kpool`).
    /// Falls back to [`Self::select`] for multi-row calls or `!select_tail`.
    pub fn select_readonly(
        &self,
        w: &IndexerWeights,
        call: Call<'_>,
        st: &mut IndexerState,
    ) -> Result<MxArray> {
        let n = call.x.shape_at(0)? as usize;
        if n != 1 || !self.g.select_tail {
            return self.select(w, call, st);
        }
        let start = call.start;
        let groups_before = start / self.g.kpool;
        if st.tail.len != start || st.pooled.groups() != groups_before {
            return Err(bad(format!(
                "select_readonly at {start} with {} positions and {} groups",
                st.tail.len,
                st.pooled.groups()
            )));
        }
        let mut plan = self.g.plan(start, 1);
        plan.cols = groups_before;
        if !plan.dense.is_empty() {
            return MxArray::from_int32(
                &dense_rows(plan.dense.start, 1, plan.kmax)?,
                &[1, plan.kmax as i64],
            );
        }
        self.sparse_cells(w, call, &plan, &st.pooled, None)
    }

    /// `select` for a paged sequence: the pooled keys go to and come from
    /// the index plane of the adapter's active request. Returns logical cell ids.
    pub fn select_paged(
        &self,
        w: &IndexerWeights,
        call: Call<'_>,
        st: &mut PagedIndexerState,
        adapter: &mut PagedKVCacheAdapter,
    ) -> Result<MxArray> {
        let pages = adapter
            .block_table()
            .filter(|t| !t.block_ids().is_empty())
            .map(|t| super::paged::page_ids(&t.block_ids()))
            .transpose()?;
        let mut keys = PlaneKeys {
            adapter,
            layer: st.layer,
            groups: &mut st.groups,
            pages,
        };
        self.select_in(w, call, &mut keys, &mut st.tail, None)
    }

    /// `select` for a paged sequence returning physical slot ids directly
    /// (`pages[c / 32] * 32 + c % 32`, `-1` kept) via `slots`.
    pub fn select_paged_slots(
        &self,
        w: &IndexerWeights,
        call: Call<'_>,
        st: &mut PagedIndexerState,
        adapter: &mut PagedKVCacheAdapter,
        slots: &super::paged::SlotMap,
    ) -> Result<MxArray> {
        let pages = slots.pages(adapter)?.clone();
        let mut keys = PlaneKeys {
            adapter,
            layer: st.layer,
            groups: &mut st.groups,
            pages: Some(pages.clone()),
        };
        self.select_in(w, call, &mut keys, &mut st.tail, Some(&pages))
    }

    fn select_in<P: PooledKeys>(
        &self,
        w: &IndexerWeights,
        call: Call<'_>,
        pooled: &mut P,
        tail: &mut TokTail,
        map_pages: Option<&MxArray>,
    ) -> Result<MxArray> {
        let n = call.x.shape_at(0)? as usize;
        let fast_rollback = (tail.len == 0 && n <= tail.target_ring_cap())
            || (tail.ring_cap > 0 && n <= tail.rollback_slack());
        if fast_rollback {
            let (prev_len, prev_held, prev_groups, had_buf) =
                (tail.len, tail.held, pooled.groups(), tail.rows.is_some());
            let out = self
                .update_in(w, call.x, call.start, pooled, tail)
                .and_then(|()| self.cells_in(w, call, pooled, tail, map_pages));
            if out.is_err() {
                pooled.cut(prev_groups)?;
                if !had_buf {
                    tail.rows = None;
                    tail.ring_cap = 0;
                    tail.held = 0;
                    tail.len = 0;
                } else {
                    tail.rollback_push(prev_len, prev_held);
                }
            }
            return out;
        }
        let mark = IndexerMark {
            tail: tail.snapshot_lazy().unwrap_or_else(|_| tail.clone()),
        };
        let out = self
            .update_in(w, call.x, call.start, pooled, tail)
            .and_then(|()| self.cells_in(w, call, pooled, tail, map_pages));
        if out.is_err() {
            restore_in(pooled, tail, mark)?;
        }
        out
    }
}

#[cfg(test)]
mod tests;
