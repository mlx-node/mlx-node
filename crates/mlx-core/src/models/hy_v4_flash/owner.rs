//! One paged sequence's state outside the KV pool (its owner cell) and the
//! table of the owners a load holds:
//!
//! | part | held |
//! |---|---|
//! | sliding trunk layer | `WindowRing`: the last `window + slack` latent rows |
//! | indexed trunk layer | `PagedIndexerState`: the pooled group count and the `tok_k` tail (keys in the index plane) |
//! | MTP layer | `Draft`: K/V rows, indexer, seeds `h_trunk` (the carried `h_trunk(s - 1)` among them) |
//! | Engram | the last `max_ngram - 1` token ids |
//!
//! A swap moves a cell out of the table and back (`take` / `put`), never a
//! copy. `state` copies the cell at a block row into a `PrefixState`; the
//! MTP rows of whole 32-row segments are copied once per cell and shared by
//! every state taken from it.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use napi::{Error, Result};

use super::cache::{WINDOW_SLACK, WindowRing};
use super::config::{Attn, HyV4FlashConfig};
use super::indexer::{IndexerGeometry, PagedIndexerState};
use super::mtp::Draft;
use super::paged::pool_layers;
use super::prefill::{BLOCK_ROWS, index_keeps, window_keeps};
use super::prefix_state::{DraftState, OpenRows, PrefixState, Segment, StateLayer};
use crate::array::MxArray;
use crate::transformer::paged_kv_cache_adapter::SeqId;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 owner cell: {msg}"))
}

/// One trunk layer's part of a cell.
pub enum CellLayer {
    Window(WindowRing),
    Index(PagedIndexerState),
}

impl CellLayer {
    /// Positions the layer holds.
    pub fn len(&self) -> usize {
        match self {
            Self::Window(r) => r.len(),
            Self::Index(ix) => ix.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn nbytes(&self) -> usize {
        match self {
            Self::Window(r) => r.nbytes(),
            Self::Index(ix) => ix.nbytes(),
        }
    }
}

/// Empty parts for every trunk layer of `cfg`, the indexed layers on their
/// pool layers.
fn layers_of(cfg: &HyV4FlashConfig) -> Result<Vec<CellLayer>> {
    let pools = pool_layers(cfg);
    let row = i64::try_from(cfg.mla.kv_row()).map_err(|_| bad("latent row exceeds i64"))?;
    cfg.trunk()
        .iter()
        .zip(&pools)
        .enumerate()
        .map(|(l, (lc, pool))| match (lc.attn, pool) {
            (Attn::Sliding, None) => Ok(CellLayer::Window(WindowRing::new(
                cfg.window,
                WINDOW_SLACK,
                row,
            ))),
            (Attn::Indexed, Some(p)) => Ok(CellLayer::Index(PagedIndexerState::new(
                *p,
                &IndexerGeometry::of(cfg, l)?,
            ))),
            (attn, p) => Err(bad(format!(
                "layer {l}: {attn:?} attention on pool layer {p:?}"
            ))),
        })
        .collect()
}

/// One paged sequence's cell.
pub struct OwnerCell {
    layers: Vec<CellLayer>,
    draft: Option<Draft>,
    recent: Vec<u32>,
    back: usize,
    kpool: usize,
    len: usize,
    segments: Vec<Arc<Segment>>,
}

impl OwnerCell {
    /// An empty cell for `cfg`'s trunk; `draft` (empty) with the MTP layer.
    pub fn new(cfg: &HyV4FlashConfig, draft: Option<Draft>) -> Result<Self> {
        let kpool = cfg.indexer.kpool;
        if kpool == 0 || !BLOCK_ROWS.is_multiple_of(kpool) {
            return Err(bad(format!(
                "groups of {kpool} do not tile a {BLOCK_ROWS}-row block"
            )));
        }
        if draft.as_ref().is_some_and(|d| !d.is_empty()) {
            return Err(bad("a new cell given a draft that holds rows"));
        }
        Ok(Self {
            layers: layers_of(cfg)?,
            draft,
            recent: Vec::new(),
            back: cfg.engram.max_ngram.saturating_sub(1),
            kpool,
            len: 0,
            segments: Vec::new(),
        })
    }

    /// The cell of a sequence resumed at `s`: `lineage` is its tokens up to
    /// `s.row`, `draft` an empty draft with the MTP layer. The pool must
    /// hold the blocks of `lineage`.
    pub fn from_state(
        cfg: &HyV4FlashConfig,
        s: &PrefixState,
        lineage: &[u32],
        draft: Option<Draft>,
    ) -> Result<Self> {
        if lineage.len() != s.row {
            return Err(bad(format!(
                "a state at row {} for a lineage of {}",
                s.row,
                lineage.len()
            )));
        }
        let mut cell = Self::new(cfg, draft)?;
        if cell.layers.len() != s.layers.len() {
            return Err(bad(format!(
                "a state of {} layers for a trunk of {}",
                s.layers.len(),
                cell.layers.len()
            )));
        }
        for (l, (c, sl)) in cell.layers.iter_mut().zip(&s.layers).enumerate() {
            let part = match (&*c, sl) {
                (CellLayer::Window(_), StateLayer::Window(r)) if r.len() == s.row => {
                    CellLayer::Window(r.clone())
                }
                (CellLayer::Index(ix), StateLayer::Tail(t)) if t.len() == s.row => {
                    CellLayer::Index(PagedIndexerState::at(ix.layer(), t.clone()))
                }
                _ => return Err(bad(format!("layer {l}: the state is not of this model"))),
            };
            *c = part;
        }
        match (cell.draft.as_mut(), &s.draft) {
            (None, None) => {}
            (Some(d), Some(ds)) => {
                fill_draft(d, ds)?;
                cell.segments = ds.segments.clone();
            }
            _ => return Err(bad("a state and a cell that differ in the MTP layer")),
        }
        cell.len = s.row;
        cell.recent = lineage[s.row.saturating_sub(cell.back)..].to_vec();
        Ok(cell)
    }

    /// Positions the trunk holds.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn layers(&self) -> &[CellLayer] {
        &self.layers
    }

    /// The trunk parts a step writes; `advance` records the step after.
    pub fn layers_mut(&mut self) -> &mut [CellLayer] {
        &mut self.layers
    }

    pub fn draft(&self) -> Option<&Draft> {
        self.draft.as_ref()
    }

    /// The draft rows below the trunk's last row are shared with stored
    /// states: they may be rewritten only after `truncate` cut them.
    pub fn draft_mut(&mut self) -> Option<&mut Draft> {
        self.draft.as_mut()
    }

    /// The last `max_ngram - 1` token ids (fewer near the start).
    pub fn recent(&self) -> &[u32] {
        &self.recent
    }

    /// Bytes the cell holds, its copied MTP segments included.
    pub fn nbytes(&self) -> u64 {
        let layers: usize = self.layers.iter().map(CellLayer::nbytes).sum();
        let segments: usize = self.segments.iter().map(|s| s.nbytes()).sum();
        (layers
            + self.draft.as_ref().map_or(0, Draft::nbytes)
            + segments
            + self.recent.len() * std::mem::size_of::<u32>()) as u64
    }

    /// Records a step that appended `ids` to every trunk layer.
    pub fn advance(&mut self, ids: &[u32]) -> Result<()> {
        let end = self.len + ids.len();
        if let Some(l) = self.layers.iter().position(|p| p.len() != end) {
            return Err(bad(format!(
                "layer {l} holds {} positions after a step to {end}",
                self.layers[l].len()
            )));
        }
        self.len = end;
        self.recent.extend_from_slice(ids);
        let over = self.recent.len().saturating_sub(self.back);
        self.recent.drain(..over);
        Ok(())
    }

    /// The draft rows `truncate(len)` leaves, or why it cannot.
    fn draft_keep(&self, len: usize) -> Result<usize> {
        let Some(d) = &self.draft else {
            return Ok(0);
        };
        if len == 0 {
            return Ok(0);
        }
        let keep = len - 1;
        if d.len() < keep {
            return Err(bad(format!(
                "the draft holds {} rows, a trunk of {len} needs {keep}",
                d.len()
            )));
        }
        if d.seed(keep).is_none() {
            return Err(bad(format!("no seed for trunk row {keep}")));
        }
        if !d.can_cut(keep) {
            return Err(bad(format!("the draft indexer cannot go back to {keep}")));
        }
        Ok(keep)
    }

    /// Whether `truncate(len, _)` would succeed.
    pub fn can_truncate(&self, len: usize) -> bool {
        if len >= self.len {
            return len == self.len;
        }
        let parts = len == 0
            || self.layers.iter().all(|p| match p {
                CellLayer::Window(r) => window_keeps(r.len(), r.kept(), r.window(), len),
                CellLayer::Index(ix) => index_keeps(ix.len(), ix.tail().held(), self.kpool, len),
            });
        parts && self.draft_keep(len).is_ok()
    }

    /// Drops the positions past `len`; `history` holds the sequence's token
    /// ids (at least `len`). A refusal changes nothing.
    pub fn truncate(&mut self, len: usize, history: &[u32]) -> Result<()> {
        if history.len() < len {
            return Err(bad(format!(
                "truncate to {len} with a history of {}",
                history.len()
            )));
        }
        if len == self.len {
            return Ok(());
        }
        if !self.can_truncate(len) {
            return Err(bad(format!(
                "truncate {} -> {len}: a window, a tail or the draft no longer holds the rows",
                self.len
            )));
        }
        let keep = self.draft_keep(len)?;
        if let Some(d) = self.draft.as_mut() {
            d.cut(keep)?;
            if len == 0 {
                d.seeds_mut().clear();
            } else {
                d.seeds_mut().cut(len);
            }
        }
        self.segments.truncate(keep / BLOCK_ROWS);
        for p in &mut self.layers {
            match p {
                CellLayer::Window(r) => r.truncate(len)?,
                CellLayer::Index(ix) => ix.truncate(len)?,
            }
        }
        self.len = len;
        self.recent = history[len.saturating_sub(self.back)..len].to_vec();
        Ok(())
    }

    /// The cell at this row in storage of its own; `None` off a block row,
    /// or with a draft not settled there (`len - 1` rows and the seed of
    /// row `len - 1`).
    pub fn state(&mut self) -> Result<Option<PrefixState>> {
        let b = self.len;
        if b == 0 || !b.is_multiple_of(BLOCK_ROWS) {
            return Ok(None);
        }
        let draft = if self.draft.is_some() {
            match self.draft_state(b)? {
                Some(d) => Some(d),
                None => return Ok(None),
            }
        } else {
            None
        };
        let layers = self
            .layers
            .iter()
            .map(|p| match p {
                CellLayer::Window(r) => StateLayer::Window(r.clone()),
                CellLayer::Index(ix) => StateLayer::Tail(ix.tail().clone()),
            })
            .collect();
        Ok(Some(PrefixState {
            row: b,
            layers,
            draft,
        }))
    }

    fn draft_state(&mut self, b: usize) -> Result<Option<DraftState>> {
        let Some(d) = self.draft.as_ref() else {
            return Ok(None);
        };
        let m = b - 1;
        let Some(seed) = d.seed(m).cloned() else {
            return Ok(None);
        };
        if d.len() != m || d.index().len() != m || d.index().pooled_len() != m / self.kpool {
            return Ok(None);
        }
        let per = BLOCK_ROWS / self.kpool;
        let full = m / BLOCK_ROWS;
        let kv = d.kv().rows()?;
        let width = kv.shape_at(1)?;
        let pooled = d.index().pooled_rows()?;
        let pw = pooled.shape_at(1)?;
        let rows = |a: &MxArray, w: i64, lo: usize, hi: usize| -> Result<MxArray> {
            a.slice(&[lo as i64, 0], &[hi as i64, w])?.deep_copy()
        };
        self.segments.truncate(full);
        let mut fresh = Vec::new();
        for s in self.segments.len()..full {
            fresh.push(Segment {
                kv: rows(kv, width, s * BLOCK_ROWS, (s + 1) * BLOCK_ROWS)?,
                pooled: rows(pooled, pw, s * per, (s + 1) * per)?,
            });
        }
        let first = full * BLOCK_ROWS;
        let open = if m > first {
            let closed = m / self.kpool;
            Some(OpenRows {
                kv: rows(kv, width, first, m)?,
                pooled: (closed > full * per)
                    .then(|| rows(pooled, pw, full * per, closed))
                    .transpose()?,
            })
        } else {
            None
        };
        let mut live: Vec<&MxArray> = fresh.iter().flat_map(|s| [&s.kv, &s.pooled]).collect();
        if let Some(o) = &open {
            live.push(&o.kv);
            live.extend(o.pooled.iter());
        }
        if !live.is_empty() {
            MxArray::eval_arrays_with_context(&live, "hy_v4_flash owner state")?;
        }
        let tail = d.index().tail().clone();
        self.segments.extend(fresh.into_iter().map(Arc::new));
        Ok(Some(DraftState {
            segments: self.segments.clone(),
            open,
            tail,
            seed,
        }))
    }
}

/// `parts` joined on rows; `None` when there are none.
fn join(parts: Vec<&MxArray>) -> Result<Option<MxArray>> {
    match parts.len() {
        0 => Ok(None),
        1 => Ok(Some(parts[0].clone())),
        _ => MxArray::concatenate_many(parts, Some(0)).map(Some),
    }
}

/// Fills the empty draft `d` with the MTP rows of `ds`.
fn fill_draft(d: &mut Draft, ds: &DraftState) -> Result<()> {
    let mut kv: Vec<&MxArray> = ds.segments.iter().map(|s| &s.kv).collect();
    let mut pooled: Vec<&MxArray> = ds.segments.iter().map(|s| &s.pooled).collect();
    if let Some(o) = &ds.open {
        kv.push(&o.kv);
        pooled.extend(o.pooled.iter());
    }
    let kv = join(kv)?;
    let pooled = join(pooled)?;
    d.fill(
        kv.as_ref(),
        pooled.as_ref(),
        ds.tail.clone(),
        ds.seed.clone(),
    )
}

/// Every owner cell of a load, by sequence; at most one active.
#[derive(Default)]
pub struct Owners {
    cells: HashMap<SeqId, OwnerCell>,
    active: Option<SeqId>,
}

impl Owners {
    pub fn new() -> Self {
        Self::default()
    }

    /// Cells in the table (a taken one is not).
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn contains(&self, seq: SeqId) -> bool {
        self.cells.contains_key(&seq)
    }

    pub fn active(&self) -> Option<SeqId> {
        self.active
    }

    /// Makes `seq` the active owner, parking the one before; a new owner
    /// gets `make()`.
    pub fn activate(
        &mut self,
        seq: SeqId,
        make: impl FnOnce() -> Result<OwnerCell>,
    ) -> Result<&mut OwnerCell> {
        let cell = match self.cells.entry(seq) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(make()?),
        };
        self.active = Some(seq);
        Ok(cell)
    }

    pub fn active_cell(&self) -> Option<&OwnerCell> {
        self.active.and_then(|s| self.cells.get(&s))
    }

    pub fn active_cell_mut(&mut self) -> Option<&mut OwnerCell> {
        self.active.and_then(|s| self.cells.get_mut(&s))
    }

    pub fn get(&self, seq: SeqId) -> Option<&OwnerCell> {
        self.cells.get(&seq)
    }

    pub fn get_mut(&mut self, seq: SeqId) -> Option<&mut OwnerCell> {
        self.cells.get_mut(&seq)
    }

    pub fn park(&mut self) {
        self.active = None;
    }

    /// Moves `seq`'s cell out (a wave holds it); it is parked first.
    pub fn take(&mut self, seq: SeqId) -> Option<OwnerCell> {
        if self.active == Some(seq) {
            self.active = None;
        }
        self.cells.remove(&seq)
    }

    /// Moves a taken cell back; refused while `seq` holds one.
    pub fn put(&mut self, seq: SeqId, cell: OwnerCell) -> Result<()> {
        if self.cells.contains_key(&seq) {
            return Err(bad(format!("owner {seq} already holds a cell")));
        }
        self.cells.insert(seq, cell);
        Ok(())
    }

    /// Drops `seq`'s cell.
    pub fn release(&mut self, seq: SeqId) -> Option<OwnerCell> {
        self.take(seq)
    }

    pub fn held_bytes(&self, seq: SeqId) -> u64 {
        self.cells.get(&seq).map_or(0, OwnerCell::nbytes)
    }

    pub fn bytes(&self) -> u64 {
        self.cells.values().map(OwnerCell::nbytes).sum()
    }

    pub fn clear(&mut self) {
        self.cells.clear();
        self.active = None;
    }
}
