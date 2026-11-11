//! Flat-lane payloads of the DFlash2 state store. A parked end-of-turn
//! state owns the flat caches and draft window it was moved out of; nothing
//! else holds a handle a later turn could write through.

use std::sync::Arc;

use mlx_paged_attn::CacheRecency;
use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::transformer::KVCache;

use super::super::arrays_cache::ArraysCache;
use super::super::dflash2::DFlash2ContextCache;
use super::super::gated_delta_net::GdnCell;
use super::super::layer_cache::Qwen3_5LayerCache;
use super::store::{Anchor, Backing, HybridStateStore, KvSource, Shrink, StateBytes};

/// Capacity slack (rows past the used ones) above which a parked KV buffer
/// is worth copying down.
pub(crate) const COMPACT_SLACK_ROWS: usize = 4096;
/// Rows a compacted buffer keeps past the used ones.
pub(crate) const COMPACT_SPARE_ROWS: usize = 256;

fn arrays_bytes<'a>(arrays: impl IntoIterator<Item = &'a MxArray>) -> u64 {
    arrays
        .into_iter()
        .fold(0, |sum, array| sum.saturating_add(array.nbytes() as u64))
}

fn kv_cache_bytes(kv: &KVCache) -> u64 {
    arrays_bytes(
        [
            kv.keys_ref(),
            kv.values_ref(),
            kv.key_scales_ref(),
            kv.value_scales_ref(),
        ]
        .into_iter()
        .flatten(),
    )
}

/// The full-attention caches of a parked state, each with its decoder-layer
/// index.
pub(crate) struct FlatKv {
    layers: Vec<(usize, KVCache)>,
}

impl FlatKv {
    pub(crate) fn layers(&self) -> &[(usize, KVCache)] {
        &self.layers
    }

    /// Drops rows past `rows` from every layer's offset; the buffers stay.
    pub(crate) fn trim(&mut self, rows: usize) {
        let rows = i32::try_from(rows).unwrap_or(i32::MAX);
        for (_, kv) in &mut self.layers {
            kv.trim(rows);
        }
    }
}

impl KvSource for FlatKv {
    fn rows(&self) -> usize {
        self.layers
            .iter()
            .map(|(_, kv)| kv.get_offset().max(0) as usize)
            .min()
            .unwrap_or(0)
    }

    fn bytes(&self) -> u64 {
        self.layers
            .iter()
            .fold(0, |sum, (_, kv)| sum.saturating_add(kv_cache_bytes(kv)))
    }

    fn shrink_plan(&self, rows: usize) -> Option<Shrink> {
        let keep = rows.min(self.rows());
        let mut plan = Shrink {
            frees: 0,
            transient: 0,
        };
        for (_, kv) in &self.layers {
            let capacity = usize::try_from(kv.capacity().ok()?).ok()?;
            if capacity.saturating_sub(keep) <= COMPACT_SLACK_ROWS {
                continue;
            }
            let bytes = kv_cache_bytes(kv);
            let kept = (bytes / capacity as u64).saturating_mul((keep + COMPACT_SPARE_ROWS) as u64);
            plan.frees = plan.frees.saturating_add(bytes.saturating_sub(kept));
            plan.transient = plan.transient.saturating_add(kept);
        }
        (plan.frees > 0).then_some(plan)
    }

    fn shrink(&mut self, rows: usize) -> Result<()> {
        let keep = rows.min(self.rows());
        for (_, kv) in &mut self.layers {
            kv.trim(keep as i32);
            let capacity = usize::try_from(kv.capacity()?).unwrap_or(0);
            if capacity.saturating_sub(keep) > COMPACT_SLACK_ROWS {
                kv.compact((keep + COMPACT_SPARE_ROWS) as i64)?;
            }
        }
        Ok(())
    }
}

/// GDN slots and the draft window at the end of a turn. The next decode
/// position is the window's logical length.
pub(crate) struct FlatEnd {
    linear: Vec<(usize, ArraysCache)>,
    context: DFlash2ContextCache,
}

impl FlatEnd {
    pub(crate) fn context(&self) -> &DFlash2ContextCache {
        &self.context
    }

    pub(crate) fn next_position(&self) -> i32 {
        self.context.logical_len()
    }
}

#[cfg(test)]
impl FlatEnd {
    /// Every GDN slot in layer order, then the draft window's arrays.
    pub(crate) fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        for (_, cache) in &self.linear {
            out.extend(cache.get(0));
            out.extend(cache.get(1));
        }
        self.context.collect_arrays(out);
    }
}

impl StateBytes for FlatEnd {
    fn state_bytes(&self) -> u64 {
        let mut arrays = Vec::new();
        for (_, cache) in &self.linear {
            arrays.extend(cache.get(0));
            arrays.extend(cache.get(1));
        }
        self.context.collect_arrays(&mut arrays);
        arrays_bytes(arrays)
    }
}

/// GDN cell and draft window snapshot at a block-aligned row.
pub(crate) struct FlatAnchor {
    pub(crate) cell: GdnCell,
    pub(crate) window: DFlash2ContextCache,
}

impl StateBytes for FlatAnchor {
    fn state_bytes(&self) -> u64 {
        let mut arrays: Vec<&MxArray> = self.cell.conv.iter().chain(&self.cell.rec).collect();
        self.window.collect_arrays(&mut arrays);
        arrays_bytes(arrays)
    }
}

pub(crate) struct FlatBacking;

impl Backing for FlatBacking {
    type Kv = FlatKv;
    type End = FlatEnd;
    type Anchor = FlatAnchor;
}

/// One loaded model's parked flat states, plus what a park of its live flat
/// state takes along: the cache salt of the turn that built it (its key) and
/// the capture C(R) on its lineage, whose KV rows are the live ones.
pub(crate) struct FlatStates {
    pub(crate) store: HybridStateStore<FlatBacking>,
    pub(crate) live_salt: u64,
    pub(crate) live_anchor: Option<Anchor<FlatAnchor>>,
    /// Stamp of the live state's last use (its turn end or install); a park
    /// keeps it.
    pub(crate) live_last_used: u64,
}

impl FlatStates {
    pub(crate) fn new(block_size: u32, recency: Arc<CacheRecency>) -> Self {
        let mut store = HybridStateStore::new(block_size);
        store.share_recency(recency);
        Self {
            store,
            live_salt: 0,
            live_anchor: None,
            live_last_used: 0,
        }
    }

    /// Drops every parked state and the live state's anchor.
    pub(crate) fn clear(&mut self) {
        self.store.clear();
        self.live_anchor = None;
    }

    /// Moves the store out, leaving an empty one with the same geometry.
    pub(crate) fn take_store(&mut self) -> HybridStateStore<FlatBacking> {
        let mut empty =
            HybridStateStore::with_state_budget(self.store.block_size(), self.store.state_budget());
        empty.share_recency(self.store.recency());
        std::mem::replace(&mut self.store, empty)
    }
}

/// Splits moved flat caches into their full-attention and GDN halves.
pub(crate) fn split_flat_caches(
    caches: Vec<Qwen3_5LayerCache>,
    context: DFlash2ContextCache,
) -> (FlatKv, FlatEnd) {
    let mut layers = Vec::new();
    let mut linear = Vec::new();
    for (index, cache) in caches.into_iter().enumerate() {
        match cache {
            Qwen3_5LayerCache::FullAttention(kv) => layers.push((index, kv)),
            Qwen3_5LayerCache::Linear(gdn) => linear.push((index, gdn)),
        }
    }
    (FlatKv { layers }, FlatEnd { linear, context })
}

/// The full-attention half of moved flat caches; the GDN slots are dropped.
pub(crate) fn flat_attention_kv(caches: Vec<Qwen3_5LayerCache>) -> FlatKv {
    let layers = caches
        .into_iter()
        .enumerate()
        .filter_map(|(index, cache)| match cache {
            Qwen3_5LayerCache::FullAttention(kv) => Some((index, kv)),
            Qwen3_5LayerCache::Linear(_) => None,
        })
        .collect();
    FlatKv { layers }
}

/// Reassembles the per-layer caches in decoder order, plus the draft window.
pub(crate) fn join_flat_caches(
    kv: FlatKv,
    end: FlatEnd,
) -> Result<(Vec<Qwen3_5LayerCache>, DFlash2ContextCache)> {
    let total = kv.layers.len() + end.linear.len();
    let mut slots: Vec<Option<Qwen3_5LayerCache>> = (0..total).map(|_| None).collect();
    let placed = kv
        .layers
        .into_iter()
        .map(|(index, cache)| (index, Qwen3_5LayerCache::FullAttention(cache)))
        .chain(
            end.linear
                .into_iter()
                .map(|(index, cache)| (index, Qwen3_5LayerCache::Linear(cache))),
        );
    for (index, cache) in placed {
        match slots.get_mut(index) {
            Some(slot @ None) => *slot = Some(cache),
            _ => {
                return Err(Error::from_reason(format!(
                    "parked DFlash2 state has no free slot for layer {index} of {total}"
                )));
            }
        }
    }
    let caches = slots
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| Error::from_reason("parked DFlash2 state is missing a layer"))?;
    Ok((caches, end.context))
}

/// Per-layer caches in decoder order from a parked state's full-attention
/// caches and a GDN cell: every slot the KV does not hold is a GDN layer and
/// takes the cell's next conv history and recurrent state, deep-copied so
/// the cell stays the anchor's alone.
pub(crate) fn join_flat_anchor(kv: FlatKv, cell: &GdnCell) -> Result<Vec<Qwen3_5LayerCache>> {
    if cell.conv.len() != cell.rec.len() {
        return Err(Error::from_reason(format!(
            "DFlash2 GDN cell holds {} conv histories for {} states",
            cell.conv.len(),
            cell.rec.len()
        )));
    }
    let total = kv.layers.len() + cell.conv.len();
    let mut slots: Vec<Option<Qwen3_5LayerCache>> = (0..total).map(|_| None).collect();
    for (index, cache) in kv.layers {
        match slots.get_mut(index) {
            Some(slot @ None) => *slot = Some(Qwen3_5LayerCache::FullAttention(cache)),
            _ => {
                return Err(Error::from_reason(format!(
                    "parked DFlash2 state has no free slot for layer {index} of {total}"
                )));
            }
        }
    }
    let mut states = cell.conv.iter().zip(&cell.rec);
    for slot in slots.iter_mut().filter(|slot| slot.is_none()) {
        let (conv, rec) = states
            .next()
            .ok_or_else(|| Error::from_reason("DFlash2 GDN cell is missing a layer"))?;
        let mut gdn = ArraysCache::new(2);
        gdn.set(0, conv.deep_copy()?)?;
        gdn.set(1, rec.deep_copy()?)?;
        *slot = Some(Qwen3_5LayerCache::Linear(gdn));
    }
    let caches = slots.into_iter().flatten().collect::<Vec<_>>();
    let mut arrays = Vec::new();
    for cache in &caches {
        if let Qwen3_5LayerCache::Linear(gdn) = cache {
            arrays.extend(gdn.get(0));
            arrays.extend(gdn.get(1));
        }
    }
    if !arrays.is_empty() {
        MxArray::eval_arrays(&arrays)?;
    }
    Ok(caches)
}
