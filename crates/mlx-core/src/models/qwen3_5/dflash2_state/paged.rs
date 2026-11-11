//! The DFlash2 state a parked replay point holds on the paged pool: the GDN
//! cell and the draft window at a block-aligned row (Splash
//! `CompositeStateLayout::cachedBytes`). The target KV is the prefix cache's
//! chain, never a copy here.

use std::sync::Arc;

use mlx_paged_attn::CacheRecency;
use napi::bindgen_prelude::*;

use crate::array::MxArray;

use super::super::arrays_cache::ArraysCache;
use super::super::dflash2::DFlash2ContextCache;
use super::super::layer_cache::Qwen3_5LayerCache;
use super::store::{EntryId, HybridStateStore, StateBytes};

/// GDN slots (full-attention slots are empty placeholders) and the draft
/// window, both at the same block-aligned row. Neither shares a buffer a
/// later turn writes: the window is a compact deep copy, and the GDN
/// arrays are snapshots a forward never writes in place.
pub(crate) struct PagedAnchor {
    pub(crate) linear: Vec<Qwen3_5LayerCache>,
    pub(crate) window: DFlash2ContextCache,
}

impl PagedAnchor {
    #[cfg(test)]
    pub(crate) fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        for cache in &self.linear {
            if matches!(cache, Qwen3_5LayerCache::Linear(_)) {
                cache.collect_arrays(out);
            }
        }
        self.window.collect_arrays(out);
    }
}

impl StateBytes for PagedAnchor {
    fn state_bytes(&self) -> u64 {
        let mut arrays = Vec::new();
        self.window.collect_arrays(&mut arrays);
        arrays.into_iter().fold(
            Qwen3_5LayerCache::linear_state_bytes(&self.linear),
            |sum, array| sum.saturating_add(array.nbytes() as u64),
        )
    }
}

/// One loaded model's parked DFlash2 states, plus the one this turn's
/// prefill published before its blocks registered: finalize keeps it only
/// once they did, an abort drops it.
pub(crate) struct DFlash2States {
    pub(crate) store: HybridStateStore<PagedAnchor>,
    pub(crate) unregistered: Option<EntryId>,
    /// The state the running turn resumes from or published at its own
    /// replay point (Splash `ActiveRequest::replayPoint`).
    pub(crate) replay_point: Option<EntryId>,
    /// The generation prompt of the turn's render, which places its replay
    /// point.
    pub(crate) generation_suffix: Option<Arc<[u32]>>,
}

impl DFlash2States {
    pub(crate) fn new(block_size: u32, recency: Arc<CacheRecency>) -> Self {
        let mut store = HybridStateStore::new(block_size);
        store.share_recency(recency);
        Self {
            store,
            unregistered: None,
            replay_point: None,
            generation_suffix: None,
        }
    }

    /// Drops every parked state.
    pub(crate) fn clear(&mut self) {
        self.store.clear();
        self.unregistered = None;
        self.replay_point = None;
    }

    /// Marks `id` as the running turn's replay point.
    pub(crate) fn use_replay_point(&mut self, id: EntryId) {
        self.end_replay_point();
        self.store.use_entry(id);
        self.replay_point = Some(id);
    }

    /// Ends the running turn's use of its replay point (Splash
    /// `StateCache::unuse`: it becomes the newest state).
    pub(crate) fn end_replay_point(&mut self) {
        if let Some(id) = self.replay_point.take() {
            self.store.unuse(id);
        }
    }

    /// A turn whose blocks will not register: its own published state goes
    /// (a state never outlives its block, Splash `StateCache::acquireBlock`).
    /// True when one went.
    pub(crate) fn drop_unregistered(&mut self) -> bool {
        self.end_replay_point();
        self.unregistered
            .take()
            .is_some_and(|id| self.store.remove(id))
    }
}

/// Puts deep copies of `anchor`'s GDN slots into `caches`' linear slots, in
/// layer order, and evaluates them; full-attention slots are left alone.
pub(crate) fn install_linear(
    caches: &mut [Qwen3_5LayerCache],
    anchor: &[Qwen3_5LayerCache],
) -> Result<()> {
    if caches.len() != anchor.len() {
        return Err(Error::from_reason(format!(
            "DFlash2 state holds {} layers for {} cache slots",
            anchor.len(),
            caches.len()
        )));
    }
    let mut arrays = Vec::new();
    for (slot, state) in caches.iter_mut().zip(anchor) {
        let Qwen3_5LayerCache::Linear(state) = state else {
            continue;
        };
        let (Some(conv), Some(rec)) = (state.get(0), state.get(1)) else {
            return Err(Error::from_reason("DFlash2 state is missing a GDN array"));
        };
        let mut gdn = ArraysCache::new(2);
        gdn.set(0, conv.deep_copy()?)?;
        gdn.set(1, rec.deep_copy()?)?;
        *slot = Qwen3_5LayerCache::Linear(gdn);
    }
    for slot in caches.iter() {
        if matches!(slot, Qwen3_5LayerCache::Linear(_)) {
            slot.collect_arrays(&mut arrays);
        }
    }
    if !arrays.is_empty() {
        MxArray::eval_arrays(&arrays)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;
    use crate::models::qwen3_5::dflash2_state::store::Lineage;

    fn anchor() -> PagedAnchor {
        let mut gdn = ArraysCache::new(2);
        gdn.set(
            0,
            MxArray::zeros(&[1, 3, 64], Some(DType::BFloat16)).unwrap(),
        )
        .unwrap();
        gdn.set(
            1,
            MxArray::zeros(&[1, 2, 16, 16], Some(DType::Float32)).unwrap(),
        )
        .unwrap();
        let config = crate::models::qwen3_5::dflash2::tiny_dflash2_config();
        PagedAnchor {
            linear: vec![
                Qwen3_5LayerCache::Linear(gdn),
                Qwen3_5LayerCache::new_full_attention(),
            ],
            window: DFlash2ContextCache::new(&config),
        }
    }

    /// A parked state holds its GDN cell and window only: its bytes do not
    /// grow with the lineage it sits on (the chain is pool blocks, charged
    /// once). Pre-pages a 78.6K state also held ~2.44 GiB of target KV.
    /// Mutation KV1 (the entry charges its chain's KV rows) → red.
    #[test]
    fn parked_state_bytes_have_no_kv_term() {
        let mut store = HybridStateStore::<PagedAnchor>::with_state_budget(16, u64::MAX);
        let want = 3 * 64 * 2 + 2 * 16 * 16 * 4;
        let short: Vec<u32> = (0..1_600).collect();
        let long: Vec<u32> = (0..78_592).collect();
        let (a, _) = store.insert(Lineage::text(&short, 0), anchor()).unwrap();
        let (b, _) = store.insert(Lineage::text(&long, 0), anchor()).unwrap();
        assert_eq!(store.get(a).unwrap().state_bytes(), want);
        assert_eq!(store.get(b).unwrap().state_bytes(), want);
        assert_eq!(store.bytes().state_bytes, 2 * want);
    }
}
