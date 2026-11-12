//! The DFlash2 state a parked replay point holds on the paged pool: the GDN
//! cell and the draft window at a block-aligned row (Splash
//! `CompositeStateLayout::cachedBytes`). The target KV is the prefix cache's
//! chain, never a copy here.

use std::collections::HashMap;
use std::sync::Arc;

use mlx_paged_attn::CacheRecency;
use napi::bindgen_prelude::*;

use crate::array::MxArray;

use super::super::arrays_cache::ArraysCache;
use super::super::config::Qwen3_5Config;
use super::super::dflash2::{DFlash2Config, DFlash2ContextCache};
use super::super::draft_context::DraftContextWindow;
use super::super::layer_cache::Qwen3_5LayerCache;
use super::plan::StatePlan;
use super::store::{Entry, EntryId, HybridStateStore, StateBytes};

/// GDN slots (full-attention slots are empty placeholders) and the 4-bit
/// draft context window, both at the same block-aligned row. Neither shares
/// a buffer a later turn writes: the window is a snapshot, and the GDN
/// arrays are snapshots a forward never writes in place.
pub(crate) struct PagedAnchor {
    pub(crate) linear: Vec<Qwen3_5LayerCache>,
    pub(crate) window: DraftContextWindow,
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
        Qwen3_5LayerCache::linear_state_bytes(&self.linear).saturating_add(self.window.nbytes())
    }
}

/// Bytes of one state of `target` with `draft`, its lineage aside (Splash
/// `CompositeStateLayout::cachedBytes`: one GDN cell and the window).
pub(crate) fn anchor_bytes(target: &Qwen3_5Config, draft: &DFlash2Config) -> u64 {
    target
        .recurrent_state_bytes()
        .saturating_add(draft.context_window_bytes())
}

/// What the turn states published, retired and missed (Splash
/// `Engine` and `CacheLookupSnapshot` counters).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StateCounters {
    pub(crate) checkpoint_publications: u64,
    pub(crate) replay_state_publications: u64,
    pub(crate) junction_materializations: u64,
    pub(crate) checkpoint_publication_failures: u64,
    pub(crate) replay_state_publication_failures: u64,
    pub(crate) junction_materialization_failures: u64,
    pub(crate) deduplicated_state_publications: u64,
    pub(crate) recycled_state_publications: u64,
    pub(crate) checkpoint_retirements: u64,
    pub(crate) lazy_junctions: u64,
    pub(crate) checkpoint_evictions: u64,
    pub(crate) evictions: u64,
    /// States no RAM copy took, written from the lane to the SSD tier
    /// (Splash `diskStatePublications`).
    pub(crate) disk_state_publications: u64,
}

/// The running DFlash2 turn's state points (Splash `Request`
/// `stateBoundaries`, `stateBoundaryCursor`, `latestCheckpoint`).
pub(crate) struct TurnStates {
    pub(crate) plan: StatePlan,
    /// The next point of `plan` the prefill reaches.
    pub(crate) cursor: usize,
    /// Checkpoints the prefill runs past without a capture (Splash
    /// `reserveCaptures` `passed`).
    pub(crate) passed: Vec<u32>,
    /// What one capture of this turn holds: a GDN cell, the window and the
    /// deepest point's lineage vectors. Known before the draft leaves the
    /// model for the prefill.
    pub(crate) anchor_bytes: u64,
    /// The lane's rolling checkpoint and its publication.
    pub(crate) latest_checkpoint: Option<(EntryId, u64)>,
    pub(crate) cache_salt: u64,
    /// The KV hit the admission found a state below, 0 for none (Splash
    /// `CacheLookup::junctionBoundary`).
    pub(crate) junction: u32,
    /// The prompt's tokens; a resumed lane replays past them (Splash
    /// `Request::promptTokens` beside `replayTokens`).
    pub(crate) prompt_tokens: u32,
    /// The turn reuses the cache: without it the turn plans no point.
    pub(crate) reuse: bool,
}

/// A host refusal of a tapped prefill's chunk (Splash
/// `Engine::suspendForGrowth`): the lane vacates and replays its prompt.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefillYield {
    /// Where the refused chunk would have ended (Splash `Engine::workEndOf`).
    pub(crate) work_end: u32,
    pub(crate) failure: Option<crate::engine::governor::governor::Failure>,
}

/// One loaded model's parked DFlash2 states and the running turn's points.
pub(crate) struct DFlash2States {
    pub(crate) store: HybridStateStore<PagedAnchor>,
    pub(crate) turn: Option<TurnStates>,
    /// The block whose state the running turn resumes from or published at
    /// its own replay point (Splash `ActiveRequest::replayPoint`).
    pub(crate) replay_point: Option<u64>,
    /// The generation prompt of the turn's render, which places its replay
    /// point.
    pub(crate) generation_suffix: Option<Arc<[u32]>>,
    /// The lane's context window a restore filled, which the turn's tapped
    /// prefill continues (Splash `QwenStateStorage::restore` writes the
    /// lane's window).
    pub(crate) lane_window: Option<DraftContextWindow>,
    pub(crate) counters: StateCounters,
    /// The finalize released the DFlash2 turn's lane; its history save
    /// takes it.
    pub(crate) released: bool,
    /// The block of the stored state each autoregressive turn's GDN prime
    /// restored from, in use until the turn ends (Splash
    /// `StateCache::useState`, `unuse`).
    pub(crate) restore_uses: HashMap<u32, u64>,
    /// A released lane's draft ring and context window, which the next
    /// DFlash2 lane takes (Splash `QwenStateStorage::releaseLane`,
    /// `acquire`).
    pub(crate) idle_lane: IdleLane,
    /// The turn's own tapped prefill on the serial lane runs: a host
    /// refusal of its chunk suspends the lane.
    pub(crate) prefill_suspends: bool,
    /// The refusal that suspended the tapped prefill, until the prefill's
    /// owner vacates the lane.
    pub(crate) prefill_yield: Option<PrefillYield>,
    /// The model's SSD tier for its states, when a load opened one.
    pub(crate) disk: Option<super::disk::StateDisk>,
    /// The disk state the turn's lane read, until the admission promotes
    /// or discards it (Splash `Engine::pollRestores`).
    pub(crate) disk_read: Option<DiskRead>,
}

/// A disk state a lane read (Splash `StateRestore`): the window the read
/// took when the lane's turn does not keep it.
pub(crate) struct DiskRead {
    pub(crate) id: EntryId,
    pub(crate) window: Option<DraftContextWindow>,
}

/// A released lane's buffers: the draft ring with its rows dropped and the
/// context window, both kept for the next lane. Both are materialized: a
/// restore credits their bytes as storage the lane already holds (Splash
/// `QwenStateStorage::releaseLane` pools only allocated `DFlashDraftRing`s).
#[derive(Default)]
pub(crate) struct IdleLane {
    pub(crate) ring: Option<DFlash2ContextCache>,
    pub(crate) window: Option<DraftContextWindow>,
}

impl IdleLane {
    /// Keeps `ring`'s buffers, its rows dropped; a ring whose buffers fail
    /// to materialize is dropped.
    pub(crate) fn give_ring(&mut self, ring: Option<DFlash2ContextCache>) {
        if let Some(mut ring) = ring
            && ring.eval().is_ok()
        {
            ring.reset();
            self.ring = Some(ring);
        }
    }

    /// Keeps `window`'s buffer, emptied; a window that fails to
    /// materialize is dropped.
    pub(crate) fn give_window(&mut self, window: Option<DraftContextWindow>) {
        if let Some(mut window) = window
            && window.eval().is_ok()
        {
            window.restart(0);
            self.window = Some(window);
        }
    }

    /// The idle ring, else a new one.
    pub(crate) fn take_ring(&mut self, config: &DFlash2Config) -> DFlash2ContextCache {
        self.ring
            .take()
            .unwrap_or_else(|| DFlash2ContextCache::new(config))
    }

    /// The idle window, else a new one.
    pub(crate) fn take_window(&mut self, config: &DFlash2Config) -> Result<DraftContextWindow> {
        self.window
            .take()
            .map_or_else(|| config.context_window(), Ok)
    }
}

impl DFlash2States {
    pub(crate) fn new(block_size: u32, recency: Arc<CacheRecency>) -> Self {
        let mut store = HybridStateStore::new(block_size);
        store.share_recency(recency);
        Self {
            store,
            turn: None,
            replay_point: None,
            generation_suffix: None,
            lane_window: None,
            counters: StateCounters::default(),
            released: false,
            restore_uses: HashMap::new(),
            idle_lane: IdleLane::default(),
            prefill_suspends: false,
            prefill_yield: None,
            disk: None,
            disk_read: None,
        }
    }

    /// Drops every parked state.
    pub(crate) fn clear(&mut self) {
        self.store.clear();
        self.turn = None;
        self.replay_point = None;
        self.lane_window = None;
        self.disk_read = None;
        self.restore_uses.clear();
        self.idle_lane = IdleLane::default();
    }

    /// Marks `id`'s block as the running turn's replay point (Splash
    /// `StateCache::useState`).
    pub(crate) fn use_replay_point(&mut self, id: EntryId) {
        if let Some(block) = self.store.get(id).map(Entry::mark) {
            self.use_replay_block(block);
        }
    }

    /// Marks `block` as the running turn's replay point, with a state there
    /// or yet to be published (Splash `StateCache::useState`).
    pub(crate) fn use_replay_block(&mut self, block: u64) {
        self.end_replay_point();
        self.store.use_state(block);
        self.replay_point = Some(block);
    }

    /// Ends the running turn's use of its replay point (Splash
    /// `StateCache::unuse`: it becomes the newest state).
    pub(crate) fn end_replay_point(&mut self) {
        if let Some(block) = self.replay_point.take() {
            self.store.end_use(block);
        }
    }

    /// An autoregressive turn on `seq` restored its GDN cell from stored
    /// state `id` (Splash `StateCache::useState`, which makes it ordinary).
    pub(crate) fn use_restored(&mut self, seq: u32, id: EntryId) {
        self.end_restored(seq);
        if let Some(block) = self.store.get(id).map(Entry::mark) {
            self.store.use_state(block);
            self.restore_uses.insert(seq, block);
        }
    }

    /// Ends `seq`'s use of the stored state it restored from: the state
    /// becomes the newest of its class (Splash `StateCache::unuse`).
    pub(crate) fn end_restored(&mut self, seq: u32) {
        if let Some(block) = self.restore_uses.remove(&seq) {
            self.store.end_use(block);
        }
    }

    pub(crate) fn end_all_restored(&mut self) {
        for (_, block) in std::mem::take(&mut self.restore_uses) {
            self.store.end_use(block);
        }
    }

    /// The turn ended (finalize, abort, cancel), after its request was
    /// released: its replay point is no longer in use, stamped newer than
    /// its KV tail, and its points go (Splash `Engine::release`). What it
    /// published stays: each state was published only once its block
    /// registered.
    pub(crate) fn end_turn(&mut self) {
        self.end_replay_point();
        self.turn = None;
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

/// Moves the GDN slots a disk read decoded into `caches`' linear slots, in
/// layer order, and evaluates them; full-attention slots are left alone.
pub(crate) fn adopt_linear(
    caches: &mut [Qwen3_5LayerCache],
    linear: Vec<Qwen3_5LayerCache>,
) -> Result<()> {
    if caches.len() != linear.len() {
        return Err(Error::from_reason(format!(
            "a DFlash2 disk state holds {} layers for {} cache slots",
            linear.len(),
            caches.len()
        )));
    }
    for (slot, state) in caches.iter_mut().zip(linear) {
        if matches!(state, Qwen3_5LayerCache::Linear(_)) {
            *slot = state;
        }
    }
    let mut arrays = Vec::new();
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
    use crate::models::qwen3_5::dflash2_state::store::{Lineage, StateClass};

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
            window: config.context_window().unwrap(),
        }
    }

    /// A parked state holds its GDN cell and window, plus its lineage's
    /// token and chain vectors: no KV term grows with the lineage it sits on
    /// (the chain is pool blocks, charged once). Pre-pages a 78.6K state
    /// also held ~2.44 GiB of target KV.
    /// Mutation KV1 (the entry charges its chain's KV rows) → red.
    #[test]
    fn parked_state_bytes_have_no_kv_term() {
        let mut store = HybridStateStore::<PagedAnchor>::new(16);
        let want = 3 * 64 * 2 + 2 * 16 * 16 * 4 + DraftContextWindow::bytes(64, 8);
        let short: Vec<u32> = (0..1_600).collect();
        let long: Vec<u32> = (0..78_592).collect();
        let a = store
            .insert(Lineage::text(&short, 0), anchor(), StateClass::Ordinary)
            .unwrap()
            .id;
        let b = store
            .insert(Lineage::text(&long, 0), anchor(), StateClass::Ordinary)
            .unwrap()
            .id;
        let lineage = |tokens: u64| tokens * 4 + tokens / 16 * 8;
        assert_eq!(store.get(a).unwrap().state_bytes(), want + lineage(1_600));
        assert_eq!(store.get(b).unwrap().state_bytes(), want + lineage(78_592));
    }

    /// A 27B state is one GDN cell and the 4-bit window (Splash
    /// `CompositeStateLayout::cachedBytes`); the store adds its lineage
    /// vectors (`state_bytes_count_the_lineage`).
    /// Mutation WB1 (the window counts bf16 rows) → red.
    #[test]
    fn state_bytes_27b_geometry() {
        let target = crate::models::qwen3_5::dflash2_decode::tests::qwen38_27b_cfg();
        let draft = crate::models::qwen3_5::dflash2::dflash2_config_with_context(5120, 2048);
        assert_eq!(target.recurrent_state_bytes(), 153_944_064);
        assert_eq!(DraftContextWindow::bytes(5120, 2048), 5_898_240);
        assert_eq!(anchor_bytes(&target, &draft), 159_842_304);
    }
}
