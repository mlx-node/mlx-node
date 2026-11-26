//! The DFlash2 state a parked replay point holds on the paged pool: the GDN
//! cell and the draft window at a block-aligned row (Splash
//! `CompositeStateLayout::cachedBytes`). The target KV is the prefix cache's
//! chain, never a copy here.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use mlx_paged_attn::CacheRecency;
use napi::bindgen_prelude::*;

use crate::array::MxArray;

use super::super::arrays_cache::ArraysCache;
use super::super::config::Qwen3_5Config;
use super::super::dflash2::{DFlash2Config, DFlash2ContextCache, DFlash2Model};
use super::super::dflash2_decode::{DraftedProposals, append_kept_taps};
use super::super::draft_context::DraftContextWindow;
use super::super::gated_delta_net::GdnLayerTape;
use super::super::gdn_blob::GdnStateBlobs;
use super::super::layer_cache::Qwen3_5LayerCache;
use super::plan::StatePlan;
use super::store::{Entry, EntryId, HybridStateStore, StateBytes};
use crate::engine::scheduled_verify::ScheduledVerifyBatch;

/// GDN slots (full-attention slots are empty placeholders) and the 4-bit
/// draft context window, both at the same block-aligned row; a load without
/// a draft keeps the GDN cell alone. Neither shares a buffer a later turn
/// writes: the window is a snapshot, and the GDN arrays are snapshots a
/// forward never writes in place.
pub(crate) struct PagedAnchor {
    pub(crate) linear: Vec<Qwen3_5LayerCache>,
    pub(crate) window: Option<DraftContextWindow>,
}

impl PagedAnchor {
    #[cfg(test)]
    pub(crate) fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        for cache in &self.linear {
            if matches!(cache, Qwen3_5LayerCache::Linear(_)) {
                cache.collect_arrays(out);
            }
        }
        if let Some(window) = &self.window {
            window.collect_arrays(out);
        }
    }
}

impl StateBytes for PagedAnchor {
    fn state_bytes(&self) -> u64 {
        Qwen3_5LayerCache::linear_state_bytes(&self.linear)
            .saturating_add(self.window.as_ref().map_or(0, DraftContextWindow::nbytes))
    }
}

/// Bytes of one state of `target` with `draft`, its lineage aside (Splash
/// `CompositeStateLayout::cachedBytes`: one GDN cell and the window; no
/// window without a draft).
pub(crate) fn anchor_bytes(target: &Qwen3_5Config, draft: Option<&DFlash2Config>) -> u64 {
    target
        .recurrent_state_bytes()
        .saturating_add(draft.map_or(0, DFlash2Config::context_window_bytes))
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

/// A running turn's state points on one sequence (Splash `Request`
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
    /// Per-block image keys of the turn's lineage, none on a text chain
    /// (Splash `CacheIdentity::images`): its blocks register and its states
    /// publish under them.
    pub(crate) extra_keys: Vec<Vec<u64>>,
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

/// One loaded model's stored states and the running turns' points.
pub(crate) struct DFlash2States {
    pub(crate) store: HybridStateStore<PagedAnchor>,
    /// Each running turn's points by its sequence (Splash `Request`): the
    /// serial lane's is sequence 0, a scheduled lane's its own.
    pub(crate) turns: HashMap<u32, TurnStates>,
    /// Block-aligned prompt head boundary per active sequence (`0` / absent when
    /// the turn has no prompt head boundary). Retained across turn planning and
    /// cleared on turn end or state clear.
    pub(crate) turn_head_boundaries: HashMap<u32, u32>,
    /// Explicit prompt-head provenance for stored states: entries published with
    /// `is_head == true` (or adopted from a persistent SSD tier at load). Excludes
    /// non-head ordinary states such as preemption-promoted reply checkpoints or
    /// lazy junctions so they never suppress a lineage's first durable prompt head.
    pub(crate) prompt_heads: HashSet<EntryId>,
    /// The block whose state each sequence's turn resumes from or published
    /// at its own replay point (Splash `ActiveRequest::replayPoint`); a
    /// vacated lane keeps it.
    pub(crate) replay_points: HashMap<u32, u64>,
    /// The generation prompt of the turn's render, which places its replay
    /// point.
    pub(crate) generation_suffix: Option<Arc<[u32]>>,
    /// The lane's context window a restore filled, which the turn's tapped
    /// prefill continues (Splash `QwenStateStorage::restore` writes the
    /// lane's window).
    pub(crate) lane_window: Option<DraftContextWindow>,
    /// Each scheduled sequence's context window on a draft load, which its
    /// prefill slices continue (Splash keeps one lane per request).
    pub(crate) seq_windows: HashMap<u32, DraftContextWindow>,
    /// Each scheduled sequence's DFlash2 decode lane on a draft load
    /// (Splash keeps one lane per request).
    pub(crate) seq_lanes: HashMap<u32, DraftLane>,
    /// The lanes' verify wave between its write and its commit: the
    /// tickets of every row in it.
    pub(crate) lane_batch: Option<ScheduledVerifyBatch>,
    /// The cells each autoregressive peer of the open wave had before its
    /// row's forward: a zero keep or an abort puts them back.
    pub(crate) peer_cells: HashMap<u32, Vec<Qwen3_5LayerCache>>,
    /// The decode cache limit of a lane's 8-row verify, held while any lane
    /// lives.
    pub(crate) lane_decode_limit: Option<crate::cache_limit::DecodeCacheLimitGuard>,
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
    /// The disk state an autoregressive prime read and checked before it
    /// armed the state's chain, until the GDN prime takes it.
    pub(crate) state_read: Option<StateRead>,
}

/// A disk state's GDN cell, read before its chain is restored.
pub(crate) struct StateRead {
    pub(crate) id: EntryId,
    pub(crate) linear: Vec<Qwen3_5LayerCache>,
}

/// A disk state a lane read (Splash `StateRestore`): the window the read
/// took when the lane's turn does not keep it.
pub(crate) struct DiskRead {
    pub(crate) id: EntryId,
    pub(crate) window: Option<DraftContextWindow>,
}

/// A scheduled sequence's DFlash2 decode state: its draft ring, its two GDN
/// cells with their pending rows, its cycle's draft and its open verify.
pub(crate) struct DraftLane {
    pub(crate) ring: DFlash2ContextCache,
    /// The position the next ring append is stamped with: the sequence's
    /// committed rows.
    pub(crate) next_position: i32,
    pub(crate) gdn: Option<GdnStateBlobs>,
    /// The cycle's whole draft: the proposals past the driver's cap are the
    /// verify's tail rows.
    pub(crate) drafted: Option<DraftedProposals>,
    pub(crate) open: Option<OpenVerify>,
    pub(crate) verifies: LaneVerifies,
    /// Planes GEMM calls of the committed verifies: a wave's calls count
    /// once for each of its lanes.
    pub(crate) planes_calls: u64,
    /// The part of `planes_calls` from verify waves.
    pub(crate) wave_planes_calls: u64,
}

/// The cycles a lane committed, by the verify that ran them: in a verify
/// wave, by the wave's lanes, or alone (its `[1, 8]` lane kernel). A verify
/// its step retracted, or that kept no row, is not counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LaneVerifies {
    /// Waves of 2, 3 and 4 lanes.
    pub(crate) waves: [u64; 3],
    pub(crate) alone: u64,
}

impl LaneVerifies {
    /// One cycle verified by `lanes` lanes together (1: alone).
    pub(crate) fn record(&mut self, lanes: usize) {
        if lanes <= 1 {
            self.alone += 1;
        } else if let Some(count) = self.waves.get_mut(lanes - 2) {
            *count += 1;
        }
    }

    pub(crate) fn wave_cycles(&self) -> u64 {
        self.waves.iter().sum()
    }

    /// `key=value` fields, space separated.
    pub(crate) fn fields(&self) -> String {
        format!(
            "wave_cycles={} lanes2={} lanes3={} lanes4={} alone_cycles={}",
            self.wave_cycles(),
            self.waves[0],
            self.waves[1],
            self.waves[2],
            self.alone
        )
    }
}

/// A lane's verify between its write and its commit; its ticket is in
/// [`DFlash2States::lane_batch`].
pub(crate) struct OpenVerify {
    pub(crate) tape: Vec<Option<GdnLayerTape>>,
    pub(crate) tapped: Vec<MxArray>,
    pub(crate) ids: Vec<u32>,
    /// The lanes that verified together, this one included.
    pub(crate) lanes: usize,
    /// Planes GEMM calls of this verify; of the whole wave for a wave lane.
    pub(crate) planes_calls: u64,
}

impl DraftLane {
    /// Appends the kept rows' taps to the ring at the lane's position and
    /// moves the position past them.
    pub(crate) fn append_kept(
        &mut self,
        draft: &DFlash2Model,
        tapped: &[MxArray],
        token_ids: &[u32],
    ) -> Result<()> {
        self.next_position =
            append_kept_taps(draft, &mut self.ring, self.next_position, tapped, token_ids)?;
        Ok(())
    }

    /// What the lane holds beside the sequence's current GDN cell: the ring
    /// buffers, the second cell and the pending GDN tape.
    pub(crate) fn held_bytes(&self) -> u64 {
        self.ring
            .buffer_bytes()
            .saturating_add(self.gdn.as_ref().map_or(0, GdnStateBlobs::spare_bytes))
            .saturating_add(self.gdn.as_ref().map_or(0, GdnStateBlobs::pending_bytes))
    }
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
    /// The napi object's handle on the state tier.
    pub(crate) fn tier_handle(&self) -> crate::disk_tier::TierHandle {
        crate::disk_tier::TierHandle::of(self.disk.as_ref().map(|disk| &disk.tier))
    }

    pub(crate) fn new(block_size: u32, recency: Arc<CacheRecency>) -> Self {
        let mut store = HybridStateStore::new(block_size);
        store.share_recency(recency);
        Self {
            store,
            turns: HashMap::new(),
            turn_head_boundaries: HashMap::new(),
            prompt_heads: HashSet::new(),
            replay_points: HashMap::new(),
            generation_suffix: None,
            lane_window: None,
            seq_windows: HashMap::new(),
            seq_lanes: HashMap::new(),
            lane_batch: None,
            peer_cells: HashMap::new(),
            lane_decode_limit: None,
            counters: StateCounters::default(),
            released: false,
            restore_uses: HashMap::new(),
            idle_lane: IdleLane::default(),
            prefill_suspends: false,
            prefill_yield: None,
            disk: None,
            disk_read: None,
            state_read: None,
        }
    }

    /// Drops every parked state.
    pub(crate) fn clear(&mut self) {
        self.store.clear();
        self.turns.clear();
        self.turn_head_boundaries.clear();
        self.prompt_heads.clear();
        self.replay_points.clear();
        self.lane_window = None;
        self.seq_windows.clear();
        self.seq_lanes.clear();
        self.lane_batch = None;
        self.peer_cells.clear();
        self.lane_decode_limit = None;
        self.disk_read = None;
        self.state_read = None;
        self.restore_uses.clear();
        self.idle_lane = IdleLane::default();
    }

    pub(crate) fn turn_head_boundary(&self, seq: u32) -> u32 {
        self.turn_head_boundaries.get(&seq).copied().unwrap_or(0)
    }

    pub(crate) fn set_turn_head_boundary(&mut self, seq: u32, head_boundary: u32) {
        if head_boundary == 0 {
            self.turn_head_boundaries.remove(&seq);
        } else {
            self.turn_head_boundaries.insert(seq, head_boundary);
        }
    }

    pub(crate) fn is_prompt_head(&self, id: EntryId) -> bool {
        self.prompt_heads.contains(&id)
    }

    pub(crate) fn mark_prompt_head(&mut self, id: EntryId) {
        if self.prompt_heads.len() >= 64 {
            let store = &self.store;
            self.prompt_heads.retain(|eid| store.get(*eid).is_some());
        }
        self.prompt_heads.insert(id);
    }

    /// Marks `id`'s block as `seq`'s turn's replay point (Splash
    /// `StateCache::useState`).
    pub(crate) fn use_replay_point(&mut self, seq: u32, id: EntryId) {
        if let Some(block) = self.store.get(id).map(Entry::mark) {
            self.use_replay_block(seq, block);
        }
    }

    /// Marks `block` as `seq`'s turn's replay point, with a state there or
    /// yet to be published (Splash `StateCache::useState`).
    pub(crate) fn use_replay_block(&mut self, seq: u32, block: u64) {
        self.end_replay_point(seq);
        self.store.use_state(block);
        self.replay_points.insert(seq, block);
    }

    /// Ends `seq`'s turn's use of its replay point (Splash
    /// `StateCache::unuse`: it becomes the newest state).
    pub(crate) fn end_replay_point(&mut self, seq: u32) {
        if let Some(block) = self.replay_points.remove(&seq) {
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

    /// `seq`'s turn ended (finalize, abort, cancel), after its request was
    /// released: its replay point is no longer in use, stamped newer than
    /// its KV tail, and its points go (Splash `Engine::release`). What it
    /// published stays: each state was published only once its block
    /// registered, and its checkpoint is not retired.
    pub(crate) fn end_turn(&mut self, seq: u32) {
        self.end_replay_point(seq);
        self.turns.remove(&seq);
        self.state_read = None;
        let window = self.seq_windows.remove(&seq);
        self.idle_lane.give_window(window);
    }

    /// `seq`'s decode lane; a new one takes the idle lane's ring (Splash
    /// `QwenStateStorage::acquire`).
    pub(crate) fn begin_lane(&mut self, seq: u32, config: &DFlash2Config) -> &mut DraftLane {
        let idle = &mut self.idle_lane;
        self.seq_lanes.entry(seq).or_insert_with(|| DraftLane {
            ring: idle.take_ring(config),
            next_position: 0,
            gdn: None,
            drafted: None,
            open: None,
            verifies: LaneVerifies::default(),
            planes_calls: 0,
            wave_planes_calls: 0,
        })
    }

    #[cfg_attr(not(test), expect(dead_code))]
    pub(crate) fn lane_mut(&mut self, seq: u32) -> Option<&mut DraftLane> {
        self.seq_lanes.get_mut(&seq)
    }

    /// `seq`'s lane ends (Splash `QwenStateStorage::releaseLane`): its ring
    /// goes to the idle lane, its cells and draft drop; the last lane drops
    /// the decode cache limit. Its pending rows drop unfolded:
    /// `scheduled_dflash2::end_lane` retracts an open wave and settles the
    /// sequence's cells first.
    pub(crate) fn release_lane(&mut self, seq: u32) {
        let Some(lane) = self.seq_lanes.remove(&seq) else {
            return;
        };
        self.idle_lane.give_ring(Some(lane.ring));
        if self.seq_lanes.is_empty() {
            self.lane_decode_limit = None;
        }
    }

    /// `seq`'s context window: the serial lane's for sequence 0.
    pub(crate) fn lane_window_of(&self, seq: u32) -> Option<&DraftContextWindow> {
        if seq == 0 {
            self.lane_window.as_ref()
        } else {
            self.seq_windows.get(&seq)
        }
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

/// Runs [`install_linear`] from `cell` into `fresh` and drops the result;
/// false when `cell` holds no materialized GDN cell.
pub(crate) fn warm_install_linear(
    cell: Option<&[Qwen3_5LayerCache]>,
    mut fresh: Vec<Qwen3_5LayerCache>,
) -> Result<bool> {
    let Some(anchor) = cell
        .and_then(crate::models::qwen3_5::paged_forward::snapshot_materialized_linear_layer_caches)
    else {
        return Ok(false);
    };
    install_linear(&mut fresh, &anchor)?;
    Ok(true)
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
            window: Some(config.context_window().unwrap()),
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
        assert_eq!(anchor_bytes(&target, Some(&draft)), 159_842_304);
        assert_eq!(anchor_bytes(&target, None), 153_944_064);
    }
}
