use super::{config::Config, math, weights::Store};
use crate::array::{DType, MxArray, scaled_dot_product_attention};
use crate::nn::Activations;
use napi::{Error, Result};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

mod attention_window;
mod batch;
mod common;
mod cuda_deferred;
mod cuda_verify;
mod device_routes;
mod draft_cycle;
mod draft_view;
mod draft_window;
mod hidden_taps;
mod hyper;
mod lanes;
mod ple;
mod prefill;
mod scheduled_batch;

use common::{AttentionProjections, GdnProjections};
pub(super) use cuda_deferred::{
    CudaPendingGdn, DeferredVerifyTicket, TreeVerifyTicket, VerifyAttnLayerTape,
    VerifyGdnLayerTape, VerifyPleLayerTape,
};
pub(super) use device_routes::RouteGuard;
#[cfg(test)]
pub(super) use draft_cycle::{
    ChosenRows, DraftRecord, DraftStepper, draft_turn, follow_history, next_verify_rows,
};
pub(super) use draft_cycle::{DRAFT_VERIFY_ROWS, DraftSlot, load_draft};
pub(super) use draft_view::{draft_target_shape_of, tapped_rows, tapped_tree_rows};
pub(super) use hidden_taps::TappedRows;
#[cfg(test)]
pub(super) use hidden_taps::stream_means;
pub(super) use lanes::{LANE_ROWS, Lane};
pub(super) use prefill::GroupLimit;
#[cfg(test)]
pub(super) use prefill::PARTIAL_GROUP_WINDOWS;

/// Slot banks for a window's expert groups: a lease per group, then the
/// release of that group's output before the slots serve the next group.
pub(super) trait GroupBanks {
    fn lease(
        &mut self,
        experts: &[u32],
    ) -> Result<Option<([Arc<super::weights::Weight>; 3], Vec<u32>, usize)>>;
    fn release(&mut self, output: &MxArray) -> Result<()>;
}

#[derive(Clone, Default)]
pub struct LayerCache {
    conv: Option<MxArray>,
    recurrent: Option<MxArray>,
    keys: Option<MxArray>,
    values: Option<MxArray>,
    /// Draft history: the rows of `keys`/`values` this cache owns. Clones
    /// share one buffer and write in place past their own rows, so no cache
    /// may read past its own rows.
    draft_rows: Option<usize>,
    /// Active token rows in preallocated non-paged flat `keys`/`values`
    /// (`[1, kv_heads, cap, head_dim]`). `None` when paged, unallocated, or
    /// managed by `draft_rows`.
    flat_kv_rows: Option<usize>,
    index_tail: Vec<MxArray>,
    /// `[cap_blocks, indexer_head_dim]` (or `[blocks, indexer_head_dim]`) of a
    /// cache without pages. Active block count is `flat_index_blocks` when
    /// `Some`, else `shape_at(0)`.
    index_keys: Option<MxArray>,
    /// Active block rows in preallocated non-paged `index_keys`.
    flat_index_blocks: Option<usize>,
    /// Complete indexer blocks of a paged sequence: their pooled keys live
    /// in the pool's index plane (Splash `PageStorage::indexOffset`).
    index_blocks: usize,
    ple_conv: Option<MxArray>,
}
impl LayerCache {
    /// The rows of a draft history.
    pub(super) fn draft_row_count(&self) -> usize {
        self.draft_rows.unwrap_or(0)
    }
    /// Whether an indexer key waits for its block.
    pub(super) fn index_pending(&self) -> bool {
        !self.index_tail.is_empty()
    }
    /// Draft rows `[from, to)` and their pooled indexer keys in storage of
    /// their own, evaluated; `None` where the history does not hold them
    /// whole (`ratio` the indexer's compress ratio).
    pub(super) fn draft_segment(
        &self,
        from: usize,
        to: usize,
        ratio: usize,
    ) -> Result<Option<super::prefix_state::Segment>> {
        let ratio = ratio.max(1);
        let (Some(keys), Some(values), Some(index)) = (&self.keys, &self.values, &self.index_keys)
        else {
            return Ok(None);
        };
        if to > self.draft_row_count()
            || !from.is_multiple_of(ratio)
            || !to.is_multiple_of(ratio)
            || to / ratio > self.index_block_count()?
        {
            return Ok(None);
        }
        let segment = super::prefix_state::Segment {
            keys: keys.slice_axis(2, from as i64, to as i64)?.deep_copy()?,
            values: values.slice_axis(2, from as i64, to as i64)?.deep_copy()?,
            index: index
                .slice_axis(0, (from / ratio) as i64, (to / ratio) as i64)?
                .deep_copy()?,
        };
        MxArray::eval_arrays(&[&segment.keys, &segment.values, &segment.index])?;
        Ok(Some(segment))
    }
    /// A draft history of `segments` in order, in a buffer of its own.
    pub(super) fn from_draft_segments(segments: &[super::prefix_state::Segment]) -> Result<Self> {
        if segments.is_empty() {
            return Ok(Self::default());
        }
        let keys = MxArray::concatenate_many(segments.iter().map(|s| &s.keys).collect(), Some(2))?;
        let values =
            MxArray::concatenate_many(segments.iter().map(|s| &s.values).collect(), Some(2))?;
        let index =
            MxArray::concatenate_many(segments.iter().map(|s| &s.index).collect(), Some(0))?;
        MxArray::eval_arrays(&[&keys, &values, &index])?;
        let rows = keys.shape_at(2)? as usize;
        let blocks = index.shape_at(0)? as usize;
        Ok(Self {
            keys: Some(keys),
            values: Some(values),
            draft_rows: Some(rows),
            index_keys: Some(index),
            flat_index_blocks: Some(blocks),
            ..Self::default()
        })
    }
    /// The pooled indexer keys of a draft history.
    #[cfg(test)]
    pub(super) fn draft_index(&self) -> Option<MxArray> {
        self.active_index_keys().ok().flatten()
    }
    /// (rows, K buffer, V buffer) of a draft history.
    #[cfg(test)]
    pub(super) fn draft_kv(&self) -> Option<(usize, &MxArray, &MxArray)> {
        Some((self.draft_rows?, self.keys.as_ref()?, self.values.as_ref()?))
    }
    /// (K/V rows, indexer blocks, indexer tail rows).
    #[cfg(test)]
    pub(super) fn history_shape(&self) -> (usize, usize, usize) {
        let rows = self.draft_rows.or(self.flat_kv_rows).unwrap_or_else(|| {
            self.keys
                .as_ref()
                .map_or(0, |keys| keys.shape_at(2).unwrap() as usize)
        });
        (
            rows,
            self.index_block_count().unwrap(),
            self.index_tail.len(),
        )
    }
    #[cfg(test)]
    pub(super) fn gdn_state(&self) -> (Option<&MxArray>, Option<&MxArray>) {
        (self.conv.as_ref(), self.recurrent.as_ref())
    }
    pub(super) fn arrays(&self) -> impl Iterator<Item = &MxArray> {
        self.conv
            .iter()
            .chain(self.recurrent.iter())
            .chain(self.ple_conv.iter())
            .chain(self.keys.iter())
            .chain(self.values.iter())
            .chain(self.index_keys.iter())
            .chain(self.index_tail.iter())
    }
}
/// A decode route mode a test pins.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RouteMode {
    /// Every decode token reads its routes back on the host, as after a
    /// route miss.
    Host,
}
/// How a decode pass over the layers ended. A stop is not an error: an
/// error resets the whole sequence.
pub(super) enum LayerPass {
    /// Every layer and the head ran: the token's output.
    Done(MxArray),
    /// The pass stopped before layer `at`: the layers before it advanced
    /// their caches, layer `at` and later are untouched, `history` too.
    Stopped { at: usize },
}
/// Which route groups a device decode pass reads at a group submit, in
/// place of the live rule (the groups submitted before this one).
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GroupPoll {
    /// None: route entries are not submitted with the groups and a miss
    /// waits for the pass end.
    Never,
    /// The group just submitted too, still in flight.
    InFlight,
    /// The group just submitted too, after the GPU completed it.
    Ready,
    /// The live rule after the GPU completed the earlier groups: the group
    /// is submitted after that wait, so it can still be in flight.
    Older,
}

#[derive(Clone, Default)]
pub struct DecoderState {
    caches: Vec<LayerCache>,
    history: Vec<u32>,
    positions: Vec<[i64; 3]>,
    rope_delta: i64,
    media_digests: Vec<crate::engine::cache::ImageCacheDigest>,
    last_chunk_hidden: Vec<MxArray>,
    cancelled: Option<Arc<AtomicBool>>,
    route_guard: RouteGuard,
}
impl DecoderState {
    pub(super) fn last_chunk_hidden(&self) -> &[MxArray] {
        &self.last_chunk_hidden
    }
    pub(super) fn arrays(&self) -> impl Iterator<Item = &MxArray> {
        self.caches
            .iter()
            .flat_map(LayerCache::arrays)
            .chain(self.last_chunk_hidden.iter())
    }
}
pub struct Decoder {
    pub prefill_chunk_size: usize,
    /// The tokens one sequence may hold: the memory plan's window once the
    /// load planned it, the trained window before.
    pub context_window: usize,
    /// The prefill scratch a host gate that refused the whole bank set
    /// reserved beside the partial slots; `None` on every other store.
    pub(super) host_prefill_scratch: Option<u64>,
    /// The lane transient the memory plan priced at its allowance context;
    /// a partial store's layer-major group runs inside it.
    pub(super) planned_scratch: Option<u64>,
    /// A test that sizes admission through the prefill window keeps its
    /// window across the turn opens that re-derive it from the plan.
    #[cfg(test)]
    pub(super) prefill_window_pin: Option<usize>,
    // Like TrackFastModel's retained configuration, share immutable metadata
    // across forward helpers instead of copying its strings/lists per layer.
    pub config: Arc<Config>,
    pub weights: Store,
    pub caches: Vec<LayerCache>,
    pub history: Vec<u32>,
    pub positions: Vec<[i64; 3]>,
    pub rope_delta: i64,
    pub media_digests: Vec<crate::engine::cache::ImageCacheDigest>,
    pub cancelled: Option<Arc<AtomicBool>>,
    /// Splash `Engine::stop` ends a lane at its pass boundary: inside a
    /// decode pass the cancel flag is left to the step-boundary snapshot,
    /// so a stop never fails the pass (or a batched pass's peer rows).
    pub(super) decode_pass: bool,
    pub paged: Option<crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter>,
    paged_chunk_start: usize,
    scope: String,
    position_override: Option<usize>,
    rotary_tables: std::cell::RefCell<math::RotaryWindowCache>,
    /// Cached `[head_dim]` BF16 Q/K norm weights for fused CUDA `qk_norm_rope_kv_store`
    /// (avoids per-step F32->BF16 casts without altering residency accounting).
    cuda_qk_norm_bf16: std::collections::HashMap<String, MxArray>,
    /// Cached FP32 `(scale, dt)` per GDN layer so `gdn_mix`/`gdn_batch` never
    /// repeat `dense().astype(F32)` on every token/verify cycle.
    cuda_gdn_scale_dt: std::collections::HashMap<usize, (MxArray, MxArray)>,
    /// Per-layer pending CUDA deferred GDN recurrent updates (flat speculative verify).
    cuda_pending_gdn: Vec<Option<CudaPendingGdn>>,
    /// Armed during `verify_chunk_deferred`; populated by `verify_layer`.
    cuda_verify_gdn_tape: Option<Vec<Option<VerifyGdnLayerTape>>>,
    /// Armed during `verify_chunk_deferred`; populated by `prefill_group_inner`.
    cuda_verify_row_caches: Option<Vec<Vec<LayerCache>>>,
    /// Carried final window + fused `output_hc_norm` output from a verify pass
    /// so `verify_chunk*` avoids slicing+re-concatenating `last_chunk_hidden`.
    cuda_verify_final_window: Option<MxArray>,
    cuda_verify_final_normed: Option<MxArray>,
    /// When true (set by `Step::forward_with_token` on flat CUDA AR), `decode_layers`
    /// flushes `logits + state` asynchronously via `cuda_async_eval` and returns lazy
    /// `logits` so `argmax` fuses into a single GPU sync per token.
    pub(super) cuda_ar_lazy_logits: bool,
    /// The target's YaRN; `None` within the native window.
    pub(super) yarn: Option<math::Yarn>,
    verification: Option<Vec<DecoderState>>,
    device_routes: Option<Vec<(usize, MxArray)>>,
    /// When set, every index-plane store pushes the plane it wrote.
    index_planes: Option<Vec<MxArray>>,
    /// The residual stream entering each layer of the decode token in
    /// flight, when kept: entry `i` is layer `i`'s input. A pass from layer
    /// `m` keeps entries `0..m` and writes its own from `m` on.
    pub(super) layer_inputs: Option<Vec<MxArray>>,
    /// Tests keep `layer_inputs` (or not) in place of the live rule.
    #[cfg(test)]
    pub(super) layer_inputs_pin: Option<bool>,
    /// Layer inputs decode passes kept.
    #[cfg(test)]
    pub(super) kept_layer_inputs: u64,
    /// Tests stop a decode pass before this layer.
    #[cfg(test)]
    pub(super) stop_before: Option<usize>,
    /// Tests zero the MoE output of this layer once, in the next pass that
    /// runs it: a stand-in for a skipped expert.
    #[cfg(test)]
    pub(super) poison_moe_at: Option<usize>,
    /// The running request's route guard.
    pub(super) route_guard: RouteGuard,
    /// Layers whose routes a pass read back to the host one at a time.
    #[cfg(test)]
    pub(super) route_readbacks: u64,
    /// Layers a tentative pass ran through the fused routed+shared kernel.
    #[cfg(test)]
    pub(super) fused_shared_layers: u64,
    /// The (layer, routes) tape of every checked device decode pass.
    #[cfg(test)]
    pub(super) route_tapes: Vec<Vec<(usize, Vec<u32>)>>,
    /// Tests read, in place of each hint, experts its layer did not route.
    #[cfg(test)]
    pub(super) wrong_hints: bool,
    /// (layer, every id it preloaded holds a slot after the token's fills),
    /// per layer a replay preloaded.
    #[cfg(test)]
    pub(super) preload_checks: Vec<(usize, bool)>,
    /// The most device layers one decode token ran again.
    #[cfg(test)]
    pub(super) device_rerun_peak: usize,
    /// Tests pick the route groups a group submit reads.
    #[cfg(test)]
    pub(super) after_group_submit: Option<GroupPoll>,
    /// Device passes a route read stopped before their last layer.
    #[cfg(test)]
    pub(super) early_stops: u64,
    /// Route reads that found an entry still in flight.
    #[cfg(test)]
    pub(super) unready_polls: std::cell::Cell<u64>,
    /// Tests pin the decode route mode in place of the live rule.
    #[cfg(test)]
    pub(super) route_mode_pin: Option<RouteMode>,
    /// Lane steps that ran.
    #[cfg(test)]
    pub(super) lane_steps: u64,
    /// When set, every one-token pass pushes (layer, residual stream after
    /// the layer), the tentative pass and its replay alike.
    #[cfg(test)]
    pub(super) layer_outputs: Option<Vec<(usize, MxArray)>>,
    /// When set, every prefill window, layer-major group and verify pass
    /// pushes (layer, each window's residual stream after the layer).
    #[cfg(test)]
    pub(super) pass_outputs: Option<Vec<(usize, Vec<MxArray>)>>,
    /// The DFlash2 draft's taps; `None` without a draft.
    hidden_taps: Option<hidden_taps::HiddenTaps>,
    #[cfg(test)]
    pub(super) submit_log: std::cell::RefCell<Vec<String>>,
    pub last_chunk_hidden: Vec<MxArray>,
    /// The final residual streams of the last layer-major group, one per
    /// window, until its views are taken.
    group_streams: Vec<MxArray>,
    /// Tests pin group mode on or off in place of the live rule.
    #[cfg(test)]
    pub(super) group_pin: Option<bool>,
    /// Per layer-major union lease: the layer and the group's routes, window
    /// after window.
    #[cfg(test)]
    pub(super) union_routes: Vec<(usize, Vec<u32>)>,
    /// Per layer-major union lease: the bytes of the arrays the group holds
    /// at its widest (every window's streams, MLP inputs, routes, scores,
    /// routed rows and sums).
    #[cfg(test)]
    pub(super) group_held: Vec<u64>,
    /// When set, per fill read ahead of its lease: the layer, whether a
    /// victim is a slot the sub-group running beside the reads leased, and
    /// whether those slots' bank rows were unchanged once its rows were
    /// submitted.
    #[cfg(test)]
    pub(super) fill_bank_checks: Option<Vec<(usize, bool, bool)>>,
    /// Tentative commits taken, each with the windows it covered.
    #[cfg(test)]
    pub(super) tentative_commits: Vec<usize>,
    /// Every (rows, elapsed) sample handed to the live, in order.
    #[cfg(test)]
    pub(super) pass_samples: Vec<(usize, std::time::Duration)>,
    /// Runs after every layer sweep of a group, with the layer; tests trip
    /// the wire or cancel between two sweeps through it.
    #[cfg(test)]
    pub(super) after_sweep: Option<Box<dyn FnMut(&mut Decoder, usize) + Send>>,
    /// The weight budget this store registers with the process-wide cache
    /// limit; re-registered whenever the residency changes, between turns or
    /// between two passes of one request.
    cache_limit_guard: Option<crate::cache_limit::CacheLimitGuard>,
    /// Runs after every observed pass; tests cut the admitted window between
    /// two passes of one prompt through it.
    #[cfg(test)]
    pub(super) after_pass: Option<Box<dyn FnMut(&mut Decoder) + Send>>,
}
impl Decoder {
    /// Register the budget in force with the process-wide cache limit.
    pub fn register_cache_limit(&mut self) {
        self.cache_limit_guard = None;
        self.cache_limit_guard =
            Some(crate::cache_limit::coordinator().register(self.weights.cache_budget()));
    }

    /// The prefill window is the chunk the memory plan prices.
    pub fn refresh_prefill_window(&mut self) {
        let window = super::MAX_PREFILL_CHUNK;
        #[cfg(test)]
        let window = self.prefill_window_pin.unwrap_or(window);
        self.prefill_chunk_size = window;
        self.weights
            .live
            .prefill_chunk_tokens
            .store(self.prefill_chunk_size as u64, Ordering::Release);
    }

    /// The load's memory plan is in force: a sequence holds up to `window`
    /// tokens. A store the host gate refused keeps the prefill scratch that
    /// gate priced at `allowance_tokens`.
    pub(super) fn adopt_plan(&mut self, window: usize, allowance_tokens: u32) {
        self.context_window = window;
        self.host_prefill_scratch = matches!(
            self.weights.live.fallback(),
            Some(super::live::Fallback::LoadGate { .. })
        )
        .then(|| super::load_plan::transient(&self.config, allowance_tokens));
        self.planned_scratch = Some(super::load_plan::transient(&self.config, allowance_tokens));
        self.residency_changed();
    }

    /// The store changed its residency: the budget it registers and the
    /// prefill window it admits follow.
    pub fn residency_changed(&mut self) {
        self.register_cache_limit();
        self.refresh_prefill_window();
    }

    /// [`Store::promote_whole_with`], then the prefill passes of a whole
    /// store: no reservation of the refused gate halves them.
    pub(super) fn promote_whole_with(
        &mut self,
        own: u64,
        gate: &mut impl FnMut(&mut super::memory::Plan, u64, u64, u64) -> Result<()>,
        admit: &mut impl FnMut(u64) -> Result<()>,
    ) -> Result<bool> {
        if !self.weights.promote_whole_with(own, gate, admit)? {
            return Ok(false);
        }
        self.host_prefill_scratch = None;
        self.device_routes = None;
        self.residency_changed();
        Ok(true)
    }

    /// Record a completed pass and run the test admission hook.
    pub(super) fn observe_pass(&mut self, rows: usize, elapsed: std::time::Duration) {
        #[cfg(test)]
        self.pass_samples.push((rows, elapsed));
        #[cfg(not(test))]
        let _ = (rows, elapsed);
        #[cfg(test)]
        if let Some(mut hook) = self.after_pass.take() {
            hook(self);
            self.after_pass = Some(hook);
        }
    }

    pub fn new(config: Config, weights: Store) -> Result<Self> {
        // Validate the checkpoint topology before the first tensor payload read.
        let context_window = config.served_context_tokens() as usize;
        let yarn = math::Yarn::of(&config)?;
        let d = Self {
            prefill_chunk_size: super::MAX_PREFILL_CHUNK,
            context_window,
            host_prefill_scratch: None,
            planned_scratch: None,
            #[cfg(test)]
            prefill_window_pin: None,
            caches: (0..config.num_hidden_layers)
                .map(|_| LayerCache::default())
                .collect(),
            cuda_pending_gdn: vec![None; config.num_hidden_layers],
            cuda_verify_gdn_tape: None,
            cuda_verify_row_caches: None,
            cuda_gdn_scale_dt: Default::default(),
            cuda_verify_final_window: None,
            cuda_verify_final_normed: None,
            cuda_ar_lazy_logits: false,
            config: Arc::new(config),
            weights,
            history: Vec::new(),
            positions: Vec::new(),
            rope_delta: 0,
            media_digests: Vec::new(),
            cancelled: None,
            decode_pass: false,
            paged: None,
            paged_chunk_start: 0,
            scope: String::new(),
            position_override: None,
            rotary_tables: Default::default(),
            cuda_qk_norm_bf16: Default::default(),
            yarn,
            verification: None,
            device_routes: None,
            index_planes: None,
            layer_inputs: None,
            #[cfg(test)]
            layer_inputs_pin: None,
            #[cfg(test)]
            kept_layer_inputs: 0,
            #[cfg(test)]
            stop_before: None,
            #[cfg(test)]
            poison_moe_at: None,
            route_guard: RouteGuard::default(),
            #[cfg(test)]
            route_readbacks: 0,
            #[cfg(test)]
            fused_shared_layers: 0,
            #[cfg(test)]
            route_tapes: Vec::new(),
            #[cfg(test)]
            wrong_hints: false,
            #[cfg(test)]
            preload_checks: Vec::new(),
            #[cfg(test)]
            device_rerun_peak: 0,
            #[cfg(test)]
            after_group_submit: None,
            #[cfg(test)]
            early_stops: 0,
            #[cfg(test)]
            unready_polls: Default::default(),
            #[cfg(test)]
            route_mode_pin: None,
            #[cfg(test)]
            lane_steps: 0,
            #[cfg(test)]
            layer_outputs: None,
            #[cfg(test)]
            pass_outputs: None,
            hidden_taps: None,
            #[cfg(test)]
            submit_log: Default::default(),
            last_chunk_hidden: Vec::new(),
            group_streams: Vec::new(),
            #[cfg(test)]
            group_pin: None,
            #[cfg(test)]
            union_routes: Vec::new(),
            #[cfg(test)]
            group_held: Vec::new(),
            #[cfg(test)]
            fill_bank_checks: None,
            #[cfg(test)]
            tentative_commits: Vec::new(),
            #[cfg(test)]
            pass_samples: Vec::new(),
            #[cfg(test)]
            after_sweep: None,
            cache_limit_guard: None,
            #[cfg(test)]
            after_pass: None,
        };
        d.preflight()?;
        Ok(d)
    }
    pub fn snapshot(&self) -> DecoderState {
        DecoderState {
            caches: self.caches.clone(),
            history: self.history.clone(),
            positions: self.positions.clone(),
            rope_delta: self.rope_delta,
            media_digests: self.media_digests.clone(),
            last_chunk_hidden: self.last_chunk_hidden.clone(),
            cancelled: self.cancelled.clone(),
            route_guard: self.route_guard.clone(),
        }
    }
    #[cfg(test)]
    pub(super) fn index_keys(&self, layer: usize) -> Option<MxArray> {
        self.index_blocks(layer, &self.caches[layer]).unwrap()
    }
    /// The index keys layer `layer` holds beside its cache.
    #[cfg(test)]
    pub(super) fn off_pool_index_keys(&self, layer: usize) -> Option<&MxArray> {
        self.caches[layer].index_keys.as_ref()
    }
    /// The pool layer of full-attention layer `i`.
    fn pool_layer(&self, i: usize) -> u32 {
        (0..i).filter(|&l| !self.config.linear(l)).count() as u32
    }
    /// Append complete blocks' pooled keys (`[n, indexer_head_dim]`) of
    /// layer `i`: into the pool's index plane for a paged sequence (Splash
    /// `block_indexer_pool`), else beside the cache. Returns what holds them.
    pub(super) fn append_index_blocks(
        &mut self,
        i: usize,
        cache: &mut LayerCache,
        keys: MxArray,
    ) -> Result<MxArray> {
        let layer = self.pool_layer(i);
        if let Some(adapter) = &mut self.paged {
            let blocks = keys.shape_at(0)? as usize;
            let plane = adapter
                .store_index_rows(layer, &keys, cache.index_blocks as u32)
                .map_err(Error::from_reason)?;
            cache.index_blocks += blocks;
            if let Some(planes) = &mut self.index_planes {
                planes.push(plane.clone());
            }
            return Ok(plane);
        }
        cache.append_index_keys(keys).cloned()
    }
    /// The pooled keys of every complete block of layer `i`, `None` before
    /// the first.
    pub(super) fn index_blocks(&self, i: usize, cache: &LayerCache) -> Result<Option<MxArray>> {
        if let Some(adapter) = &self.paged {
            if cache.index_blocks == 0 {
                return Ok(None);
            }
            return adapter
                .index_rows(self.pool_layer(i), cache.index_blocks as u32)
                .map(Some)
                .map_err(Error::from_reason);
        }
        cache.active_index_keys()
    }
    pub fn take_state(&mut self) -> DecoderState {
        let _ = self.settle_cuda_pending_gdn();
        let state = DecoderState {
            caches: std::mem::take(&mut self.caches),
            history: std::mem::take(&mut self.history),
            positions: std::mem::take(&mut self.positions),
            rope_delta: self.rope_delta,
            media_digests: self.media_digests.clone(),
            last_chunk_hidden: std::mem::take(&mut self.last_chunk_hidden),
            cancelled: self.cancelled.take(),
            route_guard: std::mem::take(&mut self.route_guard),
        };
        self.reset();
        state
    }
    pub fn restore_state(&mut self, state: DecoderState) {
        self.rotary_tables.get_mut().clear();
        self.clear_cuda_deferred_gdn();
        self.caches = state.caches;
        self.history = state.history;
        self.positions = state.positions;
        self.rope_delta = state.rope_delta;
        self.media_digests = state.media_digests;
        self.last_chunk_hidden = state.last_chunk_hidden;
        self.cancelled = state.cancelled;
        self.route_guard = state.route_guard;
    }
    /// The cell outside the pool at this row, in storage of its own; `None`
    /// where an indexer key still waits for a block or a cache holds rows
    /// the pool does not (no pages, image positions, draft history).
    pub(super) fn cell(&self, grid: usize) -> Result<Option<super::prefix_state::Cell>> {
        if self.paged.is_none() || !self.positions.is_empty() {
            return Ok(None);
        }
        let copy = |array: &Option<MxArray>| array.as_ref().map(MxArray::deep_copy).transpose();
        let mut layers = Vec::with_capacity(self.caches.len());
        for cache in &self.caches {
            if !cache.index_tail.is_empty()
                || cache.index_keys.is_some()
                || cache.draft_rows.is_some()
                || cache.keys.is_some()
                || cache.values.is_some()
            {
                return Ok(None);
            }
            layers.push(super::prefix_state::LayerCell {
                conv: copy(&cache.conv)?,
                recurrent: copy(&cache.recurrent)?,
                ple_conv: copy(&cache.ple_conv)?,
                index_blocks: cache.index_blocks,
            });
        }
        let cell = super::prefix_state::Cell {
            grid,
            layers,
            draft: None,
        };
        let arrays: Vec<&MxArray> = cell.arrays().collect();
        if !arrays.is_empty() {
            MxArray::eval_arrays(&arrays)?;
        }
        Ok(Some(cell))
    }

    /// Resume at the row `cell` was taken at: copies of its arrays become the
    /// caches and `history` the tokens before that row. The cancel flag of
    /// the running turn stays.
    pub(super) fn restore_cell(
        &mut self,
        cell: &super::prefix_state::Cell,
        history: &[u32],
    ) -> Result<()> {
        if cell.layers.len() != self.config.num_hidden_layers {
            return Err(Error::from_reason(format!(
                "Qwen4 stored state holds {} layers for {}",
                cell.layers.len(),
                self.config.num_hidden_layers
            )));
        }
        self.reset();
        let copy = |array: &Option<MxArray>| array.as_ref().map(MxArray::deep_copy).transpose();
        for (cache, layer) in self.caches.iter_mut().zip(&cell.layers) {
            cache.conv = copy(&layer.conv)?;
            cache.recurrent = copy(&layer.recurrent)?;
            cache.ple_conv = copy(&layer.ple_conv)?;
            cache.index_blocks = layer.index_blocks;
        }
        self.history = history.to_vec();
        let arrays: Vec<&MxArray> = self.caches.iter().flat_map(LayerCache::arrays).collect();
        if !arrays.is_empty() {
            MxArray::eval_arrays(&arrays)?;
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        self.rotary_tables.get_mut().clear();
        self.clear_cuda_deferred_gdn();
        self.verification = None;
        self.device_routes = None;
        self.layer_inputs = None;
        self.route_guard = RouteGuard::default();
        self.history.clear();
        self.positions.clear();
        self.rope_delta = 0;
        self.media_digests.clear();
        self.last_chunk_hidden.clear();
        self.group_streams.clear();
        if let Some(taps) = &mut self.hidden_taps {
            taps.clear();
        }
        self.caches = (0..self.config.num_hidden_layers)
            .map(|_| LayerCache::default())
            .collect();
    }

    /// Taps the outputs of `layers` on every pass from here on; `None`
    /// takes no tap.
    pub(super) fn set_hidden_taps(&mut self, layers: Option<&[usize]>) -> Result<()> {
        self.hidden_taps = layers
            .map(|layers| hidden_taps::HiddenTaps::new(layers, self.config.num_hidden_layers))
            .transpose()?;
        Ok(())
    }

    /// The taps of the passes finished since the last take, in row order.
    pub(super) fn take_hidden_taps(&mut self) -> Vec<TappedRows> {
        self.hidden_taps
            .as_mut()
            .map_or_else(Vec::new, hidden_taps::HiddenTaps::take)
    }

    /// The layers the taps set read; `None` with no taps.
    pub(super) fn hidden_tap_layers(&self) -> Option<&[usize]> {
        self.hidden_taps
            .as_ref()
            .map(hidden_taps::HiddenTaps::layers)
    }

    /// Whether the taps are set and every unread record is evaluated.
    #[cfg(test)]
    pub(super) fn hidden_taps_available(&self) -> bool {
        self.hidden_taps.as_ref().is_some_and(|taps| {
            taps.passes()
                .iter()
                .flat_map(|pass| &pass.hidden)
                .all(|tap| unsafe { mlx_sys::mlx_array_is_available(tap.as_raw_ptr()) })
        })
    }

    /// Bytes of the taps the decoder holds: the pass in flight and the
    /// finished passes not yet taken.
    pub(super) fn hidden_tap_bytes(&self) -> u64 {
        self.hidden_taps
            .as_ref()
            .map_or(0, hidden_taps::HiddenTaps::held_bytes)
    }

    /// The taps set keep only the records of their last `rows` rows.
    pub(super) fn bound_hidden_taps(&mut self, rows: Option<usize>) {
        if let Some(taps) = &mut self.hidden_taps {
            taps.keep_rows(rows);
        }
    }

    fn begin_taps(&mut self) {
        if let Some(taps) = &mut self.hidden_taps {
            taps.begin();
        }
    }

    /// Layer `i` of the pass in flight finished with `outputs`.
    fn tap_layer(&mut self, i: usize, outputs: &[MxArray]) -> Result<()> {
        let Some(taps) = &mut self.hidden_taps else {
            return Ok(());
        };
        taps.tap(i, outputs, self.config.hc_count, self.config.hidden_size)
    }

    /// The pass in flight ran its last layer; its first row is `first`.
    fn finish_taps(&mut self, first: usize) -> Result<()> {
        match &mut self.hidden_taps {
            Some(taps) => taps.finish(first),
            None => Ok(()),
        }
    }

    fn gdn(&mut self, x: &MxArray, i: usize, cache: &mut LayerCache) -> Result<MxArray> {
        let p = format!("layers.{i}.linear_attn");
        let g = format!("blk.{i}");
        let projections = self.gdn_projections(x, &p, &g)?;
        let out = self.gdn_mix(x.dtype()?, &projections, i, cache)?;
        self.linear(
            &out,
            &format!("{p}.out_proj.weight"),
            &format!("{g}.ssm_out.weight"),
        )
    }

    /// The stateful middle of one GDN row: the conv history and the recurrent
    /// state advance by exactly one token.
    fn gdn_mix(
        &mut self,
        dtype: DType,
        projections: &GdnProjections,
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<MxArray> {
        let c = self.config.clone();
        let p = format!("layers.{i}.linear_attn");
        let g = format!("blk.{i}");
        let nh = c.linear_num_value_heads as i64;
        let kh = c.linear_num_key_heads as i64;
        let kd = c.linear_key_head_dim as i64;
        let vd = c.linear_value_head_dim as i64;
        let keydim = kh * kd;
        let GdnProjections { qkv, z, a, b } = projections;
        let conv = self.dense(
            &format!("{p}.conv1d.weight"),
            &format!("{g}.ssm_conv1d.weight"),
        )?;
        if self.gguf()
            && (kh, nh, kd, vd) == (16, 48, 128, 128)
            && (2..=8).contains(&c.linear_conv_kernel_dim)
            && c.rms_norm_eps.is_finite()
            && c.rms_norm_eps > 0.0
            && c.output_gate_type == "sigmoid"
            && qkv.dtype()? == DType::BFloat16
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            let scale = self.dense(&format!("{p}.A_log"), &format!("{g}.ssm_a"))?;
            let dt = self.dense(&format!("{p}.dt_bias"), &format!("{g}.ssm_dt.bias"))?;
            let w = self.dense(&format!("{p}.norm.weight"), &format!("{g}.ssm_norm.weight"))?;
            let history = match &cache.conv {
                Some(s) => s.clone(),
                None => MxArray::zeros(
                    &[(c.linear_conv_kernel_dim - 1) as i64, 2 * keydim + nh * vd],
                    Some(qkv.dtype()?),
                )?,
            };
            let state = match &cache.recurrent {
                Some(s) => s.clone(),
                None => MxArray::zeros(&[1, nh, vd, kd], Some(DType::Float32))?,
            };
            let (out, state, history) = math::complete_gdn(
                qkv,
                z,
                a,
                b,
                &conv,
                &history,
                &scale,
                &dt,
                &state,
                &w,
                c.rms_norm_eps,
            )?;
            self.submit_state(&[&out, &state, &history], "qwen4::decoder::complete_gdn")?;
            cache.recurrent = Some(state);
            cache.conv = Some(history);
            return Ok(out);
        }
        if super::cuda_qwen4_fast_enabled()
            && (kh, nh, kd, vd) == (16, 48, 128, 128)
            && c.linear_conv_kernel_dim == 4
            && c.rms_norm_eps.is_finite()
            && c.rms_norm_eps > 0.0
            && qkv.dtype()? == DType::BFloat16
            && z.dtype()? == DType::BFloat16
            && conv.dtype()? == DType::Float32
        {
            let (scale, dt) = self.gdn_scale_dt(i, &p, &g)?;
            let w = self.dense(&format!("{p}.norm.weight"), &format!("{g}.ssm_norm.weight"))?;
            let old_conv = cache.conv.clone();
            if let Some([q, k, v, decay, beta]) =
                math::gdn_prepare(qkv, a, b, &conv, &mut cache.conv, &scale, &dt)?
            {
                let (state, can_alias) = match &cache.recurrent {
                    Some(s) => {
                        // In-place `state_dst` avoids 113.2 MB/tok allocation churn
                        // and SM121 25 MB graph splits whenever no snapshot aliases `s`.
                        let alias = self.verification.is_none()
                            && self.paged.is_none()
                            && std::sync::Arc::strong_count(&s.handle) == 1
                            && unsafe { mlx_sys::mlx_array_is_available(s.as_raw_ptr()) }
                            && s.shape()?.as_ref() == [1, nh, vd, kd]
                            && s.dtype()? == DType::Float32;
                        (s.clone(), alias)
                    }
                    None => (
                        MxArray::zeros(&[1, nh, vd, kd], Some(DType::Float32))?,
                        false,
                    ),
                };
                if let Some((out, next_state)) = crate::kernels::gdn::fused_step_with_gate(
                    &q,
                    &k,
                    &v,
                    &decay,
                    &beta,
                    &state,
                    z,
                    &w,
                    c.rms_norm_eps as f32,
                    self.gguf(),
                    c.output_gate_type == "sigmoid",
                    can_alias.then_some(&state),
                )? {
                    if let Some(history) = &cache.conv {
                        self.submit_state(
                            &[&out, &next_state, history],
                            "qwen4::decoder::complete_gdn",
                        )?;
                    } else {
                        self.submit_state(&[&out, &next_state], "qwen4::decoder::complete_gdn")?;
                    }
                    cache.recurrent = Some(next_state);
                    return Ok(out);
                }
                cache.conv = old_conv;
            }
        }
        let a_f32 = if a.dtype()? == DType::Float32 {
            a.clone()
        } else {
            a.astype(DType::Float32)?
        };
        let b_f32 = if b.dtype()? == DType::Float32 {
            b.clone()
        } else {
            b.astype(DType::Float32)?
        };
        let qkv = math::conv_with_completion(
            qkv,
            &conv,
            &mut cache.conv,
            c.linear_conv_kernel_dim,
            1,
            self.verification.is_some(),
        )?;
        let q = math::l2(&qkv.slice_axis(2, 0, keydim)?.reshape(&[1, kh, kd])?)?
            .mul_scalar((kd as f64).powf(-0.5))?;
        let k = math::l2(
            &qkv.slice_axis(2, keydim, 2 * keydim)?
                .reshape(&[1, kh, kd])?,
        )?;
        let v = qkv
            .slice_axis(2, 2 * keydim, 2 * keydim + nh * vd)?
            .reshape(&[1, nh, vd])?
            .astype(DType::Float32)?;
        let (q, k) = if self.gguf() {
            (q, k)
        } else {
            (
                q.repeat((nh / kh) as i32, 1)?,
                k.repeat((nh / kh) as i32, 1)?,
            )
        };
        let q = q.astype(DType::Float32)?.reshape(&[1, -1, 1, kd])?;
        let k = k.astype(DType::Float32)?.reshape(&[1, -1, 1, kd])?;
        let a_log = self
            .dense(&format!("{p}.A_log"), &format!("{g}.ssm_a"))?
            .astype(DType::Float32)?;
        let a_scale = if self.gguf() {
            a_log
        } else {
            a_log.exp()?.negative()?
        };
        let dt = self
            .dense(&format!("{p}.dt_bias"), &format!("{g}.ssm_dt.bias"))?
            .astype(DType::Float32)?;
        let (decay, beta) = math::gdn_gates(&a_f32, &b_f32, &a_scale, &dt)?;
        let decay = decay.reshape(&[1, nh, 1, 1])?;
        let state = match &cache.recurrent {
            Some(s) => s.clone(),
            None => MxArray::zeros(&[1, nh, vd, kd], Some(DType::Float32))?,
        };
        let (out, state) =
            math::recurrent_step(&q, &k, &v, &decay, &beta.reshape(&[1, nh, 1])?, &state)?;
        let out = out.astype(dtype)?;
        self.submit_state(&[&state], "qwen4::decoder::state")?;
        cache.recurrent = Some(state);
        let w = self.dense(&format!("{p}.norm.weight"), &format!("{g}.ssm_norm.weight"))?;
        let out = math::norm(&out, &w, c.linear_value_head_dim, c.rms_norm_eps, false)?
            .astype(DType::Float32)?;
        let z = z.astype(DType::Float32)?;
        let out = if c.output_gate_type == "sigmoid" {
            math::sigmoid_mul(&z, &out)?
        } else {
            math::swiglu(&z, &out)?
        };
        out.astype(dtype)?.reshape(&[1, 1, nh * vd])
    }
    fn attention(&mut self, x: &MxArray, i: usize, cache: &mut LayerCache) -> Result<MxArray> {
        let projections = self.attention_projections(x, i)?;
        self.attention_projected(x, i, cache, &projections, true, true)
    }

    /// Projection matrices can span a prompt window while rotary positions,
    /// sparse selection and recurrent/indexer state still advance causally.
    /// With `store` and pages, the row is written before attention and every
    /// selected row, its own included, is read from the pages (Splash
    /// `PagedAttention::addVerify`, "Encode the store before attention"):
    /// a verify row reads the int8 rows the AR step at its position reads.
    fn attention_projected(
        &mut self,
        x: &MxArray,
        i: usize,
        cache: &mut LayerCache,
        projections: &AttentionProjections,
        project_output: bool,
        store: bool,
    ) -> Result<MxArray> {
        let c = self.config.clone();
        let p = format!("layers.{i}.self_attn");
        let g = format!("blk.{i}");
        let pos = self.position_override.unwrap_or(self.history.len());
        let hd = c.head_dim as i64;
        let nh = c.num_attention_heads as i64;
        let kh = c.num_key_value_heads as i64;
        let qg = projections.qg.reshape(&[1, 1, nh, 2 * hd])?;
        if self.paged.is_none()
            && cache.draft_rows.is_none()
            && let Some((q, k_buf, v_buf, kv_len)) = self.cuda_flat_qk_rope_store(
                &qg,
                &projections.k,
                &projections.v,
                i,
                cache,
                pos,
                1,
                "qwen4::decoder::kv",
            )?
        {
            let id = c.indexer_head_dim as i64;
            cache.index_tail.push(projections.ik.reshape(&[1, id])?);
            if cache.index_tail.len() == c.indexer_compress_ratio {
                let raw = MxArray::concatenate_many(cache.index_tail.iter().collect(), Some(0))?
                    .astype(DType::Float32)?
                    .mean(Some(&[0]), Some(true))?
                    .astype(x.dtype()?)?;
                let raw = self.norm(
                    &raw,
                    &format!("{p}.indexer.k_layernorm.weight"),
                    &format!("{g}.indexer.k_norm.weight"),
                    c.indexer_head_dim,
                    true,
                )?;
                let block = self.rope(&raw, pos + 1 - c.indexer_compress_ratio)?;
                let keys = self.append_index_blocks(i, cache, block)?;
                self.submit_state(&[&keys], "qwen4::decoder::block")?;
                cache.index_tail.clear();
            }
            let topk = c.indexer_budget / c.indexer_compress_ratio;
            let nb = cache.index_block_count()?;
            let sparse = nb > topk;
            let out = if let Some(cuda_out) = self.cuda_flat_sdpa(
                &q,
                &k_buf,
                &v_buf,
                kv_len,
                &projections.iq,
                i,
                cache,
                pos,
                1,
                sparse,
            )? {
                cuda_out
            } else if sparse {
                let (selected, query_ends) = self.qsa_select(i, cache, &projections.iq, pos, 1)?;
                let k_active = k_buf.slice_axis(2, 0, kv_len as i64)?;
                let v_active = v_buf.slice_axis(2, 0, kv_len as i64)?;
                math::qsa_attention_flat(
                    &q,
                    &k_active,
                    &v_active,
                    &selected,
                    &query_ends,
                    pos,
                    (c.head_dim as f64).powf(-0.5),
                    c.indexer_compress_ratio,
                )?
            } else {
                let k_active = k_buf.slice_axis(2, 0, kv_len as i64)?;
                let v_active = v_buf.slice_axis(2, 0, kv_len as i64)?;
                scaled_dot_product_attention(
                    &q,
                    &k_active,
                    &v_active,
                    (c.head_dim as f64).powf(-0.5),
                    None,
                )?
            };
            let out = math::attention_output(&out, &qg)?;
            return if project_output {
                self.linear(
                    &out,
                    &format!("{p}.o_proj.weight"),
                    &format!("{g}.attn_output.weight"),
                )
            } else {
                Ok(out)
            };
        }
        let q = qg.slice_axis(3, 0, hd)?;
        let q = self.attention_norm_rotary(
            &q.transpose(Some(&[0, 2, 1, 3]))?,
            &format!("{p}.q_norm.weight"),
            &format!("{g}.attn_q_norm.weight"),
            pos,
            1,
            false,
        )?;
        let k = self.attention_norm_rotary(
            &projections
                .k
                .reshape(&[1, 1, kh, hd])?
                .transpose(Some(&[0, 2, 1, 3]))?,
            &format!("{p}.k_norm.weight"),
            &format!("{g}.attn_k_norm.weight"),
            pos,
            1,
            false,
        )?;
        let v = projections
            .v
            .reshape(&[1, 1, kh, hd])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let layer = (0..i).filter(|&l| !self.config.linear(l)).count() as u32;
        let stored = match &mut self.paged {
            Some(adapter) if store => {
                let shape = [1, kh, hd];
                super::paged::write_rows(
                    adapter,
                    layer,
                    &k.transpose(Some(&[0, 2, 1, 3]))?.reshape(&shape)?,
                    &v.transpose(Some(&[0, 2, 1, 3]))?.reshape(&shape)?,
                    pos as u32,
                    self.paged_chunk_start as u32,
                )?;
                true
            }
            _ => false,
        };
        let (keys, values) = if stored {
            (k, v)
        } else if cache.draft_rows.is_some() {
            let (keys, values) = cache.append_draft_kv(&k, &v)?;
            self.submit_state(&[&keys, &values], "qwen4::decoder::kv")?;
            cache.keys = Some(keys.clone());
            cache.values = Some(values.clone());
            (keys, values)
        } else if self.paged.is_none() {
            let (keys, values) = cache.append_flat_kv(&k, &v)?;
            if let (Some(k_buf), Some(v_buf)) = (&cache.keys, &cache.values) {
                self.submit_state(&[k_buf, v_buf], "qwen4::decoder::kv")?;
            }
            (keys, values)
        } else {
            let keys = match &cache.keys {
                Some(old) => MxArray::concatenate(old, &k, 2)?,
                None => k,
            };
            let values = match &cache.values {
                Some(old) => MxArray::concatenate(old, &v, 2)?,
                None => v,
            };
            self.submit_state(&[&keys, &values], "qwen4::decoder::kv")?;
            cache.keys = Some(keys.clone());
            cache.values = Some(values.clone());
            (keys, values)
        };
        let ih = c.indexer_n_heads as i64;
        let id = c.indexer_head_dim as i64;
        let (iq, ik) = (&projections.iq, &projections.ik);
        cache.index_tail.push(ik.reshape(&[1, id])?);
        if cache.index_tail.len() == c.indexer_compress_ratio {
            let raw = MxArray::concatenate_many(cache.index_tail.iter().collect(), Some(0))?
                .astype(DType::Float32)?
                .mean(Some(&[0]), Some(true))?
                .astype(x.dtype()?)?;
            let raw = self.norm(
                &raw,
                &format!("{p}.indexer.k_layernorm.weight"),
                &format!("{g}.indexer.k_norm.weight"),
                c.indexer_head_dim,
                true,
            )?;
            let block = self.rope(&raw, pos + 1 - c.indexer_compress_ratio)?;
            let keys = self.append_index_blocks(i, cache, block)?;
            self.submit_state(&[&keys], "qwen4::decoder::block")?;
            cache.index_tail.clear();
        }
        let topk = c.indexer_budget / c.indexer_compress_ratio;
        let nb = cache.index_block_count()?;
        let index_keys = if nb > topk {
            self.index_blocks(i, cache)?
        } else {
            None
        };
        let selected = if let Some(index_keys) = index_keys.as_ref() {
            let iq = iq.reshape(&[ih, id])?;
            let iq = self.norm(
                &iq,
                &format!("{p}.indexer.q_layernorm.weight"),
                &format!("{g}.indexer.q_norm.weight"),
                c.indexer_head_dim,
                true,
            )?;
            let iq = self.rope(&iq, pos)?.astype(DType::Float32)?;
            let blocks = index_keys.astype(DType::Float32)?;
            let scores = iq
                .matmul(&blocks.transpose(None)?)?
                .clip(Some(0.0), None)?
                .sum(Some(&[0]), Some(false))?;
            // ReLU produces exact ties. Use the reference MLX partition, not
            // a CPU sort with a different tie rule, to preserve selected blocks.
            let scores = scores
                .div_scalar((c.indexer_head_dim as f64).sqrt())?
                .reshape(&[1, 1, -1])?;
            let n = nb as i64;
            let indices = scores
                .argpartition(-(topk as i32), Some(-1))?
                .slice_axis(2, n - topk as i64, n)?
                .reshape(&[-1])?
                .astype(DType::Int32)?
                .sort(Some(0))?;
            let offsets = MxArray::arange(
                0.,
                c.indexer_compress_ratio as f64,
                None,
                Some(DType::Int32),
            )?;
            let ids = indices
                .expand_dims(-1)?
                .mul_scalar(c.indexer_compress_ratio as f64)?
                .astype(DType::Int32)?
                .add(&offsets)?
                .reshape(&[-1])?;
            let tail = MxArray::arange(
                (nb * c.indexer_compress_ratio) as f64,
                (pos + 1) as f64,
                None,
                Some(DType::Int32),
            )?;
            MxArray::concatenate(&ids, &tail, 0)?
        } else {
            MxArray::arange(0., (pos + 1) as f64, None, Some(DType::Int32))?
        };
        let (keys, values) = if let Some(adapter) = &self.paged {
            let base = if stored {
                pos + 1
            } else {
                self.paged_chunk_start
            };
            super::paged::gather_window(adapter, layer, &selected, &keys, &values, base)?
        } else {
            (keys.take(&selected, 2)?, values.take(&selected, 2)?)
        };
        let out =
            scaled_dot_product_attention(&q, &keys, &values, (c.head_dim as f64).powf(-0.5), None)?;
        let out = math::attention_output(&out, &qg)?;
        if project_output {
            self.linear(
                &out,
                &format!("{p}.o_proj.weight"),
                &format!("{g}.attn_output.weight"),
            )
        } else {
            Ok(out)
        }
    }
    fn moe(&mut self, x: &MxArray, i: usize) -> Result<MxArray> {
        let c = self.config.clone();
        let p = format!("layers.{i}.mlp");
        let g = format!("blk.{i}");
        let logits = self.linear(
            x,
            &format!("{p}.gate.weight"),
            &format!("{g}.ffn_gate_inp.weight"),
        )?;
        let (selected, scores) =
            if let Some(routes) = math::singleton_routes(&logits, c.num_experts_per_tok)? {
                routes
            } else {
                let probs = Activations::softmax_precise(&logits, Some(-1))?;
                let ne = c.num_experts as i64;
                let top = c.num_experts_per_tok as i64;
                let selected =
                    probs
                        .argpartition(-(top as i32), Some(-1))?
                        .slice_axis(2, ne - top, ne)?;
                let scores = probs.take_along_axis(&selected, -1)?;
                let scores = scores.div(&scores.sum(Some(&[-1]), Some(true))?)?;
                (selected, scores)
            };
        if !self.scope.is_empty()
            && let Some(sum) = self.mtp_resident_experts(x, &selected, &scores)?
        {
            let (shared, gate) = self.shared_expert_parts(x, i)?;
            return sum
                .astype(x.dtype()?)?
                .add(&math::sigmoid_mul(&gate, &shared)?);
        }
        if let Some(out) = self.tentative_shared_experts(x, &selected, &scores, i)? {
            return Ok(out);
        }
        if self.device_routes.is_none()
            && let Some(out) = self.cuda_fused_moe(x, &selected, &scores, i)?
        {
            return Ok(out);
        }
        let device = self.tentative_experts(x, &selected, &scores, i)?;
        let resident = if device.is_some() {
            device
        } else {
            self.resident_experts(x, &selected, &scores, i)?
        };
        let ids = if resident.is_none() {
            self.read_routes(&selected)?
        } else {
            Vec::new()
        };
        let sum = if let Some(sum) = resident {
            sum
        } else if let Some(sum) = self.selected_experts(x, &ids, &scores, i)? {
            sum
        } else {
            let mut weighted = Vec::new();
            for (j, &e) in ids.iter().enumerate() {
                let e = e as usize;
                self.check_cancelled()?;
                let m = c.moe_intermediate_size;
                let (gate, up) = if self.gguf() {
                    (
                        self.weights
                            .read(&format!("{g}.ffn_gate_exps.weight"), e * m, m)?
                            .linear(x)?,
                        self.weights
                            .read(&format!("{g}.ffn_up_exps.weight"), e * m, m)?
                            .linear(x)?,
                    )
                } else {
                    let key = self.key(&format!("{p}.experts.gate_up_proj"), "");
                    let both = self.weights.read(&key, e * 2 * m, 2 * m)?.linear(x)?;
                    (
                        both.slice_axis(2, 0, m as i64)?,
                        both.slice_axis(2, m as i64, 2 * m as i64)?,
                    )
                };
                let h = math::swiglu(&gate, &up)?;
                let key = self.key(
                    &format!("{p}.experts.down_proj"),
                    &format!("{g}.ffn_down_exps.weight"),
                );
                let out = self
                    .weights
                    .read(&key, e * c.hidden_size, c.hidden_size)?
                    .linear(&h)?;
                let value = out.mul(&scores.slice_axis(2, j as i64, j as i64 + 1)?)?;
                MxArray::eval_arrays_with_context(&[&value], "qwen4::decoder::value")?;
                weighted.push(value);
            }
            MxArray::concatenate_many(weighted.iter().collect(), Some(1))?
                .sum(Some(&[1]), Some(true))?
        };
        let (shared, gate) = self.shared_expert_parts(x, i)?;
        sum.astype(x.dtype()?)?
            .add(&math::sigmoid_mul(&gate, &shared)?)
    }
    /// A layer's routes on the host: the pass waits here for the GPU.
    fn read_routes(&mut self, selected: &MxArray) -> Result<Vec<u32>> {
        #[cfg(test)]
        {
            self.route_readbacks += 1;
        }
        Ok(selected.to_uint32()?.to_vec())
    }

    /// `moe` over a `[1, R, H]` verify window: full direct banks keep all
    /// routes on the device; partial banks use one routing readback, then the
    /// row-batched slot kernels. Without those kernels rows run singly.
    fn moe_rows(&mut self, x: &MxArray, i: usize) -> Result<MxArray> {
        let rows = x.shape_at(1)?;
        if rows == 1 {
            return self.moe(x, i);
        }
        let main = self.gguf();
        let c = self.config.clone();
        if main {
            let logits = self.linear_rows(
                x,
                &format!("layers.{i}.mlp.gate.weight"),
                &format!("blk.{i}.ffn_gate_inp.weight"),
            )?;
            if super::cuda_qwen4_fast_enabled() && (2..=16).contains(&rows) {
                let (selected, scores) = if let Some(routes) =
                    crate::kernels::qwen4::router_topk_cuda(&logits, c.num_experts_per_tok)?
                {
                    routes
                } else {
                    let probs = Activations::softmax_precise(&logits, Some(-1))?;
                    let ne = c.num_experts as i64;
                    let top = c.num_experts_per_tok as i64;
                    let selected =
                        probs
                            .argpartition(-(top as i32), Some(-1))?
                            .slice_axis(2, ne - top, ne)?;
                    let scores = probs.take_along_axis(&selected, -1)?;
                    let scores = scores.div(&scores.sum(Some(&[-1]), Some(true))?)?;
                    (selected, scores)
                };
                if let Some(out) = self.cuda_fused_moe(x, &selected, &scores, i)? {
                    return Ok(out);
                }
            }
            if let Some((selected, scores)) =
                math::rows::singleton_routes(&logits, c.num_experts_per_tok)?
            {
                if let Some(out) = self.tentative_shared_expert_rows(x, &selected, &scores, i)? {
                    return Ok(out);
                }
                let mut sum = self.resident_experts(x, &selected, &scores, i)?;
                if sum.is_none() {
                    let ids = self.read_routes(&selected)?;
                    if let Some(out) = self.slotted_shared_experts(x, &ids, &scores, i)? {
                        return Ok(out);
                    }
                    sum = self.selected_experts(x, &ids, &scores, i)?;
                }
                if let Some(sum) = sum {
                    let (shared, gate) = self.shared_expert_parts(x, i)?;
                    return sum
                        .astype(x.dtype()?)?
                        .add(&math::sigmoid_mul(&gate, &shared)?);
                }
            }
        }
        let parts = (0..rows)
            .map(|j| self.moe(&x.slice_axis(1, j, j + 1)?, i))
            .collect::<Result<Vec<_>>>()?;
        MxArray::concatenate_many(parts.iter().collect(), Some(1))
    }

    /// The routed and shared experts of the verify rows in one kernel over
    /// the host-checked slots: the device-route decode kernel with the
    /// routes already known.
    fn slotted_shared_experts(
        &mut self,
        x: &MxArray,
        ids: &[u32],
        scores: &MxArray,
        i: usize,
    ) -> Result<Option<MxArray>> {
        let c = self.config.clone();
        if c.num_experts_per_tok != 10
            || c.hidden_size != 2560
            || c.moe_intermediate_size != 640
            || x.dtype()? != DType::BFloat16
        {
            return Ok(None);
        }
        let shared = ["gate", "up", "down"].map(|part| {
            self.weights
                .resident_bank(&format!("blk.{i}.ffn_{part}_shexp.weight"))
        });
        let [Some(gate), Some(up), Some(down)] = shared else {
            return Ok(None);
        };
        let shared = [gate, up, down];
        let Some((banks, slots, _)) =
            self.weights
                .expert_slots(i, c.num_experts, c.num_experts_per_tok, ids)?
        else {
            return Ok(None);
        };
        let shared_gate = self.linear_rows(
            x,
            &format!("layers.{i}.mlp.shared_expert_gate.weight"),
            &format!("blk.{i}.ffn_gate_inp_shexp.weight"),
        )?;
        let slots = MxArray::from_uint32(&slots, &[slots.len() as i64])?;
        let Some(out) =
            math::rows::routed_shared_experts(x, &slots, scores, &banks, &shared, &shared_gate)?
        else {
            return Ok(None);
        };
        self.weights.finish_expert_slots(i, &out)?;
        Ok(Some(out))
    }

    /// One decoder layer over the `[1, R, 4H]` verify window. Projections,
    /// mixers, router, experts and output projections read their weights
    /// once for all rows through the row-exact kernels; PLE, GDN and
    /// attention state advance row by row (or batched via CUDA deferred GDN /
    /// batched QSA). `saved[j]` is this layer's cache after row `j`: the
    /// frontier a commit of `j + 1` tokens restores.
    fn verify_layer(
        &mut self,
        x: &MxArray,
        tokens: &[u32],
        i: usize,
        cache: &mut LayerCache,
        attn_normed: Option<&MxArray>,
    ) -> Result<(MxArray, Option<MxArray>, Vec<LayerCache>)> {
        let base = self.history.len();
        let rows = tokens.len();
        let mut saved = vec![LayerCache::default(); rows];
        let row = |a: &MxArray, j: usize| a.slice_axis(1, j as i64, j as i64 + 1);
        let has_ple = self.config.ple_layer_ids.contains(&(i + 1));
        let x = if has_ple {
            let dtype = x.dtype()?;
            let embeddings = self.ple_embeddings(tokens, i, dtype)?;
            let (key, value) = self.ple_projections(&embeddings, i)?;
            let mut parts = Vec::with_capacity(rows);
            for (j, saved) in saved.iter_mut().enumerate() {
                self.check_cancelled()?;
                let input = row(x, j)?;
                let out = self.ple_mix_with_residual(
                    &input,
                    &row(&key, j)?,
                    &row(&value, j)?,
                    i,
                    cache,
                    true,
                )?;
                parts.push(out);
                saved.ple_conv = cache.ple_conv.clone();
            }
            MxArray::concatenate_many(parts.iter().collect(), Some(1))?
        } else {
            x.clone()
        };
        let (mixed, gate) = self.hyper_rows(
            &x,
            &format!("layers.{i}.attn_hyper_connection"),
            &format!("blk.{i}.hc_attn"),
            true,
            if has_ple { None } else { attn_normed },
        )?;
        let branch = if self.config.linear(i) {
            let p = format!("layers.{i}.linear_attn");
            let g = format!("blk.{i}");
            let projections = self.gdn_projections(&mixed, &p, &g)?;
            let out = if self.cuda_verify_gdn_tape.is_some()
                && let Some((out, tape)) = self.cuda_verify_gdn_deferred(&projections, i, cache)?
            {
                if let Some(tapes) = &mut self.cuda_verify_gdn_tape {
                    tapes[i] = Some(tape);
                }
                out
            } else {
                let dtype = mixed.dtype()?;
                let mut outputs = Vec::with_capacity(rows);
                for (j, saved) in saved.iter_mut().enumerate() {
                    self.check_cancelled()?;
                    outputs.push(self.gdn_mix(dtype, &projections.row(j)?, i, cache)?);
                    saved.conv = cache.conv.clone();
                    saved.recurrent = cache.recurrent.clone();
                }
                MxArray::concatenate_many(outputs.iter().collect(), Some(1))?
            };
            self.linear_rows(
                &out,
                &format!("{p}.out_proj.weight"),
                &format!("{g}.ssm_out.weight"),
            )?
        } else {
            let projections = self.attention_projections(&mixed, i)?;
            let out = if let Some(out) =
                self.cuda_verify_attention_batch(&mixed, i, cache, &projections, &mut saved)?
            {
                out
            } else {
                let mut outputs = Vec::with_capacity(rows);
                for (j, (&token, saved)) in tokens.iter().zip(saved.iter_mut()).enumerate() {
                    self.check_cancelled()?;
                    outputs.push(self.attention_projected(
                        &row(&mixed, j)?,
                        i,
                        cache,
                        &projections.row(j)?,
                        false,
                        true,
                    )?);
                    self.history.push(token);
                    if self.paged.is_none() {
                        saved.keys = cache.keys.clone();
                        saved.values = cache.values.clone();
                        saved.flat_kv_rows = cache.flat_kv_rows;
                    }
                    saved.index_tail = cache.index_tail.clone();
                    saved.index_keys = cache.index_keys.clone();
                    saved.flat_index_blocks = cache.flat_index_blocks;
                    saved.index_blocks = cache.index_blocks;
                }
                self.history.truncate(base);
                MxArray::concatenate_many(outputs.iter().collect(), Some(1))?
            };
            self.linear_rows(
                &out,
                &format!("layers.{i}.self_attn.o_proj.weight"),
                &format!("blk.{i}.attn_output.weight"),
            )?
        };
        let gate = gate
            .ok_or_else(|| Error::from_reason("Qwen4 attention is missing its injection gate"))?;
        let (x, normed) = self.inject_for_mlp(&x, &branch, &gate, i, true)?;
        let (mixed, gate) = self.hyper_rows(
            &x,
            &format!("layers.{i}.mlp_hyper_connection"),
            &format!("blk.{i}.hc_ffn"),
            true,
            normed.as_ref(),
        )?;
        let branch = self.moe_rows(&mixed, i)?;
        let gate =
            gate.ok_or_else(|| Error::from_reason("Qwen4 MLP is missing its injection gate"))?;
        let next_layer = i + 1;
        let cuda_fast = super::cuda_qwen4_fast_enabled();
        let (stream, next_normed) = if cuda_fast && next_layer < self.config.num_hidden_layers {
            if !self.config.ple_layer_ids.contains(&(next_layer + 1)) {
                self.inject_for_hyper(
                    &x,
                    &branch,
                    &gate,
                    &format!("layers.{next_layer}.attn_hyper_connection.hc_norm.weight"),
                    &format!("blk.{next_layer}.hc_attn_norm.weight"),
                    true,
                )?
            } else {
                (Self::inject(&x, &branch, &gate)?, None)
            }
        } else if cuda_fast && next_layer == self.config.num_hidden_layers && self.scope.is_empty()
        {
            self.inject_for_hyper(
                &x,
                &branch,
                &gate,
                "hyper_connection_mixer.hc_norm.weight",
                "output_hc_norm.weight",
                true,
            )?
        } else {
            (Self::inject(&x, &branch, &gate)?, None)
        };
        Ok((stream, next_normed, saved))
    }

    pub fn step(&mut self, token: u32) -> Result<MxArray> {
        self.settle_cuda_pending_gdn()?;
        self.rotary_tables.get_mut().clear();
        let started = std::time::Instant::now();
        let pass = std::mem::replace(&mut self.decode_pass, true);
        let result = self.forward_with_device_routes(token, true);
        self.decode_pass = pass;
        self.rotary_tables.get_mut().clear();
        // A read failure can happen after earlier layers have advanced.
        // Discard the entire prefix instead of reusing partial state.
        if result.is_err() {
            self.reset();
        } else {
            self.observe_pass(1, started.elapsed());
        }
        result
    }

    fn layer_with_norm(
        &mut self,
        x: MxArray,
        token: u32,
        i: usize,
        cache: &mut LayerCache,
        attn_normed: Option<&MxArray>,
        fuse_next_norm: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let (x, normed) =
            self.attention_block_for_mlp_with_norm(x, token, i, cache, true, attn_normed)?;
        let (mixed, gate) = self.hyper_with_norm(
            &x,
            &format!("layers.{i}.mlp_hyper_connection"),
            &format!("blk.{i}.hc_ffn"),
            true,
            normed.as_ref(),
        )?;
        let branch = self.moe(&mixed, i)?;
        #[cfg(test)]
        let branch = if self.poison_moe_at == Some(i) {
            self.poison_moe_at = None;
            MxArray::zeros(&branch.shape()?, Some(branch.dtype()?))?
        } else {
            branch
        };
        let gate =
            gate.ok_or_else(|| Error::from_reason("Qwen4 MLP is missing its injection gate"))?;
        if fuse_next_norm {
            let next_layer = i + 1;
            if next_layer < self.config.num_hidden_layers {
                if !self.config.ple_layer_ids.contains(&(next_layer + 1)) {
                    return self.inject_for_hyper(
                        &x,
                        &branch,
                        &gate,
                        &format!("layers.{next_layer}.attn_hyper_connection.hc_norm.weight"),
                        &format!("blk.{next_layer}.hc_attn_norm.weight"),
                        true,
                    );
                }
            } else if self.scope.is_empty() {
                return self.inject_for_hyper(
                    &x,
                    &branch,
                    &gate,
                    "hyper_connection_mixer.hc_norm.weight",
                    "output_hc_norm.weight",
                    true,
                );
            }
        }
        Ok((Self::inject(&x, &branch, &gate)?, None))
    }

    fn attention_block(
        &mut self,
        x: MxArray,
        token: u32,
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<MxArray> {
        Ok(self.attention_block_for_mlp(x, token, i, cache, false)?.0)
    }

    fn attention_block_for_mlp(
        &mut self,
        x: MxArray,
        token: u32,
        i: usize,
        cache: &mut LayerCache,
        normalize: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        self.attention_block_for_mlp_with_norm(x, token, i, cache, normalize, None)
    }

    fn attention_block_for_mlp_with_norm(
        &mut self,
        mut x: MxArray,
        token: u32,
        i: usize,
        cache: &mut LayerCache,
        normalize: bool,
        attn_normed: Option<&MxArray>,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let has_ple = self.config.ple_layer_ids.contains(&(i + 1));
        if has_ple {
            x = self.ple_apply(&x, &[token], i, cache)?;
        }
        let (mixed, gate) = self.hyper_with_norm(
            &x,
            &format!("layers.{i}.attn_hyper_connection"),
            &format!("blk.{i}.hc_attn"),
            true,
            if has_ple { None } else { attn_normed },
        )?;
        let branch = if self.config.linear(i) {
            self.gdn(&mixed, i, cache)?
        } else {
            self.attention(&mixed, i, cache)?
        };
        let gate = gate
            .ok_or_else(|| Error::from_reason("Qwen4 attention is missing its injection gate"))?;
        self.inject_for_mlp(&x, &branch, &gate, i, normalize)
    }

    /// The token's embedding row as a lazy array: an LRU hit or an import
    /// the consuming layer submits, never a drain at the token start.
    pub fn embed_token(&mut self, token: u32) -> Result<MxArray> {
        let key = self.key("embed_tokens.weight", "token_embd.weight");
        self.weights
            .lookup_rows(&key, &[token as usize])?
            .reshape(&[1, 1, self.config.hidden_size as i64])
    }

    pub fn verify_chunk(&mut self, tokens: &[u32]) -> Result<(MxArray, Vec<DecoderState>)> {
        if tokens.is_empty() || tokens.len() > 4 {
            return Err(Error::from_reason("Qwen4 verification width must be 1-4"));
        }
        self.settle_cuda_pending_gdn()?;
        self.rotary_tables.get_mut().clear();
        self.cuda_verify_final_window = None;
        self.cuda_verify_final_normed = None;
        let started = std::time::Instant::now();
        self.verification = Some(Vec::new());
        if self.verify_routes_on_device() {
            self.device_routes = Some(Vec::with_capacity(self.config.num_hidden_layers));
        }
        let pass = std::mem::replace(&mut self.decode_pass, true);
        let result = (|| {
            if cfg!(target_os = "macos") {
                self.weights.check_forward_headroom()?;
            }
            self.prefill_chunk_inner(tokens, &[], None, false)?;
            // The pass stays open through the head: these rows are row-exact
            // too, and `linear_rows` is batched outside a verification pass.
            let window = match self.cuda_verify_final_window.take() {
                Some(w) => w,
                None => {
                    MxArray::concatenate_many(self.last_chunk_hidden.iter().collect(), Some(1))?
                }
            };
            let normed = self.cuda_verify_final_normed.take();
            let (mixed, _) = self.hyper_rows(
                &window,
                "hyper_connection_mixer",
                "output_hc",
                false,
                normed.as_ref(),
            )?;
            let logits = self
                .linear_rows(&mixed, "lm_head.weight", "output.weight")?
                .reshape(&[tokens.len() as i64, -1])?;
            let states = self.verification.take().ok_or_else(|| {
                Error::from_reason("Qwen4 verification lost its accepted-frontier snapshots")
            })?;
            let routes = self.device_routes.take();
            let tape = routes
                .as_ref()
                .filter(|routes| !routes.is_empty())
                .map(|routes| {
                    MxArray::concatenate_many(routes.iter().map(|(_, ids)| ids).collect(), Some(0))
                })
                .transpose()?;
            // Commit rolls the page cursor back and restores one of these
            // states: every pool write and frontier array must be complete.
            let completed: Vec<_> = std::iter::once(&logits)
                .chain(states.iter().flat_map(DecoderState::arrays))
                .chain(self.caches.iter().flat_map(LayerCache::arrays))
                .chain(
                    self.hidden_taps
                        .iter()
                        .flat_map(|taps| taps.last_pass_arrays().iter()),
                )
                .chain(tape.iter())
                .collect();
            if super::cuda_qwen4_fast_enabled() {
                Self::cuda_async_eval(&completed);
            } else {
                MxArray::async_eval_arrays(&completed);
            }
            if let Some(adapter) = &mut self.paged {
                adapter
                    .eval_pending_pool_writes()
                    .map_err(Error::from_reason)?;
            }
            MxArray::eval_arrays_with_context(&completed, "qwen4::decoder::verify")?;
            if let (Some(routes), Some(tape)) = (routes, tape) {
                let layers: Vec<_> = routes.iter().map(|(layer, _)| *layer).collect();
                if !self.weights.commit_device_routes(
                    &layers,
                    &tape.to_uint32()?,
                    tokens.len() * self.config.num_experts_per_tok,
                )? {
                    return Err(Error::from_reason(
                        "Qwen4 verification routes lost their expert banks",
                    ));
                }
            }
            Ok((logits, states))
        })();
        self.decode_pass = pass;
        self.rotary_tables.get_mut().clear();
        match result {
            Ok(verified) => {
                self.observe_pass(tokens.len(), started.elapsed());
                Ok(verified)
            }
            Err(error) => {
                self.reset();
                Err(error)
            }
        }
    }

    /// The once-per-token start of a decode pass: the frontier check, the
    /// page cursor's record, the row requests and the embedding, as the
    /// stream entering layer 0. `embedding_ahead` hands the token's embedding
    /// row to the reader threads with its PLE rows; else this thread reads
    /// it when the token starts.
    pub(super) fn begin_decode_token(
        &mut self,
        token: u32,
        next: &[u32],
        embedding: Option<&MxArray>,
        embedding_ahead: bool,
    ) -> Result<MxArray> {
        let base = self.history.len();
        if base + 1 > self.context_window || token as usize >= self.config.vocab_size {
            return Err(Error::from_reason(
                "Qwen4 prefill chunk exceeds token/context budget",
            ));
        }
        if let Some(e) = embedding
            && *e.shape()? != [1, 1, self.config.hidden_size as i64]
        {
            return Err(Error::from_reason("Qwen4 media embedding shape mismatch"));
        }
        self.paged_chunk_start = base;
        if let Some(adapter) = &mut self.paged {
            if adapter.request_tokens() != self.history {
                return Err(Error::from_reason(
                    "Qwen4 paged/recurrent frontier mismatch",
                ));
            }
            adapter
                .record_tokens(&[token])
                .map_err(Error::from_reason)?;
        }
        self.prefetch_window_rows(&[token], next, embedding.is_none() && embedding_ahead)?;
        self.check_cancelled()?;
        let row = match embedding {
            Some(e) => e.slice_axis(1, 0, 1)?,
            None => self.embed_token(token)?,
        };
        MxArray::tile(&row, &[1, 1, self.config.hc_count as i32])
    }

    /// Whether this decode token keeps its layer inputs: a tentative pass
    /// over a partial store.
    fn keeps_layer_inputs(&self) -> bool {
        #[cfg(test)]
        if let Some(pin) = self.layer_inputs_pin {
            return pin;
        }
        self.device_routes.is_some() && !self.weights.plan.direct
    }

    /// One decode token over every layer and the head.
    fn decode_token(
        &mut self,
        token: u32,
        next: &[u32],
        embedding: Option<&MxArray>,
        embedding_ahead: bool,
        project: bool,
    ) -> Result<MxArray> {
        let LayerPass::Done(out) =
            self.decode_pass(token, next, embedding, embedding_ahead, project)?
        else {
            return Err(Error::from_reason(
                "Qwen4 decode pass stopped before its last layer",
            ));
        };
        Ok(out)
    }

    /// `decode_token` that hands back a pass a route miss stopped.
    pub(super) fn decode_pass(
        &mut self,
        token: u32,
        next: &[u32],
        embedding: Option<&MxArray>,
        embedding_ahead: bool,
        project: bool,
    ) -> Result<LayerPass> {
        let x = self.begin_decode_token(token, next, embedding, embedding_ahead)?;
        self.begin_taps();
        self.layer_inputs = self
            .keeps_layer_inputs()
            .then(|| Vec::with_capacity(self.config.num_hidden_layers));
        let pass = self.decode_layers(0, x, token, project)?;
        self.group_streams = Vec::new();
        Ok(pass)
    }

    /// Layers `from..` of the decode token begun by `begin_decode_token`,
    /// on `x`, the stream entering layer `from`, then the head. The caller
    /// gives layers `from..` the caches they held before this token. A
    /// device pass that keeps its layer inputs submits each group's route
    /// entries with it and stops after a group submit once an earlier group's
    /// finished routes name an expert with no slot.
    pub(super) fn decode_layers(
        &mut self,
        from: usize,
        mut x: MxArray,
        token: u32,
        project: bool,
    ) -> Result<LayerPass> {
        let layers = self.config.num_hidden_layers;
        let submit_span = self.decode_submit_span();
        let defer_final = self.deferred_forward_completion();
        let polls = self.device_routes.is_some() && self.layer_inputs.is_some();
        #[cfg(test)]
        let polls = polls && self.after_group_submit != Some(GroupPoll::Never);
        // Route entries [polled, submitted) of this pass were submitted and
        // not yet read.
        let (mut polled, mut submitted) = (0, 0);
        if let Some(inputs) = &mut self.layer_inputs {
            if inputs.len() < from {
                return Err(Error::from_reason(format!(
                    "Qwen4 layer inputs hold no input for layer {from}"
                )));
            }
            inputs.truncate(from);
        }
        let cuda_fast = super::cuda_qwen4_fast_enabled();
        let cuda_defer_last = self.cuda_deferred_state();
        let mut next_normed: Option<MxArray> = None;
        for i in from..layers {
            self.check_cancelled()?;
            if let Some(inputs) = &mut self.layer_inputs {
                inputs.push(x.clone());
                #[cfg(test)]
                {
                    self.kept_layer_inputs += 1;
                }
            }
            #[cfg(test)]
            if self.stop_before == Some(i) {
                return Ok(LayerPass::Stopped { at: i });
            }
            let mut cache = std::mem::take(&mut self.caches[i]);
            let (next, carried_normed) =
                self.layer_with_norm(x, token, i, &mut cache, next_normed.as_ref(), cuda_fast)?;
            #[cfg(test)]
            if let Some(outputs) = &mut self.layer_outputs {
                outputs.push((i, next.clone()));
            }
            self.tap_layer(i, std::slice::from_ref(&next))?;
            let last = i + 1 == layers;
            let mut poll = None;
            if (defer_final || cuda_defer_last) && last {
                // TrackFastModel and CUDA direct decode return/join at the
                // final mixer/head with all state/taps in one sync.
            } else if (i + 1) % submit_span != 0 && !last {
                if self.decode_should_submit(i + 1) {
                    #[cfg(test)]
                    if polls && self.after_group_submit == Some(GroupPoll::Older) {
                        crate::array::synchronize();
                    }
                    let mut group = vec![&next];
                    if let Some(ref normed) = carried_normed {
                        group.push(normed);
                    }
                    if polls && let Some(routes) = &self.device_routes {
                        poll = Some(submitted);
                        group.extend(routes[submitted..].iter().map(|(_, ids)| ids));
                        submitted = routes.len();
                    }
                    self.submit(&group, "qwen4::decoder::next")?;
                }
            } else if let Some(ref normed) = carried_normed {
                MxArray::eval_arrays_with_context(&[&next, normed], "qwen4::decoder::next")?;
            } else {
                MxArray::eval_arrays_with_context(&[&next], "qwen4::decoder::next")?;
            }
            x = next;
            next_normed = carried_normed;
            self.caches[i] = cache;
            #[cfg(test)]
            let poll = poll.map(|older| self.pinned_poll(older, submitted));
            if let Some(upto) = poll
                && self.routes_missed(upto, &mut polled)?
            {
                return Ok(LayerPass::Stopped { at: i + 1 });
            }
            if ((i + 1) % submit_span == 0 || last)
                && !(defer_final && last)
                && let Some(adapter) = &mut self.paged
            {
                adapter
                    .eval_pending_pool_writes()
                    .map_err(Error::from_reason)?;
            }
            // Keep reusable buffers bounded after all layer consumers complete.
            super::memory::maintain_freelist(self.weights.plan.physical_bytes);
        }
        self.finish_taps(self.history.len())?;
        self.last_chunk_hidden = vec![x.clone()];
        let (mixed, _) = self.hyper_with_norm(
            &x,
            "hyper_connection_mixer",
            "output_hc",
            false,
            next_normed.as_ref(),
        )?;
        let out = if project {
            self.linear(&mixed, "lm_head.weight", "output.weight")?
        } else {
            mixed
        };
        if !defer_final {
            if cuda_fast && self.cuda_ar_lazy_logits && self.paged.is_none() {
                // `Step::eval_step` submits pending states + taps + sampled token
                // together via non-blocking `cuda_async_eval` so AR runs with 1 sync/tok.
            } else if cuda_fast {
                let mut completed = vec![&out];
                completed.extend(self.pending_state_arrays());
                if let Some(taps) = &self.hidden_taps {
                    completed.extend(taps.last_pass_arrays().iter());
                }
                MxArray::eval_arrays_with_context(&completed, "qwen4::decoder::logits")?;
            } else {
                MxArray::eval_arrays_with_context(&[&out], "qwen4::decoder::logits")?;
            }
        }
        self.history.push(token);
        Ok(LayerPass::Done(out))
    }
}
