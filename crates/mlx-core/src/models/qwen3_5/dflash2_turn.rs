//! DFlash2 pieces shared by the dense and MoE Qwen3.5 targets: the draft
//! window a tapped paged prefill advances, and the take order a refused
//! paged step reclaims through. The speculative steppers stay per family.

use std::collections::HashSet;

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::engine::backend::ChatBackend;
use crate::engine::reclaim::{
    KvNeed, KvReclaim, MetBefore, Reach, Unit, UnitKey, free_for_kv, older_first,
};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

use super::dflash2::{DFlash2ContextCache, DFlash2Model};
use super::dflash2_state::paged::{DFlash2States, PagedAnchor};
use super::dflash2_state::store::{EntryId, Lineage};

/// The draft window a tapped paged prefill built, handed to the turn's
/// stepper.
pub(crate) struct DFlash2TurnState {
    pub(crate) context: DFlash2ContextCache,
    pub(crate) next_position: i32,
}

/// First prompt row whose draft K/V a prefill of `[resume, end)` projects:
/// each layer's sliding cache keeps only the newest `window_rows` rows, so
/// older rows would be evicted unread and only advance length and
/// provenance (`record_only`). A capture armed at `capture > resume` needs
/// the rows its snapshot reads, `[capture - window_rows, capture)`.
pub(crate) fn draft_keep_from(
    resume: usize,
    end: usize,
    window_rows: usize,
    capture: Option<usize>,
) -> usize {
    let kept_end = match capture {
        Some(row) if row > resume => row.min(end),
        _ => end,
    };
    kept_end.saturating_sub(window_rows).max(resume)
}

/// The prompt a reclaim keeps: its ladder and history entries are no
/// units, and the marked blocks it hits are out of the count. With the
/// prompt's block keys and salt the hit and its ladder rungs are read by
/// hash; without them, by token prefix.
#[derive(Clone, Copy, Default)]
pub(crate) struct Keep<'a> {
    pub tokens: &'a [u32],
    pub keys: Option<(&'a [Vec<u64>], u64)>,
}

impl<'a> Keep<'a> {
    /// The salted, keyed chain hashes the prompt's lookup walks, its last
    /// token excluded (the lookup leaves one token to prefill); `None`
    /// without keys.
    pub(crate) fn hit_chain(&self, block_size: u32) -> Option<Vec<u64>> {
        let (keys, salt) = self.keys?;
        let prompt = &self.tokens[..self.tokens.len().saturating_sub(1)];
        Some(mlx_paged_attn::PrefixKeys::PerBlock(keys).chain_hashes(prompt, block_size, salt))
    }

    /// The replay points this prompt can restore, resolved once for a
    /// reclaim (Splash `RestoredChains`). The token test leaves out the
    /// last token as the lookup does (Splash `Cache::extendMatch`): a point
    /// at the prompt's own end is no replay point of it.
    pub(crate) fn points(self, block_size: Option<u32>) -> KeptPoints<'a> {
        KeptPoints {
            tokens: &self.tokens[..self.tokens.len().saturating_sub(1)],
            chain: block_size.and_then(|block_size| self.hit_chain(block_size)),
        }
    }
}

/// What a [`Keep`] protects from a reclaim. A ladder rung on another salt
/// or image-key chain is no replay point of the prompt even when its
/// tokens prefix it: the lookup rejects it by hash (Splash
/// `Walk::keepsState`).
pub(crate) struct KeptPoints<'a> {
    tokens: &'a [u32],
    chain: Option<Vec<u64>>,
}

impl KeptPoints<'_> {
    /// Whether an entry on `tokens` is kept: by hash chain when the prompt
    /// has keys and the entry sits on blocks (`block_hashes`), else by
    /// token prefix.
    pub(crate) fn keeps(&self, tokens: &[u32], block_hashes: Option<&[u64]>) -> bool {
        match (&self.chain, block_hashes) {
            (Some(chain), Some(hashes)) => !hashes.is_empty() && chain.starts_with(hashes),
            _ => !self.tokens.is_empty() && self.tokens.starts_with(tokens),
        }
    }
}

/// A Qwen3.5-family target model that runs DFlash2 turns on its paged pool.
pub(crate) trait DFlash2Target: ChatBackend {
    fn dflash2(&self) -> Option<&DFlash2Model>;
    fn dflash2_states(&self) -> &DFlash2States;
    fn dflash2_states_mut(&mut self) -> &mut DFlash2States;
    /// Dense-shaped config the load plan prices.
    fn dense_config(&self) -> std::borrow::Cow<'_, super::config::Qwen3_5Config>;
    /// Embedding output dtype: what an unallocated BF16 cache will hold.
    fn embedding_dtype(&self) -> Result<DType>;
    fn paged_adapter_ref(&self) -> Option<&PagedKVCacheAdapter>;
    fn paged_adapter_mut_ref(&mut self) -> Option<&mut PagedKVCacheAdapter>;
    /// The image-feature cache (Splash's `embeddings` pool).
    fn vision_cache(&self) -> &crate::vision::qwen::cache::VisionCache;

    /// The oldest ladder or history entry a KV need may take, of those a
    /// turn uses (`in_use`) or of the rest.
    fn oldest_gdn_state(
        &self,
        at_cap: bool,
        skip: &HashSet<UnitKey>,
        keep: Keep<'_>,
        in_use: bool,
    ) -> Option<Unit>;
    /// Drops a ladder or history entry; false when it is gone.
    fn drop_gdn_state(&mut self, key: &UnitKey) -> bool;
    /// Marks every ladder rung's and parked DFlash2 state's block.
    fn sync_gdn_state_marks(&self);
    /// Bytes of the GDN entries a reclaim may take, with `in_use` those
    /// turns use too.
    fn gdn_state_bytes(&self, keep: Keep<'_>, in_use: bool) -> u64;
    /// Marked cache-only blocks the kept prompt hits.
    fn marked_hit_blocks(&self, keep: Keep<'_>) -> u32;
    /// The blocks the ladder rungs mark.
    fn ladder_marks(&self) -> HashSet<u64>;
    /// Fresh per-layer caches for this model.
    fn fresh_layer_caches(&self) -> Vec<super::layer_cache::Qwen3_5LayerCache>;
    /// Puts the lane at a parked state's text lineage: its caches, draft
    /// window and history; no image lineage stays.
    fn adopt_parked_state(
        &mut self,
        caches: Vec<super::layer_cache::Qwen3_5LayerCache>,
        window: DFlash2ContextCache,
        history: Vec<u32>,
    );
    /// The live GDN state agrees with its tokens and carries no image
    /// lineage, so a state captured from it is exact on a text chain.
    fn replay_state_exact(&self) -> bool;
    /// Drops the draft state a tapped prefill handed the stepper.
    fn dflash2_turn_state_clear(&mut self);
}

/// Where a DFlash2 turn resumes: the live request with its draft window, or
/// a parked state installed at its block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DFlash2Hit {
    Live(u32),
    Stored(u32),
}

/// The prompt a DFlash2 prime keeps from its reclaims: a restored chain is
/// kept for its request (Splash `Cache::kept`); the live table is no unit.
pub(crate) fn prime_keep<'a>(
    plan: &'a [u32],
    keys: &'a [Vec<u64>],
    cache_salt: u64,
    hit: Option<DFlash2Hit>,
) -> Keep<'a> {
    match hit {
        Some(DFlash2Hit::Live(_)) => Keep::default(),
        None | Some(DFlash2Hit::Stored(_)) => Keep {
            tokens: plan,
            keys: Some((keys, cache_salt)),
        },
    }
}

/// The DFlash2 prepare. A turn resumes only at its state's exact block
/// (Splash resumes a state at its boundary): a live window continues the
/// live request, a parked state looks its chain up capped at its block. Any
/// other hit restarts cold.
pub(crate) fn prepare_dflash2_turn(
    adapter: &mut PagedKVCacheAdapter,
    seq_id: u32,
    plan: &[u32],
    keys: &[Vec<u64>],
    cache_salt: u64,
    hit: DFlash2Hit,
) -> Result<crate::transformer::paged_kv_cache_adapter::PagedTurnPlan> {
    let total_budget = u32::try_from(plan.len()).unwrap_or(u32::MAX);
    let (skip_lookup, candidate) = match hit {
        DFlash2Hit::Live(depth) => (depth == 0, depth),
        DFlash2Hit::Stored(depth) => (false, depth),
    };
    let prepared = adapter
        .prepare_turn_per_block_booking_hit_only(
            seq_id,
            plan,
            total_budget,
            keys,
            cache_salt,
            skip_lookup,
            candidate,
        )
        .map_err(Error::from_reason)?;
    let resumed = match hit {
        DFlash2Hit::Live(_) => prepared.continued_live_prefix,
        DFlash2Hit::Stored(_) => !prepared.continued_live_prefix,
    } && prepared.cached_prefix_len == candidate;
    if prepared.cached_prefix_len == 0 || resumed {
        return Ok(prepared);
    }
    adapter
        .prepare_turn_per_block_booking_hit_only(
            seq_id,
            plan,
            total_budget,
            keys,
            cache_salt,
            true,
            0,
        )
        .map_err(Error::from_reason)
}

/// The deepest parked state a DFlash2 turn of `tokens` resumes from when it
/// beats the live window of `live` tokens (Splash
/// `StateCache::acquireDeepest`); `None` without reuse.
pub(crate) fn parked_state_hit<T: DFlash2Target>(
    target: &T,
    tokens: &[u32],
    reuse_cache: bool,
    cache_salt: u64,
    live: usize,
) -> Option<super::dflash2_state::store::StoreHit> {
    if !reuse_cache {
        return None;
    }
    target
        .dflash2_states()
        .store
        .lookup(Lineage::text(tokens, cache_salt))
        .filter(|hit| hit.beats_live((live > 0).then_some(live)))
}

/// Installs parked state `id` into the lane of `seq`: the live request goes
/// (a restore resumes only at the state's block), the GDN cell and window
/// are deep-copied in, the history is the state's lineage, and the state
/// stays parked with a new stamp (Splash `QwenStateStorage::restore`,
/// `StateCache::release`). Returns the resume depth.
pub(crate) fn install_parked_state<T: DFlash2Target>(
    target: &mut T,
    id: EntryId,
    seq: u32,
) -> Result<u32> {
    let (mark, depth) = {
        let entry = target
            .dflash2_states()
            .store
            .get(id)
            .ok_or_else(|| Error::from_reason(format!("DFlash2 parked state {id} is gone")))?;
        (entry.mark(), entry.depth())
    };
    if !target
        .paged_adapter_ref()
        .is_some_and(|adapter| adapter.prefix_cached(mark))
    {
        return Err(Error::from_reason(
            "DFlash2 state outlived its target KV block",
        ));
    }
    if let Some(adapter) = target.paged_adapter_mut_ref() {
        adapter
            .release_request_for(seq)
            .map_err(Error::from_reason)?;
    }
    let mut caches = target.fresh_layer_caches();
    let (window, history) = {
        let entry = target
            .dflash2_states()
            .store
            .get(id)
            .ok_or_else(|| Error::from_reason(format!("DFlash2 parked state {id} is gone")))?;
        super::dflash2_state::paged::install_linear(&mut caches, &entry.state().linear)?;
        (entry.state().window.deep_copy()?, entry.tokens().to_vec())
    };
    target.adopt_parked_state(caches, window, history);
    target.dflash2_states_mut().store.touch(id);
    u32::try_from(depth).map_err(|_| Error::from_reason("DFlash2 parked state depth overflow"))
}

/// Where a DFlash2 turn of `tokens` resumes: the deepest parked state that
/// beats the live window of `live` tokens, installed, else the live window
/// (Splash `Cache::lookup` then `QwenStateStorage::restore`). Returns the
/// parked state's id beside the hit.
pub(crate) fn resolve_dflash2_resume<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    reuse_cache: bool,
    cache_salt: u64,
    seq: u32,
    live: usize,
) -> Result<(DFlash2Hit, Option<EntryId>)> {
    target.dflash2_turn_state_clear();
    let Some(hit) = parked_state_hit(&*target, tokens, reuse_cache, cache_salt, live) else {
        return Ok((DFlash2Hit::Live(u32::try_from(live).unwrap_or(0)), None));
    };
    let depth = install_parked_state(target, hit.id, seq)?;
    Ok((DFlash2Hit::Stored(depth), Some(hit.id)))
}

/// A turn that resumed parked state `stored` at `cached`, the prompt's own
/// replay point, uses it until the turn ends (Splash
/// `Engine::completeAdmission` → `StateCache::useState`).
pub(crate) fn use_resumed_replay_point<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    stored: Option<EntryId>,
    cached: usize,
) {
    let Some(id) = stored else {
        return;
    };
    let Some(block_size) = target
        .paged_adapter_ref()
        .map(PagedKVCacheAdapter::block_size)
    else {
        return;
    };
    let states = target.dflash2_states_mut();
    let replay = crate::engine::gen_prompt::prompt_replay_boundary(
        tokens,
        states.generation_suffix.as_deref(),
        block_size as usize,
    );
    if cached > 0
        && cached == replay
        && states
            .store
            .get(id)
            .is_some_and(|entry| entry.depth() == cached)
    {
        states.use_replay_point(id);
    }
}

/// The row a DFlash2 prefill of `tokens` from `cached` captures: the
/// prompt's replay point (Splash `Engine::promptReplayBoundary`) when the
/// prefill passes it and the state there is exact.
pub(crate) fn replay_capture_row<T: DFlash2Target>(
    target: &T,
    tokens: &[u32],
    cached: usize,
) -> Option<usize> {
    let block_size = target.paged_adapter_ref()?.block_size() as usize;
    let row = crate::engine::gen_prompt::prompt_replay_boundary(
        tokens,
        target.dflash2_states().generation_suffix.as_deref(),
        block_size,
    );
    (row > cached && row >= block_size && target.replay_state_exact()).then_some(row)
}

/// Publishes the prefill's replay-point state at once, marked and in use by
/// the turn (Splash `Engine::publishReachedStateBoundaries` →
/// `StateCache::useState`). Its blocks register only at finalize, which
/// keeps it or drops it.
pub(crate) fn publish_replay_state<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    cache_salt: u64,
    anchor: PagedAnchor,
) {
    let row = anchor.window.logical_len().max(0) as usize;
    let Some(lineage) = tokens.get(..row) else {
        return;
    };
    let states = target.dflash2_states_mut();
    match states
        .store
        .insert(Lineage::text(lineage, cache_salt), anchor)
    {
        Ok((id, new)) => {
            if new {
                states.unregistered = Some(id);
            }
            states.use_replay_point(id);
        }
        Err(error) => tracing::warn!(
            target: "mlx_core::inference",
            "DFlash2 replay-point state not parked: {}",
            error.reason
        ),
    }
    target.sync_gdn_state_marks();
}

/// The turn's end: this turn's published state stays only when its block
/// registered (Splash `Cache::blockAt`), and the turn's replay point is no
/// longer in use.
pub(crate) fn settle_replay_state<T: DFlash2Target>(target: &mut T, registered: bool) {
    let unregistered = target.dflash2_states_mut().unregistered.take();
    if let Some(id) = unregistered {
        let mark = target
            .dflash2_states()
            .store
            .get(id)
            .map(|entry| entry.mark());
        let cached = registered
            && mark.is_some_and(|mark| {
                target
                    .paged_adapter_ref()
                    .is_some_and(|adapter| adapter.prefix_cached(mark))
            });
        if !cached {
            drop_parked_state(target, id);
        }
    }
    target.dflash2_states_mut().end_replay_point();
}

/// The DFlash2 draft context a tapped paged prefill advances chunk by chunk
/// (Splash `DFlashDraft::addContextCommit` over the prefill's captures).
/// With a capture row, it also snapshots the draft window and the GDN cell
/// at that row: the prompt's replay point (Splash
/// `Engine::publishReachedStateBoundaries`).
pub(crate) struct DraftTapSink<'a> {
    draft: &'a DFlash2Model,
    context: DFlash2ContextCache,
    keep_from: usize,
    tap_layers: Vec<usize>,
    capture: Option<usize>,
    window: Option<DFlash2ContextCache>,
    linear: Option<Vec<super::layer_cache::Qwen3_5LayerCache>>,
}

impl<'a> DraftTapSink<'a> {
    /// `tokens` is the whole prompt; the prefill starts at `cached`, where
    /// `context` ends. `capture` is a row in `(cached, tokens.len())`.
    pub(crate) fn new(
        draft: &'a DFlash2Model,
        context: DFlash2ContextCache,
        tokens: &[u32],
        cached: usize,
        capture: Option<usize>,
    ) -> Self {
        let window_rows = draft.config.sliding_window.saturating_sub(1);
        let capture = capture.filter(|&row| row > cached && row < tokens.len());
        Self {
            keep_from: draft_keep_from(cached, tokens.len(), window_rows, capture),
            tap_layers: draft.config.target_layers.clone(),
            draft,
            context,
            capture,
            window: None,
            linear: None,
        }
    }

    /// The draft context, and the state at the capture row when the prefill
    /// reached it whole.
    pub(crate) fn into_parts(self) -> (DFlash2ContextCache, Option<PagedAnchor>) {
        let anchor = match (self.window, self.linear) {
            (Some(window), Some(linear)) => Some(PagedAnchor { linear, window }),
            _ => None,
        };
        (self.context, anchor)
    }
}

impl crate::models::qwen3_5::paged_forward::PrefillTapSink for DraftTapSink<'_> {
    fn tap_layers(&self) -> &[usize] {
        &self.tap_layers
    }

    /// Rows below `keep_from` only advance length and provenance; the rest
    /// are fused and projected in one append.
    fn chunk(&mut self, first_row: usize, chunk: &[u32], taps: &[MxArray]) -> Result<()> {
        let end = first_row + chunk.len();
        if end <= self.keep_from {
            return self.context.record_only(chunk);
        }
        let head = self.keep_from.saturating_sub(first_row);
        if head > 0 {
            self.context.record_only(&chunk[..head])?;
        }
        // Rows at or past `keep_from` survive in the sliding window: fuse +
        // project only that tail (RoPE keys are per-row at the absolute row,
        // so a sliced append is identical).
        let fused = if head == 0 {
            self.draft.fuse_context(taps)?
        } else {
            let tail_taps = taps
                .iter()
                .map(|tap| tap.slice_axis(1, head as i64, chunk.len() as i64))
                .collect::<Result<Vec<_>>>()?;
            self.draft.fuse_context(&tail_taps)?
        };
        let appended = &chunk[head..];
        let projected = self
            .context
            .project(self.draft, &fused, (first_row + head) as i32)?;
        if let Some(row) = self.capture.filter(|&row| row > first_row && row <= end) {
            self.window = Some(self.context.snapshot_window_at(
                i32::try_from(row).unwrap_or(i32::MAX),
                &projected,
                appended,
            )?);
        }
        self.context.append_projected(projected, appended)
    }

    fn eval(&self) -> Result<()> {
        self.context.eval()
    }

    fn capture_row(&self) -> Option<usize> {
        self.capture
    }

    fn capture_linear(&mut self, linear: Vec<super::layer_cache::Qwen3_5LayerCache>) {
        self.linear = Some(linear);
    }
}

/// What a reclaim took, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GrowReclaim {
    Store,
    Ladder,
    History,
    Vision,
}

#[cfg(test)]
thread_local! {
    /// What reclaims on this thread took, in order.
    pub(crate) static GROW_RECLAIM_LOG: std::cell::RefCell<Vec<GrowReclaim>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn grow_reclaim_log() -> Vec<GrowReclaim> {
    GROW_RECLAIM_LOG.with(|log| log.borrow().clone())
}

#[cfg(test)]
pub(crate) fn clear_grow_reclaim_log() {
    GROW_RECLAIM_LOG.with(|log| log.borrow_mut().clear());
}

fn log_reclaim(_taken: GrowReclaim) {
    #[cfg(test)]
    GROW_RECLAIM_LOG.with(|log| log.borrow_mut().push(_taken));
}

/// Runs `step`; while its block growth is refused and `reclaim` frees
/// memory, runs it again. A refused paged step has changed nothing:
/// `record_tokens` allocates before any layer runs.
pub(crate) fn retry_after_reclaim<T, R>(
    target: &mut T,
    mut step: impl FnMut(&mut T) -> Result<R>,
    mut reclaim: impl FnMut(&mut T) -> bool,
) -> Result<R> {
    loop {
        match step(target) {
            Err(error)
                if crate::engine::scheduler::is_paged_allocation_blocked(&error.reason)
                    && reclaim(target) => {}
            other => return other,
        }
    }
}

/// Runs `step`; while its reservation is refused and the take order frees
/// what it lacks (or the need is met in place), runs it again (Splash
/// `Engine::allocate`). `keep` is the prompt the step reserves for: its
/// ladder and history entries are no units, and the marked blocks it hits
/// are out of the count.
pub(crate) fn reclaiming<T: DFlash2Target, R>(
    target: &mut T,
    keep: Keep<'_>,
    mut step: impl FnMut(&mut T) -> Result<R>,
) -> Result<R> {
    let mut met_before = MetBefore::default();
    retry_after_reclaim(
        target,
        |target| {
            if let Some(adapter) = target.paged_adapter_ref() {
                adapter.clear_refusals();
            }
            step(target)
        },
        |target| {
            let Some(blocks) = target
                .paged_adapter_ref()
                .and_then(PagedKVCacheAdapter::take_refusal)
            else {
                return false;
            };
            if met_before.seen(&blocks) {
                return false;
            }
            let freed = free_for_kv(&mut ModelReclaim { target, keep }, KvNeed::in_place(blocks));
            met_before.retry(blocks, freed)
        },
    )
}

/// One serial paged step under [`reclaiming`], keeping no prompt.
pub(crate) fn paged_step_reclaiming<T: DFlash2Target, R>(
    target: &mut T,
    step: impl FnMut(&mut T) -> Result<R>,
) -> Result<R> {
    reclaiming(target, Keep::default(), step)
}

/// Books a prefill chunk's rows under [`reclaiming`] (Splash
/// `Engine::growLane` books each chunk's end through `Engine::allocate`).
/// The prefix the turn reuses is the request's own table, so the reclaim
/// keeps nothing.
pub(crate) fn record_chunk_reclaiming<T: DFlash2Target>(
    target: &mut T,
    chunk: &[u32],
) -> Result<()> {
    reclaiming(target, Keep::default(), |target| {
        target
            .paged_adapter_mut_ref()
            .ok_or_else(|| Error::from_reason("a paged prefill lost its paged adapter"))?
            .record_tokens(chunk)
            .map_err(Error::from_reason)
    })
}

/// The model side of the take order: parked states, ladder and history
/// entries, ordered by their own stamps.
pub(crate) struct ModelReclaim<'a, T: DFlash2Target> {
    pub(crate) target: &'a mut T,
    pub(crate) keep: Keep<'a>,
}

impl<T: DFlash2Target> KvReclaim for ModelReclaim<'_, T> {
    fn at_cap(&self) -> bool {
        self.target
            .paged_adapter_ref()
            .is_none_or(PagedKVCacheAdapter::at_cap)
    }

    fn takeable(&self) -> u32 {
        self.target
            .paged_adapter_ref()
            .map_or(0, PagedKVCacheAdapter::takeable_blocks)
    }

    fn ungrown(&self) -> u32 {
        self.target
            .paged_adapter_ref()
            .map_or(0, PagedKVCacheAdapter::ungrown_blocks)
    }

    fn bytes_per_block(&self) -> u64 {
        self.target
            .paged_adapter_ref()
            .and_then(|adapter| adapter.bytes_per_block().ok())
            .unwrap_or(0)
    }

    fn met(&mut self, need: KvNeed) -> bool {
        self.target
            .paged_adapter_mut_ref()
            .is_some_and(|adapter| adapter.kv_need_met(need.blocks, need.over_takeable))
    }

    fn grow_room(&self) -> u32 {
        self.target
            .paged_adapter_ref()
            .map_or(0, PagedKVCacheAdapter::last_grow_room)
    }

    fn sync_marks(&mut self) {
        self.target.sync_gdn_state_marks();
    }

    fn reach(&self, at_cap: bool, in_use: bool) -> Reach {
        let marked = self
            .target
            .paged_adapter_ref()
            .map_or(0, PagedKVCacheAdapter::marked_blocks);
        Reach {
            blocks: u64::from(marked.saturating_sub(self.target.marked_hit_blocks(self.keep))),
            bytes: if at_cap {
                0
            } else {
                state_reach_bytes(&*self.target, self.keep, in_use)
            },
        }
    }

    fn oldest_state(&self, at_cap: bool, skip: &HashSet<UnitKey>, in_use: bool) -> Option<Unit> {
        oldest_state_unit(&*self.target, at_cap, skip, self.keep, in_use)
    }

    fn take(&mut self, key: &UnitKey) -> bool {
        take_state_unit(self.target, key).is_some()
    }

    fn rebuilt_bytes(&self) -> u64 {
        rebuilt_bytes(&*self.target)
    }

    fn rebuild_one(&mut self) -> bool {
        release_one_rebuilt(&*self.target)
    }
}

/// Bytes of the image features nothing outside the cache holds.
pub(crate) fn rebuilt_bytes<T: DFlash2Target>(target: &T) -> u64 {
    crate::vision::qwen::cache::unheld_vision_bytes(target.vision_cache())
}

/// Drops the least recently used image feature entry nothing holds.
pub(crate) fn release_one_rebuilt<T: DFlash2Target>(target: &T) -> bool {
    let released = crate::vision::qwen::cache::release_one_unheld_vision(target.vision_cache());
    if released {
        log_reclaim(GrowReclaim::Vision);
    }
    released
}

/// What [`take_state_unit`] took.
pub(crate) enum Taken {
    Store,
    Gdn,
}

/// Bytes every parked state and every GDN entry outside `keep` hold; the
/// entries turns use count only with `in_use`.
pub(crate) fn state_reach_bytes<T: DFlash2Target>(target: &T, keep: Keep<'_>, in_use: bool) -> u64 {
    let kept = keep.points(
        target
            .paged_adapter_ref()
            .map(PagedKVCacheAdapter::block_size),
    );
    target
        .dflash2_states()
        .store
        .entries()
        .filter(|entry| {
            (in_use || !entry.in_use()) && !kept.keeps(entry.tokens(), Some(entry.chain()))
        })
        .fold(target.gdn_state_bytes(keep, in_use), |sum, entry| {
            sum.saturating_add(entry.state_bytes())
        })
}

/// The oldest state that meets a KV need. A parked DFlash2 state is a unit
/// of its class (in use by the running turn, or not) unless `keep`
/// restores through it. At the cap it must free a block: its chain holds a
/// cache-only block (the ladder's tip rule) and no ladder rung marks the same
/// block.
pub(crate) fn oldest_state_unit<T: DFlash2Target>(
    target: &T,
    at_cap: bool,
    skip: &HashSet<UnitKey>,
    keep: Keep<'_>,
    in_use: bool,
) -> Option<Unit> {
    let adapter = target.paged_adapter_ref();
    let kept = keep.points(adapter.map(PagedKVCacheAdapter::block_size));
    let tips = at_cap.then(|| {
        adapter
            .map(PagedKVCacheAdapter::tip_marks)
            .unwrap_or_default()
    });
    let rungs = at_cap.then(|| target.ladder_marks());
    let store = target
        .dflash2_states()
        .store
        .oldest_unit(|entry| {
            entry.in_use() == in_use
                && !kept.keeps(entry.tokens(), Some(entry.chain()))
                && !skip.contains(&UnitKey::Store(entry.id()))
                && tips
                    .as_ref()
                    .is_none_or(|tips| tips.contains(&entry.mark()))
                && rungs
                    .as_ref()
                    .is_none_or(|rungs| !rungs.contains(&entry.mark()))
        })
        .map(|(stamp, id)| Unit {
            stamp,
            key: UnitKey::Store(id),
        });
    older_first(store, target.oldest_gdn_state(at_cap, skip, keep, in_use))
}

/// Drops parked state `id` and re-syncs the marks, so its chain becomes
/// dead leaves (Splash `StateCache::evict` frees no page itself).
pub(crate) fn drop_parked_state<T: DFlash2Target>(target: &mut T, id: EntryId) -> bool {
    let states = target.dflash2_states_mut();
    if !states.store.remove(id) {
        return false;
    }
    if states.replay_point == Some(id) {
        states.replay_point = None;
    }
    if states.unregistered == Some(id) {
        states.unregistered = None;
    }
    target.sync_gdn_state_marks();
    true
}

/// Takes one state unit; `None` when it freed nothing.
pub(crate) fn take_state_unit<T: DFlash2Target>(target: &mut T, key: &UnitKey) -> Option<Taken> {
    match key {
        UnitKey::Store(id) => {
            let dropped = drop_parked_state(target, *id);
            if dropped {
                log_reclaim(GrowReclaim::Store);
            }
            dropped.then_some(Taken::Store)
        }
        UnitKey::Ladder { .. } | UnitKey::History => {
            let dropped = target.drop_gdn_state(key);
            if dropped {
                log_reclaim(if matches!(key, UnitKey::History) {
                    GrowReclaim::History
                } else {
                    GrowReclaim::Ladder
                });
            }
            dropped.then_some(Taken::Gdn)
        }
        UnitKey::Owner(_) => None,
    }
}

/// Grows the page pool so it can hand out every block a serial turn's
/// `prompt` takes before the turn's prefix lookup runs: an SSD restore
/// allocates from the pool and cannot grow it, so a runway-sized pool caps
/// the restore. Splash allocates a restored chain's pages through
/// `Engine::allocate` before the restore starts. Blocks the live request
/// holds count as available (the turn continues on them or releases them),
/// and so do the `marked_hit` blocks the prompt hits. The met check grows
/// first and counts cached blocks after (`kv_need_met`); a need it leaves
/// short goes to `reclaim`.
pub(crate) fn grow_paged_pool_for_prompt<T>(
    target: &mut T,
    prompt: &[u32],
    marked_hit: u32,
    adapter: impl Fn(&mut T) -> Option<&mut PagedKVCacheAdapter>,
    mut reclaim: impl FnMut(&mut T, u32) -> bool,
) {
    let Some(pool) = adapter(target) else {
        return;
    };
    let live = pool
        .active_seq_id()
        .and_then(|seq_id| pool.block_table_for(seq_id))
        .map_or(0, |table| table.num_blocks() as u32);
    let needed = u32::try_from((prompt.len() as u64).div_ceil(u64::from(pool.block_size().max(1))))
        .unwrap_or(u32::MAX);
    let blocks = needed.saturating_sub(live).saturating_sub(marked_hit);
    if blocks == 0 || pool.kv_need_met(blocks, false) {
        return;
    }
    reclaim(target, blocks);
}

/// [`grow_paged_pool_for_prompt`] on `target`'s pool, its short need sent
/// through the take order. `keys` and `cache_salt` are the prepare's, so
/// the prompt's own marked hit is read by hash. Runs after the turn parked
/// the live state, so that state is a unit too.
pub(crate) fn ensure_paged_pool_for_prompt<T: DFlash2Target>(
    target: &mut T,
    prompt: &[u32],
    keys: &[Vec<u64>],
    cache_salt: u64,
) {
    let keep = Keep {
        tokens: prompt,
        keys: Some((keys, cache_salt)),
    };
    ensure_paged_pool_keeping(target, prompt, keep);
}

/// [`ensure_paged_pool_for_prompt`] keeping `keep`. A prepare that skips
/// the lookup keeps nothing (Splash keeps the restored chain only when the
/// request does not skip the cache).
pub(crate) fn ensure_paged_pool_keeping<T: DFlash2Target>(
    target: &mut T,
    prompt: &[u32],
    keep: Keep<'_>,
) {
    let marked_hit = target.marked_hit_blocks(keep);
    let mut met_before = MetBefore::default();
    grow_paged_pool_for_prompt(
        target,
        prompt,
        marked_hit,
        |target| target.paged_adapter_mut_ref(),
        |target, blocks| {
            if met_before.seen(&blocks) {
                return false;
            }
            let need = KvNeed {
                blocks,
                over_takeable: false,
                checked: true,
                reach_in_use: false,
                rebuilt: true,
            };
            let freed = free_for_kv(&mut ModelReclaim { target, keep }, need);
            met_before.retry(blocks, freed)
        },
    );
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    /// A kept prompt protects only the points its lookup can restore, its
    /// last token left out: a point at its own end is not kept, keyed or
    /// not; a shorter one is.
    /// Mutation: pass the whole prompt to the token test in `Keep::points`
    /// → red.
    #[test]
    fn kept_points_leave_the_prompts_last_token() {
        let tokens: Vec<u32> = (1..=32).collect();
        let keyless = Keep {
            tokens: &tokens,
            keys: None,
        }
        .points(Some(16));
        assert!(!keyless.keeps(&tokens, None), "its own end");
        assert!(keyless.keeps(&tokens[..16], None));
        let keys = vec![Vec::new(); 2];
        let keyed = Keep {
            tokens: &tokens,
            keys: Some((&keys, 0)),
        };
        let chain = keyed.hit_chain(16).expect("keys");
        assert_eq!(chain.len(), 1);
        let keyed = keyed.points(Some(16));
        assert!(!keyed.keeps(&tokens, None), "a history entry at its end");
        assert!(keyed.keeps(&tokens[..16], None));
        assert!(keyed.keeps(&tokens[..16], Some(&chain)));
    }

    /// A refused paged step retries after each reclaim that frees memory,
    /// until it runs or nothing is freed; other errors never reclaim.
    /// Mutation: return the first result without retrying → red.
    /// Mutation: retry once only → red (the third run never happens).
    #[test]
    fn refused_paged_step_retries_after_reclaim() {
        let refused = || {
            Err::<u32, _>(Error::from_reason(
                "context_length_exceeded: paged decode could not reserve 1 more block(s)",
            ))
        };
        let mut calls = 0u32;
        let out = retry_after_reclaim(
            &mut calls,
            |calls| {
                *calls += 1;
                if *calls == 1 { refused() } else { Ok(7) }
            },
            |_| true,
        );
        assert_eq!((out.ok(), calls), (Some(7), 2));

        let mut calls = 0u32;
        let out = retry_after_reclaim(
            &mut calls,
            |calls| {
                *calls += 1;
                if *calls < 3 { refused() } else { Ok(7) }
            },
            |_| true,
        );
        assert_eq!((out.ok(), calls), (Some(7), 3));

        let mut calls = 0u32;
        let out = retry_after_reclaim(
            &mut calls,
            |calls| {
                *calls += 1;
                refused()
            },
            |_| false,
        );
        assert!(out.is_err() && calls == 1);

        let mut reclaimed = false;
        let out = retry_after_reclaim(
            &mut reclaimed,
            |_| Err::<u32, _>(Error::from_reason("other failure")),
            |reclaimed| {
                *reclaimed = true;
                true
            },
        );
        assert!(out.is_err() && !reclaimed);
    }

    /// Real adapter over a `initial`-block pool growing to `max` blocks of
    /// `block_size` tokens; `None` without Metal.
    fn growable_adapter(
        initial: u32,
        max: u32,
        block_size: u32,
    ) -> Option<crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter> {
        let config = mlx_paged_attn::PagedAttentionConfig {
            block_size,
            gpu_memory_mb: 256,
            head_size: 64,
            num_kv_heads: 1,
            num_layers: 2,
            max_seq_len: Some(max * block_size),
            max_batch_size: Some(1),
        };
        let pool = match mlx_paged_attn::LayerKVPool::new(
            config,
            initial,
            max,
            mlx_paged_attn::metal::MetalDtype::BFloat16,
        ) {
            Ok(pool) => Arc::new(pool),
            Err(err) if crate::test_support::metal_device_absent(&err) => return None,
            Err(err) => panic!("unexpected LayerKVPool::new failure: {err}"),
        };
        let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
            initial, max, block_size,
        )));
        let mut adapter = crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter::new(
            allocator, pool, block_size,
        )
        .expect("growable adapter");
        adapter.set_grow_headroom_probe_override(Some(Arc::new(Ok)));
        Some(adapter)
    }

    /// A serial turn's prompt restores its whole cold chain from a runway
    /// pool: the pool grows for the prompt before the lookup, since a restore
    /// allocates from the pool and cannot grow it.
    /// Mutation: skip `try_grow_pool` in `grow_paged_pool_for_prompt` → red
    /// (16 ≠ 64 restored tokens).
    #[test]
    fn cold_restore_grows_pool_before_prepare() {
        const BLOCK: u32 = 8;
        let tokens: Vec<u32> = (0..64u32).map(|i| 700 + i * 3).collect();
        let fingerprint = mlx_paged_attn::ColdCacheFingerprint::from_components([
            b"qwen3-5-runway-restore".as_slice(),
        ]);
        let root = std::env::temp_dir().join(format!(
            "mlx-runway-cold-restore-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let open = || {
            Arc::new(
                mlx_paged_attn::ColdCacheManager::open_at(root.clone(), 16 * 1024 * 1024, 0, 8)
                    .expect("temp-dir cold cache"),
            )
        };
        let tier = |manager: &Arc<mlx_paged_attn::ColdCacheManager>| {
            crate::transformer::paged_kv_cache_adapter::ColdTierContext {
                manager: Arc::clone(manager),
                fingerprint,
                sidecar_policy: None,
            }
        };
        let Some(mut source) = growable_adapter(8, 8, BLOCK) else {
            return;
        };
        let manager = open();
        source.set_cold_tier(tier(&manager));
        source.reset_for_new_request(0).expect("reset");
        source.allocate_suffix_blocks(64).expect("allocate");
        source.record_tokens(&tokens).expect("record");
        assert_eq!(
            source
                .register_full_blocks_for_reuse(&[], 0)
                .expect("register"),
            8
        );
        assert!(manager.drain(std::time::Duration::from_secs(10)));
        drop(source);
        drop(manager);

        let mut prompt = tokens.clone();
        prompt.extend(1..=8);
        let mut restore = growable_adapter(2, 16, BLOCK).expect("Metal");
        let manager = open();
        restore.set_cold_tier(tier(&manager));
        grow_paged_pool_for_prompt(
            &mut restore,
            &prompt,
            0,
            |adapter| Some(adapter),
            |_, _| false,
        );
        let plan = restore
            .prepare_turn_per_block_with_max_cache_hit_tokens(
                0,
                &prompt,
                prompt.len() as u32,
                true,
                &vec![Vec::new(); prompt.len() / BLOCK as usize],
                0,
                false,
                prompt.len() as u32 - 1,
            )
            .expect("prepare");
        assert_eq!(plan.cached_prefix_len, 64);
        drop(restore);
        drop(manager);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// An extent-growth pool grows for a serial prompt that its free blocks
    /// cannot hold before cached prefix blocks are counted: the cached block
    /// survives. Mutation: count reclaimable blocks as available for an
    /// extent pool in `kv_need_met` → red (pool 4 ≠ 6).
    #[test]
    fn prompt_grow_keeps_cached_blocks_on_an_extent_pool() {
        const BLOCK: u32 = 8;
        let Some(mut adapter) = growable_adapter(4, 16, BLOCK) else {
            return;
        };
        adapter.set_extent_growth(2);
        adapter.reset_for_new_request(0).expect("reset");
        adapter.allocate_suffix_blocks(8).expect("allocate");
        adapter
            .record_tokens(&[1, 2, 3, 4, 5, 6, 7, 8])
            .expect("record");
        assert_eq!(
            adapter
                .register_full_blocks_for_reuse(&[], 0)
                .expect("register"),
            1
        );
        adapter.release_request().expect("release");
        let prompt: Vec<u32> = (100..132).collect();
        grow_paged_pool_for_prompt(
            &mut adapter,
            &prompt,
            0,
            |adapter| Some(adapter),
            |_, _| false,
        );
        let blocks = adapter.block_telemetry().expect("telemetry");
        assert_eq!(blocks.total_blocks, 6);
        assert_eq!(blocks.reclaimable_blocks, 1);
    }

    /// A declined grow on an extent pool whose free plus cached blocks hold
    /// the prompt reclaims nothing: the met check counts them.
    /// Mutation: drop the `takeable >= blocks` arm in `kv_need_met` → red
    /// (the reclaim runs).
    #[test]
    fn prompt_grow_counts_held_and_cached_blocks_before_states() {
        const BLOCK: u32 = 8;
        let Some(mut adapter) = growable_adapter(4, 16, BLOCK) else {
            return;
        };
        adapter.set_extent_growth(2);
        adapter.set_grow_headroom_probe_override(Some(Arc::new(|_| Ok(4))));
        adapter.reset_for_new_request(0).expect("reset");
        adapter.allocate_suffix_blocks(8).expect("allocate");
        adapter
            .record_tokens(&[1, 2, 3, 4, 5, 6, 7, 8])
            .expect("record");
        adapter
            .register_full_blocks_for_reuse(&[], 0)
            .expect("register");
        adapter.release_request().expect("release");
        let prompt: Vec<u32> = (100..132).collect();
        let mut reclaimed = false;
        grow_paged_pool_for_prompt(
            &mut adapter,
            &prompt,
            0,
            |adapter| Some(adapter),
            |_, _| {
                reclaimed = true;
                false
            },
        );
        assert!(!reclaimed);
        let blocks = adapter.block_telemetry().expect("telemetry");
        assert_eq!(blocks.total_blocks, 4);
    }

    /// Without an armed capture the kept tail is the newest window of the
    /// prompt (today's schedule); a capture past the resume point keeps the
    /// window below it, clamped to the resume point when the gap is shorter
    /// than the window; a capture at or before the resume point, or past the
    /// prompt's end, keeps today's schedule.
    #[test]
    fn keep_from_follows_the_armed_capture() {
        const W: usize = 2047;
        for (resume, end, capture, want, case) in [
            (0, 5000, None, 2953, "no capture"),
            (0, 100, None, 0, "no capture, short prompt"),
            (4000, 5000, None, 4000, "no capture, short tail"),
            (3000, 6000, Some(4000), 3000, "gap < window"),
            (0, 6000, Some(4000), 1953, "gap >= window, cold"),
            (1000, 6000, Some(4000), 1953, "gap >= window, warm"),
            (2000, 6000, Some(4047), 2000, "gap == window"),
            (0, 6000, Some(100), 0, "capture inside the first window"),
            (0, 6000, Some(6000), 3953, "capture at the prompt end"),
            (4000, 9000, Some(4000), 6953, "capture at the resume point"),
            (
                4000,
                9000,
                Some(100),
                6953,
                "capture before the resume point",
            ),
            (0, 6000, Some(7000), 3953, "capture past the prompt end"),
        ] {
            assert_eq!(draft_keep_from(resume, end, W, capture), want, "{case}");
        }
    }
}
