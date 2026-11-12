//! DFlash2 pieces shared by the dense and MoE Qwen3.5 targets: the draft
//! window a tapped paged prefill advances, and the take order a refused
//! paged step reclaims through. The speculative steppers stay per family.

use std::sync::Arc;

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::engine::backend::ChatBackend;
use crate::engine::governor::allocate::{
    Admission, Denial, KvAdmission, Refusal, StateAdmission, UnitSources, Verdict, allocate,
    capacity_exhausted, judge, page_shortfall,
};
use crate::engine::governor::governor as governor_fn;
use crate::engine::governor::governor::cancelled;
use crate::engine::governor::governor::{ClaimRef, Governor, ModelGuard, whole_turn_spec};
use crate::engine::governor::ledger::{
    CertainReuse, ClaimClass, ClaimSpec, FreeOutcome, NeedTiming, ReachClass, Reserves, Stage,
    Taken as LedgerTaken, UnitKey, Victim, VictimQuery, VictimSource, WantedChains,
};
use crate::engine::governor::wait::{
    RESOURCE_WAIT_SECONDS, ResourceWait, WaitReason, Waiter, defer_resource_retry,
    resource_deadline, resource_timeout, sync_claims,
};
use crate::engine::governor::{is_gpu_out_of_memory, metal_out_of_memory};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

use super::dflash2::{DFlash2ContextCache, DFlash2Model};
use super::dflash2_state::paged::{DFlash2States, PagedAnchor};
use super::dflash2_state::store::{EntryId, Lineage, StateBytes};

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

    /// Drops a ladder or history entry; false when it is gone.
    fn drop_gdn_state(&mut self, key: &UnitKey) -> bool;
    /// Marks every ladder rung's and parked DFlash2 state's block.
    fn sync_gdn_state_marks(&self);
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
    /// The ladder and history entries, flagged when `keep` or `wanted`
    /// restores through them.
    fn gdn_state_units(&self, keep: Keep<'_>, wanted: &WantedChains) -> Vec<GdnStateUnit>;
    /// The model's registration with the memory governor, once its paged
    /// pool loaded.
    fn memory_registration(&self) -> Option<&ModelGuard<'static>>;
    /// The serial text decode this thread runs, whose lane a host refusal
    /// suspends in place; `None` elsewhere (a prefill, a scheduled lane,
    /// an image lineage, an MTP turn).
    fn serial_decode(&self) -> Option<SerialDecode>;
    fn set_serial_decode(&mut self, decode: Option<SerialDecode>);
    /// Splash `Runtime::suspend` once the lane's sequence is released: its
    /// live caches and the lane a DFlash2 verify lent (both GDN cells, the
    /// pending rows, the draft window) go, and its use of a GDN state ends.
    fn drop_serial_lane_state(&mut self);
    /// Splash `replayTokens`: the GDN prime and the prefill of `tokens`
    /// past `plan`'s restored prefix, output off, as a turn start with that
    /// prefix prepares them; a DFlash2 lane's prefill is tapped, its draft
    /// window left in the turn state. `stored` is the parked state the
    /// lookup installed.
    fn replay_serial_lane(
        &mut self,
        tokens: &[u32],
        cache_salt: u64,
        plan: &crate::transformer::paged_kv_cache_adapter::PagedTurnPlan,
        decode: SerialDecode,
        stored: Option<EntryId>,
    ) -> Result<()>;
}

/// The serial decode a whole turn runs: what its suspended step books
/// again at resume (Splash `Engine::workEndOf` for a decode command).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SerialDecode {
    Autoregressive,
    DFlash2,
}

impl SerialDecode {
    fn rows(self) -> u32 {
        match self {
            Self::Autoregressive => 1,
            Self::DFlash2 => super::dflash2::DFLASH2_VERIFY_ROWS as u32,
        }
    }
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
/// `StateCache::release`). Returns the resume depth; None when the copy's
/// optional claim was refused, the lane untouched. `prompt` is the turn's,
/// whose states the claim keeps.
pub(crate) fn install_parked_state<T: DFlash2Target>(
    target: &mut T,
    id: EntryId,
    seq: u32,
    prompt: &[u32],
) -> Result<Option<u32>> {
    let (mark, depth, bytes) = {
        let entry = target
            .dflash2_states()
            .store
            .get(id)
            .ok_or_else(|| Error::from_reason(format!("DFlash2 parked state {id} is gone")))?;
        (entry.mark(), entry.depth(), entry.state_bytes())
    };
    if !target
        .paged_adapter_ref()
        .is_some_and(|adapter| adapter.prefix_cached(mark))
    {
        return Err(Error::from_reason(
            "DFlash2 state outlived its target KV block",
        ));
    }
    let keep = Keep {
        tokens: prompt,
        keys: None,
    };
    let installed = publish_claimed(target, Publication::Optional, bytes, keep, |target| {
        if let Some(adapter) = target.paged_adapter_mut_ref() {
            adapter
                .release_request_for(seq)
                .map_err(Error::from_reason)?;
        }
        let mut caches = target.fresh_layer_caches();
        let (window, history) = {
            let entry =
                target.dflash2_states().store.get(id).ok_or_else(|| {
                    Error::from_reason(format!("DFlash2 parked state {id} is gone"))
                })?;
            super::dflash2_state::paged::install_linear(&mut caches, &entry.state().linear)?;
            (entry.state().window.deep_copy()?, entry.tokens().to_vec())
        };
        target.adopt_parked_state(caches, window, history);
        target.dflash2_states_mut().store.touch(id);
        Ok(())
    });
    match installed {
        None => Ok(None),
        Some(installed) => installed.and_then(|()| {
            u32::try_from(depth)
                .map(Some)
                .map_err(|_| Error::from_reason("DFlash2 parked state depth overflow"))
        }),
    }
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
    match install_parked_state(target, hit.id, seq, tokens)? {
        Some(depth) => Ok((DFlash2Hit::Stored(depth), Some(hit.id))),
        None => Ok((DFlash2Hit::Live(u32::try_from(live).unwrap_or(0)), None)),
    }
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
    let bytes = anchor.state_bytes();
    let keep = Keep { tokens, keys: None };
    publish_claimed(target, Publication::ReplayPoint, bytes, keep, |target| {
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
    });
    target.sync_gdn_state_marks();
}

/// What a state publication claims (Splash `Engine::stateClaim`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Publication {
    /// The prompt's replay point (`StateClass::InUse`): in service, it
    /// waits, and it reaches the states turns use.
    ReplayPoint,
    /// A parked state's install, a ladder rung or the history checkpoint:
    /// optional and immediate.
    Optional,
}

/// Runs `publish` while `bytes` are admitted for it under a claim of the
/// thread scope's request (Splash `Engine::stateClaim`), `keep`'s states
/// kept; a refusal skips the publication (None), counted, and the turn
/// goes on. A model the governor does not know publishes unclaimed.
pub(crate) fn publish_claimed<T: DFlash2Target, R>(
    target: &mut T,
    publication: Publication,
    bytes: u64,
    keep: Keep<'_>,
    publish: impl FnOnce(&mut T) -> R,
) -> Option<R> {
    let Some((governor, model)) = target
        .memory_registration()
        .map(|guard| (guard.governor(), guard.model_id()))
    else {
        return Some(publish(target));
    };
    if bytes == 0 {
        return Some(publish(target));
    }
    let request = governor
        .scope_claim()
        .and_then(|id| governor.spec(id))
        .map_or((0, Default::default()), |spec| (spec.request_id, spec.rank));
    let spec = match publication {
        Publication::ReplayPoint => ClaimSpec {
            cls: ClaimClass::InService,
            request_id: request.0,
            rank: request.1,
            reach: ReachClass::InUse,
            timing: NeedTiming::CanWait,
        },
        Publication::Optional => ClaimSpec {
            cls: ClaimClass::Optional,
            request_id: request.0,
            rank: request.1,
            reach: ReachClass::Ordinary,
            timing: NeedTiming::Immediate,
        },
    };
    let claim = governor.open(spec, model);
    let allocation = allocate(
        claim.claim(),
        spec,
        Refusal::Settled,
        false,
        &mut ModelUnits {
            target: &mut *target,
            keep,
        },
        |_| StateAdmission(governor.try_reserve(bytes)),
    );
    let Ok(reservation) = allocation.admission.0 else {
        governor.publication_refused(model);
        return None;
    };
    let published = publish(target);
    drop(reservation);
    Some(published)
}

/// The image features nothing holds go (a reset after a GPU command
/// exhausted Metal memory).
pub(crate) fn drop_unheld_vision<T: DFlash2Target>(target: &T) {
    let cache = target.vision_cache();
    for (generation, _) in crate::vision::qwen::cache::unheld_vision_units(cache) {
        crate::vision::qwen::cache::release_unheld_vision(cache, generation);
    }
}

/// Splash `MemoryControl::run` between a whole turn's steps: the
/// governor's control pass over this model's units, `keep` kept.
fn run_control_pass<T: DFlash2Target>(
    governor: &Governor,
    model: u64,
    forced: bool,
    target: &mut T,
    keep: Keep<'_>,
) {
    if model == 0 || !governor.control_due(model, forced) {
        return;
    }
    let mut units = ModelUnits { target, keep };
    if governor.control_pass(model, &mut [&mut units]).again() {
        governor.control_again(model);
    }
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
thread_local! {
    /// Each granted resume's booked tokens and its KV target.
    static RESUME_BOOKED: std::cell::RefCell<Vec<(u32, u32)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn take_resume_booked() -> Vec<(u32, u32)> {
    RESUME_BOOKED.with(|log| std::mem::take(&mut *log.borrow_mut()))
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

/// A ladder or history entry as a unit of the take order.
pub(crate) struct GdnStateUnit {
    pub key: UnitKey,
    pub stamp: u64,
    pub bytes: u64,
    /// The block it marks; none for the history entry.
    pub mark: Option<u64>,
    pub in_use: bool,
    /// The claim's own prompt restores through it.
    pub kept: bool,
    /// A request waiting for a lane restores through it.
    pub wanted: bool,
}

/// The claim an allocation of `target` runs under: the thread scope's, or
/// a transient InService claim of the engine's own; its spec and model.
fn turn_claim<T: DFlash2Target>(
    target: &T,
) -> (
    &'static Governor,
    Option<crate::engine::governor::governor::ClaimHandle<'static>>,
    ClaimRef<'static>,
    ClaimSpec,
    u64,
) {
    let (governor, model) = target.memory_registration().map_or_else(
        || (governor_fn(), 0),
        |guard| (guard.governor(), guard.model_id()),
    );
    let scoped = governor.scope_claim().and_then(|id| {
        governor
            .spec(id)
            .map(|spec| (ClaimRef { governor, id }, spec))
    });
    let (transient, claim, spec) = match scoped {
        Some((claim, spec)) => (None, claim, spec),
        None => {
            let spec = whole_turn_spec(0);
            let transient = governor.open(spec, model);
            let claim = transient.claim();
            (Some(transient), claim, spec)
        }
    };
    let model = governor.ledger(|ledger| ledger.claim(claim.id).map_or(model, |state| state.model));
    (governor, transient, claim, spec, model)
}

/// A whole turn's claim while it waits (Splash `Engine::syncClaims` for
/// that request): it holds what lands, or counts in the host's deficit,
/// only while the wait lasts.
struct TurnWait<'g> {
    governor: &'g Governor,
    claim: crate::engine::governor::ledger::ClaimId,
}

impl TurnWait<'_> {
    fn sync(&self, wait: &mut ResourceWait, now: f64, epoch: u64) {
        sync_claims(
            self.governor,
            &mut [Waiter {
                wait,
                claim: Some(self.claim),
                submitted: 0,
                finalized: false,
                retried: true,
            }],
            now,
            epoch,
        );
    }

    /// Sleeps to the wait's retry time: the turn's claim waited.
    fn sleep(&self, wait: &ResourceWait, now: f64) {
        self.governor.sleep((wait.retry_at - now).max(0.0));
        crate::engine::governor::metrics::whole_turn_claim_waited(
            (self.governor.now() - now).max(0.0),
        );
    }
}

impl Drop for TurnWait<'_> {
    fn drop(&mut self) {
        self.governor.ledger(|ledger| {
            ledger.set_waiting(self.claim, false);
            ledger.set_host_refused_waiting(self.claim, false);
        });
    }
}

/// Runs `step` as an allocation of the thread scope's claim, or of the
/// engine's own InService claim (Splash `Engine::allocate`): a refused
/// block reservation asks the ledger to free what it lacks over this
/// model's units and runs again. A refused paged step has changed nothing:
/// `record_tokens` allocates before any layer runs. `keep` is the prompt
/// the step reserves for: its ladder and history entries and parked
/// states are no units. Under a scheduled lane's admission the refusal is
/// left for the scheduler's judge. A lone lane's judge (Splash
/// `Engine::judge`): a claim that waits for what lands retries every
/// 100 ms up to the resource wait limit; one the host refused yields, which
/// suspends a serial text decode in place ([`suspend_in_place`]) and makes
/// any other step wait the same way; any other refusal ends the turn
/// `capacity_exhausted`.
pub(crate) fn claimed<T: DFlash2Target, R>(
    target: &mut T,
    keep: Keep<'_>,
    mut step: impl FnMut(&mut T) -> Result<R>,
) -> Result<R> {
    let (governor, _transient, claim, spec, model) = turn_claim(&*target);
    // A stale mark would keep its chain out of both the attempt and the
    // count.
    target.sync_gdn_state_marks();
    let mut wait = ResourceWait::default();
    let turn_wait = TurnWait {
        governor,
        claim: claim.id,
    };
    run_control_pass(governor, model, false, target, keep);
    loop {
        let mut units = ModelUnits {
            target: &mut *target,
            keep,
        };
        let allocation = allocate(
            claim,
            spec,
            Refusal::Waits,
            wait.hold_ended,
            &mut units,
            |units| {
                if let Some(adapter) = units.target.paged_adapter_ref() {
                    adapter.clear_refusals();
                }
                let result = step(&mut *units.target);
                let refusal = match &result {
                    Err(error)
                        if crate::engine::scheduler::is_paged_allocation_blocked(&error.reason) =>
                    {
                        units
                            .target
                            .paged_adapter_ref()
                            .and_then(PagedKVCacheAdapter::take_block_refusal)
                    }
                    _ => None,
                };
                KvAdmission {
                    result,
                    refusal,
                    failure: governor.failure(claim.id),
                }
            },
        );
        let denial = allocation.denial;
        let admission = allocation.admission;
        if let Err(error) = &admission.result
            && is_gpu_out_of_memory(&error.reason)
        {
            governor.note_gpu_oom(model);
            return Err(Error::from_reason(metal_out_of_memory(&error.reason)));
        }
        if admission.result.is_ok() || !admission.memory_denied() {
            return admission.result;
        }
        if crate::engine::governor::governor::scheduler_judges() {
            crate::engine::governor::governor::note_lane_denial(denial);
            return admission.result;
        }
        let verdict = judge(&denial, false);
        if verdict == Verdict::Fail {
            governor.capacity_exhausted();
            return Err(Error::from_reason(capacity_exhausted(
                "KV target",
                denial.failure,
                &page_shortfall(denial.blocks),
            )));
        }
        if verdict == Verdict::Yield
            && let Some(decode) = target.serial_decode()
        {
            suspend_in_place(target, claim, spec, model, decode, &denial)?;
            wait = ResourceWait::default();
            continue;
        }
        let now = governor.now();
        let epoch = governor.epoch(model);
        defer_resource_retry(&mut wait, now, &denial, WaitReason::Memory, true, 0, epoch);
        wait.hold_ended = wait
            .pending_since
            .is_some_and(|since| now >= since + RESOURCE_WAIT_SECONDS);
        // A cancel or stop ends the wait before its limit can (Splash
        // `Engine::settle` keeps the first end).
        if cancelled() {
            if target.serial_decode().is_some() {
                vacate_serial_lane(target)?;
                return Err(Error::from_reason(
                    crate::engine::session::LANE_VACATED_CANCELLED,
                ));
            }
            return Err(Error::from_reason(
                crate::engine::session::CHAT_SESSION_CANCELLED,
            ));
        }
        let deadline = resource_deadline(&wait, epoch);
        if deadline > 0.0 && now >= deadline {
            governor.resource_timed_out();
            return Err(Error::from_reason(resource_timeout(denial.failure)));
        }
        turn_wait.sync(&mut wait, now, epoch);
        turn_wait.sleep(&wait, now);
        run_control_pass(governor, model, true, target, keep);
    }
}

/// A speculative lookahead's rows (Splash `Engine::growLane` books a lane's
/// verify rows in its claim, InService and Immediate): the ledger frees
/// what meets them at once, and a refusal, or a lookahead past the
/// window, returns false: the turn decodes autoregressively, and that
/// step's claim decides. No wait, no judge.
pub(crate) fn lookahead_claimed<T: DFlash2Target>(
    target: &mut T,
    mut reserve: impl FnMut(&mut T) -> Result<()>,
) -> Result<bool> {
    let (governor, _transient, claim, held, model) = turn_claim(&*target);
    let spec = ClaimSpec {
        cls: ClaimClass::InService,
        timing: NeedTiming::Immediate,
        ..held
    };
    target.sync_gdn_state_marks();
    let allocation = allocate(
        claim,
        spec,
        Refusal::Settled,
        false,
        &mut ModelUnits {
            target: &mut *target,
            keep: Keep::default(),
        },
        |units| {
            if let Some(adapter) = units.target.paged_adapter_ref() {
                adapter.clear_refusals();
            }
            let result = reserve(&mut *units.target);
            let refusal = match &result {
                Err(error)
                    if crate::engine::scheduler::is_paged_allocation_blocked(&error.reason) =>
                {
                    units
                        .target
                        .paged_adapter_ref()
                        .and_then(PagedKVCacheAdapter::take_block_refusal)
                }
                _ => None,
            };
            KvAdmission {
                result,
                refusal,
                failure: governor.failure(claim.id),
            }
        },
    );
    governor.ledger(|ledger| ledger.update(claim.id, held));
    let denied = allocation.admission.memory_denied();
    match allocation.admission.result {
        Ok(()) => Ok(true),
        Err(error) if is_gpu_out_of_memory(&error.reason) => {
            governor.note_gpu_oom(model);
            Err(Error::from_reason(metal_out_of_memory(&error.reason)))
        }
        Err(_) if denied => Ok(false),
        Err(error) if error.reason.starts_with("context_length_exceeded:") => Ok(false),
        Err(error) => Err(error),
    }
}

/// Splash `NativeRuntime::executionFailed`: MLX reports a failed command
/// at a later wait, so a whole turn's GPU OOM can surface outside every
/// claim. At the turn's boundary it arms the model's recovery and fails
/// the turn `metal_out_of_memory`.
pub(crate) fn whole_turn_oom<T: DFlash2Target>(target: &T, error: Error) -> Error {
    if error.reason.starts_with("metal_out_of_memory:") || !is_gpu_out_of_memory(&error.reason) {
        return error;
    }
    if let Some(guard) = target.memory_registration() {
        guard.governor().note_gpu_oom(guard.model_id());
    }
    Error::from_reason(metal_out_of_memory(&error.reason))
}

/// One serial paged decode step under [`claimed`], keeping no prompt: a
/// token event of the whole turn this thread runs.
pub(crate) fn paged_step_claimed<T: DFlash2Target, R>(
    target: &mut T,
    step: impl FnMut(&mut T) -> Result<R>,
) -> Result<R> {
    let result = claimed(target, Keep::default(), step);
    if let Some(guard) = target.memory_registration() {
        crate::engine::governor::metrics::whole_turn_token(guard.governor());
    }
    result
}

/// Splash `Engine::suspendForGrowth` → `suspendLane` for the serial lane:
/// the lane is vacated, waits on the model thread until the host lets it
/// back, then takes its KV target at once and replays its tokens (Splash
/// `admit` of a suspended request, `replayTokens`). The driver's locals are
/// the request's continuation: nothing goes to the client meanwhile, and
/// the denied step runs again after the replay.
fn suspend_in_place<T: DFlash2Target>(
    target: &mut T,
    claim: ClaimRef<'_>,
    spec: ClaimSpec,
    model: u64,
    decode: SerialDecode,
    denial: &Denial,
) -> Result<()> {
    let governor = claim.governor;
    let (tokens, cache_salt) = vacate_serial_lane(target)?;
    let resume_target = u32::try_from(tokens.len())
        .unwrap_or(u32::MAX)
        .saturating_add(decode.rows());
    let now = governor.now();
    governor.lane_suspended(model, now, denial.failure);
    let mut wait = ResourceWait::default();
    defer_resource_retry(
        &mut wait,
        now,
        &Denial {
            failure: denial.failure,
            ..Denial::default()
        },
        WaitReason::Memory,
        true,
        0,
        governor.epoch(model),
    );
    // The replay's own claims wait: a lane is suspended once.
    target.set_serial_decode(None);
    let lane = SuspendedLane {
        tokens: &tokens,
        cache_salt,
        decode,
        resume_target,
    };
    let resumed = wait_and_resume(target, claim, spec, model, &lane, &mut wait);
    governor.end_drain(model);
    if resumed.is_ok() {
        target.set_serial_decode(Some(decode));
        governor.lane_resumed(model);
    }
    resumed
}

struct SuspendedLane<'a> {
    tokens: &'a [u32],
    cache_salt: u64,
    decode: SerialDecode,
    resume_target: u32,
}

/// The serial lane's sequence.
const SERIAL_SEQ: u32 = 0;

/// Splash `Engine::vacateLane`: the lane's full blocks register as a cached
/// chain nothing pins, its sequence is released, its live GDN cell and
/// draft window go. Returns the tokens it fed and their salt.
fn vacate_serial_lane<T: DFlash2Target>(target: &mut T) -> Result<(Vec<u32>, u64)> {
    let adapter = target
        .paged_adapter_mut_ref()
        .ok_or_else(|| Error::from_reason("a serial lane lost its paged adapter"))?;
    let tokens = adapter
        .request_tokens_for(SERIAL_SEQ)
        .ok_or_else(|| Error::from_reason("a serial lane has no request to suspend"))?
        .to_vec();
    let cache_salt = adapter.request_cache_salt_for(SERIAL_SEQ).unwrap_or(0);
    adapter
        .register_full_blocks_for_reuse_for(SERIAL_SEQ, &[], cache_salt, true)
        .map_err(Error::from_reason)?;
    adapter
        .release_request_for(SERIAL_SEQ)
        .map_err(Error::from_reason)?;
    target.drop_serial_lane_state();
    settle_replay_state(target, true);
    target.sync_gdn_state_marks();
    Ok((tokens, cache_salt))
}

/// The suspended lane's wait (Splash `deferResourceRetry` + `admitQueued`):
/// every 100 ms, on the model thread, the turn's cancel and the wait limit
/// are checked, the control pass runs and the resume is tried; the host's
/// refusal and memory still on its way keep it waiting.
fn wait_and_resume<T: DFlash2Target>(
    target: &mut T,
    claim: ClaimRef<'_>,
    spec: ClaimSpec,
    model: u64,
    lane: &SuspendedLane<'_>,
    wait: &mut ResourceWait,
) -> Result<()> {
    let governor = claim.governor;
    let keep = Keep {
        tokens: lane.tokens,
        keys: None,
    };
    let turn_wait = TurnWait {
        governor,
        claim: claim.id,
    };
    loop {
        let now = governor.now();
        let epoch = governor.epoch(model);
        wait.hold_ended = wait
            .pending_since
            .is_some_and(|since| now >= since + RESOURCE_WAIT_SECONDS);
        if cancelled() {
            return Err(Error::from_reason(
                crate::engine::session::LANE_VACATED_CANCELLED,
            ));
        }
        let deadline = resource_deadline(wait, epoch);
        if deadline > 0.0 && now >= deadline {
            governor.resource_timed_out();
            return Err(Error::from_reason(resource_timeout(wait.failure)));
        }
        turn_wait.sync(wait, now, epoch);
        turn_wait.sleep(wait, now);
        run_control_pass(governor, model, true, target, keep);
        let (hit, stored) = match lane.decode {
            SerialDecode::DFlash2 => {
                let (hit, stored) = resolve_dflash2_resume(
                    target,
                    lane.tokens,
                    true,
                    lane.cache_salt,
                    SERIAL_SEQ,
                    0,
                )?;
                (Some(hit), stored)
            }
            SerialDecode::Autoregressive => (None, None),
        };
        let allocation = allocate(
            claim,
            spec,
            Refusal::Waits,
            wait.hold_ended,
            &mut ModelUnits {
                target: &mut *target,
                keep,
            },
            |units| claim_resume(units.target, lane, hit, claim),
        );
        let denied = allocation.admission.memory_denied();
        match allocation.admission.result {
            Ok(plan) => {
                return target
                    .replay_serial_lane(lane.tokens, lane.cache_salt, &plan, lane.decode, stored)
                    .map_err(|error| {
                        if is_gpu_out_of_memory(&error.reason) {
                            governor.note_gpu_oom(model);
                            Error::from_reason(metal_out_of_memory(&error.reason))
                        } else {
                            error
                        }
                    });
            }
            Err(error) if !denied => return Err(error),
            Err(_) => {}
        }
        let denial = allocation.denial;
        if judge(&denial, false) == Verdict::Fail {
            governor.capacity_exhausted();
            return Err(Error::from_reason(capacity_exhausted(
                "KV target",
                denial.failure,
                &page_shortfall(denial.blocks),
            )));
        }
        defer_resource_retry(
            wait,
            governor.now(),
            &denial,
            WaitReason::Memory,
            false,
            0,
            governor.epoch(model),
        );
    }
}

/// The resume claim's attempt (Splash `Engine::admit` of a suspended lane):
/// the chain the lookup restores and the lane's KV target, booked at once.
/// A refusal leaves no request behind.
fn claim_resume<T: DFlash2Target>(
    target: &mut T,
    lane: &SuspendedLane<'_>,
    hit: Option<DFlash2Hit>,
    claim: ClaimRef<'_>,
) -> KvAdmission<crate::transformer::paged_kv_cache_adapter::PagedTurnPlan> {
    let Some(adapter) = target.paged_adapter_mut_ref() else {
        return KvAdmission {
            result: Err(Error::from_reason("a serial lane lost its paged adapter")),
            refusal: None,
            failure: None,
        };
    };
    adapter.clear_refusals();
    let tokens = lane.tokens;
    let len = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
    let keys = crate::engine::build_paged_extra_keys(tokens.len(), adapter.block_size(), &[]);
    let prepared = match hit {
        None => adapter
            .prepare_turn_per_block_with_max_cache_hit_tokens(
                SERIAL_SEQ,
                tokens,
                len,
                true,
                &keys,
                lane.cache_salt,
                false,
                len.saturating_sub(1),
            )
            .map_err(Error::from_reason),
        Some(hit) => prepare_dflash2_turn(adapter, SERIAL_SEQ, tokens, &keys, lane.cache_salt, hit),
    };
    let result = prepared.and_then(|plan| {
        let booked = u32::try_from(adapter.request_tokens().len()).unwrap_or(u32::MAX);
        adapter
            .reserve_rows(lane.resume_target.saturating_sub(booked))
            .map(|_| plan)
            .map_err(Error::from_reason)
    });
    #[cfg(test)]
    if result.is_ok() {
        let booked = (adapter.num_allocated_blocks() as u32).saturating_mul(adapter.block_size());
        RESUME_BOOKED.with(|log| log.borrow_mut().push((booked, lane.resume_target)));
    }
    let refusal = if result.is_err() {
        let refusal = adapter.take_block_refusal();
        let _ = adapter.release_request_for(SERIAL_SEQ);
        refusal
    } else {
        None
    };
    KvAdmission {
        result,
        refusal,
        failure: claim.governor.failure(claim.id),
    }
}

/// Books a prefill chunk's rows under [`claimed`] (Splash
/// `Engine::growLane` books each chunk's end through `Engine::allocate`).
/// The prefix the turn reuses is the request's own table, so the claim
/// keeps nothing.
pub(crate) fn record_chunk_claimed<T: DFlash2Target>(target: &mut T, chunk: &[u32]) -> Result<()> {
    claimed(target, Keep::default(), |target| {
        target
            .paged_adapter_mut_ref()
            .ok_or_else(|| Error::from_reason("a paged prefill lost its paged adapter"))?
            .record_tokens(chunk)
            .map_err(Error::from_reason)
    })
}

/// The model's accounts on the governor's books: its KV pool; its states
/// (parked DFlash2 states, GDN ladder and history entries, and the live
/// scheduled GDN cells, `gdn_cells`); its image features.
pub(crate) fn memory_accounts<T: DFlash2Target>(
    target: &T,
    gdn_cells: u64,
) -> crate::engine::governor::governor::MemoryAccounts {
    let kv = target
        .paged_adapter_ref()
        .and_then(|adapter| adapter.pool_allocated_bytes().ok())
        .unwrap_or(0);
    let gdn = target
        .gdn_state_units(Keep::default(), &WantedChains::default())
        .iter()
        .fold(0u64, |sum, unit| sum.saturating_add(unit.bytes));
    crate::engine::governor::governor::MemoryAccounts {
        kv,
        states: target
            .dflash2_states()
            .store
            .bytes()
            .state_bytes
            .saturating_add(gdn)
            .saturating_add(gdn_cells),
        embeddings: crate::vision::qwen::cache::vision_cache_bytes(target.vision_cache()),
    }
}

/// A model's units on the ledger's order (Splash's cache walk): its pool's
/// empty extents, parked DFlash2 states and GDN ladder and history entries
/// by their stamps (the ones turns use at `InUse`), and the image features
/// nothing holds (`Rebuild`). `keep`'s states are never listed.
pub(crate) struct ModelUnits<'a, T: DFlash2Target> {
    pub(crate) target: &'a mut T,
    pub(crate) keep: Keep<'a>,
}

impl<T: DFlash2Target> UnitSources for ModelUnits<'_, T> {
    fn sources(&mut self) -> Vec<&mut dyn VictimSource> {
        vec![self]
    }
}

impl<T: DFlash2Target> VictimSource for ModelUnits<'_, T> {
    fn units(&self, query: &VictimQuery) -> Vec<Victim> {
        model_units(&*self.target, self.keep, query)
    }

    fn take(&mut self, query: &VictimQuery, victim: &Victim) -> LedgerTaken {
        take_model_unit(self.target, query, victim)
    }
}

/// [`ModelUnits`]' units of `target`. A need the host refused walks the
/// units that meet it in place (Splash `Cache::walkFor`): blocks only
/// (`pagesOnly`), or with no blocks the states only (`statesOnly`), whose
/// buffers the refused allocation reuses from the MLX free pool.
pub(crate) fn model_units<T: DFlash2Target>(
    target: &T,
    keep: Keep<'_>,
    query: &VictimQuery,
) -> Vec<Victim> {
    let host_refused = query.need.host_refused;
    let pages_only = host_refused && query.need.pages > 0;
    let states_only = host_refused && query.need.pages == 0;
    let mut units = Vec::new();
    if !host_refused {
        let bytes = target.paged_adapter_ref().map_or(0, |adapter| {
            adapter.free_extent_bytes(query.rights.reserves == Reserves::None)
        });
        if bytes > 0 {
            units.push(Victim {
                key: UnitKey::Extent,
                stamp: 0,
                bytes,
                pages: 0,
                stage: Stage::FreeNow,
            });
        }
    }
    let kept = keep.points(
        target
            .paged_adapter_ref()
            .map(PagedKVCacheAdapter::block_size),
    );
    let wanted = target
        .memory_registration()
        .map_or_else(Arc::default, |guard| {
            guard.governor().wanted_chains(guard.model_id())
        });
    let state_bytes = |bytes: u64| if pages_only { 0 } else { bytes };
    let stage = |in_use: bool| if in_use { Stage::InUse } else { Stage::FreeNow };
    // Every state, each with the block it marks and whether the claim's
    // prompt or a waiting one restores through it.
    let mut all: Vec<StateUnit> = Vec::new();
    for entry in target.dflash2_states().store.entries() {
        all.push(StateUnit {
            unit: Victim {
                key: UnitKey::Store(entry.id()),
                stamp: entry.last_used(),
                bytes: state_bytes(entry.state_bytes()),
                pages: 0,
                stage: stage(entry.in_use()),
            },
            mark: Some(entry.mark()),
            kept: kept.keeps(entry.tokens(), Some(entry.chain())),
            wanted: wanted.keeps(entry.tokens()),
            ordinary: true,
        });
    }
    for unit in target.gdn_state_units(keep, &wanted) {
        all.push(StateUnit {
            ordinary: unit.key == UnitKey::History,
            unit: Victim {
                key: unit.key,
                stamp: unit.stamp,
                bytes: state_bytes(unit.bytes),
                pages: 0,
                stage: stage(unit.in_use),
            },
            mark: unit.mark,
            kept: unit.kept,
            wanted: unit.wanted,
        });
    }
    let resume_point = resume_point(&all);
    let reuse = query.rights.certain_reuse;
    // States in take order, each with the block it marks.
    let mut states: Vec<(Victim, Option<u64>)> = all
        .into_iter()
        .filter(|state| {
            !state.kept
                && (reuse == CertainReuse::All || !state.wanted)
                && (reuse != CertainReuse::None || resume_point.as_ref() != Some(&state.unit.key))
        })
        .map(|state| (state.unit, state.mark))
        .collect();
    states.sort_by_key(|(unit, _)| (unit.stage, unit.stamp));
    if !states_only && let Some(adapter) = target.paged_adapter_ref() {
        let marked: Vec<usize> = (0..states.len())
            .filter(|&index| states[index].1.is_some())
            .collect();
        let marks: Vec<u64> = marked.iter().filter_map(|&index| states[index].1).collect();
        for (index, pages) in marked.into_iter().zip(adapter.freed_by_marks(&marks)) {
            states[index].0.pages = pages;
        }
    }
    units.extend(states.into_iter().map(|(unit, _)| unit));
    if !host_refused {
        for (generation, bytes) in
            crate::vision::qwen::cache::unheld_vision_units(target.vision_cache())
        {
            units.push(Victim {
                key: UnitKey::Vision(generation),
                stamp: generation,
                bytes,
                pages: 0,
                stage: Stage::Rebuild,
            });
        }
    }
    units.retain(|unit| unit.bytes > 0 || unit.pages > 0);
    units.sort_by_key(|unit| (unit.stage, unit.stamp));
    units
}

/// A state on the take order, before the claim's rights filter it.
struct StateUnit {
    unit: Victim,
    mark: Option<u64>,
    kept: bool,
    wanted: bool,
    /// A parked state or the history entry, not a ladder checkpoint.
    ordinary: bool,
}

/// Splash `StateCache::resumePoint`: the newest state no turn uses, an
/// ordinary one before any checkpoint.
fn resume_point(states: &[StateUnit]) -> Option<UnitKey> {
    let newest = |ordinary: bool| {
        states
            .iter()
            .filter(|state| state.ordinary == ordinary && state.unit.stage != Stage::InUse)
            .max_by_key(|state| state.unit.stamp)
            .map(|state| state.unit.key.clone())
    };
    newest(true).or_else(|| newest(false))
}

/// Takes one of [`model_units`]' units. Credits what the take measurably
/// freed: MLX's active bytes (a state a running turn or a pending graph
/// still holds frees nothing), the pool bytes an extent cut gave back, and
/// the blocks it made takeable. Met whenever the unit went (Splash
/// `Cache::take`), so the ledger lists again: a state that shared its
/// block or its arrays hands them to the next one only once it is gone.
pub(crate) fn take_model_unit<T: DFlash2Target>(
    target: &mut T,
    query: &VictimQuery,
    victim: &Victim,
) -> LedgerTaken {
    let active = crate::array::memory::active_memory_bytes();
    let takeable = target
        .paged_adapter_ref()
        .map_or(0, PagedKVCacheAdapter::takeable_blocks);
    let mut pool_bytes = 0u64;
    let removed = match &victim.key {
        UnitKey::Extent => {
            let keep_one = query.rights.reserves == Reserves::None;
            if let Some(adapter) = target.paged_adapter_mut_ref() {
                pool_bytes = adapter.release_free_extents(victim.bytes, keep_one);
            }
            pool_bytes > 0
        }
        UnitKey::Vision(generation) => {
            let released = crate::vision::qwen::cache::release_unheld_vision(
                target.vision_cache(),
                *generation,
            );
            if released {
                log_reclaim(GrowReclaim::Vision);
            }
            released
        }
        key => take_state_unit(target, key),
    };
    let freed = active.saturating_sub(crate::array::memory::active_memory_bytes());
    let now_bytes = freed.saturating_add(pool_bytes);
    let pages = target
        .paged_adapter_ref()
        .map_or(0, PagedKVCacheAdapter::takeable_blocks)
        .saturating_sub(takeable);
    LedgerTaken {
        outcome: if removed || now_bytes > 0 || pages > 0 {
            FreeOutcome::Met
        } else {
            FreeOutcome::Short
        },
        now_bytes,
        later_bytes: 0,
        pages,
    }
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

/// Takes one state unit; false when it freed nothing.
pub(crate) fn take_state_unit<T: DFlash2Target>(target: &mut T, key: &UnitKey) -> bool {
    match key {
        UnitKey::Store(id) => {
            let dropped = drop_parked_state(target, *id);
            if dropped {
                log_reclaim(GrowReclaim::Store);
            }
            dropped
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
            dropped
        }
        UnitKey::Owner(_) | UnitKey::Extent | UnitKey::Vision(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::engine::governor::governor::Scope;

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

    /// A governor of its own on fake counters, with 32 GiB of host memory
    /// free.
    fn test_governor() -> &'static crate::engine::governor::governor::Governor {
        use crate::engine::governor::governor::tests::{FakeIo, G, host, stats};
        let io = FakeIo::new(
            stats(0),
            host(32 * G, crate::engine::governor::host::Pressure::Normal),
        );
        Box::leak(Box::new(crate::engine::governor::governor::Governor::new(
            io,
        )))
    }

    /// A serial turn's prompt restores its whole cold chain from a runway
    /// pool: under the governor the lookup grows the pool for the blocks it
    /// may restore, an allocation of the turn's claim, since a restore
    /// allocates from the pool and cannot grow it.
    /// Mutation R3: skip the grow before the restore walk → red (16 ≠ 64
    /// restored tokens).
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
        let governor = test_governor();
        let _model = governor
            .register_model(crate::engine::governor::governor::tests::model(1 << 40, 0))
            .expect("register");
        restore.set_admission(Arc::new(governor));
        let claim = governor.open(whole_turn_spec(1), 7);
        let _scope = Scope::enter(claim.claim());
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
