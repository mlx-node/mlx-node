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
    Taken as LedgerTaken, UnitKey, UnitRuns, Victim, VictimQuery, VictimSource, WantedChains,
};
use crate::engine::governor::wait::{
    RESOURCE_WAIT_SECONDS, ResourceWait, WaitReason, Waiter, defer_resource_retry,
    resource_deadline, resource_timeout, sync_claims,
};
use crate::engine::governor::{is_gpu_out_of_memory, metal_out_of_memory};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

use super::dflash2::{DFlash2ContextCache, DFlash2Model};
use super::dflash2_state::disk;
use super::dflash2_state::paged::{
    DFlash2States, PagedAnchor, PrefillYield, TurnStates, anchor_bytes,
};
use super::dflash2_state::plan::{
    CaptureSpan, PlanInputs, StatePlan, checkpoint_interval, command_breaks,
    configure_draft_state_plan, plan_draft_context, replay_state_boundary,
};
use super::dflash2_state::store::{EntryId, Inserted, Lineage, StateClass, lineage_bytes};
use super::draft_context::DraftContextWindow;
use super::paged_forward::{CaptureGrant, Reached};
use crate::engine::governor::governor::Reservation;

/// The draft window a tapped paged prefill built, handed to the turn's
/// stepper.
pub(crate) struct DFlash2TurnState {
    pub(crate) context: DFlash2ContextCache,
    pub(crate) next_position: i32,
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
    /// The sequence the lane running now holds: a scheduled lane's own,
    /// else the serial lane's 0 (Splash `Request` per lane).
    fn lane_seq(&self) -> u32;
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
    /// Drops the lane's draft ring: the turn starts from nothing.
    fn clear_draft_context(&mut self);
    /// The live GDN cell, evaluated: a state taken from the lane after the
    /// chunk that ended at it.
    fn snapshot_live_linear(&mut self) -> Result<Vec<super::layer_cache::Qwen3_5LayerCache>>;
    /// Per-block extra keys of the running turn's first `tokens` rows: its
    /// image lineage, none on a text chain.
    fn paged_block_keys(&self, tokens: usize) -> Vec<Vec<u64>>;
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
    /// The thread runs the serial lane on a text lineage: a host refusal
    /// may suspend it (an image lineage's replay needs its features).
    fn serial_text_lane(&self) -> bool;
    /// Splash `Runtime::suspend` before the lane's sequence is released:
    /// its live GDN cell and the cells a DFlash2 verify lent (both, with
    /// their pending rows) go, its draft ring and context window go back to
    /// the idle lane, and its use of a GDN state ends.
    fn drop_serial_lane_state(&mut self);
    /// Splash `replayTokens`: the GDN prime and the prefill of `tokens`
    /// past `plan`'s restored prefix, as a turn start with that prefix
    /// prepares them; a DFlash2 lane's prefill is tapped, its draft window
    /// left in the turn state. `stored` is the parked state the lookup
    /// installed. Returns the last row's logits, evaluated.
    fn replay_serial_lane(
        &mut self,
        tokens: &[u32],
        cache_salt: u64,
        plan: &crate::transformer::paged_kv_cache_adapter::PagedTurnPlan,
        decode: SerialDecode,
        stored: Option<EntryId>,
    ) -> Result<MxArray>;
}

/// A tapped prefill's chunk the host refused: the prefill's owner vacates
/// the lane and replays the prompt (Splash suspends between commands).
pub(crate) const DFLASH2_SUSPENDED: &str = "dflash2_suspended: the serial lane yielded its prefill";

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

/// Where a DFlash2 turn resumes: a parked state installed at its block, or
/// from nothing (Splash resumes a lane only at a state).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DFlash2Hit {
    Stored(u32),
    Cold,
}

/// The DFlash2 prepare. A turn resumes only at its state's exact block
/// (Splash resumes a state at its boundary): a parked state looks its chain
/// up capped at its block. Any other hit restarts cold.
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
        DFlash2Hit::Stored(depth) => {
            adapter.arm_state_restore(depth);
            (false, depth)
        }
        DFlash2Hit::Cold => (true, 0),
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
    if prepared.cached_prefix_len == 0
        || (!prepared.continued_live_prefix && prepared.cached_prefix_len == candidate)
    {
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

/// The deepest parked state a DFlash2 turn of `tokens` resumes from (Splash
/// `StateCache::acquireDeepest`); `None` without reuse.
/// A disk state whose chain cannot come back whole is discarded and the
/// next deepest taken.
pub(crate) fn parked_state_hit<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    reuse_cache: bool,
    cache_salt: u64,
) -> Option<super::dflash2_state::store::StoreHit> {
    if !reuse_cache {
        return None;
    }
    for _ in 0..=target.dflash2_states().store.bytes().entries {
        let hit = target
            .dflash2_states()
            .store
            .lookup(Lineage::text(tokens, cache_salt))?;
        if disk::restorable(&*target, hit.id) {
            return Some(hit);
        }
        disk::discard_state(target, hit.id);
    }
    None
}

/// The deepest state on disk alone an autoregressive prime of `tokens`
/// resumes on, at most `cap` deep: its chain comes back from the tier with
/// the KV lookup. Its object is read and checked before the chain is armed,
/// and the read waits in `state_read` for the GDN prime. One that does not
/// read back, or whose chain cannot come back whole, is discarded and the
/// next deepest taken (Splash `pollRestores` → `discardState`, then a fresh
/// lookup): no KV is restored without its state.
pub(crate) fn disk_state_depth<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    extra_keys: &[Vec<u64>],
    cache_salt: u64,
    cap: u32,
) -> Option<u32> {
    target.dflash2_states_mut().state_read = None;
    for _ in 0..=target.dflash2_states().store.bytes().entries {
        let store = &target.dflash2_states().store;
        let hit = store.lookup_within(
            Lineage {
                tokens,
                extra_keys,
                salt: cache_salt,
            },
            cap as usize,
        )?;
        store.get(hit.id).filter(|entry| entry.ram().is_none())?;
        let depth = u32::try_from(hit.depth).ok()?;
        if disk::restorable(&*target, hit.id)
            && let Some(linear) = disk::read_state(target, hit.id, None)
        {
            target.dflash2_states_mut().state_read =
                Some(super::dflash2_state::paged::StateRead { id: hit.id, linear });
            return Some(depth);
        }
        disk::discard_state(target, hit.id);
    }
    None
}

/// The turn's KV prefix came back with a disk state but no GDN state at its
/// end does: the restored KV goes and the turn prefills its whole prompt
/// from nothing (Splash `pollRestores` vacates the lane after a failed state
/// read). The lookup is skipped, so the restored chain is not hit again
/// without its state. A whole turn books its prompt; a scheduled lane books
/// nothing past its empty hit.
pub(crate) fn restart_turn_cold<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    extra_keys: &[Vec<u64>],
    cache_salt: u64,
) -> Result<()> {
    let seq = target
        .paged_adapter_ref()
        .and_then(PagedKVCacheAdapter::active_seq_id)
        .ok_or_else(|| Error::from_reason("a cold restart without an active request"))?;
    let budget = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
    claimed(target, Keep::default(), |target| {
        let adapter = target
            .paged_adapter_mut_ref()
            .ok_or_else(|| Error::from_reason("a cold restart lost its paged adapter"))?;
        let prepared = if seq == 0 {
            adapter
                .restart_prepared_turn_cold_per_block(seq, tokens, budget, extra_keys, cache_salt)
        } else {
            adapter.prepare_turn_per_block_booking_hit_only(
                seq, tokens, budget, extra_keys, cache_salt, true, 0,
            )
        };
        prepared.map(drop).map_err(Error::from_reason)
    })
}

#[cfg(test)]
thread_local! {
    static STATE_READ_LOST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// While it lives, the state read a prime armed with is lost before the
/// GDN prime takes it, and the state does not read back again.
#[cfg(test)]
pub(crate) struct StateReadLost;

#[cfg(test)]
impl StateReadLost {
    pub(crate) fn new() -> Self {
        STATE_READ_LOST.with(|lost| lost.set(true));
        Self
    }
}

#[cfg(test)]
impl Drop for StateReadLost {
    fn drop(&mut self) {
        STATE_READ_LOST.with(|lost| lost.set(false));
    }
}

/// The deepest stored state an autoregressive prime of `tokens` restores
/// its GDN cell from: at most `cached` deep, deeper than `deeper_than`
/// (the best ladder rung). Returns its id, depth and a deep copy of its GDN
/// slots (Splash `QwenStateStorage::restore` with `restoreDraftState`
/// false: a composite state serves a request that needs no draft).
/// A state on disk alone gives its cell from the read its prime armed with,
/// else from its object (the window is read and dropped); one that does
/// not read back is discarded and the next deepest taken.
pub(crate) fn stored_gdn_prefix<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    extra_keys: &[Vec<u64>],
    cache_salt: u64,
    cached: u32,
    deeper_than: u32,
) -> Option<(EntryId, u32, Vec<super::layer_cache::Qwen3_5LayerCache>)> {
    for _ in 0..=target.dflash2_states().store.bytes().entries {
        let store = &target.dflash2_states().store;
        let hit = store.lookup_within(
            Lineage {
                tokens,
                extra_keys,
                salt: cache_salt,
            },
            cached as usize,
        )?;
        let depth = u32::try_from(hit.depth).ok()?;
        if depth <= deeper_than {
            return None;
        }
        let mut caches = target.fresh_layer_caches();
        if let Some(anchor) = store.get(hit.id)?.ram() {
            super::dflash2_state::paged::install_linear(&mut caches, &anchor.linear).ok()?;
            return Some((hit.id, depth, caches));
        }
        let read = match target
            .dflash2_states_mut()
            .state_read
            .take_if(|read| read.id == hit.id)
        {
            Some(read) => Some(read.linear),
            None => disk::read_state(target, hit.id, None),
        };
        #[cfg(test)]
        let read = read.filter(|_| !STATE_READ_LOST.with(std::cell::Cell::get));
        match read {
            Some(linear) => {
                super::dflash2_state::paged::adopt_linear(&mut caches, linear).ok()?;
                return Some((hit.id, depth, caches));
            }
            None => disk::discard_state(target, hit.id),
        }
    }
    None
}

/// Installs parked state `id` into the lane, whose sequence holds no live
/// request of an earlier turn: the GDN cell is deep-copied in, the history
/// is the state's lineage, and the state stays parked with a new stamp
/// (Splash `QwenStateStorage::restore`, `StateCache::release`). With
/// `restore_draft` the draft ring is projected again from the state's
/// 4-bit window and the lane's window takes its rows; without it the ring
/// starts empty at the state's depth (Splash `restoreLengths(.., false)`).
/// A state on disk alone is read into the lane (Splash `beginRestore`: the
/// window comes along either way) and waits for the admission to promote
/// it; one that does not read back is discarded and `None` returned.
/// Returns the resume depth. The caller holds the restore's claim.
pub(crate) fn install_parked_state<T: DFlash2Target>(
    target: &mut T,
    id: EntryId,
    restore_draft: bool,
) -> Result<Option<u32>> {
    let (mark, depth, on_disk, history) = {
        let entry = target
            .dflash2_states()
            .store
            .get(id)
            .ok_or_else(|| Error::from_reason(format!("DFlash2 parked state {id} is gone")))?;
        (
            entry.mark(),
            entry.depth(),
            entry.ram().is_none(),
            entry.tokens().to_vec(),
        )
    };
    if !on_disk
        && !target
            .paged_adapter_ref()
            .is_some_and(|adapter| adapter.prefix_cached(mark))
    {
        return Err(Error::from_reason(
            "DFlash2 state outlived its target KV block",
        ));
    }
    let mut caches = target.fresh_layer_caches();
    let config = target
        .dflash2()
        .ok_or_else(|| Error::from_reason("DFlash2 state install without a draft"))?
        .config
        .clone();
    let idle = &mut target.dflash2_states_mut().idle_lane;
    let mut ring = idle.take_ring(&config);
    let mut lane = if restore_draft || on_disk {
        Some(idle.take_window(&config)?)
    } else {
        None
    };
    if on_disk {
        let Some(linear) = disk::read_state(target, id, lane.as_mut()) else {
            let idle = &mut target.dflash2_states_mut().idle_lane;
            idle.give_ring(Some(ring));
            idle.give_window(lane);
            disk::discard_state(target, id);
            return Ok(None);
        };
        super::dflash2_state::paged::adopt_linear(&mut caches, linear)?;
    } else {
        let anchor = target
            .dflash2_states()
            .store
            .get(id)
            .and_then(|entry| entry.ram())
            .ok_or_else(|| Error::from_reason(format!("DFlash2 state {id} holds no RAM copy")))?;
        super::dflash2_state::paged::install_linear(&mut caches, &anchor.linear)?;
        if let Some(lane) = lane.as_mut() {
            let window = anchor
                .window
                .as_ref()
                .ok_or_else(|| Error::from_reason(format!("DFlash2 state {id} holds no window")))?;
            lane.copy_from(window)?;
        }
    }
    let draft = target
        .dflash2()
        .ok_or_else(|| Error::from_reason("DFlash2 state install without a draft"))?;
    match lane.as_ref().filter(|_| restore_draft) {
        Some(window) => ring = DFlash2ContextCache::restore(ring, draft, window, &history)?,
        None => {
            ring.reset();
            ring.record_only(&history)?;
        }
    }
    target.adopt_parked_state(caches, ring, history);
    let states = target.dflash2_states_mut();
    let (lane, kept) = if restore_draft {
        (lane, None)
    } else {
        (None, lane)
    };
    states.lane_window = lane;
    if on_disk {
        states.disk_read = Some(super::dflash2_state::paged::DiskRead { id, window: kept });
    }
    states.store.touch(id);
    #[cfg(test)]
    SUSPEND_LOG.with(|log| log.borrow_mut().push(SuspendRecord::Restored { depth }));
    u32::try_from(depth)
        .map(Some)
        .map_err(|_| Error::from_reason("DFlash2 parked state depth overflow"))
}

/// The window a turn plans its states with: the draft's sliding window, or
/// Splash's `draftContextTokens` on a load without a draft.
fn state_window<T: DFlash2Target>(target: &T) -> Result<u32> {
    match target.dflash2() {
        Some(draft) => u32::try_from(draft.config.sliding_window)
            .map_err(|_| Error::from_reason("DFlash2 draft window exceeds u32 rows")),
        None => Ok(super::dflash2_state::plan::state_window_tokens()),
    }
}

/// Splash `Engine::configureDraftStatePlan` for a turn of `tokens` that
/// resumes at `resume` with a junction at `junction` (0 for none), with the
/// state window; `None` without a pool. Shared-prefill points are not
/// planned.
fn state_plan<T: DFlash2Target>(
    target: &T,
    tokens: &[u32],
    prompt: u32,
    resume: u32,
    junction: u32,
) -> Result<Option<(StatePlan, u32)>> {
    let Some(adapter) = target.paged_adapter_ref() else {
        return Ok(None);
    };
    let block = adapter.block_size();
    let len = u32::try_from(tokens.len())
        .map_err(|_| Error::from_reason("DFlash2 prompt exceeds u32 tokens"))?;
    let window = state_window(target)?;
    let prompt_replay = u32::try_from(crate::engine::gen_prompt::prompt_replay_boundary(
        &tokens[..(prompt as usize).min(tokens.len())],
        target.dflash2_states().generation_suffix.as_deref(),
        block as usize,
    ))
    .map_err(|_| Error::from_reason("DFlash2 replay point exceeds u32 tokens"))?;
    // Splash `plannedCheckpoints` margins with the constant
    // `prefillTokenBudget`, never the configured chunk.
    let chunk =
        super::paged_forward::prefill_chunk_rows(crate::models::forward::PREFILL_STEP_SIZE as i32);
    let plan = configure_draft_state_plan(&PlanInputs {
        resume,
        junction,
        prompt_replay,
        replay_state: replay_state_boundary(len, prompt.min(len), prompt_replay, block),
        window,
        interval: checkpoint_interval(window, block),
        chunk: u32::try_from(chunk).unwrap_or(0),
    });
    Ok(Some((plan, window)))
}

/// Splash `Cache::lookup`'s junction: the KV hit of `tokens` (its leading
/// cached blocks, capped below the prompt's last token as `extendMatch`
/// caps it) when it lies past `resume` and a state sits below its last
/// block; 0 for none. The plan may still reject it.
fn junction_probe<T: DFlash2Target>(
    target: &T,
    tokens: &[u32],
    cache_salt: u64,
    resume: u32,
) -> u32 {
    let Some(adapter) = target.paged_adapter_ref() else {
        return 0;
    };
    let block = adapter.block_size().max(1) as usize;
    let store = &target.dflash2_states().store;
    let mut chain = store.chain(Lineage::text(tokens, cache_salt));
    chain.truncate(tokens.len().saturating_sub(1) / block);
    let matched = chain
        .iter()
        .take_while(|&&hash| adapter.prefix_cached(hash))
        .count();
    let kv = u32::try_from(matched * block).unwrap_or(0);
    let below = matched > 0 && store.state_below(chain[matched - 1]);
    if kv > resume && below { kv } else { 0 }
}

/// Opens the DFlash2 turn's points for a lane of `tokens`, whose first
/// `prompt` are the prompt, resuming at `resume` (Splash
/// `Engine::configureDraftStatePlan` at admission). A turn without reuse,
/// a draft or a pool opens an empty plan. `extra_keys` are the lineage's
/// per-block image keys; `None` keeps the lane's running turn's.
#[allow(clippy::too_many_arguments)]
fn open_turn<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    prompt: u32,
    resume: u32,
    junction: u32,
    cache_salt: u64,
    reuse: bool,
    extra_keys: Option<Vec<Vec<u64>>>,
) -> Result<StatePlan> {
    let plan = if reuse {
        state_plan(&*target, tokens, prompt, resume, junction)?
            .map(|(plan, _)| plan)
            .unwrap_or_default()
    } else {
        StatePlan::default()
    };
    let anchor_bytes = turn_anchor_bytes(&*target, &plan);
    let seq = target.lane_seq();
    let extra_keys = extra_keys.unwrap_or_else(|| {
        target
            .dflash2_states_mut()
            .turns
            .get_mut(&seq)
            .map(|turn| std::mem::take(&mut turn.extra_keys))
            .unwrap_or_default()
    });
    target.dflash2_states_mut().turns.insert(
        seq,
        TurnStates {
            plan: plan.clone(),
            cursor: 0,
            passed: Vec::new(),
            anchor_bytes,
            latest_checkpoint: None,
            cache_salt,
            extra_keys,
            junction,
            prompt_tokens: prompt,
            reuse,
        },
    );
    Ok(plan)
}

/// Reopens the running turn's points from 0: its restore did not happen.
fn reopen_cold<T: DFlash2Target>(target: &mut T, tokens: &[u32], prompt: u32) -> Result<()> {
    if let Some((salt, junction, reuse)) = target
        .dflash2_states()
        .turns
        .get(&target.lane_seq())
        .map(|turn| (turn.cache_salt, turn.junction, turn.reuse))
    {
        open_turn(target, tokens, prompt, 0, junction, salt, reuse, None)?;
    }
    Ok(())
}

/// The prompt length of the running turn's lane of `tokens`: all of them
/// at a turn start, fewer for a resumed lane that replays its output.
fn turn_prompt<T: DFlash2Target>(target: &T, tokens: &[u32]) -> u32 {
    let len = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
    target
        .dflash2_states()
        .turns
        .get(&target.lane_seq())
        .map_or(len, |turn| turn.prompt_tokens.min(len))
}

/// What one capture of the turn holds: a GDN cell, the window (none
/// without a draft) and the deepest point's lineage vectors. Read while
/// the draft is on the model.
fn turn_anchor_bytes<T: DFlash2Target>(target: &T, plan: &StatePlan) -> u64 {
    let block = target
        .paged_adapter_ref()
        .map_or(1, PagedKVCacheAdapter::block_size)
        .max(1);
    let deepest = plan.boundaries.last().map_or(0, |point| point.tokens) as usize;
    anchor_bytes(
        &target.dense_config(),
        target.dflash2().map(|draft| &draft.config),
    )
    .saturating_add(lineage_bytes(deepest, deepest / block as usize))
}

/// The parked state a DFlash2 lane of `tokens` resumes from (Splash
/// `Engine::admit`: `Cache::lookup`, then `configureDraftStatePlan`): the
/// turn's points open, a junction at the KV hit among them, and the plan
/// tells whether the restore rebuilds the draft (`restoresDraftState`). No
/// restore runs and no lookup is counted here.
struct ResumePlan {
    hit: Option<super::dflash2_state::store::StoreHit>,
    restore_draft: bool,
}

fn plan_resume<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    prompt: u32,
    reuse_cache: bool,
    cache_salt: u64,
) -> Result<ResumePlan> {
    let hit = parked_state_hit(target, tokens, reuse_cache, cache_salt);
    let resume = match hit {
        Some(hit) => u32::try_from(hit.depth)
            .map_err(|_| Error::from_reason("DFlash2 parked state depth overflow"))?,
        None => 0,
    };
    let junction = if reuse_cache {
        junction_probe(&*target, tokens, cache_salt, resume)
    } else {
        0
    };
    let plan = open_turn(
        target,
        tokens,
        prompt,
        resume,
        junction,
        cache_salt,
        reuse_cache,
        Some(Vec::new()),
    )?;
    let end = u32::try_from(tokens.len())
        .map_err(|_| Error::from_reason("DFlash2 prompt exceeds u32 tokens"))?;
    let restore_draft = match (hit, target.dflash2()) {
        (Some(_), Some(draft)) if resume < end => {
            let points: Vec<u32> = plan.boundaries.iter().map(|point| point.tokens).collect();
            plan_draft_context(
                resume,
                end,
                &points,
                u32::try_from(draft.config.sliding_window).unwrap_or(u32::MAX),
            )?
            .restores_draft_state
        }
        _ => false,
    };
    Ok(ResumePlan { hit, restore_draft })
}

/// What a restore of `id` claims for the lane (Splash `admitIdleLane`:
/// the governor is asked only for what the idle lane lacks): the GDN cell,
/// the ring bytes the idle ring lacks and, when the draft is rebuilt or
/// the state is read from disk (the read takes the window along), a window
/// when the idle lane holds none.
fn restore_bytes<T: DFlash2Target>(target: &T, id: EntryId, restore_draft: bool) -> u64 {
    let on_disk = target
        .dflash2_states()
        .store
        .get(id)
        .is_some_and(|entry| entry.ram().is_none());
    let restore_draft = restore_draft || on_disk;
    let Some(draft) = target.dflash2() else {
        return 0;
    };
    let cell = target.dense_config().recurrent_state_bytes();
    let idle = &target.dflash2_states().idle_lane;
    let ring = draft.config.kv_ring_bytes().saturating_sub(
        idle.ring
            .as_ref()
            .map_or(0, DFlash2ContextCache::buffer_bytes),
    );
    let window = if restore_draft && idle.window.is_none() {
        draft.config.context_window_bytes()
    } else {
        0
    };
    cell.saturating_add(ring).saturating_add(window)
}

/// Where a fresh DFlash2 turn of `tokens` resumes (Splash `Engine::admit`):
/// the deepest parked state, installed under the turn's claim for the
/// lane's buffers, else cold. A restore the ledger fails goes cold (Splash
/// `admit` skips the cache). Returns the parked state's id beside the hit.
pub(crate) fn resolve_dflash2_resume<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    reuse_cache: bool,
    cache_salt: u64,
) -> Result<(DFlash2Hit, Option<EntryId>)> {
    target.dflash2_turn_state_clear();
    let states = target.dflash2_states_mut();
    let lane = states.lane_window.take();
    states.idle_lane.give_window(lane);
    if let Some(read) = states.disk_read.take() {
        states.idle_lane.give_window(read.window);
    }
    let prompt = u32::try_from(tokens.len())
        .map_err(|_| Error::from_reason("DFlash2 prompt exceeds u32 tokens"))?;
    // A disk state that does not read back is discarded and the lookup
    // runs again (Splash: a failed restore discards the state, and the next
    // attempt matches what remains).
    for _ in 0..=target.dflash2_states().store.bytes().entries {
        let resume = plan_resume(target, tokens, prompt, reuse_cache, cache_salt)?;
        let Some(hit) = resume.hit else {
            return Ok((DFlash2Hit::Cold, None));
        };
        let bytes = restore_bytes(&*target, hit.id, resume.restore_draft);
        let keep = Keep { tokens, keys: None };
        let reservation = match claim_restore(target, bytes, keep)? {
            StateClaim::Refused => {
                reopen_cold(target, tokens, prompt)?;
                return Ok((DFlash2Hit::Cold, None));
            }
            StateClaim::Unclaimed => None,
            StateClaim::Reserved(reservation, _) => Some(reservation),
        };
        let depth = install_parked_state(target, hit.id, resume.restore_draft)?;
        drop(reservation);
        if let Some(depth) = depth {
            return Ok((DFlash2Hit::Stored(depth), Some(hit.id)));
        }
    }
    reopen_cold(target, tokens, prompt)?;
    Ok((DFlash2Hit::Cold, None))
}

/// Splash `Engine::pollRestores` once the lane's restore of a disk state
/// is complete: the state turns resident again under an optional,
/// immediate claim of its RAM copy (`promotedCopy`), the lane's GDN cell
/// and window copied in; a refusal leaves it on disk. A restore whose KV
/// did not come back discards the state.
fn settle_disk_read<T: DFlash2Target>(target: &mut T, tokens: &[u32], resumed: Option<EntryId>) {
    let Some(read) = target.dflash2_states_mut().disk_read.take() else {
        return;
    };
    if resumed != Some(read.id) {
        disk::discard_state(target, read.id);
    } else if let Some(bytes) = target
        .dflash2_states()
        .store
        .get(read.id)
        .filter(|entry| entry.ram().is_none())
        .map(super::dflash2_state::store::Entry::resident_bytes)
    {
        let keep = Keep { tokens, keys: None };
        let snapshot = |target: &mut T| -> Result<PagedAnchor> {
            let window = match read.window.as_ref() {
                Some(window) => Some(window.snapshot()?),
                None if target.dflash2().is_none() => None,
                None => Some(
                    target
                        .dflash2_states()
                        .lane_window_of(target.lane_seq())
                        .ok_or_else(|| Error::from_reason("a disk restore left no lane window"))?
                        .snapshot()?,
                ),
            };
            Ok(PagedAnchor {
                linear: target.snapshot_live_linear()?,
                window,
            })
        };
        let mut made = None;
        let claim = claim_state(
            target,
            CaptureClass::Ordinary,
            bytes,
            keep,
            Refusal::Settled,
            |target, retry| {
                if retry {
                    return false;
                }
                made = snapshot(target).ok();
                made.is_some()
            },
        );
        let (anchor, reservation) = match claim {
            StateClaim::Unclaimed => (snapshot(target).ok(), None),
            StateClaim::Reserved(reservation, _) => (made.take(), Some(reservation)),
            StateClaim::Refused => (None, None),
        };
        let promoted = anchor.is_some_and(|anchor| disk::promote(target, read.id, anchor));
        drop(reservation);
        if !promoted {
            disk::promotion_skipped(target.dflash2_states_mut());
        }
    }
    target
        .dflash2_states_mut()
        .idle_lane
        .give_window(read.window);
}

/// Splash `Engine::completeAdmission` for a turn of `tokens` the prime put
/// at `cached`: a cold fallback reopens the points from 0; a resume at a
/// checkpoint makes it the lane's rolling checkpoint, unless it is the
/// prompt's own replay point, where it turns ordinary (a deduplicated
/// publication); the replay point is then in use until the turn ends.
pub(crate) fn admit_resumed<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    stored: Option<EntryId>,
    cached: usize,
) -> Result<()> {
    let Some(block_size) = target
        .paged_adapter_ref()
        .map(PagedKVCacheAdapter::block_size)
    else {
        return Ok(());
    };
    let resumed = stored.filter(|&id| {
        cached > 0
            && target
                .dflash2_states()
                .store
                .get(id)
                .is_some_and(|entry| entry.depth() == cached)
    });
    settle_disk_read(target, tokens, resumed);
    let prompt = turn_prompt(&*target, tokens);
    let Some(id) = resumed else {
        return reopen_cold(target, tokens, prompt);
    };
    let seq = target.lane_seq();
    let states = target.dflash2_states_mut();
    let replay = crate::engine::gen_prompt::prompt_replay_boundary(
        &tokens[..prompt as usize],
        states.generation_suffix.as_deref(),
        block_size as usize,
    );
    let mut latest = states.store.checkpoint_state(id);
    if cached == replay && latest.is_some() {
        states.store.reuse(id, false);
        states.counters.deduplicated_state_publications += 1;
        latest = None;
    }
    if let Some(turn) = states.turns.get_mut(&seq) {
        turn.latest_checkpoint = latest;
    }
    if cached == replay {
        states.use_replay_point(seq, id);
    }
    Ok(())
}

/// The DFlash2 prime of a turn of `tokens` on `seq`: a live request there
/// (an autoregressive turn's) is released, the resume resolved, and
/// `prime` prepares the prefix from it (Splash `Engine::admit` →
/// `completeAdmission`, which counts the lookup's junction once the turn
/// starts: `Cache::recordLookup`).
pub(crate) fn prime_dflash2_turn<T: DFlash2Target, P: crate::engine::backend::PagedPrefix>(
    target: &mut T,
    tokens: &[u32],
    reuse_cache: bool,
    cache_salt: u64,
    seq: u32,
    prime: impl FnOnce(&mut T, DFlash2Hit) -> Result<P>,
) -> Result<P> {
    if let Some(adapter) = target.paged_adapter_mut_ref() {
        adapter
            .release_request_for(seq)
            .map_err(Error::from_reason)?;
    }
    let (hit, stored) = resolve_dflash2_resume(target, tokens, reuse_cache, cache_salt)?;
    let prefix = prime(target, hit)?;
    let cached = prefix.effective_cached_prefix_len();
    if cached == 0 {
        target.clear_draft_context();
    }
    admit_resumed(target, tokens, stored, cached)?;
    let seq = target.lane_seq();
    let states = target.dflash2_states_mut();
    if states
        .turns
        .get(&seq)
        .is_some_and(|turn| turn.junction != 0)
    {
        states.counters.lazy_junctions += 1;
    }
    Ok(prefix)
}

/// Plans the DFlash2 prefill of `tokens` from `cached` in chunks of
/// `chunk_rows` on the turn's points (Splash `reserveCaptures` passes a
/// checkpoint the same chunk carries past). A lane whose GDN state is not
/// exact on a text chain captures nothing: its points go. Returns the plan
/// and the rows a chunk ends at to capture.
pub(crate) fn turn_plan<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    cached: usize,
    cache_salt: u64,
    chunk_rows: i32,
) -> Result<(StatePlan, Vec<u32>)> {
    let resume = u32::try_from(cached)
        .map_err(|_| Error::from_reason("DFlash2 prefix exceeds u32 tokens"))?;
    let len = u32::try_from(tokens.len())
        .map_err(|_| Error::from_reason("DFlash2 prompt exceeds u32 tokens"))?;
    let seq = target.lane_seq();
    let (junction, latest, reuse) = target
        .dflash2_states()
        .turns
        .get(&seq)
        .map_or((0, None, true), |turn| {
            (turn.junction, turn.latest_checkpoint, turn.reuse)
        });
    let prompt = turn_prompt(&*target, tokens);
    let mut plan = open_turn(
        target, tokens, prompt, resume, junction, cache_salt, reuse, None,
    )?;
    if !target.replay_state_exact() {
        plan.boundaries.clear();
    }
    let breaks = command_breaks(resume, len, u32::try_from(chunk_rows).unwrap_or(0), &plan);
    if let Some(turn) = target.dflash2_states_mut().turns.get_mut(&seq) {
        turn.plan = plan.clone();
        turn.passed = breaks.passed;
        turn.latest_checkpoint = latest;
    }
    Ok((plan, breaks.breaks))
}

/// A state's class on the turn (Splash `Engine` `StateClass`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CaptureClass {
    /// A disposable point: the lane's rolling checkpoint.
    Checkpoint,
    /// The prompt's replay point, which the turn uses.
    InUse,
    /// A junction, a parked state's install, a ladder rung or the history
    /// checkpoint.
    Ordinary,
}

/// The class of `plan`'s point at `row` (Splash `reserveCaptures`
/// `stateClass`).
fn capture_class(plan: &StatePlan, row: u32) -> Option<CaptureClass> {
    plan.boundaries
        .iter()
        .find(|point| point.tokens == row)
        .map(|point| {
            if point.disposable {
                CaptureClass::Checkpoint
            } else if point.tokens == plan.prompt_replay {
                CaptureClass::InUse
            } else {
                CaptureClass::Ordinary
            }
        })
}

/// Splash `Engine::stateClaim` on the scope's request: the state the turn
/// uses is in service, waits and reaches the states turns use; any other
/// is optional and immediate.
pub(crate) fn state_claim(class: CaptureClass, scope: Option<ClaimSpec>) -> ClaimSpec {
    let (request_id, rank) =
        scope.map_or((0, Default::default()), |spec| (spec.request_id, spec.rank));
    match class {
        CaptureClass::InUse => ClaimSpec {
            cls: ClaimClass::InService,
            request_id,
            rank,
            reach: ReachClass::InUse,
            timing: NeedTiming::CanWait,
        },
        CaptureClass::Checkpoint | CaptureClass::Ordinary => ClaimSpec {
            cls: ClaimClass::Optional,
            request_id,
            rank,
            reach: ReachClass::Ordinary,
            timing: NeedTiming::Immediate,
        },
    }
}

/// What the ledger said to a state's bytes.
enum StateClaim {
    /// The governor does not know the model, or nothing is claimed.
    Unclaimed,
    /// Granted; true when the ledger freed units for it.
    Reserved(Reservation<'static>, bool),
    Refused,
}

/// A state claim as a test's fake ledger scripts it.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScriptedClaim {
    /// Refused; nothing else can free memory.
    Refuse,
    /// Refused while memory is on its way back.
    Wait,
    /// Granted, reporting whether units were freed for it.
    Grant { freed: bool },
}

/// One state claim as the ledger saw it.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ClaimLog {
    pub cls: ClaimClass,
    pub timing: NeedTiming,
    pub reach: ReachClass,
    pub refusal: Refusal,
    pub bytes: u64,
}

#[cfg(test)]
thread_local! {
    /// The answers the next state claims on this thread get, in order;
    /// empty, the ledger answers.
    pub(crate) static STATE_CLAIM_SCRIPT: std::cell::RefCell<std::collections::VecDeque<ScriptedClaim>> =
        const { std::cell::RefCell::new(std::collections::VecDeque::new()) };
    /// Every state claim on this thread, in order.
    pub(crate) static STATE_CLAIM_LOG: std::cell::RefCell<Vec<ClaimLog>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Logs a state claim; a scripted answer replaces the ledger's.
#[cfg(test)]
fn scripted_claim(spec: ClaimSpec, refusal: Refusal, bytes: u64) -> Option<ScriptedClaim> {
    STATE_CLAIM_LOG.with(|log| {
        log.borrow_mut().push(ClaimLog {
            cls: spec.cls,
            timing: spec.timing,
            reach: spec.reach,
            refusal,
            bytes,
        });
    });
    STATE_CLAIM_SCRIPT.with(|script| script.borrow_mut().pop_front())
}

/// Reserves `bytes` for a state of `class` under a claim of the thread
/// scope's request, `keep`'s states kept (Splash `Engine::allocate` with
/// `refusal`); `attempt` runs once the bytes are reserved and returns
/// false when it could not make the state, which refuses the claim. A
/// refusal is counted.
fn claim_state<T: DFlash2Target>(
    target: &mut T,
    class: CaptureClass,
    bytes: u64,
    keep: Keep<'_>,
    refusal: Refusal,
    mut attempt: impl FnMut(&mut T, bool) -> bool,
) -> StateClaim {
    let Some((governor, model)) = target
        .memory_registration()
        .map(|guard| (guard.governor(), guard.model_id()))
    else {
        return StateClaim::Unclaimed;
    };
    if bytes == 0 {
        return StateClaim::Unclaimed;
    }
    let spec = state_claim(
        class,
        governor.scope_claim().and_then(|id| governor.spec(id)),
    );
    let claim = governor.open(spec, model);
    let mut first = true;
    let mut reserve = |target: &mut T| loop {
        let reservation = governor.try_reserve(bytes);
        let retry = reservation.is_err() && std::mem::take(&mut first);
        match reservation {
            Ok(reservation) if attempt(target, false) => break Ok(reservation),
            Ok(_) => break Err(crate::engine::governor::governor::Failure::EngineBudget),
            Err(_) if retry && attempt(target, true) => {}
            Err(failure) => break Err(failure),
        }
    };
    #[cfg(test)]
    if let Some(scripted) = scripted_claim(spec, refusal, bytes) {
        let _scope = crate::engine::governor::governor::Scope::enter(claim.claim());
        return match scripted {
            ScriptedClaim::Refuse | ScriptedClaim::Wait => {
                governor.publication_refused(model);
                StateClaim::Refused
            }
            ScriptedClaim::Grant { freed } => match reserve(target) {
                Ok(reservation) => StateClaim::Reserved(reservation, freed),
                Err(_) => StateClaim::Refused,
            },
        };
    }
    let allocation = allocate(
        claim.claim(),
        spec,
        refusal,
        false,
        &mut ModelUnits {
            target: &mut *target,
            keep,
        },
        |units| StateAdmission(reserve(&mut *units.target)),
    );
    match allocation.admission.0 {
        Ok(reservation) => StateClaim::Reserved(reservation, allocation.freed),
        Err(_) => {
            governor.publication_refused(model);
            StateClaim::Refused
        }
    }
}

/// Runs `publish` while `bytes` are admitted for it under an optional
/// claim of the thread scope's request (Splash `Engine::stateClaim` for an
/// ordinary state), `keep`'s states kept; a refusal skips the publication
/// (None), counted, and the turn goes on. A model the governor does not
/// know publishes unclaimed.
pub(crate) fn publish_claimed<T: DFlash2Target, R>(
    target: &mut T,
    bytes: u64,
    keep: Keep<'_>,
    publish: impl FnOnce(&mut T) -> R,
) -> Option<R> {
    match claim_state(
        target,
        CaptureClass::Ordinary,
        bytes,
        keep,
        Refusal::Settled,
        |_, retry| !retry,
    ) {
        StateClaim::Unclaimed => Some(publish(target)),
        StateClaim::Reserved(reservation, _) => {
            let published = publish(target);
            drop(reservation);
            Some(published)
        }
        StateClaim::Refused => None,
    }
}

/// Splash `Engine::checkpointReclaimable`: the lane's checkpoint is still
/// the one it published, and no restore pins it.
fn checkpoint_reclaimable(states: &DFlash2States, turn: &TurnStates) -> bool {
    turn.latest_checkpoint.is_some_and(|(id, publication)| {
        states
            .store
            .checkpoint_state(id)
            .is_some_and(|(_, current)| current == publication)
            && !states.store.checkpoint_pinned(id, publication)
    })
}

/// Claims the state the chunk ending at `row` captures (Splash
/// `Engine::reserveCapture`): its bytes are reserved until the store takes
/// the state. Refused while the lane holds a checkpoint it may retire, the
/// claim tries once and the state is taken from the lane after the chunk
/// (`fromLane`); refused otherwise, no state is taken.
pub(crate) fn reserve_capture<T: DFlash2Target>(target: &mut T, row: u32) -> Result<CaptureGrant> {
    let states = target.dflash2_states();
    let Some((class, bytes, from_lane)) = states.turns.get(&target.lane_seq()).and_then(|turn| {
        capture_class(&turn.plan, row).map(|class| {
            (
                class,
                turn.anchor_bytes,
                checkpoint_reclaimable(states, turn),
            )
        })
    }) else {
        return Err(Error::from_reason(format!(
            "DFlash2 prefill captures at row {row}, which the turn did not plan"
        )));
    };
    let prompt = target
        .paged_adapter_ref()
        .map(|adapter| adapter.request_tokens().to_vec())
        .unwrap_or_default();
    let keep = Keep {
        tokens: &prompt,
        keys: None,
    };
    let refusal = if from_lane {
        Refusal::SettledOnce
    } else {
        Refusal::Settled
    };
    Ok(
        match claim_state(target, class, bytes, keep, refusal, |_, retry| !retry) {
            StateClaim::Unclaimed => CaptureGrant::Reserved(None, false),
            StateClaim::Reserved(reservation, freed) => {
                CaptureGrant::Reserved(Some(reservation), freed)
            }
            StateClaim::Refused if from_lane => CaptureGrant::FromLane,
            StateClaim::Refused => CaptureGrant::Refused,
        },
    )
}

/// One try at the lane's buffers for a restore of a stored state (Splash
/// `admitIdleLane` claims them under the request's own claim, `Refusal::Waits`).
enum RestoreAttempt<'g> {
    /// Reserved, or the model is unknown to the governor.
    Granted(Option<Reservation<'g>>),
    /// Memory is on its way back or the host refuses it.
    Wait(Denial),
    /// Nothing else can free memory: the turn goes cold (Splash `admit`
    /// skips the cache).
    Fail,
}

fn restore_attempt<'g, T: DFlash2Target>(
    target: &mut T,
    claim: ClaimRef<'g>,
    spec: ClaimSpec,
    bytes: u64,
    keep: Keep<'_>,
    hold_ended: bool,
) -> RestoreAttempt<'g> {
    if bytes == 0 || target.memory_registration().is_none() {
        return RestoreAttempt::Granted(None);
    }
    let governor = claim.governor;
    #[cfg(test)]
    if let Some(scripted) = scripted_claim(spec, Refusal::Waits, bytes) {
        let denial = match scripted {
            ScriptedClaim::Grant { .. } => {
                let _scope = crate::engine::governor::governor::Scope::enter(claim);
                return match governor.try_reserve(bytes) {
                    Ok(reservation) => RestoreAttempt::Granted(Some(reservation)),
                    Err(_) => RestoreAttempt::Fail,
                };
            }
            ScriptedClaim::Refuse => Denial::default(),
            ScriptedClaim::Wait => Denial {
                pending: true,
                ..Denial::default()
            },
        };
        return restore_verdict(denial);
    }
    let allocation = allocate(
        claim,
        spec,
        Refusal::Waits,
        hold_ended,
        &mut ModelUnits {
            target: &mut *target,
            keep,
        },
        |_| StateAdmission(governor.try_reserve(bytes)),
    );
    match allocation.admission.0 {
        Ok(reservation) => RestoreAttempt::Granted(Some(reservation)),
        Err(_) => restore_verdict(allocation.denial),
    }
}

/// Splash `admit` on a refused restore: `judge` fails it, else it waits.
fn restore_verdict<'g>(denial: Denial) -> RestoreAttempt<'g> {
    if judge(&denial, false) == Verdict::Fail {
        RestoreAttempt::Fail
    } else {
        RestoreAttempt::Wait(denial)
    }
}

/// The lane's buffers for a fresh turn's restore of a stored state:
/// `bytes` under the turn's claim, waiting while memory is on its way back
/// or the host refuses it. `Refused` when the claim fails: the turn goes
/// cold.
fn claim_restore<T: DFlash2Target>(
    target: &mut T,
    bytes: u64,
    keep: Keep<'_>,
) -> Result<StateClaim> {
    if bytes == 0 || target.memory_registration().is_none() {
        return Ok(StateClaim::Unclaimed);
    }
    let (governor, _transient, claim, spec, model) = turn_claim(&*target);
    let mut wait = ResourceWait::default();
    let turn_wait = TurnWait {
        governor,
        claim: claim.id,
    };
    loop {
        let denial = match restore_attempt(target, claim, spec, bytes, keep, wait.hold_ended) {
            RestoreAttempt::Granted(Some(reservation)) => {
                return Ok(StateClaim::Reserved(reservation, false));
            }
            RestoreAttempt::Granted(None) => return Ok(StateClaim::Unclaimed),
            RestoreAttempt::Fail => return Ok(StateClaim::Refused),
            RestoreAttempt::Wait(denial) => denial,
        };
        let now = governor.now();
        let epoch = governor.epoch(model);
        defer_resource_retry(&mut wait, now, &denial, WaitReason::Memory, true, 0, epoch);
        wait.hold_ended = wait
            .pending_since
            .is_some_and(|since| now >= since + RESOURCE_WAIT_SECONDS);
        if cancelled() {
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
        turn_wait.wait_for_retry(target, &wait, now);
        run_control_pass(governor, model, true, target, keep);
    }
}

/// A non-last chunk of the turn's tapped prefill ended at `row` (Splash
/// `Engine::apply`): its committed blocks register, then the point there,
/// if any, publishes.
pub(crate) fn reached<T: DFlash2Target>(
    target: &mut T,
    row: u32,
    outcome: Reached<'_>,
) -> Result<()> {
    super::dflash2_state::disk::tick(target, super::dflash2_state::disk::Tick::Poll);
    let Some((salt, image_keys)) = target
        .dflash2_states()
        .turns
        .get(&target.lane_seq())
        .map(|turn| (turn.cache_salt, !turn.extra_keys.is_empty()))
    else {
        return Ok(());
    };
    let keys = if image_keys {
        let blocks = target.paged_adapter_ref().map_or(0, |adapter| {
            row as usize / adapter.block_size().max(1) as usize
        });
        let turn = &target.dflash2_states().turns[&target.lane_seq()];
        (0..blocks)
            .map(|block| turn.extra_keys.get(block).cloned().unwrap_or_default())
            .collect()
    } else {
        target.paged_block_keys(row as usize)
    };
    target
        .paged_adapter_mut_ref()
        .ok_or_else(|| Error::from_reason("a paged prefill lost its paged adapter"))?
        .register_committed_blocks(row as usize, &keys, salt)
        .map_err(Error::from_reason)?;
    if !matches!(outcome, Reached::NoPoint) {
        publish_reached_state(target, row, outcome)?;
    }
    target.sync_gdn_state_marks();
    Ok(())
}

/// Splash `Engine::publishReachedStateBoundaries` for the point at `row`,
/// past the checkpoints the chunk carried past. The block must be
/// registered; the replay point is in use first; a state already on the
/// block is reused; else a checkpoint or junction first retires the lane's
/// resident checkpoint and the captured state is stored, or one the ledger
/// refused before the chunk is taken from the lane now. After it the
/// lane's checkpoint retires, except at the replay point, where it stays as
/// a disposable state for a later prompt that shares it.
fn publish_reached_state<T: DFlash2Target>(
    target: &mut T,
    row: u32,
    outcome: Reached<'_>,
) -> Result<()> {
    let lineage = target
        .paged_adapter_ref()
        .and_then(|adapter| adapter.request_tokens().get(..row as usize))
        .map(<[u32]>::to_vec)
        .ok_or_else(|| Error::from_reason(format!("DFlash2 prefill has no row {row}")))?;
    let seq = target.lane_seq();
    let Some(mut turn) = target.dflash2_states_mut().turns.remove(&seq) else {
        return Ok(());
    };
    let published = publish_point(target, &mut turn, &lineage, outcome);
    target.dflash2_states_mut().turns.insert(seq, turn);
    published
}

/// The live GDN cell and the sink's lane window (none without a draft),
/// both at `row`.
fn lane_state<T: DFlash2Target>(
    target: &mut T,
    sink: &mut (dyn super::paged_forward::PrefillTapSink + '_),
    row: u32,
) -> Result<PagedAnchor> {
    Ok(PagedAnchor {
        linear: target.snapshot_live_linear()?,
        window: sink.snapshot_lane_window(row as usize)?,
    })
}

fn publish_point<T: DFlash2Target>(
    target: &mut T,
    turn: &mut TurnStates,
    lineage: &[u32],
    outcome: Reached<'_>,
) -> Result<()> {
    let row = u32::try_from(lineage.len()).unwrap_or(u32::MAX);
    while let Some(point) = turn
        .plan
        .boundaries
        .get(turn.cursor)
        .filter(|point| point.tokens < row)
    {
        if !point.disposable || !turn.passed.contains(&point.tokens) {
            return Err(Error::from_reason(
                "prefill passed a state boundary it did not capture",
            ));
        }
        turn.cursor += 1;
    }
    let Some(point) = turn
        .plan
        .boundaries
        .get(turn.cursor)
        .copied()
        .filter(|point| point.tokens == row)
    else {
        return Err(Error::from_reason(format!(
            "prefill captured a state boundary it did not plan at row {row}"
        )));
    };
    turn.cursor += 1;
    let checkpoint = point.disposable;
    let replay = !checkpoint && row == turn.plan.prompt_replay;
    let class = if checkpoint {
        CaptureClass::Checkpoint
    } else if replay {
        CaptureClass::InUse
    } else {
        CaptureClass::Ordinary
    };
    let extra_keys = turn.extra_keys.clone();
    let text = Lineage {
        tokens: lineage,
        extra_keys: &extra_keys,
        salt: turn.cache_salt,
    };
    let (block, stored) = target.dflash2_states().store.locate(text);
    let registered = target
        .paged_adapter_ref()
        .is_some_and(|adapter| adapter.prefix_cached(block));
    let failed = |states: &mut DFlash2States| {
        let counters = &mut states.counters;
        let failures = if checkpoint {
            &mut counters.checkpoint_publication_failures
        } else if replay {
            &mut counters.replay_state_publication_failures
        } else {
            &mut counters.junction_materialization_failures
        };
        *failures += 1;
    };
    if !registered {
        failed(target.dflash2_states_mut());
        return Ok(());
    }
    if replay {
        let seq = target.lane_seq();
        target.dflash2_states_mut().use_replay_block(seq, block);
    }
    let reused = stored.filter(|&id| target.dflash2_states_mut().store.reuse(id, checkpoint));
    let depth = u32::try_from(lineage.len()).unwrap_or(u32::MAX);
    let id = match reused {
        Some(id) => {
            target
                .dflash2_states_mut()
                .counters
                .deduplicated_state_publications += 1;
            id
        }
        None => {
            let states = target.dflash2_states_mut();
            if checkpoint
                && turn.latest_checkpoint.is_some_and(|(id, publication)| {
                    states.store.checkpoint_pinned(id, publication)
                })
            {
                failed(states);
                return Ok(());
            }
            // Splash `stateResident`: a checkpoint only on disk gives no
            // buffers; it retires after the publication.
            let resident = turn.latest_checkpoint.is_some_and(|(id, _)| {
                states
                    .store
                    .get(id)
                    .is_some_and(|entry| entry.ram().is_some())
            });
            if !replay && resident {
                retire_checkpoint(states, turn);
            }
            let (state, lane) = match outcome {
                Reached::Captured(anchor, reservation, recycled) => {
                    (Some((anchor, reservation, recycled)), None)
                }
                Reached::FromLane(sink) => (
                    from_lane(target, turn, &mut *sink, class, row, replay, resident)?,
                    Some(sink),
                ),
                Reached::Refused(sink) => (None, Some(sink)),
                Reached::NoPoint => (None, None),
            };
            let class = if checkpoint {
                StateClass::Checkpoint
            } else {
                StateClass::Ordinary
            };
            let Some((anchor, reservation, recycled)) = state else {
                // No RAM copy takes the state: the tier takes it from the
                // lane (Splash `publishStateToDisk` with `snapshotToDisk`).
                let published = match lane {
                    Some(sink) => {
                        super::dflash2_state::disk::publish_from_lane(target, sink, text, class)?
                    }
                    None => None,
                };
                let states = target.dflash2_states_mut();
                let Some(id) = published else {
                    failed(states);
                    return Ok(());
                };
                let counters = &mut states.counters;
                counters.disk_state_publications += 1;
                *(if checkpoint {
                    &mut counters.checkpoint_publications
                } else if replay {
                    &mut counters.replay_state_publications
                } else {
                    &mut counters.junction_materializations
                }) += 1;
                finish_publication(target, turn, id, checkpoint, replay, depth);
                return Ok(());
            };
            let states = target.dflash2_states_mut();
            match states.store.insert(text, anchor, class) {
                Ok(Inserted { id, .. }) => {
                    drop(reservation);
                    let counters = &mut states.counters;
                    *(if checkpoint {
                        &mut counters.checkpoint_publications
                    } else if replay {
                        &mut counters.replay_state_publications
                    } else {
                        &mut counters.junction_materializations
                    }) += 1;
                    if recycled {
                        counters.recycled_state_publications += 1;
                    }
                    id
                }
                Err(error) => {
                    tracing::warn!(
                        target: "mlx_core::inference",
                        "DFlash2 state at row {row} not published: {}",
                        error.reason
                    );
                    failed(states);
                    return Ok(());
                }
            }
        }
    };
    finish_publication(target, turn, id, checkpoint, replay, depth);
    Ok(())
}

/// After a point's publication (Splash `publishReachedStateBoundaries`):
/// the lane's previous checkpoint retires, except at the replay point; an
/// ordinary point joins the copies ahead.
fn finish_publication<T: DFlash2Target>(
    target: &mut T,
    turn: &mut TurnStates,
    id: EntryId,
    checkpoint: bool,
    replay: bool,
    depth: u32,
) {
    let now = super::dflash2_state::disk::clock(&*target);
    let states = target.dflash2_states_mut();
    if !replay {
        retire_checkpoint(states, turn);
    }
    turn.latest_checkpoint = if checkpoint {
        states.store.checkpoint_state(id)
    } else {
        None
    };
    if !checkpoint {
        super::dflash2_state::write_behind::published(states, id, depth, now);
    }
}

/// Splash `publishReachedStateBoundaries`' `fromLane` arm: the state a
/// claim refused before the chunk, taken from the lane under its claim
/// now; for the replay point, a refusal first retires the lane's resident
/// checkpoint and tries once more. `recycled` when the ledger freed units
/// for it; None when refused.
fn from_lane<T: DFlash2Target>(
    target: &mut T,
    turn: &mut TurnStates,
    sink: &mut (dyn super::paged_forward::PrefillTapSink + '_),
    class: CaptureClass,
    row: u32,
    replay: bool,
    resident: bool,
) -> Result<Option<(PagedAnchor, Option<Reservation<'static>>, bool)>> {
    let prompt = target
        .paged_adapter_ref()
        .map(|adapter| adapter.request_tokens().to_vec())
        .unwrap_or_default();
    let keep = Keep {
        tokens: &prompt,
        keys: None,
    };
    let mut made: Option<Result<PagedAnchor>> = None;
    let claim = claim_state(
        target,
        class,
        turn.anchor_bytes,
        keep,
        Refusal::Settled,
        |target, retry| {
            if retry {
                return replay && resident && retire_checkpoint(target.dflash2_states_mut(), turn);
            }
            made = Some(lane_state(target, &mut *sink, row));
            matches!(made, Some(Ok(_)))
        },
    );
    Ok(match claim {
        StateClaim::Unclaimed => Some((lane_state(target, sink, row)?, None, false)),
        StateClaim::Reserved(reservation, freed) => {
            let anchor = made
                .take()
                .ok_or_else(|| Error::from_reason("a granted lane state was never taken"))??;
            Some((anchor, Some(reservation), freed))
        }
        StateClaim::Refused => {
            if let Some(Err(error)) = made {
                return Err(error);
            }
            None
        }
    })
}

/// Splash `Engine::retireCheckpoint`: the lane's checkpoint goes when it is
/// still the checkpoint it published (`StateCache::retireCheckpointState`);
/// no other DFlash2 lane runs beside this one, so none shares it. True when
/// the lane holds no checkpoint after it.
fn retire_checkpoint(states: &mut DFlash2States, turn: &mut TurnStates) -> bool {
    let Some((id, publication)) = turn.latest_checkpoint.take() else {
        return true;
    };
    if states
        .store
        .get(id)
        .is_some_and(|entry| entry.checkpoint() && entry.publication() == publication)
        && super::dflash2_state::disk::erase(states, id)
    {
        states.counters.checkpoint_retirements += 1;
    }
    true
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

/// The turn ended, after its request was released (Splash
/// `Engine::release`): its replay point is no longer in use and its points
/// go; what it published stays.
pub(crate) fn end_turn_states<T: DFlash2Target>(target: &mut T) {
    let seq = target.lane_seq();
    if target.dflash2_states().turns.contains_key(&seq) {
        log_turn_end("end_use");
    }
    target.dflash2_states_mut().end_turn(seq);
}

#[cfg(test)]
thread_local! {
    /// The steps DFlash2 turns on this thread ended with, in order.
    pub(crate) static TURN_END_LOG: std::cell::RefCell<Vec<&'static str>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
thread_local! {
    /// The live GDN cells the last DFlash2 finalize on this thread released.
    pub(crate) static RELEASED_LANE: std::cell::RefCell<Option<Vec<super::layer_cache::Qwen3_5LayerCache>>> =
        const { std::cell::RefCell::new(None) };
}

/// Keeps a released lane's GDN cells for a test to read.
pub(crate) fn note_released_lane(_cells: Option<Vec<super::layer_cache::Qwen3_5LayerCache>>) {
    #[cfg(test)]
    RELEASED_LANE.with(|lane| *lane.borrow_mut() = _cells);
}

/// Notes a step of a turn's end.
pub(crate) fn log_turn_end(_step: &'static str) {
    #[cfg(test)]
    TURN_END_LOG.with(|log| log.borrow_mut().push(_step));
}

/// The DFlash2 draft context a tapped paged prefill advances chunk by chunk
/// (Splash `DFlashDraft::addContextPrefill`). The K/V ring projects only the
/// rows its sliding window keeps; the lane's 4-bit context window takes the
/// capture spans' rows below the turn's last point (Splash
/// `planDraftContext`), and at an armed point the window and the GDN cell
/// are that point's state (Splash `Engine::publishReachedStateBoundaries`).
pub(crate) struct DraftTapSink<'a> {
    draft: &'a DFlash2Model,
    context: DFlash2ContextCache,
    /// First row the K/V ring keeps (its sliding window's rows).
    kv_from: usize,
    /// The capture spans below the last point.
    spans: Vec<CaptureSpan>,
    lane: Option<DraftContextWindow>,
    tap_layers: Vec<usize>,
    captures: Vec<usize>,
    armed: Option<usize>,
    window: Option<(usize, DraftContextWindow)>,
    linear: Option<(usize, Vec<super::layer_cache::Qwen3_5LayerCache>)>,
}

impl<'a> DraftTapSink<'a> {
    /// `tokens` is the whole prompt; the prefill starts at `cached`, where
    /// `context` ends. `plan` is the turn's, its points in `(cached,
    /// tokens.len())`; `captures` the rows a chunk ends at to capture.
    /// `lane` is the lane's window: the one a restore left at `cached`, or
    /// an empty one, which a span restarts at its first row.
    pub(crate) fn new(
        draft: &'a DFlash2Model,
        context: DFlash2ContextCache,
        mut lane: DraftContextWindow,
        tokens: &[u32],
        cached: usize,
        plan: &StatePlan,
        captures: &[u32],
    ) -> Result<Self> {
        let window_rows = draft.config.sliding_window;
        let to_u32 = |row: usize| {
            u32::try_from(row).map_err(|_| Error::from_reason("DFlash2 prompt exceeds u32 rows"))
        };
        let points: Vec<u32> = plan.boundaries.iter().map(|point| point.tokens).collect();
        let spans = match points.last() {
            Some(&last) => plan_draft_context(
                to_u32(cached)?,
                to_u32(tokens.len())?,
                &points,
                to_u32(window_rows)?,
            )?
            .spans
            .into_iter()
            .filter(|span| span.begin < last)
            .map(|span| CaptureSpan {
                end: span.end.min(last),
                ..span
            })
            .collect(),
            None => Vec::new(),
        };
        if let Some(first) = spans.first()
            && (first.reset || lane.end() != first.begin)
        {
            lane.restart(first.begin);
        }
        Ok(Self {
            kv_from: tokens
                .len()
                .saturating_sub(window_rows.saturating_sub(1))
                .max(cached),
            spans,
            lane: Some(lane),
            tap_layers: draft.config.target_layers.clone(),
            draft,
            context,
            captures: captures.iter().map(|&row| row as usize).collect(),
            armed: None,
            window: None,
            linear: None,
        })
    }

    /// The sink of a request that decodes autoregressively: only the
    /// lane's window takes rows, the K/V ring none.
    pub(crate) fn without_ring(mut self) -> Self {
        self.kv_from = usize::MAX;
        self
    }

    /// The draft context the prefill advanced and the lane's window.
    pub(crate) fn into_parts(self) -> (DFlash2ContextCache, Option<DraftContextWindow>) {
        (self.context, self.lane)
    }
}

impl crate::models::qwen3_5::paged_forward::PrefillTapSink for DraftTapSink<'_> {
    fn tap_layers(&self) -> &[usize] {
        &self.tap_layers
    }

    /// Rows a span or the K/V ring needs are fused once; the rest only
    /// advance length and provenance.
    fn chunk(&mut self, first_row: usize, chunk: &[u32], taps: &[MxArray]) -> Result<()> {
        let end = first_row + chunk.len();
        let spans: Vec<(usize, usize, bool)> = self
            .spans
            .iter()
            .filter_map(|span| {
                let (begin, span_end) = (span.begin as usize, span.end as usize);
                let from = begin.max(first_row);
                let to = span_end.min(end);
                (from < to).then_some((from, to, span.reset && from == begin))
            })
            .collect();
        let kv = self.kv_from.max(first_row);
        let lo = spans
            .first()
            .map_or(kv, |&(from, _, _)| from.min(kv))
            .min(end);
        if lo >= end {
            return self.context.record_only(chunk);
        }
        let fused = if lo == first_row {
            self.draft.fuse_context(taps)?
        } else {
            let tail_taps = taps
                .iter()
                .map(|tap| tap.slice_axis(1, (lo - first_row) as i64, chunk.len() as i64))
                .collect::<Result<Vec<_>>>()?;
            self.draft.fuse_context(&tail_taps)?
        };
        for (from, to, reset) in spans {
            let lane = self
                .lane
                .as_mut()
                .ok_or_else(|| Error::from_reason("DFlash2 capture span without a lane window"))?;
            if reset {
                lane.restart(from as u32);
            }
            let rows = fused.slice_axis(1, (from - lo) as i64, (to - lo) as i64)?;
            lane.store_rows(&rows, from as u32)?;
        }
        if let Some(row) = self.armed.filter(|&row| row > first_row && row <= end) {
            let lane = self
                .lane
                .as_ref()
                .filter(|lane| lane.end() as usize == row)
                .ok_or_else(|| {
                    Error::from_reason(format!(
                        "DFlash2 lane window does not end at the capture row {row}"
                    ))
                })?;
            note_lane_snapshot();
            self.window = Some((row, lane.snapshot()?));
        }
        if kv >= end {
            return self.context.record_only(chunk);
        }
        if kv > first_row {
            self.context.record_only(&chunk[..kv - first_row])?;
        }
        // RoPE keys are per row at the absolute row, so a sliced append is
        // identical.
        let ring_rows = if kv == lo {
            fused
        } else {
            fused.slice_axis(1, (kv - lo) as i64, (end - lo) as i64)?
        };
        let projected = self.context.project(self.draft, &ring_rows, kv as i32)?;
        self.context
            .append_projected(projected, &chunk[kv - first_row..])
    }

    fn eval(&self) -> Result<()> {
        let mut arrays = Vec::new();
        self.context.collect_arrays(&mut arrays);
        if let Some(lane) = &self.lane {
            lane.collect_arrays(&mut arrays);
        }
        if arrays.is_empty() {
            Ok(())
        } else {
            MxArray::eval_arrays(&arrays)
        }
    }

    fn capture_rows(&self) -> &[usize] {
        &self.captures
    }

    fn arm(&mut self, row: usize, armed: bool) {
        self.armed = armed.then_some(row);
        self.window = None;
        self.linear = None;
    }

    fn capture_linear(&mut self, row: usize, linear: Vec<super::layer_cache::Qwen3_5LayerCache>) {
        self.linear = Some((row, linear));
    }

    fn write_lane_window(&mut self, row: usize, out: &mut [u8]) -> Result<Option<u32>> {
        match self.lane.as_ref() {
            Some(lane) if lane.end() as usize == row => {
                Ok(lane.write_bytes(out)?.then(|| lane.begin()))
            }
            _ => Ok(None),
        }
    }

    fn snapshot_lane_window(&mut self, row: usize) -> Result<Option<DraftContextWindow>> {
        let lane = self
            .lane
            .as_ref()
            .filter(|lane| lane.end() as usize == row)
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "DFlash2 lane window does not end at the state row {row}"
                ))
            })?;
        note_lane_snapshot();
        lane.snapshot().map(Some)
    }

    fn take_captured(&mut self, row: usize) -> Option<PagedAnchor> {
        self.armed = None;
        match (self.window.take(), self.linear.take()) {
            (Some((at, window)), Some((linear_at, linear))) if at == row && linear_at == row => {
                Some(PagedAnchor {
                    linear,
                    window: Some(window),
                })
            }
            _ => None,
        }
    }
}

/// The state sink of a prefill on a load without a draft (Splash
/// `publishReachedStateBoundaries` with a state that is the GDN cell
/// alone): it taps no layer, and at an armed point the GDN cell is that
/// point's state.
pub(crate) struct LaneStateSink {
    captures: Vec<usize>,
    armed: Option<usize>,
    linear: Option<(usize, Vec<super::layer_cache::Qwen3_5LayerCache>)>,
}

impl LaneStateSink {
    /// `captures` are the rows a chunk ends at to capture.
    pub(crate) fn new(captures: &[u32]) -> Self {
        Self {
            captures: captures.iter().map(|&row| row as usize).collect(),
            armed: None,
            linear: None,
        }
    }
}

impl crate::models::qwen3_5::paged_forward::PrefillTapSink for LaneStateSink {
    fn tap_layers(&self) -> &[usize] {
        &[]
    }

    fn chunk(&mut self, _first_row: usize, _chunk: &[u32], _taps: &[MxArray]) -> Result<()> {
        Ok(())
    }

    fn eval(&self) -> Result<()> {
        Ok(())
    }

    fn capture_rows(&self) -> &[usize] {
        &self.captures
    }

    fn arm(&mut self, row: usize, armed: bool) {
        self.armed = armed.then_some(row);
        self.linear = None;
    }

    fn capture_linear(&mut self, row: usize, linear: Vec<super::layer_cache::Qwen3_5LayerCache>) {
        self.linear = Some((row, linear));
    }

    /// No window: the object's window span is empty and begins at `row`.
    fn write_lane_window(&mut self, row: usize, _out: &mut [u8]) -> Result<Option<u32>> {
        Ok(u32::try_from(row).ok())
    }

    fn snapshot_lane_window(&mut self, _row: usize) -> Result<Option<DraftContextWindow>> {
        Ok(None)
    }

    fn take_captured(&mut self, row: usize) -> Option<PagedAnchor> {
        let armed = self.armed.take();
        match self.linear.take() {
            Some((at, linear)) if at == row && armed == Some(row) => Some(PagedAnchor {
                linear,
                window: None,
            }),
            _ => None,
        }
    }
}

/// Splash `Engine::admit` → `configureDraftStatePlan` and
/// `completeAdmission` for a turn of `tokens` on a load without a draft,
/// once its prime put the lane at `cached`: the turn's points open from
/// `cached` (no junction: the prime's GDN replay already reached the KV
/// hit). A prime that restored stored state `restored` at `cached` makes a
/// disk copy resident under an optional claim (Splash `promotedCopy`), a
/// checkpoint the lane's rolling one, and the prompt's replay point the
/// turn's own, in use until the turn ends. A lane whose GDN state is not
/// exact on its chain plans no point. `extra_keys` are the lineage's
/// per-block image keys (an image turn's), none on a text chain.
pub(crate) fn open_lane_turn<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    extra_keys: &[Vec<u64>],
    cached: usize,
    cache_salt: u64,
    reuse: bool,
    restored: Option<EntryId>,
) -> Result<()> {
    open_lane_turn_at(
        target, tokens, extra_keys, cached, cache_salt, reuse, 0, restored,
    )
}

/// [`open_lane_turn`] with the KV hit `junction` past `cached` (Splash
/// `Cache::lookup`'s junction; 0 for none).
#[allow(clippy::too_many_arguments)]
fn open_lane_turn_at<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    extra_keys: &[Vec<u64>],
    cached: usize,
    cache_salt: u64,
    reuse: bool,
    junction: u32,
    restored: Option<EntryId>,
) -> Result<()> {
    if target.paged_adapter_ref().is_none() {
        return Ok(());
    }
    let seq = target.lane_seq();
    let resume = u32::try_from(cached)
        .map_err(|_| Error::from_reason("a prime's prefix exceeds u32 tokens"))?;
    let prompt = u32::try_from(tokens.len())
        .map_err(|_| Error::from_reason("a prompt exceeds u32 tokens"))?;
    let plan = open_turn(
        target,
        tokens,
        prompt,
        resume,
        junction,
        cache_salt,
        reuse,
        Some(extra_keys.to_vec()),
    )?;
    if !target.replay_state_exact()
        && let Some(turn) = target.dflash2_states_mut().turns.get_mut(&seq)
    {
        turn.plan.boundaries.clear();
    }
    let resumed = restored.filter(|&id| {
        cached > 0
            && target
                .dflash2_states()
                .store
                .get(id)
                .is_some_and(|entry| entry.depth() == cached)
    });
    if let Some(id) = resumed
        && target
            .dflash2_states()
            .store
            .get(id)
            .is_some_and(|entry| entry.ram().is_none())
    {
        target.dflash2_states_mut().disk_read =
            Some(super::dflash2_state::paged::DiskRead { id, window: None });
    }
    settle_disk_read(target, tokens, resumed);
    let Some(id) = resumed else {
        return Ok(());
    };
    let states = target.dflash2_states_mut();
    let mut latest = states.store.checkpoint_state(id);
    let replay = resume == plan.prompt_replay;
    if replay && latest.is_some() {
        states.store.reuse(id, false);
        states.counters.deduplicated_state_publications += 1;
        latest = None;
    }
    if let Some(turn) = states.turns.get_mut(&seq) {
        turn.latest_checkpoint = latest;
    }
    if replay {
        states.use_replay_point(seq, id);
    }
    Ok(())
}

/// Splash `configureDraftStatePlan` and `publishReachedStateBoundaries` for
/// an image request: the image prefill of `tokens` from `cached` opens the
/// lane's turn on its image lineage (`extra_keys`) and runs `prefill` with
/// the turn's sink, so its chunks reach the plan's points. `restored` is the
/// stored state a prime installed at `cached`. On a draft load the states
/// carry the draft window the same forward's taps fill, the restored
/// state's window first in the lane; `take_draft` and `give_draft` move the
/// draft off the model for the prefill. An image turn decodes
/// autoregressively, so the draft ring goes back to the idle lane.
#[allow(clippy::too_many_arguments)]
pub(crate) fn image_lane_prefill<T, L, R>(
    target: &mut T,
    tokens: &[u32],
    extra_keys: &[Vec<u64>],
    cached: u32,
    cache_salt: u64,
    reuse: bool,
    restored: Option<EntryId>,
    chunk_rows: i32,
    take_draft: impl FnOnce(&mut T) -> Option<DFlash2Model>,
    give_draft: impl FnOnce(&mut T, DFlash2Model),
    prefill: impl FnOnce(
        &mut dyn crate::models::qwen3_5::paged_forward::PrefillTarget<L>,
        &mut dyn crate::models::qwen3_5::paged_forward::PrefillTapSink,
    ) -> Result<R>,
) -> Result<R>
where
    T: DFlash2Target + crate::models::qwen3_5::paged_forward::PrefillTarget<L>,
{
    let cached_rows = cached as usize;
    let resumed = restored.filter(|&id| {
        cached > 0
            && target
                .dflash2_states()
                .store
                .get(id)
                .is_some_and(|entry| entry.depth() == cached_rows)
    });
    let draft_config = target.dflash2().map(|draft| draft.config.clone());
    let mut lane_end = cached;
    if let Some(config) = draft_config.as_ref() {
        let mut lane = target.dflash2_states_mut().idle_lane.take_window(config)?;
        if let Some(id) = resumed {
            restored_window(target, id, &mut lane);
        }
        lane_end = lane.end();
        target.dflash2_states_mut().lane_window = Some(lane);
    }
    open_lane_turn(
        target,
        tokens,
        extra_keys,
        cached_rows,
        cache_salt,
        reuse,
        resumed,
    )?;
    keep_drawable_points(target, cached, lane_end)?;
    let captures = lane_command_breaks(target, cached_rows, tokens.len(), chunk_rows);
    let Some(config) = draft_config else {
        let mut sink = LaneStateSink::new(&captures);
        return prefill(&mut LanePrefill(target), &mut sink);
    };
    let plan = target
        .dflash2_states()
        .turns
        .get(&target.lane_seq())
        .map(|turn| turn.plan.clone())
        .unwrap_or_default();
    let states = target.dflash2_states_mut();
    let lane = match states.lane_window.take() {
        Some(lane) => lane,
        None => states.idle_lane.take_window(&config)?,
    };
    let mut ring = states.idle_lane.take_ring(&config);
    ring.reset();
    let Some(model) = take_draft(target) else {
        let idle = &mut target.dflash2_states_mut().idle_lane;
        idle.give_window(Some(lane));
        idle.give_ring(Some(ring));
        return Err(Error::from_reason(
            "an image prefill lost its DFlash2 draft",
        ));
    };
    let prefilled = ring
        .record_only(&tokens[..cached_rows])
        .and_then(|()| DraftTapSink::new(&model, ring, lane, tokens, cached_rows, &plan, &captures))
        .and_then(|mut sink| {
            let prefilled = prefill(&mut LanePrefill(&mut *target), &mut sink);
            let (ring, lane) = sink.into_parts();
            let idle = &mut target.dflash2_states_mut().idle_lane;
            idle.give_window(lane);
            idle.give_ring(Some(ring));
            prefilled
        });
    give_draft(target, model);
    prefilled
}

/// Puts the context window of stored state `id` into `lane`, which then
/// ends at the state's row; an empty window when it does not read back.
fn restored_window<T: DFlash2Target>(target: &mut T, id: EntryId, lane: &mut DraftContextWindow) {
    let ram = target
        .dflash2_states()
        .store
        .get(id)
        .and_then(|entry| entry.ram())
        .map(|anchor| match anchor.window.as_ref() {
            Some(window) => lane.copy_from(window).is_ok(),
            None => false,
        });
    let filled = match ram {
        Some(filled) => filled,
        None => disk::read_state(target, id, Some(lane)).is_some(),
    };
    if !filled {
        lane.restart(0);
    }
}

/// The deepest stored state a scheduled request of `tokens` resumes at,
/// below its last token; one whose chain cannot come back whole is
/// discarded and the next deepest taken (Splash `acquireDeepest`).
fn scheduled_state_hit<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    cache_salt: u64,
) -> Option<(EntryId, u32)> {
    let cap = tokens.len().saturating_sub(1);
    for _ in 0..=target.dflash2_states().store.bytes().entries {
        let hit = target
            .dflash2_states()
            .store
            .lookup_within(Lineage::text(tokens, cache_salt), cap)?;
        if disk::restorable(&*target, hit.id) {
            return u32::try_from(hit.depth).ok().map(|depth| (hit.id, depth));
        }
        disk::discard_state(target, hit.id);
    }
    None
}

/// Splash `Engine::admit` → `configureDraftStatePlan` for a scheduled
/// request on a draft load, which decodes autoregressively (the scheduler
/// never speculates on a draft load): it resumes only at a stored state's
/// block, else from 0 with its KV hit as the junction (Splash
/// `Cache::lookup`), and its lane's context window is the state's
/// (Splash `QwenStateStorage::restore`), kept on its sequence for its
/// prefill slices. `prime` prepares the prefix from the hit.
pub(crate) fn prime_scheduled_draft_lane<T, P>(
    target: &mut T,
    tokens: &[u32],
    reuse_cache: bool,
    cache_salt: u64,
    prime: impl FnOnce(&mut T, DFlash2Hit) -> Result<P>,
) -> Result<P>
where
    T: DFlash2Target,
    P: crate::engine::backend::PagedPrefix,
{
    let seq = target.lane_seq();
    let config = target
        .dflash2()
        .ok_or_else(|| Error::from_reason("a draft lane prime without a draft"))?
        .config
        .clone();
    let states = target.dflash2_states_mut();
    let earlier = states.seq_windows.remove(&seq);
    states.idle_lane.give_window(earlier);
    let hit = if reuse_cache {
        scheduled_state_hit(target, tokens, cache_salt)
    } else {
        None
    };
    let prefix = prime(
        target,
        hit.map_or(DFlash2Hit::Cold, |(_, depth)| DFlash2Hit::Stored(depth)),
    )?;
    let cached = prefix.effective_cached_prefix_len();
    let resume = u32::try_from(cached)
        .map_err(|_| Error::from_reason("a prime's prefix exceeds u32 tokens"))?;
    let resumed = hit
        .filter(|&(_, depth)| cached > 0 && depth == resume)
        .map(|(id, _)| id);
    let mut lane = target.dflash2_states_mut().idle_lane.take_window(&config)?;
    if let Some(id) = resumed {
        restored_window(target, id, &mut lane);
    }
    let lane_end = lane.end();
    target.dflash2_states_mut().seq_windows.insert(seq, lane);
    let junction = if reuse_cache {
        junction_probe(&*target, tokens, cache_salt, resume)
    } else {
        0
    };
    open_lane_turn_at(
        target,
        tokens,
        &[],
        cached,
        cache_salt,
        reuse_cache,
        junction,
        resumed,
    )?;
    keep_drawable_points(target, resume, lane_end)?;
    let states = target.dflash2_states_mut();
    if states
        .turns
        .get(&seq)
        .is_some_and(|turn| turn.junction != 0)
    {
        states.counters.lazy_junctions += 1;
    }
    Ok(prefix)
}

/// A lane resumed at `resume` whose context window does not end there
/// draws no point's window from rows below it: only points a whole window
/// past it stay on the turn (Splash restores a window with each state it
/// resumes at).
pub(crate) fn keep_drawable_points<T: DFlash2Target>(
    target: &mut T,
    resume: u32,
    lane_end: u32,
) -> Result<()> {
    if lane_end == resume {
        return Ok(());
    }
    let first = resume.saturating_add(state_window(&*target)?);
    let seq = target.lane_seq();
    if let Some(turn) = target.dflash2_states_mut().turns.get_mut(&seq) {
        turn.plan.boundaries.retain(|point| point.tokens >= first);
    }
    Ok(())
}

/// Splash `publishReachedStateBoundaries` for one prefill slice, rows
/// `[cached, end)` of `tokens`, of a scheduled request on a draft load:
/// its chunks reach the lane's points with the context window the same
/// forward's taps fill, continued from the sequence's window. The request
/// decodes autoregressively, so no draft ring takes rows. `take_draft` and
/// `give_draft` move the draft off the model for the slice.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scheduled_draft_prefill<T, L, R>(
    target: &mut T,
    tokens: &[u32],
    cached: usize,
    end: usize,
    chunk_rows: i32,
    take_draft: impl FnOnce(&mut T) -> Option<DFlash2Model>,
    give_draft: impl FnOnce(&mut T, DFlash2Model),
    prefill: impl FnOnce(
        &mut dyn crate::models::qwen3_5::paged_forward::PrefillTarget<L>,
        &mut dyn crate::models::qwen3_5::paged_forward::PrefillTapSink,
    ) -> Result<R>,
) -> Result<R>
where
    T: DFlash2Target + crate::models::qwen3_5::paged_forward::PrefillTarget<L>,
{
    let seq = target.lane_seq();
    let config = target
        .dflash2()
        .ok_or_else(|| Error::from_reason("a draft lane prefill without a draft"))?
        .config
        .clone();
    let captures = lane_command_breaks(target, cached, end, chunk_rows);
    let mut plan = target
        .dflash2_states()
        .turns
        .get(&seq)
        .map(|turn| turn.plan.clone())
        .unwrap_or_default();
    plan.boundaries
        .retain(|point| point.tokens as usize > cached);
    let states = target.dflash2_states_mut();
    let lane = match states.seq_windows.remove(&seq) {
        Some(lane) => lane,
        None => states.idle_lane.take_window(&config)?,
    };
    let Some(model) = take_draft(target) else {
        target.dflash2_states_mut().seq_windows.insert(seq, lane);
        return Err(Error::from_reason(
            "a scheduled prefill lost its DFlash2 draft",
        ));
    };
    let ring = DFlash2ContextCache::new(&model.config);
    let prefilled = DraftTapSink::new(&model, ring, lane, tokens, cached, &plan, &captures)
        .map(DraftTapSink::without_ring)
        .and_then(|mut sink| {
            let prefilled = prefill(&mut LanePrefill(&mut *target), &mut sink);
            let (_, lane) = sink.into_parts();
            if let Some(lane) = lane {
                target.dflash2_states_mut().seq_windows.insert(seq, lane);
            }
            prefilled
        });
    give_draft(target, model);
    prefilled
}

/// Splash `reserveCaptures` over one prefill command of the lane's turn,
/// rows `[start, end)` in chunks of `chunk` rows: the rows a chunk ends at
/// to capture, a point at `end` among them when the prompt runs on past it
/// (Splash publishes a boundary at a command's end from the lane). The
/// checkpoints the command carries past are noted on the turn.
pub(crate) fn lane_command_breaks<T: DFlash2Target>(
    target: &mut T,
    start: usize,
    end: usize,
    chunk: i32,
) -> Vec<u32> {
    let seq = target.lane_seq();
    let (Ok(start), Ok(end)) = (u32::try_from(start), u32::try_from(end)) else {
        return Vec::new();
    };
    let Some(turn) = target.dflash2_states_mut().turns.get_mut(&seq) else {
        return Vec::new();
    };
    let stop = if end < turn.prompt_tokens {
        end.saturating_add(1)
    } else {
        end
    };
    let breaks = command_breaks(start, stop, u32::try_from(chunk).unwrap_or(0), &turn.plan);
    turn.passed
        .extend(breaks.passed.into_iter().filter(|&row| row <= end));
    breaks
        .breaks
        .into_iter()
        .filter(|&row| row <= end)
        .collect()
}

/// A prefill on a load without a draft as its turn's target: the parts and
/// the plain booking of the model, the state claims and publications of
/// the turn (Splash `Engine::prepare` and `apply` for any request).
pub(crate) struct LanePrefill<'a, T>(pub(crate) &'a mut T);

impl<L, T> crate::models::qwen3_5::paged_forward::PrefillTarget<L> for LanePrefill<'_, T>
where
    T: DFlash2Target + crate::models::qwen3_5::paged_forward::PrefillTarget<L>,
{
    fn parts(&mut self) -> Result<crate::models::qwen3_5::paged_forward::PrefillParts<'_, L>> {
        self.0.parts()
    }

    fn record(&mut self, chunk: &[u32]) -> Result<()> {
        self.0
            .paged_adapter_mut_ref()
            .ok_or_else(|| Error::from_reason("a paged prefill lost its paged adapter"))?
            .record_tokens(chunk)
            .map_err(Error::from_reason)
    }

    fn prepare_capture(&mut self, row: u32) -> Result<CaptureGrant> {
        reserve_capture(self.0, row)
    }

    fn reached(&mut self, row: u32, outcome: Reached<'_>) -> Result<()> {
        reached(self.0, row, outcome)
    }
}

#[cfg(test)]
thread_local! {
    /// Lane windows the tapped prefills on this thread snapshotted.
    pub(crate) static LANE_SNAPSHOTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn note_lane_snapshot() {
    #[cfg(test)]
    LANE_SNAPSHOTS.with(|count| count.set(count.get() + 1));
}

/// What a reclaim took, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GrowReclaim {
    /// A state left with no copy, or its RAM copy beside a disk one.
    Store,
    /// A state's RAM copy left through a staged write.
    StoreWrite,
    /// A disk state's chain was asked to the tier.
    StoreChain,
    Ladder,
    History,
    Vision,
    /// An idle owner's row and table.
    Idle,
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

pub(crate) fn log_reclaim(_taken: GrowReclaim) {
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

/// What is left of a retry interval of `seconds` that began at `started`
/// on the governor's clock, at `now`: the state tier's wait already spent
/// the rest.
pub(crate) fn retry_left(seconds: f64, started: f64, now: f64) -> f64 {
    (seconds - (now - started).max(0.0)).max(0.0)
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

    /// Waits to the wait's retry time: the turn's claim waited. The wait
    /// moves an owed chain at the writer's pace, and a landing retries at
    /// once (the epoch moved).
    fn wait_for_retry<T: DFlash2Target>(&self, target: &mut T, wait: &ResourceWait, now: f64) {
        let seconds = (wait.retry_at - now).max(0.0);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(seconds);
        if !super::dflash2_state::disk::tick(
            target,
            super::dflash2_state::disk::Tick::Wait { deadline },
        ) {
            self.governor
                .sleep(retry_left(seconds, now, self.governor.now()));
        }
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
/// suspends a serial text decode in place ([`suspend_in_place`]), ends a
/// turn's own tapped prefill [`DFLASH2_SUSPENDED`] for its owner to vacate
/// and replay ([`suspendable_prefill`]), and makes any other step wait the
/// same way; any other refusal ends the turn `capacity_exhausted`.
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
        if forced_yield(&*target) {
            let denial = Denial {
                failure: Some(crate::engine::governor::governor::Failure::HostPressure),
                ..Denial::default()
            };
            if yield_lane(target, claim, spec, model, &denial)? {
                wait = ResourceWait::default();
                continue;
            }
        }
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
        if verdict == Verdict::Yield && yield_lane(target, claim, spec, model, &denial)? {
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
        turn_wait.wait_for_retry(target, &wait, now);
        run_control_pass(governor, model, true, target, keep);
    }
}

/// A lone lane the host refused yields (Splash `Engine::yieldLane`): a
/// serial text decode suspends in place and resumes (true); a turn's own
/// tapped prefill latches the refusal and ends [`DFLASH2_SUSPENDED`]: its
/// frame owns the draft and the sink, so it vacates the lane (Splash
/// suspends between commands). Any other step waits (false).
fn yield_lane<T: DFlash2Target>(
    target: &mut T,
    claim: ClaimRef<'_>,
    spec: ClaimSpec,
    model: u64,
    denial: &Denial,
) -> Result<bool> {
    if let Some(decode) = target.serial_decode() {
        suspend_in_place(target, claim, spec, model, decode, denial)?;
        return Ok(true);
    }
    if target.dflash2_states().prefill_suspends && target.serial_text_lane() {
        target.dflash2_states_mut().prefill_yield = Some(PrefillYield {
            work_end: 0,
            failure: denial.failure,
        });
        return Err(Error::from_reason(DFLASH2_SUSPENDED));
    }
    Ok(false)
}

#[cfg(test)]
thread_local! {
    /// The first claim on this thread whose lane has fed this many tokens
    /// is refused by the host, once.
    pub(crate) static YIELD_FROM: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

fn forced_yield<T: DFlash2Target>(_target: &T) -> bool {
    #[cfg(test)]
    {
        let fed = _target
            .paged_adapter_ref()
            .and_then(|adapter| adapter.request_tokens_for(SERIAL_SEQ))
            .map_or(0, |tokens| tokens.len());
        if YIELD_FROM
            .with(std::cell::Cell::get)
            .is_some_and(|from| fed >= from)
        {
            YIELD_FROM.with(|from| from.set(None));
            return true;
        }
    }
    false
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
    let vacated = vacate_serial_lane(target)?;
    let resume_target = u32::try_from(vacated.tokens.len())
        .unwrap_or(u32::MAX)
        .saturating_add(decode.rows());
    let lane = SuspendedLane {
        tokens: &vacated.tokens,
        prompt: vacated.prompt,
        cache_salt: vacated.cache_salt,
        reuse: vacated.reuse,
        decode,
        resume_target,
    };
    // The replay's own claims wait: a lane is suspended once.
    target.set_serial_decode(None);
    let resumed = suspended_wait(target, claim, spec, model, &lane, denial.failure).map(|_| ());
    if resumed.is_ok() {
        target.set_serial_decode(Some(decode));
    }
    resumed
}

/// The suspended lane waits, resumes and replays (Splash
/// `deferResourceRetry`, `admit` of a suspended request, `replayTokens`);
/// the replay's last logits.
fn suspended_wait<T: DFlash2Target>(
    target: &mut T,
    claim: ClaimRef<'_>,
    spec: ClaimSpec,
    model: u64,
    lane: &SuspendedLane<'_>,
    failure: Option<crate::engine::governor::governor::Failure>,
) -> Result<MxArray> {
    let governor = claim.governor;
    let now = governor.now();
    governor.lane_suspended(model, now, failure);
    let mut wait = ResourceWait::default();
    defer_resource_retry(
        &mut wait,
        now,
        &Denial {
            failure,
            ..Denial::default()
        },
        WaitReason::Memory,
        true,
        0,
        governor.epoch(model),
    );
    let resumed =
        wait_and_resume(target, claim, spec, model, lane, &mut wait).and_then(|(plan, stored)| {
            target
                .replay_serial_lane(lane.tokens, lane.cache_salt, &plan, lane.decode, stored)
                .map_err(|error| {
                    if is_gpu_out_of_memory(&error.reason) {
                        governor.note_gpu_oom(model);
                        Error::from_reason(metal_out_of_memory(&error.reason))
                    } else if cancelled()
                        || error.reason == crate::models::forward::PREFILL_CANCELLED
                    {
                        // Splash `Engine::stop` settles a suspended lane whatever its
                        // phase: a stop during the replay is the vacated lane's stop.
                        Error::from_reason(crate::engine::session::LANE_VACATED_CANCELLED)
                    } else {
                        error
                    }
                })
        });
    governor.end_drain(model);
    if resumed.is_ok() {
        governor.lane_resumed(model);
    }
    resumed
}

/// Runs a turn's own tapped prefill of `tokens` on the serial lane. A host
/// refusal of a chunk unwinds it (the frame put its draft back and its
/// buffers in the idle lane); the lane is vacated, waits, resumes at its
/// deepest state and replays the rest of the prompt, tapped: the caller
/// sees one prefill, the replay's logits.
pub(crate) fn suspendable_prefill<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    prefill: impl FnOnce(&mut T) -> Result<MxArray>,
) -> Result<MxArray> {
    let armed = target.serial_text_lane();
    target.dflash2_states_mut().prefill_suspends = armed;
    let prefilled = prefill(target);
    let states = target.dflash2_states_mut();
    states.prefill_suspends = false;
    let yielded = states.prefill_yield.take();
    let yielded = match (prefilled, yielded) {
        (Err(error), Some(yielded)) if error.reason == DFLASH2_SUSPENDED => yielded,
        (prefilled, _) => return prefilled,
    };
    let (_governor, _transient, claim, spec, model) = turn_claim(&*target);
    let vacated = vacate_serial_lane(target)?;
    let lane = SuspendedLane {
        tokens,
        prompt: vacated.prompt,
        cache_salt: vacated.cache_salt,
        reuse: vacated.reuse,
        decode: SerialDecode::DFlash2,
        resume_target: yielded.work_end,
    };
    suspended_wait(target, claim, spec, model, &lane, yielded.failure)
}

struct SuspendedLane<'a> {
    tokens: &'a [u32],
    /// The prompt's length: `tokens` past it are the lane's output.
    prompt: u32,
    cache_salt: u64,
    /// The turn reuses the cache.
    reuse: bool,
    decode: SerialDecode,
    resume_target: u32,
}

/// What a vacated lane fed and its prompt's length.
struct Vacated {
    tokens: Vec<u32>,
    prompt: u32,
    cache_salt: u64,
    reuse: bool,
}

/// The serial lane's sequence.
const SERIAL_SEQ: u32 = 0;

/// Splash `Engine::vacateLane`: the lane's full blocks register as a cached
/// chain nothing pins; its live GDN cell goes and its draft buffers return
/// to the idle lane (`model_.suspend`); its sequence is released
/// (`cache_.endRequest`). The turn's pending points go
/// (`discardPendingStateBoundaries`); nothing is published, and its use of
/// its replay point stays (`vacateLane` keeps `replayPoint`).
fn vacate_serial_lane<T: DFlash2Target>(target: &mut T) -> Result<Vacated> {
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
    target.drop_serial_lane_state();
    target
        .paged_adapter_mut_ref()
        .ok_or_else(|| Error::from_reason("a serial lane lost its paged adapter"))?
        .release_request_for(SERIAL_SEQ)
        .map_err(Error::from_reason)?;
    // A prefill vacated part way still replays its whole prompt.
    let (prompt, reuse) = target.dflash2_states().turns.get(&SERIAL_SEQ).map_or(
        (u32::try_from(tokens.len()).unwrap_or(u32::MAX), true),
        |turn| (turn.prompt_tokens, turn.reuse),
    );
    target.dflash2_states_mut().turns.remove(&SERIAL_SEQ);
    target.sync_gdn_state_marks();
    #[cfg(test)]
    SUSPEND_LOG.with(|log| {
        log.borrow_mut().push(SuspendRecord::Vacated {
            fed: tokens.len(),
            states: target
                .dflash2_states()
                .store
                .entries()
                .map(|entry| (entry.depth(), entry.checkpoint()))
                .collect(),
            turn: target.dflash2_states().turns.contains_key(&SERIAL_SEQ),
            replay_point: target
                .dflash2_states()
                .replay_points
                .contains_key(&SERIAL_SEQ),
        });
    });
    Ok(Vacated {
        tokens,
        prompt,
        cache_salt,
        reuse,
    })
}

/// What a suspension did, in order.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SuspendRecord {
    /// The lane fed `fed` tokens and was vacated; the store held `states`
    /// (depth, checkpoint); a turn's points remained; its replay point
    /// stayed in use.
    Vacated {
        fed: usize,
        states: Vec<(usize, bool)>,
        turn: bool,
        replay_point: bool,
    },
    /// The lane resumed at `hit` with these points (tokens, disposable).
    Resumed {
        hit: Option<DFlash2Hit>,
        points: Vec<(u32, bool)>,
    },
    /// A parked state of `depth` was installed into the lane.
    Restored { depth: usize },
}

#[cfg(test)]
thread_local! {
    pub(crate) static SUSPEND_LOG: std::cell::RefCell<Vec<SuspendRecord>> =
        const { std::cell::RefCell::new(Vec::new()) };
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
) -> Result<(
    crate::transformer::paged_kv_cache_adapter::PagedTurnPlan,
    Option<EntryId>,
)> {
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
        turn_wait.wait_for_retry(target, wait, now);
        run_control_pass(governor, model, true, target, keep);
        if cancelled() {
            return Err(Error::from_reason(
                crate::engine::session::LANE_VACATED_CANCELLED,
            ));
        }
        // Splash `admit`: the lane's buffers, then the KV target; the
        // restore runs only once both are held.
        let mut restore = None;
        // On a draft load every lane's states carry the draft window, whatever
        // it decodes with (Splash keeps one composite state per request).
        let hit = if target.dflash2().is_some() {
            let resume = plan_resume(
                target,
                lane.tokens,
                lane.prompt,
                lane.reuse,
                lane.cache_salt,
            )?;
            match resume.hit {
                None => Some(DFlash2Hit::Cold),
                Some(hit) => {
                    let bytes = restore_bytes(&*target, hit.id, resume.restore_draft);
                    match restore_attempt(target, claim, spec, bytes, keep, wait.hold_ended) {
                        RestoreAttempt::Granted(reservation) => {
                            let depth = u32::try_from(hit.depth).map_err(|_| {
                                Error::from_reason("DFlash2 parked state depth overflow")
                            })?;
                            restore = Some((hit.id, depth, resume.restore_draft, reservation));
                            Some(DFlash2Hit::Stored(depth))
                        }
                        RestoreAttempt::Fail => {
                            reopen_cold(target, lane.tokens, lane.prompt)?;
                            Some(DFlash2Hit::Cold)
                        }
                        RestoreAttempt::Wait(denial) => {
                            defer_resource_retry(
                                wait,
                                governor.now(),
                                &denial,
                                WaitReason::Memory,
                                false,
                                0,
                                governor.epoch(model),
                            );
                            continue;
                        }
                    }
                }
            }
        } else {
            None
        };
        let mut stored = None;
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
                #[cfg(test)]
                SUSPEND_LOG.with(|log| {
                    log.borrow_mut().push(SuspendRecord::Resumed {
                        hit,
                        points: target
                            .dflash2_states()
                            .turns
                            .get(&SERIAL_SEQ)
                            .map(|turn| {
                                turn.plan
                                    .boundaries
                                    .iter()
                                    .map(|point| (point.tokens, point.disposable))
                                    .collect()
                            })
                            .unwrap_or_default(),
                    });
                });
                if let Some((id, depth, restore_draft, reservation)) = restore {
                    if plan.cached_prefix_len == depth {
                        if install_parked_state(target, id, restore_draft)?.is_none() {
                            // The state did not read back: it is gone, and the
                            // next attempt matches what remains.
                            if let Some(adapter) = target.paged_adapter_mut_ref() {
                                adapter
                                    .release_request_for(SERIAL_SEQ)
                                    .map_err(Error::from_reason)?;
                            }
                            drop(reservation);
                            continue;
                        }
                        stored = Some(id);
                    }
                    drop(reservation);
                }
                return Ok((plan, stored));
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
    let work_end = target.paged_adapter_ref().map_or(0, |adapter| {
        u32::try_from(adapter.request_tokens().len() + chunk.len()).unwrap_or(u32::MAX)
    });
    let recorded = claimed(target, Keep::default(), |target| {
        target
            .paged_adapter_mut_ref()
            .ok_or_else(|| Error::from_reason("a paged prefill lost its paged adapter"))?
            .record_tokens(chunk)
            .map_err(Error::from_reason)
    });
    if let Some(yielded) = target.dflash2_states_mut().prefill_yield.as_mut() {
        yielded.work_end = work_end;
    }
    recorded
}

/// The model's accounts on the governor's books: its KV pool; its states
/// (parked DFlash2 states, GDN ladder and history entries, the live
/// scheduled GDN cells, `gdn_cells`, the idle lane's draft ring and window,
/// and the scheduled sequences' windows); its image features.
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
    let states = target.dflash2_states();
    let idle = states
        .idle_lane
        .ring
        .as_ref()
        .map_or(0, DFlash2ContextCache::buffer_bytes)
        .saturating_add(
            states
                .idle_lane
                .window
                .as_ref()
                .map_or(0, DraftContextWindow::nbytes),
        )
        .saturating_add(
            states
                .seq_windows
                .values()
                .fold(0u64, |sum, window| sum.saturating_add(window.nbytes())),
        );
    crate::engine::governor::governor::MemoryAccounts {
        kv,
        states: states
            .store
            .bytes()
            .state_bytes
            .saturating_add(gdn)
            .saturating_add(gdn_cells)
            .saturating_add(idle),
        embeddings: crate::vision::qwen::cache::vision_cache_bytes(target.vision_cache()),
    }
}

/// The model's DFlash2 states for the governor's status: the turns'
/// counters and the store as it is now.
pub(crate) fn state_status<T: DFlash2Target>(
    target: &T,
) -> crate::engine::governor::governor::DFlash2StateStatus {
    let states = target.dflash2_states();
    let counters = states.counters;
    let mut status = crate::engine::governor::governor::DFlash2StateStatus {
        checkpoint_publications: counters.checkpoint_publications,
        replay_state_publications: counters.replay_state_publications,
        junction_materializations: counters.junction_materializations,
        checkpoint_publication_failures: counters.checkpoint_publication_failures,
        replay_state_publication_failures: counters.replay_state_publication_failures,
        junction_materialization_failures: counters.junction_materialization_failures,
        deduplicated_state_publications: counters.deduplicated_state_publications,
        recycled_state_publications: counters.recycled_state_publications,
        checkpoint_retirements: counters.checkpoint_retirements,
        lazy_junctions: counters.lazy_junctions,
        checkpoint_evictions: counters.checkpoint_evictions,
        evictions: counters.evictions,
        disk_state_publications: counters.disk_state_publications,
        ..Default::default()
    };
    if let Some(disk) = states.disk.as_ref() {
        use crate::engine::governor::governor::{DFlash2DiskStatus, DiskMode};
        let behind = disk.write_behind.snapshot();
        status.offloads = disk.counters.offloads;
        status.offload_failures = disk.counters.offload_failures;
        status.invalidations = disk.counters.invalidations;
        status.disk_promotions = disk.counters.promotions;
        status.disk_promotions_skipped = disk.counters.promotions_skipped;
        status.disk = DFlash2DiskStatus {
            mode: if !disk.tier.is_open() {
                DiskMode::Off
            } else if disk.tier.persistent {
                DiskMode::Persistent
            } else {
                DiskMode::Temp
            },
            limit_bytes: disk.tier.limit,
            held_bytes: disk.tier.held,
            taken_back: disk.counters.adopted,
            write_behind_waiting: u64::from(behind.waiting),
            write_behind_durable: behind.durable,
            write_behind_unneeded: behind.unneeded,
            write_behind_refused: behind.refused,
        };
    }
    for entry in states.store.entries() {
        status.entries += 1;
        status.bytes += entry.state_bytes();
        if let Some(copy) = entry.disk() {
            status.disk_entries += 1;
            status.disk_bytes += copy.bytes;
        }
        if entry.checkpoint() {
            status.checkpoint_entries += 1;
            status.checkpoint_bytes += entry.state_bytes();
        }
        if states.store.in_use(entry.mark()) {
            status.in_use += 1;
        }
    }
    status
}

/// A model's units on the ledger's order (Splash's cache walk): its pool's
/// empty extents, its stored checkpoints, then its other DFlash2 states and
/// GDN ladder and history entries by their stamps (the ones turns use at
/// `InUse`), and the image features nothing holds (`Rebuild`). `keep`'s
/// states are never listed.
pub(crate) struct ModelUnits<'a, T: DFlash2Target> {
    pub(crate) target: &'a mut T,
    pub(crate) keep: Keep<'a>,
}

impl<T: DFlash2Target> UnitSources for ModelUnits<'_, T> {
    fn sources(&mut self) -> Vec<&mut dyn VictimSource> {
        vec![self]
    }

    fn poll_landings(&mut self) {
        super::dflash2_state::disk::tick(self.target, super::dflash2_state::disk::Tick::Poll);
    }
}

impl<T: DFlash2Target> VictimSource for ModelUnits<'_, T> {
    fn units(&self, query: &VictimQuery) -> Vec<Victim> {
        model_units(&*self.target, self.keep, query).into_units()
    }

    fn take(&mut self, query: &VictimQuery, victim: &Victim) -> LedgerTaken {
        take_model_unit(self.target, self.keep, query, victim)
    }
}

/// [`ModelUnits`]' units of `target`. A need the host refused walks the
/// units that meet it in place (Splash `Cache::walkFor`): blocks only
/// (`pagesOnly`, no checkpoints), or with no blocks the states only
/// (`statesOnly`), whose buffers the refused allocation reuses from the MLX
/// free pool.
pub(crate) fn model_units<T: DFlash2Target>(
    target: &T,
    keep: Keep<'_>,
    query: &VictimQuery,
) -> UnitRuns {
    let host_refused = query.need.host_refused;
    let pages_only = host_refused && query.need.pages > 0;
    let states_only = host_refused && query.need.pages == 0;
    let mut runs = UnitRuns::default();
    let wanted = wanted_chains(target);
    let mut dead_leaves = None;
    if !host_refused {
        let keep_one = query.rights.reserves == Reserves::None;
        let bytes = target
            .paged_adapter_ref()
            .map_or(0, |adapter| adapter.free_extent_bytes(keep_one));
        if bytes > 0 {
            runs.ahead.push(Victim {
                key: UnitKey::Extent,
                stamp: 0,
                bytes,
                pages: 0,
                stage: Stage::FreeNow,
                later: false,
            });
        }
        // Splash `Cache::takeReclaim`: a device need takes the dead leaves
        // (`Phase::DeadLeaves`, before any ordinary state) and releases the
        // extent they empty. A need of blocks takes them in place.
        if query.need.bytes > 0 && query.need.pages == 0 {
            let lookups = kept_lookups(keep, &wanted, query.rights.certain_reuse);
            let bytes = target
                .paged_adapter_ref()
                .map_or(0, |adapter| adapter.cached_extent_bytes(keep_one, &lookups));
            if bytes > 0 {
                dead_leaves = Some(Victim {
                    key: UnitKey::DeadLeaves,
                    stamp: 0,
                    bytes,
                    pages: 0,
                    stage: Stage::FreeNow,
                    later: false,
                });
            }
        }
    }
    let kept = keep.points(
        target
            .paged_adapter_ref()
            .map(PagedKVCacheAdapter::block_size),
    );
    let state_bytes = |bytes: u64| if pages_only { 0 } else { bytes };
    // Every state, each with the block it marks and whether the claim's
    // prompt or a waiting one restores through it.
    let mut all: Vec<StateUnit> = Vec::new();
    let states = target.dflash2_states();
    let store = &states.store;
    for entry in store.entries() {
        let in_use = store.in_use(entry.mark());
        let checkpoint = entry.checkpoint() && !in_use;
        let (write, bytes, marks) = match (entry.ram(), entry.disk()) {
            // Splash `StateCache::marks`: a disk state needs its chain as it
            // is until the chain is on the tier too.
            (None, Some(copy)) if copy.chain_owed => (Write::Chain, 0, true),
            (None, _) => continue,
            (Some(_), Some(copy)) if copy.landed => {
                (Write::None, entry.state_bytes(), !copy.chain_owed)
            }
            (Some(_), Some(_)) => (Write::InFlight, entry.state_bytes(), false),
            (Some(_), None) if super::dflash2_state::disk::reclaim_writes(states, entry.id()) => {
                (Write::Needed, entry.state_bytes(), in_use)
            }
            (Some(_), None) => (Write::None, entry.state_bytes(), true),
        };
        let key = if write == Write::Chain {
            UnitKey::StoreChain(entry.id())
        } else {
            UnitKey::Store(entry.id())
        };
        all.push(StateUnit {
            unit: Victim {
                key,
                stamp: entry.last_used(),
                bytes: state_bytes(bytes),
                pages: 0,
                stage: if in_use { Stage::InUse } else { Stage::FreeNow },
                later: false,
            },
            mark: marks.then(|| entry.mark()),
            kept: kept.keeps(entry.tokens(), Some(entry.chain())),
            wanted: wanted.keeps(entry.tokens()),
            ordinary: !entry.checkpoint(),
            checkpoint,
            write,
            phase: Phase::Merged,
        });
    }
    for unit in target.gdn_state_units(keep, &wanted) {
        all.push(StateUnit {
            ordinary: unit.key == UnitKey::History,
            checkpoint: false,
            unit: Victim {
                key: unit.key,
                stamp: unit.stamp,
                bytes: state_bytes(unit.bytes),
                pages: 0,
                stage: if unit.in_use {
                    Stage::InUse
                } else {
                    Stage::FreeNow
                },
                later: false,
            },
            mark: unit.mark,
            kept: unit.kept,
            wanted: unit.wanted,
            write: Write::None,
            phase: Phase::Merged,
        });
    }
    let resume_point = resume_point(&all);
    let reuse = query.rights.certain_reuse;
    let mut states: Vec<StateUnit> = all
        .into_iter()
        .filter(|state| {
            !state.kept
                && (reuse == CertainReuse::All || !state.wanted)
                && (reuse != CertainReuse::None || resume_point.as_ref() != Some(&state.unit.key))
                && !(pages_only && state.checkpoint)
        })
        .collect();
    // Splash `MemoryLedger::Order::stage` over `Cache::stateReclaim`: a
    // state that needs a write, or whose copy lands later, goes after the
    // units that free memory now; with no write in flight the first
    // writable state of the walk is staged and frees at once, the others
    // wait for the staging buffer (Splash `StateCache::reclaims`).
    states.sort_by_key(|state| state.unit.stamp);
    let busy = target
        .dflash2_states()
        .disk
        .as_ref()
        .is_some_and(super::dflash2_state::disk::StateDisk::writing);
    let staged = (!busy)
        .then(|| {
            let writes = |in_use: bool, checkpoint: Option<bool>| {
                states.iter().position(|state| {
                    state.write == Write::Needed
                        && (state.unit.stage == Stage::InUse) == in_use
                        && checkpoint.is_none_or(|checkpoint| state.checkpoint == checkpoint)
                })
            };
            writes(false, Some(true))
                .or_else(|| writes(false, Some(false)))
                .or_else(|| query.rights.in_use.then(|| writes(true, None)).flatten())
        })
        .flatten();
    for (index, state) in states.iter_mut().enumerate() {
        let later = match state.write {
            Write::None => false,
            Write::Needed => staged != Some(index),
            Write::InFlight | Write::Chain => true,
        };
        if state.write != Write::None && state.unit.stage != Stage::InUse {
            state.unit.stage = Stage::NeedsWrite;
        }
        state.unit.later = later;
        // In use: no write first, then the staged write, then the later
        // ones (Splash `UsedNow`, `UsedWritten`, `UsedLater`).
        state.phase = match (state.unit.stage, state.write) {
            (Stage::InUse, Write::None) => Phase::InUse(0),
            (Stage::InUse, _) if !state.unit.later => Phase::InUse(1),
            (Stage::InUse, _) => Phase::InUse(2),
            _ if state.checkpoint => Phase::Ahead,
            _ => Phase::Merged,
        };
    }
    // Splash `Cache::WalkCursor::Phase`: the dead leaves go after the
    // checkpoints, before any ordinary state.
    let mut listed: Vec<(Victim, Option<u64>, Phase)> = Vec::new();
    listed.extend(runs.ahead.drain(..).map(|unit| (unit, None, Phase::Ahead)));
    listed.extend(
        states
            .iter()
            .filter(|state| state.phase == Phase::Ahead)
            .map(|state| (state.unit.clone(), state.mark, Phase::Ahead)),
    );
    listed.extend(dead_leaves.map(|unit| (unit, None, Phase::Ahead)));
    listed.extend(
        states
            .iter()
            .filter(|state| state.phase == Phase::Merged)
            .map(|state| (state.unit.clone(), state.mark, Phase::Merged)),
    );
    if !host_refused {
        let mut vision: Vec<Victim> =
            crate::vision::qwen::cache::unheld_vision_units(target.vision_cache())
                .into_iter()
                .map(|(generation, bytes)| Victim {
                    key: UnitKey::Vision(generation),
                    stamp: generation,
                    bytes,
                    pages: 0,
                    stage: Stage::Rebuild,
                    later: false,
                })
                .collect();
        vision.sort_by_key(|unit| unit.stamp);
        listed.extend(vision.into_iter().map(|unit| (unit, None, Phase::After)));
    }
    for group in 0..3 {
        listed.extend(
            states
                .iter()
                .filter(|state| state.phase == Phase::InUse(group))
                .map(|state| (state.unit.clone(), state.mark, Phase::After)),
        );
    }
    listed.retain(|(unit, mark, _)| unit.bytes > 0 || unit.pages > 0 || mark.is_some());
    // The blocks each take hands back, counted in the take order.
    listed.sort_by_key(|(unit, _, _)| (unit.stage, unit.later));
    if !states_only && let Some(adapter) = target.paged_adapter_ref() {
        let marked: Vec<usize> = (0..listed.len())
            .filter(|&index| listed[index].1.is_some())
            .collect();
        let marks: Vec<u64> = marked.iter().filter_map(|&index| listed[index].1).collect();
        for (index, pages) in marked.into_iter().zip(adapter.freed_by_marks(&marks)) {
            listed[index].0.pages = pages;
        }
    }
    for (unit, _, phase) in listed {
        if unit.bytes == 0 && unit.pages == 0 {
            continue;
        }
        match phase {
            Phase::Ahead => runs.ahead.push(unit),
            Phase::Merged => runs.merged.push(unit),
            Phase::InUse(_) | Phase::After => runs.after.push(unit),
        }
    }
    runs
}

/// What a state's take writes (Splash `Cache::stateReclaim`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Write {
    /// None: it goes, or its RAM copy does beside a landed disk copy.
    None,
    /// Its RAM copy leaves through a write.
    Needed,
    /// Its copy is in flight: it frees its RAM once that lands.
    InFlight,
    /// A disk-only state whose chain blocks are still to be written.
    Chain,
}

/// Where a unit sits in the phase order the ledger's stable sort keeps
/// within each stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Extents, checkpoints no turn uses, dead leaves.
    Ahead,
    /// Stamp order, into which the scheduler merges its idle owners.
    Merged,
    /// Image features, then the states turns use.
    After,
    InUse(u8),
}

fn wanted_chains<T: DFlash2Target>(target: &T) -> Arc<WantedChains> {
    target
        .memory_registration()
        .map_or_else(Arc::default, |guard| {
            guard.governor().wanted_chains(guard.model_id())
        })
}

/// The lookups whose chains a reclaim keeps: the claim's prompt, and the
/// waiting prompts unless the claim takes certain reuse (Splash
/// `Walk::reused`, `Walk::restored`).
fn kept_lookups<'a>(
    keep: Keep<'a>,
    wanted: &'a WantedChains,
    reuse: CertainReuse,
) -> Vec<&'a [u32]> {
    let mut lookups = vec![&keep.tokens[..keep.tokens.len().saturating_sub(1)]];
    if reuse != CertainReuse::All {
        lookups.extend(wanted.lookups());
    }
    lookups.retain(|lookup| !lookup.is_empty());
    lookups
}

/// A state on the take order, before the claim's rights filter it.
struct StateUnit {
    unit: Victim,
    mark: Option<u64>,
    kept: bool,
    wanted: bool,
    /// An ordinary stored state or the history entry, not a stored or
    /// ladder checkpoint.
    ordinary: bool,
    /// A stored checkpoint no turn uses (Splash `Phase::Checkpoints`).
    checkpoint: bool,
    write: Write,
    phase: Phase,
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
    keep: Keep<'_>,
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
        UnitKey::DeadLeaves => {
            let keep_one = query.rights.reserves == Reserves::None;
            let wanted = wanted_chains(&*target);
            let lookups = kept_lookups(keep, &wanted, query.rights.certain_reuse);
            if let Some(adapter) = target.paged_adapter_mut_ref() {
                pool_bytes = adapter.release_cached_extent(keep_one, &lookups);
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
        key => match take_state_unit(target, key) {
            StateTake::Took => true,
            StateTake::Nothing => false,
            StateTake::Pending => {
                return LedgerTaken {
                    outcome: FreeOutcome::Pending,
                    ..LedgerTaken::default()
                };
            }
        },
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
#[cfg(test)]
pub(crate) fn drop_parked_state<T: DFlash2Target>(target: &mut T, id: EntryId) -> bool {
    let states = target.dflash2_states_mut();
    if !states.store.remove(id) {
        return false;
    }
    target.sync_gdn_state_marks();
    true
}

/// What one state unit's take did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StateTake {
    Took,
    /// It frees nothing until a write in flight lands.
    Pending,
    Nothing,
}

/// Takes one state unit.
pub(crate) fn take_state_unit<T: DFlash2Target>(target: &mut T, key: &UnitKey) -> StateTake {
    use super::dflash2_state::disk::{Reclaimed, reclaim};
    match key {
        UnitKey::Store(id) => {
            let checkpoint = target
                .dflash2_states()
                .store
                .get(*id)
                .is_some_and(|entry| entry.checkpoint());
            match reclaim(target, *id) {
                Reclaimed::Written => {
                    log_reclaim(GrowReclaim::StoreWrite);
                    StateTake::Took
                }
                Reclaimed::RamDropped => {
                    log_reclaim(GrowReclaim::Store);
                    StateTake::Took
                }
                Reclaimed::Erased => {
                    log_reclaim(GrowReclaim::Store);
                    let counters = &mut target.dflash2_states_mut().counters;
                    if checkpoint {
                        counters.checkpoint_evictions += 1;
                    } else {
                        counters.evictions += 1;
                    }
                    StateTake::Took
                }
                Reclaimed::Pending => StateTake::Pending,
                Reclaimed::Nothing => StateTake::Nothing,
            }
        }
        // The chain's pages free once its copies land: the writer takes
        // what its queue holds now.
        UnitKey::StoreChain(_) => {
            log_reclaim(GrowReclaim::StoreChain);
            super::dflash2_state::disk::tick(
                target,
                super::dflash2_state::disk::Tick::Wait {
                    deadline: std::time::Instant::now(),
                },
            );
            StateTake::Pending
        }
        UnitKey::Ladder { .. } | UnitKey::History => {
            let dropped = target.drop_gdn_state(key);
            if !dropped {
                return StateTake::Nothing;
            }
            log_reclaim(if matches!(key, UnitKey::History) {
                GrowReclaim::History
            } else {
                GrowReclaim::Ladder
            });
            StateTake::Took
        }
        UnitKey::Owner(_) | UnitKey::Extent | UnitKey::DeadLeaves | UnitKey::Vision(_) => {
            StateTake::Nothing
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    /// A claim's retry sleeps only what the state tier's wait left of its
    /// interval, on the governor's clock: a wait that spent 80 ms of 100
    /// sleeps 20, one that spent it all sleeps none.
    /// Mutation RS2 (the whole interval is slept again) → red.
    #[test]
    fn a_retry_sleeps_what_the_tier_wait_left() {
        assert!((retry_left(0.1, 5.0, 5.08) - 0.02).abs() < 1e-9);
        assert_eq!(retry_left(0.1, 5.0, 5.2), 0.0);
        assert_eq!(retry_left(0.1, 5.0, 5.0), 0.1);
    }

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
                backing: crate::transformer::paged_kv_cache_adapter::ColdContextBacking::KvOnly,
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
        let _scope = crate::engine::governor::governor::Scope::enter(claim.claim());
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
}
