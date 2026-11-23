//! A disk-tier restore beside other turns (Splash parks a restoring
//! request, vLLM V1 `WAITING_FOR_REMOTE_KVS`): the turn waits while the
//! tier's reader brings its chain back, then lands through the plain prime
//! on a hot chain. Nothing past the hot base is published before the batch
//! commit, and no landing walks the tier on the model thread.

use std::time::Instant;

use super::*;
use crate::engine::hybrid_scheduler::{ScheduledPrefixAdmission, ScheduledRestoreResult};
use crate::models::qwen3_5::dflash2_state::disk;
use crate::models::qwen3_5::dflash2_state::store::{Lineage, StoreHit};
use crate::models::qwen3_5::dflash2_turn::{DFlash2Hit, Keep, claimed};
use crate::transformer::paged_kv_cache_adapter::{
    ChainPrefetch, ChainPrefetchPoll, ChainPrefetchStart, RestoreUploadBudget,
};

/// A parked turn's restore.
pub(crate) struct Qwen35Restore {
    phase: Phase,
    cache_salt: u64,
    /// A second read was started (one at most).
    retried: bool,
    parked_at: Instant,
    read_wait: Option<disk::ReadWait>,
    /// When the landing first found the staging buffer out.
    read_wait_since: Option<f64>,
}

enum Phase {
    Reading(ChainPrefetch),
    Landed(ChainPrefetch),
    /// The read failed: the landing, in the turn's context, chooses again.
    Lost,
    /// The state's chain cannot come back beside other turns: the turn
    /// primes from nothing.
    Cold,
}

/// How a landing primes.
enum Landing {
    /// At the landed chain's state: the armed lookup takes it hot.
    Armed,
    /// The plain prime, which reads no block file.
    Plain,
    /// Another read for the chain of the state the prime takes now.
    Read(Box<ChainPrefetch>),
    Cold,
}

impl Qwen35Restore {
    /// The data the landing needs is in place.
    pub(crate) fn lands(&self) -> bool {
        !matches!(self.phase, Phase::Reading(_))
    }

    /// Test-only: the blocks uploaded so far, and whether the chain landed.
    #[cfg(test)]
    pub(crate) fn progress_for_test(&self) -> (usize, bool) {
        match &self.phase {
            Phase::Reading(prefetch) => (prefetch.stats().uploaded_blocks, false),
            Phase::Landed(prefetch) => (prefetch.restored_blocks(), true),
            Phase::Lost | Phase::Cold => (0, false),
        }
    }

    pub(crate) fn held_blocks(&self) -> u32 {
        match &self.phase {
            Phase::Reading(prefetch) | Phase::Landed(prefetch) => prefetch.held_blocks(),
            Phase::Lost | Phase::Cold => 0,
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Each restore poll waits for its reader's last chunk first.
    pub(crate) static AWAIT_READERS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The next landings are refused memory, as a claim the ledger makes
    /// wait for what lands.
    pub(crate) static REFUSED_LANDINGS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// Second reads started, and landings that primed cold.
    pub(crate) static RETRIES: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    pub(crate) static COLD_LANDINGS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn count(counter: &'static std::thread::LocalKey<std::cell::Cell<u32>>) {
    counter.with(|count| count.set(count.get() + 1));
}

/// A landing [`REFUSED_LANDINGS`] refuses, as its prime's claim would be.
#[cfg(test)]
fn scripted_landing_refusal() -> Result<()> {
    let refused = REFUSED_LANDINGS.with(|left| {
        let refused = left.get() > 0;
        left.set(left.get().saturating_sub(1));
        refused
    });
    if !refused {
        return Ok(());
    }
    crate::engine::governor::governor::note_lane_denial(
        crate::engine::governor::allocate::Denial {
            pending: true,
            ..Default::default()
        },
    );
    Err(Error::from_reason(
        "paged allocation blocked: a scripted landing refusal",
    ))
}

impl Qwen35Inner {
    /// A start beside other turns: a draft load's text turn that reuses
    /// the cache and resumes at a stored state whose chain is partly on
    /// the tier parks while the chain is read. Anything else primes as a
    /// lone start does: no reuse, a whole-turn draft prime, media or
    /// rope-delta lineage, a live continuation, no state tier.
    pub(crate) fn prepare_parked_prefix(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        reuse_cache: bool,
        cache_salt: u64,
        block_size: u32,
    ) -> Result<ScheduledPrefixAdmission<Qwen35PrefixState, Qwen35Restore>> {
        self.activate_paged_seq(seq_id)?;
        let start = if self.parks_restore(tokens, reuse_cache) {
            self.start_state_prefetch(seq_id, tokens, cache_salt)?
        } else {
            None
        };
        if let Some(ChainPrefetchStart::Unavailable) = start {
            return self
                .prime_cold(tokens, cache_salt)
                .map(ScheduledPrefixAdmission::Ready);
        }
        if let Some(ChainPrefetchStart::Started(prefetch)) = start {
            if inference_trace_enabled() {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-dense parked_restore_start seq={seq_id} depth={} hot_blocks={} reserved={}",
                    prefetch.depth(),
                    prefetch.hot.len(),
                    prefetch.reserved_blocks(),
                ));
            }
            return Ok(ScheduledPrefixAdmission::Waiting {
                provisional: Qwen35PrefixState {
                    effective_cached_prefix_len: 0,
                    suffix_len: tokens.len(),
                    full_tokens: tokens.to_vec(),
                    cache_salt,
                    gdn_prefix_already_primed: false,
                },
                restore: Qwen35Restore {
                    phase: Phase::Reading(*prefetch),
                    cache_salt,
                    retried: false,
                    parked_at: Instant::now(),
                    read_wait: None,
                    read_wait_since: None,
                },
            });
        }
        self.prime_prefix_state(tokens, reuse_cache, block_size as usize, &[], cache_salt)
            .map(ScheduledPrefixAdmission::Ready)
    }

    fn parks_restore(&self, tokens: &[u32], reuse_cache: bool) -> bool {
        if self.is_moe()
            || !reuse_cache
            || self.dflash2.is_none()
            || self.draft_lane_turn()
            || self.cached_rope_deltas.unwrap_or(0) != 0
            || self.cached_image_key.is_some()
            || !self.cached_paged_image_token_positions.is_empty()
            || self.dflash2_states.disk.is_none()
        {
            return false;
        }
        self.paged_adapter.as_ref().is_some_and(|adapter| {
            !(adapter.is_live_for_continue() && tokens.starts_with(adapter.request_tokens()))
        })
    }

    /// The stored state the plain prime of `tokens` resumes at
    /// (`scheduled_state_hit`'s pick, read only).
    fn resume_hit(&self, tokens: &[u32], cache_salt: u64) -> Option<StoreHit> {
        self.dflash2_states
            .store
            .deepest_accepted_within(
                Lineage::text(tokens, cache_salt),
                tokens.len().saturating_sub(1),
                |id| disk::restorable(self, id),
            )
            .filter(|hit| hit.depth > 0)
    }

    /// The reader of the chain under the state the prime resumes at, its
    /// state object held on the tier; `None` when no stored state backs a
    /// resume.
    fn begin_state_prefetch(
        &mut self,
        tokens: &[u32],
        cache_salt: u64,
    ) -> Option<ChainPrefetchStart> {
        let hit = self.resume_hit(tokens, cache_salt)?;
        let state_key = disk::state_object_key(&self.dflash2_states, hit.id);
        let adapter = self.paged_adapter.as_mut()?;
        let keys = engine::build_paged_extra_keys(tokens.len(), adapter.block_size(), &[]);
        let cap = u32::try_from(tokens.len()).ok()?.saturating_sub(1);
        Some(adapter.begin_chain_prefetch(
            tokens,
            cap,
            hit.depth,
            mlx_paged_attn::PrefixKeys::PerBlock(&keys),
            cache_salt,
            state_key,
        ))
    }

    /// A start's prefetch. Destinations the pool has no room for are an
    /// allocation of the turn's claim: the ledger frees what it can and the
    /// prefetch runs again; a refusal left then is the scheduler's to
    /// judge, and the start waits as any refused start does.
    /// Never `Refused`.
    fn start_state_prefetch(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        cache_salt: u64,
    ) -> Result<Option<ChainPrefetchStart>> {
        let start = self.begin_state_prefetch(tokens, cache_salt);
        if !matches!(start, Some(ChainPrefetchStart::Refused { .. })) {
            return Ok(start);
        }
        let block_size = self
            .paged_adapter
            .as_ref()
            .map_or(0, PagedKVCacheAdapter::block_size);
        let keys = engine::build_paged_extra_keys(tokens.len(), block_size, &[]);
        let keep = Keep {
            tokens,
            keys: Some((&keys, cache_salt)),
        };
        claimed(self, keep, |inner| {
            match inner.begin_state_prefetch(tokens, cache_salt) {
                Some(ChainPrefetchStart::Refused { blocks }) => {
                    if let Some(adapter) = inner.paged_adapter.as_ref() {
                        adapter.refuse_restore(seq_id, blocks);
                    }
                    Err(Error::from_reason(format!(
                        "paged cache could not reserve {blocks} blocks for a disk restore"
                    )))
                }
                start => Ok(start),
            }
        })
    }

    /// A prime from nothing: no lookup, no walk.
    fn prime_cold(&mut self, tokens: &[u32], cache_salt: u64) -> Result<Qwen35PrefixState> {
        crate::models::qwen3_5::dflash2_turn::prime_scheduled_draft_lane(
            self,
            tokens,
            true,
            cache_salt,
            |inner, _| inner.prime_dense_prefix(tokens, cache_salt, Some(DFlash2Hit::Cold)),
        )
    }

    /// Every block below `hit`'s state is hot.
    fn chain_hot(&self, hit: &StoreHit) -> bool {
        let (Some(adapter), Some(entry)) = (
            self.paged_adapter.as_ref(),
            self.dflash2_states.store.get(hit.id),
        ) else {
            return false;
        };
        entry
            .chain()
            .iter()
            .all(|&hash| adapter.prefix_cached(hash))
    }

    /// Beside other turns a landing primes hot only. A state
    /// whose chain is not hot gets one more read, else the turn primes
    /// cold.
    fn choose_landing(
        &mut self,
        tokens: &[u32],
        cache_salt: u64,
        landed_depth: Option<usize>,
        retried: bool,
    ) -> Landing {
        let Some(hit) = self.resume_hit(tokens, cache_salt) else {
            return Landing::Plain;
        };
        if landed_depth == Some(hit.depth) {
            return Landing::Armed;
        }
        if self.chain_hot(&hit) {
            return Landing::Plain;
        }
        if !retried
            && let Some(ChainPrefetchStart::Started(prefetch)) =
                self.begin_state_prefetch(tokens, cache_salt)
        {
            return Landing::Read(prefetch);
        }
        Landing::Cold
    }

    fn retry_read(restore: &mut Qwen35Restore, mut prefetch: ChainPrefetch) {
        prefetch.mark_retried();
        restore.retried = true;
        restore.phase = Phase::Reading(prefetch);
        #[cfg(test)]
        count(&RETRIES);
    }

    /// One poll of a parked restore's read within the pass's `uploads`.
    pub(crate) fn poll_parked_restore(
        &mut self,
        restore: &mut Qwen35Restore,
        uploads: &mut RestoreUploadBudget,
    ) -> Result<Option<ScheduledRestoreResult<Qwen35PrefixState>>> {
        let Phase::Reading(prefetch) = &mut restore.phase else {
            return Ok(None);
        };
        #[cfg(test)]
        if AWAIT_READERS.with(std::cell::Cell::get) {
            prefetch.await_reader_for_test();
        }
        let adapter = self
            .paged_adapter
            .as_ref()
            .ok_or_else(|| Error::from_reason("a parked restore lost its paged adapter"))?;
        let polled = adapter.poll_chain_prefetch(prefetch, uploads);
        if polled == ChainPrefetchPoll::Pending {
            return Ok(None);
        }
        let Phase::Reading(prefetch) = std::mem::replace(&mut restore.phase, Phase::Cold) else {
            return Ok(None);
        };
        if matches!(polled, ChainPrefetchPoll::Landed { .. }) {
            restore.phase = Phase::Landed(prefetch);
            return Ok(None);
        }
        drop(prefetch);
        restore.phase = Phase::Lost;
        Ok(None)
    }

    /// The landing of a restore whose data is in place, with the turn's
    /// sequence active and its admission context installed. `Ok(None)`:
    /// it stays parked (a second read started, or the state read would
    /// wait for the staging buffer).
    pub(crate) fn land_parked_restore(
        &mut self,
        seq_id: SeqId,
        restore: &mut Qwen35Restore,
        tokens: &[u32],
    ) -> Result<Option<ScheduledRestoreResult<Qwen35PrefixState>>> {
        let landed_depth = match &restore.phase {
            Phase::Reading(_) => return Ok(None),
            Phase::Landed(prefetch) => Some(prefetch.depth()),
            Phase::Lost | Phase::Cold => None,
        };
        let salt = restore.cache_salt;
        let landing = if matches!(restore.phase, Phase::Cold) {
            Landing::Cold
        } else {
            self.choose_landing(tokens, salt, landed_depth, restore.retried)
        };
        let landing = match landing {
            Landing::Read(prefetch) => {
                Self::retry_read(restore, *prefetch);
                return Ok(None);
            }
            Landing::Cold => Landing::Cold,
            primed => {
                if self.state_read_waits(tokens, salt) {
                    let now = self.memory_registration.as_ref().map_or_else(
                        || restore.parked_at.elapsed().as_secs_f64(),
                        |guard| guard.governor().now(),
                    );
                    let since = *restore.read_wait_since.get_or_insert(now);
                    if now - since < crate::engine::governor::wait::RESOURCE_WAIT_SECONDS {
                        if restore.read_wait.is_none() {
                            restore.read_wait = disk::read_wait(&self.dflash2_states);
                        }
                        return Ok(None);
                    }
                    // The buffer stayed out past the wait limit: as a state
                    // read that times out, prime from nothing.
                    Landing::Cold
                } else {
                    primed
                }
            }
        };
        restore.read_wait = None;
        let block_size = self
            .paged_adapter
            .as_ref()
            .map(PagedKVCacheAdapter::block_size)
            .ok_or_else(|| Error::from_reason("a parked restore lost its paged adapter"))?;
        let (path, prefix) = match landing {
            Landing::Cold => {
                #[cfg(test)]
                {
                    scripted_landing_refusal()?;
                    count(&COLD_LANDINGS);
                }
                ("cold", self.prime_cold(tokens, salt)?)
            }
            Landing::Armed => {
                if let (Phase::Landed(prefetch), Some(adapter)) =
                    (&restore.phase, self.paged_adapter.as_mut())
                {
                    adapter.arm_landed_chain(prefetch);
                }
                let primed = self.armed_prime(tokens, block_size as usize, salt);
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    adapter.disarm_landed_chain();
                }
                ("armed", primed?)
            }
            Landing::Plain | Landing::Read(_) => {
                #[cfg(test)]
                scripted_landing_refusal()?;
                let prefix =
                    self.prime_prefix_state(tokens, true, block_size as usize, &[], salt)?;
                ("plain", prefix)
            }
        };
        let bytes_restored = match (&restore.phase, path) {
            (Phase::Landed(prefetch), "armed") => prefetch.bytes_restored,
            _ => 0,
        };
        let wait = restore.parked_at.elapsed();
        if inference_trace_enabled() {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense parked_restore_land seq={seq_id} path={path} cached={} retried={} wait_ms={:.3}",
                prefix.effective_cached_prefix_len,
                u8::from(restore.retried),
                wait.as_secs_f64() * 1000.0,
            ));
        }
        let materialized_blocks = self
            .paged_adapter
            .as_ref()
            .and_then(|adapter| adapter.block_table_for(seq_id))
            .map_or(0, |table| table.num_blocks() as u32);
        let profiler_prefill_tokens = u32::try_from(prefix.suffix_len).unwrap_or(u32::MAX);
        Ok(Some(ScheduledRestoreResult {
            prefix,
            bytes_restored,
            wait,
            materialized_blocks,
            profiler_prefill_tokens,
            extra_prefill_breaks: Vec::new(),
        }))
    }

    /// The plain prime with the landed chain armed.
    fn armed_prime(
        &mut self,
        tokens: &[u32],
        block_size: usize,
        cache_salt: u64,
    ) -> Result<Qwen35PrefixState> {
        #[cfg(test)]
        scripted_landing_refusal()?;
        self.prime_prefix_state(tokens, true, block_size, &[], cache_salt)
    }

    /// The prime would read its state from the tier while the
    /// staging buffer is out.
    fn state_read_waits(&mut self, tokens: &[u32], cache_salt: u64) -> bool {
        let on_disk = self.resume_hit(tokens, cache_salt).is_some_and(|hit| {
            self.dflash2_states
                .store
                .get(hit.id)
                .is_some_and(|entry| entry.ram().is_none())
        });
        on_disk && !disk::staging_free(self)
    }
}
