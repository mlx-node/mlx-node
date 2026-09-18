//! SSD cold-tier machinery for `PagedKVCacheAdapter`.
//!
//! Extracted from `paged_kv_cache_adapter.rs`: every piece of the optional
//! SSD persistence tier that is a pure function of the cold-tier context plus
//! the shared block allocator / K/V pool lives here —
//!
//! - [`ColdTierContext`], the per-adapter handle (manager + fingerprint +
//!   optional [`mlx_paged_attn::ColdSidecarPolicy`]);
//! - [`ColdTierWalk`], the borrowed-field bundle both capture and restore
//!   walks run against, with `restore_extend` (incl. the reconcile-down a
//!   sidecar policy requires), `begin_restore_extend`, `capture_chain`,
//!   `kv_chain_upper_bound`, `deepest_backed_boundary`, and `record_decline`;
//! - [`ColdRestore`] / [`PendingColdRestore`] / [`PagedRestoreTicket`], the
//!   synchronous and asynchronous restore payloads;
//! - [`ColdCaptureBudget`] / [`ColdCaptureStop`] / [`ColdCaptureOutcome`],
//!   the capture-walk policy and reporting types;
//! - [`trace_cold_capture_walk`] and the `aux_prefix` latch helpers
//!   ([`aux_prefix_state_missing`], [`ensure_aux_prefix_primed`],
//!   [`confirm_aux_prefix_primed`]).
//!
//! ## What deliberately stays adapter-owned
//!
//! - The `aux_prefix_unbacked` LATCH itself, `restored_sidecar`,
//!   `cold_capture`, and `cold_capture_budget` are request-scoped fields on
//!   `PagedKVCacheAdapter` (and `PagedRequestState` while parked). The latch's
//!   enforcement points — `record_tokens`, `register_full_blocks_for_reuse*`,
//!   `finalize_turn_keep_live*` — are adapter methods that cannot move here,
//!   so the latch cannot either: the helpers below take it (and the fields
//!   they consult) by reference instead of owning it.
//! - `suppress_cold_restore_once` is armed/consumed by the adapter's
//!   reset/lookup methods and stays a plain adapter field.
//! - `prepare_turn_with_async_restore` / `poll_restore` orchestrate the
//!   request lifecycle (activation, block-table install, suffix allocation)
//!   around the restore; they remain adapter methods and merely consume
//!   [`PendingColdRestore`] / [`PagedRestoreTicket`].
//! - `native_pool_arrays` and every other write-ordering-coupled state stays
//!   adapter-owned; nothing here carries lazy graph outputs.
//!
//! The module is NOT macOS-gated: `mlx_paged_attn`'s cold-cache types are
//! cross-platform (only its `metal`/`extern_c` modules are gated), matching
//! the ungated status this code had inside the adapter.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mlx_paged_attn::{BlockAllocator, LayerKVPool, PhysicalBlock};

use crate::inference_trace::{enabled as inference_trace_enabled, write as write_inference_trace};

use super::paged_kv_cache_adapter::{PagedTurnPlanReason, SeqId};

/// SSD cold-tier handle for persisting full paged KV blocks across process
/// restarts. `fingerprint` binds every persisted block to one model + cache
/// layout identity; any drift (weights, config, pool geometry, dtype) must
/// produce a different fingerprint so stale blocks can never validate.
pub struct ColdTierContext {
    pub manager: std::sync::Arc<mlx_paged_attn::ColdCacheManager>,
    pub fingerprint: mlx_paged_attn::ColdCacheFingerprint,
    /// The auxiliary (non-KV) state this family REQUIRES at any boundary it
    /// resumes from, or `None` when the paged pool already holds every piece
    /// of per-token state the forward pass carries between turns.
    ///
    /// `None` is the dense-`qwen3` control: the restore walk behaves exactly
    /// as it did before sidecars existed. `Some(policy)` turns the walk into
    /// vLLM's reconcile-down — the candidate prefix is reduced to the deepest
    /// boundary a VALIDATED sidecar backs, and a boundary no sidecar backs
    /// restores nothing rather than handing back attention state whose
    /// recurrent half never existed.
    pub sidecar_policy: Option<mlx_paged_attn::ColdSidecarPolicy>,
}

/// The adapter fields the cold-tier restore and capture walks need, borrowed
/// as one bundle.
///
/// Both walks are shared verbatim between the uniform-`extra_keys` entry points
/// (`find_cached_prefix*` / `register_full_blocks_for_reuse`) and the per-block
/// ones (`find_cached_prefix_per_block*` /
/// `register_full_blocks_for_reuse_per_block`); the ONLY difference is how a
/// block index maps to its `extra_keys`, which callers supply as a closure. The
/// uniform side returns the same slice for every index, the per-block side
/// indexes its per-block vec.
///
/// This is a borrowed bundle rather than `&self` methods on the adapter because
/// the lookup path holds a `&mut` borrow of `self.block_table` across the
/// restore call; taking `&self` there would conflict, while borrowing the four
/// disjoint fields below does not.
pub(crate) struct ColdTierWalk<'a> {
    pub(crate) cold: &'a ColdTierContext,
    pub(crate) pool: &'a Arc<LayerKVPool>,
    pub(crate) allocator: &'a Arc<Mutex<BlockAllocator>>,
    pub(crate) block_size: u32,
}

/// Outcome of one cold-tier restore walk.
///
/// INVARIANT, and the whole point of this type: when `sidecar` is `Some`, it is
/// the validated auxiliary state for EXACTLY the boundary the returned `blocks`
/// end at (hot prefix + `blocks`). The two can never describe different
/// boundaries — every path that shortens `blocks` also re-derives (or drops)
/// the sidecar.
///
/// `sidecar` is always `None` for a family with no [`ColdSidecarPolicy`].
pub(crate) struct ColdRestore {
    pub(crate) blocks: Vec<Arc<PhysicalBlock>>,
    pub(crate) sidecar: Option<mlx_paged_attn::ColdSidecar>,
}

pub(crate) struct PendingColdRestore {
    pub(crate) job: mlx_paged_attn::ColdRestoreBatchJob,
    pub(crate) reserved: Vec<Arc<PhysicalBlock>>,
    pub(crate) identities: Vec<mlx_paged_attn::RestorePrefixIdentity>,
}

impl ColdRestore {
    /// Restore nothing: the hot hit stands unextended and no state is handed
    /// back. Every fail-closed exit returns this.
    fn miss() -> Self {
        Self {
            blocks: Vec::new(),
            sidecar: None,
        }
    }
}

impl ColdTierWalk<'_> {
    /// Prepare a dense-family cold restore without performing filesystem I/O
    /// or Metal work on the model thread. The consecutive on-disk chain is
    /// index-probed first, every destination is reserved next, and only then
    /// is the background read launched.
    pub(crate) fn begin_restore_extend<'k>(
        &self,
        lookup_tokens: &[u32],
        hot_cached_tokens: usize,
        cache_salt: u64,
        extra_keys_for: impl Fn(usize) -> Option<&'k [u64]>,
        hot_hashes: impl FnOnce(usize) -> Vec<u64>,
    ) -> Option<PendingColdRestore> {
        if self.cold.sidecar_policy.is_some() {
            return None;
        }
        let bs = self.block_size as usize;
        if bs == 0 {
            return None;
        }
        let mut full_blocks = lookup_tokens.len() / bs;
        let base = hot_cached_tokens.min(lookup_tokens.len()) / bs;
        if base >= full_blocks {
            return None;
        }
        let hot = hot_hashes(full_blocks);
        full_blocks = full_blocks.min(hot.len());
        if base >= full_blocks {
            return None;
        }

        let mut parent_key = None;
        for index in 0..base {
            let extra_keys = extra_keys_for(index)?;
            let tokens = lookup_tokens.get(index * bs..(index + 1) * bs)?;
            parent_key = Some(mlx_paged_attn::ColdCacheKey::chain(
                mlx_paged_attn::ColdGroup::Kv,
                self.cold.fingerprint,
                parent_key,
                tokens,
                extra_keys,
                cache_salt,
                index,
            ));
        }
        let limit = self.kv_chain_upper_bound(
            lookup_tokens,
            cache_salt,
            &extra_keys_for,
            base,
            full_blocks,
            parent_key,
        );
        if limit <= base {
            return None;
        }

        let mut keys = Vec::with_capacity(limit - base);
        let mut identities = Vec::with_capacity(limit - base);
        for index in base..limit {
            let extra_keys = extra_keys_for(index)?;
            let tokens = lookup_tokens.get(index * bs..(index + 1) * bs)?;
            let key = mlx_paged_attn::ColdCacheKey::chain(
                mlx_paged_attn::ColdGroup::Kv,
                self.cold.fingerprint,
                parent_key,
                tokens,
                extra_keys,
                cache_salt,
                index,
            );
            keys.push(key);
            identities.push(mlx_paged_attn::RestorePrefixIdentity {
                hot_hash: hot[index],
                tokens: tokens.to_vec(),
                parent_hot_hash: if index == 0 { 0 } else { hot[index - 1] },
                extra_keys: extra_keys.to_vec(),
                cache_salt,
                block_index: index,
            });
            parent_key = Some(key);
        }

        let mut reserved = Vec::with_capacity(keys.len());
        {
            let mut allocator = self
                .allocator
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for _ in 0..keys.len() {
                let Some(block) = allocator.allocate() else {
                    for block in reserved.drain(..) {
                        allocator.free(block);
                    }
                    return None;
                };
                reserved.push(block);
            }
        }
        let job =
            match self
                .cold
                .manager
                .begin_restore_batch(self.pool, keys, self.cold.fingerprint)
            {
                Ok(job) => job,
                Err(_) => {
                    let mut allocator = self
                        .allocator
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    for block in reserved {
                        allocator.free(block);
                    }
                    return None;
                }
            };
        Some(PendingColdRestore {
            job,
            reserved,
            identities,
        })
    }

    /// SSD cold-tier restore on a hot-cache prefix miss: for each full block
    /// the in-memory lookup did NOT cover, recompute the persisted chain (the
    /// capture contract below — parent-linked per block, `cache_salt` mixed
    /// into block 0 only) and transactionally restore it into a fresh physical
    /// slot before falling back to prefill. Returns the restored blocks, in
    /// order, to append to the hot hit.
    ///
    /// Fail-open everywhere: the first block that misses on disk, fails
    /// validation, or cannot be uploaded stops the extension and leaves the hot
    /// hit untouched. The hot identity each restored block is published under
    /// comes from `hot_hashes`, so a subsequent lookup on this same allocator
    /// serves it directly.
    ///
    /// `hot_hashes` is invoked at most once, with the full-block count, and
    /// must return the hot chain hashes for exactly those blocks. A SHORT
    /// return CAPS the walk: it means the caller could not establish a cache
    /// identity past that point — for the per-block path, that the per-block
    /// `extra_keys` vec ran out — and restoring under a guessed identity would
    /// publish blocks a later lookup could serve for the wrong keys. This is
    /// the same rule the hot allocator applies when `extra_keys_per_block` runs
    /// short (`find_longest_cache_hit_per_block` breaks at the first block
    /// without keys, keeping the blocks before it).
    ///
    /// ## Reconcile-down (families with a [`ColdSidecarPolicy`])
    ///
    /// Paged KV is only half the state of a hybrid family: GDN recurrent state
    /// (`qwen3_5`) and sliding-window `RotatingKVCache` state (`gemma4`) live
    /// OUTSIDE the pool, so a restored KV prefix whose auxiliary state is
    /// missing describes a model state that never existed. vLLM's rule for the
    /// same hazard is that each cache group may only REDUCE the candidate
    /// length (`vllm/v1/core/sched/scheduler.py`,
    /// `vllm/v1/core/kv_cache_coordinator.py`): "No external tokens back the
    /// deeper local hit, so its resume boundary would have no valid Mamba
    /// state. Reconcile to the boundary every group agrees on."
    ///
    /// So when a policy is present the walk runs in phases:
    ///
    ///  1. probe how far the persisted KV chain reaches (index only, no I/O);
    ///  2. descend from that ceiling to the deepest boundary a VALIDATED
    ///     sidecar backs — nothing backed means restore NOTHING, never a
    ///     "close enough" prefix;
    ///  3. restore exactly that many blocks;
    ///  4. if step 3 came up short (a block failed to decode, upload, or
    ///     publish), reconcile down AGAIN over what actually landed and free
    ///     the tail, so the returned sidecar always backs exactly the returned
    ///     prefix.
    ///
    /// The floor of every reconcile is the hot hit: this walk can only extend
    /// it, so it never claims to have reduced a prefix the in-memory cache
    /// already served.
    ///
    /// That leaves one thing this gate deliberately does NOT cover: a HOT hit
    /// is not gated here. A block that a backed restore published (or that
    /// phase 4 released — `free` decrefs it to a cache-only entry, it is not
    /// erased) stays in the allocator's prefix cache and a later lookup in the
    /// same process can serve it as a hot hit with no sidecar attached. The KV
    /// is valid; what is missing is the auxiliary half. So a hybrid family must
    /// still establish its own state for a hot prefix, or restart the turn cold
    /// via [`PagedKVCacheAdapter::restart_prepared_turn_cold_per_block`]. This
    /// walk changes only what the SSD tier is allowed to hand back; the hot
    /// half is covered by the `aux_prefix_unbacked` latch, which turns that
    /// obligation into a checked one (see its field doc).
    ///
    /// With `sidecar_policy: None` every phase above is skipped and the body is
    /// the pre-sidecar walk verbatim.
    pub(crate) fn restore_extend<'k>(
        &self,
        lookup_tokens: &[u32],
        hot_cached_tokens: usize,
        cache_salt: u64,
        extra_keys_for: impl Fn(usize) -> Option<&'k [u64]>,
        hot_hashes: impl FnOnce(usize) -> Vec<u64>,
    ) -> ColdRestore {
        let bs = self.block_size as usize;
        // `checked_div` folds the degenerate `block_size == 0` case into a
        // no-op: both counts become 0, so the extension loop never runs.
        let mut full_blocks = lookup_tokens.len().checked_div(bs).unwrap_or(0);
        // Blocks the hot hit already covers. This is the walk's floor — it can
        // extend the hot hit but never shorten it.
        let base = hot_cached_tokens
            .min(lookup_tokens.len())
            .checked_div(bs)
            .unwrap_or(0);
        let mut idx = base;
        let mut restored: Vec<Arc<PhysicalBlock>> = Vec::new();
        if idx >= full_blocks {
            return ColdRestore::miss();
        }

        let hot = hot_hashes(full_blocks);
        full_blocks = full_blocks.min(hot.len());
        if idx >= full_blocks {
            return ColdRestore::miss();
        }

        // Rebuild the cold keys of the already-covered leading blocks so the
        // first restore chains off the correct parent key.
        let mut parent_key: Option<mlx_paged_attn::ColdCacheKey> = None;
        for i in 0..idx {
            let Some(extra_keys) = extra_keys_for(i) else {
                self.record_decline(
                    "parent_chain_unavailable",
                    base,
                    full_blocks,
                    full_blocks,
                    bs,
                    lookup_tokens.len(),
                );
                return ColdRestore::miss();
            };
            parent_key = Some(mlx_paged_attn::ColdCacheKey::chain(
                mlx_paged_attn::ColdGroup::Kv,
                self.cold.fingerprint,
                parent_key,
                &lookup_tokens[i * bs..(i + 1) * bs],
                extra_keys,
                cache_salt,
                i,
            ));
        }

        // Phases 1-2: reduce the candidate to a boundary the family's auxiliary
        // state actually backs, BEFORE any Metal blit or hot-cache publish, so
        // an unbacked prefix is never materialized in the first place.
        let mut limit = full_blocks;
        let mut sidecar = None;
        if let Some(policy) = self.cold.sidecar_policy.as_ref() {
            let ceiling = self.kv_chain_upper_bound(
                lookup_tokens,
                cache_salt,
                &extra_keys_for,
                base,
                full_blocks,
                parent_key,
            );
            let Some((backed, state)) = self.deepest_backed_boundary(
                policy,
                lookup_tokens,
                cache_salt,
                &extra_keys_for,
                base,
                ceiling,
            ) else {
                // The silent zero this whole counter exists for. Both probes
                // this verdict rests on (`contains` / `contains_in`) are
                // side-effect free by contract, `load_sidecar` and
                // `restore_block` are never reached, and the walk returns
                // above its own trace line — so a refused restore moved no
                // counter and printed nothing, and the tier reported `0/0`
                // exactly like a turn that never opened it.
                self.record_decline(
                    "no_backed_boundary",
                    base,
                    ceiling,
                    full_blocks,
                    bs,
                    lookup_tokens.len(),
                );
                return ColdRestore::miss();
            };
            limit = backed;
            sidecar = Some(state);
        }

        // The restore loop has no budget and deliberately gets none: capping it
        // would cap REUSE, which is the whole feature. It is timed instead,
        // because the per-block restore cost is what decides whether reuse pays
        // at all — a block restored slower than the prefill that would have
        // recomputed it is a loss no coverage can fix.
        let restore_started = Instant::now();
        while idx < limit {
            let Some(extra_keys) = extra_keys_for(idx) else {
                break;
            };
            let toks = &lookup_tokens[idx * bs..(idx + 1) * bs];
            let key = mlx_paged_attn::ColdCacheKey::chain(
                mlx_paged_attn::ColdGroup::Kv,
                self.cold.fingerprint,
                parent_key,
                toks,
                extra_keys,
                cache_salt,
                idx,
            );
            let identity = mlx_paged_attn::RestorePrefixIdentity {
                hot_hash: hot[idx],
                tokens: toks.to_vec(),
                parent_hot_hash: if idx == 0 { 0 } else { hot[idx - 1] },
                extra_keys: extra_keys.to_vec(),
                cache_salt,
                block_index: idx,
            };
            match self.cold.manager.restore_block(
                self.pool,
                self.allocator,
                key,
                self.cold.fingerprint,
                &identity,
            ) {
                Some(block) => {
                    restored.push(block);
                    parent_key = Some(key);
                    idx += 1;
                }
                None => break,
            }
        }
        if inference_trace_enabled() {
            let elapsed_ms = restore_started.elapsed().as_secs_f64() * 1000.0;
            write_inference_trace(format_args!(
                "[MLX_TRACE] paged cold_restore_walk blocks={} base={} limit={} elapsed_ms={:.3} per_block_ms={:.3}",
                restored.len(),
                base,
                limit,
                elapsed_ms,
                if restored.is_empty() {
                    0.0
                } else {
                    elapsed_ms / restored.len() as f64
                },
            ));
        }

        // Phase 4: the restore stopped short of the boundary the sidecar backs
        // (a block failed to decode, allocate, upload, or publish). The state
        // in hand is for a prefix we do not have, so reconcile down again over
        // the blocks that actually landed and release the tail. Fail-closed:
        // when nothing shorter is backed either, the whole extension is
        // dropped rather than returned without state.
        if let Some(policy) = self.cold.sidecar_policy.as_ref()
            && idx < limit
        {
            let keep = match self.deepest_backed_boundary(
                policy,
                lookup_tokens,
                cache_salt,
                &extra_keys_for,
                base,
                idx,
            ) {
                Some((backed, state)) => {
                    sidecar = Some(state);
                    backed
                }
                None => {
                    sidecar = None;
                    base
                }
            };
            // `keep >= base` by construction (`deepest_backed_boundary` only
            // returns counts above its floor), so this can never exceed what
            // was restored; the clamp keeps a future contract slip a no-op
            // instead of an underflow panic.
            let drop_count = (base + restored.len())
                .saturating_sub(keep)
                .min(restored.len());
            if drop_count > 0 {
                let tail = restored.split_off(restored.len() - drop_count);
                let mut guard = self
                    .allocator
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                for block in tail {
                    guard.free(block);
                }
                // Everything this walk restored was handed back to the
                // allocator, so it extends the hot hit by nothing. Unlike the
                // refusals above this one has already paid for real `hits` and
                // `bytes_restored`, which is what makes it the more misleading
                // of the two: the tier reports blocks restored on a turn that
                // reuses none of them.
                if restored.is_empty() {
                    self.record_decline(
                        "restore_short_of_boundary",
                        base,
                        keep,
                        limit,
                        bs,
                        lookup_tokens.len(),
                    );
                }
            }
        }

        ColdRestore {
            blocks: restored,
            sidecar,
        }
    }

    /// Count one restore this walk refused to serve, and trace why.
    ///
    /// The counter (`ColdCacheStats::restore_declines`) is what makes a refusal
    /// exist at all for anything downstream: `hits` and `misses` count per-block
    /// lookups, a refusal happens instead of those lookups, so before this a
    /// refused restore was reported as `0 hits / 0 misses` — the same row a
    /// turn that never touched the tier produces, which reads as "nothing ran"
    /// rather than "reuse was refused".
    ///
    /// The block geometry goes to the trace instead of to counters on purpose.
    /// A ceiling is a per-walk position, and summing positions across turns
    /// produces a number with no meaning; what the diagnosis needs is the
    /// single line where `ceiling` and `full_blocks` sit next to each other —
    /// a capture that anchors one block past the restore's reach shows up here
    /// as `ceiling` one short of the boundary it would need, on the turn it
    /// happened.
    ///
    /// `ceiling_tokens` is that comparison already made. It is the DEEPEST
    /// boundary this walk could name at all — every candidate
    /// `deepest_backed_boundary` probed was at or below it — and it is directly
    /// comparable with the `boundary_tokens=` a family's capture trace prints.
    /// A capture line one block above a decline line's `ceiling_tokens`, for the
    /// same prompt, IS the aligned-prompt gap: the lookup is capped at
    /// `prompt_len - 1` (`lookup_tokens`), so a boundary at `prompt_len` has no
    /// name on this side.
    fn record_decline(
        &self,
        reason: &str,
        base: usize,
        ceiling: usize,
        full_blocks: usize,
        block_size: usize,
        lookup_tokens: usize,
    ) {
        self.cold.manager.record_restore_decline();
        if inference_trace_enabled() {
            write_inference_trace(format_args!(
                "[MLX_TRACE] paged cold_restore_declined reason={} base={} ceiling={} full_blocks={} block_size={} ceiling_tokens={} lookup_tokens={}",
                reason,
                base,
                ceiling,
                full_blocks,
                block_size,
                ceiling.saturating_mul(block_size),
                lookup_tokens,
            ));
        }
    }

    /// How far the persisted KV chain reaches past the hot hit, as a block
    /// count, using the in-memory index only — no file I/O, no decode, and no
    /// hit/miss accounting (`contains` is explicitly side-effect free).
    ///
    /// This is the ceiling the sidecar descent starts from. Without it a
    /// sidecar recorded at a deep boundary would be selected even when the KV
    /// blocks under it were evicted, and the walk would blit and publish blocks
    /// it must then free again. The index can be optimistic (an externally
    /// deleted file leaves a stale entry), which only means the restore loop
    /// stops short and phase 4 reconciles — never that an unbacked prefix is
    /// returned.
    ///
    /// `parent_key` must be the KV chain key of block `base - 1` (`None` when
    /// `base == 0`), i.e. exactly what the leading-block rebuild produced.
    fn kv_chain_upper_bound<'k>(
        &self,
        lookup_tokens: &[u32],
        cache_salt: u64,
        extra_keys_for: &impl Fn(usize) -> Option<&'k [u64]>,
        base: usize,
        full_blocks: usize,
        mut parent_key: Option<mlx_paged_attn::ColdCacheKey>,
    ) -> usize {
        let bs = self.block_size as usize;
        let mut end = base;
        for i in base..full_blocks {
            let (Some(extra_keys), Some(toks)) =
                (extra_keys_for(i), lookup_tokens.get(i * bs..(i + 1) * bs))
            else {
                break;
            };
            let key = mlx_paged_attn::ColdCacheKey::chain(
                mlx_paged_attn::ColdGroup::Kv,
                self.cold.fingerprint,
                parent_key,
                toks,
                extra_keys,
                cache_salt,
                i,
            );
            if !self.cold.manager.contains(&key) {
                break;
            }
            parent_key = Some(key);
            end = i + 1;
        }
        end
    }

    /// The reconcile-down step: the DEEPEST block count in `(floor, ceiling]`
    /// whose auxiliary state is on disk and validates against `policy`, with
    /// that state. `None` means no boundary in range is backed, which callers
    /// must treat as "restore nothing" — never as "restore anyway".
    ///
    /// The sidecar chain is the KV chain recomputed under the sidecar group's
    /// own domain tag: identical per-block arguments (tokens, `extra_keys`,
    /// `cache_salt`, block index), different group. That is vLLM's
    /// `BlockHashWithGroupId` (`vllm/v1/core/kv_cache_utils.py`) — the group is
    /// part of the key — so a sidecar key can never collide with, or be
    /// decoded as, a KV block key. Gaps are fine: a boundary that was never
    /// captured still lets deeper boundaries derive, because the chain is pure
    /// computation over the prompt, not a walk over what exists on disk.
    ///
    /// Each candidate is index-probed before it is loaded, so boundaries that
    /// were simply never captured cost no I/O and register no miss; a probe
    /// that says present and then fails to load is a real miss (and, if the
    /// bytes were readable, a corruption) counted by
    /// [`mlx_paged_attn::ColdCacheManager::load_sidecar`].
    fn deepest_backed_boundary<'k>(
        &self,
        policy: &mlx_paged_attn::ColdSidecarPolicy,
        lookup_tokens: &[u32],
        cache_salt: u64,
        extra_keys_for: &impl Fn(usize) -> Option<&'k [u64]>,
        floor: usize,
        ceiling: usize,
    ) -> Option<(usize, mlx_paged_attn::ColdSidecar)> {
        let bs = self.block_size as usize;
        if bs == 0 || ceiling <= floor {
            return None;
        }
        let mut keys: Vec<mlx_paged_attn::ColdCacheKey> = Vec::with_capacity(ceiling);
        let mut parent: Option<mlx_paged_attn::ColdCacheKey> = None;
        for i in 0..ceiling {
            let (Some(extra_keys), Some(toks)) =
                (extra_keys_for(i), lookup_tokens.get(i * bs..(i + 1) * bs))
            else {
                // No cache identity for block `i`, so the chain — and every
                // boundary past it — cannot be derived at all.
                break;
            };
            let key = mlx_paged_attn::ColdCacheKey::chain(
                policy.group(),
                self.cold.fingerprint,
                parent,
                toks,
                extra_keys,
                cache_salt,
                i,
            );
            keys.push(key);
            parent = Some(key);
        }

        for count in (floor + 1..=keys.len()).rev() {
            let key = keys[count - 1];
            if !self.cold.manager.contains_in(&key, policy.group()) {
                continue;
            }
            // A boundary past `u32::MAX` tokens cannot be expressed in the
            // layout, so it cannot match; shallower candidates still can.
            let Ok(boundary) = u32::try_from(count.saturating_mul(bs)) else {
                continue;
            };
            if let Some(state) = self.cold.manager.load_sidecar(
                key,
                self.cold.fingerprint,
                &policy.expected_at(boundary),
            ) {
                return Some((count, state));
            }
        }
        None
    }

    /// SSD cold-tier capture, fail-open: a persistence error only means the
    /// next process recomputes this prefix. Keys follow the hot chain-hash
    /// contract — parent-linked per block, `cache_salt` mixed into block 0 only
    /// — so [`Self::restore_extend`] recomputes the identical chain.
    /// `contains` dedups re-publishes of a chain already on disk without
    /// touching Metal.
    ///
    /// Reports how many leading blocks the persisted chain now covers — every
    /// block that was already on disk or was accepted by the writer queue, up
    /// to the first one that was not. A family capturing an auxiliary sidecar
    /// alongside the chain must not anchor it deeper than this: a sidecar past
    /// the chain's break can never be selected on restore
    /// ([`Self::kv_chain_upper_bound`] caps the descent at the chain's reach),
    /// so writing it would only burn quota.
    ///
    /// # What bounds this walk
    ///
    /// `budget`, and only `budget`. Until this took a budget the walk was
    /// bounded by the writer queue refusing a block, which made the per-turn
    /// capture depth an emergent property of the filesystem rather than a
    /// policy — `(Q + 1) / (1 - Tc/Tw)` blocks, measured at ~12 on this
    /// machine — so an 8 K-token prompt needed ~40 turns to persist and the
    /// restored prefix measured a few percent of the prompt. Waiting a bounded
    /// time for a queue slot (`capture_and_enqueue_before`) instead of giving
    /// up on one decouples the depth from `Tw` entirely.
    ///
    /// # Why it still breaks rather than skipping
    ///
    /// A block that did not land ends the walk, and every deeper block is left
    /// for a later turn. Skipping it instead would buy nothing on the turn that
    /// hits it: [`Self::kv_chain_upper_bound`] and [`Self::restore_extend`]
    /// both stop at the first key that is absent, so the chain's REACH is the
    /// index of the first hole under either policy. It would only pay from the
    /// turn after — at the price of a full Metal blit per skipped block, all of
    /// them on the inference thread, all of them discarded. Under a budget
    /// there is no cheap refusal left to skip anyway: `Ok(false)` now means the
    /// deadline expired, which is exactly when the walk should stop.
    pub(crate) fn capture_chain<'k>(
        &self,
        request_tokens: &[u32],
        blocks_slice: &[Arc<PhysicalBlock>],
        cache_salt: u64,
        budget: ColdCaptureBudget,
        extra_keys_for: impl Fn(usize) -> Option<&'k [u64]>,
    ) -> ColdCaptureOutcome {
        let started = Instant::now();
        let deadline = started + budget.max_walk;
        let mut outcome = ColdCaptureOutcome::default();
        let bs = self.block_size as usize;
        if bs == 0 {
            return outcome;
        }
        let mut parent: Option<mlx_paged_attn::ColdCacheKey> = None;
        for (i, block) in blocks_slice.iter().enumerate() {
            // Both lookups are infallible under the callers' own
            // preconditions; `get` keeps a contract slip a graceful stop
            // instead of a panic.
            let (Some(extra_keys), Some(toks)) =
                (extra_keys_for(i), request_tokens.get(i * bs..(i + 1) * bs))
            else {
                break;
            };
            let key = mlx_paged_attn::ColdCacheKey::chain(
                mlx_paged_attn::ColdGroup::Kv,
                self.cold.fingerprint,
                parent,
                toks,
                extra_keys,
                cache_salt,
                i,
            );
            if !self.cold.manager.contains(&key) {
                // Budget checks guard the CAPTURE, not the free `contains`
                // skip above: re-walking a chain already on disk costs an
                // in-memory index probe per block and must not consume a
                // turn's capture depth, or a long persisted prefix would stop
                // the walk before it reached the first block that needs
                // writing.
                if outcome.enqueued >= budget.max_blocks {
                    outcome.stop = ColdCaptureStop::Budget;
                    break;
                }
                if Instant::now() >= deadline {
                    outcome.stop = ColdCaptureStop::Deadline;
                    break;
                }
                match self.cold.manager.capture_and_enqueue_before(
                    self.pool,
                    block,
                    key,
                    self.cold.fingerprint,
                    toks,
                    deadline,
                ) {
                    Ok(true) => outcome.enqueued += 1,
                    // The queue stayed full for the rest of the budget: the
                    // storage device, not the walk, is the bottleneck.
                    Ok(false) => {
                        outcome.stop = ColdCaptureStop::Deadline;
                        break;
                    }
                    // A failed blit leaves nothing to chain off. Descendants
                    // must not be persisted under a missing parent — that is a
                    // chain hole, and a hole is unrestorable past its index.
                    Err(_) => {
                        outcome.stop = ColdCaptureStop::Error;
                        break;
                    }
                }
            }
            parent = Some(key);
            outcome.blocks = i + 1;
        }
        outcome.elapsed = started.elapsed();
        outcome
    }
}

/// How much of the prompt one turn's cold-tier capture walk may persist.
///
/// Two independent bounds because they answer different failure modes.
/// `max_blocks` bounds the STEADY state: how fast the persisted chain is
/// allowed to ratchet up a long prompt, which is a trade of turn tail against
/// how many turns it takes before a restore covers anything worth having.
/// `max_walk` bounds the TAIL: it is what stops a stalled storage device, or a
/// filesystem so fast that the queue never pushes back, from turning a 64 K
/// first turn into a second of dead time after the last token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColdCaptureBudget {
    pub max_blocks: usize,
    pub max_walk: Duration,
}

impl Default for ColdCaptureBudget {
    /// The process-wide budget, from `MLX_COLD_CAPTURE_BLOCKS_PER_TURN` /
    /// `MLX_COLD_CAPTURE_BUDGET_MS`.
    fn default() -> Self {
        let (max_blocks, max_walk) = crate::cold_tier::cold_capture_budget();
        Self {
            max_blocks,
            max_walk,
        }
    }
}

/// Why [`ColdTierWalk::capture_chain`] stopped.
///
/// `End` and `Budget` are the healthy states — the walk ran out of prompt, or
/// spent its depth. `Deadline` means the writer could not keep up within
/// `max_walk`, so this turn ratcheted less than it was allowed to; `Error`
/// means a Metal blit failed. Both of the latter are visible per turn in the
/// `cold_capture_walk` trace line, and `Deadline` additionally warns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColdCaptureStop {
    /// Walked every full block of the request.
    #[default]
    End,
    /// Spent `max_blocks`.
    Budget,
    /// Ran out of `max_walk` waiting on the writer queue.
    Deadline,
    /// A capture failed; the chain must stay contiguous, so the walk stopped.
    Error,
}

impl ColdCaptureStop {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::End => "end",
            Self::Budget => "budget",
            Self::Deadline => "deadline",
            Self::Error => "error",
        }
    }
}

/// Outcome of one [`ColdTierWalk::capture_chain`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ColdCaptureOutcome {
    /// Leading blocks the persisted chain covers after this walk. Includes
    /// blocks that were already on disk, so it is the number a sidecar may
    /// anchor under — NOT the number this turn wrote.
    pub blocks: usize,
    /// Blocks this walk actually handed to the writer queue. What the budget
    /// counts, and what separates "the chain advanced" from "the chain was
    /// already there".
    pub enqueued: usize,
    pub elapsed: Duration,
    pub stop: ColdCaptureStop,
}

/// In-flight asynchronous cold restore, handed to the scheduler between
/// `prepare_turn_with_async_restore` (which builds it from a
/// [`PendingColdRestore`]) and `PagedKVCacheAdapter::poll_restore` (which
/// commits it).
///
/// The `job` / `reserved` / `identities` triple is the cold-tier payload;
/// the remaining fields carry the turn context the commit must re-apply
/// (request id, budget, prompt, hot-hit bookkeeping, plan reason). Dropping
/// an uncommitted ticket returns its reserved blocks to the shared
/// allocator.
pub(crate) struct PagedRestoreTicket {
    pub(crate) seq_id: SeqId,
    pub(crate) total_budget: u32,
    pub(crate) prompt_tokens: Vec<u32>,
    pub(crate) hot_cached_prefix_len: u32,
    pub(crate) hot_cached_blocks: usize,
    pub(crate) reason: PagedTurnPlanReason,
    pub(crate) job: mlx_paged_attn::ColdRestoreBatchJob,
    pub(crate) reserved: Option<Vec<Arc<PhysicalBlock>>>,
    pub(crate) identities: Vec<mlx_paged_attn::RestorePrefixIdentity>,
    pub(crate) allocator: Arc<Mutex<BlockAllocator>>,
}

impl PagedRestoreTicket {
    pub(crate) fn reserved_blocks(&self) -> u32 {
        self.reserved
            .as_ref()
            .map_or(0, |blocks| blocks.len().try_into().unwrap_or(u32::MAX))
    }
}

impl Drop for PagedRestoreTicket {
    fn drop(&mut self) {
        let Some(blocks) = self.reserved.take() else {
            return;
        };
        let mut allocator = self
            .allocator
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for block in blocks {
            allocator.free(block);
        }
    }
}

/// Publish one capture walk's outcome.
///
/// The trace line carries the whole ratchet: `enqueued` is what this turn
/// added, `blocks` is where the chain now reaches, and `stop` says which
/// bound ended it. A `deadline` stop additionally warns, because unlike the
/// other three it is not a policy decision — it means the storage device
/// could not absorb the turn's budget, so the chain ratcheted slower than
/// configured and the next restore covers less than it should.
pub(crate) fn trace_cold_capture_walk(
    entry: &str,
    outcome: ColdCaptureOutcome,
    budget: ColdCaptureBudget,
) {
    if inference_trace_enabled() {
        write_inference_trace(format_args!(
            "[MLX_TRACE] paged cold_capture_walk entry={} blocks={} enqueued={} stop={} elapsed_ms={:.3} budget_blocks={} budget_ms={}",
            entry,
            outcome.blocks,
            outcome.enqueued,
            outcome.stop.as_str(),
            outcome.elapsed.as_secs_f64() * 1000.0,
            budget.max_blocks,
            budget.max_walk.as_millis(),
        ));
    }
    if outcome.stop == ColdCaptureStop::Deadline {
        tracing::warn!(
            target: "mlx_core::paged::cold",
            "cold-tier capture walk hit its {} ms deadline after {} of {} budgeted blocks; \
             the persisted prefix will ratchet slower than configured",
            budget.max_walk.as_millis(),
            outcome.enqueued,
            budget.max_blocks,
        );
    }
}

/// Decide whether the prefix just handed back leaves an unmet auxiliary
/// obligation.
///
/// `false` — the only possible answer without a [`ColdSidecarPolicy`] —
/// means either the family keeps ALL of its cross-token state inside the
/// paged pool, or the restore walk handed back state that backs exactly the
/// prefix it returned.
///
/// This is a free function rather than a walk method because the latch it
/// feeds lives on the adapter (and is enforced by adapter methods that
/// cannot move): the three fields it consults are passed in, so the
/// decision stays colocated with the sidecar machinery while ownership
/// stays with the request state.
pub(crate) fn aux_prefix_state_missing(
    cold_tier: Option<&ColdTierContext>,
    cached_token_count: u32,
    restored_sidecar: Option<&mlx_paged_attn::ColdSidecar>,
) -> bool {
    let Some(cold) = cold_tier else {
        return false;
    };
    if cold.sidecar_policy.is_none() {
        return false;
    }
    if cached_token_count == 0 {
        return false;
    }
    match restored_sidecar {
        // `restore_extend` reconciles the prefix and the state together, so
        // this normally holds; re-checking keeps a future contract slip
        // fail-closed instead of silently resuming on the wrong boundary.
        Some(sidecar) => sidecar.layout.boundary_tokens != cached_token_count,
        None => true,
    }
}

/// The family acknowledges that it has established the auxiliary
/// (out-of-pool) state for exactly `primed_tokens` of cached prefix — GDN
/// recurrent state for `qwen3_5*`, sliding-window `RotatingKVCache` state
/// for `gemma4` — and clears the obligation the prefix lookup latched.
///
/// No-op unless the obligation is actually outstanding. When it IS
/// outstanding, `primed_tokens` must equal the prefix the adapter reported
/// (`cached_token_count`); a mismatch means the family primed a different
/// boundary than the one it is about to resume from, which is exactly the
/// corruption this gate exists to prevent, so it returns `Err` and leaves
/// the latch set.
///
/// Takes the latch by mutable reference — it stays adapter-owned (the
/// request-scoped `aux_prefix_unbacked` field) because every enforcement
/// point is an adapter method.
pub(crate) fn confirm_aux_prefix_primed(
    aux_prefix_unbacked: &mut bool,
    cached_token_count: u32,
    primed_tokens: u32,
) -> Result<(), String> {
    if !*aux_prefix_unbacked {
        return Ok(());
    }
    if primed_tokens != cached_token_count {
        return Err(format!(
            "confirm_aux_prefix_primed: family primed {primed_tokens} tokens of auxiliary \
             state but the request resumes from a {cached_token_count} token cached prefix. \
             The out-of-pool state must cover exactly the reused prefix."
        ));
    }
    *aux_prefix_unbacked = false;
    Ok(())
}

/// Fail closed on any operation that would build on — or publish — a
/// cached prefix whose auxiliary half nobody has established.
///
/// Takes the latch by value: this is a read-only check over adapter state.
pub(crate) fn ensure_aux_prefix_primed(
    aux_prefix_unbacked: bool,
    cached_token_count: u32,
    op: &str,
) -> Result<(), String> {
    if !aux_prefix_unbacked {
        return Ok(());
    }
    Err(format!(
        "{op}: this request resumed from a {cached_token_count} token cached K/V prefix whose \
         out-of-pool state (GDN recurrent / sliding-window) was not restored with it, and the \
         model never called confirm_aux_prefix_primed. Continuing would attend over K/V that no \
         recurrent state matches. Prime the prefix (or restart the turn cold) first."
    ))
}
