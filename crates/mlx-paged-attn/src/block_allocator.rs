//! Block allocator for PagedAttention KV cache
//!
//! Manages a pool of fixed-size physical blocks that can be allocated
//! to sequences on demand. Supports:
//! - Reference counting for copy-on-write (beam search)
//! - Prefix caching via content-based hashing
//! - LRU eviction for cache management
//!
//! # Prefix-cache reference invariant
//!
//! Every entry in `prefix_cache` corresponds to **one logical reference
//! held by the cache itself**. `register_prefix` increfs on the genuine
//! insertion path; every cache-removal path (LRU eviction, Case 1
//! stale-alias displacement, and `free()`'s ref_count→0 cleanup) is
//! responsible for releasing that reference. Idempotent refresh of an
//! already-present (block, hash) pair does NOT incref again — the
//! existing logical reference is preserved across the LRU bump.
//!
//! Consequence: callers do not need to manually `incref` blocks before
//! registering them — registration itself takes the cache's ref. Once
//! all external references are released via `free()`, the cache's ref
//! is what keeps the block alive until LRU eviction (or another
//! displacement path) decrefs it back to 0 and returns it to the pool.
//!
//! # Allocation under cache pressure
//!
//! When `free_blocks` is empty, `allocate` falls back to evicting the
//! LRU oldest cache-only block (one whose ref_count is exactly 1 — the
//! cache's own logical ref, with no live request holding it) to satisfy
//! the request. This mirrors vLLM's
//! `vllm/v1/core/block_pool.py:_maybe_evict_cached_block` pattern and
//! keeps the pool from going monotonically unreachable when many unique
//! prompts cycle through `register_prefix` + `free()` faster than they
//! age out by capacity.
//!
//! # SipHash collision limitation
//!
//! `hash_tokens` uses Rust's `DefaultHasher` (SipHash-1-3, u64 output).
//! Cryptographic collision resistance is NOT guaranteed. At 1024 cache
//! entries the birthday-paradox collision probability is ~1e-14, but
//! adversarial inputs OR very large caches could produce collisions.
//! When a collision occurs, two different token chains share the same
//! block hash; `find_longest_cache_hit` will return blocks from one
//! chain when the caller intended the other → silent KV corruption.
//!
//! Mitigations:
//! - `cache_full_blocks` aborts registration on the first colliding
//!   block (so we don't WRITE a corrupted chain), unless the existing
//!   entry carries verified identical block identity metadata. In that
//!   case the caller is recomputing an already-cached prefix and can
//!   continue publishing the new tail.
//! - `find_longest_cache_hit` verifies stored block identity metadata
//!   when available before returning a cache hit.
//! - For multi-tenant deployments and adversarial settings, switch to
//!   SHA-256 by replacing `hash_tokens`'s hasher (small mechanical
//!   change).
//! - For deterministic reproducibility across processes, switch to
//!   xxhash with a fixed seed (vLLM's default).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use hashlink::LinkedHashSet;

/// A physical block in GPU memory
#[derive(Debug)]
pub struct PhysicalBlock {
    /// Unique block ID (index into the cache tensor)
    pub block_id: u32,

    /// Reference count for copy-on-write semantics
    pub ref_count: Arc<AtomicU32>,

    /// Number of tokens actually stored in this block
    pub num_tokens: u32,
}

impl PhysicalBlock {
    /// Create a new physical block
    pub fn new(block_id: u32) -> Self {
        Self {
            block_id,
            ref_count: Arc::new(AtomicU32::new(1)),
            num_tokens: 0,
        }
    }

    /// Increment the reference count
    pub fn incref(&self) {
        self.ref_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Decrement the reference count, returns true if it reached zero
    pub fn decref(&self) -> bool {
        self.ref_count.fetch_sub(1, Ordering::SeqCst) == 1
    }

    /// Get the current reference count
    pub fn get_ref_count(&self) -> u32 {
        self.ref_count.load(Ordering::SeqCst)
    }

    /// Check if this block is shared (ref_count > 1)
    pub fn is_shared(&self) -> bool {
        self.get_ref_count() > 1
    }
}

// Note: PhysicalBlock intentionally does not implement Clone.
// Use Arc::clone() for Rust ownership, and incref()/decref() for
// copy-on-write reference counting (tracking how many sequences use this block).

/// Block allocator managing a pool of physical blocks
pub struct BlockAllocator {
    /// Queue of free block IDs
    free_blocks: VecDeque<u32>,

    /// All allocated blocks (block_id -> block)
    allocated: HashMap<u32, Arc<PhysicalBlock>>,

    /// Total number of blocks in the pool
    num_blocks: u32,

    /// Block size in tokens
    block_size: u32,

    /// Prefix cache: hash -> block for reuse
    prefix_cache: HashMap<u64, Arc<PhysicalBlock>>,

    /// Prefix-cache block identity metadata for entries registered
    /// through `cache_full_blocks`.
    ///
    /// Direct `register_prefix` callers do not provide enough context to
    /// populate this table, so absence means "unknown" rather than
    /// "matching". The metadata lets a cold-prefill replay skip an
    /// already-cached leading prefix while still rejecting true hash
    /// collisions.
    prefix_cache_identities: HashMap<u64, PrefixCacheBlockIdentity>,

    /// Reverse mapping: block_id -> hash (for cleanup during free)
    block_hashes: HashMap<u32, u64>,

    /// LRU order for prefix cache eviction (oldest first).
    ///
    /// Backed by an intrusive doubly-linked hash set (`hashlink`) rather
    /// than a `VecDeque` so that "touch" (move-to-back on cache hit) and
    /// single-entry removal are O(1) amortized instead of an O(n)
    /// `retain()` over the whole order. Mirrors vLLM's
    /// `FreeKVCacheBlockQueue` / `BlockPool.touch()` intent, without a
    /// hand-rolled unsafe intrusive list.
    lru_order: LinkedHashSet<u64>,

    /// Maximum entries in prefix cache
    max_prefix_cache_entries: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PrefixCacheBlockIdentity {
    token_ids: Vec<u32>,
    parent_hash: u64,
    extra_keys: Vec<u64>,
    cache_salt: u64,
    block_index: usize,
}

impl PrefixCacheBlockIdentity {
    fn new(
        token_ids: &[u32],
        parent_hash: u64,
        extra_keys: &[u64],
        cache_salt: u64,
        block_index: usize,
    ) -> Self {
        Self {
            token_ids: token_ids.to_vec(),
            parent_hash,
            extra_keys: extra_keys.to_vec(),
            cache_salt,
            block_index,
        }
    }

    fn matches(
        &self,
        token_ids: &[u32],
        parent_hash: u64,
        extra_keys: &[u64],
        cache_salt: u64,
        block_index: usize,
    ) -> bool {
        self.token_ids.as_slice() == token_ids
            && self.parent_hash == parent_hash
            && self.extra_keys.as_slice() == extra_keys
            && self.cache_salt == cache_salt
            && self.block_index == block_index
    }
}

impl BlockAllocator {
    /// Create a new block allocator
    ///
    /// # Arguments
    /// * `num_blocks` - Total number of blocks to manage
    /// * `block_size` - Number of tokens per block
    pub fn new(num_blocks: u32, block_size: u32) -> Self {
        let free_blocks: VecDeque<u32> = (0..num_blocks).collect();

        Self {
            free_blocks,
            allocated: HashMap::with_capacity(num_blocks as usize),
            num_blocks,
            block_size,
            prefix_cache: HashMap::new(),
            prefix_cache_identities: HashMap::new(),
            block_hashes: HashMap::new(),
            lru_order: LinkedHashSet::new(),
            // Scale the prefix-cache capacity to `num_blocks` so the cache
            // can accommodate the full live block set. `num_blocks` is the
            // natural upper bound — no more than that many blocks can ever
            // be live, so this can't cause additional eviction beyond what
            // `try_evict_lru_for_allocation` already performs on
            // physical-pool exhaustion. Per-instance overrides remain
            // available via `set_max_prefix_cache_entries`.
            max_prefix_cache_entries: num_blocks as usize,
        }
    }

    /// Allocate a new block.
    ///
    /// Pops from `free_blocks` first. If the free pool is empty, falls
    /// back to evicting the LRU oldest cache-only block (see
    /// `try_evict_lru_for_allocation`). Returns None only when neither
    /// path can produce a block (free pool empty AND every cached entry
    /// is in-use by a live request).
    pub fn allocate(&mut self) -> Option<Arc<PhysicalBlock>> {
        let block_id = if let Some(id) = self.free_blocks.pop_front() {
            id
        } else {
            self.try_evict_lru_for_allocation()?
        };
        let block = Arc::new(PhysicalBlock::new(block_id));
        self.allocated.insert(block_id, Arc::clone(&block));
        Some(block)
    }

    /// Attempt to evict the LRU oldest cache-only block to make room for a
    /// new allocation. Returns the freed `block_id` if successful.
    ///
    /// "Cache-only" means the block's `ref_count` is exactly 1 (the
    /// cache's own logical ref — no request handle is alive on the
    /// block). If the LRU oldest entry has `ref_count > 1` it's still
    /// in use by a request, so it can't be evicted; we walk forward
    /// through `lru_order` looking for the first cache-only candidate.
    /// If NO entry is cache-only, returns None.
    ///
    /// Mirrors vLLM `vllm/v1/core/block_pool.py:_maybe_evict_cached_block`.
    ///
    /// Side effects on success: the chosen entry is removed from
    /// `prefix_cache`, `block_hashes`, `lru_order`, AND `allocated`. Its
    /// ref_count is decremented from 1 → 0 (releasing the cache's
    /// logical ref). The block_id is NOT pushed onto `free_blocks` —
    /// `allocate` will reuse it directly via the returned id.
    fn try_evict_lru_for_allocation(&mut self) -> Option<u32> {
        // Scan the LRU order oldest-first for the first cache-only entry.
        // Unlike the old approach (cloning the ENTIRE `lru_order` into a
        // fresh `Vec` on every call), this only visits entries up to the
        // match and remembers a single `u64` — no whole-list allocation.
        // The scan itself still has to walk past any entries that are
        // in-use (ref_count > 1); that's inherent to "oldest cache-only
        // wins" eviction, not an artifact of the container type.
        let mut evict_hash = None;
        for &hash in self.lru_order.iter() {
            match self.prefix_cache.get(&hash) {
                Some(block) if block.get_ref_count() == 1 => {
                    evict_hash = Some(hash);
                    break;
                }
                // ref_count > 1: in use by a live request — keep scanning.
                // None: desync (shouldn't happen, but defensive) — keep scanning.
                _ => continue,
            }
        }
        let hash = evict_hash?;

        // Re-borrow now that the scan's immutable borrow of `lru_order` has
        // ended, so we're free to mutate `prefix_cache` / `lru_order` below.
        let block = self
            .prefix_cache
            .get(&hash)
            .expect("hash found during the scan above; no mutation happened in between");
        let block_id = block.block_id;
        // Release the cache's logical ref (1 → 0).
        let _ = block.decref();
        // Remove all bookkeeping for this entry.
        self.prefix_cache.remove(&hash);
        self.prefix_cache_identities.remove(&hash);
        self.block_hashes.remove(&block_id);
        self.lru_order.remove(&hash);
        self.allocated.remove(&block_id);
        Some(block_id)
    }

    /// Free a block
    ///
    /// The block is only returned to the free pool if its ref_count reaches 0
    pub fn free(&mut self, block: Arc<PhysicalBlock>) {
        let block_id = block.block_id;

        // Decrement ref count
        if block.decref() {
            // Ref count reached 0, return to free pool
            self.allocated.remove(&block_id);

            // Remove from prefix cache if present
            if let Some(hash) = self.block_hashes.remove(&block_id) {
                self.prefix_cache.remove(&hash);
                self.prefix_cache_identities.remove(&hash);
                self.lru_order.retain(|&h| h != hash);
            }

            self.free_blocks.push_back(block_id);
        }
    }

    /// Evict EVERY prefix-cache entry, releasing the cache's logical
    /// reference on each block (the same per-entry cleanup the capacity
    /// eviction loop in [`Self::register_prefix`] performs). Blocks whose
    /// `ref_count` drops to 0 are removed from `allocated` and returned to
    /// the free pool; blocks still held by a live request stay allocated
    /// but are no longer discoverable via `lookup_prefix` /
    /// `find_longest_cache_hit`.
    ///
    /// This is the hard-reset primitive behind an EXPLICIT session reset
    /// (`ResetScope::Command` in mlx-core): [`Self::free`] /
    /// `release_request` deliberately keep content-addressed full blocks
    /// registered for cross-request reuse, so a reset-then-rerun of the
    /// same prompt would otherwise take the prefix-hit suffix-prefill path
    /// — which reduces bf16 in a different order than a cold full prefill
    /// and can flip a greedy near-tie (observed on LFM2). Purging restores
    /// the documented "fully cold state" contract.
    pub fn purge_prefix_cache(&mut self) {
        self.lru_order.clear();
        self.prefix_cache_identities.clear();
        // `block_hashes` only carries reverse mappings for live
        // prefix-cache entries (the register/free invariants), so it
        // empties exactly when the cache does.
        self.block_hashes.clear();
        for (_hash, block) in self.prefix_cache.drain() {
            if block.decref() {
                let block_id = block.block_id;
                self.allocated.remove(&block_id);
                self.free_blocks.push_back(block_id);
            }
        }
    }

    /// Register a block in the prefix cache
    ///
    /// The block will be reused when a sequence has matching prefix tokens.
    ///
    /// # Reference-count semantics
    ///
    /// The `prefix_cache` holds **one logical reference per entry** (see
    /// the module-level invariant). `register_prefix` is the function that
    /// takes that reference on the genuine-insert path; every removal path
    /// in this method (Case 1 stale-alias displacement, capacity eviction
    /// loop) is responsible for releasing it via `decref()`, returning the
    /// block to the free pool if the count hits zero.
    ///
    /// Idempotent refresh (same block, same hash) does NOT incref — the
    /// existing logical reference is reused across the LRU bump.
    ///
    /// # Aliasing policy & precedence
    ///
    /// `block_hashes` only tracks ONE reverse mapping per `block_id`, so we
    /// must keep `prefix_cache` and `block_hashes` consistent for `free()` to
    /// clean up correctly. The checks run in this order:
    ///
    /// 1. **Collision drop FIRST — same hash, different block** (hash
    ///    collision or caller logic error): the new registration is dropped
    ///    (no-op) and we return immediately, BEFORE touching the incoming
    ///    block's existing alias. This preserves the invariant that
    ///    `block_hashes[id]` always reflects the entry currently in
    ///    `prefix_cache`, and crucially also preserves any prior valid
    ///    registration the incoming block already had — a rejected
    ///    registration must be a true no-op for the caller's block.
    ///    Because nothing was inserted, no incref happens here.
    ///
    /// 2. **Stale-alias eviction — same block, different hash** (e.g. same
    ///    tokens cached under different `extra_keys`): the OLD hash is
    ///    evicted from `prefix_cache` and `lru_order` before inserting the
    ///    new alias. Otherwise the stale entry would survive `free()` and
    ///    could hand out a returned-to-pool block on a future
    ///    `lookup_prefix` — bypassing `extra_keys` isolation. The cache's
    ///    logical reference for the OLD hash is released here (decref); the
    ///    new alias takes a fresh ref via Step 4 below — net change in
    ///    ref_count for this block is zero (one ref consumed for the old
    ///    hash, one ref taken for the new hash; the same block stays in
    ///    the cache, just under a different key).
    ///
    /// 3. **Capacity eviction — only on genuine insertion**: the LRU eviction
    ///    loop runs only when we're about to ADD a new hash entry. A
    ///    refresh of an already-present hash doesn't grow the cache, so
    ///    skipping the loop in that case avoids evicting unrelated entries
    ///    under capacity pressure. Each evicted entry releases the cache's
    ///    logical reference (decref); if that drops the block to ref_count
    ///    0 (no other holder), the block is removed from `allocated` and
    ///    pushed back onto `free_blocks` — same cleanup `free()` performs.
    ///
    /// 4. **LRU refresh + insert**: bump the hash to the back of `lru_order`
    ///    and (re)insert into `prefix_cache` / `block_hashes`. Incref iff
    ///    this is a genuine new insertion (idempotent refresh skips the
    ///    incref so the cache holds at most ONE logical reference per
    ///    entry).
    ///
    /// # Return value
    ///
    /// Returns `true` when the registration was accepted — i.e. the
    /// passed-in `block` is now authoritative for `hash` in the prefix
    /// cache. Specifically:
    ///
    /// - Genuine insertion (capacity-eviction path): `true`.
    /// - Idempotent refresh (same `block`, same `hash`): `true` — the
    ///   block was already authoritative; we just refreshed LRU.
    /// - Case 1 stale-alias displacement (same `block`, different `hash`):
    ///   `true` — old hash entry was released, new alias installed.
    ///
    /// Returns `false` only when the registration was rejected and the
    /// caller's `block` was NOT inserted:
    ///
    /// - Cache disabled (`max_prefix_cache_entries == 0`).
    /// - Case 2 hash collision (same `hash`, different `block`): the
    ///   pre-existing entry stays authoritative.
    ///
    /// Callers that walk a hash chain (notably `cache_full_blocks`) MUST
    /// abort on `false`: a chain block whose registration was dropped
    /// means subsequent block hashes would link to ghost predecessors,
    /// producing a future `find_longest_cache_hit` return that mixes
    /// blocks across registration intents (silent KV corruption).
    pub(crate) fn register_prefix(&mut self, block: Arc<PhysicalBlock>, hash: u64) -> bool {
        // If prefix caching is disabled (max_prefix_cache_entries == 0), do nothing
        if self.max_prefix_cache_entries == 0 {
            return false;
        }

        // Step 1 (collision drop, FIRST): this hash is already mapped to a
        // DIFFERENT block. Reject the new registration as a true no-op —
        // don't touch the incoming block's prior alias (if any), don't
        // shuffle LRU, don't take the cache's ref (nothing was inserted).
        // The existing entry stays authoritative.
        // (Same block + same hash falls through to the LRU refresh below;
        // no eviction needed since the entry is already correct.)
        if let Some(existing_block) = self.prefix_cache.get(&hash)
            && existing_block.block_id != block.block_id
        {
            return false;
        }

        // Step 2 (stale-alias eviction): this block_id is already registered
        // under a DIFFERENT hash. Evict the stale alias before installing the
        // new one — otherwise the old prefix_cache entry would survive free()
        // and could leak across extra_keys boundaries. (block_hashes will be
        // overwritten below, no need to remove first.) Release the cache's
        // logical reference for the OLD hash; the new alias takes its own
        // ref in Step 4. The block survives this swap because at least one
        // of {external request handle, cache's ref about to be retaken} keeps
        // ref_count >= 1.
        if let Some(&existing_hash) = self.block_hashes.get(&block.block_id)
            && existing_hash != hash
        {
            if self.prefix_cache.remove(&existing_hash).is_some() {
                self.prefix_cache_identities.remove(&existing_hash);
                // Decref the cache's logical reference for the old alias.
                // We deliberately ignore a true return here: callers that
                // re-register a block they still hold (the common case)
                // keep ref_count >= 1, and Step 4 below restores the
                // cache's ref under the new hash. If a caller somehow ends
                // up re-registering a block they no longer hold, ref_count
                // could hit 0 — but that block is about to be re-inserted
                // under the new hash anyway, so leaving it in `allocated`
                // is safe and avoids a free→re-allocate flap.
                let _ = block.decref();
            }
            self.lru_order.remove(&existing_hash);
        }

        // Step 3 (capacity eviction, only on genuine insertion): if this
        // call is a refresh of an already-present hash it won't grow the
        // cache, so skip the eviction loop. Otherwise evict oldest entries
        // until we have room for the new insertion. Each eviction releases
        // the cache's logical reference for that block; if no external
        // holder remains (ref_count hits 0), the block is fully reclaimed.
        let is_new_insertion = !self.prefix_cache.contains_key(&hash);
        if is_new_insertion {
            while self.prefix_cache.len() >= self.max_prefix_cache_entries {
                match self.lru_order.pop_front() {
                    Some(old_hash) => {
                        // Remove evicted entry from both side tables, then
                        // release the cache's logical reference. If that
                        // drops ref_count to 0 the block goes back to the
                        // free pool — same cleanup `free()` performs.
                        if let Some(evicted_block) = self.prefix_cache.remove(&old_hash) {
                            self.prefix_cache_identities.remove(&old_hash);
                            let evicted_id = evicted_block.block_id;
                            self.block_hashes.remove(&evicted_id);
                            if evicted_block.decref() {
                                self.allocated.remove(&evicted_id);
                                self.free_blocks.push_back(evicted_id);
                            }
                        }
                    }
                    None => {
                        // Safety: If lru_order is empty but cache still has entries,
                        // this indicates a bug (desynchronization). Break to avoid infinite loop.
                        break;
                    }
                }
            }
        }

        // Step 4: LRU refresh (move to back if present, insert if not) +
        // insert into `prefix_cache`. Take the cache's logical reference
        // iff this is a genuine new insertion. An idempotent refresh leaves
        // ref_count unchanged so the cache continues to hold exactly ONE
        // logical ref per entry. `LinkedHashSet::insert` is an O(1)
        // amortized move-to-back when `hash` is already present, replacing
        // the old O(n) retain()+push_back() pair.
        self.lru_order.insert(hash);

        // Track the hash for this block (for cleanup during free)
        self.block_hashes.insert(block.block_id, hash);

        if is_new_insertion {
            // Direct register_prefix callers do not provide enough
            // identity material to verify future duplicate-hash attempts.
            // cache_full_blocks repopulates this immediately after a
            // successful insertion.
            self.prefix_cache_identities.remove(&hash);
            block.incref();
        }

        // Insert into cache
        self.prefix_cache.insert(hash, block);
        true
    }

    /// Look up a block in the prefix cache
    ///
    /// Returns the cached block if found, incrementing its ref count
    pub(crate) fn lookup_prefix(&mut self, hash: u64) -> Option<Arc<PhysicalBlock>> {
        if let Some(block) = self.prefix_cache.get(&hash) {
            // Update LRU order: O(1) amortized move-to-back (insert is a
            // no-op re-link when `hash` is already present), replacing the
            // old O(n) retain()+push_back() pair. This runs once per
            // matched block on every prefix-cache hit, so it's the hottest
            // of the four LRU-touch sites in this file.
            self.lru_order.insert(hash);

            // Increment ref count and return
            block.incref();
            Some(Arc::clone(block))
        } else {
            None
        }
    }

    fn prefix_identity_matches(
        &self,
        hash: u64,
        token_ids: &[u32],
        parent_hash: u64,
        extra_keys: &[u64],
        cache_salt: u64,
        block_index: usize,
    ) -> bool {
        self.prefix_cache_identities
            .get(&hash)
            .is_some_and(|identity| {
                identity.matches(token_ids, parent_hash, extra_keys, cache_salt, block_index)
            })
    }

    fn prefix_identity_mismatches(
        &self,
        hash: u64,
        token_ids: &[u32],
        parent_hash: u64,
        extra_keys: &[u64],
        cache_salt: u64,
        block_index: usize,
    ) -> bool {
        self.prefix_cache_identities
            .get(&hash)
            .is_some_and(|identity| {
                !identity.matches(token_ids, parent_hash, extra_keys, cache_salt, block_index)
            })
    }

    fn remember_prefix_identity(
        &mut self,
        hash: u64,
        token_ids: &[u32],
        parent_hash: u64,
        extra_keys: &[u64],
        cache_salt: u64,
        block_index: usize,
    ) {
        self.prefix_cache_identities.insert(
            hash,
            PrefixCacheBlockIdentity::new(
                token_ids,
                parent_hash,
                extra_keys,
                cache_salt,
                block_index,
            ),
        );
    }

    /// Walk a token sequence in `block_size`-aligned chunks, looking up each
    /// block in the prefix cache. Stop at the first miss. Returns the cached
    /// blocks (with their ref counts already bumped via `lookup_prefix`, in
    /// order) and the cached token count.
    ///
    /// `extra_keys` is applied uniformly per-block-hash (same value for every
    /// block in this call). For per-block extra_keys (multimodal cache
    /// isolation where different blocks carry different image hashes), use
    /// [`Self::find_longest_cache_hit_per_block`] instead.
    ///
    /// `cache_salt` is mixed into the FIRST block's hash only (when it is
    /// non-zero). Pass `0` for "no salt". The first-block-only semantics
    /// align with vLLM (`vllm/v1/core/kv_cache_utils.py:521-531`), which adds the
    /// request's `cache_salt` to `extra_keys` only when
    /// `start_token_idx == 0`. A non-zero salt isolates the per-request
    /// prefix from the cross-tenant prefix pool while still allowing
    /// later blocks (which only chain off `parent_hash`) to converge into
    /// shared content if the chain matches.
    ///
    /// Mirrors vLLM `vllm/v1/core/single_type_kv_cache_manager.py:421-468`
    /// (`FullAttentionManager.find_longest_cache_hit`).
    pub fn find_longest_cache_hit(
        &mut self,
        token_ids: &[u32],
        block_size: u32,
        extra_keys: &[u64],
        cache_salt: u64,
    ) -> (Vec<Arc<PhysicalBlock>>, usize) {
        // Defensive: 0 block_size would cause infinite loop / divide by zero
        if block_size == 0 || token_ids.is_empty() || token_ids.len() < block_size as usize {
            return (Vec::new(), 0);
        }

        let block_size_us = block_size as usize;
        let num_full_blocks = token_ids.len() / block_size_us;

        let mut blocks: Vec<Arc<PhysicalBlock>> = Vec::with_capacity(num_full_blocks);
        let mut previous_block_hash: u64 = 0;

        for n in 0..num_full_blocks {
            let start = n * block_size_us;
            let end = start + block_size_us;
            let block_tokens = &token_ids[start..end];
            let parent_hash = if n == 0 { 0 } else { previous_block_hash };
            let block_hash = hash_block(block_tokens, parent_hash, extra_keys, cache_salt, n);

            if self.prefix_identity_mismatches(
                block_hash,
                block_tokens,
                parent_hash,
                extra_keys,
                cache_salt,
                n,
            ) {
                break;
            }

            match self.lookup_prefix(block_hash) {
                Some(block) => {
                    blocks.push(block);
                    previous_block_hash = block_hash;
                }
                None => break,
            }
        }

        let cached_tokens = blocks.len() * block_size_us;
        (blocks, cached_tokens)
    }

    /// Per-block-extra_keys variant of [`Self::find_longest_cache_hit`].
    ///
    /// Each block uses its own `extra_keys` vector for the hash, indexed by
    /// block position. This is the load-bearing primitive for multimodal
    /// prefix caching: a request whose blocks contain image-token positions
    /// passes per-block image hashes here so that two requests with the same
    /// text prefix but different images produce distinct block hashes (and
    /// therefore distinct cache identities).
    ///
    /// `extra_keys_per_block.len()` MUST be at least the number of full
    /// blocks scanned (`token_ids.len() / block_size`). When shorter, the
    /// scan stops at the first block without per-block keys (treated as a
    /// cache miss). Pass an all-empty vec (e.g. produced by
    /// `compute_per_block_image_extra_keys(&[], num_blocks, block_size)`)
    /// for text-only requests to get the same hashes as
    /// `find_longest_cache_hit(token_ids, block_size, &[], cache_salt)`
    /// (when called with the same `cache_salt`).
    ///
    /// Mirrors vLLM commit 269bf46d which added per-block extra_keys to the
    /// FullAttentionManager prefix-cache walk.
    ///
    /// `cache_salt` is mixed into the FIRST block's hash only (when it is
    /// non-zero), with the same semantics as
    /// [`Self::find_longest_cache_hit`]. See vLLM
    /// `vllm/v1/core/kv_cache_utils.py:521-531`.
    pub fn find_longest_cache_hit_per_block(
        &mut self,
        token_ids: &[u32],
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> (Vec<Arc<PhysicalBlock>>, usize) {
        if block_size == 0 || token_ids.is_empty() || token_ids.len() < block_size as usize {
            return (Vec::new(), 0);
        }

        let block_size_us = block_size as usize;
        let num_full_blocks = token_ids.len() / block_size_us;

        let mut blocks: Vec<Arc<PhysicalBlock>> = Vec::with_capacity(num_full_blocks);
        let mut previous_block_hash: u64 = 0;

        for n in 0..num_full_blocks {
            let Some(per_block) = extra_keys_per_block.get(n) else {
                // Caller didn't supply keys for this block — treat as a
                // miss to keep cache identity unambiguous. (vLLM aborts
                // here too.)
                break;
            };
            let start = n * block_size_us;
            let end = start + block_size_us;
            let block_tokens = &token_ids[start..end];
            let parent_hash = if n == 0 { 0 } else { previous_block_hash };
            let block_hash = hash_block(block_tokens, parent_hash, per_block, cache_salt, n);

            if self.prefix_identity_mismatches(
                block_hash,
                block_tokens,
                parent_hash,
                per_block,
                cache_salt,
                n,
            ) {
                break;
            }

            match self.lookup_prefix(block_hash) {
                Some(block) => {
                    blocks.push(block);
                    previous_block_hash = block_hash;
                }
                None => break,
            }
        }

        let cached_tokens = blocks.len() * block_size_us;
        (blocks, cached_tokens)
    }

    /// Register a freshly computed sequence's blocks in the prefix cache.
    /// Caller has already allocated `blocks` for the sequence; this method
    /// computes the chain of block hashes and inserts each FULL block via
    /// `register_prefix`.
    ///
    /// `blocks.len() * block_size` must be `<= token_ids.len()`. Only the
    /// fully-formed blocks are registered; the trailing partial block isn't
    /// cached until it's full.
    ///
    /// `extra_keys` is applied uniformly per-block-hash (same value for
    /// every block in this call). For per-block extra_keys (multimodal
    /// cache isolation), use [`Self::cache_full_blocks_per_block`].
    ///
    /// `cache_salt` is mixed into the FIRST block's hash only (when it is
    /// non-zero). Pass `0` for "no salt". The first-block-only semantics
    /// mirror vLLM (`vllm/v1/core/kv_cache_utils.py:521-531`): the request's
    /// `cache_salt` is added to `extra_keys` only for the leading block,
    /// so two requests with the same tokens but different salts publish
    /// distinct first-block cache identities while later blocks (which
    /// only chain off `parent_hash`) stay shareable when the chain
    /// converges.
    ///
    /// Mirrors vLLM `vllm/v1/core/block_pool.py:211-320` (`cache_full_blocks`).
    ///
    /// # Collision and duplicate-prefix handling
    ///
    /// If `register_prefix` rejects a block partway through the chain
    /// (Case 2 hash collision — `hash_n` is already authoritative for a
    /// different block), we continue only when the existing cache entry
    /// has verified-identical block identity metadata. That is the normal
    /// cold-prefill replay case: the prefix was already cached under old
    /// physical blocks, and this request recomputed the same leading
    /// blocks before producing a new tail. The duplicate leading blocks
    /// are skipped, and later tail blocks can still be published.
    ///
    /// If identity is missing or different, the chain is aborted
    /// immediately. We do NOT continue computing
    /// `hash_{n+1} = H(hash_n, ...)` and registering later blocks — those
    /// would link to a predecessor (`hash_n`) that resolves to someone
    /// else's block, so a future `find_longest_cache_hit` walking the
    /// chain would mix blocks across registration intents (silent KV
    /// corruption).
    ///
    /// Returns the number of blocks actually registered. May be less than
    /// `blocks.len()` if the chain was aborted mid-way or if verified
    /// duplicate leading blocks were skipped; callers can treat any value
    /// as success — the partial registration is still internally
    /// consistent. Subsequent lookups simply miss at the first dropped
    /// block, which forces a fresh prefill (correct behavior).
    pub fn cache_full_blocks(
        &mut self,
        token_ids: &[u32],
        blocks: &[Arc<PhysicalBlock>],
        block_size: u32,
        extra_keys: &[u64],
        cache_salt: u64,
    ) -> Result<usize, &'static str> {
        if block_size == 0 {
            return Err("block_size must be > 0");
        }

        let block_size_us = block_size as usize;
        if blocks.len() * block_size_us > token_ids.len() {
            return Err("blocks exceed token_ids length");
        }

        let mut previous_block_hash: u64 = 0;
        let mut registered = 0usize;
        for (n, block) in blocks.iter().enumerate() {
            let start = n * block_size_us;
            let end = start + block_size_us;
            let block_tokens = &token_ids[start..end];
            let parent_hash = if n == 0 { 0 } else { previous_block_hash };
            let block_hash = hash_block(block_tokens, parent_hash, extra_keys, cache_salt, n);
            if !self.register_prefix(Arc::clone(block), block_hash) {
                if self.prefix_identity_matches(
                    block_hash,
                    block_tokens,
                    parent_hash,
                    extra_keys,
                    cache_salt,
                    n,
                ) {
                    // The same logical block is already cached under a
                    // different physical block. This happens after a cold
                    // prefill recomputes an existing prefix: skip the
                    // duplicate leading block but keep walking so the new
                    // tail can be published.
                    previous_block_hash = block_hash;
                    continue;
                } else {
                    // Chain broke (collision drop or cache disabled). Stop
                    // here: any further block we register would chain off a
                    // hash that resolves to someone else's block, which
                    // corrupts future find_longest_cache_hit walks. Return
                    // the count of accepted registrations up to (but not
                    // including) the dropped block.
                    break;
                }
            }
            self.remember_prefix_identity(
                block_hash,
                block_tokens,
                parent_hash,
                extra_keys,
                cache_salt,
                n,
            );
            previous_block_hash = block_hash;
            registered += 1;
        }
        Ok(registered)
    }

    /// Per-block-extra_keys variant of [`Self::cache_full_blocks`].
    ///
    /// Each block in `blocks` is hashed with its own `extra_keys` vector
    /// (`extra_keys_per_block[n]`). The chain semantics (parent_hash,
    /// abort-on-collision, partial-success return) match
    /// `cache_full_blocks` exactly. The two methods produce identical
    /// hashes when `extra_keys_per_block[n] == extra_keys` for every n
    /// (verified by unit tests).
    ///
    /// `extra_keys_per_block.len()` MUST be at least `blocks.len()`. The
    /// caller built it via [`compute_per_block_image_extra_keys`] (in the
    /// adapter module) or an analogous per-model helper that maps absolute
    /// token positions to their owning block.
    ///
    /// Returns the number of blocks actually registered. May be less than
    /// `blocks.len()` if the chain aborted on a hash collision mid-way or
    /// skipped verified duplicate leading blocks.
    ///
    /// `cache_salt` is mixed into the FIRST block's hash only (when it is
    /// non-zero), with the same semantics as
    /// [`Self::cache_full_blocks`]. See vLLM
    /// `vllm/v1/core/kv_cache_utils.py:521-531`.
    ///
    /// Mirrors vLLM commit 269bf46d (per-block extra_keys for multimodal).
    pub fn cache_full_blocks_per_block(
        &mut self,
        token_ids: &[u32],
        blocks: &[Arc<PhysicalBlock>],
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Result<usize, &'static str> {
        if block_size == 0 {
            return Err("block_size must be > 0");
        }
        let block_size_us = block_size as usize;
        if blocks.len() * block_size_us > token_ids.len() {
            return Err("blocks exceed token_ids length");
        }
        if extra_keys_per_block.len() < blocks.len() {
            return Err("extra_keys_per_block shorter than blocks");
        }

        let mut previous_block_hash: u64 = 0;
        let mut registered = 0usize;
        for (n, block) in blocks.iter().enumerate() {
            let start = n * block_size_us;
            let end = start + block_size_us;
            let block_tokens = &token_ids[start..end];
            let extra_keys = &extra_keys_per_block[n];
            let parent_hash = if n == 0 { 0 } else { previous_block_hash };
            let block_hash = hash_block(block_tokens, parent_hash, extra_keys, cache_salt, n);
            if !self.register_prefix(Arc::clone(block), block_hash) {
                if self.prefix_identity_matches(
                    block_hash,
                    block_tokens,
                    parent_hash,
                    extra_keys,
                    cache_salt,
                    n,
                ) {
                    previous_block_hash = block_hash;
                    continue;
                } else {
                    break;
                }
            }
            self.remember_prefix_identity(
                block_hash,
                block_tokens,
                parent_hash,
                extra_keys,
                cache_salt,
                n,
            );
            previous_block_hash = block_hash;
            registered += 1;
        }
        Ok(registered)
    }

    /// Get the number of free blocks
    pub fn num_free_blocks(&self) -> u32 {
        self.free_blocks.len() as u32
    }

    /// Get the number of allocated blocks
    pub fn num_allocated_blocks(&self) -> u32 {
        self.allocated.len() as u32
    }

    /// Get the total number of blocks
    pub fn total_blocks(&self) -> u32 {
        self.num_blocks
    }

    /// Alias of [`Self::total_blocks`]. Matches the naming of
    /// `LayerKVPool::num_blocks` so the adapter can validate matching
    /// capacity without surprise.
    pub fn num_blocks(&self) -> u32 {
        self.num_blocks
    }

    /// Get the block size
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Check if we can allocate the requested number of blocks.
    ///
    /// Counts both genuinely-free blocks AND evictable cache-only
    /// blocks (ref_count == 1) — the same set `allocate` can draw from
    /// via `try_evict_lru_for_allocation`. Without this, schedulers that
    /// guard `allocate` with `can_allocate` would refuse work that the
    /// allocator could in fact satisfy.
    pub fn can_allocate(&self, num_blocks: u32) -> bool {
        let free = self.num_free_blocks();
        if free >= num_blocks {
            return true;
        }
        let evictable = self
            .prefix_cache
            .values()
            .filter(|b| b.get_ref_count() == 1)
            .count() as u32;
        free + evictable >= num_blocks
    }

    /// Set the maximum number of entries the prefix cache will hold before
    /// the LRU eviction loop fires on subsequent inserts.
    ///
    /// This setter does NOT shrink the cache below an existing population —
    /// pre-existing entries are left in place and only the next genuine
    /// insertion will trigger eviction.
    pub fn set_max_prefix_cache_entries(&mut self, max_entries: usize) {
        self.max_prefix_cache_entries = max_entries;
    }
}

/// Hash function for token sequences (for prefix caching).
///
/// Computes a chained block hash in vLLM's style: feeds `parent_hash` first,
/// then each token id in order, then each entry of `extra_keys` in order.
///
/// `extra_keys` is reserved for per-block side-channel information that must
/// participate in cache identity — image content hashes, cache-salt, LoRA
/// names, etc. (see vLLM commit 269bf46d). Order matters: `[a, b]` and
/// `[b, a]` produce different hashes. Most callers should pass `&[]`.
///
/// Uses Rust's `DefaultHasher` (SipHash-1-3). vLLM uses xxhash/sha256 for
/// cross-process determinism, but our prefix cache is process-local — every
/// hash is computed and consumed in the same process — so SipHash's stronger
/// collision resistance is the better trade-off and we don't need stable
/// hashes across runs.
///
// FIXME: SipHash u64 is not cryptographically collision-resistant.
// `find_longest_cache_hit` walks chained block hashes via `lookup_prefix`, so a
// mid-chain collision between two different token chains could cause a
// mixed-prefix lookup for entries registered without block identity metadata.
// cache_full_blocks/cache_full_blocks_per_block entries are verified on lookup
// and duplicate registration, but direct register_prefix callers still have no
// token/extra-key metadata. See the module-level "SipHash collision limitation"
// doc above for details.
pub fn hash_tokens(tokens: &[u32], parent_hash: u64, extra_keys: &[u64]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    parent_hash.hash(&mut hasher);
    for &token in tokens {
        token.hash(&mut hasher);
    }
    for &key in extra_keys {
        key.hash(&mut hasher);
    }
    hasher.finish()
}

/// Per-block hash helper for the four hashing-loop sites in this module.
///
/// Mixes `cache_salt` into block 0's hash only (when `cache_salt != 0` and
/// `block_index == 0`), matching vLLM's first-block-only `cache_salt`
/// composition (`vllm/v1/core/kv_cache_utils.py:521-531`):
///
/// ```text
/// cache_salt_keys = [cache_salt] if start_token_idx == 0 and cache_salt else []
/// extra_keys      = ... + cache_salt_keys + ...
/// ```
///
/// When `cache_salt == 0` OR `block_index > 0`, this collapses to
/// `hash_tokens(tokens, parent_hash, extra_keys)` byte-for-byte — no extra
/// allocation, no salt mixed in. The leading-block + non-zero-salt branch
/// drives the `Hasher` directly (`parent_hash`, every token, every
/// `extra_keys` entry, then `cache_salt`) instead of materializing an
/// `extra_keys` + `[cache_salt]` slice; the result is bit-equal to
/// `hash_tokens(tokens, parent_hash, &[extra_keys..., cache_salt])` but
/// avoids the heap allocation that path used to do. Ordering matches vLLM:
/// `cache_salt` is hashed AFTER the existing `extra_keys` entries.
#[inline]
fn hash_block(
    tokens: &[u32],
    parent_hash: u64,
    extra_keys: &[u64],
    cache_salt: u64,
    block_index: usize,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    if cache_salt == 0 || block_index != 0 {
        return hash_tokens(tokens, parent_hash, extra_keys);
    }
    // First block + non-zero salt → drive the hasher directly. Bit-equal to
    // `hash_tokens(tokens, parent_hash, [extra_keys..., cache_salt])` because
    // `hash_tokens` writes the same prefix in the same order, then iterates
    // its `extra_keys` slice; appending `cache_salt` here matches that.
    let mut hasher = DefaultHasher::new();
    parent_hash.hash(&mut hasher);
    for &token in tokens {
        token.hash(&mut hasher);
    }
    for &key in extra_keys {
        key.hash(&mut hasher);
    }
    cache_salt.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_free() {
        let mut allocator = BlockAllocator::new(10, 32);

        assert_eq!(allocator.num_free_blocks(), 10);

        let block = allocator.allocate().unwrap();
        assert_eq!(allocator.num_free_blocks(), 9);
        assert_eq!(block.block_id, 0);

        allocator.free(block);
        assert_eq!(allocator.num_free_blocks(), 10);
    }

    #[test]
    fn test_reference_counting() {
        let mut allocator = BlockAllocator::new(10, 32);

        let block = allocator.allocate().unwrap();
        assert_eq!(block.get_ref_count(), 1);

        // Explicitly share the block (like for beam search)
        block.incref();
        let block2 = Arc::clone(&block);
        assert_eq!(block.get_ref_count(), 2);
        assert_eq!(block2.get_ref_count(), 2);

        // Free only decrements, doesn't return to pool
        allocator.free(block);
        assert_eq!(allocator.num_free_blocks(), 9);

        // Second free returns to pool
        allocator.free(block2);
        assert_eq!(allocator.num_free_blocks(), 10);
    }

    #[test]
    fn test_prefix_cache() {
        let mut allocator = BlockAllocator::new(10, 32);

        let block = allocator.allocate().unwrap();
        let hash = hash_tokens(&[1, 2, 3], 0, &[]);

        allocator.register_prefix(Arc::clone(&block), hash);

        // Lookup should find the block
        let cached = allocator.lookup_prefix(hash).unwrap();
        assert_eq!(cached.block_id, block.block_id);
        // Original (1) + register_prefix incref (1) + lookup increments (1) = 3.
        assert_eq!(cached.get_ref_count(), 3);

        // Unknown hash should return None
        assert!(allocator.lookup_prefix(12345).is_none());
    }

    #[test]
    fn test_prefix_cache_cleanup_on_free() {
        // With the cache-holds-its-own-ref design, freeing the only
        // external handle leaves the block alive at ref_count=1 (the
        // cache's logical reference). LRU eviction is what eventually
        // returns the block to the free pool — see
        // `test_lru_eviction_returns_block_to_free_pool`.
        let mut allocator = BlockAllocator::new(10, 32);

        let block = allocator.allocate().unwrap();
        let hash = hash_tokens(&[1, 2, 3], 0, &[]);

        // Register in prefix cache (incref'd for the cache's logical ref).
        allocator.register_prefix(Arc::clone(&block), hash);
        // ref_count: original(1) + register(1) = 2

        // Free the external handle. ref_count: 2 -> 1. Block stays alive
        // because the cache still holds its logical reference.
        allocator.free(block);
        assert_eq!(
            allocator.num_free_blocks(),
            9,
            "cache holds the block; free pool should not get it back yet"
        );

        // Lookup still works — cache survived the free.
        let cached = allocator.lookup_prefix(hash).unwrap();
        // ref_count: 1 (cache) + 1 (this lookup) = 2
        assert_eq!(cached.get_ref_count(), 2);

        // Free the lookup handle: ref_count 2 -> 1. Cache's ref still holds.
        allocator.free(cached);
        assert!(allocator.lookup_prefix(hash).is_some());
        assert_eq!(allocator.num_free_blocks(), 9);
    }

    #[test]
    fn test_purge_prefix_cache() {
        let mut allocator = BlockAllocator::new(10, 32);

        // Cache-only entry: external handle freed, only the cache's
        // logical ref keeps it alive (the post-`release_request` state).
        let cache_only = allocator.allocate().unwrap();
        let hash_cache_only = hash_tokens(&[1, 2, 3], 0, &[]);
        allocator.register_prefix(Arc::clone(&cache_only), hash_cache_only);
        allocator.free(cache_only);
        assert_eq!(allocator.num_free_blocks(), 9);

        // Live entry: a request still holds an external handle.
        let live = allocator.allocate().unwrap();
        let hash_live = hash_tokens(&[4, 5, 6], 0, &[]);
        allocator.register_prefix(Arc::clone(&live), hash_live);
        assert_eq!(allocator.num_free_blocks(), 8);

        allocator.purge_prefix_cache();

        // Both entries are gone from the cache.
        assert!(allocator.lookup_prefix(hash_cache_only).is_none());
        assert!(allocator.lookup_prefix(hash_live).is_none());

        // The cache-only block returned to the free pool; the live block
        // stays allocated for its holder (9 free, 1 still out).
        assert_eq!(allocator.num_free_blocks(), 9);
        assert_eq!(live.get_ref_count(), 1, "only the external ref remains");

        // The surviving handle frees cleanly afterwards.
        allocator.free(live);
        assert_eq!(allocator.num_free_blocks(), 10);

        // A purged-then-reallocated pool keeps working: re-register and
        // hit the same hash again.
        let again = allocator.allocate().unwrap();
        assert!(allocator.register_prefix(Arc::clone(&again), hash_cache_only));
        assert!(allocator.lookup_prefix(hash_cache_only).is_some());
    }

    #[test]
    fn test_prefix_cache_eviction_cleanup() {
        // This test verifies that evicted blocks are properly cleaned up
        let mut allocator = BlockAllocator::new(10, 32);
        allocator.max_prefix_cache_entries = 2; // Small cache for testing

        let block1 = allocator.allocate().unwrap();
        let hash1 = hash_tokens(&[1], 0, &[]);
        allocator.register_prefix(Arc::clone(&block1), hash1);

        let block2 = allocator.allocate().unwrap();
        let hash2 = hash_tokens(&[2], 0, &[]);
        allocator.register_prefix(Arc::clone(&block2), hash2);

        // Cache is at capacity (2 entries)
        assert_eq!(allocator.prefix_cache.len(), 2);

        // Add a third block, should evict the first (LRU)
        let block3 = allocator.allocate().unwrap();
        let hash3 = hash_tokens(&[3], 0, &[]);
        allocator.register_prefix(Arc::clone(&block3), hash3);

        // Verify hash1 was evicted
        assert!(allocator.lookup_prefix(hash1).is_none());
        assert!(allocator.lookup_prefix(hash2).is_some());
        assert!(allocator.lookup_prefix(hash3).is_some());

        // Verify block_hashes was also cleaned up
        assert!(!allocator.block_hashes.contains_key(&block1.block_id));
        assert!(allocator.block_hashes.contains_key(&block2.block_id));
        assert!(allocator.block_hashes.contains_key(&block3.block_id));

        // block1 (the evicted one) still has the external handle — its
        // ref_count is now 1 (cache's ref was released by eviction). It
        // remains in `allocated`. Drop the external handle to confirm
        // free pool comes back to 8 (10 total - block2 - block3 still held).
        assert_eq!(block1.get_ref_count(), 1, "cache ref released on eviction");
        allocator.free(block1);
        // block1 returns to pool. block2, block3, and the cached refs
        // for hash2 and hash3 keep those blocks pinned.
        assert_eq!(allocator.num_free_blocks(), 8);
    }

    #[test]
    fn test_prefix_cache_disabled() {
        // This test verifies that setting max_prefix_cache_entries = 0 disables caching
        // and doesn't cause infinite loop
        let mut allocator = BlockAllocator::new(10, 32);
        allocator.max_prefix_cache_entries = 0; // Disable prefix caching

        let block = allocator.allocate().unwrap();
        let hash = hash_tokens(&[1, 2, 3], 0, &[]);

        // Should not cache when disabled — returns false to signal nothing was inserted.
        assert!(!allocator.register_prefix(Arc::clone(&block), hash));

        // Verify nothing was cached
        assert_eq!(allocator.prefix_cache.len(), 0);
        assert!(allocator.lookup_prefix(hash).is_none());
    }

    #[test]
    fn test_prefix_cache_eviction_safety() {
        // This test verifies that even if lru_order becomes desynchronized,
        // we don't infinite loop
        let mut allocator = BlockAllocator::new(10, 32);
        allocator.max_prefix_cache_entries = 1;

        let block1 = allocator.allocate().unwrap();
        let hash1 = hash_tokens(&[1], 0, &[]);
        allocator.register_prefix(Arc::clone(&block1), hash1);

        // Manually desynchronize: clear lru_order but leave prefix_cache populated
        allocator.lru_order.clear();

        // This should not infinite loop - it will break when pop_front returns None
        let block2 = allocator.allocate().unwrap();
        let hash2 = hash_tokens(&[2], 0, &[]);
        allocator.register_prefix(Arc::clone(&block2), hash2);

        // Verify we didn't infinite loop and the function completed
        assert!(!allocator.prefix_cache.is_empty());
    }

    // hash_tokens(extra_keys), find_longest_cache_hit, cache_full_blocks,
    // refcount lifecycle, LRU eviction order.

    #[test]
    fn test_hash_tokens_extra_keys() {
        // No extra_keys vs. with extra_keys -> different hashes.
        let h_none = hash_tokens(&[1, 2, 3], 0, &[]);
        let h_one = hash_tokens(&[1, 2, 3], 0, &[42]);
        assert_ne!(h_none, h_one);

        // Same input -> deterministic.
        let h_one_again = hash_tokens(&[1, 2, 3], 0, &[42]);
        assert_eq!(h_one, h_one_again);

        // Order of extra_keys matters.
        let h_ab = hash_tokens(&[1, 2, 3], 0, &[42, 100]);
        let h_ba = hash_tokens(&[1, 2, 3], 0, &[100, 42]);
        assert_ne!(h_ab, h_ba);
    }

    #[test]
    fn test_find_longest_cache_hit_empty_registry() {
        let mut allocator = BlockAllocator::new(8, 4);
        let (blocks, n) = allocator.find_longest_cache_hit(&[1, 2, 3, 4, 5, 6, 7, 8], 4, &[], 0);
        assert!(blocks.is_empty());
        assert_eq!(n, 0);
    }

    #[test]
    fn test_find_longest_cache_hit_full_match() {
        let mut allocator = BlockAllocator::new(8, 4);
        let tokens: Vec<u32> = (0..8).collect();

        // Allocate two blocks and cache them.
        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        let blocks = [b0, b1];
        allocator
            .cache_full_blocks(&tokens, &blocks, 4, &[], 0)
            .unwrap();

        let (hit_blocks, n) = allocator.find_longest_cache_hit(&tokens, 4, &[], 0);
        assert_eq!(hit_blocks.len(), 2);
        assert_eq!(n, 8);
        assert_eq!(hit_blocks[0].block_id, blocks[0].block_id);
        assert_eq!(hit_blocks[1].block_id, blocks[1].block_id);
    }

    #[test]
    fn test_find_longest_cache_hit_partial_prefix() {
        // Cache 2 blocks (8 tokens). Lookup 12 tokens that share the first 8.
        // Third block was never cached -> hit count is 2 blocks / 8 tokens.
        let mut allocator = BlockAllocator::new(8, 4);
        let tokens_a: Vec<u32> = (0..8).collect();
        let mut tokens_b = tokens_a.clone();
        tokens_b.extend([100, 101, 102, 103]);

        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        allocator
            .cache_full_blocks(&tokens_a, &[b0, b1], 4, &[], 0)
            .unwrap();

        let (hit_blocks, n) = allocator.find_longest_cache_hit(&tokens_b, 4, &[], 0);
        assert_eq!(hit_blocks.len(), 2);
        assert_eq!(n, 8);
    }

    #[test]
    fn test_find_longest_cache_hit_chain_isolation() {
        // Cache 3 blocks for sequence A. Sequence B shares first block but
        // diverges in block 2. The hash chain must isolate -> only 1 block hits.
        let mut allocator = BlockAllocator::new(16, 4);
        let tokens_a: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let tokens_b: Vec<u32> = vec![1, 2, 3, 4, 99, 99, 99, 99, 9, 10, 11, 12];

        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        let b2 = allocator.allocate().unwrap();
        allocator
            .cache_full_blocks(&tokens_a, &[b0, b1, b2], 4, &[], 0)
            .unwrap();

        let (hit_blocks, n) = allocator.find_longest_cache_hit(&tokens_b, 4, &[], 0);
        assert_eq!(hit_blocks.len(), 1);
        assert_eq!(n, 4);
    }

    #[test]
    fn test_find_longest_cache_hit_short_input() {
        let mut allocator = BlockAllocator::new(4, 4);
        let (blocks, n) = allocator.find_longest_cache_hit(&[1, 2, 3], 4, &[], 0);
        assert!(blocks.is_empty());
        assert_eq!(n, 0);

        let (blocks, n) = allocator.find_longest_cache_hit(&[], 4, &[], 0);
        assert!(blocks.is_empty());
        assert_eq!(n, 0);
    }

    #[test]
    fn test_find_longest_cache_hit_zero_block_size() {
        // Defensive: block_size == 0 should not panic / infinite loop.
        let mut allocator = BlockAllocator::new(4, 4);
        let (blocks, n) = allocator.find_longest_cache_hit(&[1, 2, 3, 4], 0, &[], 0);
        assert!(blocks.is_empty());
        assert_eq!(n, 0);
    }

    #[test]
    fn test_cache_full_blocks_oversize_returns_err() {
        let mut allocator = BlockAllocator::new(4, 4);
        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        // 2 blocks * block_size 4 = 8 tokens required, but only 4 supplied.
        let res = allocator.cache_full_blocks(&[1, 2, 3, 4], &[b0, b1], 4, &[], 0);
        assert!(res.is_err());
    }

    #[test]
    fn test_cache_full_blocks_extra_keys_mismatch() {
        // Cache with extra_keys=[100], lookup with extra_keys=[] -> miss.
        let mut allocator = BlockAllocator::new(4, 4);
        let tokens: Vec<u32> = (0..8).collect();
        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        allocator
            .cache_full_blocks(&tokens, &[b0, b1], 4, &[100], 0)
            .unwrap();

        let (blocks, n) = allocator.find_longest_cache_hit(&tokens, 4, &[], 0);
        assert!(blocks.is_empty());
        assert_eq!(n, 0);

        // Same extra_keys -> hit.
        let (blocks, n) = allocator.find_longest_cache_hit(&tokens, 4, &[100], 0);
        assert_eq!(blocks.len(), 2);
        assert_eq!(n, 8);
    }

    /// `cache_salt` participates in block 0's hash only — not in
    /// blocks 1+ — matching vLLM's first-block-only semantics
    /// (`vllm/v1/core/kv_cache_utils.py:521-531`).
    ///
    /// Two properties are asserted here (the "salt is excluded from n > 0"
    /// invariant lives in [`cache_salt_not_mixed_into_block_n_for_n_gt_0`]
    /// where it's proved against the un-salted `hash_tokens` reference,
    /// which avoids the tautology of round-tripping with the same salt):
    ///
    /// 1. With `cache_salt = 0` (the "no salt" sentinel), behavior is
    ///    byte-equal to today: registering with `0` and looking up with `0`
    ///    is a hit; registering with `0` and looking up with non-zero is a
    ///    miss only because of the first-block-only mix-in.
    /// 2. Registering with `cache_salt = A` and looking up with
    ///    `cache_salt = B` (A != B, both non-zero) misses on block 0 — i.e.
    ///    no cross-tenant prefix reuse on the leading block.
    #[test]
    fn cache_salt_only_affects_first_block_hash() {
        // --- Property 1: cache_salt == 0 is byte-equal to today ---
        let mut alloc_zero_a = BlockAllocator::new(8, 4);
        let mut alloc_zero_b = BlockAllocator::new(8, 4);
        let tokens: Vec<u32> = (0..8).collect();
        for alloc in [&mut alloc_zero_a, &mut alloc_zero_b] {
            let b0 = alloc.allocate().unwrap();
            let b1 = alloc.allocate().unwrap();
            alloc
                .cache_full_blocks(&tokens, &[b0, b1], 4, &[], 0)
                .unwrap();
        }
        // Same salt=0 → both lookups hit fully.
        let (hits_a, n_a) = alloc_zero_a.find_longest_cache_hit(&tokens, 4, &[], 0);
        assert_eq!(hits_a.len(), 2, "salt=0 lookup must fully hit salt=0 chain");
        assert_eq!(n_a, 8);
        // salt=0 cache, salt!=0 lookup → block 0 hash differs → full miss.
        let (hits_b, n_b) = alloc_zero_b.find_longest_cache_hit(&tokens, 4, &[], 42);
        assert!(
            hits_b.is_empty(),
            "non-zero salt must NOT hit a salt=0-registered chain"
        );
        assert_eq!(n_b, 0);

        // --- Property 2: A != B isolates block 0 ---
        let mut allocator = BlockAllocator::new(8, 4);
        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        allocator
            .cache_full_blocks(&tokens, &[b0, b1], 4, &[], 0xAAAA_AAAA_AAAA_AAAA)
            .unwrap();
        // Different non-zero salt → block 0 misses → chain breaks → no hits.
        let (hits, n) = allocator.find_longest_cache_hit(&tokens, 4, &[], 0xBBBB_BBBB_BBBB_BBBB);
        assert!(
            hits.is_empty(),
            "different cache_salt must isolate block 0 (no cross-tenant first-block reuse)"
        );
        assert_eq!(n, 0);
        // Same salt → full hit.
        let (hits, n) = allocator.find_longest_cache_hit(&tokens, 4, &[], 0xAAAA_AAAA_AAAA_AAAA);
        assert_eq!(hits.len(), 2, "matching cache_salt must hit fully");
        assert_eq!(n, 8);
    }

    /// Direct-hash assertion that `cache_salt` is NOT mixed into the hash of
    /// any block at position `n > 0`. This is the strong-form proof that
    /// complements `cache_salt_only_affects_first_block_hash` (which is an
    /// end-to-end same-salt round-trip and is structurally tautological in
    /// isolation: registering with `salt=A` and looking up with `salt=A`
    /// would still pass on a buggy implementation that mixed `cache_salt`
    /// into every block, so long as the lookup used the same salt).
    ///
    /// Strategy: register a 3-block chain via the public
    /// `cache_full_blocks` API with a non-zero `cache_salt`, then walk the
    /// chain ourselves using the public `hash_tokens` primitive (no salt)
    /// and assert that each `n > 0` block is reachable at exactly the
    /// un-salted hash. We additionally assert that block 0 is NOT
    /// reachable at the un-salted hash (i.e. salt IS mixed in for n == 0)
    /// so the test isn't trivially satisfied by a hypothetical
    /// implementation that ignores salt entirely.
    ///
    /// A buggy implementation that mixed `cache_salt` into every block
    /// would fail this test: the un-salted lookup at block 1 would miss
    /// because the registered hash would have included the salt, while
    /// our recomputed reference hash would not.
    #[test]
    fn cache_salt_not_mixed_into_block_n_for_n_gt_0() {
        let mut allocator = BlockAllocator::new(8, 4);
        let toks: Vec<u32> = (0..12).collect();
        let salt = 0xDEAD_BEEF_DEAD_BEEFu64;

        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        let b2 = allocator.allocate().unwrap();
        let b0_id = b0.block_id;
        let b1_id = b1.block_id;
        let b2_id = b2.block_id;

        let blocks = vec![Arc::clone(&b0), Arc::clone(&b1), Arc::clone(&b2)];
        let registered = allocator
            .cache_full_blocks(&toks, &blocks, 4, &[], salt)
            .unwrap();
        assert_eq!(registered, 3);

        // Recompute the chain ourselves WITHOUT the salt — this is the
        // exact byte sequence `hash_block` produces for any block with
        // `block_index > 0` (or `cache_salt == 0`). We use the parent_hash
        // chain that the registered chain would produce given block 0's
        // SALTED hash; for n > 0 the salt does not appear in the
        // composition, so we feed the chained `previous_block_hash`
        // through plain `hash_tokens` and expect it to match.

        // Block 0: registered hash is the SALTED variant. The un-salted
        // hash MUST differ (otherwise salt is silently a no-op). Since
        // `register_prefix` keys the cache by the registered (salted)
        // hash, an un-salted lookup must miss.
        let unsalted_block0_hash = hash_tokens(&toks[0..4], 0, &[]);
        assert!(
            allocator.lookup_prefix(unsalted_block0_hash).is_none(),
            "block 0 must NOT be reachable at the un-salted hash when a non-zero cache_salt was registered \
             (otherwise salt is silently ignored at n == 0)"
        );

        // Reconstruct the chain's SALTED block-0 hash (parent for block 1)
        // by re-running the same effective-extra_keys composition that
        // `hash_block` does internally for the leading block: extra_keys
        // ++ [cache_salt]. This is the only place we need to know that
        // composition; for n > 0 we use plain `hash_tokens`.
        let salted_block0_hash = {
            let effective = vec![salt];
            hash_tokens(&toks[0..4], 0, &effective)
        };

        // Block 1: the registered hash MUST be `hash_tokens(block1_tokens,
        // salted_block0_hash, &[])` — i.e. NO salt mixed in for n > 0.
        // If a buggy implementation mixed the salt into every block, the
        // registered hash would differ from this and the lookup would
        // miss.
        let expected_block1_hash = hash_tokens(&toks[4..8], salted_block0_hash, &[]);
        let cached_block1 = allocator.lookup_prefix(expected_block1_hash).expect(
            "block 1 must be reachable at the un-salted hash_tokens(block1_tokens, parent_hash, &[]) — \
             salt MUST NOT be mixed into hashes for n > 0",
        );
        assert_eq!(
            cached_block1.block_id, b1_id,
            "lookup at the un-salted block-1 hash must resolve to the registered block 1"
        );
        // Drop the lookup-incref ref_count bump.
        allocator.free(cached_block1);

        // Block 2: chain off the un-salted block 1 hash and the same
        // un-salted composition.
        let expected_block2_hash = hash_tokens(&toks[8..12], expected_block1_hash, &[]);
        let cached_block2 = allocator.lookup_prefix(expected_block2_hash).expect(
            "block 2 must be reachable at the un-salted chained hash — salt MUST NOT be mixed into n > 0",
        );
        assert_eq!(
            cached_block2.block_id, b2_id,
            "lookup at the un-salted block-2 hash must resolve to the registered block 2"
        );
        allocator.free(cached_block2);

        // Sanity: block 0 IS reachable at its salted hash. Together with
        // the un-salted-block-0 miss above, this confirms salt was
        // actually mixed in at n == 0.
        let cached_block0 = allocator.lookup_prefix(salted_block0_hash).expect(
            "block 0 must be reachable at the salted hash (otherwise registration was a no-op)",
        );
        assert_eq!(cached_block0.block_id, b0_id);
        allocator.free(cached_block0);
    }

    /// `cache_salt` composes with non-empty uniform `extra_keys`
    /// per the contract `[existing_extra_keys..., cache_salt]` (salt
    /// APPENDED to the existing keys, not prepended, not replacing). This
    /// test pins down the exact byte composition by registering a chain
    /// with `cache_full_blocks` (uniform `extra_keys`) and a non-zero
    /// salt, then probing `lookup_prefix` against `hash_tokens(...)`
    /// references built with each candidate composition.
    ///
    /// The earlier salt tests (`cache_salt_only_affects_first_block_hash`
    /// and `cache_salt_not_mixed_into_block_n_for_n_gt_0`) only exercise
    /// the salt path with EMPTY `extra_keys`, so a mutation that drops
    /// `extra_keys` when `cache_salt != 0`, or reorders them to
    /// `[cache_salt, existing_extra_keys...]`, would still pass them.
    /// This test exists to kill those mutants.
    #[test]
    fn cache_salt_composes_with_uniform_extra_keys() {
        let mut allocator = BlockAllocator::new(8, 4);
        let toks: Vec<u32> = (0..8).collect();
        const K1: u64 = 0xCAFE_F00D_DEAD_BEEF;
        const K2: u64 = 0x0123_4567_89AB_CDEF;
        let salt: u64 = 0xDEAD_BEEF_DEAD_BEEFu64;

        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        let b0_id = b0.block_id;
        let b1_id = b1.block_id;

        let blocks = vec![Arc::clone(&b0), Arc::clone(&b1)];
        let registered = allocator
            .cache_full_blocks(&toks, &blocks, 4, &[K1, K2], salt)
            .unwrap();
        assert_eq!(registered, 2);

        // Property 1: block 0's hash equals
        // `hash_tokens(tokens[0..block_size], 0, &[K1, K2, salt])`.
        // Salt is APPENDED to the existing extra_keys.
        let salted_block0_hash = hash_tokens(&toks[0..4], 0, &[K1, K2, salt]);
        let cached_block0 = allocator.lookup_prefix(salted_block0_hash).expect(
            "block 0 must be reachable at hash_tokens(tokens, 0, &[K1, K2, salt]) — \
             salt is APPENDED to existing extra_keys for the leading block",
        );
        assert_eq!(
            cached_block0.block_id, b0_id,
            "lookup at the canonical [extra_keys..., salt] composition must resolve to block 0"
        );
        allocator.free(cached_block0);

        // Property 2: block 0's hash does NOT equal
        // `hash_tokens(tokens[0..block_size], 0, &[salt, K1, K2])`.
        // Order matters — salt is appended, not prepended.
        let wrong_order_hash = hash_tokens(&toks[0..4], 0, &[salt, K1, K2]);
        assert!(
            allocator.lookup_prefix(wrong_order_hash).is_none(),
            "block 0 must NOT be reachable at hash_tokens(tokens, 0, &[salt, K1, K2]) — \
             ordering is `[extra_keys..., salt]`, not `[salt, extra_keys...]`"
        );

        // Property 3: block 0's hash does NOT equal
        // `hash_tokens(tokens[0..block_size], 0, &[salt])`. Existing
        // extra_keys are preserved alongside salt, not dropped.
        let salt_only_hash = hash_tokens(&toks[0..4], 0, &[salt]);
        assert!(
            allocator.lookup_prefix(salt_only_hash).is_none(),
            "block 0 must NOT be reachable at hash_tokens(tokens, 0, &[salt]) — \
             existing extra_keys MUST NOT be dropped when cache_salt != 0"
        );

        // Property 4: block 1's hash equals
        // `hash_tokens(tokens[block_size..2*block_size], salted_block0_hash, &[K1, K2])`.
        // For n > 0 the salt is NOT mixed in, but the existing extra_keys
        // STILL ARE — they are uniform per-call.
        let expected_block1_hash = hash_tokens(&toks[4..8], salted_block0_hash, &[K1, K2]);
        let cached_block1 = allocator.lookup_prefix(expected_block1_hash).expect(
            "block 1 must be reachable at hash_tokens(tokens, parent, &[K1, K2]) — \
             extra_keys still thread through n > 0; salt does not",
        );
        assert_eq!(
            cached_block1.block_id, b1_id,
            "lookup at the un-salted block-1 hash must resolve to the registered block 1"
        );
        allocator.free(cached_block1);
    }

    /// Per-block-extra_keys variant of the salt composition
    /// contract. Each block has its OWN extra_keys vector
    /// (`extra_keys_per_block[n]`); salt is appended only to block 0's
    /// per-block keys, and blocks 1+ use their per-block keys verbatim
    /// without salt. This pins down the exact composition for the
    /// per-block path so a mutation that mishandles the
    /// non-empty-per-block-keys + non-zero-salt branch is killed.
    #[test]
    fn cache_salt_composes_with_per_block_extra_keys() {
        let mut allocator = BlockAllocator::new(8, 4);
        let toks: Vec<u32> = (0..12).collect();
        let per_block_keys: Vec<Vec<u64>> = vec![vec![0xAA], vec![0xBB], vec![0xCC]];
        let salt: u64 = 0xDEAD_BEEF_DEAD_BEEFu64;

        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        let b2 = allocator.allocate().unwrap();
        let b0_id = b0.block_id;
        let b1_id = b1.block_id;
        let b2_id = b2.block_id;

        let blocks = vec![Arc::clone(&b0), Arc::clone(&b1), Arc::clone(&b2)];
        let registered = allocator
            .cache_full_blocks_per_block(&toks, &blocks, 4, &per_block_keys, salt)
            .unwrap();
        assert_eq!(registered, 3);

        // Property 1: block 0's hash equals
        // `hash_tokens(tokens[0..block_size], 0, &[0xAA, salt])` —
        // salt appended AFTER block 0's per-block keys.
        let salted_block0_hash = hash_tokens(&toks[0..4], 0, &[0xAA, salt]);
        let cached_block0 = allocator.lookup_prefix(salted_block0_hash).expect(
            "block 0 must be reachable at hash_tokens(tokens, 0, &[0xAA, salt]) — \
             salt is APPENDED to block 0's per-block extra_keys",
        );
        assert_eq!(cached_block0.block_id, b0_id);
        allocator.free(cached_block0);

        // Property 2: block 1's hash equals
        // `hash_tokens(tokens[block_size..2*block_size], salted_block0_hash, &[0xBB])`.
        // Block 1 uses ITS per-block keys, NO salt.
        let expected_block1_hash = hash_tokens(&toks[4..8], salted_block0_hash, &[0xBB]);
        let cached_block1 = allocator.lookup_prefix(expected_block1_hash).expect(
            "block 1 must be reachable at hash_tokens(tokens, parent, &[0xBB]) — \
             block 1 uses its per-block keys; salt is NOT mixed into n > 0",
        );
        assert_eq!(cached_block1.block_id, b1_id);
        allocator.free(cached_block1);

        // Property 3: block 2's hash equals
        // `hash_tokens(tokens[2*block_size..3*block_size], expected_block1_hash, &[0xCC])`.
        let expected_block2_hash = hash_tokens(&toks[8..12], expected_block1_hash, &[0xCC]);
        let cached_block2 = allocator.lookup_prefix(expected_block2_hash).expect(
            "block 2 must be reachable at hash_tokens(tokens, parent, &[0xCC]) — \
             block 2 uses its own per-block keys",
        );
        assert_eq!(cached_block2.block_id, b2_id);
        allocator.free(cached_block2);

        // Wrong-block per-block keys must miss: looking up block 1 with
        // block 0's keys (`&[0xAA]`) instead of its own (`&[0xBB]`)
        // should NOT resolve. Guards against any "use block 0's keys for
        // every block" mutation.
        let wrong_block1_hash = hash_tokens(&toks[4..8], salted_block0_hash, &[0xAA]);
        assert!(
            allocator.lookup_prefix(wrong_block1_hash).is_none(),
            "block 1 must NOT be reachable with block 0's per-block keys — \
             per-block keys MUST be applied per-block, not uniformly"
        );
    }

    #[test]
    fn test_register_lookup_refcount_lifecycle() {
        let mut allocator = BlockAllocator::new(4, 4);
        let block = allocator.allocate().unwrap();
        // Newly allocated -> ref_count == 1.
        assert_eq!(block.get_ref_count(), 1);

        let hash = hash_tokens(&[1, 2, 3, 4], 0, &[]);
        allocator.register_prefix(Arc::clone(&block), hash);
        // register_prefix incref's so the cache holds its own logical ref:
        // allocate(1) + register(1) = 2.
        assert_eq!(block.get_ref_count(), 2);

        let cached = allocator.lookup_prefix(hash).unwrap();
        // lookup_prefix increments ref_count: 2 + 1 = 3.
        assert_eq!(cached.get_ref_count(), 3);

        // Free the lookup'd handle: 3 -> 2, block stays alive.
        allocator.free(cached);
        assert_eq!(block.get_ref_count(), 2);
        assert_eq!(allocator.num_free_blocks(), 3);

        // Free the external handle: 2 -> 1, block STILL alive (cache's
        // logical ref). No cleanup yet — that happens via LRU eviction or
        // when the cache's ref is the last and it gets removed.
        allocator.free(block);
        assert_eq!(allocator.num_free_blocks(), 3);
        assert!(allocator.lookup_prefix(hash).is_some());
    }

    #[test]
    fn test_lru_eviction_order() {
        let mut allocator = BlockAllocator::new(8, 4);
        allocator.max_prefix_cache_entries = 3;

        // Register 4 blocks with distinct hashes; the oldest registration
        // (hash1) should be evicted from the prefix_cache after the 4th insert.
        let b1 = allocator.allocate().unwrap();
        let b2 = allocator.allocate().unwrap();
        let b3 = allocator.allocate().unwrap();
        let b4 = allocator.allocate().unwrap();

        let h1 = 0xAAAA_AAAA_AAAA_AAAA;
        let h2 = 0xBBBB_BBBB_BBBB_BBBB;
        let h3 = 0xCCCC_CCCC_CCCC_CCCC;
        let h4 = 0xDDDD_DDDD_DDDD_DDDD;

        allocator.register_prefix(Arc::clone(&b1), h1);
        allocator.register_prefix(Arc::clone(&b2), h2);
        allocator.register_prefix(Arc::clone(&b3), h3);
        allocator.register_prefix(Arc::clone(&b4), h4);

        // h1 (oldest) was evicted; h2/h3/h4 remain.
        assert!(allocator.lookup_prefix(h1).is_none());
        assert!(allocator.lookup_prefix(h2).is_some());
        assert!(allocator.lookup_prefix(h3).is_some());
        assert!(allocator.lookup_prefix(h4).is_some());

        // block_hashes was also cleaned for the evicted entry.
        assert!(!allocator.block_hashes.contains_key(&b1.block_id));
        assert!(allocator.block_hashes.contains_key(&b2.block_id));
        assert!(allocator.block_hashes.contains_key(&b3.block_id));
        assert!(allocator.block_hashes.contains_key(&b4.block_id));

        // The evicted block (b1) had its cache ref released — ref_count is
        // now 1 (just the external b1 handle). The lookup_prefix(h1) call
        // returned None, so it did NOT incref. b2/b3/b4 are at higher counts
        // because lookup_prefix incref'd them.
        assert_eq!(b1.get_ref_count(), 1);
    }

    // register_prefix must evict stale aliases when the same block is
    // re-registered under a different hash, otherwise a freed block can
    // leak back through lookup_prefix on the stale hash.

    #[test]
    fn test_register_prefix_re_registers_same_block_different_hash() {
        // Allocate one block, register under hash A, then re-register the
        // SAME block under hash B. The stale alias on hash A must be
        // evicted (Case 1 displacement); the block's net ref_count is
        // unchanged since the cache decrefs the old alias and increfs
        // the new one.
        let mut allocator = BlockAllocator::new(4, 4);
        let initial_free = allocator.num_free_blocks();

        let block = allocator.allocate().unwrap();
        let hash_a = 0xAAAA_AAAA_AAAA_AAAA;
        let hash_b = 0xBBBB_BBBB_BBBB_BBBB;

        assert!(
            allocator.register_prefix(Arc::clone(&block), hash_a),
            "initial registration must be accepted",
        );
        // After first register: alloc(1) + register(1) = 2.
        assert_eq!(block.get_ref_count(), 2);

        assert!(
            allocator.register_prefix(Arc::clone(&block), hash_b),
            "Case 1 same-block-different-hash must be accepted",
        );
        // Case 1: cache decref's old hash_a ref, then increfs new hash_b
        // ref. Net unchanged: still 2.
        assert_eq!(block.get_ref_count(), 2);

        // Stale alias evicted; current alias resolves.
        assert!(allocator.lookup_prefix(hash_a).is_none());
        let cached = allocator.lookup_prefix(hash_b).unwrap();
        assert_eq!(cached.block_id, block.block_id);
        // alloc(1) + register(1) + lookup(1) = 3.
        assert_eq!(cached.get_ref_count(), 3);

        // Free the lookup and external handles — cache's ref still holds
        // the block, so it stays in the cache and is NOT in the free pool.
        allocator.free(cached); // 3 -> 2
        allocator.free(block); // 2 -> 1

        assert!(allocator.lookup_prefix(hash_a).is_none());
        // Cache still holds the block under hash_b.
        assert!(allocator.lookup_prefix(hash_b).is_some());
        assert_eq!(allocator.num_free_blocks(), initial_free - 1);
    }

    #[test]
    fn test_cache_full_blocks_extra_keys_re_register_isolation() {
        // Cache the same blocks under two different extra_keys (no_keys
        // vs [99]). After freeing, neither
        // hash can hand back the freed block via find_longest_cache_hit —
        // i.e. extra_keys isolation must hold across freed entries.
        let mut allocator = BlockAllocator::new(4, 4);
        let tokens: Vec<u32> = vec![1, 2, 3, 4];

        let b0 = allocator.allocate().unwrap();
        let blocks = [Arc::clone(&b0)];

        allocator
            .cache_full_blocks(&tokens, &blocks, 4, &[], 0)
            .unwrap();
        // After first cache_full_blocks: alloc(1) + register(1) = 2.
        assert_eq!(b0.get_ref_count(), 2);

        allocator
            .cache_full_blocks(&tokens, &blocks, 4, &[99], 0)
            .unwrap();
        // Second cache_full_blocks: same block under different hash →
        // Case 1 path. Decref old, incref new → net unchanged = 2.
        assert_eq!(b0.get_ref_count(), 2);

        // Free the external handle. ref_count: 2 -> 1. The cache still
        // holds the block under hash_b (extra_keys=[99]).
        allocator.free(b0);

        // The empty-extra_keys alias was displaced by the second register;
        // it is gone.
        let (hits_none, n_none) = allocator.find_longest_cache_hit(&tokens, 4, &[], 0);
        assert!(hits_none.is_empty(), "stale extra_keys=[] alias leaked");
        assert_eq!(n_none, 0);

        // The [99] alias still resolves — block survived because the cache
        // still holds its logical reference. This is correct: the cached
        // block is consistent with extra_keys=[99] and a future lookup
        // with the matching extra_keys is a legitimate hit.
        let (hits_99, n_99) = allocator.find_longest_cache_hit(&tokens, 4, &[99], 0);
        assert_eq!(hits_99.len(), 1, "extra_keys=[99] alias still resolves");
        assert_eq!(n_99, 4);
    }

    #[test]
    fn test_register_prefix_collision_drops_new() {
        // If two DIFFERENT blocks register under the same hash, the second
        // call is dropped (no-op). The first registration stays
        // authoritative; block_a's ref_count includes the cache's logical
        // ref; block_b's is unchanged because it was never inserted.
        let mut allocator = BlockAllocator::new(4, 4);
        let initial_free = allocator.num_free_blocks();

        let block_a = allocator.allocate().unwrap();
        let block_b = allocator.allocate().unwrap();
        assert_ne!(block_a.block_id, block_b.block_id);

        let hash_x = 0xFEED_FEED_FEED_FEED;

        assert!(
            allocator.register_prefix(Arc::clone(&block_a), hash_x),
            "first registration must be accepted",
        );
        // alloc(1) + register(1) = 2.
        assert_eq!(block_a.get_ref_count(), 2);

        assert!(
            !allocator.register_prefix(Arc::clone(&block_b), hash_x),
            "collision attempt must be rejected (returns false)",
        );
        // Collision drop: nothing inserted, block_b unchanged.
        assert_eq!(block_b.get_ref_count(), 1);
        // block_a still authoritative; ref_count unchanged.
        assert_eq!(block_a.get_ref_count(), 2);

        // block_a stays authoritative for hash_x.
        let cached = allocator.lookup_prefix(hash_x).unwrap();
        assert_eq!(cached.block_id, block_a.block_id);
        // block_b was NOT inserted into block_hashes.
        assert!(!allocator.block_hashes.contains_key(&block_b.block_id));

        // Free the lookup'd handle (decrements block_a refcount: 3 -> 2).
        allocator.free(cached);
        // Free block_a external handle → 2 -> 1. Cache still holds block_a
        // under hash_x at ref_count = 1.
        allocator.free(block_a);
        assert!(allocator.lookup_prefix(hash_x).is_some());

        // Free block_b → no-op on the cache (was never registered),
        // block returns to free pool.
        allocator.free(block_b);
        // block_a still pinned in cache; only block_b returned.
        assert_eq!(allocator.num_free_blocks(), initial_free - 1);
    }

    #[test]
    fn test_register_prefix_idempotent_at_capacity() {
        // At capacity, re-registering the SAME (block, hash) pair must be a
        // pure LRU refresh — it must NOT trigger capacity eviction of
        // unrelated entries, since the cache size doesn't grow.
        let mut allocator = BlockAllocator::new(8, 4);
        allocator.max_prefix_cache_entries = 3;

        let block_1 = allocator.allocate().unwrap();
        let block_2 = allocator.allocate().unwrap();
        let block_3 = allocator.allocate().unwrap();

        let h1 = 0x1111_1111_1111_1111;
        let h2 = 0x2222_2222_2222_2222;
        let h3 = 0x3333_3333_3333_3333;

        allocator.register_prefix(Arc::clone(&block_1), h1);
        allocator.register_prefix(Arc::clone(&block_2), h2);
        allocator.register_prefix(Arc::clone(&block_3), h3);

        // Cache at full capacity (3/3).
        assert_eq!(allocator.prefix_cache.len(), 3);
        let free_before = allocator.num_free_blocks();

        // Re-register the MIDDLE entry (block_2 under h2). This is an
        // idempotent refresh — must not evict h1 or h3.
        allocator.register_prefix(Arc::clone(&block_2), h2);

        // All three entries still resolvable.
        assert!(
            allocator.lookup_prefix(h1).is_some(),
            "h1 must not be evicted by an idempotent re-register"
        );
        assert!(
            allocator.lookup_prefix(h3).is_some(),
            "h3 must not be evicted by an idempotent re-register"
        );
        assert!(allocator.lookup_prefix(h2).is_some());

        // num_free_blocks unchanged (no extra allocations / frees).
        assert_eq!(allocator.num_free_blocks(), free_before);

        // After the refresh, h2 should be the most-recently-used entry.
        // (lookup_prefix calls above also bumped h1, h3, h2 in order, so
        // we re-check by triggering one more eviction: register a 4th hash
        // and confirm h1 — currently the LRU after the lookups — gets
        // evicted, not h2 or h3.)
        let block_4 = allocator.allocate().unwrap();
        let h4 = 0x4444_4444_4444_4444;
        allocator.register_prefix(Arc::clone(&block_4), h4);

        // h1 was the oldest (first in lru_order after the lookups).
        assert!(
            allocator.lookup_prefix(h1).is_none(),
            "h1 should be the LRU after subsequent lookups bumped h3 and h2"
        );
        assert!(allocator.lookup_prefix(h2).is_some());
        assert!(allocator.lookup_prefix(h3).is_some());
        assert!(allocator.lookup_prefix(h4).is_some());
    }

    #[test]
    fn test_register_prefix_refresh_moves_hash_to_mru() {
        // Discriminating regression test for the core Task-13 semantic:
        // `LinkedHashSet::insert` of an ALREADY-PRESENT key must move it to
        // the back (MRU), exactly like the old `retain(..); push_back(..)`.
        // A hypothetical broken swap (e.g. `replace`, which does NOT reposition
        // an existing entry) would leave the refreshed hash at the front and
        // get it wrongly evicted. This test fails under that broken variant.
        //
        // Unlike `test_register_prefix_idempotent_at_capacity`, this test
        // never calls `lookup_prefix` before the final assertions (lookup also
        // bumps LRU order and would confound the outcome). It refreshes the
        // OLDEST entry via `register_prefix` and checks membership through the
        // non-bumping `prefix_cache.contains_key`.
        let mut allocator = BlockAllocator::new(8, 4);
        allocator.max_prefix_cache_entries = 3;

        let block_1 = allocator.allocate().unwrap();
        let block_2 = allocator.allocate().unwrap();
        let block_3 = allocator.allocate().unwrap();

        let h1 = 0x1111_1111_1111_1111;
        let h2 = 0x2222_2222_2222_2222;
        let h3 = 0x3333_3333_3333_3333;

        // Insertion order → lru_order front..back = [h1, h2, h3]. h1 is oldest.
        allocator.register_prefix(Arc::clone(&block_1), h1);
        allocator.register_prefix(Arc::clone(&block_2), h2);
        allocator.register_prefix(Arc::clone(&block_3), h3);
        assert_eq!(allocator.prefix_cache.len(), 3);

        // Idempotent refresh of the OLDEST entry (block_1 under h1). Correct
        // move-to-back → order becomes [h2, h3, h1], making h2 the new LRU.
        // (Broken no-reposition → order stays [h1, h2, h3], h1 still LRU.)
        allocator.register_prefix(Arc::clone(&block_1), h1);

        // Force exactly one capacity eviction with a genuinely new hash.
        let block_4 = allocator.allocate().unwrap();
        let h4 = 0x4444_4444_4444_4444;
        allocator.register_prefix(Arc::clone(&block_4), h4);

        // Correct move-to-back: h2 (new LRU after the refresh) is evicted;
        // the refreshed h1 survives. Under a broken swap, h1 would be gone.
        assert!(
            allocator.prefix_cache.contains_key(&h1),
            "refreshing h1 must move it to MRU so it survives the next eviction"
        );
        assert!(
            !allocator.prefix_cache.contains_key(&h2),
            "h2 became the LRU after h1's refresh and must be the evicted entry"
        );
        assert!(
            allocator.prefix_cache.contains_key(&h3),
            "h3 was newer than the evicted LRU and must survive"
        );
        assert!(
            allocator.prefix_cache.contains_key(&h4),
            "h4 is the freshly inserted entry"
        );
        assert_eq!(allocator.prefix_cache.len(), 3);
    }

    #[test]
    fn test_register_prefix_collision_preserves_incoming_block_old_hash() {
        // block_a registered under hash_a, block_b registered under hash_b.
        // Then register_prefix(block_b, hash_a) — a collision attempt.
        // The collision drop must be a true no-op for block_b: hash_a stays
        // pointing at block_a, AND block_b's prior valid alias under hash_b
        // must NOT be torn down.
        let mut allocator = BlockAllocator::new(4, 4);

        let block_a = allocator.allocate().unwrap();
        let block_b = allocator.allocate().unwrap();
        assert_ne!(block_a.block_id, block_b.block_id);

        let hash_a = 0xAAAA_AAAA_AAAA_AAAA;
        let hash_b = 0xBBBB_BBBB_BBBB_BBBB;

        allocator.register_prefix(Arc::clone(&block_a), hash_a);
        allocator.register_prefix(Arc::clone(&block_b), hash_b);

        // Collision attempt: try to register block_b under hash_a.
        allocator.register_prefix(Arc::clone(&block_b), hash_a);

        // hash_a still maps to block_a (unchanged).
        let cached_a = allocator.lookup_prefix(hash_a).unwrap();
        assert_eq!(
            cached_a.block_id, block_a.block_id,
            "hash_a must still resolve to block_a after collision drop"
        );

        // hash_b STILL maps to block_b — its prior valid entry was preserved
        // despite the collision attempt.
        let cached_b = allocator.lookup_prefix(hash_b).unwrap();
        assert_eq!(
            cached_b.block_id, block_b.block_id,
            "hash_b alias of block_b must survive a collision drop on hash_a"
        );
    }

    /// Regression test for the orphaned-block leak: when the prefix_cache
    /// exceeds capacity, the LRU eviction loop must release the cache's
    /// logical reference. If the evicted block has no other holder, it
    /// must return to the free pool (otherwise the pool drains
    /// monotonically until allocation fails).
    #[test]
    fn test_lru_eviction_returns_block_to_free_pool() {
        let mut allocator = BlockAllocator::new(8, 4);
        allocator.max_prefix_cache_entries = 2;
        let initial_free = allocator.num_free_blocks();

        // Allocate three blocks, register all three. The third register
        // exceeds capacity → LRU eviction of the first.
        let b1 = allocator.allocate().unwrap();
        let h1 = 0x1111_1111_1111_1111;
        allocator.register_prefix(Arc::clone(&b1), h1);

        let b2 = allocator.allocate().unwrap();
        let h2 = 0x2222_2222_2222_2222;
        allocator.register_prefix(Arc::clone(&b2), h2);

        let b3 = allocator.allocate().unwrap();
        let h3 = 0x3333_3333_3333_3333;

        // Drop b1's external handle BEFORE the eviction so b1's only
        // remaining ref is the cache's logical ref. Triggering eviction
        // must drive ref_count to 0 and return b1 to the free pool.
        let b1_id = b1.block_id;
        allocator.free(b1); // b1: alloc(1)+reg(1)=2 → 1 (cache only)

        // b1 is still pinned in the cache — free pool didn't get it back.
        assert_eq!(
            allocator.num_free_blocks(),
            initial_free - 3,
            "before eviction: 3 blocks held (b1 by cache, b2/b3 external+cache)"
        );

        // Now register b3, evicting b1 (oldest in LRU order).
        allocator.register_prefix(Arc::clone(&b3), h3);

        // b1 was evicted, decref'd from 1 → 0, removed from `allocated`,
        // pushed back to free_blocks.
        assert!(allocator.lookup_prefix(h1).is_none());
        assert!(!allocator.allocated.contains_key(&b1_id));
        assert!(
            allocator.free_blocks.contains(&b1_id),
            "evicted block must return to free_blocks"
        );

        // b2 and b3 still alive (external handle + cache ref).
        assert!(allocator.lookup_prefix(h2).is_some());
        assert!(allocator.lookup_prefix(h3).is_some());

        // Free pool: b1 came back; b2 and b3 are still held by the cache
        // and their external handles. Net: initial_free - 2.
        assert_eq!(
            allocator.num_free_blocks(),
            initial_free - 2,
            "evicted block must return to the free pool"
        );
    }

    // -------------------------------------------------------------------
    // Allocation under cache pressure: when free_blocks is empty,
    // allocate() must evict an LRU cache-only block to satisfy the
    // request. Mirrors vLLM `_maybe_evict_cached_block`.
    // -------------------------------------------------------------------

    /// `allocate()` must fall back to evicting the LRU oldest cache-only
    /// block when the free pool is empty. The OLDEST entry (by LRU
    /// order, not insertion-of-cache-only) is the one selected.
    #[test]
    fn test_allocate_evicts_lru_when_pool_exhausted() {
        let mut allocator = BlockAllocator::new(3, 4);

        // Allocate all three blocks.
        let b1 = allocator.allocate().unwrap();
        let b2 = allocator.allocate().unwrap();
        let b3 = allocator.allocate().unwrap();
        let b1_id = b1.block_id;
        let b2_id = b2.block_id;
        let b3_id = b3.block_id;

        // Register each under a distinct hash, in order. b1 is therefore
        // the LRU oldest entry in `lru_order`.
        let h1 = 0x1111_1111_1111_1111;
        let h2 = 0x2222_2222_2222_2222;
        let h3 = 0x3333_3333_3333_3333;
        allocator.register_prefix(Arc::clone(&b1), h1);
        allocator.register_prefix(Arc::clone(&b2), h2);
        allocator.register_prefix(Arc::clone(&b3), h3);

        // Free each external handle. The cache's logical ref keeps each
        // block alive at ref_count = 1.
        allocator.free(b1);
        allocator.free(b2);
        allocator.free(b3);

        // Pool exhausted; all three blocks are cache-pinned.
        assert_eq!(allocator.num_free_blocks(), 0);
        assert_eq!(allocator.prefix_cache.len(), 3);

        // Allocate again — must succeed by evicting the LRU oldest (b1).
        let new_block = allocator
            .allocate()
            .expect("allocate must evict LRU cache-only block when pool empty");
        assert_eq!(
            new_block.block_id, b1_id,
            "evicted block must be the LRU oldest (b1)"
        );

        // b1's hash is gone from the cache; b2 and b3 remain.
        assert!(!allocator.prefix_cache.contains_key(&h1));
        assert!(allocator.prefix_cache.contains_key(&h2));
        assert!(allocator.prefix_cache.contains_key(&h3));
        assert!(!allocator.block_hashes.contains_key(&b1_id));
        assert!(allocator.block_hashes.contains_key(&b2_id));
        assert!(allocator.block_hashes.contains_key(&b3_id));
    }

    /// When NO cache entry is evictable (every cached block has a live
    /// request handle, ref_count > 1), `allocate()` returns None.
    #[test]
    fn test_allocate_returns_none_when_all_cache_in_use() {
        let mut allocator = BlockAllocator::new(2, 4);

        // Allocate two blocks; register each but DO NOT free the
        // request's handle. Each ends up at ref_count = 2 (alloc + cache).
        let b1 = allocator.allocate().unwrap();
        let b2 = allocator.allocate().unwrap();
        let h1 = 0xAAAA_AAAA_AAAA_AAAA;
        let h2 = 0xBBBB_BBBB_BBBB_BBBB;
        allocator.register_prefix(Arc::clone(&b1), h1);
        allocator.register_prefix(Arc::clone(&b2), h2);

        // Pool exhausted, but both cache entries are in-use by live
        // request handles (ref_count == 2). No eviction possible.
        assert_eq!(allocator.num_free_blocks(), 0);
        assert_eq!(b1.get_ref_count(), 2);
        assert_eq!(b2.get_ref_count(), 2);

        // allocate must return None — there's nothing it can give us.
        assert!(
            allocator.allocate().is_none(),
            "allocate must return None when free pool is empty AND every \
             cache entry has a live request handle"
        );
    }

    /// `can_allocate` must count evictable cache-only blocks alongside
    /// genuinely-free blocks — otherwise schedulers refuse work the
    /// allocator could in fact satisfy via the eviction fallback.
    #[test]
    fn test_can_allocate_counts_evictable_cache_blocks() {
        let mut allocator = BlockAllocator::new(2, 4);

        // Allocate, register, free both blocks. Both end up cache-pinned
        // at ref_count = 1.
        let b1 = allocator.allocate().unwrap();
        let b2 = allocator.allocate().unwrap();
        let h1 = 0x1111_1111_1111_1111;
        let h2 = 0x2222_2222_2222_2222;
        allocator.register_prefix(Arc::clone(&b1), h1);
        allocator.register_prefix(Arc::clone(&b2), h2);
        allocator.free(b1);
        allocator.free(b2);

        // Free pool empty, but both blocks evictable.
        assert_eq!(allocator.num_free_blocks(), 0);
        assert!(
            allocator.can_allocate(1),
            "1 cache-only block is evictable → can_allocate(1) must be true"
        );
        assert!(
            allocator.can_allocate(2),
            "2 cache-only blocks are evictable → can_allocate(2) must be true"
        );
        assert!(
            !allocator.can_allocate(3),
            "only 2 evictable; can_allocate(3) must be false"
        );
    }

    // register_prefix bool return contract + cache_full_blocks abort-on-
    // collision behavior. Closes the silent-corruption path where a
    // dropped registration in the middle of a chain would leave later
    // blocks linked to a ghost predecessor.

    /// `register_prefix` must return the right bool for each of its four
    /// paths: insertion, idempotent refresh, Case 1 displacement, Case 2
    /// collision.
    #[test]
    fn test_register_prefix_returns_correct_bool() {
        let mut allocator = BlockAllocator::new(8, 4);
        allocator.max_prefix_cache_entries = 4;

        let b1 = allocator.allocate().unwrap();
        let b2 = allocator.allocate().unwrap();
        assert_ne!(b1.block_id, b2.block_id);

        let h1 = 0x1111_1111_1111_1111;
        let h2 = 0x2222_2222_2222_2222;

        // Genuine insertion → true.
        assert!(
            allocator.register_prefix(Arc::clone(&b1), h1),
            "insertion path must return true",
        );

        // Idempotent refresh (same block, same hash) → true.
        assert!(
            allocator.register_prefix(Arc::clone(&b1), h1),
            "idempotent refresh must return true",
        );

        // Case 1 displacement (same block, different hash) → true.
        assert!(
            allocator.register_prefix(Arc::clone(&b1), h2),
            "Case 1 same-block-different-hash must return true",
        );

        // Case 2 collision (same hash, different block) → false.
        // h2 is now owned by b1; registering b2 under h2 must be rejected.
        assert!(
            !allocator.register_prefix(Arc::clone(&b2), h2),
            "Case 2 collision must return false",
        );

        // Sanity: h2 still resolves to b1 (collision drop preserved
        // existing entry).
        let cached = allocator.lookup_prefix(h2).unwrap();
        assert_eq!(cached.block_id, b1.block_id);
    }

    /// When the FIRST block in a `cache_full_blocks` chain collides with
    /// an existing prefix-cache entry, the entire chain must be aborted —
    /// no subsequent blocks may be registered. Otherwise a later
    /// `find_longest_cache_hit` would mix the existing block (under the
    /// colliding hash) with the chain's own descendants, producing a KV
    /// prefix that never coherently existed.
    #[test]
    fn test_cache_full_blocks_aborts_on_collision() {
        let mut allocator = BlockAllocator::new(8, 4);

        let tokens: Vec<u32> = (0..12).collect(); // 3 full blocks.
        let block_size = 4u32;

        // Pre-occupy block 0's hash with a DIFFERENT block. Compute the
        // hash the chain's first block would produce (parent_hash = 0,
        // first 4 tokens, no extra_keys).
        let block0_hash = hash_tokens(&tokens[0..4], 0, &[]);
        let intruder = allocator.allocate().unwrap();
        assert!(allocator.register_prefix(Arc::clone(&intruder), block0_hash));

        // Now allocate the chain's blocks and try to cache them.
        let chain_b0 = allocator.allocate().unwrap();
        let chain_b1 = allocator.allocate().unwrap();
        let chain_b2 = allocator.allocate().unwrap();
        assert_ne!(chain_b0.block_id, intruder.block_id);

        let registered = allocator
            .cache_full_blocks(
                &tokens,
                &[
                    Arc::clone(&chain_b0),
                    Arc::clone(&chain_b1),
                    Arc::clone(&chain_b2),
                ],
                block_size,
                &[],
                0,
            )
            .unwrap();

        // The chain aborted on the very first block; nothing was registered.
        assert_eq!(registered, 0, "chain must abort at block 0");

        // The intruder is still authoritative for block_0_hash.
        let resolved = allocator.lookup_prefix(block0_hash).unwrap();
        assert_eq!(resolved.block_id, intruder.block_id);

        // Crucially: blocks 1 and 2 of the chain were NOT registered. A
        // future find_longest_cache_hit on the full sequence must see at
        // most the intruder's block (which doesn't even belong to this
        // sequence's chain — but find_longest_cache_hit can't tell, so it
        // would still return the intruder for block 0). The point is that
        // chain_b1 and chain_b2 are not in the cache.
        assert!(!allocator.block_hashes.contains_key(&chain_b1.block_id));
        assert!(!allocator.block_hashes.contains_key(&chain_b2.block_id));
    }

    /// When a collision strikes mid-chain (block 1 collides, block 0 is
    /// fine), the chain registers block 0 then aborts. Blocks 2+ must not
    /// be registered. A future `find_longest_cache_hit` on the full
    /// sequence sees exactly 1 hit (block 0), then misses on block 1.
    #[test]
    fn test_cache_full_blocks_partial_chain() {
        let mut allocator = BlockAllocator::new(8, 4);

        let tokens: Vec<u32> = (0..12).collect();
        let block_size = 4u32;

        // The hash for chain block 1 chains off block 0's hash.
        let block0_hash = hash_tokens(&tokens[0..4], 0, &[]);
        let block1_hash = hash_tokens(&tokens[4..8], block0_hash, &[]);

        // Pre-occupy block 1's hash with a different block.
        let intruder = allocator.allocate().unwrap();
        assert!(allocator.register_prefix(Arc::clone(&intruder), block1_hash));

        // Run the chain.
        let chain_b0 = allocator.allocate().unwrap();
        let chain_b1 = allocator.allocate().unwrap();
        let chain_b2 = allocator.allocate().unwrap();
        let chain_b0_id = chain_b0.block_id;
        let chain_b1_id = chain_b1.block_id;
        let chain_b2_id = chain_b2.block_id;

        let registered = allocator
            .cache_full_blocks(
                &tokens,
                &[
                    Arc::clone(&chain_b0),
                    Arc::clone(&chain_b1),
                    Arc::clone(&chain_b2),
                ],
                block_size,
                &[],
                0,
            )
            .unwrap();

        // Block 0 succeeded; block 1 collided and aborted; block 2 never ran.
        assert_eq!(registered, 1);

        // Block 0 is registered under its hash.
        assert!(allocator.prefix_cache.contains_key(&block0_hash));
        let resolved_b0 = allocator.lookup_prefix(block0_hash).unwrap();
        assert_eq!(resolved_b0.block_id, chain_b0_id);

        // Block 1's hash still belongs to the intruder.
        let resolved_b1 = allocator.lookup_prefix(block1_hash).unwrap();
        assert_eq!(resolved_b1.block_id, intruder.block_id);

        // Block 2's hash was never computed/inserted.
        let block2_hash = hash_tokens(&tokens[8..12], block1_hash, &[]);
        assert!(!allocator.prefix_cache.contains_key(&block2_hash));

        // chain_b1 and chain_b2 are not tracked in block_hashes.
        assert!(!allocator.block_hashes.contains_key(&chain_b1_id));
        assert!(!allocator.block_hashes.contains_key(&chain_b2_id));

        // Walking find_longest_cache_hit on the full sequence: block 0
        // hits (chain_b0), block 1 misses (intruder is under that hash but
        // find_longest_cache_hit chains hashes off the previous block's
        // hash, not what the cache contains — so it would compute
        // block1_hash which DOES resolve to the intruder; let's just make
        // sure the function doesn't blow up and returns a sensible answer).
        // The key invariant we're testing is that we did NOT register
        // ghost descendants — chain_b2's hash isn't cached.
        assert!(!allocator.prefix_cache.contains_key(&block2_hash));
    }

    /// Cold-prefill replay can recompute a leading prefix whose blocks are
    /// already cached under older physical blocks. That must not abort the
    /// whole registration chain; otherwise the newly computed tail never
    /// becomes reusable and the next request keeps falling back to cold
    /// prefill at the same stale prefix.
    #[test]
    fn test_cache_full_blocks_continues_past_verified_existing_prefix() {
        let mut allocator = BlockAllocator::new(12, 4);
        let block_size = 4u32;

        let old_tokens: Vec<u32> = (0..8).collect();
        let old_blocks: Vec<_> = (0..2).map(|_| allocator.allocate().unwrap()).collect();
        let old_ids: Vec<_> = old_blocks.iter().map(|block| block.block_id).collect();
        let old_registered = allocator
            .cache_full_blocks(&old_tokens, &old_blocks, block_size, &[], 0)
            .unwrap();
        assert_eq!(old_registered, 2);

        // Simulate end-of-request: the prefix cache keeps the old blocks
        // alive while the request's direct handles are released.
        for block in old_blocks {
            allocator.free(block);
        }

        let new_tokens: Vec<u32> = (0..16).collect();
        let new_blocks: Vec<_> = (0..4).map(|_| allocator.allocate().unwrap()).collect();
        let new_ids: Vec<_> = new_blocks.iter().map(|block| block.block_id).collect();
        assert_ne!(new_ids[0], old_ids[0]);
        assert_ne!(new_ids[1], old_ids[1]);

        let new_registered = allocator
            .cache_full_blocks(&new_tokens, &new_blocks, block_size, &[], 0)
            .unwrap();
        assert_eq!(
            new_registered, 2,
            "only the new tail blocks should be newly registered"
        );

        let (hits, cached_tokens) =
            allocator.find_longest_cache_hit(&new_tokens, block_size, &[], 0);
        assert_eq!(cached_tokens, new_tokens.len());
        let hit_ids: Vec<_> = hits.iter().map(|block| block.block_id).collect();
        assert_eq!(
            hit_ids,
            vec![old_ids[0], old_ids[1], new_ids[2], new_ids[3]],
            "lookup should reuse the old cached prefix and the newly published tail"
        );

        assert!(
            !allocator.block_hashes.contains_key(&new_ids[0]),
            "duplicate leading block 0 must not be inserted as a cache alias"
        );
        assert!(
            !allocator.block_hashes.contains_key(&new_ids[1]),
            "duplicate leading block 1 must not be inserted as a cache alias"
        );
        assert!(allocator.block_hashes.contains_key(&new_ids[2]));
        assert!(allocator.block_hashes.contains_key(&new_ids[3]));
    }

    // Per-block extra_keys variants for multimodal cache isolation.

    /// Per-block API with all-empty per-block keys must produce IDENTICAL
    /// hashes (and therefore IDENTICAL cache hits) to the uniform API with
    /// `extra_keys=&[]`. This is the load-bearing invariant that lets
    /// text-only paged callers migrate to the per-block API without
    /// invalidating their existing cache entries.
    #[test]
    fn test_per_block_empty_matches_uniform_empty() {
        let tokens: Vec<u32> = (0..8).collect();
        let block_size = 4u32;
        let num_full = tokens.len() / block_size as usize;
        let empty_per_block: Vec<Vec<u64>> = (0..num_full).map(|_| Vec::new()).collect();

        // Cache via the uniform API.
        let mut a_uniform = BlockAllocator::new(8, block_size);
        let blocks_uniform: Vec<Arc<PhysicalBlock>> = (0..num_full)
            .map(|_| a_uniform.allocate().unwrap())
            .collect();
        a_uniform
            .cache_full_blocks(&tokens, &blocks_uniform, block_size, &[], 0)
            .expect("uniform cache_full_blocks");

        // Cache via the per-block API (in a fresh allocator to keep hashes
        // independent — but they must STILL collide if both APIs hash the
        // same way, which would manifest as a Case 2 collision drop. We
        // assert hash equivalence directly below to avoid that subtlety).
        let mut a_per_block = BlockAllocator::new(8, block_size);
        let blocks_pb: Vec<Arc<PhysicalBlock>> = (0..num_full)
            .map(|_| a_per_block.allocate().unwrap())
            .collect();
        a_per_block
            .cache_full_blocks_per_block(&tokens, &blocks_pb, block_size, &empty_per_block, 0)
            .expect("per-block cache_full_blocks");

        // Lookup with the uniform API in a_per_block: must hit if the
        // hashes match between the two APIs.
        let (hits_uniform_view, n_uniform_view) =
            a_per_block.find_longest_cache_hit(&tokens, block_size, &[], 0);
        assert_eq!(hits_uniform_view.len(), num_full);
        assert_eq!(n_uniform_view, tokens.len());

        // Lookup with the per-block API in a_uniform: must also hit.
        let (hits_pb_view, n_pb_view) =
            a_uniform.find_longest_cache_hit_per_block(&tokens, block_size, &empty_per_block, 0);
        assert_eq!(hits_pb_view.len(), num_full);
        assert_eq!(n_pb_view, tokens.len());
    }

    /// Two requests with identical text but DIFFERENT per-block image
    /// hashes must produce DIFFERENT cache identities: a stale image's KV
    /// state must NOT be reused for a request with a different image at
    /// the same positions.
    #[test]
    fn test_per_block_image_hash_isolation() {
        let tokens: Vec<u32> = (0..8).collect();
        let block_size = 4u32;
        let num_full = tokens.len() / block_size as usize;

        // Image A puts its hash on block 0; image B does the same but with
        // a different hash. Both blocks 1+ are text-only (empty).
        let per_block_a: Vec<Vec<u64>> = vec![vec![0xAAAA, 0], Vec::new()];
        let per_block_b: Vec<Vec<u64>> = vec![vec![0xBBBB, 0], Vec::new()];

        let mut allocator = BlockAllocator::new(8, block_size);

        // Request A: cache blocks under image-A's per-block keys.
        let blocks_a: Vec<Arc<PhysicalBlock>> = (0..num_full)
            .map(|_| allocator.allocate().unwrap())
            .collect();
        allocator
            .cache_full_blocks_per_block(&tokens, &blocks_a, block_size, &per_block_a, 0)
            .expect("cache_full_blocks_per_block A");

        // Request B (same tokens, different image): lookup with image-B's
        // per-block keys MUST miss block 0 (because the image hash on that
        // block differs). The chain breaks at block 0, so blocks 1+ are
        // also misses (unreachable).
        let (hits_b, n_b) =
            allocator.find_longest_cache_hit_per_block(&tokens, block_size, &per_block_b, 0);
        assert_eq!(
            hits_b.len(),
            0,
            "different image hash on block 0 must produce a cache miss; \
             otherwise stale image KV state would be reused for a \
             different image",
        );
        assert_eq!(n_b, 0);

        // Sanity: the same image's per-block keys still hit.
        let (hits_a, n_a) =
            allocator.find_longest_cache_hit_per_block(&tokens, block_size, &per_block_a, 0);
        assert_eq!(hits_a.len(), num_full);
        assert_eq!(n_a, tokens.len());
    }

    /// Per-block extra_keys mismatch only on a non-leading block isolates
    /// at the divergence point. Blocks before the mismatch hit; blocks at
    /// and after do not.
    #[test]
    fn test_per_block_image_hash_partial_isolation() {
        let tokens: Vec<u32> = (0..16).collect();
        let block_size = 4u32;
        let num_full = tokens.len() / block_size as usize;

        // Image A puts its hash on block 2 only.
        let per_block_a: Vec<Vec<u64>> = vec![Vec::new(), Vec::new(), vec![0xAAAA, 0], Vec::new()];
        // Image B differs ONLY on block 2.
        let per_block_b: Vec<Vec<u64>> = vec![Vec::new(), Vec::new(), vec![0xBBBB, 0], Vec::new()];

        let mut allocator = BlockAllocator::new(8, block_size);
        let blocks_a: Vec<Arc<PhysicalBlock>> = (0..num_full)
            .map(|_| allocator.allocate().unwrap())
            .collect();
        allocator
            .cache_full_blocks_per_block(&tokens, &blocks_a, block_size, &per_block_a, 0)
            .expect("cache_full_blocks_per_block A");

        // Request B: blocks 0 and 1 hit (text-only, identical hashes); the
        // chain breaks at block 2.
        let (hits_b, n_b) =
            allocator.find_longest_cache_hit_per_block(&tokens, block_size, &per_block_b, 0);
        assert_eq!(
            hits_b.len(),
            2,
            "blocks 0+1 share text-only hashes; block 2 differs by image hash",
        );
        assert_eq!(n_b, 2 * block_size as usize);
    }

    /// `cache_full_blocks_per_block` must reject mismatched lengths
    /// (per-block vec shorter than the block list). This catches model-
    /// integration bugs where the caller forgot to size the per-block
    /// vec to the block table's length.
    #[test]
    fn test_per_block_rejects_length_mismatch() {
        let tokens: Vec<u32> = (0..8).collect();
        let block_size = 4u32;
        let mut allocator = BlockAllocator::new(8, block_size);
        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();

        // Only one entry in per_block, but two blocks → mismatch.
        let too_short: Vec<Vec<u64>> = vec![Vec::new()];
        let res = allocator.cache_full_blocks_per_block(
            &tokens,
            &[Arc::clone(&b0), Arc::clone(&b1)],
            block_size,
            &too_short,
            0,
        );
        assert!(res.is_err());
    }

    /// `find_longest_cache_hit_per_block` with too-short keys treats the
    /// missing entry as a chain-break (no panic). Defensive test.
    #[test]
    fn test_per_block_lookup_short_keys_breaks_chain() {
        let tokens: Vec<u32> = (0..16).collect();
        let block_size = 4u32;
        let num_full = tokens.len() / block_size as usize;

        // Cache 4 blocks under all-empty per-block keys.
        let per_block_full: Vec<Vec<u64>> = (0..num_full).map(|_| Vec::new()).collect();
        let mut allocator = BlockAllocator::new(8, block_size);
        let blocks: Vec<Arc<PhysicalBlock>> = (0..num_full)
            .map(|_| allocator.allocate().unwrap())
            .collect();
        allocator
            .cache_full_blocks_per_block(&tokens, &blocks, block_size, &per_block_full, 0)
            .unwrap();

        // Lookup with only 2 entries → chain breaks at block 2.
        let too_short: Vec<Vec<u64>> = vec![Vec::new(), Vec::new()];
        let (hits, n) =
            allocator.find_longest_cache_hit_per_block(&tokens, block_size, &too_short, 0);
        assert_eq!(hits.len(), 2);
        assert_eq!(n, 2 * block_size as usize);
    }

    /// Turn 1 of a 31k-token prompt registers ~1942 blocks at
    /// block_size=16. With a fixed cap of 1024 prefix-cache entries, the
    /// first ~921 blocks would evict before finalize returned, so turn 2's
    /// lookup walking from block 0 would miss instantly and report
    /// cached_prefix_len=0. The cap scales to num_blocks to prevent this.
    #[test]
    fn long_chain_survives_registration_without_head_eviction() {
        let mut a = BlockAllocator::new(2000, 16);
        let token_ids: Vec<u32> = (0..1100 * 16).map(|i| i as u32).collect();
        let blocks: Vec<_> = (0..1100).map(|_| a.allocate().unwrap()).collect();
        let n = a
            .cache_full_blocks(&token_ids, &blocks, 16, &[], 0)
            .unwrap();
        assert_eq!(n, 1100);
        let (hits, cached) = a.find_longest_cache_hit(&token_ids, 16, &[], 0);
        assert_eq!(hits.len(), 1100, "head blocks must survive registration");
        assert_eq!(cached, 1100 * 16);
    }
}
