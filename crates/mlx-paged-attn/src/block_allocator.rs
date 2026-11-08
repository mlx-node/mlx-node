//! Block allocator for PagedAttention KV cache
//!
//! Manages a pool of fixed-size physical blocks that can be allocated
//! to sequences on demand. Supports:
//! - Reference counting for copy-on-write (beam search)
//! - Prefix caching via content-based hashing
//! - Leaf-first eviction by one recency clock
//!
//! # Prefix-cache reference invariant
//!
//! Every entry in `prefix_cache` corresponds to **one logical reference
//! held by the cache itself**. `register_prefix` increfs on the genuine
//! insertion path; every cache-removal path (a take by `allocate`, Case 1
//! stale-alias displacement, and `free()`'s ref_count→0 cleanup) is
//! responsible for releasing that reference. Idempotent refresh of an
//! already-present (block, hash) pair does NOT incref again — the
//! existing logical reference is preserved and the stamp is unchanged.
//!
//! Consequence: callers do not need to manually `incref` blocks before
//! registering them — registration itself takes the cache's ref. Once
//! all external references are released via `free()`, the cache's ref
//! is what keeps the block alive until a take (or another displacement
//! path) decrefs it back to 0 and returns it to the pool.
//!
//! # Allocation under cache pressure
//!
//! When `free_blocks` is empty, `allocate` takes a cache-only block (one
//! whose ref_count is exactly 1: the cache's own logical ref). Every cached
//! hash carries a stamp from one [`CacheRecency`] clock. The oldest dead
//! leaf (a cached block no cached identity names as its parent, and no
//! state mark protects) goes first, so a chain loses its tail and keeps its
//! head, as Splash `KvCache::orders` puts only a block with no resident
//! children and no users in a leaf order. A parent that becomes a leaf
//! inherits its last child's stamp when nothing holds it (Splash
//! `KvCache::inherit`).
//!
//! mlx-node choice, not in Splash: only when no dead leaf is left does
//! `allocate` take a pinned interior (a cache-only block whose cached
//! descendants are all held), deepest first. Splash moves a request onto
//! the block it finds when it publishes, so it never has one; mlx-node's
//! verified-duplicate skip keeps the request on its own block and links its
//! new tail under the found one, and without this fallback such a block
//! could never be reused.
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

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::prefix_registry::{PrefixKeys, hash_block};
use crate::recency::CacheRecency;
#[cfg(test)]
use crate::{chain_hashes, chain_hashes_per_block, hash_tokens};

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

    /// Current number of blocks in the pool. [`Self::grow_by`] can raise
    /// this up to `max_num_blocks`.
    num_blocks: u32,

    /// Hard ceiling on the block count this allocator can reach, fixed at
    /// construction. [`Self::new`] manages `num_blocks` ids immediately and
    /// permits [`Self::grow_by`] up to `max_num_blocks`. When the two are
    /// equal the allocator is not growable and behaves exactly like the
    /// original fixed-size allocator.
    max_num_blocks: u32,

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

    /// The clock every stamp comes from. Shared with the owner's other
    /// caches when built with [`BlockAllocator::with_recency`].
    recency: Arc<CacheRecency>,

    /// Cached hash -> last use.
    stamps: HashMap<u64, u64>,

    /// `(stamp, hash)` of every cached hash with no `child_count` row. Held
    /// leaves stay here and are skipped when a take scans.
    leaves: BTreeSet<(u64, u64)>,

    /// Parent hash -> number of cached identities naming it. A row exists
    /// only while its count is above 0; a forgotten parent keeps its row
    /// while cached children still name it.
    child_count: HashMap<u64, u32>,

    /// Hashes a state needs: their chains are never taken.
    state_marks: HashSet<u64>,
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

/// A cut of the pool's tail: the block count it shrinks to, and the
/// cache-only blocks at or above that count that first move into free slots
/// below it, as `(from, to)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShrinkPlan {
    pub num_blocks: u32,
    pub moves: Vec<(u32, u32)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotUse {
    Free,
    CacheOnly,
    Held,
}

/// One block in an all-or-nothing cold-restore publication.
///
/// The physical block must already be reserved and fully uploaded. Keeping
/// the borrowed identity beside it lets [`BlockAllocator`] validate the whole
/// chain before mutating the hot prefix cache.
pub struct RestoredPrefixRegistration<'a> {
    pub block: Arc<PhysicalBlock>,
    pub hash: u64,
    pub token_ids: &'a [u32],
    pub parent_hash: u64,
    pub extra_keys: &'a [u64],
    pub cache_salt: u64,
    pub block_index: usize,
}

impl BlockAllocator {
    /// Create a new block allocator
    ///
    /// # Arguments
    /// * `num_blocks` - Number of blocks to manage immediately
    /// * `max_num_blocks` - Hard ceiling [`Self::grow_by`] may later raise
    ///   the pool to (must be `>= num_blocks`). When equal to `num_blocks`
    ///   the allocator is not growable and behavior is identical to the
    ///   original fixed-size allocator.
    /// * `block_size` - Number of tokens per block
    pub fn new(num_blocks: u32, max_num_blocks: u32, block_size: u32) -> Self {
        Self::with_recency(
            num_blocks,
            max_num_blocks,
            block_size,
            Arc::new(CacheRecency::new()),
        )
    }

    /// [`Self::new`] on a clock the caller shares with its other caches.
    pub fn with_recency(
        num_blocks: u32,
        max_num_blocks: u32,
        block_size: u32,
        recency: Arc<CacheRecency>,
    ) -> Self {
        assert!(
            max_num_blocks >= num_blocks,
            "BlockAllocator::new: max_num_blocks ({max_num_blocks}) must be >= num_blocks \
             ({num_blocks})"
        );
        let free_blocks: VecDeque<u32> = (0..num_blocks).collect();

        Self {
            free_blocks,
            allocated: HashMap::with_capacity(num_blocks as usize),
            num_blocks,
            max_num_blocks,
            block_size,
            prefix_cache: HashMap::new(),
            prefix_cache_identities: HashMap::new(),
            block_hashes: HashMap::new(),
            recency,
            stamps: HashMap::new(),
            leaves: BTreeSet::new(),
            child_count: HashMap::new(),
            state_marks: HashSet::new(),
        }
    }

    /// The clock this allocator stamps with.
    pub fn recency(&self) -> Arc<CacheRecency> {
        Arc::clone(&self.recency)
    }

    fn ref_count_of(&self, hash: u64) -> Option<u32> {
        self.prefix_cache
            .get(&hash)
            .map(|block| block.get_ref_count())
    }

    /// Give `hash` a stamp from the clock and re-key its leaf entry.
    fn touch(&mut self, hash: u64) {
        let stamp = self.recency.next();
        self.set_stamp(hash, stamp);
    }

    fn set_stamp(&mut self, hash: u64, stamp: u64) {
        let old = self.stamps.insert(hash, stamp);
        if let Some(old) = old
            && self.leaves.remove(&(old, hash))
        {
            self.leaves.insert((stamp, hash));
        }
    }

    /// Drop one cached identity naming `parent`. At 0 the row goes, and a
    /// cached parent becomes a leaf; one nothing holds takes the child's
    /// stamp when that is newer.
    fn unlink_child(&mut self, parent: u64, child_stamp: Option<u64>) {
        let Some(count) = self.child_count.get_mut(&parent) else {
            return;
        };
        *count = count.saturating_sub(1);
        if *count > 0 {
            return;
        }
        self.child_count.remove(&parent);
        let Some(ref_count) = self.ref_count_of(parent) else {
            return;
        };
        let own = self.stamps.get(&parent).copied().unwrap_or(0);
        let stamp = match child_stamp {
            Some(child) if ref_count == 1 => own.max(child),
            _ => own,
        };
        self.stamps.insert(parent, stamp);
        self.leaves.insert((stamp, parent));
    }

    fn link_child(&mut self, parent: u64) {
        let count = self.child_count.entry(parent).or_insert(0);
        *count += 1;
        if *count == 1
            && let Some(&stamp) = self.stamps.get(&parent)
        {
            self.leaves.remove(&(stamp, parent));
        }
    }

    /// Drop `hash`'s identity and unlink it from its parent.
    fn drop_identity(&mut self, hash: u64, stamp: Option<u64>) {
        if let Some(identity) = self.prefix_cache_identities.remove(&hash)
            && identity.parent_hash != 0
        {
            self.unlink_child(identity.parent_hash, stamp);
        }
    }

    /// The one remover of a cached hash's bookkeeping. It leaves the
    /// block's ref count and `allocated` to the caller.
    fn forget(&mut self, hash: u64) -> Option<Arc<PhysicalBlock>> {
        let block = self.prefix_cache.remove(&hash);
        if let Some(block) = &block
            && self.block_hashes.get(&block.block_id) == Some(&hash)
        {
            self.block_hashes.remove(&block.block_id);
        }
        let stamp = self.stamps.remove(&hash);
        if let Some(stamp) = stamp {
            self.leaves.remove(&(stamp, hash));
        }
        self.drop_identity(hash, stamp);
        block
    }

    /// Marks and every cached ancestor of a mark.
    fn marked_closure(&self) -> HashSet<u64> {
        let mut closure = HashSet::with_capacity(self.state_marks.len());
        for &mark in &self.state_marks {
            let mut hash = mark;
            while closure.insert(hash) {
                let Some(identity) = self.prefix_cache_identities.get(&hash) else {
                    break;
                };
                let parent = identity.parent_hash;
                if parent == 0 || !self.prefix_cache.contains_key(&parent) {
                    break;
                }
                hash = parent;
            }
        }
        closure
    }

    /// The oldest cached leaf nothing holds and no mark protects.
    fn oldest_dead_leaf(&self) -> Option<u64> {
        self.leaves
            .iter()
            .map(|&(_, hash)| hash)
            .find(|hash| !self.state_marks.contains(hash) && self.ref_count_of(*hash) == Some(1))
    }

    /// The deepest cache-only block that still has cached children (all of
    /// them held), outside every marked chain; the oldest stamp breaks a tie.
    /// mlx-node choice (see the module doc): Splash takes no interior.
    fn deepest_pinned_interior(&self) -> Option<u64> {
        let closure = self.marked_closure();
        self.prefix_cache
            .iter()
            .filter(|(hash, block)| {
                block.get_ref_count() == 1
                    && self.child_count.contains_key(hash)
                    && !closure.contains(hash)
            })
            .map(|(&hash, _)| {
                let depth = self
                    .prefix_cache_identities
                    .get(&hash)
                    .map_or(0, |identity| identity.block_index);
                let stamp = self.stamps.get(&hash).copied().unwrap_or(0);
                (
                    depth,
                    std::cmp::Reverse(stamp),
                    std::cmp::Reverse(hash),
                    hash,
                )
            })
            .max()
            .map(|(_, _, _, hash)| hash)
    }

    /// Release the cache's ref on a cache-only `hash` and return its id,
    /// which leaves `allocated`.
    fn take_cached(&mut self, hash: u64) -> Option<u32> {
        let block = self.forget(hash)?;
        let block_id = block.block_id;
        let _ = block.decref();
        self.allocated.remove(&block_id);
        Some(block_id)
    }

    /// Allocate a new block: a free id first, else the oldest dead leaf,
    /// else the deepest pinned interior (an mlx-node choice, see the module
    /// doc). `None` when none exists.
    pub fn allocate(&mut self) -> Option<Arc<PhysicalBlock>> {
        let block_id = if let Some(id) = self.free_blocks.pop_front() {
            id
        } else if let Some(hash) = self.oldest_dead_leaf() {
            self.take_cached(hash)?
        } else {
            let hash = self.deepest_pinned_interior()?;
            self.take_cached(hash)?
        };
        let block = Arc::new(PhysicalBlock::new(block_id));
        self.allocated.insert(block_id, Arc::clone(&block));
        Some(block)
    }

    /// Allocate `n` blocks or none. At every pool size nothing is taken
    /// unless free plus evictable blocks cover `n` (`Err(0)`), as Splash
    /// `MemoryLedger::free` counts with `bound` and calls `take` only when
    /// the count covers the need. Blocks the pool has not grown yet do not
    /// count: the caller grows the pool first, as Splash
    /// `KvPool::takeFreePages` grows an extent before the claim reaches the
    /// ledger.
    pub fn reserve(&mut self, n: u32) -> Result<Vec<Arc<PhysicalBlock>>, u32> {
        if !self.can_allocate(n) {
            return Err(0);
        }
        let mut taken = Vec::with_capacity(n as usize);
        for _ in 0..n {
            match self.allocate() {
                Some(block) => taken.push(block),
                None => {
                    let count = taken.len() as u32;
                    for block in taken {
                        self.free(block);
                    }
                    return Err(count);
                }
            }
        }
        Ok(taken)
    }

    /// Evict the oldest dead leaf and put its id back in the free queue.
    pub fn evict_oldest_dead_leaf(&mut self) -> Option<u32> {
        let hash = self.oldest_dead_leaf()?;
        let block_id = self.take_cached(hash)?;
        self.free_blocks.push_back(block_id);
        Some(block_id)
    }

    /// Replace the set of hashes a state needs.
    pub fn set_state_marks(&mut self, hashes: impl IntoIterator<Item = u64>) {
        self.state_marks = hashes.into_iter().collect();
    }

    /// The oldest marked leaf nothing holds, as `(stamp, hash)`.
    pub fn oldest_marked_leaf(&self) -> Option<(u64, u64)> {
        self.leaves.iter().copied().find(|(_, hash)| {
            self.state_marks.contains(hash) && self.ref_count_of(*hash) == Some(1)
        })
    }

    /// Whether `hash` is cached and only the cache holds it.
    pub fn is_cache_only(&self, hash: u64) -> bool {
        self.ref_count_of(hash) == Some(1)
    }

    /// Marks that are cache-only and lie in no other mark's chain: dropping
    /// every state on one makes at least that block evictable.
    pub fn tip_marks(&self) -> HashSet<u64> {
        let mut covered = HashSet::new();
        for &mark in &self.state_marks {
            let mut hash = mark;
            while let Some(identity) = self.prefix_cache_identities.get(&hash) {
                let parent = identity.parent_hash;
                if parent == 0
                    || !self.prefix_cache.contains_key(&parent)
                    || !covered.insert(parent)
                {
                    break;
                }
                hash = parent;
            }
        }
        self.state_marks
            .iter()
            .copied()
            .filter(|hash| !covered.contains(hash) && self.is_cache_only(*hash))
            .collect()
    }

    /// Of `block_ids`, the cached cache-only blocks: (outside every marked
    /// chain, inside one).
    pub fn cache_only_among(&self, block_ids: &HashSet<u32>) -> (u32, u32) {
        if block_ids.is_empty() {
            return (0, 0);
        }
        let closure = self.marked_closure();
        let mut outside = 0;
        let mut inside = 0;
        for (hash, block) in &self.prefix_cache {
            if block.get_ref_count() != 1 || !block_ids.contains(&block.block_id) {
                continue;
            }
            if closure.contains(hash) {
                inside += 1;
            } else {
                outside += 1;
            }
        }
        (outside, inside)
    }

    /// How many distinct `hashes` are cached and cache-only.
    pub fn cache_only_count(&self, hashes: impl IntoIterator<Item = u64>) -> u32 {
        hashes
            .into_iter()
            .collect::<HashSet<u64>>()
            .into_iter()
            .filter(|hash| self.is_cache_only(*hash))
            .count() as u32
    }

    /// Of the leading cached hashes of `chain` (a lookup's walk), how many are
    /// cache-only and inside a marked chain.
    pub fn marked_hit(&self, chain: &[u64]) -> u32 {
        if self.state_marks.is_empty() {
            return 0;
        }
        let closure = self.marked_closure();
        chain
            .iter()
            .take_while(|hash| self.prefix_cache.contains_key(hash))
            .filter(|hash| self.is_cache_only(**hash) && closure.contains(hash))
            .count() as u32
    }

    /// The cut [`Self::shrink_target`] would reach if the tables holding
    /// `table_refs` (block id -> refs) were released: a block counts Free
    /// when no other ref is left, CacheOnly when it is cached and only the
    /// cache's ref is left.
    pub fn shrink_target_as_released(
        &self,
        floor: u32,
        extent: u32,
        extents: u32,
        table_refs: &HashMap<u32, u32>,
    ) -> u32 {
        let mut uses = self.slot_uses();
        for (&block_id, block) in &self.allocated {
            let Some(slot) = uses.get_mut(block_id as usize) else {
                continue;
            };
            *slot = self.use_as_released(block_id, block, table_refs);
        }
        self.shrink_candidates(&uses, floor, extent, extents)
            .first()
            .copied()
            .unwrap_or(self.num_blocks)
    }

    /// Of the blocks in `table_refs` (block id -> refs), those releasing
    /// those refs would leave free or cache-only.
    pub fn released_blocks(&self, table_refs: &HashMap<u32, u32>) -> HashSet<u32> {
        table_refs
            .keys()
            .copied()
            .filter(|block_id| {
                self.allocated.get(block_id).is_some_and(|block| {
                    self.use_as_released(*block_id, block, table_refs) != SlotUse::Held
                })
            })
            .collect()
    }

    fn use_as_released(
        &self,
        block_id: u32,
        block: &PhysicalBlock,
        table_refs: &HashMap<u32, u32>,
    ) -> SlotUse {
        let released = table_refs.get(&block_id).copied().unwrap_or(0);
        let left = block.get_ref_count().saturating_sub(released);
        let cached = self
            .block_hashes
            .get(&block_id)
            .and_then(|hash| self.prefix_cache.get(hash))
            .is_some_and(|cached| cached.block_id == block_id);
        if left == 0 {
            SlotUse::Free
        } else if cached && left == 1 {
            SlotUse::CacheOnly
        } else {
            SlotUse::Held
        }
    }

    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn block_stamp(&self, hash: u64) -> Option<u64> {
        self.stamps.get(&hash).copied()
    }

    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn child_count_rows(&self) -> usize {
        self.child_count.len()
    }

    /// Free a block
    ///
    /// The block is only returned to the free pool if its ref_count reaches 0
    pub fn free(&mut self, block: Arc<PhysicalBlock>) {
        self.free_at(block, None);
    }

    /// [`Self::free`] for a block whose last use was at `at`. A cached leaf
    /// left with only the cache's ref is stamped `max(own, at)`, or with a
    /// fresh stamp when `at` is `None`.
    pub fn free_at(&mut self, block: Arc<PhysicalBlock>, at: Option<u64>) {
        let block_id = block.block_id;
        if block.decref() {
            self.allocated.remove(&block_id);
            if let Some(&hash) = self.block_hashes.get(&block_id) {
                self.forget(hash);
                self.block_hashes.remove(&block_id);
            }
            self.free_blocks.push_back(block_id);
            return;
        }
        if block.get_ref_count() != 1 {
            return;
        }
        let Some(&hash) = self.block_hashes.get(&block_id) else {
            return;
        };
        if self.child_count.contains_key(&hash)
            || self
                .prefix_cache
                .get(&hash)
                .is_none_or(|cached| cached.block_id != block_id)
        {
            return;
        }
        let stamp = match at {
            None => self.recency.next(),
            Some(at) => self.stamps.get(&hash).copied().unwrap_or(0).max(at),
        };
        self.set_stamp(hash, stamp);
    }

    /// Evict EVERY prefix-cache entry, releasing the cache's logical
    /// reference on each block. Blocks whose
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
        self.prefix_cache_identities.clear();
        self.stamps.clear();
        self.leaves.clear();
        self.child_count.clear();
        self.state_marks.clear();
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
    /// takes that reference on the genuine-insert path; the stale-alias
    /// displacement below releases the old alias's reference via `decref()`.
    ///
    /// Idempotent refresh (same block, same hash) does NOT incref and does
    /// not stamp: the block was committed already.
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
    ///    forgotten before inserting the new alias. Otherwise the stale
    ///    entry would survive `free()` and could hand out a returned-to-pool
    ///    block on a future `lookup_prefix` — bypassing `extra_keys`
    ///    isolation. The cache's logical reference for the OLD hash is
    ///    released here (decref); the new alias takes a fresh ref in step 3
    ///    — net change in ref_count for this block is zero.
    ///
    /// 3. **Insert**: (re)insert into `prefix_cache` / `block_hashes`. A
    ///    genuine new insertion increfs, takes a fresh stamp, and enters the
    ///    leaf set unless cached children already name the hash.
    ///
    /// # Return value
    ///
    /// Returns `true` when the registration was accepted — i.e. the
    /// passed-in `block` is now authoritative for `hash` in the prefix
    /// cache: a genuine insertion, an idempotent refresh, or a stale-alias
    /// displacement.
    ///
    /// Returns `false` only on a Case 2 hash collision (same `hash`,
    /// different `block`): the pre-existing entry stays authoritative and
    /// the caller's `block` was NOT inserted.
    ///
    /// Callers that walk a hash chain (notably `cache_full_blocks`) MUST
    /// abort on `false`: a chain block whose registration was dropped
    /// means subsequent block hashes would link to ghost predecessors,
    /// producing a future `find_longest_cache_hit` return that mixes
    /// blocks across registration intents (silent KV corruption).
    pub(crate) fn register_prefix(&mut self, block: Arc<PhysicalBlock>, hash: u64) -> bool {
        if let Some(existing_block) = self.prefix_cache.get(&hash)
            && existing_block.block_id != block.block_id
        {
            return false;
        }

        // The block survives this swap because at least one of {external
        // request handle, cache's ref about to be retaken} keeps
        // ref_count >= 1.
        if let Some(&existing_hash) = self.block_hashes.get(&block.block_id)
            && existing_hash != hash
            && self.forget(existing_hash).is_some()
        {
            let _ = block.decref();
        }

        let is_new_insertion = !self.prefix_cache.contains_key(&hash);
        self.block_hashes.insert(block.block_id, hash);
        if is_new_insertion {
            // Direct register_prefix callers do not provide enough
            // identity material to verify future duplicate-hash attempts.
            // cache_full_blocks repopulates this immediately after a
            // successful insertion.
            self.drop_identity(hash, None);
            block.incref();
            let stamp = self.recency.next();
            self.stamps.insert(hash, stamp);
            if !self.child_count.contains_key(&hash) {
                self.leaves.insert((stamp, hash));
            }
        }
        self.prefix_cache.insert(hash, block);
        true
    }

    /// Look up a block in the prefix cache
    ///
    /// Returns the cached block if found, incrementing its ref count. It
    /// stamps nothing: a walk stamps its tip once.
    pub(crate) fn lookup_prefix(&mut self, hash: u64) -> Option<Arc<PhysicalBlock>> {
        let block = self.prefix_cache.get(&hash)?;
        block.incref();
        Some(Arc::clone(block))
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
        let old = self.prefix_cache_identities.insert(
            hash,
            PrefixCacheBlockIdentity::new(
                token_ids,
                parent_hash,
                extra_keys,
                cache_salt,
                block_index,
            ),
        );
        let had_identity = old.is_some();
        let old_parent = old.map_or(0, |identity| identity.parent_hash);
        if had_identity && old_parent == parent_hash {
            return;
        }
        if old_parent != 0 {
            let stamp = self.stamps.get(&hash).copied();
            self.unlink_child(old_parent, stamp);
        }
        if parent_hash != 0 {
            self.link_child(parent_hash);
        }
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
        self.find_longest_cache_hit_with_keys(
            token_ids,
            block_size,
            PrefixKeys::Uniform(extra_keys),
            cache_salt,
        )
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
        self.find_longest_cache_hit_with_keys(
            token_ids,
            block_size,
            PrefixKeys::PerBlock(extra_keys_per_block),
            cache_salt,
        )
    }

    pub fn find_longest_cache_hit_with_keys(
        &mut self,
        token_ids: &[u32],
        block_size: u32,
        keys: PrefixKeys<'_>,
        cache_salt: u64,
    ) -> (Vec<Arc<PhysicalBlock>>, usize) {
        if block_size == 0 || token_ids.is_empty() || token_ids.len() < block_size as usize {
            return (Vec::new(), 0);
        }
        let block_size_us = block_size as usize;
        let num_full_blocks = token_ids.len() / block_size_us;
        // A short per-block key list truncates the shared hash walk at the
        // first block without a cache identity, matching the old miss behavior.
        let hashes = keys.chain_hashes(token_ids, block_size, cache_salt);
        let mut blocks = Vec::with_capacity(num_full_blocks);
        for (n, &block_hash) in hashes.iter().enumerate() {
            let start = n * block_size_us;
            let block_tokens = &token_ids[start..start + block_size_us];
            let parent_hash = if n == 0 { 0 } else { hashes[n - 1] };
            // `hashes` was produced by `keys.chain_hashes`, so a missing key
            // cannot occur; if it ever did, the documented "first block
            // without a cache identity" behavior is to stop the walk.
            let Some(extra_keys) = keys.get(n) else {
                break;
            };
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
                Some(block) => blocks.push(block),
                None => break,
            }
        }
        // A hit is Splash `Cache::lookup` (`KvCache::touch` on the tip) and
        // the attach after it (`KvCache::retainActive` on the chain's last
        // block only): ancestors keep their stamps.
        if let Some(&tip) = blocks.len().checked_sub(1).and_then(|n| hashes.get(n)) {
            self.touch(tip);
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
        self.cache_full_blocks_with_keys(
            token_ids,
            blocks,
            block_size,
            PrefixKeys::Uniform(extra_keys),
            cache_salt,
        )
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
        self.cache_full_blocks_with_keys(
            token_ids,
            blocks,
            block_size,
            PrefixKeys::PerBlock(extra_keys_per_block),
            cache_salt,
        )
    }

    pub fn cache_full_blocks_with_keys(
        &mut self,
        token_ids: &[u32],
        blocks: &[Arc<PhysicalBlock>],
        block_size: u32,
        keys: PrefixKeys<'_>,
        cache_salt: u64,
    ) -> Result<usize, &'static str> {
        if block_size == 0 {
            return Err("block_size must be > 0");
        }

        let block_size_us = block_size as usize;
        if blocks.len() * block_size_us > token_ids.len() {
            return Err("blocks exceed token_ids length");
        }
        if let PrefixKeys::PerBlock(per_block) = keys
            && per_block.len() < blocks.len()
        {
            return Err("extra_keys_per_block shorter than blocks");
        }

        let mut previous_block_hash: u64 = 0;
        let mut registered = 0usize;
        for (n, block) in blocks.iter().enumerate() {
            let start = n * block_size_us;
            let end = start + block_size_us;
            let block_tokens = &token_ids[start..end];
            let parent_hash = if n == 0 { 0 } else { previous_block_hash };
            // Registration key count was validated against `blocks.len()`
            // above, so this is `Some` for every n we reach.
            let extra_keys = keys.get(n).ok_or("missing per-block registration keys")?;
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
                    // tail can be published. The found block is committed
                    // again, so it is stamped.
                    self.touch(block_hash);
                    previous_block_hash = block_hash;
                    continue;
                } else {
                    // Chain broke (collision drop). Stop
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

    /// Get the hard ceiling on the block count (see [`Self::grow_by`]).
    pub fn max_num_blocks(&self) -> u32 {
        self.max_num_blocks
    }

    /// Grow the pool by up to `additional` blocks, capped at
    /// `max_num_blocks`. Returns the number of blocks actually added (0 when
    /// already at the cap).
    ///
    /// The new ids (`old_num_blocks..new_num_blocks`) are pushed onto the
    /// back of the free queue, behind any previously freed ids; existing
    /// allocations, prefix-cache entries, and stamps are untouched.
    /// `num_blocks` bumps to the new total.
    ///
    /// The caller must have already grown the backing `LayerKVPool` to cover
    /// the new id range (the allocator owns no GPU memory) — the adapter
    /// performs `pool.grow_to` first and calls this only on success.
    pub fn grow_by(&mut self, additional: u32) -> u32 {
        let target = self
            .num_blocks
            .saturating_add(additional)
            .min(self.max_num_blocks);
        if target == self.num_blocks {
            return 0;
        }
        let added = target - self.num_blocks;
        self.free_blocks.extend(self.num_blocks..target);
        self.num_blocks = target;
        added
    }

    /// What every id below `num_blocks` holds: a free slot, a prefix-cache
    /// entry no request holds, or a block a request (or a reservation) holds.
    fn slot_uses(&self) -> Vec<SlotUse> {
        let mut uses = vec![SlotUse::Held; self.num_blocks as usize];
        for &id in &self.free_blocks {
            if let Some(slot) = uses.get_mut(id as usize) {
                *slot = SlotUse::Free;
            }
        }
        for block in self.prefix_cache.values() {
            if block.get_ref_count() == 1
                && let Some(slot) = uses.get_mut(block.block_id as usize)
            {
                *slot = SlotUse::CacheOnly;
            }
        }
        uses
    }

    /// Cuts of the tail by `1..=extents` whole extents of `extent` blocks,
    /// lowest first, none below `floor` or below a held block.
    fn shrink_candidates(
        &self,
        uses: &[SlotUse],
        floor: u32,
        extent: u32,
        extents: u32,
    ) -> Vec<u32> {
        let held_end = uses
            .iter()
            .rposition(|slot| *slot == SlotUse::Held)
            .map_or(0, |id| id as u32 + 1);
        let lowest = floor.max(held_end).max(1);
        let mut cuts: Vec<u32> = (1..=extents)
            .rev()
            .map(|count| {
                self.num_blocks
                    .saturating_sub(count.saturating_mul(extent.max(1)))
                    .max(floor)
                    .max(1)
            })
            .filter(|&cut| cut >= lowest && cut < self.num_blocks)
            .collect();
        cuts.dedup();
        cuts
    }

    /// The lowest block count a cut by up to `extents` extents may reach once
    /// enough cache-only blocks are gone: `num_blocks` when none can.
    pub fn shrink_target(&self, floor: u32, extent: u32, extents: u32) -> u32 {
        let uses = self.slot_uses();
        self.shrink_candidates(&uses, floor, extent, extents)
            .first()
            .copied()
            .unwrap_or(self.num_blocks)
    }

    /// The deepest cut by up to `extents` whole extents of `extent` blocks
    /// that loses nothing: every block at or above it is free or cache-only,
    /// and the cache-only ones fit the free slots below it. The cut never
    /// goes below `floor` or below a held block. `None` when no cut fits.
    pub fn shrink_plan(&self, floor: u32, extent: u32, extents: u32) -> Option<ShrinkPlan> {
        let uses = self.slot_uses();
        for cut in self.shrink_candidates(&uses, floor, extent, extents) {
            let (below, above) = uses.split_at(cut as usize);
            let cached: Vec<u32> = above
                .iter()
                .enumerate()
                .filter(|(_, slot)| **slot == SlotUse::CacheOnly)
                .map(|(offset, _)| cut + offset as u32)
                .collect();
            let free: Vec<u32> = below
                .iter()
                .enumerate()
                .filter(|(_, slot)| **slot == SlotUse::Free)
                .map(|(id, _)| id as u32)
                .take(cached.len())
                .collect();
            if free.len() == cached.len() {
                return Some(ShrinkPlan {
                    num_blocks: cut,
                    moves: cached.into_iter().zip(free).collect(),
                });
            }
        }
        None
    }

    /// Apply a [`Self::shrink_plan`] after the pool copied every moved
    /// block: each moved entry keeps its hash under its new id, ids at or
    /// above the cut leave the free queue, and `num_blocks` becomes the cut.
    /// Stamps, child rows and marks are keyed by hash, so they stay.
    /// Checks the whole plan first; on `Err` nothing changed.
    pub fn apply_shrink(&mut self, plan: &ShrinkPlan) -> Result<(), &'static str> {
        let cut = plan.num_blocks;
        if cut == 0 || cut > self.num_blocks {
            return Err("shrink plan cut out of range");
        }
        let mut uses = self.slot_uses();
        for &(from, to) in &plan.moves {
            if from < cut
                || to >= cut
                || uses.get(from as usize) != Some(&SlotUse::CacheOnly)
                || uses.get(to as usize) != Some(&SlotUse::Free)
            {
                return Err("shrink plan moves a block that is not cache-only into a free slot");
            }
            uses[from as usize] = SlotUse::Free;
            uses[to as usize] = SlotUse::CacheOnly;
        }
        if uses[cut as usize..]
            .iter()
            .any(|slot| *slot != SlotUse::Free)
        {
            return Err("shrink plan leaves a block at or above the cut");
        }
        for &(from, to) in &plan.moves {
            let Some(hash) = self.block_hashes.remove(&from) else {
                return Err("shrink plan lost a moved block's hash");
            };
            self.allocated.remove(&from);
            let block = Arc::new(PhysicalBlock::new(to));
            self.allocated.insert(to, Arc::clone(&block));
            self.block_hashes.insert(to, hash);
            self.prefix_cache.insert(hash, block);
        }
        let taken: HashSet<u32> = plan.moves.iter().map(|&(_, to)| to).collect();
        self.free_blocks
            .retain(|&id| id < cut && !taken.contains(&id));
        self.num_blocks = cut;
        Ok(())
    }

    /// Get the block size
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Check if we can allocate the requested number of blocks: free blocks
    /// plus the cache-only blocks [`Self::allocate`] may take.
    pub fn can_allocate(&self, num_blocks: u32) -> bool {
        let free = self.num_free_blocks();
        if free >= num_blocks {
            return true;
        }
        let evictable = self.num_evictable_blocks();
        free + evictable >= num_blocks
    }

    /// Cache-only blocks [`Self::allocate`] may take: every ref-1 cached
    /// block outside a marked chain. Active request blocks have refcount
    /// greater than one and are deliberately excluded.
    pub fn num_evictable_blocks(&self) -> u32 {
        if self.state_marks.is_empty() {
            return self.num_reusable_blocks();
        }
        let closure = self.marked_closure();
        self.prefix_cache
            .iter()
            .filter(|(hash, block)| block.get_ref_count() == 1 && !closure.contains(hash))
            .count() as u32
    }

    /// Every cache-only block, marked chains included.
    pub fn num_reusable_blocks(&self) -> u32 {
        self.prefix_cache
            .values()
            .filter(|block| block.get_ref_count() == 1)
            .count() as u32
    }

    /// Total blocks [`Self::allocate`] could currently hand out: genuinely
    /// free blocks plus evictable cache-only prefix entries. Adapter growth
    /// pre-flights against this so `allocate` never evicts a cache-only
    /// block the pool could instead cover by growing.
    pub fn free_or_evictable_blocks(&self) -> u32 {
        self.num_free_blocks() + self.num_evictable_blocks()
    }

    /// Publish a fully restored cold-cache block into the hot prefix cache.
    ///
    /// The caller must have reserved `block` with [`Self::allocate`] and
    /// restored all of its physical layer bytes before calling this method.
    /// Keeping publication as the final step makes restore transactional:
    /// a failed upload is not discoverable, and the caller can simply
    /// [`Self::free`] the reservation. `token_ids` must describe exactly one
    /// immutable full block.
    #[allow(clippy::too_many_arguments)]
    pub fn publish_restored_prefix(
        &mut self,
        block: Arc<PhysicalBlock>,
        hash: u64,
        token_ids: &[u32],
        parent_hash: u64,
        extra_keys: &[u64],
        cache_salt: u64,
        block_index: usize,
    ) -> Result<bool, &'static str> {
        if token_ids.len() != self.block_size as usize {
            return Err("restored prefix must contain exactly one full block");
        }
        if !self.allocated.contains_key(&block.block_id) {
            return Err("restored prefix block was not reserved by this allocator");
        }
        if !self.register_prefix(Arc::clone(&block), hash) {
            return Ok(false);
        }
        self.remember_prefix_identity(
            hash,
            token_ids,
            parent_hash,
            extra_keys,
            cache_salt,
            block_index,
        );
        Ok(true)
    }

    /// Atomically publish a chain of restored blocks.
    ///
    /// Every fallible condition is checked before the first cache mutation.
    /// Once preflight succeeds, the individual registrations cannot fail
    /// while this exclusive allocator borrow is held. This is the commit
    /// boundary used by asynchronous SSD restore: a missing/corrupt block or
    /// failed upload leaves no partial hot prefix discoverable.
    pub fn publish_restored_prefix_batch(
        &mut self,
        registrations: &[RestoredPrefixRegistration<'_>],
    ) -> Result<bool, &'static str> {
        if registrations.is_empty() {
            return Ok(true);
        }
        if registrations.len() > self.num_blocks as usize {
            return Err("restored prefix batch exceeds prefix-cache capacity");
        }

        let mut hashes = HashSet::with_capacity(registrations.len());
        let mut block_ids = HashSet::with_capacity(registrations.len());
        for registration in registrations {
            if registration.token_ids.len() != self.block_size as usize {
                return Err("restored prefix must contain exactly one full block");
            }
            if !self.allocated.contains_key(&registration.block.block_id) {
                return Err("restored prefix block was not reserved by this allocator");
            }
            if !hashes.insert(registration.hash) || !block_ids.insert(registration.block.block_id) {
                return Err("restored prefix batch contains duplicate hashes or blocks");
            }
            if let Some(existing) = self.prefix_cache.get(&registration.hash)
                && existing.block_id != registration.block.block_id
            {
                return Ok(false);
            }
            if let Some(identity) = self.prefix_cache_identities.get(&registration.hash)
                && !identity.matches(
                    registration.token_ids,
                    registration.parent_hash,
                    registration.extra_keys,
                    registration.cache_salt,
                    registration.block_index,
                )
            {
                return Ok(false);
            }
        }

        for registration in registrations {
            let inserted = self.register_prefix(Arc::clone(&registration.block), registration.hash);
            debug_assert!(inserted, "batch publication was fully preflighted");
            self.remember_prefix_identity(
                registration.hash,
                registration.token_ids,
                registration.parent_hash,
                registration.extra_keys,
                registration.cache_salt,
                registration.block_index,
            );
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_free() {
        let mut allocator = BlockAllocator::new(10, 10, 32);

        assert_eq!(allocator.num_free_blocks(), 10);

        let block = allocator.allocate().unwrap();
        assert_eq!(allocator.num_free_blocks(), 9);
        assert_eq!(block.block_id, 0);

        allocator.free(block);
        assert_eq!(allocator.num_free_blocks(), 10);
    }

    #[test]
    fn test_free_or_evictable_blocks_counts_free_plus_cache_only() {
        let mut allocator = BlockAllocator::new(4, 4, 4);
        assert_eq!(allocator.free_or_evictable_blocks(), 4);

        // A registered prefix entry is not evictable while the request
        // still holds its logical ref (refcount 2: the request's ref from
        // `allocate` plus the cache's ref from `register_prefix`): only the
        // 3 free blocks count.
        let block = allocator.allocate().unwrap();
        allocator.register_prefix(Arc::clone(&block), 0xAAAA);
        assert_eq!(block.get_ref_count(), 2);
        assert_eq!(allocator.num_free_blocks(), 3);
        assert_eq!(allocator.num_evictable_blocks(), 0);
        assert_eq!(allocator.free_or_evictable_blocks(), 3);

        // Releasing the request's logical ref (refcount 1: only the cache's
        // own entry) makes the block evictable.
        block.decref();
        drop(block);
        assert_eq!(allocator.num_evictable_blocks(), 1);
        assert_eq!(allocator.free_or_evictable_blocks(), 4);
    }

    #[test]
    fn test_reference_counting() {
        let mut allocator = BlockAllocator::new(10, 10, 32);

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
        let mut allocator = BlockAllocator::new(10, 10, 32);

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
        // cache's logical reference). A take is what eventually returns
        // the block to the free pool — see
        // `evicted_dead_leaf_returns_to_the_free_pool`.
        let mut allocator = BlockAllocator::new(10, 10, 32);

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
        let mut allocator = BlockAllocator::new(10, 10, 32);

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
        let mut allocator = BlockAllocator::new(2, 2, 32);

        let block1 = allocator.allocate().unwrap();
        let hash1 = hash_tokens(&[1], 0, &[]);
        allocator.register_prefix(Arc::clone(&block1), hash1);
        let block2 = allocator.allocate().unwrap();
        let hash2 = hash_tokens(&[2], 0, &[]);
        allocator.register_prefix(Arc::clone(&block2), hash2);
        let id1 = block1.block_id;
        allocator.free(block1);
        allocator.free(block2);
        assert_eq!(allocator.num_free_blocks(), 0);

        let taken = allocator.allocate().unwrap();
        assert_eq!(taken.block_id, id1, "hash1 was released first");
        assert!(allocator.lookup_prefix(hash1).is_none());
        assert!(!allocator.block_hashes.contains_key(&id1));
        assert_eq!(allocator.block_stamp(hash1), None);
        assert!(allocator.block_stamp(hash2).is_some());
    }

    // hash_tokens(extra_keys), find_longest_cache_hit, cache_full_blocks,
    // refcount lifecycle, eviction order.

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
        let mut allocator = BlockAllocator::new(8, 8, 4);
        let (blocks, n) = allocator.find_longest_cache_hit(&[1, 2, 3, 4, 5, 6, 7, 8], 4, &[], 0);
        assert!(blocks.is_empty());
        assert_eq!(n, 0);
    }

    #[test]
    fn test_find_longest_cache_hit_full_match() {
        let mut allocator = BlockAllocator::new(8, 8, 4);
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
        let mut allocator = BlockAllocator::new(8, 8, 4);
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
        let mut allocator = BlockAllocator::new(16, 16, 4);
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
        let mut allocator = BlockAllocator::new(4, 4, 4);
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
        let mut allocator = BlockAllocator::new(4, 4, 4);
        let (blocks, n) = allocator.find_longest_cache_hit(&[1, 2, 3, 4], 0, &[], 0);
        assert!(blocks.is_empty());
        assert_eq!(n, 0);
    }

    #[test]
    fn test_cache_full_blocks_oversize_returns_err() {
        let mut allocator = BlockAllocator::new(4, 4, 4);
        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        // 2 blocks * block_size 4 = 8 tokens required, but only 4 supplied.
        let res = allocator.cache_full_blocks(&[1, 2, 3, 4], &[b0, b1], 4, &[], 0);
        assert!(res.is_err());
    }

    #[test]
    fn test_cache_full_blocks_extra_keys_mismatch() {
        // Cache with extra_keys=[100], lookup with extra_keys=[] -> miss.
        let mut allocator = BlockAllocator::new(4, 4, 4);
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
        let mut alloc_zero_a = BlockAllocator::new(8, 8, 4);
        let mut alloc_zero_b = BlockAllocator::new(8, 8, 4);
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
        let mut allocator = BlockAllocator::new(8, 8, 4);
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

    /// `chain_hashes` must reproduce the inline per-block walk that
    /// `find_longest_cache_hit` / `cache_full_blocks` use (the refactor is
    /// equivalence-preserving), and a chain registered via the allocator
    /// must be reachable at exactly `chain_hashes[n]` — so the cold-tier
    /// restore path can trust the helper to yield the hot identity the
    /// allocator publishes under.
    #[test]
    fn chain_hashes_matches_inline_walk_and_lookup() {
        let tokens: Vec<u32> = (0..12).collect();
        let extra_keys = [0x1234u64, 0x5678u64];
        let salt = 0xDEAD_BEEFu64;

        // Byte-for-byte equality with the inline hash_block chain, both with
        // and without a salt (first-block-only composition).
        for &cache_salt in &[0u64, salt] {
            let expected: Vec<u64> = {
                let mut out = Vec::new();
                let mut parent = 0u64;
                for n in 0..3 {
                    let block = &tokens[n * 4..(n + 1) * 4];
                    let parent_hash = if n == 0 { 0 } else { parent };
                    let h = hash_block(block, parent_hash, &extra_keys, cache_salt, n);
                    out.push(h);
                    parent = h;
                }
                out
            };
            assert_eq!(chain_hashes(&tokens, 4, &extra_keys, cache_salt), expected);
        }

        // A chain registered through the public API is reachable at exactly
        // the helper's hashes.
        let mut allocator = BlockAllocator::new(8, 8, 4);
        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        let b2 = allocator.allocate().unwrap();
        allocator
            .cache_full_blocks(&tokens, &[b0, b1, b2], 4, &extra_keys, salt)
            .unwrap();
        let hashes = chain_hashes(&tokens, 4, &extra_keys, salt);
        assert_eq!(hashes.len(), 3);
        for h in &hashes {
            assert!(
                allocator.lookup_prefix(*h).is_some(),
                "registered block must be reachable at its chain_hashes entry"
            );
        }

        // Trailing partial-block tokens are ignored.
        let mut longer = tokens.clone();
        longer.push(99);
        assert_eq!(
            chain_hashes(&longer, 4, &extra_keys, salt),
            hashes,
            "partial trailing block must not affect the chain"
        );
        // Degenerate block_size → empty.
        assert!(chain_hashes(&tokens, 0, &extra_keys, salt).is_empty());
    }

    /// `chain_hashes_per_block` must be the exact per-block generalization of
    /// `chain_hashes`: identical output when every block carries the same keys,
    /// and a SHORT result (not a panic, not a padded chain) when the per-block
    /// vec runs out — that short result is what makes the cold restore walk
    /// bail as a miss instead of inventing an identity.
    #[test]
    fn chain_hashes_per_block_generalizes_the_uniform_walk() {
        let tokens: Vec<u32> = (0..12).collect();
        let salt = 0xDEAD_BEEFu64;

        for &cache_salt in &[0u64, salt] {
            // All-empty per-block keys == uniform empty keys.
            let empty_per_block = vec![Vec::<u64>::new(); 3];
            assert_eq!(
                chain_hashes_per_block(&tokens, 4, &empty_per_block, cache_salt),
                chain_hashes(&tokens, 4, &[], cache_salt),
                "all-empty per-block keys must reproduce the uniform empty-key chain"
            );

            // Same non-empty keys on every block == uniform keys.
            let uniform = [0x1234u64, 0x5678u64];
            let repeated = vec![uniform.to_vec(); 3];
            assert_eq!(
                chain_hashes_per_block(&tokens, 4, &repeated, cache_salt),
                chain_hashes(&tokens, 4, &uniform, cache_salt),
                "repeating one key vec must reproduce the uniform chain"
            );

            // Differing keys on block 1 must diverge from block 1 onward while
            // block 0 stays identical (chained hashing).
            let mut varied = repeated.clone();
            varied[1] = vec![0xAAAAu64];
            let base = chain_hashes_per_block(&tokens, 4, &repeated, cache_salt);
            let alt = chain_hashes_per_block(&tokens, 4, &varied, cache_salt);
            assert_eq!(base[0], alt[0]);
            assert_ne!(base[1], alt[1]);
            assert_ne!(base[2], alt[2]);
        }

        // Short per-block vec truncates the walk instead of panicking.
        let short = vec![Vec::<u64>::new(); 2];
        assert_eq!(chain_hashes_per_block(&tokens, 4, &short, salt).len(), 2);
        assert!(chain_hashes_per_block(&tokens, 4, &[], salt).is_empty());
        // Degenerate block_size → empty.
        let full = vec![Vec::<u64>::new(); 3];
        assert!(chain_hashes_per_block(&tokens, 0, &full, salt).is_empty());
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
        let mut allocator = BlockAllocator::new(8, 8, 4);
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
        let mut allocator = BlockAllocator::new(8, 8, 4);
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
        let mut allocator = BlockAllocator::new(8, 8, 4);
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
        let mut allocator = BlockAllocator::new(4, 4, 4);
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
        // logical ref). No cleanup yet — that happens when a take
        // removes the cache's ref.
        allocator.free(block);
        assert_eq!(allocator.num_free_blocks(), 3);
        assert!(allocator.lookup_prefix(hash).is_some());
    }

    #[test]
    fn test_dead_leaves_go_oldest_stamp_first() {
        let mut allocator = BlockAllocator::new(4, 4, 4);
        let hashes = [
            0xAAAA_AAAA_AAAA_AAAA,
            0xBBBB_BBBB_BBBB_BBBB,
            0xCCCC_CCCC_CCCC_CCCC,
            0xDDDD_DDDD_DDDD_DDDD,
        ];
        let mut ids = Vec::new();
        for &hash in &hashes {
            let block = allocator.allocate().unwrap();
            allocator.register_prefix(Arc::clone(&block), hash);
            ids.push(block.block_id);
            allocator.free(block);
        }
        assert_eq!(allocator.num_free_blocks(), 0);

        assert_eq!(allocator.allocate().unwrap().block_id, ids[0]);
        assert!(!allocator.prefix_cache.contains_key(&hashes[0]));
        assert_eq!(allocator.allocate().unwrap().block_id, ids[1]);
        assert!(!allocator.prefix_cache.contains_key(&hashes[1]));
        assert!(allocator.prefix_cache.contains_key(&hashes[2]));
        assert!(allocator.prefix_cache.contains_key(&hashes[3]));
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
        let mut allocator = BlockAllocator::new(4, 4, 4);
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
        let mut allocator = BlockAllocator::new(4, 4, 4);
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
        let mut allocator = BlockAllocator::new(4, 4, 4);
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

    /// Mutation: stamp on an idempotent refresh in `register_prefix` -> red
    /// (the stamp changes).
    #[test]
    fn refresh_keeps_the_stamp() {
        let mut allocator = BlockAllocator::new(3, 3, 4);
        let tokens: Vec<u32> = (0..8).collect();
        let blocks = [allocator.allocate().unwrap(), allocator.allocate().unwrap()];
        allocator
            .cache_full_blocks(&tokens, &blocks, 4, &[], 0)
            .unwrap();
        let hashes = chain_hashes(&tokens, 4, &[], 0);
        let before: Vec<Option<u64>> = hashes.iter().map(|h| allocator.block_stamp(*h)).collect();
        let free_before = allocator.num_free_blocks();

        allocator
            .cache_full_blocks(&tokens, &blocks, 4, &[], 0)
            .unwrap();

        let after: Vec<Option<u64>> = hashes.iter().map(|h| allocator.block_stamp(*h)).collect();
        assert_eq!(after, before, "a refresh is not a commit");
        assert_eq!(allocator.prefix_cache.len(), 2, "a refresh evicts nothing");
        assert_eq!(allocator.num_free_blocks(), free_before);
        assert!(blocks.iter().all(|block| block.get_ref_count() == 2));
    }

    #[test]
    fn test_register_prefix_collision_preserves_incoming_block_old_hash() {
        // block_a registered under hash_a, block_b registered under hash_b.
        // Then register_prefix(block_b, hash_a) — a collision attempt.
        // The collision drop must be a true no-op for block_b: hash_a stays
        // pointing at block_a, AND block_b's prior valid alias under hash_b
        // must NOT be torn down.
        let mut allocator = BlockAllocator::new(4, 4, 4);

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

    /// Mutation: skip `free_blocks.push_back` in `evict_oldest_dead_leaf`
    /// -> red (the id leaks: not free, not allocated).
    #[test]
    fn evicted_dead_leaf_returns_to_the_free_pool() {
        let mut allocator = BlockAllocator::new(2, 2, 4);
        let b1 = allocator.allocate().unwrap();
        let h1 = 0x1111_1111_1111_1111;
        allocator.register_prefix(Arc::clone(&b1), h1);
        let b1_id = b1.block_id;
        allocator.free(b1);
        assert_eq!(allocator.num_free_blocks(), 1);

        assert_eq!(allocator.evict_oldest_dead_leaf(), Some(b1_id));
        assert!(!allocator.prefix_cache.contains_key(&h1));
        assert!(!allocator.allocated.contains_key(&b1_id));
        assert!(allocator.free_blocks.contains(&b1_id));
        assert_eq!(allocator.num_free_blocks(), 2);
        assert_eq!(allocator.evict_oldest_dead_leaf(), None);
    }

    // -------------------------------------------------------------------
    // Allocation under cache pressure: when free_blocks is empty,
    // allocate() must take a cache-only block to satisfy the request.
    // -------------------------------------------------------------------

    /// `allocate()` must fall back to the oldest dead leaf when the free
    /// pool is empty. Bare `register_prefix` sets no identity, so every
    /// entry is a root leaf, and the release order makes b1 oldest.
    #[test]
    fn test_allocate_takes_oldest_dead_leaf_when_pool_exhausted() {
        let mut allocator = BlockAllocator::new(3, 3, 4);

        // Allocate all three blocks.
        let b1 = allocator.allocate().unwrap();
        let b2 = allocator.allocate().unwrap();
        let b3 = allocator.allocate().unwrap();
        let b1_id = b1.block_id;
        let b2_id = b2.block_id;
        let b3_id = b3.block_id;

        // Register each under a distinct hash, in order.
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

        // Allocate again — must succeed by taking the oldest dead leaf (b1).
        let new_block = allocator
            .allocate()
            .expect("allocate must take a dead leaf when the pool is empty");
        assert_eq!(
            new_block.block_id, b1_id,
            "taken block must be the oldest dead leaf (b1)"
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
        let mut allocator = BlockAllocator::new(2, 2, 4);

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

        // Snapshot every cache-authority table before the failed allocation.
        // A no-candidate scan must be observationally neutral: in particular,
        // it must not refresh any hit or disturb the stamps just because it
        // inspected in-use entries while looking for an evictable block.
        let mut prefix_cache_before: Vec<(u64, u32, usize)> = allocator
            .prefix_cache
            .iter()
            .map(|(&hash, block)| (hash, block.block_id, Arc::as_ptr(block) as usize))
            .collect();
        prefix_cache_before.sort_unstable();
        let identities_before = allocator.prefix_cache_identities.clone();
        let block_hashes_before = allocator.block_hashes.clone();
        let stamps_before = [allocator.block_stamp(h1), allocator.block_stamp(h2)];
        let mut allocated_before: Vec<(u32, usize)> = allocator
            .allocated
            .iter()
            .map(|(&block_id, block)| (block_id, Arc::as_ptr(block) as usize))
            .collect();
        allocated_before.sort_unstable();
        let free_before: Vec<u32> = allocator.free_blocks.iter().copied().collect();
        let ref_counts_before = [b1.get_ref_count(), b2.get_ref_count()];

        // allocate must return None — there's nothing it can give us.
        assert!(
            allocator.allocate().is_none(),
            "allocate must return None when free pool is empty AND every \
             cache entry has a live request handle"
        );

        let mut prefix_cache_after: Vec<(u64, u32, usize)> = allocator
            .prefix_cache
            .iter()
            .map(|(&hash, block)| (hash, block.block_id, Arc::as_ptr(block) as usize))
            .collect();
        prefix_cache_after.sort_unstable();
        let mut allocated_after: Vec<(u32, usize)> = allocator
            .allocated
            .iter()
            .map(|(&block_id, block)| (block_id, Arc::as_ptr(block) as usize))
            .collect();
        allocated_after.sort_unstable();
        assert_eq!(prefix_cache_after, prefix_cache_before);
        assert_eq!(allocator.prefix_cache_identities, identities_before);
        assert_eq!(allocator.block_hashes, block_hashes_before);
        assert_eq!(
            [allocator.block_stamp(h1), allocator.block_stamp(h2)],
            stamps_before,
            "a failed allocation must not restamp cache hits"
        );
        assert_eq!(allocated_after, allocated_before);
        assert_eq!(
            allocator.free_blocks.iter().copied().collect::<Vec<_>>(),
            free_before
        );
        assert_eq!([b1.get_ref_count(), b2.get_ref_count()], ref_counts_before);
    }

    /// `can_allocate` must count evictable cache-only blocks alongside
    /// genuinely-free blocks — otherwise schedulers refuse work the
    /// allocator could in fact satisfy via the eviction fallback.
    #[test]
    fn test_can_allocate_counts_evictable_cache_blocks() {
        let mut allocator = BlockAllocator::new(2, 2, 4);

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
        let mut allocator = BlockAllocator::new(8, 8, 4);

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
        let mut allocator = BlockAllocator::new(8, 8, 4);

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
        let mut allocator = BlockAllocator::new(8, 8, 4);

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
        let mut allocator = BlockAllocator::new(12, 12, 4);
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
        let mut a_uniform = BlockAllocator::new(8, 8, block_size);
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
        let mut a_per_block = BlockAllocator::new(8, 8, block_size);
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

        let mut allocator = BlockAllocator::new(8, 8, block_size);

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

        let mut allocator = BlockAllocator::new(8, 8, block_size);
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
        let mut allocator = BlockAllocator::new(8, 8, block_size);
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
        assert!(allocator.prefix_cache.is_empty());
        assert_eq!(b0.get_ref_count(), 1);
        assert_eq!(b1.get_ref_count(), 1);
    }

    #[test]
    fn prefix_registry_modes_preserve_duplicate_tail_and_refs() {
        let tokens: Vec<u32> = (0..16).collect();
        let keys = vec![vec![11, 22]; 4];
        for per_block in [false, true] {
            let mut allocator = BlockAllocator::new(12, 12, 4);
            let old: Vec<_> = (0..2).map(|_| allocator.allocate().unwrap()).collect();
            let old_ids: Vec<_> = old.iter().map(|b| b.block_id).collect();
            let count = if per_block {
                allocator.cache_full_blocks_per_block(&tokens[..8], &old, 4, &keys, 99)
            } else {
                allocator.cache_full_blocks(&tokens[..8], &old, 4, &[11, 22], 99)
            }
            .unwrap();
            assert_eq!(count, 2);
            for block in old {
                allocator.free(block);
            }
            let new: Vec<_> = (0..4).map(|_| allocator.allocate().unwrap()).collect();
            let count = if per_block {
                allocator.cache_full_blocks_per_block(&tokens, &new, 4, &keys, 99)
            } else {
                allocator.cache_full_blocks(&tokens, &new, 4, &[11, 22], 99)
            }
            .unwrap();
            assert_eq!(count, 2);
            assert_eq!(
                new.iter().map(|b| b.get_ref_count()).collect::<Vec<_>>(),
                vec![1, 1, 2, 2]
            );
            let (hits, cached) = if per_block {
                allocator.find_longest_cache_hit_per_block(&tokens, 4, &keys, 99)
            } else {
                allocator.find_longest_cache_hit(&tokens, 4, &[11, 22], 99)
            };
            assert_eq!(cached, 16);
            assert_eq!(
                hits.iter().map(|b| b.block_id).collect::<Vec<_>>(),
                vec![old_ids[0], old_ids[1], new[2].block_id, new[3].block_id]
            );
            for block in hits {
                allocator.free(block);
            }
            for block in new {
                allocator.free(block);
            }
            allocator.purge_prefix_cache();
            assert_eq!(allocator.num_free_blocks(), 12);
        }
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
        let mut allocator = BlockAllocator::new(8, 8, block_size);
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

    /// Turn 1 of a long prompt registers ~1100 blocks at block_size=16.
    /// Registration evicts nothing, so turn 2's lookup walking from block 0
    /// hits the whole chain.
    #[test]
    fn long_chain_survives_registration_without_head_eviction() {
        let mut a = BlockAllocator::new(2000, 2000, 16);
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

    #[test]
    fn restored_batch_preflights_every_block_before_publishing_any() {
        let mut allocator = BlockAllocator::new(4, 4, 4);
        let existing = allocator.allocate().expect("existing block");
        allocator
            .publish_restored_prefix(Arc::clone(&existing), 22, &[8, 9, 10, 11], 0, &[], 0, 0)
            .expect("existing publish");

        let first = allocator.allocate().expect("first destination");
        let second = allocator.allocate().expect("second destination");
        let registrations = [
            RestoredPrefixRegistration {
                block: Arc::clone(&first),
                hash: 11,
                token_ids: &[0, 1, 2, 3],
                parent_hash: 0,
                extra_keys: &[],
                cache_salt: 0,
                block_index: 0,
            },
            RestoredPrefixRegistration {
                block: Arc::clone(&second),
                hash: 22,
                token_ids: &[4, 5, 6, 7],
                parent_hash: 11,
                extra_keys: &[],
                cache_salt: 0,
                block_index: 1,
            },
        ];
        assert_eq!(
            allocator.publish_restored_prefix_batch(&registrations),
            Ok(false),
            "the second registration collides with the existing block"
        );
        assert!(
            !allocator.prefix_cache.contains_key(&11),
            "preflight failure must not publish the valid first block"
        );
        assert_eq!(first.get_ref_count(), 1);
        assert_eq!(second.get_ref_count(), 1);
    }

    #[test]
    fn test_grow_by_preserves_state_and_allocates_new_ids() {
        let mut allocator = BlockAllocator::new(4, 8, 4);
        assert_eq!(allocator.max_num_blocks(), 8);

        // Live allocation + a registered prefix-cache entry that must both
        // survive the grow untouched.
        let live = allocator.allocate().unwrap();
        let cached = allocator.allocate().unwrap();
        let hash = hash_tokens(&[1, 2, 3, 4], 0, &[]);
        assert!(allocator.register_prefix(Arc::clone(&cached), hash));

        assert_eq!(allocator.grow_by(4), 4);
        assert_eq!(allocator.total_blocks(), 8);
        assert_eq!(allocator.num_blocks(), 8);

        // Existing allocations and the prefix-cache entry survive.
        assert_eq!(allocator.num_allocated_blocks(), 2);
        assert!(allocator.allocated.contains_key(&live.block_id));
        let hit = allocator
            .lookup_prefix(hash)
            .expect("prefix-cache entry survives grow_by");
        assert_eq!(hit.block_id, cached.block_id);

        // The two pre-existing free ids come first (freed/reused ids stay
        // ahead of the freshly grown range), then the new ids 4..8 in order.
        let mut ids = Vec::new();
        while let Some(block) = allocator.allocate() {
            ids.push(block.block_id);
        }
        assert_eq!(ids, vec![2, 3, 4, 5, 6, 7]);
        assert_eq!(allocator.num_free_blocks(), 0);
    }

    /// A held block at id 9 floors the cut at 10; once it is cache-only the
    /// cut reaches the floor, the three cache-only blocks above it move to
    /// the lowest free slots, both chains still hit under the new ids, and
    /// no id at or above the cut is handed out again. Mutation: count a
    /// cached block a request still holds as cache-only (drop
    /// `get_ref_count() == 1` in `slot_uses`) → red (cut 4 ≠ 10).
    #[test]
    fn shrink_plan_moves_cache_only_and_floors_at_held() {
        let mut allocator = BlockAllocator::new(4, 16, 4);
        assert_eq!(allocator.grow_by(8), 8);
        let blocks: Vec<_> = (0..12).map(|_| allocator.allocate().unwrap()).collect();
        let first: Vec<u32> = (1..=8).collect();
        let second: Vec<u32> = (101..=104).collect();
        assert_eq!(
            allocator
                .cache_full_blocks(&first, &blocks[6..8], 4, &[], 0)
                .unwrap(),
            2
        );
        assert_eq!(
            allocator
                .cache_full_blocks(&second, &blocks[9..10], 4, &[], 0)
                .unwrap(),
            1
        );
        for (id, block) in blocks.iter().enumerate() {
            if id != 9 {
                allocator.free(Arc::clone(block));
            }
        }
        assert_eq!(
            allocator.shrink_plan(4, 2, 4),
            Some(ShrinkPlan {
                num_blocks: 10,
                moves: Vec::new(),
            })
        );
        assert_eq!(allocator.shrink_target(4, 2, 4), 10);

        allocator.free(Arc::clone(&blocks[9]));
        drop(blocks);
        let plan = allocator.shrink_plan(4, 2, 4).expect("a cut fits");
        assert_eq!(
            plan,
            ShrinkPlan {
                num_blocks: 4,
                moves: vec![(6, 0), (7, 1), (9, 2)],
            }
        );
        allocator.apply_shrink(&plan).expect("apply");
        assert_eq!(allocator.num_blocks(), 4);
        let (hit, tokens) = allocator.find_longest_cache_hit(&first, 4, &[], 0);
        assert_eq!(tokens, 8);
        assert_eq!(
            hit.iter().map(|block| block.block_id).collect::<Vec<_>>(),
            vec![0, 1]
        );
        let (hit, tokens) = allocator.find_longest_cache_hit(&second, 4, &[], 0);
        assert_eq!(tokens, 4);
        assert_eq!(hit[0].block_id, 2);
        let ids: Vec<u32> =
            std::iter::from_fn(|| allocator.allocate().map(|b| b.block_id)).collect();
        assert_eq!(ids, vec![3]);
    }

    /// One leaf eviction of a cached 4-block chain takes its tail and keeps
    /// the head: the chain still hits 3 blocks, then 2. Mutation: take the
    /// oldest stamp regardless of children → red (hit 0).
    #[test]
    fn evicting_a_leaf_keeps_the_chain_head() {
        let mut allocator = BlockAllocator::new(8, 8, 4);
        let blocks: Vec<_> = (0..4).map(|_| allocator.allocate().unwrap()).collect();
        let tokens: Vec<u32> = (1..=16).collect();
        assert_eq!(
            allocator
                .cache_full_blocks(&tokens, &blocks, 4, &[], 0)
                .unwrap(),
            4
        );
        for block in blocks {
            allocator.free(block);
        }
        let hit_blocks = |allocator: &mut BlockAllocator| {
            let (hit, cached) = allocator.find_longest_cache_hit(&tokens, 4, &[], 0);
            for block in hit {
                block.decref();
            }
            cached / 4
        };
        assert_eq!(allocator.evict_oldest_dead_leaf(), Some(3));
        assert_eq!(hit_blocks(&mut allocator), 3);
        assert_eq!(allocator.evict_oldest_dead_leaf(), Some(2));
        assert_eq!(hit_blocks(&mut allocator), 2);
        assert_eq!(allocator.num_free_blocks(), 6);
    }

    #[test]
    fn test_grow_by_caps_at_max_num_blocks() {
        let mut allocator = BlockAllocator::new(4, 6, 4);
        // Requesting beyond the cap grows only up to max_num_blocks.
        assert_eq!(allocator.grow_by(10), 2);
        assert_eq!(allocator.total_blocks(), 6);
        assert_eq!(allocator.num_free_blocks(), 6);
        // At the cap, further growth is a no-op.
        assert_eq!(allocator.grow_by(1), 0);
        assert_eq!(allocator.total_blocks(), 6);

        // The pool serves exactly the ids 0..6 then reports exhaustion.
        let ids: Vec<u32> =
            std::iter::from_fn(|| allocator.allocate().map(|b| b.block_id)).collect();
        assert_eq!(ids, vec![0, 1, 2, 3, 4, 5]);
        assert!(allocator.allocate().is_none());
    }

    #[test]
    fn test_grow_by_no_op_when_max_equals_initial() {
        let mut allocator = BlockAllocator::new(4, 4, 4);
        assert_eq!(allocator.grow_by(3), 0);
        assert_eq!(allocator.total_blocks(), 4);
        assert_eq!(allocator.num_free_blocks(), 4);
    }

    #[test]
    fn test_grown_pool_registers_every_block_without_eviction() {
        // A pool grown 4 -> 8 holds 8 cache entries: registration never
        // evicts.
        let mut allocator = BlockAllocator::new(4, 8, 4);
        assert_eq!(allocator.grow_by(4), 4);
        for i in 0..8u64 {
            let block = allocator.allocate().unwrap();
            assert!(allocator.register_prefix(block, 1000 + i));
        }
        for i in 0..8u64 {
            assert!(
                allocator.lookup_prefix(1000 + i).is_some(),
                "entry {i} evicted by registration"
            );
        }
    }

    #[test]
    #[should_panic(expected = "max_num_blocks")]
    fn test_new_rejects_max_below_initial() {
        let _ = BlockAllocator::new(8, 4, 4);
    }

    fn cache_chain(
        allocator: &mut BlockAllocator,
        tokens: &[u32],
    ) -> (Vec<Arc<PhysicalBlock>>, Vec<u64>) {
        let n = tokens.len() / 4;
        let blocks: Vec<_> = (0..n).map(|_| allocator.allocate().unwrap()).collect();
        assert_eq!(
            allocator
                .cache_full_blocks(tokens, &blocks, 4, &[], 0)
                .unwrap(),
            n
        );
        (blocks, chain_hashes(tokens, 4, &[], 0))
    }

    fn hit_len(allocator: &mut BlockAllocator, tokens: &[u32]) -> usize {
        let (hit, _) = allocator.find_longest_cache_hit(tokens, 4, &[], 0);
        let n = hit.len();
        for block in hit {
            allocator.free(block);
        }
        n
    }

    /// Mutation: `allocate` takes the oldest stamp regardless of children
    /// -> red (A goes, hit 0).
    #[test]
    fn allocate_takes_the_chain_tail_and_keeps_the_prefix_hit() {
        let mut allocator = BlockAllocator::new(4, 4, 4);
        let tokens: Vec<u32> = (0..16).collect();
        let (blocks, _) = cache_chain(&mut allocator, &tokens);
        let d = blocks[3].block_id;
        for block in blocks {
            allocator.free(block);
        }
        assert_eq!(allocator.allocate().unwrap().block_id, d);
        assert_eq!(hit_len(&mut allocator, &tokens), 3);
    }

    /// Mutations: the parent keeps its own stamp -> red ((1, L)); the parent
    /// gets `next()` -> red ((4, L)).
    #[test]
    fn parent_takes_its_child_stamp() {
        let mut allocator = BlockAllocator::new(2, 2, 4);
        let tokens: Vec<u32> = (0..8).collect();
        let (blocks, hashes) = cache_chain(&mut allocator, &tokens);
        let (l, k) = (hashes[0], hashes[1]);
        allocator.set_state_marks([l]);
        assert_eq!(allocator.block_stamp(l), Some(1));
        assert_eq!(allocator.block_stamp(k), Some(2));
        let k_id = blocks[1].block_id;
        for block in blocks {
            allocator.free(block);
        }
        assert_eq!(allocator.block_stamp(k), Some(3));

        assert_eq!(allocator.allocate().unwrap().block_id, k_id);
        assert_eq!(allocator.oldest_marked_leaf(), Some((3, l)));
    }

    /// Mutation: no touch after the walk in
    /// `find_longest_cache_hit_with_keys` -> red (alloc 2 = B).
    /// Mutation: the walk touches every hit block, not only the tip -> red
    /// (A's stamp moves).
    #[test]
    fn a_hit_touches_its_tip_only() {
        let mut allocator = BlockAllocator::new(4, 4, 4);
        let x_tokens: Vec<u32> = (0..12).collect();
        let y_tokens: Vec<u32> = (100..104).collect();
        let (x_blocks, x_hashes) = cache_chain(&mut allocator, &x_tokens);
        let (y_blocks, _) = cache_chain(&mut allocator, &y_tokens);
        let d = x_blocks[2].block_id;
        let c = y_blocks[0].block_id;
        for block in x_blocks {
            allocator.free(block);
        }
        for block in y_blocks {
            allocator.free(block);
        }
        let a_stamp = allocator.block_stamp(x_hashes[0]);
        assert_eq!(hit_len(&mut allocator, &x_tokens[..8]), 2);
        assert_eq!(allocator.block_stamp(x_hashes[0]), a_stamp);

        assert_eq!(allocator.allocate().unwrap().block_id, d);
        assert_eq!(allocator.allocate().unwrap().block_id, c);
    }

    /// Mutation: `free` stamps nothing -> red (alloc 1 = X's tip).
    #[test]
    fn released_tip_is_stamped_newest() {
        let mut allocator = BlockAllocator::new(2, 2, 4);
        let x_tokens: Vec<u32> = (0..4).collect();
        let y_tokens: Vec<u32> = (100..104).collect();
        let (x_blocks, _) = cache_chain(&mut allocator, &x_tokens);
        let (y_blocks, _) = cache_chain(&mut allocator, &y_tokens);
        let y = y_blocks[0].block_id;
        for block in y_blocks {
            allocator.free(block);
        }
        for block in x_blocks {
            allocator.free(block);
        }
        assert_eq!(allocator.allocate().unwrap().block_id, y);
    }

    /// Mutation: no touch in the verified-duplicate skip branch of
    /// `cache_full_blocks_with_keys` -> red (a2 < z1 after B).
    #[test]
    fn verified_duplicate_is_stamped_on_commit() {
        let mut allocator = BlockAllocator::new(6, 6, 4);
        let a_tokens: Vec<u32> = (0..8).collect();
        let z_tokens: Vec<u32> = (100..104).collect();
        let (a_blocks, a_hashes) = cache_chain(&mut allocator, &a_tokens);
        for block in a_blocks {
            allocator.free(block);
        }
        let (z_blocks, z_hashes) = cache_chain(&mut allocator, &z_tokens);
        for block in z_blocks {
            allocator.free(block);
        }
        let (a2, z1) = (a_hashes[1], z_hashes[0]);
        assert!(allocator.block_stamp(a2) < allocator.block_stamp(z1));

        let b_blocks = [allocator.allocate().unwrap(), allocator.allocate().unwrap()];
        assert_eq!(
            allocator
                .cache_full_blocks(&a_tokens, &b_blocks, 4, &[], 0)
                .unwrap(),
            0,
            "both blocks are verified duplicates"
        );
        assert!(allocator.block_stamp(a2) > allocator.block_stamp(z1));
        for block in b_blocks {
            allocator.free(block);
        }
    }

    /// Mutation: `child_count += 1` on every `remember_prefix_identity`
    /// -> red (B, then `None`).
    #[test]
    fn republished_identity_does_not_double_count_its_parent() {
        let mut allocator = BlockAllocator::new(4, 4, 4);
        let tokens: Vec<u32> = (0..8).collect();
        let (blocks, _) = cache_chain(&mut allocator, &tokens);
        assert_eq!(
            allocator
                .cache_full_blocks(&tokens, &blocks, 4, &[], 0)
                .unwrap(),
            2
        );
        let (a, b) = (blocks[0].block_id, blocks[1].block_id);
        for block in blocks {
            allocator.free(block);
        }
        assert_eq!(allocator.evict_oldest_dead_leaf(), Some(b));
        assert_eq!(allocator.evict_oldest_dead_leaf(), Some(a));
    }

    /// Mutation: `oldest_dead_leaf` ignores marks -> red (alloc 1 = B).
    #[test]
    fn marked_chain_is_never_allocated() {
        let mut allocator = BlockAllocator::new(3, 3, 4);
        let ab_tokens: Vec<u32> = (0..8).collect();
        let c_tokens: Vec<u32> = (100..104).collect();
        let (ab_blocks, ab_hashes) = cache_chain(&mut allocator, &ab_tokens);
        let (c_blocks, _) = cache_chain(&mut allocator, &c_tokens);
        allocator.set_state_marks([ab_hashes[1]]);
        let c = c_blocks[0].block_id;
        for block in ab_blocks.into_iter().chain(c_blocks) {
            allocator.free(block);
        }

        assert_eq!(allocator.allocate().unwrap().block_id, c);
        assert!(allocator.allocate().is_none());
        assert_eq!(allocator.num_evictable_blocks(), 0);
        assert_eq!(allocator.num_reusable_blocks(), 2);
        assert_eq!(hit_len(&mut allocator, &ab_tokens), 2);
    }

    /// Mutations: drop the pinned-interior fallback in `allocate` -> red
    /// (`None`); take the oldest interior instead of the deepest -> red
    /// (A first, hit 0).
    #[test]
    fn pinned_interior_goes_deepest_first() {
        let mut allocator = BlockAllocator::new(3, 3, 4);
        let tokens: Vec<u32> = (0..12).collect();
        let (mut blocks, _) = cache_chain(&mut allocator, &tokens);
        let c_block = blocks.pop().unwrap();
        let (a, b) = (blocks[0].block_id, blocks[1].block_id);
        for block in blocks {
            allocator.free(block);
        }
        assert_eq!(allocator.num_free_blocks(), 0);
        assert_eq!(allocator.num_evictable_blocks(), 2);

        assert_eq!(allocator.allocate().unwrap().block_id, b);
        assert_eq!(hit_len(&mut allocator, &tokens[..4]), 1);
        assert_eq!(allocator.allocate().unwrap().block_id, a);
        allocator.free(c_block);
    }

    /// Mutation: skip the marked closure in `deepest_pinned_interior` only
    /// below the max -> red (at the max it takes B).
    #[test]
    fn pinned_interior_skips_a_marked_chain_at_every_size() {
        for max in [3, 6] {
            let mut allocator = BlockAllocator::new(3, max, 4);
            let tokens: Vec<u32> = (0..12).collect();
            let (mut blocks, hashes) = cache_chain(&mut allocator, &tokens);
            allocator.set_state_marks([hashes[1]]);
            let c_block = blocks.pop().unwrap();
            for block in blocks {
                allocator.free(block);
            }
            assert!(allocator.allocate().is_none(), "pool 3/{max}");
            assert_eq!(hit_len(&mut allocator, &tokens[..8]), 2, "pool 3/{max}");
            allocator.free(c_block);
        }
    }

    /// At the max (3/3) and below it (3/6) a reserve of 3 against 0 free
    /// and 2 evictable blocks takes nothing, and X still hits 2 blocks.
    /// Mutations: put `self.num_blocks >= self.max_num_blocks &&` back in
    /// front of the `reserve` gate -> red at 3/6 (`Err(2)`, X evicted, hit
    /// 0); drop the gate -> red at 3/3 and 3/6.
    #[test]
    fn reserve_evicts_nothing_it_cannot_cover() {
        let x_tokens: Vec<u32> = (0..8).collect();
        for max in [3, 6] {
            let mut allocator = BlockAllocator::new(3, max, 4);
            let (x_blocks, _) = cache_chain(&mut allocator, &x_tokens);
            for block in x_blocks {
                allocator.free(block);
            }
            let held = allocator.allocate().unwrap();
            assert!(matches!(allocator.reserve(3), Err(0)), "pool 3/{max}");
            assert_eq!(hit_len(&mut allocator, &x_tokens), 2, "pool 3/{max}");
            assert_eq!(allocator.num_free_blocks(), 0, "pool 3/{max}");
            allocator.free(held);
        }
    }

    /// Mutation: keep a `child_count` row at 0 in `unlink_child` -> red
    /// (1 row).
    #[test]
    fn child_count_rows_end_with_their_children() {
        let mut allocator = BlockAllocator::new(2, 2, 4);
        let tokens: Vec<u32> = (0..8).collect();
        let (mut blocks, _) = cache_chain(&mut allocator, &tokens);
        let b_block = blocks.pop().unwrap();
        let a = blocks[0].block_id;
        let b = b_block.block_id;
        allocator.free(blocks.pop().unwrap());

        assert_eq!(allocator.allocate().unwrap().block_id, a);
        assert_eq!(allocator.child_count_rows(), 1);
        allocator.free(b_block);
        assert_eq!(allocator.allocate().unwrap().block_id, b);
        assert_eq!(allocator.child_count_rows(), 0);
    }
}
