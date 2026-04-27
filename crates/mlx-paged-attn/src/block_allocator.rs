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

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

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

    /// Reverse mapping: block_id -> hash (for cleanup during free)
    block_hashes: HashMap<u32, u64>,

    /// LRU order for prefix cache eviction (oldest first)
    lru_order: VecDeque<u64>,

    /// Maximum entries in prefix cache
    max_prefix_cache_entries: usize,
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
            block_hashes: HashMap::new(),
            lru_order: VecDeque::new(),
            max_prefix_cache_entries: 1024, // Configurable
        }
    }

    /// Allocate a new block
    ///
    /// Returns None if no free blocks are available
    pub fn allocate(&mut self) -> Option<Arc<PhysicalBlock>> {
        let block_id = self.free_blocks.pop_front()?;
        let block = Arc::new(PhysicalBlock::new(block_id));
        self.allocated.insert(block_id, Arc::clone(&block));
        Some(block)
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
                self.lru_order.retain(|&h| h != hash);
            }

            self.free_blocks.push_back(block_id);
        }
    }

    /// Perform copy-on-write for a shared block
    ///
    /// If the block is shared (ref_count > 1), allocates a new block
    /// and returns it. Otherwise returns None (no copy needed).
    pub fn copy_on_write(&mut self, block: &Arc<PhysicalBlock>) -> Option<Arc<PhysicalBlock>> {
        if !block.is_shared() {
            return None;
        }

        // Allocate new block
        let new_block = self.allocate()?;

        // Decrement old block's ref count
        block.decref();

        Some(new_block)
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
    pub fn register_prefix(&mut self, block: Arc<PhysicalBlock>, hash: u64) {
        // If prefix caching is disabled (max_prefix_cache_entries == 0), do nothing
        if self.max_prefix_cache_entries == 0 {
            return;
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
            return;
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
            self.lru_order.retain(|&h| h != existing_hash);
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

        // Step 4: LRU refresh (remove if exists, add to end) + insert.
        // Take the cache's logical reference iff this is a genuine new
        // insertion. An idempotent refresh leaves ref_count unchanged so
        // the cache continues to hold exactly ONE logical ref per entry.
        self.lru_order.retain(|&h| h != hash);
        self.lru_order.push_back(hash);

        // Track the hash for this block (for cleanup during free)
        self.block_hashes.insert(block.block_id, hash);

        if is_new_insertion {
            block.incref();
        }

        // Insert into cache
        self.prefix_cache.insert(hash, block);
    }

    /// Look up a block in the prefix cache
    ///
    /// Returns the cached block if found, incrementing its ref count
    pub fn lookup_prefix(&mut self, hash: u64) -> Option<Arc<PhysicalBlock>> {
        if let Some(block) = self.prefix_cache.get(&hash) {
            // Update LRU order
            self.lru_order.retain(|&h| h != hash);
            self.lru_order.push_back(hash);

            // Increment ref count and return
            block.incref();
            Some(Arc::clone(block))
        } else {
            None
        }
    }

    /// Walk a token sequence in `block_size`-aligned chunks, looking up each
    /// block in the prefix cache. Stop at the first miss. Returns the cached
    /// blocks (with their ref counts already bumped via `lookup_prefix`, in
    /// order) and the cached token count.
    ///
    /// `extra_keys` is applied per-block-hash (same value for every block in
    /// this call). Phase 6 will thread per-block extra_keys for multimodal
    /// (image hashes, cache-salt, LoRA names, etc.).
    ///
    /// Mirrors vLLM `vllm/v1/core/single_type_kv_cache_manager.py:421-468`
    /// (`FullAttentionManager.find_longest_cache_hit`).
    pub fn find_longest_cache_hit(
        &mut self,
        token_ids: &[u32],
        block_size: u32,
        extra_keys: &[u64],
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
            let parent_hash = if n == 0 { 0 } else { previous_block_hash };
            let block_hash = hash_tokens(&token_ids[start..end], parent_hash, extra_keys);

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
    /// `extra_keys` is applied per-block-hash (same value for every block in
    /// this call). Phase 6 will thread per-block extra_keys for multimodal.
    ///
    /// Mirrors vLLM `vllm/v1/core/block_pool.py:211-320` (`cache_full_blocks`).
    pub fn cache_full_blocks(
        &mut self,
        token_ids: &[u32],
        blocks: &[Arc<PhysicalBlock>],
        block_size: u32,
        extra_keys: &[u64],
    ) -> Result<(), &'static str> {
        if block_size == 0 {
            return Err("block_size must be > 0");
        }

        let block_size_us = block_size as usize;
        if blocks.len() * block_size_us > token_ids.len() {
            return Err("blocks exceed token_ids length");
        }

        let mut previous_block_hash: u64 = 0;
        for (n, block) in blocks.iter().enumerate() {
            let start = n * block_size_us;
            let end = start + block_size_us;
            let parent_hash = if n == 0 { 0 } else { previous_block_hash };
            let block_hash = hash_tokens(&token_ids[start..end], parent_hash, extra_keys);
            self.register_prefix(Arc::clone(block), block_hash);
            previous_block_hash = block_hash;
        }
        Ok(())
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

    /// Get the block size
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Check if we can allocate the requested number of blocks
    pub fn can_allocate(&self, num_blocks: u32) -> bool {
        self.num_free_blocks() >= num_blocks
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

    /// Get the current prefix-cache capacity ceiling.
    pub fn max_prefix_cache_entries(&self) -> usize {
        self.max_prefix_cache_entries
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
    fn test_copy_on_write() {
        let mut allocator = BlockAllocator::new(10, 32);

        let block = allocator.allocate().unwrap();

        // No copy needed when not shared
        assert!(allocator.copy_on_write(&block).is_none());

        // Share the block (like for beam search)
        block.incref();
        let block2 = Arc::clone(&block);

        // Now copy-on-write should allocate new block
        let new_block = allocator.copy_on_write(&block).unwrap();
        assert_ne!(new_block.block_id, block.block_id);
        assert_eq!(block.get_ref_count(), 1); // Decremented by copy_on_write
        assert_eq!(new_block.get_ref_count(), 1);

        // Clean up
        allocator.free(block2);
        allocator.free(new_block);
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

        // Should not cache when disabled
        allocator.register_prefix(Arc::clone(&block), hash);

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

    // -------------------------------------------------------------------
    // Phase 1A additions: hash_tokens(extra_keys), find_longest_cache_hit,
    // cache_full_blocks, refcount lifecycle, LRU eviction order.
    // -------------------------------------------------------------------

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
        let (blocks, n) = allocator.find_longest_cache_hit(&[1, 2, 3, 4, 5, 6, 7, 8], 4, &[]);
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
            .cache_full_blocks(&tokens, &blocks, 4, &[])
            .unwrap();

        let (hit_blocks, n) = allocator.find_longest_cache_hit(&tokens, 4, &[]);
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
            .cache_full_blocks(&tokens_a, &[b0, b1], 4, &[])
            .unwrap();

        let (hit_blocks, n) = allocator.find_longest_cache_hit(&tokens_b, 4, &[]);
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
            .cache_full_blocks(&tokens_a, &[b0, b1, b2], 4, &[])
            .unwrap();

        let (hit_blocks, n) = allocator.find_longest_cache_hit(&tokens_b, 4, &[]);
        assert_eq!(hit_blocks.len(), 1);
        assert_eq!(n, 4);
    }

    #[test]
    fn test_find_longest_cache_hit_short_input() {
        let mut allocator = BlockAllocator::new(4, 4);
        let (blocks, n) = allocator.find_longest_cache_hit(&[1, 2, 3], 4, &[]);
        assert!(blocks.is_empty());
        assert_eq!(n, 0);

        let (blocks, n) = allocator.find_longest_cache_hit(&[], 4, &[]);
        assert!(blocks.is_empty());
        assert_eq!(n, 0);
    }

    #[test]
    fn test_find_longest_cache_hit_zero_block_size() {
        // Defensive: block_size == 0 should not panic / infinite loop.
        let mut allocator = BlockAllocator::new(4, 4);
        let (blocks, n) = allocator.find_longest_cache_hit(&[1, 2, 3, 4], 0, &[]);
        assert!(blocks.is_empty());
        assert_eq!(n, 0);
    }

    #[test]
    fn test_cache_full_blocks_oversize_returns_err() {
        let mut allocator = BlockAllocator::new(4, 4);
        let b0 = allocator.allocate().unwrap();
        let b1 = allocator.allocate().unwrap();
        // 2 blocks * block_size 4 = 8 tokens required, but only 4 supplied.
        let res = allocator.cache_full_blocks(&[1, 2, 3, 4], &[b0, b1], 4, &[]);
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
            .cache_full_blocks(&tokens, &[b0, b1], 4, &[100])
            .unwrap();

        let (blocks, n) = allocator.find_longest_cache_hit(&tokens, 4, &[]);
        assert!(blocks.is_empty());
        assert_eq!(n, 0);

        // Same extra_keys -> hit.
        let (blocks, n) = allocator.find_longest_cache_hit(&tokens, 4, &[100]);
        assert_eq!(blocks.len(), 2);
        assert_eq!(n, 8);
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

    // -------------------------------------------------------------------
    // Phase 1A bugfix: register_prefix must evict stale aliases when the
    // same block is re-registered under a different hash, otherwise a
    // freed block can leak back through lookup_prefix on the stale hash.
    // -------------------------------------------------------------------

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

        allocator.register_prefix(Arc::clone(&block), hash_a);
        // After first register: alloc(1) + register(1) = 2.
        assert_eq!(block.get_ref_count(), 2);

        allocator.register_prefix(Arc::clone(&block), hash_b);
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
        // The Codex-identified scenario: cache the same blocks under two
        // different extra_keys (no_keys vs [99]). After freeing, neither
        // hash can hand back the freed block via find_longest_cache_hit —
        // i.e. extra_keys isolation must hold across freed entries.
        let mut allocator = BlockAllocator::new(4, 4);
        let tokens: Vec<u32> = vec![1, 2, 3, 4];

        let b0 = allocator.allocate().unwrap();
        let blocks = [Arc::clone(&b0)];

        allocator
            .cache_full_blocks(&tokens, &blocks, 4, &[])
            .unwrap();
        // After first cache_full_blocks: alloc(1) + register(1) = 2.
        assert_eq!(b0.get_ref_count(), 2);

        allocator
            .cache_full_blocks(&tokens, &blocks, 4, &[99])
            .unwrap();
        // Second cache_full_blocks: same block under different hash →
        // Case 1 path. Decref old, incref new → net unchanged = 2.
        assert_eq!(b0.get_ref_count(), 2);

        // Free the external handle. ref_count: 2 -> 1. The cache still
        // holds the block under hash_b (extra_keys=[99]).
        allocator.free(b0);

        // The empty-extra_keys alias was displaced by the second register;
        // it is gone.
        let (hits_none, n_none) = allocator.find_longest_cache_hit(&tokens, 4, &[]);
        assert!(hits_none.is_empty(), "stale extra_keys=[] alias leaked");
        assert_eq!(n_none, 0);

        // The [99] alias still resolves — block survived because the cache
        // still holds its logical reference. This is correct: the cached
        // block is consistent with extra_keys=[99] and a future lookup
        // with the matching extra_keys is a legitimate hit.
        let (hits_99, n_99) = allocator.find_longest_cache_hit(&tokens, 4, &[99]);
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

        allocator.register_prefix(Arc::clone(&block_a), hash_x);
        // alloc(1) + register(1) = 2.
        assert_eq!(block_a.get_ref_count(), 2);

        allocator.register_prefix(Arc::clone(&block_b), hash_x);
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
}
