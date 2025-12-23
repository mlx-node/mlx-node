//! Block allocator for PagedAttention KV cache
//!
//! Manages a pool of fixed-size physical blocks that can be allocated
//! to sequences on demand. Supports:
//! - Reference counting for copy-on-write (beam search)
//! - Prefix caching via content-based hashing
//! - LRU eviction for cache management

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
    /// The block will be reused when a sequence has matching prefix tokens
    pub fn register_prefix(&mut self, block: Arc<PhysicalBlock>, hash: u64) {
        // If prefix caching is disabled (max_prefix_cache_entries == 0), do nothing
        if self.max_prefix_cache_entries == 0 {
            return;
        }

        // Evict oldest entries if at capacity
        while self.prefix_cache.len() >= self.max_prefix_cache_entries {
            match self.lru_order.pop_front() {
                Some(old_hash) => {
                    // Remove evicted entry from both caches
                    if let Some(evicted_block) = self.prefix_cache.remove(&old_hash) {
                        self.block_hashes.remove(&evicted_block.block_id);
                    }
                }
                None => {
                    // Safety: If lru_order is empty but cache still has entries,
                    // this indicates a bug (desynchronization). Break to avoid infinite loop.
                    break;
                }
            }
        }

        // Update LRU order (remove if exists, add to end)
        self.lru_order.retain(|&h| h != hash);
        self.lru_order.push_back(hash);

        // Track the hash for this block (for cleanup during free)
        self.block_hashes.insert(block.block_id, hash);

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
}

/// Hash function for token sequences (for prefix caching)
pub fn hash_tokens(tokens: &[u32], parent_hash: u64) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    parent_hash.hash(&mut hasher);
    for &token in tokens {
        token.hash(&mut hasher);
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
        let hash = hash_tokens(&[1, 2, 3], 0);

        allocator.register_prefix(Arc::clone(&block), hash);

        // Lookup should find the block
        let cached = allocator.lookup_prefix(hash).unwrap();
        assert_eq!(cached.block_id, block.block_id);
        // Original (1) + lookup increments (1) = 2
        assert_eq!(cached.get_ref_count(), 2);

        // Unknown hash should return None
        assert!(allocator.lookup_prefix(12345).is_none());
    }

    #[test]
    fn test_prefix_cache_cleanup_on_free() {
        // This test verifies the memory leak fix:
        // When a block is freed, it must be removed from prefix_cache
        let mut allocator = BlockAllocator::new(10, 32);

        let block = allocator.allocate().unwrap();
        let hash = hash_tokens(&[1, 2, 3], 0);

        // Register in prefix cache (doesn't increment ref_count)
        allocator.register_prefix(Arc::clone(&block), hash);

        // Verify it's in the cache
        let cached = allocator.lookup_prefix(hash).unwrap();
        assert_eq!(cached.block_id, block.block_id);
        // ref_count: original(1) + lookup(1) = 2

        // Free both references
        allocator.free(block); // ref_count: 2 -> 1
        allocator.free(cached); // ref_count: 1 -> 0, block freed

        // Verify it's removed from prefix cache after free
        assert!(allocator.lookup_prefix(hash).is_none());

        // Verify the block is back in the free pool
        assert_eq!(allocator.num_free_blocks(), 10);
    }

    #[test]
    fn test_prefix_cache_eviction_cleanup() {
        // This test verifies that evicted blocks are properly cleaned up
        let mut allocator = BlockAllocator::new(10, 32);
        allocator.max_prefix_cache_entries = 2; // Small cache for testing

        let block1 = allocator.allocate().unwrap();
        let hash1 = hash_tokens(&[1], 0);
        allocator.register_prefix(Arc::clone(&block1), hash1);

        let block2 = allocator.allocate().unwrap();
        let hash2 = hash_tokens(&[2], 0);
        allocator.register_prefix(Arc::clone(&block2), hash2);

        // Cache is at capacity (2 entries)
        assert_eq!(allocator.prefix_cache.len(), 2);

        // Add a third block, should evict the first (LRU)
        let block3 = allocator.allocate().unwrap();
        let hash3 = hash_tokens(&[3], 0);
        allocator.register_prefix(Arc::clone(&block3), hash3);

        // Verify hash1 was evicted
        assert!(allocator.lookup_prefix(hash1).is_none());
        assert!(allocator.lookup_prefix(hash2).is_some());
        assert!(allocator.lookup_prefix(hash3).is_some());

        // Verify block_hashes was also cleaned up
        assert!(!allocator.block_hashes.contains_key(&block1.block_id));
        assert!(allocator.block_hashes.contains_key(&block2.block_id));
        assert!(allocator.block_hashes.contains_key(&block3.block_id));
    }

    #[test]
    fn test_prefix_cache_disabled() {
        // This test verifies that setting max_prefix_cache_entries = 0 disables caching
        // and doesn't cause infinite loop
        let mut allocator = BlockAllocator::new(10, 32);
        allocator.max_prefix_cache_entries = 0; // Disable prefix caching

        let block = allocator.allocate().unwrap();
        let hash = hash_tokens(&[1, 2, 3], 0);

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
        let hash1 = hash_tokens(&[1], 0);
        allocator.register_prefix(Arc::clone(&block1), hash1);

        // Manually desynchronize: clear lru_order but leave prefix_cache populated
        allocator.lru_order.clear();

        // This should not infinite loop - it will break when pop_front returns None
        let block2 = allocator.allocate().unwrap();
        let hash2 = hash_tokens(&[2], 0);
        allocator.register_prefix(Arc::clone(&block2), hash2);

        // Verify we didn't infinite loop and the function completed
        assert!(!allocator.prefix_cache.is_empty());
    }
}
