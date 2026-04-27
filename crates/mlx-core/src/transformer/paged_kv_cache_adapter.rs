//! PagedKVCacheAdapter — session-friendly wrapper over `mlx_paged_attn::BlockAllocator`
//!
//! Replaces per-model `Vec<KVCache>` storage with block-paged KV. Multiple
//! conversations sharing a system prompt can reference the same physical SYS
//! blocks (refcount > 1) without evicting each other — the vLLM block-paged
//! design (see `vllm/v1/core/block_pool.py` and `kv_cache_utils.py`).
//!
//! This step (P1C-1) only handles block lifecycle, prefix lookup, and
//! registration. Metal kernel dispatch (reshape_and_cache, paged_attention)
//! and GPU buffer management are out of scope; they land in P1C-2 / P1C-3.
//!
//! ## Scope
//!
//! The adapter is for **FULL ATTENTION layers only**. Sliding-window /
//! rotating-cache layers and hybrid recurrent (e.g. Lfm2 GDN) layers continue
//! to use their existing dedicated cache types and are outside the
//! responsibility of this adapter.
//!
//! ## Lifecycle contract
//!
//! Each adapter instance is scoped to ONE in-flight request at a time. The
//! caller flow is:
//!
//! 1. `reset_for_new_request(seq_id)` — releases any prior request.
//! 2. `find_cached_prefix(prompt_tokens, extra_keys)` — populates block_table
//!    with reused prefix blocks (refcounts already incremented).
//! 3. `allocate_suffix_blocks(total_tokens)` — allocates fresh blocks to
//!    cover the remainder of the prompt + decode budget.
//! 4. `record_tokens(...)` — every token consumed (prefill batch + each
//!    decoded token), in order.
//! 5. On success, optionally `register_full_blocks_for_reuse(extra_keys)` to
//!    publish full blocks for cross-request prefix reuse.
//! 6. `release_request()` — decrefs every block in the table. Blocks still
//!    referenced by the prefix cache (registered above) survive at refcount
//!    > 0; otherwise return to the free pool.

use std::sync::{Arc, Mutex};

use mlx_paged_attn::{BlockAllocator, PhysicalBlock, SequenceBlockTable};

/// Result of a prefix-cache lookup.
#[derive(Debug)]
pub struct CachedPrefix {
    /// Physical blocks reused from the prefix cache (refcount already incremented).
    pub blocks: Vec<Arc<PhysicalBlock>>,
    /// Number of tokens covered by `blocks` (always a multiple of `block_size`).
    pub cached_token_count: u32,
}

/// Per-model session-friendly KV cache adapter.
///
/// Holds shared `BlockAllocator` (`Arc<Mutex<...>>`) so multiple in-flight
/// requests on the same model can share blocks. Each adapter instance is
/// scoped to ONE request at a time — call `reset_for_new_request` between
/// requests.
pub struct PagedKVCacheAdapter {
    allocator: Arc<Mutex<BlockAllocator>>,
    block_size: u32,

    /// Block table for the active request. None between requests.
    block_table: Option<SequenceBlockTable>,

    /// Tokens reused from the prefix cache (NOT prefilled by this request).
    cached_token_count: u32,

    /// Full token sequence for the active request, in order. Used by
    /// `register_full_blocks_for_reuse` on completion.
    request_tokens: Vec<u32>,
}

impl PagedKVCacheAdapter {
    /// Construct a new adapter sharing the given allocator.
    ///
    /// `block_size` MUST equal `allocator.block_size()` — checked. Returns
    /// `Err` with a descriptive message on mismatch.
    pub fn new(allocator: Arc<Mutex<BlockAllocator>>, block_size: u32) -> Result<Self, String> {
        let allocator_block_size = {
            let guard = allocator
                .lock()
                .map_err(|e| format!("BlockAllocator mutex poisoned: {e}"))?;
            guard.block_size()
        };
        if block_size != allocator_block_size {
            return Err(format!(
                "block_size mismatch: adapter requested {block_size}, allocator has \
                 {allocator_block_size}"
            ));
        }
        Ok(Self {
            allocator,
            block_size,
            block_table: None,
            cached_token_count: 0,
            request_tokens: Vec::new(),
        })
    }

    /// Begin a new request. Releases any prior request's blocks first.
    /// `seq_id` is a logical request identifier (caller's choice; not
    /// interpreted by the adapter beyond passing it to `SequenceBlockTable`).
    pub fn reset_for_new_request(&mut self, seq_id: u32) -> Result<(), String> {
        // If there's a prior request, release its blocks first. We must NOT
        // silently leak; the prior caller forgot to call release_request.
        if self.block_table.is_some() {
            self.release_request()?;
        }
        self.block_table = Some(SequenceBlockTable::new(seq_id, self.block_size));
        self.cached_token_count = 0;
        self.request_tokens.clear();
        Ok(())
    }

    /// Look up the longest cached prefix matching `prompt_tokens` and
    /// populate the request's block_table with those blocks. Returns the
    /// cached prefix length so the caller knows where prefill must start.
    ///
    /// Calls `BlockAllocator::find_longest_cache_hit` which increments
    /// refcount on matched blocks. The adapter takes ownership (`Arc` clones)
    /// so subsequent `release_request()` correctly decrements.
    ///
    /// Idempotent in the sense that calling it twice with the same tokens
    /// returns the same prefix; but it expects to be called once per
    /// request, after `reset_for_new_request`.
    pub fn find_cached_prefix(
        &mut self,
        prompt_tokens: &[u32],
        extra_keys: &[u64],
    ) -> Result<CachedPrefix, String> {
        let block_table = self
            .block_table
            .as_mut()
            .ok_or_else(|| "find_cached_prefix called before reset_for_new_request".to_string())?;

        let (blocks, cached_tokens) = {
            let mut guard = self
                .allocator
                .lock()
                .map_err(|e| format!("BlockAllocator mutex poisoned: {e}"))?;
            guard.find_longest_cache_hit(prompt_tokens, self.block_size, extra_keys)
        };

        for block in &blocks {
            block_table.add_block(Arc::clone(block));
        }

        let cached_token_count = cached_tokens as u32;
        self.cached_token_count = cached_token_count;

        Ok(CachedPrefix {
            blocks,
            cached_token_count,
        })
    }

    /// Allocate enough new blocks to hold `total_tokens` tokens beyond
    /// the cached prefix. Appends them to the block_table. Returns the
    /// number of NEW blocks allocated.
    ///
    /// Errors if the allocator can't fulfil the request (no free blocks).
    /// On partial failure (some allocations succeeded before the pool ran
    /// out), the already-allocated blocks are rolled back into the pool to
    /// avoid leaks.
    pub fn allocate_suffix_blocks(&mut self, total_tokens: u32) -> Result<u32, String> {
        let block_table = self.block_table.as_mut().ok_or_else(|| {
            "allocate_suffix_blocks called before reset_for_new_request".to_string()
        })?;

        // Tokens that need fresh blocks = total_tokens - cached prefix tokens.
        let cached = self.cached_token_count;
        if total_tokens <= cached {
            return Ok(0);
        }
        let suffix_tokens = total_tokens - cached;
        let needed_blocks = suffix_tokens.div_ceil(self.block_size);
        if needed_blocks == 0 {
            return Ok(0);
        }

        let mut guard = self
            .allocator
            .lock()
            .map_err(|e| format!("BlockAllocator mutex poisoned: {e}"))?;

        let mut newly_allocated: Vec<Arc<PhysicalBlock>> =
            Vec::with_capacity(needed_blocks as usize);
        for i in 0..needed_blocks {
            match guard.allocate() {
                Some(block) => newly_allocated.push(block),
                None => {
                    // Roll back partial allocations to keep the pool consistent.
                    for partial in newly_allocated.drain(..) {
                        guard.free(partial);
                    }
                    return Err(format!(
                        "BlockAllocator exhausted: needed {needed_blocks} blocks, allocated {i} \
                         before running out"
                    ));
                }
            }
        }
        drop(guard);

        for block in newly_allocated {
            block_table.add_block(block);
        }
        Ok(needed_blocks)
    }

    /// Record tokens emitted/consumed in this request. Updates
    /// `request_tokens` and `block_table.num_tokens`. Caller passes
    /// EVERY token (prompt prefill + decoded output) in order.
    ///
    /// `tokens.len()` may be 1 (decode step) or N (full prefill batch).
    pub fn record_tokens(&mut self, tokens: &[u32]) -> Result<(), String> {
        let block_table = self
            .block_table
            .as_mut()
            .ok_or_else(|| "record_tokens called before reset_for_new_request".to_string())?;

        self.request_tokens.extend_from_slice(tokens);
        let new_total = self.request_tokens.len() as u32;
        block_table.set_num_tokens(new_total);
        Ok(())
    }

    /// Register the request's FULL blocks in the prefix cache so future
    /// requests with the same prompt prefix can reuse them. Call once
    /// per request, after generation finishes (success path only — do
    /// NOT call on error/abort).
    ///
    /// Only fully-formed blocks are registered (partial trailing block is
    /// not eligible). `extra_keys` is the same value the caller passed to
    /// `find_cached_prefix`; it MUST match for future cross-request
    /// reuse to work.
    ///
    /// Returns the number of blocks registered (i.e. the number of full
    /// blocks covered by `request_tokens`).
    ///
    /// ## Refcount semantics
    ///
    /// `BlockAllocator::register_prefix` itself does NOT bump the logical
    /// `ref_count` — its `prefix_cache` map holds an `Arc` clone, but the
    /// `Arc` strong count is separate from the `AtomicU32 ref_count` that
    /// `free()` decrements. So a freshly-allocated block at `ref_count = 1`
    /// (held only by the request) would be evicted from the prefix cache
    /// by `release_request()` — defeating cross-request reuse.
    ///
    /// To keep registered blocks alive across the owning request's
    /// `release_request()`, the adapter calls `incref()` on each
    /// **freshly-allocated** block before registering it. The first
    /// `cached_token_count / block_size` blocks of the table are
    /// **prefix-cache hits** (already incref'd by `lookup_prefix` during
    /// `find_cached_prefix`); registering them is a pure LRU refresh and
    /// must not double-incref or `release_request` will leave a permanent
    /// leak.
    ///
    /// After release the block lands at `ref_count = 1` (the prefix-cache
    /// reference), surviving until it is evicted by capacity pressure or
    /// the next owner frees the last reference.
    pub fn register_full_blocks_for_reuse(&mut self, extra_keys: &[u64]) -> Result<u32, String> {
        let block_table = self.block_table.as_ref().ok_or_else(|| {
            "register_full_blocks_for_reuse called before reset_for_new_request".to_string()
        })?;

        // Only count blocks fully covered by the recorded tokens; the
        // BlockAllocator caches per-block, so the trailing partial block
        // (if any) cannot be registered until it's filled.
        let block_size_us = self.block_size as usize;
        if block_size_us == 0 {
            return Err("block_size must be > 0".to_string());
        }
        let num_full_blocks = self.request_tokens.len() / block_size_us;
        if num_full_blocks == 0 {
            return Ok(0);
        }

        // Take only the first `num_full_blocks` from the table — there may
        // be a trailing under-filled block beyond this.
        let blocks_slice = &block_table.blocks()[..num_full_blocks.min(block_table.num_blocks())];
        let actual_blocks_to_register = blocks_slice.len();
        if actual_blocks_to_register == 0 {
            return Ok(0);
        }

        // The first `cached_blocks_count` blocks were reused from the prefix
        // cache (already incref'd by lookup_prefix during find_cached_prefix);
        // skip incref'ing them here. The remaining are freshly allocated by
        // `allocate_suffix_blocks` — those need a +1 to outlive the
        // owning request's release_request.
        let cached_blocks_count = (self.cached_token_count as usize) / block_size_us;
        for block in blocks_slice.iter().skip(cached_blocks_count) {
            block.incref();
        }

        let mut guard = self
            .allocator
            .lock()
            .map_err(|e| format!("BlockAllocator mutex poisoned: {e}"))?;

        guard
            .cache_full_blocks(
                &self.request_tokens[..actual_blocks_to_register * block_size_us],
                blocks_slice,
                self.block_size,
                extra_keys,
            )
            .map_err(|e| {
                // On failure, roll back the increfs to keep ref_count balanced.
                for block in blocks_slice.iter().skip(cached_blocks_count) {
                    let _ = block.decref();
                }
                format!("cache_full_blocks failed: {e}")
            })?;

        Ok(actual_blocks_to_register as u32)
    }

    /// Release this request's block references. Decrefs every block in
    /// the block_table. Blocks with refcount > 0 (still referenced by
    /// the prefix cache or another in-flight request) survive; blocks
    /// at refcount 0 return to the free pool.
    ///
    /// Call exactly once per request, in the cleanup path (success or
    /// failure). Subsequent operations on this adapter require
    /// `reset_for_new_request` first. Calling twice in a row is a no-op
    /// (second call sees `block_table == None`).
    ///
    /// Returns the number of block Arc references that were freed (i.e.
    /// the number of blocks in the table at release time).
    pub fn release_request(&mut self) -> Result<u32, String> {
        let Some(table) = self.block_table.take() else {
            return Ok(0);
        };

        let blocks = table.blocks().to_vec();
        let count = blocks.len() as u32;

        let mut guard = self
            .allocator
            .lock()
            .map_err(|e| format!("BlockAllocator mutex poisoned: {e}"))?;
        for block in blocks {
            guard.free(block);
        }
        drop(guard);

        self.cached_token_count = 0;
        self.request_tokens.clear();
        Ok(count)
    }

    // ------------------------ Getters ------------------------

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn cached_token_count(&self) -> u32 {
        self.cached_token_count
    }

    pub fn current_token_count(&self) -> u32 {
        self.request_tokens.len() as u32
    }

    pub fn num_allocated_blocks(&self) -> usize {
        self.block_table
            .as_ref()
            .map(|t| t.num_blocks())
            .unwrap_or(0)
    }

    pub fn block_table(&self) -> Option<&SequenceBlockTable> {
        self.block_table.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_allocator(num_blocks: u32, block_size: u32) -> Arc<Mutex<BlockAllocator>> {
        Arc::new(Mutex::new(BlockAllocator::new(num_blocks, block_size)))
    }

    /// Convenience: simulates a previous completed request that registered
    /// its blocks for cross-request reuse. Mirrors the combined effect of
    /// `register_full_blocks_for_reuse` followed by `release_request`:
    /// incref each block (so the prefix-cache reference outlives release),
    /// register, then free (drop request handle). After return, each block
    /// is at ref_count = 1 — the prefix-cache's long-lived logical reference.
    fn seed_prefix_cache(
        allocator: &Arc<Mutex<BlockAllocator>>,
        tokens: &[u32],
        block_size: u32,
        extra_keys: &[u64],
    ) {
        let mut guard = allocator.lock().unwrap();
        let block_size_us = block_size as usize;
        let num_full = tokens.len() / block_size_us;
        let mut blocks = Vec::with_capacity(num_full);
        for _ in 0..num_full {
            blocks.push(guard.allocate().expect("seed_prefix_cache: free block"));
        }
        // Incref each (mirror register_full_blocks_for_reuse) then register.
        for b in &blocks {
            b.incref();
        }
        guard
            .cache_full_blocks(tokens, &blocks, block_size, extra_keys)
            .expect("seed_prefix_cache: cache_full_blocks");
        // Now free each — decrefs from 2 → 1, blocks survive in prefix cache.
        for b in blocks {
            guard.free(b);
        }
    }

    #[test]
    fn test_new_validates_block_size() {
        let allocator = new_allocator(8, 4);
        let bad = PagedKVCacheAdapter::new(Arc::clone(&allocator), 8);
        assert!(bad.is_err(), "expected mismatch error, got Ok");
        let ok = PagedKVCacheAdapter::new(allocator, 4);
        assert!(ok.is_ok(), "expected Ok, got {:?}", ok.err());
    }

    #[test]
    fn test_reset_for_new_request_initializes_state() {
        let allocator = new_allocator(8, 4);
        let mut adapter = PagedKVCacheAdapter::new(allocator, 4).unwrap();
        adapter.reset_for_new_request(7).unwrap();
        let table = adapter.block_table().expect("block_table populated");
        assert_eq!(table.seq_id, 7);
        assert_eq!(table.num_blocks(), 0);
        assert_eq!(table.num_tokens(), 0);
        assert_eq!(adapter.cached_token_count(), 0);
        assert_eq!(adapter.current_token_count(), 0);
    }

    #[test]
    fn test_find_cached_prefix_miss() {
        let allocator = new_allocator(8, 4);
        let mut adapter = PagedKVCacheAdapter::new(allocator, 4).unwrap();
        adapter.reset_for_new_request(0).unwrap();
        let res = adapter.find_cached_prefix(&[1, 2, 3, 4, 5], &[]).unwrap();
        assert!(res.blocks.is_empty());
        assert_eq!(res.cached_token_count, 0);
        assert_eq!(adapter.block_table().unwrap().num_blocks(), 0);
        assert_eq!(adapter.cached_token_count(), 0);
    }

    #[test]
    fn test_find_cached_prefix_hit_after_register() {
        let allocator = new_allocator(8, 4);
        let tokens: Vec<u32> = (0..8).collect();
        seed_prefix_cache(&allocator, &tokens, 4, &[]);

        let mut adapter = PagedKVCacheAdapter::new(allocator, 4).unwrap();
        adapter.reset_for_new_request(1).unwrap();
        // Look up with same 8 tokens — should hit both blocks.
        let res = adapter.find_cached_prefix(&tokens, &[]).unwrap();
        assert_eq!(res.blocks.len(), 2);
        assert_eq!(res.cached_token_count, 8);
        assert_eq!(adapter.block_table().unwrap().num_blocks(), 2);
        // Each lookup increments refcount; seed left blocks at 1 (prefix-cache
        // reference). After lookup_prefix we expect ref_count == 2.
        for b in &res.blocks {
            assert_eq!(b.get_ref_count(), 2, "lookup must incref");
        }
    }

    #[test]
    fn test_allocate_suffix_blocks_no_prefix() {
        let allocator = new_allocator(8, 4);
        let mut adapter = PagedKVCacheAdapter::new(allocator, 4).unwrap();
        adapter.reset_for_new_request(0).unwrap();
        // 10 tokens, block_size=4 -> ceil(10/4) = 3 new blocks.
        let n = adapter.allocate_suffix_blocks(10).unwrap();
        assert_eq!(n, 3);
        assert_eq!(adapter.block_table().unwrap().num_blocks(), 3);
        // record_tokens not called yet, so num_tokens stays 0.
        assert_eq!(adapter.block_table().unwrap().num_tokens(), 0);
    }

    #[test]
    fn test_allocate_suffix_blocks_after_prefix() {
        let allocator = new_allocator(8, 4);
        let prefix_tokens: Vec<u32> = (0..8).collect();
        seed_prefix_cache(&allocator, &prefix_tokens, 4, &[]);

        let mut adapter = PagedKVCacheAdapter::new(allocator, 4).unwrap();
        adapter.reset_for_new_request(2).unwrap();
        let res = adapter.find_cached_prefix(&prefix_tokens, &[]).unwrap();
        assert_eq!(res.cached_token_count, 8);
        assert_eq!(res.blocks.len(), 2);

        // Want 13 total tokens; 8 already cached → 5 more → ceil(5/4) = 2 blocks.
        let n = adapter.allocate_suffix_blocks(13).unwrap();
        assert_eq!(n, 2);
        assert_eq!(adapter.block_table().unwrap().num_blocks(), 4);
    }

    #[test]
    fn test_record_tokens_appends_to_request_tokens_and_updates_num_tokens() {
        let allocator = new_allocator(8, 4);
        let mut adapter = PagedKVCacheAdapter::new(allocator, 4).unwrap();
        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(8).unwrap();

        adapter.record_tokens(&[10, 20, 30]).unwrap();
        assert_eq!(adapter.current_token_count(), 3);
        assert_eq!(adapter.block_table().unwrap().num_tokens(), 3);

        adapter.record_tokens(&[40]).unwrap();
        assert_eq!(adapter.current_token_count(), 4);
        assert_eq!(adapter.block_table().unwrap().num_tokens(), 4);
    }

    #[test]
    fn test_register_full_blocks_for_reuse_idempotent() {
        let allocator = new_allocator(8, 4);
        let mut adapter = PagedKVCacheAdapter::new(Arc::clone(&allocator), 4).unwrap();
        adapter.reset_for_new_request(0).unwrap();

        adapter.allocate_suffix_blocks(8).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let registered = adapter.register_full_blocks_for_reuse(&[]).unwrap();
        assert_eq!(registered, 2, "two full blocks of size 4 = 8 tokens");

        // Second adapter on the same allocator should now see the cached prefix.
        let mut adapter2 = PagedKVCacheAdapter::new(allocator, 4).unwrap();
        adapter2.reset_for_new_request(1).unwrap();
        let res = adapter2
            .find_cached_prefix(&[1, 2, 3, 4, 5, 6, 7, 8], &[])
            .unwrap();
        assert_eq!(res.cached_token_count, 8);
        assert_eq!(res.blocks.len(), 2);
    }

    #[test]
    fn test_release_request_decrefs_blocks() {
        let allocator = new_allocator(8, 4);
        let initial_free = allocator.lock().unwrap().num_free_blocks();
        let mut adapter = PagedKVCacheAdapter::new(Arc::clone(&allocator), 4).unwrap();

        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(8).unwrap(); // 2 blocks
        assert_eq!(
            allocator.lock().unwrap().num_free_blocks(),
            initial_free - 2
        );

        let freed = adapter.release_request().unwrap();
        assert_eq!(freed, 2);
        assert!(adapter.block_table().is_none());
        assert_eq!(allocator.lock().unwrap().num_free_blocks(), initial_free);

        // Calling twice is a no-op.
        let again = adapter.release_request().unwrap();
        assert_eq!(again, 0);
        assert_eq!(allocator.lock().unwrap().num_free_blocks(), initial_free);
    }

    #[test]
    fn test_register_then_release_keeps_blocks_alive_in_prefix_cache() {
        let allocator = new_allocator(8, 4);
        let mut adapter = PagedKVCacheAdapter::new(Arc::clone(&allocator), 4).unwrap();
        adapter.reset_for_new_request(0).unwrap();

        adapter.allocate_suffix_blocks(8).unwrap();
        let tokens: Vec<u32> = (10..18).collect();
        adapter.record_tokens(&tokens).unwrap();
        let registered = adapter.register_full_blocks_for_reuse(&[]).unwrap();
        assert_eq!(registered, 2);

        // Release this request. Because register_prefix doesn't bump the
        // refcount but DOES hold an Arc clone in prefix_cache, freeing the
        // request's reference brings refcount to 1 (held by the cache map +
        // the `allocated` map). The block survives prefix lookup.
        let freed = adapter.release_request().unwrap();
        assert_eq!(freed, 2);

        // A fresh adapter on the same allocator can resurrect the prefix.
        let mut adapter2 = PagedKVCacheAdapter::new(allocator, 4).unwrap();
        adapter2.reset_for_new_request(1).unwrap();
        let res = adapter2.find_cached_prefix(&tokens, &[]).unwrap();
        assert_eq!(
            res.cached_token_count, 8,
            "prefix cache must survive release_request"
        );
        assert_eq!(res.blocks.len(), 2);
    }

    #[test]
    fn test_two_adapters_share_prefix() {
        // Adapter A finishes a request whose first 8 tokens are SYS_8 and
        // next 4 are USER_A_4 → 3 full blocks. Register A's blocks, release.
        // Adapter B starts a new request with SYS_8 + USER_B_4. The first 2
        // blocks (SYS_8) are shared via the prefix cache; USER_B_4 differs
        // so block 3 is a miss.
        let allocator = new_allocator(16, 4);
        let sys_tokens: Vec<u32> = (1..=8).collect();
        let user_a_tokens: Vec<u32> = vec![100, 101, 102, 103];
        let user_b_tokens: Vec<u32> = vec![200, 201, 202, 203];

        let mut full_a = sys_tokens.clone();
        full_a.extend_from_slice(&user_a_tokens);
        let mut full_b = sys_tokens.clone();
        full_b.extend_from_slice(&user_b_tokens);

        // Adapter A.
        let mut adapter_a = PagedKVCacheAdapter::new(Arc::clone(&allocator), 4).unwrap();
        adapter_a.reset_for_new_request(0).unwrap();
        adapter_a
            .allocate_suffix_blocks(full_a.len() as u32)
            .unwrap();
        adapter_a.record_tokens(&full_a).unwrap();
        let reg_a = adapter_a.register_full_blocks_for_reuse(&[]).unwrap();
        assert_eq!(reg_a, 3);
        adapter_a.release_request().unwrap();

        // Adapter B.
        let mut adapter_b = PagedKVCacheAdapter::new(allocator, 4).unwrap();
        adapter_b.reset_for_new_request(1).unwrap();
        let res = adapter_b.find_cached_prefix(&full_b, &[]).unwrap();
        // SYS prefix shared (8 tokens / 2 blocks); USER_B differs → miss.
        assert_eq!(
            res.cached_token_count, 8,
            "shared SYS prefix must hit even when USER suffix differs"
        );
        assert_eq!(res.blocks.len(), 2);
    }
}
