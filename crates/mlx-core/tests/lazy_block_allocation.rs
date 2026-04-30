//! Lazy block allocation regression tests for `PagedKVCacheAdapter`.
//!
//! Pre-fix bug: `allocate_suffix_blocks(prompt_len + max_new_tokens)`
//! pre-reserved blocks for the speculative `max_new_tokens` budget upfront.
//! Claude Code routinely sends `max_tokens=128000` (~9750 blocks at the default
//! `block_size=16`) even when actual generation is far less, blowing out the
//! shared `BlockAllocator` pool: `BlockAllocator exhausted: needed 9943
//! blocks, allocated 6553 before running out`.
//!
//! Post-fix behavior (matches vLLM's `kv_cache_manager.py`):
//! - `allocate_suffix_blocks` is called with `prompt_tokens.len()` only.
//! - `record_tokens` lazily allocates one block at a time as decode crosses
//!   block boundaries.
//! - Pool exhaustion mid-decode surfaces a clean error and leaves
//!   caller-visible state unchanged.
//!
//! These tests use `BlockAllocator` directly via the `LayerKVPool::new_for_test`
//! shim — no Metal device required for the bookkeeping checks. The pool tests
//! still skip cleanly when Metal is unavailable.

use std::sync::{Arc, Mutex};

use mlx_core::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use mlx_paged_attn::{BlockAllocator, LayerKVPool, PagedAttentionConfig, metal::MetalDtype};

fn build_test_pool(num_blocks: u32, block_size: u32) -> Option<Arc<LayerKVPool>> {
    let cfg = PagedAttentionConfig {
        block_size,
        num_kv_heads: 1,
        head_size: 32,
        num_layers: 2,
        ..PagedAttentionConfig::default()
    };
    match LayerKVPool::new_for_test(cfg, num_blocks, 2, MetalDtype::Float16) {
        Ok(p) => Some(Arc::new(p)),
        Err(e) if e.contains("No Metal device found") => None,
        Err(e) => panic!("unexpected new_for_test failure: {e}"),
    }
}

fn build_adapter(num_blocks: u32, block_size: u32) -> Option<PagedKVCacheAdapter> {
    let pool = build_test_pool(num_blocks, block_size)?;
    let allocator = Arc::new(Mutex::new(BlockAllocator::new(num_blocks, block_size)));
    Some(PagedKVCacheAdapter::new(allocator, pool, block_size).expect("adapter ctor"))
}

/// Repro of the pool-exhaustion scenario the bug reproduced in production:
/// with a small pool (128 blocks) and a tiny prompt (16 tokens = 1 block),
/// recording 200 decode tokens one-by-one must NOT pre-allocate the full
/// `max_new_tokens` budget. Total blocks needed = ceil((16 + 200) / 16) = 14
/// blocks, far below the 128-block pool capacity.
#[test]
fn lazy_alloc_decode_does_not_exhaust_small_pool() {
    let block_size: u32 = 16;
    let pool_blocks: u32 = 128;
    let Some(mut adapter) = build_adapter(pool_blocks, block_size) else {
        eprintln!("skipping lazy_alloc_decode_does_not_exhaust_small_pool: Metal unavailable");
        return;
    };

    adapter.reset_for_new_request(0).unwrap();

    // Step 1: cold prefix lookup (miss) + suffix-block allocation.
    let prompt: Vec<u32> = (1..=16).collect(); // 1 block
    let prefix = adapter.find_cached_prefix(&prompt, &[], 0, false).unwrap();
    assert_eq!(prefix.cached_token_count, 0);

    // CRITICAL: pass prompt-only length, NOT prompt + max_new_tokens.
    // After this call we should have exactly the prompt's blocks reserved
    // and NOTHING for the speculative decode budget.
    let allocated = adapter
        .allocate_suffix_blocks(prompt.len() as u32)
        .expect("allocate_suffix_blocks");
    assert_eq!(
        allocated, 1,
        "prompt of 16 tokens should reserve exactly 1 block (got {allocated})"
    );
    assert_eq!(
        adapter.num_allocated_blocks(),
        1,
        "after allocate_suffix_blocks(16) only the prompt's block should be reserved \
         (no max_new_tokens pre-reserve)"
    );

    // Step 2: prefill records all 16 prompt tokens at once.
    adapter.record_tokens(&prompt).unwrap();
    assert_eq!(adapter.current_token_count(), 16);
    assert_eq!(adapter.num_allocated_blocks(), 1);

    // Step 3: decode loop records 200 tokens one at a time. Each block
    // boundary crossing should trigger a single lazy block allocation.
    let decode_count: u32 = 200;
    for i in 0..decode_count {
        let token = 1000 + i;
        adapter
            .record_tokens(&[token])
            .unwrap_or_else(|e| panic!("record_tokens at decode step {i} failed: {e}"));
    }

    // After 16 prompt + 200 decode = 216 tokens, expected block count is
    // ceil(216 / 16) = 14 blocks. Plenty of headroom in the 128-block pool.
    let expected_blocks = (16 + decode_count).div_ceil(block_size) as usize;
    assert_eq!(
        adapter.num_allocated_blocks(),
        expected_blocks,
        "expected {expected_blocks} blocks after 16 prompt + {decode_count} decode tokens"
    );
    assert_eq!(adapter.current_token_count(), 16 + decode_count);
    assert!(
        adapter.num_allocated_blocks() < pool_blocks as usize,
        "lazy alloc must stay well below pool capacity"
    );
}

/// Direct invariant check tied to the production bug: with the equivalent of
/// `paged_cache_memory_mb=2048` (which yields ~6553 blocks at the default
/// dtype × head config) and `max_tokens=128000` on a 16-token prompt, the
/// adapter must NOT pre-reserve the speculative budget. Only the prompt's
/// 1 block should be present immediately after `allocate_suffix_blocks(16)`.
#[test]
fn allocate_suffix_blocks_does_not_pre_reserve_max_new_tokens() {
    let block_size: u32 = 16;
    // Generously sized pool — the assertion is about what's allocated,
    // not what the pool could have given.
    let Some(mut adapter) = build_adapter(256, block_size) else {
        eprintln!(
            "skipping allocate_suffix_blocks_does_not_pre_reserve_max_new_tokens: Metal unavailable"
        );
        return;
    };

    adapter.reset_for_new_request(0).unwrap();
    let prompt: Vec<u32> = (1..=16).collect();
    let _ = adapter.find_cached_prefix(&prompt, &[], 0, false).unwrap();
    adapter
        .allocate_suffix_blocks(prompt.len() as u32)
        .expect("allocate_suffix_blocks");

    // The whole point of the fix: ONE block reserved for the prompt; no
    // max_new_tokens pre-reserve.
    assert_eq!(
        adapter.num_allocated_blocks(),
        1,
        "allocate_suffix_blocks(16) must reserve exactly 1 block (the prompt's), NOT \
         pre-allocate the speculative max_new_tokens budget"
    );
}

/// `record_tokens` must surface a clean error when lazy allocation fails
/// mid-decode (pool truly exhausted) and leave caller-visible state
/// unchanged so the model can stop generation gracefully without writing
/// to a non-existent block.
#[test]
fn record_tokens_lazy_alloc_propagates_pool_exhaustion() {
    let block_size: u32 = 4;
    // 2-block pool → 8 token slots total.
    let Some(mut adapter) = build_adapter(2, block_size) else {
        eprintln!(
            "skipping record_tokens_lazy_alloc_propagates_pool_exhaustion: Metal unavailable"
        );
        return;
    };
    adapter.reset_for_new_request(0).unwrap();
    // Reserve 1 block for the prompt suffix.
    adapter.allocate_suffix_blocks(4).unwrap();
    // Fill block 0.
    adapter.record_tokens(&[1, 2, 3, 4]).unwrap();
    // Crossing into block 1 (token 5) lazily allocates the second (and
    // last) pool block.
    adapter.record_tokens(&[5]).unwrap();
    assert_eq!(adapter.num_allocated_blocks(), 2);
    // Fill block 1.
    adapter.record_tokens(&[6, 7, 8]).unwrap();
    assert_eq!(adapter.current_token_count(), 8);
    assert_eq!(adapter.num_allocated_blocks(), 2);

    // 9th token would require a 3rd block; pool is exhausted.
    let prior_count = adapter.current_token_count();
    let prior_blocks = adapter.num_allocated_blocks();
    let res = adapter.record_tokens(&[9]);
    assert!(res.is_err(), "expected pool-exhaustion error");
    let msg = res.err().unwrap();
    assert!(
        msg.contains("BlockAllocator exhausted") || msg.contains("running out"),
        "error must indicate allocator exhaustion, got: {msg}"
    );
    // State must be unchanged so the model can abort cleanly without
    // writing to a non-existent block.
    assert_eq!(adapter.current_token_count(), prior_count);
    assert_eq!(adapter.num_allocated_blocks(), prior_blocks);
}
