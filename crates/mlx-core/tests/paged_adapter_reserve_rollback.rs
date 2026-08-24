//! Bookkeeping tests for the `PagedKVCacheAdapter` speculative seams:
//! lookahead row reservation (`reserve_rows` / `reserve_rows_for`) and
//! per-sequence rollback (`rollback_last_tokens_for`).
//!
//! - `reserve_rows` must allocate blocks for `current + extra` positions
//!   WITHOUT advancing the token cursor, so a later `record_tokens` for
//!   those rows takes the no-op allocation branch (vLLM's
//!   `num_lookahead_tokens` region in `allocate_slots`).
//! - `rollback_last_tokens_for` must truncate exactly the named sequence
//!   and leave every parked peer untouched.
//! - Allocator exhaustion inside a reservation must roll back the partial
//!   allocation atomically so the caller can fall back to plain
//!   autoregressive decode with unchanged state.
//!
//! Like `lazy_block_allocation.rs`, these use `LayerKVPool::new_for_test`
//! — pure bookkeeping, no KV writes — and skip cleanly when Metal is
//! unavailable.

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

/// Adapter plus a handle on its shared allocator so tests can assert on
/// pool-level free counts (the only observer that catches a leaked or
/// non-rolled-back partial allocation).
fn build_fixture(
    num_blocks: u32,
    block_size: u32,
) -> Option<(PagedKVCacheAdapter, Arc<Mutex<BlockAllocator>>)> {
    let pool = build_test_pool(num_blocks, block_size)?;
    let allocator = Arc::new(Mutex::new(BlockAllocator::new(
        num_blocks, num_blocks, block_size,
    )));
    let adapter =
        PagedKVCacheAdapter::new(Arc::clone(&allocator), pool, block_size).expect("adapter ctor");
    Some((adapter, allocator))
}

fn free_blocks(allocator: &Arc<Mutex<BlockAllocator>>) -> u32 {
    allocator.lock().unwrap().num_free_blocks()
}

#[test]
fn reserve_rows_does_not_advance_cursor() {
    let block_size: u32 = 16;
    let Some((mut adapter, allocator)) = build_fixture(64, block_size) else {
        eprintln!("skipping reserve_rows_does_not_advance_cursor: Metal unavailable");
        return;
    };

    adapter.reset_for_new_request(0).unwrap();
    let prompt: Vec<u32> = (1..=16).collect(); // exactly 1 block
    let prefix = adapter.find_cached_prefix(&prompt, &[], 0, false).unwrap();
    assert_eq!(prefix.cached_token_count, 0);
    adapter.allocate_suffix_blocks(prompt.len() as u32).unwrap();
    adapter.record_tokens(&prompt).unwrap();
    assert_eq!(adapter.current_token_count(), 16);
    assert_eq!(adapter.num_allocated_blocks(), 1);

    // Reserve a depth-3 verify cycle's rows (anchor + 3 drafts). The
    // cursor sits on a block boundary, so 4 lookahead rows need exactly
    // one new block.
    let free_before = free_blocks(&allocator);
    let newly_allocated = adapter.reserve_rows(4).expect("reserve_rows");
    assert_eq!(newly_allocated, 1);
    assert_eq!(adapter.num_allocated_blocks(), 2);
    assert_eq!(free_before - free_blocks(&allocator), 1);

    // The reservation must NOT advance either cursor surface.
    assert_eq!(adapter.current_token_count(), 16);
    assert_eq!(adapter.block_table_for(0).unwrap().num_tokens(), 16);

    // Recording the reserved rows takes the no-op allocation branch:
    // zero new blocks leave the pool.
    let free_after_reserve = free_blocks(&allocator);
    for token in [100u32, 101, 102, 103] {
        adapter.record_tokens(&[token]).unwrap();
    }
    assert_eq!(adapter.current_token_count(), 20);
    assert_eq!(adapter.block_table_for(0).unwrap().num_tokens(), 20);
    assert_eq!(adapter.num_allocated_blocks(), 2);
    assert_eq!(
        free_blocks(&allocator),
        free_after_reserve,
        "recording into reserved rows must not touch the allocator"
    );

    // Re-reserving already-covered rows is a no-op.
    assert_eq!(adapter.reserve_rows(4).unwrap(), 0);
    assert_eq!(free_blocks(&allocator), free_after_reserve);
}

#[test]
fn rollback_last_tokens_for_targets_named_seq() {
    let block_size: u32 = 4;
    let Some((mut adapter, allocator)) = build_fixture(64, block_size) else {
        eprintln!("skipping rollback_last_tokens_for_targets_named_seq: Metal unavailable");
        return;
    };

    adapter.begin_request(1).unwrap();
    for i in 0..6u32 {
        adapter.record_token_for(1, 100 + i).unwrap();
    }
    adapter.begin_request(2).unwrap();
    for i in 0..5u32 {
        adapter.record_token_for(2, 200 + i).unwrap();
    }
    let seq2_tokens_before: Vec<u32> = adapter.request_tokens_for(2).unwrap().to_vec();
    assert_eq!(seq2_tokens_before.len(), 5);

    // Roll back two tokens of seq 1 while seq 2 is the active row.
    let free_before = free_blocks(&allocator);
    adapter.rollback_last_tokens_for(1, 2).unwrap();

    assert_eq!(adapter.current_token_count_for(1), Some(4));
    assert_eq!(
        adapter.request_tokens_for(1).unwrap(),
        &[100, 101, 102, 103]
    );
    assert_eq!(adapter.block_table_for(1).unwrap().num_tokens(), 4);

    // The peer sequence is untouched on every surface.
    assert_eq!(adapter.current_token_count_for(2), Some(5));
    assert_eq!(adapter.request_tokens_for(2).unwrap(), &seq2_tokens_before);
    assert_eq!(adapter.block_table_for(2).unwrap().num_tokens(), 5);

    // Bookkeeping-only rollback: no block was freed, and the vacated
    // slots are overwritten in place with zero new allocation.
    assert_eq!(adapter.block_table_for(1).unwrap().num_blocks(), 2);
    assert_eq!(free_blocks(&allocator), free_before);
    adapter.record_token_for(1, 300).unwrap();
    assert_eq!(adapter.current_token_count_for(1), Some(5));
    assert_eq!(free_blocks(&allocator), free_before);
}

#[test]
fn reserve_rows_exhaustion_rolls_back_partial_allocation() {
    let block_size: u32 = 4;
    // 4-block pool → 16 token slots of physical capacity.
    let Some((mut adapter, allocator)) = build_fixture(4, block_size) else {
        eprintln!(
            "skipping reserve_rows_exhaustion_rolls_back_partial_allocation: Metal unavailable"
        );
        return;
    };

    adapter.begin_request(1).unwrap();
    for i in 0..8u32 {
        adapter.record_token_for(1, 100 + i).unwrap();
    }
    adapter.begin_request(2).unwrap();
    for i in 0..4u32 {
        adapter.record_token_for(2, 200 + i).unwrap();
    }
    assert_eq!(free_blocks(&allocator), 1);

    // Seq 2 asks for 12 lookahead rows = 3 more blocks. new_total (16)
    // passes the per-request capacity check, but the shared pool has only
    // one free block: the reservation must fail AFTER allocating one
    // block and roll that partial allocation back atomically.
    let err = adapter.reserve_rows_for(2, 12).unwrap_err();
    assert!(
        err.starts_with("context_length_exceeded:"),
        "expected canonical exhaustion error, got: {err}"
    );
    assert_eq!(
        free_blocks(&allocator),
        1,
        "partial allocation must be returned to the pool"
    );
    assert_eq!(adapter.block_table_for(2).unwrap().num_blocks(), 1);
    assert_eq!(adapter.block_table_for(2).unwrap().num_tokens(), 4);
    assert_eq!(adapter.current_token_count_for(2), Some(4));

    // The peer sequence is untouched.
    assert_eq!(adapter.current_token_count_for(1), Some(8));
    assert_eq!(adapter.block_table_for(1).unwrap().num_blocks(), 2);

    // A reservation that fits still succeeds afterwards and covers the
    // subsequent decode writes without touching the allocator again.
    assert_eq!(adapter.reserve_rows_for(2, 4).unwrap(), 1);
    assert_eq!(free_blocks(&allocator), 0);
    for i in 0..4u32 {
        adapter.record_token_for(2, 300 + i).unwrap();
    }
    assert_eq!(adapter.current_token_count_for(2), Some(8));
    assert_eq!(free_blocks(&allocator), 0);
}
