//! Prefix-cache hit-walk benchmark.
//!
//! A long shared prefix (e.g. a 4096-token system prompt at block_size=16
//! -> 256 blocks) is re-walked by every new request that shares it, while
//! `num_blocks` background entries (other concurrent requests' cached
//! blocks) occupy the rest of the prefix cache. A walk stamps only its tip,
//! so the per-walk cost should not scale with `num_blocks`.
//!
//! Run with: cargo bench --package mlx-paged-attn --bench prefix_cache_hit_walk_bench

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mlx_paged_attn::BlockAllocator;
use std::hint::black_box;

const BLOCK_SIZE: u32 = 16;
/// 4096 tokens at block_size=16, matching the backlog item's example.
const PREFIX_BLOCKS: usize = 256;

/// Build a `BlockAllocator` whose prefix cache holds exactly `num_blocks`
/// entries: `PREFIX_BLOCKS` of them belonging to one long shared prefix
/// (registered last, so it holds the newest stamps at the start of the
/// benchmark), and the rest distinct
/// single-block "background" prefixes standing in for other concurrent
/// requests' cached blocks.
///
/// Returns the allocator and the shared prefix's token_ids so the
/// benchmark closure can re-walk it via `find_longest_cache_hit`.
fn build_populated_allocator(num_blocks: usize) -> (BlockAllocator, Vec<u32>) {
    assert!(
        num_blocks > PREFIX_BLOCKS,
        "num_blocks must exceed the shared-prefix block count"
    );
    let mut allocator = BlockAllocator::new(num_blocks as u32, num_blocks as u32, BLOCK_SIZE);

    let filler_count = num_blocks - PREFIX_BLOCKS;
    for i in 0..filler_count {
        let block = allocator.allocate().expect("pool sized for num_blocks");
        // Distinct single-block token content per filler entry so its hash
        // can't collide with the shared prefix or with other fillers.
        let tokens: Vec<u32> = (0..BLOCK_SIZE)
            .map(|t| {
                0x1000_0000u32
                    .wrapping_add((i as u32).wrapping_mul(1000))
                    .wrapping_add(t)
            })
            .collect();
        allocator
            .cache_full_blocks(&tokens, std::slice::from_ref(&block), BLOCK_SIZE, &[], 0)
            .expect("filler registration must succeed");
    }

    // The long shared prefix: PREFIX_BLOCKS blocks of small, distinct
    // token content (disjoint range from the filler tokens above).
    let prefix_tokens: Vec<u32> = (0..(PREFIX_BLOCKS as u32 * BLOCK_SIZE)).collect();
    let prefix_blocks: Vec<_> = (0..PREFIX_BLOCKS)
        .map(|_| allocator.allocate().expect("pool sized for num_blocks"))
        .collect();
    let registered = allocator
        .cache_full_blocks(&prefix_tokens, &prefix_blocks, BLOCK_SIZE, &[], 0)
        .expect("shared-prefix registration must succeed");
    assert_eq!(
        registered, PREFIX_BLOCKS,
        "whole shared prefix must register"
    );

    (allocator, prefix_tokens)
}

/// Simulates one new request walking in and hitting the full shared
/// prefix: `find_longest_cache_hit` looks up all `PREFIX_BLOCKS` blocks in
/// order and stamps only the last. `num_blocks` (500/2000/8000)
/// stands in for realistic prefix-cache population sizes (bigger
/// `gpu_memory_mb`, or an int8 pool's near-double block count).
fn shared_prefix_cache_hit_walk(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_cache_hit_walk");
    for &num_blocks in &[500usize, 2000, 8000] {
        let (mut allocator, prefix_tokens) = build_populated_allocator(num_blocks);
        group.bench_with_input(
            BenchmarkId::from_parameter(num_blocks),
            &num_blocks,
            |b, _| {
                b.iter(|| {
                    let (blocks, cached_tokens) = allocator.find_longest_cache_hit(
                        black_box(&prefix_tokens),
                        BLOCK_SIZE,
                        &[],
                        0,
                    );
                    assert_eq!(cached_tokens, prefix_tokens.len());
                    black_box(blocks);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, shared_prefix_cache_hit_walk);
criterion_main!(benches);
