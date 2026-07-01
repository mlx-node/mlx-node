//! Prefix-cache LRU touch benchmark.
//!
//! Reproduces the scenario from the "BlockAllocator LRU order O(n) touch"
//! perf backlog item: a long shared prefix (e.g. a 4096-token system prompt
//! at block_size=16 -> 256 blocks) re-touched by every new request that
//! shares it, while `num_blocks` background entries (other concurrent
//! requests' cached blocks) occupy the rest of the prefix cache.
//!
//! Before the fix, `lookup_prefix`'s LRU touch was `VecDeque::retain()` +
//! `push_back()` -- O(current lru_order length) per touched block, so one
//! 256-block cache-hit walk costs O(256 * num_blocks). After the fix it's
//! an O(1)-amortized `LinkedHashSet::insert()` per touched block, so the
//! per-walk cost should stop scaling with `num_blocks`.
//!
//! Run with: cargo bench --package mlx-paged-attn --bench prefix_cache_lru_bench

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mlx_paged_attn::BlockAllocator;
use std::hint::black_box;

const BLOCK_SIZE: u32 = 16;
/// 4096 tokens at block_size=16, matching the backlog item's example.
const PREFIX_BLOCKS: usize = 256;

/// Build a `BlockAllocator` whose prefix cache holds exactly `num_blocks`
/// entries: `PREFIX_BLOCKS` of them belonging to one long shared prefix
/// (registered last, so it's the most-recently-touched segment of LRU
/// order at the start of the benchmark), and the rest distinct
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
    let mut allocator = BlockAllocator::new(num_blocks as u32, BLOCK_SIZE);

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
/// order, LRU-touching every one of them. `num_blocks` (500/2000/8000)
/// stands in for realistic prefix-cache population sizes (bigger
/// `gpu_memory_mb`, or FP8 cache doubling block count).
fn shared_prefix_cache_hit_walk(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_cache_lru_touch");
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
