//! Multimodal cache-isolation integration tests for the paged
//! adapter's per-block extra_keys API.
//!
//! Pinpoints the load-bearing property: two requests with identical text
//! tokens but different image content MUST produce distinct cache
//! identities so the second request does NOT silently reuse the first
//! request's image-conditioned KV state. Conversely, two requests with
//! identical text AND identical images MUST hit the cache.
//!
//! These tests exercise the production code path:
//!
//! 1. `compute_per_block_image_extra_keys(token_image_positions, num_blocks,
//!    block_size)` — the helper VLM model code uses to build per-block
//!    extra_keys.
//! 2. `PagedKVCacheAdapter::find_cached_prefix_per_block` and
//!    `register_full_blocks_for_reuse_per_block` — the per-block API the
//!    Qwen3.5 paged dispatcher (and any future VLM-paged caller) goes
//!    through.
//!
//! Pure-CPU bookkeeping checks (no GPU dispatch). The fixture goes through
//! `LayerKVPool::new_for_validation_only`, which never touches the Metal
//! device, so these tests run identically on Metal hosts and on sandboxed
//! CI VMs without a Metal device. `new_for_test` must NOT be used here: it
//! calls `MetalState::get` and silently early-returns `None` on non-Metal
//! hosts — producing green test runs that never exercise the load-bearing
//! adapter code.

use std::sync::{Arc, Mutex};

use mlx_core::transformer::paged_kv_cache_adapter::{
    PagedKVCacheAdapter, compute_per_block_image_extra_keys,
};
use mlx_paged_attn::{BlockAllocator, LayerKVPool, PagedAttentionConfig, metal::MetalDtype};

const NUM_BLOCKS: u32 = 32;
const BLOCK_SIZE: u32 = 8;

/// Build a placeholder pool + adapter pair. Uses
/// `LayerKVPool::new_for_validation_only` so construction succeeds on
/// every platform (Metal or not) — the load-bearing
/// `PagedKVCacheAdapter::find_cached_prefix_per_block` /
/// `register_full_blocks_for_reuse_per_block` paths these tests exercise
/// only touch the `BlockAllocator` (pure CPU) and the adapter
/// constructor's block_size/num_blocks validation, which only reads
/// `LayerKVPool::block_size` / `num_blocks`. **No GPU dispatch occurs.**
fn build_adapter() -> PagedKVCacheAdapter {
    let cfg = PagedAttentionConfig {
        block_size: BLOCK_SIZE,
        num_kv_heads: 1,
        head_size: 32,
        num_layers: 2,
        ..PagedAttentionConfig::default()
    };
    let pool = LayerKVPool::new_for_validation_only(cfg, NUM_BLOCKS, 2, MetalDtype::Float16)
        .expect("new_for_validation_only must succeed on every platform");
    let allocator = Arc::new(Mutex::new(BlockAllocator::new(NUM_BLOCKS, BLOCK_SIZE)));
    PagedKVCacheAdapter::new(allocator, Arc::new(pool), BLOCK_SIZE).expect("adapter ctor")
}

/// Drive a request through the per-block paged lifecycle: reset → cached-
/// prefix lookup (with image-aware per-block extra_keys) → suffix
/// allocation → record tokens → register for reuse → release.
///
/// Returns the cached_token_count from `find_cached_prefix_per_block` so
/// callers can assert hit/miss outcomes.
fn run_request(
    adapter: &mut PagedKVCacheAdapter,
    seq_id: u32,
    tokens: &[u32],
    token_image_positions: &[(u32, u64)],
    register_for_reuse: bool,
) -> u32 {
    adapter.reset_for_new_request(seq_id).expect("reset");
    let num_blocks = tokens.len().div_ceil(BLOCK_SIZE as usize);
    let per_block =
        compute_per_block_image_extra_keys(token_image_positions, num_blocks, BLOCK_SIZE);
    let prefix = adapter
        .find_cached_prefix_per_block(tokens, &per_block, 0, false)
        .expect("find_cached_prefix_per_block");
    let cached = prefix.cached_token_count;
    adapter
        .allocate_suffix_blocks(tokens.len() as u32)
        .expect("allocate_suffix_blocks");
    let cached_us = cached as usize;
    if cached_us < tokens.len() {
        adapter
            .record_tokens(&tokens[cached_us..])
            .expect("record_tokens (suffix)");
    }
    if register_for_reuse {
        adapter
            .register_full_blocks_for_reuse_per_block(&per_block, 0)
            .expect("register_full_blocks_for_reuse_per_block");
    }
    adapter.release_request().expect("release_request");
    cached
}

/// Two requests with the SAME text and SAME image produce a cache hit.
/// The cross-conversation reuse half — the load-bearing
/// "same prompt + same image → block reuse" property.
#[test]
fn same_text_same_image_hits_cache() {
    let mut adapter = build_adapter();

    // 16 tokens = 2 full blocks. Image positions 4..8 are entirely inside
    // block 0 (which carries the image hash); block 1 is text-only.
    let tokens: Vec<u32> = (1..=16).collect();
    let image_a_positions: Vec<(u32, u64)> = (4u32..8).map(|p| (p, 0xAAAA_AAAA)).collect();

    // Request 1: register the blocks under image_a's per-block keys.
    let cached_first = run_request(&mut adapter, 0, &tokens, &image_a_positions, true);
    assert_eq!(cached_first, 0, "first request must miss the empty cache");

    // Request 2: same text + same image → must hit both blocks.
    let cached_second = run_request(&mut adapter, 1, &tokens, &image_a_positions, false);
    assert_eq!(
        cached_second, 16,
        "second request with identical (tokens, image) must hit the full prefix; \
         got {cached_second} cached tokens out of 16"
    );
}

/// Two requests with the SAME text but DIFFERENT images MUST miss the
/// cache. The load-bearing isolation half — preventing
/// stale-image KV state from being silently reused for a request bearing
/// a different image.
#[test]
fn same_text_different_image_misses_cache() {
    let mut adapter = build_adapter();

    let tokens: Vec<u32> = (1..=16).collect();
    let image_a_positions: Vec<(u32, u64)> = (4u32..8).map(|p| (p, 0xAAAA_AAAA)).collect();
    let image_b_positions: Vec<(u32, u64)> = (4u32..8).map(|p| (p, 0xBBBB_BBBB)).collect();

    // Request 1: register under image A.
    run_request(&mut adapter, 0, &tokens, &image_a_positions, true);

    // Request 2: same text + different image → block 0's hash differs, so
    // the chain breaks at block 0 → 0 cached tokens.
    let cached = run_request(&mut adapter, 1, &tokens, &image_b_positions, false);
    assert_eq!(
        cached, 0,
        "different image hash on a block-spanning image MUST produce a cache miss; \
         got {cached} cached tokens (would be a stale-image KV reuse)"
    );

    // Request 3 sanity: same text + image A again → still hits.
    let cached_a_again = run_request(&mut adapter, 2, &tokens, &image_a_positions, false);
    assert_eq!(
        cached_a_again, 16,
        "image A's cached blocks must still resolve under the original keys"
    );
}

/// Image differences ONLY on a non-leading block isolate at the
/// divergence point. Blocks before the mismatch hit; blocks at and after
/// are misses.
#[test]
fn image_diff_on_later_block_isolates_at_divergence() {
    let mut adapter = build_adapter();

    // 24 tokens = 3 full blocks. Image positions 16..20 are entirely
    // inside block 2.
    let tokens: Vec<u32> = (1..=24).collect();
    let image_a_positions: Vec<(u32, u64)> = (16u32..20).map(|p| (p, 0xAAAA_AAAA)).collect();
    let image_b_positions: Vec<(u32, u64)> = (16u32..20).map(|p| (p, 0xBBBB_BBBB)).collect();

    run_request(&mut adapter, 0, &tokens, &image_a_positions, true);

    let cached_b = run_request(&mut adapter, 1, &tokens, &image_b_positions, false);
    assert_eq!(
        cached_b,
        2 * BLOCK_SIZE,
        "blocks 0+1 share text-only hashes (no image positions in those blocks); \
         block 2's image hash differs so the chain breaks there. Expected {} cached \
         tokens, got {cached_b}",
        2 * BLOCK_SIZE,
    );
}

/// Per-block API with empty image positions (text-only baseline) is
/// bit-equal to the uniform API with `&[]`. This pins the migration
/// invariant: callers that swap the uniform API for the per-block API
/// with `compute_per_block_image_extra_keys(&[], ...)` must NOT
/// invalidate their pre-existing cache entries.
#[test]
fn per_block_empty_matches_uniform_for_text_only() {
    let mut adapter = build_adapter();

    let tokens: Vec<u32> = (1..=16).collect();
    let num_blocks = tokens.len() / BLOCK_SIZE as usize;
    let empty_per_block: Vec<Vec<u64>> = (0..num_blocks).map(|_| Vec::new()).collect();

    // Register via the uniform API.
    adapter.reset_for_new_request(0).unwrap();
    adapter
        .find_cached_prefix(&tokens, &[], 0, false)
        .expect("uniform find_cached_prefix");
    adapter
        .allocate_suffix_blocks(tokens.len() as u32)
        .expect("allocate_suffix_blocks");
    adapter.record_tokens(&tokens).expect("record_tokens");
    adapter
        .register_full_blocks_for_reuse(&[], 0)
        .expect("uniform register");
    adapter.release_request().expect("release_request");

    // Look up via the per-block API with all-empty per-block keys → must
    // hit the same blocks the uniform path registered.
    adapter.reset_for_new_request(1).unwrap();
    let prefix = adapter
        .find_cached_prefix_per_block(&tokens, &empty_per_block, 0, false)
        .expect("per-block find_cached_prefix");
    assert_eq!(
        prefix.cached_token_count,
        tokens.len() as u32,
        "per-block API with empty positions must hit blocks registered via the \
         uniform API (compatibility invariant for text-only callers migrating to \
         the per-block API)"
    );
    adapter.release_request().expect("release");
}

/// Per-block helper output for an empty image-position list is functionally
/// equivalent to passing `&[]` uniformly across all blocks. This is the CPU-
/// only contract test that the helper itself respects the text-only
/// baseline (separate from the adapter's runtime equivalence above).
#[test]
fn helper_empty_positions_yields_all_empty_per_block() {
    let num_blocks = 4;
    let per_block = compute_per_block_image_extra_keys(&[], num_blocks, BLOCK_SIZE);
    assert_eq!(per_block.len(), num_blocks);
    for entry in &per_block {
        assert!(entry.is_empty());
    }
}
