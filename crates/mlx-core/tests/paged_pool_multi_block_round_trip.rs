//! Multi-block paged-pool round-trip diagnostic for the Gemma4 paged-decode bug.
//!
//! This test reproduces the exact LayerKVPool config used by the Gemma4 e2b paged
//! adapter (head_size=512, num_kv_heads=1, block_size=16, BFloat16) and:
//!
//! 1. Writes 16 tokens at logical position 0 (fills block 0).
//! 2. Writes 1 token at logical position 16 (start of block 1) — same pattern as
//!    the first decode step on a 16-token prompt.
//! 3. Reads `read_kv_range(0, 0, 17)` and verifies that:
//!    - The first 16 positions return the original block-0 values.
//!    - Position 16 returns the just-written block-1 values.
//!
//! Greedy parity uses the host-side gather (`read_kv_range` + SDPA) for decode,
//! so a single-layer multi-block round-trip with a fresh write into block 1 is
//! the smallest reproduction of the Gemma4 decode pattern.

use std::sync::{Arc, Mutex};

use mlx_core::array::{DType, MxArray};
use mlx_core::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use mlx_paged_attn::{BlockAllocator, LayerKVPool, PagedAttentionConfig, metal::MetalDtype};

/// Convert an f32 to BF16 bits via the same bit-truncation MLX uses.
fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    // Round-to-nearest-even truncation. For positive non-NaN small values we can
    // get away with simple truncation, which matches MLX's lossy cast for these
    // diagnostic values.
    (bits >> 16) as u16
}

#[test]
fn multi_block_round_trip_bf16_head_dim_512() {
    let cfg = PagedAttentionConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_size: 512,
        num_layers: 2,
        gpu_memory_mb: 256,
        use_fp8_cache: Some(false),
        max_seq_len: Some(64),
        max_batch_size: Some(2),
    };
    let pool = match LayerKVPool::new(cfg.clone(), 8, MetalDtype::BFloat16) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            eprintln!("skipping multi_block_round_trip_bf16_head_dim_512: {e}");
            return;
        }
    };
    let allocator = Arc::new(Mutex::new(BlockAllocator::new(8, 16)));
    let mut adapter = PagedKVCacheAdapter::new(allocator, pool, 16).expect("adapter");
    adapter.reset_for_new_request(0).unwrap();
    adapter.allocate_suffix_blocks(32).unwrap();

    // Step 1: write 16 tokens at position 0. K = 0.5, V = 1.0 (all positions and
    // dims).
    let prefix_tokens: Vec<u32> = (1..=16).collect();
    adapter.record_tokens(&prefix_tokens).unwrap();
    let k_prefix_bits = vec![f32_to_bf16_bits(0.5_f32); 16 * 512];
    let v_prefix_bits = vec![f32_to_bf16_bits(1.0_f32); 16 * 512];
    let k_prefix = MxArray::from_bfloat16(&k_prefix_bits, &[16, 1, 512]).unwrap();
    let v_prefix = MxArray::from_bfloat16(&v_prefix_bits, &[16, 1, 512]).unwrap();
    k_prefix.eval();
    v_prefix.eval();
    match adapter.update_keys_values(0, &k_prefix, &v_prefix, 0) {
        Ok(()) => {}
        Err(e) if e.contains("Metal GPU not available") => {
            eprintln!("skipping multi_block_round_trip_bf16_head_dim_512 (prefix): {e}");
            return;
        }
        Err(e) => panic!("update_keys_values prefix: {e}"),
    }

    // Step 2: write 1 token at position 16 (start of block 1). K = 0.25, V = 0.75.
    adapter.record_tokens(&[17]).unwrap();
    let k_suffix_bits = vec![f32_to_bf16_bits(0.25_f32); 512];
    let v_suffix_bits = vec![f32_to_bf16_bits(0.75_f32); 512];
    let k_suffix = MxArray::from_bfloat16(&k_suffix_bits, &[1, 1, 512]).unwrap();
    let v_suffix = MxArray::from_bfloat16(&v_suffix_bits, &[1, 1, 512]).unwrap();
    k_suffix.eval();
    v_suffix.eval();
    match adapter.update_keys_values(0, &k_suffix, &v_suffix, 16) {
        Ok(()) => {}
        Err(e) if e.contains("Metal GPU not available") => {
            eprintln!("skipping multi_block_round_trip_bf16_head_dim_512 (suffix): {e}");
            return;
        }
        Err(e) => panic!("update_keys_values suffix: {e}"),
    }

    // Step 3: read [0, 17) and verify each position has the expected value.
    let (k_out, v_out) = match adapter.read_kv_range(0, 0, 17) {
        Ok(t) => t,
        Err(e) if e.contains("Metal GPU not available") => {
            eprintln!("skipping multi_block_round_trip_bf16_head_dim_512 (read): {e}");
            return;
        }
        Err(e) => panic!("read_kv_range: {e}"),
    };

    assert_eq!(k_out.shape_at(0).unwrap(), 1);
    assert_eq!(k_out.shape_at(1).unwrap(), 1);
    assert_eq!(k_out.shape_at(2).unwrap(), 17);
    assert_eq!(k_out.shape_at(3).unwrap(), 512);

    let k_f32 = k_out.astype(DType::Float32).unwrap();
    let v_f32 = v_out.astype(DType::Float32).unwrap();
    k_f32.eval();
    v_f32.eval();

    // Layout: [1, 1, 17, 512]. To inspect token t dim d:
    //   index = t * 512 + d (since num_kv_heads=1 and batch=1).
    let head_size = 512usize;

    // Verify positions 0..15 (block 0): K = 0.5, V = 1.0.
    for t in 0..16 {
        let idx = t * head_size; // d=0
        let k_val = k_f32.item_at_float32(idx).unwrap();
        let v_val = v_f32.item_at_float32(idx).unwrap();
        assert!(
            (k_val - 0.5_f32).abs() < 0.01,
            "K[t={t}, d=0] = {k_val}, expected 0.5"
        );
        assert!(
            (v_val - 1.0_f32).abs() < 0.01,
            "V[t={t}, d=0] = {v_val}, expected 1.0"
        );
    }

    // Position 16 (block 1, offset 0): K = 0.25, V = 0.75.
    let idx16 = 16 * head_size;
    let k16 = k_f32.item_at_float32(idx16).unwrap();
    let v16 = v_f32.item_at_float32(idx16).unwrap();
    assert!(
        (k16 - 0.25_f32).abs() < 0.01,
        "K[t=16, d=0] = {k16}, expected 0.25 (just-written block-1 value)"
    );
    assert!(
        (v16 - 0.75_f32).abs() < 0.01,
        "V[t=16, d=0] = {v16}, expected 0.75 (just-written block-1 value)"
    );

    // Also check the LAST dim of position 16 to make sure layout works for d != 0.
    let idx16_last = 16 * head_size + (head_size - 1);
    let k16_last = k_f32.item_at_float32(idx16_last).unwrap();
    let v16_last = v_f32.item_at_float32(idx16_last).unwrap();
    assert!(
        (k16_last - 0.25_f32).abs() < 0.01,
        "K[t=16, d={}] = {k16_last}, expected 0.25",
        head_size - 1
    );
    assert!(
        (v16_last - 0.75_f32).abs() < 0.01,
        "V[t=16, d={}] = {v16_last}, expected 0.75",
        head_size - 1
    );
}
