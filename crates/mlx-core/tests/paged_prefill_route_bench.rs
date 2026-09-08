#![cfg(target_os = "macos")]

use std::sync::{Arc, Mutex};
use std::time::Instant;

use mlx_core::array::mask::create_causal_mask;
use mlx_core::array::{DType, MxArray, scaled_dot_product_attention};
use mlx_core::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{BlockAllocator, LayerKVPool, PagedAttentionConfig};

/// Includes graph construction, dense KV gather and mask construction. Resident
/// KV is established before timing. Both routes use exactly the same Q/K/V.
#[test]
#[ignore = "isolated release benchmark on Metal"]
fn compare_short_suffix_attention_routes() {
    for context in [512u32, 4096, 32768] {
        let blocks = context / 16 + 1;
        let config = PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb: 256,
            head_size: 128,
            num_kv_heads: 8,
            num_layers: 1,
            use_fp8_cache: Some(false),
            max_seq_len: Some(context),
            max_batch_size: Some(1),
        };
        let pool =
            Arc::new(LayerKVPool::new(config, blocks, blocks, MetalDtype::BFloat16).unwrap());
        let allocator = Arc::new(Mutex::new(BlockAllocator::new(blocks, blocks, 16)));
        let mut adapter = PagedKVCacheAdapter::new(allocator, pool, 16).unwrap();
        adapter.reset_for_new_request(1).unwrap();
        adapter.allocate_suffix_blocks(context).unwrap();
        adapter.record_tokens(&vec![1; context as usize]).unwrap();
        let k = MxArray::random_normal(&[context as i64, 8, 128], 0.0, 0.1, Some(DType::BFloat16))
            .unwrap();
        let v = MxArray::random_normal(&[context as i64, 8, 128], 0.0, 0.1, Some(DType::BFloat16))
            .unwrap();
        adapter.update_keys_values_native(0, &k, &v, 0).unwrap();
        let (ready_k, ready_v) = adapter.gather_kv_for_prefill_sdpa(0, context).unwrap();
        ready_k.eval();
        ready_v.eval();
        for query in [1u32, 4, 8, 16, 64, 157] {
            let q =
                MxArray::random_normal(&[query as i64, 16, 128], 0.0, 0.1, Some(DType::BFloat16))
                    .unwrap();
            q.eval();
            let scale = 1.0 / 128.0f64.sqrt();
            let mut build = |route| {
                if route == 0 {
                    let (keys, values) = adapter.gather_kv_for_prefill_sdpa(0, context).unwrap();
                    let queries = q
                        .reshape(&[1, query as i64, 16, 128])
                        .unwrap()
                        .transpose(Some(&[0, 2, 1, 3]))
                        .unwrap();
                    let mask =
                        create_causal_mask(query as i32, Some((context - query) as i32), None)
                            .unwrap();
                    scaled_dot_product_attention(&queries, &keys, &values, scale, Some(&mask))
                        .unwrap()
                        .transpose(Some(&[0, 2, 1, 3]))
                        .unwrap()
                        .reshape(&[query as i64, 16, 128])
                        .unwrap()
                } else {
                    adapter
                        .gather_kv_for_prefill_chunk_varlen(0, &q, context - query, scale as f32)
                        .unwrap()
                }
            };
            let a = build(0).to_float32().unwrap();
            let b = build(1).to_float32().unwrap();
            let max_error = a
                .iter()
                .zip(b.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_error < 0.002,
                "context={context} query={query}: error={max_error}"
            );
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..8 {
                for route in [round % 2, 1 - round % 2] {
                    let start = Instant::now();
                    for _ in 0..20 {
                        build(route).eval();
                    }
                    if round > 0 {
                        times[route].push(start.elapsed().as_secs_f64() * 1e6 / 20.0);
                    }
                }
            }
            for row in &mut times {
                row.sort_by(f64::total_cmp);
            }
            eprintln!(
                "route_bench context={context} query={query} sdpa_us={:.3} paged_us={:.3} max_error={max_error}",
                times[0][3], times[1][3]
            );
        }
        adapter.release_request().unwrap();
    }
}
