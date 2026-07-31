//! Raw-Metal parity for every runtime geometry accepted by the staged BF16
//! D512/BS16 grouped paged-attention capability.

#![cfg(all(target_os = "macos", mlx_node_metal_enabled))]

use std::ffi::c_void;
use std::time::{Duration, Instant};

use metal::{Buffer, MTLResourceOptions};
use mlx_paged_attn::metal::{
    MetalDtype, MetalState, PagedAttentionParams, PagedAttentionRouteHint, RawBufferInfo,
    dispatch_paged_attention_v2_raw_with_route,
};

const HEAD_SIZE: u32 = 512;
const BLOCK_SIZE: u32 = 16;
const X_PACK: usize = 8;

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let bias = 0x7fff + ((bits >> 16) & 1);
    bits.wrapping_add(bias).wrapping_shr(16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn value_for(token: usize, kv_head: usize, dim: usize) -> f32 {
    // Every term is exactly representable in BF16 and varies independently
    // with logical token, KV head, and dimension.
    ((token % 31) as f32 - 15.0) / 64.0 + kv_head as f32 / 8.0 + (dim % 8) as f32 / 128.0
}

fn read_bf16(state: &MetalState, source: &Buffer, elements: usize) -> Vec<f32> {
    let bytes = elements * std::mem::size_of::<u16>();
    let shared = state
        .device
        .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    let command_buffer = state.command_queue.new_command_buffer();
    let encoder = command_buffer.new_blit_command_encoder();
    encoder.copy_from_buffer(source, 0, &shared, 0, bytes as u64);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let bits = unsafe { std::slice::from_raw_parts(shared.contents() as *const u16, elements) };
    bits.iter().copied().map(bf16_bits_to_f32).collect()
}

fn zeroed_shared_buffer(state: &MetalState, bytes: usize) -> Buffer {
    let buffer = state
        .device
        .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    unsafe { std::ptr::write_bytes(buffer.contents() as *mut u8, 0, bytes) };
    buffer
}

fn run_case(state: &MetalState, num_heads: u32, num_kv_heads: u32, context_len: usize) {
    let logical_blocks = context_len.div_ceil(BLOCK_SIZE as usize);
    let physical_blocks = logical_blocks + 2;
    let block_table: Vec<u32> = (0..logical_blocks)
        .map(|logical| (logical_blocks - logical) as u32)
        .collect();
    assert!(
        block_table
            .iter()
            .enumerate()
            .all(|(logical, &physical)| logical != physical as usize)
    );

    let per_head = HEAD_SIZE as usize * BLOCK_SIZE as usize;
    let per_block = num_kv_heads as usize * per_head;
    // Quiet BF16 NaNs poison unused physical blocks and the partial tail.
    let mut key_pool = vec![0x7fc1u16; physical_blocks * per_block];
    let mut value_pool = vec![0x7fc1u16; physical_blocks * per_block];

    for token in 0..context_len {
        let logical_block = token / BLOCK_SIZE as usize;
        let block_offset = token % BLOCK_SIZE as usize;
        let physical_block = block_table[logical_block] as usize;
        for kv_head in 0..num_kv_heads as usize {
            let head_base = physical_block * per_block + kv_head * per_head;
            for dim in 0..HEAD_SIZE as usize {
                // K: [physical_block, kv_head, D/8, BS16, 8].
                let k_index = head_base
                    + (dim / X_PACK) * BLOCK_SIZE as usize * X_PACK
                    + block_offset * X_PACK
                    + dim % X_PACK;
                // Zero Q and zero K produce a uniform softmax.
                key_pool[k_index] = f32_to_bf16_bits(0.0);

                // V: [physical_block, kv_head, D, BS16].
                let v_index = head_base + dim * BLOCK_SIZE as usize + block_offset;
                value_pool[v_index] = f32_to_bf16_bits(value_for(token, kv_head, dim));
            }
        }
    }

    let queries = vec![f32_to_bf16_bits(0.0); num_heads as usize * HEAD_SIZE as usize];
    let context_lens = [context_len as u32];
    let key_buffer = state
        .device
        .new_buffer_with_slice(key_pool.as_ref(), MTLResourceOptions::StorageModeShared);
    let value_buffer = state
        .device
        .new_buffer_with_slice(value_pool.as_ref(), MTLResourceOptions::StorageModeShared);
    let query_buffer = state
        .device
        .new_buffer_with_slice(queries.as_ref(), MTLResourceOptions::StorageModeShared);
    let table_buffer = state
        .device
        .new_buffer_with_slice(block_table.as_ref(), MTLResourceOptions::StorageModeShared);
    let lens_buffer = state
        .device
        .new_buffer_with_slice(context_lens.as_ref(), MTLResourceOptions::StorageModeShared);

    let query = RawBufferInfo {
        ptr: query_buffer.as_ptr() as *mut c_void,
        offset: 0,
    };
    let params = PagedAttentionParams {
        num_seqs: 1,
        num_heads,
        num_kv_heads,
        head_size: HEAD_SIZE,
        block_size: BLOCK_SIZE,
        max_seq_len: context_len as u32,
        max_num_blocks_per_seq: logical_blocks as u32,
        scale: 1.0,
        softcapping: 1.0,
        q_stride: (num_heads * HEAD_SIZE) as i32,
        kv_block_stride: (num_kv_heads * HEAD_SIZE * BLOCK_SIZE) as i32,
        kv_head_stride: (HEAD_SIZE * BLOCK_SIZE) as i32,
        k_scale: 1.0,
        v_scale: 1.0,
        sliding_window: 0,
    };
    let output = unsafe {
        dispatch_paged_attention_v2_raw_with_route(
            &query,
            &key_buffer,
            &value_buffer,
            &table_buffer,
            &lens_buffer,
            &params,
            MetalDtype::BFloat16,
            MetalDtype::BFloat16,
            PagedAttentionRouteHint::ForceD512Staged,
        )
    }
    .expect("staged D512 raw dispatch must succeed");
    assert!(
        output.used_grouped_d512,
        "q={num_heads} kv={num_kv_heads} silently used generic V2"
    );

    let actual = read_bf16(
        state,
        &output.buffer,
        num_heads as usize * HEAD_SIZE as usize,
    );
    let gqa_factor = num_heads as usize / num_kv_heads as usize;
    let mut worst = (0.0f32, 0usize, 0.0f32);
    for head in 0..num_heads as usize {
        let kv_head = head / gqa_factor;
        for dim in 0..HEAD_SIZE as usize {
            let expected = (0..context_len)
                .map(|token| value_for(token, kv_head, dim))
                .sum::<f32>()
                / context_len as f32;
            let index = head * HEAD_SIZE as usize + dim;
            let difference = (actual[index] - expected).abs();
            if !actual[index].is_finite() || difference > worst.0 {
                worst = (difference, index, expected);
            }
        }
    }
    assert!(
        worst.0 <= 1.2e-2,
        "q={num_heads} kv={num_kv_heads} context={context_len}: \
         mismatch at head={}, dim={}: actual={}, expected={}, diff={}",
        worst.1 / HEAD_SIZE as usize,
        worst.1 % HEAD_SIZE as usize,
        actual[worst.1],
        worst.2,
        worst.0,
    );
}

#[test]
fn grouped_d512_raw_decode_matches_uniform_reference_for_every_geometry() {
    let state = match MetalState::get() {
        Ok(state) => state,
        Err(error) if error.contains("No Metal device found") => {
            eprintln!("skipping grouped D512 raw parity: {error}");
            return;
        }
        Err(error) => panic!("unexpected MetalState::get failure: {error}"),
    };

    for (num_heads, num_kv_heads) in [(8, 1), (16, 1), (16, 2), (32, 4)] {
        run_case(state, num_heads, num_kv_heads, 513);
    }
}

fn benchmark_dispatch(
    state: &MetalState,
    num_heads: u32,
    num_kv_heads: u32,
    context_len: usize,
    route_hint: PagedAttentionRouteHint,
) -> Duration {
    let logical_blocks = context_len.div_ceil(BLOCK_SIZE as usize);
    let pool_elements =
        logical_blocks * num_kv_heads as usize * HEAD_SIZE as usize * BLOCK_SIZE as usize;
    let pool_bytes = pool_elements * std::mem::size_of::<u16>();
    let key_buffer = zeroed_shared_buffer(state, pool_bytes);
    let value_buffer = zeroed_shared_buffer(state, pool_bytes);
    let query_buffer = zeroed_shared_buffer(
        state,
        num_heads as usize * HEAD_SIZE as usize * std::mem::size_of::<u16>(),
    );
    let block_table: Vec<u32> = (0..logical_blocks as u32).collect();
    let context_lens = [context_len as u32];
    let table_buffer = state
        .device
        .new_buffer_with_slice(block_table.as_ref(), MTLResourceOptions::StorageModeShared);
    let lens_buffer = state
        .device
        .new_buffer_with_slice(context_lens.as_ref(), MTLResourceOptions::StorageModeShared);
    let query = RawBufferInfo {
        ptr: query_buffer.as_ptr() as *mut c_void,
        offset: 0,
    };
    let params = PagedAttentionParams {
        num_seqs: 1,
        num_heads,
        num_kv_heads,
        head_size: HEAD_SIZE,
        block_size: BLOCK_SIZE,
        max_seq_len: context_len as u32,
        max_num_blocks_per_seq: logical_blocks as u32,
        scale: 1.0,
        softcapping: 1.0,
        q_stride: (num_heads * HEAD_SIZE) as i32,
        kv_block_stride: (num_kv_heads * HEAD_SIZE * BLOCK_SIZE) as i32,
        kv_head_stride: (HEAD_SIZE * BLOCK_SIZE) as i32,
        k_scale: 1.0,
        v_scale: 1.0,
        sliding_window: 0,
    };

    let started = Instant::now();
    let output = unsafe {
        dispatch_paged_attention_v2_raw_with_route(
            &query,
            &key_buffer,
            &value_buffer,
            &table_buffer,
            &lens_buffer,
            &params,
            MetalDtype::BFloat16,
            MetalDtype::BFloat16,
            route_hint,
        )
    }
    .expect("D512 benchmark dispatch must succeed");
    assert_eq!(
        output.used_grouped_d512,
        route_hint == PagedAttentionRouteHint::ForceD512Staged,
        "benchmark route did not honor its explicit hint"
    );
    std::hint::black_box(output.buffer_ptr());
    started.elapsed()
}

fn benchmark_route_pair(
    state: &MetalState,
    num_heads: u32,
    num_kv_heads: u32,
    context_len: usize,
    warmups: usize,
    iterations: usize,
) -> (f64, f64) {
    for _ in 0..warmups {
        let _ = benchmark_dispatch(
            state,
            num_heads,
            num_kv_heads,
            context_len,
            PagedAttentionRouteHint::ForceD512Staged,
        );
        let _ = benchmark_dispatch(
            state,
            num_heads,
            num_kv_heads,
            context_len,
            PagedAttentionRouteHint::ForceGeneric,
        );
    }

    let mut grouped = Duration::ZERO;
    let mut generic = Duration::ZERO;
    for iteration in 0..iterations {
        let routes = if iteration.is_multiple_of(2) {
            [
                PagedAttentionRouteHint::ForceD512Staged,
                PagedAttentionRouteHint::ForceGeneric,
            ]
        } else {
            [
                PagedAttentionRouteHint::ForceGeneric,
                PagedAttentionRouteHint::ForceD512Staged,
            ]
        };
        for route in routes {
            let elapsed = benchmark_dispatch(state, num_heads, num_kv_heads, context_len, route);
            match route {
                PagedAttentionRouteHint::ForceD512Staged => grouped += elapsed,
                PagedAttentionRouteHint::ForceGeneric => generic += elapsed,
                PagedAttentionRouteHint::Auto => unreachable!(),
            }
        }
    }
    (
        grouped.as_secs_f64() * 1_000.0 / iterations as f64,
        generic.as_secs_f64() * 1_000.0 / iterations as f64,
    )
}

#[test]
#[ignore = "manual Metal performance benchmark"]
fn grouped_d512_hq16_hkv2_alternating_benchmark() {
    const WARMUPS: usize = 2;
    const ITERATIONS: usize = 7;

    let state = match MetalState::get() {
        Ok(state) => state,
        Err(error) if error.contains("No Metal device found") => {
            eprintln!("skipping grouped D512 benchmark: {error}");
            return;
        }
        Err(error) => panic!("unexpected MetalState::get failure: {error}"),
    };

    eprintln!(
        "D512 Hq16/Hkv2 raw benchmark: warmups={WARMUPS}, iterations={ITERATIONS}, \
         stripe_override={}",
        std::env::var("MLX_PAGED_GROUPED_D512_STRIPES")
            .or_else(|_| std::env::var("MLX_PAGED_GROUPED_GEMMA4_STRIPES"))
            .unwrap_or_else(|_| "default".to_string())
    );
    for context_len in [4_096, 16_384, 32_768, 65_536, 91_765, 112_000] {
        let (grouped_ms, generic_ms) =
            benchmark_route_pair(state, 16, 2, context_len, WARMUPS, ITERATIONS);
        eprintln!(
            "context={context_len:>6} grouped_ms={grouped_ms:>9.3} \
             generic_ms={generic_ms:>9.3} generic/grouped={:>6.3}x",
            generic_ms / grouped_ms
        );
    }
}

#[test]
#[ignore = "manual Metal performance benchmark"]
fn grouped_d512_all_geometry_long_context_benchmark() {
    const WARMUPS: usize = 2;
    const ITERATIONS: usize = 7;

    let state = match MetalState::get() {
        Ok(state) => state,
        Err(error) if error.contains("No Metal device found") => {
            eprintln!("skipping grouped D512 geometry benchmark: {error}");
            return;
        }
        Err(error) => panic!("unexpected MetalState::get failure: {error}"),
    };

    eprintln!(
        "D512 all-geometry raw benchmark: warmups={WARMUPS}, iterations={ITERATIONS}, \
         Hkv-aware default stripes"
    );
    for (num_heads, num_kv_heads) in [(8, 1), (16, 1), (16, 2), (32, 4)] {
        for context_len in [91_765, 112_000] {
            let (grouped_ms, generic_ms) = benchmark_route_pair(
                state,
                num_heads,
                num_kv_heads,
                context_len,
                WARMUPS,
                ITERATIONS,
            );
            eprintln!(
                "q={num_heads:>2} kv={num_kv_heads} context={context_len:>6} \
                 grouped_ms={grouped_ms:>9.3} generic_ms={generic_ms:>9.3} \
                 generic/grouped={:>6.3}x",
                generic_ms / grouped_ms
            );
        }
    }
}
