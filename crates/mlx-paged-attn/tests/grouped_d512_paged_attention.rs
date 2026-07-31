//! Raw-Metal parity for every runtime geometry accepted by the grouped BF16
//! D512/BS16 grouped paged-attention capability.

#![cfg(all(target_os = "macos", mlx_node_metal_enabled))]

use std::ffi::c_void;
use std::time::{Duration, Instant};

use metal::{Buffer, ComputePipelineState, MTLResourceOptions, MTLSize};
use mlx_paged_attn::metal::{
    MetalDtype, MetalState, PagedAttentionParams, PagedAttentionRouteHint, RawBufferInfo,
    dispatch_paged_attention_v2_raw_with_route,
};

const HEAD_SIZE: u32 = 512;
const BLOCK_SIZE: u32 = 16;
const X_PACK: usize = 8;
const TARGET_CONTEXT: usize = 91_795;
const TARGET_NUM_HEADS: u32 = 16;
const TARGET_NUM_KV_HEADS: u32 = 2;
const TARGET_STRIPES: usize = 128;
const LOGICAL_LAYERS: usize = 5;
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

fn read_bf16_bits(state: &MetalState, source: &Buffer, elements: usize) -> Vec<u16> {
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
    bits.to_vec()
}

fn read_bf16(state: &MetalState, source: &Buffer, elements: usize) -> Vec<f32> {
    read_bf16_bits(state, source, elements)
        .into_iter()
        .map(bf16_bits_to_f32)
        .collect()
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
    .expect("grouped D512 raw dispatch must succeed");
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

#[test]
fn grouped_d512_production_and_staged_rollback_pipelines_load() {
    let state = match MetalState::get() {
        Ok(state) => state,
        Err(error) if error.contains("No Metal device found") => {
            eprintln!("skipping grouped D512 pipeline capability check: {error}");
            return;
        }
        Err(error) => panic!("unexpected MetalState::get failure: {error}"),
    };
    let production_name = MetalState::paged_attention_grouped_d512_kernel_name();
    let staged_name = MetalState::paged_attention_grouped_d512_staged_kernel_name();
    assert_ne!(production_name, staged_name);
    const D512_PAGE_BYTES: u64 =
        (HEAD_SIZE as usize * BLOCK_SIZE as usize * std::mem::size_of::<u16>()) as u64;
    for (label, name) in [
        ("production direct", production_name),
        ("staged rollback", staged_name),
    ] {
        let pipeline = state
            .get_pipeline(name)
            .unwrap_or_else(|error| panic!("{label} D512 pipeline must load: {error}"));
        assert!(
            pipeline.max_total_threads_per_threadgroup() >= 512,
            "{label} D512 pipeline cannot launch the maximum Hq16/Hkv1 threadgroup"
        );
        let static_memory = pipeline.static_threadgroup_memory_length();
        if name == production_name {
            assert!(
                static_memory < D512_PAGE_BYTES,
                "production D512 pipeline unexpectedly stages a full K/V page"
            );
        } else {
            assert!(
                static_memory >= D512_PAGE_BYTES,
                "staged rollback D512 pipeline does not reserve a full K/V page"
            );
        }
    }
    let reducer = state
        .get_pipeline(MetalState::paged_attention_grouped_d512_reduce_kernel_name())
        .expect("D512 reducer pipeline must load");
    assert!(
        reducer.max_total_threads_per_threadgroup() >= 1024,
        "D512 reducer cannot launch its required 1024-thread threadgroup"
    );
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

struct D512OperatorBenchmark {
    output: Buffer,
    exp_sums: Buffer,
    max_logits: Buffer,
    partials: Buffer,
    key_buffer: Buffer,
    value_buffer: Buffer,
    query_buffer: Buffer,
    table_buffer: Buffer,
    lens_buffer: Buffer,
    k_scale: Buffer,
    v_scale: Buffer,
    num_kv_heads: Buffer,
    scale: Buffer,
    softcapping: Buffer,
    max_num_blocks: Buffer,
    alibi_slopes: Buffer,
    q_stride: Buffer,
    kv_block_stride: Buffer,
    kv_head_stride: Buffer,
    sliding_window: Buffer,
    num_stripes: Buffer,
    staged_pipeline: ComputePipelineState,
    direct_pipeline: ComputePipelineState,
    reduce_pipeline: ComputePipelineState,
}

#[derive(Clone, Copy, Debug)]
enum D512StageMode {
    Staged,
    Direct,
}

fn exact_benchmark_value(kv_head: usize, dim: usize) -> f32 {
    ((kv_head * 3 + dim) % 5) as f32 / 8.0 - 0.25
}

fn exact_benchmark_query_value(head: usize, dim: usize) -> f32 {
    ((head * 11 + dim * 5) % 17) as f32 / 256.0 - 8.0 / 256.0
}

fn exact_benchmark_key_value(token: usize, kv_head: usize, dim: usize) -> f32 {
    ((token * 7 + kv_head * 13 + dim * 3) % 19) as f32 / 256.0 - 9.0 / 256.0
}

fn assert_bf16_identical(label: &str, left: &[u16], right: &[u16]) {
    assert_eq!(left.len(), right.len(), "{label}: output lengths differ");
    if let Some(index) = left
        .iter()
        .zip(right)
        .position(|(left, right)| left != right)
    {
        panic!(
            "{label}: mismatch at flat index {index}: left={} (0x{:04x}), \
             right={} (0x{:04x})",
            bf16_bits_to_f32(left[index]),
            left[index],
            bf16_bits_to_f32(right[index]),
            right[index],
        );
    }
}

fn bf16_fnv1a(bits: &[u16]) -> u64 {
    bits.iter()
        .flat_map(|bits| bits.to_le_bytes())
        .fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
}

struct D512ParityChecksums {
    uniform: u64,
    nonzero_qkv: u64,
}

fn exact_benchmark_value_buffer(
    state: &MetalState,
    logical_blocks: usize,
    num_kv_heads: usize,
) -> Buffer {
    let per_head = HEAD_SIZE as usize * BLOCK_SIZE as usize;
    let per_block = num_kv_heads * per_head;
    let elements = logical_blocks * per_block;
    let buffer = state.device.new_buffer(
        (elements * std::mem::size_of::<u16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let values = unsafe { std::slice::from_raw_parts_mut(buffer.contents() as *mut u16, elements) };
    for physical_block in 0..logical_blocks {
        for kv_head in 0..num_kv_heads {
            let head_base = physical_block * per_block + kv_head * per_head;
            for dim in 0..HEAD_SIZE as usize {
                let row_start = head_base + dim * BLOCK_SIZE as usize;
                values[row_start..row_start + BLOCK_SIZE as usize]
                    .fill(f32_to_bf16_bits(exact_benchmark_value(kv_head, dim)));
            }
        }
    }
    buffer
}

impl D512OperatorBenchmark {
    fn new(state: &MetalState) -> Self {
        let logical_blocks = TARGET_CONTEXT.div_ceil(BLOCK_SIZE as usize);
        let pool_elements = logical_blocks
            * TARGET_NUM_KV_HEADS as usize
            * HEAD_SIZE as usize
            * BLOCK_SIZE as usize;
        let pool_bytes = pool_elements * std::mem::size_of::<u16>();
        let key_buffer = zeroed_shared_buffer(state, pool_bytes);
        let value_buffer =
            exact_benchmark_value_buffer(state, logical_blocks, TARGET_NUM_KV_HEADS as usize);
        let query_buffer = zeroed_shared_buffer(
            state,
            TARGET_NUM_HEADS as usize * HEAD_SIZE as usize * std::mem::size_of::<u16>(),
        );
        let block_table: Vec<u32> = (0..logical_blocks as u32).collect();
        let context_lens = [TARGET_CONTEXT as u32];
        let table_buffer = state
            .device
            .new_buffer_with_slice(block_table.as_ref(), MTLResourceOptions::StorageModeShared);
        let lens_buffer = state
            .device
            .new_buffer_with_slice(context_lens.as_ref(), MTLResourceOptions::StorageModeShared);
        let params = PagedAttentionParams {
            num_seqs: 1,
            num_heads: TARGET_NUM_HEADS,
            num_kv_heads: TARGET_NUM_KV_HEADS,
            head_size: HEAD_SIZE,
            block_size: BLOCK_SIZE,
            max_seq_len: TARGET_CONTEXT as u32,
            max_num_blocks_per_seq: logical_blocks as u32,
            scale: 1.0,
            softcapping: 1.0,
            q_stride: (TARGET_NUM_HEADS * HEAD_SIZE) as i32,
            kv_block_stride: (TARGET_NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE) as i32,
            kv_head_stride: (HEAD_SIZE * BLOCK_SIZE) as i32,
            k_scale: 1.0,
            v_scale: 1.0,
            sliding_window: 0,
        };
        let output = state.device.new_buffer(
            (TARGET_NUM_HEADS as usize * HEAD_SIZE as usize * std::mem::size_of::<u16>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let stats_elements = TARGET_NUM_HEADS as usize * TARGET_STRIPES;
        let exp_sums = state.device.new_buffer(
            (stats_elements * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let max_logits = state.device.new_buffer(
            (stats_elements * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let partials = state.device.new_buffer(
            (stats_elements * HEAD_SIZE as usize * std::mem::size_of::<u16>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let shared = MTLResourceOptions::StorageModeShared;
        let k_scale = state.device.new_buffer_with_value(&params.k_scale, shared);
        let v_scale = state.device.new_buffer_with_value(&params.v_scale, shared);
        let num_kv_heads = state
            .device
            .new_buffer_with_value(&(params.num_kv_heads as i32), shared);
        let scale = state.device.new_buffer_with_value(&params.scale, shared);
        let softcapping = state
            .device
            .new_buffer_with_value(&params.softcapping, shared);
        let max_num_blocks = state
            .device
            .new_buffer_with_value(&(params.max_num_blocks_per_seq as i32), shared);
        let alibi_slopes = state.device.new_buffer_with_value(&0.0f32, shared);
        let q_stride = state.device.new_buffer_with_value(&params.q_stride, shared);
        let kv_block_stride = state
            .device
            .new_buffer_with_value(&params.kv_block_stride, shared);
        let kv_head_stride = state
            .device
            .new_buffer_with_value(&params.kv_head_stride, shared);
        let sliding_window = state
            .device
            .new_buffer_with_value(&params.sliding_window, shared);
        let num_stripes = state
            .device
            .new_buffer_with_value(&(TARGET_STRIPES as i32), shared);
        let staged_pipeline = state
            .get_pipeline(MetalState::paged_attention_grouped_d512_staged_kernel_name())
            .expect("staged rollback D512 benchmark pipeline must load");
        let direct_pipeline = state
            .get_pipeline(MetalState::paged_attention_grouped_d512_kernel_name())
            .expect("production direct D512 benchmark pipeline must load");
        let reduce_pipeline = state
            .get_pipeline(MetalState::paged_attention_grouped_d512_reduce_kernel_name())
            .expect("D512 benchmark reducer pipeline must load");

        Self {
            output,
            exp_sums,
            max_logits,
            partials,
            key_buffer,
            value_buffer,
            query_buffer,
            table_buffer,
            lens_buffer,
            k_scale,
            v_scale,
            num_kv_heads,
            scale,
            softcapping,
            max_num_blocks,
            alibi_slopes,
            q_stride,
            kv_block_stride,
            kv_head_stride,
            sliding_window,
            num_stripes,
            staged_pipeline,
            direct_pipeline,
            reduce_pipeline,
        }
    }

    fn dispatch_layer(&self, state: &MetalState, mode: D512StageMode) {
        let stage_pipeline = match mode {
            D512StageMode::Staged => &self.staged_pipeline,
            D512StageMode::Direct => &self.direct_pipeline,
        };
        let command_buffer = state.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(stage_pipeline);
        encoder.set_buffer(0, Some(&self.exp_sums), 0);
        encoder.set_buffer(1, Some(&self.max_logits), 0);
        encoder.set_buffer(2, Some(&self.partials), 0);
        encoder.set_buffer(3, Some(&self.query_buffer), 0);
        encoder.set_buffer(4, Some(&self.key_buffer), 0);
        encoder.set_buffer(5, Some(&self.value_buffer), 0);
        encoder.set_buffer(6, Some(&self.k_scale), 0);
        encoder.set_buffer(7, Some(&self.v_scale), 0);
        encoder.set_buffer(8, Some(&self.num_kv_heads), 0);
        encoder.set_buffer(9, Some(&self.scale), 0);
        encoder.set_buffer(10, Some(&self.softcapping), 0);
        encoder.set_buffer(11, Some(&self.table_buffer), 0);
        encoder.set_buffer(12, Some(&self.lens_buffer), 0);
        encoder.set_buffer(13, Some(&self.max_num_blocks), 0);
        encoder.set_buffer(14, Some(&self.alibi_slopes), 0);
        encoder.set_buffer(15, Some(&self.q_stride), 0);
        encoder.set_buffer(16, Some(&self.kv_block_stride), 0);
        encoder.set_buffer(17, Some(&self.kv_head_stride), 0);
        encoder.set_buffer(18, Some(&self.sliding_window), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(TARGET_NUM_KV_HEADS as u64, 1, TARGET_STRIPES as u64),
            MTLSize::new(32, (TARGET_NUM_HEADS / TARGET_NUM_KV_HEADS) as u64, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let command_buffer = state.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.reduce_pipeline);
        encoder.set_buffer(0, Some(&self.output), 0);
        encoder.set_buffer(1, Some(&self.exp_sums), 0);
        encoder.set_buffer(2, Some(&self.max_logits), 0);
        encoder.set_buffer(3, Some(&self.partials), 0);
        encoder.set_buffer(4, Some(&self.lens_buffer), 0);
        encoder.set_buffer(5, Some(&self.num_stripes), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(TARGET_NUM_HEADS as u64, 1, 1),
            MTLSize::new(1024, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        std::hint::black_box(self.output.as_ptr());
    }

    fn dispatch_logical_layers(&self, state: &MetalState, mode: D512StageMode) {
        for _ in 0..LOGICAL_LAYERS {
            self.dispatch_layer(state, mode);
        }
    }

    fn load_nonzero_query_and_keys(&self) {
        let query_elements = TARGET_NUM_HEADS as usize * HEAD_SIZE as usize;
        let query = unsafe {
            std::slice::from_raw_parts_mut(self.query_buffer.contents() as *mut u16, query_elements)
        };
        for head in 0..TARGET_NUM_HEADS as usize {
            for dim in 0..HEAD_SIZE as usize {
                query[head * HEAD_SIZE as usize + dim] =
                    f32_to_bf16_bits(exact_benchmark_query_value(head, dim));
            }
        }

        let logical_blocks = TARGET_CONTEXT.div_ceil(BLOCK_SIZE as usize);
        let per_head = HEAD_SIZE as usize * BLOCK_SIZE as usize;
        let per_block = TARGET_NUM_KV_HEADS as usize * per_head;
        let key_elements = logical_blocks * per_block;
        let keys = unsafe {
            std::slice::from_raw_parts_mut(self.key_buffer.contents() as *mut u16, key_elements)
        };
        for physical_block in 0..logical_blocks {
            for kv_head in 0..TARGET_NUM_KV_HEADS as usize {
                let head_base = physical_block * per_block + kv_head * per_head;
                for pack in 0..HEAD_SIZE as usize / X_PACK {
                    for token_in_block in 0..BLOCK_SIZE as usize {
                        let token = physical_block * BLOCK_SIZE as usize + token_in_block;
                        let pack_base =
                            head_base + (pack * BLOCK_SIZE as usize + token_in_block) * X_PACK;
                        for element in 0..X_PACK {
                            let dim = pack * X_PACK + element;
                            keys[pack_base + element] =
                                f32_to_bf16_bits(exact_benchmark_key_value(token, kv_head, dim));
                        }
                    }
                }
            }
        }
    }

    fn assert_exact_parity_and_reference(&self, state: &MetalState) -> D512ParityChecksums {
        let elements = TARGET_NUM_HEADS as usize * HEAD_SIZE as usize;
        self.dispatch_layer(state, D512StageMode::Staged);
        let staged = read_bf16_bits(state, &self.output, elements);
        self.dispatch_layer(state, D512StageMode::Direct);
        let direct = read_bf16_bits(state, &self.output, elements);
        assert_bf16_identical("uniform staged/direct exact BF16 parity", &staged, &direct);
        let uniform = bf16_fnv1a(&staged);

        let gqa_factor = TARGET_NUM_HEADS as usize / TARGET_NUM_KV_HEADS as usize;
        for head in 0..TARGET_NUM_HEADS as usize {
            let kv_head = head / gqa_factor;
            for dim in 0..HEAD_SIZE as usize {
                let index = head * HEAD_SIZE as usize + dim;
                let expected = f32_to_bf16_bits(exact_benchmark_value(kv_head, dim));
                assert_eq!(
                    staged[index],
                    expected,
                    "exact BF16 reference mismatch at head={head}, dim={dim}: \
                     actual={} expected={}",
                    bf16_bits_to_f32(staged[index]),
                    bf16_bits_to_f32(expected),
                );
            }
        }

        self.load_nonzero_query_and_keys();
        self.dispatch_layer(state, D512StageMode::Staged);
        let staged = read_bf16_bits(state, &self.output, elements);
        self.dispatch_layer(state, D512StageMode::Direct);
        let direct = read_bf16_bits(state, &self.output, elements);
        assert_bf16_identical(
            "nonzero Q/K/V staged/direct exact BF16 parity",
            &staged,
            &direct,
        );
        D512ParityChecksums {
            uniform,
            nonzero_qkv: bf16_fnv1a(&staged),
        }
    }
}

struct D512ReducerBenchmark {
    output: Buffer,
    exp_sums: Buffer,
    max_logits: Buffer,
    partials: Buffer,
    context_lens: Buffer,
    num_stripes: Buffer,
    pipeline: ComputePipelineState,
}

impl D512ReducerBenchmark {
    fn new(state: &MetalState) -> Self {
        let rows = TARGET_NUM_HEADS as usize;
        let stats_elements = rows * TARGET_STRIPES;
        let mut exp_sums = Vec::with_capacity(stats_elements);
        let mut max_logits = Vec::with_capacity(stats_elements);
        let mut partials = Vec::with_capacity(stats_elements * HEAD_SIZE as usize);
        for row in 0..rows {
            for stripe in 0..TARGET_STRIPES {
                exp_sums.push(0.5 + ((row * 7 + stripe * 13) % 31) as f32 / 32.0);
                max_logits.push(((row * 11 + stripe * 17) % 67) as f32 / 64.0 - 0.5);
                for dim in 0..HEAD_SIZE as usize {
                    let value = ((row * 3 + stripe * 5 + dim * 7) % 41) as f32 / 128.0 - 0.125;
                    partials.push(f32_to_bf16_bits(value));
                }
            }
        }

        let output = state.device.new_buffer(
            (rows * HEAD_SIZE as usize * std::mem::size_of::<u16>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let exp_sums = state
            .device
            .new_buffer_with_slice(exp_sums.as_ref(), MTLResourceOptions::StorageModeShared);
        let max_logits = state
            .device
            .new_buffer_with_slice(max_logits.as_ref(), MTLResourceOptions::StorageModeShared);
        let partials = state
            .device
            .new_buffer_with_slice(partials.as_ref(), MTLResourceOptions::StorageModeShared);
        let context_lens = state.device.new_buffer_with_slice(
            &[TARGET_CONTEXT as u32],
            MTLResourceOptions::StorageModeShared,
        );
        let num_stripes = state.device.new_buffer_with_value(
            &(TARGET_STRIPES as i32),
            MTLResourceOptions::StorageModeShared,
        );
        let pipeline = state
            .get_pipeline(MetalState::paged_attention_grouped_d512_reduce_kernel_name())
            .expect("D512 reducer pipeline must load");

        Self {
            output,
            exp_sums,
            max_logits,
            partials,
            context_lens,
            num_stripes,
            pipeline,
        }
    }

    fn dispatch_logical_layers(&self, state: &MetalState) {
        let command_buffer = state.command_queue.new_command_buffer();
        for _ in 0..LOGICAL_LAYERS {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_buffer(0, Some(&self.output), 0);
            encoder.set_buffer(1, Some(&self.exp_sums), 0);
            encoder.set_buffer(2, Some(&self.max_logits), 0);
            encoder.set_buffer(3, Some(&self.partials), 0);
            encoder.set_buffer(4, Some(&self.context_lens), 0);
            encoder.set_buffer(5, Some(&self.num_stripes), 0);
            encoder.dispatch_thread_groups(
                MTLSize::new(TARGET_NUM_HEADS as u64, 1, 1),
                MTLSize::new(1024, 1, 1),
            );
            encoder.end_encoding();
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}

fn benchmark_samples(
    warmups: usize,
    samples: usize,
    rounds_per_sample: usize,
    mut dispatch: impl FnMut(),
) -> Vec<f64> {
    for _ in 0..warmups {
        dispatch();
    }
    (0..samples)
        .map(|_| {
            let started = Instant::now();
            for _ in 0..rounds_per_sample {
                dispatch();
            }
            started.elapsed().as_secs_f64() * 1_000.0 / rounds_per_sample as f64
        })
        .collect()
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    }
}

fn measure_operator(
    state: &MetalState,
    operator: &D512OperatorBenchmark,
    mode: D512StageMode,
) -> f64 {
    let started = Instant::now();
    operator.dispatch_logical_layers(state, mode);
    started.elapsed().as_secs_f64() * 1_000.0
}

fn paired_operator_samples(
    state: &MetalState,
    operator: &D512OperatorBenchmark,
    warmups: usize,
    samples: usize,
) -> Vec<(f64, f64)> {
    for pair in 0..warmups {
        let order = if pair.is_multiple_of(2) {
            [D512StageMode::Staged, D512StageMode::Direct]
        } else {
            [D512StageMode::Direct, D512StageMode::Staged]
        };
        for mode in order {
            operator.dispatch_logical_layers(state, mode);
        }
    }

    (0..samples)
        .map(|pair| {
            if pair.is_multiple_of(2) {
                let staged = measure_operator(state, operator, D512StageMode::Staged);
                let direct = measure_operator(state, operator, D512StageMode::Direct);
                (staged, direct)
            } else {
                let direct = measure_operator(state, operator, D512StageMode::Direct);
                let staged = measure_operator(state, operator, D512StageMode::Staged);
                (staged, direct)
            }
        })
        .collect()
}

#[test]
#[ignore = "manual exact-shape D512 staged/direct parity"]
fn grouped_d512_exact_context_staged_direct_parity() {
    let state = match MetalState::get() {
        Ok(state) => state,
        Err(error) if error.contains("No Metal device found") => {
            eprintln!("skipping exact-shape D512 staged/direct parity: {error}");
            return;
        }
        Err(error) => panic!("unexpected MetalState::get failure: {error}"),
    };
    let operator = D512OperatorBenchmark::new(state);
    let checksums = operator.assert_exact_parity_and_reference(state);
    eprintln!(
        "D512 exact parity: Hq={TARGET_NUM_HEADS} Hkv={TARGET_NUM_KV_HEADS} \
         D={HEAD_SIZE} context={TARGET_CONTEXT} stripes={TARGET_STRIPES} \
         uniform_reference=pass uniform_exact_bf16_parity=pass \
         nonzero_qkv_exact_bf16_parity=pass uniform_staged_fnv=0x{:016x} \
         uniform_direct_fnv=0x{:016x} nonzero_qkv_staged_fnv=0x{:016x} \
         nonzero_qkv_direct_fnv=0x{:016x}",
        checksums.uniform, checksums.uniform, checksums.nonzero_qkv, checksums.nonzero_qkv,
    );
}

#[test]
#[ignore = "manual exact-shape D512 staged/direct operator benchmark"]
fn grouped_d512_exact_context_reducer_and_operator_benchmark() {
    const WARMUPS: usize = 5;
    const SAMPLES: usize = 7;
    const REDUCER_ROUNDS_PER_SAMPLE: usize = 20;

    unsafe {
        std::env::set_var("MLX_PAGED_GROUPED_D512_STRIPES", TARGET_STRIPES.to_string());
    }
    let state = match MetalState::get() {
        Ok(state) => state,
        Err(error) if error.contains("No Metal device found") => {
            eprintln!("skipping exact-shape D512 staged/direct benchmark: {error}");
            return;
        }
        Err(error) => panic!("unexpected MetalState::get failure: {error}"),
    };
    let reducer = D512ReducerBenchmark::new(state);
    let operator = D512OperatorBenchmark::new(state);
    let checksums = operator.assert_exact_parity_and_reference(state);

    let reducer_samples = benchmark_samples(WARMUPS, SAMPLES, REDUCER_ROUNDS_PER_SAMPLE, || {
        reducer.dispatch_logical_layers(state);
    });
    let operator_pairs = paired_operator_samples(state, &operator, WARMUPS, SAMPLES);
    let staged_samples: Vec<f64> = operator_pairs.iter().map(|pair| pair.0).collect();
    let direct_samples: Vec<f64> = operator_pairs.iter().map(|pair| pair.1).collect();

    eprintln!(
        "D512 exact benchmark: Hq={TARGET_NUM_HEADS} Hkv={TARGET_NUM_KV_HEADS} \
         D={HEAD_SIZE} context={TARGET_CONTEXT} stripes={TARGET_STRIPES} \
         logical_layers={LOGICAL_LAYERS} warmups={WARMUPS} samples={SAMPLES}"
    );
    for (index, sample) in reducer_samples.iter().enumerate() {
        eprintln!(
            "reducer sample={:02} five_layer_ms={sample:.6} per_layer_ms={:.6}",
            index + 1,
            sample / LOGICAL_LAYERS as f64,
        );
    }
    eprintln!(
        "reducer median_five_layer_ms={:.6} median_per_layer_ms={:.6}",
        median(&reducer_samples),
        median(&reducer_samples) / LOGICAL_LAYERS as f64,
    );
    eprintln!(
        "operator uniform_reference=pass uniform_exact_bf16_parity=pass \
         nonzero_qkv_exact_bf16_parity=pass paired_order=alternating \
         uniform_staged_fnv=0x{:016x} uniform_direct_fnv=0x{:016x} \
         nonzero_qkv_staged_fnv=0x{:016x} nonzero_qkv_direct_fnv=0x{:016x}",
        checksums.uniform, checksums.uniform, checksums.nonzero_qkv, checksums.nonzero_qkv,
    );
    for (index, (staged, direct)) in operator_pairs.iter().enumerate() {
        eprintln!(
            "operator pair={:02} order={} staged_five_layer_ms={staged:.6} \
             direct_five_layer_ms={direct:.6} direct_vs_staged_pct={:+.3}% \
             staged_over_direct={:.4}x",
            index + 1,
            if index.is_multiple_of(2) {
                "staged,direct"
            } else {
                "direct,staged"
            },
            (direct / staged - 1.0) * 100.0,
            staged / direct,
        );
    }
    let staged_median = median(&staged_samples);
    let direct_median = median(&direct_samples);
    eprintln!(
        "operator staged_median_five_layer_ms={staged_median:.6} \
         staged_median_per_layer_ms={:.6} \
         direct_median_five_layer_ms={direct_median:.6} \
         direct_median_per_layer_ms={:.6} direct_vs_staged_median_pct={:+.3}% \
         staged_over_direct_median={:.4}x",
        staged_median / LOGICAL_LAYERS as f64,
        direct_median / LOGICAL_LAYERS as f64,
        (direct_median / staged_median - 1.0) * 100.0,
        staged_median / direct_median,
    );
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
