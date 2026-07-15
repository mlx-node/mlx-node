//! Exact-shape coverage for the grouped Qwen3.5/Qwen3.6 paged-attention
//! specialization.
//!
//! The optimized stage-1 kernel is selected only for BF16 Q/K/V with one
//! sequence, either the dense 24Q/4KV or MoE 16Q/2KV head layout, D256, BS16,
//! and either one decode query or two varlen/MTP queries. This test drives both
//! raw V2 dispatches at each layout's measured grouped-route context thresholds
//! and compares every output head/dimension against a host reference.
//!
//! The physical pool contains poisoned holes and the block table is an affine
//! permutation into that pool. This catches code that accidentally treats a
//! logical block index as a physical block index. The context also ends in a
//! partial block. The decode case enables a sliding window; the two-row case
//! checks the per-row causal effective-context boundary.
//!
//! The ignored benchmark at the bottom allocates one context at a time and
//! reports raw-dispatch latency for 16K, 64K, and 112K contexts. Run it in
//! separate processes for a grouped/generic A/B because the escape hatch is
//! cached on first dispatch:
//!
//! ```text
//! cargo test -p mlx-paged-attn --test grouped_qwen35_paged_attention \
//!   grouped_qwen35_exact_shape_sequential_benchmark -- --ignored --exact --nocapture
//! MLX_PAGED_GROUPED_QWEN35=0 cargo test -p mlx-paged-attn \
//!   --test grouped_qwen35_paged_attention \
//!   grouped_qwen35_exact_shape_sequential_benchmark -- --ignored --exact --nocapture
//! ```

#![cfg(all(target_os = "macos", mlx_node_metal_enabled))]

use std::ffi::c_void;
use std::time::{Duration, Instant};

use metal::{Buffer, MTLResourceOptions};
use mlx_paged_attn::metal::{
    MetalDtype, MetalState, PagedAttentionParams, PagedAttentionVarlenParams, RawBufferInfo,
    dispatch_paged_attention_v2_raw, dispatch_paged_attention_varlen_v2_raw,
};

const NUM_SEQS: u32 = 1;
const NUM_HEADS: u32 = 24;
const NUM_KV_HEADS: u32 = 4;
const GQA_FACTOR: usize = (NUM_HEADS / NUM_KV_HEADS) as usize;
const HEAD_SIZE: u32 = 256;
const BLOCK_SIZE: u32 = 16;
const X_PACK: usize = 8;
const SCALE: f32 = 1.0 / 16.0;

#[derive(Clone, Copy)]
struct ExactShape {
    label: &'static str,
    num_heads: u32,
    num_kv_heads: u32,
    decode_context_len: usize,
    varlen_context_len: usize,
}

const EXACT_SHAPES: [ExactShape; 2] = [
    ExactShape {
        label: "dense 24Q/4KV",
        num_heads: 24,
        num_kv_heads: 4,
        decode_context_len: 16_385,
        varlen_context_len: 8_194,
    },
    ExactShape {
        label: "MoE 16Q/2KV",
        num_heads: 16,
        num_kv_heads: 2,
        decode_context_len: 32_769,
        varlen_context_len: 16_386,
    },
];

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    bits.wrapping_add(rounding_bias).wrapping_shr(16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn round_bf16(value: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(value))
}

fn synthetic_k(token: usize, kv_head: usize, dim: usize) -> f32 {
    let phase = token as f32 * 0.013 + kv_head as f32 * 0.71 + dim as f32 * 0.017;
    round_bf16(phase.sin() * 0.42 + (phase * 0.37).cos() * 0.06)
}

fn synthetic_v(token: usize, kv_head: usize, dim: usize) -> f32 {
    let phase = token as f32 * 0.007 + kv_head as f32 * 0.53 + dim as f32 * 0.023;
    round_bf16(phase.sin() * 0.28 + kv_head as f32 * 0.035)
}

fn synthetic_q(row: usize, head: usize, dim: usize) -> f32 {
    let phase = row as f32 * 0.43 + head as f32 * 0.11 + dim as f32 * 0.019;
    round_bf16(phase.cos() * 0.36 + (phase * 0.41).sin() * 0.04)
}

struct ExactShapeInputs {
    shape: ExactShape,
    context_len: usize,
    block_table: Vec<u32>,
    q_bf16: Vec<u16>,
    q_f32: Vec<f32>,
    logical_k: Vec<f32>,
    logical_v: Vec<f32>,
    k_pool_bf16: Vec<u16>,
    v_pool_bf16: Vec<u16>,
}

impl ExactShapeInputs {
    fn new(shape: ExactShape, context_len: usize, query_rows: usize) -> Self {
        let logical_blocks = context_len.div_ceil(BLOCK_SIZE as usize);

        // Leave seven physical blocks unused and choose a pool size coprime
        // with seven so the affine mapping remains injective at every tested
        // context length.
        let mut physical_blocks = logical_blocks + 7;
        while physical_blocks.is_multiple_of(7) {
            physical_blocks += 1;
        }
        let block_table: Vec<u32> = (0..logical_blocks)
            .map(|logical| ((logical * 7 + 3) % physical_blocks) as u32)
            .collect();
        let mut uniqueness = block_table.clone();
        uniqueness.sort_unstable();
        uniqueness.dedup();
        assert_eq!(uniqueness.len(), logical_blocks);
        assert!(
            block_table
                .iter()
                .enumerate()
                .any(|(i, &p)| i != p as usize)
        );

        let logical_kv_len = context_len * shape.num_kv_heads as usize * HEAD_SIZE as usize;
        let mut logical_k = vec![0.0f32; logical_kv_len];
        let mut logical_v = vec![0.0f32; logical_kv_len];

        let elements_per_physical_block =
            shape.num_kv_heads as usize * HEAD_SIZE as usize * BLOCK_SIZE as usize;
        // Poison every unused slot. If the kernel ignores the block table or
        // reads beyond the partial tail, the result diverges dramatically.
        let poison = f32_to_bf16_bits(37.0);
        let mut k_pool_bf16 = vec![poison; physical_blocks * elements_per_physical_block];
        let mut v_pool_bf16 = vec![poison; physical_blocks * elements_per_physical_block];

        for token in 0..context_len {
            let logical_block = token / BLOCK_SIZE as usize;
            let block_offset = token % BLOCK_SIZE as usize;
            let physical_block = block_table[logical_block] as usize;
            for kv_head in 0..shape.num_kv_heads as usize {
                for dim in 0..HEAD_SIZE as usize {
                    let logical_idx =
                        (token * shape.num_kv_heads as usize + kv_head) * HEAD_SIZE as usize + dim;
                    let k = synthetic_k(token, kv_head, dim);
                    let v = synthetic_v(token, kv_head, dim);
                    logical_k[logical_idx] = k;
                    logical_v[logical_idx] = v;

                    // K: [physical_block, kv_head, D/8, BS16, 8].
                    let k_idx = physical_block * elements_per_physical_block
                        + kv_head * HEAD_SIZE as usize * BLOCK_SIZE as usize
                        + (dim / X_PACK) * BLOCK_SIZE as usize * X_PACK
                        + block_offset * X_PACK
                        + dim % X_PACK;
                    // V: [physical_block, kv_head, D, BS16].
                    let v_idx = physical_block * elements_per_physical_block
                        + kv_head * HEAD_SIZE as usize * BLOCK_SIZE as usize
                        + dim * BLOCK_SIZE as usize
                        + block_offset;
                    k_pool_bf16[k_idx] = f32_to_bf16_bits(k);
                    v_pool_bf16[v_idx] = f32_to_bf16_bits(v);
                }
            }
        }

        let q_len = query_rows * shape.num_heads as usize * HEAD_SIZE as usize;
        let mut q_f32 = Vec::with_capacity(q_len);
        let mut q_bf16 = Vec::with_capacity(q_len);
        for row in 0..query_rows {
            for head in 0..shape.num_heads as usize {
                for dim in 0..HEAD_SIZE as usize {
                    let q = synthetic_q(row, head, dim);
                    q_f32.push(q);
                    q_bf16.push(f32_to_bf16_bits(q));
                }
            }
        }

        Self {
            shape,
            context_len,
            block_table,
            q_bf16,
            q_f32,
            logical_k,
            logical_v,
            k_pool_bf16,
            v_pool_bf16,
        }
    }

    fn host_reference(&self, query_rows: usize, sliding_window: i32) -> Vec<f32> {
        let mut output =
            vec![0.0f32; query_rows * self.shape.num_heads as usize * HEAD_SIZE as usize];
        let gqa_factor = (self.shape.num_heads / self.shape.num_kv_heads) as usize;

        for row in 0..query_rows {
            let effective_context = self.context_len - query_rows + row + 1;
            let lower = if sliding_window > 0 && effective_context > sliding_window as usize {
                effective_context - sliding_window as usize
            } else {
                0
            };

            for head in 0..self.shape.num_heads as usize {
                let kv_head = head / gqa_factor;
                let q_start = (row * self.shape.num_heads as usize + head) * HEAD_SIZE as usize;
                let q = &self.q_f32[q_start..q_start + HEAD_SIZE as usize];
                let mut scores = Vec::with_capacity(effective_context - lower);
                for token in lower..effective_context {
                    let kv_start =
                        (token * self.shape.num_kv_heads as usize + kv_head) * HEAD_SIZE as usize;
                    let k = &self.logical_k[kv_start..kv_start + HEAD_SIZE as usize];
                    let dot = q.iter().zip(k).map(|(&qv, &kv)| qv * kv).sum::<f32>();
                    scores.push(dot * SCALE);
                }

                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for score in &mut scores {
                    *score = (*score - max_score).exp();
                    sum_exp += *score;
                }
                let inv_sum = 1.0 / (sum_exp + 1e-6);
                for score in &mut scores {
                    *score *= inv_sum;
                }

                let out_start = (row * self.shape.num_heads as usize + head) * HEAD_SIZE as usize;
                for dim in 0..HEAD_SIZE as usize {
                    let mut acc = 0.0f32;
                    for (weight_idx, &weight) in scores.iter().enumerate() {
                        let token = lower + weight_idx;
                        let v_idx = (token * self.shape.num_kv_heads as usize + kv_head)
                            * HEAD_SIZE as usize
                            + dim;
                        acc += weight * self.logical_v[v_idx];
                    }
                    output[out_start + dim] = acc;
                }
            }
        }

        output
    }
}

struct MetalInputs {
    key_pool: Buffer,
    value_pool: Buffer,
    q: Buffer,
    block_table: Buffer,
    context_lens: Buffer,
}

impl MetalInputs {
    fn new(state: &MetalState, inputs: &ExactShapeInputs) -> Self {
        let key_pool = state.device.new_buffer_with_slice(
            inputs.k_pool_bf16.as_ref(),
            MTLResourceOptions::StorageModeShared,
        );
        let value_pool = state.device.new_buffer_with_slice(
            inputs.v_pool_bf16.as_ref(),
            MTLResourceOptions::StorageModeShared,
        );
        let q = state.device.new_buffer_with_slice(
            inputs.q_bf16.as_ref(),
            MTLResourceOptions::StorageModeShared,
        );
        let block_table = state.device.new_buffer_with_slice(
            inputs.block_table.as_ref(),
            MTLResourceOptions::StorageModeShared,
        );
        let context_lens = [inputs.context_len as u32];
        let context_lens = state
            .device
            .new_buffer_with_slice(context_lens.as_ref(), MTLResourceOptions::StorageModeShared);
        Self {
            key_pool,
            value_pool,
            q,
            block_table,
            context_lens,
        }
    }
}

fn common_strides(num_heads: u32, num_kv_heads: u32) -> (i32, i32, i32) {
    (
        (num_heads * HEAD_SIZE) as i32,
        (num_kv_heads * HEAD_SIZE * BLOCK_SIZE) as i32,
        (HEAD_SIZE * BLOCK_SIZE) as i32,
    )
}

fn read_bf16_output(state: &MetalState, source: &Buffer, elements: usize) -> Vec<f32> {
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

fn dispatch_decode(
    state: &MetalState,
    inputs: &ExactShapeInputs,
    sliding_window: i32,
) -> (Vec<f32>, bool) {
    let metal = MetalInputs::new(state, inputs);
    let (q_stride, kv_block_stride, kv_head_stride) =
        common_strides(inputs.shape.num_heads, inputs.shape.num_kv_heads);
    let params = PagedAttentionParams {
        num_seqs: NUM_SEQS,
        num_heads: inputs.shape.num_heads,
        num_kv_heads: inputs.shape.num_kv_heads,
        head_size: HEAD_SIZE,
        block_size: BLOCK_SIZE,
        max_seq_len: inputs.context_len as u32,
        max_num_blocks_per_seq: inputs.block_table.len() as u32,
        scale: SCALE,
        softcapping: 1.0,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        k_scale: 1.0,
        v_scale: 1.0,
        sliding_window,
    };
    let q = RawBufferInfo {
        ptr: metal.q.as_ptr() as *mut c_void,
        offset: 0,
    };
    let output = unsafe {
        dispatch_paged_attention_v2_raw(
            &q,
            &metal.key_pool,
            &metal.value_pool,
            &metal.block_table,
            &metal.context_lens,
            &params,
            MetalDtype::BFloat16,
            MetalDtype::BFloat16,
        )
    }
    .expect("exact-shape decode V2 dispatch must succeed");
    let used_grouped = output.used_grouped_qwen35;
    (
        read_bf16_output(
            state,
            &output.buffer,
            inputs.shape.num_heads as usize * HEAD_SIZE as usize,
        ),
        used_grouped,
    )
}

fn dispatch_varlen_two_rows(
    state: &MetalState,
    inputs: &ExactShapeInputs,
    sliding_window: i32,
) -> (Vec<f32>, bool) {
    let metal = MetalInputs::new(state, inputs);
    let cu_seqlens = [0i32, 2];
    let cu_seqlens = state
        .device
        .new_buffer_with_slice(cu_seqlens.as_ref(), MTLResourceOptions::StorageModeShared);
    let (q_stride, kv_block_stride, kv_head_stride) =
        common_strides(inputs.shape.num_heads, inputs.shape.num_kv_heads);
    let params = PagedAttentionVarlenParams {
        num_seqs: NUM_SEQS,
        total_queries: 2,
        num_heads: inputs.shape.num_heads,
        num_kv_heads: inputs.shape.num_kv_heads,
        head_size: HEAD_SIZE,
        block_size: BLOCK_SIZE,
        max_seq_len: inputs.context_len as u32,
        max_num_blocks_per_seq: inputs.block_table.len() as u32,
        scale: SCALE,
        softcapping: 1.0,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        k_scale: 1.0,
        v_scale: 1.0,
        sliding_window,
    };
    let q = RawBufferInfo {
        ptr: metal.q.as_ptr() as *mut c_void,
        offset: 0,
    };
    let output = unsafe {
        dispatch_paged_attention_varlen_v2_raw(
            &q,
            &metal.key_pool,
            &metal.value_pool,
            &metal.block_table,
            &metal.context_lens,
            &cu_seqlens,
            &params,
            MetalDtype::BFloat16,
            MetalDtype::BFloat16,
        )
    }
    .expect("exact-shape two-row varlen V2 dispatch must succeed");
    let used_grouped = output.used_grouped_qwen35;
    (
        read_bf16_output(
            state,
            &output.buffer,
            2 * inputs.shape.num_heads as usize * HEAD_SIZE as usize,
        ),
        used_grouped,
    )
}

fn assert_matches_host(label: &str, num_heads: u32, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let mut worst = (0.0f32, 0usize);
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual_value - expected_value).abs();
        if !actual_value.is_finite() || !diff.is_finite() {
            panic!(
                "{label}: non-finite output at flat index {index}: actual={actual_value}, \
                 expected={expected_value}, abs_diff={diff}"
            );
        }
        if diff > worst.0 {
            worst = (diff, index);
        }
    }
    // Stage 1 writes each partition's output to BF16 before stage 2 combines
    // partitions, then stage 2 rounds to BF16 once more. The synthetic values
    // stay below one, so 2e-2 is generous while still catching wrong heads,
    // blocks, masks, or layouts by a wide margin.
    const ABS_TOLERANCE: f32 = 2.0e-2;
    assert!(
        worst.0 <= ABS_TOLERANCE,
        "{label}: worst mismatch at flat index {} (row={}, head={}, dim={}): actual={}, \
         expected={}, abs_diff={} > {}",
        worst.1,
        worst.1 / (num_heads as usize * HEAD_SIZE as usize),
        (worst.1 / HEAD_SIZE as usize) % num_heads as usize,
        worst.1 % HEAD_SIZE as usize,
        actual[worst.1],
        expected[worst.1],
        worst.0,
        ABS_TOLERANCE,
    );
}

#[test]
fn grouped_qwen35_decode_and_varlen_match_host_reference() {
    let state = match MetalState::get() {
        Ok(state) => state,
        Err(error) if error.contains("No Metal device found") => {
            eprintln!("skipping grouped exact-shape test: {error}");
            return;
        }
        Err(error) => panic!("unexpected MetalState::get failure: {error}"),
    };

    for shape in EXACT_SHAPES {
        // Each q_len=1 context is one token past its measured grouped-route
        // threshold and one token into a physical block. Scope every large
        // pool so no two exact-shape fixtures coexist in memory.
        {
            let decode_inputs = ExactShapeInputs::new(shape, shape.decode_context_len, 1);
            let decode_expected = decode_inputs.host_reference(1, 73);
            let (decode_actual, used_grouped) = dispatch_decode(state, &decode_inputs, 73);
            assert!(
                used_grouped,
                "{} q_len=1 parity silently used the generic V2 route",
                shape.label
            );
            assert_matches_host(
                &format!("{} q_len=1 sliding decode", shape.label),
                shape.num_heads,
                &decode_actual,
                &decode_expected,
            );
        }

        // The q_len=2 context is two tokens past its route threshold. Row 0
        // and row 1 have distinct causal tails; the same bounded window still
        // crosses several physical pages.
        let varlen_inputs = ExactShapeInputs::new(shape, shape.varlen_context_len, 2);
        let varlen_expected = varlen_inputs.host_reference(2, 73);
        let (varlen_actual, used_grouped) = dispatch_varlen_two_rows(state, &varlen_inputs, 73);
        assert!(
            used_grouped,
            "{} q_len=2 parity silently used the generic V2 route",
            shape.label
        );
        assert_matches_host(
            &format!("{} q_len=2 causal varlen", shape.label),
            shape.num_heads,
            &varlen_actual,
            &varlen_expected,
        );

        let row_size = shape.num_heads as usize * HEAD_SIZE as usize;
        let row_difference = varlen_actual[..row_size]
            .iter()
            .zip(&varlen_actual[row_size..])
            .map(|(&left, &right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(
            row_difference > 1.0e-3,
            "{} causal rows unexpectedly produced identical output",
            shape.label
        );
    }
}

fn full_context_block_numerator(logical_block: usize) -> i32 {
    ((logical_block * 7) % 29) as i32 - 14
}

fn full_context_v_value(
    logical_block: usize,
    block_offset: usize,
    kv_head: usize,
    dim: usize,
) -> f32 {
    // Every term is a multiple of 1/128 and the result stays in (-1, 1), so
    // this is exactly representable in BF16. The block and in-block terms
    // make omissions, tail mistakes, and physical-page addressing visible.
    let base_numerator = (kv_head * 8 + dim % 16) as i32 - 8;
    base_numerator as f32 / 64.0
        + full_context_block_numerator(logical_block) as f32 / 32.0
        + (block_offset as i32 - 7) as f32 / 128.0
}

fn full_context_pattern_mean(context_len: usize) -> f32 {
    let full_blocks = context_len / BLOCK_SIZE as usize;
    let tail = context_len % BLOCK_SIZE as usize;

    // `block * 7 mod 29` permutes 0..29, whose centered sum is zero.
    // Therefore only the remainder after complete 29-block cycles matters.
    let block_remainder = full_blocks % 29;
    let full_block_numerator_sum: i64 = (0..block_remainder)
        .map(|block| i64::from(full_context_block_numerator(block)))
        .sum();
    let full_offset_numerator_sum =
        (BLOCK_SIZE as i64 * (BLOCK_SIZE as i64 - 1)) / 2 - 7 * BLOCK_SIZE as i64;
    let tail_offset_numerator_sum = (tail as i64 * (tail as i64 - 1)) / 2 - 7 * tail as i64;

    let sum = full_block_numerator_sum as f64 * BLOCK_SIZE as f64 / 32.0
        + full_blocks as f64 * full_offset_numerator_sum as f64 / 128.0
        + tail as f64 * f64::from(full_context_block_numerator(full_blocks)) / 32.0
        + tail_offset_numerator_sum as f64 / 128.0;
    (sum / context_len as f64) as f32
}

fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    left
}

fn run_full_context_stage1_case(state: &MetalState, context_len: usize) {
    let logical_blocks = context_len.div_ceil(BLOCK_SIZE as usize);
    let mut physical_blocks = logical_blocks + 17;
    while gcd(physical_blocks, 13) != 1 {
        physical_blocks += 1;
    }
    let block_table: Vec<u32> = (0..logical_blocks)
        .map(|logical| ((logical * 13 + 5) % physical_blocks) as u32)
        .collect();
    let mut uniqueness = block_table.clone();
    uniqueness.sort_unstable();
    uniqueness.dedup();
    assert_eq!(uniqueness.len(), logical_blocks);
    assert!(
        block_table
            .iter()
            .enumerate()
            .any(|(logical, &physical)| logical != physical as usize)
    );

    let elements_per_physical_block =
        NUM_KV_HEADS as usize * HEAD_SIZE as usize * BLOCK_SIZE as usize;
    let pool_elements = physical_blocks * elements_per_physical_block;
    let pool_bytes = pool_elements * std::mem::size_of::<u16>();
    let key_pool = state
        .device
        .new_buffer(pool_bytes as u64, MTLResourceOptions::StorageModeShared);
    let value_pool = state
        .device
        .new_buffer(pool_bytes as u64, MTLResourceOptions::StorageModeShared);

    // Allocate directly in shared Metal storage so the 65K fixture does not
    // retain duplicate 128 MiB host vectors. Unmapped pages and partial-block
    // tails stay poisoned; only logical tokens are overwritten below.
    let key_bits =
        unsafe { std::slice::from_raw_parts_mut(key_pool.contents() as *mut u16, pool_elements) };
    let value_bits =
        unsafe { std::slice::from_raw_parts_mut(value_pool.contents() as *mut u16, pool_elements) };
    key_bits.fill(u16::MAX);
    value_bits.fill(u16::MAX);

    let mut value_pattern = vec![0u16; 29 * elements_per_physical_block];
    for block_class in 0..29 {
        for kv_head in 0..NUM_KV_HEADS as usize {
            for dim in 0..HEAD_SIZE as usize {
                for block_offset in 0..BLOCK_SIZE as usize {
                    let value = full_context_v_value(block_class, block_offset, kv_head, dim);
                    let bits = f32_to_bf16_bits(value);
                    assert_eq!(
                        bf16_bits_to_f32(bits),
                        value,
                        "fixture value must be exactly BF16"
                    );
                    let pattern_idx = block_class * elements_per_physical_block
                        + kv_head * HEAD_SIZE as usize * BLOCK_SIZE as usize
                        + dim * BLOCK_SIZE as usize
                        + block_offset;
                    value_pattern[pattern_idx] = bits;
                }
            }
        }
    }

    for (logical_block, &physical_block) in block_table.iter().enumerate() {
        let physical_block = physical_block as usize;
        let physical_start = physical_block * elements_per_physical_block;
        let block_start_token = logical_block * BLOCK_SIZE as usize;
        let valid_tokens = (context_len - block_start_token).min(BLOCK_SIZE as usize);

        if valid_tokens == BLOCK_SIZE as usize {
            key_bits[physical_start..physical_start + elements_per_physical_block].fill(0);
        } else {
            for kv_head in 0..NUM_KV_HEADS as usize {
                for dim in 0..HEAD_SIZE as usize {
                    for block_offset in 0..valid_tokens {
                        let k_idx = physical_start
                            + kv_head * HEAD_SIZE as usize * BLOCK_SIZE as usize
                            + (dim / X_PACK) * BLOCK_SIZE as usize * X_PACK
                            + block_offset * X_PACK
                            + dim % X_PACK;
                        key_bits[k_idx] = 0;
                    }
                }
            }
        }

        for kv_head in 0..NUM_KV_HEADS as usize {
            for dim in 0..HEAD_SIZE as usize {
                for block_offset in 0..valid_tokens {
                    let pattern_idx = (logical_block % 29) * elements_per_physical_block
                        + kv_head * HEAD_SIZE as usize * BLOCK_SIZE as usize
                        + dim * BLOCK_SIZE as usize
                        + block_offset;
                    let v_idx = physical_start
                        + kv_head * HEAD_SIZE as usize * BLOCK_SIZE as usize
                        + dim * BLOCK_SIZE as usize
                        + block_offset;
                    value_bits[v_idx] = value_pattern[pattern_idx];
                }
            }
        }
    }

    let q = zeroed_shared_buffer(
        state,
        NUM_HEADS as usize * HEAD_SIZE as usize * std::mem::size_of::<u16>(),
    );
    let block_table_buffer = state
        .device
        .new_buffer_with_slice(block_table.as_ref(), MTLResourceOptions::StorageModeShared);
    let context_lens = [context_len as u32];
    let context_lens_buffer = state
        .device
        .new_buffer_with_slice(context_lens.as_ref(), MTLResourceOptions::StorageModeShared);
    let q_raw = RawBufferInfo {
        ptr: q.as_ptr() as *mut c_void,
        offset: 0,
    };
    let (q_stride, kv_block_stride, kv_head_stride) = common_strides(NUM_HEADS, NUM_KV_HEADS);
    let params = PagedAttentionParams {
        num_seqs: NUM_SEQS,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_size: HEAD_SIZE,
        block_size: BLOCK_SIZE,
        max_seq_len: context_len as u32,
        max_num_blocks_per_seq: logical_blocks as u32,
        scale: SCALE,
        softcapping: 1.0,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        k_scale: 1.0,
        v_scale: 1.0,
        sliding_window: 0,
    };
    let output = unsafe {
        dispatch_paged_attention_v2_raw(
            &q_raw,
            &key_pool,
            &value_pool,
            &block_table_buffer,
            &context_lens_buffer,
            &params,
            MetalDtype::BFloat16,
            MetalDtype::BFloat16,
        )
    }
    .expect("full-context grouped stage-1 dispatch must succeed");
    assert!(
        output.used_grouped_qwen35,
        "context={context_len} silently used generic V2"
    );
    let actual = read_bf16_output(
        state,
        &output.buffer,
        NUM_HEADS as usize * HEAD_SIZE as usize,
    );

    let pattern_mean = full_context_pattern_mean(context_len);
    let mut worst = (0.0f32, 0usize, 0.0f32);
    for head in 0..NUM_HEADS as usize {
        let kv_head = head / GQA_FACTOR;
        for dim in 0..HEAD_SIZE as usize {
            let base_numerator = (kv_head * 8 + dim % 16) as i32 - 8;
            let expected = base_numerator as f32 / 64.0 + pattern_mean;
            let index = head * HEAD_SIZE as usize + dim;
            let difference = (actual[index] - expected).abs();
            if !actual[index].is_finite() || difference > worst.0 {
                worst = (difference, index, expected);
            }
        }
    }
    const ABS_TOLERANCE: f32 = 1.2e-2;
    assert!(
        worst.0 <= ABS_TOLERANCE,
        "context={context_len}: stage-1/full-context mean mismatch at head={}, dim={}: \
         actual={}, expected={}, abs_diff={} > {}",
        worst.1 / HEAD_SIZE as usize,
        worst.1 % HEAD_SIZE as usize,
        actual[worst.1],
        worst.2,
        worst.0,
        ABS_TOLERANCE,
    );
    eprintln!(
        "context={context_len}: full-context stage-1 max_abs_diff={} pool={:.1} MiB/cache",
        worst.0,
        pool_bytes as f64 / (1024.0 * 1024.0),
    );
}

#[test]
fn grouped_qwen35_full_context_stage1_matches_analytic_mean() {
    let state = match MetalState::get() {
        Ok(state) => state,
        Err(error) if error.contains("No Metal device found") => {
            eprintln!("skipping grouped full-context stage-1 test: {error}");
            return;
        }
        Err(error) => panic!("unexpected MetalState::get failure: {error}"),
    };

    // These one-token-tail contexts force each production stripe-count tier:
    // S=256, S=512, and S=1024. Each fixture (including its Metal buffers and
    // auxiliary output) is dropped before the next allocation begins.
    for context_len in [16_385, 32_769, 65_537] {
        run_full_context_stage1_case(state, context_len);
    }
}

fn zeroed_shared_buffer(state: &MetalState, bytes: usize) -> Buffer {
    let buffer = state
        .device
        .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    unsafe { std::ptr::write_bytes(buffer.contents() as *mut u8, 0, bytes) };
    buffer
}

fn average_duration(total: Duration, iterations: u32) -> Duration {
    Duration::from_secs_f64(total.as_secs_f64() / iterations as f64)
}

fn benchmark_context(state: &MetalState, context_len: u32) {
    let logical_blocks = context_len.div_ceil(BLOCK_SIZE);
    let elements_per_pool =
        logical_blocks as usize * NUM_KV_HEADS as usize * HEAD_SIZE as usize * BLOCK_SIZE as usize;
    let pool_bytes = elements_per_pool * std::mem::size_of::<u16>();

    // Allocate and release each context inside this function. The caller runs
    // contexts sequentially, so the 112K case never overlaps another pool.
    let key_pool = zeroed_shared_buffer(state, pool_bytes);
    let value_pool = zeroed_shared_buffer(state, pool_bytes);
    let block_table: Vec<u32> = (0..logical_blocks).collect();
    let block_table = state
        .device
        .new_buffer_with_slice(block_table.as_ref(), MTLResourceOptions::StorageModeShared);
    let context_lens = [context_len];
    let context_lens = state
        .device
        .new_buffer_with_slice(context_lens.as_ref(), MTLResourceOptions::StorageModeShared);
    let q_elements = 2 * NUM_HEADS as usize * HEAD_SIZE as usize;
    let q = zeroed_shared_buffer(state, q_elements * std::mem::size_of::<u16>());
    let q_raw = RawBufferInfo {
        ptr: q.as_ptr() as *mut c_void,
        offset: 0,
    };
    let (q_stride, kv_block_stride, kv_head_stride) = common_strides(NUM_HEADS, NUM_KV_HEADS);

    let decode_params = PagedAttentionParams {
        num_seqs: 1,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_size: HEAD_SIZE,
        block_size: BLOCK_SIZE,
        max_seq_len: context_len,
        max_num_blocks_per_seq: logical_blocks,
        scale: SCALE,
        softcapping: 1.0,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        k_scale: 1.0,
        v_scale: 1.0,
        sliding_window: 0,
    };
    let varlen_params = PagedAttentionVarlenParams {
        num_seqs: 1,
        total_queries: 2,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_size: HEAD_SIZE,
        block_size: BLOCK_SIZE,
        max_seq_len: context_len,
        max_num_blocks_per_seq: logical_blocks,
        scale: SCALE,
        softcapping: 1.0,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        k_scale: 1.0,
        v_scale: 1.0,
        sliding_window: 0,
    };
    let cu_seqlens = [0i32, 2];
    let cu_seqlens = state
        .device
        .new_buffer_with_slice(cu_seqlens.as_ref(), MTLResourceOptions::StorageModeShared);

    let run_decode = || unsafe {
        dispatch_paged_attention_v2_raw(
            &q_raw,
            &key_pool,
            &value_pool,
            &block_table,
            &context_lens,
            &decode_params,
            MetalDtype::BFloat16,
            MetalDtype::BFloat16,
        )
        .expect("benchmark decode dispatch")
    };
    let run_varlen = || unsafe {
        dispatch_paged_attention_varlen_v2_raw(
            &q_raw,
            &key_pool,
            &value_pool,
            &block_table,
            &context_lens,
            &cu_seqlens,
            &varlen_params,
            MetalDtype::BFloat16,
            MetalDtype::BFloat16,
        )
        .expect("benchmark varlen dispatch")
    };

    // Pipeline compile and first-touch residency are not part of the timed
    // dispatch. The APIs synchronously wait for both stage 1 and stage 2.
    drop(run_decode());
    drop(run_varlen());

    const ITERATIONS: u32 = 5;
    let decode_start = Instant::now();
    for _ in 0..ITERATIONS {
        drop(run_decode());
    }
    let decode_average = average_duration(decode_start.elapsed(), ITERATIONS);

    let varlen_start = Instant::now();
    for _ in 0..ITERATIONS {
        drop(run_varlen());
    }
    let varlen_average = average_duration(varlen_start.elapsed(), ITERATIONS);

    println!(
        "context={context_len:>6} pool={:.1} MiB/cache q1={:.3} ms q2={:.3} ms",
        pool_bytes as f64 / (1024.0 * 1024.0),
        decode_average.as_secs_f64() * 1_000.0,
        varlen_average.as_secs_f64() * 1_000.0,
    );
}

#[test]
#[ignore = "manual sequential Metal microbenchmark; run exact test in its own process"]
fn grouped_qwen35_exact_shape_sequential_benchmark() {
    let state = match MetalState::get() {
        Ok(state) => state,
        Err(error) if error.contains("No Metal device found") => {
            eprintln!("skipping grouped exact-shape benchmark: {error}");
            return;
        }
        Err(error) => panic!("unexpected MetalState::get failure: {error}"),
    };
    println!(
        "MLX_PAGED_GROUPED_QWEN35={:?}",
        std::env::var("MLX_PAGED_GROUPED_QWEN35").ok()
    );
    for context_len in [
        1024,
        4 * 1024,
        8 * 1024,
        12 * 1024,
        16 * 1024,
        32 * 1024,
        64 * 1024,
        112 * 1024,
    ] {
        benchmark_context(state, context_len);
    }
}
