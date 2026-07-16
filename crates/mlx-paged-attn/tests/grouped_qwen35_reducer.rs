//! Model-free numerical parity for the grouped Qwen3.5/Qwen3.6 D256 and
//! Gemma 4 D512 striped-attention reducers.
//!
//! The stage-1 grouped kernel emits one unnormalized BF16 numerator plus
//! an f32 local maximum and exponential sum per stripe. The dedicated reducer
//! must reconstruct the global softmax result across 32 SIMD-group stripe
//! classes. This test covers both head sizes and Gemma's diagnostic 4/8/16
//! stripe counts, including deterministic empty-stripe sentinels.

#![cfg(all(target_os = "macos", mlx_node_metal_enabled))]

use metal::{MTLResourceOptions, MTLSize};
use mlx_paged_attn::metal::MetalState;

const NUM_HEADS: usize = 3;
const QUERY_ROWS: usize = 2;
const NUM_ROWS: usize = NUM_HEADS * QUERY_ROWS;
const REDUCE_THREADS: u64 = 1024;

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn is_empty_stripe(row: usize, stripe: usize) -> bool {
    // Mix a regular cadence with one row-dependent cadence so every row has
    // empty stripes, including entries in different SIMD lane/classes.
    (stripe + row * 11).is_multiple_of(37) || (stripe * 3 + row).is_multiple_of(113)
}

struct ReducerInputs {
    exp_sums: Vec<f32>,
    max_logits: Vec<f32>,
    partials_bf16: Vec<u16>,
}

impl ReducerInputs {
    fn new(num_stripes: usize, head_size: usize) -> Self {
        let mut exp_sums = vec![0.0f32; NUM_ROWS * num_stripes];
        let mut max_logits = vec![-f32::MAX; NUM_ROWS * num_stripes];
        let mut partials_bf16 = vec![0u16; NUM_ROWS * num_stripes * head_size];

        for row in 0..NUM_ROWS {
            for stripe in 0..num_stripes {
                let stats_index = row * num_stripes + stripe;
                let partial_start = stats_index * head_size;
                if is_empty_stripe(row, stripe) {
                    // Stage 1 represents an empty stripe with this exact pair.
                    // Poison the associated numerator: the reducer must give it
                    // zero weight rather than accidentally consuming it.
                    exp_sums[stats_index] = 0.0;
                    max_logits[stats_index] = -f32::MAX;
                    partials_bf16[partial_start..partial_start + head_size]
                        .fill(f32_to_bf16_bits(29.0 + row as f32));
                    continue;
                }

                let max_bucket = ((stripe * 17 + row * 29) % 101) as f32;
                let local_max = (max_bucket - 50.0) * 0.041 + row as f32 * 0.013;
                let local_sum = 0.25 + ((stripe * 13 + row * 7) % 47) as f32 * 0.03125;
                max_logits[stats_index] = local_max;
                exp_sums[stats_index] = local_sum;

                for dim in 0..head_size {
                    let phase = stripe as f32 * 0.019 + row as f32 * 0.31 + dim as f32 * 0.027;
                    // An unnormalized local softmax numerator should generally
                    // scale with its local exponential sum. Round here so the
                    // host reference sees exactly the BF16 values read by Metal.
                    let partial = local_sum * (phase.sin() * 0.37 + (phase * 0.43).cos() * 0.08);
                    partials_bf16[partial_start + dim] = f32_to_bf16_bits(partial);
                }
            }
        }

        Self {
            exp_sums,
            max_logits,
            partials_bf16,
        }
    }

    fn host_reference(&self, num_stripes: usize, head_size: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; NUM_ROWS * head_size];
        for row in 0..NUM_ROWS {
            let stats_start = row * num_stripes;
            let stats_end = stats_start + num_stripes;
            let row_maxs = &self.max_logits[stats_start..stats_end];
            let row_sums = &self.exp_sums[stats_start..stats_end];
            let global_max = row_maxs.iter().copied().fold(-f32::MAX, f32::max);

            let mut factors = Vec::with_capacity(num_stripes);
            let mut global_sum = 0.0f32;
            for stripe in 0..num_stripes {
                let factor = (row_maxs[stripe] - global_max).exp();
                factors.push(factor);
                global_sum += factor * row_sums[stripe];
            }
            assert!(global_sum.is_finite() && global_sum > 0.0);

            for dim in 0..head_size {
                let mut numerator = 0.0f32;
                for (stripe, &factor) in factors.iter().enumerate() {
                    let partial_index = (stats_start + stripe) * head_size + dim;
                    let partial = bf16_bits_to_f32(self.partials_bf16[partial_index]);
                    numerator += factor * partial;
                }
                output[row * head_size + dim] = numerator / global_sum;
            }
        }
        output
    }
}

fn run_reducer_case(state: &MetalState, head_size: usize, num_stripes: usize) {
    let inputs = ReducerInputs::new(num_stripes, head_size);
    let expected = inputs.host_reference(num_stripes, head_size);

    let output_bytes = NUM_ROWS * head_size * std::mem::size_of::<u16>();
    let output = state
        .device
        .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
    let exp_sums = state.device.new_buffer_with_slice(
        inputs.exp_sums.as_ref(),
        MTLResourceOptions::StorageModeShared,
    );
    let max_logits = state.device.new_buffer_with_slice(
        inputs.max_logits.as_ref(),
        MTLResourceOptions::StorageModeShared,
    );
    let partials = state.device.new_buffer_with_slice(
        inputs.partials_bf16.as_ref(),
        MTLResourceOptions::StorageModeShared,
    );
    let context_lens = [num_stripes as u32; QUERY_ROWS];
    let context_lens = state
        .device
        .new_buffer_with_slice(context_lens.as_ref(), MTLResourceOptions::StorageModeShared);
    let num_stripes_i32 = num_stripes as i32;
    let num_stripes_buffer = state
        .device
        .new_buffer_with_value(&num_stripes_i32, MTLResourceOptions::StorageModeShared);

    let pipeline_name = match head_size {
        256 => MetalState::paged_attention_grouped_qwen35_reduce_kernel_name(),
        512 => MetalState::paged_attention_grouped_gemma4_reduce_kernel_name(),
        _ => panic!("unsupported grouped reducer head size {head_size}"),
    };
    let pipeline = state
        .get_pipeline(pipeline_name)
        .expect("grouped striped reducer pipeline must load");
    let command_buffer = state.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&output), 0);
    encoder.set_buffer(1, Some(&exp_sums), 0);
    encoder.set_buffer(2, Some(&max_logits), 0);
    encoder.set_buffer(3, Some(&partials), 0);
    encoder.set_buffer(4, Some(&context_lens), 0);
    encoder.set_buffer(5, Some(&num_stripes_buffer), 0);
    encoder.dispatch_thread_groups(
        MTLSize::new(NUM_HEADS as u64, QUERY_ROWS as u64, 1),
        MTLSize::new(REDUCE_THREADS, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let actual_bits = unsafe {
        std::slice::from_raw_parts(output.contents() as *const u16, NUM_ROWS * head_size)
    };
    let actual: Vec<f32> = actual_bits.iter().copied().map(bf16_bits_to_f32).collect();

    let mut worst = (0.0f32, 0usize);
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(&expected).enumerate() {
        let difference = (actual_value - expected_value).abs();
        if difference > worst.0 {
            worst = (difference, index);
        }
    }

    // Metal uses fast::exp and reduces in SIMD/stripe-class order, while the
    // host walks stripes linearly. The final value is rounded to BF16. This
    // mixed absolute/relative bound is tight enough to catch missing stripe
    // classes, row-indexing mistakes, and bad empty-stripe handling.
    let expected_value = expected[worst.1];
    let tolerance = 3.0e-3 + expected_value.abs() * 5.0e-3;
    assert!(
        worst.0 <= tolerance,
        "S={num_stripes}: mismatch at flat index {} (query_row={}, head={}, dim={}): \
         actual={}, expected={}, abs_diff={}, tolerance={}",
        worst.1,
        worst.1 / (NUM_HEADS * head_size),
        (worst.1 / head_size) % NUM_HEADS,
        worst.1 % head_size,
        actual[worst.1],
        expected_value,
        worst.0,
        tolerance,
    );
    eprintln!(
        "D={head_size} S={num_stripes}: max_abs_diff={} at flat index {}",
        worst.0, worst.1,
    );
}

#[test]
fn grouped_striped_reducers_match_host_reference() {
    let state = match MetalState::get() {
        Ok(state) => state,
        Err(error) if error.contains("No Metal device found") => {
            eprintln!("skipping grouped striped reducer test: {error}");
            return;
        }
        Err(error) => panic!("unexpected MetalState::get failure: {error}"),
    };

    // Keep allocations and GPU work sequential. Each case drops all of its
    // buffers before the next representative production stripe count begins.
    for (head_size, stripe_counts) in [
        (256, &[256, 512, 1024][..]),
        (512, &[4, 8, 16, 32, 64, 128][..]),
    ] {
        for &num_stripes in stripe_counts {
            run_reducer_case(state, head_size, num_stripes);
        }
    }
}
