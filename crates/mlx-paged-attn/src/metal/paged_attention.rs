//! paged_attention Metal kernel dispatch
//!
//! This kernel computes attention using the paged KV cache.
//! Supports both V1 (no partitioning) and V2 (with partitioning) modes.

use super::state::{MetalDtype, MetalState};
use metal::{Buffer, MTLSize};

/// Parameters for paged_attention kernel
pub struct PagedAttentionParams {
    /// Number of sequences in the batch
    pub num_seqs: u32,
    /// Number of query heads
    pub num_heads: u32,
    /// Number of KV heads
    pub num_kv_heads: u32,
    /// Size of each head
    pub head_size: u32,
    /// Block size for paged attention
    pub block_size: u32,
    /// Maximum sequence length
    pub max_seq_len: u32,
    /// Maximum number of blocks per sequence
    pub max_num_blocks_per_seq: u32,
    /// Attention scale factor (1/sqrt(head_size))
    pub scale: f32,
    /// Softcapping value (1.0 = disabled)
    pub softcapping: f32,
    /// Query stride (num_heads * head_size)
    pub q_stride: i32,
    /// KV block stride
    pub kv_block_stride: i32,
    /// KV head stride
    pub kv_head_stride: i32,
}

/// Partition size for V2 kernel
const PARTITION_SIZE: u32 = 512;

/// Dispatch paged_attention V1 kernel (no partitioning, for short sequences)
///
/// # Buffer Layout
/// - buffer(0): exp_sums [unused in V1]
/// - buffer(1): max_logits [unused in V1]
/// - buffer(2): output [num_seqs, num_heads, head_size]
/// - buffer(3): queries [num_seqs, num_heads, head_size]
/// - buffer(4): key_cache [num_blocks, num_kv_heads, head_size/x, block_size, x]
/// - buffer(5): value_cache [num_blocks, num_kv_heads, head_size, block_size]
/// - buffer(6): k_scale (unused for non-FP8)
/// - buffer(7): v_scale (unused for non-FP8)
/// - buffer(8-17): constants and block_tables/context_lens
#[allow(clippy::too_many_arguments)]
pub fn dispatch_paged_attention_v1(
    output: &Buffer,
    queries: &Buffer,
    queries_offset: usize,
    key_cache: &Buffer,
    value_cache: &Buffer,
    block_tables: &Buffer,
    context_lens: &Buffer,
    params: &PagedAttentionParams,
    dtype: MetalDtype,
) -> Result<(), String> {
    let state = MetalState::get()?;

    // Get V1 pipeline (partition_size = 0)
    // FORKED: Pass use_alibi=false since we don't support ALiBi yet
    let kernel_name = MetalState::paged_attention_v1_kernel_name(
        dtype,
        params.head_size,
        params.block_size,
        false,
    );
    let pipeline = state.get_pipeline(&kernel_name)?;

    // Create command buffer and encoder
    let command_queue = state.device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);

    // Create dummy buffers for exp_sums/max_logits (unused in V1)
    let dummy_float: f32 = 0.0;
    let dummy_buffer = state.device.new_buffer_with_data(
        &dummy_float as *const f32 as *const _,
        std::mem::size_of::<f32>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Set buffers
    encoder.set_buffer(0, Some(&dummy_buffer), 0); // exp_sums (unused)
    encoder.set_buffer(1, Some(&dummy_buffer), 0); // max_logits (unused)
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_buffer(3, Some(queries), queries_offset as u64);
    encoder.set_buffer(4, Some(key_cache), 0);
    encoder.set_buffer(5, Some(value_cache), 0);

    // k_scale and v_scale (unused for non-FP8)
    let one_f32: f32 = 1.0;
    let scale_buffer = state.device.new_buffer_with_data(
        &one_f32 as *const f32 as *const _,
        std::mem::size_of::<f32>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    encoder.set_buffer(6, Some(&scale_buffer), 0);
    encoder.set_buffer(7, Some(&scale_buffer), 0);

    // Set constant buffers
    let create_int_buffer = |value: i32| {
        state.device.new_buffer_with_data(
            &value as *const i32 as *const _,
            std::mem::size_of::<i32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        )
    };

    let create_float_buffer = |value: f32| {
        state.device.new_buffer_with_data(
            &value as *const f32 as *const _,
            std::mem::size_of::<f32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        )
    };

    let num_kv_heads_buf = create_int_buffer(params.num_kv_heads as i32);
    let scale_buf = create_float_buffer(params.scale);
    let softcapping_buf = create_float_buffer(params.softcapping);
    let max_num_blocks_buf = create_int_buffer(params.max_num_blocks_per_seq as i32);
    let q_stride_buf = create_int_buffer(params.q_stride);
    let kv_block_stride_buf = create_int_buffer(params.kv_block_stride);
    let kv_head_stride_buf = create_int_buffer(params.kv_head_stride);

    encoder.set_buffer(8, Some(&num_kv_heads_buf), 0);
    encoder.set_buffer(9, Some(&scale_buf), 0);
    encoder.set_buffer(10, Some(&softcapping_buf), 0);
    encoder.set_buffer(11, Some(block_tables), 0);
    encoder.set_buffer(12, Some(context_lens), 0);
    encoder.set_buffer(13, Some(&max_num_blocks_buf), 0);

    // alibi_slopes (unused, set to dummy)
    encoder.set_buffer(14, Some(&dummy_buffer), 0);

    encoder.set_buffer(15, Some(&q_stride_buf), 0);
    encoder.set_buffer(16, Some(&kv_block_stride_buf), 0);
    encoder.set_buffer(17, Some(&kv_head_stride_buf), 0);

    // Calculate threadgroup memory size
    // Need space for logits (max_seq_len floats) and reduction workspace
    let threadgroup_mem_size = (params.max_seq_len as usize * std::mem::size_of::<f32>())
        + (2 * 8 * std::mem::size_of::<f32>()); // 2 * NUM_WARPS * sizeof(float)
    encoder.set_threadgroup_memory_length(0, threadgroup_mem_size as u64);

    // Dispatch: (num_heads, num_seqs, 1) threadgroups, 256 threads each
    let threads_per_threadgroup = MTLSize::new(256, 1, 1);
    let threadgroups = MTLSize::new(
        params.num_heads as u64,
        params.num_seqs as u64,
        1, // No partitioning in V1
    );

    encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

/// Dispatch paged_attention V2 kernel (with partitioning, for long sequences)
///
/// This is a two-phase kernel:
/// 1. Compute partial attention for each partition
/// 2. Reduce partitions to final output
#[allow(clippy::too_many_arguments)]
pub fn dispatch_paged_attention_v2(
    output: &Buffer,
    queries: &Buffer,
    queries_offset: usize,
    key_cache: &Buffer,
    value_cache: &Buffer,
    block_tables: &Buffer,
    context_lens: &Buffer,
    params: &PagedAttentionParams,
    dtype: MetalDtype,
) -> Result<(), String> {
    let state = MetalState::get()?;

    // Calculate number of partitions
    let max_num_partitions = params.max_seq_len.div_ceil(PARTITION_SIZE);

    // Allocate temporary buffers
    let exp_sums_size = (params.num_seqs * params.num_heads * max_num_partitions) as usize
        * std::mem::size_of::<f32>();
    let max_logits_size = exp_sums_size;
    let tmp_out_size = (params.num_seqs * params.num_heads * max_num_partitions * params.head_size)
        as usize
        * dtype.size();

    let exp_sums = state.device.new_buffer(
        exp_sums_size as u64,
        metal::MTLResourceOptions::StorageModePrivate,
    );
    let max_logits = state.device.new_buffer(
        max_logits_size as u64,
        metal::MTLResourceOptions::StorageModePrivate,
    );
    let tmp_out = state.device.new_buffer(
        tmp_out_size as u64,
        metal::MTLResourceOptions::StorageModePrivate,
    );

    // Phase 1: Compute partitioned attention
    {
        // FORKED: Pass use_alibi=false since we don't support ALiBi yet
        let kernel_name = MetalState::paged_attention_v2_kernel_name(
            dtype,
            params.head_size,
            params.block_size,
            false,
        );
        let pipeline = state.get_pipeline(&kernel_name)?;

        let command_queue = state.device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);

        // Set buffers (same as V1 but with exp_sums/max_logits/tmp_out used)
        encoder.set_buffer(0, Some(&exp_sums), 0);
        encoder.set_buffer(1, Some(&max_logits), 0);
        encoder.set_buffer(2, Some(&tmp_out), 0);
        encoder.set_buffer(3, Some(queries), queries_offset as u64);
        encoder.set_buffer(4, Some(key_cache), 0);
        encoder.set_buffer(5, Some(value_cache), 0);

        // k_scale and v_scale
        let one_f32: f32 = 1.0;
        let scale_buffer = state.device.new_buffer_with_data(
            &one_f32 as *const f32 as *const _,
            std::mem::size_of::<f32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(6, Some(&scale_buffer), 0);
        encoder.set_buffer(7, Some(&scale_buffer), 0);

        // Set constant buffers
        let create_int_buffer = |value: i32| {
            state.device.new_buffer_with_data(
                &value as *const i32 as *const _,
                std::mem::size_of::<i32>() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };

        let create_float_buffer = |value: f32| {
            state.device.new_buffer_with_data(
                &value as *const f32 as *const _,
                std::mem::size_of::<f32>() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };

        let num_kv_heads_buf = create_int_buffer(params.num_kv_heads as i32);
        let scale_buf = create_float_buffer(params.scale);
        let softcapping_buf = create_float_buffer(params.softcapping);
        let max_num_blocks_buf = create_int_buffer(params.max_num_blocks_per_seq as i32);
        let q_stride_buf = create_int_buffer(params.q_stride);
        let kv_block_stride_buf = create_int_buffer(params.kv_block_stride);
        let kv_head_stride_buf = create_int_buffer(params.kv_head_stride);

        let dummy_float: f32 = 0.0;
        let dummy_buffer = state.device.new_buffer_with_data(
            &dummy_float as *const f32 as *const _,
            std::mem::size_of::<f32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(8, Some(&num_kv_heads_buf), 0);
        encoder.set_buffer(9, Some(&scale_buf), 0);
        encoder.set_buffer(10, Some(&softcapping_buf), 0);
        encoder.set_buffer(11, Some(block_tables), 0);
        encoder.set_buffer(12, Some(context_lens), 0);
        encoder.set_buffer(13, Some(&max_num_blocks_buf), 0);
        encoder.set_buffer(14, Some(&dummy_buffer), 0); // alibi_slopes
        encoder.set_buffer(15, Some(&q_stride_buf), 0);
        encoder.set_buffer(16, Some(&kv_block_stride_buf), 0);
        encoder.set_buffer(17, Some(&kv_head_stride_buf), 0);

        // Threadgroup memory
        let threadgroup_mem_size = (PARTITION_SIZE as usize * std::mem::size_of::<f32>())
            + (2 * 8 * std::mem::size_of::<f32>());
        encoder.set_threadgroup_memory_length(0, threadgroup_mem_size as u64);

        // Dispatch: (num_heads, num_seqs, max_num_partitions)
        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let threadgroups = MTLSize::new(
            params.num_heads as u64,
            params.num_seqs as u64,
            max_num_partitions as u64,
        );

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // Phase 2: Reduce partitions
    {
        let kernel_name =
            MetalState::paged_attention_v2_reduce_kernel_name(dtype, params.head_size);
        let pipeline = state.get_pipeline(&kernel_name)?;

        let command_queue = state.device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_buffer(0, Some(output), 0);
        encoder.set_buffer(1, Some(&exp_sums), 0);
        encoder.set_buffer(2, Some(&max_logits), 0);
        encoder.set_buffer(3, Some(&tmp_out), 0);
        encoder.set_buffer(4, Some(context_lens), 0);

        let max_num_partitions_buf = state.device.new_buffer_with_data(
            &(max_num_partitions as i32) as *const i32 as *const _,
            std::mem::size_of::<i32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(5, Some(&max_num_partitions_buf), 0);

        // Threadgroup memory for reduce
        let threadgroup_mem_size = 2 * (max_num_partitions as usize) * std::mem::size_of::<f32>();
        encoder.set_threadgroup_memory_length(0, threadgroup_mem_size as u64);

        // Dispatch: (num_heads, num_seqs, 1)
        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        let threadgroups = MTLSize::new(params.num_heads as u64, params.num_seqs as u64, 1);

        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params() {
        let params = PagedAttentionParams {
            num_seqs: 1,
            num_heads: 12,
            num_kv_heads: 2,
            head_size: 128,
            block_size: 16,
            max_seq_len: 2048,
            max_num_blocks_per_seq: 128,
            scale: 0.088388, // 1/sqrt(128)
            softcapping: 1.0,
            q_stride: 1536, // 12 * 128
            kv_block_stride: 32768,
            kv_head_stride: 16384,
        };
        assert_eq!(params.num_heads / params.num_kv_heads, 6); // GQA ratio
    }
}
