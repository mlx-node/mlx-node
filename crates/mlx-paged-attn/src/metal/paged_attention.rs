//! paged_attention Metal kernel dispatch
//!
//! This kernel computes attention using the paged KV cache.
//! Supports both V1 (no partitioning) and V2 (with partitioning) modes.

use super::reshape_and_cache::RawBufferInfo;
use super::state::{MetalDtype, MetalState};
use metal::foreign_types::ForeignTypeRef;
use metal::{Buffer, BufferRef, MTLSize};
use std::ffi::c_void;

/// Convert IEEE 754 half-precision (f16) to single-precision (f32)
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal - convert to normalized f32
            let mut m = mant;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let f32_exp = ((127 - 15 + 1 + e) as u32) << 23;
            f32::from_bits((sign << 31) | f32_exp | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | 0x7F800000 | (mant << 13))
    } else {
        // Normalized
        let f32_exp = ((exp as i32 - 15 + 127) as u32) << 23;
        f32::from_bits((sign << 31) | f32_exp | (mant << 13))
    }
}

/// Convert bfloat16 (bf16) to single-precision (f32)
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    // bf16 is just the upper 16 bits of f32
    f32::from_bits((bits as u32) << 16)
}

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
    /// K scale for FP8 quantization (1.0 for non-FP8)
    pub k_scale: f32,
    /// V scale for FP8 quantization (1.0 for non-FP8)
    pub v_scale: f32,
}

impl Default for PagedAttentionParams {
    fn default() -> Self {
        Self {
            num_seqs: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_size: 128,
            block_size: 16,
            max_seq_len: 0,
            max_num_blocks_per_seq: 0,
            scale: 1.0,
            softcapping: 1.0,
            q_stride: 0,
            kv_block_stride: 0,
            kv_head_stride: 0,
            k_scale: 1.0,
            v_scale: 1.0,
        }
    }
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

    // k_scale and v_scale - use params values for FP8 support
    let k_scale_buffer = state.device.new_buffer_with_data(
        &params.k_scale as *const f32 as *const _,
        std::mem::size_of::<f32>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let v_scale_buffer = state.device.new_buffer_with_data(
        &params.v_scale as *const f32 as *const _,
        std::mem::size_of::<f32>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    encoder.set_buffer(6, Some(&k_scale_buffer), 0);
    encoder.set_buffer(7, Some(&v_scale_buffer), 0);

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

        // k_scale and v_scale - use params values for FP8 support
        let k_scale_buffer = state.device.new_buffer_with_data(
            &params.k_scale as *const f32 as *const _,
            std::mem::size_of::<f32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let v_scale_buffer = state.device.new_buffer_with_data(
            &params.v_scale as *const f32 as *const _,
            std::mem::size_of::<f32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(6, Some(&k_scale_buffer), 0);
        encoder.set_buffer(7, Some(&v_scale_buffer), 0);

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

/// Output from paged attention dispatch
///
/// Contains the output buffer and metadata for creating an MLX array.
pub struct PagedAttentionOutput {
    /// Output buffer [num_seqs, num_heads, head_size]
    pub buffer: Buffer,
    /// Number of sequences
    pub num_seqs: u32,
    /// Number of query heads
    pub num_heads: u32,
    /// Head size
    pub head_size: u32,
    /// Data type
    pub dtype: MetalDtype,
}

impl PagedAttentionOutput {
    /// Get the raw buffer pointer
    pub fn buffer_ptr(&self) -> *mut c_void {
        use metal::foreign_types::ForeignType;
        self.buffer.as_ptr() as *mut c_void
    }

    /// Get the total number of elements
    pub fn num_elements(&self) -> usize {
        (self.num_seqs * self.num_heads * self.head_size) as usize
    }

    /// Get the buffer size in bytes
    pub fn size_bytes(&self) -> usize {
        self.num_elements() * self.dtype.size()
    }

    /// Get output shape as [num_seqs, num_heads, head_size]
    pub fn shape(&self) -> [i64; 3] {
        [
            self.num_seqs as i64,
            self.num_heads as i64,
            self.head_size as i64,
        ]
    }

    /// Copy output data to host memory as f32
    ///
    /// This copies the GPU-private buffer to CPU-accessible memory and converts to f32.
    /// Use this to create an MLX array from the attention output.
    ///
    /// # Performance Note
    /// This involves a GPU->CPU copy. For best performance, consider batching
    /// multiple layers' attention outputs before copying to host.
    ///
    /// # Returns
    /// Vector of f32 values with shape [num_seqs, num_heads, head_size]
    pub fn to_host_f32(&self) -> Result<Vec<f32>, String> {
        let state = MetalState::get()?;
        let size_bytes = self.size_bytes();

        // Create shared (CPU-accessible) buffer
        let shared_buffer = state.device.new_buffer(
            size_bytes as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Blit copy from private to shared
        let command_queue = state.device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let blit_encoder = command_buffer.new_blit_command_encoder();

        blit_encoder.copy_from_buffer(&self.buffer, 0, &shared_buffer, 0, size_bytes as u64);
        blit_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read data from shared buffer
        let num_elements = self.num_elements();
        let mut result = vec![0.0f32; num_elements];

        match self.dtype {
            MetalDtype::Float32 => {
                // Direct copy for f32
                let ptr = shared_buffer.contents() as *const f32;
                unsafe {
                    std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), num_elements);
                }
            }
            MetalDtype::Float16 => {
                // Convert f16 to f32
                let ptr = shared_buffer.contents() as *const u16;
                let src = unsafe { std::slice::from_raw_parts(ptr, num_elements) };
                for (dst, &bits) in result.iter_mut().zip(src.iter()) {
                    *dst = f16_to_f32(bits);
                }
            }
            MetalDtype::BFloat16 => {
                // Convert bf16 to f32
                let ptr = shared_buffer.contents() as *const u16;
                let src = unsafe { std::slice::from_raw_parts(ptr, num_elements) };
                for (dst, &bits) in result.iter_mut().zip(src.iter()) {
                    *dst = bf16_to_f32(bits);
                }
            }
            MetalDtype::UChar => {
                // FP8 E4M3 format cannot be correctly converted to f32 without the quantization scales.
                // The raw byte cast (byte as f32) produces completely wrong values because FP8 E4M3
                // has a different bit layout (1 sign, 4 exponent, 3 mantissa bits).
                // Proper dequantization requires: f32_value = fp8_to_f32(byte) / scale
                // where scale was used during quantization.
                return Err(
                    "FP8 (UChar) to f32 conversion not yet implemented. \
                    FP8 dequantization requires k_scale/v_scale values which are not available here. \
                    Use the raw buffer directly or implement to_host_f32_with_scales() for FP8 data."
                        .to_string(),
                );
            }
        }

        Ok(result)
    }

    /// Convert output to an MLX array
    ///
    /// Creates a new MLX array from the attention output data.
    /// This involves copying data from GPU to CPU and back to MLX.
    ///
    /// # Safety
    /// The returned pointer must be managed by the caller (typically wrapped in MxArray).
    ///
    /// # Performance Note
    /// This involves GPU->CPU->GPU copies. For zero-copy integration,
    /// use the raw buffer pointer with MLX's metal operations directly.
    pub unsafe fn to_mlx_array(&self) -> Result<*mut mlx_sys::mlx_array, String> {
        let data = self.to_host_f32()?;
        let shape = self.shape();

        // SAFETY: data and shape are valid pointers with correct length
        let arr = unsafe {
            mlx_sys::mlx_array_from_float32(
                data.as_ptr(),
                shape.as_ptr(),
                3, // ndim
            )
        };

        if arr.is_null() {
            return Err("Failed to create MLX array from attention output".to_string());
        }

        Ok(arr)
    }
}

/// Dispatch paged_attention V1 with raw buffer pointers
///
/// This variant is used when integrating with MLX arrays.
///
/// # Safety
///
/// - queries must be a valid MTLBuffer* with shape [num_seqs, num_heads, head_size]
/// - key_cache and value_cache must be valid cache buffers
/// - block_tables must have shape [num_seqs, max_blocks_per_seq]
/// - context_lens must have shape [num_seqs]
///
/// # Returns
///
/// A new buffer containing the attention output [num_seqs, num_heads, head_size]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dispatch_paged_attention_v1_raw(
    queries: &RawBufferInfo,
    key_cache: &Buffer,
    value_cache: &Buffer,
    block_tables: &Buffer,
    context_lens: &Buffer,
    params: &PagedAttentionParams,
    dtype: MetalDtype,
) -> Result<PagedAttentionOutput, String> {
    let state = MetalState::get()?;

    // Allocate output buffer - always use float16 size since kernel outputs float16
    // even when using FP8 cache (kernel dequantizes internally)
    let output_element_size = MetalDtype::Float16.size() as u64;
    let output_size =
        (params.num_seqs * params.num_heads * params.head_size) as u64 * output_element_size;
    let output = state
        .device
        .new_buffer(output_size, metal::MTLResourceOptions::StorageModePrivate);

    // Get V1 pipeline (partition_size = 0)
    let kernel_name = MetalState::paged_attention_v1_kernel_name(
        dtype,
        params.head_size,
        params.block_size,
        false, // use_alibi
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

    // Convert queries raw pointer to BufferRef
    // SAFETY: Caller guarantees queries.ptr is a valid MTLBuffer*
    let queries_ref: &BufferRef = unsafe { ForeignTypeRef::from_ptr(queries.ptr as *mut _) };

    // Set buffers
    encoder.set_buffer(0, Some(&dummy_buffer), 0); // exp_sums (unused)
    encoder.set_buffer(1, Some(&dummy_buffer), 0); // max_logits (unused)
    encoder.set_buffer(2, Some(&output), 0);
    encoder.set_buffer(3, Some(queries_ref), queries.offset as u64);
    encoder.set_buffer(4, Some(key_cache), 0);
    encoder.set_buffer(5, Some(value_cache), 0);

    // k_scale and v_scale - use params values for FP8 support
    let k_scale_buffer = state.device.new_buffer_with_data(
        &params.k_scale as *const f32 as *const _,
        std::mem::size_of::<f32>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let v_scale_buffer = state.device.new_buffer_with_data(
        &params.v_scale as *const f32 as *const _,
        std::mem::size_of::<f32>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    encoder.set_buffer(6, Some(&k_scale_buffer), 0);
    encoder.set_buffer(7, Some(&v_scale_buffer), 0);

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

    // Output is always float16 regardless of cache dtype
    Ok(PagedAttentionOutput {
        buffer: output,
        num_seqs: params.num_seqs,
        num_heads: params.num_heads,
        head_size: params.head_size,
        dtype: MetalDtype::Float16,
    })
}

/// Dispatch paged_attention V2 with raw buffer pointers (for long sequences)
///
/// This variant uses partitioning for sequences longer than PARTITION_SIZE tokens.
///
/// # Safety
///
/// Same requirements as V1.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dispatch_paged_attention_v2_raw(
    queries: &RawBufferInfo,
    key_cache: &Buffer,
    value_cache: &Buffer,
    block_tables: &Buffer,
    context_lens: &Buffer,
    params: &PagedAttentionParams,
    dtype: MetalDtype,
) -> Result<PagedAttentionOutput, String> {
    let state = MetalState::get()?;

    // Allocate output buffer - always use float16 size since kernel outputs float16
    // even when using FP8 cache (kernel dequantizes internally)
    let output_element_size = MetalDtype::Float16.size() as u64;
    let output_size =
        (params.num_seqs * params.num_heads * params.head_size) as u64 * output_element_size;
    let output = state
        .device
        .new_buffer(output_size, metal::MTLResourceOptions::StorageModePrivate);

    // Calculate number of partitions
    let max_num_partitions = params.max_seq_len.div_ceil(PARTITION_SIZE);

    // Allocate temporary buffers - tmp_out also uses float16 (kernel output dtype)
    let exp_sums_size = (params.num_seqs * params.num_heads * max_num_partitions) as usize
        * std::mem::size_of::<f32>();
    let max_logits_size = exp_sums_size;
    let tmp_out_size = (params.num_seqs * params.num_heads * max_num_partitions * params.head_size)
        as usize
        * MetalDtype::Float16.size();

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

    // Convert queries raw pointer to BufferRef
    // SAFETY: Caller guarantees queries.ptr is a valid MTLBuffer*
    let queries_ref: &BufferRef = unsafe { ForeignTypeRef::from_ptr(queries.ptr as *mut _) };

    // Phase 1: Compute partitioned attention
    {
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

        // Set buffers
        encoder.set_buffer(0, Some(&exp_sums), 0);
        encoder.set_buffer(1, Some(&max_logits), 0);
        encoder.set_buffer(2, Some(&tmp_out), 0);
        encoder.set_buffer(3, Some(queries_ref), queries.offset as u64);
        encoder.set_buffer(4, Some(key_cache), 0);
        encoder.set_buffer(5, Some(value_cache), 0);

        // k_scale and v_scale - use params values for FP8 support
        let k_scale_buffer = state.device.new_buffer_with_data(
            &params.k_scale as *const f32 as *const _,
            std::mem::size_of::<f32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let v_scale_buffer = state.device.new_buffer_with_data(
            &params.v_scale as *const f32 as *const _,
            std::mem::size_of::<f32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(6, Some(&k_scale_buffer), 0);
        encoder.set_buffer(7, Some(&v_scale_buffer), 0);

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

        encoder.set_buffer(0, Some(&output), 0);
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

    // Output is always float16 regardless of cache dtype
    Ok(PagedAttentionOutput {
        buffer: output,
        num_seqs: params.num_seqs,
        num_heads: params.num_heads,
        head_size: params.head_size,
        dtype: MetalDtype::Float16,
    })
}

/// Automatically dispatch V1 or V2 based on max context length
///
/// Uses V1 for short sequences (< PARTITION_SIZE tokens), V2 for longer.
///
/// # Safety
/// - `queries` must contain a valid MTLBuffer pointer
/// - `key_cache` and `value_cache` must be valid cache buffers
/// - `block_tables` must have shape [num_seqs, max_blocks_per_seq]
/// - `context_lens` must have shape [num_seqs]
/// - All buffers must remain valid until the kernel completes
#[allow(clippy::too_many_arguments)]
pub unsafe fn dispatch_paged_attention_auto(
    queries: &RawBufferInfo,
    key_cache: &Buffer,
    value_cache: &Buffer,
    block_tables: &Buffer,
    context_lens: &Buffer,
    max_context_len: u32,
    params: &PagedAttentionParams,
    dtype: MetalDtype,
) -> Result<PagedAttentionOutput, String> {
    if max_context_len <= PARTITION_SIZE {
        // SAFETY: Caller guarantees all buffer pointers are valid
        unsafe {
            dispatch_paged_attention_v1_raw(
                queries,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                params,
                dtype,
            )
        }
    } else {
        // SAFETY: Caller guarantees all buffer pointers are valid
        unsafe {
            dispatch_paged_attention_v2_raw(
                queries,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                params,
                dtype,
            )
        }
    }
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
            k_scale: 1.0,
            v_scale: 1.0,
        };
        assert_eq!(params.num_heads / params.num_kv_heads, 6); // GQA ratio
    }

    #[test]
    fn test_fp8_params() {
        let params = PagedAttentionParams {
            k_scale: 0.5,  // FP8 quantization scale
            v_scale: 0.25, // FP8 quantization scale
            ..Default::default()
        };
        assert_eq!(params.k_scale, 0.5);
        assert_eq!(params.v_scale, 0.25);
    }
}
