//! reshape_and_cache Metal kernel dispatch
//!
//! This kernel writes new KV pairs into the paged KV cache.

use super::state::{MetalDtype, MetalState};
use metal::foreign_types::ForeignTypeRef;
use metal::{Buffer, BufferRef, MTLSize};
use std::ffi::c_void;

/// Parameters for reshape_and_cache kernel
pub struct ReshapeAndCacheParams {
    /// Number of tokens being processed
    pub num_tokens: u32,
    /// Number of KV heads
    pub num_heads: u32,
    /// Size of each head
    pub head_size: u32,
    /// Block size for paged attention
    pub block_size: u32,
    /// Stride between tokens in key tensor
    pub key_stride: i32,
    /// Stride between tokens in value tensor
    pub value_stride: i32,
    /// X factor for key cache layout (typically 16 / sizeof(dtype))
    pub x: i32,
    /// K scale for FP8 quantization (1.0 for non-FP8)
    pub k_scale: f32,
    /// V scale for FP8 quantization (1.0 for non-FP8)
    pub v_scale: f32,
}

impl Default for ReshapeAndCacheParams {
    fn default() -> Self {
        Self {
            num_tokens: 0,
            num_heads: 0,
            head_size: 0,
            block_size: 0,
            key_stride: 0,
            value_stride: 0,
            x: 8,
            k_scale: 1.0,
            v_scale: 1.0,
        }
    }
}

/// Dispatch the reshape_and_cache kernel
///
/// # Buffer Layout
/// - buffer(0): key [num_tokens, num_heads, head_size]
/// - buffer(1): value [num_tokens, num_heads, head_size]
/// - buffer(2): key_cache [num_blocks, num_heads, head_size/x, block_size, x]
/// - buffer(3): value_cache [num_blocks, num_heads, head_size, block_size]
/// - buffer(4): slot_mapping [num_tokens] - int64
/// - buffer(5): k_scale [1] - float (unused when not FP8)
/// - buffer(6): v_scale [1] - float (unused when not FP8)
/// - buffer(7-12): dimension constants
#[allow(clippy::too_many_arguments)]
pub fn dispatch_reshape_and_cache(
    key: &Buffer,
    key_offset: usize,
    value: &Buffer,
    value_offset: usize,
    key_cache: &Buffer,
    value_cache: &Buffer,
    slot_mapping: &Buffer,
    params: &ReshapeAndCacheParams,
    dtype: MetalDtype,
) -> Result<(), String> {
    let state = MetalState::get()?;

    // Determine if using FP8 based on cache dtype
    let use_fp8 = dtype.is_fp8();

    // Get pipeline for this dtype
    let kernel_name = MetalState::reshape_and_cache_kernel_name(dtype, use_fp8);
    let pipeline = state.get_pipeline(&kernel_name)?;

    // Create command buffer and encoder
    let command_queue = state.device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);

    // Set buffers
    encoder.set_buffer(0, Some(key), key_offset as u64);
    encoder.set_buffer(1, Some(value), value_offset as u64);
    encoder.set_buffer(2, Some(key_cache), 0);
    encoder.set_buffer(3, Some(value_cache), 0);
    encoder.set_buffer(4, Some(slot_mapping), 0);

    // k_scale and v_scale - use params values for FP8
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
    encoder.set_buffer(5, Some(&k_scale_buffer), 0);
    encoder.set_buffer(6, Some(&v_scale_buffer), 0);

    // Set dimension constants as buffers (Metal kernel expects device buffers)
    let key_stride = params.key_stride;
    let value_stride = params.value_stride;
    let num_heads = params.num_heads as i32;
    let head_size = params.head_size as i32;
    let block_size = params.block_size as i32;
    let x = params.x;

    // Create constant buffers
    let create_const_buffer = |value: i32| {
        state.device.new_buffer_with_data(
            &value as *const i32 as *const _,
            std::mem::size_of::<i32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        )
    };

    let key_stride_buf = create_const_buffer(key_stride);
    let value_stride_buf = create_const_buffer(value_stride);
    let num_heads_buf = create_const_buffer(num_heads);
    let head_size_buf = create_const_buffer(head_size);
    let block_size_buf = create_const_buffer(block_size);
    let x_buf = create_const_buffer(x);

    encoder.set_buffer(7, Some(&key_stride_buf), 0);
    encoder.set_buffer(8, Some(&value_stride_buf), 0);
    encoder.set_buffer(9, Some(&num_heads_buf), 0);
    encoder.set_buffer(10, Some(&head_size_buf), 0);
    encoder.set_buffer(11, Some(&block_size_buf), 0);
    encoder.set_buffer(12, Some(&x_buf), 0);

    // Dispatch: 1 threadgroup per token, 256 threads per threadgroup
    let threads_per_threadgroup = MTLSize::new(256, 1, 1);
    let threadgroups = MTLSize::new(params.num_tokens as u64, 1, 1);

    encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

/// Raw buffer pointers for reshape_and_cache dispatch
///
/// Used when working with MLX arrays where we extract raw Metal buffer pointers.
#[derive(Debug)]
pub struct RawBufferInfo {
    /// Raw MTLBuffer pointer (as void*)
    pub ptr: *mut c_void,
    /// Byte offset into the buffer
    pub offset: usize,
}

/// Dispatch reshape_and_cache kernel with raw buffer pointers
///
/// This variant is used when integrating with MLX arrays, where we extract
/// raw Metal buffer pointers instead of having owned `Buffer` references.
///
/// # Safety
///
/// - All buffer pointers must be valid MTLBuffer* pointers
/// - key and value buffers must have shape [num_tokens, num_heads, head_size]
/// - key_cache must have shape [num_blocks, num_heads, head_size/x, block_size, x]
/// - value_cache must have shape [num_blocks, num_heads, head_size, block_size]
/// - slot_mapping must have num_tokens elements of i64
/// - All buffers must remain valid until the kernel completes
#[allow(clippy::too_many_arguments)]
pub unsafe fn dispatch_reshape_and_cache_raw(
    key: &RawBufferInfo,
    value: &RawBufferInfo,
    key_cache: &Buffer,
    value_cache: &Buffer,
    slot_mapping: &RawBufferInfo,
    params: &ReshapeAndCacheParams,
    dtype: MetalDtype,
) -> Result<(), String> {
    let state = MetalState::get()?;

    // Determine if using FP8 based on cache dtype
    let use_fp8 = dtype.is_fp8();

    // Get pipeline for this dtype
    let kernel_name = MetalState::reshape_and_cache_kernel_name(dtype, use_fp8);
    let pipeline = state.get_pipeline(&kernel_name)?;

    // Create command buffer and encoder
    let command_queue = state.device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);

    // Set buffers - convert raw pointers to BufferRef
    // SAFETY: Caller guarantees these are valid MTLBuffer* pointers
    let key_buffer_ref: &BufferRef = unsafe { ForeignTypeRef::from_ptr(key.ptr as *mut _) };
    let value_buffer_ref: &BufferRef = unsafe { ForeignTypeRef::from_ptr(value.ptr as *mut _) };
    let slot_buffer_ref: &BufferRef =
        unsafe { ForeignTypeRef::from_ptr(slot_mapping.ptr as *mut _) };

    encoder.set_buffer(0, Some(key_buffer_ref), key.offset as u64);
    encoder.set_buffer(1, Some(value_buffer_ref), value.offset as u64);
    encoder.set_buffer(2, Some(key_cache), 0);
    encoder.set_buffer(3, Some(value_cache), 0);
    encoder.set_buffer(4, Some(slot_buffer_ref), slot_mapping.offset as u64);

    // k_scale and v_scale - use params values for FP8
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
    encoder.set_buffer(5, Some(&k_scale_buffer), 0);
    encoder.set_buffer(6, Some(&v_scale_buffer), 0);

    // Set dimension constants as buffers
    let key_stride = params.key_stride;
    let value_stride = params.value_stride;
    let num_heads = params.num_heads as i32;
    let head_size = params.head_size as i32;
    let block_size = params.block_size as i32;
    let x = params.x;

    let create_const_buffer = |value: i32| {
        state.device.new_buffer_with_data(
            &value as *const i32 as *const _,
            std::mem::size_of::<i32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        )
    };

    let key_stride_buf = create_const_buffer(key_stride);
    let value_stride_buf = create_const_buffer(value_stride);
    let num_heads_buf = create_const_buffer(num_heads);
    let head_size_buf = create_const_buffer(head_size);
    let block_size_buf = create_const_buffer(block_size);
    let x_buf = create_const_buffer(x);

    encoder.set_buffer(7, Some(&key_stride_buf), 0);
    encoder.set_buffer(8, Some(&value_stride_buf), 0);
    encoder.set_buffer(9, Some(&num_heads_buf), 0);
    encoder.set_buffer(10, Some(&head_size_buf), 0);
    encoder.set_buffer(11, Some(&block_size_buf), 0);
    encoder.set_buffer(12, Some(&x_buf), 0);

    // Dispatch: 1 threadgroup per token, 256 threads per threadgroup
    let threads_per_threadgroup = MTLSize::new(256, 1, 1);
    let threadgroups = MTLSize::new(params.num_tokens as u64, 1, 1);

    encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_params() {
        let params = ReshapeAndCacheParams {
            num_tokens: 1,
            num_heads: 4,
            head_size: 128,
            block_size: 16,
            key_stride: 512, // 4 * 128
            value_stride: 512,
            x: 8, // 16 / sizeof(half)
            k_scale: 1.0,
            v_scale: 1.0,
        };
        assert_eq!(params.num_tokens, 1);
    }

    #[test]
    fn test_fp8_params() {
        let params = ReshapeAndCacheParams {
            num_tokens: 1,
            num_heads: 4,
            head_size: 128,
            block_size: 16,
            key_stride: 512,
            value_stride: 512,
            x: 16,         // 16 / sizeof(uchar) for FP8
            k_scale: 0.5,  // Example FP8 scale
            v_scale: 0.25, // Example FP8 scale
        };
        assert_eq!(params.x, 16);
        assert_eq!(params.k_scale, 0.5);
        assert_eq!(params.v_scale, 0.25);
    }
}
