//! copy_blocks Metal kernel dispatch
//!
//! This kernel copies blocks for copy-on-write semantics (used in beam search).

use super::state::{MetalDtype, MetalState};
use metal::foreign_types::ForeignTypeRef;
use metal::{Buffer, BufferRef, MTLSize};
use std::ffi::c_void;

/// Parameters for copy_blocks kernel
#[derive(Debug, Clone)]
pub struct CopyBlocksParams {
    /// Number of block pairs to copy
    pub num_pairs: u32,
    /// Number of elements per block (num_kv_heads * head_size * block_size)
    pub numel_per_block: u32,
}

impl CopyBlocksParams {
    /// Create params from cache config
    pub fn from_config(num_pairs: u32, num_kv_heads: u32, head_size: u32, block_size: u32) -> Self {
        Self {
            num_pairs,
            numel_per_block: num_kv_heads * head_size * block_size,
        }
    }
}

impl MetalState {
    /// Get the copy_blocks kernel name for a dtype
    ///
    /// # Arguments
    /// * `dtype` - Data type for cache elements
    pub fn copy_blocks_kernel_name(dtype: MetalDtype) -> String {
        let type_str = dtype.type_string();
        format!("copy_blocks_{}", type_str)
    }
}

/// Dispatch the copy_blocks kernel
///
/// Copies blocks from source to destination for copy-on-write semantics.
/// This is used during beam search when a sequence branches.
///
/// # Buffer Layout
/// - buffer(0): key_cache [num_blocks, num_kv_heads, head_size/x, block_size, x]
/// - buffer(1): value_cache [num_blocks, num_kv_heads, head_size, block_size]
/// - buffer(2): block_mapping [num_pairs, 2] - int64 pairs of (src_block, dst_block)
/// - buffer(3): numel_per_block - int32
///
/// # Arguments
/// * `key_cache` - Key cache buffer
/// * `value_cache` - Value cache buffer
/// * `block_mapping` - Buffer containing (src, dst) block pairs
/// * `params` - Copy parameters
/// * `dtype` - Data type of cache elements
pub fn dispatch_copy_blocks(
    key_cache: &Buffer,
    value_cache: &Buffer,
    block_mapping: &Buffer,
    params: &CopyBlocksParams,
    dtype: MetalDtype,
) -> Result<(), String> {
    if params.num_pairs == 0 {
        return Ok(()); // Nothing to copy
    }

    let state = MetalState::get()?;

    // Get pipeline for this dtype
    let kernel_name = MetalState::copy_blocks_kernel_name(dtype);
    let pipeline = state.get_pipeline(&kernel_name)?;

    // Create command buffer and encoder
    let command_queue = state.device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);

    // Set buffers
    encoder.set_buffer(0, Some(key_cache), 0);
    encoder.set_buffer(1, Some(value_cache), 0);
    encoder.set_buffer(2, Some(block_mapping), 0);

    // Set numel_per_block constant
    let numel = params.numel_per_block as i32;
    let numel_buffer = state.device.new_buffer_with_data(
        &numel as *const i32 as *const _,
        std::mem::size_of::<i32>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    encoder.set_buffer(3, Some(&numel_buffer), 0);

    // Dispatch: 1 threadgroup per block pair, 256 threads per threadgroup
    let threads_per_threadgroup = MTLSize::new(256, 1, 1);
    let threadgroups = MTLSize::new(params.num_pairs as u64, 1, 1);

    encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

/// Raw buffer info for copy_blocks dispatch
#[derive(Debug)]
pub struct RawCopyBlocksBuffers {
    /// Key cache raw pointer
    pub key_cache_ptr: *mut c_void,
    /// Value cache raw pointer
    pub value_cache_ptr: *mut c_void,
}

/// Dispatch copy_blocks kernel with raw buffer pointers
///
/// # Safety
///
/// - All buffer pointers must be valid MTLBuffer* pointers
/// - key_cache and value_cache must be properly sized
/// - block_mapping must have num_pairs * 2 elements of i64
/// - All buffers must remain valid until the kernel completes
pub unsafe fn dispatch_copy_blocks_raw(
    caches: &RawCopyBlocksBuffers,
    block_mapping: &Buffer,
    params: &CopyBlocksParams,
    dtype: MetalDtype,
) -> Result<(), String> {
    if params.num_pairs == 0 {
        return Ok(()); // Nothing to copy
    }

    let state = MetalState::get()?;

    // Get pipeline for this dtype
    let kernel_name = MetalState::copy_blocks_kernel_name(dtype);
    let pipeline = state.get_pipeline(&kernel_name)?;

    // Create command buffer and encoder
    let command_queue = state.device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);

    // Set buffers - convert raw pointers to BufferRef
    // SAFETY: Caller guarantees these are valid MTLBuffer* pointers
    let key_cache_ref: &BufferRef =
        unsafe { ForeignTypeRef::from_ptr(caches.key_cache_ptr as *mut _) };
    let value_cache_ref: &BufferRef =
        unsafe { ForeignTypeRef::from_ptr(caches.value_cache_ptr as *mut _) };

    encoder.set_buffer(0, Some(key_cache_ref), 0);
    encoder.set_buffer(1, Some(value_cache_ref), 0);
    encoder.set_buffer(2, Some(block_mapping), 0);

    // Set numel_per_block constant
    let numel = params.numel_per_block as i32;
    let numel_buffer = state.device.new_buffer_with_data(
        &numel as *const i32 as *const _,
        std::mem::size_of::<i32>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    encoder.set_buffer(3, Some(&numel_buffer), 0);

    // Dispatch: 1 threadgroup per block pair, 256 threads per threadgroup
    let threads_per_threadgroup = MTLSize::new(256, 1, 1);
    let threadgroups = MTLSize::new(params.num_pairs as u64, 1, 1);

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
    fn test_copy_blocks_params() {
        let params = CopyBlocksParams::from_config(
            2,   // num_pairs
            4,   // num_kv_heads
            128, // head_size
            16,  // block_size
        );
        assert_eq!(params.num_pairs, 2);
        assert_eq!(params.numel_per_block, 4 * 128 * 16);
    }

    #[test]
    fn test_copy_blocks_kernel_name() {
        assert_eq!(
            MetalState::copy_blocks_kernel_name(MetalDtype::Float16),
            "copy_blocks_half"
        );
        assert_eq!(
            MetalState::copy_blocks_kernel_name(MetalDtype::Float32),
            "copy_blocks_float"
        );
        assert_eq!(
            MetalState::copy_blocks_kernel_name(MetalDtype::BFloat16),
            "copy_blocks_bfloat16_t"
        );
    }

    #[test]
    fn test_get_copy_blocks_pipeline() {
        let state = MetalState::get().expect("Failed to init Metal state");
        let kernel_name = MetalState::copy_blocks_kernel_name(MetalDtype::Float16);
        let pipeline = state.get_pipeline(&kernel_name);
        assert!(
            pipeline.is_ok(),
            "Failed to get copy_blocks pipeline: {:?}",
            pipeline.err()
        );
    }
}
