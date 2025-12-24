//! swap_blocks - CPU offload support for paged attention
//!
//! This module provides infrastructure for swapping KV cache blocks
//! between GPU and CPU memory when GPU memory is constrained.
//! Uses Metal blit commands for efficient async copy.

use super::state::MetalState;
use metal::{Buffer, MTLResourceOptions};
use std::collections::HashMap;

/// CPU cache for swapped-out blocks
///
/// Provides a CPU-side storage for KV cache blocks that have been
/// swapped out from GPU memory to make room for other sequences.
pub struct CpuBlockCache {
    /// CPU buffers for swapped key blocks
    /// Maps block_id -> (key_buffer, value_buffer)
    swapped_blocks: HashMap<u32, (Buffer, Buffer)>,

    /// Block size (tokens per block)
    block_size: u32,
}

impl CpuBlockCache {
    /// Create a new CPU block cache
    ///
    /// # Arguments
    /// * `num_kv_heads` - Number of KV heads (unused, kept for API compatibility)
    /// * `head_size` - Size of each head (unused, kept for API compatibility)
    /// * `block_size` - Block size (tokens per block)
    pub fn new(_num_kv_heads: u32, _head_size: u32, block_size: u32) -> Self {
        Self {
            swapped_blocks: HashMap::new(),
            block_size,
        }
    }

    /// Check if a block is currently swapped out
    pub fn is_swapped(&self, block_id: u32) -> bool {
        self.swapped_blocks.contains_key(&block_id)
    }

    /// Get the number of swapped blocks
    pub fn num_swapped(&self) -> usize {
        self.swapped_blocks.len()
    }

    /// Get swapped block buffers
    pub fn get_swapped(&self, block_id: u32) -> Option<(&Buffer, &Buffer)> {
        self.swapped_blocks.get(&block_id).map(|(k, v)| (k, v))
    }

    /// Remove a swapped block (after swap-in)
    pub fn remove_swapped(&mut self, block_id: u32) -> Option<(Buffer, Buffer)> {
        self.swapped_blocks.remove(&block_id)
    }

    /// Get block size configuration
    pub fn block_size(&self) -> u32 {
        self.block_size
    }
}

/// Parameters for swap operations
#[derive(Debug, Clone)]
pub struct SwapBlocksParams {
    /// Number of KV heads
    pub num_kv_heads: u32,
    /// Size of each head
    pub head_size: u32,
    /// Block size (tokens per block)
    pub block_size: u32,
    /// Whether to use FP8 cache (affects element size)
    pub use_fp8: bool,
}

impl SwapBlocksParams {
    /// Get element size in bytes based on dtype
    fn element_size(&self) -> u64 {
        if self.use_fp8 { 1 } else { 2 }
    }

    /// Get x value (16 bytes / element_size) for key cache layout
    fn x(&self) -> u32 {
        if self.use_fp8 { 16 } else { 8 }
    }

    /// Calculate key block offset in bytes
    pub fn key_block_offset(&self, block_id: u32) -> u64 {
        let x = self.x();
        let element_size = self.element_size();
        let block_elements = self.num_kv_heads as u64
            * (self.head_size as u64 / x as u64)
            * self.block_size as u64
            * x as u64;
        block_id as u64 * block_elements * element_size
    }

    /// Calculate value block offset in bytes
    pub fn value_block_offset(&self, block_id: u32) -> u64 {
        let element_size = self.element_size();
        let block_elements =
            self.num_kv_heads as u64 * self.head_size as u64 * self.block_size as u64;
        block_id as u64 * block_elements * element_size
    }

    /// Calculate key block size in bytes
    pub fn key_block_size(&self) -> u64 {
        let x = self.x();
        let element_size = self.element_size();
        self.num_kv_heads as u64
            * (self.head_size as u64 / x as u64)
            * self.block_size as u64
            * x as u64
            * element_size
    }

    /// Calculate value block size in bytes
    pub fn value_block_size(&self) -> u64 {
        let element_size = self.element_size();
        self.num_kv_heads as u64 * self.head_size as u64 * self.block_size as u64 * element_size
    }
}

/// Swap blocks from GPU to CPU (swap out)
///
/// Copies the specified blocks from GPU key/value caches to CPU memory.
/// The blocks are stored in the cpu_cache for later swap-in.
///
/// # Arguments
/// * `key_cache` - GPU key cache buffer
/// * `value_cache` - GPU value cache buffer
/// * `block_ids` - Block IDs to swap out
/// * `cpu_cache` - CPU cache to store swapped blocks
/// * `params` - Swap parameters
///
/// # Returns
/// * Ok(()) on success
pub fn swap_out(
    key_cache: &Buffer,
    value_cache: &Buffer,
    block_ids: &[u32],
    cpu_cache: &mut CpuBlockCache,
    params: &SwapBlocksParams,
) -> Result<(), String> {
    if block_ids.is_empty() {
        return Ok(());
    }

    let state = MetalState::get()?;

    // Create command buffer for blit operations
    let command_queue = state.device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let blit_encoder = command_buffer.new_blit_command_encoder();

    let key_block_size = params.key_block_size();
    let value_block_size = params.value_block_size();

    for &block_id in block_ids {
        if cpu_cache.is_swapped(block_id) {
            continue; // Already swapped out
        }

        // Allocate CPU buffers for this block
        let cpu_key = state
            .device
            .new_buffer(key_block_size, MTLResourceOptions::StorageModeShared);
        let cpu_value = state
            .device
            .new_buffer(value_block_size, MTLResourceOptions::StorageModeShared);

        // Copy key block from GPU to CPU
        let key_offset = params.key_block_offset(block_id);
        blit_encoder.copy_from_buffer(key_cache, key_offset, &cpu_key, 0, key_block_size);

        // Copy value block from GPU to CPU
        let value_offset = params.value_block_offset(block_id);
        blit_encoder.copy_from_buffer(value_cache, value_offset, &cpu_value, 0, value_block_size);

        cpu_cache
            .swapped_blocks
            .insert(block_id, (cpu_key, cpu_value));
    }

    blit_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

/// Swap blocks from CPU back to GPU (swap in)
///
/// Copies the specified blocks from CPU memory back to GPU key/value caches.
/// The blocks are removed from the cpu_cache after swap-in.
///
/// # Arguments
/// * `key_cache` - GPU key cache buffer
/// * `value_cache` - GPU value cache buffer
/// * `block_ids` - Block IDs to swap in
/// * `cpu_cache` - CPU cache containing swapped blocks
/// * `params` - Swap parameters
///
/// # Returns
/// * Ok(()) on success
pub fn swap_in(
    key_cache: &Buffer,
    value_cache: &Buffer,
    block_ids: &[u32],
    cpu_cache: &mut CpuBlockCache,
    params: &SwapBlocksParams,
) -> Result<(), String> {
    if block_ids.is_empty() {
        return Ok(());
    }

    let state = MetalState::get()?;

    // Create command buffer for blit operations
    let command_queue = state.device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let blit_encoder = command_buffer.new_blit_command_encoder();

    let key_block_size = params.key_block_size();
    let value_block_size = params.value_block_size();

    for &block_id in block_ids {
        let (cpu_key, cpu_value) = cpu_cache
            .get_swapped(block_id)
            .ok_or_else(|| format!("Block {} not found in CPU cache", block_id))?;

        // Copy key block from CPU to GPU
        let key_offset = params.key_block_offset(block_id);
        blit_encoder.copy_from_buffer(cpu_key, 0, key_cache, key_offset, key_block_size);

        // Copy value block from CPU to GPU
        let value_offset = params.value_block_offset(block_id);
        blit_encoder.copy_from_buffer(cpu_value, 0, value_cache, value_offset, value_block_size);
    }

    blit_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Remove swapped blocks from CPU cache
    for &block_id in block_ids {
        cpu_cache.remove_swapped(block_id);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_block_cache() {
        let cache = CpuBlockCache::new(4, 128, 16);
        assert_eq!(cache.num_swapped(), 0);
        assert!(!cache.is_swapped(0));
    }

    #[test]
    fn test_swap_params_fp16() {
        let params = SwapBlocksParams {
            num_kv_heads: 4,
            head_size: 128,
            block_size: 16,
            use_fp8: false,
        };

        // Key block size: 4 * (128/8) * 16 * 8 * 2 = 4 * 16 * 16 * 8 * 2 = 16384
        assert_eq!(params.key_block_size(), 16384);

        // Value block size: 4 * 128 * 16 * 2 = 16384
        assert_eq!(params.value_block_size(), 16384);

        // Block 0 offset should be 0
        assert_eq!(params.key_block_offset(0), 0);
        assert_eq!(params.value_block_offset(0), 0);

        // Block 1 offset
        assert_eq!(params.key_block_offset(1), 16384);
        assert_eq!(params.value_block_offset(1), 16384);
    }

    #[test]
    fn test_swap_params_fp8() {
        let params = SwapBlocksParams {
            num_kv_heads: 4,
            head_size: 128,
            block_size: 16,
            use_fp8: true,
        };

        // FP8: element_size = 1, x = 16
        // Key block size: 4 * (128/16) * 16 * 16 * 1 = 4 * 8 * 16 * 16 * 1 = 8192
        assert_eq!(params.key_block_size(), 8192);

        // Value block size: 4 * 128 * 16 * 1 = 8192
        assert_eq!(params.value_block_size(), 8192);

        // Block 0 offset should be 0
        assert_eq!(params.key_block_offset(0), 0);
        assert_eq!(params.value_block_offset(0), 0);

        // Block 1 offset (half of FP16 due to 1 byte elements)
        assert_eq!(params.key_block_offset(1), 8192);
        assert_eq!(params.value_block_offset(1), 8192);
    }
}
