//! Paged Attention - Integration layer for PagedAttention with MLX tensors
//!
//! This module bridges the block management from mlx-paged-attn with
//! actual tensor operations using MLX's FFI.
//!
//! ## Metal Kernel Dispatch
//!
//! The `mlx-paged-attn` crate contains compiled Metal kernels for:
//! - `reshape_and_cache` - Updates KV cache with new tokens
//! - `paged_attention` - Computes attention using paged KV cache
//!
//! Currently, operations use the C++ software fallback in `mlx-sys`.
//! Full Metal kernel dispatch requires extracting MTLBuffer from MLX arrays,
//! which is not yet supported by the MLX public API.
//!
//! The Metal dispatch infrastructure is ready in `mlx_paged_attn::metal` and
//! can be enabled once buffer extraction is available.

use crate::array::MxArray;
use mlx_sys as sys;
use std::ptr;

// Re-export PagedKVCache, config, and scheduler from mlx-paged-attn
pub use mlx_paged_attn::{
    CompletedSequence, ContinuousBatchingScheduler, MemoryStats, PagedAttentionConfig,
    PagedKVCache, PendingRequest, ScheduledBatch, SchedulerConfig, SchedulerStats, TokenOutput,
};

/// Paged Attention layer that integrates with PagedKVCache
///
/// This layer provides the tensor-level operations for paged attention:
/// - Cache update (reshape_and_cache kernel)
/// - Attention computation (paged_attention kernel)
///
/// ## Usage with Qwen3
/// ```ignore
/// let cache = PagedKVCache::new(config)?;
/// let attn = PagedAttentionLayer::new(config)?;
///
/// // During generation
/// attn.update_cache(layer_idx, &keys, &values, &slot_mapping)?;
/// let output = attn.forward(&queries, &block_tables, &context_lens, scale, layer_idx)?;
/// ```
pub struct PagedAttentionLayer {
    /// C++ cache handle for the underlying buffers
    cache_handle: *mut sys::PagedAttnCache,
    /// Configuration
    config: PagedAttentionConfig,
    /// Whether we own the cache handle (for cleanup)
    owns_handle: bool,
}

// SAFETY: PagedAttentionLayer is Send+Sync because:
// - The cache_handle is a C++ pointer to GPU memory managed by MLX
// - MLX handles thread safety internally for GPU resources
// - We don't mutate the pointer, only pass it to FFI calls
unsafe impl Send for PagedAttentionLayer {}
unsafe impl Sync for PagedAttentionLayer {}

impl PagedAttentionLayer {
    /// Create a new PagedAttentionLayer with the given configuration
    ///
    /// This allocates the underlying KV cache buffers on GPU.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Configuration validation fails (invalid block_size, head_size, etc.)
    /// - FP8 cache is requested (not yet implemented)
    /// - GPU memory allocation fails
    pub fn new(config: PagedAttentionConfig) -> Result<Self, String> {
        // Validate configuration (rejects FP8, invalid sizes, etc.)
        config.validate()?;

        // Create FFI config
        // dtype mapping: 0=float16, 1=bfloat16, 2=float32
        let ffi_config = sys::PagedAttnConfig {
            block_size: config.block_size,
            num_blocks: config.calculate_num_blocks(),
            head_size: config.head_size,
            num_kv_heads: config.num_kv_heads,
            num_layers: config.num_layers,
            dtype: 0, // Always use float16 until FP8 is properly wired
        };

        let handle = unsafe { sys::mlx_paged_attn_create_cache(&ffi_config) };
        if handle.is_null() {
            return Err("Failed to create paged attention cache".to_string());
        }

        Ok(Self {
            cache_handle: handle,
            config,
            owns_handle: true,
        })
    }

    /// Update the KV cache with new keys and values
    ///
    /// # Arguments
    /// * `layer_idx` - Which transformer layer
    /// * `keys` - New key vectors, shape: [num_tokens, num_kv_heads, head_size]
    /// * `values` - New value vectors, shape: [num_tokens, num_kv_heads, head_size]
    /// * `slot_mapping` - Slot indices for each token, shape: [num_tokens]
    pub fn update_cache(
        &self,
        layer_idx: u32,
        keys: &MxArray,
        values: &MxArray,
        slot_mapping: &MxArray,
    ) -> Result<(), String> {
        unsafe {
            sys::mlx_paged_attn_reshape_and_cache(
                self.cache_handle,
                layer_idx,
                keys.handle.0,
                values.handle.0,
                slot_mapping.handle.0,
            );
        }
        Ok(())
    }

    /// Run paged attention forward pass
    ///
    /// # Arguments
    /// * `queries` - Query vectors, shape: [num_seqs, num_heads, head_size]
    /// * `block_tables` - Block table array, shape: [num_seqs, max_blocks_per_seq]
    /// * `context_lens` - Context length for each sequence, shape: [num_seqs]
    /// * `scale` - Attention scale factor (1/sqrt(head_dim))
    /// * `layer_idx` - Which transformer layer
    ///
    /// # Returns
    /// * Attention output, shape: [num_seqs, num_heads, head_size]
    pub fn forward(
        &self,
        queries: &MxArray,
        block_tables: &MxArray,
        context_lens: &MxArray,
        scale: f64,
        layer_idx: u32,
    ) -> Result<MxArray, String> {
        // Get the key and value caches for this layer
        let key_cache = unsafe { sys::mlx_paged_attn_get_key_cache(self.cache_handle, layer_idx) };
        let value_cache =
            unsafe { sys::mlx_paged_attn_get_value_cache(self.cache_handle, layer_idx) };

        // Check for errors and clean up any allocated handles
        if key_cache.is_null() || value_cache.is_null() {
            // Free any successfully allocated handles before returning error
            if !key_cache.is_null() {
                unsafe { sys::mlx_array_delete(key_cache) };
            }
            if !value_cache.is_null() {
                unsafe { sys::mlx_array_delete(value_cache) };
            }
            return Err(format!("Failed to get cache for layer {}", layer_idx));
        }

        // Run paged attention
        let output = unsafe {
            sys::mlx_paged_attn_forward(
                queries.handle.0,
                key_cache,
                value_cache,
                block_tables.handle.0,
                context_lens.handle.0,
                scale as f32,
                self.config.block_size,
                self.config.max_seq_len(),
            )
        };

        // Clean up temporary cache array handles (always, regardless of output status)
        unsafe {
            sys::mlx_array_delete(key_cache);
            sys::mlx_array_delete(value_cache);
        }

        if output.is_null() {
            return Err("Paged attention forward failed".to_string());
        }

        MxArray::from_handle(output, "paged_attention_output").map_err(|e| e.reason.clone())
    }

    /// Get the key cache tensor for a layer (for debugging)
    pub fn get_key_cache(&self, layer_idx: u32) -> Result<MxArray, String> {
        let handle = unsafe { sys::mlx_paged_attn_get_key_cache(self.cache_handle, layer_idx) };
        if handle.is_null() {
            return Err(format!("Failed to get key cache for layer {}", layer_idx));
        }
        // If from_handle fails, we need to clean up the allocated handle
        MxArray::from_handle(handle, "paged_key_cache").map_err(|e| {
            unsafe { sys::mlx_array_delete(handle) };
            e.reason.clone()
        })
    }

    /// Get the value cache tensor for a layer (for debugging)
    pub fn get_value_cache(&self, layer_idx: u32) -> Result<MxArray, String> {
        let handle = unsafe { sys::mlx_paged_attn_get_value_cache(self.cache_handle, layer_idx) };
        if handle.is_null() {
            return Err(format!("Failed to get value cache for layer {}", layer_idx));
        }
        // If from_handle fails, we need to clean up the allocated handle
        MxArray::from_handle(handle, "paged_value_cache").map_err(|e| {
            unsafe { sys::mlx_array_delete(handle) };
            e.reason.clone()
        })
    }

    /// Copy blocks for copy-on-write semantics
    ///
    /// # Arguments
    /// * `layer_idx` - Which transformer layer
    /// * `block_mapping` - Source and destination block pairs, shape: [num_pairs, 2]
    pub fn copy_blocks(&self, layer_idx: u32, block_mapping: &MxArray) -> Result<(), String> {
        unsafe {
            sys::mlx_paged_attn_copy_blocks(self.cache_handle, layer_idx, block_mapping.handle.0);
        }
        Ok(())
    }

    /// Get the configuration
    pub fn config(&self) -> PagedAttentionConfig {
        self.config.clone()
    }
}

impl Drop for PagedAttentionLayer {
    fn drop(&mut self) {
        if self.owns_handle && !self.cache_handle.is_null() {
            unsafe {
                sys::mlx_paged_attn_free_cache(self.cache_handle);
            }
            self.cache_handle = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_attention_layer_creation() {
        let config = PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb: 512,
            head_size: 128,
            num_kv_heads: 4,
            num_layers: 2, // Small for testing
            ..Default::default()
        };

        let layer = PagedAttentionLayer::new(config);
        assert!(layer.is_ok());
    }
}
