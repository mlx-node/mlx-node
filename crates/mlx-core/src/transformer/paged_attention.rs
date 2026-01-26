//! Paged Attention - Re-exports from mlx-paged-attn
//!
//! This module provides access to the paged attention infrastructure from the
//! `mlx-paged-attn` crate, which implements Metal kernel-based paged attention.
//!
//! ## Metal Kernel Dispatch
//!
//! The `mlx-paged-attn` crate contains compiled Metal kernels for:
//! - `reshape_and_cache` - Updates KV cache with new tokens
//! - `paged_attention` - Computes attention using paged KV cache
//!
//! Use `PagedKVCache` from `mlx-paged-attn` directly for Metal kernel dispatch:
//!
//! ```ignore
//! use mlx_paged_attn::{PagedKVCache, PagedAttentionConfig};
//!
//! let mut cache = PagedKVCache::new(config)?;
//! cache.initialize()?;  // Allocate Metal buffers
//!
//! // Direct Metal kernel dispatch
//! unsafe {
//!     cache.update(layer_idx, keys_ptr, values_ptr, slot_mapping_ptr)?;
//!     let output = cache.attention(layer_idx, queries_ptr, &seq_ids, num_heads, scale)?;
//! }
//! ```

// Re-export PagedKVCache, config, and scheduler from mlx-paged-attn
pub use mlx_paged_attn::{
    CompletedSequence, ContinuousBatchingScheduler, MemoryStats, PagedAttentionConfig,
    PagedKVCache, PendingRequest, ScheduledBatch, SchedulerConfig, SchedulerStats, TokenOutput,
};
