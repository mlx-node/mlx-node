use crate::transformer::KVCache;

use super::arrays_cache::ArraysCache;

/// Mixed cache type for Qwen3.5 layers.
///
/// Linear attention layers use ArraysCache (conv_state + recurrent_state).
/// Full attention layers use standard KVCache.
pub enum Qwen3_5LayerCache {
    Linear(ArraysCache),
    FullAttention(KVCache),
}

impl Qwen3_5LayerCache {
    /// Create a cache for a linear attention layer.
    pub fn new_linear() -> Self {
        Self::Linear(ArraysCache::new(2)) // 2 slots: conv_state, recurrent_state
    }

    /// Create a cache for a full attention layer.
    pub fn new_full_attention() -> Self {
        Self::FullAttention(KVCache::new())
    }

    /// Get as mutable ArraysCache, or None if this is a full-attention cache.
    pub fn as_arrays_cache_mut(&mut self) -> Option<&mut ArraysCache> {
        match self {
            Self::Linear(c) => Some(c),
            _ => None,
        }
    }

    /// Get as mutable KVCache, or None if this is a linear-attention cache.
    pub fn as_kv_cache_mut(&mut self) -> Option<&mut KVCache> {
        match self {
            Self::FullAttention(c) => Some(c),
            _ => None,
        }
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        match self {
            Self::Linear(c) => c.reset(),
            Self::FullAttention(c) => c.reset(),
        }
    }

    /// Get the cache offset (for mask generation).
    /// For linear layers, returns 0 (not position-based).
    /// For full attention layers, returns the KVCache offset.
    pub fn offset(&self) -> i32 {
        match self {
            Self::Linear(_) => 0,
            Self::FullAttention(c) => c.get_offset(),
        }
    }
}
