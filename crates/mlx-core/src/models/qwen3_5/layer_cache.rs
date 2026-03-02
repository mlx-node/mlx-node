use crate::array::MxArray;
use crate::transformer::KVCache;
use mlx_sys as sys;

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

    /// Export cache as 2 raw pointers for the fused C++ forward pass.
    ///
    /// For linear layers: (conv_state_ptr, recurrent_state_ptr)
    /// For full attention layers: (keys_ptr, values_ptr)
    ///
    /// Returns null pointers if the cache slot is empty.
    pub fn export_ptrs(&self) -> (*mut sys::mlx_array, *mut sys::mlx_array) {
        match self {
            Self::Linear(c) => {
                let p0 = c.get(0).map_or(std::ptr::null_mut(), |a| a.as_raw_ptr());
                let p1 = c.get(1).map_or(std::ptr::null_mut(), |a| a.as_raw_ptr());
                (p0, p1)
            }
            Self::FullAttention(c) => {
                let keys_ptr = c
                    .keys_ref()
                    .map_or(std::ptr::null_mut(), |a| a.as_raw_ptr());
                let values_ptr = c
                    .values_ref()
                    .map_or(std::ptr::null_mut(), |a| a.as_raw_ptr());
                (keys_ptr, values_ptr)
            }
        }
    }

    /// Collect references to all stored arrays in this cache slot.
    ///
    /// Used after import_ptrs to gather arrays for async_eval to prevent
    /// compute graph accumulation across decode steps.
    pub fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        match self {
            Self::Linear(c) => {
                if let Some(arr) = c.get(0) {
                    out.push(arr);
                }
                if let Some(arr) = c.get(1) {
                    out.push(arr);
                }
            }
            Self::FullAttention(c) => {
                if let Some(k) = c.keys_ref() {
                    out.push(k);
                }
                if let Some(v) = c.values_ref() {
                    out.push(v);
                }
            }
        }
    }

    /// Import cache from 2 raw pointers returned by the fused C++ forward pass.
    ///
    /// Takes ownership of the pointers (wraps them in MxArray).
    pub fn import_ptrs(
        &mut self,
        p0: *mut sys::mlx_array,
        p1: *mut sys::mlx_array,
        new_offset: i32,
    ) {
        match self {
            Self::Linear(c) => {
                if !p0.is_null()
                    && let Ok(arr) = MxArray::from_handle(p0, "fused_conv_state")
                {
                    c.set(0, arr);
                }
                if !p1.is_null()
                    && let Ok(arr) = MxArray::from_handle(p1, "fused_recurrent_state")
                {
                    c.set(1, arr);
                }
            }
            Self::FullAttention(c) => {
                if !p0.is_null()
                    && let Ok(keys) = MxArray::from_handle(p0, "fused_kv_keys")
                {
                    c.set_keys(keys);
                }
                if !p1.is_null()
                    && let Ok(values) = MxArray::from_handle(p1, "fused_kv_values")
                {
                    c.set_values(values);
                }
                c.set_offset(new_offset);
            }
        }
    }
}
