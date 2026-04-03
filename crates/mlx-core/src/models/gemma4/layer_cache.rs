use crate::array::MxArray;
use crate::transformer::{KVCache, RotatingKVCache};

/// Per-layer cache for Gemma4 decoder layers.
///
/// Global (full attention) layers use KVCache.
/// Sliding (local attention) layers use RotatingKVCache with window size.
pub enum Gemma4LayerCache {
    Global(KVCache),
    Sliding(RotatingKVCache),
}

impl Gemma4LayerCache {
    pub fn new_global() -> Self {
        Gemma4LayerCache::Global(KVCache::new())
    }

    pub fn new_sliding(window_size: i32) -> Self {
        Gemma4LayerCache::Sliding(RotatingKVCache::new(window_size, None))
    }

    /// Get the current offset (number of tokens cached).
    pub fn get_offset(&self) -> i32 {
        match self {
            Gemma4LayerCache::Global(c) => c.get_offset(),
            Gemma4LayerCache::Sliding(c) => c.get_offset(),
        }
    }

    /// Get the current cached K/V as (keys, values).
    ///
    /// Returns the valid portion of the cache (sliced to current offset).
    /// For KVCache: returns keys/values sliced to [0..offset].
    /// For RotatingKVCache: returns the current window contents.
    ///
    /// Returns None if the cache is empty.
    pub fn get_cached_kv(&self) -> Option<(MxArray, MxArray)> {
        match self {
            Gemma4LayerCache::Global(c) => {
                let offset = c.get_offset();
                if offset == 0 {
                    return None;
                }
                let keys = c.keys_ref()?;
                let values = c.values_ref()?;
                // Slice to valid portion [0..offset]
                let keys = keys.slice_axis(2, 0, offset as i64).ok()?;
                let values = values.slice_axis(2, 0, offset as i64).ok()?;
                Some((keys, values))
            }
            Gemma4LayerCache::Sliding(c) => c.fetch_current_kv(),
        }
    }
}
