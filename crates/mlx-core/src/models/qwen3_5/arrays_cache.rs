use crate::array::MxArray;

/// Simple array cache for linear attention (GatedDeltaNet) layers.
///
/// Holds N arrays (typically 2 slots: conv_state and recurrent_state).
/// This is the equivalent of Python's ArraysCache.
pub struct ArraysCache {
    cache: Vec<Option<MxArray>>,
}

impl ArraysCache {
    /// Create a new cache with `size` slots.
    pub fn new(size: usize) -> Self {
        let cache = (0..size).map(|_| None).collect();
        Self { cache }
    }

    /// Get the array at the given index.
    pub fn get(&self, idx: usize) -> Option<&MxArray> {
        self.cache.get(idx).and_then(|v| v.as_ref())
    }

    /// Set the array at the given index.
    /// Panics if index is out of bounds (indicates a programming bug).
    pub fn set(&mut self, idx: usize, value: MxArray) {
        assert!(
            idx < self.cache.len(),
            "ArraysCache::set() index {} out of bounds (size {})",
            idx,
            self.cache.len()
        );
        self.cache[idx] = Some(value);
    }

    /// Check if a specific slot is populated.
    pub fn has(&self, idx: usize) -> bool {
        self.cache.get(idx).is_some_and(|v| v.is_some())
    }

    /// Reset all cache entries.
    pub fn reset(&mut self) {
        for slot in &mut self.cache {
            *slot = None;
        }
    }

    /// Number of slots in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

impl Clone for ArraysCache {
    fn clone(&self) -> Self {
        Self {
            cache: self.cache.clone(),
        }
    }
}
