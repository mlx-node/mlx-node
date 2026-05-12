use crate::array::MxArray;
use crate::models::qwen3_5::arrays_cache::ArraysCache;
use crate::transformer::KVCache;

/// Per-layer cache for LFM2 decoder layers.
///
/// Attention layers use KVCache (standard transformer K/V cache).
/// Conv layers use ArraysCache (1 slot for conv state).
pub enum Lfm2LayerCache {
    Conv(ArraysCache),
    Attention(KVCache),
}

impl Lfm2LayerCache {
    /// Create a new conv layer cache (1 slot for conv state).
    pub fn new_conv() -> Self {
        Lfm2LayerCache::Conv(ArraysCache::new(1))
    }

    /// Create a new attention layer cache.
    pub fn new_attention() -> Self {
        Lfm2LayerCache::Attention(KVCache::new())
    }

    /// Collect cache arrays for eval.
    ///
    /// Gathers references to all internal arrays so they can be
    /// evaluated between chunked prefill steps.
    pub fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        match self {
            Lfm2LayerCache::Conv(c) => {
                for i in 0..c.len() {
                    if let Some(arr) = c.get(i) {
                        out.push(arr);
                    }
                }
            }
            Lfm2LayerCache::Attention(c) => {
                if let Some(k) = c.keys_ref() {
                    out.push(k);
                }
                if let Some(v) = c.values_ref() {
                    out.push(v);
                }
            }
        }
    }

    /// Get mutable reference to the inner KVCache (for attention layers).
    pub fn as_kv_cache_mut(&mut self) -> Option<&mut KVCache> {
        match self {
            Lfm2LayerCache::Attention(c) => Some(c),
            Lfm2LayerCache::Conv(_) => None,
        }
    }

    /// Get mutable reference to the inner ArraysCache (for conv layers).
    pub fn as_conv_cache_mut(&mut self) -> Option<&mut ArraysCache> {
        match self {
            Lfm2LayerCache::Conv(c) => Some(c),
            Lfm2LayerCache::Attention(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_as_kv_cache_mut_attention() {
        let mut cache = Lfm2LayerCache::new_attention();
        assert!(cache.as_kv_cache_mut().is_some());
        assert!(cache.as_conv_cache_mut().is_none());
    }

    #[test]
    fn test_as_conv_cache_mut_conv() {
        let mut cache = Lfm2LayerCache::new_conv();
        assert!(cache.as_conv_cache_mut().is_some());
        assert!(cache.as_kv_cache_mut().is_none());
    }

    #[test]
    fn test_collect_arrays_empty() {
        let cache = Lfm2LayerCache::new_attention();
        let mut out = Vec::new();
        cache.collect_arrays(&mut out);
        assert!(out.is_empty());
    }
}
