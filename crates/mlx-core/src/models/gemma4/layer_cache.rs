use crate::array::MxArray;
use crate::transformer::{KVCache, RotatingKVCache};
use napi::bindgen_prelude::*;

/// Internal cache type discriminant.
enum CacheType {
    Global(KVCache),
    Sliding(RotatingKVCache),
}

/// Per-layer cache for Gemma4 decoder layers.
///
/// Global (full attention) layers use KVCache.
/// Sliding (local attention) layers use RotatingKVCache with window size.
///
/// Includes a K/V stash for correct KV sharing. During prefill, the
/// RotatingKVCache returns the FULL untrimmed sequence for attention but stores
/// only the trimmed window internally. The stash captures the returned K/V so
/// that shared layers receive the same context the anchor attention actually used.
pub struct Gemma4LayerCache {
    inner: CacheType,
    /// Stashed K/V from the last `update_and_fetch_stash` call.
    /// During prefill this is the FULL untrimmed sequence (even if the cache
    /// stores only a sliding window). During decode this is the current
    /// window/full state. Used by KV sharing to get the K/V the anchor
    /// attention actually used.
    stashed_kv: Option<(MxArray, MxArray)>,
}

impl Gemma4LayerCache {
    pub fn new_global() -> Self {
        Gemma4LayerCache {
            inner: CacheType::Global(KVCache::new()),
            stashed_kv: None,
        }
    }

    pub fn new_sliding(window_size: i32) -> Self {
        Gemma4LayerCache {
            inner: CacheType::Sliding(RotatingKVCache::new(window_size, None)),
            stashed_kv: None,
        }
    }

    /// Get the current offset (number of tokens cached).
    pub fn get_offset(&self) -> i32 {
        match &self.inner {
            CacheType::Global(c) => c.get_offset(),
            CacheType::Sliding(c) => c.get_offset(),
        }
    }

    /// Get the current cached K/V as (keys, values).
    ///
    /// Returns the valid portion of the cache (sliced to current offset).
    /// For KVCache: returns keys/values sliced to [0..offset].
    /// For RotatingKVCache: returns the current window contents.
    ///
    /// **Note**: For sliding caches after a long prefill this returns the
    /// TRIMMED window, not the full sequence. Use `take_stashed_kv` when you
    /// need the K/V the anchor attention actually used.
    ///
    /// Returns None if the cache is empty.
    pub fn get_cached_kv(&self) -> Option<(MxArray, MxArray)> {
        match &self.inner {
            CacheType::Global(c) => {
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
            CacheType::Sliding(c) => c.fetch_current_kv(),
        }
    }

    /// Update the cache and stash the returned K/V.
    ///
    /// Delegates to the inner cache's `update_and_fetch`, then stashes the
    /// returned K/V pair. Returns the same K/V that was stashed.
    ///
    /// During prefill of a long prompt, the RotatingKVCache returns the FULL
    /// untrimmed sequence for attention while storing only the trimmed window.
    /// The stash captures the full sequence so shared layers get proper context.
    pub fn update_and_fetch_stash(
        &mut self,
        keys: &MxArray,
        values: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        let (k, v) = match &mut self.inner {
            CacheType::Global(kvc) => kvc.update_and_fetch(keys, values)?,
            CacheType::Sliding(rkvc) => {
                let kv = rkvc.update_and_fetch(keys, values)?;
                (kv[0].clone(), kv[1].clone())
            }
        };
        self.stashed_kv = Some((k.clone(), v.clone()));
        Ok((k, v))
    }

    /// Update the cache and return K/V without stashing.
    /// Use this when KV sharing is disabled (num_kv_shared_layers=0).
    pub fn update_and_fetch(
        &mut self,
        keys: &MxArray,
        values: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        match &mut self.inner {
            CacheType::Global(kvc) => kvc.update_and_fetch(keys, values),
            CacheType::Sliding(rkvc) => {
                let kv = rkvc.update_and_fetch(keys, values)?;
                Ok((kv[0].clone(), kv[1].clone()))
            }
        }
    }

    /// Save K/V into the stash (replaces any previous stash).
    pub fn stash_kv(&mut self, keys: MxArray, values: MxArray) {
        self.stashed_kv = Some((keys, values));
    }

    /// Take the stashed K/V, clearing the stash.
    ///
    /// Returns the K/V that was saved by the last `update_and_fetch_stash` or
    /// `stash_kv` call. Returns None if the stash is empty (already taken or
    /// never populated).
    pub fn take_stashed_kv(&mut self) -> Option<(MxArray, MxArray)> {
        self.stashed_kv.take()
    }

    /// Collect references to the raw internal K/V arrays for eval between
    /// chunked prefill steps. Matches Qwen3.5's `collect_arrays` pattern.
    pub fn collect_cache_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        match &self.inner {
            CacheType::Global(c) => {
                if let Some(k) = c.keys_ref() {
                    out.push(k);
                }
                if let Some(v) = c.values_ref() {
                    out.push(v);
                }
            }
            CacheType::Sliding(c) => {
                if let Some(k) = c.keys_ref() {
                    out.push(k);
                }
                if let Some(v) = c.values_ref() {
                    out.push(v);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sliding_cache_stash_preserves_full_prefill() {
        // Create a sliding cache with small window (4 tokens)
        let mut cache = Gemma4LayerCache::new_sliding(4);

        // Simulate a long prefill with 8 tokens (exceeds window of 4)
        // K/V shape: [1, 1, 8, 16] = [B, H, T, D]
        let keys = MxArray::ones(&[1, 1, 8, 16], None).unwrap();
        let values = MxArray::ones(&[1, 1, 8, 16], None).unwrap();

        // update_and_fetch_stash should return full 8-token sequence
        let (ret_k, ret_v) = cache.update_and_fetch_stash(&keys, &values).unwrap();
        assert_eq!(
            ret_k.shape_at(2).unwrap(),
            8,
            "returned K should have full 8 tokens"
        );
        assert_eq!(
            ret_v.shape_at(2).unwrap(),
            8,
            "returned V should have full 8 tokens"
        );

        // The stash should also have full 8 tokens
        let (stash_k, stash_v) = cache.take_stashed_kv().unwrap();
        assert_eq!(
            stash_k.shape_at(2).unwrap(),
            8,
            "stashed K should have full 8 tokens"
        );
        assert_eq!(
            stash_v.shape_at(2).unwrap(),
            8,
            "stashed V should have full 8 tokens"
        );

        // But get_cached_kv (reading from stored cache) should only have window-sized cache
        let (cached_k, _cached_v) = cache.get_cached_kv().unwrap();
        assert!(
            cached_k.shape_at(2).unwrap() <= 4,
            "stored cache should be trimmed to window"
        );

        // Stash should be consumed (take clears it)
        assert!(
            cache.take_stashed_kv().is_none(),
            "stash should be empty after take"
        );
    }

    #[test]
    fn test_global_cache_stash_matches_stored() {
        // Global cache stores everything, so stash == stored
        let mut cache = Gemma4LayerCache::new_global();

        let keys = MxArray::ones(&[1, 1, 8, 16], None).unwrap();
        let values = MxArray::ones(&[1, 1, 8, 16], None).unwrap();

        let (ret_k, _) = cache.update_and_fetch_stash(&keys, &values).unwrap();
        assert_eq!(ret_k.shape_at(2).unwrap(), 8);

        let (stash_k, _) = cache.take_stashed_kv().unwrap();
        assert_eq!(stash_k.shape_at(2).unwrap(), 8);

        let (cached_k, _) = cache.get_cached_kv().unwrap();
        assert_eq!(
            cached_k.shape_at(2).unwrap(),
            8,
            "global cache stores everything"
        );
    }
}
