use crate::array::MxArray;
use crate::transformer::rotating_kv_cache::{RotatingKVCacheSnapshot, RotatingKVCacheState};
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

    /// Returns true when this Gemma layer owns a sliding rotating cache.
    pub fn is_sliding(&self) -> bool {
        matches!(self.inner, CacheType::Sliding(_))
    }

    /// Return logical state for a sliding cache.
    ///
    /// Global layers return `None` so callers can iterate over all Gemma4
    /// layers without separately checking the layer kind.
    pub fn sliding_state(&self) -> Result<Option<RotatingKVCacheState>> {
        match &self.inner {
            CacheType::Global(_) => Ok(None),
            CacheType::Sliding(c) => Ok(Some(c.state()?)),
        }
    }

    /// Check that a sliding cache is initialized and aligned to `offset`.
    pub fn sliding_offset_matches(&self, offset: i32) -> Result<bool> {
        match &self.inner {
            CacheType::Global(_) => Ok(false),
            CacheType::Sliding(c) => {
                let state = c.state()?;
                Ok(state.initialized && state.offset == offset)
            }
        }
    }

    /// Snapshot a sliding cache's ordered K/V tail.
    ///
    /// Global layers and empty sliding caches return `None`.
    pub fn snapshot_sliding(&self) -> Result<Option<RotatingKVCacheSnapshot>> {
        match &self.inner {
            CacheType::Global(_) => Ok(None),
            CacheType::Sliding(c) => c.snapshot(),
        }
    }

    /// Restore a sliding cache from an ordered K/V tail snapshot.
    ///
    /// This intentionally errors on global layers to prevent accidentally
    /// loading sliding-window state into a full-attention cache.
    pub fn restore_sliding_snapshot(&mut self, snapshot: &RotatingKVCacheSnapshot) -> Result<()> {
        match &mut self.inner {
            CacheType::Global(_) => Err(Error::new(
                Status::InvalidArg,
                "cannot restore a sliding snapshot into a Gemma4 global cache",
            )),
            CacheType::Sliding(c) => {
                c.restore_snapshot(snapshot)?;
                self.stashed_kv = None;
                Ok(())
            }
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

    fn assert_float_data(arr: &MxArray, expected: &[f32]) {
        arr.eval();
        let data = arr.to_float32().unwrap().to_vec();
        assert_eq!(data, expected);
    }

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

    #[test]
    fn test_sliding_snapshot_restore_after_wrap() {
        let mut source = Gemma4LayerCache::new_sliding(4);

        let keys1 = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]).unwrap();
        let values1 = MxArray::from_float32(&[10.0, 20.0, 30.0, 40.0], &[1, 1, 4, 1]).unwrap();
        source.update_and_fetch(&keys1, &values1).unwrap();

        let keys2 = MxArray::from_float32(&[5.0], &[1, 1, 1, 1]).unwrap();
        let values2 = MxArray::from_float32(&[50.0], &[1, 1, 1, 1]).unwrap();
        source.update_and_fetch(&keys2, &values2).unwrap();

        let keys3 = MxArray::from_float32(&[6.0], &[1, 1, 1, 1]).unwrap();
        let values3 = MxArray::from_float32(&[60.0], &[1, 1, 1, 1]).unwrap();
        source.update_and_fetch(&keys3, &values3).unwrap();

        assert!(source.is_sliding());
        assert!(source.sliding_offset_matches(6).unwrap());
        let source_state = source.sliding_state().unwrap().unwrap();
        assert_eq!(source_state.offset, 6);
        assert_eq!(source_state.cached_tokens, 4);

        let snapshot = source.snapshot_sliding().unwrap().unwrap();
        let mut restored = Gemma4LayerCache::new_sliding(4);
        restored.restore_sliding_snapshot(&snapshot).unwrap();
        assert!(restored.sliding_offset_matches(6).unwrap());

        let (restored_keys, restored_values) = restored.get_cached_kv().unwrap();
        assert_float_data(&restored_keys, &[3.0, 4.0, 5.0, 6.0]);
        assert_float_data(&restored_values, &[30.0, 40.0, 50.0, 60.0]);

        let append_keys = MxArray::from_float32(&[7.0, 8.0], &[1, 1, 2, 1]).unwrap();
        let append_values = MxArray::from_float32(&[70.0, 80.0], &[1, 1, 2, 1]).unwrap();
        let (attention_keys, attention_values) = restored
            .update_and_fetch(&append_keys, &append_values)
            .unwrap();
        assert_float_data(&attention_keys, &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_float_data(&attention_values, &[30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);

        let (restored_tail, _) = restored.get_cached_kv().unwrap();
        assert_float_data(&restored_tail, &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_sliding_snapshot_not_restored_into_global_cache() {
        let mut sliding = Gemma4LayerCache::new_sliding(4);
        let keys = MxArray::ones(&[1, 1, 4, 1], None).unwrap();
        let values = MxArray::ones(&[1, 1, 4, 1], None).unwrap();
        sliding.update_and_fetch(&keys, &values).unwrap();
        let snapshot = sliding.snapshot_sliding().unwrap().unwrap();

        let mut global = Gemma4LayerCache::new_global();
        assert!(!global.is_sliding());
        assert!(global.snapshot_sliding().unwrap().is_none());
        assert!(global.restore_sliding_snapshot(&snapshot).is_err());
    }
}
