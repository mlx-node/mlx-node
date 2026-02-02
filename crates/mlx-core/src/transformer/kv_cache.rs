use crate::array::MxArray;
use napi::bindgen_prelude::*;

/// Key-Value cache for efficient transformer inference.
///
/// Uses pre-allocated buffers with in-place assignment to avoid O(N²) concatenation overhead.
/// Allocates memory in 256-token chunks (matching MLX-LM's step size).
pub struct KVCache {
    keys: Option<MxArray>,
    values: Option<MxArray>,
    offset: i32,
    step: i32,
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KVCache {
    /// Creates a new empty KV cache.
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            step: 256, // Pre-allocate 256 tokens at a time (matching MLX-LM)
        }
    }

    /// Updates the cache with new keys and values, and returns all cached keys/values.
    ///
    /// # Arguments
    /// * `keys` - New keys to add, shape: (batch, n_kv_heads, seq_len, head_dim)
    /// * `values` - New values to add, shape: (batch, n_kv_heads, seq_len, head_dim)
    ///
    /// # Returns
    /// Array containing [cached_keys, cached_values] including the new entries
    pub fn update_and_fetch(
        &mut self,
        keys: &MxArray,
        values: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        // Extract dimensions without copying entire shape vectors
        let batch_size = keys.shape_at(0)?;
        let n_kv_heads = keys.shape_at(1)?;
        let seq_len = keys.shape_at(2)? as i32;
        let k_head_dim = keys.shape_at(3)?;
        let v_head_dim = values.shape_at(3)?;

        let prev = self.offset;

        // Check if we need to grow the buffer
        if self.keys.is_none() || (prev + seq_len) > self.keys.as_ref().unwrap().shape_at(2)? as i32
        {
            // Calculate how many steps we need to allocate
            let n_steps = (self.step + seq_len - 1) / self.step;
            let k_shape = [
                batch_size,
                n_kv_heads,
                n_steps as i64 * self.step as i64,
                k_head_dim,
            ];
            let v_shape = [
                batch_size,
                n_kv_heads,
                n_steps as i64 * self.step as i64,
                v_head_dim,
            ];

            // Pre-allocate new buffer filled with zeros
            let new_k = MxArray::zeros(&k_shape, Some(keys.dtype()?))?;
            let new_v = MxArray::zeros(&v_shape, Some(values.dtype()?))?;

            if let Some(cached_keys) = &self.keys {
                // Align to step boundary if needed
                let cached_keys = if prev % self.step != 0 {
                    cached_keys.slice_axis(2, 0, prev as i64)?
                } else {
                    cached_keys.clone()
                };
                let cached_values = if prev % self.step != 0 {
                    self.values
                        .as_ref()
                        .unwrap()
                        .slice_axis(2, 0, prev as i64)?
                } else {
                    self.values.as_ref().unwrap().clone()
                };

                // Only concatenate when growing buffer (rare!)
                self.keys = Some(MxArray::concatenate(&cached_keys, &new_k, 2)?);
                self.values = Some(MxArray::concatenate(&cached_values, &new_v, 2)?);
            } else {
                // First allocation
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
        }

        // In-place assignment: write new keys/values to pre-allocated buffer
        // This is O(N) instead of O(N²) concatenation!
        self.offset += seq_len;

        // Get mutable references and perform TRUE in-place updates
        // This modifies the pre-allocated buffers directly without creating new arrays!
        if let Some(cached_keys) = self.keys.as_mut() {
            cached_keys.slice_assign_axis_inplace(2, prev as i64, self.offset as i64, keys)?;
        }
        if let Some(cached_values) = self.values.as_mut() {
            cached_values.slice_assign_axis_inplace(2, prev as i64, self.offset as i64, values)?;
        }

        // Return slice of buffer containing valid data [0:offset]
        // This creates new arrays only for the return value
        let result_keys = self
            .keys
            .as_ref()
            .unwrap()
            .slice_axis(2, 0, self.offset as i64)?;
        let result_values = self
            .values
            .as_ref()
            .unwrap()
            .slice_axis(2, 0, self.offset as i64)?;

        Ok((result_keys, result_values))
    }

    /// Resets the cache, clearing all stored keys and values.
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
    }

    /// Returns the current offset (number of cached tokens).
    pub fn get_offset(&self) -> i32 {
        self.offset
    }

    /// Trim the cache to keep only the first `new_len` tokens.
    ///
    /// This is used in speculative decoding to rewind the cache when draft tokens
    /// are rejected. After trimming, subsequent calls to `update_and_fetch` will
    /// overwrite the trimmed portion.
    ///
    /// # Arguments
    /// * `new_len` - New length of the cache (must be <= current offset)
    ///
    /// # Note
    /// This doesn't actually deallocate memory - it just updates the offset.
    /// The next `update_and_fetch` call will overwrite the trimmed data in-place.
    pub fn trim(&mut self, new_len: i32) {
        if new_len < 0 {
            self.offset = 0;
        } else if new_len < self.offset {
            self.offset = new_len;
        }
        // If new_len >= offset, do nothing (can't grow via trim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_shape(arr: &MxArray, expected: &[i64]) {
        let shape = arr.shape().unwrap();
        assert_eq!(shape.len(), expected.len(), "Shape dimension mismatch");
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(shape[i], exp, "Shape mismatch at dimension {}", i);
        }
    }

    #[test]
    fn test_cache_creation() {
        let cache = KVCache::new();
        assert_eq!(cache.get_offset(), 0);
    }

    #[test]
    fn test_cache_default() {
        let cache = KVCache::default();
        assert_eq!(cache.get_offset(), 0);
    }

    #[test]
    fn test_single_update() {
        let mut cache = KVCache::new();
        let keys = MxArray::zeros(&[1, 2, 4, 8], None).unwrap();
        let values = MxArray::zeros(&[1, 2, 4, 8], None).unwrap();

        let (result_k, result_v) = cache.update_and_fetch(&keys, &values).unwrap();

        assert_eq!(cache.get_offset(), 4);
        assert_shape(&result_k, &[1, 2, 4, 8]);
        assert_shape(&result_v, &[1, 2, 4, 8]);
    }

    #[test]
    fn test_multiple_updates() {
        let mut cache = KVCache::new();

        // First update: 4 tokens
        let keys1 = MxArray::zeros(&[1, 2, 4, 8], None).unwrap();
        let values1 = MxArray::zeros(&[1, 2, 4, 8], None).unwrap();
        cache.update_and_fetch(&keys1, &values1).unwrap();
        assert_eq!(cache.get_offset(), 4);

        // Second update: 3 more tokens
        let keys2 = MxArray::zeros(&[1, 2, 3, 8], None).unwrap();
        let values2 = MxArray::zeros(&[1, 2, 3, 8], None).unwrap();
        let (result_k, result_v) = cache.update_and_fetch(&keys2, &values2).unwrap();

        assert_eq!(cache.get_offset(), 7);
        assert_shape(&result_k, &[1, 2, 7, 8]);
        assert_shape(&result_v, &[1, 2, 7, 8]);
    }

    #[test]
    fn test_single_token_updates() {
        let mut cache = KVCache::new();

        // Initial prefill
        let keys1 = MxArray::zeros(&[1, 4, 5, 16], None).unwrap();
        let values1 = MxArray::zeros(&[1, 4, 5, 16], None).unwrap();
        cache.update_and_fetch(&keys1, &values1).unwrap();
        assert_eq!(cache.get_offset(), 5);

        // Single token updates (generation)
        for i in 0..3 {
            let key_token = MxArray::zeros(&[1, 4, 1, 16], None).unwrap();
            let value_token = MxArray::zeros(&[1, 4, 1, 16], None).unwrap();
            let (result_k, _) = cache.update_and_fetch(&key_token, &value_token).unwrap();

            assert_eq!(cache.get_offset(), 5 + i + 1);
            assert_shape(&result_k, &[1, 4, 5 + i as i64 + 1, 16]);
        }
    }

    #[test]
    fn test_reset() {
        let mut cache = KVCache::new();

        let keys = MxArray::zeros(&[2, 8, 6, 32], None).unwrap();
        let values = MxArray::zeros(&[2, 8, 6, 32], None).unwrap();
        cache.update_and_fetch(&keys, &values).unwrap();
        assert_eq!(cache.get_offset(), 6);

        cache.reset();
        assert_eq!(cache.get_offset(), 0);

        // After reset, can add new data
        let keys2 = MxArray::zeros(&[2, 8, 5, 32], None).unwrap();
        let values2 = MxArray::zeros(&[2, 8, 5, 32], None).unwrap();
        let (result_k, _) = cache.update_and_fetch(&keys2, &values2).unwrap();

        assert_eq!(cache.get_offset(), 5);
        assert_shape(&result_k, &[2, 8, 5, 32]);
    }

    #[test]
    fn test_trim() {
        let mut cache = KVCache::new();

        let keys = MxArray::zeros(&[1, 2, 10, 8], None).unwrap();
        let values = MxArray::zeros(&[1, 2, 10, 8], None).unwrap();
        cache.update_and_fetch(&keys, &values).unwrap();
        assert_eq!(cache.get_offset(), 10);

        // Trim to 5
        cache.trim(5);
        assert_eq!(cache.get_offset(), 5);

        // Add more tokens after trim
        let keys2 = MxArray::zeros(&[1, 2, 3, 8], None).unwrap();
        let values2 = MxArray::zeros(&[1, 2, 3, 8], None).unwrap();
        let (result_k, _) = cache.update_and_fetch(&keys2, &values2).unwrap();

        assert_eq!(cache.get_offset(), 8);
        assert_shape(&result_k, &[1, 2, 8, 8]);
    }

    #[test]
    fn test_trim_negative() {
        let mut cache = KVCache::new();

        let keys = MxArray::zeros(&[1, 2, 10, 8], None).unwrap();
        let values = MxArray::zeros(&[1, 2, 10, 8], None).unwrap();
        cache.update_and_fetch(&keys, &values).unwrap();

        // Trim with negative value should reset to 0
        cache.trim(-5);
        assert_eq!(cache.get_offset(), 0);
    }

    #[test]
    fn test_trim_larger_than_offset() {
        let mut cache = KVCache::new();

        let keys = MxArray::zeros(&[1, 2, 10, 8], None).unwrap();
        let values = MxArray::zeros(&[1, 2, 10, 8], None).unwrap();
        cache.update_and_fetch(&keys, &values).unwrap();

        // Trim to larger than offset should do nothing
        cache.trim(100);
        assert_eq!(cache.get_offset(), 10);
    }

    #[test]
    fn test_data_integrity() {
        let mut cache = KVCache::new();

        let keys1 = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]).unwrap();
        let values1 = MxArray::from_float32(&[10.0, 20.0, 30.0, 40.0], &[1, 1, 4, 1]).unwrap();

        let (result_k, result_v) = cache.update_and_fetch(&keys1, &values1).unwrap();
        result_k.eval();
        result_v.eval();
        let keys1_data = result_k.to_float32().unwrap().to_vec();
        let values1_data = result_v.to_float32().unwrap().to_vec();

        assert_eq!(keys1_data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(values1_data, vec![10.0, 20.0, 30.0, 40.0]);

        let keys2 = MxArray::from_float32(&[5.0, 6.0], &[1, 1, 2, 1]).unwrap();
        let values2 = MxArray::from_float32(&[50.0, 60.0], &[1, 1, 2, 1]).unwrap();

        let (result_k2, result_v2) = cache.update_and_fetch(&keys2, &values2).unwrap();
        result_k2.eval();
        result_v2.eval();
        let keys2_data = result_k2.to_float32().unwrap().to_vec();
        let values2_data = result_v2.to_float32().unwrap().to_vec();

        assert_eq!(keys2_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(values2_data, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn test_batch_size_greater_than_one() {
        let mut cache = KVCache::new();

        let keys = MxArray::zeros(&[4, 2, 8, 16], None).unwrap();
        let values = MxArray::zeros(&[4, 2, 8, 16], None).unwrap();
        let (result_k, result_v) = cache.update_and_fetch(&keys, &values).unwrap();

        assert_eq!(cache.get_offset(), 8);
        assert_shape(&result_k, &[4, 2, 8, 16]);
        assert_shape(&result_v, &[4, 2, 8, 16]);
    }
}
