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
    /// Rows the first allocation must hold (set by [`Self::reserve`] while
    /// the cache is still empty); 0 when nothing is reserved.
    pending_rows: i64,
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
            pending_rows: 0,
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
        let needs_grow = match &self.keys {
            Some(cached_keys) => (prev + seq_len) > cached_keys.shape_at(2)? as i32,
            None => true,
        };
        if needs_grow {
            // Calculate how many steps we need to allocate
            let n_steps = (self.step + seq_len - 1) / self.step;
            let step_rows = n_steps as i64 * self.step as i64;
            let new_rows = if self.keys.is_some() {
                step_rows
            } else {
                step_rows.max(std::mem::take(&mut self.pending_rows))
            };
            let k_shape = [batch_size, n_kv_heads, new_rows, k_head_dim];
            let v_shape = [batch_size, n_kv_heads, new_rows, v_head_dim];

            // Pre-allocate new buffer filled with zeros
            let new_k = MxArray::zeros(&k_shape, Some(keys.dtype()?))?;
            let new_v = MxArray::zeros(&v_shape, Some(values.dtype()?))?;

            // Align to step boundary if needed; only concatenate when growing
            // the buffer (rare!)
            let keep_rows = if prev % self.step != 0 {
                Some(prev as i64)
            } else {
                None
            };
            self.append_rows(keep_rows, new_k, new_v)?;
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
            .ok_or_else(|| Error::from_reason("KV cache keys missing after buffer update"))?
            .slice_axis(2, 0, self.offset as i64)?;
        let result_values = self
            .values
            .as_ref()
            .ok_or_else(|| Error::from_reason("KV cache values missing after buffer update"))?
            .slice_axis(2, 0, self.offset as i64)?;

        Ok((result_keys, result_values))
    }

    /// Resets the cache, clearing all stored keys and values.
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
        self.pending_rows = 0;
    }

    /// Make the buffer hold at least `rows` rows (rounded up to the step), so
    /// appends up to that frontier never grow it. An empty cache only records
    /// the size for its first allocation; a live cache copies `[0:offset]`
    /// into the larger buffer once. Rows past `offset` are never read.
    pub(crate) fn reserve(&mut self, rows: i64) -> Result<()> {
        if rows <= 0 {
            return Ok(());
        }
        let step = self.step as i64;
        let target = rows
            .checked_add(step - 1)
            .map(|rows| rows / step * step)
            .filter(|&target| target <= i32::MAX as i64)
            .ok_or_else(|| {
                Error::from_reason(format!("KV cache reservation of {rows} rows is too large"))
            })?;
        let Some(keys) = &self.keys else {
            self.pending_rows = self.pending_rows.max(target);
            return Ok(());
        };
        if self.capacity()? >= rows {
            return Ok(());
        }
        let values = self
            .values
            .as_ref()
            .ok_or_else(|| Error::from_reason("KV cache values missing while keys are present"))?;
        let extra = target - self.offset as i64;
        let zeros_like = |cached: &MxArray| -> Result<MxArray> {
            MxArray::zeros(
                &[
                    cached.shape_at(0)?,
                    cached.shape_at(1)?,
                    extra,
                    cached.shape_at(3)?,
                ],
                Some(cached.dtype()?),
            )
        };
        let new_k = zeros_like(keys)?;
        let new_v = zeros_like(values)?;
        self.append_rows(Some(self.offset as i64), new_k, new_v)
    }

    /// Rows the key buffer can hold without growing (0 before the first
    /// allocation).
    pub(crate) fn capacity(&self) -> Result<i64> {
        self.keys.as_ref().map_or(Ok(0), |keys| keys.shape_at(2))
    }

    /// Replace the buffers with `buffer[0:keep_rows]` (the whole buffer when
    /// `None`) followed by `new_k` / `new_v`, or with `new_k` / `new_v` alone
    /// before the first allocation or when no row is kept.
    fn append_rows(
        &mut self,
        keep_rows: Option<i64>,
        new_k: MxArray,
        new_v: MxArray,
    ) -> Result<()> {
        let (Some(cached_keys), Some(cached_values)) = (&self.keys, &self.values) else {
            if self.keys.is_some() {
                return Err(Error::from_reason(
                    "KV cache values missing while keys are present",
                ));
            }
            self.keys = Some(new_k);
            self.values = Some(new_v);
            return Ok(());
        };
        if keep_rows == Some(0) {
            self.keys = Some(new_k);
            self.values = Some(new_v);
            return Ok(());
        }
        let keep = |cached: &MxArray| -> Result<MxArray> {
            match keep_rows {
                Some(rows) => cached.slice_axis(2, 0, rows),
                None => Ok(cached.clone()),
            }
        };
        let (kept_keys, kept_values) = (keep(cached_keys)?, keep(cached_values)?);
        self.keys = Some(MxArray::concatenate(&kept_keys, &new_k, 2)?);
        self.values = Some(MxArray::concatenate(&kept_values, &new_v, 2)?);
        Ok(())
    }

    /// Returns the current offset (number of cached tokens).
    pub fn get_offset(&self) -> i32 {
        self.offset
    }

    /// Get a reference to the cached keys.
    pub fn keys_ref(&self) -> Option<&MxArray> {
        self.keys.as_ref()
    }

    /// Get a reference to the cached values.
    pub fn values_ref(&self) -> Option<&MxArray> {
        self.values.as_ref()
    }

    /// Consume the cache and transfer its backing arrays to a different
    /// decoder implementation without cloning their reference-counted
    /// handles. This is used when a dense prefill hands ownership to a fused
    /// token-step kernel.
    pub(crate) fn into_parts(self) -> (Option<MxArray>, Option<MxArray>, i32) {
        (self.keys, self.values, self.offset)
    }

    /// Set the cached keys directly (used by fused forward pass).
    pub fn set_keys(&mut self, keys: MxArray) {
        self.keys = Some(keys);
    }

    /// Set the cached values directly (used by fused forward pass).
    pub fn set_values(&mut self, values: MxArray) {
        self.values = Some(values);
    }

    /// Set the cache offset directly (used by fused forward pass).
    pub fn set_offset(&mut self, offset: i32) {
        self.offset = offset;
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

    fn rows(n: i64, seed: f64) -> (MxArray, MxArray) {
        let k = MxArray::random_normal(&[1, 2, n, 8], seed, 1.0, None).unwrap();
        let v = MxArray::random_normal(&[1, 2, n, 8], -seed, 1.0, None).unwrap();
        (k, v)
    }

    fn bits(arr: &MxArray) -> (Vec<i64>, Vec<u32>) {
        arr.eval();
        (
            arr.shape().unwrap().to_vec(),
            arr.to_float32()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect(),
        )
    }

    #[test]
    fn reserve_before_first_append_allocates_reserved_rows_once() {
        let mut cache = KVCache::new();
        cache.reserve(600).unwrap();
        assert_eq!(
            cache.capacity().unwrap(),
            0,
            "reserve on an empty cache is lazy"
        );
        let (k, v) = rows(4, 0.5);
        cache.update_and_fetch(&k, &v).unwrap();
        assert_eq!(cache.capacity().unwrap(), 768);
        assert_eq!(cache.get_offset(), 4);
        while cache.get_offset() < 600 {
            let (k, v) = rows(8, 0.25);
            cache.update_and_fetch(&k, &v).unwrap();
            assert_eq!(cache.capacity().unwrap(), 768);
        }
    }

    #[test]
    fn reserve_after_append_copies_prefix_once_and_is_bit_identical() {
        let mut plain = KVCache::new();
        let mut reserved = KVCache::new();
        let (k, v) = rows(300, 1.0);
        plain.update_and_fetch(&k, &v).unwrap();
        reserved.update_and_fetch(&k, &v).unwrap();
        reserved.reserve(2000).unwrap();
        assert_eq!(reserved.capacity().unwrap(), 2048);
        assert_eq!(reserved.get_offset(), 300);

        let mut plain_capacities = vec![plain.capacity().unwrap()];
        for (cycle, keep) in [8, 1, 5, 8, 3, 8, 2, 7]
            .iter()
            .cycle()
            .take(200)
            .enumerate()
        {
            let (k, v) = rows(8, cycle as f64 * 0.01);
            let base = plain.get_offset();
            let (pk, pv) = plain.update_and_fetch(&k, &v).unwrap();
            let (rk, rv) = reserved.update_and_fetch(&k, &v).unwrap();
            assert_eq!(bits(&pk), bits(&rk), "keys cycle={cycle}");
            assert_eq!(bits(&pv), bits(&rv), "values cycle={cycle}");
            plain.trim(base + keep);
            reserved.trim(base + keep);
            assert_eq!(reserved.capacity().unwrap(), 2048, "cycle={cycle}");
            plain_capacities.push(plain.capacity().unwrap());
        }
        assert!(
            reserved.get_offset() > 300 + 4 * 256,
            "the trace must cross four 256-row boundaries"
        );
        plain_capacities.dedup();
        assert!(
            plain_capacities.len() >= 5,
            "the unreserved twin must grow at every boundary: {plain_capacities:?}"
        );
    }

    #[test]
    fn reserve_is_noop_when_capacity_suffices() {
        let mut cache = KVCache::new();
        let (k, v) = rows(300, 1.0);
        cache.update_and_fetch(&k, &v).unwrap();
        let before = cache.keys_ref().unwrap().clone();
        cache.reserve(512).unwrap();
        cache.reserve(400).unwrap();
        assert_eq!(cache.capacity().unwrap(), 512);
        assert_eq!(bits(cache.keys_ref().unwrap()), bits(&before));
    }

    #[test]
    fn reserve_nonpositive_is_noop() {
        let mut cache = KVCache::new();
        cache.reserve(0).unwrap();
        cache.reserve(-5).unwrap();
        let (k, v) = rows(4, 1.0);
        cache.update_and_fetch(&k, &v).unwrap();
        assert_eq!(cache.capacity().unwrap(), 256);
        assert!(
            cache.reserve(i32::MAX as i64).is_err(),
            "rounding past i32 must fail"
        );
    }

    #[test]
    fn reset_clears_pending_rows() {
        let mut cache = KVCache::new();
        cache.reserve(4096).unwrap();
        cache.reset();
        let (k, v) = rows(300, 1.0);
        cache.update_and_fetch(&k, &v).unwrap();
        assert_eq!(cache.capacity().unwrap(), 512);
    }

    #[test]
    fn unreserved_growth_is_unchanged() {
        let mut cache = KVCache::new();
        let (k, v) = rows(300, 1.0);
        cache.update_and_fetch(&k, &v).unwrap();
        assert_eq!(cache.capacity().unwrap(), 512);
        cache.trim(257);
        let (k, v) = rows(300, 2.0);
        cache.update_and_fetch(&k, &v).unwrap();
        assert_eq!(cache.capacity().unwrap(), 257 + 2 * 256);
        cache.trim(512);
        let (k, v) = rows(300, 3.0);
        cache.update_and_fetch(&k, &v).unwrap();
        assert_eq!(
            cache.capacity().unwrap(),
            769 + 2 * 256,
            "a step-aligned frontier keeps the whole old buffer"
        );
    }
}
