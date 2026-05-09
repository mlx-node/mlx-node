use crate::array::MxArray;
use napi::bindgen_prelude::*;

/// Rotating Key-Value cache with fixed maximum size (internal).
///
/// Once the cache reaches `max_size`, old entries are evicted (except for `keep` tokens).
/// Used internally by transformer models for long context generation.
pub struct RotatingKVCache {
    keys: Option<MxArray>,
    values: Option<MxArray>,
    offset: i32,
    max_size: i32,
    keep: i32,
    idx: i32,
}

/// Cheap snapshot of a [`RotatingKVCache`] valid K/V tail.
///
/// `keys` and `values` are MLX arrays in temporal attention order. Cloning this
/// struct clones array handles; it does not read K/V data back to the host.
#[derive(Clone)]
pub struct RotatingKVCacheSnapshot {
    pub keys: MxArray,
    pub values: MxArray,
    /// Total number of tokens processed by the cache when snapshotted.
    pub offset: i32,
    /// Sliding window size used by the source cache.
    pub max_size: i32,
    /// Number of leading tokens preserved by the source cache.
    pub keep: i32,
    /// Number of valid cached tokens stored in `keys`/`values`.
    pub cached_tokens: i32,
}

/// Lightweight logical state for validating a rotating cache without reading
/// tensor contents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RotatingKVCacheState {
    pub offset: i32,
    pub window_size: i32,
    pub keep: i32,
    pub idx: i32,
    pub cached_tokens: i32,
    pub initialized: bool,
}

impl RotatingKVCache {
    /// Creates a new rotating KV cache.
    pub fn new(max_size: i32, keep: Option<i32>) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            max_size,
            keep: keep.unwrap_or(0),
            idx: 0,
        }
    }

    pub fn keys_ref(&self) -> Option<&MxArray> {
        self.keys.as_ref()
    }

    pub fn values_ref(&self) -> Option<&MxArray> {
        self.values.as_ref()
    }

    fn validate_config(max_size: i32, keep: i32) -> Result<()> {
        if max_size <= 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("RotatingKVCache max_size must be positive, got {max_size}"),
            ));
        }
        if keep < 0 || keep > max_size {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "RotatingKVCache keep must be in [0, max_size], got keep={keep} max_size={max_size}"
                ),
            ));
        }
        Ok(())
    }

    fn expected_cached_tokens(offset: i32, max_size: i32) -> Result<i32> {
        if offset < 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("RotatingKVCache offset must be non-negative, got {offset}"),
            ));
        }
        if offset == 0 {
            Ok(0)
        } else {
            Ok(std::cmp::min(offset, max_size))
        }
    }

    /// Helper: Trim cache by removing middle section
    fn trim(&self, trim_size: i32, v: &MxArray, append: Option<&MxArray>) -> Result<MxArray> {
        let mut to_cat: Vec<MxArray> = Vec::new();
        let v_shape = v.shape()?;

        if trim_size > 0 {
            let starts = vec![0, 0, 0, 0];
            let stops = vec![v_shape[0], v_shape[1], self.keep as i64, v_shape[3]];
            let keep_start = v.slice(&starts, &stops)?;
            to_cat.push(keep_start);

            let starts = vec![0, 0, (trim_size + self.keep) as i64, 0];
            let stops = vec![v_shape[0], v_shape[1], v_shape[2], v_shape[3]];
            let keep_end = v.slice(&starts, &stops)?;
            to_cat.push(keep_end);
        } else {
            to_cat.push(v.clone());
        }

        if let Some(append_arr) = append {
            to_cat.push(append_arr.clone());
        }

        let refs: Vec<&MxArray> = to_cat.iter().collect();
        MxArray::concatenate_many(refs, Some(2))
    }

    /// Select the bounded K/V tail that should remain stored after a
    /// multi-token append. The attention view may be larger than `max_size`
    /// (`previous_window + current_chunk`), but the rotating cache storage
    /// itself must stay within `max_size`.
    fn bounded_storage_from_attention_view(&self, attention: &MxArray) -> Result<MxArray> {
        let shape = attention.shape()?;
        let total_len = shape[2] as i32;
        if total_len <= self.max_size {
            return Ok(attention.clone());
        }

        if self.keep > 0 {
            let keep_len = self.keep.min(self.max_size);
            let tail_len = self.max_size - keep_len;
            let keep = attention.slice_axis(2, 0, keep_len as i64)?;
            if tail_len == 0 {
                return Ok(keep);
            }
            let tail_start = (total_len - tail_len) as i64;
            let tail = attention.slice_axis(2, tail_start, total_len as i64)?;
            MxArray::concatenate(&keep, &tail, 2)
        } else {
            let start = (total_len - self.max_size) as i64;
            attention.slice_axis(2, start, total_len as i64)
        }
    }

    /// Helper: Rearrange cache into temporal order
    fn temporal_order(&self, v: &MxArray) -> Result<MxArray> {
        let v_shape = v.shape()?;
        let cache_len = v_shape[2] as i32;

        if self.idx == cache_len {
            Ok(v.clone())
        } else if self.idx < self.offset {
            let starts = vec![0, 0, 0, 0];
            let stops = vec![v_shape[0], v_shape[1], self.keep as i64, v_shape[3]];
            let keep_section = v.slice(&starts, &stops)?;

            let starts = vec![0, 0, self.idx as i64, 0];
            let stops = vec![v_shape[0], v_shape[1], v_shape[2], v_shape[3]];
            let tail = v.slice(&starts, &stops)?;

            if self.keep < self.idx {
                let starts = vec![0, 0, self.keep as i64, 0];
                let stops = vec![v_shape[0], v_shape[1], self.idx as i64, v_shape[3]];
                let middle = v.slice(&starts, &stops)?;
                MxArray::concatenate_many(vec![&keep_section, &tail, &middle], Some(2))
            } else {
                MxArray::concatenate(&keep_section, &tail, 2)
            }
        } else {
            let starts = [0, 0, 0, 0];
            let stops = [v_shape[0], v_shape[1], self.idx as i64, v_shape[3]];
            v.slice(&starts, &stops)
        }
    }

    /// Update with concatenation (multi-token updates)
    fn update_concat(&mut self, keys: &MxArray, values: &MxArray) -> Result<Vec<MxArray>> {
        let seq_len = keys.shape_at(2)? as i32;

        if self.keys.is_none() {
            // If initial sequence exceeds max_size, trim to keep only the tail
            let (stored_keys, stored_values, stored_idx) = if seq_len > self.max_size {
                if self.keep > 0 {
                    // Preserve first `keep` tokens + last `max_size - keep` tokens.
                    let kept_keys = keys.slice_axis(2, 0, self.keep as i64)?;
                    let kept_values = values.slice_axis(2, 0, self.keep as i64)?;
                    let tail_len = self.max_size - self.keep;
                    let tail_start = (seq_len - tail_len) as i64;
                    let tail_keys = keys.slice_axis(2, tail_start, seq_len as i64)?;
                    let tail_values = values.slice_axis(2, tail_start, seq_len as i64)?;
                    let trimmed_keys = MxArray::concatenate(&kept_keys, &tail_keys, 2)?;
                    let trimmed_values = MxArray::concatenate(&kept_values, &tail_values, 2)?;
                    (trimmed_keys, trimmed_values, self.max_size)
                } else {
                    let start = (seq_len - self.max_size) as i64;
                    let trimmed_keys = keys.slice_axis(2, start, seq_len as i64)?;
                    let trimmed_values = values.slice_axis(2, start, seq_len as i64)?;
                    (trimmed_keys, trimmed_values, self.max_size)
                }
            } else {
                (keys.clone(), values.clone(), seq_len)
            };

            self.keys = Some(stored_keys);
            self.values = Some(stored_values);
            self.offset = seq_len; // offset tracks TOTAL tokens seen, not stored
            self.idx = stored_idx;

            // Return the FULL (untrimmed) keys/values for prefill attention.
            // The caller needs to attend over the complete sequence during prefill,
            // even though only the window is stored internally.
            return Ok(vec![keys.clone(), values.clone()]);
        }

        let ordered_keys = self.temporal_order(self.keys.as_ref().unwrap())?;
        let ordered_values = self.temporal_order(self.values.as_ref().unwrap())?;

        let current_len = ordered_keys.shape_at(2)? as i32;
        self.idx = current_len;

        // For multi-token prefill, attention must see the previous sliding
        // window plus the entire current chunk. Store the trimmed rotating
        // window below, but return the untrimmed attention view just like the
        // empty-cache branch does for an over-window initial prefill.
        let attention_keys = MxArray::concatenate(&ordered_keys, keys, 2)?;
        let attention_values = MxArray::concatenate(&ordered_values, values, 2)?;

        let new_keys = self.bounded_storage_from_attention_view(&attention_keys)?;
        let new_values = self.bounded_storage_from_attention_view(&attention_values)?;

        self.keys = Some(new_keys.clone());
        self.values = Some(new_values.clone());
        self.offset += seq_len;
        self.idx = new_keys.shape_at(2)? as i32;

        Ok(vec![attention_keys, attention_values])
    }

    /// Update in-place (single-token updates)
    fn update_in_place(&mut self, keys: &MxArray, values: &MxArray) -> Result<Vec<MxArray>> {
        let batch_size = keys.shape_at(0)? as i32;
        let n_kv_heads = keys.shape_at(1)? as i32;
        let seq_len = keys.shape_at(2)? as i32;
        let k_head_dim = keys.shape_at(3)? as i32;
        let v_head_dim = values.shape_at(3)? as i32;

        let prev = self.offset;

        let needs_grow = if let Some(existing_keys) = &self.keys {
            let cache_size = existing_keys.shape_at(2)? as i32;
            prev >= cache_size && cache_size < self.max_size
        } else {
            true
        };

        if needs_grow {
            let new_size = std::cmp::min(256, self.max_size - prev);
            let k_shape = vec![
                batch_size as i64,
                n_kv_heads as i64,
                new_size as i64,
                k_head_dim as i64,
            ];
            let v_shape = vec![
                batch_size as i64,
                n_kv_heads as i64,
                new_size as i64,
                v_head_dim as i64,
            ];

            let new_k = MxArray::zeros(&k_shape, Some(keys.dtype()?))?;
            let new_v = MxArray::zeros(&v_shape, Some(values.dtype()?))?;

            if let Some(existing_keys) = &self.keys {
                self.keys = Some(MxArray::concatenate(existing_keys, &new_k, 2)?);
                self.values = Some(MxArray::concatenate(
                    self.values.as_ref().unwrap(),
                    &new_v,
                    2,
                )?);
            } else {
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
            self.idx = prev;
        }

        let self_keys = self.keys.as_ref().ok_or_else(|| {
            Error::new(Status::InvalidArg, "Keys not existing on rotating kv cache")
        })?;
        let self_values = self.values.as_ref().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "Values not existing on rotating kv cache",
            )
        })?;

        let current_cache_size = self_keys.shape_at(2)? as i32;
        let trim_size = current_cache_size - self.max_size;
        if trim_size > 0 {
            let trimmed_keys = self.trim(trim_size, self_keys, None)?;
            let trimmed_values = self.trim(trim_size, self_values, None)?;
            self.keys = Some(trimmed_keys);
            self.values = Some(trimmed_values);
            self.idx = self.max_size;
        }

        if self.idx == self.max_size {
            self.idx = self.keep;
        }

        let start = self.idx;
        let end = self.idx + seq_len;

        if let Some(cache_keys) = self.keys.as_mut() {
            cache_keys.slice_assign_axis_inplace(2, start as i64, end as i64, keys)?;
        }
        if let Some(cache_values) = self.values.as_mut() {
            cache_values.slice_assign_axis_inplace(2, start as i64, end as i64, values)?;
        }
        self.offset += seq_len;
        self.idx += seq_len;

        let self_keys = self.keys.as_ref().ok_or_else(|| {
            Error::new(Status::InvalidArg, "Keys not existing on rotating kv cache")
        })?;
        let self_values = self.values.as_ref().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "Values not existing on rotating kv cache",
            )
        })?;

        if self.offset < self.max_size {
            let result_shape = self_keys.shape()?;
            let starts = [0, 0, 0, 0];
            let stops = [
                result_shape[0],
                result_shape[1],
                self.offset as i64,
                result_shape[3],
            ];
            let keys_result = self_keys.slice(&starts, &stops)?;
            let values_result = self_values.slice(&starts, &stops)?;
            return Ok(vec![keys_result, values_result]);
        }

        Ok(vec![self_keys.clone(), self_values.clone()])
    }

    /// Updates the cache with new keys and values.
    pub fn update_and_fetch(&mut self, keys: &MxArray, values: &MxArray) -> Result<Vec<MxArray>> {
        let seq_len = keys.shape_at(2)? as i32;
        if seq_len == 1 {
            self.update_in_place(keys, values)
        } else {
            self.update_concat(keys, values)
        }
    }

    /// Resets the cache.
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
        self.idx = 0;
    }

    /// Returns whether the cache has allocated K/V arrays.
    pub fn is_initialized(&self) -> bool {
        self.keys.is_some() && self.values.is_some()
    }

    /// Returns the current offset.
    pub fn get_offset(&self) -> i32 {
        self.offset
    }

    /// Returns the maximum cache size.
    pub fn get_max_size(&self) -> i32 {
        self.max_size
    }

    /// Returns the number of tokens to keep.
    pub fn get_keep(&self) -> i32 {
        self.keep
    }

    /// Returns the current write index.
    pub fn get_idx(&self) -> i32 {
        self.idx
    }

    /// Returns the logical number of valid cached tokens.
    ///
    /// This can be smaller than the raw backing array length after single-token
    /// growth because the backing array may be preallocated ahead of `offset`.
    pub fn get_cached_token_count(&self) -> Result<i32> {
        if !self.is_initialized() {
            return Ok(0);
        }
        Self::expected_cached_tokens(self.offset, self.max_size)
    }

    /// Returns a compact state summary suitable for offset/window validation.
    pub fn state(&self) -> Result<RotatingKVCacheState> {
        Ok(RotatingKVCacheState {
            offset: self.offset,
            window_size: self.max_size,
            keep: self.keep,
            idx: self.idx,
            cached_tokens: self.get_cached_token_count()?,
            initialized: self.is_initialized(),
        })
    }

    /// Snapshot the current valid K/V tail in temporal attention order.
    ///
    /// Returns `None` for an empty cache. The returned arrays are lazy MLX array
    /// views/concats over existing device state; no host K/V readback happens.
    pub fn snapshot(&self) -> Result<Option<RotatingKVCacheSnapshot>> {
        Self::validate_config(self.max_size, self.keep)?;
        if self.offset == 0 {
            return Ok(None);
        }

        let keys = self.keys.as_ref().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "RotatingKVCache snapshot requested with offset but no keys",
            )
        })?;
        let values = self.values.as_ref().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "RotatingKVCache snapshot requested with offset but no values",
            )
        })?;

        let ordered_keys = self.temporal_order(keys)?;
        let ordered_values = self.temporal_order(values)?;
        let cached_tokens = ordered_keys.shape_at(2)? as i32;
        let value_cached_tokens = ordered_values.shape_at(2)? as i32;
        let expected_cached_tokens = Self::expected_cached_tokens(self.offset, self.max_size)?;

        if cached_tokens != value_cached_tokens {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "RotatingKVCache snapshot K/V token counts differ: keys={cached_tokens} values={value_cached_tokens}"
                ),
            ));
        }
        if cached_tokens != expected_cached_tokens {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "RotatingKVCache snapshot has {cached_tokens} cached tokens but expected {expected_cached_tokens} for offset={} max_size={}",
                    self.offset, self.max_size
                ),
            ));
        }

        Ok(Some(RotatingKVCacheSnapshot {
            keys: ordered_keys,
            values: ordered_values,
            offset: self.offset,
            max_size: self.max_size,
            keep: self.keep,
            cached_tokens,
        }))
    }

    /// Restore a previously snapshotted ordered K/V tail.
    ///
    /// The restored cache has the same logical `offset` as the source cache and
    /// stores the snapshot in temporal order. Setting `idx` to the cached length
    /// makes the next append observe the same attention view as a cache that had
    /// naturally processed the prefix.
    pub fn restore_snapshot(&mut self, snapshot: &RotatingKVCacheSnapshot) -> Result<()> {
        Self::validate_config(snapshot.max_size, snapshot.keep)?;
        if snapshot.max_size != self.max_size || snapshot.keep != self.keep {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "RotatingKVCache snapshot config mismatch: cache max_size={} keep={} snapshot max_size={} keep={}",
                    self.max_size, self.keep, snapshot.max_size, snapshot.keep
                ),
            ));
        }

        let keys_shape = snapshot.keys.shape()?;
        let values_shape = snapshot.values.shape()?;
        if keys_shape.len() != 4 || values_shape.len() != 4 {
            let keys_shape_vec = keys_shape.to_vec();
            let values_shape_vec = values_shape.to_vec();
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "RotatingKVCache snapshot expects 4D K/V tensors, got keys={keys_shape_vec:?} values={values_shape_vec:?}"
                ),
            ));
        }
        if keys_shape[0] != values_shape[0]
            || keys_shape[1] != values_shape[1]
            || keys_shape[2] != values_shape[2]
        {
            let keys_shape_vec = keys_shape.to_vec();
            let values_shape_vec = values_shape.to_vec();
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "RotatingKVCache snapshot K/V leading shapes differ: keys={keys_shape_vec:?} values={values_shape_vec:?}"
                ),
            ));
        }

        let cached_tokens = keys_shape[2] as i32;
        let expected_cached_tokens =
            Self::expected_cached_tokens(snapshot.offset, snapshot.max_size)?;
        if cached_tokens != snapshot.cached_tokens
            || cached_tokens != expected_cached_tokens
            || cached_tokens == 0
        {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "RotatingKVCache snapshot token count mismatch: shape_tokens={cached_tokens} snapshot_tokens={} expected={expected_cached_tokens}",
                    snapshot.cached_tokens
                ),
            ));
        }

        self.keys = Some(snapshot.keys.clone());
        self.values = Some(snapshot.values.clone());
        self.offset = snapshot.offset;
        self.idx = cached_tokens;
        Ok(())
    }

    /// Get the current cached K/V in temporal order.
    ///
    /// Returns the valid portion of the cache rearranged into temporal order.
    /// This is useful for KV cache sharing where another layer needs to read
    /// this cache's contents without modifying it.
    ///
    /// Returns None if the cache is empty.
    pub fn fetch_current_kv(&self) -> Option<(MxArray, MxArray)> {
        if self.offset == 0 {
            return None;
        }
        let keys = self.keys.as_ref()?;
        let values = self.values.as_ref()?;
        let ordered_keys = self.temporal_order(keys).ok()?;
        let ordered_values = self.temporal_order(values).ok()?;
        Some((ordered_keys, ordered_values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_kv(batch: i64, heads: i64, seq: i64, dim: i64) -> (MxArray, MxArray) {
        let keys = MxArray::random_normal(&[batch, heads, seq, dim], 0.0, 1.0, None).unwrap();
        let values = MxArray::random_normal(&[batch, heads, seq, dim], 0.0, 1.0, None).unwrap();
        (keys, values)
    }

    fn assert_shape(arr: &MxArray, expected: &[i64]) {
        let shape = arr.shape().unwrap();
        assert_eq!(shape.len(), expected.len(), "Shape dimension mismatch");
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(shape[i], exp, "Shape mismatch at dimension {}", i);
        }
    }

    fn assert_float_data(arr: &MxArray, expected: &[f32]) {
        arr.eval();
        let data = arr.to_float32().unwrap().to_vec();
        assert_eq!(data, expected);
    }

    #[test]
    fn test_cache_creation() {
        let cache = RotatingKVCache::new(8, None);
        assert_eq!(cache.get_offset(), 0);
        assert_eq!(cache.get_max_size(), 8);
        assert_eq!(cache.get_keep(), 0);
        assert_eq!(cache.get_idx(), 0);
    }

    #[test]
    fn test_cache_creation_with_keep() {
        let cache = RotatingKVCache::new(8, Some(2));
        assert_eq!(cache.get_max_size(), 8);
        assert_eq!(cache.get_keep(), 2);
    }

    #[test]
    fn test_update_below_max_size() {
        let mut cache = RotatingKVCache::new(10, None);
        let (keys, values) = random_kv(1, 2, 6, 8);

        let result = cache.update_and_fetch(&keys, &values).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(cache.get_offset(), 6);
        assert_shape(&result[0], &[1, 2, 6, 8]);
        assert_shape(&result[1], &[1, 2, 6, 8]);
    }

    #[test]
    fn test_multiple_updates_below_max() {
        let mut cache = RotatingKVCache::new(10, None);

        let (keys1, values1) = random_kv(1, 2, 4, 8);
        cache.update_and_fetch(&keys1, &values1).unwrap();

        let (keys2, values2) = random_kv(1, 2, 3, 8);
        let result = cache.update_and_fetch(&keys2, &values2).unwrap();

        assert_eq!(cache.get_offset(), 7);
        assert_shape(&result[0], &[1, 2, 7, 8]);
    }

    #[test]
    fn test_rotation_exceeds_max_size() {
        let mut cache = RotatingKVCache::new(4, Some(0));
        let (keys, values) = random_kv(1, 2, 10, 8);

        let result = cache.update_and_fetch(&keys, &values).unwrap();

        assert_eq!(cache.get_offset(), 10);
        let seq_len = result[0].shape().unwrap()[2];
        assert!(seq_len >= 4);
        assert!(seq_len <= 13);
    }

    #[test]
    fn test_keep_tokens_during_rotation() {
        let mut cache = RotatingKVCache::new(8, Some(2));
        let (keys, values) = random_kv(1, 2, 12, 8);

        let result = cache.update_and_fetch(&keys, &values).unwrap();

        assert_eq!(cache.get_offset(), 12);
        let seq_len = result[0].shape().unwrap()[2];
        assert!(seq_len >= 8);
    }

    #[test]
    fn test_single_token_updates() {
        let mut cache = RotatingKVCache::new(10, Some(0));

        let (keys1, values1) = random_kv(1, 4, 5, 16);
        cache.update_and_fetch(&keys1, &values1).unwrap();
        assert_eq!(cache.get_offset(), 5);

        for i in 0..3 {
            let (key_token, value_token) = random_kv(1, 4, 1, 16);
            let result = cache.update_and_fetch(&key_token, &value_token).unwrap();

            assert_eq!(cache.get_offset(), 5 + i + 1);
            assert_shape(&result[0], &[1, 4, 5 + i as i64 + 1, 16]);
        }
    }

    #[test]
    fn test_single_token_rotation() {
        let mut cache = RotatingKVCache::new(6, Some(0));

        let (keys1, values1) = random_kv(1, 2, 6, 8);
        cache.update_and_fetch(&keys1, &values1).unwrap();

        for i in 0..4 {
            let (key_token, value_token) = random_kv(1, 2, 1, 8);
            let result = cache.update_and_fetch(&key_token, &value_token).unwrap();

            assert_eq!(cache.get_offset(), 6 + i + 1);
            assert_shape(&result[0], &[1, 2, 6, 8]);
        }
    }

    #[test]
    fn test_reset() {
        let mut cache = RotatingKVCache::new(8, Some(2));

        let (keys, values) = random_kv(2, 8, 6, 32);
        cache.update_and_fetch(&keys, &values).unwrap();
        assert_eq!(cache.get_offset(), 6);

        cache.reset();
        assert_eq!(cache.get_offset(), 0);
        assert_eq!(cache.get_idx(), 0);

        let (keys2, values2) = random_kv(2, 8, 5, 32);
        let result = cache.update_and_fetch(&keys2, &values2).unwrap();

        assert_eq!(cache.get_offset(), 5);
        assert_shape(&result[0], &[2, 8, 5, 32]);
    }

    #[test]
    fn test_max_size_one() {
        let mut cache = RotatingKVCache::new(1, Some(0));

        for i in 0..3 {
            let (keys, values) = random_kv(1, 2, 1, 8);
            let result = cache.update_and_fetch(&keys, &values).unwrap();

            assert_eq!(cache.get_offset(), i + 1);
            assert_shape(&result[0], &[1, 2, 1, 8]);
        }
    }

    #[test]
    fn test_batch_size_greater_than_one() {
        let mut cache = RotatingKVCache::new(10, Some(2));

        let (keys, values) = random_kv(4, 2, 8, 16);
        let result = cache.update_and_fetch(&keys, &values).unwrap();

        assert_eq!(cache.get_offset(), 8);
        assert_shape(&result[0], &[4, 2, 8, 16]);
    }

    #[test]
    fn test_data_integrity() {
        let mut cache = RotatingKVCache::new(10, Some(0));

        let keys1 = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]).unwrap();
        let values1 = MxArray::from_float32(&[10.0, 20.0, 30.0, 40.0], &[1, 1, 4, 1]).unwrap();

        let result1 = cache.update_and_fetch(&keys1, &values1).unwrap();
        result1[0].eval();
        result1[1].eval();
        let keys1_data = result1[0].to_float32().unwrap().to_vec();
        let values1_data = result1[1].to_float32().unwrap().to_vec();

        assert_eq!(keys1_data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(values1_data, vec![10.0, 20.0, 30.0, 40.0]);

        let keys2 = MxArray::from_float32(&[5.0, 6.0], &[1, 1, 2, 1]).unwrap();
        let values2 = MxArray::from_float32(&[50.0, 60.0], &[1, 1, 2, 1]).unwrap();

        let result2 = cache.update_and_fetch(&keys2, &values2).unwrap();
        result2[0].eval();
        result2[1].eval();
        let keys2_data = result2[0].to_float32().unwrap().to_vec();
        let values2_data = result2[1].to_float32().unwrap().to_vec();

        assert_eq!(keys2_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(values2_data, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn test_long_generation_with_rotation() {
        let mut cache = RotatingKVCache::new(8, Some(2));

        let (keys1, values1) = random_kv(1, 2, 5, 8);
        cache.update_and_fetch(&keys1, &values1).unwrap();

        for i in 0..20 {
            let (key_token, value_token) = random_kv(1, 2, 1, 8);
            let result = cache.update_and_fetch(&key_token, &value_token).unwrap();

            assert_eq!(cache.get_offset(), 5 + i + 1);
            if 5 + i + 1 >= 8 {
                assert_shape(&result[0], &[1, 2, 8, 8]);
            }
        }

        assert_eq!(cache.get_offset(), 25);
    }

    #[test]
    fn test_multi_chunk_stays_within_window() {
        let mut cache = RotatingKVCache::new(8, None);
        let k1 = MxArray::ones(&[1, 1, 6, 4], None).unwrap();
        let v1 = k1.clone();
        cache.update_and_fetch(&k1, &v1).unwrap();

        let k2 = MxArray::ones(&[1, 1, 6, 4], None).unwrap();
        let v2 = k2.clone();
        let r2 = cache.update_and_fetch(&k2, &v2).unwrap();
        assert_eq!(
            r2[0].shape_at(2).unwrap(),
            12,
            "prefill attention sees previous window plus current chunk"
        );

        // Single token after should yield window-bounded cache
        let k3 = MxArray::ones(&[1, 1, 1, 4], None).unwrap();
        let v3 = k3.clone();
        let r = cache.update_and_fetch(&k3, &v3).unwrap();
        assert!(
            r[0].shape_at(2).unwrap() <= 8,
            "cache must not exceed window"
        );
        let (stored_k, _) = cache.fetch_current_kv().unwrap();
        assert!(
            stored_k.shape_at(2).unwrap() <= 8,
            "stored rotating cache must remain window bounded"
        );
    }

    #[test]
    fn test_multi_token_append_larger_than_window_stores_only_tail() {
        let mut cache = RotatingKVCache::new(4, None);

        let prefix_keys = MxArray::from_float32(&[1.0], &[1, 1, 1, 1]).unwrap();
        let prefix_values = MxArray::from_float32(&[10.0], &[1, 1, 1, 1]).unwrap();
        cache
            .update_and_fetch(&prefix_keys, &prefix_values)
            .unwrap();

        let chunk1_keys =
            MxArray::from_float32(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[1, 1, 6, 1]).unwrap();
        let chunk1_values =
            MxArray::from_float32(&[20.0, 30.0, 40.0, 50.0, 60.0, 70.0], &[1, 1, 6, 1]).unwrap();
        let chunk1_attention = cache
            .update_and_fetch(&chunk1_keys, &chunk1_values)
            .unwrap();
        assert_shape(&chunk1_attention[0], &[1, 1, 7, 1]);

        let (stored_after_chunk1, _) = cache.fetch_current_kv().unwrap();
        assert_shape(&stored_after_chunk1, &[1, 1, 4, 1]);
        assert_float_data(&stored_after_chunk1, &[4.0, 5.0, 6.0, 7.0]);

        let chunk2_keys =
            MxArray::from_float32(&[8.0, 9.0, 10.0, 11.0, 12.0, 13.0], &[1, 1, 6, 1]).unwrap();
        let chunk2_values =
            MxArray::from_float32(&[80.0, 90.0, 100.0, 110.0, 120.0, 130.0], &[1, 1, 6, 1])
                .unwrap();
        let chunk2_attention = cache
            .update_and_fetch(&chunk2_keys, &chunk2_values)
            .unwrap();

        assert_shape(&chunk2_attention[0], &[1, 1, 10, 1]);
        assert_float_data(
            &chunk2_attention[0],
            &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        );

        let (stored_after_chunk2, _) = cache.fetch_current_kv().unwrap();
        assert_shape(&stored_after_chunk2, &[1, 1, 4, 1]);
        assert_float_data(&stored_after_chunk2, &[10.0, 11.0, 12.0, 13.0]);
        assert_eq!(cache.get_offset(), 13);
        assert_eq!(cache.get_cached_token_count().unwrap(), 4);
    }

    #[test]
    fn test_snapshot_restore_after_wrap_preserves_temporal_tail() {
        let mut source = RotatingKVCache::new(4, None);
        let keys1 = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]).unwrap();
        let values1 = MxArray::from_float32(&[10.0, 20.0, 30.0, 40.0], &[1, 1, 4, 1]).unwrap();
        source.update_and_fetch(&keys1, &values1).unwrap();

        let keys2 = MxArray::from_float32(&[5.0], &[1, 1, 1, 1]).unwrap();
        let values2 = MxArray::from_float32(&[50.0], &[1, 1, 1, 1]).unwrap();
        source.update_and_fetch(&keys2, &values2).unwrap();

        let keys3 = MxArray::from_float32(&[6.0], &[1, 1, 1, 1]).unwrap();
        let values3 = MxArray::from_float32(&[60.0], &[1, 1, 1, 1]).unwrap();
        source.update_and_fetch(&keys3, &values3).unwrap();

        let snapshot = source.snapshot().unwrap().unwrap();
        assert_eq!(snapshot.offset, 6);
        assert_eq!(snapshot.max_size, 4);
        assert_eq!(snapshot.cached_tokens, 4);
        assert_float_data(&snapshot.keys, &[3.0, 4.0, 5.0, 6.0]);
        assert_float_data(&snapshot.values, &[30.0, 40.0, 50.0, 60.0]);

        let mut restored = RotatingKVCache::new(4, None);
        restored.restore_snapshot(&snapshot).unwrap();

        let state = restored.state().unwrap();
        assert!(state.initialized);
        assert_eq!(state.offset, 6);
        assert_eq!(state.cached_tokens, 4);
        assert_eq!(state.window_size, 4);

        let (restored_keys, restored_values) = restored.fetch_current_kv().unwrap();
        assert_float_data(&restored_keys, &[3.0, 4.0, 5.0, 6.0]);
        assert_float_data(&restored_values, &[30.0, 40.0, 50.0, 60.0]);

        let append_keys = MxArray::from_float32(&[7.0, 8.0], &[1, 1, 2, 1]).unwrap();
        let append_values = MxArray::from_float32(&[70.0, 80.0], &[1, 1, 2, 1]).unwrap();

        let source_attention = source
            .update_and_fetch(&append_keys, &append_values)
            .unwrap();
        let restored_attention = restored
            .update_and_fetch(&append_keys, &append_values)
            .unwrap();

        assert_float_data(&source_attention[0], &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_float_data(&source_attention[1], &[30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
        assert_float_data(&restored_attention[0], &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_float_data(
            &restored_attention[1],
            &[30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        );

        let (source_tail, _) = source.fetch_current_kv().unwrap();
        let (restored_tail, _) = restored.fetch_current_kv().unwrap();
        assert_float_data(&source_tail, &[5.0, 6.0, 7.0, 8.0]);
        assert_float_data(&restored_tail, &[5.0, 6.0, 7.0, 8.0]);
    }
}
