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
            let zero_scalar = MxArray::full(&[1], Either::A(0.0), None)?;
            to_cat.push(v.add(&zero_scalar)?);
        }

        if let Some(append_arr) = append {
            let zero_scalar = MxArray::full(&[1], Either::A(0.0), None)?;
            to_cat.push(append_arr.add(&zero_scalar)?);
        }

        let refs: Vec<&MxArray> = to_cat.iter().collect();
        MxArray::concatenate_many(refs, Some(2))
    }

    /// Helper: Rearrange cache into temporal order
    fn temporal_order(&self, v: &MxArray) -> Result<MxArray> {
        let v_shape = v.shape()?;
        let cache_len = v_shape[2] as i32;

        if self.idx == cache_len {
            let zero_scalar = MxArray::full(&[1], Either::A(0.0), None)?;
            Ok(v.add(&zero_scalar)?)
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
            let zero_scalar = MxArray::full(&[1], Either::A(0.0), None)?;
            self.keys = Some(keys.add(&zero_scalar)?);
            self.values = Some(values.add(&zero_scalar)?);
            self.offset = seq_len;
            self.idx = seq_len;
            return Ok(vec![keys.add(&zero_scalar)?, values.add(&zero_scalar)?]);
        }

        let ordered_keys = self.temporal_order(self.keys.as_ref().unwrap())?;
        let ordered_values = self.temporal_order(self.values.as_ref().unwrap())?;

        let current_len = ordered_keys.shape_at(2)? as i32;
        self.idx = current_len;

        let trim_size = current_len - self.max_size + 1;

        let new_keys = self.trim(trim_size, &ordered_keys, Some(keys))?;
        let new_values = self.trim(trim_size, &ordered_values, Some(values))?;

        let zero_scalar = MxArray::full(&[1], Either::A(0.0), None)?;
        self.keys = Some(new_keys.add(&zero_scalar)?);
        self.values = Some(new_values.add(&zero_scalar)?);
        self.offset += seq_len;
        self.idx = new_keys.shape_at(2)? as i32;

        Ok(vec![new_keys, new_values])
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

            let new_k = MxArray::zeros(&k_shape, None)?;
            let new_v = MxArray::zeros(&v_shape, None)?;

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

        let zero_scalar = MxArray::full(&[1], Either::A(0.0), None)?;
        Ok(vec![
            self_keys.add(&zero_scalar)?,
            self_values.add(&zero_scalar)?,
        ])
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
}
