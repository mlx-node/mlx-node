use crate::array::MxArray;
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Rotating Key-Value cache with fixed maximum size.
///
/// Once the cache reaches `max_size`, old entries are evicted (except for `keep` tokens).
/// This is useful for long conversations where we want to limit memory usage while
/// keeping important context (e.g., system prompts).
///
/// Reference: mlx-lm/mlx_lm/models/cache.py:RotatingKVCache
#[napi(js_name = "RotatingKVCache")]
pub struct RotatingKVCache {
    keys: Option<MxArray>,
    values: Option<MxArray>,
    offset: i32,   // Total number of tokens processed
    max_size: i32, // Maximum cache size
    keep: i32,     // Number of initial tokens to always keep
    idx: i32,      // Current write position (for rotation)
}

#[napi]
impl RotatingKVCache {
    /// Creates a new rotating KV cache.
    ///
    /// # Arguments
    /// * `max_size` - Maximum number of tokens to cache
    /// * `keep` - Number of initial tokens to never evict (default: 0)
    ///
    /// # Example
    /// ```js
    /// // Keep system prompt (first 10 tokens), max cache 100 tokens
    /// const cache = new RotatingKVCache(100, 10);
    /// ```
    #[napi(constructor)]
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

    /// Helper: Trim cache by removing middle section (keep `keep` tokens, remove some, keep rest)
    fn trim(&self, trim_size: i32, v: &MxArray, append: Option<&MxArray>) -> Result<MxArray> {
        let mut to_cat: Vec<MxArray> = Vec::new();
        let v_shape = v.shape()?;

        if trim_size > 0 {
            // Slice [:, :, 0:keep, :] - always keep first `keep` tokens
            let starts = vec![0, 0, 0, 0];
            let stops = vec![v_shape[0], v_shape[1], self.keep as i64, v_shape[3]];
            let keep_start = v.slice(&starts, &stops)?;
            to_cat.push(keep_start);

            // Slice [:, :, trim_size + keep:, :] - keep tokens after the trimmed section
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

    /// Helper: Rearrange cache into temporal order (handles rotation)
    fn temporal_order(&self, v: &MxArray) -> Result<MxArray> {
        let v_shape = v.shape()?;
        let cache_len = v_shape[2] as i32;

        if self.idx == cache_len {
            // No rotation needed, cache is in order
            let zero_scalar = MxArray::full(&[1], Either::A(0.0), None)?;
            Ok(v.add(&zero_scalar)?)
        } else if self.idx < self.offset {
            // Cache has been rotated, need to reorder:
            // [:, :, 0:keep, :] + [:, :, idx:, :] + [:, :, keep:idx, :]

            // Slice [:, :, 0:keep, :]
            let starts = vec![0, 0, 0, 0];
            let stops = vec![v_shape[0], v_shape[1], self.keep as i64, v_shape[3]];
            let keep_section = v.slice(&starts, &stops)?;

            // Slice [:, :, idx:, :]
            let starts = vec![0, 0, self.idx as i64, 0];
            let stops = vec![v_shape[0], v_shape[1], v_shape[2], v_shape[3]];
            let tail = v.slice(&starts, &stops)?;

            // Slice [:, :, keep:idx, :]
            let starts = vec![0, 0, self.keep as i64, 0];
            let stops = vec![v_shape[0], v_shape[1], self.idx as i64, v_shape[3]];
            let middle = v.slice(&starts, &stops)?;

            MxArray::concatenate_many(vec![&keep_section, &tail, &middle], Some(2))
        } else {
            // Cache not yet full, slice off unused portion [:, :, 0:idx, :]
            let starts = [0, 0, 0, 0];
            let stops = [v_shape[0], v_shape[1], self.idx as i64, v_shape[3]];
            v.slice(&starts, &stops)
        }
    }

    /// Update with concatenation (used for multi-token updates)
    fn update_concat(&mut self, keys: &MxArray, values: &MxArray) -> Result<Vec<MxArray>> {
        let seq_len = keys.shape_at(2)? as i32;

        if self.keys.is_none() {
            // First update - store copies
            let zero_scalar = MxArray::full(&[1], Either::A(0.0), None)?;
            self.keys = Some(keys.add(&zero_scalar)?);
            self.values = Some(values.add(&zero_scalar)?);
            self.offset = seq_len;
            self.idx = seq_len;

            return Ok(vec![keys.add(&zero_scalar)?, values.add(&zero_scalar)?]);
        }

        // Put keys/values in temporal order to preserve context
        let ordered_keys = self.temporal_order(self.keys.as_ref().unwrap())?;
        let ordered_values = self.temporal_order(self.values.as_ref().unwrap())?;

        let current_len = ordered_keys.shape_at(2)? as i32;
        self.idx = current_len;

        // The largest size is self.max_size + S - 1 to ensure
        // every token gets at least self.max_size context
        let trim_size = current_len - self.max_size + 1;

        let new_keys = self.trim(trim_size, &ordered_keys, Some(keys))?;
        let new_values = self.trim(trim_size, &ordered_values, Some(values))?;

        // Store copies
        let zero_scalar = MxArray::full(&[1], Either::A(0.0), None)?;
        self.keys = Some(new_keys.add(&zero_scalar)?);
        self.values = Some(new_values.add(&zero_scalar)?);
        self.offset += seq_len;

        self.idx = new_keys.shape_at(2)? as i32;

        Ok(vec![new_keys, new_values])
    }

    /// Update in-place (used for single-token updates)
    fn update_in_place(&mut self, keys: &MxArray, values: &MxArray) -> Result<Vec<MxArray>> {
        // Extract dimensions without copying entire shape vectors
        let batch_size = keys.shape_at(0)? as i32;
        let n_kv_heads = keys.shape_at(1)? as i32;
        let seq_len = keys.shape_at(2)? as i32; // Should be 1 for single-token
        let k_head_dim = keys.shape_at(3)? as i32;
        let v_head_dim = values.shape_at(3)? as i32;

        let prev = self.offset;

        // May not have hit max size yet, so potentially keep growing the cache
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

        // Trim if needed
        let current_cache_size = self_keys.shape_at(2)? as i32;
        let trim_size = current_cache_size - self.max_size;
        if trim_size > 0 {
            let trimmed_keys = self.trim(trim_size, self_keys, None)?;
            let trimmed_values = self.trim(trim_size, self_values, None)?;
            self.keys = Some(trimmed_keys);
            self.values = Some(trimmed_values);
            self.idx = self.max_size;
        }

        // Rotate if we've reached max_size
        if self.idx == self.max_size {
            self.idx = self.keep;
        }

        // Assign new keys/values at current position using OPTIMIZED in-place assignment
        let start = self.idx;
        let end = self.idx + seq_len;

        // Use optimized in-place slice assignment - no concatenation needed!
        // This directly modifies the rotating buffer without creating new arrays
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

        // If the buffer is not full, slice off the end
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

    /// Updates the cache with new keys and values, and returns all cached keys/values.
    ///
    /// Uses different strategies based on sequence length:
    /// - Single token (seq_len=1): In-place update with rotation
    /// - Multiple tokens: Concatenation with temporal reordering
    ///
    /// # Arguments
    /// * `keys` - New keys to add, shape: (batch, n_kv_heads, seq_len, head_dim)
    /// * `values` - New values to add, shape: (batch, n_kv_heads, seq_len, head_dim)
    ///
    /// # Returns
    /// Array containing [cached_keys, cached_values] including the new entries
    #[napi]
    pub fn update_and_fetch(&mut self, keys: &MxArray, values: &MxArray) -> Result<Vec<MxArray>> {
        let seq_len = keys.shape_at(2)? as i32;

        if seq_len == 1 {
            self.update_in_place(keys, values)
        } else {
            self.update_concat(keys, values)
        }
    }

    /// Resets the cache, clearing all stored keys and values.
    #[napi]
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
        self.idx = 0;
    }

    /// Returns the current offset (total number of tokens processed).
    #[napi]
    pub fn get_offset(&self) -> i32 {
        self.offset
    }

    /// Returns the maximum cache size.
    #[napi]
    pub fn get_max_size(&self) -> i32 {
        self.max_size
    }

    /// Returns the number of tokens to keep (never evict).
    #[napi]
    pub fn get_keep(&self) -> i32 {
        self.keep
    }

    /// Returns the current write index (for rotation).
    #[napi]
    pub fn get_idx(&self) -> i32 {
        self.idx
    }
}
