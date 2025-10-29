use crate::array::MxArray;
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Batch Key-Value cache for efficient batch inference with left-padding support.
///
/// The BatchKVCache expects inputs to be left-padded. For example, given prompts:
/// - [1, 3, 5]
/// - [7]
/// - [2, 6, 8, 9]
///
/// They should be padded like:
/// - [0, 1, 3, 5]
/// - [0, 0, 0, 7]
/// - [2, 6, 8, 9]
///
/// And `left_padding = [1, 3, 0]` specifies the padding for each batch element.
///
/// Reference: mlx-lm/mlx_lm/models/cache.py:BatchKVCache (lines 662-787)
#[napi(js_name = "BatchKVCache")]
pub struct BatchKVCache {
    keys: Option<MxArray>,
    values: Option<MxArray>,
    left_padding: Int32Array, // Amount of left padding for each batch element
    offset: Vec<i32>,         // Offset for each batch element (starts negative due to padding)
    idx: i32,                 // Current write position (shared across batch)
}

#[napi]
impl BatchKVCache {
    /// Creates a new batch KV cache with left-padding information.
    ///
    /// # Arguments
    /// * `left_padding` - Array specifying left padding for each batch element
    ///
    /// # Example
    /// ```js
    /// // Three sequences with different padding amounts
    /// const cache = new BatchKVCache([1, 3, 0]);
    /// ```
    #[napi(constructor)]
    pub fn new(left_padding: Int32Array) -> Self {
        // Offset starts negative because of left padding
        let offset: Vec<i32> = left_padding.iter().map(|&l| -l).collect();

        Self {
            keys: None,
            values: None,
            left_padding,
            offset,
            idx: 0,
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
    #[napi]
    pub fn update_and_fetch(&mut self, keys: &MxArray, values: &MxArray) -> Result<Vec<MxArray>> {
        // Extract dimensions without copying entire shape vectors
        let batch_size = keys.shape_at(0)? as usize;
        let n_kv_heads = keys.shape_at(1)?;
        let seq_len = keys.shape_at(2)? as i32;
        let k_head_dim = keys.shape_at(3)?;
        let v_head_dim = values.shape_at(3)?;

        let prev = self.idx;

        // Check if we need to allocate or expand cache
        let step = 256; // Allocation step size (matches MLX-LM)
        if self.keys.is_none() || (prev + seq_len) > self.keys.as_ref().unwrap().shape_at(2)? as i32
        {
            let n_steps = (step + seq_len - 1) / step;
            let k_shape = vec![
                batch_size as i64,
                n_kv_heads,
                (n_steps * step) as i64,
                k_head_dim,
            ];
            let v_shape = vec![
                batch_size as i64,
                n_kv_heads,
                (n_steps * step) as i64,
                v_head_dim,
            ];

            let new_k = MxArray::zeros(&k_shape, None)?;
            let new_v = MxArray::zeros(&v_shape, None)?;

            if let Some(existing_keys) = &self.keys {
                // Expand existing cache
                if prev % step != 0 {
                    // Trim to prev if not on step boundary
                    let starts = vec![0, 0, 0, 0];
                    let stops = vec![batch_size as i64, n_kv_heads, prev as i64, k_head_dim];
                    self.keys = Some(existing_keys.slice(&starts, &stops)?);

                    let stops_v = vec![batch_size as i64, n_kv_heads, prev as i64, v_head_dim];
                    self.values = Some(self.values.as_ref().unwrap().slice(&starts, &stops_v)?);
                }

                // Concatenate with new space
                self.keys = Some(MxArray::concatenate(
                    self.keys.as_ref().unwrap(),
                    &new_k,
                    2,
                )?);
                self.values = Some(MxArray::concatenate(
                    self.values.as_ref().unwrap(),
                    &new_v,
                    2,
                )?);
            } else {
                // First allocation
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
        }

        // Update offset for each batch element
        for i in 0..batch_size {
            self.offset[i] += seq_len;
        }
        self.idx += seq_len;

        // Write new keys/values into cache using OPTIMIZED in-place assignment
        // This directly modifies the pre-allocated buffers without concatenation!
        if let Some(cached_keys) = self.keys.as_mut() {
            cached_keys.slice_assign_axis_inplace(2, prev as i64, self.idx as i64, keys)?;
        }
        if let Some(cached_values) = self.values.as_mut() {
            cached_values.slice_assign_axis_inplace(2, prev as i64, self.idx as i64, values)?;
        }

        // Return keys and values up to idx
        let return_starts = vec![0, 0, 0, 0];
        let return_stops_k = vec![batch_size as i64, n_kv_heads, self.idx as i64, k_head_dim];
        let return_stops_v = vec![batch_size as i64, n_kv_heads, self.idx as i64, v_head_dim];

        let return_keys = self
            .keys
            .as_ref()
            .unwrap()
            .slice(&return_starts, &return_stops_k)?;
        let return_values = self
            .values
            .as_ref()
            .unwrap()
            .slice(&return_starts, &return_stops_v)?;

        Ok(vec![return_keys, return_values])
    }

    /// Filters the cache to keep only specified batch indices.
    ///
    /// # Arguments
    /// * `batch_indices` - Indices of batch elements to keep
    ///
    /// # Example
    /// ```js
    /// // Keep only batch elements 0 and 2
    /// cache.filter([0, 2]);
    /// ```
    #[napi]
    pub fn filter(&mut self, batch_indices: &[i32]) -> Result<()> {
        // Filter offset and left_padding arrays (always do this, even if cache is empty)
        let new_offset: Vec<i32> = batch_indices
            .iter()
            .map(|&i| self.offset[i as usize])
            .collect();
        let new_left_padding: Vec<i32> = batch_indices
            .iter()
            .map(|&i| self.left_padding[i as usize])
            .collect();

        self.offset = new_offset;
        self.left_padding = new_left_padding.clone().into();

        // If cache is empty, we're done
        if self.keys.is_none() {
            return Ok(());
        }

        // Filter keys and values
        let keys = self.keys.as_ref().unwrap();
        let values = self.values.as_ref().unwrap();

        // Convert batch_indices to MxArray
        let indices_array = MxArray::from_int32(batch_indices, &[batch_indices.len() as i64])?;

        // Take along batch axis (axis 0)
        let filtered_keys = keys.take(&indices_array, 0)?;
        let filtered_values = values.take(&indices_array, 0)?;

        self.keys = Some(filtered_keys);
        self.values = Some(filtered_values);

        // Shift left to reduce padding
        let min_left_pad = *new_left_padding.iter().min().unwrap_or(&0);
        if min_left_pad > 0 {
            let keys_shape = self.keys.as_ref().unwrap().shape()?;

            // Slice off the left padding: [:, :, min_left_pad:, :]
            let starts = vec![0, 0, min_left_pad as i64, 0];
            let stops = keys_shape.as_ref().to_vec();

            self.keys = Some(self.keys.as_ref().unwrap().slice(&starts, &stops)?);
            self.values = Some(self.values.as_ref().unwrap().slice(&starts, &stops)?);
            self.idx -= min_left_pad;

            // Adjust both left_padding and offset since we shifted left
            self.left_padding = self
                .left_padding
                .iter()
                .map(|&l| l - min_left_pad)
                .collect::<Vec<_>>()
                .into();
            self.offset = self.offset.iter().map(|&o| o - min_left_pad).collect();
        }

        Ok(())
    }

    /// Extends this cache with another cache (concatenates along batch dimension).
    ///
    /// # Arguments
    /// * `other` - Another BatchKVCache to concatenate
    #[napi]
    pub fn extend(&mut self, other: &BatchKVCache) -> Result<()> {
        if self.keys.is_none() || other.keys.is_none() {
            return Err(Error::new(
                Status::InvalidArg,
                "Cannot extend empty caches".to_string(),
            ));
        }

        let max_idx = self.idx.max(other.idx);
        let self_shape = self.keys.as_ref().unwrap().shape()?;
        let other_shape = other.keys.as_ref().unwrap().shape()?;
        let max_size = (self_shape[2] as i32).max(other_shape[2] as i32);

        // Helper to pad cache
        let pad_cache = |cache_keys: &MxArray,
                         cache_values: &MxArray,
                         cache_idx: i32,
                         cache_offset: &[i32],
                         cache_left_padding: &[i32]|
         -> Result<(MxArray, MxArray, Vec<i32>, Vec<i32>)> {
            let left = max_idx - cache_idx;
            let keys_shape = cache_keys.shape()?;
            let right = max_size - keys_shape[2] as i32 - left;

            let mut k = cache_keys.clone();
            let mut v = cache_values.clone();

            if right < 0 {
                // Slice off the end
                let starts = vec![0, 0, 0, 0];
                let stops = vec![
                    keys_shape[0],
                    keys_shape[1],
                    (keys_shape[2] as i32 + right) as i64,
                    keys_shape[3],
                ];
                k = k.slice(&starts, &stops)?;

                let v_shape = cache_values.shape()?;
                let stops_v = vec![
                    v_shape[0],
                    v_shape[1],
                    (v_shape[2] as i32 + right) as i64,
                    v_shape[3],
                ];
                v = v.slice(&starts, &stops_v)?;
            }

            // Apply padding if needed
            if left != 0 || right.max(0) != 0 {
                // Pad: [(0, 0), (0, 0), (left, right), (0, 0)]
                // Flatten to [left0, right0, left1, right1, left2, right2, left3, right3]
                let pad_spec = vec![
                    0,
                    0, // axis 0
                    0,
                    0, // axis 1
                    left,
                    right.max(0), // axis 2
                    0,
                    0, // axis 3
                ];

                k = k.pad(&pad_spec, 0.0)?;
                v = v.pad(&pad_spec, 0.0)?;
            }

            let new_left_padding: Vec<i32> = cache_left_padding.iter().map(|&l| l + left).collect();
            let new_offset: Vec<i32> = cache_offset.iter().map(|&o| o + left).collect();

            Ok((k, v, new_offset, new_left_padding))
        };

        let (self_k, self_v, self_offset, self_lp) = pad_cache(
            self.keys.as_ref().unwrap(),
            self.values.as_ref().unwrap(),
            self.idx,
            &self.offset,
            &self.left_padding,
        )?;

        let (other_k, other_v, other_offset, other_lp) = pad_cache(
            other.keys.as_ref().unwrap(),
            other.values.as_ref().unwrap(),
            other.idx,
            &other.offset,
            &other.left_padding,
        )?;

        // Concatenate along batch dimension (axis 0)
        self.keys = Some(MxArray::concatenate(&self_k, &other_k, 0)?);
        self.values = Some(MxArray::concatenate(&self_v, &other_v, 0)?);

        // Concatenate metadata (replace, don't extend)
        self.offset = self_offset
            .iter()
            .chain(other_offset.iter())
            .copied()
            .collect();
        self.left_padding = self_lp
            .iter()
            .chain(other_lp.iter())
            .copied()
            .collect::<Vec<_>>()
            .into();
        self.idx = max_idx;

        Ok(())
    }

    /// Resets the cache, clearing all stored keys and values.
    #[napi]
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.idx = 0;
        // Reset offsets to initial values (negative of left padding)
        self.offset = self.left_padding.iter().map(|&l| -l).collect();
    }

    /// Returns the current index (number of cached tokens, not accounting for padding).
    #[napi]
    pub fn get_idx(&self) -> i32 {
        self.idx
    }

    /// Returns the offset array (per-batch offsets).
    #[napi]
    pub fn get_offsets(&self) -> Vec<i32> {
        self.offset.clone()
    }

    /// Returns the left padding array.
    #[napi]
    pub fn get_left_padding(&mut self) -> &mut Int32Array {
        &mut self.left_padding
    }
}
