use crate::array::MxArray;
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Key-Value cache for efficient transformer inference.
///
/// Uses pre-allocated buffers with in-place assignment to avoid O(N²) concatenation overhead.
/// Allocates memory in 256-token chunks (matching MLX-LM's step size).
#[napi(js_name = "KVCache")]
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

#[napi]
impl KVCache {
    /// Creates a new empty KV cache.
    #[napi(constructor)]
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
    #[napi]
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
    #[napi]
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
    }

    /// Returns the current offset (number of cached tokens).
    #[napi]
    pub fn get_offset(&self) -> i32 {
        self.offset
    }
}
