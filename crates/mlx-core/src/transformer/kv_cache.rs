use crate::array::kv_int8::Int8KvRows;
use crate::array::{DType, MxArray};
use napi::bindgen_prelude::*;

/// Element format of a flat [`KVCache`]'s rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KvFormat {
    /// Rows stored as produced (BF16 for the Qwen3.5 family).
    #[default]
    Bf16,
    /// Per-(token, head) symmetric int8 rows with one fp32 scale each
    /// (Splash's target KV format); see `crate::array::kv_int8`.
    Int8,
}

impl KvFormat {
    /// `"int8"` / `"bf16"` (case-insensitive); `None` is BF16.
    pub fn parse(value: Option<&str>) -> std::result::Result<Self, String> {
        match value.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") | Some("bf16") => Ok(Self::Bf16),
            Some("int8") => Ok(Self::Int8),
            Some(other) => Err(format!(
                "unknown kv_format {other:?}: expected \"int8\" or \"bf16\""
            )),
        }
    }

    /// The format a cache takes when none is requested: int8 (Splash's
    /// default) where the int8 segmented SDPA kernels serve the geometry —
    /// the Metal device and a 256-wide head — and BF16 everywhere else, so a
    /// geometry the kernels do not cover never lands on the dequantizing
    /// fallback by default.
    pub fn default_for_geometry(head_dim: i64) -> Self {
        let metal = unsafe { mlx_sys::mlx_metal_is_available() }
            && unsafe { mlx_sys::mlx_default_device() } == 1;
        if metal && head_dim == 256 {
            Self::Int8
        } else {
            Self::Bf16
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Int8 => "int8",
        }
    }

    /// Bytes one (token, head) row of `head_dim` costs in one K or V tensor,
    /// including the int8 row's fp32 scale.
    pub fn row_bytes(self, head_dim: i64, activation_bytes: u64) -> u64 {
        match self {
            Self::Bf16 => head_dim.max(0) as u64 * activation_bytes,
            Self::Int8 => head_dim.max(0) as u64 + 4,
        }
    }
}

/// Key-Value cache for efficient transformer inference.
///
/// Uses pre-allocated buffers with in-place assignment to avoid O(N²) concatenation overhead.
/// Allocates memory in 256-token chunks (matching MLX-LM's step size).
///
/// In [`KvFormat::Int8`] mode `keys` / `values` are int8 `[B, H, capacity,
/// D]` and `key_scales` / `value_scales` float32 `[B, H, capacity]`; rows are
/// quantized on write ([`Self::update_and_fetch_int8`]) or arrive quantized
/// ([`Self::append_quantized`]) and are read through [`Self::int8_view`].
/// `reserve`, `trim`, `capacity` and the offset bookkeeping are format
/// independent.
pub struct KVCache {
    keys: Option<MxArray>,
    values: Option<MxArray>,
    /// Int8 mode only: fp32 scales `[B, H, capacity]` of `keys` / `values`.
    key_scales: Option<MxArray>,
    value_scales: Option<MxArray>,
    offset: i32,
    step: i32,
    /// Rows the first allocation must hold (set by [`Self::reserve`] while
    /// the cache is still empty); 0 when nothing is reserved.
    pending_rows: i64,
    format: KvFormat,
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KVCache {
    /// Creates a new empty KV cache.
    pub fn new() -> Self {
        Self::with_format(KvFormat::Bf16)
    }

    /// Creates a new empty KV cache holding rows in `format`.
    pub fn with_format(format: KvFormat) -> Self {
        Self {
            keys: None,
            values: None,
            key_scales: None,
            value_scales: None,
            offset: 0,
            step: 256, // Pre-allocate 256 tokens at a time (matching MLX-LM)
            pending_rows: 0,
            format,
        }
    }

    /// Element format of the rows this cache holds.
    pub fn format(&self) -> KvFormat {
        self.format
    }

    /// Grow the buffers so rows `[prev, prev + seq_len)` exist; `k_dtype` /
    /// `v_dtype` and the head dims describe the row buffers (int8 for the
    /// int8 format). Allocates scale buffers in int8 mode.
    fn ensure_rows(
        &mut self,
        batch_size: i64,
        n_kv_heads: i64,
        seq_len: i32,
        k_head_dim: i64,
        v_head_dim: i64,
        k_dtype: DType,
        v_dtype: DType,
    ) -> Result<()> {
        let prev = self.offset;
        let needs_grow = match &self.keys {
            Some(cached_keys) => (prev + seq_len) > cached_keys.shape_at(2)? as i32,
            None => true,
        };
        if !needs_grow {
            return Ok(());
        }
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
        let new_k = MxArray::zeros(&k_shape, Some(k_dtype))?;
        let new_v = MxArray::zeros(&v_shape, Some(v_dtype))?;
        let new_scales = match self.format {
            KvFormat::Bf16 => None,
            KvFormat::Int8 => {
                let shape = [batch_size, n_kv_heads, new_rows];
                Some((
                    MxArray::zeros(&shape, Some(DType::Float32))?,
                    MxArray::zeros(&shape, Some(DType::Float32))?,
                ))
            }
        };

        // Align to step boundary if needed; only concatenate when growing
        // the buffer (rare!)
        let keep_rows = if prev % self.step != 0 {
            Some(prev as i64)
        } else {
            None
        };
        self.append_rows(keep_rows, new_k, new_v, new_scales)
    }

    /// Updates the cache with new keys and values, and returns all cached keys/values.
    ///
    /// BF16 format only: an int8 cache quantizes through
    /// [`Self::update_and_fetch_int8`] (or [`Self::append_quantized`]) and is
    /// read through [`Self::int8_view`].
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
        if self.format == KvFormat::Int8 {
            return Err(Error::from_reason(
                "KVCache::update_and_fetch on an int8 cache: use update_and_fetch_int8 / \
                 append_quantized",
            ));
        }
        // Extract dimensions without copying entire shape vectors
        let batch_size = keys.shape_at(0)?;
        let n_kv_heads = keys.shape_at(1)?;
        let seq_len = keys.shape_at(2)? as i32;
        let k_head_dim = keys.shape_at(3)?;
        let v_head_dim = values.shape_at(3)?;

        let prev = self.offset;
        self.ensure_rows(
            batch_size,
            n_kv_heads,
            seq_len,
            k_head_dim,
            v_head_dim,
            keys.dtype()?,
            values.dtype()?,
        )?;

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

    /// Int8 format: quantize the BF16 `keys` / `values` block
    /// `[B, H, T, D]` per row and append it; returns the whole cache
    /// `[0:offset]` as int8 views plus scales.
    pub fn update_and_fetch_int8(
        &mut self,
        keys: &MxArray,
        values: &MxArray,
    ) -> Result<Int8KvRows> {
        let rows = Int8KvRows::quantize(keys, values)?;
        self.append_quantized(&rows)
    }

    /// Int8 format: append rows that are already quantized (the compiled
    /// verify tape emits them); returns the whole cache `[0:offset]`.
    pub fn append_quantized(&mut self, rows: &Int8KvRows) -> Result<Int8KvRows> {
        if self.format != KvFormat::Int8 {
            return Err(Error::from_reason(
                "KVCache::append_quantized on a BF16 cache: use update_and_fetch",
            ));
        }
        if rows.keys.dtype()? != DType::Int8
            || rows.values.dtype()? != DType::Int8
            || rows.key_scales.dtype()? != DType::Float32
            || rows.value_scales.dtype()? != DType::Float32
        {
            return Err(Error::from_reason(
                "KVCache::append_quantized expects int8 rows with float32 scales",
            ));
        }
        let batch_size = rows.keys.shape_at(0)?;
        let n_kv_heads = rows.keys.shape_at(1)?;
        let seq_len = rows.keys.shape_at(2)? as i32;
        let k_head_dim = rows.keys.shape_at(3)?;
        let v_head_dim = rows.values.shape_at(3)?;
        let prev = self.offset;
        self.ensure_rows(
            batch_size,
            n_kv_heads,
            seq_len,
            k_head_dim,
            v_head_dim,
            DType::Int8,
            DType::Int8,
        )?;
        self.offset += seq_len;
        let (start, end) = (prev as i64, self.offset as i64);
        for (buffer, update) in [
            (&mut self.keys, &rows.keys),
            (&mut self.values, &rows.values),
            (&mut self.key_scales, &rows.key_scales),
            (&mut self.value_scales, &rows.value_scales),
        ] {
            let Some(buffer) = buffer.as_mut() else {
                return Err(Error::from_reason(
                    "KV cache int8 buffers missing after buffer update",
                ));
            };
            buffer.slice_assign_axis_inplace(2, start, end, update)?;
        }
        self.int8_view()
            .ok_or_else(|| Error::from_reason("KV cache int8 buffers missing after append"))
    }

    /// Int8 format: views of rows `[0:offset]` (keys, values, scales), or
    /// `None` before the first allocation or on a BF16 cache.
    pub fn int8_view(&self) -> Option<Int8KvRows> {
        if self.format != KvFormat::Int8 {
            return None;
        }
        let end = self.offset as i64;
        Some(Int8KvRows {
            keys: self.keys.as_ref()?.slice_axis(2, 0, end).ok()?,
            values: self.values.as_ref()?.slice_axis(2, 0, end).ok()?,
            key_scales: self.key_scales.as_ref()?.slice_axis(2, 0, end).ok()?,
            value_scales: self.value_scales.as_ref()?.slice_axis(2, 0, end).ok()?,
        })
    }

    /// Resets the cache, clearing all stored keys and values.
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.key_scales = None;
        self.value_scales = None;
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
        let new_scales = match self.format {
            KvFormat::Bf16 => None,
            KvFormat::Int8 => {
                let shape = [keys.shape_at(0)?, keys.shape_at(1)?, extra];
                Some((
                    MxArray::zeros(&shape, Some(DType::Float32))?,
                    MxArray::zeros(&shape, Some(DType::Float32))?,
                ))
            }
        };
        self.append_rows(Some(self.offset as i64), new_k, new_v, new_scales)
    }

    /// Rows the key buffer can hold without growing (0 before the first
    /// allocation).
    pub(crate) fn capacity(&self) -> Result<i64> {
        self.keys.as_ref().map_or(Ok(0), |keys| keys.shape_at(2))
    }

    /// Replace the buffers with `buffer[0:keep_rows]` (the whole buffer when
    /// `None`) followed by `new_k` / `new_v` (and, in int8 mode, the matching
    /// scale rows), or with the new buffers alone before the first allocation
    /// or when no row is kept.
    fn append_rows(
        &mut self,
        keep_rows: Option<i64>,
        new_k: MxArray,
        new_v: MxArray,
        new_scales: Option<(MxArray, MxArray)>,
    ) -> Result<()> {
        if (self.format == KvFormat::Int8) != new_scales.is_some() {
            return Err(Error::from_reason(
                "KV cache append_rows: scale buffers must accompany int8 rows only",
            ));
        }
        let (Some(cached_keys), Some(cached_values)) = (&self.keys, &self.values) else {
            if self.keys.is_some() {
                return Err(Error::from_reason(
                    "KV cache values missing while keys are present",
                ));
            }
            self.keys = Some(new_k);
            self.values = Some(new_v);
            if let Some((ks, vs)) = new_scales {
                self.key_scales = Some(ks);
                self.value_scales = Some(vs);
            }
            return Ok(());
        };
        if keep_rows == Some(0) {
            self.keys = Some(new_k);
            self.values = Some(new_v);
            if let Some((ks, vs)) = new_scales {
                self.key_scales = Some(ks);
                self.value_scales = Some(vs);
            }
            return Ok(());
        }
        let keep = |cached: &MxArray| -> Result<MxArray> {
            match keep_rows {
                Some(rows) => cached.slice_axis(2, 0, rows),
                None => Ok(cached.clone()),
            }
        };
        let (kept_keys, kept_values) = (keep(cached_keys)?, keep(cached_values)?);
        let new_keys = MxArray::concatenate(&kept_keys, &new_k, 2)?;
        let new_values = MxArray::concatenate(&kept_values, &new_v, 2)?;
        if let Some((new_ks, new_vs)) = new_scales {
            let (Some(cached_ks), Some(cached_vs)) = (&self.key_scales, &self.value_scales) else {
                return Err(Error::from_reason(
                    "KV cache int8 scale buffers missing while rows are present",
                ));
            };
            let (kept_ks, kept_vs) = (keep(cached_ks)?, keep(cached_vs)?);
            self.key_scales = Some(MxArray::concatenate(&kept_ks, &new_ks, 2)?);
            self.value_scales = Some(MxArray::concatenate(&kept_vs, &new_vs, 2)?);
        }
        self.keys = Some(new_keys);
        self.values = Some(new_values);
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

    /// Int8 format: the whole key-scale buffer `[B, H, capacity]`.
    pub fn key_scales_ref(&self) -> Option<&MxArray> {
        self.key_scales.as_ref()
    }

    /// Int8 format: the whole value-scale buffer `[B, H, capacity]`.
    pub fn value_scales_ref(&self) -> Option<&MxArray> {
        self.value_scales.as_ref()
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

    /// BF16 `[1, H, n, 256]` rows for the int8 tests (the int8 format is
    /// defined at D = 256).
    fn rows256(heads: i64, n: i64, seed: f64) -> (MxArray, MxArray) {
        let k =
            MxArray::random_normal(&[1, heads, n, 256], seed, 1.0, Some(DType::BFloat16)).unwrap();
        let v =
            MxArray::random_normal(&[1, heads, n, 256], -seed, 1.0, Some(DType::BFloat16)).unwrap();
        (k, v)
    }

    fn metal() -> bool {
        unsafe { mlx_sys::mlx_metal_is_available() }
    }

    /// Quantize -> dequantize keeps every element within half a quantization
    /// step of its row (`max|x| / 127 / 2`, plus the BF16 rounding of the
    /// dequantized value), and every row's scale is `max|x| / 127`.
    #[test]
    fn int8_round_trip_error_is_bounded_by_the_row_step() {
        if !metal() {
            return;
        }
        let (k, v) = rows256(4, 40, 3.0);
        let quantized = Int8KvRows::quantize(&k, &v).unwrap();
        assert_eq!(quantized.keys.dtype().unwrap(), DType::Int8);
        assert_eq!(quantized.key_scales.dtype().unwrap(), DType::Float32);
        assert_eq!(
            quantized.key_scales.shape().unwrap().to_vec(),
            vec![1, 4, 40]
        );
        let (dk, dv) = quantized.dequantize().unwrap();
        for (orig, deq, scales) in [
            (&k, &dk, &quantized.key_scales),
            (&v, &dv, &quantized.value_scales),
        ] {
            let x = orig.to_float32().unwrap();
            let y = deq.to_float32().unwrap();
            let s = scales.to_float32().unwrap();
            let mut worst_ratio = 0f32;
            for row in 0..(4 * 40) {
                let xs = &x[row * 256..(row + 1) * 256];
                let ys = &y[row * 256..(row + 1) * 256];
                let max = xs.iter().fold(0f32, |m, a| m.max(a.abs()));
                let scale = s[row];
                assert!(
                    (scale - max / 127.0).abs() <= max / 127.0 * 1e-6 + 1e-12,
                    "row {row}: scale {scale} != max/127 {}",
                    max / 127.0
                );
                // Half a step plus the BF16 rounding of the dequantized value
                // (|y| <= max, so <= max * 2^-8).
                let bound = scale / 2.0 + max / 256.0;
                for (a, b) in xs.iter().zip(ys) {
                    let diff = (a - b).abs();
                    assert!(
                        diff <= bound + 1e-6,
                        "row {row}: |{a} - {b}| = {diff} > {bound}"
                    );
                    worst_ratio = worst_ratio.max(diff / bound.max(1e-12));
                }
            }
            assert!(worst_ratio > 0.0, "the data must exercise the rounding");
        }
    }

    /// The Metal quantizer and the MLX-op reference produce identical int8
    /// rows and scales.
    #[test]
    fn int8_quantizer_matches_its_mlx_op_reference() {
        if !metal() {
            return;
        }
        let (k, _) = rows256(4, 37, 11.0);
        // A row of zeros quantizes to zeros with scale 0.
        let zeros = MxArray::zeros(&[1, 4, 1, 256], Some(DType::BFloat16)).unwrap();
        let k = MxArray::concatenate(&k, &zeros, 2).unwrap();
        let (q, s) = crate::array::kv_int8::quantize_kv_rows(&k).unwrap();
        let (rq, rs) = crate::array::kv_int8::quantize_kv_rows_reference(&k).unwrap();
        assert_eq!(q.to_int8().unwrap(), rq.to_int8().unwrap());
        assert_eq!(bits(&s), bits(&rs));
        let s = s.to_float32().unwrap();
        for head in 0..4 {
            assert_eq!(s[head * 38 + 37], 0.0, "zero row scale");
        }
        let q = q.to_int8().unwrap();
        assert!(q.iter().all(|&x| (-127..=127).contains(&x)));
        assert!(
            q.iter().any(|&x| x == 127 || x == -127),
            "every row hits its max"
        );
    }

    /// An int8 cache grows, reserves, trims and appends like the BF16 one;
    /// its views carry the scales and its rows survive a reserve copy
    /// bit-for-bit.
    #[test]
    fn int8_cache_reserve_trim_append_round_trip() {
        if !metal() {
            return;
        }
        let mut cache = KVCache::with_format(KvFormat::Int8);
        assert_eq!(cache.format(), KvFormat::Int8);
        assert!(cache.int8_view().is_none(), "nothing allocated yet");
        let (k, v) = rows256(4, 300, 1.0);
        assert!(
            cache.update_and_fetch(&k, &v).is_err(),
            "the BF16 entry point refuses an int8 cache"
        );
        let view = cache.update_and_fetch_int8(&k, &v).unwrap();
        assert_eq!(cache.get_offset(), 300);
        assert_eq!(cache.capacity().unwrap(), 512);
        assert_eq!(view.keys.shape().unwrap().to_vec(), vec![1, 4, 300, 256]);
        assert_eq!(view.key_scales.shape().unwrap().to_vec(), vec![1, 4, 300]);
        assert_eq!(cache.key_scales_ref().unwrap().shape().unwrap()[2], 512);
        let expected = Int8KvRows::quantize(&k, &v).unwrap();
        assert_eq!(bits(&view.keys), bits(&expected.keys));
        assert_eq!(bits(&view.value_scales), bits(&expected.value_scales));

        // Reserve copies the prefix once into a larger buffer.
        cache.reserve(2000).unwrap();
        assert_eq!(cache.capacity().unwrap(), 2048);
        let after = cache.int8_view().unwrap();
        assert_eq!(bits(&after.keys), bits(&expected.keys));
        assert_eq!(bits(&after.key_scales), bits(&expected.key_scales));
        assert_eq!(bits(&after.values), bits(&expected.values));

        // Verify-style cycles: append 8, keep some, repeat across the 512
        // boundary without growing; the kept rows equal a fresh quantization
        // of the kept BF16 rows.
        let mut kept_k = k.clone();
        let mut kept_v = v.clone();
        for (cycle, keep) in [8, 1, 5, 8, 3, 8, 2, 7].iter().cycle().take(60).enumerate() {
            let (nk, nv) = rows256(4, 8, cycle as f64 * 0.1 + 0.05);
            let base = cache.get_offset();
            cache.update_and_fetch_int8(&nk, &nv).unwrap();
            assert_eq!(cache.get_offset(), base + 8);
            cache.trim(base + keep);
            assert_eq!(cache.capacity().unwrap(), 2048, "cycle={cycle}");
            kept_k = MxArray::concatenate(&kept_k, &nk.slice_axis(2, 0, *keep as i64).unwrap(), 2)
                .unwrap();
            kept_v = MxArray::concatenate(&kept_v, &nv.slice_axis(2, 0, *keep as i64).unwrap(), 2)
                .unwrap();
        }
        assert!(
            cache.get_offset() > 512,
            "the trace crosses the 512 boundary"
        );
        let live = cache.int8_view().unwrap();
        let fresh = Int8KvRows::quantize(&kept_k, &kept_v).unwrap();
        assert_eq!(live.len().unwrap(), kept_k.shape_at(2).unwrap());
        assert_eq!(bits(&live.keys), bits(&fresh.keys));
        assert_eq!(bits(&live.values), bits(&fresh.values));
        assert_eq!(bits(&live.key_scales), bits(&fresh.key_scales));
        assert_eq!(bits(&live.value_scales), bits(&fresh.value_scales));

        // Already-quantized rows append as they are.
        let (nk, nv) = rows256(4, 3, 99.0);
        let pre = Int8KvRows::quantize(&nk, &nv).unwrap();
        let base = cache.get_offset() as i64;
        let all = cache.append_quantized(&pre).unwrap();
        let tail = all.slice_tokens(base, base + 3).unwrap();
        assert_eq!(bits(&tail.keys), bits(&pre.keys));
        assert_eq!(bits(&tail.key_scales), bits(&pre.key_scales));
        assert!(
            KVCache::new().append_quantized(&pre).is_err(),
            "a BF16 cache refuses quantized rows"
        );

        cache.reset();
        assert_eq!(cache.get_offset(), 0);
        assert!(cache.int8_view().is_none());
        assert!(cache.key_scales_ref().is_none());
    }

    /// Growth across the step boundary without a reservation concatenates
    /// the scale buffers alongside the rows.
    #[test]
    fn int8_cache_grows_scale_buffers_with_rows() {
        if !metal() {
            return;
        }
        let mut cache = KVCache::with_format(KvFormat::Int8);
        let (k, v) = rows256(2, 250, 5.0);
        cache.update_and_fetch_int8(&k, &v).unwrap();
        assert_eq!(cache.capacity().unwrap(), 256);
        let (k2, v2) = rows256(2, 20, 6.0);
        let view = cache.update_and_fetch_int8(&k2, &v2).unwrap();
        assert_eq!(cache.capacity().unwrap(), 250 + 256);
        assert_eq!(
            cache.key_scales_ref().unwrap().shape().unwrap()[2],
            250 + 256
        );
        let expected = Int8KvRows::quantize(
            &MxArray::concatenate(&k, &k2, 2).unwrap(),
            &MxArray::concatenate(&v, &v2, 2).unwrap(),
        )
        .unwrap();
        assert_eq!(bits(&view.keys), bits(&expected.keys));
        assert_eq!(bits(&view.key_scales), bits(&expected.key_scales));
        assert_eq!(bits(&view.value_scales), bits(&expected.value_scales));
    }

    #[test]
    fn kv_format_parses_its_two_names() {
        assert_eq!(KvFormat::parse(None), Ok(KvFormat::Bf16));
        assert_eq!(KvFormat::parse(Some("")), Ok(KvFormat::Bf16));
        assert_eq!(KvFormat::parse(Some("bf16")), Ok(KvFormat::Bf16));
        assert_eq!(KvFormat::parse(Some(" INT8 ")), Ok(KvFormat::Int8));
        assert!(KvFormat::parse(Some("fp8")).is_err());
        assert_eq!(KvFormat::Bf16.row_bytes(256, 2), 512);
        assert_eq!(KvFormat::Int8.row_bytes(256, 2), 260);
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
