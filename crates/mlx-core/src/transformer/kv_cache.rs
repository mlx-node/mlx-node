use crate::array::{DType, MxArray};
use napi::bindgen_prelude::*;

/// Element format of the full-attention K/V rows a paged pool stores.
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

    /// The format a cache takes when none is requested: int8 (Splash
    /// `RuntimeResourcesConfig::kvFormat`) where the int8 kernels serve the
    /// model — the Metal device, BF16 `compute` (the kernels take BF16
    /// queries; Splash is BF16-only), a 256-wide head and whole GQA groups (a
    /// paged read maps query head h to KV head h / gqa at any gqa) — and BF16
    /// everywhere else.
    pub fn default_for_geometry(
        head_dim: i64,
        q_heads: i64,
        kv_heads: i64,
        compute: DType,
    ) -> Self {
        let metal = unsafe { mlx_sys::mlx_metal_is_available() }
            && unsafe { mlx_sys::mlx_default_device() } == 1;
        let whole_groups = kv_heads > 0 && q_heads % kv_heads == 0;
        if metal && compute == DType::BFloat16 && head_dim == 256 && whole_groups {
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

/// One layer's block of new rows, for [`KVCache::store_block`].
pub(crate) enum KvBlock {
    /// BF16 `[B, H, T, D]` rows (any strides).
    Bf16 { keys: MxArray, values: MxArray },
}

/// One cache's share of a fused row store: the (cache, block) handle pairs
/// and the row the block lands on.
struct StorePlan {
    dst: Vec<*mut mlx_sys::mlx_array>,
    src: Vec<*mut mlx_sys::mlx_array>,
    offset: i32,
    seq_len: i32,
}

/// Key-Value cache for efficient transformer inference.
///
/// Uses pre-allocated buffers with in-place assignment to avoid O(N²) concatenation overhead.
/// Defaults to 256-token allocation chunks. Short, bounded sequences can use a
/// smaller growth step without changing the logical cache or reset semantics.
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

    /// Choose allocation granularity for a workload with a known sequence bound.
    /// This controls spare capacity, not how many tokens attention can observe.
    pub(crate) fn with_growth_step(rows: usize) -> Result<Self> {
        let step = i32::try_from(rows)
            .ok()
            .filter(|&rows| rows > 0)
            .ok_or_else(|| {
                Error::from_reason("KV cache growth step must be within 1..=i32::MAX")
            })?;
        Ok(Self {
            step,
            ..Self::new()
        })
    }

    /// Grow the buffers so rows `[prev, prev + seq_len)` exist; `k_dtype` /
    /// `v_dtype` and the head dims describe the row buffers.
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
        let next = prev
            .checked_add(seq_len)
            .ok_or_else(|| Error::from_reason("KV cache sequence is too large"))?;
        let needs_grow = match &self.keys {
            Some(cached_keys) => i64::from(next) > cached_keys.shape_at(2)?,
            None => true,
        };
        if !needs_grow {
            return Ok(());
        }
        // Calculate how many steps we need to allocate
        let step = i64::from(self.step);
        let step_rows = (step + i64::from(seq_len) - 1) / step * step;
        let new_rows = if self.keys.is_some() {
            step_rows
        } else {
            step_rows.max(self.pending_rows)
        };
        let keep_rows = if prev % self.step != 0 {
            Some(i64::from(prev))
        } else {
            None
        };
        let retained = keep_rows.unwrap_or(self.capacity()?);
        if retained + new_rows > i64::from(i32::MAX) {
            return Err(Error::from_reason("KV cache allocation is too large"));
        }
        let k_shape = [batch_size, n_kv_heads, new_rows, k_head_dim];
        let v_shape = [batch_size, n_kv_heads, new_rows, v_head_dim];

        // Pre-allocate new buffer filled with zeros
        let new_k = MxArray::zeros(&k_shape, Some(k_dtype))?;
        let new_v = MxArray::zeros(&v_shape, Some(v_dtype))?;

        // Align to step boundary if needed; only concatenate when growing
        // the buffer (rare!)
        self.append_rows(keep_rows, new_k, new_v)?;
        self.pending_rows = 0;
        Ok(())
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
        let seq_len = i32::try_from(keys.shape_at(2)?)
            .map_err(|_| Error::from_reason("KV cache input sequence is too large"))?;
        let k_head_dim = keys.shape_at(3)?;
        let v_head_dim = values.shape_at(3)?;

        let prev = self.offset;
        let next = prev
            .checked_add(seq_len)
            .ok_or_else(|| Error::from_reason("KV cache sequence is too large"))?;
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
        self.offset = next;

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

    /// Append a block of K and V rows with ONE store dispatch for the whole
    /// layer, in place into the cache buffers. The bytes equal the
    /// slice_update path ([`Self::update_and_fetch`]), which stays the
    /// fallback when the fused store declines (no Metal, a geometry it does
    /// not serve).
    pub(crate) fn store_block(&mut self, block: &KvBlock) -> Result<()> {
        let plan = self.plan_store(block)?;
        let mut out: [*mut mlx_sys::mlx_array; 2] = [std::ptr::null_mut(); 2];
        let offsets = [plan.offset; 2];
        // SAFETY: every pointer is a live array handle for the call; `out`
        // receives owned handles (same buffers as `dst`) or stays null.
        let ok = unsafe {
            mlx_sys::mlx_kv_store_rows(
                plan.dst.len() as i32,
                plan.dst.as_ptr(),
                plan.src.as_ptr(),
                offsets.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if !ok {
            return self.store_block_unfused(block);
        }
        self.adopt_store(&out[..plan.dst.len()], plan.seq_len)
    }

    /// Grow for `block` and gather the store's (cache, rows) handle pairs:
    /// K and V. The pointers stay valid while the cache and block are
    /// untouched.
    fn plan_store(&mut self, block: &KvBlock) -> Result<StorePlan> {
        let KvBlock::Bf16 { keys, values } = block;
        let seq_len = i32::try_from(keys.shape_at(2)?)
            .map_err(|_| Error::from_reason("KV cache input sequence is too large"))?;
        let offset = self.offset;
        self.ensure_rows(
            keys.shape_at(0)?,
            keys.shape_at(1)?,
            seq_len,
            keys.shape_at(3)?,
            values.shape_at(3)?,
            keys.dtype()?,
            values.dtype()?,
        )?;
        let (Some(cached_keys), Some(cached_values)) = (&self.keys, &self.values) else {
            return Err(Error::from_reason(
                "KV cache buffers missing after buffer update",
            ));
        };
        Ok(StorePlan {
            dst: vec![cached_keys.as_raw_ptr(), cached_values.as_raw_ptr()],
            src: vec![keys.as_raw_ptr(), values.as_raw_ptr()],
            offset,
            seq_len,
        })
    }

    /// Replace the buffer handles with the store primitive's outputs (same
    /// buffers) and advance the offset.
    fn adopt_store(&mut self, out: &[*mut mlx_sys::mlx_array], seq_len: i32) -> Result<()> {
        self.keys = Some(MxArray::from_handle(out[0], "kv_store_rows:keys")?);
        self.values = Some(MxArray::from_handle(out[1], "kv_store_rows:values")?);
        self.offset += seq_len;
        Ok(())
    }

    fn store_block_unfused(&mut self, block: &KvBlock) -> Result<()> {
        let KvBlock::Bf16 { keys, values } = block;
        self.update_and_fetch(keys, values).map(|_| ())
    }

    /// Resets the cache, clearing all stored keys and values.
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
        self.pending_rows = 0;
    }

    /// Reset the logical sequence while retaining the allocated buffers.
    /// No value beyond the new offset is observable by update_and_fetch.
    pub(crate) fn reset_keep_capacity(&mut self) {
        self.offset = 0;
    }

    /// Fork a reusable prefix into an independently writable cache. Cloning an
    /// MxArray handle would alias the buffers modified by update_and_fetch.
    pub(crate) fn fork(&self) -> Result<Self> {
        Ok(Self {
            keys: self.keys.as_ref().map(MxArray::deep_copy).transpose()?,
            values: self.values.as_ref().map(MxArray::deep_copy).transpose()?,
            offset: self.offset,
            step: self.step,
            pending_rows: self.pending_rows,
        })
    }

    /// Retain a bounded suffix. Position encodings must use a separate absolute
    /// cursor: this cache's offset counts resident rows only.
    pub(crate) fn retain_recent(&mut self, rows: usize) -> Result<()> {
        if rows >= self.offset as usize {
            return Ok(());
        }
        let start = i64::from(self.offset) - rows as i64;
        for buffer in [&mut self.keys, &mut self.values] {
            *buffer = buffer
                .as_ref()
                .map(|x| x.slice_axis(2, start, i64::from(self.offset))?.deep_copy())
                .transpose()?;
        }
        self.offset = rows as i32;
        Ok(())
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
    /// `None`) followed by `new_k` / `new_v`, or with the new buffers alone
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
        let new_keys = MxArray::concatenate(&kept_keys, &new_k, 2)?;
        let new_values = MxArray::concatenate(&kept_values, &new_v, 2)?;
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
    use crate::array::kv_int8::Int8KvRows;

    /// Gate: the geometry default is int8 wherever the int8 kernels read the
    /// model: D256 with whole GQA groups at BF16 compute, the 122B's 32 / 2
    /// included; a non-divisible split, zero KV heads, D128 and f16 / f32
    /// compute stay BF16.
    /// Mutation D1: `default_for_geometry` keeps a `gqa <= 8` clause → red
    /// at "122B".
    #[test]
    fn kv_format_default_is_int8_wherever_the_pool_reads_it() {
        let metal = unsafe { mlx_sys::mlx_metal_is_available() }
            && unsafe { mlx_sys::mlx_default_device() } == 1;
        let int8 = if metal {
            KvFormat::Int8
        } else {
            KvFormat::Bf16
        };
        let bf16 = DType::BFloat16;
        assert_eq!(
            KvFormat::default_for_geometry(256, 24, 4, bf16),
            int8,
            "27B"
        );
        assert_eq!(
            KvFormat::default_for_geometry(256, 16, 2, bf16),
            int8,
            "35B"
        );
        assert_eq!(
            KvFormat::default_for_geometry(256, 32, 2, bf16),
            int8,
            "122B"
        );
        assert_eq!(
            KvFormat::default_for_geometry(256, 24, 5, bf16),
            KvFormat::Bf16
        );
        assert_eq!(
            KvFormat::default_for_geometry(256, 24, 0, bf16),
            KvFormat::Bf16
        );
        assert_eq!(
            KvFormat::default_for_geometry(128, 24, 4, bf16),
            KvFormat::Bf16
        );
        for compute in [DType::Float16, DType::Float32] {
            assert_eq!(
                KvFormat::default_for_geometry(256, 16, 2, compute),
                KvFormat::Bf16,
                "35B at {compute:?} compute"
            );
            assert_eq!(
                KvFormat::default_for_geometry(256, 24, 4, compute),
                KvFormat::Bf16,
                "27B at {compute:?} compute"
            );
        }
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
        let cache = KVCache::new();
        assert_eq!(cache.get_offset(), 0);
    }

    fn rand_bf16(shape: &[i64]) -> MxArray {
        MxArray::random_normal(shape, 0.0, 1.0, Some(DType::Float32))
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
    }

    fn bits_equal(a: &MxArray, b: &MxArray) -> bool {
        let af = a.astype(DType::Float32).unwrap().to_float32().unwrap();
        let bf = b.astype(DType::Float32).unwrap().to_float32().unwrap();
        a.shape().unwrap().as_ref() == b.shape().unwrap().as_ref()
            && af
                .as_ref()
                .iter()
                .zip(bf.as_ref().iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// The fused store must leave the cache buffers byte-identical to the
    /// slice_update path, with the transposed `values` view a verify emits,
    /// across a prefill, two verify blocks and a trim in between.
    #[test]
    #[cfg(target_os = "macos")]
    fn store_block_matches_slice_update_path() {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return;
        }
        let (b, h, d) = (1i64, 4i64, 256i64);
        let mut reference = KVCache::new();
        let mut fused = KVCache::new();
        // Verify-shaped blocks: `values` is a [B, T, H, D] -> [B, H, T, D]
        // transpose view, keys contiguous.
        let make = |t: i64| {
            let keys = rand_bf16(&[b, h, t, d]);
            let values = rand_bf16(&[b, t, h, d])
                .transpose(Some(&[0, 2, 1, 3]))
                .unwrap();
            (keys, values)
        };
        let steps: [(i64, Option<i32>); 4] =
            [(300, None), (8, Some(303)), (8, Some(306)), (8, None)];
        for (t, trim_to) in steps {
            let (keys, values) = make(t);
            reference.update_and_fetch(&keys, &values).unwrap();
            fused
                .store_block(&KvBlock::Bf16 {
                    keys: keys.clone(),
                    values: values.clone(),
                })
                .unwrap();
            if let Some(to) = trim_to {
                reference.trim(to);
                fused.trim(to);
            }
            assert_eq!(reference.get_offset(), fused.get_offset());
        }
        for (i, (r, f)) in [
            (reference.keys_ref(), fused.keys_ref()),
            (reference.values_ref(), fused.values_ref()),
        ]
        .into_iter()
        .enumerate()
        {
            assert!(bits_equal(r.unwrap(), f.unwrap()), "buffer {i} differs");
        }
    }

    /// One store dispatch per block, writing in place (the cache handle keeps
    /// its buffer).
    #[test]
    #[cfg(target_os = "macos")]
    fn store_block_is_one_in_place_dispatch() {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return;
        }
        let mut cache = KVCache::new();
        cache
            .store_block(&KvBlock::Bf16 {
                keys: rand_bf16(&[1, 2, 20, 256]),
                values: rand_bf16(&[1, 2, 20, 256]),
            })
            .unwrap();
        let before = cache.keys_ref().unwrap().clone();
        before.eval();
        let (keys, values) = (rand_bf16(&[1, 2, 8, 256]), rand_bf16(&[1, 2, 8, 256]));
        MxArray::eval_arrays(&[&keys, &values]).unwrap();
        unsafe { mlx_sys::mlx_test_kquant_counting(true) };
        cache
            .store_block(&KvBlock::Bf16 {
                keys: keys.clone(),
                values,
            })
            .unwrap();
        let after = cache.keys_ref().unwrap().clone();
        after.eval();
        let stores = unsafe { mlx_sys::mlx_test_kquant_family_count(c"kv_store_rows".as_ptr()) };
        unsafe { mlx_sys::mlx_test_kquant_counting(false) };
        assert_eq!(stores, 1, "one KvStoreRows evaluation for the block");
        // In place: the pre-store handle sees the new rows too.
        assert!(
            bits_equal(&before, &after),
            "store must write into the same buffer"
        );
        assert!(bits_equal(&after.slice_axis(2, 20, 28).unwrap(), &keys));
    }

    #[test]
    fn test_growth_step_validation() {
        assert!(KVCache::with_growth_step(0).is_err());
        assert!(KVCache::with_growth_step(i32::MAX as usize + 1).is_err());
        assert!(KVCache::with_growth_step(i32::MAX as usize).is_ok());
    }

    #[test]
    fn test_growth_overflow_is_rejected_before_allocation_or_mutation() {
        let row = MxArray::from_float32(&[1.], &[1, 1, 1, 1]).unwrap();
        let mut cache = KVCache::new();
        cache.update_and_fetch(&row, &row).unwrap();
        cache.step = i32::MAX;
        cache.offset = 256;
        // One retained buffer plus one huge step exceeds the index range.
        assert!(cache.update_and_fetch(&row, &row).is_err());
        assert_eq!(cache.offset, 256);
        assert_eq!(cache.capacity().unwrap(), 256);
        cache.offset = i32::MAX;
        assert!(cache.update_and_fetch(&row, &row).is_err());
        assert_eq!(cache.offset, i32::MAX);
        assert_eq!(cache.capacity().unwrap(), 256);
    }

    #[test]
    fn test_small_growth_steps_preserve_values_reset_reservation_and_forks() {
        for step in [1, 3, 16, 256] {
            let mut cache = KVCache::with_growth_step(step).unwrap();
            let mut expected = Vec::new();
            for count in [2, 1, 5, 17] {
                let input: Vec<f32> = (expected.len()..expected.len() + count)
                    .map(|i| i as f32)
                    .collect();
                let values = MxArray::from_float32(&input, &[1, 1, count as i64, 1]).unwrap();
                let (keys, vals) = cache.update_and_fetch(&values, &values).unwrap();
                expected.extend(input);
                assert_eq!(keys.to_float32().unwrap().as_ref(), expected.as_slice());
                assert_eq!(vals.to_float32().unwrap().as_ref(), expected.as_slice());
                assert!(cache.capacity().unwrap() < (expected.len() + step) as i64);
            }
            cache.reserve(cache.capacity().unwrap() + 40).unwrap();
            let capacity = cache.capacity().unwrap();
            assert_eq!(capacity % step as i64, 0);
            let mut fork = cache.fork().unwrap();
            assert_eq!(fork.step, step as i32);
            fork.reset_keep_capacity();
            let replacement = MxArray::from_float32(&[91., 92.], &[1, 1, 2, 1]).unwrap();
            let (keys, _) = fork.update_and_fetch(&replacement, &replacement).unwrap();
            assert_eq!(keys.to_float32().unwrap().as_ref(), &[91., 92.]);
            assert_eq!(fork.capacity().unwrap(), capacity);
            assert_eq!(cache.get_offset(), expected.len() as i32);
            assert_eq!(
                cache
                    .keys_ref()
                    .unwrap()
                    .slice_axis(2, 0, expected.len() as i64)
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .as_ref(),
                expected.as_slice()
            );
            cache.reset();
            cache.reserve(5).unwrap();
            cache.update_and_fetch(&replacement, &replacement).unwrap();
            assert_eq!(
                cache.capacity().unwrap(),
                5usize.div_ceil(step) as i64 * step as i64
            );
        }
    }

    #[test]
    fn test_reuse_capacity_and_bounded_suffix_do_not_expose_old_rows() {
        let mut cache = KVCache::new();
        let old = MxArray::from_float32(&[1., 2., 3., 4.], &[1, 1, 4, 1]).unwrap();
        cache.update_and_fetch(&old, &old).unwrap();
        let capacity = cache.capacity().unwrap();
        cache.reset_keep_capacity();
        assert_eq!(cache.capacity().unwrap(), capacity);
        let next = MxArray::from_float32(&[7., 8., 9.], &[1, 1, 3, 1]).unwrap();
        let (keys, _) = cache.update_and_fetch(&next, &next).unwrap();
        assert_eq!(keys.to_float32().unwrap().as_ref(), &[7., 8., 9.]);
        cache.retain_recent(2).unwrap();
        let last = MxArray::from_float32(&[10.], &[1, 1, 1, 1]).unwrap();
        let (keys, _) = cache.update_and_fetch(&last, &last).unwrap();
        assert_eq!(keys.to_float32().unwrap().as_ref(), &[8., 9., 10.]);
    }

    #[test]
    fn test_cache_default() {
        let cache = KVCache::default();
        assert_eq!(cache.get_offset(), 0);
    }

    #[test]
    fn test_fork_isolates_in_place_writes_and_keeps_capacity() {
        let mut prefix = KVCache::new();
        let input = MxArray::from_float32(&[1., 2., 3.], &[1, 1, 3, 1]).unwrap();
        let (keys, _) = prefix.update_and_fetch(&input, &input).unwrap();
        keys.eval();
        let mut fork = prefix.fork().unwrap();
        assert_eq!(fork.capacity().unwrap(), prefix.capacity().unwrap());
        fork.trim(1);
        let replacement = MxArray::from_float32(&[9., 8.], &[1, 1, 2, 1]).unwrap();
        let (keys, _) = fork.update_and_fetch(&replacement, &replacement).unwrap();
        assert_eq!(keys.to_float32().unwrap().as_ref(), &[1., 9., 8.]);
        let suffix = MxArray::from_float32(&[4.], &[1, 1, 1, 1]).unwrap();
        let (keys, _) = prefix.update_and_fetch(&suffix, &suffix).unwrap();
        assert_eq!(keys.to_float32().unwrap().as_ref(), &[1., 2., 3., 4.]);
        assert_eq!(fork.get_offset(), 3);
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
