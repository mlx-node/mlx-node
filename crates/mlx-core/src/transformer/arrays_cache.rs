use crate::array::MxArray;
use napi::bindgen_prelude::*;

/// Simple array cache for linear attention / short-conv layers (GatedDeltaNet, LFM2).
///
/// Holds N arrays (e.g. 2 slots for conv_state + recurrent_state, or 1 slot for conv state).
/// This is the equivalent of Python's ArraysCache.
pub struct ArraysCache {
    cache: Vec<Option<MxArray>>,
}

impl ArraysCache {
    /// Create a new cache with `size` slots.
    pub fn new(size: usize) -> Self {
        let cache = (0..size).map(|_| None).collect();
        Self { cache }
    }

    /// Get the array at the given index.
    pub fn get(&self, idx: usize) -> Option<&MxArray> {
        self.cache.get(idx).and_then(|v| v.as_ref())
    }

    /// Set the array at the given index.
    pub fn set(&mut self, idx: usize, value: MxArray) -> Result<()> {
        if idx >= self.cache.len() {
            return Err(Error::from_reason(format!(
                "ArraysCache::set index {idx} out of bounds for {} slots",
                self.cache.len()
            )));
        }
        self.cache[idx] = Some(value);
        Ok(())
    }

    /// Reset all cache entries.
    pub fn reset(&mut self) {
        for slot in &mut self.cache {
            *slot = None;
        }
    }

    /// Number of slots in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Stack identical request-local cache slots along their batch axis.
    /// Decode requires every slot to be materialized; a missing row is an
    /// admission/lifecycle bug and fails closed instead of borrowing a peer's
    /// state through broadcasting.
    pub(crate) fn stack_rows(rows: &[&Self]) -> Result<Self> {
        if rows.is_empty() {
            return Err(Error::from_reason(
                "ArraysCache::stack_rows requires at least one row",
            ));
        }
        let slots = rows[0].len();
        if rows.iter().any(|row| row.len() != slots) {
            return Err(Error::from_reason(
                "ArraysCache::stack_rows cache slot count mismatch",
            ));
        }
        let mut stacked = Self::new(slots);
        for slot in 0..slots {
            let arrays = rows
                .iter()
                .enumerate()
                .map(|(row, cache)| {
                    cache.get(slot).ok_or_else(|| {
                        Error::from_reason(format!(
                            "ArraysCache::stack_rows row {row} slot {slot} is uninitialized"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            stacked.set(slot, MxArray::concatenate_many(arrays, Some(0))?)?;
        }
        Ok(stacked)
    }

    /// Return one independent batch slice for `row`.
    pub(crate) fn row(&self, row: usize, batch: usize) -> Result<Self> {
        if row >= batch {
            return Err(Error::from_reason(format!(
                "ArraysCache::row index {row} out of range for batch {batch}"
            )));
        }
        let mut result = Self::new(self.len());
        for slot in 0..self.len() {
            let array = self.get(slot).ok_or_else(|| {
                Error::from_reason(format!("ArraysCache::row slot {slot} is uninitialized"))
            })?;
            if array.shape_at(0)? != batch as i64 {
                return Err(Error::from_reason(format!(
                    "ArraysCache::row slot {slot} batch {} does not match {batch}",
                    array.shape_at(0)?
                )));
            }
            result.set(slot, array.slice_axis(0, row as i64, row as i64 + 1)?)?;
        }
        Ok(result)
    }

    /// Broadcast a batch-1 cache to `batch` independent rows.
    ///
    /// Returns an error if `batch < 1` or if any populated slot does not have
    /// batch dimension 1. `None` slots remain `None`.
    pub fn broadcast_batch(&self, batch: i64) -> Result<Self> {
        let batch_i32 = i32::try_from(batch)
            .ok()
            .filter(|&b| b >= 1)
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "ArraysCache::broadcast_batch requires batch in 1..=i32::MAX, got {batch}"
                ))
            })?;
        let mut out = Self::new(self.len());
        for slot in 0..self.len() {
            if let Some(arr) = self.get(slot) {
                let ndim = arr.ndim()? as usize;
                if ndim == 0 {
                    return Err(Error::from_reason(format!(
                        "ArraysCache::broadcast_batch slot {slot} must have rank >= 1"
                    )));
                }
                let cur_batch = arr.shape_at(0)?;
                if cur_batch != 1 {
                    return Err(Error::from_reason(format!(
                        "ArraysCache::broadcast_batch requires batch=1, got {cur_batch} at slot {slot}"
                    )));
                }
                let tiled = if batch == 1 {
                    arr.deep_copy()?
                } else {
                    let mut reps = vec![1i32; ndim];
                    reps[0] = batch_i32;
                    MxArray::tile(arr, &reps)?
                };
                out.set(slot, tiled)?;
            }
        }
        Ok(out)
    }

    /// Compact the batch dimension (axis 0) in place by selecting rows at `indices`.
    ///
    /// Replaces each populated slot with `arr.take(indices, 0)?` while leaving
    /// `None` slots as `None`.
    pub fn select_batch_rows(&mut self, indices: &MxArray) -> Result<()> {
        if indices.ndim()? != 1 || indices.shape_at(0)? < 1 {
            return Err(Error::from_reason(
                "ArraysCache::select_batch_rows requires a non-empty 1D index array",
            ));
        }
        let mut updated = Vec::with_capacity(self.cache.len());
        for (slot, entry) in self.cache.iter().enumerate() {
            match entry {
                Some(arr) => {
                    if arr.ndim()? == 0 {
                        return Err(Error::from_reason(format!(
                            "ArraysCache::select_batch_rows slot {slot} must have rank >= 1"
                        )));
                    }
                    updated.push(Some(arr.take(indices, 0)?));
                }
                None => updated.push(None),
            }
        }
        self.cache = updated;
        Ok(())
    }
}

/// Advance independent recurrent rows through their singleton projection
/// graphs, then restack the activations and cache slots in input row order.
pub(crate) fn forward_rows_independently(
    x: &MxArray,
    cache: &ArraysCache,
    mut forward: impl FnMut(&MxArray, &mut ArraysCache) -> Result<MxArray>,
) -> Result<(MxArray, ArraysCache)> {
    let batch = usize::try_from(x.shape_at(0)?).map_err(|_| {
        Error::from_reason("forward_rows_independently received a negative batch dimension")
    })?;
    if batch == 0 {
        return Err(Error::from_reason(
            "forward_rows_independently requires at least one row",
        ));
    }
    let mut outputs = Vec::with_capacity(batch);
    let mut caches = Vec::with_capacity(batch);
    for row in 0..batch {
        let row_x = x.slice_axis(0, row as i64, row as i64 + 1)?;
        let mut row_cache = cache.row(row, batch)?;
        outputs.push(forward(&row_x, &mut row_cache)?);
        caches.push(row_cache);
    }
    let output = MxArray::concatenate_many(outputs.iter().collect(), Some(0))?;
    let cache_refs = caches.iter().collect::<Vec<_>>();
    Ok((output, ArraysCache::stack_rows(&cache_refs)?))
}

impl Clone for ArraysCache {
    fn clone(&self) -> Self {
        Self {
            cache: self.cache.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_and_scatter_preserve_request_rows() {
        let mut a = ArraysCache::new(2);
        a.set(0, MxArray::from_float32(&[1.0, 2.0], &[1, 2]).unwrap())
            .unwrap();
        a.set(1, MxArray::from_float32(&[3.0], &[1, 1]).unwrap())
            .unwrap();
        let mut b = ArraysCache::new(2);
        b.set(0, MxArray::from_float32(&[10.0, 20.0], &[1, 2]).unwrap())
            .unwrap();
        b.set(1, MxArray::from_float32(&[30.0], &[1, 1]).unwrap())
            .unwrap();

        let stacked = ArraysCache::stack_rows(&[&a, &b]).unwrap();
        assert_eq!(stacked.get(0).unwrap().shape().unwrap().as_ref(), [2, 2]);
        let row0 = stacked.row(0, 2).unwrap();
        let row1 = stacked.row(1, 2).unwrap();
        assert_eq!(
            row0.get(0).unwrap().to_float32().unwrap().as_ref(),
            [1.0, 2.0]
        );
        assert_eq!(
            row1.get(0).unwrap().to_float32().unwrap().as_ref(),
            [10.0, 20.0]
        );
        assert_eq!(row0.get(1).unwrap().to_float32().unwrap().as_ref(), [3.0]);
        assert_eq!(row1.get(1).unwrap().to_float32().unwrap().as_ref(), [30.0]);
    }
}
