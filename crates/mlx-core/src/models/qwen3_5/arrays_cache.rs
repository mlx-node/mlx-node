use crate::array::MxArray;
use napi::bindgen_prelude::*;

/// Simple array cache for linear attention (GatedDeltaNet) layers.
///
/// Holds N arrays (typically 2 slots: conv_state and recurrent_state).
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
