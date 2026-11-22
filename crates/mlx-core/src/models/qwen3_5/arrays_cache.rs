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

    #[test]
    fn kv_cache_and_arrays_cache_broadcast_batch() {
        use super::super::attention::Qwen3_5Attention;
        use super::super::config::Qwen3_5Config;
        use super::super::gated_delta_net::GatedDeltaNet;
        use crate::array::DType;
        use crate::transformer::KVCache;

        let cfg = Qwen3_5Config {
            rope_scaling: None,
            qwen35_gguf_gdn_layout: None,
            kv_format: None,
            vocab_size: 32,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 16,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 128,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 2,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 2,
            partial_rotary_factor: 0.5,
            rope_theta: 10_000.0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            n_mtp_layers: 0,
        };

        let mut attn = Qwen3_5Attention::new(&cfg).unwrap();
        let mut gdn = GatedDeltaNet::new(&cfg).unwrap();

        // Deterministic weights so parity assertions are reproducible across runs.
        let det = |shape: &[i64], salt: f32, scale: f32| -> MxArray {
            let n: usize = shape.iter().map(|&d| d as usize).product();
            let vals: Vec<f32> = (0..n)
                .map(|i| ((i as f32) * 0.3719 + salt).sin() * scale)
                .collect();
            MxArray::from_float32(&vals, shape).unwrap()
        };
        let hidden = cfg.hidden_size as i64;
        let h = cfg.num_heads as i64;
        let kv = cfg.num_kv_heads as i64;
        let d = cfg.head_dim as i64;
        attn.set_q_proj_weight(&det(&[2 * h * d, hidden], 1.0, 0.12))
            .unwrap();
        attn.set_k_proj_weight(&det(&[kv * d, hidden], 2.0, 0.12))
            .unwrap();
        attn.set_v_proj_weight(&det(&[kv * d, hidden], 3.0, 0.12))
            .unwrap();
        attn.set_o_proj_weight(&det(&[hidden, h * d], 4.0, 0.12))
            .unwrap();

        let key_dim = cfg.linear_key_dim() as i64;
        let val_dim = cfg.linear_value_dim() as i64;
        let conv_dim = cfg.linear_conv_dim() as i64;
        let hv = cfg.linear_num_value_heads as i64;
        gdn.set_in_proj_qkvz_weight(&det(&[2 * key_dim + 2 * val_dim, hidden], 5.0, 0.12))
            .unwrap();
        gdn.set_in_proj_ba_weight(&det(&[2 * hv, hidden], 6.0, 0.12))
            .unwrap();
        gdn.set_conv1d_weight(&det(&[conv_dim, 4, 1], 7.0, 0.2), DType::Float32)
            .unwrap();
        gdn.set_out_proj_weight(&det(&[hidden, val_dim], 8.0, 0.12))
            .unwrap();

        // Shared prefix (B=1, T=5)
        let prefix_vals: Vec<f32> = (0..(5 * 64))
            .map(|i| ((i as f32) * 0.037).sin() * 0.25)
            .collect();
        let prefix = MxArray::from_float32(&prefix_vals, &[1, 5, 64]).unwrap();

        let mut kv_prefix = KVCache::new();
        let mut gdn_prefix = ArraysCache::new(2);
        let _ = attn
            .forward(&prefix, None, Some(&mut kv_prefix), None)
            .unwrap();
        let _ = gdn
            .forward(&prefix, None, Some(&mut gdn_prefix), false)
            .unwrap();

        // Validation: batch < 1 fails
        assert!(kv_prefix.broadcast_batch(0).is_err());
        assert!(gdn_prefix.broadcast_batch(0).is_err());

        // Broadcast B=1 -> B=3
        let mut kv_batched = kv_prefix.broadcast_batch(3).unwrap();
        let mut gdn_batched = gdn_prefix.broadcast_batch(3).unwrap();
        assert_eq!(kv_batched.get_offset(), 5);
        assert_eq!(
            kv_batched.keys_ref().unwrap().shape().unwrap().as_ref(),
            [3, 2, 5, 16]
        );
        assert_eq!(
            gdn_batched.get(0).unwrap().shape().unwrap().as_ref(),
            [3, 3, conv_dim]
        );
        assert_eq!(
            gdn_batched.get(1).unwrap().shape().unwrap().as_ref(),
            [3, 4, 16, 16]
        );

        // Broadcasted rows must be bit-identical to the B=1 prefix before divergence.
        let prefix_k = kv_prefix
            .keys_ref()
            .unwrap()
            .slice_axis(2, 0, 5)
            .unwrap()
            .to_float32()
            .unwrap();
        let prefix_conv = gdn_prefix.get(0).unwrap().to_float32().unwrap();
        let prefix_rec = gdn_prefix.get(1).unwrap().to_float32().unwrap();
        for b in 0..3i64 {
            let row_k = kv_batched
                .keys_ref()
                .unwrap()
                .slice_axis(0, b, b + 1)
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(row_k.as_ref(), prefix_k.as_ref());
            let row_conv = gdn_batched
                .get(0)
                .unwrap()
                .slice_axis(0, b, b + 1)
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(row_conv.as_ref(), prefix_conv.as_ref());
            let row_rec = gdn_batched
                .get(1)
                .unwrap()
                .slice_axis(0, b, b + 1)
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(row_rec.as_ref(), prefix_rec.as_ref());
        }

        // Validation: broadcasting a B=3 cache fails
        assert!(kv_batched.broadcast_batch(2).is_err());
        assert!(gdn_batched.broadcast_batch(2).is_err());

        // Independent B=1 clones to diverge separately
        let mut kv_singles: Vec<KVCache> = (0..3)
            .map(|_| kv_prefix.broadcast_batch(1).unwrap())
            .collect();
        let mut gdn_singles: Vec<ArraysCache> = (0..3)
            .map(|_| gdn_prefix.broadcast_batch(1).unwrap())
            .collect();

        // Run 2 distinct decode steps (T=1 each) and assert parity across every row
        for step in 0..2 {
            let mut step_rows = Vec::with_capacity(3);
            for b in 0..3 {
                let data: Vec<f32> = (0..64)
                    .map(|i| (((b * 97 + step * 31 + i) as f32) * 0.11).cos() * 0.3)
                    .collect();
                step_rows.push(MxArray::from_float32(&data, &[1, 1, 64]).unwrap());
            }
            let x_batch = MxArray::concatenate_many(step_rows.iter().collect(), Some(0)).unwrap();

            let attn_batch_out = attn
                .forward(&x_batch, None, Some(&mut kv_batched), None)
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            let gdn_batch_out = gdn
                .forward(&x_batch, None, Some(&mut gdn_batched), false)
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();

            for b in 0..3 {
                let attn_single = attn
                    .forward(&step_rows[b], None, Some(&mut kv_singles[b]), None)
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap();
                let gdn_single = gdn
                    .forward(&step_rows[b], None, Some(&mut gdn_singles[b]), false)
                    .unwrap()
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap();

                let offset = b * 64;
                let attn_diff = attn_single
                    .as_ref()
                    .iter()
                    .zip(&attn_batch_out.as_ref()[offset..offset + 64])
                    .map(|(a, c)| (a - c).abs())
                    .fold(0.0f32, f32::max);
                let gdn_diff = gdn_single
                    .as_ref()
                    .iter()
                    .zip(&gdn_batch_out.as_ref()[offset..offset + 64])
                    .map(|(a, c)| (a - c).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    attn_diff < 1e-4,
                    "KVCache broadcast_batch mismatch at step={step}, row={b}: max_diff={attn_diff}"
                );
                assert!(
                    gdn_diff < 1e-4,
                    "ArraysCache broadcast_batch mismatch at step={step}, row={b}: max_diff={gdn_diff}"
                );
            }
        }

        // Original B=1 prefix caches remain unmutated
        assert_eq!(kv_prefix.get_offset(), 5);
        assert_eq!(gdn_prefix.get(0).unwrap().shape_at(0).unwrap(), 1);
    }
}
