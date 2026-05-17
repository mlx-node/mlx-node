use crate::array::MxArray;
use crate::transformer::KVCache;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::arrays_cache::ArraysCache;

/// Mixed cache type for Qwen3.5 layers.
///
/// Linear attention layers use ArraysCache (conv_state + recurrent_state).
/// Full attention layers use standard KVCache.
pub enum Qwen3_5LayerCache {
    Linear(ArraysCache),
    FullAttention(KVCache),
}

impl Qwen3_5LayerCache {
    /// Create a cache for a linear attention layer.
    pub fn new_linear() -> Self {
        Self::Linear(ArraysCache::new(2)) // 2 slots: conv_state, recurrent_state
    }

    /// Create a cache for a full attention layer.
    pub fn new_full_attention() -> Self {
        Self::FullAttention(KVCache::new())
    }

    /// Get as mutable ArraysCache, or None if this is a full-attention cache.
    pub fn as_arrays_cache_mut(&mut self) -> Option<&mut ArraysCache> {
        match self {
            Self::Linear(c) => Some(c),
            _ => None,
        }
    }

    /// Get as mutable KVCache, or None if this is a linear-attention cache.
    pub fn as_kv_cache_mut(&mut self) -> Option<&mut KVCache> {
        match self {
            Self::FullAttention(c) => Some(c),
            _ => None,
        }
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        match self {
            Self::Linear(c) => c.reset(),
            Self::FullAttention(c) => c.reset(),
        }
    }

    /// Get the cache offset (for mask generation).
    /// For linear layers, returns 0 (not position-based).
    /// For full attention layers, returns the KVCache offset.
    pub fn offset(&self) -> i32 {
        match self {
            Self::Linear(_) => 0,
            Self::FullAttention(c) => c.get_offset(),
        }
    }

    /// Export cache as 2 raw pointers for the fused C++ forward pass.
    ///
    /// For linear layers: (conv_state_ptr, recurrent_state_ptr)
    /// For full attention layers: (keys_ptr, values_ptr)
    ///
    /// Returns null pointers if the cache slot is empty.
    pub fn export_ptrs(&self) -> (*mut sys::mlx_array, *mut sys::mlx_array) {
        match self {
            Self::Linear(c) => {
                let p0 = c.get(0).map_or(std::ptr::null_mut(), |a| a.as_raw_ptr());
                let p1 = c.get(1).map_or(std::ptr::null_mut(), |a| a.as_raw_ptr());
                (p0, p1)
            }
            Self::FullAttention(c) => {
                let keys_ptr = c
                    .keys_ref()
                    .map_or(std::ptr::null_mut(), |a| a.as_raw_ptr());
                let values_ptr = c
                    .values_ref()
                    .map_or(std::ptr::null_mut(), |a| a.as_raw_ptr());
                (keys_ptr, values_ptr)
            }
        }
    }

    /// Collect references to all stored arrays in this cache slot.
    ///
    /// Used after import_ptrs to gather arrays for async_eval to prevent
    /// compute graph accumulation across decode steps.
    pub fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        match self {
            Self::Linear(c) => {
                if let Some(arr) = c.get(0) {
                    out.push(arr);
                }
                if let Some(arr) = c.get(1) {
                    out.push(arr);
                }
            }
            Self::FullAttention(c) => {
                if let Some(k) = c.keys_ref() {
                    out.push(k);
                }
                if let Some(v) = c.values_ref() {
                    out.push(v);
                }
            }
        }
    }

    /// Import cache from 2 raw pointers returned by the fused C++ forward pass.
    ///
    /// Takes ownership of the pointers (wraps them in MxArray).
    pub fn import_ptrs(
        &mut self,
        p0: *mut sys::mlx_array,
        p1: *mut sys::mlx_array,
        new_offset: i32,
    ) {
        match self {
            Self::Linear(c) => {
                if !p0.is_null()
                    && let Ok(arr) = MxArray::from_handle(p0, "fused_conv_state")
                {
                    c.set(0, arr);
                }
                if !p1.is_null()
                    && let Ok(arr) = MxArray::from_handle(p1, "fused_recurrent_state")
                {
                    c.set(1, arr);
                }
            }
            Self::FullAttention(c) => {
                if !p0.is_null()
                    && let Ok(keys) = MxArray::from_handle(p0, "fused_kv_keys")
                {
                    c.set_keys(keys);
                }
                if !p1.is_null()
                    && let Ok(values) = MxArray::from_handle(p1, "fused_kv_values")
                {
                    c.set_values(values);
                }
                c.set_offset(new_offset);
            }
        }
    }

    /// Capture a restore point for speculative decoding.
    ///
    /// * **FullAttention**: snapshots only the logical `offset` — the pre-allocated
    ///   K/V buffer is reused in place. After [`Self::restore`], the slots beyond
    ///   the snapshotted offset become "free for reuse" and the next
    ///   `update_and_fetch` overwrites them. No tensor data is copied.
    /// * **Linear / GDN**: clones `conv_state` and `recurrent_state`. Both arrays
    ///   are replaced (not mutated in place) on every decode step, but we issue
    ///   an explicit `MxArray::copy()` so the saved handle owns independent
    ///   storage in the MLX graph — matching MTPLX's
    ///   `cache_state._clone_tree` invariant and guarding against any future
    ///   GDN path that introduces in-place mutation.
    pub fn snapshot(&self) -> Result<Qwen3_5LayerSnapshot> {
        match self {
            Self::FullAttention(c) => Ok(Qwen3_5LayerSnapshot::FullAttention {
                offset: c.get_offset(),
            }),
            Self::Linear(c) => {
                let conv_state = match c.get(0) {
                    Some(arr) => Some(arr.copy()?),
                    None => None,
                };
                let recurrent_state = match c.get(1) {
                    Some(arr) => Some(arr.copy()?),
                    None => None,
                };
                Ok(Qwen3_5LayerSnapshot::Linear {
                    conv_state,
                    recurrent_state,
                })
            }
        }
    }

    /// Roll the cache back to a previously captured [`Qwen3_5LayerSnapshot`].
    ///
    /// * **FullAttention**: rewinds the logical offset via [`KVCache::trim`].
    ///   The underlying K/V buffer is unchanged; subsequent writes will
    ///   overwrite the stale tail in place.
    /// * **Linear**: swaps `conv_state` / `recurrent_state` back to the
    ///   snapshotted copies.
    ///
    /// Returns an error if the snapshot variant does not match the cache
    /// variant — speculative decoding code must keep the two in lock-step.
    pub fn restore(&mut self, snap: &Qwen3_5LayerSnapshot) -> Result<()> {
        match (self, snap) {
            (Self::FullAttention(c), Qwen3_5LayerSnapshot::FullAttention { offset }) => {
                c.trim(*offset);
                Ok(())
            }
            (
                Self::Linear(c),
                Qwen3_5LayerSnapshot::Linear {
                    conv_state,
                    recurrent_state,
                },
            ) => {
                c.reset();
                if let Some(arr) = conv_state {
                    c.set(0, arr.clone());
                }
                if let Some(arr) = recurrent_state {
                    c.set(1, arr.clone());
                }
                Ok(())
            }
            (Self::FullAttention(_), Qwen3_5LayerSnapshot::Linear { .. }) => {
                Err(Error::from_reason(
                    "Qwen3_5LayerCache::restore: snapshot is Linear but cache slot is FullAttention",
                ))
            }
            (Self::Linear(_), Qwen3_5LayerSnapshot::FullAttention { .. }) => {
                Err(Error::from_reason(
                    "Qwen3_5LayerCache::restore: snapshot is FullAttention but cache slot is Linear",
                ))
            }
        }
    }
}

/// Per-layer restore point captured by [`Qwen3_5LayerCache::snapshot`].
///
/// Designed for the W6 speculative-decode loop: snapshot before drafting,
/// restore (`rollback_after_verify` equivalent) if the verifier rejects.
pub enum Qwen3_5LayerSnapshot {
    /// Logical token offset of the full-attention KV cache. The underlying
    /// pre-allocated buffer is shared with the live cache and is rewound by
    /// trimming the offset on restore — zero tensor copy.
    FullAttention { offset: i32 },
    /// Deep-cloned conv + recurrent state for a GatedDeltaNet layer.
    Linear {
        conv_state: Option<MxArray>,
        recurrent_state: Option<MxArray>,
    },
}

/// Snapshot every layer's cache in one shot.
pub fn snapshot_all(caches: &[Qwen3_5LayerCache]) -> Result<Vec<Qwen3_5LayerSnapshot>> {
    caches.iter().map(|c| c.snapshot()).collect()
}

/// Restore every layer's cache from a matching slice of snapshots.
///
/// Returns an error if the slice lengths differ or if any variant pair is
/// mismatched (caller is responsible for keeping snapshots aligned with the
/// cache vector).
pub fn restore_all(caches: &mut [Qwen3_5LayerCache], snaps: &[Qwen3_5LayerSnapshot]) -> Result<()> {
    if caches.len() != snaps.len() {
        return Err(Error::from_reason(format!(
            "Qwen3_5LayerCache::restore_all: cache length {} != snapshot length {}",
            caches.len(),
            snaps.len()
        )));
    }
    for (cache, snap) in caches.iter_mut().zip(snaps.iter()) {
        cache.restore(snap)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Best-effort eval that swallows Metal-unavailable failures so the test
    /// suite still runs on hosts without a GPU.
    fn try_eval_or_skip(arr: &MxArray, label: &str) -> bool {
        // `MxArray::eval` is infallible at the FFI level, but the subsequent
        // data-copy (`to_float32`) is the path that surfaces Metal init
        // failures. We try it once and signal the caller to skip on error.
        arr.eval();
        if let Err(e) = arr.to_float32() {
            let msg = e.reason.to_string();
            if msg.contains("Metal") || msg.contains("device") || msg.contains("metal") {
                eprintln!("skipping {label}: Metal unavailable ({msg})");
                return false;
            }
            // Surface unexpected errors loudly.
            panic!("unexpected eval failure in {label}: {msg}");
        }
        true
    }

    fn vec_of(arr: &MxArray) -> Vec<f32> {
        arr.eval();
        arr.to_float32().expect("to_float32").to_vec()
    }

    #[test]
    fn snapshot_full_attention_byte_equality() {
        let mut cache = Qwen3_5LayerCache::new_full_attention();
        let Qwen3_5LayerCache::FullAttention(kv) = &mut cache else {
            unreachable!()
        };

        // Prime to offset N=4 with recognizable data.
        let keys1 = match MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]) {
            Ok(a) => a,
            Err(_) => return,
        };
        let values1 = match MxArray::from_float32(&[10.0, 20.0, 30.0, 40.0], &[1, 1, 4, 1]) {
            Ok(a) => a,
            Err(_) => return,
        };
        let (rk1, _) = match kv.update_and_fetch(&keys1, &values1) {
            Ok(v) => v,
            Err(_) => return,
        };
        if !try_eval_or_skip(&rk1, "snapshot_full_attention_byte_equality::prime") {
            return;
        }
        assert_eq!(kv.get_offset(), 4);

        // Snapshot at offset=4, then write 3 more entries.
        let snap = cache.snapshot().expect("snapshot");
        match &snap {
            Qwen3_5LayerSnapshot::FullAttention { offset } => assert_eq!(*offset, 4),
            _ => panic!("expected FullAttention snapshot"),
        }

        let Qwen3_5LayerCache::FullAttention(kv) = &mut cache else {
            unreachable!()
        };
        let keys2 = MxArray::from_float32(&[5.0, 6.0, 7.0], &[1, 1, 3, 1]).unwrap();
        let values2 = MxArray::from_float32(&[50.0, 60.0, 70.0], &[1, 1, 3, 1]).unwrap();
        kv.update_and_fetch(&keys2, &values2).unwrap();
        assert_eq!(kv.get_offset(), 7);

        // Restore: offset must rewind to 4; the trailing 3 slots stay in the
        // pre-allocated buffer but are no longer logically valid.
        cache.restore(&snap).expect("restore");
        let Qwen3_5LayerCache::FullAttention(kv) = &cache else {
            unreachable!()
        };
        assert_eq!(kv.get_offset(), 4);

        // Subsequent writes must resume from offset=4 and overwrite the stale
        // tail in place — i.e. the buffer's logical tail equals the new data.
        let Qwen3_5LayerCache::FullAttention(kv) = &mut cache else {
            unreachable!()
        };
        let keys3 = MxArray::from_float32(&[100.0, 200.0], &[1, 1, 2, 1]).unwrap();
        let values3 = MxArray::from_float32(&[1000.0, 2000.0], &[1, 1, 2, 1]).unwrap();
        let (rk, rv) = kv.update_and_fetch(&keys3, &values3).unwrap();
        assert_eq!(kv.get_offset(), 6);
        assert_eq!(vec_of(&rk), vec![1.0, 2.0, 3.0, 4.0, 100.0, 200.0]);
        assert_eq!(vec_of(&rv), vec![10.0, 20.0, 30.0, 40.0, 1000.0, 2000.0]);
    }

    #[test]
    fn snapshot_linear_byte_equality() {
        let mut cache = Qwen3_5LayerCache::new_linear();
        let arrays = cache.as_arrays_cache_mut().expect("Linear cache");

        let conv0 = match MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]) {
            Ok(a) => a,
            Err(_) => return,
        };
        let rec0 = match MxArray::from_float32(&[5.0, 6.0, 7.0, 8.0, 9.0], &[1, 5]) {
            Ok(a) => a,
            Err(_) => return,
        };
        if !try_eval_or_skip(&conv0, "snapshot_linear_byte_equality::seed") {
            return;
        }
        arrays.set(0, conv0);
        arrays.set(1, rec0);

        let snap = cache.snapshot().expect("snapshot");
        match &snap {
            Qwen3_5LayerSnapshot::Linear {
                conv_state,
                recurrent_state,
            } => {
                assert!(conv_state.is_some());
                assert!(recurrent_state.is_some());
            }
            _ => panic!("expected Linear snapshot"),
        }

        // Mutate both arrays by replacing the cache slots with junk.
        let arrays = cache.as_arrays_cache_mut().unwrap();
        arrays.set(
            0,
            MxArray::from_float32(&[-1.0, -2.0, -3.0, -4.0], &[1, 4]).unwrap(),
        );
        arrays.set(
            1,
            MxArray::from_float32(&[-5.0, -6.0, -7.0, -8.0, -9.0], &[1, 5]).unwrap(),
        );

        // Restore and confirm byte-for-byte equality with the original seeds.
        cache.restore(&snap).expect("restore");
        let arrays = cache.as_arrays_cache_mut().unwrap();
        assert_eq!(
            vec_of(arrays.get(0).expect("conv_state restored")),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            vec_of(arrays.get(1).expect("recurrent_state restored")),
            vec![5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn snapshot_linear_empty_slots_round_trip() {
        // Snapshotting an empty Linear cache must yield None/None and restore
        // must reset the cache (no panics, no dangling state).
        let mut cache = Qwen3_5LayerCache::new_linear();
        let snap = cache.snapshot().expect("snapshot empty");
        match &snap {
            Qwen3_5LayerSnapshot::Linear {
                conv_state,
                recurrent_state,
            } => {
                assert!(conv_state.is_none());
                assert!(recurrent_state.is_none());
            }
            _ => panic!("expected Linear snapshot"),
        }

        // Fill the cache, then restore from the empty snapshot — both slots
        // should clear.
        let arrays = cache.as_arrays_cache_mut().unwrap();
        let Ok(filler) = MxArray::from_float32(&[1.0], &[1, 1]) else {
            return;
        };
        arrays.set(0, filler.clone());
        arrays.set(1, filler);

        cache.restore(&snap).expect("restore");
        let arrays = cache.as_arrays_cache_mut().unwrap();
        assert!(arrays.get(0).is_none());
        assert!(arrays.get(1).is_none());
    }

    #[test]
    fn restore_variant_mismatch_errors() {
        // Snapshot a FullAttention slot, try to restore into a Linear cache.
        let full = Qwen3_5LayerCache::new_full_attention();
        let full_snap = full.snapshot().expect("snapshot full");
        let mut linear = Qwen3_5LayerCache::new_linear();
        let err = linear
            .restore(&full_snap)
            .expect_err("Linear cache must reject FullAttention snapshot");
        assert!(
            err.reason.to_string().contains("FullAttention"),
            "error message should name the offending variant: {err}",
        );

        // And the reverse direction.
        let linear = Qwen3_5LayerCache::new_linear();
        let linear_snap = linear.snapshot().expect("snapshot linear");
        let mut full = Qwen3_5LayerCache::new_full_attention();
        let err = full
            .restore(&linear_snap)
            .expect_err("FullAttention cache must reject Linear snapshot");
        assert!(
            err.reason.to_string().contains("Linear"),
            "error message should name the offending variant: {err}",
        );
    }

    #[test]
    fn snapshot_all_restore_all_round_trip() {
        let mut caches = vec![
            Qwen3_5LayerCache::new_full_attention(),
            Qwen3_5LayerCache::new_linear(),
        ];

        // Prime the full-attention cache.
        if let Qwen3_5LayerCache::FullAttention(kv) = &mut caches[0] {
            let Ok(k) = MxArray::from_float32(&[1.0, 2.0], &[1, 1, 2, 1]) else {
                return;
            };
            let Ok(v) = MxArray::from_float32(&[10.0, 20.0], &[1, 1, 2, 1]) else {
                return;
            };
            let Ok((rk, _)) = kv.update_and_fetch(&k, &v) else {
                return;
            };
            if !try_eval_or_skip(&rk, "snapshot_all_restore_all_round_trip::prime") {
                return;
            }
        }
        // Prime the linear cache.
        if let Some(arrays) = caches[1].as_arrays_cache_mut() {
            let Ok(c0) = MxArray::from_float32(&[7.0, 8.0], &[1, 2]) else {
                return;
            };
            arrays.set(0, c0);
        }

        let snaps = snapshot_all(&caches).expect("snapshot_all");
        assert_eq!(snaps.len(), 2);

        // Advance both caches past the snapshot.
        if let Qwen3_5LayerCache::FullAttention(kv) = &mut caches[0] {
            let k = MxArray::from_float32(&[99.0], &[1, 1, 1, 1]).unwrap();
            let v = MxArray::from_float32(&[99.0], &[1, 1, 1, 1]).unwrap();
            kv.update_and_fetch(&k, &v).unwrap();
            assert_eq!(kv.get_offset(), 3);
        }
        if let Some(arrays) = caches[1].as_arrays_cache_mut() {
            arrays.set(0, MxArray::from_float32(&[-1.0, -2.0], &[1, 2]).unwrap());
        }

        restore_all(&mut caches, &snaps).expect("restore_all");

        if let Qwen3_5LayerCache::FullAttention(kv) = &caches[0] {
            assert_eq!(kv.get_offset(), 2);
        } else {
            panic!("layer 0 must remain FullAttention");
        }
        if let Some(arrays) = caches[1].as_arrays_cache_mut() {
            assert_eq!(
                vec_of(arrays.get(0).expect("conv_state restored")),
                vec![7.0, 8.0]
            );
        }
    }

    #[test]
    fn restore_all_length_mismatch_errors() {
        let mut caches = vec![Qwen3_5LayerCache::new_full_attention()];
        let snaps: Vec<Qwen3_5LayerSnapshot> = vec![];
        let err = restore_all(&mut caches, &snaps).expect_err("length mismatch");
        assert!(err.reason.to_string().contains("length"));
    }
}
