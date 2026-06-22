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
    ///
    /// # Lazy-copy invariant
    ///
    /// `MxArray::copy()` is **lazy**: it appends a copy node to the MLX
    /// compute graph rather than performing an immediate `memcpy`. The
    /// snapshot is therefore captured against whatever the live cache array
    /// represents *at evaluation time*, not at the moment `snapshot()`
    /// returns.
    ///
    /// **Safety contract**: no in-place mutation of `conv_state` /
    /// `recurrent_state` is performed by any forward path in the qwen3_5 /
    /// qwen3_5_moe stack — every decode step replaces the cache slot with a
    /// fresh `MxArray` handle. As long as that contract holds, the lazy copy
    /// is functionally equivalent to a materialized snapshot for the
    /// purposes of speculative decode rollback.
    ///
    /// If a future change introduces in-place mutation on Linear caches
    /// (e.g. writing into an existing `conv_state` buffer rather than
    /// allocating a new one), this API MUST be updated to call `eval()` on
    /// each cloned `MxArray` before the mutation occurs, so the snapshot
    /// materializes against the pre-mutation tensor.
    ///
    /// # Paged-KV caveat
    ///
    /// This API targets the **flat (non-paged) cache path only**. When
    /// `Qwen3_5Config::use_block_paged_cache = true`, the FullAttention
    /// state lives in a `PagedKVCacheAdapter` and the `KVCache` slot held
    /// inside `Qwen3_5LayerCache::FullAttention` is a vestigial shell with
    /// `offset == 0` and empty K/V buffers. Snapshotting in that
    /// configuration would silently capture `offset = 0` and the
    /// corresponding [`Self::restore`] would no-op — yielding an invalid
    /// rollback that pretends to succeed.
    ///
    /// **Callers must guard with the paged adapter being `None` before
    /// snapshotting.** A `debug_assert!` below provides a louder failure in
    /// dev builds when a FullAttention slot is snapshotted with empty
    /// underlying buffers (the most reliable in-process signal that the
    /// paged path is active).
    #[allow(dead_code)] // Wired in by W6 speculative-decode loop.
    pub(crate) fn snapshot(&self) -> Result<Qwen3_5LayerSnapshot> {
        match self {
            Self::FullAttention(c) => {
                debug_assert!(
                    c.keys_ref().is_some() || c.get_offset() > 0,
                    "Qwen3_5LayerCache::snapshot: FullAttention cache has empty K/V buffer \
                     and offset=0 — likely the paged-KV path is active and this snapshot \
                     would silently capture nothing. See the Paged-KV caveat in the doc."
                );
                Ok(Qwen3_5LayerSnapshot::FullAttention {
                    offset: c.get_offset(),
                })
            }
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
    ///
    /// Also returns an error if the snapshot's FullAttention offset exceeds
    /// the current cache offset. [`KVCache::trim`] silently no-ops when
    /// `new_len >= offset`, which would otherwise hide bugs such as
    /// restoring a stale snapshot from a longer-running cache onto a
    /// shorter one.
    #[allow(dead_code)] // Wired in by W6 speculative-decode loop.
    pub(crate) fn restore(&mut self, snap: &Qwen3_5LayerSnapshot) -> Result<()> {
        match (self, snap) {
            (Self::FullAttention(c), Qwen3_5LayerSnapshot::FullAttention { offset }) => {
                let snap_offset = *offset;
                let cur_offset = c.get_offset();
                if snap_offset > cur_offset {
                    return Err(Error::from_reason(format!(
                        "Qwen3_5LayerCache::restore: snapshot offset {snap_offset} \
                         exceeds current cache offset {cur_offset}; KVCache::trim cannot \
                         grow the cache via restore (would silently no-op)",
                    )));
                }
                c.trim(snap_offset);
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
#[allow(dead_code)] // Wired in by W6 speculative-decode loop.
pub(crate) enum Qwen3_5LayerSnapshot {
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
#[allow(dead_code)] // Wired in by W6 speculative-decode loop.
pub(crate) fn snapshot_all(caches: &[Qwen3_5LayerCache]) -> Result<Vec<Qwen3_5LayerSnapshot>> {
    caches.iter().map(|c| c.snapshot()).collect()
}

/// Snapshot every layer for the eager-MTP rollback, paged-backend aware.
///
/// Identical to [`snapshot_all`] on the flat path. On the **paged** path the
/// FullAttention K/V lives in the `PagedKVCacheAdapter` pool rather than in
/// `caches`, so each `Qwen3_5LayerCache::FullAttention` slot is a vestigial
/// shell (empty K/V buffer, offset 0). The paged MTP rollback rewinds those
/// slots via `adapter.rollback_last_tokens` and never reads their snapshot
/// (see `DenseMtpStepper::rollback`), so this emits a benign
/// `FullAttention { offset: 0 }` placeholder for them instead of calling
/// [`Qwen3_5LayerCache::snapshot`] — which would trip that method's flat-only
/// empty-buffer `debug_assert!`. Linear (GDN) slots, whose recurrent state
/// lives in `caches` on both paths, are snapshotted normally.
pub(crate) fn snapshot_all_mtp(
    caches: &[Qwen3_5LayerCache],
    paged: bool,
) -> Result<Vec<Qwen3_5LayerSnapshot>> {
    caches
        .iter()
        .map(|c| match c {
            Qwen3_5LayerCache::FullAttention(_) if paged => {
                Ok(Qwen3_5LayerSnapshot::FullAttention { offset: 0 })
            }
            _ => c.snapshot(),
        })
        .collect()
}

/// Restore every layer's cache from a matching slice of snapshots.
///
/// Returns an error if the slice lengths differ or if any variant pair is
/// mismatched (caller is responsible for keeping snapshots aligned with the
/// cache vector).
///
/// # Atomicity
///
/// All variant pairs are pre-validated **before** any cache is mutated.
/// If any pair mismatches, the function returns `Err` and every cache is
/// left in its original state. After validation, the per-layer
/// [`Qwen3_5LayerCache::restore`] can still fail on the inner
/// `KVCache::trim` offset check; in that case the caches up to (but not
/// including) the failing index will have been rolled back. The variant
/// pre-validation removes the most common source of partial-rollback
/// inconsistency observed during speculative decoding.
#[allow(dead_code)] // Wired in by W6 speculative-decode loop.
pub(crate) fn restore_all(
    caches: &mut [Qwen3_5LayerCache],
    snaps: &[Qwen3_5LayerSnapshot],
) -> Result<()> {
    if caches.len() != snaps.len() {
        return Err(Error::from_reason(format!(
            "Qwen3_5LayerCache::restore_all: cache length {} != snapshot length {}",
            caches.len(),
            snaps.len()
        )));
    }
    // Pre-validate: ensure every (cache, snap) variant pair matches BEFORE
    // mutating anything. This prevents leaving the cache vector in a
    // partially-rolled-back state if a mismatch is detected mid-loop.
    for (idx, (cache, snap)) in caches.iter().zip(snaps.iter()).enumerate() {
        let matches = matches!(
            (cache, snap),
            (
                Qwen3_5LayerCache::FullAttention(_),
                Qwen3_5LayerSnapshot::FullAttention { .. },
            ) | (
                Qwen3_5LayerCache::Linear(_),
                Qwen3_5LayerSnapshot::Linear { .. },
            )
        );
        if !matches {
            let cache_kind = match cache {
                Qwen3_5LayerCache::FullAttention(_) => "FullAttention",
                Qwen3_5LayerCache::Linear(_) => "Linear",
            };
            let snap_kind = match snap {
                Qwen3_5LayerSnapshot::FullAttention { .. } => "FullAttention",
                Qwen3_5LayerSnapshot::Linear { .. } => "Linear",
            };
            return Err(Error::from_reason(format!(
                "Qwen3_5LayerCache::restore_all: variant mismatch at index {idx} \
                 (cache is {cache_kind}, snapshot is {snap_kind}); no caches mutated",
            )));
        }
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

        // Prime to offset N=4 with recognizable data. `from_float32` only
        // validates shape; if it errors here it's a real bug worth panicking
        // on. Metal-skip is handled in `try_eval_or_skip` below.
        let keys1 = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]).unwrap();
        let values1 = MxArray::from_float32(&[10.0, 20.0, 30.0, 40.0], &[1, 1, 4, 1]).unwrap();
        let (rk1, _) = kv.update_and_fetch(&keys1, &values1).unwrap();
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

        let conv0 = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let rec0 = MxArray::from_float32(&[5.0, 6.0, 7.0, 8.0, 9.0], &[1, 5]).unwrap();
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
                // Document the lazy-eval contract: `snapshot()` produces a
                // lazy `MxArray::copy()` graph node. Forcing `eval()` here
                // materializes the copy against the *current* tensor data,
                // BEFORE we mutate the live cache below. Without this, the
                // restore round-trip would still pass today (because the
                // cache slot is replaced, not mutated in place) — but it
                // would silently start failing the moment any future GDN
                // path introduces in-place mutation.
                if let Some(arr) = conv_state {
                    arr.eval();
                }
                if let Some(arr) = recurrent_state {
                    arr.eval();
                }
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
        let filler = MxArray::from_float32(&[1.0], &[1, 1]).unwrap();
        arrays.set(0, filler.clone());
        arrays.set(1, filler);

        cache.restore(&snap).expect("restore");
        let arrays = cache.as_arrays_cache_mut().unwrap();
        assert!(arrays.get(0).is_none());
        assert!(arrays.get(1).is_none());
    }

    #[test]
    fn restore_variant_mismatch_errors() {
        // Construct snapshots directly (rather than via `snapshot()`) so we
        // don't trip the Paged-KV `debug_assert!` on empty FullAttention
        // caches — the assertion is irrelevant to variant-mismatch testing.
        let full_snap = Qwen3_5LayerSnapshot::FullAttention { offset: 0 };
        let mut linear = Qwen3_5LayerCache::new_linear();
        let err = linear
            .restore(&full_snap)
            .expect_err("Linear cache must reject FullAttention snapshot");
        assert!(
            err.reason.to_string().contains("FullAttention"),
            "error message should name the offending variant: {err}",
        );

        // And the reverse direction.
        let linear_snap = Qwen3_5LayerSnapshot::Linear {
            conv_state: None,
            recurrent_state: None,
        };
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
            let k = MxArray::from_float32(&[1.0, 2.0], &[1, 1, 2, 1]).unwrap();
            let v = MxArray::from_float32(&[10.0, 20.0], &[1, 1, 2, 1]).unwrap();
            let (rk, _) = kv.update_and_fetch(&k, &v).unwrap();
            if !try_eval_or_skip(&rk, "snapshot_all_restore_all_round_trip::prime") {
                return;
            }
        }
        // Prime the linear cache.
        if let Some(arrays) = caches[1].as_arrays_cache_mut() {
            let c0 = MxArray::from_float32(&[7.0, 8.0], &[1, 2]).unwrap();
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
    fn snapshot_all_mtp_skips_empty_fullattention_shell_on_paged() {
        // Mirror the paged-MTP layout: a FullAttention slot is a vestigial
        // shell (empty K/V, offset 0) because its real K/V lives in the paged
        // adapter pool, alongside a Linear (GDN) slot whose recurrent state
        // DOES live in `caches` on both backends.
        let mut caches = vec![
            Qwen3_5LayerCache::new_full_attention(), // empty shell, offset 0
            Qwen3_5LayerCache::new_linear(),
        ];
        if let Some(arrays) = caches[1].as_arrays_cache_mut() {
            arrays.set(0, MxArray::from_float32(&[7.0, 8.0], &[1, 2]).unwrap());
        }

        // The paged-aware snapshot must NOT call `snapshot()` on the empty
        // shell (that trips its flat-only debug-assert in dev builds); it
        // emits a placeholder instead and snapshots the Linear slot normally.
        let snaps = snapshot_all_mtp(&caches, true).expect("paged snapshot must not panic");
        assert_eq!(snaps.len(), 2);
        assert!(
            matches!(snaps[0], Qwen3_5LayerSnapshot::FullAttention { offset: 0 }),
            "paged FullAttention shell must snapshot as an offset-0 placeholder",
        );
        assert!(
            matches!(snaps[1], Qwen3_5LayerSnapshot::Linear { .. }),
            "Linear (GDN) slot must be snapshotted normally even on the paged path",
        );
    }

    #[test]
    fn restore_all_length_mismatch_errors() {
        let mut caches = vec![Qwen3_5LayerCache::new_full_attention()];
        let snaps: Vec<Qwen3_5LayerSnapshot> = vec![];
        let err = restore_all(&mut caches, &snaps).expect_err("length mismatch");
        assert!(err.reason.to_string().contains("length"));
    }

    #[test]
    fn restore_all_atomic_on_variant_mismatch_no_mutation() {
        // Build [FullAttention, Linear] caches. Prime FullAttention to a
        // non-zero offset and Linear with a recognizable conv_state so we
        // can assert nothing was touched after the failed restore_all.
        let mut caches = vec![
            Qwen3_5LayerCache::new_full_attention(),
            Qwen3_5LayerCache::new_linear(),
        ];

        let mut primed_full_offset = 0;
        if let Qwen3_5LayerCache::FullAttention(kv) = &mut caches[0] {
            let k = MxArray::from_float32(&[1.0, 2.0], &[1, 1, 2, 1]).unwrap();
            let v = MxArray::from_float32(&[10.0, 20.0], &[1, 1, 2, 1]).unwrap();
            let (rk, _) = kv.update_and_fetch(&k, &v).unwrap();
            if !try_eval_or_skip(&rk, "restore_all_atomic::prime_full") {
                return;
            }
            primed_full_offset = kv.get_offset();
            assert_eq!(primed_full_offset, 2);
        }
        if let Some(arrays) = caches[1].as_arrays_cache_mut() {
            let c0 = MxArray::from_float32(&[42.0, 43.0], &[1, 2]).unwrap();
            arrays.set(0, c0);
        }

        // Build a snaps vec whose index 0 is deliberately Linear (mismatches
        // the FullAttention cache at index 0). Index 1 stays valid so we
        // verify that the FIRST mismatch detected aborts before the second
        // (otherwise-valid) restore runs.
        let valid_full_snap = caches[0].snapshot().expect("snapshot full");
        let valid_linear_snap = caches[1].snapshot().expect("snapshot linear");
        let snaps = vec![valid_linear_snap, valid_full_snap];

        let err = restore_all(&mut caches, &snaps)
            .expect_err("mismatched variant pair must error before any mutation");
        let msg = err.reason.to_string();
        assert!(
            msg.contains("variant mismatch") && msg.contains("no caches mutated"),
            "error must call out atomicity guarantee: {msg}",
        );

        // Verify both caches are unchanged.
        if let Qwen3_5LayerCache::FullAttention(kv) = &caches[0] {
            assert_eq!(
                kv.get_offset(),
                primed_full_offset,
                "FullAttention offset must not have moved",
            );
        } else {
            panic!("layer 0 must still be FullAttention");
        }
        if let Some(arrays) = caches[1].as_arrays_cache_mut() {
            assert_eq!(
                vec_of(arrays.get(0).expect("conv_state must still be present")),
                vec![42.0, 43.0],
                "Linear conv_state must not have been touched",
            );
        } else {
            panic!("layer 1 must still be Linear");
        }
    }

    #[test]
    fn restore_full_attention_rejects_grow_attempt() {
        // Prime cache to offset=4, then try to restore with a snapshot
        // whose offset is 10 — that would be a "grow via restore", which
        // KVCache::trim silently no-ops. The new defensive check must
        // surface an error naming both offsets.
        let mut cache = Qwen3_5LayerCache::new_full_attention();
        let Qwen3_5LayerCache::FullAttention(kv) = &mut cache else {
            unreachable!()
        };
        let k = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]).unwrap();
        let v = MxArray::from_float32(&[10.0, 20.0, 30.0, 40.0], &[1, 1, 4, 1]).unwrap();
        let (rk, _) = kv.update_and_fetch(&k, &v).unwrap();
        if !try_eval_or_skip(&rk, "restore_full_attention_rejects_grow_attempt::prime") {
            return;
        }
        assert_eq!(kv.get_offset(), 4);

        let bogus_snap = Qwen3_5LayerSnapshot::FullAttention { offset: 10 };
        let err = cache
            .restore(&bogus_snap)
            .expect_err("must reject snapshot offset > current offset");
        let msg = err.reason.to_string();
        assert!(
            msg.contains("10") && msg.contains("4"),
            "error message must include both offsets (got: {msg})",
        );

        // Offset must be unchanged after the rejected restore.
        if let Qwen3_5LayerCache::FullAttention(kv) = &cache {
            assert_eq!(kv.get_offset(), 4);
        } else {
            unreachable!();
        }
    }
}
