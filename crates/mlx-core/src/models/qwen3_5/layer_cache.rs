use crate::array::MxArray;
use crate::transformer::KVCache;
use napi::bindgen_prelude::*;

use super::arrays_cache::ArraysCache;
use super::config::Qwen3_5Config;

/// Mixed cache type for Qwen3.5 layers.
///
/// Linear attention layers use ArraysCache (conv_state + recurrent_state).
/// Full attention layers use standard KVCache.
pub enum Qwen3_5LayerCache {
    Linear(ArraysCache),
    FullAttention(KVCache),
}

impl Qwen3_5LayerCache {
    /// Bytes of every linear layer's arrays in `caches`, by their real dtype.
    pub(crate) fn linear_state_bytes(caches: &[Self]) -> u64 {
        caches
            .iter()
            .filter_map(|cache| match cache {
                Self::Linear(arrays) => Some(arrays),
                Self::FullAttention(_) => None,
            })
            .flat_map(|arrays| (0..arrays.len()).filter_map(|idx| arrays.get(idx)))
            .fold(0u64, |sum, array| sum.saturating_add(array.nbytes() as u64))
    }

    /// Create a cache for a linear attention layer.
    pub fn new_linear() -> Self {
        Self::Linear(ArraysCache::new(2)) // 2 slots: conv_state, recurrent_state
    }

    /// Create a cache for a full attention layer (BF16 rows).
    pub fn new_full_attention() -> Self {
        Self::FullAttention(KVCache::new())
    }

    /// The per-layer cache slot for decoder layer `index` of `config`: a
    /// GDN state cache for linear layers, a bf16 full-attention K/V cache
    /// otherwise (a pool load's full-attention K/V lives in the pool).
    pub fn for_layer(config: &Qwen3_5Config, index: usize) -> Self {
        if config.is_linear_layer(index) {
            Self::new_linear()
        } else {
            Self::new_full_attention()
        }
    }

    /// One cache slot per decoder layer of `config` (see [`Self::for_layer`]).
    pub fn fresh_layer_caches(config: &Qwen3_5Config) -> Vec<Self> {
        (0..config.num_layers.max(0) as usize)
            .map(|index| Self::for_layer(config, index))
            .collect()
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

    #[cfg(test)]
    pub fn offset(&self) -> i32 {
        match self {
            Self::Linear(_) => 0,
            Self::FullAttention(c) => c.get_offset(),
        }
    }

    /// Collect references to all stored arrays in this cache slot.
    ///
    /// Gathers arrays for async_eval to prevent compute graph accumulation
    /// across decode steps.
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

    /// Capture a restore point for speculative decoding.
    ///
    /// * **FullAttention**: a placeholder; its K/V lives in the pool.
    /// * **Linear / GDN**: retains immutable aliases of `conv_state` and
    ///   `recurrent_state`. Each decode step replaces those cache slots with
    ///   new arrays, so the saved handles keep the pre-verify values alive
    ///   without adding copy operations to the graph.
    ///
    /// # Immutable-state invariant
    ///
    /// **Safety contract**: no in-place mutation of `conv_state` /
    /// `recurrent_state` is performed by any forward path in the qwen3_5 /
    /// qwen3_5_moe stack — every decode step replaces the cache slot with a
    /// fresh `MxArray` handle. As long as that contract holds, the alias
    /// is functionally equivalent to a materialized snapshot for the
    /// purposes of speculative decode rollback.
    ///
    /// If a future change introduces in-place mutation on Linear caches
    /// (e.g. writing into an existing `conv_state` buffer rather than
    /// allocating a new one), this API MUST first create independent copies
    /// and evaluate them before the mutation. Evaluating an alias alone does
    /// not detach its storage or protect the snapshot against mutation.
    pub(crate) fn snapshot(&self) -> Result<Qwen3_5LayerSnapshot> {
        match self {
            Self::FullAttention(_) => Ok(Qwen3_5LayerSnapshot::FullAttention),
            Self::Linear(c) => {
                // Lazy aliases suffice: every GDN state update is functional
                // (`ArraysCache::set` swaps in a new MxArray; `get` yields only
                // &MxArray, so no caller can `overwrite` the stored handle in
                // place). The snapshotted Arc keeps the pre-verify descriptor
                // and buffer alive and untouched until `restore` rebinds it —
                // same values as `.copy()` without ~72 copy kernels per
                // verify cycle.
                let conv_state = c.get(0).cloned();
                let recurrent_state = c.get(1).cloned();
                Ok(Qwen3_5LayerSnapshot::Linear {
                    conv_state,
                    recurrent_state,
                })
            }
        }
    }

    /// Roll the cache back to a previously captured [`Qwen3_5LayerSnapshot`].
    ///
    /// * **FullAttention**: nothing (its K/V lives in the pool).
    /// * **Linear**: swaps `conv_state` / `recurrent_state` back to the
    ///   snapshotted copies.
    ///
    /// Returns an error if the snapshot variant does not match the cache
    /// variant — speculative decoding code must keep the two in lock-step.
    #[cfg(test)]
    pub(crate) fn restore(&mut self, snap: &Qwen3_5LayerSnapshot) -> Result<()> {
        match (self, snap) {
            (Self::FullAttention(_), Qwen3_5LayerSnapshot::FullAttention) => Ok(()),
            (
                Self::Linear(c),
                Qwen3_5LayerSnapshot::Linear {
                    conv_state,
                    recurrent_state,
                },
            ) => {
                c.reset();
                if let Some(arr) = conv_state {
                    c.set(0, arr.clone())?;
                }
                if let Some(arr) = recurrent_state {
                    c.set(1, arr.clone())?;
                }
                Ok(())
            }
            (Self::FullAttention(_), Qwen3_5LayerSnapshot::Linear { .. }) => {
                Err(Error::from_reason(
                    "Qwen3_5LayerCache::restore: snapshot is Linear but cache slot is FullAttention",
                ))
            }
            (Self::Linear(_), Qwen3_5LayerSnapshot::FullAttention) => Err(Error::from_reason(
                "Qwen3_5LayerCache::restore: snapshot is FullAttention but cache slot is Linear",
            )),
        }
    }
}

/// Per-layer restore point captured by [`Qwen3_5LayerCache::snapshot`].
///
/// Used by speculative decode: snapshot before drafting,
/// restore (`rollback_after_verify` equivalent) if the verifier rejects.
pub(crate) enum Qwen3_5LayerSnapshot {
    /// A full-attention layer: its K/V lives in the pool, which the adapter
    /// rewinds, so there is nothing to keep here.
    FullAttention,
    /// Deep-cloned conv + recurrent state for a GatedDeltaNet layer.
    Linear {
        conv_state: Option<MxArray>,
        recurrent_state: Option<MxArray>,
    },
}

/// Restore `snap` into `caches` and replay the first `steps` recorded tape
/// steps — the eager-MTP GDN rewind both the dense and the MoE stepper drive
/// from their `rollback` and paged `rollback_unemitted` hooks.
///
/// Pure over `(snap, tape, steps)`: `steps == 0` degenerates to a bare
/// snapshot restore. A full-attention layer (no tape) keeps its K/V in the
/// pool, which the adapter rewinds, so its vestigial shell here is skipped.
/// `context` prefixes every error so the caller's family and hook stay
/// identifiable.
pub(crate) fn replay_mtp_snapshot_to(
    caches: &mut [Qwen3_5LayerCache],
    snap: &[Qwen3_5LayerSnapshot],
    tape: &[Option<super::gated_delta_net::GdnLayerTape>],
    steps: usize,
    context: &str,
) -> Result<()> {
    if caches.len() != snap.len() || caches.len() != tape.len() {
        return Err(Error::from_reason(format!(
            "{context}: length mismatch (caches {}, snapshot {}, tape {})",
            caches.len(),
            snap.len(),
            tape.len(),
        )));
    }
    for (idx, cache) in caches.iter_mut().enumerate() {
        let Some(layer_tape) = tape[idx].as_ref() else {
            match (&*cache, &snap[idx]) {
                (Qwen3_5LayerCache::FullAttention(_), Qwen3_5LayerSnapshot::FullAttention) => {
                    continue;
                }
                (_, Qwen3_5LayerSnapshot::FullAttention) => {
                    return Err(Error::from_reason(format!(
                        "{context}: layer {idx} has a FullAttention snapshot but its cache \
                         slot is not FullAttention",
                    )));
                }
                (_, Qwen3_5LayerSnapshot::Linear { .. }) => {
                    return Err(Error::from_reason(format!(
                        "{context}: layer {idx} has no GDN tape but a Linear snapshot",
                    )));
                }
            }
        };
        let arrays = cache.as_arrays_cache_mut().ok_or_else(|| {
            Error::from_reason(format!(
                "{context}: layer {idx} has a GDN tape but its cache slot is not Linear",
            ))
        })?;
        let (snap_conv, snap_rec) = match &snap[idx] {
            Qwen3_5LayerSnapshot::Linear {
                conv_state,
                recurrent_state,
            } => (conv_state.as_ref(), recurrent_state.as_ref()),
            Qwen3_5LayerSnapshot::FullAttention => {
                return Err(Error::from_reason(format!(
                    "{context}: layer {idx} GDN tape but FullAttention snapshot",
                )));
            }
        };
        let window = layer_tape.kernel.window_len()? as usize;
        if steps > window {
            return Err(Error::from_reason(format!(
                "{context}: target steps {steps} exceeds recorded window {window} at \
                 layer {idx}",
            )));
        }
        layer_tape.replay_into(arrays, snap_conv, snap_rec, steps)?;
    }
    Ok(())
}

/// Snapshot every layer for the eager-MTP rollback: the GDN state of each
/// Linear slot, and a placeholder for each FullAttention slot (its K/V
/// lives in the pool, rewound through `adapter.rollback_last_tokens`).
pub(crate) fn snapshot_all_mtp(caches: &[Qwen3_5LayerCache]) -> Result<Vec<Qwen3_5LayerSnapshot>> {
    caches.iter().map(|c| c.snapshot()).collect()
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
/// left in its original state.
#[cfg(test)]
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
                Qwen3_5LayerSnapshot::FullAttention,
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
                Qwen3_5LayerSnapshot::FullAttention => "FullAttention",
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
    fn snapshot_linear_byte_equality() {
        let mut cache = Qwen3_5LayerCache::new_linear();
        let arrays = cache.as_arrays_cache_mut().expect("Linear cache");

        let conv0 = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let rec0 = MxArray::from_float32(&[5.0, 6.0, 7.0, 8.0, 9.0], &[1, 5]).unwrap();
        if !try_eval_or_skip(&conv0, "snapshot_linear_byte_equality::seed") {
            return;
        }
        arrays.set(0, conv0).unwrap();
        arrays.set(1, rec0).unwrap();

        let snap = cache.snapshot().expect("snapshot");
        match &snap {
            Qwen3_5LayerSnapshot::Linear {
                conv_state,
                recurrent_state,
            } => {
                assert!(conv_state.is_some());
                assert!(recurrent_state.is_some());
                // Do not evaluate the snapshot before replacing the live
                // slots: its immutable aliases must preserve the old values
                // independently of when the lazy graph is evaluated.
            }
            _ => panic!("expected Linear snapshot"),
        }

        // Mutate both arrays by replacing the cache slots with junk.
        let arrays = cache.as_arrays_cache_mut().unwrap();
        arrays
            .set(
                0,
                MxArray::from_float32(&[-1.0, -2.0, -3.0, -4.0], &[1, 4]).unwrap(),
            )
            .unwrap();
        arrays
            .set(
                1,
                MxArray::from_float32(&[-5.0, -6.0, -7.0, -8.0, -9.0], &[1, 5]).unwrap(),
            )
            .unwrap();

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
        arrays.set(0, filler.clone()).unwrap();
        arrays.set(1, filler).unwrap();

        cache.restore(&snap).expect("restore");
        let arrays = cache.as_arrays_cache_mut().unwrap();
        assert!(arrays.get(0).is_none());
        assert!(arrays.get(1).is_none());
    }

    #[test]
    fn restore_variant_mismatch_errors() {
        let full_snap = Qwen3_5LayerSnapshot::FullAttention;
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

        // Prime the linear cache.
        if let Some(arrays) = caches[1].as_arrays_cache_mut() {
            let c0 = MxArray::from_float32(&[7.0, 8.0], &[1, 2]).unwrap();
            arrays.set(0, c0).unwrap();
        }

        let snaps = snapshot_all_mtp(&caches).expect("snapshot_all_mtp");
        assert_eq!(snaps.len(), 2);

        // Advance the linear cache past the snapshot.
        if let Some(arrays) = caches[1].as_arrays_cache_mut() {
            arrays
                .set(0, MxArray::from_float32(&[-1.0, -2.0], &[1, 2]).unwrap())
                .unwrap();
        }

        restore_all(&mut caches, &snaps).expect("restore_all");

        if let Some(arrays) = caches[1].as_arrays_cache_mut() {
            assert_eq!(
                vec_of(arrays.get(0).expect("conv_state restored")),
                vec![7.0, 8.0]
            );
        }
    }

    #[test]
    fn snapshot_all_mtp_keeps_a_fullattention_placeholder() {
        // Mirror the paged-MTP layout: a FullAttention slot is a vestigial
        // shell (empty K/V, offset 0) because its real K/V lives in the paged
        // adapter pool, alongside a Linear (GDN) slot whose recurrent state
        // DOES live in `caches` on both backends.
        let mut caches = vec![
            Qwen3_5LayerCache::new_full_attention(), // empty shell, offset 0
            Qwen3_5LayerCache::new_linear(),
        ];
        if let Some(arrays) = caches[1].as_arrays_cache_mut() {
            arrays
                .set(0, MxArray::from_float32(&[7.0, 8.0], &[1, 2]).unwrap())
                .unwrap();
        }

        let snaps = snapshot_all_mtp(&caches).expect("snapshot");
        assert_eq!(snaps.len(), 2);
        assert!(
            matches!(snaps[0], Qwen3_5LayerSnapshot::FullAttention),
            "the FullAttention shell snapshots as a placeholder",
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
            arrays.set(0, c0).unwrap();
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

    /// A tape-less (full-attention) layer keeps its K/V in the adapter pool,
    /// so the replay leaves the vestigial shell alone.
    #[test]
    fn replay_skips_the_full_attention_shell() {
        let mut caches = vec![Qwen3_5LayerCache::new_full_attention()];
        let k = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]).unwrap();
        let v = MxArray::from_float32(&[5.0, 6.0, 7.0, 8.0], &[1, 1, 4, 1]).unwrap();
        if !try_eval_or_skip(&k, "replay_skips_the_full_attention_shell::prime") {
            return;
        }
        {
            let kv = caches[0].as_kv_cache_mut().unwrap();
            kv.update_and_fetch(&k, &v).unwrap();
            assert_eq!(kv.get_offset(), 4);
        }
        let snap = vec![Qwen3_5LayerSnapshot::FullAttention];
        let tape: Vec<Option<crate::models::qwen3_5::gated_delta_net::GdnLayerTape>> = vec![None];
        replay_mtp_snapshot_to(&mut caches, &snap, &tape, 1, "replay")
            .expect("the replay must skip the shell, not fail on it");
        assert_eq!(
            caches[0].offset(),
            4,
            "a replay must not move the vestigial full-attention shell"
        );
    }

    /// Length disagreement between the three parallel slices is reported with
    /// all three counts rather than panicking on an index — the shape a
    /// mismatched layer_kinds/tape wiring takes.
    #[test]
    fn replay_reports_a_length_mismatch_instead_of_indexing_past_the_tape() {
        let mut caches = vec![Qwen3_5LayerCache::new_linear()];
        let snap = vec![Qwen3_5LayerSnapshot::Linear {
            conv_state: None,
            recurrent_state: None,
        }];
        let tape: Vec<Option<crate::models::qwen3_5::gated_delta_net::GdnLayerTape>> = Vec::new();
        let err = replay_mtp_snapshot_to(&mut caches, &snap, &tape, 0, "ctx")
            .expect_err("a short tape must be refused");
        assert!(err.reason.contains("length mismatch"), "{}", err.reason);
        assert!(err.reason.contains("ctx"), "{}", err.reason);
    }

    #[test]
    fn replay_rejects_a_missing_linear_tape() {
        let mut caches = vec![Qwen3_5LayerCache::new_linear()];
        let snap = vec![Qwen3_5LayerSnapshot::Linear {
            conv_state: None,
            recurrent_state: None,
        }];
        for steps in [0, 1] {
            let err = replay_mtp_snapshot_to(&mut caches, &snap, &[None], steps, "missing tape")
                .expect_err("paged K/V ownership does not excuse a missing recurrent tape");
            assert!(err.reason.contains("no GDN tape"), "{}", err.reason);
        }
    }
}
