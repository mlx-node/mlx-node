use crate::array::MxArray;
use crate::transformer::rotating_kv_cache::{RotatingKVCacheSnapshot, RotatingKVCacheState};
use crate::transformer::{KVCache, RotatingKVCache};
use napi::bindgen_prelude::*;

/// Internal cache type discriminant.
enum CacheType {
    Global(KVCache),
    Sliding(RotatingKVCache),
}

/// Per-layer cache for Gemma4 decoder layers.
///
/// Global (full attention) layers use KVCache.
/// Sliding (local attention) layers use RotatingKVCache with window size.
///
/// Includes a K/V stash for correct KV sharing. During prefill, the
/// RotatingKVCache returns the FULL untrimmed sequence for attention but stores
/// only the trimmed window internally. The stash captures the returned K/V so
/// that shared layers receive the same context the anchor attention actually used.
pub struct Gemma4LayerCache {
    inner: CacheType,
    /// Stashed K/V from the last `update_and_fetch_stash` call.
    /// During prefill this is the FULL untrimmed sequence (even if the cache
    /// stores only a sliding window). During decode this is the current
    /// window/full state. Used by KV sharing to get the K/V the anchor
    /// attention actually used.
    stashed_kv: Option<(MxArray, MxArray)>,
}

impl Gemma4LayerCache {
    pub fn new_global() -> Self {
        Gemma4LayerCache {
            inner: CacheType::Global(KVCache::new()),
            stashed_kv: None,
        }
    }

    pub fn new_sliding(window_size: i32) -> Self {
        Gemma4LayerCache {
            inner: CacheType::Sliding(RotatingKVCache::new(window_size, None)),
            stashed_kv: None,
        }
    }

    /// Get the current offset (number of tokens cached).
    pub fn get_offset(&self) -> i32 {
        match &self.inner {
            CacheType::Global(c) => c.get_offset(),
            CacheType::Sliding(c) => c.get_offset(),
        }
    }

    /// Returns true when this Gemma layer owns a sliding rotating cache.
    pub fn is_sliding(&self) -> bool {
        matches!(self.inner, CacheType::Sliding(_))
    }

    /// Return logical state for a sliding cache.
    ///
    /// Global layers return `None` so callers can iterate over all Gemma4
    /// layers without separately checking the layer kind.
    pub fn sliding_state(&self) -> Result<Option<RotatingKVCacheState>> {
        match &self.inner {
            CacheType::Global(_) => Ok(None),
            CacheType::Sliding(c) => Ok(Some(c.state()?)),
        }
    }

    /// Check that a sliding cache is initialized and aligned to `offset`.
    pub fn sliding_offset_matches(&self, offset: i32) -> Result<bool> {
        match &self.inner {
            CacheType::Global(_) => Ok(false),
            CacheType::Sliding(c) => {
                let state = c.state()?;
                Ok(state.initialized && state.offset == offset)
            }
        }
    }

    /// Snapshot a sliding cache's ordered K/V tail.
    ///
    /// Global layers and empty sliding caches return `None`.
    pub fn snapshot_sliding(&self) -> Result<Option<RotatingKVCacheSnapshot>> {
        match &self.inner {
            CacheType::Global(_) => Ok(None),
            CacheType::Sliding(c) => c.snapshot(),
        }
    }

    /// Restore a sliding cache from an ordered K/V tail snapshot.
    ///
    /// This intentionally errors on global layers to prevent accidentally
    /// loading sliding-window state into a full-attention cache.
    pub fn restore_sliding_snapshot(&mut self, snapshot: &RotatingKVCacheSnapshot) -> Result<()> {
        match &mut self.inner {
            CacheType::Global(_) => Err(Error::new(
                Status::InvalidArg,
                "cannot restore a sliding snapshot into a Gemma4 global cache",
            )),
            CacheType::Sliding(c) => {
                c.restore_snapshot(snapshot)?;
                self.stashed_kv = None;
                Ok(())
            }
        }
    }

    /// Get the current cached K/V as (keys, values).
    ///
    /// Returns the valid portion of the cache (sliced to current offset).
    /// For KVCache: returns keys/values sliced to [0..offset].
    /// For RotatingKVCache: returns the current window contents.
    ///
    /// **Note**: For sliding caches after a long prefill this returns the
    /// TRIMMED window, not the full sequence. Use `take_stashed_kv` when you
    /// need the K/V the anchor attention actually used.
    ///
    /// Returns None if the cache is empty.
    pub fn get_cached_kv(&self) -> Option<(MxArray, MxArray)> {
        match &self.inner {
            CacheType::Global(c) => {
                let offset = c.get_offset();
                if offset == 0 {
                    return None;
                }
                let keys = c.keys_ref()?;
                let values = c.values_ref()?;
                // Slice to valid portion [0..offset]
                let keys = keys.slice_axis(2, 0, offset as i64).ok()?;
                let values = values.slice_axis(2, 0, offset as i64).ok()?;
                Some((keys, values))
            }
            CacheType::Sliding(c) => c.fetch_current_kv(),
        }
    }

    /// Update the cache and stash the returned K/V.
    ///
    /// Delegates to the inner cache's `update_and_fetch`, then stashes the
    /// returned K/V pair. Returns the same K/V that was stashed.
    ///
    /// During prefill of a long prompt, the RotatingKVCache returns the FULL
    /// untrimmed sequence for attention while storing only the trimmed window.
    /// The stash captures the full sequence so shared layers get proper context.
    pub fn update_and_fetch_stash(
        &mut self,
        keys: &MxArray,
        values: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        let (k, v) = match &mut self.inner {
            CacheType::Global(kvc) => kvc.update_and_fetch(keys, values)?,
            CacheType::Sliding(rkvc) => {
                let kv = rkvc.update_and_fetch(keys, values)?;
                (kv[0].clone(), kv[1].clone())
            }
        };
        self.stashed_kv = Some((k.clone(), v.clone()));
        Ok((k, v))
    }

    /// Update the cache and return K/V without stashing.
    /// Use this when KV sharing is disabled (num_kv_shared_layers=0).
    pub fn update_and_fetch(
        &mut self,
        keys: &MxArray,
        values: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        match &mut self.inner {
            CacheType::Global(kvc) => kvc.update_and_fetch(keys, values),
            CacheType::Sliding(rkvc) => {
                let kv = rkvc.update_and_fetch(keys, values)?;
                Ok((kv[0].clone(), kv[1].clone()))
            }
        }
    }

    /// Save K/V into the stash (replaces any previous stash).
    pub fn stash_kv(&mut self, keys: MxArray, values: MxArray) {
        self.stashed_kv = Some((keys, values));
    }

    /// Take the stashed K/V, clearing the stash.
    ///
    /// Returns the K/V that was saved by the last `update_and_fetch_stash` or
    /// `stash_kv` call. Returns None if the stash is empty (already taken or
    /// never populated).
    pub fn take_stashed_kv(&mut self) -> Option<(MxArray, MxArray)> {
        self.stashed_kv.take()
    }

    /// Rewind a global cache to `new_len` tokens (passthrough to
    /// `KVCache::trim`; the next append overwrites the trimmed region in
    /// place). Errors on sliding caches — a rotating window cannot be
    /// rewound by moving the offset; use the snapshot/restore rollback path.
    pub(crate) fn trim_global(&mut self, new_len: i32) -> Result<()> {
        match &mut self.inner {
            CacheType::Global(c) => {
                c.trim(new_len);
                self.stashed_kv = None;
                Ok(())
            }
            CacheType::Sliding(_) => Err(Error::new(
                Status::InvalidArg,
                "trim_global cannot rewind a Gemma4 sliding cache; use snapshot/restore rollback",
            )),
        }
    }

    /// The LAST `n` cached entries of a sliding cache in temporal order,
    /// as `(keys, values)` slices of the current window contents.
    ///
    /// After a T-token verify append on a window >= T, the last T entries
    /// are exactly the verify block's K/V. Errors on global caches and when
    /// `n` is zero or exceeds the cached token count.
    pub(crate) fn sliding_tail(&self, n: usize) -> Result<(MxArray, MxArray)> {
        match &self.inner {
            CacheType::Global(_) => Err(Error::new(
                Status::InvalidArg,
                "sliding_tail is only valid on a Gemma4 sliding cache",
            )),
            CacheType::Sliding(c) => {
                let (keys, values) = c.fetch_current_kv().ok_or_else(|| {
                    Error::new(
                        Status::InvalidArg,
                        "sliding_tail requested on an empty Gemma4 sliding cache",
                    )
                })?;
                let cached = keys.shape_at(2)?;
                let n = n as i64;
                if n < 1 || n > cached {
                    return Err(Error::new(
                        Status::InvalidArg,
                        format!("sliding_tail: n={n} out of range for {cached} cached tokens"),
                    ));
                }
                Ok((
                    keys.slice_axis(2, cached - n, cached)?,
                    values.slice_axis(2, cached - n, cached)?,
                ))
            }
        }
    }

    /// Clear a sliding cache back to empty. Errors on global caches.
    fn reset_sliding(&mut self) -> Result<()> {
        match &mut self.inner {
            CacheType::Global(_) => Err(Error::new(
                Status::InvalidArg,
                "reset_sliding is only valid on a Gemma4 sliding cache",
            )),
            CacheType::Sliding(c) => {
                c.reset();
                self.stashed_kv = None;
                Ok(())
            }
        }
    }

    /// Collect references to the raw internal K/V arrays for eval between
    /// chunked prefill steps. Matches Qwen3.5's `collect_arrays` pattern.
    pub fn collect_cache_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        match &self.inner {
            CacheType::Global(c) => {
                if let Some(k) = c.keys_ref() {
                    out.push(k);
                }
                if let Some(v) = c.values_ref() {
                    out.push(v);
                }
            }
            CacheType::Sliding(c) => {
                if let Some(k) = c.keys_ref() {
                    out.push(k);
                }
                if let Some(v) = c.values_ref() {
                    out.push(v);
                }
            }
        }
    }
}

/// Cache state captured before a DSpark verify forward, used by
/// [`commit_after_verify`] to roll every cache back to the kept prefix when
/// the target accepts only part of the verified block.
pub(crate) struct Gemma4VerifyRollback {
    /// Per-cache sliding snapshots, index-aligned with the caches slice.
    /// `None` for global layers and for sliding caches that were empty.
    snapshots: Vec<Option<RotatingKVCacheSnapshot>>,
    /// Per-cache offsets before the verify forward.
    start_offsets: Vec<i32>,
    /// Caller-declared KV-shared slots, index-aligned with the caches slice.
    /// A shared layer reads its anchor's cache; its own vec entry is never
    /// written by a forward pass and must not move across a verify.
    shared_slots: Vec<bool>,
    /// Tokens the verify forward appends to every ACTIVE (non-shared) cache.
    total_written: usize,
}

/// Snapshot every cache before a verify forward that will append
/// `total_to_write` tokens.
///
/// `shared_slots` must mark exactly the cache vec entries that belong to
/// KV-shared layers (`Gemma4Config::is_kv_shared_layer` per index — see
/// `dspark_shared_slot_mask`). Declaring them explicitly lets commit enforce
/// the exact-advance invariant on every active cache instead of inferring
/// activity from offsets.
///
/// Global caches need no snapshot (rollback is a `trim`); sliding caches
/// snapshot their ordered window tail. Errors when a sliding window is
/// smaller than the verify block, because the block tail could then not be
/// recovered for a partial-keep rollback.
pub(crate) fn snapshot_before_verify(
    caches: &[Gemma4LayerCache],
    total_to_write: usize,
    shared_slots: &[bool],
) -> Result<Gemma4VerifyRollback> {
    if total_to_write == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "snapshot_before_verify: total_to_write must be at least 1",
        ));
    }
    if shared_slots.len() != caches.len() {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "snapshot_before_verify: {} caches but shared_slots covers {}",
                caches.len(),
                shared_slots.len()
            ),
        ));
    }
    let mut snapshots = Vec::with_capacity(caches.len());
    let mut start_offsets = Vec::with_capacity(caches.len());
    for (idx, cache) in caches.iter().enumerate() {
        if let Some(state) = cache.sliding_state()?
            && (state.window_size as i64) < total_to_write as i64
        {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "snapshot_before_verify: cache {idx} sliding window {} is smaller than the {total_to_write}-token verify block; its tail could not be recovered for rollback",
                    state.window_size
                ),
            ));
        }
        start_offsets.push(cache.get_offset());
        snapshots.push(cache.snapshot_sliding()?);
    }
    Ok(Gemma4VerifyRollback {
        snapshots,
        start_offsets,
        shared_slots: shared_slots.to_vec(),
        total_written: total_to_write,
    })
}

/// Commit the first `keep` tokens of a verify block and roll back the rest.
///
/// Preconditions: `rb` was taken right before the verify forward, and the
/// verify forward appended `rb.total_written` tokens to every ACTIVE cache.
/// EVERY commit (including full keep) validates the whole vec against the
/// declared `shared_slots`: an active cache must sit at exactly
/// `start + total_written` and a shared slot must not have moved, else a
/// hard error — a routing bug that leaves an owner cache unwritten can
/// never be silently committed. Validation runs over the full vec before
/// any cache is mutated, so a failed commit leaves all caches untouched.
///
/// * `keep == total_written`: validated no-op — the appended block is kept.
/// * `keep < total_written`: global caches trim to `start + keep`; sliding
///   caches slice the block tail, restore the pre-verify snapshot, then
///   re-append the first `keep` tail rows through the normal update path,
///   leaving state identical to a cache that only ever saw the kept prefix.
///   `keep == 0` discards the block entirely. Shared slots are untouched,
///   so each physical cache is touched exactly once.
pub(crate) fn commit_after_verify(
    caches: &mut [Gemma4LayerCache],
    rb: &Gemma4VerifyRollback,
    keep: usize,
) -> Result<()> {
    if caches.len() != rb.snapshots.len() {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "commit_after_verify: {} caches but rollback snapshot covers {}",
                caches.len(),
                rb.snapshots.len()
            ),
        ));
    }
    if keep > rb.total_written {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "commit_after_verify: keep={keep} exceeds the {}-token verify block",
                rb.total_written
            ),
        ));
    }

    for (idx, cache) in caches.iter().enumerate() {
        let start = rb.start_offsets[idx];
        let current = cache.get_offset();
        if rb.shared_slots[idx] {
            if current != start {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!(
                        "commit_after_verify: cache {idx} is a KV-shared slot but moved from offset {start} to {current}; shared slots are never written by a verify forward"
                    ),
                ));
            }
        } else {
            let expected = start + rb.total_written as i32;
            if current != expected {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!(
                        "commit_after_verify: active cache {idx} is at offset {current}, expected {expected} (start {start} + {}-token verify block)",
                        rb.total_written
                    ),
                ));
            }
        }
    }

    if keep == rb.total_written {
        return Ok(());
    }

    for (idx, (cache, (snapshot, &start))) in caches
        .iter_mut()
        .zip(rb.snapshots.iter().zip(rb.start_offsets.iter()))
        .enumerate()
    {
        if rb.shared_slots[idx] {
            continue;
        }

        if cache.is_sliding() {
            let (tail_keys, tail_values) = cache.sliding_tail(rb.total_written)?;
            match snapshot {
                Some(snapshot) => cache.restore_sliding_snapshot(snapshot)?,
                // The sliding cache was empty before the verify forward;
                // rolling back means clearing it entirely.
                None => cache.reset_sliding()?,
            }
            if keep > 0 {
                let kept_keys = tail_keys.slice_axis(2, 0, keep as i64)?;
                let kept_values = tail_values.slice_axis(2, 0, keep as i64)?;
                cache.update_and_fetch(&kept_keys, &kept_values)?;
            }
        } else {
            cache.trim_global(start + keep as i32)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_float_data(arr: &MxArray, expected: &[f32]) {
        arr.eval();
        let data = arr.to_float32().unwrap().to_vec();
        assert_eq!(data, expected);
    }

    /// Build a K/V pair shaped [1, 1, n, 1]: keys carry `values` verbatim,
    /// values carry `values * 10` so K and V mixups are visible.
    fn kv_pair(values: &[f32]) -> (MxArray, MxArray) {
        let n = values.len() as i64;
        let keys = MxArray::from_float32(values, &[1, 1, n, 1]).unwrap();
        let scaled: Vec<f32> = values.iter().map(|v| v * 10.0).collect();
        let vals = MxArray::from_float32(&scaled, &[1, 1, n, 1]).unwrap();
        (keys, vals)
    }

    /// Apply a sequence of appends (each inner vec is one `update_and_fetch`
    /// call; single-element vecs take the single-token path).
    fn apply_appends(cache: &mut Gemma4LayerCache, appends: &[Vec<f32>]) {
        for chunk in appends {
            let (k, v) = kv_pair(chunk);
            cache.update_and_fetch(&k, &v).unwrap();
        }
    }

    fn assert_same_f32(a: &MxArray, b: &MxArray, ctx: &str) {
        a.eval();
        b.eval();
        assert_eq!(
            a.shape().unwrap().to_vec(),
            b.shape().unwrap().to_vec(),
            "{ctx}: shape"
        );
        let a_bits: Vec<u32> = a
            .to_float32()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let b_bits: Vec<u32> = b
            .to_float32()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(a_bits, b_bits, "{ctx}: contents");
    }

    /// Compare a rolled-back cache against a reference cache that only ever
    /// saw prefill + kept prefix: logical state, temporal contents, and the
    /// behavior of one subsequent single-token append.
    ///
    /// `compare_idx` gates comparing the raw write index and raw attention
    /// views: a rollback restores storage in temporal order, so when the
    /// reference's backing buffer is ROTATED (idx mid-window) the physical
    /// layout legitimately differs while offset, temporal contents, and all
    /// subsequent appends remain identical.
    fn assert_rollback_matches_reference(
        live: &mut Gemma4LayerCache,
        reference: &mut Gemma4LayerCache,
        compare_idx: bool,
        ctx: &str,
    ) {
        let ls = live.sliding_state().unwrap().unwrap();
        let rs = reference.sliding_state().unwrap().unwrap();
        if compare_idx {
            assert_eq!(ls, rs, "{ctx}: state");
        } else {
            assert_eq!(ls.offset, rs.offset, "{ctx}: offset");
            assert_eq!(ls.window_size, rs.window_size, "{ctx}: window");
            assert_eq!(ls.keep, rs.keep, "{ctx}: keep param");
            assert_eq!(ls.cached_tokens, rs.cached_tokens, "{ctx}: cached_tokens");
            assert_eq!(ls.initialized, rs.initialized, "{ctx}: initialized");
        }

        match (live.get_cached_kv(), reference.get_cached_kv()) {
            (Some((lk, lv)), Some((rk, rv))) => {
                assert_same_f32(&lk, &rk, &format!("{ctx}: temporal keys"));
                assert_same_f32(&lv, &rv, &format!("{ctx}: temporal values"));
            }
            (None, None) => {}
            (l, r) => panic!(
                "{ctx}: cache emptiness mismatch: live={} reference={}",
                l.is_some(),
                r.is_some()
            ),
        }

        let (ak, av) = kv_pair(&[201.0]);
        let (live_k, live_v) = live.update_and_fetch(&ak, &av).unwrap();
        let (ref_k, ref_v) = reference.update_and_fetch(&ak, &av).unwrap();
        if compare_idx {
            assert_same_f32(
                &live_k,
                &ref_k,
                &format!("{ctx}: post-append attention keys"),
            );
            assert_same_f32(
                &live_v,
                &ref_v,
                &format!("{ctx}: post-append attention values"),
            );
        }
        assert_eq!(
            live.get_offset(),
            reference.get_offset(),
            "{ctx}: post-append offset"
        );
        let (lk2, lv2) = live.get_cached_kv().unwrap();
        let (rk2, rv2) = reference.get_cached_kv().unwrap();
        assert_same_f32(&lk2, &rk2, &format!("{ctx}: post-append temporal keys"));
        assert_same_f32(&lv2, &rv2, &format!("{ctx}: post-append temporal values"));
    }

    /// The critical rollback invariant: snapshot → multi-token verify append
    /// → rollback-to-keep leaves the sliding cache byte-identical to a cache
    /// that only ever saw prefill + kept prefix, for every keep in 0..=T,
    /// pre-wrap, post-wrap, rotated-storage, and empty-prefill.
    #[test]
    fn sliding_reappend_matches_reference() {
        let window = 8;
        let block: Vec<f32> = (101..=108).map(|v| v as f32).collect();

        let rotated_prefill: Vec<Vec<f32>> = {
            let mut appends = vec![(1..=5).map(|v| v as f32).collect::<Vec<f32>>()];
            for v in 6..=12 {
                appends.push(vec![v as f32]);
            }
            appends
        };
        let scenarios: Vec<(&str, Vec<Vec<f32>>, bool)> = vec![
            ("pre_wrap", vec![(1..=5).map(|v| v as f32).collect()], true),
            (
                "post_wrap",
                vec![(1..=40).map(|v| v as f32).collect()],
                true,
            ),
            ("rotated_idx", rotated_prefill, false),
            ("empty_prefill", vec![], true),
        ];

        for (name, prefill, compare_idx) in &scenarios {
            for keep in 0..=block.len() {
                let ctx = format!("{name} keep={keep}");

                let mut live = [Gemma4LayerCache::new_sliding(window)];
                apply_appends(&mut live[0], prefill);
                let rollback = snapshot_before_verify(&live, block.len(), &[false]).unwrap();
                let (bk, bv) = kv_pair(&block);
                live[0].update_and_fetch(&bk, &bv).unwrap();
                commit_after_verify(&mut live, &rollback, keep).unwrap();

                let mut reference = Gemma4LayerCache::new_sliding(window);
                apply_appends(&mut reference, prefill);
                if keep > 0 {
                    let (kk, kv) = kv_pair(&block[..keep]);
                    reference.update_and_fetch(&kk, &kv).unwrap();
                }

                assert_rollback_matches_reference(&mut live[0], &mut reference, *compare_idx, &ctx);
            }
        }
    }

    #[test]
    fn trim_global_trims_global_and_errors_on_sliding() {
        let mut global = Gemma4LayerCache::new_global();
        let (k, v) = kv_pair(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        global.update_and_fetch(&k, &v).unwrap();
        assert_eq!(global.get_offset(), 8);

        global.trim_global(5).unwrap();
        assert_eq!(global.get_offset(), 5);
        let (cached_k, cached_v) = global.get_cached_kv().unwrap();
        assert_float_data(&cached_k, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_float_data(&cached_v, &[10.0, 20.0, 30.0, 40.0, 50.0]);

        // The next append overwrites the trimmed region in place.
        let (nk, nv) = kv_pair(&[99.0]);
        global.update_and_fetch(&nk, &nv).unwrap();
        assert_eq!(global.get_offset(), 6);
        let (cached_k, cached_v) = global.get_cached_kv().unwrap();
        assert_float_data(&cached_k, &[1.0, 2.0, 3.0, 4.0, 5.0, 99.0]);
        assert_float_data(&cached_v, &[10.0, 20.0, 30.0, 40.0, 50.0, 990.0]);

        let mut sliding = Gemma4LayerCache::new_sliding(4);
        let (sk, sv) = kv_pair(&[1.0, 2.0]);
        sliding.update_and_fetch(&sk, &sv).unwrap();
        assert!(sliding.trim_global(1).is_err());
        assert_eq!(sliding.get_offset(), 2, "failed trim must not mutate");
    }

    #[test]
    fn sliding_tail_returns_block_rows_and_errors_on_global() {
        let mut sliding = Gemma4LayerCache::new_sliding(8);
        let (pk, pv) = kv_pair(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        sliding.update_and_fetch(&pk, &pv).unwrap();
        let (bk, bv) = kv_pair(&[101.0, 102.0, 103.0, 104.0]);
        sliding.update_and_fetch(&bk, &bv).unwrap();

        let (tail_k, tail_v) = sliding.sliding_tail(4).unwrap();
        assert_float_data(&tail_k, &[101.0, 102.0, 103.0, 104.0]);
        assert_float_data(&tail_v, &[1010.0, 1020.0, 1030.0, 1040.0]);

        // n beyond the cached token count and n == 0 are rejected.
        assert!(sliding.sliding_tail(9).is_err());
        assert!(sliding.sliding_tail(0).is_err());

        let mut global = Gemma4LayerCache::new_global();
        let (gk, gv) = kv_pair(&[1.0, 2.0]);
        global.update_and_fetch(&gk, &gv).unwrap();
        assert!(global.sliding_tail(1).is_err());

        let empty = Gemma4LayerCache::new_sliding(8);
        assert!(empty.sliding_tail(1).is_err());
    }

    /// Mixed vec of global + sliding caches, including entries declared as
    /// KV-shared slots (their vec entry is never written by a forward).
    const MIXED_ACTIVE: [usize; 4] = [0, 1, 2, 3];
    const MIXED_SHARED: [usize; 2] = [4, 5];
    const MIXED_MASK: [bool; 6] = [false, false, false, false, true, true];

    fn build_mixed_caches() -> Vec<Gemma4LayerCache> {
        let mut caches = vec![
            Gemma4LayerCache::new_global(),
            Gemma4LayerCache::new_sliding(8),
            Gemma4LayerCache::new_global(),
            Gemma4LayerCache::new_sliding(8),
            // KV-shared entries: allocated but never written.
            Gemma4LayerCache::new_sliding(8),
            Gemma4LayerCache::new_global(),
        ];
        for &i in &MIXED_ACTIVE {
            let (k, v) = kv_pair(&[1.0, 2.0, 3.0, 4.0, 5.0]);
            caches[i].update_and_fetch(&k, &v).unwrap();
        }
        caches
    }

    fn append_block_to(caches: &mut [Gemma4LayerCache], indices: &[usize], block: &[f32]) {
        for &i in indices {
            let (k, v) = kv_pair(block);
            caches[i].update_and_fetch(&k, &v).unwrap();
        }
    }

    #[test]
    fn commit_after_verify_mixed_caches() {
        let block = [101.0f32, 102.0, 103.0];

        // Full keep: validated no-op, all active offsets stay at n + T.
        let mut caches = build_mixed_caches();
        let rollback = snapshot_before_verify(&caches, block.len(), &MIXED_MASK).unwrap();
        append_block_to(&mut caches, &MIXED_ACTIVE, &block);
        commit_after_verify(&mut caches, &rollback, block.len()).unwrap();
        for &i in &MIXED_ACTIVE {
            assert_eq!(caches[i].get_offset(), 8, "cache {i} full-keep offset");
        }
        for &i in &MIXED_SHARED {
            assert_eq!(caches[i].get_offset(), 0, "shared slot {i} must stay empty");
        }

        // Partial keep: every active cache lands at n + keep with exactly the
        // kept prefix appended; shared slots untouched.
        let mut caches = build_mixed_caches();
        let rollback = snapshot_before_verify(&caches, block.len(), &MIXED_MASK).unwrap();
        append_block_to(&mut caches, &MIXED_ACTIVE, &block);
        commit_after_verify(&mut caches, &rollback, 1).unwrap();
        for &i in &MIXED_ACTIVE {
            assert_eq!(caches[i].get_offset(), 6, "cache {i} partial-keep offset");
            let (k, v) = caches[i].get_cached_kv().unwrap();
            assert_float_data(&k, &[1.0, 2.0, 3.0, 4.0, 5.0, 101.0]);
            assert_float_data(&v, &[10.0, 20.0, 30.0, 40.0, 50.0, 1010.0]);
        }
        for &i in &MIXED_SHARED {
            assert_eq!(caches[i].get_offset(), 0, "shared slot {i} must stay empty");
        }

        // keep == 0: the verify block is fully discarded.
        let mut caches = build_mixed_caches();
        let rollback = snapshot_before_verify(&caches, block.len(), &MIXED_MASK).unwrap();
        append_block_to(&mut caches, &MIXED_ACTIVE, &block);
        commit_after_verify(&mut caches, &rollback, 0).unwrap();
        for &i in &MIXED_ACTIVE {
            assert_eq!(caches[i].get_offset(), 5, "cache {i} keep-0 offset");
            let (k, _) = caches[i].get_cached_kv().unwrap();
            assert_float_data(&k, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        }

        // keep > total_written is rejected.
        let mut caches = build_mixed_caches();
        let rollback = snapshot_before_verify(&caches, block.len(), &MIXED_MASK).unwrap();
        append_block_to(&mut caches, &MIXED_ACTIVE, &block);
        assert!(commit_after_verify(&mut caches, &rollback, 4).is_err());

        // A cache that advanced by anything other than total_written is a
        // contract violation, not something to silently "fix".
        let mut caches = build_mixed_caches();
        let rollback = snapshot_before_verify(&caches, block.len(), &MIXED_MASK).unwrap();
        append_block_to(&mut caches, &MIXED_ACTIVE, &block[..2]);
        assert!(commit_after_verify(&mut caches, &rollback, 1).is_err());

        // Cache-count mismatch between snapshot and commit is rejected.
        let mut caches = build_mixed_caches();
        let rollback = snapshot_before_verify(&caches, block.len(), &MIXED_MASK).unwrap();
        append_block_to(&mut caches, &MIXED_ACTIVE, &block);
        assert!(commit_after_verify(&mut caches[..5], &rollback, 1).is_err());
    }

    /// An ACTIVE cache the verify forward failed to write is a hard error on
    /// partial keep — and commit must not have mutated ANY cache (validation
    /// runs over the whole vec before rollback starts).
    #[test]
    fn commit_rejects_unmoved_active_cache_on_partial_keep() {
        let block = [101.0f32, 102.0, 103.0];
        let mut caches = build_mixed_caches();
        let rollback = snapshot_before_verify(&caches, block.len(), &MIXED_MASK).unwrap();
        // Cache 2 (active, global) is skipped by the "verify forward".
        append_block_to(&mut caches, &[0, 1, 3], &block);

        let err = commit_after_verify(&mut caches, &rollback, 1)
            .expect_err("unmoved active cache must fail partial-keep commit");
        assert!(
            err.reason.contains("active cache 2"),
            "error must name the unmoved active cache, got: {}",
            err.reason
        );
        // No cache was rolled back by the failed commit.
        for &i in &[0usize, 1, 3] {
            assert_eq!(caches[i].get_offset(), 8, "cache {i} must be untouched");
        }
        assert_eq!(caches[2].get_offset(), 5, "cache 2 must be untouched");
    }

    /// Full keep is no longer a blind no-op: an active cache at the wrong
    /// offset fails commit even when keep == total_written.
    #[test]
    fn commit_validates_active_offsets_on_full_keep() {
        let block = [101.0f32, 102.0, 103.0];
        let mut caches = build_mixed_caches();
        let rollback = snapshot_before_verify(&caches, block.len(), &MIXED_MASK).unwrap();
        // Cache 1 (active, sliding) is skipped by the "verify forward".
        append_block_to(&mut caches, &[0, 2, 3], &block);

        let err = commit_after_verify(&mut caches, &rollback, block.len())
            .expect_err("unmoved active cache must fail full-keep commit");
        assert!(
            err.reason.contains("active cache 1"),
            "error must name the unmoved active cache, got: {}",
            err.reason
        );
    }

    /// A slot declared KV-shared that MOVED during verify is a routing bug,
    /// rejected on both full and partial keep.
    #[test]
    fn commit_rejects_moved_shared_slot() {
        let block = [101.0f32, 102.0, 103.0];
        for keep in [block.len(), 1] {
            let mut caches = build_mixed_caches();
            let rollback = snapshot_before_verify(&caches, block.len(), &MIXED_MASK).unwrap();
            // Shared slot 4 is unexpectedly written too.
            append_block_to(&mut caches, &[0, 1, 2, 3, 4], &block);

            let err = commit_after_verify(&mut caches, &rollback, keep)
                .expect_err("moved shared slot must fail commit");
            assert!(
                err.reason.contains("cache 4") && err.reason.contains("KV-shared"),
                "error must name the moved shared slot, got: {}",
                err.reason
            );
        }
    }

    #[test]
    fn snapshot_before_verify_validates_window_block_and_mask() {
        let caches = vec![Gemma4LayerCache::new_sliding(2)];
        assert!(
            snapshot_before_verify(&caches, 3, &[false]).is_err(),
            "window smaller than the verify block cannot be rolled back"
        );
        assert!(snapshot_before_verify(&caches, 0, &[false]).is_err());
        assert!(
            snapshot_before_verify(&caches, 2, &[false, true]).is_err(),
            "shared_slots length must match the caches vec"
        );
        assert!(snapshot_before_verify(&caches, 2, &[false]).is_ok());
    }

    /// A one-token verify block (T = 1 + 0 drafts) takes the rotating
    /// cache's IN-PLACE single-token update path, which overwrites the
    /// backing array's descriptor rather than rebinding it. The pre-verify
    /// snapshot must not alias that array, or the rollback would restore
    /// post-verify data. Regression test for the `temporal_order` clone
    /// branch (`idx == cache_len`) sharing the live handle.
    #[test]
    fn sliding_rollback_survives_single_token_verify_append() {
        // Post-wrap single-concat prefill leaves idx == cached length ==
        // window, exactly the state where the snapshot would alias the
        // live storage and the next single-token append writes in place.
        let prefill: Vec<f32> = (1..=40).map(|v| v as f32).collect();

        let mut live = [Gemma4LayerCache::new_sliding(8)];
        apply_appends(&mut live[0], std::slice::from_ref(&prefill));
        let rollback = snapshot_before_verify(&live, 1, &[false]).unwrap();
        let (bk, bv) = kv_pair(&[999.0]);
        live[0].update_and_fetch(&bk, &bv).unwrap();
        commit_after_verify(&mut live, &rollback, 0).unwrap();

        let mut reference = Gemma4LayerCache::new_sliding(8);
        apply_appends(&mut reference, &[prefill]);

        assert_rollback_matches_reference(
            &mut live[0],
            &mut reference,
            false,
            "single-token verify keep=0",
        );
    }

    #[test]
    fn test_sliding_cache_stash_preserves_full_prefill() {
        // Create a sliding cache with small window (4 tokens)
        let mut cache = Gemma4LayerCache::new_sliding(4);

        // Simulate a long prefill with 8 tokens (exceeds window of 4)
        // K/V shape: [1, 1, 8, 16] = [B, H, T, D]
        let keys = MxArray::ones(&[1, 1, 8, 16], None).unwrap();
        let values = MxArray::ones(&[1, 1, 8, 16], None).unwrap();

        // update_and_fetch_stash should return full 8-token sequence
        let (ret_k, ret_v) = cache.update_and_fetch_stash(&keys, &values).unwrap();
        assert_eq!(
            ret_k.shape_at(2).unwrap(),
            8,
            "returned K should have full 8 tokens"
        );
        assert_eq!(
            ret_v.shape_at(2).unwrap(),
            8,
            "returned V should have full 8 tokens"
        );

        // The stash should also have full 8 tokens
        let (stash_k, stash_v) = cache.take_stashed_kv().unwrap();
        assert_eq!(
            stash_k.shape_at(2).unwrap(),
            8,
            "stashed K should have full 8 tokens"
        );
        assert_eq!(
            stash_v.shape_at(2).unwrap(),
            8,
            "stashed V should have full 8 tokens"
        );

        // But get_cached_kv (reading from stored cache) should only have window-sized cache
        let (cached_k, _cached_v) = cache.get_cached_kv().unwrap();
        assert!(
            cached_k.shape_at(2).unwrap() <= 4,
            "stored cache should be trimmed to window"
        );

        // Stash should be consumed (take clears it)
        assert!(
            cache.take_stashed_kv().is_none(),
            "stash should be empty after take"
        );
    }

    #[test]
    fn test_global_cache_stash_matches_stored() {
        // Global cache stores everything, so stash == stored
        let mut cache = Gemma4LayerCache::new_global();

        let keys = MxArray::ones(&[1, 1, 8, 16], None).unwrap();
        let values = MxArray::ones(&[1, 1, 8, 16], None).unwrap();

        let (ret_k, _) = cache.update_and_fetch_stash(&keys, &values).unwrap();
        assert_eq!(ret_k.shape_at(2).unwrap(), 8);

        let (stash_k, _) = cache.take_stashed_kv().unwrap();
        assert_eq!(stash_k.shape_at(2).unwrap(), 8);

        let (cached_k, _) = cache.get_cached_kv().unwrap();
        assert_eq!(
            cached_k.shape_at(2).unwrap(),
            8,
            "global cache stores everything"
        );
    }

    #[test]
    fn test_sliding_snapshot_restore_after_wrap() {
        let mut source = Gemma4LayerCache::new_sliding(4);

        let keys1 = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]).unwrap();
        let values1 = MxArray::from_float32(&[10.0, 20.0, 30.0, 40.0], &[1, 1, 4, 1]).unwrap();
        source.update_and_fetch(&keys1, &values1).unwrap();

        let keys2 = MxArray::from_float32(&[5.0], &[1, 1, 1, 1]).unwrap();
        let values2 = MxArray::from_float32(&[50.0], &[1, 1, 1, 1]).unwrap();
        source.update_and_fetch(&keys2, &values2).unwrap();

        let keys3 = MxArray::from_float32(&[6.0], &[1, 1, 1, 1]).unwrap();
        let values3 = MxArray::from_float32(&[60.0], &[1, 1, 1, 1]).unwrap();
        source.update_and_fetch(&keys3, &values3).unwrap();

        assert!(source.is_sliding());
        assert!(source.sliding_offset_matches(6).unwrap());
        let source_state = source.sliding_state().unwrap().unwrap();
        assert_eq!(source_state.offset, 6);
        assert_eq!(source_state.cached_tokens, 4);

        let snapshot = source.snapshot_sliding().unwrap().unwrap();
        let mut restored = Gemma4LayerCache::new_sliding(4);
        restored.restore_sliding_snapshot(&snapshot).unwrap();
        assert!(restored.sliding_offset_matches(6).unwrap());

        let (restored_keys, restored_values) = restored.get_cached_kv().unwrap();
        assert_float_data(&restored_keys, &[3.0, 4.0, 5.0, 6.0]);
        assert_float_data(&restored_values, &[30.0, 40.0, 50.0, 60.0]);

        let append_keys = MxArray::from_float32(&[7.0, 8.0], &[1, 1, 2, 1]).unwrap();
        let append_values = MxArray::from_float32(&[70.0, 80.0], &[1, 1, 2, 1]).unwrap();
        let (attention_keys, attention_values) = restored
            .update_and_fetch(&append_keys, &append_values)
            .unwrap();
        assert_float_data(&attention_keys, &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_float_data(&attention_values, &[30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);

        let (restored_tail, _) = restored.get_cached_kv().unwrap();
        assert_float_data(&restored_tail, &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_sliding_snapshot_not_restored_into_global_cache() {
        let mut sliding = Gemma4LayerCache::new_sliding(4);
        let keys = MxArray::ones(&[1, 1, 4, 1], None).unwrap();
        let values = MxArray::ones(&[1, 1, 4, 1], None).unwrap();
        sliding.update_and_fetch(&keys, &values).unwrap();
        let snapshot = sliding.snapshot_sliding().unwrap().unwrap();

        let mut global = Gemma4LayerCache::new_global();
        assert!(!global.is_sliding());
        assert!(global.snapshot_sliding().unwrap().is_none());
        assert!(global.restore_sliding_snapshot(&snapshot).is_err());
    }
}
