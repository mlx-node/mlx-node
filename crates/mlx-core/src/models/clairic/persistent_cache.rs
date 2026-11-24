//! Content-addressed, bounded-LRU persistent caches owned by `Qwen35Inner`
//! across Clairic extraction requests.
//!
//! Two independent tiers are maintained per loaded model instance:
//! - Vision encoder outputs (`media_features`), keyed by `image_digest`
//!   (SHA-256 of image dimensions + RGB bytes), bounded to 16 entries / 128 MiB.
//! - Shared `B=1` prefix layer snapshots (`prefix_snapshots`), keyed strictly by
//!   `(media_content_digest, prefix_tokens_hash)`, bounded to 4 entries / 512 MiB.
//!
//! Set `MLX_CLAIRIC_PERSIST_CACHE=0` to disable cross-request persistence.

use std::collections::VecDeque;
use std::sync::OnceLock;

use napi::bindgen_prelude::Result;

use crate::array::MxArray;
use crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache;

/// Maximum cached vision feature entries across Clairic requests.
const MEDIA_MAX_ENTRIES: usize = 16;
/// Maximum byte budget for cached vision features (128 MiB).
const MEDIA_MAX_BYTES: usize = 128 << 20;

/// Maximum cached `B=1` prefix KV/GDN snapshots across Clairic requests.
const PREFIX_MAX_ENTRIES: usize = 4;
/// Maximum byte budget for cached `B=1` prefix snapshots (512 MiB).
const PREFIX_MAX_BYTES: usize = 512 << 20;

#[allow(dead_code)]
static PERSIST_CACHE_ENABLED: OnceLock<bool> = OnceLock::new();

#[allow(dead_code)]
fn parse_persist_cache_flag(value: Option<&str>) -> bool {
    match value {
        Some(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        }
        None => true,
    }
}

/// Cached `B=1` layer states after running a shared media or text prefix.
#[allow(dead_code)]
pub(crate) struct PrefixSnapshot {
    pub(crate) caches: Vec<Qwen3_5LayerCache>,
    pub(crate) cache_len: i32,
    pub(crate) next_position: i32,
}

struct LruEntry<K, V> {
    #[allow(dead_code)]
    key: K,
    #[allow(dead_code)]
    value: V,
    #[allow(dead_code)]
    bytes: usize,
}

/// Small bounded LRU cache enforcing both entry-count and byte-budget limits.
///
/// Oldest entries live at the front of `entries`; most-recently-used entries
/// live at the back.
struct BoundedLru<K, V> {
    #[allow(dead_code)]
    max_entries: usize,
    #[allow(dead_code)]
    max_bytes: usize,
    entries: VecDeque<LruEntry<K, V>>,
    total_bytes: usize,
}

impl<K, V> BoundedLru<K, V> {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            max_entries,
            max_bytes,
            entries: VecDeque::new(),
            total_bytes: 0,
        }
    }

    #[allow(dead_code)]
    fn get_by(&mut self, matches: impl Fn(&K) -> bool) -> Option<&V> {
        let pos = self.entries.iter().position(|entry| matches(&entry.key))?;
        if pos + 1 != self.entries.len()
            && let Some(entry) = self.entries.remove(pos)
        {
            self.entries.push_back(entry);
        }
        self.entries.back().map(|entry| &entry.value)
    }

    #[allow(dead_code)]
    fn insert(&mut self, key: K, value: V, bytes: usize) -> Option<V>
    where
        K: PartialEq,
    {
        if self.max_entries == 0 || bytes > self.max_bytes {
            return Some(value);
        }
        if let Some(pos) = self.entries.iter().position(|entry| entry.key == key)
            && let Some(old) = self.entries.remove(pos)
        {
            self.total_bytes = self.total_bytes.saturating_sub(old.bytes);
        }
        while (self.entries.len() >= self.max_entries
            || bytes > self.max_bytes.saturating_sub(self.total_bytes))
            && let Some(oldest) = self.entries.pop_front()
        {
            self.total_bytes = self.total_bytes.saturating_sub(oldest.bytes);
        }
        self.total_bytes = self.total_bytes.saturating_add(bytes);
        self.entries.push_back(LruEntry { key, value, bytes });
        None
    }

    #[allow(dead_code)]
    fn evict_oldest(&mut self) -> bool {
        if let Some(oldest) = self.entries.pop_front() {
            self.total_bytes = self.total_bytes.saturating_sub(oldest.bytes);
            true
        } else {
            false
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.total_bytes = 0;
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    fn total_bytes(&self) -> usize {
        self.total_bytes
    }
}

fn retained_array_bytes(arr: &MxArray) -> usize {
    arr.nbytes().max(arr.backing_buffer_bytes())
}

#[allow(dead_code)]
fn estimate_prefix_bytes(snapshot: &PrefixSnapshot) -> usize {
    let mut arrays: Vec<&MxArray> = Vec::with_capacity(snapshot.caches.len().saturating_mul(2));
    for layer in &snapshot.caches {
        layer.collect_arrays(&mut arrays);
    }
    arrays
        .into_iter()
        .fold(0usize, |acc, arr| acc.saturating_add(arr.nbytes()))
}

fn estimate_retained_prefix_bytes(snapshot: &PrefixSnapshot) -> usize {
    let mut arrays: Vec<&MxArray> = Vec::with_capacity(snapshot.caches.len().saturating_mul(2));
    for layer in &snapshot.caches {
        layer.collect_arrays(&mut arrays);
    }
    arrays.into_iter().fold(0usize, |acc, arr| {
        acc.saturating_add(retained_array_bytes(arr))
    })
}

enum DetachTarget {
    Linear { layer: usize, slot: usize },
    FullKeys { layer: usize },
    FullValues { layer: usize },
}

fn detach_array_if_view(arr: &MxArray, force_copy: bool) -> Result<Option<MxArray>> {
    let has_storage = arr.nbytes() > 0 || arr.backing_buffer_bytes() > 0;
    if (force_copy && has_storage) || arr.is_oversized_view() {
        Ok(Some(arr.deep_copy()?))
    } else {
        Ok(None)
    }
}

/// Materialize independent tightly-sized copies for any array in `snapshot`
/// that may pin a larger backing allocation (such as GDN `conv_state` split
/// views over `[1, T + 3, conv_dim]` `conv_input` buffers or zero-offset prefix
/// slices), then evaluate all detached buffers together before swapping them
/// into `snapshot` so parent allocations are released immediately and
/// `snapshot` remains untouched on error.
fn detach_snapshot_views(snapshot: &mut PrefixSnapshot) -> Result<()> {
    // Ensure any lazy arrays are materialized before inspecting backing buffers.
    // In production `compute_prefix_snapshot` already runs `eval_caches_only()`,
    // so `unready` is empty.
    {
        let mut arrays: Vec<&MxArray> = Vec::with_capacity(snapshot.caches.len().saturating_mul(2));
        for layer in &snapshot.caches {
            layer.collect_arrays(&mut arrays);
        }
        let unready: Vec<&MxArray> = arrays
            .into_iter()
            .filter(|arr| !arr.is_available())
            .collect();
        if !unready.is_empty() {
            MxArray::eval_arrays_with_context(
                &unready,
                "clairic_persistent_cache::detach_snapshot_views::eval",
            )?;
        }
    }

    let mut pending: Vec<(DetachTarget, MxArray)> = Vec::new();
    for (layer_idx, layer) in snapshot.caches.iter().enumerate() {
        match layer {
            Qwen3_5LayerCache::Linear(ac) => {
                for slot in 0..ac.len() {
                    // Slot 0 (`conv_state`) is always tiny (`[1, 3, conv_dim]`) and
                    // is typically a split view over a huge `conv_input` allocation
                    // on CUDA / non-fused Metal.
                    if let Some(arr) = ac.get(slot)
                        && let Some(copied) = detach_array_if_view(arr, slot == 0)?
                    {
                        pending.push((
                            DetachTarget::Linear {
                                layer: layer_idx,
                                slot,
                            },
                            copied,
                        ));
                    }
                }
            }
            Qwen3_5LayerCache::FullAttention(kv) => {
                if let Some(keys) = kv.keys_ref()
                    && let Some(copied) = detach_array_if_view(keys, false)?
                {
                    pending.push((DetachTarget::FullKeys { layer: layer_idx }, copied));
                }
                if let Some(values) = kv.values_ref()
                    && let Some(copied) = detach_array_if_view(values, false)?
                {
                    pending.push((DetachTarget::FullValues { layer: layer_idx }, copied));
                }
            }
        }
    }

    if pending.is_empty() {
        return Ok(());
    }

    let refs: Vec<&MxArray> = pending.iter().map(|(_, arr)| arr).collect();
    MxArray::eval_arrays_with_context(
        &refs,
        "clairic_persistent_cache::detach_snapshot_views::copy",
    )?;

    for (target, detached) in pending {
        match target {
            DetachTarget::Linear { layer, slot } => {
                if let Qwen3_5LayerCache::Linear(ac) = &mut snapshot.caches[layer] {
                    ac.set(slot, detached)?;
                }
            }
            DetachTarget::FullKeys { layer } => {
                if let Qwen3_5LayerCache::FullAttention(kv) = &mut snapshot.caches[layer] {
                    kv.set_keys(detached);
                }
            }
            DetachTarget::FullValues { layer } => {
                if let Qwen3_5LayerCache::FullAttention(kv) = &mut snapshot.caches[layer] {
                    kv.set_values(detached);
                }
            }
        }
    }

    Ok(())
}

/// Model-lifetime, content-addressed LRU caches shared across Clairic requests.
pub(crate) struct ClairicPersistentCache {
    media_features: BoundedLru<String, (MxArray, [i32; 3])>,
    prefix_snapshots: BoundedLru<(String, String), PrefixSnapshot>,
    #[cfg(test)]
    enabled_override: Option<bool>,
}

impl Default for ClairicPersistentCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ClairicPersistentCache {
    pub(crate) fn new() -> Self {
        Self {
            media_features: BoundedLru::new(MEDIA_MAX_ENTRIES, MEDIA_MAX_BYTES),
            prefix_snapshots: BoundedLru::new(PREFIX_MAX_ENTRIES, PREFIX_MAX_BYTES),
            #[cfg(test)]
            enabled_override: None,
        }
    }

    #[cfg(test)]
    fn with_limits_for_test(
        enabled: bool,
        media_max_entries: usize,
        media_max_bytes: usize,
        prefix_max_entries: usize,
        prefix_max_bytes: usize,
    ) -> Self {
        Self {
            media_features: BoundedLru::new(media_max_entries, media_max_bytes),
            prefix_snapshots: BoundedLru::new(prefix_max_entries, prefix_max_bytes),
            enabled_override: Some(enabled),
        }
    }

    /// Returns whether cross-request Clairic caching is enabled
    /// (`MLX_CLAIRIC_PERSIST_CACHE=0` disables it; enabled by default).
    #[allow(dead_code)]
    pub(crate) fn is_enabled() -> bool {
        *PERSIST_CACHE_ENABLED.get_or_init(|| {
            parse_persist_cache_flag(std::env::var("MLX_CLAIRIC_PERSIST_CACHE").ok().as_deref())
        })
    }

    #[allow(dead_code)]
    fn effective_enabled(&self) -> bool {
        #[cfg(test)]
        if let Some(enabled) = self.enabled_override {
            return enabled;
        }
        Self::is_enabled()
    }

    /// Look up cached vision features by `image_digest`, touching LRU recency
    /// and cloning the `MxArray` handle on hit.
    #[allow(dead_code)]
    pub(crate) fn get_media(&mut self, image_digest: &str) -> Option<(MxArray, [i32; 3])> {
        if !self.effective_enabled() {
            return None;
        }
        self.media_features
            .get_by(|key| key.as_str() == image_digest)
            .map(|(features, grid)| (features.clone(), *grid))
    }

    /// Insert vision features keyed by `image_digest`, detaching oversized
    /// slice/strided views so cached entries do not pin larger backing buffers
    /// and charging actual retained backing bytes against the LRU budget.
    /// No-op when disabled, when a single entry exceeds the byte budget, or if
    /// materializing a detached copy fails.
    #[allow(dead_code)]
    pub(crate) fn insert_media(
        &mut self,
        image_digest: String,
        mut features: MxArray,
        grid: [i32; 3],
    ) {
        if !self.effective_enabled() {
            return;
        }
        if self.media_features.max_entries == 0 || features.nbytes() > self.media_features.max_bytes
        {
            return;
        }
        if !features.is_available()
            && let Err(err) = MxArray::eval_arrays_with_context(
                &[&features],
                "clairic_persistent_cache::insert_media::eval",
            )
        {
            tracing::warn!(
                target: "mlx_core::clairic",
                error = %err,
                "Failed to evaluate vision features before persistent caching; skipping insert"
            );
            return;
        }
        match detach_array_if_view(&features, false) {
            Ok(Some(detached)) => {
                if let Err(err) = MxArray::eval_arrays_with_context(
                    &[&detached],
                    "clairic_persistent_cache::insert_media::copy",
                ) {
                    tracing::warn!(
                        target: "mlx_core::clairic",
                        error = %err,
                        "Failed to materialize detached vision features view; skipping insert"
                    );
                    return;
                }
                features = detached;
            }
            Ok(None) => {}
            Err(err) => {
                tracing::warn!(
                    target: "mlx_core::clairic",
                    error = %err,
                    "Failed to detach oversized vision features view; skipping insert"
                );
                return;
            }
        }
        let bytes = retained_array_bytes(&features);
        let _ = self
            .media_features
            .insert(image_digest, (features, grid), bytes);
    }

    /// Look up a cached `B=1` prefix snapshot keyed strictly by
    /// `(media_content_digest, prefix_hash)`, touching LRU recency on hit.
    #[allow(dead_code)]
    pub(crate) fn get_prefix(
        &mut self,
        media_content_digest: &str,
        prefix_hash: &str,
    ) -> Option<&PrefixSnapshot> {
        if !self.effective_enabled() {
            return None;
        }
        self.prefix_snapshots.get_by(|(media_digest, tok_hash)| {
            media_digest.as_str() == media_content_digest && tok_hash.as_str() == prefix_hash
        })
    }

    /// Insert a `B=1` prefix snapshot keyed strictly by
    /// `(media_content_digest, prefix_hash)`, detaching view-backed layer state
    /// (such as GDN `conv_state` split views) so parent allocations are
    /// released immediately and charging actual retained backing bytes
    /// (`arr.nbytes().max(arr.backing_buffer_bytes())`) against the LRU budget.
    /// Evicts oldest entries when over count or byte budget.
    ///
    /// Returns `None` when stored in the persistent cache, or `Some(snapshot)`
    /// when declined (persistent cache disabled, single entry exceeds the byte
    /// budget, or view detachment fails), allowing the caller to retain it in
    /// request-local storage.
    #[allow(dead_code)]
    pub(crate) fn insert_prefix(
        &mut self,
        media_content_digest: String,
        prefix_hash: String,
        mut snapshot: PrefixSnapshot,
    ) -> Option<PrefixSnapshot> {
        if !self.effective_enabled() {
            return Some(snapshot);
        }
        // Reject entries whose logical size already exceeds the budget before
        // paying any copy/eval work to detach views.
        if self.prefix_snapshots.max_entries == 0
            || estimate_prefix_bytes(&snapshot) > self.prefix_snapshots.max_bytes
        {
            return Some(snapshot);
        }
        if let Err(err) = detach_snapshot_views(&mut snapshot) {
            tracing::warn!(
                target: "mlx_core::clairic",
                error = %err,
                "Failed to detach views from prefix snapshot; skipping persistent cache insert"
            );
            return Some(snapshot);
        }
        let bytes = estimate_retained_prefix_bytes(&snapshot);
        self.prefix_snapshots
            .insert((media_content_digest, prefix_hash), snapshot, bytes)
    }

    /// Evict the oldest cached prefix snapshot to relieve memory pressure.
    /// Returns `true` if an entry was evicted, `false` if empty.
    #[allow(dead_code)]
    pub(crate) fn evict_oldest_prefix(&mut self) -> bool {
        self.prefix_snapshots.evict_oldest()
    }

    /// Evict the oldest cached vision feature entry to relieve memory pressure.
    /// Returns `true` if an entry was evicted, `false` if empty.
    #[allow(dead_code)]
    pub(crate) fn evict_oldest_media(&mut self) -> bool {
        self.media_features.evict_oldest()
    }

    /// Drop all cached vision features and prefix snapshots.
    pub(crate) fn clear(&mut self) {
        self.media_features.clear();
        self.prefix_snapshots.clear();
    }
}

#[cfg(test)]
mod tests {
    use napi::bindgen_prelude::{Error, Result};

    use super::*;

    fn f32_tensor(len: usize, fill: f32) -> Result<MxArray> {
        let data = vec![fill; len];
        MxArray::from_float32(&data, &[1, len as i64])
    }

    fn make_prefix_snapshot(
        num_floats_per_slot: usize,
        cache_len: i32,
        next_position: i32,
    ) -> Result<PrefixSnapshot> {
        let mut linear = Qwen3_5LayerCache::new_linear();
        let arrays = linear
            .as_arrays_cache_mut()
            .ok_or_else(|| Error::from_reason("expected Linear cache"))?;
        arrays.set(0, f32_tensor(num_floats_per_slot, 1.0)?)?;
        arrays.set(1, f32_tensor(num_floats_per_slot, 2.0)?)?;

        let mut full = Qwen3_5LayerCache::new_full_attention();
        let kv = full
            .as_kv_cache_mut()
            .ok_or_else(|| Error::from_reason("expected FullAttention cache"))?;
        kv.set_keys(f32_tensor(num_floats_per_slot, 3.0)?);
        kv.set_values(f32_tensor(num_floats_per_slot, 4.0)?);
        kv.set_offset(cache_len);

        Ok(PrefixSnapshot {
            caches: vec![linear, full],
            cache_len,
            next_position,
        })
    }

    #[test]
    fn parse_persist_cache_flag_respects_kill_switch() {
        assert!(parse_persist_cache_flag(None));
        assert!(parse_persist_cache_flag(Some("1")));
        assert!(parse_persist_cache_flag(Some("true")));
        assert!(parse_persist_cache_flag(Some("yes")));
        assert!(parse_persist_cache_flag(Some("")));

        assert!(!parse_persist_cache_flag(Some("0")));
        assert!(!parse_persist_cache_flag(Some(" 0 ")));
        assert!(!parse_persist_cache_flag(Some("false")));
        assert!(!parse_persist_cache_flag(Some("FALSE")));
        assert!(!parse_persist_cache_flag(Some("off")));
        assert!(!parse_persist_cache_flag(Some("Off")));
    }

    #[test]
    fn bounded_lru_evicts_oldest_by_count_and_touches_recency() {
        let mut lru: BoundedLru<String, u32> = BoundedLru::new(3, 1024);
        assert_eq!(lru.insert("a".to_string(), 1, 10), None);
        assert_eq!(lru.insert("b".to_string(), 2, 10), None);
        assert_eq!(lru.insert("c".to_string(), 3, 10), None);
        assert_eq!(lru.len(), 3);
        assert_eq!(lru.total_bytes(), 30);

        // Touch "a" so "b" becomes oldest.
        assert_eq!(lru.get_by(|k| k == "a"), Some(&1));

        // Inserting "d" evicts "b".
        assert_eq!(lru.insert("d".to_string(), 4, 10), None);
        assert_eq!(lru.len(), 3);
        assert_eq!(lru.total_bytes(), 30);
        assert_eq!(lru.get_by(|k| k == "b"), None);
        assert_eq!(lru.get_by(|k| k == "a"), Some(&1));
        assert_eq!(lru.get_by(|k| k == "c"), Some(&3));
        assert_eq!(lru.get_by(|k| k == "d"), Some(&4));

        // Re-inserting existing key "a" replaces in place without evicting others.
        assert_eq!(lru.insert("a".to_string(), 100, 25), None);
        assert_eq!(lru.len(), 3);
        assert_eq!(lru.total_bytes(), 45);
        assert_eq!(lru.get_by(|k| k == "a"), Some(&100));
        assert_eq!(lru.get_by(|k| k == "c"), Some(&3));
        assert_eq!(lru.get_by(|k| k == "d"), Some(&4));
    }

    #[test]
    fn bounded_lru_enforces_byte_budget_and_skips_oversized_entry() {
        let mut lru: BoundedLru<String, u32> = BoundedLru::new(10, 100);
        assert_eq!(lru.insert("a".to_string(), 1, 40), None);
        assert_eq!(lru.insert("b".to_string(), 2, 40), None);
        assert_eq!(lru.total_bytes(), 80);

        // 80 + 30 > 100 -> evicts "a" (40B), leaving "b" (40B) + "c" (30B) = 70B.
        assert_eq!(lru.insert("c".to_string(), 3, 30), None);
        assert_eq!(lru.len(), 2);
        assert_eq!(lru.total_bytes(), 70);
        assert_eq!(lru.get_by(|k| k == "a"), None);

        // Oversized single entry (101B > 100B) is declined and returned without evicting live entries.
        assert_eq!(lru.insert("huge".to_string(), 99, 101), Some(99));
        assert_eq!(lru.len(), 2);
        assert_eq!(lru.total_bytes(), 70);
        assert_eq!(lru.get_by(|k| k == "huge"), None);
        assert_eq!(lru.get_by(|k| k == "b"), Some(&2));
        assert_eq!(lru.get_by(|k| k == "c"), Some(&3));

        // Entry equal to the full budget evicts all existing entries and fits.
        assert_eq!(lru.insert("full".to_string(), 4, 100), None);
        assert_eq!(lru.len(), 1);
        assert_eq!(lru.total_bytes(), 100);
        assert_eq!(lru.get_by(|k| k == "full"), Some(&4));

        assert!(lru.evict_oldest());
        assert_eq!(lru.len(), 0);
        assert_eq!(lru.total_bytes(), 0);
        assert!(!lru.evict_oldest());
    }

    #[test]
    fn persistent_cache_media_lru_and_byte_budget() -> Result<()> {
        // Measure actual retained backing bytes per tensor so the test is
        // deterministic across CUDA (power-of-2 buckets), Metal, and CPU
        // regardless of allocator cache reuse slack.
        let t_a = f32_tensor(4, 1.0)?;
        let t_b = f32_tensor(4, 2.0)?;
        let t_c = f32_tensor(4, 3.0)?;
        let t_big = f32_tensor(16, 5.0)?;
        let t_huge = f32_tensor(64, 9.0)?;

        let b_a = retained_array_bytes(&t_a);
        let b_b = retained_array_bytes(&t_b);
        let b_big = retained_array_bytes(&t_big);
        let b_huge = retained_array_bytes(&t_huge);
        // Budget holds two small tensors (a + b) or one `t_big`, never `t_big + small`
        // and never `t_huge`.
        let max_bytes = (b_a + b_b).max(b_big);
        assert!(b_big > b_a && b_big > b_b);
        assert!(b_huge > max_bytes);

        let mut cache = ClairicPersistentCache::with_limits_for_test(true, 2, max_bytes, 2, 128);

        cache.insert_media("img_a".to_string(), t_a, [1, 2, 2]);
        cache.insert_media("img_b".to_string(), t_b, [1, 4, 4]);
        assert_eq!(cache.media_features.len(), 2);
        assert_eq!(cache.media_features.total_bytes(), b_a + b_b);

        // Touch img_a so img_b becomes LRU.
        let hit_a = cache
            .get_media("img_a")
            .ok_or_else(|| Error::from_reason("expected img_a hit"))?;
        assert_eq!(hit_a.1, [1, 2, 2]);
        assert_eq!(hit_a.0.nbytes(), 16);

        // Inserting img_c evicts img_b by entry count.
        let b_c = retained_array_bytes(&t_c);
        assert!(b_a + b_c <= max_bytes);
        cache.insert_media("img_c".to_string(), t_c, [1, 6, 6]);
        assert!(cache.get_media("img_b").is_none());
        assert!(cache.get_media("img_a").is_some());
        assert!(cache.get_media("img_c").is_some());

        // Single entry exceeding max_bytes is ignored without evicting.
        cache.insert_media("img_huge".to_string(), t_huge, [1, 8, 8]);
        assert!(cache.get_media("img_huge").is_none());
        assert_eq!(cache.media_features.len(), 2);

        // Entry of size `b_big` fits only after evicting both smaller entries.
        cache.insert_media("img_big".to_string(), t_big, [1, 10, 10]);
        assert_eq!(cache.media_features.len(), 1);
        assert_eq!(cache.media_features.total_bytes(), b_big);
        let hit_big = cache
            .get_media("img_big")
            .ok_or_else(|| Error::from_reason("expected img_big hit"))?;
        assert_eq!(hit_big.1, [1, 10, 10]);

        cache.clear();
        assert_eq!(cache.media_features.len(), 0);
        assert_eq!(cache.media_features.total_bytes(), 0);

        // Partial byte-budget eviction with generous entry cap: evicts only as many LRU
        // entries as needed to fit the incoming tensor.
        let a = f32_tensor(4, 1.0)?;
        let b = f32_tensor(4, 2.0)?;
        let c = f32_tensor(4, 3.0)?;
        let d = f32_tensor(8, 4.0)?;
        let a_b = retained_array_bytes(&a);
        let b_b = retained_array_bytes(&b);
        let c_b = retained_array_bytes(&c);
        let d_b = retained_array_bytes(&d);
        // `d` is 32B logical while `a, b, c` are 16B logical (< 32B backing), so
        // `d_b > b_b` guarantees `a_b + c_b + d_b > max(a_b + b_b + c_b, a_b + d_b)`.
        let partial_budget = (a_b + b_b + c_b).max(a_b + d_b);
        assert!(a_b + c_b + d_b > partial_budget);

        let mut by_bytes =
            ClairicPersistentCache::with_limits_for_test(true, 8, partial_budget, 2, 128);
        by_bytes.insert_media("a".to_string(), a, [1, 1, 1]);
        by_bytes.insert_media("b".to_string(), b, [1, 1, 1]);
        by_bytes.insert_media("c".to_string(), c, [1, 1, 1]);
        assert_eq!(by_bytes.media_features.total_bytes(), a_b + b_b + c_b);
        assert!(by_bytes.get_media("a").is_some()); // order: [b, c, a]
        // Evicts b and c, keeping a alongside d.
        by_bytes.insert_media("d".to_string(), d, [1, 2, 3]);
        assert_eq!(by_bytes.media_features.len(), 2);
        assert_eq!(by_bytes.media_features.total_bytes(), a_b + d_b);
        assert!(by_bytes.get_media("b").is_none());
        assert!(by_bytes.get_media("c").is_none());
        assert!(by_bytes.get_media("a").is_some());
        assert!(by_bytes.get_media("d").is_some());
        Ok(())
    }

    #[test]
    fn persistent_cache_prefix_keys_budgets_eviction_and_disabled() -> Result<()> {
        // Each 4-float-per-slot snapshot has 4 * 16B = 64B logical (<= 124B retained
        // with worst-case <2x allocator slack). Budget of 256B always fits two
        // 4-float snapshots while rejecting a 32-float snapshot (512B logical).
        let mut cache = ClairicPersistentCache::with_limits_for_test(true, 4, 256, 2, 256);

        assert!(
            cache
                .insert_prefix(
                    "media_1".to_string(),
                    "tok_shared".to_string(),
                    make_prefix_snapshot(4, 10, 10)?,
                )
                .is_none()
        );
        assert!(
            cache
                .insert_prefix(
                    "media_2".to_string(),
                    "tok_shared".to_string(),
                    make_prefix_snapshot(4, 20, 22)?,
                )
                .is_none()
        );
        assert_eq!(cache.prefix_snapshots.len(), 2);
        let stored_1_bytes = estimate_retained_prefix_bytes(
            cache
                .get_prefix("media_1", "tok_shared")
                .ok_or_else(|| Error::from_reason("expected media_1 hit"))?,
        );
        let stored_2_bytes = estimate_retained_prefix_bytes(
            cache
                .get_prefix("media_2", "tok_shared")
                .ok_or_else(|| Error::from_reason("expected media_2 hit"))?,
        );
        assert_eq!(
            cache.prefix_snapshots.total_bytes(),
            stored_1_bytes + stored_2_bytes
        );

        // Strict composite key: same prefix_hash with distinct media digest returns distinct entry.
        let snap1 = cache
            .get_prefix("media_1", "tok_shared")
            .ok_or_else(|| Error::from_reason("expected media_1 hit"))?;
        assert_eq!(snap1.cache_len, 10);
        assert_eq!(snap1.next_position, 10);

        // media_1 was just touched, so media_2 is now oldest.
        // Inserting media_3 evicts media_2.
        assert!(
            cache
                .insert_prefix(
                    "media_3".to_string(),
                    "tok_shared".to_string(),
                    make_prefix_snapshot(4, 30, 35)?,
                )
                .is_none()
        );
        assert!(cache.get_prefix("media_2", "tok_shared").is_none());
        assert!(cache.get_prefix("media_1", "tok_shared").is_some());
        assert!(cache.get_prefix("media_3", "tok_shared").is_some());

        // Oversized snapshot (4 arrays * 32 floats * 4B = 512B > 256B) is returned back
        // to the caller rather than silently dropped, leaving existing entries intact.
        let returned_huge = cache.insert_prefix(
            "media_huge".to_string(),
            "tok_huge".to_string(),
            make_prefix_snapshot(32, 99, 99)?,
        );
        let returned_huge = returned_huge
            .ok_or_else(|| Error::from_reason("expected oversized snapshot to be returned"))?;
        assert_eq!(returned_huge.cache_len, 99);
        assert_eq!(returned_huge.next_position, 99);
        assert!(cache.get_prefix("media_huge", "tok_huge").is_none());
        assert_eq!(cache.prefix_snapshots.len(), 2);

        // Memory-pressure relief evicts oldest first (media_1, since media_3 was touched last).
        assert!(cache.evict_oldest_prefix());
        assert_eq!(cache.prefix_snapshots.len(), 1);
        assert!(cache.get_prefix("media_1", "tok_shared").is_none());
        assert!(cache.get_prefix("media_3", "tok_shared").is_some());

        assert!(cache.evict_oldest_prefix());
        assert_eq!(cache.prefix_snapshots.len(), 0);
        assert_eq!(cache.prefix_snapshots.total_bytes(), 0);
        assert!(!cache.evict_oldest_prefix());

        // Disabled cache declines inserts and returns prefix snapshots to the caller.
        let mut disabled = ClairicPersistentCache::with_limits_for_test(false, 4, 256, 4, 256);
        disabled.insert_media("img".to_string(), f32_tensor(4, 1.0)?, [1, 2, 2]);
        let returned_disabled = disabled.insert_prefix(
            "media".to_string(),
            "tok".to_string(),
            make_prefix_snapshot(4, 5, 5)?,
        );
        let returned_disabled = returned_disabled
            .ok_or_else(|| Error::from_reason("expected disabled cache to return snapshot"))?;
        assert_eq!(returned_disabled.cache_len, 5);
        assert_eq!(returned_disabled.next_position, 5);
        assert!(disabled.get_media("img").is_none());
        assert!(disabled.get_prefix("media", "tok").is_none());
        assert_eq!(disabled.media_features.len(), 0);
        assert_eq!(disabled.prefix_snapshots.len(), 0);
        Ok(())
    }

    #[test]
    fn persistent_cache_returns_oversized_prefix_snapshot_for_local_fallback() -> Result<()> {
        // Budget of 128B fits a 4-float snapshot (64B logical, <= 124B retained)
        // but rejects a 16-float snapshot (256B logical > 128B).
        let mut cache = ClairicPersistentCache::with_limits_for_test(true, 4, 256, 4, 128);

        assert!(
            cache
                .insert_prefix(
                    "media_small".to_string(),
                    "tok_small".to_string(),
                    make_prefix_snapshot(4, 12, 14)?,
                )
                .is_none()
        );
        assert_eq!(cache.prefix_snapshots.len(), 1);
        let small_bytes = estimate_retained_prefix_bytes(
            cache
                .get_prefix("media_small", "tok_small")
                .ok_or_else(|| Error::from_reason("expected media_small hit"))?,
        );
        assert_eq!(cache.prefix_snapshots.total_bytes(), small_bytes);
        assert!((64..=128).contains(&small_bytes));

        let oversized = make_prefix_snapshot(16, 256, 260)?;
        let declined = cache.insert_prefix(
            "media_oversized".to_string(),
            "tok_oversized".to_string(),
            oversized,
        );
        let returned = declined
            .ok_or_else(|| Error::from_reason("oversized prefix snapshot must be returned"))?;
        assert_eq!(returned.cache_len, 256);
        assert_eq!(returned.next_position, 260);
        assert_eq!(returned.caches.len(), 2);

        // Persistent cache remains unchanged and still holds the earlier small entry.
        assert!(
            cache
                .get_prefix("media_oversized", "tok_oversized")
                .is_none()
        );
        assert_eq!(cache.prefix_snapshots.len(), 1);
        assert_eq!(cache.prefix_snapshots.total_bytes(), small_bytes);
        assert!(cache.get_prefix("media_small", "tok_small").is_some());
        Ok(())
    }

    #[test]
    fn persistent_cache_evict_oldest_media_follows_lru_order() -> Result<()> {
        let mut cache = ClairicPersistentCache::with_limits_for_test(true, 4, 256, 2, 128);
        assert!(!cache.evict_oldest_media());

        let img_1 = f32_tensor(4, 1.0)?;
        let img_2 = f32_tensor(6, 2.0)?;
        let img_3 = f32_tensor(4, 3.0)?;
        let b_1 = retained_array_bytes(&img_1);
        let b_2 = retained_array_bytes(&img_2);
        let b_3 = retained_array_bytes(&img_3);

        cache.insert_media("img_1".to_string(), img_1, [1, 2, 2]);
        cache.insert_media("img_2".to_string(), img_2, [1, 4, 4]);
        cache.insert_media("img_3".to_string(), img_3, [1, 6, 6]);
        assert_eq!(cache.media_features.len(), 3);
        assert_eq!(cache.media_features.total_bytes(), b_1 + b_2 + b_3);

        // Touch img_1 so LRU order becomes [img_2, img_3, img_1].
        assert!(cache.get_media("img_1").is_some());

        assert!(cache.evict_oldest_media());
        assert_eq!(cache.media_features.len(), 2);
        assert_eq!(cache.media_features.total_bytes(), b_1 + b_3);
        assert!(cache.get_media("img_2").is_none());
        assert!(cache.get_media("img_3").is_some());
        assert!(cache.get_media("img_1").is_some());

        // Above get_media calls moved img_3 then img_1 to MRU; img_3 is now oldest.
        assert!(cache.evict_oldest_media());
        assert_eq!(cache.media_features.len(), 1);
        assert_eq!(cache.media_features.total_bytes(), b_1);
        assert!(cache.get_media("img_3").is_none());
        assert!(cache.get_media("img_1").is_some());

        assert!(cache.evict_oldest_media());
        assert_eq!(cache.media_features.len(), 0);
        assert_eq!(cache.media_features.total_bytes(), 0);
        assert!(!cache.evict_oldest_media());
        Ok(())
    }

    #[test]
    fn insert_prefix_detaches_split_view_conv_state_and_bounds_backing_memory() -> Result<()> {
        const ROWS: i64 = 4096;
        const COLS: i64 = 512;
        const KEEP_ROWS: i64 = 3;
        let logical_bytes = (KEEP_ROWS * COLS) as usize * std::mem::size_of::<f32>();

        // 4096 * 512 * 4 = 8 MiB conv_input with exact distinct f32 values per (row, col).
        let mut data = Vec::with_capacity((ROWS * COLS) as usize);
        for r in 0..ROWS {
            let base = (r as f32) * 1024.0;
            for c in 0..COLS {
                data.push(base + (c as f32));
            }
        }
        let conv_input = MxArray::from_float32(&data, &[1, ROWS, COLS])?;
        let parts = conv_input.split_sections(&[ROWS - KEEP_ROWS], 1)?;
        assert_eq!(parts.len(), 2);
        let conv_state = parts[1].clone();
        MxArray::eval_arrays(&[&conv_state])?;

        // Confirm the split view really pins the 8 MiB parent buffer at a nonzero offset.
        assert_eq!(conv_state.nbytes(), logical_bytes);
        assert!(
            conv_state.backing_buffer_bytes() >= (8 << 20),
            "expected split view to retain >= 8 MiB backing buffer, got {}",
            conv_state.backing_buffer_bytes()
        );
        let pre_offset = unsafe { mlx_sys::mlx_array_get_buffer_offset(conv_state.as_raw_ptr()) };
        assert!(pre_offset > 0, "expected nonzero byte offset on tail split");
        assert!(conv_state.is_oversized_view());

        let expected = conv_state.to_float32()?.to_vec();
        drop(parts);
        drop(conv_input);

        let mut linear = Qwen3_5LayerCache::new_linear();
        linear
            .as_arrays_cache_mut()
            .ok_or_else(|| Error::from_reason("expected Linear cache"))?
            .set(0, conv_state)?;
        let snapshot = PrefixSnapshot {
            caches: vec![linear],
            cache_len: ROWS as i32,
            next_position: ROWS as i32,
        };

        let mut cache = ClairicPersistentCache::with_limits_for_test(true, 4, 1 << 20, 4, 1 << 20);
        assert!(
            cache
                .insert_prefix("media_split".to_string(), "tok_split".to_string(), snapshot)
                .is_none()
        );
        assert!(
            cache.prefix_snapshots.total_bytes() <= (64 << 10),
            "expected detached LRU byte charge <= 64 KiB, got {}",
            cache.prefix_snapshots.total_bytes()
        );
        assert!(cache.prefix_snapshots.total_bytes() >= logical_bytes);

        let stored = cache
            .get_prefix("media_split", "tok_split")
            .ok_or_else(|| Error::from_reason("expected media_split hit"))?;
        let Qwen3_5LayerCache::Linear(ac) = &stored.caches[0] else {
            return Err(Error::from_reason(
                "expected Linear layer in stored snapshot",
            ));
        };
        let stored_conv = ac
            .get(0)
            .ok_or_else(|| Error::from_reason("expected stored slot 0"))?;
        assert_eq!(stored_conv.nbytes(), logical_bytes);
        assert!(
            stored_conv.backing_buffer_bytes() <= (64 << 10),
            "expected detached conv_state backing buffer <= 64 KiB, got {}",
            stored_conv.backing_buffer_bytes()
        );
        let post_offset = unsafe { mlx_sys::mlx_array_get_buffer_offset(stored_conv.as_raw_ptr()) };
        assert_eq!(post_offset, 0);
        assert!(!stored_conv.is_oversized_view());
        assert_eq!(stored_conv.to_float32()?.as_ref(), expected.as_slice());
        Ok(())
    }

    #[test]
    fn insert_prefix_detaches_prefix_slice_view_with_zero_offset() -> Result<()> {
        const ROWS: i64 = 4096;
        const COLS: i64 = 512;
        const KEEP_ROWS: i64 = 3;
        let logical_elems = (KEEP_ROWS * COLS) as usize;
        let logical_bytes = logical_elems * std::mem::size_of::<f32>();

        let mut data = Vec::with_capacity((ROWS * COLS) as usize);
        for r in 0..ROWS {
            let base = (r as f32) * 1024.0;
            for c in 0..COLS {
                data.push(base + (c as f32));
            }
        }
        let big = MxArray::from_float32(&data, &[1, ROWS, COLS])?;
        let prefix_slice = big.slice_axis(1, 0, KEEP_ROWS)?;
        MxArray::eval_arrays(&[&prefix_slice])?;

        // Confirm zero-offset, contiguous prefix slice still pins the 8 MiB backing buffer.
        let pre_offset = unsafe { mlx_sys::mlx_array_get_buffer_offset(prefix_slice.as_raw_ptr()) };
        let pre_data_size = unsafe { mlx_sys::mlx_array_get_data_size(prefix_slice.as_raw_ptr()) };
        assert_eq!(pre_offset, 0);
        assert_eq!(pre_data_size, logical_elems);
        assert_eq!(prefix_slice.nbytes(), logical_bytes);
        assert!(
            prefix_slice.backing_buffer_bytes() >= (8 << 20),
            "expected zero-offset slice to retain >= 8 MiB backing buffer, got {}",
            prefix_slice.backing_buffer_bytes()
        );
        assert!(prefix_slice.is_oversized_view());

        let expected = prefix_slice.to_float32()?.to_vec();
        drop(big);

        // 1) Non-forced Linear slot 1 (`recurrent_state`) path.
        let mut linear = Qwen3_5LayerCache::new_linear();
        linear
            .as_arrays_cache_mut()
            .ok_or_else(|| Error::from_reason("expected Linear cache"))?
            .set(1, prefix_slice.clone())?;
        let snapshot = PrefixSnapshot {
            caches: vec![linear],
            cache_len: KEEP_ROWS as i32,
            next_position: KEEP_ROWS as i32,
        };

        let mut cache = ClairicPersistentCache::with_limits_for_test(true, 4, 1 << 20, 4, 1 << 20);
        assert!(
            cache
                .insert_prefix(
                    "media_prefix_slice".to_string(),
                    "tok_prefix_slice".to_string(),
                    snapshot,
                )
                .is_none()
        );
        assert!(
            cache.prefix_snapshots.total_bytes() <= (64 << 10),
            "expected detached slot 1 LRU charge <= 64 KiB, got {}",
            cache.prefix_snapshots.total_bytes()
        );
        assert!(cache.prefix_snapshots.total_bytes() >= logical_bytes);

        let stored = cache
            .get_prefix("media_prefix_slice", "tok_prefix_slice")
            .ok_or_else(|| Error::from_reason("expected media_prefix_slice hit"))?;
        let Qwen3_5LayerCache::Linear(ac) = &stored.caches[0] else {
            return Err(Error::from_reason(
                "expected Linear layer in stored snapshot",
            ));
        };
        let stored_slot1 = ac
            .get(1)
            .ok_or_else(|| Error::from_reason("expected stored slot 1"))?;
        assert_eq!(stored_slot1.nbytes(), logical_bytes);
        assert!(
            stored_slot1.backing_buffer_bytes() <= (64 << 10),
            "expected detached slot 1 backing buffer <= 64 KiB, got {}",
            stored_slot1.backing_buffer_bytes()
        );
        assert!(!stored_slot1.is_oversized_view());
        assert_eq!(stored_slot1.to_float32()?.as_ref(), expected.as_slice());

        // 2) Non-forced FullAttention KVCache (`keys` / `values`) path.
        let mut full = Qwen3_5LayerCache::new_full_attention();
        let kv = full
            .as_kv_cache_mut()
            .ok_or_else(|| Error::from_reason("expected FullAttention cache"))?;
        kv.set_keys(prefix_slice.clone());
        kv.set_values(prefix_slice.clone());
        kv.set_offset(KEEP_ROWS as i32);
        let kv_snapshot = PrefixSnapshot {
            caches: vec![full],
            cache_len: KEEP_ROWS as i32,
            next_position: KEEP_ROWS as i32,
        };
        assert!(
            cache
                .insert_prefix(
                    "media_kv_slice".to_string(),
                    "tok_kv_slice".to_string(),
                    kv_snapshot,
                )
                .is_none()
        );
        let stored_kv = cache
            .get_prefix("media_kv_slice", "tok_kv_slice")
            .ok_or_else(|| Error::from_reason("expected media_kv_slice hit"))?;
        let Qwen3_5LayerCache::FullAttention(kv_out) = &stored_kv.caches[0] else {
            return Err(Error::from_reason(
                "expected FullAttention layer in stored snapshot",
            ));
        };
        let k_out = kv_out
            .keys_ref()
            .ok_or_else(|| Error::from_reason("expected stored keys"))?;
        let v_out = kv_out
            .values_ref()
            .ok_or_else(|| Error::from_reason("expected stored values"))?;
        assert!(k_out.backing_buffer_bytes() <= (64 << 10));
        assert!(v_out.backing_buffer_bytes() <= (64 << 10));
        assert!(!k_out.is_oversized_view());
        assert!(!v_out.is_oversized_view());
        assert_eq!(k_out.to_float32()?.as_ref(), expected.as_slice());
        assert_eq!(v_out.to_float32()?.as_ref(), expected.as_slice());

        // 3) `insert_media` non-forced `is_oversized_view()` path.
        cache.insert_media("img_prefix_slice".to_string(), prefix_slice, [1, 1, 3]);
        assert!(
            cache.media_features.total_bytes() <= (64 << 10),
            "expected detached media LRU charge <= 64 KiB, got {}",
            cache.media_features.total_bytes()
        );
        assert!(cache.media_features.total_bytes() >= logical_bytes);
        let (stored_media, grid) = cache
            .get_media("img_prefix_slice")
            .ok_or_else(|| Error::from_reason("expected img_prefix_slice hit"))?;
        assert_eq!(grid, [1, 1, 3]);
        assert_eq!(stored_media.nbytes(), logical_bytes);
        assert!(
            stored_media.backing_buffer_bytes() <= (64 << 10),
            "expected detached media backing buffer <= 64 KiB, got {}",
            stored_media.backing_buffer_bytes()
        );
        assert!(!stored_media.is_oversized_view());
        assert_eq!(stored_media.to_float32()?.as_ref(), expected.as_slice());
        Ok(())
    }
}
