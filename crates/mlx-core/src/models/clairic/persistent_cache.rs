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

    /// Insert vision features keyed by `image_digest`, evicting the oldest
    /// entries when over count or byte budget. No-op when disabled or when a
    /// single entry exceeds the byte budget.
    #[allow(dead_code)]
    pub(crate) fn insert_media(&mut self, image_digest: String, features: MxArray, grid: [i32; 3]) {
        if !self.effective_enabled() {
            return;
        }
        let bytes = features.nbytes();
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
    /// `(media_content_digest, prefix_hash)`, estimating its footprint by
    /// summing `arr.nbytes()` over `collect_arrays`. Evicts oldest entries when
    /// over count or byte budget.
    ///
    /// Returns `None` when stored in the persistent cache, or `Some(snapshot)`
    /// when declined (persistent cache disabled or single entry exceeds the
    /// byte budget), allowing the caller to retain it in request-local storage.
    #[allow(dead_code)]
    pub(crate) fn insert_prefix(
        &mut self,
        media_content_digest: String,
        prefix_hash: String,
        snapshot: PrefixSnapshot,
    ) -> Option<PrefixSnapshot> {
        if !self.effective_enabled() {
            return Some(snapshot);
        }
        let bytes = estimate_prefix_bytes(&snapshot);
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
        // Each f32 tensor of length 4 is 16 bytes.
        // Budget: max 2 entries, max 40 bytes (holds two 16B entries = 32B, not three).
        let mut cache = ClairicPersistentCache::with_limits_for_test(true, 2, 40, 2, 128);

        cache.insert_media("img_a".to_string(), f32_tensor(4, 1.0)?, [1, 2, 2]);
        cache.insert_media("img_b".to_string(), f32_tensor(4, 2.0)?, [1, 4, 4]);
        assert_eq!(cache.media_features.len(), 2);
        assert_eq!(cache.media_features.total_bytes(), 32);

        // Touch img_a so img_b becomes LRU.
        let hit_a = cache
            .get_media("img_a")
            .ok_or_else(|| Error::from_reason("expected img_a hit"))?;
        assert_eq!(hit_a.1, [1, 2, 2]);
        assert_eq!(hit_a.0.nbytes(), 16);

        // Inserting img_c (16B) evicts img_b.
        cache.insert_media("img_c".to_string(), f32_tensor(4, 3.0)?, [1, 6, 6]);
        assert!(cache.get_media("img_b").is_none());
        assert!(cache.get_media("img_a").is_some());
        assert!(cache.get_media("img_c").is_some());

        // Single entry exceeding 40B (12 floats = 48B) is ignored without evicting.
        cache.insert_media("img_huge".to_string(), f32_tensor(12, 9.0)?, [1, 8, 8]);
        assert!(cache.get_media("img_huge").is_none());
        assert_eq!(cache.media_features.len(), 2);

        // Entry of 32B (8 floats) fits in 40B only after evicting both 16B entries.
        cache.insert_media("img_big".to_string(), f32_tensor(8, 5.0)?, [1, 10, 10]);
        assert_eq!(cache.media_features.len(), 1);
        assert_eq!(cache.media_features.total_bytes(), 32);
        let hit_big = cache
            .get_media("img_big")
            .ok_or_else(|| Error::from_reason("expected img_big hit"))?;
        assert_eq!(hit_big.1, [1, 10, 10]);

        cache.clear();
        assert_eq!(cache.media_features.len(), 0);
        assert_eq!(cache.media_features.total_bytes(), 0);

        // Partial byte-budget eviction with generous entry cap: evicts only as many LRU
        // entries as needed to fit the incoming tensor.
        let mut by_bytes = ClairicPersistentCache::with_limits_for_test(true, 8, 48, 2, 128);
        by_bytes.insert_media("a".to_string(), f32_tensor(4, 1.0)?, [1, 1, 1]);
        by_bytes.insert_media("b".to_string(), f32_tensor(4, 2.0)?, [1, 1, 1]);
        by_bytes.insert_media("c".to_string(), f32_tensor(4, 3.0)?, [1, 1, 1]);
        assert_eq!(by_bytes.media_features.total_bytes(), 48);
        assert!(by_bytes.get_media("a").is_some()); // order: [b, c, a]
        // 24B (6 floats) requires freeing >= 24B -> evicts b (16B) and c (16B), keeping a (16B).
        by_bytes.insert_media("d".to_string(), f32_tensor(6, 4.0)?, [1, 2, 3]);
        assert_eq!(by_bytes.media_features.len(), 2);
        assert_eq!(by_bytes.media_features.total_bytes(), 40);
        assert!(by_bytes.get_media("b").is_none());
        assert!(by_bytes.get_media("c").is_none());
        assert!(by_bytes.get_media("a").is_some());
        assert!(by_bytes.get_media("d").is_some());
        Ok(())
    }

    #[test]
    fn persistent_cache_prefix_keys_budgets_eviction_and_disabled() -> Result<()> {
        // Each snapshot with 4 floats per slot has 4 arrays * 16B = 64B.
        // Budget: max 2 entries, max 140 bytes (holds two 64B snapshots = 128B).
        let mut cache = ClairicPersistentCache::with_limits_for_test(true, 4, 256, 2, 140);

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
        assert_eq!(cache.prefix_snapshots.total_bytes(), 128);

        // Strict composite key: same prefix_hash with distinct media digest returns distinct entry.
        let snap1 = cache
            .get_prefix("media_1", "tok_shared")
            .ok_or_else(|| Error::from_reason("expected media_1 hit"))?;
        assert_eq!(snap1.cache_len, 10);
        assert_eq!(snap1.next_position, 10);

        // media_1 was just touched, so media_2 is now oldest.
        // Inserting media_3 (64B) evicts media_2.
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

        // Oversized snapshot (4 arrays * 10 floats * 4B = 160B > 140B) is returned back
        // to the caller rather than silently dropped, leaving existing entries intact.
        let returned_huge = cache.insert_prefix(
            "media_huge".to_string(),
            "tok_huge".to_string(),
            make_prefix_snapshot(10, 99, 99)?,
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
        // Budget of 64 bytes fits a 4-float snapshot (64B) but rejects an 8-float snapshot (128B).
        let mut cache = ClairicPersistentCache::with_limits_for_test(true, 4, 256, 4, 64);

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
        assert_eq!(cache.prefix_snapshots.total_bytes(), 64);

        let oversized = make_prefix_snapshot(8, 256, 260)?;
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
        assert_eq!(cache.prefix_snapshots.total_bytes(), 64);
        assert!(cache.get_prefix("media_small", "tok_small").is_some());
        Ok(())
    }

    #[test]
    fn persistent_cache_evict_oldest_media_follows_lru_order() -> Result<()> {
        let mut cache = ClairicPersistentCache::with_limits_for_test(true, 4, 256, 2, 128);
        assert!(!cache.evict_oldest_media());

        cache.insert_media("img_1".to_string(), f32_tensor(4, 1.0)?, [1, 2, 2]);
        cache.insert_media("img_2".to_string(), f32_tensor(6, 2.0)?, [1, 4, 4]);
        cache.insert_media("img_3".to_string(), f32_tensor(4, 3.0)?, [1, 6, 6]);
        assert_eq!(cache.media_features.len(), 3);
        assert_eq!(cache.media_features.total_bytes(), 16 + 24 + 16);

        // Touch img_1 so LRU order becomes [img_2, img_3, img_1].
        assert!(cache.get_media("img_1").is_some());

        assert!(cache.evict_oldest_media());
        assert_eq!(cache.media_features.len(), 2);
        assert_eq!(cache.media_features.total_bytes(), 32);
        assert!(cache.get_media("img_2").is_none());
        assert!(cache.get_media("img_3").is_some());
        assert!(cache.get_media("img_1").is_some());

        // Above get_media calls moved img_3 then img_1 to MRU; img_3 is now oldest.
        assert!(cache.evict_oldest_media());
        assert_eq!(cache.media_features.len(), 1);
        assert_eq!(cache.media_features.total_bytes(), 16);
        assert!(cache.get_media("img_3").is_none());
        assert!(cache.get_media("img_1").is_some());

        assert!(cache.evict_oldest_media());
        assert_eq!(cache.media_features.len(), 0);
        assert_eq!(cache.media_features.total_bytes(), 0);
        assert!(!cache.evict_oldest_media());
        Ok(())
    }
}
