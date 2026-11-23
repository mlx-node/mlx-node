//! Per-request / LRU image and vision-view cache with RGB + tensor byte accounting.
//!
//! Mirrors upstream `clairic_runtime/media_cache.py`.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
};

use image::RgbImage;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::{embedded::ValidatedImage, media::hex_digest};

/// Default byte budget matching upstream `ImageCache(max_bytes=128 * 1024 * 1024)`.
pub const DEFAULT_CACHE_MAX_BYTES: usize = 128 * 1024 * 1024;
/// Default maximum entry count matching upstream `ImageCache(max_entries=8)`.
pub const DEFAULT_CACHE_MAX_ENTRIES: usize = 8;

/// Borrowed descriptor of one media input used to build a cache key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaKeyItem<'a> {
    pub content_type: &'a str,
    pub base64: &'a str,
    pub rotation: u16,
    pub explicit_rotation: bool,
}

impl<'a> From<&'a ValidatedImage> for MediaKeyItem<'a> {
    fn from(image: &'a ValidatedImage) -> Self {
        Self {
            content_type: &image.content_type,
            base64: &image.base64,
            rotation: image.rotation_degrees(),
            explicit_rotation: image.has_explicit_rotation(),
        }
    }
}

/// Compute a deterministic SHA-256 cache key matching `media_cache.py::ImageCache.key`.
pub fn media_cache_key(items: &[MediaKeyItem<'_>]) -> String {
    let mut hasher = Sha256::new();
    for item in items {
        let content = item.base64.as_bytes();
        hasher.update(item.content_type.as_bytes());
        hasher.update(item.rotation.to_string().as_bytes());
        hasher.update([u8::from(item.explicit_rotation)]);
        hasher.update((content.len() as u64).to_be_bytes());
        hasher.update(content);
    }
    hex_digest(&hasher.finalize())
}

/// Uncompressed RGB byte footprint of an `RgbImage` (`width * height * 3`).
pub fn rgb_image_bytes(image: &RgbImage) -> usize {
    (image.width() as usize)
        .saturating_mul(image.height() as usize)
        .saturating_mul(3)
}

/// Deduplicated RGB byte footprint across `images` and `source_images` (matching
/// `retained = {id(image): image for image in [*images, *source_images]}` in `media_cache.py`).
pub fn retained_rgb_bytes(images: &[Arc<RgbImage>], source_images: &[Arc<RgbImage>]) -> usize {
    let mut seen: HashSet<usize> = HashSet::new();
    let mut total: usize = 0;
    for img in images.iter().chain(source_images.iter()) {
        let ptr = Arc::as_ptr(img) as usize;
        if seen.insert(ptr) {
            total = total.saturating_add(rgb_image_bytes(img));
        }
    }
    total
}

/// Cached prepared images, optional source views, optional vision processor outputs,
/// and per-entry salt / byte size (port of upstream `PreparedImages`).
pub struct PreparedImages<P, M = ()> {
    pub images: Vec<Arc<RgbImage>>,
    pub source_images: Vec<Arc<RgbImage>>,
    pub processed: Vec<Arc<P>>,
    pub metadata: M,
    pub salt: String,
    pub size_bytes: usize,
}

impl<P, M: Clone> Clone for PreparedImages<P, M> {
    fn clone(&self) -> Self {
        Self {
            images: self.images.clone(),
            source_images: self.source_images.clone(),
            processed: self.processed.clone(),
            metadata: self.metadata.clone(),
            salt: self.salt.clone(),
            size_bytes: self.size_bytes,
        }
    }
}

/// LRU cache holding prepared RGB views and vision processor outputs, bounded by
/// total byte budget and entry count (port of upstream `media_cache.py::ImageCache`).
pub struct MediaCache<P, M = ()> {
    max_bytes: usize,
    max_entries: usize,
    namespace: String,
    entries: HashMap<String, PreparedImages<P, M>>,
    order: VecDeque<String>,
    size_bytes: usize,
}

impl<P, M> Default for MediaCache<P, M> {
    fn default() -> Self {
        Self::new(DEFAULT_CACHE_MAX_BYTES, DEFAULT_CACHE_MAX_ENTRIES)
    }
}

impl<P, M> MediaCache<P, M> {
    pub fn new(max_bytes: usize, max_entries: usize) -> Self {
        Self {
            max_bytes,
            max_entries,
            namespace: Uuid::new_v4().simple().to_string(),
            entries: HashMap::new(),
            order: VecDeque::new(),
            size_bytes: 0,
        }
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.size_bytes = 0;
    }

    /// Build a cache key for a slice of validated images.
    pub fn key_for_images(images: &[ValidatedImage]) -> String {
        let items: Vec<MediaKeyItem<'_>> = images.iter().map(MediaKeyItem::from).collect();
        media_cache_key(&items)
    }

    /// Retrieve an entry by key and promote it to most-recently-used.
    pub fn get(&mut self, key: &str) -> Option<&PreparedImages<P, M>> {
        if !self.entries.contains_key(key) {
            return None;
        }
        self.touch(key);
        self.entries.get(key)
    }

    /// Insert prepared views & optional processed outputs, evicting LRU entries as needed.
    ///
    /// If the entry alone exceeds `max_bytes`, it is returned without being stored
    /// (matching upstream `if size > self.max_bytes: return entry`).
    pub fn put(
        &mut self,
        key: impl Into<String>,
        images: Vec<Arc<RgbImage>>,
        source_images: Option<Vec<Arc<RgbImage>>>,
        processed: Vec<Arc<P>>,
        tensor_bytes: usize,
        metadata: M,
    ) -> PreparedImages<P, M>
    where
        M: Clone,
    {
        let key = key.into();
        let source_images = source_images.unwrap_or_else(|| images.clone());
        let size_bytes = tensor_bytes.saturating_add(retained_rgb_bytes(&images, &source_images));
        let entry = PreparedImages {
            images,
            source_images,
            processed,
            metadata,
            salt: Uuid::new_v4().simple().to_string(),
            size_bytes,
        };

        if size_bytes > self.max_bytes || self.max_entries == 0 {
            return entry;
        }

        if let Some(old) = self.entries.remove(&key) {
            self.size_bytes = self.size_bytes.saturating_sub(old.size_bytes);
            self.order.retain(|k| k != &key);
        }

        while !self.entries.is_empty()
            && (self.size_bytes.saturating_add(size_bytes) > self.max_bytes
                || self.entries.len() >= self.max_entries)
        {
            if let Some(oldest_key) = self.order.pop_front() {
                if let Some(old) = self.entries.remove(&oldest_key) {
                    self.size_bytes = self.size_bytes.saturating_sub(old.size_bytes);
                }
            } else {
                break;
            }
        }

        self.size_bytes = self.size_bytes.saturating_add(size_bytes);
        self.entries.insert(key.clone(), entry.clone());
        self.order.push_back(key);
        entry
    }

    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key)
            && let Some(k) = self.order.remove(pos)
        {
            self.order.push_back(k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn key_distinguishes_rotation_and_explicit_flag() {
        let a = MediaKeyItem {
            content_type: "image/png",
            base64: "AAAA",
            rotation: 0,
            explicit_rotation: false,
        };
        let b = MediaKeyItem {
            explicit_rotation: true,
            ..a
        };
        let c = MediaKeyItem {
            rotation: 90,
            explicit_rotation: true,
            ..a
        };
        assert_ne!(media_cache_key(&[a]), media_cache_key(&[b]));
        assert_ne!(media_cache_key(&[b]), media_cache_key(&[c]));
    }

    #[test]
    fn lru_eviction_and_deduplicated_byte_accounting() {
        // Each 10x10 RGB image is 300 bytes.
        let img1 = Arc::new(RgbImage::from_pixel(10, 10, Rgb([1, 2, 3])));
        let img2 = Arc::new(RgbImage::from_pixel(10, 10, Rgb([4, 5, 6])));
        let img3 = Arc::new(RgbImage::from_pixel(10, 10, Rgb([7, 8, 9])));

        // Budget holds 2 images (600 bytes), max_entries = 2.
        let mut cache: MediaCache<(), &str> = MediaCache::new(600, 2);

        // Passing the same Arc in images and source_images counts it once (300 bytes).
        let e1 = cache.put("k1", vec![img1.clone()], None, Vec::new(), 0, "m1");
        assert_eq!(e1.size_bytes, 300);
        assert_eq!(cache.size_bytes(), 300);

        let _e2 = cache.put("k2", vec![img2.clone()], None, Vec::new(), 0, "m2");
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.size_bytes(), 600);

        // Touch k1 so k2 becomes LRU.
        assert!(cache.get("k1").is_some());

        // Inserting k3 should evict k2.
        let _e3 = cache.put("k3", vec![img3], None, Vec::new(), 0, "m3");
        assert_eq!(cache.len(), 2);
        assert!(cache.get("k2").is_none());
        assert!(cache.get("k1").is_some());
        assert!(cache.get("k3").is_some());

        // Entry larger than max_bytes is returned without caching.
        let huge = Arc::new(RgbImage::from_pixel(20, 20, Rgb([0, 0, 0]))); // 1200 bytes > 600
        let eh = cache.put("huge", vec![huge], None, Vec::new(), 0, "mh");
        assert_eq!(eh.size_bytes, 1200);
        assert!(cache.get("huge").is_none());
        assert_eq!(cache.len(), 2);
    }
}
