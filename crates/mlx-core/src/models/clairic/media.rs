//! Media pixel digests, batch normalization, and warmup prefix grouping.
//!
//! Mirrors upstream `clairic_runtime/media.py` plus SHA-256 pixel digest helpers.

use std::{collections::HashMap, fmt::Write as _};

use image::RgbImage;
use sha2::{Digest, Sha256};

/// Minimum common prefix length for media warmup groups (upstream default).
pub const DEFAULT_MIN_PREFIX_TOKENS: usize = 128;

/// Format a 32-byte SHA-256 digest as lowercase hex.
pub(crate) fn hex_digest(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Compute a deterministic SHA-256 hex digest over an `RgbImage`'s dimensions and raw RGB bytes.
pub fn rgb_image_digest(image: &RgbImage) -> String {
    let mut hasher = Sha256::new();
    hasher.update(image.width().to_be_bytes());
    hasher.update(image.height().to_be_bytes());
    hasher.update(image.as_raw());
    hex_digest(&hasher.finalize())
}

/// Compute a deterministic SHA-256 hex digest over a sequence of `RgbImage`s.
pub fn rgb_images_digest(images: &[RgbImage]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((images.len() as u64).to_be_bytes());
    for img in images {
        hasher.update(img.width().to_be_bytes());
        hasher.update(img.height().to_be_bytes());
        hasher.update(img.as_raw());
    }
    hex_digest(&hasher.finalize())
}

/// Normalize a batch of optional media payloads matching `media.py::batch_image_data`.
///
/// If every item is `None`, returns `None` for every slot; otherwise replaces `None`
/// with an empty `Vec` so mixed text/vision batches have uniform list shapes.
pub fn batch_image_data<T: Clone>(media: &[Option<Vec<T>>]) -> Vec<Option<Vec<T>>> {
    if media.iter().all(Option::is_none) {
        return media.to_vec();
    }
    media
        .iter()
        .map(|item| Some(item.clone().unwrap_or_default()))
        .collect()
}

/// Identity of one prepared processor-output image item (matching `media.py::_media_identity`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreparedItemIdentity {
    pub clair_media_digest: String,
    pub pixel_values_id: usize,
    pub image_grid_thw: Vec<i32>,
}

/// Partitioning identity for a task's media payload (matching `media.py::_media_identity`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MediaIdentity {
    /// Text-only task (`media is None`).
    Text,
    /// Prepared processor outputs sharing digest, tensor identity, and grid shape.
    Prepared(Vec<PreparedItemIdentity>),
    /// Unrecognized or non-deduplicable media object keyed by unique handle/address.
    Opaque(usize),
}

/// One autoregressive task input to [`media_warmup_groups`].
#[derive(Debug, Clone)]
pub struct ArMediaTask<M> {
    pub name: String,
    pub token_ids: Vec<u32>,
    pub media_identity: MediaIdentity,
    pub media: M,
}

/// A shared-prefix warmup group emitted by [`media_warmup_groups`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaWarmupGroup<M> {
    pub prefix_ids: Vec<u32>,
    pub media: M,
    pub salt: String,
    pub group_prefix_ids: Vec<Vec<u32>>,
}

/// Compute the longest common prefix across a non-empty slice of token sequences.
///
/// Matches `media.py::_common_prefix` by comparing the lexicographically smallest
/// and largest sequences.
pub fn common_prefix(sequences: &[&[u32]]) -> Vec<u32> {
    let (Some(first), Some(last)) = (sequences.iter().min(), sequences.iter().max()) else {
        return Vec::new();
    };
    let len = first
        .iter()
        .zip(last.iter())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| first.len().min(last.len()));
    first[..len].to_vec()
}

/// Partition tasks by `(salt, media_identity)` and extract shared prefix warmup groups
/// and per-parent subgroup prefixes.
///
/// Exact port of `clairic_runtime/media.py::media_warmup_groups`.
pub fn media_warmup_groups<M: Clone>(
    tasks: &[ArMediaTask<M>],
    default_salt: &str,
    task_salts: &HashMap<String, String>,
    parent_annotations: &HashMap<String, String>,
    min_prefix_tokens: usize,
) -> Vec<MediaWarmupGroup<M>> {
    struct Partition<'a, M> {
        media: M,
        salt: String,
        members: Vec<&'a ArMediaTask<M>>,
    }

    let mut key_to_idx: HashMap<(String, MediaIdentity), usize> = HashMap::new();
    let mut partitions: Vec<Partition<'_, M>> = Vec::new();

    for task in tasks {
        let salt = task_salts
            .get(&task.name)
            .map(String::as_str)
            .unwrap_or(default_salt)
            .to_string();
        let key = (salt.clone(), task.media_identity.clone());
        if let Some(&idx) = key_to_idx.get(&key) {
            partitions[idx].members.push(task);
        } else {
            let idx = partitions.len();
            key_to_idx.insert(key, idx);
            partitions.push(Partition {
                media: task.media.clone(),
                salt,
                members: vec![task],
            });
        }
    }

    let mut records = Vec::new();
    for partition in partitions {
        if partition.members.len() < 2 {
            continue;
        }
        let member_slices: Vec<&[u32]> = partition
            .members
            .iter()
            .map(|t| t.token_ids.as_slice())
            .collect();
        let mut prefix = common_prefix(&member_slices);
        if prefix.len() < min_prefix_tokens {
            prefix.clear();
        }

        // Preserve insertion order of parents (matching Python dict).
        let mut parent_to_idx: HashMap<&str, usize> = HashMap::new();
        let mut parent_groups: Vec<Vec<&[u32]>> = Vec::new();
        for task in &partition.members {
            if let Some(parent) = parent_annotations.get(&task.name)
                && !parent.is_empty()
            {
                if let Some(&idx) = parent_to_idx.get(parent.as_str()) {
                    parent_groups[idx].push(task.token_ids.as_slice());
                } else {
                    let idx = parent_groups.len();
                    parent_to_idx.insert(parent.as_str(), idx);
                    parent_groups.push(vec![task.token_ids.as_slice()]);
                }
            }
        }

        let subgroups: Vec<Vec<u32>> = parent_groups
            .into_iter()
            .filter(|group| group.len() > 1)
            .map(|group| common_prefix(&group))
            .filter(|group| !group.is_empty() && *group != prefix)
            .collect();

        if !prefix.is_empty() || !subgroups.is_empty() {
            records.push(MediaWarmupGroup {
                prefix_ids: prefix,
                media: partition.media,
                salt: partition.salt,
                group_prefix_ids: subgroups,
            });
        }
    }

    records
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn pixel_digests_are_deterministic_and_sensitive() {
        let a = RgbImage::from_pixel(4, 4, Rgb([10, 20, 30]));
        let b = RgbImage::from_pixel(4, 4, Rgb([10, 20, 30]));
        let c = RgbImage::from_pixel(4, 4, Rgb([10, 20, 31]));
        let d = RgbImage::from_pixel(8, 2, Rgb([10, 20, 30]));
        assert_eq!(rgb_image_digest(&a), rgb_image_digest(&b));
        assert_ne!(rgb_image_digest(&a), rgb_image_digest(&c));
        assert_ne!(rgb_image_digest(&a), rgb_image_digest(&d));
        assert_eq!(rgb_image_digest(&a).len(), 64);
    }

    #[test]
    fn batch_image_data_matches_upstream() {
        let all_none: Vec<Option<Vec<u32>>> = vec![None, None];
        assert_eq!(batch_image_data(&all_none), vec![None, None]);

        let mixed = vec![Some(vec![1u32, 2]), None];
        assert_eq!(
            batch_image_data(&mixed),
            vec![Some(vec![1u32, 2]), Some(vec![])]
        );
    }

    #[test]
    fn warmup_groups_partition_and_find_parent_subgroups() {
        let id = MediaIdentity::Prepared(vec![PreparedItemIdentity {
            clair_media_digest: "abc".to_string(),
            pixel_values_id: 42,
            image_grid_thw: vec![1, 4, 4],
        }]);
        let tasks = vec![
            ArMediaTask {
                name: "items.0".to_string(),
                token_ids: vec![1, 2, 3, 4, 10, 11],
                media_identity: id.clone(),
                media: "m1",
            },
            ArMediaTask {
                name: "items.1".to_string(),
                token_ids: vec![1, 2, 3, 4, 10, 12],
                media_identity: id.clone(),
                media: "m1",
            },
            ArMediaTask {
                name: "other".to_string(),
                token_ids: vec![1, 2, 3, 4, 99, 100],
                media_identity: id,
                media: "m1",
            },
        ];
        let mut parents = HashMap::new();
        parents.insert("items.0".to_string(), "items".to_string());
        parents.insert("items.1".to_string(), "items".to_string());

        let groups = media_warmup_groups(&tasks, "salt0", &HashMap::new(), &parents, 4);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].prefix_ids, vec![1, 2, 3, 4]);
        assert_eq!(groups[0].group_prefix_ids, vec![vec![1, 2, 3, 4, 10]]);
        assert_eq!(groups[0].salt, "salt0");
    }
}
