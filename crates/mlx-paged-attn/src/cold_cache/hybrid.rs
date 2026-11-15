//! Index operations a family with its own state objects needs beside the
//! block and sidecar paths: removal of one object, recency for a chain it
//! still needs, a sidecar read whose layout the caller checks, and a write
//! that waits for a queue slot and gives the object back when none freed.

use super::*;

/// A block or a sidecar for the writer.
pub enum ColdObject {
    Block(ColdCacheBlock),
    Sidecar(ColdSidecar),
}

impl ColdObject {
    fn into_job(self) -> WriteJob {
        match self {
            Self::Block(block) => WriteJob::Block(block),
            Self::Sidecar(sidecar) => WriteJob::Sidecar(Box::new(sidecar)),
        }
    }

    fn from_job(job: WriteJob) -> Option<Self> {
        match job {
            WriteJob::Block(block) => Some(Self::Block(block)),
            WriteJob::Sidecar(sidecar) => Some(Self::Sidecar(*sidecar)),
            _ => None,
        }
    }
}

impl ColdCacheManager {
    /// Removes the object `key` of `group`: true once its name is clear and
    /// the index no longer holds it.
    pub fn remove_object(&self, key: ColdCacheKey, group: ColdGroup) -> bool {
        let Ok(index) = self.shared.index.lock() else {
            return false;
        };
        let Some(entry) = index.entries.get(&key) else {
            return true;
        };
        if entry.group != group {
            return false;
        }
        drop(index);
        self.remove(&key) || !self.contains_in(&key, group)
    }

    /// Marks `keys` as used now in the index, so the eviction takes them
    /// after every object used before.
    pub fn touch(&self, keys: &[ColdCacheKey]) {
        let tick = now_tick();
        if let Ok(mut index) = self.shared.index.lock() {
            for key in keys {
                if let Some(entry) = index.entries.get_mut(key) {
                    entry.last_access = tick;
                }
            }
        }
    }

    /// The sidecar `key` of `group`, read through the same checks as
    /// [`Self::load_sidecar`] (identity, ABI, checksum, at most `max_bytes`
    /// read) except the layout, which the caller compares itself. A failed
    /// read is a miss that removes the object.
    pub fn load_object(
        &self,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        group: ColdGroup,
        max_bytes: u64,
    ) -> Option<ColdSidecar> {
        if group == ColdGroup::Kv {
            return None;
        }
        self.load_object_bounded(key, group, max_bytes, |bytes| {
            decode_sidecar(bytes, key, fingerprint, group)
        })
    }

    /// Queues `object` for the writer, waiting until `deadline` for a slot;
    /// the object comes back when none freed.
    pub fn offer(
        &self,
        object: ColdObject,
        deadline: Instant,
    ) -> Result<Option<ColdObject>, String> {
        match &object {
            ColdObject::Block(block) => block.validate()?,
            ColdObject::Sidecar(sidecar) => sidecar.validate()?,
        }
        let mut job = object.into_job();
        loop {
            match self.try_send_block_lane(job).map_err(|refused| *refused) {
                Ok(()) => {
                    self.shared.stats.enqueued.fetch_add(1, Ordering::Relaxed);
                    return Ok(None);
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err("cold-cache writer stopped".to_string());
                }
                Err(TrySendError::Full(returned)) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        return Ok(ColdObject::from_job(returned));
                    }
                    job = returned;
                    std::thread::sleep(Duration::from_millis(1).min(remaining));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("mlx-hybrid-{tag}-{}-{nanos}", std::process::id()))
    }

    fn sidecar(group: ColdGroup, seed: u32, bytes: usize) -> ColdSidecar {
        let fingerprint = ColdCacheFingerprint::from_components([b"hybrid".as_slice()]);
        let key = ColdCacheKey::chain(group, fingerprint, None, &[seed; 32], &[], 0, 0);
        ColdSidecar {
            key,
            fingerprint,
            layout: ColdSidecarLayout {
                group,
                boundary_tokens: 32,
                num_layers: 1,
                tensors_per_layer: 1,
                dtype: "u8".to_string(),
                dims: vec![2, 1, 32],
                bytes_per_tensor: bytes,
            },
            tensors: vec![vec![seed as u8; bytes]],
        }
    }

    fn state(
        manager: &ColdCacheManager,
        group: ColdGroup,
        seed: u32,
        bytes: usize,
    ) -> ColdCacheKey {
        let sidecar = sidecar(group, seed, bytes);
        let key = sidecar.key;
        assert_eq!(manager.enqueue_sidecar(sidecar), Ok(true));
        assert!(manager.drain(Duration::from_secs(10)));
        key
    }

    /// An offered object goes through the block lane's count: once the
    /// writer took it the lane has room again, and the object is indexed.
    /// Mutation: `offer` sends on the raw channel → red (the writer's take
    /// underflows the count, so the lane never has room).
    #[test]
    fn an_offer_keeps_the_block_lane_count() {
        let dir = root("offer");
        let manager = ColdCacheManager::open_at(dir.clone(), 1 << 30, 0, 4).unwrap();
        let object = sidecar(ColdGroup::HybridDraft, 9, 64);
        let key = object.key;
        let deadline = Instant::now() + Duration::from_secs(10);
        assert!(matches!(
            manager.offer(ColdObject::Sidecar(object), deadline),
            Ok(None)
        ));
        assert!(manager.drain(Duration::from_secs(10)));
        assert!(manager.has_block_room());
        assert!(manager.contains_in(&key, ColdGroup::HybridDraft));
        let _ = fs::remove_dir_all(dir);
    }

    /// A family object under the store's one quota: a touch makes it the
    /// newest, so a write that needs room takes an object used before it; a
    /// read of another group or a smaller bound is a miss that removes it,
    /// and a removal frees its file and its bytes.
    /// Mutation: `touch` leaves the recency alone → red (the touched object
    /// is the one evicted).
    #[test]
    fn a_family_object_is_touched_read_and_removed() {
        let probe = root("probe");
        let one = {
            let manager = ColdCacheManager::open_at(probe.clone(), 1 << 30, 0, 4).unwrap();
            state(&manager, ColdGroup::HybridDraft, 9, 100);
            manager.total_bytes()
        };
        let _ = fs::remove_dir_all(probe);
        let dir = root("touch");
        let manager = ColdCacheManager::open_at(dir.clone(), one * 5 / 2, 0, 4).unwrap();
        let first = state(&manager, ColdGroup::HybridDraft, 1, 100);
        let second = state(&manager, ColdGroup::HybridDraft, 2, 100);
        manager.touch(&[first]);
        let third = state(&manager, ColdGroup::HybridDraft, 3, 100);
        assert!(manager.contains_in(&first, ColdGroup::HybridDraft));
        assert!(!manager.contains_in(&second, ColdGroup::HybridDraft));
        assert!(manager.contains_in(&third, ColdGroup::HybridDraft));
        let fingerprint = ColdCacheFingerprint::from_components([b"hybrid".as_slice()]);
        assert!(
            manager
                .load_object(first, fingerprint, ColdGroup::HybridDraft, 1 << 20)
                .is_some()
        );
        assert!(
            manager
                .load_object(first, fingerprint, ColdGroup::HybridDraft, 50)
                .is_none(),
            "a read past the bound is a miss"
        );
        assert!(!manager.contains_in(&first, ColdGroup::HybridDraft));
        let other = state(&manager, ColdGroup::ConvState, 4, 100);
        assert!(!manager.remove_object(other, ColdGroup::HybridDraft));
        let before = manager.total_bytes();
        assert!(manager.remove_object(third, ColdGroup::HybridDraft));
        assert!(!manager.contains_in(&third, ColdGroup::HybridDraft));
        assert!(manager.total_bytes() < before);
        assert!(manager.contains_in(&other, ColdGroup::ConvState));
        let _ = fs::remove_dir_all(dir);
    }
}
