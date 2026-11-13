//! Index operations a family with its own state objects needs beside the
//! block and sidecar paths: a byte budget over some groups (oldest first),
//! removal of one object, recency for a chain it still needs, a sidecar read
//! whose layout the caller checks, and a write that waits for a queue slot
//! and gives the object back when none freed.

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
    /// Bytes the index holds in `groups`.
    pub fn group_bytes(&self, groups: &[ColdGroup]) -> u64 {
        self.shared.index.lock().map_or(0, |index| {
            index
                .entries
                .values()
                .filter(|entry| groups.contains(&entry.group))
                .map(|entry| entry.size)
                .sum()
        })
    }

    /// The least recently used object of `groups`.
    pub fn oldest_in(&self, groups: &[ColdGroup]) -> Option<(ColdCacheKey, ColdGroup)> {
        let index = self.shared.index.lock().ok()?;
        index
            .entries
            .iter()
            .filter(|(_, entry)| groups.contains(&entry.group))
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(key, entry)| (*key, entry.group))
    }

    /// Removes the object `key` of `group`: true once its name is clear and
    /// the index no longer holds it.
    pub fn remove_object(&self, key: ColdCacheKey, group: ColdGroup) -> bool {
        let Ok(mut index) = self.shared.index.lock() else {
            return false;
        };
        let Some(entry) = index.entries.get(&key) else {
            return true;
        };
        if entry.group != group {
            return false;
        }
        let name = entry.file_name.clone();
        let cleared = match self.shared.root.stat_identity(&name) {
            Some((_, kind)) => clear_entry(&self.shared.root, &name, kind),
            None => entry_gone(self.shared.root.unlink(&name)),
        };
        if cleared && let Some(entry) = index.entries.remove(&key) {
            index.total_bytes = index.total_bytes.saturating_sub(entry.size);
        }
        cleared
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
            match self.sender.try_send(job) {
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

    fn state(
        manager: &ColdCacheManager,
        group: ColdGroup,
        seed: u32,
        bytes: usize,
    ) -> ColdCacheKey {
        let fingerprint = ColdCacheFingerprint::from_components([b"hybrid".as_slice()]);
        let key = ColdCacheKey::chain(group, fingerprint, None, &[seed; 32], &[], 0, 0);
        let sidecar = ColdSidecar {
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
        };
        assert_eq!(manager.enqueue_sidecar(sidecar), Ok(true));
        assert!(manager.drain(Duration::from_secs(10)));
        key
    }

    /// The budget's view of the index: bytes per group set, the oldest
    /// object first, a touch makes an object the newest, a removal frees its
    /// file and its bytes, a read of another group or a smaller bound is a
    /// miss. Mutation: `touch` leaves the recency alone → red (the touched
    /// object is still the oldest).
    #[test]
    fn a_state_budget_sees_its_groups_oldest_first() {
        let dir = root("budget");
        let manager = ColdCacheManager::open_at(dir.clone(), 1 << 30, 0, 4).unwrap();
        let groups = [ColdGroup::HybridCell, ColdGroup::HybridDraft];
        let first = state(&manager, ColdGroup::HybridCell, 1, 100);
        let second = state(&manager, ColdGroup::HybridDraft, 2, 200);
        let other = state(&manager, ColdGroup::GdnState, 3, 300);
        let total = manager.group_bytes(&groups);
        let gdn = manager.group_bytes(&[ColdGroup::GdnState]);
        assert!(total > 300 && gdn > 300, "{total} {gdn}");
        assert_eq!(
            manager.group_bytes(&[
                ColdGroup::HybridCell,
                ColdGroup::HybridDraft,
                ColdGroup::GdnState
            ]),
            total + gdn
        );
        assert_eq!(
            manager.oldest_in(&groups),
            Some((first, ColdGroup::HybridCell))
        );
        manager.touch(&[first]);
        assert_eq!(
            manager.oldest_in(&groups),
            Some((second, ColdGroup::HybridDraft))
        );
        let fingerprint = ColdCacheFingerprint::from_components([b"hybrid".as_slice()]);
        assert!(
            manager
                .load_object(first, fingerprint, ColdGroup::HybridCell, 1 << 20)
                .is_some()
        );
        assert!(
            manager
                .load_object(first, fingerprint, ColdGroup::HybridCell, 50)
                .is_none(),
            "a read past the bound is a miss"
        );
        assert!(!manager.contains_in(&first, ColdGroup::HybridCell));
        assert!(!manager.remove_object(other, ColdGroup::HybridCell));
        assert!(manager.remove_object(second, ColdGroup::HybridDraft));
        assert!(!manager.contains_in(&second, ColdGroup::HybridDraft));
        assert_eq!(manager.group_bytes(&groups), 0);
        assert!(manager.contains_in(&other, ColdGroup::GdnState));
        let _ = fs::remove_dir_all(dir);
    }
}
