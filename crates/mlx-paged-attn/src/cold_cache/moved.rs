//! What a load that moves another store's objects into its own needs of the
//! manager and of the source's reader: a tracked write of a block made on
//! the host, a read back that leaves the object's time alone, an object's
//! time set to its source's, room made only among objects older than a
//! given time, a device flush before a source object goes, and that
//! object's removal.

use std::collections::HashSet;
use std::fs::FileTimes;
use std::io::Read;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant, UNIX_EPOCH};

use sha2::Digest;

use super::source_reader::block_bound;
use super::{
    ColdCacheBlock, ColdCacheFingerprint, ColdCacheKey, ColdCacheLayout, ColdCacheManager,
    ColdGroup, ColdSourceReader, ColdStateLabel, ColdStateLayout, Landed, MakeRoom, ObjectRead,
    STATE_LABEL_HEADER_CAP, TrackedBlock, WriteJob, clear_entry, decode_block, entry_gone,
    held_keys, hex_encode, object_file_name, read_state_header, state_checksum_prefix,
};

/// Bytes of a state payload checked per read.
const STATE_READ_CHUNK: usize = 1 << 20;

/// What a source directory holds under a block's key.
#[derive(Debug)]
pub enum ColdSourceBlock {
    Read(Box<ColdCacheBlock>),
    /// No object of that name.
    Absent,
    /// An object a restore's decode refuses: it comes back for nobody.
    Damaged,
    /// An object that is sound, or could not be judged, and is not a block
    /// of the expected layout: nothing is known about what it holds.
    Foreign,
}

impl ColdSourceReader {
    /// The KV block for `key` as a restore decodes it, with the reason when
    /// it does not come back as a block of `expected`.
    pub fn probe_block(
        &self,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        expected: &ColdCacheLayout,
    ) -> ColdSourceBlock {
        let Some(bound) = block_bound(expected) else {
            return ColdSourceBlock::Foreign;
        };
        let file = match self
            .root()
            .open_existing(&object_file_name(&key, ColdGroup::Kv))
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return ColdSourceBlock::Absent;
            }
            Err(_) => return ColdSourceBlock::Foreign,
        };
        let mut bytes = Vec::new();
        if file
            .take(bound.saturating_add(1))
            .read_to_end(&mut bytes)
            .is_err()
            || bytes.len() as u64 > bound
        {
            return ColdSourceBlock::Foreign;
        }
        match decode_block(&bytes, key, fingerprint) {
            Ok(block) if block.layout == *expected => ColdSourceBlock::Read(Box::new(block)),
            Ok(_) => ColdSourceBlock::Foreign,
            Err(_) => ColdSourceBlock::Damaged,
        }
    }

    /// Takes the object for `key` out of the directory, the one call of a
    /// reader that changes it: only for an object whose rows another store
    /// holds. True when no object of that name is left.
    pub fn remove_moved(&self, key: ColdCacheKey, group: ColdGroup) -> bool {
        entry_gone(self.root().unlink(&object_file_name(&key, group)))
    }
}

impl ColdCacheManager {
    /// [`Self::capture_and_enqueue_tracked`] for a block made on the host:
    /// the writer reports it as [`Landed::Block`] on `done`. `Ok(false)`: no
    /// queue slot until `deadline`, a `queue_drops`, nothing is written.
    pub fn enqueue_tracked(
        &self,
        block: ColdCacheBlock,
        room: MakeRoom,
        deadline: Instant,
        done: Sender<Landed>,
    ) -> Result<bool, String> {
        block.validate()?;
        if !self.takes_tracked_writes() {
            return Err("cold-cache tracked writes stopped after a failed write".to_string());
        }
        if !self.wait_block_room(deadline)? {
            self.shared
                .stats
                .queue_drops
                .fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        self.send_before(
            WriteJob::Tracked(Box::new(TrackedBlock { block, room, done })),
            deadline,
        )
    }

    /// The block for `key` as a restore decodes it, its time and the reuse
    /// counters left alone; `None` unless its layout is `expected`. A
    /// damaged object goes, as after any failed load.
    pub fn read_back(
        &self,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        expected: &ColdCacheLayout,
    ) -> Option<ColdCacheBlock> {
        self.read_block_of(key, fingerprint, expected, false)
    }

    /// [`Self::read_back`] as a restore's use of the block: its time moves
    /// to now, as a restored block's does, so it leaves the tier no sooner
    /// than the blocks restored with it, and it counts as a hit.
    pub fn load_with_layout(
        &self,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        expected: &ColdCacheLayout,
    ) -> Option<ColdCacheBlock> {
        let block = self.read_block_of(key, fingerprint, expected, true)?;
        self.shared.stats.hits.fetch_add(1, Ordering::Relaxed);
        self.shared
            .stats
            .bytes_restored
            .fetch_add(block.encoded_len(), Ordering::Relaxed);
        Some(block)
    }

    fn read_block_of(
        &self,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        expected: &ColdCacheLayout,
        touch: bool,
    ) -> Option<ColdCacheBlock> {
        let bound = block_bound(expected)?;
        self.load_object_with(key, ColdGroup::Kv, touch, |file| {
            let mut bytes = Vec::new();
            if file
                .take(bound.saturating_add(1))
                .read_to_end(&mut bytes)
                .is_err()
                || bytes.len() as u64 > bound
            {
                return ObjectRead::Corrupt;
            }
            match decode_block(&bytes, key, fingerprint) {
                Ok(block) if block.layout == *expected => ObjectRead::Ok(block),
                Ok(_) => ObjectRead::Mismatch,
                Err(_) => ObjectRead::Corrupt,
            }
        })
    }

    /// The state object for `key` carries `layout` and `label`, its bytes
    /// hash to its header's checksum and, with `payload`, are that payload.
    /// Its time is left alone; a damaged object goes.
    pub fn state_reads_back(
        &self,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        layout: &ColdStateLayout,
        label: &ColdStateLabel,
        payload: Option<&[u8]>,
    ) -> bool {
        self.load_object_with(key, ColdGroup::DFlash2State, false, |file| {
            let Ok(header) = read_state_header(file, key, fingerprint, STATE_LABEL_HEADER_CAP)
            else {
                return ObjectRead::Corrupt;
            };
            if header.layout != *layout || header.label != *label {
                return ObjectRead::Mismatch;
            }
            let Some(bytes) = header
                .layout
                .payload_bytes()
                .and_then(|bytes| usize::try_from(bytes).ok())
            else {
                return ObjectRead::Corrupt;
            };
            if payload.is_some_and(|payload| payload.len() != bytes) {
                return ObjectRead::Mismatch;
            }
            let mut hasher = state_checksum_prefix(&header.layout, &header.label);
            let mut chunk = vec![0u8; STATE_READ_CHUNK.min(bytes.max(1))];
            let mut at = 0;
            let mut same = true;
            while at < bytes {
                let want = chunk.len().min(bytes - at);
                if file.read_exact(&mut chunk[..want]).is_err() {
                    return ObjectRead::Corrupt;
                }
                hasher.update(&chunk[..want]);
                same &= payload.is_none_or(|payload| payload[at..at + want] == chunk[..want]);
                at += want;
            }
            let mut tail = [0u8; 1];
            if !matches!(file.read(&mut tail), Ok(0))
                || hex_encode(&hasher.finalize()) != header.checksum
            {
                return ObjectRead::Corrupt;
            }
            if same {
                ObjectRead::Ok(())
            } else {
                ObjectRead::Mismatch
            }
        })
        .is_some()
    }

    /// When `key`'s object was last used, in nanoseconds since the Unix
    /// epoch: what eviction orders it by.
    pub fn modified(&self, key: &ColdCacheKey) -> Option<u128> {
        self.shared
            .index
            .lock()
            .ok()?
            .entries
            .get(key)
            .map(|entry| entry.last_access)
    }

    /// Sets the time of `key`'s object, on disk and in the index, to
    /// `modified` nanoseconds since the Unix epoch: an object moved from
    /// another store keeps its place in the eviction order.
    pub fn set_modified(&self, key: &ColdCacheKey, modified: u128) -> bool {
        let Ok(mut index) = self.shared.index.lock() else {
            return false;
        };
        let Some(entry) = index.entries.get_mut(key) else {
            return false;
        };
        let Ok(nanos) = u64::try_from(modified) else {
            return false;
        };
        let Ok(file) = self.shared.root.open_existing(&entry.file_name) else {
            return false;
        };
        let when = UNIX_EPOCH + Duration::from_nanos(nanos);
        if file.set_times(FileTimes::new().set_modified(when)).is_err() {
            return false;
        }
        entry.last_access = modified;
        true
    }

    /// Makes room under the quota for `incoming` more bytes by clearing
    /// objects last used before `older_than`, oldest first, none of `keep`
    /// and none a restore holds. Nothing goes unless enough of them can;
    /// true when `incoming` fits afterwards.
    pub fn make_room_among_older(
        &self,
        incoming: u64,
        older_than: u128,
        keep: &HashSet<ColdCacheKey>,
    ) -> bool {
        let Ok(mut index) = self.shared.index.lock() else {
            return false;
        };
        let quota = self.shared.quota_bytes;
        let fits = |taken: u64| taken.saturating_add(incoming) <= quota;
        if fits(index.total_bytes) {
            return true;
        }
        let held: HashSet<ColdCacheKey> = held_keys(&self.shared).into_iter().collect();
        let mut older: Vec<(u128, ColdCacheKey, u64)> = index
            .entries
            .iter()
            .filter(|&(key, entry)| {
                entry.last_access < older_than && !keep.contains(key) && !held.contains(key)
            })
            .map(|(key, entry)| (entry.last_access, *key, entry.size))
            .collect();
        let reclaimable = older
            .iter()
            .fold(0u64, |sum, &(_, _, size)| sum.saturating_add(size));
        if !fits(index.total_bytes.saturating_sub(reclaimable)) {
            return false;
        }
        older.sort_unstable_by_key(|&(used, key, _)| (used, *key.as_bytes()));
        for (_, key, _) in older {
            if fits(index.total_bytes) {
                break;
            }
            let Some(name) = index.entries.get(&key).map(|entry| entry.file_name.clone()) else {
                continue;
            };
            let cleared = match self.shared.root.stat_identity(&name) {
                Some((_, kind)) => clear_entry(&self.shared.root, &name, kind),
                None => entry_gone(self.shared.root.unlink(&name)),
            };
            if !cleared {
                continue;
            }
            if let Some(entry) = index.entries.remove(&key) {
                index.total_bytes = index.total_bytes.saturating_sub(entry.size);
                self.shared.stats.evictions.fetch_add(1, Ordering::Relaxed);
                if entry.group == ColdGroup::DFlash2State
                    && let Ok(mut evicted) = self.shared.evicted.lock()
                {
                    evicted.push(key);
                }
            }
        }
        fits(index.total_bytes)
    }

    /// Asks the volume's device to put on permanent storage what the writer
    /// handed it. A landed write survives a process death, not a power cut
    /// (`sync_payload`): an object that replaces the only other copy of its
    /// rows needs this first.
    pub fn flush_device(&self) -> bool {
        #[cfg(target_vendor = "apple")]
        if rustix::fs::fcntl_fullfsync(&self.shared.root.fd).is_ok() {
            return true;
        }
        self.shared.sync().is_ok()
    }
}

#[cfg(test)]
mod tests;
