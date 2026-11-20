//! Read-only access to a cache directory another load wrote. Unlike
//! [`ColdCacheManager`](super::ColdCacheManager) it keeps no index, removes
//! no writer temp file, prunes no damaged or foreign object and touches no
//! file's time: the directory is byte-for-byte the same after any read.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use sha2::Digest;

use super::{
    ColdCacheBlock, ColdCacheFingerprint, ColdCacheKey, ColdCacheLayout, ColdGroup, ColdStateLabel,
    ColdStateLayout, RootDir, STATE_LABEL_HEADER_CAP, decode_block, header_overhead, hex_encode,
    object_file_name, parse_object_name, read_state_header, state_checksum_prefix,
};

/// One canonical object of a source directory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColdSourceEntry {
    pub key: ColdCacheKey,
    pub group: ColdGroup,
    pub bytes: u64,
    /// Modification time, nanoseconds since the Unix epoch.
    pub modified: u128,
}

/// A cache directory opened for reading only.
pub struct ColdSourceReader {
    root: RootDir,
}

impl ColdSourceReader {
    /// Opens `dir`, which must exist, be a directory (not a symlink) and be
    /// owned by this user; nothing is created.
    pub fn open(dir: &Path) -> Result<Self, String> {
        Ok(Self {
            root: open_read_only(dir)?,
        })
    }

    pub fn path(&self) -> &Path {
        &self.root.path
    }

    /// Every canonical object, by file name; writer temps and foreign files
    /// are left out and left alone.
    pub fn entries(&self) -> Result<Vec<ColdSourceEntry>, String> {
        let mut names = self.root.entry_names()?;
        names.sort();
        Ok(names
            .iter()
            .filter_map(|name| {
                let (key, group) = parse_object_name(name)?;
                let (bytes, modified) = self.root.stat_file(name)?;
                Some(ColdSourceEntry {
                    key,
                    group,
                    bytes,
                    modified,
                })
            })
            .collect())
    }

    /// The layout and label of the state object for `key` from its header
    /// alone; the file's size must be exactly header + spans.
    pub fn state_label(
        &self,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
    ) -> Option<(ColdStateLayout, ColdStateLabel)> {
        self.read_object(key, ColdGroup::DFlash2State, |file| {
            let header = read_state_header(file, key, fingerprint, STATE_LABEL_HEADER_CAP).ok()?;
            let expected = header
                .layout
                .payload_bytes()?
                .checked_add(header.header_len)?;
            (file.metadata().ok()?.len() == expected).then_some((header.layout, header.label))
        })
    }

    /// The state object for `key`, its payload read into `into` and checked
    /// against the header's checksum. On `None` the contents of `into` are
    /// unspecified.
    pub fn read_state(
        &self,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        into: &mut Vec<u8>,
    ) -> Option<(ColdStateLayout, ColdStateLabel)> {
        self.read_object(key, ColdGroup::DFlash2State, |file| {
            let header = read_state_header(file, key, fingerprint, STATE_LABEL_HEADER_CAP).ok()?;
            let payload = header.layout.payload_bytes()?;
            if file.metadata().ok()?.len() != payload.checked_add(header.header_len)? {
                return None;
            }
            into.clear();
            into.resize(usize::try_from(payload).ok()?, 0);
            let mut tail = [0u8; 1];
            if file.read_exact(into).is_err() || !matches!(file.read(&mut tail), Ok(0)) {
                return None;
            }
            let mut hasher = state_checksum_prefix(&header.layout, &header.label);
            hasher.update(&*into);
            (hex_encode(&hasher.finalize()) == header.checksum)
                .then_some((header.layout, header.label))
        })
    }

    /// The KV block for `key`, decoded and checked as a restore checks it;
    /// `None` unless its layout is exactly `expected`.
    pub fn read_block(
        &self,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        expected: &ColdCacheLayout,
    ) -> Option<ColdCacheBlock> {
        let bound = block_bound(expected)?;
        self.read_object(key, ColdGroup::Kv, |file| {
            let mut bytes = Vec::new();
            file.take(bound.saturating_add(1))
                .read_to_end(&mut bytes)
                .ok()?;
            if bytes.len() as u64 > bound {
                return None;
            }
            decode_block(&bytes, key, fingerprint)
                .ok()
                .filter(|block| block.layout == *expected)
        })
    }

    fn read_object<T>(
        &self,
        key: ColdCacheKey,
        group: ColdGroup,
        read: impl FnOnce(&mut File) -> Option<T>,
    ) -> Option<T> {
        let name = object_file_name(&key, group);
        let mut file = self.root.open_existing(&name).ok()?;
        read(&mut file)
    }

    pub(super) fn root(&self) -> &RootDir {
        &self.root
    }
}

/// The encoded size of a block of `layout` (`ColdCacheBlock::encoded_len`).
pub(super) fn block_bound(layout: &ColdCacheLayout) -> Option<u64> {
    let layer =
        (layout.key_bytes_per_layer as u64).checked_add(layout.value_bytes_per_layer as u64)?;
    layer
        .checked_mul(u64::from(layout.num_layers))?
        .checked_add(u64::from(layout.block_size).checked_mul(4)?)?
        .checked_add(header_overhead(u64::from(layout.num_layers)))
}

#[cfg(unix)]
fn open_read_only(dir: &Path) -> Result<RootDir, String> {
    use rustix::fs::{Mode, OFlags, fstat, open};
    let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let fd = open(dir, flags, Mode::empty())
        .map_err(|e| format!("open cold-cache source {}: {e}", dir.display()))?;
    let stat = fstat(&fd).map_err(|e| format!("stat cold-cache source: {e}"))?;
    // SAFETY: geteuid has no preconditions and cannot fail.
    if stat.st_uid != unsafe { libc::geteuid() } {
        return Err("cold-cache source is not owned by the current user".to_string());
    }
    Ok(RootDir {
        path: dir.to_path_buf(),
        fd,
    })
}

#[cfg(not(unix))]
fn open_read_only(dir: &Path) -> Result<RootDir, String> {
    match std::fs::symlink_metadata(dir) {
        Ok(meta) if meta.is_dir() && !meta.file_type().is_symlink() => Ok(RootDir {
            path: dir.to_path_buf(),
        }),
        _ => Err(format!(
            "cold-cache source {} is not a plain directory",
            dir.display()
        )),
    }
}

#[cfg(test)]
mod tests;
