//! The open of a tier that moves another namespace's objects into its own:
//! the source namespace is held from the open on, so no start's quota
//! measure or 14-day rule removes it while its objects move.

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(unix)]
use {
    super::{
        CacheDisk, Taken, idle_caches, named, namespace_name, open_tier, owned_by_me, remove_stale,
        take_lock, usage,
    },
    std::fs,
    std::path::Path,
    std::time::SystemTime,
};

use super::{DiskTier, DiskTierOptions, PersistentIdentity};

/// Another namespace of the cache root whose objects this load moves into
/// its own, with its lock taken.
#[cfg_attr(not(unix), allow(dead_code))]
pub(crate) struct SourceHold {
    pub dir: PathBuf,
    pub identity: PersistentIdentity,
    /// Its objects' bytes when the hold was taken.
    pub bytes: u64,
    _lock: File,
}

#[cfg(unix)]
impl DiskTier {
    /// `read` of the held source namespace; `None` when this tier holds none.
    pub(crate) fn with_source<R>(&self, read: impl FnOnce(&SourceHold) -> R) -> Option<R> {
        self.source
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map(read)
    }

    /// Removes the held source namespace under its lock, then lets the lock
    /// go. False when this tier holds none, the lock no longer names the
    /// directory, or the directory could not go; the hold ends either way.
    pub(crate) fn remove_source(&self) -> bool {
        let Some(hold) = self
            .source
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        else {
            return false;
        };
        named(&hold.dir, &hold._lock) && fs::remove_dir_all(&hold.dir).is_ok()
    }
}

/// [`super::open_disk_tier`] for a load that moves the objects of the
/// namespace `source` names into its own: once its own persistent tier is
/// open, and before any cache of the root is measured or aged out, the
/// source's lock is taken and kept in the tier ([`DiskTier::with_source`])
/// and its time set to now, so neither this start nor any other removes it.
/// `source` runs only for a persistent tier; a source nobody made, or one
/// another process holds, is not held.
#[cfg(unix)]
pub(crate) fn open_disk_tier_with_source(
    options: &DiskTierOptions,
    identity: impl FnOnce() -> Option<PersistentIdentity>,
    source: impl FnOnce() -> Option<PersistentIdentity>,
    state_bytes: u64,
) -> Option<Arc<DiskTier>> {
    let root = options.cache_root();
    let mut caches = root.as_deref().map_or_else(Vec::new, idle_caches);
    let tier = (options.cache_disk != CacheDisk::Off)
        .then(|| open_tier(options, identity, state_bytes, &caches))
        .flatten();
    if let (Some(tier), Some(root)) = (tier.as_ref().filter(|tier| tier.persistent), root)
        && let Some(source) = source()
    {
        let dir = root.join(&source.namespace);
        let own = source.namespace == tier.namespace;
        let held = hold_source(&root, source, &tier.namespace);
        // A source another process locked after the scan is not idle
        // either; one with no lock stays in the list and ages out.
        if !own && !matches!(held, Held::Not) {
            caches.retain(|cache| cache.path != dir);
        }
        if let Held::Mine(hold) = held {
            *tier
                .source
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(hold);
        }
    }
    let mut others = remove_stale(caches);
    if let Some(tier) = tier.as_ref().filter(|tier| tier.persistent) {
        others.retain(|cache| cache.path != tier.manager.root());
        others.sort_by_key(|cache| cache.last_use);
        *tier
            .others
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = others.into();
        tier.measure();
    }
    tier
}

#[cfg(unix)]
enum Held {
    Mine(SourceHold),
    /// Someone else holds its lock.
    Other,
    Not,
}

/// The source namespace `identity` names under `root`, its lock taken and
/// its time set to now; `Not` when it is this tier's own, is no directory
/// of this user, or has no lock.
#[cfg(unix)]
fn hold_source(root: &Path, identity: PersistentIdentity, own: &str) -> Held {
    if identity.namespace == own || !namespace_name(&identity.namespace) {
        return Held::Not;
    }
    let dir = root.join(&identity.namespace);
    let Ok(meta) = fs::symlink_metadata(&dir) else {
        return Held::Not;
    };
    if !meta.is_dir() || !owned_by_me(&meta) {
        return Held::Not;
    }
    let lock = match take_lock(&dir) {
        Taken::Lock(lock) => lock,
        Taken::Held => return Held::Other,
        Taken::NoLock => return Held::Not,
    };
    let _ = lock.set_modified(SystemTime::now());
    let (bytes, _) = usage(&dir);
    Held::Mine(SourceHold {
        dir,
        identity,
        bytes,
        _lock: lock,
    })
}

#[cfg(windows)]
impl DiskTier {
    pub(crate) fn with_source<R>(&self, _read: impl FnOnce(&SourceHold) -> R) -> Option<R> {
        None
    }

    pub(crate) fn remove_source(&self) -> bool {
        false
    }
}

#[cfg(windows)]
pub(crate) fn open_disk_tier_with_source(
    options: &DiskTierOptions,
    identity: impl FnOnce() -> Option<PersistentIdentity>,
    _source: impl FnOnce() -> Option<PersistentIdentity>,
    state_bytes: u64,
) -> Option<Arc<DiskTier>> {
    super::open_disk_tier(options, identity, state_bytes)
}

#[cfg(all(test, unix))]
mod tests;
