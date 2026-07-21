//! Persistent SSD-backed cold tier for immutable PagedAttention prefix blocks.
//!
//! The hot allocator remains authoritative. This module stores only complete,
//! immutable blocks and restores them transactionally: bytes are validated and
//! uploaded into a reserved physical slot before the prefix is published.
//! Every I/O error is a cache miss, never an inference failure.
//!
//! On unix the cache root is held as an `O_DIRECTORY` descriptor acquired by
//! a no-follow component walk, and every mutating filesystem operation is
//! descriptor-relative, so a pathname replaced with a symlink can never
//! redirect cache I/O. Non-unix platforms keep path-based operations behind a
//! static pre-open symlink check (no-follow hardening is unix-only, matching
//! the supported platforms).

use std::collections::HashMap;
use std::fmt;
#[cfg(not(unix))]
use std::fs::OpenOptions;
use std::fs::{self, File};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use safetensors::tensor::{Dtype, TensorView};
use safetensors::{SafeTensors, serialize};
use sha2::{Digest, Sha256};

use crate::{BlockAllocator, LayerKVPool, PhysicalBlock};

const CACHE_ABI: &str = "mlx-paged-v1";
const DEFAULT_QUEUE_DEPTH: usize = 8;
const GIB: u64 = 1024 * 1024 * 1024;
const MAX_DEFAULT_QUOTA: u64 = 100 * GIB;
const MIN_FREE_RESERVE: u64 = 5 * GIB;

/// Stable model/cache identity. Callers should hash exact weight shards plus
/// tokenizer/template, quantization, RoPE/MTP, and cache-layout components.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct ColdCacheFingerprint([u8; 32]);

impl ColdCacheFingerprint {
    /// Domain-separated SHA-256 over length-prefixed components.
    pub fn from_components<'a>(components: impl IntoIterator<Item = &'a [u8]>) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"mlx-node:cold-cache-fingerprint:v1\0");
        for component in components {
            hasher.update((component.len() as u64).to_le_bytes());
            hasher.update(component);
        }
        Self(hasher.finalize().into())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        hex_encode(&self.0)
    }
}

impl fmt::Debug for ColdCacheFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ColdCacheFingerprint")
            .field(&self.to_hex())
            .finish()
    }
}

/// Stable, collision-resistant chained key for one logical prefix block.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct ColdCacheKey([u8; 32]);

impl ColdCacheKey {
    /// Build a block key. `parent` is `None` for the first block and the
    /// preceding block key thereafter. Integer encoding is explicitly LE so
    /// the key is stable across processes and Rust versions.
    pub fn chain(
        fingerprint: ColdCacheFingerprint,
        parent: Option<Self>,
        tokens: &[u32],
        extra_keys: &[u64],
        cache_salt: u64,
        block_index: usize,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"mlx-node:cold-prefix-block:v1\0");
        hasher.update(fingerprint.as_bytes());
        hasher.update(parent.map_or([0u8; 32], |key| key.0));
        hasher.update((block_index as u64).to_le_bytes());
        hasher.update((tokens.len() as u64).to_le_bytes());
        for token in tokens {
            hasher.update(token.to_le_bytes());
        }
        hasher.update((extra_keys.len() as u64).to_le_bytes());
        for key in extra_keys {
            hasher.update(key.to_le_bytes());
        }
        // Match the hot-cache contract: salt isolates only block zero.
        hasher.update(if block_index == 0 { cache_salt } else { 0 }.to_le_bytes());
        Self(hasher.finalize().into())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        hex_encode(&self.0)
    }

    fn from_hex(value: &str) -> Option<Self> {
        let bytes = hex_decode_32(value)?;
        Some(Self(bytes))
    }
}

impl fmt::Debug for ColdCacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ColdCacheKey").field(&self.to_hex()).finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColdCacheLayout {
    pub block_size: u32,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_size: u32,
    pub cache_dtype: String,
    pub key_bytes_per_layer: usize,
    pub value_bytes_per_layer: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColdLayerBlock {
    pub keys: Vec<u8>,
    pub values: Vec<u8>,
}

/// Owned host representation of one complete physical block across all layers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColdCacheBlock {
    pub key: ColdCacheKey,
    pub fingerprint: ColdCacheFingerprint,
    pub tokens: Vec<u32>,
    pub layout: ColdCacheLayout,
    pub layers: Vec<ColdLayerBlock>,
}

impl ColdCacheBlock {
    fn validate(&self) -> Result<(), String> {
        if self.tokens.len() != self.layout.block_size as usize {
            return Err("cold cache accepts immutable full blocks only".to_string());
        }
        if self.layers.len() != self.layout.num_layers as usize {
            return Err("cold-cache layer count does not match layout".to_string());
        }
        for layer in &self.layers {
            if layer.keys.len() != self.layout.key_bytes_per_layer
                || layer.values.len() != self.layout.value_bytes_per_layer
            {
                return Err("cold-cache layer byte length does not match layout".to_string());
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> u64 {
        self.layers
            .iter()
            .map(|layer| (layer.keys.len() + layer.values.len()) as u64)
            .sum::<u64>()
            + (self.tokens.len() * size_of::<u32>()) as u64
            + 4096
    }
}

#[derive(Clone, Debug)]
pub struct RestorePrefixIdentity {
    pub hot_hash: u64,
    pub tokens: Vec<u32>,
    pub parent_hot_hash: u64,
    pub extra_keys: Vec<u64>,
    pub cache_salt: u64,
    pub block_index: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ColdCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub enqueued: u64,
    pub queue_drops: u64,
    pub bytes_written: u64,
    pub bytes_restored: u64,
    pub evictions: u64,
    pub corruptions: u64,
}

#[derive(Default)]
struct AtomicStats {
    hits: AtomicU64,
    misses: AtomicU64,
    enqueued: AtomicU64,
    queue_drops: AtomicU64,
    bytes_written: AtomicU64,
    bytes_restored: AtomicU64,
    evictions: AtomicU64,
    corruptions: AtomicU64,
}

impl AtomicStats {
    fn snapshot(&self) -> ColdCacheStats {
        let load = |value: &AtomicU64| value.load(Ordering::Relaxed);
        ColdCacheStats {
            hits: load(&self.hits),
            misses: load(&self.misses),
            enqueued: load(&self.enqueued),
            queue_drops: load(&self.queue_drops),
            bytes_written: load(&self.bytes_written),
            bytes_restored: load(&self.bytes_restored),
            evictions: load(&self.evictions),
            corruptions: load(&self.corruptions),
        }
    }
}

#[derive(Clone, Debug)]
struct IndexEntry {
    file_name: String,
    size: u64,
    last_access: u128,
}

#[derive(Default)]
struct CacheIndex {
    entries: HashMap<ColdCacheKey, IndexEntry>,
    total_bytes: u64,
}

/// Handle to the cache root directory. On unix it owns the directory file
/// descriptor from the no-follow opener and performs every mutating
/// operation relative to that descriptor (`openat`/`renameat`/`unlinkat`/
/// `fchmod`/`fsync`), so replacing the root pathname after open cannot
/// redirect writes, eviction, or cleanup. Non-unix stores only the path and
/// keeps the previous path-based operations.
struct RootDir {
    path: PathBuf,
    #[cfg(unix)]
    fd: OwnedFd,
}

#[cfg(unix)]
impl RootDir {
    /// Secure opener: absolutize `root`, absolutely open its deepest
    /// existing strict ancestor (the caller-trusted base), then walk every
    /// remaining component with `O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC`,
    /// creating missing ones with `mkdirat` mode 0700. The final directory
    /// must be owned by the current effective uid. Any symlink at or below
    /// the first walked component is refused.
    fn open_at_path(root: PathBuf) -> Result<Self, String> {
        let absolute =
            std::path::absolute(&root).map_err(|e| format!("resolve cold-cache root: {e}"))?;
        let Some(parent) = absolute.parent() else {
            return Err("cold-cache root must not be a filesystem root".to_string());
        };
        let mut anchor = parent;
        while fs::symlink_metadata(anchor).is_err() {
            anchor = anchor
                .parent()
                .ok_or_else(|| "cold-cache root has no existing ancestor".to_string())?;
        }
        let rel = absolute
            .strip_prefix(anchor)
            .expect("anchor is a lexical ancestor")
            .to_path_buf();
        Self::open_beneath(anchor, &rel, root)
    }

    fn open_beneath(anchor: &Path, rel: &Path, display: PathBuf) -> Result<Self, String> {
        use rustix::fs::{Mode, OFlags, mkdirat, open, openat};
        let dir_flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC;
        let no_follow = dir_flags | OFlags::NOFOLLOW;
        let mut fd = open(anchor, dir_flags, Mode::empty())
            .map_err(|e| format!("open cold-cache ancestor {}: {e}", anchor.display()))?;
        for component in rel.components() {
            let std::path::Component::Normal(name) = component else {
                return Err(format!(
                    "cold-cache root {} has a non-plain path component",
                    display.display()
                ));
            };
            fd = match openat(&fd, name, no_follow, Mode::empty()) {
                Ok(next) => next,
                Err(e) if e == rustix::io::Errno::NOENT => {
                    if let Err(e) = mkdirat(&fd, name, Mode::RWXU)
                        && e != rustix::io::Errno::EXIST
                    {
                        return Err(format!("create cold-cache root: {e}"));
                    }
                    openat(&fd, name, no_follow, Mode::empty())
                        .map_err(|e| format!("open cold-cache root: {e}"))?
                }
                Err(e) => {
                    return Err(format!(
                        "open cold-cache root component {}: {e}",
                        name.to_string_lossy()
                    ));
                }
            };
        }
        let stat = rustix::fs::fstat(&fd).map_err(|e| format!("stat cold-cache root: {e}"))?;
        if !file_type_of(&stat).is_dir() {
            return Err("cold-cache root is not a directory".to_string());
        }
        // SAFETY: geteuid has no preconditions and cannot fail.
        if stat.st_uid != unsafe { libc::geteuid() } {
            return Err("cold-cache root is not owned by the current user".to_string());
        }
        Ok(Self { path: display, fd })
    }

    fn set_root_permissions(&self) -> Result<(), String> {
        rustix::fs::fchmod(&self.fd, rustix::fs::Mode::RWXU)
            .map_err(|e| format!("set cold-cache directory permissions: {e}"))
    }

    fn open_existing(&self, name: &str) -> std::io::Result<File> {
        use rustix::fs::{Mode, OFlags, openat};
        let flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
        openat(&self.fd, name, flags, Mode::empty())
            .map(File::from)
            .map_err(std::io::Error::from)
    }

    fn create_exclusive(&self, name: &str) -> Result<File, String> {
        use rustix::fs::{Mode, OFlags, fchmod, openat};
        let flags =
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC;
        let mode = Mode::RUSR | Mode::WUSR;
        let fd = openat(&self.fd, name, flags, mode)
            .map_err(|e| format!("create cold-cache temp file: {e}"))?;
        fchmod(&fd, mode).map_err(|e| format!("set cold-cache file permissions: {e}"))?;
        Ok(File::from(fd))
    }

    fn rename(&self, from: &str, to: &str) -> Result<(), String> {
        rustix::fs::renameat(&self.fd, from, &self.fd, to)
            .map_err(|e| format!("commit cold-cache file: {e}"))
    }

    fn unlink(&self, name: &str) -> std::io::Result<()> {
        rustix::fs::unlinkat(&self.fd, name, rustix::fs::AtFlags::empty())
            .map_err(std::io::Error::from)
    }

    fn sync(&self) -> Result<(), String> {
        rustix::fs::fsync(&self.fd).map_err(|e| format!("sync cold-cache directory: {e}"))
    }

    fn space(&self) -> Result<(u64, u64), String> {
        let vfs =
            rustix::fs::fstatvfs(&self.fd).map_err(|e| format!("statvfs cold-cache root: {e}"))?;
        Ok((
            vfs.f_blocks.saturating_mul(vfs.f_frsize),
            vfs.f_bavail.saturating_mul(vfs.f_frsize),
        ))
    }

    fn entry_names(&self) -> Result<Vec<String>, String> {
        let dir = rustix::fs::Dir::read_from(&self.fd)
            .map_err(|e| format!("scan cold-cache root: {e}"))?;
        let mut names = Vec::new();
        for entry in dir {
            let Ok(entry) = entry else { continue };
            let Ok(name) = entry.file_name().to_str() else {
                continue;
            };
            if name != "." && name != ".." {
                names.push(name.to_string());
            }
        }
        Ok(names)
    }

    /// Size and mtime of `name` when it is a regular file (never following
    /// symlinks); `None` otherwise, so symlinked entries are never indexed.
    fn stat_file(&self, name: &str) -> Option<(u64, u128)> {
        let stat = self.stat_no_follow(name)?;
        if !file_type_of(&stat).is_file() {
            return None;
        }
        Some((
            u64::try_from(stat.st_size).unwrap_or(0),
            mtime_nanos_of(&stat),
        ))
    }

    fn stat_identity(&self, name: &str) -> Option<FileIdentity> {
        self.stat_no_follow(name).map(|stat| identity_of(&stat))
    }

    fn stat_no_follow(&self, name: &str) -> Option<rustix::fs::Stat> {
        rustix::fs::statat(&self.fd, name, rustix::fs::AtFlags::SYMLINK_NOFOLLOW).ok()
    }
}

// Stat field widths vary across unix targets, so some of these casts are
// identities on one platform and lossless widenings on another.
#[cfg(unix)]
#[allow(clippy::unnecessary_cast)]
fn file_type_of(stat: &rustix::fs::Stat) -> rustix::fs::FileType {
    rustix::fs::FileType::from_raw_mode(stat.st_mode as rustix::fs::RawMode)
}

#[cfg(unix)]
#[allow(clippy::unnecessary_cast)]
fn identity_of(stat: &rustix::fs::Stat) -> FileIdentity {
    FileIdentity {
        device: stat.st_dev as u64,
        inode: stat.st_ino as u64,
    }
}

#[cfg(unix)]
#[allow(clippy::unnecessary_cast)]
fn mtime_nanos_of(stat: &rustix::fs::Stat) -> u128 {
    if stat.st_mtime < 0 {
        return 0;
    }
    (stat.st_mtime as u128) * 1_000_000_000 + (stat.st_mtime_nsec.max(0) as u128)
}

#[cfg(not(unix))]
impl RootDir {
    fn open_at_path(root: PathBuf) -> Result<Self, String> {
        match fs::symlink_metadata(&root) {
            Ok(meta) if meta.file_type().is_symlink() || !meta.is_dir() => {
                return Err("cold-cache root exists but is not a plain directory".to_string());
            }
            _ => {}
        }
        fs::create_dir_all(&root).map_err(|e| format!("create cold-cache root: {e}"))?;
        Ok(Self { path: root })
    }

    fn set_root_permissions(&self) -> Result<(), String> {
        Ok(())
    }

    fn open_existing(&self, name: &str) -> std::io::Result<File> {
        File::open(self.path.join(name))
    }

    fn create_exclusive(&self, name: &str) -> Result<File, String> {
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(self.path.join(name))
            .map_err(|e| format!("create cold-cache temp file: {e}"))
    }

    fn rename(&self, from: &str, to: &str) -> Result<(), String> {
        fs::rename(self.path.join(from), self.path.join(to))
            .map_err(|e| format!("commit cold-cache file: {e}"))
    }

    fn unlink(&self, name: &str) -> std::io::Result<()> {
        fs::remove_file(self.path.join(name))
    }

    fn sync(&self) -> Result<(), String> {
        Ok(())
    }

    fn space(&self) -> Result<(u64, u64), String> {
        Err("automatic cold-cache quota requires a Unix statvfs implementation".to_string())
    }

    fn entry_names(&self) -> Result<Vec<String>, String> {
        let mut names = Vec::new();
        for entry in fs::read_dir(&self.path).map_err(|e| format!("scan cold-cache root: {e}"))? {
            let Ok(entry) = entry else { continue };
            if let Ok(name) = entry.file_name().into_string() {
                names.push(name);
            }
        }
        Ok(names)
    }

    fn stat_file(&self, name: &str) -> Option<(u64, u128)> {
        let meta = fs::symlink_metadata(self.path.join(name)).ok()?;
        if !meta.is_file() {
            return None;
        }
        let mtime = meta
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |duration| duration.as_nanos());
        Some((meta.len(), mtime))
    }

    fn stat_identity(&self, _name: &str) -> Option<FileIdentity> {
        None
    }
}

struct Shared {
    root: RootDir,
    quota_bytes: u64,
    reserve_bytes: u64,
    index: Mutex<CacheIndex>,
    stats: AtomicStats,
    /// Invoked between a failed read and its cleanup so tests can commit a
    /// writer replacement at exactly that interleaving point.
    #[cfg(test)]
    failed_load_cleanup_hook: Mutex<Option<Box<dyn Fn() + Send>>>,
}

struct WriteJob {
    block: ColdCacheBlock,
}

/// Bounded background SSD cache. Clones share one queue/index.
#[derive(Clone)]
pub struct ColdCacheManager {
    shared: Arc<Shared>,
    sender: SyncSender<WriteJob>,
}

impl ColdCacheManager {
    /// Open the automatic cache root (`~/.mlx-node/cache/paged/v1`) with a
    /// quota of 10% of filesystem capacity, capped at 100 GiB. At least 5%
    /// or 5 GiB (whichever is larger) remains reserved for the filesystem.
    pub fn open_default() -> Result<Self, String> {
        let home = std::env::var_os("HOME")
            .ok_or_else(|| "HOME is not set; cannot locate the paged cache".to_string())?;
        Self::open_default_at(PathBuf::from(home).join(".mlx-node/cache/paged/v1"))
    }

    /// Open a custom root with the same automatic quota policy as
    /// [`Self::open_default`]: 10% of filesystem capacity capped at 100 GiB,
    /// a 5%-or-5-GiB free reserve, and the default queue depth.
    pub fn open_default_at(root: PathBuf) -> Result<Self, String> {
        let root = RootDir::open_at_path(root)?;
        let (total, _) = root.space()?;
        let quota = (total / 10).min(MAX_DEFAULT_QUOTA);
        let reserve = (total / 20).max(MIN_FREE_RESERVE);
        Self::open_prepared(root, quota, reserve, DEFAULT_QUEUE_DEPTH)
    }

    /// Explicit constructor used by tests and embedders with custom policy.
    /// The manager takes ownership of `root`: opening chmods it 0700 and
    /// removes leftover writer temp files, so callers must pass a directory
    /// dedicated to this cache, never a shared/user directory. On unix the
    /// root must resolve without symlinks below its deepest pre-existing
    /// ancestor, must be owned by the current effective uid, and is held as
    /// a directory descriptor for all later cache I/O.
    pub fn open_at(
        root: PathBuf,
        quota_bytes: u64,
        reserve_bytes: u64,
        queue_depth: usize,
    ) -> Result<Self, String> {
        Self::open_prepared(
            RootDir::open_at_path(root)?,
            quota_bytes,
            reserve_bytes,
            queue_depth,
        )
    }

    fn open_prepared(
        root: RootDir,
        quota_bytes: u64,
        reserve_bytes: u64,
        queue_depth: usize,
    ) -> Result<Self, String> {
        if quota_bytes == 0 || queue_depth == 0 {
            return Err("cold-cache quota and queue depth must be non-zero".to_string());
        }
        root.set_root_permissions()?;
        let index = rebuild_index(&root)?;
        let shared = Arc::new(Shared {
            root,
            quota_bytes,
            reserve_bytes,
            index: Mutex::new(index),
            stats: AtomicStats::default(),
            #[cfg(test)]
            failed_load_cleanup_hook: Mutex::new(None),
        });
        let (sender, receiver) = mpsc::sync_channel::<WriteJob>(queue_depth);
        let worker_shared = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("mlx-paged-ssd-writer".to_string())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    // Fail-open: inference already has a valid hot block. A
                    // persistence error only means the next process recomputes.
                    let _ = persist_block(&worker_shared, &job.block);
                }
            })
            .map_err(|e| format!("spawn cold-cache writer: {e}"))?;
        Ok(Self { shared, sender })
    }

    pub fn root(&self) -> &Path {
        &self.shared.root.path
    }

    pub fn quota_bytes(&self) -> u64 {
        self.shared.quota_bytes
    }

    pub fn stats(&self) -> ColdCacheStats {
        self.shared.stats.snapshot()
    }

    /// Whether a persisted block for `key` is present in the in-memory
    /// index. No filesystem I/O and no stats side effects, so callers can
    /// probe before deciding to capture without inflating hit/miss counts.
    /// A file deleted externally leaves a stale `true` only until the next
    /// `load` for that key misses and prunes the entry.
    pub fn contains(&self, key: &ColdCacheKey) -> bool {
        self.shared
            .index
            .lock()
            .map(|index| index.entries.contains_key(key))
            .unwrap_or(false)
    }

    /// Capture one pinned physical block from Metal, then enqueue only the
    /// owned host bytes. The writer thread never calls MLX/Metal and never
    /// holds the allocator lock.
    pub fn capture_and_enqueue(
        &self,
        pool: &LayerKVPool,
        block: &Arc<PhysicalBlock>,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        tokens: &[u32],
    ) -> Result<bool, String> {
        if tokens.len() != pool.block_size() as usize {
            return Err("cold cache captures full blocks only".to_string());
        }

        // Logical pin prevents allocator eviction/reuse while Metal blits run.
        block.incref();
        let captured: Result<ColdCacheBlock, String> = (|| {
            let mut layers = Vec::with_capacity(pool.num_layers());
            for layer in 0..pool.num_layers() as u32 {
                let (keys, values) = pool.read_blocks_to_host(layer, &[block.block_id])?;
                layers.push(ColdLayerBlock { keys, values });
            }
            let first = layers
                .first()
                .ok_or_else(|| "cannot persist a pool with zero layers".to_string())?;
            let layout = ColdCacheLayout {
                block_size: pool.block_size(),
                num_layers: pool.num_layers() as u32,
                num_kv_heads: pool.config().num_kv_heads,
                head_size: pool.config().head_size,
                cache_dtype: format!("{:?}", pool.cache_dtype()),
                key_bytes_per_layer: first.keys.len(),
                value_bytes_per_layer: first.values.len(),
            };
            Ok(ColdCacheBlock {
                key,
                fingerprint,
                tokens: tokens.to_vec(),
                layout,
                layers,
            })
        })();
        let _ = block.decref();
        self.enqueue(captured?)
    }

    /// Non-blocking enqueue. A saturated queue deliberately drops the cold
    /// write so host buffers cannot grow without bound.
    pub fn enqueue(&self, block: ColdCacheBlock) -> Result<bool, String> {
        block.validate()?;
        match self.sender.try_send(WriteJob { block }) {
            Ok(()) => {
                self.shared.stats.enqueued.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            }
            Err(TrySendError::Full(_)) => {
                self.shared
                    .stats
                    .queue_drops
                    .fetch_add(1, Ordering::Relaxed);
                Ok(false)
            }
            Err(TrySendError::Disconnected(_)) => Err("cold-cache writer stopped".to_string()),
        }
    }

    /// Load and validate a block. Every failed read is a miss; a payload
    /// that existed but failed validation additionally counts as a
    /// corruption (a missing file never does). Failure cleanup unlinks only
    /// the exact file generation (dev+inode) that was read, under the same
    /// index lock the writer holds across [rename + index publish], so an
    /// in-process writer's freshly committed replacement is never deleted
    /// or de-indexed. Coordination is in-process only: a concurrent
    /// *process* mutating the same root stays fail-open — the identity
    /// guard still refuses to unlink its replacement, and the worst case is
    /// a stale index entry or one recomputed prefix.
    pub fn load(
        &self,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
    ) -> Option<ColdCacheBlock> {
        let name = block_file_name(&key);
        let mut read_identity = None;
        let mut opened_file = None;
        let result = match self.shared.root.open_existing(&name) {
            Ok(mut file) => {
                read_identity = open_identity(&file);
                let mut bytes = Vec::new();
                let read = file
                    .read_to_end(&mut bytes)
                    .map_err(|e| e.to_string())
                    .and_then(|_| decode_block(&bytes, key, fingerprint));
                opened_file = Some(file);
                read
            }
            Err(e) => Err(e.to_string()),
        };
        match result {
            Ok(block) => {
                self.shared.stats.hits.fetch_add(1, Ordering::Relaxed);
                self.shared
                    .stats
                    .bytes_restored
                    .fetch_add(block.encoded_len(), Ordering::Relaxed);
                // Startup rebuild derives recency from file mtime. Persist
                // every validated hit (on the descriptor that was read, so a
                // swapped pathname is never touched) so a process restart
                // preserves the same LRU order instead of reverting to
                // original write age. Touch failure is deliberately
                // fail-open: the block is already validated and useful to
                // inference; only future eviction precision is affected.
                let touched_at = SystemTime::now();
                if let Some(file) = &opened_file {
                    let _ = file.set_times(std::fs::FileTimes::new().set_modified(touched_at));
                }
                let touched_tick = touched_at
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos();
                if let Ok(mut index) = self.shared.index.lock()
                    && let Some(entry) = index.entries.get_mut(&key)
                {
                    entry.last_access = touched_tick;
                }
                Some(block)
            }
            Err(_) => {
                self.shared.stats.misses.fetch_add(1, Ordering::Relaxed);
                if opened_file.is_some() {
                    self.shared
                        .stats
                        .corruptions
                        .fetch_add(1, Ordering::Relaxed);
                }
                #[cfg(test)]
                if let Ok(hook) = self.shared.failed_load_cleanup_hook.lock()
                    && let Some(hook) = hook.as_ref()
                {
                    hook();
                }
                prune_failed_load(&self.shared, key, &name, read_identity);
                None
            }
        }
    }

    /// Restore one block transactionally. Returns `None` on every cold-tier
    /// failure so the caller can perform ordinary prefill.
    pub fn restore_block(
        &self,
        pool: &LayerKVPool,
        allocator: &Mutex<BlockAllocator>,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        identity: &RestorePrefixIdentity,
    ) -> Option<Arc<PhysicalBlock>> {
        let cold = self.load(key, fingerprint)?;
        if cold.tokens != identity.tokens || !layout_matches_pool(&cold.layout, pool) {
            return None;
        }
        let block = allocator
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .allocate()?;

        for (layer_idx, layer) in cold.layers.iter().enumerate() {
            if pool
                .write_blocks_from_host(
                    layer_idx as u32,
                    &[block.block_id],
                    &layer.keys,
                    &layer.values,
                )
                .is_err()
            {
                allocator
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .free(Arc::clone(&block));
                return None;
            }
        }

        let published = allocator
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .publish_restored_prefix(
                Arc::clone(&block),
                identity.hot_hash,
                &identity.tokens,
                identity.parent_hot_hash,
                &identity.extra_keys,
                identity.cache_salt,
                identity.block_index,
            );
        match published {
            Ok(true) => Some(block),
            _ => {
                allocator
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .free(block);
                None
            }
        }
    }
}

fn layout_matches_pool(layout: &ColdCacheLayout, pool: &LayerKVPool) -> bool {
    layout.block_size == pool.block_size()
        && layout.num_layers as usize == pool.num_layers()
        && layout.num_kv_heads == pool.config().num_kv_heads
        && layout.head_size == pool.config().head_size
        && layout.cache_dtype == format!("{:?}", pool.cache_dtype())
}

fn persist_block(shared: &Shared, block: &ColdCacheBlock) -> Result<(), String> {
    block.validate()?;
    let bytes = encode_block(block)?;
    evict_for_write(shared, bytes.len() as u64)?;
    let destination = block_file_name(&block.key);
    let temp = format!(
        ".{}.{}.{}.tmp",
        block.key.to_hex(),
        std::process::id(),
        now_tick()
    );
    let mut file = shared.root.create_exclusive(&temp)?;
    let size = bytes.len() as u64;
    if let Err(error) = (|| -> Result<(), String> {
        file.write_all(&bytes)
            .map_err(|e| format!("write cold-cache file: {e}"))?;
        file.sync_all()
            .map_err(|e| format!("sync cold-cache file: {e}"))?;
        // The index lock spans [rename + index publish] so a concurrent
        // failed-load cleanup can never observe the renamed file without
        // its index entry (or delete it in between).
        let mut index = shared
            .index
            .lock()
            .map_err(|_| "cold-cache index mutex poisoned".to_string())?;
        shared.root.rename(&temp, &destination)?;
        shared.root.sync()?;
        if let Some(old) = index.entries.insert(
            block.key,
            IndexEntry {
                file_name: destination.clone(),
                size,
                last_access: now_tick(),
            },
        ) {
            index.total_bytes = index.total_bytes.saturating_sub(old.size);
        }
        index.total_bytes = index.total_bytes.saturating_add(size);
        Ok(())
    })() {
        let _ = shared.root.unlink(&temp);
        return Err(error);
    }
    shared
        .stats
        .bytes_written
        .fetch_add(size, Ordering::Relaxed);
    Ok(())
}

fn evict_for_write(shared: &Shared, incoming: u64) -> Result<(), String> {
    let (_, mut available) = shared.root.space()?;
    let mut index = shared
        .index
        .lock()
        .map_err(|_| "cold-cache index mutex poisoned".to_string())?;
    while index.total_bytes.saturating_add(incoming) > shared.quota_bytes
        || available < shared.reserve_bytes.saturating_add(incoming)
    {
        let Some((&key, _)) = index.entries.iter().min_by_key(|(_, e)| e.last_access) else {
            return Err("insufficient disk space for cold-cache write".to_string());
        };
        if let Some(entry) = index.entries.remove(&key) {
            let _ = shared.root.unlink(&entry.file_name);
            index.total_bytes = index.total_bytes.saturating_sub(entry.size);
            available = available.saturating_add(entry.size);
            shared.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }
    Ok(())
}

/// Cleanup after a failed load, under the same index lock the writer holds
/// across [rename + index publish]. Unlinks (descriptor-relative) only when
/// the directory entry `name` is still the generation that was read
/// (`read_identity`); a renamed-in replacement (new inode) is left on disk
/// and keeps its index entry.
fn prune_failed_load(
    shared: &Shared,
    key: ColdCacheKey,
    name: &str,
    read_identity: Option<FileIdentity>,
) {
    let Ok(mut index) = shared.index.lock() else {
        return;
    };
    match shared.root.stat_identity(name) {
        Some(current) if read_identity == Some(current) => {
            let _ = shared.root.unlink(name);
            if let Some(entry) = index.entries.remove(&key) {
                index.total_bytes = index.total_bytes.saturating_sub(entry.size);
            }
        }
        Some(_) => {}
        None => {
            if let Some(entry) = index.entries.remove(&key) {
                index.total_bytes = index.total_bytes.saturating_sub(entry.size);
            }
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
}

#[cfg(unix)]
fn open_identity(file: &File) -> Option<FileIdentity> {
    use std::os::unix::fs::MetadataExt;
    file.metadata().ok().map(|metadata| FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

#[cfg(not(unix))]
fn open_identity(_file: &File) -> Option<FileIdentity> {
    None
}

fn block_file_name(key: &ColdCacheKey) -> String {
    format!("{}.safetensors", key.to_hex())
}

fn encode_block(block: &ColdCacheBlock) -> Result<Vec<u8>, String> {
    let token_bytes: Vec<u8> = block.tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut owned: Vec<(String, Vec<u8>)> = Vec::with_capacity(1 + block.layers.len() * 2);
    owned.push(("tokens".to_string(), token_bytes));
    for (i, layer) in block.layers.iter().enumerate() {
        owned.push((format!("layer.{i}.key"), layer.keys.clone()));
        owned.push((format!("layer.{i}.value"), layer.values.clone()));
    }
    let checksum = payload_checksum(&owned);
    let views: Result<Vec<_>, _> = owned
        .iter()
        .map(|(name, data)| {
            TensorView::new(Dtype::U8, vec![data.len()], data).map(|view| (name.as_str(), view))
        })
        .collect();
    let mut metadata = HashMap::new();
    metadata.insert("abi".to_string(), CACHE_ABI.to_string());
    metadata.insert("key".to_string(), block.key.to_hex());
    metadata.insert("fingerprint".to_string(), block.fingerprint.to_hex());
    metadata.insert("checksum".to_string(), checksum);
    metadata.insert(
        "block_size".to_string(),
        block.layout.block_size.to_string(),
    );
    metadata.insert(
        "num_layers".to_string(),
        block.layout.num_layers.to_string(),
    );
    metadata.insert(
        "num_kv_heads".to_string(),
        block.layout.num_kv_heads.to_string(),
    );
    metadata.insert("head_size".to_string(), block.layout.head_size.to_string());
    metadata.insert("cache_dtype".to_string(), block.layout.cache_dtype.clone());
    metadata.insert(
        "key_bytes".to_string(),
        block.layout.key_bytes_per_layer.to_string(),
    );
    metadata.insert(
        "value_bytes".to_string(),
        block.layout.value_bytes_per_layer.to_string(),
    );
    serialize(views.map_err(|e| e.to_string())?, Some(metadata)).map_err(|e| e.to_string())
}

fn decode_block(
    bytes: &[u8],
    expected_key: ColdCacheKey,
    expected_fingerprint: ColdCacheFingerprint,
) -> Result<ColdCacheBlock, String> {
    let (_, header) = SafeTensors::read_metadata(bytes).map_err(|e| e.to_string())?;
    let metadata = header
        .metadata()
        .as_ref()
        .ok_or_else(|| "cold-cache metadata missing".to_string())?;
    let tensors = SafeTensors::deserialize(bytes).map_err(|e| e.to_string())?;
    let get = |name: &str| {
        metadata
            .get(name)
            .cloned()
            .ok_or_else(|| format!("cold-cache metadata `{name}` missing"))
    };
    if get("abi")? != CACHE_ABI
        || get("key")? != expected_key.to_hex()
        || get("fingerprint")? != expected_fingerprint.to_hex()
    {
        return Err("cold-cache identity/ABI mismatch".to_string());
    }
    let parse = |name: &str| -> Result<u32, String> {
        get(name)?
            .parse::<u32>()
            .map_err(|_| format!("invalid cold-cache metadata `{name}`"))
    };
    let parse_usize = |name: &str| -> Result<usize, String> {
        get(name)?
            .parse::<usize>()
            .map_err(|_| format!("invalid cold-cache metadata `{name}`"))
    };
    let layout = ColdCacheLayout {
        block_size: parse("block_size")?,
        num_layers: parse("num_layers")?,
        num_kv_heads: parse("num_kv_heads")?,
        head_size: parse("head_size")?,
        cache_dtype: get("cache_dtype")?,
        key_bytes_per_layer: parse_usize("key_bytes")?,
        value_bytes_per_layer: parse_usize("value_bytes")?,
    };
    let token_data = tensors.tensor("tokens").map_err(|e| e.to_string())?;
    let token_bytes = token_data.data();
    if token_bytes.len() % 4 != 0 {
        return Err("cold-cache tokens have invalid byte length".to_string());
    }
    let tokens = token_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();
    let mut layers = Vec::with_capacity(layout.num_layers as usize);
    for i in 0..layout.num_layers as usize {
        layers.push(ColdLayerBlock {
            keys: tensors
                .tensor(&format!("layer.{i}.key"))
                .map_err(|e| e.to_string())?
                .data()
                .to_vec(),
            values: tensors
                .tensor(&format!("layer.{i}.value"))
                .map_err(|e| e.to_string())?
                .data()
                .to_vec(),
        });
    }
    let block = ColdCacheBlock {
        key: expected_key,
        fingerprint: expected_fingerprint,
        tokens,
        layout,
        layers,
    };
    block.validate()?;

    let mut owned = Vec::with_capacity(1 + block.layers.len() * 2);
    owned.push((
        "tokens".to_string(),
        block.tokens.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ));
    for (i, layer) in block.layers.iter().enumerate() {
        owned.push((format!("layer.{i}.key"), layer.keys.clone()));
        owned.push((format!("layer.{i}.value"), layer.values.clone()));
    }
    if payload_checksum(&owned) != get("checksum")? {
        return Err("cold-cache payload checksum mismatch".to_string());
    }
    Ok(block)
}

fn payload_checksum(tensors: &[(String, Vec<u8>)]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"mlx-node:cold-cache-payload:v1\0");
    for (name, data) in tensors {
        hasher.update((name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update((data.len() as u64).to_le_bytes());
        hasher.update(data);
    }
    hex_encode(&hasher.finalize())
}

fn rebuild_index(root: &RootDir) -> Result<CacheIndex, String> {
    let mut index = CacheIndex::default();
    for name in root.entry_names()? {
        let Some(stem) = name.strip_suffix(".safetensors") else {
            if is_cold_cache_temp_file(&name) {
                let _ = root.unlink(&name);
            }
            continue;
        };
        let Some(key) = ColdCacheKey::from_hex(stem) else {
            continue;
        };
        let Some((size, last_access)) = root.stat_file(&name) else {
            continue;
        };
        index.entries.insert(
            key,
            IndexEntry {
                file_name: name,
                size,
                last_access,
            },
        );
        index.total_bytes = index.total_bytes.saturating_add(size);
    }
    Ok(index)
}

/// Matches exactly the temp-file names `persist_block` creates
/// (`.{64-hex key}.{pid}.{tick}.tmp`) so startup cleanup can never remove
/// foreign files from a directory it was mistakenly pointed at.
fn is_cold_cache_temp_file(name: &str) -> bool {
    let Some(body) = name
        .strip_prefix('.')
        .and_then(|rest| rest.strip_suffix(".tmp"))
    else {
        return false;
    };
    let mut parts = body.split('.');
    let (Some(key), Some(pid), Some(tick), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    let is_digits = |value: &str| !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit());
    hex_decode_32(key).is_some() && is_digits(pid) && is_digits(tick)
}

fn now_tick() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn hex_decode_32(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 {
        return None;
    }
    fn nibble(value: u8) -> Option<u8> {
        match value {
            b'0'..=b'9' => Some(value - b'0'),
            b'a'..=b'f' => Some(value - b'a' + 10),
            b'A'..=b'F' => Some(value - b'A' + 10),
            _ => None,
        }
    }
    let mut output = [0u8; 32];
    for (i, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        output[i] = nibble(pair[0])? << 4 | nibble(pair[1])?;
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint() -> ColdCacheFingerprint {
        ColdCacheFingerprint::from_components([b"model".as_slice(), b"tokenizer".as_slice()])
    }

    fn block(key: ColdCacheKey) -> ColdCacheBlock {
        ColdCacheBlock {
            key,
            fingerprint: fingerprint(),
            tokens: vec![1, 2, 3, 4],
            layout: ColdCacheLayout {
                block_size: 4,
                num_layers: 2,
                num_kv_heads: 1,
                head_size: 2,
                cache_dtype: "BFloat16".to_string(),
                key_bytes_per_layer: 4,
                value_bytes_per_layer: 4,
            },
            layers: vec![
                ColdLayerBlock {
                    keys: vec![1, 2, 3, 4],
                    values: vec![5, 6, 7, 8],
                },
                ColdLayerBlock {
                    keys: vec![9, 10, 11, 12],
                    values: vec![13, 14, 15, 16],
                },
            ],
        }
    }

    fn temp_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "mlx-paged-cold-cache-{name}-{}-{}",
            std::process::id(),
            now_tick()
        ))
    }

    #[test]
    fn stable_chain_is_parent_and_fingerprint_sensitive() {
        let fp = fingerprint();
        let first = ColdCacheKey::chain(fp, None, &[1, 2, 3, 4], &[], 0, 0);
        assert_eq!(
            first,
            ColdCacheKey::chain(fp, None, &[1, 2, 3, 4], &[], 0, 0)
        );
        assert_ne!(
            first,
            ColdCacheKey::chain(fp, None, &[1, 2, 3, 5], &[], 0, 0)
        );
        assert_ne!(
            ColdCacheKey::chain(fp, Some(first), &[5, 6, 7, 8], &[], 0, 1),
            ColdCacheKey::chain(fp, None, &[5, 6, 7, 8], &[], 0, 1)
        );
    }

    #[test]
    fn safetensors_roundtrip_and_checksum() {
        let key = ColdCacheKey::chain(fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let original = block(key);
        let encoded = encode_block(&original).unwrap();
        let decoded = decode_block(&encoded, key, fingerprint()).unwrap();
        assert_eq!(decoded, original);

        let mut corrupt = encoded;
        *corrupt.last_mut().unwrap() ^= 0xff;
        assert!(decode_block(&corrupt, key, fingerprint()).is_err());
    }

    #[test]
    fn full_blocks_only() {
        let key = ColdCacheKey::chain(fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let mut partial = block(key);
        partial.tokens.pop();
        assert!(partial.validate().is_err());
    }

    #[test]
    fn writer_is_atomic_and_index_rebuilds() {
        let root = temp_root("roundtrip");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let expected = block(key);
        assert!(manager.enqueue(expected.clone()).unwrap());

        let path = root.join(format!("{}.safetensors", key.to_hex()));
        for _ in 0..100 {
            if path.exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(manager.load(key, fingerprint()), Some(expected));
        drop(manager);

        let reopened = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        assert!(reopened.load(key, fingerprint()).is_some());
        assert_eq!(reopened.shared.index.lock().unwrap().entries.len(), 1);
        drop(reopened);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn restart_lru_uses_persisted_read_recency() {
        fn wait_for(path: &Path) {
            for _ in 0..200 {
                if path.exists() {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            panic!("timed out waiting for {}", path.display());
        }

        let root = temp_root("restart-lru");
        let fp = fingerprint();
        let key_a = ColdCacheKey::chain(fp, None, &[1, 2, 3, 4], &[1], 0, 0);
        let key_b = ColdCacheKey::chain(fp, None, &[1, 2, 3, 4], &[2], 0, 0);
        let key_c = ColdCacheKey::chain(fp, None, &[1, 2, 3, 4], &[3], 0, 0);
        let path_a = root.join(format!("{}.safetensors", key_a.to_hex()));
        let path_b = root.join(format!("{}.safetensors", key_b.to_hex()));
        let path_c = root.join(format!("{}.safetensors", key_c.to_hex()));

        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        manager.enqueue(block(key_a)).unwrap();
        wait_for(&path_a);
        // Keep write mtimes strictly ordered even on coarse filesystems.
        std::thread::sleep(std::time::Duration::from_millis(20));
        manager.enqueue(block(key_b)).unwrap();
        wait_for(&path_b);
        std::thread::sleep(std::time::Duration::from_millis(20));

        // A was written first but read last. The hit must persist that fact
        // in mtime so a new manager evicts B before A.
        assert!(manager.load(key_a, fp).is_some());
        let size_a = fs::metadata(&path_a).unwrap().len();
        let size_b = fs::metadata(&path_b).unwrap().len();
        drop(manager);
        std::thread::sleep(std::time::Duration::from_millis(10));

        let reopened = ColdCacheManager::open_at(root.clone(), size_a + size_b, 0, 1).unwrap();
        reopened.enqueue(block(key_c)).unwrap();
        wait_for(&path_c);
        // The writer updates the index immediately after rename; wait for the
        // old-file removal/index commit to be visible too.
        for _ in 0..200 {
            if path_a.exists() && !path_b.exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(path_a.exists(), "recently read A must survive restart LRU");
        assert!(
            !path_b.exists(),
            "older unread B must be evicted after restart"
        );
        assert!(path_c.exists());

        drop(reopened);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn contains_checks_index_without_stats_side_effects() {
        let root = temp_root("contains");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        assert!(!manager.contains(&key));
        assert!(manager.enqueue(block(key)).unwrap());
        for _ in 0..200 {
            if manager.contains(&key) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(manager.contains(&key));
        let stats = manager.stats();
        assert_eq!(stats.hits, 0, "contains must not count as a hit");
        assert_eq!(stats.misses, 0, "contains must not count as a miss");
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_miss_after_external_delete_prunes_index() {
        let root = temp_root("external-delete");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        assert!(manager.enqueue(block(key)).unwrap());
        for _ in 0..200 {
            if manager.contains(&key) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(manager.contains(&key));

        let path = root.join(format!("{}.safetensors", key.to_hex()));
        fs::remove_file(&path).unwrap();
        assert!(manager.load(key, fingerprint()).is_none());
        assert!(
            !manager.contains(&key),
            "externally deleted entry must leave the index on the next load miss"
        );
        let stats = manager.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.corruptions, 0, "a missing file is not a corruption");
        assert_eq!(manager.shared.index.lock().unwrap().total_bytes, 0);
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn failed_load_cleanup_spares_concurrent_writer_replacement() {
        let root = temp_root("replace-race");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let path = root.join(format!("{}.safetensors", key.to_hex()));
        fs::write(&path, b"corrupt generation").unwrap();

        let shared = Arc::clone(&manager.shared);
        let replacement = block(key);
        let commit = replacement.clone();
        *manager.shared.failed_load_cleanup_hook.lock().unwrap() = Some(Box::new(move || {
            persist_block(&shared, &commit).unwrap();
        }));

        assert!(manager.load(key, fingerprint()).is_none());
        assert!(
            path.exists(),
            "cleanup must not delete the writer's renamed-in replacement"
        );
        assert!(
            manager.contains(&key),
            "the writer's index publish must survive failed-load cleanup"
        );
        let stats = manager.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.corruptions, 1, "the generation read was corrupt");

        *manager.shared.failed_load_cleanup_hook.lock().unwrap() = None;
        assert_eq!(manager.load(key, fingerprint()), Some(replacement));
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn startup_cleanup_removes_only_writer_temp_files() {
        let root = temp_root("tmp-cleanup");
        fs::create_dir_all(&root).unwrap();
        let key = ColdCacheKey::chain(fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let stale_writer_tmp = root.join(format!(".{}.{}.{}.tmp", key.to_hex(), 4242, 7));
        let foreign_tmp = root.join("foo.tmp");
        let foreign_data = root.join("data.txt");
        fs::write(&stale_writer_tmp, b"stale").unwrap();
        fs::write(&foreign_tmp, b"foreign").unwrap();
        fs::write(&foreign_data, b"data").unwrap();

        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 1).unwrap();
        assert!(
            !stale_writer_tmp.exists(),
            "leftover writer temp files must be cleaned at startup"
        );
        assert!(foreign_tmp.exists(), "unrelated *.tmp files must survive");
        assert!(foreign_data.exists());
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn default_shape_symlink_child_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let base = temp_root("default-symlink");
        let victim = base.join("victim");
        fs::create_dir_all(&victim).unwrap();
        let marker = victim.join("marker.txt");
        fs::write(&marker, b"marker").unwrap();
        let key = ColdCacheKey::chain(fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let victim_tmp = victim.join(format!(".{}.{}.{}.tmp", key.to_hex(), 4242, 7));
        fs::write(&victim_tmp, b"stale").unwrap();
        let victim_mode = fs::metadata(&victim).unwrap().permissions().mode() & 0o7777;

        let parent = base.join("home/.mlx-node/cache/paged");
        fs::create_dir_all(&parent).unwrap();
        std::os::unix::fs::symlink(&victim, parent.join("v1")).unwrap();
        assert!(
            ColdCacheManager::open_default_at(parent.join("v1")).is_err(),
            "a symlinked default root must be refused, not followed"
        );
        assert!(marker.exists());
        assert!(
            victim_tmp.exists(),
            "refusal must precede writer-temp cleanup through the link"
        );
        assert_eq!(
            fs::metadata(&victim).unwrap().permissions().mode() & 0o7777,
            victim_mode,
            "refusal must precede any chmod through the link"
        );

        let fresh = base.join("home2/.mlx-node/cache/paged/v1");
        let manager = ColdCacheManager::open_default_at(fresh.clone()).unwrap();
        assert_eq!(manager.root(), fresh.as_path());
        assert!(fresh.is_dir());
        drop(manager);
        let _ = fs::remove_dir_all(base);
    }

    #[cfg(unix)]
    #[test]
    fn post_open_symlink_swap_cannot_redirect_io() {
        use std::os::unix::fs::PermissionsExt;
        let base = temp_root("swap");
        let root = base.join("root");
        let moved = base.join("moved");
        let victim = base.join("victim");
        fs::create_dir_all(&victim).unwrap();
        let marker = victim.join("marker.txt");
        fs::write(&marker, b"marker").unwrap();
        let victim_mode = fs::metadata(&victim).unwrap().permissions().mode() & 0o7777;

        let fp = fingerprint();
        let key_a = ColdCacheKey::chain(fp, None, &[1, 2, 3, 4], &[1], 0, 0);
        let key_b = ColdCacheKey::chain(fp, None, &[1, 2, 3, 4], &[2], 0, 0);
        let one = encode_block(&block(key_a)).unwrap().len() as u64;
        let manager = ColdCacheManager::open_at(root.clone(), one + one / 2, 0, 2).unwrap();

        assert!(manager.enqueue(block(key_a)).unwrap());
        for _ in 0..200 {
            if manager.contains(&key_a) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(manager.contains(&key_a));

        fs::rename(&root, &moved).unwrap();
        std::os::unix::fs::symlink(&victim, &root).unwrap();

        assert!(manager.enqueue(block(key_b)).unwrap());
        for _ in 0..200 {
            if manager.contains(&key_b) && !manager.contains(&key_a) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let name_a = format!("{}.safetensors", key_a.to_hex());
        let name_b = format!("{}.safetensors", key_b.to_hex());
        assert!(
            moved.join(&name_b).exists(),
            "persist must land via the dirfd in the original directory"
        );
        assert!(
            !moved.join(&name_a).exists(),
            "eviction must unlink via the dirfd in the original directory"
        );
        assert!(
            !victim.join(&name_a).exists() && !victim.join(&name_b).exists(),
            "victim behind the swapped-in symlink must never receive cache I/O"
        );
        assert_eq!(manager.stats().evictions, 1);
        assert_eq!(
            manager.load(key_b, fp),
            Some(block(key_b)),
            "load must read via the dirfd, not the swapped pathname"
        );
        assert!(marker.exists());
        assert_eq!(
            fs::metadata(&victim).unwrap().permissions().mode() & 0o7777,
            victim_mode
        );
        assert_eq!(
            fs::read_dir(&victim).unwrap().count(),
            1,
            "victim must contain exactly its own marker file"
        );
        drop(manager);
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn open_default_at_applies_auto_quota_policy() {
        let root = temp_root("default-at");
        let manager = ColdCacheManager::open_default_at(root.clone()).unwrap();
        assert_eq!(manager.root(), root.as_path());
        assert!(manager.quota_bytes() > 0);
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn corrupt_file_fails_open_and_is_removed() {
        let root = temp_root("corrupt");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 1).unwrap();
        let key = ColdCacheKey::chain(fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let path = root.join(format!("{}.safetensors", key.to_hex()));
        fs::write(&path, b"not a safetensors file").unwrap();
        assert!(manager.load(key, fingerprint()).is_none());
        assert!(!path.exists());
        assert_eq!(manager.stats().corruptions, 1);
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn transactional_restore_uploads_then_publishes() {
        use crate::metal::MetalDtype;
        use crate::{PagedAttentionConfig, hash_tokens};

        let config = PagedAttentionConfig {
            block_size: 8,
            gpu_memory_mb: 256,
            head_size: 64,
            num_kv_heads: 1,
            num_layers: 1,
            use_fp8_cache: Some(false),
            max_seq_len: Some(32),
            max_batch_size: Some(1),
        };
        let pool = match LayerKVPool::new(config, 2, MetalDtype::BFloat16) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping transactional_restore_uploads_then_publishes: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        let allocator = Mutex::new(BlockAllocator::new(2, 8));
        let source = allocator.lock().unwrap().allocate().unwrap();
        let bytes_per_side = 64 * 8 * 2;
        let keys: Vec<u8> = (0..bytes_per_side).map(|i| (i % 251) as u8).collect();
        let values: Vec<u8> = (0..bytes_per_side)
            .map(|i| (250 - (i % 251)) as u8)
            .collect();
        pool.write_blocks_from_host(0, &[source.block_id], &keys, &values)
            .unwrap();

        let root = temp_root("restore");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let key = ColdCacheKey::chain(fingerprint(), None, &tokens, &[], 0, 0);
        assert!(
            manager
                .capture_and_enqueue(&pool, &source, key, fingerprint(), &tokens)
                .unwrap()
        );
        let path = root.join(format!("{}.safetensors", key.to_hex()));
        for _ in 0..100 {
            if path.exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        allocator.lock().unwrap().free(source);

        let identity = RestorePrefixIdentity {
            hot_hash: hash_tokens(&tokens, 0, &[]),
            tokens: tokens.clone(),
            parent_hot_hash: 0,
            extra_keys: vec![],
            cache_salt: 0,
            block_index: 0,
        };
        let restored = manager
            .restore_block(&pool, &allocator, key, fingerprint(), &identity)
            .expect("cold block restore");
        let (restored_keys, restored_values) =
            pool.read_blocks_to_host(0, &[restored.block_id]).unwrap();
        assert_eq!(restored_keys, keys);
        assert_eq!(restored_values, values);

        let (hits, hit_tokens) =
            allocator
                .lock()
                .unwrap()
                .find_longest_cache_hit(&tokens, 8, &[], 0);
        assert_eq!(hit_tokens, 8, "publish must happen after complete upload");
        assert_eq!(hits[0].block_id, restored.block_id);
        {
            let mut allocator = allocator.lock().unwrap();
            allocator.free(restored);
            for hit in hits {
                allocator.free(hit);
            }
        }
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }
}
