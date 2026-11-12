//! A loaded model's SSD cache tier for its DFlash2 states and their KV
//! chains (Splash `openDiskTier`, `DiskSpace`, `CacheDirectory`).
//!
//! Temporary by default: a locked directory under the temp root that the
//! model's `closeCache` removes at a clean stop and the next start sweeps
//! after a crash. `--persistent-cache` keeps it in a namespace directory of
//! the cache root across restarts.

use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

use mlx_paged_attn::{ColdCacheFingerprint, ColdCacheManager, DEFAULT_QUEUE_DEPTH};
use sha2::{Digest, Sha256};

const GIB: u64 = 1 << 30;

/// Splash `kAutomaticCacheDiskBytes`.
pub(crate) const AUTOMATIC_CACHE_DISK_BYTES: u64 = 32 * GIB;

/// Splash `kStaleCache`: a namespace nobody used for this long goes at the
/// next start.
const STALE_CACHE: Duration = Duration::from_secs(14 * 24 * 60 * 60);

/// Splash `CacheDirectory::kNamespaceDigits`.
const NAMESPACE_DIGITS: usize = 32;

const LOCK_FILE: &str = "lock";
const TEMP_PREFIX: &str = "mlx-node-cache-";
const TEMP_CREATING_PREFIX: &str = ".mlx-node-cache-new-";

/// Splash `kCacheLockWait`: a restarting server's new load can race the end
/// of the old one.
fn cache_lock_wait() -> Duration {
    if cfg!(test) {
        Duration::ZERO
    } else {
        Duration::from_secs(20)
    }
}

/// `--max-cache-disk`: `auto`, a size, or `0` (off).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum CacheDisk {
    #[default]
    Auto,
    Bytes(u64),
    Off,
}

impl CacheDisk {
    /// The load option's text: `auto`, or a decimal byte count (`0` = off).
    pub(crate) fn parse(text: &str) -> Result<Self, String> {
        if text == "auto" {
            return Ok(Self::Auto);
        }
        if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!(
                "cacheDisk must be 'auto' or a byte count, not {text:?}"
            ));
        }
        match text.parse::<u64>() {
            Ok(0) => Ok(Self::Off),
            Ok(bytes) => Ok(Self::Bytes(bytes)),
            Err(_) => Err(format!("cacheDisk {text:?} is out of range")),
        }
    }

    /// The quota the plan sets a state's staging aside for (Splash
    /// `cacheDiskBytes`: the automatic one at its nominal size).
    pub(crate) fn nominal_bytes(self) -> u64 {
        match self {
            Self::Auto => AUTOMATIC_CACHE_DISK_BYTES,
            Self::Bytes(bytes) => bytes,
            Self::Off => 0,
        }
    }
}

/// How a load keeps its SSD cache.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DiskTierOptions {
    pub cache_disk: CacheDisk,
    /// `--persistent-cache`: keep the cache across restarts.
    pub persistent: bool,
    /// `--cache-dir`, only with `persistent`.
    pub cache_dir: Option<PathBuf>,
    /// Where temporary caches go instead of the process temp directory.
    pub temp_root: Option<PathBuf>,
}

impl DiskTierOptions {
    /// The load options, refused as Splash `check_serve_arguments` refuses
    /// the flags.
    pub(crate) fn from_load(
        cache_disk: Option<&str>,
        persistent: Option<bool>,
        cache_dir: Option<String>,
    ) -> Result<Self, String> {
        let cache_disk = cache_disk.map_or(Ok(CacheDisk::Auto), CacheDisk::parse)?;
        let persistent = persistent.unwrap_or(false);
        if cache_dir.is_some() && !persistent {
            return Err("--cache-dir needs --persistent-cache".to_string());
        }
        if persistent && cache_disk == CacheDisk::Off {
            return Err(
                "--persistent-cache needs the SSD cache --max-cache-disk 0 turns off".to_string(),
            );
        }
        Ok(Self {
            cache_disk,
            persistent,
            cache_dir: cache_dir.map(PathBuf::from),
            temp_root: None,
        })
    }

    /// Whether the load reads its weights' identity: only a persistent tier
    /// keys anything by it.
    pub(crate) fn reads_identity(&self) -> bool {
        self.persistent && self.cache_disk != CacheDisk::Off
    }

    fn temp_root(&self) -> PathBuf {
        self.temp_root.clone().unwrap_or_else(std::env::temp_dir)
    }

    fn cache_root(&self) -> Option<PathBuf> {
        if cfg!(test) {
            return self.cache_dir.clone();
        }
        self.cache_dir.clone().or_else(|| {
            std::env::var_os("HOME")
                .map(|home| PathBuf::from(home).join(".mlx-node/cache/prefix-cache"))
        })
    }
}

/// Splash `diskFloorBytes`.
pub(crate) fn disk_floor_bytes(volume: u64) -> u64 {
    (10 * GIB).min(volume / 20)
}

/// Splash `diskRoomBytes`.
pub(crate) fn disk_room_bytes(available: u64, held: u64, floor: u64) -> u64 {
    available.saturating_add(held).saturating_sub(floor)
}

/// Splash `diskLimitBytes`: an automatic tier keeps a quarter of the room.
pub(crate) fn disk_limit_bytes(quota: u64, available: u64, held: u64, floor: u64) -> u64 {
    let room = disk_room_bytes(available, held, floor);
    quota.min(room / 4)
}

/// Splash `diskActiveLimitBytes`: a size may take all of the room.
pub(crate) fn disk_active_limit_bytes(quota: u64, available: u64, held: u64, floor: u64) -> u64 {
    quota.min(disk_room_bytes(available, held, floor))
}

/// Splash `DiskVolume`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Volume {
    pub available: u64,
    pub size: u64,
    pub local: bool,
    pub mount: String,
}

/// Splash `volumeOf`: the volume of `path`, or of its nearest existing
/// ancestor.
pub(crate) fn volume_of(path: &Path) -> Option<Volume> {
    #[cfg(test)]
    if let Some(volume) = VOLUMES.with(|volumes| {
        volumes
            .borrow()
            .iter()
            .find(|(root, _)| path.starts_with(root))
            .map(|(_, volume)| volume.clone())
    }) {
        return volume;
    }
    let mut directory = path.to_path_buf();
    loop {
        if directory.is_dir() {
            return measure_volume(&directory);
        }
        if !directory.pop() {
            return None;
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Volumes the tests give paths under these roots.
    pub(crate) static VOLUMES: std::cell::RefCell<Vec<(PathBuf, Option<Volume>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// The writer's queue depth and free-space reserve the tiers a test
    /// opens take.
    pub(crate) static TEST_WRITER: std::cell::Cell<Option<(usize, u64)>> =
        const { std::cell::Cell::new(None) };
}

/// The writer's queue depth and its free-space reserve over a volume whose
/// floor is `floor`.
fn writer_shape(floor: u64) -> (usize, u64) {
    #[cfg(test)]
    if let Some(shape) = TEST_WRITER.with(std::cell::Cell::get) {
        return shape;
    }
    (DEFAULT_QUEUE_DEPTH, floor)
}

#[cfg(target_os = "macos")]
fn measure_volume(directory: &Path) -> Option<Volume> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(directory.as_os_str().as_bytes()).ok()?;
    let mut status = std::mem::MaybeUninit::<libc::statfs>::zeroed();
    // SAFETY: `path` is NUL-terminated and `status` is writable statfs storage.
    if unsafe { libc::statfs(path.as_ptr(), status.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: statfs succeeded and filled `status`.
    let status = unsafe { status.assume_init() };
    let block = u64::from(status.f_bsize);
    // SAFETY: `f_mntonname` is a NUL-terminated C string from the kernel.
    let mount = unsafe { std::ffi::CStr::from_ptr(status.f_mntonname.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Some(Volume {
        available: status.f_bavail.saturating_mul(block),
        size: status.f_blocks.saturating_mul(block),
        local: status.f_flags & (libc::MNT_LOCAL as u32) != 0,
        mount,
    })
}

#[cfg(not(target_os = "macos"))]
fn measure_volume(directory: &Path) -> Option<Volume> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(directory.as_os_str().as_bytes()).ok()?;
    let mut status = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
    // SAFETY: `path` is NUL-terminated and `status` is writable statvfs storage.
    if unsafe { libc::statvfs(path.as_ptr(), status.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: statvfs succeeded and filled `status`.
    let status = unsafe { status.assume_init() };
    let block = status.f_frsize as u64;
    Some(Volume {
        available: (status.f_bavail as u64).saturating_mul(block),
        size: (status.f_blocks as u64).saturating_mul(block),
        local: true,
        mount: directory.display().to_string(),
    })
}

fn gibibytes(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB as f64)
}

/// Splash `automaticDiskRefusal`: why an automatic tier stays off on
/// `volume`, whose caches nobody holds come to `held` bytes.
pub(crate) fn automatic_disk_refusal(
    volume: Option<&Volume>,
    held: u64,
    minimum: u64,
) -> Option<String> {
    let Some(volume) = volume else {
        return Some("its volume cannot be measured".to_string());
    };
    if !volume.local {
        return Some(format!("{} is not a local volume", volume.mount));
    }
    let floor = disk_floor_bytes(volume.size);
    if disk_limit_bytes(AUTOMATIC_CACHE_DISK_BYTES, volume.available, held, floor) < minimum {
        return Some(format!(
            "{} available on {} is too little beyond the {} it keeps free",
            gibibytes(volume.available),
            volume.mount,
            gibibytes(floor)
        ));
    }
    None
}

/// The limit a tier of `cache_disk` keeps on `volume` (Splash `DiskSpace`):
/// an automatic one a quarter of the room, a size all of it; a volume the
/// tier does not follow leaves the size as it is.
fn tier_limit(cache_disk: CacheDisk, volume: Option<&Volume>, held: u64) -> u64 {
    let quota = cache_disk.nominal_bytes();
    match volume.filter(|volume| volume.local) {
        Some(volume) => {
            let floor = disk_floor_bytes(volume.size);
            match cache_disk {
                CacheDisk::Auto => disk_limit_bytes(quota, volume.available, held, floor),
                _ => disk_active_limit_bytes(quota, volume.available, held, floor),
            }
        }
        None => quota,
    }
}

/// Splash `persistentCacheNamespace`'s inputs.
pub(crate) struct CacheNamespace<'a> {
    pub target: ColdCacheFingerprint,
    pub draft: ColdCacheFingerprint,
    /// Block size, layers, KV heads, head size and the cache dtype.
    pub kv: (u32, u32, u32, u32, &'a str),
    /// GDN layers, conv history, conv channels, value heads, value and key
    /// head dims; the recurrent state is fp32.
    pub gdn: [u32; 6],
    /// Draft layers, KV heads, head size and sliding window.
    pub draft_geometry: (u32, u32, u32, u32),
    pub context_width: u32,
}

/// The `gdn`, `draft` and `context` lines: a state object's geometry.
pub(crate) fn state_geometry(
    gdn: [u32; 6],
    draft: (u32, u32, u32, u32),
    context_width: u32,
) -> String {
    let [layers, history, channels, groups, rows, cols] = gdn;
    let (draft_layers, heads, head, window) = draft;
    let mut text = format!("gdn {layers} {history} {channels} {groups} {rows} {cols}\n");
    text.push_str(&format!("draft {draft_layers} {heads} {head} {window}\n"));
    text.push_str(&format!("context {context_width} q4 64\n"));
    text
}

/// The canonical text Splash `persistentCacheNamespace` hashes.
pub(crate) fn cache_namespace_text(ns: &CacheNamespace<'_>) -> String {
    let (block, layers, heads, head, dtype) = ns.kv;
    let mut text = String::from("mlx-node-persistent-cache-v1\n");
    text.push_str(&ns.target.to_hex());
    text.push('\n');
    text.push_str(&ns.draft.to_hex());
    text.push('\n');
    text.push_str(&format!("kv {block} {layers} {heads} {head} {dtype}\n"));
    text.push_str(&state_geometry(ns.gdn, ns.draft_geometry, ns.context_width));
    text
}

/// The namespace (32 hex digits of the text's SHA-256) and the fingerprint
/// every state object of a persistent tier is keyed by.
pub(crate) fn cache_namespace(ns: &CacheNamespace<'_>) -> (String, ColdCacheFingerprint) {
    let text = cache_namespace_text(ns);
    let digest = Sha256::digest(text.as_bytes());
    let mut name = String::with_capacity(NAMESPACE_DIGITS);
    for byte in digest.iter().take(NAMESPACE_DIGITS / 2) {
        name.push_str(&format!("{byte:02x}"));
    }
    (
        name,
        ColdCacheFingerprint::from_components([text.as_bytes()]),
    )
}

/// A temporary tier dies with its process, so it is keyed by random bytes
/// and hashes no weights.
fn temporary_fingerprint() -> ColdCacheFingerprint {
    let random = uuid::Uuid::new_v4();
    ColdCacheFingerprint::from_components([b"mlx-node-temp-tier".as_slice(), random.as_bytes()])
}

/// A persistent tier's identity, built only when `--persistent-cache` asks
/// for one.
pub(crate) struct PersistentIdentity {
    pub namespace: String,
    pub state: ColdCacheFingerprint,
    pub kv: ColdCacheFingerprint,
}

enum TierHold {
    Temp { dir: PathBuf, _lock: File },
    Persistent { _lock: File },
}

/// One loaded model's open SSD tier.
#[allow(dead_code)]
pub(crate) struct DiskTier {
    pub manager: Arc<ColdCacheManager>,
    pub persistent: bool,
    /// The persistent namespace; empty for a temporary tier.
    pub namespace: String,
    pub limit: u64,
    /// The bytes of the root's caches nobody holds, this one's included.
    pub held: u64,
    pub state_fingerprint: ColdCacheFingerprint,
    pub kv_fingerprint: ColdCacheFingerprint,
    hold: Mutex<Option<TierHold>>,
    /// The root's other caches nobody held at open, oldest first; none for a
    /// temporary tier.
    others: Mutex<VecDeque<Idle>>,
}

impl DiskTier {
    /// Until [`Self::close`], and while no write failed for anything but
    /// room: the writer takes jobs.
    pub(crate) fn is_open(&self) -> bool {
        self.hold
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some()
            && self.manager.takes_tracked_writes()
    }

    /// Stops the writer and, for a temporary tier, removes its directory; a
    /// persistent tier first drains what it queued until `deadline`. Every
    /// write is refused afterwards. False when a persistent drain missed the
    /// deadline.
    pub(crate) fn close(&self, deadline: Instant) -> bool {
        let Some(hold) = self
            .hold
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        else {
            return true;
        };
        match hold {
            TierHold::Persistent { _lock } => {
                self.manager.hold_until_writer_exits(Box::new(_lock));
                self.manager
                    .close(deadline.saturating_duration_since(Instant::now()))
            }
            TierHold::Temp { dir, _lock } => {
                self.manager.close(Duration::ZERO);
                remove_temp_dir(&dir, deadline);
                true
            }
        }
    }

    /// Splash `DiskSpace::measure` over the root: one someone else removed
    /// stops counting, and while this tier and the others hold more than
    /// the limit the oldest other goes.
    pub(crate) fn measure(&self) {
        let mut others = self
            .others
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if others.is_empty() {
            return;
        }
        others.retain(|cache| {
            !matches!(fs::symlink_metadata(&cache.path),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound)
        });
        let own = self.manager.total_bytes();
        let mut others_bytes = others.iter().map(|cache| cache.bytes).sum::<u64>();
        while own.saturating_add(others_bytes) > self.limit {
            let Some(oldest) = others.pop_front() else {
                break;
            };
            others_bytes = others_bytes.saturating_sub(oldest.bytes);
            if remove_idle(&oldest) {
                tracing::info!(
                    "Persistent cache: removed {} ({}), the oldest cache nobody holds, to keep \
                     the cache root within {}",
                    oldest.path.display(),
                    gibibytes(oldest.bytes),
                    gibibytes(self.limit)
                );
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn dir(&self) -> PathBuf {
        self.manager.root().to_path_buf()
    }
}

impl Drop for DiskTier {
    fn drop(&mut self) {
        self.close(Instant::now());
    }
}

/// Removes a closed temporary tier's directory. The stopped writer may still
/// be finishing one write in it, so a removal that finds the directory
/// refilled tries again until `deadline`.
fn remove_temp_dir(dir: &Path, deadline: Instant) {
    let mut result = fs::remove_dir_all(dir);
    while let Err(error) = &result {
        if error.kind() == std::io::ErrorKind::NotFound || Instant::now() >= deadline {
            break;
        }
        std::thread::yield_now();
        result = fs::remove_dir_all(dir);
    }
    if let Err(error) = result
        && error.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!(
            "SSD cache: could not remove {} ({error}); the next start removes it",
            dir.display()
        );
    }
}

fn owned_by_me(meta: &fs::Metadata) -> bool {
    // SAFETY: geteuid has no preconditions and cannot fail.
    meta.uid() == unsafe { libc::geteuid() }
}

/// Takes `file`'s lock without waiting.
fn try_lock(file: &File) -> bool {
    // SAFETY: the descriptor is open for the call's duration.
    unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) == 0 }
}

/// The `lock` of the cache directory `dir` taken, when nobody holds it.
/// A directory whose lock is missing is nobody's.
enum Taken {
    Lock(File),
    NoLock,
    Held,
}

/// Splash `named`: a held lock gives its directory only while the
/// directory's path still names it. A removal of the namespace as a cache
/// nobody holds takes the lock away, so a process that held, or waited for,
/// the same lock may hold one that names nothing.
pub(crate) fn named(dir: &Path, lock: &File) -> bool {
    let (Ok(held), Ok(current)) = (lock.metadata(), fs::symlink_metadata(dir.join(LOCK_FILE)))
    else {
        return false;
    };
    held.dev() == current.dev() && held.ino() == current.ino()
}

fn take_lock(dir: &Path) -> Taken {
    match OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(dir.join(LOCK_FILE))
    {
        Ok(file) if try_lock(&file) && named(dir, &file) => Taken::Lock(file),
        Ok(_) => Taken::Held,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Taken::NoLock,
        Err(_) => Taken::Held,
    }
}

/// Removes every published temporary cache under `temp_root` whose lock
/// nobody holds (a crashed process's). A directory still being made has a
/// dot name and is never touched.
pub(crate) fn sweep_temp_leftovers(temp_root: &Path) -> usize {
    let Ok(entries) = fs::read_dir(temp_root) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with(TEMP_PREFIX) {
            continue;
        }
        let path = entry.path();
        let Ok(meta) = fs::symlink_metadata(&path) else {
            continue;
        };
        if !meta.is_dir() || !owned_by_me(&meta) {
            continue;
        }
        let held = match take_lock(&path) {
            Taken::Lock(lock) => Some(lock),
            Taken::NoLock => None,
            Taken::Held => continue,
        };
        if fs::remove_dir_all(&path).is_ok() {
            removed += 1;
        }
        drop(held);
    }
    removed
}

/// Makes a temporary cache: created under a dot name, its lock taken, then
/// published under the name a sweep matches.
fn make_temp_dir(temp_root: &Path) -> Result<(PathBuf, File), String> {
    use std::os::unix::ffi::OsStrExt;
    let template = temp_root.join(format!("{TEMP_CREATING_PREFIX}XXXXXX"));
    let mut bytes = template.as_os_str().as_bytes().to_vec();
    bytes.push(0);
    // SAFETY: `bytes` is a writable NUL-terminated template mkdtemp fills in.
    let made = unsafe { libc::mkdtemp(bytes.as_mut_ptr().cast()) };
    if made.is_null() {
        return Err(format!(
            "make a temporary cache in {}: {}",
            temp_root.display(),
            std::io::Error::last_os_error()
        ));
    }
    bytes.pop();
    let creating = PathBuf::from(std::ffi::OsStr::from_bytes(&bytes));
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(creating.join(LOCK_FILE))
        .map_err(|error| format!("make the temporary cache's lock: {error}"))?;
    if !try_lock(&lock) {
        let _ = fs::remove_dir_all(&creating);
        return Err("lock the temporary cache".to_string());
    }
    let suffix = creating
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix(TEMP_CREATING_PREFIX))
        .ok_or_else(|| "temporary cache name".to_string())?
        .to_string();
    let published = temp_root.join(format!("{TEMP_PREFIX}{suffix}"));
    if let Err(error) = fs::rename(&creating, &published) {
        let _ = fs::remove_dir_all(&creating);
        return Err(format!("publish the temporary cache: {error}"));
    }
    Ok((published, lock))
}

/// A namespace directory of the cache root that nobody holds (Splash
/// `CacheDirectory::Idle`), with its lock taken while it is looked at.
struct Idle {
    path: PathBuf,
    bytes: u64,
    last_use: SystemTime,
}

fn namespace_name(name: &str) -> bool {
    name.len() == NAMESPACE_DIGITS
        && name
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Bytes and the newest modification time of the files in `dir`.
fn usage(dir: &Path) -> (u64, SystemTime) {
    let mut bytes = 0u64;
    let mut last_use = SystemTime::UNIX_EPOCH;
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if !meta.is_file() {
                continue;
            }
            if entry.file_name() != LOCK_FILE {
                bytes = bytes.saturating_add(meta.len());
            }
            if let Ok(modified) = meta.modified() {
                last_use = last_use.max(modified);
            }
        }
    }
    (bytes, last_use)
}

/// Splash `CacheDirectory::idle`: the root's namespaces whose lock is free.
fn idle_caches(root: &Path) -> Vec<Idle> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut idle = Vec::new();
    for entry in entries.flatten() {
        if !namespace_name(&entry.file_name().to_string_lossy()) {
            continue;
        }
        let path = entry.path();
        let Ok(meta) = fs::symlink_metadata(&path) else {
            continue;
        };
        if !meta.is_dir() || !owned_by_me(&meta) {
            continue;
        }
        if matches!(take_lock(&path), Taken::Held) {
            continue;
        }
        let (bytes, last_use) = usage(&path);
        idle.push(Idle {
            path,
            bytes,
            last_use,
        });
    }
    idle
}

/// Removes `cache` when its lock is still free.
fn remove_idle(cache: &Idle) -> bool {
    let held = match take_lock(&cache.path) {
        Taken::Lock(lock) => Some(lock),
        Taken::NoLock => None,
        Taken::Held => return false,
    };
    let removed = fs::remove_dir_all(&cache.path).is_ok();
    drop(held);
    removed
}

enum PersistentOpen {
    Open { dir: PathBuf, lock: File },
    Held(PathBuf),
    Failed(String),
}

/// Splash `CacheDirectory::open`: the namespace's directory with its lock,
/// waiting up to [`cache_lock_wait`] for another process to let go.
fn open_persistent_dir(root: &Path, namespace: &str) -> PersistentOpen {
    if let Err(error) = fs::create_dir_all(root) {
        return PersistentOpen::Failed(format!("make {}: {error}", root.display()));
    }
    let dir = root.join(namespace);
    let deadline = Instant::now() + cache_lock_wait();
    let lock = loop {
        // Another process that removes the namespace as a cache nobody holds
        // can take its directory away before this one opens its lock: the
        // next turn makes it again, empty.
        if let Err(error) = fs::create_dir(&dir)
            && error.kind() != std::io::ErrorKind::AlreadyExists
        {
            return PersistentOpen::Failed(format!("make {}: {error}", dir.display()));
        }
        match fs::symlink_metadata(&dir) {
            Ok(meta) if meta.is_dir() && owned_by_me(&meta) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            _ => {
                return PersistentOpen::Failed(format!(
                    "{} belongs to another user or is not a directory",
                    dir.display()
                ));
            }
        }
        let lock = match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(dir.join(LOCK_FILE))
        {
            Ok(lock) => lock,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if Instant::now() >= deadline {
                    return PersistentOpen::Held(dir);
                }
                continue;
            }
            Err(error) => return PersistentOpen::Failed(format!("open its lock: {error}")),
        };
        while !try_lock(&lock) {
            if Instant::now() >= deadline {
                return PersistentOpen::Held(dir);
            }
            std::thread::sleep(Duration::from_millis(200).min(deadline - Instant::now()));
        }
        // A removal may have taken the lock away while this process waited
        // for it: the namespace is whoever holds the lock its path names now.
        if named(&dir, &lock) {
            break lock;
        }
        if Instant::now() >= deadline {
            return PersistentOpen::Held(dir);
        }
    };
    // The lock's time says when the namespace was last opened.
    let _ = lock.set_modified(SystemTime::now());
    PersistentOpen::Open { dir, lock }
}

/// Splash `openDiskTier`: the tier of a load whose state is `state_bytes`,
/// or `None` (logged) when it stays off. `identity` runs only when the
/// options read one ([`DiskTierOptions::reads_identity`]).
pub(crate) fn open_disk_tier(
    options: &DiskTierOptions,
    identity: impl FnOnce() -> Option<PersistentIdentity>,
    state_bytes: u64,
) -> Option<Arc<DiskTier>> {
    // The cache root's caches nobody holds, looked at once: room an
    // automatic persistent tier may take; after the open, which holds this
    // start's own, the stale ones go whatever the tier, and the rest share a
    // persistent tier's limit.
    let caches = options
        .cache_root()
        .map_or_else(Vec::new, |root| idle_caches(&root));
    let tier = (options.cache_disk != CacheDisk::Off)
        .then(|| open_tier(options, identity, state_bytes, &caches))
        .flatten();
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

/// Splash `CacheDirectory::removeStale`: each cache unused for
/// [`STALE_CACHE`] goes while nobody holds it; the others are kept.
fn remove_stale(caches: Vec<Idle>) -> Vec<Idle> {
    let now = SystemTime::now();
    caches
        .into_iter()
        .filter(|cache| {
            let stale = now
                .duration_since(cache.last_use)
                .is_ok_and(|age| age > STALE_CACHE);
            if stale && remove_idle(cache) {
                tracing::info!(
                    "SSD cache: removed {}, unused for 14 days",
                    cache.path.display()
                );
            }
            !stale
        })
        .collect()
}

fn open_tier(
    options: &DiskTierOptions,
    identity: impl FnOnce() -> Option<PersistentIdentity>,
    state_bytes: u64,
    caches: &[Idle],
) -> Option<Arc<DiskTier>> {
    let persistent = options.reads_identity().then(identity).flatten();
    let temp_root = options.temp_root();
    sweep_temp_leftovers(&temp_root);
    let automatic = options.cache_disk == CacheDisk::Auto;
    let minimum = state_bytes.saturating_mul(2).saturating_add(GIB);
    let temp_volume = volume_of(&temp_root);
    let temporary_refusal = automatic
        .then(|| automatic_disk_refusal(temp_volume.as_ref(), 0, minimum))
        .flatten();

    let mut root_refusal = None;
    if options.persistent {
        match (options.cache_root(), persistent) {
            (Some(root), Some(identity)) => {
                let held: u64 = caches.iter().map(|cache| cache.bytes).sum();
                let root_volume = volume_of(&root);
                root_refusal = automatic
                    .then(|| automatic_disk_refusal(root_volume.as_ref(), held, minimum))
                    .flatten();
                if root_refusal.is_none() {
                    match open_persistent(options, &root, identity, held, root_volume.as_ref()) {
                        Ok(tier) => return Some(tier),
                        Err(line) => tracing::warn!("{line}"),
                    }
                }
            }
            (None, _) => tracing::warn!(
                "Persistent cache disabled (no home directory for its root); this process \
                 keeps a temporary cache."
            ),
            (Some(_), None) => tracing::warn!(
                "Persistent cache disabled (the model's identity could not be read); this \
                 process keeps a temporary cache."
            ),
        }
    }
    if let Some(temporary) = temporary_refusal {
        let reason = match (&root_refusal, options.persistent) {
            (Some(root), _) => format!("for its root, {root}; for temporary files, {temporary}"),
            (None, true) => format!("for temporary files, {temporary}"),
            (None, false) => temporary,
        };
        tracing::warn!(
            "SSD cache off: {reason}; the state staging the plan set aside stays unused."
        );
        return None;
    }
    if let Some(root) = root_refusal {
        tracing::warn!("Persistent cache off: {root}; the SSD cache keeps temporary files.");
    }
    match open_temporary(options, &temp_root, temp_volume.as_ref()) {
        Ok(tier) => Some(tier),
        Err(error) => {
            tracing::warn!(
                "SSD cache off: {error}; the state staging the plan set aside stays unused."
            );
            None
        }
    }
}

fn open_temporary(
    options: &DiskTierOptions,
    temp_root: &Path,
    volume: Option<&Volume>,
) -> Result<Arc<DiskTier>, String> {
    let limit = tier_limit(options.cache_disk, volume, 0);
    let (dir, lock) = make_temp_dir(temp_root)?;
    let (depth, reserve) = writer_shape(volume.map_or(0, |volume| disk_floor_bytes(volume.size)));
    let manager = match ColdCacheManager::open_at(dir.clone(), limit, reserve, depth) {
        Ok(manager) => manager,
        Err(error) => {
            let _ = fs::remove_dir_all(&dir);
            return Err(error);
        }
    };
    let fingerprint = temporary_fingerprint();
    tracing::info!(
        "SSD cache: temporary files in {}, limit {}",
        dir.display(),
        gibibytes(limit)
    );
    Ok(Arc::new(DiskTier {
        manager: Arc::new(manager),
        persistent: false,
        namespace: String::new(),
        limit,
        held: 0,
        state_fingerprint: fingerprint,
        kv_fingerprint: fingerprint,
        hold: Mutex::new(Some(TierHold::Temp { dir, _lock: lock })),
        others: Mutex::new(VecDeque::new()),
    }))
}

fn open_persistent(
    options: &DiskTierOptions,
    root: &Path,
    identity: PersistentIdentity,
    held: u64,
    volume: Option<&Volume>,
) -> Result<Arc<DiskTier>, String> {
    let (dir, lock) = match open_persistent_dir(root, &identity.namespace) {
        PersistentOpen::Open { dir, lock } => (dir, lock),
        PersistentOpen::Held(dir) => {
            return Err(format!(
                "Persistent cache {} is in use by another process; this one keeps a temporary \
                 cache.",
                dir.display()
            ));
        }
        PersistentOpen::Failed(error) => {
            return Err(format!(
                "Persistent cache disabled ({error}); this process keeps a temporary cache."
            ));
        }
    };
    let limit = tier_limit(options.cache_disk, volume, held);
    let (depth, reserve) = writer_shape(volume.map_or(0, |volume| disk_floor_bytes(volume.size)));
    let manager =
        ColdCacheManager::open_at(dir.clone(), limit, reserve, depth).map_err(|error| {
            format!("Persistent cache disabled ({error}); this process keeps a temporary cache.")
        })?;
    tracing::info!(
        "SSD cache: persistent files in {}, limit {}",
        dir.display(),
        gibibytes(limit)
    );
    Ok(Arc::new(DiskTier {
        manager: Arc::new(manager),
        persistent: true,
        namespace: identity.namespace,
        limit,
        held,
        state_fingerprint: identity.state,
        kv_fingerprint: identity.kv,
        hold: Mutex::new(Some(TierHold::Persistent { _lock: lock })),
        others: Mutex::new(VecDeque::new()),
    }))
}

#[cfg(test)]
mod tests;
