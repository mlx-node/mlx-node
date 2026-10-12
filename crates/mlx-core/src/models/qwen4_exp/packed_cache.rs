//! Optional, lossless SSD cache of already imported quantized matrix chunks.
//! Opening a Store only fingerprints source metadata. Entries are created on
//! demand, atomically, and contain at most one bounded row/expert chunk.
use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use napi::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{debug, info};

use super::weights::{MAX_READ_BYTES, Weight};
use crate::array::{DType, MxArray};

const MAGIC: &[u8; 8] = b"Q4PACK02";
const MAX_HEADER: usize = 4096;
const DISK_BUDGET: u64 = 96 << 30;
const FREE_RESERVE: u64 = 16 << 30;
/// Idle identity dirs are stale after this; `.pack`/budget mtime stamps on
/// verified reads keep a live dir young.
const PACKED_SWEEP_MIN_AGE: Duration = Duration::from_secs(3600);
const DIR_PREFIX: &str = ".mlx-qwen4-packed-v2-";
/// Optional total cap across all identity dirs; unset/<=0 keeps the per-dir
/// DISK_BUDGET behaviour.
const TOTAL_CAP_ENV: &str = "MLX_QWEN4_PACKED_CACHE_MAX_GB";

fn err(e: impl std::fmt::Display) -> Error {
    Error::from_reason(e.to_string())
}

#[derive(Serialize, Deserialize)]
struct ArrayInfo {
    dtype: String,
    shape: Vec<i64>,
    bytes: usize,
}
#[derive(Serialize, Deserialize)]
struct Header {
    key: String,
    group: i32,
    bits: i32,
    mode: String,
    // values, scales, optional biases
    arrays: Vec<ArrayInfo>,
}

/// Fully verified owned bytes; no file mapping or path-based trust survives a read.
pub(super) struct VerifiedChunk {
    header: Header,
    data: Vec<u8>,
    file_len: u64,
}

pub(super) struct PackedCache {
    root: PathBuf,
    /// `.manifest` bytes written by `new_at`; `write` republishes them if a
    /// sweep deletes the file (or the whole dir) mid-session. None only for
    /// tests constructing the struct literally.
    manifest: Option<Vec<u8>>,
}

pub(super) fn chunk_key(
    path: &Path,
    offset: u64,
    bytes: u64,
    rows: usize,
    width: usize,
    ty: u32,
) -> String {
    let mut hash = Sha256::new();
    hash.update(path.as_os_str().as_bytes());
    for n in [offset, bytes, rows as u64, width as u64, ty as u64] {
        hash.update(n.to_le_bytes());
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn packed_identity(files: impl Iterator<Item = PathBuf>) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(MAGIC);
    for path in files {
        let canonical = path.canonicalize().map_err(err)?;
        let meta = fs::metadata(&canonical).map_err(err)?;
        hash.update(canonical.as_os_str().as_bytes());
        for n in [
            meta.dev(),
            meta.ino(),
            meta.len(),
            meta.mtime() as u64,
            meta.mtime_nsec() as u64,
            meta.ctime() as u64,
            meta.ctime_nsec() as u64,
        ] {
            hash.update(n.to_le_bytes());
        }
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Recompute an identity dir's source fingerprint from its `.manifest`. Any
/// missing source file or unreadable manifest is Err so callers can treat the
/// dir as orphaned.
fn identity_from_manifest(dir: &Path) -> Result<String> {
    let data = fs::read(dir.join(".manifest")).map_err(err)?;
    let mut lines = data.split(|&b| b == b'\n');
    let base = lines
        .next()
        .and_then(|line| line.strip_prefix(b"path=".as_slice()))
        .map(|raw| PathBuf::from(OsStr::from_bytes(raw)))
        .ok_or_else(|| err("packed manifest missing path="))?;
    let files: Vec<PathBuf> = lines
        .filter_map(|line| line.strip_prefix(b"file=".as_slice()))
        .map(|raw| base.join(OsStr::from_bytes(raw)))
        .collect();
    if files.is_empty() {
        return Err(err("packed manifest lists no files"));
    }
    packed_identity(files.into_iter())
}

/// Serialize canonical source paths into `.manifest` bytes: the common base
/// on `path=` and each file relative to it on `file=` lines.
fn pack_manifest(canonical: &[PathBuf]) -> Vec<u8> {
    let base = canonical
        .first()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_default();
    let mut manifest = b"path=".to_vec();
    manifest.extend_from_slice(base.as_os_str().as_bytes());
    manifest.push(b'\n');
    for path in canonical {
        let rel = path.strip_prefix(&base).unwrap_or(path);
        manifest.extend_from_slice(b"file=");
        manifest.extend_from_slice(rel.as_os_str().as_bytes());
        manifest.push(b'\n');
    }
    manifest
}

fn dir_name_identity(dir: &Path) -> Option<&str> {
    dir.file_name()?.to_str()?.strip_prefix(DIR_PREFIX)
}

/// Newest mtime over the dir and every direct entry (.manifest, budget,
/// .pack, in-flight .tmp); a present-but-unreadable mtime counts as fresh
/// (None) so the dir is never deleted on a bad clock.
fn packed_dir_mtime(dir: &Path) -> Option<SystemTime> {
    let mut newest = fs::metadata(dir).ok()?.modified().ok()?;
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        newest = newest.max(entry.metadata().ok()?.modified().ok()?);
    }
    Some(newest)
}

/// True when the newest witness in `dir` is older than the sweep grace. Every
/// deletion goes through this so a dir another process just opened or a
/// writer mid-staging is never removed.
fn packed_dir_idle(dir: &Path) -> bool {
    packed_dir_mtime(dir).is_some_and(|newest| newest < SystemTime::now() - PACKED_SWEEP_MIN_AGE)
}

/// None keeps the dir (manifest-less dirs are never orphan-deleted: their
/// staleness cannot be proven). The idle gate applies to every reason: a
/// fresh dir keeps its packs until the grace window passes.
fn packed_orphan_reason(dir: &Path) -> Option<&'static str> {
    let manifest = dir.join(".manifest");
    if !manifest.is_file() || !packed_dir_idle(dir) {
        return None;
    }
    match identity_from_manifest(dir) {
        Err(_) => return Some("unreadable manifest or missing source"),
        Ok(identity) if dir_name_identity(dir) != Some(identity.as_str()) => {
            return Some("source fingerprint changed");
        }
        Ok(_) => {}
    }
    Some("idle beyond sweep grace")
}

fn dir_size(dir: &Path) -> u64 {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// Bytes actually held by .pack files — the same quantity the budget ledger
/// records.
fn pack_bytes(dir: &Path) -> u64 {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| e.path().extension().is_some_and(|s| s == "pack"))
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// Best-effort sibling spend estimate: the ledger u64 when it reads, else a
/// live .pack byte count. A ledger value can lag reality (a skipped/failed
/// replacement write leaves the old target on disk but absent from `used`),
/// so a readable ledger is floored at the real byte count — the estimate can
/// overcount pending spend, never undercount. Advisory only — the sibling's
/// flock stays with its writer and any failure degrades to the byte count,
/// never to an error.
fn sibling_pack_estimate(dir: &Path) -> u64 {
    let mut n = [0u8; 8];
    let from_ledger = File::open(dir.join("budget"))
        .and_then(|mut f| f.read_exact(&mut n))
        .is_ok();
    if from_ledger {
        u64::from_le_bytes(n).max(pack_bytes(dir))
    } else {
        pack_bytes(dir)
    }
}

/// Root-wide spend: `current_used` for `current` plus `sibling_bytes` over
/// every other identity dir under `parent`. Read errors count as 0 — this is
/// an advisory estimate, never a load failure.
fn root_pack_spend(
    parent: &Path,
    current: &Path,
    current_used: u64,
    sibling_bytes: impl Fn(&Path) -> u64,
) -> u64 {
    let siblings: u64 = fs::read_dir(parent)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p != current && p.is_dir() && dir_name_identity(p).is_some())
                .map(|p| sibling_bytes(&p))
                .sum()
        })
        .unwrap_or(0);
    current_used.saturating_add(siblings)
}

/// Try the writer flock non-blockingly. Some(_) keeps the fd as a guard; the
/// inner None means no ledger exists (nothing can be holding a lock). None
/// means a live writer or an error: keep the dir.
fn try_budget_lock(dir: &Path) -> Option<Option<File>> {
    let path = dir.join("budget");
    // macOS flock requires a writable fd for LOCK_EX; a read-only ledger falls
    // back to read-only and relies on the kernel accepting or refusing it.
    let file = match OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .or_else(|_| File::open(&path))
    {
        Ok(f) => f,
        // No ledger means no writer ever locked it; treat as free.
        Err(e) if e.kind() == ErrorKind::NotFound => return Some(None),
        Err(_) => return None,
    };
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return None;
    }
    // The ledger could have been unlinked between open and flock; only act on
    // the dir whose ledger we actually hold.
    let held = file.metadata().ok();
    let live = fs::metadata(&path).ok();
    match (held, live) {
        (Some(a), Some(b)) if a.dev() == b.dev() && a.ino() == b.ino() => Some(Some(file)),
        _ => None,
    }
}

/// Authoritative sibling spend for the under-root-lock recheck. A crash
/// between `record` and the temp-file rename leaves a ledger inflated by a
/// reservation that never became a `.pack`: if the sibling's budget lock is
/// free (no live writer), reconcile the ledger to real pack bytes — phantom
/// spend must not veto other identities' writes forever. A busy lock means a
/// live writer, so its reservation is real pending spend and is trusted.
fn reconciled_sibling_pack_spend(dir: &Path) -> u64 {
    let real = pack_bytes(dir);
    match try_budget_lock(dir) {
        Some(guard) => {
            if let Ok(mut ledger) = OpenOptions::new().write(true).open(dir.join("budget")) {
                let _ = ledger.write_all(&real.to_le_bytes());
                let _ = ledger.set_len(8);
            }
            drop(guard);
            real
        }
        None => sibling_pack_estimate(dir),
    }
}

fn locked_remove(dir: &Path, reason: &str, still_stale: impl Fn(&Path) -> bool) -> u64 {
    let Some(_guard) = try_budget_lock(dir) else {
        debug!(
            "packed cache: keeping {} (writer holds budget lock)",
            dir.display()
        );
        return 0;
    };
    // Recheck under the held lock: the dir may have been rebuilt while we
    // waited, in which case our lock no longer proves staleness.
    if !still_stale(dir) {
        return 0;
    }
    let size = dir_size(dir);
    match fs::remove_dir_all(dir) {
        Ok(()) => {
            debug!(
                "packed cache: removed {} ({reason}, {size} bytes)",
                dir.display()
            );
            size
        }
        Err(e) => {
            debug!("packed cache: failed to remove {}: {e}", dir.display());
            0
        }
    }
}

fn remove_packed_dir(dir: &Path, reason: &str) -> u64 {
    locked_remove(dir, reason, |d| packed_orphan_reason(d).is_some())
}

/// Best-effort mtime stamp; keeps the dir young for idle-expiry and LRU.
fn stamp_mtime(path: &Path) {
    let Ok(file) = OpenOptions::new().write(true).open(path) else {
        return;
    };
    let _ = file.set_modified(SystemTime::now());
}

fn packed_total_cap() -> Option<u64> {
    let raw = std::env::var(TOTAL_CAP_ENV).ok()?;
    raw.trim()
        .parse::<f64>()
        .ok()
        .filter(|gb| gb.is_finite() && *gb > 0.0)
        .map(|gb| (gb * (1u64 << 30) as f64) as u64)
}

/// Root-wide admission lock for the total cap: one flock per cache root
/// serializes the whole estimate→evict→reserve critical section across
/// processes. The name is a dot-file under the root, so `dir_name_identity`
/// keeps it invisible to `sweep_packed_cache`, `enforce_packed_total_cap`
/// and `root_pack_spend`, and it is never unlinked (flock mutual exclusion
/// depends on a stable inode). None on busy or error — cache writes never
/// stall.
fn packed_root_cap_lock(parent: &Path) -> Option<File> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(parent.join(".mlx-qwen4-packed-cap.lock"))
        .ok()?;
    // SAFETY: `file` owns the descriptor for the flock call, and the returned
    // guard keeps it — and the lock — alive until the caller drops it.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return None;
    }
    Some(file)
}

/// Enforce the optional total cap: evict least-recently-used sibling dirs
/// first, then shed the oldest `.pack` files inside the current dir under its
/// own ledger lock.
fn enforce_packed_total_cap(parent: &Path, current: &Path, cap: u64) {
    let mut dirs: Vec<(PathBuf, Option<SystemTime>)> = fs::read_dir(parent)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir() && dir_name_identity(p).is_some())
                .map(|p| {
                    let mtime = packed_dir_mtime(&p);
                    (p, mtime)
                })
                .collect()
        })
        .unwrap_or_default();
    dirs.sort_by_key(|(_, mtime)| mtime.unwrap_or(SystemTime::UNIX_EPOCH));
    let mut total: u64 = dirs.iter().map(|(p, _)| dir_size(p)).sum();
    if !dirs.iter().any(|(p, _)| p == current) {
        // new_at creates the current dir eagerly, but tests may sweep roots
        // whose PackedCache was constructed without one.
        total += dir_size(current);
    }
    // Same idle grace as the orphan sweep: a dir that became active
    // mid-enforcement is rechecked under the budget lock before removal, and
    // manifest-less dirs are never cap-evicted.
    let cutoff = SystemTime::now() - PACKED_SWEEP_MIN_AGE;
    let mut reclaimed = 0u64;
    for (dir, snapshot) in dirs.iter().filter(|(p, _)| p != current) {
        if total <= cap {
            break;
        }
        if !snapshot.is_some_and(|mtime| mtime < cutoff) {
            continue;
        }
        let removed = locked_remove(dir, "total cap LRU eviction", |d| {
            dir_name_identity(d).is_some()
                && d.join(".manifest").is_file()
                && packed_dir_mtime(d)
                    .is_some_and(|mtime| snapshot.is_some_and(|snap| mtime <= snap))
        });
        total = total.saturating_sub(removed);
        reclaimed += removed;
    }
    if total > cap {
        // Shed oldest .pack files in the current dir under its own budget lock.
        let Some(_guard) = try_budget_lock(current) else {
            if reclaimed > 0 {
                info!("packed cache: reclaimed {reclaimed} bytes under {TOTAL_CAP_ENV}");
            }
            return;
        };
        let mut packs: Vec<(SystemTime, PathBuf, u64)> = fs::read_dir(current)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|s| s == "pack"))
                    .filter_map(|p| {
                        let m = fs::metadata(&p).ok()?;
                        Some((m.modified().ok()?, p, m.len()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        packs.sort_by_key(|(t, _, _)| *t);
        for (_, path, len) in packs {
            if total <= cap {
                break;
            }
            if fs::remove_file(&path).is_ok() {
                debug!("packed cache: shed {} (total cap)", path.display());
                total = total.saturating_sub(len);
                reclaimed += len;
            }
        }
        // Re-reconcile the ledger after removals.
        let _ = OpenOptions::new()
            .write(true)
            .open(current.join("budget"))
            .and_then(|mut f| f.write_all(&pack_bytes(current).to_le_bytes()));
    }
    if reclaimed > 0 {
        info!("packed cache: reclaimed {reclaimed} bytes under {TOTAL_CAP_ENV}");
    }
}

/// Sweep sibling `.mlx-qwen4-packed-v2-*` dirs for orphans and enforce the
/// optional total cap. Never fatal.
fn sweep_packed_cache(parent: &Path, current: &Path) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Ok(entries) = fs::read_dir(parent) else {
            return;
        };
        for entry in entries.flatten() {
            let dir = entry.path();
            if dir == current || !dir.is_dir() || dir_name_identity(&dir).is_none() {
                continue;
            }
            if let Some(reason) = packed_orphan_reason(&dir) {
                remove_packed_dir(&dir, reason);
            }
        }
        // Crash-leftover `{uuid}.tmp` files carry no ledger bytes: drop them
        // once they age past the sweep grace (a live writer's tmp is fresh).
        if let Ok(entries) = fs::read_dir(current) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|s| s == "tmp")
                    && let Ok(mtime) = entry.metadata().and_then(|m| m.modified())
                    && mtime < SystemTime::now() - PACKED_SWEEP_MIN_AGE
                    && fs::remove_file(&path).is_ok()
                {
                    debug!(
                        "packed cache: removed {} (stale staging tmp)",
                        path.display()
                    );
                }
            }
        }
        if let Some(cap) = packed_total_cap() {
            enforce_packed_total_cap(parent, current, cap);
        }
    }));
}

/// Serialize writers in-process: BSD flock is per-process, so two threads
/// opening the ledger could both hold LOCK_EX.
fn writer_mutex() -> &'static std::sync::Mutex<()> {
    static MUTEX: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    MUTEX.get_or_init(|| std::sync::Mutex::new(()))
}

/// Remove the oldest `.pack` files in `dir` (by mtime — the recency witness
/// stamped by verified reads) until the remaining pack bytes fit `headroom`,
/// or nothing is left to shed. Returns the real remaining pack bytes, i.e.
/// the new `used`. `exclude` protects the write's own target. Callers hold
/// the dir's budget lock, so no concurrent writer can shed underneath us and
/// `enforce_packed_total_cap`'s own-packs phase cannot (it never wins our
/// flock) — this is the only place self-shedding can happen during a write.
fn shed_oldest_packs(dir: &Path, exclude: &Path, headroom: u64) -> u64 {
    let mut packs: Vec<(SystemTime, PathBuf, u64)> = fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|s| s == "pack") && p != exclude)
                .filter_map(|p| {
                    let meta = fs::metadata(&p).ok()?;
                    Some((meta.modified().ok()?, p.clone(), meta.len()))
                })
                .collect()
        })
        .unwrap_or_default();
    packs.sort_by_key(|(mtime, _, _)| *mtime);
    let mut used: u64 = packs.iter().map(|(_, _, len)| len).sum();
    for (_, path, len) in packs {
        if used <= headroom {
            break;
        }
        if fs::remove_file(&path).is_ok() {
            debug!("packed cache: shed {} to admit a new write", path.display());
            used = used.saturating_sub(len);
        }
    }
    used
}

impl PackedCache {
    pub fn new(source: &Path, files: BTreeSet<PathBuf>) -> Result<Self> {
        // The source checkpoint may live on a slow external volume. Keep the
        // reusable working set in the OS application cache by default.
        let root = std::env::var_os("MLX_QWEN4_PACKED_CACHE_DIR")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|home| {
                    PathBuf::from(home).join(if cfg!(target_os = "macos") {
                        "Library/Caches/mlx-node/qwen4-packed"
                    } else {
                        ".cache/mlx-node/qwen4-packed"
                    })
                })
            })
            .unwrap_or_else(|| source.parent().unwrap_or(Path::new(".")).to_path_buf());
        Self::new_at(&root, files)
    }

    pub(super) fn new_at(root: &Path, files: BTreeSet<PathBuf>) -> Result<Self> {
        let mut canonical = Vec::with_capacity(files.len());
        for path in &files {
            canonical.push(path.canonicalize().map_err(err)?);
        }
        let identity = packed_identity(canonical.iter().cloned())?;
        let dir = root.join(format!("{DIR_PREFIX}{identity}"));
        // Eagerly create the identity dir and record its provenance so the
        // sweep can re-verify staleness later. A missing/failed manifest is
        // harmless: manifest-less dirs are never orphan-deleted.
        let manifest = pack_manifest(&canonical);
        if fs::create_dir_all(&dir).is_ok() {
            let _ = fs::write(dir.join(".manifest"), &manifest);
        }
        let cache = Self {
            root: dir,
            manifest: Some(manifest),
        };
        sweep_packed_cache(root, &cache.root);
        Ok(cache)
    }

    fn path(&self, key: &str) -> PathBuf {
        self.root.join(format!("{key}.pack"))
    }

    pub fn read(&self, key: &str) -> Result<Option<(Weight, u64)>> {
        self.read_verified(key, MAX_READ_BYTES)?
            .map(VerifiedChunk::into_weight)
            .transpose()
    }

    /// IO and hashing run without MLX arrays or Store mutation. Bounds are
    /// validated before spawning workers and enforced again on each file.
    pub(super) fn read_batch(
        &self,
        requests: &[(String, u64)],
    ) -> Result<Vec<Option<VerifiedChunk>>> {
        let total = requests
            .iter()
            .try_fold(0u64, |n, (_, bytes)| n.checked_add(*bytes));
        if requests.len() > 24 || total.is_none_or(|n| n > MAX_READ_BYTES) {
            return Err(err("Packed batch exceeds staging budget"));
        }
        let workers = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(4)
            .min(requests.len());
        if workers <= 1 {
            return Ok(requests
                .iter()
                .map(|(key, bytes)| self.read_verified(key, *bytes).ok().flatten())
                .collect());
        }
        std::thread::scope(|scope| {
            let mut jobs = Vec::new();
            for worker in 0..workers {
                jobs.push(scope.spawn(move || {
                    (worker..requests.len())
                        .step_by(workers)
                        .map(|i| {
                            let (key, bytes) = &requests[i];
                            (i, self.read_verified(key, *bytes).ok().flatten())
                        })
                        .collect::<Vec<_>>()
                }));
            }
            let mut out: Vec<_> = (0..requests.len()).map(|_| None).collect();
            for job in jobs {
                let completed = job
                    .join()
                    .map_err(|_| err("Packed reader worker panicked"))?;
                for (i, chunk) in completed {
                    out[i] = chunk;
                }
            }
            Ok(out)
        })
    }

    fn read_verified(&self, key: &str, payload_limit: u64) -> Result<Option<VerifiedChunk>> {
        let mut file = match File::open(self.path(key)) {
            Ok(f) => f,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(err(e)),
        };
        let len = file.metadata().map_err(err)?.len();
        // All bounds and the checksum are checked before creating MLX arrays.
        if !(44..=44 + MAX_HEADER as u64 + MAX_READ_BYTES).contains(&len) {
            return Err(err("Packed chunk exceeds staging budget"));
        }
        let mut prefix = [0u8; 44];
        file.read_exact(&mut prefix).map_err(err)?;
        let header_len = u32::from_le_bytes(
            prefix[8..12]
                .try_into()
                .map_err(|_| err("Invalid packed chunk header length field"))?,
        ) as usize;
        if &prefix[..8] != MAGIC || header_len > MAX_HEADER || 44 + header_len as u64 > len {
            return Err(err("Invalid packed chunk header"));
        }
        let mut header_bytes = vec![0; header_len];
        file.read_exact(&mut header_bytes).map_err(err)?;
        let header: Header = serde_json::from_slice(&header_bytes).map_err(err)?;
        if header.key != key
            || !(2..=3).contains(&header.arrays.len())
            || !(2..=8).contains(&header.bits)
            || ![16, 32, 64, 128].contains(&header.group)
            || header.mode.len() > 16
        {
            return Err(err("Invalid packed chunk quantization metadata"));
        }
        let mut total = 0usize;
        for a in &header.arrays {
            let element_bytes = match a.dtype.as_str() {
                "U32" | "F32" => 4,
                "F16" | "BF16" => 2,
                "U8" | "I8" => 1,
                _ => return Err(err("Invalid packed chunk dtype")),
            };
            if a.shape.is_empty() || a.shape.len() > 4 || a.shape.iter().any(|&d| d <= 0) {
                return Err(err("Invalid packed chunk shape"));
            }
            let expected = a
                .shape
                .iter()
                .try_fold(element_bytes, |n: usize, &d| n.checked_mul(d as usize));
            if expected != Some(a.bytes) {
                return Err(err("Invalid packed chunk array length"));
            }
            total = total
                .checked_add(a.bytes)
                .ok_or_else(|| err("Packed chunk size overflow"))?;
        }
        if total as u64 > MAX_READ_BYTES.min(payload_limit)
            || total as u64 + header_len as u64 + 44 != len
        {
            return Err(err("Truncated or oversized packed chunk"));
        }
        let mut data = vec![0; total];
        file.read_exact(&mut data).map_err(err)?;
        let digest = {
            Sha256::new()
                .chain_update(&header_bytes)
                .chain_update(&data)
                .finalize()
        };
        if digest[..] != prefix[12..44] {
            return Err(err("Packed chunk checksum mismatch"));
        }
        // Verified hit: stamp recency so idle-expiry and LRU see live entries.
        drop(file);
        stamp_mtime(&self.path(key));
        stamp_mtime(&self.root.join("budget"));
        Ok(Some(VerifiedChunk {
            header,
            data,
            file_len: len,
        }))
    }

    pub fn write(&self, key: &str, weight: &Weight) -> Result<()> {
        self.write_with_total_cap(key, weight, packed_total_cap())
    }

    /// `total_cap` is injectable so tests can exercise the root-wide cap
    /// without mutating the process environment.
    fn write_with_total_cap(
        &self,
        key: &str,
        weight: &Weight,
        total_cap: Option<u64>,
    ) -> Result<()> {
        if weight.bytes()? > MAX_READ_BYTES || weight.scales.is_none() {
            return Ok(());
        }
        fs::create_dir_all(&self.root).map_err(err)?;
        // A sweep may have deleted the .manifest (or the whole dir) between
        // new_at and this write; republish so the recreated dir stays
        // verifiable instead of becoming an unaccountable manifest-less dir.
        if let Some(manifest) = &self.manifest
            && !self.root.join(".manifest").is_file()
        {
            let _ = fs::write(self.root.join(".manifest"), manifest);
        }
        // Serialize outside the lock so the ledger can reserve exact bytes.
        let mut header = Header {
            key: key.into(),
            group: weight.group,
            bits: weight.bits,
            mode: weight.mode.clone(),
            arrays: Vec::new(),
        };
        let mut data = Vec::with_capacity(weight.bytes()? as usize);
        for a in [
            Some(&weight.values),
            weight.scales.as_ref(),
            weight.biases.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            let (dtype, bytes) = encode_array(a)?;
            header.arrays.push(ArrayInfo {
                dtype: dtype.into(),
                shape: a.shape()?.to_vec(),
                bytes: bytes.len(),
            });
            data.extend_from_slice(&bytes);
        }
        let header = serde_json::to_vec(&header).map_err(err)?;
        if header.len() > MAX_HEADER || data.len() as u64 > MAX_READ_BYTES {
            return Err(err("Packed chunk exceeds staging budget"));
        }
        let reserve = 44 + header.len() as u64 + data.len() as u64;
        // Serialize writers across processes. A busy writer never stalls inference.
        let _in_process = writer_mutex()
            .lock()
            .map_err(|_| err("Packed writer mutex poisoned"))?;
        let mut ledger = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.root.join("budget"))
            .map_err(err)?;
        if unsafe { libc::flock(ledger.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Ok(());
        }
        // Closing ledger releases the lock on every return/error path.
        let mut n = [0u8; 8];
        let recorded = if ledger.metadata().map_err(err)?.len() == 8 {
            ledger.read_exact(&mut n).map_err(err)?;
            Some(u64::from_le_bytes(n))
        } else {
            None
        };
        // Reconcile the ledger to actual .pack bytes: externally deleted files
        // must not permanently inflate the recorded spend. `used` excludes the
        // target's own bytes — rename replaces it atomically, so its current
        // size is not new spend and must not veto repairing a corrupt or
        // oversized entry. `real_used` includes them: the ledger can be
        // persisted on any of the early returns below while the old target
        // stays on disk, so the recorded value must never undercount reality.
        let target = self.path(key);
        let target_len = fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
        let mut used: u64 = fs::read_dir(&self.root)
            .map_err(err)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                let path = e.path();
                path.extension().is_some_and(|s| s == "pack") && path != target
            })
            .try_fold(0u64, |sum, e| {
                sum.checked_add(e.metadata().map_err(err)?.len())
                    .ok_or_else(|| err("Packed cache budget overflow"))
            })?;
        let real_used = used.saturating_add(target_len);
        // A configured total cap bounds the whole cache root but never loosens
        // the per-dir DISK_BUDGET invariant.
        let budget = total_cap.map_or(DISK_BUDGET, |cap| cap.min(DISK_BUDGET));
        let record = |ledger: &mut File, value: u64| -> Result<()> {
            ledger.seek(SeekFrom::Start(0)).map_err(err)?;
            ledger.write_all(&value.to_le_bytes()).map_err(err)?;
            ledger.set_len(8).map_err(err)
        };
        if recorded != Some(real_used) {
            record(&mut ledger, real_used)?;
        }
        if !has_disk_space(&self.root, reserve) {
            return Ok(());
        }
        // A chunk larger than the configured cap can never be admitted.
        if total_cap.is_some_and(|cap| reserve > cap) {
            return Ok(());
        }
        // Bound the configured cap against ROOT-WIDE spend, not just this
        // dir's: sibling caches share the same cap. The whole estimate →
        // evict → reconcile → shed → reserve critical section runs under a
        // root-wide flock; without it two writers on different identity dirs
        // could both admit writes that jointly overspend. A busy lock skips
        // this chunk: cache writes never stall.
        // Lock order stays acyclic: own budget lock (held above) → root cap
        // lock → siblings' budget locks, non-blocking, inside enforce.
        let _root_guard = if let Some((cap, parent)) = total_cap.zip(self.root.parent()) {
            let guard = packed_root_cap_lock(parent);
            if guard.is_none() {
                return Ok(());
            }
            // The fast path estimates siblings from their ledgers (a full
            // byte scan per tensor write would be too hot); only a plausible
            // overspend pays the real enforcement sweep.
            let estimate = root_pack_spend(parent, &self.root, used, sibling_pack_estimate);
            if estimate.saturating_add(reserve) > cap || used.saturating_add(reserve) > budget {
                // Evict LRU siblings down to `cap - reserve` — the target is
                // the room this write needs, so admission can roll instead of
                // wedging once aggregate usage reaches the cap.
                enforce_packed_total_cap(parent, &self.root, cap - reserve);
                // Authoritative recheck: reconcile each unheld sibling ledger
                // to real pack bytes (a crashed reservation is phantom spend
                // that would otherwise veto forever); a held budget lock
                // means a live writer whose pending reservation is trusted.
                let reconciled =
                    root_pack_spend(parent, &self.root, used, reconciled_sibling_pack_spend);
                let siblings = reconciled.saturating_sub(used);
                // Shed our own oldest packs when self-shedding can admit the
                // write: enforcement could not (it never wins the budget
                // flock we hold). `headroom` bounds both the per-dir budget
                // and the global cap net of sibling spend.
                if used.saturating_add(reserve) > budget || reconciled.saturating_add(reserve) > cap
                {
                    let headroom = budget
                        .min(cap.saturating_sub(siblings))
                        .saturating_sub(reserve);
                    used = shed_oldest_packs(&self.root, &target, headroom);
                }
                let actual = siblings.saturating_add(used);
                if actual.saturating_add(reserve) > cap || used.saturating_add(reserve) > budget {
                    return Ok(());
                }
            }
            guard
        } else {
            // No configured cap: the per-dir budget is the only bound.
            if used.saturating_add(reserve) > budget {
                return Ok(());
            }
            None
        };
        // Reserve first: an interrupted write can overcount, never overspend.
        // With a cap configured the reservation stays under the root lock, so
        // admission and reservation are atomic across processes. The record is
        // max(reservation, real bytes): the rename swaps the target's bytes
        // atomically, so if this write later fails the old pack persists and
        // the ledger must never fall below real spend.
        record(
            &mut ledger,
            (used + reserve).max(used.saturating_add(target_len)),
        )?;
        drop(_root_guard);
        let digest = Sha256::new()
            .chain_update(&header)
            .chain_update(&data)
            .finalize();
        let temp = self.root.join(format!("{}.tmp", uuid::Uuid::new_v4()));
        let result = (|| {
            let mut f = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temp)
                .map_err(err)?;
            f.write_all(MAGIC).map_err(err)?;
            f.write_all(&(header.len() as u32).to_le_bytes())
                .map_err(err)?;
            f.write_all(&digest).map_err(err)?;
            f.write_all(&header).map_err(err)?;
            f.write_all(&data).map_err(err)?;
            drop(f);
            fs::rename(&temp, self.path(key)).map_err(err)
        })();
        let _ = fs::remove_file(temp);
        result
    }
}

impl VerifiedChunk {
    pub(super) fn into_weight(self) -> Result<(Weight, u64)> {
        let Self {
            header,
            data,
            file_len,
        } = self;
        if !(2..=3).contains(&header.arrays.len()) {
            return Err(err(
                "Packed chunk requires values, scales and optional biases",
            ));
        }
        let mut arrays = Vec::with_capacity(3);
        let mut remaining = data.as_slice();
        for info in &header.arrays {
            let (bytes, tail) = remaining
                .split_at_checked(info.bytes)
                .ok_or_else(|| err("Truncated packed chunk array payload"))?;
            arrays.push(decode_array(info, bytes)?);
            remaining = tail;
        }
        if !remaining.is_empty() {
            return Err(err("Packed chunk has trailing array payload"));
        }
        let biases = if arrays.len() == 3 {
            arrays.pop()
        } else {
            None
        };
        let scales = arrays.pop();
        let values = arrays
            .pop()
            .ok_or_else(|| err("Packed chunk is missing its values array"))?;
        Ok((
            Weight {
                dense_bf16: None,
                values,
                scales,
                biases,
                group: header.group,
                bits: header.bits,
                mode: header.mode,
            },
            file_len,
        ))
    }
}

fn has_disk_space(path: &Path, needed: u64) -> bool {
    let Ok(path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    if unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return false;
    }
    let stat = unsafe { stat.assume_init() };
    (stat.f_bavail as u64).saturating_mul(stat.f_frsize) >= FREE_RESERVE + needed
}

fn encode_array(a: &MxArray) -> Result<(&'static str, Vec<u8>)> {
    Ok(match a.dtype()? {
        DType::Uint32 => (
            "U32",
            a.to_uint32()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        DType::Float32 => (
            "F32",
            a.to_float32()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        DType::Float16 | DType::BFloat16 => (
            if a.dtype()? == DType::Float16 {
                "F16"
            } else {
                "BF16"
            },
            a.to_uint16_native()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        DType::Uint8 => ("U8", a.to_uint8()?),
        DType::Int8 => ("I8", a.to_int8()?.into_iter().map(|v| v as u8).collect()),
        _ => return Err(err("Unsupported packed cache dtype")),
    })
}
fn decode_array(a: &ArrayInfo, bytes: &[u8]) -> Result<MxArray> {
    match a.dtype.as_str() {
        "U32" => MxArray::from_uint32(
            &bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect::<Vec<_>>(),
            &a.shape,
        ),
        "F32" => MxArray::from_float32(
            &bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>(),
            &a.shape,
        ),
        "F16" | "BF16" => {
            let data = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b))
                .collect::<Vec<_>>();
            if a.dtype == "F16" {
                MxArray::from_float16(&data, &a.shape)
            } else {
                MxArray::from_bfloat16(&data, &a.shape)
            }
        }
        "U8" => MxArray::from_uint8(bytes, &a.shape),
        "I8" => MxArray::from_int8(
            &bytes.iter().map(|&b| b as i8).collect::<Vec<_>>(),
            &a.shape,
        ),
        _ => Err(err("Unsupported packed cache dtype")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_verified_chunks_are_rejected_before_array_creation() {
        for (count, data) in [(0, Vec::new()), (1, Vec::new()), (2, vec![0; 3])] {
            let chunk = VerifiedChunk {
                header: Header {
                    key: "malformed".into(),
                    group: 32,
                    bits: 4,
                    mode: "affine".into(),
                    arrays: (0..count)
                        .map(|_| ArrayInfo {
                            dtype: "U32".into(),
                            shape: vec![1],
                            bytes: 4,
                        })
                        .collect(),
                },
                data,
                file_len: 0,
            };
            assert!(chunk.into_weight().is_err());
        }
    }

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("qwen4-packed-{}", uuid::Uuid::new_v4()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn weight(mode: &str) -> Weight {
        let (bits, group, width, scales, bias) = match mode {
            "q6k" => (
                6,
                16,
                48,
                MxArray::from_int8(&[-7; 32], &[2, 16]).unwrap(),
                MxArray::from_float16(&[half::f16::from_f32(0.01).to_bits(); 2], &[2, 1]).unwrap(),
            ),
            "q4k" => (
                4,
                32,
                32,
                MxArray::from_uint8(&[3; 32], &[2, 16]).unwrap(),
                MxArray::from_float16(&[half::f16::from_f32(0.03).to_bits(); 4], &[2, 2]).unwrap(),
            ),
            _ => (
                5,
                32,
                40,
                MxArray::from_float16(&[half::f16::from_f32(0.25).to_bits(); 16], &[2, 8]).unwrap(),
                MxArray::from_float16(&[half::f16::from_f32(-2.0).to_bits(); 16], &[2, 8]).unwrap(),
            ),
        };
        Weight {
            dense_bf16: None,
            values: MxArray::from_uint32(&vec![0x11223344; 2 * width], &[2, width as i64]).unwrap(),
            scales: Some(scales),
            biases: Some(bias),
            bits,
            group,
            mode: mode.into(),
        }
    }

    fn age(path: &Path, seconds: u64) {
        let old = SystemTime::now() - Duration::from_secs(seconds);
        let file = OpenOptions::new().write(true).open(path).unwrap();
        file.set_modified(old).unwrap();
    }

    /// `age` needs a writable fd a directory cannot provide; futimens on a
    /// read-only dir fd reaches the same witness.
    fn age_dir(path: &Path, seconds: u64) {
        let old = SystemTime::now() - Duration::from_secs(seconds);
        let file = File::open(path).unwrap();
        file.set_modified(old).unwrap();
    }

    #[test]
    fn packed_chunks_preserve_codes_signed_scales_and_quantized_outputs() {
        let temp = Temp::new();
        let cache = PackedCache {
            root: temp.0.join("cache"),
            manifest: None,
        };
        let x = MxArray::from_float32(
            &(0..256)
                .map(|i| (i as f32 * 0.07).sin())
                .collect::<Vec<_>>(),
            &[1, 256],
        )
        .unwrap();
        for mode in ["affine", "q4k", "q6k"] {
            let w = weight(mode);
            cache.write(mode, &w).unwrap();
            let (loaded, _) = cache.read(mode).unwrap().unwrap();
            for (a, b) in [
                (&w.values, &loaded.values),
                (w.scales.as_ref().unwrap(), loaded.scales.as_ref().unwrap()),
                (w.biases.as_ref().unwrap(), loaded.biases.as_ref().unwrap()),
            ] {
                assert_eq!(encode_array(a).unwrap(), encode_array(b).unwrap());
                assert_eq!(&*a.shape().unwrap(), &*b.shape().unwrap());
            }
            assert_eq!(
                w.linear(&x).unwrap().to_float32().unwrap().as_ref(),
                loaded.linear(&x).unwrap().to_float32().unwrap().as_ref()
            );
            let original = std::sync::Arc::new(w.clone());
            let originals = vec![original; 3];
            let stacked = Weight::stack(&originals).unwrap().unwrap();
            let actual = stacked
                .linear(
                    &x.reshape(&[1, 1, 256])
                        .unwrap()
                        .broadcast_to(&[3, 1, 256])
                        .unwrap(),
                )
                .unwrap()
                .to_float32()
                .unwrap();
            let expected = w.linear(&x).unwrap().to_float32().unwrap();
            for row in actual.chunks(2) {
                assert!(
                    row.iter()
                        .zip(expected.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-4),
                    "{mode}: batched expert arithmetic changed"
                );
            }
        }
    }

    #[test]
    fn resident_gather_matches_selected_quantized_experts_for_prompt_and_decode() {
        use std::sync::Arc;
        for mode in ["affine", "q4k", "q6k"] {
            let experts: Vec<_> = (0..3)
                .map(|expert| {
                    let mut w = weight(mode);
                    let shape = w.values.shape().unwrap();
                    let codes = w
                        .values
                        .to_uint32()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(i, v)| v.wrapping_add((expert * 7919 + i * 101) as u32))
                        .collect::<Vec<_>>();
                    w.values = MxArray::from_uint32(&codes, &shape).unwrap();
                    Arc::new(w)
                })
                .collect();
            let bank = Weight::concatenate_rows(&experts).unwrap();
            let ids = [2u32, 0, 1, 1, 0, 2];
            let indices = MxArray::from_uint32(&ids, &[3, 2]).unwrap();
            for dtype in [DType::Float32, DType::BFloat16] {
                for fanout in [1, 2] {
                    let x = MxArray::from_float32(
                        &(0..3 * fanout * 256)
                            .map(|i| (i as f32 * 0.037).sin() * 0.1)
                            .collect::<Vec<_>>(),
                        &[3, fanout as i64, 1, 256],
                    )
                    .unwrap()
                    .astype(dtype)
                    .unwrap();
                    let got = bank
                        .gather_rows(&x, &indices, 3, false)
                        .unwrap()
                        .to_float32()
                        .unwrap();
                    for (j, &expert) in ids.iter().enumerate() {
                        let row = x
                            .reshape(&[(3 * fanout) as i64, 256])
                            .unwrap()
                            .slice_axis(
                                0,
                                ((j / 2) * fanout + if fanout == 1 { 0 } else { j % 2 }) as i64,
                                ((j / 2) * fanout + if fanout == 1 { 0 } else { j % 2 } + 1) as i64,
                            )
                            .unwrap();
                        let want = experts[expert as usize]
                            .linear(&row)
                            .unwrap()
                            .to_float32()
                            .unwrap();
                        for k in 0..2 {
                            let tolerance = if dtype == DType::Float32 { 1e-4 } else { 0.04 };
                            assert!(
                                (got[j * 2 + k] - want[k]).abs() < tolerance,
                                "mode={mode}, dtype={dtype:?}, fanout={fanout}, row={j}, got={}, want={}",
                                got[j * 2 + k],
                                want[k]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn corrupt_or_oversized_cache_is_rejected_before_array_creation_and_can_be_repaired() {
        let temp = Temp::new();
        let cache = PackedCache {
            root: temp.0.join("cache"),
            manifest: None,
        };
        let w = weight("affine");
        cache.write("key", &w).unwrap();
        let path = cache.path("key");
        let original = fs::read(&path).unwrap();
        for bad in [
            original[..20].to_vec(),
            {
                let mut b = original.clone();
                *b.last_mut().unwrap() ^= 1;
                b
            },
            {
                let mut b = original.clone();
                b[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
                b
            },
        ] {
            fs::write(&path, bad).unwrap();
            assert!(cache.read("key").is_err());
        }
        let f = File::create(&path).unwrap();
        f.set_len(1 << 40).unwrap();
        assert!(cache.read("key").is_err());
        drop(f);
        cache.write("key", &w).unwrap();
        assert!(cache.read("key").unwrap().is_some());
    }

    #[test]
    fn source_change_invalidates_prepared_chunks_without_reading_payloads() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"first").unwrap();
        let a = PackedCache::new_at(&parent, BTreeSet::from([source.clone()])).unwrap();
        assert!(a.root.join(".manifest").is_file());
        fs::write(&source, b"replacement").unwrap();
        let b = PackedCache::new_at(&parent, BTreeSet::from([source.clone()])).unwrap();
        assert_ne!(a.root, b.root);
    }

    #[test]
    fn manifest_records_source_dir_and_recomputes_identity() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source.clone()])).unwrap();
        let manifest = fs::read_to_string(cache.root.join(".manifest")).unwrap();
        let canonical = source.canonicalize().unwrap();
        assert_eq!(
            manifest.lines().next().unwrap(),
            format!("path={}", canonical.parent().unwrap().display())
        );
        assert_eq!(manifest.lines().nth(1).unwrap(), "file=source.gguf");
        assert_eq!(
            identity_from_manifest(&cache.root).unwrap(),
            dir_name_identity(&cache.root).unwrap()
        );
        // Deleting the source makes the identity unverifiable; the orphan
        // verdict still waits out the idle grace on the newest witness.
        fs::remove_file(&source).unwrap();
        assert!(identity_from_manifest(&cache.root).is_err());
        assert_eq!(packed_orphan_reason(&cache.root), None);
        age(&cache.root.join(".manifest"), 7200);
        age_dir(&cache.root, 7200);
        assert_eq!(
            packed_orphan_reason(&cache.root),
            Some("unreadable manifest or missing source")
        );
    }

    #[test]
    fn stale_sibling_identity_is_swept_current_is_preserved() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        fs::create_dir(&parent).unwrap();
        // A stale sibling: manifest present but its source is gone, aged past
        // the idle grace on every witness.
        let stale = parent.join(format!("{DIR_PREFIX}{}", "a".repeat(64)));
        fs::create_dir(&stale).unwrap();
        fs::write(stale.join(".manifest"), "path=/gone\nfile=x.gguf\n").unwrap();
        age(&stale.join(".manifest"), 7200);
        age_dir(&stale, 7200);
        // A manifest-less sibling is never orphan-deleted.
        let unknown = parent.join(format!("{DIR_PREFIX}{}", "b".repeat(64)));
        fs::create_dir(&unknown).unwrap();
        fs::write(unknown.join("leftover.pack"), b"data").unwrap();
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();
        assert!(!stale.exists());
        assert!(unknown.exists());
        assert!(cache.root.join(".manifest").is_file());
    }

    // flock is per-process on BSD/macOS, so a held writer lock can only be
    // observed from a second process.
    #[test]
    #[ignore = "helper process for held_budget_lock_skips_sweep_deletion"]
    fn packed_budget_lock_child() {
        let path = PathBuf::from(
            std::env::var_os("MLX_NODE_PACKED_LOCK_TEST_PATH").expect("lock-test child path"),
        );
        let acquired = PathBuf::from(
            std::env::var_os("MLX_NODE_PACKED_LOCK_TEST_ACQUIRED")
                .expect("lock-test child acquired path"),
        );
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        assert_eq!(unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) }, 0);
        fs::write(acquired, b"acquired").unwrap();
        // The parent kills this process once the sweep assertion completes.
        std::thread::sleep(Duration::from_secs(60));
    }

    #[test]
    fn held_budget_lock_skips_sweep_deletion() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        fs::create_dir(&parent).unwrap();
        let stale = parent.join(format!("{DIR_PREFIX}{}", "a".repeat(64)));
        fs::create_dir(&stale).unwrap();
        fs::write(stale.join(".manifest"), "path=/gone\nfile=x.gguf\n").unwrap();
        fs::write(stale.join("budget"), 0u64.to_le_bytes()).unwrap();
        age(&stale.join(".manifest"), 7200);
        age(&stale.join("budget"), 7200);
        age_dir(&stale, 7200);
        let acquired = temp.0.join("child.acquired");
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("models::qwen4_exp::packed_cache::tests::packed_budget_lock_child")
            .arg("--ignored")
            .env("MLX_NODE_PACKED_LOCK_TEST_PATH", stale.join("budget"))
            .env("MLX_NODE_PACKED_LOCK_TEST_ACQUIRED", &acquired)
            .spawn()
            .unwrap();
        for _ in 0..500 {
            if acquired.is_file() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(acquired.is_file(), "child never acquired the budget lock");
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let _cache = PackedCache::new_at(&parent, BTreeSet::from([source.clone()])).unwrap();
        // A live writer's held flock makes the sweep keep even an orphan dir.
        assert!(stale.exists());
        child.kill().unwrap();
        let _ = child.wait();
        let _cache = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();
        assert!(!stale.exists());
    }

    #[test]
    fn ledger_reconciles_externally_deleted_packs() {
        let temp = Temp::new();
        let cache = PackedCache {
            root: temp.0.join("cache"),
            manifest: None,
        };
        let w = weight("affine");
        cache.write("k1", &w).unwrap();
        cache.write("k2", &w).unwrap();
        // Simulated external cleanup inflates the recorded spend.
        fs::remove_file(cache.path("k1")).unwrap();
        cache.write("k3", &w).unwrap();
        let mut n = [0u8; 8];
        File::open(cache.root.join("budget"))
            .unwrap()
            .read_exact(&mut n)
            .unwrap();
        let actual: u64 = fs::read_dir(&cache.root)
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|s| s == "pack"))
            .map(|e| e.metadata().unwrap().len())
            .sum();
        // Recorded spend = reconciled used + exact reserve for the new pack,
        // which now equals the true on-disk bytes.
        assert_eq!(u64::from_le_bytes(n), actual);
        cache.write("k4", &w).unwrap();
        File::open(cache.root.join("budget"))
            .unwrap()
            .read_exact(&mut n)
            .unwrap();
        assert_eq!(
            u64::from_le_bytes(n),
            fs::metadata(cache.path("k4")).unwrap().len()
                + fs::metadata(cache.path("k2")).unwrap().len()
                + fs::metadata(cache.path("k3")).unwrap().len()
        );
    }

    #[test]
    fn total_cap_sheds_oldest_packs_in_current_dir() {
        let temp = Temp::new();
        let cache = PackedCache {
            root: temp.0.join("cache"),
            manifest: None,
        };
        let w = weight("affine");
        for (key, seconds) in [("old", 300u64), ("mid", 200), ("new", 100)] {
            cache.write(key, &w).unwrap();
            age(&cache.path(key), seconds);
        }
        let pack = fs::metadata(cache.path("old")).unwrap().len();
        // Cap fits exactly two packs plus ledger bytes: shed only the oldest.
        enforce_packed_total_cap(&temp.0, &cache.root, 2 * pack + 64);
        assert!(!cache.path("old").exists());
        assert!(cache.path("mid").exists());
        assert!(cache.path("new").exists());
        // Ledger re-reconciled to actual bytes.
        let mut n = [0u8; 8];
        File::open(cache.root.join("budget"))
            .unwrap()
            .read_exact(&mut n)
            .unwrap();
        assert_eq!(u64::from_le_bytes(n), 2 * pack);
    }

    #[test]
    fn write_republishes_manifest_and_sweep_drops_stale_tmps() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();
        // A sweep-deleted manifest comes back on the next write.
        fs::remove_file(cache.root.join(".manifest")).unwrap();
        cache.write("key", &weight("affine")).unwrap();
        assert!(cache.root.join(".manifest").is_file());
        // Crash leftovers in the current dir leave through the sweep once
        // aged; a fresh .tmp (a plausibly live writer) is kept.
        let stale_tmp = cache.root.join("old.tmp");
        let live_tmp = cache.root.join("live.tmp");
        fs::write(&stale_tmp, b"x").unwrap();
        fs::write(&live_tmp, b"x").unwrap();
        age(&stale_tmp, 7200);
        sweep_packed_cache(&parent, &cache.root);
        assert!(!stale_tmp.exists());
        assert!(live_tmp.exists());
    }

    #[test]
    fn total_cap_evicts_only_idle_manifested_siblings() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        fs::create_dir(&parent).unwrap();
        // Eldest with a manifest and past the idle grace: the only eviction.
        let oldest = parent.join(format!("{DIR_PREFIX}{}", "a".repeat(64)));
        fs::create_dir(&oldest).unwrap();
        fs::write(oldest.join("a.pack"), vec![0_u8; 100]).unwrap();
        fs::write(oldest.join(".manifest"), "path=/gone\nfile=x.gguf\n").unwrap();
        age(&oldest.join("a.pack"), 7200);
        age(&oldest.join(".manifest"), 7200);
        age_dir(&oldest, 7200);
        // Fresh manifest dir and manifest-less dir both survive the cap: the
        // idle grace shields the first, the manifest requirement the second.
        let fresh = parent.join(format!("{DIR_PREFIX}{}", "b".repeat(64)));
        fs::create_dir(&fresh).unwrap();
        fs::write(fresh.join("b.pack"), vec![0_u8; 100]).unwrap();
        fs::write(fresh.join(".manifest"), "path=/gone\nfile=x.gguf\n").unwrap();
        let bare = parent.join(format!("{DIR_PREFIX}{}", "c".repeat(64)));
        fs::create_dir(&bare).unwrap();
        fs::write(bare.join("c.pack"), vec![0_u8; 100]).unwrap();
        age(&bare.join("c.pack"), 7200);
        age_dir(&bare, 7200);
        let cache = PackedCache {
            root: temp.0.join("current"),
            manifest: None,
        };
        fs::create_dir(&cache.root).unwrap();
        enforce_packed_total_cap(&parent, &cache.root, 150);
        assert!(!oldest.exists());
        assert!(fresh.exists());
        assert!(bare.exists());
    }

    #[test]
    fn verified_read_stamps_recency() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();
        cache.write("key", &weight("affine")).unwrap();
        age(&cache.path("key"), 7200);
        age(&cache.root.join("budget"), 7200);
        age(&cache.root.join(".manifest"), 7200);
        let before = fs::metadata(cache.path("key")).unwrap().modified().unwrap();
        assert!(before < SystemTime::now() - PACKED_SWEEP_MIN_AGE);
        assert!(cache.read("key").unwrap().is_some());
        let after = fs::metadata(cache.path("key")).unwrap().modified().unwrap();
        assert!(after > before);
        // Identity verifies and the stamped pack keeps the dir's newest mtime
        // inside the grace window (the dir mtime itself is also fresh).
        assert_eq!(packed_orphan_reason(&cache.root), None);
    }

    #[test]
    fn parallel_reads_preserve_order_ownership_and_revalidate_changed_files() {
        let temp = Temp::new();
        let cache = PackedCache {
            root: temp.0.join("cache"),
            manifest: None,
        };
        for mode in ["affine", "q4k", "q6k"] {
            cache.write(mode, &weight(mode)).unwrap();
        }
        let requests: Vec<_> = ["q6k", "affine", "missing", "q4k", "q6k"]
            .into_iter()
            .map(|key| (key.to_string(), 4096))
            .collect();
        let chunks = cache.read_batch(&requests).unwrap();
        // Returned bytes own the verified generation even if the path changes
        // before main-thread import. No mmap or stale verification is reused.
        fs::write(cache.path("q6k"), b"truncated").unwrap();
        let mut corrupt = fs::read(cache.path("affine")).unwrap();
        *corrupt.last_mut().unwrap() ^= 1;
        fs::write(cache.path("affine"), corrupt).unwrap();
        for ((key, _), chunk) in requests.iter().zip(chunks) {
            if key == "missing" {
                assert!(chunk.is_none());
                continue;
            }
            let (got, _) = chunk.unwrap().into_weight().unwrap();
            let want = weight(key);
            assert_eq!(
                encode_array(&got.values).unwrap(),
                encode_array(&want.values).unwrap()
            );
            assert_eq!(
                encode_array(got.scales.as_ref().unwrap()).unwrap(),
                encode_array(want.scales.as_ref().unwrap()).unwrap()
            );
        }
        let changed = cache.read_batch(&requests).unwrap();
        assert!(
            changed[0].is_none()
                && changed[1].is_none()
                && changed[2].is_none()
                && changed[4].is_none()
        );
        assert!(changed[3].is_some());
        assert!(cache.read_batch(&[("q4k".into(), 1)]).unwrap()[0].is_none());
        assert!(
            cache
                .read_batch(&[("q4k".into(), MAX_READ_BYTES + 1)])
                .is_err()
        );
        assert!(cache.read_batch(&vec![("q4k".into(), 1); 25]).is_err());
    }

    #[test]
    fn total_cap_admission_counts_sibling_reservations_and_busy_lock_skips() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        let source_a = temp.0.join("a.gguf");
        let source_b = temp.0.join("b.gguf");
        fs::write(&source_a, b"first").unwrap();
        fs::write(&source_b, b"second").unwrap();
        let a = PackedCache::new_at(&parent, BTreeSet::from([source_a])).unwrap();
        let b = PackedCache::new_at(&parent, BTreeSet::from([source_b])).unwrap();
        assert_ne!(a.root, b.root);
        let w = weight("affine");
        // Under the cap the write goes through and the ledger reserves it.
        a.write_with_total_cap("k1", &w, Some(u64::MAX)).unwrap();
        let pack = fs::metadata(a.path("k1")).unwrap().len();
        assert!(pack > 0);
        // A sibling ledger reservation counts toward the shared cap: b's
        // write is vetoed even though its own dir is empty, and no eviction
        // can free it (a's dir is fresh).
        b.write_with_total_cap("k1", &w, Some(pack)).unwrap();
        assert!(!b.path("k1").exists());
        // A busy root cap lock skips the chunk instead of stalling; the next
        // write after the guard drops succeeds.
        let guard = packed_root_cap_lock(&parent).unwrap();
        a.write_with_total_cap("k2", &w, Some(u64::MAX)).unwrap();
        assert!(!a.path("k2").exists());
        drop(guard);
        a.write_with_total_cap("k2", &w, Some(u64::MAX)).unwrap();
        assert!(a.path("k2").exists());
        // The cap lock file itself is never treated as a cache dir.
        assert!(parent.join(".mlx-qwen4-packed-cap.lock").is_file());
        assert_eq!(
            root_pack_spend(&parent, &b.root, 0, sibling_pack_estimate),
            pack * 2,
            "the lock file must not be counted as spend"
        );
    }

    #[test]
    fn total_cap_sheds_own_oldest_packs_to_admit_new_write() {
        let temp = Temp::new();
        let cache = PackedCache {
            root: temp.0.join("cache"),
            manifest: None,
        };
        let w = weight("affine");
        for (key, seconds) in [("old", 300u64), ("mid", 200), ("new", 100)] {
            cache.write(key, &w).unwrap();
            age(&cache.path(key), seconds);
        }
        let pack = fs::metadata(cache.path("old")).unwrap().len();
        // A cap that fits exactly two packs plus the pending write: the
        // writer must shed its own oldest pack (enforcement cannot — it
        // never wins the budget flock the writer holds), turning the dir
        // into a true LRU instead of wedging once usage reaches the cap.
        // ("nxt" stays 3 chars so reserve equals the existing pack size.)
        cache
            .write_with_total_cap("nxt", &w, Some(2 * pack + pack))
            .unwrap();
        assert!(!cache.path("old").exists(), "oldest pack is shed for room");
        assert!(cache.path("mid").exists());
        assert!(cache.path("new").exists());
        assert!(cache.path("nxt").exists(), "the new write must be admitted");
        // Ledger records reconciled used + reserve = 3 packs.
        let mut n = [0u8; 8];
        File::open(cache.root.join("budget"))
            .unwrap()
            .read_exact(&mut n)
            .unwrap();
        assert_eq!(u64::from_le_bytes(n), 3 * pack);
    }

    #[test]
    fn crashed_sibling_reservation_is_reconciled_at_recheck() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        fs::create_dir(&parent).unwrap();
        // Sibling identity dir with a small real pack but a ledger claiming
        // u64::MAX: a crash between `record` and rename leaves phantom spend
        // that must not veto other identities forever.
        let sibling = parent.join(format!("{DIR_PREFIX}{}", "a".repeat(64)));
        fs::create_dir(&sibling).unwrap();
        fs::write(sibling.join("a.pack"), vec![0_u8; 64]).unwrap();
        let ledger_path = sibling.join("budget");
        fs::write(&ledger_path, u64::MAX.to_le_bytes()).unwrap();
        let source = temp.0.join("b.gguf");
        fs::write(&source, b"second").unwrap();
        let b = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();
        let w = weight("affine");
        // The sibling is fresh and manifest-less: nothing can evict it, so
        // admission is possible only if the recheck reconciles the phantom
        // ledger to its real pack bytes.
        b.write_with_total_cap("k1", &w, Some(u64::MAX)).unwrap();
        let pack = fs::metadata(b.path("k1")).unwrap().len();
        fs::write(&ledger_path, u64::MAX.to_le_bytes()).unwrap();
        b.write_with_total_cap("k2", &w, Some(pack * 10)).unwrap();
        assert!(
            b.path("k2").exists(),
            "a reconciled phantom reservation must not veto the write"
        );
        let mut n = [0u8; 8];
        File::open(&ledger_path)
            .unwrap()
            .read_exact(&mut n)
            .unwrap();
        assert_eq!(
            u64::from_le_bytes(n),
            64,
            "the unheld sibling ledger is rewritten to real pack bytes"
        );
        // While the sibling's budget lock is held the ledger is a live
        // writer's pending reservation and stays trusted: the write vetoes.
        fs::write(&ledger_path, u64::MAX.to_le_bytes()).unwrap();
        let held = try_budget_lock(&sibling).unwrap().unwrap();
        b.write_with_total_cap("k3", &w, Some(pack * 10)).unwrap();
        assert!(!b.path("k3").exists());
        drop(held);
    }

    #[test]
    fn skipped_replacement_write_keeps_ledger_at_or_above_real_bytes() {
        let temp = Temp::new();
        let cache = PackedCache {
            root: temp.0.join("cache"),
            manifest: None,
        };
        let w = weight("affine");
        cache.write("k1", &w).unwrap();
        let pack = fs::metadata(cache.path("k1")).unwrap().len();
        // A vetoed REPLACEMENT write leaves the old target on disk: the
        // ledger reconcile must record real spend (target included), not the
        // ex-target `used` — an undercounted ledger would let siblings
        // overspend the configured cap.
        cache.write_with_total_cap("k1", &w, Some(1)).unwrap();
        let mut n = [0u8; 8];
        File::open(cache.root.join("budget"))
            .unwrap()
            .read_exact(&mut n)
            .unwrap();
        assert!(
            u64::from_le_bytes(n) >= pack,
            "ledger {} must stay >= real pack bytes {}",
            u64::from_le_bytes(n),
            pack
        );
        assert!(cache.path("k1").exists());
        // And sibling estimates floor a stale-low ledger at real bytes.
        fs::write(cache.root.join("budget"), 0u64.to_le_bytes()).unwrap();
        assert_eq!(sibling_pack_estimate(&cache.root), pack);
    }
}
