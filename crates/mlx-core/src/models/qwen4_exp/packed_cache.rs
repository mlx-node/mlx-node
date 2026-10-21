//! The mapped expert pack's cache root: the source fingerprint that names
//! a checkpoint's pack dir and its `.manifest`, the stale-dir sweep, the
//! descriptor-relative file helpers the pack builder writes through, and
//! the per-expert array encoding the pack's sections hold.
use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use napi::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::debug;

use super::weights::Weight;
use crate::array::{DType, MxArray};

/// Seeds the source fingerprint that names the pack dir: a new value
/// renames every checkpoint's dir, so each load would rebuild its pack.
const MAGIC: &[u8; 8] = b"Q4PACK02";
const FREE_RESERVE: u64 = 16 << 30;
/// An orphan reason waits out this grace on the dir's newest mtime; the
/// manifest stamp of a serving pack keeps a live dir young.
const PACKED_SWEEP_MIN_AGE: Duration = Duration::from_secs(3600);
pub(super) const MAPPED_DIR_PREFIX: &str = ".mlx-qwen4-mapped-v3-";
/// A mapped dir's stable lock file: SH while a model reads or builds the
/// pack, EX by an evictor. Never renamed or unlinked (flock needs the inode).
const MAPPED_LOCK: &str = "lock";
/// Per-expert chunk dirs of the retired format, which older releases still
/// write (up to 96 GiB each). Nothing reads them; the sweep removes them.
const RETIRED_CHUNK_PREFIX: &str = ".mlx-qwen4-packed-v2-";
/// The retired format's ledger: every older build holds it EX while writing
/// a chunk; builds after v0.0.16 also hold it SH while reading one.
const RETIRED_CHUNK_LOCK: &str = "budget";

fn err(e: impl std::fmt::Display) -> Error {
    Error::from_reason(e.to_string())
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub(super) struct ArrayInfo {
    pub(super) dtype: String,
    pub(super) shape: Vec<i64>,
    pub(super) bytes: usize,
}
#[derive(Serialize, Deserialize)]
pub(super) struct Header {
    pub(super) key: String,
    pub(super) group: i32,
    pub(super) bits: i32,
    pub(super) mode: String,
    // values, scales, optional biases
    pub(super) arrays: Vec<ArrayInfo>,
}

/// Fully verified owned bytes; no file mapping or path-based trust survives a read.
pub(super) struct VerifiedChunk {
    pub(super) header: Header,
    pub(super) data: Vec<u8>,
    pub(super) file_len: u64,
}

/// One checkpoint's place under the cache root: its mapped pack dir, named
/// by the source fingerprint, and the `.manifest` the pack carries.
pub(super) struct PackedCache {
    dir: PathBuf,
    manifest: Vec<u8>,
}

/// Canonical path and metadata of each source file, in order.
fn stat_sources(files: impl Iterator<Item = PathBuf>) -> Vec<io::Result<(PathBuf, fs::Metadata)>> {
    files
        .map(|path| {
            let canonical = path.canonicalize()?;
            let meta = fs::metadata(&canonical)?;
            Ok((canonical, meta))
        })
        .collect()
}

/// No device number: macOS renumbers volumes across reboots and re-plugs.
fn packed_identity(files: &[(PathBuf, fs::Metadata)]) -> String {
    let mut hash = Sha256::new();
    hash.update(MAGIC);
    for (canonical, meta) in files {
        hash.update(canonical.as_os_str().as_bytes());
        for n in [
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
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Lowercase-hex encode raw bytes — the `.manifest` `path_hex=`/`file_hex=`
/// shape. The plain `path=`/`file=` lines carry raw OS bytes for readability,
/// but the manifest is line-delimited: a newline inside a filename splits one
/// record into bogus lines and non-UTF-8 bytes cannot round-trip through a
/// reader that inspects text. The hex twins are authoritative on parse; a
/// present-but-corrupt hex field is a corrupt manifest (Err), never license
/// to trust the lossy line.
fn packed_hex_encode(bytes: &[u8]) -> Vec<u8> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize]);
        out.push(HEX[(byte & 0x0f) as usize]);
    }
    out
}

/// Strict decode of `packed_hex_encode` output: even length, lowercase hex
/// only. `None` means malformed.
fn packed_hex_decode(field: &[u8]) -> Option<Vec<u8>> {
    if !field.len().is_multiple_of(2) {
        return None;
    }
    let nibble = |byte: u8| -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        }
    };
    field
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Some((nibble(pair[0])? << 4) | nibble(pair[1])?))
        .collect()
}

/// What a pack's `.manifest` proves about its source now.
#[derive(Debug, PartialEq, Eq)]
enum Source {
    /// Every file stats; the identity it hashes to.
    Present(String),
    /// A file is gone while the mount point the manifest recorded is mounted.
    Deleted,
    /// Its volume is not mounted, or a stat failed for another reason.
    Unreachable,
}

/// Err = unreadable or corrupt manifest. `path_hex=`/`file_hex=` lines (raw
/// OS bytes as hex) are preferred over the lossy `path=`/`file=` lines they
/// twin; pre-hex manifests keep working through the raw fields.
fn manifest_source(dir: &Path) -> Result<Source> {
    let data = fs::read(dir.join(".manifest")).map_err(err)?;
    let mut path_raw = None;
    let mut path_hex = None;
    let mut mount_hex = None;
    let mut file_raws: Vec<&[u8]> = Vec::new();
    let mut file_hexes: Vec<&[u8]> = Vec::new();
    for line in data.split(|&b| b == b'\n') {
        if let Some(raw) = line.strip_prefix(b"path_hex=".as_slice()) {
            path_hex = Some(raw);
        } else if let Some(raw) = line.strip_prefix(b"path=".as_slice()) {
            path_raw = Some(raw);
        } else if let Some(raw) = line.strip_prefix(b"mount_hex=".as_slice()) {
            mount_hex = Some(raw);
        } else if let Some(raw) = line.strip_prefix(b"file_hex=".as_slice()) {
            file_hexes.push(raw);
        } else if let Some(raw) = line.strip_prefix(b"file=".as_slice()) {
            file_raws.push(raw);
        }
    }
    let base = if let Some(hex) = path_hex {
        packed_hex_decode(hex)
            .map(|bytes| PathBuf::from(OsStr::from_bytes(&bytes)))
            .ok_or_else(|| err("packed manifest has corrupt path_hex="))?
    } else {
        path_raw
            .map(|raw| PathBuf::from(OsStr::from_bytes(raw)))
            .ok_or_else(|| err("packed manifest missing path="))?
    };
    let mount = mount_hex
        .map(|hex| {
            packed_hex_decode(hex)
                .map(|bytes| PathBuf::from(OsStr::from_bytes(&bytes)))
                .ok_or_else(|| err("packed manifest has corrupt mount_hex="))
        })
        .transpose()?;
    let files: Vec<PathBuf> = if !file_hexes.is_empty() {
        let mut files = Vec::with_capacity(file_hexes.len());
        for hex in file_hexes {
            let bytes = packed_hex_decode(hex)
                .ok_or_else(|| err("packed manifest has corrupt file_hex="))?;
            files.push(base.join(OsStr::from_bytes(&bytes)));
        }
        files
    } else {
        file_raws
            .iter()
            .map(|raw| base.join(OsStr::from_bytes(raw)))
            .collect()
    };
    if files.is_empty() {
        return Err(err("packed manifest lists no files"));
    }
    let mut stats = Vec::with_capacity(files.len());
    let mut missing = false;
    for stat in stat_sources(files.into_iter()) {
        match stat {
            Ok(stat) => stats.push(stat),
            Err(e) if e.kind() == ErrorKind::NotFound => missing = true,
            Err(_) => return Ok(Source::Unreachable),
        }
    }
    Ok(if !missing {
        Source::Present(packed_identity(&stats))
    } else if mount.is_some_and(|mount| is_mounted(&mount)) {
        Source::Deleted
    } else {
        Source::Unreachable
    })
}

/// Serialize canonical source paths into `.manifest` bytes: the hex records
/// come first (`path_hex=`, `mount_hex=`, then every `file_hex=`), the raw
/// `path=`/`file=` display lines last; the mount point has no raw twin. A
/// raw line whose bytes contain a newline is omitted
/// entirely — a newline is the only byte that can forge a record line, and a
/// path holding "\nfile_hex=evil" would inject an earlier authoritative-
/// looking line into the same file. The raw lines keep readable manifests
/// where names allow; the hex twins carry the same bytes losslessly so
/// newline-bearing or non-UTF-8 names cannot split or mangle a record.
fn pack_manifest(canonical: &[PathBuf]) -> Vec<u8> {
    let base = canonical
        .first()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_default();
    let mut manifest = b"path_hex=".to_vec();
    manifest.extend_from_slice(&packed_hex_encode(base.as_os_str().as_bytes()));
    manifest.push(b'\n');
    manifest.extend_from_slice(b"mount_hex=");
    manifest.extend_from_slice(&packed_hex_encode(
        mount_point(&base).as_os_str().as_bytes(),
    ));
    manifest.push(b'\n');
    let mut raw = Vec::new();
    if !base.as_os_str().as_bytes().contains(&b'\n') {
        raw.extend_from_slice(b"path=");
        raw.extend_from_slice(base.as_os_str().as_bytes());
        raw.push(b'\n');
    }
    for path in canonical {
        let rel = path.strip_prefix(&base).unwrap_or(path);
        manifest.extend_from_slice(b"file_hex=");
        manifest.extend_from_slice(&packed_hex_encode(rel.as_os_str().as_bytes()));
        manifest.push(b'\n');
        if !rel.as_os_str().as_bytes().contains(&b'\n') {
            raw.extend_from_slice(b"file=");
            raw.extend_from_slice(rel.as_os_str().as_bytes());
            raw.push(b'\n');
        }
    }
    manifest.extend_from_slice(&raw);
    manifest
}

/// A cache dir on disk is a REAL directory — never a symlink. `Path::is_dir`
/// follows links, so a `.mlx-qwen4-mapped-v3-*` symlink would admit its
/// external target into the sweep's eviction. The producers only ever emit
/// real dirs, so links are foreign.
pub(super) fn is_real_dir(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|meta| meta.is_dir())
}

/// The highest ancestor of `path` on its device: the mount point it sits on.
fn mount_point(path: &Path) -> PathBuf {
    let device = |p: &Path| fs::symlink_metadata(p).ok().map(|m| m.dev());
    let dev = device(path);
    let mut mount = path;
    while let Some(parent) = mount.parent() {
        if dev.is_none() || device(parent) != dev {
            break;
        }
        mount = parent;
    }
    mount.to_path_buf()
}

/// `mount` is a mount point now: `/`, or a real dir on another device than
/// its parent. A left-behind mount dir is a plain dir on its parent's device.
fn is_mounted(mount: &Path) -> bool {
    let Ok(meta) = fs::symlink_metadata(mount) else {
        return false;
    };
    meta.is_dir()
        && mount
            .parent()
            .is_none_or(|parent| fs::symlink_metadata(parent).is_ok_and(|p| p.dev() != meta.dev()))
}

/// Unlink `path` only when it is a symlink — `fs::remove_file` removes just
/// the link, never its target. Any other occupant or lookup failure is left
/// for the caller's `create_dir_all` to surface. Prevents a planted link
/// from redirecting dir creation and writes outside the cache root.
pub(super) fn unlink_if_symlink(path: &Path) {
    if fs::symlink_metadata(path).is_ok_and(|meta| meta.is_symlink()) {
        let _ = fs::remove_file(path);
    }
}

// ── Descriptor-relative I/O ────────────────────────────────────────────────
// A path check cannot pin what it validated: an actor able to mutate the
// cache parent could `rename` the real identity dir away and plant a
// symlink between the check and any `dir.join(name)` open, redirecting
// .manifest/.tmp/.mapped writes outside the cache root. Every write under
// a pack dir therefore runs relative to a directory fd opened once
// with O_DIRECTORY|O_NOFOLLOW: the fd pins THE validated inode, and a
// swapped path afterwards resolves nothing (writes then land in the — now
// unlinked — real dir and are reclaimed on close, never in the link's
// target). Names are bare filenames; the reject is a hard error, never an
// escape.

/// `name` must be a bare filename usable as an `openat`/`renameat`/
/// `unlinkat`/`fstatat` argument: no NUL, no path separator.
fn dir_c_name(name: &OsStr) -> io::Result<std::ffi::CString> {
    if name.as_bytes().contains(&b'/') {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "dir-relative name contains a separator",
        ));
    }
    std::ffi::CString::new(name.as_bytes())
        .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "dir-relative name contains NUL"))
}

/// Open `path` as a real directory without following a trailing symlink —
/// the descriptor-relative root for all writes under it. `None` on any
/// failure (symlink, missing, non-dir): callers treat that exactly like
/// the old not-a-real-dir check.
pub(super) fn open_dir_nofollow(path: &Path) -> Option<File> {
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: `c_path` outlives the call; the fd is wrapped in `File`,
    // which owns it.
    let fd = unsafe {
        libc::open(
            c_path.as_ptr(),
            libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_RDONLY | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return None;
    }
    // SAFETY: `fd` is a freshly opened descriptor this function owns.
    Some(unsafe { File::from_raw_fd(fd) })
}

pub(super) fn dir_open_at(
    dirfd: &File,
    name: &OsStr,
    flags: libc::c_int,
    mode: libc::c_int,
) -> io::Result<File> {
    let c_name = dir_c_name(name)?;
    // SAFETY: `dirfd` is an open directory descriptor and `c_name` is a
    // live NUL-terminated name; the returned fd is owned by `File`.
    let fd = unsafe { libc::openat(dirfd.as_raw_fd(), c_name.as_ptr(), flags, mode) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `fd` is a freshly opened descriptor this function owns.
    Ok(unsafe { File::from_raw_fd(fd) })
}

/// Create-or-truncate `name` inside `dirfd` and write `contents`.
/// O_NOFOLLOW keeps a planted link at `name` from redirecting the write.
pub(super) fn dir_write_file(dirfd: &File, name: &str, contents: &[u8]) -> io::Result<()> {
    let mut file = dir_open_at(
        dirfd,
        OsStr::new(name),
        libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        0o644,
    )?;
    file.write_all(contents)
}

/// Atomic `rename` inside the pinned dir — both endpoints relative to
/// `dirfd`, so a swapped `self.root` path cannot redirect either side.
pub(super) fn dir_renameat(dirfd: &File, from: &str, to: &str) -> io::Result<()> {
    let c_from = dir_c_name(OsStr::new(from))?;
    let c_to = dir_c_name(OsStr::new(to))?;
    // SAFETY: `dirfd` is an open directory descriptor; both names are live
    // NUL-terminated strings for the duration of the call.
    let result = unsafe {
        libc::renameat(
            dirfd.as_raw_fd(),
            c_from.as_ptr(),
            dirfd.as_raw_fd(),
            c_to.as_ptr(),
        )
    };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Unlink `name` inside the pinned dir. Errors are surfaced so callers can
/// tell a real removal from a miss.
pub(super) fn dir_unlinkat(dirfd: &File, name: &OsStr) -> io::Result<()> {
    let c_name = dir_c_name(name)?;
    // SAFETY: `dirfd` is an open directory descriptor and `c_name` is a
    // live NUL-terminated name; flag 0 removes files only, never dirs.
    let result = unsafe { libc::unlinkat(dirfd.as_raw_fd(), c_name.as_ptr(), 0) };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// `lstat` relative to the pinned dir: a symlink at `name` reports its own
/// metadata, never the target's — same contract as `fs::symlink_metadata`.
pub(super) fn dir_stat(dirfd: &File, name: &OsStr) -> Option<libc::stat> {
    let c_name = dir_c_name(name).ok()?;
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `dirfd` is an open directory descriptor, `c_name` is a live
    // NUL-terminated name, and `stat` is valid memory to fill.
    let result = unsafe {
        libc::fstatat(
            dirfd.as_raw_fd(),
            c_name.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if result != 0 {
        return None;
    }
    // SAFETY: fstatat returned success, so `stat` is fully initialized.
    Some(unsafe { stat.assume_init() })
}

/// Producer identities are exactly 64 lowercase hex chars (the source
/// fingerprint) under their prefix. Anything else under it — a `-backup`
/// suffix, a typo, a manual copy — is foreign: never swept.
fn prefixed_identity<'a>(dir: &'a Path, prefix: &str) -> Option<&'a str> {
    let suffix = dir.file_name()?.to_str()?.strip_prefix(prefix)?;
    (suffix.len() == 64
        && suffix
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()))
    .then_some(suffix)
}

pub(super) fn dir_name_identity(dir: &Path) -> Option<&str> {
    prefixed_identity(dir, MAPPED_DIR_PREFIX)
}

fn is_retired_chunk_dir(dir: &Path) -> bool {
    prefixed_identity(dir, RETIRED_CHUNK_PREFIX).is_some()
}

/// Newest mtime over the dir and every direct entry (.manifest, lock,
/// .mapped, in-flight .tmp); a present-but-unreadable mtime counts as fresh
/// (None) so the dir is never deleted on a bad clock.
fn packed_dir_mtime(dir: &Path) -> Option<SystemTime> {
    // symlink_metadata at both levels: a link's own mtime counts, its target
    // never does — a dangling link must not void recency and a hot external
    // target must not keep the dir artificially fresh.
    let mut newest = fs::symlink_metadata(dir).ok()?.modified().ok()?;
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        newest = newest.max(fs::symlink_metadata(entry.path()).ok()?.modified().ok()?);
    }
    Some(newest)
}

/// True when the newest witness in `dir` is older than the sweep grace. Every
/// deletion goes through this so a dir another process just opened or a
/// writer mid-staging is never removed.
fn packed_dir_idle(dir: &Path) -> bool {
    packed_dir_mtime(dir).is_some_and(|newest| newest < SystemTime::now() - PACKED_SWEEP_MIN_AGE)
}

/// None keeps the dir. Only proof removes one: a present source that hashes
/// to another identity, or a file gone from a mount point that is still
/// mounted. An unplugged volume, an unreadable file, a corrupt manifest or
/// no manifest proves nothing. The idle gate is a race grace, never a reason.
fn packed_orphan_reason(dir: &Path) -> Option<&'static str> {
    let manifest = dir.join(".manifest");
    if !manifest.is_file() || !packed_dir_idle(dir) {
        return None;
    }
    match manifest_source(dir) {
        Ok(Source::Present(identity)) if dir_name_identity(dir) != Some(identity.as_str()) => {
            Some("source fingerprint changed")
        }
        Ok(Source::Deleted) => Some("source deleted"),
        _ => None,
    }
}

/// Every count uses `symlink_metadata`: a symlinked entry counts only its own
/// (tiny) link and the target is never followed — bytes outside the cache
/// dir must not drive eviction or recorded spend.
fn dir_size(dir: &Path) -> u64 {
    if !is_real_dir(dir) {
        return 0;
    }
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|e| fs::symlink_metadata(e.path()).ok())
                .filter(|m| m.is_file() || m.is_symlink())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// Try an exclusive flock on `path` non-blockingly. Some(_) keeps the fd as
/// a guard; the inner None means the file does not exist (nothing can be
/// holding a lock). None means a live holder or an error: keep the dir.
fn try_exclusive_lock(path: &Path) -> Option<Option<File>> {
    // macOS flock requires a writable fd for LOCK_EX; a read-only file falls
    // back to read-only and relies on the kernel accepting or refusing it.
    let file = match OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .or_else(|_| File::open(path))
    {
        Ok(f) => f,
        Err(e) if e.kind() == ErrorKind::NotFound => return Some(None),
        Err(_) => return None,
    };
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return None;
    }
    // The file could have been unlinked between open and flock; only act on
    // the dir whose lock inode we actually hold.
    let held = file.metadata().ok();
    let live = fs::metadata(path).ok();
    match (held, live) {
        (Some(a), Some(b)) if a.dev() == b.dev() && a.ino() == b.ino() => Some(Some(file)),
        _ => None,
    }
}

/// `lock` names the file every user of the dir flocks: `MAPPED_LOCK` for a
/// mapped dir, `RETIRED_CHUNK_LOCK` for a retired chunk dir.
fn locked_remove(dir: &Path, lock: &str, reason: &str, still_stale: impl Fn(&Path) -> bool) -> u64 {
    // In-process exclusion first: a same-process holder's SH flock cannot
    // stop this process's EX request, so the flock alone would let us delete
    // a dir a pack of this process serves. The registry lock is held through
    // remove_dir_all so a holder racing in is refused registration and can
    // only observe the deletion.
    let Some(_reader_exclusion) = packed_deletion_guard(dir) else {
        debug!(
            "packed cache: keeping {} (mapped pack is held in this process)",
            dir.display()
        );
        return 0;
    };
    // A mapped dir has no creation window: `hold_mapped_dir` takes the SH
    // lock before the manifest is written, so a lock-free manifested dir is
    // one nobody is reading or building.
    let Some(_guard) = try_exclusive_lock(&dir.join(lock)) else {
        debug!(
            "packed cache: keeping {} (its {lock} file is held)",
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
    locked_remove(dir, MAPPED_LOCK, reason, |d| {
        is_real_dir(d) && packed_orphan_reason(d).is_some()
    })
}

/// Best-effort mtime stamp; keeps the dir young for idle-expiry and LRU.
pub(super) fn stamp_mtime(path: &Path) {
    let Ok(file) = OpenOptions::new().write(true).open(path) else {
        return;
    };
    let _ = file.set_modified(SystemTime::now());
}

/// Admission for one mapped-pack file of `file_bytes`: the free-space
/// reserve on the volume `dirfd` pins. On success the 0-length `staging`
/// file is sized to `file_bytes`. `Err` carries the reason.
pub(super) fn admit_mapped_file(dirfd: &File, staging: &File, file_bytes: u64) -> Result<()> {
    if !has_disk_space(dirfd, file_bytes) {
        return Err(err(format!(
            "Not enough disk space for {:.2} GiB plus the 16 GiB reserve",
            file_bytes as f64 / (1u64 << 30) as f64
        )));
    }
    staging.set_len(file_bytes).map_err(err)
}

/// Remove orphaned sibling `.mlx-qwen4-mapped-v3-*` dirs, and every
/// `.mlx-qwen4-packed-v2-<64 lowercase hex>` dir once it is idle and
/// unlocked. Never fatal.
fn sweep_packed_cache(parent: &Path, current: &Path) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Ok(entries) = fs::read_dir(parent) else {
            return;
        };
        for entry in entries.flatten() {
            let dir = entry.path();
            if dir == current || !is_real_dir(&dir) {
                continue;
            }
            if is_retired_chunk_dir(&dir) {
                // A v0.0.16 process takes no lock and stamps nothing on
                // reads, and stops writing once its ledger is full, so an
                // idle dir can be removed under it: its reads then miss and
                // re-import, never see torn data. A dir with no `budget` yet
                // has nothing to lock; only its fresh mtime keeps it.
                if packed_dir_idle(&dir) {
                    locked_remove(&dir, RETIRED_CHUNK_LOCK, "retired chunk format", |d| {
                        is_real_dir(d) && packed_dir_idle(d)
                    });
                }
                continue;
            }
            if dir_name_identity(&dir).is_none() {
                continue;
            }
            if let Some(reason) = packed_orphan_reason(&dir) {
                remove_packed_dir(&dir, reason);
            }
        }
    }));
}

/// In-process holder registry, keyed by pack dir. `hold_mapped_dir` bumps
/// its dir's count for the pack's lifetime; the sweep's `locked_remove`
/// must hold this lock while it unlinks — which atomically covers both
/// "holder already counted" (skip the delete) and "deletion already in
/// flight" (the holder registers after it and finds the dir gone). Needed
/// because BSD flock is per-process: the pack's lock file alone cannot
/// exclude a same-process holder.
fn packed_readers() -> &'static std::sync::Mutex<std::collections::HashMap<PathBuf, usize>> {
    static REGISTRY: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<PathBuf, usize>>,
    > = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Exclusive side for deletion paths: Some(guard) only while the registry
/// can be locked AND no reader holds this dir. Hold the guard through the
/// unlink so a reader racing in is refused the registry lock and goes
/// unguarded — it can only observe the deletion, never straddle it.
fn packed_deletion_guard(
    dir: &Path,
) -> Option<std::sync::MutexGuard<'static, std::collections::HashMap<PathBuf, usize>>> {
    let mut registry = match packed_readers().try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
        // A reader is mid-registration or another delete is running.
        Err(std::sync::TryLockError::WouldBlock) => return None,
    };
    if registry.get(dir).is_some_and(|&n| n > 0) {
        return None;
    }
    // Vacant entries accumulate one per deleted dir; reap while held.
    registry.retain(|_, n| *n > 0);
    Some(registry)
}

/// Held for a mapped pack's lifetime: a per-dir in-process count plus an SH
/// flock on the dir's lock file. The sweep must win packed_deletion_guard
/// AND an EX flock on that file, so a held pack makes both fail — the dir
/// cannot be removed out from under a model that serves or builds it.
pub(super) struct ReadGuard {
    dir: Option<PathBuf>,
    _flock: Option<File>,
}

impl ReadGuard {
    /// Register `dir` in the holder registry and keep `flock` for the
    /// guard's lifetime. Holders BLOCK on the registry mutex — its critical
    /// sections are tiny (a count bump, or a whole remove_dir_all a deleter
    /// holds it through). A blocking lock still cannot straddle a delete:
    /// either we register before the deleter's try_lock (it sees the count
    /// and skips) or after it finishes (the dir is gone and the hold's
    /// lock-file check fails).
    fn registered(dir: &Path, flock: Option<File>) -> Self {
        let mut registry = packed_readers()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        *registry.entry(dir.to_path_buf()).or_insert(0) += 1;
        drop(registry);
        Self {
            dir: Some(dir.to_path_buf()),
            _flock: flock,
        }
    }
}

/// Hold a mapped dir for a `MappedPack`'s lifetime: the in-process registry
/// entry plus an SH flock on its lock file, both of which `locked_remove`
/// must win before removing the dir. Taken before anything is written into
/// the dir, so the dir is never evictable while a load builds or reads it.
/// `Err` means an evictor holds the lock or removed the dir meanwhile; the
/// load then reads its experts from the checkpoint.
pub(super) fn hold_mapped_dir(dir: &Path, dirfd: &File) -> io::Result<ReadGuard> {
    let lock = dir_open_at(
        dirfd,
        OsStr::new(MAPPED_LOCK),
        libc::O_RDWR | libc::O_CREAT | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        0o644,
    )?;
    // SAFETY: `lock` owns an open descriptor for the flock's lifetime.
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) } != 0 {
        return Err(io::Error::new(
            ErrorKind::WouldBlock,
            "mapped pack dir is being evicted",
        ));
    }
    // An evictor that won the EX lock before our open removes the whole dir:
    // the name we hold is then gone from the (unlinked) directory.
    let held = lock.metadata()?;
    let live = dir_stat(dirfd, OsStr::new(MAPPED_LOCK))
        .ok_or_else(|| io::Error::new(ErrorKind::NotFound, "mapped pack dir was removed"))?;
    if held.dev() != live.st_dev as u64 || held.ino() != live.st_ino {
        return Err(io::Error::new(
            ErrorKind::NotFound,
            "mapped pack dir was replaced",
        ));
    }
    Ok(ReadGuard::registered(dir, Some(lock)))
}

impl Drop for ReadGuard {
    fn drop(&mut self) {
        let Some(dir) = self.dir.take() else {
            return;
        };
        let mut registry = match packed_readers().lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if let Some(n) = registry.get_mut(&dir) {
            *n = n.saturating_sub(1);
        }
    }
}

impl PackedCache {
    pub fn new(source: &Path, files: BTreeSet<PathBuf>) -> Result<Self> {
        // The source checkpoint may live on a slow external volume. Keep the
        // pack in the OS application cache.
        let root = std::env::var_os("HOME")
            .map(|home| {
                PathBuf::from(home).join(if cfg!(target_os = "macos") {
                    "Library/Caches/mlx-node/qwen4-packed"
                } else {
                    ".cache/mlx-node/qwen4-packed"
                })
            })
            .unwrap_or_else(|| source.parent().unwrap_or(Path::new(".")).to_path_buf());
        Self::new_at(&root, files)
    }

    /// Fingerprint the sources and sweep orphaned sibling packs under
    /// `root`. Nothing is created here: the pack builder creates the dir,
    /// takes its lock and only then publishes the manifest.
    pub(super) fn new_at(root: &Path, files: BTreeSet<PathBuf>) -> Result<Self> {
        let stats = stat_sources(files.into_iter())
            .into_iter()
            .collect::<io::Result<Vec<_>>>()
            .map_err(err)?;
        let identity = packed_identity(&stats);
        let canonical: Vec<PathBuf> = stats.into_iter().map(|(path, _)| path).collect();
        let cache = Self {
            dir: root.join(format!("{MAPPED_DIR_PREFIX}{identity}")),
            manifest: pack_manifest(&canonical),
        };
        sweep_packed_cache(root, &cache.dir);
        Ok(cache)
    }

    /// The source fingerprint the pack dir is named for.
    pub(super) fn identity(&self) -> Option<&str> {
        dir_name_identity(&self.dir)
    }

    pub(super) fn mapped_dir(&self) -> &Path {
        &self.dir
    }

    pub(super) fn manifest(&self) -> &[u8] {
        &self.manifest
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

/// Free-space check on the writer's pinned dir fd: the reservation guards
/// the volume the fd's inode actually lives on, not whatever `self.root`
/// might resolve to after a swap.
pub(super) fn has_disk_space(dirfd: &File, needed: u64) -> bool {
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `dirfd` is an open descriptor and `stat` is valid memory to fill.
    if unsafe { libc::fstatvfs(dirfd.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
        return false;
    }
    // SAFETY: fstatvfs returned success, so `stat` is fully initialized.
    let stat = unsafe { stat.assume_init() };
    (stat.f_bavail as u64).saturating_mul(stat.f_frsize) >= FREE_RESERVE + needed
}

/// The three arrays of a quantized chunk as their on-disk `ArrayInfo` list
/// and one concatenated little-endian payload: the per-expert record the
/// mapped pack's sections hold.
pub(super) fn encode_weight(weight: &Weight) -> Result<(Vec<ArrayInfo>, Vec<u8>)> {
    let mut arrays = Vec::with_capacity(3);
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
        arrays.push(ArrayInfo {
            dtype: dtype.into(),
            shape: a.shape()?.to_vec(),
            bytes: bytes.len(),
        });
        data.extend_from_slice(&bytes);
    }
    Ok((arrays, data))
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

    /// Publish `cache`'s manifest in its pack dir, as the pack builder does
    /// once it holds the dir.
    fn published(cache: &PackedCache) -> PathBuf {
        let dir = cache.mapped_dir().to_path_buf();
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(".manifest"), cache.manifest()).unwrap();
        dir
    }

    #[test]
    fn chunk_encoding_preserves_codes_signed_scales_and_quantized_outputs() {
        let x = MxArray::from_float32(
            &(0..256)
                .map(|i| (i as f32 * 0.07).sin())
                .collect::<Vec<_>>(),
            &[1, 256],
        )
        .unwrap();
        for mode in ["affine", "q4k", "q6k"] {
            let w = weight(mode);
            let (arrays, data) = encode_weight(&w).unwrap();
            let (loaded, _) = VerifiedChunk {
                header: Header {
                    key: mode.into(),
                    group: w.group,
                    bits: w.bits,
                    mode: w.mode.clone(),
                    arrays,
                },
                file_len: data.len() as u64,
                data,
            }
            .into_weight()
            .unwrap();
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
    fn source_change_renames_the_pack_dir_without_reading_payloads() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"first").unwrap();
        let a = PackedCache::new_at(&parent, BTreeSet::from([source.clone()])).unwrap();
        fs::write(&source, b"replacement").unwrap();
        let b = PackedCache::new_at(&parent, BTreeSet::from([source.clone()])).unwrap();
        assert_ne!(a.mapped_dir(), b.mapped_dir());
    }

    #[test]
    fn manifest_records_source_dir_and_recomputes_identity() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source.clone()])).unwrap();
        let dir = published(&cache);
        let manifest = fs::read_to_string(dir.join(".manifest")).unwrap();
        let canonical = source.canonicalize().unwrap();
        let source_dir = canonical.parent().unwrap();
        let mut lines = manifest.lines();
        assert_eq!(
            lines.next().unwrap(),
            format!("path_hex={}", hex(source_dir.as_os_str().as_bytes())),
            "the authoritative hex record leads the manifest:\n{manifest}"
        );
        let mount = mount_point(source_dir);
        assert_eq!(
            lines.next().unwrap(),
            format!("mount_hex={}", hex(mount.as_os_str().as_bytes())),
            "the source's mount point follows it:\n{manifest}"
        );
        assert!(is_mounted(&mount), "{} must be mounted", mount.display());
        assert!(
            manifest.lines().any(|line| line == "file=source.gguf"),
            "the raw file record follows its hex twin:\n{manifest}"
        );
        assert_eq!(
            manifest_source(&dir).unwrap(),
            Source::Present(dir_name_identity(&dir).unwrap().to_owned())
        );
        // A file gone from a mounted volume proves the pack stale; the verdict
        // still waits out the idle grace on the newest witness.
        fs::remove_file(&source).unwrap();
        assert_eq!(manifest_source(&dir).unwrap(), Source::Deleted);
        assert_eq!(packed_orphan_reason(&dir), None);
        age(&dir.join(".manifest"), 7200);
        age_dir(&dir, 7200);
        assert_eq!(packed_orphan_reason(&dir), Some("source deleted"));
    }

    /// A raw `file=` line carries the filename's bytes verbatim, so a name
    /// holding `\nfile_hex=6262` would otherwise split into a bogus
    /// authoritative-looking record. `pack_manifest` must omit the raw line
    /// for that file and let only its real `file_hex=` twin stand, so the
    /// recomputed identity still matches the dir name.
    #[test]
    fn manifest_omits_raw_lines_that_could_inject_records() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        let source = temp.0.join("a\nfile_hex=6262\npath_hex=6262.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source.clone()])).unwrap();
        let dir = published(&cache);
        let manifest = fs::read_to_string(dir.join(".manifest")).unwrap();
        assert!(
            manifest.lines().all(|line| !line.starts_with("file=")),
            "the newline-bearing filename must not emit a raw record:\n{manifest}"
        );
        assert_eq!(
            manifest
                .lines()
                .filter(|line| line.starts_with("file_hex="))
                .count(),
            1,
            "only the authoritative file record may exist:\n{manifest}"
        );
        assert!(
            !manifest.lines().any(|line| line == "file_hex=6262"),
            "the injected record must never be written:\n{manifest}"
        );
        assert!(
            manifest.lines().all(|line| !line.starts_with("mount=")),
            "the mount point is written as hex only:\n{manifest}"
        );
        assert_eq!(
            manifest_source(&dir).unwrap(),
            Source::Present(dir_name_identity(&dir).unwrap().to_owned()),
            "the real file still verifies through its hex twin"
        );
    }

    #[test]
    fn stale_sibling_pack_is_swept_while_the_current_and_foreign_dirs_stay() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        fs::create_dir(&parent).unwrap();
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source.clone()])).unwrap();
        // The current identity's dir, with the same orphan manifest and age
        // as the stale sibling: only being current keeps it.
        let current = cache.mapped_dir().to_path_buf();
        fs::create_dir(&current).unwrap();
        fs::write(current.join("x.mapped"), vec![0_u8; 100]).unwrap();
        fs::write(current.join(".manifest"), deleted_manifest(&parent)).unwrap();
        age(&current.join("x.mapped"), 7200);
        age(&current.join(".manifest"), 7200);
        age_dir(&current, 7200);
        assert_eq!(packed_orphan_reason(&current), Some("source deleted"));
        let stale = mapped_fixture(&parent, 'a', 100, true);
        assert_eq!(packed_orphan_reason(&stale), Some("source deleted"));
        // A manifest-less dir is never orphan-deleted: nothing proves it stale.
        let unknown = parent.join(format!("{MAPPED_DIR_PREFIX}{}", "b".repeat(64)));
        fs::create_dir(&unknown).unwrap();
        fs::write(unknown.join("x.mapped"), b"data").unwrap();
        // A name under the prefix that is not a 64-hex identity is foreign;
        // an idle chunk dir of the retired format is removed.
        let foreign = parent.join(format!("{MAPPED_DIR_PREFIX}backup"));
        let chunks = parent.join(format!("{RETIRED_CHUNK_PREFIX}{}", "d".repeat(64)));
        for dir in [&foreign, &chunks] {
            fs::create_dir(dir).unwrap();
            fs::write(dir.join(".manifest"), deleted_manifest(&parent)).unwrap();
            age(&dir.join(".manifest"), 7200);
            age_dir(dir, 7200);
        }
        assert_eq!(packed_orphan_reason(&foreign), Some("source deleted"));
        until_gone(
            || drop(PackedCache::new_at(&parent, BTreeSet::from([source.clone()])).unwrap()),
            || !stale.exists() && !chunks.exists(),
        );
        assert!(
            !stale.exists(),
            "an aged mapped dir with a dead source is an orphan"
        );
        assert!(current.exists(), "the current identity's dir is current");
        assert!(unknown.exists());
        assert!(foreign.exists());
        assert!(!chunks.exists());
    }

    #[test]
    fn idle_but_identity_matched_dir_is_never_an_orphan() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();
        let dir = published(&cache);
        // Age every witness past the sweep grace: the dir is as idle as a
        // cache can be, yet its manifest still recomputes to the dir-name
        // identity — a healthy-but-idle model cache, not an orphan.
        age(&dir.join(".manifest"), 7200);
        age_dir(&dir, 7200);
        assert!(
            packed_dir_idle(&dir),
            "the fixture must read as idle for this regression to bite"
        );
        assert_eq!(packed_orphan_reason(&dir), None);
        // And the full sweep must leave it alone.
        sweep_packed_cache(&parent, &temp.0.join("nonexistent-current"));
        assert!(dir.exists());
        assert!(dir.join(".manifest").is_file());
    }

    #[test]
    fn symlinked_identity_dir_is_foreign_to_the_sweep() {
        let temp = Temp::new();
        let parent = temp.0.join("root");
        fs::create_dir(&parent).unwrap();
        // Bytes outside the cache root must never be deleted, even behind a
        // stale-looking manifest.
        let outside = temp.0.join("outside");
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("x.mapped"), vec![0_u8; 1 << 20]).unwrap();
        fs::write(outside.join(".manifest"), deleted_manifest(&temp.0)).unwrap();
        age(&outside.join("x.mapped"), 7200);
        age(&outside.join(".manifest"), 7200);
        age_dir(&outside, 7200);
        assert_eq!(packed_orphan_reason(&outside), Some("source deleted"));
        // A root-level entry whose NAME claims an identity but whose inode is
        // a symlink: foreign, immune to the sweep.
        let link = parent.join(format!("{MAPPED_DIR_PREFIX}{}", "a".repeat(64)));
        std::os::unix::fs::symlink(&outside, &link).unwrap();
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let _cache = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();
        assert!(link.is_symlink(), "the sweep must never touch a link");
        assert_eq!(
            fs::canonicalize(&link).unwrap(),
            outside.canonicalize().unwrap(),
            "the link target is intact"
        );
        assert!(outside.join("x.mapped").is_file());
    }

    // flock is per-process on BSD/macOS, so a held pack lock can only be
    // observed from a second process.
    #[test]
    #[ignore = "helper process for the held-lock sweep tests"]
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
        let operation = if std::env::var_os("MLX_NODE_PACKED_LOCK_TEST_SHARED").is_some() {
            libc::LOCK_SH
        } else {
            libc::LOCK_EX
        };
        assert_eq!(unsafe { libc::flock(file.as_raw_fd(), operation) }, 0);
        fs::write(acquired, b"acquired").unwrap();
        // The parent kills this process once the sweep assertion completes.
        std::thread::sleep(Duration::from_secs(60));
    }

    /// Dropping a bare `Child` leaves it running: a failed assertion would
    /// keep the lock and the test's output pipe open for the child's sleep.
    struct LockHolder(std::process::Child);

    impl Drop for LockHolder {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    /// A second process holding `path` shared until dropped.
    fn shared_lock_holder(temp: &Path, path: &Path) -> LockHolder {
        let acquired = temp.join(format!("{}.acquired", uuid::Uuid::new_v4()));
        let holder = LockHolder(
            std::process::Command::new(std::env::current_exe().unwrap())
                .arg("--exact")
                .arg("models::qwen4_exp::packed_cache::tests::packed_budget_lock_child")
                .arg("--ignored")
                .env("MLX_NODE_PACKED_LOCK_TEST_PATH", path)
                .env("MLX_NODE_PACKED_LOCK_TEST_ACQUIRED", &acquired)
                .env("MLX_NODE_PACKED_LOCK_TEST_SHARED", "1")
                .spawn()
                .unwrap(),
        );
        for _ in 0..500 {
            if acquired.is_file() {
                return holder;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("child never acquired {}", path.display());
    }

    fn hex(bytes: &[u8]) -> String {
        String::from_utf8(packed_hex_encode(bytes)).unwrap()
    }

    /// A manifest naming `source_dir/x.gguf`, recorded on `mount`.
    fn manifest_naming(source_dir: &Path, mount: &Path) -> String {
        format!(
            "path_hex={}\nmount_hex={}\nfile_hex={}\n",
            hex(source_dir.as_os_str().as_bytes()),
            hex(mount.as_os_str().as_bytes()),
            hex(b"x.gguf")
        )
    }

    /// A manifest proving its source deleted: no `x.gguf` in `source_dir`,
    /// whose mount point is mounted.
    fn deleted_manifest(source_dir: &Path) -> String {
        let source_dir = source_dir.canonicalize().unwrap();
        assert!(!source_dir.join("x.gguf").exists());
        manifest_naming(&source_dir, &mount_point(&source_dir))
    }

    /// A mapped dir fixture: `<letter>*64` identity, one `x.mapped` of
    /// `bytes`, a manifest whose source is deleted, every witness aged past
    /// the sweep grace when `aged`.
    fn mapped_fixture(parent: &Path, letter: char, bytes: usize, aged: bool) -> PathBuf {
        let dir = parent.join(format!(
            "{MAPPED_DIR_PREFIX}{}",
            letter.to_string().repeat(64)
        ));
        fs::create_dir(&dir).unwrap();
        fs::write(dir.join("x.mapped"), vec![0_u8; bytes]).unwrap();
        fs::write(dir.join(".manifest"), deleted_manifest(parent)).unwrap();
        fs::write(dir.join(MAPPED_LOCK), b"").unwrap();
        if aged {
            for entry in ["x.mapped", ".manifest", MAPPED_LOCK] {
                age(&dir.join(entry), 7200);
            }
            age_dir(&dir, 7200);
        }
        dir
    }

    /// The sweep keeps a dir whenever the process-wide holder registry is
    /// busy (another test of this binary opening or sweeping a pack), so
    /// `sweep` is repeated until `gone` holds or 1 s has passed; a dir the
    /// sweep keeps on purpose survives every repeat.
    fn until_gone(mut sweep: impl FnMut(), gone: impl Fn() -> bool) {
        for _ in 0..200 {
            sweep();
            if gone() {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn held_mapped_dir_lock_blocks_the_sweep() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        fs::create_dir(&parent).unwrap();
        let current = parent.join(format!("{MAPPED_DIR_PREFIX}{}", "c".repeat(64)));
        fs::create_dir(&current).unwrap();
        // Another process reads the pack: its SH lock keeps the orphan.
        let stale = mapped_fixture(&parent, 'a', 100, true);
        let holder = shared_lock_holder(&temp.0, &stale.join(MAPPED_LOCK));
        sweep_packed_cache(&parent, &current);
        assert!(stale.exists(), "an open pack is not swept");
        drop(holder);
        until_gone(|| sweep_packed_cache(&parent, &current), || !stale.exists());
        assert!(!stale.exists());
        // The same process holding the dir through `hold_mapped_dir` (a
        // loaded MappedPack) is excluded too, by the registry and the lock.
        let held = mapped_fixture(&parent, 'b', 100, true);
        let dirfd = open_dir_nofollow(&held).unwrap();
        let guard = hold_mapped_dir(&held, &dirfd).unwrap();
        assert_eq!(locked_remove(&held, MAPPED_LOCK, "test", |_| true), 0);
        assert!(held.exists());
        drop(guard);
        let removed = std::cell::Cell::new(0);
        until_gone(
            || removed.set(removed.get() + locked_remove(&held, MAPPED_LOCK, "test", |_| true)),
            || removed.get() > 0,
        );
        assert!(removed.get() > 0);
        assert!(!held.exists());
        // Holding a dir an evictor removed meanwhile is refused.
        let gone = mapped_fixture(&parent, 'd', 100, true);
        let dirfd = open_dir_nofollow(&gone).unwrap();
        fs::remove_dir_all(&gone).unwrap();
        assert!(hold_mapped_dir(&gone, &dirfd).is_err());
    }

    /// A retired chunk dir holding `files`, every witness aged past the sweep
    /// grace when `aged`.
    fn chunk_fixture(parent: &Path, suffix: &str, files: &[(&str, &[u8])], aged: bool) -> PathBuf {
        let dir = parent.join(format!("{RETIRED_CHUNK_PREFIX}{suffix}"));
        fs::create_dir(&dir).unwrap();
        for (name, bytes) in files {
            fs::write(dir.join(name), bytes).unwrap();
        }
        if aged {
            for entry in fs::read_dir(&dir).unwrap() {
                age(&entry.unwrap().path(), 7200);
            }
            age_dir(&dir, 7200);
        }
        dir
    }

    #[test]
    fn retired_chunk_dirs_are_swept_once_idle_and_unlocked() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        fs::create_dir(&parent).unwrap();
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source.clone()])).unwrap();
        let identity = cache.identity().unwrap();
        // v0.0.16 writes a `budget` ledger and `.pack` chunks, no manifest.
        let ledger = 64_u64.to_le_bytes();
        let written: [(&str, &[u8]); 2] = [(RETIRED_CHUNK_LOCK, &ledger), ("x.pack", &[0; 64])];
        let released = chunk_fixture(&parent, &"a".repeat(64), &written, true);
        // Named for the source being loaded, with a manifest that still
        // verifies: the format is retired, so a live source keeps nothing.
        let manifested = [written[0], written[1], (".manifest", cache.manifest())];
        let same_source = chunk_fixture(&parent, identity, &manifested, true);
        assert_eq!(
            manifest_source(&same_source).unwrap(),
            Source::Present(identity.to_owned())
        );
        // origin/main writes the manifest at load and `budget` only with its
        // first chunk, so a dir it never wrote to has no ledger to lock.
        let stub_files: [(&str, &[u8]); 1] = [(".manifest", b"path=/gone\nfile=x.gguf\n")];
        let stub = chunk_fixture(&parent, &"e".repeat(64), &stub_files, true);
        let fresh = chunk_fixture(&parent, &"b".repeat(64), &written, false);
        let locked = chunk_fixture(&parent, &"c".repeat(64), &written, true);
        let foreign = chunk_fixture(&parent, "backup", &written, true);
        for dir in [&released, &same_source, &stub, &locked, &foreign] {
            assert!(packed_dir_idle(dir), "{} must read as idle", dir.display());
        }
        assert!(!packed_dir_idle(&fresh));
        // An old build reading the dir holds its ledger shared.
        let holder = shared_lock_holder(&temp.0, &locked.join(RETIRED_CHUNK_LOCK));
        until_gone(
            || drop(PackedCache::new_at(&parent, BTreeSet::from([source.clone()])).unwrap()),
            || !released.exists() && !same_source.exists(),
        );
        assert!(
            !released.exists(),
            "an idle manifest-less dir v0.0.16 wrote is removed"
        );
        assert!(
            !same_source.exists(),
            "an idle dir of the current source is removed"
        );
        assert!(
            !stub.exists(),
            "an idle manifest-only dir with no ledger is removed"
        );
        assert!(fresh.exists(), "a dir used within the hour stays");
        assert!(locked.exists(), "a dir whose ledger is held stays");
        assert!(
            foreign.exists(),
            "a name that is not a 64-hex identity is foreign"
        );
        drop(holder);
        until_gone(
            || drop(PackedCache::new_at(&parent, BTreeSet::from([source.clone()])).unwrap()),
            || !locked.exists(),
        );
        assert!(!locked.exists(), "the released ledger no longer keeps it");
        assert!(fresh.exists());
        assert!(foreign.exists());
    }

    #[test]
    fn identity_hashes_path_inode_size_and_times_but_not_the_device() {
        let temp = Temp::new();
        let first = temp.0.join("a.gguf");
        let second = temp.0.join("b.gguf");
        fs::write(&first, b"first split").unwrap();
        fs::write(&second, b"second").unwrap();
        let cache = PackedCache::new_at(
            &temp.0.join("cache"),
            BTreeSet::from([second.clone(), first.clone()]),
        )
        .unwrap();
        let mut hash = Sha256::new();
        hash.update(MAGIC);
        for path in [&first, &second] {
            let canonical = path.canonicalize().unwrap();
            let meta = fs::metadata(&canonical).unwrap();
            hash.update(canonical.as_os_str().as_bytes());
            for n in [
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
        let expected: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(cache.identity(), Some(expected.as_str()));
    }

    #[test]
    fn an_unmounted_source_volume_keeps_its_pack() {
        let temp = Temp::new();
        let base = temp.0.canonicalize().unwrap();
        let parent = base.join("cache");
        fs::create_dir(&parent).unwrap();
        let unplugged = base.join("Volumes/T7");
        let leftover = base.join("leftover");
        fs::create_dir(&leftover).unwrap();
        let live = mount_point(&base);
        assert!(!is_mounted(&unplugged));
        assert!(
            !is_mounted(&leftover),
            "a plain dir on its parent's device is no mount point"
        );
        assert!(is_mounted(&live), "{} must be mounted", live.display());
        let fixture = |letter: char, manifest: String| {
            let dir = mapped_fixture(&parent, letter, 100, true);
            fs::write(dir.join(".manifest"), manifest).unwrap();
            age(&dir.join(".manifest"), 7200);
            age_dir(&dir, 7200);
            dir
        };
        let away = fixture('1', manifest_naming(&unplugged.join("models"), &unplugged));
        let left_behind = fixture('2', manifest_naming(&leftover.join("models"), &leftover));
        let deleted = fixture('3', manifest_naming(&base, &live));
        let unrecorded = fixture(
            '4',
            format!(
                "path_hex={}\nfile_hex={}\n",
                hex(base.as_os_str().as_bytes()),
                hex(b"x.gguf")
            ),
        );
        assert_eq!(manifest_source(&away).unwrap(), Source::Unreachable);
        assert_eq!(manifest_source(&left_behind).unwrap(), Source::Unreachable);
        assert_eq!(manifest_source(&unrecorded).unwrap(), Source::Unreachable);
        assert_eq!(packed_orphan_reason(&away), None);
        assert_eq!(packed_orphan_reason(&left_behind), None);
        assert_eq!(packed_orphan_reason(&unrecorded), None);
        assert_eq!(packed_orphan_reason(&deleted), Some("source deleted"));
        let current = parent.join("current");
        until_gone(
            || sweep_packed_cache(&parent, &current),
            || !deleted.exists(),
        );
        assert!(!deleted.exists(), "a file gone from a live mount is swept");
        assert!(away.exists(), "an unplugged volume keeps its pack");
        assert!(
            left_behind.exists(),
            "a left-behind mount dir keeps its pack"
        );
        assert!(
            unrecorded.exists(),
            "a missing file with no mount record keeps its pack"
        );
    }

    /// Resets a `chmod 000` before `Temp` removes the tree.
    struct Mode(PathBuf, u32);

    impl Drop for Mode {
        fn drop(&mut self) {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(self.1));
        }
    }

    fn chmod(path: &Path, mode: u32) {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
    }

    #[test]
    fn a_corrupt_manifest_or_an_unreadable_source_keeps_the_pack() {
        let temp = Temp::new();
        let base = temp.0.canonicalize().unwrap();
        let parent = base.join("cache");
        fs::create_dir(&parent).unwrap();
        let path = hex(base.as_os_str().as_bytes());
        let mount = hex(mount_point(&base).as_os_str().as_bytes());
        let file = hex(b"x.gguf");
        let corrupt = [
            (
                '1',
                format!("path_hex=zz\nmount_hex={mount}\nfile_hex={file}\n"),
            ),
            ('2', format!("mount_hex={mount}\nfile_hex={file}\n")),
            ('3', format!("path_hex={path}\nmount_hex={mount}\n")),
            (
                '4',
                format!("path_hex={path}\nmount_hex=zz\nfile_hex={file}\n"),
            ),
        ];
        let mut kept = Vec::new();
        for (letter, manifest) in corrupt {
            let dir = mapped_fixture(&parent, letter, 100, true);
            fs::write(dir.join(".manifest"), &manifest).unwrap();
            age(&dir.join(".manifest"), 7200);
            age_dir(&dir, 7200);
            assert!(manifest_source(&dir).is_err(), "{manifest}");
            assert_eq!(packed_orphan_reason(&dir), None, "{manifest}");
            kept.push(dir);
        }
        // Root reads through any mode.
        let mut restore = Vec::new();
        if unsafe { libc::geteuid() } != 0 {
            let unreadable = mapped_fixture(&parent, '5', 100, true);
            restore.push(Mode(unreadable.join(".manifest"), 0o644));
            chmod(&unreadable.join(".manifest"), 0o000);
            assert!(manifest_source(&unreadable).is_err());
            assert_eq!(packed_orphan_reason(&unreadable), None);
            kept.push(unreadable);
            let source_dir = base.join("src");
            fs::create_dir(&source_dir).unwrap();
            fs::write(source_dir.join("x.gguf"), b"weights").unwrap();
            let cache =
                PackedCache::new_at(&parent, BTreeSet::from([source_dir.join("x.gguf")])).unwrap();
            let denied = published(&cache);
            age(&denied.join(".manifest"), 7200);
            age_dir(&denied, 7200);
            restore.push(Mode(source_dir.clone(), 0o755));
            chmod(&source_dir, 0o000);
            assert_eq!(manifest_source(&denied).unwrap(), Source::Unreachable);
            assert_eq!(packed_orphan_reason(&denied), None);
            kept.push(denied);
        }
        sweep_packed_cache(&parent, &parent.join("current"));
        for dir in &kept {
            assert!(dir.exists(), "{} was swept", dir.display());
        }
    }

    #[test]
    fn a_split_set_missing_one_file_on_a_mounted_volume_is_an_orphan() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        let pack = |name: &str| {
            let source_dir = temp.0.join(name);
            fs::create_dir(&source_dir).unwrap();
            let files = [source_dir.join("a.gguf"), source_dir.join("b.gguf")];
            for file in &files {
                fs::write(file, file.as_os_str().as_bytes()).unwrap();
            }
            let cache = PackedCache::new_at(&parent, BTreeSet::from(files.clone())).unwrap();
            (files, published(&cache))
        };
        let (files, removed) = pack("removed");
        fs::remove_file(&files[1]).unwrap();
        let (files, rewritten) = pack("rewritten");
        fs::write(&files[0], b"new weights").unwrap();
        for dir in [&removed, &rewritten] {
            age(&dir.join(".manifest"), 7200);
            age_dir(dir, 7200);
        }
        assert_eq!(manifest_source(&removed).unwrap(), Source::Deleted);
        assert_eq!(packed_orphan_reason(&removed), Some("source deleted"));
        assert_eq!(
            packed_orphan_reason(&rewritten),
            Some("source fingerprint changed")
        );
    }

    fn fixture_gguf() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp/model.gguf")
    }

    #[test]
    #[ignore = "helper process for fixture_loads_cannot_sweep_an_external_cache"]
    fn fixture_open_child() {
        super::super::weights::Store::open_fixture(&fixture_gguf(), None).unwrap();
    }

    /// HOME is overridden only in a child: other tests read it unlocked.
    #[test]
    fn fixture_loads_cannot_sweep_an_external_cache() {
        let temp = Temp::new();
        let home = temp.0.join("home");
        let parent = home.join(if cfg!(target_os = "macos") {
            "Library/Caches/mlx-node/qwen4-packed"
        } else {
            ".cache/mlx-node/qwen4-packed"
        });
        fs::create_dir_all(&parent).unwrap();
        let stub: [(&str, &[u8]); 1] = [(".manifest", b"path=/gone\nfile=x.gguf\n")];
        let retired = chunk_fixture(&parent, &"a".repeat(64), &stub, true);
        let orphan = mapped_fixture(&parent, 'b', 100, true);
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("models::qwen4_exp::packed_cache::tests::fixture_open_child")
            .arg("--ignored")
            .env("HOME", &home)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&child.stdout);
        assert!(
            child.status.success() && stdout.contains("test result: ok. 1 passed;"),
            "fixture child did not run: {stdout}{}",
            String::from_utf8_lossy(&child.stderr)
        );
        assert!(
            retired.exists(),
            "a fixture load swept the user's retired chunk dir"
        );
        assert!(
            orphan.exists(),
            "a fixture load swept the user's orphaned mapped pack"
        );
        until_gone(
            || drop(PackedCache::new_at(&parent, BTreeSet::from([fixture_gguf()])).unwrap()),
            || !retired.exists() && !orphan.exists(),
        );
        assert!(
            !retired.exists() && !orphan.exists(),
            "both dirs must be sweep-eligible or the checks above prove nothing"
        );
    }
}
