//! Optional, lossless SSD cache of already imported quantized matrix chunks.
//! Opening a Store only fingerprints source metadata. Entries are created on
//! demand, atomically, and contain at most one bounded row/expert chunk.
use std::collections::BTreeSet;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd};
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
/// The mapped expert pack of the same identity lives beside the chunk dir
/// under this prefix; both are one checkpoint's cache to the sweep and cap.
pub(super) const MAPPED_DIR_PREFIX: &str = ".mlx-qwen4-mapped-v3-";
/// A mapped dir's stable lock file: SH while a model reads or builds the
/// pack, EX by an evictor. Never renamed or unlinked (flock needs the inode).
const MAPPED_LOCK: &str = "lock";
/// Optional total cap across all identity dirs; unset/<=0 keeps the per-dir
/// DISK_BUDGET behaviour.
const TOTAL_CAP_ENV: &str = "MLX_QWEN4_PACKED_CACHE_MAX_GB";

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

/// Recompute an identity dir's source fingerprint from its `.manifest`. Any
/// missing source file or unreadable manifest is Err so callers can treat the
/// dir as orphaned. `path_hex=`/`file_hex=` lines (raw OS bytes as hex) are
/// preferred over the lossy `path=`/`file=` lines they twin; pre-hex
/// manifests keep working through the raw fields.
fn identity_from_manifest(dir: &Path) -> Result<String> {
    let data = fs::read(dir.join(".manifest")).map_err(err)?;
    let mut path_raw = None;
    let mut path_hex = None;
    let mut file_raws: Vec<&[u8]> = Vec::new();
    let mut file_hexes: Vec<&[u8]> = Vec::new();
    for line in data.split(|&b| b == b'\n') {
        if let Some(raw) = line.strip_prefix(b"path_hex=".as_slice()) {
            path_hex = Some(raw);
        } else if let Some(raw) = line.strip_prefix(b"path=".as_slice()) {
            path_raw = Some(raw);
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
    packed_identity(files.into_iter())
}

/// Serialize canonical source paths into `.manifest` bytes: the hex twins
/// come first (`path_hex=`, then every `file_hex=`), the raw `path=`/`file=`
/// display lines last. A raw line whose bytes contain a newline is omitted
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
/// follows links, so a `.mlx-qwen4-packed-v2-*` symlink would admit its
/// external target into sizing (`read_dir` walks it) and eviction. The
/// producers only ever emit real dirs, so links are foreign.
pub(super) fn is_real_dir(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|meta| meta.is_dir())
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
// .manifest/budget/.tmp/.pack writes outside the cache root. Every write
// under `self.root` therefore runs relative to a directory fd opened once
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

/// Open `name` inside `dirfd` for the `budget` ledger: read-write when the
/// ledger must be (re)created, read-only otherwise. O_NOFOLLOW keeps a
/// planted link from redirecting the flock and bytes outside the pinned
/// dir.
fn dir_open_file(dirfd: &File, name: &str, create: bool) -> io::Result<File> {
    let flags = if create {
        libc::O_RDWR | libc::O_CREAT | libc::O_NOFOLLOW | libc::O_CLOEXEC
    } else {
        libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC
    };
    dir_open_at(dirfd, OsStr::new(name), flags, 0o644)
}

/// `create_new` staging temp inside `dirfd`: O_EXCL keeps a pre-existing
/// name (or planted link) from being opened.
fn dir_create_new_file(dirfd: &File, name: &str) -> io::Result<File> {
    dir_open_at(
        dirfd,
        OsStr::new(name),
        libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        0o644,
    )
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DirKind {
    Chunks,
    Mapped,
}

/// Producer identities are exactly 64 lowercase hex chars (the blake-style
/// source fingerprint) under the chunk prefix or the mapped prefix. Anything
/// else under either prefix — `.mlx-qwen4-packed-v2-backup`, a typo, a manual
/// copy — is foreign: never sized, never swept, never evicted.
fn identity_dir_kind(dir: &Path) -> Option<(DirKind, &str)> {
    let name = dir.file_name()?.to_str()?;
    let (kind, suffix) = match name.strip_prefix(DIR_PREFIX) {
        Some(suffix) => (DirKind::Chunks, suffix),
        None => (DirKind::Mapped, name.strip_prefix(MAPPED_DIR_PREFIX)?),
    };
    (suffix.len() == 64
        && suffix
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()))
    .then_some((kind, suffix))
}

pub(super) fn dir_name_identity(dir: &Path) -> Option<&str> {
    identity_dir_kind(dir).map(|(_, identity)| identity)
}

fn is_mapped_dir(dir: &Path) -> bool {
    identity_dir_kind(dir).is_some_and(|(kind, _)| kind == DirKind::Mapped)
}

/// The other format's dir of the same identity. A load uses both, so the
/// sweep and the cap treat the current dir's twin as current too.
fn twin_dir(dir: &Path) -> Option<PathBuf> {
    let (kind, identity) = identity_dir_kind(dir)?;
    let prefix = match kind {
        DirKind::Chunks => MAPPED_DIR_PREFIX,
        DirKind::Mapped => DIR_PREFIX,
    };
    Some(dir.parent()?.join(format!("{prefix}{identity}")))
}

fn is_current(dir: &Path, current: &Path) -> bool {
    dir == current || twin_dir(current).is_some_and(|twin| twin == dir)
}

/// Producer chunk files are `{chunk_key}.pack` where chunk_key is a SHA-256
/// hexdigest — 64 lowercase hex chars. A `backup.pack` or `notes.pack`
/// dropped inside an identity dir is foreign: never counted as spend and
/// never a deletion candidate.
fn is_pack_name(name: &OsStr) -> bool {
    let Some(stem) = name.to_str().and_then(|n| n.strip_suffix(".pack")) else {
        return false;
    };
    stem.len() == 64
        && stem
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// Producer staging files are `{Uuid::new_v4()}.tmp` — 36 chars:
/// 8-4-4-4-12 lowercase hex separated by dashes, version nibble `4` at
/// position 14, variant `[89ab]` at 19. A `notes.tmp` or `snapshot.tmp` in
/// the dir is foreign: never counted as retained spend and never unlinked
/// by the stale-tmp reaper.
fn is_tmp_name(name: &OsStr) -> bool {
    let Some(stem) = name.to_str().and_then(|n| n.strip_suffix(".tmp")) else {
        return false;
    };
    let b = stem.as_bytes();
    b.len() == 36
        && [8usize, 13, 18, 23].iter().all(|&i| b[i] == b'-')
        && b[14] == b'4'
        && matches!(b[19], b'8' | b'9' | b'a' | b'b')
        && b.iter().enumerate().all(|(i, c)| {
            [8usize, 13, 18, 23].contains(&i) || (c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        })
}

/// `libc::stat` timestamps are signed: a restored or manually stamped file
/// can carry a pre-epoch mtime, and casting a negative `st_mtime` to u64
/// wraps to a near-`u64::MAX` duration whose epoch addition panics. Clamp
/// negative values to UNIX_EPOCH — such a file is simply the oldest.
fn stat_mtime(stat: &libc::stat) -> SystemTime {
    if stat.st_mtime < 0 || stat.st_mtime_nsec < 0 {
        return SystemTime::UNIX_EPOCH;
    }
    // A corrupt far-future mtime must read as FRESH (kept, never swept):
    // the conservative failure is retention, not deletion.
    SystemTime::UNIX_EPOCH
        .checked_add(Duration::from_secs(stat.st_mtime as u64))
        .and_then(|t| t.checked_add(Duration::from_nanos(stat.st_mtime_nsec as u64)))
        .unwrap_or_else(SystemTime::now)
}

/// Newest mtime over the dir and every direct entry (.manifest, budget,
/// .pack, in-flight .tmp); a present-but-unreadable mtime counts as fresh
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

/// None keeps the dir (manifest-less dirs are never orphan-deleted: their
/// staleness cannot be proven). The idle gate is a safety grace for the
/// orphan reasons, never itself a reason: an idle dir whose manifest still
/// recomputes to its name identity is a healthy cache, and deleting it would
/// only force a full repack on the next load.
fn packed_orphan_reason(dir: &Path) -> Option<&'static str> {
    let manifest = dir.join(".manifest");
    if !manifest.is_file() || !packed_dir_idle(dir) {
        return None;
    }
    match identity_from_manifest(dir) {
        Err(_) => Some("unreadable manifest or missing source"),
        Ok(identity) if dir_name_identity(dir) != Some(identity.as_str()) => {
            Some("source fingerprint changed")
        }
        // Identity matches the dir name: the contents are still correct for
        // their source — idleness alone is not an orphan reason.
        Ok(_) => None,
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

/// Bytes actually held by .pack files — the same quantity the budget ledger
/// records. A .pack that is itself a symlink contributes only its link bytes,
/// and a symlinked `dir` is foreign: read_dir would traverse its target.
fn pack_bytes(dir: &Path) -> u64 {
    if !is_real_dir(dir) {
        return 0;
    }
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| is_pack_name(&e.file_name()))
                .filter_map(|e| fs::symlink_metadata(e.path()).ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// A mapped dir's spend: its `<tensor>.mapped` files and the `.<tensor>.tmp`
/// a builder is filling or a crash left (the next load rebuilds it in
/// place, so it stays spend until then). Lock files and the manifest are
/// not counted, like the chunk dir's ledger.
fn is_mapped_entry(name: &OsStr) -> bool {
    name.to_str().is_some_and(|name| {
        name.ends_with(".mapped") || (name.starts_with('.') && name.ends_with(".tmp"))
    })
}

fn mapped_bytes(dir: &Path) -> u64 {
    if !is_real_dir(dir) {
        return 0;
    }
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| is_mapped_entry(&e.file_name()))
                .filter_map(|e| fs::symlink_metadata(e.path()).ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// Bytes the sweep deliberately retains in `{uuid}.tmp` crash leftovers:
/// regular files younger than `PACKED_SWEEP_MIN_AGE` — the same freshness
/// gate the sweep applies before unlinking one. Aged tmps and non-regular
/// files are debris, not spend: the sweep reclaims them on its next pass.
/// These count toward admission estimates (a writer that crashed between
/// reservation and rename still owns the bytes on disk, and repeated
/// interrupted writes must not sail past the configured cap) but never the
/// `budget` ledger, which records committed `.pack` bytes only — the same
/// split `reconciled_sibling_pack_spend` and `shed_oldest_packs` keep.
fn retained_tmp_bytes(dir: &Path) -> u64 {
    if !is_real_dir(dir) {
        return 0;
    }
    let cutoff = SystemTime::now() - PACKED_SWEEP_MIN_AGE;
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| is_tmp_name(&e.file_name()))
                .filter_map(|e| fs::symlink_metadata(e.path()).ok())
                .filter(|m| m.is_file() && m.modified().is_ok_and(|t| t > cutoff))
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// Best-effort sibling spend estimate: the ledger u64 when it reads, else a
/// live .pack byte count, plus retained `.tmp` bytes in both cases — a crash
/// between reservation and rename owns those bytes and the sweep keeps them
/// for the whole grace, so they are spend either way. A ledger value can lag
/// reality (a skipped/failed replacement write leaves the old target on disk
/// but absent from `used`), so a readable ledger is floored at the real byte
/// count — the estimate can overcount pending spend, never undercount (a
/// live writer's in-flight `.tmp` momentarily doubles its own reservation,
/// which is the safe direction). Advisory only — the sibling's flock stays
/// with its writer and any failure degrades to the byte count, never to an
/// error.
fn sibling_pack_estimate(dir: &Path) -> u64 {
    let mut n = [0u8; 8];
    let from_ledger = File::open(dir.join("budget"))
        .and_then(|mut f| f.read_exact(&mut n))
        .is_ok();
    let packs = pack_bytes(dir);
    let committed = if from_ledger {
        u64::from_le_bytes(n).max(packs)
    } else {
        packs
    };
    committed.saturating_add(retained_tmp_bytes(dir))
}

/// Root-wide spend: `current_used` for `current` plus `sibling_bytes` over
/// every other chunk dir under `parent` and `mapped_bytes` over every other
/// mapped dir (the current dir's twin included: its bytes are real spend,
/// only eviction treats it as current). Both inputs are expected to include
/// retained `.tmp` leftovers — `sibling_pack_estimate`/
/// `reconciled_sibling_pack_spend` already add them, and the writer passes
/// `used + retained_tmp_bytes` as `current_used`. Read errors count as 0 —
/// this is an advisory estimate, never a load failure.
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
                .filter(|p| p != current && is_real_dir(p) && dir_name_identity(p).is_some())
                .map(|p| {
                    if is_mapped_dir(&p) {
                        mapped_bytes(&p)
                    } else {
                        sibling_bytes(&p)
                    }
                })
                .sum()
        })
        .unwrap_or(0);
    current_used.saturating_add(siblings)
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

/// The chunk dir's writer lock: its `budget` ledger.
fn try_budget_lock(dir: &Path) -> Option<Option<File>> {
    try_exclusive_lock(&dir.join("budget"))
}

/// The mapped dir's lock file; every open `MappedPack` holds it SH.
fn try_mapped_lock(dir: &Path) -> Option<Option<File>> {
    try_exclusive_lock(&dir.join(MAPPED_LOCK))
}

/// `try_budget_lock` relative to a pinned dir fd: the ledger is opened
/// through `dirfd` so the flock lands on the inode `try_budget_lock`'s
/// path-open could no longer prove (a swapped `dir` would open some other
/// dir's ledger). Same returns: Some(Some(file)) holds the lock, Some(None)
/// means no ledger, None means a live writer or an error.
fn try_budget_lock_dirfd(dirfd: &File) -> Option<Option<File>> {
    let file = match dir_open_at(
        dirfd,
        OsStr::new("budget"),
        libc::O_RDWR | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        0o644,
    )
    .or_else(|_| dir_open_file(dirfd, "budget", false))
    {
        Ok(file) => file,
        Err(e) if e.kind() == ErrorKind::NotFound => return Some(None),
        Err(_) => return None,
    };
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return None;
    }
    // The ledger could have been unlinked and recreated between openat and
    // flock; only act on the inode we actually hold.
    let held = file.metadata().ok();
    let live = dir_stat(dirfd, OsStr::new("budget"));
    match (held, live) {
        (Some(a), Some(b)) if a.dev() == b.st_dev as u64 && a.ino() == b.st_ino => Some(Some(file)),
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
    // Lock BEFORE scanning: a writer that publishes between the scan and the
    // lock would leave the scan stale, and rewriting the ledger under the
    // now-free lock would erase its fresh reservation — the authoritative
    // recheck would then undercount real spend and over-admit. Scanning and
    // rewriting under the held flock keeps the ledger we write equal to what
    // is on disk at handoff; a busy lock means a live writer whose pending
    // reservation is trusted.
    match try_budget_lock(dir) {
        Some(guard) => {
            // The ledger records committed `.pack` bytes only — rewrite it
            // to exactly that. Retained `.tmp` leftovers were never
            // reservations: they inflate the returned spend estimate, not
            // the ledger.
            let real = pack_bytes(dir);
            if let Ok(mut ledger) = OpenOptions::new().write(true).open(dir.join("budget")) {
                let _ = ledger.write_all(&real.to_le_bytes());
                let _ = ledger.set_len(8);
            }
            let spend = real.saturating_add(retained_tmp_bytes(dir));
            drop(guard);
            spend
        }
        None => sibling_pack_estimate(dir),
    }
}

fn locked_remove(dir: &Path, reason: &str, still_stale: impl Fn(&Path) -> bool) -> u64 {
    // In-process exclusion first: a same-process reader's SH flock cannot
    // stop this process's EX request, so the flock alone would let us delete
    // a dir mid-read. The registry lock is held through remove_dir_all so a
    // reader racing in is refused registration and can only observe the
    // deletion — the window between the under-lock staleness recheck and
    // the unlink stays exclusive.
    let Some(_reader_exclusion) = packed_deletion_guard(dir) else {
        debug!(
            "packed cache: keeping {} (verified read in flight)",
            dir.display()
        );
        return 0;
    };
    let _guard = if is_mapped_dir(dir) {
        // A mapped dir has no creation window: `hold_mapped_dir` takes the
        // SH lock before the manifest is written, so a lock-free manifested
        // dir is one nobody is reading or building.
        let Some(guard) = try_mapped_lock(dir) else {
            debug!(
                "packed cache: keeping {} (mapped pack is open or being built)",
                dir.display()
            );
            return 0;
        };
        guard
    } else {
        let Some(guard) = try_budget_lock(dir) else {
            debug!(
                "packed cache: keeping {} (writer holds budget lock)",
                dir.display()
            );
            return 0;
        };
        // A dir without a budget ledger whose manifest still recomputes to
        // its dir-name identity sits in `new_at`'s creation window: the
        // manifest is written eagerly but `budget` only appears at the first
        // `write`. Deleting it would vanish a live load's cache root
        // mid-write. A stale or mismatched manifest means abandonment, not
        // creation — fall through to the caller's verdict.
        if guard.is_none()
            && is_real_dir(dir)
            && dir.join(".manifest").is_file()
            && identity_from_manifest(dir)
                .ok()
                .is_some_and(|identity| dir_name_identity(dir) == Some(identity.as_str()))
        {
            debug!(
                "packed cache: keeping {} (verified manifest without ledger is mid-creation)",
                dir.display()
            );
            return 0;
        }
        guard
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
    locked_remove(dir, reason, |d| {
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

pub(super) fn packed_total_cap() -> Option<u64> {
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

/// The mapped builder's root cap lock: a chunk writer's estimate→reserve
/// section lasts milliseconds, so wait up to a second for it instead of
/// losing a 0.5-1 GiB file build to a busy lock.
fn packed_root_cap_lock_wait(parent: &Path) -> Option<File> {
    for _ in 0..50 {
        if let Some(guard) = packed_root_cap_lock(parent) {
            return Some(guard);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    None
}

/// Admission for one mapped-pack file of `file_bytes` in the mapped `dir`:
/// the free-space reserve, and under a configured total cap the root-wide
/// spend after evicting idle siblings, under the root cap lock so a chunk
/// writer cannot admit against the same room. `Err` carries the reason.
pub(super) fn admit_mapped_file(
    dir: &Path,
    dirfd: &File,
    file_bytes: u64,
    cap: Option<u64>,
) -> Result<()> {
    let gib = file_bytes as f64 / (1u64 << 30) as f64;
    if !has_disk_space(dirfd, file_bytes) {
        return Err(err(format!(
            "Not enough disk space for {gib:.2} GiB plus the 16 GiB reserve"
        )));
    }
    let Some((cap, parent)) = cap.zip(dir.parent()) else {
        return Ok(());
    };
    let _root_guard = packed_root_cap_lock_wait(parent)
        .ok_or_else(|| err(format!("{TOTAL_CAP_ENV} admission lock stayed busy")))?;
    let own = mapped_bytes(dir);
    let fits = |spend: u64| spend.saturating_add(file_bytes) <= cap;
    if !fits(root_pack_spend(parent, dir, own, sibling_pack_estimate)) {
        enforce_packed_total_cap(parent, dir, cap.saturating_sub(file_bytes));
        if !fits(root_pack_spend(
            parent,
            dir,
            own,
            reconciled_sibling_pack_spend,
        )) {
            return Err(err(format!(
                "{TOTAL_CAP_ENV} leaves no room for a {gib:.2} GiB mapped file"
            )));
        }
    }
    Ok(())
}

/// Enforce the optional total cap: evict least-recently-used sibling dirs
/// of either format first (never `current` or its twin), then shed the
/// oldest `.pack` files inside a current chunk dir under its own ledger lock.
fn enforce_packed_total_cap(parent: &Path, current: &Path, cap: u64) {
    let mut dirs: Vec<(PathBuf, Option<SystemTime>)> = fs::read_dir(parent)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| is_real_dir(p) && dir_name_identity(p).is_some())
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
    for (dir, snapshot) in dirs.iter().filter(|(p, _)| !is_current(p, current)) {
        if total <= cap {
            break;
        }
        if !snapshot.is_some_and(|mtime| mtime < cutoff) {
            continue;
        }
        let removed = locked_remove(dir, "total cap LRU eviction", |d| {
            is_real_dir(d)
                && dir_name_identity(d).is_some()
                && d.join(".manifest").is_file()
                && packed_dir_mtime(d)
                    .is_some_and(|mtime| snapshot.is_some_and(|snap| mtime <= snap))
        });
        total = total.saturating_sub(removed);
        reclaimed += removed;
    }
    if total > cap && !is_mapped_dir(current) {
        // Shed oldest .pack files in the current dir under its own budget
        // lock. Both the lock and the unlinks anchor to a pinned dir fd: a
        // `current` path swapped for a symlink since its check cannot make
        // the ledger open or the deletions follow it outside the root.
        let dirfd = open_dir_nofollow(current);
        let guard = dirfd.as_ref().and_then(try_budget_lock_dirfd);
        let (Some(dirfd), Some(_guard)) = (dirfd, guard) else {
            if reclaimed > 0 {
                info!("packed cache: reclaimed {reclaimed} bytes under {TOTAL_CAP_ENV}");
            }
            return;
        };
        // Same exclusion as locked_remove: unlinking packs a verified read
        // has not opened yet forces a rebuild. A reader active right now
        // means this whole shed pass defers.
        let Some(_reader_exclusion) = packed_deletion_guard(current) else {
            if reclaimed > 0 {
                info!("packed cache: reclaimed {reclaimed} bytes under {TOTAL_CAP_ENV}");
            }
            return;
        };
        let names: Vec<OsString> = fs::read_dir(current)
            .map(|entries| entries.flatten().map(|e| e.file_name()).collect())
            .unwrap_or_default();
        let mut packs: Vec<(SystemTime, OsString, u64)> = names
            .iter()
            .filter(|name| is_pack_name(name.as_os_str()))
            .filter_map(|name| {
                let stat = dir_stat(&dirfd, name)?;
                Some((stat_mtime(&stat), name.clone(), stat.st_size as u64))
            })
            .collect();
        packs.sort_by_key(|(t, _, _)| *t);
        let mut remaining: u64 = packs.iter().map(|(_, _, len)| len).sum();
        for (_, name, len) in packs {
            if total <= cap {
                break;
            }
            if dir_unlinkat(&dirfd, &name).is_ok() {
                debug!(
                    "packed cache: shed {} (total cap)",
                    current.join(&name).display()
                );
                total = total.saturating_sub(len);
                remaining = remaining.saturating_sub(len);
                reclaimed += len;
            }
        }
        // Re-reconcile the ledger after removals — relative to the same
        // pinned fd so the bytes land on the inode this pass touched.
        let _ = dir_write_file(&dirfd, "budget", &remaining.to_le_bytes());
    }
    if reclaimed > 0 {
        info!("packed cache: reclaimed {reclaimed} bytes under {TOTAL_CAP_ENV}");
    }
}

/// Sweep sibling `.mlx-qwen4-packed-v2-*` and `.mlx-qwen4-mapped-v3-*` dirs
/// for orphans and enforce the optional total cap. Never fatal.
fn sweep_packed_cache(parent: &Path, current: &Path) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Ok(entries) = fs::read_dir(parent) else {
            return;
        };
        for entry in entries.flatten() {
            let dir = entry.path();
            if is_current(&dir, current) || !is_real_dir(&dir) || dir_name_identity(&dir).is_none()
            {
                continue;
            }
            if let Some(reason) = packed_orphan_reason(&dir) {
                remove_packed_dir(&dir, reason);
            }
        }
        // Crash-leftover `{uuid}.tmp` files carry no ledger bytes: drop them
        // once they age past the sweep grace (a live writer's tmp is fresh).
        // The fd pins the validated inode: age checks and unlinks stay
        // inside it even if the `current` path was swapped for a link.
        if let Some(dirfd) = open_dir_nofollow(current)
            && let Ok(entries) = fs::read_dir(current)
        {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let path = current.join(&name);
                if is_tmp_name(&name)
                    && let Some(stat) = dir_stat(&dirfd, &name)
                    && (stat.st_mode & libc::S_IFMT) == libc::S_IFREG
                    && stat_mtime(&stat) < SystemTime::now() - PACKED_SWEEP_MIN_AGE
                    && dir_unlinkat(&dirfd, &name).is_ok()
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

/// In-process reader registry, keyed by cache dir. `read_verified` bumps
/// its dir's count for the whole read; every deletion path (sweep/LRU
/// `locked_remove`, own-pack shedding) must hold this lock while it
/// unlinks — which atomically covers both "reader already counted" (skip
/// the delete) and "deletion already in flight" (the reader's try_lock
/// fails, it proceeds unguarded and observes the removal). Needed because
/// BSD flock is per-process: the budget ledger lock alone cannot exclude a
/// same-process reader.
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

/// Held through a verified read: a per-dir in-process count plus an SH
/// flock on the dir's budget ledger. Every deletion path must win
/// packed_deletion_guard AND an EX flock on this ledger, so a read already
/// in flight makes both sides fail — the dir cannot be removed out from
/// under an open. A read that loses the race proceeds unguarded and simply
/// observes the removal (miss or fd-survives), never corrupt data.
pub(super) struct ReadGuard {
    dir: Option<PathBuf>,
    _flock: Option<File>,
}

impl ReadGuard {
    /// A read is guarded only while BOTH exclusions hold: the in-process
    /// registry entry stops same-process deleters (flock is per-process) and
    /// the SH flock stops foreign ones. Either leg missing means a deleter
    /// can proceed concurrently — the caller must report a cache miss
    /// instead of reading unguarded, or an open could race remove_dir_all.
    fn live(&self) -> bool {
        self.dir.is_some() && self._flock.is_some()
    }

    /// Register `dir` in the reader registry and keep `flock` for the
    /// guard's lifetime. Readers BLOCK on the registry mutex — its critical
    /// sections are tiny (a count bump, or a whole remove_dir_all a deleter
    /// holds it through). A blocking lock still cannot straddle a delete:
    /// either we register before the deleter's try_lock (it sees the count
    /// and skips) or after it finishes (the dir is gone and the open
    /// misses). try_lock would be wrong here — a parallel reader's brief
    /// bump would fail the lock and spuriously degrade a healthy read to a
    /// miss.
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
/// load then keeps the chunk reader.
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

fn read_guard(root: &Path) -> ReadGuard {
    let flock = File::open(root.join("budget")).ok().and_then(|file| {
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) } != 0 {
            return None;
        }
        // The ledger could have been unlinked between open and flock; only a
        // lock on the inode the evictor will open excludes it.
        let held = file.metadata().ok();
        let live = fs::metadata(root.join("budget")).ok();
        match (held, live) {
            (Some(a), Some(b)) if a.dev() == b.dev() && a.ino() == b.ino() => Some(file),
            _ => None,
        }
    });
    ReadGuard::registered(root, flock)
}

/// Remove the oldest `.pack` files in `dir` (by mtime — the recency witness
/// stamped by verified reads) until the remaining pack bytes fit `headroom`,
/// or nothing is left to shed. Returns the real remaining pack bytes, i.e.
/// the new `used`. `exclude` protects the write's own target. Callers hold
/// the dir's budget lock, so no concurrent writer can shed underneath us and
/// `enforce_packed_total_cap`'s own-packs phase cannot (it never wins our
/// flock) — this is the only place self-shedding can happen during a write.
/// `dirfd` pins `dir`'s validated inode: sizing and unlinks run relative to
/// it, so a `dir` path swapped for a symlink between the open and this call
/// sheds nothing outside the cache root.
fn shed_oldest_packs(dir: &Path, dirfd: &File, exclude: &OsStr, headroom: u64) -> u64 {
    let names: Vec<OsString> = fs::read_dir(dir)
        .map(|entries| entries.flatten().map(|e| e.file_name()).collect())
        .unwrap_or_default();
    let mut packs: Vec<(SystemTime, OsString, u64)> = names
        .iter()
        .filter(|name| is_pack_name(name.as_os_str()) && name.as_os_str() != exclude)
        .filter_map(|name| {
            let stat = dir_stat(dirfd, name)?;
            Some((stat_mtime(&stat), name.clone(), stat.st_size as u64))
        })
        .collect();
    packs.sort_by_key(|(mtime, _, _)| *mtime);
    let mut used: u64 = packs.iter().map(|(_, _, len)| len).sum();
    for (_, name, len) in packs {
        if used <= headroom {
            break;
        }
        if dir_unlinkat(dirfd, &name).is_ok() {
            debug!(
                "packed cache: shed {} to admit a new write",
                dir.join(&name).display()
            );
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
        // harmless: manifest-less dirs are never orphan-deleted. The dir
        // must be REAL: `create_dir_all` accepts a symlink to an external
        // directory, which would route the manifest and every later .pack/
        // budget/.tmp write through the link — outside the root, invisible
        // to the sweep and the total cap — so a planted link is unlinked
        // first (the external target is never touched) and the manifest is
        // published only when the dir verifies as real (covers a link
        // re-planted between unlink and create). The write goes through a
        // dir fd opened O_DIRECTORY|O_NOFOLLOW: the validated inode stays
        // pinned for the write, so a link swapped in between the check and
        // the open still cannot redirect it outside the root.
        let manifest = pack_manifest(&canonical);
        unlink_if_symlink(&dir);
        if fs::create_dir_all(&dir).is_ok()
            && let Some(dirfd) = open_dir_nofollow(&dir)
        {
            let _ = dir_write_file(&dirfd, ".manifest", &manifest);
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

    /// The source fingerprint this identity dir was named for; the mapped
    /// pack reuses it so both formats of one checkpoint share the suffix.
    pub(super) fn identity(&self) -> Option<&str> {
        dir_name_identity(&self.root)
    }

    /// This checkpoint's mapped pack dir beside the chunk dir.
    pub(super) fn mapped_dir(&self) -> Option<PathBuf> {
        twin_dir(&self.root)
    }

    pub(super) fn manifest(&self) -> Option<&[u8]> {
        self.manifest.as_deref()
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
        // One guard for the whole batch: per-chunk guards would let a
        // writer's shed_oldest_packs interleave between two chunk opens and
        // delete a pack this batch has not reached yet. A non-live guard
        // means a deleter can proceed concurrently right now — report
        // all-miss rather than racing remove_dir_all between chunks.
        let _batch_guard = read_guard(&self.root);
        if !_batch_guard.live() {
            return Ok(requests.iter().map(|_| None).collect());
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

    pub(super) fn read_verified(
        &self,
        key: &str,
        payload_limit: u64,
    ) -> Result<Option<VerifiedChunk>> {
        // Deletion exclusion for the whole read: an mtime stamp alone is an
        // unlocked witness — an evictor can pass its under-lock staleness
        // recheck before the stamp lands and still remove_dir_all. The guard
        // is taken BEFORE the open so deletion is atomic against the read:
        // either the evictor won its locks first (the read may observe the
        // removal and report a miss), or this guard makes both of its locks
        // fail and the dir survives until the read finishes.
        let _guard = read_guard(&self.root);
        if !_guard.live() {
            // Contention = miss: the dir may already be inside a foreign or
            // same-process removal. Failing the open path as NotFound gives
            // the caller a clean rebuild instead of a mid-read delete.
            return Ok(None);
        }
        let mut file = match File::open(self.path(key)) {
            Ok(f) => f,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(err(e)),
        };
        // Stamp at read start, not only at verified-hit: the idle-expiry
        // snapshot taken before the evictor's lock must also see this read
        // as recent, and freshness matters for LRU order too.
        stamp_mtime(&self.path(key));
        stamp_mtime(&self.root.join("budget"));
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
        // new_at verified the dir is real, but a link could be swapped in
        // after construction: unlink one found here, then recheck — a dir
        // that is still not real after creation (a hostile occupant) makes
        // the write skip rather than follow a link outside the cache root.
        // The check alone cannot pin what it validated, so the dir is
        // opened once O_DIRECTORY|O_NOFOLLOW and every open/write/rename/
        // unlink below runs relative to the fd: a link planted after this
        // point resolves nothing, and a dir deleted after the open just
        // lands the bytes in an unlinked inode the OS reclaims on close.
        unlink_if_symlink(&self.root);
        fs::create_dir_all(&self.root).map_err(err)?;
        let Some(dirfd) = open_dir_nofollow(&self.root) else {
            debug!(
                "packed cache: skipping write, {} is not a real directory",
                self.root.display()
            );
            return Ok(());
        };
        // A sweep may have deleted the .manifest (or the whole dir) between
        // new_at and this write; republish so the recreated dir stays
        // verifiable instead of becoming an unaccountable manifest-less dir.
        // A symlink at `.manifest` is not a manifest, and O_NOFOLLOW keeps
        // the republish from writing through it.
        if let Some(manifest) = &self.manifest
            && dir_stat(&dirfd, OsStr::new(".manifest"))
                .is_none_or(|stat| (stat.st_mode & libc::S_IFMT) != libc::S_IFREG)
        {
            let _ = dir_write_file(&dirfd, ".manifest", manifest);
        }
        // Serialize outside the lock so the ledger can reserve exact bytes.
        let (arrays, data) = encode_weight(weight)?;
        let header = Header {
            key: key.into(),
            group: weight.group,
            bits: weight.bits,
            mode: weight.mode.clone(),
            arrays,
        };
        let header = serde_json::to_vec(&header).map_err(err)?;
        if header.len() > MAX_HEADER || data.len() as u64 > MAX_READ_BYTES {
            return Err(err("Packed chunk exceeds staging budget"));
        }
        let reserve = 44 + header.len() as u64 + data.len() as u64;
        // Serialize writers across processes. A busy writer never stalls inference.
        let _in_process = writer_mutex()
            .lock()
            .map_err(|_| err("Packed writer mutex poisoned"))?;
        let mut ledger = dir_open_file(&dirfd, "budget", true).map_err(err)?;
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
        let target_name = format!("{key}.pack");
        let target_len = dir_stat(&dirfd, OsStr::new(&target_name))
            .map(|stat| stat.st_size as u64)
            .unwrap_or(0);
        let mut used: u64 = fs::read_dir(&self.root)
            .map_err(err)?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .filter(|name| {
                is_pack_name(name.as_os_str()) && name.as_os_str() != OsStr::new(&target_name)
            })
            .try_fold(0u64, |sum, name| {
                let len = dir_stat(&dirfd, &name)
                    .map(|stat| stat.st_size as u64)
                    .ok_or_else(|| err("Packed cache entry vanished"))?;
                sum.checked_add(len)
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
        // fstatvfs on the pinned fd: a swapped path could point at a volume
        // with different space than the inode the write will land in.
        if !has_disk_space(&dirfd, reserve) {
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
            // Retained `.tmp` leftovers spend cap room exactly like packs —
            // a crashed writer's staged bytes sit on disk for the whole
            // sweep grace — so every root estimate adds them on top of the
            // pack-only `used`/ledger values.
            let tmp = retained_tmp_bytes(&self.root);
            // The fast path estimates siblings from their ledgers (a full
            // byte scan per tensor write would be too hot); only a plausible
            // overspend pays the real enforcement sweep.
            let estimate = root_pack_spend(
                parent,
                &self.root,
                used.saturating_add(tmp),
                sibling_pack_estimate,
            );
            if estimate.saturating_add(reserve) > cap || used.saturating_add(reserve) > budget {
                // Evict LRU siblings down to `cap - reserve` — the target is
                // the room this write needs, so admission can roll instead of
                // wedging once aggregate usage reaches the cap.
                enforce_packed_total_cap(parent, &self.root, cap - reserve);
                // Authoritative recheck: reconcile each unheld sibling ledger
                // to real pack bytes (a crashed reservation is phantom spend
                // that would otherwise veto forever); a held budget lock
                // means a live writer whose pending reservation is trusted.
                let reconciled = root_pack_spend(
                    parent,
                    &self.root,
                    used.saturating_add(tmp),
                    reconciled_sibling_pack_spend,
                );
                let siblings = reconciled.saturating_sub(used.saturating_add(tmp));
                // Shed our own oldest packs when self-shedding can admit the
                // write: enforcement could not (it never wins the budget
                // flock we hold). `headroom` bounds both the per-dir budget
                // and the global cap net of sibling spend.
                if used.saturating_add(reserve) > budget || reconciled.saturating_add(reserve) > cap
                {
                    // Shed only when no verified read is in flight; the
                    // budget flock we hold cannot exclude a same-process
                    // reader, so this needs the in-process guard too.
                    if let Some(_reader_exclusion) = packed_deletion_guard(&self.root) {
                        let headroom = budget
                            .min(cap.saturating_sub(siblings).saturating_sub(tmp))
                            .saturating_sub(reserve);
                        used = shed_oldest_packs(
                            &self.root,
                            &dirfd,
                            OsStr::new(&target_name),
                            headroom,
                        );
                    }
                }
                // `.tmp` bytes shed with nothing — they stay spend until the
                // sweep ages them out.
                let actual = siblings.saturating_add(used).saturating_add(tmp);
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
        let temp_name = format!("{}.tmp", uuid::Uuid::new_v4());
        let result = (|| {
            let mut f = dir_create_new_file(&dirfd, &temp_name).map_err(err)?;
            f.write_all(MAGIC).map_err(err)?;
            f.write_all(&(header.len() as u32).to_le_bytes())
                .map_err(err)?;
            f.write_all(&digest).map_err(err)?;
            f.write_all(&header).map_err(err)?;
            f.write_all(&data).map_err(err)?;
            drop(f);
            dir_renameat(&dirfd, &temp_name, &target_name).map_err(err)
        })();
        let _ = dir_unlinkat(&dirfd, OsStr::new(&temp_name));
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
/// and one concatenated little-endian payload — the `.pack` body and the
/// per-expert record of the mapped pack share this encoding.
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

    /// Grammar-valid pack STEM (see `is_pack_name`): 64 lowercase hex —
    /// use as a `write`/`path` key or append `.pack` for a file name.
    fn pack_name(letter: char) -> String {
        letter.to_string().repeat(64)
    }

    /// Grammar-valid pack FILE name: `pack_name` plus the extension.
    fn pack_file(letter: char) -> String {
        format!("{}.pack", pack_name(letter))
    }

    /// Grammar-valid staging name (see `is_tmp_name`): a v4 UUID —
    /// version nibble `4` at index 14, variant `8` at index 19.
    fn tmp_name(seed: &str) -> String {
        format!("{seed}-0000-4000-8000-000000000000.tmp")
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
            format!(
                "path_hex={}",
                String::from_utf8(packed_hex_encode(
                    canonical.parent().unwrap().as_os_str().as_bytes()
                ))
                .unwrap()
            ),
            "the authoritative hex record leads the manifest:\n{manifest}"
        );
        assert!(
            manifest.lines().any(|line| line == "file=source.gguf"),
            "the raw file record follows its hex twin:\n{manifest}"
        );
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
        let manifest = fs::read_to_string(cache.root.join(".manifest")).unwrap();
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
        assert_eq!(
            identity_from_manifest(&cache.root).unwrap(),
            dir_name_identity(&cache.root).unwrap(),
            "the real file still verifies through its hex twin"
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
        fs::write(unknown.join(pack_file('9')), b"data").unwrap();
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();
        assert!(!stale.exists());
        assert!(unknown.exists());
        assert!(cache.root.join(".manifest").is_file());
    }

    #[test]
    fn idle_but_identity_matched_dir_is_never_an_orphan() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();
        // Age every witness past the sweep grace: the dir is as idle as a
        // cache can be, yet its manifest still recomputes to the dir-name
        // identity — a healthy-but-idle model cache, not an orphan.
        age(&cache.root.join(".manifest"), 7200);
        age_dir(&cache.root, 7200);
        assert!(
            packed_dir_idle(&cache.root),
            "the fixture must read as idle for this regression to bite"
        );
        assert_eq!(packed_orphan_reason(&cache.root), None);
        // And the full sweep must leave it alone.
        sweep_packed_cache(&parent, &temp.0.join("nonexistent-current"));
        assert!(cache.root.exists());
        assert!(cache.root.join(".manifest").is_file());
    }

    #[test]
    fn manifested_dir_without_ledger_is_mid_creation_and_survives_removal() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();
        // new_at writes .manifest long before the first write creates the
        // budget ledger; that window is creation, not abandonment. With no
        // ledger nothing can hold the lock, so without the mid-creation gate
        // locked_remove would delete it.
        assert!(cache.root.join(".manifest").is_file());
        assert!(!cache.root.join("budget").exists());
        // An always-stale verdict isolates the ledger gate: without the
        // mid-creation check locked_remove would delete this dir.
        assert_eq!(
            locked_remove(&cache.root, "test stale", |_| true),
            0,
            "a manifested dir with no budget ledger is mid-creation"
        );
        assert!(cache.root.exists());
        // Once a ledger exists the same verdict removes it.
        fs::write(cache.root.join("budget"), 0u64.to_le_bytes()).unwrap();
        assert!(locked_remove(&cache.root, "test stale", |_| true) > 0);
        assert!(!cache.root.exists());
    }

    #[test]
    fn symlinked_entries_never_inflate_dir_or_pack_bytes() {
        let temp = Temp::new();
        let dir = temp.0.join("cache");
        fs::create_dir(&dir).unwrap();
        // A large file outside the cache dir, linked in as a .pack and as a
        // regular entry, plus a symlinked directory: targets must never be
        // counted.
        let outside = temp.0.join("outside");
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("big.bin"), vec![0_u8; 1 << 20]).unwrap();
        std::os::unix::fs::symlink(outside.join("big.bin"), dir.join(pack_file('e'))).unwrap();
        std::os::unix::fs::symlink(outside.join("big.bin"), dir.join("linked.bin")).unwrap();
        std::os::unix::fs::symlink(&outside, dir.join("linked-dir")).unwrap();
        fs::write(dir.join(pack_file('f')), vec![0_u8; 64]).unwrap();
        let link_len = |name: &str| fs::symlink_metadata(dir.join(name)).unwrap().len();
        // A symlink contributes only its own link bytes — never the target's.
        assert_eq!(pack_bytes(&dir), 64 + link_len(&pack_file('e')));
        assert_eq!(
            dir_size(&dir),
            64 + link_len(&pack_file('e')) + link_len("linked.bin") + link_len("linked-dir")
        );
        assert!(
            pack_bytes(&dir) < 1 << 16,
            "pack_bytes followed a symlink: {}",
            pack_bytes(&dir)
        );
    }

    #[test]
    fn symlinked_identity_dir_is_foreign_to_sweep_and_spend() {
        let temp = Temp::new();
        let parent = temp.0.join("root");
        fs::create_dir(&parent).unwrap();
        // Bytes outside the cache root must never be counted or deleted.
        let outside = temp.0.join("outside");
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("big.pack"), vec![0_u8; 1 << 20]).unwrap();
        // A root-level entry whose NAME claims an identity but whose inode is
        // a symlink: foreign — invisible to sizing and immune to the sweep.
        let link = parent.join(format!("{DIR_PREFIX}{}", "a".repeat(64)));
        std::os::unix::fs::symlink(&outside, &link).unwrap();
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();
        assert!(link.is_symlink(), "the sweep must never touch a link");
        assert_eq!(
            fs::canonicalize(&link).unwrap(),
            outside.canonicalize().unwrap(),
            "the link target is intact"
        );
        // Sizing ignores the link's target bytes entirely.
        assert_eq!(root_pack_spend(&parent, &cache.root, 0, pack_bytes), 0);
        assert_eq!(reconciled_sibling_pack_spend(&link), 0);
        assert_eq!(sibling_pack_estimate(&link), 0);
    }

    /// When the cache's OWN identity name is occupied by a symlink,
    /// `new_at` must unlink it and create a real dir. `create_dir_all`
    /// accepts a link-to-dir, which would land the manifest and every later
    /// write in the external target while the sweep and the cap count the
    /// link itself as foreign — an escaped, uncounted, unswept cache.
    #[test]
    fn packed_cache_new_at_replaces_symlinked_identity_dir() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        fs::create_dir(&parent).unwrap();
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        // The real identity dir for `source` is unlinked and replaced with a
        // symlink whose target carries a marker; construction must admit no
        // bytes outside the root.
        let identity = packed_identity(BTreeSet::from([source.clone()]).into_iter()).unwrap();
        let expected = parent.join(format!("{DIR_PREFIX}{identity}"));
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source.clone()])).unwrap();
        assert_eq!(cache.root, expected);
        fs::remove_dir_all(&cache.root).unwrap();
        let outside = temp.0.join("outside");
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("marker"), b"keep").unwrap();
        std::os::unix::fs::symlink(&outside, &cache.root).unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();
        assert!(
            is_real_dir(&cache.root),
            "the link must be replaced by a real dir"
        );
        assert_ne!(
            fs::canonicalize(&cache.root).unwrap(),
            outside.canonicalize().unwrap(),
            "the real dir must not resolve to the external target"
        );
        assert!(cache.root.join(".manifest").is_file());
        cache.write("key", &weight("affine")).unwrap();
        assert!(cache.path("key").is_file());
        // The external target is untouched: the marker survives and nothing
        // the cache wrote landed there.
        assert_eq!(fs::read(outside.join("marker")).unwrap(), b"keep");
        assert!(!outside.join(".manifest").exists());
        assert!(!outside.join("key.pack").exists());
    }

    /// `new_at` validates the identity dir by path, but a path check cannot
    /// pin the inode: rename the real dir aside and plant a symlink to an
    /// external dir at `cache.root`, and every `dir.join(name)` write would
    /// follow it outside the cache root. `write` re-validates by opening the
    /// dir O_DIRECTORY|O_NOFOLLOW and then does ALL child opens/writes/
    /// renames/unlinks relative to that fd — the swap either keeps the fd's
    /// inode (writes land in the renamed-aside real dir) or makes the open
    /// fail (write skips, and the replanted link is unlinked and replaced by
    /// a fresh real dir first). In every case the external target stays
    /// byte-for-byte untouched.
    #[test]
    fn packed_cache_writes_stay_in_the_validated_inode() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        fs::create_dir(&parent).unwrap();
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();

        // Swap the validated dir for a link: real dir aside, link planted.
        let aside = temp.0.join("renamed-aside");
        fs::rename(&cache.root, &aside).unwrap();
        let outside = temp.0.join("outside");
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("marker"), b"keep").unwrap();
        std::os::unix::fs::symlink(&outside, &cache.root).unwrap();

        cache.write("key", &weight("affine")).unwrap();

        // The external target absorbed nothing: no republished manifest, no
        // budget ledger, no staging temp, no pack.
        assert_eq!(fs::read(outside.join("marker")).unwrap(), b"keep");
        let names: Vec<_> = fs::read_dir(&outside)
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert_eq!(
            names,
            vec![OsString::from("marker")],
            "descriptor-relative writes must not reach the link target"
        );
        // The write either recreated a real dir at the path (link unlinked)
        // or landed in the pinned aside inode — never through the link.
        if is_real_dir(&cache.root) {
            assert!(cache.path("key").is_file());
        } else {
            assert!(aside.join("key.pack").is_file());
        }
        assert!(aside.join(".manifest").is_file());
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

    /// A mapped dir fixture: `<letter>*64` identity, one `x.mapped` of
    /// `bytes`, a manifest whose source is gone, every witness aged past the
    /// sweep grace when `aged`.
    fn mapped_fixture(parent: &Path, letter: char, bytes: usize, aged: bool) -> PathBuf {
        let dir = parent.join(format!(
            "{MAPPED_DIR_PREFIX}{}",
            letter.to_string().repeat(64)
        ));
        fs::create_dir(&dir).unwrap();
        fs::write(dir.join("x.mapped"), vec![0_u8; bytes]).unwrap();
        fs::write(dir.join(".manifest"), "path=/gone\nfile=x.gguf\n").unwrap();
        fs::write(dir.join(MAPPED_LOCK), b"").unwrap();
        if aged {
            for entry in ["x.mapped", ".manifest", MAPPED_LOCK] {
                age(&dir.join(entry), 7200);
            }
            age_dir(&dir, 7200);
        }
        dir
    }

    #[test]
    fn stale_mapped_dir_is_swept_and_the_current_identity_twin_is_kept() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        fs::create_dir(&parent).unwrap();
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source.clone()])).unwrap();
        // The current identity's mapped dir, with the same orphan-looking
        // manifest and age as the stale sibling: only `is_current` keeps it.
        let twin = cache.mapped_dir().unwrap();
        assert_eq!(twin_dir(&twin).as_deref(), Some(cache.root.as_path()));
        fs::create_dir(&twin).unwrap();
        fs::write(twin.join("x.mapped"), vec![0_u8; 100]).unwrap();
        fs::write(twin.join(".manifest"), "path=/gone\nfile=x.gguf\n").unwrap();
        age(&twin.join("x.mapped"), 7200);
        age(&twin.join(".manifest"), 7200);
        age_dir(&twin, 7200);
        let stale = mapped_fixture(&parent, 'a', 100, true);
        assert_eq!(
            packed_orphan_reason(&stale),
            Some("unreadable manifest or missing source")
        );
        // A mapped dir whose name is not a 64-hex identity is foreign.
        let foreign = parent.join(format!("{MAPPED_DIR_PREFIX}backup"));
        fs::create_dir(&foreign).unwrap();
        fs::write(foreign.join(".manifest"), "path=/gone\nfile=x.gguf\n").unwrap();
        age(&foreign.join(".manifest"), 7200);
        age_dir(&foreign, 7200);
        let _cache = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();
        assert!(
            !stale.exists(),
            "an aged mapped dir with a dead source is an orphan"
        );
        assert!(
            twin.exists(),
            "the current identity's mapped dir is current"
        );
        assert!(foreign.exists());
        assert!(cache.root.exists());
    }

    #[test]
    fn total_cap_counts_mapped_bytes_and_evicts_idle_mapped_dirs_but_never_the_current_twin() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        fs::create_dir(&parent).unwrap();
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();
        let twin = cache.mapped_dir().unwrap();
        fs::create_dir(&twin).unwrap();
        fs::write(twin.join("x.mapped"), vec![0_u8; 100]).unwrap();
        fs::write(twin.join(".blk.1.tmp"), vec![0_u8; 50]).unwrap();
        fs::write(twin.join(".manifest"), "path=/gone\nfile=x.gguf\n").unwrap();
        fs::write(twin.join(MAPPED_LOCK), b"").unwrap();
        for entry in ["x.mapped", ".blk.1.tmp", ".manifest", MAPPED_LOCK] {
            age(&twin.join(entry), 7200);
        }
        age_dir(&twin, 7200);
        let idle = mapped_fixture(&parent, 'a', 100, true);
        let fresh = mapped_fixture(&parent, 'b', 100, false);
        // Mapped files and staging files are spend; manifests and locks not.
        assert_eq!(mapped_bytes(&twin), 150);
        assert_eq!(
            root_pack_spend(&parent, &cache.root, 0, sibling_pack_estimate),
            350
        );
        assert_eq!(
            root_pack_spend(&parent, &twin, mapped_bytes(&twin), sibling_pack_estimate),
            350,
            "seen from the mapped dir the chunk twin is an empty sibling"
        );
        // Over the cap: the idle sibling is the only eviction — the fresh
        // one is inside the grace and the aged twin is current.
        enforce_packed_total_cap(&parent, &cache.root, 10);
        assert!(!idle.exists());
        assert!(fresh.exists());
        assert!(twin.exists());
        assert!(
            twin.join("x.mapped").is_file(),
            "a mapped dir is never shed file by file"
        );
    }

    #[test]
    fn held_mapped_dir_lock_blocks_sweep_and_cap_eviction() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        fs::create_dir(&parent).unwrap();
        let current = parent.join(format!("{DIR_PREFIX}{}", "c".repeat(64)));
        fs::create_dir(&current).unwrap();
        // Another process reads the pack: its SH lock keeps the orphan.
        let stale = mapped_fixture(&parent, 'a', 100, true);
        let acquired = temp.0.join("child.acquired");
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("models::qwen4_exp::packed_cache::tests::packed_budget_lock_child")
            .arg("--ignored")
            .env("MLX_NODE_PACKED_LOCK_TEST_PATH", stale.join(MAPPED_LOCK))
            .env("MLX_NODE_PACKED_LOCK_TEST_ACQUIRED", &acquired)
            .env("MLX_NODE_PACKED_LOCK_TEST_SHARED", "1")
            .spawn()
            .unwrap();
        for _ in 0..500 {
            if acquired.is_file() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(acquired.is_file(), "child never acquired the mapped lock");
        sweep_packed_cache(&parent, &current);
        assert!(stale.exists(), "an open pack is not swept");
        enforce_packed_total_cap(&parent, &current, 0);
        assert!(stale.exists(), "an open pack is not cap-evicted");
        child.kill().unwrap();
        let _ = child.wait();
        sweep_packed_cache(&parent, &current);
        assert!(!stale.exists());
        // The same process holding the dir through `hold_mapped_dir` (a
        // loaded MappedPack) is excluded too, by the registry and the lock.
        let held = mapped_fixture(&parent, 'b', 100, true);
        let dirfd = open_dir_nofollow(&held).unwrap();
        let guard = hold_mapped_dir(&held, &dirfd).unwrap();
        assert!(guard.live());
        assert_eq!(locked_remove(&held, "test", |_| true), 0);
        assert!(held.exists());
        drop(guard);
        assert!(locked_remove(&held, "test", |_| true) > 0);
        assert!(!held.exists());
        // Holding a dir an evictor removed meanwhile is refused.
        let gone = mapped_fixture(&parent, 'd', 100, true);
        let dirfd = open_dir_nofollow(&gone).unwrap();
        fs::remove_dir_all(&gone).unwrap();
        assert!(hold_mapped_dir(&gone, &dirfd).is_err());
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

    // An in-flight verified read holds packed_readers shared; BSD flock is
    // per-process so the budget lock alone cannot exclude same-process
    // readers — locked_remove must defer while a read guard is live.
    #[test]
    fn locked_remove_defers_while_a_verified_read_is_in_flight() {
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
        assert!(packed_orphan_reason(&stale).is_some());
        let guard = read_guard(&stale);
        assert_eq!(locked_remove(&stale, "test", |_| true), 0);
        assert!(stale.exists());
        drop(guard);
        assert!(locked_remove(&stale, "test", |_| true) > 0);
        assert!(!stale.exists());
    }

    // Names under the prefix that don't match the 64-hex producer grammar
    // are foreign: an aged `.mlx-qwen4-packed-v2-backup` with a malformed
    // manifest must survive the sweep, and a pack stamped before the epoch
    // must order as oldest rather than panic.
    #[test]
    fn sweep_leaves_non_identity_prefixed_dirs_and_handles_pre_epoch_packs() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        fs::create_dir(&parent).unwrap();
        let foreign = parent.join(format!("{DIR_PREFIX}backup"));
        fs::create_dir(&foreign).unwrap();
        fs::write(foreign.join(".manifest"), "not a manifest\n").unwrap();
        age(&foreign.join(".manifest"), 7200);
        age_dir(&foreign, 7200);
        sweep_packed_cache(&parent, &parent.join("current"));
        assert!(foreign.exists());
        // st_mtime before the epoch used to wrap into a panic-inducing
        // duration; it now orders as the oldest entry.
        let cache_dir = parent.join(format!("{DIR_PREFIX}{}", "d".repeat(64)));
        fs::create_dir(&cache_dir).unwrap();
        let pack = cache_dir.join(pack_file('0'));
        fs::write(&pack, b"x").unwrap();
        let epoch_minus_one = SystemTime::UNIX_EPOCH - Duration::from_secs(1);
        OpenOptions::new()
            .write(true)
            .open(&pack)
            .unwrap()
            .set_modified(epoch_minus_one)
            .unwrap();
        let stat = dir_stat(
            &open_dir_nofollow(&cache_dir).unwrap(),
            OsStr::new(&pack_file('0')),
        )
        .unwrap();
        assert_eq!(stat_mtime(&stat), SystemTime::UNIX_EPOCH);
    }

    #[test]
    fn ledger_reconciles_externally_deleted_packs() {
        let temp = Temp::new();
        let cache = PackedCache {
            root: temp.0.join("cache"),
            manifest: None,
        };
        let w = weight("affine");
        cache.write(&pack_name('a'), &w).unwrap();
        cache.write(&pack_name('b'), &w).unwrap();
        // Simulated external cleanup inflates the recorded spend.
        fs::remove_file(cache.path(&pack_name('a'))).unwrap();
        cache.write(&pack_name('c'), &w).unwrap();
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
        cache.write(&pack_name('d'), &w).unwrap();
        File::open(cache.root.join("budget"))
            .unwrap()
            .read_exact(&mut n)
            .unwrap();
        assert_eq!(
            u64::from_le_bytes(n),
            fs::metadata(cache.path(&pack_name('d'))).unwrap().len()
                + fs::metadata(cache.path(&pack_name('b'))).unwrap().len()
                + fs::metadata(cache.path(&pack_name('c'))).unwrap().len()
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
        for (key, seconds) in [
            (pack_name('a'), 300u64),
            (pack_name('b'), 200),
            (pack_name('c'), 100),
        ] {
            cache.write(&key, &w).unwrap();
            age(&cache.path(&key), seconds);
        }
        let pack = fs::metadata(cache.path(&pack_name('a'))).unwrap().len();
        // Cap fits exactly two packs plus ledger bytes: shed only the oldest.
        enforce_packed_total_cap(&temp.0, &cache.root, 2 * pack + 64);
        assert!(!cache.path(&pack_name('a')).exists());
        assert!(cache.path(&pack_name('b')).exists());
        assert!(cache.path(&pack_name('c')).exists());
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
        let stale_tmp = cache.root.join(tmp_name("00000000"));
        let live_tmp = cache.root.join(tmp_name("11111111"));
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
        fs::write(oldest.join(pack_file('a')), vec![0_u8; 100]).unwrap();
        fs::write(oldest.join(".manifest"), "path=/gone\nfile=x.gguf\n").unwrap();
        age(&oldest.join(pack_file('a')), 7200);
        age(&oldest.join(".manifest"), 7200);
        age_dir(&oldest, 7200);
        // Fresh manifest dir and manifest-less dir both survive the cap: the
        // idle grace shields the first, the manifest requirement the second.
        let fresh = parent.join(format!("{DIR_PREFIX}{}", "b".repeat(64)));
        fs::create_dir(&fresh).unwrap();
        fs::write(fresh.join(pack_file('b')), vec![0_u8; 100]).unwrap();
        fs::write(fresh.join(".manifest"), "path=/gone\nfile=x.gguf\n").unwrap();
        let bare = parent.join(format!("{DIR_PREFIX}{}", "c".repeat(64)));
        fs::create_dir(&bare).unwrap();
        fs::write(bare.join(pack_file('c')), vec![0_u8; 100]).unwrap();
        age(&bare.join(pack_file('c')), 7200);
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
    fn read_stamps_witnesses_before_verification_completes() {
        let temp = Temp::new();
        let parent = temp.0.join("cache");
        let source = temp.0.join("source.gguf");
        fs::write(&source, b"weights").unwrap();
        let cache = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();
        cache.write("key", &weight("affine")).unwrap();
        let mut corrupt = fs::read(cache.path("key")).unwrap();
        *corrupt.last_mut().unwrap() ^= 1;
        fs::write(cache.path("key"), corrupt).unwrap();
        age(&cache.path("key"), 7200);
        age(&cache.root.join("budget"), 7200);
        age(&cache.root.join(".manifest"), 7200);
        age_dir(&cache.root, 7200);
        assert!(packed_dir_idle(&cache.root));
        // Even though verification fails, the open already refreshed the
        // witnesses: an evictor racing this read sees a live dir.
        assert!(cache.read("key").is_err());
        assert!(!packed_dir_idle(&cache.root));
    }

    // A read whose guard cannot take BOTH exclusions reports a miss instead
    // of racing remove_dir_all: here the ledger is absent, so the flock leg
    // has nothing to pin and the guard is not live.
    #[test]
    fn read_reports_miss_when_exclusion_is_incomplete() {
        let temp = Temp::new();
        let cache = PackedCache {
            root: temp.0.join("cache"),
            manifest: None,
        };
        cache.write(&pack_name('a'), &weight("affine")).unwrap();
        fs::remove_file(cache.root.join("budget")).unwrap();
        assert!(cache.read(&pack_name('a')).unwrap().is_none());
        // With the ledger restored both legs hold and reads hit again.
        fs::write(cache.root.join("budget"), 0u64.to_le_bytes()).unwrap();
        assert!(cache.read(&pack_name('a')).unwrap().is_some());
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
        a.write_with_total_cap(&pack_name('e'), &w, Some(u64::MAX))
            .unwrap();
        let pack = fs::metadata(a.path(&pack_name('e'))).unwrap().len();
        assert!(pack > 0);
        // A sibling ledger reservation counts toward the shared cap: b's
        // write is vetoed even though its own dir is empty, and no eviction
        // can free it (a's dir is fresh).
        b.write_with_total_cap(&pack_name('e'), &w, Some(pack))
            .unwrap();
        assert!(!b.path(&pack_name('e')).exists());
        // A busy root cap lock skips the chunk instead of stalling; the next
        // write after the guard drops succeeds.
        let guard = packed_root_cap_lock(&parent).unwrap();
        a.write_with_total_cap(&pack_name('f'), &w, Some(u64::MAX))
            .unwrap();
        assert!(!a.path(&pack_name('f')).exists());
        drop(guard);
        a.write_with_total_cap(&pack_name('f'), &w, Some(u64::MAX))
            .unwrap();
        assert!(a.path(&pack_name('f')).exists());
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
        for (key, seconds) in [
            (pack_name('a'), 300u64),
            (pack_name('b'), 200),
            (pack_name('c'), 100),
        ] {
            cache.write(&key, &w).unwrap();
            age(&cache.path(&key), seconds);
        }
        let pack = fs::metadata(cache.path(&pack_name('a'))).unwrap().len();
        // A cap that fits exactly two packs plus the pending write: the
        // writer must shed its own oldest pack (enforcement cannot — it
        // never wins the budget flock the writer holds), turning the dir
        // into a true LRU instead of wedging once usage reaches the cap.
        // (pack names must satisfy is_pack_name: 64 lowercase hex.)
        cache
            .write_with_total_cap(&pack_name('d'), &w, Some(2 * pack + pack))
            .unwrap();
        assert!(
            !cache.path(&pack_name('a')).exists(),
            "oldest pack is shed for room"
        );
        assert!(cache.path(&pack_name('b')).exists());
        assert!(cache.path(&pack_name('c')).exists());
        assert!(
            cache.path(&pack_name('d')).exists(),
            "the new write must be admitted"
        );
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
        fs::write(sibling.join(pack_file('a')), vec![0_u8; 64]).unwrap();
        let ledger_path = sibling.join("budget");
        fs::write(&ledger_path, u64::MAX.to_le_bytes()).unwrap();
        let source = temp.0.join("b.gguf");
        fs::write(&source, b"second").unwrap();
        let b = PackedCache::new_at(&parent, BTreeSet::from([source])).unwrap();
        let w = weight("affine");
        // The sibling is fresh and manifest-less: nothing can evict it, so
        // admission is possible only if the recheck reconciles the phantom
        // ledger to its real pack bytes.
        b.write_with_total_cap(&pack_name('e'), &w, Some(u64::MAX))
            .unwrap();
        let pack = fs::metadata(b.path(&pack_name('e'))).unwrap().len();
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
        cache.write(&pack_name('a'), &w).unwrap();
        let pack = fs::metadata(cache.path(&pack_name('a'))).unwrap().len();
        // A vetoed REPLACEMENT write leaves the old target on disk: the
        // ledger reconcile must record real spend (target included), not the
        // ex-target `used` — an undercounted ledger would let siblings
        // overspend the configured cap.
        cache
            .write_with_total_cap(&pack_name('a'), &w, Some(1))
            .unwrap();
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
        assert!(cache.path(&pack_name('a')).exists());
        // And sibling estimates floor a stale-low ledger at real bytes.
        fs::write(cache.root.join("budget"), 0u64.to_le_bytes()).unwrap();
        assert_eq!(sibling_pack_estimate(&cache.root), pack);
    }

    /// A `{uuid}.tmp` crash leftover is kept by the sweep for the whole
    /// grace, so it must count as spend in every admission estimate —
    /// otherwise repeated interrupted writes sail past the configured cap.
    /// Retained = regular and inside the sweep grace; aged-out leftovers and
    /// non-regular `.tmp` names are debris, never spend.
    #[test]
    fn retained_tmp_files_count_toward_admission_estimates() {
        let temp = Temp::new();
        let dir = temp.0.join("cache");
        fs::create_dir(&dir).unwrap();
        fs::write(dir.join(pack_file('f')), vec![0_u8; 128]).unwrap();
        fs::write(dir.join(tmp_name("22222222")), vec![0_u8; 512]).unwrap();
        let stale_tmp = dir.join(tmp_name("33333333"));
        fs::write(&stale_tmp, vec![0_u8; 256]).unwrap();
        age(&stale_tmp, 2 * PACKED_SWEEP_MIN_AGE.as_secs());
        fs::create_dir(dir.join(tmp_name("44444444"))).unwrap();
        let outside = temp.0.join("outside.bin");
        fs::write(&outside, vec![0_u8; 1 << 20]).unwrap();
        std::os::unix::fs::symlink(&outside, dir.join(tmp_name("55555555"))).unwrap();
        assert_eq!(pack_bytes(&dir), 128);
        assert_eq!(retained_tmp_bytes(&dir), 512);
        // No ledger: estimate = pack bytes + retained tmp bytes.
        assert_eq!(sibling_pack_estimate(&dir), 640);
        assert_eq!(reconciled_sibling_pack_spend(&dir), 640);
        // A ledger reservation does not hide the tmp spend either, and the
        // reconcile rewrites the ledger to pack bytes only — `.tmp`
        // leftovers were never reservations.
        fs::write(dir.join("budget"), 64u64.to_le_bytes()).unwrap();
        assert_eq!(sibling_pack_estimate(&dir), 128 + 512);
        assert_eq!(reconciled_sibling_pack_spend(&dir), 640);
        let mut n = [0u8; 8];
        File::open(dir.join("budget"))
            .unwrap()
            .read_exact(&mut n)
            .unwrap();
        assert_eq!(u64::from_le_bytes(n), 128);
        // The writer's own dir counts the same way through root_pack_spend.
        assert_eq!(
            root_pack_spend(&temp.0, &dir, 128 + 512, sibling_pack_estimate),
            640
        );
        // And admission itself must see the retained bytes: a cap with room
        // for the write beside the pack alone must veto once the tmp spend
        // is counted, then admit after the leftover ages out.
        let w = weight("affine");
        let cache = PackedCache {
            root: dir.clone(),
            manifest: None,
        };
        // The write reservation equals a published pack's length — measure
        // it once in a scratch dir.
        let reserve = {
            let probe = PackedCache {
                root: temp.0.join("probe"),
                manifest: None,
            };
            probe.write("k", &w).unwrap();
            fs::metadata(probe.path("k")).unwrap().len()
        };
        fs::remove_file(dir.join("budget")).unwrap();
        cache
            .write_with_total_cap("k", &w, Some(128 + reserve))
            .unwrap();
        assert!(
            !cache.path("k").exists(),
            "a retained crash leftover must veto admission over the cap"
        );
        fs::remove_file(dir.join(tmp_name("22222222"))).unwrap();
        cache
            .write_with_total_cap("k", &w, Some(128 + reserve))
            .unwrap();
        assert!(
            cache.path("k").exists(),
            "once the leftover is gone the same cap admits the write"
        );
    }
}
