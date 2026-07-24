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
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use safetensors::tensor::{Dtype, TensorView};
use safetensors::{SafeTensors, serialize};
use sha2::{Digest, Sha256};

use crate::{BlockAllocator, LayerKVPool, PhysicalBlock};

const CACHE_ABI: &str = "mlx-paged-v1";
/// Filename suffix shared by every cold object (KV blocks and sidecars).
const OBJECT_SUFFIX: &str = ".safetensors";
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

/// Cache group a cold object belongs to — the cold-tier analogue of vLLM's
/// `BlockHashWithGroupId` (`vllm/v1/core/kv_cache_utils.py`), which folds a
/// group id into the hash key so blocks of one KV-cache group can never be
/// mistaken for another's. vLLM concatenates a 4-byte group id onto the
/// block hash; a fixed-width 32-byte key cannot grow, so the group is folded
/// in as the hashed domain-separation prefix instead — strictly stronger,
/// since the discriminant is inside the SHA-256 message rather than beside
/// it.
///
/// [`ColdGroup::Kv`] deliberately carries the pre-group domain tag verbatim,
/// so KV keys are byte-identical to the derivation that shipped before groups
/// existed (pinned by `kv_group_key_is_byte_identical_to_pre_group_derivation`).
///
/// Groups are also the on-disk namespace: [`object_file_name`] gives each
/// non-KV group its own filename suffix, so a sidecar can never be opened,
/// decoded, or restored as a KV block even if a key somehow repeated.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ColdGroup {
    /// Paged KV blocks: `1 + 2*num_layers` tensors per 16-token block.
    Kv,
    /// GDN recurrent state at a block boundary (qwen3_5 / qwen3_5_moe), which
    /// lives outside the paged pool and is therefore not covered by any KV
    /// block.
    GdnState,
    /// Sliding-window (`RotatingKVCache`) state at a block boundary (gemma4),
    /// likewise outside the paged pool.
    SlidingWindow,
}

impl ColdGroup {
    /// Every non-KV group, in a stable order. Used by name parsing and by the
    /// dashboard-facing filename contract.
    pub const SIDECAR_GROUPS: [Self; 2] = [Self::GdnState, Self::SlidingWindow];

    /// Domain-separation tag hashed as the first component of every key.
    ///
    /// Tags are NUL-terminated and NUL-free, and differ from one another
    /// before their first NUL, so no two groups can produce the same hasher
    /// input for any argument list — group separation does not rely on the
    /// (fixed-width) components that follow.
    const fn domain_tag(self) -> &'static [u8] {
        match self {
            // Byte-identical to the pre-group constant: DO NOT EDIT.
            Self::Kv => b"mlx-node:cold-prefix-block:v1\0",
            Self::GdnState => b"mlx-node:cold-sidecar-gdn-state:v1\0",
            Self::SlidingWindow => b"mlx-node:cold-sidecar-sliding-window:v1\0",
        }
    }

    /// Stable on-disk label: the filename infix for sidecars and the `group`
    /// metadata value. KV keeps the empty label so its canonical filename
    /// stays `<64-hex>.safetensors`.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Kv => "kv",
            Self::GdnState => "gdn_state",
            Self::SlidingWindow => "sliding_window",
        }
    }

    fn from_label(label: &str) -> Option<Self> {
        match label {
            "kv" => Some(Self::Kv),
            "gdn_state" => Some(Self::GdnState),
            "sliding_window" => Some(Self::SlidingWindow),
            _ => None,
        }
    }
}

/// Stable, collision-resistant chained key for one logical prefix block.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct ColdCacheKey([u8; 32]);

impl ColdCacheKey {
    /// Build a cold-object key within `group`. `parent` is `None` for the
    /// first block and the preceding block key thereafter. Integer encoding
    /// is explicitly LE so the key is stable across processes and Rust
    /// versions.
    ///
    /// `group` is hashed first (as its domain tag), so the same prefix in two
    /// groups yields two unrelated keys; [`ColdGroup::Kv`] reproduces the
    /// pre-group derivation exactly.
    pub fn chain(
        group: ColdGroup,
        fingerprint: ColdCacheFingerprint,
        parent: Option<Self>,
        tokens: &[u32],
        extra_keys: &[u64],
        cache_salt: u64,
        block_index: usize,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(group.domain_tag());
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
            + header_overhead(self.layers.len() as u64)
    }
}

/// Upper bound on [`ColdSidecarLayout::tensors_per_layer`]. Sidecar payloads
/// are a handful of state tensors per layer (e.g. GDN conv + recurrent
/// state); the cap keeps the descriptor count — and so the header bound in
/// [`header_overhead_for_descriptors`] — provably tied to `num_layers`.
const MAX_SIDECAR_TENSORS_PER_LAYER: u32 = 16;

/// Upper bound on [`ColdSidecarLayout::dims`]. `dims` is serialized into the
/// safetensors `__metadata__` object, so it must stay well inside
/// [`HEADER_METADATA_BYTES`]: 8 dims is at most 8*10 digits + 7 separators =
/// 87 bytes on top of the ~450-byte block metadata worst case.
const MAX_SIDECAR_DIMS: usize = 8;

/// Geometry of one persisted sidecar: the non-paged state a hybrid family
/// carries alongside its KV blocks.
///
/// A sidecar is anchored at a BOUNDARY — `boundary_tokens` prefix tokens have
/// been consumed — because recurrent/rotating state is only meaningful at an
/// exact token count. Restore reconciles DOWN to a boundary a sidecar
/// actually backs (vLLM `kv_cache_coordinator.py`: each group may only reduce
/// the candidate length), never up.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColdSidecarLayout {
    /// Which non-KV group this payload belongs to. Never [`ColdGroup::Kv`].
    pub group: ColdGroup,
    /// Prefix length (in tokens) the state is valid at. Must be a positive
    /// multiple of the KV block size so it names a real block boundary.
    pub boundary_tokens: u32,
    pub num_layers: u32,
    /// Tensors persisted per layer; group-specific (e.g. conv + recurrent).
    pub tensors_per_layer: u32,
    /// Element dtype label of the state tensors, e.g. `"BFloat16"`.
    pub dtype: String,
    /// Group-specific per-tensor geometry (e.g. `[num_heads, head_dim]`).
    pub dims: Vec<u32>,
    /// Byte length of every individual state tensor.
    pub bytes_per_tensor: usize,
}

impl ColdSidecarLayout {
    /// Total tensor count, `None` on overflow.
    pub fn tensor_count(&self) -> Option<usize> {
        (self.num_layers as usize).checked_mul(self.tensors_per_layer as usize)
    }

    /// The invariants that describe GEOMETRY alone — group, layer/tensor
    /// counts, dims, per-tensor byte length — with no reference to a boundary
    /// or to payload bytes.
    ///
    /// [`ColdSidecar::validate`] layers the boundary and payload checks on top,
    /// so a value accepted here still cannot be persisted in a shape the
    /// decoder would reject. [`ColdSidecarPolicy::new`] needs exactly this
    /// half: a policy is a geometry TEMPLATE whose boundary is only known once
    /// a candidate prefix is in hand.
    pub fn validate_geometry(&self) -> Result<(), String> {
        if self.group == ColdGroup::Kv {
            return Err("cold-cache sidecars must not use the KV group".to_string());
        }
        if self.num_layers == 0 || self.tensors_per_layer == 0 {
            return Err("cold-cache sidecar must carry at least one state tensor".to_string());
        }
        if self.tensors_per_layer > MAX_SIDECAR_TENSORS_PER_LAYER {
            return Err("cold-cache sidecar tensors-per-layer exceeds the bound".to_string());
        }
        if self.dims.is_empty() || self.dims.len() > MAX_SIDECAR_DIMS {
            return Err("cold-cache sidecar dims count out of range".to_string());
        }
        if self.dims.contains(&0) {
            return Err("cold-cache sidecar dims must be positive".to_string());
        }
        if self.bytes_per_tensor == 0 {
            return Err("cold-cache sidecar tensors must be non-empty".to_string());
        }
        Ok(())
    }
}

/// What a model family REQUIRES at every prefix boundary it resumes from: one
/// auxiliary (non-KV) group plus the exact geometry a sidecar of that group
/// must have. A family whose whole per-token state lives inside the paged pool
/// (dense `qwen3`) has NO policy; a hybrid family that keeps GDN recurrent or
/// sliding-window state outside the pool has one, and the cold-tier restore
/// walk refuses to hand back any prefix a matching sidecar does not back.
///
/// This is the cold-tier form of vLLM's per-group reconcile-down
/// (`vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/kv_cache_coordinator.py`):
/// every group may only REDUCE the candidate prefix length, never extend it,
/// and the reused prefix is the boundary every group agrees on.
///
/// `boundary_tokens` is deliberately NOT part of a policy — it is the one
/// layout field that varies per candidate — so [`Self::new`] normalizes it to
/// zero and [`Self::expected_at`] stamps the candidate boundary in.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColdSidecarPolicy {
    layout: ColdSidecarLayout,
}

impl ColdSidecarPolicy {
    /// Build a policy from a geometry template. Rejects [`ColdGroup::Kv`] and
    /// any geometry a sidecar could never legally be written with, so an
    /// impossible policy cannot be installed and then silently suppress every
    /// restore forever.
    pub fn new(layout: ColdSidecarLayout) -> Result<Self, String> {
        layout.validate_geometry()?;
        Ok(Self {
            layout: ColdSidecarLayout {
                boundary_tokens: 0,
                ..layout
            },
        })
    }

    /// The auxiliary group whose keys this policy probes. Never
    /// [`ColdGroup::Kv`].
    pub fn group(&self) -> ColdGroup {
        self.layout.group
    }

    /// The exact layout a sidecar anchored at `boundary_tokens` must have.
    /// [`ColdCacheManager::load_sidecar`] compares layouts for equality, so a
    /// sidecar recorded at a different boundary, dtype, or tensor shape is a
    /// miss rather than a reinterpretation of its bytes.
    pub fn expected_at(&self, boundary_tokens: u32) -> ColdSidecarLayout {
        ColdSidecarLayout {
            boundary_tokens,
            ..self.layout.clone()
        }
    }
}

/// Owned host representation of one sidecar object. Stored as its own file
/// under its own group-tagged key, so it is never reachable through the KV
/// block namespace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColdSidecar {
    pub key: ColdCacheKey,
    pub fingerprint: ColdCacheFingerprint,
    pub layout: ColdSidecarLayout,
    /// Layer-major: `tensors[layer * tensors_per_layer + slot]`.
    pub tensors: Vec<Vec<u8>>,
}

impl ColdSidecar {
    /// Every structural invariant the decoder also enforces, so a sidecar can
    /// never be written in a shape that would later fail to decode.
    fn validate(&self) -> Result<(), String> {
        self.layout.validate_geometry()?;
        if self.layout.boundary_tokens == 0 {
            return Err("cold-cache sidecar boundary must be a positive token count".to_string());
        }
        if self.layout.tensor_count() != Some(self.tensors.len()) {
            return Err("cold-cache sidecar tensor count does not match layout".to_string());
        }
        if self
            .tensors
            .iter()
            .any(|tensor| tensor.len() != self.layout.bytes_per_tensor)
        {
            return Err("cold-cache sidecar tensor byte length does not match layout".to_string());
        }
        Ok(())
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
    /// Writes dropped without landing on disk: the bounded queue was full
    /// at enqueue, or the commit rename failed after the queue accepted
    /// the job.
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
    /// Group the on-disk object belongs to. Kept alongside `file_name` so
    /// eviction and quota accounting cover sidecars as well as KV blocks —
    /// an unaccounted sidecar would sit outside the quota forever.
    group: ColdGroup,
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

    /// Opens only regular files. `NONBLOCK` keeps a FIFO swapped in for a
    /// block file from parking the open until a writer appears; the `fstat`
    /// gate then rejects every non-regular type. `O_NONBLOCK` has no effect
    /// on regular-file reads, so the returned `File` needs no flag reset.
    fn open_existing(&self, name: &str) -> std::io::Result<File> {
        use rustix::fs::{Mode, OFlags, openat};
        let flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK;
        let fd = openat(&self.fd, name, flags, Mode::empty()).map_err(std::io::Error::from)?;
        let stat = rustix::fs::fstat(&fd).map_err(std::io::Error::from)?;
        if !file_type_of(&stat).is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cold-cache entry is not a regular file",
            ));
        }
        Ok(File::from(fd))
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

    /// Identity and concrete type of the current directory entry, never
    /// following symlinks; `None` when no entry exists.
    fn stat_identity(&self, name: &str) -> Option<(FileIdentity, EntryKind)> {
        self.stat_no_follow(name).map(|stat| {
            let file_type = file_type_of(&stat);
            let kind = if file_type.is_file() {
                EntryKind::Regular
            } else if file_type.is_dir() {
                EntryKind::Directory
            } else {
                EntryKind::Other
            };
            (identity_of(&stat), kind)
        })
    }

    fn remove_dir_entry(&self, name: &str) -> std::io::Result<()> {
        rustix::fs::unlinkat(&self.fd, name, rustix::fs::AtFlags::REMOVEDIR)
            .map_err(std::io::Error::from)
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

    fn stat_identity(&self, _name: &str) -> Option<(FileIdentity, EntryKind)> {
        None
    }

    fn remove_dir_entry(&self, name: &str) -> std::io::Result<()> {
        fs::remove_dir(self.path.join(name))
    }
}

#[cfg(test)]
type TestSpaceOverride = Mutex<Option<Box<dyn Fn() -> Result<(u64, u64), String> + Send>>>;

#[cfg(test)]
type TestSyncOverride = Mutex<Option<Box<dyn Fn() -> Result<(), String> + Send>>>;

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
    /// Invoked between a failed open and its identity snapshot so tests can
    /// race a writer commit into exactly that window.
    #[cfg(test)]
    failed_open_identity_hook: Mutex<Option<Box<dyn Fn() + Send>>>,
    /// Replaces filesystem space probes so reserve-floor decisions are
    /// deterministic under test.
    #[cfg(test)]
    space_override: TestSpaceOverride,
    /// Forces the writer commit's directory fsync to fail so tests can drive
    /// a post-rename dir-sync error and assert index accounting stays
    /// consistent with the on-disk canonical file.
    #[cfg(test)]
    dir_sync_override: TestSyncOverride,
}

impl Shared {
    /// Filesystem `(total, available)` bytes backing eviction decisions.
    fn space(&self) -> Result<(u64, u64), String> {
        #[cfg(test)]
        if let Ok(hook) = self.space_override.lock()
            && let Some(hook) = hook.as_ref()
        {
            return hook();
        }
        self.root.space()
    }

    /// Directory fsync backing the writer commit's durability barrier.
    fn sync(&self) -> Result<(), String> {
        #[cfg(test)]
        if let Ok(hook) = self.dir_sync_override.lock()
            && let Some(hook) = hook.as_ref()
        {
            return hook();
        }
        self.root.sync()
    }
}

/// A unit of work for the single background writer thread. FIFO delivery on
/// the bounded channel means a `Barrier` enqueued after a run of `Block`s is
/// processed only once every one of those blocks has been fully persisted —
/// the property [`ColdCacheManager::drain`] relies on for a shutdown flush.
enum WriteJob {
    Block(ColdCacheBlock),
    /// A non-KV state sidecar. Persisted through the same durable path and
    /// covered by the same barrier semantics as `Block`.
    Sidecar(Box<ColdSidecar>),
    /// Drain marker: after every earlier `Block` has been persisted the writer
    /// acks it (unblocking `drain`) with whether all of those blocks since the
    /// previous barrier persisted successfully — `true` only when none failed.
    /// A dropped `rx` makes the ack a harmless no-op.
    Barrier(SyncSender<bool>),
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
            #[cfg(test)]
            failed_open_identity_hook: Mutex::new(None),
            #[cfg(test)]
            space_override: Mutex::new(None),
            #[cfg(test)]
            dir_sync_override: Mutex::new(None),
        });
        let (sender, receiver) = mpsc::sync_channel::<WriteJob>(queue_depth);
        let worker_shared = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("mlx-paged-ssd-writer".to_string())
            .spawn(move || {
                // Whether any block since the last barrier failed to persist.
                // Inference is still fail-open (the hot block is valid), but the
                // flag lets a covering drain barrier report durability honestly
                // instead of acking success unconditionally. Reset after each
                // barrier so every drain reports only on its own window.
                let mut failed = false;
                while let Ok(job) = receiver.recv() {
                    match job {
                        // Fail-open: inference already has a valid hot block. A
                        // persistence error only means the next process
                        // recomputes — but a pending drain barrier must learn
                        // that this covered block did not become durable.
                        WriteJob::Block(block) => {
                            if persist_block(&worker_shared, &block).is_err() {
                                failed = true;
                            }
                        }
                        WriteJob::Sidecar(sidecar) => {
                            if persist_sidecar(&worker_shared, &sidecar).is_err() {
                                failed = true;
                            }
                        }
                        // Every earlier `Block` has already been persisted (FIFO,
                        // single consumer), so acking here signals the drain is
                        // complete; the ack now reports whether all of them
                        // succeeded. A gone receiver (drain timed out) is fine.
                        WriteJob::Barrier(ack) => {
                            let _ = ack.send(!failed);
                            failed = false;
                        }
                    }
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
        self.contains_in(key, ColdGroup::Kv)
    }

    /// [`Self::contains`] restricted to one group: true only when the indexed
    /// object for `key` was written in `group`. Keys are already
    /// group-derived, so this can only differ from `contains` if a key ever
    /// repeated across groups — in which case the group-specific answer is
    /// the safe one, since it matches the file the loader would open.
    pub fn contains_in(&self, key: &ColdCacheKey, group: ColdGroup) -> bool {
        self.shared
            .index
            .lock()
            .map(|index| {
                index
                    .entries
                    .get(key)
                    .is_some_and(|entry| entry.group == group)
            })
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
        match self.sender.try_send(WriteJob::Block(block)) {
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

    /// Non-blocking enqueue of a state sidecar, with the same bounded-queue
    /// drop policy as [`Self::enqueue`]. A dropped sidecar is not a
    /// correctness problem: without it the next restore simply reconciles the
    /// candidate prefix down past that boundary and recomputes.
    pub fn enqueue_sidecar(&self, sidecar: ColdSidecar) -> Result<bool, String> {
        sidecar.validate()?;
        match self.sender.try_send(WriteJob::Sidecar(Box::new(sidecar))) {
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

    /// Block until every write accepted before this call is fully durable
    /// (payload fsync + rename + directory fsync), or `timeout` elapses.
    ///
    /// The WHOLE drain is bounded by `timeout`: a deadline is computed up front
    /// and bounds BOTH barrier admission and the ack wait. `std::sync::mpsc`'s
    /// `SyncSender` has no timed `send`, so a blocking `send` onto a full queue
    /// behind a stuck fsync could exceed the timeout or hang process exit;
    /// instead the barrier is admitted with `try_send` retried until a slot
    /// frees or the deadline passes, then the ack is awaited for the remaining
    /// time. FIFO ordering on the single-consumer writer guarantees the ack
    /// lands only after every earlier `Block`'s `persist_block` has returned,
    /// and the ack is `true` only when all of those blocks persisted — so
    /// `drain` returns `true` iff every `enqueue` that returned `Ok(true)`
    /// before this call is on disk. Returns `false` when the barrier cannot be
    /// admitted or acked within the deadline (a stuck fsync cannot hang exit)
    /// or when a covered block failed to persist, and `true` immediately when
    /// the writer is already gone (tier disabled/torn down: nothing to flush).
    pub fn drain(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let (tx, rx) = mpsc::sync_channel::<bool>(1);
        // Deadline-bounded admission: retry `try_send` (recovering the barrier
        // job on each `Full`) until a queue slot frees or the deadline passes.
        let mut job = WriteJob::Barrier(tx);
        loop {
            match self.sender.try_send(job) {
                Ok(()) => break,
                // Writer thread absent/stopped: no queued block can still be in
                // flight, so the drain is trivially satisfied.
                Err(TrySendError::Disconnected(_)) => return true,
                Err(TrySendError::Full(returned)) => {
                    if Instant::now() >= deadline {
                        // Could not even admit the barrier within the timeout.
                        return false;
                    }
                    job = returned;
                    std::thread::sleep(
                        Duration::from_millis(5)
                            .min(deadline.saturating_duration_since(Instant::now())),
                    );
                }
            }
        }
        // Success only on an honest `true` ack within the remaining time; a
        // timeout or a persist-failure ack (`false`) is a failed drain.
        matches!(
            rx.recv_timeout(deadline.saturating_duration_since(Instant::now())),
            Ok(true)
        )
    }

    /// Load and validate a block, bounding the restore read by the manager's
    /// quota — no single persisted entry can exceed it, since the writer
    /// evicts to keep the whole index within quota. The geometry-aware restore
    /// path ([`Self::restore_block`]) instead passes a tighter, pool-derived
    /// cap via [`Self::load_bounded`]. See [`Self::load_bounded`] for the full
    /// failure/pruning contract.
    pub fn load(
        &self,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
    ) -> Option<ColdCacheBlock> {
        let block = self.load_bounded(key, fingerprint, self.shared.quota_bytes)?;
        // Decode-level hit: this API hands back the decoded block directly, so a
        // successful load IS the realized outcome here (unlike `restore_block`,
        // which counts only after its transactional publish commits).
        self.shared.stats.hits.fetch_add(1, Ordering::Relaxed);
        self.shared
            .stats
            .bytes_restored
            .fetch_add(block.encoded_len(), Ordering::Relaxed);
        Some(block)
    }

    /// Load and validate the sidecar for `key`, which must be a key derived
    /// in `expected.group`. The read is bounded by `expected`'s own geometry,
    /// and the decoded layout must equal `expected` exactly — a sidecar
    /// recorded under a different dtype, layer count, tensor count, or
    /// boundary is a miss, never a reinterpretation of its bytes.
    ///
    /// `None` covers absent, unreadable, malformed, over-sized, and
    /// mismatched sidecars alike. Callers reconcile DOWN on `None`: drop the
    /// candidate prefix back to the last boundary a sidecar does back
    /// (vLLM `kv_cache_coordinator.py`), never restore an attention-only
    /// prefix whose recurrent state is missing.
    ///
    /// No `hits`/`bytes_restored` bump: a sidecar is a precondition for reuse,
    /// not reuse itself, and realized reuse is counted once by
    /// [`Self::restore_block`] per KV block actually published.
    pub fn load_sidecar(
        &self,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        expected: &ColdSidecarLayout,
    ) -> Option<ColdSidecar> {
        if expected.group == ColdGroup::Kv {
            return None;
        }
        let max_encoded = max_encoded_len_for_sidecar(expected)?;
        let sidecar = self.load_object_bounded(key, expected.group, max_encoded, |bytes| {
            decode_sidecar(bytes, key, fingerprint, expected.group)
        })?;
        if &sidecar.layout != expected {
            // A structurally valid sidecar for a different geometry is a
            // fall-back to recompute, exactly like a layout-mismatched block
            // in `restore_block`, and is counted the same way.
            self.shared.stats.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        Some(sidecar)
    }

    /// Load and validate a block, reading at most `max_encoded` bytes so a
    /// corrupt/tampered oversized entry (possibly a sparse regular file whose
    /// `st_size` reports gigabytes) can never drive an unbounded allocation:
    /// the read streams through `take(max_encoded + 1)` and an entry longer
    /// than `max_encoded` is treated as corruption (miss + `corruptions`
    /// bump + prune), identical to any decode failure — fail-open.
    ///
    /// Every failed read is a miss; a payload that existed but failed
    /// validation additionally counts as a corruption (an entry that could
    /// not be opened never does). Failure cleanup runs under the same index
    /// lock the writer holds across [rename + index publish];
    /// [`prune_failed_load`] clears the canonical name only when the entry
    /// there is the one observed to fail (dev+inode) or is a non-regular type
    /// that can never be a writer commit, so an in-process writer's freshly
    /// committed replacement is never deleted or de-indexed; the failed-open
    /// identity snapshot is itself taken under that lock, so the writer can
    /// never publish between a failed open and the snapshot. Coordination is
    /// in-process only: a concurrent *process* mutating the same root stays
    /// fail-open — the worst case is a stale index entry, one recomputed
    /// prefix, or one lost persist (an external actor swapping the entry
    /// inside the stat window right after a failed open).
    ///
    /// The byte quota is therefore a per-process *best-effort* cap, not a
    /// strict cross-process invariant: each process admits blocks against its
    /// own startup-scan view, so N processes that each start on the same root
    /// before either writes may transiently hold up to ~N×quota on disk. This
    /// self-corrects — the next process whose [`rebuild_index`] scan sees the
    /// combined on-disk total evicts LRU down to the quota on its first write.
    /// The free-space floor, by contrast, is checked against a live `statvfs`
    /// re-sampled after every eviction, so the only cross-process slack there
    /// is the handful of in-flight block writes, far below the reserve. The
    /// strict-quota fix (an interprocess lock spanning scan→evict→reserve→
    /// rename→publish) would invert this deliberately lock-free design and is
    /// out of scope for the v1 best-effort cache.
    fn load_bounded(
        &self,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        max_encoded: u64,
    ) -> Option<ColdCacheBlock> {
        self.load_object_bounded(key, ColdGroup::Kv, max_encoded, |bytes| {
            decode_block(bytes, key, fingerprint)
        })
    }

    /// Group-generic body of [`Self::load_bounded`]: open the canonical name
    /// for `(key, group)`, slurp at most `max_encoded` bytes, and hand them to
    /// `decode`. The open/prune/touch/statistics contract documented on
    /// [`Self::load_bounded`] applies verbatim to every group — only the
    /// decoder differs, so a sidecar can never be decoded by the block
    /// decoder (or vice versa) and every malformed payload is a graceful miss.
    fn load_object_bounded<T>(
        &self,
        key: ColdCacheKey,
        group: ColdGroup,
        max_encoded: u64,
        decode: impl FnOnce(&[u8]) -> Result<T, String>,
    ) -> Option<T> {
        let name = object_file_name(&key, group);
        let mut observed_identity = None;
        let mut opened_file = None;
        // The index lock spans [open → failed-open identity snapshot]: the
        // writer publishes replacements under the same lock, so an identity
        // captured here is genuinely the entry that failed, never a
        // replacement renamed in between the failed open and the stat.
        // Released before any read/decode work; a successful open needs no
        // exclusion because its identity comes from the descriptor itself.
        let open_result = {
            let _index_guard = self.shared.index.lock().ok();
            let result = self.shared.root.open_existing(&name);
            if let Err(e) = &result
                && e.kind() != std::io::ErrorKind::NotFound
            {
                #[cfg(test)]
                if let Ok(hook) = self.shared.failed_open_identity_hook.lock()
                    && let Some(hook) = hook.as_ref()
                {
                    hook();
                }
                // Capture the identity of the entry that made the open
                // fail so pruning can distinguish it from a later writer
                // replacement. Skipped for NotFound: an entry committed
                // after a plain miss must never be mistaken for the one
                // that failed.
                observed_identity = self
                    .shared
                    .root
                    .stat_identity(&name)
                    .map(|(identity, _)| identity);
            }
            result
        };
        let result = match open_result {
            Ok(mut file) => {
                observed_identity = open_identity(&file);
                let mut bytes = Vec::new();
                // Bounded slurp: cap the read at the caller's geometry-derived
                // maximum (+1 so an over-cap file is detectable). A sparse
                // regular file reports a huge `st_size`, but `take` caps the
                // allocation; anything exceeding the bound is treated as
                // corruption and never read in full.
                let read = (&mut file)
                    .take(max_encoded.saturating_add(1))
                    .read_to_end(&mut bytes)
                    .map_err(|e| e.to_string())
                    .and_then(|_| {
                        if bytes.len() as u64 > max_encoded {
                            Err("cold-cache entry exceeds geometry bound".to_string())
                        } else {
                            decode(&bytes)
                        }
                    });
                opened_file = Some(file);
                read
            }
            Err(e) => Err(e.to_string()),
        };
        match result {
            Ok(decoded) => {
                // NOTE: the `hits` / `bytes_restored` reuse counters are NOT
                // bumped here. A successful decode is not yet realized reuse —
                // `restore_block` still has to validate layout, allocate a
                // physical block, upload to the GPU, and publish the prefix,
                // any of which can fail and fall back to prefill. The counters
                // are incremented at each caller's true success boundary: the
                // public `load` (decode-level API) below, and `restore_block`
                // only after `publish_restored_prefix` commits.
                //
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
                Some(decoded)
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
                prune_failed_load(&self.shared, key, &name, observed_identity);
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
        // Bound the restore read by the exact pool geometry this block's layout
        // is validated against just below, so a tampered oversized entry at the
        // canonical name is a bounded miss, never a gigabyte allocation.
        let cold = self.load_bounded(key, fingerprint, max_encoded_len_for_pool(pool))?;
        // Each post-decode failure below is a real fall-back to ordinary prefill,
        // so it must count exactly one miss. The decode itself counted neither
        // hit nor miss (`load_bounded` bumps `misses` only for a failed decode in
        // its `Err` arm), and each path here is reached only after `load_bounded`
        // returned `Some`, so there is no double-count with that decode-level miss.
        if cold.tokens != identity.tokens || !layout_matches_pool(&cold.layout, pool) {
            self.shared.stats.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let block = match allocator
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .allocate()
        {
            Some(block) => block,
            None => {
                self.shared.stats.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };

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
                self.shared.stats.misses.fetch_add(1, Ordering::Relaxed);
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
            Ok(true) => {
                // Realized reuse: the decoded prefix is now allocated, uploaded,
                // and published into the pool. Count the hit and restored bytes
                // only here so the dashboard/trace never report reuse for a
                // block that decoded but fell back to prefill (layout mismatch,
                // allocation exhaustion, upload error, or a lost publish race).
                self.shared.stats.hits.fetch_add(1, Ordering::Relaxed);
                self.shared
                    .stats
                    .bytes_restored
                    .fetch_add(cold.encoded_len(), Ordering::Relaxed);
                Some(block)
            }
            _ => {
                allocator
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .free(block);
                self.shared.stats.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }
}

/// Per-tensor safetensors descriptor allowance. `encode_block`'s longest
/// descriptor is
/// `"layer.<i>.value":{"dtype":"U8","shape":[N],"data_offsets":[A,B]}` plus a
/// separating comma: 60 bytes of fixed punctuation/keywords, at most
/// `digits(i)` index digits, and three integer fields (one shape, two offsets),
/// each a payload offset and so at most 20 decimal digits (`u64`). That caps
/// any real descriptor at 60 + 20 + 3*20 = 140 bytes; 256 leaves generous
/// headroom for every layer count.
const HEADER_BYTES_PER_DESCRIPTOR: u64 = 256;

/// Fixed allowance for the `__metadata__` object: `abi` +
/// `key`/`fingerprint`/`checksum` (three 64-char hex strings) + the numeric
/// layout fields + JSON syntax (~450 bytes worst case).
const HEADER_METADATA_BYTES: u64 = 1024;

/// safetensors framing: the 8-byte little-endian header-length prefix plus up
/// to 7 bytes of padding that 8-byte-aligns the JSON header.
const HEADER_FRAMING_BYTES: u64 = 8 + 7;

/// Upper bound on the safetensors header + framing `encode_block` wraps around
/// the raw K/V/token payload. The container is `[8-byte header length][JSON
/// header, 8-byte aligned][payload]`, and the JSON header carries
/// `1 + 2*num_layers` tensor descriptors (`tokens` plus each layer's
/// key/value) and one `__metadata__` object, so the overhead grows with layer
/// count — a flat constant cannot cover deep models. Shared by
/// [`ColdCacheBlock::encoded_len`] and [`max_encoded_len_for_pool`] so the two
/// bounds can never drift.
fn header_overhead(num_layers: u64) -> u64 {
    header_overhead_for_descriptors(num_layers.saturating_mul(2).saturating_add(1))
}

/// [`header_overhead`] for an arbitrary descriptor count — the sidecar
/// encoder writes `num_layers * tensors_per_layer` descriptors and no
/// `tokens` tensor, so it cannot use the block formula. The per-descriptor
/// and metadata allowances are shared, and a sidecar's longest descriptor
/// name (`layer.<i>.state.<j>`, at most 25 chars for `u32` indices) stays
/// well inside [`HEADER_BYTES_PER_DESCRIPTOR`].
fn header_overhead_for_descriptors(descriptors: u64) -> u64 {
    descriptors
        .saturating_mul(HEADER_BYTES_PER_DESCRIPTOR)
        .saturating_add(HEADER_METADATA_BYTES)
        .saturating_add(HEADER_FRAMING_BYTES)
}

/// Upper bound on a legitimately-encoded block for `pool`, mirroring
/// [`ColdCacheBlock::encoded_len`]: all layers' K+V bytes (via
/// [`crate::profile::bytes_per_block`], which is exactly
/// `num_layers * (key_bytes_per_layer + value_bytes_per_layer)`), the block's
/// per-token `u32` ids, and the encoder's header/framing overhead
/// ([`header_overhead`], which scales with the `1 + 2*num_layers` safetensors
/// descriptor count). A degenerate (zero-factor) geometry yields the
/// overhead-only floor, which fails closed by rejecting every real block — the
/// pool would fail [`layout_matches_pool`] anyway.
fn max_encoded_len_for_pool(pool: &LayerKVPool) -> u64 {
    let kv_bytes = crate::profile::bytes_per_block(
        pool.num_layers() as u32,
        pool.config().num_kv_heads,
        pool.config().head_size,
        pool.block_size(),
        pool.cache_dtype(),
    )
    .unwrap_or(0);
    let token_bytes = pool.block_size() as u64 * size_of::<u32>() as u64;
    kv_bytes
        .saturating_add(token_bytes)
        .saturating_add(header_overhead(pool.num_layers() as u64))
}

/// Upper bound on a legitimately-encoded sidecar with `layout`: every state
/// tensor's bytes (`num_layers * tensors_per_layer * bytes_per_tensor`, all
/// pinned by [`ColdSidecar::validate`]) plus the encoder's header/framing
/// overhead for that many descriptors. `None` when the layout's own counts
/// overflow or exceed the structural caps, which fails the read closed (no
/// bound, no load).
fn max_encoded_len_for_sidecar(layout: &ColdSidecarLayout) -> Option<u64> {
    if layout.tensors_per_layer > MAX_SIDECAR_TENSORS_PER_LAYER {
        return None;
    }
    let tensors = layout.tensor_count()? as u64;
    let payload = tensors.checked_mul(layout.bytes_per_tensor as u64)?;
    payload.checked_add(header_overhead_for_descriptors(tensors))
}

/// Per-layer K/V byte lengths one block occupies in `pool`, mirroring
/// `LayerKVPool::read_blocks_to_host` / `write_blocks_from_host` exactly
/// (including the `head_size / x` integer division on the K side). `None`
/// for a pool whose dtype/FP8 combination has no kernel layout, which makes
/// [`layout_matches_pool`] fail closed.
fn pool_layer_bytes(pool: &LayerKVPool) -> Option<(usize, usize)> {
    let x = pool.cache_pack_factor().ok()? as u64;
    if x == 0 {
        return None;
    }
    let element = crate::profile::dtype_size_for(pool.cache_dtype()) as u64;
    let heads = pool.config().num_kv_heads as u64;
    let head_size = pool.config().head_size as u64;
    let block_size = pool.block_size() as u64;
    let key = heads * (head_size / x) * x * block_size * element;
    let value = heads * head_size * block_size * element;
    Some((usize::try_from(key).ok()?, usize::try_from(value).ok()?))
}

/// Whether a decoded block's layout is exactly this pool's geometry.
///
/// The per-layer byte lengths are compared here, not left to
/// `write_blocks_from_host`: a block that agrees on
/// `(block_size, num_layers, num_kv_heads, head_size, cache_dtype)` but not
/// on the packed K/V byte lengths — a different kernel pack factor `x`, or a
/// `head_size` not divisible by `x` — would otherwise pass validation and be
/// caught only mid-upload, after a physical block had been allocated and
/// earlier layers already written into the pool.
fn layout_matches_pool(layout: &ColdCacheLayout, pool: &LayerKVPool) -> bool {
    let Some((key_bytes, value_bytes)) = pool_layer_bytes(pool) else {
        return false;
    };
    layout.block_size == pool.block_size()
        && layout.num_layers as usize == pool.num_layers()
        && layout.num_kv_heads == pool.config().num_kv_heads
        && layout.head_size == pool.config().head_size
        && layout.cache_dtype == format!("{:?}", pool.cache_dtype())
        && layout.key_bytes_per_layer == key_bytes
        && layout.value_bytes_per_layer == value_bytes
}

fn persist_block(shared: &Shared, block: &ColdCacheBlock) -> Result<(), String> {
    block.validate()?;
    let bytes = encode_block(block)?;
    persist_encoded(shared, block.key, ColdGroup::Kv, &bytes)
}

/// Sidecars go through the identical durable path as blocks — evict to
/// quota, write to a writer temp, `fsync`, `renameat`, publish under the
/// index lock, `fsync` the directory — only the encoder and the canonical
/// name differ.
fn persist_sidecar(shared: &Shared, sidecar: &ColdSidecar) -> Result<(), String> {
    sidecar.validate()?;
    let bytes = encode_sidecar(sidecar)?;
    persist_encoded(shared, sidecar.key, sidecar.layout.group, &bytes)
}

fn persist_encoded(
    shared: &Shared,
    key: ColdCacheKey,
    group: ColdGroup,
    bytes: &[u8],
) -> Result<(), String> {
    evict_for_write(shared, bytes.len() as u64)?;
    let destination = object_file_name(&key, group);
    let temp = format!(
        ".{}.{}.{}.tmp",
        key.to_hex(),
        std::process::id(),
        now_tick()
    );
    let mut file = shared.root.create_exclusive(&temp)?;
    let size = bytes.len() as u64;
    if let Err(error) = (|| -> Result<(), String> {
        file.write_all(bytes)
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
        // A failed commit rename drops a write the queue already accepted;
        // counting it in `queue_drops` keeps enqueue/write accounting
        // honest, since the worker itself is fail-open and returns the
        // error to nobody.
        shared.root.rename(&temp, &destination).inspect_err(|_| {
            shared.stats.queue_drops.fetch_add(1, Ordering::Relaxed);
        })?;
        if let Some(old) = index.entries.insert(
            key,
            IndexEntry {
                group,
                file_name: destination.clone(),
                size,
                last_access: now_tick(),
            },
        ) {
            index.total_bytes = index.total_bytes.saturating_sub(old.size);
        }
        index.total_bytes = index.total_bytes.saturating_add(size);
        // The rename is the true commit point (the payload was already
        // `sync_all`'d), so the index publish above is bound to rename
        // success, not to this directory fsync. A dir-fsync failure here
        // therefore leaves in-process accounting consistent with the
        // renamed canonical file — the cleanup `unlink(&temp)` stays a
        // harmless NotFound no-op and `rebuild_index` still heals on
        // restart — rather than orphaning the block outside the quota.
        shared.sync()?;
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

/// Evict least-recently-used entries until `incoming` fits both the
/// logical quota and the physical free-space reserve. Clearing goes
/// through the same type-safe [`clear_entry`] path as failed-load pruning,
/// and an entry is de-indexed (bytes debited, eviction counted) only once
/// its canonical name is actually clear — quarantining an obstructing
/// directory counts, since the name becomes writable again. A name that
/// cannot be cleared keeps its index entry and is skipped for the rest of
/// this pass, so one stuck entry can neither spin the loop nor falsify
/// accounting. The reserve check never trusts logical index sizes:
/// availability is re-sampled (statvfs) after every clearing, so entries
/// that free no bytes — already missing, or quarantined rather than
/// deleted — can never admit a write that would breach the reserve floor.
fn evict_for_write(shared: &Shared, incoming: u64) -> Result<(), String> {
    let (_, mut available) = shared.space()?;
    let mut index = shared
        .index
        .lock()
        .map_err(|_| "cold-cache index mutex poisoned".to_string())?;
    let mut unclearable: Vec<ColdCacheKey> = Vec::new();
    while index.total_bytes.saturating_add(incoming) > shared.quota_bytes
        || available < shared.reserve_bytes.saturating_add(incoming)
    {
        let Some((&key, entry)) = index
            .entries
            .iter()
            .filter(|&(key, _)| !unclearable.contains(key))
            .min_by_key(|(_, entry)| entry.last_access)
        else {
            return Err("insufficient disk space for cold-cache write".to_string());
        };
        let name = entry.file_name.clone();
        let cleared = match shared.root.stat_identity(&name) {
            Some((_, kind)) => clear_entry(&shared.root, &name, kind),
            // Either the entry already vanished (the unlink then observes
            // NotFound) or this platform reports no identities and the
            // plain unlink decides.
            None => entry_gone(shared.root.unlink(&name)),
        };
        if !cleared {
            unclearable.push(key);
            continue;
        }
        if let Some(entry) = index.entries.remove(&key) {
            index.total_bytes = index.total_bytes.saturating_sub(entry.size);
            shared.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }
        let (_, resampled) = shared.space()?;
        available = resampled;
    }
    Ok(())
}

/// Cleanup after a failed load, under the same index lock the writer holds
/// across [rename + index publish]. The key is de-indexed only once the
/// canonical name is actually clear, so a de-indexed key can never leave
/// behind an obstruction that fails every later writer commit rename.
///
/// `observed_identity` identifies the entry that produced the failure:
/// `fstat` of the descriptor when the open succeeded, else a no-follow
/// stat taken right after a non-NotFound open failure, under the index
/// lock, so it can never be a writer replacement. A regular file at
/// the name is preserved (with its index entry) only on positive
/// replacement evidence — it carries a different identity than the
/// observed one, or it appeared where the failed open found nothing —
/// because only then can it be a writer's freshly renamed-in commit.
/// Without such evidence the entry can only fail again (corrupt payload,
/// or unopenable, e.g. mode 000), and a non-regular entry can never be a
/// writer commit nor be opened at all (`open_existing` rejects every
/// non-regular type after `fstat`), so both are cleared via
/// [`clear_entry`] and then de-indexed. When clearing fails the index
/// entry stays, and the next load miss for the key retries.
fn prune_failed_load(
    shared: &Shared,
    key: ColdCacheKey,
    name: &str,
    observed_identity: Option<FileIdentity>,
) {
    let Ok(mut index) = shared.index.lock() else {
        return;
    };
    let cleared = match shared.root.stat_identity(name) {
        Some((current, EntryKind::Regular)) if observed_identity != Some(current) => false,
        Some((_, kind)) => clear_entry(&shared.root, name, kind),
        None => true,
    };
    // De-index only the entry that actually names the file just cleared. The
    // group is part of the key derivation, so a sidecar and a block can never
    // share a key in the first place; checking the name keeps that a local,
    // checkable property instead of a cross-module assumption.
    if cleared
        && index
            .entries
            .get(&key)
            .is_some_and(|entry| entry.file_name == name)
        && let Some(entry) = index.entries.remove(&key)
    {
        index.total_bytes = index.total_bytes.saturating_sub(entry.size);
    }
}

/// Clear whatever entry currently occupies canonical `name`, by observed
/// type — the single clearing path shared by eviction and failed-load
/// pruning. Regular and other non-directory entries have their directory
/// entry unlinked (`unlinkat` removes the entry itself, never a symlink's
/// target). An empty directory is removed with `unlinkat(REMOVEDIR)`; a
/// non-empty one is renamed aside to a quarantine name — unknown content
/// is never deleted — that the index scanner and startup cleanup ignore.
/// Returns whether the canonical name is clear afterwards (an entry that
/// vanished concurrently counts as cleared).
fn clear_entry(root: &RootDir, name: &str, kind: EntryKind) -> bool {
    match kind {
        EntryKind::Regular | EntryKind::Other => entry_gone(root.unlink(name)),
        EntryKind::Directory => match root.remove_dir_entry(name) {
            Ok(()) => true,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
            Err(_) => root.rename(name, &quarantine_name(name)).is_ok(),
        },
    }
}

fn entry_gone(result: std::io::Result<()>) -> bool {
    match result {
        Ok(()) => true,
        Err(e) => e.kind() == std::io::ErrorKind::NotFound,
    }
}

/// Quarantine name for a directory obstructing a canonical block name.
/// Shaped like the writer temp convention (leading dot, pid + tick for
/// uniqueness) but matches neither `*.safetensors` nor
/// [`is_cold_cache_temp_file`], so quarantined directories are never
/// indexed and never deleted by startup cleanup.
fn quarantine_name(name: &str) -> String {
    format!(".blocked.{name}.{}.{}", std::process::id(), now_tick())
}

#[cfg_attr(not(unix), allow(dead_code))]
#[derive(Clone, Copy, Eq, PartialEq)]
enum EntryKind {
    Regular,
    Directory,
    Other,
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

/// Canonical filename for a cold object. KV keeps the historical
/// `<64-hex>.safetensors`; every other group gets its label as an infix
/// (`<64-hex>.gdn_state.safetensors`), so the two namespaces are disjoint on
/// disk as well as in the key derivation, and the dashboard can account them
/// separately (packages/dashboard/src/cache.ts).
fn object_file_name(key: &ColdCacheKey, group: ColdGroup) -> String {
    match group {
        ColdGroup::Kv => format!("{}{OBJECT_SUFFIX}", key.to_hex()),
        other => format!("{}.{}{OBJECT_SUFFIX}", key.to_hex(), other.label()),
    }
}

/// Inverse of [`object_file_name`]. `None` for anything that is not a
/// canonical cold object (writer temps, quarantined obstructions, foreign
/// files), so the index scanner never adopts a name it could not later
/// resolve back to the same file.
fn parse_object_name(name: &str) -> Option<(ColdCacheKey, ColdGroup)> {
    let stem = name.strip_suffix(OBJECT_SUFFIX)?;
    if let Some(key) = ColdCacheKey::from_hex(stem) {
        return Some((key, ColdGroup::Kv));
    }
    let (hex, label) = stem.split_once('.')?;
    let key = ColdCacheKey::from_hex(hex)?;
    let group = ColdGroup::from_label(label)?;
    // `kv` as an infix would name a second file for a KV key; only the bare
    // hex form is canonical for that group.
    (group != ColdGroup::Kv).then_some((key, group))
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
    // `num_layers` comes from untrusted metadata; a valid block has exactly
    // `1 + 2*num_layers` tensors (`tokens` plus each layer's key/value).
    // Checking it against the actually-deserialized tensor count (itself
    // bounded by the byte-capped read) keeps the `Vec::with_capacity` below
    // from being sized by a forged huge value.
    if (layout.num_layers as usize)
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        != Some(tensors.len())
    {
        return Err("cold-cache tensor count does not match num_layers".to_string());
    }
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

fn sidecar_tensor_name(layer: usize, slot: usize) -> String {
    format!("layer.{layer}.state.{slot}")
}

/// Serialize a sidecar into its own safetensors object. The container shape
/// is deliberately NOT the block shape — no `tokens` tensor, `state`-named
/// descriptors, and a `group` metadata field — so the two object types can
/// never be confused even before their disjoint keys and filenames are
/// considered.
fn encode_sidecar(sidecar: &ColdSidecar) -> Result<Vec<u8>, String> {
    sidecar.validate()?;
    let per_layer = sidecar.layout.tensors_per_layer as usize;
    let mut owned: Vec<(String, Vec<u8>)> = Vec::with_capacity(sidecar.tensors.len());
    for (index, tensor) in sidecar.tensors.iter().enumerate() {
        owned.push((
            sidecar_tensor_name(index / per_layer, index % per_layer),
            tensor.clone(),
        ));
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
    metadata.insert(
        "group".to_string(),
        sidecar.layout.group.label().to_string(),
    );
    metadata.insert("key".to_string(), sidecar.key.to_hex());
    metadata.insert("fingerprint".to_string(), sidecar.fingerprint.to_hex());
    metadata.insert("checksum".to_string(), checksum);
    metadata.insert(
        "boundary_tokens".to_string(),
        sidecar.layout.boundary_tokens.to_string(),
    );
    metadata.insert(
        "num_layers".to_string(),
        sidecar.layout.num_layers.to_string(),
    );
    metadata.insert(
        "tensors_per_layer".to_string(),
        sidecar.layout.tensors_per_layer.to_string(),
    );
    metadata.insert("dtype".to_string(), sidecar.layout.dtype.clone());
    metadata.insert(
        "dims".to_string(),
        sidecar
            .layout
            .dims
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","),
    );
    metadata.insert(
        "bytes_per_tensor".to_string(),
        sidecar.layout.bytes_per_tensor.to_string(),
    );
    serialize(views.map_err(|e| e.to_string())?, Some(metadata)).map_err(|e| e.to_string())
}

/// Inverse of [`encode_sidecar`], fail-closed at every step: a missing or
/// unparsable field, a `group`/`key`/`fingerprint`/`abi` that does not match
/// what the caller asked for, a tensor count that disagrees with
/// `num_layers * tensors_per_layer`, a tensor of the wrong byte length, or a
/// payload checksum mismatch all return `Err` — which the loader turns into a
/// miss plus a corruption bump plus a prune. Nothing here can panic or
/// allocate from an untrusted count: the `Vec` is reserved only after the
/// metadata counts have been checked against the deserialized tensor count,
/// which is itself bounded by the byte-capped read.
fn decode_sidecar(
    bytes: &[u8],
    expected_key: ColdCacheKey,
    expected_fingerprint: ColdCacheFingerprint,
    expected_group: ColdGroup,
) -> Result<ColdSidecar, String> {
    if expected_group == ColdGroup::Kv {
        return Err("cold-cache sidecars must not use the KV group".to_string());
    }
    let (_, header) = SafeTensors::read_metadata(bytes).map_err(|e| e.to_string())?;
    let metadata = header
        .metadata()
        .as_ref()
        .ok_or_else(|| "cold-cache sidecar metadata missing".to_string())?;
    let tensors = SafeTensors::deserialize(bytes).map_err(|e| e.to_string())?;
    let get = |name: &str| {
        metadata
            .get(name)
            .cloned()
            .ok_or_else(|| format!("cold-cache sidecar metadata `{name}` missing"))
    };
    if get("abi")? != CACHE_ABI
        || get("group")? != expected_group.label()
        || get("key")? != expected_key.to_hex()
        || get("fingerprint")? != expected_fingerprint.to_hex()
    {
        return Err("cold-cache sidecar identity/ABI mismatch".to_string());
    }
    let parse = |name: &str| -> Result<u32, String> {
        get(name)?
            .parse::<u32>()
            .map_err(|_| format!("invalid cold-cache sidecar metadata `{name}`"))
    };
    let dims_field = get("dims")?;
    let mut dims = Vec::new();
    for part in dims_field.split(',') {
        if dims.len() == MAX_SIDECAR_DIMS {
            return Err("cold-cache sidecar dims count out of range".to_string());
        }
        dims.push(
            part.parse::<u32>()
                .map_err(|_| "invalid cold-cache sidecar metadata `dims`".to_string())?,
        );
    }
    let layout = ColdSidecarLayout {
        group: expected_group,
        boundary_tokens: parse("boundary_tokens")?,
        num_layers: parse("num_layers")?,
        tensors_per_layer: parse("tensors_per_layer")?,
        dtype: get("dtype")?,
        dims,
        bytes_per_tensor: get("bytes_per_tensor")?
            .parse::<usize>()
            .map_err(|_| "invalid cold-cache sidecar metadata `bytes_per_tensor`".to_string())?,
    };
    // `num_layers`/`tensors_per_layer` are untrusted metadata; a valid
    // sidecar has exactly their product of tensors. Checking that against the
    // actually-deserialized tensor count (bounded by the byte-capped read)
    // keeps the reservation below from being sized by a forged value.
    if layout.tensors_per_layer > MAX_SIDECAR_TENSORS_PER_LAYER
        || layout.tensor_count() != Some(tensors.len())
    {
        return Err("cold-cache sidecar tensor count does not match layout".to_string());
    }
    let per_layer = layout.tensors_per_layer as usize;
    let mut state = Vec::with_capacity(tensors.len());
    for index in 0..tensors.len() {
        state.push(
            tensors
                .tensor(&sidecar_tensor_name(index / per_layer, index % per_layer))
                .map_err(|e| e.to_string())?
                .data()
                .to_vec(),
        );
    }
    let sidecar = ColdSidecar {
        key: expected_key,
        fingerprint: expected_fingerprint,
        layout,
        tensors: state,
    };
    // Structural validation (including per-tensor byte lengths) before the
    // checksum, so a forged header can never make the checksum the only gate.
    sidecar.validate()?;

    let mut owned = Vec::with_capacity(sidecar.tensors.len());
    for (index, tensor) in sidecar.tensors.iter().enumerate() {
        owned.push((
            sidecar_tensor_name(index / per_layer, index % per_layer),
            tensor.clone(),
        ));
    }
    if payload_checksum(&owned) != get("checksum")? {
        return Err("cold-cache sidecar payload checksum mismatch".to_string());
    }
    Ok(sidecar)
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
        let Some((key, group)) = parse_object_name(&name) else {
            if is_cold_cache_temp_file(&name) {
                let _ = root.unlink(&name);
            }
            continue;
        };
        let Some((size, last_access)) = root.stat_file(&name) else {
            continue;
        };
        index.entries.insert(
            key,
            IndexEntry {
                group,
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

    fn sidecar_layout(group: ColdGroup) -> ColdSidecarLayout {
        ColdSidecarLayout {
            group,
            boundary_tokens: 16,
            num_layers: 2,
            tensors_per_layer: 2,
            dtype: "BFloat16".to_string(),
            dims: vec![4, 8, 2],
            bytes_per_tensor: 6,
        }
    }

    fn sidecar(key: ColdCacheKey, group: ColdGroup) -> ColdSidecar {
        let layout = sidecar_layout(group);
        let count = layout.tensor_count().unwrap();
        ColdSidecar {
            key,
            fingerprint: fingerprint(),
            layout,
            // Distinct per-tensor content so a decoder that reorders or
            // aliases tensors cannot round-trip.
            tensors: (0..count)
                .map(|i| (0..6u8).map(|b| i as u8 * 16 + b).collect())
                .collect(),
        }
    }

    /// Byte-for-byte reimplementation of the key derivation as it existed
    /// before [`ColdGroup`] — the reference the KV group must still match.
    fn pre_group_chain(
        fingerprint: ColdCacheFingerprint,
        parent: Option<ColdCacheKey>,
        tokens: &[u32],
        extra_keys: &[u64],
        cache_salt: u64,
        block_index: usize,
    ) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"mlx-node:cold-prefix-block:v1\0");
        hasher.update(fingerprint.as_bytes());
        hasher.update(parent.map_or([0u8; 32], |key| *key.as_bytes()));
        hasher.update((block_index as u64).to_le_bytes());
        hasher.update((tokens.len() as u64).to_le_bytes());
        for token in tokens {
            hasher.update(token.to_le_bytes());
        }
        hasher.update((extra_keys.len() as u64).to_le_bytes());
        for key in extra_keys {
            hasher.update(key.to_le_bytes());
        }
        hasher.update(if block_index == 0 { cache_salt } else { 0 }.to_le_bytes());
        hasher.finalize().into()
    }

    /// Adding the group discriminant must not move a single KV key: an
    /// existing chain on disk (and the hot-chain contract the adapter mirrors)
    /// still derives to exactly the same bytes.
    #[test]
    fn kv_group_key_is_byte_identical_to_pre_group_derivation() {
        let fp = fingerprint();
        // Frozen golden value for the canonical first block, so a future edit
        // to the hashed component order fails here even if the reference
        // implementation below were edited alongside it.
        assert_eq!(
            ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[], 0, 0).to_hex(),
            "150ac769fca99a77c26a4b3776143c1912d837c90fad2889719e83ef7896a6d7",
            "the KV key derivation is a persisted on-disk contract"
        );

        let parent = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[], 7, 0);
        for (parent, tokens, extra, salt, index) in [
            (None, vec![1u32, 2, 3, 4], vec![], 0u64, 0usize),
            (None, vec![1, 2, 3, 4], vec![9u64], 7, 0),
            (Some(parent), vec![5, 6, 7, 8], vec![9, 10], 7, 1),
            (Some(parent), vec![], vec![], u64::MAX, 3),
        ] {
            assert_eq!(
                ColdCacheKey::chain(ColdGroup::Kv, fp, parent, &tokens, &extra, salt, index)
                    .as_bytes(),
                &pre_group_chain(fp, parent, &tokens, &extra, salt, index),
                "ColdGroup::Kv must reproduce the pre-group derivation exactly"
            );
        }
    }

    /// vLLM folds the cache-group id into the block hash key
    /// (`BlockHashWithGroupId`) precisely so one group's entry can never be
    /// served for another's. Same inputs, different group ⇒ different key.
    #[test]
    fn groups_never_collide_for_identical_inputs() {
        let fp = fingerprint();
        let groups = [ColdGroup::Kv, ColdGroup::GdnState, ColdGroup::SlidingWindow];
        let parent = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[], 0, 0);
        for (parent, tokens, extra, salt, index) in [
            (None, vec![1u32, 2, 3, 4], vec![], 0u64, 0usize),
            (Some(parent), vec![5, 6, 7, 8], vec![11u64], 3, 1),
        ] {
            let keys: Vec<ColdCacheKey> = groups
                .iter()
                .map(|&group| ColdCacheKey::chain(group, fp, parent, &tokens, &extra, salt, index))
                .collect();
            for i in 0..keys.len() {
                for j in (i + 1)..keys.len() {
                    assert_ne!(
                        keys[i], keys[j],
                        "{:?} and {:?} must not share a key",
                        groups[i], groups[j]
                    );
                }
            }
        }
        // Domain tags must also stay pairwise distinct as literals, since key
        // separation rests entirely on them.
        let tags: Vec<&[u8]> = groups.iter().map(|g| g.domain_tag()).collect();
        for i in 0..tags.len() {
            for j in (i + 1)..tags.len() {
                assert_ne!(tags[i], tags[j]);
            }
        }
    }

    /// A sidecar lives in its own filename namespace, so it can never be
    /// opened — let alone decoded — through the KV block path.
    #[test]
    fn sidecar_names_are_disjoint_from_block_names() {
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let kv_name = object_file_name(&key, ColdGroup::Kv);
        assert_eq!(kv_name, format!("{}.safetensors", key.to_hex()));
        assert_eq!(parse_object_name(&kv_name), Some((key, ColdGroup::Kv)));

        for group in ColdGroup::SIDECAR_GROUPS {
            let name = object_file_name(&key, group);
            assert_ne!(name, kv_name);
            assert_eq!(
                name,
                format!("{}.{}.safetensors", key.to_hex(), group.label())
            );
            assert_eq!(parse_object_name(&name), Some((key, group)));
        }

        // Non-canonical shapes are never adopted by the index scanner.
        assert_eq!(
            parse_object_name(&format!("{}.kv.safetensors", key.to_hex())),
            None
        );
        assert_eq!(
            parse_object_name(&format!("{}.bogus.safetensors", key.to_hex())),
            None
        );
        assert_eq!(
            parse_object_name(&format!("{}.safetensors.tmp", key.to_hex())),
            None
        );
        assert_eq!(parse_object_name("not-a-key.gdn_state.safetensors"), None);
    }

    #[test]
    fn sidecar_roundtrip_preserves_dtype_and_dims() {
        let fp = fingerprint();
        let key = ColdCacheKey::chain(ColdGroup::GdnState, fp, None, &[1, 2, 3, 4], &[], 0, 0);
        let original = sidecar(key, ColdGroup::GdnState);
        let encoded = encode_sidecar(&original).unwrap();
        assert!(
            encoded.len() as u64 <= max_encoded_len_for_sidecar(&original.layout).unwrap(),
            "the read bound must be a true upper bound on the encoder output"
        );

        let decoded = decode_sidecar(&encoded, key, fp, ColdGroup::GdnState).unwrap();
        assert_eq!(decoded, original);
        assert_eq!(decoded.layout.dtype, "BFloat16");
        assert_eq!(decoded.layout.dims, vec![4, 8, 2]);
        assert_eq!(decoded.layout.boundary_tokens, 16);
        assert_eq!(decoded.layout.tensors_per_layer, 2);

        // Wrong group, wrong key, wrong fingerprint: all refused.
        assert!(decode_sidecar(&encoded, key, fp, ColdGroup::SlidingWindow).is_err());
        assert!(decode_sidecar(&encoded, key, fp, ColdGroup::Kv).is_err());
        let other = ColdCacheKey::chain(ColdGroup::GdnState, fp, None, &[9, 9, 9, 9], &[], 0, 0);
        assert!(decode_sidecar(&encoded, other, fp, ColdGroup::GdnState).is_err());
        let other_fp = ColdCacheFingerprint::from_components([b"other".as_slice()]);
        assert!(decode_sidecar(&encoded, key, other_fp, ColdGroup::GdnState).is_err());

        // A corrupt payload byte fails the checksum.
        let mut corrupt = encoded.clone();
        *corrupt.last_mut().unwrap() ^= 0xff;
        assert!(decode_sidecar(&corrupt, key, fp, ColdGroup::GdnState).is_err());

        // The two object types are mutually undecodable even with matching
        // identity metadata: neither decoder can be fed the other's bytes.
        assert!(decode_block(&encoded, key, fp).is_err());
        let kv_key = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[], 0, 0);
        let block_bytes = encode_block(&block(kv_key)).unwrap();
        assert!(decode_sidecar(&block_bytes, kv_key, fp, ColdGroup::GdnState).is_err());
    }

    #[test]
    fn sidecar_rejects_out_of_range_geometry() {
        let key = ColdCacheKey::chain(ColdGroup::GdnState, fingerprint(), None, &[1], &[], 0, 0);
        let mut kv_group = sidecar(key, ColdGroup::GdnState);
        kv_group.layout.group = ColdGroup::Kv;
        assert!(
            kv_group.validate().is_err(),
            "sidecars must not use the KV group"
        );

        let mut zero_boundary = sidecar(key, ColdGroup::GdnState);
        zero_boundary.layout.boundary_tokens = 0;
        assert!(zero_boundary.validate().is_err());

        let mut too_many = sidecar(key, ColdGroup::GdnState);
        too_many.layout.tensors_per_layer = MAX_SIDECAR_TENSORS_PER_LAYER + 1;
        assert!(too_many.validate().is_err());
        assert_eq!(max_encoded_len_for_sidecar(&too_many.layout), None);

        let mut too_many_dims = sidecar(key, ColdGroup::GdnState);
        too_many_dims.layout.dims = vec![1; MAX_SIDECAR_DIMS + 1];
        assert!(too_many_dims.validate().is_err());

        let mut short_tensor = sidecar(key, ColdGroup::GdnState);
        short_tensor.tensors[1].pop();
        assert!(short_tensor.validate().is_err());

        let mut missing_tensor = sidecar(key, ColdGroup::GdnState);
        missing_tensor.tensors.pop();
        assert!(missing_tensor.validate().is_err());
    }

    /// A policy is a geometry TEMPLATE, not a boundary: the boundary is the one
    /// layout field that varies per candidate prefix, so the constructor drops
    /// whatever was passed and `expected_at` stamps in the candidate's own. Any
    /// geometry a sidecar could never legally be written with is refused up
    /// front, so an impossible policy cannot be installed and then silently
    /// suppress every restore forever.
    #[test]
    fn sidecar_policy_is_a_boundary_free_geometry_template() {
        let policy = ColdSidecarPolicy::new(ColdSidecarLayout {
            boundary_tokens: 4096,
            ..sidecar_layout(ColdGroup::GdnState)
        })
        .expect("a legal geometry must build a policy");
        assert_eq!(policy.group(), ColdGroup::GdnState);
        assert_eq!(
            policy.expected_at(32),
            ColdSidecarLayout {
                boundary_tokens: 32,
                ..sidecar_layout(ColdGroup::GdnState)
            },
            "the constructor's boundary must be dropped and the candidate's used"
        );

        // KV is not an auxiliary group: a policy in it would mint keys that
        // collide with the block namespace.
        assert!(ColdSidecarPolicy::new(sidecar_layout(ColdGroup::Kv)).is_err());
        // Every geometry bound `ColdSidecar::validate` enforces applies here.
        assert!(
            ColdSidecarPolicy::new(ColdSidecarLayout {
                tensors_per_layer: MAX_SIDECAR_TENSORS_PER_LAYER + 1,
                ..sidecar_layout(ColdGroup::GdnState)
            })
            .is_err()
        );
        assert!(
            ColdSidecarPolicy::new(ColdSidecarLayout {
                dims: Vec::new(),
                ..sidecar_layout(ColdGroup::GdnState)
            })
            .is_err()
        );
        assert!(
            ColdSidecarPolicy::new(ColdSidecarLayout {
                bytes_per_tensor: 0,
                ..sidecar_layout(ColdGroup::GdnState)
            })
            .is_err()
        );
        assert!(
            ColdSidecarPolicy::new(ColdSidecarLayout {
                num_layers: 0,
                ..sidecar_layout(ColdGroup::GdnState)
            })
            .is_err()
        );
    }

    /// A forged sidecar header must never size an allocation: the tensor
    /// count is checked against what actually deserialized, exactly as the
    /// block decoder checks `1 + 2*num_layers`. The test returning at all
    /// proves no multi-GB reservation happened.
    #[test]
    fn sidecar_decode_rejects_forged_counts() {
        let key = ColdCacheKey::chain(ColdGroup::GdnState, fingerprint(), None, &[1], &[], 0, 0);
        let payload: Vec<u8> = vec![7; 6];
        let view = TensorView::new(Dtype::U8, vec![payload.len()], &payload).unwrap();
        let forged = |num_layers: &str, per_layer: &str| {
            let mut metadata = HashMap::new();
            metadata.insert("abi".to_string(), CACHE_ABI.to_string());
            metadata.insert("group".to_string(), ColdGroup::GdnState.label().to_string());
            metadata.insert("key".to_string(), key.to_hex());
            metadata.insert("fingerprint".to_string(), fingerprint().to_hex());
            metadata.insert("checksum".to_string(), "unused".to_string());
            metadata.insert("boundary_tokens".to_string(), "16".to_string());
            metadata.insert("num_layers".to_string(), num_layers.to_string());
            metadata.insert("tensors_per_layer".to_string(), per_layer.to_string());
            metadata.insert("dtype".to_string(), "BFloat16".to_string());
            metadata.insert("dims".to_string(), "4,8,2".to_string());
            metadata.insert("bytes_per_tensor".to_string(), "6".to_string());
            serialize(
                vec![(sidecar_tensor_name(0, 0).as_str(), view.clone())],
                Some(metadata),
            )
            .unwrap()
        };

        for (layers, per_layer) in [
            (u32::MAX.to_string(), "16".to_string()),
            (u32::MAX.to_string(), u32::MAX.to_string()),
            ("1".to_string(), "0".to_string()),
            ("2".to_string(), "1".to_string()),
        ] {
            assert!(
                decode_sidecar(
                    &forged(&layers, &per_layer),
                    key,
                    fingerprint(),
                    ColdGroup::GdnState
                )
                .is_err(),
                "forged counts ({layers}, {per_layer}) must be rejected before allocating"
            );
        }
    }

    /// End-to-end fail-closed contract: a sidecar that lands on disk and is
    /// then truncated must MISS, count exactly one corruption, and have its
    /// file pruned — never panic, and never hand back partial state.
    #[test]
    fn truncated_sidecar_is_a_graceful_miss_that_prunes_and_counts_corruption() {
        let root = temp_root("sidecar-truncated");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let fp = fingerprint();
        let key = ColdCacheKey::chain(ColdGroup::GdnState, fp, None, &[1, 2, 3, 4], &[], 0, 0);
        let expected = sidecar(key, ColdGroup::GdnState);

        assert!(manager.enqueue_sidecar(expected.clone()).unwrap());
        assert!(manager.drain(Duration::from_secs(5)));

        let path = root.join(object_file_name(&key, ColdGroup::GdnState));
        assert!(path.exists(), "the sidecar must land under its own name");
        assert!(
            !root.join(format!("{}.safetensors", key.to_hex())).exists(),
            "a sidecar must never occupy the KV block name"
        );
        assert!(manager.contains_in(&key, ColdGroup::GdnState));
        assert!(!manager.contains(&key), "a sidecar is not a KV block");
        assert_eq!(
            manager.load_sidecar(key, fp, &sidecar_layout(ColdGroup::GdnState)),
            Some(expected.clone())
        );

        // Truncate in place (same inode), so pruning sees the very entry that
        // failed and is allowed to clear it.
        let bytes = fs::read(&path).unwrap();
        fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();

        let before = manager.stats();
        assert_eq!(
            manager.load_sidecar(key, fp, &sidecar_layout(ColdGroup::GdnState)),
            None,
            "a truncated sidecar must miss, not panic or return partial state"
        );
        let after = manager.stats();
        assert_eq!(after.corruptions, before.corruptions + 1);
        assert_eq!(after.misses, before.misses + 1);
        assert!(!path.exists(), "the corrupt sidecar file must be pruned");
        assert!(!manager.contains_in(&key, ColdGroup::GdnState));

        // A sidecar asked for under the wrong group, or with a layout that
        // does not match what was written, is likewise a miss.
        assert_eq!(
            manager.load_sidecar(key, fp, &sidecar_layout(ColdGroup::Kv)),
            None
        );

        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    /// A sidecar whose on-disk geometry differs from what the caller expects
    /// is a miss, not a reinterpretation — the sidecar analogue of
    /// `layout_matches_pool`.
    #[test]
    fn sidecar_layout_mismatch_is_a_miss() {
        let root = temp_root("sidecar-layout-mismatch");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let fp = fingerprint();
        let key = ColdCacheKey::chain(ColdGroup::GdnState, fp, None, &[1, 2, 3, 4], &[], 0, 0);
        assert!(
            manager
                .enqueue_sidecar(sidecar(key, ColdGroup::GdnState))
                .unwrap()
        );
        assert!(manager.drain(Duration::from_secs(5)));

        let mut wrong_dtype = sidecar_layout(ColdGroup::GdnState);
        wrong_dtype.dtype = "Float16".to_string();
        assert_eq!(manager.load_sidecar(key, fp, &wrong_dtype), None);

        let mut wrong_dims = sidecar_layout(ColdGroup::GdnState);
        wrong_dims.dims = vec![4, 8, 3];
        assert_eq!(manager.load_sidecar(key, fp, &wrong_dims), None);

        let mut wrong_boundary = sidecar_layout(ColdGroup::GdnState);
        wrong_boundary.boundary_tokens = 32;
        assert_eq!(manager.load_sidecar(key, fp, &wrong_boundary), None);

        // The file survives every mismatch: it is valid, just not what this
        // caller asked for.
        assert!(
            root.join(object_file_name(&key, ColdGroup::GdnState))
                .exists()
        );
        assert_eq!(
            manager.load_sidecar(key, fp, &sidecar_layout(ColdGroup::GdnState)),
            Some(sidecar(key, ColdGroup::GdnState))
        );

        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    /// `layout_matches_pool` must reject a block whose per-layer K/V byte
    /// lengths disagree with the pool, instead of leaving it to
    /// `write_blocks_from_host` — by then a physical block is allocated and
    /// earlier layers are already uploaded.
    #[cfg(target_os = "macos")]
    #[test]
    fn layout_mismatch_on_layer_bytes_is_rejected_at_validation() {
        use crate::PagedAttentionConfig;
        use crate::metal::MetalDtype;

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
                eprintln!("skipping layout_mismatch_on_layer_bytes_is_rejected_at_validation: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };

        // The pool's own per-layer byte lengths, cross-checked against what
        // `read_blocks_to_host` actually produces.
        let (key_bytes, value_bytes) = pool_layer_bytes(&pool).unwrap();
        let (keys, values) = pool.read_blocks_to_host(0, &[0]).unwrap();
        assert_eq!((keys.len(), values.len()), (key_bytes, value_bytes));

        let matching = ColdCacheLayout {
            block_size: pool.block_size(),
            num_layers: pool.num_layers() as u32,
            num_kv_heads: pool.config().num_kv_heads,
            head_size: pool.config().head_size,
            cache_dtype: format!("{:?}", pool.cache_dtype()),
            key_bytes_per_layer: key_bytes,
            value_bytes_per_layer: value_bytes,
        };
        assert!(layout_matches_pool(&matching, &pool));

        let mut wrong_keys = matching.clone();
        wrong_keys.key_bytes_per_layer = key_bytes / 2;
        assert!(
            !layout_matches_pool(&wrong_keys, &pool),
            "a key_bytes mismatch must fail validation, not the upload"
        );

        let mut wrong_values = matching.clone();
        wrong_values.value_bytes_per_layer = value_bytes + 2;
        assert!(
            !layout_matches_pool(&wrong_values, &pool),
            "a value_bytes mismatch must fail validation, not the upload"
        );
    }

    /// Sidecars occupy quota like any other object, so the startup scan must
    /// index them — an unaccounted file would sit outside eviction forever.
    #[test]
    fn sidecars_are_indexed_and_accounted_across_restart() {
        let root = temp_root("sidecar-accounting");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let fp = fingerprint();
        let kv_key = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[], 0, 0);
        let side_key = ColdCacheKey::chain(ColdGroup::GdnState, fp, None, &[1, 2, 3, 4], &[], 0, 0);
        assert!(manager.enqueue(block(kv_key)).unwrap());
        assert!(
            manager
                .enqueue_sidecar(sidecar(side_key, ColdGroup::GdnState))
                .unwrap()
        );
        assert!(manager.drain(Duration::from_secs(5)));
        drop(manager);

        let reopened = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let index = reopened.shared.index.lock().unwrap();
        assert_eq!(index.entries.len(), 2, "both objects must be indexed");
        assert_eq!(index.entries[&kv_key].group, ColdGroup::Kv);
        assert_eq!(index.entries[&side_key].group, ColdGroup::GdnState);
        let on_disk: u64 = [kv_key, side_key]
            .iter()
            .map(|key| index.entries[key].size)
            .sum();
        assert_eq!(index.total_bytes, on_disk);
        drop(index);
        drop(reopened);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn stable_chain_is_parent_and_fingerprint_sensitive() {
        let fp = fingerprint();
        let first = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[], 0, 0);
        assert_eq!(
            first,
            ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[], 0, 0)
        );
        assert_ne!(
            first,
            ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 5], &[], 0, 0)
        );
        assert_ne!(
            ColdCacheKey::chain(ColdGroup::Kv, fp, Some(first), &[5, 6, 7, 8], &[], 0, 1),
            ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[5, 6, 7, 8], &[], 0, 1)
        );
    }

    #[test]
    fn safetensors_roundtrip_and_checksum() {
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
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
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let mut partial = block(key);
        partial.tokens.pop();
        assert!(partial.validate().is_err());
    }

    #[test]
    fn writer_is_atomic_and_index_rebuilds() {
        let root = temp_root("roundtrip");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
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

    // A shutdown drain must guarantee every ACCEPTED (`Ok(true)`) block is on
    // disk before it returns, even when more blocks were pushed than the queue
    // depth so the barrier has to wait behind in-flight writes. `persist_block`
    // is filesystem-only, so this exercises the full FIFO ordering contract
    // without Metal.
    #[test]
    fn drain_flushes_accepted_blocks_before_returning() {
        let root = temp_root("drain-accepted");
        // Queue depth 2 with 8 rapid enqueues forces the barrier to sit behind
        // blocks the writer has not yet persisted.
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let mut accepted = Vec::new();
        for i in 0..8u32 {
            let toks = vec![i, i + 100, i + 200, i + 300];
            let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &toks, &[], 0, 0);
            let mut candidate = block(key);
            candidate.tokens = toks;
            // Non-blocking enqueue may drop under a momentarily full queue; only
            // ACCEPTED blocks carry the drain durability guarantee.
            if manager.enqueue(candidate).unwrap() {
                accepted.push(key);
            }
        }

        assert!(
            manager.drain(Duration::from_secs(5)),
            "drain must ack within the timeout"
        );
        for key in &accepted {
            let path = root.join(format!("{}.safetensors", key.to_hex()));
            assert!(
                path.exists(),
                "every accepted block must be fsynced to disk before drain returns"
            );
        }
        // The barrier is one-shot: a second drain over a now-idle writer also
        // returns true.
        assert!(manager.drain(Duration::from_secs(5)));
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn drain_returns_true_when_empty() {
        let root = temp_root("drain-empty");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        assert!(
            manager.drain(Duration::from_secs(5)),
            "an empty tier drains immediately"
        );
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    // A drain must report durability HONESTLY: if any block it covers failed
    // to persist, the barrier ack is `false`, so `drain` returns `false`
    // rather than falsely reporting the write as durable. The dir-fsync
    // override forces `persist_block` to return `Err` after the rename, the
    // same failure seam the post-rename test uses.
    #[test]
    fn drain_reports_false_when_a_covered_block_fails_to_persist() {
        let root = temp_root("drain-persist-fail");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        *manager.shared.dir_sync_override.lock().unwrap() =
            Some(Box::new(|| Err("injected dir fsync failure".to_string())));

        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        assert!(
            manager.enqueue(block(key)).unwrap(),
            "the block must be accepted so the barrier covers it"
        );

        assert!(
            !manager.drain(Duration::from_secs(5)),
            "a covered block that failed to persist must make drain report false"
        );

        *manager.shared.dir_sync_override.lock().unwrap() = None;
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    // The WHOLE drain must stay bounded by `timeout` even when the bounded
    // queue is full and the writer is stuck mid-persist: barrier admission is
    // deadline-aware `try_send`, not a blocking `send` that could outlast the
    // timeout or hang exit. A safety timer releases the writer well after the
    // short timeout so a regression (blocking admission) terminates instead of
    // hanging the suite.
    #[test]
    fn drain_is_bounded_by_timeout_under_a_saturated_queue() {
        use std::sync::atomic::AtomicBool;

        let root = temp_root("drain-bounded");
        let depth = 2usize;
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, depth).unwrap();

        // Park the writer inside `persist_block`'s dir fsync so it consumes
        // exactly one block and then stops draining the queue.
        let release = Arc::new(AtomicBool::new(false));
        let release_writer = Arc::clone(&release);
        *manager.shared.dir_sync_override.lock().unwrap() = Some(Box::new(move || {
            while !release_writer.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(())
        }));

        let make_block = |i: u32| {
            let toks = vec![i, i + 100, i + 200, i + 300];
            let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &toks, &[], 0, 0);
            let mut candidate = block(key);
            candidate.tokens = toks;
            candidate
        };

        // First block is dequeued by the writer, which then parks in the
        // fsync override; give it a moment to get there.
        assert!(manager.enqueue(make_block(0)).unwrap());
        std::thread::sleep(Duration::from_millis(100));

        // Fill the bounded buffer until enqueue starts dropping — a drop proves
        // the queue is saturated behind the parked writer.
        let mut saturated = false;
        for i in 1..(depth as u32 + 6) {
            if !manager.enqueue(make_block(i)).unwrap() {
                saturated = true;
                break;
            }
        }
        assert!(saturated, "the bounded queue must be full before draining");

        // Safety net: releases the writer long after the short drain timeout,
        // so even a regressed blocking drain unblocks rather than hanging.
        let release_timer = Arc::clone(&release);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(3));
            release_timer.store(true, Ordering::Relaxed);
        });

        let timeout = Duration::from_millis(200);
        let start = Instant::now();
        let drained = manager.drain(timeout);
        let elapsed = start.elapsed();

        assert!(
            !drained,
            "a saturated queue behind a stuck writer cannot drain within the timeout"
        );
        assert!(
            elapsed < timeout + Duration::from_millis(500),
            "drain must stay bounded by the timeout, took {elapsed:?}"
        );

        // Release the writer so teardown drains cleanly.
        release.store(true, Ordering::Relaxed);
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    // A post-rename directory fsync failure must leave in-process accounting
    // consistent with the on-disk canonical file: the rename is the true
    // commit point (the payload was already `sync_all`'d), so the index entry
    // and its byte credit belong to a renamed block even when the durability
    // barrier afterwards fails. Otherwise the file is orphaned outside the
    // quota until the next `rebuild_index` re-credits it on restart.
    #[test]
    fn post_rename_dir_sync_failure_keeps_index_consistent() {
        let root = temp_root("dir-sync-fail");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let expected = block(key);
        // The index credits the actual encoded byte length, matching
        // `persist_block`'s `size = bytes.len()` (not the `encoded_len`
        // upper bound used for read-time allocation caps).
        let size = encode_block(&expected).unwrap().len() as u64;

        *manager.shared.dir_sync_override.lock().unwrap() =
            Some(Box::new(|| Err("injected dir fsync failure".to_string())));

        let result = persist_block(&manager.shared, &expected);
        assert!(
            result.is_err(),
            "the injected dir fsync error must surface to the fail-open worker"
        );

        let path = root.join(format!("{}.safetensors", key.to_hex()));
        assert!(
            path.exists(),
            "the renamed canonical file survives a post-rename fsync failure"
        );

        let index = manager.shared.index.lock().unwrap();
        assert!(
            index.entries.contains_key(&key),
            "the index entry must be published for the renamed canonical file"
        );
        assert_eq!(
            index.total_bytes, size,
            "the renamed block must be credited so it stays inside the quota"
        );
        drop(index);

        *manager.shared.dir_sync_override.lock().unwrap() = None;
        drop(manager);
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
        let key_a = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[1], 0, 0);
        let key_b = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[2], 0, 0);
        let key_c = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[3], 0, 0);
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
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
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
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
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

    #[cfg(unix)]
    #[test]
    fn load_miss_after_symlink_swap_prunes_index_and_spares_target() {
        let base = temp_root("symlink-swap-entry");
        let root = base.join("root");
        let victim = base.join("victim.bin");
        fs::create_dir_all(&base).unwrap();
        fs::write(&victim, b"victim payload").unwrap();

        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
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
        std::os::unix::fs::symlink(&victim, &path).unwrap();

        assert!(manager.load(key, fingerprint()).is_none());
        assert!(
            !manager.contains(&key),
            "an entry replaced by a symlink must leave the index on the next load miss"
        );
        assert_eq!(manager.shared.index.lock().unwrap().total_bytes, 0);
        assert!(
            fs::symlink_metadata(&path).is_err(),
            "the dead symlink directory entry itself must be unlinked"
        );
        assert_eq!(
            fs::read(&victim).unwrap(),
            b"victim payload",
            "the symlink target must never be followed or unlinked"
        );
        let stats = manager.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.corruptions, 0, "nothing was opened, nothing was read");

        assert!(
            manager.enqueue(block(key)).unwrap(),
            "pruning must unblock re-persisting the same key"
        );
        for _ in 0..200 {
            if manager.contains(&key) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(manager.load(key, fingerprint()), Some(block(key)));
        drop(manager);
        let _ = fs::remove_dir_all(base);
    }

    #[cfg(unix)]
    #[test]
    fn load_returns_promptly_when_entry_replaced_by_fifo() {
        use std::os::unix::ffi::OsStrExt;

        let root = temp_root("fifo-swap");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
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
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        // SAFETY: c_path is a valid NUL-terminated path for the whole call.
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);

        // A blocking read-only open of a writerless FIFO parks forever, so a
        // regression must fail this timeout instead of hanging the suite.
        let manager = Arc::new(manager);
        let (done, loaded) = std::sync::mpsc::channel();
        let loader = Arc::clone(&manager);
        std::thread::spawn(move || {
            let _ = done.send(loader.load(key, fingerprint()));
        });
        let result = loaded
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("load of a FIFO-swapped entry must return promptly, not block for a writer");
        assert!(result.is_none());

        assert!(
            !manager.contains(&key),
            "an entry replaced by a FIFO must leave the index on the load miss"
        );
        assert_eq!(manager.shared.index.lock().unwrap().total_bytes, 0);
        assert!(
            fs::symlink_metadata(&path).is_err(),
            "the dead FIFO directory entry itself must be unlinked"
        );
        let stats = manager.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.corruptions, 0, "a type mismatch is not a corruption");

        assert!(
            manager.enqueue(block(key)).unwrap(),
            "pruning must unblock re-persisting the same key"
        );
        for _ in 0..200 {
            if manager.contains(&key) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(manager.load(key, fingerprint()), Some(block(key)));
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn load_miss_after_empty_dir_swap_removes_dir_and_unblocks_key() {
        let root = temp_root("empty-dir-swap");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
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
        fs::create_dir(&path).unwrap();

        assert!(manager.load(key, fingerprint()).is_none());
        assert!(
            fs::symlink_metadata(&path).is_err(),
            "an empty directory swapped onto the canonical name must be removed"
        );
        assert!(
            !manager.contains(&key),
            "the cleared entry must leave the index on the load miss"
        );
        assert_eq!(manager.shared.index.lock().unwrap().total_bytes, 0);
        let stats = manager.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.corruptions, 0, "nothing was opened, nothing was read");

        assert!(
            manager.enqueue(block(key)).unwrap(),
            "removing the directory must unblock re-persisting the same key"
        );
        for _ in 0..200 {
            if manager.contains(&key) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(manager.load(key, fingerprint()), Some(block(key)));
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn load_miss_after_nonempty_dir_swap_quarantines_without_deleting_content() {
        let root = temp_root("nonempty-dir-swap");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        assert!(manager.enqueue(block(key)).unwrap());
        for _ in 0..200 {
            if manager.contains(&key) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(manager.contains(&key));

        let name = format!("{}.safetensors", key.to_hex());
        let path = root.join(&name);
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        fs::write(path.join("marker.txt"), b"marker").unwrap();

        assert!(manager.load(key, fingerprint()).is_none());
        assert!(
            fs::symlink_metadata(&path).is_err(),
            "the canonical name must be freed on the load miss"
        );
        assert!(!manager.contains(&key));
        assert_eq!(manager.shared.index.lock().unwrap().total_bytes, 0);

        let quarantine_prefix = format!(".blocked.{name}.");
        let quarantined: Vec<PathBuf> = fs::read_dir(&root)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|n| n.starts_with(&quarantine_prefix))
            })
            .map(|entry| entry.path())
            .collect();
        assert_eq!(
            quarantined.len(),
            1,
            "the obstructing directory must be renamed aside, not deleted"
        );
        assert_eq!(
            fs::read(quarantined[0].join("marker.txt")).unwrap(),
            b"marker",
            "quarantine must preserve the directory's content"
        );

        assert!(
            manager.enqueue(block(key)).unwrap(),
            "quarantining must unblock re-persisting the same key"
        );
        for _ in 0..200 {
            if manager.contains(&key) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(manager.load(key, fingerprint()), Some(block(key)));
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn load_miss_after_unreadable_entry_clears_index_and_name() {
        use std::os::unix::fs::PermissionsExt;

        let root = temp_root("unreadable-entry");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        persist_block(&manager.shared, &block(key)).unwrap();
        assert!(manager.contains(&key));

        let path = root.join(format!("{}.safetensors", key.to_hex()));
        fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).unwrap();

        assert!(manager.load(key, fingerprint()).is_none());
        assert!(
            !manager.contains(&key),
            "an unopenable entry must leave the index on the load miss"
        );
        assert!(
            fs::symlink_metadata(&path).is_err(),
            "the unopenable file must be unlinked from the canonical name"
        );
        assert_eq!(manager.shared.index.lock().unwrap().total_bytes, 0);
        let stats = manager.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.corruptions, 0, "nothing was opened, nothing was read");

        persist_block(&manager.shared, &block(key))
            .expect("clearing must unblock re-persisting the same key");
        assert_eq!(manager.load(key, fingerprint()), Some(block(key)));
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn load_bounded_rejects_oversized_entry_without_unbounded_alloc() {
        let root = temp_root("bounded-oversized");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let good = block(key);
        // Tight, geometry-derived cap: the legit entry's encoded upper bound.
        let max_encoded = good.encoded_len();
        persist_block(&manager.shared, &good).unwrap();
        assert!(manager.contains(&key));
        assert_eq!(
            manager.load_bounded(key, fingerprint(), max_encoded),
            Some(good.clone()),
            "the legitimate entry must still load within its own encoded bound"
        );

        // Replace the committed entry with a huge SPARSE regular file:
        // `st_size` reports gigabytes but no data blocks are allocated. An
        // unbounded `read_to_end` would try to allocate that many bytes; the
        // bounded read must cap the allocation and treat it as corruption.
        let path = root.join(format!("{}.safetensors", key.to_hex()));
        fs::remove_file(&path).unwrap();
        let huge = fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)
            .unwrap();
        huge.set_len(8 * GIB).unwrap();
        drop(huge);

        let before = manager.stats().corruptions;
        assert!(
            manager
                .load_bounded(key, fingerprint(), max_encoded)
                .is_none(),
            "an entry exceeding the geometry bound must miss, not slurp gigabytes"
        );
        let after = manager.stats();
        assert_eq!(
            after.corruptions,
            before + 1,
            "an oversized entry counts as a corruption, like any decode failure"
        );
        assert_eq!(after.misses, 1);
        assert!(
            !manager.contains(&key),
            "the oversized entry must be pruned from the index on the miss"
        );
        assert!(
            fs::symlink_metadata(&path).is_err(),
            "the oversized file must be cleared from the canonical name"
        );

        // Pruning must unblock re-persisting the same key, which then loads
        // back cleanly through the geometry-free public wrapper.
        persist_block(&manager.shared, &good)
            .expect("clearing must unblock re-persisting the same key");
        assert_eq!(manager.load(key, fingerprint()), Some(good));
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn eviction_quarantines_directory_swapped_onto_lru_entry() {
        let root = temp_root("evict-dir-swap");
        let fp = fingerprint();
        let key_a = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[1], 0, 0);
        let key_b = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[2], 0, 0);
        let one = encode_block(&block(key_a)).unwrap().len() as u64;
        let manager = ColdCacheManager::open_at(root.clone(), one + one / 2, 0, 2).unwrap();

        persist_block(&manager.shared, &block(key_a)).unwrap();
        assert!(manager.contains(&key_a));

        let name_a = format!("{}.safetensors", key_a.to_hex());
        let path_a = root.join(&name_a);
        fs::remove_file(&path_a).unwrap();
        fs::create_dir(&path_a).unwrap();
        fs::write(path_a.join("marker.txt"), b"marker").unwrap();

        persist_block(&manager.shared, &block(key_b))
            .expect("eviction must clear the obstructed LRU name and let the write proceed");
        assert!(manager.contains(&key_b));
        assert!(!manager.contains(&key_a));
        assert!(
            fs::symlink_metadata(&path_a).is_err(),
            "the canonical name must actually be clear after the eviction pass"
        );

        let quarantine_prefix = format!(".blocked.{name_a}.");
        let quarantined: Vec<PathBuf> = fs::read_dir(&root)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|n| n.starts_with(&quarantine_prefix))
            })
            .map(|entry| entry.path())
            .collect();
        assert_eq!(
            quarantined.len(),
            1,
            "the obstructing directory must be quarantined, not deleted or left in place"
        );
        assert_eq!(
            fs::read(quarantined[0].join("marker.txt")).unwrap(),
            b"marker",
            "quarantine must preserve the directory's content"
        );

        let stats = manager.stats();
        assert_eq!(
            stats.evictions, 1,
            "only the actually-cleared entry may count as an eviction"
        );
        assert_eq!(
            manager.shared.index.lock().unwrap().total_bytes,
            one,
            "byte accounting must reflect exactly the surviving entry"
        );

        persist_block(&manager.shared, &block(key_a))
            .expect("subsequent writes to the freed key must succeed");
        assert_eq!(manager.load(key_a, fp), Some(block(key_a)));
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn eviction_retains_unclearable_entry_and_terminates() {
        use std::os::unix::fs::PermissionsExt;

        let root = temp_root("evict-unclearable");
        let fp = fingerprint();
        let key_a = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[1], 0, 0);
        let key_b = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[2], 0, 0);
        let one = encode_block(&block(key_a)).unwrap().len() as u64;
        let manager = ColdCacheManager::open_at(root.clone(), one + one / 2, 0, 2).unwrap();

        persist_block(&manager.shared, &block(key_a)).unwrap();
        assert!(manager.contains(&key_a));

        // A write-protected root makes every unlink fail: the pass must
        // skip the entry (keeping it indexed and counted) and end in an
        // error instead of spinning or falsifying accounting.
        fs::set_permissions(&root, fs::Permissions::from_mode(0o500)).unwrap();
        assert!(
            persist_block(&manager.shared, &block(key_b)).is_err(),
            "an eviction pass with no clearable candidate must fail the write"
        );
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();

        assert!(
            manager.contains(&key_a),
            "an entry whose name could not be cleared must stay indexed"
        );
        assert_eq!(manager.shared.index.lock().unwrap().total_bytes, one);
        assert_eq!(
            manager.stats().evictions,
            0,
            "a failed clearing must not count as an eviction"
        );
        assert_eq!(manager.load(key_a, fp), Some(block(key_a)));
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn failed_open_identity_snapshot_excludes_concurrent_writer() {
        use std::os::unix::fs::PermissionsExt;
        use std::time::Duration;

        let root = temp_root("failed-open-writer-race");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        persist_block(&manager.shared, &block(key)).unwrap();
        let size = encode_block(&block(key)).unwrap().len() as u64;

        let path = root.join(format!("{}.safetensors", key.to_hex()));
        fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).unwrap();

        let (start_tx, start_rx) = mpsc::channel::<()>();
        let (published_tx, published_rx) = mpsc::channel::<()>();
        let writer_shared = Arc::clone(&manager.shared);
        let replacement = block(key);
        let writer = std::thread::spawn(move || {
            start_rx.recv().unwrap();
            persist_block(&writer_shared, &replacement).unwrap();
            let _ = published_tx.send(());
        });
        *manager.shared.failed_open_identity_hook.lock().unwrap() = Some(Box::new(move || {
            let _ = start_tx.send(());
            // Unfixed, the writer publishes its replacement inside this
            // window and the recv succeeds; fixed, the writer blocks on the
            // index lock until the snapshot is done and the wait expires.
            let _ = published_rx.recv_timeout(Duration::from_secs(1));
        }));

        assert!(manager.load(key, fingerprint()).is_none());
        writer.join().unwrap();
        *manager.shared.failed_open_identity_hook.lock().unwrap() = None;

        assert!(
            manager.contains(&key),
            "the writer's replacement index entry must survive failed-load pruning"
        );
        assert!(
            path.exists(),
            "the writer's replacement file must survive failed-load pruning"
        );
        assert_eq!(
            manager.load(key, fingerprint()),
            Some(block(key)),
            "the surviving replacement must be loadable"
        );
        let stats = manager.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.corruptions, 0, "nothing was opened, nothing was read");
        assert_eq!(
            stats.bytes_written,
            size * 2,
            "both persisted generations must stay accounted"
        );
        assert_eq!(manager.shared.index.lock().unwrap().total_bytes, size);
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    fn dir_regular_bytes(dir: &Path) -> u64 {
        let Ok(entries) = fs::read_dir(dir) else {
            return 0;
        };
        let mut total = 0;
        for entry in entries.flatten() {
            let Ok(meta) = fs::symlink_metadata(entry.path()) else {
                continue;
            };
            if meta.is_file() {
                total += meta.len();
            } else if meta.is_dir() {
                total += dir_regular_bytes(&entry.path());
            }
        }
        total
    }

    /// Emulates a filesystem whose free space is `base` minus the regular
    /// bytes physically present under `root`: unlinking a file frees its
    /// size, quarantining a directory frees nothing, and an already-missing
    /// entry is already reflected — exactly the physics the reserve floor
    /// must respect.
    #[cfg(unix)]
    fn install_space_override(manager: &ColdCacheManager, root: &Path, base: &Arc<AtomicU64>) {
        let root = root.to_path_buf();
        let base = Arc::clone(base);
        *manager.shared.space_override.lock().unwrap() = Some(Box::new(move || {
            Ok((
                u64::MAX,
                base.load(Ordering::Relaxed)
                    .saturating_sub(dir_regular_bytes(&root)),
            ))
        }));
    }

    #[cfg(unix)]
    #[test]
    fn eviction_of_missing_entry_frees_nothing_and_keeps_reserve() {
        let root = temp_root("evict-missing-reserve");
        let fp = fingerprint();
        let key_a = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[1], 0, 0);
        let key_b = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[2], 0, 0);
        let one = encode_block(&block(key_a)).unwrap().len() as u64;
        let reserve = 5 * one;
        let manager = ColdCacheManager::open_at(root.clone(), GIB, reserve, 2).unwrap();
        let base = Arc::new(AtomicU64::new(reserve + 2 * one));
        install_space_override(&manager, &root, &base);

        persist_block(&manager.shared, &block(key_a)).unwrap();
        assert!(manager.contains(&key_a));
        fs::remove_file(root.join(format!("{}.safetensors", key_a.to_hex()))).unwrap();

        // Available space is now exactly the reserve: clearing the
        // already-missing LRU entry frees zero bytes, so the write must be
        // refused instead of dipping below the floor.
        base.store(reserve, Ordering::Relaxed);
        assert!(
            persist_block(&manager.shared, &block(key_b)).is_err(),
            "clearing an already-missing entry must not admit a write below the reserve"
        );
        assert!(
            !root
                .join(format!("{}.safetensors", key_b.to_hex()))
                .exists(),
            "the refused write must not land on disk"
        );
        assert!(
            !manager.contains(&key_a),
            "the dead index entry must still be pruned by the pass"
        );
        assert!(!manager.contains(&key_b));
        assert_eq!(manager.shared.index.lock().unwrap().total_bytes, 0);
        assert_eq!(manager.stats().evictions, 1);

        base.store(reserve + 2 * one, Ordering::Relaxed);
        persist_block(&manager.shared, &block(key_b))
            .expect("restored headroom must admit the write again");
        assert_eq!(manager.load(key_b, fp), Some(block(key_b)));
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn eviction_quarantine_frees_nothing_and_keeps_reserve() {
        let root = temp_root("evict-quarantine-reserve");
        let fp = fingerprint();
        let key_a = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[1], 0, 0);
        let key_b = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[2], 0, 0);
        let one = encode_block(&block(key_a)).unwrap().len() as u64;
        let reserve = 5 * one;
        let marker = b"marker";
        let manager = ColdCacheManager::open_at(root.clone(), GIB, reserve, 2).unwrap();
        let base = Arc::new(AtomicU64::new(reserve + 2 * one));
        install_space_override(&manager, &root, &base);

        persist_block(&manager.shared, &block(key_a)).unwrap();
        let name_a = format!("{}.safetensors", key_a.to_hex());
        let path_a = root.join(&name_a);
        fs::remove_file(&path_a).unwrap();
        fs::create_dir(&path_a).unwrap();
        fs::write(path_a.join("marker.txt"), marker).unwrap();

        // Quarantining the obstructing directory renames it aside without
        // freeing a byte, so the reserve floor must still refuse the write.
        base.store(reserve + marker.len() as u64, Ordering::Relaxed);
        assert!(
            persist_block(&manager.shared, &block(key_b)).is_err(),
            "a quarantine that frees no bytes must not admit a write below the reserve"
        );
        assert!(
            !root
                .join(format!("{}.safetensors", key_b.to_hex()))
                .exists(),
            "the refused write must not land on disk"
        );
        assert!(
            fs::symlink_metadata(&path_a).is_err(),
            "the canonical name must still be cleared by the pass"
        );
        assert!(!manager.contains(&key_a));
        assert_eq!(manager.shared.index.lock().unwrap().total_bytes, 0);
        assert_eq!(manager.stats().evictions, 1);
        let quarantine_prefix = format!(".blocked.{name_a}.");
        let quarantined: Vec<PathBuf> = fs::read_dir(&root)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|n| n.starts_with(&quarantine_prefix))
            })
            .map(|entry| entry.path())
            .collect();
        assert_eq!(quarantined.len(), 1);
        assert_eq!(fs::read(quarantined[0].join("marker.txt")).unwrap(), marker);

        base.store(reserve + marker.len() as u64 + 2 * one, Ordering::Relaxed);
        persist_block(&manager.shared, &block(key_b))
            .expect("restored headroom must admit the write again");
        assert_eq!(manager.load(key_b, fp), Some(block(key_b)));
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn eviction_of_regular_file_frees_space_and_admits_write() {
        let root = temp_root("evict-regular-reserve");
        let fp = fingerprint();
        let key_a = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[1], 0, 0);
        let key_b = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[2], 0, 0);
        let one = encode_block(&block(key_a)).unwrap().len() as u64;
        let reserve = 5 * one;
        let manager = ColdCacheManager::open_at(root.clone(), GIB, reserve, 2).unwrap();
        let base = Arc::new(AtomicU64::new(reserve + 2 * one));
        install_space_override(&manager, &root, &base);

        persist_block(&manager.shared, &block(key_a)).unwrap();

        // Unlinking the LRU file genuinely frees its bytes, which is
        // exactly enough to clear the reserve floor for the incoming write.
        base.store(reserve + one, Ordering::Relaxed);
        persist_block(&manager.shared, &block(key_b))
            .expect("a genuine regular-file eviction must still admit the write");
        assert!(!manager.contains(&key_a));
        assert!(manager.contains(&key_b));
        assert_eq!(manager.stats().evictions, 1);
        assert_eq!(manager.shared.index.lock().unwrap().total_bytes, one);
        assert_eq!(manager.load(key_b, fp), Some(block(key_b)));
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn startup_rebuild_ignores_quarantined_directories() {
        let root = temp_root("quarantine-rebuild");
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let quarantined = root.join(format!(".blocked.{}.safetensors.4242.7", key.to_hex()));
        fs::create_dir_all(&quarantined).unwrap();
        let marker = quarantined.join("marker.txt");
        fs::write(&marker, b"marker").unwrap();

        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 1).unwrap();
        assert!(
            quarantined.is_dir() && marker.exists(),
            "startup cleanup must never delete quarantined directories"
        );
        assert_eq!(
            manager.shared.index.lock().unwrap().entries.len(),
            0,
            "quarantined names must never be indexed"
        );
        assert!(!is_cold_cache_temp_file(
            quarantined.file_name().unwrap().to_str().unwrap()
        ));
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn writer_commit_rename_failure_counts_drop_and_removes_temp() {
        let root = temp_root("commit-rename-failure");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let path = root.join(format!("{}.safetensors", key.to_hex()));
        fs::create_dir(&path).unwrap();
        fs::write(path.join("marker.txt"), b"marker").unwrap();

        assert!(manager.enqueue(block(key)).unwrap());
        for _ in 0..200 {
            if manager.stats().queue_drops >= 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let stats = manager.stats();
        assert_eq!(
            stats.queue_drops, 1,
            "a failed commit rename must be counted, not silently discarded"
        );
        assert_eq!(stats.bytes_written, 0);
        assert!(!manager.contains(&key));
        assert!(
            !fs::read_dir(&root).unwrap().any(|entry| {
                entry
                    .unwrap()
                    .file_name()
                    .to_str()
                    .is_some_and(|n| n.ends_with(".tmp"))
            }),
            "the orphaned temp file must be removed after a failed commit"
        );
        assert!(path.is_dir() && path.join("marker.txt").exists());
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn failed_load_cleanup_spares_concurrent_writer_replacement() {
        let root = temp_root("replace-race");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
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
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
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
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
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
        let key_a = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[1], 0, 0);
        let key_b = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[2], 0, 0);
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
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
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
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &tokens, &[], 0, 0);
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

    /// A block that decodes cleanly but then fails a post-decode restore step is
    /// a real fall-back to ordinary prefill, so it must count exactly one miss
    /// and zero hits — the hit/bytes_restored accounting is reserved for a fully
    /// published prefix. The token-mismatch guard is the deterministic
    /// post-decode failure: the stored block decodes against its own
    /// key/fingerprint, but the caller's `identity.tokens` differ, so
    /// `restore_block` bails before allocating a physical block.
    #[cfg(target_os = "macos")]
    #[test]
    fn post_decode_restore_failure_counts_one_miss() {
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
                eprintln!("skipping post_decode_restore_failure_counts_one_miss: {e}");
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

        let root = temp_root("restore-miss");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &tokens, &[], 0, 0);
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

        // Same key + fingerprint, so `load_bounded` decodes the stored block
        // successfully (the decode itself counts neither hit nor miss). But the
        // caller's prefix tokens differ from the stored block's, so the
        // token-mismatch guard rejects the restore — the deterministic
        // post-decode failure this test targets.
        let mismatched_tokens = vec![9, 10, 11, 12, 13, 14, 15, 16];
        let identity = RestorePrefixIdentity {
            hot_hash: hash_tokens(&mismatched_tokens, 0, &[]),
            tokens: mismatched_tokens,
            parent_hot_hash: 0,
            extra_keys: vec![],
            cache_salt: 0,
            block_index: 0,
        };
        let restored = manager.restore_block(&pool, &allocator, key, fingerprint(), &identity);
        assert!(
            restored.is_none(),
            "a token mismatch must abort the restore (fall back to prefill)"
        );

        let stats = manager.stats();
        assert_eq!(
            stats.misses, 1,
            "a decoded-then-rejected block must count exactly one miss"
        );
        assert_eq!(stats.hits, 0, "no prefix was published, so no hit");
        assert_eq!(
            stats.bytes_restored, 0,
            "nothing was restored into the pool"
        );

        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    /// End-to-end multi-block restart: capture a two-block prefix, drop the
    /// manager, reopen the cache from disk (index rebuilt), and restore both
    /// blocks into a FRESH allocator + pool by mirroring the exact chain the
    /// restore hook uses — hot hashes from [`chain_hashes`], cold keys from
    /// [`ColdCacheKey::chain`]. Each restored block must be byte-identical to
    /// the captured source, and the fresh hot cache must then serve the whole
    /// prefix. This is the cold_cache-level proof for Task A4's restore loop.
    #[cfg(target_os = "macos")]
    #[test]
    fn multi_block_prefix_restores_after_restart() {
        use crate::metal::MetalDtype;
        use crate::{PagedAttentionConfig, chain_hashes};

        let config = PagedAttentionConfig {
            block_size: 8,
            gpu_memory_mb: 256,
            head_size: 64,
            num_kv_heads: 1,
            num_layers: 1,
            use_fp8_cache: Some(false),
            max_seq_len: Some(64),
            max_batch_size: Some(1),
        };
        // Separate capture and restore pools so a byte match can only come
        // from the cold tier, never from source bytes lingering in a shared
        // physical block (a genuine restart discards the GPU buffers).
        let pool_src = match LayerKVPool::new(config.clone(), 4, MetalDtype::BFloat16) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping multi_block_prefix_restores_after_restart: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        let pool_dst = LayerKVPool::new(config, 4, MetalDtype::BFloat16).unwrap();

        let bytes_per_side = 64 * 8 * 2usize;
        let pattern = |seed: usize| -> (Vec<u8>, Vec<u8>) {
            let keys = (0..bytes_per_side)
                .map(|i| ((i + seed * 7) % 251) as u8)
                .collect();
            let values = (0..bytes_per_side)
                .map(|i| ((i * 3 + seed * 13) % 251) as u8)
                .collect();
            (keys, values)
        };
        let (k0, v0) = pattern(1);
        let (k1, v1) = pattern(2);

        let capture_alloc = Mutex::new(BlockAllocator::new(4, 8));
        let src0 = capture_alloc.lock().unwrap().allocate().unwrap();
        let src1 = capture_alloc.lock().unwrap().allocate().unwrap();
        pool_src
            .write_blocks_from_host(0, &[src0.block_id], &k0, &v0)
            .unwrap();
        pool_src
            .write_blocks_from_host(0, &[src1.block_id], &k1, &v1)
            .unwrap();

        let tokens: Vec<u32> = (1..=16).collect();
        let extra_keys: &[u64] = &[];
        let cache_salt = 0u64;
        let fp = fingerprint();
        let key0 = ColdCacheKey::chain(
            ColdGroup::Kv,
            fp,
            None,
            &tokens[0..8],
            extra_keys,
            cache_salt,
            0,
        );
        let key1 = ColdCacheKey::chain(
            ColdGroup::Kv,
            fp,
            Some(key0),
            &tokens[8..16],
            extra_keys,
            cache_salt,
            1,
        );

        let root = temp_root("multi-restore");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 4).unwrap();
        assert!(
            manager
                .capture_and_enqueue(&pool_src, &src0, key0, fp, &tokens[0..8])
                .unwrap()
        );
        assert!(
            manager
                .capture_and_enqueue(&pool_src, &src1, key1, fp, &tokens[8..16])
                .unwrap()
        );

        let path0 = root.join(format!("{}.safetensors", key0.to_hex()));
        let path1 = root.join(format!("{}.safetensors", key1.to_hex()));
        for _ in 0..200 {
            if path0.exists() && path1.exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(path0.exists() && path1.exists(), "both blocks must persist");

        // Simulate a process restart: release the source handles, tear down
        // the manager, reopen the cache from disk with a fresh allocator.
        capture_alloc.lock().unwrap().free(src0);
        capture_alloc.lock().unwrap().free(src1);
        drop(manager);

        let reopened = ColdCacheManager::open_at(root.clone(), GIB, 0, 4).unwrap();
        let fresh_alloc = Mutex::new(BlockAllocator::new(4, 8));

        let hot = chain_hashes(&tokens, 8, extra_keys, cache_salt);
        assert_eq!(hot.len(), 2);
        let mut parent_key: Option<ColdCacheKey> = None;
        let mut restored = Vec::new();
        for idx in 0..2usize {
            let toks = &tokens[idx * 8..(idx + 1) * 8];
            let key = ColdCacheKey::chain(
                ColdGroup::Kv,
                fp,
                parent_key,
                toks,
                extra_keys,
                cache_salt,
                idx,
            );
            let identity = RestorePrefixIdentity {
                hot_hash: hot[idx],
                tokens: toks.to_vec(),
                parent_hot_hash: if idx == 0 { 0 } else { hot[idx - 1] },
                extra_keys: extra_keys.to_vec(),
                cache_salt,
                block_index: idx,
            };
            let block = reopened
                .restore_block(&pool_dst, &fresh_alloc, key, fp, &identity)
                .expect("cold block restore");
            let (rk, rv) = pool_dst.read_blocks_to_host(0, &[block.block_id]).unwrap();
            let (ek, ev) = if idx == 0 { (&k0, &v0) } else { (&k1, &v1) };
            assert_eq!(&rk, ek, "restored keys must match captured block {idx}");
            assert_eq!(&rv, ev, "restored values must match captured block {idx}");
            parent_key = Some(key);
            restored.push(block);
        }

        // The fresh hot cache now serves the entire two-block prefix.
        let (hits, hit_tokens) = fresh_alloc
            .lock()
            .unwrap()
            .find_longest_cache_hit(&tokens, 8, extra_keys, cache_salt);
        assert_eq!(hit_tokens, 16, "restored prefix must be fully hot-hittable");
        assert_eq!(hits.len(), 2);

        {
            let mut allocator = fresh_alloc.lock().unwrap();
            for block in restored {
                allocator.free(block);
            }
            for hit in hits {
                allocator.free(hit);
            }
        }
        drop(reopened);
        let _ = fs::remove_dir_all(root);
    }

    /// Production Qwen3 KV geometry (block_size=16, num_kv_heads=8,
    /// head_size=128, bf16) at an arbitrary layer count, so the fixture
    /// exercises the O(num_layers) safetensors header the flat `+4096` bound
    /// underestimated.
    fn deep_block(key: ColdCacheKey, num_layers: u32) -> ColdCacheBlock {
        let block_size = 16u32;
        let num_kv_heads = 8u32;
        let head_size = 128u32;
        let dtype_bytes = 2usize; // bf16
        let side_bytes =
            num_kv_heads as usize * head_size as usize * block_size as usize * dtype_bytes;
        let tokens: Vec<u32> = (0..block_size).collect();
        let layers = (0..num_layers as usize)
            .map(|i| ColdLayerBlock {
                keys: (0..side_bytes).map(|b| ((b + i) % 251) as u8).collect(),
                values: (0..side_bytes)
                    .map(|b| ((b * 3 + i * 7) % 251) as u8)
                    .collect(),
            })
            .collect();
        ColdCacheBlock {
            key,
            fingerprint: fingerprint(),
            tokens,
            layout: ColdCacheLayout {
                block_size,
                num_layers,
                num_kv_heads,
                head_size,
                cache_dtype: "BFloat16".to_string(),
                key_bytes_per_layer: side_bytes,
                value_bytes_per_layer: side_bytes,
            },
            layers,
        }
    }

    #[test]
    fn deep_blocks_persist_and_load_within_geometry_bound() {
        // Regression: the O(num_layers) safetensors header overruns a flat
        // +4096 allowance at real Qwen3 depths, so every persisted block was
        // rejected as corruption on restart. Each depth must round-trip within
        // its own encoded bound (numerically the max_encoded_len_for_pool the
        // matching pool would derive — both use header_overhead with equal
        // payload terms).
        for &num_layers in &[28u32, 32, 64] {
            let root = temp_root(&format!("deep-roundtrip-{num_layers}"));
            let manager = ColdCacheManager::open_at(root.clone(), 8 * GIB, 0, 2).unwrap();
            let key = ColdCacheKey::chain(
                ColdGroup::Kv,
                fingerprint(),
                None,
                &[1, 2, 3, 4],
                &[num_layers as u64],
                0,
                0,
            );
            let original = deep_block(key, num_layers);
            let bound = original.encoded_len();
            assert!(
                encode_block(&original).unwrap().len() as u64 <= bound,
                "L={num_layers}: encoded block must fit within its own geometry bound"
            );
            persist_block(&manager.shared, &original).unwrap();
            assert_eq!(
                manager.load_bounded(key, fingerprint(), bound),
                Some(original),
                "L={num_layers}: a legitimate deep block must load within the bound, not miss"
            );
            drop(manager);
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn encoded_len_mirrors_pool_bound_and_upper_bounds_encoder() {
        for &num_layers in &[1u32, 28, 32, 64, 80] {
            let key = ColdCacheKey::chain(
                ColdGroup::Kv,
                fingerprint(),
                None,
                &[1, 2, 3, 4],
                &[num_layers as u64],
                0,
                0,
            );
            let block = deep_block(key, num_layers);
            // encoded_len must equal the geometry-only max_encoded_len_for_pool
            // arithmetic (kv payload + tokens + header_overhead), proving the
            // two bounds stay mirrored without constructing a GPU pool.
            let kv_bytes = crate::profile::bytes_per_block(
                num_layers,
                8,
                128,
                16,
                crate::metal::MetalDtype::BFloat16,
            )
            .unwrap();
            let token_bytes = 16u64 * size_of::<u32>() as u64;
            let pool_bound = kv_bytes + token_bytes + header_overhead(num_layers as u64);
            assert_eq!(
                block.encoded_len(),
                pool_bound,
                "L={num_layers}: encoded_len must mirror max_encoded_len_for_pool arithmetic"
            );
            assert!(
                encode_block(&block).unwrap().len() as u64 <= block.encoded_len(),
                "L={num_layers}: the bound must be a true upper bound on the encoder output"
            );
        }
    }

    #[test]
    fn decode_rejects_forged_huge_num_layers() {
        // A tiny file with correct abi/key/fingerprint but a forged
        // num_layers=u32::MAX and only a `tokens` tensor. Before the guard the
        // decoder ran `Vec::with_capacity(u32::MAX)` (~206 GB) and aborted; the
        // tensor-count check now rejects it. The test returning at all proves
        // no multi-GB allocation happened.
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let fp = fingerprint();
        let token_bytes: Vec<u8> = vec![1, 0, 0, 0]; // one u32
        let view = TensorView::new(Dtype::U8, vec![token_bytes.len()], &token_bytes).unwrap();
        let mut metadata = HashMap::new();
        metadata.insert("abi".to_string(), CACHE_ABI.to_string());
        metadata.insert("key".to_string(), key.to_hex());
        metadata.insert("fingerprint".to_string(), fp.to_hex());
        metadata.insert("checksum".to_string(), "unused".to_string());
        metadata.insert("block_size".to_string(), "1".to_string());
        metadata.insert("num_layers".to_string(), u32::MAX.to_string());
        metadata.insert("num_kv_heads".to_string(), "1".to_string());
        metadata.insert("head_size".to_string(), "1".to_string());
        metadata.insert("cache_dtype".to_string(), "BFloat16".to_string());
        metadata.insert("key_bytes".to_string(), "0".to_string());
        metadata.insert("value_bytes".to_string(), "0".to_string());
        let bytes = serialize(vec![("tokens", view)], Some(metadata)).unwrap();

        assert!(
            decode_block(&bytes, key, fp).is_err(),
            "a forged num_layers must be rejected before allocating layer storage"
        );

        // Delivered through the public load path, the same file must miss and
        // count as a corruption — never abort.
        let root = temp_root("forged-num-layers");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let path = root.join(format!("{}.safetensors", key.to_hex()));
        fs::write(&path, &bytes).unwrap();
        assert!(
            manager.load(key, fp).is_none(),
            "the forged entry must miss, not abort"
        );
        assert_eq!(manager.stats().corruptions, 1);
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }
}
