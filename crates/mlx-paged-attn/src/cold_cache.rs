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
/// How many captured blocks may sit in host memory waiting for the writer.
///
/// This is a MEMORY bound and nothing else: `queue_depth * block_bytes` is the
/// worst-case host footprint of the write-behind queue (8 x 1.84 MB = 15 MB on
/// qwen3-0.6b). It used to double as the per-turn capture rate, because
/// `ColdTierWalk::capture_chain` stopped at the first refusal, so a turn
/// persisted `(Q + 1) / (1 - Tc/Tw)` blocks — a number nobody chose, that
/// moved with the speed of the filesystem, and that went UNBOUNDED on a RAM
/// disk. Capture depth is now the walk's own explicit budget
/// (`MLX_COLD_CAPTURE_BLOCKS_PER_TURN`), which waits for a slot rather than
/// giving up on one, so raising this constant no longer buys reach — it only
/// buys queue slack, at the cost of host memory.
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
///
/// A boundary may ALSO scale exactly one declared `dims` axis
/// ([`Self::new_boundary_scaled`]), for a family whose payload carries
/// `min(boundary, extent)` rows rather than a fixed count — gemma4's rotating
/// sliding window, which holds `min(offset, window)` tokens. A scaled axis is
/// a PAYLOAD-FORMAT property and NEVER a hit rule: it says how many bytes a
/// sidecar at a given boundary carries, and says nothing about which
/// boundaries are eligible. Eligibility stays entirely with the caller's own
/// representability check and with the restore walk's reconcile-down. That is
/// the separation vLLM keeps in `vllm/v1/core/single_type_kv_cache_manager.py`
/// between `SlidingWindowManager::find_longest_cache_hit` (hit discipline,
/// which explicitly accepts a prefix-anchored sub-window run) and
/// `reachable_block_mask` (retention discipline): neither dictates the other,
/// and neither is inferred from the payload's shape.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColdSidecarPolicy {
    layout: ColdSidecarLayout,
    /// Index into `layout.dims` whose extent tracks the candidate boundary,
    /// or `None` when the payload is boundary-invariant (every family but
    /// gemma4 today). `Some(axis)` is only ever produced by
    /// [`Self::new_boundary_scaled`], which pre-validates it so
    /// [`Self::expected_at`] cannot fail.
    boundary_scaled_axis: Option<usize>,
}

impl ColdSidecarPolicy {
    /// Build a policy from a geometry template. Rejects [`ColdGroup::Kv`] and
    /// any geometry a sidecar could never legally be written with, so an
    /// impossible policy cannot be installed and then silently suppress every
    /// restore forever.
    ///
    /// The resulting policy is boundary-INVARIANT: [`Self::expected_at`]
    /// varies `boundary_tokens` and nothing else.
    pub fn new(layout: ColdSidecarLayout) -> Result<Self, String> {
        layout.validate_geometry()?;
        Ok(Self {
            layout: ColdSidecarLayout {
                boundary_tokens: 0,
                ..layout
            },
            boundary_scaled_axis: None,
        })
    }

    /// Build a policy whose `dims[axis]` — and, proportionally,
    /// `bytes_per_tensor` — follow the candidate boundary, clamped at the
    /// template's own extent.
    ///
    /// Everything [`Self::new`] rejects is rejected here too, plus the three
    /// facts that make [`Self::expected_at`] INFALLIBLE: the axis is in range,
    /// its extent is non-zero, and it divides `bytes_per_tensor` evenly (so
    /// the per-row byte cost is exact and the scaled length can never be a
    /// rounded approximation of the payload a capture would write).
    pub fn new_boundary_scaled(layout: ColdSidecarLayout, axis: usize) -> Result<Self, String> {
        layout.validate_geometry()?;
        let Some(&extent) = layout.dims.get(axis) else {
            return Err(
                "cold-cache sidecar boundary-scaled axis is outside the layout dims".to_string(),
            );
        };
        // `validate_geometry` already refuses a zero dim; restated so the
        // division below stays obviously safe if that ever loosens.
        if extent == 0 {
            return Err("cold-cache sidecar boundary-scaled axis must be positive".to_string());
        }
        if !layout.bytes_per_tensor.is_multiple_of(extent as usize) {
            return Err(
                "cold-cache sidecar boundary-scaled axis must divide bytes_per_tensor".to_string(),
            );
        }
        Ok(Self {
            layout: ColdSidecarLayout {
                boundary_tokens: 0,
                ..layout
            },
            boundary_scaled_axis: Some(axis),
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
    ///
    /// With a boundary-scaled axis the shallow end also shrinks the payload:
    /// `dims[axis] = min(boundary_tokens, template_extent)` and
    /// `bytes_per_tensor` scales with it. At and above the template extent the
    /// rule is the IDENTITY, so a policy that gains a scaled axis keeps
    /// describing every sidecar written before it existed — no fingerprint or
    /// on-disk format change. At `boundary_tokens == 0` it yields a zero-length
    /// payload, which [`ColdSidecar::validate`] already refuses to write and no
    /// stored sidecar can match: the fail-closed direction.
    pub fn expected_at(&self, boundary_tokens: u32) -> ColdSidecarLayout {
        let mut layout = ColdSidecarLayout {
            boundary_tokens,
            ..self.layout.clone()
        };
        if let Some(axis) = self.boundary_scaled_axis {
            // `new_boundary_scaled` pinned the axis in range with a non-zero
            // extent that divides `bytes_per_tensor`, so both operations below
            // are total. `scaled <= extent` keeps the product bounded by
            // `bytes_per_tensor`, so it cannot overflow either.
            let extent = self.layout.dims[axis];
            let scaled = boundary_tokens.min(extent);
            layout.dims[axis] = scaled;
            layout.bytes_per_tensor =
                self.layout.bytes_per_tensor / extent as usize * scaled as usize;
        }
        layout
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
    /// Writes ACCEPTED onto the bounded queue — every object that took a slot,
    /// blocks and family state sidecars alike. Object-scoped on purpose: the
    /// queue is shared, so "is the writer keeping up?" can only be answered by
    /// counting everything that entered it. The sidecar telemetry in
    /// `mlx-core` counts a SUB-set of this, not a disjoint peer — never sum
    /// the two.
    pub enqueued: u64,
    /// Writes REFUSED at admission because the bounded queue was full. Same
    /// object scope as [`Self::enqueued`], and a starved sidecar is the failure
    /// this counter most needs to show: a turn that loses one restores nothing
    /// at all. The
    /// counterpart of [`Self::write_errors`], and deliberately disjoint from
    /// it: this one is the producer's problem (offer rate above writer
    /// throughput), that one is the storage's. A single write can only ever be
    /// one or the other — refused before the queue, or accepted and then
    /// failed — so the two must never be summed into one "lost writes" number.
    /// `write_errors` is a subset of `enqueued`; this is not.
    pub queue_drops: u64,
    /// Bytes that LANDED: credited only after `write_all`, the payload sync,
    /// the commit `renameat` and the directory `fsync` have all returned `Ok`
    /// for one object. Not an enqueue-time estimate — a write that fails at
    /// any of those steps adds nothing here and adds one to
    /// [`Self::write_errors`] instead.
    ///
    /// The one thing it is NOT is synchronous with the caller: the writer
    /// thread credits it, so a reader sampling right after `enqueue` sees the
    /// bytes still in flight. `drain` closes that window; `write_errors` is
    /// what distinguishes "still in flight" from "never going to land".
    pub bytes_written: u64,
    pub bytes_restored: u64,
    pub evictions: u64,
    pub corruptions: u64,
    /// Writes the queue ACCEPTED that never reached disk — a read-only,
    /// full, or unmounted cache root, a quota the object cannot fit, a failed
    /// commit rename, a failed fsync.
    ///
    /// The writer is deliberately fail-open: it swallows every persist error
    /// so a broken cache root cannot alter a single emitted token. Before this
    /// counter that fail-open was also fail-SILENT — an operator whose root
    /// was read-only saw `queue_drops 0`, `corruptions 0`, an empty stderr,
    /// and a dashboard reporting a healthy cache that in fact held nothing.
    pub write_errors: u64,
    /// Restores the walk REFUSED to serve, having found candidates on disk.
    ///
    /// Neither a hit nor a miss, and that is exactly why it is here: those two
    /// count per-BLOCK lookups, while a refusal happens before any block is
    /// looked up. A refused restore therefore reports `0/0` — bit-for-bit the
    /// signature of a turn that never consulted the tier at all, which reads
    /// as "nothing ran" rather than "reuse was refused". Recorded by the
    /// caller through [`ColdCacheManager::record_restore_decline`], since the
    /// refusal is a policy decision the manager itself never sees.
    pub restore_declines: u64,
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
    write_errors: AtomicU64,
    restore_declines: AtomicU64,
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
            write_errors: load(&self.write_errors),
            restore_declines: load(&self.restore_declines),
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
                        // that this covered block did not become durable, and
                        // `write_errors` must learn it too. That counter is the
                        // ONLY thing standing between a fail-open writer and a
                        // fail-silent one: the error is returned to nobody, so
                        // without the bump here a root that accepts no writes
                        // at all still reports a spotless cache. Counted once
                        // per accepted job, whatever step of the persist failed,
                        // because this is the one place that sees every
                        // outcome of every job the queue admitted.
                        WriteJob::Block(block) => {
                            if persist_block(&worker_shared, &block).is_err() {
                                worker_shared
                                    .stats
                                    .write_errors
                                    .fetch_add(1, Ordering::Relaxed);
                                failed = true;
                            }
                        }
                        WriteJob::Sidecar(sidecar) => {
                            if persist_sidecar(&worker_shared, &sidecar).is_err() {
                                worker_shared
                                    .stats
                                    .write_errors
                                    .fetch_add(1, Ordering::Relaxed);
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

    /// Record one restore this tier held candidates for and the caller refused
    /// to serve — see [`ColdCacheStats::restore_declines`].
    ///
    /// Recorded by the caller because the refusal is decided outside this
    /// crate: the restore walk lives in `mlx-core`, reaches its verdict from
    /// side-effect-free `contains` / `contains_in` probes, and returns without
    /// ever asking the manager to load anything. So the manager sees no
    /// lookup, counts no hit and no miss, and the refusal is invisible unless
    /// the walk says so here.
    ///
    /// Deliberately a counter and not a log line: a decline is a steady-state
    /// event on the first turn of any prompt, so it must be cheap and
    /// aggregatable, and the diagnosis (which boundary, which ceiling) belongs
    /// in the walk's `MLX_TRACE` line where the numbers still have context.
    pub fn record_restore_decline(&self) {
        self.shared
            .stats
            .restore_declines
            .fetch_add(1, Ordering::Relaxed);
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
    ///
    /// Non-blocking admission: a full queue drops the write. Callers that
    /// would rather wait a bounded time for a slot than lose the block use
    /// [`Self::capture_and_enqueue_before`].
    pub fn capture_and_enqueue(
        &self,
        pool: &LayerKVPool,
        block: &Arc<PhysicalBlock>,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        tokens: &[u32],
    ) -> Result<bool, String> {
        self.capture_and_enqueue_before(pool, block, key, fingerprint, tokens, Instant::now())
    }

    /// [`Self::capture_and_enqueue`] with a bounded wait for a queue slot.
    ///
    /// The Metal blit happens first and unconditionally — it is the expensive
    /// half and it must run while the block is pinned — then the owned host
    /// bytes are offered to the writer until `deadline`. A caller walking a
    /// chain of blocks passes the SAME deadline for every block, so the wait
    /// is a budget over the whole walk, not per block.
    pub fn capture_and_enqueue_before(
        &self,
        pool: &LayerKVPool,
        block: &Arc<PhysicalBlock>,
        key: ColdCacheKey,
        fingerprint: ColdCacheFingerprint,
        tokens: &[u32],
        deadline: Instant,
    ) -> Result<bool, String> {
        if tokens.len() != pool.block_size() as usize {
            return Err("cold cache captures full blocks only".to_string());
        }

        // Logical pin prevents allocator eviction/reuse while Metal blits run.
        block.incref();
        let captured: Result<ColdCacheBlock, String> = (|| {
            // One command buffer for the whole block. Reading layer by layer
            // cost one blocking GPU round-trip per layer, all of them on the
            // inference thread and all of them inside the pin above. A failure
            // still returns here with `layers` dropped, so the `decref` below
            // runs and no half-populated block can reach `enqueue`.
            let layers: Vec<ColdLayerBlock> = pool
                .read_block_all_layers(block.block_id)?
                .into_iter()
                .map(|(keys, values)| ColdLayerBlock { keys, values })
                .collect();
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
        let captured = captured?;
        captured.validate()?;
        self.send_before(WriteJob::Block(captured), deadline)
    }

    /// Offer `job` to the bounded writer queue, retrying until a slot frees or
    /// `deadline` passes.
    ///
    /// `deadline <= now` degenerates to exactly one `try_send`, which is the
    /// historical non-blocking contract of [`Self::enqueue`] /
    /// [`Self::enqueue_sidecar`] — the retry loop below cannot sleep, because
    /// the deadline check runs before the sleep. That degeneracy is what lets
    /// this be the single admission implementation for every caller.
    ///
    /// `Ok(true)` accepted, `Ok(false)` refused within the deadline (the
    /// caller decides whether that is a `queue_drops`), `Err` writer gone.
    fn send_before(&self, job: WriteJob, deadline: Instant) -> Result<bool, String> {
        let mut job = job;
        loop {
            match self.sender.try_send(job) {
                Ok(()) => {
                    self.shared.stats.enqueued.fetch_add(1, Ordering::Relaxed);
                    return Ok(true);
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err("cold-cache writer stopped".to_string());
                }
                Err(TrySendError::Full(returned)) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        self.shared
                            .stats
                            .queue_drops
                            .fetch_add(1, Ordering::Relaxed);
                        return Ok(false);
                    }
                    job = returned;
                    // A commit is ~0.4-1.3 ms, so 1 ms is roughly one slot's
                    // worth of wait — short enough that the walk's block
                    // budget, not the poll granularity, is what bounds it.
                    std::thread::sleep(Duration::from_millis(1).min(remaining));
                }
            }
        }
    }

    /// Non-blocking enqueue. A saturated queue deliberately drops the cold
    /// write so host buffers cannot grow without bound.
    pub fn enqueue(&self, block: ColdCacheBlock) -> Result<bool, String> {
        block.validate()?;
        self.send_before(WriteJob::Block(block), Instant::now())
    }

    /// Non-blocking enqueue of a state sidecar, with the same bounded-queue
    /// drop policy as [`Self::enqueue`]. A dropped sidecar is not a
    /// correctness problem: without it the next restore simply reconciles the
    /// candidate prefix down past that boundary and recomputes.
    pub fn enqueue_sidecar(&self, sidecar: ColdSidecar) -> Result<bool, String> {
        self.enqueue_sidecar_before(sidecar, Instant::now())
    }

    /// [`Self::enqueue_sidecar`] with a bounded wait for a queue slot.
    ///
    /// Every hybrid family offers its sidecar MICROSECONDS after its K/V
    /// capture walk returns, and that walk stops precisely when the queue is
    /// saturated — so a non-blocking sidecar enqueue is offered to a queue the
    /// walk just filled, and loses. A dropped sidecar is worse than a dropped
    /// block: without it the whole restore reconciles down to nothing
    /// (`ColdRestore::miss()`), so the turn's entire persisted chain is
    /// unusable. Families therefore wait out the same walk budget here.
    pub fn enqueue_sidecar_before(
        &self,
        sidecar: ColdSidecar,
        deadline: Instant,
    ) -> Result<bool, String> {
        sidecar.validate()?;
        self.send_before(WriteJob::Sidecar(Box::new(sidecar)), deadline)
    }

    /// Block until every write accepted before this call has been committed
    /// (payload [`sync_payload`] + rename + directory fsync), or `timeout`
    /// elapses.
    ///
    /// "Committed" is exactly what [`sync_payload`] provides: the object
    /// survives process death and kernel panic, but NOT a sudden power loss,
    /// which can leave a renamed object whose payload extents never reached
    /// the drive. Such an object fails its payload checksum on the next read
    /// and is pruned as a miss.
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
        //
        // Deliberately NOT [`Self::send_before`], despite the identical shape.
        // A barrier is not a write: admitting one must not bump `enqueued`,
        // failing to admit one must not bump `queue_drops`, and a disconnected
        // writer means the drain is already satisfied (`true`) rather than an
        // error. Sharing the loop would have to thread all three differences
        // through as flags, which is more code than the loop.
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

        // One command buffer for the whole block instead of one per layer.
        // Two different failures land here. Validation (layout, block id, per
        // layer byte lengths) rejects before the first blit is encoded, so
        // nothing was written. A command-buffer abort is reported after the
        // blits were submitted, so some layers of this block may have been
        // applied and others not.
        //
        // Both are safe for the same reason, and it is not "nothing was
        // written": `BlockAllocator::allocate` never zeroes, so every freshly
        // handed-out block already holds a previous owner's bytes. The
        // invariant restore depends on is that the block is never PUBLISHED —
        // freeing it here without reaching `publish_restored_prefix` leaves it
        // unreachable through the prefix cache, and its slots are only ever
        // read again after `reshape_and_cache` rewrites them.
        let layer_bytes: Vec<(&[u8], &[u8])> = cold
            .layers
            .iter()
            .map(|layer| (layer.keys.as_slice(), layer.values.as_slice()))
            .collect();
        if pool
            .write_block_all_layers(block.block_id, &layer_bytes)
            .is_err()
        {
            allocator
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .free(Arc::clone(&block));
            self.shared.stats.misses.fetch_add(1, Ordering::Relaxed);
            return None;
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
        sync_payload(&file)?;
        // The index lock spans [rename + index publish] so a concurrent
        // failed-load cleanup can never observe the renamed file without
        // its index entry (or delete it in between).
        let mut index = shared
            .index
            .lock()
            .map_err(|_| "cold-cache index mutex poisoned".to_string())?;
        // A failed commit rename is a write the queue ACCEPTED and could not
        // land, so it is a `write_errors` — counted by the worker loop, which
        // sees this `Err` along with every other way a persist can fail. It
        // used to be counted here as a `queue_drops` because no write-error
        // counter existed; that made one event move a counter named after a
        // completely different cause (a full queue at admission) and left the
        // other failure modes — a read-only root, a full disk, a failed fsync
        // — with no counter at all.
        shared.root.rename(&temp, &destination)?;
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
        // The rename is the true commit point (the payload already went
        // through `sync_payload`), so the index publish above is bound to rename
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
    // Refuse an impossible write BEFORE evicting anything. Both loop
    // conditions below are insensitive to how much has already been reclaimed
    // once `incoming` alone cannot fit: `total_bytes + incoming > quota` stays
    // true at `total_bytes == 0`, and the free-space branch stays true even
    // after every entry is gone. The loop would then evict the ENTIRE cache
    // one LRU entry at a time and still return the same error, so a single
    // oversized object (a gemma4 sliding sidecar runs to hundreds of MiB)
    // against a small quota — or any write on a nearly-full disk — silently
    // wipes a cache it was never able to join.
    if incoming > shared.quota_bytes {
        return Err(format!(
            "cold-cache write of {incoming} bytes exceeds the {} byte quota",
            shared.quota_bytes
        ));
    }
    // Everything still indexed and not yet ruled out is the most this eviction
    // can reclaim — an UPPER bound, not a promise. An entry whose file another
    // process already deleted counts its size here and frees nothing when
    // cleared, so a single estimate taken up front can clear the bar while the
    // real reclaim falls short. Re-tested at the top of every iteration
    // instead, against what actually remains: the bound drops by the entry's
    // own size whether that entry was cleared or found unclearable, so it
    // converges on the truth as stale entries are discovered.
    let mut reclaimable = index.total_bytes;
    let mut unclearable: Vec<ColdCacheKey> = Vec::new();
    while index.total_bytes.saturating_add(incoming) > shared.quota_bytes
        || available < shared.reserve_bytes.saturating_add(incoming)
    {
        // Refuse before destroying one more live entry for a write the
        // survivors can no longer make room for. Past this point every
        // eviction is pure loss: the write fails either way, and what it
        // takes with it is warm cache someone was about to reuse. Guards the
        // free-space axis only — under quota pressure alone `available`
        // already clears `reserve + incoming`, so this cannot fire and
        // mislabel a quota eviction as a disk-space failure.
        if available.saturating_add(reclaimable) < shared.reserve_bytes.saturating_add(incoming) {
            return Err("insufficient disk space for cold-cache write".to_string());
        }
        let Some((&key, entry)) = index
            .entries
            .iter()
            .filter(|&(key, _)| !unclearable.contains(key))
            .min_by_key(|(_, entry)| entry.last_access)
        else {
            return Err("insufficient disk space for cold-cache write".to_string());
        };
        let name = entry.file_name.clone();
        let candidate_size = entry.size;
        let cleared = match shared.root.stat_identity(&name) {
            Some((_, kind)) => clear_entry(&shared.root, &name, kind),
            // Either the entry already vanished (the unlink then observes
            // NotFound) or this platform reports no identities and the
            // plain unlink decides.
            None => entry_gone(shared.root.unlink(&name)),
        };
        if !cleared {
            unclearable.push(key);
            reclaimable = reclaimable.saturating_sub(candidate_size);
            continue;
        }
        if let Some(entry) = index.entries.remove(&key) {
            index.total_bytes = index.total_bytes.saturating_sub(entry.size);
            reclaimable = reclaimable.saturating_sub(entry.size);
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

/// Push one cold object's bytes toward the storage device before the rename
/// that commits it.
///
/// Deliberately NOT [`File::sync_all`]: on Apple targets that is
/// `fcntl(F_FULLFSYNC)`, which flushes the drive's own volatile write cache
/// and is device-wide. Measured by `bench_write_path_phase_decomposition` on
/// a qwen3_5 dense block, the flush was 3.876 ms of a 4.267 ms per-object
/// service time; `fsync(2)` costs 0.089 ms of 0.432 ms in the same harness.
/// Because the writer is a single thread, every queued block waits behind that
/// drive round trip, so `Tw` is what a capture walk's per-block wait converges
/// to once its budget outruns the queue. Under `F_FULLFSYNC` that wait pinned
/// the persisted chain at exactly 9 blocks per turn no matter how fast the
/// producer ran (`bench_chain_advance_per_turn`); the walk no longer stops at a
/// refusal, so `Tw` now sets how long a budgeted walk TAKES rather than how far
/// it REACHES.
///
/// The guarantee that is dropped is power-loss durability, and only that:
/// `fsync(2)` still hands the bytes to the kernel's device queue, so process
/// death and kernel panic are unaffected. It also drops the implicit device
/// ordering barrier, so after a hard power cut the (journalled) rename may be
/// present while the payload extents are not.
///
/// That is affordable here because this is a cache whose every read
/// re-derives [`payload_checksum`] over the tensor payload and compares it
/// against the value recorded at write time ([`decode_block`],
/// [`decode_sidecar`]). A mismatch is an `Err`, and `load_object_bounded`
/// turns `Err` into a miss plus a prune, so a torn object costs one
/// recomputed prefix and can never be handed to inference as data.
#[cfg(unix)]
fn sync_payload(file: &File) -> Result<(), String> {
    rustix::fs::fsync(file).map_err(|e| format!("sync cold-cache file: {e}"))
}

/// Non-unix keeps `sync_all`. The measurement above is a macOS/APFS result,
/// and this platform's [`RootDir::sync`] is already a no-op, so it is not the
/// platform this trade-off was made for.
#[cfg(not(unix))]
fn sync_payload(file: &File) -> Result<(), String> {
    file.sync_all()
        .map_err(|e| format!("sync cold-cache file: {e}"))
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

#[cfg(all(test, target_os = "macos"))]
mod restore_decomposition_bench {
    //! Measurement-only harness (temporary, `#[ignore]`d, not part of any gate).
    //! Decomposes the per-block cold-tier restore cost into named phases using
    //! the real `qwen3-0.6b-mlx-bf16` pool geometry.
    use super::tests::{fingerprint, temp_root};
    use super::*;
    use crate::metal::MetalDtype;
    use crate::{PagedAttentionConfig, hash_tokens};

    const BENCH_LAYERS: u32 = 28;
    const BENCH_KV_HEADS: u32 = 8;
    const BENCH_HEAD_SIZE: u32 = 128;
    const BENCH_BLOCK_SIZE: u32 = 16;
    const BENCH_BLOCKS: usize = 64;

    fn ms(d: Duration) -> f64 {
        d.as_secs_f64() * 1e3
    }

    fn pct(part: f64, whole: f64) -> f64 {
        if whole <= 0.0 {
            0.0
        } else {
            part / whole * 100.0
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "measurement harness; run explicitly with --ignored"]
    fn bench_restore_block_phase_decomposition() {
        let config = PagedAttentionConfig {
            block_size: BENCH_BLOCK_SIZE,
            gpu_memory_mb: 4096,
            head_size: BENCH_HEAD_SIZE,
            num_kv_heads: BENCH_KV_HEADS,
            num_layers: BENCH_LAYERS,
            use_fp8_cache: Some(false),
            max_seq_len: Some(4096),
            max_batch_size: Some(1),
        };
        let total_blocks = (BENCH_BLOCKS * 2 + 4) as u32;
        let pool = match LayerKVPool::new(config, total_blocks, MetalDtype::BFloat16) {
            Ok(pool) => pool,
            Err(e) => {
                eprintln!("skipping bench: {e}");
                return;
            }
        };
        let per_side = (BENCH_KV_HEADS * BENCH_HEAD_SIZE * BENCH_BLOCK_SIZE * 2) as usize;
        let payload_bytes = per_side * 2 * BENCH_LAYERS as usize;
        eprintln!(
            "geometry: {BENCH_LAYERS} layers x {BENCH_KV_HEADS} kv-heads x {BENCH_HEAD_SIZE} dim \
             x {BENCH_BLOCK_SIZE} tok bf16 = {payload_bytes} B/block ({:.2} MB)",
            payload_bytes as f64 / 1e6
        );

        let allocator = Mutex::new(BlockAllocator::new(total_blocks, BENCH_BLOCK_SIZE));
        let root = temp_root("bench-restore");
        let manager = ColdCacheManager::open_at(root.clone(), 8 * GIB, 0, 32).unwrap();

        // ---- populate: capture BENCH_BLOCKS real objects through the real writer.
        let source = allocator.lock().unwrap().allocate().unwrap();
        let keys: Vec<u8> = (0..per_side).map(|i| (i % 251) as u8).collect();
        let values: Vec<u8> = (0..per_side).map(|i| (250 - (i % 251)) as u8).collect();
        let mut capture_perlayer = Duration::ZERO;
        let t = Instant::now();
        for layer in 0..BENCH_LAYERS {
            pool.write_blocks_from_host(layer, &[source.block_id], &keys, &values)
                .unwrap();
        }
        capture_perlayer += t.elapsed();
        eprintln!(
            "seed upload of 1 block ({BENCH_LAYERS} write_blocks_from_host calls): {:.3} ms",
            ms(capture_perlayer)
        );

        let mut all_keys = Vec::new();
        let mut parent = None;
        // Capture is ~20x faster now that a block is one command buffer, so the
        // producer readily outruns the writer and `capture_and_enqueue` starts
        // returning `Ok(false)` (bounded-queue drop). The measurement needs all
        // BENCH_BLOCKS objects on disk, so back off and retry; `queue_full`
        // counts how often the frontier was hit.
        let mut queue_full = 0usize;
        for i in 0..BENCH_BLOCKS {
            let toks: Vec<u32> = (0..BENCH_BLOCK_SIZE)
                .map(|t| (i as u32) * 1000 + t)
                .collect();
            let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), parent, &toks, &[], 0, i);
            while !manager
                .capture_and_enqueue(&pool, &source, key, fingerprint(), &toks)
                .unwrap()
            {
                queue_full += 1;
                assert!(
                    manager.drain(Duration::from_secs(120)),
                    "writer must drain when the queue is full"
                );
            }
            parent = Some(key);
            all_keys.push((key, toks));
        }
        eprintln!(
            "populate: {BENCH_BLOCKS} blocks captured, write queue hit its bound {queue_full} \
             time(s)"
        );
        assert!(
            manager.drain(Duration::from_secs(120)),
            "writer must drain before measuring reads"
        );
        allocator.lock().unwrap().free(source);

        // ---- phase A: raw file read only (open + read syscalls, no decode).
        let mut raw_read = Duration::ZERO;
        let mut raw_bytes = 0usize;
        for (key, _) in &all_keys {
            let path = root.join(object_file_name(key, ColdGroup::Kv));
            let t = Instant::now();
            let bytes = fs::read(&path).unwrap();
            raw_read += t.elapsed();
            raw_bytes += bytes.len();
        }
        eprintln!(
            "[A] raw fs::read x{BENCH_BLOCKS} ({} B total): {:.3} ms total, {:.4} ms/block",
            raw_bytes,
            ms(raw_read),
            ms(raw_read) / BENCH_BLOCKS as f64
        );

        // ---- phase A2: same read again (page cache now warm) for cold-vs-warm split.
        let mut raw_read_warm = Duration::ZERO;
        for (key, _) in &all_keys {
            let path = root.join(object_file_name(key, ColdGroup::Kv));
            let t = Instant::now();
            let _ = fs::read(&path).unwrap();
            raw_read_warm += t.elapsed();
        }
        eprintln!(
            "[A2] raw fs::read again (warm page cache): {:.3} ms total, {:.4} ms/block",
            ms(raw_read_warm),
            ms(raw_read_warm) / BENCH_BLOCKS as f64
        );

        // ---- phase A3: TRUE cold read. `sudo purge` needs a password, so instead
        //      bypass the unified buffer cache per-descriptor with F_NOCACHE, which
        //      forces every read to reach the device.
        let mut nocache_read = Duration::ZERO;
        for (key, _) in &all_keys {
            let path = root.join(object_file_name(key, ColdGroup::Kv));
            let t = Instant::now();
            let f = File::open(&path).unwrap();
            unsafe {
                use std::os::fd::AsRawFd;
                assert_eq!(libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1), 0);
            }
            let mut buf = Vec::new();
            let mut f = f;
            f.read_to_end(&mut buf).unwrap();
            nocache_read += t.elapsed();
            assert_eq!(buf.len(), raw_bytes / BENCH_BLOCKS);
        }
        eprintln!(
            "[A3] fs read with F_NOCACHE (page cache bypassed, true device read): {:.3} ms total, \
             {:.4} ms/block",
            ms(nocache_read),
            ms(nocache_read) / BENCH_BLOCKS as f64
        );

        // ---- phase B: decode + checksum (load_bounded minus the raw read).
        let max_encoded = max_encoded_len_for_pool(&pool);
        let mut load_total = Duration::ZERO;
        for (key, _) in &all_keys {
            let t = Instant::now();
            let got = manager.load_bounded(*key, fingerprint(), max_encoded);
            load_total += t.elapsed();
            assert!(got.is_some(), "bench object must decode");
        }
        eprintln!(
            "[B] load_bounded (open+read+decode+sha256+touch) x{BENCH_BLOCKS}: {:.3} ms total, \
             {:.4} ms/block",
            ms(load_total),
            ms(load_total) / BENCH_BLOCKS as f64
        );

        // ---- phase C: sha256 alone over one payload, to size the checksum term.
        let sample = manager
            .load_bounded(all_keys[0].0, fingerprint(), max_encoded)
            .unwrap();
        let mut owned = Vec::with_capacity(1 + sample.layers.len() * 2);
        owned.push((
            "tokens".to_string(),
            sample.tokens.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
        for (i, layer) in sample.layers.iter().enumerate() {
            owned.push((format!("layer.{i}.key"), layer.keys.clone()));
            owned.push((format!("layer.{i}.value"), layer.values.clone()));
        }
        let t = Instant::now();
        for _ in 0..BENCH_BLOCKS {
            let _ = payload_checksum(&owned);
        }
        let sha_total = t.elapsed();
        eprintln!(
            "[C] payload_checksum (sha256 over {} B) x{BENCH_BLOCKS}: {:.4} ms/block",
            payload_bytes,
            ms(sha_total) / BENCH_BLOCKS as f64
        );

        // ---- phase C2: the extra clone decode_block makes just to feed the checksum.
        let t = Instant::now();
        for _ in 0..BENCH_BLOCKS {
            let mut o = Vec::with_capacity(1 + sample.layers.len() * 2);
            o.push((
                "tokens".to_string(),
                sample.tokens.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ));
            for (i, layer) in sample.layers.iter().enumerate() {
                o.push((format!("layer.{i}.key"), layer.keys.clone()));
                o.push((format!("layer.{i}.value"), layer.values.clone()));
            }
            std::hint::black_box(&o);
        }
        let clone_total = t.elapsed();
        eprintln!(
            "[C2] decode_block's checksum-only payload clone x{BENCH_BLOCKS}: {:.4} ms/block",
            ms(clone_total) / BENCH_BLOCKS as f64
        );

        // ---- phase D: device upload, exactly as restore_block issues it
        //      (one write_blocks_from_host per layer => per-layer commit+wait).
        let scratch = allocator.lock().unwrap().allocate().unwrap();
        let mut upload_total = Duration::ZERO;
        for _ in 0..BENCH_BLOCKS {
            let t = Instant::now();
            for (layer_idx, layer) in sample.layers.iter().enumerate() {
                pool.write_blocks_from_host(
                    layer_idx as u32,
                    &[scratch.block_id],
                    &layer.keys,
                    &layer.values,
                )
                .unwrap();
            }
            upload_total += t.elapsed();
        }
        eprintln!(
            "[D] device upload, {BENCH_LAYERS} x write_blocks_from_host (per-layer commit+wait) \
             x{BENCH_BLOCKS}: {:.3} ms total, {:.4} ms/block",
            ms(upload_total),
            ms(upload_total) / BENCH_BLOCKS as f64
        );

        // ---- phase D2: the same upload as restore_block issues it TODAY —
        //      one write_block_all_layers => a single commit+wait per block.
        let borrowed: Vec<(&[u8], &[u8])> = sample
            .layers
            .iter()
            .map(|layer| (layer.keys.as_slice(), layer.values.as_slice()))
            .collect();
        let mut upload_batched = Duration::ZERO;
        for _ in 0..BENCH_BLOCKS {
            let t = Instant::now();
            pool.write_block_all_layers(scratch.block_id, &borrowed)
                .unwrap();
            upload_batched += t.elapsed();
        }
        eprintln!(
            "[D2] device upload, 1 x write_block_all_layers (single commit+wait) x{BENCH_BLOCKS}: \
             {:.3} ms total, {:.4} ms/block  ({:.2}x faster than [D])",
            ms(upload_batched),
            ms(upload_batched) / BENCH_BLOCKS as f64,
            ms(upload_total) / ms(upload_batched)
        );

        // ---- phase G: the CAPTURE direction, which runs on the inference
        //      thread on every turn that captures — not just on restore turns.
        //      G is the old per-layer readback shape, G2 the batched one that
        //      capture_and_enqueue uses now.
        let mut capture_per_layer = Duration::ZERO;
        for _ in 0..BENCH_BLOCKS {
            let t = Instant::now();
            let mut layers = Vec::with_capacity(BENCH_LAYERS as usize);
            for layer in 0..BENCH_LAYERS {
                layers.push(
                    pool.read_blocks_to_host(layer, &[scratch.block_id])
                        .unwrap(),
                );
            }
            capture_per_layer += t.elapsed();
            std::hint::black_box(&layers);
        }
        let mut capture_batched = Duration::ZERO;
        for _ in 0..BENCH_BLOCKS {
            let t = Instant::now();
            let layers = pool.read_block_all_layers(scratch.block_id).unwrap();
            capture_batched += t.elapsed();
            std::hint::black_box(&layers);
        }
        eprintln!(
            "[G] capture readback, {BENCH_LAYERS} x read_blocks_to_host (per-layer commit+wait) \
             x{BENCH_BLOCKS}: {:.3} ms total, {:.4} ms/block",
            ms(capture_per_layer),
            ms(capture_per_layer) / BENCH_BLOCKS as f64
        );
        eprintln!(
            "[G2] capture readback, 1 x read_block_all_layers (single commit+wait) \
             x{BENCH_BLOCKS}: {:.3} ms total, {:.4} ms/block  ({:.2}x faster than [G])",
            ms(capture_batched),
            ms(capture_batched) / BENCH_BLOCKS as f64,
            ms(capture_per_layer) / ms(capture_batched)
        );
        allocator.lock().unwrap().free(scratch);

        // ---- phase E: full restore_block, production path, as ground truth.
        let mut restore_total = Duration::ZERO;
        let mut restored_blocks = Vec::new();
        let mut parent_ok = 0usize;
        for (i, (key, toks)) in all_keys.iter().enumerate() {
            let identity = RestorePrefixIdentity {
                hot_hash: hash_tokens(toks, if i == 0 { 0 } else { parent_ok as u64 }, &[]),
                tokens: toks.clone(),
                parent_hot_hash: 0,
                extra_keys: vec![],
                cache_salt: 0,
                block_index: i,
            };
            let t = Instant::now();
            let got = manager.restore_block(&pool, &allocator, *key, fingerprint(), &identity);
            restore_total += t.elapsed();
            if let Some(b) = got {
                restored_blocks.push(b);
                parent_ok += 1;
            }
        }
        eprintln!(
            "[E] restore_block end-to-end x{BENCH_BLOCKS} ({} succeeded): {:.3} ms total, {:.4} \
             ms/block",
            restored_blocks.len(),
            ms(restore_total),
            ms(restore_total) / BENCH_BLOCKS as f64
        );

        let n = BENCH_BLOCKS as f64;
        let e2e = ms(restore_total) / n;
        let a = ms(raw_read_warm) / n;
        let b = ms(load_total) / n;
        // `restore_block` uploads through the batched path now, so the
        // decomposition of the measured [E] must use [D2], not [D].
        let d = ms(upload_batched) / n;
        eprintln!("\n=== per-block decomposition (ms, % of restore_block e2e) ===");
        eprintln!("  open+read (warm)        {:8.3}  {:5.1}%", a, pct(a, e2e));
        eprintln!(
            "  decode+sha256 (B - A)   {:8.3}  {:5.1}%",
            b - a,
            pct(b - a, e2e)
        );
        eprintln!(
            "    of which sha256       {:8.3}  {:5.1}%",
            ms(sha_total) / n,
            pct(ms(sha_total) / n, e2e)
        );
        eprintln!(
            "    of which extra clone  {:8.3}  {:5.1}%",
            ms(clone_total) / n,
            pct(ms(clone_total) / n, e2e)
        );
        eprintln!("  device upload (batched) {:8.3}  {:5.1}%", d, pct(d, e2e));
        eprintln!(
            "  bookkeeping (E-B-D2)    {:8.3}  {:5.1}%",
            e2e - b - d,
            pct(e2e - b - d, e2e)
        );
        eprintln!("  ------------------------------------------");
        eprintln!("  restore_block e2e       {:8.3}  100.0%", e2e);
        eprintln!(
            "  (upload before batching was {:.3} ms/block; the old e2e would be {:.3} ms)",
            ms(upload_total) / n,
            e2e - d + ms(upload_total) / n
        );

        {
            let mut g = allocator.lock().unwrap();
            for blk in restored_blocks {
                g.free(blk);
            }
        }

        // ---- phase F: CONTENDED load. The reader's `openat` sits inside the same
        //      index mutex the writer holds across `renameat` + directory fsync
        //      (cold_cache.rs:1727-1758), and nothing drains the write queue before
        //      a restore. Reproduce that overlap: keep the writer committing while
        //      the same objects are loaded.
        // A CPU-only clone of one captured block; `enqueue` needs no GPU, so the
        // writer thread can drive real commits (encode + fsync + rename + dir
        // fsync) without touching the pool.
        let template = sample.clone();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_w = Arc::clone(&stop);
        let mgr2 = Arc::new(ColdCacheManager::open_at(root.clone(), 8 * GIB, 0, 8).unwrap());
        let mgr2_w = Arc::clone(&mgr2);
        let writer = std::thread::spawn(move || {
            let mut accepted = 0usize;
            let mut i = 1_000_000usize;
            while !stop_w.load(Ordering::Relaxed) {
                let toks: Vec<u32> = (0..BENCH_BLOCK_SIZE).map(|t| i as u32 * 31 + t).collect();
                let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &toks, &[], 0, 0);
                let mut b = template.clone();
                b.key = key;
                b.tokens = toks;
                if matches!(mgr2_w.enqueue(b), Ok(true)) {
                    accepted += 1;
                }
                i += 1;
            }
            accepted
        });
        std::thread::sleep(Duration::from_millis(500));
        let mut load_contended = Duration::ZERO;
        for (key, _) in &all_keys {
            let t = Instant::now();
            let _ = manager.load_bounded(*key, fingerprint(), max_encoded);
            load_contended += t.elapsed();
        }
        stop.store(true, Ordering::Relaxed);
        let writes = writer.join().unwrap();
        eprintln!(
            "\n[F] load_bounded while a writer commits concurrently ({writes} writes accepted): \
             {:.3} ms total, {:.4} ms/block  ({:.2}x the quiet {:.4} ms/block)",
            ms(load_contended),
            ms(load_contended) / n,
            (ms(load_contended) / n) / (ms(load_total) / n),
            ms(load_total) / n
        );

        drop(mgr2);
        drop(manager);
        let _ = fs::remove_dir_all(&root);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod write_decomposition_bench {
    //! Measurement-only harness (temporary, `#[ignore]`d, not part of any
    //! gate). Decomposes the writer thread's per-object service time `Tw` —
    //! which is exactly [`persist_block`]: `validate` + [`encode_block`] +
    //! [`persist_encoded`] — into named phases, then re-measures the same
    //! phases with the payload `fsync` swapped for cheaper barriers.
    //!
    //! Nothing here touches Metal. The writer thread only ever sees owned host
    //! bytes, so the payload is synthesised on the host at the real
    //! `qwen3-0.6b-mlx-bf16` geometry (28 layers x 8 KV heads x 128 head dim x
    //! 16 tokens, bf16 = 1,835,008 payload bytes per block).
    //!
    //! Sampling is ROUND-ROBIN across the sync variants, rotated one position
    //! per round, because this machine is not guaranteed quiet. Block
    //! structured sampling ("run 128 of variant A, then 128 of variant B")
    //! attributes any drift in background load to whichever variant happened
    //! to be running during it; round-robin spreads that drift across all
    //! variants, and the rotation stops any variant from permanently owning
    //! the first (cache-cold) or last position within a round. Each round is
    //! labelled with whether a `rustc`/`cargo` was running immediately before
    //! AND immediately after it, so a quiet-only subset can be reported
    //! alongside the full set.
    //!
    //! The variants exist to separate three different costs that
    //! `File::sync_all` conflates on macOS:
    //! - `sync_all` — what production did before [`sync_payload`], kept as the
    //!   speedup baseline.
    //! - plain `fsync(2)` — the POSIX flush, which on APFS does NOT push the
    //!   drive's own write cache. Comparing it against `sync_all` is the
    //!   direct in-process test of whether `sync_all` really issues
    //!   `F_FULLFSYNC` on this platform. This is what production calls now.
    //! - `F_BARRIERFSYNC` — an I/O ordering barrier with no device flush.
    //! - nothing at all — the floor.
    use super::tests::{fingerprint, temp_root};
    use super::*;
    use crate::PagedAttentionConfig;
    use crate::metal::MetalDtype;
    use std::os::fd::AsRawFd;

    const BENCH_BLOCK_SIZE: usize = 16;
    /// bf16.
    const BENCH_ELEMENT: usize = 2;

    /// One family's paged-KV block shape. Only full-attention layers hold
    /// paged KV, so `layers` is the full-attention layer count, not the
    /// model's depth.
    #[derive(Clone, Copy)]
    struct Geometry {
        label: &'static str,
        layers: usize,
        kv_heads: usize,
        head_size: usize,
        /// Object size this shape is chosen to stand in for. Checked (to 1%)
        /// against what the production encoder actually emits before any
        /// timing runs, so a later edit to the numbers above cannot silently
        /// move the bench onto a payload no family writes.
        object_bytes: usize,
        /// Producer cost `Tc` for this shape, from
        /// [`bench_capture_cost_per_geometry`] on an M5 Max. The frontier `N`
        /// is steep in it, so it is per-geometry rather than one constant:
        /// a smaller block moves less data over the same GPU round trip.
        capture_ms: f64,
    }

    impl Geometry {
        /// K (and, identically, V) bytes for one layer of one block.
        const fn per_side(&self) -> usize {
            self.kv_heads * self.head_size * BENCH_BLOCK_SIZE * BENCH_ELEMENT
        }

        const fn payload_bytes(&self) -> usize {
            self.per_side() * 2 * self.layers
        }
    }

    /// The bench originally measured only the first of these. It is ~9x
    /// larger than the block the hybrid families actually persist, which
    /// inflates every payload-proportional phase (host clone, checksum,
    /// `write_all`) against the fixed per-object syscall costs — so `Tw`, and
    /// with it the frontier `N`, does not transfer between them.
    ///
    /// The two hybrid entries are shapes CHOSEN to land on the measured
    /// per-block object sizes (198,072 B dense, 329,760 B MoE); only
    /// full-attention layers hold paged KV, which is why their layer counts
    /// are far below model depth. They are not claimed to be those models'
    /// exact `(layers, kv_heads, head_size)` — what the writer's cost depends
    /// on is bytes per object and descriptor count, and both are matched to
    /// well under 1%.
    const GEOMETRIES: [Geometry; 3] = [
        Geometry {
            label: "qwen3-0.6b block  28 attn layers x 8 kv x 128",
            layers: 28,
            kv_heads: 8,
            head_size: 128,
            object_bytes: 1_839_952,
            capture_ms: 0.394,
        },
        Geometry {
            label: "qwen3_5 dense block  12 attn layers x 2 kv x 128",
            layers: 12,
            kv_heads: 2,
            head_size: 128,
            object_bytes: 198_072,
            capture_ms: 0.215,
        },
        Geometry {
            label: "qwen3_5 MoE block  20 attn layers x 2 kv x 128",
            layers: 20,
            kv_heads: 2,
            head_size: 128,
            object_bytes: 329_760,
            capture_ms: 0.201,
        },
    ];

    /// Measured rounds. One round writes one object per variant.
    const ROUNDS: usize = 128;
    /// Discarded rounds up front (first-touch page faults, directory growth,
    /// allocator warm-up).
    const WARMUP_ROUNDS: usize = 8;

    /// [`DEFAULT_QUEUE_DEPTH`].
    const QUEUE_DEPTH: f64 = 8.0;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum SyncMode {
        /// `File::sync_all` — what production did before [`sync_payload`],
        /// kept as the speedup baseline.
        SyncAll,
        /// Production: POSIX `fsync(2)` via [`sync_payload`], no
        /// `F_FULLFSYNC`. Calling the production helper rather than
        /// re-implementing it makes the `(PRODUCTION)` label below a
        /// compile-time fact: reverting `sync_payload` to `sync_all` collapses
        /// this variant onto the baseline and the speedup column reads 1.00x.
        PlainFsync,
        /// `fcntl(F_BARRIERFSYNC)`: ordering barrier, no device flush.
        Barrier,
        /// No durability call at all.
        NoSync,
    }

    impl SyncMode {
        const ALL: [SyncMode; 4] = [
            SyncMode::SyncAll,
            SyncMode::PlainFsync,
            SyncMode::Barrier,
            SyncMode::NoSync,
        ];

        fn label(self) -> &'static str {
            match self {
                SyncMode::SyncAll => "sync_all (was production)",
                SyncMode::PlainFsync => "fsync(2) (PRODUCTION)",
                SyncMode::Barrier => "F_BARRIERFSYNC",
                SyncMode::NoSync => "none",
            }
        }

        fn slug(self) -> &'static str {
            match self {
                SyncMode::SyncAll => "syncall",
                SyncMode::PlainFsync => "fsync",
                SyncMode::Barrier => "barrier",
                SyncMode::NoSync => "nosync",
            }
        }

        fn apply(self, file: &File) -> Result<(), String> {
            match self {
                SyncMode::SyncAll => file
                    .sync_all()
                    .map_err(|e| format!("sync cold-cache file: {e}")),
                SyncMode::PlainFsync => sync_payload(file),
                SyncMode::Barrier => {
                    // SAFETY: `F_BARRIERFSYNC` takes no third argument and the
                    // descriptor is an open, writable regular file owned by
                    // `file` for the whole call.
                    let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_BARRIERFSYNC) };
                    if rc == -1 {
                        return Err(format!(
                            "F_BARRIERFSYNC cold-cache file: {}",
                            std::io::Error::last_os_error()
                        ));
                    }
                    Ok(())
                }
                SyncMode::NoSync => Ok(()),
            }
        }
    }

    /// Display order. The three `of which` rows sit INSIDE `encode_block`, so
    /// the percentage column deliberately sums to more than 100%.
    const PHASE_NAMES: [&str; 10] = [
        "encode_block+checksum",
        "  of which host clone",
        "  of which payload_checksum",
        "  of which serialize",
        "evict_for_write",
        "create_exclusive",
        "write_all",
        "file sync",
        "renameat+index publish",
        "directory fsync",
    ];

    #[derive(Clone, Copy, Default)]
    struct Phases {
        encode: f64,
        encode_clone: f64,
        encode_checksum: f64,
        encode_serialize: f64,
        evict: f64,
        create: f64,
        write: f64,
        sync: f64,
        commit: f64,
        dir_fsync: f64,
        total: f64,
    }

    impl Phases {
        fn phase(&self, index: usize) -> f64 {
            match index {
                0 => self.encode,
                1 => self.encode_clone,
                2 => self.encode_checksum,
                3 => self.encode_serialize,
                4 => self.evict,
                5 => self.create,
                6 => self.write,
                7 => self.sync,
                8 => self.commit,
                9 => self.dir_fsync,
                _ => unreachable!("phase index out of range"),
            }
        }
    }

    fn ms(d: Duration) -> f64 {
        d.as_secs_f64() * 1e3
    }

    fn pct(part: f64, whole: f64) -> f64 {
        if whole <= 0.0 {
            0.0
        } else {
            part / whole * 100.0
        }
    }

    /// Nearest-rank percentile over an already-sorted slice.
    fn percentile(sorted: &[f64], p: f64) -> f64 {
        if sorted.is_empty() {
            return f64::NAN;
        }
        let rank = (p * sorted.len() as f64).ceil().max(1.0) as usize;
        sorted[rank.min(sorted.len()) - 1]
    }

    fn summarize(values: &[f64]) -> (f64, f64, f64) {
        if values.is_empty() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("durations are never NaN"));
        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        (percentile(&sorted, 0.5), percentile(&sorted, 0.9), mean)
    }

    /// Whether a compile is running right now. Recorded as a per-round LABEL,
    /// not used as a gate: the background build belongs to another worktree
    /// and may outlive this run.
    fn compiler_running() -> bool {
        ["rustc", "cargo"].iter().any(|name| {
            std::process::Command::new("/usr/bin/pgrep")
                .args(["-x", name])
                .output()
                .map(|out| out.status.success())
                .unwrap_or(false)
        })
    }

    /// A block at one family's real geometry. Layer bytes differ per layer
    /// so no encoder or checksum can collapse them.
    fn bench_block(geometry: &Geometry, key: ColdCacheKey, tokens: Vec<u32>) -> ColdCacheBlock {
        let per_side = geometry.per_side();
        let layers = (0..geometry.layers)
            .map(|layer| ColdLayerBlock {
                keys: (0..per_side).map(|i| ((i + layer) % 251) as u8).collect(),
                values: (0..per_side)
                    .map(|i| (250 - ((i + layer) % 251)) as u8)
                    .collect(),
            })
            .collect();
        ColdCacheBlock {
            key,
            fingerprint: fingerprint(),
            tokens,
            layout: ColdCacheLayout {
                block_size: BENCH_BLOCK_SIZE as u32,
                num_layers: geometry.layers as u32,
                num_kv_heads: geometry.kv_heads as u32,
                head_size: geometry.head_size as u32,
                cache_dtype: "BFloat16".to_string(),
                key_bytes_per_layer: per_side,
                value_bytes_per_layer: per_side,
            },
            layers,
        }
    }

    /// Instrumented clone of [`encode_block`] (cold_cache.rs:2008-2051),
    /// split into the three costs it hides: the host clone that materialises
    /// every layer's K/V bytes into an owned `Vec` per tensor, the SHA-256
    /// over that payload, and the safetensors `serialize` that concatenates
    /// it all into the final buffer. Byte-for-byte identical output.
    fn encode_block_instrumented(
        block: &ColdCacheBlock,
        phases: &mut Phases,
    ) -> Result<Vec<u8>, String> {
        let t = Instant::now();
        let token_bytes: Vec<u8> = block.tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut owned: Vec<(String, Vec<u8>)> = Vec::with_capacity(1 + block.layers.len() * 2);
        owned.push(("tokens".to_string(), token_bytes));
        for (i, layer) in block.layers.iter().enumerate() {
            owned.push((format!("layer.{i}.key"), layer.keys.clone()));
            owned.push((format!("layer.{i}.value"), layer.values.clone()));
        }
        phases.encode_clone = ms(t.elapsed());

        let t = Instant::now();
        let checksum = payload_checksum(&owned);
        phases.encode_checksum = ms(t.elapsed());

        let t = Instant::now();
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
        let bytes =
            serialize(views.map_err(|e| e.to_string())?, Some(metadata)).map_err(|e| e.to_string());
        phases.encode_serialize = ms(t.elapsed());
        bytes
    }

    /// Instrumented clone of [`persist_block`] + [`persist_encoded`]
    /// (cold_cache.rs:1695-1776). Phase boundaries and lock scopes match the
    /// production body exactly — in particular the index guard spans
    /// `renameat`, the index publish AND the directory fsync — with only the
    /// payload-durability call parameterised. Production is untouched.
    fn persist_instrumented(
        shared: &Shared,
        block: &ColdCacheBlock,
        mode: SyncMode,
    ) -> Result<Phases, String> {
        let mut phases = Phases::default();
        let started = Instant::now();

        let t = Instant::now();
        block.validate()?;
        let bytes = encode_block_instrumented(block, &mut phases)?;
        phases.encode = ms(t.elapsed());

        let key = block.key;
        let group = ColdGroup::Kv;
        let size = bytes.len() as u64;

        let t = Instant::now();
        evict_for_write(shared, size)?;
        phases.evict = ms(t.elapsed());

        let destination = object_file_name(&key, group);
        let temp = format!(
            ".{}.{}.{}.tmp",
            key.to_hex(),
            std::process::id(),
            now_tick()
        );

        let t = Instant::now();
        let mut file = shared.root.create_exclusive(&temp)?;
        phases.create = ms(t.elapsed());

        let outcome = (|| -> Result<(), String> {
            let t = Instant::now();
            file.write_all(&bytes)
                .map_err(|e| format!("write cold-cache file: {e}"))?;
            phases.write = ms(t.elapsed());

            let t = Instant::now();
            mode.apply(&file)?;
            phases.sync = ms(t.elapsed());

            let t = Instant::now();
            let mut index = shared
                .index
                .lock()
                .map_err(|_| "cold-cache index mutex poisoned".to_string())?;
            // No stats bump on the rename error: this harness calls the persist
            // body directly instead of going through the writer loop, and the
            // loop is where `write_errors` is counted. Matching production here
            // would count an error production counts elsewhere.
            shared.root.rename(&temp, &destination)?;
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
            phases.commit = ms(t.elapsed());

            let t = Instant::now();
            shared.sync()?;
            phases.dir_fsync = ms(t.elapsed());
            Ok(())
        })();
        if let Err(error) = outcome {
            let _ = shared.root.unlink(&temp);
            return Err(error);
        }
        shared
            .stats
            .bytes_written
            .fetch_add(size, Ordering::Relaxed);
        phases.total = ms(started.elapsed());
        Ok(phases)
    }

    fn report_variant(mode: SyncMode, samples: &[Phases], label: &str) {
        if samples.is_empty() {
            eprintln!("\n=== {} [{label}] === no samples", mode.label());
            return;
        }
        let totals: Vec<f64> = samples.iter().map(|s| s.total).collect();
        let (median_total, p90_total, mean_total) = summarize(&totals);
        eprintln!(
            "\n=== {} [{label}] === n={}   Tw median {:.3} ms  p90 {:.3} ms  mean {:.3} ms",
            mode.label(),
            samples.len(),
            median_total,
            p90_total,
            mean_total
        );
        eprintln!("  phase                     median      p90     mean   %of median Tw");
        for (index, name) in PHASE_NAMES.iter().enumerate() {
            let values: Vec<f64> = samples.iter().map(|s| s.phase(index)).collect();
            let (median, p90, mean) = summarize(&values);
            eprintln!(
                "  {name:<24}{median:8.3} {p90:8.3} {mean:8.3}   {:5.1}%",
                pct(median, median_total)
            );
        }
        eprintln!("  ---------------------------------------------------------------");
        eprintln!(
            "  {:<24}{:8.3} {:8.3} {:8.3}   100.0%",
            "Tw (total)", median_total, p90_total, mean_total
        );
    }

    /// Frontier fixed point `N = (Q+1)/(1 - Tc/Tw)`: how many blocks a
    /// back-to-back producer can enqueue before the bounded queue drops one.
    /// `None` once the writer is at least as fast as the producer, where the
    /// backlog never grows and no drop is reachable.
    fn frontier(tc: f64, tw: f64, q: f64) -> Option<f64> {
        let slack = 1.0 - tc / tw;
        (slack > 0.0).then(|| (q + 1.0) / slack)
    }

    /// Inverse of [`frontier`]: the queue depth that pushes the first drop out
    /// past `n` blocks. `None` when no depth is needed.
    fn required_depth(tc: f64, tw: f64, n: f64) -> Option<f64> {
        let slack = 1.0 - tc / tw;
        (slack > 0.0).then(|| (n * slack - 1.0).max(0.0))
    }

    #[test]
    #[ignore = "measurement harness; run explicitly with --ignored"]
    fn bench_write_path_phase_decomposition() {
        eprintln!(
            "sampling: {ROUNDS} measured rounds (+{WARMUP_ROUNDS} warmup) per geometry, \
             round-robin over {} variants, order rotated one position per round; one dedicated \
             cache root per variant so directory size and index size stay identical across them",
            SyncMode::ALL.len()
        );
        for geometry in &GEOMETRIES {
            measure_geometry(geometry);
        }
    }

    fn measure_geometry(geometry: &Geometry) {
        let payload_bytes = geometry.payload_bytes();
        eprintln!(
            "\n\n############ geometry: {} x {BENCH_BLOCK_SIZE} tok bf16 = {payload_bytes} B \
             payload/block ({:.3} MB), {} B encoded ############",
            geometry.label,
            payload_bytes as f64 / 1e6,
            geometry.object_bytes
        );

        // Fidelity gate: if the instrumented encoder does not produce the
        // object the production encoder produces, every number below is
        // measuring the wrong work. Compared through `decode_block` rather
        // than byte-wise because safetensors emits `__metadata__` in
        // `HashMap` iteration order, which differs between two calls.
        {
            let probe_tokens: Vec<u32> = (0..BENCH_BLOCK_SIZE as u32).collect();
            let probe_key =
                ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &probe_tokens, &[], 0, 0);
            let probe = bench_block(geometry, probe_key, probe_tokens);
            let mut scratch = Phases::default();
            let instrumented = encode_block_instrumented(&probe, &mut scratch).unwrap();
            let production = encode_block(&probe).unwrap();
            assert_eq!(
                instrumented.len(),
                production.len(),
                "instrumented encoder must match encode_block's byte length"
            );
            let drift = (production.len() as f64 - geometry.object_bytes as f64).abs()
                / geometry.object_bytes as f64;
            assert!(
                drift < 0.01,
                "{} encodes to {} B but stands in for a {} B object ({:.2}% off)",
                geometry.label,
                production.len(),
                geometry.object_bytes,
                drift * 100.0
            );
            assert_eq!(
                decode_block(&instrumented, probe_key, fingerprint()).unwrap(),
                decode_block(&production, probe_key, fingerprint()).unwrap(),
                "instrumented encoder must round-trip identically to encode_block"
            );
        }

        // One root per variant. Sharing a root would make the directory (and
        // so the directory fsync, and the index) 4x larger than production for
        // the same per-variant sample count, and would let one variant's
        // eviction pressure land on another's measurement.
        let roots: Vec<PathBuf> = SyncMode::ALL
            .iter()
            .map(|mode| temp_root(&format!("bench-write-{}-{}", geometry.layers, mode.slug())))
            .collect();
        // 8 GiB quota against ~235 MB written per root: eviction never fires,
        // so `evict_for_write` here is its steady-state no-evict cost
        // (fstatvfs + index mutex), which is what the writer pays on every
        // object until the cache is actually full.
        let managers: Vec<ColdCacheManager> = roots
            .iter()
            .map(|root| ColdCacheManager::open_at(root.clone(), 8 * GIB, 0, 8).unwrap())
            .collect();

        let mut samples: Vec<Vec<Phases>> = vec![Vec::new(); SyncMode::ALL.len()];
        let mut quiet_samples: Vec<Vec<Phases>> = vec![Vec::new(); SyncMode::ALL.len()];
        let mut quiet_rounds = 0usize;
        let mut serial = 0usize;

        for round in 0..(WARMUP_ROUNDS + ROUNDS) {
            let measured = round >= WARMUP_ROUNDS;
            let busy_before = compiler_running();
            for step in 0..SyncMode::ALL.len() {
                // Rotate so every variant occupies every position in the round
                // an equal number of times.
                let slot = (round + step) % SyncMode::ALL.len();
                let mode = SyncMode::ALL[slot];
                let tokens: Vec<u32> = (0..BENCH_BLOCK_SIZE)
                    .map(|t| (serial as u32) * 1000 + t as u32)
                    .collect();
                let key = ColdCacheKey::chain(
                    ColdGroup::Kv,
                    fingerprint(),
                    None,
                    &tokens,
                    &[],
                    0,
                    serial,
                );
                serial += 1;
                let block = bench_block(geometry, key, tokens);
                let phases = persist_instrumented(&managers[slot].shared, &block, mode)
                    .expect("bench write must persist");
                if measured {
                    samples[slot].push(phases);
                }
            }
            let quiet = !busy_before && !compiler_running();
            if measured && quiet {
                quiet_rounds += 1;
                for (slot, per_mode) in quiet_samples.iter_mut().enumerate() {
                    per_mode.push(*samples[slot].last().expect("round pushed a sample"));
                }
            }
        }

        eprintln!(
            "\nbackground load: {quiet_rounds}/{ROUNDS} measured rounds had NO rustc/cargo running \
             both immediately before and immediately after the round"
        );

        for (slot, mode) in SyncMode::ALL.iter().enumerate() {
            report_variant(*mode, &samples[slot], "all samples");
        }
        if quiet_rounds > 0 && quiet_rounds < ROUNDS {
            for (slot, mode) in SyncMode::ALL.iter().enumerate() {
                report_variant(*mode, &quiet_samples[slot], "quiet subset");
            }
        }

        for (set_label, set) in [("all samples", &samples), ("quiet subset", &quiet_samples)] {
            if set.iter().all(|s| s.is_empty()) {
                continue;
            }
            eprintln!(
                "\n=== Tw summary + frontier N, {set_label} (Tc = {} ms) ===",
                geometry.capture_ms
            );
            eprintln!(
                "  variant                  Tw med   Tw p90   speedup   N@Q=8   N@Q=8 (p90)   \
                 Q for N=128   Q for N=512"
            );
            let baseline = {
                let totals: Vec<f64> = set[0].iter().map(|s| s.total).collect();
                summarize(&totals).0
            };
            for (slot, mode) in SyncMode::ALL.iter().enumerate() {
                let totals: Vec<f64> = set[slot].iter().map(|s| s.total).collect();
                if totals.is_empty() {
                    continue;
                }
                let (median, p90, _) = summarize(&totals);
                let render = |value: Option<f64>| match value {
                    Some(v) => format!("{v:.1}"),
                    None => "never".to_string(),
                };
                eprintln!(
                    "  {:<22}{:8.3} {:8.3}   {:6.2}x  {:>7} {:>13} {:>13} {:>13}",
                    mode.label(),
                    median,
                    p90,
                    baseline / median,
                    render(frontier(geometry.capture_ms, median, QUEUE_DEPTH)),
                    render(frontier(geometry.capture_ms, p90, QUEUE_DEPTH)),
                    render(required_depth(geometry.capture_ms, median, 128.0)),
                    render(required_depth(geometry.capture_ms, median, 512.0)),
                );
            }
            eprintln!(
                "  N is the number of consecutive captured blocks before the bounded queue drops \
                 one. A 2048-token prompt at block_size 16 is 128 blocks, 8192 tokens is 512."
            );

            // Every variant above is still dominated by SHA-256 once the
            // device flush is gone. Subtracting the MEASURED per-sample
            // checksum time bounds what any checksum change could ever buy:
            // no cheaper hash can beat a free one. This is a bound derived
            // from measured data, not a measurement of a real implementation.
            eprintln!(
                "\n  projection: same samples with the measured payload_checksum time subtracted \
                 (upper bound on any checksum optimisation, e.g. a non-cryptographic hash)"
            );
            eprintln!(
                "  variant                 Tw' med  Tw' p90             N@Q=8   N@Q=8 (p90)   \
                 Q for N=128   Q for N=512"
            );
            for (slot, mode) in SyncMode::ALL.iter().enumerate() {
                let totals: Vec<f64> = set[slot]
                    .iter()
                    .map(|s| (s.total - s.encode_checksum).max(1e-6))
                    .collect();
                if totals.is_empty() {
                    continue;
                }
                let (median, p90, _) = summarize(&totals);
                let render = |value: Option<f64>| match value {
                    Some(v) => format!("{v:.1}"),
                    None => "never".to_string(),
                };
                eprintln!(
                    "  {:<22}{:8.3} {:8.3}            {:>7} {:>13} {:>13} {:>13}",
                    mode.label(),
                    median,
                    p90,
                    render(frontier(geometry.capture_ms, median, QUEUE_DEPTH)),
                    render(frontier(geometry.capture_ms, p90, QUEUE_DEPTH)),
                    render(required_depth(geometry.capture_ms, median, 128.0)),
                    render(required_depth(geometry.capture_ms, median, 512.0)),
                );
            }
            eprintln!(
                "  \"never\" means Tw <= Tc: the writer keeps up with the producer, the backlog \
                 never grows, and no queue depth is needed at all."
            );
        }

        drop(managers);
        for root in &roots {
            let _ = fs::remove_dir_all(root);
        }
    }

    /// Producer intervals to sweep in [`bench_chain_advance_per_turn`].
    /// Brackets every geometry's measured `Geometry::capture_ms`, because the
    /// frontier `N` is steep in `Tc` and a single point would hide that.
    const PRODUCER_INTERVALS_MS: [f64; 5] = [0.10, 0.20, 0.32, 0.50, 1.00];

    /// Cap on host bytes held as pre-built blocks in one trial.
    const CHAIN_TRIAL_BYTES: usize = 256 * 1024 * 1024;

    /// OBSERVED per-turn chain advance: how many consecutive blocks
    /// `ColdTierWalk::capture_chain` gets to persist before the bounded queue
    /// refuses one and the walk breaks.
    ///
    /// Everything reported about the chain so far has been arithmetic on the
    /// frontier `N = (Q+1)/(1 - Tc/Tw)`. This drives the real
    /// [`ColdCacheManager::enqueue`] against the real writer thread at a fixed
    /// producer interval and counts the accepted prefix directly, so the model
    /// can be checked rather than assumed. No Metal and no weights: the writer
    /// only ever sees owned host bytes, and the capture cost it is racing is
    /// reproduced by the interval.
    ///
    /// This is a measurement, NOT a gate. The separation between `sync_all`
    /// and `fsync(2)` here is a wall-clock effect of a few blocks on a shared
    /// machine, and asserting on it would be a flaky test dressed up as a
    /// guarantee. The defence against someone quietly restoring `sync_all` is
    /// [`sync_payload`]'s doc, the note in `docs/paged-cache.md`, and
    /// `SyncMode::PlainFsync` calling the production helper — not an
    /// assertion here.
    #[test]
    #[ignore = "measurement harness; run explicitly with --ignored"]
    fn bench_chain_advance_per_turn() {
        eprintln!(
            "observed chain advance: blocks accepted by enqueue() before the first Ok(false), \
             driving the real writer thread at queue depth {DEFAULT_QUEUE_DEPTH}"
        );
        for geometry in &GEOMETRIES {
            let cap = (CHAIN_TRIAL_BYTES / geometry.object_bytes).min(600);
            eprintln!(
                "\n=== {} ({} B/object, cap {cap} blocks/trial) ===",
                geometry.label, geometry.object_bytes
            );
            eprintln!(
                "  producer Tc     accepted   elapsed   implied Tw   tokens/turn   \
                 turns to 8192 tok"
            );
            for interval_ms in PRODUCER_INTERVALS_MS {
                let (accepted, elapsed_ms, hit_cap) =
                    chain_advance_trial(geometry, interval_ms, cap);
                // From the frontier relation, the Tw that would produce this
                // observed N: Tw = Tc / (1 - (Q+1)/N).
                let implied_tw = {
                    let slack = 1.0 - (QUEUE_DEPTH + 1.0) / accepted as f64;
                    if slack > 0.0 {
                        format!("{:8.3}", interval_ms / slack)
                    } else {
                        "       -".to_string()
                    }
                };
                let tokens = accepted * BENCH_BLOCK_SIZE;
                eprintln!(
                    "  {interval_ms:8.2} ms {:10}{}{elapsed_ms:9.1} ms {implied_tw}   \
                     {tokens:11}   {:16.1}",
                    accepted,
                    if hit_cap { "+" } else { " " },
                    512.0 / accepted as f64,
                );
            }
        }
        eprintln!(
            "\n  `+` marks a trial that hit the block cap without ever being refused. \
             8192 tokens at block_size {BENCH_BLOCK_SIZE} is 512 blocks."
        );
    }

    /// The producer cost `Tc` that [`bench_chain_advance_per_turn`] can only
    /// sweep: one `read_block_all_layers` round trip, which is exactly what
    /// [`ColdCacheManager::capture_and_enqueue`] pays on the inference thread
    /// per block. Source of every `Geometry::capture_ms`.
    ///
    /// `Tc` had only ever been measured at the qwen3-0.6b geometry, and the
    /// frontier `N` is steep in it, so the two shipping block sizes were the
    /// biggest unknown in every chain-advance number. Needs Metal; skips
    /// itself when a pool cannot be built.
    #[test]
    #[ignore = "measurement harness; run explicitly with --ignored"]
    fn bench_capture_cost_per_geometry() {
        eprintln!(
            "Tc: one read_block_all_layers round trip per block ({ROUNDS} rounds, \
             +{WARMUP_ROUNDS} warmup)"
        );
        for geometry in &GEOMETRIES {
            let total_blocks = 32u32;
            let config = PagedAttentionConfig {
                block_size: BENCH_BLOCK_SIZE as u32,
                gpu_memory_mb: 4096,
                head_size: geometry.head_size as u32,
                num_kv_heads: geometry.kv_heads as u32,
                num_layers: geometry.layers as u32,
                use_fp8_cache: Some(false),
                max_seq_len: Some(4096),
                max_batch_size: Some(1),
            };
            let pool = match LayerKVPool::new(config, total_blocks, MetalDtype::BFloat16) {
                Ok(pool) => pool,
                Err(e) => {
                    eprintln!("  {}: skipped ({e})", geometry.label);
                    continue;
                }
            };
            let allocator = Mutex::new(BlockAllocator::new(total_blocks, BENCH_BLOCK_SIZE as u32));
            let Some(block) = allocator.lock().unwrap().allocate() else {
                eprintln!("  {}: skipped (no free block)", geometry.label);
                continue;
            };

            // Seed every layer so the readback moves real bytes rather than
            // whatever an untouched buffer holds.
            let per_side = geometry.per_side();
            let keys: Vec<u8> = (0..per_side).map(|i| (i % 251) as u8).collect();
            let values: Vec<u8> = (0..per_side).map(|i| (250 - (i % 251)) as u8).collect();
            for layer in 0..geometry.layers as u32 {
                pool.write_blocks_from_host(layer, &[block.block_id], &keys, &values)
                    .unwrap();
            }

            let mut samples = Vec::with_capacity(ROUNDS);
            for round in 0..(WARMUP_ROUNDS + ROUNDS) {
                let started = Instant::now();
                let read = pool.read_block_all_layers(block.block_id).unwrap();
                let elapsed = ms(started.elapsed());
                assert_eq!(
                    read.len(),
                    geometry.layers,
                    "readback must cover every layer"
                );
                if round >= WARMUP_ROUNDS {
                    samples.push(elapsed);
                }
            }
            let (median, p90, mean) = summarize(&samples);
            eprintln!(
                "  {:<52} Tc median {median:.3} ms  p90 {p90:.3} ms  mean {mean:.3} ms",
                geometry.label
            );
        }
    }

    /// One trial: feed pre-built blocks to `enqueue` exactly `interval_ms`
    /// apart and return `(accepted, elapsed_ms, hit_cap)`.
    ///
    /// Blocks are built up front so allocation never counts against the
    /// producer interval, and the pacing is a spin rather than a sleep
    /// because `thread::sleep` cannot resolve 0.1 ms on macOS.
    fn chain_advance_trial(
        geometry: &Geometry,
        interval_ms: f64,
        cap: usize,
    ) -> (usize, f64, bool) {
        let mut blocks: Vec<ColdCacheBlock> = Vec::with_capacity(cap);
        for serial in 0..cap {
            let tokens: Vec<u32> = (0..BENCH_BLOCK_SIZE)
                .map(|t| (serial as u32) * 1000 + t as u32)
                .collect();
            let key =
                ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &tokens, &[], 0, serial);
            blocks.push(bench_block(geometry, key, tokens));
        }

        let root = temp_root(&format!(
            "bench-chain-{}-{}",
            geometry.layers,
            (interval_ms * 100.0) as u64
        ));
        let manager = ColdCacheManager::open_at(root.clone(), 8 * GIB, 0, DEFAULT_QUEUE_DEPTH)
            .expect("open bench cache");

        let interval = Duration::from_secs_f64(interval_ms / 1000.0);
        let started = Instant::now();
        let mut accepted = 0usize;
        let mut hit_cap = true;
        for (index, block) in blocks.into_iter().enumerate() {
            let due = started + interval.saturating_mul(index as u32);
            while Instant::now() < due {
                std::hint::spin_loop();
            }
            match manager.enqueue(block).expect("writer must stay alive") {
                true => accepted += 1,
                false => {
                    hit_cap = false;
                    break;
                }
            }
        }
        let elapsed_ms = ms(started.elapsed());

        // Let the writer finish so the next trial starts from an idle drive
        // rather than inheriting this one's backlog.
        manager.drain(Duration::from_secs(60));
        drop(manager);
        let _ = fs::remove_dir_all(&root);
        (accepted, elapsed_ms, hit_cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn fingerprint() -> ColdCacheFingerprint {
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

    pub(super) fn temp_root(name: &str) -> PathBuf {
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

    /// CONTROL for the boundary-scaled axis, and the `qwen3_5` /
    /// `qwen3_5_moe` no-change proof: those families build their policy with
    /// [`ColdSidecarPolicy::new`], whose `expected_at` stamps the candidate
    /// boundary and NOTHING else. `dims` and `bytes_per_tensor` stay frozen at
    /// the template's values at every boundary, shallow or deep, so their
    /// restore walk keeps probing exactly the layouts it probes today.
    ///
    /// This test must be green both BEFORE and AFTER a scaled axis exists.
    #[test]
    fn unscaled_policy_stamps_the_boundary_and_nothing_else() {
        let template = sidecar_layout(ColdGroup::GdnState);
        let policy =
            ColdSidecarPolicy::new(template.clone()).expect("a legal geometry must build a policy");
        for boundary in [1u32, 16, 17, 32, 512, 4096, u32::MAX] {
            let expected = policy.expected_at(boundary);
            assert_eq!(
                expected,
                ColdSidecarLayout {
                    boundary_tokens: boundary,
                    ..template.clone()
                },
                "an unscaled policy must vary only `boundary_tokens` (at {boundary})"
            );
            assert_eq!(expected.dims, template.dims);
            assert_eq!(expected.bytes_per_tensor, template.bytes_per_tensor);
        }
    }

    /// A rotating-window-shaped template: `[batch, kv_heads, window, head_dim]`
    /// with the token axis at index 2, sized so `bytes_per_tensor` is the exact
    /// bf16 byte count.
    fn scaled_sidecar_layout() -> ColdSidecarLayout {
        ColdSidecarLayout {
            group: ColdGroup::SlidingWindow,
            boundary_tokens: 0,
            num_layers: 4,
            tensors_per_layer: 2,
            dtype: "BFloat16".to_string(),
            dims: vec![1, 2, 1024, 4],
            bytes_per_tensor: 2 * 1024 * 4 * 2,
        }
    }

    /// BACKWARD-COMPAT PROOF, and the reason a scaled axis needs no
    /// fingerprint or on-disk format bump: at and above the template extent
    /// `min(b, extent) == extent`, so a scaled policy stamps byte-for-byte
    /// what the unscaled one stamps, and every sidecar already on disk still
    /// compares equal. Only BELOW the extent does anything move — and there,
    /// nothing was ever written.
    #[test]
    fn boundary_scaled_policy_is_the_identity_at_and_above_the_template_extent() {
        let template = scaled_sidecar_layout();
        let scaled = ColdSidecarPolicy::new_boundary_scaled(template.clone(), 2)
            .expect("a legal geometry with an in-range axis must build a policy");
        let unscaled =
            ColdSidecarPolicy::new(template.clone()).expect("the same geometry, unscaled");

        for boundary in [1024u32, 1040, 2048, 65536, u32::MAX] {
            assert_eq!(
                scaled.expected_at(boundary),
                unscaled.expected_at(boundary),
                "scaling must be the identity at boundary {boundary}"
            );
            assert_eq!(scaled.expected_at(boundary).dims, template.dims);
            assert_eq!(
                scaled.expected_at(boundary).bytes_per_tensor,
                template.bytes_per_tensor
            );
        }

        // Below the extent exactly one axis moves, and the byte length moves
        // with it — proportionally, never rounded.
        let half = scaled.expected_at(512);
        assert_eq!(half.boundary_tokens, 512);
        assert_eq!(half.dims, vec![1, 2, 512, 4]);
        assert_eq!(half.bytes_per_tensor, template.bytes_per_tensor / 2);
        let quarter = scaled.expected_at(256);
        assert_eq!(quarter.dims, vec![1, 2, 256, 4]);
        assert_eq!(quarter.bytes_per_tensor, template.bytes_per_tensor / 4);
        // Everything that is not the scaled axis stays geometry.
        assert_eq!(half.group, template.group);
        assert_eq!(half.num_layers, template.num_layers);
        assert_eq!(half.tensors_per_layer, template.tensors_per_layer);
        assert_eq!(half.dtype, template.dtype);

        // A zero boundary yields an empty payload — refused by
        // `ColdSidecar::validate`, and matched by no stored sidecar.
        assert_eq!(scaled.expected_at(0).bytes_per_tensor, 0);
    }

    /// The three facts that make `expected_at` infallible are checked up
    /// front, so an unusable scaled policy can never be installed and then
    /// silently mis-size every probe.
    #[test]
    fn boundary_scaled_policy_refuses_an_unusable_axis() {
        // Out of range: `dims` has 4 entries.
        assert!(ColdSidecarPolicy::new_boundary_scaled(scaled_sidecar_layout(), 4).is_err());
        assert!(
            ColdSidecarPolicy::new_boundary_scaled(scaled_sidecar_layout(), usize::MAX).is_err()
        );
        // The extent must divide `bytes_per_tensor`, or the per-row cost —
        // and so every scaled length — would be a rounded lie.
        assert!(
            ColdSidecarPolicy::new_boundary_scaled(
                ColdSidecarLayout {
                    bytes_per_tensor: 2 * 1024 * 4 * 2 + 1,
                    ..scaled_sidecar_layout()
                },
                2,
            )
            .is_err()
        );
        // Every geometry bound `new` enforces still applies.
        assert!(
            ColdSidecarPolicy::new_boundary_scaled(
                ColdSidecarLayout {
                    group: ColdGroup::Kv,
                    ..scaled_sidecar_layout()
                },
                2,
            )
            .is_err()
        );
        assert!(
            ColdSidecarPolicy::new_boundary_scaled(
                ColdSidecarLayout {
                    dims: vec![1, 2, 0, 4],
                    ..scaled_sidecar_layout()
                },
                2,
            )
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

    /// The failure mode [`sync_payload`] deliberately admits: a KV block whose
    /// bytes on disk are not the bytes we wrote, while its length, its
    /// safetensors framing and every metadata field are still intact. Only
    /// the payload checksum can tell, so this is the end-to-end proof that a
    /// torn object is a MISS that prunes — never partial KV handed to
    /// inference.
    ///
    /// Blocks had no such test. `corrupt_file_fails_open_and_is_removed`
    /// writes non-safetensors bytes (rejected by the parser) and
    /// `decode_rejects_forged_huge_num_layers` trips the tensor-count guard;
    /// neither reaches the checksum comparison.
    #[test]
    fn a_bit_flipped_block_is_a_miss_that_prunes_and_counts_corruption() {
        let root = temp_root("block-bitflip");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let expected = block(key);

        assert!(manager.enqueue(expected.clone()).unwrap());
        assert!(manager.drain(Duration::from_secs(5)));
        let path = root.join(format!("{}.safetensors", key.to_hex()));
        assert_eq!(
            manager.load(key, fingerprint()),
            Some(expected),
            "the fixture must be live before it is damaged"
        );

        // Flip one payload byte in place: same inode (so pruning may clear the
        // very entry that failed) and same length, so the safetensors header,
        // the `1 + 2*num_layers` tensor-count check and `validate()` all still
        // pass. The checksum is the only gate this flip can trip, which is
        // what stops the test going green for some other reason.
        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        fs::write(&path, &bytes).unwrap();

        let before = manager.stats();
        assert_eq!(
            manager.load(key, fingerprint()),
            None,
            "a torn block must miss, not hand back the wrong KV bytes"
        );
        let after = manager.stats();
        assert_eq!(after.corruptions, before.corruptions + 1);
        assert_eq!(after.misses, before.misses + 1);
        assert!(!path.exists(), "the torn block must be pruned");
        assert!(!manager.contains(&key));

        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    /// An object that predates the payload checksum must be refused, not
    /// trusted. The whole reason `sync_payload` is affordable is that every
    /// read re-derives the checksum, so a decoder that verified only when the
    /// field happened to be present would silently reopen the hole for every
    /// object written before it existed.
    #[test]
    fn a_block_without_a_payload_checksum_is_refused() {
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let source = block(key);

        // Everything `encode_block` writes except the `checksum` entry, so the
        // object is well-formed on every other axis and the missing field is
        // the only thing that can reject it.
        let token_bytes: Vec<u8> = source.tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut owned: Vec<(String, Vec<u8>)> = vec![("tokens".to_string(), token_bytes)];
        for (i, layer) in source.layers.iter().enumerate() {
            owned.push((format!("layer.{i}.key"), layer.keys.clone()));
            owned.push((format!("layer.{i}.value"), layer.values.clone()));
        }
        let views: Vec<_> = owned
            .iter()
            .map(|(name, data)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::U8, vec![data.len()], data).unwrap(),
                )
            })
            .collect();
        let mut metadata = HashMap::new();
        metadata.insert("abi".to_string(), CACHE_ABI.to_string());
        metadata.insert("key".to_string(), key.to_hex());
        metadata.insert("fingerprint".to_string(), fingerprint().to_hex());
        metadata.insert(
            "block_size".to_string(),
            source.layout.block_size.to_string(),
        );
        metadata.insert(
            "num_layers".to_string(),
            source.layout.num_layers.to_string(),
        );
        metadata.insert(
            "num_kv_heads".to_string(),
            source.layout.num_kv_heads.to_string(),
        );
        metadata.insert("head_size".to_string(), source.layout.head_size.to_string());
        metadata.insert("cache_dtype".to_string(), source.layout.cache_dtype.clone());
        metadata.insert(
            "key_bytes".to_string(),
            source.layout.key_bytes_per_layer.to_string(),
        );
        metadata.insert(
            "value_bytes".to_string(),
            source.layout.value_bytes_per_layer.to_string(),
        );
        let bytes = serialize(views, Some(metadata)).unwrap();

        assert!(
            decode_block(&bytes, key, fingerprint()).is_err(),
            "a block carrying no payload checksum must not decode"
        );

        // And through the public loader: a miss that is counted and pruned,
        // exactly like a damaged object, so an unverifiable generation cannot
        // linger in the cache.
        let root = temp_root("block-no-checksum");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let path = root.join(format!("{}.safetensors", key.to_hex()));
        fs::write(&path, &bytes).unwrap();
        assert_eq!(
            manager.load(key, fingerprint()),
            None,
            "an unchecksummed object must never be read as if it had one"
        );
        assert_eq!(manager.stats().corruptions, 1);
        assert!(!path.exists(), "the unverifiable object must be pruned");

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

    /// Park the writer inside its post-rename directory fsync for `commit_ms`
    /// per object, so the queue drains at a known, slow, controllable rate.
    /// Returns the flag that releases it; hold it alive for the test's body.
    ///
    /// Returns only once the writer is PROVABLY parked. Installing the hook
    /// alone does not park anything — the hook runs on the writer's first
    /// commit, and until then the writer is sitting in `recv` popping whatever
    /// `saturate` sends. A caller that raced that lost roughly one run in
    /// three: `saturate` filled the two-slot channel before the writer popped
    /// anything, the writer then popped one and freed a slot, and the refusal
    /// the test exists to pin never happened. So a wedge job is pushed here and
    /// waited on until the hook reports it has been entered; from that point the
    /// writer cannot pop, and a full queue stays full.
    fn slow_writer(
        manager: &ColdCacheManager,
        commit_ms: u64,
    ) -> Arc<std::sync::atomic::AtomicBool> {
        let release = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let parked = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let for_writer = Arc::clone(&release);
        let parked_by_writer = Arc::clone(&parked);
        *manager.shared.dir_sync_override.lock().unwrap() = Some(Box::new(move || {
            parked_by_writer.store(true, Ordering::Relaxed);
            let until = Instant::now() + Duration::from_millis(commit_ms);
            while Instant::now() < until && !for_writer.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(1));
            }
            Ok(())
        }));

        // The wedge. Sent on the raw channel for the same reason `saturate`
        // bypasses the public API: what these tests measure IS the public
        // admission path, so the fixture must not consume any of its budget.
        let toks = vec![u32::MAX, u32::MAX - 1, u32::MAX - 2, u32::MAX - 3];
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &toks, &[], 0, 0);
        let mut wedge = block(key);
        wedge.tokens = toks;
        assert!(
            manager.sender.try_send(WriteJob::Block(wedge)).is_ok(),
            "the wedge job must fit an empty queue"
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        while !parked.load(Ordering::Relaxed) {
            assert!(
                Instant::now() < deadline,
                "the writer never reached the commit dir-sync the wedge installs"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        release
    }

    /// Fill the queue until the RAW channel refuses, proving it is full behind
    /// the parked writer. Returns how many were accepted.
    ///
    /// Deliberately bypasses `enqueue`: every test below exists to pin some
    /// property of `send_before`, and a fixture that reached saturation THROUGH
    /// `send_before` would be defeated by the very mutations it must catch —
    /// the helper would block or over-accept and panic before the test's own
    /// assertion ever ran.
    fn saturate(manager: &ColdCacheManager, depth: usize) -> usize {
        let mut accepted = 0usize;
        for i in 0..(depth as u32 + 8) {
            let toks = vec![i, i + 100, i + 200, i + 300];
            let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &toks, &[], 0, 0);
            let mut candidate = block(key);
            candidate.tokens = toks;
            match manager.sender.try_send(WriteJob::Block(candidate)) {
                Ok(()) => accepted += 1,
                Err(TrySendError::Full(_)) => return accepted,
                Err(TrySendError::Disconnected(_)) => panic!("writer thread is gone"),
            }
        }
        panic!("the bounded queue never refused a block");
    }

    // `enqueue` is the non-blocking admission API and MUST stay that way. The
    // deadline-aware `send_before` it now delegates to degenerates to exactly
    // one `try_send` when the deadline is already past, so a full queue behind
    // a stuck writer is refused promptly and counted as a drop.
    //
    // Without this, a refactor that gave `enqueue` a non-zero default deadline
    // would turn every existing caller blocking behind the writer's back —
    // silently, since the return value would still be `Ok(true)`. The wedge is
    // short for the same reason as in `a_wedged_writer_cannot_hang_a_bounded_enqueue`:
    // under that mutation `saturate` itself starts blocking, and it must reach
    // a verdict rather than stall the suite.
    #[test]
    fn enqueue_keeps_its_non_blocking_contract() {
        let root = temp_root("enqueue-nonblocking");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let release = slow_writer(&manager, 400);
        saturate(&manager, 2);

        let before = manager.stats().queue_drops;
        let toks = vec![9u32, 8, 7, 6];
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &toks, &[], 0, 0);
        let mut candidate = block(key);
        candidate.tokens = toks;

        let start = Instant::now();
        let accepted = manager.enqueue(candidate).unwrap();
        let elapsed = start.elapsed();

        assert!(!accepted, "a full queue must refuse a non-blocking enqueue");
        assert!(
            elapsed < Duration::from_millis(50),
            "enqueue must not wait for a slot, took {elapsed:?}"
        );
        assert_eq!(
            manager.stats().queue_drops - before,
            1,
            "the refusal must be counted"
        );

        release.store(true, Ordering::Relaxed);
        *manager.shared.dir_sync_override.lock().unwrap() = None;
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    // A saturated queue must not cost a family its state sidecar.
    //
    // Every hybrid family offers its sidecar microseconds after its K/V
    // capture walk returns, and that walk now SPENDS its budget filling the
    // queue — so a non-blocking sidecar offer is guaranteed to arrive at a full
    // queue. A dropped sidecar is worse than a dropped block: the restore
    // reconciles down to the deepest boundary a validated sidecar backs, so
    // losing it makes the turn's whole persisted chain unusable.
    #[test]
    fn a_saturated_queue_still_admits_the_family_sidecar() {
        let root = temp_root("sidecar-admit");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        // 30 ms per commit: the queue only frees a slot by draining, so an
        // admission that does not wait cannot possibly succeed here.
        let release = slow_writer(&manager, 30);
        saturate(&manager, 2);

        let key = ColdCacheKey::chain(
            ColdGroup::GdnState,
            fingerprint(),
            None,
            &[1, 2, 3, 4],
            &[],
            0,
            0,
        );
        let start = Instant::now();
        let accepted = manager
            .enqueue_sidecar_before(
                sidecar(key, ColdGroup::GdnState),
                Instant::now() + Duration::from_millis(2_000),
            )
            .unwrap();

        assert!(
            accepted,
            "the sidecar must wait out a slot rather than be dropped ({:?})",
            start.elapsed()
        );

        release.store(true, Ordering::Relaxed);
        *manager.shared.dir_sync_override.lock().unwrap() = None;
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    // The bounded wait must be BOUNDED. A writer wedged for far longer than the
    // deadline must yield a refusal at the deadline — this is what keeps a
    // stalled storage device from turning a capture walk into unbounded turn
    // tail (`docs/paged-cache.md`).
    //
    // The wedge is 2 s rather than "forever" on purpose: a walk that lost its
    // deadline check would HANG on a forever-wedge, and a hung suite is a much
    // worse signal than a failed assert. At 2 s the unbounded version instead
    // succeeds ~20x past the deadline and both asserts below fire.
    #[test]
    fn a_wedged_writer_cannot_hang_a_bounded_enqueue() {
        let root = temp_root("wedged-writer");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let release = slow_writer(&manager, 2_000);
        // Exactly the depth, which is only true because `slow_writer` returned
        // with the writer already parked. A writer still popping absorbs one of
        // these and leaves a slot free behind the "full" queue, and the refusal
        // below then never happens.
        assert_eq!(
            saturate(&manager, 2),
            2,
            "the queue must be full at its depth before the deadline is measured"
        );

        let deadline_ms = 100u64;
        let key = ColdCacheKey::chain(
            ColdGroup::GdnState,
            fingerprint(),
            None,
            &[5, 6, 7, 8],
            &[],
            0,
            0,
        );
        let start = Instant::now();
        let accepted = manager
            .enqueue_sidecar_before(
                sidecar(key, ColdGroup::GdnState),
                Instant::now() + Duration::from_millis(deadline_ms),
            )
            .unwrap();
        let elapsed = start.elapsed();

        assert!(
            !accepted,
            "a writer wedged past the deadline must produce a refusal"
        );
        assert!(
            elapsed >= Duration::from_millis(deadline_ms),
            "the wait must actually run to the deadline, took {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_millis(deadline_ms) + Duration::from_millis(400),
            "the wait must STOP at the deadline, took {elapsed:?}"
        );

        release.store(true, Ordering::Relaxed);
        *manager.shared.dir_sync_override.lock().unwrap() = None;
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    // A refused sidecar is a queue drop. `queue_drops` answers "did the write
    // queue refuse work?", and the queue is shared: a sidecar occupies a slot
    // exactly as a block does.
    //
    // Scoping this counter to blocks would read like a tightening and would
    // instead hide the failure this whole bounded-wait path exists to prevent.
    // `ColdSidecarTelemetry.queue_drops` is not among the sidecar fields the
    // dashboard stores, so a run whose capture walk fills the queue and starves
    // the sidecar every turn — the chain then restores nothing — would report a
    // perfectly healthy queue and leave the drop alarm silent.
    //
    // `saturate` fills the queue through the raw sender rather than `enqueue`,
    // so the counters below see the sidecar and nothing else.
    #[test]
    fn a_refused_sidecar_is_counted_as_a_queue_drop() {
        let root = temp_root("sidecar-queue-drop");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let release = slow_writer(&manager, 2_000);
        assert_eq!(
            saturate(&manager, 2),
            2,
            "the queue must be full before the refusal is measured"
        );

        let key = ColdCacheKey::chain(
            ColdGroup::GdnState,
            fingerprint(),
            None,
            &[4, 3, 2, 1],
            &[],
            0,
            0,
        );
        assert!(
            !manager
                .enqueue_sidecar(sidecar(key, ColdGroup::GdnState))
                .unwrap(),
            "a full queue refuses the non-blocking sidecar offer"
        );

        let stats = manager.stats();
        assert_eq!(
            stats.queue_drops, 1,
            "the refused sidecar is the queue's refusal to count"
        );
        assert_eq!(
            stats.enqueued, 0,
            "nothing was admitted through `send_before`"
        );

        release.store(true, Ordering::Relaxed);
        *manager.shared.dir_sync_override.lock().unwrap() = None;
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

    /// A stale index entry must not cost a LIVE one its place.
    ///
    /// The up-front feasibility estimate counts every indexed byte as
    /// reclaimable, which an entry another process already deleted is not: it
    /// frees zero. With that entry at the LRU head the estimate clears the bar,
    /// the pass prunes it, and the real reclaim then falls short — so a
    /// single-shot check would keep going and evict the valid entry behind it,
    /// destroying warm cache for a write that fails anyway.
    ///
    /// Sized so the distinction is the whole test: `available + A + B` exactly
    /// meets `reserve + incoming`, and `available + B` alone is one block
    /// short.
    #[cfg(unix)]
    #[test]
    fn a_stale_entry_does_not_drag_a_valid_one_down_with_it() {
        let root = temp_root("evict-stale-spares-valid");
        let fp = fingerprint();
        let key_a = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[1], 0, 0);
        let key_b = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[2], 0, 0);
        let key_c = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[3], 0, 0);
        let one = encode_block(&block(key_a)).unwrap().len() as u64;
        let reserve = 5 * one;

        // Quota is roomy, so only the free-space axis is under test.
        let manager = ColdCacheManager::open_at(root.clone(), GIB, reserve, 2).unwrap();
        let base = Arc::new(AtomicU64::new(GIB));
        install_space_override(&manager, &root, &base);

        // A first, so it is the LRU head; B is the valid entry behind it.
        persist_block(&manager.shared, &block(key_a)).unwrap();
        persist_block(&manager.shared, &block(key_b)).unwrap();
        assert!(manager.contains(&key_a) && manager.contains(&key_b));

        // Another process removed A's payload; the index still carries it.
        fs::remove_file(root.join(format!("{}.safetensors", key_a.to_hex()))).unwrap();

        // Only B's bytes remain on disk, so `available` settles at 4 blocks:
        //   4 + (A 1 + B 1) == reserve 5 + incoming 1   -> first check passes
        //   4 +        (B 1) <  reserve 5 + incoming 1   -> second must refuse
        base.store(5 * one, Ordering::Relaxed);
        assert!(
            persist_block(&manager.shared, &block(key_c)).is_err(),
            "a write the survivors cannot make room for must be refused"
        );

        assert!(
            manager.contains(&key_b),
            "the valid entry must survive: clearing it could never have admitted the write"
        );
        assert_eq!(
            manager.load(key_b, fp),
            Some(block(key_b)),
            "and it must still be readable, not merely indexed"
        );
        assert!(
            !manager.contains(&key_a),
            "the dead index entry is still pruned by the pass"
        );
        assert!(!manager.contains(&key_c));
        assert_eq!(
            manager.stats().evictions,
            1,
            "exactly one eviction: the stale entry, and nothing beyond it"
        );

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

    /// An object that cannot fit the quota at all must be refused BEFORE the
    /// LRU loop runs. Otherwise both loop conditions stay true no matter how
    /// much is reclaimed, so the write evicts every entry in turn and then
    /// fails anyway — destroying a warm cache on behalf of a write that could
    /// never have joined it.
    #[test]
    fn oversized_write_is_refused_without_evicting_anything() {
        let root = temp_root("evict-oversized-refused");
        let fp = fingerprint();
        let key_a = ColdCacheKey::chain(ColdGroup::Kv, fp, None, &[1, 2, 3, 4], &[1], 0, 0);
        let one = encode_block(&block(key_a)).unwrap().len() as u64;

        // Quota admits the resident entry but is smaller than a second object,
        // so `incoming` alone can never fit.
        let manager = ColdCacheManager::open_at(root.clone(), one + one / 2, 0, 2).unwrap();
        let base = Arc::new(AtomicU64::new(GIB));
        install_space_override(&manager, &root, &base);

        persist_block(&manager.shared, &block(key_a)).unwrap();
        assert!(manager.contains(&key_a));

        assert!(
            evict_for_write(&manager.shared, one * 4).is_err(),
            "a write larger than the whole quota must be refused"
        );
        assert!(
            manager.contains(&key_a),
            "the resident entry must survive a write that could never fit"
        );
        assert_eq!(manager.stats().evictions, 0);
        assert_eq!(manager.shared.index.lock().unwrap().total_bytes, one);

        // Same guarantee on the disk-space axis: reclaiming everything indexed
        // still cannot clear the reserve floor, so nothing is evicted.
        let space_root = temp_root("evict-oversized-space");
        let roomy = ColdCacheManager::open_at(space_root.clone(), GIB, GIB, 2).unwrap();
        let space = Arc::new(AtomicU64::new(0));
        install_space_override(&roomy, &space_root, &space);
        assert!(
            evict_for_write(&roomy.shared, one).is_err(),
            "a hopeless free-space request must be refused up front"
        );
        assert_eq!(roomy.stats().evictions, 0);

        drop(roomy);
        drop(manager);
        let _ = fs::remove_dir_all(space_root);
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
    fn writer_commit_rename_failure_counts_write_error_and_removes_temp() {
        let root = temp_root("commit-rename-failure");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);
        let path = root.join(format!("{}.safetensors", key.to_hex()));
        fs::create_dir(&path).unwrap();
        fs::write(path.join("marker.txt"), b"marker").unwrap();

        assert!(manager.enqueue(block(key)).unwrap());
        for _ in 0..200 {
            if manager.stats().write_errors >= 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let stats = manager.stats();
        assert_eq!(
            stats.write_errors, 1,
            "a failed commit rename must be counted, not silently discarded"
        );
        assert_eq!(
            stats.queue_drops, 0,
            "the queue accepted this write, so it is not a queue drop: \
             `queue_drops` is admission refusals only"
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

    /// A cache root the process cannot write to must be VISIBLE, not merely
    /// survivable.
    ///
    /// The failure is induced for real — the root is chmod'ed 0500 after the
    /// manager has opened it, so the writer's `openat(O_CREAT)` returns EACCES
    /// from the kernel — rather than simulated through a hook, because the
    /// thing under test is precisely that a genuine storage refusal reaches a
    /// counter.
    ///
    /// Every other counter is asserted to stay at zero on purpose: that
    /// all-zeros row IS the reported bug. An operator whose root was read-only
    /// saw `queue_drops 0`, `corruptions 0`, `bytes_written 0`, an empty
    /// stderr and a successful turn, which is indistinguishable from a healthy
    /// cache that had nothing to write.
    #[cfg(unix)]
    #[test]
    fn a_read_only_root_counts_write_errors_instead_of_failing_silently() {
        use std::os::unix::fs::PermissionsExt;

        let root = temp_root("read-only-root");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 4).unwrap();
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &[1, 2, 3, 4], &[], 0, 0);

        // Revoke write permission on the root the manager already holds open.
        fs::set_permissions(&root, fs::Permissions::from_mode(0o500)).unwrap();

        let accepted = manager.enqueue(block(key)).unwrap();
        let durable = manager.drain(std::time::Duration::from_secs(10));

        let stats = manager.stats();
        // Every observation is taken above and every assertion below, so that
        // restoring write permission here cannot be unwound past by a failing
        // assertion — that would leave an undeletable directory behind.
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();

        assert!(
            accepted,
            "the queue accepts the write; only the disk refuses it"
        );
        assert!(
            !durable,
            "a drain covering a failed write must report the failure"
        );
        assert_eq!(
            stats.write_errors, 1,
            "a write the disk refused must be counted exactly once"
        );
        assert_eq!(
            stats.bytes_written, 0,
            "no bytes landed, so no bytes may be credited"
        );
        assert_eq!(stats.enqueued, 1, "the queue did accept it");
        assert_eq!(
            stats.queue_drops, 0,
            "the queue had room; this was not an admission refusal"
        );
        assert_eq!(stats.corruptions, 0, "nothing was written, nothing to read");
        assert!(
            !manager.contains(&key),
            "a write that never landed must not be indexed as present"
        );
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    /// The reported operator scenario, reproduced: the cache root is removed
    /// out from under a running process, and the next turn writes a sidecar.
    ///
    /// This is the test that settles what `bytes_written` means. It is
    /// credited only after `write_all` + payload sync + commit rename +
    /// directory fsync all return `Ok` (`persist_encoded`), so a root that no
    /// longer exists produces zero bytes written and one write error — not,
    /// as an enqueue-time estimate would, a healthy-looking byte total for
    /// data that never reached the disk.
    ///
    /// Uses a sidecar rather than a block so the sidecar arm of the writer
    /// loop is covered too; its `persist_sidecar` is a separate call site and
    /// would otherwise be counted only by inspection.
    #[cfg(unix)]
    #[test]
    fn a_deleted_root_counts_write_errors_and_credits_no_bytes() {
        let root = temp_root("deleted-root");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 4).unwrap();
        let key = ColdCacheKey::chain(
            ColdGroup::GdnState,
            fingerprint(),
            None,
            &[9, 9, 9, 9],
            &[],
            0,
            0,
        );

        fs::remove_dir_all(&root).unwrap();
        assert!(
            !root.exists(),
            "the root is gone before the write is offered"
        );

        assert!(
            manager
                .enqueue_sidecar(sidecar(key, ColdGroup::GdnState))
                .unwrap()
        );
        assert!(
            !manager.drain(std::time::Duration::from_secs(10)),
            "a drain covering a failed sidecar write must report the failure"
        );

        let stats = manager.stats();
        assert_eq!(stats.write_errors, 1, "the write had nowhere to land");
        assert_eq!(
            stats.bytes_written, 0,
            "`bytes_written` is landed bytes: a deleted root must credit none"
        );
        // `write_errors` is documented as a SUBSET of `enqueued`, and this row is
        // the only one where the subset is non-trivial: the object that failed is
        // a sidecar. Counting admissions per block kind would leave `enqueued` at
        // 0 beside a `write_errors` of 1 and break the invariant here.
        assert_eq!(
            stats.enqueued, 1,
            "a sidecar took a queue slot, so it is an admission"
        );
        assert!(
            stats.write_errors <= stats.enqueued,
            "`write_errors` must stay a subset of `enqueued` ({} > {})",
            stats.write_errors,
            stats.enqueued
        );
        assert_eq!(stats.queue_drops, 0, "the queue was never full");
        assert_eq!(stats.corruptions, 0);
        assert!(!root.exists(), "nothing may recreate the root behind us");
        drop(manager);
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

    /// A capture whose GPU read aborts must not enqueue anything and must
    /// still release the pin it took, otherwise the block is leaked as
    /// permanently referenced and a half-read block reaches disk as valid
    /// cold data for a later process to restore.
    ///
    /// Scope: the armed seam substitutes for reading the command buffer's
    /// status. A real device fault is NOT covered — see
    /// `crate::metal::command_buffer::arm_failure`.
    #[cfg(target_os = "macos")]
    #[test]
    fn capture_command_buffer_failure_enqueues_nothing_and_unpins() {
        use crate::PagedAttentionConfig;
        use crate::metal::MetalDtype;
        use crate::metal::command_buffer::arm_failure;

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
                eprintln!(
                    "skipping capture_command_buffer_failure_enqueues_nothing_and_unpins: {e}"
                );
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        let allocator = Mutex::new(BlockAllocator::new(2, 8));
        let source = allocator.lock().unwrap().allocate().unwrap();

        let root = temp_root("capture-cb-fail");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &tokens, &[], 0, 0);

        let refs_before = source.get_ref_count();
        let armed = arm_failure("LayerKVPool::read_block_all_layers");
        let captured = manager.capture_and_enqueue(&pool, &source, key, fingerprint(), &tokens);
        drop(armed);

        // Checked before the error is unwrapped, so a regression that lets the
        // capture succeed on an aborted read is caught here rather than
        // hidden behind an earlier `expect_err` panic.
        let stats = manager.stats();
        assert_eq!(
            stats.enqueued, 0,
            "a half-read block must never reach the writer queue"
        );
        assert_eq!(stats.queue_drops, 0, "the queue was never offered anything");
        assert_eq!(
            source.get_ref_count(),
            refs_before,
            "the capture pin must be released on the failure path too"
        );

        let err = captured.expect_err("an aborted GPU read must fail the capture");
        assert!(
            err.contains("LayerKVPool::read_block_all_layers"),
            "the capture must surface which submission failed: {err}"
        );
        assert!(
            manager.drain(Duration::from_secs(5)),
            "the writer queue must drain within the timeout"
        );
        assert!(
            !manager.contains(&key),
            "nothing must be persisted for a failed capture"
        );

        allocator.lock().unwrap().free(source);
        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    /// A restore whose GPU upload aborts must free the block it allocated,
    /// count exactly one miss, and never publish the prefix. Publishing would
    /// make a block the GPU may have written only partially reachable through
    /// the prefix cache, which decodes to wrong tokens with no error anywhere.
    ///
    /// Scope: the armed seam substitutes for reading the command buffer's
    /// status. A real device fault is NOT covered — see
    /// `crate::metal::command_buffer::arm_failure`. The sibling test
    /// `restore_post_allocate_upload_error_frees_block_and_counts_one_miss`
    /// drives the same arm through a real, unseamed `Err`.
    #[cfg(target_os = "macos")]
    #[test]
    fn restore_command_buffer_failure_frees_block_and_counts_one_miss() {
        use crate::metal::MetalDtype;
        use crate::metal::command_buffer::arm_failure;
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
                eprintln!(
                    "skipping restore_command_buffer_failure_frees_block_and_counts_one_miss: {e}"
                );
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

        let root = temp_root("restore-cb-fail");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &tokens, &[], 0, 0);
        assert!(
            manager
                .capture_and_enqueue(&pool, &source, key, fingerprint(), &tokens)
                .unwrap()
        );
        assert!(
            manager.drain(Duration::from_secs(5)),
            "the captured block must be durable before the restore"
        );
        allocator.lock().unwrap().free(source);

        let identity = RestorePrefixIdentity {
            hot_hash: hash_tokens(&tokens, 0, &[]),
            tokens: tokens.clone(),
            parent_hot_hash: 0,
            extra_keys: vec![],
            cache_salt: 0,
            block_index: 0,
        };
        let free_before = allocator.lock().unwrap().num_free_blocks();
        let misses_before = manager.stats().misses;

        let armed = arm_failure("LayerKVPool::write_block_all_layers");
        let restored = manager.restore_block(&pool, &allocator, key, fingerprint(), &identity);
        drop(armed);

        assert!(
            restored.is_none(),
            "an aborted upload must fall back to prefill"
        );
        let stats = manager.stats();
        assert_eq!(
            stats.misses - misses_before,
            1,
            "a failed upload is exactly one fall-back to prefill, counted once"
        );
        assert_eq!(stats.hits, 0, "no prefix was published, so no hit");
        assert_eq!(stats.bytes_restored, 0, "nothing landed in the pool");
        // Checked before the free-block count: "never published" is the
        // invariant the restore actually depends on, and a failure arm that
        // published would still be caught here even if the block were also
        // handed back to the free pool.
        let (hits, hit_tokens) =
            allocator
                .lock()
                .unwrap()
                .find_longest_cache_hit(&tokens, 8, &[], 0);
        assert_eq!(
            hit_tokens, 0,
            "a block whose upload aborted must never become reachable through the prefix cache"
        );
        assert!(hits.is_empty());

        assert_eq!(
            allocator.lock().unwrap().num_free_blocks(),
            free_before,
            "the block allocated for the restore must be freed again"
        );

        drop(manager);
        let _ = fs::remove_dir_all(root);
    }

    /// The same cleanup arm as the test above, reached by a REAL `Err` from
    /// `write_block_all_layers` with no seam involved: the allocator owns one
    /// more block than the pool, so the block handed to the restore is out of
    /// the pool's range and the upload is rejected AFTER the allocation.
    ///
    /// Every earlier guard (`layout_matches_pool`, the token comparison) runs
    /// before `allocate`, so this is the only way to reach the free-the-block
    /// branch without a test hook — which makes it the evidence that the
    /// branch is not seam-dependent.
    #[cfg(target_os = "macos")]
    #[test]
    fn restore_post_allocate_upload_error_frees_block_and_counts_one_miss() {
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
                eprintln!(
                    "skipping restore_post_allocate_upload_error_frees_block_and_counts_one_miss: {e}"
                );
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        // Three allocator blocks against a two-block pool: ids 0 and 1 are
        // valid pool blocks, id 2 is not.
        let allocator = Mutex::new(BlockAllocator::new(3, 8));
        let source = allocator.lock().unwrap().allocate().unwrap();
        let held = allocator.lock().unwrap().allocate().unwrap();
        assert_eq!(source.block_id, 0);
        assert_eq!(held.block_id, 1);

        let bytes_per_side = 64 * 8 * 2;
        let keys: Vec<u8> = (0..bytes_per_side).map(|i| (i % 251) as u8).collect();
        let values: Vec<u8> = (0..bytes_per_side)
            .map(|i| (250 - (i % 251)) as u8)
            .collect();
        pool.write_blocks_from_host(0, &[source.block_id], &keys, &values)
            .unwrap();

        let root = temp_root("restore-oob-upload");
        let manager = ColdCacheManager::open_at(root.clone(), GIB, 0, 2).unwrap();
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let key = ColdCacheKey::chain(ColdGroup::Kv, fingerprint(), None, &tokens, &[], 0, 0);
        assert!(
            manager
                .capture_and_enqueue(&pool, &source, key, fingerprint(), &tokens)
                .unwrap()
        );
        assert!(
            manager.drain(Duration::from_secs(5)),
            "the captured block must be durable before the restore"
        );

        // `source` and `held` stay alive, so the restore's `allocate` draws
        // block id 2 — past the end of the pool.
        let identity = RestorePrefixIdentity {
            hot_hash: hash_tokens(&tokens, 0, &[]),
            tokens: tokens.clone(),
            parent_hot_hash: 0,
            extra_keys: vec![],
            cache_salt: 0,
            block_index: 0,
        };
        let free_before = allocator.lock().unwrap().num_free_blocks();
        assert_eq!(free_before, 1, "only block 2 is left for the restore");
        let misses_before = manager.stats().misses;

        let restored = manager.restore_block(&pool, &allocator, key, fingerprint(), &identity);
        assert!(
            restored.is_none(),
            "an upload the pool rejects must fall back to prefill"
        );
        let stats = manager.stats();
        assert_eq!(
            stats.misses - misses_before,
            1,
            "a rejected upload is exactly one fall-back to prefill, counted once"
        );
        assert_eq!(stats.hits, 0, "no prefix was published, so no hit");
        assert_eq!(stats.bytes_restored, 0, "nothing landed in the pool");
        // Checked before the free-block count: "never published" is the
        // invariant the restore actually depends on, and a failure arm that
        // published would still be caught here even if the block were also
        // handed back to the free pool.
        let (hits, hit_tokens) =
            allocator
                .lock()
                .unwrap()
                .find_longest_cache_hit(&tokens, 8, &[], 0);
        assert_eq!(
            hit_tokens, 0,
            "a block whose upload failed must never become reachable through the prefix cache"
        );
        assert!(hits.is_empty());

        assert_eq!(
            allocator.lock().unwrap().num_free_blocks(),
            free_before,
            "the block allocated for the restore must be freed again"
        );

        {
            let mut allocator = allocator.lock().unwrap();
            allocator.free(source);
            allocator.free(held);
        }
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
