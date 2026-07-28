//! Process-global registry for the SSD cold tier
//! (`mlx_paged_attn::ColdCacheManager`) plus its NAPI stats surface.
//!
//! The tier is opened lazily on first use and is fail-open: if the root
//! cannot be opened (no HOME, unwritable disk, ...) inference proceeds
//! without persistence and `coldCacheStats()` reports `enabled: false`.

use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use mlx_paged_attn::{ColdCacheFingerprint, ColdCacheManager, ColdCacheStats};
use napi_derive::napi;
use sha2::{Digest, Sha256};

use crate::array::memory::WeightsResident;
use crate::models::qwen3_5::gdn_checkpoint_store::GDN_PREFIX_CHECKPOINT_LIMIT;

static GLOBAL: OnceLock<Option<Arc<ColdCacheManager>>> = OnceLock::new();

/// Overrides the cold-tier parent directory (primarily for tests). Read
/// once on first `global_cold_cache()` call; an empty value means the
/// default root (`~/.mlx-node/cache/paged/v1`).
const COLD_CACHE_DIR_ENV: &str = "MLX_COLD_CACHE_DIR";

/// Child of `MLX_COLD_CACHE_DIR` the tier actually operates in. Opening a
/// root chmods it 0700 and deletes leftover writer temp files, so those
/// behaviors must only ever touch a directory the cache created itself,
/// never the user-supplied directory verbatim.
const MANAGED_SUBDIR: &str = "mlx-paged-v1";

/// How many blocks one turn's cold-tier capture walk may add to the persisted
/// chain, and how long that walk may take. Both read once, on first use.
const CAPTURE_BLOCKS_ENV: &str = "MLX_COLD_CAPTURE_BLOCKS_PER_TURN";
const CAPTURE_BUDGET_MS_ENV: &str = "MLX_COLD_CAPTURE_BUDGET_MS";

/// Default per-turn capture depth, in blocks.
///
/// The walk used to have no policy at all: it stopped at the first block the
/// bounded writer queue refused, which made the per-turn rate an emergent
/// function of the filesystem — `(Q + 1) / (1 - Tc/Tw)`, ~12 blocks on this
/// machine, 9 under `F_FULLFSYNC`, and UNBOUNDED on a RAM disk. At ~12 blocks
/// (192 tokens) an 8 K-token prompt needed ~40 turns to persist, which is why
/// the restored prefix measured 1.4-6.2% of the prompt.
///
/// 128 blocks is chosen so a typical agent prompt is fully covered within a few
/// turns while the tail stays inside the deadline below at the measured `Tw`
/// (~0.43-1.27 ms/object): 128 x 1.27 ms = 163 ms worst case, 55 ms typical.
const DEFAULT_CAPTURE_BLOCKS_PER_TURN: usize = 128;

/// Wall-clock ceiling on one capture walk. Bounds the turn tail even when the
/// storage device stalls, and is what makes the walk safe on a filesystem fast
/// enough that the queue never refuses (where the old walk covered the whole
/// prompt in one turn: ~0.9 s of tail at 64 K tokens).
const DEFAULT_CAPTURE_BUDGET_MS: u64 = 250;

/// Per-turn budget for `ColdTierWalk::capture_chain`, read once from the
/// environment.
///
/// Deliberately a policy the walk carries rather than a property of the writer
/// queue: the queue's depth is a host-MEMORY bound (blocks in flight), while
/// this is a TURN-TAIL bound (how much of the prompt one turn is willing to pay
/// to persist). Conflating them is what made the old rate move with `Tw`.
pub fn cold_capture_budget() -> (usize, std::time::Duration) {
    static BUDGET: OnceLock<(usize, std::time::Duration)> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let blocks = std::env::var(CAPTURE_BLOCKS_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_CAPTURE_BLOCKS_PER_TURN);
        let millis = std::env::var(CAPTURE_BUDGET_MS_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_CAPTURE_BUDGET_MS);
        (blocks, std::time::Duration::from_millis(millis))
    })
}

/// Lazily open the process-wide cold tier. Returns `None` when the tier
/// cannot be opened (fail-open: inference proceeds without persistence).
pub fn global_cold_cache() -> Option<Arc<ColdCacheManager>> {
    GLOBAL
        .get_or_init(|| match std::env::var(COLD_CACHE_DIR_ENV) {
            Ok(dir) if !dir.is_empty() => open_managed_cold_cache(Path::new(&dir)),
            _ => ColdCacheManager::open_default().ok().map(Arc::new),
        })
        .clone()
}

/// Open the tier in the `mlx-paged-v1` child of `parent`, creating the
/// child when absent. Validation lives in the cold-cache root opener: on
/// unix the child is opened descriptor-relative with `O_NOFOLLOW` (and the
/// manager keeps that dirfd for all later I/O), elsewhere a static pre-open
/// check applies; either way a pre-existing child that is a symlink or not
/// a directory is refused (`None`, fail-open) so the tier's chmod/temp
/// cleanup/eviction can never follow a planted link into a foreign
/// directory.
fn open_managed_cold_cache(parent: &Path) -> Option<Arc<ColdCacheManager>> {
    ColdCacheManager::open_default_at(parent.join(MANAGED_SUBDIR))
        .ok()
        .map(Arc::new)
}

/// Counter snapshot of the global tier for Rust-side consumers (per-turn
/// trace deltas). `None` while the tier is uninitialized or disabled;
/// never forces the tier open.
pub fn cold_cache_stats_snapshot() -> Option<ColdCacheStats> {
    GLOBAL.get()?.as_ref().map(|manager| manager.stats())
}

/// Sidecar-write accounting for the recurrent/sliding state that lives
/// OUTSIDE the paged K/V pool.
///
/// `ColdCacheStats` counts block writes and cannot distinguish them from
/// sidecar writes, and a turn that returns before `enqueue_sidecar` is
/// invisible there entirely. Without these a family whose sidecar is never
/// written looks exactly like a family that needs no sidecar: both report clean
/// block traffic and a restore that quietly reuses nothing.
///
/// These belong in `ColdCacheStats` alongside the block counters the dashboard
/// and the parity harness already read — a manager is in hand at every record
/// site, so nothing here needs to outlive it. They live in this crate only
/// because `ColdCacheStats` is defined in `mlx-paged-attn`, which this change
/// may not touch. The cost of that split is real and unfixed here: the counts
/// are per-process, so a host running several sessions cannot attribute a skip
/// to one of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColdSidecarTelemetry {
    /// Turns that reached a family's sidecar capture at all. Every other counter
    /// here is a sub-count of this one, so it is what separates "the capture ran
    /// and declined" from "the turn's finalize path never calls the capture" —
    /// the exact confusion a planned-MTP turn produced while its manual finalize
    /// had no capture hook.
    pub capture_reached: u64,
    /// Turns whose persisted K/V chain covered no whole block of the request,
    /// so there was no prefix to anchor recurrent state under at all.
    pub chain_empty: u64,
    /// Turns whose chain did cover blocks, but where no retained in-memory
    /// checkpoint sat at or below its reach.
    pub boundary_skips: u64,
    /// Turns that selected a boundary but found that exact chain already on
    /// disk, so nothing was written and nothing needed to be.
    ///
    /// Its own counter rather than silence, because silence made a HEALTHY run
    /// wear the signature of a broken one. The steady state of a repeated
    /// prompt is: the first turn that can reach a rung enqueues it, and every
    /// later turn re-selects the SAME rung and dedups here. A window that
    /// starts after that first turn therefore sees `enqueued == 0` while the
    /// per-turn shallow boundaries it could not reach still count as
    /// `boundary_skips`, which is bit-for-bit the "the ladder collapsed to its
    /// deepest rung" signature the parity harness triages on. Only this
    /// counter tells the two apart.
    pub already_persisted: u64,
    /// Sidecars handed to the bounded writer queue.
    pub enqueued: u64,
    /// Sidecars the bounded writer queue refused because it was full.
    pub queue_drops: u64,
    /// Restored sidecars a family actually INSTALLED as its live per-turn state.
    ///
    /// The only counter on the read side, and it exists because every other
    /// signal a restore emits is satisfiable with the sidecar thrown away.
    /// `install_*_gdn_cold_sidecar` has eight `Ok(false)` arms and each one falls
    /// through to a full O(prefix) recurrent replay that produces CORRECT state
    /// — so `cached_tokens`, `hits`, `corruptions`, and text/`num_tokens` parity
    /// all stay exactly as they were. Without this counter a regression from
    /// "restored and used" to "restored and re-derived from scratch" is
    /// invisible, and that regression is the whole feature.
    pub installed: u64,
    /// Restores a family THREW AWAY after the walk had already served them.
    ///
    /// The second read-side counter, and a different event from
    /// `ColdCacheStats::restore_declines`: there the walk refused to hand
    /// anything back, here it handed something back and the family released it
    /// and restarted the turn cold (gemma4's large-sliding suppression, which
    /// does a `release_request` + `reset_for_new_request` + forced-miss
    /// re-lookup). Both end at "the prefix was recomputed", so they are
    /// indistinguishable downstream — and only this one is preceded by real
    /// `hits` and `bytes_restored`, which is what makes it look like reuse
    /// worked. Split from the walk's counter so a suppression that starts
    /// firing on every turn cannot hide inside a decline count.
    pub restore_suppressed: u64,
}

/// The counters behind [`ColdSidecarTelemetry`].
///
/// A type rather than loose statics so a test can exercise a fresh, isolated
/// instance and assert exact deltas — the only assertion that catches a
/// recorder wired to the wrong field.
#[derive(Debug, Default)]
pub struct ColdSidecarCounters {
    capture_reached: AtomicU64,
    chain_empty: AtomicU64,
    boundary_skips: AtomicU64,
    already_persisted: AtomicU64,
    enqueued: AtomicU64,
    queue_drops: AtomicU64,
    installed: AtomicU64,
    restore_suppressed: AtomicU64,
}

impl ColdSidecarCounters {
    const fn new() -> Self {
        Self {
            capture_reached: AtomicU64::new(0),
            chain_empty: AtomicU64::new(0),
            boundary_skips: AtomicU64::new(0),
            already_persisted: AtomicU64::new(0),
            enqueued: AtomicU64::new(0),
            queue_drops: AtomicU64::new(0),
            installed: AtomicU64::new(0),
            restore_suppressed: AtomicU64::new(0),
        }
    }

    /// Record a turn that entered a family's sidecar capture. Recorded first
    /// thing in the capture, before any of its skip arms.
    pub fn record_capture_reached(&self) {
        self.capture_reached.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a turn whose persisted K/V chain covered no whole block.
    pub fn record_chain_empty(&self) {
        self.chain_empty.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a turn whose sidecar capture found no usable boundary under the
    /// chain's reach.
    pub fn record_boundary_skip(&self) {
        self.boundary_skips.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a turn whose selected chain was already on disk.
    pub fn record_already_persisted(&self) {
        self.already_persisted.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a sidecar the writer queue accepted.
    pub fn record_enqueued(&self) {
        self.enqueued.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a sidecar the writer queue refused because it was full.
    pub fn record_queue_drop(&self) {
        self.queue_drops.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a restored sidecar a family installed as its live per-turn state.
    ///
    /// Recorded at the point of no return — after the decoded caches have
    /// replaced `self.caches`, so no later `Ok(false)` arm can undo it and a
    /// decode failure (which returns `Err` and fails the turn) never reaches it.
    pub fn record_install(&self) {
        self.installed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a restored prefix a family discarded rather than reuse.
    ///
    /// Recorded at the point the family commits to the discard — after the
    /// decision to suppress is final and before the adapter is reset — so a
    /// turn that merely evaluated the suppression rule and kept its prefix
    /// never lands here.
    pub fn record_restore_suppressed(&self) {
        self.restore_suppressed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> ColdSidecarTelemetry {
        let load = |value: &AtomicU64| value.load(Ordering::Relaxed);
        ColdSidecarTelemetry {
            capture_reached: load(&self.capture_reached),
            chain_empty: load(&self.chain_empty),
            boundary_skips: load(&self.boundary_skips),
            already_persisted: load(&self.already_persisted),
            enqueued: load(&self.enqueued),
            queue_drops: load(&self.queue_drops),
            installed: load(&self.installed),
            restore_suppressed: load(&self.restore_suppressed),
        }
    }
}

static SIDECAR_COUNTERS: ColdSidecarCounters = ColdSidecarCounters::new();

/// The counters the capture paths record into.
pub fn cold_sidecar_counters() -> &'static ColdSidecarCounters {
    &SIDECAR_COUNTERS
}

/// Process-wide sidecar counters. Unlike `cold_cache_stats_snapshot` these read
/// without forcing the tier open.
pub fn cold_sidecar_telemetry() -> ColdSidecarTelemetry {
    SIDECAR_COUNTERS.snapshot()
}

/// Serializes tests that assert on a DELTA of the process-wide counters.
///
/// `SIDECAR_COUNTERS` is one static shared by every test thread, and cargo runs
/// tests in parallel. Two tests that each assert "my call moved this counter by
/// exactly one" can both be correct and still fail, or worse pass while a bug
/// hides in the other's increment. Every test that reads a global delta holds
/// this for the whole read-act-read window.
#[cfg(test)]
pub(crate) fn sidecar_counter_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Model families whose paged prefix blocks can be restored from the SSD
/// cold tier soundly. The single source of truth on the native side, and the
/// exact mirror of `COLD_TIER_RESTORE_FAMILIES` in
/// `packages/agent/src/provider/model-host.ts`; the two gate the same
/// decision from opposite ends, so a test asserts they never drift.
///
/// A family belongs here only when EVERY piece of per-token state its forward
/// pass carries between turns is reconstructible from the tier — either because
/// it all lives inside the paged pool, or because the part that does not is
/// persisted alongside as a `ColdSidecar` and the restore reconciles down to a
/// boundary that sidecar actually backs:
///
///  * `qwen3` (dense) sizes its pool over all layers, so the pool holds the
///    complete KV for the prefix and needs no sidecar
///    (`ColdTierContext::sidecar_policy` is `None`).
///  * `gemma4` sizes its pool over the full-attention layers only, but persists
///    its out-of-pool sliding-window `RotatingKVCache` state as a
///    `ColdGroup::SlidingWindow` sidecar
///    (`crate::models::gemma4::sliding_sidecar`). Its `ColdSidecarPolicy` makes
///    the restore walk refuse any boundary a validated sidecar does not back.
///  * `qwen3_5` (dense) sizes its pool over the full-attention layers only, but
///    persists its out-of-pool GDN recurrent state (conv + recurrent) as a
///    `ColdGroup::GdnState` sidecar (`crate::models::qwen3_5::gdn_sidecar`). A
///    GDN recurrent state is a running summary of every preceding token, so it
///    is valid ONLY at its exact block-aligned prefix length (vLLM `MambaSpec`);
///    its `ColdSidecarPolicy` reconciles the restore down to the deepest such
///    boundary a validated sidecar backs, or to zero.
///  * `qwen3_5_moe` keeps the SAME GDN recurrent state outside the pool — same
///    shapes, same dtype, same layer mapping, projected by
///    `Qwen3_5MoeConfig::to_dense_config` — so it shares the dense family's
///    sidecar codec and `ColdSidecarPolicy` verbatim. Witnessed on
///    `Qwen3.6-35b-a3b-UD-Q2_K_XL-mlx`: the restore reconciled onto rung 304 of
///    the ladder `[16, 64, 304, 1248]` with `hits=42`, `corruptions=0`, and text
///    matching a no-persist baseline
///    (`crates/mlx-core/tests/qwen3_5_moe_cold_tier_parity.rs`).
///  * `lfm2` / `lfm2_moe` keep short-conv state outside the pool with no
///    serialization path for it at all, AND drive the uniform adapter API
///    whose restore branch is already wired to the tier — attaching a context
///    to them would restore an incomplete prefix silently.
///
/// Widening this list is a correctness decision, never a perf one, and it is
/// authorized by exactly one thing: the family's restart-parity gate
/// (`crates/mlx-core/tests/cold_tier_parity_harness.rs`) passing on real
/// weights with `hits > 0` and `corruptions == 0`.
const COLD_RESTORE_FAMILIES: &[&str] = &["gemma4", "qwen3", "qwen3_5", "qwen3_5_moe"];

/// Whether the cold tier may restore persisted paged blocks for `model_type`
/// (see [`COLD_RESTORE_FAMILIES`]). Fails closed: an unknown or misspelled
/// family is unsupported, so the tier stays off and the prefix is recomputed.
pub fn cold_restore_supported(model_type: &str) -> bool {
    COLD_RESTORE_FAMILIES.contains(&model_type)
}

/// The native cold-restore allowlist, exposed so a test can assert it agrees
/// exactly with the TypeScript `COLD_TIER_RESTORE_FAMILIES` set.
#[napi]
pub fn cold_restore_families() -> Vec<String> {
    COLD_RESTORE_FAMILIES
        .iter()
        .map(|family| (*family).to_string())
        .collect()
}

/// How many GDN prefix checkpoints the native store holds across all owners
/// ([`GDN_PREFIX_CHECKPOINT_LIMIT`]), exposed for the same reason
/// [`cold_restore_families`] is: it is one half of a cross-language invariant
/// and nothing else carries it over the boundary.
///
/// The other half is `MAX_CONCURRENCY` in
/// `packages/agent/src/extensions/subagent.ts`. The store's demand is
/// `MAX_CONCURRENCY + 1` — one owner per concurrent child loop plus the root
/// session — and the cliff sits exactly one owner past the cap, where every
/// owner holds a single entry and the store is still over it, so each publish
/// takes somebody's last checkpoint. `retention_sim` measures that as 0 blind
/// turns at five owners and 28 of 40 at six.
///
/// No Rust gate can see a TypeScript-only edit, so raising the fleet alone
/// would land in that regime with every Rust gate still green.
/// `packages/agent/__test__/gdn-checkpoint-capacity.test.ts` is what stops it.
#[napi]
pub fn gdn_prefix_checkpoint_limit() -> u32 {
    GDN_PREFIX_CHECKPOINT_LIMIT as u32
}

// ---------------------------------------------------------------------------
// Load-time guard shared by every family that attaches a cold tier
// ---------------------------------------------------------------------------
//
// The fingerprint above describes the bytes it READ; these helpers are what
// let a loader prove those are the same bytes its mmap actually took. They
// live here rather than in one family's persistence module because every
// family attaching a `ColdTierContext` owes the identical bracket, and two
// copies of a fail-safe check drift into one copy of a fail-open one.

/// Filesystem identity of one weight shard, used to detect a mid-load
/// model-directory swap. `dev`/`ino` change when a separate process
/// reinstalls a shard as a new file (the common reinstall pattern), and
/// `len`/`mtime` catch an in-place rewrite. Equality across a snapshot pair
/// means the mmap'd inode is still the file the cold-tier fingerprint will
/// read.
#[derive(PartialEq)]
pub(crate) struct ShardIdentity {
    len: u64,
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
    #[cfg(unix)]
    mtime: i64,
    #[cfg(unix)]
    mtime_nsec: i64,
    #[cfg(not(unix))]
    mtime: Option<std::time::SystemTime>,
}

impl ShardIdentity {
    fn from_metadata(meta: &fs::Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            ShardIdentity {
                len: meta.len(),
                dev: meta.dev(),
                ino: meta.ino(),
                mtime: meta.mtime(),
                mtime_nsec: meta.mtime_nsec(),
            }
        }
        #[cfg(not(unix))]
        {
            ShardIdentity {
                len: meta.len(),
                mtime: meta.modified().ok(),
            }
        }
    }
}

/// Snapshot the on-disk identity of every top-level `.safetensors` shard in
/// `dir` — the SAME shard set the cold-tier fingerprint
/// (`cold_tier::build_model_fingerprint` → `shard_file_names`) covers.
///
/// Returns `None` if the directory or any shard's metadata cannot be read;
/// the caller treats `None` as "changed" and leaves cold persistence off
/// (fail-safe). Keyed by file name so an added/removed shard between two
/// snapshots also compares unequal.
pub(crate) fn snapshot_shard_identities(dir: &Path) -> Option<HashMap<String, ShardIdentity>> {
    let mut identities = HashMap::new();
    for entry in fs::read_dir(dir).ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".safetensors") {
            continue;
        }
        // `metadata` follows symlinks, matching the fingerprint's shard reads.
        let meta = fs::metadata(entry.path()).ok()?;
        identities.insert(name, ShardIdentity::from_metadata(&meta));
    }
    Some(identities)
}

/// True only if the shard identity is provably unchanged across all three
/// bracketing snapshots: before the mmap, right after the mmap, and after the
/// cold-tier fingerprint has read the shards. Any `None` (a shard/directory
/// that could not be stat'd at some checkpoint) or any difference means the
/// mmap'd bytes might not be the bytes the fingerprint described, so the caller
/// must leave cold persistence off (fail-safe).
///
/// Requiring equality across the full bracket rejects the atomic-rename swap
/// (a reinstall changes dev/ino, and an in-place rewrite changes len/mtime, so
/// at least one snapshot differs). It does NOT prove *continuous* identity: a
/// coordinated local ABA swap — shard A present for the before/at snapshots,
/// replaced by B only while the fingerprint reads it, then A restored before
/// the final snapshot — makes all three snapshots compare equal yet binds A's
/// stat identity to B's fingerprinted bytes. That residual is accepted and out
/// of scope; closing it soundly would require the fingerprint to hash the same
/// held fd that backs the loaded arrays (an fd-coupled loader restructure), not
/// point-in-time stat snapshots.
///
/// The bracket also covers materialization, not just the mmap. MLX reads shard
/// bytes lazily through a held `O_RDONLY` fd, so a same-inode in-place rewrite
/// after the identity read used to bind persisted KV to bytes the model never
/// ran. Every loader now materializes its weights BEFORE building the
/// fingerprint — enforced by the `WeightsResident` witness
/// `build_model_fingerprint` demands — so the third snapshot lands after the
/// bulk `pread`, and a rewrite anywhere in `[before-mmap .. after-fingerprint]`
/// changes a shard's len or mtime and fails this check closed. The residuals
/// that remain are the coordinated ABA swap above and the mtime-preserving
/// same-size swap; both would need the same fd-coupled fingerprint to close and
/// stay out of scope.
pub(crate) fn shard_identities_stable(
    before_mmap: &Option<HashMap<String, ShardIdentity>>,
    at_mmap: &Option<HashMap<String, ShardIdentity>>,
    after_fingerprint: &Option<HashMap<String, ShardIdentity>>,
) -> bool {
    match (before_mmap, at_mmap, after_fingerprint) {
        (Some(before), Some(at), Some(after)) => before == at && at == after,
        _ => false,
    }
}

/// Lenient boolean parse for the `MLX_PERSIST_PAGED_CACHE` override.
/// `1/true/on/yes` → `Some(true)`, `0/false/off/no` → `Some(false)`
/// (case-insensitive, surrounding whitespace ignored); anything else →
/// `None` so the caller falls through to the config alias.
pub(crate) fn parse_bool_env(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

/// Resolve whether cold KV persistence is on, with precedence
/// **explicit config > env > off**. An *explicit* per-model config decision
/// (`persist_paged_cache` / `persistPagedCache` present as a bool, i.e.
/// `Some`) is authoritative and wins over the environment — this is how an
/// explicit CLI decision reaches the loader: `mlx agent --no-persist-cache`
/// writes `persist_paged_cache: false` into the model's config overlay, and
/// that per-invocation intent must beat an ambient `MLX_PERSIST_PAGED_CACHE`
/// default. The `MLX_PERSIST_PAGED_CACHE` environment variable only supplies a
/// default when the config is unset (`None`), so a direct library caller —
/// which has no config plumbing — can still enable or disable persistence,
/// matching the Phase-A "per-model config + env override, off by default at
/// library level" contract. With neither an explicit config nor a recognized
/// env value, persistence is off. `env_value` is passed in (rather than read
/// here) so the precedence logic is unit-testable without mutating
/// process-global env.
pub(crate) fn resolve_persist_cold(
    model_type: &str,
    env_value: Option<&str>,
    config_flag: Option<bool>,
) -> bool {
    // The family allowlist is the single source of truth for whether the cold
    // tier may engage, enforced HERE at the native choke point rather than only
    // in the agent. A family that is off the list never persists or restores a
    // cold tier — not via an explicit `persist_paged_cache` config, not via
    // `MLX_PERSIST_PAGED_CACHE`, not via any direct library caller that bypasses
    // the agent entirely. A family's loader may wire a full cold bracket ahead
    // of proving it; this gate keeps that bracket dormant until the family is
    // admitted, so an unproven path can never be exercised in the field.
    if !cold_restore_supported(model_type) {
        return false;
    }
    if let Some(explicit) = config_flag {
        return explicit;
    }
    env_value.and_then(parse_bool_env).unwrap_or(false)
}

/// Filename the dashboard downloader writes as the last step of an atomic
/// publish (see `packages/dashboard/src/download.ts`). Carries the pinned
/// HuggingFace commit `revision` — an immutable content identity for the
/// whole snapshot — plus the list of `files` the snapshot contains.
const DOWNLOAD_COMPLETE_MARKER: &str = ".mlx-download-complete.json";

/// Read-buffer size for the streaming full-shard digest. Bounds peak memory
/// during the O(weight-bytes) fallback hash regardless of shard size.
const FULL_DIGEST_CHUNK: usize = 1024 * 1024;

/// Cache geometry whose drift must invalidate persisted blocks.
pub(crate) struct ColdTierGeometry {
    pub block_size: u64,
    pub num_layers: u64,
    pub num_kv_heads: u64,
    pub head_size: u64,
    pub cache_dtype: String,
}

/// Build the cold-tier fingerprint for a model directory, binding it to the
/// COMPLETE weight identity of every shard rather than a sampled slice.
///
/// Two same-architecture finetunes of the same base normally have identical
/// shard filenames AND identical byte sizes (same tensor shapes), so a
/// name+size fingerprint would collide and let one model restore the other's
/// persisted KV for a shared token prefix — silent output corruption. A
/// windowed sample of each shard is not enough either: two checkpoints that
/// differ only in an UNSAMPLED tensor between the windows would still collide.
/// The identity must therefore be complete. Two paths deliver that, both
/// sound and both deterministic:
///
///  * **Manifest path (cheap, preferred).** When the dashboard download marker
///    is present, pins an immutable commit `revision`, and its `files` list
///    covers every on-disk shard, that repo+revision cryptographically
///    identifies the exact bytes of the whole snapshot — so the weight bytes
///    need not be read at all. This is the common case (catalog models
///    installed via the dashboard downloader). Each shard's on-disk size AND
///    modification time (nanoseconds) are folded in from the metadata already
///    read, so any out-of-band write that changes a shard's size OR mtime
///    yields a different key (safe miss) instead of silently reusing another
///    revision's persisted KV. The only residual evasion — a deliberate
///    mtime-preserving, same-size in-place swap — is out of scope (the sound
///    alternative, a full per-load digest, is a deliberate cold-cache perf
///    non-goal here).
///  * **Full-digest fallback (sound, O(weight bytes)).** Otherwise stream a
///    SHA-256 over each shard's entire bytes. This runs once per model load,
///    only when persistence is enabled and no trusted manifest covers the
///    shards, so a manually-placed checkpoint without a marker pays a one-time
///    full hash at load. Correct KV identity outweighs that one-time cost.
///
/// The weight_map index and geometry are folded in additively either way.
///
/// Returns `None` when the directory holds no shard, or when a shard on the
/// full-digest path cannot be read to completion: the caller then leaves
/// persistence OFF (fail-safe) rather than persisting under a partial identity
/// that could collide with another model.
///
/// **Ordering contract (lazy materialization).** MLX reads shard bytes lazily:
/// the loader holds one `O_RDONLY` fd per shard and the bulk bytes arrive at
/// `array::memory::materialize_weights`. An identity read before that pass
/// describes bytes a same-inode in-place rewrite can still replace, and — unlike
/// the mtime-preserving swap guarded above — such a rewrite need not preserve
/// mtime, because no identity snapshot would follow it. Nothing corrects a wrong
/// identity later: `cold_cache` compares a stored object's fingerprint against
/// the ATTACH-TIME value, never a re-derived one.
///
/// So this function requires a [`WeightsResident`] witness, which only
/// `materialize_weights` can mint. Every loader must materialize first, and the
/// compiler enforces it. A witness that materialized nothing is refused, so an
/// empty `materialize_weights(&[])` cannot launder the ordering.
pub(crate) fn build_model_fingerprint(
    model_type: &str,
    model_path: &str,
    config_json: Option<&[u8]>,
    geometry: &ColdTierGeometry,
    weights: &WeightsResident,
) -> Option<ColdCacheFingerprint> {
    // A zero-coverage witness proves nothing about the shards this identity is
    // about to read, so fail safe rather than bind KV to unread weights.
    if weights.bytes() == 0 {
        tracing::warn!(
            "cold-tier persistence disabled for {model_path}: the weight-materialization \
             witness covers {} arrays / {} bytes, so the shard bytes this identity reads \
             are not provably the bytes the model runs",
            weights.arrays(),
            weights.bytes(),
        );
        return None;
    }

    let dir = Path::new(model_path);

    // Deterministic (sorted) shard enumeration. A missing/unreadable directory
    // or an empty shard set means there is no weight content to bind to.
    let shard_names = shard_file_names(dir)?;
    if shard_names.is_empty() {
        return None;
    }

    let mut components: Vec<Vec<u8>> = vec![model_type.as_bytes().to_vec()];
    if let Some(cfg) = config_json {
        components.push(cfg.to_vec());
    }

    let manifest = read_download_manifest(dir);
    let trusted = manifest
        .as_ref()
        .filter(|m| m.covers_all_shards(&shard_names));

    match trusted {
        // Cheap manifest identity: the immutable repo+revision pins the exact
        // snapshot bytes, so shard bytes are NOT read. Names, sizes, and
        // modification times come from the SAME metadata (no data read) and
        // guard against an on-disk shard swap: any out-of-band write that
        // changes a shard's size OR mtime yields a different key (safe miss),
        // so a same-size in-place rewrite can no longer silently reuse another
        // revision's persisted KV. A shard whose mtime cannot be read disables
        // the cheap path for this load (`?` -> None, fail-safe) rather than
        // trusting it.
        Some(m) => {
            components.push(b"cold-identity:manifest:v1\0".to_vec());
            for name in &shard_names {
                let meta = std::fs::metadata(dir.join(name)).ok()?;
                let size = meta.len();
                let mtime_nanos = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_nanos())?;
                components.push(name.as_bytes().to_vec());
                components.push(size.to_le_bytes().to_vec());
                components.push(mtime_nanos.to_le_bytes().to_vec());
            }
            components.push(m.identity_bytes());
        }
        // Sound fallback: a full streaming digest of every shard's bytes. A
        // single shard that cannot be read to completion disables persistence.
        None => {
            components.push(b"cold-identity:full-digest:v1\0".to_vec());
            for name in &shard_names {
                let (size, digest) = full_shard_digest(&dir.join(name))?;
                components.push(name.as_bytes().to_vec());
                components.push(size.to_le_bytes().to_vec());
                components.push(digest.to_vec());
            }
            // A present-but-incomplete marker still contributes its revision
            // additively; a stronger identity than the bytes alone never hurts.
            if let Some(m) = &manifest {
                components.push(b"download-marker\0".to_vec());
                components.push(m.identity_bytes());
            }
        }
    }

    // Additive: the weight_map index encodes the tensor->shard layout. Read
    // without touching weight bytes; absent for single-shard checkpoints.
    if let Ok(index_bytes) = std::fs::read(dir.join("model.safetensors.index.json")) {
        components.push(b"index.json\0".to_vec());
        components.push(index_bytes);
    }

    components.push(geometry.block_size.to_le_bytes().to_vec());
    components.push(geometry.num_layers.to_le_bytes().to_vec());
    components.push(geometry.num_kv_heads.to_le_bytes().to_vec());
    components.push(geometry.head_size.to_le_bytes().to_vec());
    components.push(geometry.cache_dtype.clone().into_bytes());

    Some(ColdCacheFingerprint::from_components(
        components.iter().map(|c| c.as_slice()),
    ))
}

/// Sorted file names of every top-level `.safetensors` shard in `dir`.
/// `None` only if the directory itself cannot be read.
fn shard_file_names(dir: &Path) -> Option<Vec<String>> {
    let read_dir = std::fs::read_dir(dir).ok()?;
    let mut names: Vec<String> = read_dir
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .filter(|name| name.ends_with(".safetensors"))
        .collect();
    names.sort();
    Some(names)
}

/// Streaming SHA-256 over a shard's ENTIRE bytes (domain-separated, size
/// prefixed), read in bounded chunks so peak memory is independent of shard
/// size. `File::open` follows symlinks so symlinked checkpoints digest the
/// real weight bytes. `None` on any hard I/O error — a shard whose complete
/// identity cannot be established must disable persistence, not weaken it.
fn full_shard_digest(path: &Path) -> Option<(u64, [u8; 32])> {
    let mut file = File::open(path).ok()?;
    let size = file.metadata().ok()?.len();

    let mut hasher = Sha256::new();
    hasher.update(b"mlx-node:cold-shard-full:v1\0");
    hasher.update(size.to_le_bytes());

    let mut buf = vec![0u8; FULL_DIGEST_CHUNK];
    loop {
        match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => hasher.update(&buf[..n]),
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return None,
        }
    }
    Some((size, hasher.finalize().into()))
}

/// The dashboard download manifest's trusted fields: the pinned commit
/// `revision`, the source `repo`, and the `files` the snapshot contains.
struct DownloadManifest {
    repo: Option<String>,
    revision: String,
    files: Vec<String>,
}

impl DownloadManifest {
    /// True when the manifest lists every on-disk shard, so its immutable
    /// repo+revision fully pins those exact bytes. Coverage is decided by raw
    /// string equality: on-disk shards are top-level basenames, so only a
    /// byte-for-byte equal manifest entry names the same repo-relative file. A
    /// directory prefix, absolute path, or trailing slash/dot is a DIFFERENT or
    /// malformed entry and never covers a bare basename.
    fn covers_all_shards(&self, shard_names: &[String]) -> bool {
        shard_names.iter().all(|shard| {
            // Raw equality, NOT path normalization: `Path::components()` would
            // fold `x/`, `x/.`, `x//` back to `x` and wrongly report coverage.
            self.files.iter().any(|file| file == shard)
        })
    }

    /// Immutable identity bytes: `repo\0revision`. Excludes the manifest's
    /// `completedAt` timestamp so re-downloading the SAME snapshot does not
    /// shift the fingerprint.
    fn identity_bytes(&self) -> Vec<u8> {
        let mut identity = Vec::new();
        if let Some(repo) = &self.repo {
            identity.extend_from_slice(repo.as_bytes());
        }
        identity.push(0);
        identity.extend_from_slice(self.revision.as_bytes());
        identity
    }
}

/// Parse the download marker. `None` when the marker is absent, unparseable,
/// or missing the required `revision` — it is trusted provenance only when
/// fully formed; otherwise the full-digest fallback establishes identity.
fn read_download_manifest(dir: &Path) -> Option<DownloadManifest> {
    let bytes = std::fs::read(dir.join(DOWNLOAD_COMPLETE_MARKER)).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let revision = value.get("revision")?.as_str()?.to_string();
    // A cold-tier identity is only trustworthy if the revision is an immutable
    // commit sha; a mutable ref (e.g. "main") could point at different weights
    // across downloads. Reject anything but 40 hex chars -> full digest fallback.
    if revision.len() != 40 || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let repo = value
        .get("repo")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let files = value
        .get("files")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    Some(DownloadManifest {
        repo,
        revision,
        files,
    })
}

/// Snapshot of the process-wide SSD cold tier for paged prefix blocks.
/// Counters are cumulative since the tier was opened; all numeric values
/// are returned as `f64` to avoid BigInt round-trips in JS.
#[napi(object, js_name = "ColdCacheStats")]
#[derive(Clone, Debug, Default)]
pub struct ColdCacheStatsJs {
    /// `false` until the tier is first opened by inference, or when opening
    /// failed (fail-open: inference then runs without persistence).
    pub enabled: bool,
    /// Cache root directory (empty while disabled).
    pub root: String,
    /// Disk quota in bytes.
    pub quota_bytes: f64,
    /// Blocks restored from disk after validation.
    pub hits: f64,
    /// Lookups that found no usable block (includes corrupt entries).
    pub misses: f64,
    /// Objects accepted onto the background write queue — K/V blocks and
    /// family state sidecars alike, since both take a slot in the same queue.
    /// `coldSidecarEnqueued` counts a subset of this; never sum them.
    pub enqueued: f64,
    /// Writes REFUSED at admission because the bounded queue was full. Same
    /// object scope as `enqueued`. Disjoint from `writeErrors`, which counts
    /// accepted writes that then failed to land — never sum the two into one
    /// "lost writes" number.
    pub queue_drops: f64,
    /// Bytes that LANDED, credited after the payload sync, the commit rename
    /// and the directory fsync all succeeded. Not an enqueue-time estimate:
    /// a failed write credits nothing here and one `writeErrors` instead.
    pub bytes_written: f64,
    /// Total bytes read back on validated hits.
    pub bytes_restored: f64,
    /// Entries evicted to respect the quota / free-space reserve.
    pub evictions: f64,
    /// Entries that failed checksum/identity validation and were removed.
    pub corruptions: f64,
    /// Writes the queue accepted that never reached disk — a read-only, full
    /// or unmounted cache root, a failed rename, a failed fsync. The writer is
    /// fail-open and reports the error to nobody, so without this a cache that
    /// stores nothing at all still looks perfectly healthy.
    pub write_errors: f64,
    /// Restores refused before any block was looked up. Neither a hit nor a
    /// miss — so a refused restore reads as `0/0`, exactly like a turn that
    /// never consulted the tier.
    pub restore_declines: f64,
}

/// Return a snapshot of the process-wide cold tier. Read-only: never opens
/// the tier itself, so it reports `enabled: false` until inference first
/// initializes the tier.
///
/// The source snapshot is DESTRUCTURED rather than field-accessed, so a
/// counter added to `ColdCacheStats` and forgotten here fails to compile.
/// The failure this guards is silent and has already happened twice: a native
/// counter that never reaches this struct reaches no JS consumer either, and
/// nothing downstream can tell "the counter is zero" from "the counter was
/// never carried across".
#[napi]
pub fn cold_cache_stats() -> ColdCacheStatsJs {
    match GLOBAL.get().and_then(|slot| slot.clone()) {
        Some(manager) => {
            let mlx_paged_attn::ColdCacheStats {
                hits,
                misses,
                enqueued,
                queue_drops,
                bytes_written,
                bytes_restored,
                evictions,
                corruptions,
                write_errors,
                restore_declines,
            } = manager.stats();
            ColdCacheStatsJs {
                enabled: true,
                root: manager.root().display().to_string(),
                quota_bytes: manager.quota_bytes() as f64,
                hits: hits as f64,
                misses: misses as f64,
                enqueued: enqueued as f64,
                queue_drops: queue_drops as f64,
                bytes_written: bytes_written as f64,
                bytes_restored: bytes_restored as f64,
                evictions: evictions as f64,
                corruptions: corruptions as f64,
                write_errors: write_errors as f64,
                restore_declines: restore_declines as f64,
            }
        }
        None => ColdCacheStatsJs::default(),
    }
}

/// Snapshot of [`ColdSidecarTelemetry`] for JS. Counters are cumulative since
/// process start; all values are `f64` to avoid BigInt round-trips.
///
/// Separate from [`ColdCacheStatsJs`] because this one isolates SIDECARS — the
/// recurrent and sliding-window state that lives outside the pool — while that
/// one is scoped to the write queue as a whole. Both structs carry an
/// `enqueued` and a `queue_drops`, and they are NOT disjoint: a sidecar
/// admission bumps both, so these are a subset of those and summing them
/// double-counts. The JS side keeps them apart by prefix (`coldEnqueued` vs
/// `coldSidecarEnqueued`); read the sidecar pair when you need "did the family
/// state persist?", the block pair when you need "is the writer keeping up?".
#[napi(object, js_name = "ColdSidecarStats")]
#[derive(Clone, Debug, Default)]
pub struct ColdSidecarStatsJs {
    /// Turns that reached a family's sidecar capture at all. Every other
    /// counter here is a sub-count of this one, so `captureReached == 0`
    /// separates "the finalize path never calls the capture" from "the capture
    /// ran and declined".
    pub capture_reached: f64,
    /// Turns whose persisted K/V chain covered no whole block, so there was no
    /// prefix to anchor recurrent state under.
    pub chain_empty: f64,
    /// Turns whose chain covered blocks but where no retained checkpoint sat at
    /// or below its reach.
    pub boundary_skips: f64,
    /// Turns that selected a boundary already on disk — nothing written, and
    /// nothing needed to be. The steady state of a repeated prompt, and the
    /// only thing that tells a healthy run from a collapsed ladder.
    pub already_persisted: f64,
    /// Sidecars handed to the bounded writer queue.
    pub enqueued: f64,
    /// Sidecars the bounded writer queue refused because it was full.
    pub queue_drops: f64,
    /// Restored sidecars a family actually INSTALLED as its live per-turn
    /// state. The one read-side counter, and the only signal that separates
    /// "restored and used" from "restored and silently re-derived by a full
    /// O(prefix) replay" — every other counter, and text parity itself, is
    /// satisfied by the replay.
    pub installed: f64,
    /// Restores a family THREW AWAY after the walk served them, restarting the
    /// turn cold. Unlike `ColdCacheStats.restoreDeclines` this one comes AFTER
    /// real `coldHits` and `coldBytesRestored`, so the turn looks like it
    /// reused a prefix right up to the point where it recomputed all of it.
    pub restore_suppressed: f64,
}

/// Return the process-wide sidecar counters. Unlike [`cold_cache_stats`] this
/// never consults the tier at all, so it is valid before any inference has run
/// and reports honestly even when the tier failed to open.
///
/// The plain-Rust [`cold_sidecar_telemetry`] stays as it is: the parity harness
/// and the in-crate tests destructure `ColdSidecarTelemetry`, and a napi
/// `#[napi(object)]` return type cannot serve both.
#[napi]
pub fn cold_sidecar_stats() -> ColdSidecarStatsJs {
    // Destructured for the same reason [`cold_cache_stats`] is: a counter
    // added to the telemetry and forgotten here must not compile.
    let ColdSidecarTelemetry {
        capture_reached,
        chain_empty,
        boundary_skips,
        already_persisted,
        enqueued,
        queue_drops,
        installed,
        restore_suppressed,
    } = cold_sidecar_telemetry();
    ColdSidecarStatsJs {
        capture_reached: capture_reached as f64,
        chain_empty: chain_empty as f64,
        boundary_skips: boundary_skips as f64,
        already_persisted: already_persisted as f64,
        enqueued: enqueued as f64,
        queue_drops: queue_drops as f64,
        installed: installed as f64,
        restore_suppressed: restore_suppressed as f64,
    }
}

/// Flush every accepted cold-tier block to disk, blocking until the
/// background writer has fsync+renamed each write enqueued before this call,
/// or `timeout_ms` elapses. Returns `true` when the drain completed — or when
/// the tier was never opened (nothing to flush) — and `false` on timeout.
///
/// Called from the agent's one-shot (`mlx agent -p`) shutdown so a prompt's
/// just-persisted prefix blocks reach the drive before the process exits,
/// rather than being abandoned in the write queue.
///
/// The payload flush is `fsync(2)`, not `F_FULLFSYNC`: a drained block
/// survives process death and kernel panic, but a sudden power loss can leave
/// it torn. Torn objects fail the payload checksum on the next read and are
/// pruned as a miss, so the cost is a recomputed prefix, never wrong state.
#[napi]
pub fn cold_cache_drain(timeout_ms: u32) -> bool {
    match GLOBAL.get().and_then(|slot| slot.clone()) {
        Some(manager) => manager.drain(std::time::Duration::from_millis(timeout_ms as u64)),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    fn unique_tmp(tag: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("mlx-{tag}-{}-{n}-{nanos}", std::process::id()))
    }

    fn geometry() -> ColdTierGeometry {
        ColdTierGeometry {
            block_size: 16,
            num_layers: 24,
            num_kv_heads: 2,
            head_size: 128,
            cache_dtype: "BFloat16".to_string(),
        }
    }

    /// Write a minimal but well-formed safetensors file: the 8-byte LE header
    /// length prefix, the JSON header, then the raw data payload.
    fn write_safetensors(path: &Path, header: &str, data: &[u8]) {
        let header_bytes = header.as_bytes();
        let mut bytes = Vec::with_capacity(8 + header_bytes.len() + data.len());
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header_bytes);
        bytes.extend_from_slice(data);
        std::fs::write(path, bytes).unwrap();
    }

    /// Shared header: two finetunes of the same base carry the identical
    /// tensor layout (names/shapes/offsets), so their headers and total byte
    /// sizes match — only the weight bytes differ.
    const SHARED_HEADER: &str = r#"{"w":{"dtype":"U8","shape":[4096],"data_offsets":[0,4096]}}"#;

    /// A witness standing in for a real `materialize_weights` pass. Its counts
    /// only have to be non-zero: `build_model_fingerprint` refuses a witness
    /// that materialized nothing, and every identity test here is about the
    /// shard bytes, not the pass size.
    fn materialized() -> WeightsResident {
        WeightsResident::for_test(1, 4096)
    }

    fn build(dir: &Path) -> Option<ColdCacheFingerprint> {
        build_with_witness(dir, &materialized())
    }

    fn build_with_witness(dir: &Path, weights: &WeightsResident) -> Option<ColdCacheFingerprint> {
        let config = br#"{"model_type":"qwen3","hidden_size":1024}"#;
        build_model_fingerprint(
            "qwen3",
            dir.to_str().unwrap(),
            Some(config),
            &geometry(),
            weights,
        )
    }

    /// Pin a shard file's modification time. The cheap manifest path now folds
    /// mtime into the fingerprint, so tests that assert cross-dir EQUALITY of
    /// the cheap path must equalize the shard mtime across the two dirs.
    fn set_shard_mtime(path: &Path, time: std::time::SystemTime) {
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(time)
            .unwrap();
    }

    #[test]
    fn cold_restore_allowlist_covers_gated_families_only() {
        // Dense: the pool holds the whole prefix.
        assert!(cold_restore_supported("qwen3"));
        // Hybrid, admitted only because its out-of-pool sliding-window state is
        // persisted as a `ColdGroup::SlidingWindow` sidecar AND its
        // restart-parity gate passes on real weights
        // (`crates/mlx-core/tests/gemma4_cold_tier_parity.rs`).
        assert!(cold_restore_supported("gemma4"));
        // Hybrid, admitted only because its out-of-pool GDN recurrent state is
        // persisted as a `ColdGroup::GdnState` sidecar AND its restart-parity
        // gate passes on real weights
        // (`crates/mlx-core/tests/qwen3_5_cold_tier_parity.rs`).
        assert!(cold_restore_supported("qwen3_5"));
        // Shares qwen3_5's GDN state type and sidecar codec, driven through
        // `Qwen3_5MoeConfig::to_dense_config()`, and admitted on its OWN
        // restart-parity gate against a real MoE checkpoint
        // (`crates/mlx-core/tests/qwen3_5_moe_cold_tier_parity.rs`) — sharing a
        // codec with a passing family is not evidence, so the MoE ran its own.
        assert!(cold_restore_supported("qwen3_5_moe"));
        // Every remaining family still keeps per-token state outside the paged
        // pool with no cold-tier attach wired. lfm2 / lfm2_moe matter most:
        // they drive the uniform adapter API whose restore branch is already
        // wired to the tier, so admitting them would corrupt output rather
        // than merely slow it down.
        for family in [
            "lfm2",
            "lfm2_moe",
            // Fails closed on anything unrecognized, including near-misses.
            "Qwen3",
            "qwen3 ",
            "qwen3_dense",
            "Qwen3_5",
            "qwen3_5 ",
            "Gemma4",
            "gemma4_text",
            "",
        ] {
            assert!(
                !cold_restore_supported(family),
                "{family:?} must not be cold-restorable"
            );
        }
        // The napi surface a TypeScript test compares against must expose the
        // same list, not a superset.
        assert_eq!(
            cold_restore_families(),
            vec![
                "gemma4".to_string(),
                "qwen3".to_string(),
                "qwen3_5".to_string(),
                "qwen3_5_moe".to_string()
            ]
        );
    }

    #[test]
    fn fingerprint_binds_to_full_content_beyond_sampled_windows() {
        // Regression for the sampling gap the round-1 fix left open: identical
        // config, identical shard filename, identical shard SIZE, and bytes
        // that match inside the old head/middle/tail 256 KiB windows — they
        // differ ONLY at an offset BETWEEN the windows. Fixed-window sampling
        // returned EQUAL here (silent cross-model KV restore); a full content
        // digest must return DIFFERENT. Data region is 2 MiB so the three
        // 256 KiB windows leave uncovered gaps; the mutated offset (500_000)
        // sits in the first gap `[256 KiB, ~896 KiB)`.
        const DATA_LEN: usize = 2 * 1024 * 1024;
        const GAP_OFFSET: usize = 500_000;
        let base = unique_tmp("cold-fp-gap");
        let dir_a = base.join("a");
        let dir_b = base.join("b");
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();

        let mut data_a = vec![0u8; DATA_LEN];
        let mut data_b = vec![0u8; DATA_LEN];
        // Mutate ONE byte deep in the sampling gap; everything the old windows
        // covered (first/middle/last 256 KiB) stays byte-identical.
        data_a[GAP_OFFSET] = 0x11;
        data_b[GAP_OFFSET] = 0x22;
        write_safetensors(&dir_a.join("model.safetensors"), SHARED_HEADER, &data_a);
        write_safetensors(&dir_b.join("model.safetensors"), SHARED_HEADER, &data_b);

        let size_a = std::fs::metadata(dir_a.join("model.safetensors"))
            .unwrap()
            .len();
        let size_b = std::fs::metadata(dir_b.join("model.safetensors"))
            .unwrap()
            .len();
        assert_eq!(size_a, size_b, "test fixture must keep shard sizes equal");

        let fp_a = build(&dir_a).expect("fingerprint a");
        let fp_b = build(&dir_b).expect("fingerprint b");
        assert_ne!(
            fp_a, fp_b,
            "a byte differing only OUTSIDE the sampled windows must still split the fingerprint"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn fingerprint_is_deterministic_for_identical_bytes() {
        let base = unique_tmp("cold-fp-determinism");
        let dir_a = base.join("a");
        let dir_b = base.join("b");
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        write_safetensors(
            &dir_a.join("model.safetensors"),
            SHARED_HEADER,
            &[3u8; 4096],
        );
        write_safetensors(
            &dir_b.join("model.safetensors"),
            SHARED_HEADER,
            &[3u8; 4096],
        );

        let fp_a = build(&dir_a).expect("fingerprint a");
        let fp_b = build(&dir_b).expect("fingerprint b");
        assert_eq!(
            fp_a, fp_b,
            "identical bytes must yield identical fingerprints"
        );
        // Same directory re-read must reproduce the fingerprint (restart
        // determinism — persistence hinges on this).
        assert_eq!(fp_a, build(&dir_a).unwrap());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn unreadable_shard_disables_persistence() {
        // A shard the full digest cannot read to completion (here a dangling
        // symlink whose `File::open` fails) means a COMPLETE identity cannot be
        // established, so the builder must fail-safe to `None` rather than bind
        // under a partial identity that could collide with another model.
        let dir = unique_tmp("cold-fp-unreadable");
        std::fs::create_dir_all(&dir).unwrap();
        std::os::unix::fs::symlink(
            dir.join("nonexistent-target"),
            dir.join("model.safetensors"),
        )
        .unwrap();
        assert!(
            build(&dir).is_none(),
            "an unreadable shard must disable persistence, not weaken the fingerprint"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fingerprint_refuses_a_witness_that_materialized_nothing() {
        // The `WeightsResident` parameter is a compile-time ordering gate: the
        // identity read must run AFTER the loader's lazy `pread` of the shards.
        // A bare ZST gate would be launderable — `materialize_weights(&[])?`
        // above the attach mints a witness while reading nothing — so the
        // witness carries counts and a zero-coverage one is refused here.
        let dir = unique_tmp("cold-fp-empty-witness");
        std::fs::create_dir_all(&dir).unwrap();
        write_safetensors(&dir.join("model.safetensors"), SHARED_HEADER, &[7u8; 4096]);

        assert!(
            build_with_witness(&dir, &materialized()).is_some(),
            "a real materialize pass over this shard must yield a fingerprint"
        );
        assert!(
            build_with_witness(&dir, &WeightsResident::for_test(0, 0)).is_none(),
            "a witness that materialized nothing proves nothing about these shard \
             bytes, so persistence must stay off"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_shards_disables_persistence() {
        let dir = unique_tmp("cold-fp-noshards");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        assert!(
            build(&dir).is_none(),
            "a directory with no weight shard has no content to bind"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_revision_discriminates_identical_weights() {
        // Cheap manifest path: markers whose `files` cover the shard, so the
        // immutable repo+revision identifies the bytes without reading them.
        let base = unique_tmp("cold-fp-marker");
        let dir_a = base.join("a");
        let dir_b = base.join("b");
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        // Byte-identical weights: without a marker they collide (intended —
        // truly identical checkpoints share persisted KV).
        write_safetensors(
            &dir_a.join("model.safetensors"),
            SHARED_HEADER,
            &[5u8; 4096],
        );
        write_safetensors(
            &dir_b.join("model.safetensors"),
            SHARED_HEADER,
            &[5u8; 4096],
        );
        assert_eq!(build(&dir_a).unwrap(), build(&dir_b).unwrap());

        // A differing pinned revision must split them even though every weight
        // byte matches.
        std::fs::write(
            dir_a.join(DOWNLOAD_COMPLETE_MARKER),
            br#"{"repo":"acme/base","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","files":["config.json","model.safetensors"],"completedAt":"2026-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        std::fs::write(
            dir_b.join(DOWNLOAD_COMPLETE_MARKER),
            br#"{"repo":"acme/base","revision":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","files":["config.json","model.safetensors"],"completedAt":"2026-01-02T00:00:00Z"}"#,
        )
        .unwrap();
        assert_ne!(
            build(&dir_a).unwrap(),
            build(&dir_b).unwrap(),
            "a differing manifest revision must split identical weights"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn manifest_path_uses_provenance_not_weight_bytes() {
        // When the marker covers the shard, the fingerprint is driven by the
        // immutable repo+revision and NOT by the weight bytes. Two dirs with
        // the SAME marker but DIFFERENT shard bytes therefore fingerprint
        // EQUAL — observable proof the cheap path never reads the weights.
        let base = unique_tmp("cold-fp-marker-cheap");
        let dir_a = base.join("a");
        let dir_b = base.join("b");
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        // Same size, DIFFERENT bytes — a full digest would split them.
        write_safetensors(
            &dir_a.join("model.safetensors"),
            SHARED_HEADER,
            &[1u8; 4096],
        );
        write_safetensors(
            &dir_b.join("model.safetensors"),
            SHARED_HEADER,
            &[2u8; 4096],
        );
        let marker = br#"{"repo":"acme/base","revision":"deadbeefdeadbeefdeadbeefdeadbeefdeadbeef","files":["config.json","model.safetensors"],"completedAt":"2026-01-01T00:00:00Z"}"#;
        std::fs::write(dir_a.join(DOWNLOAD_COMPLETE_MARKER), marker).unwrap();
        std::fs::write(dir_b.join(DOWNLOAD_COMPLETE_MARKER), marker).unwrap();
        // Equalize the shard mtime so the ONLY difference is the weight bytes:
        // the cheap path folds in name+size+mtime+provenance, so proving it
        // still fingerprints EQUAL here proves it never read the weight bytes.
        let mtime = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
        set_shard_mtime(&dir_a.join("model.safetensors"), mtime);
        set_shard_mtime(&dir_b.join("model.safetensors"), mtime);
        assert_eq!(
            build(&dir_a).unwrap(),
            build(&dir_b).unwrap(),
            "the manifest path must bind to repo+revision and skip reading weight bytes"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn manifest_mtime_discriminates_identical_provenance() {
        // Regression guard for the cheap-path mtime binding: two dirs identical
        // in marker (repo+revision), shard filename, shard BYTES, and size, but
        // with DIFFERENT shard mtimes, must fingerprint DIFFERENTLY. An
        // out-of-band in-place rewrite that preserves size still moves mtime, so
        // stale KV can no longer silently cross-restore under an unchanged
        // marker.
        let base = unique_tmp("cold-fp-marker-mtime");
        let dir_a = base.join("a");
        let dir_b = base.join("b");
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        // Byte-identical, same size: only the differing mtime can split them.
        write_safetensors(
            &dir_a.join("model.safetensors"),
            SHARED_HEADER,
            &[7u8; 4096],
        );
        write_safetensors(
            &dir_b.join("model.safetensors"),
            SHARED_HEADER,
            &[7u8; 4096],
        );
        let marker = br#"{"repo":"acme/base","revision":"deadbeefdeadbeefdeadbeefdeadbeefdeadbeef","files":["config.json","model.safetensors"],"completedAt":"2026-01-01T00:00:00Z"}"#;
        std::fs::write(dir_a.join(DOWNLOAD_COMPLETE_MARKER), marker).unwrap();
        std::fs::write(dir_b.join(DOWNLOAD_COMPLETE_MARKER), marker).unwrap();
        set_shard_mtime(
            &dir_a.join("model.safetensors"),
            std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000),
        );
        set_shard_mtime(
            &dir_b.join("model.safetensors"),
            std::time::UNIX_EPOCH + std::time::Duration::from_secs(2_000_000),
        );
        assert_ne!(
            build(&dir_a).unwrap(),
            build(&dir_b).unwrap(),
            "the cheap path must fold in shard mtime so a same-size in-place rewrite splits the fingerprint"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn incomplete_manifest_falls_back_to_full_digest() {
        // A marker that does NOT list every on-disk shard cannot be trusted to
        // pin the bytes, so the builder falls back to the full digest — which
        // splits differing bytes even under an otherwise identical marker.
        let base = unique_tmp("cold-fp-marker-partial");
        let dir_a = base.join("a");
        let dir_b = base.join("b");
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        write_safetensors(
            &dir_a.join("model.safetensors"),
            SHARED_HEADER,
            &[3u8; 4096],
        );
        write_safetensors(
            &dir_b.join("model.safetensors"),
            SHARED_HEADER,
            &[4u8; 4096],
        );
        // `files` omits the shard -> not covered -> fallback path.
        let marker = br#"{"repo":"acme/base","revision":"cafef00dcafef00dcafef00dcafef00dcafef00d","files":["config.json"],"completedAt":"2026-01-01T00:00:00Z"}"#;
        std::fs::write(dir_a.join(DOWNLOAD_COMPLETE_MARKER), marker).unwrap();
        std::fs::write(dir_b.join(DOWNLOAD_COMPLETE_MARKER), marker).unwrap();
        assert_ne!(
            build(&dir_a).unwrap(),
            build(&dir_b).unwrap(),
            "an incomplete marker must not suppress the full-digest split"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn subdir_manifest_entry_does_not_cover_top_level_shard() {
        // On-disk shards enumerate as top-level basenames. A marker whose
        // `files` list the real weight under a subdir names a DIFFERENT file, so
        // it must NOT cover a distinct top-level shard of the same basename:
        // coverage fails, the builder falls to the full digest, and
        // byte-different top-level shards of equal size fingerprint DIFFERENTLY
        // (no basename-collision cross-model KV restore).
        let base = unique_tmp("cold-fp-subdir");
        let dir_a = base.join("a");
        let dir_b = base.join("b");
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        // Same size, DIFFERENT bytes: only the full digest can split them.
        write_safetensors(
            &dir_a.join("model.safetensors"),
            SHARED_HEADER,
            &[1u8; 4096],
        );
        write_safetensors(
            &dir_b.join("model.safetensors"),
            SHARED_HEADER,
            &[2u8; 4096],
        );
        // `files` lists the shard under a subdir -> does not cover the top-level
        // `model.safetensors` -> full-digest fallback.
        let marker = br#"{"repo":"acme/base","revision":"deadbeefdeadbeefdeadbeefdeadbeefdeadbeef","files":["config.json","weights/model.safetensors"],"completedAt":"2026-01-01T00:00:00Z"}"#;
        std::fs::write(dir_a.join(DOWNLOAD_COMPLETE_MARKER), marker).unwrap();
        std::fs::write(dir_b.join(DOWNLOAD_COMPLETE_MARKER), marker).unwrap();
        assert_ne!(
            build(&dir_a).unwrap(),
            build(&dir_b).unwrap(),
            "a subdir-prefixed manifest entry must not cover a distinct top-level shard"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn mutable_revision_marker_is_rejected() {
        // A legacy marker can pin a MUTABLE ref (e.g. "main") that could point
        // at different weights across downloads. `read_download_manifest` must
        // reject any non-40-hex revision, forcing the sound full-digest path.
        let base = unique_tmp("cold-fp-mutable-rev");
        let dir_a = base.join("a");
        let dir_b = base.join("b");
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        // Same size, DIFFERENT bytes: the cheap path (same marker) would bind
        // them EQUAL; only a full digest splits them.
        write_safetensors(
            &dir_a.join("model.safetensors"),
            SHARED_HEADER,
            &[1u8; 4096],
        );
        write_safetensors(
            &dir_b.join("model.safetensors"),
            SHARED_HEADER,
            &[2u8; 4096],
        );
        // Otherwise valid marker (files cover the shard) but a mutable revision.
        let marker = br#"{"repo":"acme/base","revision":"main","files":["config.json","model.safetensors"],"completedAt":"2026-01-01T00:00:00Z"}"#;
        std::fs::write(dir_a.join(DOWNLOAD_COMPLETE_MARKER), marker).unwrap();
        std::fs::write(dir_b.join(DOWNLOAD_COMPLETE_MARKER), marker).unwrap();
        assert!(
            read_download_manifest(&dir_a).is_none(),
            "a mutable (non-40-hex) revision must be rejected outright"
        );
        assert_ne!(
            build(&dir_a).unwrap(),
            build(&dir_b).unwrap(),
            "a mutable-revision marker must not take the cheap path: full digest must split differing bytes"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn covers_all_shards_requires_single_component_basename() {
        let shards = vec!["model.safetensors".to_string()];
        let manifest = |files: &[&str]| DownloadManifest {
            repo: Some("acme/base".to_string()),
            revision: "deadbeef".to_string(),
            files: files.iter().map(|s| s.to_string()).collect(),
        };
        // Positive: the bare top-level basename covers (cheap path preserved).
        assert!(manifest(&["model.safetensors"]).covers_all_shards(&shards));
        // Negative: any directory prefix / absolute / dot segment names a
        // DIFFERENT (or malformed) file and must fall through to full digest.
        assert!(!manifest(&["weights/model.safetensors"]).covers_all_shards(&shards));
        assert!(!manifest(&["/model.safetensors"]).covers_all_shards(&shards));
        assert!(!manifest(&["../model.safetensors"]).covers_all_shards(&shards));
        assert!(!manifest(&["./model.safetensors"]).covers_all_shards(&shards));
        // Negative: a trailing slash/dot is a DIFFERENT raw string. `Path`
        // normalization would fold these back to `model.safetensors` and
        // wrongly report coverage; raw equality must reject them.
        assert!(!manifest(&["model.safetensors/"]).covers_all_shards(&shards));
        assert!(!manifest(&["model.safetensors/."]).covers_all_shards(&shards));
        assert!(!manifest(&["model.safetensors//"]).covers_all_shards(&shards));
    }

    #[test]
    fn manifest_timestamp_does_not_shift_fingerprint() {
        // Re-downloading the SAME snapshot rewrites `completedAt` but keeps
        // `repo`/`revision`; the fingerprint must not move (no false miss).
        let base = unique_tmp("cold-fp-marker-ts");
        let dir_a = base.join("a");
        let dir_b = base.join("b");
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        write_safetensors(
            &dir_a.join("model.safetensors"),
            SHARED_HEADER,
            &[9u8; 4096],
        );
        write_safetensors(
            &dir_b.join("model.safetensors"),
            SHARED_HEADER,
            &[9u8; 4096],
        );
        std::fs::write(
            dir_a.join(DOWNLOAD_COMPLETE_MARKER),
            br#"{"repo":"acme/base","revision":"cccccccccccccccccccccccccccccccccccccccc","files":["config.json","model.safetensors"],"completedAt":"2026-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        std::fs::write(
            dir_b.join(DOWNLOAD_COMPLETE_MARKER),
            br#"{"repo":"acme/base","revision":"cccccccccccccccccccccccccccccccccccccccc","files":["config.json","model.safetensors"],"completedAt":"2026-07-20T12:34:56Z"}"#,
        )
        .unwrap();
        // Only the manifest `completedAt` differs; equalize the shard mtime so
        // the cheap-path fingerprint (name+size+mtime+provenance) is driven
        // purely by the identical repo+revision.
        let mtime = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
        set_shard_mtime(&dir_a.join("model.safetensors"), mtime);
        set_shard_mtime(&dir_b.join("model.safetensors"), mtime);
        assert_eq!(
            build(&dir_a).unwrap(),
            build(&dir_b).unwrap(),
            "only the manifest timestamp differs — the fingerprint must be stable"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    // GLOBAL is a process-wide OnceLock, so this must stay the ONLY test in
    // the binary that initializes it: one test, ordered assertions.
    #[test]
    fn stats_reflect_env_overridden_init() {
        let before = cold_cache_stats();
        assert!(
            !before.enabled,
            "GLOBAL initialized before the only init test ran"
        );
        assert_eq!(before.hits, 0.0);
        assert!(before.root.is_empty());
        assert!(cold_cache_stats_snapshot().is_none());

        let dir = std::env::temp_dir().join(format!("mlx-cold-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let foreign_tmp = dir.join("foo.tmp");
        let foreign_data = dir.join("data.txt");
        std::fs::write(&foreign_tmp, b"foreign").unwrap();
        std::fs::write(&foreign_data, b"data").unwrap();
        #[cfg(unix)]
        let parent_mode = {
            use std::os::unix::fs::PermissionsExt;
            std::fs::metadata(&dir).unwrap().permissions().mode() & 0o7777
        };

        // SAFETY: this is the only test in the binary touching
        // MLX_COLD_CACHE_DIR and nothing else reads it concurrently.
        unsafe { std::env::set_var(COLD_CACHE_DIR_ENV, &dir) };
        let manager = global_cold_cache().expect("temp-dir cold tier must open");
        assert_eq!(manager.root(), dir.join("mlx-paged-v1"));
        assert!(manager.root().is_dir());
        assert!(
            foreign_tmp.exists(),
            "init must not delete files in the user-supplied parent dir"
        );
        assert!(foreign_data.exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&dir).unwrap().permissions().mode() & 0o7777,
                parent_mode,
                "init must not chmod the user-supplied parent dir"
            );
        }

        let after = cold_cache_stats();
        assert!(after.enabled);
        assert_eq!(after.root, manager.root().display().to_string());
        assert!(after.quota_bytes > 0.0);
        assert!(cold_cache_stats_snapshot().is_some());

        // SAFETY: see above.
        unsafe { std::env::remove_var(COLD_CACHE_DIR_ENV) };
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Uses the non-global helper only: never initializes GLOBAL and never
    // reads MLX_COLD_CACHE_DIR, so it stays independent of the single
    // OnceLock-initializing test above.
    #[cfg(unix)]
    #[test]
    fn managed_child_symlink_or_file_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("mlx-cold-symlink-{}", std::process::id()));
        let victim = base.join("victim");
        std::fs::create_dir_all(&victim).unwrap();
        let marker = victim.join("marker.txt");
        let victim_tmp = victim.join("foo.tmp");
        std::fs::write(&marker, b"marker").unwrap();
        std::fs::write(&victim_tmp, b"tmp").unwrap();
        let victim_mode = std::fs::metadata(&victim).unwrap().permissions().mode() & 0o7777;

        let parent = base.join("parent");
        std::fs::create_dir_all(&parent).unwrap();
        std::os::unix::fs::symlink(&victim, parent.join(MANAGED_SUBDIR)).unwrap();
        assert!(
            open_managed_cold_cache(&parent).is_none(),
            "a symlinked managed child must be refused, not followed"
        );
        assert!(marker.exists());
        assert!(victim_tmp.exists());
        assert_eq!(
            std::fs::metadata(&victim).unwrap().permissions().mode() & 0o7777,
            victim_mode,
            "refusal must precede any chmod through the link"
        );

        let file_parent = base.join("parent-file");
        std::fs::create_dir_all(&file_parent).unwrap();
        std::fs::write(file_parent.join(MANAGED_SUBDIR), b"not a directory").unwrap();
        assert!(
            open_managed_cold_cache(&file_parent).is_none(),
            "a non-directory managed child must be refused"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    /// Each recorder must move its own counter and ONLY its own. Run against a
    /// fresh instance, so every count starts at zero and the whole snapshot can
    /// be compared exactly — a recorder wired to the wrong field, or bumping
    /// two at once, changes a field this asserts on.
    #[test]
    fn each_sidecar_recorder_moves_only_its_own_counter() {
        let counters = ColdSidecarCounters::default();
        assert_eq!(counters.snapshot(), ColdSidecarTelemetry::default());

        counters.record_capture_reached();
        assert_eq!(
            counters.snapshot(),
            ColdSidecarTelemetry {
                capture_reached: 1,
                ..Default::default()
            }
        );

        counters.record_chain_empty();
        assert_eq!(
            counters.snapshot(),
            ColdSidecarTelemetry {
                capture_reached: 1,
                chain_empty: 1,
                ..Default::default()
            }
        );

        counters.record_boundary_skip();
        assert_eq!(
            counters.snapshot(),
            ColdSidecarTelemetry {
                capture_reached: 1,
                chain_empty: 1,
                boundary_skips: 1,
                ..Default::default()
            }
        );

        counters.record_already_persisted();
        assert_eq!(
            counters.snapshot(),
            ColdSidecarTelemetry {
                capture_reached: 1,
                chain_empty: 1,
                boundary_skips: 1,
                already_persisted: 1,
                ..Default::default()
            }
        );

        counters.record_enqueued();
        assert_eq!(
            counters.snapshot(),
            ColdSidecarTelemetry {
                capture_reached: 1,
                chain_empty: 1,
                boundary_skips: 1,
                already_persisted: 1,
                enqueued: 1,
                ..Default::default()
            }
        );

        counters.record_queue_drop();
        assert_eq!(
            counters.snapshot(),
            ColdSidecarTelemetry {
                capture_reached: 1,
                chain_empty: 1,
                boundary_skips: 1,
                already_persisted: 1,
                enqueued: 1,
                queue_drops: 1,
                ..Default::default()
            }
        );

        counters.record_install();
        assert_eq!(
            counters.snapshot(),
            ColdSidecarTelemetry {
                capture_reached: 1,
                chain_empty: 1,
                boundary_skips: 1,
                already_persisted: 1,
                enqueued: 1,
                queue_drops: 1,
                installed: 1,
                ..Default::default()
            }
        );

        counters.record_restore_suppressed();
        assert_eq!(
            counters.snapshot(),
            ColdSidecarTelemetry {
                capture_reached: 1,
                chain_empty: 1,
                boundary_skips: 1,
                already_persisted: 1,
                enqueued: 1,
                queue_drops: 1,
                installed: 1,
                restore_suppressed: 1,
            }
        );
    }

    /// The process-wide instance the capture paths actually record into is the
    /// same type, so a call through the free function lands in the same field.
    #[test]
    fn the_process_wide_counters_are_the_ones_the_capture_paths_record_into() {
        let _serialized = sidecar_counter_test_lock();
        let before = cold_sidecar_telemetry();
        cold_sidecar_counters().record_boundary_skip();
        cold_sidecar_counters().record_capture_reached();
        let after = cold_sidecar_telemetry();
        assert_eq!(after.boundary_skips, before.boundary_skips + 1);
        assert_eq!(after.capture_reached, before.capture_reached + 1);
    }

    /// The napi mirror must carry each counter into the field of the SAME name.
    ///
    /// Asserted as a DELTA with a distinct increment per field, never as
    /// equality against a fixed snapshot: the counters are one process-wide
    /// static that earlier tests have already moved, so absolute values are not
    /// reproducible — and, more importantly, equal absolute values would let a
    /// cross-wired field (`capture_reached: telemetry.chain_empty`) pass
    /// whenever the two counters happened to agree. Distinct increments make
    /// every permutation observable.
    #[test]
    fn the_napi_sidecar_mirror_carries_each_counter_into_its_own_field() {
        let _serialized = sidecar_counter_test_lock();
        let before = cold_sidecar_stats();
        let counters = cold_sidecar_counters();
        for _ in 0..1 {
            counters.record_capture_reached();
        }
        for _ in 0..2 {
            counters.record_chain_empty();
        }
        for _ in 0..3 {
            counters.record_boundary_skip();
        }
        for _ in 0..4 {
            counters.record_already_persisted();
        }
        for _ in 0..5 {
            counters.record_enqueued();
        }
        for _ in 0..6 {
            counters.record_queue_drop();
        }
        for _ in 0..7 {
            counters.record_install();
        }
        for _ in 0..8 {
            counters.record_restore_suppressed();
        }
        let after = cold_sidecar_stats();
        assert_eq!(after.capture_reached - before.capture_reached, 1.0);
        assert_eq!(after.chain_empty - before.chain_empty, 2.0);
        assert_eq!(after.boundary_skips - before.boundary_skips, 3.0);
        assert_eq!(after.already_persisted - before.already_persisted, 4.0);
        assert_eq!(after.enqueued - before.enqueued, 5.0);
        assert_eq!(after.queue_drops - before.queue_drops, 6.0);
        assert_eq!(after.installed - before.installed, 7.0);
        assert_eq!(after.restore_suppressed - before.restore_suppressed, 8.0);

        // …and it must agree with the plain-Rust reader the parity harness
        // uses, so the two can never report different truths about the same
        // static.
        let telemetry = cold_sidecar_telemetry();
        let mirror = cold_sidecar_stats();
        assert_eq!(mirror.capture_reached, telemetry.capture_reached as f64);
        assert_eq!(mirror.chain_empty, telemetry.chain_empty as f64);
        assert_eq!(mirror.boundary_skips, telemetry.boundary_skips as f64);
        assert_eq!(mirror.already_persisted, telemetry.already_persisted as f64);
        assert_eq!(mirror.enqueued, telemetry.enqueued as f64);
        assert_eq!(mirror.queue_drops, telemetry.queue_drops as f64);
        assert_eq!(mirror.installed, telemetry.installed as f64);
        assert_eq!(
            mirror.restore_suppressed,
            telemetry.restore_suppressed as f64
        );
    }

    /// A real manager, a real write failure, and the counter read back through
    /// the same `stats()` snapshot the napi mirror maps from.
    ///
    /// The point is that BOTH new block counters move only for their own
    /// event: the failed write must not register as a decline, and the two
    /// declines must not register as writes. Asserted with distinct counts (1
    /// vs 2) so a crossed pair cannot pass.
    #[cfg(unix)]
    #[test]
    fn a_failed_write_and_a_refused_restore_land_in_different_block_counters() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "mlx-cold-tier-block-counters-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::remove_dir_all(&root);
        let Ok(manager) = mlx_paged_attn::ColdCacheManager::open_at(root.clone(), 1 << 30, 0, 4)
        else {
            eprintln!("skipping block-counter test: cold cache root would not open");
            return;
        };

        // Revoke write permission on the already-open root, so the enqueued
        // sidecar fails against the real filesystem rather than a hook.
        let _ = std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o500));
        let key = mlx_paged_attn::ColdCacheKey::chain(
            mlx_paged_attn::ColdGroup::GdnState,
            mlx_paged_attn::ColdCacheFingerprint::from_components([b"counters".as_slice()]),
            None,
            &[1, 2, 3, 4],
            &[],
            0,
            0,
        );
        let layout = mlx_paged_attn::ColdSidecarLayout {
            group: mlx_paged_attn::ColdGroup::GdnState,
            boundary_tokens: 16,
            num_layers: 1,
            tensors_per_layer: 1,
            dtype: "BFloat16".to_string(),
            dims: vec![2, 2],
            bytes_per_tensor: 8,
        };
        let accepted = manager
            .enqueue_sidecar(mlx_paged_attn::ColdSidecar {
                key,
                fingerprint: mlx_paged_attn::ColdCacheFingerprint::from_components([
                    b"counters".as_slice()
                ]),
                layout,
                tensors: vec![vec![0u8; 8]],
            })
            .unwrap_or(false);
        let durable = manager.drain(std::time::Duration::from_secs(10));

        manager.record_restore_decline();
        manager.record_restore_decline();

        let stats = manager.stats();
        // Every observation is taken before the root is made writable again,
        // and every assertion after it: a panic here must not unwind past the
        // restore and leave an undeletable directory in the temp dir.
        let _ = std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700));

        assert!(accepted, "the queue must accept the write it cannot land");
        assert!(
            !durable,
            "the drain must report the covered write as not durable"
        );
        assert_eq!(stats.write_errors, 1, "one write, refused by the disk");
        assert_eq!(stats.restore_declines, 2, "two restores, refused by policy");
        assert_eq!(stats.bytes_written, 0, "nothing landed");
        assert_eq!(stats.queue_drops, 0, "the queue was never full");
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);

        drop(manager);
        let _ = std::fs::remove_dir_all(&root);
    }
}
