//! Process-global registry for the SSD cold tier
//! (`mlx_paged_attn::ColdCacheManager`) plus its NAPI stats surface.
//!
//! The tier is opened lazily on first use and is fail-open: if the root
//! cannot be opened (no HOME, unwritable disk, ...) inference proceeds
//! without persistence and `coldCacheStats()` reports `enabled: false`.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use mlx_paged_attn::{ColdCacheFingerprint, ColdCacheManager, ColdCacheStats};
use napi_derive::napi;
use sha2::{Digest, Sha256};

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
pub(crate) fn build_model_fingerprint(
    model_type: &str,
    model_path: &str,
    config_json: Option<&[u8]>,
    geometry: &ColdTierGeometry,
) -> Option<ColdCacheFingerprint> {
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
    /// Blocks accepted onto the background write queue.
    pub enqueued: f64,
    /// Writes dropped without landing on disk: queue full at enqueue, or the commit rename failed.
    pub queue_drops: f64,
    /// Total bytes committed to disk.
    pub bytes_written: f64,
    /// Total bytes read back on validated hits.
    pub bytes_restored: f64,
    /// Entries evicted to respect the quota / free-space reserve.
    pub evictions: f64,
    /// Entries that failed checksum/identity validation and were removed.
    pub corruptions: f64,
}

/// Return a snapshot of the process-wide cold tier. Read-only: never opens
/// the tier itself, so it reports `enabled: false` until inference first
/// initializes the tier.
#[napi]
pub fn cold_cache_stats() -> ColdCacheStatsJs {
    match GLOBAL.get().and_then(|slot| slot.clone()) {
        Some(manager) => {
            let stats = manager.stats();
            ColdCacheStatsJs {
                enabled: true,
                root: manager.root().display().to_string(),
                quota_bytes: manager.quota_bytes() as f64,
                hits: stats.hits as f64,
                misses: stats.misses as f64,
                enqueued: stats.enqueued as f64,
                queue_drops: stats.queue_drops as f64,
                bytes_written: stats.bytes_written as f64,
                bytes_restored: stats.bytes_restored as f64,
                evictions: stats.evictions as f64,
                corruptions: stats.corruptions as f64,
            }
        }
        None => ColdCacheStatsJs::default(),
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

    fn build(dir: &Path) -> Option<ColdCacheFingerprint> {
        let config = br#"{"model_type":"qwen3","hidden_size":1024}"#;
        build_model_fingerprint("qwen3", dir.to_str().unwrap(), Some(config), &geometry())
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
}
