//! Process-global registry for the SSD cold tier
//! (`mlx_paged_attn::ColdCacheManager`) plus its NAPI stats surface.
//!
//! The tier is opened lazily on first use and is fail-open: if the root
//! cannot be opened (no HOME, unwritable disk, ...) inference proceeds
//! without persistence and `coldCacheStats()` reports `enabled: false`.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Arc, OnceLock};

use mlx_paged_attn::{ColdCacheFingerprint, ColdCacheManager, ColdCacheStats};
use napi_derive::napi;

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
/// publish. Carries the pinned HuggingFace commit `revision`, the strongest
/// (cryptographic) available identity for a downloaded checkpoint.
const DOWNLOAD_COMPLETE_MARKER: &str = ".mlx-download-complete.json";

/// Fixed weight-data window sampled at three offsets per shard. Two
/// same-architecture finetunes share tensor names/shapes/offsets — so their
/// safetensors headers AND total byte sizes are identical — yet differ in
/// the actual weight bytes, so the fingerprint MUST sample the data region.
const SHARD_DATA_WINDOW: u64 = 256 * 1024;

/// Upper bound on the safetensors header bytes folded per shard, so a
/// corrupt or absurd header-length prefix can never drive unbounded I/O.
const SHARD_HEADER_SAMPLE_CAP: u64 = 1024 * 1024;

/// Cache geometry whose drift must invalidate persisted blocks.
pub(crate) struct ColdTierGeometry {
    pub block_size: u64,
    pub num_layers: u64,
    pub num_kv_heads: u64,
    pub head_size: u64,
    pub cache_dtype: String,
}

/// Build the cold-tier fingerprint for a model directory, binding it to
/// weight CONTENT rather than only shard names and sizes.
///
/// Two same-architecture finetunes of the same base normally have identical
/// shard filenames AND identical byte sizes (same tensor shapes), so a
/// name+size fingerprint would collide and let one model restore the other's
/// persisted KV for a shared token prefix — silent output corruption. To
/// discriminate them cheaply and deterministically, each shard contributes a
/// bounded, content-sensitive sample of its weight DATA region (three fixed
/// 256 KiB windows) plus its safetensors header; total I/O is O(shards) and
/// independent of model size. When present, the weight_map index and the
/// dashboard download manifest's immutable revision are folded in as
/// additive (stronger) identities.
///
/// Returns `None` when any `.safetensors` shard cannot be read for a
/// complete content sample, or when the directory holds no shard at all: the
/// caller then leaves persistence OFF (fail-safe) rather than persisting
/// under a weak fingerprint that could collide with another model.
pub(crate) fn build_model_fingerprint(
    model_type: &str,
    model_path: &str,
    config_json: Option<&[u8]>,
    geometry: &ColdTierGeometry,
) -> Option<ColdCacheFingerprint> {
    let dir = Path::new(model_path);
    let mut components: Vec<Vec<u8>> = vec![model_type.as_bytes().to_vec()];
    if let Some(cfg) = config_json {
        components.push(cfg.to_vec());
    }

    // Provenance-independent baseline: weight-content sample per shard. A
    // single unreadable shard disables persistence for this model.
    let shards = sample_shard_digests(dir)?;
    if shards.is_empty() {
        return None;
    }
    for (name, size, digest) in &shards {
        components.push(name.as_bytes().to_vec());
        components.push(size.to_le_bytes().to_vec());
        components.push(digest.to_vec());
    }

    // Additive strengthenings that never read weights. The weight_map index
    // encodes the tensor->shard layout; the download manifest carries the
    // immutable commit revision (excluding its timestamp so an innocuous
    // re-download of the SAME snapshot never shifts the fingerprint).
    if let Ok(index_bytes) = std::fs::read(dir.join("model.safetensors.index.json")) {
        components.push(b"index.json\0".to_vec());
        components.push(index_bytes);
    }
    if let Some(revision) = download_marker_identity(dir) {
        components.push(b"download-marker\0".to_vec());
        components.push(revision);
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

/// Per-shard `(file_name, size, content_digest)` for every `.safetensors`
/// file in `dir`, sorted by name for restart-determinism. `None` if ANY
/// shard cannot be opened or a required sample cannot be read — a partial
/// identity must disable persistence rather than weaken it.
fn sample_shard_digests(dir: &Path) -> Option<Vec<(String, u64, [u8; 32])>> {
    let read_dir = std::fs::read_dir(dir).ok()?;
    let mut shards = Vec::new();
    for entry in read_dir.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".safetensors") {
            continue;
        }
        let (size, digest) = sample_one_shard(&dir.join(&name))?;
        shards.push((name, size, digest));
    }
    shards.sort();
    Some(shards)
}

/// Bounded content digest of one safetensors shard: total size, the header
/// length prefix, a bounded slice of the JSON header (tensor layout), and
/// three fixed data-region windows (head, middle, tail). `File::open`
/// follows symlinks so symlinked checkpoints sample the real weight bytes.
/// `None` on any hard I/O error, including a shard too short to hold the
/// 8-byte safetensors length prefix.
fn sample_one_shard(path: &Path) -> Option<(u64, [u8; 32])> {
    let mut file = File::open(path).ok()?;
    let size = file.metadata().ok()?.len();

    let mut len_prefix = [0u8; 8];
    file.read_exact(&mut len_prefix).ok()?;
    let header_len = u64::from_le_bytes(len_prefix);
    let header_sample = read_window(&mut file, 8, header_len.min(SHARD_HEADER_SAMPLE_CAP))?;

    let data_start = 8u64.saturating_add(header_len);
    let data_len = size.saturating_sub(data_start);
    let mut windows: Vec<Vec<u8>> = Vec::with_capacity(3);
    for offset in data_window_offsets(data_start, data_len) {
        windows.push(read_window(&mut file, offset, SHARD_DATA_WINDOW)?);
    }

    let size_le = size.to_le_bytes();
    let mut components: Vec<&[u8]> = vec![
        b"mlx-node:cold-shard-sample:v1".as_slice(),
        &size_le,
        &len_prefix,
        &header_sample,
    ];
    for window in &windows {
        components.push(window.as_slice());
    }
    let digest = ColdCacheFingerprint::from_components(components);
    Some((size, *digest.as_bytes()))
}

/// Head/middle/tail window start offsets inside the data region
/// `[data_start, data_start + data_len)`. Saturating arithmetic clamps every
/// offset into the region so a data region smaller than the window overlaps
/// deterministically instead of underflowing.
fn data_window_offsets(data_start: u64, data_len: u64) -> [u64; 3] {
    let head = data_start;
    let middle = data_start + (data_len / 2).saturating_sub(SHARD_DATA_WINDOW / 2);
    let tail = data_start + data_len.saturating_sub(SHARD_DATA_WINDOW);
    [head, middle, tail]
}

/// Read up to `len` bytes at `offset`. A short read at EOF returns fewer
/// bytes (offsets past EOF yield an empty window); only a hard I/O error
/// returns `None`, so the caller disables persistence.
fn read_window(file: &mut File, offset: u64, len: u64) -> Option<Vec<u8>> {
    file.seek(SeekFrom::Start(offset)).ok()?;
    let mut buf = vec![0u8; len as usize];
    let mut filled = 0usize;
    while filled < buf.len() {
        match file.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return None,
        }
    }
    buf.truncate(filled);
    Some(buf)
}

/// Immutable identity from the dashboard download manifest: `repo` plus the
/// pinned commit `revision`. Excludes the manifest's `completedAt` timestamp
/// so re-downloading the SAME snapshot does not shift the fingerprint.
/// `None` when the marker is absent or unparseable — it is additive only,
/// and the weight-window sample already discriminates every model.
fn download_marker_identity(dir: &Path) -> Option<Vec<u8>> {
    let bytes = std::fs::read(dir.join(DOWNLOAD_COMPLETE_MARKER)).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let revision = value.get("revision")?.as_str()?;
    let mut identity = Vec::new();
    if let Some(repo) = value.get("repo").and_then(|v| v.as_str()) {
        identity.extend_from_slice(repo.as_bytes());
    }
    identity.push(0);
    identity.extend_from_slice(revision.as_bytes());
    Some(identity)
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

    #[test]
    fn fingerprint_binds_to_weight_content_not_name_or_size() {
        // Regression for the cross-model collision: identical config,
        // identical shard filename, identical shard SIZE — but different
        // weight bytes. A name+size fingerprint (the old builder) returned
        // EQUAL here; binding to content must return DIFFERENT.
        let base = unique_tmp("cold-fp-content");
        let dir_a = base.join("a");
        let dir_b = base.join("b");
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        write_safetensors(
            &dir_a.join("model.safetensors"),
            SHARED_HEADER,
            &[0u8; 4096],
        );
        write_safetensors(
            &dir_b.join("model.safetensors"),
            SHARED_HEADER,
            &[7u8; 4096],
        );

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
            "same shard name+size but different weight bytes must not collide"
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

    #[test]
    fn unreadable_shard_disables_persistence() {
        // A shard too short to even hold the 8-byte safetensors length prefix
        // is an I/O read failure: a complete identity cannot be established,
        // so the builder must fail-safe to `None`.
        let dir = unique_tmp("cold-fp-unreadable");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), [0u8; 3]).unwrap();
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
    fn download_marker_revision_discriminates_identical_weights() {
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

        // A differing pinned revision must split them even though every
        // sampled weight window matches.
        std::fs::write(
            dir_a.join(DOWNLOAD_COMPLETE_MARKER),
            br#"{"repo":"acme/base","revision":"aaaaaaaa","files":["config.json"],"completedAt":"2026-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        std::fs::write(
            dir_b.join(DOWNLOAD_COMPLETE_MARKER),
            br#"{"repo":"acme/base","revision":"bbbbbbbb","files":["config.json"],"completedAt":"2026-01-02T00:00:00Z"}"#,
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
    fn download_marker_timestamp_does_not_shift_fingerprint() {
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
            br#"{"repo":"acme/base","revision":"cccccccc","files":["config.json"],"completedAt":"2026-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        std::fs::write(
            dir_b.join(DOWNLOAD_COMPLETE_MARKER),
            br#"{"repo":"acme/base","revision":"cccccccc","files":["config.json"],"completedAt":"2026-07-20T12:34:56Z"}"#,
        )
        .unwrap();
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
