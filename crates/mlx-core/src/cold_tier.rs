//! Process-global registry for the SSD cold tier
//! (`mlx_paged_attn::ColdCacheManager`) plus its NAPI stats surface.
//!
//! The tier is opened lazily on first use and is fail-open: if the root
//! cannot be opened (no HOME, unwritable disk, ...) inference proceeds
//! without persistence and `coldCacheStats()` reports `enabled: false`.

use std::path::Path;
use std::sync::{Arc, OnceLock};

use mlx_paged_attn::{ColdCacheManager, ColdCacheStats};
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
    use super::*;

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
