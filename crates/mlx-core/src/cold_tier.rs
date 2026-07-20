//! Process-global registry for the SSD cold tier
//! (`mlx_paged_attn::ColdCacheManager`) plus its NAPI stats surface.
//!
//! The tier is opened lazily on first use and is fail-open: if the root
//! cannot be opened (no HOME, unwritable disk, ...) inference proceeds
//! without persistence and `coldCacheStats()` reports `enabled: false`.

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
        .get_or_init(|| {
            let opened = match std::env::var(COLD_CACHE_DIR_ENV) {
                Ok(dir) if !dir.is_empty() => ColdCacheManager::open_default_at(
                    std::path::PathBuf::from(dir).join(MANAGED_SUBDIR),
                ),
                _ => ColdCacheManager::open_default(),
            };
            opened.ok().map(Arc::new)
        })
        .clone()
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
    /// Writes dropped because the bounded queue was full.
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
}
