//! One-release cleanup for retired Flash Next weight packs. Direct GGUF
//! residency never creates these directories; only exact legacy names under
//! the old cache root are eligible, and removal is best-effort.
use std::fs::{self, File, OpenOptions};
use std::io::ErrorKind;
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use tracing::debug;

const MAPPED_DIR_PREFIX: &str = ".mlx-qwen4-mapped-v3-";
const RETIRED_CHUNK_PREFIX: &str = ".mlx-qwen4-packed-v2-";
const MAPPED_LOCK: &str = "lock";
const RETIRED_CHUNK_LOCK: &str = "budget";
// Old chunk readers did not always flock. The grace also covers a loader
// between directory creation and its first lock.
const RETIRED_SWEEP_MIN_AGE: Duration = Duration::from_secs(3600);

#[derive(Clone, Copy)]
enum RetiredPack {
    Mapped,
    Chunks,
}

impl RetiredPack {
    fn lock(self) -> &'static str {
        match self {
            Self::Mapped => MAPPED_LOCK,
            Self::Chunks => RETIRED_CHUNK_LOCK,
        }
    }

    fn reason(self) -> &'static str {
        match self {
            Self::Mapped => "retired mapped pack",
            Self::Chunks => "retired chunk pack",
        }
    }
}

fn has_identity_suffix(name: &str, prefix: &str) -> bool {
    name.strip_prefix(prefix).is_some_and(|suffix| {
        suffix.len() == 64
            && suffix
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    })
}

fn retired_pack(dir: &Path) -> Option<RetiredPack> {
    let name = dir.file_name()?.to_str()?;
    if has_identity_suffix(name, MAPPED_DIR_PREFIX) {
        Some(RetiredPack::Mapped)
    } else if has_identity_suffix(name, RETIRED_CHUNK_PREFIX) {
        Some(RetiredPack::Chunks)
    } else {
        None
    }
}

fn is_real_dir(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|meta| meta.is_dir())
}

fn newest_mtime(dir: &Path) -> Option<SystemTime> {
    let mut newest = fs::symlink_metadata(dir).ok()?.modified().ok()?;
    for entry in fs::read_dir(dir).ok()?.flatten() {
        newest = newest.max(fs::symlink_metadata(entry.path()).ok()?.modified().ok()?);
    }
    Some(newest)
}

fn idle(dir: &Path) -> bool {
    SystemTime::now()
        .checked_sub(RETIRED_SWEEP_MIN_AGE)
        .is_some_and(|cutoff| newest_mtime(dir).is_some_and(|mtime| mtime < cutoff))
}

/// Return the open lock file, or `None` inside `Some` when the legacy writer
/// had not created its lock yet. Any live holder or metadata error keeps the
/// directory. The inode check refuses a lock replaced during acquisition.
fn try_exclusive_lock(path: &Path) -> Option<Option<File>> {
    let file = match OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .or_else(|_| File::open(path))
    {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::NotFound => return Some(None),
        Err(_) => return None,
    };
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return None;
    }
    let held = file.metadata().ok()?;
    let live = fs::metadata(path).ok()?;
    (held.dev() == live.dev() && held.ino() == live.ino()).then_some(Some(file))
}

fn remove_retired(dir: &Path, kind: RetiredPack) {
    let Ok(before) = fs::symlink_metadata(dir) else {
        return;
    };
    if !before.is_dir() {
        return;
    }
    let Some(_guard) = try_exclusive_lock(&dir.join(kind.lock())) else {
        debug!(
            "qwen4 cache: keeping {} (legacy lock is held)",
            dir.display()
        );
        return;
    };
    // Recheck the name, inode, and grace with the lock held. A newly created
    // or replaced directory is not the one whose idleness was observed.
    let still_retired = fs::symlink_metadata(dir).is_ok_and(|now| {
        now.is_dir()
            && now.dev() == before.dev()
            && now.ino() == before.ino()
            && retired_pack(dir).is_some()
            && idle(dir)
    });
    if !still_retired {
        return;
    }
    match fs::remove_dir_all(dir) {
        Ok(()) => debug!("qwen4 cache: removed {} ({})", dir.display(), kind.reason()),
        Err(error) => debug!("qwen4 cache: failed to remove {}: {error}", dir.display()),
    }
}

/// Remove the two retired on-disk weight formats after one release of direct
/// residency. No root or marker is created. Symlinks, files, foreign suffixes,
/// fresh directories, and locked legacy readers are left untouched.
pub(super) fn sweep_old_packs(root: &Path) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Ok(entries) = fs::read_dir(root) else {
            return;
        };
        for entry in entries.flatten() {
            let dir = entry.path();
            if !is_real_dir(&dir) || !idle(&dir) {
                continue;
            }
            if let Some(kind) = retired_pack(&dir) {
                remove_retired(&dir, kind);
            }
        }
    }));
}

pub(super) fn cache_root(source: &Path) -> PathBuf {
    std::env::var_os("HOME").map_or_else(
        || source.parent().unwrap_or(Path::new(".")).to_path_buf(),
        |home| {
            PathBuf::from(home).join(if cfg!(target_os = "macos") {
                "Library/Caches/mlx-node/qwen4-packed"
            } else {
                ".cache/mlx-node/qwen4-packed"
            })
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Temp(PathBuf);

    impl Temp {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("qwen4-retired-{}", uuid::Uuid::new_v4()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn age(path: &Path) {
        let old = SystemTime::now() - Duration::from_secs(7200);
        File::open(path).unwrap().set_modified(old).unwrap();
    }

    fn old_dir(root: &Path, prefix: &str, suffix: &str, lock: &str, aged: bool) -> PathBuf {
        let dir = root.join(format!("{prefix}{suffix}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(lock), b"lock").unwrap();
        fs::write(dir.join("weights"), vec![7; 32]).unwrap();
        if aged {
            age(&dir.join(lock));
            age(&dir.join("weights"));
            age(&dir);
        }
        dir
    }

    fn mapped(root: &Path, suffix: &str, aged: bool) -> PathBuf {
        old_dir(root, MAPPED_DIR_PREFIX, suffix, MAPPED_LOCK, aged)
    }

    fn chunks(root: &Path, suffix: &str, aged: bool) -> PathBuf {
        old_dir(root, RETIRED_CHUNK_PREFIX, suffix, RETIRED_CHUNK_LOCK, aged)
    }

    #[test]
    fn retired_formats_are_removed_only_when_idle_and_exactly_named() {
        let temp = Temp::new();
        let id = "a".repeat(64);
        let old_mapped = mapped(&temp.0, &id, true);
        let old_chunks = chunks(&temp.0, &"b".repeat(64), true);
        let fresh = mapped(&temp.0, &"c".repeat(64), false);
        let foreign = [
            mapped(&temp.0, "abc", true),
            mapped(&temp.0, &"F".repeat(64), true),
            mapped(&temp.0, &format!("{}-backup", "d".repeat(64)), true),
            chunks(&temp.0, &"e".repeat(63), true),
        ];
        sweep_old_packs(&temp.0);
        assert!(!old_mapped.exists());
        assert!(!old_chunks.exists());
        assert!(fresh.exists(), "the idle grace protects new loaders");
        assert!(foreign.iter().all(|dir| dir.exists()));
    }

    #[test]
    fn symlinked_pack_and_validly_named_file_are_not_removed() {
        let temp = Temp::new();
        let outside = temp.0.join("outside");
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("weights"), b"keep").unwrap();
        age(&outside.join("weights"));
        age(&outside);
        let link = temp
            .0
            .join(format!("{MAPPED_DIR_PREFIX}{}", "a".repeat(64)));
        std::os::unix::fs::symlink(&outside, &link).unwrap();
        age(&link);
        let file = temp
            .0
            .join(format!("{RETIRED_CHUNK_PREFIX}{}", "b".repeat(64)));
        fs::write(&file, b"not a directory").unwrap();
        age(&file);
        sweep_old_packs(&temp.0);
        assert!(link.symlink_metadata().unwrap().is_symlink());
        assert_eq!(fs::read(outside.join("weights")).unwrap(), b"keep");
        assert!(file.exists());
    }

    // BSD flock is per-process, so the contended-lock case needs a child.
    #[test]
    #[ignore = "helper process for legacy lock cleanup"]
    fn cleanup_lock_child() {
        let path = std::env::var_os("MLX_QWEN4_CLEANUP_TEST_LOCK").unwrap();
        let acquired = std::env::var_os("MLX_QWEN4_CLEANUP_TEST_ACQUIRED").unwrap();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        assert_eq!(unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_SH) }, 0);
        fs::write(acquired, b"acquired").unwrap();
        std::thread::sleep(Duration::from_secs(60));
    }

    struct LockHolder(std::process::Child);

    impl Drop for LockHolder {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    fn shared_lock_holder(dir: &Path) -> LockHolder {
        let acquired = dir
            .parent()
            .unwrap()
            .join(format!("{}.acquired", uuid::Uuid::new_v4()));
        let holder = LockHolder(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "models::qwen4_exp::packed_cache::tests::cleanup_lock_child",
                    "--ignored",
                ])
                .env("MLX_QWEN4_CLEANUP_TEST_LOCK", dir.join(MAPPED_LOCK))
                .env("MLX_QWEN4_CLEANUP_TEST_ACQUIRED", &acquired)
                .spawn()
                .unwrap(),
        );
        for _ in 0..500 {
            if acquired.is_file() {
                return holder;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("child never acquired the legacy lock");
    }

    #[test]
    fn shared_legacy_lock_blocks_removal_until_the_reader_leaves() {
        let temp = Temp::new();
        let dir = mapped(&temp.0, &"a".repeat(64), true);
        let holder = shared_lock_holder(&dir);
        sweep_old_packs(&temp.0);
        assert!(dir.exists(), "a legacy reader's shared lock was ignored");
        drop(holder);
        sweep_old_packs(&temp.0);
        assert!(!dir.exists());
    }

    #[test]
    fn direct_fixture_load_creates_no_pack_root() {
        let temp = Temp::new();
        let root = temp.0.join("fresh-cache");
        let model =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp/model.gguf");
        let _store = super::super::weights::Store::open_fixture(&model, Some(&root)).unwrap();
        assert!(!root.exists(), "a direct load created a weight-pack root");
    }
}
