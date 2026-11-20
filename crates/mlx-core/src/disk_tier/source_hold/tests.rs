use std::cell::Cell;
use std::sync::mpsc;
use std::time::Duration;

use mlx_paged_attn::{
    ColdCacheFingerprint, ColdCacheKey, ColdGroup, ColdStateLabel, ColdStateLayout,
    ColdStateObject, Landed, MakeRoom,
};

use super::*;
use crate::disk_tier::{GIB, LOCK_FILE, VOLUMES, Volume, open_disk_tier};

const OWN: &str = "0123456789abcdef0123456789abcdef";
const SOURCE: &str = "55555555555555555555555555555555";
const OTHER: &str = "cccccccccccccccccccccccccccccccc";
const NEXT: &str = "dddddddddddddddddddddddddddddddd";
const DAY: Duration = Duration::from_secs(24 * 60 * 60);

/// A fresh directory for one test, removed on drop.
struct Root(PathBuf);

impl Root {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "mlx-source-hold-test-{tag}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(path.join("tmp")).unwrap();
        fs::create_dir_all(path.join("cache")).unwrap();
        VOLUMES.with(|volumes| {
            volumes.borrow_mut().push((
                path.clone(),
                Some(Volume {
                    available: 2048 * GIB,
                    size: 4096 * GIB,
                    local: true,
                    mount: "/Volumes/Test".to_string(),
                }),
            ))
        });
        Self(path)
    }

    fn cache(&self) -> PathBuf {
        self.0.join("cache")
    }

    fn options(&self, persistent: bool, limit: u64) -> DiskTierOptions {
        DiskTierOptions {
            cache_disk: CacheDisk::Bytes(limit),
            persistent,
            cache_dir: Some(self.cache()),
            temp_root: Some(self.0.join("tmp")),
        }
    }
}

impl Drop for Root {
    fn drop(&mut self) {
        VOLUMES.with(|volumes| volumes.borrow_mut().clear());
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn identity(namespace: &str) -> Option<PersistentIdentity> {
    let fingerprint = ColdCacheFingerprint::from_components([namespace.as_bytes()]);
    Some(PersistentIdentity {
        namespace: namespace.to_string(),
        state: fingerprint,
        kv: fingerprint,
    })
}

/// A namespace a block-16 load left: its lock and one object of `bytes`,
/// last used `age` ago.
fn plant(cache: &Path, name: &str, bytes: u64, age: Duration) -> PathBuf {
    let dir = cache.join(name);
    fs::create_dir_all(&dir).unwrap();
    let when = SystemTime::now() - age;
    for file in [LOCK_FILE, "object.safetensors"] {
        let handle = File::create(dir.join(file)).unwrap();
        if file != LOCK_FILE {
            handle.set_len(bytes).unwrap();
        }
        handle.set_modified(when).unwrap();
    }
    dir
}

/// One state object of `bytes` through the tier's tracked writer.
fn write_state(tier: &DiskTier, bytes: usize) {
    let tokens: Vec<u32> = (0..16).collect();
    let fingerprint = tier.state_fingerprint;
    let key = ColdCacheKey::chain(
        ColdGroup::DFlash2State,
        fingerprint,
        None,
        &tokens,
        &[],
        0,
        0,
    );
    let object = ColdStateObject {
        key,
        fingerprint,
        layout: ColdStateLayout {
            boundary_tokens: 16,
            spans: vec![bytes as u64 - 64, 64],
            geometry: "gdn 1 3 14 2 4 3\n".to_string(),
        },
        label: ColdStateLabel {
            tokens,
            extra_keys: Vec::new(),
            salt: 0,
            last_used: 1,
            checkpoint: false,
        },
        payload: vec![7u8; bytes],
    };
    let (tx, rx) = mpsc::channel();
    tier.manager
        .enqueue_state(object, MakeRoom::Evict, tx)
        .unwrap();
    assert!(matches!(
        rx.recv_timeout(Duration::from_secs(10)).unwrap(),
        Landed::State { ok: true, .. }
    ));
}

/// A source store past the 14-day rule, in a root whose limit is 1.3x the
/// store: the open holds it before the rule and the measure run, so the
/// rule leaves it, its lock time is now, and neither this tier's measure
/// (own objects + source past the limit) nor another start's measure takes
/// it; once released it is an idle cache again and goes.
/// Mutation no-lock (the open takes no source hold) → red.
#[test]
fn a_held_source_outlives_the_stale_rule_and_every_measure() {
    let root = Root::new("held");
    let store: u64 = 1 << 20;
    let source = plant(&root.cache(), SOURCE, store, 15 * DAY);
    let opened = SystemTime::now();
    let tier = open_disk_tier_with_source(
        &root.options(true, store * 13 / 10),
        || identity(OWN),
        || identity(SOURCE),
        160,
    )
    .expect("a persistent tier");
    assert!(tier.persistent);
    assert!(source.is_dir(), "the 14-day rule leaves a held source");
    assert_eq!(
        tier.with_source(|hold| (
            hold.dir.clone(),
            hold.identity.namespace.clone(),
            hold.bytes
        )),
        Some((source.clone(), SOURCE.to_string(), store))
    );
    let lock_time = fs::metadata(source.join(LOCK_FILE))
        .unwrap()
        .modified()
        .unwrap();
    assert!(lock_time >= opened, "the source lock's time is the open's");

    write_state(&tier, (store / 2) as usize);
    assert!(tier.manager.total_bytes() + store > tier.limit);
    tier.measure();
    assert!(
        source.is_dir(),
        "this tier's measure leaves the held source"
    );

    let other = open_disk_tier(&root.options(true, store / 2), || identity(OTHER), 160)
        .expect("another persistent tier");
    assert!(other.persistent);
    other.measure();
    assert!(source.is_dir(), "another start's measure leaves it");
    other.close(std::time::Instant::now());
    drop(other);

    assert!(tier.release_source());
    assert!(tier.with_source(|_| ()).is_none());
    let next = open_disk_tier(&root.options(true, store / 2), || identity(NEXT), 160);
    assert!(next.is_some_and(|next| next.persistent));
    assert!(!source.exists(), "a released source is an idle cache again");
}

/// A temporary tier runs no source identity and holds nothing; a
/// persistent one holds no source another process holds (it stays) and
/// makes none that is not on disk.
#[test]
fn only_a_persistent_open_holds_a_free_source() {
    let root = Root::new("free");
    let source = plant(&root.cache(), SOURCE, 4096, DAY);
    let ran = Cell::new(false);
    let temporary = open_disk_tier_with_source(
        &root.options(false, 1 << 30),
        || identity(OWN),
        || {
            ran.set(true);
            identity(SOURCE)
        },
        160,
    )
    .expect("a temporary tier");
    assert!(!temporary.persistent);
    assert!(!ran.get(), "a temporary tier computes no source");
    assert!(temporary.with_source(|_| ()).is_none());
    drop(temporary);

    let Taken::Lock(held) = take_lock(&source) else {
        panic!("the source's lock");
    };
    let tier = open_disk_tier_with_source(
        &root.options(true, 1 << 30),
        || identity(OWN),
        || identity(SOURCE),
        160,
    )
    .expect("a persistent tier");
    assert!(tier.persistent);
    assert!(tier.with_source(|_| ()).is_none(), "someone holds it");
    assert!(source.is_dir());
    drop(held);
    tier.close(std::time::Instant::now());
    drop(tier);

    let absent = root.cache().join(OTHER);
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let tier = loop {
        let tier = open_disk_tier_with_source(
            &root.options(true, 1 << 30),
            || identity(OWN),
            || identity(OTHER),
            160,
        )
        .expect("a tier");
        if tier.persistent || std::time::Instant::now() >= deadline {
            break tier;
        }
        drop(tier);
        std::thread::sleep(Duration::from_millis(10));
    };
    assert!(tier.persistent);
    assert!(tier.with_source(|_| ()).is_none());
    assert!(!absent.exists(), "a source not on disk is not made");
}

/// A source another process locks after this open scanned the root and
/// before its hold is not held, and it is not on this tier's list of idle
/// caches either: once that process lets the lock go, this tier's measure
/// (own objects + source past the limit) leaves the source.
#[test]
fn a_source_locked_between_the_scan_and_the_hold_stays() {
    let root = Root::new("raced");
    let store: u64 = 1 << 20;
    let source = plant(&root.cache(), SOURCE, store, DAY);
    let held = std::cell::RefCell::new(None);
    let tier = open_disk_tier_with_source(
        &root.options(true, store * 13 / 10),
        || identity(OWN),
        || {
            let Taken::Lock(lock) = take_lock(&source) else {
                panic!("the source's lock");
            };
            *held.borrow_mut() = Some(lock);
            identity(SOURCE)
        },
        160,
    )
    .expect("a persistent tier");
    assert!(tier.persistent);
    assert!(tier.with_source(|_| ()).is_none(), "someone holds it");
    assert!(
        held.borrow_mut().take().is_some(),
        "the other holder lets it go"
    );

    write_state(&tier, (store / 2) as usize);
    assert!(tier.manager.total_bytes() + store > tier.limit);
    tier.measure();
    assert!(source.is_dir(), "this tier's measure leaves the source");
}

/// A source directory with no lock file is not held and nobody holds it: it
/// stays an idle cache, so this tier's measure removes it past the limit.
#[test]
fn a_source_with_no_lock_stays_an_idle_cache() {
    let root = Root::new("lockless");
    let store: u64 = 1 << 20;
    let source = plant(&root.cache(), SOURCE, store, DAY);
    std::fs::remove_file(source.join(LOCK_FILE)).expect("the lock file");
    let tier = open_disk_tier_with_source(
        &root.options(true, store * 13 / 10),
        || identity(OWN),
        || identity(SOURCE),
        160,
    )
    .expect("a persistent tier");
    assert!(tier.persistent);
    assert!(tier.with_source(|_| ()).is_none(), "no lock, no hold");

    write_state(&tier, (store / 2) as usize);
    assert!(tier.manager.total_bytes() + store > tier.limit);
    tier.measure();
    assert!(!source.is_dir(), "an idle cache past the limit goes");
}
