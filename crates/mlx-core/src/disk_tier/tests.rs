use std::sync::mpsc;

use mlx_paged_attn::{
    ColdCacheKey, ColdGroup, ColdStateLabel, ColdStateLayout, ColdStateObject, EnqueueRefused,
    Landed, MakeRoom,
};

use super::*;

const STATE_27B: u64 = 159_842_304;

/// A fresh directory for one test, removed by [`Root::drop`].
struct Root(PathBuf);

impl Root {
    fn new(tag: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("mlx-disk-tier-test-{tag}-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Root {
    fn drop(&mut self) {
        VOLUMES.with(|volumes| volumes.borrow_mut().clear());
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn volume(size_gib: u64, available_gib: u64) -> Volume {
    Volume {
        available: available_gib * GIB,
        size: size_gib * GIB,
        local: true,
        mount: "/Volumes/Test".to_string(),
    }
}

fn set_volume(root: &Path, volume: Option<Volume>) {
    VOLUMES.with(|volumes| volumes.borrow_mut().push((root.to_path_buf(), volume)));
}

fn temp_options(root: &Root) -> DiskTierOptions {
    DiskTierOptions {
        temp_root: Some(root.join("tmp")),
        ..DiskTierOptions::default()
    }
}

fn persistent_options(root: &Root, cache_disk: CacheDisk) -> DiskTierOptions {
    DiskTierOptions {
        cache_disk,
        persistent: true,
        cache_dir: Some(root.join("cache")),
        temp_root: Some(root.join("tmp")),
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

const NS: &str = "0123456789abcdef0123456789abcdef";
const NS_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const NS_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

/// One idle namespace of `bytes` (a sparse object) last used `age` ago.
fn plant_namespace(cache: &Path, name: &str, bytes: u64, age: Duration) -> PathBuf {
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

fn tiny_state(fingerprint: ColdCacheFingerprint) -> ColdStateObject {
    let tokens: Vec<u32> = (0..16).collect();
    let key = ColdCacheKey::chain(
        ColdGroup::DFlash2State,
        fingerprint,
        None,
        &tokens,
        &[],
        0,
        0,
    );
    ColdStateObject {
        key,
        fingerprint,
        layout: ColdStateLayout {
            boundary_tokens: 16,
            spans: vec![96, 64],
            geometry: "gdn 1 3 14 2 4 3\n".to_string(),
        },
        label: ColdStateLabel {
            tokens,
            salt: 0,
            last_used: 1,
            checkpoint: false,
        },
        payload: vec![7u8; 160],
    }
}

/// Splash `DiskSpace`: the floor is min(10 GiB, volume / 20); an automatic
/// tier keeps a quarter of the room up to 32 GiB and stays off below two
/// states and 1 GiB; a size takes all of the room.
/// Mutation DQ1 (the automatic limit takes all of the room) → red.
#[test]
fn auto_quota_follows_the_volume() {
    let auto = |volume: &Volume| tier_limit(CacheDisk::Auto, Some(volume), 0);
    let minimum = 2 * STATE_27B + GIB;

    let big = volume(1024, 200);
    assert_eq!(disk_floor_bytes(big.size), 10 * GIB);
    assert_eq!(auto(&big), 32 * GIB);
    assert_eq!(automatic_disk_refusal(Some(&big), 0, minimum), None);

    let mid = volume(256, 50);
    assert_eq!(disk_floor_bytes(mid.size), 10 * GIB);
    assert_eq!(auto(&mid), 10 * GIB);

    let small = volume(128, 20);
    let floor = disk_floor_bytes(small.size);
    assert_eq!(floor, 128 * GIB / 20);
    assert_eq!(auto(&small), (20 * GIB - floor) / 4);
    assert_eq!(gibibytes(auto(&small)), "3.4 GiB");

    let full = volume(128, 7);
    assert!(
        automatic_disk_refusal(Some(&full), 0, minimum)
            .is_some_and(|reason| reason.contains("is too little beyond the 6.4 GiB"))
    );
    let remote = Volume {
        local: false,
        ..volume(1024, 200)
    };
    assert_eq!(
        automatic_disk_refusal(Some(&remote), 0, minimum).as_deref(),
        Some("/Volumes/Test is not a local volume")
    );
    assert!(automatic_disk_refusal(None, 0, minimum).is_some());

    assert_eq!(
        tier_limit(CacheDisk::Bytes(64 * GIB), Some(&volume(256, 50)), 0),
        40 * GIB
    );
}

/// Splash `persistentCacheNamespace`: the text names the target, the draft,
/// the KV, GDN, draft and context geometry; its digest's first 32 hex
/// digits name the directory.
/// Mutation NS1 (no `context` line) → red (a width change keeps the digest).
#[test]
fn namespace_names_the_draft_and_the_window() {
    let target = ColdCacheFingerprint::from_components([b"target".as_slice()]);
    let draft = ColdCacheFingerprint::from_components([b"draft".as_slice()]);
    let ns = CacheNamespace {
        target,
        draft,
        kv: (16, 16, 4, 256, "Int8"),
        gdn: [48, 3, 10240, 48, 128, 128],
        draft_geometry: (5, 8, 128, 2048),
        context_width: 5120,
    };
    assert_eq!(
        cache_namespace_text(&ns),
        format!(
            "mlx-node-persistent-cache-v1\n{}\n{}\nkv 16 16 4 256 Int8\ngdn 48 3 10240 48 128 \
             128\ndraft 5 8 128 2048\ncontext 5120 q4 64\n",
            target.to_hex(),
            draft.to_hex()
        )
    );
    let (name, state) = cache_namespace(&ns);
    assert_eq!(name.len(), 32);
    assert!(namespace_name(&name));

    let other_draft = CacheNamespace {
        draft: target,
        ..ns
    };
    assert_ne!(cache_namespace(&other_draft).0, name);
    let other_window = CacheNamespace {
        draft_geometry: (5, 8, 128, 4096),
        ..ns
    };
    assert_ne!(cache_namespace(&other_window).0, name);
    let other_width = CacheNamespace {
        context_width: 4096,
        ..ns
    };
    let (width_name, width_state) = cache_namespace(&other_width);
    assert_ne!(width_name, name);
    assert_ne!(width_state, state);
}

/// A temporary tier is a locked, published directory; its close removes it
/// with the tier still referenced and refuses every later write. A sweep
/// takes only a published directory whose lock nobody holds.
/// Mutation TT1 (`close` keeps the directory) → red.
#[test]
fn temp_tier_leaves_nothing_behind() {
    let root = Root::new("temp");
    let tmp = root.join("tmp");
    fs::create_dir_all(&tmp).unwrap();
    let options = temp_options(&root);
    let tier = open_disk_tier(&options, || None, 160).expect("temporary tier");
    assert!(!tier.persistent);
    let dir = tier.dir();
    let name = dir.file_name().unwrap().to_string_lossy().into_owned();
    assert!(name.starts_with(TEMP_PREFIX), "{name}");
    assert!(matches!(take_lock(&dir), Taken::Held));
    let names: Vec<String> = fs::read_dir(&tmp)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, vec![name.clone()], "no dot-named directory is left");

    let (tx, rx) = mpsc::channel();
    let object = tiny_state(tier.state_fingerprint);
    let key = object.key;
    tier.manager
        .enqueue_state(object, MakeRoom::Evict, tx.clone())
        .expect("state lane");
    assert!(matches!(rx.recv().unwrap(), Landed::State { ok: true, .. }));
    assert!(tier.manager.contains_in(&key, ColdGroup::DFlash2State));

    let held = Arc::clone(&tier);
    assert!(held.close(Instant::now() + Duration::from_secs(5)));
    assert!(!dir.exists(), "close removes the temporary directory");
    let refused =
        tier.manager
            .enqueue_state(tiny_state(tier.state_fingerprint), MakeRoom::Evict, tx);
    assert!(matches!(refused, Err(EnqueueRefused::Gone(_))));
    drop(held);
    drop(tier);

    let plant = |name: &str| {
        let dir = tmp.join(name);
        fs::create_dir_all(&dir).unwrap();
        File::create(dir.join(LOCK_FILE)).unwrap();
        dir
    };
    let crashed = plant("mlx-node-cache-crashed");
    let live = plant("mlx-node-cache-live");
    let live_lock = File::open(live.join(LOCK_FILE)).unwrap();
    assert!(try_lock(&live_lock));
    let creating = plant(".mlx-node-cache-new-x");
    assert_eq!(sweep_temp_leftovers(&tmp), 1);
    assert!(!crashed.exists());
    assert!(live.exists(), "a held lock keeps its directory");
    assert!(
        creating.exists(),
        "a directory still being made is never swept"
    );

    let crashed = plant("mlx-node-cache-again");
    let tier = open_disk_tier(&options, || None, 160).expect("temporary tier");
    assert!(!crashed.exists(), "the next open sweeps");
    assert!(live.exists());
    drop(tier);
    drop(live_lock);
}

/// Splash `openDiskTier` / `CacheDirectory`: a held namespace keeps a
/// temporary tier; idle namespaces count as held room, the stale ones go,
/// and the oldest go first while the root is over its limit; a refused or
/// unmakeable root keeps a temporary tier.
/// Mutations PL1 (the lock is ignored) and HD1 (`held` counts the tier's
/// own namespace only) → red.
#[test]
fn held_namespace_falls_back_to_temp() {
    let root = Root::new("held");
    fs::create_dir_all(root.join("tmp")).unwrap();
    let cache = root.join("cache");
    fs::create_dir_all(&cache).unwrap();
    set_volume(&root.0, Some(volume(4096, 2048)));

    let options = persistent_options(&root, CacheDisk::Bytes(1 << 40));
    let first = open_disk_tier(&options, || identity(NS), 160).expect("persistent tier");
    assert!(first.persistent);
    assert_eq!(first.dir(), cache.join(NS));
    let second = open_disk_tier(&options, || identity(NS), 160).expect("a tier");
    assert!(
        !second.persistent,
        "a held namespace keeps a temporary tier"
    );
    assert_ne!(second.dir(), first.dir());
    drop(second);
    drop(first);

    let day = Duration::from_secs(24 * 60 * 60);
    let stale = plant_namespace(&cache, NS_A, GIB, 15 * day);
    let fresh = plant_namespace(&cache, NS_B, 5 * GIB, day);
    let tier = open_disk_tier(&options, || identity(NS), 160).expect("persistent tier");
    assert!(tier.persistent);
    assert!(!stale.exists(), "a namespace unused for 14 days goes");
    assert!(fresh.exists());
    drop(tier);

    let older = plant_namespace(&cache, NS_A, 5 * GIB, 2 * day);
    let tier = open_disk_tier(&options, || identity(NS), 160).expect("persistent tier");
    assert_eq!(tier.held, 10 * GIB, "both idle siblings count");
    drop(tier);

    let tight = persistent_options(&root, CacheDisk::Bytes(6 * GIB));
    let tier = open_disk_tier(&tight, || identity(NS), 160).expect("persistent tier");
    assert!(!older.exists(), "the oldest sibling goes first");
    assert!(fresh.exists());
    drop(tier);

    set_volume(
        &cache,
        Some(Volume {
            local: false,
            ..volume(4096, 2048)
        }),
    );
    VOLUMES.with(|volumes| volumes.borrow_mut().reverse());
    let auto = persistent_options(&root, CacheDisk::Auto);
    let tier = open_disk_tier(&auto, || identity(NS), 160).expect("a tier");
    assert!(!tier.persistent, "a refused root keeps a temporary tier");
    drop(tier);

    let blocked = root.join("file");
    File::create(&blocked).unwrap();
    let unmakeable = DiskTierOptions {
        cache_dir: Some(blocked.join("cache")),
        ..persistent_options(&root, CacheDisk::Bytes(1 << 40))
    };
    let tier = open_disk_tier(&unmakeable, || identity(NS), 160).expect("a tier");
    assert!(
        !tier.persistent,
        "an unmakeable root keeps a temporary tier"
    );
}

/// Splash `serve_options.py`: the default is the automatic temporary tier;
/// `--cache-dir` needs `--persistent-cache`, which needs the tier on.
/// Mutation DF1 (`persistent` defaults to true) → red.
#[test]
fn default_options_open_a_temp_tier() {
    let default = DiskTierOptions::default();
    assert_eq!(default.cache_disk, CacheDisk::Auto);
    assert!(!default.persistent);
    assert_eq!(
        DiskTierOptions::from_load(None, None, None).unwrap(),
        default
    );
    assert_eq!(
        DiskTierOptions::from_load(None, None, Some("/x".to_string())).unwrap_err(),
        "--cache-dir needs --persistent-cache"
    );
    assert_eq!(
        DiskTierOptions::from_load(Some("0"), Some(true), None).unwrap_err(),
        "--persistent-cache needs the SSD cache --max-cache-disk 0 turns off"
    );
    assert_eq!(CacheDisk::parse("0"), Ok(CacheDisk::Off));
    assert_eq!(
        CacheDisk::parse("68719476736"),
        Ok(CacheDisk::Bytes(64 * GIB))
    );
    assert!(CacheDisk::parse("64G").is_err());

    let root = Root::new("default");
    fs::create_dir_all(root.join("tmp")).unwrap();
    let off = DiskTierOptions {
        cache_disk: CacheDisk::Off,
        ..temp_options(&root)
    };
    assert!(open_disk_tier(&off, || identity(NS), 160).is_none());
    set_volume(&root.0, Some(volume(1024, 200)));
    let options = DiskTierOptions {
        temp_root: Some(root.join("tmp")),
        ..DiskTierOptions::from_load(None, None, None).unwrap()
    };
    let tier = open_disk_tier(&options, || identity(NS), 160).expect("temporary tier");
    assert!(!tier.persistent);
    assert_eq!(tier.limit, 32 * GIB);
}

/// A temporary tier hashes no weights: its identity is random, the load
/// guard stays off, and only a persistent open reads the checkpoint's
/// identity, once.
/// Mutation TI1 (a temporary tier builds the namespace) → red.
#[test]
fn temp_tier_hashes_no_weights() {
    use std::sync::atomic::Ordering;
    let root = Root::new("identity");
    fs::create_dir_all(root.join("tmp")).unwrap();
    let checkpoint = root.join("checkpoint");
    fs::create_dir_all(&checkpoint).unwrap();
    fs::write(checkpoint.join("model.safetensors"), [0u8; 64]).unwrap();
    set_volume(&root.0, Some(volume(1024, 200)));
    let read_identity = || {
        let mut components = Vec::new();
        crate::cold_tier::push_dir_identity(&checkpoint, &mut components)?;
        identity(NS)
    };
    let reads = || crate::cold_tier::IDENTITY_READS.load(Ordering::Relaxed);

    let options = temp_options(&root);
    assert!(!options.reads_identity(), "the load guard stays off");
    let before = reads();
    let one = open_disk_tier(&options, read_identity, 160).expect("temporary tier");
    let two = open_disk_tier(&options, read_identity, 160).expect("temporary tier");
    assert_eq!(reads(), before, "a temporary tier reads no identity");
    assert_ne!(one.state_fingerprint, two.state_fingerprint);
    assert_eq!(one.state_fingerprint, one.kv_fingerprint);
    drop(one);
    drop(two);

    let persistent = persistent_options(&root, CacheDisk::Bytes(1 << 40));
    assert!(persistent.reads_identity());
    let tier = open_disk_tier(&persistent, read_identity, 160).expect("persistent tier");
    assert!(tier.persistent);
    assert_eq!(reads(), before + 1);
}

/// Splash `openDiskTier` runs `CacheDirectory::removeStale` on every start:
/// a temporary tier and a tier turned off remove a namespace nobody used
/// for 14 days from the cache root as a persistent open does, and keep a
/// fresh one.
/// Mutation SC1 (only a persistent open removes stale namespaces) → red.
#[test]
fn every_start_removes_stale_namespaces() {
    let root = Root::new("stale");
    fs::create_dir_all(root.join("tmp")).unwrap();
    let cache = root.join("cache");
    fs::create_dir_all(&cache).unwrap();
    set_volume(&root.0, Some(volume(4096, 2048)));
    let day = Duration::from_secs(24 * 60 * 60);
    let temp = DiskTierOptions {
        cache_dir: Some(cache.clone()),
        ..temp_options(&root)
    };
    let stale = plant_namespace(&cache, NS_A, GIB, 15 * day);
    let fresh = plant_namespace(&cache, NS_B, GIB, day);
    let tier = open_disk_tier(&temp, || identity(NS), 160).expect("temporary tier");
    assert!(!tier.persistent);
    assert!(
        !stale.exists(),
        "a temporary start removes a stale namespace"
    );
    assert!(fresh.exists());
    drop(tier);

    let stale = plant_namespace(&cache, NS_A, GIB, 15 * day);
    let off = DiskTierOptions {
        cache_disk: CacheDisk::Off,
        ..temp.clone()
    };
    assert!(open_disk_tier(&off, || identity(NS), 160).is_none());
    assert!(!stale.exists(), "a start with the tier off removes it too");
    assert!(fresh.exists());
}

/// Splash `named`: a lock gives its namespace only while the namespace's
/// path still names the locked file. Once a removal took the directory and
/// another open made it again, the old lock names nothing: neither a
/// waiting open nor a removal may act on it.
/// Mutation NL1 (`named` always holds) → red.
#[test]
fn a_lock_whose_file_was_replaced_names_nothing() {
    let root = Root::new("named");
    let dir = root.join(NS);
    fs::create_dir_all(&dir).unwrap();
    let held = File::create(dir.join(LOCK_FILE)).unwrap();
    assert!(named(&dir, &held));
    fs::remove_dir_all(&dir).unwrap();
    assert!(!named(&dir, &held), "the lock's file is gone");
    fs::create_dir_all(&dir).unwrap();
    File::create(dir.join(LOCK_FILE)).unwrap();
    assert!(!named(&dir, &held), "the path names another lock now");
}

/// Splash `DiskSpace::measure`: the root's idle caches counted at open
/// keep counting, and as the tier grows past the root's limit with them
/// the oldest goes; one whose lock someone took meanwhile is that
/// process's and stays.
/// Mutation Q1a (`measure` does nothing) → red.
#[test]
fn a_growing_tier_removes_the_oldest_idle_sibling() {
    let root = Root::new("measure");
    fs::create_dir_all(root.join("tmp")).unwrap();
    let cache = root.join("cache");
    fs::create_dir_all(&cache).unwrap();
    set_volume(&root.0, Some(volume(4096, 2048)));
    let day = Duration::from_secs(24 * 60 * 60);
    let older = plant_namespace(&cache, NS_A, 600 << 10, 2 * day);
    let newer = plant_namespace(&cache, NS_B, 100 << 10, day);
    let options = persistent_options(&root, CacheDisk::Bytes(1 << 20));
    let tier = open_disk_tier(&options, || identity(NS), 160).expect("persistent tier");
    assert!(tier.persistent);
    assert_eq!(tier.limit, 1 << 20);
    assert!(older.exists() && newer.exists(), "the root fits at open");

    let write = |payload: usize, salt: u32| {
        let mut object = tiny_state(tier.state_fingerprint);
        object.label.tokens = (salt..salt + 16).collect();
        object.key = ColdCacheKey::chain(
            ColdGroup::DFlash2State,
            tier.state_fingerprint,
            None,
            &object.label.tokens,
            &[],
            0,
            0,
        );
        object.layout.spans = vec![payload as u64 - 64, 64];
        object.payload = vec![7u8; payload];
        let (tx, rx) = mpsc::channel();
        tier.manager
            .enqueue_state(object, MakeRoom::Evict, tx)
            .unwrap();
        assert!(matches!(
            rx.recv_timeout(Duration::from_secs(10)).unwrap(),
            Landed::State { ok: true, .. }
        ));
    };
    write(600 << 10, 0);
    tier.measure();
    assert!(!older.exists(), "the oldest idle sibling went");
    assert!(newer.exists());
    assert!(tier.manager.total_bytes() + (100 << 10) <= tier.limit);

    let Taken::Lock(held) = take_lock(&newer) else {
        panic!("the sibling's lock");
    };
    write(340 << 10, 100);
    tier.measure();
    assert!(newer.exists(), "a sibling someone holds stays");
    drop(held);
}

/// Splash `~SlotFile` joins its worker before `~CacheDirectory` closes the
/// lock: a closed persistent tier lets its namespace go once its writer
/// exits, so the next open takes it.
#[test]
fn persistent_close_releases_the_namespace_after_the_writer() {
    let root = Root::new("close-lock");
    fs::create_dir_all(root.join("tmp")).unwrap();
    set_volume(&root.0, Some(volume(4096, 2048)));
    let options = persistent_options(&root, CacheDisk::Bytes(1 << 40));
    let tier = open_disk_tier(&options, || identity(NS), 160).expect("persistent tier");
    assert!(tier.persistent);
    tier.close(Instant::now());
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match open_persistent_dir(&root.join("cache"), NS) {
            PersistentOpen::Open { .. } => break,
            PersistentOpen::Held(_) => {
                assert!(Instant::now() < deadline, "the namespace stays held");
                std::thread::sleep(Duration::from_millis(5));
            }
            PersistentOpen::Failed(error) => panic!("{error}"),
        }
    }
    drop(tier);
}
