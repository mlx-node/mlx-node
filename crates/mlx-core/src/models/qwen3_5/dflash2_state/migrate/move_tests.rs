use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use mlx_paged_attn::{
    ColdCacheFingerprint, ColdCacheManager, ColdSourceReader, ColdStateLayout, ColdStateObject,
    Landed, MakeRoom,
};

use super::move_source::hooks::{self, Act, Step};
use super::move_source::{MoveShape, Moved, move_source};
use super::*;
use crate::disk_tier::source_hold::open_disk_tier_with_source;
use crate::disk_tier::{
    CacheDisk, DiskTier, DiskTierOptions, TEST_WRITER, VOLUMES, Volume, open_disk_tier,
};
use crate::models::qwen3_5::dflash2_state::disk::adopt_states_on;
use crate::models::qwen3_5::dflash2_state::store::{HybridStateStore, StateBytes};

const LAYERS: u32 = 2;
const HEADS: u32 = 2;
const DIM: u32 = 16;
const SPANS: [u64; 2] = [4000, 96];
const GEOMETRY: &str = "gdn 1 3 14 2 4 3\ndraft 1 2 8 64\ncontext 8 q4 64\n";
const SALT: u64 = 0x5a17;
const OWN: &str = "0123456789abcdef0123456789abcdef";
const SOURCE: &str = "55555555555555555555555555555555";
const GIB: u64 = 1 << 30;
const SECOND: u128 = 1_000_000_000;
const HOUR: u64 = 60 * 60;
const DAY: u64 = 24 * HOUR;
const WAIT: Duration = Duration::from_secs(10);

/// A fresh cache root for one test, removed on drop.
struct Root(PathBuf);

impl Root {
    fn new(tag: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("mlx-move-{tag}-{}-{}", std::process::id(), now()));
        std::fs::create_dir_all(path.join("tmp")).expect("temp root");
        std::fs::create_dir_all(path.join("cache")).expect("cache root");
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
        TEST_WRITER.with(|writer| writer.set(Some((8, 0))));
        hooks::observe(None);
        hooks::take_conversions();
        Self(path)
    }

    fn dir(&self, namespace: &str) -> PathBuf {
        self.0.join("cache").join(namespace)
    }

    fn options(&self, limit: u64) -> DiskTierOptions {
        DiskTierOptions {
            cache_disk: CacheDisk::Bytes(limit),
            persistent: true,
            cache_dir: Some(self.0.join("cache")),
            temp_root: Some(self.0.join("tmp")),
        }
    }
}

impl Drop for Root {
    fn drop(&mut self) {
        hooks::observe(None);
        TEST_WRITER.with(|writer| writer.set(None));
        VOLUMES.with(|volumes| volumes.borrow_mut().clear());
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn now() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn identity(namespace: &str) -> crate::disk_tier::PersistentIdentity {
    let of = |part: &[u8]| ColdCacheFingerprint::from_components([namespace.as_bytes(), part]);
    crate::disk_tier::PersistentIdentity {
        namespace: namespace.to_string(),
        state: of(b" state"),
        kv: of(b" kv"),
    }
}

/// The fingerprints of a move from [`SOURCE`] into [`OWN`].
fn fingerprints() -> Fingerprints {
    let own = identity(OWN);
    Fingerprints {
        source_kv: identity(SOURCE).kv,
        kv: own.kv,
        state: own.state,
    }
}

fn block_layout() -> ColdCacheLayout {
    source_layout(LAYERS, HEADS, DIM).expect("a source layout")
}

fn shape(layout: &ColdCacheLayout) -> MoveShape<'_> {
    MoveShape {
        spans: &SPANS,
        geometry: GEOMETRY,
        source_layout: layout.clone(),
    }
}

/// One stored chat of the source: its rows, its stamp, and the times of
/// its blocks and of its state object.
#[derive(Clone)]
struct Chat {
    tokens: Vec<u32>,
    /// Image keys per source block.
    extra_keys: Vec<Vec<u64>>,
    last_used: u64,
    blocks_at: u128,
    state_at: u128,
    /// Added to the time of every odd block.
    stagger: u128,
}

impl Chat {
    fn new(seed: u32, rows: usize, last_used: u64, age: u64) -> Self {
        Self::extending(&[], seed, rows, last_used, age)
    }

    /// A chat that goes on from `prefix` to `rows` rows.
    fn extending(prefix: &[u32], seed: u32, rows: usize, last_used: u64, age: u64) -> Self {
        let mut tokens = prefix.to_vec();
        tokens.extend((prefix.len()..rows).map(|row| seed * 100_000 + row as u32));
        let at = now() - u128::from(age) * SECOND;
        Self {
            tokens,
            extra_keys: Vec::new(),
            last_used,
            blocks_at: at,
            state_at: at,
            stagger: 0,
        }
    }

    fn lineage(&self) -> Lineage<'_> {
        Lineage {
            tokens: &self.tokens,
            extra_keys: &self.extra_keys,
            salt: SALT,
        }
    }

    fn label(&self) -> ColdStateLabel {
        ColdStateLabel {
            tokens: self.tokens.clone(),
            extra_keys: self.extra_keys.clone(),
            salt: SALT,
            last_used: self.last_used,
            checkpoint: false,
        }
    }

    fn layout(&self) -> ColdStateLayout {
        ColdStateLayout {
            boundary_tokens: self.tokens.len() as u32,
            spans: SPANS.to_vec(),
            geometry: GEOMETRY.to_string(),
        }
    }

    fn payload(&self) -> Vec<u8> {
        let seed = self.tokens.last().copied().unwrap_or(0) as usize + self.tokens.len();
        (0..SPANS.iter().sum::<u64>() as usize)
            .map(|byte| ((byte * 31 + seed) % 251) as u8)
            .collect()
    }

    fn block_time(&self, index: usize) -> u128 {
        self.blocks_at + (index % 2) as u128 * self.stagger
    }

    fn source_keys(&self) -> Vec<ColdCacheKey> {
        lineage_keys(
            ColdGroup::Kv,
            identity(SOURCE).kv,
            self.lineage(),
            SOURCE_BLOCK,
        )
    }

    fn source_state_key(&self) -> ColdCacheKey {
        lineage_keys(
            ColdGroup::DFlash2State,
            identity(SOURCE).state,
            self.lineage(),
            SOURCE_BLOCK,
        )
        .last()
        .copied()
        .expect("a state key")
    }
}

/// The block the source holds under `key`: bytes the key alone decides.
fn source_block(key: ColdCacheKey, tokens: &[u32], layout: &ColdCacheLayout) -> ColdCacheBlock {
    let seed = *key.as_bytes();
    let bytes = |len: usize, salt: usize| -> Vec<u8> {
        (0..len)
            .map(|at| seed[(at * 7 + salt) % 32] ^ (at as u8))
            .collect()
    };
    ColdCacheBlock {
        key,
        fingerprint: identity(SOURCE).kv,
        tokens: tokens.to_vec(),
        layers: (0..layout.num_layers as usize)
            .map(|layer| ColdLayerBlock {
                keys: bytes(layout.key_bytes_per_layer, layer),
                values: bytes(layout.value_bytes_per_layer, layer + 13),
            })
            .collect(),
        layout: layout.clone(),
    }
}

fn land_block(manager: &ColdCacheManager, block: ColdCacheBlock, room: MakeRoom) -> bool {
    let (done, landed) = mpsc::channel();
    assert_eq!(
        manager.enqueue_tracked(block, room, Instant::now() + WAIT, done),
        Ok(true)
    );
    matches!(
        landed.recv_timeout(WAIT),
        Ok(Landed::Block { ok: true, .. })
    )
}

fn land_state(manager: &ColdCacheManager, object: ColdStateObject) {
    let (done, landed) = mpsc::channel();
    assert!(manager.enqueue_state(object, MakeRoom::Evict, done).is_ok());
    assert!(matches!(
        landed.recv_timeout(WAIT),
        Ok(Landed::State { ok: true, .. })
    ));
}

fn close(tier: Arc<DiskTier>) {
    tier.close(Instant::now() + WAIT);
}

/// A persistent tier of `namespace` with no source; a namespace whose
/// earlier writer still holds its lock is opened again until it lets go.
fn open_plain(root: &Root, namespace: &str, limit: u64) -> Arc<DiskTier> {
    let deadline = Instant::now() + WAIT;
    loop {
        let tier = open_disk_tier(&root.options(limit), || Some(identity(namespace)), 160)
            .expect("a tier");
        if tier.persistent || Instant::now() >= deadline {
            assert!(tier.persistent, "the namespace's lock is free in time");
            return tier;
        }
        close(tier);
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The load's own tier, opened as a load that moves [`SOURCE`] opens it.
/// With a source directory on disk it is opened again until it holds the
/// source too, for [`WAIT`] at most.
fn open_own(root: &Root, limit: u64) -> Arc<DiskTier> {
    let deadline = Instant::now() + WAIT;
    loop {
        let wants_source = root.dir(SOURCE).is_dir();
        let tier = open_disk_tier_with_source(
            &root.options(limit),
            || Some(identity(OWN)),
            || Some(identity(SOURCE)),
            160,
        )
        .expect("a tier");
        let held = tier.with_source(|_| ()).is_some();
        if (tier.persistent && (held || !wants_source)) || Instant::now() >= deadline {
            assert!(tier.persistent, "the own namespace's lock is free in time");
            return tier;
        }
        close(tier);
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Writes `chats` as the block-16 load left them: every block of every
/// chain once, each chat's state object, with the chats' times (a block
/// two chats share has the newer chat's).
fn write_source(root: &Root, chats: &[Chat], layout: &ColdCacheLayout) {
    let tier = open_plain(root, SOURCE, GIB);
    let manager = &tier.manager;
    let source = identity(SOURCE);
    let mut order: Vec<&Chat> = chats.iter().collect();
    order.sort_by_key(|chat| chat.blocks_at);
    for chat in order {
        for (index, key) in chat.source_keys().into_iter().enumerate() {
            if !manager.contains(&key) {
                let rows = &chat.tokens[index * 16..(index + 1) * 16];
                assert!(land_block(
                    manager,
                    source_block(key, rows, layout),
                    MakeRoom::Evict
                ));
            }
            assert!(manager.set_modified(&key, chat.block_time(index)));
        }
        let key = chat.source_state_key();
        land_state(
            manager,
            ColdStateObject {
                key,
                fingerprint: source.state,
                layout: chat.layout(),
                label: chat.label(),
                payload: chat.payload(),
            },
        );
        assert!(manager.set_modified(&key, chat.state_at));
    }
    close(tier);
}

/// What a chat is in the own store: its plan and the objects the converter
/// makes of its source blocks.
struct Expected {
    plan: StatePlan,
    pages: Vec<ColdCacheBlock>,
    fragment: Option<ColdCacheBlock>,
    label: ColdStateLabel,
}

fn expected(chat: &Chat, layout: &ColdCacheLayout) -> Expected {
    let fingerprints = fingerprints();
    let label = chat.label();
    let plan = plan_state(&label, &fingerprints).expect("a plan");
    let keys = chat.source_keys();
    let block = |key: ColdCacheKey| {
        let index = keys
            .iter()
            .position(|at| *at == key)
            .expect("a block of the chat");
        source_block(key, &chat.tokens[index * 16..(index + 1) * 16], layout)
    };
    let pages = plan
        .pages
        .iter()
        .map(|page| {
            convert_page(
                page,
                &chat.tokens,
                &block(page.sources[0]),
                &block(page.sources[1]),
                &fingerprints,
            )
            .expect("a page")
        })
        .collect();
    let fragment = plan.fragment.as_ref().map(|fragment| {
        convert_fragment(
            fragment,
            &chat.tokens,
            &block(fragment.source),
            &fingerprints,
        )
        .expect("a fragment")
    });
    Expected {
        label: ColdStateLabel {
            extra_keys: plan.extra_keys.clone(),
            ..label
        },
        plan,
        pages,
        fragment,
    }
}

/// `chat` is in the own store whole: every page and its fragment decode to
/// the converter's bytes and its state object holds its payload.
fn restores(tier: &DiskTier, chat: &Chat, layout: &ColdCacheLayout) -> bool {
    let expected = expected(chat, layout);
    let fingerprints = fingerprints();
    let manager = &tier.manager;
    expected
        .pages
        .iter()
        .chain(expected.fragment.as_ref())
        .all(|block| {
            manager
                .read_back(block.key, fingerprints.kv, &block.layout)
                .as_ref()
                == Some(block)
        })
        && manager.state_reads_back(
            expected.plan.state_key,
            fingerprints.state,
            &chat.layout(),
            &expected.label,
            Some(&chat.payload()),
        )
}

/// Every row and the state of `chat` still decode from one of the two
/// directories: a page from the own store, or both its halves from the
/// source; its fragment, or its tail block; its state object from either.
fn recoverable(root: &Root, chat: &Chat, layout: &ColdCacheLayout) -> bool {
    let expected = expected(chat, layout);
    let fingerprints = fingerprints();
    let own = ColdSourceReader::open(&root.dir(OWN)).ok();
    let source = ColdSourceReader::open(&root.dir(SOURCE)).ok();
    let in_own = |block: &ColdCacheBlock| {
        own.as_ref().is_some_and(|own| {
            own.read_block(block.key, fingerprints.kv, &block.layout)
                .as_ref()
                == Some(block)
        })
    };
    let in_source = |key: &ColdCacheKey| {
        source.as_ref().is_some_and(|source| {
            source
                .read_block(*key, fingerprints.source_kv, layout)
                .is_some()
        })
    };
    let pages = expected
        .pages
        .iter()
        .zip(&expected.plan.pages)
        .all(|(block, page)| in_own(block) || page.sources.iter().all(in_source));
    let fragment = expected
        .fragment
        .iter()
        .zip(&expected.plan.fragment)
        .all(|(block, fragment)| in_own(block) || in_source(&fragment.source));
    let payload = chat.payload();
    let mut read = Vec::new();
    let state_in_own = own.as_ref().is_some_and(|own| {
        own.read_state(expected.plan.state_key, fingerprints.state, &mut read)
            == Some((chat.layout(), expected.label.clone()))
            && read == payload
    });
    let state_in_source = source.as_ref().is_some_and(|source| {
        source.read_state(chat.source_state_key(), identity(SOURCE).state, &mut read)
            == Some((chat.layout(), chat.label()))
            && read == payload
    });
    pages && fragment && (state_in_own || state_in_source)
}

struct NoRam;

impl StateBytes for NoRam {
    fn state_bytes(&self) -> u64 {
        0
    }
}

/// The rows and stamps of the states a load of the own store takes back,
/// in row order.
fn adopted(tier: &DiskTier) -> Vec<(usize, u64)> {
    let mut store = HybridStateStore::<NoRam>::new(PAGE_BLOCK);
    let count = adopt_states_on(&mut store, tier, &SPANS, GEOMETRY, true, |_| Some((160, 0)));
    let mut rows: Vec<(usize, u64)> = store
        .entries()
        .map(|entry| (entry.depth(), entry.last_used()))
        .collect();
    assert_eq!(rows.len() as u64, count);
    rows.sort_unstable();
    rows
}

fn rows_of(chats: &[&Chat]) -> Vec<(usize, u64)> {
    let mut rows: Vec<(usize, u64)> = chats
        .iter()
        .map(|chat| (chat.tokens.len(), chat.last_used))
        .collect();
    rows.sort_unstable();
    rows
}

/// Bytes of the files of `dir`, its lock aside; 0 for no directory.
fn bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map_or(0, |entries| {
        entries
            .flatten()
            .filter(|entry| entry.file_name() != "lock")
            .filter_map(|entry| entry.metadata().ok())
            .map(|meta| meta.len())
            .sum()
    })
}

/// Name, size and time of every file of `dir`, its lock aside.
fn listing(dir: &Path) -> Vec<(String, u64, SystemTime)> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .expect("list")
        .map(|entry| entry.expect("an entry"))
        .filter(|entry| entry.file_name() != "lock")
        .map(|entry| {
            let meta = entry.metadata().expect("stat");
            (
                entry.file_name().to_string_lossy().into_owned(),
                meta.len(),
                meta.modified().expect("mtime"),
            )
        })
        .collect();
    entries.sort();
    entries
}

/// Watches the runs of this thread: every step is logged; step `cut_at`
/// ends the run as a killed process does; write `act.0` gets `act.1`.
fn watch(cut_at: Option<usize>, act: Option<(usize, Act)>) -> Rc<RefCell<Vec<Step>>> {
    let log = Rc::new(RefCell::new(Vec::new()));
    let seen = Rc::clone(&log);
    let mut writes = 0;
    hooks::observe(Some(Box::new(move |step| {
        let mut seen = seen.borrow_mut();
        let index = seen.len();
        seen.push(step);
        if cut_at == Some(index) {
            return Act::Cut;
        }
        if matches!(step, Step::Write(_)) {
            writes += 1;
            if let Some((at, act)) = act
                && at + 1 == writes
            {
                return act;
            }
        }
        Act::Go
    })));
    log
}

fn writes_in(log: &[Step]) -> usize {
    log.iter()
        .filter(|step| matches!(step, Step::Write(_)))
        .count()
}

/// Every source object a run took out went after every own object that
/// holds its rows read back and a device flush followed; no state's tail
/// block went at all.
fn assert_order(log: &[Step], chats: &[Chat], case: &str) {
    let fingerprints = fingerprints();
    let mut holders: HashMap<ColdCacheKey, Vec<ColdCacheKey>> = HashMap::new();
    let labels: Vec<ColdStateLabel> = chats.iter().map(Chat::label).collect();
    for (chat, label) in chats.iter().zip(&labels) {
        let plan = plan_state(label, &fingerprints).expect("a plan");
        for page in &plan.pages {
            for source in page.sources {
                let pages = holders.entry(source).or_default();
                if !pages.contains(&page.key) {
                    pages.push(page.key);
                }
            }
        }
        holders
            .entry(chat.source_state_key())
            .or_default()
            .push(plan.state_key);
    }
    let tails = tail_keys(&labels, fingerprints.source_kv);
    for (at, step) in log.iter().enumerate() {
        let Step::Unlink(key) = step else {
            continue;
        };
        assert!(
            !tails.contains(key),
            "{case}: a tail block leaves only with the source directory"
        );
        let last_read = holders
            .get(key)
            .expect("a source object of a chat")
            .iter()
            .map(|holder| {
                log[..at]
                    .iter()
                    .rposition(|earlier| *earlier == Step::ReadBack(*holder))
            })
            .collect::<Option<Vec<usize>>>()
            .and_then(|reads| reads.into_iter().max());
        assert!(
            last_read.is_some_and(|read| log[read..at].contains(&Step::Flush)),
            "{case}: a source object goes after all its rows read back and reached the device"
        );
    }
}

/// X (48 rows), Y (96 rows, X's chat gone on) and Z (64 rows, parted from
/// X after 16 rows): 9 blocks and 3 state objects, 12 objects. X's last
/// block is the first half of a page of Y; X's first block is the first
/// half of a page of Y and of another page of Z.
fn twelve_objects() -> Vec<Chat> {
    let x = Chat::new(1, 48, 1, 3 * DAY);
    let y = Chat::extending(&x.tokens, 2, 96, 3, DAY);
    let z = Chat::extending(&x.tokens[..16], 3, 64, 2, 2 * DAY);
    vec![x, y, z]
}

/// P (64 rows) shared by three chats of 112, 128 and 80 rows.
fn shared_prefix() -> Vec<Chat> {
    let prefix: Vec<u32> = (0..64).map(|row| 900_000 + row).collect();
    vec![
        Chat::extending(&prefix, 1, 112, 3, DAY),
        Chat::extending(&prefix, 2, 128, 2, 2 * DAY),
        Chat::extending(&prefix, 3, 80, 1, 3 * DAY),
    ]
}

/// Three chats with a shared prefix move into an own store whose
/// limit is 1.3x the source: each comes back with its old rows and stamp
/// and decodes to its source's rows; the shared pages are made once; the
/// source directory is gone and the own store is within its limit.
/// Mutation no-lock (the open takes no source hold) → red.
#[test]
fn chats_with_a_shared_prefix_all_come_back_with_their_rows() {
    let root = Root::new("shared");
    let layout = block_layout();
    let chats = shared_prefix();
    write_source(&root, &chats, &layout);
    let store = bytes(&root.dir(SOURCE));
    let limit = store * 13 / 10;
    let tier = open_own(&root, limit);
    let moved = move_source(&tier, &shape(&layout));
    assert_eq!(
        adopted(&tier),
        rows_of(&chats.iter().collect::<Vec<_>>()),
        "all three chats are taken back at their old rows"
    );
    assert_eq!(
        moved,
        Moved {
            states: 3,
            pages: 5,
            fragments: 2,
            source_removed: true,
            ..Moved::default()
        }
    );
    for chat in &chats {
        assert!(restores(&tier, chat, &layout), "{} rows", chat.tokens.len());
    }
    assert!(!root.dir(SOURCE).exists());
    assert!(tier.with_source(|_| ()).is_none());
    assert!(tier.manager.total_bytes() <= limit);
    assert_eq!(tier.manager.total_bytes(), bytes(&root.dir(OWN)));
}

/// Every object is converted once (a page two chats share is not
/// made again); a second load converts nothing, with no source left and
/// with a source whose removal or whose last state object's removal the
/// first load did not reach.
/// Mutation skip-off (a page in the own store is made again) → red.
#[test]
fn a_second_load_converts_nothing() {
    let layout = block_layout();
    let chats = shared_prefix();
    let all = rows_of(&chats.iter().collect::<Vec<_>>());

    let root = Root::new("second");
    write_source(&root, &chats, &layout);
    let tier = open_own(&root, GIB);
    let log = watch(None, None);
    let moved = move_source(&tier, &shape(&layout));
    hooks::observe(None);
    assert_eq!(
        hooks::take_conversions(),
        5 + 2 + 3,
        "5 pages, 2 fragments and 3 states are converted, each once"
    );
    assert!(moved.source_removed);
    let steps = log.borrow().clone();
    close(tier);
    let tier = open_own(&root, GIB);
    assert_eq!(move_source(&tier, &shape(&layout)), Moved::default());
    assert_eq!(
        hooks::take_conversions(),
        0,
        "a second load converts nothing"
    );
    assert_eq!(adopted(&tier), all);
    close(tier);
    drop(root);

    let last_state = steps
        .iter()
        .rposition(|step| matches!(step, Step::ReadBack(_)))
        .expect("a state read back");
    let removal = steps
        .iter()
        .position(|step| *step == Step::Remove)
        .expect("a removal");
    for cut_at in [last_state, removal] {
        let root = Root::new("second-cut");
        write_source(&root, &chats, &layout);
        let tier = open_own(&root, GIB);
        watch(Some(cut_at), None);
        let first = move_source(&tier, &shape(&layout));
        hooks::observe(None);
        assert!(!first.source_removed && root.dir(SOURCE).is_dir());
        hooks::take_conversions();
        close(tier);
        let tier = open_own(&root, GIB);
        let second = move_source(&tier, &shape(&layout));
        assert_eq!(
            hooks::take_conversions(),
            0,
            "cut at step {cut_at}: what the own store holds is not converted again"
        );
        assert!(second.source_removed && !second.stopped);
        assert_eq!((second.pages, second.fragments), (0, 0));
        assert_eq!(adopted(&tier), all);
        for chat in &chats {
            assert!(restores(&tier, chat, &layout));
        }
    }
}

/// A run cut at any step, a write the queue does not take, a write
/// that fails and a write the writer refuses for room, each at every write
/// of a 12-object source: at the stop every chat still decodes from the two
/// directories, no source object went before the object that holds its
/// rows read back and reached the device, and the next run brings every
/// chat back.
/// Mutation unlink-before-durable (a page's halves go before it is
/// written) → red.
#[test]
fn a_stop_at_any_point_loses_no_chat() {
    let layout = block_layout();
    let chats = twelve_objects();
    let all = rows_of(&chats.iter().collect::<Vec<_>>());

    let (steps, writes) = {
        let root = Root::new("clean");
        write_source(&root, &chats, &layout);
        assert_eq!(listing(&root.dir(SOURCE)).len(), 12);
        let tier = open_own(&root, GIB);
        let log = watch(None, None);
        let moved = move_source(&tier, &shape(&layout));
        hooks::observe(None);
        assert!(moved.source_removed && !moved.stopped);
        let log = log.borrow();
        assert_order(&log, &chats, "no stop");
        (log.len(), writes_in(&log))
    };
    assert_eq!(writes, 9, "5 pages, 1 fragment, 3 states");

    let mut cases: Vec<(Option<usize>, Option<(usize, Act)>)> =
        (0..steps).map(|step| (Some(step), None)).collect();
    for write in 0..writes {
        cases.push((None, Some((write, Act::Drop))));
        cases.push((None, Some((write, Act::Fail))));
    }
    for (cut_at, act) in cases {
        let case = format!("cut {cut_at:?}, write {act:?}");
        let root = Root::new("stop");
        write_source(&root, &chats, &layout);
        let tier = open_own(&root, GIB);
        let log = watch(cut_at, act);
        let first = move_source(&tier, &shape(&layout));
        hooks::observe(None);
        assert!(!first.source_removed, "{case}");
        assert_order(&log.borrow(), &chats, &case);
        for chat in &chats {
            assert!(
                recoverable(&root, chat, &layout),
                "{case}: no row of a chat is gone from both directories"
            );
        }
        close(tier);
        let tier = open_own(&root, GIB);
        let second = move_source(&tier, &shape(&layout));
        assert!(second.source_removed && !second.stopped, "{case}");
        assert!(!root.dir(SOURCE).exists(), "{case}");
        assert_eq!(adopted(&tier), all, "{case}");
        for chat in &chats {
            assert!(restores(&tier, chat, &layout), "{case}");
        }
    }

    for write in 0..writes {
        let case = format!("write {write} refused for room");
        let root = Root::new("refused");
        write_source(&root, &chats, &layout);
        let tier = open_own(&root, 256 * 1024);
        let junk = Rc::new(RefCell::new(Vec::new()));
        let (filler, keys) = (Arc::clone(&tier), Rc::clone(&junk));
        let mut seen = 0;
        hooks::observe(Some(Box::new(move |step| {
            if matches!(step, Step::Write(_)) {
                if seen == write {
                    fill(&filler, &mut keys.borrow_mut());
                }
                seen += 1;
            }
            Act::Go
        })));
        let first = move_source(&tier, &shape(&layout));
        hooks::observe(None);
        assert!(first.stopped && !first.source_removed, "{case}");
        for chat in &chats {
            assert!(recoverable(&root, chat, &layout), "{case}");
        }
        assert!(!junk.borrow().is_empty());
        for key in junk.borrow().iter() {
            assert!(tier.manager.remove(key));
        }
        let second = move_source(&tier, &shape(&layout));
        assert!(second.source_removed && !second.stopped, "{case}");
        assert_eq!(adopted(&tier), all, "{case}");
        for chat in &chats {
            assert!(restores(&tier, chat, &layout), "{case}");
        }
    }
}

/// Fills the own store to within less than any object of a chat of its
/// limit, with objects of no chat.
fn fill(tier: &DiskTier, junk: &mut Vec<ColdCacheKey>) {
    let manager = &tier.manager;
    let room = || manager.quota_bytes().saturating_sub(manager.total_bytes());
    loop {
        let free = room();
        let landed = (2..40u64)
            .map(|step| step * 128)
            .filter(|slack| free > *slack)
            .any(|slack| {
                let tokens = [junk.len() as u32];
                let layout = ColdCacheLayout {
                    block_size: 1,
                    num_layers: 1,
                    num_kv_heads: 1,
                    head_size: 1,
                    cache_dtype: "Junk".to_string(),
                    key_bytes_per_layer: (free - slack) as usize,
                    value_bytes_per_layer: 1,
                };
                let block = ColdCacheBlock {
                    key: ColdCacheKey::chain(
                        ColdGroup::Kv,
                        tier.kv_fingerprint,
                        None,
                        &tokens,
                        &[],
                        0xdead,
                        0,
                    ),
                    fingerprint: tier.kv_fingerprint,
                    tokens: tokens.to_vec(),
                    layers: vec![ColdLayerBlock {
                        keys: vec![0x5a; layout.key_bytes_per_layer],
                        values: vec![0xa5],
                    }],
                    layout,
                };
                let key = block.key;
                let landed = land_block(manager, block, MakeRoom::Refuse);
                if landed {
                    junk.push(key);
                }
                landed
            });
        if !landed {
            break;
        }
    }
    assert!(room() < 256, "the own store is full");
}

/// The bytes both directories hold at every step of a run, and the largest
/// state object and block of the source.
fn sums(root: &Root, tier: &Arc<DiskTier>, shape: &MoveShape<'_>) -> (Moved, Vec<u64>) {
    let sums = Rc::new(RefCell::new(Vec::new()));
    let (seen, own, source) = (Rc::clone(&sums), root.dir(OWN), root.dir(SOURCE));
    hooks::observe(Some(Box::new(move |_| {
        seen.borrow_mut().push(bytes(&own) + bytes(&source));
        Act::Go
    })));
    let moved = move_source(tier, shape);
    hooks::observe(None);
    let sums = sums.borrow().clone();
    (moved, sums)
}

/// The most two directories may hold during a move of a source of `store`
/// bytes with `tails` odd states: the store, one state object on disk
/// twice, the batch of source blocks that wait for a device flush with the
/// page in flight, and each odd state's tail block beside its fragment and
/// beside the page of a longer chat.
fn bound(store: u64, state: u64, block: u64, tails: u64) -> u64 {
    store + state + (2 * mlx_paged_attn::DEFAULT_QUEUE_DEPTH as u64 + 4 + 2 * tails) * block
}

fn largest(dir: &Path, state: bool) -> u64 {
    listing(dir)
        .into_iter()
        .filter(|(name, ..)| name.contains("dflash2_state") == state)
        .map(|(_, len, _)| len)
        .max()
        .unwrap_or(0)
}

/// (a) an own limit short of the source: the two newest chats come
/// back, the oldest is refused whole and goes with the source, and the two
/// directories never hold more than the source did plus what is in flight;
/// (b) a volume below the writer's reserve: the run stops at its first
/// write, the source is as it was, and a later load with room brings
/// every chat back.
/// Mutation copy-mode (no source object goes before the directory) → red.
#[test]
fn a_short_limit_keeps_the_newest_and_a_full_volume_loses_nothing() {
    let layout = block_layout();
    let chats = [
        Chat::new(1, 640, 3, HOUR),
        Chat::new(2, 512, 2, 2 * HOUR),
        Chat::new(3, 768, 1, 3 * HOUR),
    ];
    let root = Root::new("limit");
    write_source(&root, &chats, &layout);
    let source = root.dir(SOURCE);
    let store = bytes(&source);
    let (state, block) = (largest(&source, true), largest(&source, false));
    let oldest = chats[2].source_keys().len() as u64 * block;
    let limit = store - oldest / 2;
    let tier = open_own(&root, limit);
    let (moved, sums) = sums(&root, &tier, &shape(&layout));
    assert_eq!(
        (moved.states, moved.refused, moved.stopped),
        (2, 1, false),
        "the oldest chat is past the limit"
    );
    assert_eq!(
        adopted(&tier),
        rows_of(&[&chats[0], &chats[1]]),
        "the two newest chats are kept"
    );
    assert!(restores(&tier, &chats[0], &layout) && restores(&tier, &chats[1], &layout));
    assert!(moved.source_removed && !source.exists());
    assert!(tier.manager.total_bytes() <= limit);
    let most = sums.iter().copied().max().unwrap_or(0);
    assert!(
        most <= bound(store, state, block, 0),
        "the two directories held {most} bytes of a {store} byte source"
    );
    close(tier);
    drop(root);

    let root = Root::new("volume");
    write_source(&root, &chats, &layout);
    let before = listing(&root.dir(SOURCE));
    TEST_WRITER.with(|writer| writer.set(Some((8, u64::MAX / 4))));
    let tier = open_own(&root, GIB);
    let stopped = move_source(&tier, &shape(&layout));
    assert_eq!(
        stopped,
        Moved {
            stopped: true,
            ..Moved::default()
        },
        "the writer refuses the first write below its reserve"
    );
    assert_eq!(
        listing(&root.dir(SOURCE)),
        before,
        "nothing left the source"
    );
    assert_eq!(bytes(&root.dir(OWN)), 0);
    assert!(tier.with_source(|_| ()).is_some(), "the source stays held");
    close(tier);
    TEST_WRITER.with(|writer| writer.set(Some((8, 0))));
    let tier = open_own(&root, GIB);
    let moved = move_source(&tier, &shape(&layout));
    assert!(moved.source_removed && moved.states == 3);
    assert_eq!(adopted(&tier), rows_of(&chats.iter().collect::<Vec<_>>()));
}

/// A moved object has its source's time (a page: the newer of its
/// halves), on disk and in the index, and a moved state its stamp; a later
/// load with a limit one byte short gives up the chat the source would.
/// Mutation time-stamped (a moved object keeps its write time) → red.
#[test]
fn moved_objects_keep_their_times_and_their_eviction_order() {
    let layout = block_layout();
    let mut chats = [
        Chat::new(1, 80, 7, 3 * DAY),
        Chat::new(2, 96, 9, 2 * DAY),
        Chat::new(3, 112, 8, DAY),
    ];
    chats[2].stagger = SECOND;
    chats[1].state_at += 5 * SECOND;
    let root = Root::new("times");
    write_source(&root, &chats, &layout);
    let tier = open_own(&root, GIB);
    let moved = move_source(&tier, &shape(&layout));
    assert!(moved.source_removed && moved.states == 3);
    let file_time = |name: String| {
        std::fs::metadata(root.dir(OWN).join(name))
            .and_then(|meta| meta.modified())
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|time| time.as_nanos())
    };
    for chat in &chats {
        let expected = expected(chat, &layout);
        for (index, page) in expected.plan.pages.iter().enumerate() {
            let want = chat
                .block_time(2 * index)
                .max(chat.block_time(2 * index + 1));
            assert_eq!(
                tier.manager.modified(&page.key),
                Some(want),
                "a page has the time of the newer of its halves"
            );
            assert_eq!(
                file_time(format!("{}.safetensors", page.key.to_hex())),
                Some(want)
            );
        }
        if let Some(fragment) = &expected.plan.fragment {
            let want = chat.block_time(2 * expected.plan.pages.len());
            assert_eq!(tier.manager.modified(&fragment.key), Some(want));
        }
        let state = expected.plan.state_key;
        assert_eq!(tier.manager.modified(&state), Some(chat.state_at));
        assert_eq!(
            file_time(format!("{}.dflash2_state.safetensors", state.to_hex())),
            Some(chat.state_at)
        );
    }
    assert_eq!(
        adopted(&tier),
        rows_of(&chats.iter().collect::<Vec<_>>()),
        "the stamps are the source's"
    );
    let total = tier.manager.total_bytes();
    close(tier);

    let tier = open_own(&root, total - 1);
    assert_eq!(
        adopted(&tier),
        rows_of(&[&chats[1], &chats[2]]),
        "one byte short, the chat with the oldest source objects gives way"
    );
}

/// A newer chat goes on past an older chat's odd last block, so
/// that block is the first half of one of its pages. The block stays in
/// the source after that page is written, the older chat's fragment is
/// made of it, and both chats come back.
/// Mutation tail-unlinked (the block goes with its page's other half) →
/// red.
#[test]
fn a_tail_block_a_newer_chat_extends_stays_for_its_own_chat() {
    let layout = block_layout();
    let older = Chat::new(1, 48, 1, 2 * DAY);
    let newer = Chat::extending(&older.tokens, 2, 96, 2, DAY);
    let chats = [older, newer];
    let tail = chats[0].source_keys()[2];
    assert_eq!(chats[1].source_keys()[2], tail);
    let root = Root::new("tail");
    write_source(&root, &chats, &layout);
    let tier = open_own(&root, GIB);
    let log = watch(None, None);
    let moved = move_source(&tier, &shape(&layout));
    hooks::observe(None);
    assert_eq!(
        adopted(&tier),
        rows_of(&chats.iter().collect::<Vec<_>>()),
        "both chats come back"
    );
    assert_eq!(
        (moved.states, moved.pages, moved.fragments, moved.unreadable),
        (2, 3, 1, 0)
    );
    assert!(restores(&tier, &chats[0], &layout) && restores(&tier, &chats[1], &layout));
    let log = log.borrow();
    assert!(
        !log.contains(&Step::Unlink(tail)),
        "the tail block leaves only with the source directory"
    );
    assert_order(&log, &chats, "shared tail");
    assert!(moved.source_removed);
}

/// The own store is full of chats an earlier move brought, and the
/// block-16 load wrote newer chats since. The newer chats all come back;
/// own objects older than them give way; no chat is refused.
/// Mutation no-evict-older (a full own store refuses) → red.
#[test]
fn newer_chats_of_the_source_take_the_room_of_older_own_chats() {
    let layout = block_layout();
    let old = [Chat::new(1, 64, 1, 10 * DAY), Chat::new(2, 128, 2, 9 * DAY)];
    let new = [Chat::new(3, 160, 4, HOUR), Chat::new(4, 96, 3, 2 * HOUR)];
    let root = Root::new("forward");
    write_source(&root, &old, &layout);
    let tier = open_own(&root, GIB);
    assert!(move_source(&tier, &shape(&layout)).source_removed);
    assert_eq!(adopted(&tier), rows_of(&old.iter().collect::<Vec<_>>()));
    let own = tier.manager.total_bytes();
    close(tier);

    write_source(&root, &new, &layout);
    let limit = own + bytes(&root.dir(SOURCE)) / 2;
    let tier = open_own(&root, limit);
    let moved = move_source(&tier, &shape(&layout));
    let rows = adopted(&tier);
    for chat in &new {
        assert!(
            rows.contains(&(chat.tokens.len(), chat.last_used)) && restores(&tier, chat, &layout),
            "the newer chat of {} rows is kept",
            chat.tokens.len()
        );
    }
    assert_eq!((moved.states, moved.refused, moved.stopped), (2, 0, false));
    assert!(moved.source_removed);
    assert!(rows.len() < 4, "older own chats gave way");
    assert!(tier.manager.total_bytes() <= limit);
}

/// At every step of a move the two directories hold at most the
/// source, one state object and the batch in flight more.
/// Mutation copy-mode (no source object goes before the directory) → red.
#[test]
fn both_directories_never_hold_more_than_one_store_and_the_batch_in_flight() {
    let layout = block_layout();
    let chats = [Chat::new(1, 656, 2, HOUR), Chat::new(2, 640, 1, 2 * HOUR)];
    let root = Root::new("sum");
    write_source(&root, &chats, &layout);
    let source = root.dir(SOURCE);
    let store = bytes(&source);
    let (state, block) = (largest(&source, true), largest(&source, false));
    let tier = open_own(&root, GIB);
    let (moved, sums) = sums(&root, &tier, &shape(&layout));
    assert!(moved.source_removed && moved.states == 2);
    assert!(
        sums.len() > 2 * (20 + 20),
        "a sum at every step of 40 pages"
    );
    let most = sums.iter().copied().max().unwrap_or(0);
    assert!(
        most <= bound(store, state, block, 1),
        "the two directories held {most} bytes of a {store} byte source"
    );
    assert!(
        bytes(&root.dir(OWN)) <= store,
        "pages take no more room than their halves"
    );
}

/// An own state newer than the one in hand keeps its objects, however old
/// their times: a run cut after the newest chat moved, then a load with no
/// room for the older chat, refuses the older chat and keeps the newest.
#[test]
fn an_older_chat_takes_no_room_from_a_newer_own_chat() {
    let layout = block_layout();
    let mut newest = Chat::new(1, 96, 2, HOUR);
    newest.blocks_at -= u128::from(5 * DAY) * SECOND;
    let mut older = Chat::new(2, 96, 1, 2 * HOUR);
    older.blocks_at -= u128::from(3 * DAY) * SECOND;
    let chats = [newest, older];
    let root = Root::new("newer-own");
    write_source(&root, &chats, &layout);
    let tier = open_own(&root, GIB);
    watch(None, Some((4, Act::Cut)));
    let first = move_source(&tier, &shape(&layout));
    hooks::observe(None);
    assert_eq!((first.states, first.stopped), (1, true));
    let own = tier.manager.total_bytes();
    close(tier);

    let tier = open_own(&root, own + 1024);
    let second = move_source(&tier, &shape(&layout));
    assert_eq!(
        (second.states, second.refused, second.source_removed),
        (0, 1, true)
    );
    assert_eq!(adopted(&tier), rows_of(&[&chats[0]]));
    assert!(restores(&tier, &chats[0], &layout));
}

/// A damaged source block loses its own chat, as for the old load, and no
/// other; a sound source object that is no block of the expected layout
/// stops the run before anything moves and the source stays.
#[test]
fn a_damaged_block_loses_one_chat_and_a_foreign_one_holds_the_source() {
    let layout = block_layout();
    let chats = [
        Chat::new(1, 64, 3, HOUR),
        Chat::new(2, 64, 2, 2 * HOUR),
        Chat::new(3, 80, 1, 3 * HOUR),
    ];
    let name = |key: ColdCacheKey| format!("{}.safetensors", key.to_hex());
    let root = Root::new("damaged");
    write_source(&root, &chats, &layout);
    let path = root.dir(SOURCE).join(name(chats[1].source_keys()[1]));
    let mut damaged = std::fs::read(&path).expect("a block");
    let last = damaged.len() - 1;
    damaged[last] ^= 0xff;
    std::fs::write(&path, damaged).expect("damage it");
    let tier = open_own(&root, GIB);
    let moved = move_source(&tier, &shape(&layout));
    assert_eq!(
        (
            moved.states,
            moved.unreadable,
            moved.held_back,
            moved.stopped
        ),
        (2, 1, 0, false)
    );
    assert!(moved.source_removed);
    assert_eq!(adopted(&tier), rows_of(&[&chats[0], &chats[2]]));
    assert!(restores(&tier, &chats[0], &layout) && restores(&tier, &chats[2], &layout));
    close(tier);
    drop(root);

    let root = Root::new("foreign");
    write_source(&root, &chats, &layout);
    let wider = source_layout(LAYERS, HEADS, 2 * DIM).expect("another layout");
    let key = chats[0].source_keys()[0];
    {
        let writer = ColdCacheManager::open_at(root.dir(SOURCE), GIB, 0, 8).expect("the source");
        assert!(land_block(
            &writer,
            source_block(key, &chats[0].tokens[..16], &wider),
            MakeRoom::Evict
        ));
        assert!(writer.close(WAIT));
    }
    let before = listing(&root.dir(SOURCE));
    let tier = open_own(&root, GIB);
    let moved = move_source(&tier, &shape(&layout));
    assert_eq!(
        moved,
        Moved {
            held_back: 1,
            stopped: true,
            ..Moved::default()
        }
    );
    assert_eq!(listing(&root.dir(SOURCE)), before);
    assert!(recoverable(&root, &chats[1], &layout) && recoverable(&root, &chats[2], &layout));
}

/// Two chats part inside a page: the block before the parting is the first
/// half of a page of each. It stays in the source until both pages are in
/// the own store and read back, and both chats come back.
/// Mutation first-page-unlinks (the block goes after the first page made
/// of it) → red.
#[test]
fn a_block_two_chats_share_inside_a_page_stays_for_both_pages() {
    let layout = block_layout();
    let prefix: Vec<u32> = (0..48).map(|row| 700_000 + row).collect();
    let chats = [
        Chat::extending(&prefix, 1, 96, 2, DAY),
        Chat::extending(&prefix, 2, 112, 1, 2 * DAY),
    ];
    let shared = chats[0].source_keys()[2];
    assert_eq!(chats[1].source_keys()[2], shared);
    let pages: Vec<ColdCacheKey> = chats
        .iter()
        .map(|chat| expected(chat, &layout).plan.pages[1].key)
        .collect();
    assert_ne!(
        pages[0], pages[1],
        "each chat has its own page on the block"
    );
    let root = Root::new("parting");
    write_source(&root, &chats, &layout);
    let tier = open_own(&root, GIB);
    let log = watch(None, None);
    let moved = move_source(&tier, &shape(&layout));
    hooks::observe(None);
    assert_eq!(
        adopted(&tier),
        rows_of(&chats.iter().collect::<Vec<_>>()),
        "both chats come back"
    );
    assert!(restores(&tier, &chats[0], &layout) && restores(&tier, &chats[1], &layout));
    assert_eq!(
        moved,
        Moved {
            states: 2,
            pages: 5,
            fragments: 1,
            source_removed: true,
            ..Moved::default()
        }
    );
    let log = log.borrow();
    let gone = log
        .iter()
        .position(|step| *step == Step::Unlink(shared))
        .expect("the shared block goes before the directory");
    for page in &pages {
        assert!(
            log[..gone].contains(&Step::ReadBack(*page)),
            "the shared block goes after both pages read back"
        );
    }
    assert_order(&log, &chats, "parting");
}

/// The old load wrote a state again at rows the own store has one for: the
/// source's object takes the own one's place with its stamp and time, its
/// pages are not made again and take the newer time.
#[test]
fn a_state_the_old_load_wrote_again_takes_the_place_of_the_own_one() {
    let layout = block_layout();
    let first = Chat::new(1, 64, 5, 2 * DAY);
    let root = Root::new("again");
    write_source(&root, std::slice::from_ref(&first), &layout);
    let tier = open_own(&root, GIB);
    assert!(move_source(&tier, &shape(&layout)).source_removed);
    assert_eq!(adopted(&tier), rows_of(&[&first]));
    close(tier);

    let again = Chat::new(1, 64, 9, HOUR);
    assert_eq!(again.tokens, first.tokens);
    write_source(&root, std::slice::from_ref(&again), &layout);
    let tier = open_own(&root, GIB);
    hooks::take_conversions();
    let moved = move_source(&tier, &shape(&layout));
    assert_eq!(
        hooks::take_conversions(),
        1,
        "the state object alone is converted"
    );
    assert_eq!(
        moved,
        Moved {
            states: 1,
            source_removed: true,
            ..Moved::default()
        }
    );
    assert_eq!(adopted(&tier), rows_of(&[&again]), "the newer stamp");
    assert!(restores(&tier, &again, &layout));
    let plan = expected(&again, &layout).plan;
    assert_eq!(tier.manager.modified(&plan.state_key), Some(again.state_at));
    for page in &plan.pages {
        assert_eq!(tier.manager.modified(&page.key), Some(again.blocks_at));
    }
}

/// A state the old load takes back and the move cannot convert, and a
/// source the converter does not take, each hold the whole source: nothing
/// is converted, nothing leaves, and the source stays held.
#[test]
fn a_source_the_move_cannot_convert_stays_whole() {
    let layout = block_layout();
    let plain = Chat::new(1, 64, 2, HOUR);
    let mut odd_keys = Chat::new(2, 32, 1, 2 * HOUR);
    odd_keys.extra_keys = vec![vec![1, 2, 3]];
    let root = Root::new("held-back");
    write_source(&root, &[plain.clone(), odd_keys], &layout);
    let before = listing(&root.dir(SOURCE));
    let tier = open_own(&root, GIB);
    hooks::take_conversions();
    assert_eq!(
        move_source(&tier, &shape(&layout)),
        Moved {
            held_back: 1,
            stopped: true,
            ..Moved::default()
        }
    );
    assert_eq!(hooks::take_conversions(), 0);
    assert_eq!(listing(&root.dir(SOURCE)), before);
    assert_eq!(bytes(&root.dir(OWN)), 0);
    assert!(tier.with_source(|_| ()).is_some());
    close(tier);
    drop(root);

    let side = (HEADS * 8 * 16 + HEADS * 16 * 4) as usize;
    let narrow = ColdCacheLayout {
        block_size: SOURCE_BLOCK,
        num_layers: LAYERS,
        num_kv_heads: HEADS,
        head_size: 8,
        cache_dtype: "Int8".to_string(),
        key_bytes_per_layer: side,
        value_bytes_per_layer: side,
    };
    let root = Root::new("no-repack");
    write_source(&root, std::slice::from_ref(&plain), &narrow);
    let before = listing(&root.dir(SOURCE));
    let tier = open_own(&root, GIB);
    assert_eq!(
        move_source(&tier, &shape(&narrow)),
        Moved {
            held_back: 1,
            stopped: true,
            ..Moved::default()
        }
    );
    assert_eq!(listing(&root.dir(SOURCE)), before);
    assert_eq!(bytes(&root.dir(OWN)), 0);
}
