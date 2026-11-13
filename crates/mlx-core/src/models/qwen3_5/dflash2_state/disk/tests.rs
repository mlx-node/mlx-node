use std::path::PathBuf;
use std::sync::Arc;

use mlx_paged_attn::MakeRoom;

use super::*;
use crate::array::DType;
use crate::disk_tier::{CacheDisk, DiskTier, DiskTierOptions, open_disk_tier};
use crate::engine::governor::ledger::{
    CertainReuse, FreeOutcome, Need, Stage, TakeRights, UnitKey, Victim, VictimQuery, VictimSource,
};
use crate::models::qwen3_5::config::Qwen3_5Config;
use crate::models::qwen3_5::dflash2_state::paged::PagedAnchor;
use crate::models::qwen3_5::dflash2_state::store::{Lineage, StateClass};
use crate::models::qwen3_5::dflash2_turn::{
    GrowReclaim, Keep, ModelUnits, clear_grow_reclaim_log, grow_reclaim_log,
};
use crate::models::qwen3_5::draft_context::DraftContextWindow;
use crate::models::qwen3_5::model::Qwen35Inner;
use crate::models::qwen3_5::model::dflash2_park_tests::{
    cache_chain, chain_tokens, install_pool, passing_probe,
};

/// A temporary SSD tier under a fresh test root.
pub(crate) struct TestTier {
    pub(crate) tier: Arc<DiskTier>,
    root: PathBuf,
}

impl Drop for TestTier {
    fn drop(&mut self) {
        self.tier.close(std::time::Instant::now());
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

pub(crate) fn test_tier(tag: &str, limit: u64) -> TestTier {
    open_test_tier(tag, limit, false)
}

/// A persistent SSD tier under a fresh test root.
pub(crate) fn persistent_test_tier(tag: &str, limit: u64) -> TestTier {
    open_test_tier(tag, limit, true)
}

impl TestTier {
    /// The same tier as the next process opens it: this one closed, its
    /// files kept.
    pub(crate) fn reopen(mut self, tag: &str, limit: u64) -> TestTier {
        let root = std::mem::take(&mut self.root);
        let persistent = self.tier.persistent;
        drop(self);
        let tier = tier_at(&root, tag, limit, persistent);
        TestTier { tier, root }
    }
}

/// A temporary SSD tier whose writer queues `depth` block jobs and keeps
/// `reserve` bytes of the volume free.
pub(crate) fn test_tier_with(tag: &str, limit: u64, depth: usize, reserve: u64) -> TestTier {
    crate::disk_tier::TEST_WRITER.with(|writer| writer.set(Some((depth, reserve))));
    let tier = open_test_tier(tag, limit, false);
    crate::disk_tier::TEST_WRITER.with(|writer| writer.set(None));
    tier
}

fn open_test_tier(tag: &str, limit: u64, persistent: bool) -> TestTier {
    let root = std::env::temp_dir().join(format!(
        "mlx-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).expect("temp root");
    let tier = tier_at(&root, tag, limit, persistent);
    TestTier { tier, root }
}

fn tier_at(root: &std::path::Path, tag: &str, limit: u64, persistent: bool) -> Arc<DiskTier> {
    let options = DiskTierOptions {
        cache_disk: CacheDisk::Bytes(limit),
        persistent,
        cache_dir: persistent.then(|| root.join("cache")),
        temp_root: Some(root.join("tmp")),
    };
    std::fs::create_dir_all(root.join("tmp")).expect("temp root");
    let identity = || {
        let fingerprint = mlx_paged_attn::ColdCacheFingerprint::from_components([tag.as_bytes()]);
        Some(crate::disk_tier::PersistentIdentity {
            namespace: "0123456789abcdef0123456789abcdef".to_string(),
            state: fingerprint,
            kv: fingerprint,
        })
    };
    let tier = open_disk_tier(&options, identity, 1 << 20).expect("a tier");
    assert_eq!(tier.persistent, persistent);
    tier
}

/// The model's states on `tier`, their window `window` bytes (0: a load
/// without a draft, whose state is the GDN cell alone).
pub(crate) fn attach_tier(
    states: &mut DFlash2States,
    config: &Qwen3_5Config,
    adapter: &PagedKVCacheAdapter,
    tier: &TestTier,
    window: u64,
    slots: u32,
) {
    let gdn = gdn_sidecar::geometry(config, DType::BFloat16).expect("a GDN geometry");
    let block_bytes = mlx_paged_attn::block_object_bytes(adapter.layer_kv_pool());
    states.disk = Some(
        StateDisk::new(
            Arc::clone(&tier.tier),
            gdn,
            (window > 0).then_some(window),
            slots,
            "test geometry".to_string(),
            block_bytes,
        )
        .expect("a state disk"),
    );
}

/// A state of `config`'s GDN geometry the tier can write, with `window`.
pub(crate) fn writable_anchor(config: &Qwen3_5Config, window: DraftContextWindow) -> PagedAnchor {
    let gdn = gdn_sidecar::geometry(config, DType::BFloat16).expect("a GDN geometry");
    let cell = vec![0x3c; gdn.cell_bytes().expect("cell bytes") as usize];
    let linear = gdn_sidecar::decode_caches(config, &gdn, &cell)
        .expect("decode")
        .expect("the geometry's cell");
    let mut arrays = Vec::new();
    for cache in &linear {
        cache.collect_arrays(&mut arrays);
    }
    crate::array::MxArray::eval_arrays(&arrays).expect("eval");
    PagedAnchor {
        linear,
        window: Some(window),
    }
}

fn fixture(seed: u64, tag: &str) -> Option<(Qwen35Inner, TestTier)> {
    fixture_on(seed, 8, 8, test_tier(tag, 1 << 30))
}

/// A tiny model with a pool of `initial` blocks growing to `max`, its
/// states on `tier`.
pub(crate) fn fixture_on(
    seed: u64,
    initial: u32,
    max: u32,
    tier: TestTier,
) -> Option<(Qwen35Inner, TestTier)> {
    if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device() {
        return None;
    }
    let mut inner =
        crate::models::qwen3_5::dflash2_decode::tests::tiny_dflash_inner_with_config(seed, |_| {});
    install_pool(&mut inner, initial, max, passing_probe()).expect("pool");
    let window = window(&inner);
    attach_tier(
        &mut inner.dflash2_states,
        &inner.config,
        inner.paged_adapter.as_ref().expect("pool"),
        &tier,
        window.nbytes(),
        window.slots(),
    );
    Some((inner, tier))
}

pub(crate) fn window(inner: &Qwen35Inner) -> DraftContextWindow {
    inner
        .dflash2
        .as_ref()
        .expect("draft")
        .config
        .context_window()
        .expect("window")
}

/// A state on a cached one-block chain of `seed`.
fn park(inner: &mut Qwen35Inner, seed: u32, class: StateClass) -> EntryId {
    let block_size = inner.paged_adapter.as_ref().expect("pool").block_size();
    let tokens = chain_tokens(seed, 1, block_size);
    cache_chain(inner, &tokens);
    let anchor = writable_anchor(&inner.config, window(inner));
    let id = inner
        .dflash2_states
        .store
        .insert(Lineage::text(&tokens, 0), anchor, class)
        .expect("a whole-block state")
        .id;
    inner.sync_gdn_state_marks();
    id
}

fn waits_for_writes() -> VictimQuery {
    VictimQuery {
        rights: TakeRights {
            units: true,
            waits_for_writes: true,
            certain_reuse: CertainReuse::All,
            ..TakeRights::default()
        },
        need: Need {
            bytes: 1,
            ..Need::default()
        },
    }
}

/// The store's units: key, stage, later.
fn store_units(inner: &mut Qwen35Inner, query: &VictimQuery) -> Vec<(UnitKey, Stage, bool)> {
    ModelUnits {
        target: inner,
        keep: Keep::default(),
    }
    .units(query)
    .into_iter()
    .filter(|unit| matches!(unit.key, UnitKey::Store(_)))
    .map(|unit| (unit.key, unit.stage, unit.later))
    .collect()
}

fn unit_of(inner: &mut Qwen35Inner, query: &VictimQuery, id: EntryId) -> Victim {
    ModelUnits {
        target: inner,
        keep: Keep::default(),
    }
    .units(query)
    .into_iter()
    .find(|unit| unit.key == UnitKey::Store(id))
    .expect("a unit of the state")
}

fn take(
    inner: &mut Qwen35Inner,
    query: &VictimQuery,
    id: EntryId,
) -> crate::engine::governor::ledger::Taken {
    let unit = unit_of(inner, query, id);
    ModelUnits {
        target: inner,
        keep: Keep::default(),
    }
    .take(query, &unit)
}

/// Takes the writer's next landing off the state tier's channel.
fn hold_landing(inner: &Qwen35Inner, tier: &TestTier) -> Landed {
    assert!(tier.tier.manager.drain(std::time::Duration::from_secs(10)));
    inner
        .dflash2_states
        .disk
        .as_ref()
        .expect("tier")
        .landed_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("a landing")
}

fn release_landing(inner: &Qwen35Inner, landing: Landed) {
    inner
        .dflash2_states
        .disk
        .as_ref()
        .expect("tier")
        .landed_tx
        .send(landing)
        .expect("the tier's channel");
}

pub(crate) fn landed(inner: &mut Qwen35Inner, tier: &TestTier) {
    assert!(tier.tier.manager.drain(std::time::Duration::from_secs(10)));
    assert!(poll_offload(inner), "a write landed");
}

pub(crate) fn ram(inner: &Qwen35Inner, id: EntryId) -> bool {
    inner
        .dflash2_states
        .store
        .get(id)
        .is_some_and(|entry| entry.ram().is_some())
}

pub(crate) fn disk_copy(inner: &Qwen35Inner, id: EntryId) -> Option<DiskCopy> {
    inner
        .dflash2_states
        .store
        .get(id)
        .and_then(|entry| entry.disk().copied())
}

/// Splash `StateCache::reclaim` over the walk's order: with no write in
/// flight the first writable state of the walk, the checkpoint, is
/// staged whatever its stamp, and its RAM frees at once; the other
/// writable states wait for the staging buffer (Pending). A RAM copy
/// whose own copy is in flight frees nothing until it lands; one beside a
/// landed copy frees now. With the tier off every state frees now.
/// Mutations RS1 (every writable state is staged) → red (o1 listed with
/// `later` false); IF1 (an in-flight copy's RAM goes at the take) → red.
#[test]
fn reclaim_stages_one_write_and_frees_at_once() {
    let Some((mut inner, tier)) = fixture(0x5713_1301, "state-tier-reclaim") else {
        return;
    };
    let query = waits_for_writes();
    let o1 = park(&mut inner, 31, StateClass::Ordinary);
    let o2 = park(&mut inner, 32, StateClass::Ordinary);
    let c = park(&mut inner, 33, StateClass::Checkpoint);
    assert_eq!(
        store_units(&mut inner, &query),
        vec![
            (UnitKey::Store(c), Stage::NeedsWrite, false),
            (UnitKey::Store(o1), Stage::NeedsWrite, true),
            (UnitKey::Store(o2), Stage::NeedsWrite, true),
        ]
    );

    let taken = take(&mut inner, &query, c);
    assert_eq!(taken.outcome, FreeOutcome::Met);
    assert!(taken.now_bytes > 0, "the staged write frees the RAM copy");
    assert!(!ram(&inner, c));
    assert!(disk_copy(&inner, c).is_some_and(|copy| !copy.landed));
    let disk = inner.dflash2_states.disk.as_ref().expect("tier");
    assert_eq!(disk.counters.offloads, 1);
    assert!(disk.writing());

    let taken = take(&mut inner, &query, o1);
    assert_eq!(taken.outcome, FreeOutcome::Pending);
    assert!(ram(&inner, o1) && disk_copy(&inner, o1).is_none());

    landed(&mut inner, &tier);
    assert!(disk_copy(&inner, c).is_some_and(|copy| copy.landed));
    assert_eq!(
        store_units(&mut inner, &query),
        vec![
            (UnitKey::Store(o1), Stage::NeedsWrite, false),
            (UnitKey::Store(o2), Stage::NeedsWrite, true),
        ]
    );

    // A write-behind copy of o2: its RAM stays until it lands.
    assert_eq!(
        stage(&mut inner, o2, MakeRoom::Evict, true),
        Staged::Started
    );
    assert!(store_units(&mut inner, &query).contains(&(
        UnitKey::Store(o2),
        Stage::NeedsWrite,
        true
    )));
    let taken = take(&mut inner, &query, o2);
    assert_eq!(taken.outcome, FreeOutcome::Pending);
    assert!(ram(&inner, o2), "a copy in flight reads this RAM copy");
    landed(&mut inner, &tier);
    assert!(store_units(&mut inner, &query).contains(&(UnitKey::Store(o2), Stage::FreeNow, false)));

    inner.dflash2_states.disk = None;
    assert!(
        store_units(&mut inner, &query)
            .iter()
            .all(|(_, stage, later)| *stage == Stage::FreeNow && !later)
    );
}

/// A failed state write leaves no copy (Splash `StateCache::pollOffload`:
/// a failed offload drops the disk copy, and a state with no copy left is
/// invalid and goes); the staging buffer comes back and the chain's mark
/// goes with the state.
/// Mutation WF1 (the failed copy is kept) → red.
#[test]
fn a_failed_write_leaves_no_state() {
    let Some((mut inner, tier)) = fixture(0x5713_1501, "state-tier-failed-write") else {
        return;
    };
    let query = waits_for_writes();
    let c = park(&mut inner, 41, StateClass::Checkpoint);
    let marked = |inner: &Qwen35Inner| inner.paged_adapter.as_ref().expect("pool").marked_blocks();
    assert_eq!(marked(&inner), 1);
    // The tier's directory is gone: the writer cannot create the object.
    std::fs::remove_dir_all(tier.tier.dir()).expect("remove the tier's directory");
    assert_eq!(take(&mut inner, &query, c).outcome, FreeOutcome::Met);
    assert!(
        !tier.tier.manager.drain(std::time::Duration::from_secs(10)),
        "the write failed"
    );
    assert!(poll_offload(&mut inner), "the failure landed");
    let disk = inner.dflash2_states.disk.as_ref().expect("tier");
    assert_eq!(disk.counters.offload_failures, 1);
    assert!(disk.staging.is_some() && !disk.writing());
    assert!(inner.dflash2_states.store.get(c).is_none(), "no copy left");
    assert_eq!(marked(&inner), 0);
}

/// A claim that waits for writes and meets a state the busy staging buffer
/// holds back goes Pending and waits (Splash `MemoryLedger::take` with a
/// pending unit); a lower claim waits behind its hold; an Immediate claim
/// passes the write and is Short at once. Once the write in flight lands,
/// the waiting claim's retry stages o1 and is granted.
/// Mutation PH1 (a busy writable state's take is Short) → red (A is Short).
#[test]
fn pending_write_holds_a_waiting_claim() {
    use crate::engine::governor::allocate::{Admission, Refusal, StateAdmission, allocate};
    use crate::engine::governor::governor::Governor;
    use crate::engine::governor::governor::tests::{FakeIo, G, host, lane, model, stats};
    use crate::engine::governor::host::Pressure;
    use crate::engine::governor::ledger::{ClaimClass, ClaimSpec, NeedTiming};
    let Some((mut inner, tier)) = fixture(0x5713_1401, "state-tier-pending-hold") else {
        return;
    };
    let query = waits_for_writes();
    let c = park(&mut inner, 51, StateClass::Checkpoint);
    let o1 = park(&mut inner, 52, StateClass::Ordinary);
    assert_eq!(take(&mut inner, &query, c).outcome, FreeOutcome::Met);
    // The write stays in flight until the test hands its landing back: every
    // allocation polls the landings first.
    let held = hold_landing(&inner, &tier);
    let o1_bytes = inner
        .dflash2_states
        .store
        .get(o1)
        .expect("o1")
        .state_bytes();
    const ROOM: u64 = 4096;
    let active = G - ROOM;
    let io = FakeIo::new(stats(active), host(32 * G, Pressure::Normal));
    let governor: &'static Governor = Box::leak(Box::new(Governor::new(io.clone())));
    inner.memory_registration = Some(governor.register_model(model(G, 0)).expect("register"));
    let need = ROOM + o1_bytes / 2;
    let reserve = |bytes: u64| {
        let io = io.clone();
        move |units: &mut ModelUnits<'_, Qwen35Inner>| {
            if !ram(units.target, o1) {
                io.with(|state| state.stats.active = active - o1_bytes);
            }
            StateAdmission(governor.try_reserve(bytes))
        }
    };

    let a_spec = ClaimSpec {
        rank: crate::engine::governor::ledger::ClaimRank {
            decoding: true,
            ..lane(ClaimClass::InService, 1, 900).rank
        },
        ..lane(ClaimClass::InService, 1, 900)
    };
    let a = governor.open(a_spec, 7);
    let mut units = ModelUnits {
        target: &mut inner,
        keep: Keep::default(),
    };
    let first = allocate(
        a.claim(),
        a_spec,
        Refusal::Waits,
        false,
        &mut units,
        reserve(need),
    );
    assert!(!first.admission.granted());
    assert!(first.denial.pending, "o1 waits for the write in flight");
    assert!(governor.ledger(|ledger| ledger.claim(a.id()).is_some_and(|claim| claim.waiting)));
    assert!(ram(units.target, o1));

    let b_spec = lane(ClaimClass::InService, 2, 10);
    let b = governor.open(b_spec, 7);
    let second = allocate(
        b.claim(),
        b_spec,
        Refusal::Waits,
        false,
        &mut units,
        reserve(ROOM / 2),
    );
    assert!(!second.admission.granted(), "A's hold keeps the room");
    assert!(second.denial.pending);

    let c_spec = ClaimSpec {
        timing: NeedTiming::Immediate,
        ..lane(ClaimClass::Optional, 3, 0)
    };
    let optional = governor.open(c_spec, 7);
    let third = allocate(
        optional.claim(),
        c_spec,
        Refusal::Settled,
        false,
        &mut units,
        reserve(need),
    );
    assert!(!third.admission.granted());
    assert!(
        !third.denial.pending,
        "an Immediate claim waits for nothing"
    );
    assert!(governor.ledger(|ledger| {
        ledger
            .claim(optional.id())
            .is_some_and(|claim| !claim.waiting)
    }));
    drop(b);
    drop(optional);

    release_landing(units.target, held);
    landed(units.target, &tier);
    let epoch = governor.epoch(7);
    governor.signal_progress(7);
    assert!(governor.epoch(7) > epoch);
    clear_grow_reclaim_log();
    let retried = allocate(
        a.claim(),
        a_spec,
        Refusal::Waits,
        false,
        &mut units,
        reserve(need),
    );
    assert!(retried.admission.granted());
    assert_eq!(grow_reclaim_log(), vec![GrowReclaim::StoreWrite]);
    assert!(!ram(units.target, o1));
    assert!(disk_copy(units.target, o1).is_some());
}

/// A state of `class` on `tokens`, whose chain the prefix cache holds.
pub(crate) fn insert_state(inner: &mut Qwen35Inner, tokens: &[u32], class: StateClass) -> EntryId {
    let anchor = writable_anchor(&inner.config, window(inner));
    let id = inner
        .dflash2_states
        .store
        .insert(Lineage::text(tokens, 0), anchor, class)
        .expect("a whole-block state")
        .id;
    inner.sync_gdn_state_marks();
    id
}

/// A state on a cached `blocks`-block chain of `seed`.
pub(crate) fn park_blocks(
    inner: &mut Qwen35Inner,
    seed: u32,
    blocks: u32,
    class: StateClass,
) -> EntryId {
    let block_size = inner.paged_adapter.as_ref().expect("pool").block_size();
    let tokens = chain_tokens(seed, blocks, block_size);
    cache_chain(inner, &tokens);
    insert_state(inner, &tokens, class)
}

pub(crate) fn marked(inner: &Qwen35Inner) -> u32 {
    inner.paged_adapter.as_ref().expect("pool").marked_blocks()
}

fn owed(inner: &Qwen35Inner, id: EntryId) -> bool {
    disk_copy(inner, id).is_some_and(|copy| copy.chain_owed)
}

/// The KV object key of block `index` of `id`'s chain.
pub(crate) fn kv_key(
    inner: &Qwen35Inner,
    tier: &TestTier,
    id: EntryId,
    index: usize,
) -> ColdCacheKey {
    let entry = inner.dflash2_states.store.get(id).expect("the state");
    chain_keys(ColdGroup::Kv, tier.tier.kv_fingerprint, entry)[index]
}

pub(crate) fn drained(tier: &TestTier) {
    assert!(tier.tier.manager.drain(std::time::Duration::from_secs(10)));
}

/// A written state's chain stays marked until every block is on the tier,
/// its landed state notwithstanding (Splash `StateCache::marks`: a disk
/// state needs its KV as it is); the ledger sees the blocks in flight as
/// landing pages; once the chain lands its pages are takeable. A state
/// whose chain is on the tier already marks nothing at its write. A walk
/// that finds nothing in flight walks the chain once more from block 0
/// (Splash `Cache::persist`), so a block the tier lost below the cursor
/// is written again before the chain counts as landed.
/// Mutations CM1 (the mark goes when the state lands) → red; RW1 (no walk
/// from block 0) → red.
#[test]
fn disk_state_keeps_its_chain_until_it_lands() {
    use crate::engine::governor::governor::Governor;
    use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
    use crate::engine::governor::host::Pressure;
    let Some((mut inner, tier)) = fixture(0x5713_1701, "state-tier-chain") else {
        return;
    };
    let io = FakeIo::new(stats(G), host(32 * G, Pressure::Normal));
    let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
    inner.memory_registration = Some(governor.register_model(model(16 * G, 0)).expect("register"));
    let returning = || governor.ledger(|ledger| ledger.returning().landing_pages);
    let takeable = |inner: &Qwen35Inner| {
        inner
            .paged_adapter
            .as_ref()
            .expect("pool")
            .takeable_blocks()
    };

    let b = park_blocks(&mut inner, 61, 3, StateClass::Ordinary);
    assert_eq!(marked(&inner), 3);
    let before = takeable(&inner);
    assert_eq!(
        stage(&mut inner, b, MakeRoom::Evict, false),
        Staged::Started
    );
    assert!(owed(&inner, b));
    landed(&mut inner, &tier);
    inner.sync_gdn_state_marks();
    assert!(disk_copy(&inner, b).is_some_and(|copy| copy.landed));
    assert_eq!(marked(&inner), 3, "the landed state still needs its chain");
    let written = tier.tier.manager.stats().enqueued;
    assert!(!tick(&mut inner, Tick::Full { command_pages: 0 }));
    assert_eq!(tier.tier.manager.stats().enqueued, written + 3);
    assert!(owed(&inner, b));
    assert_eq!(landing_pages(&inner), 3);
    assert_eq!(returning(), 3, "the ledger sees the pages in flight");
    drained(&tier);
    assert!(tick(&mut inner, Tick::Full { command_pages: 0 }));
    assert!(!owed(&inner, b), "every block landed");
    assert_eq!(marked(&inner), 0);
    assert_eq!(takeable(&inner), before + 3);
    assert_eq!(landing_pages(&inner), 0);
    assert_eq!(returning(), 0);

    // Its first two blocks are on the tier: a state there marks nothing.
    let block_size = inner.paged_adapter.as_ref().expect("pool").block_size();
    let a = insert_state(
        &mut inner,
        &chain_tokens(61, 2, block_size),
        StateClass::Checkpoint,
    );
    assert_eq!(marked(&inner), 2);
    assert_eq!(
        stage(&mut inner, a, MakeRoom::Evict, false),
        Staged::Started
    );
    assert!(!owed(&inner, a));
    assert_eq!(marked(&inner), 0);
    landed(&mut inner, &tier);

    // Block 0 leaves the tier while the cursor stands past it.
    let c = park_blocks(&mut inner, 61, 4, StateClass::Ordinary);
    assert_eq!(
        stage(&mut inner, c, MakeRoom::Evict, false),
        Staged::Started
    );
    landed(&mut inner, &tier);
    assert_eq!(chain_step(&mut inner, Instant::now()).enqueued, 1);
    drained(&tier);
    assert!(poll_offload(&mut inner));
    assert!(tier.tier.manager.remove(&kv_key(&inner, &tier, c, 0)));
    let step = chain_step(&mut inner, Instant::now());
    assert_eq!(step.enqueued, 1, "block 0 is written again");
    assert!(!step.landed_all && owed(&inner, c));
    drained(&tier);
    assert!(tick(&mut inner, Tick::Full { command_pages: 0 }));
    assert!(!owed(&inner, c));
    assert!(
        tier.tier
            .manager
            .contains_in(&kv_key(&inner, &tier, c, 0), ColdGroup::Kv)
    );
}

/// Splash `Cache::pointFits`, `pollTransfers` and `Cache::persist`'s
/// `onDisk` check: a state the tier cannot hold with its chain leaves with
/// no copy and nothing written; a chain block whose write failed fails its
/// point (the disk-only state goes, its mark with it, and nothing is
/// written again); a state whose object left the tier fails its point.
/// A copy ahead the tier cannot hold is refused before anything is
/// written (Splash `Cache::persist`: waiting for room would hold back the
/// points after it).
/// Mutations PF1 (no `point_fits`) → red; FB1 (a failed block is
/// ignored) → red; OD1 (no `onDisk` check) → red.
#[test]
fn chain_refuses_and_fails_like_splash() {
    let query = waits_for_writes();
    let enqueued = |tier: &TestTier| tier.tier.manager.stats().enqueued;

    // The tier's limit holds one block object and no state.
    let Some((mut inner, _)) = fixture(0x5713_3201, "state-tier-fits") else {
        return;
    };
    let block_bytes = mlx_paged_attn::block_object_bytes(
        inner.paged_adapter.as_ref().expect("pool").layer_kv_pool(),
    );
    let small = test_tier("state-tier-fits-small", block_bytes);
    let lane = window(&inner);
    attach_tier(
        &mut inner.dflash2_states,
        &inner.config,
        inner.paged_adapter.as_ref().expect("pool"),
        &small,
        lane.nbytes(),
        lane.slots(),
    );
    let o = park(&mut inner, 71, StateClass::Ordinary);
    assert_eq!(take(&mut inner, &query, o).outcome, FreeOutcome::Met);
    assert!(inner.dflash2_states.store.get(o).is_none(), "no copy left");
    assert_eq!(enqueued(&small), 0);
    assert_eq!(marked(&inner), 0);
    // A copy ahead the tier cannot hold with its chain is refused.
    use crate::models::qwen3_5::dflash2_state::write_behind::{
        DELAY_SECONDS, MINIMUM_TOKENS, published, run,
    };
    let block_size = inner.paged_adapter.as_ref().expect("pool").block_size();
    let blocks = MINIMUM_TOKENS.div_ceil(block_size);
    let p = park_blocks(&mut inner, 74, blocks, StateClass::Ordinary);
    published(&mut inner.dflash2_states, p, blocks * block_size, 0.0);
    assert!(run(&mut inner, DELAY_SECONDS, u64::MAX, false));
    let disk = inner.dflash2_states.disk.as_ref().expect("tier");
    assert_eq!(disk.write_behind.snapshot().refused, 1);
    assert_eq!(enqueued(&small), 0);
    assert!(ram(&inner, p) && disk_copy(&inner, p).is_none());

    // A chain block's write fails.
    let Some((mut inner, tier)) = fixture(0x5713_3202, "state-tier-block-fails") else {
        return;
    };
    let o = park(&mut inner, 72, StateClass::Ordinary);
    assert_eq!(take(&mut inner, &query, o).outcome, FreeOutcome::Met);
    landed(&mut inner, &tier);
    std::fs::remove_dir_all(tier.tier.dir()).expect("remove the tier's directory");
    assert_eq!(chain_step(&mut inner, Instant::now()).enqueued, 1);
    assert!(!tier.tier.manager.drain(std::time::Duration::from_secs(10)));
    tick(&mut inner, Tick::Full { command_pages: 0 });
    let disk = inner.dflash2_states.disk.as_ref().expect("tier");
    assert_eq!(disk.counters.offload_failures, 1);
    assert!(disk.chain.is_none());
    assert!(inner.dflash2_states.store.get(o).is_none());
    assert_eq!(marked(&inner), 0);
    let written = enqueued(&tier);
    tick(&mut inner, Tick::Full { command_pages: 0 });
    assert_eq!(enqueued(&tier), written, "nothing is written again");

    // The state's object leaves the tier under its chain.
    let Some((mut inner, tier)) = fixture(0x5713_3203, "state-tier-state-gone") else {
        return;
    };
    let o = park(&mut inner, 73, StateClass::Ordinary);
    assert_eq!(take(&mut inner, &query, o).outcome, FreeOutcome::Met);
    landed(&mut inner, &tier);
    let key = disk_copy(&inner, o).expect("a disk copy").key;
    assert!(tier.tier.manager.remove(&key));
    let written = enqueued(&tier);
    let step = chain_step(&mut inner, Instant::now());
    assert!(step.failed && step.enqueued == 0);
    assert_eq!(enqueued(&tier), written);
    assert!(inner.dflash2_states.store.get(o).is_none());
    assert_eq!(marked(&inner), 0);
}

/// Splash `Cache::stateReclaim`: the walk lists a state as `reclaim` takes
/// it. One the tier cannot hold with its chain leaves with no write, so a
/// need of blocks reaches it now, its chain's blocks counted, and its take
/// frees them.
/// Mutation MU1 (the listing writes every RAM-only state) → red.
#[test]
fn a_state_the_tier_cannot_hold_frees_its_chain_now() {
    let Some((mut inner, _)) = fixture(0x5713_3204, "state-tier-unfit") else {
        return;
    };
    let block_bytes = mlx_paged_attn::block_object_bytes(
        inner.paged_adapter.as_ref().expect("pool").layer_kv_pool(),
    );
    let small = test_tier("state-tier-unfit-small", block_bytes);
    let lane = window(&inner);
    attach_tier(
        &mut inner.dflash2_states,
        &inner.config,
        inner.paged_adapter.as_ref().expect("pool"),
        &small,
        lane.nbytes(),
        lane.slots(),
    );
    let o = park(&mut inner, 76, StateClass::Ordinary);
    let blocks = VictimQuery {
        rights: TakeRights {
            units: true,
            waits_for_writes: true,
            certain_reuse: CertainReuse::All,
            ..TakeRights::default()
        },
        need: Need {
            pages: 1,
            host_refused: true,
            ..Need::default()
        },
    };
    let unit = unit_of(&mut inner, &blocks, o);
    assert_eq!((unit.stage, unit.later), (Stage::FreeNow, false));
    assert!(unit.pages > 0, "its chain's blocks are counted");
    let taken = ModelUnits {
        target: &mut inner,
        keep: Keep::default(),
    }
    .take(&blocks, &unit);
    assert_eq!(taken.outcome, FreeOutcome::Met);
    assert_eq!(taken.pages, unit.pages);
    assert!(inner.dflash2_states.store.get(o).is_none());
    assert_eq!(marked(&inner), 0);
    assert_eq!(small.tier.manager.stats().enqueued, 0);
}

/// The governor's status carries the state tier (Splash `/status`
/// `state` disk counters and `disk`): after a reclaim wrote one state, one
/// offload and one disk entry with its bytes, a temporary tier and its
/// limit.
/// Mutation ST1 (the tier's counters are not copied into the status) → red.
#[test]
fn status_reports_the_disk_tier() {
    use crate::engine::governor::governor::Governor;
    use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
    use crate::engine::governor::host::Pressure;
    use crate::engine::governor::status::{MemoryGovernorStatus, status};
    use crate::models::qwen3_5::dflash2_turn::state_status;
    let Some((mut inner, tier)) = fixture(0x5713_2601, "state-tier-status") else {
        return;
    };
    let query = waits_for_writes();
    let c = park(&mut inner, 33, StateClass::Checkpoint);
    assert_eq!(take(&mut inner, &query, c).outcome, FreeOutcome::Met);
    landed(&mut inner, &tier);
    let governor = Governor::new(FakeIo::new(stats(6 * G), host(32 * G, Pressure::Normal)));
    let _registration = governor.register_model(model(16 * G, 0)).expect("register");
    governor.publish_dflash2_states(7, state_status(&inner));
    let js = MemoryGovernorStatus::from(status(&governor).expect("registered")).dflash2_states;
    assert_eq!((js.offloads, js.disk_entries), (1.0, 1.0));
    assert!(js.disk_bytes > 0.0);
    assert_eq!(js.disk.mode, "temp");
    assert_eq!(js.disk.limit_bytes, (1u64 << 30) as f64);
}

/// A reclaim's chain frees memory once it lands, so it takes the cursor
/// from a copy ahead (Splash `Cache::persist` yields while a reclaim's write
/// is in flight): with P's copy ahead holding the cursor and its chain owed,
/// the next wait tick writes R's chain and none of P's.
/// Mutation CY1 (the cursor stays on the copy ahead) → red.
#[test]
fn a_reclaim_chain_goes_before_a_copy_ahead() {
    let Some((mut inner, tier)) =
        fixture_on(0x5713_1702, 16, 16, test_tier("state-tier-yield", 1 << 30))
    else {
        return;
    };
    let p = park_blocks(&mut inner, 62, 3, StateClass::Ordinary);
    assert_eq!(
        stage(&mut inner, p, MakeRoom::Refuse, true),
        Staged::Started
    );
    landed(&mut inner, &tier);
    let r = park_blocks(&mut inner, 63, 3, StateClass::Ordinary);
    assert_eq!(
        stage(&mut inner, r, MakeRoom::Evict, false),
        Staged::Started
    );
    landed(&mut inner, &tier);
    assert!(owed(&inner, p) && owed(&inner, r));
    let disk = inner.dflash2_states.disk.as_mut().expect("tier");
    let entry = inner.dflash2_states.store.get(p).expect("p");
    disk.chain = Some(ChainCursor {
        id: p,
        keys: chain_keys(ColdGroup::Kv, disk.tier.kv_fingerprint, entry),
        landed: 0,
        in_flight: HashSet::new(),
        room: MakeRoom::Refuse,
    });
    tick(
        &mut inner,
        Tick::Wait {
            deadline: Instant::now(),
        },
    );
    drained(&tier);
    let on_tier = |inner: &Qwen35Inner, id: EntryId| {
        (0..3)
            .filter(|&index| {
                tier.tier
                    .manager
                    .contains_in(&kv_key(inner, &tier, id, index), ColdGroup::Kv)
            })
            .count()
    };
    assert_eq!((on_tier(&inner, r), on_tier(&inner, p)), (3, 0));
    assert!(landing_pages(&inner) > 0 || !owed(&inner, r));
}

/// Splash `Cache::adopt` ends with `fitDiskLimit`: a load whose limit is
/// lower than what the last process left takes the oldest copies first,
/// so it takes back fewer states and the tier fits its limit.
/// Mutation FQ1 (no fit at adoption) → red.
#[test]
fn adoption_fits_a_lower_limit() {
    let Some((mut inner, tier)) = fixture_on(
        0x5713_2401,
        16,
        16,
        persistent_test_tier("state-tier-adopt-fit", 1 << 30),
    ) else {
        return;
    };
    for seed in [64, 65] {
        let id = park_blocks(&mut inner, seed, 2, StateClass::Checkpoint);
        assert_eq!(reclaim(&mut inner, id), Reclaimed::Written);
        for _ in 0..8 {
            drained(&tier);
            tick(&mut inner, Tick::Full { command_pages: 0 });
            if disk_copy(&inner, id).is_some_and(|copy| copy.landed && !copy.chain_owed) {
                break;
            }
        }
        assert!(disk_copy(&inner, id).is_some_and(|copy| copy.landed && !copy.chain_owed));
    }
    let total = tier.tier.manager.total_bytes();
    drop(inner);
    let tier = tier.reopen("state-tier-adopt-fit", total - 1);
    let Some((mut inner, tier)) = fixture_on(0x5713_2401, 16, 16, tier) else {
        return;
    };
    assert_eq!(adopt_all(&mut inner), 1);
    assert!(tier.tier.manager.total_bytes() < total);
}

/// Splash `SlotFile::store`: a write that fails for anything but room
/// stops the tier, so a copy ahead whose write failed is not encoded and
/// written again at every landing; it is refused once.
/// Mutation WS1 (the tier takes writes after the failure) → red.
#[test]
fn a_failed_write_stops_the_tier() {
    use crate::models::qwen3_5::dflash2_state::write_behind::{
        DELAY_SECONDS, MINIMUM_TOKENS, published, run,
    };
    let Some((mut inner, tier)) =
        fixture_on(0x5713_1503, 16, 16, test_tier("state-tier-stop", 1 << 30))
    else {
        return;
    };
    let block_size = inner.paged_adapter.as_ref().expect("pool").block_size();
    let blocks = MINIMUM_TOKENS.div_ceil(block_size);
    let p = park_blocks(&mut inner, 66, blocks, StateClass::Ordinary);
    published(&mut inner.dflash2_states, p, blocks * block_size, 0.0);
    std::fs::remove_dir_all(tier.tier.dir()).expect("remove the tier's directory");
    for _ in 0..4 {
        run(&mut inner, DELAY_SECONDS, u64::MAX, false);
        let _ = tier.tier.manager.drain(std::time::Duration::from_secs(10));
        poll_offload(&mut inner);
    }
    let disk = inner.dflash2_states.disk.as_ref().expect("tier");
    assert_eq!(disk.counters.offloads, 1, "one write, never encoded again");
    assert!(!disk.writable());
    assert_eq!(disk.write_behind.snapshot().refused, 1);
    assert!(ram(&inner, p) && disk_copy(&inner, p).is_none());
}

/// Splash `DiskBudget::reportFull`: a volume without room above the
/// writer's reserve refuses a write before its state is encoded: a reclaim
/// leaves with no copy and a copy ahead is refused, nothing written.
/// Mutation FS1 (`point_fits` ignores the volume's free space) → red.
#[test]
fn a_full_volume_refuses_before_writing() {
    use crate::models::qwen3_5::dflash2_state::write_behind::{
        DELAY_SECONDS, MINIMUM_TOKENS, published, run,
    };
    let tier = test_tier_with("state-tier-full", 1 << 30, 8, u64::MAX / 2);
    let Some((mut inner, tier)) = fixture_on(0x5713_1504, 16, 16, tier) else {
        return;
    };
    let o = park(&mut inner, 67, StateClass::Ordinary);
    assert_eq!(reclaim(&mut inner, o), Reclaimed::Erased);
    let block_size = inner.paged_adapter.as_ref().expect("pool").block_size();
    let blocks = MINIMUM_TOKENS.div_ceil(block_size);
    let p = park_blocks(&mut inner, 68, blocks, StateClass::Ordinary);
    published(&mut inner.dflash2_states, p, blocks * block_size, 0.0);
    assert!(run(&mut inner, DELAY_SECONDS, u64::MAX, false));
    let disk = inner.dflash2_states.disk.as_ref().expect("tier");
    assert_eq!(disk.write_behind.snapshot().refused, 1);
    assert_eq!(disk.counters.offloads, 0);
    assert_eq!(tier.tier.manager.stats().enqueued, 0);
}

/// Drains and ticks until `id`'s copy and its chain are on the tier.
fn settle(inner: &mut Qwen35Inner, tier: &TestTier, id: EntryId) {
    for _ in 0..8 {
        drained(tier);
        tick(inner, Tick::Full { command_pages: 0 });
        if disk_copy(inner, id).is_some_and(|copy| copy.landed && !copy.chain_owed) {
            return;
        }
    }
    panic!("the point never landed whole");
}

/// Splash `Cache::freeDiskSpace`: the tier makes room only by evicting the
/// state with its copy, so a disk-only state whose object a quota eviction
/// cleared leaves the store, and one with a RAM copy keeps RAM and loses
/// only its disk copy; no entry outlives its object.
/// Mutation EV1 (`poll_offload` skips the tier's evictions) → red.
#[test]
fn a_quota_eviction_takes_the_disk_only_state() {
    let Some((mut probe, probe_tier)) = fixture_on(
        0x5713_4101,
        16,
        16,
        test_tier("state-tier-evict-probe", 1 << 30),
    ) else {
        return;
    };
    let id = park(&mut probe, 70, StateClass::Checkpoint);
    assert_eq!(reclaim(&mut probe, id), Reclaimed::Written);
    settle(&mut probe, &probe_tier, id);
    let point = probe_tier.tier.manager.total_bytes();
    drop(probe);
    drop(probe_tier);

    let limit = 2 * point + point / 2;
    let Some((mut inner, tier)) =
        fixture_on(0x5713_4101, 16, 16, test_tier("state-tier-evict", limit))
    else {
        return;
    };
    let a = park(&mut inner, 70, StateClass::Checkpoint);
    assert_eq!(reclaim(&mut inner, a), Reclaimed::Written);
    settle(&mut inner, &tier, a);
    let p = park(&mut inner, 71, StateClass::Ordinary);
    assert_eq!(
        stage(&mut inner, p, MakeRoom::Refuse, true),
        Staged::Started
    );
    settle(&mut inner, &tier, p);
    assert!(!ram(&inner, a) && ram(&inner, p));
    let key = |inner: &Qwen35Inner, id: EntryId| disk_copy(inner, id).expect("a copy").key;
    let (a_key, p_key) = (key(&inner, a), key(&inner, p));
    let on_tier = |key: &ColdCacheKey| tier.tier.manager.contains_in(key, ColdGroup::DFlash2State);

    let mut last = p;
    for seed in 72..80 {
        if !on_tier(&a_key) && !on_tier(&p_key) {
            break;
        }
        last = park(&mut inner, seed, StateClass::Checkpoint);
        assert_eq!(reclaim(&mut inner, last), Reclaimed::Written);
        settle(&mut inner, &tier, last);
    }
    assert!(!on_tier(&a_key) && !on_tier(&p_key), "both objects went");
    drained(&tier);
    poll_offload(&mut inner);

    assert!(
        inner.dflash2_states.store.get(a).is_none(),
        "the disk-only state left with its object"
    );
    assert!(ram(&inner, p) && disk_copy(&inner, p).is_none());
    assert!(disk_copy(&inner, last).is_some_and(|copy| copy.landed));
    let status = crate::models::qwen3_5::dflash2_turn::state_status(&inner);
    let indexed = inner
        .dflash2_states
        .store
        .entries()
        .filter(|entry| entry.disk().is_some())
        .filter(|entry| on_tier(&entry.disk().expect("a copy").key))
        .count() as u64;
    assert_eq!(
        status.disk_entries, indexed,
        "every disk entry has its object"
    );
}

/// Splash `DiskSpace::measure` at every control pass: a Full tick of a
/// persistent tier that grew past the root's limit with an idle sibling
/// takes the sibling.
/// Mutation Q1b (the Full tick does not measure) → red.
#[test]
fn a_full_tick_measures_the_cache_root() {
    let Some((mut probe, probe_tier)) = fixture_on(
        0x5713_4201,
        16,
        16,
        persistent_test_tier("state-tier-measure-probe", 1 << 30),
    ) else {
        return;
    };
    let id = park(&mut probe, 81, StateClass::Checkpoint);
    assert_eq!(reclaim(&mut probe, id), Reclaimed::Written);
    settle(&mut probe, &probe_tier, id);
    let point = probe_tier.tier.manager.total_bytes();
    drop(probe);
    drop(probe_tier);

    let root = std::env::temp_dir().join(format!(
        "mlx-state-tier-measure-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let sibling = root.join("cache").join("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    std::fs::create_dir_all(&sibling).expect("a sibling");
    std::fs::File::create(sibling.join("object.safetensors"))
        .and_then(|file| file.set_len(point))
        .expect("the sibling's object");
    let limit = point + point / 2;
    let tier = TestTier {
        tier: tier_at(&root, "state-tier-measure", limit, true),
        root,
    };
    assert!(sibling.exists(), "the root fits its limit at open");
    let Some((mut inner, tier)) = fixture_on(0x5713_4201, 16, 16, tier) else {
        return;
    };
    let id = park(&mut inner, 81, StateClass::Checkpoint);
    assert_eq!(reclaim(&mut inner, id), Reclaimed::Written);
    settle(&mut inner, &tier, id);
    tick(&mut inner, Tick::Full { command_pages: 0 });
    assert!(!sibling.exists(), "the tier grew: the idle sibling went");
}

/// Codex disk.rs:1480: a draft replaced between its load and the identity
/// read keys nothing by the replacement.
#[test]
fn a_draft_changed_during_load_keeps_the_tier_temporary() {
    if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device() {
        return;
    }
    let mut inner = crate::models::qwen3_5::dflash2_decode::tests::tiny_dflash_inner_with_config(
        0x5713_1480,
        |_| {},
    );
    install_pool(&mut inner, 8, 8, passing_probe()).expect("pool");
    let root = std::env::temp_dir().join(format!(
        "mlx-state-tier-draft-guard-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let target = root.join("target");
    let draft = root.join("draft");
    std::fs::create_dir_all(&target).expect("target dir");
    std::fs::create_dir_all(&draft).expect("draft dir");
    std::fs::create_dir_all(root.join("tmp")).expect("temp root");
    std::fs::write(target.join("model.safetensors"), b"target").expect("target shard");
    std::fs::write(draft.join("model.safetensors"), b"draft").expect("draft shard");
    std::fs::write(draft.join("config.json"), b"{}").expect("draft config");
    let options = DiskTierOptions {
        cache_disk: CacheDisk::Bytes(1 << 30),
        persistent: true,
        cache_dir: Some(root.join("cache")),
        temp_root: Some(root.join("tmp")),
    };
    let weights = WeightsResident::for_test(1, 4096);
    let files = CheckpointFiles::Dir(&target);
    let open = |replace_draft: bool| {
        let mut guard = CheckpointLoadGuard::before_read(files, true);
        guard.record_read();
        let draft_files = DraftLoadGuard::before_read(&draft, true);
        if replace_draft {
            std::fs::write(draft.join("model.safetensors"), b"replacement draft")
                .expect("replacement");
        }
        let load = TierLoad {
            family: "qwen3_5",
            config_json: b"{}".to_vec(),
            dense: &inner.config,
            activation: DType::BFloat16,
            draft: Some(DraftTier {
                model: inner.dflash2.as_ref().expect("a draft"),
                files: &draft_files,
            }),
            adapter: inner.paged_adapter.as_ref().expect("pool"),
            files,
            weights: &weights,
            guard: &guard,
        };
        let (_, disk) = open_state_disk(&load, &options, inner.model_id).expect("a tier");
        let persistent = disk.tier.persistent;
        disk.tier.close(Instant::now());
        persistent
    };
    assert!(open(false), "an unchanged draft keys a persistent tier");
    assert!(
        !open(true),
        "a draft replaced during load keeps the tier temporary"
    );
    let _ = std::fs::remove_dir_all(&root);
}
