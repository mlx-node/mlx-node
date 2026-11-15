//! The stored states' disk tier on the paged fixture: the model's own
//! persistent tier under a temporary cache root, the fixture's fingerprint,
//! and a restart that drops the model and builds a new one over the same
//! namespace reopened.
use super::prefix_reuse::{
    GRID, cold_on, cycles, finish, fork, greedy, prompt, reuse_inner, reuse_over, speculative_turn,
    turn,
};
use super::*;
use crate::disk_tier::DiskTierOptions;
use crate::engine::backend::PagedBackend;
use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
use mlx_paged_attn::{ColdCacheFingerprint, ColdCacheManager, ColdGroup};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A cache root for the tier's namespaces and a temporary root that must
/// stay empty.
struct Root {
    cache: Temp,
    tmp: Temp,
}

impl Root {
    fn new() -> Self {
        Self {
            cache: Temp::new(),
            tmp: Temp::new(),
        }
    }

    /// The persistent tier of a 1 GiB cache under this root.
    fn options(&self) -> DiskTierOptions {
        DiskTierOptions {
            temp_root: Some(self.tmp.0.clone()),
            ..DiskTierOptions::from_load(
                Some("1073741824"),
                true,
                Some(self.cache.0.to_string_lossy().into_owned()),
            )
            .unwrap()
        }
    }

    /// The namespace directories under the cache root.
    fn namespaces(&self) -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = std::fs::read_dir(&self.cache.0)
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect();
        out.sort();
        out
    }

    /// The one namespace directory under the cache root.
    fn namespace(&self) -> PathBuf {
        let mut namespaces = self.namespaces();
        assert_eq!(namespaces.len(), 1, "{namespaces:?}");
        namespaces.remove(0)
    }

    fn temp_is_empty(&self) -> bool {
        std::fs::read_dir(&self.tmp.0).unwrap().next().is_none()
    }
}

/// The fingerprint a load of the fixture under checkpoint `identity` keys
/// its objects with.
fn fixture_fingerprint(inner: &Inner, identity: &str) -> ColdCacheFingerprint {
    prefix_disk::fingerprint(
        identity,
        None,
        &inner.decoder.config,
        inner.decoder.paged.as_ref().unwrap().layer_kv_pool(),
        inner.decoder.prefill_slice_size(),
        &inner.prefix_states.open,
        inner.decoder.host_prefill_scratch,
        inner.has_mtp,
    )
    .unwrap()
}

/// One open of `inner`'s disk tier under `root` as a load of checkpoint
/// `identity` opens it, its writer queueing `depth` jobs and keeping
/// `reserve` bytes free.
fn open_once(inner: &mut Inner, root: &Root, identity: &str, (depth, reserve): (usize, u64)) {
    crate::disk_tier::TEST_WRITER.with(|writer| writer.set(Some((depth, reserve))));
    inner.open_prefix_disk(&root.options(), Some(identity), None);
    crate::disk_tier::TEST_WRITER.with(|writer| writer.set(None));
}

/// `inner` with its disk tier open under `root`. The last tier of the same
/// namespace lets go of its lock only when its writer thread exits, after
/// its close returned.
fn on_disk_with(mut inner: Inner, root: &Root, identity: &str, writer: (usize, u64)) -> Inner {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        open_once(&mut inner, root, identity, writer);
        if inner.disk.is_some() {
            return inner;
        }
        assert!(Instant::now() < deadline, "the namespace stays held");
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn on_disk_as(inner: Inner, root: &Root, identity: &str) -> Inner {
    on_disk_with(inner, root, identity, (8, 0))
}

fn on_disk(inner: Inner, root: &Root) -> Inner {
    on_disk_as(inner, root, "fixture")
}

fn disk(inner: &Inner) -> &prefix_disk::Disk {
    inner.disk.as_ref().unwrap()
}

/// The tier's store.
fn store(inner: &Inner) -> Arc<ColdCacheManager> {
    Arc::clone(&disk(inner).tier().manager)
}

/// Everything parked and copied out lands: the parker hands it over, the
/// store's writer commits it.
fn settle(inner: &Inner) {
    disk(inner).flush(Instant::now() + Duration::from_secs(20));
    assert!(store(inner).drain(Duration::from_secs(20)));
}

/// The object files of `group` in every namespace under `root`.
fn files(root: &Root, group: &str) -> Vec<PathBuf> {
    let suffix = format!(".{group}.safetensors");
    let mut out: Vec<PathBuf> = root
        .namespaces()
        .iter()
        .flat_map(|dir| std::fs::read_dir(dir).unwrap().flatten())
        .map(|entry| entry.path())
        .filter(|path| path.to_string_lossy().ends_with(&suffix))
        .collect();
    out.sort();
    out
}

fn objects(root: &Root) -> usize {
    root.namespaces()
        .iter()
        .flat_map(|dir| std::fs::read_dir(dir).unwrap().flatten())
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .ends_with(".safetensors")
        })
        .count()
}

fn count(counter: &std::sync::atomic::AtomicU64) -> u64 {
    counter.load(Ordering::Relaxed)
}

/// The first conversation of a store: a 100-token prompt on sequence 1 (its
/// replay state at row 96) under checkpoint `identity`, written out; the
/// model and its tier are gone after.
fn first_conversation_as(root: &Root, make: &dyn Fn() -> Inner, identity: &str) -> Vec<u32> {
    let mut inner = on_disk_as(make(), root, identity);
    let a = prompt(100, 1);
    assert_eq!(turn(&mut inner, 1, &a).0, 0);
    finish(&mut inner);
    settle(&inner);
    a
}

fn first_conversation(root: &Root, make: &dyn Fn() -> Inner) -> Vec<u32> {
    first_conversation_as(root, make, "fixture")
}

/// The keys of `tokens`' first `blocks` blocks in `group`, as the fixture's
/// plain 32-row passes key them.
fn keys_of(
    inner: &Inner,
    tokens: &[u32],
    group: ColdGroup,
    blocks: usize,
) -> Vec<mlx_paged_attn::ColdCacheKey> {
    let ends = prefix_state::pass_ends(tokens.len(), &[], |_| GRID);
    let keys = prefix_state::pass_keys(tokens, &ends, GRID);
    prefix_disk::chain(
        group,
        fixture_fingerprint(inner, "fixture"),
        tokens,
        &keys,
        0,
        blocks,
    )
}

/// A restarted model resumes a follow-up at the state the first model left
/// on disk: it reports row 96 cached and gives a cold model's logits and
/// greedy steps bit for bit, on a bf16 pool and on an int8 pool with the
/// wide head; the store counts no corruption.
/// Mutation: `encode_cell` writes no recurrent matrices → red (the state
/// does not read back, the turn is cold).
#[test]
fn a_restart_resumes_from_the_disk_state_and_matches_cold() {
    use crate::transformer::KvFormat;
    let _host = memory::scoped_fixture_host();
    let wide = Temp::new();
    write_wide_head_fixture(&wide.0);
    let cases: [(&str, &dyn Fn() -> Inner); 2] = [
        ("bf16", &reuse_inner),
        ("int8", &|| {
            reuse_over(wide_head_inner(&wide.0, KvFormat::Int8))
        }),
    ];
    for (name, make) in cases {
        let root = Root::new();
        let a = first_conversation(&root, make);
        assert_eq!(files(&root, "hybrid_cell").len(), 1, "{name}");
        let mut inner = on_disk(make(), &root);
        let b = fork(&a, 98, 110, 2);
        let (cached, logits) = turn(&mut inner, 1, &b);
        assert_eq!(cached, 96, "{name}");
        assert_eq!(count(&inner.telemetry.disk_restored), 1, "{name}");
        let steps = greedy(&mut inner, &logits, 8);
        let (want_logits, want_steps) = cold_on(make(), &b);
        assert_eq!(logits, want_logits, "{name}: first logits");
        assert_eq!(steps, want_steps, "{name}: greedy steps");
        assert_eq!(store(&inner).stats().corruptions, 0, "{name}");
    }
}

/// A restarted model's speculative turn drafts as a cold one: speculation
/// begins at the row restored from disk, the draft history after the
/// prefill is the cold one's bit for bit, and six greedy cycles give the
/// same drafts, kept rows and verify logits.
/// Mutation: the segments are read back in reverse order → red.
#[test]
fn a_restarted_draft_turn_matches_cold() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let a = prompt(100, 1);
    {
        let mut inner = on_disk(reuse_inner(), &root);
        let (_, begun, _) = speculative_turn(&mut inner, &a);
        assert!(begun);
        finish(&mut inner);
        inner.release_scheduled_speculation(1);
        settle(&inner);
    }
    assert_eq!(files(&root, "hybrid_draft").len(), 2);
    let mut inner = on_disk(reuse_inner(), &root);
    let b = fork(&a, 98, 104, 2);
    let (cached, begun, logits) = speculative_turn(&mut inner, &b);
    assert_eq!((cached, begun), (96, true));
    let warm = inner.mtp.draft_bits(1).unwrap();
    let mut cold = reuse_inner();
    let (cached, begun, cold_logits) = speculative_turn(&mut cold, &b);
    assert_eq!((cached, begun), (0, true));
    assert_eq!(logits, cold_logits);
    assert_eq!(warm, cold.mtp.draft_bits(1).unwrap());
    assert_eq!(
        cycles(&mut inner, &logits, 6),
        cycles(&mut cold, &cold_logits, 6)
    );
}

/// A state object cut short, one with a flipped payload byte and one of
/// another checkpoint under this state's name are each refused: the
/// restarted turn prefills as a cold one and the file is gone.
/// Mutation: the sidecar read skips its fingerprint comparison → red (the
/// other checkpoint's state is taken, 96 cached, the file stays).
#[test]
fn a_corrupt_or_foreign_state_is_refused_and_removed() {
    let _host = memory::scoped_fixture_host();
    for case in ["truncated", "checksum", "identity"] {
        let root = Root::new();
        let a = first_conversation(&root, &reuse_inner);
        let state = files(&root, "hybrid_cell").remove(0);
        let mut bytes = std::fs::read(&state).unwrap();
        match case {
            "truncated" => bytes.truncate(bytes.len() / 2),
            "checksum" => {
                let last = bytes.len() - 1;
                bytes[last] ^= 0x40;
            }
            _ => {
                let other = Root::new();
                first_conversation_as(&other, &reuse_inner, "another checkpoint");
                bytes = std::fs::read(files(&other, "hybrid_cell").remove(0)).unwrap();
            }
        }
        std::fs::write(&state, &bytes).unwrap();
        let mut inner = on_disk(reuse_inner(), &root);
        let b = fork(&a, 98, 110, 2);
        let (cached, logits) = turn(&mut inner, 1, &b);
        assert_eq!(cached, 0, "{case}");
        assert_eq!(count(&inner.telemetry.disk_restored), 0, "{case}");
        assert!(!state.exists(), "{case}: the refused object is removed");
        assert_eq!(logits, cold_on(reuse_inner(), &b).0, "{case}");
    }
}

/// A state on disk whose chain lost a KV block is dropped at the restart's
/// lookup: the turn prefills as a cold one, the state object is removed.
/// Mutation: the candidate skips the KV check → red (the state stays).
#[test]
fn a_state_with_a_missing_block_is_dropped() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let a = first_conversation(&root, &reuse_inner);
    let kv = keys_of(&reuse_inner(), &a, ColdGroup::Kv, 3);
    let block = root
        .namespace()
        .join(format!("{}.safetensors", kv[1].to_hex()));
    assert!(block.exists());
    std::fs::remove_file(&block).unwrap();
    let mut inner = on_disk(reuse_inner(), &root);
    let b = fork(&a, 98, 110, 2);
    let (cached, logits) = turn(&mut inner, 1, &b);
    assert_eq!(cached, 0);
    assert_eq!(count(&inner.telemetry.dropped_chain), 1);
    assert!(files(&root, "hybrid_cell").is_empty());
    assert_eq!(logits, cold_on(reuse_inner(), &b).0);
}

/// A conversation that goes on inside the delay writes only its newest
/// state: the parked one is replaced.
/// Mutation: a parked state is due at once → red (two state objects).
#[test]
fn a_conversation_that_went_on_writes_one_state() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    let a = prompt(70, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    let b = fork(&a, 68, 100, 3);
    assert_eq!(turn(&mut inner, 1, &b).0, 64);
    finish(&mut inner);
    assert_eq!(count(&inner.telemetry.unneeded), 1);
    settle(&inner);
    let cells = keys_of(&inner, &b, ColdGroup::HybridCell, 3);
    assert_eq!(files(&root, "hybrid_cell").len(), 1);
    assert!(store(&inner).contains_in(&cells[2], ColdGroup::HybridCell));
    assert_eq!(count(&inner.telemetry.durable), 1);
}

/// A state is written only after every object of its chain: while the
/// block and segment copies are held back the parked state stays out of
/// the store; once they land, it lands.
/// Mutation: the hand-over ignores the chain → red (the state is written
/// while its blocks are not).
#[test]
fn a_state_waits_for_its_chain() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    disk(&inner).hold(true);
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    disk(&inner).flush(Instant::now() + Duration::from_millis(300));
    assert!(store(&inner).drain(Duration::from_secs(20)));
    assert!(files(&root, "hybrid_cell").is_empty());
    assert_eq!(objects(&root), 0);
    assert_eq!(count(&inner.telemetry.waiting), 1);
    disk(&inner).hold(false);
    settle(&inner);
    assert_eq!(files(&root, "hybrid_cell").len(), 1);
    assert_eq!(count(&inner.telemetry.waiting), 0);
}

/// Turns never wait for the disk: with the writer held, two turns run and
/// end, their copies wait, nothing is written.
/// Mutation: `park` waits for the hand-over (a flush of 30 s) → red.
#[test]
fn a_turn_never_waits_for_the_writer() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    disk(&inner).hold(true);
    let started = Instant::now();
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    turn(&mut inner, 2, &prompt(100, 2));
    finish(&mut inner);
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "{:?}",
        started.elapsed()
    );
    assert!(count(&inner.telemetry.owed) > 0);
    assert_eq!(objects(&root), 0);
    disk(&inner).hold(false);
    settle(&inner);
    assert_eq!(files(&root, "hybrid_cell").len(), 2);
}

/// States on disk stay inside their byte budget, the oldest going first:
/// with room for two and a half states, a third conversation's state
/// replaces the first one's.
/// Mutation: the hand-over evicts nothing → red (three states).
#[test]
fn the_state_budget_drops_the_oldest() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let without_head = || {
        let mut inner = reuse_inner();
        inner.has_mtp = false;
        inner
    };
    let mut inner = on_disk(without_head(), &root);
    let manager = store(&inner);
    let groups = [ColdGroup::HybridCell, ColdGroup::HybridDraft];
    let mut cells = Vec::new();
    for seq in 1..=3u32 {
        let tokens = prompt(100, seq);
        assert_eq!(turn(&mut inner, seq, &tokens).0, 0);
        finish(&mut inner);
        settle(&inner);
        if seq == 1 {
            let unit = manager.group_bytes(&groups);
            assert!(unit > 0);
            disk(&inner).set_budget(unit * 5 / 2);
        }
        cells.push(keys_of(&inner, &tokens, ColdGroup::HybridCell, 3)[2]);
    }
    assert_eq!(files(&root, "hybrid_cell").len(), 2);
    assert!(!manager.contains_in(&cells[0], ColdGroup::HybridCell));
    assert!(manager.contains_in(&cells[1], ColdGroup::HybridCell));
    assert!(manager.contains_in(&cells[2], ColdGroup::HybridCell));
}

/// A flush stops waiting once nothing more can land: with every write
/// failing (the store keeps a reserve no disk has) a turn's state can never
/// land, and the flush, a second one and the unload each return long before
/// their deadline, leaving the store's own drain its time.
/// Mutation: the flush waits until no state is parked → red.
#[test]
fn a_flush_does_not_wait_for_a_state_that_cannot_land() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk_with(reuse_inner(), &root, "fixture", (8, 1 << 62));
    let manager = store(&inner);
    turn(&mut inner, 1, &prompt(100, 1));
    finish(&mut inner);
    for round in ["first", "second"] {
        let started = Instant::now();
        disk(&inner).flush(started + Duration::from_secs(4));
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{round} flush: {:?}",
            started.elapsed()
        );
    }
    assert_eq!(count(&inner.telemetry.waiting), 1);
    assert_eq!(count(&inner.telemetry.durable), 0);
    assert!(manager.stats().write_errors > 0);
    let started = Instant::now();
    inner.disk = None;
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(objects(&root), 0);
}

/// A block or segment that finds no writer slot stays first in line: with
/// a one-slot writer queue and a single try per offer, the copies of a
/// conversation's chain keep coming back, and still the whole chain and its
/// state land.
/// Mutation: an object that finds no slot is dropped → red (the chain has
/// a hole, the state never lands).
#[test]
fn a_copy_that_finds_no_writer_slot_stays_in_line() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk_with(reuse_inner(), &root, "fixture", (1, 0));
    disk(&inner).set_slot_wait(Duration::ZERO);
    disk(&inner).hold(true);
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    disk(&inner).hold(false);
    settle(&inner);
    assert_eq!(files(&root, "hybrid_cell").len(), 1);
    assert_eq!(count(&inner.telemetry.durable), 1);
    assert_eq!(count(&inner.telemetry.owed), 0);
}

/// Every lane's conversation can park its state, and a state past them is
/// refused before it is copied out; a parked state's bytes are within the
/// one state the memory plan books for each.
/// Mutation: the parcel count is checked after the copy → red (the refused
/// state was copied out).
#[test]
fn every_lane_parks_and_a_refused_state_is_never_copied() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    disk(&inner).hold(true);
    let lanes = prefix_disk::PARCELS as u32;
    for seq in 1..=lanes + 1 {
        turn(&mut inner, seq, &prompt(70, seq));
        inner.finalize_paged_turn(false, 0);
    }
    assert_eq!(count(&inner.telemetry.waiting), u64::from(lanes));
    assert_eq!(count(&inner.telemetry.refused), 1);
    assert_eq!(disk(&inner).encoded, prefix_disk::PARCELS);
    let booked = memory::encoded_state_bytes(&inner.decoder.config, inner.has_mtp);
    let parked = disk(&inner).parked_bytes();
    assert_eq!(parked.len(), prefix_disk::PARCELS);
    assert!(
        parked.iter().all(|&bytes| bytes <= booked),
        "{parked:?} > {booked}"
    );
}

/// Decode steps copy blocks out at most once per interval: a copy waits for
/// the GPU.
/// Mutation: every step copies → red.
#[test]
fn decode_steps_copy_out_at_most_once_per_interval() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    for _ in 0..3 {
        inner.top_up_disk();
    }
    assert_eq!(disk(&inner).step_runs, 1);
    std::thread::sleep(Duration::from_millis(300));
    inner.top_up_disk();
    assert_eq!(disk(&inner).step_runs, 2);
}

/// After a command reset the next turn prefills the purged prefix again
/// although its state is on disk; a later turn restores it.
/// Mutation: the reset leaves the disk on for the next turn → red (96
/// cached).
#[test]
fn a_reset_does_not_restore_from_disk_next() {
    use crate::engine::backend::ResetScope;
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let a = first_conversation(&root, &reuse_inner);
    let mut inner = on_disk(reuse_inner(), &root);
    inner.reset_caches(ResetScope::Command).unwrap();
    let b = fork(&a, 98, 110, 2);
    assert_eq!(turn(&mut inner, 1, &b).0, 0);
    finish(&mut inner);
    inner.reset_caches(ResetScope::Command).unwrap();
    assert_eq!(turn(&mut inner, 1, &prompt(60, 5)).0, 0);
    finish(&mut inner);
    assert_eq!(turn(&mut inner, 1, &b).0, 96);
    assert_eq!(count(&inner.telemetry.disk_restored), 1);
}

/// The fingerprint binds what changes a state's bits: the checkpoint, the
/// prefill window, the draft head, the assistant opening, the scratch that
/// halves the passes; the same load gives the same one.
/// Mutation: the window is left out → red.
#[test]
fn the_fingerprint_binds_the_math() {
    let _host = memory::scoped_fixture_host();
    let inner = reuse_inner();
    let config = &inner.decoder.config;
    let pool = inner.decoder.paged.as_ref().unwrap().layer_kv_pool();
    let of = |identity: &str, window: usize, open: &[u32], scratch: Option<u64>, head: bool| {
        prefix_disk::fingerprint(identity, None, config, pool, window, open, scratch, head).unwrap()
    };
    let base = of("a", 32, &[], None, true);
    assert_eq!(base, of("a", 32, &[], None, true));
    assert_eq!(base, fixture_fingerprint(&inner, "a"));
    for (name, other) in [
        ("checkpoint", of("b", 32, &[], None, true)),
        ("window", of("a", 64, &[], None, true)),
        ("opening", of("a", 32, &[62, 63], None, true)),
        ("scratch", of("a", 32, &[], Some(1 << 20), true)),
        ("head", of("a", 32, &[], None, false)),
    ] {
        assert_ne!(base, other, "{name}");
    }
}

/// A restarted model finds its states in its own namespace directory, the
/// first 32 hex digits of its fingerprint under the cache root; a load of
/// another checkpoint opens another directory and finds nothing.
/// Mutation: the namespace is a constant → red.
#[test]
fn a_restart_restores_from_the_models_own_directory() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let a = first_conversation(&root, &reuse_inner);
    let own = root
        .cache
        .0
        .join(&fixture_fingerprint(&reuse_inner(), "fixture").to_hex()[..32]);
    assert_eq!(root.namespaces(), vec![own.clone()]);
    assert_eq!(files(&root, "hybrid_cell").len(), 1);
    let b = fork(&a, 98, 110, 2);
    {
        let mut inner = on_disk(reuse_inner(), &root);
        assert_eq!(disk(&inner).tier().dir(), own);
        assert_eq!(turn(&mut inner, 1, &b).0, 96);
    }
    let mut other = on_disk_as(reuse_inner(), &root, "another checkpoint");
    assert_ne!(disk(&other).tier().dir(), own);
    assert_eq!(turn(&mut other, 1, &b).0, 0);
    assert_eq!(count(&other.telemetry.disk_restored), 0);
    assert!(root.temp_is_empty());
}

/// A second load whose namespace another load holds keeps its states in
/// memory: no disk, nothing in the temporary root.
/// Mutation: the persistent-only open falls back to a temporary tier → red.
#[test]
fn a_held_namespace_leaves_the_disk_off() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let first = on_disk(reuse_inner(), &root);
    let mut second = reuse_inner();
    open_once(&mut second, &root, "fixture", (8, 0));
    assert!(second.disk.is_none());
    assert_eq!(count(&second.telemetry.disk_enabled), 0);
    assert!(root.temp_is_empty());
    assert_eq!(root.namespaces().len(), 1);
    drop(first);
}

/// The disk plan follows the paged cache's persistence switch: the load's
/// own option over `MLX_PERSIST_PAGED_CACHE`, `--max-cache-disk 0`, the GGUF
/// identity, and `--cache-dir` only with the switch on. A plan that is on
/// opens the tier and writes the conversation; one that is off opens
/// nothing anywhere.
/// Mutation: the plan ignores the resolved switch → red.
#[test]
fn the_switch_decides_the_tier() {
    let with = |cache_disk: Option<&str>, persist: Option<bool>, cache_dir: Option<&str>| {
        Qwen4ExpLoadOptions {
            cache_disk: cache_disk.map(str::to_string),
            persist_paged_cache: persist,
            cache_dir: cache_dir.map(str::to_string),
            ..Default::default()
        }
    };
    let plain = Qwen4ExpLoadOptions::default();
    for (name, options, env, identity, on) in [
        ("env on", &plain, Some("1"), Some("id"), true),
        ("env unset", &plain, None, Some("id"), false),
        ("env off", &plain, Some("0"), Some("id"), false),
        (
            "option off",
            &with(None, Some(false), None),
            Some("1"),
            Some("id"),
            false,
        ),
        (
            "option on",
            &with(None, Some(true), None),
            Some("0"),
            Some("id"),
            true,
        ),
        (
            "size 0",
            &with(Some("0"), None, None),
            Some("1"),
            Some("id"),
            false,
        ),
        ("no identity", &plain, Some("1"), None, false),
    ] {
        let (tier, plan) = disk_plan("Qwen4Exp", options, env, identity).unwrap();
        assert_eq!(plan, on, "{name}");
        assert_eq!(tier.persistent, on || name == "no identity", "{name}");
    }
    let error = disk_plan("Qwen4Exp", &with(None, None, Some("/x")), None, Some("id"))
        .err()
        .unwrap();
    assert!(
        error.reason.starts_with("Qwen4Exp load: --cache-dir needs"),
        "{}",
        error.reason
    );

    let _host = memory::scoped_fixture_host();
    for (env, on) in [(Some("1"), true), (Some("0"), false), (None, false)] {
        let root = Root::new();
        let cache_dir = on.then(|| root.cache.0.to_string_lossy().into_owned());
        let options = with(Some("1073741824"), None, cache_dir.as_deref());
        let (mut tier, plan) = disk_plan("Qwen4Exp", &options, env, Some("fixture")).unwrap();
        assert_eq!(plan, on, "{env:?}");
        tier.temp_root = Some(root.tmp.0.clone());
        let mut inner = reuse_inner();
        crate::disk_tier::TEST_WRITER.with(|writer| writer.set(Some((8, 0))));
        inner.open_prefix_disk(&tier, Some("fixture"), None);
        crate::disk_tier::TEST_WRITER.with(|writer| writer.set(None));
        assert_eq!(inner.disk.is_some(), on, "{env:?}");
        turn(&mut inner, 1, &prompt(100, 1));
        finish(&mut inner);
        if on {
            settle(&inner);
        }
        assert_eq!(objects(&root) > 0, on, "{env:?}");
        assert_eq!(root.namespaces().is_empty(), !on, "{env:?}");
        assert!(root.temp_is_empty(), "{env:?}");
        assert_eq!(
            count(&inner.telemetry.disk_enabled),
            u64::from(on),
            "{env:?}"
        );
    }
}

/// What every host does (`persistPagedCache: { byDefault: true }`, written
/// to `MLX_PERSIST_PAGED_CACHE` before any load) gives a GGUF Flash Next
/// load with no option of its own a persistent tier that opens; the
/// cache root stands in for `~/.mlx-node/cache/prefix-cache`, which a test
/// build never reads.
/// Mutation: the plan does not read the switch's environment value → red.
#[test]
fn a_host_load_keeps_its_states_on_disk() {
    let (mut tier, on) = disk_plan(
        "Qwen4Exp",
        &Qwen4ExpLoadOptions::default(),
        Some("1"),
        Some("fixture"),
    )
    .unwrap();
    assert!(on);
    assert!(tier.persistent);
    assert_eq!(tier.cache_disk, crate::disk_tier::CacheDisk::Auto);
    // An automatic size is refused on a nearly full volume.
    tier.cache_disk = crate::disk_tier::CacheDisk::Bytes(1 << 30);
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    tier.cache_dir = Some(root.cache.0.clone());
    tier.temp_root = Some(root.tmp.0.clone());
    let mut inner = reuse_inner();
    crate::disk_tier::TEST_WRITER.with(|writer| writer.set(Some((8, 0))));
    inner.open_prefix_disk(&tier, Some("fixture"), None);
    crate::disk_tier::TEST_WRITER.with(|writer| writer.set(None));
    assert!(disk(&inner).tier().persistent);
    assert_eq!(count(&inner.telemetry.disk_enabled), 1);
}

/// `closeCache`'s close hands a state parked with the whole delay ahead to
/// the writer: it is on disk after the close and restores after a restart.
/// The closed tier takes nothing more: a later turn's end encodes no state
/// and copies no block.
/// Mutation: the close skips the parker's flush → red (no state on disk).
#[test]
fn close_cache_hands_parked_states_over() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let a = prompt(100, 1);
    {
        let mut inner = on_disk(reuse_inner(), &root);
        assert_eq!(disk(&inner).delay, prefix_disk::DELAY);
        turn(&mut inner, 1, &a);
        finish(&mut inner);
        assert_eq!(count(&inner.telemetry.waiting), 1);
        assert!(
            disk(&inner)
                .closer()
                .upgrade()
                .unwrap()
                .close(Instant::now() + Duration::from_secs(5))
        );
        assert_eq!(count(&inner.telemetry.disk_enabled), 0);
        assert_eq!(files(&root, "hybrid_cell").len(), 1);
        let (encoded, written) = (disk(&inner).encoded, objects(&root));
        turn(&mut inner, 2, &prompt(100, 2));
        finish(&mut inner);
        assert_eq!(disk(&inner).encoded, encoded);
        assert_eq!(count(&inner.telemetry.owed), 0);
        assert_eq!(objects(&root), written);
    }
    let mut inner = on_disk(reuse_inner(), &root);
    assert_eq!(turn(&mut inner, 1, &fork(&a, 98, 110, 2)).0, 96);
}

/// The napi object outlives a dispose until V8 collects it: the closer
/// handle it keeps does not hold the parked states, the copies or the tier.
/// Mutation: the handle keeps the closer alive → red.
#[test]
fn a_dispose_frees_what_the_closer_reaches() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    turn(&mut inner, 1, &prompt(100, 1));
    finish(&mut inner);
    assert_eq!(count(&inner.telemetry.waiting), 1);
    let handle = inner.disk.as_ref().map(prefix_disk::Disk::closer).unwrap();
    let tier = Arc::downgrade(disk(&inner).tier());
    inner.disk = None;
    assert!(handle.upgrade().is_none(), "the handle keeps the closer");
    assert!(tier.upgrade().is_none(), "the tier outlives the dispose");
}

/// A dispose keeps what a running turn handed the writer: with the writer
/// stalled behind one state write, a turn's block copies wait in its queue,
/// and dropping the model's disk still lands them.
/// Mutation: the drop closes the tier at once → red (the blocks are lost).
#[test]
fn dispose_keeps_what_the_writer_holds() {
    use mlx_paged_attn::{
        ColdCacheKey, ColdStateLabel, ColdStateLayout, ColdStateObject, MakeRoom,
    };
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk_with(reuse_inner(), &root, "fixture", (64, 0));
    let manager = store(&inner);
    let (entered, stalled) = std::sync::mpsc::channel::<()>();
    let (release, released) = std::sync::mpsc::channel::<()>();
    let released = std::sync::Mutex::new(Some(released));
    manager.set_landing_wake(Some(Arc::new(move || {
        let _ = entered.send(());
        let released = released.lock().unwrap().take();
        if let Some(released) = released {
            let _ = released.recv_timeout(Duration::from_secs(20));
        }
    })));
    let tokens: Vec<u32> = (0..16).collect();
    let fingerprint = disk(&inner).tier().state_fingerprint;
    let (done, _landed) = std::sync::mpsc::channel();
    manager
        .enqueue_state(
            ColdStateObject {
                key: ColdCacheKey::chain(
                    ColdGroup::DFlash2State,
                    fingerprint,
                    None,
                    &tokens,
                    &[],
                    0,
                    0,
                ),
                fingerprint,
                layout: ColdStateLayout {
                    boundary_tokens: 16,
                    spans: vec![96, 64],
                    geometry: "gdn 1 3 14 2 4 3\n".to_string(),
                },
                label: ColdStateLabel {
                    tokens,
                    extra_keys: Vec::new(),
                    salt: 0,
                    last_used: 1,
                    checkpoint: false,
                },
                payload: vec![7u8; 160],
            },
            MakeRoom::Evict,
            done,
        )
        .unwrap();
    stalled.recv_timeout(Duration::from_secs(20)).unwrap();
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    let deadline = Instant::now() + Duration::from_secs(20);
    while count(&inner.telemetry.owed) > 0 {
        assert!(
            Instant::now() < deadline,
            "the copies never reached the writer"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let kv = keys_of(&inner, &a, ColdGroup::Kv, 3);
    assert!(kv.iter().all(|key| !manager.contains(key)));
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        let _ = release.send(());
    });
    inner.disk = None;
    releaser.join().unwrap();
    drop(manager);
    let again = on_disk(reuse_inner(), &root);
    let reopened = store(&again);
    assert!(
        kv.iter().all(|key| reopened.contains(key)),
        "the blocks the writer held are not on disk"
    );
}
