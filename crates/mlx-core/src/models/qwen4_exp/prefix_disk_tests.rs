//! The stored states' disk tier on the paged fixture: a store in a temporary
//! folder, the fixture's fingerprint, and a restart that drops the model and
//! builds a new one over the same store reopened.
use super::prefix_reuse::{
    GRID, cold_on, cycles, finish, fork, greedy, prompt, reuse_inner, reuse_over, speculative_turn,
    turn,
};
use super::*;
use crate::engine::backend::PagedBackend;
use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
use mlx_paged_attn::{ColdCacheFingerprint, ColdCacheManager, ColdGroup};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn open(dir: &Temp) -> Arc<ColdCacheManager> {
    Arc::new(ColdCacheManager::open_at(dir.0.clone(), 1 << 30, 0, 8).unwrap())
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

/// `inner` with its states' disk tier over `manager`, as a load of the
/// fixture checkpoint `identity` opens it.
fn on_disk_as(mut inner: Inner, manager: &Arc<ColdCacheManager>, identity: &str) -> Inner {
    let fingerprint = fixture_fingerprint(&inner, identity);
    let disk = prefix_disk::Disk::open(
        Arc::clone(manager),
        fingerprint,
        Arc::clone(&inner.telemetry),
    )
    .unwrap();
    disk.attach(inner.decoder.paged.as_mut().unwrap());
    inner.disk = Some(disk);
    inner
}

fn on_disk(inner: Inner, manager: &Arc<ColdCacheManager>) -> Inner {
    on_disk_as(inner, manager, "fixture")
}

fn disk(inner: &Inner) -> &prefix_disk::Disk {
    inner.disk.as_ref().unwrap()
}

/// Everything parked and copied out lands: the parker hands it over, the
/// store's writer commits it.
fn settle(inner: &Inner, manager: &ColdCacheManager) {
    disk(inner).flush(Instant::now() + Duration::from_secs(20));
    assert!(manager.drain(Duration::from_secs(20)));
}

/// The store's object files of `group`.
fn files(dir: &Temp, group: &str) -> Vec<PathBuf> {
    let suffix = format!(".{group}.safetensors");
    let mut out: Vec<PathBuf> = std::fs::read_dir(&dir.0)
        .unwrap()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.to_string_lossy().ends_with(&suffix))
        .collect();
    out.sort();
    out
}

fn objects(dir: &Temp) -> usize {
    std::fs::read_dir(&dir.0)
        .unwrap()
        .flatten()
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
/// model and the store are gone after.
fn first_conversation_as(dir: &Temp, make: &dyn Fn() -> Inner, identity: &str) -> Vec<u32> {
    let manager = open(dir);
    let mut inner = on_disk_as(make(), &manager, identity);
    let a = prompt(100, 1);
    assert_eq!(turn(&mut inner, 1, &a).0, 0);
    finish(&mut inner);
    settle(&inner, &manager);
    a
}

fn first_conversation(dir: &Temp, make: &dyn Fn() -> Inner) -> Vec<u32> {
    first_conversation_as(dir, make, "fixture")
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
        let dir = Temp::new();
        let a = first_conversation(&dir, make);
        assert_eq!(files(&dir, "hybrid_cell").len(), 1, "{name}");
        let manager = open(&dir);
        let mut inner = on_disk(make(), &manager);
        let b = fork(&a, 98, 110, 2);
        let (cached, logits) = turn(&mut inner, 1, &b);
        assert_eq!(cached, 96, "{name}");
        assert_eq!(count(&inner.telemetry.disk_restored), 1, "{name}");
        let steps = greedy(&mut inner, &logits, 8);
        let (want_logits, want_steps) = cold_on(make(), &b);
        assert_eq!(logits, want_logits, "{name}: first logits");
        assert_eq!(steps, want_steps, "{name}: greedy steps");
        assert_eq!(manager.stats().corruptions, 0, "{name}");
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
    let dir = Temp::new();
    let a = prompt(100, 1);
    {
        let manager = open(&dir);
        let mut inner = on_disk(reuse_inner(), &manager);
        let (_, begun, _) = speculative_turn(&mut inner, &a);
        assert!(begun);
        finish(&mut inner);
        inner.release_scheduled_speculation(1);
        settle(&inner, &manager);
    }
    assert_eq!(files(&dir, "hybrid_draft").len(), 2);
    let manager = open(&dir);
    let mut inner = on_disk(reuse_inner(), &manager);
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
        let dir = Temp::new();
        let a = first_conversation(&dir, &reuse_inner);
        let state = files(&dir, "hybrid_cell").remove(0);
        let mut bytes = std::fs::read(&state).unwrap();
        match case {
            "truncated" => bytes.truncate(bytes.len() / 2),
            "checksum" => {
                let last = bytes.len() - 1;
                bytes[last] ^= 0x40;
            }
            _ => {
                let other = Temp::new();
                first_conversation_as(&other, &reuse_inner, "another checkpoint");
                bytes = std::fs::read(files(&other, "hybrid_cell").remove(0)).unwrap();
            }
        }
        std::fs::write(&state, &bytes).unwrap();
        let manager = open(&dir);
        let mut inner = on_disk(reuse_inner(), &manager);
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
    let dir = Temp::new();
    let a = first_conversation(&dir, &reuse_inner);
    let kv = keys_of(&reuse_inner(), &a, ColdGroup::Kv, 3);
    let block = dir.0.join(format!("{}.safetensors", kv[1].to_hex()));
    assert!(block.exists());
    std::fs::remove_file(&block).unwrap();
    let manager = open(&dir);
    let mut inner = on_disk(reuse_inner(), &manager);
    let b = fork(&a, 98, 110, 2);
    let (cached, logits) = turn(&mut inner, 1, &b);
    assert_eq!(cached, 0);
    assert_eq!(count(&inner.telemetry.dropped_chain), 1);
    assert!(files(&dir, "hybrid_cell").is_empty());
    assert_eq!(logits, cold_on(reuse_inner(), &b).0);
}

/// A conversation that goes on inside the delay writes only its newest
/// state: the parked one is replaced.
/// Mutation: a parked state is due at once → red (two state objects).
#[test]
fn a_conversation_that_went_on_writes_one_state() {
    let _host = memory::scoped_fixture_host();
    let dir = Temp::new();
    let manager = open(&dir);
    let mut inner = on_disk(reuse_inner(), &manager);
    let a = prompt(70, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    let b = fork(&a, 68, 100, 3);
    assert_eq!(turn(&mut inner, 1, &b).0, 64);
    finish(&mut inner);
    assert_eq!(count(&inner.telemetry.unneeded), 1);
    settle(&inner, &manager);
    let cells = keys_of(&inner, &b, ColdGroup::HybridCell, 3);
    assert_eq!(files(&dir, "hybrid_cell").len(), 1);
    assert!(manager.contains_in(&cells[2], ColdGroup::HybridCell));
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
    let dir = Temp::new();
    let manager = open(&dir);
    let mut inner = on_disk(reuse_inner(), &manager);
    disk(&inner).hold(true);
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    disk(&inner).flush(Instant::now() + Duration::from_millis(300));
    assert!(manager.drain(Duration::from_secs(20)));
    assert!(files(&dir, "hybrid_cell").is_empty());
    assert_eq!(objects(&dir), 0);
    assert_eq!(count(&inner.telemetry.waiting), 1);
    disk(&inner).hold(false);
    settle(&inner, &manager);
    assert_eq!(files(&dir, "hybrid_cell").len(), 1);
    assert_eq!(count(&inner.telemetry.waiting), 0);
}

/// Turns never wait for the disk: with the writer held, two turns run and
/// end, their copies wait, nothing is written.
/// Mutation: `park` waits for the hand-over (a flush of 30 s) → red.
#[test]
fn a_turn_never_waits_for_the_writer() {
    let _host = memory::scoped_fixture_host();
    let dir = Temp::new();
    let manager = open(&dir);
    let mut inner = on_disk(reuse_inner(), &manager);
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
    assert_eq!(objects(&dir), 0);
    disk(&inner).hold(false);
    settle(&inner, &manager);
    assert_eq!(files(&dir, "hybrid_cell").len(), 2);
}

/// States on disk stay inside their byte budget, the oldest going first:
/// with room for two and a half states, a third conversation's state
/// replaces the first one's.
/// Mutation: the hand-over evicts nothing → red (three states).
#[test]
fn the_state_budget_drops_the_oldest() {
    let _host = memory::scoped_fixture_host();
    let dir = Temp::new();
    let manager = open(&dir);
    let without_head = || {
        let mut inner = reuse_inner();
        inner.has_mtp = false;
        inner
    };
    let mut inner = on_disk(without_head(), &manager);
    let groups = [ColdGroup::HybridCell, ColdGroup::HybridDraft];
    let mut cells = Vec::new();
    for seq in 1..=3u32 {
        let tokens = prompt(100, seq);
        assert_eq!(turn(&mut inner, seq, &tokens).0, 0);
        finish(&mut inner);
        settle(&inner, &manager);
        if seq == 1 {
            let unit = manager.group_bytes(&groups);
            assert!(unit > 0);
            disk(&inner).set_budget(unit * 5 / 2);
        }
        cells.push(keys_of(&inner, &tokens, ColdGroup::HybridCell, 3)[2]);
    }
    assert_eq!(files(&dir, "hybrid_cell").len(), 2);
    assert!(!manager.contains_in(&cells[0], ColdGroup::HybridCell));
    assert!(manager.contains_in(&cells[1], ColdGroup::HybridCell));
    assert!(manager.contains_in(&cells[2], ColdGroup::HybridCell));
}

/// The tier follows the paged cache's persistence switch: off by `0`, off
/// on a library load that leaves it unset, off without a GGUF identity. A
/// load the switch leaves off opens no tier and writes nothing; on, it
/// writes the conversation.
/// Mutation: the load opens the tier whatever the switch says → red.
#[test]
fn the_tier_is_off_by_the_switch() {
    assert!(!disk_switch(Some("0"), Some("identity")));
    assert!(!disk_switch(None, Some("identity")));
    assert!(!disk_switch(Some("1"), None));
    assert!(disk_switch(Some("1"), Some("identity")));
    let _host = memory::scoped_fixture_host();
    for (env, on) in [(Some("0"), false), (None, false), (Some("1"), true)] {
        let dir = Temp::new();
        let manager = open(&dir);
        let mut inner = reuse_inner();
        inner.open_prefix_disk(env, Some("fixture"), None, || Some(Arc::clone(&manager)));
        assert_eq!(inner.disk.is_some(), on, "{env:?}");
        turn(&mut inner, 1, &prompt(100, 1));
        finish(&mut inner);
        if on {
            settle(&inner, &manager);
        }
        assert!(manager.drain(Duration::from_secs(20)));
        assert_eq!(objects(&dir) > 0, on, "{env:?}");
        assert_eq!(
            count(&inner.telemetry.disk_enabled),
            u64::from(on),
            "{env:?}"
        );
    }
}

/// A flush stops waiting once nothing more can land: with every write
/// failing (the store keeps a reserve no disk has) a turn's state can never
/// land, and the flush, a second one and the unload each return long before
/// their deadline, leaving the store's own drain its time.
/// Mutation: the flush waits until no state is parked → red.
#[test]
fn a_flush_does_not_wait_for_a_state_that_cannot_land() {
    let _host = memory::scoped_fixture_host();
    let dir = Temp::new();
    let manager = Arc::new(ColdCacheManager::open_at(dir.0.clone(), 1 << 30, 1 << 62, 8).unwrap());
    let mut inner = on_disk(reuse_inner(), &manager);
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
    assert_eq!(objects(&dir), 0);
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
    let dir = Temp::new();
    let manager = Arc::new(ColdCacheManager::open_at(dir.0.clone(), 1 << 30, 0, 1).unwrap());
    let mut inner = on_disk(reuse_inner(), &manager);
    disk(&inner).set_slot_wait(Duration::ZERO);
    disk(&inner).hold(true);
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    disk(&inner).hold(false);
    settle(&inner, &manager);
    assert_eq!(files(&dir, "hybrid_cell").len(), 1);
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
    let dir = Temp::new();
    let manager = open(&dir);
    let mut inner = on_disk(reuse_inner(), &manager);
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
    let dir = Temp::new();
    let manager = open(&dir);
    let mut inner = on_disk(reuse_inner(), &manager);
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
    let dir = Temp::new();
    let a = first_conversation(&dir, &reuse_inner);
    let manager = open(&dir);
    let mut inner = on_disk(reuse_inner(), &manager);
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
