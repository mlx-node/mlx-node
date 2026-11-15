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
        self.options_of(1 << 30)
    }

    /// The persistent tier of a `bytes` cache under this root.
    fn options_of(&self, bytes: u64) -> DiskTierOptions {
        DiskTierOptions {
            temp_root: Some(self.tmp.0.clone()),
            ..DiskTierOptions::from_load(
                Some(&bytes.to_string()),
                true,
                Some(self.cache.0.to_string_lossy().into_owned()),
            )
            .unwrap()
        }
    }

    /// The bytes of every object file in every namespace.
    fn object_bytes(&self) -> u64 {
        self.namespaces()
            .iter()
            .flat_map(|dir| std::fs::read_dir(dir).unwrap().flatten())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".safetensors")
            })
            .map(|entry| entry.metadata().unwrap().len())
            .sum()
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

    /// The temporary caches under the temporary root.
    fn temp_dirs(&self) -> Vec<PathBuf> {
        std::fs::read_dir(&self.tmp.0)
            .unwrap()
            .flatten()
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("mlx-node-cache-")
            })
            .map(|entry| entry.path())
            .collect()
    }

    /// The object files of every temporary cache whose names end in
    /// `suffix`.
    fn temp_objects(&self, suffix: &str) -> Vec<PathBuf> {
        self.temp_dirs()
            .iter()
            .flat_map(|dir| std::fs::read_dir(dir).unwrap().flatten())
            .map(|entry| entry.path())
            .filter(|path| path.to_string_lossy().ends_with(suffix))
            .collect()
    }

    /// The tier of a `--no-persist-cache` load: temporary, under this
    /// root's temporary root.
    fn switch_off(&self) -> DiskTierOptions {
        DiskTierOptions {
            temp_root: Some(self.tmp.0.clone()),
            ..DiskTierOptions::from_load(Some("1073741824"), false, None).unwrap()
        }
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

/// One open of `inner`'s disk tier of `options` as a load of checkpoint
/// `identity` opens it, its writer queueing `depth` jobs and keeping
/// `reserve` bytes free. A turn of the fixture's own size writes: the
/// minimum lineage is one pass.
fn open_tier_once(
    inner: &mut Inner,
    options: &DiskTierOptions,
    identity: Option<&str>,
    (depth, reserve): (usize, u64),
) {
    crate::disk_tier::TEST_WRITER.with(|writer| writer.set(Some((depth, reserve))));
    inner.open_prefix_disk(options, identity, None);
    crate::disk_tier::TEST_WRITER.with(|writer| writer.set(None));
    if let Some(disk) = inner.disk.as_mut() {
        disk.minimum = GRID;
    }
}

/// One open of `inner`'s disk tier under `root`.
fn open_once(inner: &mut Inner, root: &Root, identity: &str, writer: (usize, u64)) {
    open_tier_once(inner, &root.options(), Some(identity), writer);
}

/// `inner` with the persistent tier of `options` open. The last tier of
/// the same namespace lets go of its lock only when its writer thread
/// exits, after its close returned; until then an open gives a temporary
/// tier, which goes again.
fn on_disk_under(
    mut inner: Inner,
    options: &DiskTierOptions,
    identity: &str,
    writer: (usize, u64),
) -> Inner {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        open_tier_once(&mut inner, options, Some(identity), writer);
        if inner
            .disk
            .as_ref()
            .is_some_and(|disk| disk.tier().persistent)
        {
            return inner;
        }
        inner.disk = None;
        assert!(Instant::now() < deadline, "the namespace stays held");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// `inner` with its disk tier open under `root`.
fn on_disk_with(inner: Inner, root: &Root, identity: &str, writer: (usize, u64)) -> Inner {
    on_disk_under(inner, &root.options(), identity, writer)
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

fn disk_mut(inner: &mut Inner) -> &mut prefix_disk::Disk {
    inner.disk.as_mut().unwrap()
}

/// The tier's store.
fn store(inner: &Inner) -> Arc<ColdCacheManager> {
    Arc::clone(&disk(inner).tier().manager)
}

/// Everything waiting and copied out lands, due or not: the model thread
/// writes the waiting states newest first, the parker hands the copies
/// over, the store's writer commits them. A test of the delay must not call
/// it.
fn settle(inner: &mut Inner) {
    inner.flush_disk(Instant::now() + Duration::from_secs(20));
    assert!(store(inner).drain(Duration::from_secs(20)));
}

/// The model thread's disk pass, again and again, until `until` holds.
fn drive(inner: &mut Inner, until: impl Fn(&Inner) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        inner.step_disk_now();
        if until(inner) {
            return;
        }
        assert!(Instant::now() < deadline, "the disk never got there");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// One drive pass's disk tick, as the scheduler makes it.
fn tick(inner: &mut Inner) {
    inner.tick_state_disk(crate::models::qwen3_5::dflash2_state::disk::Tick::Poll);
}

/// The entry of the stored state at `depth`.
fn entry_at(
    inner: &Inner,
    depth: usize,
) -> &crate::models::qwen3_5::dflash2_state::store::Entry<prefix_state::Cell> {
    inner
        .prefix_states
        .store
        .entries()
        .find(|entry| entry.depth() == depth)
        .unwrap()
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
    settle(&mut inner);
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

/// A restarted model takes the state the first model left on disk back at
/// load, on disk only, and resumes a follow-up there: it reports row 96
/// cached, the state has its RAM copy back beside its disk copy, and the
/// turn gives a cold model's logits and greedy steps bit for bit, on a bf16
/// pool and on an int8 pool with the wide head; the store counts no
/// corruption.
/// Mutation: the read stores the cell as a new state in place of giving
/// the entry its RAM copy back → red (the turn finds no RAM copy).
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
        assert_eq!(files(&root, "dflash2_state").len(), 1, "{name}");
        let mut inner = on_disk(make(), &root);
        assert_eq!(count(&inner.telemetry.adopted), 1, "{name}");
        assert!(entry_at(&inner, 96).ram().is_none(), "{name}");
        let b = fork(&a, 98, 110, 2);
        let (cached, logits) = turn(&mut inner, 1, &b);
        assert_eq!(cached, 96, "{name}");
        assert_eq!(count(&inner.telemetry.disk_restored), 1, "{name}");
        let restored = entry_at(&inner, 96);
        assert!(restored.ram().is_some(), "{name}");
        assert!(prefix_state::durable(restored), "{name}");
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
        settle(&mut inner);
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
        let state = files(&root, "dflash2_state").remove(0);
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
                bytes = std::fs::read(files(&other, "dflash2_state").remove(0)).unwrap();
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

/// Turns never wait for the disk: with the copies held, two turns run and
/// end, their states wait for their delay, nothing is written.
/// Mutation: a turn's end flushes the waiting states → red.
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
    settle(&mut inner);
    assert_eq!(files(&root, "dflash2_state").len(), 2);
}

/// States share the tier's one quota with their chains: a state is
/// refused only when the tier cannot hold it, not past a share of its own.
/// The tier holds one conversation (one block, no draft head) and half a
/// state more; the state takes more than a quarter of it and lands.
/// Mutation: a quarter of the quota for states → red (the state is
/// refused).
#[test]
fn states_share_the_tier_quota() {
    let _host = memory::scoped_fixture_host();
    let without_head = || {
        let mut inner = reuse_inner();
        inner.has_mtp = false;
        inner
    };
    let tokens = prompt(40, 1);
    let (unit, state) = {
        let root = Root::new();
        let mut inner = on_disk(without_head(), &root);
        turn(&mut inner, 1, &tokens);
        finish(&mut inner);
        settle(&mut inner);
        let cells = files(&root, "dflash2_state");
        assert_eq!(cells.len(), 1);
        (
            store(&inner).total_bytes(),
            std::fs::metadata(&cells[0]).unwrap().len(),
        )
    };
    let quota = unit + state / 2;
    assert!(quota / 4 < state, "{quota} {state}");
    let root = Root::new();
    let mut inner = on_disk_under(without_head(), &root.options_of(quota), "fixture", (8, 0));
    turn(&mut inner, 1, &tokens);
    finish(&mut inner);
    settle(&mut inner);
    assert_eq!(files(&root, "dflash2_state").len(), 1);
    assert_eq!(count(&inner.telemetry.refused), 0);
    assert_eq!(count(&inner.telemetry.durable), 1);
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
    settle(&mut inner);
    assert_eq!(files(&root, "dflash2_state").len(), 1);
    assert_eq!(count(&inner.telemetry.durable), 1);
    assert_eq!(count(&inner.telemetry.owed), 0);
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
    assert_eq!(files(&root, "dflash2_state").len(), 1);
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

/// A second load whose namespace another load holds keeps a temporary
/// cache, not none.
/// Mutation: the open keeps no disk when the tier is temporary → red.
#[test]
fn a_held_namespace_keeps_a_temporary_cache() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let first = on_disk(reuse_inner(), &root);
    let mut second = reuse_inner();
    open_once(&mut second, &root, "fixture", (8, 0));
    assert!(second.disk.is_some());
    assert!(!disk(&second).tier().persistent);
    assert_eq!(root.temp_dirs().len(), 1);
    assert_eq!(count(&second.telemetry.disk_enabled), 2);
    assert_eq!(root.namespaces().len(), 1);
    drop(second);
    assert!(root.temp_is_empty());
    drop(first);
}

/// The disk plan follows `--max-cache-disk` alone: on unless it is 0. The
/// persistence switch (the load's own option over `MLX_PERSIST_PAGED_CACHE`)
/// and the GGUF identity decide whether the tier is persistent, and
/// `--cache-dir` needs the switch. A persistent open writes the
/// conversation into its namespace; a temporary one (switch off, or no
/// identity) opens one temporary cache, writes nothing without memory
/// demand and makes no namespace.
/// Mutation: the plan's switch ignores `--max-cache-disk` → red.
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
    for (name, options, env, on, persistent) in [
        ("env on", &plain, Some("1"), true, true),
        ("env unset", &plain, None, true, false),
        ("env off", &plain, Some("0"), true, false),
        (
            "option off",
            &with(None, Some(false), None),
            Some("1"),
            true,
            false,
        ),
        (
            "option on",
            &with(None, Some(true), None),
            Some("0"),
            true,
            true,
        ),
        (
            "size 0",
            &with(Some("0"), None, None),
            Some("1"),
            false,
            false,
        ),
    ] {
        let (tier, plan) = disk_plan("Qwen4Exp", options, env).unwrap();
        assert_eq!(plan, on, "{name}");
        assert_eq!(tier.persistent, persistent, "{name}");
    }
    let error = disk_plan("Qwen4Exp", &with(None, None, Some("/x")), None)
        .err()
        .unwrap();
    assert!(
        error.reason.starts_with("Qwen4Exp load: --cache-dir needs"),
        "{}",
        error.reason
    );

    let _host = memory::scoped_fixture_host();
    for (env, identity, persistent) in [
        (Some("1"), Some("fixture"), true),
        (Some("0"), Some("fixture"), false),
        (None, Some("fixture"), false),
        (Some("1"), None, false),
    ] {
        let case = format!("{env:?} {identity:?}");
        let root = Root::new();
        let switch = env == Some("1");
        let cache_dir = switch.then(|| root.cache.0.to_string_lossy().into_owned());
        let options = with(Some("1073741824"), None, cache_dir.as_deref());
        let (mut tier, plan) = disk_plan("Qwen4Exp", &options, env).unwrap();
        assert!(plan, "{case}");
        assert_eq!(tier.persistent, switch, "{case}");
        tier.temp_root = Some(root.tmp.0.clone());
        let mut inner = reuse_inner();
        open_tier_once(&mut inner, &tier, identity, (8, 0));
        assert!(inner.disk.is_some(), "{case}");
        assert_eq!(disk(&inner).tier().persistent, persistent, "{case}");
        turn(&mut inner, 1, &prompt(100, 1));
        finish(&mut inner);
        settle(&mut inner);
        assert_eq!(objects(&root) > 0, persistent, "{case}");
        assert_eq!(root.namespaces().is_empty(), !persistent, "{case}");
        assert_eq!(root.temp_is_empty(), persistent, "{case}");
        assert_eq!(root.temp_dirs().len(), usize::from(!persistent), "{case}");
        assert!(root.temp_objects(".safetensors").is_empty(), "{case}");
        assert_eq!(
            count(&inner.telemetry.disk_enabled),
            if persistent { 1 } else { 2 },
            "{case}"
        );
    }
}

/// What every host does (`persistPagedCache: { byDefault: true }`, written
/// to `MLX_PERSIST_PAGED_CACHE` before any load) gives a GGUF Flash Next
/// load with no option of its own a persistent tier that opens, never a
/// silent temporary one: its namespace under the cache root, nothing in the
/// temporary root. The cache root stands in for
/// `~/.mlx-node/cache/prefix-cache`, which a test build never reads.
/// Mutation: the plan does not read the switch's environment value → red.
#[test]
fn a_host_load_keeps_its_states_on_disk() {
    let (mut tier, on) = disk_plan("Qwen4Exp", &Qwen4ExpLoadOptions::default(), Some("1")).unwrap();
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
    open_tier_once(&mut inner, &tier, Some("fixture"), (8, 0));
    assert!(disk(&inner).tier().persistent);
    assert_eq!(count(&inner.telemetry.disk_enabled), 1);
    assert_eq!(
        root.namespace(),
        root.cache
            .0
            .join(&fixture_fingerprint(&inner, "fixture").to_hex()[..32])
    );
    assert!(root.temp_is_empty());
}

/// The napi object outlives a dispose until V8 collects it: the closer
/// handle it keeps does not hold the waiting states, the copies or the tier,
/// and neither do the wakes, the load's or the writer's after a landing.
/// Mutation: the handle keeps the closer alive → red. Mutation: the
/// writer's landing wake holds the closer → red.
#[test]
fn a_dispose_frees_what_the_closer_reaches() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    let (woke, _wakes) = std::sync::mpsc::channel::<()>();
    disk(&inner)
        .closer()
        .upgrade()
        .unwrap()
        .set_wake(Arc::new(move || {
            let _ = woke.send(());
        }));
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

/// Stalls `inner`'s tier writer behind one tracked state write until the
/// returned sender sends (or 10 s pass): every copy queued after it waits.
fn stall_writer(inner: &Inner) -> std::sync::mpsc::Sender<()> {
    use mlx_paged_attn::{
        ColdCacheKey, ColdStateLabel, ColdStateLayout, ColdStateObject, MakeRoom,
    };
    let manager = store(inner);
    let (entered, stalled) = std::sync::mpsc::channel::<()>();
    let (release, released) = std::sync::mpsc::channel::<()>();
    let released = std::sync::Mutex::new(Some(released));
    manager.set_landing_wake(Some(Arc::new(move || {
        let _ = entered.send(());
        let released = released.lock().unwrap().take();
        if let Some(released) = released {
            let _ = released.recv_timeout(Duration::from_secs(10));
        }
    })));
    let tokens: Vec<u32> = (0..16).collect();
    let fingerprint = disk(inner).tier().state_fingerprint;
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
    release
}

/// `tokens`' turn on sequence 1 with its block copies handed to a stalled
/// writer: their keys, none of them on disk yet.
fn queued_behind_a_stall(inner: &mut Inner, tokens: &[u32]) -> Vec<mlx_paged_attn::ColdCacheKey> {
    turn(inner, 1, tokens);
    let deadline = Instant::now() + Duration::from_secs(20);
    while count(&inner.telemetry.owed) > 0 {
        assert!(
            Instant::now() < deadline,
            "the copies never reached the writer"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let kv = keys_of(inner, tokens, ColdGroup::Kv, 3);
    assert!(kv.iter().all(|key| !store(inner).contains(key)));
    kv
}

/// A reopen under a lower `cacheDisk` than the last process had trims the
/// store to the new limit at once, before any write and before it takes
/// any state back: the index and the files on disk are inside it, every
/// state it took back has its object and whole chain, and a follow-up
/// gives a cold model's logits.
/// Mutation: the open trims only after it took the states back → red (a
/// state whose chain the trim broke came back).
#[test]
fn a_reopen_under_a_lower_quota_trims_at_once() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let a = first_conversation(&root, &reuse_inner);
    let before = root.object_bytes();
    let limit = before - 1;
    assert!(limit > 0, "{before}");
    let inner = on_disk_under(reuse_inner(), &root.options_of(limit), "fixture", (8, 0));
    assert!(disk(&inner).tier().persistent);
    let indexed = store(&inner).total_bytes();
    assert!(indexed <= limit, "{indexed} indexed over the {limit} limit");
    let on_disk = root.object_bytes();
    assert!(
        on_disk <= limit,
        "{on_disk} on disk over the {limit} limit (was {before})"
    );
    assert!(on_disk > 0, "the trim takes only what is over the limit");
    let manager = store(&inner);
    let adapter = inner.decoder.paged.as_ref().unwrap();
    for entry in inner.prefix_states.store.entries() {
        let copy = entry.disk().unwrap();
        assert!(manager.contains_in(&copy.key, ColdGroup::DFlash2State));
        assert!(
            disk(&inner)
                .chain_whole(
                    entry.tokens(),
                    entry.extra_keys(),
                    entry.salt(),
                    entry.depth(),
                    adapter,
                    Some(GRID),
                )
                .is_some(),
            "a state the trim broke came back"
        );
    }
    let mut inner = inner;
    let b = fork(&a, 98, 110, 2);
    let (_, logits) = turn(&mut inner, 1, &b);
    assert_eq!(logits, cold_on(reuse_inner(), &b).0);
}

/// Overlapping closes share one outcome: a close that comes while another
/// drains waits for it and never reports true before the copies queued
/// ahead of it are on disk, it does not cut the running close's deadline
/// short, and a close after both gives the same outcome.
/// Mutation: every close runs on its own → red.
#[test]
fn overlapping_closes_share_one_outcome() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk_with(reuse_inner(), &root, "fixture", (64, 0));
    let manager = store(&inner);
    let release = stall_writer(&inner);
    let kv = queued_behind_a_stall(&mut inner, &prompt(100, 1));
    let closer = disk(&inner).closer().upgrade().unwrap();
    let first = {
        let closer = Arc::clone(&closer);
        std::thread::spawn(move || closer.close(Instant::now() + Duration::from_secs(20)))
    };
    std::thread::sleep(Duration::from_millis(200));
    let second = closer.close(Instant::now() + Duration::from_millis(300));
    let landed = kv.iter().all(|key| manager.contains(key));
    assert!(!landed, "the writer is stalled");
    assert!(!second, "a close reported true before its copies landed");
    release.send(()).unwrap();
    let first = first.join().unwrap();
    assert!(
        kv.iter().all(|key| manager.contains(key)),
        "the copies queued before the closes are lost"
    );
    assert!(first, "the running close missed its deadline");
    assert!(closer.close(Instant::now() + Duration::from_secs(1)));
    assert_eq!(count(&inner.telemetry.disk_enabled), 0);
}

/// A `closeCache` that comes while a dispose closes the tier gets the
/// dispose's outcome: with the writer stalled past the dispose's 2 s the
/// copies are lost, and it reports false, never true.
/// Mutation: every close runs on its own → red.
#[test]
fn close_cache_during_a_dispose_shares_its_outcome() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk_with(reuse_inner(), &root, "fixture", (64, 0));
    let manager = store(&inner);
    let release = stall_writer(&inner);
    let kv = queued_behind_a_stall(&mut inner, &prompt(100, 1));
    let closer = disk(&inner).closer().upgrade().unwrap();
    let late = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        closer.close(Instant::now() + Duration::from_secs(20))
    });
    inner.disk = None;
    let _ = release.send(());
    let reported = late.join().unwrap();
    let landed = kv.iter().all(|key| manager.contains(key));
    assert!(
        !reported || landed,
        "closeCache reported true and its copies are not on disk"
    );
    assert!(!reported, "the dispose's close missed its deadline");
}

/// A `closeCache` that is waiting for the model thread when a dispose comes
/// gets false when the dispose's 2 s leave a state unwritten, even though
/// the writer then drains everything it was handed: with the writer stalled,
/// the dispose writes the newer of two waiting states and runs out of time
/// for the older one.
/// Mutation: the close ignores the model thread's flush and the parker's
/// hand-over → red (true with a state unwritten).
#[test]
fn close_cache_before_a_dispose_reports_a_state_left_unwritten() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk_with(reuse_inner(), &root, "fixture", (64, 0));
    let prompts = [prompt(100, 1), prompt(100, 2)];
    for (seq, tokens) in [1, 2].into_iter().zip(&prompts) {
        turn(&mut inner, seq, tokens);
        finish(&mut inner);
    }
    assert_eq!(count(&inner.telemetry.waiting), 2);
    let older = keys_of(&inner, &prompts[0], ColdGroup::DFlash2State, 3)[2];
    let manager = store(&inner);
    let release = stall_writer(&inner);
    let (woke, _wakes) = std::sync::mpsc::channel::<()>();
    let closer = disk(&inner).closer().upgrade().unwrap();
    closer.set_wake(Arc::new(move || {
        let _ = woke.send(());
    }));
    let closing = std::thread::spawn(move || closer.close(Instant::now() + Duration::from_secs(8)));
    std::thread::sleep(Duration::from_millis(300));
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(3));
        let _ = release.send(());
    });
    drop(inner);
    releaser.join().unwrap();
    let reported = closing.join().unwrap();
    assert!(
        !manager.contains_in(&older, ColdGroup::DFlash2State),
        "the dispose wrote the older state"
    );
    assert!(
        !reported,
        "closeCache reported true with a waiting state unwritten"
    );
}

/// A state write that fails for anything but room stops the tier: it takes
/// no write after, and the turn's row reads `off`, as Qwen3.5 reports it.
/// Mutation: the mode ignores whether the tier still takes writes → red.
#[test]
fn a_failed_state_write_turns_the_mode_off() {
    use std::os::unix::fs::PermissionsExt;
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    disk_mut(&mut inner).delay = Duration::ZERO;
    let handle = disk(&inner).tier_handle();
    assert_eq!(inner.telemetry.mode(&handle), 1);
    let namespace = root.namespace();
    std::fs::set_permissions(&namespace, std::fs::Permissions::from_mode(0o555)).unwrap();
    turn(&mut inner, 1, &prompt(100, 1));
    finish(&mut inner);
    drive(&mut inner, |inner| disk(inner).idle());
    std::fs::set_permissions(&namespace, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(!disk(&inner).tier().is_open());
    assert_eq!(count(&inner.telemetry.disk_enabled), 1);
    assert_eq!(inner.telemetry.mode(&handle), 0);
}

/// After `closeCache` this process removes nothing from the directory it let
/// go of: a turn over a state on disk only whose chain lost a block drops
/// the state from the store and leaves its object where it is.
/// Mutation: a removal does not ask whether the tier still holds its
/// directory → red (the object is gone).
#[test]
fn a_closed_tier_removes_nothing() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let a = first_conversation(&root, &reuse_inner);
    let mut inner = on_disk(reuse_inner(), &root);
    assert_eq!(count(&inner.telemetry.adopted), 1);
    let kv = keys_of(&inner, &a, ColdGroup::Kv, 3);
    assert!(store(&inner).remove(&kv[1]));
    let closer = disk(&inner).closer().upgrade().unwrap();
    assert!(closer.close(Instant::now() + Duration::from_secs(5)));
    assert_eq!(files(&root, "dflash2_state").len(), 1);
    let (cached, _) = turn(&mut inner, 1, &fork(&a, 98, 110, 2));
    finish(&mut inner);
    assert_eq!(cached, 0);
    assert_eq!(count(&inner.telemetry.dropped_chain), 1);
    assert_eq!(
        files(&root, "dflash2_state").len(),
        1,
        "a closed tier removed an object"
    );
}

/// The model thread never measures the cache root (a measure can remove
/// another cache's whole directory): a state write hands the measure to the
/// parker thread.
/// Mutation: the write measures on the model thread → red.
#[test]
fn a_state_write_measures_the_root_off_the_model_thread() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    disk_mut(&mut inner).delay = Duration::ZERO;
    crate::disk_tier::MEASURES.with(|count| count.set(0));
    turn(&mut inner, 1, &prompt(100, 1));
    finish(&mut inner);
    drive(&mut inner, |_| files(&root, "dflash2_state").len() == 1);
    assert_eq!(
        crate::disk_tier::MEASURES.with(std::cell::Cell::get),
        0,
        "the model thread measured the cache root"
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    while disk(&inner).measured() == 0 {
        assert!(Instant::now() < deadline, "the parker never measured");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// A turn whose lineage is under the minimum queues no state and copies no
/// block; one at the minimum writes. A load's minimum is Splash's 2,048.
/// Mutation: `start` without the minimum check → red (the blocks are
/// copied).
#[test]
fn a_state_under_2048_tokens_is_not_parked() {
    assert_eq!(prefix_disk::MINIMUM_ROWS, 2048);
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    {
        let mut inner = reuse_inner();
        crate::disk_tier::TEST_WRITER.with(|writer| writer.set(Some((8, 0))));
        inner.open_prefix_disk(&root.options(), Some("fixture"), None);
        crate::disk_tier::TEST_WRITER.with(|writer| writer.set(None));
        assert_eq!(disk(&inner).minimum, prefix_disk::MINIMUM_ROWS);
    }
    let mut inner = on_disk(reuse_inner(), &root);
    disk_mut(&mut inner).minimum = 128;
    turn(&mut inner, 1, &prompt(100, 1));
    finish(&mut inner);
    assert_eq!(disk(&inner).encoded, 0);
    assert_eq!(count(&inner.telemetry.waiting), 0);
    assert_eq!(count(&inner.telemetry.owed), 0);
    settle(&mut inner);
    assert_eq!(objects(&root), 0);
    disk_mut(&mut inner).minimum = 96;
    turn(&mut inner, 2, &prompt(100, 2));
    finish(&mut inner);
    settle(&mut inner);
    assert_eq!(disk(&inner).encoded, 1);
    assert_eq!(files(&root, "dflash2_state").len(), 1);
}

/// A temporary tier writes nothing while memory drops no state; once it
/// does, the next conversation's chain and state go to the temporary cache,
/// and the state, its RAM copy dropped, comes back from it in the same
/// process, as a cold prefill would give it.
/// Mutation: a temporary tier copies without demand → red.
#[test]
fn a_temporary_tier_writes_only_on_demand() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = reuse_inner();
    open_tier_once(&mut inner, &root.switch_off(), Some("fixture"), (8, 0));
    assert!(!disk(&inner).tier().persistent);
    disk_mut(&mut inner).delay = Duration::ZERO;
    turn(&mut inner, 1, &prompt(100, 1));
    finish(&mut inner);
    for _ in 0..5 {
        inner.step_disk_now();
    }
    assert!(store(&inner).drain(Duration::from_secs(20)));
    assert!(root.temp_objects(".safetensors").is_empty());
    assert_eq!(disk(&inner).encoded, 0);
    inner.prefix_states.counters.dropped_memory += 1;
    let a = prompt(100, 2);
    turn(&mut inner, 2, &a);
    finish(&mut inner);
    let durable = |inner: &Inner| {
        inner.prefix_states.store.entries().any(|entry| {
            entry.tokens() == &a[..96]
                && entry
                    .disk()
                    .is_some_and(|copy| copy.landed && !copy.chain_owed)
        })
    };
    drive(&mut inner, durable);
    assert!(!root.temp_objects(".dflash2_state.safetensors").is_empty());
    let ids: Vec<_> = inner
        .prefix_states
        .store
        .entries()
        .map(|entry| (entry.id(), prefix_state::durable(entry)))
        .collect();
    for (id, kept) in ids {
        if kept {
            inner.prefix_states.store.drop_ram(id);
        } else {
            inner.prefix_states.store.remove(id);
        }
    }
    let b = fork(&a, 98, 110, 3);
    let (cached, logits) = turn(&mut inner, 3, &b);
    assert_eq!(cached, 96);
    assert_eq!(count(&inner.telemetry.disk_restored), 1);
    assert_eq!(logits, cold_on(reuse_inner(), &b).0);
}

/// `closeCache` on a temporary tier writes nothing and removes its
/// directory: a state waiting under demand with the whole delay ahead is
/// never encoded.
/// Mutation: the close leaves a temporary tier open → red (the directory
/// stays).
#[test]
fn a_temporary_cache_goes_at_close() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = reuse_inner();
    open_tier_once(&mut inner, &root.switch_off(), Some("fixture"), (8, 0));
    inner.prefix_states.counters.dropped_memory += 1;
    turn(&mut inner, 1, &prompt(100, 1));
    finish(&mut inner);
    assert_eq!(count(&inner.telemetry.waiting), 1);
    assert_eq!(root.temp_dirs().len(), 1);
    let closer = disk(&inner).closer().upgrade().unwrap();
    assert!(closer.close(Instant::now() + Duration::from_secs(5)));
    assert!(root.temp_dirs().is_empty());
    assert_eq!(disk(&inner).encoded, 0);
    assert_eq!(count(&inner.telemetry.disk_enabled), 0);
}

/// An array's shape and bits; `None` for an absent one.
fn array_bits(array: Option<&MxArray>) -> Option<(Vec<i64>, Vec<u8>)> {
    let array = array?;
    let shape = (0..array.ndim().unwrap())
        .map(|axis| array.shape_at(axis).unwrap())
        .collect();
    let bits = if array.dtype().unwrap() == crate::array::DType::Float32 {
        array
            .to_float32()
            .unwrap()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    } else {
        array
            .to_uint16_native()
            .unwrap()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    Some((shape, bits))
}

/// What a state record holds of `cell`, bit for bit: the grid, each layer's
/// indexer blocks and arrays, the draft's waiting rows and seed.
type RecordBits = (usize, Vec<usize>, Vec<Option<(Vec<i64>, Vec<u8>)>>);

fn record_bits(cell: &prefix_state::Cell) -> RecordBits {
    let mut arrays = Vec::new();
    for layer in &cell.layers {
        arrays.extend([
            array_bits(layer.conv.as_ref()),
            array_bits(layer.recurrent.as_ref()),
            array_bits(layer.ple_conv.as_ref()),
        ]);
    }
    if let Some(draft) = &cell.draft {
        arrays.push(array_bits(draft.waiting.as_ref()));
        arrays.push(array_bits(Some(&draft.seed)));
    }
    let blocks = cell.layers.iter().map(|layer| layer.index_blocks).collect();
    (cell.grid, blocks, arrays)
}

/// Every state record is one fixed span: a state and a smaller one encoded
/// one after the other into one buffer leave it as the smaller one in a
/// fresh buffer, each decodes to its own arrays bit for bit, and a buffer
/// one byte short of a state takes nothing.
/// Mutation: no zeros after the last array → red (the reused buffer keeps
/// the larger state's tail).
#[test]
fn a_state_record_has_one_fixed_span() {
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    assert!(inner.has_mtp);
    turn(&mut inner, 1, &prompt(100, 1));
    finish(&mut inner);
    let store = &inner.prefix_states.store;
    let cell = store
        .entries()
        .find(|entry| entry.depth() == 96)
        .and_then(|entry| entry.ram())
        .unwrap();
    let draft = cell.draft.as_ref().unwrap();
    assert!(draft.waiting.is_some());
    let smaller = prefix_state::Cell {
        grid: cell.grid,
        layers: cell
            .layers
            .iter()
            .map(|layer| prefix_state::LayerCell {
                conv: layer.conv.clone(),
                recurrent: layer.recurrent.clone(),
                ple_conv: layer.ple_conv.clone(),
                index_blocks: layer.index_blocks,
            })
            .collect(),
        draft: Some(prefix_state::DraftCell {
            segments: Vec::new(),
            waiting: None,
            seed: draft.seed.clone(),
        }),
    };
    let span = memory::encoded_state_bytes(&inner.decoder.config, true) as usize;
    let (large, small) = (
        prefix_disk::encoded_len(cell),
        prefix_disk::encoded_len(&smaller),
    );
    assert!(small < large && large <= span, "{small} {large} {span}");
    let mut reused = vec![0u8; span];
    assert!(prefix_disk::encode_cell_into(cell, &mut reused).unwrap());
    let larger = reused.clone();
    assert!(prefix_disk::encode_cell_into(&smaller, &mut reused).unwrap());
    let mut fresh = vec![0u8; span];
    assert!(prefix_disk::encode_cell_into(&smaller, &mut fresh).unwrap());
    assert!(
        reused == fresh,
        "the reused buffer keeps bytes of the last state"
    );
    let layers = cell.layers.len();
    for (bytes, want) in [(&larger, cell), (&fresh, &smaller)] {
        let got = prefix_disk::decode_cell(bytes, layers, true, Vec::new()).unwrap();
        assert!(record_bits(&got) == record_bits(want));
    }
    let mut short = vec![0u8; large - 1];
    assert!(!prefix_disk::encode_cell_into(cell, &mut short).unwrap());
    let mut exact = vec![0u8; large];
    assert!(prefix_disk::encode_cell_into(cell, &mut exact).unwrap());
}

/// One staging buffer: with the writer stalled, four lanes end their turns
/// with their states due and only the first is encoded; the others wait.
/// Released, every state lands in publication order, each one record of the
/// load's fixed span.
/// Mutation: a write takes a buffer of its own while the lane's is out →
/// red.
#[test]
fn one_staging_buffer_serves_every_lane() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk_with(reuse_inner(), &root, "fixture", (64, 0));
    disk_mut(&mut inner).delay = Duration::ZERO;
    let release = stall_writer(&inner);
    let lanes = crate::models::qwen4_exp::LANES as u32;
    let prompts: Vec<Vec<u32>> = (1..=lanes).map(|seq| prompt(70, seq)).collect();
    for (seq, tokens) in (1..=lanes).zip(&prompts) {
        turn(&mut inner, seq, tokens);
        inner.finalize_paged_turn(false, 0);
    }
    assert_eq!(disk(&inner).encoded, 1);
    assert_eq!(count(&inner.telemetry.waiting), u64::from(lanes));
    release.send(()).unwrap();
    drive(&mut inner, |inner| disk(inner).idle());
    let want: Vec<_> = prompts
        .iter()
        .map(|tokens| keys_of(&inner, tokens, ColdGroup::DFlash2State, 2)[1])
        .collect();
    assert_eq!(disk(&inner).landed_states, want);
    assert_eq!(disk(&inner).encoded, lanes as usize);
    // The stalling write's own record lies beside them.
    assert_eq!(files(&root, "dflash2_state").len(), lanes as usize + 1);
    let span = memory::encoded_state_bytes(&inner.decoder.config, inner.has_mtp);
    assert_eq!(disk(&inner).spans(), &[span]);
    let fingerprint = disk(&inner).tier().state_fingerprint;
    for key in want {
        let (layout, label) = store(&inner).state_label(key, fingerprint).unwrap();
        assert_eq!(layout.spans, vec![span]);
        assert_eq!(label.tokens.len(), 64);
    }
}

/// A state counts as durable only with its whole chain, and a state on disk
/// without it is never read: with the copies held back the state lands
/// first; a copy of the directory then (what a crash leaves) restores
/// nothing and its follow-up is a cold one; released, the chain lands and
/// the state is durable.
/// Mutation: a state is durable at its landing → red.
#[test]
fn a_state_is_durable_only_with_its_chain() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    disk_mut(&mut inner).delay = Duration::ZERO;
    disk(&inner).hold(true);
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    drive(&mut inner, |inner| !disk(inner).landed_states.is_empty());
    assert_eq!(files(&root, "dflash2_state").len(), 1);
    assert_eq!(count(&inner.telemetry.durable), 0);
    let crash = Root::new();
    let namespace = root.namespace();
    let copied = crash.cache.0.join(namespace.file_name().unwrap());
    std::fs::create_dir_all(&copied).unwrap();
    for entry in std::fs::read_dir(&namespace).unwrap().flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().ends_with(".safetensors") {
            std::fs::copy(entry.path(), copied.join(&name)).unwrap();
        }
    }
    let b = fork(&a, 98, 110, 2);
    {
        let mut after = on_disk(reuse_inner(), &crash);
        let (cached, logits) = turn(&mut after, 1, &b);
        assert_eq!(cached, 0);
        assert_eq!(count(&after.telemetry.disk_restored), 0);
        assert_eq!(logits, cold_on(reuse_inner(), &b).0);
    }
    disk(&inner).hold(false);
    drive(&mut inner, |inner| count(&inner.telemetry.durable) == 1);
}

/// A written state keeps its copy while its chain can still land: with the
/// copies held in the queue far past the give-up time, the state stays.
/// Mutation: the give-up clock ignores the queued copies → red.
#[test]
fn a_state_keeps_its_copy_while_its_chain_is_queued() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    disk_mut(&mut inner).delay = Duration::ZERO;
    disk_mut(&mut inner).give_up = Duration::from_millis(50);
    disk(&inner).hold(true);
    turn(&mut inner, 1, &prompt(100, 1));
    finish(&mut inner);
    drive(&mut inner, |inner| !disk(inner).landed_states.is_empty());
    std::thread::sleep(Duration::from_millis(200));
    inner.step_disk_now();
    assert_eq!(files(&root, "dflash2_state").len(), 1);
    assert_eq!(count(&inner.telemetry.refused), 0);
    assert!(entry_at(&inner, 96).disk().is_some());
}

/// A chain that can no longer land drops its state: with the queued copies
/// gone the state goes after the give-up time, and nothing stays owed.
/// Mutation: an owed chain is never given up → red.
#[test]
fn a_chain_that_cannot_land_drops_its_state() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    disk_mut(&mut inner).delay = Duration::ZERO;
    disk_mut(&mut inner).give_up = Duration::from_millis(50);
    disk(&inner).hold(true);
    turn(&mut inner, 1, &prompt(100, 1));
    finish(&mut inner);
    drive(&mut inner, |inner| !disk(inner).landed_states.is_empty());
    disk(&inner).discard_queue();
    std::thread::sleep(Duration::from_millis(100));
    drive(&mut inner, |inner| disk(inner).owed_states() == 0);
    assert!(files(&root, "dflash2_state").is_empty());
    assert!(entry_at(&inner, 96).disk().is_none());
    assert_eq!(count(&inner.telemetry.refused), 1);
    assert!(!disk(&inner).plan_owing());
}

/// A persistent tier writes a state its delay after publication while its
/// turn still runs; a temporary one only once the turn ended, under memory
/// demand.
/// Mutation: a point in use waits on a persistent tier too → red.
#[test]
fn a_point_is_copied_while_its_turn_runs() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    disk_mut(&mut inner).delay = Duration::ZERO;
    turn(&mut inner, 1, &prompt(100, 1));
    drive(&mut inner, |_| files(&root, "dflash2_state").len() == 1);

    let root = Root::new();
    let mut inner = reuse_inner();
    open_tier_once(&mut inner, &root.switch_off(), Some("fixture"), (8, 0));
    disk_mut(&mut inner).delay = Duration::ZERO;
    inner.prefix_states.counters.dropped_memory += 1;
    turn(&mut inner, 1, &prompt(100, 1));
    for _ in 0..5 {
        inner.step_disk_now();
    }
    assert!(root.temp_objects(".dflash2_state.safetensors").is_empty());
    assert_eq!(disk(&inner).encoded, 0);
    finish(&mut inner);
    drive(&mut inner, |_| {
        root.temp_objects(".dflash2_state.safetensors").len() == 1
    });
}

/// A conversation that goes on writes only its newest state: the one it
/// went on from is not written, on one sequence and across two.
/// Mutation: a waiting state's conversation is not asked → red (two state
/// objects).
#[test]
fn a_conversation_that_went_on_writes_one_state() {
    let _host = memory::scoped_fixture_host();
    for next in [1, 2] {
        let root = Root::new();
        let mut inner = on_disk(reuse_inner(), &root);
        let a = prompt(70, 1);
        turn(&mut inner, 1, &a);
        finish(&mut inner);
        let b = fork(&a, 68, 100, 3);
        assert_eq!(turn(&mut inner, next, &b).0, 64, "seq {next}");
        finish(&mut inner);
        settle(&mut inner);
        let states = keys_of(&inner, &b, ColdGroup::DFlash2State, 3);
        assert_eq!(files(&root, "dflash2_state").len(), 1, "seq {next}");
        assert!(store(&inner).contains_in(&states[2], ColdGroup::DFlash2State));
        assert_eq!(count(&inner.telemetry.unneeded), 1, "seq {next}");
        assert_eq!(count(&inner.telemetry.durable), 1, "seq {next}");
    }
}

/// `closeCache` writes the waiting states through the model thread, the
/// newest first, and returns by its deadline: two states with their delay
/// ahead are on disk after it, and the closed tier takes nothing more. With
/// a model thread that never answers, the close still returns by its
/// deadline with the turn's blocks on disk, and reports false: its waiting
/// state was not written.
/// Mutation: the model thread's pass ignores a close → red (no state on
/// disk). Mutation: the close ignores the model thread's flush and the
/// parker's hand-over → red (true with the state unwritten).
#[test]
fn close_cache_writes_waiting_points_newest_first() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    assert_eq!(disk(&inner).delay, prefix_disk::DELAY);
    let prompts = [prompt(100, 1), prompt(100, 2)];
    for (seq, tokens) in [1, 2].into_iter().zip(&prompts) {
        turn(&mut inner, seq, tokens);
        finish(&mut inner);
    }
    assert_eq!(count(&inner.telemetry.waiting), 2);
    let (woke, wakes) = std::sync::mpsc::channel::<()>();
    let closer = disk(&inner).closer().upgrade().unwrap();
    closer.set_wake(Arc::new(move || {
        let _ = woke.send(());
    }));
    let closing = std::thread::spawn(move || closer.close(Instant::now() + Duration::from_secs(5)));
    while !closing.is_finished() {
        if wakes.recv_timeout(Duration::from_millis(20)).is_ok() {
            tick(&mut inner);
        }
    }
    assert!(closing.join().unwrap());
    let want: Vec<_> = prompts
        .iter()
        .rev()
        .map(|tokens| keys_of(&inner, tokens, ColdGroup::DFlash2State, 3)[2])
        .collect();
    assert_eq!(disk(&inner).landed_states, want);
    assert_eq!(files(&root, "dflash2_state").len(), 2);
    assert_eq!(count(&inner.telemetry.disk_enabled), 0);
    let (encoded, written) = (disk(&inner).encoded, objects(&root));
    turn(&mut inner, 3, &prompt(100, 3));
    finish(&mut inner);
    inner.step_disk_now();
    assert_eq!(disk(&inner).encoded, encoded);
    assert_eq!(count(&inner.telemetry.owed), 0);
    assert_eq!(objects(&root), written);

    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    let closer = disk(&inner).closer().upgrade().unwrap();
    closer.set_wake(Arc::new(|| {}));
    let started = Instant::now();
    let reported = closer.close(started + Duration::from_secs(2));
    assert!(
        started.elapsed() < Duration::from_millis(2500),
        "{:?}",
        started.elapsed()
    );
    assert!(files(&root, "dflash2_state").is_empty());
    assert!(
        !reported,
        "closeCache reported true with its waiting state unwritten"
    );
    let namespace = root.namespace();
    for key in keys_of(&inner, &a, ColdGroup::Kv, 3) {
        assert!(
            namespace
                .join(format!("{}.safetensors", key.to_hex()))
                .exists()
        );
    }
}

/// A write the tier refuses leaves no disk copy: the staging buffer comes
/// back, the RAM state still serves, and a flush and a dispose do not wait
/// for it. Every write fails: the store keeps a reserve no disk has.
/// Mutation: a failed write keeps its disk copy → red.
#[test]
fn a_refused_write_leaves_no_disk_copy() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk_with(reuse_inner(), &root, "fixture", (8, 1 << 62));
    disk_mut(&mut inner).delay = Duration::ZERO;
    let manager = store(&inner);
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    drive(&mut inner, |inner| disk(inner).idle());
    assert!(entry_at(&inner, 96).disk().is_none());
    assert_eq!(count(&inner.telemetry.refused), 1);
    assert_eq!(count(&inner.telemetry.durable), 0);
    assert!(manager.stats().write_errors > 0);
    let (cached, _) = turn(&mut inner, 3, &fork(&a, 98, 110, 2));
    assert_eq!(cached, 96);
    finish(&mut inner);
    turn(&mut inner, 2, &prompt(100, 2));
    finish(&mut inner);
    drive(&mut inner, |inner| disk(inner).idle());
    assert_eq!(disk(&inner).encoded, 2);
    for round in ["first", "second"] {
        let started = Instant::now();
        inner.flush_disk(started + Duration::from_secs(4));
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{round} flush: {:?}",
            started.elapsed()
        );
    }
    let started = Instant::now();
    inner.disk = None;
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(objects(&root), 0);
}

/// The disk's pass copies blocks out at most once per interval, from the
/// decode steps and the drive passes alike: a copy waits for the GPU.
/// Mutation: every call runs the pass → red.
#[test]
fn decode_steps_copy_out_at_most_once_per_interval() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    for _ in 0..3 {
        inner.top_up_disk();
        tick(&mut inner);
    }
    assert_eq!(disk(&inner).step_runs, 1);
    std::thread::sleep(Duration::from_millis(300));
    tick(&mut inner);
    assert_eq!(disk(&inner).step_runs, 2);
}

/// An idle model is woken for its disk work: a state due later and the
/// chain held back under it are written and made durable with no request,
/// the model thread making one pass per wake.
/// Mutation: the parker never wakes the model for a due point → red.
#[test]
fn an_idle_model_is_woken_for_a_due_point() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    disk_mut(&mut inner).delay = Duration::from_millis(200);
    let (woke, wakes) = std::sync::mpsc::channel::<()>();
    disk(&inner)
        .closer()
        .upgrade()
        .unwrap()
        .set_wake(Arc::new(move || {
            let _ = woke.send(());
        }));
    disk(&inner).hold(true);
    turn(&mut inner, 1, &prompt(100, 1));
    finish(&mut inner);
    assert!(files(&root, "dflash2_state").is_empty());
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut held = true;
    while files(&root, "dflash2_state").len() != 1 || count(&inner.telemetry.durable) != 1 {
        assert!(Instant::now() < deadline, "the model was never woken");
        if wakes.recv_timeout(Duration::from_millis(50)).is_ok() {
            if held {
                disk(&inner).hold(false);
                held = false;
            }
            tick(&mut inner);
        }
    }
}

/// A dispose gives the waiting states its 2 s on the model thread and never
/// waits for anything else: a state with its whole delay ahead is on disk
/// after the model is dropped and restores; with nothing waiting the drop is
/// quick.
/// Mutation: the model's drop does not write the waiting states → red.
#[test]
fn a_dispose_writes_a_waiting_point() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let a = prompt(100, 1);
    let mut inner = on_disk(reuse_inner(), &root);
    disk(&inner)
        .closer()
        .upgrade()
        .unwrap()
        .set_wake(Arc::new(|| {}));
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    assert_eq!(count(&inner.telemetry.waiting), 1);
    let started = Instant::now();
    drop(inner);
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(files(&root, "dflash2_state").len(), 1);
    let mut inner = on_disk(reuse_inner(), &root);
    disk(&inner)
        .closer()
        .upgrade()
        .unwrap()
        .set_wake(Arc::new(|| {}));
    assert_eq!(turn(&mut inner, 1, &fork(&a, 98, 110, 2)).0, 96);
    finish(&mut inner);
    assert_eq!(count(&inner.telemetry.waiting), 0);
    let started = Instant::now();
    drop(inner);
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "{:?}",
        started.elapsed()
    );
}

/// Memory takes a RAM copy and keeps its disk state: with room for one
/// state, a second conversation's publication takes the first one's RAM
/// copy, whose whole chain is on disk; its entry stays on disk only and
/// memory counts no drop. A follow-up of the first conversation reads it
/// back as a cold model computes it. With only states on disk only left,
/// none of them makes room.
/// Mutation: memory removes a state whose chain is on disk → red.
#[test]
fn memory_takes_the_ram_copy_and_keeps_the_disk_state() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    settle(&mut inner);
    let first = entry_at(&inner, 96).id();
    assert!(prefix_state::durable(entry_at(&inner, 96)));
    let others: Vec<_> = inner
        .prefix_states
        .store
        .entries()
        .map(|entry| entry.id())
        .filter(|&id| id != first)
        .collect();
    for id in others {
        inner.prefix_states.store.remove(id);
    }
    let held = inner.prefix_states.bytes();
    let need = memory::state_bytes(&inner.decoder.config, 32, GRID as u64, inner.has_mtp)
        + crate::models::qwen3_5::dflash2_state::store::lineage_bytes(32, 1);
    inner.prefix_states.room_override = Some(held + need - 1);
    turn(&mut inner, 2, &prompt(40, 2));
    finish(&mut inner);
    let kept = inner
        .prefix_states
        .store
        .get(first)
        .expect("the state stays");
    assert!(kept.ram().is_none());
    assert!(prefix_state::durable(kept));
    assert_eq!(inner.prefix_states.counters.dropped_memory, 0);
    inner.prefix_states.room_override = None;
    let b = fork(&a, 98, 110, 2);
    let (cached, logits) = turn(&mut inner, 3, &b);
    assert_eq!(cached, 96);
    assert_eq!(count(&inner.telemetry.disk_restored), 1);
    assert_eq!(logits, cold_on(reuse_inner(), &b).0);
    finish(&mut inner);
    inner.prefix_states.reset(true);
    let left = inner.prefix_states.store.bytes().entries;
    assert!(left > 0);
    assert!(
        inner
            .prefix_states
            .store
            .entries()
            .all(|entry| entry.ram().is_none())
    );
    assert!(inner.prefix_states.room_for(0, 1).is_none());
    assert_eq!(inner.prefix_states.store.bytes().entries, left);
}

/// A state published again at the block of a state on disk only takes its
/// RAM copy back there: one entry at the block, both copies, nothing
/// waiting and nothing encoded again.
/// Mutation: the publication stores the cell as a new state → red (the
/// entry keeps no RAM copy).
#[test]
fn a_republished_state_takes_its_ram_copy_back() {
    use crate::engine::backend::ResetScope;
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = on_disk(reuse_inner(), &root);
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    settle(&mut inner);
    assert!(prefix_state::durable(entry_at(&inner, 96)));
    let encoded = disk(&inner).encoded;
    inner.reset_caches(ResetScope::Command).unwrap();
    assert!(entry_at(&inner, 96).ram().is_none());
    assert_eq!(turn(&mut inner, 1, &a).0, 0);
    finish(&mut inner);
    let at_block: Vec<_> = inner
        .prefix_states
        .store
        .entries()
        .filter(|entry| entry.depth() == 96)
        .collect();
    assert_eq!(at_block.len(), 1);
    assert!(at_block[0].ram().is_some());
    assert!(prefix_state::durable(at_block[0]));
    assert_eq!(count(&inner.telemetry.waiting), 0);
    assert_eq!(disk(&inner).encoded, encoded);
}

/// A restarted persistent load takes back, before any turn, each state
/// whose whole chain is on the tier, as a state on disk only: of two
/// conversations written, the one that lost a KV block stays out.
/// Mutation: the open takes nothing back → red.
#[test]
fn a_restart_adopts_states_with_whole_chains() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let (a, b) = (prompt(100, 1), prompt(100, 2));
    {
        let mut inner = on_disk(reuse_inner(), &root);
        for (seq, tokens) in [(1, &a), (2, &b)] {
            turn(&mut inner, seq, tokens);
            finish(&mut inner);
        }
        settle(&mut inner);
        assert_eq!(files(&root, "dflash2_state").len(), 2);
    }
    let lost = keys_of(&reuse_inner(), &b, ColdGroup::Kv, 3)[1];
    std::fs::remove_file(
        root.namespace()
            .join(format!("{}.safetensors", lost.to_hex())),
    )
    .unwrap();
    let inner = on_disk(reuse_inner(), &root);
    assert_eq!(count(&inner.telemetry.adopted), 1);
    let entries: Vec<_> = inner.prefix_states.store.entries().collect();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].tokens(), &a[..96]);
    assert!(entries[0].ram().is_none());
    assert!(prefix_state::durable(entries[0]));
    assert_eq!(count(&inner.telemetry.states), 0);
}

/// On a load with the MTP head a state without one of its draft segments
/// is not taken back, and its follow-up prefills as a cold one.
/// Mutation: the adoption skips the segment check → red.
#[test]
fn a_state_without_its_draft_segments_is_not_adopted() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let a = first_conversation(&root, &reuse_inner);
    let segments = files(&root, "hybrid_draft");
    assert_eq!(segments.len(), 2);
    std::fs::remove_file(&segments[0]).unwrap();
    let mut inner = on_disk(reuse_inner(), &root);
    assert!(inner.has_mtp);
    assert_eq!(count(&inner.telemetry.adopted), 0);
    let b = fork(&a, 98, 110, 2);
    let (cached, logits) = turn(&mut inner, 1, &b);
    assert_eq!(cached, 0);
    assert_eq!(logits, cold_on(reuse_inner(), &b).0);
}

/// A crash between a write and its landing leaves nothing that is read: a
/// writer's temporary file beside a whole state goes at the reopen and the
/// state comes back; a state file cut short is not taken back and goes; a
/// whole state whose last KV block never landed is not taken back. Each
/// follow-up gives a cold model's logits.
/// Mutation: the adoption skips the chain check → red (the third case
/// takes the state back).
#[test]
fn a_half_written_state_is_never_read() {
    let _host = memory::scoped_fixture_host();
    for case in ["writer temp", "cut short", "last block"] {
        let root = Root::new();
        let a = first_conversation(&root, &reuse_inner);
        let fixture = reuse_inner();
        let state = keys_of(&fixture, &a, ColdGroup::DFlash2State, 3)[2];
        let kv = keys_of(&fixture, &a, ColdGroup::Kv, 3);
        let namespace = root.namespace();
        let state_file = files(&root, "dflash2_state").remove(0);
        let bytes = std::fs::read(&state_file).unwrap();
        let temp = namespace.join(format!(".{}.{}.{}.tmp", state.to_hex(), 4242, 7));
        let want = match case {
            "writer temp" => {
                std::fs::write(&temp, &bytes[..bytes.len() / 2]).unwrap();
                1
            }
            "cut short" => {
                std::fs::write(&state_file, &bytes[..bytes.len() / 2]).unwrap();
                0
            }
            _ => {
                std::fs::remove_file(namespace.join(format!("{}.safetensors", kv[2].to_hex())))
                    .unwrap();
                0
            }
        };
        let mut inner = on_disk(reuse_inner(), &root);
        assert_eq!(count(&inner.telemetry.adopted), want, "{case}");
        assert!(!temp.exists(), "{case}");
        if case == "cut short" {
            assert!(!state_file.exists(), "{case}");
        }
        let b = fork(&a, 98, 110, 2);
        let (cached, logits) = turn(&mut inner, 1, &b);
        assert_eq!(cached, if want == 1 { 96 } else { 0 }, "{case}");
        assert_eq!(logits, cold_on(reuse_inner(), &b).0, "{case}");
    }
}

/// Only a persistent tier with an empty store takes states back: a
/// temporary tier holding a whole state takes back none, and neither does a
/// persistent one whose store holds one.
/// Mutation: the guard does not ask whether the tier is persistent → red.
#[test]
fn a_temporary_tier_adopts_nothing() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let mut inner = reuse_inner();
    open_tier_once(&mut inner, &root.switch_off(), Some("fixture"), (8, 0));
    assert!(!disk(&inner).tier().persistent);
    disk_mut(&mut inner).delay = Duration::ZERO;
    inner.prefix_states.counters.dropped_memory += 1;
    turn(&mut inner, 1, &prompt(100, 1));
    finish(&mut inner);
    drive(&mut inner, |inner| {
        inner
            .prefix_states
            .store
            .entries()
            .any(prefix_state::durable)
    });
    inner.prefix_states.reset(false);
    let config = inner.decoder.config.clone();
    let disk = inner.disk.as_mut().unwrap();
    assert_eq!(disk.adopt(&mut inner.prefix_states.store, &config), 0);

    let root = Root::new();
    first_conversation(&root, &reuse_inner);
    let mut inner = on_disk(reuse_inner(), &root);
    assert_eq!(inner.prefix_states.store.bytes().entries, 1);
    let config = inner.decoder.config.clone();
    let disk = inner.disk.as_mut().unwrap();
    assert_eq!(disk.adopt(&mut inner.prefix_states.store, &config), 0);
    assert_eq!(inner.prefix_states.store.bytes().entries, 1);
}

/// The napi object's handle names the namespace directory its persistent
/// tier opened, read from the tier; after the close, and for a temporary
/// tier, it names none.
/// Mutation: the tier names its namespace after its close → red.
#[test]
fn the_tier_handle_names_the_open_namespace() {
    let _host = memory::scoped_fixture_host();
    let root = Root::new();
    let inner = on_disk(reuse_inner(), &root);
    let handle = disk(&inner).tier_handle();
    let dir = root.namespace();
    assert_eq!(dir, disk(&inner).tier().dir());
    assert_eq!(
        handle.namespace().as_deref(),
        dir.file_name().and_then(|name| name.to_str())
    );
    let closer = disk(&inner).closer().upgrade().unwrap();
    assert!(closer.close(Instant::now() + Duration::from_secs(5)));
    assert_eq!(handle.namespace(), None);
    let mut temporary = reuse_inner();
    open_tier_once(&mut temporary, &root.switch_off(), Some("fixture"), (8, 0));
    assert!(!disk(&temporary).tier().persistent);
    assert_eq!(disk(&temporary).tier_handle().namespace(), None);
}

/// `tokens`' turn on sequence 1 written with its state landing before its
/// chain, so the state is the conversation's oldest object; its entry and
/// state key.
fn written_state_first(
    inner: &mut Inner,
    tokens: &[u32],
) -> (
    crate::models::qwen3_5::dflash2_state::store::EntryId,
    mlx_paged_attn::ColdCacheKey,
) {
    disk_mut(inner).delay = Duration::ZERO;
    disk(inner).hold(true);
    turn(inner, 1, tokens);
    finish(inner);
    drive(inner, |inner| !disk(inner).landed_states.is_empty());
    disk(inner).hold(false);
    drive(inner, |inner| count(&inner.telemetry.durable) == 1);
    let entry = entry_at(inner, 96);
    (entry.id(), entry.disk().unwrap().key)
}

/// A state object the tier's quota takes leaves the store: the next
/// conversation's writes take the first one's state, its oldest object; an
/// entry with its RAM copy loses its disk copy, one on disk only goes and
/// its follow-up reads nothing from disk and gives a cold model's logits.
/// Mutation: the pass does not read the quota's evictions → red.
#[test]
fn an_evicted_state_leaves_the_store() {
    let _host = memory::scoped_fixture_host();
    let a = prompt(100, 1);
    let (unit, block) = {
        let root = Root::new();
        let mut inner = on_disk(reuse_inner(), &root);
        written_state_first(&mut inner, &a);
        let kv = keys_of(&inner, &a, ColdGroup::Kv, 1)[0];
        let block = std::fs::metadata(
            root.namespace()
                .join(format!("{}.safetensors", kv.to_hex())),
        )
        .unwrap()
        .len();
        (store(&inner).total_bytes(), block)
    };
    for disk_only in [false, true] {
        let root = Root::new();
        let options = root.options_of(unit + block / 2);
        let mut inner = on_disk_under(reuse_inner(), &options, "fixture", (8, 0));
        let (first, key) = written_state_first(&mut inner, &a);
        let manager = store(&inner);
        assert!(manager.contains_in(&key, ColdGroup::DFlash2State));
        if disk_only {
            inner.prefix_states.release(first);
            assert!(
                inner
                    .prefix_states
                    .store
                    .get(first)
                    .unwrap()
                    .ram()
                    .is_none()
            );
        }
        turn(&mut inner, 2, &prompt(100, 2));
        finish(&mut inner);
        drive(&mut inner, |inner| {
            !manager.contains_in(&key, ColdGroup::DFlash2State)
                && inner
                    .prefix_states
                    .store
                    .get(first)
                    .is_none_or(|entry| entry.disk().is_none())
        });
        let case = format!("disk only {disk_only}");
        match inner.prefix_states.store.get(first) {
            Some(entry) => {
                assert!(!disk_only, "{case}: the entry stays");
                assert!(entry.ram().is_some(), "{case}");
            }
            None => assert!(disk_only, "{case}: the entry goes"),
        }
        if disk_only {
            let b = fork(&a, 98, 110, 3);
            let (cached, logits) = turn(&mut inner, 3, &b);
            assert!(cached < 96, "{case}: {cached}");
            assert_eq!(count(&inner.telemetry.disk_restored), 0, "{case}");
            assert_eq!(logits, cold_on(reuse_inner(), &b).0, "{case}");
        }
    }
}
