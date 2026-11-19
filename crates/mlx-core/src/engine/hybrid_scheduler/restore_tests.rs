//! A disk-tier restore beside a decoding lane, through the scheduler on a
//! draft load: the restoring start parks while the tier's reader reads its
//! chain and lands on a hot chain. Each request gives its tokens and the
//! GDN cells it publishes alone, and a restored one those of no cache.

use std::collections::HashMap;
use std::sync::atomic::AtomicU32;
use std::sync::{Mutex, OnceLock};

use super::*;
use crate::array::DType;
use crate::engine::governor::governor::Governor;
use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
use crate::models::qwen3_5::dflash2_decode::paged_dflash2_tests::{
    cell_bits, tiny_paged_int8_dflash_inner,
};
use crate::models::qwen3_5::dflash2_decode::tests::{
    content_tokenizer, metal_is_default_device, rendered_prompt, session_start_with,
};
use crate::models::qwen3_5::dflash2_state::disk::tests::{
    TestTier, attach_tier, persistent_test_tier,
};
use crate::models::qwen3_5::dflash2_state::disk::{
    self, adopt_all, close_cache, lineage_keys, state_context,
};
use crate::models::qwen3_5::dflash2_state::store::Lineage;
use crate::models::qwen3_5::dflash2_state::write_behind::Stop;
use crate::models::qwen3_5::dflash2_turn::stored_resume_depth;
use crate::models::qwen3_5::model::dflash2_park_tests::{install_pool, passing_probe};
use crate::models::qwen3_5::model::restore::Qwen35Restore;
use crate::models::qwen3_5::model::{
    AWAIT_READERS, COLD_LANDINGS, Qwen35Inner, REFUSED_LANDINGS, RETRIES,
};
use crate::models::qwen3_5::paged_forward::{
    PREFILL_CHUNK_ROWS_OVERRIDE, PREFILL_PASS_BUDGET_OVERRIDE, PREFILL_TAILS_STORED,
};
use crate::transformer::cold_tier::sync_restore_blocks;

type State = HybridSchedulerState<Qwen35Inner>;
type Command = ModelCommand<<Qwen35Inner as HybridSchedulerBackend>::FamilyCommand>;

const SEED: u64 = 0x5_81C0;
const TIER: u64 = 1 << 30;
const X_MAX: i32 = 120;
const Y_MAX: i32 = 8;

/// `len` prompt tokens of one pattern, from row `from`.
fn rows(pattern: u32, from: u32, len: usize) -> Vec<u32> {
    (from..from + len as u32)
        .map(|i| (i * pattern + 3) % 16)
        .collect()
}

pub(super) fn body(pattern: u32, len: usize) -> Vec<u32> {
    rows(pattern, 0, len)
}

fn x_body() -> Vec<u32> {
    body(3, 40)
}

/// A pool of `initial` blocks that grows to `max` in `extent` steps on the
/// governor's admission.
#[derive(Clone, Copy)]
pub(super) struct Pool {
    initial: u32,
    max: u32,
    extent: u32,
}

/// The tiny dense model with a 40-row DFlash2 draft window and four lanes.
fn fixture_inner() -> Qwen35Inner {
    let mut inner = tiny_paged_int8_dflash_inner(SEED);
    inner.dflash2.as_mut().expect("draft").config.sliding_window = 40;
    inner.grant_scheduled_lanes(4);
    inner.dflash2_wave_ready = true;
    inner
}

/// The tiny dense load with a 40-row DFlash2 draft window, four lanes, a
/// governor on a fake clock and, with `tier`, the state tier.
pub(super) fn load(tier: Option<&TestTier>, pool: Option<Pool>) -> (State, Arc<FakeIo>) {
    load_sharing(tier, pool, None)
}

/// [`load`] whose decoding lanes are owed `decode_share` of a prefill
/// unit's wall (`None`: the load's default).
fn load_sharing(
    tier: Option<&TestTier>,
    pool: Option<Pool>,
    decode_share: Option<f64>,
) -> (State, Arc<FakeIo>) {
    let mut inner = fixture_inner();
    if let Some(share) = decode_share {
        inner.decode_share = share;
    }
    if let Some(pool) = pool {
        install_pool(&mut inner, pool.initial, pool.max, passing_probe()).expect("a small pool");
    }
    if let Some(tier) = tier {
        let window = inner
            .dflash2
            .as_ref()
            .expect("draft")
            .config
            .context_window()
            .expect("the draft window");
        attach_tier(
            &mut inner.dflash2_states,
            &inner.config,
            inner.paged_adapter.as_ref().expect("pool"),
            tier,
            window.nbytes(),
            window.slots(),
        );
        inner
            .paged_adapter
            .as_mut()
            .expect("pool")
            .set_cold_tier(state_context(&tier.tier));
    }
    scheduled(inner, pool)
}

/// `inner` on the content tokenizer and a governor on a fake clock, in a
/// scheduler.
fn scheduled(mut inner: Qwen35Inner, pool: Option<Pool>) -> (State, Arc<FakeIo>) {
    inner.tokenizer = Some(content_tokenizer());
    let io = FakeIo::new(
        stats(0),
        host(32 * G, crate::engine::governor::host::Pressure::Normal),
    );
    let governor: &'static Governor = Box::leak(Box::new(Governor::new(io.clone())));
    inner.memory_registration = Some(governor.register_model(model(16 * G, 0)).expect("register"));
    if let Some(pool) = pool {
        let adapter = inner.paged_adapter.as_mut().expect("pool");
        adapter.set_extent_growth(pool.extent);
        adapter.set_admission(Arc::new(governor));
    }
    let mut state = HybridSchedulerState::new(inner).expect("scheduler");
    state.enable_continuous_batching();
    (state, io)
}

/// A load after a restart whose earlier process ran each of `seeds` alone
/// (one token each) and closed its cache: their states are on disk only.
fn seeded(tag: &str, seeds: &[&[u32]], pool: Option<Pool>) -> (State, Arc<FakeIo>, TestTier) {
    seeded_sharing(tag, seeds, pool, None)
}

/// [`seeded`] on [`load_sharing`]'s load.
fn seeded_sharing(
    tag: &str,
    seeds: &[&[u32]],
    pool: Option<Pool>,
    decode_share: Option<f64>,
) -> (State, Arc<FakeIo>, TestTier) {
    let tier = persistent_test_tier(tag, TIER);
    {
        let (mut state, _io) = load_sharing(Some(&tier), pool, decode_share);
        for (n, seed) in seeds.iter().enumerate() {
            let mut run = Run::new();
            run.send(&format!("seed{n}"), seed, 1);
            run.finish(&mut state);
        }
        assert!(
            close_cache(&mut state.inner, Instant::now() + Duration::from_secs(30)),
            "the seeds reach the tier"
        );
    }
    let tier = tier.reopen(tag, TIER);
    let (mut state, io) = load_sharing(Some(&tier), pool, decode_share);
    assert!(
        adopt_all(&mut state.inner) >= 1,
        "the restart adopts the seeds"
    );
    (state, io, tier)
}

struct Session {
    owner: String,
    reply: tokio::sync::oneshot::Receiver<Result<ChatResult>>,
    cancel: Arc<AtomicBool>,
    done: Option<std::result::Result<ChatResult, String>>,
}

/// Greedy DFlash2 session starts (the app's turn shape on a draft load),
/// driven pass by pass.
pub(super) struct Run {
    sender: tokio::sync::mpsc::UnboundedSender<Command>,
    receiver: tokio::sync::mpsc::UnboundedReceiver<Command>,
    sessions: Vec<Session>,
}

impl Run {
    pub(super) fn new() -> Self {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        Self {
            sender,
            receiver,
            sessions: Vec::new(),
        }
    }

    pub(super) fn send(&mut self, owner: &str, body: &[u32], max: i32) -> usize {
        self.send_with(owner, body, max, |_| {})
    }

    /// [`Self::send`] with `edit` applied to the request's config.
    pub(super) fn send_with(
        &mut self,
        owner: &str,
        body: &[u32],
        max: i32,
        edit: impl FnOnce(&mut crate::engine::types::ChatConfig),
    ) -> usize {
        let (mut command, reply, cancel) =
            session_start_with(owner, body, max, true).expect("a session start");
        if let crate::engine::cmd::ChatCmd::SessionStart { config, .. } = &mut command {
            edit(config);
        }
        assert!(self.sender.send(ModelCommand::from_chat(command)).is_ok());
        self.sessions.push(Session {
            owner: owner.to_string(),
            reply,
            cancel,
            done: None,
        });
        self.sessions.len() - 1
    }

    fn pass(&mut self, state: &mut State) {
        assert!(!self.all_done(), "a pass with nothing to run blocks");
        assert!(matches!(
            state.drive(&mut self.receiver),
            LoopControl::Continue
        ));
        for session in &mut self.sessions {
            if session.done.is_none()
                && let Ok(reply) = session.reply.try_recv()
            {
                session.done = Some(reply.map_err(|error| error.reason.clone()));
            }
        }
    }

    fn all_done(&self) -> bool {
        self.sessions.iter().all(|session| session.done.is_some())
    }

    fn done(&self, owner: &str) -> bool {
        self.sessions
            .iter()
            .any(|session| session.owner == owner && session.done.is_some())
    }

    fn until(&mut self, state: &mut State, mut stop: impl FnMut(&mut State, &Self) -> bool) {
        for _ in 0..4000 {
            if stop(state, self) {
                return;
            }
            self.pass(state);
        }
        panic!("the condition never held");
    }

    pub(super) fn finish(&mut self, state: &mut State) {
        self.until(state, |_, run| run.all_done());
    }

    pub(super) fn result(&self, index: usize) -> &ChatResult {
        let session = &self.sessions[index];
        match session.done.as_ref().expect("the request finished") {
            Ok(result) => result,
            Err(error) => panic!("request {} failed: {error}", session.owner),
        }
    }
}

/// The tokens `owner`'s turn generated so far; `None` once it ended.
fn generated(state: &mut State, owner: &str) -> Option<usize> {
    let seq = *state.owner_sequences.get(owner)?;
    state
        .turn_mut(seq)
        .map(|turn| turn.payload.generated_tokens.len())
}

fn seq_of(state: &State, owner: &str) -> SeqId {
    *state
        .owner_sequences
        .get(owner)
        .expect("the owner's sequence")
}

fn is_parked(state: &State, owner: &str) -> bool {
    state
        .owner_sequences
        .get(owner)
        .is_some_and(|seq| state.pending_restores.contains_key(seq))
}

/// The lane `x` decodes: it has generated 3 tokens.
fn lane(run: &mut Run, state: &mut State) -> usize {
    let x = run.send("x", &x_body(), X_MAX);
    run.until(state, |state, _| {
        generated(state, "x").is_some_and(|n| n >= 3)
    });
    x
}

/// Passes until `owner` has its first token; every pass `lanes` decode
/// in while they run. True when `owner` was seen parked.
fn until_first_token(run: &mut Run, state: &mut State, owner: &str, lanes: &[&str]) -> bool {
    let mut parked = false;
    for _ in 0..4000 {
        if generated(state, owner).is_some_and(|n| n > 0) || run.done(owner) {
            return parked;
        }
        let before: Vec<_> = lanes.iter().map(|lane| generated(state, lane)).collect();
        run.pass(state);
        parked |= is_parked(state, owner);
        for (lane, before) in lanes.iter().zip(before) {
            if let (Some(before), Some(after)) = (before, generated(state, lane)) {
                assert!(
                    after > before,
                    "lane {lane} decodes in every pass while {owner} restores"
                );
            }
        }
    }
    panic!("{owner} never gave a token");
}

fn invalidations(state: &State) -> u64 {
    state
        .inner
        .dflash2_states
        .disk
        .as_ref()
        .map_or(0, |disk| disk.counters.invalidations)
}

fn total_blocks(state: &State) -> u32 {
    state
        .inner
        .paged_adapter
        .as_ref()
        .expect("pool")
        .block_telemetry()
        .expect("telemetry")
        .total_blocks
}

/// Blocks held by anything but `owner`'s request: neither free nor
/// evictable, nor in its block table.
fn held_beside(state: &State, owner: &str) -> u32 {
    let adapter = state.inner.paged_adapter.as_ref().expect("pool");
    let telemetry = adapter.block_telemetry().expect("telemetry");
    let own = adapter
        .block_table_for(seq_of(state, owner))
        .map_or(0, |table| table.num_blocks() as u32);
    telemetry.total_blocks - telemetry.free_blocks - telemetry.evictable_blocks - own
}

fn counted(counter: &'static std::thread::LocalKey<std::cell::Cell<u32>>) -> u32 {
    counter.with(std::cell::Cell::get)
}

/// A published state: its depth, its GDN cell bits and its window rows.
type Published = (usize, Vec<Vec<u32>>, Vec<u32>);

/// What a request gave: its text, its token count and the states with a
/// RAM copy it published on its prompt past `past`.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Gave {
    text: String,
    tokens: u32,
    states: Vec<Published>,
}

fn bits(array: &MxArray) -> Vec<u32> {
    let f = array.astype(DType::Float32).expect("f32");
    f.eval();
    f.to_float32()
        .expect("host values")
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

fn states_past(inner: &Qwen35Inner, prompt: &[u32], past: usize) -> Vec<Published> {
    let mut states: Vec<Published> = inner
        .dflash2_states
        .store
        .entries()
        .filter(|entry| entry.depth() > past && prompt.starts_with(entry.tokens()))
        .filter_map(|entry| {
            let anchor = entry.ram()?;
            let window = anchor.window.as_ref().map_or_else(Vec::new, |window| {
                bits(
                    &window
                        .load_rows(window.begin(), window.end() - window.begin())
                        .expect("window rows"),
                )
            });
            Some((
                entry.depth(),
                cell_bits(&anchor.linear).expect("cells"),
                window,
            ))
        })
        .collect();
    states.sort_by_key(|state| state.0);
    states
}

pub(super) fn gave(run: &Run, index: usize, state: &State, body: &[u32], past: usize) -> Gave {
    let result = run.result(index);
    Gave {
        text: result.raw_text.clone(),
        tokens: result.num_tokens,
        states: states_past(&state.inner, &rendered_prompt(body), past),
    }
}

fn memo(key: String, make: impl FnOnce() -> Gave) -> Gave {
    static MEMO: OnceLock<Mutex<HashMap<String, Gave>>> = OnceLock::new();
    let memo = MEMO.get_or_init(Default::default);
    if let Some(found) = memo.lock().expect("memo").get(&key) {
        return found.clone();
    }
    let made = make();
    memo.lock().expect("memo").insert(key, made.clone());
    made
}

/// `body` for `owner` alone on a load with no cache.
fn no_cache(owner: &str, body: &[u32], max: i32, past: usize) -> Gave {
    memo(format!("none {owner} {body:?} {max} {past}"), || {
        let (mut state, _io) = load(None, None);
        let mut run = Run::new();
        let index = run.send(owner, body, max);
        run.finish(&mut state);
        assert_eq!(run.result(index).cached_tokens, 0);
        gave(&run, index, &state, body, past)
    })
}

/// `body` for `owner` alone after a restart from `seeds`: a lone restore.
fn alone(seeds: &[&[u32]], owner: &str, body: &[u32], max: i32, past: usize) -> Gave {
    memo(
        format!("alone {seeds:?} {owner} {body:?} {max} {past}"),
        || {
            let (mut state, _io, _tier) = seeded("rs-alone", seeds, None);
            let mut run = Run::new();
            let index = run.send(owner, body, max);
            run.finish(&mut state);
            gave(&run, index, &state, body, past)
        },
    )
}

/// The test's switches, reset when it ends.
pub(super) struct Hooks {
    _ctx: StreamContext,
}

impl Hooks {
    pub(super) fn new(name: &str) -> Option<Self> {
        if crate::test_support::paged_prefill_unsupported(name) || !metal_is_default_device() {
            return None;
        }
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
        AWAIT_READERS.with(|on| on.set(true));
        REFUSED_LANDINGS.with(|left| left.set(0));
        Some(Self {
            _ctx: StreamContext::new(Stream::generation()),
        })
    }
}

impl Drop for Hooks {
    fn drop(&mut self) {
        PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
        PREFILL_PASS_BUDGET_OVERRIDE.with(|ms| ms.set(None));
        AWAIT_READERS.with(|on| on.set(false));
        REFUSED_LANDINGS.with(|left| left.set(0));
        mlx_paged_attn::set_restore_chunk_bytes_for_test(None);
        mlx_paged_attn::fail_restore_commits_for_test(0);
    }
}

/// Request y's state (144) is on disk only; it starts beside decoding
/// lane x, parks while the reader reads its nine blocks, and lands on a
/// hot chain: no block is walked on the model thread, x decodes in every
/// pass until y's first token, y resumes at 144 and gives its tokens and
/// cells alone and those of no cache, and x gives its tokens alone.
#[test]
fn a_restore_beside_a_lane_parks_and_the_lane_keeps_stepping() {
    let Some(_hooks) = Hooks::new("a_restore_beside_a_lane_parks_and_the_lane_keeps_stepping")
    else {
        return;
    };
    let (seed, y) = (body(7, 150), body(7, 180));
    let (mut state, _io, _tier) = seeded("rs-s1", &[&seed], None);
    let mut run = Run::new();
    let x = lane(&mut run, &mut state);
    let walks = sync_restore_blocks();
    let yi = run.send("y", &y, Y_MAX);
    let parked = until_first_token(&mut run, &mut state, "y", &["x"]);
    assert!(parked, "the restore beside the lane parks");
    assert_eq!(
        sync_restore_blocks(),
        walks,
        "no block is walked beside the lane"
    );
    run.finish(&mut state);
    assert_eq!(run.result(yi).cached_tokens, 144, "y resumed at its state");
    let got = gave(&run, yi, &state, &y, 144);
    assert!(!got.states.is_empty(), "y published a state past 144");
    assert_eq!(
        got,
        alone(&[&seed], "y", &y, Y_MAX, 144),
        "y equals y alone"
    );
    assert_eq!(got, no_cache("y", &y, Y_MAX, 144), "y equals no cache");
    assert_eq!(
        gave(&run, x, &state, &x_body(), 0),
        no_cache("x", &x_body(), X_MAX, 0),
        "x equals x alone"
    );
}

/// Two restores start together beside lane x, and each prefix unit
/// (a start, a landing) costs 60 ms on the governor's clock: the second
/// start and the second landing each wait for the next pass, so a decode
/// step of x lies between them; all three equal themselves alone.
#[test]
fn two_restores_beside_a_lane_each_follow_a_decode_step() {
    let Some(_hooks) = Hooks::new("two_restores_beside_a_lane_each_follow_a_decode_step") else {
        return;
    };
    let (ys, ws) = (body(7, 150), body(5, 150));
    let (y, w) = (body(7, 180), body(5, 180));
    let seeds: [&[u32]; 2] = [&ys, &ws];
    let (mut state, io, _tier) = seeded("rs-s2", &seeds, None);
    let mut run = Run::new();
    let x = lane(&mut run, &mut state);
    let starts = Arc::new(AtomicU32::new(0));
    let landings = Arc::new(AtomicU32::new(0));
    for (counter, slot) in [(&starts, 0), (&landings, 1)] {
        let (clock, counter) = (Arc::clone(&io), Arc::clone(counter));
        let hook: Box<dyn FnMut() + Send> = Box::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            clock.with(|fake| fake.now += 0.060);
        });
        if slot == 0 {
            state.after_prefix_prepare = Some(hook);
        } else {
            state.after_restore_landing = Some(hook);
        }
    }
    let yi = run.send("y", &y, Y_MAX);
    let wi = run.send("w", &w, Y_MAX);
    for _ in 0..4000 {
        let started = |state: &mut State, owner| generated(state, owner).is_some_and(|n| n > 0);
        if (started(&mut state, "y") || run.done("y"))
            && (started(&mut state, "w") || run.done("w"))
        {
            break;
        }
        let (s0, l0) = (
            starts.load(Ordering::SeqCst),
            landings.load(Ordering::SeqCst),
        );
        let x0 = generated(&mut state, "x");
        run.pass(&mut state);
        assert!(
            starts.load(Ordering::SeqCst) - s0 <= 1,
            "the second restore start waits for the next pass"
        );
        assert!(
            landings.load(Ordering::SeqCst) - l0 <= 1,
            "the second landing waits for the next pass"
        );
        if let (Some(x0), Some(x1)) = (x0, generated(&mut state, "x")) {
            assert!(x1 > x0, "x decodes in every pass");
        }
    }
    assert_eq!(starts.load(Ordering::SeqCst), 2);
    assert_eq!(landings.load(Ordering::SeqCst), 2);
    run.finish(&mut state);
    for (index, owner, prompt) in [(yi, "y", &y), (wi, "w", &w)] {
        assert_eq!(run.result(index).cached_tokens, 144);
        let got = gave(&run, index, &state, prompt, 144);
        assert_eq!(
            got,
            alone(&seeds, owner, prompt, Y_MAX, 144),
            "{owner} alone"
        );
        assert_eq!(got, no_cache(owner, prompt, Y_MAX, 144), "{owner} no cache");
    }
    assert_eq!(
        gave(&run, x, &state, &x_body(), 0),
        no_cache("x", &x_body(), X_MAX, 0)
    );
}

/// With nothing beside it, a restore takes the uncontended path: no turn
/// parks, no start takes the contended prepare, and the walk restores the
/// state's nine blocks on the model thread.
#[test]
fn a_lone_restore_takes_the_whole_walk() {
    let Some(_hooks) = Hooks::new("a_lone_restore_takes_the_whole_walk") else {
        return;
    };
    let (seed, y) = (body(7, 150), body(7, 180));
    let (mut state, _io, _tier) = seeded("rs-s3", &[&seed], None);
    let mut run = Run::new();
    let walks = sync_restore_blocks();
    let yi = run.send("y", &y, Y_MAX);
    let mut parked = false;
    while !run.all_done() {
        run.pass(&mut state);
        parked |= is_parked(&state, "y");
    }
    assert!(!parked, "a lone restore does not park");
    assert_eq!(state.contended_prepares, 0);
    assert_eq!(
        sync_restore_blocks() - walks,
        9,
        "the lone walk restores nine blocks"
    );
    assert_eq!(run.result(yi).cached_tokens, 144);
    assert_eq!(
        gave(&run, yi, &state, &y, 144),
        no_cache("y", &y, Y_MAX, 144)
    );
}

/// Request y is cancelled while parked: its ticket's blocks are free again, no
/// block of its chain is hot, its state is still restorable and not
/// discarded, x equals x alone; the same prompt again restores at 144 and
/// equals no cache.
#[test]
fn a_restore_cancelled_while_parked_leaves_nothing() {
    let Some(_hooks) = Hooks::new("a_restore_cancelled_while_parked_leaves_nothing") else {
        return;
    };
    let (seed, y) = (body(7, 150), body(7, 180));
    let (mut state, _io, _tier) = seeded("rs-s4", &[&seed], None);
    let mut run = Run::new();
    let x = lane(&mut run, &mut state);
    let held = held_beside(&state, "x");
    let yi = run.send("y", &y, Y_MAX);
    run.until(&mut state, |state, _| is_parked(state, "y"));
    run.sessions[yi].cancel.store(true, Ordering::Relaxed);
    run.pass(&mut state);
    assert!(
        run.sessions[yi].done.is_some(),
        "the cancelled request ends"
    );
    assert!(state.pending_restores.is_empty(), "its ticket is gone");
    assert_eq!(held_beside(&state, "x"), held, "its blocks are free again");
    let prompt = rendered_prompt(&y);
    let stored = state
        .inner
        .dflash2_states
        .store
        .entries()
        .find(|entry| entry.depth() == 144 && prompt.starts_with(entry.tokens()))
        .expect("the stored state");
    let adapter = state.inner.paged_adapter.as_ref().expect("pool");
    assert!(
        stored
            .chain()
            .iter()
            .all(|&hash| !adapter.prefix_cached(hash)),
        "no block of its chain is a hit"
    );
    assert_eq!(stored_resume_depth(&state.inner, &prompt, 0), 144);
    assert_eq!(invalidations(&state), 0, "the state was not discarded");
    let again = run.send("y", &y, Y_MAX);
    run.finish(&mut state);
    assert_eq!(run.result(again).cached_tokens, 144);
    assert_eq!(
        gave(&run, again, &state, &y, 144),
        no_cache("y", &y, Y_MAX, 144)
    );
    assert_eq!(
        gave(&run, x, &state, &x_body(), 0),
        no_cache("x", &x_body(), X_MAX, 0)
    );
}

/// The file of block `index` of the chain under the state `depth` deep on
/// `prompt`'s lineage goes from the tier's root; its index entry stays.
fn remove_block_file(state: &State, tier: &TestTier, prompt: &[u32], depth: usize, index: usize) {
    let entry = state
        .inner
        .dflash2_states
        .store
        .entries()
        .find(|entry| entry.depth() == depth && prompt.starts_with(entry.tokens()))
        .expect("the stored state");
    let keys = lineage_keys(
        mlx_paged_attn::ColdGroup::Kv,
        tier.tier.kv_fingerprint,
        Lineage {
            tokens: entry.tokens(),
            extra_keys: entry.extra_keys(),
            salt: entry.salt(),
        },
        entry.block_size(),
    );
    let hex = keys[index].to_hex();
    let root = tier.tier.manager.root().to_path_buf();
    let file = std::fs::read_dir(&root)
        .expect("the tier's root")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(&hex))
        })
        .expect("the block's file");
    std::fs::remove_file(file).expect("remove the block's file");
}

/// Block 6 of y's chain (rows 96-111) is gone from the disk though
/// the index lists it: the reader fails, the commit publishes nothing, the
/// tier forgets the block and the state at 144 is no longer restorable.
/// The landing walks nothing on the model thread, reads nothing again and
/// primes cold: y equals no cache, x equals x alone.
#[test]
fn a_restore_whose_block_is_gone_restarts_cold() {
    let Some(_hooks) = Hooks::new("a_restore_whose_block_is_gone_restarts_cold") else {
        return;
    };
    let (seed, y) = (body(7, 150), body(7, 180));
    let (mut state, _io, tier) = seeded("rs-s5", &[&seed], None);
    let mut run = Run::new();
    let x = lane(&mut run, &mut state);
    remove_block_file(&state, &tier, &rendered_prompt(&y), 144, 6);
    let (walks, retries, colds) = (
        sync_restore_blocks(),
        counted(&RETRIES),
        counted(&COLD_LANDINGS),
    );
    let yi = run.send("y", &y, Y_MAX);
    let parked = until_first_token(&mut run, &mut state, "y", &["x"]);
    assert!(parked);
    run.finish(&mut state);
    assert_eq!(
        sync_restore_blocks(),
        walks,
        "no block is walked beside the lane"
    );
    assert_eq!(
        (
            counted(&RETRIES) - retries,
            counted(&COLD_LANDINGS) - colds,
            run.result(yi).cached_tokens
        ),
        (0, 0, 0),
        "no second read, no forced cold prime: the prime finds no state"
    );
    assert_eq!(
        gave(&run, yi, &state, &y, 144),
        no_cache("y", &y, Y_MAX, 144)
    );
    assert_eq!(
        gave(&run, x, &state, &x_body(), 0),
        no_cache("x", &x_body(), X_MAX, 0)
    );
}

/// Request y's landing is refused memory once, as a claim the ledger makes
/// wait for what lands: y stays parked and is not failed, leaves no armed
/// chain behind, lands no sooner than its wait's retry, and equals y alone
/// and no cache.
#[test]
fn a_restore_refused_memory_at_landing_waits_and_lands() {
    let Some(_hooks) = Hooks::new("a_restore_refused_memory_at_landing_waits_and_lands") else {
        return;
    };
    let (seed, y) = (body(7, 150), body(7, 180));
    let (mut state, io, _tier) = seeded("rs-s6", &[&seed], None);
    let mut run = Run::new();
    let x = lane(&mut run, &mut state);
    let yi = run.send("y", &y, Y_MAX);
    run.until(&mut state, |state, _| is_parked(state, "y"));
    let landings = Arc::new(AtomicU32::new(0));
    let counted_landings = Arc::clone(&landings);
    state.after_restore_landing = Some(Box::new(move || {
        counted_landings.fetch_add(1, Ordering::SeqCst);
    }));
    REFUSED_LANDINGS.with(|left| left.set(1));
    run.pass(&mut state);
    assert_eq!(
        REFUSED_LANDINGS.with(std::cell::Cell::get),
        0,
        "the landing ran"
    );
    assert!(
        is_parked(&state, "y") && !run.done("y"),
        "the refused landing stays parked"
    );
    assert!(
        !state
            .inner
            .paged_adapter
            .as_ref()
            .expect("pool")
            .landed_chain_armed(),
        "the refused landing leaves no armed chain"
    );
    let refused_at = landings.load(Ordering::SeqCst);
    assert!(
        !state.landing_retry_ready(seq_of(&state, "y")),
        "the refusal set a retry"
    );
    run.pass(&mut state);
    assert_eq!(
        landings.load(Ordering::SeqCst),
        refused_at,
        "no landing before its retry"
    );
    run.until(&mut state, |state, _| {
        io.with(|fake| fake.now += 1.0);
        !is_parked(state, "y")
    });
    run.finish(&mut state);
    assert_eq!(run.result(yi).cached_tokens, 144);
    let got = gave(&run, yi, &state, &y, 144);
    assert_eq!(got, alone(&[&seed], "y", &y, Y_MAX, 144));
    assert_eq!(got, no_cache("y", &y, Y_MAX, 144));
    assert_eq!(
        gave(&run, x, &state, &x_body(), 0),
        no_cache("x", &x_body(), X_MAX, 0)
    );
}

/// Request y's chain is read one block a chunk, more chunks than a pass
/// uploads; a peer's prefill unit grows the pool after y's first uploads
/// and before its commit: y lands whole with its first read, walks
/// nothing, and y, the peer and x equal no cache. Lane x is owed no
/// decode time (share 0), so the pass after z's start runs z's first
/// unit whatever the walls of x's prefill and decode steps were.
#[test]
fn a_pool_grow_beside_a_parked_restore_keeps_it() {
    let Some(_hooks) = Hooks::new("a_pool_grow_beside_a_parked_restore_keeps_it") else {
        return;
    };
    let pool = Pool {
        initial: 24,
        max: 256,
        extent: 8,
    };
    let (seed, y, z) = (body(7, 260), body(7, 290), body(5, 200));
    let (mut state, _io, _tier) = seeded_sharing("rs-s7", &[&seed], Some(pool), Some(0.0));
    let depth = stored_resume_depth(&state.inner, &rendered_prompt(&y), 0) as usize;
    assert!(
        depth / 16 > mlx_paged_attn::RESTORE_CHUNKS_PER_PASS,
        "the chain ({depth} rows) spans more chunks than a pass uploads"
    );
    let mut run = Run::new();
    let x = lane(&mut run, &mut state);
    let (walks, retries) = (sync_restore_blocks(), counted(&RETRIES));
    mlx_paged_attn::set_restore_chunk_bytes_for_test(Some(1));
    let yi = run.send("y", &y, Y_MAX);
    run.until(&mut state, |state, _| is_parked(state, "y"));
    let before = total_blocks(&state);
    let zi = run.send("z", &z, Y_MAX);
    run.pass(&mut state);
    let (uploaded, landed) = state.pending_restores[&seq_of(&state, "y")].progress_for_test();
    assert!(
        uploaded > 0 && uploaded < depth / 16 && !landed,
        "y's chain is part uploaded ({uploaded})"
    );
    assert!(
        total_blocks(&state) > before,
        "the peer's unit grew the pool while y was part uploaded"
    );
    run.finish(&mut state);
    assert_eq!(counted(&RETRIES), retries, "y landed with its first read");
    assert_eq!(sync_restore_blocks(), walks);
    assert_eq!(run.result(yi).cached_tokens as usize, depth);
    assert_eq!(
        gave(&run, yi, &state, &y, depth),
        no_cache("y", &y, Y_MAX, depth)
    );
    assert_eq!(gave(&run, zi, &state, &z, 0), no_cache("z", &z, Y_MAX, 0));
    assert_eq!(
        gave(&run, x, &state, &x_body(), 0),
        no_cache("x", &x_body(), X_MAX, 0)
    );
}

/// A landing reads y's state as often as a lone start
/// of y does (its GDN cell, then its window: twice), and never while the
/// staging buffer is out.
#[test]
fn a_landing_reads_its_state_as_a_lone_start_does() {
    let Some(_hooks) = Hooks::new("a_landing_reads_its_state_as_a_lone_start_does") else {
        return;
    };
    let (seed, y) = (body(7, 150), body(7, 180));
    let lone = {
        let (mut state, _io, _tier) = seeded("rs-s9-lone", &[&seed], None);
        let mut run = Run::new();
        let reads = disk::state_reads();
        run.send("y", &y, Y_MAX);
        run.finish(&mut state);
        disk::state_reads() - reads
    };
    let (mut state, _io, _tier) = seeded("rs-s9", &[&seed], None);
    let mut run = Run::new();
    lane(&mut run, &mut state);
    let reads = disk::state_reads();
    let yi = run.send("y", &y, Y_MAX);
    run.finish(&mut state);
    assert_eq!(run.result(yi).cached_tokens, 144);
    assert_eq!(lone, 2, "a lone start reads its cell and its window");
    assert_eq!(
        disk::state_reads() - reads,
        lone,
        "the landing reads the state as a lone start does"
    );
}

/// The staging buffer is out (a state write in flight) when y's
/// chain lands: y stays parked, waits on nothing, and the write-behind
/// stops for it; with the buffer back y lands, its state not discarded,
/// resumes at 144 and equals y alone and no cache.
#[test]
fn a_landing_waits_for_the_staging_buffer_and_stops_the_write_behind() {
    let Some(_hooks) =
        Hooks::new("a_landing_waits_for_the_staging_buffer_and_stops_the_write_behind")
    else {
        return;
    };
    let (seed, y) = (body(7, 150), body(7, 180));
    let (mut state, _io, _tier) = seeded("rs-s10", &[&seed], None);
    let mut run = Run::new();
    let x = lane(&mut run, &mut state);
    let yi = run.send("y", &y, Y_MAX);
    run.until(&mut state, |state, _| is_parked(state, "y"));
    let staging = disk::hold_staging(&mut state.inner.dflash2_states).expect("a free buffer");
    for _ in 0..2 {
        run.pass(&mut state);
        assert!(
            is_parked(&state, "y"),
            "the landing waits while the buffer is out"
        );
        let stopped = state
            .inner
            .dflash2_states
            .disk
            .as_ref()
            .expect("tier")
            .write_behind
            .stopped();
        assert_eq!(
            stopped,
            Stop::ClaimWaiting,
            "the write-behind waits for the read"
        );
    }
    disk::return_staging(&mut state.inner.dflash2_states, staging);
    run.until(&mut state, |state, _| !is_parked(state, "y"));
    let disk_tier = |state: &State| -> u32 {
        state
            .inner
            .dflash2_states
            .disk
            .as_ref()
            .expect("tier")
            .read_waiters()
    };
    assert_eq!(disk_tier(&state), 0, "the landing let go of its read wait");
    run.pass(&mut state);
    assert_ne!(
        state
            .inner
            .dflash2_states
            .disk
            .as_ref()
            .expect("tier")
            .write_behind
            .stopped(),
        Stop::ClaimWaiting,
        "the write-behind runs again after the landing"
    );
    run.finish(&mut state);
    assert_eq!(invalidations(&state), 0, "the state was not discarded");
    assert_eq!(run.result(yi).cached_tokens, 144);
    let got = gave(&run, yi, &state, &y, 144);
    assert_eq!(got, alone(&[&seed], "y", &y, Y_MAX, 144));
    assert_eq!(got, no_cache("y", &y, Y_MAX, 144));
    assert_eq!(
        gave(&run, x, &state, &x_body(), 0),
        no_cache("x", &x_body(), X_MAX, 0)
    );
}

/// Requests y and w share their first 100 rows (six blocks) and both states
/// are on disk only; both park beside x and read the shared blocks into
/// their own destinations: the second commit adopts what the first
/// published, both land with their first read and walk nothing, and each
/// equals itself alone and no cache.
#[test]
fn two_parked_restores_sharing_a_prefix_both_land_whole() {
    let Some(_hooks) = Hooks::new("two_parked_restores_sharing_a_prefix_both_land_whole") else {
        return;
    };
    let shared = body(7, 100);
    let seed_of = |pattern: u32, len: usize| {
        let mut tokens = shared.clone();
        tokens.extend(rows(pattern, 100, len));
        tokens
    };
    let (ys, ws, y, w) = (
        seed_of(5, 50),
        seed_of(3, 50),
        seed_of(5, 80),
        seed_of(3, 80),
    );
    let seeds: [&[u32]; 2] = [&ys, &ws];
    let (mut state, _io, _tier) = seeded("rs-s11", &seeds, None);
    let mut run = Run::new();
    let x = lane(&mut run, &mut state);
    let (walks, retries, colds) = (
        sync_restore_blocks(),
        counted(&RETRIES),
        counted(&COLD_LANDINGS),
    );
    let yi = run.send("y", &y, Y_MAX);
    let wi = run.send("w", &w, Y_MAX);
    let (mut y_parked, mut w_parked) = (false, false);
    while !run.all_done() {
        run.pass(&mut state);
        y_parked |= is_parked(&state, "y");
        w_parked |= is_parked(&state, "w");
    }
    assert!(y_parked && w_parked, "both park");
    assert_eq!(
        counted(&RETRIES),
        retries,
        "both land with their first read"
    );
    assert_eq!(counted(&COLD_LANDINGS), colds);
    assert_eq!(sync_restore_blocks(), walks);
    for (index, owner, prompt) in [(yi, "y", &y), (wi, "w", &w)] {
        assert_eq!(run.result(index).cached_tokens, 144, "{owner} resumed");
        let got = gave(&run, index, &state, prompt, 144);
        assert_eq!(
            got,
            alone(&seeds, owner, prompt, Y_MAX, 144),
            "{owner} alone"
        );
        assert_eq!(got, no_cache(owner, prompt, Y_MAX, 144), "{owner} no cache");
    }
    assert_eq!(
        gave(&run, x, &state, &x_body(), 0),
        no_cache("x", &x_body(), X_MAX, 0)
    );
}

/// Peer z's prefill unit runs in passes of one layer beside x; while
/// a unit of it is held between passes, y's restore start grows the pool:
/// z's held unit goes on, and x, z and y equal no cache.
#[test]
fn a_restore_start_that_grows_the_pool_between_held_passes_keeps_all_exact() {
    let Some(_hooks) =
        Hooks::new("a_restore_start_that_grows_the_pool_between_held_passes_keeps_all_exact")
    else {
        return;
    };
    PREFILL_PASS_BUDGET_OVERRIDE.with(|ms| ms.set(Some(0.0)));
    let pool = Pool {
        initial: 16,
        max: 256,
        extent: 8,
    };
    let (seed, y, z) = (body(7, 150), body(7, 180), body(5, 150));
    let (mut state, _io, _tier) = seeded("rs-s12", &[&seed], Some(pool));
    let mut run = Run::new();
    let x = lane(&mut run, &mut state);
    let zi = run.send("z", &z, Y_MAX);
    let held = |state: &mut State| {
        let Some(&seq) = state.owner_sequences.get("z") else {
            return false;
        };
        state
            .scheduler
            .running()
            .iter()
            .any(|turn| turn.seq_id == seq && turn.prefill_unit.is_some())
    };
    run.until(&mut state, |state, _| held(state));
    let before = total_blocks(&state);
    let yi = run.send("y", &y, Y_MAX);
    run.pass(&mut state);
    assert!(is_parked(&state, "y"), "y parks");
    assert!(total_blocks(&state) > before, "y's start grew the pool");
    assert!(held(&mut state), "z's unit is still held between passes");
    run.finish(&mut state);
    assert_eq!(run.result(yi).cached_tokens, 144);
    assert_eq!(
        gave(&run, yi, &state, &y, 144),
        no_cache("y", &y, Y_MAX, 144)
    );
    assert_eq!(gave(&run, zi, &state, &z, 0), no_cache("z", &z, Y_MAX, 0));
    assert_eq!(
        gave(&run, x, &state, &x_body(), 0),
        no_cache("x", &x_body(), X_MAX, 0)
    );
}

/// The pool has no room for y's chain beside lane x (a peer holds
/// all but four takeable blocks): y's start is refused as any start the
/// pool cannot take, waits (neither parked nor failed) and walks nothing;
/// once the peer lets go, y parks, lands at 144 and equals y alone and no
/// cache.
#[test]
fn a_restore_with_no_room_beside_a_lane_waits_and_walks_nothing() {
    let Some(_hooks) = Hooks::new("a_restore_with_no_room_beside_a_lane_waits_and_walks_nothing")
    else {
        return;
    };
    let (seed, y) = (body(7, 150), body(7, 180));
    let (mut state, io, _tier) = seeded("rs-s13", &[&seed], None);
    let mut run = Run::new();
    let x = lane(&mut run, &mut state);
    let walks = sync_restore_blocks();
    let allocator = Arc::clone(
        state
            .inner
            .paged_adapter
            .as_ref()
            .expect("pool")
            .allocator(),
    );
    let peer = {
        let mut allocator = allocator.lock().expect("allocator");
        let takeable = allocator.free_or_evictable_blocks();
        allocator
            .reserve(takeable.saturating_sub(4), 0)
            .expect("the peer's blocks")
    };
    let yi = run.send("y", &y, Y_MAX);
    for _ in 0..3 {
        let x0 = generated(&mut state, "x");
        run.pass(&mut state);
        io.with(|fake| fake.now += 1.0);
        assert!(
            !is_parked(&state, "y") && !run.done("y"),
            "y waits for room"
        );
        assert_eq!(sync_restore_blocks(), walks, "no block is walked beside x");
        if let (Some(x0), Some(x1)) = (x0, generated(&mut state, "x")) {
            assert!(x1 > x0, "x decodes while y waits");
        }
    }
    {
        let mut allocator = allocator.lock().expect("allocator");
        for block in peer {
            allocator.free(block);
        }
    }
    let mut parked = false;
    run.until(&mut state, |state, run| {
        io.with(|fake| fake.now += 1.0);
        parked |= is_parked(state, "y");
        generated(state, "y").is_some_and(|n| n > 0) || run.done("y")
    });
    assert!(parked, "y parks once there is room");
    run.finish(&mut state);
    assert_eq!(sync_restore_blocks(), walks, "no block is walked beside x");
    assert_eq!(run.result(yi).cached_tokens, 144);
    let got = gave(&run, yi, &state, &y, 144);
    assert_eq!(
        got,
        alone(&[&seed], "y", &y, Y_MAX, 144),
        "y equals y alone"
    );
    assert_eq!(got, no_cache("y", &y, Y_MAX, 144), "y equals no cache");
    assert_eq!(
        gave(&run, x, &state, &x_body(), 0),
        no_cache("x", &x_body(), X_MAX, 0)
    );
}

/// The staging buffer stays out past the wait limit after y's chain
/// landed: y primes from nothing, lets go of its read wait and walks
/// nothing; y and x equal no cache.
#[test]
fn a_landing_whose_staging_buffer_stays_out_primes_cold() {
    let Some(_hooks) = Hooks::new("a_landing_whose_staging_buffer_stays_out_primes_cold") else {
        return;
    };
    let (seed, y) = (body(7, 150), body(7, 180));
    let (mut state, io, _tier) = seeded("rs-s14", &[&seed], None);
    let mut run = Run::new();
    let x = lane(&mut run, &mut state);
    let (walks, colds) = (sync_restore_blocks(), counted(&COLD_LANDINGS));
    let yi = run.send("y", &y, Y_MAX);
    run.until(&mut state, |state, _| is_parked(state, "y"));
    let staging = disk::hold_staging(&mut state.inner.dflash2_states).expect("a free buffer");
    let mut passes = 0;
    while is_parked(&state, "y") {
        assert!(passes < 20, "the read wait ends at its limit");
        io.with(|fake| fake.now += 5.0);
        run.pass(&mut state);
        passes += 1;
    }
    assert!(passes > 1, "y waited for the buffer first");
    assert_eq!(counted(&COLD_LANDINGS) - colds, 1, "y primed from nothing");
    assert_eq!(
        state
            .inner
            .dflash2_states
            .disk
            .as_ref()
            .expect("tier")
            .read_waiters(),
        0
    );
    disk::return_staging(&mut state.inner.dflash2_states, staging);
    run.finish(&mut state);
    assert_eq!(sync_restore_blocks(), walks, "no block is walked beside x");
    assert_eq!(run.result(yi).cached_tokens, 0);
    assert_eq!(
        gave(&run, yi, &state, &y, 144),
        no_cache("y", &y, Y_MAX, 144)
    );
    assert_eq!(
        gave(&run, x, &state, &x_body(), 0),
        no_cache("x", &x_body(), X_MAX, 0)
    );
}

/// Y's first commit fails as a failed read does: the landing starts
/// its one second read, which lands. y walks nothing on the model thread,
/// resumes at 144 and equals y alone and no cache.
#[test]
fn a_failed_first_read_reads_once_more_and_lands_whole() {
    let Some(_hooks) = Hooks::new("a_failed_first_read_reads_once_more_and_lands_whole") else {
        return;
    };
    let (seed, y) = (body(7, 150), body(7, 180));
    let (mut state, _io, _tier) = seeded("rs-s15", &[&seed], None);
    let mut run = Run::new();
    let x = lane(&mut run, &mut state);
    let (walks, retries, colds) = (
        sync_restore_blocks(),
        counted(&RETRIES),
        counted(&COLD_LANDINGS),
    );
    mlx_paged_attn::fail_restore_commits_for_test(1);
    let yi = run.send("y", &y, Y_MAX);
    let parked = until_first_token(&mut run, &mut state, "y", &["x"]);
    assert!(parked);
    run.finish(&mut state);
    assert_eq!(sync_restore_blocks(), walks, "no block is walked beside x");
    assert_eq!(counted(&RETRIES) - retries, 1, "one second read");
    assert_eq!(counted(&COLD_LANDINGS), colds);
    assert_eq!(run.result(yi).cached_tokens, 144);
    let got = gave(&run, yi, &state, &y, 144);
    assert_eq!(
        got,
        alone(&[&seed], "y", &y, Y_MAX, 144),
        "y equals y alone"
    );
    assert_eq!(got, no_cache("y", &y, Y_MAX, 144), "y equals no cache");
    assert_eq!(
        gave(&run, x, &state, &x_body(), 0),
        no_cache("x", &x_body(), X_MAX, 0)
    );
}

/// Three restores park together beside lane x, their chains read one
/// block a chunk, and each poll of a reading restore costs 30 ms. Their
/// uploads share one budget of 12 chunks a pass, the oldest first; the
/// oldest, whose chain came in while the pass's budget was spent, lands
/// first in the next pass, before the later restores' uploads.
#[test]
fn three_parked_restores_share_one_upload_budget_and_land_oldest_first() {
    let Some(_hooks) =
        Hooks::new("three_parked_restores_share_one_upload_budget_and_land_oldest_first")
    else {
        return;
    };
    let owners = ["y1", "y2", "y3"];
    let seeds: Vec<Vec<u32>> = [7, 5, 9].iter().map(|&p| body(p, 150)).collect();
    let prompts: Vec<Vec<u32>> = [7, 5, 9].iter().map(|&p| body(p, 180)).collect();
    let seed_refs: Vec<&[u32]> = seeds.iter().map(Vec::as_slice).collect();
    let (mut state, io, _tier) = seeded("rs-s16", &seed_refs, None);
    let mut run = Run::new();
    let x = lane(&mut run, &mut state);
    mlx_paged_attn::set_restore_chunk_bytes_for_test(Some(1));
    let sent: Vec<usize> = owners
        .iter()
        .zip(&prompts)
        .map(|(owner, prompt)| run.send(owner, prompt, Y_MAX))
        .collect();
    run.until(&mut state, |state, _| is_parked(state, "y1"));
    let progress = |state: &State, owner: &str| {
        state
            .pending_restores
            .get(&seq_of(state, owner))
            .map(Qwen35Restore::progress_for_test)
    };
    for owner in owners {
        assert_eq!(progress(&state, owner), Some((0, false)), "{owner} parked");
    }
    let clock = Arc::clone(&io);
    state.after_restore_poll = Some(Box::new(move || {
        clock.with(|fake| fake.now += 0.030);
    }));
    run.pass(&mut state);
    let (first, second, third) = (
        progress(&state, "y1").expect("y1 parked"),
        progress(&state, "y2").expect("y2 parked"),
        progress(&state, "y3").expect("y3 parked"),
    );
    assert!(first.1, "the oldest restore's chain came in first");
    assert_eq!(third, (0, false), "the newest waits for the budget");
    assert_eq!(
        first.0 + second.0 + third.0,
        mlx_paged_attn::RESTORE_CHUNKS_PER_PASS,
        "one budget a pass for all restores"
    );
    run.pass(&mut state);
    assert!(
        !is_parked(&state, "y1"),
        "the oldest lands in the next pass"
    );
    assert!(is_parked(&state, "y3"));
    state.after_restore_poll = None;
    run.finish(&mut state);
    for ((owner, prompt), index) in owners.iter().zip(&prompts).zip(sent) {
        assert_eq!(run.result(index).cached_tokens, 144, "{owner} resumed");
        assert_eq!(
            gave(&run, index, &state, prompt, 144),
            no_cache(owner, prompt, Y_MAX, 144),
            "{owner} equals no cache"
        );
    }
    assert_eq!(
        gave(&run, x, &state, &x_body(), 0),
        no_cache("x", &x_body(), X_MAX, 0)
    );
}

/// A landed turn whose suffix runs beside lane x in one-layer passes of
/// launches of 64 rows takes the fused GDN row scan on the restored cells
/// and stores its non-final launches' tails, walks nothing, and gives its
/// tokens and cells alone.
#[test]
fn a_landed_suffix_of_64_row_launches_takes_the_row_scan_and_equals_alone() {
    let Some(_hooks) =
        Hooks::new("a_landed_suffix_of_64_row_launches_takes_the_row_scan_and_equals_alone")
    else {
        return;
    };
    if !crate::models::qwen3_5::gated_delta::tests::fused_prefill_launches() {
        return;
    }
    PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(64)));
    PREFILL_PASS_BUDGET_OVERRIDE.with(|ms| ms.set(Some(0.0)));
    let (seed, y) = (body(7, 150), body(7, 340));
    let (mut state, _io, _tier) = seeded("rs-landed-64", &[&seed], None);
    let depth = stored_resume_depth(&state.inner, &rendered_prompt(&y), 0) as usize;
    assert!(depth > 0, "y resumes at a stored state");
    let mut run = Run::new();
    lane(&mut run, &mut state);
    let walks = sync_restore_blocks();
    PREFILL_TAILS_STORED.with(|count| count.set(0));
    // SAFETY: test-only thread-local counter switch.
    unsafe { mlx_sys::mlx_test_kquant_counting(true) };
    let yi = run.send("y", &y, Y_MAX);
    let parked = until_first_token(&mut run, &mut state, "y", &["x"]);
    // SAFETY: NUL-terminated name; test-only counter.
    let scans = unsafe { mlx_sys::mlx_test_kquant_family_count(c"gdn_prefill_row_scan".as_ptr()) };
    // SAFETY: as above.
    unsafe { mlx_sys::mlx_test_kquant_counting(false) };
    assert!(parked, "the restore beside the lane parks");
    assert_eq!(sync_restore_blocks(), walks, "no block is walked beside x");
    assert!(scans > 0, "the landed suffix took no GDN row scan");
    assert!(
        PREFILL_TAILS_STORED.with(std::cell::Cell::get) > 0,
        "the landed suffix stored no tail"
    );
    run.finish(&mut state);
    assert_eq!(run.result(yi).cached_tokens as usize, depth);
    let got = gave(&run, yi, &state, &y, depth);
    assert!(
        !got.states.is_empty(),
        "y published a state past its resume"
    );
    assert_eq!(
        got,
        alone(&[&seed], "y", &y, Y_MAX, depth),
        "y equals y alone"
    );
}

/// Request y lands while peer z's prefill unit is held between one-layer
/// passes beside lane x: x, z and y equal no cache.
#[test]
fn a_landing_between_held_passes_of_a_peer_keeps_all_exact() {
    let Some(_hooks) = Hooks::new("a_landing_between_held_passes_of_a_peer_keeps_all_exact") else {
        return;
    };
    PREFILL_PASS_BUDGET_OVERRIDE.with(|ms| ms.set(Some(0.0)));
    let (seed, y, z) = (body(7, 150), body(7, 180), body(5, 150));
    let (mut state, _io, _tier) = seeded("rs-landing-held", &[&seed], None);
    let mut run = Run::new();
    let x = lane(&mut run, &mut state);
    let landings = Arc::new(AtomicU32::new(0));
    let counted_landings = Arc::clone(&landings);
    state.after_restore_landing = Some(Box::new(move || {
        counted_landings.fetch_add(1, Ordering::SeqCst);
    }));
    let held = |state: &mut State| {
        let Some(&seq) = state.owner_sequences.get("z") else {
            return false;
        };
        state
            .scheduler
            .running()
            .iter()
            .any(|turn| turn.seq_id == seq && turn.prefill_unit.is_some())
    };
    let zi = run.send("z", &z, Y_MAX);
    run.until(&mut state, |state, _| held(state));
    let yi = run.send("y", &y, Y_MAX);
    let mut held_at_landing = false;
    run.until(&mut state, |state, _| {
        if landings.load(Ordering::SeqCst) > 0 {
            return true;
        }
        held_at_landing = held(state);
        false
    });
    assert!(held_at_landing, "z's unit was held when y landed");
    assert!(!is_parked(&state, "y"), "y landed");
    run.finish(&mut state);
    assert_eq!(run.result(yi).cached_tokens, 144);
    assert_eq!(
        gave(&run, yi, &state, &y, 144),
        no_cache("y", &y, Y_MAX, 144)
    );
    assert_eq!(gave(&run, zi, &state, &z, 0), no_cache("z", &z, Y_MAX, 0));
    assert_eq!(
        gave(&run, x, &state, &x_body(), 0),
        no_cache("x", &x_body(), X_MAX, 0)
    );
}

/// The load as a restart opens its state tier under `root`: the product's
/// namespace of the pool's identity, the target's files and the draft's,
/// on activations `activations` (INT8 queries from `from` on, when given).
fn product_tier_load(
    root: &std::path::Path,
    activations: crate::transformer::Activations,
    from: Option<u32>,
) -> (State, Arc<FakeIo>) {
    use crate::array::memory::WeightsResident;
    use crate::cold_tier::{CheckpointFiles, CheckpointLoadGuard};
    use crate::disk_tier::{CacheDisk, DiskTierOptions};
    use crate::models::qwen3_5::dflash2_state::disk::{
        DraftLoadGuard, DraftTier, TierLoad, open_state_disk,
    };
    let mut inner = fixture_inner();
    let adapter = inner.paged_adapter.as_mut().expect("pool");
    adapter.set_activations(activations);
    if let Some(from) = from {
        adapter.set_int8_query_threshold(from);
    }
    let target = root.join("target");
    let draft = root.join("draft");
    let options = DiskTierOptions {
        cache_disk: CacheDisk::Bytes(TIER),
        persistent: true,
        cache_dir: Some(root.join("cache")),
        temp_root: Some(root.join("tmp")),
    };
    let weights = WeightsResident::for_test(1, 4096);
    let files = CheckpointFiles::Dir(&target);
    // The last open's writer lets go of the namespace's lock only when its
    // thread exits, after its close returned.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let mut guard = CheckpointLoadGuard::before_read(files, true);
        guard.record_read();
        let draft_files = DraftLoadGuard::before_read(&draft, true);
        let load = TierLoad {
            family: "qwen3_5",
            config_json: b"{}".to_vec(),
            dense: &inner.config,
            activation: DType::BFloat16,
            draft: Some(DraftTier {
                model: inner.dflash2.as_ref().expect("draft"),
                files: &draft_files,
            }),
            adapter: inner.paged_adapter.as_ref().expect("pool"),
            files,
            weights: &weights,
            guard: &guard,
        };
        let (context, disk) = open_state_disk(&load, &options, inner.model_id).expect("a tier");
        if disk.tier.persistent || Instant::now() >= deadline {
            assert!(disk.tier.persistent, "the namespace's last writer let go");
            inner
                .paged_adapter
                .as_mut()
                .expect("pool")
                .set_cold_tier(context);
            inner.dflash2_states.disk = Some(disk);
            return scheduled(inner, None);
        }
        disk.tier.close(Instant::now());
        drop(disk);
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// A store a load with bf16 queries wrote restores under a load that scores
/// INT8 queries past row 64 (the dense default's rule, its threshold moved
/// into the fixture): the namespace is the same, and each of two restarts
/// adopts the state, restores the request at its stored depth with no cold
/// prefill, scores INT8 queries past the threshold and gives the reply the
/// other restart gives.
/// Mutation q8-kept (the identity names the threshold) → red.
#[test]
fn a_bf16_query_store_restores_under_int8_queries() {
    use crate::transformer::Activations;
    let Some(_hooks) = Hooks::new("a_bf16_query_store_restores_under_int8_queries") else {
        return;
    };
    let root = std::env::temp_dir().join(format!(
        "mlx-rs-mixed-queries-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    for dir in ["target", "draft", "tmp"] {
        std::fs::create_dir_all(root.join(dir)).expect("test dir");
    }
    std::fs::write(root.join("target/model.safetensors"), b"target").expect("target shard");
    std::fs::write(root.join("draft/model.safetensors"), b"draft").expect("draft shard");
    std::fs::write(root.join("draft/config.json"), b"{}").expect("draft config");
    let identity = |state: &State| {
        state
            .inner
            .paged_adapter
            .as_ref()
            .expect("pool")
            .cache_identity()
    };
    let seed = body(7, 150);
    {
        let (mut state, _io) = product_tier_load(&root, Activations::Bf16, None);
        assert_eq!(identity(&state), "Int8", "the writer's identity");
        let mut run = Run::new();
        run.send("seed", &seed, 1);
        run.finish(&mut state);
        assert!(
            close_cache(&mut state.inner, Instant::now() + Duration::from_secs(30)),
            "the seed reaches the tier"
        );
    }
    let mut replies = Vec::new();
    for restart in 0..2 {
        let (mut state, _io) = product_tier_load(&root, Activations::Auto, Some(64));
        if state
            .inner
            .paged_adapter
            .as_ref()
            .expect("pool")
            .int8_queries_from()
            .is_none()
        {
            let _ = std::fs::remove_dir_all(&root);
            return;
        }
        assert!(
            adopt_all(&mut state.inner) >= 1,
            "restart {restart}: the int8-query load adopts the bf16-query state"
        );
        let mut run = Run::new();
        let index = run.send("y", &seed, Y_MAX);
        run.finish(&mut state);
        let result = run.result(index);
        let launches = state
            .inner
            .paged_adapter
            .as_ref()
            .expect("pool")
            .tile_query_launches();
        println!(
            "MIXEDRESTORE restart {restart}: identity {}, cached {}, tokens {}, launches \
             [bf16, int8] {launches:?}",
            identity(&state),
            result.cached_tokens,
            result.num_tokens
        );
        assert!(
            result.cached_tokens >= 144,
            "restart {restart}: restored at {}, not at the stored 144",
            result.cached_tokens
        );
        assert!(launches[1] > 0, "restart {restart}: no INT8-query launch");
        replies.push((
            result.cached_tokens,
            result.raw_text.clone(),
            result.num_tokens,
        ));
        assert!(
            close_cache(&mut state.inner, Instant::now() + Duration::from_secs(30)),
            "restart {restart}: the tier closes"
        );
    }
    assert_eq!(replies[0], replies[1], "a restored turn repeats itself");
    let _ = std::fs::remove_dir_all(&root);
}
