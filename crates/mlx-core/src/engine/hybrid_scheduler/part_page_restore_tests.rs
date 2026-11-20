//! A chat a 16-token store holds 16 rows past its last whole 32-token
//! page, moved into a keys-token-major store, restores in every flow of a
//! draft load at the row it had: a scheduled lane (DFlash2 and
//! autoregressive), a whole DFlash2 turn and a whole autoregressive turn.
//! It gives the tokens and the states the same chat gives restored from the
//! 16-token store, a lane restored beside a decoding lane gives what each
//! gives alone, and the states a turn on it leaves sit on whole pages, where
//! the next turn resumes.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{BlockAllocator, ColdCacheFingerprint, LayerKVPool, PageLayout};

use super::restore_tests::{Hooks, body};
use super::{HybridSchedulerBackend, HybridSchedulerState};
use crate::array::{DType, MxArray};
use crate::disk_tier::source_hold::open_disk_tier_with_source;
use crate::disk_tier::{CacheDisk, DiskTier, DiskTierOptions, PersistentIdentity, open_disk_tier};
use crate::engine::governor::governor::Governor;
use crate::engine::governor::governor::tests::{FakeIo, G, host, model, stats};
use crate::engine::model_command::ModelCommand;
use crate::engine::types::ChatResult;
use crate::model_thread::LoopControl;
use crate::models::quantized_linear::LinearProj;
use crate::models::qwen3_5::dflash2::{DFLASH2_DRAFT_PROPOSALS, tiny_dflash2_model_with_block};
use crate::models::qwen3_5::dflash2_decode::paged_dflash2_tests::cell_bits;
use crate::models::qwen3_5::dflash2_decode::tests::{
    content_tokenizer, rendered_prompt, session_start_with,
};
use crate::models::qwen3_5::dflash2_state::disk::{
    StateDisk, adopt_all, close_cache, state_context,
};
use crate::models::qwen3_5::dflash2_state::migrate::move_source::{MoveShape, Moved, move_source};
use crate::models::qwen3_5::dflash2_state::migrate::source_layout;
use crate::models::qwen3_5::dflash2_state::paged::DFlash2States;
use crate::models::qwen3_5::gdn_sidecar;
use crate::models::qwen3_5::model::Qwen35Inner;
use crate::models::qwen3_5::model::scheduled_mtp::seeded_inner_with_config;
use crate::nn::Linear;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

type State = HybridSchedulerState<Qwen35Inner>;
type Command = ModelCommand<<Qwen35Inner as HybridSchedulerBackend>::FamilyCommand>;

const SEED: u64 = 0x5_81D0;
const VOCAB: u32 = 16;
const TIER: u64 = 1 << 30;
/// Rows of every pool here, at either page.
const CONTEXT: u32 = 4096;
/// The 16-token store's namespace, and the one it moves into.
const SOURCE: &str = "16161616161616161616161616161616";
const MOVED: &str = "32323232323232323232323232323232";
/// The stored state the requests resume at: 4 pages and 16 rows.
const RESUME: usize = 144;
const Y_MAX: i32 = 8;
const X_MAX: i32 = 120;

/// The page of a load's pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Page {
    /// The vLLM page of 16 tokens, the store's writer.
    Block16,
    /// 32-token int8 pages, keys token-major.
    KeysTokenMajor,
}

impl Page {
    fn block(self) -> u32 {
        match self {
            Self::Block16 => 16,
            Self::KeysTokenMajor => 32,
        }
    }
}

/// How the model thread runs a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Flow {
    /// A scheduled DFlash2 lane (the app's text turn).
    Lane,
    /// A scheduled lane with its draft off.
    ArLane,
    /// A whole DFlash2 turn in the exclusive lane.
    Whole,
    /// A whole autoregressive turn on the draft load.
    ArWhole,
}

impl Flow {
    fn speculate(self) -> bool {
        matches!(self, Self::Lane | Self::Whole)
    }

    fn whole(self) -> bool {
        matches!(self, Self::Whole | Self::ArWhole)
    }
}

/// A fresh test root, removed on drop.
struct Root(PathBuf);

impl Root {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "mlx-part-page-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(path.join("tmp")).expect("temp root");
        std::fs::create_dir_all(path.join("cache")).expect("cache root");
        Self(path)
    }

    fn options(&self) -> DiskTierOptions {
        DiskTierOptions {
            cache_disk: CacheDisk::Bytes(TIER),
            persistent: true,
            cache_dir: Some(self.0.join("cache")),
            temp_root: Some(self.0.join("tmp")),
        }
    }

    /// A root whose cache is a copy of this one's.
    fn copy(&self, tag: &str) -> Self {
        let root = Self::new(tag);
        copy_dir(&self.0.join("cache"), &root.0.join("cache"));
        root
    }
}

impl Drop for Root {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("a copy's dir");
    for entry in std::fs::read_dir(from).expect("a dir to copy") {
        let entry = entry.expect("a dir entry");
        let target = to.join(entry.file_name());
        if entry.file_type().expect("a file type").is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).expect("a file copy");
        }
    }
}

fn identity(namespace: &str) -> PersistentIdentity {
    let of = |part: &[u8]| ColdCacheFingerprint::from_components([namespace.as_bytes(), part]);
    PersistentIdentity {
        namespace: namespace.to_string(),
        state: of(b" state"),
        kv: of(b" kv"),
    }
}

/// The persistent tier of `page` under `root`: the 16-token store, or the
/// keys-token-major store holding the 16-token one as its source.
fn open_tier(root: &Root, page: Page) -> Arc<DiskTier> {
    let options = root.options();
    // The last open's writer lets go of the namespace's lock only when its
    // thread exits, after its close returned.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let tier = match page {
            Page::Block16 => open_disk_tier(&options, || Some(identity(SOURCE)), 1 << 20),
            Page::KeysTokenMajor => open_disk_tier_with_source(
                &options,
                || Some(identity(MOVED)),
                || Some(identity(SOURCE)),
                1 << 20,
            ),
        }
        .expect("a tier");
        let ready =
            tier.persistent && (page == Page::Block16 || tier.with_source(|_| ()).is_some());
        if ready || Instant::now() >= deadline {
            assert!(ready, "the namespaces' last writer let go");
            return tier;
        }
        tier.close(Instant::now());
        drop(tier);
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The tiny dense draft load with the 27B's attention heads (24 / 4, head
/// 256) on an int8 pool of `page`: a 40-row draft window, four lanes.
fn inner_on(page: Page) -> Qwen35Inner {
    let mut inner = seeded_inner_with_config(SEED, |config| {
        config.num_heads = 24;
        config.num_kv_heads = 4;
        config.head_dim = 256;
        config.hidden_size = 256;
        config.kv_format = Some("int8".to_string());
        config.linear_key_head_dim = 128;
        config.linear_value_head_dim = 128;
        config.linear_num_key_heads = 2;
        config.linear_num_value_heads = 2;
    });
    let mut head =
        Linear::new(inner.config.hidden_size as u32, VOCAB, Some(false)).expect("tiny head");
    head.set_weight(
        &MxArray::random_normal(
            &[i64::from(VOCAB), inner.config.hidden_size as i64],
            0.0,
            0.02,
            Some(DType::BFloat16),
        )
        .expect("tiny head weights"),
    )
    .expect("install tiny head");
    inner.lm_head = Some(LinearProj::Standard(head));
    let mut draft = tiny_dflash2_model_with_block(&inner.config, DFLASH2_DRAFT_PROPOSALS)
        .expect("tiny 7-proposal draft");
    draft
        .validate_target(&inner.config)
        .expect("draft fits target");
    draft.config.sliding_window = 40;
    inner.dflash2 = Some(draft);
    inner.grant_scheduled_lanes(4);
    inner.dflash2_wave_ready = true;

    let old = inner.paged_adapter.take().expect("a pool");
    let block = page.block();
    let blocks = CONTEXT / block;
    let mut config = old.layer_kv_pool().config().clone();
    config.block_size = block;
    config.max_seq_len = Some(CONTEXT);
    let layout = PageLayout {
        keys_token_major: page == Page::KeysTokenMajor,
        ..PageLayout::default()
    };
    let pool = LayerKVPool::new_with_layout(config, blocks, blocks, MetalDtype::Int8, layout)
        .expect("an int8 pool");
    let allocator = Arc::new(Mutex::new(BlockAllocator::with_recency(
        blocks,
        blocks,
        block,
        Arc::clone(&inner.cache_recency),
    )));
    let mut adapter =
        PagedKVCacheAdapter::new(allocator, Arc::new(pool), block).expect("an adapter");
    adapter.set_activations(inner.activations);
    if let Some(ceiling) = old.memory_ceiling() {
        adapter.set_memory_ceiling(ceiling);
    }
    adapter.set_admission(Arc::new(crate::engine::governor::governor()));
    drop(old);
    inner.paged_adapter = Some(adapter);
    inner.dflash2_states = DFlash2States::new(block, Arc::clone(&inner.cache_recency));
    inner.sync_gdn_state_marks();
    inner
}

/// `inner`'s states on `tier`.
fn attach(inner: &mut Qwen35Inner, tier: &Arc<DiskTier>) {
    let window = inner
        .dflash2
        .as_ref()
        .expect("draft")
        .config
        .context_window()
        .expect("the draft window");
    let gdn = gdn_sidecar::geometry(&inner.config, DType::BFloat16).expect("a GDN geometry");
    let block_bytes = mlx_paged_attn::block_object_bytes(
        inner.paged_adapter.as_ref().expect("pool").layer_kv_pool(),
    );
    inner.dflash2_states.disk = Some(
        StateDisk::new(
            Arc::clone(tier),
            gdn,
            Some(window.nbytes()),
            window.slots(),
            "test geometry".to_string(),
            block_bytes,
        )
        .expect("a state disk"),
    );
    inner
        .paged_adapter
        .as_mut()
        .expect("pool")
        .set_cold_tier(state_context(tier));
}

/// `inner` on the content tokenizer and a governor on a fake clock, in a
/// scheduler that runs whole turns when `whole`.
fn scheduled(mut inner: Qwen35Inner, whole: bool) -> State {
    inner.tokenizer = Some(content_tokenizer());
    let io = FakeIo::new(
        stats(0),
        host(32 * G, crate::engine::governor::host::Pressure::Normal),
    );
    let governor: &'static Governor = Box::leak(Box::new(Governor::new(io)));
    inner.memory_registration = Some(governor.register_model(model(16 * G, 0)).expect("register"));
    let mut state = HybridSchedulerState::new(inner).expect("scheduler");
    if whole {
        state.serve_whole_turns();
    } else {
        state.enable_continuous_batching();
    }
    state
}

struct Session {
    owner: String,
    reply: tokio::sync::oneshot::Receiver<napi::bindgen_prelude::Result<ChatResult>>,
    done: Option<std::result::Result<ChatResult, String>>,
}

/// Greedy session starts, driven pass by pass.
struct Run {
    sender: tokio::sync::mpsc::UnboundedSender<Command>,
    receiver: tokio::sync::mpsc::UnboundedReceiver<Command>,
    sessions: Vec<Session>,
}

impl Run {
    fn new() -> Self {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        Self {
            sender,
            receiver,
            sessions: Vec::new(),
        }
    }

    fn send(&mut self, owner: &str, body: &[u32], max: i32, speculate: bool) -> usize {
        let (command, reply, _cancel) =
            session_start_with(owner, body, max, speculate).expect("a session start");
        assert!(self.sender.send(ModelCommand::from_chat(command)).is_ok());
        self.sessions.push(Session {
            owner: owner.to_string(),
            reply,
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

    fn finish(&mut self, state: &mut State) {
        self.until(state, |_, run| run.all_done());
    }

    fn result(&self, index: usize) -> &ChatResult {
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

fn is_parked(state: &State, owner: &str) -> bool {
    state
        .owner_sequences
        .get(owner)
        .is_some_and(|seq| state.pending_restores.contains_key(seq))
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

/// A published state: its depth, its GDN cell bits and its window rows.
type Published = (usize, Vec<Vec<u32>>, Vec<u32>);

/// What a request gave: the rows it resumed at, its text, its token count
/// and the states with a RAM copy it published on its prompt past `past`.
#[derive(Clone, Debug, PartialEq)]
struct Gave {
    cached: u32,
    text: String,
    tokens: u32,
    states: Vec<Published>,
}

impl Gave {
    fn depths(&self) -> Vec<usize> {
        self.states.iter().map(|state| state.0).collect()
    }

    fn state_at(&self, depth: usize) -> Option<&Published> {
        self.states.iter().find(|state| state.0 == depth)
    }
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

fn gave_of(run: &Run, index: usize, state: &State, body: &[u32], past: usize) -> Gave {
    let result = run.result(index);
    Gave {
        cached: result.cached_tokens,
        text: result.raw_text.clone(),
        tokens: result.num_tokens,
        states: states_past(&state.inner, &rendered_prompt(body), past),
    }
}

/// `body` for `owner` alone on `state`.
fn alone(state: &mut State, owner: &str, body: &[u32], max: i32, flow: Flow, past: usize) -> Gave {
    let mut run = Run::new();
    let index = run.send(owner, body, max, flow.speculate());
    run.finish(state);
    gave_of(&run, index, state, body, past)
}

fn closed(state: &mut State) {
    assert!(
        close_cache(&mut state.inner, Instant::now() + Duration::from_secs(30)),
        "the tier closes"
    );
}

/// The 16-token store a block-16 draft load wrote for one seed chat
/// (body 7, 150 rows, one token), its states 16 rows past whole 32-token
/// pages.
fn source_store(tag: &str) -> Root {
    let root = Root::new(tag);
    let tier = open_tier(&root, Page::Block16);
    let mut inner = inner_on(Page::Block16);
    attach(&mut inner, &tier);
    let mut state = scheduled(inner, false);
    let seed = body(7, 150);
    let mut run = Run::new();
    run.send("seed", &seed, 1, true);
    run.finish(&mut state);
    let depths: Vec<usize> = state
        .inner
        .dflash2_states
        .store
        .entries()
        .map(|entry| entry.depth())
        .collect();
    assert!(
        depths.contains(&RESUME) && depths.iter().all(|depth| depth % 32 == 16),
        "the seed's states sit 16 rows past whole pages: {depths:?}"
    );
    closed(&mut state);
    drop(state);
    drop(tier);
    root
}

/// A restart of `page` on `root`'s store, its scheduler running whole turns
/// when `whole`: the keys-token-major load first moves the 16-token store
/// into its own (what it moved is returned); then the load adopts its states.
fn restart(root: &Root, page: Page, whole: bool) -> (State, Option<Moved>) {
    let tier = open_tier(root, page);
    let mut inner = inner_on(page);
    attach(&mut inner, &tier);
    let moved = (page == Page::KeysTokenMajor).then(|| {
        let disk = inner.dflash2_states.disk.as_ref().expect("a tier");
        let config = inner
            .paged_adapter
            .as_ref()
            .expect("pool")
            .layer_kv_pool()
            .config();
        let layout = source_layout(config.num_layers, config.num_kv_heads, config.head_size)
            .expect("the 16-token layout");
        move_source(
            &disk.tier,
            &MoveShape {
                spans: &disk.spans,
                geometry: &disk.geometry,
                source_layout: layout,
            },
        )
    });
    assert!(adopt_all(&mut inner) >= 1, "the restart adopts the store");
    (scheduled(inner, whole), moved)
}

/// The resumed chat `y` (body 7, 180 rows) in `flow` after a restart of
/// `page` on a copy of `source`, alone.
fn restored(source: &Root, page: Page, flow: Flow, tag: &str) -> (Gave, Option<Moved>) {
    let root = source.copy(tag);
    let (mut state, moved) = restart(&root, page, flow.whole());
    let got = alone(&mut state, "y", &body(7, 180), Y_MAX, flow, RESUME);
    closed(&mut state);
    drop(state);
    (got, moved)
}

fn hash(words: impl IntoIterator<Item = u32>) -> u64 {
    words.into_iter().fold(0xcbf2_9ce4_8422_2325, |hash, word| {
        (hash ^ u64::from(word)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn state_hash(state: &Published) -> u64 {
    hash(state.1.iter().flatten().chain(&state.2).copied())
}

/// The moved chat restores in `flow` with the cached rows of the old
/// store, its tokens and the states both loads publish at one depth.
fn restores_as_the_16_token_store(flow: Flow, tag: &str) {
    let source = source_store(tag);
    let (old, _) = restored(&source, Page::Block16, flow, "old");
    let (new, moved) = restored(&source, Page::KeysTokenMajor, flow, "new");
    let moved = moved.expect("the move ran");
    let common: Vec<usize> = new
        .depths()
        .into_iter()
        .filter(|depth| old.state_at(*depth).is_some())
        .collect();
    println!(
        "PART_PAGE_FLOW {flow:?}: cached old {} new {}; tokens old {} new {}; text equal {}; \
         depths old {:?} new {:?}; common {common:?}; hashes old {:?} new {:?}; moved {moved:?}",
        old.cached,
        new.cached,
        old.tokens,
        new.tokens,
        old.text == new.text,
        old.depths(),
        new.depths(),
        common
            .iter()
            .filter_map(|depth| old.state_at(*depth).map(state_hash))
            .collect::<Vec<_>>(),
        common
            .iter()
            .filter_map(|depth| new.state_at(*depth).map(state_hash))
            .collect::<Vec<_>>(),
    );
    assert!(
        moved.fragments > 0 && moved.refused == 0 && moved.dropped == 0 && moved.source_removed,
        "{flow:?}: the move converted the store whole: {moved:?}"
    );
    assert_eq!(
        old.cached, RESUME as u32,
        "{flow:?}: the 16-token store restores at its row"
    );
    assert_eq!(
        new.cached, old.cached,
        "{flow:?}: the moved chat restores at the old row"
    );
    assert!(
        new.text == old.text && new.tokens == old.tokens,
        "{flow:?}: the moved chat gives the tokens of the 16-token store"
    );
    assert!(
        !common.is_empty(),
        "{flow:?}: both loads publish a state at one depth"
    );
    for depth in common {
        assert!(
            new.state_at(depth) == old.state_at(depth),
            "{flow:?}: the state at {depth} is the 16-token store's"
        );
    }
}

/// The moved chat in a scheduled DFlash2 lane (the app's text turn).
#[test]
fn a_moved_part_page_chat_restores_in_a_scheduled_lane() {
    let Some(_hooks) = Hooks::new("a_moved_part_page_chat_restores_in_a_scheduled_lane") else {
        return;
    };
    restores_as_the_16_token_store(Flow::Lane, "lane");
}

#[test]
fn a_moved_part_page_chat_restores_in_a_scheduled_ar_lane() {
    let Some(_hooks) = Hooks::new("a_moved_part_page_chat_restores_in_a_scheduled_ar_lane") else {
        return;
    };
    restores_as_the_16_token_store(Flow::ArLane, "ar-lane");
}

#[test]
fn a_moved_part_page_chat_restores_in_a_whole_dflash2_turn() {
    let Some(_hooks) = Hooks::new("a_moved_part_page_chat_restores_in_a_whole_dflash2_turn") else {
        return;
    };
    restores_as_the_16_token_store(Flow::Whole, "whole");
}

#[test]
fn a_moved_part_page_chat_restores_in_a_whole_ar_turn() {
    let Some(_hooks) = Hooks::new("a_moved_part_page_chat_restores_in_a_whole_ar_turn") else {
        return;
    };
    restores_as_the_16_token_store(Flow::ArWhole, "ar-whole");
}

/// A turn resumed 16 rows past a whole page leaves its states on whole
/// pages, and the next turn of a longer prompt resumes at the deepest of
/// them, not at the part page.
#[test]
fn a_turn_on_a_part_page_leaves_whole_page_states_the_next_turn_resumes() {
    let Some(_hooks) =
        Hooks::new("a_turn_on_a_part_page_leaves_whole_page_states_the_next_turn_resumes")
    else {
        return;
    };
    let source = source_store("next");
    let root = source.copy("next-moved");
    let (mut state, _) = restart(&root, Page::KeysTokenMajor, false);
    let y = alone(&mut state, "y", &body(7, 180), Y_MAX, Flow::Lane, RESUME);
    assert_eq!(y.cached, RESUME as u32, "y resumed at the part page");
    let depths = y.depths();
    println!("PART_PAGE_NEXT y states past {RESUME}: {depths:?}");
    assert!(
        !depths.is_empty() && depths.iter().all(|depth| depth % 32 == 0),
        "the turn on the part page leaves states on whole pages: {depths:?}"
    );
    let z_body = body(7, 220);
    let z_prompt = rendered_prompt(&z_body);
    let deepest = depths
        .iter()
        .copied()
        .filter(|&depth| depth < z_prompt.len())
        .max()
        .expect("a state on z's prompt");
    let z = alone(&mut state, "z", &z_body, Y_MAX, Flow::Lane, deepest);
    println!("PART_PAGE_NEXT z cached {} (deepest {deepest})", z.cached);
    assert_eq!(
        z.cached as usize, deepest,
        "the next turn resumes from the state the turn on the part page left"
    );
    closed(&mut state);
}

/// Lane x decodes; y, whose state is 16 rows past a whole page on disk
/// only, starts beside it, parks while the reader reads its pages and lands:
/// y resumes at its row and gives its tokens and states alone, and x gives
/// its tokens alone.
#[test]
fn a_part_page_restore_beside_a_lane_gives_both_lanes_alone() {
    let Some(_hooks) = Hooks::new("a_part_page_restore_beside_a_lane_gives_both_lanes_alone")
    else {
        return;
    };
    let x_body = body(3, 40);
    let y_body = body(7, 180);
    let source = source_store("beside");
    let (y_alone, _) = restored(&source, Page::KeysTokenMajor, Flow::Lane, "beside-alone");
    assert_eq!(y_alone.cached, RESUME as u32);
    let x_alone = {
        let mut state = scheduled(inner_on(Page::KeysTokenMajor), false);
        alone(&mut state, "x", &x_body, X_MAX, Flow::Lane, 0)
    };

    let root = source.copy("beside-both");
    let (mut state, _) = restart(&root, Page::KeysTokenMajor, false);
    let mut run = Run::new();
    let x = run.send("x", &x_body, X_MAX, true);
    run.until(&mut state, |state, _| {
        generated(state, "x").is_some_and(|n| n >= 3)
    });
    let y = run.send("y", &y_body, Y_MAX, true);
    let mut parked = false;
    run.until(&mut state, |state, run| {
        parked |= is_parked(state, "y");
        generated(state, "y").is_some_and(|n| n > 0) || run.done("y")
    });
    run.finish(&mut state);
    let y_got = gave_of(&run, y, &state, &y_body, RESUME);
    let x_got = gave_of(&run, x, &state, &x_body, 0);
    println!(
        "PART_PAGE_BESIDE parked {parked}; y cached {} tokens {} depths {:?}; x tokens {}",
        y_got.cached,
        y_got.tokens,
        y_got.depths(),
        x_got.tokens
    );
    assert!(parked, "the restore beside the lane parks");
    assert_eq!(y_got.cached, RESUME as u32, "y resumed at its state");
    assert!(
        y_got == y_alone,
        "y beside the lane gives what it gives alone"
    );
    assert!(x_got == x_alone, "x gives its tokens alone");
    closed(&mut state);
}

/// The device the product loads below plan on: a 16 GiB working set, no
/// active memory, no other pool.
struct PlannedMemory;

impl PlannedMemory {
    fn set() -> Self {
        crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|live| {
            live.set(Some(crate::engine::memory_plan::LiveMemory {
                working_set_bytes: 16 << 30,
                active_bytes: 0,
                registered_pool_bytes: 0,
            }))
        });
        Self
    }
}

impl Drop for PlannedMemory {
    fn drop(&mut self) {
        crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|live| live.set(None));
    }
}

/// The tiny dense draft load of [`inner_on`] with its pool built by the
/// load's own page rule (`initialize_paged_adapter`), `explicit` as its
/// config's `paged_block_size`.
fn product_inner(explicit: Option<u32>) -> Qwen35Inner {
    let _memory = PlannedMemory::set();
    let mut inner = seeded_inner_with_config(SEED, |config| {
        config.num_heads = 24;
        config.num_kv_heads = 4;
        config.head_dim = 256;
        config.hidden_size = 256;
        config.kv_format = Some("int8".to_string());
        config.linear_key_head_dim = 128;
        config.linear_value_head_dim = 128;
        config.linear_num_key_heads = 2;
        config.linear_num_value_heads = 2;
        config.paged_block_size = explicit;
    });
    let mut head =
        Linear::new(inner.config.hidden_size as u32, VOCAB, Some(false)).expect("tiny head");
    head.set_weight(
        &MxArray::random_normal(
            &[i64::from(VOCAB), inner.config.hidden_size as i64],
            0.0,
            0.02,
            Some(DType::BFloat16),
        )
        .expect("tiny head weights"),
    )
    .expect("install tiny head");
    inner.lm_head = Some(LinearProj::Standard(head));
    let mut draft = tiny_dflash2_model_with_block(&inner.config, DFLASH2_DRAFT_PROPOSALS)
        .expect("tiny 7-proposal draft");
    draft
        .validate_target(&inner.config)
        .expect("draft fits target");
    draft.config.sliding_window = 40;
    inner.dflash2 = Some(draft);
    inner.paged_adapter = None;
    inner
        .initialize_paged_adapter()
        .expect("the load's own pool");
    inner.grant_scheduled_lanes(4);
    inner.dflash2_wave_ready = true;
    inner
}

/// The model files the product loads below fingerprint.
struct ModelFiles(PathBuf);

impl ModelFiles {
    fn new(tag: &str) -> Self {
        let files = Self(std::env::temp_dir().join(format!(
            "mlx-part-page-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        )));
        for dir in ["target", "draft"] {
            std::fs::create_dir_all(files.0.join(dir)).expect("a model dir");
        }
        std::fs::write(files.0.join("target/model.safetensors"), b"target").expect("target");
        std::fs::write(files.0.join("draft/model.safetensors"), b"draft").expect("draft");
        std::fs::write(files.0.join("draft/config.json"), b"{}").expect("draft config");
        files
    }
}

impl Drop for ModelFiles {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Opens `inner`'s state tier under `root` as the load does
/// (`open_state_disk`: its pool's identity, `files`' target and draft; on
/// the page with token-major keys the move of the 16-token store first),
/// then takes its states back; returns how many came back.
fn product_open(root: &Root, files: &ModelFiles, inner: &mut Qwen35Inner) -> u64 {
    use crate::array::memory::WeightsResident;
    use crate::cold_tier::{CheckpointFiles, CheckpointLoadGuard};
    use crate::models::qwen3_5::dflash2_state::disk::{
        DraftLoadGuard, DraftTier, TierLoad, open_state_disk,
    };
    let target = files.0.join("target");
    let draft = files.0.join("draft");
    let weights = WeightsResident::for_test(1, 4096);
    let checkpoint = CheckpointFiles::Dir(&target);
    // The last open's writer lets go of the namespace's lock only when its
    // thread exits, after its close returned.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let opened = {
            let mut guard = CheckpointLoadGuard::before_read(checkpoint, true);
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
                files: checkpoint,
                weights: &weights,
                guard: &guard,
            };
            open_state_disk(&load, &root.options(), inner.model_id).expect("a tier")
        };
        let (context, disk) = opened;
        if disk.tier.persistent || Instant::now() >= deadline {
            assert!(disk.tier.persistent, "the namespace's last writer let go");
            inner
                .paged_adapter
                .as_mut()
                .expect("pool")
                .set_cold_tier(context);
            inner.dflash2_states.disk = Some(disk);
            return adopt_all(inner);
        }
        disk.tier.close(Instant::now());
        drop(disk);
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn page_of(inner: &Qwen35Inner) -> (u32, bool, String) {
    let adapter = inner.paged_adapter.as_ref().expect("pool");
    (
        adapter.block_size(),
        adapter.layer_kv_pool().layout().keys_token_major,
        adapter.cache_identity(),
    )
}

/// The load lines printed from `mark` on.
fn load_lines_from(mark: usize) -> Vec<String> {
    crate::utils::gguf_planes::LOAD_LINES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(mark..)
        .map(<[String]>::to_vec)
        .unwrap_or_default()
}

fn load_lines_mark() -> usize {
    crate::utils::gguf_planes::LOAD_LINES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .len()
}

/// The namespace directories of `root`'s cache.
fn namespaces(root: &Root) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(root.0.join("cache"))
        .expect("the cache root")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            entry
                .file_type()
                .ok()?
                .is_dir()
                .then(|| entry.file_name().to_string_lossy().into_owned())
        })
        .collect();
    names.sort();
    names
}

/// End to end on the load path: a draft load whose config names
/// `paged_block_size` 16 writes a chat 16 rows past a whole page; the next
/// load takes the page with token-major keys by its own rule, moves the
/// 16-token store at its first open (its load line counts the move), and
/// restores the chat at the row the 16-token load restores it, with its
/// tokens; a restart converts nothing and restores it again, and one
/// namespace is left.
/// Mutation no-move (the load does not move the store it holds) → red.
#[test]
fn a_block_16_store_moves_at_the_first_load_and_restores_at_its_row() {
    use crate::models::qwen3_5::dflash2_state::migrate::move_source::hooks;
    let Some(_hooks) =
        Hooks::new("a_block_16_store_moves_at_the_first_load_and_restores_at_its_row")
    else {
        return;
    };
    if !crate::models::qwen3_5::load_plan::device_builds_tile() {
        return;
    }
    let files = ModelFiles::new("load-move-files");
    let root = Root::new("load-move");
    {
        let mut inner = product_inner(Some(16));
        assert_eq!(
            page_of(&inner),
            (16, false, "Int8".to_string()),
            "the writer's page"
        );
        product_open(&root, &files, &mut inner);
        let mut state = scheduled(inner, false);
        let mut run = Run::new();
        run.send("seed", &body(7, 150), 1, true);
        run.finish(&mut state);
        assert!(
            state
                .inner
                .dflash2_states
                .store
                .entries()
                .any(|entry| entry.depth() == RESUME),
            "the seed's state sits 16 rows past a whole page"
        );
        closed(&mut state);
    }
    let source_namespaces = namespaces(&root);
    let old = {
        let copy = root.copy("load-move-old");
        let mut inner = product_inner(Some(16));
        assert!(
            product_open(&copy, &files, &mut inner) >= 1,
            "the 16-token load adopts"
        );
        let mut state = scheduled(inner, false);
        let got = alone(&mut state, "y", &body(7, 180), Y_MAX, Flow::Lane, RESUME);
        closed(&mut state);
        got
    };
    let _ = hooks::take_conversions();
    let mark = load_lines_mark();
    let new = {
        let mut inner = product_inner(None);
        assert_eq!(
            page_of(&inner),
            (32, true, "Int8 ktm".to_string()),
            "the draft load's own page"
        );
        let adopted = product_open(&root, &files, &mut inner);
        let lines = load_lines_from(mark);
        let moved = lines
            .iter()
            .find(|line| line.starts_with("Qwen3.5 state store move: "))
            .cloned();
        println!(
            "LOAD_MOVE lines {lines:?}; adopted {adopted}; conversions {}; namespaces before \
             {source_namespaces:?} after {:?}",
            hooks::take_conversions(),
            namespaces(&root)
        );
        let moved = moved.expect("the first load moves the 16-token store");
        for part in [
            " states=1 ",
            " fragments=1 ",
            " refused=0 ",
            " unreadable=0 ",
            " dropped=0 ",
            " held_back=0 ",
            " stopped=false ",
            " source_removed=true ",
        ] {
            assert!(moved.contains(part), "the move line has{part}: {moved}");
        }
        assert!(
            lines
                .iter()
                .any(|line| line == "Qwen3.5 KV page: block_size=32 keys=token-major"),
            "the load line names the page"
        );
        assert!(adopted >= 1, "the moved state comes back");
        assert_eq!(namespaces(&root).len(), 1, "the source namespace is gone");
        let mut state = scheduled(inner, false);
        let got = alone(&mut state, "y", &body(7, 180), Y_MAX, Flow::Lane, RESUME);
        closed(&mut state);
        got
    };
    println!(
        "LOAD_MOVE y cached old {} new {}; tokens old {} new {}; text equal {}",
        old.cached,
        new.cached,
        old.tokens,
        new.tokens,
        old.text == new.text
    );
    assert_eq!(
        old.cached, RESUME as u32,
        "the 16-token load restores at its row"
    );
    assert_eq!(
        new.cached, old.cached,
        "the moved chat restores at the old row"
    );
    assert!(
        new.text == old.text && new.tokens == old.tokens,
        "the moved chat gives the tokens of the 16-token store"
    );
    let mark = load_lines_mark();
    let mut inner = product_inner(None);
    let adopted = product_open(&root, &files, &mut inner);
    let conversions = hooks::take_conversions();
    let moved_again = load_lines_from(mark)
        .iter()
        .any(|line| line.starts_with("Qwen3.5 state store move: "));
    let mut state = scheduled(inner, false);
    let again = alone(&mut state, "w", &body(7, 150), Y_MAX, Flow::Lane, RESUME);
    println!(
        "LOAD_MOVE restart: adopted {adopted}, conversions {conversions}, moved {moved_again}, \
         w cached {}",
        again.cached
    );
    assert_eq!(conversions, 0, "a restart converts nothing");
    assert!(!moved_again, "a restart holds no source");
    assert!(adopted >= 1, "the restart adopts");
    assert_eq!(
        again.cached, RESUME as u32,
        "the restart restores the chat at its row"
    );
    closed(&mut state);
}

/// The page with token-major keys names its store apart: its identity
/// `Int8 ktm` makes the 27B's namespace text `kv 32 16 4 256 Int8 ktm`
/// against `kv 32 16 4 256 Int8` on a 32-token vLLM page, so a store of
/// either page sits in its own namespace and neither load takes back what
/// the other wrote.
/// Mutation ktm-dropped (the identity leaves ` ktm` out) → red.
#[test]
fn the_token_major_page_and_the_32_token_vllm_page_never_adopt_each_other() {
    use crate::disk_tier::{CacheNamespace, cache_namespace, cache_namespace_text};
    let Some(_hooks) =
        Hooks::new("the_token_major_page_and_the_32_token_vllm_page_never_adopt_each_other")
    else {
        return;
    };
    if !crate::models::qwen3_5::load_plan::device_builds_tile() {
        return;
    }
    let files = ModelFiles::new("pages-apart-files");
    let target = ColdCacheFingerprint::from_components([b"target".as_slice()]);
    let namespace_27b = |identity: &str| {
        let ns = CacheNamespace {
            target,
            draft: None,
            kv: (32, 16, 4, 256, identity),
            gdn: [48, 3, 10240, 48, 128, 128],
        };
        (cache_namespace_text(&ns), cache_namespace(&ns).0)
    };
    for (writer, reader) in [(Some(32), None), (None, Some(32))] {
        let root = Root::new("pages-apart");
        let writer_identity = {
            let mut inner = product_inner(writer);
            let (block, ktm, identity) = page_of(&inner);
            println!("PAGES_APART writer {writer:?}: block {block} ktm {ktm} identity {identity}");
            assert_eq!((block, ktm), (32, writer.is_none()), "the writer's page");
            product_open(&root, &files, &mut inner);
            let mut state = scheduled(inner, false);
            let mut run = Run::new();
            run.send("seed", &body(7, 150), 1, true);
            run.finish(&mut state);
            closed(&mut state);
            identity
        };
        let mut inner = product_inner(reader);
        let reader_identity = page_of(&inner).2;
        let adopted = product_open(&root, &files, &mut inner);
        let mut state = scheduled(inner, false);
        let got = alone(&mut state, "y", &body(7, 180), Y_MAX, Flow::Lane, 0);
        println!(
            "PAGES_APART reader {reader:?}: identity {reader_identity}, adopted {adopted}, \
             namespaces {:?}, y cached {}",
            namespaces(&root),
            got.cached
        );
        closed(&mut state);
        assert_eq!(
            namespaces(&root).len(),
            2,
            "the two pages keep two namespaces"
        );
        assert_eq!(
            adopted, 0,
            "the {reader:?} load adopts nothing of the other page"
        );
        assert_eq!(got.cached, 0, "nothing restores across the pages");
        let (token_major, vllm) = if writer.is_none() {
            (writer_identity, reader_identity)
        } else {
            (reader_identity, writer_identity)
        };
        let (token_major_text, token_major_name) = namespace_27b(&token_major);
        let (vllm_text, vllm_name) = namespace_27b(&vllm);
        println!(
            "PAGES_APART 27B kv lines: {:?} / {:?}",
            token_major_text.lines().nth(3),
            vllm_text.lines().nth(3)
        );
        assert!(
            token_major_text.contains("\nkv 32 16 4 256 Int8 ktm\n")
                && vllm_text.contains("\nkv 32 16 4 256 Int8\n"),
            "the 27B's namespace texts"
        );
        assert_ne!(token_major_name, vllm_name, "the 27B's two namespaces");
    }
}

/// The plan of the page with token-major keys holds the bytes of the
/// 16-token page's plan in half the blocks: max blocks are the 16-token page's / 2 (+- 1),
/// the window and the lanes are equal.
#[test]
fn the_token_major_page_plans_half_the_blocks_and_the_same_window() {
    let Some(_hooks) = Hooks::new("the_token_major_page_plans_half_the_blocks_and_the_same_window")
    else {
        return;
    };
    if !crate::models::qwen3_5::load_plan::device_builds_tile() {
        return;
    }
    let plan = |explicit| {
        let inner = product_inner(explicit);
        let pool = inner
            .memory_plan
            .as_ref()
            .and_then(|plan| plan.pool.as_ref())
            .expect("a planned pool");
        (
            page_of(&inner).0,
            pool.max_blocks,
            pool.lane.window.maximum(),
            pool.lanes,
        )
    };
    let today = plan(Some(16));
    let new = plan(None);
    println!("PAGE_PLAN today {today:?} new {new:?}");
    assert_eq!((today.0, new.0), (16, 32), "the pages");
    assert!(
        new.1.abs_diff(today.1 / 2) <= 1,
        "max blocks {} against today's {} / 2",
        new.1,
        today.1
    );
    assert_eq!(new.2, today.2, "the window");
    assert_eq!(new.3, today.3, "the lanes");
}

/// Greedy DFlash2 lanes on the page with token-major keys, with INT8
/// queries from row 64 on (each lane crosses it at its own cycle): in waves
/// of 2, 3 and 4 lanes every lane gives its reply and its states alone, and
/// the waves scored rows on both sides of the threshold.
#[test]
fn waves_on_the_token_major_page_equal_each_lane_alone() {
    use crate::transformer::Activations;
    const MAX: i32 = 32;
    let Some(_hooks) = Hooks::new("waves_on_the_token_major_page_equal_each_lane_alone") else {
        return;
    };
    if !crate::models::qwen3_5::load_plan::device_builds_tile() {
        return;
    }
    let fresh = |int8: bool| {
        let mut inner = product_inner(None);
        assert!(page_of(&inner).1, "the page with token-major keys");
        let adapter = inner.paged_adapter.as_mut().expect("pool");
        if int8 {
            adapter.set_activations(Activations::Auto);
            adapter.int8_queries_from()?;
            adapter.set_int8_query_threshold(64);
        } else {
            adapter.set_activations(Activations::Bf16);
        }
        Some(scheduled(inner, false))
    };
    let bodies: Vec<Vec<u32>> = [(3, 40), (5, 44), (7, 48), (11, 52)]
        .into_iter()
        .map(|(pattern, len)| body(pattern, len))
        .collect();
    for int8 in [false, true] {
        let mut alone_lanes = Vec::new();
        for (lane, prompt) in bodies.iter().enumerate() {
            let Some(mut state) = fresh(int8) else {
                return;
            };
            alone_lanes.push(alone(
                &mut state,
                &format!("lane{lane}"),
                prompt,
                MAX,
                Flow::Lane,
                0,
            ));
        }
        for width in [2usize, 3, 4] {
            let mut state = fresh(int8).expect("as the lanes alone");
            let mut run = Run::new();
            let indices: Vec<usize> = bodies[..width]
                .iter()
                .enumerate()
                .map(|(lane, prompt)| run.send(&format!("lane{lane}"), prompt, MAX, true))
                .collect();
            run.finish(&mut state);
            let occupancy = state.scheduler.stats().max_batch_occupancy;
            let launches = state
                .inner
                .paged_adapter
                .as_ref()
                .expect("pool")
                .tile_query_launches();
            println!(
                "KTM_WAVE int8 {int8} width {width}: occupancy {occupancy}, launches [bf16, \
                 int8] {launches:?}"
            );
            assert_eq!(occupancy, width, "{width} lanes share a step");
            if int8 {
                assert!(
                    launches[0] > 0 && launches[1] > 0,
                    "{width} lanes: rows on both sides of the threshold {launches:?}"
                );
            }
            for (lane, &index) in indices.iter().enumerate() {
                assert!(
                    gave_of(&run, index, &state, &bodies[lane], 0) == alone_lanes[lane],
                    "int8 {int8}: lane {lane} of {width} equals the lane alone"
                );
            }
        }
    }
}
