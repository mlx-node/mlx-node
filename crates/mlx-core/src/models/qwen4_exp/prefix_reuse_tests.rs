//! Warm prefix reuse on the paged fixture (with the MTP head): 32-row
//! passes (one KV block), checkpoints every 32 rows, a 128-token window;
//! the prompt-cut tests run 64-row passes.
use super::*;
use crate::engine::backend::{PagedBackend, PagedPrefix};
use crate::engine::hybrid_scheduler::HybridSchedulerBackend;

pub(super) const GRID: usize = 32;

/// The assistant opening of the fixture's tests (`prompt` never makes 62
/// or 63).
const OPEN: [u32; 2] = [62, 63];

pub(super) fn gpu() -> Stream {
    Stream::default(crate::stream::DeviceType::Gpu)
}

/// The paged fixture with 32-row passes and a checkpoint every 32 rows.
pub(super) fn reuse_inner() -> Inner {
    reuse_over(tiny_inner())
}

pub(super) fn reuse_over(mut inner: Inner) -> Inner {
    inner.decoder.prefill_window_pin = Some(GRID);
    inner.decoder.refresh_prefill_window();
    inner.prefix_states.interval = GRID;
    inner
}

/// `len` fixture token ids in `3..62`, distinct per `seed`.
pub(super) fn prompt(len: usize, seed: u32) -> Vec<u32> {
    (0..len as u32)
        .map(|i| 3 + (i * 7 + seed * 13) % 59)
        .collect()
}

/// `base[..at]` then `len - at` tokens of another conversation.
pub(super) fn fork(base: &[u32], at: usize, len: usize, seed: u32) -> Vec<u32> {
    let mut tokens = base[..at].to_vec();
    tokens.extend(prompt(len - at, seed));
    tokens
}

/// Primes and prefills `tokens` on `seq`; the cached prefix and the logits.
pub(super) fn turn(inner: &mut Inner, seq: u32, tokens: &[u32]) -> (usize, Vec<f32>) {
    inner.activate_paged_seq(seq).unwrap();
    let prefix = inner.prime_prefix_state(tokens, true, 32, &[], 0).unwrap();
    let cached = prefix.effective_cached_prefix_len();
    let logits = inner
        .paged_prefill(&tokens[cached..], &prefix, gpu())
        .unwrap();
    (cached, f32s(&logits))
}

/// Ends the active turn as the epilogue does: the row stays live.
pub(super) fn finish(inner: &mut Inner) {
    inner.finalize_paged_turn(true, 0);
    inner.save_paged_history(&[], &[], false, true).unwrap();
}

/// `count` greedy steps after `logits`, each step's logits.
pub(super) fn greedy(inner: &mut Inner, logits: &[f32], count: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    let mut last = logits.to_vec();
    for _ in 0..count {
        let token = last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap();
        last = f32s(&inner.decoder.step(token).unwrap());
        out.push(last.clone());
    }
    out
}

/// The cold reference: a fresh model given the whole prompt.
fn cold(tokens: &[u32]) -> (Vec<f32>, Vec<Vec<f32>>) {
    cold_on(reuse_inner(), tokens)
}

pub(super) fn cold_on(mut inner: Inner, tokens: &[u32]) -> (Vec<f32>, Vec<Vec<f32>>) {
    let (cached, logits) = turn(&mut inner, 1, tokens);
    assert_eq!(cached, 0);
    let steps = greedy(&mut inner, &logits, 8);
    (logits, steps)
}

fn depths(inner: &Inner) -> Vec<(usize, bool)> {
    let mut depths: Vec<_> = inner
        .prefix_states
        .store
        .entries()
        .map(|entry| (entry.depth(), entry.checkpoint()))
        .collect();
    depths.sort_unstable();
    depths
}

/// A follow-up whose prompt leaves the row's history (the next render drops
/// the reasoning) resumes at the stored replay point: it reports that row
/// as cached and gives a cold model's logits and greedy steps bit for bit,
/// on a bf16 pool, on an int8 pool (head 128), and with the passes a
/// refused gate halves past 64 rows.
/// Mutation: `restore_cell` leaves the recurrent matrices out → red.
#[test]
fn a_follow_up_turn_resumes_at_the_stored_state_and_matches_a_cold_turn() {
    use crate::transformer::KvFormat;
    let _host = memory::scoped_fixture_host();
    let dir = Temp::new();
    write_wide_head_fixture(&dir.0);
    let halved = || {
        let mut inner = reuse_inner();
        let config = Arc::clone(&inner.decoder.config);
        inner.decoder.host_prefill_scratch = Some(memory::prefill_scratch(&config, 32, 64));
        inner
    };
    let cases: [(&str, &dyn Fn() -> Inner); 3] = [
        ("bf16", &reuse_inner),
        ("int8", &|| {
            reuse_over(wide_head_inner(&dir.0, KvFormat::Int8))
        }),
        ("halved", &halved),
    ];
    for (name, make) in cases {
        let mut inner = make();
        let a = prompt(100, 1);
        assert_eq!(turn(&mut inner, 1, &a).0, 0, "{name}");
        finish(&mut inner);
        assert_eq!(depths(&inner), [(64, true), (96, false)], "{name}");
        let b = fork(&a, 98, 110, 2);
        let (cached, logits) = turn(&mut inner, 1, &b);
        assert_eq!(cached, 96, "{name}");
        assert_eq!(inner.prefix_states.counters.restored, 1, "{name}");
        if name == "halved" {
            assert!(
                inner
                    .decoder
                    .pass_samples
                    .iter()
                    .any(|(rows, _)| *rows < GRID),
                "the passes past 64 rows are halved"
            );
        }
        let steps = greedy(&mut inner, &logits, 8);
        let (want_logits, want_steps) = cold_on(make(), &b);
        assert_eq!(logits, want_logits, "{name}: first logits");
        assert_eq!(steps, want_steps, "{name}: greedy steps");
    }
}

/// A prompt that leaves the stored lineage resumes at the last state before
/// the difference, and equals cold there; one that leaves it before every
/// state is cold.
/// Mutation: the lookup takes the deepest state of any lineage → red (96
/// cached).
#[test]
fn a_prompt_that_differs_inside_the_prefix_resumes_at_the_last_state_before_it() {
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    let b = fork(&a, 80, 110, 3);
    let (cached, logits) = turn(&mut inner, 2, &b);
    assert_eq!(cached, 64);
    let steps = greedy(&mut inner, &logits, 8);
    let (want_logits, want_steps) = cold(&b);
    assert_eq!(logits, want_logits);
    assert_eq!(steps, want_steps);
    finish(&mut inner);
    let c = fork(&a, 50, 110, 4);
    assert_eq!(turn(&mut inner, 3, &c).0, 0);
}

/// A cancel that lands after the pass ending at 96 fails the prefill in
/// that pass's draft history, before the point there is published: the
/// state at 64 stays, none sits above it, and a retry resumes there and
/// equals cold.
/// Mutation: `abort_paged_turn` clears the store → red (the retry is cold).
#[test]
fn a_cancel_in_prefill_keeps_the_states_behind_it_and_nothing_past_it() {
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let a = prompt(100, 1);
    let flag = Arc::new(AtomicBool::new(false));
    inner.set_turn_cancel_flag(Some(Arc::clone(&flag)));
    let trip = Arc::clone(&flag);
    let mut passes = 0;
    inner.decoder.after_pass = Some(Box::new(move |_| {
        passes += 1;
        if passes == 3 {
            trip.store(true, Ordering::Release);
        }
    }));
    inner.activate_paged_seq(1).unwrap();
    let prefix = inner.prime_prefix_state(&a, true, 32, &[], 0).unwrap();
    assert!(inner.paged_prefill(&a, &prefix, gpu()).is_err());
    inner.abort_paged_turn();
    assert_eq!(depths(&inner), [(64, true)]);
    inner.decoder.after_pass = None;
    flag.store(false, Ordering::Release);
    let (cached, logits) = turn(&mut inner, 1, &a);
    assert_eq!(cached, 64);
    let steps = greedy(&mut inner, &logits, 4);
    let (want_logits, want_steps) = cold(&a);
    assert_eq!(logits, want_logits);
    assert_eq!(steps, want_steps[..4]);
}

/// Other sequences take back the blocks of a released conversation: its
/// states go with them and its follow-up runs cold.
/// Mutation: `prune` keeps states whose chain lost a block → red (the
/// entries stay).
#[test]
fn states_follow_their_kv_blocks_out_of_the_pool() {
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let config = inner.decoder.config.clone();
    inner.decoder.paged = Some(super::super::paged::create_with_blocks(&config, 8, 8).unwrap());
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    assert_eq!(depths(&inner), [(64, true), (96, false)]);
    inner.release_scheduled_recurrent_for(1);
    inner
        .decoder
        .paged
        .as_mut()
        .unwrap()
        .release_request_for(1)
        .unwrap();
    turn(&mut inner, 2, &prompt(120, 5));
    finish(&mut inner);
    turn(&mut inner, 3, &prompt(120, 6));
    finish(&mut inner);
    for seq in [2, 3] {
        inner.release_scheduled_recurrent_for(seq);
        inner
            .decoder
            .paged
            .as_mut()
            .unwrap()
            .release_request_for(seq)
            .unwrap();
    }
    let follow = fork(&a, 98, 110, 2);
    inner.activate_paged_seq(4).unwrap();
    inner
        .prefix_states
        .prune(inner.decoder.paged.as_ref().unwrap(), GRID);
    assert!(
        inner
            .prefix_states
            .store
            .entries()
            .all(|entry| !a.starts_with(entry.tokens())),
        "every state of the evicted chain is gone"
    );
    assert!(inner.prefix_states.counters.dropped_kv > 0);
    assert_eq!(turn(&mut inner, 4, &follow).0, 0);
}

/// A second lane resumes from the first lane's state while the first still
/// decodes; both give the bits they give alone.
/// Mutation: the store lookup is keyed by sequence (lineage of this seq
/// only) → red (cached 0).
#[test]
fn two_lanes_share_a_published_prefix() {
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let a = prompt(100, 1);
    let (_, first) = turn(&mut inner, 1, &a);
    let early = greedy(&mut inner, &first, 3);
    let b = fork(&a, 98, 110, 2);
    let (cached, logits) = turn(&mut inner, 2, &b);
    assert_eq!(cached, 96);
    let (want_logits, want_steps) = cold(&b);
    assert_eq!(logits, want_logits);
    assert_eq!(greedy(&mut inner, &logits, 8), want_steps);
    inner.activate_paged_seq(1).unwrap();
    let late = greedy(&mut inner, early.last().unwrap(), 3);
    let mut alone = reuse_inner();
    let (_, alone_first) = turn(&mut alone, 1, &a);
    let alone_steps = greedy(&mut alone, &alone_first, 6);
    assert_eq!([early, late].concat(), alone_steps);
}

/// A scheduler step whose budget cuts a slice off the grid makes every
/// later pass differ from a cold prefill's: the turn publishes nothing past
/// the cut, and a follow-up resumes below it and equals cold.
/// Mutation: the publication ignores the canonical flag → red (a state at
/// 96 over a 48-row cut).
#[test]
fn a_cut_off_the_grid_publishes_nothing_past_it() {
    use crate::engine::hybrid_scheduler::ScheduledPrefixAdmission;
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let a = prompt(100, 1);
    let Ok(ScheduledPrefixAdmission::Ready(prefix)) =
        inner.prepare_scheduled_prefix(1, &a, &[], true, 0, 32)
    else {
        panic!("a text prompt is admitted at once");
    };
    assert_eq!(prefix.effective_cached_prefix_len(), 0);
    for (start, end) in [(0, 32), (32, 48), (48, 64), (64, 96), (96, 100)] {
        inner.activate_paged_seq(1).unwrap();
        inner
            .run_scheduled_prefill_slice(1, &a, &prefix, start, end, gpu(), start == 0)
            .unwrap();
    }
    finish(&mut inner);
    assert_eq!(depths(&inner), [(32, true)]);
    assert!(inner.prefix_states.counters.refused_off_grid >= 1);
    let b = fork(&a, 98, 110, 2);
    let (cached, logits) = turn(&mut inner, 2, &b);
    assert_eq!(cached, 32);
    let (want_logits, want_steps) = cold(&b);
    assert_eq!(logits, want_logits);
    assert_eq!(greedy(&mut inner, &logits, 8), want_steps);
}

/// Scheduler slices on the grid publish as the exclusive lane does.
#[test]
fn scheduled_slices_on_the_grid_publish_their_states() {
    use crate::engine::hybrid_scheduler::{ScheduledPrefixAdmission, prefill_slice_ends};
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let a = prompt(100, 1);
    let Ok(ScheduledPrefixAdmission::Ready(prefix)) =
        inner.prepare_scheduled_prefix(1, &a, &[], true, 0, 32)
    else {
        panic!("a text prompt is admitted at once");
    };
    let slice = inner.scheduler_prefill_slice_tokens();
    let mut start = 0;
    for end in prefill_slice_ends(0, a.len() as u32, slice, []) {
        inner.activate_paged_seq(1).unwrap();
        inner
            .run_scheduled_prefill_slice(1, &a, &prefix, start, end as usize, gpu(), start == 0)
            .unwrap();
        start = end as usize;
    }
    finish(&mut inner);
    assert_eq!(depths(&inner), [(64, true), (96, false)]);
}

/// The live continuation (the row's whole history leads the prompt) runs on
/// the live row and publishes nothing.
/// Mutation: a live start opens the turn's points → red (a state at 96).
#[test]
fn a_live_continuation_publishes_nothing() {
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let a = prompt(64, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    assert_eq!(depths(&inner), [(32, false)]);
    let b = [a.clone(), prompt(56, 7)].concat();
    assert_eq!(turn(&mut inner, 1, &b).0, 64);
    finish(&mut inner);
    assert_eq!(depths(&inner), [(32, false)]);
}

/// Rolling checkpoints: each retires the turn's previous one, the last
/// stays beside the replay point.
/// Mutation: no retirement → red (a checkpoint at 32 stays).
#[test]
fn rolling_checkpoints_retire_and_the_last_one_stays() {
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    turn(&mut inner, 1, &prompt(100, 1));
    assert_eq!(depths(&inner), [(64, true), (96, false)]);
    assert_eq!(inner.prefix_states.counters.published, 3);
    assert_eq!(inner.prefix_states.counters.retired, 1);
}

/// States live in the room the plan's dynamic budget leaves: a publication
/// makes room by dropping checkpoints first, never a state in use; the
/// scheduler's state ledger counts the store, and a state shortfall takes
/// stored states before anything else.
/// Mutation: `held` leaves the store out → red.
#[test]
fn the_state_store_stays_inside_its_room_and_gives_way_to_rows() {
    use crate::models::qwen3_5::dflash2_state::store::lineage_bytes;
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let state = |row: usize| memory::state_bytes(&inner.decoder.config, row as u64, 32, true);
    let room = state(96) + lineage_bytes(96, 3) + state(32) / 2;
    inner.prefix_states.room_override = Some(room);
    turn(&mut inner, 1, &prompt(100, 1));
    assert_eq!(depths(&inner), [(96, false)]);
    assert!(inner.prefix_states.bytes() <= room);
    assert_eq!(inner.prefix_states.counters.dropped_memory, 1);
    assert_eq!(inner.prefix_states.counters.retired, 1);
    let c = inner.decoder.config.clone();
    assert_eq!(
        inner.scheduled_state_capacity().unwrap().held,
        memory::row_state_bytes(&c) + inner.mtp.held_bytes(&c) + inner.prefix_states.bytes()
    );
    assert!(
        !inner.reclaim_for_scheduled_state(1) || inner.prefix_states.bytes() > 0,
        "the replay point is in use until its turn ends"
    );
    finish(&mut inner);
    assert!(inner.reclaim_for_scheduled_state(1));
    assert_eq!(inner.prefix_states.bytes(), 0);
}

/// Once a conversation's states are gone but its blocks stay cached, its next
/// cold run of the same rows publishes again over those blocks (the same
/// passes over the same tokens), and a follow-up resumes there and equals
/// cold.
/// Mutation: a cached block below the point refuses the publication → red.
#[test]
fn a_cold_run_publishes_over_the_blocks_an_earlier_run_of_its_passes_left() {
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    assert_eq!(depths(&inner), [(64, true), (96, false)]);
    let chain = inner
        .prefix_states
        .store
        .entries()
        .map(|entry| entry.chain().to_vec())
        .max_by_key(Vec::len)
        .unwrap();
    inner.prefix_states.store.clear();
    let (cached, _) = turn(&mut inner, 2, &a);
    assert_eq!(cached, 0);
    finish(&mut inner);
    assert_eq!(depths(&inner), [(64, true), (96, false)]);
    let adapter = inner.decoder.paged.as_ref().unwrap();
    assert!(chain.iter().all(|&hash| adapter.prefix_cached(hash)));
    let b = fork(&a, 98, 110, 2);
    let (cached, logits) = turn(&mut inner, 3, &b);
    assert_eq!(cached, 96);
    let (want_logits, want_steps) = cold(&b);
    assert_eq!(logits, want_logits);
    assert_eq!(greedy(&mut inner, &logits, 8), want_steps);
}

/// The fixture on a 64-row grid (two KV blocks per pass), checkpoints every
/// 64 rows.
fn wide_grid_inner() -> Inner {
    let mut inner = tiny_inner();
    inner.decoder.prefill_window_pin = Some(2 * GRID);
    inner.decoder.refresh_prefill_window();
    inner.prefix_states.interval = 2 * GRID;
    inner
}

/// Two conversations that share a whole first block but leave each other
/// inside its pass: the second one's blocks never take the first one's key
/// (their passes ran over other rows), so it publishes its own state, and its
/// follow-up resumes there and equals cold.
/// Mutation: blocks keyed by their own tokens only (no pass key) → red (the
/// second conversation's state sits on the first one's block 0).
#[test]
fn a_conversation_that_leaves_another_inside_a_pass_keeps_its_own_blocks() {
    let _host = memory::scoped_fixture_host();
    let mut inner = wide_grid_inner();
    let a = prompt(120, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    assert_eq!(depths(&inner), [(64, false)]);
    let b = fork(&a, 40, 110, 2);
    assert_eq!(turn(&mut inner, 2, &b).0, 0);
    finish(&mut inner);
    assert_eq!(depths(&inner), [(64, false), (64, false)]);
    let chains: Vec<Vec<u64>> = inner
        .prefix_states
        .store
        .entries()
        .map(|entry| entry.chain().to_vec())
        .collect();
    assert!(
        chains[0].iter().all(|hash| !chains[1].contains(hash)),
        "the two passes over rows 0..64 differ, so no block is shared"
    );
    let c = fork(&b, 100, 112, 3);
    let (cached, logits) = turn(&mut inner, 3, &c);
    assert_eq!(cached, 64);
    let (want_logits, want_steps) = cold_on(wide_grid_inner(), &c);
    assert_eq!(logits, want_logits);
    assert_eq!(greedy(&mut inner, &logits, 8), want_steps);
}

/// A publication the host's live memory refuses drops no stored state: the
/// room it would have made is only chosen, never taken.
/// Mutation: the victims are dropped before the live check → red.
#[test]
fn a_refused_publication_drops_no_stored_state() {
    use crate::models::qwen3_5::dflash2_state::store::lineage_bytes;
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let state = |row: usize| memory::state_bytes(&inner.decoder.config, row as u64, 32, true);
    inner.prefix_states.room_override = Some(state(96) + lineage_bytes(96, 3) + state(32) / 2);
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    assert_eq!(depths(&inner), [(96, false)]);
    let dropped = inner.prefix_states.counters.dropped_memory;
    {
        let _full = memory::scoped_test_host(memory::HostMemory {
            free: 0,
            file: 0,
            ..memory::fixture_host()
        });
        turn(&mut inner, 2, &prompt(100, 5));
        finish(&mut inner);
    }
    assert_eq!(inner.prefix_states.counters.refused_live, 3);
    assert_eq!(inner.prefix_states.counters.dropped_memory, dropped);
    assert_eq!(depths(&inner), [(96, false)]);
    assert_eq!(turn(&mut inner, 3, &fork(&a, 98, 110, 2)).0, 96);
}

/// A layer-major group ends at the deepest checkpoint and the replay point
/// inside it: the states it publishes are the window-at-a-time ones bit for
/// bit, and it never holds more than the cell it publishes.
/// Mutation: the group runs unsplit → red (no state).
#[test]
fn a_group_publishes_at_its_cuts_the_states_of_window_passes() {
    let _host = memory::scoped_fixture_host();
    let cells = |pin: bool| {
        let mut inner = reuse_inner();
        inner.decoder.group_pin = Some(pin);
        turn(&mut inner, 1, &prompt(100, 1));
        let log = inner.decoder.submit_log.borrow().clone();
        let mut cells: Vec<_> = inner
            .prefix_states
            .store
            .entries()
            .map(|entry| {
                (
                    entry.depth(),
                    entry.state().arrays().map(f32s).collect::<Vec<_>>(),
                )
            })
            .collect();
        cells.sort_by_key(|(depth, _)| *depth);
        (cells, log)
    };
    let (grouped, log) = cells(true);
    let (windowed, _) = cells(false);
    assert_eq!(
        log.iter().filter(|e| *e == "qwen4::group::end").count(),
        1,
        "the rows below the checkpoint ran as one group"
    );
    assert_eq!(
        grouped.iter().map(|(d, _)| *d).collect::<Vec<_>>(),
        [64, 96]
    );
    assert_eq!(grouped, windowed);
}

/// An image prompt neither resumes nor plans a state.
/// Mutation: the text-only check removed → red (cached 64).
#[test]
fn an_image_prompt_neither_resumes_nor_publishes() {
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    let mut b = fork(&a, 80, 110, 3);
    b[100] = crate::vision::qwen::prompt::IMAGE_TOKEN_ID as u32;
    inner.activate_paged_seq(2).unwrap();
    let prefix = inner.prime_prefix_state(&b, true, 32, &[], 0).unwrap();
    assert_eq!(prefix.effective_cached_prefix_len(), 0);
    assert!(!inner.prefix_states.turns.contains_key(&2));
    inner.abort_paged_turn();
}

/// A command reset drops every state and every published block.
/// Mutation: the reset keeps the store → red.
#[test]
fn a_reset_drops_every_state_and_published_block() {
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    finish(&mut inner);
    let chain = inner
        .prefix_states
        .store
        .entries()
        .next()
        .unwrap()
        .chain()
        .to_vec();
    inner
        .reset_caches(crate::engine::backend::ResetScope::Command)
        .unwrap();
    assert!(depths(&inner).is_empty());
    let adapter = inner.decoder.paged.as_ref().unwrap();
    assert!(chain.iter().all(|&hash| !adapter.prefix_cached(hash)));
    assert_eq!(turn(&mut inner, 1, &fork(&a, 98, 110, 2)).0, 0);
}

/// Preemption hands the row's blocks back without publishing: no block
/// above the last state (decode rows) enters the prefix cache.
/// Mutation: the default preemption (publish every full block) → red.
#[test]
fn a_preempted_turn_publishes_no_block_past_its_states() {
    use crate::engine::scheduler::PreemptionMode;
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let a = prompt(70, 1);
    let (_, logits) = turn(&mut inner, 1, &a);
    greedy(&mut inner, &logits, 26);
    let history = inner.decoder.history.clone();
    assert_eq!(history.len(), 96);
    inner
        .preempt_scheduled_cache(1, 0, PreemptionMode::Recompute)
        .unwrap();
    inner.release_scheduled_recurrent_for(1);
    // Keys of the prompt's passes below 64, and of the decode rows 64..96 as
    // if a pass had run over them.
    let chain = crate::models::qwen3_5::gdn_checkpoint_store::compute_paged_prefix_block_hashes(
        &history,
        96,
        32,
        &prefix_state::pass_keys(&history, &[32, 64, 96, 97], GRID),
        0,
    )
    .unwrap();
    let adapter = inner.decoder.paged.as_ref().unwrap();
    assert!(adapter.prefix_cached(chain[1]), "the replay point's blocks");
    assert!(!adapter.prefix_cached(chain[2]), "a block of decode rows");
}

/// A restored turn's sequence replaces the draft owner an earlier turn left
/// at the same row with the state's draft history: speculation begins there
/// from the history a cold prefill of the lineage leaves.
/// Mutation: the restore keeps the old owner → red.
#[test]
fn a_restored_turn_replaces_a_stale_draft_owner() {
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    inner.activate_paged_seq(1).unwrap();
    assert!(inner.begin_mtp(1, 0).unwrap());
    turn(&mut inner, 1, &prompt(64, 9));
    finish(&mut inner);
    // The scheduler ends an MTP turn this way: an owner at the row's
    // frontier stays for the next turn.
    inner.release_scheduled_speculation(1);
    assert_eq!(inner.mtp.owner_frontier(1), Some(64));
    let a = prompt(100, 1);
    turn(&mut inner, 2, &a);
    finish(&mut inner);
    // The scheduler begins speculation at the first slice, before it runs.
    inner.activate_paged_seq(1).unwrap();
    let prefix = inner
        .prime_prefix_state(&fork(&a, 80, 110, 3), true, 32, &[], 0)
        .unwrap();
    assert_eq!(prefix.effective_cached_prefix_len(), 64);
    assert!(inner.begin_scheduled_speculation(1, 64).unwrap());
    let restored = inner.mtp.draft_bits(1).unwrap();
    inner.abort_paged_turn();
    let mut cold = reuse_inner();
    cold.activate_paged_seq(1).unwrap();
    let prefix = cold.prime_prefix_state(&a[..64], true, 32, &[], 0).unwrap();
    assert!(cold.begin_scheduled_speculation(1, 0).unwrap());
    cold.paged_prefill(&a[..64], &prefix, gpu()).unwrap();
    assert_eq!(restored, cold.mtp.draft_bits(1).unwrap());
}

/// The turn result reports the reused rows as `cachedTokens`, from the
/// prefix the prime chose.
/// Mutation: the prime reports 0 cached rows after a restore → red.
#[test]
fn the_turn_result_reports_the_reused_rows() {
    use crate::engine::backend::{TurnOutput, WholeTurnArgs};
    use crate::engine::plan::{DecoderPlan, MediaCapabilities, MediaInputs, TurnPlan};
    let _host = memory::scoped_fixture_host();
    let run = |inner: &mut Inner, tokens: &[u32]| {
        let config = crate::engine::types::ChatConfig {
            max_new_tokens: Some(4),
            temperature: Some(0.0),
            reuse_cache: Some(true),
            ..Default::default()
        };
        let params = ChatBackend::resolve_params(inner, &config);
        let thinking = ChatBackend::thinking_setup(inner, &config);
        let tokenizer = Arc::clone(&inner.tokenizer);
        let mut args = WholeTurnArgs {
            tokens,
            tokenizer: &tokenizer,
            eos_id: u32::MAX,
            config: &config,
            params: &params,
            thinking,
            plan: TurnPlan {
                is_delta: false,
                input_media: MediaCapabilities::NONE,
                context_media: MediaCapabilities::NONE,
                use_paged_attention: true,
                decoder: DecoderPlan::Autoregressive,
            },
            sink: None,
            cancelled: None,
            media: MediaInputs {
                images: &[],
                audio: &[],
            },
        };
        let TurnOutput::Complete(result) = ChatBackend::run_paged_turn(inner, &mut args).unwrap()
        else {
            panic!("a sync turn completes");
        };
        result
    };
    let mut inner = reuse_inner();
    let a = prompt(100, 1);
    assert_eq!(run(&mut inner, &a).cached_tokens, 0);
    let b = fork(&a, 98, 110, 2);
    let warm = run(&mut inner, &b);
    assert_eq!(warm.cached_tokens, 96);
    let mut fresh = reuse_inner();
    let cold = run(&mut fresh, &b);
    assert_eq!(cold.cached_tokens, 0);
    assert_eq!(warm.num_tokens, cold.num_tokens);
    assert_eq!(warm.text, cold.text);
}

/// Every way a turn ends gives back the states it used (the one it resumed
/// from, the replay point it published): finalize, abort, the scheduler's
/// release, preemption, and the next prime of the same sequence. A use left
/// behind would keep a state from ever giving way to memory.
/// Mutation: the scheduler's release keeps the turn → red.
#[test]
fn every_turn_end_gives_back_the_states_it_used() {
    use crate::engine::scheduler::PreemptionMode;
    let _host = memory::scoped_fixture_host();
    let a = prompt(100, 1);
    let ends: [(&str, &dyn Fn(&mut Inner)); 5] = [
        ("finalize", &|inner| finish(inner)),
        ("abort", &|inner| inner.abort_paged_turn()),
        ("release", &|inner| inner.release_scheduled_recurrent_for(1)),
        ("preempt", &|inner| {
            inner
                .preempt_scheduled_cache(1, 0, PreemptionMode::Recompute)
                .unwrap()
        }),
        ("prime", &|inner| {
            inner
                .prime_prefix_state(&prompt(20, 4), true, 32, &[], 0)
                .unwrap();
        }),
    ];
    for (name, end) in ends {
        let mut inner = reuse_inner();
        turn(&mut inner, 2, &a);
        finish(&mut inner);
        assert_eq!(turn(&mut inner, 1, &fork(&a, 70, 110, 3)).0, 64, "{name}");
        let in_use = |inner: &Inner| {
            inner
                .prefix_states
                .store
                .entries()
                .filter(|entry| inner.prefix_states.store.in_use(entry.mark()))
                .count()
        };
        assert!(
            in_use(&inner) > 0,
            "{name}: the running turn uses its states"
        );
        end(&mut inner);
        assert_eq!(in_use(&inner), 0, "{name}");
        assert!(
            inner
                .prefix_states
                .turns
                .get(&1)
                .is_none_or(|turn| turn.uses.is_empty() && turn.prompt_len == 20),
            "{name}: only the next prompt's turn may be open"
        );
    }
}

/// `inner` on a 64-row prefill window (checkpoints every 64 rows) whose
/// prompts open assistant messages with [`OPEN`].
fn cut_over(mut inner: Inner) -> Inner {
    inner.decoder.prefill_window_pin = Some(2 * GRID);
    inner.decoder.refresh_prefill_window();
    inner.prefix_states.interval = 2 * GRID;
    inner.prefix_states.open = OPEN.to_vec();
    inner
}

fn cut_inner() -> Inner {
    cut_over(tiny_inner())
}

/// The parts in order.
fn join(parts: &[&[u32]]) -> Vec<u32> {
    parts.concat()
}

/// A: an assistant opening at row 45, a user opening (`62, 5`) at row 67.
/// B: A up to the opening, another reply and B's own generation prompt at
/// row 87.
fn conversation() -> (Vec<u32>, Vec<u32>) {
    let a = join(&[
        &prompt(45, 1),
        &OPEN,
        &prompt(20, 3),
        &[62, 5],
        &prompt(3, 4),
    ]);
    let b = join(&[&a[..47], &prompt(40, 2), &OPEN, &prompt(3, 5)]);
    (a, b)
}

/// The rows of every pass the decoder ran since the last call.
fn passes(inner: &mut Inner) -> Vec<usize> {
    std::mem::take(&mut inner.decoder.pass_samples)
        .into_iter()
        .map(|(rows, _)| rows)
        .collect()
}

/// A follow-up resumes at the block before its conversation's last
/// generation prompt (row 32 for an opening at row 45, where the 64-row
/// grid would give 0), and gives a cold model's logits and greedy steps bit
/// for bit, on a bf16 pool, an int8 pool (head 128), and with the passes a
/// refused gate halves.
/// Mutation: `boundaries` ignores the opening's role token → red (the user
/// opening at row 67 moves the replay point to 64, past B's lineage).
#[test]
fn a_follow_up_resumes_at_the_block_before_the_generation_prompt() {
    use crate::transformer::KvFormat;
    let _host = memory::scoped_fixture_host();
    let dir = Temp::new();
    write_wide_head_fixture(&dir.0);
    let halved = || {
        let mut inner = cut_inner();
        let config = Arc::clone(&inner.decoder.config);
        inner.decoder.host_prefill_scratch = Some(memory::prefill_scratch(&config, 32, 64));
        inner
    };
    let cases: [(&str, &dyn Fn() -> Inner); 3] = [
        ("bf16", &cut_inner),
        ("int8", &|| {
            cut_over(wide_head_inner(&dir.0, KvFormat::Int8))
        }),
        ("halved", &halved),
    ];
    let (a, b) = conversation();
    for (name, make) in cases {
        let mut inner = make();
        assert_eq!(turn(&mut inner, 1, &a).0, 0, "{name}");
        finish(&mut inner);
        assert_eq!(depths(&inner), [(32, false)], "{name}");
        let (cached, logits) = turn(&mut inner, 1, &b);
        assert_eq!(cached, 32, "{name}");
        if name == "halved" {
            assert!(
                inner
                    .decoder
                    .pass_samples
                    .iter()
                    .any(|(rows, _)| *rows < 2 * GRID),
                "{name}: the window halves past the scratch"
            );
        }
        let steps = greedy(&mut inner, &logits, 8);
        let (want_logits, want_steps) = cold_on(make(), &b);
        assert_eq!(logits, want_logits, "{name}: first logits");
        assert_eq!(steps, want_steps, "{name}: greedy steps");
    }
}

/// A cold prefill of the follow-up runs the passes the first turn ran below
/// the point the follow-up resumes at, then the follow-up's own.
/// Mutation: a cold turn walks the plain grid (no cut at its openings) →
/// red.
#[test]
fn a_cold_run_cuts_where_an_earlier_turn_resumed() {
    let _host = memory::scoped_fixture_host();
    let (a, b) = conversation();
    let mut inner = cut_inner();
    inner.decoder.pass_samples.clear();
    turn(&mut inner, 1, &a);
    let first = passes(&mut inner);
    assert_eq!(first, [32, 40]);
    finish(&mut inner);
    let (cached, _) = turn(&mut inner, 1, &b);
    assert_eq!(cached, 32);
    let warm = passes(&mut inner);
    assert_eq!(warm, [32, 28]);
    let mut cold = cut_inner();
    cold.decoder.pass_samples.clear();
    assert_eq!(turn(&mut cold, 1, &b).0, 0);
    assert_eq!(passes(&mut cold), [&first[..1], &warm[..]].concat());
}

/// A prompt that shares a state's lineage but leaves the conversation inside
/// the block of the opening the state was cut for does not cut there, so it
/// takes the deepest earlier state of its own passes, and equals cold.
/// Mutation: the lookup ignores the pass keys → red (the state at 64 is
/// tried, its blocks miss, the turn runs cold).
#[test]
fn a_prompt_that_leaves_inside_the_open_block_does_not_take_that_state() {
    let _host = memory::scoped_fixture_host();
    let mut inner = cut_inner();
    let first = join(&[&prompt(45, 1), &OPEN, &prompt(3, 3)]);
    turn(&mut inner, 1, &first);
    finish(&mut inner);
    let a = join(&[&first[..47], &prompt(18, 3), &OPEN, &prompt(3, 4)]);
    assert_eq!(turn(&mut inner, 2, &a).0, 32);
    finish(&mut inner);
    assert_eq!(depths(&inner), [(32, false), (64, false)]);
    let q = join(&[&a[..65], &prompt(40, 6), &OPEN, &prompt(3, 7)]);
    let (cached, logits) = turn(&mut inner, 3, &q);
    assert_eq!(cached, 32);
    let steps = greedy(&mut inner, &logits, 8);
    let (want_logits, want_steps) = cold_on(cut_inner(), &q);
    assert_eq!(logits, want_logits);
    assert_eq!(steps, want_steps);
}

/// The scheduler slices a text prompt at its pass ends, not on a grid
/// walked from the cached prefix: slices run through the scheduled lane
/// publish the replay point a cold prefill publishes.
/// Mutation: `scheduler_prefill_ends` answers `None` → red (the walked grid
/// cuts at 64, off the passes, and nothing is published).
#[test]
fn scheduled_slices_follow_the_prompt_ends() {
    use crate::engine::hybrid_scheduler::{ScheduledPrefixAdmission, prefill_slice_ends};
    let _host = memory::scoped_fixture_host();
    let mut inner = cut_inner();
    let a = join(&[&prompt(45, 1), &OPEN, &prompt(58, 3), &OPEN, &prompt(3, 4)]);
    let Ok(ScheduledPrefixAdmission::Ready(prefix)) =
        inner.prepare_scheduled_prefix(1, &a, &[], true, 0, 32)
    else {
        panic!("a text prompt is admitted at once");
    };
    let len = a.len() as u32;
    let ends = inner.scheduler_prefill_ends(1, len, 0);
    assert_eq!(ends.as_deref(), Some(&[32, 96, 110][..]));
    assert_eq!(
        ends.as_ref()
            .unwrap()
            .iter()
            .map(|&e| e as usize)
            .collect::<Vec<_>>(),
        inner.prefix_states.turns[&1].ends
    );
    let slices = ends
        .unwrap_or_else(|| prefill_slice_ends(0, len, inner.scheduler_prefill_slice_tokens(), []));
    let mut start = 0;
    for end in slices {
        inner.activate_paged_seq(1).unwrap();
        inner
            .run_scheduled_prefill_slice(1, &a, &prefix, start, end as usize, gpu(), start == 0)
            .unwrap();
        start = end as usize;
    }
    finish(&mut inner);
    assert_eq!(depths(&inner), [(96, false)]);
}

/// A layer-major group holds only whole windows of the turn's passes: with
/// the window halved past 64 rows, the short passes run one at a time, and
/// the states equal the window-at-a-time ones bit for bit.
/// Mutation: a run takes passes of any length → red (a run of short
/// passes).
#[test]
fn a_group_holds_only_whole_windows() {
    let _host = memory::scoped_fixture_host();
    let run = |pin: bool| {
        let mut inner = reuse_inner();
        let config = Arc::clone(&inner.decoder.config);
        inner.decoder.host_prefill_scratch = Some(memory::prefill_scratch(&config, 32, 64));
        inner.decoder.group_pin = Some(pin);
        let (_, logits) = turn(&mut inner, 1, &prompt(100, 1));
        let turn = &inner.prefix_states.turns[&1];
        assert!(
            turn.ends.windows(2).any(|pair| pair[1] - pair[0] < GRID),
            "the window halves past 64 rows: {:?}",
            turn.ends
        );
        for run in turn.runs(0, 100, GRID) {
            assert!(
                run.len() == 1 || run.iter().all(|&(start, end)| end - start == GRID),
                "a run of several passes holds whole windows only: {run:?}"
            );
        }
        let log = inner.decoder.submit_log.borrow().clone();
        let mut cells: Vec<_> = inner
            .prefix_states
            .store
            .entries()
            .map(|entry| {
                (
                    entry.depth(),
                    entry.state().arrays().map(f32s).collect::<Vec<_>>(),
                )
            })
            .collect();
        cells.sort_by_key(|(depth, _)| *depth);
        let passes = inner.prefix_states.counters;
        (cells, log, logits, passes)
    };
    let (grouped, log, grouped_logits, counters) = run(true);
    let (windowed, _, windowed_logits, _) = run(false);
    assert_eq!(
        log.iter().filter(|e| *e == "qwen4::group::end").count(),
        1,
        "only the two whole windows below the checkpoint form a group"
    );
    assert_eq!(counters.refused_off_grid, 0);
    assert_eq!(
        grouped.iter().map(|(d, _)| *d).collect::<Vec<_>>(),
        [64, 96]
    );
    assert_eq!(grouped, windowed);
    assert_eq!(grouped_logits, windowed_logits);
}

/// `count` greedy speculative cycles from `logits`: propose two drafts,
/// verify them with the anchor, keep the matching prefix. Each cycle's
/// draft ids, kept rows and verify logits.
pub(super) fn cycles(
    inner: &mut Inner,
    logits: &[f32],
    count: usize,
) -> Vec<(Vec<i32>, usize, Vec<f32>)> {
    use crate::engine::hybrid_scheduler::{ScheduledVerifyCommit, ScheduledVerifyRow};
    use rand::SeedableRng;
    let params = inner.resolve_params(&crate::engine::types::ChatConfig {
        temperature: Some(0.0),
        enable_mtp: Some(true),
        ..Default::default()
    });
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap()
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let mut last = logits.to_vec();
    let mut out = Vec::new();
    for _ in 0..count {
        let anchor = argmax(&last);
        let proposal = inner
            .propose_mtp(1, anchor, 2, &params, &mut rng, false)
            .unwrap();
        let mut tokens = vec![anchor];
        tokens.extend(proposal.draft_ids.iter().map(|&t| t as u32));
        let verified = f32s(
            &inner
                .verify_mtp(&[ScheduledVerifyRow {
                    seq_id: 1,
                    first_position: inner.decoder.history.len() as u32,
                    tokens: tokens.clone(),
                    speculative: true,
                }])
                .unwrap(),
        );
        let rows: Vec<&[f32]> = verified.chunks(verified.len() / tokens.len()).collect();
        let mut keep = 1;
        while keep < tokens.len() && argmax(rows[keep - 1]) == tokens[keep] {
            keep += 1;
        }
        inner
            .commit_mtp(&[ScheduledVerifyCommit { seq_id: 1, keep }])
            .unwrap()
            .remove(0)
            .unwrap();
        last = rows[keep - 1].to_vec();
        out.push((proposal.draft_ids, keep, verified));
    }
    out
}

/// Primes `tokens` on sequence 1, begins speculation at the prefix it
/// resumes at as the scheduler does, and prefills: the cached prefix, the
/// begin answer and the logits.
pub(super) fn speculative_turn(inner: &mut Inner, tokens: &[u32]) -> (usize, bool, Vec<f32>) {
    inner.activate_paged_seq(1).unwrap();
    let prefix = inner.prime_prefix_state(tokens, true, 32, &[], 0).unwrap();
    let cached = prefix.effective_cached_prefix_len();
    let begun = inner.begin_scheduled_speculation(1, cached as u32).unwrap();
    let logits = inner
        .paged_prefill(&tokens[cached..], &prefix, gpu())
        .unwrap();
    (cached, begun, f32s(&logits))
}

/// A turn restored from a state drafts as a cold turn of its prompt does:
/// speculation begins at the restored row, the draft history after the
/// prefill is the cold one's bit for bit, and six greedy cycles give the
/// same drafts, kept rows and verify logits.
/// Mutation: `mtp.restore` drops the waiting rows → red.
#[test]
fn a_restored_turn_drafts_as_a_cold_turn_does() {
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let a = prompt(100, 1);
    let (_, begun, _) = speculative_turn(&mut inner, &a);
    assert!(begun);
    finish(&mut inner);
    inner.release_scheduled_speculation(1);
    let b = fork(&a, 98, 104, 2);
    let (cached, begun, logits) = speculative_turn(&mut inner, &b);
    assert_eq!((cached, begun), (96, true));
    let warm = inner.mtp.draft_bits(1).unwrap();
    let mut cold = reuse_inner();
    let (cached, begun, cold_logits) = speculative_turn(&mut cold, &b);
    assert_eq!((cached, begun), (0, true));
    assert_eq!(logits, cold_logits);
    assert_eq!(warm, cold.mtp.draft_bits(1).unwrap());
    assert_eq!(warm.rows, 96);
    assert_eq!(
        cycles(&mut inner, &logits, 6),
        cycles(&mut cold, &cold_logits, 6)
    );
}

/// On a load with the MTP head an autoregressive turn still publishes its
/// states with the draft history and drops its owner after its prefill; a
/// follow-up that speculates resumes there and drafts as a cold one.
/// Mutation: the prime starts no owner for a turn that has not begun
/// speculation → red (no state is published).
#[test]
fn an_autoregressive_turn_on_a_draft_load_publishes_draft_states() {
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let a = prompt(100, 1);
    turn(&mut inner, 1, &a);
    assert_eq!(inner.mtp.owner_frontier(1), None);
    finish(&mut inner);
    assert_eq!(depths(&inner), [(64, true), (96, false)]);
    assert!(
        inner
            .prefix_states
            .store
            .entries()
            .all(|entry| entry.state().draft.is_some())
    );
    let b = fork(&a, 98, 104, 2);
    let (cached, begun, logits) = speculative_turn(&mut inner, &b);
    assert_eq!((cached, begun), (96, true));
    let mut cold = reuse_inner();
    let (_, _, cold_logits) = speculative_turn(&mut cold, &b);
    assert_eq!(logits, cold_logits);
    assert_eq!(
        inner.mtp.draft_bits(1).unwrap(),
        cold.mtp.draft_bits(1).unwrap()
    );
}

/// A point whose owner no longer holds the cold prefill's draft passes is
/// refused, never published without its draft history.
/// Mutation: the publication ignores a refused capture → red (a state at 96).
#[test]
fn a_state_without_its_draft_is_not_published() {
    use crate::engine::hybrid_scheduler::ScheduledPrefixAdmission;
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let a = prompt(100, 1);
    let Ok(ScheduledPrefixAdmission::Ready(prefix)) =
        inner.prepare_scheduled_prefix(1, &a, &[], true, 0, 32)
    else {
        panic!("a text prompt is admitted at once");
    };
    for (start, end) in [(0, 32), (32, 64), (64, 96), (96, 100)] {
        inner.activate_paged_seq(1).unwrap();
        inner
            .run_scheduled_prefill_slice(1, &a, &prefix, start, end, gpu(), start == 0)
            .unwrap();
        if end == 64 {
            inner.mtp.misalign(1);
        }
    }
    finish(&mut inner);
    assert_eq!(inner.prefix_states.counters.refused_draft, 1);
    assert_eq!(depths(&inner), [(64, true)]);
}

/// The states of one conversation share the draft passes they have in
/// common: the shallow state's segments are the deep state's first ones.
/// Mutation: `capture` copies the draft cache for every state → red.
#[test]
fn states_of_one_lineage_share_their_draft_segments() {
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    turn(&mut inner, 1, &prompt(100, 1));
    finish(&mut inner);
    let segments = |depth: usize| {
        let entry = inner
            .prefix_states
            .store
            .entries()
            .find(|entry| entry.depth() == depth)
            .unwrap();
        entry
            .state()
            .draft
            .as_ref()
            .unwrap()
            .segments
            .iter()
            .map(|segment| segment.keys.as_raw_ptr())
            .collect::<Vec<_>>()
    };
    let (shallow, deep) = (segments(64), segments(96));
    assert_eq!((shallow.len(), deep.len()), (1, 2));
    assert_eq!(shallow[..], deep[..1]);
}

/// On a load with the MTP head an autoregressive text prefill is charged at
/// admission for the draft history it keeps (draft K/V and indexer keys of
/// its rows, at most a window of hidden rows waiting); without the head it
/// is charged its row only. An owner's held bytes count the segments cut
/// out of its cache.
/// Mutation: only a speculative turn is charged its draft → red.
#[test]
fn an_autoregressive_prefill_on_a_head_load_is_charged_its_draft() {
    let _host = memory::scoped_fixture_host();
    let mut inner = reuse_inner();
    let c = inner.decoder.config.clone();
    let row = memory::row_state_bytes(&c);
    let window = inner.decoder.prefill_slice_size() as u64;
    let waiting = 100.min(window + 1);
    let ar = memory::draft_owner_bytes(&c, 100 - waiting, waiting, false)
        .max(memory::draft_owner_bytes(&c, 100, 0, false));
    assert_eq!(
        inner.scheduled_state_candidate_bytes(1, 100, false),
        row + ar
    );
    let a = prompt(100, 1);
    let (_, begun, _) = speculative_turn(&mut inner, &a);
    assert!(begun);
    let (_, waiting) = inner.mtp.draft_rows(1).unwrap();
    let frontier = inner.mtp.owner_frontier(1).unwrap() as u64;
    let hidden = waiting as u64 + 1;
    // The replay state at row 96 shares the segments cut out of the owner.
    let replay = inner
        .prefix_states
        .store
        .entries()
        .find(|entry| entry.depth() == 96)
        .unwrap();
    let segments: u64 = replay
        .state()
        .draft
        .as_ref()
        .unwrap()
        .segments
        .iter()
        .map(|segment| {
            (segment.keys.nbytes() + segment.values.nbytes() + segment.index.nbytes()) as u64
        })
        .sum();
    assert!(segments > 0);
    assert_eq!(
        inner.mtp.owner_held_bytes(&c, 1),
        memory::draft_owner_bytes(&c, frontier - hidden, hidden, true) + segments
    );
    let mut without = reuse_inner();
    without.has_mtp = false;
    assert_eq!(without.scheduled_state_candidate_bytes(1, 100, false), row);
}
