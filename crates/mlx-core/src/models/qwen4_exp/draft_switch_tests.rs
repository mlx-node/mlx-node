//! Which scheduled turns run the draft's cycle: only a turn that starts
//! alone on a whole store with a draft attached. A turn that starts beside
//! another, or on a partial store, takes the plain path; a drafted turn that
//! gets a peer verifies its anchor alone from its next cycle. Every path
//! gives the target's own greedy tokens.
use super::draft_cycle::{drafted_turn, moe_draft, nothing, tapped};
use super::lanes::direct_inner;
use super::prefix_reuse::{prompt, reuse_over};
use super::promote::{bank_bytes, host_gate_model, host_with, own};
use super::*;
use crate::engine::backend::ChatBackend;
use crate::engine::cmd::ChatCmd;
use crate::engine::hybrid_scheduler::{HybridSchedulerBackend, HybridSchedulerState};
use crate::engine::model_command::ModelCommand;
use crate::engine::types::{ChatConfig, ChatResult};
use crate::model_thread::LoopControl;
use crate::models::qwen3_5::dflash2::{DFlash2ContextCache, DFlash2Model};
use crate::models::qwen4_exp::live::Residency;
use std::cell::RefCell;
use std::rc::Rc;

pub(super) const ROWS: usize = 4;

/// A tokenizer whose chat template writes each message's content and a
/// space, and whose words `t0`..`t{vocab-1}` are the ids.
fn word_tokenizer(vocab: usize) -> Arc<Qwen3Tokenizer> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<usize, Arc<Qwen3Tokenizer>>>,
    > = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(tok) = cache.lock().unwrap().get(&vocab) {
        return Arc::clone(tok);
    }
    let dir = Temp::new();
    std::fs::write(
        dir.0.join("tokenizer_config.json"),
        r#"{"chat_template":"{% for m in messages %}{{ m.content }} {% endfor %}"}"#,
    )
    .unwrap();
    let words: serde_json::Map<String, serde_json::Value> = (0..vocab)
        .map(|id| (format!("t{id}"), serde_json::json!(id)))
        .collect();
    let tokenizer = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": words, "unk_token": "t2"}
    });
    let path = dir.0.join("tokenizer.json");
    std::fs::write(&path, tokenizer.to_string()).unwrap();
    let tok = Arc::new(Qwen3Tokenizer::load_from_file_sync(path.to_str().unwrap()).unwrap());
    let mut map = cache.lock().unwrap();
    Arc::clone(map.entry(vocab).or_insert(tok))
}

fn words(ids: &[u32]) -> String {
    ids.iter()
        .map(|id| format!("t{id}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `inner` serving chat with [`word_tokenizer`].
pub(super) fn chat_model(mut inner: Inner) -> Inner {
    inner.tokenizer = word_tokenizer(inner.decoder.config.vocab_size);
    inner
}

pub(super) type Reply = tokio::sync::oneshot::Receiver<Result<ChatResult>>;

/// A greedy start of `prompt` for `owner`: `max_new` tokens, the draft's
/// cycle asked for with `speculate` at 4 rows.
pub(super) fn start(
    owner: &str,
    prompt: &[u32],
    max_new: i32,
    speculate: bool,
) -> (ModelCommand, Reply) {
    let config = ChatConfig {
        max_new_tokens: Some(max_new),
        temperature: Some(0.0),
        enable_mtp: Some(speculate),
        mtp_depth: Some(ROWS as i32 - 1),
        mtp_adaptive_depth: Some(false),
        ..Default::default()
    };
    start_with(owner, prompt, config)
}

/// A start of `prompt` for `owner` under `config`.
pub(super) fn start_with(owner: &str, prompt: &[u32], config: ChatConfig) -> (ModelCommand, Reply) {
    let message: crate::tokenizer::ChatMessage =
        serde_json::from_value(serde_json::json!({"role": "user", "content": words(prompt)}))
            .unwrap();
    let config = ChatConfig {
        cache_owner_id: Some(owner.to_string()),
        ..config
    };
    let (reply, rx) = tokio::sync::oneshot::channel();
    let command = ChatCmd::SessionStart {
        messages: vec![message],
        config,
        reply,
        cancelled: Arc::new(AtomicBool::new(false)),
    };
    (ModelCommand::from_chat(command), rx)
}

/// What a turn gave: its text, its finish and the owner's tokens.
#[derive(Debug, PartialEq)]
pub(super) struct Turn {
    pub(super) text: String,
    pub(super) finish: String,
    pub(super) tokens: Vec<u32>,
}

pub(super) fn turn_of(
    state: &HybridSchedulerState<Inner>,
    owner: &str,
    result: ChatResult,
) -> Turn {
    Turn {
        text: result.text,
        finish: result.finish_reason,
        tokens: state.owner_states.get(owner).cloned().unwrap_or_default(),
    }
}

/// Drives `state` until every reply in `replies` arrived; the results in
/// order.
pub(super) fn drive_all(
    state: &mut HybridSchedulerState<Inner>,
    receiver: &mut tokio::sync::mpsc::UnboundedReceiver<ModelCommand>,
    replies: &mut [Reply],
) -> Vec<Result<ChatResult>> {
    let mut done: Vec<Option<Result<ChatResult>>> = replies.iter().map(|_| None).collect();
    for _ in 0..4096 {
        for (slot, rx) in done.iter_mut().zip(replies.iter_mut()) {
            if slot.is_none()
                && let Ok(result) = rx.try_recv()
            {
                *slot = Some(result);
            }
        }
        if done.iter().all(Option::is_some) {
            return done.into_iter().map(Option::unwrap).collect();
        }
        assert!(matches!(state.drive(receiver), LoopControl::Continue));
    }
    panic!("the turns did not finish");
}

/// One turn of `owner` alone on `inner`; the scheduler after it.
fn alone(
    inner: Inner,
    owner: &str,
    prompt: &[u32],
    max_new: i32,
    speculate: bool,
) -> (HybridSchedulerState<Inner>, Turn) {
    let mut state = HybridSchedulerState::new(inner).unwrap();
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let (command, rx) = start(owner, prompt, max_new, speculate);
    assert!(sender.send(command).is_ok());
    let result = drive_all(&mut state, &mut receiver, &mut [rx])
        .remove(0)
        .unwrap();
    let turn = turn_of(&state, owner, result);
    (state, turn)
}

/// A prompt of `len` words whose plain greedy turn on `model()` runs all
/// `max_new` tokens; the prompt and that turn.
pub(super) fn full_length(
    model: impl Fn() -> Inner,
    len: usize,
    max_new: i32,
    seed: u32,
) -> (Vec<u32>, Turn) {
    let mut state = HybridSchedulerState::new(model()).unwrap();
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    for k in 0..16 {
        let tokens = prompt(len, seed + 17 * k);
        let (command, rx) = start("plain", &tokens, max_new, false);
        assert!(sender.send(command).is_ok());
        let result = drive_all(&mut state, &mut receiver, &mut [rx])
            .remove(0)
            .unwrap();
        let turn = turn_of(&state, "plain", result);
        if turn.finish == "length" {
            return (tokens, turn);
        }
        let (reply, mut rx) = tokio::sync::oneshot::channel();
        assert!(
            sender
                .send(ModelCommand::from_chat(ChatCmd::ReleaseCacheOwner {
                    owner_id: "plain".to_string(),
                    reply,
                }))
                .is_ok()
        );
        for _ in 0..64 {
            if rx.try_recv().is_ok() {
                break;
            }
            assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
        }
        state
            .inner
            .reset_caches(crate::engine::backend::ResetScope::Command)
            .unwrap();
        state.inner.prefix_states.reset(true);
    }
    panic!("no prompt ran {max_new} tokens");
}

/// Proposals of the plain turn's tokens `full`: the cycle at anchor row
/// `p` proposes `full[p + 1..]` right for its first `k` rows and wrong
/// after, `k` turning through 0..ROWS-1. The `k` of each cycle is pushed to
/// `ks`.
pub(super) fn forced(
    full: Vec<u32>,
    vocab: u32,
    ks: Rc<RefCell<Vec<usize>>>,
) -> Box<dyn FnMut(usize, &[u32]) -> Vec<u32>> {
    Box::new(move |position, ids| {
        let k = ks.borrow().len() % ROWS;
        ks.borrow_mut().push(k);
        (0..ids.len())
            .map(|i| match full.get(position + 1 + i) {
                Some(&t) if i < k => t,
                Some(&t) => (t + 1) % vocab,
                None => 0,
            })
            .collect()
    })
}

fn drafted(mut inner: Inner, draft: DFlash2Model) -> Inner {
    inner.set_draft(Some(Arc::new(draft))).unwrap();
    inner
}

fn slot(state: &mut HybridSchedulerState<Inner>) -> &mut decoder::DraftSlot {
    state.inner.draft_slot().expect("a draft is attached")
}

/// Each drafted cycle of `seq` that a later proposal of `seq` follows kept
/// the forced length `ks[c]` of its proposals, capped by the proposals it
/// handed the engine: the anchor rows of the two proposals are that many
/// tokens and the anchor apart. The number of cycles checked.
pub(super) fn assert_forced_accepts(
    slot: &decoder::DraftSlot,
    seq: u32,
    ks: &[usize],
    what: &str,
) -> usize {
    let rows: Vec<usize> = slot
        .proposals
        .iter()
        .filter(|(owner, ..)| *owner == seq)
        .map(|&(_, row, _)| row)
        .collect();
    let offered: Vec<(usize, usize)> = slot
        .offered
        .iter()
        .filter(|(owner, ..)| *owner == seq)
        .map(|&(_, row, count)| (row, count))
        .collect();
    assert_eq!(
        offered.len(),
        ks.len(),
        "{what}: one forced length per drafted cycle"
    );
    let mut checked = 0;
    for (c, (&(row, count), &k)) in offered.iter().zip(ks).enumerate() {
        let Some(next) = rows.iter().find(|&&later| later > row) else {
            continue;
        };
        assert_eq!(
            next - row - 1,
            k.min(count),
            "{what}: cycle {c} at row {row}: accepted proposals"
        );
        checked += 1;
    }
    checked
}

/// Both contexts hold the same rows: length, tokens and the bits of each
/// layer's live K/V rows.
pub(super) fn assert_same_rows(got: &DFlash2ContextCache, want: &DFlash2ContextCache, what: &str) {
    assert_eq!(got.logical_len(), want.logical_len(), "{what}: length");
    assert_eq!(got.token_history(), want.token_history(), "{what}: tokens");
    let bits = |a: &MxArray| -> Vec<u32> { f32s(a).iter().map(|v| v.to_bits()).collect() };
    let (got, want) = (got.window_views(), want.window_views());
    assert_eq!(got.len(), want.len(), "{what}: layers");
    for (layer, (got, want)) in got.iter().zip(&want).enumerate() {
        let same = match (got, want) {
            (Some((gk, gv)), Some((wk, wv))) => {
                gk.shape().unwrap().as_ref() == wk.shape().unwrap().as_ref()
                    && bits(gk) == bits(wk)
                    && bits(gv) == bits(wv)
            }
            (None, None) => true,
            _ => false,
        };
        assert!(same, "{what}: layer {layer}: the K/V rows");
    }
}

/// The switch's rule and each of its branches on the model: no draft offers
/// no speculation; a draft offers it and starts the cycle for a turn alone
/// on a whole store, with the draft's taps; a second turn open at the start
/// or a partial store keep the plain path and no tap; a peer that opens
/// beside a drafted turn ends its cycle (no proposal, no tap); the turn's
/// end turns the taps off.
/// MUTATION: the rule ignores the store → red (a partial store).
/// MUTATION: the rule ignores the open turns → red (beside another turn).
/// MUTATION: speculation is offered with no draft → red (no draft).
/// MUTATION: a peer leaves the drafted turn's cycle on → red (a peer).
/// MUTATION: the turn's end leaves the taps on → red (the turn's end).
/// MUTATION: the open turns leave out an exclusive turn → red (beside an
/// exclusive turn).
#[test]
fn the_switch_drafts_only_a_lone_turn_on_a_whole_store() {
    let _host = memory::scoped_fixture_host();
    for (attached, direct, filled, open, on) in [
        (true, true, true, 1, true),
        (true, true, true, 0, true),
        (false, true, true, 1, false),
        (true, false, true, 1, false),
        (true, true, false, 1, false),
        (true, true, false, 0, false),
        (true, true, true, 2, false),
        (true, true, true, 4, false),
        (false, false, false, 2, false),
    ] {
        assert_eq!(
            decoder::draft_turn(attached, direct, filled, open),
            on,
            "attached {attached}, direct {direct}, filled {filled}, {open} open"
        );
    }
    let c = super::route_step::moe_shape_config();
    let draft = || moe_draft(c.hidden_size, c.vocab_size, c.num_hidden_layers);
    let layers = draft().config.target_layers.clone();
    let greedy = crate::engine::params::extract_chat_params(&ChatConfig {
        temperature: Some(0.0),
        ..Default::default()
    });

    let mut inner = direct_inner();
    assert!(
        !inner.supports_scheduled_speculation() && inner.execution_plan().speculative.is_none(),
        "no draft: no speculation offered"
    );
    inner.scheduled_turn_opened(1);
    assert!(!inner.begin_draft_turn(1).unwrap(), "no draft: plain");
    inner.scheduled_turn_closed(1);

    let mut inner = drafted(direct_inner(), draft());
    assert!(inner.supports_scheduled_speculation());
    assert_eq!(
        inner.execution_plan().speculative.map(|plan| plan.kind),
        Some(crate::engine::plan::SpeculativeKind::DraftModel)
    );
    assert!(
        inner
            .resolve_params(&ChatConfig {
                temperature: Some(0.0),
                enable_mtp: Some(true),
                mtp_adaptive_depth: Some(true),
                ..Default::default()
            })
            .mtp_adaptive_depth
            && inner.supports_adaptive_scheduled_speculation()
            && inner.scheduled_verification_budget().is_none(),
        "an adaptive request: the draft's own rows, no measured budget, no exclusive lane"
    );
    inner.scheduled_turn_opened(1);
    assert!(
        inner.begin_scheduled_speculation(1, 0).unwrap(),
        "a lone turn on a whole store drafts"
    );
    assert_eq!(inner.decoder.hidden_tap_layers(), Some(layers.as_slice()));
    inner.scheduled_turn_closed(1);
    assert_eq!(
        inner.decoder.hidden_tap_layers(),
        None,
        "the turn's end turns the taps off"
    );

    inner.scheduled_turn_opened(2);
    inner.scheduled_turn_opened(3);
    assert!(
        !inner.begin_scheduled_speculation(2, 0).unwrap(),
        "a turn beside another open turn stays plain"
    );
    assert_eq!(inner.decoder.hidden_tap_layers(), None);
    inner.scheduled_turn_closed(2);
    inner.scheduled_turn_closed(3);

    inner.exclusive_turn = true;
    inner.scheduled_turn_opened(6);
    assert!(
        !inner.begin_scheduled_speculation(6, 0).unwrap(),
        "a turn beside an exclusive turn stays plain"
    );
    assert_eq!(inner.decoder.hidden_tap_layers(), None);
    inner.scheduled_turn_closed(6);
    inner.exclusive_turn = false;

    inner.scheduled_turn_opened(4);
    assert!(inner.begin_scheduled_speculation(4, 0).unwrap());
    inner.scheduled_turn_opened(5);
    assert_eq!(
        inner.decoder.hidden_tap_layers(),
        None,
        "a peer turns the taps off"
    );
    let proposal = inner
        .propose_scheduled(4, 7, ROWS - 1, &greedy, &mut rand::rng(), false)
        .unwrap();
    assert!(
        proposal.draft_ids.is_empty(),
        "a peer ends the drafted turn's proposals"
    );
    assert!(!inner.begin_scheduled_speculation(5, 0).unwrap());
    inner.scheduled_turn_closed(4);
    inner.scheduled_turn_closed(5);
    let starts = &inner.draft_slot().unwrap().starts;
    assert_eq!(
        starts.as_slice(),
        [(1, true), (2, false), (6, false), (4, true), (5, false)]
    );

    let mut partial = drafted(super::route_step::moe_shape_inner(12), draft());
    assert!(!partial.decoder.weights.plan.direct);
    partial.scheduled_turn_opened(1);
    assert!(
        !partial.begin_scheduled_speculation(1, 0).unwrap(),
        "a partial store keeps the plain path"
    );
    assert_eq!(partial.decoder.hidden_tap_layers(), None);
}

/// Owner a starts alone on a whole store and runs the draft's cycle,
/// forced through accept lengths 0..3; owner b opens beside it mid-turn.
/// From its next cycle a proposes nothing, b never drafts, and no pass is
/// tapped. Each owner's text and tokens equal its plain turn alone. Each
/// drafted cycle kept its forced length. The draft's context, which the
/// scheduled cycles fill from the verify passes' taps, holds the rows and
/// K/V bits the stepper's own commits give over the same four cycles.
/// MUTATION: a peer leaves the drafted turn's cycle on → red (no proposal
/// after the peer opened).
/// MUTATION: a proposal reaches the engine one row late → red (accepted
/// proposals).
/// MUTATION: the context takes no K/V row from the taps → red (the K/V
/// rows).
#[test]
fn a_peer_ends_the_drafted_cycles_at_a_cycle_boundary() {
    let _host = memory::scoped_fixture_host();
    const MAX_NEW: i32 = 24;
    let model = || chat_model(reuse_over(direct_inner()));
    let c = super::route_step::moe_shape_config();
    let draft = || moe_draft(c.hidden_size, c.vocab_size, c.num_hidden_layers);
    let (a, plain_a) = full_length(model, 40, MAX_NEW, 1);
    let (b, plain_b) = full_length(model, 33, MAX_NEW, 5);
    let mut full = plain_a.tokens.clone();
    full.push(0);

    let (_, lone) = alone(drafted(model(), draft()), "plain", &a, MAX_NEW, true);
    assert_eq!(lone, plain_a, "a drafted alone");

    let ks: Rc<RefCell<Vec<usize>>> = Rc::default();
    let mut state = HybridSchedulerState::new(drafted(model(), draft())).unwrap();
    slot(&mut state).forced = Some(forced(full, c.vocab_size as u32, Rc::clone(&ks)));
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let (command, rx_a) = start("plain", &a, MAX_NEW, true);
    assert!(sender.send(command).is_ok());
    let mut rx_a = rx_a;
    let mut steps = 0;
    while ks.borrow().len() < ROWS + 1 {
        assert!(rx_a.try_recv().is_err(), "a ended before b opened");
        assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
        steps += 1;
        assert!(steps < 4096);
    }
    let before = slot(&mut state).proposals.len();
    let (command, rx_b) = start("b", &b, MAX_NEW, true);
    assert!(sender.send(command).is_ok());
    let results = drive_all(&mut state, &mut receiver, &mut [rx_a, rx_b]);
    let seq_a = state.owner_sequences["plain"];
    let seq_b = state.owner_sequences["b"];
    let records = slot(&mut state);
    let proposals = records.proposals.clone();
    let starts = records.starts.clone();
    let after: Vec<_> = proposals[before..]
        .iter()
        .filter(|(seq, ..)| *seq == seq_a)
        .collect();
    assert!(
        proposals[..before]
            .iter()
            .all(|&(seq, _, drafted)| seq == seq_a && drafted),
        "a drafted alone: {proposals:?}"
    );
    assert!(!after.is_empty(), "a ran cycles beside b");
    assert!(
        after.iter().all(|(_, _, drafted)| !drafted),
        "no proposal after the peer opened: {after:?}"
    );
    assert!(
        after.windows(2).all(|pair| pair[0].1 < pair[1].1),
        "a's cycles go on from where they were"
    );
    assert_eq!(starts, [(seq_a, true), (seq_b, false)]);
    let forced_ks: std::collections::BTreeSet<usize> = ks.borrow().iter().copied().collect();
    eprintln!(
        "peer: a ran {before} drafted cycles alone and {} plain cycles beside b; forced lengths {:?}; a {} tokens, b {} tokens",
        after.len(),
        ks.borrow(),
        plain_a.tokens.len(),
        plain_b.tokens.len()
    );
    assert_eq!(forced_ks, (0..ROWS).collect(), "every accept length ran");
    let checked = assert_forced_accepts(slot(&mut state), seq_a, &ks.borrow(), "peer");
    assert_eq!(checked, ROWS + 1, "peer: every drafted cycle checked");
    assert_eq!(state.inner.decoder.hidden_tap_layers(), None);

    assert_eq!(plain_a.tokens[..a.len()], a, "the prompt is the words' ids");
    let mut reference = tapped(model(), &Arc::new(draft()));
    drafted_turn(
        &mut reference,
        seq_a,
        &a,
        ROWS,
        ROWS * (ROWS + 1) / 2 + 1,
        Some(&plain_a.tokens[a.len()..]),
        true,
        nothing,
    );
    let (_, got) = state.inner.draft_parts().unwrap();
    let (_, want) = reference.draft_parts().unwrap();
    assert_eq!(
        got.token_history(),
        &plain_a.tokens[..got.token_history().len()],
        "the context follows a's rows"
    );
    assert_same_rows(got, want, "the context after four cycles");
    let mut results = results.into_iter();
    let got_a = results.next().unwrap().unwrap();
    let got_b = results.next().unwrap().unwrap();
    assert_eq!(turn_of(&state, "plain", got_a), plain_a, "a beside b");
    assert_eq!(turn_of(&state, "b", got_b), plain_b, "b beside a");
}

/// A host-gate store with the draft attached: turn 1 starts on the partial
/// store and takes the plain path (no tap; the submit log of the same turn
/// with no draft); its close moves the store up to whole and its banks start
/// to fill, held until a pass waits for a layer; turn 2 starts while they
/// fill and takes the plain path; turn 3 starts on the filled banks and runs
/// the draft's cycle. Every turn's text and tokens equal the same turns
/// with no draft.
/// MUTATION: the rule ignores the store → red (turn 1 drafted).
/// MUTATION: the fill gate is ignored → red (turn 2 drafted).
#[test]
fn a_store_that_moves_up_drafts_the_turn_after_its_fill() {
    const MAX_NEW: i32 = 9;
    let bank = bank_bytes();
    let model = || chat_model(host_gate_model(bank));
    let c = super::route_step::deep_moe_shape_config();
    let draft = || moe_draft(c.hidden_size, c.vocab_size, c.num_hidden_layers);
    let quiet = 3u64 << 30;
    let first = prompt(16, 2);
    let second = prompt(14, 4);
    let (third, _) = full_length(
        || chat_model(super::promote::model_direct()),
        15,
        MAX_NEW,
        6,
    );
    let run = |inner: Inner, ks: Option<Rc<RefCell<Vec<usize>>>>, full: Vec<u32>| {
        let _host = memory::scoped_test_host(host_with(quiet + own(&inner)));
        let mut state = HybridSchedulerState::new(inner).unwrap();
        state.inner.decoder.weights.live.fill_hold.hold();
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        state.inner.decoder.submit_log.borrow_mut().clear();
        let mut turn = |state: &mut HybridSchedulerState<Inner>, owner: &str, tokens: &[u32]| {
            let (command, rx) = start(owner, tokens, MAX_NEW, true);
            assert!(sender.send(command).is_ok());
            let result = drive_all(state, &mut receiver, &mut [rx])
                .remove(0)
                .unwrap();
            turn_of(state, owner, result)
        };
        let one = turn(&mut state, "one", &first);
        let log = state.inner.decoder.submit_log.borrow().clone();
        let moved = state.inner.decoder.weights.live.residency();
        let two = turn(&mut state, "two", &second);
        let filling = state.inner.decoder.weights.direct_fill_pass();
        if let Some(ks) = ks {
            slot(&mut state).forced = Some(forced(full, c.vocab_size as u32, ks));
        }
        let three = turn(&mut state, "three", &third);
        (state, [one, two, three], log, moved, filling)
    };
    let (_, plain, plain_log, plain_moved, plain_filling) = run(model(), None, Vec::new());
    assert_eq!(
        plain_moved,
        Residency::Direct,
        "turn 1's close moved the store"
    );
    assert!(plain_filling, "turn 2 started while the banks filled");
    let mut full = plain[2].tokens.clone();
    full.push(0);
    let ks: Rc<RefCell<Vec<usize>>> = Rc::default();
    let (mut state, turns, log, moved, filling) =
        run(drafted(model(), draft()), Some(Rc::clone(&ks)), full);
    assert_eq!(moved, Residency::Direct);
    assert!(filling, "turn 2 started while the banks filled");
    let seqs = ["one", "two", "three"].map(|owner| state.owner_sequences[owner]);
    let records = slot(&mut state);
    assert_eq!(
        records.starts,
        [(seqs[0], false), (seqs[1], false), (seqs[2], true)],
        "turn 1 plain on the partial store, turn 2 plain while the banks fill, turn 3 drafted"
    );
    assert!(
        records
            .proposals
            .iter()
            .all(|&(seq, _, drafted)| seq == seqs[2] && drafted)
    );
    assert!(!ks.borrow().is_empty(), "turn 3 ran the draft's cycle");
    let checked = assert_forced_accepts(records, seqs[2], &ks.borrow(), "after the fill");
    assert!(
        checked + 1 >= ks.borrow().len() && checked >= ROWS - 1,
        "after the fill: {checked} drafted cycles checked"
    );
    eprintln!(
        "fill gate: starts {:?}; turn 3 drafted cycles {}; forced lengths {:?}; submit log {} events",
        records.starts,
        records.proposals.len(),
        ks.borrow(),
        log.len()
    );
    assert!(
        log == plain_log,
        "turn 1: the submit log of the turn with no draft"
    );
    for (k, (got, want)) in turns.iter().zip(&plain).enumerate() {
        assert_eq!(got, want, "turn {}", k + 1);
    }
    assert_eq!(state.inner.decoder.hidden_tap_layers(), None);
}
