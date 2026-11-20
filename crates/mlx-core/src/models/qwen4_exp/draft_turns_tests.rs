//! The draft's context across scheduled turns, restores and the state
//! store. The next turn on a sequence drafts on the context the drafted
//! turn's end left. A turn restored from a stored state (in memory, or from
//! disk after a restart) rebuilds the context from the rows its own prefill
//! tapped: nothing of the draft is stored. A regenerate and an edit of the
//! last message do the same. Every drafted turn gives the plain turn's text
//! and tokens, and drafted turns leave the plain turns' files on disk.
use super::draft_cycle::{drafted_turn, moe_draft, nothing, tapped};
use super::draft_switch::{
    ROWS, Reply, Turn, assert_forced_accepts, assert_same_rows, chat_model, drive_all, forced,
    full_length, start, start_with, turn_of,
};
use super::lanes::direct_inner;
use super::prefix_disk_tests::{Root, disk, on_disk, settle};
use super::prefix_reuse::{finish, prompt, reuse_over, turn};
use super::*;
use crate::engine::hybrid_scheduler::HybridSchedulerState;
use crate::engine::types::ChatConfig;
use crate::model_thread::LoopControl;
use crate::models::qwen3_5::dflash2::{DFlash2ContextCache, DFlash2Model};
use std::cell::RefCell;
use std::rc::Rc;

/// The row of the state a turn of these prompts restores: the replay point
/// of the 66-token first prompt.
const RESTORED: usize = 64;

/// A scheduler over one model and the channel its commands come in on.
struct Driver {
    state: HybridSchedulerState<Inner>,
    sender: tokio::sync::mpsc::UnboundedSender<ModelCommand>,
    receiver: tokio::sync::mpsc::UnboundedReceiver<ModelCommand>,
    /// A turn that asks for the cycle leaves its rows to the model (the
    /// server's request: no depth), not 4 rows.
    rule: bool,
}

/// What a turn gave and the rows it took from a cache.
struct Said {
    turn: Turn,
    cached: u32,
}

impl Driver {
    fn new(inner: Inner) -> Self {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        Self {
            state: HybridSchedulerState::new(inner).unwrap(),
            sender,
            receiver,
            rule: false,
        }
    }

    fn slot(&mut self) -> &mut decoder::DraftSlot {
        self.state.inner.draft_slot().expect("a draft is attached")
    }

    /// The draft proposes the tokens of the plain turn `plain` (the owner's
    /// tokens after it), forced through accept lengths 0..ROWS-1; the
    /// forced length of each cycle.
    fn force(&mut self, plain: &[u32]) -> Rc<RefCell<Vec<usize>>> {
        let vocab = self.state.inner.decoder.config.vocab_size as u32;
        let mut full = plain.to_vec();
        full.push(0);
        let ks: Rc<RefCell<Vec<usize>>> = Rc::default();
        self.slot().forced = Some(forced(full, vocab, Rc::clone(&ks)));
        ks
    }

    /// Starts a greedy turn of `owner` over `tokens`; a drafted model's
    /// proposal records start again.
    fn send(&mut self, owner: &str, tokens: &[u32], max_new: i32, speculate: bool) -> Reply {
        if let Some(slot) = self.state.inner.draft_slot() {
            slot.proposals.clear();
            slot.offered.clear();
            slot.chosen.clear();
        }
        let (command, rx) = if self.rule && speculate {
            let config = ChatConfig {
                max_new_tokens: Some(max_new),
                temperature: Some(0.0),
                enable_mtp: Some(true),
                ..Default::default()
            };
            start_with(owner, tokens, config)
        } else {
            start(owner, tokens, max_new, speculate)
        };
        assert!(self.sender.send(command).is_ok());
        rx
    }

    fn finish(&mut self, owner: &str, rx: Reply) -> Said {
        let result = drive_all(&mut self.state, &mut self.receiver, &mut [rx])
            .remove(0)
            .unwrap();
        let cached = result.cached_tokens;
        Said {
            turn: turn_of(&self.state, owner, result),
            cached,
        }
    }

    fn turn(&mut self, owner: &str, tokens: &[u32], max_new: i32, speculate: bool) -> Said {
        let rx = self.send(owner, tokens, max_new, speculate);
        self.finish(owner, rx)
    }

    /// A turn that asks for the draft's cycle; `at_first` gets the model
    /// right after the turn's first proposal.
    fn drafted(
        &mut self,
        owner: &str,
        tokens: &[u32],
        max_new: i32,
        at_first: impl FnOnce(&mut Inner),
    ) -> Said {
        let mut rx = self.send(owner, tokens, max_new, true);
        let mut steps = 0;
        while self.slot().offered.is_empty() {
            assert!(
                rx.try_recv().is_err(),
                "{owner}: the turn ended before it proposed"
            );
            assert!(matches!(
                self.state.drive(&mut self.receiver),
                LoopControl::Continue
            ));
            steps += 1;
            assert!(steps < 4096, "{owner}: no proposal");
        }
        at_first(&mut self.state.inner);
        self.finish(owner, rx)
    }

    fn seq(&self, owner: &str) -> u32 {
        self.state.owner_sequences[owner]
    }
}

fn with_draft(mut inner: Inner, draft: &Arc<DFlash2Model>) -> Inner {
    inner.set_draft(Some(Arc::clone(draft))).unwrap();
    inner
}

fn context(inner: &Inner) -> &DFlash2ContextCache {
    inner.draft_parts().expect("a draft is attached").1
}

/// The context a turn of `tokens` restored at row `restored` holds at its
/// first proposal: the rows below `restored` counted only, then the rows of
/// the ring from its prefill's taps. The taps come from a cold prefill of
/// `tokens` on `model`, whose passes end where the restored turn's do and
/// carry the same bits from `restored` on.
fn rebuilt(
    model: Inner,
    draft: &Arc<DFlash2Model>,
    tokens: &[u32],
    restored: usize,
) -> DFlash2ContextCache {
    let mut cold = tapped(model, draft);
    let (cached, _) = turn(&mut cold, 1, tokens);
    assert_eq!(cached, 0, "the reference prefills cold");
    let passes = cold.decoder.take_hidden_taps();
    let mut context = DFlash2ContextCache::new(&draft.config);
    context.record_only(&tokens[..restored]).unwrap();
    decoder::follow_history(draft, &mut context, &passes, tokens).unwrap();
    context
}

/// Every object file of the one namespace under `root`: name and bytes,
/// by name.
fn objects(root: &Root) -> Vec<(String, Vec<u8>)> {
    let mut out: Vec<(String, Vec<u8>)> = std::fs::read_dir(root.namespace())
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".safetensors"))
        .map(|name| {
            let bytes = std::fs::read(root.namespace().join(&name)).unwrap();
            (name, bytes)
        })
        .collect();
    out.sort();
    out
}

/// An object file's header (its JSON; a block's metadata map has no fixed
/// key order) and the bytes after it.
fn object_parts(bytes: &[u8]) -> (serde_json::Value, &[u8]) {
    let mut len = [0u8; 8];
    len.copy_from_slice(&bytes[..8]);
    let end = 8 + u64::from_le_bytes(len) as usize;
    (
        serde_json::from_slice(&bytes[8..end]).unwrap(),
        &bytes[end..],
    )
}

/// The parts of a state record that the turns alone decide: its header less
/// `last_used` and the checksum over it, and its payload. `last_used` is the
/// store's stamp when the write goes out (`label_of` in `Disk::write`):
/// before the turn's end restamps its replay state (`end_use`) when the turn
/// outlasts `DELAY`, after it otherwise.
fn record_parts(bytes: &[u8]) -> (serde_json::Value, &[u8]) {
    let (mut header, payload) = object_parts(bytes);
    let meta = header["__metadata__"]
        .as_object_mut()
        .expect("a state record's header has its metadata");
    assert!(meta.remove("last_used").is_some(), "a recency stamp");
    assert!(meta.remove("checksum").is_some(), "a checksum");
    (header, payload)
}

fn concat(parts: &[&[u32]]) -> Vec<u32> {
    parts.concat()
}

/// A drafted turn, then a second turn on the same sequence. At the first
/// turn's end the context holds the turn's rows with the K/V a stepper's
/// own cycles over the same forced proposals give, and no layer is tapped.
/// The second turn resumes the live row, drafts, and at its first proposal
/// holds that context followed by its prompt's rows, as the stepper's
/// does. Both turns give the plain turns' text and tokens.
/// MUTATION: the turn's end leaves the taps on → red (turn 1's end turns
/// the taps off).
#[test]
fn a_drafted_turn_carries_its_context_into_the_next_turn() {
    let _host = memory::scoped_fixture_host();
    const FIRST: i32 = 12;
    const SECOND: i32 = 8;
    let model = || chat_model(reuse_over(direct_inner()));
    let c = model().decoder.config.clone();
    let draft = Arc::new(moe_draft(c.hidden_size, c.vocab_size, c.num_hidden_layers));
    let (first, _) = full_length(model, 40, FIRST, 1);

    let mut plain = Driver::new(model());
    let plain1 = plain.turn("a", &first, FIRST, false);
    let h1 = plain1.turn.tokens.clone();
    let second = concat(&[&h1, &prompt(3, 7)]);
    let plain2 = plain.turn("a", &second, SECOND, false);
    assert_eq!(plain2.cached as usize, h1.len(), "plain: turn 2 resumes");
    drop(plain);

    let mut run = Driver::new(with_draft(model(), &draft));
    let ks1 = run.force(&plain1.turn.tokens);
    let got1 = run.turn("a", &first, FIRST, true);
    assert_eq!(got1.turn, plain1.turn, "turn 1");
    assert_eq!(
        run.state.inner.decoder.hidden_tap_layers(),
        None,
        "turn 1's end turns the taps off"
    );
    let seq = run.seq("a");
    let checked = assert_forced_accepts(run.slot(), seq, &ks1.borrow(), "turn 1");
    assert!(
        checked >= ROWS - 1,
        "turn 1: {checked} drafted cycles checked"
    );

    let mut reference = tapped(model(), &draft);
    drafted_turn(
        &mut reference,
        seq,
        &first,
        ROWS,
        FIRST as usize,
        Some(&h1[first.len()..]),
        true,
        nothing,
    );
    assert_eq!(
        context(&run.state.inner).token_history(),
        h1.as_slice(),
        "turn 1's end: the context holds the turn's rows"
    );
    assert_same_rows(
        context(&run.state.inner),
        context(&reference),
        "turn 1's end: the context",
    );
    finish(&mut reference);
    let (cached, _) = turn(&mut reference, seq, &second);
    assert_eq!(cached, h1.len(), "the reference resumes the live row");
    assert!(
        reference.draft_stepper(ROWS).unwrap().is_some(),
        "the reference follows its history"
    );

    let ks2 = run.force(&plain2.turn.tokens);
    let got2 = run.drafted("a", &second, SECOND, |inner| {
        assert_same_rows(
            context(inner),
            context(&reference),
            "turn 2's first proposal: the context",
        );
    });
    assert_eq!(got2.turn, plain2.turn, "turn 2");
    assert_eq!(
        got2.cached as usize,
        h1.len(),
        "turn 2 resumes the live row"
    );
    let records = run.slot();
    assert_eq!(records.starts, [(seq, true), (seq, true)]);
    let checked = assert_forced_accepts(records, seq, &ks2.borrow(), "turn 2");
    assert!(checked >= 1, "turn 2: {checked} drafted cycles checked");
    eprintln!(
        "carry: turn 1 forced {:?}, turn 2 forced {:?}; history {} then {} rows",
        ks1.borrow(),
        ks2.borrow(),
        h1.len(),
        got2.turn.tokens.len()
    );
    assert_eq!(run.state.inner.decoder.hidden_tap_layers(), None);
}

/// The server's request (no depth: the model's rule sets each verify's
/// rows) on a drafted turn whose first proposals are refused and later ones
/// right, then a plain turn on the same live row. The rule picks more than
/// one row count; the drafted turn gives the plain
/// turn's text and tokens and leaves its history and pages; after the plain
/// turn every state array, page, scale and index block equals the plain
/// model's.
#[test]
fn rows_from_the_rule_leave_the_plain_state() {
    let _host = memory::scoped_fixture_host();
    const FIRST: i32 = 24;
    const SECOND: i32 = 8;
    // The first cycles' proposals are all refused, so the rule's rows fall;
    // every later one is right.
    const REFUSED: usize = 8;
    let model = || chat_model(reuse_over(direct_inner()));
    let c = model().decoder.config.clone();
    let draft = Arc::new(moe_draft(c.hidden_size, c.vocab_size, c.num_hidden_layers));
    let (first, _) = full_length(model, 40, FIRST, 5);

    let mut plain = Driver::new(model());
    let plain1 = plain.turn("a", &first, FIRST, false);
    let mut run = Driver::new(with_draft(model(), &draft));
    run.rule = true;
    let vocab = c.vocab_size as u32;
    let mut full = plain1.turn.tokens.clone();
    full.push(0);
    let ks: Rc<RefCell<Vec<usize>>> = Rc::default();
    let record = Rc::clone(&ks);
    run.slot().forced = Some(Box::new(move |position, ids: &[u32]| {
        let k = if record.borrow().len() < REFUSED {
            0
        } else {
            ROWS - 1
        };
        record.borrow_mut().push(k);
        (0..ids.len())
            .map(|i| match full.get(position + 1 + i) {
                Some(&t) if i < k => t,
                Some(&t) => (t + 1) % vocab,
                None => 0,
            })
            .collect()
    }));
    let got1 = run.turn("a", &first, FIRST, true);
    let seq = run.seq("a");
    let rows: std::collections::BTreeSet<usize> =
        run.slot().chosen.iter().map(|chosen| chosen.4).collect();
    eprintln!(
        "rule rows {rows:?}; forced {:?}; offered {:?}",
        ks.borrow(),
        run.slot()
            .offered
            .iter()
            .map(|&(_, _, n)| n)
            .collect::<Vec<_>>()
    );
    assert!(
        rows.len() >= 2,
        "the rule picks more than one row count: {rows:?}"
    );
    let checked = assert_forced_accepts(run.slot(), seq, &ks.borrow(), "turn 1");
    assert!(
        checked >= ROWS - 1,
        "turn 1: {checked} drafted cycles checked"
    );
    assert_eq!(got1.turn, plain1.turn, "turn 1");
    assert_eq!(
        run.state.inner.decoder.history, plain.state.inner.decoder.history,
        "turn 1: history"
    );
    assert_eq!(
        run.state
            .inner
            .decoder
            .paged
            .as_ref()
            .map(|p| p.request_tokens()),
        plain
            .state
            .inner
            .decoder
            .paged
            .as_ref()
            .map(|p| p.request_tokens()),
        "turn 1: pages"
    );

    let second = concat(&[&plain1.turn.tokens, &prompt(3, 11)]);
    let plain2 = plain.turn("a", &second, SECOND, false);
    let got2 = run.turn("a", &second, SECOND, false);
    assert_eq!(got2.turn, plain2.turn, "turn 2");
    assert_eq!(
        (got2.cached, plain2.cached),
        (
            plain1.turn.tokens.len() as u32,
            plain1.turn.tokens.len() as u32
        ),
        "turn 2 resumes the live row"
    );
    super::route_step::assert_same_decoder_state(
        &mut run.state.inner.decoder,
        &mut plain.state.inner.decoder,
        "after the plain turn",
    );
}

/// Drafted turns on a model whose state store has its persistent tier: a
/// first turn of owner a, a turn of owner b on a new sequence restored from
/// the state turn a1 published (in memory), and owner a's next turn on its
/// live row. Each drafted turn's row reads the persistent tier, b's context
/// at its first proposal is the one rebuilt from its own prefill's taps,
/// and the store after the turns holds the files of the same turns with no
/// draft: the state records' headers but their recency stamp and its
/// checksum, and their payloads byte for byte; every block object's header
/// and data (a block header's keys come in no fixed order, run to run). After a
/// restart (a new model over the same
/// directory) a turn of the conversation restores from disk: a drafted
/// model over either store rebuilds its context from its prefill's taps,
/// drafts and gives the plain text; a model with no draft restores the
/// drafted turns' store and gives the plain text.
/// MUTATION: a new sequence keeps the context of the older turn → red (b's
/// context at its first proposal).
/// MUTATION: a drafted model's turn does not resume its live row (it
/// restores and publishes one more state) → red (the drafted turns' state
/// records: names).
#[test]
fn drafted_turns_keep_the_persistent_cache() {
    keeps_the_persistent_cache(false, false);
}

/// [`drafted_turns_keep_the_persistent_cache`] with a1's state written before
/// a1's end, as a turn that outlasts the write delay writes it: its record
/// has an older recency stamp than the plain one and is otherwise the same.
#[test]
fn drafted_turns_written_before_their_end_keep_the_persistent_cache() {
    keeps_the_persistent_cache(false, true);
}

/// [`drafted_turns_keep_the_persistent_cache`] with the server's request:
/// no depth, so the model's rule sets each verify's rows.
#[test]
fn drafted_turns_on_the_rows_rule_keep_the_persistent_cache() {
    keeps_the_persistent_cache(true, false);
}

/// The persistent cache proof, every drafted turn at 4 rows or on the
/// rows rule (`rule`); `early` writes a1's state before a1's end.
fn keeps_the_persistent_cache(rule: bool, early: bool) {
    let _host = memory::scoped_fixture_host();
    let model = || chat_model(reuse_over(direct_inner()));
    let c = model().decoder.config.clone();
    let draft = Arc::new(moe_draft(c.hidden_size, c.vocab_size, c.num_hidden_layers));
    let (first, _) = full_length(model, 66, 4, 3);

    let plain_root = Root::new();
    let mut plain = Driver::new(on_disk(model(), &plain_root));
    let a1 = plain.turn("a", &first, 4, false);
    settle(&mut plain.state.inner);
    let h1 = a1.turn.tokens.clone();
    let of_b = concat(&[&h1, &prompt(1, 21)]);
    let b = plain.turn("b", &of_b, 4, false);
    settle(&mut plain.state.inner);
    let second = concat(&[&h1, &prompt(30, 23)]);
    let a2 = plain.turn("a", &second, 6, false);
    settle(&mut plain.state.inner);
    assert_eq!(b.cached as usize, RESTORED, "plain: b restores a1's state");
    assert_eq!(a2.cached as usize, h1.len(), "plain: a2 resumes its row");
    drop(plain);

    let drafted_root = Root::new();
    let mut run = Driver::new(with_draft(on_disk(model(), &drafted_root), &draft));
    run.rule = rule;
    let handle = disk(&run.state.inner).tier_handle();
    run.force(&a1.turn.tokens);
    let got_a1 = if early {
        run.state.inner.disk.as_mut().unwrap().delay = std::time::Duration::ZERO;
        run.drafted("a", &first, 4, |inner| inner.step_disk_now())
    } else {
        run.turn("a", &first, 4, true)
    };
    let mode_a1 = run.state.inner.telemetry.mode(&handle);
    let offered_a1 = run.slot().offered.len();
    let mut ruled = vec![run.slot().chosen.len()];
    settle(&mut run.state.inner);
    let want_b = rebuilt(model(), &draft, &of_b, RESTORED);
    run.force(&b.turn.tokens);
    let got_b = run.drafted("b", &of_b, 4, |inner| {
        assert_same_rows(
            context(inner),
            &want_b,
            "b, restored in memory: the context at its first proposal",
        );
    });
    let mode_b = run.state.inner.telemetry.mode(&handle);
    let offered_b = run.slot().offered.len();
    ruled.push(run.slot().chosen.len());
    settle(&mut run.state.inner);
    run.force(&a2.turn.tokens);
    let got_a2 = run.turn("a", &second, 6, true);
    let mode_a2 = run.state.inner.telemetry.mode(&handle);
    let offered_a2 = run.slot().offered.len();
    ruled.push(run.slot().chosen.len());
    settle(&mut run.state.inner);
    let (seq_a, seq_b) = (run.seq("a"), run.seq("b"));
    let starts = run.slot().starts.clone();
    let taps = run.state.inner.decoder.hidden_tap_layers().is_none();
    drop(run);
    assert!(
        ruled.iter().all(|&calls| (calls > 0) == rule),
        "rule {rule}: the rows rule's calls per drafted turn: {ruled:?}"
    );

    let plain_files = objects(&plain_root);
    let drafted_files = objects(&drafted_root);
    const RECORD: &str = ".dflash2_state.safetensors";
    let names = |files: &[(String, Vec<u8>)], records: bool| -> Vec<String> {
        files
            .iter()
            .filter(|(name, _)| !records || name.ends_with(RECORD))
            .map(|(name, _)| name.clone())
            .collect()
    };
    if early {
        let stamps = |files: &[(String, Vec<u8>)]| -> Vec<serde_json::Value> {
            files
                .iter()
                .filter(|(name, _)| name.ends_with(RECORD))
                .map(|(_, bytes)| object_parts(bytes).0["__metadata__"]["last_used"].clone())
                .collect()
        };
        assert_ne!(
            stamps(&drafted_files),
            stamps(&plain_files),
            "a1's record written before a1's end: an older stamp"
        );
    }
    let states = names(&plain_files, true).len();
    assert!(states >= 1, "the plain turns wrote a state");
    assert_eq!(
        names(&drafted_files, true),
        names(&plain_files, true),
        "the drafted turns' state records: names"
    );
    assert_eq!(
        names(&drafted_files, false),
        names(&plain_files, false),
        "the drafted turns' files: names"
    );
    for ((name, got), (_, want)) in drafted_files.iter().zip(&plain_files) {
        let parts: fn(&[u8]) -> (serde_json::Value, &[u8]) = if name.ends_with(RECORD) {
            record_parts
        } else {
            object_parts
        };
        let ((got_header, got_data), (want_header, want_data)) = (parts(got), parts(want));
        assert_eq!(got_header, want_header, "{name}: the drafted turns' header");
        assert!(
            got_data == want_data,
            "{name}: the drafted turns' data: first differing byte {:?}, {} and {} bytes",
            got_data.iter().zip(want_data).position(|(a, b)| a != b),
            got_data.len(),
            want_data.len()
        );
    }
    eprintln!(
        "store: {} files ({states} states, {} bytes) on both stores",
        plain_files.len(),
        plain_files
            .iter()
            .map(|(_, bytes)| bytes.len())
            .sum::<usize>()
    );

    assert_eq!(got_a1.turn, a1.turn, "a1");
    assert_eq!(got_b.turn, b.turn, "b");
    assert_eq!(got_a2.turn, a2.turn, "a2");
    assert_eq!(got_b.cached as usize, RESTORED, "b restores a1's state");
    assert_eq!(got_a2.cached as usize, h1.len(), "a2 resumes its row");
    assert_eq!(
        [mode_a1, mode_b, mode_a2],
        [1, 1, 1],
        "every row: persistent"
    );
    assert!(
        offered_a1 > 0 && offered_b > 0 && offered_a2 > 0,
        "every turn drafted: {offered_a1}, {offered_b}, {offered_a2}"
    );
    assert_eq!(starts, [(seq_a, true), (seq_b, true), (seq_a, true)]);
    assert!(taps, "no layer is tapped after the turns");

    let third = concat(&[&h1, &prompt(5, 29)]);
    let mut reference = Driver::new(on_disk(model(), &plain_root));
    let plain3 = reference.turn("a", &third, 6, false);
    let restored = reference
        .state
        .inner
        .telemetry
        .disk_restored
        .load(Ordering::Relaxed);
    assert_eq!(
        (plain3.cached as usize, restored),
        (RESTORED, 1),
        "plain store, plain model: restored from disk"
    );
    drop(reference);
    let want3 = rebuilt(model(), &draft, &third, RESTORED);
    for (name, root, attach) in [
        ("drafted store, drafted model", &drafted_root, true),
        ("plain store, drafted model", &plain_root, true),
        ("drafted store, plain model", &drafted_root, false),
    ] {
        let inner = on_disk(model(), root);
        let inner = if attach {
            with_draft(inner, &draft)
        } else {
            inner
        };
        let mut run = Driver::new(inner);
        run.rule = rule;
        let handle = disk(&run.state.inner).tier_handle();
        let got = if attach {
            run.force(&plain3.turn.tokens);
            run.drafted("a", &third, 6, |inner| {
                assert_same_rows(
                    context(inner),
                    &want3,
                    &format!("{name}: the context at the first proposal"),
                );
            })
        } else {
            run.turn("a", &third, 6, false)
        };
        assert_eq!(got.turn, plain3.turn, "{name}: the plain text");
        let restored = run
            .state
            .inner
            .telemetry
            .disk_restored
            .load(Ordering::Relaxed);
        assert_eq!(
            (got.cached as usize, restored),
            (RESTORED, 1),
            "{name}: restored from disk"
        );
        assert_eq!(
            run.state.inner.telemetry.mode(&handle),
            1,
            "{name}: persistent"
        );
        if attach {
            let seq = run.seq("a");
            let records = run.slot();
            assert_eq!(records.starts, [(seq, true)], "{name}: drafted");
            assert!(!records.offered.is_empty(), "{name}: proposed");
            assert_eq!(
                !records.chosen.is_empty(),
                rule,
                "{name}: the rows rule ran"
            );
            assert_eq!(run.state.inner.decoder.hidden_tap_layers(), None);
        }
    }
}

/// The app's regenerate (the same conversation again: the history goes back
/// to a stored state) and edit (the last message changed: the history
/// leaves the row inside its last turn) on a sequence with the draft
/// attached. Each restores the state turn 1 published, rebuilds the context
/// from its own prefill's taps, drafts, and gives the plain model's text and
/// tokens for the same turns.
#[test]
fn a_regenerate_and_an_edit_draft_on_a_rebuilt_context() {
    let _host = memory::scoped_fixture_host();
    const MAX_NEW: i32 = 12;
    let model = || chat_model(reuse_over(direct_inner()));
    let c = model().decoder.config.clone();
    let draft = Arc::new(moe_draft(c.hidden_size, c.vocab_size, c.num_hidden_layers));
    let (first, _) = full_length(model, 66, MAX_NEW, 9);

    let mut plain = Driver::new(model());
    let a1 = plain.turn("a", &first, MAX_NEW, false);
    let h1 = a1.turn.tokens.clone();
    let second = concat(&[&h1, &prompt(5, 31)]);
    let a2 = plain.turn("a", &second, MAX_NEW, false);
    let again = plain.turn("a", &second, MAX_NEW, false);
    let edited = concat(&[&h1, &prompt(5, 37)]);
    let edit = plain.turn("a", &edited, MAX_NEW, false);
    assert_eq!(
        [
            a2.cached as usize,
            again.cached as usize,
            edit.cached as usize
        ],
        [h1.len(), RESTORED, RESTORED],
        "plain: a2 resumes, the regenerate and the edit restore"
    );
    drop(plain);

    let mut run = Driver::new(with_draft(model(), &draft));
    run.force(&a1.turn.tokens);
    let got_a1 = run.turn("a", &first, MAX_NEW, true);
    assert_eq!(got_a1.turn, a1.turn, "a1");
    run.force(&a2.turn.tokens);
    let got_a2 = run.turn("a", &second, MAX_NEW, true);
    assert_eq!(got_a2.turn, a2.turn, "a2");
    let seq = run.seq("a");
    for (name, tokens, plain_turn) in [
        ("the regenerate", &second, &again),
        ("the edit", &edited, &edit),
    ] {
        let want = rebuilt(model(), &draft, tokens, RESTORED);
        let ks = run.force(&plain_turn.turn.tokens);
        let got = run.drafted("a", tokens, MAX_NEW, |inner| {
            assert_same_rows(
                context(inner),
                &want,
                &format!("{name}: the context at its first proposal"),
            );
        });
        assert_eq!(got.turn, plain_turn.turn, "{name}");
        assert_eq!(got.cached as usize, RESTORED, "{name}: restored");
        assert_eq!(run.seq("a"), seq, "{name}: the same sequence");
        let checked = assert_forced_accepts(run.slot(), seq, &ks.borrow(), name);
        assert!(checked >= 1, "{name}: {checked} drafted cycles checked");
        assert_eq!(
            context(&run.state.inner).token_history(),
            got.turn.tokens.as_slice(),
            "{name}: the context ends on the turn's rows"
        );
        assert_eq!(run.state.inner.decoder.hidden_tap_layers(), None);
    }
    assert_eq!(run.slot().starts, [(seq, true); 4]);
}
