//! Chat backend. Fakes (no Metal): the sampling defaults and the MTP depth
//! clamp, the eos check, and the conversation state across turns: reuse of
//! the sequence and the ladder, a cancelled prefill that keeps its chunks
//! through the fail-closed reset, resets and saves. Real (`HY4_DIR`): the
//! GGUF markers, one engine turn with the reasoning split and the eos stop,
//! and a cancelled prefill that leaves the sequence at its history.

use std::cell::{Cell, RefCell};
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use napi::{Error, Result};

use super::chat::{
    Conv, EOS, HY_START, MTP_DEPTH_MAX, PREFILL_CANCELLED, TEMPERATURE, TOP_P, assistant_open,
    check_eos, generation_defaults, resolve,
};
use super::ladder::{Kind, Snapshot};
use super::model::Want;
use super::prefill::{Grid, Runner, Seq};
use crate::engine::backend::{ChatBackend, ResetScope};
use crate::engine::types::ChatConfig;
use crate::stream::Stream;

#[derive(Clone, Debug)]
struct Snap {
    len: usize,
    last: Option<u64>,
}

impl Snapshot for Snap {
    fn len(&self) -> usize {
        self.len
    }
}

/// Tokens plus the call that wrote each position (a rewritten row refuses
/// an older snapshot, as `Session::restore` does).
struct Fake {
    tokens: Vec<u32>,
    written: Vec<u64>,
    calls: u64,
    cap: usize,
}

impl Fake {
    fn new(cap: usize) -> Self {
        Self {
            tokens: Vec::new(),
            written: Vec::new(),
            calls: 0,
            cap,
        }
    }

    fn push(&mut self, ids: &[u32]) {
        self.calls += 1;
        self.tokens.extend_from_slice(ids);
        self.written
            .extend(std::iter::repeat_n(self.calls, ids.len()));
    }
}

impl Seq for Fake {
    type Snap = Snap;

    fn len(&self) -> usize {
        self.tokens.len()
    }

    fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    fn cap(&self) -> usize {
        self.cap
    }

    fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.tokens.len() {
            return Err(Error::from_reason("truncate past the end"));
        }
        self.tokens.truncate(len);
        self.written.truncate(len);
        Ok(())
    }

    fn snap(&self) -> Snap {
        Snap {
            len: self.tokens.len(),
            last: self.written.last().copied(),
        }
    }

    fn restore(&mut self, s: &Snap) -> Result<()> {
        let last = s
            .len
            .checked_sub(1)
            .and_then(|i| self.written.get(i).copied());
        if s.len > self.tokens.len() || last != s.last {
            return Err(Error::from_reason("rewritten since"));
        }
        self.truncate(s.len)
    }
}

/// Records the rows of every chunk; `stop_after` chunks run before the
/// cancel fires, `fail` errs without a cancel.
#[derive(Default)]
struct FakeRunner {
    rows: RefCell<Vec<Range<usize>>>,
    stop_after: Cell<Option<usize>>,
    fail: Cell<bool>,
}

impl FakeRunner {
    fn rows(&self) -> Vec<Range<usize>> {
        self.rows.borrow().clone()
    }

    fn clear(&self) {
        self.rows.borrow_mut().clear();
    }
}

impl Runner<Fake> for FakeRunner {
    type Out = usize;

    fn can_truncate(&self, seq: &Fake, len: usize) -> bool {
        len <= seq.len()
    }

    fn run(
        &self,
        seq: &mut Fake,
        ids: &[u32],
        _want: Want,
        cancel: &dyn Fn(usize) -> bool,
    ) -> Result<usize> {
        if self.fail.get() {
            return Err(Error::from_reason("forward failed"));
        }
        if cancel(0) {
            return Err(Error::from_reason("cancelled at layer 0"));
        }
        let start = seq.len();
        seq.push(ids);
        self.rows.borrow_mut().push(start..seq.len());
        Ok(seq.len())
    }
}

const CAP: usize = 4096;

fn grid() -> Grid {
    Grid::new(32, Vec::new()).unwrap()
}

fn prompt_of(n: usize, salt: u32) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 7 + salt) % 1000 + 1).collect()
}

/// `prefill` on `conv` with a cancel that fires once `runner.stop_after`
/// chunks ran; the chunk rows it reported.
fn prefill(
    conv: &mut Conv<Fake>,
    runner: &FakeRunner,
    tokens: &[u32],
) -> (Result<usize>, Vec<Range<usize>>) {
    let cancel = |_: usize| {
        runner
            .stop_after
            .get()
            .is_some_and(|n| runner.rows.borrow().len() >= n)
    };
    let mut seen = Vec::new();
    let out = conv.prefill(runner, &grid(), tokens, &cancel, &mut |r| seen.push(r));
    (out, seen)
}

/// The engine's flat turn: verify, prefill the tail past the hit (the whole
/// prompt after a miss reset), then the generated rows and the save.
fn turn(conv: &mut Conv<Fake>, runner: &FakeRunner, prompt: &[u32], generated: &[u32]) -> usize {
    let hit = conv.verify(runner, &grid(), prompt, true);
    let tail = if hit > 0 && hit < prompt.len() {
        &prompt[hit..]
    } else {
        conv.reset(ResetScope::PrefixMiss).unwrap();
        prompt
    };
    prefill(conv, runner, tail).0.unwrap();
    conv.seq_mut().push(generated);
    conv.save(true).unwrap();
    hit
}

#[test]
fn sampling_defaults_and_mtp_depth_clamp() {
    let d = generation_defaults();
    let p = resolve(&ChatConfig::default(), &d);
    let s = p.sampling_config.unwrap();
    assert_eq!(s.temperature, Some(TEMPERATURE));
    assert_eq!(s.top_p, Some(TOP_P));
    assert_eq!(s.top_k, None);
    assert_eq!(p.mtp_depth, 1, "depth default 1");
    assert!(!p.mtp_adaptive_depth);
    for (asked, got) in [
        (-2, 1),
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 3),
        (5, 3),
        (9, 3),
    ] {
        let cfg = ChatConfig {
            mtp_depth: Some(asked),
            mtp_adaptive_depth: Some(true),
            ..Default::default()
        };
        let p = resolve(&cfg, &d);
        assert_eq!(p.mtp_depth, got, "mtpDepth {asked}");
        assert!(p.mtp_depth <= MTP_DEPTH_MAX);
        assert!(!p.mtp_adaptive_depth, "adaptive stays off");
    }
    let greedy = ChatConfig {
        temperature: Some(0.0),
        top_p: Some(0.5),
        ..Default::default()
    };
    let s = resolve(&greedy, &d).sampling_config.unwrap();
    assert_eq!(
        (s.temperature, s.top_p),
        (Some(0.0), Some(0.5)),
        "the request wins"
    );
}

#[test]
fn eos_is_hy_end_only() {
    assert_eq!(check_eos(120_025).unwrap(), EOS);
    for other in [120_008, 120_026, 2, 0] {
        assert!(check_eos(other).is_err(), "{other}");
    }
}

/// A follow-up extends the sequence; a regenerated reply goes back to the
/// prompt's last row; the saved history is the sequence.
#[test]
fn turns_reuse_the_sequence() {
    let runner = FakeRunner::default();
    let mut conv = Conv::new(Fake::new(CAP));
    let p1 = prompt_of(40, 3);
    assert_eq!(turn(&mut conv, &runner, &p1, &[5, 6]), 0);
    assert_eq!(runner.rows(), vec![0..32, 32..40]);
    assert_eq!(conv.history(), conv.seq().tokens());
    assert_eq!(conv.history().len(), 42);
    assert_eq!(conv.ladder().kind(42), Some(Kind::Turn));

    runner.clear();
    let mut p2 = conv.history().to_vec();
    p2.extend([7, 8, 9]);
    assert_eq!(turn(&mut conv, &runner, &p2, &[10]), 42);
    assert_eq!(runner.rows(), vec![42..45]);
    assert_eq!(conv.history().len(), 46);

    runner.clear();
    assert_eq!(turn(&mut conv, &runner, &p1, &[11]), 39, "regenerate");
    assert_eq!(runner.rows(), vec![39..40]);
    assert_eq!(&conv.seq().tokens()[..40], p1.as_slice());
    assert!(conv.ladder().lens().iter().all(|&l| l <= conv.seq().len()));

    assert_eq!(conv.verify(&runner, &grid(), &p2, false), 0, "reuse off");
}

/// A cancel keeps the finished chunks and a ladder entry; the fail-closed
/// reset of the same turn keeps them; the next turn resumes there.
#[test]
fn cancel_keeps_the_finished_chunks() {
    let runner = FakeRunner::default();
    let mut conv = Conv::new(Fake::new(CAP));
    let p = prompt_of(100, 9);
    runner.stop_after.set(Some(2));
    assert_eq!(conv.verify(&runner, &grid(), &p, true), 0);
    let (out, seen) = prefill(&mut conv, &runner, &p);
    assert_eq!(out.err().unwrap().reason, PREFILL_CANCELLED);
    assert_eq!(seen, vec![0..32, 32..64]);
    assert_eq!(conv.seq().tokens(), &p[..64]);
    assert!(conv.kept());
    assert_eq!(conv.ladder().kind(64), Some(Kind::Cancel));

    conv.reset(ResetScope::Command).unwrap();
    assert_eq!(
        conv.seq().len(),
        64,
        "the fail-closed reset keeps the chunks"
    );
    assert!(conv.history().is_empty());
    assert!(!conv.kept());
    conv.turn_over();

    runner.stop_after.set(None);
    runner.clear();
    assert_eq!(conv.verify(&runner, &grid(), &p, true), 64);
    let (out, _) = prefill(&mut conv, &runner, &p[64..]);
    assert_eq!(out.unwrap(), 100);
    assert_eq!(runner.rows(), vec![64..96, 96..100]);
    assert_eq!(conv.seq().tokens(), p.as_slice());

    runner.stop_after.set(Some(0));
    let mut q = p.clone();
    q.extend(prompt_of(50, 1));
    assert_eq!(conv.verify(&runner, &grid(), &q, true), 100);
    let (out, _) = prefill(&mut conv, &runner, &q[100..]);
    assert_eq!(out.err().unwrap().reason, PREFILL_CANCELLED);
    assert_eq!(conv.seq().len(), 100, "cancelled before its first chunk");
    conv.turn_over();
    conv.reset(ResetScope::Command).unwrap();
    assert!(conv.seq().is_empty(), "a reset after the turn drops it all");
    assert!(conv.ladder().is_empty());
}

#[test]
fn a_failed_prefill_is_not_kept() {
    let runner = FakeRunner::default();
    let mut conv = Conv::new(Fake::new(CAP));
    let p = prompt_of(40, 2);
    turn(&mut conv, &runner, &p, &[1]);
    runner.fail.set(true);
    let mut q = conv.history().to_vec();
    q.extend([2, 3]);
    assert_eq!(conv.verify(&runner, &grid(), &q, true), 41);
    let (out, _) = prefill(&mut conv, &runner, &q[41..]);
    let e = out.err().unwrap();
    assert_ne!(e.reason, PREFILL_CANCELLED);
    assert!(!conv.kept());
    conv.reset(ResetScope::Command).unwrap();
    assert!(conv.seq().is_empty());
    assert!(conv.history().is_empty());
}

/// A miss reset drops only the history; the prefill of the new prompt then
/// starts from row 0 and replaces the sequence.
#[test]
fn prefix_miss_leaves_the_sequence_to_the_prefill() {
    let runner = FakeRunner::default();
    let mut conv = Conv::new(Fake::new(CAP));
    turn(&mut conv, &runner, &prompt_of(40, 1), &[4]);
    let other: Vec<u32> = (2000..2050).collect();
    assert_eq!(conv.verify(&runner, &grid(), &other, true), 0);
    conv.reset(ResetScope::PrefixMiss).unwrap();
    assert!(conv.history().is_empty());
    assert_eq!(conv.seq().len(), 41);
    runner.clear();
    prefill(&mut conv, &runner, &other).0.unwrap();
    assert_eq!(runner.rows(), vec![0..32, 32..50]);
    assert_eq!(conv.seq().tokens(), other.as_slice());
    assert!(conv.ladder().lens().iter().all(|&l| l <= 50));
}

#[test]
fn save_without_reuse_and_command_reset_drop_the_conversation() {
    let runner = FakeRunner::default();
    let mut conv = Conv::new(Fake::new(CAP));
    let p = prompt_of(40, 5);
    conv.verify(&runner, &grid(), &p, false);
    prefill(&mut conv, &runner, &p).0.unwrap();
    conv.save(false).unwrap();
    assert!(conv.seq().is_empty());
    assert!(conv.history().is_empty());
    assert!(conv.ladder().is_empty());

    turn(&mut conv, &runner, &p, &[1, 2]);
    assert!(!conv.history().is_empty());
    conv.reset(ResetScope::Command).unwrap();
    assert!(conv.seq().is_empty());
    assert!(conv.history().is_empty());
    assert!(conv.ladder().is_empty());
    assert_eq!(conv.verify(&runner, &grid(), &p, true), 0);
}

fn hy4_dir() -> PathBuf {
    PathBuf::from(std::env::var("HY4_DIR").expect("HY4_DIR = directory of the two Hy4 shards"))
}

fn chat_msg(role: &str, content: &str) -> crate::tokenizer::ChatMessage {
    serde_json::from_value(serde_json::json!({"role": role, "content": content})).unwrap()
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_gguf_markers() {
    let paths = crate::models::deepseek_v41::chat::shard_paths(&hy4_dir()).unwrap();
    let (tok, eos) = crate::models::deepseek_v41::chat::gguf_tokenizer(&paths[0]).unwrap();
    assert_eq!(check_eos(eos).unwrap(), EOS);
    assert_eq!(assistant_open(&tok).unwrap(), vec![HY_START, 611, 10_372]);
    super::finalize::Hy4Tags::from_tokenizer(&tok).unwrap();
}

/// One greedy engine turn with reasoning and one without: the reasoning is
/// split off, an eos stop ends on `EOS`; then a prefill cancelled before
/// its first chunk leaves the sequence at the saved history and the next
/// turn reuses it.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_chat_turn_split_stop_and_cancel() {
    let mut inner = super::chat::Inner::load(&hy4_dir(), Some(16_384), Default::default()).unwrap();
    assert_eq!(inner.context_token_limit(), Some(inner.store.ctx));
    let flag = Arc::new(AtomicBool::new(false));
    let think = ChatConfig {
        max_new_tokens: Some(256),
        temperature: Some(0.0),
        ..Default::default()
    };
    let msgs = || vec![chat_msg("user", "What is 2 + 3? Answer with one number.")];
    let r = crate::engine::session::session_start(&mut inner, msgs(), think, &flag).unwrap();
    let close = super::finalize::Hy4Tags::from_tokenizer(&inner.tokenizer().unwrap())
        .unwrap()
        .think_close()
        .to_string();
    println!(
        "think turn: {:?} thinking {:?} ({}, {} tokens)",
        r.text, r.thinking, r.finish_reason, r.num_tokens
    );
    assert!(!r.text.contains(&close));
    let (_, out) = inner.turn_ids().unwrap();
    if r.finish_reason == "stop" {
        assert_eq!(out.last(), Some(&EOS));
    }

    inner.reset_caches(ResetScope::Command).unwrap();
    let plain = ChatConfig {
        max_new_tokens: Some(64),
        temperature: Some(0.0),
        reasoning_effort: Some("none".into()),
        ..Default::default()
    };
    let r = crate::engine::session::session_start(&mut inner, msgs(), plain, &flag).unwrap();
    println!("no-think turn: {:?} ({})", r.text, r.finish_reason);
    assert!(r.thinking.is_none());

    let held = inner.conv().seq().tokens().to_vec();
    let mut next = held.clone();
    next.extend(
        inner
            .tokenizer()
            .unwrap()
            .encode_sync("more", Some(false))
            .unwrap(),
    );
    let cancelled = Arc::new(AtomicBool::new(true));
    inner.set_turn_cancel_flag(Some(Arc::clone(&cancelled)));
    let hit = inner.verify_cache_prefix(&next, true);
    assert_eq!(hit, held.len());
    let e = inner
        .prefill(&next[hit..], Stream::generation())
        .err()
        .unwrap();
    assert_eq!(e.reason, PREFILL_CANCELLED);
    inner.reset_caches(ResetScope::Command).unwrap();
    inner.set_turn_cancel_flag(None);
    assert_eq!(inner.conv().seq().tokens(), held.as_slice());
    cancelled.store(false, Ordering::Relaxed);
    assert_eq!(inner.verify_cache_prefix(&next, true), held.len());
}
