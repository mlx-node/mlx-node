use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::ops::Range;
use std::path::PathBuf;

use napi::{Error, Result};
use serde_json::Value;

use super::ladder::{Kind, Ladder, RAM_STATES, Snapshot};
use super::model::{Logits, Want};
use super::prefill::{
    BLOCK_ROWS, Driver, FAR_FROM, Grid, NEAR_EVERY, Plan, Prefilled, Runner, Seq, StateRow,
    checkpoints, common_prefix, index_keeps, is_checkpoint, turn_row, window_keeps,
};

fn fixture(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hy-v4-flash")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

fn u(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(u).collect()
}

const FILL: u32 = 7;
/// Rows a fake sequence can drop by `truncate` (the window slack).
const SLACK: usize = 3;
const LAYERS: usize = 4;

#[derive(Clone, Debug)]
struct Snap {
    len: usize,
    stamp: Option<u64>,
}

impl Snapshot for Snap {
    fn len(&self) -> usize {
        self.len
    }
}

/// A sequence that can drop at most `SLACK` rows by truncation, and whose
/// snapshots are refused once their last row was rewritten (or when listed
/// in `refuse`).
struct Fake {
    tokens: Vec<u32>,
    written: Vec<u64>,
    calls: u64,
    cap: usize,
    refuse: HashSet<usize>,
}

impl Fake {
    fn new(cap: usize) -> Self {
        Self {
            tokens: Vec::new(),
            written: Vec::new(),
            calls: 0,
            cap,
            refuse: HashSet::new(),
        }
    }

    fn push(&mut self, ids: &[u32]) {
        self.calls += 1;
        self.tokens.extend_from_slice(ids);
        self.written
            .extend(std::iter::repeat_n(self.calls, ids.len()));
    }
}

fn fake_keeps(now: usize, len: usize) -> bool {
    len <= now && (len == 0 || now - len <= SLACK)
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
        if !fake_keeps(self.tokens.len(), len) {
            return Err(Error::from_reason(format!(
                "truncate {} -> {len}",
                self.tokens.len()
            )));
        }
        self.tokens.truncate(len);
        self.written.truncate(len);
        Ok(())
    }

    fn snap(&self) -> Snap {
        Snap {
            len: self.tokens.len(),
            stamp: self.written.last().copied(),
        }
    }

    fn restore(&mut self, s: &Snap) -> Result<()> {
        let last = s
            .len
            .checked_sub(1)
            .and_then(|i| self.written.get(i).copied());
        if s.len > self.tokens.len() || last != s.stamp || self.refuse.contains(&s.len) {
            return Err(Error::from_reason(format!("restore to {}", s.len)));
        }
        self.tokens.truncate(s.len);
        self.written.truncate(s.len);
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Call {
    rows: Range<usize>,
    want: Want,
}

/// Runs a chunk as `LAYERS` layers; trips the stop flag at `trip` (call,
/// layer) and fails without cancel at call `fail`.
#[derive(Default)]
struct FakeRunner {
    calls: RefCell<Vec<Call>>,
    stop: Cell<bool>,
    trip: Option<(usize, usize)>,
    fail: Option<usize>,
}

impl FakeRunner {
    fn rows(&self) -> Vec<Range<usize>> {
        self.calls.borrow().iter().map(|c| c.rows.clone()).collect()
    }
}

impl Runner<Fake> for FakeRunner {
    type Out = Call;

    fn can_truncate(&self, seq: &Fake, len: usize) -> bool {
        fake_keeps(seq.tokens.len(), len)
    }

    fn run(
        &self,
        seq: &mut Fake,
        ids: &[u32],
        want: Want,
        cancel: &dyn Fn(usize) -> bool,
    ) -> Result<Call> {
        let i = self.calls.borrow().len();
        let start = seq.tokens.len();
        let call = Call {
            rows: start..start + ids.len(),
            want,
        };
        self.calls.borrow_mut().push(call.clone());
        for l in 0..LAYERS {
            if self.trip == Some((i, l)) {
                self.stop.set(true);
            }
            if cancel(l) {
                return Err(Error::from_reason(format!("cancelled before layer {l}")));
            }
        }
        if self.fail == Some(i) {
            return Err(Error::from_reason("chunk failed"));
        }
        seq.push(ids);
        Ok(call)
    }
}

const CAP: usize = 262_144;
/// `<hy_start>assistant` of the template's generation prompt.
const OPEN: [u32; 3] = [120_000, 611, 10_372];
const GEN: [u32; 5] = [120_000, 611, 10_372, 120_001, 120_029];

/// `len` tokens with the generation prompt at the end.
fn prompt_of(len: usize) -> Vec<u32> {
    let mut p: Vec<u32> = (0..len).map(|i| FILL + (i % 97) as u32).collect();
    p[len - GEN.len()..].copy_from_slice(&GEN);
    p
}

fn grid(chunk: usize) -> Grid {
    Grid::new(chunk, OPEN.to_vec()).unwrap()
}

struct Run {
    out: Result<Prefilled<Call>>,
    chunks: Vec<Range<usize>>,
}

fn prefill(
    runner: &FakeRunner,
    g: &Grid,
    seq: &mut Fake,
    ladder: &mut Ladder<Snap>,
    plan: Option<Plan>,
    tokens: &[u32],
) -> Run {
    let d = Driver {
        runner,
        grid: g,
        hidden: false,
    };
    let mut chunks = Vec::new();
    let cancel = |_: usize| runner.stop.get();
    let out = d.prefill(
        seq,
        ladder,
        plan,
        tokens,
        &cancel,
        &mut |r: Range<usize>, _: &Call| {
            chunks.push(r);
            Ok(())
        },
    );
    Run { out, chunks }
}

fn plan(runner: &FakeRunner, g: &Grid, seq: &Fake, ladder: &Ladder<Snap>, p: &[u32]) -> Plan {
    Driver {
        runner,
        grid: g,
        hidden: false,
    }
    .plan(seq, ladder, p, true)
}

fn done_from(out: &Result<Prefilled<Call>>) -> usize {
    match out {
        Ok(Prefilled::Done { from, .. }) => *from,
        Ok(Prefilled::Cancelled { at }) => panic!("cancelled at {at}"),
        Err(e) => panic!("{}", e.reason),
    }
}

/// Grid ends and state rows of every block-rows.json sample; T is the
/// block row below the turn's KV end, not repeated when it is a state row.
#[test]
fn state_rows_match_the_block_rows_fixture() {
    let f = fixture("block-rows.json");
    let gen_ids: Vec<u32> = usizes(&f["gen_prompt"]["ids"])
        .into_iter()
        .map(|v| v as u32)
        .collect();
    assert_eq!(gen_ids, GEN);
    for s in f["samples"].as_array().unwrap() {
        let name = s["name"].as_str().unwrap();
        let (len, start, chunk) = (u(&s["prompt_len"]), u(&s["start"]), u(&s["chunk"]));
        assert_eq!(u(&s["gen_prompt_start"]), len - GEN.len(), "{name}");
        let p = prompt_of(len);
        let g = grid(chunk);
        assert_eq!(
            g.ends(start, &p),
            usizes(&s["prefill_ends"]),
            "{name}: ends"
        );
        let mut want = Vec::new();
        let mut t = None;
        for st in s["states"].as_array().unwrap() {
            let row = u(&st["row"]);
            assert_eq!(
                row.is_multiple_of(chunk),
                st["on_chunk_grid"].as_bool().unwrap(),
                "{name}: grid flag of {row}"
            );
            match st["kind"].as_str().unwrap() {
                "checkpoint" => want.push(StateRow {
                    row,
                    kind: Kind::Checkpoint,
                }),
                "P" => want.push(StateRow {
                    row,
                    kind: Kind::Prompt,
                }),
                "T" => t = Some(row),
                k => panic!("{name}: kind {k}"),
            }
        }
        let got = g.states(start, &p);
        assert_eq!(got, want, "{name}: states");
        assert!(got.iter().all(|s| s.row.is_multiple_of(BLOCK_ROWS)));
        let tr = turn_row(u(&s["kv_end"]));
        let fresh = !got.iter().any(|s| s.row == tr);
        assert_eq!(t, fresh.then_some(tr), "{name}: T");
    }
}

/// The cross-owner hits of block-rows.json: the deepest state row no longer
/// than the shared prefix, through `plan` on a sequence holding owner A.
#[test]
fn cross_owner_hits_take_the_deepest_state_row() {
    let f = fixture("block-rows.json");
    let g = grid(8192);
    for c in f["cross_owner"].as_array().unwrap() {
        let shared = u(&c["shared_prefix"]);
        let rows = usizes(&c["owner_a_state_rows"]);
        let a = prompt_of(60_000);
        let mut seq = Fake::new(CAP);
        let mut ladder = Ladder::new();
        let mut at = 0;
        for &r in &rows {
            seq.push(&a[at..r]);
            ladder.put(Kind::Checkpoint, seq.snap());
            at = r;
        }
        seq.push(&a[at..]);
        let mut b = a[..shared].to_vec();
        b.extend((0..100).map(|i| 50_000 + i));
        let runner = FakeRunner::default();
        let pl = plan(&runner, &g, &seq, &ladder, &b);
        assert_eq!(pl.reuse(), u(&c["hit_row"]), "shared {shared}");
        let tail = b[pl.reuse()..].to_vec();
        let run = prefill(&runner, &g, &mut seq, &mut ladder, Some(pl), &tail);
        assert_eq!(done_from(&run.out), u(&c["hit_row"]), "shared {shared}");
        assert_eq!(seq.tokens, b, "shared {shared}");
        assert_eq!(runner.rows()[0].start, u(&c["hit_row"]));
    }
}

/// Every state row is a whole block, so no `tok_k` row is open there and
/// the k-pool bookkeeping of kpool-splits.json closes `row / kpool` groups.
#[test]
fn state_rows_close_whole_kpool_groups() {
    let f = fixture("kpool-splits.json");
    let kpool = u(&f["geometry"]["kpool"]);
    let disk = u(&f["geometry"]["disk_block_rows"]);
    assert!(BLOCK_ROWS.is_multiple_of(kpool));
    assert!(NEAR_EVERY.is_multiple_of(disk) && FAR_FROM.is_multiple_of(NEAR_EVERY));
    let mut aligned = 0;
    for (len, st) in f["state_after_len"].as_object().unwrap() {
        let len: usize = len.parse().unwrap();
        if !len.is_multiple_of(BLOCK_ROWS) {
            continue;
        }
        aligned += 1;
        assert!(st["tail"].as_array().unwrap().is_empty(), "len {len}");
        assert_eq!(u(&st["closed"]), len / kpool, "len {len}");
        assert_eq!(u(&st["disk_blocks_closed"]), len / disk, "len {len}");
    }
    assert!(aligned > 0);
}

#[test]
fn checkpoint_rows() {
    assert!(!is_checkpoint(0));
    for r in [8192, 16_384, 24_576, 32_768, 65_536, 229_376] {
        assert!(is_checkpoint(r), "{r}");
    }
    for r in [4096, 40_960, 49_152, 57_344, 73_728] {
        assert!(!is_checkpoint(r), "{r}");
    }
    assert_eq!(
        checkpoints(0, 100_000),
        vec![8192, 16_384, 24_576, 32_768, 65_536, 98_304]
    );
    assert_eq!(checkpoints(8192, 16_384), Vec::<usize>::new());
    assert_eq!(checkpoints(8191, 16_385), vec![8192, 16_384]);
    assert_eq!(turn_row(4097), 4096);
    assert_eq!(turn_row(31), 0);
}

/// Ends are absolute: a resume at an odd row runs to the next grid line,
/// every gap is at most C, every end but the last is a block row, and the
/// ends past any later row are the ends a resume from there makes.
#[test]
fn grid_ends_are_absolute() {
    for chunk in [256, 768, 2048, 4096, 8192] {
        let g = grid(chunk);
        for len in [33, 900, 8197, 20_000, 70_001] {
            let p = prompt_of(len);
            let cold = g.ends(0, &p);
            for start in [0, 1, 1000, 8191, 8192, len / 2, len - 1] {
                if start >= len {
                    continue;
                }
                let ends = g.ends(start, &p);
                let what = format!("C {chunk} len {len} start {start}");
                assert_eq!(ends.last(), Some(&len), "{what}");
                assert!(ends.windows(2).all(|w| w[0] < w[1]), "{what}");
                let mut at = start;
                for &e in &ends {
                    assert!(e > at && e - at <= chunk, "{what}: {at} -> {e}");
                    at = e;
                }
                assert!(
                    ends[..ends.len() - 1]
                        .iter()
                        .all(|e| e.is_multiple_of(BLOCK_ROWS)),
                    "{what}"
                );
                let mut m = (start / chunk + 1) * chunk;
                while m < len {
                    assert!(ends.contains(&m), "{what}: grid line {m}");
                    m += chunk;
                }
                let tail: Vec<usize> = cold.iter().copied().filter(|&e| e > start).collect();
                assert_eq!(ends, tail, "{what}: resume");
            }
        }
    }
}

#[test]
fn grid_rules() {
    assert!(Grid::new(0, OPEN.to_vec()).is_err());
    assert!(Grid::new(100, OPEN.to_vec()).is_err());
    let g = grid(8192);
    let p = prompt_of(900);
    assert_eq!(g.replay_row(&p), 864);
    assert_eq!(g.ends(900, &p), Vec::<usize>::new());
    assert_eq!(g.states(864, &p), Vec::new());
    let raw = Grid::new(8192, Vec::new()).unwrap();
    let plain: Vec<u32> = vec![FILL; 1000];
    assert_eq!(raw.replay_row(&plain), 992);
    assert_eq!(raw.ends(0, &plain), vec![992, 1000]);
    let mut two = prompt_of(5000);
    two[100..103].copy_from_slice(&OPEN);
    assert_eq!(g.replay_row(&two), 4992);
    let mut first_only: Vec<u32> = vec![FILL; 50];
    first_only[..3].copy_from_slice(&OPEN);
    assert_eq!(g.replay_row(&first_only), 32);
}

/// The truncate rules of `WindowRing` and `IndexerState`.
#[test]
fn truncate_rules() {
    let (w, keep) = (512, 515);
    assert!(window_keeps(1000, keep, w, 1000));
    assert!(!window_keeps(1000, keep, w, 1001));
    assert!(window_keeps(1000, keep, w, 997));
    assert!(!window_keeps(1000, keep, w, 996));
    assert!(window_keeps(1000, keep, w, 0));
    assert!(window_keeps(100, 100, w, 1));
    assert!(window_keeps(515, 515, w, 3));
    assert!(window_keeps(997, 512, w, 997));
    assert!(!window_keeps(997, 512, w, 996));

    assert!(index_keeps(1000, 6, 4, 996));
    assert!(index_keeps(1000, 6, 4, 997));
    assert!(index_keeps(1000, 6, 4, 4));
    assert!(!index_keeps(1000, 1, 4, 997));
    assert!(index_keeps(1000, 1, 4, 996));
    assert!(!index_keeps(1000, 6, 0, 996));
    assert!(!index_keeps(10, 6, 4, 11));
}

#[test]
fn ladder_keeps_the_newest_and_drops_on_cuts() {
    let s = |len| Snap { len, stamp: None };
    let mut l: Ladder<Snap> = Ladder::new();
    for len in [300, 100, 200] {
        l.put(Kind::Checkpoint, s(len));
    }
    assert_eq!(l.lens(), vec![100, 200, 300]);
    l.put(Kind::Turn, s(200));
    assert_eq!(l.len(), 3);
    assert_eq!(l.kind(200), Some(Kind::Turn));
    assert_eq!(l.deepest(250, 1000), Some(200));
    assert_eq!(l.deepest(1000, 250), Some(200));
    assert_eq!(l.deepest(99, 1000), None);
    l.drop_above(200);
    assert_eq!(l.lens(), vec![100, 200]);
    l.remove(100);
    assert_eq!(l.lens(), vec![200]);
    l.clear();
    assert!(l.is_empty());

    for len in 1..=RAM_STATES + 3 {
        l.put(Kind::Checkpoint, s(len * 10));
    }
    assert_eq!(l.len(), RAM_STATES);
    assert_eq!(l.lens().first(), Some(&40));
    l.put(Kind::Prompt, s(5));
    assert_eq!(l.len(), RAM_STATES);
    assert!(l.contains(5) && !l.contains(40));
    assert!(s(0).is_empty());
}

/// A cold prefill cuts on the grid and at the state rows, asks logits of
/// the last chunk only and snapshots every state row.
#[test]
fn cold_prefill_cuts_on_the_grid() {
    let runner = FakeRunner::default();
    let g = grid(8192);
    let p = prompt_of(20_000);
    let mut seq = Fake::new(CAP);
    let mut ladder = Ladder::new();
    let pl = plan(&runner, &g, &seq, &ladder, &p);
    assert_eq!(pl.reuse(), 0);
    let run = prefill(&runner, &g, &mut seq, &mut ladder, Some(pl), &p);
    assert_eq!(done_from(&run.out), 0);
    let want = vec![0..8192, 8192..16_384, 16_384..19_968, 19_968..20_000];
    assert_eq!(runner.rows(), want);
    assert_eq!(run.chunks, want);
    let logits: Vec<Logits> = runner
        .calls
        .borrow()
        .iter()
        .map(|c| c.want.logits)
        .collect();
    assert_eq!(
        logits,
        vec![Logits::None, Logits::None, Logits::None, Logits::Last]
    );
    assert!(runner.calls.borrow().iter().all(|c| !c.want.hidden));
    assert_eq!(ladder.lens(), vec![8192, 16_384, 19_968]);
    assert_eq!(ladder.kind(16_384), Some(Kind::Checkpoint));
    assert_eq!(ladder.kind(19_968), Some(Kind::Prompt));
    assert_eq!(seq.tokens, p);
}

#[test]
fn hidden_rows_are_asked_of_every_chunk() {
    let runner = FakeRunner::default();
    let g = grid(2048);
    let p = prompt_of(5000);
    let mut seq = Fake::new(CAP);
    let mut ladder = Ladder::new();
    let d = Driver {
        runner: &runner,
        grid: &g,
        hidden: true,
    };
    let out = d.prefill(
        &mut seq,
        &mut ladder,
        None,
        &p,
        &|_: usize| false,
        &mut |_: Range<usize>, _: &Call| Ok(()),
    );
    assert_eq!(done_from(&out), 0);
    assert!(runner.calls.borrow().iter().all(|c| c.want.hidden));
    assert_eq!(runner.rows().len(), 4);
}

/// A cancel inside a chunk keeps the chunks before it and marks the point;
/// the same prompt again resumes there with the cold run's cuts.
#[test]
fn cancel_keeps_finished_chunks() {
    let g = grid(8192);
    let p = prompt_of(20_000);
    let mut seq = Fake::new(CAP);
    let mut ladder = Ladder::new();
    let runner = FakeRunner {
        trip: Some((2, 1)),
        ..FakeRunner::default()
    };
    let run = prefill(&runner, &g, &mut seq, &mut ladder, None, &p);
    assert!(matches!(run.out, Ok(Prefilled::Cancelled { at: 16_384 })));
    assert_eq!(seq.tokens, p[..16_384]);
    assert_eq!(run.chunks, vec![0..8192, 8192..16_384]);
    assert_eq!(ladder.kind(16_384), Some(Kind::Checkpoint));

    let again = FakeRunner::default();
    let pl = plan(&again, &g, &seq, &ladder, &p);
    assert_eq!(pl.reuse(), 16_384);
    let tail = p[16_384..].to_vec();
    let run = prefill(&again, &g, &mut seq, &mut ladder, Some(pl), &tail);
    assert_eq!(done_from(&run.out), 16_384);
    assert_eq!(again.rows(), vec![16_384..19_968, 19_968..20_000]);
    assert_eq!(seq.tokens, p);
}

#[test]
fn cancel_off_a_state_row_leaves_a_cancel_entry() {
    let g = grid(2048);
    let p = prompt_of(5000);
    let mut seq = Fake::new(CAP);
    let mut ladder = Ladder::new();
    let runner = FakeRunner {
        trip: Some((1, 3)),
        ..FakeRunner::default()
    };
    let run = prefill(&runner, &g, &mut seq, &mut ladder, None, &p);
    assert!(matches!(run.out, Ok(Prefilled::Cancelled { at: 2048 })));
    assert_eq!(ladder.kind(2048), Some(Kind::Cancel));

    let mut seq = Fake::new(CAP);
    let mut ladder = Ladder::new();
    let first = FakeRunner {
        trip: Some((0, 0)),
        ..FakeRunner::default()
    };
    let run = prefill(&first, &g, &mut seq, &mut ladder, None, &p);
    assert!(matches!(run.out, Ok(Prefilled::Cancelled { at: 0 })));
    assert!(ladder.is_empty() && seq.tokens.is_empty());
}

#[test]
fn a_failed_chunk_is_an_error_and_keeps_the_chunks_before() {
    let g = grid(8192);
    let p = prompt_of(20_000);
    let mut seq = Fake::new(CAP);
    let mut ladder = Ladder::new();
    let runner = FakeRunner {
        fail: Some(1),
        ..FakeRunner::default()
    };
    let run = prefill(&runner, &g, &mut seq, &mut ladder, None, &p);
    assert!(run.out.is_err());
    assert_eq!(seq.tokens, p[..8192]);
}

/// Regenerating a reply goes back to the replay row; the same prompt as the
/// whole sequence re-runs only its last row.
#[test]
fn reuse_points() {
    let g = grid(8192);
    let p = prompt_of(20_000);
    let mut seq = Fake::new(CAP);
    let mut ladder = Ladder::new();
    let runner = FakeRunner::default();
    prefill(&runner, &g, &mut seq, &mut ladder, None, &p)
        .out
        .unwrap();
    seq.push(&[1, 2, 3, 4, 5, 6]);
    ladder.put(Kind::Turn, seq.snap());

    let again = FakeRunner::default();
    let pl = plan(&again, &g, &seq, &ladder, &p);
    assert_eq!(pl.reuse(), 19_968);
    let tail = p[19_968..].to_vec();
    let run = prefill(&again, &g, &mut seq, &mut ladder, Some(pl), &tail);
    assert_eq!(done_from(&run.out), 19_968);
    assert_eq!(again.rows(), vec![19_968..20_000]);
    assert_eq!(ladder.lens(), vec![8192, 16_384, 19_968]);
    assert_eq!(seq.tokens, p);

    let same = FakeRunner::default();
    let pl = plan(&same, &g, &seq, &ladder, &p);
    assert_eq!(pl.reuse(), 19_999);
    let run = prefill(&same, &g, &mut seq, &mut ladder, Some(pl), &p[19_999..]);
    assert_eq!(done_from(&run.out), 19_999);
    assert_eq!(same.rows(), vec![19_999..20_000]);

    let mut longer = p.clone();
    longer.extend([9, 9, 9]);
    let ext = FakeRunner::default();
    let pl = plan(&ext, &g, &seq, &ladder, &longer);
    assert_eq!(pl.reuse(), 20_000);
    let off = Driver {
        runner: &ext,
        grid: &g,
        hidden: false,
    }
    .plan(&seq, &ladder, &longer, false);
    assert_eq!(off.reuse(), 0);
}

/// A promised point refused at prefill falls to a deeper one and runs the
/// rows between.
#[test]
fn a_refused_point_recomputes_from_a_deeper_one() {
    let g = grid(8192);
    let p = prompt_of(20_000);
    let mut seq = Fake::new(CAP);
    let mut ladder = Ladder::new();
    prefill(&FakeRunner::default(), &g, &mut seq, &mut ladder, None, &p)
        .out
        .unwrap();
    seq.push(&[1, 2, 3, 4, 5, 6]);
    let runner = FakeRunner::default();
    let pl = plan(&runner, &g, &seq, &ladder, &p);
    assert_eq!(pl.reuse(), 19_968);
    seq.refuse.insert(19_968);
    let tail = p[19_968..].to_vec();
    let run = prefill(&runner, &g, &mut seq, &mut ladder, Some(pl), &tail);
    assert_eq!(done_from(&run.out), 16_384);
    assert_eq!(runner.rows(), vec![16_384..19_968, 19_968..20_000]);
    assert_eq!(seq.tokens, p);

    seq.refuse.extend([8192, 16_384]);
    seq.push(&[1, 2, 3, 4, 5, 6]);
    let all = FakeRunner::default();
    let pl = plan(&all, &g, &seq, &ladder, &p);
    let run = prefill(&all, &g, &mut seq, &mut ladder, Some(pl), &tail);
    assert_eq!(done_from(&run.out), 0);
    assert_eq!(all.rows().first(), Some(&(0..8192)));
    assert_eq!(seq.tokens, p);
}

/// A plan that does not end in the prefill input is ignored: the input
/// extends the sequence.
#[test]
fn a_stale_plan_extends_the_sequence() {
    let g = grid(8192);
    let p = prompt_of(1000);
    let mut seq = Fake::new(CAP);
    let mut ladder = Ladder::new();
    seq.push(&p[..100]);
    let runner = FakeRunner::default();
    let other = plan(&runner, &g, &seq, &ladder, &prompt_of(3000));
    let run = prefill(&runner, &g, &mut seq, &mut ladder, Some(other), &p[100..]);
    assert_eq!(done_from(&run.out), 100);
    assert_eq!(runner.rows(), vec![100..992, 992..1000]);
    assert_eq!(seq.tokens, p);
}

#[test]
fn refused_inputs_leave_the_sequence() {
    let g = grid(8192);
    let mut seq = Fake::new(64);
    let mut ladder = Ladder::new();
    seq.push(&[1, 2, 3]);
    let runner = FakeRunner::default();
    assert!(
        prefill(&runner, &g, &mut seq, &mut ladder, None, &[])
            .out
            .is_err()
    );
    let long = prompt_of(100);
    assert!(
        prefill(&runner, &g, &mut seq, &mut ladder, None, &long)
            .out
            .is_err()
    );
    assert_eq!(seq.tokens, vec![1, 2, 3]);
    assert!(runner.rows().is_empty());
    assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 4]), 2);
}
