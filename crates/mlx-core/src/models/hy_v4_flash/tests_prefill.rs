use std::ops::Range;

use napi::Result;

use super::ladder::{Kind, Ladder, RAM_STATES, Snapshot};
use super::model::Logits;
use super::prefill::{
    BLOCK_ROWS, Driver, FAR_FROM, Grid, NEAR_EVERY, Plan, Prefilled, Seq, StateRow, checkpoints,
    common_prefix, index_keeps, is_checkpoint, turn_row, window_keeps,
};
use super::test_support::{
    FakeCall, FakeRunner, FakeSeq, FakeSnap, fixture, j_usize as u, j_usizes as usizes,
};

const FILL: u32 = 7;
/// Rows a fake sequence can drop by `truncate` (the window slack).
const SLACK: usize = 3;
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
    out: Result<Prefilled<FakeCall>>,
    chunks: Vec<Range<usize>>,
}

struct Env {
    grid: Grid,
    seq: FakeSeq,
    ladder: Ladder<FakeSnap>,
}

impl Env {
    fn new(chunk: usize) -> Self {
        Self::with_cap(chunk, CAP)
    }
    fn with_cap(chunk: usize, cap: usize) -> Self {
        let (grid, seq, ladder) = (grid(chunk), FakeSeq::with_slack(cap, SLACK), Ladder::new());
        Self { grid, seq, ladder }
    }
    fn plan(&self, runner: &FakeRunner, p: &[u32], reuse: bool) -> Plan {
        let d = Driver {
            runner,
            grid: &self.grid,
            hidden: false,
        };
        d.plan(&self.seq, &self.ladder, p, reuse)
    }
    fn run(&mut self, runner: &FakeRunner, pl: Option<Plan>, toks: &[u32], hidden: bool) -> Run {
        let mut chunks = Vec::new();
        let cancel = |_: usize| runner.stop.get();
        let d = Driver {
            runner,
            grid: &self.grid,
            hidden,
        };
        let mut cb = |r, _: &FakeCall| {
            chunks.push(r);
            Ok(())
        };
        let out = d.prefill(&mut self.seq, &mut self.ladder, pl, toks, &cancel, &mut cb);
        Run { out, chunks }
    }
    fn resume(&mut self, runner: &FakeRunner, p: &[u32], want_reuse: usize) -> Run {
        let pl = self.plan(runner, p, true);
        assert_eq!(pl.reuse(), want_reuse);
        let run = self.run(runner, Some(pl), &p[want_reuse..], false);
        assert_eq!(done_from(&run.out), want_reuse);
        run
    }
}

fn trip_runner(call: usize, layer: usize) -> FakeRunner {
    FakeRunner {
        trip: Some((call, layer)),
        ..FakeRunner::default()
    }
}

fn done_from(out: &Result<Prefilled<FakeCall>>) -> usize {
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
    let gids: Vec<u32> = usizes(&f["gen_prompt"]["ids"])
        .into_iter()
        .map(|v| v as u32)
        .collect();
    assert_eq!(gids, GEN);
    for s in f["samples"].as_array().unwrap() {
        let name = s["name"].as_str().unwrap();
        let (len, start, chunk) = (u(&s["prompt_len"]), u(&s["start"]), u(&s["chunk"]));
        assert_eq!(u(&s["gen_prompt_start"]), len - GEN.len(), "{name}");
        let (p, g) = (prompt_of(len), grid(chunk));
        assert_eq!(g.ends(start, &p), usizes(&s["prefill_ends"]), "{name}");
        let (mut want, mut t) = (Vec::new(), None);
        for st in s["states"].as_array().unwrap() {
            let row = u(&st["row"]);
            assert_eq!(
                row.is_multiple_of(chunk),
                st["on_chunk_grid"].as_bool().unwrap()
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
        assert_eq!(
            t,
            (!got.iter().any(|s| s.row == tr)).then_some(tr),
            "{name}: T"
        );
    }
}

#[test]
fn cross_owner_hits_take_the_deepest_state_row() {
    let f = fixture("block-rows.json");
    for c in f["cross_owner"].as_array().unwrap() {
        let (shared, hit, a) = (u(&c["shared_prefix"]), u(&c["hit_row"]), prompt_of(60_000));
        let (mut env, mut at) = (Env::new(8192), 0);
        for r in usizes(&c["owner_a_state_rows"]) {
            env.seq.push(&a[at..r]);
            env.ladder.put(Kind::Checkpoint, env.seq.snap());
            at = r;
        }
        env.seq.push(&a[at..]);
        let mut b = a[..shared].to_vec();
        b.extend((0..100).map(|i| 50_000 + i));
        let runner = FakeRunner::default();
        env.resume(&runner, &b, hit);
        assert_eq!(env.seq.tokens, b, "shared {shared}");
        assert_eq!(runner.rows()[0].start, hit);
    }
}

#[test]
fn state_rows_close_whole_kpool_groups() {
    let f = fixture("kpool-splits.json");
    let (kpool, disk) = (
        u(&f["geometry"]["kpool"]),
        u(&f["geometry"]["disk_block_rows"]),
    );
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
        assert_eq!(
            (u(&st["closed"]), u(&st["disk_blocks_closed"])),
            (len / kpool, len / disk)
        );
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
        [8192, 16_384, 24_576, 32_768, 65_536, 98_304]
    );
    assert_eq!(checkpoints(8192, 16_384), Vec::<usize>::new());
    assert_eq!(checkpoints(8191, 16_385), [8192, 16_384]);
    assert_eq!((turn_row(4097), turn_row(31)), (4096, 0));
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
    assert!(Grid::new(0, OPEN.to_vec()).is_err() && Grid::new(100, OPEN.to_vec()).is_err());
    let (g, p) = (grid(8192), prompt_of(900));
    assert_eq!(g.replay_row(&p), 864);
    assert!(g.ends(900, &p).is_empty() && g.states(864, &p).is_empty());
    let (raw, plain) = (Grid::new(8192, Vec::new()).unwrap(), vec![FILL; 1000]);
    assert_eq!(
        (raw.replay_row(&plain), raw.ends(0, &plain)),
        (992, vec![992, 1000])
    );
    let mut two = prompt_of(5000);
    two[100..103].copy_from_slice(&OPEN);
    assert_eq!(g.replay_row(&two), 4992);
    let mut first_only = vec![FILL; 50];
    first_only[..3].copy_from_slice(&OPEN);
    assert_eq!(g.replay_row(&first_only), 32);
}

/// Historical assistant openings inside a cold multi-turn prompt must NOT
/// multiply prefill waves: a 4,096-token prompt containing 100 short historical
/// turns on distinct block rows still cuts into only 2 waves (`replay_row` and
/// prompt end), matching a single-turn prompt.
#[test]
fn many_historical_openings_keep_cold_prefill_waves_bounded() {
    let g = grid(4096);
    let mut p = prompt_of(4096);
    for turn in 0..100usize {
        let at = (turn + 1) * BLOCK_ROWS + 5;
        p[at..at + OPEN.len()].copy_from_slice(&OPEN);
    }
    let replay = g.replay_row(&p);
    assert_eq!(replay, 4064);
    assert_eq!(g.ends(0, &p), vec![4064, 4096]);
    assert_eq!(
        g.states(0, &p),
        vec![StateRow {
            row: 4064,
            kind: Kind::Prompt
        }]
    );
}

/// The truncate rules of `WindowRing` and `IndexerState`.
#[test]
fn truncate_rules() {
    let win = [
        (1000, 515, 1000, true),
        (1000, 515, 1001, false),
        (1000, 515, 997, true),
        (1000, 515, 996, false),
        (1000, 515, 0, true),
        (100, 100, 1, true),
        (515, 515, 3, true),
        (997, 512, 997, true),
        (997, 512, 996, false),
    ];
    win.iter()
        .for_each(|&(n, k, l, w)| assert_eq!(window_keeps(n, k, 512, l), w));
    let idx = [
        (1000, 6, 4, 996, true),
        (1000, 6, 4, 997, true),
        (1000, 6, 4, 4, true),
        (1000, 1, 4, 997, false),
        (1000, 1, 4, 996, true),
        (1000, 6, 0, 996, false),
        (10, 6, 4, 11, false),
    ];
    idx.iter()
        .for_each(|&(n, h, kp, l, w)| assert_eq!(index_keeps(n, h, kp, l), w));
}

#[test]
fn ladder_keeps_the_newest_and_drops_on_cuts() {
    let s = |len| FakeSnap { len, stamp: None };
    let mut l: Ladder<FakeSnap> = Ladder::new();
    for n in [300, 100, 200] {
        l.put(Kind::Checkpoint, s(n));
    }
    assert_eq!(l.lens(), [100, 200, 300]);
    l.put(Kind::Turn, s(200));
    assert_eq!((l.len(), l.kind(200)), (3, Some(Kind::Turn)));
    assert_eq!(l.deepest(250, 1000), Some(200));
    assert_eq!(
        (l.deepest(1000, 250), l.deepest(99, 1000)),
        (Some(200), None)
    );
    l.drop_above(200);
    assert_eq!(l.lens(), [100, 200]);
    l.remove(100);
    assert_eq!(l.lens(), [200]);
    l.clear();
    assert!(l.is_empty());

    (1..=RAM_STATES + 3).for_each(|n| l.put(Kind::Checkpoint, s(n * 10)));
    assert_eq!((l.len(), l.lens().first()), (RAM_STATES, Some(&40)));
    l.put(Kind::Prompt, s(5));
    assert!(l.len() == RAM_STATES && l.contains(5) && !l.contains(40) && s(0).is_empty());
}

#[test]
fn cold_prefill_cuts_on_the_grid() {
    let (runner, p, mut env) = (FakeRunner::default(), prompt_of(20_000), Env::new(8192));
    let run = env.resume(&runner, &p, 0);
    let want = [0..8192, 8192..16_384, 16_384..19_968, 19_968..20_000];
    assert_eq!((runner.rows(), run.chunks), (want.to_vec(), want.to_vec()));
    let calls = runner.calls.borrow();
    let logits: Vec<Logits> = calls.iter().map(|c| c.want.logits).collect();
    assert_eq!(
        logits,
        [Logits::None, Logits::None, Logits::None, Logits::Last]
    );
    assert!(calls.iter().all(|c| !c.want.hidden));
    assert_eq!(env.ladder.lens(), [8192, 16_384, 19_968]);
    assert_eq!(env.ladder.kind(16_384), Some(Kind::Checkpoint));
    assert_eq!(env.ladder.kind(19_968), Some(Kind::Prompt));
    assert_eq!(env.seq.tokens, p);
}

#[test]
fn hidden_rows_are_asked_of_every_chunk() {
    let (runner, p, mut env) = (FakeRunner::default(), prompt_of(5000), Env::new(2048));
    assert_eq!(done_from(&env.run(&runner, None, &p, true).out), 0);
    assert!(runner.calls.borrow().iter().all(|c| c.want.hidden));
    assert_eq!(runner.rows().len(), 4);
}

#[test]
fn cancel_keeps_finished_chunks() {
    let (p, mut env) = (prompt_of(20_000), Env::new(8192));
    let run = env.run(&trip_runner(2, 1), None, &p, false);
    assert!(matches!(run.out, Ok(Prefilled::Cancelled { at: 16_384 })));
    assert_eq!(env.seq.tokens, p[..16_384]);
    assert_eq!(run.chunks, [0..8192, 8192..16_384]);
    assert_eq!(env.ladder.kind(16_384), Some(Kind::Checkpoint));

    let again = FakeRunner::default();
    env.resume(&again, &p, 16_384);
    assert_eq!(again.rows(), [16_384..19_968, 19_968..20_000]);
    assert_eq!(env.seq.tokens, p);
}

#[test]
fn cancel_off_a_state_row_leaves_a_cancel_entry() {
    let (p, mut env) = (prompt_of(5000), Env::new(2048));
    let r = env.run(&trip_runner(1, 3), None, &p, false).out;
    assert!(matches!(r, Ok(Prefilled::Cancelled { at: 2048 })));
    assert_eq!(env.ladder.kind(2048), Some(Kind::Cancel));

    let mut env = Env::new(2048);
    let r = env.run(&trip_runner(0, 0), None, &p, false).out;
    assert!(matches!(r, Ok(Prefilled::Cancelled { at: 0 })));
    assert!(env.ladder.is_empty() && env.seq.tokens.is_empty());
}

#[test]
fn a_failed_chunk_is_an_error_and_keeps_the_chunks_before() {
    let (p, mut env) = (prompt_of(20_000), Env::new(8192));
    let runner = FakeRunner {
        fail_at: Some(1),
        ..FakeRunner::default()
    };
    assert!(env.run(&runner, None, &p, false).out.is_err());
    assert_eq!(env.seq.tokens, p[..8192]);
}

#[test]
fn reuse_points() {
    let (p, mut env) = (prompt_of(20_000), Env::new(8192));
    env.run(&FakeRunner::default(), None, &p, false)
        .out
        .unwrap();
    env.seq.push(&[1, 2, 3, 4, 5, 6]);
    env.ladder.put(Kind::Turn, env.seq.snap());

    let again = FakeRunner::default();
    env.resume(&again, &p, 19_968);
    assert_eq!(again.rows(), std::slice::from_ref(&(19_968..20_000)));
    assert_eq!(env.ladder.lens(), [8192, 16_384, 19_968]);
    assert_eq!(env.seq.tokens, p);

    let same = FakeRunner::default();
    env.resume(&same, &p, 19_999);
    assert_eq!(same.rows(), std::slice::from_ref(&(19_999..20_000)));

    let mut longer = p.clone();
    longer.extend([9, 9, 9]);
    let ext = FakeRunner::default();
    assert_eq!(env.plan(&ext, &longer, true).reuse(), 20_000);
    assert_eq!(env.plan(&ext, &longer, false).reuse(), 0);
}

#[test]
fn a_refused_point_recomputes_from_a_deeper_one() {
    let (p, mut env) = (prompt_of(20_000), Env::new(8192));
    env.run(&FakeRunner::default(), None, &p, false)
        .out
        .unwrap();
    env.seq.push(&[1, 2, 3, 4, 5, 6]);
    let runner = FakeRunner::default();
    let pl = env.plan(&runner, &p, true);
    assert_eq!(pl.reuse(), 19_968);
    env.seq.refuse.insert(19_968);
    let run = env.run(&runner, Some(pl), &p[19_968..], false);
    assert_eq!(done_from(&run.out), 16_384);
    assert_eq!(runner.rows(), [16_384..19_968, 19_968..20_000]);
    assert_eq!(env.seq.tokens, p);

    env.seq.refuse.extend([8192, 16_384]);
    env.seq.push(&[1, 2, 3, 4, 5, 6]);
    let all = FakeRunner::default();
    env.resume(&all, &p, 0);
    assert_eq!(all.rows().first(), Some(&(0..8192)));
    assert_eq!(env.seq.tokens, p);
}

#[test]
fn a_stale_plan_extends_the_sequence() {
    let (p, mut env, runner) = (prompt_of(1000), Env::new(8192), FakeRunner::default());
    env.seq.push(&p[..100]);
    let other = env.plan(&runner, &prompt_of(3000), true);
    let run = env.run(&runner, Some(other), &p[100..], false);
    assert_eq!(done_from(&run.out), 100);
    assert_eq!(runner.rows(), [100..992, 992..1000]);
    assert_eq!(env.seq.tokens, p);
}

#[test]
fn refused_inputs_leave_the_sequence() {
    let (mut env, runner) = (Env::with_cap(8192, 64), FakeRunner::default());
    env.seq.push(&[1, 2, 3]);
    assert!(env.run(&runner, None, &[], false).out.is_err());
    assert!(env.run(&runner, None, &prompt_of(100), false).out.is_err());
    assert_eq!(env.seq.tokens, [1, 2, 3]);
    assert!(runner.rows().is_empty());
    assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 4]), 2);
}
