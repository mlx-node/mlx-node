//! States at block rows: the rows a turn stores (fixture block-rows.json),
//! the T candidate, the RAM bound, the cut rule and shared bytes. CPU only.

use std::path::PathBuf;

use serde_json::Value;

use super::ladder::{Kind, RAM_STATES};
use super::prefill::{BLOCK_ROWS, Grid, StateRow, turn_row};
use super::prefix_state::{Key, PrefixState, States, Stored, Turn};

const OPEN: [u32; 3] = [120_000, 611, 10_372];
const GEN: [u32; 5] = [120_000, 611, 10_372, 120_001, 120_029];
const SALT: u64 = 7;

fn fixture() -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hy-v4-flash/block-rows.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

fn u(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

/// `len` tokens with the generation prompt at the end.
fn prompt_of(len: usize) -> Vec<u32> {
    let mut p: Vec<u32> = (0..len).map(|i| 7 + (i % 97) as u32).collect();
    p[len - GEN.len()..].copy_from_slice(&GEN);
    p
}

fn bare(row: usize) -> PrefixState {
    PrefixState {
        row,
        layers: Vec::new(),
        draft: None,
    }
}

/// A turn that prefills `prompt_len` rows from `start` and decodes to
/// `kv_end`, holding a state at every block row it reaches past the
/// prompt's state rows; its stored rows and its T.
fn run_turn(g: &Grid, start: usize, prompt_len: usize, kv_end: usize) -> Vec<StateRow> {
    let p = prompt_of(prompt_len);
    let mut turn = Turn::new(g, start, &p);
    let mut rows = turn.rows().to_vec();
    let first = prompt_len.next_multiple_of(BLOCK_ROWS);
    for row in (first..=kv_end).step_by(BLOCK_ROWS) {
        turn.hold(bare(row));
    }
    if let Some((t, s)) = turn.close(kv_end) {
        assert_eq!(t.row, s.row);
        rows.push(t);
    }
    rows
}

/// Every block-rows.json sample: checkpoints every 8,192 below 32,768 and
/// every 32,768 above, P, then T = the block row below the KV end, not
/// stored again when it is P.
/// Mutation: `Turn::close` keeps `kv_end` instead of `turn_row` → red.
#[test]
fn state_rows_follow_the_block_rows_fixture() {
    let f = fixture();
    for s in f["samples"].as_array().unwrap() {
        let name = s["name"].as_str().unwrap();
        let g = Grid::new(u(&s["chunk"]), OPEN.to_vec()).unwrap();
        let got = run_turn(&g, u(&s["start"]), u(&s["prompt_len"]), u(&s["kv_end"]));
        let want: Vec<StateRow> = s["states"]
            .as_array()
            .unwrap()
            .iter()
            .map(|st| StateRow {
                row: u(&st["row"]),
                kind: match st["kind"].as_str().unwrap() {
                    "checkpoint" => Kind::Checkpoint,
                    "P" => Kind::Prompt,
                    "T" => Kind::Turn,
                    k => panic!("{name}: kind {k}"),
                },
            })
            .collect();
        assert_eq!(got, want, "{name}");
        assert!(got.iter().all(|r| r.row.is_multiple_of(BLOCK_ROWS)));
    }
}

/// T: a state at or below the prompt's last state row is not held, an
/// older one never replaces a newer one, a cut drops a held state past
/// it, and a held state past `turn_row(kv_end)` is not T; a commit that
/// stepped over that row leaves the deepest one below.
#[test]
fn turn_candidate_rules() {
    let g = Grid::new(8192, OPEN.to_vec()).unwrap();
    let p = prompt_of(900);
    let mut t = Turn::new(&g, 0, &p);
    assert_eq!(t.prompt_kind(864), Some(Kind::Prompt));
    t.hold(bare(864));
    assert_eq!(t.held_row(), None);
    t.hold(bare(960));
    t.hold(bare(928));
    assert_eq!(t.held_row(), Some(960));
    t.cut(950);
    assert_eq!(t.held_row(), None);
    t.hold(bare(992));
    assert!(t.close(980).is_none());
    t.hold(bare(1024));
    let (row, _) = t.close(1100).unwrap();
    assert_eq!(row.row, 1024);
    assert_ne!(row.row, turn_row(1100));
    assert_eq!(row.kind, Kind::Turn);
    assert_eq!(t.held_row(), None);
}

struct Fake {
    own: u64,
    shared: Vec<(usize, u64)>,
}

impl Stored for Fake {
    fn bytes(&self) -> u64 {
        self.own + self.shared.iter().map(|s| s.1).sum::<u64>()
    }

    fn shared(&self) -> Vec<(usize, u64)> {
        self.shared.clone()
    }
}

fn fake(own: u64) -> Fake {
    Fake {
        own,
        shared: Vec::new(),
    }
}

/// `blocks` whole blocks of a lineage numbered `n`.
fn lineage(n: u32, blocks: usize) -> Vec<u32> {
    (0..blocks * BLOCK_ROWS)
        .map(|i| n * 1000 + i as u32)
        .collect()
}

fn publish(st: &mut States<Fake>, tokens: &[u32], kind: Kind, own: u64) -> Option<u64> {
    st.publish(Key::text(tokens, SALT), fake(own), kind)
        .unwrap()
}

fn hit(st: &States<Fake>, prompt: &[u32], cached: bool) -> Option<usize> {
    st.lookup(Key::text(prompt, SALT), &|_| cached).map(|h| h.1)
}

fn longer(tokens: &[u32]) -> Vec<u32> {
    let mut p = tokens.to_vec();
    p.push(9);
    p
}

/// Nine states: the oldest RAM copy goes, eight stay.
/// Mutation: the bound check uses `>` → red (nine kept).
#[test]
fn ram_bound_keeps_the_newest_eight() {
    let mut st = States::<Fake>::new();
    let all: Vec<Vec<u32>> = (1..=RAM_STATES as u32 + 1).map(|n| lineage(n, 1)).collect();
    for l in &all {
        assert!(publish(&mut st, l, Kind::Prompt, 10).is_some());
    }
    assert_eq!(st.ram_len(), RAM_STATES);
    assert_eq!(hit(&st, &longer(&all[0]), true), None);
    for l in &all[1..] {
        assert_eq!(hit(&st, &longer(l), true), Some(BLOCK_ROWS));
    }
}

/// A checkpoint goes before an older ordinary state.
#[test]
fn a_checkpoint_goes_first() {
    let mut st = States::<Fake>::with_cap(2);
    let (a, b, c) = (lineage(1, 1), lineage(2, 1), lineage(3, 1));
    publish(&mut st, &a, Kind::Prompt, 1);
    publish(&mut st, &b, Kind::Checkpoint, 1);
    publish(&mut st, &c, Kind::Turn, 1);
    assert_eq!(hit(&st, &longer(&a), true), Some(BLOCK_ROWS));
    assert_eq!(hit(&st, &longer(&b), true), None);
    assert_eq!(hit(&st, &longer(&c), true), Some(BLOCK_ROWS));
}

/// A cut below a state drops it unless its blocks stay cached; shallower
/// states stay.
/// Mutation: `cut` keeps every state → red.
#[test]
fn a_cut_drops_the_states_past_it() {
    let tokens = lineage(4, 3);
    let fill = |st: &mut States<Fake>| {
        for b in 1..=3 {
            publish(st, &tokens[..b * BLOCK_ROWS], Kind::Checkpoint, 1);
        }
    };
    let mut st = States::<Fake>::new();
    fill(&mut st);
    st.cut(&tokens, SALT, 40, &|_| false);
    assert_eq!(st.ram_rows(), vec![BLOCK_ROWS]);
    let mut kept = States::<Fake>::new();
    fill(&mut kept);
    kept.cut(&tokens, SALT, 40, &|_| true);
    assert_eq!(kept.ram_rows(), vec![32, 64, 96]);
    let mut other = States::<Fake>::new();
    fill(&mut other);
    other.cut(&tokens, SALT + 1, 0, &|_| false);
    assert_eq!(other.ram_rows(), vec![32, 64, 96]);
}

/// The deepest strict prefix with its blocks cached; none when the pool
/// lost them.
#[test]
fn lookup_takes_the_deepest_cached_prefix() {
    let tokens = lineage(5, 2);
    let mut st = States::<Fake>::new();
    publish(&mut st, &tokens[..BLOCK_ROWS], Kind::Checkpoint, 1);
    publish(&mut st, &tokens, Kind::Prompt, 1);
    assert_eq!(hit(&st, &longer(&tokens), true), Some(64));
    assert_eq!(hit(&st, &tokens, true), Some(32));
    let mut fork = tokens[..40].to_vec();
    fork.extend([1, 2, 3]);
    assert_eq!(hit(&st, &fork, true), Some(32));
    assert_eq!(hit(&st, &longer(&tokens), false), None);
    st.prune(&|_| false);
    assert_eq!(st.ram_len(), 0);
}

/// Shared parts count once in the held bytes; off-block rows are refused.
#[test]
fn held_bytes_count_shared_parts_once() {
    let mut st = States::<Fake>::new();
    for (n, own) in [(6, 10), (7, 20)] {
        let s = Fake {
            own,
            shared: vec![(1, 100), (n as usize, 5)],
        };
        st.publish(Key::text(&lineage(n, 1), SALT), s, Kind::Prompt)
            .unwrap();
    }
    assert_eq!(st.held_bytes(), 10 + 20 + 100 + 5 + 5);
    let off = lineage(8, 1);
    assert!(
        st.publish(Key::text(&off[..31], SALT), fake(1), Kind::Prompt)
            .is_err()
    );
}
