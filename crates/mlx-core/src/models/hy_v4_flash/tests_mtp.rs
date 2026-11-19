//! MTP head: held sizes against `mtp-tensors.json`, the prefill lag
//! schedule, a lens model of the stepper's lengths over whole cycles at
//! depth 1..3, the K/V buffer and seeds; real (`HY4_DIR`): the held layer
//! size and a greedy MTP turn against the AR turn.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use serde_json::Value;

use super::cache::WINDOW_SLACK;
use super::mtp::{KvRows, Lag, SEEDS, Seeds, lag};
use super::mtp_weights::{Held, held, held_bytes};
use super::spec::{LOOKAHEAD_MAX, commit_row, cycle_rows, unemitted_keep, verify_keep};
use crate::array::MxArray;
use crate::engine::types::ChatConfig;
use crate::models::qwen3_5::mtp_decode::MtpCommitAnchor;
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hy-v4-flash/mtp-tensors.json");
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

fn info(t: &Value) -> GgufTensorInfo {
    let dims: Vec<u64> = t["ggml_dims"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_u64().unwrap())
        .collect();
    GgufTensorInfo {
        name: t["name"].as_str().unwrap().to_string(),
        n_dims: dims.len() as u32,
        dims,
        tensor_type: match t["type"].as_str().unwrap() {
            "BF16" => GgufTensorType::BF16,
            "F32" => GgufTensorType::F32,
            other => panic!("unexpected type {other}"),
        },
        offset: 0,
    }
}

#[test]
fn held_bytes_match_the_fixture() {
    let f = fixture();
    let mut total = 0u64;
    let mut experts = 0u64;
    for t in f["tensors"].as_array().unwrap() {
        let expert = t["category"] == "routed_expert";
        let i = info(t);
        let want = t["plan_form"]["bytes"].as_u64().unwrap();
        assert_eq!(held_bytes(&i, expert).unwrap(), want, "{}", i.name);
        let form = match held(&i, expert).unwrap() {
            Held::Qmm => "qmm_affine8_g32_f32_scale_bias",
            Held::F32 => "f32",
            Held::Affine4 => "affine4_g64_bf16_scale_bias",
        };
        assert_eq!(t["plan_form"]["form"], form, "{}", i.name);
        total += want;
        if expert {
            experts += want;
        }
    }
    assert_eq!(total, f["totals"]["plan_bytes"].as_u64().unwrap());
    assert_eq!(
        experts,
        f["totals"]["routed_expert_plan_bytes"].as_u64().unwrap()
    );
}

#[test]
fn held_refuses_types_the_layer_does_not_use() {
    let mut i = info(&fixture()["tensors"][0]);
    i.tensor_type = GgufTensorType::Q8_0;
    assert!(held(&i, false).is_err());
    i.tensor_type = GgufTensorType::F32;
    assert!(held(&i, true).is_err());
}

/// Every MTP row `p` pairs trunk output `p` with token `p + 1`, exactly
/// once and in order, however the prompt is cut into chunks.
#[test]
fn lag_rows_pair_each_trunk_row_with_the_next_token() {
    assert_eq!(
        lag(0, 5),
        Lag {
            at: 0,
            seed: false,
            rows: 4
        }
    );
    assert_eq!(lag(0, 1).rows, 0);
    assert_eq!(
        lag(8, 3),
        Lag {
            at: 7,
            seed: true,
            rows: 3
        }
    );
    for cuts in [
        vec![1, 1, 1, 1],
        vec![5],
        vec![1, 4, 2048, 3],
        vec![2048, 2048, 7],
        vec![32, 1, 31, 64],
    ] {
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut start = 0;
        for n in &cuts {
            let l = lag(start, *n);
            assert_eq!(l.at, pairs.len(), "cuts {cuts:?}");
            let mut hidden: Vec<usize> = Vec::new();
            if l.seed {
                hidden.push(start - 1);
            }
            hidden.extend(start..start + n - 1);
            let ids: Vec<usize> = (start..start + n).skip(n - l.rows).collect();
            assert_eq!(hidden.len(), l.rows);
            pairs.extend(hidden.into_iter().zip(ids));
            start += n;
        }
        let want: Vec<(usize, usize)> = (0..start - 1).map(|p| (p, p + 1)).collect();
        assert_eq!(pairs, want, "cuts {cuts:?}");
    }
}

/// A draft row: `Some((hidden position, token position))` for an exact
/// pair, `None` for a draft.
type Row = Option<(usize, usize)>;

/// The stepper's lengths through `run_mtp_cycle`'s calls.
struct Lens {
    trunk: usize,
    rows: Vec<Row>,
    committed: usize,
    snap: usize,
}

impl Lens {
    /// A settled sequence after a prompt of `t` tokens.
    fn new(t: usize) -> Self {
        Self {
            trunk: t,
            rows: (0..t - 1).map(|p| Some((p, p + 1))).collect(),
            committed: t - 1,
            snap: 0,
        }
    }

    fn step_a(&mut self) {
        assert_eq!(self.rows.len(), self.committed);
        if self.rows.len() + 1 == self.trunk {
            self.rows.push(Some((self.trunk - 1, self.trunk)));
        }
        assert_eq!(self.rows.len(), self.trunk);
        self.trunk += 1;
        self.committed = self.rows.len();
    }

    fn cut(&mut self, keep: usize) {
        assert!(keep <= self.rows.len());
        assert!(self.rows.len() - keep <= WINDOW_SLACK);
        self.rows.truncate(keep);
    }

    fn cycle(&mut self, depth: usize, accepted: usize, chained: bool) {
        assert!(accepted <= depth);
        self.cut(cycle_rows(self.committed, chained).unwrap());
        assert_eq!(
            self.rows.len() + 1,
            self.trunk,
            "first draft row is trunk - 1"
        );
        self.rows.push(Some((self.trunk - 1, self.trunk)));
        for _ in 1..depth {
            self.rows.push(None);
        }
        self.snap = self.trunk;
        assert!(depth < LOOKAHEAD_MAX);
        self.trunk += depth + 1;
        let anchor = if chained {
            MtpCommitAnchor::SkipAlreadyCommittedAnchor
        } else {
            MtpCommitAnchor::IncludeAnchor
        };
        let m = match anchor {
            MtpCommitAnchor::IncludeAnchor => accepted + 2,
            MtpCommitAnchor::SkipAlreadyCommittedAnchor => accepted + 1,
        };
        let at = commit_row(anchor, self.snap).unwrap();
        assert_eq!(at, self.committed);
        self.cut(at);
        // the hiddens of rows at.. are the seed (Step A) or verify rows
        for i in 0..m {
            self.rows.push(Some((at + i, at + i + 1)));
        }
        self.committed = self.rows.len();
        let keep = verify_keep(self.snap, accepted);
        assert!(self.trunk - keep <= WINDOW_SLACK);
        self.trunk = keep;
        self.check(self.trunk);
    }

    fn unemitted(&mut self, u: usize) {
        self.trunk = unemitted_keep(self.trunk, u);
        let keep = self.rows.len().min(self.trunk - 1);
        self.cut(keep);
        self.committed = keep;
        self.check(self.trunk - 1);
    }

    fn settle(&mut self) {
        let keep = self.rows.len().min(self.trunk - 1);
        self.rows.truncate(keep);
        self.check(self.trunk - 1);
    }

    /// `len` rows, each the exact pair of its own position.
    fn check(&self, len: usize) {
        assert_eq!(self.rows.len(), len);
        for (p, r) in self.rows.iter().enumerate() {
            assert_eq!(*r, Some((p, p + 1)), "row {p} of {:?}", self.rows);
        }
    }
}

/// Step-A and chained cycles at depth 1..3 with every accept count, a
/// near-tail skip (two Step A in a row) and a mid-cycle stop: the draft
/// rows stay the exact pairs of their positions and every cut stays
/// within the slack.
#[test]
fn lens_model_keeps_exact_rows_through_cycles() {
    for depth in 1..=3 {
        for chained in [false, true] {
            let mut l = Lens::new(37);
            l.step_a();
            for k in 0..=depth {
                l.cycle(depth, k, false);
                if chained {
                    l.cycle(depth, depth - k, true);
                } else {
                    l.step_a();
                }
                l.step_a();
            }
            for u in 1..=depth {
                let mut stop = Lens::new(20);
                stop.step_a();
                stop.cycle(depth, depth, false);
                stop.unemitted(u);
            }
            l.settle();
        }
    }
}

#[test]
fn trunk_cuts_fit_the_slack() {
    for depth in 1..=3 {
        for k in 0..=depth {
            let snap = 100;
            assert!(snap + depth + 1 - verify_keep(snap, k) <= WINDOW_SLACK);
        }
        for u in 1..=depth {
            assert!(100 - unemitted_keep(100, u) <= WINDOW_SLACK);
        }
    }
    assert_eq!(cycle_rows(0, true), None);
    assert_eq!(commit_row(MtpCommitAnchor::IncludeAnchor, 0), None);
}

fn rows(n: usize, from: usize) -> MxArray {
    let v: Vec<f32> = (from * 4..(from + n) * 4).map(|x| x as f32).collect();
    MxArray::from_float32(&v, &[n as i64, 4]).unwrap()
}

fn read(x: &MxArray, n: usize) -> Vec<f32> {
    x.slice(&[0, 0], &[n as i64, 4])
        .unwrap()
        .astype(crate::array::DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .to_vec()
}

/// Writes past the buffer move the rows to one of twice the rows (a
/// multiple of 32, at most the cap) and read back in order; a draft read is the buffer itself; a
/// rollback rewrites in place; a deep cut shrinks it and a cut to 0 frees it.
#[test]
fn kv_rows_grow_cut_and_rewrite() {
    let mut b = KvRows::new(4, 300);
    let mut caps = Vec::new();
    let mut at = 0;
    for n in [1, 31, 40, 5, 200] {
        b.write(&rows(n, at), at).unwrap();
        at += n;
        caps.push(b.capacity());
    }
    assert_eq!(caps, vec![32, 32, 96, 96, 288]);
    assert_eq!(b.len(), at);
    assert_eq!(read(b.rows().unwrap(), at), read(&rows(at, 0), at));
    assert!(b.write(&rows(24, at), at).is_err());
    assert_eq!(b.len(), at);

    b.truncate(at - 3).unwrap();
    assert_eq!(b.capacity(), 288);
    b.write(&rows(3, at - 3), at - 3).unwrap();
    assert_eq!(read(b.rows().unwrap(), at), read(&rows(at, 0), at));

    b.truncate(70).unwrap();
    assert_eq!(b.capacity(), 288);
    b.truncate(40).unwrap();
    assert_eq!(b.capacity(), 128);
    assert_eq!(b.rows().unwrap().shape_at(0).unwrap(), 128);
    b.write(&rows(3, 40), 40).unwrap();
    assert_eq!(read(b.rows().unwrap(), 43), read(&rows(43, 0), 43));
    b.truncate(0).unwrap();
    assert_eq!((b.capacity(), b.nbytes()), (0, 0));
    assert!(b.rows().is_err());
    assert!(b.write(&rows(1, 0), 1).is_err());
}

#[test]
fn seeds_keep_the_last_rows_by_position() {
    let h = rows(12, 0);
    let mut s = Seeds::default();
    s.extend(Seeds::copies(&h, 100).unwrap());
    assert!(s.get(100 + 12 - SEEDS - 1).is_none());
    assert_eq!(read(s.get(111).unwrap(), 1), read(&rows(1, 11), 1));
    s.cut(110);
    assert!(s.get(110).is_none() && s.get(109).is_some());
    s.extend(Seeds::copies(&rows(1, 50), 110).unwrap());
    assert_eq!(read(s.get(110).unwrap(), 1), read(&rows(1, 50), 1));
}

fn hy4_dir() -> PathBuf {
    PathBuf::from(std::env::var("HY4_DIR").expect("HY4_DIR = directory of the two Hy4 shards"))
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_held_layer_matches_the_fixture() {
    let paths = crate::models::deepseek_v41::chat::shard_paths(&hy4_dir()).unwrap();
    let opts = super::store::Options::from_env(Some(16_384), true).unwrap();
    let store = super::store::Store::open(&paths, &opts).unwrap();
    let m = super::mtp::MtpLayer::load(&store.scan, &store.shards, &store.model).unwrap();
    assert_eq!(
        m.bytes(),
        fixture()["totals"]["plan_bytes"].as_u64().unwrap()
    );
}

fn chat_msg(role: &str, content: &str) -> crate::tokenizer::ChatMessage {
    serde_json::from_value(serde_json::json!({"role": role, "content": content})).unwrap()
}

/// Greedy MTP at depth 1..3 gives the AR tokens; cycles run.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_greedy_mtp_matches_ar() {
    use crate::engine::backend::{ChatBackend, ResetScope};
    let mut inner = super::chat::Inner::load(&hy4_dir(), Some(16_384), Default::default()).unwrap();
    let flag = Arc::new(AtomicBool::new(false));
    let msgs = || vec![chat_msg("user", "Count from 1 to 40, comma separated.")];
    let cfg = |mtp: bool, depth: i32| ChatConfig {
        max_new_tokens: Some(96),
        temperature: Some(0.0),
        enable_mtp: Some(mtp),
        mtp_depth: Some(depth),
        report_performance: Some(true),
        ..Default::default()
    };
    let ar =
        crate::engine::session::session_start(&mut inner, msgs(), cfg(false, 1), &flag).unwrap();
    for depth in 1..=3 {
        inner.reset_caches(ResetScope::Command).unwrap();
        let r = crate::engine::session::session_start(&mut inner, msgs(), cfg(true, depth), &flag)
            .unwrap();
        println!(
            "depth {depth}: {:?} perf {:?}",
            r.text,
            r.performance
                .as_ref()
                .map(|p| (p.mtp_cycles, p.mtp_mean_accepted_tokens))
        );
        assert_eq!(r.text, ar.text, "depth {depth}");
        assert!(
            r.performance
                .as_ref()
                .and_then(|p| p.mtp_cycles)
                .is_some_and(|c| c > 0)
        );
    }
}
