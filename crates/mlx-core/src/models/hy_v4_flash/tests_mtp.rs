//! MTP head: held sizes against `mtp-tensors.json`, prefill lag schedule,
//! cycle lens model at depth 1..3, K/V buffer & seeds, batched drafting,
//! draft head modes, and real-weight parity (`HY4_DIR`).

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use serde_json::Value;

use super::cache::WINDOW_SLACK;
use super::mtp::{
    Draft, KvRows, Lag, MtpHeadMode, MtpLayer, SEEDS, Seeds, lag, parse_mtp_head_mode,
    with_mtp_head,
};
use super::mtp_weights::{
    Held, MtpDenseMode, held, held_bytes, parse_mtp_dense_mode, with_mtp_dense,
};
use super::spec::{LOOKAHEAD_MAX, commit_row, cycle_rows, unemitted_keep, verify_keep};
use super::stats::{self, DeferScope};
use super::test_support::{
    MiniMtpFixture, Rng, arr, fixture, hy4_dir, j_u64, j_u64s, mini_mtp_cfg, mini_mtp_fixture,
    mini_mtp_weights, read_bits, read_f32,
};
use super::weights::Dense;
use crate::array::{DType, MxArray};
use crate::engine::mtp_decode::MtpCommitAnchor;
use crate::engine::types::ChatConfig;
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};

fn info(t: &Value) -> GgufTensorInfo {
    let dims = j_u64s(&t["ggml_dims"]);
    let tensor_type = match t["type"].as_str().unwrap() {
        "BF16" => GgufTensorType::BF16,
        "F32" => GgufTensorType::F32,
        other => panic!("unexpected type {other}"),
    };
    GgufTensorInfo {
        name: t["name"].as_str().unwrap().into(),
        n_dims: dims.len() as u32,
        dims,
        tensor_type,
        offset: 0,
    }
}

#[test]
fn held_bytes_match_the_fixture() {
    for (raw, want) in [
        (None, MtpDenseMode::SymQ8),
        (Some(""), MtpDenseMode::SymQ8),
        (Some("symq8"), MtpDenseMode::SymQ8),
        (Some("q4"), MtpDenseMode::Q4),
        (Some(" Q4 "), MtpDenseMode::Q4),
    ] {
        assert_eq!(parse_mtp_dense_mode(raw).unwrap(), want, "{raw:?}");
    }
    assert!(parse_mtp_dense_mode(Some("bf16")).is_err());

    let f = fixture("mtp-tensors.json");
    let tensors = f["tensors"].as_array().unwrap();
    with_mtp_dense(MtpDenseMode::SymQ8, || {
        let (mut total, mut experts) = (0u64, 0u64);
        for t in tensors {
            let (expert, i) = (t["category"] == "routed_expert", info(t));
            let want = j_u64(&t["plan_form"]["bytes"]);
            assert_eq!(held_bytes(&i, expert).unwrap(), want, "{}", i.name);
            let form = match held(&i, expert).unwrap() {
                Held::SymQ8 => "symq8_g32_f16_scale",
                Held::F32 => "f32",
                Held::Affine4 => "affine4_g64_bf16_scale_bias",
            };
            assert_eq!(t["plan_form"]["form"], form, "{}", i.name);
            total += want;
            if expert {
                experts += want;
            }
        }
        assert_eq!(total, j_u64(&f["totals"]["plan_bytes"]));
        assert_eq!(experts, j_u64(&f["totals"]["routed_expert_plan_bytes"]));
    });

    with_mtp_dense(MtpDenseMode::Q4, || {
        let mut q4_non_expert = 0u64;
        for t in tensors {
            let (expert, i) = (t["category"] == "routed_expert", info(t));
            let (n, got, got_b) = (
                i.num_elements().unwrap(),
                held(&i, expert).unwrap(),
                held_bytes(&i, expert).unwrap(),
            );
            if expert {
                assert_eq!(got, Held::Affine4, "{}", i.name);
            } else if t["plan_form"]["form"] == "symq8_g32_f16_scale" {
                assert_eq!((got, got_b), (Held::Affine4, n / 2 + n / 16), "{}", i.name);
                q4_non_expert += got_b;
            } else {
                assert_eq!(
                    (got, got_b),
                    (Held::F32, j_u64(&t["plan_form"]["bytes"])),
                    "{}",
                    i.name
                );
                q4_non_expert += got_b;
            }
        }
        let sym_non_expert = j_u64(&f["totals"]["non_expert_plan_bytes"]);
        assert!(
            q4_non_expert + 50_000_000 < sym_non_expert,
            "Q4 {q4_non_expert} vs {sym_non_expert}"
        );
    });
}

#[test]
fn held_refuses_types_the_layer_does_not_use() {
    let mut i = info(&fixture("mtp-tensors.json")["tensors"][0]);
    i.tensor_type = GgufTensorType::Q8_0;
    assert!(held(&i, false).is_err());
    i.tensor_type = GgufTensorType::F32;
    assert!(held(&i, true).is_err());
}

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
        let (mut pairs, mut start) = (Vec::new(), 0);
        for &n in &cuts {
            let l = lag(start, n);
            assert_eq!(l.at, pairs.len(), "cuts {cuts:?}");
            let mut hidden: Vec<usize> = l.seed.then(|| start - 1).into_iter().collect();
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

/// The stepper's lengths through `run_mtp_cycle`'s calls.
struct Lens {
    trunk: usize,
    rows: Vec<Option<(usize, usize)>>,
    committed: usize,
    snap: usize,
}

impl Lens {
    fn new(t: usize) -> Self {
        let rows = (0..t - 1).map(|p| Some((p, p + 1))).collect();
        Self {
            trunk: t,
            rows,
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
        assert!(keep <= self.rows.len() && self.rows.len() - keep <= WINDOW_SLACK);
        self.rows.truncate(keep);
    }
    fn cycle(&mut self, depth: usize, accepted: usize, chained: bool) {
        assert!(accepted <= depth && depth < LOOKAHEAD_MAX);
        self.cut(cycle_rows(self.committed, chained).unwrap());
        assert_eq!(self.rows.len() + 1, self.trunk);
        self.rows.push(Some((self.trunk - 1, self.trunk)));
        self.rows.extend(std::iter::repeat_n(None, depth - 1));
        self.snap = self.trunk;
        self.trunk += depth + 1;
        let (anchor, m) = if chained {
            (MtpCommitAnchor::SkipAlreadyCommittedAnchor, accepted + 1)
        } else {
            (MtpCommitAnchor::IncludeAnchor, accepted + 2)
        };
        let at = commit_row(anchor, self.snap).unwrap();
        assert_eq!(at, self.committed);
        self.cut(at);
        self.rows.extend((0..m).map(|i| Some((at + i, at + i + 1))));
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
        self.rows.truncate(self.rows.len().min(self.trunk - 1));
        self.check(self.trunk - 1);
    }
    fn check(&self, len: usize) {
        assert_eq!(self.rows.len(), len);
        for (p, &r) in self.rows.iter().enumerate() {
            assert_eq!(r, Some((p, p + 1)), "row {p}");
        }
    }
}

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
            assert!(100 + depth + 1 - verify_keep(100, k) <= WINDOW_SLACK);
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
    arr(&v, &[n, 4])
}
fn read(x: &MxArray, n: usize) -> Vec<f32> {
    read_f32(&x.slice(&[0, 0], &[n as i64, 4]).unwrap())
}
fn kv_bits(x: &MxArray, n: usize) -> Vec<u32> {
    read_bits(&x.slice(&[0, 0], &[n as i64, 4]).unwrap())
}

#[test]
fn kv_rows_grow_cut_and_rewrite() {
    let mut b = KvRows::new(4, 300);
    let (mut caps, mut at) = (Vec::new(), 0);
    for n in [1, 31, 40, 5, 200] {
        b.write(&rows(n, at), at).unwrap();
        at += n;
        caps.push(b.capacity());
    }
    assert_eq!((caps, b.len()), (vec![32, 32, 96, 96, 288], at));
    assert_eq!(read(b.rows().unwrap(), at), read(&rows(at, 0), at));
    assert!(b.write(&rows(24, at), at).is_err() && b.len() == at);

    b.truncate(at - 3).unwrap();
    assert_eq!(b.capacity(), 288);
    b.write(&rows(3, at - 3), at - 3).unwrap();
    assert_eq!(read(b.rows().unwrap(), at), read(&rows(at, 0), at));

    b.truncate(70).unwrap();
    assert_eq!(b.capacity(), 288);
    b.truncate(40).unwrap();
    assert_eq!(
        (b.capacity(), b.rows().unwrap().shape_at(0).unwrap()),
        (128, 128)
    );
    b.write(&rows(3, 40), 40).unwrap();
    assert_eq!(read(b.rows().unwrap(), 43), read(&rows(43, 0), 43));
    b.truncate(0).unwrap();
    assert_eq!((b.capacity(), b.nbytes()), (0, 0));
    assert!(b.rows().is_err() && b.write(&rows(1, 0), 1).is_err());
}

#[test]
fn seeds_keep_the_last_rows_by_position() {
    let mut s = Seeds::default();
    s.extend(Seeds::copies(&rows(12, 0), 100).unwrap());
    assert!(s.get(100 + 12 - SEEDS - 1).is_none());
    assert_eq!(read(s.get(111).unwrap(), 1), read(&rows(1, 11), 1));
    s.cut(110);
    assert!(s.get(110).is_none() && s.get(109).is_some());
    s.extend(Seeds::copies(&rows(1, 50), 110).unwrap());
    assert_eq!(read(s.get(110).unwrap(), 1), read(&rows(1, 50), 1));
}

#[test]
fn kv_rows_and_seeds_defer_in_scope_and_stay_bit_identical_after_rollback() {
    let mut kv = KvRows::new(4, 128);
    kv.write(&rows(4, 0), 0).unwrap();
    let mut seeds = Seeds::default();
    let (want_kv, want_10, want_11) = (
        kv_bits(&rows(7, 0), 7),
        kv_bits(&rows(1, 10), 1),
        kv_bits(&rows(1, 11), 1),
    );
    let check = |kv: &KvRows, seeds: &Seeds| {
        assert_eq!(kv_bits(kv.rows().unwrap(), 7), want_kv);
        assert_eq!(kv_bits(seeds.get(10).unwrap(), 1), want_10);
        assert_eq!(kv_bits(seeds.get(11).unwrap(), 1), want_11);
    };

    for (seed_n, seed_from, do_rollback) in [(4usize, 8usize, true), (2, 10, false)] {
        {
            let _scope = DeferScope::enter();
            let ev = stats::thread_evals();
            kv.write(&rows(3, 4), 4).unwrap();
            assert_eq!((stats::deferred(), stats::thread_evals()), (1, ev));
            let copied = Seeds::copies(&rows(seed_n, seed_from), seed_from).unwrap();
            assert_eq!(copied.len(), seed_n);
            assert_eq!((stats::deferred(), stats::thread_evals()), (1 + seed_n, ev));
            seeds.extend(copied);
            stats::flush("test mtp defer").unwrap();
            assert_eq!((stats::deferred(), stats::thread_evals()), (0, ev + 1));
        }
        check(&kv, &seeds);
        if do_rollback {
            {
                let _scope = DeferScope::enter();
                kv.write(&rows(2, 7).mul_scalar(99.0).unwrap(), 7).unwrap();
                assert_eq!(stats::deferred(), 1);
                kv.truncate(4).unwrap();
            }
            assert_eq!(stats::deferred(), 0);
            seeds.cut(10);
        }
    }
}

struct DraftRig(MiniMtpFixture);

impl DraftRig {
    fn init_pair(
        &self,
        r: &mut Rng,
        hist_lens: &[usize],
    ) -> (Vec<Draft>, Vec<Draft>, Vec<MxArray>, Vec<u32>) {
        let (v, d, ctx) = (
            self.0.cfg.vocab_size,
            self.0.cfg.dim,
            self.0.cfg.context_length,
        );
        let mut a: Vec<Draft> = (0..hist_lens.len())
            .map(|_| self.0.mtp.new_draft(ctx))
            .collect();
        let mut b: Vec<Draft> = (0..hist_lens.len())
            .map(|_| self.0.mtp.new_draft(ctx))
            .collect();
        let (mut seeds_h, mut anchors) = (Vec::new(), Vec::new());
        for (i, &hl) in hist_lens.iter().enumerate() {
            let (hh, hids) = (arr(&r.vec_f32(hl * d, 1.0), &[hl, d]), r.ids(hl, v));
            self.0
                .mtp
                .commit(&self.0.model, &mut a[i], &hh, &hids)
                .unwrap();
            self.0
                .mtp
                .commit(&self.0.model, &mut b[i], &hh, &hids)
                .unwrap();
            seeds_h.push(arr(&r.vec_f32(d, 1.0), &[1, d]));
            anchors.push(r.ids(1, v)[0]);
        }
        (a, b, seeds_h, anchors)
    }

    fn run_batched(
        &self,
        drafts: &mut [Draft],
        seeds_h: &[MxArray],
        anchors: &[u32],
        depths: &[usize],
        skip_last: Option<bool>,
    ) -> (Vec<Vec<u32>>, u64) {
        let (n, d) = (drafts.len(), self.0.cfg.dim as i64);
        let max_depth = depths.iter().copied().max().unwrap_or(0);
        let mut hs = seeds_h.to_vec();
        let mut ts: Vec<MxArray> = anchors
            .iter()
            .map(|&t| MxArray::from_uint32(&[t], &[1]).unwrap())
            .collect();
        let mut step_toks = Vec::with_capacity(max_depth);
        let ev0 = stats::thread_evals();
        {
            let _scope = DeferScope::enter();
            for step in 0..max_depth {
                let active: Vec<usize> = (0..n).filter(|&i| step < depths[i]).collect();
                let cat = |xs: Vec<&MxArray>| {
                    if xs.len() == 1 {
                        xs[0].clone()
                    } else {
                        MxArray::concatenate_many(xs, Some(0)).unwrap()
                    }
                };
                let hb = cat(active.iter().map(|&i| &hs[i]).collect());
                let tb = cat(active.iter().map(|&i| &ts[i]).collect());
                let eb = self.0.model.weights().token_embd().rows(&tb).unwrap();
                let mut lanes: Vec<&mut Draft> = drafts
                    .iter_mut()
                    .enumerate()
                    .filter(|&(i, _)| step < depths[i])
                    .map(|(_, d)| d)
                    .collect();
                let (h_next, logits) = match skip_last {
                    None => self
                        .0
                        .mtp
                        .draft_batch(&self.0.model, &mut lanes, &hb, &eb)
                        .unwrap(),
                    Some(skip) => {
                        let p: Vec<bool> = active
                            .iter()
                            .map(|&i| !skip || step + 1 < depths[i])
                            .collect();
                        self.0
                            .mtp
                            .draft_batch_with_persist(&self.0.model, &mut lanes, &hb, &eb, &p)
                            .unwrap()
                    }
                };
                let next_ids = logits.argmax(-1, Some(false)).unwrap();
                for (k, &lane) in active.iter().enumerate() {
                    let lo = k as i64;
                    hs[lane] = h_next.slice(&[lo, 0], &[lo + 1, d]).unwrap();
                    ts[lane] = next_ids.slice(&[lo], &[lo + 1]).unwrap();
                }
                step_toks.push((active, next_ids));
            }
            let refs: Vec<&MxArray> = step_toks.iter().map(|(_, t)| t).collect();
            stats::eval(&refs, "test mtp draft_batch").unwrap();
        }
        let evals = stats::thread_evals() - ev0;
        let mut out = vec![Vec::new(); n];
        for (active, ids) in &step_toks {
            let vals = ids.to_int32().unwrap();
            for (k, &lane) in active.iter().enumerate() {
                out[lane].push(vals[k] as u32);
            }
        }
        (out, evals)
    }
}

#[test]
fn mtp_draft_batch_matches_serial_multi_lane_multi_depth() {
    crate::models::hy_v4_flash::exact_f32();
    let rig = DraftRig(mini_mtp_fixture(777));
    let d = rig.0.cfg.dim as i64;
    let mut r = Rng::new(779);
    let scenarios: &[(&str, &[usize], &[usize])] = &[
        ("n2_d1", &[6, 11], &[1, 1]),
        ("n2_d2", &[6, 11], &[2, 2]),
        ("n2_d3", &[6, 11], &[3, 3]),
        ("n4_d1", &[5, 9, 13, 7], &[1, 1, 1, 1]),
        ("n4_d2", &[5, 9, 13, 7], &[2, 2, 2, 2]),
        ("n4_d3", &[5, 9, 13, 7], &[3, 3, 3, 3]),
        ("n4_hetero", &[5, 9, 13, 7], &[3, 2, 1, 3]),
    ];

    for &(label, hist_lens, depths) in scenarios {
        let (mut serial, mut batch, seeds_h, anchors) = rig.init_pair(&mut r, hist_lens);
        let mut serial_toks = vec![Vec::new(); hist_lens.len()];
        for (i, &depth) in depths.iter().enumerate() {
            let (mut h, mut tok) = (seeds_h[i].clone(), anchors[i]);
            for _ in 0..depth {
                let idx = MxArray::from_uint32(&[tok], &[1]).unwrap();
                let emb = rig.0.model.weights().token_embd().rows(&idx).unwrap();
                let (h_next, logits) = rig
                    .0
                    .mtp
                    .draft(&rig.0.model, &mut serial[i], &h, &emb)
                    .unwrap();
                let tid = logits.argmax(-1, Some(false)).unwrap();
                tid.eval();
                tok = tid.to_int32().unwrap()[0] as u32;
                serial_toks[i].push(tok);
                h = h_next.reshape(&[1, d]).unwrap();
            }
        }
        let (batch_toks, evals) = rig.run_batched(&mut batch, &seeds_h, &anchors, depths, None);
        assert_eq!(evals, 1, "{label}: expected 1 eval, got {evals}");
        assert_eq!(batch_toks, serial_toks, "{label}: draft_batch vs serial");
    }

    // Skipping `d.index.update` on each lane's final step (`persist = step + 1 < depth`).
    for &(label, hist_lens, depths) in scenarios {
        let (mut full, mut skip, seeds_h, anchors) = rig.init_pair(&mut r, hist_lens);
        let (full_toks, _) = rig.run_batched(&mut full, &seeds_h, &anchors, depths, Some(false));
        let (skip_toks, _) = rig.run_batched(&mut skip, &seeds_h, &anchors, depths, Some(true));
        assert_eq!(
            skip_toks, full_toks,
            "{label}: persist=false changed tokens"
        );
        for i in 0..hist_lens.len() {
            assert_eq!(
                skip[i].index().len(),
                hist_lens[i] + depths[i] - 1,
                "{label} lane {i}"
            );
            assert_eq!(skip[i].len(), hist_lens[i] + depths[i]);
            skip[i].cut(hist_lens[i]).unwrap();
            assert_eq!(
                (skip[i].len(), skip[i].index().len()),
                (hist_lens[i], hist_lens[i])
            );
        }
    }
}

#[test]
fn mtp_head_env_and_symq8_draft_head_accounting_and_parity() {
    for (raw, want) in [
        (None, MtpHeadMode::Q4),
        (Some(""), MtpHeadMode::Q4),
        (Some("q4"), MtpHeadMode::Q4),
        (Some(" Q4 "), MtpHeadMode::Q4),
        (Some("symq8"), MtpHeadMode::SymQ8),
        (Some(" SYMQ8 "), MtpHeadMode::SymQ8),
        (Some("bf16"), MtpHeadMode::Bf16),
        (Some("BF16"), MtpHeadMode::Bf16),
    ] {
        assert_eq!(parse_mtp_head_mode(raw).unwrap(), want, "{raw:?}");
    }
    assert!(parse_mtp_head_mode(Some("q2")).is_err());

    let mut cfg = mini_mtp_cfg();
    cfg.dim = 64; // Q4 g64 draft head requires dim % 64 == 0
    let (v, d) = (cfg.vocab_size, cfg.dim);
    let mut r = Rng::new(701);
    let tok = Dense::Array(arr(&r.vec_f32(v * d, 0.2), &[v, d]));
    let out = Dense::Array(
        arr(&r.vec_f32(v * d, 0.2), &[v, d])
            .astype(DType::BFloat16)
            .unwrap(),
    );
    let build = |mode| {
        with_mtp_head(mode, || {
            let mw = mini_mtp_weights(&cfg, 11, 32, 1234);
            MtpLayer::new_with_head(&cfg, mw, &tok, Some(&out)).unwrap()
        })
    };

    let q4 = build(MtpHeadMode::Q4);
    assert!(matches!(q4.draft_head(), Some(Dense::Qmm4 { .. })));
    assert_eq!(q4.bytes(), 1234 + (v * d / 2 + v * d / 16) as u64);

    let sym = build(MtpHeadMode::SymQ8);
    assert!(matches!(sym.draft_head(), Some(Dense::SymQ8 { .. })));
    assert_eq!(sym.bytes(), 1234 + (v * d + v * d / 16) as u64);

    let bf16 = build(MtpHeadMode::Bf16);
    assert!(bf16.draft_head().is_none() && bf16.bytes() == 1234);
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_held_layer_matches_the_fixture() {
    let paths = crate::models::deepseek_v41::chat::shard_paths(&hy4_dir()).unwrap();
    let opts = super::store::Options::from_env(Some(16_384), true).unwrap();
    let store = super::store::Store::open(&paths, &opts).unwrap();
    let m = MtpLayer::load(&store.scan, &store.shards, &store.model).unwrap();
    assert_eq!(m.bytes(), super::store::mtp_bytes(&store.scan).unwrap());
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
    println!(
        "AR: tps={:.2}",
        ar.performance
            .as_ref()
            .map_or(0.0, |p| p.decode_tokens_per_second)
    );
    for depth in 1..=3 {
        inner.reset_caches(ResetScope::Command).unwrap();
        let r = crate::engine::session::session_start(&mut inner, msgs(), cfg(true, depth), &flag)
            .unwrap();
        let p = r.performance.as_ref();
        println!(
            "depth {depth}: {:?} perf {:?} tps={:.2}",
            r.text,
            p.map(|p| (p.mtp_cycles, p.mtp_mean_accepted_tokens)),
            p.map_or(0.0, |p| p.decode_tokens_per_second)
        );
        assert_eq!(r.text, ar.text, "depth {depth}");
        assert!(p.and_then(|p| p.mtp_cycles).is_some_and(|c| c > 0));
    }
}
