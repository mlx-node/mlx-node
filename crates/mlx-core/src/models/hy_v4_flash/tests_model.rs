//! Tiny synthetic trunk covering sliding & indexed layers, Engram, Q4_K & Q6_K
//! expert bank (streamed / direct / mmap / prefetch), BF16 head, multi-owner
//! decode waves, rollback/restore, circular window wrap, and Engram lookahead.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::bank::{KqBank, KqStats, Residency};
use super::cache::{WINDOW_SLACK, with_row_dtype};
use super::config::HyV4FlashConfig;
use super::engram::Hasher;
use super::ladder::Ladder;
use super::model::{Logits, Model, Output, OwnerRows, Session, Trace, Want, call_engram_rows};
use super::moe::{ExpertSource, RowKind, forced, with_prefill_indirect_gu};
use super::ops::head_logits;
use super::prefill::{Driver, Grid, Prefilled};
use super::rows::{EngramRows, F8Table, PrefetchHandle};
use super::scan::{ExpertLayer, Located};
use super::stats::{self, PHASES, Phase, phase_seconds, set_profiling};
#[allow(unused_imports)]
pub(super) use super::test_support::tiny_cfg;
use super::test_support::{
    DenseKind, Rng, TINY_CLASSES, TINY_CTX, TINY_D, TINY_V, Temp, TinyModelFixture,
    TinyWeightsOpts, arr, bank_fixture, fixture, kquant_layer, max_abs_err, read_bits, read_f32,
    real_scan, tiny_engram_cfg, tiny_engram_table, tiny_model_fixture, tiny_q8_cfg,
    tiny_weights_opts,
};
use super::weights::{Dense, Resident};
use crate::array::{DType, MxArray};
use crate::moe_stream::io::{IoPool, ShardSet};
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};

enum BankKind {
    Streamed { cache: [usize; 2], staging: usize },
    Direct,
    Mmap,
}

/// Shared synthetic K-quant shard file spawning streamed, direct, or mmap `ModelRig`s.
struct ShardRig {
    _dir: Temp,
    path: PathBuf,
    len: u64,
    cfg: HyV4FlashConfig,
    experts: Vec<Option<ExpertLayer>>,
}

impl ShardRig {
    fn new(cfg: HyV4FlashConfig, align: usize, seed: u64) -> Self {
        let mut r = Rng::new(seed);
        let mut file = r.bytes(if align > 1 { 1024 } else { 1000 });
        let experts = TINY_CLASSES
            .iter()
            .enumerate()
            .map(|(il, c)| c.map(|c| kquant_layer(&mut r, &cfg.moe, il, c, align, &mut file)))
            .collect();
        let dir = Temp::new("model");
        let len = file.len() as u64;
        let path = dir.write("experts.bin", &file);
        Self {
            _dir: dir,
            path,
            len,
            cfg,
            experts,
        }
    }

    fn model(
        &self,
        kind: BankKind,
        opts: TinyWeightsOpts,
        engram: Box<dyn EngramRows>,
    ) -> ModelRig {
        let shards = Arc::new(ShardSet::open(&[self.path.as_path()]).unwrap());
        let pool = IoPool::new(shards, 4).unwrap();
        let bank = Arc::new(match kind {
            BankKind::Streamed { cache, staging } => {
                KqBank::new(pool, &self.cfg.moe, &self.experts, cache, staging).unwrap()
            }
            BankKind::Direct => KqBank::new_direct(pool, &self.cfg.moe, &self.experts).unwrap(),
            BankKind::Mmap => {
                let ps = std::slice::from_ref(&self.path);
                KqBank::new_mmap(pool, &self.cfg.moe, &self.experts, ps, self.len, false).unwrap()
            }
        });
        let b = Arc::clone(&bank);
        let w = Box::new(tiny_weights_opts(&self.cfg, opts));
        let model = Model::new(
            self.cfg.clone(),
            w,
            move |_| Ok(ExpertSource::Bank(Arc::clone(&b))),
            engram,
        )
        .unwrap();
        ModelRig {
            model,
            bank,
            _dir: None,
        }
    }

    fn with_table(&self, kind: BankKind, opts: TinyWeightsOpts, table_seed: u64) -> ModelRig {
        self.model(
            kind,
            opts,
            Box::new(tiny_engram_table(&self.cfg.engram, table_seed)),
        )
    }
}

struct ModelRig {
    model: Model,
    bank: Arc<KqBank>,
    _dir: Option<Temp>,
}

impl ModelRig {
    fn q8_bf16() -> Self {
        let cfg = tiny_q8_cfg();
        let (bank, _, dir, _) = bank_fixture(&cfg.moe, &TINY_CLASSES, [4, 4], 0, 29);
        let opts = TinyWeightsOpts {
            seed: 31,
            dense: DenseKind::SymQ8,
            shared_ff: 32,
            router_bias: true,
        };
        let table = tiny_engram_table(&cfg.engram, 37);
        let b = Arc::clone(&bank);
        let model = Model::new(
            cfg.clone(),
            Box::new(tiny_weights_opts(&cfg, opts)),
            move |_| Ok(ExpertSource::Bank(Arc::clone(&b))),
            Box::new(table),
        )
        .unwrap();
        Self {
            model,
            bank,
            _dir: Some(dir),
        }
    }

    fn pf_delta(&self, before: &[KqStats; 2]) -> [usize; 2] {
        let a = self.bank.stats();
        [
            a[0].prefetch_slots - before[0].prefetch_slots,
            a[1].prefetch_slots - before[1].prefetch_slots,
        ]
    }
}

impl Drop for ModelRig {
    fn drop(&mut self) {
        let _ = self.bank.end_turn();
    }
}

fn model_fixture() -> TinyModelFixture {
    tiny_model_fixture(13)
}
fn prompt() -> Vec<u32> {
    Rng::new(5).ids(24, TINY_V)
}
fn make_prompts(specs: &[(u64, usize)]) -> Vec<Vec<u32>> {
    specs
        .iter()
        .map(|&(s, n)| Rng::new(s).ids(n, TINY_V))
        .collect()
}

fn fwd(m: &Model, s: &mut Session, ids: &[u32], kind: RowKind, want: Want) -> Output {
    m.forward(s, ids, kind, want, None, None).unwrap()
}

fn fwd_traced(
    m: &Model,
    s: &mut Session,
    ids: &[u32],
    kind: RowKind,
    want: Want,
) -> (Output, Trace) {
    let mut t = Trace::default();
    let out = m.forward(s, ids, kind, want, None, Some(&mut t)).unwrap();
    (out, t)
}

fn fwd_last(m: &Model, s: &mut Session, ids: &[u32], kind: RowKind) -> Vec<f32> {
    read_f32(&m.forward_last(s, ids, kind, None).unwrap())
}

fn step_wave(
    model: &Model,
    sessions: &mut [Session],
    ids: &[&[u32]],
    wants: &[Want],
    trace: Option<&mut Trace>,
) -> (Vec<Output>, u64) {
    let before = stats::thread_evals();
    let mut owners: Vec<OwnerRows<'_>> = sessions
        .iter_mut()
        .zip(ids)
        .zip(wants)
        .map(|((sess, &ids), &want)| OwnerRows { sess, ids, want })
        .collect();
    let outs = model
        .step(&mut owners, RowKind::Decode, None, trace)
        .unwrap();
    (outs, stats::thread_evals() - before)
}

fn run_split(model: &Model, ids: &[u32], cuts: &[usize]) -> (Vec<Vec<f32>>, Vec<Vec<u32>>) {
    assert_eq!(cuts.iter().sum::<usize>(), ids.len());
    let mut sess = model.new_session(TINY_CTX).unwrap();
    let (mut rows, mut routes, mut at) = (Vec::new(), vec![Vec::new(); model.cfg().n_layers], 0);
    for &n in cuts {
        let kind = if n == 1 && at > 0 {
            RowKind::Decode
        } else {
            RowKind::Prefill
        };
        let (out, t) = fwd_traced(model, &mut sess, &ids[at..at + n], kind, Want::ALL);
        rows.extend(logit_rows(out.logits.as_ref().unwrap()));
        for (l, r) in t.routes.into_iter().enumerate() {
            if let Some(r) = r {
                routes[l].extend(r.ids);
            }
        }
        at += n;
    }
    assert_eq!(sess.len(), ids.len());
    assert!(sess.layers().iter().all(|st| st.len() == ids.len()));
    (rows, routes)
}

fn assert_close(got: &[Vec<f32>], want: &[Vec<f32>], what: &str) {
    assert_within(got, want, what, 1e-4, 1e-4);
}

fn assert_within(got: &[Vec<f32>], want: &[Vec<f32>], what: &str, abs: f32, rel: f32) {
    assert_eq!(got.len(), want.len(), "{what}: rows");
    for (t, (g, w)) in got.iter().zip(want).enumerate() {
        for (j, (&a, &b)) in g.iter().zip(w).enumerate() {
            assert!(a.is_finite(), "{what}: row {t} col {j} not finite");
            assert!(
                (a - b).abs() <= abs + rel * b.abs(),
                "{what}: row {t} col {j}: {a} vs {b}"
            );
        }
    }
}

fn assert_near_arr(a: &MxArray, b: &MxArray, label: &str) {
    let diff = max_abs_err(&read_f32(a), &read_f32(b));
    assert!(diff <= 1e-2, "{label}: max_diff={diff}");
}

fn decode(model: &Model, sess: &mut Session, ids: &[u32]) -> Vec<Vec<f32>> {
    ids.iter()
        .map(|&id| fwd_last(model, sess, &[id], RowKind::Decode))
        .collect()
}

fn prefilled(model: &Model, ids: &[u32]) -> Session {
    let mut s = model.new_session(TINY_CTX).unwrap();
    fwd(model, &mut s, ids, RowKind::Prefill, Want::STATE);
    s
}

fn routes_of(t: &Trace) -> Vec<Vec<u32>> {
    t.routes
        .iter()
        .map(|r| r.as_ref().map_or_else(Vec::new, |r| r.ids.clone()))
        .collect()
}

fn logit_rows(x: &MxArray) -> Vec<Vec<f32>> {
    read_f32(x).chunks(TINY_V).map(<[f32]>::to_vec).collect()
}

fn prefill_mma_tol() -> f32 {
    let on = unsafe { mlx_sys::mlx_metal_is_available() }
        && std::env::var("MLX_HY4_PREFILL_NAX").as_deref() != Ok("0");
    if on { 5e-3 } else { 1e-4 }
}

#[test]
fn decode_wave_matches_each_owner_alone() {
    let f = model_fixture();
    let prompts = make_prompts(&[(5, 9), (6, 14), (8, 20)]);
    let next: [&[u32]; 3] = [&[7], &[3, 41], &[60]];
    let alone: Vec<_> = prompts
        .iter()
        .zip(next)
        .map(|(p, ids)| {
            let mut s = prefilled(&f.model, p);
            let (out, t) = fwd_traced(&f.model, &mut s, ids, RowKind::Decode, Want::ALL);
            (logit_rows(out.logits.as_ref().unwrap()), routes_of(&t))
        })
        .collect();
    let mut sessions: Vec<Session> = prompts.iter().map(|p| prefilled(&f.model, p)).collect();
    let mut t = Trace::default();
    let (outs, _) = step_wave(
        &f.model,
        &mut sessions,
        &next,
        &[Want::ALL; 3],
        Some(&mut t),
    );
    assert_eq!(outs.len(), 3);
    let (wave, k, mut row) = (routes_of(&t), f.model.cfg().moe.n_used, 0);
    for (i, ((want, want_routes), out)) in alone.iter().zip(&outs).enumerate() {
        let n = next[i].len();
        assert_close(
            &logit_rows(out.logits.as_ref().unwrap()),
            want,
            &format!("owner {i}"),
        );
        for (l, r) in wave.iter().enumerate() {
            if !r.is_empty() {
                assert_eq!(
                    r[row * k..(row + n) * k],
                    want_routes[l][..],
                    "owner {i} layer {l}"
                );
            }
        }
        assert_eq!(sessions[i].len(), prompts[i].len() + n);
        assert_eq!(&sessions[i].tokens()[prompts[i].len()..], next[i]);
        row += n;
    }
}

#[test]
fn refused_waves_leave_every_session() {
    let f = model_fixture();
    let (pa, pb) = (Rng::new(5).ids(12, TINY_V), Rng::new(6).ids(10, TINY_V));
    let (mut a, mut b) = (prefilled(&f.model, &pa), prefilled(&f.model, &pb));
    let step = |a: &mut Session,
                b: &mut Session,
                ids: [&[u32]; 2],
                kind,
                cancel: Option<&dyn Fn(usize) -> bool>| {
        let mut owners = [
            OwnerRows {
                sess: a,
                ids: ids[0],
                want: Want::LAST,
            },
            OwnerRows {
                sess: b,
                ids: ids[1],
                want: Want::LAST,
            },
        ];
        f.model.step(&mut owners, kind, cancel, None).map(|outs| {
            outs.iter()
                .map(|o| read_f32(o.logits.as_ref().unwrap()))
                .collect::<Vec<_>>()
        })
    };
    let lens = |s: &Session| {
        (
            s.len(),
            s.layers().iter().map(|st| st.len()).collect::<Vec<_>>(),
        )
    };
    let (la, lb) = (lens(&a), lens(&b));
    assert!(
        step(
            &mut a,
            &mut b,
            [&[1, 2][..], &[3][..]],
            RowKind::Prefill,
            None
        )
        .is_err()
    );
    assert!(
        step(
            &mut a,
            &mut b,
            [&[1][..], &[TINY_V as u32][..]],
            RowKind::Decode,
            None
        )
        .is_err()
    );
    let at_two = |l: usize| l == 2;
    assert!(
        step(
            &mut a,
            &mut b,
            [&[1][..], &[3][..]],
            RowKind::Decode,
            Some(&at_two)
        )
        .is_err()
    );
    assert_eq!((lens(&a), lens(&b)), (la, lb));
    let got = step(&mut a, &mut b, [&[1][..], &[3][..]], RowKind::Decode, None).unwrap();
    for (i, (p, id)) in [(&pa, 1u32), (&pb, 3)].into_iter().enumerate() {
        let mut s = prefilled(&f.model, p);
        assert_close(
            &[got[i].clone()],
            &[fwd_last(&f.model, &mut s, &[id], RowKind::Decode)],
            &format!("owner {i}"),
        );
    }
}

#[test]
fn pad_positions_ignore_the_engram_table() {
    let (f13, f17) = (tiny_model_fixture(13), tiny_model_fixture(17));
    let run = |f: &TinyModelFixture, ids: &[u32]| {
        let mut s = f.model.new_session(TINY_CTX).unwrap();
        fwd_last(&f.model, &mut s, ids, RowKind::Prefill)
    };
    let pads = [(TINY_V - 1) as u32; 5];
    assert_eq!(run(&f13, &pads), run(&f17, &pads));
    assert_ne!(
        run(&f13, &[3, 17, 40, 2, 61]),
        run(&f17, &[3, 17, 40, 2, 61])
    );
}

#[test]
fn engram_rows_hash_only_the_reach() {
    let cfg = tiny_engram_cfg(TINY_V);
    let hasher = Hasher::new(&cfg).unwrap();
    let all = Rng::new(3).ids(30, TINY_V);
    for s in [0usize, 1, 2, 3, 7, 29] {
        let got = call_engram_rows(&hasher, cfg.max_ngram - 1, &all[..s], &all[s..]).unwrap();
        assert_eq!(got, hasher.call_rows(&all, s).unwrap(), "split at {s}");
    }
}

#[test]
fn split_calls_match_one_call() {
    let (f, ids) = (model_fixture(), prompt());
    let dec = |k: usize| {
        std::iter::once(k)
            .chain(std::iter::repeat_n(1, 24 - k))
            .collect::<Vec<_>>()
    };
    let patterns = [
        dec(12),
        dec(1),
        vec![7, 9, 8],
        vec![16, 8],
        vec![8, 1, 15],
        vec![3, 5, 4, 12],
    ];
    let check = |rows: &str, abs: f32, rel: f32| {
        let (want, want_routes) = run_split(&f.model, &ids, &[24]);
        for cuts in &patterns {
            let (got, routes) = run_split(&f.model, &ids, cuts);
            let what = format!("{rows} rows, cuts {cuts:?}");
            assert_within(&got, &want, &what, abs, rel);
            assert_eq!(routes, want_routes, "{what}: routes");
        }
    };
    let f32_tol = prefill_mma_tol();
    with_row_dtype(DType::Float32, || check("f32", f32_tol, f32_tol));
    let f16_step = 2f32.powi(-10);
    check("f16", (2.0 * f16_step).max(f32_tol), f16_step.max(f32_tol));
}

#[test]
fn last_row_and_hidden_rows() {
    let (f, ids) = (model_fixture(), prompt());
    let (all, _) = run_split(&f.model, &ids, &[24]);

    let mut a = f.model.new_session(TINY_CTX).unwrap();
    let last = fwd_last(&f.model, &mut a, &ids, RowKind::Prefill);
    assert_close(std::slice::from_ref(&last), &all[23..], "last row");

    let mut b = f.model.new_session(TINY_CTX).unwrap();
    let want_lh = Want {
        logits: Logits::Last,
        hidden: true,
    };
    let out = fwd(&f.model, &mut b, &ids, RowKind::Prefill, want_lh);
    assert_eq!(
        read_f32(out.logits.as_ref().unwrap()),
        last,
        "hidden changed logits"
    );
    let hidden = out.hidden.unwrap();
    assert_eq!(hidden.shape().unwrap().as_ref(), &[24, TINY_D as i64]);
    let rows = logit_rows(&head_logits(&hidden, f.model.weights().output()).unwrap());
    assert_close(&rows, &all, "head of the hidden rows");

    let mut c = f.model.new_session(TINY_CTX).unwrap();
    let Output { logits, hidden } =
        fwd(&f.model, &mut c, &ids[..20], RowKind::Prefill, Want::STATE);
    assert!(logits.is_none() && hidden.is_none() && c.len() == 20);
    let tail = decode(&f.model, &mut c, &ids[20..]);
    let tol = prefill_mma_tol();
    assert_within(&tail, &all[20..], "decode after state-only", tol, tol);
}

#[test]
fn failed_call_leaves_the_session() {
    let (f, ids) = (model_fixture(), prompt());
    let mut clean = prefilled(&f.model, &ids[..20]);
    let want = fwd_last(&f.model, &mut clean, &ids[20..], RowKind::Prefill);

    let mut sess = prefilled(&f.model, &ids[..20]);
    for stop in 0..4usize {
        let cancel = move |l: usize| l == stop;
        assert!(
            f.model
                .forward_last(&mut sess, &ids[20..], RowKind::Prefill, Some(&cancel))
                .is_err()
        );
        assert!(sess.len() == 20 && sess.layers().iter().all(|st| st.len() == 20));
    }
    assert!(
        f.model
            .forward_last(&mut sess, &[TINY_V as u32], RowKind::Decode, None)
            .is_err()
    );
    assert!(
        f.model
            .forward_last(&mut sess, &[], RowKind::Prefill, None)
            .is_err()
    );
    let past = vec![1u32; TINY_CTX - 19];
    assert!(
        f.model
            .forward_last(&mut sess, &past, RowKind::Prefill, None)
            .is_err()
    );
    assert_eq!(sess.len(), 20);
    assert_eq!(
        fwd_last(&f.model, &mut sess, &ids[20..], RowKind::Prefill),
        want
    );
}

#[test]
fn mark_restore_replays_exactly() {
    let (f, ids) = (model_fixture(), prompt());
    let mut sess = prefilled(&f.model, &ids[..14]);
    let mark = sess.mark();
    assert_eq!(mark.len(), 14);
    let first = decode(&f.model, &mut sess, &ids[14..]);
    sess.restore(&mark).unwrap();
    assert_eq!(sess.len(), 14);
    assert_eq!(decode(&f.model, &mut sess, &ids[14..]), first);
    let later = sess.mark();
    sess.restore(&mark).unwrap();
    assert!(sess.restore(&later).is_err(), "a mark past the session");
}

#[test]
fn restore_refuses_a_rewritten_mark() {
    let (f, ids) = (model_fixture(), prompt());
    let mut sess = prefilled(&f.model, &ids[..14]);
    let early = sess.mark();
    decode(&f.model, &mut sess, &ids[14..]);
    let late = sess.mark();
    sess.restore(&early).unwrap();
    let other = vec![5u32; 10];
    let want = decode(&f.model, &mut sess, &other);
    assert_eq!(sess.len(), 24);
    assert!(sess.restore(&late).is_err(), "restored over rewritten rows");
    assert_eq!((sess.len(), &sess.tokens()[14..]), (24, other.as_slice()));
    sess.restore(&early).unwrap();
    assert_eq!(sess.len(), 14);
    assert_eq!(decode(&f.model, &mut sess, &other), want);
}

#[test]
fn profiled_call_times_every_trunk_phase() {
    let (f, ids) = (model_fixture(), prompt());
    let mut plain = f.model.new_session(TINY_CTX).unwrap();
    let want = fwd_last(&f.model, &mut plain, &ids, RowKind::Prefill);
    let before = phase_seconds();
    set_profiling(true);
    let mut sess = f.model.new_session(TINY_CTX).unwrap();
    let got = f
        .model
        .forward_last(&mut sess, &ids, RowKind::Prefill, None);
    set_profiling(false);
    let after = phase_seconds();
    assert_close(&[read_f32(&got.unwrap())], &[want], "profiled logits");
    for (i, &p) in PHASES.iter().enumerate() {
        if p != Phase::Mtp {
            assert!(after[i] > before[i], "{} not timed", p.name());
        }
    }
}

#[test]
fn truncate_within_the_slack_replays_exactly() {
    let (f, ids) = (model_fixture(), prompt());
    let mut sess = prefilled(&f.model, &ids[..20]);
    let first = decode(&f.model, &mut sess, &ids[20..]);

    sess.truncate(21).unwrap();
    assert!(sess.len() == 21 && sess.layers().iter().all(|st| st.len() == 21));
    assert_eq!(decode(&f.model, &mut sess, &ids[21..]), first[1..].to_vec());

    assert!(sess.truncate(10).is_err(), "past window slack");
    assert!(sess.len() == 24 && sess.layers().iter().all(|st| st.len() == 24));
    let mut clean = prefilled(&f.model, &ids[..20]);
    decode(&f.model, &mut clean, &ids[20..]);
    assert_eq!(
        decode(&f.model, &mut sess, &[3]),
        decode(&f.model, &mut clean, &[3])
    );

    sess.truncate(0).unwrap();
    assert!(sess.is_empty());
    let mut fresh = f.model.new_session(TINY_CTX).unwrap();
    assert_eq!(
        fwd_last(&f.model, &mut sess, &ids, RowKind::Prefill),
        fwd_last(&f.model, &mut fresh, &ids, RowKind::Prefill),
    );
}

#[test]
fn new_refuses_mismatched_weights() {
    let cfg = tiny_cfg();
    let (bank, _, _dir, _) = bank_fixture(&cfg.moe, &TINY_CLASSES, [4, 4], 0, 7);
    let build = |w| {
        let b = Arc::clone(&bank);
        let tbl = Box::new(tiny_engram_table(&tiny_cfg().engram, 13));
        Model::new(
            tiny_cfg(),
            Box::new(w),
            move |_| Ok(ExpertSource::Bank(Arc::clone(&b))),
            tbl,
        )
    };
    let fresh = || tiny_weights_opts(&cfg, TinyWeightsOpts::default());

    let mut w = fresh();
    w.layers[1].engram = None;
    assert!(build(w).is_err(), "Engram layer without weights");
    let mut w = fresh();
    w.layers[0].indexer = fresh().layers[1].indexer.take();
    assert!(build(w).is_err(), "indexer on sliding layer");
    let mut w = fresh();
    w.layers.pop();
    assert!(build(w).is_err(), "missing layer");
    let mut w = fresh();
    w.output = Dense::Array(arr(
        &vec![0.0; (TINY_V - 1) * TINY_D],
        &[TINY_V - 1, TINY_D],
    ));
    assert!(build(w).is_err(), "head rows != vocab");

    let ok = build(fresh()).unwrap();
    assert!(ok.new_session(0).is_err() && ok.new_session(TINY_CTX + 1).is_err());
    bank.end_turn().unwrap();
}

#[test]
fn batched_bf16_head_four_owner_decode_wave_matches_alone_bitwise() {
    let f = ModelRig::q8_bf16();
    let prompts = make_prompts(&[(5, 9), (6, 14), (8, 18), (9, 22)]);

    let single: [&[u32]; 4] = [&[7], &[3], &[41], &[60]];
    let mut evals_1 = 0u64;
    let alone_last: Vec<Vec<u32>> = prompts
        .iter()
        .zip(single)
        .enumerate()
        .map(|(i, (p, ids))| {
            let mut s = prefilled(&f.model, p);
            let ev = stats::thread_evals();
            let out = fwd(&f.model, &mut s, ids, RowKind::Decode, Want::LAST);
            if i == 0 {
                evals_1 = stats::thread_evals() - ev;
            }
            read_bits(out.logits.as_ref().unwrap())
        })
        .collect();

    let mut sessions: Vec<Session> = prompts.iter().map(|p| prefilled(&f.model, p)).collect();
    let (outs, evals_4) = step_wave(&f.model, &mut sessions, &single, &[Want::LAST; 4], None);
    assert_eq!(evals_4, evals_1, "4-owner BF16-head wave evals");
    for (i, (want, out)) in alone_last.iter().zip(&outs).enumerate() {
        assert_eq!(read_bits(out.logits.as_ref().unwrap()), *want, "owner {i}");
        assert!(out.hidden.is_none());
    }

    let mixed_ids: [&[u32]; 4] = [&[11], &[12, 13], &[14], &[15, 16]];
    let mixed_wants = [
        Want::LAST,
        Want::ALL,
        Want::STATE,
        Want {
            logits: Logits::Last,
            hidden: true,
        },
    ];
    let alone_mixed: Vec<_> = prompts
        .iter()
        .zip(mixed_ids)
        .zip(mixed_wants)
        .map(|((p, ids), want)| {
            let mut s = prefilled(&f.model, p);
            let out = fwd(&f.model, &mut s, ids, RowKind::Decode, want);
            (
                out.logits.as_ref().map(read_bits),
                out.hidden.as_ref().map(read_bits),
            )
        })
        .collect();
    let mut sessions: Vec<Session> = prompts.iter().map(|p| prefilled(&f.model, p)).collect();
    let (outs, evals_mixed) = step_wave(&f.model, &mut sessions, &mixed_ids, &mixed_wants, None);
    assert_eq!(evals_mixed, evals_1, "mixed-wants 4-owner wave evals");
    for (i, ((wl, wh), out)) in alone_mixed.iter().zip(&outs).enumerate() {
        assert_eq!(
            out.logits.as_ref().map(read_bits),
            *wl,
            "mixed owner {i} logits"
        );
        assert_eq!(
            out.hidden.as_ref().map(read_bits),
            *wh,
            "mixed owner {i} hidden"
        );
    }
}

#[test]
fn prefill_with_prefetch_matches_without_prefetch_bitwise() {
    let mut cfg = tiny_cfg();
    cfg.context_length = 300;
    let shard = ShardRig::new(cfg, 0, 7);
    let rig = shard.with_table(
        BankKind::Streamed {
            cache: [4, 4],
            staging: 2,
        },
        Default::default(),
        13,
    );
    let ids = Rng::new(53).ids(super::block::READ_AHEAD_MIN_ROWS, TINY_V);

    let run = |on: bool| {
        rig.bank.end_turn().unwrap();
        rig.bank.set_prefetch(on);
        rig.bank.test_forget_where(|_, _| true);
        let st = rig.bank.stats();
        let mut s = rig.model.new_session(300).unwrap();
        let (out, t) = fwd_traced(&rig.model, &mut s, &ids, RowKind::Prefill, Want::LAST);
        (
            read_bits(out.logits.as_ref().unwrap()),
            routes_of(&t),
            rig.pf_delta(&st),
        )
    };

    let (want_bits, routes_off, pf_off) = run(false);
    assert_eq!(pf_off, [0, 0]);
    let (got_bits, routes_on, pf_on) = run(true);
    assert_eq!((got_bits, routes_on), (want_bits, routes_off));
    assert!(
        pf_on[0] > 0 && pf_on[1] > 0,
        "both classes must prefetch, got {pf_on:?}"
    );
}

#[test]
fn decode_with_prefetch_matches_without_prefetch_bitwise_and_zero_extra_evals() {
    let shard = ShardRig::new(tiny_cfg(), 0, 7);
    let rig = shard.with_table(
        BankKind::Streamed {
            cache: [4, 4],
            staging: 2,
        },
        Default::default(),
        13,
    );
    let (prefill_ids, decode_ids) = (Rng::new(19).ids(12, TINY_V), [7u32, 23, 41, 60]);

    let run_seq = |on: bool| {
        rig.bank.end_turn().unwrap();
        rig.bank.set_decode_prefetch(on);
        let mut s = prefilled(&rig.model, &prefill_ids);
        rig.bank.test_forget_where(|_, _| true);
        let (st, ev) = (rig.bank.stats(), stats::thread_evals());
        let (mut all_bits, mut all_routes) = (Vec::new(), Vec::new());
        for &id in &decode_ids {
            let (out, t) = fwd_traced(&rig.model, &mut s, &[id], RowKind::Decode, Want::LAST);
            all_bits.push(read_bits(out.logits.as_ref().unwrap()));
            all_routes.push(routes_of(&t));
        }
        (
            all_bits,
            all_routes,
            stats::thread_evals() - ev,
            rig.pf_delta(&st),
        )
    };
    let (b_off, r_off, ev_off, pf_off) = run_seq(false);
    let (b_on, r_on, ev_on, pf_on) = run_seq(true);
    assert_eq!((pf_off, b_on, r_on, ev_on), ([0, 0], b_off, r_off, ev_off));
    assert!(
        pf_on[0] > 0 && pf_on[1] > 0,
        "M=1 decode_prefetch: {pf_on:?}"
    );

    // Multi-row decode wave (3 rows <= 4, exercises `forward_split`).
    let wave_prompts = make_prompts(&[(31, 10), (37, 14)]);
    let run_wave = |on: bool| {
        rig.bank.end_turn().unwrap();
        rig.bank.set_decode_prefetch(on);
        let mut sess: Vec<Session> = wave_prompts
            .iter()
            .map(|p| prefilled(&rig.model, p))
            .collect();
        rig.bank.test_forget_where(|_, _| true);
        let st = rig.bank.stats();
        let (outs, ev) = step_wave(
            &rig.model,
            &mut sess,
            &[&[11][..], &[12, 13][..]],
            &[Want::LAST, Want::ALL],
            None,
        );
        let out_bits: Vec<_> = outs
            .iter()
            .map(|o| read_bits(o.logits.as_ref().unwrap()))
            .collect();
        (out_bits, ev, rig.pf_delta(&st))
    };
    let (w_off, we_off, wpf_off) = run_wave(false);
    let (w_on, we_on, wpf_on) = run_wave(true);
    assert_eq!((wpf_off, w_on, we_on), ([0, 0], w_off, we_off));
    assert!(
        wpf_on[0] > 0 && wpf_on[1] > 0,
        "M=3 wave prefetch: {wpf_on:?}"
    );

    // Wide decode wave (5 rows > 4) must not prefetch.
    rig.bank.end_turn().unwrap();
    rig.bank.set_decode_prefetch(true);
    let mut wide_s = prefilled(&rig.model, &wave_prompts[0]);
    rig.bank.test_forget_where(|_, _| true);
    let st = rig.bank.stats();
    fwd(
        &rig.model,
        &mut wide_s,
        &[1, 2, 3, 4, 5],
        RowKind::Decode,
        Want::LAST,
    );
    assert_eq!(
        rig.pf_delta(&st),
        [0, 0],
        "waves past 4 rows must skip read-ahead"
    );

    // Single-staging bank (`staging_layers == 1`) must never decode-prefetch.
    let single = model_fixture();
    single.bank.set_decode_prefetch(true);
    let mut ss = prefilled(&single.model, &prefill_ids);
    single.bank.test_forget_where(|_, _| true);
    let st = single.bank.stats();
    fwd(&single.model, &mut ss, &[7], RowKind::Decode, Want::LAST);
    let st2 = single.bank.stats();
    assert_eq!(
        (st2[0].prefetch_slots - st[0].prefetch_slots)
            + (st2[1].prefetch_slots - st[1].prefetch_slots),
        0
    );
}

#[test]
fn direct_mode_decode_wave_matches_partial_bitwise_with_single_eval() {
    let shard = ShardRig::new(tiny_cfg(), 0, 7);
    let partial = shard.with_table(
        BankKind::Streamed {
            cache: [4, 4],
            staging: 1,
        },
        Default::default(),
        13,
    );
    let direct = shard.with_table(BankKind::Direct, Default::default(), 13);
    direct.bank.wait_direct_fill().unwrap();
    assert!(!partial.model.bank_is_direct() && direct.model.bank_is_direct());

    let prompts = make_prompts(&[(5, 9), (6, 14), (8, 18), (9, 22)]);
    let mut p_sess = partial.model.new_session(TINY_CTX).unwrap();
    let mut d_sess = direct.model.new_session(TINY_CTX).unwrap();
    let p_pre = fwd(
        &partial.model,
        &mut p_sess,
        &prompts[0],
        RowKind::Prefill,
        Want::LAST,
    );
    let d_pre = fwd(
        &direct.model,
        &mut d_sess,
        &prompts[0],
        RowKind::Prefill,
        Want::LAST,
    );
    assert_eq!(
        read_bits(d_pre.logits.as_ref().unwrap()),
        read_bits(p_pre.logits.as_ref().unwrap())
    );

    let ev = stats::thread_evals();
    let p_dec1 = fwd(
        &partial.model,
        &mut p_sess,
        &[7],
        RowKind::Decode,
        Want::LAST,
    );
    assert!(
        stats::thread_evals() - ev > 1,
        "partial mode must sync on router reads"
    );

    let ev = stats::thread_evals();
    let d_dec1 = fwd(
        &direct.model,
        &mut d_sess,
        &[7],
        RowKind::Decode,
        Want::LAST,
    );
    assert_eq!(
        stats::thread_evals() - ev,
        1,
        "direct 1-row decode must use 1 eval"
    );
    assert_eq!(
        read_bits(d_dec1.logits.as_ref().unwrap()),
        read_bits(p_dec1.logits.as_ref().unwrap())
    );

    let mut p_sesss: Vec<Session> = prompts
        .iter()
        .map(|p| prefilled(&partial.model, p))
        .collect();
    let mut d_sesss: Vec<Session> = prompts
        .iter()
        .map(|p| prefilled(&direct.model, p))
        .collect();
    let wave_ids: [&[u32]; 4] = [&[11], &[12, 13], &[14], &[15, 16]];
    let wave_wants = [
        Want::LAST,
        Want::ALL,
        Want::LAST,
        Want {
            logits: Logits::All,
            hidden: true,
        },
    ];
    let (p_outs, _) = step_wave(&partial.model, &mut p_sesss, &wave_ids, &wave_wants, None);
    let (d_outs, d_ev) = step_wave(&direct.model, &mut d_sesss, &wave_ids, &wave_wants, None);
    assert_eq!(
        d_ev, 1,
        "direct multi-owner wave must use 1 eval, got {d_ev}"
    );
    for (i, (po, do_)) in p_outs.iter().zip(&d_outs).enumerate() {
        assert_eq!(
            do_.logits.as_ref().map(read_bits),
            po.logits.as_ref().map(read_bits),
            "owner {i} logits"
        );
        assert_eq!(
            do_.hidden.as_ref().map(read_bits),
            po.hidden.as_ref().map(read_bits),
            "owner {i} hidden"
        );
    }
}

#[test]
fn mmap_mode_decode_and_prefill_match_direct_bank() {
    // SAFETY: pure process-wide Metal availability probe.
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let shard = ShardRig::new(tiny_cfg(), 32, 23);
    let direct = shard.with_table(BankKind::Direct, Default::default(), 13);
    let mmap = shard.with_table(BankKind::Mmap, Default::default(), 13);
    direct.bank.wait_direct_fill().unwrap();
    mmap.bank.wait_direct_fill().unwrap();

    assert!(mmap.bank.is_direct() && mmap.bank.is_mmap() && mmap.model.bank_is_direct());
    assert_eq!(
        (mmap.bank.nbytes(), mmap.bank.residency().unwrap()),
        (0, Residency::default())
    );
    assert!(mmap.bank.mapped_bytes() > 0);

    let prompts = make_prompts(&[(5, 9), (6, 14), (8, 18), (9, 22)]);
    let mut d_sess = direct.model.new_session(TINY_CTX).unwrap();
    let mut m_sess = mmap.model.new_session(TINY_CTX).unwrap();

    // 1. Prefill (<= 32 tiny-tail AND > 32 indirect NAX) matches DirectBank.
    let d_pre = fwd(
        &direct.model,
        &mut d_sess,
        &prompts[0],
        RowKind::Prefill,
        Want::LAST,
    );
    let m_pre = fwd(
        &mmap.model,
        &mut m_sess,
        &prompts[0],
        RowKind::Prefill,
        Want::LAST,
    );
    assert_near_arr(
        m_pre.logits.as_ref().unwrap(),
        d_pre.logits.as_ref().unwrap(),
        "tiny-tail prefill",
    );

    let long_p = Rng::new(42).ids(40, TINY_V);
    let (mut d_long, mut m_long) = (
        direct.model.new_session(TINY_CTX).unwrap(),
        mmap.model.new_session(TINY_CTX).unwrap(),
    );
    let dl = fwd(
        &direct.model,
        &mut d_long,
        &long_p,
        RowKind::Prefill,
        Want::LAST,
    );
    let ml = fwd(
        &mmap.model,
        &mut m_long,
        &long_p,
        RowKind::Prefill,
        Want::LAST,
    );
    assert_near_arr(
        ml.logits.as_ref().unwrap(),
        dl.logits.as_ref().unwrap(),
        ">32-row prefill",
    );

    // 2. Prefill non-NAX fallback (32-row decode chunks).
    let (mut d_fb, mut m_fb) = (
        direct.model.new_session(TINY_CTX).unwrap(),
        mmap.model.new_session(TINY_CTX).unwrap(),
    );
    let (do_fb, mo_fb) = with_prefill_indirect_gu(false, || {
        (
            fwd(
                &direct.model,
                &mut d_fb,
                &prompts[1],
                RowKind::Prefill,
                Want::LAST,
            ),
            fwd(
                &mmap.model,
                &mut m_fb,
                &prompts[1],
                RowKind::Prefill,
                Want::LAST,
            ),
        )
    });
    assert_near_arr(
        mo_fb.logits.as_ref().unwrap(),
        do_fb.logits.as_ref().unwrap(),
        "prefill fallback",
    );

    // 3. 1-row Decode: 1 eval and matches DirectBank.
    let d_dec1 = fwd(
        &direct.model,
        &mut d_sess,
        &[7],
        RowKind::Decode,
        Want::LAST,
    );
    let ev = stats::thread_evals();
    let m_dec1 = fwd(&mmap.model, &mut m_sess, &[7], RowKind::Decode, Want::LAST);
    assert_eq!(
        stats::thread_evals() - ev,
        1,
        "mmap 1-row decode must use 1 eval"
    );
    assert_near_arr(
        m_dec1.logits.as_ref().unwrap(),
        d_dec1.logits.as_ref().unwrap(),
        "1-row decode",
    );

    // 4. Multi-owner + verify2 Decode wave (6 rows, 4 owners): 1 eval and matches DirectBank.
    let mut d_sesss: Vec<Session> = prompts
        .iter()
        .map(|p| prefilled(&direct.model, p))
        .collect();
    let mut m_sesss: Vec<Session> = prompts.iter().map(|p| prefilled(&mmap.model, p)).collect();
    let wave_ids: [&[u32]; 4] = [&[11], &[12, 13], &[14], &[15, 16]];
    let wave_wants = [
        Want::LAST,
        Want::ALL,
        Want::LAST,
        Want {
            logits: Logits::All,
            hidden: true,
        },
    ];
    let (d_outs, _) = step_wave(&direct.model, &mut d_sesss, &wave_ids, &wave_wants, None);
    let (m_outs, m_ev) = step_wave(&mmap.model, &mut m_sesss, &wave_ids, &wave_wants, None);
    assert_eq!(m_ev, 1, "mmap multi-owner wave must use 1 eval, got {m_ev}");
    for (i, (do_, mo)) in d_outs.iter().zip(&m_outs).enumerate() {
        assert_near_arr(
            mo.logits.as_ref().unwrap(),
            do_.logits.as_ref().unwrap(),
            &format!("wave {i} logits"),
        );
        if let (Some(mh), Some(dh)) = (mo.hidden.as_ref(), do_.hidden.as_ref()) {
            assert_near_arr(mh, dh, &format!("wave {i} hidden"));
        }
    }

    // 5. Host-routed trace + forced routes work in Mmap mode.
    let forced_map: HashMap<usize, Vec<u32>> = TINY_CLASSES
        .iter()
        .enumerate()
        .filter_map(|(il, c)| c.map(|_| (il, vec![0u32, 3])))
        .collect();
    let ((df, d_tr), (mf, m_tr)) = forced::with(forced_map, || {
        (
            fwd_traced(
                &direct.model,
                &mut d_sess,
                &[21],
                RowKind::Decode,
                Want::LAST,
            ),
            fwd_traced(&mmap.model, &mut m_sess, &[21], RowKind::Decode, Want::LAST),
        )
    });
    assert_eq!(m_tr.routes.len(), d_tr.routes.len());
    assert_near_arr(
        mf.logits.as_ref().unwrap(),
        df.logits.as_ref().unwrap(),
        "forced decode",
    );
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_trunk_forward_last() {
    let scan = real_scan();
    let shards = Arc::new(ShardSet::open(&scan.paths).unwrap());
    let pool = IoPool::new(Arc::clone(&shards), 8).unwrap();
    let bank = Arc::new(KqBank::new(pool, &scan.cfg.moe, scan.trunk_experts(), [0, 0], 1).unwrap());
    let resident = Resident::load(&scan, &shards).unwrap();
    let table = F8Table::open(&scan).unwrap();
    let b = Arc::clone(&bank);
    let model = Model::new(
        scan.cfg.clone(),
        Box::new(resident),
        move |_| Ok(ExpertSource::Bank(Arc::clone(&b))),
        Box::new(table),
    )
    .unwrap();
    let vocab = scan.cfg.vocab_size;

    let stored: Option<serde_json::Value> = std::env::var("HY4_LLAMA_TOP1")
        .ok()
        .map(|p| serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap());
    let ids: Vec<u32> = match &stored {
        Some(v) => v["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect(),
        None => fixture("tokenizer-cases.json")["cases"][1]["ids"]
            .as_array()
            .unwrap()[..16]
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect(),
    };
    assert_eq!(ids.len(), 16);

    let mut a = model.new_session(64).unwrap();
    let all = read_f32(
        fwd(&model, &mut a, &ids, RowKind::Prefill, Want::ALL)
            .logits
            .as_ref()
            .unwrap(),
    );
    assert!(all.iter().all(|v| v.is_finite()), "non-finite logits");
    let rows: Vec<Vec<f32>> = all.chunks(vocab).map(<[f32]>::to_vec).collect();
    let mut b2 = model.new_session(64).unwrap();
    let last = fwd_last(&model, &mut b2, &ids, RowKind::Prefill);
    assert!(
        last.iter()
            .zip(&rows[15])
            .all(|(&g, &w)| (g - w).abs() <= 2e-3 + 2e-3 * w.abs())
    );
    let top1: Vec<u32> = rows
        .iter()
        .map(|r| {
            r.iter()
                .enumerate()
                .fold(
                    (0, f32::NEG_INFINITY),
                    |m, (i, &v)| if v > m.1 { (i, v) } else { m },
                )
                .0 as u32
        })
        .collect();
    eprintln!("hy4 real trunk top-1 per row: {top1:?}");
    if let Some(v) = stored {
        let want: Vec<u32> = v["top1"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(top1, want, "top-1 vs stored llama.cpp rows");
    }
    bank.end_turn().unwrap();
}

#[test]
fn circular_window_decode_wrap_truncate_and_restore_match_reference() {
    let (f, ids) = (model_fixture(), prompt());
    assert!(ids.len() > f.model.cfg().window + WINDOW_SLACK);

    let mut pre = f.model.new_session(TINY_CTX).unwrap();
    let want_prefill = fwd_last(&f.model, &mut pre, &ids, RowKind::Prefill);
    let mut clean = f.model.new_session(TINY_CTX).unwrap();
    let want_decode = decode(&f.model, &mut clean, &ids).pop().unwrap();

    let mut sess = f.model.new_session(TINY_CTX).unwrap();
    decode(&f.model, &mut sess, &ids[..18]);
    let mark = sess.mark();
    decode(&f.model, &mut sess, &ids[18..21]);
    sess.truncate(18).unwrap();
    assert_eq!(sess.len(), 18);

    let last = decode(&f.model, &mut sess, &ids[18..]).pop().unwrap();
    assert_eq!(last, want_decode, "wrapped circular ring + truncate");
    assert_within(
        &[last],
        &[want_prefill],
        "wrapped circular ring vs prefill",
        2e-3,
        2e-3,
    );

    sess.restore(&mark).unwrap();
    assert_eq!(
        decode(&f.model, &mut sess, &ids[18..]).pop().unwrap(),
        want_decode
    );
}

#[test]
fn cross_chunk_engram_lookahead_bit_exact_and_prefetches_next_chunk() {
    let mut cfg = tiny_cfg();
    cfg.context_length = 128;
    let head_dim = cfg.engram.head_dim;
    let total_rows = cfg
        .engram
        .head_offsets
        .iter()
        .zip(&cfg.engram.head_vocab_sizes)
        .map(|(&o, &n)| o + n)
        .max()
        .unwrap();
    let raw_f8: Vec<u8> = Rng::new(77)
        .bytes(total_rows as usize * head_dim)
        .into_iter()
        .map(|b| if (b & 0x7F) == 0x7F { b & 0xFE } else { b })
        .collect();
    let dir = Temp::new("engram-la");
    let path = dir.write("engram.bin", &raw_f8);
    let loc = Located {
        shard: 0,
        data_offset: 0,
        info: GgufTensorInfo {
            name: "per_layer_token_embd.weight".into(),
            n_dims: 2,
            dims: vec![head_dim as u64, total_rows],
            tensor_type: GgufTensorType::MXFP8,
            offset: 0,
        },
    };

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Ev {
        Prefetch(usize, usize),
        Take(usize),
        SyncRows(usize),
    }
    struct SpyRows {
        inner: F8Table,
        events: Arc<Mutex<Vec<Ev>>>,
    }
    impl EngramRows for SpyRows {
        fn rows(&self, layer: usize, ids: &[u32]) -> napi::Result<Vec<f32>> {
            self.events.lock().unwrap().push(Ev::SyncRows(layer));
            self.inner.rows(layer, ids)
        }
        fn prefetch(&self, layer: usize, ids: &[u32]) -> napi::Result<Option<PrefetchHandle>> {
            self.events
                .lock()
                .unwrap()
                .push(Ev::Prefetch(layer, ids.len()));
            self.inner.prefetch(layer, ids)
        }
        fn take_prefetched(&self, handle: PrefetchHandle) -> napi::Result<Vec<f32>> {
            self.events.lock().unwrap().push(Ev::Take(0));
            self.inner.take_prefetched(handle)
        }
    }

    let events = Arc::new(Mutex::new(Vec::new()));
    let shard = ShardRig::new(cfg.clone(), 0, 91);
    let opts = TinyWeightsOpts {
        seed: 19,
        ..Default::default()
    };
    let bk = BankKind::Streamed {
        cache: [4, 4],
        staging: 1,
    };
    let la_rig = shard.model(
        bk,
        opts,
        Box::new(SpyRows {
            inner: F8Table::open_at(&path, &loc, &cfg.engram).unwrap(),
            events: Arc::clone(&events),
        }),
    );
    let ref_rig = shard.model(
        BankKind::Streamed {
            cache: [4, 4],
            staging: 1,
        },
        opts,
        Box::new(F8Table::open_at(&path, &loc, &cfg.engram).unwrap()),
    );
    let ids = Rng::new(123).ids(96, TINY_V);

    let mut ref_sess = ref_rig.model.new_session(128).unwrap();
    let mut ref_out = None;
    for chunk in ids.chunks(32) {
        ref_out = Some(fwd(
            &ref_rig.model,
            &mut ref_sess,
            chunk,
            RowKind::Prefill,
            Want::LAST,
        ));
    }
    let want_bits = read_bits(ref_out.unwrap().logits.as_ref().unwrap());

    let grid = Grid::new(32, Vec::new()).unwrap();
    let driver = Driver {
        runner: &la_rig.model,
        grid: &grid,
        hidden: false,
    };
    let (mut sess, mut ladder, mut ran) = (
        la_rig.model.new_session(128).unwrap(),
        Ladder::new(),
        Vec::new(),
    );
    let Prefilled::Done { out, .. } = driver
        .prefill(
            &mut sess,
            &mut ladder,
            None,
            &ids,
            &|_| false,
            &mut |r, _| {
                ran.push(r);
                Ok(())
            },
        )
        .unwrap()
    else {
        panic!("expected Prefilled::Done");
    };
    assert_eq!(ran, vec![0..32, 32..64, 64..96]);
    assert_eq!(
        read_bits(out.logits.as_ref().unwrap()),
        want_bits,
        "lookahead vs manual chunks"
    );

    let n_cols = cfg.engram.n_cols();
    let pf = Ev::Prefetch(0, 32 * n_cols);
    assert_eq!(
        *events.lock().unwrap(),
        vec![
            pf.clone(),
            Ev::Take(0),
            pf.clone(),
            Ev::Take(0),
            pf,
            Ev::Take(0)
        ]
    );
    assert!(!la_rig.model.has_prefill_engram(&ids, 0));

    let (mut cs, mut cl) = (la_rig.model.new_session(128).unwrap(), Ladder::new());
    let res = driver
        .prefill(
            &mut cs,
            &mut cl,
            None,
            &ids,
            &|l| l >= 2,
            &mut |_, _| Ok(()),
        )
        .unwrap();
    assert!(matches!(res, Prefilled::Cancelled { at: 0 }));
    assert!(!la_rig.model.has_prefill_engram(&ids, 32));
}
