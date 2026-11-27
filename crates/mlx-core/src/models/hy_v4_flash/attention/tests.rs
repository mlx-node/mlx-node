use serde_json::Value;

use super::*;
use crate::models::deepseek_v41::ops::rope_angle;
use crate::models::hy_v4_flash::cache::{
    LatentBuf, LayerCache, WINDOW_SLACK, WindowKeys, WindowRing,
};
use crate::models::hy_v4_flash::indexer::{Indexer, IndexerGeometry, IndexerState, Indexing};
use crate::models::hy_v4_flash::moe::RowKind;
use crate::models::hy_v4_flash::sparse::{
    gather_rows_reference, gather_sink_attention, gather_sink_attention_in,
    gather_sink_attention_rows,
};
#[cfg(target_os = "macos")]
use crate::models::hy_v4_flash::sparse::{
    gather_sink_attention_for, gather_sink_attention_rows_for,
};
use crate::models::hy_v4_flash::test_support::{
    Rng, arr, assert_close_f64, assert_row_invariant, error_stats, fixture, j_usize as u, make_qmm,
    read_bits, read_f32,
};
use crate::models::hy_v4_flash::weights::IndexerWeights;

const D: usize = 12;
const Q_LORA: usize = 6;
const MLA_D: usize = 576;
const MLA_V: i64 = 512;
const MLA_DV: usize = 512;
#[cfg(target_os = "macos")]
const MLA_DR: usize = 64;

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

#[test]
fn window_rows_match_fixture_index_rows() {
    let w = &fixture("ops.json")["window"];
    let win = u(&w["window"]);
    for r in w["index_rows"].as_array().unwrap() {
        let (p, first) = (u(&r["p"]), u(&r["first"]));
        let row: Vec<i64> = window_rows(p, 1, first, win)
            .into_iter()
            .map(i64::from)
            .collect();
        assert_eq!(row.len(), win, "p {p}");
        assert_eq!(&row[..3], ints(&r["head"]).as_slice());
        assert_eq!(&row[win - 3..], ints(&r["tail"]).as_slice());
        assert_eq!(
            row.iter().filter(|&&k| k >= 0).count(),
            u(&r["n_valid"]),
            "p {p}"
        );
    }
}

#[test]
fn window_rows_match_fixture_visibility() {
    let w = &fixture("ops.json")["window"];
    let win = u(&w["window"]);
    for c in w["cases"].as_array().unwrap() {
        let t = u(&c["t"]);
        let raw = window_rows(t, 1, 0, win);
        let keys: Vec<i64> = raw.into_iter().filter(|&k| k >= 0).map(i64::from).collect();
        let (lo, hi) = (c["lo"].as_i64().unwrap(), c["hi"].as_i64().unwrap());
        assert_eq!(keys, (lo..=hi).collect::<Vec<_>>(), "t {t}");
        assert_eq!(keys.len(), u(&c["n_keys"]), "t {t}");
        for (dist, vis) in c["dist_visible"].as_object().unwrap() {
            let d: i64 = dist.parse().unwrap();
            assert_eq!(
                keys.contains(&(t as i64 - d)),
                vis.as_bool().unwrap(),
                "t {t} d {d}"
            );
        }
    }
}

#[test]
fn window_rows_at_510_to_513() {
    for p in 510usize..=513 {
        let lo = (p + 1).saturating_sub(512);
        let pad = std::iter::repeat_n(-1, 512 - (p + 1 - lo));
        let want: Vec<i32> = (lo..=p).map(|k| k as i32).chain(pad).collect();
        assert_eq!(window_rows(p, 1, 0, 512), want, "p {p}");
    }
    let two = window_rows(511, 2, 0, 512);
    assert_eq!(two[..512], window_rows(511, 1, 0, 512));
    assert_eq!(two[512..], window_rows(512, 1, 0, 512));
}

fn geometry(attn: Attn) -> Geometry {
    Geometry {
        attn,
        n_heads: 2,
        kv_lora: 6,
        nope: 4,
        rope_dim: 4,
        head_v: 5,
        eps: 1e-5,
        scale: 1.0 / 8f32.sqrt(),
        rope_theta: if attn == Attn::Sliding { 1e4 } else { 1e7 },
        window: 4,
    }
}

/// Host copies of the weights, row-major `[out, in]`.
struct RefW {
    norm: Vec<f32>,
    sinks: Vec<f32>,
    q_a: Vec<f32>,
    q_a_norm: Vec<f32>,
    q_b: Vec<f32>,
    kv_a: Vec<f32>,
    kv_a_norm: Vec<f32>,
    k_b: Vec<f32>,
    v_b: Vec<f32>,
    output: Vec<f32>,
    gate: Vec<f32>,
    post_norm: Vec<f32>,
}

fn dense(v: &[f32], rows: usize, cols: usize) -> Dense {
    Dense::Array(arr(v, &[rows, cols]))
}

fn norm_vec(r: &mut Rng, n: usize) -> Vec<f32> {
    r.vec_f32(n, 0.5).into_iter().map(|x| 1.0 + x).collect()
}

fn weights(g: &Geometry, seed: u64) -> (AttnWeights, RefW) {
    let mut r = Rng::new(seed);
    let (h, l, v, qk) = (g.n_heads, g.kv_lora, g.head_v, g.head_qk());
    let mat = |r: &mut Rng, rows, cols| r.vec_f32(rows * cols, 1.5 / (cols as f32).sqrt());
    let rw = RefW {
        norm: norm_vec(&mut r, D),
        sinks: r.vec_f32(h, 1.0),
        q_a: mat(&mut r, Q_LORA, D),
        q_a_norm: norm_vec(&mut r, Q_LORA),
        q_b: mat(&mut r, h * qk, Q_LORA),
        kv_a: mat(&mut r, g.row(), D),
        kv_a_norm: norm_vec(&mut r, l),
        k_b: mat(&mut r, h * l, g.nope),
        v_b: mat(&mut r, h * v, l),
        output: mat(&mut r, D, h * v),
        gate: mat(&mut r, h * v, D),
        post_norm: norm_vec(&mut r, D),
    };
    let w = AttnWeights {
        norm: arr(&rw.norm, &[D]),
        sinks: arr(&rw.sinks, &[h]),
        q_a: dense(&rw.q_a, Q_LORA, D),
        q_a_norm: arr(&rw.q_a_norm, &[Q_LORA]),
        q_b: dense(&rw.q_b, h * qk, Q_LORA),
        kv_a: dense(&rw.kv_a, g.row(), D),
        kv_a_norm: arr(&rw.kv_a_norm, &[l]),
        k_b: dense(&rw.k_b, h * l, g.nope),
        v_b: dense(&rw.v_b, h * v, l),
        output: dense(&rw.output, D, h * v),
        gate: dense(&rw.gate, h * v, D),
        post_norm: arr(&rw.post_norm, &[D]),
        q_kv: None,
        q_kv_gate: None,
    };
    (w, rw)
}

fn make_mla(attn: Attn, seed: u64) -> (Geometry, AttnWeights, Mla) {
    let g = geometry(attn);
    let (w, _) = weights(&g, seed);
    let mla = Mla::new(g.clone(), &w).unwrap();
    (g, w, mla)
}

fn mv(w: &[f32], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    assert_eq!((w.len(), x.len()), (rows * cols, cols));
    (0..rows)
        .map(|r| {
            let sl = &w[r * cols..(r + 1) * cols];
            sl.iter().zip(x).map(|(&a, &b)| f64::from(a) * b).sum()
        })
        .collect()
}

fn rms(x: &[f64], w: &[f32], eps: f32) -> Vec<f64> {
    let mean = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (mean + f64::from(eps)).sqrt();
    x.iter()
        .zip(w)
        .map(|(&v, &g)| v * r * f64::from(g))
        .collect()
}

fn rope(v: &mut [f64], pos: usize, inv: &[f32]) {
    for (i, &f) in inv.iter().enumerate() {
        let ang = f64::from(rope_angle(pos as u64, f));
        let (c, s) = (f64::from(ang.cos() as f32), f64::from(ang.sin() as f32));
        let (a, b) = (v[2 * i], v[2 * i + 1]);
        (v[2 * i], v[2 * i + 1]) = (a * c - b * s, a * s + b * c);
    }
}

fn f16r(v: f64) -> f64 {
    half::f16::from_f64(v).to_f64()
}

fn softmax_sink<'a>(
    q: &[f64],
    keys: impl IntoIterator<Item = &'a [f64]>,
    vw: usize,
    sink: f32,
    scale: f32,
) -> Vec<f64> {
    let ks: Vec<&[f64]> = keys.into_iter().collect();
    let dot = |k: &[f64]| q.iter().zip(k).map(|(x, y)| x * y).sum::<f64>();
    let sc: Vec<f64> = ks.iter().map(|k| f64::from(scale) * dot(k)).collect();
    let m = sc.iter().copied().fold(-1e30, f64::max);
    let e: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
    let den = e.iter().sum::<f64>() + (f64::from(sink) - m).exp();
    (0..vw)
        .map(|c| ks.iter().zip(&e).map(|(k, &p)| p * k[c]).sum::<f64>() / den)
        .collect()
}

/// The layer output for every token of `hs` (positions from 0), f64 math with f16 cached rows.
fn reference(g: &Geometry, r: &RefW, hs: &[Vec<f32>]) -> Vec<Vec<f64>> {
    let (h, l, nope, rd, v) = (g.n_heads, g.kv_lora, g.nope, g.rope_dim, g.head_v);
    let (qk, inv) = (nope + rd, inv_freqs(rd, g.rope_theta, None));
    let (mut keys, mut outs) = (Vec::<Vec<f64>>::new(), Vec::new());
    for (t, row) in hs.iter().enumerate() {
        let hx: Vec<f64> = row.iter().map(|&x| f64::from(x)).collect();
        let x = rms(&hx, &r.norm, g.eps);
        let qr = rms(&mv(&r.q_a, Q_LORA, D, &x), &r.q_a_norm, g.eps);
        let mut q = mv(&r.q_b, h * qk, Q_LORA, &qr);
        (0..h).for_each(|hh| rope(&mut q[hh * qk + nope..(hh + 1) * qk], t, &inv));
        let ckv = mv(&r.kv_a, l + rd, D, &x);
        let (mut k, mut pe) = (rms(&ckv[..l], &r.kv_a_norm, g.eps), ckv[l..].to_vec());
        rope(&mut pe, t, &inv);
        k.extend(pe);
        keys.push(k.into_iter().map(f16r).collect());
        let lo = if g.attn == Attn::Sliding {
            (t + 1).saturating_sub(g.window)
        } else {
            0
        };
        let mut o = vec![0.0; h * v];
        for hh in 0..h {
            let wkb = &r.k_b[hh * l * nope..(hh + 1) * l * nope];
            let mut qq = mv(wkb, l, nope, &q[hh * qk..hh * qk + nope]);
            qq.extend_from_slice(&q[hh * qk + nope..(hh + 1) * qk]);
            let ks = (lo..=t).map(|j| keys[j].as_slice());
            let ol = softmax_sink(&qq, ks, l, r.sinks[hh], g.scale);
            let wvb = &r.v_b[hh * v * l..(hh + 1) * v * l];
            o[hh * v..(hh + 1) * v].copy_from_slice(&mv(wvb, v, l, &ol));
        }
        let gt = mv(&r.gate, h * v, D, &x);
        o.iter_mut()
            .zip(gt)
            .for_each(|(x, g)| *x /= 1.0 + (-g).exp());
        outs.push(rms(&mv(&r.output, D, h * v, &o), &r.post_norm, g.eps));
    }
    outs
}

fn inputs(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.vec_f32(D, 2.0)).collect()
}

fn rows_arr(hs: &[Vec<f32>]) -> MxArray {
    let flat: Vec<f32> = hs.iter().flatten().copied().collect();
    arr(&flat, &[hs.len(), D])
}

fn assert_rows(got: &[f32], want: &[Vec<f64>], what: &str) {
    let flat: Vec<f64> = want.iter().flatten().copied().collect();
    assert_close_f64(got, &flat, 2e-3, 2e-3, what);
}

/// An indexer whose `n_top` (3 groups of 4) keeps every query of up to 16 tokens dense.
fn dense_indexer(seed: u64) -> (Indexer, IndexerWeights, IndexerGeometry) {
    let ig = IndexerGeometry {
        n_heads: 2,
        head_dim: 8,
        rope_dim: 4,
        kpool: 4,
        n_top: 3,
        select_tail: true,
        eps: 1e-5,
        weight_scale: 0.25,
        rope_theta: 1e7,
    };
    let mut r = Rng::new(seed);
    let (h, d, kp) = (ig.n_heads, ig.head_dim, ig.kpool);
    let iw = IndexerWeights {
        attn_k: dense(&r.vec_f32(d * D, 0.5), d, D),
        k_norm: arr(&r.vec_f32(d, 0.5), &[d]),
        q_b: dense(&r.vec_f32(h * d * Q_LORA, 0.5), h * d, Q_LORA),
        proj: dense(&r.vec_f32(h * D, 0.5), h, D),
        compressor_value: arr(&r.vec_f32(d * d, 0.5), &[d, d]),
        compressor_gate: arr(&r.vec_f32(d * d, 0.5), &[d, d]),
        compressor_ape: arr(&r.vec_f32(kp * d, 0.5), &[kp, d]),
        compressor_norm: arr(&r.vec_f32(d, 0.5), &[d]),
    };
    (Indexer::new(ig.clone(), &iw).unwrap(), iw, ig)
}

struct Layer {
    cache: LayerCache,
    index: Option<(Indexer, IndexerWeights, IndexerState)>,
}

impl Layer {
    fn new(g: &Geometry, cap: usize) -> Self {
        let index = (g.attn == Attn::Indexed).then(|| {
            let (ix, iw, ig) = dense_indexer(0x51);
            (ix, iw, IndexerState::new(cap, &ig))
        });
        let cache = LayerCache::new(g.attn, g.window, cap, g.row() as i64);
        Self { cache, index }
    }
    fn index_len(&self) -> Option<usize> {
        self.index.as_ref().map(|(_, _, s)| s.len())
    }
    fn truncate(&mut self, len: usize) -> Result<()> {
        self.cache.truncate(len)?;
        if let Some((_, _, s)) = self.index.as_mut() {
            s.truncate(len)?;
        }
        Ok(())
    }
    fn forward(
        &mut self,
        mla: &Mla,
        w: &AttnWeights,
        h: &MxArray,
        start: usize,
    ) -> Result<MxArray> {
        let ix = self.index.as_mut().map(|(i, iw, s)| Indexing {
            indexer: &*i,
            weights: &*iw,
            state: s,
            kind: RowKind::Prefill,
        });
        mla.forward(w, h, start, &mut self.cache, ix)
    }
}

fn run_split(
    mla: &Mla,
    w: &AttnWeights,
    layer: &mut Layer,
    hs: &[Vec<f32>],
    start: usize,
    sizes: &[usize],
) -> Vec<f32> {
    let (mut out, mut at) = (Vec::new(), 0);
    for &n in sizes {
        let y = layer
            .forward(mla, w, &rows_arr(&hs[at..at + n]), start + at)
            .unwrap();
        out.extend(read_f32(&y));
        at += n;
    }
    assert_eq!(at, hs.len());
    out
}

fn check_layer(attn: Attn, seed: u64) {
    crate::models::hy_v4_flash::exact_f32();
    let g = geometry(attn);
    let (w, rw) = weights(&g, seed);
    let (mla, hs) = (Mla::new(g.clone(), &w).unwrap(), inputs(9, seed ^ 0x9e37));
    let want = reference(&g, &rw, &hs);
    for sizes in [vec![9], vec![1, 3, 5], vec![1; 9], vec![4, 1, 4]] {
        let mut layer = Layer::new(&g, 16);
        let got = run_split(&mla, &w, &mut layer, &hs, 0, &sizes);
        assert_rows(&got, &want, &format!("{attn:?} {sizes:?}"));
        assert_eq!(layer.cache.len(), 9);
        if attn == Attn::Indexed {
            assert_eq!(layer.index_len(), Some(9));
        }
    }
}

#[test]
fn sliding_layer_matches_reference_for_every_split() {
    check_layer(Attn::Sliding, 7);
}

#[test]
fn indexed_layer_matches_reference_for_every_split() {
    check_layer(Attn::Indexed, 11);
}

#[test]
fn truncate_then_replay_equals_the_first_pass() {
    crate::models::hy_v4_flash::exact_f32();
    for attn in [Attn::Sliding, Attn::Indexed] {
        let g = geometry(attn);
        let (w, rw) = weights(&g, 23);
        let (mla, hs) = (Mla::new(g.clone(), &w).unwrap(), inputs(9, 5));
        let want = reference(&g, &rw, &hs);
        let mut layer = Layer::new(&g, 16);
        run_split(&mla, &w, &mut layer, &hs, 0, &[9]);
        layer.truncate(9 - WINDOW_SLACK).unwrap();
        assert_eq!(layer.cache.len(), 6);
        let got = run_split(&mla, &w, &mut layer, &hs[6..], 6, &[1, 2]);
        assert_rows(&got, &want[6..], &format!("{attn:?} replay"));
    }
}

#[test]
fn sliding_ring_refuses_a_truncate_past_its_slack() {
    let (g, w, mla) = make_mla(Attn::Sliding, 3);
    let mut layer = Layer::new(&g, 16);
    run_split(&mla, &w, &mut layer, &inputs(9, 8), 0, &[9]);
    assert!(layer.truncate(9 - WINDOW_SLACK - 1).is_err() && layer.cache.len() == 9);
    layer.truncate(0).unwrap();
    assert!(layer.cache.is_empty());
}

#[test]
fn indexed_layer_needs_its_indexer_and_keeps_its_cache() {
    let (g, w, mla) = make_mla(Attn::Indexed, 9);
    let hs = inputs(9, 2);
    let mut cache = LayerCache::new(g.attn, g.window, 16, g.row() as i64);
    assert!(
        mla.forward(&w, &rows_arr(&hs), 0, &mut cache, None)
            .is_err()
    );
    assert_eq!(cache.len(), 0);
    let mut layer = Layer::new(&g, 16);
    run_split(&mla, &w, &mut layer, &hs[..8], 0, &[8]);
    assert!(layer.forward(&mla, &w, &rows_arr(&hs[8..]), 7).is_err());
    assert_eq!((layer.cache.len(), layer.index_len()), (8, Some(8)));
}

#[test]
fn indexed_layer_refuses_a_call_past_the_context() {
    let (g, w, mla) = make_mla(Attn::Indexed, 13);
    let mut layer = Layer::new(&g, 5);
    assert!(
        layer
            .forward(&mla, &w, &rows_arr(&inputs(6, 4)), 0)
            .is_err()
    );
    assert_eq!((layer.cache.len(), layer.index_len()), (0, Some(0)));
}

#[test]
fn forward_refuses_the_other_cache_kind() {
    let (g, w, mla) = make_mla(Attn::Sliding, 1);
    let mut cache = LayerCache::new(Attn::Indexed, g.window, 16, g.row() as i64);
    assert!(
        mla.forward(&w, &rows_arr(&inputs(1, 1)), 0, &mut cache, None)
            .is_err()
    );
}

#[test]
fn new_refuses_a_swapped_k_b() {
    let g = geometry(Attn::Indexed);
    let (mut w, rw) = weights(&g, 17);
    w.k_b = dense(&rw.k_b, g.n_heads * g.nope, g.kv_lora);
    assert!(Mla::new(g.clone(), &w).is_err());
    let (mut w, rw) = weights(&g, 17);
    w.v_b = dense(&rw.v_b, g.n_heads * g.kv_lora, g.head_v);
    assert!(Mla::new(g, &w).is_err());
}

fn positions(rows: std::ops::Range<usize>) -> MxArray {
    let v: Vec<f32> = rows.clone().flat_map(|p| [p as f32; 2]).collect();
    arr(&v, &[rows.len(), 2])
}

fn col0(x: &MxArray) -> Vec<f32> {
    read_f32(x).chunks(2).map(|c| c[0]).collect()
}

fn ring_vals(wk: WindowKeys<'_>, range: std::ops::Range<usize>) -> (Vec<f32>, usize) {
    match wk {
        WindowKeys::Ring { keys, cap } => {
            let v = col0(keys);
            (range.map(|p| v[p % cap]).collect(), cap)
        }
        WindowKeys::Linear { .. } => panic!("expected Ring"),
    }
}

#[test]
fn window_ring_keeps_window_plus_slack() {
    let mut ring = WindowRing::new(4, WINDOW_SLACK, 2);
    let (v, cap) = ring_vals(ring.append(&positions(0..5), 0).unwrap(), 0..5);
    assert_eq!((cap, v.as_slice()), (7, &[0., 1., 2., 3., 4.][..]));
    let (v, cap) = ring_vals(ring.append(&positions(5..9), 5).unwrap(), 2..9);
    assert_eq!(
        (cap, ring.kept(), v.as_slice()),
        (7, 7, &[2., 3., 4., 5., 6., 7., 8.][..])
    );
    let (v, _) = ring_vals(ring.append(&positions(9..10), 9).unwrap(), 3..10);
    assert_eq!(v, [3., 4., 5., 6., 7., 8., 9.]);
    assert!(ring.append(&positions(0..1), 4).is_err());

    let snap = ring.snapshot().unwrap();
    ring.truncate(7).unwrap();
    assert_eq!((ring.len(), ring.kept()), (7, 4));
    let rep = positions(7..8).mul_scalar(10.0).unwrap();
    assert_eq!(
        ring_vals(ring.append(&rep, 7).unwrap(), 3..8).0,
        [3., 4., 5., 6., 70.]
    );
    assert!(ring.truncate(5).is_err());
    assert_eq!((snap.len(), snap.kept()), (10, 7));
    assert_eq!(col0(snap.rows().unwrap()), [3., 4., 5., 6., 7., 8., 9.]);
}

#[test]
fn window_ring_defers_appends_and_truncates_within_slack_without_eval() {
    use crate::models::hy_v4_flash::stats::{self, DeferScope};
    let (mut ring, mut latent) = (WindowRing::new(4, WINDOW_SLACK, 2), LatentBuf::new(16, 2));
    latent.write(&positions(0..1), 0).unwrap();
    {
        let _scope = DeferScope::enter();
        let before = stats::thread_evals();
        assert_eq!(stats::deferred(), 0);
        for (i, r) in [0..5, 5..8, 8..16].into_iter().enumerate() {
            ring.append(&positions(r.clone()), r.start).unwrap();
            assert_eq!((stats::deferred(), stats::thread_evals()), (i + 1, before));
        }
        latent.write(&positions(1..3), 1).unwrap();
        assert_eq!((stats::deferred(), stats::thread_evals()), (4, before));
        stats::flush("test window flush").unwrap();
        assert_eq!((stats::deferred(), stats::thread_evals()), (0, before + 1));
    }
    let snap = ring.snapshot().unwrap();
    let want_snap = col0(snap.rows().unwrap());
    assert_eq!(want_snap, [9., 10., 11., 12., 13., 14., 15.]);

    let before = stats::thread_evals();
    ring.truncate(16 - WINDOW_SLACK).unwrap();
    assert_eq!(
        (stats::thread_evals(), ring.len(), ring.kept()),
        (before, 13, 4)
    );
    assert_eq!(
        col0(ring.snapshot().unwrap().rows().unwrap()),
        [9., 10., 11., 12.]
    );

    let rep = positions(13..14).mul_scalar(10.0).unwrap();
    let (v, _) = ring_vals(ring.append(&rep, 13).unwrap(), 9..14);
    assert_eq!(v, [9., 10., 11., 12., 130.]);
    assert_eq!(col0(snap.rows().unwrap()), want_snap);
    assert_eq!(col0(latent.rows().unwrap())[..3], [0., 1., 2.]);
}

#[test]
fn latent_buf_writes_rows_and_truncates_by_length() {
    let mut buf = LatentBuf::new(6, 2);
    assert!(buf.rows().is_err());
    buf.write(&positions(0..3), 0).unwrap();
    buf.write(&positions(3..5), 3).unwrap();
    assert_eq!(buf.len(), 5);
    let rows = buf.rows().unwrap();
    assert_eq!(rows.dtype().unwrap(), DType::Float16);
    assert_eq!(col0(rows)[..5], [0., 1., 2., 3., 4.]);
    buf.truncate(2).unwrap();
    buf.write(&positions(20..21), 2).unwrap();
    assert_eq!(col0(buf.rows().unwrap())[..4], [0., 1., 20., 3.]);
    assert!(buf.write(&positions(0..4), 3).is_err());
    assert!(buf.write(&positions(0..1), 2).is_err() && buf.truncate(4).is_err());
    assert_eq!((buf.len(), buf.nbytes()), (3, 6 * 2 * 2));
}

#[test]
fn gather_core_matches_host_softmax_with_sinks() {
    crate::models::hy_v4_flash::exact_f32();
    let (s, h, d, n, kmax, vw, scale) = (5usize, 2usize, 6usize, 7usize, 4usize, 4usize, 0.37f32);
    let mut r = Rng::new(31);
    let (q, sinks) = (r.vec_f32(s * h * d, 1.5), r.vec_f32(h, 1.0));
    let raw_k = r.vec_f32(n * d, 1.5);
    let keys: Vec<f32> = raw_k
        .into_iter()
        .map(|x| f16r(f64::from(x)) as f32)
        .collect();
    let idx = [
        0, 1, 2, 3, 6, -1, 2, -1, -1, -1, -1, -1, 5, 5, 0, -1, 3, 4, 6, 1,
    ];
    let (qa, sa) = (arr(&q, &[s, h, d]), arr(&sinks, &[h]));
    let ka = arr(&keys, &[n, d]).astype(DType::Float16).unwrap();
    let ia = MxArray::from_int32(&idx, &[s as i64, kmax as i64]).unwrap();
    let one = read_f32(&gather_sink_attention(&qa, &ka, &ia, vw as i64, &sa, scale).unwrap());
    let tiny =
        read_f32(&gather_sink_attention_in(&qa, &ka, &ia, vw as i64, &sa, scale, 1).unwrap());
    assert!(one.len() == s * h * vw && error_stats(&one, &tiny).0 <= 1e-6);

    let to64 = |sl: &[f32]| sl.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let k64: Vec<Vec<f64>> = keys.chunks(d).map(to64).collect();
    let mut want = Vec::with_capacity(s * h * vw);
    for row in 0..s {
        let sl = &idx[row * kmax..(row + 1) * kmax];
        let ids: Vec<usize> = sl
            .iter()
            .filter(|&&j| j >= 0)
            .map(|&j| j as usize)
            .collect();
        for hh in 0..h {
            let qv = to64(&q[(row * h + hh) * d..(row * h + hh + 1) * d]);
            let ks = ids.iter().map(|&j| k64[j].as_slice());
            want.extend(softmax_sink(&qv, ks, vw, sinks[hh], scale));
        }
    }
    assert_close_f64(&one, &want, 1e-5, 1e-5, "gather_core");
}

#[test]
fn gather_core_row_bits_do_not_depend_on_the_call_height() {
    let scale = 1.0 / (MLA_D as f32).sqrt();
    for (h, kmax) in [(64usize, 512usize), (32, 2048)] {
        let n = kmax + 5;
        let idx_fn = |r, j| {
            if j + r >= kmax {
                -1
            } else {
                ((j + 3 * r) % n) as i32
            }
        };
        let m = MlaInputs::new(4, h, n, kmax, 1.5, None, 53 + h as u64, idx_fn);
        let lbl = format!("gather h={h} k={kmax}");
        assert_row_invariant(
            |r| m.call_slice(r, 1, scale),
            |b| m.call_slice(0, b, scale),
            &[2, 3, 4],
            &lbl,
        );
    }
}

/// An unlisted id reads key row 0; a non-finite row there leaves the output finite and bit-equal.
#[test]
fn gather_core_ignores_a_non_finite_unlisted_row() {
    let (s, h, d, n, kmax, vw) = (3usize, 2usize, 6usize, 7usize, 4usize, 4i64);
    let mut r = Rng::new(59);
    let (qa, sa) = (
        arr(&r.vec_f32(s * h * d, 1.5), &[s, h, d]),
        arr(&r.vec_f32(h, 1.0), &[h]),
    );
    let mut keys = r.vec_f32(n * d, 1.5);
    let idx = [3, -1, 2, 6, -1, -1, -1, -1, 5, 4, -1, 1];
    let ia = MxArray::from_int32(&idx, &[s as i64, kmax as i64]).unwrap();
    let run = |kv: &[f32]| {
        let ka = arr(kv, &[n, d]).astype(DType::Float16).unwrap();
        read_bits(&gather_sink_attention(&qa, &ka, &ia, vw, &sa, 0.37).unwrap())
    };
    keys[..d].fill(0.0);
    let want = run(&keys);
    keys[..d].copy_from_slice(&[
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1.0,
        2.0,
    ]);
    let got = run(&keys);
    assert!(got.iter().all(|&b| f32::from_bits(b).is_finite()) && got == want);
}

#[test]
fn gather_core_refuses_mismatched_shapes() {
    let q = MxArray::zeros(&[2, 2, 6], None).unwrap();
    let k = MxArray::zeros(&[3, 6], Some(DType::Float16)).unwrap();
    let sinks = MxArray::zeros(&[2], None).unwrap();
    let idx = MxArray::from_int32(&[0, 1, 2, 0], &[2, 2]).unwrap();
    assert!(gather_sink_attention(&q, &k, &idx, 4, &sinks, 1.0).is_ok());
    let short = MxArray::from_int32(&[0, 1], &[1, 2]).unwrap();
    let narrow = MxArray::zeros(&[3, 5], Some(DType::Float16)).unwrap();
    let sinks3 = MxArray::zeros(&[3], None).unwrap();
    assert!(gather_sink_attention(&q, &k, &short, 4, &sinks, 1.0).is_err());
    assert!(gather_sink_attention(&q, &k, &idx, 7, &sinks, 1.0).is_err());
    assert!(gather_sink_attention(&q, &narrow, &idx, 4, &sinks, 1.0).is_err());
    assert!(gather_sink_attention(&q, &k, &idx, 4, &sinks3, 1.0).is_err());
}

/// With Q8 projections (`k_b`, `v_b` grouped), a row alone projects and finishes
/// to the bits of the same row in a 3-row call.
#[test]
fn q8_projections_of_a_row_do_not_depend_on_the_call() {
    let g = Geometry {
        attn: Attn::Indexed,
        n_heads: 2,
        kv_lora: 32,
        nope: 32,
        rope_dim: 32,
        head_v: 16,
        eps: 1e-5,
        scale: 1.0 / 8f32.sqrt(),
        rope_theta: 1e7,
        window: 4,
    };
    let (dim, ql, h, l, v) = (64, 64, g.n_heads, g.kv_lora, g.head_v);
    let mut r = Rng::new(71);
    let m = |r: &mut Rng, rows, cols| make_qmm(r, rows, cols, 1.5 / (cols as f32).sqrt());
    let nw = |r: &mut Rng, n| arr(&norm_vec(r, n), &[n]);
    let w = AttnWeights {
        norm: nw(&mut r, dim),
        sinks: arr(&r.vec_f32(h, 1.0), &[h]),
        q_a: m(&mut r, ql, dim),
        q_a_norm: nw(&mut r, ql),
        q_b: m(&mut r, h * g.head_qk(), ql),
        kv_a: m(&mut r, g.row(), dim),
        kv_a_norm: nw(&mut r, l),
        k_b: m(&mut r, h * l, g.nope),
        v_b: m(&mut r, h * v, l),
        output: m(&mut r, dim, h * v),
        gate: m(&mut r, h * v, dim),
        post_norm: nw(&mut r, dim),
        q_kv: None,
        q_kv_gate: None,
    };
    let mla = Mla::new(g, &w).unwrap();
    let (hs, lat, start) = (r.vec_f32(3 * dim, 2.0), r.vec_f32(3 * h * l, 1.0), 9);
    let all = mla.project(&w, &arr(&hs, &[3, dim]), start).unwrap();
    let out = mla.finish(&w, &all.x, &arr(&lat, &[3, h, l])).unwrap();
    let row = |a: &MxArray, i: usize| {
        let vals = read_f32(a);
        let per = vals.len() / 3;
        vals[i * per..(i + 1) * per].to_vec()
    };
    for i in 0..3 {
        let xi = arr(&hs[i * dim..(i + 1) * dim], &[1, dim]);
        let one = mla.project(&w, &xi, start + i).unwrap();
        let pairs = [
            ("x", &all.x, &one.x),
            ("qr", &all.qr, &one.qr),
            ("q", &all.q, &one.q),
            ("kv", &all.kv, &one.kv),
        ];
        for (tag, a, b) in pairs {
            assert_eq!(row(a, i), read_f32(b), "row {i} {tag}");
        }
        let li = arr(&lat[i * h * l..(i + 1) * h * l], &[1, h, l]);
        let fo = mla.finish(&w, &one.x, &li).unwrap();
        assert_eq!(row(&out, i), read_f32(&fo), "row {i} finish");
    }
}

// ---------- Full-geometry Hy4 MLA (D=576, V=512) helpers & tests ----------

fn split_kv(keys: &MxArray, n: usize) -> (MxArray, MxArray) {
    let k0 = keys.slice(&[0, 0], &[n as i64, 288]).unwrap();
    let k1 = keys.slice(&[0, 288], &[n as i64, MLA_D as i64]).unwrap();
    (k0, k1)
}

fn make_sinks(r: &mut Rng, h: usize, hi: Option<f32>) -> MxArray {
    let mut v = r.vec_f32(h, 2.0);
    if let Some(hi) = hi.filter(|_| h >= 3) {
        (v[0], v[1], v[2], v[h - 1]) = (-20.0, hi, 0.0, hi);
    }
    arr(&v, &[h])
}

fn mla_run(q: &MxArray, kr: KeyRows<'_>, ia: &MxArray, sinks: &MxArray, scale: f32) -> MxArray {
    gather_sink_attention_rows(q, kr, ia, MLA_V, sinks, scale).unwrap()
}

fn mla_ref(q: &MxArray, kr: KeyRows<'_>, ia: &MxArray, sinks: &MxArray, scale: f32) -> MxArray {
    gather_rows_reference(q, kr, ia, MLA_V, sinks, scale).unwrap()
}

fn assert_bits_eq(a: &MxArray, b: &MxArray, msg: &str) {
    assert_eq!(read_bits(a), read_bits(b), "{msg}");
}

fn assert_mla_parity(got: &MxArray, want: &MxArray, tol: (f32, Option<f32>), label: &str) {
    let (max_abs, rel_l2) = error_stats(&read_f32(got), &read_f32(want));
    let ok = max_abs <= tol.0 && tol.1.is_none_or(|rt| rel_l2 <= rt);
    assert!(ok, "{label}: max_abs={max_abs:e} rel_l2={rel_l2:e}");
}

fn mixed_idx(row: usize, j: usize, kmax: usize, n: usize) -> i32 {
    match row % 5 {
        0 => ((j * 7 + row * 13 + 3) % n) as i32,
        1 if j < kmax / 3 => ((j * 11 + row * 5 + 1) % n) as i32,
        2 => match j % 5 {
            0 | 1 => ((j / 5 + row) % n) as i32,
            2 | 3 => ((j / 5 + 42 + row) % n) as i32,
            _ => -1,
        },
        3 if j == 0 => (n - 1 - (row % 7)) as i32,
        _ => -1,
    }
}

struct MlaInputs {
    q: MxArray,
    keys: MxArray,
    k0: MxArray,
    k1: MxArray,
    sinks: MxArray,
    idx: MxArray,
    idx_raw: Vec<i32>,
    s: usize,
    h: usize,
    kmax: usize,
}

impl MlaInputs {
    #[allow(clippy::too_many_arguments)]
    fn new(
        s: usize,
        h: usize,
        n: usize,
        kmax: usize,
        q_sc: f32,
        hi: Option<f32>,
        seed: u64,
        idx_fn: impl Fn(usize, usize) -> i32,
    ) -> Self {
        let mut r = Rng::new(seed);
        let q = arr(&r.vec_f32(s * h * MLA_D, q_sc), &[s, h, MLA_D]);
        let keys = arr(&r.vec_f32(n * MLA_D, q_sc), &[n, MLA_D])
            .astype(DType::Float16)
            .unwrap();
        let (k0, k1) = split_kv(&keys, n);
        let sinks = make_sinks(&mut r, h, hi);
        let idx_raw: Vec<i32> = (0..s * kmax).map(|i| idx_fn(i / kmax, i % kmax)).collect();
        let idx = MxArray::from_int32(&idx_raw, &[s as i64, kmax as i64]).unwrap();
        Self {
            q,
            keys,
            k0,
            k1,
            sinks,
            idx,
            idx_raw,
            s,
            h,
            kmax,
        }
    }
    fn whole(&self) -> KeyRows<'_> {
        KeyRows::Whole(&self.keys)
    }
    fn split(&self) -> KeyRows<'_> {
        KeyRows::Split(&self.k0, &self.k1)
    }
    fn slice(&self, lo: usize, rows: usize) -> (MxArray, MxArray) {
        let (l, r) = (lo as i64, (lo + rows) as i64);
        let q = self
            .q
            .slice(&[l, 0, 0], &[r, self.h as i64, MLA_D as i64])
            .unwrap();
        let i = self.idx.slice(&[l, 0], &[r, self.kmax as i64]).unwrap();
        (q, i)
    }
    fn call_slice(&self, lo: usize, rows: usize, scale: f32) -> MxArray {
        let (qs, is) = self.slice(lo, rows);
        gather_sink_attention(&qs, &self.keys, &is, MLA_V, &self.sinks, scale).unwrap()
    }
    fn pad_idx(&self, extra: usize) -> MxArray {
        let kp = self.kmax + extra;
        let mut pad = vec![-1i32; self.s * kp];
        for row in 0..self.s {
            let src = &self.idx_raw[row * self.kmax..(row + 1) * self.kmax];
            pad[row * kp..row * kp + self.kmax].copy_from_slice(src);
        }
        MxArray::from_int32(&pad, &[self.s as i64, kp as i64]).unwrap()
    }
}

#[allow(clippy::too_many_arguments)]
fn check_keyless_nan_split(
    s: usize,
    h: usize,
    n: usize,
    kmax: usize,
    seed: u64,
    is_keyless: impl Fn(usize) -> bool,
    run_fn: impl Fn(&MxArray, KeyRows<'_>, &MxArray, &MxArray) -> MxArray,
) {
    let mut r = Rng::new(seed);
    let q = arr(&r.vec_f32(s * h * MLA_D, 1.5), &[s, h, MLA_D]);
    let mut keys_v = r.vec_f32(n * MLA_D, 1.5);
    let sinks = make_sinks(&mut r, h, Some(50.0));
    let mut idx = vec![-1i32; s * kmax];
    for row in (0..s).filter(|&row| !is_keyless(row)) {
        for j in 0..37 {
            idx[row * kmax + j * 2] = (1 + (j * 5 + row) % (n - 1)) as i32;
        }
    }
    let ia = MxArray::from_int32(&idx, &[s as i64, kmax as i64]).unwrap();
    let run = |kv: &[f32]| {
        let ka = arr(kv, &[n, MLA_D]).astype(DType::Float16).unwrap();
        let (k0, k1) = split_kv(&ka, n);
        let w = read_bits(&run_fn(&q, KeyRows::Whole(&ka), &ia, &sinks));
        let sp = read_bits(&run_fn(&q, KeyRows::Split(&k0, &k1), &ia, &sinks));
        (w, sp)
    };
    keys_v[..MLA_D].fill(0.0);
    let (ww, ws) = run(&keys_v);
    assert_eq!(ww, ws);
    for (c, v) in keys_v[..MLA_D].iter_mut().enumerate() {
        *v = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][c % 3];
    }
    let (gw, gs) = run(&keys_v);
    assert_eq!((&gw, &gs), (&ww, &ws));
    let (per, zero) = (h * MLA_DV, 0.0f32.to_bits());
    for row in (0..s).filter(|&row| is_keyless(row)) {
        let rng = row * per..(row + 1) * per;
        assert!(
            gw[rng.clone()].iter().all(|&b| b == zero),
            "Whole row {row}"
        );
        assert!(gs[rng].iter().all(|&b| b == zero), "Split row {row}");
    }
}

#[cfg(target_os = "macos")]
fn metal_ok() -> bool {
    unsafe { mlx_sys::mlx_metal_is_available() }
}

#[cfg(target_os = "macos")]
fn prefill_tol() -> (f32, Option<f32>) {
    if std::env::var("MLX_HY4_MLA_PREFILL_2MMA").as_deref() == Ok("1") {
        (1e-5, Some(1e-5))
    } else {
        (3e-3, Some(1e-3))
    }
}

#[cfg(target_os = "macos")]
fn call_mla_prefill_nax(
    q: &MxArray,
    kr: KeyRows<'_>,
    ia: &MxArray,
    sinks: &MxArray,
    scale: f32,
) -> Option<MxArray> {
    let (k0, k1) = match kr {
        KeyRows::Whole(k) => (k.as_raw_ptr(), std::ptr::null_mut()),
        KeyRows::Split(k, v) => (k.as_raw_ptr(), v.as_raw_ptr()),
    };
    let out_f16 = q.dtype().unwrap() == DType::Float16;
    let raw = unsafe {
        mlx_sys::mlx_hy4_mla_prefill_nax(
            q.as_raw_ptr(),
            std::ptr::null_mut(),
            k0,
            k1,
            ia.as_raw_ptr(),
            sinks.as_raw_ptr(),
            MLA_V as i32,
            scale,
            out_f16,
        )
    };
    (!raw.is_null()).then(|| MxArray::from_handle(raw, "test mla_prefill_nax").unwrap())
}

#[cfg(target_os = "macos")]
fn slide_nax(
    q: &MxArray,
    q_pe: Option<&MxArray>,
    keys: &MxArray,
    sinks: &MxArray,
    scale: f32,
    pos: (usize, bool, usize, usize),
) -> MxArray {
    let r = try_sliding_sink_attention_prefill(
        q, q_pe, keys, sinks, MLA_V, scale, pos.0, pos.1, pos.2, 512, pos.3,
    );
    r.unwrap().expect("sliding NAX prefill")
}

#[cfg(target_os = "macos")]
fn make_slide_kv(
    r: &mut Rng,
    s: usize,
    h: usize,
    pos: (usize, bool, usize, usize),
    hi: Option<f32>,
) -> (MxArray, MxArray) {
    let rows = if pos.1 { pos.2 } else { pos.0 + s - pos.3 };
    let keys = arr(&r.vec_f32(rows * MLA_D, 1.0), &[rows, MLA_D])
        .astype(DType::Float16)
        .unwrap();
    (keys, make_sinks(r, h, hi))
}

#[cfg(target_os = "macos")]
fn mla_split_q_prefill(
    q: &MxArray,
    q_pe: Option<&MxArray>,
    kr: KeyRows<'_>,
    ia: &MxArray,
    sinks: &MxArray,
    scale: f32,
) -> MxArray {
    let k = RowKind::Prefill;
    match kr {
        KeyRows::Whole(w) => {
            gather_sink_attention_split_q_for(q, q_pe, w, ia, MLA_V, sinks, scale, k)
        }
        KeyRows::Split(..) => {
            gather_sink_attention_rows_split_q_for(q, q_pe, kr, ia, MLA_V, sinks, scale, k)
        }
    }
    .unwrap()
}

#[cfg(target_os = "macos")]
fn make_split_q(r: &mut Rng, s: usize, h: usize, f16: bool) -> (MxArray, MxArray, MxArray) {
    let mut a = arr(&r.vec_f32(s * h * MLA_DV, 1.0), &[s, h, MLA_DV]);
    let mut p = arr(&r.vec_f32(s * h * MLA_DR, 1.0), &[s, h, MLA_DR]);
    if f16 {
        (a, p) = (
            a.astype(DType::Float16).unwrap(),
            p.astype(DType::Float16).unwrap(),
        );
    }
    let full = MxArray::concatenate(&a, &p, -1).unwrap();
    (a, p, full)
}

#[test]
fn mla_sink_attention_matches_reference_full_geometry() {
    crate::models::hy_v4_flash::exact_f32();
    let scale = 1.0 / (MLA_D as f32).sqrt();
    for (h, kmax, seed) in [(32usize, 2048usize, 131u64), (64, 512, 137)] {
        let n = kmax + 17;
        let m = MlaInputs::new(5, h, n, kmax, 1.5, Some(50.0), seed, |r, j| {
            mixed_idx(r, j, kmax, n)
        });
        #[cfg(target_os = "macos")]
        if metal_ok() {
            let raw = unsafe {
                mlx_sys::mlx_hy4_mla_sink_attention(
                    m.q.as_raw_ptr(),
                    m.keys.as_raw_ptr(),
                    std::ptr::null_mut(),
                    m.idx.as_raw_ptr(),
                    m.sinks.as_raw_ptr(),
                    MLA_V as i32,
                    scale,
                )
            };
            assert!(!raw.is_null(), "expected Metal kernel H={h} kmax={kmax}");
            drop(MxArray::from_handle(raw, "test mla probe").unwrap());
        }
        let got = mla_run(&m.q, m.whole(), &m.idx, &m.sinks, scale);
        let want = mla_ref(&m.q, m.whole(), &m.idx, &m.sinks, scale);
        assert_mla_parity(&got, &want, (1e-3, None), &format!("H={h} k={kmax}"));
    }
}

#[test]
fn mla_sink_attention_keyless_row_exact_zero_with_nan_slot_zero() {
    let scale = 1.0 / (MLA_D as f32).sqrt();
    check_keyless_nan_split(
        3,
        32,
        64,
        512,
        149,
        |r| r == 0 || r == 2,
        |q, kr, ia, s| mla_run(q, kr, ia, s, scale),
    );
}

#[test]
fn mla_sink_attention_strict_row_invariance_up_to_256() {
    let scale = 1.0 / (MLA_D as f32).sqrt();
    let probes = [0usize, 1, 2, 3, 127, 255];
    for (h, kmax, seed) in [(32usize, 2048usize, 163u64), (64, 512, 167)] {
        let n = kmax + 9;
        let idx = |r, j| {
            if j >= kmax - (r % 16) {
                -1
            } else {
                ((j + r * 7) % n) as i32
            }
        };
        let m = MlaInputs::new(256, h, n, kmax, 1.5, None, seed, idx);
        let per = h * MLA_DV;
        let alone: Vec<_> = probes
            .iter()
            .map(|&r| (r, read_bits(&m.call_slice(r, 1, scale))))
            .collect();
        for bs in [2usize, 4, 256] {
            let all = read_bits(&m.call_slice(0, bs, scale));
            for (row, want) in alone.iter().filter(|(r, _)| *r < bs) {
                assert_eq!(
                    &all[*row * per..(*row + 1) * per],
                    want.as_slice(),
                    "H={h} S={bs} r={row}"
                );
            }
        }
    }
    // Split-K chunked batch (S=130 > chunk boundary): boundary rows match S=1 bit-for-bit.
    let (s, h, kmax, n) = (130usize, 64usize, 512usize, 529usize);
    let idx = |r, j| {
        if j >= kmax - (r % 23) {
            -1
        } else {
            ((j * 5 + r * 11) % n) as i32
        }
    };
    let m = MlaInputs::new(s, h, n, kmax, 1.5, None, 173, idx);
    let all = read_bits(&m.call_slice(0, s, scale));
    let per = h * MLA_DV;
    for row in [0usize, 1, 2, 63, 64, 65, 128, 129] {
        let want = read_bits(&m.call_slice(row, 1, scale));
        assert_eq!(
            &all[row * per..(row + 1) * per],
            want.as_slice(),
            "splitk S=130 r={row}"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn mla_prefill_nax_matches_reference_full_geometry() {
    if !metal_ok() {
        return;
    }
    crate::models::hy_v4_flash::exact_f32();
    let (scale, tol) = (0.0625f32, prefill_tol());
    let cases = [
        (64usize, 32usize, 512usize, 211u64),
        (64, 32, 2048, 223),
        (64, 64, 512, 227),
        (256, 64, 512, 229),
        (32, 32, 500, 233),
    ];
    for (s, h, kmax, seed) in cases {
        let n = kmax + 17;
        let m = MlaInputs::new(s, h, n, kmax, 1.0, Some(50.0), seed, |r, j| {
            mixed_idx(r, j, kmax, n)
        });
        let got = call_mla_prefill_nax(&m.q, m.whole(), &m.idx, &m.sinks, scale).expect("NAX");
        let want = mla_ref(&m.q, m.whole(), &m.idx, &m.sinks, scale);
        assert_mla_parity(&got, &want, tol, &format!("prefill S={s} H={h} k={kmax}"));
    }
}

#[cfg(target_os = "macos")]
#[test]
fn mla_prefill_nax_keyless_nan_slot_zero_and_whole_vs_split_and_row_invariance() {
    if !metal_ok() {
        return;
    }
    let scale = 0.0625f32;
    check_keyless_nan_split(
        32,
        32,
        64,
        512,
        239,
        |r| r % 3 == 0,
        |q, kr, ia, s| call_mla_prefill_nax(q, kr, ia, s, scale).unwrap(),
    );
    for (h, kmax, seed) in [(32usize, 2048usize, 241u64), (64, 512, 251)] {
        let (max_s, n) = (64usize, kmax + 11);
        let idx = |r, j| {
            if (j + r) % 13 == 0 {
                -1
            } else {
                ((j * 3 + r * 17) % n) as i32
            }
        };
        let m = MlaInputs::new(max_s, h, n, kmax, 1.5, None, seed, idx);
        let whole_a = call_mla_prefill_nax(&m.q, m.whole(), &m.idx, &m.sinks, scale).unwrap();
        let split_a = call_mla_prefill_nax(&m.q, m.split(), &m.idx, &m.sinks, scale).unwrap();
        let (whole_all, split_all) = (read_bits(&whole_a), read_bits(&split_a));
        assert_eq!(whole_all, split_all, "H={h} kmax={kmax} Whole vs Split");
        let per = h * MLA_DV;
        for row in [0usize, 1, 15, 31, 63] {
            let (qs, is) = m.slice(row, 1);
            let alone =
                read_bits(&call_mla_prefill_nax(&qs, m.whole(), &is, &m.sinks, scale).unwrap());
            assert_eq!(
                &whole_all[row * per..(row + 1) * per],
                alone.as_slice(),
                "H={h} r={row}"
            );
        }
        if std::env::var("MLX_HY4_MLA_PREFILL_NAX").as_deref() != Ok("0") {
            let k = RowKind::Prefill;
            let dw = gather_sink_attention_for(&m.q, &m.keys, &m.idx, MLA_V, &m.sinks, scale, k)
                .unwrap();
            let ds =
                gather_sink_attention_rows_for(&m.q, m.split(), &m.idx, MLA_V, &m.sinks, scale, k)
                    .unwrap();
            assert_eq!(read_bits(&dw), whole_all, "H={h} Whole dispatch");
            assert_eq!(read_bits(&ds), split_all, "H={h} Split dispatch");
        }
    }
}

/// Covers decode NAX / Metal 3 `v2` accuracy across all `(H, S, kmax)`,
/// `KeyRows::Whole == KeyRows::Split` bit-identity, and trailing `-1` padding
/// determinism across `SINGLE_SPLIT` (`kmax=64, 128`) and multi-split boundaries.
#[test]
fn mla_decode_nax_accuracy_all_s_and_trailing_unlisted_determinism() {
    crate::models::hy_v4_flash::exact_f32();
    let scale = 1.0 / (MLA_D as f32).sqrt();
    let check_pad = |m: &MlaInputs, whole: &MxArray, extra: usize| {
        let pad = m.pad_idx(extra);
        assert_bits_eq(
            whole,
            &mla_run(&m.q, m.whole(), &pad, &m.sinks, scale),
            "W pad",
        );
        assert_bits_eq(
            whole,
            &mla_run(&m.q, m.split(), &pad, &m.sinks, scale),
            "S pad",
        );
    };
    let grid: &[(usize, usize, usize, u64)] = &[
        (16, 1, 17, 503),
        (16, 2, 64, 503),
        (16, 4, 128, 503),
        (16, 16, 512, 503),
        (16, 4, 836, 503),
        (16, 2, 2048, 503),
        (32, 1, 64, 509),
        (32, 2, 128, 509),
        (32, 4, 512, 509),
        (32, 16, 17, 509),
        (32, 16, 836, 509),
        (32, 4, 2048, 509),
        (64, 1, 128, 521),
        (64, 2, 512, 521),
        (64, 4, 17, 521),
        (64, 16, 64, 521),
        (64, 4, 836, 521),
        (64, 2, 2048, 521),
    ];
    for &(h, s, kmax, seed_base) in grid {
        let (n, seed) = (kmax + 19, seed_base + (s * 1009 + kmax * 37) as u64);
        let idx = |r, j| {
            let drop = (r == 1 && j % 17 == 16)
                || (r == 2 && j >= (kmax / 2).max(1))
                || (r == 3 && s >= 4);
            if drop {
                -1
            } else {
                ((j * 7 + r * 13 + 3) % n) as i32
            }
        };
        let m = MlaInputs::new(s, h, n, kmax, 1.5, Some(35.0), seed, idx);
        let whole = mla_run(&m.q, m.whole(), &m.idx, &m.sinks, scale);
        let split = mla_run(&m.q, m.split(), &m.idx, &m.sinks, scale);
        let want = mla_ref(&m.q, m.whole(), &m.idx, &m.sinks, scale);
        assert_mla_parity(
            &whole,
            &want,
            (1e-3, None),
            &format!("H={h} S={s} k={kmax}"),
        );
        assert_bits_eq(&whole, &split, "Whole vs Split");
        let extra = match kmax {
            64 | 128 | 512 | 836 => Some(1),
            2048 => Some(4),
            _ => None,
        };
        if let Some(d) = extra {
            check_pad(&m, &whole, d);
        }
    }
    for (h, ks, kl, seed) in [
        (32usize, 256usize, 257usize, 181u64),
        (32, 512, 513, 187),
        (32, 1024, 1025, 193),
        (32, 1800, 1801, 199),
        (64, 128, 129, 211),
        (64, 512, 513, 223),
    ] {
        let n = kl + 7;
        let m = MlaInputs::new(1, h, n, ks, 1.5, None, seed, |_, j| {
            ((j * 7 + 3) % n) as i32
        });
        let ws = mla_run(&m.q, m.whole(), &m.idx, &m.sinks, scale);
        assert_bits_eq(
            &ws,
            &mla_run(&m.q, m.split(), &m.idx, &m.sinks, scale),
            "Whole vs Split",
        );
        check_pad(&m, &ws, kl - ks);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn mla_prefill_sliding_nax_matches_reference_ring_and_linear() {
    if !metal_ok() {
        return;
    }
    crate::models::hy_v4_flash::exact_f32();
    let (win, scale, tol) = (512usize, 0.0625f32, prefill_tol());
    let slide_ref =
        |q: &MxArray, k: &MxArray, snk: &MxArray, s: usize, pos: (usize, bool, usize, usize)| {
            let idx = if pos.1 {
                window_ring_rows(pos.0, s, pos.2, win)
            } else {
                window_rows(pos.0, s, pos.3, win)
            };
            let ia = MxArray::from_int32(&idx, &[s as i64, win as i64]).unwrap();
            mla_ref(q, KeyRows::Whole(k), &ia, snk, scale)
        };
    let cases = [
        (16usize, 64usize, (0usize, true, 540usize, 0usize), 401u64),
        (17, 64, (500, true, 540, 0), 409),
        (50, 32, (600, true, 650, 0), 419),
        (64, 64, (600, true, 650, 0), 421),
        (128, 64, (700, true, 650, 0), 431),
        (64, 64, (0, false, 0, 0), 433),
        (128, 64, (512, false, 0, 12), 439),
        (128, 32, (1024, false, 0, 513), 443),
    ];
    for (s, h, pos, seed) in cases {
        let mut r = Rng::new(seed);
        let q = arr(&r.vec_f32(s * h * MLA_D, 1.0), &[s, h, MLA_D]);
        let (keys, sinks) = make_slide_kv(&mut r, s, h, pos, Some(40.0));
        let got = slide_nax(&q, None, &keys, &sinks, scale, pos);
        let want = slide_ref(&q, &keys, &sinks, s, pos);
        assert_mla_parity(&got, &want, tol, &format!("sliding S={s} H={h}"));
    }
    let (mut r, h) = (Rng::new(449), 64usize);
    let mut ring = WindowRing::new(win, WINDOW_SLACK, MLA_D as i64);
    ring.append(&arr(&r.vec_f32(600 * MLA_D, 1.0), &[600, MLA_D]), 0)
        .unwrap();
    for (start, s) in [(600usize, 16usize), (616, 64), (680, 128)] {
        let wk = ring
            .append(&arr(&r.vec_f32(s * MLA_D, 1.0), &[s, MLA_D]), start)
            .unwrap();
        let q = arr(&r.vec_f32(s * h * MLA_D, 1.0), &[s, h, MLA_D]);
        let sinks = arr(&r.vec_f32(h, 2.0), &[h]);
        let (keys, pos) = match &wk {
            WindowKeys::Ring { keys, cap } => (*keys, (start, true, *cap, 0)),
            WindowKeys::Linear { keys, first } => (keys, (start, false, 0, *first)),
        };
        let got = slide_nax(&q, None, keys, &sinks, scale, pos);
        let want = slide_ref(&q, keys, &sinks, s, pos);
        assert_mla_parity(
            &got,
            &want,
            (tol.0, None),
            &format!("WindowRing {start}/{s}"),
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn mla_prefill_split_q_matches_combined_bitwise_sliding_and_indexed() {
    crate::models::hy_v4_flash::exact_f32();
    let scale = 1.0 / (MLA_D as f32).sqrt();
    if metal_ok() || std::env::var("MLX_TEST_REQUIRE_NAX").as_deref() == Ok("1") {
        for (s, h, pos, seed) in [
            (17usize, 6usize, (500usize, true, 540usize, 0usize), 701u64),
            (35, 32, (600, true, 650, 0), 709),
            (64, 64, (512, false, 0, 12), 719),
        ] {
            let mut r = Rng::new(seed);
            let (q_abs, q_pe, q_full) = make_split_q(&mut r, s, h, false);
            let (keys, sinks) = make_slide_kv(&mut r, s, h, pos, None);
            let got = slide_nax(&q_abs, Some(&q_pe), &keys, &sinks, scale, pos);
            let want = slide_nax(&q_full, None, &keys, &sinks, scale, pos);
            assert_bits_eq(&got, &want, &format!("sliding split-Q S={s} H={h}"));
        }
    }
    for (s, h, kmax, split_kv, seed) in [
        (19usize, 32usize, 512usize, false, 727u64),
        (64, 64, 512, true, 733),
        (32, 32, 2048, true, 739),
    ] {
        let n = kmax + 37;
        let idx = |r, j| {
            let i = r * kmax + j;
            if i % 41 == 40 {
                -1
            } else {
                ((i * 11 + 5) % n) as i32
            }
        };
        let m = MlaInputs::new(s, h, n, kmax, 1.0, None, seed, idx);
        let (q_abs, q_pe, q_full) = make_split_q(&mut Rng::new(seed ^ 1), s, h, false);
        let kr = if split_kv { m.split() } else { m.whole() };
        let got = mla_split_q_prefill(&q_abs, Some(&q_pe), kr, &m.idx, &m.sinks, scale);
        let want = mla_split_q_prefill(&q_full, None, kr, &m.idx, &m.sinks, scale);
        assert_bits_eq(
            &got,
            &want,
            &format!("idx split-Q S={s} H={h} kv={split_kv}"),
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn mla_prefill_f16_io_matches_f32_cast() {
    if !metal_ok() {
        return;
    }
    crate::models::hy_v4_flash::exact_f32();
    let scale = 1.0 / (MLA_D as f32).sqrt();
    for (s, h, pos, seed) in [
        (19usize, 5usize, (500usize, true, 540usize, 0usize), 811u64),
        (64, 64, (512, false, 0, 12), 821),
    ] {
        let mut r = Rng::new(seed);
        let (a16, p16, _) = make_split_q(&mut r, s, h, true);
        let (a32, p32) = (
            a16.astype(DType::Float32).unwrap(),
            p16.astype(DType::Float32).unwrap(),
        );
        let (keys, sinks) = make_slide_kv(&mut r, s, h, pos, None);
        let f16 = slide_nax(&a16, Some(&p16), &keys, &sinks, scale, pos);
        let f32 = slide_nax(&a32, Some(&p32), &keys, &sinks, scale, pos);
        assert_eq!(
            (f16.dtype().unwrap(), f32.dtype().unwrap()),
            (DType::Float16, DType::Float32)
        );
        assert_bits_eq(&f16, &f32.astype(DType::Float16).unwrap(), "slide f16 IO");
    }

    let (s, h, kmax, n) = (32usize, 32usize, 512usize, 549usize);
    let idx = |r, j| {
        if r % 7 == 0 || (32..64).contains(&j) {
            -1
        } else {
            (((r * kmax + j) * 11 + 5) % n) as i32
        }
    };
    let m = MlaInputs::new(s, h, n, kmax, 1.0, None, 827, idx);
    let (a16, p16, _) = make_split_q(&mut Rng::new(829), s, h, true);
    let (a32, p32) = (
        a16.astype(DType::Float32).unwrap(),
        p16.astype(DType::Float32).unwrap(),
    );
    let f16 = mla_split_q_prefill(&a16, Some(&p16), m.split(), &m.idx, &m.sinks, scale);
    let f32 = mla_split_q_prefill(&a32, Some(&p32), m.split(), &m.idx, &m.sinks, scale);
    assert_eq!(f16.dtype().unwrap(), DType::Float16);
    assert_bits_eq(&f16, &f32.astype(DType::Float16).unwrap(), "indexed f16 IO");

    // Mixed-validity tiles alternating with all-padding tiles: zero s_slots overwrite race.
    let mut alt = vec![-1i32; s * kmax];
    for row in 0..s {
        for tile in (0..kmax / 32).step_by(2) {
            for j in 0..16 {
                alt[row * kmax + tile * 32 + j * 2] = ((row * 17 + tile * 13 + j * 3) % n) as i32;
            }
        }
    }
    let alt_ia = MxArray::from_int32(&alt, &[s as i64, kmax as i64]).unwrap();
    let got = mla_split_q_prefill(&a32, Some(&p32), m.split(), &alt_ia, &m.sinks, scale);
    let q_full = MxArray::concatenate(&a32, &p32, -1).unwrap();
    let want = mla_ref(&q_full, m.split(), &alt_ia, &m.sinks, scale);
    assert_mla_parity(
        &got,
        &want,
        (3e-3, None),
        "alternating mixed/all-padding tiles",
    );
}
