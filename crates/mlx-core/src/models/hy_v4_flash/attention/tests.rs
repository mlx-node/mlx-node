use std::path::PathBuf;

use serde_json::Value;

use super::*;
use crate::models::deepseek_v41::ops::rope_angle;
use crate::models::hy_v4_flash::cache::{LatentBuf, LayerCache, WINDOW_SLACK, WindowRing};
use crate::models::hy_v4_flash::indexer::{Indexer, IndexerGeometry, IndexerState, Indexing};
use crate::models::hy_v4_flash::moe::RowKind;
use crate::models::hy_v4_flash::sparse::gather_sink_attention_in;
use crate::models::hy_v4_flash::weights::IndexerWeights;

fn ops() -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hy-v4-flash/ops.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

#[test]
fn window_rows_match_fixture_index_rows() {
    let o = ops();
    let w = &o["window"];
    let window = w["window"].as_u64().unwrap() as usize;
    for r in w["index_rows"].as_array().unwrap() {
        let p = r["p"].as_u64().unwrap() as usize;
        let first = r["first"].as_u64().unwrap() as usize;
        let row: Vec<i64> = window_rows(p, 1, first, window)
            .into_iter()
            .map(i64::from)
            .collect();
        assert_eq!(row.len(), window, "p {p}");
        assert_eq!(row[..3].to_vec(), ints(&r["head"]), "p {p} head");
        assert_eq!(row[window - 3..].to_vec(), ints(&r["tail"]), "p {p} tail");
        let valid = row.iter().filter(|&&k| k >= 0).count() as u64;
        assert_eq!(valid, r["n_valid"].as_u64().unwrap(), "p {p} valid");
    }
}

#[test]
fn window_rows_match_fixture_visibility() {
    let o = ops();
    let w = &o["window"];
    let window = w["window"].as_u64().unwrap() as usize;
    for c in w["cases"].as_array().unwrap() {
        let t = c["t"].as_u64().unwrap() as usize;
        let row = window_rows(t, 1, 0, window);
        let keys: Vec<i64> = row
            .iter()
            .filter(|&&k| k >= 0)
            .map(|&k| i64::from(k))
            .collect();
        let lo = c["lo"].as_i64().unwrap();
        let hi = c["hi"].as_i64().unwrap();
        assert_eq!(keys, (lo..=hi).collect::<Vec<_>>(), "t {t}");
        assert_eq!(keys.len() as u64, c["n_keys"].as_u64().unwrap(), "t {t}");
        for (dist, vis) in c["dist_visible"].as_object().unwrap() {
            let d: i64 = dist.parse().unwrap();
            let k = t as i64 - d;
            assert_eq!(keys.contains(&k), vis.as_bool().unwrap(), "t {t} dist {d}");
        }
    }
}

#[test]
fn window_rows_at_510_to_513() {
    for p in 510..=513 {
        let row = window_rows(p, 1, 0, 512);
        let lo = (p + 1).saturating_sub(512);
        let want: Vec<i32> = (lo..=p)
            .map(|k| k as i32)
            .chain(std::iter::repeat_n(-1, 512 - (p + 1 - lo)))
            .collect();
        assert_eq!(row, want, "p {p}");
    }
    let two = window_rows(511, 2, 0, 512);
    assert_eq!(two[..512].to_vec(), window_rows(511, 1, 0, 512));
    assert_eq!(two[512..].to_vec(), window_rows(512, 1, 0, 512));
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

const D: usize = 12;
const Q_LORA: usize = 6;

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
        rope_theta: match attn {
            Attn::Sliding => 1e4,
            Attn::Indexed => 1e7,
        },
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

fn arr(v: &[f32], shape: &[usize]) -> MxArray {
    let s: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
    MxArray::from_float32(v, &s).unwrap()
}

fn dense(v: &[f32], rows: usize, cols: usize) -> Dense {
    Dense::Array(arr(v, &[rows, cols]))
}

fn weights(g: &Geometry, seed: u64) -> (AttnWeights, RefW) {
    let mut r = Rng(seed);
    let (h, l, v) = (g.n_heads, g.kv_lora, g.head_v);
    let qk = g.head_qk();
    let mat = |r: &mut Rng, rows: usize, cols: usize| -> Vec<f32> {
        r.vec(rows * cols, 1.5 / (cols as f32).sqrt())
    };
    let norm_w = |r: &mut Rng, n: usize| -> Vec<f32> {
        r.vec(n, 0.5).into_iter().map(|x| 1.0 + x).collect()
    };
    let rw = RefW {
        norm: norm_w(&mut r, D),
        sinks: r.vec(h, 1.0),
        q_a: mat(&mut r, Q_LORA, D),
        q_a_norm: norm_w(&mut r, Q_LORA),
        q_b: mat(&mut r, h * qk, Q_LORA),
        kv_a: mat(&mut r, g.row(), D),
        kv_a_norm: norm_w(&mut r, l),
        k_b: mat(&mut r, h * l, g.nope),
        v_b: mat(&mut r, h * v, l),
        output: mat(&mut r, D, h * v),
        gate: mat(&mut r, h * v, D),
        post_norm: norm_w(&mut r, D),
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
    };
    (w, rw)
}

fn mv(w: &[f32], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    assert_eq!(w.len(), rows * cols);
    assert_eq!(x.len(), cols);
    (0..rows)
        .map(|r| {
            w[r * cols..(r + 1) * cols]
                .iter()
                .zip(x)
                .map(|(&a, &b)| f64::from(a) * b)
                .sum()
        })
        .collect()
}

fn rms(x: &[f64], w: &[f32], eps: f32) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ms + f64::from(eps)).sqrt();
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
        v[2 * i] = a * c - b * s;
        v[2 * i + 1] = a * s + b * c;
    }
}

fn f16r(v: f64) -> f64 {
    half::f16::from_f64(v).to_f64()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The layer output for every token of `hs` (positions from 0), f64 math
/// with the cached rows rounded to f16.
fn reference(g: &Geometry, r: &RefW, hs: &[Vec<f32>]) -> Vec<Vec<f64>> {
    let (h, l, nope, rd, v) = (g.n_heads, g.kv_lora, g.nope, g.rope_dim, g.head_v);
    let qk = nope + rd;
    let inv = inv_freqs(rd, g.rope_theta, None);
    let mut keys: Vec<Vec<f64>> = Vec::new();
    let mut outs = Vec::new();
    for (t, row) in hs.iter().enumerate() {
        let hx: Vec<f64> = row.iter().map(|&x| f64::from(x)).collect();
        let x = rms(&hx, &r.norm, g.eps);
        let qr = rms(&mv(&r.q_a, Q_LORA, D, &x), &r.q_a_norm, g.eps);
        let mut q = mv(&r.q_b, h * qk, Q_LORA, &qr);
        for hh in 0..h {
            rope(&mut q[hh * qk + nope..(hh + 1) * qk], t, &inv);
        }
        let ckv = mv(&r.kv_a, l + rd, D, &x);
        let mut k = rms(&ckv[..l], &r.kv_a_norm, g.eps);
        let mut pe = ckv[l..].to_vec();
        rope(&mut pe, t, &inv);
        k.extend(pe);
        keys.push(k.into_iter().map(f16r).collect());
        let lo = match g.attn {
            Attn::Sliding => (t + 1).saturating_sub(g.window),
            Attn::Indexed => 0,
        };
        let mut o = vec![0.0; h * v];
        for hh in 0..h {
            let wkb = &r.k_b[hh * l * nope..(hh + 1) * l * nope];
            let mut qq = mv(wkb, l, nope, &q[hh * qk..hh * qk + nope]);
            qq.extend_from_slice(&q[hh * qk + nope..(hh + 1) * qk]);
            let sc: Vec<f64> = (lo..=t)
                .map(|j| f64::from(g.scale) * dot(&qq, &keys[j]))
                .collect();
            let m = sc.iter().copied().fold(-1e30, f64::max);
            let e: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
            let den = e.iter().sum::<f64>() + (f64::from(r.sinks[hh]) - m).exp();
            let mut ol = vec![0.0; l];
            for (i, j) in (lo..=t).enumerate() {
                for (c, x) in ol.iter_mut().enumerate() {
                    *x += e[i] * keys[j][c];
                }
            }
            for x in &mut ol {
                *x /= den;
            }
            let wvb = &r.v_b[hh * v * l..(hh + 1) * v * l];
            o[hh * v..(hh + 1) * v].copy_from_slice(&mv(wvb, v, l, &ol));
        }
        let gate = mv(&r.gate, h * v, D, &x);
        for (x, gt) in o.iter_mut().zip(gate) {
            *x *= 1.0 / (1.0 + (-gt).exp());
        }
        let y = mv(&r.output, D, h * v, &o);
        outs.push(rms(&y, &r.post_norm, g.eps));
    }
    outs
}

fn inputs(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut r = Rng(seed);
    (0..n).map(|_| r.vec(D, 2.0)).collect()
}

fn rows_arr(hs: &[Vec<f32>]) -> MxArray {
    let flat: Vec<f32> = hs.iter().flatten().copied().collect();
    arr(&flat, &[hs.len(), D])
}

fn read(x: &MxArray) -> Vec<f32> {
    x.astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .as_ref()
        .to_vec()
}

fn assert_rows(got: &[f32], want: &[Vec<f64>], what: &str) {
    assert_eq!(got.len(), want.len() * D, "{what} len");
    for (t, w) in want.iter().enumerate() {
        for (c, &wv) in w.iter().enumerate() {
            let gv = f64::from(got[t * D + c]);
            let tol = 2e-3 + 2e-3 * wv.abs();
            assert!(
                (gv - wv).abs() <= tol,
                "{what} row {t} col {c}: {gv} vs {wv}"
            );
        }
    }
}

/// An indexer whose `n_top` (3 groups of 4) keeps every query of up to 16
/// tokens dense.
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
    let mut r = Rng(seed);
    let (h, d, kp) = (ig.n_heads, ig.head_dim, ig.kpool);
    let iw = IndexerWeights {
        attn_k: dense(&r.vec(d * D, 0.5), d, D),
        k_norm: arr(&r.vec(d, 0.5), &[d]),
        q_b: dense(&r.vec(h * d * Q_LORA, 0.5), h * d, Q_LORA),
        proj: dense(&r.vec(h * D, 0.5), h, D),
        compressor_value: arr(&r.vec(d * d, 0.5), &[d, d]),
        compressor_gate: arr(&r.vec(d * d, 0.5), &[d, d]),
        compressor_ape: arr(&r.vec(kp * d, 0.5), &[kp, d]),
        compressor_norm: arr(&r.vec(d, 0.5), &[d]),
    };
    let ix = Indexer::new(ig.clone(), &iw).unwrap();
    (ix, iw, ig)
}

/// One layer's state: the attention cache and, for an indexed layer, its
/// indexer.
struct Layer {
    cache: LayerCache,
    index: Option<(Indexer, IndexerWeights, IndexerState)>,
}

impl Layer {
    fn len(&self) -> usize {
        self.cache.len()
    }

    fn is_empty(&self) -> bool {
        self.cache.is_empty()
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

fn cache_for(g: &Geometry, cap: usize) -> Layer {
    let index = match g.attn {
        Attn::Sliding => None,
        Attn::Indexed => {
            let (ix, iw, ig) = dense_indexer(0x51);
            let st = IndexerState::new(cap, &ig);
            Some((ix, iw, st))
        }
    };
    Layer {
        cache: LayerCache::new(g.attn, g.window, cap, g.row() as i64),
        index,
    }
}

/// Runs `hs` through one layer in calls of the given sizes.
fn run_split(
    mla: &Mla,
    w: &AttnWeights,
    cache: &mut Layer,
    hs: &[Vec<f32>],
    start: usize,
    sizes: &[usize],
) -> Vec<f32> {
    let mut out = Vec::new();
    let mut at = 0;
    for &n in sizes {
        let y = cache
            .forward(mla, w, &rows_arr(&hs[at..at + n]), start + at)
            .unwrap();
        out.extend(read(&y));
        at += n;
    }
    assert_eq!(at, hs.len());
    out
}

fn check_layer(attn: Attn, seed: u64) {
    let g = geometry(attn);
    let (w, rw) = weights(&g, seed);
    let mla = Mla::new(g.clone(), &w).unwrap();
    let hs = inputs(9, seed ^ 0x9e37);
    let want = reference(&g, &rw, &hs);
    for sizes in [vec![9], vec![1, 3, 5], vec![1; 9], vec![4, 1, 4]] {
        let mut cache = cache_for(&g, 16);
        let got = run_split(&mla, &w, &mut cache, &hs, 0, &sizes);
        assert_rows(&got, &want, &format!("{attn:?} split {sizes:?}"));
        assert_eq!(cache.len(), 9);
        if attn == Attn::Indexed {
            assert_eq!(cache.index_len(), Some(9));
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
    for attn in [Attn::Sliding, Attn::Indexed] {
        let g = geometry(attn);
        let (w, rw) = weights(&g, 23);
        let mla = Mla::new(g.clone(), &w).unwrap();
        let hs = inputs(9, 5);
        let want = reference(&g, &rw, &hs);
        let mut cache = cache_for(&g, 16);
        run_split(&mla, &w, &mut cache, &hs, 0, &[9]);
        cache.truncate(9 - WINDOW_SLACK).unwrap();
        assert_eq!(cache.len(), 6);
        let got = run_split(&mla, &w, &mut cache, &hs[6..], 6, &[1, 2]);
        assert_rows(&got, &want[6..], &format!("{attn:?} replay"));
    }
}

#[test]
fn sliding_ring_refuses_a_truncate_past_its_slack() {
    let g = geometry(Attn::Sliding);
    let (w, _) = weights(&g, 3);
    let mla = Mla::new(g.clone(), &w).unwrap();
    let hs = inputs(9, 8);
    let mut cache = cache_for(&g, 16);
    run_split(&mla, &w, &mut cache, &hs, 0, &[9]);
    assert!(cache.truncate(9 - WINDOW_SLACK - 1).is_err());
    assert_eq!(cache.len(), 9);
    cache.truncate(0).unwrap();
    assert!(cache.is_empty());
}

#[test]
fn indexed_layer_needs_its_indexer_and_keeps_its_cache() {
    let g = geometry(Attn::Indexed);
    let (w, _) = weights(&g, 9);
    let mla = Mla::new(g.clone(), &w).unwrap();
    let hs = inputs(9, 2);
    let mut cache = LayerCache::new(g.attn, g.window, 16, g.row() as i64);
    assert!(
        mla.forward(&w, &rows_arr(&hs), 0, &mut cache, None)
            .is_err()
    );
    assert_eq!(cache.len(), 0);
    let mut layer = cache_for(&g, 16);
    run_split(&mla, &w, &mut layer, &hs[..8], 0, &[8]);
    assert!(layer.forward(&mla, &w, &rows_arr(&hs[8..]), 7).is_err());
    assert_eq!((layer.len(), layer.index_len()), (8, Some(8)));
}

#[test]
fn indexed_layer_refuses_a_call_past_the_context() {
    let g = geometry(Attn::Indexed);
    let (w, _) = weights(&g, 13);
    let mla = Mla::new(g.clone(), &w).unwrap();
    let hs = inputs(6, 4);
    let mut cache = cache_for(&g, 5);
    assert!(cache.forward(&mla, &w, &rows_arr(&hs), 0).is_err());
    assert_eq!((cache.len(), cache.index_len()), (0, Some(0)));
}

#[test]
fn forward_refuses_the_other_cache_kind() {
    let g = geometry(Attn::Sliding);
    let (w, _) = weights(&g, 1);
    let mla = Mla::new(g.clone(), &w).unwrap();
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

fn positions(rows: std::ops::Range<usize>, width: usize) -> MxArray {
    let n = rows.len();
    let v: Vec<f32> = rows
        .flat_map(|p| std::iter::repeat_n(p as f32, width))
        .collect();
    arr(&v, &[n, width])
}

fn first_col(x: &MxArray, width: usize) -> Vec<f32> {
    read(x).chunks(width).map(|c| c[0]).collect()
}

#[test]
fn window_ring_keeps_window_plus_slack() {
    let mut ring = WindowRing::new(4, WINDOW_SLACK, 2);
    let (keys, first) = ring.append(&positions(0..5, 2), 0).unwrap();
    assert_eq!((first, first_col(&keys, 2)), (0, vec![0., 1., 2., 3., 4.]));
    let (keys, first) = ring.append(&positions(5..9, 2), 5).unwrap();
    assert_eq!(first, 0);
    assert_eq!(keys.shape_at(0).unwrap(), 9);
    assert_eq!(ring.kept(), 7);
    let (keys, first) = ring.append(&positions(9..10, 2), 9).unwrap();
    assert_eq!(
        (first, first_col(&keys, 2)),
        (2, (2..10).map(|p| p as f32).collect())
    );
    assert!(ring.append(&positions(0..1, 2), 4).is_err());

    let snap = ring.clone();
    ring.truncate(7).unwrap();
    assert_eq!((ring.len(), ring.kept()), (7, 4));
    let mut v = positions(7..8, 2);
    v = v.mul_scalar(10.0).unwrap();
    let (keys, first) = ring.append(&v, 7).unwrap();
    assert_eq!((first, first_col(&keys, 2)), (3, vec![3., 4., 5., 6., 70.]));
    assert!(ring.truncate(5).is_err());
    assert_eq!(snap.len(), 10);
    assert_eq!(snap.kept(), 7);
}

#[test]
fn latent_buf_writes_rows_and_truncates_by_length() {
    let mut buf = LatentBuf::new(6, 2);
    assert!(buf.rows().is_err());
    buf.write(&positions(0..3, 2), 0).unwrap();
    buf.write(&positions(3..5, 2), 3).unwrap();
    assert_eq!(buf.len(), 5);
    let rows = buf.rows().unwrap();
    assert_eq!(rows.dtype().unwrap(), DType::Float16);
    assert_eq!(first_col(rows, 2)[..5].to_vec(), vec![0., 1., 2., 3., 4.]);
    buf.truncate(2).unwrap();
    buf.write(&positions(20..21, 2), 2).unwrap();
    assert_eq!(
        first_col(buf.rows().unwrap(), 2)[..4].to_vec(),
        vec![0., 1., 20., 3.]
    );
    assert!(buf.write(&positions(0..4, 2), 3).is_err());
    assert!(buf.write(&positions(0..1, 2), 2).is_err());
    assert!(buf.truncate(4).is_err());
    assert_eq!(buf.len(), 3);
    assert_eq!(buf.nbytes(), 6 * 2 * 2);
}

#[test]
fn gather_core_matches_host_softmax_with_sinks() {
    let (s, h, d, n, kmax, vw) = (5usize, 2usize, 6usize, 7usize, 4usize, 4usize);
    let mut r = Rng(31);
    let q = r.vec(s * h * d, 1.5);
    let keys: Vec<f32> = r
        .vec(n * d, 1.5)
        .into_iter()
        .map(|x| f16r(f64::from(x)) as f32)
        .collect();
    let sinks = r.vec(h, 1.0);
    let idx: Vec<i32> = vec![
        0, 1, 2, 3, //
        6, -1, 2, -1, //
        -1, -1, -1, -1, //
        5, 5, 0, -1, //
        3, 4, 6, 1,
    ];
    let scale = 0.37f32;
    let qa = arr(&q, &[s, h, d]);
    let ka = arr(&keys, &[n, d]).astype(DType::Float16).unwrap();
    let ia = MxArray::from_int32(&idx, &[s as i64, kmax as i64]).unwrap();
    let sa = arr(&sinks, &[h]);
    let one = gather_sink_attention(&qa, &ka, &ia, vw as i64, &sa, scale).unwrap();
    let tiny = gather_sink_attention_in(&qa, &ka, &ia, vw as i64, &sa, scale, 1).unwrap();
    let (one, tiny) = (read(&one), read(&tiny));
    assert_eq!(one.len(), s * h * vw);
    for (a, b) in one.iter().zip(&tiny) {
        assert!((a - b).abs() <= 1e-6, "chunked {b} vs {a}");
    }
    for row in 0..s {
        let ids: Vec<usize> = idx[row * kmax..(row + 1) * kmax]
            .iter()
            .filter(|&&j| j >= 0)
            .map(|&j| j as usize)
            .collect();
        for hh in 0..h {
            let qv: Vec<f64> = q[(row * h + hh) * d..(row * h + hh + 1) * d]
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            let kv = |j: usize| -> Vec<f64> {
                keys[j * d..(j + 1) * d]
                    .iter()
                    .map(|&x| f64::from(x))
                    .collect()
            };
            let sc: Vec<f64> = ids
                .iter()
                .map(|&j| f64::from(scale) * dot(&qv, &kv(j)))
                .collect();
            let m = sc.iter().copied().fold(-1e30, f64::max);
            let e: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
            let den = e.iter().sum::<f64>() + (f64::from(sinks[hh]) - m).exp();
            for c in 0..vw {
                let want: f64 = ids
                    .iter()
                    .zip(&e)
                    .map(|(&j, &p)| p * f64::from(keys[j * d + c]))
                    .sum::<f64>()
                    / den;
                let got = f64::from(one[(row * h + hh) * vw + c]);
                assert!(
                    (got - want).abs() <= 1e-5 + 1e-5 * want.abs(),
                    "row {row} head {hh} col {c}: {got} vs {want}"
                );
            }
        }
    }
}

fn bits(x: &MxArray) -> Vec<u32> {
    read(x).iter().map(|v| v.to_bits()).collect()
}

/// A query row gives the same bits alone as inside a call of 2..4 rows, at
/// the real sliding (64 heads, 512 keys) and indexed (32 heads, 2048 keys)
/// sizes.
#[test]
fn gather_core_row_bits_do_not_depend_on_the_call_height() {
    for (h, kmax) in [(64usize, 512usize), (32, 2048)] {
        let (s, d, vw, n) = (4usize, 576usize, 512i64, kmax + 5);
        let mut r = Rng(53 + h as u64);
        let q = arr(&r.vec(s * h * d, 1.5), &[s, h, d]);
        let keys = arr(&r.vec(n * d, 1.5), &[n, d])
            .astype(DType::Float16)
            .unwrap();
        let sinks = arr(&r.vec(h, 1.0), &[h]);
        let idx: Vec<i32> = (0..s * kmax)
            .map(|i| {
                let (row, j) = (i / kmax, i % kmax);
                if j + row >= kmax {
                    -1
                } else {
                    ((j + 3 * row) % n) as i32
                }
            })
            .collect();
        let ia = MxArray::from_int32(&idx, &[s as i64, kmax as i64]).unwrap();
        let scale = 1.0 / (d as f32).sqrt();
        let (hi, di, ki) = (h as i64, d as i64, kmax as i64);
        let call = |lo: i64, rows: i64| {
            let qs = q.slice(&[lo, 0, 0], &[lo + rows, hi, di]).unwrap();
            let is = ia.slice(&[lo, 0], &[lo + rows, ki]).unwrap();
            bits(&gather_sink_attention(&qs, &keys, &is, vw, &sinks, scale).unwrap())
        };
        let per = h * vw as usize;
        let alone: Vec<Vec<u32>> = (0..s as i64).map(|j| call(j, 1)).collect();
        for rows in 2..=s {
            let all = call(0, rows as i64);
            for (j, want) in alone.iter().enumerate().take(rows) {
                assert!(
                    all[j * per..(j + 1) * per] == want[..],
                    "h {h} kmax {kmax}: row {j} of {rows} differs from the row alone"
                );
            }
        }
    }
}

/// An unlisted id reads key row 0; a non-finite row there leaves the
/// output finite and bit-equal to the run with that row zeroed.
#[test]
fn gather_core_ignores_a_non_finite_unlisted_row() {
    let (s, h, d, n, kmax, vw) = (3usize, 2usize, 6usize, 7usize, 4usize, 4i64);
    let mut r = Rng(59);
    let qa = arr(&r.vec(s * h * d, 1.5), &[s, h, d]);
    let mut keys = r.vec(n * d, 1.5);
    let sa = arr(&r.vec(h, 1.0), &[h]);
    let idx: Vec<i32> = vec![
        3, -1, 2, 6, //
        -1, -1, -1, -1, //
        5, 4, -1, 1,
    ];
    let ia = MxArray::from_int32(&idx, &[s as i64, kmax as i64]).unwrap();
    let run = |keys: &[f32]| {
        let ka = arr(keys, &[n, d]).astype(DType::Float16).unwrap();
        bits(&gather_sink_attention(&qa, &ka, &ia, vw, &sa, 0.37).unwrap())
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
    assert!(got.iter().all(|&b| f32::from_bits(b).is_finite()));
    assert_eq!(got, want);
}

#[test]
fn gather_core_refuses_mismatched_shapes() {
    let q = MxArray::zeros(&[2, 2, 6], None).unwrap();
    let k = MxArray::zeros(&[3, 6], Some(DType::Float16)).unwrap();
    let sinks = MxArray::zeros(&[2], None).unwrap();
    let idx = MxArray::from_int32(&[0, 1, 2, 0], &[2, 2]).unwrap();
    assert!(gather_sink_attention(&q, &k, &idx, 4, &sinks, 1.0).is_ok());
    let short = MxArray::from_int32(&[0, 1], &[1, 2]).unwrap();
    assert!(gather_sink_attention(&q, &k, &short, 4, &sinks, 1.0).is_err());
    assert!(gather_sink_attention(&q, &k, &idx, 7, &sinks, 1.0).is_err());
    let narrow = MxArray::zeros(&[3, 5], Some(DType::Float16)).unwrap();
    assert!(gather_sink_attention(&q, &narrow, &idx, 4, &sinks, 1.0).is_err());
    let sinks3 = MxArray::zeros(&[3], None).unwrap();
    assert!(gather_sink_attention(&q, &k, &idx, 4, &sinks3, 1.0).is_err());
}

/// Affine-8 g32 codes (`Dense::Qmm`) of `v [rows, cols]`.
fn qmm(v: &[f32], rows: usize, cols: usize) -> Dense {
    let q = crate::models::hy_v4_flash::moe::AffineProj::quantize(&arr(v, &[1, rows, cols]), 32, 8)
        .unwrap();
    let sq = |a: &MxArray| {
        a.reshape(&[a.shape_at(1).unwrap(), a.shape_at(2).unwrap()])
            .unwrap()
    };
    Dense::Qmm {
        w: sq(&q.w),
        scales: sq(&q.s).astype(DType::Float32).unwrap(),
        biases: sq(&q.b).astype(DType::Float32).unwrap(),
    }
}

/// With Q8 projections (`k_b`, `v_b` grouped), a row alone projects and
/// finishes to the bits of the same row in a 3-row call: every Q8 matrix
/// sees at least two rows, so both run `qmv_wide`.
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
    let (dim, q_lora) = (64, 64);
    let (h, l, v) = (g.n_heads, g.kv_lora, g.head_v);
    let mut r = Rng(71);
    let mat = |r: &mut Rng, rows: usize, cols: usize| {
        qmm(&r.vec(rows * cols, 1.5 / (cols as f32).sqrt()), rows, cols)
    };
    let norm = |r: &mut Rng, n: usize| {
        let v: Vec<f32> = r.vec(n, 0.5).into_iter().map(|x| 1.0 + x).collect();
        arr(&v, &[n])
    };
    let w = AttnWeights {
        norm: norm(&mut r, dim),
        sinks: arr(&r.vec(h, 1.0), &[h]),
        q_a: mat(&mut r, q_lora, dim),
        q_a_norm: norm(&mut r, q_lora),
        q_b: mat(&mut r, h * g.head_qk(), q_lora),
        kv_a: mat(&mut r, g.row(), dim),
        kv_a_norm: norm(&mut r, l),
        k_b: mat(&mut r, h * l, g.nope),
        v_b: mat(&mut r, h * v, l),
        output: mat(&mut r, dim, h * v),
        gate: mat(&mut r, h * v, dim),
        post_norm: norm(&mut r, dim),
    };
    let mla = Mla::new(g, &w).unwrap();
    let hs = r.vec(3 * dim, 2.0);
    let lat = r.vec(3 * h * l, 1.0);
    let start = 9;
    let all = mla.project(&w, &arr(&hs, &[3, dim]), start).unwrap();
    let out = mla.finish(&w, &all.x, &arr(&lat, &[3, h, l])).unwrap();
    let row = |a: &MxArray, i: usize| {
        let v = read(a);
        let per = v.len() / 3;
        v[i * per..(i + 1) * per].to_vec()
    };
    for i in 0..3 {
        let one = mla
            .project(&w, &arr(&hs[i * dim..(i + 1) * dim], &[1, dim]), start + i)
            .unwrap();
        for (what, a, b) in [
            ("x", &all.x, &one.x),
            ("qr", &all.qr, &one.qr),
            ("q", &all.q, &one.q),
            ("kv", &all.kv, &one.kv),
        ] {
            assert_eq!(row(a, i), read(b), "row {i} {what}");
        }
        let fo = mla
            .finish(
                &w,
                &one.x,
                &arr(&lat[i * h * l..(i + 1) * h * l], &[1, h, l]),
            )
            .unwrap();
        assert_eq!(row(&out, i), read(&fo), "row {i} finish");
    }
}
