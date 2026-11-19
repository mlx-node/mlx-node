use std::path::PathBuf;

use serde_json::Value;

use super::*;
use crate::models::deepseek_v41::ops::rope_angle;
use crate::models::hy_v4_flash::cache::{LatentBuf, LayerCache, WINDOW_SLACK, WindowRing};
use crate::models::hy_v4_flash::sparse::gather_sink_attention_in;

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

#[test]
fn causal_rows_list_every_earlier_cell() {
    let idx = causal_rows(2, 3);
    assert_eq!(idx, vec![0, 1, 2, -1, -1, 0, 1, 2, 3, -1, 0, 1, 2, 3, 4]);
    assert_eq!(causal_rows(0, 1), vec![0]);
}

#[test]
fn dense_reach_ends_below_top_k() {
    assert!(check_dense_reach(0, 2048, 2048).is_ok());
    assert!(check_dense_reach(2047, 1, 2048).is_ok());
    assert!(check_dense_reach(2047, 2, 2048).is_err());
    assert!(check_dense_reach(2048, 1, 2048).is_err());
    assert!(check_dense_reach(0, 0, 2048).is_err());
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
        top_k: 16,
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

fn cache_for(g: &Geometry, cap: usize) -> LayerCache {
    LayerCache::new(g.attn, g.window, cap, g.row() as i64)
}

/// Runs `hs` through one layer in calls of the given sizes.
fn run_split(
    mla: &Mla,
    w: &AttnWeights,
    cache: &mut LayerCache,
    hs: &[Vec<f32>],
    start: usize,
    sizes: &[usize],
) -> Vec<f32> {
    let mut out = Vec::new();
    let mut at = 0;
    for &n in sizes {
        let y = mla
            .forward(w, &rows_arr(&hs[at..at + n]), start + at, cache)
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
fn indexed_layer_refuses_queries_at_top_k_and_keeps_its_cache() {
    let mut g = geometry(Attn::Indexed);
    g.top_k = 8;
    let (w, _) = weights(&g, 9);
    let mla = Mla::new(g.clone(), &w).unwrap();
    let hs = inputs(9, 2);
    let mut cache = cache_for(&g, 16);
    assert!(mla.forward(&w, &rows_arr(&hs), 0, &mut cache).is_err());
    assert_eq!(cache.len(), 0);
    run_split(&mla, &w, &mut cache, &hs[..8], 0, &[8]);
    assert!(mla.forward(&w, &rows_arr(&hs[8..]), 8, &mut cache).is_err());
    assert_eq!(cache.len(), 8);
}

#[test]
fn indexed_layer_refuses_a_call_past_the_context() {
    let g = geometry(Attn::Indexed);
    let (w, _) = weights(&g, 13);
    let mla = Mla::new(g.clone(), &w).unwrap();
    let hs = inputs(6, 4);
    let mut cache = cache_for(&g, 5);
    assert!(mla.forward(&w, &rows_arr(&hs), 0, &mut cache).is_err());
    assert_eq!(cache.len(), 0);
}

#[test]
fn forward_refuses_the_other_cache_kind() {
    let g = geometry(Attn::Sliding);
    let (w, _) = weights(&g, 1);
    let mla = Mla::new(g.clone(), &w).unwrap();
    let mut cache = LayerCache::new(Attn::Indexed, g.window, 16, g.row() as i64);
    assert!(
        mla.forward(&w, &rows_arr(&inputs(1, 1)), 0, &mut cache)
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
