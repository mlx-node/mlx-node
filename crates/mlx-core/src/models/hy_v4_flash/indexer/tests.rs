use std::path::PathBuf;

use serde_json::Value;

use super::*;
use crate::models::deepseek_v41::ops::rope_angle;
use crate::models::hy_v4_flash::attention::{Geometry, Mla};
use crate::models::hy_v4_flash::cache::LayerCache;
use crate::models::hy_v4_flash::weights::AttnWeights;

/// A prefill call of `x`/`qr` at `start`.
fn prefill<'a>(x: &'a MxArray, qr: &'a MxArray, start: usize) -> Call<'a> {
    Call {
        x,
        qr,
        start,
        kind: RowKind::Prefill,
    }
}

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

fn f32s(v: &Value) -> Vec<f32> {
    let mut out = Vec::new();
    flatten(v, &mut out);
    out
}

fn flatten(v: &Value, out: &mut Vec<f32>) {
    match v {
        Value::Array(a) => a.iter().for_each(|x| flatten(x, out)),
        x => out.push(x.as_f64().unwrap() as f32),
    }
}

/// The real file's indexer (I layer: 16 heads of 128, rope 64, kpool 4).
fn real() -> IndexerGeometry {
    IndexerGeometry {
        n_heads: 16,
        head_dim: 128,
        rope_dim: 64,
        kpool: 4,
        n_top: 511,
        select_tail: true,
        eps: 1e-5,
        weight_scale: (2048f32).sqrt().recip(),
        rope_theta: 1e7,
    }
}

fn close(got: f32, want: f32, abs: f64, rel: f64, what: &str) {
    let (g, w) = (f64::from(got), f64::from(want));
    assert!((g - w).abs() <= abs + rel * w.abs(), "{what}: {g} vs {w}");
}

#[test]
fn row_rules_match_kpool_splits_rows() {
    let g = real();
    let f = fixture("kpool-splits.json");
    let check = |r: &Value| {
        let t = u(&r["t"]);
        assert_eq!(g.visible(t), u(&r["visible_groups"]), "t {t} visible");
        assert_eq!(g.picked(t), u(&r["picked_groups"]), "t {t} picked");
        assert_eq!(!g.is_dense(t), r["topk"].as_bool().unwrap(), "t {t} topk");
        assert_eq!(g.is_dense(t), r["dense"].as_bool().unwrap(), "t {t} dense");
        assert_eq!(t / g.kpool, u(&r["own_group"]), "t {t} own group");
        assert_eq!(g.tail_cells(t).len(), u(&r["own_cells"]), "t {t} own cells");
        assert_eq!(g.cells(t), u(&r["cells"]), "t {t} cells");
        assert!(g.cells(t) <= g.kmax());
    };
    for r in f["rows"].as_array().unwrap() {
        check(r);
    }
    let calls = f["chain"]["calls"]
        .as_array()
        .unwrap()
        .iter()
        .chain(
            f["two_call_splits"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|s| s["calls"].as_array().unwrap()),
        )
        .chain(f["decode_after_2045"].as_array().unwrap());
    for c in calls {
        for key in ["first_row", "last_row"] {
            if !c[key].is_null() {
                check(&c[key]);
            }
        }
    }
}

fn check_call(g: &IndexerGeometry, c: &Value, budget: usize) {
    let (s, e) = (u(&c["s"]), u(&c["e"]));
    let what = format!("call [{s}, {e})");
    let p = g.plan(s, e - s);
    let closed = usizes(&c["groups_closed_by_call"]);
    assert_eq!(vec![p.groups.start, p.groups.end], closed, "{what} groups");
    assert_eq!(usizes(&c["pooled_write_range"]), closed, "{what} writes");
    assert_eq!(p.groups.start, u(&c["groups_closed_before"]), "{what}");
    assert_eq!(p.groups.end, u(&c["groups_closed_after"]), "{what}");
    assert_eq!(
        p.open_in.collect::<Vec<_>>(),
        usizes(&c["tail_in"]),
        "{what} tail in"
    );
    assert_eq!(
        p.tail_out.clone().collect::<Vec<_>>(),
        usizes(&c["tail_out"]),
        "{what} tail out"
    );
    assert!(p.tail_out.len() < g.kpool);
    assert_eq!(p.cols, u(&c["score_cols"]), "{what} cols");
    assert_eq!(
        score_row_bytes(g.n_heads, p.cols),
        u(&c["score_row_bytes"]),
        "{what} row bytes"
    );
    let rows = e - s;
    let per = score_rows(g.n_heads, p.cols, budget).min(rows);
    assert_eq!(
        per,
        u(&c["score_rows_per_sub_chunk"]),
        "{what} sub-chunk rows"
    );
    assert_eq!(
        rows.div_ceil(per),
        u(&c["score_sub_chunks"]),
        "{what} sub-chunks"
    );
    assert_eq!(
        p.sparse.len(),
        u(&c["rows_needing_topk"]),
        "{what} top-k rows"
    );
    assert_eq!(p.dense.len() + p.sparse.len(), rows);
    let max_cells = (s..e).map(|t| g.cells(t)).max().unwrap();
    assert_eq!(max_cells, u(&c["max_cells"]), "{what} max cells");
    assert!(max_cells <= p.kmax, "{what} width");
    if p.sparse.is_empty() {
        assert_eq!(p.kmax, e, "{what} dense width");
    } else {
        assert_eq!(p.kmax, g.kmax(), "{what} sparse width");
    }
}

#[test]
fn call_plans_match_kpool_splits() {
    let g = real();
    let f = fixture("kpool-splits.json");
    let geo = &f["geometry"];
    assert_eq!(u(&geo["kpool"]), g.kpool);
    assert_eq!(u(&geo["n_top_pool"]), g.n_top);
    assert_eq!(u(&geo["indexer_heads"]), g.n_heads);
    let budget = u(&geo["score_sub_chunk_budget"]);
    assert_eq!(budget, INDEX_CHUNK_BYTES);
    for c in f["chain"]["calls"].as_array().unwrap() {
        check_call(&g, c, budget);
    }
    for split in f["two_call_splits"].as_array().unwrap() {
        let calls = split["calls"].as_array().unwrap();
        for c in calls {
            check_call(&g, c, budget);
        }
        let last = calls.last().unwrap();
        let one = g.plan(0, u(&last["e"]));
        assert_eq!(one.groups.end, u(&last["groups_closed_after"]));
        assert_eq!(one.tail_out.collect::<Vec<_>>(), usizes(&last["tail_out"]));
    }
    for c in f["decode_after_2045"].as_array().unwrap() {
        check_call(&g, c, budget);
    }
    for (len, st) in f["state_after_len"].as_object().unwrap() {
        let len: usize = len.parse().unwrap();
        let p = g.plan(0, len);
        assert_eq!(p.groups.end, u(&st["closed"]), "len {len}");
        assert_eq!(p.groups.end, u(&st["pooled_len"]), "len {len}");
        assert_eq!(
            p.tail_out.collect::<Vec<_>>(),
            usizes(&st["tail"]),
            "len {len}"
        );
    }
}

#[test]
fn select_rules_match_ops_kpool_select() {
    let g = real();
    let o = fixture("ops.json");
    let k = &o["kpool_select"];
    assert_eq!(u(&k["top_pool"]), g.n_top);
    assert_eq!(u(&k["kpool"]), g.kpool);
    for c in k["cases"].as_array().unwrap() {
        let t = u(&c["t"]);
        assert_eq!(g.visible(t), u(&c["n_visible"]), "t {t}");
        assert_eq!(g.picked(t), u(&c["n_picked"]), "t {t}");
        assert_eq!(
            g.tail_cells(t).collect::<Vec<_>>(),
            usizes(&c["tail_cells"]),
            "t {t}"
        );
        assert_eq!(g.cells(t), u(&c["n_cells"]), "t {t}");
        let dense = c["all_earlier_visible"].as_bool().unwrap();
        assert_eq!(g.is_dense(t), dense, "t {t}");
        if dense {
            assert_eq!(
                usizes(&c["cells_sorted"]),
                (0..=t).collect::<Vec<_>>(),
                "t {t}"
            );
            let row = dense_rows(t, 1, g.kmax()).unwrap();
            assert_eq!(row.iter().filter(|&&x| x >= 0).count(), t + 1);
        }
    }
}

#[test]
fn plan_splits_dense_and_scored_rows() {
    let g = real();
    assert_eq!(g.first_sparse(), 2048);
    assert_eq!((g.tail_slots(), g.kmax()), (4, 2048));
    let p = g.plan(0, 8192);
    assert_eq!(
        (p.dense, p.sparse.clone(), p.kmax),
        (0..2048, 2048..8192, 2048)
    );
    assert_eq!((p.groups, p.cols), (0..2048, 2048));
    let p = g.plan(100, 50);
    assert_eq!(
        (p.dense, p.sparse.is_empty(), p.kmax),
        (100..150, true, 150)
    );
    let p = g.plan(5000, 1);
    assert_eq!(
        (p.dense.is_empty(), p.sparse, p.kmax),
        (true, 5000..5001, 2048)
    );

    let mut nt = real();
    nt.select_tail = false;
    nt.n_top = 512;
    assert_eq!(nt.visible(3), 1);
    assert_eq!(nt.tail_cells(3), 4..4);
    assert_eq!(nt.tail_cells(6), 4..7);
    assert_eq!(
        (nt.first_sparse(), nt.tail_slots(), nt.kmax()),
        (2051, 3, 2051)
    );
    for t in [0, 1, 2047, 2050, 2051, 5000] {
        assert!(nt.cells(t) <= nt.kmax(), "t {t}");
        assert_eq!(nt.is_dense(t), nt.cells(t) == t + 1, "t {t}");
    }
}

#[test]
fn dense_rows_pad_to_their_width() {
    assert_eq!(
        dense_rows(2, 3, 5).unwrap(),
        vec![0, 1, 2, -1, -1, 0, 1, 2, 3, -1, 0, 1, 2, 3, 4]
    );
    assert_eq!(dense_rows(0, 1, 1).unwrap(), vec![0]);
    assert!(dense_rows(4, 2, 5).is_err());
}

#[test]
fn score_rows_stay_within_the_budget() {
    assert_eq!(score_row_bytes(16, 65_536), 4 << 20);
    assert_eq!(score_rows(16, 65_536, INDEX_CHUNK_BYTES), 64);
    assert_eq!(score_rows(16, 0, INDEX_CHUNK_BYTES), INDEX_CHUNK_BYTES);
    assert_eq!(score_rows(16, 1 << 30, 1), 1);
}

// Everything below builds MLX arrays.

fn arr(v: &[f32], shape: &[usize]) -> MxArray {
    let s: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
    MxArray::from_float32(v, &s).unwrap()
}

fn dense(v: &[f32], rows: usize, cols: usize) -> Dense {
    Dense::Array(arr(v, &[rows, cols]))
}

fn read(x: &MxArray) -> Vec<f32> {
    x.astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .as_ref()
        .to_vec()
}

fn read_i32(x: &MxArray) -> Vec<i32> {
    x.astype(DType::Int32)
        .unwrap()
        .to_int32()
        .unwrap()
        .as_ref()
        .to_vec()
}

/// Index rows `[n, width]` -> each row's valid cells, ascending.
fn cell_sets(idx: &MxArray) -> Vec<Vec<i32>> {
    let width = idx.shape_at(1).unwrap() as usize;
    read_i32(idx)
        .chunks(width)
        .map(|r| {
            let mut v: Vec<i32> = r.iter().copied().filter(|&c| c >= 0).collect();
            v.sort_unstable();
            v
        })
        .collect()
}

#[test]
fn pool_rope_and_scores_match_ops_fixture() {
    crate::models::hy_v4_flash::exact_f32();
    let o = fixture("ops.json");
    let k = &o["kpool_pool_score"];
    let dm = &k["dims"];
    let (t, d, h, kp, rd) = (
        u(&dm["t"]),
        u(&dm["head_dim"]),
        u(&dm["heads"]),
        u(&dm["kpool"]),
        u(&dm["n_rot"]),
    );
    let g = IndexerGeometry {
        n_heads: h,
        head_dim: d,
        rope_dim: rd,
        kpool: kp,
        n_top: 2,
        select_tail: true,
        eps: k["norm_eps"].as_f64().unwrap() as f32,
        weight_scale: k["weight_scale"].as_f64().unwrap() as f32,
        rope_theta: k["theta"].as_f64().unwrap() as f32,
    };
    let dim = 8;
    let w = IndexerWeights {
        attn_k: dense(&vec![0.0; d * dim], d, dim),
        k_norm: arr(&vec![1.0; d], &[d]),
        q_b: dense(&vec![0.0; h * d * dim], h * d, dim),
        proj: dense(&vec![0.0; h * dim], h, dim),
        compressor_value: arr(&f32s(&k["wv"]), &[d, d]),
        compressor_gate: arr(&f32s(&k["wg"]), &[d, d]),
        compressor_ape: arr(&f32s(&k["ape"]), &[kp, d]),
        compressor_norm: arr(&f32s(&k["norm_w"]), &[d]),
    };
    let ix = Indexer::new(g.clone(), &w).unwrap();
    let tok = f32s(&k["tok_k"]);
    let n_groups = t / kp;
    let grp = arr(&tok[..n_groups * kp * d], &[n_groups, kp, d]);
    let keys = ix.pool(&w, &grp, 0).unwrap();
    let want = f32s(&k["pooled_keys_roped_f32"]);
    for (i, (a, b)) in read(&keys).iter().zip(&want).enumerate() {
        close(*a, *b, 1e-5, 1e-4, &format!("pooled key {i}"));
    }

    let iq = arr(&f32s(&k["iq"]), &[t, h, d]);
    let pos: Vec<u64> = (0..t as u64).collect();
    let q = ix.rope(&iq, &pos).unwrap();
    let want = f32s(&k["q_roped_f32"]);
    for (i, (a, b)) in read(&q).iter().zip(&want).enumerate() {
        close(*a, *b, 1e-5, 1e-5, &format!("roped query {i}"));
    }

    let wr: Vec<f32> = f32s(&k["w_raw"])
        .into_iter()
        .map(|x| x * g.weight_scale)
        .collect();
    let keys_f = arr(&f32s(&k["pooled_keys_roped_f32"]), &[n_groups, d]);
    let qf = arr(&f32s(&k["q_roped_f32"]), &[t, h, d]);
    let sc = read(&scores(&qf, &arr(&wr, &[t, h]), &keys_f).unwrap());
    for (ti, row) in k["score"].as_array().unwrap().iter().enumerate() {
        for (gi, v) in row.as_array().unwrap().iter().enumerate() {
            assert_eq!(v.is_null(), gi >= g.visible(ti), "t {ti} group {gi}");
            if let Some(want) = v.as_f64() {
                close(
                    sc[ti * n_groups + gi],
                    want as f32,
                    1e-4,
                    1e-4,
                    &format!("score t {ti} g {gi}"),
                );
            }
        }
    }
}

#[test]
fn pick_groups_ties_to_the_lowest_group() {
    let g = real();
    let o = fixture("ops.json");
    let k = &o["kpool_select"];
    let gs = f32s(&k["group_scores"]);
    let cols = gs.len();
    let cases = k["cases"].as_array().unwrap();
    let rows: Vec<f32> = cases.iter().flat_map(|_| gs.iter().copied()).collect();
    let visible: Vec<i32> = cases.iter().map(|c| u(&c["n_visible"]) as i32).collect();
    let sc = arr(&rows, &[cases.len(), cols]);
    let ids = pick_groups(&sc, &visible, g.n_top).unwrap();
    assert_eq!(ids.shape_at(1).unwrap() as usize, g.n_top);
    let got = read_i32(&ids);
    let cells = cell_sets(&group_cells(&ids, g.kpool).unwrap());
    for (i, c) in cases.iter().enumerate() {
        let t = u(&c["t"]);
        let row = &got[i * g.n_top..(i + 1) * g.n_top];
        let n = u(&c["n_picked"]);
        let want: Vec<i32> = usizes(&c["picked_sorted"])
            .into_iter()
            .map(|x| x as i32)
            .collect();
        assert_eq!(row[..n].to_vec(), want, "t {t} picks");
        assert!(row[n..].iter().all(|&x| x == -1), "t {t} padding");
        let mut all: Vec<i32> = cells[i].clone();
        all.extend(g.tail_cells(t).map(|x| x as i32));
        all.sort_unstable();
        let want: Vec<i32> = usizes(&c["cells_sorted"])
            .into_iter()
            .map(|x| x as i32)
            .collect();
        assert_eq!(all, want, "t {t} cells");
    }
    assert!(pick_groups(&sc, &visible[1..], g.n_top).is_err());
    let narrow = arr(&gs[..100], &[1, 100]);
    assert!(pick_groups(&narrow, &[100], g.n_top).is_err());
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

/// kpool 4, 2 picked groups: queries from 12 on pick by score.
fn small() -> IndexerGeometry {
    IndexerGeometry {
        n_heads: 2,
        head_dim: 8,
        rope_dim: 4,
        kpool: 4,
        n_top: 2,
        select_tail: true,
        eps: 1e-5,
        weight_scale: 0.25,
        rope_theta: 1e7,
    }
}

/// Host copies of the indexer weights, row-major `[out, in]`.
struct HostIx {
    attn_k: Vec<f32>,
    k_norm: Vec<f32>,
    q_b: Vec<f32>,
    proj: Vec<f32>,
    value: Vec<f32>,
    gate: Vec<f32>,
    ape: Vec<f32>,
    norm: Vec<f32>,
}

fn indexer_weights(g: &IndexerGeometry, seed: u64) -> (IndexerWeights, HostIx) {
    let mut r = Rng(seed);
    let (h, d, kp) = (g.n_heads, g.head_dim, g.kpool);
    let mat =
        |r: &mut Rng, rows: usize, cols: usize| r.vec(rows * cols, 1.5 / (cols as f32).sqrt());
    let norm_w = |r: &mut Rng, n: usize| -> Vec<f32> {
        r.vec(n, 0.5).into_iter().map(|x| 1.0 + x).collect()
    };
    let hw = HostIx {
        attn_k: mat(&mut r, d, D),
        k_norm: norm_w(&mut r, d),
        q_b: mat(&mut r, h * d, Q_LORA),
        proj: mat(&mut r, h, D),
        value: mat(&mut r, d, d),
        gate: mat(&mut r, d, d),
        ape: r.vec(kp * d, 1.0),
        norm: norm_w(&mut r, d),
    };
    let w = IndexerWeights {
        attn_k: dense(&hw.attn_k, d, D),
        k_norm: arr(&hw.k_norm, &[d]),
        q_b: dense(&hw.q_b, h * d, Q_LORA),
        proj: dense(&hw.proj, h, D),
        compressor_value: arr(&hw.value, &[d, d]),
        compressor_gate: arr(&hw.gate, &[d, d]),
        compressor_ape: arr(&hw.ape, &[kp, d]),
        compressor_norm: arr(&hw.norm, &[d]),
    };
    (w, hw)
}

fn mv(w: &[f32], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
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

/// RoPE on the last `2·inv.len()` values of `v`, consecutive pairs.
fn rope(v: &mut [f64], pos: usize, inv: &[f32]) {
    let off = v.len() - 2 * inv.len();
    for (i, &f) in inv.iter().enumerate() {
        let ang = f64::from(rope_angle(pos as u64, f));
        let (c, s) = (f64::from(ang.cos() as f32), f64::from(ang.sin() as f32));
        let (a, b) = (v[off + 2 * i], v[off + 2 * i + 1]);
        v[off + 2 * i] = a * c - b * s;
        v[off + 2 * i + 1] = a * s + b * c;
    }
}

fn f16r(v: f64) -> f64 {
    half::f16::from_f64(v).to_f64()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Each query's cells, f64 math with the pooled keys rounded to f16; picks
/// by score descending, ties to the lowest group.
fn host_cells(
    g: &IndexerGeometry,
    hw: &HostIx,
    xs: &[Vec<f32>],
    qrs: &[Vec<f32>],
) -> Vec<Vec<i32>> {
    let (h, d, kp) = (g.n_heads, g.head_dim, g.kpool);
    let inv = inv_freqs(g.rope_dim, g.rope_theta, None);
    let f64s = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| f64::from(x)).collect() };
    let tok: Vec<Vec<f64>> = xs
        .iter()
        .map(|x| rms(&mv(&hw.attn_k, d, D, &f64s(x)), &hw.k_norm, g.eps))
        .collect();
    let keys: Vec<Vec<f64>> = (0..xs.len() / kp)
        .map(|gi| {
            let members = &tok[gi * kp..(gi + 1) * kp];
            let v: Vec<Vec<f64>> = members.iter().map(|m| mv(&hw.value, d, d, m)).collect();
            let gt: Vec<Vec<f64>> = members
                .iter()
                .enumerate()
                .map(|(s, m)| {
                    mv(&hw.gate, d, d, m)
                        .into_iter()
                        .zip(&hw.ape[s * d..(s + 1) * d])
                        .map(|(a, &b)| a + f64::from(b))
                        .collect()
                })
                .collect();
            let pooled: Vec<f64> = (0..d)
                .map(|c| {
                    let m = (0..kp).map(|s| gt[s][c]).fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = (0..kp).map(|s| (gt[s][c] - m).exp()).collect();
                    let z: f64 = e.iter().sum();
                    (0..kp).map(|s| v[s][c] * e[s] / z).sum()
                })
                .collect();
            let mut k = rms(&pooled, &hw.norm, g.eps);
            rope(&mut k, gi * kp, &inv);
            k.into_iter().map(f16r).collect()
        })
        .collect();
    (0..xs.len())
        .map(|t| {
            let nv = g.visible(t);
            let mut groups: Vec<usize> = (0..nv).collect();
            if nv > g.n_top {
                let qr = f64s(&qrs[t]);
                let x = f64s(&xs[t]);
                let wt = mv(&hw.proj, h, D, &x);
                let score: Vec<f64> = (0..nv)
                    .map(|gi| {
                        (0..h)
                            .map(|hh| {
                                let mut q = mv(
                                    &hw.q_b[hh * d * Q_LORA..(hh + 1) * d * Q_LORA],
                                    d,
                                    Q_LORA,
                                    &qr,
                                );
                                rope(&mut q, t, &inv);
                                wt[hh] * f64::from(g.weight_scale) * dot(&q, &keys[gi]).max(0.0)
                            })
                            .sum()
                    })
                    .collect();
                groups.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
                groups.truncate(g.n_top);
                groups.sort_unstable();
            }
            let mut cells: Vec<i32> = groups
                .iter()
                .flat_map(|&gi| (gi * kp..(gi + 1) * kp).map(|c| c as i32))
                .collect();
            cells.extend(g.tail_cells(t).map(|c| c as i32));
            cells
        })
        .collect()
}

fn rows_of(v: &[Vec<f32>]) -> MxArray {
    let flat: Vec<f32> = v.iter().flatten().copied().collect();
    arr(&flat, &[v.len(), v[0].len()])
}

/// Runs the indexer over `xs`/`qrs` in calls of `sizes`; returns each row's
/// cells.
fn select_split(
    ix: &Indexer,
    w: &IndexerWeights,
    st: &mut IndexerState,
    xs: &[Vec<f32>],
    qrs: &[Vec<f32>],
    start: usize,
    sizes: &[usize],
) -> Vec<Vec<i32>> {
    let mut out = Vec::new();
    let mut at = 0;
    for &n in sizes {
        let (x, qr) = (rows_of(&xs[at..at + n]), rows_of(&qrs[at..at + n]));
        let idx = ix.select(w, prefill(&x, &qr, start + at), st).unwrap();
        assert_eq!(idx.shape_at(0).unwrap() as usize, n);
        assert_eq!(
            idx.shape_at(1).unwrap() as usize,
            ix.geometry().plan(start + at, n).kmax
        );
        out.extend(cell_sets(&idx));
        at += n;
    }
    assert_eq!(at, xs.len());
    out
}

fn inputs(n: usize, width: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut r = Rng(seed);
    (0..n).map(|_| r.vec(width, 2.0)).collect()
}

fn pooled_host(st: &IndexerState, d: usize) -> Vec<f32> {
    read(st.pooled_rows().unwrap())[..st.pooled_len() * d].to_vec()
}

#[test]
fn split_calls_pick_what_one_call_picks() {
    let g = small();
    let (w, hw) = indexer_weights(&g, 41);
    let ix = Indexer::new(g.clone(), &w).unwrap();
    let n = 23;
    let xs = inputs(n, D, 3);
    let qrs = inputs(n, Q_LORA, 4);
    let want = host_cells(&g, &hw, &xs, &qrs);
    let mut one = IndexerState::new(32, &g);
    let got = select_split(&ix, &w, &mut one, &xs, &qrs, 0, &[n]);
    assert_eq!(got, want, "one call vs host");
    let pooled = pooled_host(&one, g.head_dim);
    assert_eq!(one.pooled_len(), n / g.kpool);
    assert_eq!(one.open(), 20..23);
    assert_eq!(one.held(), g.kpool - 1 + WINDOW_SLACK);
    for sizes in [
        vec![1, 3, 4, 5, 10],
        vec![12, 11],
        vec![11, 1, 11],
        vec![13, 10],
        vec![1; n],
    ] {
        let mut st = IndexerState::new(32, &g);
        let got = select_split(&ix, &w, &mut st, &xs, &qrs, 0, &sizes);
        assert_eq!(got, want, "split {sizes:?}");
        assert_eq!((st.len(), st.pooled_len(), st.open()), (n, 5, 20..23));
        for (a, b) in pooled_host(&st, g.head_dim).iter().zip(&pooled) {
            close(*a, *b, 1e-3, 1e-3, &format!("pooled split {sizes:?}"));
        }
    }
}

#[test]
fn truncate_and_restore_keep_the_picks() {
    let g = small();
    let (w, _) = indexer_weights(&g, 43);
    let ix = Indexer::new(g.clone(), &w).unwrap();
    let n = 23;
    let xs = inputs(n, D, 9);
    let qrs = inputs(n, Q_LORA, 10);
    let mut st = IndexerState::new(32, &g);
    let want = select_split(&ix, &w, &mut st, &xs, &qrs, 0, &[n]);
    let full = st.mark();
    assert_eq!(full.len(), n);

    assert!(st.truncate(15).is_err());
    assert_eq!(st.len(), n);
    assert!(st.truncate(24).is_err());
    st.truncate(21).unwrap();
    assert_eq!((st.len(), st.pooled_len(), st.open()), (21, 5, 20..21));
    let got = select_split(&ix, &w, &mut st, &xs[21..], &qrs[21..], 21, &[2]);
    assert_eq!(got, want[21..].to_vec(), "replay after 21");

    st.truncate(16).unwrap();
    assert_eq!((st.len(), st.pooled_len(), st.held()), (16, 4, 0));
    assert!(st.restore(full.clone()).is_err());
    let got = select_split(&ix, &w, &mut st, &xs[16..], &qrs[16..], 16, &[3, 4]);
    assert_eq!(got, want[16..].to_vec(), "replay after 16");

    let before = (st.len(), st.pooled_len(), st.open(), st.held());
    let x = rows_of(&xs[..1]);
    let qr = rows_of(&qrs[..1]);
    assert!(ix.select(&w, prefill(&x, &qr, 5), &mut st).is_err());
    assert_eq!((st.len(), st.pooled_len(), st.open(), st.held()), before);

    let mut short = IndexerState::new(9, &g);
    select_split(&ix, &w, &mut short, &xs[..6], &qrs[..6], 0, &[6]);
    let before = (short.len(), short.pooled_len(), short.open(), short.held());
    let (x, qr) = (rows_of(&xs[6..12]), rows_of(&qrs[6..12]));
    assert!(ix.select(&w, prefill(&x, &qr, 6), &mut short).is_err());
    assert_eq!(
        (short.len(), short.pooled_len(), short.open(), short.held()),
        before
    );

    st.truncate(0).unwrap();
    assert!(st.is_empty());
    let got = select_split(&ix, &w, &mut st, &xs, &qrs, 0, &[n]);
    assert_eq!(got, want, "from empty");
}

/// A flat `update` that fails after it pooled a group (the tail step)
/// leaves the state as it was, and the same call then succeeds.
#[test]
fn a_failed_update_keeps_the_indexer_state() {
    let g = small();
    let (w, _) = indexer_weights(&g, 47);
    let ix = Indexer::new(g.clone(), &w).unwrap();
    let xs = inputs(12, D, 21);
    let mut st = IndexerState::new(32, &g);
    ix.update(&w, &rows_of(&xs[..8]), 0, &mut st).unwrap();
    assert_eq!(
        (st.len(), st.pooled_len(), st.held()),
        (8, 2, g.kpool - 1 + WINDOW_SLACK)
    );
    let before = (st.len(), st.pooled_len(), st.held());
    {
        let _armed = crate::array::fail_native_call_at(
            "hy_v4_flash indexer tail",
            "test: indexer tail eval failed",
        );
        assert!(ix.update(&w, &rows_of(&xs[8..12]), 8, &mut st).is_err());
    }
    assert_eq!((st.len(), st.pooled_len(), st.held()), before);
    ix.update(&w, &rows_of(&xs[8..12]), 8, &mut st).unwrap();
    assert_eq!((st.len(), st.pooled_len()), (12, 3));
}

#[test]
fn indexer_refuses_mismatched_weights() {
    let g = small();
    let (mut w, hw) = indexer_weights(&g, 47);
    assert!(Indexer::new(g.clone(), &w).is_ok());
    w.compressor_ape = arr(&hw.ape, &[g.head_dim, g.kpool]);
    assert!(Indexer::new(g.clone(), &w).is_err());
    let (mut w, hw) = indexer_weights(&g, 47);
    w.q_b = dense(&hw.q_b, g.head_dim, g.n_heads * Q_LORA);
    assert!(Indexer::new(g.clone(), &w).is_err());
    let (w, _) = indexer_weights(&g, 47);
    let mut bad_g = g;
    bad_g.rope_dim = bad_g.head_dim;
    assert!(Indexer::new(bad_g, &w).is_err());
}

fn attn_geometry() -> Geometry {
    Geometry {
        attn: Attn::Indexed,
        n_heads: 2,
        kv_lora: 6,
        nope: 4,
        rope_dim: 4,
        head_v: 5,
        eps: 1e-5,
        scale: 1.0 / 8f32.sqrt(),
        rope_theta: 1e7,
        window: 4,
    }
}

fn attn_weights(g: &Geometry, seed: u64) -> AttnWeights {
    let mut r = Rng(seed);
    let (h, l, v) = (g.n_heads, g.kv_lora, g.head_v);
    let mut mat = |rows: usize, cols: usize| {
        dense(&r.vec(rows * cols, 1.5 / (cols as f32).sqrt()), rows, cols)
    };
    let q_a = mat(Q_LORA, D);
    let q_b = mat(h * g.head_qk(), Q_LORA);
    let kv_a = mat(g.row(), D);
    let k_b = mat(h * l, g.nope);
    let v_b = mat(h * v, l);
    let output = mat(D, h * v);
    let gate = mat(h * v, D);
    let ones = |n: usize| arr(&vec![1.0; n], &[n]);
    AttnWeights {
        norm: ones(D),
        sinks: arr(&[0.3, -0.2], &[h]),
        q_a,
        q_a_norm: ones(Q_LORA),
        q_b,
        kv_a,
        kv_a_norm: ones(l),
        k_b,
        v_b,
        output,
        gate,
        post_norm: ones(D),
        q_kv: None,
        q_kv_gate: None,
    }
}

#[test]
fn indexed_layer_past_top_k_gives_the_same_rows_for_every_split() {
    crate::models::hy_v4_flash::exact_f32();
    let ag = attn_geometry();
    let aw = attn_weights(&ag, 61);
    let mla = Mla::new(ag.clone(), &aw).unwrap();
    let g = small();
    let (w, _) = indexer_weights(&g, 62);
    let ix = Indexer::new(g.clone(), &w).unwrap();
    let n = 24;
    let hs = inputs(n, D, 63);
    let run = |sizes: &[usize]| -> Vec<f32> {
        let mut cache = LayerCache::new(Attn::Indexed, ag.window, 32, ag.row() as i64);
        let mut st = IndexerState::new(32, &g);
        let mut out = Vec::new();
        let mut at = 0;
        for &k in sizes {
            let y = mla
                .forward(
                    &aw,
                    &rows_of(&hs[at..at + k]),
                    at,
                    &mut cache,
                    Some(Indexing {
                        indexer: &ix,
                        weights: &w,
                        state: &mut st,
                        kind: RowKind::Prefill,
                    }),
                )
                .unwrap();
            out.extend(read(&y));
            at += k;
        }
        assert_eq!((cache.len(), st.len()), (n, n));
        out
    };
    let want = run(&[n]);
    assert_eq!(want.len(), n * D);
    assert!(want.iter().all(|x| x.is_finite()));
    for sizes in [vec![12, 12], vec![1, 23], vec![1; n], vec![5, 7, 1, 11]] {
        let got = run(&sizes);
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            close(*a, *b, 1e-4, 1e-4, &format!("split {sizes:?} value {i}"));
        }
    }
}

/// `attn_geometry` with a 64-column latent row: planes of 32, a pool head
/// size.
fn wide_geometry() -> Geometry {
    Geometry {
        kv_lora: 60,
        ..attn_geometry()
    }
}

#[cfg(target_os = "macos")]
mod paged {
    use super::*;
    use crate::models::hy_v4_flash::attention::PagedPool;
    use crate::models::hy_v4_flash::paged::{PageGeometry, SlotMap, create};
    use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

    const OWNER: u32 = 1;
    const OTHER: u32 = 9;
    const N: usize = 40;

    struct Case {
        ag: Geometry,
        aw: AttnWeights,
        mla: Mla,
        g: IndexerGeometry,
        w: IndexerWeights,
        ix: Indexer,
        hs: Vec<Vec<f32>>,
    }

    fn case() -> Case {
        let ag = wide_geometry();
        let aw = attn_weights(&ag, 71);
        let mla = Mla::new(ag.clone(), &aw).unwrap();
        let g = small();
        let (w, _) = indexer_weights(&g, 72);
        let ix = Indexer::new(g.clone(), &w).unwrap();
        let hs = inputs(N, D, 73);
        Case {
            ag,
            aw,
            mla,
            g,
            w,
            ix,
            hs,
        }
    }

    /// A one-layer pool of this case's page, `None` without Metal.
    fn pool(c: &Case) -> Option<PagedKVCacheAdapter> {
        let pg = PageGeometry {
            layers: 1,
            half: (c.ag.row() / 2) as u32,
            index_rows: (32 / c.g.kpool) as u32,
            index_dim: c.g.head_dim as u32,
        };
        match create(&pg, 1024, 8, 8, u64::MAX) {
            Ok(a) => Some(a),
            Err(e) => {
                eprintln!("skipping Hy4 paged indexer test: {e}");
                None
            }
        }
    }

    fn bits(x: &MxArray) -> Vec<u32> {
        x.to_float32()
            .unwrap()
            .as_ref()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    fn flat(c: &Case, sizes: &[usize]) -> (Vec<u32>, Vec<u16>) {
        let mut cache = LayerCache::new(Attn::Indexed, c.ag.window, 64, c.ag.row() as i64);
        let mut st = IndexerState::new(64, &c.g);
        let mut out = Vec::new();
        let mut at = 0;
        for &k in sizes {
            let y = c
                .mla
                .forward(
                    &c.aw,
                    &rows_of(&c.hs[at..at + k]),
                    at,
                    &mut cache,
                    Some(Indexing {
                        indexer: &c.ix,
                        weights: &c.w,
                        state: &mut st,
                        kind: RowKind::Prefill,
                    }),
                )
                .unwrap();
            out.extend(bits(&y));
            at += k;
        }
        let pooled = st.pooled_rows().unwrap().to_uint16_native().unwrap()
            [..st.pooled_len() * c.g.head_dim]
            .to_vec();
        (out, pooled)
    }

    fn paged_call(
        c: &Case,
        adapter: &mut PagedKVCacheAdapter,
        st: &mut PagedIndexerState,
        slots: &SlotMap,
        at: usize,
        k: usize,
    ) -> Result<MxArray> {
        c.mla.forward_indexed_paged(
            &c.aw,
            &rows_of(&c.hs[at..at + k]),
            at,
            PagedPool {
                adapter,
                floor: at as u32,
                slots,
            },
            PagedIndexing {
                indexer: &c.ix,
                weights: &c.w,
                state: st,
                kind: RowKind::Prefill,
            },
        )
    }

    /// `OWNER`'s rows in calls of `sizes`, with a page of `OTHER` taken
    /// before its first and its second call.
    fn paged(c: &Case, adapter: &mut PagedKVCacheAdapter, sizes: &[usize]) -> (Vec<u32>, Vec<u16>) {
        let mut st = PagedIndexerState::new(0, &c.g);
        let mut out = Vec::new();
        let mut at = 0;
        for (i, &k) in sizes.iter().enumerate() {
            if i < 2 {
                adapter.record_tokens_for(OTHER, &[3; 32]).unwrap();
            }
            adapter.record_tokens_for(OWNER, &vec![3; k]).unwrap();
            let slots = SlotMap::of(adapter).unwrap();
            let y = paged_call(c, adapter, &mut st, &slots, at, k).unwrap();
            out.extend(bits(&y));
            at += k;
        }
        assert_eq!((st.len(), st.pooled_len()), (N, N / c.g.kpool));
        let pooled = adapter
            .index_rows(0, st.pooled_len() as u32)
            .unwrap()
            .to_uint16_native()
            .unwrap();
        (out, pooled)
    }

    /// One indexed layer over the paged pool (half rows in two planes,
    /// pooled keys in the f16 index plane, scattered pages, rows past
    /// `top_k` scored) gives the flat layer's outputs and pooled keys bit for
    /// bit, for every split.
    /// Mutation: the slot map ignores the block table (slot = row) → red.
    #[test]
    fn paged_indexed_layer_equals_the_flat_layer_bit_for_bit() {
        let c = case();
        for sizes in [vec![N], vec![12, 12, 16], vec![1; N], vec![5, 7, 1, 27]] {
            let Some(mut adapter) = pool(&c) else {
                return;
            };
            let (want, want_pooled) = flat(&c, &sizes);
            let (got, got_pooled) = paged(&c, &mut adapter, &sizes);
            assert_eq!(adapter.active_seq_id(), Some(OWNER));
            assert_ne!(adapter.block_table().unwrap().block_ids(), vec![0, 1]);
            assert_eq!(got.len(), N * D, "split {sizes:?}");
            assert!(got == want, "split {sizes:?} outputs differ");
            assert_eq!(got_pooled, want_pooled, "split {sizes:?} pooled keys");
        }
    }

    /// A call whose slot map was built on older pages fails after its
    /// indexer update and leaves the indexer state as it was; the same call
    /// with a fresh map then succeeds.
    /// Mutation: no restore on error → the retry is refused → red.
    #[test]
    fn a_failed_paged_call_keeps_the_indexer_state() {
        let c = case();
        let Some(mut adapter) = pool(&c) else {
            return;
        };
        let mut st = PagedIndexerState::new(0, &c.g);
        adapter.record_tokens_for(OWNER, &[3; 30]).unwrap();
        let slots = SlotMap::of(&adapter).unwrap();
        paged_call(&c, &mut adapter, &mut st, &slots, 0, 30).unwrap();
        adapter.record_tokens_for(OWNER, &[3; 6]).unwrap();
        let before = (st.len(), st.pooled_len(), st.open(), st.tail().held());
        assert!(paged_call(&c, &mut adapter, &mut st, &slots, 30, 6).is_err());
        assert_eq!(
            (st.len(), st.pooled_len(), st.open(), st.tail().held()),
            before
        );
        let fresh = SlotMap::of(&adapter).unwrap();
        paged_call(&c, &mut adapter, &mut st, &fresh, 30, 6).unwrap();
        assert_eq!((st.len(), st.pooled_len()), (36, 9));
    }
}

/// Decode rows take the head weights one row per call: each row of a
/// 3-row decode call is that row's 1-row call bit for bit; a prefill call
/// is one matmul over its rows.
#[test]
fn decode_head_weights_run_per_row() {
    let g = small();
    let (w, _) = indexer_weights(&g, 5);
    let ix = Indexer::new(g.clone(), &w).unwrap();
    let xs = inputs(3, D, 17);
    let all = read(&ix.head_weights(&w, &rows_of(&xs), RowKind::Decode).unwrap());
    let h = g.n_heads;
    for (i, x) in xs.iter().enumerate() {
        let one = read(
            &ix.head_weights(&w, &rows_of(std::slice::from_ref(x)), RowKind::Prefill)
                .unwrap(),
        );
        assert_eq!(all[i * h..(i + 1) * h], one[..], "row {i}");
    }
    let pre = read(
        &ix.head_weights(&w, &rows_of(&xs), RowKind::Prefill)
            .unwrap(),
    );
    assert_eq!(pre.len(), 3 * h);
}

#[test]
fn decode_head_weights_q8_runs_batched_bit_exact() {
    use crate::models::hy_v4_flash::weights::quantize_sym_q8_array;

    const Q8_D: usize = 32;
    let g = small();
    let (mut w, _) = indexer_weights(&g, 23);
    let mut r = Rng(23);
    let attn_k = arr(&r.vec(g.head_dim * Q8_D, 0.25), &[g.head_dim, Q8_D]);
    let proj_arr = arr(&r.vec(g.n_heads * Q8_D, 0.25), &[g.n_heads, Q8_D]);
    w.attn_k = quantize_sym_q8_array(&attn_k).unwrap();
    w.proj = quantize_sym_q8_array(&proj_arr).unwrap();
    let ix = Indexer::new(g.clone(), &w).unwrap();
    let xs = inputs(4, Q8_D, 29);
    let all = read(&ix.head_weights(&w, &rows_of(&xs), RowKind::Decode).unwrap());
    let h = g.n_heads;
    for (i, x) in xs.iter().enumerate() {
        let one = read(
            &ix.head_weights(&w, &rows_of(std::slice::from_ref(x)), RowKind::Decode)
                .unwrap(),
        );
        assert_eq!(all[i * h..(i + 1) * h], one[..], "q8 row {i}");
    }
}

#[cfg(target_os = "macos")]
mod fused_metal {
    use super::*;
    use crate::models::hy_v4_flash::ops::{try_indexer_score, try_indexer_topk_cells};
    use crate::models::hy_v4_flash::paged::{BLOCK, page_ids, slot_ids};

    /// Builds a scattered physical page table of `n_pages` distinct physical
    /// blocks inside `n_blocks` total blocks (`n_blocks >= n_pages`).
    fn scattered_pages(n_pages: usize, n_blocks: usize) -> Vec<u32> {
        let mut pages = Vec::with_capacity(n_pages);
        for i in 0..n_pages {
            pages.push(((i * 7 + 3) % n_blocks) as u32);
        }
        // Guarantee uniqueness if stride shares a factor with n_blocks.
        let mut seen = vec![false; n_blocks];
        for p in &mut pages {
            while seen[*p as usize] {
                *p = ((*p as usize + 1) % n_blocks) as u32;
            }
            seen[*p as usize] = true;
        }
        pages
    }

    /// Packs logical `[n_groups, 128]` f16 keys into a paged
    /// `[n_blocks, index_rows, 128]` f16 pool at `(pages[(start_g + g)/index_rows], (start_g + g)%index_rows)`.
    fn pack_paged_keys(
        logical_f16: &[u16],
        n_groups: usize,
        start_g: usize,
        pages: &[u32],
        n_blocks: usize,
        index_rows: usize,
        d: usize,
    ) -> MxArray {
        let mut buf = vec![0u16; n_blocks * index_rows * d];
        for g in 0..n_groups {
            let g_abs = start_g + g;
            let phys = pages[g_abs / index_rows] as usize;
            let local = g_abs % index_rows;
            let dst = (phys * index_rows + local) * d;
            let src = g * d;
            buf[dst..dst + d].copy_from_slice(&logical_f16[src..src + d]);
        }
        MxArray::from_float16(&buf, &[n_blocks as i64, index_rows as i64, d as i64]).unwrap()
    }

    #[test]
    fn indexer_score_decode_and_prefill_flat_and_paged_match_reference() {
        crate::models::hy_v4_flash::exact_f32();
        let h = 16usize;
        let d = 128usize;
        let index_rows = 8usize;

        for (s, n_groups, start_g, seed) in [
            (1usize, 37usize, 0usize, 101u64),
            (1usize, 256usize, 11usize, 102u64),
            (4usize, 193usize, 5usize, 103u64),
            (19usize, 73usize, 0usize, 104u64),
            (64usize, 260usize, 8usize, 105u64),
        ] {
            let mut rng = Rng(seed);
            let q_vec = rng.vec(s * h * d, 1.5);
            let w_vec = rng.vec(s * h, 0.2);
            let k_f32_vec = rng.vec(n_groups * d, 1.5);

            let q = arr(&q_vec, &[s, h, d]);
            let w = arr(&w_vec, &[s, h]);
            let keys_f16 = arr(&k_f32_vec, &[n_groups, d])
                .astype(DType::Float16)
                .unwrap();
            let keys_ref_f32 = keys_f16.astype(DType::Float32).unwrap();
            let want = read(&scores(&q, &w, &keys_ref_f32).unwrap());

            // Flat mode (with leading padding of `start_g` rows + trailing slack).
            let cap_g = start_g + n_groups + 13;
            let keys_bits = keys_f16.to_uint16_native().unwrap().to_vec();
            let mut flat_bits = vec![0u16; cap_g * d];
            flat_bits[start_g * d..(start_g + n_groups) * d].copy_from_slice(&keys_bits);
            let flat_pool = MxArray::from_float16(&flat_bits, &[cap_g as i64, d as i64]).unwrap();

            let Some(got_flat_arr) =
                try_indexer_score(&q, &w, &flat_pool, None, n_groups, start_g).unwrap()
            else {
                eprintln!("skipping hy4_indexer_score Metal test (no GPU)");
                return;
            };
            let got_flat = read(&got_flat_arr);
            assert_eq!(got_flat.len(), want.len());
            let mut max_diff_flat = 0.0f32;
            for (g, wv) in got_flat.iter().zip(&want) {
                max_diff_flat = max_diff_flat.max((g - wv).abs());
            }
            assert!(
                max_diff_flat <= 1e-3,
                "flat S={s} N_g={n_groups} start_g={start_g}: max_abs_diff={max_diff_flat}"
            );

            // Paged mode with scattered non-monotonic physical blocks.
            let n_pages = (start_g + n_groups).div_ceil(index_rows);
            let n_blocks = n_pages + 9;
            let pages_u32 = scattered_pages(n_pages, n_blocks);
            let pages_arr = page_ids(&pages_u32).unwrap();
            let paged_pool = pack_paged_keys(
                &keys_bits, n_groups, start_g, &pages_u32, n_blocks, index_rows, d,
            );

            let got_paged_arr =
                try_indexer_score(&q, &w, &paged_pool, Some(&pages_arr), n_groups, start_g)
                    .unwrap()
                    .expect("paged indexer_score");
            let got_paged = read(&got_paged_arr);
            assert_eq!(
                got_paged.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                got_flat.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "paged vs flat bit mismatch for S={s} N_g={n_groups} start_g={start_g}"
            );

            // Decode multi-row invariance: S=4 matches 4 independent S=1 calls bit-for-bit.
            if s == 4 {
                for row in 0..s {
                    let qr = q
                        .slice(&[row as i64, 0, 0], &[(row + 1) as i64, h as i64, d as i64])
                        .unwrap();
                    let wr = w
                        .slice(&[row as i64, 0], &[(row + 1) as i64, h as i64])
                        .unwrap();
                    let one = read(
                        &try_indexer_score(
                            &qr,
                            &wr,
                            &paged_pool,
                            Some(&pages_arr),
                            n_groups,
                            start_g,
                        )
                        .unwrap()
                        .unwrap(),
                    );
                    let row_got = &got_paged[row * n_groups..(row + 1) * n_groups];
                    assert_eq!(
                        row_got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        one.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "decode row {row} invariance"
                    );
                }
            }
        }
    }

    /// Reference implementation matching `pick_groups + group_cells + tail + slot_ids`
    /// for sparse rows (`vis > n_top`), and packed `0..vis*kpool + tail + -1` + `slot_ids`
    /// for dense rows (`vis <= n_top`).
    fn ref_topk_cells(
        sc: &MxArray,
        visible: &[i32],
        tail_starts: &[i32],
        tail_lens: &[i32],
        pages: Option<&MxArray>,
        n_top: usize,
        kpool: usize,
        kmax: usize,
    ) -> Vec<i32> {
        let s = visible.len();
        let n_g = sc.shape_at(1).unwrap() as usize;
        let can_pick = n_g >= n_top;
        let picked_all = if can_pick {
            Some(read_i32(&pick_groups(sc, visible, n_top).unwrap()))
        } else {
            None
        };
        let mut logical = vec![-1i32; s * kmax];
        for r in 0..s {
            let vis = visible[r].max(0) as usize;
            let row = &mut logical[r * kmax..(r + 1) * kmax];
            let ts = tail_starts[r];
            let tl = tail_lens[r].max(0) as usize;
            if vis <= n_top {
                let n_vis = vis * kpool;
                for (i, slot) in row.iter_mut().enumerate().take(n_vis.min(kmax)) {
                    *slot = i as i32;
                }
                for off in 0..tl {
                    if n_vis + off < kmax {
                        row[n_vis + off] = ts + off as i32;
                    }
                }
            } else {
                let ids = &picked_all.as_ref().unwrap()[r * n_top..(r + 1) * n_top];
                for (gi, &g_id) in ids.iter().enumerate() {
                    if g_id >= 0 {
                        for sub in 0..kpool {
                            row[gi * kpool + sub] = g_id * kpool as i32 + sub as i32;
                        }
                    }
                }
                let base = n_top * kpool;
                for off in 0..tl {
                    if base + off < kmax {
                        row[base + off] = ts + off as i32;
                    }
                }
            }
        }
        let logical_arr = MxArray::from_int32(&logical, &[s as i64, kmax as i64]).unwrap();
        match pages {
            Some(p) => read_i32(&slot_ids(p, &logical_arr).unwrap()),
            None => logical,
        }
    }

    fn check_topk_case(
        label: &str,
        score_rows: &[Vec<f32>],
        visible: &[i32],
        tail_starts: &[i32],
        tail_lens: &[i32],
        n_top: usize,
        kpool: usize,
        kmax: usize,
    ) {
        let s = score_rows.len();
        let n_g = score_rows[0].len();
        let flat_scores: Vec<f32> = score_rows.iter().flatten().copied().collect();
        let sc = arr(&flat_scores, &[s, n_g]);
        let vis_arr = MxArray::from_int32(visible, &[s as i64]).unwrap();
        let ts_arr = MxArray::from_int32(tail_starts, &[s as i64]).unwrap();
        let tl_arr = MxArray::from_int32(tail_lens, &[s as i64]).unwrap();

        let max_cell = visible
            .iter()
            .zip(tail_starts)
            .zip(tail_lens)
            .map(|((&v, &ts), &tl)| ((v.max(0) as usize) * kpool).max((ts + tl).max(0) as usize))
            .max()
            .unwrap_or(1)
            .max(1);
        let n_pages = max_cell.div_ceil(BLOCK as usize);
        let n_blocks = n_pages + 11;
        let pages_u32 = scattered_pages(n_pages, n_blocks);
        let pages_arr = page_ids(&pages_u32).unwrap();

        // Flat (PAGED=false)
        let Some(got_flat_arr) = try_indexer_topk_cells(
            &sc,
            &vis_arr,
            &ts_arr,
            &tl_arr,
            None,
            n_top,
            kpool,
            kmax,
            BLOCK as usize,
        )
        .unwrap() else {
            eprintln!("skipping hy4_indexer_topk_cells Metal test (no GPU)");
            return;
        };
        let got_flat = read_i32(&got_flat_arr);
        let want_flat = ref_topk_cells(
            &sc,
            visible,
            tail_starts,
            tail_lens,
            None,
            n_top,
            kpool,
            kmax,
        );
        assert_eq!(got_flat, want_flat, "{label} (flat)");

        // Paged (PAGED=true)
        let got_paged_arr = try_indexer_topk_cells(
            &sc,
            &vis_arr,
            &ts_arr,
            &tl_arr,
            Some(&pages_arr),
            n_top,
            kpool,
            kmax,
            BLOCK as usize,
        )
        .unwrap()
        .expect("paged topk_cells");
        let got_paged = read_i32(&got_paged_arr);
        let want_paged = ref_topk_cells(
            &sc,
            visible,
            tail_starts,
            tail_lens,
            Some(&pages_arr),
            n_top,
            kpool,
            kmax,
        );
        assert_eq!(got_paged, want_paged, "{label} (paged)");
    }

    #[test]
    fn indexer_topk_cells_dense_fast_path_and_distinct_large_contexts() {
        let n_top = 511usize;
        let kpool = 4usize;
        let kmax = 2048usize;

        // 1. Dense fast path: vis <= n_top (0, 1, 17, 510, 511)
        let mut rng = Rng(201);
        let dense_vis = vec![0i32, 1, 17, 510, 511];
        let dense_ts: Vec<i32> = dense_vis.iter().map(|&v| v * kpool as i32).collect();
        let dense_tl = vec![1i32, 4, 2, 3, 4];
        let dense_scores: Vec<Vec<f32>> = (0..dense_vis.len()).map(|_| rng.vec(520, 5.0)).collect();
        check_topk_case(
            "dense vis <= n_top",
            &dense_scores,
            &dense_vis,
            &dense_ts,
            &dense_tl,
            n_top,
            kpool,
            kmax,
        );

        // 2. Large contexts: vis = 1024, 4096, 8192 with distinct scores
        for (vis, s, seed) in [(1024usize, 3usize, 202u64), (4096, 10, 203), (8192, 4, 204)] {
            let mut r = Rng(seed);
            let cols = vis + 17;
            let mut rows_sc = Vec::with_capacity(s);
            let mut vis_v = Vec::with_capacity(s);
            let mut ts_v = Vec::with_capacity(s);
            let mut tl_v = Vec::with_capacity(s);
            for row in 0..s {
                let v = vis - (row % 3);
                vis_v.push(v as i32);
                ts_v.push((v * kpool) as i32);
                tl_v.push(((row % 4) + 1) as i32);
                // Distinct scores with wide dynamic range (positive + negative)
                rows_sc.push(
                    (0..cols)
                        .map(|g| r.next() * 50.0 + (g as f32) * 1e-5)
                        .collect(),
                );
            }
            check_topk_case(
                &format!("distinct vis={vis} s={s}"),
                &rows_sc,
                &vis_v,
                &ts_v,
                &tl_v,
                n_top,
                kpool,
                kmax,
            );
        }
    }

    #[test]
    fn indexer_topk_cells_heavy_ties_negatives_zeros_and_signed_zero_bit_exact() {
        let n_top = 511usize;
        let kpool = 4usize;
        let kmax = 2048usize;
        let vis = 2048usize;
        let cols = vis + 16;

        // Row 0: all equal positive scores
        let r0 = vec![1.25f32; cols];

        // Row 1: all equal negative scores
        let r1 = vec![-3.5f32; cols];

        // Row 2: all zeros, alternating -0.0f and +0.0f (must tie and pick 0..511!)
        let r2: Vec<f32> = (0..cols)
            .map(|g| if g % 2 == 0 { -0.0f32 } else { 0.0f32 })
            .collect();

        // Row 3: 50% tied at threshold 0.0 (mixing -0.0f and +0.0f), 200 strictly
        // above, rest strictly below
        let r3: Vec<f32> = (0..cols)
            .map(|g| {
                if g % 10 == 7 && g < 2000 {
                    5.0 + (g as f32) * 1e-4
                } else if g % 2 == 0 {
                    if g % 4 == 0 { -0.0f32 } else { 0.0f32 }
                } else {
                    -1.0 - (g as f32) * 1e-3
                }
            })
            .collect();

        // Row 4: 50% tied at negative threshold -2.5f, 300 strictly above (still negative!)
        let r4: Vec<f32> = (0..cols)
            .map(|g| {
                if g % 7 == 3 && g < 2100 {
                    -0.1 - (g as f32) * 1e-5
                } else if g % 2 == 1 {
                    -2.5f32
                } else {
                    -50.0f32
                }
            })
            .collect();

        // Row 5: strictly negative distinct values with subnormals & -inf/+inf
        let mut r5: Vec<f32> = (0..cols).map(|g| -(g as f32) * 0.01 - 1e-30).collect();
        r5[1500] = f32::INFINITY;
        r5[1501] = f32::MAX;
        r5[0] = f32::NEG_INFINITY;
        r5[10] = -0.0f32;
        r5[11] = 0.0f32;
        r5[12] = -f32::from_bits(1); // negative min subnormal
        r5[13] = f32::from_bits(1); // positive min subnormal

        // Row 6: threshold strictly at -0.0f/+0.0f with > 511 zeros and some -0.0
        // earlier than +0.0
        let mut r6 = vec![-1.0f32; cols];
        for g in 0..100 {
            r6[cols - 1 - g] = 1.0f32; // invisible or late high values
        }
        for (g, v) in r6.iter_mut().enumerate().take(1500).skip(100) {
            *v = if g < 800 { -0.0f32 } else { 0.0f32 };
        }

        // Row 7: every visible entry is -0.0f except 100 late entries are +0.0f
        let mut r7 = vec![-0.0f32; cols];
        r7[1800..cols].fill(0.0f32);

        let rows_sc = vec![r0, r1, r2, r3, r4, r5, r6, r7];
        let visible = vec![vis as i32; rows_sc.len()];
        let tail_starts = vec![(vis * kpool) as i32; rows_sc.len()];
        let tail_lens = vec![1i32, 2, 3, 4, 1, 2, 3, 4];

        check_topk_case(
            "heavy ties & signed zero",
            &rows_sc,
            &visible,
            &tail_starts,
            &tail_lens,
            n_top,
            kpool,
            kmax,
        );

        // Also test NaN handling explicitly: NaNs must rank strictly below -inf.
        let mut nan_row = vec![f32::NAN; 1024];
        nan_row[700] = f32::NEG_INFINITY;
        nan_row[200] = -1000.0;
        nan_row[900] = 0.0;
        let sc = arr(&nan_row, &[1, 1024]);
        let vis_arr = MxArray::from_int32(&[1024], &[1]).unwrap();
        let ts_arr = MxArray::from_int32(&[4096], &[1]).unwrap();
        let tl_arr = MxArray::from_int32(&[2], &[1]).unwrap();
        if let Some(got_arr) = try_indexer_topk_cells(
            &sc,
            &vis_arr,
            &ts_arr,
            &tl_arr,
            None,
            4,
            4,
            20,
            BLOCK as usize,
        )
        .unwrap()
        {
            let got = read_i32(&got_arr);
            // Groups 200, 700, 900 are finite/-inf; 4th pick is lowest-index NaN = group 0!
            // Sorted ascending: groups [0, 200, 700, 900], plus tail [4096, 4097], padded -1.
            let expected = vec![
                0, 1, 2, 3, 800, 801, 802, 803, 2800, 2801, 2802, 2803, 3600, 3601, 3602, 3603,
                4096, 4097, -1, -1,
            ];
            assert_eq!(
                got, expected,
                "NaN ranks below -inf with lowest-index tie-break"
            );
        }
    }

    #[test]
    fn end_to_end_real_dim_128_flat_and_paged_slots_match_reference() {
        use crate::models::hy_v4_flash::paged::{PageGeometry, SlotMap, create};

        crate::models::hy_v4_flash::exact_f32();
        let g = IndexerGeometry {
            n_heads: 16,
            head_dim: 128,
            rope_dim: 64,
            kpool: 4,
            n_top: 8,
            select_tail: true,
            eps: 1e-5,
            weight_scale: (2048f32).sqrt().recip(),
            rope_theta: 1e7,
        };
        let (w, hw) = indexer_weights(&g, 303);
        let ix = Indexer::new(g.clone(), &w).unwrap();
        let n = 68usize; // 17 groups > n_top=8, crosses 2 pages of 32 tokens
        let xs = inputs(n, D, 304);
        let qrs = inputs(n, Q_LORA, 305);

        let want = host_cells(&g, &hw, &xs, &qrs);

        // Flat IndexerState (exercises hy4_indexer_score PAGED=false + hy4_indexer_topk_cells PAGED=false)
        let mut flat_st = IndexerState::new(128, &g);
        let flat_prefill = ix
            .select(
                &w,
                prefill(&rows_of(&xs[..64]), &rows_of(&qrs[..64]), 0),
                &mut flat_st,
            )
            .unwrap();
        let flat_decode = ix
            .select(
                &w,
                Call {
                    x: &rows_of(&xs[64..]),
                    qr: &rows_of(&qrs[64..]),
                    start: 64,
                    kind: RowKind::Decode,
                },
                &mut flat_st,
            )
            .unwrap();
        let mut got_flat = cell_sets(&flat_prefill);
        got_flat.extend(cell_sets(&flat_decode));
        assert_eq!(got_flat, want, "flat D=128 H=16 vs host");

        // Paged IndexerState + select_paged_slots (exercises hy4_indexer_score PAGED=true + hy4_indexer_topk_cells PAGED=true)
        let pg = PageGeometry {
            layers: 1,
            half: 32,
            index_rows: (BLOCK as usize / g.kpool) as u32,
            index_dim: g.head_dim as u32,
        };
        let Ok(mut adapter) = create(&pg, 1024, 12, 12, u64::MAX) else {
            return;
        };
        let mut paged_st = PagedIndexerState::new(0, &g);
        // Interleave another sequence's pages so physical page IDs are scattered.
        adapter.record_tokens_for(99, &[1; 32]).unwrap();
        adapter.record_tokens_for(1, &[2; 64]).unwrap();
        let slots0 = SlotMap::of(&adapter).unwrap();
        let got_p0 = ix
            .select_paged_slots(
                &w,
                prefill(&rows_of(&xs[..64]), &rows_of(&qrs[..64]), 0),
                &mut paged_st,
                &mut adapter,
                &slots0,
            )
            .unwrap();
        let want_p0 = slots0.slots(&adapter, &flat_prefill).unwrap();
        assert_eq!(read_i32(&got_p0), read_i32(&want_p0), "paged prefill D=128");

        adapter.record_tokens_for(99, &[1; 32]).unwrap();
        adapter.record_tokens_for(1, &[2; 4]).unwrap();
        let slots1 = SlotMap::of(&adapter).unwrap();
        let got_p1 = ix
            .select_paged_slots(
                &w,
                Call {
                    x: &rows_of(&xs[64..]),
                    qr: &rows_of(&qrs[64..]),
                    start: 64,
                    kind: RowKind::Decode,
                },
                &mut paged_st,
                &mut adapter,
                &slots1,
            )
            .unwrap();
        let want_p1 = slots1.slots(&adapter, &flat_decode).unwrap();
        assert_eq!(read_i32(&got_p1), read_i32(&want_p1), "paged decode D=128");
    }

    #[test]
    fn tok_tail_ring_wrap_truncate_rollback_and_snapshot_parity() {
        crate::models::hy_v4_flash::exact_f32();
        let g = IndexerGeometry {
            n_heads: 16,
            head_dim: 128,
            rope_dim: 64,
            kpool: 4,
            n_top: 8,
            select_tail: true,
            eps: 1e-5,
            weight_scale: (2048f32).sqrt().recip(),
            rope_theta: 1e7,
        };
        let keep = g.kpool - 1 + WINDOW_SLACK;
        let all_rows = inputs(64, g.head_dim, 401);

        // Push across multiple circular ring wraps with M = 1, 2, 4, 3 steps.
        let mut tail = TokTail::new(&g);
        let mut pos = 0usize;
        let steps = [
            1usize, 2, 4, 3, 1, 1, 2, 4, 1, 3, 2, 4, 1, 2, 4, 1, 3, 2, 4, 3,
        ];
        for &n in &steps {
            let chunk = rows_of(&all_rows[pos..pos + n]);
            tail.push(&chunk).unwrap();
            pos += n;
            assert_eq!(tail.len(), pos);
            assert_eq!(tail.held(), pos.min(keep));
            let snap = tail.snapshot().unwrap();
            assert_eq!(snap.len(), pos);
            assert_eq!(snap.held(), pos.min(keep));
            let got = read(snap.rows().unwrap());
            let start = pos - pos.min(keep);
            let want: Vec<f32> = all_rows[start..pos].iter().flatten().copied().collect();
            assert_eq!(got, want, "ring snapshot mismatch at pos={pos}");

            // Roundtrip through of_rows produces identical linear snapshot.
            let rebuilt = TokTail::of_rows(&g, snap.rows().cloned(), pos).unwrap();
            assert_eq!(read(rebuilt.rows().unwrap()), want);
        }

        // O(1) rollback within rollback_slack (1..=4) restores all `keep` retained rows.
        assert!(tail.rollback_slack() >= 4);
        let before_pos = pos;
        let before_held = tail.held();
        let push4 = rows_of(&all_rows[pos..pos + 4]);
        tail.push(&push4).unwrap();
        assert_eq!(tail.len(), before_pos + 4);
        tail.rollback_push(before_pos, before_held);
        assert_eq!(tail.len(), before_pos);
        assert_eq!(tail.held(), before_held);
        let snap_after_rb = tail.snapshot().unwrap();
        let want_after_rb: Vec<f32> = all_rows[before_pos - keep..before_pos]
            .iter()
            .flatten()
            .copied()
            .collect();
        assert_eq!(
            read(snap_after_rb.rows().unwrap()),
            want_after_rb,
            "ring O(1) rollback corrupted retained rows"
        );

        // End-to-end real D=128 kpool=4 decode chunks crossing ring wraps and group closures.
        let (w, hw) = indexer_weights(&g, 409);
        let ix = Indexer::new(g.clone(), &w).unwrap();
        let n = 40usize;
        let xs = inputs(n, D, 410);
        let qrs = inputs(n, Q_LORA, 411);
        let want = host_cells(&g, &hw, &xs, &qrs);

        let run_schedule = |sched: &[usize]| -> (Vec<Vec<i32>>, Vec<f32>) {
            let mut st = IndexerState::new(64, &g);
            let mut cells = Vec::new();
            let mut at = 0usize;
            for &m in sched {
                let got = ix
                    .select(
                        &w,
                        Call {
                            x: &rows_of(&xs[at..at + m]),
                            qr: &rows_of(&qrs[at..at + m]),
                            start: at,
                            kind: RowKind::Decode,
                        },
                        &mut st,
                    )
                    .unwrap();
                cells.extend(cell_sets(&got));
                at += m;
            }
            let pooled = read(st.pooled.rows().unwrap());
            (cells, pooled)
        };

        for (dt, tol) in [(DType::Float32, 1e-5f64), (DType::Float16, 1e-3f64)] {
            crate::models::hy_v4_flash::cache::with_row_dtype(dt, || {
                let (cells_1, pooled_1) = run_schedule(&vec![1; n]);
                assert_eq!(cells_1, want, "D=128 {dt:?} M=1 schedule vs host");
                for sched in [
                    vec![2; n / 2],
                    vec![4; n / 4],
                    vec![3, 1, 2, 4, 1, 3, 2, 4, 1, 2, 3, 4, 2, 1, 3, 4],
                ] {
                    let (cells, pooled) = run_schedule(&sched);
                    assert_eq!(cells, want, "D=128 {dt:?} schedule {sched:?} vs host");
                    assert_eq!(pooled.len(), pooled_1.len());
                    for (i, (a, b)) in pooled.iter().zip(&pooled_1).enumerate() {
                        close(
                            *a,
                            *b,
                            tol,
                            tol,
                            &format!("D=128 {dt:?} pooled {sched:?}[{i}]"),
                        );
                    }
                }
            });
        }

        // Failed decode update on real D=128 fast-rollback path leaves state bit-exact.
        let mut st = IndexerState::new(64, &g);
        ix.update(&w, &rows_of(&xs[..18]), 0, &mut st).unwrap();
        let snap_before = read(st.tail().snapshot().unwrap().rows().unwrap());
        let live_pooled = |s: &IndexerState| {
            read(
                &s.pooled
                    .rows()
                    .unwrap()
                    .slice(&[0, 0], &[s.pooled_len() as i64, g.head_dim as i64])
                    .unwrap(),
            )
        };
        let pooled_before = live_pooled(&st);
        let meta_before = (st.len(), st.pooled_len(), st.held());
        {
            let _armed = crate::array::fail_native_call_at(
                "hy_v4_flash indexer tail",
                "test: ring tail eval failed",
            );
            assert!(ix.update(&w, &rows_of(&xs[18..22]), 18, &mut st).is_err());
        }
        assert_eq!((st.len(), st.pooled_len(), st.held()), meta_before);
        assert_eq!(
            read(st.tail().snapshot().unwrap().rows().unwrap()),
            snap_before
        );
        assert_eq!(live_pooled(&st), pooled_before);
        ix.update(&w, &rows_of(&xs[18..22]), 18, &mut st).unwrap();
        assert_eq!((st.len(), st.pooled_len()), (22, 5));
    }

    #[test]
    fn indexer_score_decode_nax_large_contexts_flat_and_paged_match_reference() {
        crate::models::hy_v4_flash::exact_f32();
        let d = 128usize;
        let index_rows = 8usize;

        for (h, s, n_groups, start_g, seed) in [
            (16usize, 1usize, 256usize, 0usize, 701u64),
            (16, 2, 256, 7, 702),
            (16, 4, 256, 13, 703),
            (16, 1, 2048, 0, 704),
            (16, 2, 2048, 9, 705),
            (16, 4, 2048, 5, 706),
            (16, 1, 7763, 3, 707),
            (16, 1, 8192, 0, 708),
            (16, 2, 8192, 11, 709),
            (16, 4, 8192, 4, 710),
            (8, 2, 256, 3, 711),
        ] {
            let mut rng = Rng(seed);
            let q_vec = rng.vec(s * h * d, 1.5);
            let w_vec = rng.vec(s * h, 0.2);
            let k_f32_vec = rng.vec(n_groups * d, 1.5);

            let q = arr(&q_vec, &[s, h, d]);
            let w = arr(&w_vec, &[s, h]);
            let keys_f16 = arr(&k_f32_vec, &[n_groups, d])
                .astype(DType::Float16)
                .unwrap();
            let keys_ref_f32 = keys_f16.astype(DType::Float32).unwrap();
            let want = read(&scores(&q, &w, &keys_ref_f32).unwrap());

            let cap_g = start_g + n_groups + 17;
            let keys_bits = keys_f16.to_uint16_native().unwrap().to_vec();
            let mut flat_bits = vec![0u16; cap_g * d];
            flat_bits[start_g * d..(start_g + n_groups) * d].copy_from_slice(&keys_bits);
            let flat_pool = MxArray::from_float16(&flat_bits, &[cap_g as i64, d as i64]).unwrap();

            let Some(got_flat_arr) =
                try_indexer_score(&q, &w, &flat_pool, None, n_groups, start_g).unwrap()
            else {
                eprintln!("skipping indexer_score large context test (no GPU)");
                return;
            };
            let got_flat = read(&got_flat_arr);
            assert_eq!(got_flat.len(), want.len());
            let mut max_diff = 0.0f32;
            for (&g, &wv) in got_flat.iter().zip(&want) {
                assert!(g.is_finite(), "non-finite flat H={h} S={s} N_g={n_groups}");
                max_diff = max_diff.max((g - wv).abs());
            }
            assert!(
                max_diff <= 1e-3,
                "flat H={h} S={s} N_g={n_groups} start_g={start_g}: max_abs={max_diff:e} > 1e-3"
            );

            let n_pages = (start_g + n_groups).div_ceil(index_rows);
            let n_blocks = n_pages + 9;
            let pages_u32 = scattered_pages(n_pages, n_blocks);
            let pages_arr = page_ids(&pages_u32).unwrap();
            let paged_pool = pack_paged_keys(
                &keys_bits, n_groups, start_g, &pages_u32, n_blocks, index_rows, d,
            );
            let got_paged_arr =
                try_indexer_score(&q, &w, &paged_pool, Some(&pages_arr), n_groups, start_g)
                    .unwrap()
                    .expect("paged indexer_score");
            let got_paged = read(&got_paged_arr);
            assert_eq!(
                got_paged.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                got_flat.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "paged vs flat mismatch H={h} S={s} N_g={n_groups}"
            );

            if s == 4 && n_groups == 2048 {
                for row in 0..s {
                    let qr = q
                        .slice(&[row as i64, 0, 0], &[(row + 1) as i64, h as i64, d as i64])
                        .unwrap();
                    let wr = w
                        .slice(&[row as i64, 0], &[(row + 1) as i64, h as i64])
                        .unwrap();
                    let one = read(
                        &try_indexer_score(
                            &qr,
                            &wr,
                            &paged_pool,
                            Some(&pages_arr),
                            n_groups,
                            start_g,
                        )
                        .unwrap()
                        .unwrap(),
                    );
                    let row_got = &got_paged[row * n_groups..(row + 1) * n_groups];
                    assert_eq!(
                        row_got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        one.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "row {row} invariance N_g={n_groups}"
                    );
                }
            }
        }
    }

    #[test]
    fn indexer_score_decode_bench_timing() {
        let h = 16usize;
        let d = 128usize;
        let index_rows = 8usize;
        let mode = if std::env::var("MLX_HY4_INDEXER_DECODE_NAX").as_deref() == Ok("0") {
            "scalar"
        } else {
            "nax"
        };
        let iters = 20usize;
        let batch_calls = 36usize;

        for (s, n_groups, paged) in [
            (1usize, 2048usize, true),
            (1, 7763, true),
            (2, 7763, true),
            (4, 7763, true),
            (1, 8192, false),
            (4, 8192, false),
        ] {
            let mut rng = Rng(801 + (s * 17 + n_groups) as u64);
            let q = arr(&rng.vec(s * h * d, 1.0), &[s, h, d]);
            let w = arr(&rng.vec(s * h, 0.2), &[s, h]);
            let keys_f16 = arr(&rng.vec(n_groups * d, 1.0), &[n_groups, d])
                .astype(DType::Float16)
                .unwrap();
            let keys_bits = keys_f16.to_uint16_native().unwrap().to_vec();
            let (pool, pages) = if paged {
                let n_pages = n_groups.div_ceil(index_rows);
                let n_blocks = n_pages + 9;
                let pages_u32 = scattered_pages(n_pages, n_blocks);
                let pages_arr = page_ids(&pages_u32).unwrap();
                let paged_pool =
                    pack_paged_keys(&keys_bits, n_groups, 0, &pages_u32, n_blocks, index_rows, d);
                paged_pool.eval();
                pages_arr.eval();
                (paged_pool, Some(pages_arr))
            } else {
                keys_f16.eval();
                (keys_f16, None)
            };
            q.eval();
            w.eval();

            for _ in 0..4 {
                let Some(o) =
                    try_indexer_score(&q, &w, &pool, pages.as_ref(), n_groups, 0).unwrap()
                else {
                    return;
                };
                o.eval();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let mut outs = Vec::with_capacity(batch_calls);
                for _ in 0..batch_calls {
                    outs.push(
                        try_indexer_score(&q, &w, &pool, pages.as_ref(), n_groups, 0)
                            .unwrap()
                            .unwrap(),
                    );
                }
                let refs: Vec<&MxArray> = outs.iter().collect();
                crate::models::hy_v4_flash::stats::eval(&refs, "indexer bench").unwrap();
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / (iters * batch_calls) as f64;
            eprintln!(
                "[indexer_score_bench:{mode}] S={s} N_g={n_groups} paged={paged}: {us:.2} us/layer (12L={:.2} ms)",
                us * 12.0 / 1000.0
            );
        }
    }
}
