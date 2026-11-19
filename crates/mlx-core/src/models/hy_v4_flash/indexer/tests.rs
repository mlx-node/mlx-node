use std::path::PathBuf;

use serde_json::Value;

use super::*;
use crate::models::deepseek_v41::ops::rope_angle;
use crate::models::hy_v4_flash::attention::{Geometry, Mla};
use crate::models::hy_v4_flash::cache::LayerCache;
use crate::models::hy_v4_flash::weights::AttnWeights;

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
        let idx = ix.select(w, &x, &qr, start + at, st).unwrap();
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
    assert!(ix.select(&w, &x, &qr, 5, &mut st).is_err());
    assert_eq!((st.len(), st.pooled_len(), st.open(), st.held()), before);

    let mut short = IndexerState::new(9, &g);
    select_split(&ix, &w, &mut short, &xs[..6], &qrs[..6], 0, &[6]);
    let before = (short.len(), short.pooled_len(), short.open(), short.held());
    let (x, qr) = (rows_of(&xs[6..12]), rows_of(&qrs[6..12]));
    assert!(ix.select(&w, &x, &qr, 6, &mut short).is_err());
    assert_eq!(
        (short.len(), short.pooled_len(), short.open(), short.held()),
        before
    );

    st.truncate(0).unwrap();
    assert!(st.is_empty());
    let got = select_split(&ix, &w, &mut st, &xs, &qrs, 0, &[n]);
    assert_eq!(got, want, "from empty");
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
    }
}

#[test]
fn indexed_layer_past_top_k_gives_the_same_rows_for_every_split() {
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
