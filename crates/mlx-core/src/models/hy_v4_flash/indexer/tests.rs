use serde_json::Value;

use super::*;
use crate::models::deepseek_v41::ops::rope_angle;
use crate::models::hy_v4_flash::attention::{Geometry, Mla};
use crate::models::hy_v4_flash::cache::LayerCache;
use crate::models::hy_v4_flash::test_support::{
    Rng, arr, assert_close_f64, assert_row_invariant, error_stats, fixture, j_f32s as f32s,
    j_usize as u, j_usizes as usizes, make_sym_q8, read_f32, read_i32,
};
use crate::models::hy_v4_flash::weights::AttnWeights;

const D: usize = 12;
const Q_LORA: usize = 6;

fn call<'a>(x: &'a MxArray, qr: &'a MxArray, start: usize, kind: RowKind) -> Call<'a> {
    Call { x, qr, start, kind }
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

fn dense(v: &[f32], rows: usize, cols: usize) -> Dense {
    Dense::Array(arr(v, &[rows, cols]))
}

fn assert_close(got: &[f32], want: &[f32], abs: f64, rel: f64, what: &str) {
    let wf: Vec<f64> = want.iter().map(|&x| f64::from(x)).collect();
    assert_close_f64(got, &wf, abs, rel, what);
}

#[test]
fn row_rules_match_kpool_splits_rows() {
    let (g, f) = (real(), fixture("kpool-splits.json"));
    let check = |r: &Value| {
        let t = u(&r["t"]);
        assert_eq!(g.visible(t), u(&r["visible_groups"]), "t {t}");
        assert_eq!(g.picked(t), u(&r["picked_groups"]), "t {t}");
        assert_eq!(!g.is_dense(t), r["topk"].as_bool().unwrap());
        assert_eq!(g.is_dense(t), r["dense"].as_bool().unwrap());
        assert_eq!(
            (t / g.kpool, g.tail_cells(t).len()),
            (u(&r["own_group"]), u(&r["own_cells"]))
        );
        assert!(
            g.cells(t) == u(&r["cells"]) && g.cells(t) <= g.kmax(),
            "t {t}"
        );
    };
    f["rows"].as_array().unwrap().iter().for_each(check);
    let splits = f["two_call_splits"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|s| s["calls"].as_array().unwrap());
    let chain = f["chain"]["calls"].as_array().unwrap().iter();
    let dec = f["decode_after_2045"].as_array().unwrap().iter();
    for c in chain.chain(splits).chain(dec) {
        for key in ["first_row", "last_row"] {
            if !c[key].is_null() {
                check(&c[key]);
            }
        }
    }
}

fn check_call(g: &IndexerGeometry, c: &Value, budget: usize) {
    let (s, e) = (u(&c["s"]), u(&c["e"]));
    let (p, closed, rows) = (g.plan(s, e - s), usizes(&c["groups_closed_by_call"]), e - s);
    assert_eq!([p.groups.start, p.groups.end], closed.as_slice());
    assert_eq!(usizes(&c["pooled_write_range"]), closed);
    assert_eq!(p.groups.start, u(&c["groups_closed_before"]));
    assert_eq!(p.groups.end, u(&c["groups_closed_after"]));
    assert_eq!(
        p.open_in.collect::<Vec<_>>(),
        usizes(&c["tail_in"]),
        "[{s},{e}) tail_in"
    );
    assert_eq!(
        p.tail_out.clone().collect::<Vec<_>>(),
        usizes(&c["tail_out"]),
        "[{s},{e})"
    );
    assert!(p.tail_out.len() < g.kpool);
    assert_eq!(p.cols, u(&c["score_cols"]));
    assert_eq!(score_row_bytes(g.n_heads, p.cols), u(&c["score_row_bytes"]));
    let per = score_rows(g.n_heads, p.cols, budget).min(rows);
    assert_eq!(per, u(&c["score_rows_per_sub_chunk"]));
    assert_eq!(rows.div_ceil(per), u(&c["score_sub_chunks"]));
    assert_eq!(
        (p.sparse.len(), p.dense.len() + p.sparse.len()),
        (u(&c["rows_needing_topk"]), rows)
    );
    let max_cells = (s..e).map(|t| g.cells(t)).max().unwrap();
    assert!(max_cells == u(&c["max_cells"]) && max_cells <= p.kmax);
    assert_eq!(p.kmax, if p.sparse.is_empty() { e } else { g.kmax() });
}

#[test]
fn call_plans_match_kpool_splits() {
    let (g, f) = (real(), fixture("kpool-splits.json"));
    let (geo, budget) = (&f["geometry"], u(&f["geometry"]["score_sub_chunk_budget"]));
    assert_eq!(
        (u(&geo["kpool"]), u(&geo["n_top_pool"])),
        (g.kpool, g.n_top)
    );
    assert_eq!(
        (u(&geo["indexer_heads"]), budget),
        (g.n_heads, INDEX_CHUNK_BYTES)
    );
    for c in f["chain"]["calls"].as_array().unwrap() {
        check_call(&g, c, budget);
    }
    for split in f["two_call_splits"].as_array().unwrap() {
        let calls = split["calls"].as_array().unwrap();
        calls.iter().for_each(|c| check_call(&g, c, budget));
        let last = calls.last().unwrap();
        let one = g.plan(0, u(&last["e"]));
        assert_eq!(one.groups.end, u(&last["groups_closed_after"]));
        assert_eq!(one.tail_out.collect::<Vec<_>>(), usizes(&last["tail_out"]));
    }
    for c in f["decode_after_2045"].as_array().unwrap() {
        check_call(&g, c, budget);
    }
    for (len, st) in f["state_after_len"].as_object().unwrap() {
        let p = g.plan(0, len.parse().unwrap());
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
    let (g, k) = (real(), &fixture("ops.json")["kpool_select"]);
    assert_eq!((u(&k["top_pool"]), u(&k["kpool"])), (g.n_top, g.kpool));
    for c in k["cases"].as_array().unwrap() {
        let t = u(&c["t"]);
        assert_eq!(g.visible(t), u(&c["n_visible"]));
        assert_eq!(
            (g.picked(t), g.cells(t)),
            (u(&c["n_picked"]), u(&c["n_cells"]))
        );
        assert_eq!(
            g.tail_cells(t).collect::<Vec<_>>(),
            usizes(&c["tail_cells"]),
            "t {t}"
        );
        let dense = c["all_earlier_visible"].as_bool().unwrap();
        assert_eq!(g.is_dense(t), dense, "t {t}");
        if dense {
            assert_eq!(
                usizes(&c["cells_sorted"]),
                (0..=t).collect::<Vec<_>>(),
                "t {t}"
            );
            let dr = dense_rows(t, 1, g.kmax()).unwrap();
            assert_eq!(dr.iter().filter(|&&x| x >= 0).count(), t + 1);
        }
    }
}

#[test]
fn plan_splits_dense_and_scored_rows() {
    let g = real();
    assert_eq!(
        (g.first_sparse(), g.tail_slots(), g.kmax()),
        (2048, 4, 2048)
    );
    let p = g.plan(0, 8192);
    assert_eq!((p.dense, p.sparse, p.kmax), (0..2048, 2048..8192, 2048));
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

    let nt = IndexerGeometry {
        select_tail: false,
        n_top: 512,
        ..real()
    };
    assert_eq!(
        (nt.visible(3), nt.tail_cells(3), nt.tail_cells(6)),
        (1, 4..4, 4..7)
    );
    assert_eq!(
        (nt.first_sparse(), nt.tail_slots(), nt.kmax()),
        (2051, 3, 2051)
    );
    for t in [0, 1, 2047, 2050, 2051, 5000] {
        assert!(
            nt.cells(t) <= nt.kmax() && nt.is_dense(t) == (nt.cells(t) == t + 1),
            "t {t}"
        );
    }
}

#[test]
fn dense_rows_pad_to_their_width() {
    assert_eq!(
        dense_rows(2, 3, 5).unwrap(),
        [0, 1, 2, -1, -1, 0, 1, 2, 3, -1, 0, 1, 2, 3, 4]
    );
    assert_eq!(dense_rows(0, 1, 1).unwrap(), [0]);
    assert!(dense_rows(4, 2, 5).is_err());
}

#[test]
fn score_rows_stay_within_the_budget() {
    assert_eq!(score_row_bytes(16, 65_536), 4 << 20);
    assert_eq!(score_rows(16, 65_536, INDEX_CHUNK_BYTES), 64);
    assert_eq!(score_rows(16, 0, INDEX_CHUNK_BYTES), INDEX_CHUNK_BYTES);
    assert_eq!(score_rows(16, 1 << 30, 1), 1);
}

fn cell_sets(idx: &MxArray) -> Vec<Vec<i32>> {
    let w = idx.shape_at(1).unwrap() as usize;
    read_i32(idx)
        .chunks(w)
        .map(|r| {
            let mut v: Vec<i32> = r.iter().copied().filter(|&c| c >= 0).collect();
            v.sort_unstable();
            v
        })
        .collect()
}

fn zero_weights(g: &IndexerGeometry, dim: usize, q_lora: usize) -> IndexerWeights {
    let (h, d, kp) = (g.n_heads, g.head_dim, g.kpool);
    IndexerWeights {
        attn_k: dense(&vec![0.0; d * dim], d, dim),
        k_norm: arr(&vec![1.0; d], &[d]),
        q_b: dense(&vec![0.0; h * d * q_lora], h * d, q_lora),
        proj: dense(&vec![0.0; h * dim], h, dim),
        compressor_value: arr(&vec![0.0; d * d], &[d, d]),
        compressor_gate: arr(&vec![0.0; d * d], &[d, d]),
        compressor_ape: arr(&vec![0.0; kp * d], &[kp, d]),
        compressor_norm: arr(&vec![1.0; d], &[d]),
    }
}

#[test]
fn pool_rope_and_scores_match_ops_fixture() {
    crate::models::hy_v4_flash::exact_f32();
    let k = &fixture("ops.json")["kpool_pool_score"];
    let dm = &k["dims"];
    let (t, d, h) = (u(&dm["t"]), u(&dm["head_dim"]), u(&dm["heads"]));
    let (kp, rd) = (u(&dm["kpool"]), u(&dm["n_rot"]));
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
    let mut w = zero_weights(&g, 8, 8);
    w.compressor_value = arr(&f32s(&k["wv"]), &[d, d]);
    w.compressor_gate = arr(&f32s(&k["wg"]), &[d, d]);
    w.compressor_ape = arr(&f32s(&k["ape"]), &[kp, d]);
    w.compressor_norm = arr(&f32s(&k["norm_w"]), &[d]);
    let (ix, ng, tok) = (
        Indexer::new(g.clone(), &w).unwrap(),
        t / kp,
        f32s(&k["tok_k"]),
    );
    let keys = ix
        .pool(&w, &arr(&tok[..ng * kp * d], &[ng, kp, d]), 0)
        .unwrap();
    let want_keys = f32s(&k["pooled_keys_roped_f32"]);
    assert_close(&read_f32(&keys), &want_keys, 1e-5, 1e-4, "pooled key");

    let pos: Vec<u64> = (0..t as u64).collect();
    let q = ix.rope(&arr(&f32s(&k["iq"]), &[t, h, d]), &pos).unwrap();
    let want_q = f32s(&k["q_roped_f32"]);
    assert_close(&read_f32(&q), &want_q, 1e-5, 1e-5, "roped query");

    let wr: Vec<f32> = f32s(&k["w_raw"])
        .into_iter()
        .map(|x| x * g.weight_scale)
        .collect();
    let sc_a = scores(
        &arr(&want_q, &[t, h, d]),
        &arr(&wr, &[t, h]),
        &arr(&want_keys, &[ng, d]),
    );
    let sc = read_f32(&sc_a.unwrap());
    for (ti, row) in k["score"].as_array().unwrap().iter().enumerate() {
        for (gi, v) in row.as_array().unwrap().iter().enumerate() {
            assert_eq!(v.is_null(), gi >= g.visible(ti), "t {ti} g {gi}");
            if let Some(want) = v.as_f64() {
                assert_close_f64(
                    &[sc[ti * ng + gi]],
                    &[want],
                    1e-4,
                    1e-4,
                    &format!("{ti},{gi}"),
                );
            }
        }
    }
}

#[test]
fn pick_groups_ties_to_the_lowest_group() {
    let (g, k) = (real(), &fixture("ops.json")["kpool_select"]);
    let (gs, cases) = (f32s(&k["group_scores"]), k["cases"].as_array().unwrap());
    let rows: Vec<f32> = cases.iter().flat_map(|_| gs.iter().copied()).collect();
    let visible: Vec<i32> = cases.iter().map(|c| u(&c["n_visible"]) as i32).collect();
    let sc = arr(&rows, &[cases.len(), gs.len()]);
    let ids = pick_groups(&sc, &visible, g.n_top).unwrap();
    assert_eq!(ids.shape_at(1).unwrap() as usize, g.n_top);
    let (got, cells) = (
        read_i32(&ids),
        cell_sets(&group_cells(&ids, g.kpool).unwrap()),
    );
    for (i, c) in cases.iter().enumerate() {
        let (t, n) = (u(&c["t"]), u(&c["n_picked"]));
        let row = &got[i * g.n_top..(i + 1) * g.n_top];
        let want_p: Vec<i32> = usizes(&c["picked_sorted"])
            .into_iter()
            .map(|x| x as i32)
            .collect();
        assert!(
            row[..n] == want_p && row[n..].iter().all(|&x| x == -1),
            "t {t} picks"
        );
        let mut all = cells[i].clone();
        all.extend(g.tail_cells(t).map(|x| x as i32));
        all.sort_unstable();
        let want_c: Vec<i32> = usizes(&c["cells_sorted"])
            .into_iter()
            .map(|x| x as i32)
            .collect();
        assert_eq!(all, want_c, "t {t}");
    }
    assert!(pick_groups(&sc, &visible[1..], g.n_top).is_err());
    assert!(pick_groups(&arr(&gs[..100], &[1, 100]), &[100], g.n_top).is_err());
}

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
    let mut r = Rng::new(seed);
    let (h, d, kp) = (g.n_heads, g.head_dim, g.kpool);
    let mat = |r: &mut Rng, rows, cols| r.vec_f32(rows * cols, 1.5 / (cols as f32).sqrt());
    let nw = |r: &mut Rng, n| {
        r.vec_f32(n, 0.5)
            .into_iter()
            .map(|x| 1.0 + x)
            .collect::<Vec<_>>()
    };
    let hw = HostIx {
        attn_k: mat(&mut r, d, D),
        k_norm: nw(&mut r, d),
        q_b: mat(&mut r, h * d, Q_LORA),
        proj: mat(&mut r, h, D),
        value: mat(&mut r, d, d),
        gate: mat(&mut r, d, d),
        ape: r.vec_f32(kp * d, 1.0),
        norm: nw(&mut r, d),
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
    let off = v.len() - 2 * inv.len();
    for (i, &f) in inv.iter().enumerate() {
        let ang = f64::from(rope_angle(pos as u64, f));
        let (c, s) = (f64::from(ang.cos() as f32), f64::from(ang.sin() as f32));
        let (a, b) = (v[off + 2 * i], v[off + 2 * i + 1]);
        (v[off + 2 * i], v[off + 2 * i + 1]) = (a * c - b * s, a * s + b * c);
    }
}

/// Each query's cells, f64 math with pooled keys rounded to f16; score desc, ties to lowest group.
fn host_cells(
    g: &IndexerGeometry,
    hw: &HostIx,
    xs: &[Vec<f32>],
    qrs: &[Vec<f32>],
) -> Vec<Vec<i32>> {
    let (h, d, kp) = (g.n_heads, g.head_dim, g.kpool);
    let inv = inv_freqs(g.rope_dim, g.rope_theta, None);
    let f64s = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let tok: Vec<_> = xs
        .iter()
        .map(|x| rms(&mv(&hw.attn_k, d, D, &f64s(x)), &hw.k_norm, g.eps))
        .collect();
    let mut keys = Vec::with_capacity(xs.len() / kp);
    for gi in 0..xs.len() / kp {
        let mem = &tok[gi * kp..(gi + 1) * kp];
        let v: Vec<_> = mem.iter().map(|m| mv(&hw.value, d, d, m)).collect();
        let gt: Vec<Vec<f64>> = mem
            .iter()
            .enumerate()
            .map(|(s, m)| {
                let ape = &hw.ape[s * d..(s + 1) * d];
                mv(&hw.gate, d, d, m)
                    .into_iter()
                    .zip(ape)
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
        keys.push(
            k.into_iter()
                .map(|x| half::f16::from_f64(x).to_f64())
                .collect::<Vec<_>>(),
        );
    }
    let mut out = Vec::with_capacity(xs.len());
    for t in 0..xs.len() {
        let nv = g.visible(t);
        let mut groups: Vec<usize> = (0..nv).collect();
        if nv > g.n_top {
            let (qr, wt) = (f64s(&qrs[t]), mv(&hw.proj, h, D, &f64s(&xs[t])));
            let qs: Vec<Vec<f64>> = (0..h)
                .map(|hh| {
                    let qb = &hw.q_b[hh * d * Q_LORA..(hh + 1) * d * Q_LORA];
                    let mut q = mv(qb, d, Q_LORA, &qr);
                    rope(&mut q, t, &inv);
                    q
                })
                .collect();
            let ws = f64::from(g.weight_scale);
            let sc: Vec<f64> = (0..nv)
                .map(|gi| {
                    (0..h)
                        .map(|hh| wt[hh] * ws * dot(&qs[hh], &keys[gi]).max(0.0))
                        .sum()
                })
                .collect();
            groups.sort_by(|&a, &b| sc[b].total_cmp(&sc[a]).then(a.cmp(&b)));
            groups.truncate(g.n_top);
            groups.sort_unstable();
        }
        let mut cells: Vec<i32> = groups
            .iter()
            .flat_map(|&gi| (gi * kp..(gi + 1) * kp).map(|c| c as i32))
            .collect();
        cells.extend(g.tail_cells(t).map(|c| c as i32));
        out.push(cells);
    }
    out
}

fn rows_of(v: &[Vec<f32>]) -> MxArray {
    let flat: Vec<f32> = v.iter().flatten().copied().collect();
    arr(&flat, &[v.len(), v[0].len()])
}

fn inputs(n: usize, width: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.vec_f32(width, 2.0)).collect()
}

fn pooled_host(st: &IndexerState, d: usize) -> Vec<f32> {
    read_f32(st.pooled_rows().unwrap())[..st.pooled_len() * d].to_vec()
}

fn select_split(
    ix: &Indexer,
    w: &IndexerWeights,
    st: &mut IndexerState,
    xs: &[Vec<f32>],
    qrs: &[Vec<f32>],
    start: usize,
    sizes: &[usize],
) -> Vec<Vec<i32>> {
    let (mut out, mut at) = (Vec::new(), 0);
    for &n in sizes {
        let (x, qr) = (rows_of(&xs[at..at + n]), rows_of(&qrs[at..at + n]));
        let idx = ix
            .select(w, call(&x, &qr, start + at, RowKind::Prefill), st)
            .unwrap();
        let want_w = ix.geometry().plan(start + at, n).kmax;
        assert_eq!(
            (
                idx.shape_at(0).unwrap() as usize,
                idx.shape_at(1).unwrap() as usize
            ),
            (n, want_w)
        );
        out.extend(cell_sets(&idx));
        at += n;
    }
    assert_eq!(at, xs.len());
    out
}

#[test]
fn split_calls_pick_what_one_call_picks() {
    let g = small();
    let (w, hw) = indexer_weights(&g, 41);
    let ix = Indexer::new(g.clone(), &w).unwrap();
    let (n, xs, qrs) = (23, inputs(23, D, 3), inputs(23, Q_LORA, 4));
    let (want, mut one) = (host_cells(&g, &hw, &xs, &qrs), IndexerState::new(32, &g));
    assert_eq!(
        select_split(&ix, &w, &mut one, &xs, &qrs, 0, &[n]),
        want,
        "one call"
    );
    let pooled = pooled_host(&one, g.head_dim);
    assert_eq!((one.pooled_len(), one.open()), (n / g.kpool, 20..23));
    assert_eq!(one.held(), g.kpool - 1 + WINDOW_SLACK);
    for sizes in [
        vec![1, 3, 4, 5, 10],
        vec![12, 11],
        vec![11, 1, 11],
        vec![13, 10],
        vec![1; n],
    ] {
        let mut st = IndexerState::new(32, &g);
        assert_eq!(
            select_split(&ix, &w, &mut st, &xs, &qrs, 0, &sizes),
            want,
            "split {sizes:?}"
        );
        assert_eq!((st.len(), st.pooled_len(), st.open()), (n, 5, 20..23));
        assert_close(
            &pooled_host(&st, g.head_dim),
            &pooled,
            1e-3,
            1e-3,
            &format!("{sizes:?}"),
        );
    }
}

#[test]
fn truncate_and_restore_keep_the_picks() {
    let g = small();
    let (w, _) = indexer_weights(&g, 43);
    let ix = Indexer::new(g.clone(), &w).unwrap();
    let (n, xs, qrs) = (23, inputs(23, D, 9), inputs(23, Q_LORA, 10));
    let mut st = IndexerState::new(32, &g);
    let want = select_split(&ix, &w, &mut st, &xs, &qrs, 0, &[n]);
    let full = st.mark();
    assert!(full.len() == n && st.truncate(15).is_err() && st.len() == n);
    assert!(st.truncate(24).is_err());

    st.truncate(21).unwrap();
    assert_eq!((st.len(), st.pooled_len(), st.open()), (21, 5, 20..21));
    assert_eq!(
        select_split(&ix, &w, &mut st, &xs[21..], &qrs[21..], 21, &[2]),
        want[21..]
    );

    st.truncate(16).unwrap();
    assert_eq!((st.len(), st.pooled_len(), st.held()), (16, 4, 0));
    assert!(st.restore(full).is_err());
    assert_eq!(
        select_split(&ix, &w, &mut st, &xs[16..], &qrs[16..], 16, &[3, 4]),
        want[16..]
    );

    let meta = |s: &IndexerState| (s.len(), s.pooled_len(), s.open(), s.held());
    let before = meta(&st);
    let (x0, q0) = (rows_of(&xs[..1]), rows_of(&qrs[..1]));
    assert!(
        ix.select(&w, call(&x0, &q0, 5, RowKind::Prefill), &mut st)
            .is_err()
    );
    assert_eq!(meta(&st), before);

    let mut short = IndexerState::new(9, &g);
    select_split(&ix, &w, &mut short, &xs[..6], &qrs[..6], 0, &[6]);
    let before = meta(&short);
    let (x1, q1) = (rows_of(&xs[6..12]), rows_of(&qrs[6..12]));
    assert!(
        ix.select(&w, call(&x1, &q1, 6, RowKind::Prefill), &mut short)
            .is_err()
    );
    assert_eq!(meta(&short), before);

    st.truncate(0).unwrap();
    assert!(st.is_empty());
    assert_eq!(select_split(&ix, &w, &mut st, &xs, &qrs, 0, &[n]), want);
}

#[test]
fn a_failed_update_keeps_the_indexer_state() {
    let g = small();
    let (w, _) = indexer_weights(&g, 47);
    let ix = Indexer::new(g.clone(), &w).unwrap();
    let (xs, mut st) = (inputs(12, D, 21), IndexerState::new(32, &g));
    ix.update(&w, &rows_of(&xs[..8]), 0, &mut st).unwrap();
    let before = (st.len(), st.pooled_len(), st.held());
    assert_eq!(before, (8, 2, g.kpool - 1 + WINDOW_SLACK));
    {
        let _a = crate::array::fail_native_call_at("hy_v4_flash indexer tail", "test: tail failed");
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
    assert!(
        Indexer::new(
            IndexerGeometry {
                rope_dim: g.head_dim,
                ..g
            },
            &w
        )
        .is_err()
    );
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
    let mut r = Rng::new(seed);
    let (h, l, v) = (g.n_heads, g.kv_lora, g.head_v);
    let mut mat = |rows, cols| {
        dense(
            &r.vec_f32(rows * cols, 1.5 / (cols as f32).sqrt()),
            rows,
            cols,
        )
    };
    let ones = |n| arr(&vec![1.0; n], &[n]);
    AttnWeights {
        norm: ones(D),
        sinks: arr(&[0.3, -0.2], &[h]),
        q_a: mat(Q_LORA, D),
        q_a_norm: ones(Q_LORA),
        q_b: mat(h * g.head_qk(), Q_LORA),
        kv_a: mat(g.row(), D),
        kv_a_norm: ones(l),
        k_b: mat(h * l, g.nope),
        v_b: mat(h * v, l),
        output: mat(D, h * v),
        gate: mat(h * v, D),
        post_norm: ones(D),
        q_kv: None,
        q_kv_gate: None,
    }
}

#[test]
fn indexed_layer_past_top_k_gives_the_same_rows_for_every_split() {
    crate::models::hy_v4_flash::exact_f32();
    let (ag, g) = (attn_geometry(), small());
    let (aw, (w, _)) = (attn_weights(&ag, 61), indexer_weights(&g, 62));
    let mla = Mla::new(ag.clone(), &aw).unwrap();
    let ix = Indexer::new(g.clone(), &w).unwrap();
    let (n, hs) = (24, inputs(24, D, 63));
    let run = |sizes: &[usize]| -> Vec<f32> {
        let mut cache = LayerCache::new(Attn::Indexed, ag.window, 32, ag.row() as i64);
        let (mut st, mut out, mut at) = (IndexerState::new(32, &g), Vec::new(), 0);
        for &k in sizes {
            let idx = Indexing {
                indexer: &ix,
                weights: &w,
                state: &mut st,
                kind: RowKind::Prefill,
            };
            let h = rows_of(&hs[at..at + k]);
            out.extend(read_f32(
                &mla.forward(&aw, &h, at, &mut cache, Some(idx)).unwrap(),
            ));
            at += k;
        }
        assert_eq!((cache.len(), st.len()), (n, n));
        out
    };
    let want = run(&[n]);
    assert!(want.len() == n * D && want.iter().all(|x| x.is_finite()));
    for sizes in [vec![12, 12], vec![1, 23], vec![1; n], vec![5, 7, 1, 11]] {
        assert_close(&run(&sizes), &want, 1e-4, 1e-4, &format!("split {sizes:?}"));
    }
}

/// Parameterized decode head-weights bit-exactness across small F32, real dim=3072 F32, and SymQ8.
#[test]
fn decode_head_weights_bit_exact_across_dtypes_and_dims() {
    crate::models::hy_v4_flash::exact_f32();
    let cases = [
        (
            "small_f32",
            small(),
            D,
            false,
            &[3usize][..],
            RowKind::Prefill,
            false,
        ),
        (
            "real_f32",
            real(),
            3072,
            false,
            &[2, 4, 8],
            RowKind::Decode,
            true,
        ),
        ("small_q8", small(), 32, true, &[4], RowKind::Decode, false),
    ];
    for (ci, (label, g, dim, q8, ms, skind, check_ref)) in cases.into_iter().enumerate() {
        let (mut r, mut w) = (Rng::new(41 + ci as u64 * 17), zero_weights(&g, dim, Q_LORA));
        if q8 {
            w.attn_k = make_sym_q8(&mut r, g.head_dim, dim, 0.25);
            w.proj = make_sym_q8(&mut r, g.n_heads, dim, 0.25);
        } else {
            w.proj = dense(&r.vec_f32(g.n_heads * dim, 0.25), g.n_heads, dim);
        }
        let (ix, max_m) = (
            Indexer::new(g.clone(), &w).unwrap(),
            *ms.iter().max().unwrap(),
        );
        let xs = inputs(max_m, dim, 17 + ci as u64 * 13);
        let row_fn = |i| ix.head_weights(&w, &rows_of(&xs[i..i + 1]), skind).unwrap();
        let batch_fn = |m| {
            ix.head_weights(&w, &rows_of(&xs[..m]), RowKind::Decode)
                .unwrap()
        };
        assert_row_invariant(row_fn, batch_fn, ms, label);
        if skind == RowKind::Prefill {
            let pre = ix
                .head_weights(&w, &rows_of(&xs), RowKind::Prefill)
                .unwrap();
            assert_eq!(read_f32(&pre).len(), max_m * g.n_heads);
        }
        if check_ref {
            for &m in ms {
                let x = rows_of(&xs[..m]);
                let got = read_f32(&ix.head_weights(&w, &x, RowKind::Decode).unwrap());
                let mm = linear(&x, &w.proj)
                    .unwrap()
                    .mul_scalar(f64::from(g.weight_scale))
                    .unwrap();
                assert!(error_stats(&got, &read_f32(&mm)).0 <= 1e-4, "{label} M={m}");
            }
        }
    }
}

#[cfg(target_os = "macos")]
mod paged {
    use super::*;
    use crate::models::hy_v4_flash::attention::PagedPool;
    use crate::models::hy_v4_flash::paged::{PageGeometry, SlotMap, create};
    use crate::models::hy_v4_flash::test_support::read_bits;
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

    impl Case {
        fn new() -> Self {
            let (ag, g) = (
                Geometry {
                    kv_lora: 60,
                    ..attn_geometry()
                },
                small(),
            );
            let (aw, (w, _)) = (attn_weights(&ag, 71), indexer_weights(&g, 72));
            let mla = Mla::new(ag.clone(), &aw).unwrap();
            let ix = Indexer::new(g.clone(), &w).unwrap();
            Self {
                ag,
                aw,
                mla,
                g,
                w,
                ix,
                hs: inputs(N, D, 73),
            }
        }
        fn pool(&self) -> Option<PagedKVCacheAdapter> {
            let pg = PageGeometry {
                layers: 1,
                half: (self.ag.row() / 2) as u32,
                index_rows: (32 / self.g.kpool) as u32,
                index_dim: self.g.head_dim as u32,
            };
            create(&pg, 1024, 8, 8, u64::MAX).ok()
        }
        fn flat(&self, sizes: &[usize]) -> (Vec<u32>, Vec<u16>) {
            let mut cache =
                LayerCache::new(Attn::Indexed, self.ag.window, 64, self.ag.row() as i64);
            let (mut st, mut out, mut at) = (IndexerState::new(64, &self.g), Vec::new(), 0);
            for &k in sizes {
                let ix = Indexing {
                    indexer: &self.ix,
                    weights: &self.w,
                    state: &mut st,
                    kind: RowKind::Prefill,
                };
                let h = rows_of(&self.hs[at..at + k]);
                out.extend(read_bits(
                    &self
                        .mla
                        .forward(&self.aw, &h, at, &mut cache, Some(ix))
                        .unwrap(),
                ));
                at += k;
            }
            let raw = st.pooled_rows().unwrap().to_uint16_native().unwrap();
            (out, raw[..st.pooled_len() * self.g.head_dim].to_vec())
        }
        fn call_paged(
            &self,
            adapter: &mut PagedKVCacheAdapter,
            st: &mut PagedIndexerState,
            slots: &SlotMap,
            at: usize,
            k: usize,
        ) -> Result<MxArray> {
            let pool = PagedPool {
                adapter,
                floor: at as u32,
                slots,
            };
            let ix = PagedIndexing {
                indexer: &self.ix,
                weights: &self.w,
                state: st,
                kind: RowKind::Prefill,
            };
            let h = rows_of(&self.hs[at..at + k]);
            self.mla.forward_indexed_paged(&self.aw, &h, at, pool, ix)
        }
        fn paged(
            &self,
            adapter: &mut PagedKVCacheAdapter,
            sizes: &[usize],
        ) -> (Vec<u32>, Vec<u16>) {
            let (mut st, mut out, mut at) = (PagedIndexerState::new(0, &self.g), Vec::new(), 0);
            for (i, &k) in sizes.iter().enumerate() {
                if i < 2 {
                    adapter.record_tokens_for(OTHER, &[3; 32]).unwrap();
                }
                adapter.record_tokens_for(OWNER, &vec![3; k]).unwrap();
                let slots = SlotMap::of(adapter).unwrap();
                out.extend(read_bits(
                    &self.call_paged(adapter, &mut st, &slots, at, k).unwrap(),
                ));
                at += k;
            }
            assert_eq!((st.len(), st.pooled_len()), (N, N / self.g.kpool));
            let pooled = adapter
                .index_rows(0, st.pooled_len() as u32)
                .unwrap()
                .to_uint16_native()
                .unwrap();
            (out, pooled)
        }
    }

    #[test]
    fn paged_indexed_layer_equals_the_flat_layer_bit_for_bit() {
        let c = Case::new();
        for sizes in [vec![N], vec![12, 12, 16], vec![1; N], vec![5, 7, 1, 27]] {
            let Some(mut adapter) = c.pool() else { return };
            let ((want, want_p), (got, got_p)) = (c.flat(&sizes), c.paged(&mut adapter, &sizes));
            assert_eq!(adapter.active_seq_id(), Some(OWNER));
            assert_ne!(adapter.block_table().unwrap().block_ids(), [0, 1]);
            assert!(
                got.len() == N * D && got == want && got_p == want_p,
                "split {sizes:?}"
            );
        }
    }

    #[test]
    fn a_failed_paged_call_keeps_the_indexer_state() {
        let c = Case::new();
        let Some(mut adapter) = c.pool() else { return };
        let mut st = PagedIndexerState::new(0, &c.g);
        adapter.record_tokens_for(OWNER, &[3; 30]).unwrap();
        let slots = SlotMap::of(&adapter).unwrap();
        c.call_paged(&mut adapter, &mut st, &slots, 0, 30).unwrap();
        adapter.record_tokens_for(OWNER, &[3; 6]).unwrap();
        let before = (st.len(), st.pooled_len(), st.open(), st.tail().held());
        assert!(c.call_paged(&mut adapter, &mut st, &slots, 30, 6).is_err());
        assert_eq!(
            (st.len(), st.pooled_len(), st.open(), st.tail().held()),
            before
        );
        let fresh = SlotMap::of(&adapter).unwrap();
        c.call_paged(&mut adapter, &mut st, &fresh, 30, 6).unwrap();
        assert_eq!((st.len(), st.pooled_len()), (36, 9));
    }
}

#[cfg(target_os = "macos")]
mod fused_metal {
    use super::*;
    use crate::models::hy_v4_flash::ops::{
        try_indexer_score, try_indexer_score_causal, try_indexer_topk_cells,
    };
    use crate::models::hy_v4_flash::paged::{
        BLOCK, PageGeometry, SlotMap, create, page_ids, slot_ids,
    };
    use crate::models::hy_v4_flash::test_support::read_bits;

    const D_SCORE: usize = 128;
    const INDEX_ROWS: usize = 8;

    fn i32_arr(v: &[i32]) -> MxArray {
        MxArray::from_int32(v, &[v.len() as i64]).unwrap()
    }

    fn scattered_pages(n_pages: usize, n_blocks: usize) -> Vec<u32> {
        let (mut pages, mut seen) = (Vec::with_capacity(n_pages), vec![false; n_blocks]);
        for i in 0..n_pages {
            let mut p = (i * 7 + 3) % n_blocks;
            while seen[p] {
                p = (p + 1) % n_blocks;
            }
            seen[p] = true;
            pages.push(p as u32);
        }
        pages
    }

    /// Packs logical `[ng, D_SCORE]` f16 keys into a scattered paged `[nb, 8, D_SCORE]` pool.
    fn make_paged_pool(
        keys_bits: &[u16],
        ng: usize,
        sg: usize,
        pad_blocks: usize,
    ) -> (MxArray, MxArray) {
        let np = (sg + ng).div_ceil(INDEX_ROWS);
        let nb = np + pad_blocks;
        let pages_u32 = scattered_pages(np, nb);
        let mut buf = vec![0u16; nb * INDEX_ROWS * D_SCORE];
        for g in 0..ng {
            let ga = sg + g;
            let dst =
                (pages_u32[ga / INDEX_ROWS] as usize * INDEX_ROWS + ga % INDEX_ROWS) * D_SCORE;
            buf[dst..dst + D_SCORE].copy_from_slice(&keys_bits[g * D_SCORE..(g + 1) * D_SCORE]);
        }
        let shape = [nb as i64, INDEX_ROWS as i64, D_SCORE as i64];
        (
            MxArray::from_float16(&buf, &shape).unwrap(),
            page_ids(&pages_u32).unwrap(),
        )
    }

    fn topk_call(
        sc: &MxArray,
        vis: &MxArray,
        ts: &MxArray,
        tl: &MxArray,
        pages: Option<&MxArray>,
        spec: (usize, usize, usize),
    ) -> Option<MxArray> {
        try_indexer_topk_cells(
            sc,
            vis,
            ts,
            tl,
            pages,
            spec.0,
            spec.1,
            spec.2,
            BLOCK as usize,
        )
        .unwrap()
    }

    fn rand_score_inputs(h: usize, s: usize, ng: usize, seed: u64) -> (MxArray, MxArray, MxArray) {
        let mut rng = Rng::new(seed);
        let q = arr(&rng.vec_f32(s * h * D_SCORE, 1.5), &[s, h, D_SCORE]);
        let w = arr(&rng.vec_f32(s * h, 0.2), &[s, h]);
        let k = arr(&rng.vec_f32(ng * D_SCORE, 1.5), &[ng, D_SCORE])
            .astype(DType::Float16)
            .unwrap();
        (q, w, k)
    }

    /// Flat + paged `try_indexer_score` parity against `scores()`, flat == paged bit-exactness,
    /// and optional `S == S × 1`-row invariance. Returns `false` if Metal is unavailable.
    fn check_indexer_score_case(
        h: usize,
        s: usize,
        ng: usize,
        sg: usize,
        seed: u64,
        row_inv: bool,
    ) -> bool {
        let (q, w, keys_f16) = rand_score_inputs(h, s, ng, seed);
        let want = read_f32(&scores(&q, &w, &keys_f16.astype(DType::Float32).unwrap()).unwrap());
        let (cap_g, keys_bits) = (sg + ng + 13, keys_f16.to_uint16_native().unwrap().to_vec());
        let mut flat_bits = vec![0u16; cap_g * D_SCORE];
        flat_bits[sg * D_SCORE..(sg + ng) * D_SCORE].copy_from_slice(&keys_bits);
        let flat_pool = MxArray::from_float16(&flat_bits, &[cap_g as i64, D_SCORE as i64]).unwrap();

        let Some(got_flat) = try_indexer_score(&q, &w, &flat_pool, None, ng, sg).unwrap() else {
            return false;
        };
        let diff = error_stats(&read_f32(&got_flat), &want).0;
        let tol = if s > 16 { 5e-3 } else { 1e-3 };
        assert!(diff <= tol, "flat H={h} S={s} Ng={ng} sg={sg}: {diff:e}");

        let (paged_pool, pages_arr) = make_paged_pool(&keys_bits, ng, sg, 9);
        let got_paged = try_indexer_score(&q, &w, &paged_pool, Some(&pages_arr), ng, sg)
            .unwrap()
            .unwrap();
        assert_eq!(
            read_bits(&got_paged),
            read_bits(&got_flat),
            "paged vs flat H={h} S={s} Ng={ng}"
        );
        if row_inv {
            let row_fn = |r: usize| {
                let qr = q
                    .slice(
                        &[r as i64, 0, 0],
                        &[(r + 1) as i64, h as i64, D_SCORE as i64],
                    )
                    .unwrap();
                let wr = w
                    .slice(&[r as i64, 0], &[(r + 1) as i64, h as i64])
                    .unwrap();
                try_indexer_score(&qr, &wr, &paged_pool, Some(&pages_arr), ng, sg)
                    .unwrap()
                    .unwrap()
            };
            assert_row_invariant(
                row_fn,
                |_| got_paged.clone(),
                &[s],
                &format!("H={h} S={s} Ng={ng}"),
            );
        }
        true
    }

    #[test]
    fn indexer_score_decode_prefill_large_and_causal_match_reference() {
        crate::models::hy_v4_flash::exact_f32();
        let cases = [
            // Decode & prefill small/medium
            (16usize, 1usize, 37usize, 0usize, 101u64, false),
            (16, 1, 256, 11, 102, false),
            (16, 4, 193, 5, 103, true),
            (16, 19, 73, 0, 104, false),
            (16, 64, 260, 8, 105, false),
            // Decode NAX large contexts & SHARE_S (S=1, 2, 4)
            (16, 1, 256, 0, 701, false),
            (16, 2, 256, 7, 702, false),
            (16, 4, 256, 13, 703, false),
            (16, 1, 2048, 0, 704, false),
            (16, 2, 2048, 9, 705, false),
            (16, 4, 2048, 5, 706, true),
            (16, 1, 7763, 3, 707, false),
            (16, 1, 8192, 0, 708, false),
            (16, 2, 8192, 11, 709, false),
            (16, 4, 8192, 4, 710, false),
            (8, 2, 256, 3, 711, false),
            // Prefill BN_G=128 sub-tile boundaries (1..4 active 32-col sub-tiles) & S boundaries
            (16, 17, 1, 0, 901, false),
            (16, 19, 31, 0, 902, false),
            (16, 32, 32, 3, 903, false),
            (16, 63, 33, 0, 904, false),
            (16, 64, 64, 5, 905, false),
            (16, 65, 65, 0, 906, false),
            (16, 64, 96, 7, 907, false),
            (16, 64, 97, 0, 908, false),
            (16, 65, 127, 0, 909, false),
            (16, 64, 128, 4, 910, false),
            (16, 129, 129, 0, 911, false),
            (16, 128, 300, 9, 912, false),
        ];
        for (h, s, ng, sg, seed, inv) in cases {
            if !check_indexer_score_case(h, s, ng, sg, seed, inv) {
                return;
            }
        }

        // Causal g_tile skip: rows 0..192 with monotonically increasing visible_groups 40..520.
        let (h, s, ng, spec) = (16usize, 192usize, 512usize, (16usize, 4usize, 68usize));
        let (q, w, keys_f16) = rand_score_inputs(h, s, ng, 999);
        let kbits = keys_f16.to_uint16_native().unwrap();
        let (paged_pool, pages_arr) = make_paged_pool(&kbits, ng, 0, 7);
        let visible: Vec<i32> = (0..s).map(|r| (40 + r * 5 / 2).min(ng) as i32).collect();
        let ts: Vec<i32> = visible.iter().map(|&v| v * spec.1 as i32).collect();
        let tl: Vec<i32> = (0..s).map(|r| ((r % 4) + 1) as i32).collect();
        let (vis_a, ts_a, tl_a) = (i32_arr(&visible), i32_arr(&ts), i32_arr(&tl));
        for (tag, pool, pages) in [
            ("flat", &keys_f16, None),
            ("paged", &paged_pool, Some(&pages_arr)),
        ] {
            let full = try_indexer_score(&q, &w, pool, pages, ng, 0)
                .unwrap()
                .unwrap();
            let caus = try_indexer_score_causal(&q, &w, pool, pages, Some(&vis_a), ng, 0)
                .unwrap()
                .unwrap();
            let (fb, cb) = (read_bits(&full), read_bits(&caus));
            for (r, &v) in visible.iter().enumerate() {
                let rng = r * ng..r * ng + v as usize;
                assert_eq!(&cb[rng.clone()], &fb[rng], "{tag} r={r}");
            }
            let tk =
                |sc: &MxArray| read_i32(&topk_call(sc, &vis_a, &ts_a, &tl_a, pages, spec).unwrap());
            assert_eq!(tk(&caus), tk(&full), "topk ({tag})");
            let pk = |sc: &MxArray| read_i32(&pick_groups(sc, &visible, spec.0).unwrap());
            assert_eq!(pk(&caus), pk(&full), "pick ({tag})");
        }
    }

    fn ref_topk_cells(
        sc: &MxArray,
        visible: &[i32],
        ts: &[i32],
        tl: &[i32],
        pages: Option<&MxArray>,
        (n_top, kpool, kmax): (usize, usize, usize),
    ) -> Vec<i32> {
        let s = visible.len();
        let ok = sc.shape_at(1).unwrap() as usize >= n_top;
        let picked = ok.then(|| read_i32(&pick_groups(sc, visible, n_top).unwrap()));
        let mut logical = vec![-1i32; s * kmax];
        for r in 0..s {
            let (vis, row) = (
                visible[r].max(0) as usize,
                &mut logical[r * kmax..(r + 1) * kmax],
            );
            let base = if vis <= n_top {
                let nv = vis * kpool;
                for (i, slot) in row.iter_mut().enumerate().take(nv.min(kmax)) {
                    *slot = i as i32;
                }
                nv
            } else {
                let pr = &picked.as_ref().unwrap()[r * n_top..(r + 1) * n_top];
                for (gi, &gid) in pr.iter().enumerate() {
                    if gid >= 0 {
                        for sub in 0..kpool {
                            row[gi * kpool + sub] = gid * kpool as i32 + sub as i32;
                        }
                    }
                }
                n_top * kpool
            };
            for off in 0..tl[r].max(0) as usize {
                if base + off < kmax {
                    row[base + off] = ts[r] + off as i32;
                }
            }
        }
        let a = MxArray::from_int32(&logical, &[s as i64, kmax as i64]).unwrap();
        pages.map_or(logical, |p| read_i32(&slot_ids(p, &a).unwrap()))
    }

    fn check_topk_case(
        label: &str,
        rows: &[Vec<f32>],
        vis: &[i32],
        ts: &[i32],
        tl: &[i32],
        spec: (usize, usize, usize),
    ) {
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let sc = arr(&flat, &[rows.len(), rows[0].len()]);
        let (vis_a, ts_a, tl_a) = (i32_arr(vis), i32_arr(ts), i32_arr(tl));
        let max_cell = vis
            .iter()
            .zip(ts)
            .zip(tl)
            .map(|((&v, &s), &l)| ((v.max(0) as usize) * spec.1).max((s + l).max(0) as usize))
            .max()
            .unwrap_or(1)
            .max(1);
        let np = max_cell.div_ceil(BLOCK as usize);
        let pages_a = page_ids(&scattered_pages(np, np + 11)).unwrap();
        for (mode, pages) in [("flat", None), ("paged", Some(&pages_a))] {
            let Some(got) = topk_call(&sc, &vis_a, &ts_a, &tl_a, pages, spec) else {
                return;
            };
            assert_eq!(
                read_i32(&got),
                ref_topk_cells(&sc, vis, ts, tl, pages, spec),
                "{label} ({mode})"
            );
        }
    }

    #[test]
    fn indexer_topk_cells_dense_fast_path_and_distinct_large_contexts() {
        let (spec, mut rng) = ((511usize, 4usize, 2048usize), Rng::new(201));
        let dvis = [0i32, 1, 17, 510, 511];
        let dts: Vec<i32> = dvis.iter().map(|&v| v * 4).collect();
        let dsc: Vec<Vec<f32>> = (0..dvis.len()).map(|_| rng.vec_f32(520, 5.0)).collect();
        check_topk_case(
            "dense vis <= n_top",
            &dsc,
            &dvis,
            &dts,
            &[1, 4, 2, 3, 4],
            spec,
        );

        for (vis, s, seed) in [(1024usize, 3usize, 202u64), (4096, 10, 203), (8192, 4, 204)] {
            let (mut r, cols) = (Rng::new(seed), vis + 17);
            let vv: Vec<i32> = (0..s).map(|row| (vis - (row % 3)) as i32).collect();
            let ts: Vec<i32> = vv.iter().map(|&v| v * 4).collect();
            let tl: Vec<i32> = (0..s).map(|row| ((row % 4) + 1) as i32).collect();
            let row_fn = |_| {
                (0..cols)
                    .map(|g| r.f32() * 50.0 + g as f32 * 1e-5)
                    .collect()
            };
            let sc: Vec<Vec<f32>> = (0..s).map(row_fn).collect();
            check_topk_case(
                &format!("distinct vis={vis} s={s}"),
                &sc,
                &vv,
                &ts,
                &tl,
                spec,
            );
        }
    }

    #[test]
    fn indexer_topk_cells_heavy_ties_negatives_zeros_and_signed_zero_bit_exact() {
        let (vis, spec) = (2048usize, (511usize, 4usize, 2048usize));
        let cols = vis + 16;
        let r0 = vec![1.25f32; cols];
        let r1 = vec![-3.5f32; cols];
        let r2: Vec<f32> = (0..cols)
            .map(|g| if g % 2 == 0 { -0.0 } else { 0.0 })
            .collect();
        let r3: Vec<f32> = (0..cols)
            .map(|g| match g {
                _ if g % 10 == 7 && g < 2000 => 5.0 + g as f32 * 1e-4,
                _ if g % 4 == 0 => -0.0,
                _ if g % 2 == 0 => 0.0,
                _ => -1.0 - g as f32 * 1e-3,
            })
            .collect();
        let r4: Vec<f32> = (0..cols)
            .map(|g| {
                if g % 7 == 3 && g < 2100 {
                    -0.1 - g as f32 * 1e-5
                } else if g % 2 == 1 {
                    -2.5
                } else {
                    -50.0
                }
            })
            .collect();
        let mut r5: Vec<f32> = (0..cols).map(|g| -(g as f32) * 0.01 - 1e-30).collect();
        (r5[1500], r5[1501], r5[0]) = (f32::INFINITY, f32::MAX, f32::NEG_INFINITY);
        (r5[10], r5[11], r5[12], r5[13]) = (-0.0, 0.0, -f32::from_bits(1), f32::from_bits(1));
        let mut r6 = vec![-1.0f32; cols];
        r6[cols - 100..].fill(1.0);
        r6[100..800].fill(-0.0);
        r6[800..1500].fill(0.0);
        let mut r7 = vec![-0.0f32; cols];
        r7[1800..].fill(0.0);
        let rows = [r0, r1, r2, r3, r4, r5, r6, r7];
        let (vv, ts, tl) = (
            [vis as i32; 8],
            [(vis * 4) as i32; 8],
            [1, 2, 3, 4, 1, 2, 3, 4],
        );
        check_topk_case("heavy ties & signed zero", &rows, &vv, &ts, &tl, spec);

        // NaNs rank strictly below -inf with lowest-index tie-break.
        let mut nan = vec![f32::NAN; 1024];
        (nan[700], nan[200], nan[900]) = (f32::NEG_INFINITY, -1000.0, 0.0);
        let sc = arr(&nan, &[1, 1024]);
        if let Some(got) = topk_call(
            &sc,
            &i32_arr(&[1024]),
            &i32_arr(&[4096]),
            &i32_arr(&[2]),
            None,
            (4, 4, 20),
        ) {
            let want = [
                0, 1, 2, 3, 800, 801, 802, 803, 2800, 2801, 2802, 2803, 3600, 3601, 3602, 3603,
                4096, 4097, -1, -1,
            ];
            assert_eq!(read_i32(&got), want, "NaN ranks below -inf");
        }
    }

    #[test]
    fn end_to_end_real_dim_128_flat_and_paged_slots_match_reference() {
        crate::models::hy_v4_flash::exact_f32();
        let g = IndexerGeometry { n_top: 8, ..real() };
        let (w, hw) = indexer_weights(&g, 303);
        let ix = Indexer::new(g.clone(), &w).unwrap();
        let (n, xs, qrs) = (68usize, inputs(68, D, 304), inputs(68, Q_LORA, 305));
        let want = host_cells(&g, &hw, &xs, &qrs);

        let mut flat_st = IndexerState::new(128, &g);
        let (x0, q0) = (rows_of(&xs[..64]), rows_of(&qrs[..64]));
        let (x1, q1) = (rows_of(&xs[64..]), rows_of(&qrs[64..]));
        let flat_pre = ix
            .select(&w, call(&x0, &q0, 0, RowKind::Prefill), &mut flat_st)
            .unwrap();
        let flat_dec = ix
            .select(&w, call(&x1, &q1, 64, RowKind::Decode), &mut flat_st)
            .unwrap();
        let mut got_flat = cell_sets(&flat_pre);
        got_flat.extend(cell_sets(&flat_dec));
        assert_eq!(got_flat, want, "flat D=128 H=16 vs host");

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
        let phases = [
            ("paged prefill", 0..64, RowKind::Prefill, &flat_pre),
            ("paged decode", 64..n, RowKind::Decode, &flat_dec),
        ];
        for (tag, rng, kind, flat_idx) in phases {
            adapter.record_tokens_for(99, &[1; 32]).unwrap();
            adapter.record_tokens_for(1, &vec![2; rng.len()]).unwrap();
            let slots = SlotMap::of(&adapter).unwrap();
            let (xr, qr) = (rows_of(&xs[rng.clone()]), rows_of(&qrs[rng.clone()]));
            let c = call(&xr, &qr, rng.start, kind);
            let got = ix
                .select_paged_slots(&w, c, &mut paged_st, &mut adapter, &slots)
                .unwrap();
            let want_p = slots.slots(&adapter, flat_idx).unwrap();
            assert_eq!(read_i32(&got), read_i32(&want_p), "{tag}");
        }
    }

    #[test]
    fn tok_tail_ring_wrap_truncate_rollback_and_snapshot_parity() {
        crate::models::hy_v4_flash::exact_f32();
        let g = IndexerGeometry { n_top: 8, ..real() };
        let (keep, all_rows) = (g.kpool - 1 + WINDOW_SLACK, inputs(64, g.head_dim, 401));
        let flat_slice =
            |r: Range<usize>| all_rows[r].iter().flatten().copied().collect::<Vec<_>>();

        let (mut tail, mut pos) = (TokTail::new(&g), 0usize);
        for n in [
            1usize, 2, 4, 3, 1, 1, 2, 4, 1, 3, 2, 4, 1, 2, 4, 1, 3, 2, 4, 3,
        ] {
            tail.push(&rows_of(&all_rows[pos..pos + n])).unwrap();
            pos += n;
            let (held, snap) = (pos.min(keep), tail.snapshot().unwrap());
            let want = flat_slice(pos - held..pos);
            assert_eq!(
                (tail.len(), tail.held(), snap.len(), snap.held()),
                (pos, held, pos, held)
            );
            assert_eq!(read_f32(snap.rows().unwrap()), want, "ring snap pos={pos}");
            let rebuilt = TokTail::of_rows(&g, snap.rows().cloned(), pos).unwrap();
            assert_eq!(read_f32(rebuilt.rows().unwrap()), want);
        }

        // O(1) rollback within rollback_slack (1..=4) restores all `keep` retained rows.
        assert!(tail.rollback_slack() >= 4);
        let (bpos, bheld) = (pos, tail.held());
        tail.push(&rows_of(&all_rows[pos..pos + 4])).unwrap();
        assert_eq!(tail.len(), bpos + 4);
        tail.rollback_push(bpos, bheld);
        assert_eq!((tail.len(), tail.held()), (bpos, bheld));
        let after = read_f32(tail.snapshot().unwrap().rows().unwrap());
        assert_eq!(after, flat_slice(bpos - keep..bpos));

        // End-to-end real D=128 kpool=4 decode chunks crossing ring wraps and group closures.
        let (w, hw) = indexer_weights(&g, 409);
        let ix = Indexer::new(g.clone(), &w).unwrap();
        let (n, xs, qrs) = (40usize, inputs(40, D, 410), inputs(40, Q_LORA, 411));
        let want = host_cells(&g, &hw, &xs, &qrs);
        let run_sched = |sched: &[usize]| -> (Vec<Vec<i32>>, Vec<f32>) {
            let (mut st, mut cells, mut at) = (IndexerState::new(64, &g), Vec::new(), 0usize);
            for &m in sched {
                let (xm, qm) = (rows_of(&xs[at..at + m]), rows_of(&qrs[at..at + m]));
                let idx = ix
                    .select(&w, call(&xm, &qm, at, RowKind::Decode), &mut st)
                    .unwrap();
                cells.extend(cell_sets(&idx));
                at += m;
            }
            (cells, read_f32(st.pooled.rows().unwrap()))
        };
        let scheds = [
            vec![2; n / 2],
            vec![4; n / 4],
            vec![3, 1, 2, 4, 1, 3, 2, 4, 1, 2, 3, 4, 2, 1, 3, 4],
        ];
        for (dt, tol) in [(DType::Float32, 1e-5f64), (DType::Float16, 1e-3f64)] {
            crate::models::hy_v4_flash::cache::with_row_dtype(dt, || {
                let (c1, p1) = run_sched(&vec![1; n]);
                assert_eq!(c1, want, "{dt:?} M=1");
                for sched in &scheds {
                    let (c, p) = run_sched(sched);
                    assert_eq!(c, want, "{dt:?} {sched:?}");
                    assert_close(&p, &p1, tol, tol, &format!("{dt:?} pooled {sched:?}"));
                }
            });
        }

        // Failed decode update on real D=128 fast-rollback path leaves state bit-exact.
        let mut st = IndexerState::new(64, &g);
        ix.update(&w, &rows_of(&xs[..18]), 0, &mut st).unwrap();
        let snap_v = |s: &IndexerState| read_f32(s.tail().snapshot().unwrap().rows().unwrap());
        let live_p = |s: &IndexerState| {
            let r = s.pooled.rows().unwrap();
            read_f32(
                &r.slice(&[0, 0], &[s.pooled_len() as i64, g.head_dim as i64])
                    .unwrap(),
            )
        };
        let (snap0, pool0, meta0) = (
            snap_v(&st),
            live_p(&st),
            (st.len(), st.pooled_len(), st.held()),
        );
        {
            let _a = crate::array::fail_native_call_at("hy_v4_flash indexer tail", "test: fail");
            assert!(ix.update(&w, &rows_of(&xs[18..22]), 18, &mut st).is_err());
        }
        assert_eq!((st.len(), st.pooled_len(), st.held()), meta0);
        assert_eq!((snap_v(&st), live_p(&st)), (snap0, pool0));
        ix.update(&w, &rows_of(&xs[18..22]), 18, &mut st).unwrap();
        assert_eq!((st.len(), st.pooled_len()), (22, 5));
    }
}
