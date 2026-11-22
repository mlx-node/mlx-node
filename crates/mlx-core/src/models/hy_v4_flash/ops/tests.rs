use serde_json::Value;

use super::*;
use crate::models::deepseek_v41::EngramConfig;
use crate::models::deepseek_v41::engram_hash::EngramHasher;
use crate::models::deepseek_v41::ops::{
    apply_partial_rope, inv_freqs, rope_tables, route_select as ds_route_select,
};
use crate::models::hy_v4_flash::attention::grouped_mm_t_mgh;
use crate::models::hy_v4_flash::engram::{
    self, ENGRAM_EPS, Hasher, IMAGE_TOKEN, VIDEO_TOKEN, keeps, key_value, normed_rows, signal,
    stream_dot, stream_gate,
};
use crate::models::hy_v4_flash::test_support::{
    Rng, arr, assert_close_f64, assert_row_invariant, fixture, j_f32s, j_flat_f64, j_u64, j_u64s,
    j_usize, j_usizes, make_sym_q8, max_abs_err, read_bits, read_f32, read_u32, real_scan,
};
use crate::models::hy_v4_flash::weights::{EngramWeights, quantize_affine4_array};

fn ops() -> Value {
    fixture("ops.json")
}
fn j_u32s(v: &Value) -> Vec<u32> {
    j_flat_f64(v).into_iter().map(|x| x as u32).collect()
}
fn j_arr(v: &Value, shape: &[usize]) -> MxArray {
    arr(&j_f32s(v), shape)
}
fn shape(x: &MxArray) -> Vec<i64> {
    x.shape().unwrap().as_ref().to_vec()
}
fn f32_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}
fn as_dt(x: &MxArray, dt: DType) -> MxArray {
    x.astype(dt).unwrap()
}
fn i32_arr(v: &[i32]) -> MxArray {
    MxArray::from_int32(v, &[v.len() as i64]).unwrap()
}

fn slice_axis(x: &MxArray, last: bool, lo: usize, hi: usize) -> MxArray {
    let mut stop = shape(x);
    let mut start = vec![0i64; stop.len()];
    let ax = if last { stop.len() - 1 } else { 0 };
    start[ax] = lo as i64;
    stop[ax] = hi as i64;
    x.slice(&start, &stop).unwrap()
}

fn assert_oracle(got: &[f32], want: &[f64], what: &str) {
    assert_close_f64(got, want, 1e-6, 1e-5, what);
}
fn assert_oracle_arr(got: &MxArray, want: &Value, what: &str) {
    assert_oracle(&read_f32(got), &j_flat_f64(want), what);
}
fn assert_err(got: &MxArray, want: &MxArray, tol: f32, tag: &str) {
    let e = max_abs_err(&read_f32(got), &read_f32(want));
    assert!(e <= tol, "{tag}: max_abs={e:e} > {tol:e}");
}
fn norm_w(rng: &mut Rng, s: &[usize]) -> MxArray {
    let vals: Vec<f32> = rng
        .vec_f32(s.iter().product(), 0.5)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect();
    arr(&vals, s)
}
fn assert_tol_f32(got: &[f32], want: &[f32], abs: f32, rel: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what} len");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= abs + rel * w.abs(),
            "{what}[{i}]: {g} vs {w}"
        );
    }
}
fn assert_sampled_inv(single: &[(usize, Vec<u32>)], batch: &[u32], m: usize, tag: &str) {
    let stride = batch.len() / m;
    for (r, want) in single {
        if *r < m {
            assert_eq!(
                &batch[r * stride..(r + 1) * stride],
                want.as_slice(),
                "{tag} M={m} r={r}"
            );
        }
    }
}
fn moe_cfg(e: usize, k: usize, scale: f32, norm_topk: bool) -> MoeConfig {
    MoeConfig {
        n_experts: e,
        n_used: k,
        n_shared: 1,
        ff: 1536,
        latent: 1536,
        route_scale: scale,
        norm_topk,
    }
}
fn hc_cfg(o: &Value) -> HcConfig {
    HcConfig {
        count: j_usize(&o["dims"]["hc"]),
        eps: o["hc_eps"].as_f64().unwrap() as f32,
        magnitude: o["magnitude"].as_f64().unwrap_or(2.0) as f32,
    }
}
fn hc_weights(o: &Value, r: usize, c: usize, ns: usize) -> HcWeights {
    HcWeights {
        weight: j_arr(&o["fn"], &[r, c]),
        base: j_arr(&o["base"], &[r]),
        scale: j_arr(&o["scale"], &[ns]),
    }
}
fn norm_eps(o: &Value) -> f32 {
    o["norm_eps"].as_f64().unwrap() as f32
}

#[test]
fn hc_ops_match_the_oracle_and_check_shapes() {
    crate::models::hy_v4_flash::exact_f32();
    let all = ops();
    let (o, op, oh) = (&all["hc_mix"], &all["hc_post"], &all["hc_head"]);
    let d = &o["dims"];
    let x = j_arr(
        &o["x"],
        &[j_usize(&d["n"]), j_usize(&d["hc"]), j_usize(&d["d"])],
    );
    let w = hc_weights(o, 8, 64, 2);
    let mixes = hc_mixes(&x, &w.weight, norm_eps(o)).unwrap();
    assert_eq!(shape(&mixes), [3, 8]);
    assert_oracle_arr(&mixes, &o["mixes"], "mixes");
    let g = hc_mix(&x, &w, norm_eps(o), &hc_cfg(o)).unwrap();
    assert_oracle_arr(&g.pre, &o["pre"], "pre");
    assert_oracle_arr(&g.post, &o["post"], "post");
    let red = hc_reduce(&x, &g.pre).unwrap();
    assert_eq!(shape(&red), [3, 16]);
    assert_oracle_arr(&red, &o["reduced"], "reduced");

    let out = hc_post_identity(
        &j_arr(&op["y"], &[3, 16]),
        &x,
        &j_arr(&op["post_f32"], &[3, 4]),
    )
    .unwrap();
    assert_eq!(shape(&out), [3, 4, 16]);
    assert_oracle_arr(&out, &op["out"], "hc_post");

    let (wh, eps) = (
        hc_weights(oh, 4, 64, 1),
        oh["hc_eps"].as_f64().unwrap() as f32,
    );
    assert_oracle_arr(
        &hc_mixes(&x, &wh.weight, norm_eps(oh)).unwrap(),
        &oh["mixes"],
        "head mixes",
    );
    assert_oracle_arr(
        &hc_head_pre(&x, &wh, norm_eps(oh), eps).unwrap(),
        &oh["pre"],
        "head pre",
    );
    let h = hc_head(&x, &wh, norm_eps(oh), eps).unwrap();
    assert_eq!(shape(&h), [3, 16]);
    assert_oracle_arr(&h, &oh["h"], "head h");

    let bad = HcConfig {
        count: 3,
        ..hc_cfg(o)
    };
    let short = HcWeights {
        weight: j_arr(&o["fn"], &[8, 64]),
        base: j_arr(&o["base"], &[8]),
        scale: arr(&[0.7], &[1]),
    };
    assert!(hc_mix(&x, &w, norm_eps(o), &bad).is_err());
    assert!(hc_mix(&x, &short, norm_eps(o), &hc_cfg(o)).is_err());
    assert!(hc_mixes(&x, &arr(&[0.0; 256], &[8, 32]), 1e-5).is_err());
    assert!(hc_mixes(&x.reshape(&[3, 64]).unwrap(), &w.weight, 1e-5).is_err());
    assert!(hc_reduce(&x, &arr(&[0.0; 9], &[3, 3])).is_err());
    assert!(hc_post_identity(&arr(&[0.0; 45], &[3, 15]), &x, &arr(&[0.0; 12], &[3, 4])).is_err());
    assert!(hc_head(&x, &w, norm_eps(oh), eps).is_err());
}

#[test]
fn router_oracle_and_route_select_fast_bitwise() {
    crate::models::hy_v4_flash::exact_f32();
    let o = &ops()["router"];
    let m = moe_cfg(
        j_usize(&o["dims"]["experts"]),
        j_usize(&o["dims"]["k"]),
        o["route_scale"].as_f64().unwrap() as f32,
        o["norm"].as_bool().unwrap(),
    );
    let (scores, biases) = (
        o["scores_f32_sigmoid"].as_array().unwrap(),
        o["bias_rows"].as_array().unwrap(),
    );
    for r in 0..3 {
        let (ids, w) = route_host(&j_f32s(&scores[r]), Some(&j_f32s(&biases[r])), &m).unwrap();
        assert_eq!(ids, j_u32s(&o["ids"][r]));
        assert_oracle(&w, &j_flat_f64(&o["weights"][r]), &format!("row {r} w"));
    }
    assert_eq!(j_u32s(&o["ids"][2]), (100..116).collect::<Vec<u32>>());
    let s0 = j_f32s(&scores[0]);
    assert!(
        route_host(&s0[..511], None, &m).is_err() && route_host(&s0, Some(&s0[..100]), &m).is_err()
    );
    assert!(
        route_host(
            &s0,
            None,
            &moe_cfg(m.n_experts, 513, m.route_scale, m.norm_topk)
        )
        .is_err()
    );
    assert_eq!(route_host(&s0, None, &m).unwrap().0.len(), 16);

    let n = m.n_experts;
    let mut eye = vec![0f32; n * n];
    for i in 0..n {
        eye[i * n + i] = 1.0;
    }
    let logits = j_arr(&o["logits"], &[3, n]);
    let gscores = router_scores(&logits, &Dense::Array(arr(&eye, &[n, n]))).unwrap();
    assert_eq!(gscores.dtype().unwrap(), DType::Float32);
    assert!(max_abs_err(&read_f32(&gscores), &j_f32s(&o["scores_f32_sigmoid"])) <= 1e-6);
    for r in 0..3 {
        let (ids, _) = route(
            &slice_axis(&gscores, false, r, r + 1),
            Some(&j_f32s(&biases[r])),
            &m,
        )
        .unwrap();
        assert_eq!(ids, j_u32s(&o["ids"][r]));
    }
    assert!(route(&logits.reshape(&[3, 2, 256]).unwrap(), None, &m).is_err());

    let bias = Rng::new(901).vec_f32(512, 0.25);
    let all_cfgs: &[(usize, bool, Option<&[f32]>, f32)] = &[
        (16, true, Some(&bias), 2.5),
        (16, true, None, 1.0),
        (16, false, Some(&bias), 0.5),
        (1, true, Some(&bias), 1.0),
        (2, true, None, 1.75),
    ];
    for &rows in &[1usize, 2, 16, 2048] {
        let mut sc = Rng::new(1000 + rows as u64).vec_f32(rows * 512, 2.0);
        for tr in [0usize, rows - 1] {
            let b = tr * 512;
            sc[b + 100..b + 124].fill(-0.375);
            (sc[b + 124], sc[b + 125]) = (-0.0, 0.0);
            sc[b + 200..b + 210].fill(1.5);
        }
        let cfgs = if rows == 2048 {
            &all_cfgs[..1]
        } else {
            all_cfgs
        };
        for &(k, norm, b, scale) in cfgs {
            let biased: Vec<f32> = match b {
                Some(bv) => sc
                    .iter()
                    .zip(bv.iter().cycle())
                    .map(|(&s, &bi)| s + bi)
                    .collect(),
                None => sc.clone(),
            };
            let (wi, ww) = ds_route_select(&biased, &sc, 512, k, scale, norm);
            let (gi, gw) = route_select_fast(&sc, b, 512, k, scale, norm);
            assert_eq!((gi, f32_bits(&gw)), (wi, f32_bits(&ww)));
        }
    }
}

fn engram_weights(o: &Value) -> EngramWeights {
    let d = |k: &str| j_usize(&o["dims"][k]);
    let (hc, dim, cols, hd) = (d("hc"), d("d"), d("n_cols"), d("head_dim"));
    EngramWeights {
        kv: Dense::Array(j_arr(&o["kv_w"], &[(hc + 1) * dim, cols * hd])),
        kv_b: j_arr(&o["kv_b"], &[hc + 1, dim]),
        gate: j_arr(&o["gate_w"], &[hc, dim]),
        lookup_norm: j_arr(&o["lookup_norm"], &[cols, hd]),
        pre_gamma: j_arr(&o["pre_gamma"], &[dim]),
        hc_scale: j_arr(&o["hc_scale"], &[hc, dim]),
    }
}

#[test]
fn engram_block_keeps_and_hasher_match_the_oracle() {
    crate::models::hy_v4_flash::exact_f32();
    let o = &ops()["engram_block"];
    assert_eq!(o["eps"].as_f64().unwrap() as f32, ENGRAM_EPS);
    let (w, x, rows) = (
        engram_weights(o),
        j_arr(&o["x"], &[2, 4, 8]),
        j_f32s(&o["rows"]),
    );
    let r = normed_rows(&rows, 2, &w).unwrap();
    assert_eq!(shape(&r), [2, 8]);
    assert_oracle_arr(&r, &o["rows_normed"], "rows_normed");
    let kv = key_value(&r, &w).unwrap();
    assert_eq!(shape(&kv), [2, 5, 8]);
    assert_oracle_arr(&kv, &o["kv"], "kv");
    let (key, val) = (
        kv.slice(&[0, 0, 0], &[2, 4, 8]).unwrap(),
        kv.slice(&[0, 4, 0], &[2, 5, 8])
            .unwrap()
            .reshape(&[2, 8])
            .unwrap(),
    );
    let dot = stream_dot(&x, &key, &w.gate).unwrap();
    assert_oracle_arr(&dot, &o["dot"], "dot");
    assert_oracle_arr(&stream_gate(&dot).unwrap(), &o["gate"], "gate");
    assert_oracle_arr(&signal(&val, &w.pre_gamma).unwrap(), &o["signal"], "signal");
    let out = engram::forward(&x, &rows, &[true, true], &w).unwrap();
    assert_eq!(
        (shape(&out), out.dtype().unwrap()),
        (vec![2, 4, 8], DType::Float32)
    );
    assert_oracle_arr(&out, &o["out"], "out");

    let (all, xs) = (read_f32(&out), read_f32(&x));
    let half = xs.len() / 2;
    for (keep, kept) in [([true, false], 0), ([false, true], 1)] {
        let got = read_f32(&engram::forward(&x, &rows, &keep, &w).unwrap());
        assert_eq!(
            got[kept * half..(kept + 1) * half],
            all[kept * half..(kept + 1) * half]
        );
        assert_eq!(
            got[(1 - kept) * half..(2 - kept) * half],
            xs[(1 - kept) * half..(2 - kept) * half]
        );
    }
    assert_eq!(
        read_f32(&engram::forward(&x, &rows, &[false, false], &w).unwrap()),
        xs
    );
    let g0 = read_f32(&stream_gate(&arr(&[0.0, -0.0, 4.0, -4.0], &[1, 4])).unwrap());
    let sig = |v: f64| 1.0 / (1.0 + (-v).exp());
    assert_oracle(
        &g0,
        &[sig(1e-3), sig(1e-3), sig(2.0), sig(-2.0)],
        "gate zero sign",
    );
    assert!(g0[0] > 0.5 && g0[1] > 0.5);
    assert!(engram::forward(&x, &rows, &[true], &w).is_err());
    assert!(engram::forward(&x, &rows[1..], &[true, true], &w).is_err());
    assert!(engram::forward(&x, &[], &[true, true], &w).is_err());
    assert!(
        engram::forward(
            &x.slice(&[0, 0, 0], &[2, 3, 8]).unwrap(),
            &rows,
            &[true, true],
            &w
        )
        .is_err()
    );
    assert!(engram::forward(&x.reshape(&[2, 32]).unwrap(), &rows, &[true, true], &w).is_err());

    let want = fixture("engram-hash.json");
    let pad = j_usize(&want["pad_token_id"]);
    assert_eq!(
        (pad, IMAGE_TOKEN, VIDEO_TOKEN),
        (
            120_002,
            j_u64(&want["image_token_id"]) as u32,
            j_u64(&want["video_token_id"]) as u32
        )
    );
    let mut masked = 0;
    for s in want["sequences"].as_array().unwrap() {
        let mask: Vec<bool> = s["engram_mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_bool().unwrap())
            .collect();
        assert_eq!(keeps(&j_u32s(&s["tokens"]), pad), mask);
        masked += mask.iter().filter(|&&k| !k).count();
    }
    assert!(masked >= 4);

    let cfg = hash_config(&want);
    let h = Hasher::new(&cfg).unwrap();
    assert_eq!((h.n_layers(), h.n_cols()), (4, 16));
    check_hasher(&h, &want);
    let full = EngramHasher::new(&cfg).unwrap();
    let tokens: Vec<u32> = (0..40u32).map(|i| (i * 7919 + 13) % 120832).collect();
    for st in 0..=tokens.len() {
        assert_eq!(h.call_rows(&tokens, st).unwrap(), full.rows(&tokens, st));
    }
    let mut b1 = cfg.clone();
    b1.max_ngram = 1;
    let mut b2 = cfg.clone();
    b2.head_vocab_sizes[3] = 0;
    let mut b3 = cfg;
    b3.multipliers.pop();
    assert!(Hasher::new(&b1).is_err() && Hasher::new(&b2).is_err() && Hasher::new(&b3).is_err());
}

fn hash_config(want: &Value) -> EngramConfig {
    let cfg = fixture("config.json");
    let e = &cfg["engram"];
    let mut token_map = vec![0u32; j_usize(&e["token_map_len"])];
    let pad = j_usize(&want["pad_token_id"]);
    token_map[pad] = j_u64(&want["token_map_pad"]) as u32;
    for s in want["sequences"].as_array().unwrap() {
        for (t, slot) in j_u32s(&s["tokens"])
            .into_iter()
            .zip(s["compressed_slots"].as_array().unwrap())
        {
            let c = j_u64(&slot[0]) as u32;
            let m = &mut token_map[t as usize];
            assert!(*m == 0 || *m == c || t as usize == pad);
            *m = c;
        }
    }
    EngramConfig {
        layers: j_usizes(&e["layers"]),
        max_ngram: j_usize(&e["ngram_size"]),
        n_heads: j_usize(&e["heads_per_ngram"]),
        head_dim: j_usize(&e["head_dim"]),
        multipliers: j_u64s(&e["multipliers"]),
        head_offsets: j_u64s(&e["head_offsets"]),
        head_vocab_sizes: j_u64s(&e["head_vocab_sizes"]),
        pad_token_id: pad,
        compressed_vocab_size: token_map.iter().max().map_or(0, |&x| x as usize + 1),
        token_map,
        vocab_size: None,
    }
}

fn check_hasher(h: &Hasher, want: &Value) {
    let pp = h.n_layers() * h.n_cols();
    for s in want["sequences"].as_array().unwrap() {
        let (toks, rows) = (j_u32s(&s["tokens"]), j_u32s(&s["rows"]));
        assert_eq!(rows.len(), toks.len() * pp);
        for st in 0..=toks.len() {
            assert_eq!(h.call_rows(&toks, st).unwrap(), rows[st * pp..]);
        }
        for e in 0..h.n_layers() {
            let we: Vec<u32> = rows
                .chunks(pp)
                .flat_map(|p| p[e * h.n_cols()..(e + 1) * h.n_cols()].to_vec())
                .collect();
            assert_eq!(h.layer_ids(&rows, e).unwrap(), we);
        }
        assert!(h.call_rows(&toks, toks.len() + 1).is_err());
    }
    assert!(h.layer_ids(&[0; 64], 4).is_err() && h.layer_ids(&[0; 63], 0).is_err());
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_hasher_matches_the_oracle() {
    let want = fixture("engram-hash.json");
    let cfg = real_scan().cfg.engram;
    let pad = cfg.token_map[cfg.pad_token_id];
    assert_eq!(u64::from(pad), j_u64(&want["token_map_pad"]));
    for s in want["sequences"].as_array().unwrap() {
        let toks = j_u32s(&s["tokens"]);
        for (p, slot) in s["compressed_slots"].as_array().unwrap().iter().enumerate() {
            let got: Vec<u32> = (0..3)
                .map(|k| {
                    if p < k {
                        pad
                    } else {
                        cfg.token_map[toks[p - k] as usize]
                    }
                })
                .collect();
            assert_eq!(got, j_u32s(slot));
        }
    }
    check_hasher(&Hasher::new(&cfg).unwrap(), &want);
}

#[test]
fn head_logits_rope_rms_norm_and_swiglu_clamped() {
    // 1. Chunked BF16 LM head vs f64 matmul.
    let (vocab, d) = (37usize, 16usize);
    let words: Vec<u16> = (0..vocab * d)
        .map(|i| half::bf16::from_f32(((i * 37 % 101) as f32 - 50.0) / 64.0).to_bits())
        .collect();
    let head = Dense::Array(MxArray::from_bfloat16(&words, &[vocab as i64, d as i64]).unwrap());
    let xv: Vec<f32> = (0..3 * d)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 8.0)
        .collect();
    let wf: Vec<f32> = words
        .iter()
        .map(|&b| f32::from_bits(u32::from(b) << 16))
        .collect();
    let (xvr, wfr) = (&xv, &wf);
    let want: Vec<f64> = (0..3)
        .flat_map(|r| {
            (0..vocab).map(move |v| {
                (0..d)
                    .map(|k| f64::from(xvr[r * d + k]) * f64::from(wfr[v * d + k]))
                    .sum()
            })
        })
        .collect();
    let x = arr(&xv, &[3, d]);
    for chunk in [8, 37, 1000] {
        let l = head_logits_chunked(&x, &head, chunk).unwrap();
        assert_eq!(
            (shape(&l), l.dtype().unwrap()),
            (vec![3, vocab as i64], DType::Float32)
        );
        assert_oracle(&read_f32(&l), &want, &format!("chunk {chunk}"));
    }
    assert!(head_logits_chunked(&x, &head, 0).is_err());

    // 2. fast_rope & fast_partial_rope_tail vs apply_partial_rope + row invariance.
    let (rd, h, td) = (64usize, 4usize, 128usize);
    let inv = inv_freqs(rd, 10_000.0, None);
    let freqs = rope_inv_freqs(&inv).unwrap();
    let mut rng = Rng::new(71);
    for (name, pos) in [
        ("contig0", (0..6u64).collect::<Vec<_>>()),
        ("contig_off", (13..19).collect()),
        ("kpool4", (0..6).map(|g| g * 4).collect()),
        ("wave", vec![9, 14, 15, 20]),
        ("big1", vec![262_144]),
        ("big2", vec![262_144, 262_145]),
        ("big_s", vec![0, 262_140, 262_144]),
    ] {
        let (n, (cos, sin)) = (pos.len(), rope_tables(&inv, &pos).unwrap());
        for (tag, x, stride) in [
            ("2D", arr(&rng.vec_f32(n * rd, 3.0), &[n, rd]), rd),
            (
                "3D",
                arr(&rng.vec_f32(n * h * rd, 3.0), &[n, h, rd]),
                h * rd,
            ),
        ] {
            let got = fast_rope(&x, &freqs, &pos).unwrap();
            let ref_r = apply_partial_rope(&x, rd as i64, &cos, &sin, false).unwrap();
            assert_tol_f32(
                &read_f32(&got),
                &read_f32(&ref_r),
                1e-5,
                1e-5,
                &format!("{name} {tag}"),
            );
            let b = read_bits(&got);
            for (r, &p) in pos.iter().enumerate() {
                assert_eq!(
                    read_bits(&fast_rope(&slice_axis(&x, false, r, r + 1), &freqs, &[p]).unwrap()),
                    &b[r * stride..(r + 1) * stride]
                );
            }
        }
        for (tag, xt) in [
            ("t2D", arr(&rng.vec_f32(n * td, 3.0), &[n, td])),
            ("t3D", arr(&rng.vec_f32(n * h * td, 3.0), &[n, h, td])),
        ] {
            let got = fast_partial_rope_tail(&xt, &freqs, &pos).unwrap();
            let ref_t = apply_partial_rope(&xt, rd as i64, &cos, &sin, false).unwrap();
            assert_tol_f32(
                &read_f32(&got),
                &read_f32(&ref_t),
                1e-5,
                1e-5,
                &format!("{name} {tag}"),
            );
        }
    }
    assert!(
        fast_rope(
            &arr(&vec![1.0; rd], &[1, rd]),
            &freqs,
            &[u64::from(u32::MAX)]
        )
        .is_err()
    );

    // 3. rms_norm & rms_norm_unweighted.
    let (eps, rows, dim) = (1e-5f32, 5usize, 64usize);
    let xv = rng.vec_f32(rows * dim, 3.0);
    let wv: Vec<f32> = (0..dim).map(|i| 0.7 + (i as f32) * 0.02).collect();
    let (x, w) = (arr(&xv, &[rows, dim]), arr(&wv, &[dim]));
    let (mut want_u, mut want_w) = (
        Vec::with_capacity(rows * dim),
        Vec::with_capacity(rows * dim),
    );
    for row in xv.chunks_exact(dim) {
        let inv_r = 1.0
            / (row
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                / dim as f64
                + f64::from(eps))
            .sqrt();
        for (c, &v) in row.iter().enumerate() {
            want_u.push(f64::from(v) * inv_r);
            want_w.push(f64::from(v) * inv_r * f64::from(wv[c]));
        }
    }
    assert_oracle(
        &read_f32(&rms_norm(&x, &w, eps).unwrap()),
        &want_w,
        "rms_norm",
    );
    assert_oracle(
        &read_f32(&rms_norm_unweighted(&x, eps).unwrap()),
        &want_u,
        "rms_unw",
    );
    let (xb, wb) = (as_dt(&x, DType::BFloat16), as_dt(&w, DType::BFloat16));
    assert_eq!(
        rms_norm(&xb, &wb, eps).unwrap().dtype().unwrap(),
        DType::BFloat16
    );
    assert_eq!(
        rms_norm_unweighted(&xb, eps).unwrap().dtype().unwrap(),
        DType::BFloat16
    );
    assert!(rms_norm(&x, &arr(&vec![1.0; 2 * dim], &[2, dim]), eps).is_err());

    // 4. Asymmetric SwiGLU clamp: g <= +lim ONLY (NEVER lower-clamped at -lim).
    let limit = 7.0f32;
    let gs = [-20.0f32, -8.5, -7.0, -2.0, 0.0, 2.5, 7.0, 9.5, 25.0];
    let us = [-15.0f32, -7.5, -7.0, -3.0, 0.0, 3.0, 7.0, 8.0, 18.0];
    let (mut ga_v, mut ua_v) = (Vec::new(), Vec::new());
    for &g in &gs {
        for &u in &us {
            ga_v.push(g);
            ua_v.push(u);
        }
    }
    let (ga, ua) = (arr(&ga_v, &[1, ga_v.len()]), arr(&ua_v, &[1, ua_v.len()]));
    let sw_want: Vec<f64> = ga_v
        .iter()
        .zip(&ua_v)
        .map(|(&g, &u)| {
            let (gc, uc) = (f64::from(g.min(limit)), f64::from(u.clamp(-limit, limit)));
            (1.0 / (1.0 + (-gc).exp())) * gc * uc
        })
        .collect();
    assert_oracle(
        &read_f32(&Activations::swiglu_clamped_compiled(&ga, &ua, limit).unwrap()),
        &sw_want,
        "swiglu_clamped",
    );
    let neg_v = read_f32(
        &Activations::swiglu_clamped_compiled(&arr(&[-20.0], &[1]), &arr(&[3.0], &[1]), limit)
            .unwrap(),
    )[0];
    assert!(neg_v.abs() < 1e-4, "g=-20 lower-clamped: {neg_v}");
    let unclamped = read_bits(&Activations::swiglu_compiled(&ga, &ua).unwrap());
    for l in [0.0f32, -1.0] {
        assert_eq!(
            read_bits(&Activations::swiglu_clamped_compiled(&ga, &ua, l).unwrap()),
            unclamped
        );
    }
    for dt in [DType::Float32, DType::Float16, DType::BFloat16] {
        let out = Activations::swiglu_clamped_compiled(
            &as_dt(&ga.reshape(&[9, 9]).unwrap(), dt),
            &as_dt(&ua.reshape(&[9, 9]).unwrap(), dt),
            limit,
        )
        .unwrap();
        assert_eq!((out.dtype().unwrap(), shape(&out)), (dt, vec![9, 9]));
    }
}

/// Unified harness for base HC kernels, fused HC bridge suite (incl. lazy broadcast `res`),
/// and prefill `ROW2` (M=33 vs M=64/65).
struct HcRig {
    hc: usize,
    d: usize,
    norm_eps: f32,
    cfg: HcConfig,
    w_mix: HcWeights,
    w_head: HcWeights,
    post_nw: MxArray,
    next_nw: MxArray,
    x: MxArray,
    y: MxArray,
    post_prev: MxArray,
}

impl HcRig {
    fn new(d: usize, max_m: usize, norm_eps: f32, hc_eps: f32, seed: u64) -> Self {
        let mut rng = Rng::new(seed + d as u64);
        let (hc, hd) = (4usize, 4 * d);
        let sw = 1.0 / (hd as f32).sqrt();
        Self {
            hc,
            d,
            norm_eps,
            cfg: HcConfig {
                count: hc,
                eps: hc_eps,
                magnitude: 2.0,
            },
            w_mix: HcWeights {
                weight: arr(&rng.vec_f32(2 * hc * hd, sw), &[2 * hc, hd]),
                scale: arr(&[0.75, 1.25], &[2]),
                base: arr(&rng.vec_f32(2 * hc, 0.5), &[2 * hc]),
            },
            w_head: HcWeights {
                weight: arr(&rng.vec_f32(hc * hd, sw), &[hc, hd]),
                scale: arr(&[0.85], &[1]),
                base: arr(&rng.vec_f32(hc, 0.5), &[hc]),
            },
            post_nw: norm_w(&mut rng, &[d]),
            next_nw: norm_w(&mut rng, &[d]),
            x: arr(&rng.vec_f32(max_m * hd, 1.5), &[max_m, hc, d]),
            y: arr(&rng.vec_f32(max_m * d, 1.5), &[max_m, d]),
            post_prev: norm_w(&mut rng, &[max_m, hc]),
        }
    }
    fn slice(&self, lo: usize, hi: usize) -> (MxArray, MxArray, MxArray) {
        (
            slice_axis(&self.x, false, lo, hi),
            slice_axis(&self.y, false, lo, hi),
            slice_axis(&self.post_prev, false, lo, hi),
        )
    }
    fn mix_red(&self, x: &MxArray) -> (MxArray, MxArray) {
        hc_mix_reduce(x, &self.w_mix, self.norm_eps, &self.cfg).unwrap()
    }
    fn head(&self, x: &MxArray) -> MxArray {
        hc_head(x, &self.w_head, self.norm_eps, self.cfg.eps).unwrap()
    }
    fn mrn(&self, x: &MxArray) -> (MxArray, MxArray) {
        hc_mix_reduce_norm(x, &self.w_mix, &self.next_nw, self.norm_eps, &self.cfg).unwrap()
    }
    fn hn(&self, x: &MxArray) -> MxArray {
        hc_head_norm(x, &self.w_head, &self.next_nw, self.norm_eps, self.cfg.eps).unwrap()
    }
    fn pnhp(&self, y: &MxArray, x: &MxArray, p: &MxArray) -> MxArray {
        post_norm_hc_post(y, x, p, &self.post_nw, self.norm_eps).unwrap()
    }
    fn bridge(
        &self,
        y: &MxArray,
        x: &MxArray,
        p: &MxArray,
        norm: bool,
    ) -> (MxArray, MxArray, MxArray) {
        hc_bridge(
            y,
            x,
            p,
            &self.post_nw,
            &self.w_mix,
            norm.then_some(&self.next_nw),
            self.norm_eps,
            &self.cfg,
        )
        .unwrap()
    }
    fn bhead(&self, y: &MxArray, x: &MxArray, p: &MxArray, norm: bool) -> MxArray {
        hc_bridge_head(
            y,
            x,
            p,
            &self.post_nw,
            &self.w_head,
            norm.then_some(&self.next_nw),
            self.norm_eps,
            self.cfg.eps,
        )
        .unwrap()
    }
    fn rn(&self, a: &MxArray) -> MxArray {
        rms_norm(a, &self.next_nw, self.norm_eps).unwrap()
    }

    fn check_base(&self, ms: &[usize], sample: &[usize]) {
        let mut bases: [Vec<(usize, Vec<u32>)>; 4] = Default::default();
        for &r in sample {
            let (xr, yr, _) = self.slice(r, r + 1);
            let (h1, p1) = self.mix_red(&xr);
            let post = hc_post_identity(&yr, &xr, &p1).unwrap();
            for (i, a) in [h1, p1, self.head(&xr), post].into_iter().enumerate() {
                bases[i].push((r, read_bits(&a)));
            }
        }
        for &m in ms {
            let (xm, ym, _) = self.slice(0, m);
            let (hm, pm) = self.mix_red(&xm);
            let (headm, outm) = (self.head(&xm), hc_post_identity(&ym, &xm, &pm).unwrap());
            for (i, a) in [&hm, &pm, &headm, &outm].into_iter().enumerate() {
                assert_sampled_inv(
                    &bases[i],
                    &read_bits(a),
                    m,
                    &format!("hc_base[{i}] d={}", self.d),
                );
            }
            if m <= 17 {
                let rg = hc_mix(&xm, &self.w_mix, self.norm_eps, &self.cfg).unwrap();
                let rhp = hc_head_pre(&xm, &self.w_head, self.norm_eps, self.cfg.eps).unwrap();
                let pb = pm.reshape(&[m as i64, self.hc as i64, 1]).unwrap();
                let yb = ym.reshape(&[m as i64, 1, self.d as i64]).unwrap();
                let rout = xm.add(&pb.mul(&yb).unwrap()).unwrap();
                assert_err(&hm, &hc_reduce(&xm, &rg.pre).unwrap(), 1e-5, "mix_reduce h");
                assert_err(&pm, &rg.post, 1e-5, "mix_reduce p");
                assert_err(&headm, &hc_reduce(&xm, &rhp).unwrap(), 1e-5, "hc_head");
                assert_err(&outm, &rout, 1e-5, "hc_post");
            }
        }
    }

    fn check_fused_suite(&self, ms: &[usize], sample: &[usize]) {
        let mut bases: [Vec<(usize, Vec<u32>)>; 9] = Default::default();
        for &r in sample {
            let (xr, yr, pr) = self.slice(r, r + 1);
            let ((mh, mp), (bx, bh, bp), (_, sh, _)) = (
                self.mrn(&xr),
                self.bridge(&yr, &xr, &pr, true),
                self.bridge(&yr, &xr, &pr, false),
            );
            let arrs = [
                mh,
                mp,
                self.hn(&xr),
                self.pnhp(&yr, &xr, &pr),
                bx,
                bh,
                bp,
                sh,
                self.bhead(&yr, &xr, &pr, true),
            ];
            for (i, a) in arrs.into_iter().enumerate() {
                bases[i].push((r, read_bits(&a)));
            }
        }
        for &m in ms {
            let (xm, ym, pm) = self.slice(0, m);
            let ((mh, mp), hn, pnhp) = (self.mrn(&xm), self.hn(&xm), self.pnhp(&ym, &xm, &pm));
            let ((bx, bh, bp), (sx, sh, sp)) = (
                self.bridge(&ym, &xm, &pm, true),
                self.bridge(&ym, &xm, &pm, false),
            );
            let (bhead, bhead_s) = (
                self.bhead(&ym, &xm, &pm, true),
                self.bhead(&ym, &xm, &pm, false),
            );
            let (rh, rp) = self.mix_red(&xm);
            let rx = hc_post_identity(
                &rms_norm(&ym, &self.post_nw, self.norm_eps).unwrap(),
                &xm,
                &pm,
            )
            .unwrap();
            let ((rbh_raw, rbp), rhead_raw) = (self.mix_red(&rx), self.head(&rx));
            let (rn_rh, rn_hx, rn_rbh, rn_rhx) = (
                self.rn(&rh),
                self.rn(&self.head(&xm)),
                self.rn(&rbh_raw),
                self.rn(&rhead_raw),
            );
            for (tag, g, w) in [
                ("mh", &mh, &rn_rh),
                ("mp", &mp, &rp),
                ("hn", &hn, &rn_hx),
                ("pnhp", &pnhp, &rx),
                ("bx", &bx, &rx),
                ("bh", &bh, &rn_rbh),
                ("bp", &bp, &rbp),
                ("sx", &sx, &rx),
                ("sh", &sh, &rbh_raw),
                ("sp", &sp, &rbp),
                ("bhead", &bhead, &rn_rhx),
                ("bhead_s", &bhead_s, &rhead_raw),
            ] {
                assert_err(g, w, 1e-5, &format!("hc_suite {tag} d={} M={m}", self.d));
            }
            for (i, a) in [&mh, &mp, &hn, &pnhp, &bx, &bh, &bp, &sh, &bhead]
                .into_iter()
                .enumerate()
            {
                assert_sampled_inv(
                    &bases[i],
                    &read_bits(a),
                    m,
                    &format!("hc_suite[{i}] d={}", self.d),
                );
            }
            // Lazy Broadcast `res` (Layer-0 embedding fanout).
            let res_b = ym
                .reshape(&[m as i64, 1, self.d as i64])
                .unwrap()
                .broadcast_to(&[m as i64, self.hc as i64, self.d as i64])
                .unwrap();
            let (bxb, bhb, bpb) = self.bridge(&ym, &res_b, &pm, true);
            let (bxd, bhd, bpd) = self.bridge(&ym, &res_b.deep_copy().unwrap(), &pm, true);
            assert_eq!(
                (read_bits(&bxb), read_bits(&bhb), read_bits(&bpb)),
                (read_bits(&bxd), read_bits(&bhd), read_bits(&bpd))
            );
        }
    }

    fn check_prefill_row2(&self) {
        let (x33, y33, p33) = self.slice(0, 33);
        let (bx33, bh33, bp33) = self.bridge(&y33, &x33, &p33, true);
        let bhead33 = self.bhead(&y33, &x33, &p33, true);
        let ref33 = [
            read_bits(&bx33),
            read_bits(&bh33),
            read_bits(&bp33),
            read_bits(&bhead33),
        ];
        for &m in &[64usize, 65] {
            let (xm, ym, pm) = self.slice(0, m);
            let ((bx, bh, bp), (_, sh, _), bhead) = (
                self.bridge(&ym, &xm, &pm, true),
                self.bridge(&ym, &xm, &pm, false),
                self.bhead(&ym, &xm, &pm, true),
            );
            let rx = self.pnhp(&ym, &xm, &pm);
            let (rh_raw, rp) = self.mix_red(&rx);
            let (rn_rh, rn_hd) = (self.rn(&rh_raw), self.rn(&self.head(&rx)));
            for (tag, g, w) in [
                ("x", &bx, &rx),
                ("h", &bh, &rn_rh),
                ("p", &bp, &rp),
                ("skip", &sh, &rh_raw),
                ("head", &bhead, &rn_hd),
            ] {
                assert_err(g, w, 1e-5, &format!("ROW2 {tag} M={m}"));
            }
            for (got, want) in [&bx, &bh, &bp, &bhead].into_iter().zip(&ref33) {
                assert_eq!(&read_bits(got)[..want.len()], want.as_slice());
            }
        }
    }
}

#[test]
fn hc_fused_suite_and_prefill_row2_parity_and_invariance() {
    crate::models::hy_v4_flash::exact_f32();
    for &d in &[64usize, 66, 3072] {
        HcRig::new(d, 256, 1e-6, 1e-4, 101).check_base(&[1, 2, 4, 17, 256], &[0, 1, 3, 16, 255]);
        HcRig::new(d, 17, 1e-5, 1e-6, 2101).check_fused_suite(&[1, 2, 4, 17], &[0, 1, 3, 16]);
    }
    HcRig::new(3072, 65, 1e-5, 1e-6, 9401).check_prefill_row2();
}

#[test]
fn bf16_lm_head_and_f32_head_weights_row_invariant() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(701);
    for &v in &[32usize, 37, 4096] {
        for &k in &[64usize, 192, 3072] {
            let wb: Vec<u16> = rng
                .vec_f32(v * k, 1.0 / (k as f32).sqrt())
                .into_iter()
                .map(|val| half::bf16::from_f32(val).to_bits())
                .collect();
            let w_bf16 = MxArray::from_bfloat16(&wb, &[v as i64, k as i64]).unwrap();
            let (head, wt) = (
                Dense::Array(w_bf16.clone()),
                as_dt(&w_bf16, DType::Float32)
                    .transpose(Some(&[1, 0]))
                    .unwrap(),
            );
            let x_all = arr(&rng.vec_f32(16 * k, 1.0), &[16, k]);
            assert_row_invariant(
                |r| head_logits(&slice_axis(&x_all, false, r, r + 1), &head).unwrap(),
                |m| {
                    let xm = slice_axis(&x_all, false, 0, m);
                    let got = head_logits(&xm, &head).unwrap();
                    assert_eq!(
                        (shape(&got), got.dtype().unwrap()),
                        (vec![m as i64, v as i64], DType::Float32)
                    );
                    assert_err(
                        &got,
                        &xm.matmul(&wt).unwrap(),
                        1e-4,
                        &format!("bf16_head V={v} K={k} M={m}"),
                    );
                    got
                },
                &[1, 2, 4, 7, 16],
                &format!("bf16_lm_head ({v},{k})"),
            );
        }
    }

    for &(n, k) in &[(16usize, 3072usize), (2, 12), (32, 128)] {
        let scale = ((n * 128) as f32).sqrt().recip();
        let w = Dense::Array(arr(&rng.vec_f32(n * k, 0.25), &[n, k]));
        let x_all = arr(&rng.vec_f32(16 * k, 1.0), &[16, k]);
        if try_f32_head_weights(&slice_axis(&x_all, false, 0, 1), &w, scale)
            .unwrap()
            .is_none()
        {
            return;
        }
        assert_row_invariant(
            |r| {
                try_f32_head_weights(&slice_axis(&x_all, false, r, r + 1), &w, scale)
                    .unwrap()
                    .unwrap()
            },
            |m| {
                let xm = slice_axis(&x_all, false, 0, m);
                let got = try_f32_head_weights(&xm, &w, scale).unwrap().unwrap();
                let want = linear(&xm, &w)
                    .unwrap()
                    .mul_scalar(f64::from(scale))
                    .unwrap();
                assert_err(&got, &want, 1e-4, &format!("head_w ({n},{k}) M={m}"));
                got
            },
            &[1, 2, 4, 8, 16],
            &format!("f32_head_weights ({n},{k})"),
        );
    }
}

fn sort_routes_asc(ids: &[u32], w: &[f32], rows: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    let (mut oi, mut ow) = (Vec::with_capacity(rows * k), Vec::with_capacity(rows * k));
    for r in 0..rows {
        let mut pairs: Vec<(u32, f32)> = ids[r * k..(r + 1) * k]
            .iter()
            .copied()
            .zip(w[r * k..(r + 1) * k].iter().copied())
            .collect();
        pairs.sort_by_key(|p| p.0);
        oi.extend(pairs.iter().map(|p| p.0));
        ow.extend(pairs.iter().map(|p| p.1));
    }
    (oi, ow)
}

#[test]
fn router_topk_gpu_matches_route_select_fast_and_is_row_invariant() {
    crate::models::hy_v4_flash::exact_f32();
    let (e, k) = (512usize, 16usize);
    let bias_v = Rng::new(1201).vec_f32(e, 0.25);
    let bias_arr = arr(&bias_v, &[e]);
    let mut lv = Rng::new(1202).vec_f32(256 * e, 4.0);
    for &r in &[0usize, 1, 3, 15, 255] {
        lv[r * e + 50..r * e + 74].fill(-0.625);
        lv[r * e + 200..r * e + 220].fill(2.25);
        (lv[r * e + 300], lv[r * e + 301]) = (-40.0, 0.0);
    }
    let logits_all = arr(&lv, &[256, e]);

    for &(has_b, norm, scale) in &[
        (true, true, 2.5f32),
        (false, true, 1.0),
        (true, false, 0.75),
    ] {
        let cfg = moe_cfg(e, k, scale, norm);
        let (bo, bs) = (
            has_b.then_some(&bias_arr),
            has_b.then_some(bias_v.as_slice()),
        );
        let (mut s_ids, mut s_w) = (Vec::with_capacity(16), Vec::with_capacity(16));
        for r in 0..16 {
            let Some((gi, gw)) =
                router_topk_gpu(&slice_axis(&logits_all, false, r, r + 1), bo, &cfg).unwrap()
            else {
                return;
            };
            s_ids.push(read_u32(&gi));
            s_w.push(read_bits(&gw));
        }
        for &m in &[1usize, 2, 4, 16, 256] {
            let lm = slice_axis(&logits_all, false, 0, m);
            let (gi, gw) = router_topk_gpu(&lm, bo, &cfg).unwrap().unwrap();
            let sh = vec![m as i64, k as i64];
            assert_eq!(
                (shape(&gi), gi.dtype().unwrap()),
                (sh.clone(), DType::Uint32)
            );
            assert_eq!((shape(&gw), gw.dtype().unwrap()), (sh, DType::Float32));
            let (got_i, got_w) = (read_u32(&gi), read_f32(&gw));
            let (ri, rw) = route_select_fast(
                &read_f32(&Activations::sigmoid(&lm).unwrap()),
                bs,
                e,
                k,
                scale,
                norm,
            );
            let (wi, ww) = sort_routes_asc(&ri, &rw, m, k);
            assert_eq!(got_i, wi);
            assert!(max_abs_err(&got_w, &ww) <= 1e-6);
            for r in 0..m.min(16) {
                assert_eq!(&got_i[r * k..(r + 1) * k], s_ids[r].as_slice());
                assert_eq!(f32_bits(&got_w[r * k..(r + 1) * k]), s_w[r]);
            }
        }
    }

    // E=1024 + NaN safety.
    let mut big = Rng::new(1299).vec_f32(2048, 3.0);
    (big[7], big[1537]) = (f32::NAN, f32::NAN);
    if let Some((gi, gw)) =
        router_topk_gpu(&arr(&big, &[2, 1024]), None, &moe_cfg(1024, 16, 1.0, true)).unwrap()
    {
        let ids = read_u32(&gi);
        assert!(
            read_f32(&gw).iter().all(|v| v.is_finite())
                && !ids[..16].contains(&7)
                && !ids[16..].contains(&513)
        );
        assert!(
            ids.as_chunks::<16>()
                .0
                .iter()
                .all(|row| row.windows(2).all(|w| w[0] < w[1]))
        );
    }

    // Edge (E, K) values with global + 24-way multi-SG ties vs CPU route_host bit-exact.
    for &(ev, kv) in &[
        (16usize, 16usize),
        (32, 1),
        (500, 16),
        (512, 16),
        (512, 24),
        (1000, 32),
        (1024, 16),
    ] {
        let mut lv = Rng::new(1300 + (ev * 31 + kv) as u64).vec_f32(4 * ev, 3.0);
        lv[ev..2 * ev].fill(1.25);
        for t in 0..24.min(ev) {
            lv[2 * ev + (t * 37) % ev] = 10.0;
        }
        let (la, bv) = (
            arr(&lv, &[4, ev]),
            Rng::new(1400 + ev as u64).vec_f32(ev, 0.25),
        );
        let (ba, cfg) = (arr(&bv, &[ev]), moe_cfg(ev, kv, 2.5, true));
        for use_b in [false, true] {
            let Some((gi, gw)) = router_topk_gpu(&la, use_b.then_some(&ba), &cfg).unwrap() else {
                return;
            };
            let (ri, rw) = route_host(
                &read_f32(&Activations::sigmoid(&la).unwrap()),
                use_b.then_some(bv.as_slice()),
                &cfg,
            )
            .unwrap();
            let (wi, ww) = sort_routes_asc(&ri, &rw, 4, kv);
            assert_eq!((read_u32(&gi), read_bits(&gw)), (wi, f32_bits(&ww)));
        }
    }
}

#[test]
fn mla_prep_q_kv_and_q8_grouped_k192_rope_row_invariant() {
    crate::models::hy_v4_flash::exact_f32();
    let (eps, pos_u64, pos_i32) = (
        1e-5f32,
        [0u64, 17, 4095, 262_144],
        [0i32, 17, 4095, 262_144],
    );
    let mut rng = Rng::new(3101);

    // 1. try_mla_prep_q_kv (F32, F16 unsplit, F16 split, f32/i32 pos).
    for &(ql, kvl, rd, gw) in &[(1024usize, 512usize, 64usize, 4096usize), (6, 6, 4, 10)] {
        let (kvr, total) = (kvl + rd, ql + kvl + rd + gw);
        let inv = rope_inv_freqs(&inv_freqs(rd, 10_000.0, None)).unwrap();
        let (q_nw, kv_nw, all) = (
            norm_w(&mut rng, &[ql]),
            norm_w(&mut rng, &[kvl]),
            arr(&rng.vec_f32(4 * total, 1.5), &[4, total]),
        );
        let prep = |sub: &MxArray, pos: &MxArray, f16: bool, split: bool| {
            try_mla_prep_q_kv(
                sub, &q_nw, &kv_nw, pos, &inv, ql as i64, kvl as i64, rd as i64, eps, f16, split,
            )
            .unwrap()
        };
        let (mut s_qr, mut s_kv, mut s_k0, mut s_k1) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (r, &pv) in pos_u64.iter().enumerate() {
            let (row, p1) = (
                slice_axis(&all, false, r, r + 1),
                rope_positions_f32(&[pv]).unwrap(),
            );
            let Some((qr1, kv1, _)) = prep(&row, &p1, true, false) else {
                return;
            };
            let (qr1s, k0, k1) = prep(&row, &p1, true, true).unwrap();
            assert_eq!(read_bits(&qr1), read_bits(&qr1s));
            s_qr.push(read_bits(&qr1));
            s_kv.push(kv1.to_uint16_native().unwrap());
            s_k0.push(k0.to_uint16_native().unwrap());
            s_k1.push(k1.unwrap().to_uint16_native().unwrap());
        }
        for &m in &[1usize, 2, 4] {
            let (sub, pf, pi) = (
                slice_axis(&all, false, 0, m),
                rope_positions_f32(&pos_u64[..m]).unwrap(),
                i32_arr(&pos_i32[..m]),
            );
            let r_qr = rms_norm(&slice_axis(&sub, true, 0, ql), &q_nw, eps).unwrap();
            let r_kvn = rms_norm(&slice_axis(&sub, true, ql, ql + kvl), &kv_nw, eps).unwrap();
            let r_kpe =
                fast_rope_with_pos(&slice_axis(&sub, true, ql + kvl, ql + kvr), &inv, &pf).unwrap();
            let r_kv = MxArray::concatenate(&r_kvn, &r_kpe, -1).unwrap();
            let r_kv16 = as_dt(&r_kv, DType::Float16);
            for pin in [&pf, &pi] {
                let (qr, kv32, none1) = prep(&sub, pin, false, false).unwrap();
                assert!(none1.is_none() && kv32.dtype().unwrap() == DType::Float32);
                assert_err(&qr, &r_qr, 1e-5, "mla_prep qr");
                assert_err(&kv32, &r_kv, 1e-5, "mla_prep kv32");
                let (qrh, kvh, _) = prep(&sub, pin, true, false).unwrap();
                assert_eq!(kvh.dtype().unwrap(), DType::Float16);
                assert_err(&kvh, &r_kv16, 1e-5, "mla_prep kv16");
                let (qrs, k0, k1_opt) = prep(&sub, pin, true, true).unwrap();
                let (k1, half) = (k1_opt.unwrap(), kvr / 2);
                assert_eq!(
                    (k0.dtype().unwrap(), k1.dtype().unwrap()),
                    (DType::Float16, DType::Float16)
                );
                assert_eq!(
                    MxArray::concatenate(&k0, &k1, -1)
                        .unwrap()
                        .to_uint16_native()
                        .unwrap(),
                    kvh.to_uint16_native().unwrap()
                );
                let (qrs_b, qrh_b, kv_b, k0_b, k1_b) = (
                    read_bits(&qrs),
                    read_bits(&qrh),
                    kvh.to_uint16_native().unwrap(),
                    k0.to_uint16_native().unwrap(),
                    k1.to_uint16_native().unwrap(),
                );
                for r in 0..m {
                    assert_eq!(&qrs_b[r * ql..(r + 1) * ql], s_qr[r].as_slice());
                    assert_eq!(&qrh_b[r * ql..(r + 1) * ql], s_qr[r].as_slice());
                    assert_eq!(&kv_b[r * kvr..(r + 1) * kvr], s_kv[r].as_slice());
                    assert_eq!(&k0_b[r * half..(r + 1) * half], s_k0[r].as_slice());
                    assert_eq!(&k1_b[r * half..(r + 1) * half], s_k1[r].as_slice());
                }
            }
        }
    }

    // 2. try_q8_grouped_k192_rope (SymQ8 + Qmm, H=32/N=512 and H=2/N=8).
    let (nope, rd) = (192usize, 64usize);
    let inv = rope_inv_freqs(&inv_freqs(rd, 10_000.0, None)).unwrap();
    for &(heads, n_out) in &[(32usize, 512usize), (2, 8)] {
        let kb_sym = make_sym_q8(&mut rng, heads * n_out, nope, 0.2);
        let Dense::SymQ8 { w, scales } = &kb_sym else {
            unreachable!()
        };
        let sf = as_dt(scales, DType::Float32);
        let kb_qmm = Dense::Qmm {
            w: w.clone(),
            scales: sf.clone(),
            biases: sf.mul_scalar(-128.0).unwrap(),
        };
        let qb = arr(
            &rng.vec_f32(4 * heads * (nope + rd), 1.0),
            &[4, heads, nope + rd],
        );
        for (tag, kb) in [("sym_q8", &kb_sym), ("qmm_f32", &kb_qmm)] {
            let run = |x: &MxArray, p: &MxArray| {
                try_q8_grouped_k192_rope(x, kb, heads as i64, p, &inv).unwrap()
            };
            if run(
                &slice_axis(&qb, false, 0, 1),
                &rope_positions_f32(&[pos_u64[0]]).unwrap(),
            )
            .is_none()
            {
                return;
            }
            for use_i32 in [false, true] {
                assert_row_invariant(
                    |r| {
                        run(
                            &slice_axis(&qb, false, r, r + 1),
                            &rope_positions_f32(&[pos_u64[r]]).unwrap(),
                        )
                        .unwrap()
                    },
                    |m| {
                        let (sub, pf) = (
                            slice_axis(&qb, false, 0, m),
                            rope_positions_f32(&pos_u64[..m]).unwrap(),
                        );
                        let pin = if use_i32 {
                            i32_arr(&pos_i32[..m])
                        } else {
                            pf.clone()
                        };
                        let got = run(&sub, &pin).unwrap();
                        let r_abs =
                            grouped_mm_t_mgh(kb, &slice_axis(&sub, true, 0, nope), heads as i64)
                                .unwrap();
                        let r_pe =
                            fast_rope_with_pos(&slice_axis(&sub, true, nope, nope + rd), &inv, &pf)
                                .unwrap();
                        assert_err(
                            &got,
                            &MxArray::concatenate(&r_abs, &r_pe, -1).unwrap(),
                            1e-5,
                            &format!("k192 {tag} M={m}"),
                        );
                        got
                    },
                    &[1, 2, 4],
                    &format!("k192_rope {tag} H={heads} i32={use_i32}"),
                );
            }
        }
    }
}

#[test]
fn kpool_compress_and_partial_rope_tail_match_unfused() {
    crate::models::hy_v4_flash::exact_f32();
    let (eps, d, rd, tail_cap) = (1e-5f32, 128usize, 64usize, 10usize);
    let inv = rope_inv_freqs(&inv_freqs(rd, 1e7, None)).unwrap();
    let mut rng = Rng::new(3601);
    let (wv, wg, ape, cn) = (
        arr(&rng.vec_f32(d * d, 0.2), &[d, d]),
        arr(&rng.vec_f32(d * d, 0.2), &[d, d]),
        arr(&rng.vec_f32(4 * d, 0.5), &[4, d]),
        norm_w(&mut rng, &[d]),
    );
    let (ring_v, tok_v) = (rng.vec_f32(tail_cap * d, 1.0), rng.vec_f32(16 * d, 1.0));
    let (ring_a, tok_a) = (arr(&ring_v, &[tail_cap, d]), arr(&tok_v, &[16, d]));
    let call = |ng, fg, open, start, dt| {
        try_kpool_compress(
            (open > 0).then_some(&ring_a),
            &tok_a,
            &wv,
            &wg,
            &ape,
            &cn,
            &inv,
            ng,
            fg,
            open,
            start,
            tail_cap,
            eps,
            dt,
        )
        .unwrap()
    };

    for &(ng, fg, open, start) in &[
        (1usize, 0usize, 0usize, 0usize),
        (1, 17, 1, 9),
        (1, 1024, 2, 9),
        (1, 65_535, 3, 8),
        (2, 5, 3, 9),
        (4, 100, 2, 9),
    ] {
        let mut lin = Vec::with_capacity(ng * 4 * d);
        for t in 0..ng * 4 {
            let (src, idx) = if t < open {
                (&ring_v, (start + t) % tail_cap)
            } else {
                (&tok_v, t - open)
            };
            lin.extend_from_slice(&src[idx * d..(idx + 1) * d]);
        }
        let grp = arr(&lin, &[ng, 4, d]);
        let v = grp.matmul(&wv.transpose(Some(&[1, 0])).unwrap()).unwrap();
        let gt = grp
            .matmul(&wg.transpose(Some(&[1, 0])).unwrap())
            .unwrap()
            .add(&ape)
            .unwrap();
        let e = gt
            .sub(&gt.max(Some(&[1]), Some(true)).unwrap())
            .unwrap()
            .exp()
            .unwrap();
        let pool = v
            .mul(&e.div(&e.sum(Some(&[1]), Some(true)).unwrap()).unwrap())
            .unwrap()
            .sum(Some(&[1]), Some(false))
            .unwrap();
        let normed = rms_norm(&pool, &cn, eps).unwrap();
        let pos: Vec<u64> = (fg..fg + ng).map(|g| (g * 4) as u64).collect();
        let tail = fast_rope_with_pos(
            &slice_axis(&normed, true, d - rd, d),
            &inv,
            &rope_positions_f32(&pos).unwrap(),
        )
        .unwrap();
        let want = MxArray::concatenate(&slice_axis(&normed, true, 0, d - rd), &tail, -1).unwrap();

        let Some(got) = call(ng, fg, open, start, DType::Float32) else {
            return;
        };
        assert_eq!(shape(&got), [ng as i64, d as i64]);
        assert_err(&got, &want, 1e-5, &format!("kpool ng={ng} open={open}"));
        for dt in [DType::Float16, DType::BFloat16] {
            let gd = call(ng, fg, open, start, dt).unwrap();
            assert_eq!(
                (gd.dtype().unwrap(), read_bits(&gd)),
                (dt, read_bits(&as_dt(&got, dt)))
            );
        }
    }

    let (pos_all, x3) = (
        [0u64, 7, 1024, 262_144],
        arr(&rng.vec_f32(4 * 16 * 128, 1.5), &[4, 16, 128]),
    );
    assert_row_invariant(
        |r| {
            try_partial_rope_tail(
                &slice_axis(&x3, false, r, r + 1),
                &rope_positions_f32(&[pos_all[r]]).unwrap(),
                &inv,
            )
            .unwrap()
            .unwrap()
        },
        |m| {
            let (xm, pm) = (
                slice_axis(&x3, false, 0, m),
                rope_positions_f32(&pos_all[..m]).unwrap(),
            );
            let got = try_partial_rope_tail(&xm, &pm, &inv).unwrap().unwrap();
            let want = MxArray::concatenate(
                &slice_axis(&xm, true, 0, 64),
                &fast_rope_with_pos(&slice_axis(&xm, true, 64, 128), &inv, &pm).unwrap(),
                -1,
            )
            .unwrap();
            assert_err(&got, &want, 1e-5, &format!("partial_rope_tail M={m}"));
            got
        },
        &[1, 2, 4],
        "partial_rope_tail 3D",
    );

    for &(m, h) in &[(17usize, 32usize), (32, 64)] {
        let pm = rope_positions_f32(&(0..m).map(|i| 100 + i as u64).collect::<Vec<_>>()).unwrap();
        let q_raw = arr(&rng.vec_f32(m * h * 256, 1.5), &[m, h, 256]);
        let q32 = slice_axis(&q_raw, true, 192, 256);
        assert_err(
            &try_partial_rope_tail_out(&q32, &pm, &inv, false)
                .unwrap()
                .unwrap(),
            &fast_rope_with_pos(&q32, &inv, &pm).unwrap(),
            1e-5,
            "strided q_pe F32",
        );
        let q16 = slice_axis(&as_dt(&q_raw, DType::Float16), true, 192, 256);
        let g16 = try_partial_rope_tail_out(&q16, &pm, &inv, true)
            .unwrap()
            .unwrap();
        assert_eq!(g16.dtype().unwrap(), DType::Float16);
        assert_err(
            &g16,
            &as_dt(
                &fast_rope_with_pos(&as_dt(&q16, DType::Float32), &inv, &pm).unwrap(),
                DType::Float16,
            ),
            2e-3,
            "strided q_pe F16",
        );
    }
}

fn engram_gate_add_ref(
    xm: &MxArray,
    kvm: &MxArray,
    w: &[&MxArray; 4],
    (m, hc, d): (i64, i64, i64),
) -> MxArray {
    let [kv_b, gw, pg, hcs] = *w;
    let kv = kvm.reshape(&[m, hc + 1, d]).unwrap().add(kv_b).unwrap();
    let (key, val) = (
        kv.slice(&[0, 0, 0], &[m, hc, d]).unwrap(),
        kv.slice(&[0, hc, 0], &[m, hc + 1, d])
            .unwrap()
            .reshape(&[m, d])
            .unwrap(),
    );
    let sc = stream_gate(&stream_dot(xm, &key, gw).unwrap())
        .unwrap()
        .reshape(&[m, hc, 1])
        .unwrap()
        .add(hcs)
        .unwrap();
    xm.add(
        &signal(&val, pg)
            .unwrap()
            .reshape(&[m, 1, d])
            .unwrap()
            .mul(&sc)
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn engram_fused_norm_and_gate_add_match_unfused_and_are_row_invariant() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(3701);

    for &(hc, d, cols, hd) in &[
        (4usize, 3072usize, 16usize, 96usize),
        (4, 8, 2, 4),
        (4, 66, 4, 12),
    ] {
        let (ln, kv_b, gw, pg, hcs) = (
            norm_w(&mut rng, &[cols, hd]),
            arr(&rng.vec_f32((hc + 1) * d, 0.25), &[hc + 1, d]),
            arr(&rng.vec_f32(hc * d, 0.5), &[hc, d]),
            norm_w(&mut rng, &[d]),
            arr(&rng.vec_f32(hc * d, 0.5), &[hc, d]),
        );
        let (rows4, x4, kv4) = (
            arr(&rng.vec_f32(4 * cols * hd, 1.5), &[4, cols, hd]),
            arr(&rng.vec_f32(4 * hc * d, 1.5), &[4, hc, d]),
            arr(&rng.vec_f32(4 * (hc + 1) * d, 1.2), &[4, (hc + 1) * d]),
        );
        if try_engram_norm_rows(&slice_axis(&rows4, false, 0, 1), &ln, ENGRAM_EPS)
            .unwrap()
            .is_none()
        {
            return;
        }
        assert_row_invariant(
            |r| {
                try_engram_norm_rows(&slice_axis(&rows4, false, r, r + 1), &ln, ENGRAM_EPS)
                    .unwrap()
                    .unwrap()
            },
            |m| {
                let rm = slice_axis(&rows4, false, 0, m);
                let got = try_engram_norm_rows(&rm, &ln, ENGRAM_EPS).unwrap().unwrap();
                let want = rms_norm_unweighted(&rm, ENGRAM_EPS)
                    .unwrap()
                    .mul(&ln)
                    .unwrap()
                    .reshape(&[m as i64, (cols * hd) as i64])
                    .unwrap();
                assert_err(&got, &want, 1e-5, &format!("norm_rows d={d} M={m}"));
                got
            },
            &[1, 2, 4],
            &format!("engram_norm_rows d={d}"),
        );

        let gate_add = |x: &MxArray, kv: &MxArray, mask: Option<&MxArray>| {
            try_engram_gate_add(x, kv, &kv_b, &gw, &pg, &hcs, mask, ENGRAM_EPS)
                .unwrap()
                .unwrap()
        };
        assert_row_invariant(
            |r| {
                gate_add(
                    &slice_axis(&x4, false, r, r + 1),
                    &slice_axis(&kv4, false, r, r + 1),
                    None,
                )
            },
            |m| {
                let (xm, kvm) = (slice_axis(&x4, false, 0, m), slice_axis(&kv4, false, 0, m));
                let got = gate_add(&xm, &kvm, None);
                assert_err(
                    &got,
                    &engram_gate_add_ref(
                        &xm,
                        &kvm,
                        &[&kv_b, &gw, &pg, &hcs],
                        (m as i64, hc as i64, d as i64),
                    ),
                    1e-5,
                    &format!("gate_add d={d} M={m}"),
                );
                let mv: Vec<u8> = (0..m).map(|r| u8::from(r % 2 == 0)).collect();
                let mb = read_bits(&gate_add(
                    &xm,
                    &kvm,
                    Some(&MxArray::from_uint8(&mv, &[m as i64]).unwrap()),
                ));
                let (xb, gb, st) = (read_bits(&xm), read_bits(&got), hc * d);
                for (r, &keep) in mv.iter().enumerate() {
                    let want = if keep == 0 {
                        &xb[r * st..(r + 1) * st]
                    } else {
                        &gb[r * st..(r + 1) * st]
                    };
                    assert_eq!(&mb[r * st..(r + 1) * st], want);
                }
                got
            },
            &[1, 2, 4],
            &format!("engram_gate_add d={d}"),
        );
    }

    // M=512, HC=4, D=3072 repeated launch race stress (with negative gate_w).
    let (hc, d, m) = (4usize, 3072usize, 512usize);
    let kv_b = arr(&rng.vec_f32((hc + 1) * d, 0.25), &[hc + 1, d]);
    let gw = arr(
        &rng.vec_f32(hc * d, 1.0)
            .into_iter()
            .map(|v| -v.abs() - 0.5)
            .collect::<Vec<_>>(),
        &[hc, d],
    );
    let (pg, hcs) = (
        norm_w(&mut rng, &[d]),
        arr(&rng.vec_f32(hc * d, 0.5), &[hc, d]),
    );
    let (xm, kvm) = (
        arr(&rng.vec_f32(m * hc * d, 1.5), &[m, hc, d]),
        arr(&rng.vec_f32(m * (hc + 1) * d, 1.2), &[m, (hc + 1) * d]),
    );
    let ref_out = engram_gate_add_ref(
        &xm,
        &kvm,
        &[&kv_b, &gw, &pg, &hcs],
        (m as i64, hc as i64, d as i64),
    );
    let mut first: Option<Vec<u32>> = None;
    for rep in 0..8 {
        let got = try_engram_gate_add(&xm, &kvm, &kv_b, &gw, &pg, &hcs, None, ENGRAM_EPS)
            .unwrap()
            .unwrap();
        assert_err(&got, &ref_out, 1e-5, &format!("stress rep={rep}"));
        let b = read_bits(&got);
        if let Some(prev) = &first {
            assert_eq!(b, *prev, "stress rep={rep} diverged");
        } else {
            first = Some(b);
        }
    }
}

#[test]
fn mtp_fused_norms_match_unfused_and_are_row_invariant() {
    crate::models::hy_v4_flash::exact_f32();
    let (eps, mut rng) = (1e-5f32, Rng::new(9301));
    for &d in &[3072usize, 192] {
        let (enorm, hnorm) = (norm_w(&mut rng, &[d]), norm_w(&mut rng, &[d]));
        let (e_all, h_all) = (
            arr(&rng.vec_f32(17 * d, 1.5), &[17, d]),
            arr(&rng.vec_f32(17 * d, 1.5), &[17, d]),
        );
        let eh =
            |e: &MxArray, h: &MxArray| try_mtp_eh_norm_concat(e, h, &enorm, &hnorm, eps).unwrap();
        if eh(
            &slice_axis(&e_all, false, 0, 1),
            &slice_axis(&h_all, false, 0, 1),
        )
        .is_none()
        {
            return;
        }
        for dt in [DType::Float32, DType::Float16, DType::BFloat16] {
            let (ed, hd) = (as_dt(&e_all, dt), as_dt(&h_all, dt));
            let r1: Vec<(usize, Vec<u32>)> = (0..4)
                .map(|r| {
                    (
                        r,
                        read_bits(
                            &eh(
                                &slice_axis(&ed, false, r, r + 1),
                                &slice_axis(&hd, false, r, r + 1),
                            )
                            .unwrap(),
                        ),
                    )
                })
                .collect();
            for &m in &[1usize, 2, 4, 17] {
                let (em, hm) = (slice_axis(&ed, false, 0, m), slice_axis(&hd, false, 0, m));
                let got = eh(&em, &hm).unwrap();
                let want = MxArray::concatenate(
                    &rms_norm(&as_dt(&em, DType::Float32), &enorm, eps).unwrap(),
                    &rms_norm(&as_dt(&hm, DType::Float32), &hnorm, eps).unwrap(),
                    1,
                )
                .unwrap();
                assert_err(&got, &want, 1e-5, &format!("eh d={d} {dt:?} M={m}"));
                assert_sampled_inv(&r1, &read_bits(&got), m, &format!("eh d={d} {dt:?}"));
            }
        }
        let pn = |r: &MxArray, y: &MxArray, nx: bool| {
            try_post_norm_add_rms_norm(r, y, &enorm, &hnorm, eps, nx)
                .unwrap()
                .unwrap()
        };
        let (mut px, mut ph) = (Vec::new(), Vec::new());
        for r in 0..4 {
            let (xo, ho) = pn(
                &slice_axis(&e_all, false, r, r + 1),
                &slice_axis(&h_all, false, r, r + 1),
                true,
            );
            px.push((r, read_bits(&xo.unwrap())));
            ph.push((r, read_bits(&ho)));
        }
        for &m in &[1usize, 2, 4, 17] {
            let (rm, ym) = (
                slice_axis(&e_all, false, 0, m),
                slice_axis(&h_all, false, 0, m),
            );
            let ((xo, ho), (xo_none, ho2)) = (pn(&rm, &ym, true), pn(&rm, &ym, false));
            let xo = xo.unwrap();
            assert!(xo_none.is_none());
            let rx = rm.add(&rms_norm(&ym, &enorm, eps).unwrap()).unwrap();
            assert_err(&xo, &rx, 1e-5, "post_norm x");
            assert_err(
                &ho,
                &rms_norm(&rx, &hnorm, eps).unwrap(),
                1e-5,
                "post_norm h",
            );
            assert_eq!(read_bits(&ho), read_bits(&ho2));
            assert_sampled_inv(&px, &read_bits(&xo), m, "pn x");
            assert_sampled_inv(&ph, &read_bits(&ho), m, "pn h");
        }
    }
}

#[test]
fn affine4_dense_decode_row_invariant_and_codex_counterexample() {
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        assert_ne!(std::env::var("MLX_TEST_REQUIRE_METAL").as_deref(), Ok("1"));
        return;
    }
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(9600);

    // 1. Exact Codex counterexample: scale=1, bias=2^-24, codes=[1, 2, 0..], inputs=[2, -1, 0..].
    {
        let mut ww = vec![0u32; 8];
        ww[0] = 1u32 | (2u32 << 4);
        let b_val = 2.0f32.powi(-24);
        let (w, s, b) = (
            MxArray::from_uint32(&ww, &[1, 8]).unwrap(),
            as_dt(&arr(&[1.0], &[1, 1]), DType::BFloat16),
            as_dt(&arr(&[b_val], &[1, 1]), DType::BFloat16),
        );
        let q4 = Dense::Qmm4 {
            w: w.clone(),
            scales: s.clone(),
            biases: b.clone(),
        };
        let mut xr = vec![0.0f32; 8 * 64];
        (xr[0], xr[1]) = (2.0, -1.0);
        for r in 1..8 {
            xr[r * 64] = (r as f32) * 0.75 + 1.0;
            xr[r * 64 + 1] = -(r as f32) * 0.5;
            xr[r * 64 + 5] = 0.25 * (r as f32);
        }
        let x_all = arr(&xr, &[8, 64]);
        let call = |x: &MxArray| {
            try_affine4_dense_decode(x, &w, &s, &b)
                .unwrap()
                .expect("affine4 Metal")
        };
        assert_eq!(
            read_bits(&call(&slice_axis(&x_all, false, 0, 1))),
            vec![b_val.to_bits()]
        );
        assert_row_invariant(
            |r| call(&slice_axis(&x_all, false, r, r + 1)),
            |m| call(&slice_axis(&x_all, false, 0, m)),
            &[2, 3, 4, 5, 8],
            "codex kernel",
        );
        assert_row_invariant(
            |r| q4.mm_t(&slice_axis(&x_all, false, r, r + 1)).unwrap(),
            |m| q4.mm_t(&slice_axis(&x_all, false, 0, m)).unwrap(),
            &[2, 4],
            "codex mm_t",
        );
        assert_row_invariant(
            |r| head_logits(&slice_axis(&x_all, false, r, r + 1), &q4).unwrap(),
            |m| head_logits(&slice_axis(&x_all, false, 0, m), &q4).unwrap(),
            &[2, 4],
            "codex head",
        );
    }

    // 2. General shapes: non-FAST_K (64, 192, 576) and FAST_K (512, 1024) with odd N.
    let mode = std::ffi::CString::new("affine").unwrap();
    for &(n, k) in &[
        (37usize, 64usize),
        (53, 192),
        (67, 512),
        (29, 576),
        (41, 1024),
    ] {
        let q4 = quantize_affine4_array(&arr(&rng.vec_f32(n * k, 0.25), &[n, k])).unwrap();
        let Dense::Qmm4 {
            ref w,
            ref scales,
            ref biases,
        } = q4
        else {
            unreachable!()
        };
        let (sf, bf) = (as_dt(scales, DType::Float32), as_dt(biases, DType::Float32));
        let h = unsafe {
            mlx_sys::mlx_dequantize(
                w.as_raw_ptr(),
                sf.as_raw_ptr(),
                bf.as_raw_ptr(),
                64,
                4,
                DType::Float32 as i32,
                mode.as_ptr(),
            )
        };
        let deq = MxArray::from_handle(h, "test affine4 f32 dequant").unwrap();
        let x_all = arr(&rng.vec_f32(8 * k, 1.0), &[8, k]);
        let want_all = read_f32(
            &x_all
                .matmul(&deq.transpose(Some(&[1, 0])).unwrap())
                .unwrap(),
        );
        let call = |x: &MxArray| {
            try_affine4_dense_decode(x, w, scales, biases)
                .unwrap()
                .unwrap()
        };
        assert_row_invariant(
            |r| {
                let y1 = call(&slice_axis(&x_all, false, r, r + 1));
                assert!(max_abs_err(&read_f32(&y1), &want_all[r * n..(r + 1) * n]) <= 1e-4);
                y1
            },
            |m| call(&slice_axis(&x_all, false, 0, m)),
            &[2, 3, 4, 5, 8],
            &format!("affine4 ({n},{k})"),
        );
    }
}
