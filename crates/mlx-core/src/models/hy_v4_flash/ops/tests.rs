use std::path::{Path, PathBuf};

use serde_json::Value;

use super::*;
use crate::models::deepseek_v41::EngramConfig;
use crate::models::deepseek_v41::engram_hash::EngramHasher;
use crate::models::hy_v4_flash::engram::{
    self, ENGRAM_EPS, Hasher, IMAGE_TOKEN, VIDEO_TOKEN, keeps, key_value, normed_rows, signal,
    stream_dot, stream_gate,
};
use crate::models::hy_v4_flash::scan::Scan;
use crate::models::hy_v4_flash::weights::EngramWeights;

fn fixture(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hy-v4-flash")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

fn ops() -> Value {
    fixture("ops.json")
}

/// Every number of a nested JSON array, row-major.
fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(flat).collect(),
        x => vec![x.as_f64().unwrap()],
    }
}

fn f32s(v: &Value) -> Vec<f32> {
    flat(v).into_iter().map(|x| x as f32).collect()
}

fn u32s(v: &Value) -> Vec<u32> {
    flat(v).into_iter().map(|x| x as u32).collect()
}

fn u64s(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap())
        .collect()
}

fn arr(v: &Value, shape: &[i64]) -> MxArray {
    MxArray::from_float32(&f32s(v), shape).unwrap()
}

fn read(x: &MxArray) -> Vec<f32> {
    x.astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .as_ref()
        .to_vec()
}

fn shape(x: &MxArray) -> Vec<i64> {
    x.shape().unwrap().as_ref().to_vec()
}

/// `|got - want| <= 1e-6 + 1e-5 · |want|` per element (f32 vs f64 oracle).
fn assert_close(got: &[f32], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what} len");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (f64::from(g) - w).abs();
        let tol = 1e-6 + 1e-5 * w.abs();
        assert!(err <= tol, "{what}[{i}]: {g} vs {w} (err {err:e})");
    }
}

fn hc_cfg(o: &Value) -> HcConfig {
    HcConfig {
        count: o["dims"]["hc"].as_u64().unwrap() as usize,
        eps: o["hc_eps"].as_f64().unwrap() as f32,
        magnitude: o["magnitude"].as_f64().unwrap_or(2.0) as f32,
    }
}

fn hc_x(o: &Value) -> MxArray {
    let d = &o["hc_mix"]["dims"];
    let dim = |k: &str| d[k].as_i64().unwrap();
    arr(&o["hc_mix"]["x"], &[dim("n"), dim("hc"), dim("d")])
}

fn hc_weights(o: &Value, rows: i64, cols: i64, n_scale: i64) -> HcWeights {
    HcWeights {
        weight: arr(&o["fn"], &[rows, cols]),
        base: arr(&o["base"], &[rows]),
        scale: arr(&o["scale"], &[n_scale]),
    }
}

fn norm_eps(o: &Value) -> f32 {
    o["norm_eps"].as_f64().unwrap() as f32
}

#[test]
fn hc_mix_matches_the_oracle() {
    crate::models::hy_v4_flash::exact_f32();
    let all = ops();
    let o = &all["hc_mix"];
    let (hc, d) = (4, 16);
    let x = hc_x(&all);
    let w = hc_weights(o, 2 * hc, hc * d, 2);
    let mixes = hc_mixes(&x, &w.weight, norm_eps(o)).unwrap();
    assert_eq!(shape(&mixes), [3, 2 * hc]);
    assert_close(&read(&mixes), &flat(&o["mixes"]), "mixes");
    let g = hc_mix(&x, &w, norm_eps(o), &hc_cfg(o)).unwrap();
    assert_close(&read(&g.pre), &flat(&o["pre"]), "pre");
    assert_close(&read(&g.post), &flat(&o["post"]), "post");
    let red = hc_reduce(&x, &g.pre).unwrap();
    assert_eq!(shape(&red), [3, d]);
    assert_close(&read(&red), &flat(&o["reduced"]), "reduced");
}

#[test]
fn hc_shapes_are_checked() {
    let all = ops();
    let o = &all["hc_mix"];
    let x = hc_x(&all);
    let w = hc_weights(o, 8, 64, 2);
    let mut cfg = hc_cfg(o);
    cfg.count = 3;
    assert!(hc_mix(&x, &w, norm_eps(o), &cfg).is_err(), "config count");
    let short = HcWeights {
        weight: arr(&o["fn"], &[8, 64]),
        base: arr(&o["base"], &[8]),
        scale: MxArray::from_float32(&[0.7], &[1]).unwrap(),
    };
    assert!(
        hc_mix(&x, &short, norm_eps(o), &hc_cfg(o)).is_err(),
        "scale [1]"
    );
    let narrow = MxArray::from_float32(&[0.0; 8 * 32], &[8, 32]).unwrap();
    assert!(hc_mixes(&x, &narrow, 1e-5).is_err(), "fn width");
    let flat_x = x.reshape(&[3, 64]).unwrap();
    assert!(
        hc_mixes(&flat_x, &w.weight, 1e-5).is_err(),
        "rank 2 streams"
    );
    let pre = MxArray::from_float32(&[0.0; 9], &[3, 3]).unwrap();
    assert!(hc_reduce(&x, &pre).is_err(), "pre width");
    let y = MxArray::from_float32(&[0.0; 3 * 15], &[3, 15]).unwrap();
    let post = MxArray::from_float32(&[0.0; 12], &[3, 4]).unwrap();
    assert!(hc_post_identity(&y, &x, &post).is_err(), "y width");
}

#[test]
fn hc_post_identity_matches_the_oracle() {
    let all = ops();
    let o = &all["hc_post"];
    let x = hc_x(&all);
    let y = arr(&o["y"], &[3, 16]);
    let post = arr(&o["post_f32"], &[3, 4]);
    let out = hc_post_identity(&y, &x, &post).unwrap();
    assert_eq!(shape(&out), [3, 4, 16]);
    assert_close(&read(&out), &flat(&o["out"]), "out");
}

#[test]
fn hc_head_matches_the_oracle() {
    crate::models::hy_v4_flash::exact_f32();
    let all = ops();
    let o = &all["hc_head"];
    let x = hc_x(&all);
    let w = hc_weights(o, 4, 64, 1);
    let eps = o["hc_eps"].as_f64().unwrap() as f32;
    let mixes = hc_mixes(&x, &w.weight, norm_eps(o)).unwrap();
    assert_close(&read(&mixes), &flat(&o["mixes"]), "head mixes");
    let pre = hc_head_pre(&x, &w, norm_eps(o), eps).unwrap();
    assert_close(&read(&pre), &flat(&o["pre"]), "head pre");
    let h = hc_head(&x, &w, norm_eps(o), eps).unwrap();
    assert_eq!(shape(&h), [3, 16]);
    assert_close(&read(&h), &flat(&o["h"]), "h");
    let mix_w = hc_weights(&all["hc_mix"], 8, 64, 2);
    assert!(hc_head(&x, &mix_w, norm_eps(o), eps).is_err(), "8-row fn");
}

fn moe(o: &Value) -> MoeConfig {
    MoeConfig {
        n_experts: o["dims"]["experts"].as_u64().unwrap() as usize,
        n_used: o["dims"]["k"].as_u64().unwrap() as usize,
        n_shared: 1,
        ff: 1536,
        latent: 1536,
        route_scale: o["route_scale"].as_f64().unwrap() as f32,
        norm_topk: o["norm"].as_bool().unwrap(),
    }
}

/// Pure host: the oracle's f32 sigmoid scores and per-row biases give its
/// ids exactly (row 2: a 20-way tie resolved to the lowest ids).
#[test]
fn route_host_matches_the_oracle() {
    let o = &ops()["router"];
    let m = moe(o);
    let scores = o["scores_f32_sigmoid"].as_array().unwrap();
    let biases = o["bias_rows"].as_array().unwrap();
    for r in 0..3 {
        let (ids, w) = route_host(&f32s(&scores[r]), Some(&f32s(&biases[r])), &m).unwrap();
        assert_eq!(ids, u32s(&o["ids"][r]), "row {r} ids");
        assert_close(&w, &flat(&o["weights"][r]), &format!("row {r} weights"));
    }
    assert_eq!(u32s(&o["ids"][2]), (100..116).collect::<Vec<u32>>());
}

#[test]
fn route_host_refuses_bad_sizes() {
    let o = &ops()["router"];
    let m = moe(o);
    let s = f32s(&o["scores_f32_sigmoid"][0]);
    assert!(route_host(&s[..511], None, &m).is_err(), "short row");
    assert!(route_host(&s, Some(&s[..100]), &m).is_err(), "short bias");
    let mut many = m.clone();
    many.n_used = 513;
    assert!(route_host(&s, None, &many).is_err(), "k > n");
    let (ids, _) = route_host(&s, None, &m).unwrap();
    assert_eq!(ids.len(), 16);
}

/// GPU sigmoid through an identity router (exact logits) vs the oracle's
/// f32 sigmoid; the readback path then picks the oracle ids.
#[test]
fn router_scores_and_route_match_the_oracle() {
    crate::models::hy_v4_flash::exact_f32();
    let o = &ops()["router"];
    let m = moe(o);
    let n = m.n_experts;
    let mut eye = vec![0f32; n * n];
    for i in 0..n {
        eye[i * n + i] = 1.0;
    }
    let router = Dense::Array(MxArray::from_float32(&eye, &[n as i64, n as i64]).unwrap());
    let logits = arr(&o["logits"], &[3, n as i64]);
    let scores = router_scores(&logits, &router).unwrap();
    assert_eq!(scores.dtype().unwrap(), DType::Float32);
    let got = read(&scores);
    let want = f32s(&o["scores_f32_sigmoid"]);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() <= 1e-6, "score {i}: {g} vs {w}");
    }
    for r in 0..3 {
        let row = scores.slice(&[r, 0], &[r + 1, n as i64]).unwrap();
        let bias = f32s(&o["bias_rows"][r as usize]);
        let (ids, _) = route(&row, Some(&bias), &m).unwrap();
        assert_eq!(ids, u32s(&o["ids"][r as usize]), "row {r}");
    }
    assert!(route(&logits.reshape(&[3, 2, 256]).unwrap(), None, &m).is_err());
}

#[test]
fn route_select_fast_matches_deepseek_route_select_bitwise() {
    use crate::models::deepseek_v41::ops::route_select as ds_route_select;

    let n = 512usize;
    let bias = det_f32(901, n, 0.25);

    for &rows in &[1usize, 2, 16, 2048] {
        let mut scores = det_f32(1000 + rows as u64, rows * n, 2.0);
        // Inject exact ties, negative values, and -0.0 / +0.0 on row 0 (and last row).
        for target_row in [0usize, rows - 1] {
            let base = target_row * n;
            for j in 100..124 {
                scores[base + j] = -0.375;
            }
            scores[base + 124] = -0.0;
            scores[base + 125] = 0.0;
            for j in 200..210 {
                scores[base + j] = 1.5;
            }
        }

        let configs: &[(usize, bool, Option<&[f32]>, f32)] = if rows == 2048 {
            &[(16, true, Some(bias.as_slice()), 2.5)]
        } else {
            &[
                (16, true, Some(bias.as_slice()), 2.5),
                (16, true, None, 1.0),
                (16, false, Some(bias.as_slice()), 0.5),
                (1, true, Some(bias.as_slice()), 1.0),
                (2, true, None, 1.75),
            ]
        };

        for &(k, norm, b, scale) in configs {
            let biased: Vec<f32> = match b {
                Some(bv) => scores
                    .iter()
                    .zip(bv.iter().cycle())
                    .map(|(&s, &bi)| s + bi)
                    .collect(),
                None => scores.clone(),
            };
            let (want_ids, want_w) = ds_route_select(&biased, &scores, n, k, scale, norm);
            let (got_ids, got_w) = route_select_fast(&scores, b, n, k, scale, norm);
            assert_eq!(
                got_ids,
                want_ids,
                "ids mismatch rows={rows} k={k} norm={norm} bias={}",
                b.is_some()
            );
            assert_eq!(
                to_bits(&got_w),
                to_bits(&want_w),
                "weights bits mismatch rows={rows} k={k} norm={norm} bias={}",
                b.is_some()
            );
        }
    }
}

fn engram_weights(o: &Value) -> EngramWeights {
    let dim = |k: &str| o["dims"][k].as_i64().unwrap();
    let (hc, d, cols, hd) = (dim("hc"), dim("d"), dim("n_cols"), dim("head_dim"));
    EngramWeights {
        kv: Dense::Array(arr(&o["kv_w"], &[(hc + 1) * d, cols * hd])),
        kv_b: arr(&o["kv_b"], &[hc + 1, d]),
        gate: arr(&o["gate_w"], &[hc, d]),
        lookup_norm: arr(&o["lookup_norm"], &[cols, hd]),
        pre_gamma: arr(&o["pre_gamma"], &[d]),
        hc_scale: arr(&o["hc_scale"], &[hc, d]),
    }
}

#[test]
fn engram_block_matches_the_oracle() {
    crate::models::hy_v4_flash::exact_f32();
    let o = &ops()["engram_block"];
    assert_eq!(o["eps"].as_f64().unwrap() as f32, ENGRAM_EPS);
    let w = engram_weights(o);
    let (n, hc, d) = (2, 4, 8);
    let x = arr(&o["x"], &[n, hc, d]);
    let rows = f32s(&o["rows"]);
    let r = normed_rows(&rows, n, &w).unwrap();
    assert_eq!(shape(&r), [n, 8]);
    assert_close(&read(&r), &flat(&o["rows_normed"]), "rows_normed");
    let kv = key_value(&r, &w).unwrap();
    assert_eq!(shape(&kv), [n, hc + 1, d]);
    assert_close(&read(&kv), &flat(&o["kv"]), "kv");
    let key = kv.slice(&[0, 0, 0], &[n, hc, d]).unwrap();
    let value = kv
        .slice(&[0, hc, 0], &[n, hc + 1, d])
        .unwrap()
        .reshape(&[n, d])
        .unwrap();
    let dot = stream_dot(&x, &key, &w.gate).unwrap();
    assert_close(&read(&dot), &flat(&o["dot"]), "dot");
    assert_close(
        &read(&stream_gate(&dot).unwrap()),
        &flat(&o["gate"]),
        "gate",
    );
    assert_close(
        &read(&signal(&value, &w.pre_gamma).unwrap()),
        &flat(&o["signal"]),
        "signal",
    );
    let out = engram::forward(&x, &rows, &[true, true], &w).unwrap();
    assert_eq!(shape(&out), [n, hc, d]);
    assert_eq!(out.dtype().unwrap(), DType::Float32);
    assert_close(&read(&out), &flat(&o["out"]), "out");
}

/// A masked row keeps its streams bit for bit; the kept row is the
/// unmasked call's row.
#[test]
fn engram_mask_leaves_masked_rows() {
    let o = &ops()["engram_block"];
    let w = engram_weights(o);
    let x = arr(&o["x"], &[2, 4, 8]);
    let rows = f32s(&o["rows"]);
    let all = read(&engram::forward(&x, &rows, &[true, true], &w).unwrap());
    let xs = read(&x);
    let half = xs.len() / 2;
    for (keep, kept) in [([true, false], 0), ([false, true], 1)] {
        let got = read(&engram::forward(&x, &rows, &keep, &w).unwrap());
        let (k, m) = (kept * half, (1 - kept) * half);
        assert_eq!(got[k..k + half], all[k..k + half], "{keep:?}: kept row");
        assert_eq!(got[m..m + half], xs[m..m + half], "{keep:?}: masked row");
    }
    let none = read(&engram::forward(&x, &rows, &[false, false], &w).unwrap());
    assert_eq!(none, xs, "every row masked");
    assert!(
        engram::forward(&x, &rows, &[true], &w).is_err(),
        "short mask"
    );
}

/// The fixture's per-position mask: pad 120002, image 120120 and video
/// 120124 get no delta; ids >= 120818 and specials do.
#[test]
fn engram_keeps_matches_the_oracle_mask() {
    let want = fixture("engram-hash.json");
    let pad = want["pad_token_id"].as_u64().unwrap() as usize;
    assert_eq!(pad, 120_002);
    assert_eq!(
        (IMAGE_TOKEN, VIDEO_TOKEN),
        (
            want["image_token_id"].as_u64().unwrap() as u32,
            want["video_token_id"].as_u64().unwrap() as u32
        )
    );
    let mut masked = 0;
    for s in want["sequences"].as_array().unwrap() {
        let tokens = u32s(&s["tokens"]);
        let mask: Vec<bool> = s["engram_mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_bool().unwrap())
            .collect();
        assert_eq!(keeps(&tokens, pad), mask, "{}", s["name"]);
        masked += mask.iter().filter(|&&k| !k).count();
    }
    assert!(
        masked >= 4,
        "the oracle masks pad, image and video positions"
    );
}

/// A dot of exactly 0 gates with sign +1: σ(+√1e-6), not 0.5.
#[test]
fn engram_gate_sign_of_zero_is_plus() {
    let dot = MxArray::from_float32(&[0.0, -0.0, 4.0, -4.0], &[1, 4]).unwrap();
    let g = read(&stream_gate(&dot).unwrap());
    let s = |v: f64| 1.0 / (1.0 + (-v).exp());
    let want = [s(1e-3), s(1e-3), s(2.0), s(-2.0)];
    assert_close(&g, &want, "gate");
    assert!(g[0] > 0.5 && g[1] > 0.5);
}

#[test]
fn engram_forward_refuses_bad_shapes() {
    let o = &ops()["engram_block"];
    let w = engram_weights(o);
    let x = arr(&o["x"], &[2, 4, 8]);
    let rows = f32s(&o["rows"]);
    let keep = [true, true];
    assert!(
        engram::forward(&x, &rows[1..], &keep, &w).is_err(),
        "short rows"
    );
    assert!(engram::forward(&x, &[], &keep, &w).is_err(), "no rows");
    let x3 = x.slice(&[0, 0, 0], &[2, 3, 8]).unwrap();
    assert!(engram::forward(&x3, &rows, &keep, &w).is_err(), "3 streams");
    let x2 = x.reshape(&[2, 32]).unwrap();
    assert!(engram::forward(&x2, &rows, &keep, &w).is_err(), "rank 2");
}

/// Small vocab, chunks of 8 rows: the chunked BF16 head equals one f32
/// matmul of the widened weights; a chunk past the vocab is one chunk.
#[test]
fn head_logits_chunks_match_one_matmul() {
    let (vocab, d) = (37i64, 16i64);
    let words: Vec<u16> = (0..vocab * d)
        .map(|i| half::bf16::from_f32(((i * 37 % 101) as f32 - 50.0) / 64.0).to_bits())
        .collect();
    let head = Dense::Array(MxArray::from_bfloat16(&words, &[vocab, d]).unwrap());
    let xv: Vec<f32> = (0..3 * d)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 8.0)
        .collect();
    let x = MxArray::from_float32(&xv, &[3, d]).unwrap();
    let wf: Vec<f32> = words
        .iter()
        .map(|&b| f32::from_bits(u32::from(b) << 16))
        .collect();
    let want: Vec<f64> = (0..3)
        .flat_map(|r| {
            let (xv, wf) = (&xv, &wf);
            (0..vocab as usize).map(move |v| {
                (0..d as usize)
                    .map(|k| f64::from(xv[r * d as usize + k]) * f64::from(wf[v * d as usize + k]))
                    .sum::<f64>()
            })
        })
        .collect();
    for chunk in [8, 37, 1000] {
        let l = head_logits_chunked(&x, &head, chunk).unwrap();
        assert_eq!(shape(&l), [3, vocab], "chunk {chunk}");
        assert_eq!(l.dtype().unwrap(), DType::Float32);
        assert_close(&read(&l), &want, &format!("chunk {chunk}"));
    }
    assert!(head_logits_chunked(&x, &head, 0).is_err());
}

/// `engram-hash.json` sequences with the token map entries they use.
fn hash_config(want: &Value) -> EngramConfig {
    let cfg = fixture("config.json");
    let e = &cfg["engram"];
    let n = |k: &str| e[k].as_u64().unwrap() as usize;
    let mut token_map = vec![0u32; n("token_map_len")];
    let pad = want["pad_token_id"].as_u64().unwrap() as usize;
    token_map[pad] = want["token_map_pad"].as_u64().unwrap() as u32;
    for s in want["sequences"].as_array().unwrap() {
        let slots = s["compressed_slots"].as_array().unwrap();
        for (t, slot) in u32s(&s["tokens"]).into_iter().zip(slots) {
            let c = slot[0].as_u64().unwrap() as u32;
            let m = &mut token_map[t as usize];
            assert!(
                *m == 0 || *m == c || t as usize == pad,
                "token {t} maps to {m} and {c}"
            );
            *m = c;
        }
    }
    EngramConfig {
        layers: e["layers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect(),
        max_ngram: n("ngram_size"),
        n_heads: n("heads_per_ngram"),
        head_dim: n("head_dim"),
        multipliers: u64s(&e["multipliers"]),
        head_offsets: u64s(&e["head_offsets"]),
        head_vocab_sizes: u64s(&e["head_vocab_sizes"]),
        pad_token_id: pad,
        compressed_vocab_size: token_map.iter().max().map_or(0, |&x| x as usize + 1),
        token_map,
        vocab_size: None,
    }
}

/// Every sequence as one prefill, as prefill + one-token decodes from each
/// split, and per Engram layer.
fn check_hasher(h: &Hasher, want: &Value) {
    let per_pos = h.n_layers() * h.n_cols();
    for s in want["sequences"].as_array().unwrap() {
        let name = s["name"].as_str().unwrap();
        let tokens = u32s(&s["tokens"]);
        let rows = u32s(&s["rows"]);
        assert_eq!(rows.len(), tokens.len() * per_pos, "{name} oracle size");
        assert_eq!(h.call_rows(&tokens, 0).unwrap(), rows, "{name} prefill");
        for start in 0..=tokens.len() {
            let got = h.call_rows(&tokens, start).unwrap();
            assert_eq!(got, rows[start * per_pos..], "{name} from {start}");
        }
        for e in 0..h.n_layers() {
            let ids = h.layer_ids(&rows, e).unwrap();
            let want_e: Vec<u32> = rows
                .chunks(per_pos)
                .flat_map(|p| p[e * h.n_cols()..(e + 1) * h.n_cols()].to_vec())
                .collect();
            assert_eq!(ids, want_e, "{name} layer {e}");
        }
        assert!(h.call_rows(&tokens, tokens.len() + 1).is_err());
    }
    assert!(h.layer_ids(&[0; 64], 4).is_err());
    assert!(h.layer_ids(&[0; 63], 0).is_err());
}

/// Pure host: Hy4's `ple.*` constants through DS `EngramHasher` give the
/// oracle's global rows (pad slots, the raw pad id, ids >= 120818).
#[test]
fn hasher_matches_the_oracle() {
    let want = fixture("engram-hash.json");
    let h = Hasher::new(&hash_config(&want)).unwrap();
    assert_eq!((h.n_layers(), h.n_cols()), (4, 16));
    check_hasher(&h, &want);
}

/// Pure host: a call hashes only the n-gram look-back, and still equals the
/// whole-history hash at every start.
#[test]
fn call_rows_match_the_whole_history() {
    let want = fixture("engram-hash.json");
    let cfg = hash_config(&want);
    let h = Hasher::new(&cfg).unwrap();
    let full = EngramHasher::new(&cfg).unwrap();
    let tokens: Vec<u32> = (0..40u32).map(|i| (i * 7919 + 13) % 120832).collect();
    for start in 0..=tokens.len() {
        assert_eq!(
            h.call_rows(&tokens, start).unwrap(),
            full.rows(&tokens, start),
            "start {start}"
        );
    }
    let mut bad = cfg.clone();
    bad.max_ngram = 1;
    assert!(Hasher::new(&bad).is_err());
    let mut bad = cfg.clone();
    bad.head_vocab_sizes[3] = 0;
    assert!(Hasher::new(&bad).is_err());
    let mut bad = cfg;
    bad.multipliers.pop();
    assert!(Hasher::new(&bad).is_err());
}

fn real_scan() -> Scan {
    let dir = std::env::var("HY4_DIR").expect("HY4_DIR = directory of the two Hy4 shards");
    let paths = crate::models::deepseek_v41::chat::shard_paths(Path::new(&dir)).unwrap();
    assert_eq!(paths.len(), 2);
    Scan::open(&paths).unwrap()
}

/// Real headers only (CPU): the file's token map gives the oracle's
/// compressed slots, and its Engram config the oracle's rows.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_hasher_matches_the_oracle() {
    let want = fixture("engram-hash.json");
    let cfg = real_scan().cfg.engram;
    let pad = cfg.token_map[cfg.pad_token_id];
    assert_eq!(u64::from(pad), want["token_map_pad"].as_u64().unwrap());
    for s in want["sequences"].as_array().unwrap() {
        let tokens = u32s(&s["tokens"]);
        for (p, slot) in s["compressed_slots"].as_array().unwrap().iter().enumerate() {
            let got: Vec<u32> = (0..3)
                .map(|k| {
                    if p < k {
                        pad
                    } else {
                        cfg.token_map[tokens[p - k] as usize]
                    }
                })
                .collect();
            assert_eq!(got, u32s(slot), "{} pos {p}", s["name"]);
        }
    }
    check_hasher(&Hasher::new(&cfg).unwrap(), &want);
}

#[test]
fn fast_rope_and_partial_rope_tail_match_apply_partial_rope() {
    use crate::models::deepseek_v41::ops::{apply_partial_rope, inv_freqs, rope_tables};

    let rd = 64usize;
    let theta = 10_000.0f32;
    let inv = inv_freqs(rd, theta, None);
    let freqs = rope_inv_freqs(&inv).unwrap();

    let det = |seed: usize, len: usize| -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 37 + seed * 13) % 101) as f32 - 50.0) / 17.0)
            .collect()
    };

    let cases: &[(&str, Vec<u64>)] = &[
        ("contiguous_0", (0..6).collect()),
        ("contiguous_off", (13..19).collect()),
        ("strided_kpool4", (0..6).map(|g| g * 4).collect()),
        ("multi_owner_wave", vec![9, 14, 15, 20]),
        ("large_pos_single", vec![262_144]),
        ("large_pos_contig", vec![262_144, 262_145]),
        ("large_pos_strided", vec![0, 262_140, 262_144]),
    ];

    for (name, pos) in cases {
        let n = pos.len() as i64;
        let (cos, sin) = rope_tables(&inv, pos).unwrap();

        // 2-D [n, rd]
        let x2 = MxArray::from_float32(&det(1, pos.len() * rd), &[n, rd as i64]).unwrap();
        let got2 = read(&fast_rope(&x2, &freqs, pos).unwrap());
        let want2 = read(&apply_partial_rope(&x2, rd as i64, &cos, &sin, false).unwrap());
        for (i, (&g, &w)) in got2.iter().zip(&want2).enumerate() {
            let err = (g - w).abs();
            let tol = 1e-5 + 1e-5 * w.abs();
            assert!(err <= tol, "{name} 2-D[{i}]: {g} vs {w} (err {err:e})");
        }

        // Row-invariance on 2-D: each row alone matches its slice in the multi-row call bit-for-bit.
        for (r, &p) in pos.iter().enumerate() {
            let row_in = x2
                .slice(&[r as i64, 0], &[r as i64 + 1, rd as i64])
                .unwrap();
            let single = read(&fast_rope(&row_in, &freqs, &[p]).unwrap());
            let slice = &got2[r * rd..(r + 1) * rd];
            let sb: Vec<u32> = single.iter().map(|v| v.to_bits()).collect();
            let mb: Vec<u32> = slice.iter().map(|v| v.to_bits()).collect();
            assert_eq!(sb, mb, "{name} 2-D row {r} invariance");
        }

        // 3-D [n, H, rd]
        let h = 4usize;
        let x3 =
            MxArray::from_float32(&det(2, pos.len() * h * rd), &[n, h as i64, rd as i64]).unwrap();
        let got3 = read(&fast_rope(&x3, &freqs, pos).unwrap());
        let want3 = read(&apply_partial_rope(&x3, rd as i64, &cos, &sin, false).unwrap());
        for (i, (&g, &w)) in got3.iter().zip(&want3).enumerate() {
            let err = (g - w).abs();
            let tol = 1e-5 + 1e-5 * w.abs();
            assert!(err <= tol, "{name} 3-D[{i}]: {g} vs {w} (err {err:e})");
        }

        // Row-invariance on 3-D
        for (r, &p) in pos.iter().enumerate() {
            let row_in = x3
                .slice(&[r as i64, 0, 0], &[r as i64 + 1, h as i64, rd as i64])
                .unwrap();
            let single = read(&fast_rope(&row_in, &freqs, &[p]).unwrap());
            let slice = &got3[r * h * rd..(r + 1) * h * rd];
            let sb: Vec<u32> = single.iter().map(|v| v.to_bits()).collect();
            let mb: Vec<u32> = slice.iter().map(|v| v.to_bits()).collect();
            assert_eq!(sb, mb, "{name} 3-D row {r} invariance");
        }

        // Partial tail on [n, 128] and [n, H, 128] (head 64 untouched, tail 64 rotated)
        let d = 128usize;
        let xt2 = MxArray::from_float32(&det(3, pos.len() * d), &[n, d as i64]).unwrap();
        let got_t2 = read(&fast_partial_rope_tail(&xt2, &freqs, pos).unwrap());
        let want_t2 = read(&apply_partial_rope(&xt2, rd as i64, &cos, &sin, false).unwrap());
        for (i, (&g, &w)) in got_t2.iter().zip(&want_t2).enumerate() {
            let err = (g - w).abs();
            let tol = 1e-5 + 1e-5 * w.abs();
            assert!(err <= tol, "{name} tail 2-D[{i}]: {g} vs {w} (err {err:e})");
        }

        let xt3 =
            MxArray::from_float32(&det(4, pos.len() * h * d), &[n, h as i64, d as i64]).unwrap();
        let got_t3 = read(&fast_partial_rope_tail(&xt3, &freqs, pos).unwrap());
        let want_t3 = read(&apply_partial_rope(&xt3, rd as i64, &cos, &sin, false).unwrap());
        for (i, (&g, &w)) in got_t3.iter().zip(&want_t3).enumerate() {
            let err = (g - w).abs();
            let tol = 1e-5 + 1e-5 * w.abs();
            assert!(err <= tol, "{name} tail 3-D[{i}]: {g} vs {w} (err {err:e})");
        }
    }

    // Overflowing u64 position is rejected cleanly
    let x1 = MxArray::from_float32(&vec![1.0; rd], &[1, rd as i64]).unwrap();
    assert!(fast_rope(&x1, &freqs, &[u64::from(u32::MAX)]).is_err());
}

#[test]
fn rms_norm_and_unweighted_match_oracle() {
    let eps = 1e-5f32;
    let (rows, dim) = (5usize, 64usize);
    let xv: Vec<f32> = (0..rows * dim)
        .map(|i| ((i * 29 % 97) as f32 - 48.0) / 13.0)
        .collect();
    let wv: Vec<f32> = (0..dim).map(|i| 0.7 + (i as f32) * 0.02).collect();

    let x = MxArray::from_float32(&xv, &[rows as i64, dim as i64]).unwrap();
    let w = MxArray::from_float32(&wv, &[dim as i64]).unwrap();

    let mut want_unw = Vec::with_capacity(rows * dim);
    let mut want_w = Vec::with_capacity(rows * dim);
    for r in 0..rows {
        let row = &xv[r * dim..(r + 1) * dim];
        let mean_sq = row
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / dim as f64;
        let inv = 1.0 / (mean_sq + f64::from(eps)).sqrt();
        for (c, &v) in row.iter().enumerate() {
            let u = f64::from(v) * inv;
            want_unw.push(u);
            want_w.push(u * f64::from(wv[c]));
        }
    }

    let got_w = rms_norm(&x, &w, eps).unwrap();
    assert_eq!(got_w.dtype().unwrap(), DType::Float32);
    assert_close(&read(&got_w), &want_w, "rms_norm f32");

    let got_unw = rms_norm_unweighted(&x, eps).unwrap();
    assert_eq!(got_unw.dtype().unwrap(), DType::Float32);
    assert_close(&read(&got_unw), &want_unw, "rms_norm_unweighted f32");

    // Dtype round-trip (bf16 -> f32 -> bf16)
    let xb = x.astype(DType::BFloat16).unwrap();
    let wb = w.astype(DType::BFloat16).unwrap();
    assert_eq!(
        rms_norm(&xb, &wb, eps).unwrap().dtype().unwrap(),
        DType::BFloat16
    );
    assert_eq!(
        rms_norm_unweighted(&xb, eps).unwrap().dtype().unwrap(),
        DType::BFloat16
    );

    // 2-D weight is rejected
    let w2 = MxArray::from_float32(&vec![1.0; 2 * dim], &[2, dim as i64]).unwrap();
    assert!(rms_norm(&x, &w2, eps).is_err());
}

#[test]
fn swiglu_clamped_compiled_matches_asymmetric_clamp() {
    let limit = 7.0f32;
    // Covers g > L, g < -L (must NOT be lower-clamped!), |g| <= L, |u| > L, |u| <= L.
    let gs = [-20.0f32, -8.5, -7.0, -2.0, 0.0, 2.5, 7.0, 9.5, 25.0];
    let us = [-15.0f32, -7.5, -7.0, -3.0, 0.0, 3.0, 7.0, 8.0, 18.0];
    let mut g_all = Vec::new();
    let mut u_all = Vec::new();
    for &g in &gs {
        for &u in &us {
            g_all.push(g);
            u_all.push(u);
        }
    }
    let n = g_all.len() as i64;
    let ga = MxArray::from_float32(&g_all, &[1, n]).unwrap();
    let ua = MxArray::from_float32(&u_all, &[1, n]).unwrap();

    let got = read(&Activations::swiglu_clamped_compiled(&ga, &ua, limit).unwrap());
    let want: Vec<f64> = g_all
        .iter()
        .zip(&u_all)
        .map(|(&g, &u)| {
            let gc = f64::from(g.min(limit));
            let uc = f64::from(u.clamp(-limit, limit));
            (1.0 / (1.0 + (-gc).exp())) * gc * uc
        })
        .collect();
    assert_close(&got, &want, "swiglu_clamped_compiled");

    // Verify g < -L is NOT lower-clamped: compare against a deliberately wrong symmetric clamp
    // at g = -20, u = 3 (silu(-20)*3 ≈ -1.2e-7 vs silu(-7)*3 ≈ -0.019).
    let neg_g = MxArray::from_float32(&[-20.0], &[1]).unwrap();
    let pos_u = MxArray::from_float32(&[3.0], &[1]).unwrap();
    let val = read(&Activations::swiglu_clamped_compiled(&neg_g, &pos_u, limit).unwrap())[0];
    assert!(
        val.abs() < 1e-4,
        "g = -20 must not be lower-clamped to -7 (got {val})"
    );

    // limit <= 0 falls back to unclamped swiglu_compiled bit-for-bit
    let unclamped = read(&Activations::swiglu_compiled(&ga, &ua).unwrap());
    for l in [0.0f32, -1.0f32] {
        let g0 = read(&Activations::swiglu_clamped_compiled(&ga, &ua, l).unwrap());
        let b0: Vec<u32> = g0.iter().map(|v| v.to_bits()).collect();
        let bu: Vec<u32> = unclamped.iter().map(|v| v.to_bits()).collect();
        assert_eq!(b0, bu, "limit={l} must equal unclamped");
    }

    // Shapeless compile handles different shapes and preserves dtype
    for dt in [DType::Float32, DType::Float16, DType::BFloat16] {
        let g3 = ga.reshape(&[9, 9]).unwrap().astype(dt).unwrap();
        let u3 = ua.reshape(&[9, 9]).unwrap().astype(dt).unwrap();
        let out = Activations::swiglu_clamped_compiled(&g3, &u3, limit).unwrap();
        assert_eq!(out.dtype().unwrap(), dt);
        assert_eq!(shape(&out), [9, 9]);
    }
}

fn det_f32(seed: u64, len: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn to_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn fused_hc_row_invariance_and_f32_parity() {
    crate::models::hy_v4_flash::exact_f32();
    let hc = 4i64;
    let norm_eps = 1e-6f32;
    let hc_eps = 1e-4f32;
    let magnitude = 2.0f32;
    let cfg = HcConfig {
        count: hc as usize,
        eps: hc_eps,
        magnitude,
    };

    for &d in &[64i64, 66, 3072] {
        let hd = (hc * d) as usize;
        let scale_w = 1.0 / (hd as f32).sqrt();
        let w_mix = HcWeights {
            weight: MxArray::from_float32(
                &det_f32(101 + d as u64, (2 * hc) as usize * hd, scale_w),
                &[2 * hc, hc * d],
            )
            .unwrap(),
            scale: MxArray::from_float32(&[0.75, 1.25], &[2]).unwrap(),
            base: MxArray::from_float32(
                &det_f32(201 + d as u64, (2 * hc) as usize, 0.5),
                &[2 * hc],
            )
            .unwrap(),
        };
        let w_head = HcWeights {
            weight: MxArray::from_float32(
                &det_f32(301 + d as u64, hc as usize * hd, scale_w),
                &[hc, hc * d],
            )
            .unwrap(),
            scale: MxArray::from_float32(&[0.85], &[1]).unwrap(),
            base: MxArray::from_float32(&det_f32(401 + d as u64, hc as usize, 0.5), &[hc]).unwrap(),
        };

        let max_m = 256usize;
        let x_all_v = det_f32(501 + d as u64, max_m * hd, 1.5);
        let y_all_v = det_f32(601 + d as u64, max_m * d as usize, 1.5);
        let x_all = MxArray::from_float32(&x_all_v, &[max_m as i64, hc, d]).unwrap();
        let y_all = MxArray::from_float32(&y_all_v, &[max_m as i64, d]).unwrap();

        // Single-row baselines for sampled rows across every tested M.
        let sample_rows = [0usize, 1, 3, 16, 255];
        let mut single_h = std::collections::HashMap::new();
        let mut single_post = std::collections::HashMap::new();
        let mut single_head = std::collections::HashMap::new();
        let mut single_out = std::collections::HashMap::new();
        for &r in &sample_rows {
            let xr = x_all
                .slice(&[r as i64, 0, 0], &[r as i64 + 1, hc, d])
                .unwrap();
            let yr = y_all.slice(&[r as i64, 0], &[r as i64 + 1, d]).unwrap();
            let (h1, p1) = hc_mix_reduce(&xr, &w_mix, norm_eps, &cfg).unwrap();
            let head1 = hc_head(&xr, &w_head, norm_eps, hc_eps).unwrap();
            let out1 = hc_post_identity(&yr, &xr, &p1).unwrap();
            single_h.insert(r, to_bits(&read(&h1)));
            single_post.insert(r, to_bits(&read(&p1)));
            single_head.insert(r, to_bits(&read(&head1)));
            single_out.insert(r, to_bits(&read(&out1)));
        }

        for &m in &[1usize, 2, 4, 17, 256] {
            let xm = x_all.slice(&[0, 0, 0], &[m as i64, hc, d]).unwrap();
            let ym = y_all.slice(&[0, 0], &[m as i64, d]).unwrap();
            let (hm, pm) = hc_mix_reduce(&xm, &w_mix, norm_eps, &cfg).unwrap();
            let headm = hc_head(&xm, &w_head, norm_eps, hc_eps).unwrap();
            let outm = hc_post_identity(&ym, &xm, &pm).unwrap();

            let hm_v = read(&hm);
            let pm_v = read(&pm);
            let headm_v = read(&headm);
            let outm_v = read(&outm);

            for &r in &sample_rows {
                if r >= m {
                    continue;
                }
                assert_eq!(
                    to_bits(&hm_v[r * d as usize..(r + 1) * d as usize]),
                    single_h[&r],
                    "hc_mix_reduce h invariance d={d} M={m} row={r}"
                );
                assert_eq!(
                    to_bits(&pm_v[r * hc as usize..(r + 1) * hc as usize]),
                    single_post[&r],
                    "hc_mix_reduce post invariance d={d} M={m} row={r}"
                );
                assert_eq!(
                    to_bits(&headm_v[r * d as usize..(r + 1) * d as usize]),
                    single_head[&r],
                    "hc_head invariance d={d} M={m} row={r}"
                );
                assert_eq!(
                    to_bits(&outm_v[r * hd..(r + 1) * hd]),
                    single_out[&r],
                    "hc_post_identity invariance d={d} M={m} row={r}"
                );
            }

            if m <= 17 {
                // F32 reference parity (pure MLX ops)
                let ref_g = hc_mix(&xm, &w_mix, norm_eps, &cfg).unwrap();
                let ref_h = read(&hc_reduce(&xm, &ref_g.pre).unwrap());
                let ref_p = read(&ref_g.post);
                let ref_head_pre = hc_head_pre(&xm, &w_head, norm_eps, hc_eps).unwrap();
                let ref_head = read(&hc_reduce(&xm, &ref_head_pre).unwrap());
                let p_b = pm.reshape(&[m as i64, hc, 1]).unwrap();
                let y_b = ym.reshape(&[m as i64, 1, d]).unwrap();
                let ref_out = read(&xm.add(&p_b.mul(&y_b).unwrap()).unwrap());

                let max_err = |a: &[f32], b: &[f32]| -> f32 {
                    a.iter()
                        .zip(b)
                        .map(|(&x, &y)| (x - y).abs())
                        .fold(0.0f32, f32::max)
                };
                let eh = max_err(&hm_v, &ref_h);
                let ep = max_err(&pm_v, &ref_p);
                let ehead = max_err(&headm_v, &ref_head);
                let eout = max_err(&outm_v, &ref_out);
                assert!(eh <= 1e-5, "hc_mix_reduce h parity d={d} M={m}: {eh:e}");
                assert!(ep <= 1e-5, "hc_mix_reduce post parity d={d} M={m}: {ep:e}");
                assert!(ehead <= 1e-5, "hc_head parity d={d} M={m}: {ehead:e}");
                assert!(
                    eout <= 1e-5,
                    "hc_post_identity parity d={d} M={m}: {eout:e}"
                );
            }
        }
    }
}

#[test]
fn bf16_lm_head_row_invariance_and_f32_parity() {
    crate::models::hy_v4_flash::exact_f32();
    for &v in &[32i64, 37, 4096] {
        for &k in &[64i64, 192, 3072] {
            let w_f32 = det_f32(
                701 + (v * 31 + k) as u64,
                (v * k) as usize,
                1.0 / (k as f32).sqrt(),
            );
            let w_bits: Vec<u16> = w_f32
                .iter()
                .map(|&val| half::bf16::from_f32(val).to_bits())
                .collect();
            let w_bf16 = MxArray::from_bfloat16(&w_bits, &[v, k]).unwrap();
            let head = Dense::Array(w_bf16.clone());
            let wt_f32 = w_bf16
                .astype(DType::Float32)
                .unwrap()
                .transpose(Some(&[1, 0]))
                .unwrap();

            let x_all = MxArray::from_float32(
                &det_f32(801 + (v * 17 + k) as u64, 16 * k as usize, 1.0),
                &[16, k],
            )
            .unwrap();

            let single: Vec<Vec<u32>> = (0..16i64)
                .map(|r| {
                    let xr = x_all.slice(&[r, 0], &[r + 1, k]).unwrap();
                    to_bits(&read(&head_logits(&xr, &head).unwrap()))
                })
                .collect();

            for &m in &[1i64, 2, 4, 7, 16] {
                let xm = x_all.slice(&[0, 0], &[m, k]).unwrap();
                let got = head_logits(&xm, &head).unwrap();
                assert_eq!(shape(&got), [m, v]);
                assert_eq!(got.dtype().unwrap(), DType::Float32);
                let got_v = read(&got);

                for r in 0..m as usize {
                    assert_eq!(
                        to_bits(&got_v[r * v as usize..(r + 1) * v as usize]),
                        single[r],
                        "bf16_lm_head invariance V={v} K={k} M={m} row={r}"
                    );
                }

                let ref_v = read(&xm.matmul(&wt_f32).unwrap());
                let max_abs = got_v
                    .iter()
                    .zip(&ref_v)
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs <= 1e-4,
                    "bf16_lm_head parity V={v} K={k} M={m}: max_abs={max_abs:e}"
                );
            }
        }
    }
}

#[test]
fn router_topk_gpu_matches_route_select_fast_and_is_row_invariant() {
    crate::models::hy_v4_flash::exact_f32();
    let e = 512usize;
    let k = 16usize;
    let bias_v = det_f32(1201, e, 0.25);
    let bias_arr = MxArray::from_float32(&bias_v, &[e as i64]).unwrap();

    let max_m = 256usize;
    let mut logits_v = det_f32(1202, max_m * e, 4.0);
    // Inject exact ties, large negative logits, zeros, and NaN on sampled rows.
    for &r in &[0usize, 1, 3, 15, 255] {
        let base = r * e;
        for j in 50..74 {
            logits_v[base + j] = -0.625;
        }
        for j in 200..220 {
            logits_v[base + j] = 2.25;
        }
        logits_v[base + 300] = -40.0;
        logits_v[base + 301] = 0.0;
    }
    let logits_all = MxArray::from_float32(&logits_v, &[max_m as i64, e as i64]).unwrap();

    for &(has_bias, norm_topk, scale) in &[
        (true, true, 2.5f32),
        (false, true, 1.0f32),
        (true, false, 0.75f32),
    ] {
        let cfg = MoeConfig {
            n_experts: e,
            n_used: k,
            n_shared: 1,
            ff: 1536,
            latent: 1536,
            route_scale: scale,
            norm_topk,
        };
        let b_opt = has_bias.then_some(&bias_arr);
        let b_slice = has_bias.then_some(bias_v.as_slice());

        // Single-row baselines for strict row invariance on rows 0..16.
        let mut single_ids = Vec::with_capacity(16);
        let mut single_w_bits = Vec::with_capacity(16);
        for r in 0..16i64 {
            let row = logits_all.slice(&[r, 0], &[r + 1, e as i64]).unwrap();
            let Some((gi, gw)) = router_topk_gpu(&row, b_opt, &cfg).unwrap() else {
                return;
            };
            single_ids.push(gi.to_uint32().unwrap().to_vec());
            single_w_bits.push(to_bits(&read(&gw)));
        }

        for &m in &[1usize, 2, 4, 16, 256] {
            let lm = logits_all.slice(&[0, 0], &[m as i64, e as i64]).unwrap();
            let (gi, gw) = router_topk_gpu(&lm, b_opt, &cfg).unwrap().unwrap();
            assert_eq!(shape(&gi), [m as i64, k as i64]);
            assert_eq!(shape(&gw), [m as i64, k as i64]);
            assert_eq!(gi.dtype().unwrap(), DType::Uint32);
            assert_eq!(gw.dtype().unwrap(), DType::Float32);
            let got_ids = gi.to_uint32().unwrap().to_vec();
            let got_w = read(&gw);

            // Reference: MLX sigmoid + host route_select_fast sorted ascending by id.
            let scores = read(&Activations::sigmoid(&lm).unwrap());
            let (ref_ids_rank, ref_w_rank) =
                route_select_fast(&scores, b_slice, e, k, scale, norm_topk);
            let mut want_ids = Vec::with_capacity(m * k);
            let mut want_w = Vec::with_capacity(m * k);
            for r in 0..m {
                let mut pairs: Vec<(u32, f32)> = ref_ids_rank[r * k..(r + 1) * k]
                    .iter()
                    .copied()
                    .zip(ref_w_rank[r * k..(r + 1) * k].iter().copied())
                    .collect();
                pairs.sort_by_key(|p| p.0);
                want_ids.extend(pairs.iter().map(|p| p.0));
                want_w.extend(pairs.iter().map(|p| p.1));
            }

            assert_eq!(
                got_ids, want_ids,
                "router_topk ids mismatch M={m} bias={has_bias} norm={norm_topk}"
            );
            let max_abs = got_w
                .iter()
                .zip(&want_w)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs <= 1e-6,
                "router_topk weights M={m} bias={has_bias} norm={norm_topk}: max_abs={max_abs:e}"
            );

            // Strict row invariance: rows 0..min(m, 16) bit-identical to M=1.
            for r in 0..m.min(16) {
                assert_eq!(
                    &got_ids[r * k..(r + 1) * k],
                    single_ids[r].as_slice(),
                    "router_topk ids invariance M={m} row={r}"
                );
                assert_eq!(
                    to_bits(&got_w[r * k..(r + 1) * k]),
                    single_w_bits[r],
                    "router_topk weights invariance M={m} row={r}"
                );
            }
        }
    }

    // Verify E=1024 (2nd MAX_E bucket) and NaN safety.
    let e_big = 1024usize;
    let mut big_logits = det_f32(1299, 2 * e_big, 3.0);
    big_logits[7] = f32::NAN;
    big_logits[e_big + 513] = f32::NAN;
    let cfg_big = MoeConfig {
        n_experts: e_big,
        n_used: 16,
        n_shared: 1,
        ff: 1536,
        latent: 1536,
        route_scale: 1.0,
        norm_topk: true,
    };
    let big_arr = MxArray::from_float32(&big_logits, &[2, e_big as i64]).unwrap();
    if let Some((gi, gw)) = router_topk_gpu(&big_arr, None, &cfg_big).unwrap() {
        let ids = gi.to_uint32().unwrap().to_vec();
        let ws = read(&gw);
        assert!(ws.iter().all(|v| v.is_finite()));
        assert!(!ids[..16].contains(&7));
        assert!(!ids[16..].contains(&513));
        for row in ids.as_chunks::<16>().0 {
            assert!(row.windows(2).all(|w| w[0] < w[1]));
        }
    }

    // Verify bit-exact parity against CPU route_host across edge E and K values
    // including heavy multi-SG ties.
    for &(ev, kv) in &[
        (16usize, 16usize),
        (32, 1),
        (500, 16),
        (512, 16),
        (512, 24),
        (1000, 32),
        (1024, 16),
    ] {
        let rows = 4usize;
        let mut lv = det_f32(1300 + (ev * 31 + kv) as u64, rows * ev, 3.0);
        // Row 1: global tie across every expert (tests lowest-id tie-breaking across all 4 SGs).
        for j in 0..ev {
            lv[ev + j] = 1.25;
        }
        // Row 2: 24-way tie scattered across SG 0..3 above random background.
        for t in 0..24.min(ev) {
            let idx = (t * 37) % ev;
            lv[2 * ev + idx] = 10.0;
        }
        let l_arr = MxArray::from_float32(&lv, &[rows as i64, ev as i64]).unwrap();
        let bv = det_f32(1400 + ev as u64, ev, 0.25);
        let b_arr = MxArray::from_float32(&bv, &[ev as i64]).unwrap();
        let cfg = MoeConfig {
            n_experts: ev,
            n_used: kv,
            n_shared: 1,
            ff: 1536,
            latent: 1536,
            route_scale: 2.5,
            norm_topk: true,
        };
        for use_bias in [false, true] {
            let Some((gi, gw)) = router_topk_gpu(&l_arr, use_bias.then_some(&b_arr), &cfg).unwrap()
            else {
                return;
            };
            let got_ids = gi.to_uint32().unwrap().to_vec();
            let got_w = read(&gw);
            let scores = read(&Activations::sigmoid(&l_arr).unwrap());
            let (ref_ids, ref_w) =
                route_host(&scores, use_bias.then_some(bv.as_slice()), &cfg).unwrap();
            let mut want_ids = Vec::with_capacity(rows * kv);
            let mut want_w = Vec::with_capacity(rows * kv);
            for r in 0..rows {
                let mut pairs: Vec<(u32, f32)> = ref_ids[r * kv..(r + 1) * kv]
                    .iter()
                    .copied()
                    .zip(ref_w[r * kv..(r + 1) * kv].iter().copied())
                    .collect();
                pairs.sort_by_key(|p| p.0);
                want_ids.extend(pairs.iter().map(|p| p.0));
                want_w.extend(pairs.iter().map(|p| p.1));
            }
            assert_eq!(
                got_ids, want_ids,
                "router_topk edge ids mismatch E={ev} K={kv} bias={use_bias}"
            );
            assert_eq!(
                to_bits(&got_w),
                to_bits(&want_w),
                "router_topk edge weights bit mismatch E={ev} K={kv} bias={use_bias}"
            );
        }
    }
}

#[test]
fn hc_fused_suite_matches_unfused_and_is_row_invariant() {
    crate::models::hy_v4_flash::exact_f32();
    let hc = 4i64;
    let norm_eps = 1e-5f32;
    let hc_eps = 1e-6f32;
    let magnitude = 2.0f32;
    let cfg = HcConfig {
        count: hc as usize,
        eps: hc_eps,
        magnitude,
    };
    let max_err = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    for &d in &[64i64, 66, 3072] {
        let hd = (hc * d) as usize;
        let scale_w = 1.0 / (hd as f32).sqrt();
        let w_next = HcWeights {
            weight: MxArray::from_float32(
                &det_f32(2101 + d as u64, (2 * hc) as usize * hd, scale_w),
                &[2 * hc, hc * d],
            )
            .unwrap(),
            scale: MxArray::from_float32(&[0.75, 1.25], &[2]).unwrap(),
            base: MxArray::from_float32(
                &det_f32(2201 + d as u64, (2 * hc) as usize, 0.5),
                &[2 * hc],
            )
            .unwrap(),
        };
        let w_head = HcWeights {
            weight: MxArray::from_float32(
                &det_f32(2301 + d as u64, hc as usize * hd, scale_w),
                &[hc, hc * d],
            )
            .unwrap(),
            scale: MxArray::from_float32(&[0.85], &[1]).unwrap(),
            base: MxArray::from_float32(&det_f32(2401 + d as u64, hc as usize, 0.5), &[hc])
                .unwrap(),
        };
        let post_norm_w = MxArray::from_float32(
            &det_f32(2501 + d as u64, d as usize, 0.5)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
            &[d],
        )
        .unwrap();
        let next_norm_w = MxArray::from_float32(
            &det_f32(2601 + d as u64, d as usize, 0.5)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
            &[d],
        )
        .unwrap();

        let max_m = 17usize;
        let x_all = MxArray::from_float32(
            &det_f32(2701 + d as u64, max_m * hd, 1.5),
            &[max_m as i64, hc, d],
        )
        .unwrap();
        let y_all = MxArray::from_float32(
            &det_f32(2801 + d as u64, max_m * d as usize, 1.5),
            &[max_m as i64, d],
        )
        .unwrap();
        let post_prev_all = MxArray::from_float32(
            &det_f32(2901 + d as u64, max_m * hc as usize, 0.5)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
            &[max_m as i64, hc],
        )
        .unwrap();

        let sample_rows = [0usize, 1, 3, 16];
        let mut s_mrn_h = std::collections::HashMap::new();
        let mut s_mrn_p = std::collections::HashMap::new();
        let mut s_hn = std::collections::HashMap::new();
        let mut s_pnhp = std::collections::HashMap::new();
        let mut s_br_x = std::collections::HashMap::new();
        let mut s_br_h = std::collections::HashMap::new();
        let mut s_br_p = std::collections::HashMap::new();
        let mut s_br_skip_h = std::collections::HashMap::new();
        let mut s_br_head = std::collections::HashMap::new();

        for &r in &sample_rows {
            let xr = x_all
                .slice(&[r as i64, 0, 0], &[r as i64 + 1, hc, d])
                .unwrap();
            let yr = y_all.slice(&[r as i64, 0], &[r as i64 + 1, d]).unwrap();
            let pr = post_prev_all
                .slice(&[r as i64, 0], &[r as i64 + 1, hc])
                .unwrap();

            let (mrn_h, mrn_p) =
                hc_mix_reduce_norm(&xr, &w_next, &next_norm_w, norm_eps, &cfg).unwrap();
            let hn = hc_head_norm(&xr, &w_head, &next_norm_w, norm_eps, hc_eps).unwrap();
            let pnhp = post_norm_hc_post(&yr, &xr, &pr, &post_norm_w, norm_eps).unwrap();
            let (br_x, br_h, br_p) = hc_bridge(
                &yr,
                &xr,
                &pr,
                &post_norm_w,
                &w_next,
                Some(&next_norm_w),
                norm_eps,
                &cfg,
            )
            .unwrap();
            let (_, br_skip_h, _) =
                hc_bridge(&yr, &xr, &pr, &post_norm_w, &w_next, None, norm_eps, &cfg).unwrap();
            let br_head = hc_bridge_head(
                &yr,
                &xr,
                &pr,
                &post_norm_w,
                &w_head,
                Some(&next_norm_w),
                norm_eps,
                hc_eps,
            )
            .unwrap();

            s_mrn_h.insert(r, to_bits(&read(&mrn_h)));
            s_mrn_p.insert(r, to_bits(&read(&mrn_p)));
            s_hn.insert(r, to_bits(&read(&hn)));
            s_pnhp.insert(r, to_bits(&read(&pnhp)));
            s_br_x.insert(r, to_bits(&read(&br_x)));
            s_br_h.insert(r, to_bits(&read(&br_h)));
            s_br_p.insert(r, to_bits(&read(&br_p)));
            s_br_skip_h.insert(r, to_bits(&read(&br_skip_h)));
            s_br_head.insert(r, to_bits(&read(&br_head)));
        }

        for &m in &[1usize, 2, 4, 17] {
            let xm = x_all.slice(&[0, 0, 0], &[m as i64, hc, d]).unwrap();
            let ym = y_all.slice(&[0, 0], &[m as i64, d]).unwrap();
            let pm = post_prev_all.slice(&[0, 0], &[m as i64, hc]).unwrap();

            // 1. hc_mix_reduce_norm (HEAD_ONLY=false)
            let (got_mrn_h, got_mrn_p) =
                hc_mix_reduce_norm(&xm, &w_next, &next_norm_w, norm_eps, &cfg).unwrap();
            let (ref_h, ref_p) = hc_mix_reduce(&xm, &w_next, norm_eps, &cfg).unwrap();
            let ref_mrn_h = read(&rms_norm(&ref_h, &next_norm_w, norm_eps).unwrap());
            let got_mrn_hv = read(&got_mrn_h);
            let got_mrn_pv = read(&got_mrn_p);
            assert!(
                max_err(&got_mrn_hv, &ref_mrn_h) <= 1e-5,
                "hc_mix_reduce_norm h d={d} M={m}"
            );
            assert!(
                max_err(&got_mrn_pv, &read(&ref_p)) <= 1e-5,
                "hc_mix_reduce_norm post d={d} M={m}"
            );

            // 2. hc_head_norm (HEAD_ONLY=true)
            let got_hn = read(&hc_head_norm(&xm, &w_head, &next_norm_w, norm_eps, hc_eps).unwrap());
            let ref_hn = read(
                &rms_norm(
                    &hc_head(&xm, &w_head, norm_eps, hc_eps).unwrap(),
                    &next_norm_w,
                    norm_eps,
                )
                .unwrap(),
            );
            assert!(
                max_err(&got_hn, &ref_hn) <= 1e-5,
                "hc_head_norm d={d} M={m}"
            );

            // 3. post_norm_hc_post
            let got_pnhp = read(&post_norm_hc_post(&ym, &xm, &pm, &post_norm_w, norm_eps).unwrap());
            let ref_yn = rms_norm(&ym, &post_norm_w, norm_eps).unwrap();
            let ref_x_next = hc_post_identity(&ref_yn, &xm, &pm).unwrap();
            let ref_x_next_v = read(&ref_x_next);
            assert!(
                max_err(&got_pnhp, &ref_x_next_v) <= 1e-5,
                "post_norm_hc_post d={d} M={m}"
            );

            // 4. hc_bridge (normal: SKIP_NEXT_NORM=false, HEAD_ONLY=false)
            let (got_bx, got_bh, got_bp) = hc_bridge(
                &ym,
                &xm,
                &pm,
                &post_norm_w,
                &w_next,
                Some(&next_norm_w),
                norm_eps,
                &cfg,
            )
            .unwrap();
            let (ref_bh_raw, ref_bp) = hc_mix_reduce(&ref_x_next, &w_next, norm_eps, &cfg).unwrap();
            let ref_bh_norm = read(&rms_norm(&ref_bh_raw, &next_norm_w, norm_eps).unwrap());
            let got_bx_v = read(&got_bx);
            let got_bh_v = read(&got_bh);
            let got_bp_v = read(&got_bp);
            assert!(
                max_err(&got_bx_v, &ref_x_next_v) <= 1e-5,
                "hc_bridge x_out d={d} M={m}"
            );
            assert!(
                max_err(&got_bh_v, &ref_bh_norm) <= 1e-5,
                "hc_bridge h_out d={d} M={m}"
            );
            assert!(
                max_err(&got_bp_v, &read(&ref_bp)) <= 1e-5,
                "hc_bridge post_next d={d} M={m}"
            );

            // 5. hc_bridge (SKIP_NEXT_NORM=true, HEAD_ONLY=false)
            let (got_sx, got_sh, got_sp) =
                hc_bridge(&ym, &xm, &pm, &post_norm_w, &w_next, None, norm_eps, &cfg).unwrap();
            let got_sh_v = read(&got_sh);
            assert!(
                max_err(&read(&got_sx), &ref_x_next_v) <= 1e-5,
                "hc_bridge skip x_out d={d} M={m}"
            );
            assert!(
                max_err(&got_sh_v, &read(&ref_bh_raw)) <= 1e-5,
                "hc_bridge skip h_out d={d} M={m}"
            );
            assert!(
                max_err(&read(&got_sp), &read(&ref_bp)) <= 1e-5,
                "hc_bridge skip post_next d={d} M={m}"
            );

            // 6. hc_bridge_head (HEAD_ONLY=true, with output_norm and with SKIP_NEXT_NORM=true)
            let got_bhead = read(
                &hc_bridge_head(
                    &ym,
                    &xm,
                    &pm,
                    &post_norm_w,
                    &w_head,
                    Some(&next_norm_w),
                    norm_eps,
                    hc_eps,
                )
                .unwrap(),
            );
            let ref_head_raw = hc_head(&ref_x_next, &w_head, norm_eps, hc_eps).unwrap();
            let ref_bhead = read(&rms_norm(&ref_head_raw, &next_norm_w, norm_eps).unwrap());
            assert!(
                max_err(&got_bhead, &ref_bhead) <= 1e-5,
                "hc_bridge_head d={d} M={m}"
            );
            let got_bhead_skip = read(
                &hc_bridge_head(&ym, &xm, &pm, &post_norm_w, &w_head, None, norm_eps, hc_eps)
                    .unwrap(),
            );
            assert!(
                max_err(&got_bhead_skip, &read(&ref_head_raw)) <= 1e-5,
                "hc_bridge_head skip d={d} M={m}"
            );

            // Strict bitwise row invariance against M=1.
            let du = d as usize;
            let hcu = hc as usize;
            for &r in &sample_rows {
                if r >= m {
                    continue;
                }
                assert_eq!(
                    to_bits(&got_mrn_hv[r * du..(r + 1) * du]),
                    s_mrn_h[&r],
                    "hc_mix_reduce_norm h inv d={d} M={m} r={r}"
                );
                assert_eq!(
                    to_bits(&got_mrn_pv[r * hcu..(r + 1) * hcu]),
                    s_mrn_p[&r],
                    "hc_mix_reduce_norm p inv d={d} M={m} r={r}"
                );
                assert_eq!(
                    to_bits(&got_hn[r * du..(r + 1) * du]),
                    s_hn[&r],
                    "hc_head_norm inv d={d} M={m} r={r}"
                );
                assert_eq!(
                    to_bits(&got_pnhp[r * hd..(r + 1) * hd]),
                    s_pnhp[&r],
                    "post_norm_hc_post inv d={d} M={m} r={r}"
                );
                assert_eq!(
                    to_bits(&got_bx_v[r * hd..(r + 1) * hd]),
                    s_br_x[&r],
                    "hc_bridge x inv d={d} M={m} r={r}"
                );
                assert_eq!(
                    to_bits(&got_bh_v[r * du..(r + 1) * du]),
                    s_br_h[&r],
                    "hc_bridge h inv d={d} M={m} r={r}"
                );
                assert_eq!(
                    to_bits(&got_bp_v[r * hcu..(r + 1) * hcu]),
                    s_br_p[&r],
                    "hc_bridge p inv d={d} M={m} r={r}"
                );
                assert_eq!(
                    to_bits(&got_sh_v[r * du..(r + 1) * du]),
                    s_br_skip_h[&r],
                    "hc_bridge skip h inv d={d} M={m} r={r}"
                );
                assert_eq!(
                    to_bits(&got_bhead[r * du..(r + 1) * du]),
                    s_br_head[&r],
                    "hc_bridge_head inv d={d} M={m} r={r}"
                );
            }

            // Lazy Broadcast `res` (matches Layer-0 embedding fanout into multi-TG Hy4HcBridge).
            let res_bcast = ym
                .reshape(&[m as i64, 1, d])
                .unwrap()
                .broadcast_to(&[m as i64, hc, d])
                .unwrap();
            let res_dense = res_bcast.deep_copy().unwrap();
            let (bx_bcast, bh_bcast, bp_bcast) = hc_bridge(
                &ym,
                &res_bcast,
                &pm,
                &post_norm_w,
                &w_next,
                Some(&next_norm_w),
                norm_eps,
                &cfg,
            )
            .unwrap();
            let (bx_dense, bh_dense, bp_dense) = hc_bridge(
                &ym,
                &res_dense,
                &pm,
                &post_norm_w,
                &w_next,
                Some(&next_norm_w),
                norm_eps,
                &cfg,
            )
            .unwrap();
            assert_eq!(
                to_bits(&read(&bx_bcast)),
                to_bits(&read(&bx_dense)),
                "hc_bridge broadcast x_out d={d} M={m}"
            );
            assert_eq!(
                to_bits(&read(&bh_bcast)),
                to_bits(&read(&bh_dense)),
                "hc_bridge broadcast h_out d={d} M={m}"
            );
            assert_eq!(
                to_bits(&read(&bp_bcast)),
                to_bits(&read(&bp_dense)),
                "hc_bridge broadcast post_next d={d} M={m}"
            );
        }
    }
}

#[test]
fn mla_prep_q_kv_matches_unfused_and_is_row_invariant() {
    use crate::models::deepseek_v41::ops::inv_freqs;

    crate::models::hy_v4_flash::exact_f32();
    let eps = 1e-5f32;
    let max_err = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    for &(q_lora, kv_lora, rd, gate_w) in &[(1024i64, 512i64, 64i64, 4096i64), (6, 6, 4, 10)] {
        let kv_row = kv_lora + rd;
        let total = q_lora + kv_row + gate_w;
        let inv = inv_freqs(rd as usize, 10_000.0, None);
        let inv_arr = rope_inv_freqs(&inv).unwrap();
        let q_nw = MxArray::from_float32(
            &det_f32(3101 + q_lora as u64, q_lora as usize, 0.5)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
            &[q_lora],
        )
        .unwrap();
        let kv_nw = MxArray::from_float32(
            &det_f32(3201 + kv_lora as u64, kv_lora as usize, 0.5)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
            &[kv_lora],
        )
        .unwrap();

        let all_4 = MxArray::from_float32(
            &det_f32(3301 + total as u64, 4 * total as usize, 1.5),
            &[4, total],
        )
        .unwrap();
        let pos_all_u64 = [0u64, 17, 4095, 262_144];
        let pos_all_i32 = [0i32, 17, 4095, 262_144];

        // Single-row baselines for strict bitwise row invariance (out_f16=true, split=false/true).
        let mut s_qr = Vec::new();
        let mut s_kv_f16 = Vec::new();
        let mut s_k0 = Vec::new();
        let mut s_k1 = Vec::new();
        for (r, &pos_val) in pos_all_u64.iter().enumerate() {
            let row = all_4.slice(&[r as i64, 0], &[r as i64 + 1, total]).unwrap();
            let pos1 = rope_positions_f32(&[pos_val]).unwrap();
            let Some((qr1, kv1, _)) = try_mla_prep_q_kv(
                &row, &q_nw, &kv_nw, &pos1, &inv_arr, q_lora, kv_lora, rd, eps, true, false,
            )
            .unwrap() else {
                return;
            };
            let (qr1s, k0_1, k1_1) = try_mla_prep_q_kv(
                &row, &q_nw, &kv_nw, &pos1, &inv_arr, q_lora, kv_lora, rd, eps, true, true,
            )
            .unwrap()
            .unwrap();
            assert_eq!(to_bits(&read(&qr1)), to_bits(&read(&qr1s)));
            s_qr.push(to_bits(&read(&qr1)));
            s_kv_f16.push(kv1.to_uint16_native().unwrap().to_vec());
            s_k0.push(k0_1.to_uint16_native().unwrap().to_vec());
            s_k1.push(k1_1.unwrap().to_uint16_native().unwrap().to_vec());
        }

        for &m in &[1usize, 2, 4] {
            let sub = all_4.slice(&[0, 0], &[m as i64, total]).unwrap();
            let pos_f32 = rope_positions_f32(&pos_all_u64[..m]).unwrap();
            let pos_i32 = MxArray::from_int32(&pos_all_i32[..m], &[m as i64]).unwrap();

            // Unfused reference.
            let ref_qr = rms_norm(
                &sub.slice(&[0, 0], &[m as i64, q_lora]).unwrap(),
                &q_nw,
                eps,
            )
            .unwrap();
            let ref_kv_n = rms_norm(
                &sub.slice(&[0, q_lora], &[m as i64, q_lora + kv_lora])
                    .unwrap(),
                &kv_nw,
                eps,
            )
            .unwrap();
            let ref_kpe = fast_rope_with_pos(
                &sub.slice(&[0, q_lora + kv_lora], &[m as i64, q_lora + kv_row])
                    .unwrap(),
                &inv_arr,
                &pos_f32,
            )
            .unwrap();
            let ref_kv_f32 = MxArray::concatenate(&ref_kv_n, &ref_kpe, -1).unwrap();
            let ref_kv_f16 = ref_kv_f32.astype(DType::Float16).unwrap();

            for (pos_tag, pos_in) in [("f32", &pos_f32), ("i32", &pos_i32)] {
                // F32 output mode.
                let (qr, kv_f32, none1) = try_mla_prep_q_kv(
                    &sub, &q_nw, &kv_nw, pos_in, &inv_arr, q_lora, kv_lora, rd, eps, false, false,
                )
                .unwrap()
                .unwrap();
                assert!(none1.is_none());
                assert_eq!(shape(&qr), [m as i64, q_lora]);
                assert_eq!(shape(&kv_f32), [m as i64, kv_row]);
                assert_eq!(kv_f32.dtype().unwrap(), DType::Float32);
                assert!(
                    max_err(&read(&qr), &read(&ref_qr)) <= 1e-5,
                    "mla_prep qr f32 q={q_lora} M={m} pos={pos_tag}"
                );
                assert!(
                    max_err(&read(&kv_f32), &read(&ref_kv_f32)) <= 1e-5,
                    "mla_prep kv f32 q={q_lora} M={m} pos={pos_tag}"
                );

                // F16 unsplit output mode.
                let (qr_h, kv_h, _) = try_mla_prep_q_kv(
                    &sub, &q_nw, &kv_nw, pos_in, &inv_arr, q_lora, kv_lora, rd, eps, true, false,
                )
                .unwrap()
                .unwrap();
                assert_eq!(kv_h.dtype().unwrap(), DType::Float16);
                assert!(
                    max_err(&read(&kv_h), &read(&ref_kv_f16)) <= 1e-5,
                    "mla_prep kv f16 q={q_lora} M={m} pos={pos_tag}"
                );

                // F16 split output mode.
                let (qr_s, k0, k1_opt) = try_mla_prep_q_kv(
                    &sub, &q_nw, &kv_nw, pos_in, &inv_arr, q_lora, kv_lora, rd, eps, true, true,
                )
                .unwrap()
                .unwrap();
                let k1 = k1_opt.unwrap();
                let half = kv_row / 2;
                assert_eq!(shape(&k0), [m as i64, half]);
                assert_eq!(shape(&k1), [m as i64, half]);
                assert_eq!(k0.dtype().unwrap(), DType::Float16);
                assert_eq!(k1.dtype().unwrap(), DType::Float16);
                let joined = MxArray::concatenate(&k0, &k1, -1).unwrap();
                assert_eq!(
                    joined.to_uint16_native().unwrap().to_vec(),
                    kv_h.to_uint16_native().unwrap().to_vec(),
                    "mla_prep split matches unsplit bit-exact q={q_lora} M={m} pos={pos_tag}"
                );

                // Bitwise row invariance against M=1.
                let qr_v = read(&qr_s);
                let kv_bits = kv_h.to_uint16_native().unwrap().to_vec();
                let k0_bits = k0.to_uint16_native().unwrap().to_vec();
                let k1_bits = k1.to_uint16_native().unwrap().to_vec();
                let ql_u = q_lora as usize;
                let kvr_u = kv_row as usize;
                let h_u = half as usize;
                for r in 0..m {
                    assert_eq!(
                        to_bits(&qr_v[r * ql_u..(r + 1) * ql_u]),
                        s_qr[r],
                        "mla_prep qr inv q={q_lora} M={m} r={r} pos={pos_tag}"
                    );
                    assert_eq!(to_bits(&read(&qr_h)[r * ql_u..(r + 1) * ql_u]), s_qr[r]);
                    assert_eq!(
                        &kv_bits[r * kvr_u..(r + 1) * kvr_u],
                        s_kv_f16[r].as_slice(),
                        "mla_prep kv inv q={q_lora} M={m} r={r} pos={pos_tag}"
                    );
                    assert_eq!(
                        &k0_bits[r * h_u..(r + 1) * h_u],
                        s_k0[r].as_slice(),
                        "mla_prep k0 inv q={q_lora} M={m} r={r} pos={pos_tag}"
                    );
                    assert_eq!(
                        &k1_bits[r * h_u..(r + 1) * h_u],
                        s_k1[r].as_slice(),
                        "mla_prep k1 inv q={q_lora} M={m} r={r} pos={pos_tag}"
                    );
                }
            }
        }
    }
}

#[test]
fn q8_grouped_k192_rope_matches_unfused_and_is_row_invariant() {
    use crate::models::deepseek_v41::ops::inv_freqs;
    use crate::models::hy_v4_flash::attention::grouped_mm_t_mgh;
    use crate::models::hy_v4_flash::weights::quantize_sym_q8_array;

    crate::models::hy_v4_flash::exact_f32();
    let max_err = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    let nope = 192i64;
    let rd = 64i64;
    let qk = nope + rd;
    let inv = inv_freqs(rd as usize, 10_000.0, None);
    let inv_arr = rope_inv_freqs(&inv).unwrap();
    let pos_all_u64 = [0u64, 23, 509, 262_144];
    let pos_all_i32 = [0i32, 23, 509, 262_144];

    // Covers real Hy4 (H=32, N=512) and N < 16 edge case (H=2, N=8).
    for &(heads, n_out) in &[(32i64, 512i64), (2, 8)] {
        let kb_f32 = MxArray::from_float32(
            &det_f32(3401 + heads as u64, (heads * n_out * nope) as usize, 0.2),
            &[heads * n_out, nope],
        )
        .unwrap();
        let kb_sym = quantize_sym_q8_array(&kb_f32).unwrap();
        let Dense::SymQ8 { w, scales } = &kb_sym else {
            unreachable!();
        };
        let scales_f32 = scales.astype(DType::Float32).unwrap();
        let biases_f32 = scales_f32.mul_scalar(-128.0).unwrap();
        let kb_qmm = Dense::Qmm {
            w: w.clone(),
            scales: scales_f32,
            biases: biases_f32,
        };

        let qb_4 = MxArray::from_float32(
            &det_f32(3501 + heads as u64, (4 * heads * qk) as usize, 1.0),
            &[4, heads, qk],
        )
        .unwrap();

        for (tag, kb) in [("sym_q8", &kb_sym), ("qmm_f32", &kb_qmm)] {
            let mut single_rows = Vec::new();
            for (r, &pos_val) in pos_all_u64.iter().enumerate() {
                let row = qb_4
                    .slice(&[r as i64, 0, 0], &[r as i64 + 1, heads, qk])
                    .unwrap();
                let pos1 = rope_positions_f32(&[pos_val]).unwrap();
                let Some(q1) = try_q8_grouped_k192_rope(&row, kb, heads, &pos1, &inv_arr).unwrap()
                else {
                    return;
                };
                single_rows.push(to_bits(&read(&q1)));
            }

            let out_n = (n_out + rd) as usize;
            for &m in &[1usize, 2, 4] {
                let sub = qb_4.slice(&[0, 0, 0], &[m as i64, heads, qk]).unwrap();
                let pos_f32 = rope_positions_f32(&pos_all_u64[..m]).unwrap();
                let pos_i32 = MxArray::from_int32(&pos_all_i32[..m], &[m as i64]).unwrap();

                let ref_nope = sub.slice(&[0, 0, 0], &[m as i64, heads, nope]).unwrap();
                let ref_pe = fast_rope_with_pos(
                    &sub.slice(&[0, 0, nope], &[m as i64, heads, qk]).unwrap(),
                    &inv_arr,
                    &pos_f32,
                )
                .unwrap();
                let ref_abs = grouped_mm_t_mgh(kb, &ref_nope, heads).unwrap();
                let ref_q = read(&MxArray::concatenate(&ref_abs, &ref_pe, -1).unwrap());

                for (ptag, pin) in [("f32", &pos_f32), ("i32", &pos_i32)] {
                    let got = try_q8_grouped_k192_rope(&sub, kb, heads, pin, &inv_arr)
                        .unwrap()
                        .unwrap();
                    assert_eq!(shape(&got), [m as i64, heads, n_out + rd]);
                    let got_v = read(&got);
                    let err = max_err(&got_v, &ref_q);
                    assert!(
                        err <= 1e-5,
                        "q8_grouped_k192_rope {tag} H={heads} N={n_out} M={m} pos={ptag}: err={err:e}"
                    );
                    let stride = heads as usize * out_n;
                    for r in 0..m {
                        assert_eq!(
                            to_bits(&got_v[r * stride..(r + 1) * stride]),
                            single_rows[r],
                            "q8_grouped_k192_rope inv {tag} H={heads} M={m} r={r} pos={ptag}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn kpool_compress_and_partial_rope_tail_match_unfused() {
    use crate::models::deepseek_v41::ops::inv_freqs;

    crate::models::hy_v4_flash::exact_f32();
    let eps = 1e-5f32;
    let d = 128i64;
    let rd = 64i64;
    let inv = inv_freqs(rd as usize, 1e7, None);
    let inv_arr = rope_inv_freqs(&inv).unwrap();
    let max_err = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    let w_val = MxArray::from_float32(&det_f32(3601, (d * d) as usize, 0.2), &[d, d]).unwrap();
    let w_gate = MxArray::from_float32(&det_f32(3602, (d * d) as usize, 0.2), &[d, d]).unwrap();
    let ape = MxArray::from_float32(&det_f32(3603, (4 * d) as usize, 0.5), &[4, d]).unwrap();
    let comp_norm = MxArray::from_float32(
        &det_f32(3604, d as usize, 0.5)
            .into_iter()
            .map(|v| 1.0 + v)
            .collect::<Vec<_>>(),
        &[d],
    )
    .unwrap();

    let ref_pool = |grp: &MxArray, first: usize| -> Vec<f32> {
        let k = grp.shape_at(0).unwrap() as usize;
        let v = grp
            .matmul(&w_val.transpose(Some(&[1, 0])).unwrap())
            .unwrap();
        let gt = grp
            .matmul(&w_gate.transpose(Some(&[1, 0])).unwrap())
            .unwrap()
            .add(&ape)
            .unwrap();
        let mx = gt.max(Some(&[1]), Some(true)).unwrap();
        let e = gt.sub(&mx).unwrap().exp().unwrap();
        let p = e.div(&e.sum(Some(&[1]), Some(true)).unwrap()).unwrap();
        let pooled = v.mul(&p).unwrap().sum(Some(&[1]), Some(false)).unwrap();
        let normed = rms_norm(&pooled, &comp_norm, eps).unwrap();
        let pos: Vec<u64> = (first..first + k).map(|g| (g * 4) as u64).collect();
        let pos_arr = rope_positions_f32(&pos).unwrap();
        let head = normed.slice(&[0, 0], &[k as i64, d - rd]).unwrap();
        let tail = normed.slice(&[0, d - rd], &[k as i64, d]).unwrap();
        let rot = fast_rope_with_pos(&tail, &inv_arr, &pos_arr).unwrap();
        read(&MxArray::concatenate(&head, &rot, -1).unwrap())
    };

    // Exercise circular ring wrap-around, open_len 0..=3, n_groups 1, 2, 4, and f32/f16/bf16 outputs.
    let tail_cap = 10usize;
    let ring_rows = det_f32(3605, tail_cap * d as usize, 1.0);
    let ring_arr = MxArray::from_float32(&ring_rows, &[tail_cap as i64, d]).unwrap();
    let tok_rows = det_f32(3606, 16 * d as usize, 1.0);
    let tok_arr = MxArray::from_float32(&tok_rows, &[16, d]).unwrap();

    for &(n_groups, first_group, open_len, start_slot) in &[
        (1usize, 0usize, 0usize, 0usize),
        (1, 17, 1, 9),
        (1, 1024, 2, 9),
        (1, 65_535, 3, 8),
        (2, 5, 3, 9),
        (4, 100, 2, 9),
    ] {
        let mut linear = Vec::with_capacity(n_groups * 4 * d as usize);
        for t in 0..n_groups * 4 {
            if t < open_len {
                let slot = (start_slot + t) % tail_cap;
                linear.extend_from_slice(&ring_rows[slot * d as usize..(slot + 1) * d as usize]);
            } else {
                let tr = t - open_len;
                linear.extend_from_slice(&tok_rows[tr * d as usize..(tr + 1) * d as usize]);
            }
        }
        let grp = MxArray::from_float32(&linear, &[n_groups as i64, 4, d]).unwrap();
        let want_f32 = ref_pool(&grp, first_group);

        let Some(got_f32) = try_kpool_compress(
            (open_len > 0).then_some(&ring_arr),
            &tok_arr,
            &w_val,
            &w_gate,
            &ape,
            &comp_norm,
            &inv_arr,
            n_groups,
            first_group,
            open_len,
            start_slot,
            tail_cap,
            eps,
            DType::Float32,
        )
        .unwrap() else {
            return;
        };
        assert_eq!(shape(&got_f32), [n_groups as i64, d]);
        let err = max_err(&read(&got_f32), &want_f32);
        assert!(
            err <= 1e-5,
            "kpool_compress f32 ng={n_groups} fg={first_group} open={open_len}: err={err:e}"
        );

        for dt in [DType::Float16, DType::BFloat16] {
            let got_dt = try_kpool_compress(
                (open_len > 0).then_some(&ring_arr),
                &tok_arr,
                &w_val,
                &w_gate,
                &ape,
                &comp_norm,
                &inv_arr,
                n_groups,
                first_group,
                open_len,
                start_slot,
                tail_cap,
                eps,
                dt,
            )
            .unwrap()
            .unwrap();
            assert_eq!(got_dt.dtype().unwrap(), dt);
            let want_dt = read(&got_f32.astype(dt).unwrap());
            assert_eq!(
                to_bits(&read(&got_dt)),
                to_bits(&want_dt),
                "kpool_compress cast {dt:?} ng={n_groups} open={open_len}"
            );
        }
    }

    // Fused partial RoPE tail on [M, 128] and [M, 16, 128] across M = 1, 2, 4.
    let pos_all = [0u64, 7, 1024, 262_144];
    let x3_all = MxArray::from_float32(&det_f32(3610, 4 * 16 * 128, 1.5), &[4, 16, 128]).unwrap();
    let mut single3 = Vec::new();
    for (r, &pos_val) in pos_all.iter().enumerate() {
        let row = x3_all
            .slice(&[r as i64, 0, 0], &[r as i64 + 1, 16, 128])
            .unwrap();
        let pos1 = rope_positions_f32(&[pos_val]).unwrap();
        let y1 = try_partial_rope_tail(&row, &pos1, &inv_arr)
            .unwrap()
            .unwrap();
        single3.push(to_bits(&read(&y1)));
    }
    for &m in &[1usize, 2, 4] {
        let xm = x3_all.slice(&[0, 0, 0], &[m as i64, 16, 128]).unwrap();
        let pos_m = rope_positions_f32(&pos_all[..m]).unwrap();
        let got = try_partial_rope_tail(&xm, &pos_m, &inv_arr)
            .unwrap()
            .unwrap();
        let head = xm.slice(&[0, 0, 0], &[m as i64, 16, 64]).unwrap();
        let tail = xm.slice(&[0, 0, 64], &[m as i64, 16, 128]).unwrap();
        let want = read(
            &MxArray::concatenate(
                &head,
                &fast_rope_with_pos(&tail, &inv_arr, &pos_m).unwrap(),
                -1,
            )
            .unwrap(),
        );
        let got_v = read(&got);
        let err = max_err(&got_v, &want);
        assert!(err <= 1e-5, "partial_rope_tail 3D M={m}: err={err:e}");
        for r in 0..m {
            assert_eq!(
                to_bits(&got_v[r * 16 * 128..(r + 1) * 16 * 128]),
                single3[r],
                "partial_rope_tail 3D inv M={m} r={r}"
            );
        }
    }

    // Strided q_pe slice (head_w=0, rope_dim=64) in both FP32 and FP16 IO.
    for &(m, h) in &[(17usize, 32usize), (32, 64)] {
        let pos_v: Vec<u64> = (0..m).map(|i| 100 + i as u64).collect();
        let pos_m = rope_positions_f32(&pos_v).unwrap();
        let q_raw = MxArray::from_float32(
            &det_f32(3690 + h as u64, m * h * 256, 1.5),
            &[m as i64, h as i64, 256],
        )
        .unwrap();
        let q_pe_f32 = q_raw
            .slice(&[0, 0, 192], &[m as i64, h as i64, 256])
            .unwrap();
        let got_f32 = try_partial_rope_tail_out(&q_pe_f32, &pos_m, &inv_arr, false)
            .unwrap()
            .unwrap();
        let want_f32 = fast_rope_with_pos(&q_pe_f32, &inv_arr, &pos_m).unwrap();
        let err_f32 = max_err(&read(&got_f32), &read(&want_f32));
        assert!(
            err_f32 <= 1e-5,
            "strided q_pe partial_rope_tail F32 M={m} H={h}: err={err_f32:e}"
        );

        let q_raw_f16 = q_raw.astype(DType::Float16).unwrap();
        let q_pe_f16 = q_raw_f16
            .slice(&[0, 0, 192], &[m as i64, h as i64, 256])
            .unwrap();
        let got_f16 = try_partial_rope_tail_out(&q_pe_f16, &pos_m, &inv_arr, true)
            .unwrap()
            .unwrap();
        assert_eq!(got_f16.dtype().unwrap(), DType::Float16);
        let want_f16 =
            fast_rope_with_pos(&q_pe_f16.astype(DType::Float32).unwrap(), &inv_arr, &pos_m)
                .unwrap()
                .astype(DType::Float16)
                .unwrap();
        let err_f16 = max_err(&read(&got_f16), &read(&want_f16));
        assert!(
            err_f16 <= 2e-3,
            "strided q_pe partial_rope_tail F16 M={m} H={h}: err={err_f16:e}"
        );
    }
}

#[test]
fn engram_fused_norm_and_gate_add_match_unfused_and_are_row_invariant() {
    crate::models::hy_v4_flash::exact_f32();
    let max_err = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    // Real Hy4 (HC=4, D=3072, COLS=16, HD=96 -> USE_REG=true), small VEC4 (D=8), and scalar (D=66).
    for &(hc, d, cols, hd) in &[(4i64, 3072i64, 16i64, 96i64), (4, 8, 2, 4), (4, 66, 4, 12)] {
        let lookup_norm = MxArray::from_float32(
            &det_f32(3701 + d as u64, (cols * hd) as usize, 0.5)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
            &[cols, hd],
        )
        .unwrap();
        let kv_b = MxArray::from_float32(
            &det_f32(3702 + d as u64, ((hc + 1) * d) as usize, 0.25),
            &[hc + 1, d],
        )
        .unwrap();
        let gate_w =
            MxArray::from_float32(&det_f32(3703 + d as u64, (hc * d) as usize, 0.5), &[hc, d])
                .unwrap();
        let pre_gamma = MxArray::from_float32(
            &det_f32(3704 + d as u64, d as usize, 0.5)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
            &[d],
        )
        .unwrap();
        let hc_scale =
            MxArray::from_float32(&det_f32(3705 + d as u64, (hc * d) as usize, 0.5), &[hc, d])
                .unwrap();

        let rows_4 = MxArray::from_float32(
            &det_f32(3706 + d as u64, (4 * cols * hd) as usize, 1.5),
            &[4, cols, hd],
        )
        .unwrap();
        let x_4 = MxArray::from_float32(
            &det_f32(3707 + d as u64, (4 * hc * d) as usize, 1.5),
            &[4, hc, d],
        )
        .unwrap();
        let kv_raw_4 = MxArray::from_float32(
            &det_f32(3708 + d as u64, (4 * (hc + 1) * d) as usize, 1.2),
            &[4, (hc + 1) * d],
        )
        .unwrap();

        // Single-row baselines for strict bitwise row invariance.
        let mut s_nr = Vec::new();
        let mut s_out = Vec::new();
        for r in 0..4usize {
            let rr = rows_4
                .slice(&[r as i64, 0, 0], &[r as i64 + 1, cols, hd])
                .unwrap();
            let xr = x_4
                .slice(&[r as i64, 0, 0], &[r as i64 + 1, hc, d])
                .unwrap();
            let kvr = kv_raw_4
                .slice(&[r as i64, 0], &[r as i64 + 1, (hc + 1) * d])
                .unwrap();
            let Some(nr1) = try_engram_norm_rows(&rr, &lookup_norm, ENGRAM_EPS).unwrap() else {
                return;
            };
            let out1 = try_engram_gate_add(
                &xr, &kvr, &kv_b, &gate_w, &pre_gamma, &hc_scale, None, ENGRAM_EPS,
            )
            .unwrap()
            .unwrap();
            s_nr.push(to_bits(&read(&nr1)));
            s_out.push(to_bits(&read(&out1)));
        }

        for &m in &[1usize, 2, 4] {
            let rm = rows_4.slice(&[0, 0, 0], &[m as i64, cols, hd]).unwrap();
            let xm = x_4.slice(&[0, 0, 0], &[m as i64, hc, d]).unwrap();
            let kvm = kv_raw_4.slice(&[0, 0], &[m as i64, (hc + 1) * d]).unwrap();

            // 1. normed_rows parity + invariance.
            let got_nr = try_engram_norm_rows(&rm, &lookup_norm, ENGRAM_EPS)
                .unwrap()
                .unwrap();
            let ref_nr = read(
                &rms_norm_unweighted(&rm, ENGRAM_EPS)
                    .unwrap()
                    .mul(&lookup_norm)
                    .unwrap()
                    .reshape(&[m as i64, cols * hd])
                    .unwrap(),
            );
            let got_nr_v = read(&got_nr);
            assert!(
                max_err(&got_nr_v, &ref_nr) <= 1e-5,
                "engram_norm_rows d={d} M={m}"
            );
            let nr_stride = (cols * hd) as usize;
            for r in 0..m {
                assert_eq!(
                    to_bits(&got_nr_v[r * nr_stride..(r + 1) * nr_stride]),
                    s_nr[r],
                    "engram_norm_rows inv d={d} M={m} r={r}"
                );
            }

            // 2. engram_gate_add unmasked parity + invariance.
            let kv = kvm
                .reshape(&[m as i64, hc + 1, d])
                .unwrap()
                .add(&kv_b)
                .unwrap();
            let key = kv.slice(&[0, 0, 0], &[m as i64, hc, d]).unwrap();
            let value = kv
                .slice(&[0, hc, 0], &[m as i64, hc + 1, d])
                .unwrap()
                .reshape(&[m as i64, d])
                .unwrap();
            let ref_gate = stream_gate(&stream_dot(&xm, &key, &gate_w).unwrap()).unwrap();
            let ref_scale = ref_gate
                .reshape(&[m as i64, hc, 1])
                .unwrap()
                .add(&hc_scale)
                .unwrap();
            let ref_sig = signal(&value, &pre_gamma)
                .unwrap()
                .reshape(&[m as i64, 1, d])
                .unwrap();
            let ref_out = read(&xm.add(&ref_sig.mul(&ref_scale).unwrap()).unwrap());

            let got_out = try_engram_gate_add(
                &xm, &kvm, &kv_b, &gate_w, &pre_gamma, &hc_scale, None, ENGRAM_EPS,
            )
            .unwrap()
            .unwrap();
            let got_out_v = read(&got_out);
            let err = max_err(&got_out_v, &ref_out);
            assert!(err <= 1e-5, "engram_gate_add d={d} M={m}: err={err:e}");
            let row_stride = (hc * d) as usize;
            for r in 0..m {
                assert_eq!(
                    to_bits(&got_out_v[r * row_stride..(r + 1) * row_stride]),
                    s_out[r],
                    "engram_gate_add inv d={d} M={m} r={r}"
                );
            }

            // 3. Masked rows stay bit-identical to x; kept rows match unmasked bit-for-bit.
            let mask_v: Vec<u8> = (0..m).map(|r| u8::from(r % 2 == 0)).collect();
            let mask_arr = MxArray::from_uint8(&mask_v, &[m as i64]).unwrap();
            let masked_out = read(
                &try_engram_gate_add(
                    &xm,
                    &kvm,
                    &kv_b,
                    &gate_w,
                    &pre_gamma,
                    &hc_scale,
                    Some(&mask_arr),
                    ENGRAM_EPS,
                )
                .unwrap()
                .unwrap(),
            );
            let xm_v = read(&xm);
            for (r, &keep) in mask_v.iter().enumerate() {
                let lo = r * row_stride;
                let hi = (r + 1) * row_stride;
                if keep == 0 {
                    assert_eq!(
                        to_bits(&masked_out[lo..hi]),
                        to_bits(&xm_v[lo..hi]),
                        "engram_gate_add masked row d={d} M={m} r={r}"
                    );
                } else {
                    assert_eq!(
                        to_bits(&masked_out[lo..hi]),
                        to_bits(&got_out_v[lo..hi]),
                        "engram_gate_add kept row d={d} M={m} r={r}"
                    );
                }
            }
        }
    }

    // Stress M = 512, HC = 4, D = 3072 across repeated launches to catch any
    // threadgroup reduction race on tg_red (including negative gate dot sums).
    let (hc, d, m) = (4i64, 3072i64, 512i64);
    let kv_b =
        MxArray::from_float32(&det_f32(3801, ((hc + 1) * d) as usize, 0.25), &[hc + 1, d]).unwrap();
    let gate_w = MxArray::from_float32(
        &det_f32(3802, (hc * d) as usize, 1.0)
            .into_iter()
            .map(|v| -v.abs() - 0.5)
            .collect::<Vec<_>>(),
        &[hc, d],
    )
    .unwrap();
    let pre_gamma = MxArray::from_float32(
        &det_f32(3803, d as usize, 0.5)
            .into_iter()
            .map(|v| 1.0 + v)
            .collect::<Vec<_>>(),
        &[d],
    )
    .unwrap();
    let hc_scale = MxArray::from_float32(&det_f32(3804, (hc * d) as usize, 0.5), &[hc, d]).unwrap();
    let xm =
        MxArray::from_float32(&det_f32(3805, (m * hc * d) as usize, 1.5), &[m, hc, d]).unwrap();
    let kvm = MxArray::from_float32(
        &det_f32(3806, (m * (hc + 1) * d) as usize, 1.2),
        &[m, (hc + 1) * d],
    )
    .unwrap();
    let kv = kvm.reshape(&[m, hc + 1, d]).unwrap().add(&kv_b).unwrap();
    let key = kv.slice(&[0, 0, 0], &[m, hc, d]).unwrap();
    let value = kv
        .slice(&[0, hc, 0], &[m, hc + 1, d])
        .unwrap()
        .reshape(&[m, d])
        .unwrap();
    let ref_gate = stream_gate(&stream_dot(&xm, &key, &gate_w).unwrap()).unwrap();
    let ref_scale = ref_gate
        .reshape(&[m, hc, 1])
        .unwrap()
        .add(&hc_scale)
        .unwrap();
    let ref_sig = signal(&value, &pre_gamma)
        .unwrap()
        .reshape(&[m, 1, d])
        .unwrap();
    let ref_out = read(&xm.add(&ref_sig.mul(&ref_scale).unwrap()).unwrap());

    let mut first_bits: Option<Vec<u32>> = None;
    for rep in 0..8 {
        let got = read(
            &try_engram_gate_add(
                &xm, &kvm, &kv_b, &gate_w, &pre_gamma, &hc_scale, None, ENGRAM_EPS,
            )
            .unwrap()
            .unwrap(),
        );
        assert!(
            got.iter().all(|v| v.is_finite()),
            "engram_gate_add stress rep={rep} produced non-finite value"
        );
        let err = max_err(&got, &ref_out);
        assert!(
            err <= 1e-5,
            "engram_gate_add stress M=512 rep={rep}: err={err:e}"
        );
        let bits = to_bits(&got);
        if let Some(prev) = &first_bits {
            assert_eq!(
                bits, *prev,
                "engram_gate_add stress M=512 rep={rep} diverged from rep=0"
            );
        } else {
            first_bits = Some(bits);
        }
    }
}

#[test]
#[ignore = "microbenchmark for hc_bridge 1-TG vs multi-TG"]
fn bench_hc_bridge_latency() {
    crate::models::hy_v4_flash::exact_f32();
    let hc = 4i64;
    let d = 3072i64;
    let hd = (hc * d) as usize;
    let cfg = HcConfig {
        count: hc as usize,
        eps: 1e-6,
        magnitude: 2.0,
    };
    let n_cold = 400usize; // 400 * 393 KiB = 157 MB > SLC
    let weights: Vec<(HcWeights, MxArray, MxArray)> = (0..n_cold)
        .map(|l| {
            let scale_w = 1.0 / (hd as f32).sqrt();
            let w = HcWeights {
                weight: MxArray::from_float32(
                    &det_f32(5000 + l as u64, (2 * hc) as usize * hd, scale_w),
                    &[2 * hc, hc * d],
                )
                .unwrap(),
                scale: MxArray::from_float32(&[0.75, 1.25], &[2]).unwrap(),
                base: MxArray::from_float32(
                    &det_f32(6000 + l as u64, (2 * hc) as usize, 0.5),
                    &[2 * hc],
                )
                .unwrap(),
            };
            let pnw = MxArray::from_float32(&vec![1.0f32; d as usize], &[d]).unwrap();
            let nnw = MxArray::from_float32(&vec![1.0f32; d as usize], &[d]).unwrap();
            (w, pnw, nnw)
        })
        .collect();
    // Materialize weights once and boost GPU clocks before timing.
    let w_refs: Vec<&MxArray> = weights.iter().map(|(w, _, _)| &w.weight).collect();
    MxArray::eval_arrays_with_context(&w_refs, "bench weights").unwrap();

    for (label, slice_len, iters) in [("warm83", 83usize, 30usize), ("cold400", n_cold, 10usize)] {
        let w_slice = &weights[..slice_len];
        for &m in &[1usize, 2, 4] {
            let x0 =
                MxArray::from_float32(&det_f32(7000 + m as u64, m * hd, 1.0), &[m as i64, hc, d])
                    .unwrap();
            let y0 = MxArray::from_float32(
                &det_f32(8000 + m as u64, m * d as usize, 1.0),
                &[m as i64, d],
            )
            .unwrap();
            let p0 =
                MxArray::from_float32(&vec![0.5f32; m * hc as usize], &[m as i64, hc]).unwrap();
            MxArray::eval_arrays_with_context(&[&x0, &y0, &p0], "bench warmup inputs").unwrap();

            let run_once = || {
                let mut x = x0.clone();
                let mut y = y0.clone();
                let mut p = p0.clone();
                for (w, pnw, nnw) in w_slice {
                    let (nx, nh, np) =
                        hc_bridge(&y, &x, &p, pnw, w, Some(nnw), 1e-5, &cfg).unwrap();
                    x = nx;
                    y = nh;
                    p = np;
                }
                MxArray::eval_arrays_with_context(&[&x, &y, &p], "bench hc_bridge").unwrap();
            };
            for _ in 0..8 {
                run_once();
            }
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = std::time::Instant::now();
                run_once();
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med_ms = times[times.len() / 2];
            let per_us = med_ms * 1000.0 / slice_len as f64;
            let eq83_ms = per_us * 83.0 / 1000.0;
            eprintln!(
                "hc_bridge {label} M={m} splits={}: {per_us:.2} us/call (83x = {eq83_ms:.3} ms)",
                std::env::var("MLX_HY4_HC_BRIDGE_SPLITS").unwrap_or_else(|_| "default".into()),
            );
        }
    }
}

#[test]
fn f32_head_weights_matches_unfused_and_is_row_invariant() {
    crate::models::hy_v4_flash::exact_f32();
    let max_err = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let to_bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|&x| x.to_bits()).collect() };

    for &(n, k) in &[(16usize, 3072usize), (2, 12), (32, 128)] {
        let scale = ((n * 128) as f32).sqrt().recip();
        let w = Dense::Array(
            MxArray::from_float32(
                &det_f32(9101 + k as u64, n * k, 0.25),
                &[n as i64, k as i64],
            )
            .unwrap(),
        );
        let x_all =
            MxArray::from_float32(&det_f32(9201 + k as u64, 16 * k, 1.0), &[16, k as i64]).unwrap();
        let mut row1: Vec<Vec<u32>> = Vec::new();
        for r in 0..16i64 {
            let xr = x_all.slice(&[r, 0], &[r + 1, k as i64]).unwrap();
            let Some(got) = try_f32_head_weights(&xr, &w, scale).unwrap() else {
                eprintln!("skipping f32_head_weights Metal test (no GPU)");
                return;
            };
            row1.push(to_bits(&read(&got)));
        }
        for &m in &[1usize, 2, 4, 8, 16] {
            let xm = x_all.slice(&[0, 0], &[m as i64, k as i64]).unwrap();
            let got = read(&try_f32_head_weights(&xm, &w, scale).unwrap().unwrap());
            let want = read(
                &linear(&xm, &w)
                    .unwrap()
                    .mul_scalar(f64::from(scale))
                    .unwrap(),
            );
            let err = max_err(&got, &want);
            assert!(err <= 1e-4, "f32_head_weights N={n} K={k} M={m}: err={err}");
            for r in 0..m {
                assert_eq!(
                    to_bits(&got[r * n..(r + 1) * n]),
                    row1[r],
                    "f32_head_weights invariance N={n} K={k} M={m} r={r}"
                );
            }
        }
    }
}

#[test]
fn mtp_fused_norms_match_unfused_and_are_row_invariant() {
    crate::models::hy_v4_flash::exact_f32();
    let eps = 1e-5f32;
    let max_err = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let to_bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|&x| x.to_bits()).collect() };

    for &d in &[3072i64, 192] {
        let du = d as usize;
        let enorm = MxArray::from_float32(
            &det_f32(9301 + d as u64, du, 0.5)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
            &[d],
        )
        .unwrap();
        let hnorm = MxArray::from_float32(
            &det_f32(9302 + d as u64, du, 0.5)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
            &[d],
        )
        .unwrap();
        let e_all =
            MxArray::from_float32(&det_f32(9303 + d as u64, 17 * du, 1.5), &[17, d]).unwrap();
        let h_all =
            MxArray::from_float32(&det_f32(9304 + d as u64, 17 * du, 1.5), &[17, d]).unwrap();

        for dt in [DType::Float32, DType::Float16, DType::BFloat16] {
            let e_dt = e_all.astype(dt).unwrap();
            let h_dt = h_all.astype(dt).unwrap();
            let mut eh_r1 = Vec::new();
            for r in 0..4i64 {
                let er = e_dt.slice(&[r, 0], &[r + 1, d]).unwrap();
                let hr = h_dt.slice(&[r, 0], &[r + 1, d]).unwrap();
                let Some(got) = try_mtp_eh_norm_concat(&er, &hr, &enorm, &hnorm, eps).unwrap()
                else {
                    eprintln!("skipping mtp_fused_norms Metal test (no GPU)");
                    return;
                };
                eh_r1.push(to_bits(&read(&got)));
            }
            for &m in &[1usize, 2, 4, 17] {
                let em = e_dt.slice(&[0, 0], &[m as i64, d]).unwrap();
                let hm = h_dt.slice(&[0, 0], &[m as i64, d]).unwrap();
                let got = read(
                    &try_mtp_eh_norm_concat(&em, &hm, &enorm, &hnorm, eps)
                        .unwrap()
                        .unwrap(),
                );
                let ref_a = rms_norm(&em.astype(DType::Float32).unwrap(), &enorm, eps).unwrap();
                let ref_b = rms_norm(&hm.astype(DType::Float32).unwrap(), &hnorm, eps).unwrap();
                let want = read(&MxArray::concatenate(&ref_a, &ref_b, 1).unwrap());
                let err = max_err(&got, &want);
                assert!(
                    err <= 1e-5,
                    "eh_norm_concat d={d} dt={dt:?} M={m}: err={err}"
                );
                for r in 0..m.min(4) {
                    assert_eq!(
                        to_bits(&got[r * 2 * du..(r + 1) * 2 * du]),
                        eh_r1[r],
                        "eh_norm_concat invariance d={d} dt={dt:?} M={m} r={r}"
                    );
                }
            }
        }

        let mut pn_x1 = Vec::new();
        let mut pn_h1 = Vec::new();
        for r in 0..4i64 {
            let rr = e_all.slice(&[r, 0], &[r + 1, d]).unwrap();
            let yr = h_all.slice(&[r, 0], &[r + 1, d]).unwrap();
            let (Some(xo), ho) = try_post_norm_add_rms_norm(&rr, &yr, &enorm, &hnorm, eps, true)
                .unwrap()
                .unwrap()
            else {
                panic!("need_x_out=true must return x_out");
            };
            pn_x1.push(to_bits(&read(&xo)));
            pn_h1.push(to_bits(&read(&ho)));
        }
        for &m in &[1usize, 2, 4, 17] {
            let rm = e_all.slice(&[0, 0], &[m as i64, d]).unwrap();
            let ym = h_all.slice(&[0, 0], &[m as i64, d]).unwrap();
            let (Some(xo), ho) = try_post_norm_add_rms_norm(&rm, &ym, &enorm, &hnorm, eps, true)
                .unwrap()
                .unwrap()
            else {
                panic!("need_x_out=true must return x_out");
            };
            let (xo_none, ho_only) =
                try_post_norm_add_rms_norm(&rm, &ym, &enorm, &hnorm, eps, false)
                    .unwrap()
                    .unwrap();
            assert!(xo_none.is_none());
            let ref_x = rm.add(&rms_norm(&ym, &enorm, eps).unwrap()).unwrap();
            let ref_h = rms_norm(&ref_x, &hnorm, eps).unwrap();
            let gx = read(&xo);
            let gh = read(&ho);
            let gh2 = read(&ho_only);
            assert!(
                max_err(&gx, &read(&ref_x)) <= 1e-5,
                "post_norm x d={d} M={m}"
            );
            assert!(
                max_err(&gh, &read(&ref_h)) <= 1e-5,
                "post_norm h d={d} M={m}"
            );
            assert_eq!(to_bits(&gh), to_bits(&gh2), "need_x_out false matches true");
            for r in 0..m.min(4) {
                assert_eq!(
                    to_bits(&gx[r * du..(r + 1) * du]),
                    pn_x1[r],
                    "post_norm x inv d={d} M={m} r={r}"
                );
                assert_eq!(
                    to_bits(&gh[r * du..(r + 1) * du]),
                    pn_h1[r],
                    "post_norm h inv d={d} M={m} r={r}"
                );
            }
        }
    }
}

#[test]
fn hc_bridge_prefill_reg_and_row2_match_unfused_and_row1() {
    crate::models::hy_v4_flash::exact_f32();
    let hc = 4i64;
    let d = 3072i64;
    let hd = (hc * d) as usize;
    let du = d as usize;
    let hcu = hc as usize;
    let norm_eps = 1e-5f32;
    let hc_eps = 1e-6f32;
    let cfg = HcConfig {
        count: hcu,
        eps: hc_eps,
        magnitude: 2.0,
    };
    let max_err = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let to_bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|&x| x.to_bits()).collect() };

    let w_next = HcWeights {
        weight: MxArray::from_float32(
            &det_f32(9401, (2 * hc) as usize * hd, 0.02),
            &[2 * hc, hc * d],
        )
        .unwrap(),
        scale: MxArray::from_float32(&[0.75, 1.25], &[2]).unwrap(),
        base: MxArray::from_float32(&det_f32(9402, (2 * hc) as usize, 0.5), &[2 * hc]).unwrap(),
    };
    let w_head = HcWeights {
        weight: MxArray::from_float32(&det_f32(9403, hcu * hd, 0.02), &[hc, hc * d]).unwrap(),
        scale: MxArray::from_float32(&[0.9], &[1]).unwrap(),
        base: MxArray::from_float32(&det_f32(9404, hcu, 0.5), &[hc]).unwrap(),
    };
    let post_norm_w = MxArray::from_float32(
        &det_f32(9405, du, 0.5)
            .into_iter()
            .map(|v| 1.0 + v)
            .collect::<Vec<_>>(),
        &[d],
    )
    .unwrap();
    let next_norm_w = MxArray::from_float32(
        &det_f32(9406, du, 0.5)
            .into_iter()
            .map(|v| 1.0 + v)
            .collect::<Vec<_>>(),
        &[d],
    )
    .unwrap();

    let max_m = 65usize;
    let x_all =
        MxArray::from_float32(&det_f32(9407, max_m * hd, 1.2), &[max_m as i64, hc, d]).unwrap();
    let y_all = MxArray::from_float32(&det_f32(9408, max_m * du, 1.2), &[max_m as i64, d]).unwrap();
    let p_all =
        MxArray::from_float32(&det_f32(9409, max_m * hcu, 0.8), &[max_m as i64, hc]).unwrap();

    // Baseline M = 33 uses prefill 1-row/TG (USE_REG=true, ROW2=false).
    let (bx33, bh33, bp33) = hc_bridge(
        &y_all.slice(&[0, 0], &[33, d]).unwrap(),
        &x_all.slice(&[0, 0, 0], &[33, hc, d]).unwrap(),
        &p_all.slice(&[0, 0], &[33, hc]).unwrap(),
        &post_norm_w,
        &w_next,
        Some(&next_norm_w),
        norm_eps,
        &cfg,
    )
    .unwrap();
    let (bx33_v, bh33_v, bp33_v) = (read(&bx33), read(&bh33), read(&bp33));
    let bhead33_v = read(
        &hc_bridge_head(
            &y_all.slice(&[0, 0], &[33, d]).unwrap(),
            &x_all.slice(&[0, 0, 0], &[33, hc, d]).unwrap(),
            &p_all.slice(&[0, 0], &[33, hc]).unwrap(),
            &post_norm_w,
            &w_head,
            Some(&next_norm_w),
            norm_eps,
            hc_eps,
        )
        .unwrap(),
    );

    // M = 64 and odd M = 65 exercise prefill 2-row/TG (ROW2=true) including odd tail row 64.
    for &m in &[64usize, 65] {
        let ym = y_all.slice(&[0, 0], &[m as i64, d]).unwrap();
        let xm = x_all.slice(&[0, 0, 0], &[m as i64, hc, d]).unwrap();
        let pm = p_all.slice(&[0, 0], &[m as i64, hc]).unwrap();

        let (bx, bh, bp) = hc_bridge(
            &ym,
            &xm,
            &pm,
            &post_norm_w,
            &w_next,
            Some(&next_norm_w),
            norm_eps,
            &cfg,
        )
        .unwrap();
        let (bx_v, bh_v, bp_v) = (read(&bx), read(&bh), read(&bp));
        let ref_x = post_norm_hc_post(&ym, &xm, &pm, &post_norm_w, norm_eps).unwrap();
        let (ref_h_raw, ref_p) = hc_mix_reduce(&ref_x, &w_next, norm_eps, &cfg).unwrap();
        let ref_h = rms_norm(&ref_h_raw, &next_norm_w, norm_eps).unwrap();
        assert!(
            max_err(&bx_v, &read(&ref_x)) <= 1e-5,
            "prefill ROW2 x M={m}"
        );
        assert!(
            max_err(&bh_v, &read(&ref_h)) <= 1e-5,
            "prefill ROW2 h M={m}"
        );
        assert!(
            max_err(&bp_v, &read(&ref_p)) <= 1e-5,
            "prefill ROW2 p M={m}"
        );

        // Bit-exact match between ROW2 (M=64/65) and ROW1 (M=33) on shared rows 0..33.
        assert_eq!(
            to_bits(&bx_v[..33 * hd]),
            to_bits(&bx33_v),
            "ROW2 vs ROW1 x M={m}"
        );
        assert_eq!(
            to_bits(&bh_v[..33 * du]),
            to_bits(&bh33_v),
            "ROW2 vs ROW1 h M={m}"
        );
        assert_eq!(
            to_bits(&bp_v[..33 * hcu]),
            to_bits(&bp33_v),
            "ROW2 vs ROW1 p M={m}"
        );

        // SKIP_NEXT_NORM + HEAD_ONLY in ROW2.
        let (_, sh, _) =
            hc_bridge(&ym, &xm, &pm, &post_norm_w, &w_next, None, norm_eps, &cfg).unwrap();
        assert!(
            max_err(&read(&sh), &read(&ref_h_raw)) <= 1e-5,
            "prefill ROW2 skip_next_norm M={m}"
        );
        let bhead = read(
            &hc_bridge_head(
                &ym,
                &xm,
                &pm,
                &post_norm_w,
                &w_head,
                Some(&next_norm_w),
                norm_eps,
                hc_eps,
            )
            .unwrap(),
        );
        let ref_head = rms_norm(
            &hc_head(&ref_x, &w_head, norm_eps, hc_eps).unwrap(),
            &next_norm_w,
            norm_eps,
        )
        .unwrap();
        assert!(
            max_err(&bhead, &read(&ref_head)) <= 1e-5,
            "prefill ROW2 head M={m}"
        );
        assert_eq!(
            to_bits(&bhead[..33 * du]),
            to_bits(&bhead33_v),
            "ROW2 vs ROW1 head M={m}"
        );
    }
}

#[test]
fn affine4_dense_decode_row_invariant_and_codex_counterexample() {
    use super::super::weights::quantize_affine4_array;

    // SAFETY: pure process-wide Metal availability probe.
    let has_metal = unsafe { mlx_sys::mlx_metal_is_available() };
    if !has_metal {
        assert_ne!(
            std::env::var("MLX_TEST_REQUIRE_METAL").as_deref(),
            Ok("1"),
            "MLX_TEST_REQUIRE_METAL=1 requires Metal"
        );
        return;
    }
    crate::models::hy_v4_flash::exact_f32();
    let to_bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|&x| x.to_bits()).collect() };

    // 1. Exact Codex counterexample: scale=1, bias=2^-24, codes=[1, 2, 0..], inputs=[2, -1, 0..].
    // Under MLX quantized_matmul, B=1 (qmv) yields 2^-24 while B>=2 (qmv_wide) yields 0.0.
    // Our hy4_affine4_dense_decode kernel must be strictly bitwise identical across M=1..8.
    {
        let mut w_words = vec![0u32; 8];
        w_words[0] = 1u32 | (2u32 << 4);
        let w = MxArray::from_uint32(&w_words, &[1, 8]).unwrap();
        let s = MxArray::from_float32(&[1.0f32], &[1, 1])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let b_val = 2.0f32.powi(-24);
        let b = MxArray::from_float32(&[b_val], &[1, 1])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();

        let mut x_rows = vec![0.0f32; 8 * 64];
        x_rows[0] = 2.0;
        x_rows[1] = -1.0;
        for r in 1..8 {
            let base = r * 64;
            x_rows[base] = (r as f32) * 0.75 + 1.0;
            x_rows[base + 1] = -(r as f32) * 0.5;
            x_rows[base + 5] = 0.25 * (r as f32);
        }
        let x_all = MxArray::from_float32(&x_rows, &[8, 64]).unwrap();

        let mut row1_bits = Vec::with_capacity(8);
        for r in 0..8 {
            let xr = x_all.slice(&[r as i64, 0], &[(r + 1) as i64, 64]).unwrap();
            let y1 = try_affine4_dense_decode(&xr, &w, &s, &b)
                .unwrap()
                .expect("affine4_dense_decode must run on Metal");
            row1_bits.push(to_bits(&read(&y1)));
        }
        assert_eq!(row1_bits[0], vec![b_val.to_bits()]);

        for &m in &[2usize, 3, 4, 5, 8] {
            let xm = x_all.slice(&[0, 0], &[m as i64, 64]).unwrap();
            let ym = try_affine4_dense_decode(&xm, &w, &s, &b)
                .unwrap()
                .expect("affine4_dense_decode batch must run on Metal");
            let yb = to_bits(&read(&ym));
            for r in 0..m {
                assert_eq!(
                    &yb[r..r + 1],
                    row1_bits[r].as_slice(),
                    "counterexample M={m} row={r}"
                );
            }
        }
    }

    // 2. General shapes: non-FAST_K (K=64, 192, 576) and FAST_K (K=512, 1024) with odd N.
    for (ci, &(n, k)) in [
        (37usize, 64usize),
        (53, 192),
        (67, 512),
        (29, 576),
        (41, 1024),
    ]
    .iter()
    .enumerate()
    {
        let w_f32 = MxArray::from_float32(
            &det_f32(9600 + ci as u64, n * k, 0.25),
            &[n as i64, k as i64],
        )
        .unwrap();
        let q4 = quantize_affine4_array(&w_f32).unwrap();
        let Dense::Qmm4 {
            ref w,
            ref scales,
            ref biases,
        } = q4
        else {
            panic!("expected Dense::Qmm4");
        };
        let mode = std::ffi::CString::new("affine").unwrap();
        let sf32 = scales.astype(DType::Float32).unwrap();
        let bf32 = biases.astype(DType::Float32).unwrap();
        // Dequantize in f32 (not BF16) to match exact f32 kernel arithmetic.
        // SAFETY: live array handles; returns a new f32 array.
        let deq_h = unsafe {
            mlx_sys::mlx_dequantize(
                w.as_raw_ptr(),
                sf32.as_raw_ptr(),
                bf32.as_raw_ptr(),
                64,
                4,
                DType::Float32 as i32,
                mode.as_ptr(),
            )
        };
        let deq = MxArray::from_handle(deq_h, "test affine4 f32 dequant").unwrap();
        let x_all =
            MxArray::from_float32(&det_f32(9700 + ci as u64, 8 * k, 1.0), &[8, k as i64]).unwrap();
        let want_all = read(
            &x_all
                .matmul(&deq.transpose(Some(&[1, 0])).unwrap())
                .unwrap(),
        );

        let mut single_bits = Vec::with_capacity(8);
        for r in 0..8 {
            let xr = x_all
                .slice(&[r as i64, 0], &[(r + 1) as i64, k as i64])
                .unwrap();
            let y1 = try_affine4_dense_decode(&xr, w, scales, biases)
                .unwrap()
                .expect("affine4_dense_decode must run on Metal");
            let y1v = read(&y1);
            let max_diff = y1v
                .iter()
                .zip(&want_all[r * n..(r + 1) * n])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff <= 1e-4,
                "N={n} K={k} row={r}: affine4_dense_decode vs f32 dequant max_diff={max_diff:e}"
            );
            single_bits.push(to_bits(&y1v));
        }

        for &m in &[2usize, 3, 4, 5, 8] {
            let xm = x_all.slice(&[0, 0], &[m as i64, k as i64]).unwrap();
            let ym = try_affine4_dense_decode(&xm, w, scales, biases)
                .unwrap()
                .expect("affine4_dense_decode batch must run on Metal");
            let yb = to_bits(&read(&ym));
            for r in 0..m {
                assert_eq!(
                    &yb[r * n..(r + 1) * n],
                    single_bits[r].as_slice(),
                    "N={n} K={k} M={m} row={r}: not bit-exact with M=1"
                );
            }
        }
    }
}
