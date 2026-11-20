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
}
