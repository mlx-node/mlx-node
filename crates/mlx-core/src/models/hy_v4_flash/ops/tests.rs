use std::path::{Path, PathBuf};

use serde_json::Value;

use super::*;
use crate::models::deepseek_v41::EngramConfig;
use crate::models::deepseek_v41::engram_hash::EngramHasher;
use crate::models::hy_v4_flash::engram::{
    self, ENGRAM_EPS, Hasher, key_value, normed_rows, signal, stream_dot, stream_gate,
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
    let out = engram::forward(&x, &rows, &w).unwrap();
    assert_eq!(shape(&out), [n, hc, d]);
    assert_eq!(out.dtype().unwrap(), DType::Float32);
    assert_close(&read(&out), &flat(&o["out"]), "out");
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
    assert!(engram::forward(&x, &rows[1..], &w).is_err(), "short rows");
    assert!(engram::forward(&x, &[], &w).is_err(), "no rows");
    let x3 = x.slice(&[0, 0, 0], &[2, 3, 8]).unwrap();
    assert!(engram::forward(&x3, &rows, &w).is_err(), "3 streams");
    let x2 = x.reshape(&[2, 32]).unwrap();
    assert!(engram::forward(&x2, &rows, &w).is_err(), "rank 2");
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
