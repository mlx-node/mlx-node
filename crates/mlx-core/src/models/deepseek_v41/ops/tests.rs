use std::path::PathBuf;

use serde_json::Value;

use super::*;
use crate::models::deepseek_v41::DeepseekV41Config;
use crate::utils::gguf::parse_gguf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/deepseek-v41")
        .join(name)
}

fn ops() -> Value {
    serde_json::from_slice(&std::fs::read(fixture("ops.json")).unwrap()).unwrap()
}

fn tiny() -> DeepseekV41Config {
    DeepseekV41Config::from_gguf(&parse_gguf(fixture("tiny-q8_0.gguf")).unwrap()).unwrap()
}

/// `ops.json` floats: hex of the i32 view (`-40771baa` = negative i32).
fn bits(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(
            |s| f32::from_bits(i64::from_str_radix(s.as_str().unwrap(), 16).unwrap() as i32 as u32),
        )
        .collect()
}

fn read(x: &MxArray) -> Vec<f32> {
    x.to_float32().unwrap().as_ref().to_vec()
}

fn arr(v: &[f32], shape: &[i64]) -> MxArray {
    MxArray::from_float32(v, shape).unwrap()
}

/// Deterministic values in `[-scale, scale)`.
fn lcg(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what} len");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}[{i}]: {g} vs {w}");
    }
}

/// Elementwise `|got - want| <= rel * |want| + floor`; returns the worst ratio.
fn assert_rel(got: &[f32], want: &[f64], rel: f64, floor: f64, what: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{what} len");
    let mut worst = 0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (f64::from(g) - w).abs();
        let tol = rel * w.abs() + floor;
        worst = worst.max(err / tol);
        assert!(
            err <= tol,
            "{what}[{i}]: {g} vs {w} (err {err:e}, tol {tol:e})"
        );
    }
    worst
}

fn yarn_of(t: &Value) -> YarnConfig {
    YarnConfig {
        factor: t["factor"].as_f64().unwrap() as f32,
        original_seq_len: t["original_seq_len"].as_u64().unwrap() as usize,
        beta_fast: t["beta_fast"].as_f64().unwrap() as f32,
        beta_slow: t["beta_slow"].as_f64().unwrap() as f32,
    }
}

const YARN_TAGS: [&str; 4] = ["tiny.main", "tiny.compress", "real.main", "real.compress"];

/// `ops.json` `inv_freq_f32_bits` came through `torch.angle(polar(1, f_i))`,
/// which moves 9 of the 80 values by 1 ulp. The reference's own `f_i` (the
/// `torch.polar` angle argument inside `precompute_freqs_cis`, captured by the
/// research oracle `s5_yarn_capture.py`) differs from `ops.json` exactly here:
/// (tag, index, f_i bits - ops.json bits).
const TRUE_FI_DELTA: [(&str, usize, i64); 9] = [
    ("tiny.main", 2, -1),
    ("real.main", 2, 1),
    ("real.main", 3, -1),
    ("real.main", 6, -1),
    ("real.main", 15, -1),
    ("real.main", 18, 1),
    ("real.compress", 3, 1),
    ("real.compress", 11, 1),
    ("real.compress", 15, -1),
];

#[test]
fn yarn_inverse_freqs_within_one_ulp() {
    let o = ops();
    let (mut exact, mut total) = (0, 0);
    for tag in YARN_TAGS {
        let t = &o["yarn"][tag];
        let dim = t["dim"].as_u64().unwrap() as usize;
        let got = inv_freqs(dim, t["theta"].as_f64().unwrap() as f32, Some(&yarn_of(t)));
        let want = bits(&t["inv_freq_f32_bits"]);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let delta = i64::from(g.to_bits()) - i64::from(w.to_bits());
            assert!(delta.abs() <= 1, "{tag} f[{i}]: {g} vs {w}");
            let want_delta = TRUE_FI_DELTA
                .iter()
                .find(|&&(t, j, _)| t == tag && j == i)
                .map_or(0, |e| e.2);
            assert_eq!(delta, want_delta, "{tag} f[{i}] vs the reference f_i");
            exact += usize::from(delta == 0);
            total += 1;
        }
    }
    for (i, f) in inv_freqs(64, 10000.0, None).iter().enumerate() {
        let want = 1.0 / 10000f64.powf(2.0 * i as f64 / 64.0);
        assert!(
            (f64::from(*f) - want).abs() <= want * 2f64.powi(-23),
            "plain f[{i}]"
        );
    }
    println!(
        "yarn inverse freqs: {exact}/{total} bitwise to ops.json, all bitwise to the reference"
    );
}

/// The angle formula reproduces `ops.json` angles bitwise from its `f_i`;
/// RoPE through MLX within the per-position bound
/// `|mlx - ora64| <= max(2 |ora32 - ora64|, 1e-6 |ora64|)`, where ora32 /
/// ora64 rotate by the reference's f32 / f64 angle of its own `f_i` (the
/// yarn test proves `inv_freqs` is that `f_i`); nope dims untouched; the
/// inverse undoes it; dtype kept.
#[test]
fn rope_matches_reference_per_position() {
    let o = ops();
    for tag in YARN_TAGS {
        let t = &o["yarn"][tag];
        let rd = t["dim"].as_u64().unwrap() as usize;
        let half = rd / 2;
        let inv = inv_freqs(rd, t["theta"].as_f64().unwrap() as f32, Some(&yarn_of(t)));
        let inv_o = bits(&t["inv_freq_f32_bits"]);
        let pos: Vec<u64> = t["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_u64().unwrap())
            .collect();
        let ang32 = bits(&t["angle_f32_bits"]);
        for (pi, &p) in pos.iter().enumerate() {
            for i in 0..half {
                let a = rope_angle(p, inv_o[i]);
                assert_eq!(
                    a.to_bits(),
                    ang32[pi * half + i].to_bits(),
                    "{tag} angle p={p} i={i}"
                );
            }
        }
        let (s, h, nope) = (pos.len(), 2, 8);
        let d = nope + rd;
        let x = lcg(rd as u64 + pos.len() as u64, s * h * d, 1.0);
        let xa = arr(&x, &[s as i64, h as i64, d as i64]);
        let (c, sn) = rope_tables(&inv, &pos).unwrap();
        let ya = apply_partial_rope(&xa, rd as i64, &c, &sn, false).unwrap();
        let y = read(&ya);
        let (mut worst, mut same32) = (0f64, 0usize);
        for (pi, &p) in pos.iter().enumerate() {
            for hh in 0..h {
                let base = (pi * h + hh) * d;
                assert_bits(&y[base..base + nope], &x[base..base + nope], "nope");
                for i in 0..half {
                    let (a, b) = (x[base + nope + 2 * i], x[base + nope + 2 * i + 1]);
                    let t64 = p as f64 * f64::from(inv[i]);
                    let o64 = [
                        f64::from(a) * t64.cos() - f64::from(b) * t64.sin(),
                        f64::from(a) * t64.sin() + f64::from(b) * t64.cos(),
                    ];
                    let t32 = f64::from(rope_angle(p, inv[i]));
                    let (c32, s32) = (t32.cos() as f32, t32.sin() as f32);
                    let o32 = [a * c32 - b * s32, a * s32 + b * c32];
                    for k in 0..2 {
                        let got = f64::from(y[base + nope + 2 * i + k]);
                        let tol =
                            (2.0 * (f64::from(o32[k]) - o64[k]).abs()).max(1e-6 * o64[k].abs());
                        let err = (got - o64[k]).abs();
                        assert!(
                            err <= tol,
                            "{tag} p={} h={hh} i={i} k={k}: {got} vs {} (tol {tol:e})",
                            p,
                            o64[k]
                        );
                        worst = worst.max(if tol > 0.0 { err / tol } else { 0.0 });
                        same32 += usize::from(got as f32 == o32[k]);
                    }
                }
            }
        }
        let back = read(&apply_partial_rope(&ya, rd as i64, &c, &sn, true).unwrap());
        for (g, w) in back.iter().zip(&x) {
            assert!((g - w).abs() <= 1e-5, "{tag} inverse: {g} vs {w}");
        }
        let yb = apply_partial_rope(
            &xa.astype(DType::BFloat16).unwrap(),
            rd as i64,
            &c,
            &sn,
            false,
        )
        .unwrap();
        assert_eq!(yb.dtype().unwrap(), DType::BFloat16);
        println!(
            "rope {tag}: worst err/tol {worst:.3}, {same32}/{} values equal the f32 reference",
            s * h * rd
        );
    }
}

/// Activation fake quant, bit for bit: fp8 e4m3 / ue8m0 per 32, fp4 e2m1 /
/// ue8m0 per 32, fp4 e2m1 / e4m3 per 16, the bf16 fp8 path, and the e2m1 ties.
#[test]
fn fake_quant_roundtrips_bitwise() {
    let o = ops();
    for c in o["fake_quant_roundtrips"].as_array().unwrap() {
        let scale = c["scale"].as_f64().unwrap();
        let x = arr(&bits(&c["x_bits"]), &[2, 64]);
        let what = |k: &str| format!("scale {scale} {k}");
        assert_bits(
            &read(&fp8_roundtrip(&x, 32).unwrap()),
            &bits(&c["fp8_b32_bits"]),
            &what("fp8"),
        );
        assert_bits(
            &read(&fp4_roundtrip(&x, 32, false).unwrap()),
            &bits(&c["fp4_e8m0_b32_bits"]),
            &what("fp4 e8m0"),
        );
        assert_bits(
            &read(&fp4_roundtrip(&x, 16, true).unwrap()),
            &bits(&c["fp4_e4m3_b16_bits"]),
            &what("fp4 e4m3"),
        );
        let yb = fp8_roundtrip(&x.astype(DType::BFloat16).unwrap(), 32).unwrap();
        assert_eq!(yb.dtype().unwrap(), DType::BFloat16);
        assert_bits(
            &read(&yb),
            &bits(&c["x_bf16_fp8_b32_bits"]),
            &what("bf16 fp8"),
        );
    }
    let f = |k: &str| -> Vec<f32> {
        o["e2m1_ties"][k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    };
    let x = f("x");
    assert_bits(
        &read(&e2m1_round(&arr(&x, &[x.len() as i64])).unwrap()),
        &f("y"),
        "e2m1 ties",
    );
    assert!(fp8_roundtrip(&arr(&[1.0; 48], &[48]), 32).is_err());
}

/// `pow2_ceil` == the kernel's bit form at, just above and just below every
/// normal power of two.
#[test]
fn pow2_ceil_exact_at_powers_of_two() {
    let mut r = Vec::new();
    for k in -125i32..=126 {
        let p = 2f32.powi(k);
        r.extend([
            p,
            f32::from_bits(p.to_bits() + 1),
            f32::from_bits(p.to_bits() - 1),
        ]);
    }
    r.extend(lcg(7, 64, 1000.0).into_iter().map(f32::abs));
    let want: Vec<f32> = r
        .iter()
        .map(|x| {
            let b = x.to_bits();
            let e = ((b >> 23) & 0xFF) as i32 - 127 + i32::from(b & 0x7F_FFFF != 0);
            f32::from_bits(((e + 127) as u32) << 23)
        })
        .collect();
    assert_bits(
        &read(&pow2_ceil(&arr(&r, &[r.len() as i64])).unwrap()),
        &want,
        "pow2_ceil",
    );
}

/// Router: scores rel 1e-6, ids equal (exact ties to the lowest index),
/// weights rel 1e-6, for the biased and the zero-bias case.
#[test]
fn router_matches_reference() {
    let c = tiny();
    let (n, k) = (c.n_routed_experts, c.n_activated_experts);
    for case in ops()["router"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let logits = bits(&case["logits_bits"]);
        let rows = logits.len() / n;
        let scores_a = router_scores(&arr(&logits, &[rows as i64, n as i64])).unwrap();
        let scores = read(&scores_a);
        let want: Vec<f64> = bits(&case["scores_bits"])
            .iter()
            .map(|&x| f64::from(x))
            .collect();
        assert_rel(&scores, &want, 1e-6, 0.0, &format!("{name} scores"));
        let biased = read(
            &scores_a
                .add(&arr(&bits(&case["bias_bits"]), &[n as i64]))
                .unwrap(),
        );
        let (ids, w) = route_select(&biased, &scores, n, k, c.route_scale, c.norm_topk_prob);
        let want_ids: Vec<u32> = case["idx"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
            })
            .collect();
        assert_eq!(ids, want_ids, "{name} ids");
        let want_w: Vec<f64> = bits(&case["weights_bits"])
            .iter()
            .map(|&x| f64::from(x))
            .collect();
        assert_rel(&w, &want_w, 1e-6, 0.0, &format!("{name} weights"));
    }
}

/// Sinkhorn split vs the oracle (mild and peaky inputs), rel 1e-6.
#[test]
fn sinkhorn_matches_reference() {
    let o = ops();
    let sk = &o["sinkhorn"];
    let (iters, eps) = (
        sk["iters"].as_u64().unwrap() as usize,
        sk["eps"].as_f64().unwrap() as f32,
    );
    let hc = tiny().hc_mult as i64;
    let mix = (2 + hc) * hc;
    for c in sk["cases"].as_array().unwrap() {
        let std = c["std"].as_f64().unwrap();
        let mixes = bits(&c["mixes_bits"]);
        let n = mixes.len() as i64 / mix;
        let (pre, post, comb) = hc_split_sinkhorn(
            &arr(&mixes, &[n, mix]),
            &arr(&bits(&c["scale_bits"]), &[3]),
            &arr(&bits(&c["base_bits"]), &[mix]),
            hc,
            iters,
            eps,
        )
        .unwrap();
        for (got, key) in [(pre, "pre_bits"), (post, "post_bits"), (comb, "comb_bits")] {
            let want: Vec<f64> = bits(&c[key]).iter().map(|&x| f64::from(x)).collect();
            let worst = assert_rel(&read(&got), &want, 1e-6, 1e-12, &format!("std {std} {key}"));
            println!("sinkhorn std {std} {key}: worst err/tol {worst:.3}");
        }
    }
}

/// `hc_mixes` / `hc_pre` / `hc_post` vs an f64 transcription of
/// `Block.hc_mixes` / `hc_pre` / `hc_post`, rel 1e-6.
#[test]
fn hc_ops_match_f64_reference() {
    crate::models::deepseek_v41::test_fixture::exact_f32();
    let (n, hc, d) = (3usize, 4usize, 64usize);
    let mix = (2 + hc) * hc;
    let x = lcg(11, n * hc * d, 2.0);
    let hc_fn = lcg(12, mix * hc * d, 0.1);
    let scale = [0.7f32, 0.9, 1.1];
    let base = lcg(13, mix, 0.5);
    let (norm_eps, hc_eps, iters) = (1e-20f32, 1e-6f32, 20usize);
    let xa = arr(&x, &[n as i64, hc as i64, d as i64]);
    let (pre, post, comb) = hc_mixes(
        &xa,
        &arr(&hc_fn, &[mix as i64, (hc * d) as i64]),
        &arr(&scale, &[3]),
        &arr(&base, &[mix as i64]),
        norm_eps,
        hc as i64,
        iters,
        hc_eps,
    )
    .unwrap();
    let sig = |v: f64| 1.0 / (1.0 + (-v).exp());
    let e = f64::from(hc_eps);
    let (mut wpre, mut wpost, mut wcomb) = (vec![], vec![], vec![]);
    for r in 0..n {
        let xr = &x[r * hc * d..(r + 1) * hc * d];
        let ms = xr.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / (hc * d) as f64;
        let rs = 1.0 / (ms + f64::from(norm_eps)).sqrt();
        let m: Vec<f64> = (0..mix)
            .map(|j| {
                let row = &hc_fn[j * hc * d..(j + 1) * hc * d];
                xr.iter()
                    .zip(row)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum::<f64>()
                    * rs
            })
            .collect();
        let sc: Vec<f64> = scale.iter().map(|&v| f64::from(v)).collect();
        let bs: Vec<f64> = base.iter().map(|&v| f64::from(v)).collect();
        wpre.extend((0..hc).map(|j| sig(m[j] * sc[0] + bs[j]) + e));
        wpost.extend((0..hc).map(|j| 2.0 * sig(m[hc + j] * sc[1] + bs[hc + j])));
        let mut cm: Vec<f64> = (0..hc * hc)
            .map(|j| m[2 * hc + j] * sc[2] + bs[2 * hc + j])
            .collect();
        for j in 0..hc {
            let row = &mut cm[j * hc..(j + 1) * hc];
            let mx = row.iter().cloned().fold(f64::MIN, f64::max);
            let s: f64 = row.iter().map(|v| (v - mx).exp()).sum();
            for v in row.iter_mut() {
                *v = (*v - mx).exp() / s + e;
            }
        }
        let norm_cols = |cm: &mut Vec<f64>| {
            for k in 0..hc {
                let s: f64 = (0..hc).map(|j| cm[j * hc + k]).sum::<f64>() + e;
                (0..hc).for_each(|j| cm[j * hc + k] /= s);
            }
        };
        norm_cols(&mut cm);
        for _ in 1..iters {
            for j in 0..hc {
                let s: f64 = cm[j * hc..(j + 1) * hc].iter().sum::<f64>() + e;
                cm[j * hc..(j + 1) * hc].iter_mut().for_each(|v| *v /= s);
            }
            norm_cols(&mut cm);
        }
        wcomb.extend(cm);
    }
    let (pre_v, post_v, comb_v) = (read(&pre), read(&post), read(&comb));
    assert_rel(&pre_v, &wpre, 1e-6, 1e-9, "pre");
    assert_rel(&post_v, &wpost, 1e-6, 1e-9, "post");
    assert_rel(&comb_v, &wcomb, 1e-6, 1e-9, "comb");

    let y = lcg(14, n * d, 1.0);
    let got_pre = read(&hc_pre(&xa, &pre).unwrap());
    let got_post = read(&hc_post(&arr(&y, &[n as i64, d as i64]), &xa, &post, &comb).unwrap());
    let (mut want_pre, mut want_post) = (vec![], vec![]);
    for r in 0..n {
        let xv = |j: usize, i: usize| f64::from(x[(r * hc + j) * d + i]);
        for i in 0..d {
            want_pre.push(
                (0..hc)
                    .map(|j| f64::from(pre_v[r * hc + j]) * xv(j, i))
                    .sum(),
            );
        }
        for k in 0..hc {
            for i in 0..d {
                let mixed: f64 = (0..hc)
                    .map(|j| f64::from(comb_v[(r * hc + j) * hc + k]) * xv(j, i))
                    .sum();
                want_post.push(f64::from(post_v[r * hc + k]) * f64::from(y[r * d + i]) + mixed);
            }
        }
    }
    assert_rel(&got_pre, &want_pre, 1e-6, 1e-6, "hc_pre");
    assert_rel(&got_post, &want_post, 1e-6, 1e-6, "hc_post");
    let pb = hc_pre(&xa.astype(DType::BFloat16).unwrap(), &pre).unwrap();
    assert_eq!(pb.dtype().unwrap(), DType::BFloat16);
}

/// SwiGLU clamps bit for bit (up both sides, gate from above, 0 = off);
/// `silu(gate) * up * w` rel 1e-6 vs f64; output dtype as asked.
#[test]
fn swiglu_clamp_bitwise() {
    let mut g = lcg(21, 61, 15.0);
    let mut u = lcg(22, 61, 15.0);
    g.extend([10.0, -10.0, 10.5, f32::from_bits(10f32.to_bits() + 1)]);
    u.extend([10.0, -10.0, -10.5, f32::from_bits(10f32.to_bits() + 1)]);
    let n = g.len() as i64;
    let (ga, ua) = (arr(&g, &[n, 1]), arr(&u, &[n, 1]));
    let (gc, uc) = swiglu_clamp(&ga, &ua, 10.0).unwrap();
    let want_g: Vec<f32> = g.iter().map(|&v| v.min(10.0)).collect();
    let want_u: Vec<f32> = u.iter().map(|&v| v.clamp(-10.0, 10.0)).collect();
    assert_bits(&read(&gc), &want_g, "gate clamp");
    assert_bits(&read(&uc), &want_u, "up clamp");
    let (g0, u0) = swiglu_clamp(&ga, &ua, 0.0).unwrap();
    assert_bits(&read(&g0), &g, "gate no clamp");
    assert_bits(&read(&u0), &u, "up no clamp");
    let w = lcg(23, g.len(), 1.0);
    let y = swiglu(&ga, &ua, 10.0, Some(&arr(&w, &[n, 1])), DType::Float32).unwrap();
    let want: Vec<f64> = (0..g.len())
        .map(|i| {
            let gv = f64::from(want_g[i]);
            gv / (1.0 + (-gv).exp()) * f64::from(want_u[i]) * f64::from(w[i])
        })
        .collect();
    assert_rel(&read(&y), &want, 1e-6, 1e-9, "swiglu");
    let yb = swiglu(&ga, &ua, 10.0, None, DType::BFloat16).unwrap();
    assert_eq!(yb.dtype().unwrap(), DType::BFloat16);
}

/// Engram gate vs an f64 transcription of `Engram.forward` after `wkv`.
#[test]
fn engram_gate_matches_f64_reference() {
    let (n, hc, d) = (3usize, 4usize, 64usize);
    let eps = 1e-20f32;
    let x = lcg(31, n * hc * d, 2.0);
    let key = lcg(32, n * hc * d, 1.5);
    let value = lcg(33, n * d, 1.0);
    let weight = lcg(34, hc * d, 1.2);
    let got = engram_gate(
        &arr(&x, &[n as i64, hc as i64, d as i64]),
        &arr(&key, &[n as i64, (hc * d) as i64]),
        &arr(&value, &[n as i64, d as i64]),
        &arr(&weight, &[hc as i64, d as i64]),
        eps,
    )
    .unwrap();
    let mut want = Vec::new();
    for r in 0..n {
        for j in 0..hc {
            let o = (r * hc + j) * d;
            let h: Vec<f64> = x[o..o + d].iter().map(|&v| f64::from(v)).collect();
            let k: Vec<f64> = key[o..o + d].iter().map(|&v| f64::from(v)).collect();
            let ms = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>() / d as f64;
            let rstd = 1.0 / (ms(&h) + f64::from(eps)).sqrt() / (ms(&k) + f64::from(eps)).sqrt();
            let dot = (0..d)
                .map(|i| h[i] * f64::from(weight[j * d + i]) * k[i])
                .sum::<f64>()
                * rstd
                / (d as f64).sqrt();
            let gate = 1.0 / (1.0 + (-(dot.abs().max(1e-6).sqrt().copysign(dot))).exp());
            want.extend((0..d).map(|i| h[i] + gate * f64::from(value[r * d + i])));
        }
    }
    assert_rel(&read(&got), &want, 1e-6, 1e-6, "engram gate");
}
