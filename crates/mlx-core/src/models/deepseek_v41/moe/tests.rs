//! MoE vs the oracle (`moe.npz`: MoE input after `ffn_norm`, router ids /
//! weights and output of every layer, one 24-row prefill per run; bf16 runs
//! also carry `out32` = the same MoE on the same input in f32).

use std::ffi::CString;

use super::*;
use crate::models::deepseek_v41::test_fixture::{Npz, TinyGguf, read, rel_l2};

const RUNS: [&str; 3] = ["c1.fp32_on", "c1.bf16_on", "pad.fp32_on"];
const ROWS: usize = 24;

struct Fix {
    gguf: TinyGguf,
    cfg: DeepseekV41Config,
    npz: Npz,
}

fn fix() -> Fix {
    let gguf = TinyGguf::open();
    let cfg = DeepseekV41Config::from_gguf(&gguf.gguf).unwrap();
    Fix {
        gguf,
        cfg,
        npz: Npz::open("moe.npz"),
    }
}

fn bank(f: &Fix, name: &str) -> Result<ExpertBank> {
    let (info, raw) = f.gguf.raw(name);
    ExpertBank::from_mxfp4(info, &raw)
}

fn moe(f: &Fix, layer: usize) -> Moe {
    let w =
        MoeWeights::from_gguf(layer, |n| f.gguf.dense(n).map(Dense::from), |n| bank(f, n)).unwrap();
    Moe::new(&f.cfg, layer, w).unwrap()
}

fn dtype(run: &str) -> DType {
    if run.contains("bf16") {
        DType::BFloat16
    } else {
        DType::Float32
    }
}

/// The run's MoE input of `layer` (bf16 runs: exact bf16 values stored as f32).
fn input(f: &Fix, run: &str, layer: usize) -> MxArray {
    f.npz
        .rows(&format!("{run}.L{layer}.x"), 0, ROWS)
        .astype(dtype(run))
        .unwrap()
}

fn vals<'a>(f: &'a Fix, run: &str, layer: usize, what: &str) -> &'a [f32] {
    &f.npz.get(&format!("{run}.L{layer}.{what}")).1
}

/// ggml `dequantize_row_mxfp4`: `kvalues_mxfp4[code] * E8M0_TO_FP32_HALF(e)`.
fn ggml_mxfp4(block: &[u8]) -> [f32; 32] {
    const KV: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    let e = u32::from(block[0]);
    let half = f32::from_bits(if e < 2 {
        0x0020_0000 << e
    } else {
        (e - 1) << 23
    });
    let mut out = [0f32; 32];
    for j in 0..16 {
        out[j] = f32::from(KV[usize::from(block[1 + j] & 0x0F)]) * half;
        out[j + 16] = f32::from(KV[usize::from(block[1 + j] >> 4)]) * half;
    }
    out
}

fn dequant_mxfp4(b: &ExpertBank) -> Vec<f32> {
    let mode = CString::new("mxfp4").unwrap();
    // SAFETY: live arrays for the call; null biases are allowed for mxfp4.
    let h = unsafe {
        mlx_sys::mlx_dequantize(
            b.weight.as_raw_ptr(),
            b.scales.as_raw_ptr(),
            std::ptr::null_mut(),
            32,
            4,
            0,
            mode.as_ptr(),
        )
    };
    read(&MxArray::from_handle(h, "dequant mxfp4").unwrap())
}

/// Bank bytes == an independent repack of the GGUF blocks (nibble j / j+16
/// of byte j -> codes LSB first in 4 words, E8M0 byte -> scale), and MLX's
/// `mxfp4` dequant of the bank == ggml's MXFP4 dequant, bit for bit.
#[test]
fn bank_bytes_and_values_match_gguf() {
    let f = fix();
    let mut tensors = 0;
    for layer in 0..f.cfg.n_layers {
        for p in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
            let name = format!("blk.{layer}.{p}.weight");
            let (_, raw) = f.gguf.raw(&name);
            let b = bank(&f, &name).unwrap();
            let blocks = raw.as_chunks::<17>().0;
            let (mut want_w, mut want_s, mut want_v) = (Vec::new(), Vec::new(), Vec::new());
            for blk in blocks {
                let mut words = [0u32; 4];
                for v in 0..32 {
                    let byte = blk[1 + v % 16];
                    let code = if v < 16 { byte & 0x0F } else { byte >> 4 };
                    words[v / 8] |= u32::from(code) << (4 * (v % 8));
                }
                want_w.extend(words);
                want_s.push(blk[0]);
                want_v.extend(ggml_mxfp4(blk));
            }
            assert_eq!(b.weight.dtype().unwrap(), DType::Uint32, "{name}");
            assert_eq!(
                b.weight.to_uint32().unwrap().to_vec(),
                want_w,
                "{name} codes"
            );
            assert_eq!(b.scales.to_uint8().unwrap(), want_s, "{name} scales");
            let got_v = dequant_mxfp4(&b);
            let bad = got_v
                .iter()
                .zip(&want_v)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!((got_v.len(), bad), (want_v.len(), 0), "{name} values");
            let (lo, hi) = (want_s.iter().min().unwrap(), want_s.iter().max().unwrap());
            assert!(
                *lo >= 2 && *hi < 255,
                "{name}: scale bytes {lo}..={hi} hit an E8M0 edge"
            );
            tensors += 1;
        }
    }
    assert_eq!(tensors, 3 * f.cfg.n_layers);
}

/// Router ids (top-k rank order) bitwise, weights within 1e-6, every layer
/// and run; bf16 runs route on the f32 upcast of the bf16 input.
#[test]
fn routes_match_oracle() {
    let f = fix();
    for run in RUNS {
        for layer in 0..f.cfg.n_layers {
            let r = moe(&f, layer).route(&input(&f, run, layer)).unwrap();
            let want_ids: Vec<u32> = vals(&f, run, layer, "ids")
                .iter()
                .map(|&v| v as u32)
                .collect();
            assert_eq!(r.ids, want_ids, "{run} L{layer} ids");
            let want_w = vals(&f, run, layer, "w");
            for (i, (&g, &w)) in r.weights.iter().zip(want_w).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-6 * w.abs(),
                    "{run} L{layer} weight {i}: {g} vs {w}"
                );
            }
        }
    }
}

/// f32 output rel <= 1e-5 per layer; the clamps matter (L3 routed, L6 shared
/// carry clamp hits in the fixture): dropping them breaks the match.
#[test]
fn f32_output_matches_oracle() {
    let f = fix();
    let mut worst = 0f64;
    for run in ["c1.fp32_on", "pad.fp32_on"] {
        for layer in 0..f.cfg.n_layers {
            let (y, _) = moe(&f, layer).forward(&input(&f, run, layer)).unwrap();
            assert_eq!(y.dtype().unwrap(), DType::Float32);
            let r = rel_l2(&read(&y), vals(&f, run, layer, "out"));
            assert!(r <= 1e-5, "{run} L{layer}: rel {r:e}");
            worst = worst.max(r);
        }
    }
    eprintln!("moe f32 worst rel {worst:e}");
    for (layer, shared) in [(3usize, false), (6, true)] {
        let mut m = moe(&f, layer);
        if shared {
            m.shared_limit = 0.0;
        } else {
            m.limit = 0.0;
        }
        let (y, _) = m.forward(&input(&f, "c1.fp32_on", layer)).unwrap();
        let r = rel_l2(&read(&y), vals(&f, "c1.fp32_on", layer, "out"));
        assert!(
            r > 1e-3,
            "L{layer} without the clamp still matches (rel {r:e})"
        );
    }
}

/// bf16: output in bf16 and within the noise floor: relL2(mlx, out32) <=
/// max(2 relL2(oracle bf16, out32), 1e-3), out32 = the oracle MoE in f32 on
/// the same input.
#[test]
fn bf16_output_within_noise_floor() {
    let f = fix();
    let run = "c1.bf16_on";
    for layer in 0..f.cfg.n_layers {
        let (y, _) = moe(&f, layer).forward(&input(&f, run, layer)).unwrap();
        assert_eq!(y.dtype().unwrap(), DType::BFloat16);
        let ref32 = vals(&f, run, layer, "out32");
        let floor = rel_l2(vals(&f, run, layer, "out"), ref32);
        let r = rel_l2(&read(&y), ref32);
        eprintln!("moe bf16 L{layer}: rel {r:e} floor {floor:e}");
        assert!(
            r <= (2.0 * floor).max(1e-3),
            "L{layer}: rel {r:e} vs floor {floor:e}"
        );
    }
}

/// Given the same routes, each row's routed outputs do not depend on the
/// other rows of the call: all rows == chunks of 1 and 7, bitwise, f32 and
/// bf16. Routing per chunk: same ids, weights within 1e-6; the full output
/// (router and shared expert GEMMs included) within 1e-5 (f32) / the bf16
/// noise floor (bf16).
#[test]
fn rows_independent_of_call_split() {
    let f = fix();
    for run in ["c1.fp32_on", "c1.bf16_on"] {
        for layer in 0..f.cfg.n_layers {
            let m = moe(&f, layer);
            let x = input(&f, run, layer);
            let routes = m.route(&x).unwrap();
            let all = read(&m.routed(&x, &routes).unwrap());
            let (full, _) = m.forward(&x).unwrap();
            let full = read(&full);
            let per_row = all.len() / ROWS;
            let dim = f.cfg.dim;
            // bf16: the shared expert's bf16 GEMM outputs round per M; bound
            // by the layer's bf16 noise floor as in the bf16 test
            let tol = if run.contains("bf16") {
                (2.0 * rel_l2(vals(&f, run, layer, "out"), vals(&f, run, layer, "out32"))).max(1e-3)
            } else {
                1e-5
            };
            for chunk in [1usize, 7] {
                let mut lo = 0;
                while lo < ROWS {
                    let hi = (lo + chunk).min(ROWS);
                    let xs = x.slice(&[lo as i64, 0], &[hi as i64, dim as i64]).unwrap();
                    let span = lo * m.k..hi * m.k;
                    let rs = m.route(&xs).unwrap();
                    assert_eq!(rs.ids[..], routes.ids[span.clone()], "{run} L{layer}");
                    for (g, w) in rs.weights.iter().zip(&routes.weights[span.clone()]) {
                        assert!(
                            (g - w).abs() <= 1e-6 * w.abs(),
                            "{run} L{layer}: {g} vs {w}"
                        );
                    }
                    // the router GEMM may round differently per M; given the
                    // same routes, the expert outputs must not move a bit
                    let same = Routes {
                        k: m.k,
                        ids: routes.ids[span.clone()].to_vec(),
                        weights: routes.weights[span].to_vec(),
                        biased: Vec::new(),
                    };
                    let got = read(&m.routed(&xs, &same).unwrap());
                    let want = &all[lo * per_row..hi * per_row];
                    assert!(
                        got.iter()
                            .zip(want)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{run} L{layer} rows {lo}..{hi} (chunk {chunk}) routed bits differ"
                    );
                    let (y, _) = m.forward(&xs).unwrap();
                    let r = rel_l2(&read(&y), &full[lo * dim..hi * dim]);
                    assert!(
                        r <= tol,
                        "{run} L{layer} rows {lo}..{hi}: rel {r:e} > {tol:e}"
                    );
                    lo = hi;
                }
            }
        }
    }
}

/// Routes sorted by expert id keep each (id, weight) pair together.
#[test]
fn by_expert_keeps_pairs() {
    let r = Routes {
        k: 3,
        ids: vec![7, 2, 5, 0, 9, 1],
        weights: vec![0.7, 0.2, 0.5, 0.0, 0.9, 0.1],
        biased: Vec::new(),
    };
    let (ids, w) = r.by_expert();
    assert_eq!(ids, vec![2, 5, 7, 0, 1, 9]);
    assert_eq!(w, vec![0.2, 0.5, 0.7, 0.0, 0.1, 0.9]);
}

#[test]
fn refuses_bad_inputs() {
    let f = fix();
    let (info, raw) = f.gguf.raw("blk.0.ffn_gate_shexp.weight");
    assert!(
        ExpertBank::from_mxfp4(info, &raw).is_err(),
        "Q8_0 is not an expert bank"
    );
    let (info, raw) = f.gguf.raw("blk.0.ffn_gate_exps.weight");
    assert!(
        ExpertBank::from_mxfp4(info, &raw[..raw.len() - 17]).is_err(),
        "short data"
    );
    let mut cfg = f.cfg.clone();
    cfg.n_shared_experts = 2;
    let w =
        MoeWeights::from_gguf(0, |n| f.gguf.dense(n).map(Dense::from), |n| bank(&f, n)).unwrap();
    assert!(Moe::new(&cfg, 0, w).is_err(), "2 shared experts");
    let m = moe(&f, 0);
    let x = input(&f, "c1.fp32_on", 0);
    let mut r = m.route(&x).unwrap();
    r.ids.truncate(r.ids.len() - m.k);
    r.weights.truncate(r.weights.len() - m.k);
    assert!(m.routed(&x, &r).is_err(), "routes for fewer rows");
}
