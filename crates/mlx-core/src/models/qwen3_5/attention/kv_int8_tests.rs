//! The int8 K/V readers against a CPU reference over the SAME dequantized
//! rows: the tensor-op and simdgroup-matrix verify kernels, the vector
//! routes (the decode row included), the route counters, and the eager
//! `Qwen3_5Attention::forward` over an int8 cache against a BF16 cache.
//!
//!   cargo test -p mlx-core --release qwen3_5::attention::kv_int8_tests

use super::*;
use crate::array::kv_int8::{Int8KvRows, segmented_sdpa_int8, segmented_sdpa_int8_with_route};

const D: usize = 256;
const SCALE: f32 = 0.0625;

fn metal() -> bool {
    unsafe { mlx_sys::mlx_metal_is_available() }
}

/// Zero-mean BF16 normal draws from MLX's global stream; `seed_rng` fixes
/// the stream once per test (tests run serially on the shared device).
fn normal(shape: &[i64], _seed: f64, std: f64) -> MxArray {
    MxArray::random_normal(shape, 0.0, std, Some(DType::BFloat16)).unwrap()
}

fn seed_rng(seed: u64) {
    unsafe { mlx_sys::mlx_seed(seed) };
}

/// One verify block: `rows` BF16 queries per head over `prefix` int8 prefix
/// rows and `rows` int8 new rows (the block's own K/V), plus the BF16 rows
/// they were quantized from.
struct Case {
    q: MxArray,
    prefix: Int8KvRows,
    new: Int8KvRows,
    bf16_prefix: (MxArray, MxArray),
    bf16_new: (MxArray, MxArray),
}

fn make_case(q_heads: i64, kv_heads: i64, rows: i64, prefix: i64, seed: f64, q_std: f64) -> Case {
    let q = normal(&[1, q_heads, rows, D as i64], seed, q_std);
    let pk = normal(&[1, kv_heads, prefix, D as i64], seed + 1.0, 1.0);
    let pv = normal(&[1, kv_heads, prefix, D as i64], seed + 2.0, 1.0);
    let nk = normal(&[1, kv_heads, rows, D as i64], seed + 3.0, 1.0);
    let nv = normal(&[1, kv_heads, rows, D as i64], seed + 4.0, 1.0);
    MxArray::eval_arrays(&[&q, &pk, &pv, &nk, &nv]).unwrap();
    Case {
        q,
        prefix: Int8KvRows::quantize(&pk, &pv).unwrap(),
        new: Int8KvRows::quantize(&nk, &nv).unwrap(),
        bf16_prefix: (pk, pv),
        bf16_new: (nk, nv),
    }
}

fn f32s(a: &MxArray) -> Vec<f32> {
    a.to_float32().unwrap().to_vec()
}

/// `[kv_heads][n][D]` f64 rows: int8 x per-row scale in f64 (exact), or the
/// BF16 rows as they are.
fn dequantized_f64(rows: &Int8KvRows) -> (Vec<f64>, Vec<f64>) {
    let (qk, qv) = (f32s(&rows.keys), f32s(&rows.values));
    let (sk, sv) = (f32s(&rows.key_scales), f32s(&rows.value_scales));
    let expand = |q: &[f32], s: &[f32]| -> Vec<f64> {
        q.iter()
            .enumerate()
            .map(|(i, &x)| f64::from(x) * f64::from(s[i / D]))
            .collect()
    };
    (expand(&qk, &sk), expand(&qv, &sv))
}

fn bf16_f64(k: &MxArray, v: &MxArray) -> (Vec<f64>, Vec<f64>) {
    let to = |a: &MxArray| f32s(a).into_iter().map(f64::from).collect::<Vec<_>>();
    (to(k), to(v))
}

/// f64 attention of `q` `[Hq][rows][D]` over prefix then new rows (each
/// `[Hkv][n][D]`); causal blocks let row r see `prefix + r + 1` keys.
fn reference(
    q: &[f32],
    q_heads: usize,
    kv_heads: usize,
    rows: usize,
    prefix: (&[f64], &[f64]),
    new: (&[f64], &[f64]),
    causal: bool,
) -> Vec<f32> {
    let gqa = q_heads / kv_heads;
    let prefix_n = prefix.0.len() / (kv_heads * D);
    let new_n = new.0.len() / (kv_heads * D);
    let n = prefix_n + new_n;
    let key = |h: usize, i: usize, d: usize| -> f64 {
        if i < prefix_n {
            prefix.0[(h * prefix_n + i) * D + d]
        } else {
            new.0[(h * new_n + i - prefix_n) * D + d]
        }
    };
    let value = |h: usize, i: usize, d: usize| -> f64 {
        if i < prefix_n {
            prefix.1[(h * prefix_n + i) * D + d]
        } else {
            new.1[(h * new_n + i - prefix_n) * D + d]
        }
    };
    let mut out = vec![0f32; q_heads * rows * D];
    for qh in 0..q_heads {
        let h = qh / gqa;
        for r in 0..rows {
            let visible = if causal { n - rows + r + 1 } else { n };
            let qrow = &q[(qh * rows + r) * D..(qh * rows + r + 1) * D];
            let mut scores = vec![0f64; visible];
            let mut max = f64::NEG_INFINITY;
            for (i, score) in scores.iter_mut().enumerate() {
                let mut s = 0f64;
                for (d, &qd) in qrow.iter().enumerate() {
                    s += f64::from(qd) * key(h, i, d);
                }
                *score = s * f64::from(SCALE);
                max = max.max(*score);
            }
            let mut sum = 0f64;
            for score in scores.iter_mut() {
                *score = (*score - max).exp();
                sum += *score;
            }
            for d in 0..D {
                let mut acc = 0f64;
                for (i, &p) in scores.iter().enumerate() {
                    acc += p * value(h, i, d);
                }
                out[(qh * rows + r) * D + d] = (acc / sum) as f32;
            }
        }
    }
    out
}

fn bf16_ulp(x: f32) -> f32 {
    let exponent = x.abs().max(f32::from_bits(0x0080_0000)).log2().floor();
    2f32.powf(exponent - 7.0)
}

/// Max and mean |got - expected| in BF16 ulps of the output magnitude.
fn error_ulps(got: &[f32], expected: &[f32]) -> (f32, f64) {
    assert_eq!(got.len(), expected.len());
    let magnitude = expected.iter().fold(0f32, |m, e| m.max(e.abs()));
    let ulp = bf16_ulp(magnitude);
    let mut max = 0f32;
    let mut sum = 0f64;
    for (&g, &e) in got.iter().zip(expected) {
        assert!(g.is_finite(), "non-finite output {g}");
        let diff = (g - e).abs();
        max = max.max(diff);
        sum += f64::from(diff);
    }
    (max / ulp, sum / got.len() as f64 / f64::from(ulp))
}

fn int8_reference(
    c: &Case,
    rows: usize,
    q_heads: usize,
    kv_heads: usize,
    causal: bool,
) -> Vec<f32> {
    let prefix = dequantized_f64(&c.prefix);
    let new = dequantized_f64(&c.new);
    reference(
        &f32s(&c.q),
        q_heads,
        kv_heads,
        rows,
        (&prefix.0, &prefix.1),
        (&new.0, &new.1),
        causal,
    )
}

fn nax_supported(q_heads: i64, kv_heads: i64, rows: i64, total: i64) -> bool {
    let mut plan = [0u32; 6];
    unsafe {
        mlx_sys::mlx_segmented_sdpa_test_nax_plan(
            q_heads as i32,
            kv_heads as i32,
            rows as i32,
            total as i32,
            plan.as_mut_ptr(),
        ) == 1
    }
}

fn tile_supported(q_heads: i64, kv_heads: i64, rows: i64, total: i64) -> bool {
    let mut plan = [0u32; 5];
    unsafe {
        mlx_sys::mlx_segmented_sdpa_test_tile_plan(
            q_heads as i32,
            kv_heads as i32,
            rows as i32,
            total as i32,
            plan.as_mut_ptr(),
        ) == 1
    }
}

/// Every int8 verify route (tensor-op where the device has it, the
/// simdgroup-matrix tile kernel, and the vector routes) stays within a few
/// BF16 ulps of the f64 reference over the same dequantized rows, over
/// prefix boundaries, block widths and sharp softmaxes.
#[test]
#[cfg(target_os = "macos")]
fn int8_verify_routes_match_the_dequantized_reference() -> Result<()> {
    if !metal() {
        eprintln!("SKIP: no Metal device");
        return Ok(());
    }
    seed_rng(0x1A7_0001);
    const HKV: i64 = 4;
    let prefixes = [1_i64, 7, 31, 33, 87, 1023, 1024, 4095, 4096, 6219];
    let layouts: [(i64, Vec<i64>); 2] = [(24, vec![4, 8]), (32, vec![2, 3, 8])];
    let mut worst = [(0f32, 0f64); 3];
    let mut checked = [0usize; 3];
    let (mut any_tile, mut any_nax) = (false, false);
    for (q_heads, rows_list) in &layouts {
        for &rows in rows_list {
            for &prefix in &prefixes {
                for (set, q_std) in [1.0f64, 8.0, 64.0].into_iter().enumerate() {
                    let seed = prefix as f64 * 0.37 + rows as f64 + set as f64 * 100.0;
                    let c = make_case(*q_heads, HKV, rows, prefix, seed, q_std);
                    let expected =
                        int8_reference(&c, rows as usize, *q_heads as usize, HKV as usize, true);
                    // Vector route (route 0): fp32 P, so the tightest.
                    let got = f32s(&segmented_sdpa_int8_with_route(
                        &c.q, &c.prefix, &c.new, SCALE, true, 0,
                    )?);
                    let (max, mean) = error_ulps(&got, &expected);
                    assert!(
                        max <= 2.0 && mean <= 0.35,
                        "vector q_heads={q_heads} rows={rows} prefix={prefix} set={set}: \
                         max {max} / mean {mean:.4} ulps"
                    );
                    worst[0] = (worst[0].0.max(max), worst[0].1.max(mean));
                    checked[0] += 1;
                    let tile_here = tile_supported(*q_heads, HKV, rows, prefix + rows);
                    any_tile |= tile_here;
                    if tile_here {
                        let got = f32s(&segmented_sdpa_int8_with_route(
                            &c.q, &c.prefix, &c.new, SCALE, true, 1,
                        )?);
                        let (max, mean) = error_ulps(&got, &expected);
                        assert!(
                            max <= 3.0 && mean <= 0.35,
                            "tile q_heads={q_heads} rows={rows} prefix={prefix} set={set}: \
                             max {max} / mean {mean:.4} ulps"
                        );
                        worst[1] = (worst[1].0.max(max), worst[1].1.max(mean));
                        checked[1] += 1;
                    }
                    let nax_here = nax_supported(*q_heads, HKV, rows, prefix + rows);
                    any_nax |= nax_here;
                    if nax_here {
                        let got = f32s(&segmented_sdpa_int8_with_route(
                            &c.q, &c.prefix, &c.new, SCALE, true, 2,
                        )?);
                        let (max, mean) = error_ulps(&got, &expected);
                        assert!(
                            max <= 3.0 && mean <= 0.35,
                            "nax q_heads={q_heads} rows={rows} prefix={prefix} set={set}: \
                             max {max} / mean {mean:.4} ulps"
                        );
                        worst[2] = (worst[2].0.max(max), worst[2].1.max(mean));
                        checked[2] += 1;
                    }
                }
            }
        }
    }
    // The vector route runs on every Metal device; a block route only
    // where its probe says this device can launch the kernel — a
    // virtualized GPU without GPUFamilyApple7 never probes tile-supported,
    // and checked[1] == 0 is the device's answer rather than a regression.
    assert!(checked[0] > 0, "the vector route must run somewhere");
    assert!(
        checked[1] > 0 || !any_tile,
        "the tile route is supported on this device but never ran"
    );
    assert!(
        checked[2] > 0 || !any_nax,
        "the nax route is supported on this device but never ran"
    );
    eprintln!(
        "int8 verify vs f64 reference (BF16 ulps of output magnitude): vector {} blocks worst \
         max {} / mean {:.4}; tile {} blocks worst max {} / mean {:.4}; nax {} blocks worst max \
         {} / mean {:.4}",
        checked[0],
        worst[0].0,
        worst[0].1,
        checked[1],
        worst[1].0,
        worst[1].1,
        checked[2],
        worst[2].0,
        worst[2].1
    );
    Ok(())
}

/// The decode row (one query, no causal mask) through the production int8
/// entry over short and long prefixes, on both GQA layouts.
#[test]
#[cfg(target_os = "macos")]
fn int8_decode_row_matches_the_dequantized_reference() -> Result<()> {
    if !metal() {
        return Ok(());
    }
    seed_rng(0x1A7_0002);
    let mut worst = (0f32, 0f64);
    for (q_heads, kv_heads) in [(24_i64, 4_i64), (32, 8), (8, 8)] {
        for prefix in [1_i64, 5, 255, 1000, 5000, 9000] {
            let c = make_case(
                q_heads,
                kv_heads,
                1,
                prefix,
                prefix as f64 + q_heads as f64,
                4.0,
            );
            let expected = int8_reference(&c, 1, q_heads as usize, kv_heads as usize, false);
            let got = f32s(&segmented_sdpa_int8(&c.q, &c.prefix, &c.new, SCALE, false)?);
            let (max, mean) = error_ulps(&got, &expected);
            assert!(
                max <= 2.0 && mean <= 0.35,
                "decode q_heads={q_heads} kv={kv_heads} prefix={prefix}: max {max} / mean \
                 {mean:.4} ulps"
            );
            worst = (worst.0.max(max), worst.1.max(mean));
        }
    }
    eprintln!(
        "int8 decode row vs f64 reference: worst max {} / mean {:.4} BF16 ulps",
        worst.0, worst.1
    );
    Ok(())
}

/// Kernel-level quality gate (a): the int8 verify output against the BF16
/// verify output on the same random K/V, next to the error int8 rounding
/// alone predicts (the f64 reference over int8 rows against the f64
/// reference over the BF16 rows). The kernel adds nothing material beyond
/// the quantization itself.
#[test]
#[cfg(target_os = "macos")]
fn int8_verify_error_against_bf16_is_the_quantization_noise() -> Result<()> {
    if !metal() {
        return Ok(());
    }
    seed_rng(0x1A7_0003);
    const Q_HEADS: i64 = 24;
    const HKV: i64 = 4;
    const ROWS: i64 = 8;
    for (prefix, q_std) in [(1024_i64, 1.0f64), (1024, 8.0), (8192, 1.0), (8192, 8.0)] {
        let c = make_case(Q_HEADS, HKV, ROWS, prefix, prefix as f64 + q_std, q_std);
        let (pk, pv) = bf16_f64(&c.bf16_prefix.0, &c.bf16_prefix.1);
        let (nk, nv) = bf16_f64(&c.bf16_new.0, &c.bf16_new.1);
        let q = f32s(&c.q);
        let ref_bf16 = reference(
            &q,
            Q_HEADS as usize,
            HKV as usize,
            ROWS as usize,
            (&pk, &pv),
            (&nk, &nv),
            true,
        );
        let ref_int8 = int8_reference(&c, ROWS as usize, Q_HEADS as usize, HKV as usize, true);
        let magnitude = ref_bf16.iter().fold(0f32, |m, e| m.max(e.abs()));
        let rel = |got: &[f32], exp: &[f32]| -> (f32, f32) {
            let mut max = 0f32;
            let mut sum = 0f64;
            for (g, e) in got.iter().zip(exp) {
                let d = (g - e).abs();
                max = max.max(d);
                sum += f64::from(d);
            }
            (max / magnitude, (sum / got.len() as f64) as f32 / magnitude)
        };
        let expected = rel(&ref_int8, &ref_bf16);
        // BF16 verify over the BF16 rows (production entry).
        let bf16_out = {
            let handle = unsafe {
                mlx_sys::mlx_segmented_sdpa_forward(
                    c.q.as_raw_ptr(),
                    c.bf16_prefix.0.as_raw_ptr(),
                    c.bf16_prefix.1.as_raw_ptr(),
                    c.bf16_new.0.as_raw_ptr(),
                    c.bf16_new.1.as_raw_ptr(),
                    SCALE,
                    true,
                )
            };
            f32s(&MxArray::from_handle(handle, "bf16 verify")?)
        };
        let int8_out = f32s(&segmented_sdpa_int8(&c.q, &c.prefix, &c.new, SCALE, true)?);
        let kernel = rel(&int8_out, &bf16_out);
        let kernel_vs_own_ref = rel(&int8_out, &ref_int8);
        eprintln!(
            "int8 vs bf16 verify, prefix={prefix} q_std={q_std}: kernel max {:.3e} / mean {:.3e} \
             of output magnitude; int8 rounding alone max {:.3e} / mean {:.3e}; kernel vs its \
             own dequantized reference max {:.3e} / mean {:.3e}",
            kernel.0, kernel.1, expected.0, expected.1, kernel_vs_own_ref.0, kernel_vs_own_ref.1
        );
        // The kernel's error against BF16 is the quantization's, plus the
        // BF16-P block-kernel noise (a few ulps of the magnitude).
        let ulp = bf16_ulp(magnitude) / magnitude;
        assert!(
            kernel.0 <= expected.0 + 4.0 * ulp && kernel.1 <= expected.1 * 1.5 + ulp,
            "prefix={prefix} q_std={q_std}: kernel error {kernel:?} exceeds quantization noise \
             {expected:?} + kernel tolerance"
        );
    }
    Ok(())
}

/// The int8 dispatcher records its own counter and the block route it took;
/// the forced routes take exactly their own kernel.
#[test]
#[cfg(target_os = "macos")]
fn int8_routes_are_counted() -> Result<()> {
    if !metal() {
        return Ok(());
    }
    seed_rng(0x1A7_0004);
    let c = make_case(24, 4, 8, 1000, 3.0, 1.0);
    let count =
        |family: &std::ffi::CStr| unsafe { mlx_sys::mlx_test_kquant_family_count(family.as_ptr()) };
    if tile_supported(24, 4, 8, 1008) {
        unsafe { mlx_sys::mlx_test_kquant_counting(true) };
        let out = segmented_sdpa_int8_with_route(&c.q, &c.prefix, &c.new, SCALE, true, 1)
            .and_then(|o| o.to_float32());
        let counts = [
            count(c"segmented_sdpa_int8"),
            count(c"segmented_sdpa_route_tile"),
            count(c"segmented_sdpa_verify_tile_2pass_1"),
            count(c"segmented_sdpa_route_nax"),
        ];
        unsafe { mlx_sys::mlx_test_kquant_counting(false) };
        out?;
        assert_eq!(counts, [1, 1, 1, 0], "tile route counters");
    }
    if nax_supported(24, 4, 8, 1008) {
        unsafe { mlx_sys::mlx_test_kquant_counting(true) };
        let out = segmented_sdpa_int8_with_route(&c.q, &c.prefix, &c.new, SCALE, true, 2)
            .and_then(|o| o.to_float32());
        let counts = [
            count(c"segmented_sdpa_int8"),
            count(c"segmented_sdpa_route_nax"),
            count(c"segmented_sdpa_verify_nax_2pass_1"),
            count(c"segmented_sdpa_route_tile"),
        ];
        unsafe { mlx_sys::mlx_test_kquant_counting(false) };
        out?;
        assert_eq!(counts, [1, 1, 1, 0], "nax route counters");
    }
    // The vector route never reaches a block kernel.
    unsafe { mlx_sys::mlx_test_kquant_counting(true) };
    let out = segmented_sdpa_int8_with_route(&c.q, &c.prefix, &c.new, SCALE, true, 0)
        .and_then(|o| o.to_float32());
    let counts = [
        count(c"segmented_sdpa_int8"),
        count(c"segmented_sdpa_route_nax"),
        count(c"segmented_sdpa_route_tile"),
        count(c"segmented_sdpa_route_unified"),
    ];
    unsafe { mlx_sys::mlx_test_kquant_counting(false) };
    out?;
    assert_eq!(
        counts,
        [1, 0, 0, 0],
        "vector route counters (no unified int8 kernel)"
    );
    Ok(())
}

/// A production-geometry attention layer (24 x 4 heads, D = 256) over an
/// int8 cache against the same layer over a BF16 cache: a prefill chunk, a
/// second chunk (dequantized prefix + fresh rows), decode rows (the int8
/// decode kernel) and a verify block (the int8 block kernels). The outputs
/// stay within the quantization noise of each other and the int8 cache
/// holds exactly the quantized rows.
#[test]
#[cfg(target_os = "macos")]
fn eager_forward_over_an_int8_cache_tracks_the_bf16_cache() -> Result<()> {
    if !metal() {
        return Ok(());
    }
    seed_rng(0x1A7_0005);
    let cfg = Qwen3_5Config {
        hidden_size: 64,
        num_heads: 24,
        num_kv_heads: 4,
        head_dim: 256,
        partial_rotary_factor: 0.25,
        ..super::tests::tiny_cfg()
    };
    let mut attn = Qwen3_5Attention::new(&cfg)?;
    let (h, d, hidden, kv) = (24_i64, 256_i64, 64_i64, 4_i64);
    attn.set_q_proj_weight(&normal(&[2 * h * d, hidden], 1.0, 0.08))?;
    attn.set_k_proj_weight(&normal(&[kv * d, hidden], 2.0, 0.08))?;
    attn.set_v_proj_weight(&normal(&[kv * d, hidden], 3.0, 0.08))?;
    attn.set_o_proj_weight(&normal(&[hidden, h * d], 4.0, 0.02))?;
    // BF16 norms keep the queries/keys BF16 (the production dtype the int8
    // kernels require; f32 would take the dequantizing fallback).
    let ones = MxArray::ones(&[d], Some(DType::BFloat16))?;
    attn.set_q_norm_weight(&ones, DType::BFloat16)?;
    attn.set_k_norm_weight(&ones, DType::BFloat16)?;

    let mut bf16 = KVCache::new();
    let mut int8 = KVCache::with_format(KvFormat::Int8);
    let mut worst = 0f32;
    let compare = |label: &str,
                   x: &MxArray,
                   tolerance: f32,
                   bf16: &mut KVCache,
                   int8: &mut KVCache,
                   worst: &mut f32|
     -> Result<()> {
        let a = f32s(&attn.forward(x, None, Some(bf16), None)?);
        let b = f32s(&attn.forward(x, None, Some(int8), None)?);
        assert_eq!(bf16.get_offset(), int8.get_offset());
        let magnitude = a.iter().fold(0f32, |m, v| m.max(v.abs()));
        let max = a
            .iter()
            .zip(&b)
            .fold(0f32, |m, (p, q)| m.max((p - q).abs()))
            / magnitude;
        eprintln!("{label}: max |bf16 - int8| = {max:.3e} of output magnitude");
        assert!(max <= tolerance, "{label}: {max} > {tolerance}");
        *worst = worst.max(max);
        Ok(())
    };
    let x = |seq: i64, seed: f64| normal(&[1, seq, hidden], seed, 1.0);
    // First prefill chunk (empty prefix: BF16 route, exact modulo nothing).
    compare(
        "prefill 300",
        &x(300, 10.0),
        0.02,
        &mut bf16,
        &mut int8,
        &mut worst,
    )?;
    // Second chunk: dequantized int8 prefix + fresh BF16 rows.
    compare(
        "prefill +200",
        &x(200, 11.0),
        0.03,
        &mut bf16,
        &mut int8,
        &mut worst,
    )?;
    // Decode rows: the int8 vector kernel over 500+ rows.
    for step in 0..3 {
        let label = format!("decode {step}");
        compare(
            &label,
            &x(1, 20.0 + step as f64),
            0.03,
            &mut bf16,
            &mut int8,
            &mut worst,
        )?;
    }
    // Verify blocks: the int8 block kernels (or vector split below the
    // calibrated crossover), then a rejection rewind and a re-append.
    compare(
        "verify 8",
        &x(8, 30.0),
        0.03,
        &mut bf16,
        &mut int8,
        &mut worst,
    )?;
    bf16.trim(bf16.get_offset() - 5);
    int8.trim(int8.get_offset() - 5);
    compare(
        "verify 8 after trim",
        &x(8, 31.0),
        0.03,
        &mut bf16,
        &mut int8,
        &mut worst,
    )?;
    compare(
        "verify 5",
        &x(5, 32.0),
        0.03,
        &mut bf16,
        &mut int8,
        &mut worst,
    )?;
    assert_eq!(int8.format(), KvFormat::Int8);
    let view = int8.int8_view().expect("int8 cache allocated");
    assert_eq!(view.keys.dtype()?, DType::Int8);
    assert_eq!(view.len()?, int8.get_offset() as i64);
    eprintln!("eager int8 cache vs bf16 cache: worst {worst:.3e} of output magnitude");
    Ok(())
}

/// Manual: the per-layer cost of the prefill reader's dequantize + concat
/// over a 30K int8 prefix, with and without `clear_cache` between
/// iterations (the chunked prefill clears the pool after every chunk).
///
///   cargo test -p mlx-core --release kv_int8_tests::profile_prefill_dequant -- --ignored --nocapture
#[test]
#[ignore]
#[cfg(target_os = "macos")]
fn profile_prefill_dequant() -> Result<()> {
    if !metal() {
        return Ok(());
    }
    seed_rng(7);
    const P: i64 = 30_720;
    const T: i64 = 2048;
    let pk = normal(&[1, 4, P, D as i64], 0.0, 1.0);
    let pv = normal(&[1, 4, P, D as i64], 0.0, 1.0);
    let rows = Int8KvRows::quantize(&pk, &pv)?;
    let fresh_k = normal(&[1, 4, T, D as i64], 0.0, 1.0);
    let fresh_v = normal(&[1, 4, T, D as i64], 0.0, 1.0);
    MxArray::eval_arrays(&[
        &rows.keys,
        &rows.values,
        &rows.key_scales,
        &rows.value_scales,
    ])?;
    MxArray::eval_arrays(&[&fresh_k, &fresh_v])?;
    for clear in [false, true] {
        let mut total = 0f64;
        let mut best = f64::INFINITY;
        for i in 0..12 {
            if clear {
                crate::array::clear_cache();
            }
            let t = std::time::Instant::now();
            let (dk, dv) = rows.dequantize()?;
            let k = MxArray::concatenate(&dk, &fresh_k, 2)?;
            let v = MxArray::concatenate(&dv, &fresh_v, 2)?;
            MxArray::eval_arrays(&[&k, &v])?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if i >= 2 {
                total += ms;
                best = best.min(ms);
            }
        }
        eprintln!(
            "dequant+concat P={P} (one layer, K and V): clear_cache={clear}: mean {:.2} ms, best \
             {best:.2} ms",
            total / 10.0
        );
    }
    Ok(())
}
