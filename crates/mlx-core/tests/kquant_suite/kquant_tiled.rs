#![cfg_attr(not(target_os = "macos"), allow(dead_code, unused_imports))]
//! The Tiled64 K-quant weight layout (`mlx_kquant.h`, mode suffix `@t64`):
//! the row-major `.weight` / `.scales` / `.biases` bytes permuted so the 64
//! rows of a tile interleave per 32-code unit. A pure permutation, so every
//! ported kernel (`qmv_wide_t64`, `qmm_m8_nax_t64`, `qmm_t_nax_t64`,
//! `qmm_t_splitk_t64`, the CPU reference) must give the same bits as its
//! row-major twin, and no tiled tensor may reach an unported kernel.
//!
//!   cargo test -p mlx-core --release --test kquant_suite kquant_tiled:: -- --nocapture

use std::ffi::CString;

use crate::kquant_support::*;
use mlx_core::array::{DType, MxArray};
use mlx_core::models::quant_dispatch::{
    KQUANT_TILED_SUFFIX, kquant_tile_rows, kquant_tileable, kquant_untile_rows,
};

/// (K, N): the Qwen3.8-27B projections (q4k/q5k/q6k) the tiling targets.
const SHAPES: [(i64, i64); 6] = [
    (5120, 17408),
    (17408, 5120),
    (5120, 6144),
    (6144, 5120),
    (5120, 1024),
    (5120, 16384),
];

/// `BF16_TILE_TOL` of kquant_m8_nax.rs: the tensor-op kernel rounds every
/// decoded weight to half and reassociates the K sum across splits.
const BF16_TILE_TOL: f32 = 3e-2;

/// The same bytes in the Tiled64 order: codes per unit, companions per
/// super-block (`.scales`: super_ratio * bytes per group; `.biases`: its
/// entries per super-block).
fn tiled(w: &Weights, kq: &KQuant) -> Weights {
    let t = Weights {
        w: kquant_tile_rows(&w.w, i64::from(kq.bits)).expect("tile weight"),
        scales: kquant_tile_rows(&w.scales, kq.super_ratio() * kq.scale_bytes_per_group())
            .expect("tile scales"),
        biases: kquant_tile_rows(&w.biases, kq.bias_entries_per_super_block())
            .expect("tile biases"),
    };
    t.w.eval();
    t.scales.eval();
    t.biases.eval();
    t
}

fn tiled_mode(kq: &KQuant) -> CString {
    CString::new(format!("{}{KQUANT_TILED_SUFFIX}", kq.mode)).expect("mode")
}

fn qmm_mode(
    x: &MxArray,
    w: &Weights,
    transpose: bool,
    kq: &KQuant,
    mode: &CString,
    device: i32,
) -> *mut mlx_sys::mlx_array {
    // SAFETY: every handle outlives the call.
    unsafe {
        mlx_sys::mlx_test_kquant_quantized_matmul(
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            transpose,
            kq.group_size,
            kq.bits,
            mode.as_ptr(),
            device,
        )
    }
}

fn qmm_tiled(x: &MxArray, w: &Weights, kq: &KQuant, device: i32) -> *mut mlx_sys::mlx_array {
    qmm_mode(x, w, true, kq, &tiled_mode(kq), device)
}

/// `read_output` widens 16-bit outputs to u32 bit patterns; the activations
/// here are bfloat16, so every output (CPU included) is bfloat16 bits.
fn bf16(bits: u32) -> f32 {
    f32::from_bits(bits << 16)
}

/// max |a - b| over bfloat16 bit patterns, and the peak |b|. Panics if any
/// element is non-finite or if the reference is identically zero.
fn worst_abs(ours: &[u32], reference: &[u32]) -> (f32, f32) {
    assert_eq!(ours.len(), reference.len(), "output lengths differ");
    let mut worst = 0f32;
    let mut peak = 0f32;
    for (i, (&o, &r)) in ours.iter().zip(reference).enumerate() {
        let ov = bf16(o);
        let rv = bf16(r);
        assert!(
            ov.is_finite() && rv.is_finite(),
            "non-finite bf16 element at {i}: ours={ov} (0x{o:04x}), reference={rv} (0x{r:04x})"
        );
        worst = worst.max((ov - rv).abs());
        peak = peak.max(rv.abs());
    }
    assert!(peak > 0.0, "reference output is identically zero");
    (worst, peak)
}

/// Every row-major kernel family a tiled tensor must never reach.
const ROW_MAJOR_FAMILIES: [&str; 11] = [
    "qmv",
    "qmv_fast",
    "qmv_wide",
    "qmv_sg8",
    "qmm_m8_nax",
    "qmm_t",
    "qmm_t_nax",
    "qmm_t_splitk",
    "qmm_n",
    "qvm",
    "qvm_split_k",
];

fn assert_no_row_major_route(what: &str) {
    for family in ROW_MAJOR_FAMILIES {
        assert_eq!(
            family_count(family),
            0,
            "{what}: a tiled tensor reached the row-major {family} kernel"
        );
    }
}

/// (c) Tiling then untiling every array restores the original bytes, for
/// every mode's unit size, on a tile-boundary-exercising shape.
#[test]
fn tile_round_trip_restores_bytes() {
    for (ki, kq) in KQUANTS.iter().enumerate() {
        let (k, n) = (1024i64, 192i64);
        assert!(kquant_tileable(n, k));
        let w = weights(kq, &[n], k, 0x7100 + ki as u32);
        let t = tiled(&w, kq);
        let back = [
            kquant_untile_rows(&t.w, i64::from(kq.bits)).unwrap(),
            kquant_untile_rows(&t.scales, kq.super_ratio() * kq.scale_bytes_per_group()).unwrap(),
            kquant_untile_rows(&t.biases, kq.bias_entries_per_super_block()).unwrap(),
        ];
        let originals = [&w.w, &w.scales, &w.biases];
        for (i, (b, o)) in back.iter().zip(originals).enumerate() {
            // Compare through a common integer view of the raw bytes.
            let (bb, ob) = match o.dtype().unwrap() {
                DType::Uint32 => (
                    b.to_uint32().unwrap().to_vec(),
                    o.to_uint32().unwrap().to_vec(),
                ),
                _ => {
                    let as_u32 = |a: &MxArray| {
                        let f = a.astype(DType::Float32).unwrap();
                        f.to_float32()
                            .unwrap()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    };
                    (as_u32(b), as_u32(o))
                }
            };
            assert_eq!(
                bb, ob,
                "{} array {i}: tile/untile is not the identity",
                kq.mode
            );
            assert_eq!(
                b.shape().unwrap().to_vec(),
                o.shape().unwrap().to_vec(),
                "{} array {i}: shape changed",
                kq.mode
            );
        }
        // And the tiled bytes really moved: tile t, unit 0 holds rows 64t..
        let orig = w.w.to_uint32().unwrap().to_vec();
        let moved = t.w.to_uint32().unwrap().to_vec();
        let units = k / 32;
        let u = i64::from(kq.bits);
        let cols = units * u;
        // Row 65, unit 3, its last word -> tile 1, unit 3, row-in-tile 1.
        let (row, unit, word) = (65i64, 3i64, u - 1);
        let src = (row * cols + unit * u + word) as usize;
        let dst = ((((row / 64) * units + unit) * 64 + row % 64) * u + word) as usize;
        assert_eq!(moved[dst], orig[src], "{}: tiled address differs", kq.mode);
        println!("  {:<6} tile/untile round trip ok", kq.mode);
    }
}

/// The CPU reference reads both layouts with the same arithmetic order, so
/// the tiled result is bit-identical, in every activation dtype.
#[test]
fn cpu_tiled_matches_cpu_row_major_bitwise() {
    for (ki, kq) in KQUANTS.iter().enumerate() {
        let (k, n) = (768i64, 128i64);
        let w = weights(kq, &[n], k, 0x7200 + ki as u32);
        let t = tiled(&w, kq);
        for dtype in DTYPES {
            for m in [1i64, 3, 8] {
                let x = activation(&[m, k], 0x7300 + ki as u32 + m as u32, dtype);
                let ours = qmm_tiled(&x, &t, kq, CPU);
                let reference = quantized_matmul(&x, &w, true, kq, CPU);
                assert_identical(&format!("cpu {} M={m} {dtype:?}", kq.mode), ours, reference);
            }
        }
        println!("  {:<6} cpu tiled == cpu row-major", kq.mode);
    }
}

/// (a) M = 2..7 and 9 (under the device's qmv batch limit) take qmv_wide in
/// both layouts, same lane map and sum order: bit-identical on the Qwen3.8
/// shapes.
#[cfg(target_os = "macos")]
#[test]
fn qmv_wide_tiled_is_bit_identical() {
    assert!(gpu_gen() > 0, "no Metal device");
    if gpu_gen() < 15 {
        eprintln!("skipping: row-major qmv_wide routes on gen-15+ only");
        return;
    }
    for (ki, kq) in KQUANTS.iter().enumerate() {
        for (si, &(k, n)) in SHAPES.iter().enumerate() {
            let w = weights(kq, &[n], k, 0x7400 + (ki * 16 + si) as u32);
            let t = tiled(&w, kq);
            for m in [3i64, 5, 7, 9] {
                // A tensor-op tier, or the GEMM from the device's qmv batch
                // limit (6 on gen-13/14): not qmv_wide, covered elsewhere.
                if tiled_family(m, n, k, kq, "gemm") != "qmv_wide_t64" {
                    continue;
                }
                let x = activation(&[m, k], 0x7500 + (ki * 16 + si) as u32, DType::BFloat16);
                start_counting();
                let ours = qmm_tiled(&x, &t, kq, GPU);
                let (_, _, ours_bits) = read_output("tiled", ours);
                let what = format!("{} K={k} N={n} M={m}", kq.mode);
                assert_eq!(
                    family_count("qmv_wide_t64"),
                    1,
                    "{what}: must take qmv_wide_t64"
                );
                assert_no_row_major_route(&what);
                stop_counting();
                start_counting();
                let reference = quantized_matmul(&x, &w, true, kq, GPU);
                let (_, _, ref_bits) = read_output("row-major", reference);
                assert!(
                    family_count("qmv_wide") == 1,
                    "{what}: reference must take qmv_wide"
                );
                stop_counting();
                assert_eq!(
                    ours_bits, ref_bits,
                    "{what}: tiled qmv_wide differs from row-major"
                );
            }
        }
        println!("  {:<6} qmv_wide tiled == row-major (M=3,5,7,9)", kq.mode);
    }
}

/// M = 1: the tiled matvec (qmv_t64, lane = row) is the only tiled M = 1
/// route and is checked against the CPU reference. Every mode covers both
/// `k_splits = 8` (`K < 8192`) and `k_splits = 16` (`K >= 8192`) on compact
/// tiled shapes; three decode-class representatives (q4k, iq4nl, iq2xxs) also
/// cover the full Qwen3.8 shapes.
#[cfg(target_os = "macos")]
#[test]
fn qmv_m1_tiled_matches_cpu() {
    assert!(gpu_gen() > 0, "no Metal device");
    const COMPACT_M1_SHAPES: [(i64, i64); 2] = [(512, 1024), (8192, 128)];
    for (ki, kq) in KQUANTS.iter().enumerate() {
        let mut shapes: Vec<(i64, i64)> = COMPACT_M1_SHAPES.to_vec();
        if matches!(kq.mode, "q4k" | "iq4nl" | "iq2xxs") {
            shapes.extend_from_slice(&SHAPES[..4]);
        }
        for (si, (k, n)) in shapes.into_iter().enumerate() {
            let w = weights(kq, &[n], k, 0x7600 + (ki * 16 + si) as u32);
            let t = tiled(&w, kq);
            let x1 = activation(&[1, k], 0x7700 + (ki * 16 + si) as u32, DType::BFloat16);
            let (_, _, cpu) = read_output("cpu", quantized_matmul(&x1, &w, true, kq, CPU));
            start_counting();
            let (_, _, ours) = read_output("tiled", qmm_tiled(&x1, &t, kq, GPU));
            let what = format!("{} K={k} N={n} M=1", kq.mode);
            assert_eq!(family_count("qmv_t64"), 1, "{what}: must take qmv_t64");
            assert_no_row_major_route(&what);
            stop_counting();
            let (worst, peak) = worst_abs(&ours, &cpu);
            assert!(
                worst / peak < 1e-2,
                "{what}: tiled M=1 off the CPU reference by {worst} (peak {peak})"
            );
        }
        println!("  {:<6} M=1 tiled: qmv_t64 ~cpu", kq.mode);
    }
}

/// (b) M = 8: every tiled mode takes the tensor op, within the tile
/// tolerance of the CPU reference. q3k, q2k, iq4nl and the grid formats also
/// take it row-major, and there the two layouts are bit-identical (same split
/// count, same order).
#[cfg(target_os = "macos")]
#[test]
fn m8_nax_tiled_matches_cpu_and_row_major_tensor_op() {
    assert!(gpu_gen() > 0, "no Metal device");
    if !nax_available() {
        eprintln!("skipping: no NAX tensor-op kernels on this host");
        return;
    }
    for (ki, kq) in KQUANTS.iter().enumerate() {
        let mut worst_rel = 0f32;
        for (si, &(k, n)) in SHAPES.iter().enumerate() {
            let w = weights(kq, &[n], k, 0x7800 + (ki * 16 + si) as u32);
            let t = tiled(&w, kq);
            let x = activation(&[8, k], 0x7900 + (ki * 16 + si) as u32, DType::BFloat16);
            let what = format!("{} K={k} N={n} M=8", kq.mode);

            // Every K-quant mode takes the 8-row tier at M = 8 (the grid rule
            // is the affine contracts'); derived from the predicate anyway.
            let takes_tensor_op = tensor_op_tier(8, n, k, kq).is_some();
            let gemm = if nax_available() {
                "qmm_t_nax_t64"
            } else {
                "qmm_t_t64"
            };
            let family = tiled_family(8, n, k, kq, gemm);
            start_counting();
            let (_, _, ours) = read_output("tiled", qmm_tiled(&x, &t, kq, GPU));
            let took = if family == gemm && family_count("qmm_t_splitk_t64") == 1 {
                "qmm_t_splitk_t64"
            } else {
                family
            };
            assert_eq!(family_count(took), 1, "{what}: must take {took}");
            assert_no_row_major_route(&what);
            let splits = [1u64, 2, 4, 8]
                .into_iter()
                .find(|s| family_count(&format!("qmm_m8_nax_splits{s}")) > 0)
                .unwrap_or(0);
            stop_counting();

            if takes_tensor_op && (matches!(kq.mode, "q3k" | "q2k" | "iq4nl") || kq.is_grid()) {
                start_counting();
                let (_, _, reference) =
                    read_output("row-major m8", quantized_matmul(&x, &w, true, kq, GPU));
                assert_eq!(
                    family_count("qmm_m8_nax"),
                    1,
                    "{what}: row-major {} must take qmm_m8_nax",
                    kq.mode
                );
                stop_counting();
                assert_eq!(
                    ours, reference,
                    "{what}: tiled m8_nax differs from row-major m8_nax"
                );
            }

            let (_, _, cpu) = read_output("cpu", quantized_matmul(&x, &w, true, kq, CPU));
            let (worst, peak) = worst_abs(&ours, &cpu);
            let rel = worst / peak;
            assert!(
                rel <= BF16_TILE_TOL,
                "{what}: rel {rel:e} > {BF16_TILE_TOL:e}"
            );
            worst_rel = worst_rel.max(rel);
            println!(
                "  {:<6} K={k:>5} N={n:>5} splits={splits} rel {rel:.2e}",
                kq.mode
            );
        }
        println!("  {:<6} worst rel {worst_rel:.2e}", kq.mode);
    }
}

/// (b') M = 12, 24, 32, 40 for every K-quant mode: the 16- and 32-row
/// tensor-op tiers are the affine contracts' (kquant_affine_tiled.rs), so a
/// K-quant mode never takes one — the route is qmv_wide_t64 under the device's
/// qmv batch limit, else the GEMM (M = 40 included: no 32-row tier plus a
/// tail) — and the result stays within the tile tolerance of the CPU
/// reference and of MLX's row-major route.
#[cfg(target_os = "macos")]
#[test]
fn kquant_modes_take_no_row_tier_past_m8() {
    assert!(gpu_gen() > 0, "no Metal device");
    if !nax_available() {
        eprintln!("skipping: no NAX tensor-op kernels on this host");
        return;
    }
    let gemm = "qmm_t_nax_t64";
    for (ki, kq) in KQUANTS.iter().enumerate() {
        let mut worst_rel = 0f32;
        let mut routes = 0;
        // One shape per split count of the grid rule. `activation` draws its
        // values in order, so the same seed at every M is a row prefix of the
        // M = 40 input and one CPU reference per shape serves every M.
        for (si, &(k, n)) in SHAPES[..3].iter().enumerate() {
            let w = weights(kq, &[n], k, 0x7c00 + (ki * 16 + si) as u32);
            let t = tiled(&w, kq);
            let seed = 0x7d00 + (ki * 16 + si) as u32;
            let x40 = activation(&[40, k], seed, DType::BFloat16);
            let (_, _, cpu40) = read_output("cpu", quantized_matmul(&x40, &w, true, kq, CPU));
            for m in [12i64, 24, 32, 40] {
                let x = activation(&[m, k], seed, DType::BFloat16);
                let what = format!("{} K={k} N={n} M={m}", kq.mode);
                let tier = tensor_op_tier(m, n, k, kq);
                assert!(
                    tier.is_none(),
                    "{what}: the 16/32-row tiers are affine-only, got {tier:?}"
                );
                let family = tiled_family(m, n, k, kq, gemm);
                start_counting();
                let (_, _, ours) = read_output("tiled", qmm_tiled(&x, &t, kq, GPU));
                let took = if family == gemm && family_count("qmm_t_splitk_t64") == 1 {
                    "qmm_t_splitk_t64"
                } else {
                    family
                };
                assert_eq!(family_count(took), 1, "{what}: must take {took}");
                assert_eq!(
                    family_count("qmm_m32_nax_t64") + family_count("qmm_m16_nax_t64"),
                    0,
                    "{what}: no row tier"
                );
                assert_no_row_major_route(&what);
                stop_counting();
                routes += 1;

                let cpu = &cpu40[..(m * n) as usize];
                let (worst, peak) = worst_abs(&ours, cpu);
                let rel = worst / peak;
                assert!(
                    rel <= BF16_TILE_TOL,
                    "{what}: rel {rel:e} > {BF16_TILE_TOL:e} off the CPU reference"
                );
                worst_rel = worst_rel.max(rel);

                let (_, _, row_major) =
                    read_output("row-major", quantized_matmul(&x, &w, true, kq, GPU));
                let (worst, peak) = worst_abs(&ours, &row_major);
                let rel = worst / peak;
                assert!(
                    rel <= BF16_TILE_TOL,
                    "{what}: rel {rel:e} > {BF16_TILE_TOL:e} off the row-major route"
                );
            }
        }
        println!(
            "  {:<6} M = 12..40 on {routes} (shape, M) cases, no row tier, worst rel {worst_rel:.2e}",
            kq.mode
        );
    }
}

/// (a) Prefill: compact tiled shapes verify `qmm_t_nax`/`qmm_t` at M = 64
/// (split_k = 1 on both NAX and non-NAX: 2 x 132 = 264 >= 257 tiles), the
/// Metal 4 BM = 128 prefill shader at M = 130 >= 65 (1 full tile + 2-row tail;
/// 5 x 52 = 260 >= 257 tiles on non-NAX), and `qmm_t_splitk` across all 15
/// modes; three decode-class representatives (q4k, iq4nl, iq2xxs) also verify
/// the full Qwen3.8 shapes (including M = 2048).
#[cfg(target_os = "macos")]
#[test]
fn qmm_tiled_is_bit_identical() {
    assert!(gpu_gen() > 0, "no Metal device");
    const COMPACT_QMM: [(i64, i64, i64, &str); 3] = [
        (512, 4224, 64, "qmm_t_nax"),
        (512, 1664, 130, "qmm_t_nax"),
        (512, 512, 40, "qmm_t_splitk"),
    ];
    const FULL_QMM: [(i64, i64, i64, &str); 4] = [
        (5120, 5120, 64, "qmm_t_nax"),
        (5120, 17408, 64, "qmm_t_nax"),
        (5120, 6144, 2048, "qmm_t_nax"),
        (5120, 1280, 40, "qmm_t_splitk"),
    ];
    for (ki, kq) in KQUANTS.iter().enumerate() {
        let mut cases: Vec<(i64, i64, i64, &str)> = COMPACT_QMM.to_vec();
        if matches!(kq.mode, "q4k" | "iq4nl" | "iq2xxs") {
            cases.extend_from_slice(&FULL_QMM);
        }
        for (k, n, m, family) in cases {
            let w = weights(kq, &[n], k, 0x7a00 + ki as u32);
            let t = tiled(&w, kq);
            let x = activation(&[m, k], 0x7b00 + ki as u32 + m as u32, DType::BFloat16);
            let what = format!("{} K={k} N={n} M={m}", kq.mode);
            let want = if family == "qmm_t_nax" && !nax_available() {
                "qmm_t"
            } else {
                family
            };
            let custom_metallib = std::env::var_os("MLX_PAGED_ATTN_METALLIB").is_some();
            let expect_p128 =
                family == "qmm_t_nax" && m >= 65 && nax_available() && !custom_metallib;
            start_counting();
            let (_, _, ours) = read_output("tiled", qmm_tiled(&x, &t, kq, GPU));
            assert_eq!(
                family_count(&format!("{want}_t64")),
                1,
                "{what}: must take {want}_t64"
            );
            let used_p128 = family_count("qmm_t_nax_t64_p128") == 1;
            if expect_p128 {
                assert!(
                    used_p128,
                    "{what}: must take qmm_t_nax_t64_p128 on the built-in NAX metallib"
                );
            }
            assert_no_row_major_route(&what);
            stop_counting();
            start_counting();
            let (_, _, reference) =
                read_output("row-major", quantized_matmul(&x, &w, true, kq, GPU));
            assert_eq!(family_count(want), 1, "{what}: reference must take {want}");
            stop_counting();
            if used_p128 {
                // `kquant_prefill_nax.h` stages decoded weights in FP16 and
                // accumulates with NAX tensor ops, so compare within
                // `BF16_TILE_TOL` against row-major `qmm_t_nax`.
                let (worst, peak) = worst_abs(&ours, &reference);
                let rel = worst / peak;
                assert!(
                    rel <= BF16_TILE_TOL,
                    "{what}: tiled p128 rel {rel:e} > {BF16_TILE_TOL:e} off row-major {want}"
                );
            } else {
                assert_eq!(
                    ours, reference,
                    "{what}: tiled {want} differs from row-major"
                );
            }
        }
        println!(
            "  {:<6} qmm_t_nax / p128 / qmm_t_splitk tiled parity ok",
            kq.mode
        );
    }
}

/// (d) Route guard: a tiled tensor is refused wherever only row-major
/// kernels exist, at construction, never silently decoded.
#[test]
fn tiled_is_refused_off_the_ported_routes() {
    let kq = kquant("q4k");
    let (k, n) = (512i64, 128i64);
    let w = weights(kq, &[n], k, 0x7c00);
    let t = tiled(&w, kq);
    let mode = tiled_mode(kq);
    let x = activation(&[2, k], 0x7d00, DType::BFloat16);
    // transpose = false (x @ w) reads columns of the row-major stream.
    let xn = activation(&[2, n], 0x7d01, DType::BFloat16);
    assert!(
        qmm_mode(&xn, &t, false, kq, &mode, CPU).is_null(),
        "tiled with transpose=false must be rejected"
    );
    // Whole tiles only.
    let w_tail = weights(kq, &[n + 32], k, 0x7c01);
    assert!(
        qmm_mode(&x, &w_tail, true, kq, &mode, CPU).is_null(),
        "N % 64 != 0 must be rejected"
    );
    // Whole 256-value super-blocks only: IQ4_NL packs in 32s, so K = 544 is a
    // valid row-major tensor that cannot tile.
    let nl = kquant("iq4nl");
    let w_k = weights(nl, &[n], k + 32, 0x7c02);
    let x_k = activation(&[2, k + 32], 0x7d02, DType::BFloat16);
    assert!(
        qmm_mode(&x_k, &w_k, true, nl, &tiled_mode(nl), CPU).is_null(),
        "K % 256 != 0 must be rejected"
    );
    // 3-D (expert) weights stay row-major.
    let w3 = weights(kq, &[2, n], k, 0x7c03);
    let x3 = activation(&[2, 2, k], 0x7d03, DType::BFloat16);
    assert!(
        qmm_mode(&x3, &w3, true, kq, &mode, CPU).is_null(),
        "a 3-D tiled weight must be rejected"
    );
    // gather_qmm and dequantize have no tiled kernels.
    let ids = indices(&[0, 1]);
    // SAFETY: every handle outlives the call.
    let g = unsafe {
        mlx_sys::mlx_gather_qmm(
            x3.as_raw_ptr(),
            w3.w.as_raw_ptr(),
            w3.scales.as_raw_ptr(),
            w3.biases.as_raw_ptr(),
            std::ptr::null_mut(),
            ids.as_raw_ptr(),
            true,
            kq.group_size,
            kq.bits,
            mode.as_ptr(),
            false,
        )
    };
    assert!(g.is_null(), "tiled gather_qmm must be rejected");
    // SAFETY: every handle outlives the call.
    let d = unsafe {
        mlx_sys::mlx_dequantize(
            t.w.as_raw_ptr(),
            t.scales.as_raw_ptr(),
            t.biases.as_raw_ptr(),
            kq.group_size,
            kq.bits,
            DType::BFloat16 as i32,
            mode.as_ptr(),
        )
    };
    assert!(d.is_null(), "tiled dequantize must be rejected");
    // The production entry accepts the tagged mode for the ported route.
    // SAFETY: every handle outlives the call.
    let ok = unsafe {
        mlx_sys::mlx_quantized_matmul(
            x.as_raw_ptr(),
            t.w.as_raw_ptr(),
            t.scales.as_raw_ptr(),
            t.biases.as_raw_ptr(),
            true,
            kq.group_size,
            kq.bits,
            mode.as_ptr(),
        )
    };
    let (_, _, ours) = read_output("production tiled", ok);
    let (_, _, reference) = read_output("cpu", quantized_matmul(&x, &w, true, kq, CPU));
    let (worst, peak) = worst_abs(&ours, &reference);
    assert!(
        worst / peak < 1e-2,
        "production tiled route off the CPU by {worst} (peak {peak})"
    );
    println!("  tiled refused on transpose=false, partial tiles, 3-D, gather, dequantize");
}

/// Row-concatenate two weight sets (`[a; b]`).
fn concat_rows(a: &Weights, b: &Weights) -> Weights {
    let cat = |x: &MxArray, y: &MxArray| MxArray::concatenate(x, y, 0).expect("concat rows");
    Weights {
        w: cat(&a.w, &b.w),
        scales: cat(&a.scales, &b.scales),
        biases: cat(&a.biases, &b.biases),
    }
}

/// `w` with `rows` zero rows appended to every array: zero codes, zero
/// sub-scales, zero super-scales.
fn zero_pad_rows(w: &Weights, rows: i64) -> Weights {
    let pad = |a: &MxArray| {
        let cols = a.shape().expect("shape")[1];
        let zeros = MxArray::zeros(&[rows, cols], Some(a.dtype().expect("dtype"))).expect("zeros");
        MxArray::concatenate(a, &zeros, 0).expect("pad rows")
    };
    Weights {
        w: pad(&w.w),
        scales: pad(&w.scales),
        biases: pad(&w.biases),
    }
}

/// The GDN in_proj merge at its Qwen3.8 width: the 96-row `in_proj_ba` is
/// zero-padded to 128 rows so it tiles and row-merges with the tiled
/// 16384-row `in_proj_qkvz` (one `_t64` dispatch instead of two). In every
/// K-quant mode the padded rows must decode to exactly 0 (`d = 0`, `sc = 0`:
/// `d*sc*q - dmin*m`, `d*sc*(q-32)`, `d*sc*grid[q]` are all 0) and the real
/// columns must equal the unpadded row-major merge: bit-identical on the CPU
/// reference and on the shared prefill kernel (N this wide never splits K),
/// within the route tolerance where M = 1 / M = 8 change kernels with the
/// layout.
#[test]
fn padded_merge_is_invisible_in_every_mode() {
    let (k, n_ba, pad) = (512i64, 96i64, 32i64);
    let has_gpu = gpu_gen() > 0;
    for (ki, kq) in KQUANTS.iter().enumerate() {
        // Every mode tests:
        //   - compact N = 256+96+32 (384) at M = 1, 8 (CPU bitwise + GPU);
        //   - route-equivalent unsplit prefill N = 4096+96+32 (4224; 2*132 = 264 >= 257
        //     tiles so both tiled and row-major M = 64 take unsplit GEMM, never split-K);
        // plus three decode-class representatives (q4k, q6k, iq2xxs) also verify
        // the full Qwen3.8 width N = 16384+96+32 across M = 1, 8, 64.
        let cases: &[(&[i64], i64)] = if matches!(kq.mode, "q4k" | "q6k" | "iq2xxs") {
            if has_gpu {
                &[(&[1, 8], 256), (&[1, 8, 64], 16384)]
            } else {
                &[(&[1, 8], 256)]
            }
        } else if has_gpu {
            &[(&[1, 8], 256), (&[64], 4096)]
        } else {
            &[(&[1, 8], 256)]
        };
        for &(ms, n_qkvz) in cases {
            let n_real = n_qkvz + n_ba;
            let n_padded = n_real + pad;
            let qkvz = weights(kq, &[n_qkvz], k, 0x8300 + ki as u32);
            let ba = weights(kq, &[n_ba], k, 0x8400 + ki as u32);
            let reference = concat_rows(&qkvz, &ba);
            assert!(!kquant_tileable(n_real, k) && kquant_tileable(n_padded, k));
            let padded = tiled(&concat_rows(&qkvz, &zero_pad_rows(&ba, pad)), kq);
            for &m in ms {
                let x = activation(&[m, k], 0x8500 + ki as u32 + m as u32, DType::BFloat16);
                let what = format!("{} N={n_qkvz}+{n_ba}+{pad} M={m}", kq.mode);
                let check_cpu_padded = !has_gpu || (n_qkvz == 256 && m != 64);
                let cpu_ref = if check_cpu_padded || m != 64 {
                    Some(read_output(&what, quantized_matmul(&x, &reference, true, kq, CPU)).2)
                } else {
                    None
                };
                if check_cpu_padded {
                    // Verify CPU tiled == row-major bitwise and exact-zero padding
                    // on the compact padded fixture even on Metal CI hosts (where
                    // CPU-only tests never run separately).
                    let (shape, _, cpu_tiled) = read_output(&what, qmm_tiled(&x, &padded, kq, CPU));
                    assert_eq!(shape, vec![m, n_padded], "{what}: padded output width");
                    let cpu_ref = cpu_ref.as_ref().unwrap();
                    for row in 0..m as usize {
                        let ours =
                            &cpu_tiled[row * n_padded as usize..(row + 1) * n_padded as usize];
                        let theirs = &cpu_ref[row * n_real as usize..(row + 1) * n_real as usize];
                        assert_eq!(
                            &ours[..n_real as usize],
                            theirs,
                            "{what}: cpu real columns differ in row {row}"
                        );
                        assert!(
                            ours[n_real as usize..].iter().all(|&b| bf16(b) == 0.0),
                            "{what}: cpu padding columns are not zero in row {row}"
                        );
                    }
                }
                if !has_gpu {
                    continue;
                }
                // GPU: exactly one tiled dispatch, no row-major kernel.
                start_counting();
                let (shape, _, ours) = read_output(&what, qmm_tiled(&x, &padded, kq, GPU));
                assert_eq!(shape, vec![m, n_padded], "{what}: padded output width");
                let gemm = if nax_available() {
                    "qmm_t_nax_t64"
                } else {
                    "qmm_t_t64"
                };
                if m == 64 {
                    assert_eq!(
                        (family_count(gemm), family_count("qmm_t_splitk_t64")),
                        (1, 0),
                        "{what}: M=64 padded merge must take unsplit {gemm} without split-K"
                    );
                } else {
                    let family = tiled_family(m, n_padded, k, kq, gemm);
                    let took = if family == gemm && family_count("qmm_t_splitk_t64") == 1 {
                        "qmm_t_splitk_t64"
                    } else {
                        family
                    };
                    assert_eq!(family_count(took), 1, "{what}: must take {took}");
                }
                assert_no_row_major_route(&what);
                stop_counting();
                let mut real = Vec::with_capacity((m * n_real) as usize);
                for row in 0..m as usize {
                    let row_slice = &ours[row * n_padded as usize..(row + 1) * n_padded as usize];
                    assert!(
                        row_slice[n_real as usize..].iter().all(|&b| bf16(b) == 0.0),
                        "{what}: gpu padding columns are not zero in row {row}"
                    );
                    real.extend_from_slice(&row_slice[..n_real as usize]);
                }
                if m == 64 {
                    let ref_gemm = if nax_available() {
                        "qmm_t_nax"
                    } else {
                        "qmm_t"
                    };
                    start_counting();
                    let (_, _, theirs) =
                        read_output(&what, quantized_matmul(&x, &reference, true, kq, GPU));
                    assert_eq!(
                        (family_count(ref_gemm), family_count("qmm_t_splitk")),
                        (1, 0),
                        "{what}: M=64 row-major reference must take unsplit {ref_gemm} without split-K"
                    );
                    stop_counting();
                    assert_eq!(
                        real, theirs,
                        "{what}: padded tiled prefill differs from the row-major merge"
                    );
                } else {
                    let cpu_ref = cpu_ref.as_ref().unwrap();
                    let (worst, peak) = worst_abs(&real, cpu_ref);
                    let tol = if m == 8 { BF16_TILE_TOL } else { 1e-2 };
                    assert!(
                        worst / peak < tol,
                        "{what}: padded tiled route off the CPU reference by {worst} (peak {peak})"
                    );
                }
            }
            println!(
                "  {:<6} padded [{n_qkvz}; 96 -> 128] merge invisible",
                kq.mode
            );
        }
    }
}

/// (e) Repeated evaluations of the tiled M = 1 and M = 8 routes are
/// byte-identical.
///
/// Run-to-run variation can only come from scheduling. `qmv_t64` reduces its
/// k-splits inside the threadgroup in a fixed order, and `qmm_m8_nax_t64`'s
/// split-K epilogue (partials, atomic arrival counter, last arriver summing
/// in split order; kquant_m8_nax.h) is the same code for every mode — the
/// per-mode decode is straight-line. So this runs one mode per decode class
/// (plain q4k, codebook iq4nl, grid iq2xxs) rather than all fifteen; the
/// split shapes themselves are `kquant_m8_nax.rs::m8_nax_is_deterministic`'s.
#[cfg(target_os = "macos")]
#[test]
fn tiled_routes_are_deterministic() {
    assert!(gpu_gen() > 0, "no Metal device");
    const REPEATS: usize = 3;
    for (ki, kq) in KQUANTS
        .iter()
        .enumerate()
        .filter(|(_, kq)| matches!(kq.mode, "q4k" | "iq4nl" | "iq2xxs"))
    {
        let (k, n) = (17408i64, 5120i64);
        let w = weights(kq, &[n], k, 0x7e00 + ki as u32);
        let t = tiled(&w, kq);
        for m in [1i64, 8] {
            let x = activation(&[m, k], 0x7f00 + ki as u32 + m as u32, DType::BFloat16);
            let (_, _, first) = read_output("first", qmm_tiled(&x, &t, kq, GPU));
            for i in 1..REPEATS {
                let (_, _, again) = read_output("again", qmm_tiled(&x, &t, kq, GPU));
                assert!(again == first, "{} M={m}: evaluation {i} differs", kq.mode);
            }
        }
        println!(
            "  {:<6} M=1 and M=8 tiled: {REPEATS} evaluations identical",
            kq.mode
        );
    }
}

/// The layout is part of the primitive: a shapeless compile traced tiled
/// replays tiled, bit-identical to eager.
#[test]
fn tiled_shapeless_replay_matches_eager() {
    let kq = kquant("q4k");
    let (k, n) = (512i64, 128i64);
    let w = weights(kq, &[n], k, 0x8000);
    let t = tiled(&w, kq);
    let mode = tiled_mode(kq);
    let trace_x = activation(&[1, 3, k], 0x8100, DType::BFloat16);
    let x = activation(&[2, 5, k], 0x8200, DType::BFloat16);
    for device in [CPU, GPU] {
        if device == GPU && gpu_gen() <= 0 {
            continue;
        }
        let mut replayed = std::ptr::null_mut();
        let mut eager = std::ptr::null_mut();
        // SAFETY: every handle outlives the call; both outputs are owned on success.
        let ok = unsafe {
            mlx_sys::mlx_test_kquant_shapeless_replay(
                trace_x.as_raw_ptr(),
                x.as_raw_ptr(),
                t.w.as_raw_ptr(),
                t.scales.as_raw_ptr(),
                t.biases.as_raw_ptr(),
                true,
                kq.group_size,
                kq.bits,
                mode.as_ptr(),
                device,
                &mut replayed,
                &mut eager,
            )
        };
        assert!(ok, "tiled shapeless replay failed");
        assert_identical("tiled shapeless replay", replayed, eager);
        // And the eager tiled result equals the row-major one on the CPU.
        if device == CPU {
            let reference = quantized_matmul(&x, &w, true, kq, CPU);
            assert_identical(
                "tiled vs row-major (cpu, batched x)",
                qmm_tiled(&x, &t, kq, CPU),
                reference,
            );
        }
    }
}
