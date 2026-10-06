//! The Tiled64 K-quant weight layout (`mlx_kquant.h`, mode suffix `@t64`):
//! the row-major `.weight` / `.scales` / `.biases` bytes permuted so the 64
//! rows of a tile interleave per 32-code unit. A pure permutation, so every
//! ported kernel (`qmv_wide_t64`, `qmm_m8_nax_t64`, `qmm_t_nax_t64`,
//! `qmm_t_splitk_t64`, the CPU reference) must give the same bits as its
//! row-major twin, and no tiled tensor may reach an unported kernel.
//!
//!   cargo test -p mlx-core --release --test kquant_tiled -- --nocapture

mod kquant_support;

use std::ffi::CString;

use kquant_support::*;
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

fn per_group(kq: &KQuant) -> i64 {
    if kq.scales_signed { 1 } else { 2 }
}

/// Groups per super-block (IQ4_NL: one 32-value block).
fn super_ratio(kq: &KQuant) -> i64 {
    match kq.mode {
        "q6k" | "q3k" | "q2k" => 16,
        "iq4nl" => 1,
        _ => 8,
    }
}

/// The same bytes in the Tiled64 order: codes per unit, companions per
/// super-block.
fn tiled(w: &Weights, kq: &KQuant) -> Weights {
    let pg = per_group(kq);
    let t = Weights {
        w: kquant_tile_rows(&w.w, i64::from(kq.bits)).expect("tile weight"),
        scales: kquant_tile_rows(&w.scales, super_ratio(kq) * pg).expect("tile scales"),
        biases: kquant_tile_rows(&w.biases, pg).expect("tile biases"),
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

/// max |a - b| over bfloat16 bit patterns, and the peak |b|.
fn worst_abs(ours: &[u32], reference: &[u32]) -> (f32, f32) {
    assert_eq!(ours.len(), reference.len(), "output lengths differ");
    let worst = ours
        .iter()
        .zip(reference)
        .map(|(&o, &r)| (bf16(o) - bf16(r)).abs())
        .fold(0f32, f32::max);
    let peak = reference.iter().fold(0f32, |m, &r| m.max(bf16(r).abs()));
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
        let pg = per_group(kq);
        let back = [
            kquant_untile_rows(&t.w, i64::from(kq.bits)).unwrap(),
            kquant_untile_rows(&t.scales, super_ratio(kq) * pg).unwrap(),
            kquant_untile_rows(&t.biases, pg).unwrap(),
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
        // Row 65, unit 3, word 1 -> tile 1, unit 3, row-in-tile 1.
        let (row, unit, word) = (65i64, 3i64, 1i64);
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

/// (a) M = 2..7 and 9 take qmv_wide in both layouts, same lane map and sum
/// order: bit-identical on the Qwen3.8 shapes.
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
/// route and is checked against the CPU reference.
#[cfg(target_os = "macos")]
#[test]
fn qmv_m1_tiled_matches_cpu() {
    assert!(gpu_gen() > 0, "no Metal device");
    for (ki, kq) in KQUANTS.iter().enumerate() {
        for (si, &(k, n)) in SHAPES.iter().enumerate().take(4) {
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
/// tolerance of the CPU reference. q3k, q2k and iq4nl also take it row-major,
/// and there the two layouts are bit-identical (same split count, same order).
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

            start_counting();
            let (_, _, ours) = read_output("tiled", qmm_tiled(&x, &t, kq, GPU));
            assert_eq!(
                family_count("qmm_m8_nax_t64"),
                1,
                "{what}: must take qmm_m8_nax_t64"
            );
            assert_no_row_major_route(&what);
            let splits = [1u64, 2, 4, 8]
                .into_iter()
                .find(|s| family_count(&format!("qmm_m8_nax_splits{s}")) > 0)
                .unwrap_or(0);
            stop_counting();

            if matches!(kq.mode, "q3k" | "q2k" | "iq4nl") {
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

/// (a) Prefill: M = 64 and 2048 take qmm_t_nax in both layouts, M = 16 on
/// N = 5120 takes qmm_t_splitk; all bit-identical.
#[cfg(target_os = "macos")]
#[test]
fn qmm_tiled_is_bit_identical() {
    assert!(gpu_gen() > 0, "no Metal device");
    for (ki, kq) in KQUANTS.iter().enumerate() {
        for (k, n, m, family) in [
            (5120i64, 5120i64, 64i64, "qmm_t_nax"),
            (5120, 17408, 64, "qmm_t_nax"),
            (5120, 6144, 2048, "qmm_t_nax"),
            (5120, 5120, 16, "qmm_t_splitk"),
        ] {
            let w = weights(kq, &[n], k, 0x7a00 + ki as u32);
            let t = tiled(&w, kq);
            let x = activation(&[m, k], 0x7b00 + ki as u32 + m as u32, DType::BFloat16);
            let what = format!("{} K={k} N={n} M={m}", kq.mode);
            let want = if family == "qmm_t_nax" && !nax_available() {
                "qmm_t"
            } else {
                family
            };
            start_counting();
            let (_, _, ours) = read_output("tiled", qmm_tiled(&x, &t, kq, GPU));
            assert_eq!(
                family_count(&format!("{want}_t64")),
                1,
                "{what}: must take {want}_t64"
            );
            assert_no_row_major_route(&what);
            stop_counting();
            start_counting();
            let (_, _, reference) =
                read_output("row-major", quantized_matmul(&x, &w, true, kq, GPU));
            assert_eq!(family_count(want), 1, "{what}: reference must take {want}");
            stop_counting();
            assert_eq!(
                ours, reference,
                "{what}: tiled {want} differs from row-major"
            );
        }
        println!(
            "  {:<6} qmm_t_nax / qmm_t_splitk tiled == row-major",
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
    let (k, n_qkvz, n_ba, pad) = (512i64, 16384i64, 96i64, 32i64);
    let n_real = n_qkvz + n_ba;
    let n_padded = n_real + pad;
    for (ki, kq) in KQUANTS.iter().enumerate() {
        let qkvz = weights(kq, &[n_qkvz], k, 0x8300 + ki as u32);
        let ba = weights(kq, &[n_ba], k, 0x8400 + ki as u32);
        let reference = concat_rows(&qkvz, &ba);
        assert!(!kquant_tileable(n_real, k) && kquant_tileable(n_padded, k));
        let padded = tiled(&concat_rows(&qkvz, &zero_pad_rows(&ba, pad)), kq);
        for m in [1i64, 8, 64] {
            let x = activation(&[m, k], 0x8500 + ki as u32 + m as u32, DType::BFloat16);
            let what = format!("{} M={m}", kq.mode);
            // CPU: the tiled reference reads the same arithmetic order, so the
            // real columns are bit-identical and the padding is zero.
            let (shape, _, cpu_tiled) = read_output(&what, qmm_tiled(&x, &padded, kq, CPU));
            assert_eq!(shape, vec![m, n_padded], "{what}: padded output width");
            let (_, _, cpu_ref) =
                read_output(&what, quantized_matmul(&x, &reference, true, kq, CPU));
            for row in 0..m as usize {
                let ours = &cpu_tiled[row * n_padded as usize..(row + 1) * n_padded as usize];
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
            if gpu_gen() <= 0 {
                continue;
            }
            // GPU: exactly one tiled dispatch, no row-major kernel.
            start_counting();
            let (_, _, ours) = read_output(&what, qmm_tiled(&x, &padded, kq, GPU));
            let family = match m {
                1 => "qmv_t64",
                8 if nax_available() => "qmm_m8_nax_t64",
                8 => "qmv_wide_t64",
                _ if nax_available() => "qmm_t_nax_t64",
                _ => "qmm_t_t64",
            };
            assert_eq!(family_count(family), 1, "{what}: must take {family}");
            assert_no_row_major_route(&what);
            stop_counting();
            let (_, _, theirs) =
                read_output(&what, quantized_matmul(&x, &reference, true, kq, GPU));
            let mut real = Vec::with_capacity((m * n_real) as usize);
            for row in 0..m as usize {
                let ours = &ours[row * n_padded as usize..(row + 1) * n_padded as usize];
                assert!(
                    ours[n_real as usize..].iter().all(|&b| bf16(b) == 0.0),
                    "{what}: gpu padding columns are not zero in row {row}"
                );
                real.extend_from_slice(&ours[..n_real as usize]);
            }
            if m == 64 {
                assert_eq!(
                    real, theirs,
                    "{what}: padded tiled prefill differs from the row-major merge"
                );
            } else {
                let (worst, peak) = worst_abs(&real, &cpu_ref);
                let tol = if m == 8 { BF16_TILE_TOL } else { 1e-2 };
                assert!(
                    worst / peak < tol,
                    "{what}: padded tiled route off the CPU reference by {worst} (peak {peak})"
                );
            }
        }
        println!("  {:<6} padded [16384; 96 -> 128] merge invisible", kq.mode);
    }
}

/// (e) Twenty evaluations of the tiled M = 1 and M = 8 routes are
/// byte-identical.
#[cfg(target_os = "macos")]
#[test]
fn tiled_routes_are_deterministic() {
    assert!(gpu_gen() > 0, "no Metal device");
    for (ki, kq) in KQUANTS.iter().enumerate() {
        let (k, n) = (17408i64, 5120i64);
        let w = weights(kq, &[n], k, 0x7e00 + ki as u32);
        let t = tiled(&w, kq);
        for m in [1i64, 8] {
            let x = activation(&[m, k], 0x7f00 + ki as u32 + m as u32, DType::BFloat16);
            let (_, _, first) = read_output("first", qmm_tiled(&x, &t, kq, GPU));
            for i in 1..20 {
                let (_, _, again) = read_output("again", qmm_tiled(&x, &t, kq, GPU));
                assert!(again == first, "{} M={m}: evaluation {i} differs", kq.mode);
            }
        }
        println!(
            "  {:<6} M=1 and M=8 tiled: 20 evaluations identical",
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
