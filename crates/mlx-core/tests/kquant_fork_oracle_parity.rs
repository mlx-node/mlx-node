//! The bridge's K-quant ops (`mlx_kquant*.cpp`) against MLX's own K-quant path,
//! bit for bit, on Metal and on the CPU.
//!
//! The oracle is reachable only while `crates/mlx-sys/mlx` is the MLX fork
//! pinned at 053e43fec; `mlx_test_fork_kquant_*` call `mlx::core` with the
//! mode string. When the pin moves to upstream MLX this file becomes a
//! reference-dequantize test.
//!
//! Both sides take an explicit device, so the tests are race-free under any
//! `--test-threads`. The Metal tests count the kernel families the bridge
//! dispatched on their own thread and require every family the device's
//! generation can reach. Run all three generations:
//!
//!   cargo test -p mlx-core --release --test kquant_fork_oracle_parity
//!   MLX_METAL_GPU_ARCH=applegpu_g16s cargo test -p mlx-core --release --test kquant_fork_oracle_parity
//!   MLX_METAL_GPU_ARCH=applegpu_g14s cargo test -p mlx-core --release --test kquant_fork_oracle_parity
//!
//! The native run (gen 17 with NAX) reaches qmm_t_nax and qmv_sg8; g16s turns
//! both off; g14s also turns qmv_wide off, so multi-row matvecs take qmv.

mod golden_support;
mod kquant_support;

use golden_support::{Golden, Mode};
use kquant_support::*;
use mlx_core::array::{DType, MxArray};

/// Row counts that cross every dispatch boundary: qmv, qmv_wide nv2..8, the
/// bfloat16 M = 8 sg8 kernel, the batch limit, split-K qmm, and NAX qmm.
const MULTI_ROW: [i64; 7] = [2, 3, 4, 5, 6, 7, 8];
const OTHER_ROWS: [i64; 8] = [1, 9, 12, 17, 24, 33, 64, 512];

fn label(device: i32) -> &'static str {
    if device == GPU { "gpu" } else { "cpu" }
}

fn fork_quantized_matmul(
    x: &MxArray,
    w: &Weights,
    transpose: bool,
    kq: &KQuant,
    device: i32,
) -> *mut mlx_sys::mlx_array {
    let mode = mode_cstr(kq);
    // SAFETY: every handle outlives the call.
    unsafe {
        mlx_sys::mlx_test_fork_kquant_quantized_matmul(
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

#[allow(clippy::too_many_arguments)]
fn fork_gather_qmm(
    x: &MxArray,
    w: &Weights,
    lhs: Option<&MxArray>,
    rhs: &MxArray,
    transpose: bool,
    sorted: bool,
    kq: &KQuant,
    device: i32,
) -> *mut mlx_sys::mlx_array {
    let mode = mode_cstr(kq);
    // SAFETY: every handle outlives the call; null lhs means "derive from x".
    unsafe {
        mlx_sys::mlx_test_fork_kquant_gather_qmm(
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            ptr(lhs),
            rhs.as_raw_ptr(),
            transpose,
            kq.group_size,
            kq.bits,
            mode.as_ptr(),
            sorted,
            device,
        )
    }
}

fn fork_dequantize(w: &Weights, dtype: DType, kq: &KQuant, device: i32) -> *mut mlx_sys::mlx_array {
    let mode = mode_cstr(kq);
    // SAFETY: every handle outlives the call.
    unsafe {
        mlx_sys::mlx_test_fork_kquant_dequantize(
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            kq.group_size,
            kq.bits,
            dtype as i32,
            mode.as_ptr(),
            device,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn qmm_pair(
    g: &mut Golden,
    cases: &mut usize,
    what: &str,
    x: &MxArray,
    w: &Weights,
    transpose: bool,
    kq: &KQuant,
    device: i32,
) {
    let (shape, dtype, bits) = assert_identical(
        what,
        quantized_matmul(x, w, transpose, kq, device),
        fork_quantized_matmul(x, w, transpose, kq, device),
    );
    g.record_bits(what, &shape, dtype, &bits);
    *cases += 1;
}

#[allow(clippy::too_many_arguments)]
fn gather_pair(
    g: &mut Golden,
    cases: &mut usize,
    what: &str,
    x: &MxArray,
    w: &Weights,
    lhs: Option<&MxArray>,
    rhs: &MxArray,
    transpose: bool,
    sorted: bool,
    kq: &KQuant,
    device: i32,
) {
    let (shape, dtype, bits) = assert_identical(
        what,
        gather_qmm(x, w, lhs, rhs, transpose, sorted, kq, device),
        fork_gather_qmm(x, w, lhs, rhs, transpose, sorted, kq, device),
    );
    g.record_bits(what, &shape, dtype, &bits);
    *cases += 1;
}

fn run_matmuls(g: &mut Golden, cases: &mut usize, device: i32, ms: &[i64]) {
    let dev = label(device);
    for (ki, kq) in KQUANTS.iter().enumerate() {
        let seed = 0x1000 + ki as u32;
        // Transposed: (N, K) = aligned and fast-qmv, then unaligned N with
        // K % 512 != 0, then batched weights.
        let wide = weights(kq, &[2048], 1024, seed);
        let odd = weights(kq, &[136], 768, seed + 1);
        let batched = weights(kq, &[2, 136], 768, seed + 2);
        let batched_wide = weights(kq, &[2, 256], 1024, seed + 5);
        // Untransposed: the packed axis is the output; K < 1024 is qvm,
        // K >= 1024 is qvm_split_k, M >= 4 is qmm_n.
        let n_short = weights(kq, &[256], 512, seed + 3);
        let n_deep = weights(kq, &[1024], 512, seed + 4);
        for dtype in DTYPES {
            let m_ = kq.mode;
            for &m in ms {
                if device == GPU || m <= 33 {
                    let x = activation(&[m, 1024], seed + m as u32, dtype);
                    qmm_pair(
                        g,
                        cases,
                        &format!("{dev} {m_} {dtype:?} t M={m} N=2048 K=1024"),
                        &x,
                        &wide,
                        true,
                        kq,
                        device,
                    );
                }
                let x = activation(&[m, 768], seed + 7 + m as u32, dtype);
                qmm_pair(
                    g,
                    cases,
                    &format!("{dev} {m_} {dtype:?} t M={m} N=136 K=768"),
                    &x,
                    &odd,
                    true,
                    kq,
                    device,
                );
                if m > 64 {
                    continue;
                }
                let x = activation(&[2, m, 768], seed + 11 + m as u32, dtype);
                qmm_pair(
                    g,
                    cases,
                    &format!("{dev} {m_} {dtype:?} t batched M={m}"),
                    &x,
                    &batched,
                    true,
                    kq,
                    device,
                );
                let x = activation(&[2, m, 1024], seed + 23 + m as u32, dtype);
                qmm_pair(
                    g,
                    cases,
                    &format!("{dev} {m_} {dtype:?} t batched-aligned M={m}"),
                    &x,
                    &batched_wide,
                    true,
                    kq,
                    device,
                );
                let x = activation(&[3, m, 768], seed + 13 + m as u32, dtype);
                qmm_pair(
                    g,
                    cases,
                    &format!("{dev} {m_} {dtype:?} t x-batched M={m}"),
                    &x,
                    &odd,
                    true,
                    kq,
                    device,
                );
                let x = activation(&[m, 256], seed + 17 + m as u32, dtype);
                qmm_pair(
                    g,
                    cases,
                    &format!("{dev} {m_} {dtype:?} n M={m} K=256"),
                    &x,
                    &n_short,
                    false,
                    kq,
                    device,
                );
                let x = activation(&[m, 1024], seed + 19 + m as u32, dtype);
                qmm_pair(
                    g,
                    cases,
                    &format!("{dev} {m_} {dtype:?} n M={m} K=1024"),
                    &x,
                    &n_deep,
                    false,
                    kq,
                    device,
                );
            }
        }
    }
}

/// K = 544 is the one transposed K-quant reduction that is not a multiple of
/// 64 (IQ4_NL's 32-value blocks), so on a NAX device it is the only shape that
/// keeps qmm on the simdgroup `qmm_t`. The batched weight makes B = 2, which
/// skips split-K.
fn run_simdgroup_qmm_t(g: &mut Golden, cases: &mut usize, device: i32) {
    let kq = kquant("iq4nl");
    let w = weights(kq, &[2, 136], 544, 0x1500);
    for dtype in DTYPES {
        let x = activation(&[2, 64, 544], 0x1501, dtype);
        qmm_pair(
            g,
            cases,
            &format!("{} iq4nl {dtype:?} t batched M=64 K=544", label(device)),
            &x,
            &w,
            true,
            kq,
            device,
        );
    }
}

fn run_gathers(g: &mut Golden, cases: &mut usize, device: i32) {
    let dev = label(device);
    const E: i64 = 8;
    for (ki, kq) in KQUANTS.iter().enumerate() {
        let seed = 0x2000 + ki as u32;
        let m_ = kq.mode;
        let experts_t = weights(kq, &[E, 128], 512, seed);
        let experts_n = weights(kq, &[E, 256], 512, seed + 1);
        let experts_odd = weights(kq, &[E, 136], 768, seed + 2);
        let random = random_ids(seed, 32, E as u32);
        let mut sorted_ids = random.clone();
        sorted_ids.sort_unstable();
        let lhs = random_ids(seed + 9, 6, 4);
        let rhs6 = random_ids(seed + 10, 6, E as u32);
        for dtype in DTYPES {
            // One row per token, experts on the right only: gather_qmv, and
            // with sorted indices the gather_qmm_rhs walk.
            let x = activation(&[32, 1, 512], seed + 3, dtype);
            for (sorted, ids, tag) in [
                (false, &random, "unsorted"),
                (true, &sorted_ids, "sorted"),
                (false, &sorted_ids, "sorted-ids unsorted-flag"),
            ] {
                gather_pair(
                    g,
                    cases,
                    &format!("{dev} {m_} {dtype:?} gather t {tag}"),
                    &x,
                    &experts_t,
                    None,
                    &indices(ids),
                    true,
                    sorted,
                    kq,
                    device,
                );
            }
            let x = activation(&[32, 1, 768], seed + 4, dtype);
            gather_pair(
                g,
                cases,
                &format!("{dev} {m_} {dtype:?} gather t odd unsorted"),
                &x,
                &experts_odd,
                None,
                &indices(&random),
                true,
                false,
                kq,
                device,
            );
            gather_pair(
                g,
                cases,
                &format!("{dev} {m_} {dtype:?} gather t odd sorted"),
                &x,
                &experts_odd,
                None,
                &indices(&sorted_ids),
                true,
                true,
                kq,
                device,
            );
            let xn = activation(&[32, 1, 256], seed + 5, dtype);
            gather_pair(
                g,
                cases,
                &format!("{dev} {m_} {dtype:?} gather n sorted"),
                &xn,
                &experts_n,
                None,
                &indices(&sorted_ids),
                false,
                true,
                kq,
                device,
            );
            gather_pair(
                g,
                cases,
                &format!("{dev} {m_} {dtype:?} gather n unsorted"),
                &xn,
                &experts_n,
                None,
                &indices(&random),
                false,
                false,
                kq,
                device,
            );
            // Both index sets: vector and matrix kernels on either layout.
            for m in [1i64, 4, 33] {
                let x = activation(&[4, m, 512], seed + 7 + m as u32, dtype);
                gather_pair(
                    g,
                    cases,
                    &format!("{dev} {m_} {dtype:?} gather t lhs M={m}"),
                    &x,
                    &experts_t,
                    Some(&indices(&lhs)),
                    &indices(&rhs6),
                    true,
                    false,
                    kq,
                    device,
                );
                let x = activation(&[4, m, 256], seed + 9 + m as u32, dtype);
                gather_pair(
                    g,
                    cases,
                    &format!("{dev} {m_} {dtype:?} gather n lhs M={m}"),
                    &x,
                    &experts_n,
                    Some(&indices(&lhs)),
                    &indices(&rhs6),
                    false,
                    false,
                    kq,
                    device,
                );
            }
        }
    }
}

fn run_dequantize(g: &mut Golden, cases: &mut usize, device: i32) {
    let dev = label(device);
    for (ki, kq) in KQUANTS.iter().enumerate() {
        let flat = weights(kq, &[96], 1024, 0x3000 + ki as u32);
        let stacked = weights(kq, &[3, 40], 512, 0x3100 + ki as u32);
        for dtype in DTYPES {
            for (w, tag) in [(&flat, "2-D"), (&stacked, "3-D")] {
                let what = format!("{dev} {} {dtype:?} dequantize {tag}", kq.mode);
                let (shape, out_dtype, bits) = assert_identical(
                    &what,
                    dequantize(w, dtype, kq, device),
                    fork_dequantize(w, dtype, kq, device),
                );
                g.record_bits(&what, &shape, out_dtype, &bits);
                *cases += 1;
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn require_metal() -> i32 {
    let arch = gpu_gen();
    assert!(arch > 0, "no Metal device: these macOS tests need one");
    arch
}

fn expect_hit(families: &[&str]) {
    for family in families {
        let n = family_count(family);
        println!("  {family:<20} {n}");
        assert!(n > 0, "kernel family {family} never ran on this thread");
    }
}

fn expect_absent(families: &[&str]) {
    for family in families {
        assert_eq!(
            family_count(family),
            0,
            "kernel family {family} must not run here"
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_matmul_matches_fork_bitwise() {
    let arch = require_metal();
    let nax = nax_available();
    println!("gpu gen {arch}, nax {nax}");
    let mut g = Golden::metal("kquant_metal_matmul", Mode::from_env());
    let mut cases = 0;

    // Multi-row matvecs alone first, to see which kernel they take.
    start_counting();
    run_matmuls(&mut g, &mut cases, GPU, &MULTI_ROW);
    let multi_row_qmv = family_count("qmv") + family_count("qmv_fast");
    let multi_row_wide = family_count("qmv_wide");
    let multi_row_sg8 = family_count("qmv_sg8");
    println!("multi-row: qmv {multi_row_qmv}, qmv_wide {multi_row_wide}, qmv_sg8 {multi_row_sg8}");
    if arch >= 15 {
        assert_eq!(
            multi_row_qmv, 0,
            "gen {arch}: multi-row matvecs must take qmv_wide"
        );
        assert!(multi_row_wide > 0);
    } else {
        assert!(
            multi_row_qmv > 0,
            "gen {arch}: multi-row matvecs must take qmv"
        );
        assert_eq!(multi_row_wide, 0);
    }
    if arch >= 17 {
        assert!(
            multi_row_sg8 > 0,
            "gen {arch}: bfloat16 M = 8 must take qmv_sg8"
        );
    }

    start_counting();
    run_matmuls(&mut g, &mut cases, GPU, &MULTI_ROW);
    run_matmuls(&mut g, &mut cases, GPU, &OTHER_ROWS);
    run_simdgroup_qmm_t(&mut g, &mut cases, GPU);
    println!("metal quantized_matmul: {cases} cases bit-identical");
    expect_hit(&[
        "qmv",
        "qmv_fast",
        "qmm_t",
        "qmm_t_splitk",
        "qmm_n",
        "qvm",
        "qvm_split_k",
    ]);
    if arch >= 15 {
        expect_hit(&[
            "qmv_wide",
            "qmv_wide_nv2",
            "qmv_wide_nv3",
            "qmv_wide_nv4",
            "qmv_wide_nv5",
            "qmv_wide_nv6",
            "qmv_wide_nv7",
            "qmv_wide_nv8",
        ]);
    } else {
        expect_absent(&["qmv_wide"]);
    }
    if arch >= 17 {
        expect_hit(&["qmv_sg8"]);
    } else {
        expect_absent(&["qmv_sg8"]);
    }
    if nax {
        expect_hit(&["qmm_t_nax"]);
    } else {
        expect_absent(&["qmm_t_nax"]);
    }
    stop_counting();
    g.finish();
}

#[cfg(target_os = "macos")]
#[test]
fn metal_gather_and_dequantize_match_fork_bitwise() {
    require_metal();
    let mut g = Golden::metal("kquant_metal_gather_dequantize", Mode::from_env());
    start_counting();
    let mut gather = 0;
    run_gathers(&mut g, &mut gather, GPU);
    println!("metal gather_qmm: {gather} cases bit-identical");
    let mut deq = 0;
    run_dequantize(&mut g, &mut deq, GPU);
    println!("metal dequantize: {deq} cases bit-identical");
    expect_hit(&[
        "gather_qmv",
        "gather_qmv_fast",
        "gather_qmm_rhs_nt",
        "gather_qmm_rhs_nn",
        "gather_qvm",
        "gather_qmm_t",
        "gather_qmm_n",
        "dequantize",
    ]);
    stop_counting();
    g.finish();
}

#[test]
fn cpu_matches_fork_bitwise() {
    let mut g = Golden::cpu("kquant_cpu", Mode::from_env());
    start_counting();
    let mut mm = 0;
    run_matmuls(&mut g, &mut mm, CPU, &[1, 2, 8, 9, 33]);
    run_simdgroup_qmm_t(&mut g, &mut mm, CPU);
    println!("cpu quantized_matmul: {mm} cases bit-identical");
    let mut gather = 0;
    run_gathers(&mut g, &mut gather, CPU);
    println!("cpu gather_qmm: {gather} cases bit-identical");
    let mut deq = 0;
    run_dequantize(&mut g, &mut deq, CPU);
    println!("cpu dequantize: {deq} cases bit-identical");
    // Nothing on this thread may have been encoded for Metal.
    expect_absent(&[
        "qmv",
        "qmv_fast",
        "qmm_t",
        "qmm_n",
        "gather_qmv",
        "dequantize",
    ]);
    stop_counting();
    g.finish();
}
