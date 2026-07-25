//! Prefill-shaped `quantized_matmul` benchmark, and the gate that keeps the
//! K-quants on the NAX tensor-op kernels.
//!
//! A K-quant `qmm_t` reaches `kquant_qmm_t_nax` only if `nax_supports_mode`
//! (`mlx/backend/metal/quantized.cpp:66`) admits it. With that check closed on
//! purpose, this shape falls back to the simdgroup `qmm_t` and runs at roughly
//! 12.2 to 12.4 TFLOP/s against affine's 41.7 to 42.4 — a ratio of 3.40 to 3.49
//! across the three K-quants, over two sweeps, while the A/A control in the
//! same sweeps held 0.98 to 0.99. With the kernels reachable it is at parity.
//! Figures are stated as ranges because a single sweep does not reproduce to
//! the precision its own error bars suggest; the ceiling below is set against
//! the range, not against any one run. The assertion at the end is a ratio, not
//! an absolute time, so a loaded or throttled machine moves every config
//! together and does not turn it red.
//!
//! Method, so the number means something:
//!   * one sweep holds every config, so they all see the same thermal state;
//!   * two *identical* affine configs are included as an A/A control -- the
//!     ratio between them is the measurement floor, measured not assumed;
//!   * rounds are start-rotated, so per-config position in the round cancels;
//!   * every handle is constructed before the clock starts, so only `eval` --
//!     i.e. the kernel -- is timed;
//!   * the per-config statistic is a trimmed mean over rounds, and the spread
//!     is reported.
//!
//! Run with:
//!   cargo test -p mlx-core --release --test kquant_nax_bench -- --nocapture
//!
//! On a host without the NAX kernels this declines instead of measuring, and
//! libtest hides the reason unless `--nocapture` is passed — so in a plain
//! `cargo test` log the line is just `kquant_vs_affine_prefill ... ok`, which
//! is what CI's macos-26 runners produce today. `KQ_NAX_REQUIRE=1` turns that
//! into a failure; `kquant_mode_guards.rs` reads the same variable, so one
//! setting pins both gates on a box that is supposed to have the kernels.

use std::ffi::CString;
use std::time::Instant;

use mlx_core::array::MxArray;

const M: i64 = 512;
const N: i64 = 8192;
const K: i64 = 8192;

/// Ceiling on `kquant / affine`. What the test checks, in order: the GPU has
/// NAX kernels at all; the shape's split-K works out to 1 so `quantized_matmul`
/// reaches the `qmm` function holding the NAX branch; and then this ratio.
///
/// Over three interleaved sweeps the K-quant ratios spanned 0.95 to 1.09 —
/// worst 1.0878 +/- 0.0272 (q6k) — while the A/A control spanned 0.98 to 1.00.
/// A range rather than one sweep on purpose: run-to-run drift here is a few
/// percent, so a single quoted figure goes stale against the test's own output.
/// 2.0 is 1.8x above the worst of those and 1.7x below the 3.3992 the simdgroup
/// fallback measured with `nax_supports_mode` closed, so it sits between the two
/// regimes with room on both sides.
///
/// It does not police small kernel drift, and it is not by itself proof that
/// the K-quants executed on NAX — a fallback that happened to be fast would
/// pass. `gpu_matches_cpu_on_every_quantized_matmul_kernel` in
/// `kquant_mode_guards.rs` carries that proof, by requiring a tf32-sized gap
/// against the CPU on the float32 `qmm_t` shapes.
const KQUANT_VS_AFFINE_CEILING: f64 = 2.0;

/// Evals per timed measurement.
const REPS: usize = 12;
/// Sweeps. Each rotates which config goes first.
const ROUNDS: usize = 9;
/// Untimed evals before the first round.
const WARMUP: usize = 6;

/// Whether this host can run the NAX tensor-op kernels at all.
///
/// `metal::is_nax_available()` (`mlx/backend/metal/device.cpp:944-962`) is the
/// same cached predicate the dispatcher's own gate reads first
/// (`metal/quantized.cpp:968`), and it folds all three ways a host can lack the
/// kernels into one bool. `mlx_gpu_architecture_gen() >= 17` is not a
/// substitute: that is a separate parse in `mlx_gated_delta.cpp:407-428` which
/// knows nothing about macOS 26.2, the `'p' -> 18` rule, or `MLX_METAL_NO_NAX`.
///
/// `MLX_METAL_GPU_ARCH` feeds the generation this ultimately reads
/// (`metal/device.cpp:589-601`), so an override left in the environment turns
/// the bench off exactly the way older hardware would.
fn nax_available() -> bool {
    // SAFETY: a nullary FFI predicate that catches internally and reports false
    // when the Metal device is unavailable (`mlx-sys/src/mlx_stream.cpp:140`).
    unsafe { mlx_sys::mlx_metal_is_nax_available() }
}

/// MLX's split-K choice for this shape, recomputed from `qmm_splitk`
/// (`metal/quantized.cpp:1071-1092`). `QuantizedMatmul::eval_gpu` sends a
/// transposed B == 1 matmul to `qmm_splitk` (:1749), which only falls through to
/// `qmm` — the function holding the NAX branch — when this comes out as 1.
/// `k_align` is `max(group_size, 32)` and every config below has a group size of
/// 16 or 32, so it is 32 for all of them.
fn split_k_for_bench_shape() -> i64 {
    let n_tiles = (N + 31) / 32;
    let m_tiles = (M + 31) / 32;
    let split_k = std::cmp::max(1, 512 / (n_tiles * m_tiles));
    std::cmp::min(split_k, K / 32)
}

fn select_gpu() -> bool {
    // SAFETY: plain global setters in the MLX FFI; both catch internally.
    unsafe {
        mlx_sys::mlx_set_default_device(1);
        if mlx_sys::mlx_default_device() != 1 {
            return false;
        }
        let stream = mlx_sys::mlx_default_stream(1);
        mlx_sys::mlx_set_default_stream(stream);
    }
    true
}

fn eval(handle: *mut mlx_sys::mlx_array) {
    let mut buf = [0i8; 512];
    let mut h = handle;
    // SAFETY: `handle` is a live array; `buf` is a 512-byte sink for the error.
    let ok = unsafe { mlx_sys::mlx_eval_with_error(&mut h, 1, buf.as_mut_ptr(), buf.len()) };
    assert!(ok, "eval failed");
}

fn drop_array(handle: *mut mlx_sys::mlx_array) {
    if !handle.is_null() {
        // SAFETY: the handle is owned here and not referenced afterwards.
        unsafe { mlx_sys::mlx_array_delete(handle) };
    }
}

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn activation(shape: &[i64], seed: u32) -> MxArray {
    let n: i64 = shape.iter().product();
    let mut st = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let bits: Vec<u16> = (0..n)
        .map(|_| {
            let v = f32::from((lcg(&mut st) >> 24) as i8) / 128.0;
            (v.to_bits() >> 16) as u16
        })
        .collect();
    MxArray::from_bfloat16(&bits, shape).expect("activation")
}

const HALF_SCALES: [u16; 6] = [0x3000, 0x2C00, 0x3400, 0x2E66, 0x3155, 0x2800];
/// bfloat16 patterns in the same magnitude band as `HALF_SCALES`.
const BF16_SCALES: [u16; 6] = [0x3C00, 0x3B80, 0x3C80, 0x3BCD, 0x3C2A, 0x3B00];

type Weights = (MxArray, MxArray, MxArray);

struct Config {
    label: &'static str,
    mode: &'static str,
    bits: i32,
    group_size: i32,
    weights: Weights,
}

/// Both K-quant scale levels populated with non-degenerate values, matching
/// `filled_kquant_weights` in `kquant_mode_guards.rs`.
fn kquant_weights(
    seed: u32,
    scales_signed: bool,
    weight_cols_per_super: i64,
    scales_cols_per_super: i64,
    biases_cols_per_super: i64,
) -> Weights {
    let supers = K / 256;
    let (wc, sc, bc) = (
        weight_cols_per_super * supers,
        scales_cols_per_super * supers,
        biases_cols_per_super * supers,
    );
    let mut st = seed;
    let weight: Vec<u32> = (0..N * wc).map(|_| lcg(&mut st)).collect();
    let scales_len = (N * sc) as usize;
    let scales = if scales_signed {
        let v: Vec<i8> = (0..scales_len)
            .map(|_| (lcg(&mut st) % 17) as i8 - 8)
            .collect();
        MxArray::from_int8(&v, &[N, sc])
    } else {
        let v: Vec<u8> = (0..scales_len).map(|_| (lcg(&mut st) % 64) as u8).collect();
        MxArray::from_uint8(&v, &[N, sc])
    }
    .expect("scales");
    let biases: Vec<u16> = (0..N * bc)
        .map(|_| HALF_SCALES[(lcg(&mut st) % 6) as usize])
        .collect();
    (
        MxArray::from_uint32(&weight, &[N, wc]).expect("weight"),
        scales,
        MxArray::from_float16(&biases, &[N, bc]).expect("biases"),
    )
}

fn affine_weights(seed: u32, bits: i64, group_size: i64) -> Weights {
    let wc = K * bits / 32;
    let gc = K / group_size;
    let mut st = seed;
    let weight: Vec<u32> = (0..N * wc).map(|_| lcg(&mut st)).collect();
    let scales: Vec<u16> = (0..N * gc)
        .map(|_| BF16_SCALES[(lcg(&mut st) % 6) as usize])
        .collect();
    let biases: Vec<u16> = (0..N * gc)
        .map(|_| BF16_SCALES[(lcg(&mut st) % 6) as usize] | 0x8000)
        .collect();
    (
        MxArray::from_uint32(&weight, &[N, wc]).expect("weight"),
        MxArray::from_bfloat16(&scales, &[N, gc]).expect("scales"),
        MxArray::from_bfloat16(&biases, &[N, gc]).expect("biases"),
    )
}

fn qmm(x: &MxArray, c: &Config) -> *mut mlx_sys::mlx_array {
    let mode = CString::new(c.mode).expect("mode");
    // SAFETY: every handle outlives the call.
    let h = unsafe {
        mlx_sys::mlx_quantized_matmul(
            x.as_raw_ptr(),
            c.weights.0.as_raw_ptr(),
            c.weights.1.as_raw_ptr(),
            c.weights.2.as_raw_ptr(),
            true,
            c.group_size,
            c.bits,
            mode.as_ptr(),
        )
    };
    assert!(!h.is_null(), "{}: rejected at construction", c.label);
    h
}

/// Build REPS graphs, then time only the evals.
fn measure(x: &MxArray, c: &Config) -> f64 {
    let handles: Vec<_> = (0..REPS).map(|_| qmm(x, c)).collect();
    let t0 = Instant::now();
    for &h in &handles {
        eval(h);
    }
    let dt = t0.elapsed().as_secs_f64();
    for h in handles {
        drop_array(h);
    }
    dt / REPS as f64
}

fn trimmed_mean(v: &mut [f64]) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let trim = v.len() / 4;
    let body = &v[trim..v.len() - trim];
    let mean = body.iter().sum::<f64>() / body.len() as f64;
    let var = body.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / body.len() as f64;
    (mean, var.sqrt())
}

#[test]
fn kquant_vs_affine_prefill() {
    if !select_gpu() {
        eprintln!("skipping: no GPU device");
        return;
    }

    // Without the NAX kernels both arms of the comparison fall back to the same
    // simdgroup `qmm_t`, the ratio sits at ~1.0 and the run proves nothing. Say
    // so and stop before allocating the weights, rather than reporting a pass.
    if !nax_available() {
        let msg = "this GPU has no NAX kernels, so the K-quant and affine arms would both \
                   run the simdgroup qmm_t and the ratio would mean nothing. Causes, per \
                   mlx/backend/metal/device.cpp:944-962: GPU generation below 17 (below 18 \
                   when the architecture string ends in 'p'); macOS older than 26.2; or a \
                   metallib built without the NAX kernels (MLX_METAL_NO_NAX). Set \
                   KQ_NAX_REQUIRE=1 to make this a failure instead of a skip.";
        assert!(
            std::env::var("KQ_NAX_REQUIRE").as_deref() != Ok("1"),
            "KQ_NAX_REQUIRE=1 but {msg}"
        );
        eprintln!("skipping kquant_vs_affine_prefill: {msg}");
        return;
    }
    println!("  nax=true");

    // Both arms must reach `qmm`, which is where the NAX branch lives. A
    // split_k above 1 would keep them in `qmm_t_splitk` instead — still moving
    // together, still landing near 1.0, still proving nothing.
    let split_k = split_k_for_bench_shape();
    assert_eq!(
        split_k, 1,
        "M={M} N={N} K={K} gives split_k={split_k}, so quantized_matmul stays in \
         qmm_t_splitk and never reaches the NAX branch in qmm \
         (mlx/backend/metal/quantized.cpp:1071-1092). Pick a shape with at least 512 \
         32x32 output tiles."
    );

    let x = activation(&[M, K], 7);

    // q6k  bits 6 gs 16: 48 u32 / 16 int8 / 1 f16 per 256-value super-block
    // q4k  bits 4 gs 32: 32 u32 / 16 u8   / 2 f16
    // q5k  bits 5 gs 32: 40 u32 / 16 u8   / 2 f16
    let configs = vec![
        Config {
            label: "affine_A (control)",
            mode: "affine",
            bits: 4,
            group_size: 32,
            weights: affine_weights(0x1111_2222, 4, 32),
        },
        Config {
            label: "affine_B (control)",
            mode: "affine",
            bits: 4,
            group_size: 32,
            weights: affine_weights(0x3333_4444, 4, 32),
        },
        Config {
            label: "q6k",
            mode: "q6k",
            bits: 6,
            group_size: 16,
            weights: kquant_weights(0x5eed_1234, true, 48, 16, 1),
        },
        Config {
            label: "q4k",
            mode: "q4k",
            bits: 4,
            group_size: 32,
            weights: kquant_weights(0x5eed_5678, false, 32, 16, 2),
        },
        Config {
            label: "q5k",
            mode: "q5k",
            bits: 5,
            group_size: 32,
            weights: kquant_weights(0x5eed_9abc, false, 40, 16, 2),
        },
    ];

    for c in &configs {
        for _ in 0..WARMUP {
            let h = qmm(&x, c);
            eval(h);
            drop_array(h);
        }
    }

    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); configs.len()];
    for round in 0..ROUNDS {
        for step in 0..configs.len() {
            let i = (round + step) % configs.len();
            samples[i].push(measure(&x, &configs[i]));
        }
    }

    println!("\n  M={M} N={N} K={K} bf16  reps={REPS} rounds={ROUNDS}");
    println!(
        "  {:<20} {:>12} {:>10} {:>12} {:>12}",
        "config", "ms/call", "rel-sd", "TFLOP/s", "vs affine_A"
    );
    let flops = 2.0 * M as f64 * N as f64 * K as f64;
    let mut base = 0.0;
    let mut stats = Vec::new();
    for (i, c) in configs.iter().enumerate() {
        let (mean, sd) = trimmed_mean(&mut samples[i]);
        if i == 0 {
            base = mean;
        }
        stats.push((mean, sd));
        println!(
            "  {:<20} {:>12.4} {:>9.2}% {:>12.2} {:>12.4}",
            c.label,
            mean * 1e3,
            100.0 * sd / mean,
            flops / mean / 1e12,
            mean / base
        );
    }
    // Ratio error bar: relative sds add in quadrature.
    println!();
    let mut worst: Option<(&str, f64)> = None;
    for (i, c) in configs.iter().enumerate().skip(1) {
        let (m, s) = stats[i];
        let (m0, s0) = stats[0];
        let r = m / m0;
        let re = r * ((s / m).powi(2) + (s0 / m0).powi(2)).sqrt();
        println!("  {:<20} vs affine_A: {r:.4} +/- {re:.4}", c.label);
        if c.mode != "affine" && worst.is_none_or(|(_, w)| r > w) {
            worst = Some((c.label, r));
        }
    }
    println!();

    let (label, ratio) = worst.expect("at least one K-quant config");
    assert!(
        ratio <= KQUANT_VS_AFFINE_CEILING,
        "{label} ran {ratio:.4}x the affine control, over the {KQUANT_VS_AFFINE_CEILING:.1}x \
         ceiling. NAX is available and the shape reaches qmm, both checked above, so the \
         K-quant qmm_t took the NAX branch and was still slow, or nax_supports_mode \
         (mlx/backend/metal/quantized.cpp:66) stopped admitting it and it is back on the \
         simdgroup kernel. gpu_matches_cpu_on_every_quantized_matmul_kernel in \
         kquant_mode_guards.rs distinguishes the two."
    );
}
