//! Manual benchmark of the Tiled64 K-quant layout (`@t64`) against the
//! row-major layout, on the Qwen3.8-27B shapes: M = 1 (row-major `qmv`
//! against `qmv_t64`), M = 8 (the row-major route, `qmv_sg8` for the sg8
//! modes and `qmm_m8_nax` for q2k, against `qmm_m8_nax_t64`) and, with
//! `MLX_KQUANT_TILED_BENCH_PREFILL=1`, the prefill rows `PREFILL_ROWS`
//! (`qmm_t_nax` / `qmm_t_splitk` in both layouts).
//! Samples interleave the layouts inside one process so both see the same
//! clock state; the statistic is the median over `REPS` samples of a
//! `BATCH`-matmul eval (host overhead paid once). Every matmul of a batch
//! reads the next copy of a >= `RING_BYTES` weight ring, as the Splash
//! harness does, so the weights come from DRAM rather than the system cache
//! (a decode step streams the whole model once). Run only on an idle GPU:
//!
//! ```text
//! MLX_KQUANT_TILED_BENCH=1 cargo test -p mlx-core --release \
//!   --test kquant_tiled_bench -- --ignored --nocapture
//! ```

mod kquant_support;

use std::ffi::CString;
use std::time::Instant;

use kquant_support::*;
use mlx_core::array::{DType, MxArray};
use mlx_core::models::quant_dispatch::{KQUANT_TILED_SUFFIX, kquant_tile_rows};

const WARMUP: usize = 4;
/// `MLX_KQUANT_TILED_BENCH_REPS` overrides (odd, >= 3).
const REPS: usize = 21;
const BATCH: usize = 8;
const RING_BYTES: usize = 384 << 20;

/// (K, N): the Qwen3.8-27B verify projections.
const SHAPES: [(i64, i64); 6] = [
    (5120, 17408),
    (17408, 5120),
    (5120, 6144),
    (6144, 5120),
    (5120, 1024),
    (5120, 16384),
];

/// The modes a UD-Q4_K_M Qwen3.8 GGUF carries, plus q2k (the 2-bit K-quant,
/// measured next to q4k) and the seven grid formats of the Unsloth UD-IQ* /
/// UD-Q2_K_XL mixes (their decode is table lookups, so the question is
/// whether M = 1 stays bandwidth-bound at 1.56-3.56 bpw).
const MODES: [&str; 11] = [
    "q4k", "q5k", "q6k", "q2k", "iq2xxs", "iq2xs", "iq2s", "iq3xxs", "iq1s", "iq1m", "iq3s",
];
/// Prefill row counts: one NAX tile of rows, and a long prompt.
const PREFILL_ROWS: [i64; 2] = [64, 2048];

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

fn qmm(x: &MxArray, w: &Weights, kq: &KQuant, mode: &CString) -> *mut mlx_sys::mlx_array {
    // SAFETY: every operand outlives the call.
    let h = unsafe {
        mlx_sys::mlx_quantized_matmul(
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            true,
            kq.group_size,
            kq.bits,
            mode.as_ptr(),
        )
    };
    assert!(!h.is_null(), "{} qmm construction failed", kq.mode);
    h
}

fn eval_all(hs: &mut [*mut mlx_sys::mlx_array]) {
    let mut buf = [0i8; 512];
    // SAFETY: live handles; `buf` receives the error text.
    let ok = unsafe {
        mlx_sys::mlx_eval_with_error(hs.as_mut_ptr(), hs.len(), buf.as_mut_ptr(), buf.len())
    };
    assert!(ok, "qmm eval failed");
    for &h in hs.iter() {
        // SAFETY: owned handle, not used afterwards.
        unsafe { mlx_sys::mlx_array_delete(h) };
    }
}

/// A DRAM-cold ring of weight copies, one layout.
struct Ring {
    copies: Vec<Weights>,
    next: std::cell::Cell<usize>,
    /// Bytes one matmul streams (`.weight` + `.scales` + `.biases`).
    bytes: usize,
}

impl Ring {
    fn new(kq: &KQuant, n: i64, k: i64, seed: u32, tile: bool) -> Self {
        let first = weights(kq, &[n], k, seed);
        let bytes = first.w.nbytes() + first.scales.nbytes() + first.biases.nbytes();
        let count = RING_BYTES.div_ceil(bytes).clamp(3, 64);
        let copies = (0..count)
            .map(|i| {
                let w = if i == 0 {
                    weights(kq, &[n], k, seed)
                } else {
                    weights(kq, &[n], k, seed + i as u32)
                };
                if tile {
                    tiled(&w, kq)
                } else {
                    w.w.eval();
                    w.scales.eval();
                    w.biases.eval();
                    w
                }
            })
            .collect();
        Self {
            copies,
            next: std::cell::Cell::new(0),
            bytes,
        }
    }

    fn take(&self) -> &Weights {
        let i = self.next.get();
        self.next.set(i + 1);
        &self.copies[i % self.copies.len()]
    }
}

fn batch_ms(x: &MxArray, ring: &Ring, kq: &KQuant, mode: &CString) -> f64 {
    let mut hs: Vec<_> = (0..BATCH).map(|_| qmm(x, ring.take(), kq, mode)).collect();
    let t = Instant::now();
    eval_all(&mut hs);
    t.elapsed().as_secs_f64() * 1e3 / BATCH as f64
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn reps() -> usize {
    std::env::var("MLX_KQUANT_TILED_BENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(REPS)
}

fn modes() -> Vec<&'static str> {
    match std::env::var("MLX_KQUANT_TILED_BENCH_MODES") {
        Ok(v) => MODES
            .iter()
            .copied()
            .filter(|m| v.split(',').any(|s| s == *m))
            .collect(),
        Err(_) => MODES.to_vec(),
    }
}

struct Arm<'a> {
    w: &'a Ring,
    mode: CString,
}

/// Interleaved medians of every arm; the first arm is the baseline.
fn measure(x: &MxArray, kq: &KQuant, arms: &[Arm]) -> Vec<f64> {
    for _ in 0..WARMUP {
        for a in arms {
            let _ = batch_ms(x, a.w, kq, &a.mode);
        }
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    for i in 0..reps() {
        let order: Vec<usize> = if i % 2 == 0 {
            (0..arms.len()).collect()
        } else {
            (0..arms.len()).rev().collect()
        };
        for ai in order {
            let a = &arms[ai];
            samples[ai].push(batch_ms(x, a.w, kq, &a.mode));
        }
    }
    samples.iter_mut().map(|s| median(s)).collect()
}

#[test]
#[ignore = "manual Tiled64 vs row-major K-quant microbenchmark"]
fn qwen38_tiled_vs_row_major() {
    if std::env::var("MLX_KQUANT_TILED_BENCH").as_deref() != Ok("1") {
        eprintln!("set MLX_KQUANT_TILED_BENCH=1 to run this benchmark");
        return;
    }
    assert!(gpu_gen() > 0, "no Metal device");
    println!(
        "\n  Tiled64 vs row-major, bfloat16, warmup={WARMUP}, reps={REPS} (medians, interleaved), \
         per matmul of {BATCH} per eval; ratios are row-major / tiled (>1 = tiled faster)"
    );
    for mode in modes() {
        let kq = kquant(mode);
        println!(
            "\n  {:<6} {:>6} {:>6} | {:>8} {:>8} {:>6} {:>7} | {:>8} {:>8} {:>6}  (ms, DRAM-cold ring; GB/s = weight bytes / M1 t64 ms)",
            "mode", "K", "N", "M1 qmv", "M1 t64", "ratio", "GB/s", "M8 sg8", "M8 t64", "ratio"
        );
        for (si, &(k, n)) in SHAPES.iter().enumerate() {
            let w = Ring::new(kq, n, k, 0x9000 + si as u32, false);
            let t = Ring::new(kq, n, k, 0x9000 + si as u32, true);
            let arms = [
                Arm {
                    w: &w,
                    mode: mode_cstr(kq),
                },
                Arm {
                    w: &t,
                    mode: CString::new(format!("{mode}{KQUANT_TILED_SUFFIX}")).unwrap(),
                },
            ];
            let x1 = activation(&[1, k], 0x9100 + si as u32, DType::BFloat16);
            let m1 = measure(&x1, kq, &arms);
            let x8 = activation(&[8, k], 0x9200 + si as u32, DType::BFloat16);
            let m8 = measure(&x8, kq, &arms);
            println!(
                "  {mode:<6} {k:>6} {n:>6} | {:>8.4} {:>8.4} {:>5.2}x {:>7.1} | {:>8.4} {:>8.4} {:>5.2}x",
                m1[0],
                m1[1],
                m1[0] / m1[1],
                t.bytes as f64 / 1e6 / m1[1],
                m8[0],
                m8[1],
                m8[0] / m8[1]
            );
            if std::env::var("MLX_KQUANT_TILED_BENCH_PREFILL").as_deref() == Ok("1") {
                for rows in PREFILL_ROWS {
                    let xp = activation(&[rows, k], 0x9300 + si as u32, DType::BFloat16);
                    let mp = measure(&xp, kq, &arms);
                    println!(
                        "  {mode:<6} {k:>6} {n:>6} | prefill M={rows:<5} row-major {:>8.4} t64 {:>8.4} {:>5.2}x",
                        mp[0],
                        mp[1],
                        mp[0] / mp[1]
                    );
                }
            }
        }
    }
}
