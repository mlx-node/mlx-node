//! Shape-scoped T64 register-MMA output projection and explicit rollback.
mod kquant_support;

use kquant_support::*;
use mlx_core::array::{DType, MxArray};
use mlx_core::models::quant_dispatch::kquant_tile_rows;
use std::ffi::CString;

const K: i64 = 4096;
const N: i64 = 2048;
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn fixture() -> (&'static KQuant, Weights, Weights, MxArray) {
    let q = KQUANTS.iter().find(|q| q.mode == "q4k").unwrap();
    let mut w = weights(q, &[N], K, 0x8791);
    w.biases = w
        .biases
        .mul_scalar(0.001)
        .unwrap()
        .astype(DType::Float16)
        .unwrap();
    let tiled = Weights {
        w: kquant_tile_rows(&w.w, 4).unwrap(),
        scales: kquant_tile_rows(&w.scales, 16).unwrap(),
        biases: kquant_tile_rows(&w.biases, 2).unwrap(),
    };
    let x = activation(&[8, K], 0x8712, DType::BFloat16);
    (q, w, tiled, x)
}

fn tiled_call(x: &MxArray, w: &Weights, q: &KQuant) -> *mut mlx_sys::mlx_array {
    let mode = CString::new("q4k@t64").unwrap();
    unsafe {
        mlx_sys::mlx_test_kquant_quantized_matmul(
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            true,
            q.group_size,
            q.bits,
            mode.as_ptr(),
            GPU,
        )
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "run with MLX_KQUANT_REGISTER_OUTPUT=1 to exercise the opt-in Metal path"]
fn register_t64_preserves_row_major_sg8_bits() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(
        std::env::var("MLX_KQUANT_REGISTER_OUTPUT").as_deref(),
        Ok("1")
    );
    assert!(gpu_gen() >= 17);
    let (q, w, tiled, x) = fixture();
    start_counting();
    for seed in [0x8712, 0x8713, 0x8714] {
        let x = if seed == 0x8712 {
            x.clone()
        } else {
            activation(&[8, K], seed, DType::BFloat16)
        };
        assert_identical(
            "T64 register projection",
            tiled_call(&x, &tiled, q),
            quantized_matmul(&x, &w, true, q, GPU),
        );
    }
    assert_eq!(family_count("qmv_sg8_t64"), 3);
    assert_eq!(family_count("qmv_sg8"), 3);
    stop_counting();
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "bounded isolated microbenchmark; run in separate processes with the experiment on and off"]
fn register_projection_microbenchmark() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    assert!(gpu_gen() >= 17);
    let (q, _, w, x) = fixture();
    // Batch independent roots so per-op synchronization does not dominate
    // the approximately tens-of-microseconds GPU projection being measured.
    const BATCH: usize = 64;
    let evaluate = || {
        let mut outputs: Vec<_> = (0..BATCH).map(|_| tiled_call(&x, &w, q)).collect();
        assert!(outputs.iter().all(|h| !h.is_null()));
        let mut error = [0i8; 512];
        let start = std::time::Instant::now();
        assert!(
            unsafe {
                mlx_sys::mlx_eval_with_error(
                    outputs.as_mut_ptr(),
                    outputs.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            "projection GPU evaluation failed"
        );
        let microseconds = start.elapsed().as_secs_f64() * 1e6 / BATCH as f64;
        let last = outputs.pop().unwrap();
        for output in outputs {
            unsafe { mlx_sys::mlx_array_delete(output) };
        }
        let (_, _, bits) = read_output("timed projection", last);
        (microseconds, bits[0])
    };
    for _ in 0..10 {
        evaluate();
    }
    let mut timings = Vec::with_capacity(100);
    let mut checksum = 0u32;
    for _ in 0..100 {
        let (time, first) = evaluate();
        timings.push(time);
        checksum = checksum.wrapping_mul(16777619) ^ first;
    }
    timings.sort_by(f64::total_cmp);
    println!(
        "register_output={} M=8 N={N} K={K} median_us={:.3} p10_us={:.3} checksum={checksum}",
        std::env::var("MLX_KQUANT_REGISTER_OUTPUT").unwrap_or_else(|_| "unset".into()),
        timings[50],
        timings[10]
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "independent exact-dequantized CPU oracle, run with register output both off and on"]
fn register_projection_cpu_oracle() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    assert!(gpu_gen() >= 17);
    let (q, row_major, tiled, x) = fixture();
    let (_, _, decoded) = read_bits(
        "CPU dequantized weights",
        dequantize(&row_major, DType::Float32, q, CPU),
    );
    let decoded: Vec<f32> = decoded.into_iter().map(f32::from_bits).collect();
    let input = x.astype(DType::Float32).unwrap().to_float32().unwrap();
    let (_, _, bits) = read_output("GPU projection", tiled_call(&x, &tiled, q));
    let mut max_error = 0.0f64;
    let mut peak = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut squared_reference = 0.0f64;
    for (r, input) in input.as_chunks::<{ K as usize }>().0.iter().enumerate() {
        for (n, w) in decoded.as_chunks::<{ K as usize }>().0.iter().enumerate() {
            let reference = w
                .iter()
                .zip(input)
                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                .sum::<f64>();
            let actual = f64::from(half::bf16::from_bits(bits[r * N as usize + n] as u16).to_f32());
            assert!(actual.is_finite());
            let error = actual - reference;
            max_error = max_error.max(error.abs());
            peak = peak.max(reference.abs());
            squared_error += error * error;
            squared_reference += reference * reference;
        }
    }
    let relative_rms = (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt();
    println!(
        "register_output={} cpu_oracle max_error={} peak={} normalized_peak_error={} relative_rms={}",
        std::env::var("MLX_KQUANT_REGISTER_OUTPUT").unwrap_or_else(|_| "unset".into()),
        max_error,
        peak,
        max_error / peak,
        relative_rms
    );
    assert!(
        max_error <= 0.01 * peak,
        "projection differs from exact dequantized matvec by >1% of peak"
    );
    assert!(
        relative_rms <= 0.005,
        "projection relative RMS error exceeds 0.5%"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "exclusive GPU, stable register output unset/1/0 dispatch policy"]
fn register_output_default_and_optout_dispatch() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    assert!(gpu_gen() >= 17);
    struct Restore(&'static str, Option<std::ffi::OsString>);
    impl Drop for Restore {
        fn drop(&mut self) {
            unsafe {
                match &self.1 {
                    Some(v) => std::env::set_var(self.0, v),
                    None => std::env::remove_var(self.0),
                }
            }
        }
    }
    let flag = "MLX_KQUANT_REGISTER_OUTPUT";
    let _flag = Restore(flag, std::env::var_os(flag));
    let _split = Restore(
        "MLX_KQUANT_SG8_SPLIT_K",
        std::env::var_os("MLX_KQUANT_SG8_SPLIT_K"),
    );
    unsafe { std::env::set_var("MLX_KQUANT_SG8_SPLIT_K", "0") };
    let (q, _, tiled, x) = fixture();
    let mut default_output = None;
    for value in [None, Some("1"), Some("0"), Some("invalid")] {
        unsafe {
            match value {
                Some(v) => std::env::set_var(flag, v),
                None => std::env::remove_var(flag),
            }
        }
        start_counting();
        let output = read_output("register output default policy", tiled_call(&x, &tiled, q));
        let enabled = value.is_none_or(|v| v == "1");
        assert_eq!(family_count("qmv_sg8_t64"), u64::from(enabled));
        assert_eq!(family_count("qmm_m8_nax_t64"), u64::from(!enabled));
        stop_counting();
        if value.is_none() {
            default_output = Some(output.clone());
        } else if enabled {
            assert_eq!(Some(output), default_output, "unset must equal explicit 1");
        }
    }
    // Changing only the flag cannot make an unsupported row count/dtype
    // eligible. Compare exact fallback outputs across unset/1/0.
    for (rows, dtype) in [
        (1, DType::BFloat16),
        (7, DType::BFloat16),
        (33, DType::BFloat16),
        (8, DType::Float32),
    ] {
        let x = activation(&[rows, K], 8719, dtype);
        let mut reference = None;
        for value in [None, Some("1"), Some("0")] {
            unsafe {
                match value {
                    Some(v) => std::env::set_var(flag, v),
                    None => std::env::remove_var(flag),
                }
            }
            start_counting();
            let output = read_output("register output fallback", tiled_call(&x, &tiled, q));
            assert_eq!(family_count("qmv_sg8_t64"), 0);
            stop_counting();
            if let Some(expected) = &reference {
                assert_eq!(&output, expected);
            } else {
                reference = Some(output);
            }
        }
    }
}
