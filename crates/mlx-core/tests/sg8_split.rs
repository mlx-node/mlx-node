//! Opt-in SG8 K splits, independent decoded-weight F64 oracle and replay.
mod kquant_support;
use kquant_support::*;
use mlx_core::array::{DType, MxArray};
use mlx_core::models::quant_dispatch::{kquant_tile_rows, kquant_untile_rows};
use mlx_core::stream::{DeviceType, Stream, StreamContext};
use std::ffi::CString;
use std::sync::atomic::{AtomicUsize, Ordering};
const FLAG: &str = "MLX_KQUANT_SG8_SPLIT_K";
fn setting(on: bool) {
    unsafe { std::env::set_var(FLAG, if on { "1" } else { "0" }) }
}
struct Restore(Vec<(&'static str, Option<std::ffi::OsString>)>);
impl Restore {
    fn new() -> Self {
        Self(
            [
                FLAG,
                "MLX_KQUANT_REGISTER_OUTPUT",
                "MLX_KQUANT_REGISTER_AFFINE",
            ]
            .into_iter()
            .map(|s| (s, std::env::var_os(s)))
            .collect(),
        )
    }
}
impl Drop for Restore {
    fn drop(&mut self) {
        for (k, v) in &self.0 {
            unsafe {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
    }
}
fn call(x: &MxArray, w: &Weights, q: &KQuant, tiled: bool) -> *mut mlx_sys::mlx_array {
    let mode = CString::new(format!("{}{}", q.mode, if tiled { "@t64" } else { "" })).unwrap();
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
fn tile(w: &Weights, q: &KQuant) -> Weights {
    Weights {
        w: kquant_tile_rows(&w.w, q.bits as i64).unwrap(),
        scales: kquant_tile_rows(&w.scales, if q.mode == "a4g64" { 4 } else { 16 }).unwrap(),
        biases: kquant_tile_rows(&w.biases, if q.mode == "a4g64" { 4 } else { 2 }).unwrap(),
    }
}
fn oracle(q: &KQuant, w: &Weights, x: &MxArray, actual: &[u32], label: &str) {
    let (_, dtype, decoded) =
        read_bits("CPU decoded weights", dequantize(w, DType::Float32, q, CPU));
    assert_eq!(dtype, DType::Float32);
    let input = x.astype(DType::Float32).unwrap().to_float32().unwrap();
    let k = x.shape().unwrap()[1] as usize;
    let n = w.w.shape().unwrap()[0] as usize;
    let (mut peak, mut max, mut square, mut reference) = (0f64, 0f64, 0f64, 0f64);
    for r in 0..8 {
        for c in 0..n {
            let want = (0..k)
                .map(|j| {
                    f64::from(f32::from_bits(decoded[c * k + j])) * f64::from(input[r * k + j])
                })
                .sum::<f64>();
            let got = half::bf16::from_bits(actual[r * n + c] as u16).to_f64();
            assert!(got.is_finite());
            let error = (got - want).abs();
            peak = peak.max(want.abs());
            max = max.max(error);
            square += error * error;
            reference += want * want;
        }
    }
    let rms = (square / reference.max(f64::MIN_POSITIVE)).sqrt();
    eprintln!("SG8 {label} peak_error={} relative_rms={rms}", max / peak);
    assert!(
        max <= 0.01 * peak + 1e-8 && rms <= 0.005,
        "{label}: independent F64 oracle"
    );
}
fn affine(n: i64, k: i64) -> (KQuant, Weights) {
    let q = KQuant {
        mode: "a4g64",
        bits: 4,
        group_size: 64,
        scales_signed: false,
        weight_cols: 32,
        scales_cols: 4,
        biases_cols: 4,
    };
    let mut state = 0xa430u32;
    let mut next = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        state
    };
    let codes: Vec<_> = (0..n * k / 8).map(|_| next()).collect();
    let scales: Vec<_> = (0..n * k / 64)
        .map(|_| half::bf16::from_f32(0.004 + (next() >> 8) as f32 / 16777216.0 * 0.016).to_bits())
        .collect();
    let biases: Vec<_> = (0..n * k / 64)
        .map(|_| half::bf16::from_f32(-((next() >> 8) as f32) / 16777216.0 * 0.15).to_bits())
        .collect();
    (
        q,
        Weights {
            w: MxArray::from_uint32(&codes, &[n, k / 8]).unwrap(),
            scales: MxArray::from_bfloat16(&scales, &[n, k / 64]).unwrap(),
            biases: MxArray::from_bfloat16(&biases, &[n, k / 64]).unwrap(),
        },
    )
}
#[test]
#[ignore = "exclusive GPU; process-local experiment flags"]
fn sg8_split_formats_and_fallback_match_f64() {
    assert!(gpu_gen() >= 17);
    let _restore = Restore::new();
    for mode in ["q4k", "q5k", "q6k", "iq4xs"] {
        let q = kquant(mode);
        for (n, k) in [(64, 512), (512, 2048), (512, 4096), (96, 2304), (64, 256)] {
            let mut w = weights(q, &[n], k, 0x8000 + k as u32);
            w.biases = w
                .biases
                .mul_scalar(0.001)
                .unwrap()
                .astype(DType::Float16)
                .unwrap();
            let x = activation(&[8, k], 0x8001, DType::BFloat16);
            setting(false);
            let off = read_output("unsplit", call(&x, &w, q, false));
            setting(true);
            start_counting();
            let on = read_output("split", call(&x, &w, q, false));
            let selected = family_count("qmv_sg8_split");
            stop_counting();
            oracle(q, &w, &x, &on.2, &format!("{mode} N{n} K{k}"));
            if k == 2304 || k == 256 {
                assert_eq!(on, off, "whole-superblock fallback must stay exact");
                assert_eq!(selected, 0);
            } else {
                assert_eq!(selected, 1);
            }
        }
    }
    unsafe {
        std::env::set_var("MLX_KQUANT_REGISTER_AFFINE", "1");
        std::env::set_var("MLX_KQUANT_REGISTER_OUTPUT", "1")
    };
    for k in [2048, 4096, 6144] {
        let (q, w) = affine(2048, k);
        let tiled = tile(&w, &q);
        let x = activation(&[8, k], 0x8002, DType::BFloat16);
        setting(true);
        start_counting();
        let out = read_output("affine split", call(&x, &tiled, &q, true));
        assert_eq!(family_count("qmv_sg8_split_t64"), 1);
        stop_counting();
        oracle(&q, &w, &x, &out.2, &format!("A4 N2048 K{k}"));
    }
    let q = kquant("q4k");
    let w = weights(q, &[2048], 4096, 0x8003);
    let tiled = tile(&w, q);
    let x = activation(&[8, 4096], 0x8004, DType::BFloat16);
    for stream in [Stream::default(DeviceType::Gpu), Stream::generation()] {
        let _stream = StreamContext::new(stream);
        for _ in 0..8 {
            setting(true);
            let actual = read_output("T64 split", call(&x, &tiled, q, true));
            let reference = read_output("row split", call(&x, &w, q, false));
            assert_eq!(actual, reference, "T64 split addressing or counter reuse");
            // Interleave staged-NAX dispatches sharing the same stream's counter cache.
            unsafe { std::env::set_var("MLX_KQUANT_REGISTER_OUTPUT", "0") };
            let _ = read_output("interleaved staged", call(&x, &tiled, q, true));
            unsafe { std::env::set_var("MLX_KQUANT_REGISTER_OUTPUT", "1") };
        }
    }
}
struct Builder {
    traces: AtomicUsize,
}
unsafe extern "C" fn build(
    context: *mut std::ffi::c_void,
    inputs: *const *const mlx_sys::mlx_array,
    n: usize,
    outputs: *mut *mut mlx_sys::mlx_array,
    m: usize,
) -> bool {
    if context.is_null() || n != 4 || m != 1 {
        return false;
    }
    let state = unsafe { &*context.cast::<Builder>() };
    state.traces.fetch_add(1, Ordering::Relaxed);
    let a = unsafe { std::slice::from_raw_parts(inputs, n) };
    let result = unsafe {
        mlx_sys::mlx_test_kquant_quantized_matmul(
            a[0].cast_mut(),
            a[1].cast_mut(),
            a[2].cast_mut(),
            a[3].cast_mut(),
            true,
            32,
            4,
            c"q4k".as_ptr(),
            GPU,
        )
    };
    for &h in a {
        unsafe { mlx_sys::mlx_array_delete(h.cast_mut()) }
    }
    unsafe { *outputs = result };
    !result.is_null()
}
#[test]
#[ignore = "exclusive GPU; dynamic graph and persistent-counter replay"]
fn sg8_split_compiled_weights_inputs_and_counter_reuse() {
    assert!(gpu_gen() >= 17);
    let _restore = Restore::new();
    setting(true);
    let id = 0x514B_0000_8;
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, id) };
    let mut state = Builder {
        traces: AtomicUsize::new(0),
    };
    let q = kquant("q4k");
    for seed in 1..=8 {
        let w = weights(q, &[512], 4096, seed);
        let x = activation(&[8, 4096], seed + 10, DType::BFloat16);
        let expected = read_output("eager split", call(&x, &w, q, false));
        let inputs = [
            x.as_raw_ptr().cast_const(),
            w.w.as_raw_ptr().cast_const(),
            w.scales.as_raw_ptr().cast_const(),
            w.biases.as_raw_ptr().cast_const(),
        ];
        let mut output = std::ptr::null_mut();
        assert!(unsafe {
            mlx_sys::mlx_compiled_graph_invoke(
                id,
                Some(build),
                (&mut state as *mut Builder).cast(),
                inputs.as_ptr(),
                4,
                &mut output,
                1,
                true,
            )
        });
        assert_eq!(read_output("compiled split", output), expected);
        assert_eq!(state.traces.load(Ordering::Relaxed), 1);
    }
    unsafe { mlx_sys::mlx_compiled_graph_erase_matching(u64::MAX, id) };
}
#[test]
#[ignore = "requires SG8_WEIGHTS prepared checkpoint; bounded single projection only"]
fn sg8_split_actual_target_output_matches_f64() {
    let path =
        std::env::var("SG8_WEIGHTS").expect("SG8_WEIGHTS points at prepared model.safetensors");
    let _restore = Restore::new();
    setting(true);
    unsafe { std::env::set_var("MLX_KQUANT_REGISTER_OUTPUT", "1") };
    let prefix = std::env::var("SG8_PROJECTION")
        .unwrap_or_else(|_| "model.layers.0.linear_attn.out_proj".into());
    let mut tensors = mlx_core::utils::load_safetensors_lazy(path).unwrap();
    let mut get = |suffix: &str| tensors.remove(&format!("{prefix}.{suffix}")).unwrap();
    let w = Weights {
        w: get("weight"),
        scales: get("scales"),
        biases: get("biases"),
    };
    drop(tensors);
    assert_eq!(w.w.shape().unwrap().as_ref(), [2048, 512]);
    let row = Weights {
        w: kquant_untile_rows(&w.w, 4).unwrap(),
        scales: kquant_untile_rows(&w.scales, 16).unwrap(),
        biases: kquant_untile_rows(&w.biases, 2).unwrap(),
    };
    let q = kquant("q4k");
    let x = activation(&[8, 4096], 0xa811, DType::BFloat16);
    start_counting();
    let out = read_output("actual split", call(&x, &w, q, true));
    assert_eq!(family_count("qmv_sg8_split_t64"), 1);
    stop_counting();
    oracle(q, &row, &x, &out.2, &prefix);
}

#[test]
#[ignore = "exclusive quiet GPU; fixed flags per process; batched projection roots"]
fn sg8_split_projection_microbenchmark() {
    assert!(gpu_gen() >= 17);
    let q = kquant("q4k");
    let w = weights(q, &[2048], 4096, 0xab81);
    let x = activation(&[8, 4096], 0xab82, DType::BFloat16);
    let evaluate = |roots: &mut [*mut mlx_sys::mlx_array]| {
        let mut error = [0i8; 1024];
        assert!(unsafe {
            mlx_sys::mlx_eval_with_error(
                roots.as_mut_ptr(),
                roots.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        });
        for &h in roots.iter() {
            unsafe { mlx_sys::mlx_array_delete(h) };
        }
    };
    for _ in 0..16 {
        let mut roots = (0..16).map(|_| call(&x, &w, q, false)).collect::<Vec<_>>();
        evaluate(&mut roots);
    }
    let mut samples = Vec::new();
    for _ in 0..31 {
        let mut roots = (0..16).map(|_| call(&x, &w, q, false)).collect::<Vec<_>>();
        let start = std::time::Instant::now();
        evaluate(&mut roots);
        samples.push(start.elapsed().as_secs_f64() * 1e6 / 16.0);
    }
    samples.sort_by(f64::total_cmp);
    eprintln!(
        "SG8_SPLIT {:?} N2048 K4096 median_us={} p10_us={}",
        std::env::var(FLAG),
        samples[15],
        samples[3]
    );
}
