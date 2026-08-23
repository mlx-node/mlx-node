//! Manual small-M benchmark for the native GGUF K/IQ matmul path.
//!
//! The default shape is Qwen3.8-27B's dominant MLP projection
//! `[N=17408, K=5120]`; M=1 measures decode and M=2/M=4/M=6 cover the
//! verification widths reachable by the current depth-1 through depth-5 MTP
//! policy. Run only on an otherwise idle GPU:
//!
//! ```text
//! MLX_KQUANT_SMALL_M_BENCH=1 cargo test -p mlx-core --release \
//!   --test kquant_small_m_bench -- --ignored --nocapture
//! ```

use std::ffi::CString;
use std::time::Instant;

use mlx_core::array::MxArray;

const N: i64 = 17_408;
const K: i64 = 5_120;
const WARMUP: usize = 4;
const REPS: usize = 17;

#[derive(Clone, Copy)]
struct Format {
    mode: &'static str,
    bits: i32,
    group_size: i32,
    scales_cols: i64,
    biases_cols: i64,
    signed_scales: bool,
}

const FORMATS: [Format; 4] = [
    Format {
        mode: "q5k",
        bits: 5,
        group_size: 32,
        scales_cols: 2 * K / 32,
        biases_cols: 2 * K / 256,
        signed_scales: false,
    },
    Format {
        mode: "iq4xs",
        bits: 4,
        group_size: 32,
        scales_cols: K / 32,
        biases_cols: K / 256,
        signed_scales: true,
    },
    Format {
        mode: "q6k",
        bits: 6,
        group_size: 16,
        scales_cols: K / 16,
        biases_cols: K / 256,
        signed_scales: true,
    },
    Format {
        mode: "q4k",
        bits: 4,
        group_size: 32,
        scales_cols: 2 * K / 32,
        biases_cols: 2 * K / 256,
        signed_scales: false,
    },
];

fn select_gpu() -> bool {
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

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn activation(m: i64, seed: u32) -> MxArray {
    let mut state = seed;
    let values: Vec<u16> = (0..m * K)
        .map(|_| {
            // Small, finite BF16 values in roughly [-1, 1].
            let value = ((lcg(&mut state) >> 16) as i32 - 32_768) as f32 / 32_768.0;
            half::bf16::from_f32(value).to_bits()
        })
        .collect();
    MxArray::from_bfloat16(&values, &[m, K]).expect("activation")
}

fn packed_arrays(format: Format, seed: u32) -> (MxArray, MxArray, MxArray, u64) {
    let mut state = seed;
    let weight_cols = K * i64::from(format.bits) / 32;
    let weight: Vec<u32> = (0..N * weight_cols).map(|_| lcg(&mut state)).collect();
    let scales_len = N * format.scales_cols;
    let scales = if format.signed_scales {
        let values: Vec<i8> = (0..scales_len)
            .map(|_| (lcg(&mut state) % 31) as i8 - 15)
            .collect();
        MxArray::from_int8(&values, &[N, format.scales_cols]).expect("signed scales")
    } else {
        let values: Vec<u8> = (0..scales_len)
            .map(|_| (lcg(&mut state) % 48 + 1) as u8)
            .collect();
        MxArray::from_uint8(&values, &[N, format.scales_cols]).expect("unsigned scales")
    };
    let half_scales = [0x2800u16, 0x2c00, 0x3000, 0x3200, 0x3400, 0x3600];
    let biases: Vec<u16> = (0..N * format.biases_cols)
        .map(|_| half_scales[(lcg(&mut state) as usize) % half_scales.len()])
        .collect();
    let resident_bytes = weight.len() as u64 * 4 + scales_len as u64 + biases.len() as u64 * 2;
    (
        MxArray::from_uint32(&weight, &[N, weight_cols]).expect("weight"),
        scales,
        MxArray::from_float16(&biases, &[N, format.biases_cols]).expect("biases"),
        resident_bytes,
    )
}

fn qmm(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    format: Format,
) -> *mut mlx_sys::mlx_array {
    let mode = CString::new(format.mode).expect("mode");
    let handle = unsafe {
        mlx_sys::mlx_quantized_matmul(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
            true,
            format.group_size,
            format.bits,
            mode.as_ptr(),
        )
    };
    assert!(!handle.is_null(), "{} qmm construction failed", format.mode);
    handle
}

fn eval(handle: *mut mlx_sys::mlx_array) {
    let mut error = [0i8; 512];
    let mut handle = handle;
    let ok =
        unsafe { mlx_sys::mlx_eval_with_error(&mut handle, 1, error.as_mut_ptr(), error.len()) };
    assert!(ok, "qmm eval failed");
}

fn drop_array(handle: *mut mlx_sys::mlx_array) {
    unsafe { mlx_sys::mlx_array_delete(handle) };
}

fn median_ms(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2] * 1e3
}

fn measure(x: &MxArray, w: &MxArray, scales: &MxArray, biases: &MxArray, format: Format) -> f64 {
    for _ in 0..WARMUP {
        let handle = qmm(x, w, scales, biases, format);
        eval(handle);
        drop_array(handle);
    }
    let handles: Vec<_> = (0..REPS)
        .map(|_| qmm(x, w, scales, biases, format))
        .collect();
    let mut samples = Vec::with_capacity(REPS);
    for handle in handles {
        let started = Instant::now();
        eval(handle);
        samples.push(started.elapsed().as_secs_f64());
        drop_array(handle);
    }
    median_ms(&mut samples)
}

#[test]
#[ignore = "manual exact-shape K/IQ small-M microbenchmark"]
fn qwen38_dominant_small_m_qmm() {
    if std::env::var("MLX_KQUANT_SMALL_M_BENCH").as_deref() != Ok("1") {
        eprintln!("set MLX_KQUANT_SMALL_M_BENCH=1 to run this benchmark");
        return;
    }
    if !select_gpu() {
        eprintln!("skipping: no GPU device");
        return;
    }

    println!("\n  BF16 input, N={N}, K={K}, warmup={WARMUP}, reps={REPS}");
    println!(
        "  {:<8} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "mode", "M1 ms", "M2 ms", "M4 ms", "M6 ms", "M2 GB/s"
    );
    for (index, format) in FORMATS.into_iter().enumerate() {
        let (w, scales, biases, resident_bytes) = packed_arrays(format, 0x51a7_0000 + index as u32);
        w.eval();
        scales.eval();
        biases.eval();
        let m1 = measure(&activation(1, 0x1010), &w, &scales, &biases, format);
        let m2 = measure(&activation(2, 0x2020), &w, &scales, &biases, format);
        let m4 = measure(&activation(4, 0x4040), &w, &scales, &biases, format);
        let m6 = measure(&activation(6, 0x6060), &w, &scales, &biases, format);
        let gb_s = resident_bytes as f64 / (m2 / 1e3) / 1e9;
        println!(
            "  {:<8} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>12.1}",
            format.mode, m1, m2, m4, m6, gb_s
        );
    }
}
