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
use mlx_core::models::quant_dispatch::kquant_tile_rows;

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

/// lm_head shape [N=248320, K=5120] q6k — DFlash2 verify proposes L+1 rows
/// and the draft selector proposes L rows; M sweeps the depth-4..8 window.
#[test]
#[ignore = "manual exact-shape K/IQ small-M microbenchmark"]
fn qwen38_lm_head_small_m_qmm() {
    if std::env::var("MLX_KQUANT_SMALL_M_BENCH").as_deref() != Ok("1") {
        eprintln!("set MLX_KQUANT_SMALL_M_BENCH=1 to run this benchmark");
        return;
    }
    if !select_gpu() {
        eprintln!("skipping: no GPU device");
        return;
    }
    const LN: i64 = 248_320;
    let format = Format {
        mode: "q6k",
        bits: 6,
        group_size: 16,
        scales_cols: K / 16,
        biases_cols: K / 256,
        signed_scales: true,
    };
    let mut state = 0x51a7_0000u32;
    let weight_cols = K * i64::from(format.bits) / 32;
    let weight: Vec<u32> = (0..LN * weight_cols).map(|_| lcg(&mut state)).collect();
    let scales_len = LN * format.scales_cols;
    let scales: Vec<i8> = (0..scales_len)
        .map(|_| (lcg(&mut state) % 31) as i8 - 15)
        .collect();
    let biases: Vec<u16> = (0..LN * format.biases_cols)
        .map(|_| half_scales_for(lcg(&mut state)))
        .collect();
    fn half_scales_for(v: u32) -> u16 {
        [0x2800u16, 0x2c00, 0x3000, 0x3200, 0x3400, 0x3600][(v as usize) % 6]
    }
    let w = MxArray::from_uint32(&weight, &[LN, weight_cols]).expect("weight");
    let sc = MxArray::from_int8(&scales, &[LN, format.scales_cols]).expect("scales");
    let bi = MxArray::from_float16(&biases, &[LN, format.biases_cols]).expect("biases");
    w.eval();
    sc.eval();
    bi.eval();
    // The same weights in the Tiled64 layout: scales tile per super-block
    // (16 groups x 1 byte), biases per super-block (1 entry), codes per
    // 32-value unit (`bits` words).
    let tiled = Format {
        mode: "q6k@t64",
        ..format
    };
    let tw = kquant_tile_rows(&w, i64::from(format.bits)).expect("tile weight");
    let tsc = kquant_tile_rows(&sc, 16).expect("tile scales");
    let tbi = kquant_tile_rows(&bi, 1).expect("tile biases");
    tw.eval();
    tsc.eval();
    tbi.eval();
    println!("\n  lm_head BF16 x, N={LN}, K={K}, q6k: row-major | t64 (ms)");
    for m in [1i64, 4, 5, 6, 7, 8, 9, 12] {
        let x = activation(m, 0x9090 + m as u32);
        println!(
            "  M={m:>2}: {:.4} | {:.4}",
            measure(&x, &w, &sc, &bi, format),
            measure(&x, &tw, &tsc, &tbi, tiled)
        );
    }
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

/// The DFlash2 draft's affine Q4/group-64 projections at the verify widths:
/// MLX's own `quantized_matmul` route (M < 13 takes `qmv`, one threadgroup
/// per row), on draft-sized shapes. Eight matmuls per eval over a ring of
/// eight weight copies (DRAM-cold, launch latency amortised), as
/// `kquant_tiled_bench` does. If M=7/8 costs well over M=1, the draft
/// re-reads its weights per row and an 8-row kernel is the lever.
#[test]
#[ignore = "manual exact-shape affine small-M microbenchmark"]
fn draft_affine_q4_small_m_qmm() {
    if std::env::var("MLX_KQUANT_SMALL_M_BENCH").as_deref() != Ok("1") {
        eprintln!("set MLX_KQUANT_SMALL_M_BENCH=1 to run this benchmark");
        return;
    }
    if !select_gpu() {
        eprintln!("skipping: no GPU device");
        return;
    }
    const GROUP: i64 = 64;
    const RING: usize = 8;
    for (n, k) in [(27_648i64, 5_120i64), (5_120, 13_824), (7_168, 5_120)] {
        let format = Format {
            mode: "affine",
            bits: 4,
            group_size: GROUP as i32,
            scales_cols: k / GROUP,
            biases_cols: k / GROUP,
            signed_scales: false,
        };
        let ring: Vec<(MxArray, MxArray, MxArray)> = (0..RING)
            .map(|i| {
                let mut state = 0xa44e_0000u32 + i as u32;
                let weight: Vec<u32> = (0..n * k / 8).map(|_| lcg(&mut state)).collect();
                let scales: Vec<u16> = (0..n * k / GROUP)
                    .map(|_| half::bf16::from_f32(0.01).to_bits())
                    .collect();
                let w = MxArray::from_uint32(&weight, &[n, k / 8]).expect("weight");
                let sc = MxArray::from_bfloat16(&scales, &[n, k / GROUP]).expect("scales");
                let bi = MxArray::from_bfloat16(&scales, &[n, k / GROUP]).expect("biases");
                w.eval();
                sc.eval();
                bi.eval();
                (w, sc, bi)
            })
            .collect();
        let bytes = (n * k / 2 + 2 * n * k / GROUP * 2) as f64;
        println!(
            "\n  draft affine Q4/g64, N={n}, K={k}, {:.0} MB, {RING} matmuls per eval",
            bytes / 1e6
        );
        for m in [1i64, 4, 7, 8, 12, 16] {
            let mut st = 0x7777u32;
            let x: Vec<u16> = (0..m * k)
                .map(|_| {
                    half::bf16::from_f32(((lcg(&mut st) >> 16) as i32 - 32_768) as f32 / 32_768.0)
                        .to_bits()
                })
                .collect();
            let x = MxArray::from_bfloat16(&x, &[m, k]).expect("x");
            let batch = |ring: &[(MxArray, MxArray, MxArray)]| -> f64 {
                let mut handles: Vec<_> = ring
                    .iter()
                    .map(|(w, sc, bi)| qmm(&x, w, sc, bi, format))
                    .collect();
                let mut error = [0i8; 512];
                let started = Instant::now();
                let ok = unsafe {
                    mlx_sys::mlx_eval_with_error(
                        handles.as_mut_ptr(),
                        handles.len(),
                        error.as_mut_ptr(),
                        error.len(),
                    )
                };
                let elapsed = started.elapsed().as_secs_f64();
                assert!(ok, "affine qmm eval failed");
                for h in handles {
                    drop_array(h);
                }
                elapsed / RING as f64
            };
            for _ in 0..WARMUP {
                batch(&ring);
            }
            let mut samples: Vec<f64> = (0..REPS).map(|_| batch(&ring)).collect();
            let ms = median_ms(&mut samples);
            println!(
                "  M={m:>2}: {ms:.4} ms/matmul  {:.0} GB/s",
                bytes / (ms / 1e3) / 1e9
            );
        }
    }
}
