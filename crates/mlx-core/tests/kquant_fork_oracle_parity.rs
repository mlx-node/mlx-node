//! The bridge's K-quant ops (`mlx_kquant*.cpp`) against MLX's own K-quant path,
//! bit for bit, on Metal and on the CPU.
//!
//! The oracle is reachable only while `crates/mlx-sys/mlx` is the MLX fork
//! pinned at 053e43fec; `mlx_test_fork_kquant_*` call `mlx::core` with the
//! mode string. When the pin moves to upstream MLX this file becomes a
//! reference-dequantize test.
//!
//! Run: cargo test -p mlx-core --release --test kquant_fork_oracle_parity

use std::ffi::CString;

use mlx_core::array::{DType, MxArray};

struct KQuant {
    mode: &'static str,
    bits: i32,
    group_size: i32,
    scales_signed: bool,
    // Columns per 256 decoded values.
    weight_cols: i64,
    scales_cols: i64,
    biases_cols: i64,
}

const KQUANTS: [KQuant; 7] = [
    KQuant {
        mode: "q3k",
        bits: 3,
        group_size: 16,
        scales_signed: true,
        weight_cols: 24,
        scales_cols: 16,
        biases_cols: 1,
    },
    KQuant {
        mode: "q6k",
        bits: 6,
        group_size: 16,
        scales_signed: true,
        weight_cols: 48,
        scales_cols: 16,
        biases_cols: 1,
    },
    KQuant {
        mode: "q4k",
        bits: 4,
        group_size: 32,
        scales_signed: false,
        weight_cols: 32,
        scales_cols: 16,
        biases_cols: 2,
    },
    KQuant {
        mode: "q5k",
        bits: 5,
        group_size: 32,
        scales_signed: false,
        weight_cols: 40,
        scales_cols: 16,
        biases_cols: 2,
    },
    KQuant {
        mode: "iq4nl",
        bits: 4,
        group_size: 32,
        scales_signed: true,
        weight_cols: 32,
        scales_cols: 8,
        biases_cols: 8,
    },
    KQuant {
        mode: "iq4xs",
        bits: 4,
        group_size: 32,
        scales_signed: true,
        weight_cols: 32,
        scales_cols: 8,
        biases_cols: 1,
    },
    KQuant {
        mode: "iq3s",
        bits: 8,
        group_size: 32,
        scales_signed: true,
        weight_cols: 64,
        scales_cols: 8,
        biases_cols: 1,
    },
];

const DTYPES: [DType; 3] = [DType::Float32, DType::Float16, DType::BFloat16];

/// Row counts that cross every dispatch boundary: qmv, qmv_wide nv2..8, the
/// bfloat16 M = 8 sg8 kernel, the batch limit, split-K qmm, and NAX qmm.
const MS: [i64; 15] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 17, 24, 33, 64, 512];

fn select(gpu: bool) -> bool {
    let code = i32::from(gpu);
    // SAFETY: plain global setters in the MLX FFI; tests run single-threaded.
    unsafe {
        mlx_sys::mlx_set_default_device(code);
        if mlx_sys::mlx_default_device() != code {
            return false;
        }
        let stream = mlx_sys::mlx_default_stream(code);
        mlx_sys::mlx_set_default_stream(stream);
    }
    true
}

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

const HALF_SCALES: [u16; 6] = [0x3000, 0x2C00, 0x3400, 0x2E66, 0x3155, 0x2800];

struct Weights {
    w: MxArray,
    scales: MxArray,
    biases: MxArray,
}

/// `leading` precedes the packed axis, which expands to `packed` values.
fn weights(kq: &KQuant, leading: &[i64], packed: i64, seed: u32) -> Weights {
    assert_eq!(packed % 256, 0);
    let rows: i64 = leading.iter().product();
    let supers = packed / 256;
    let shape = |cols: i64| {
        let mut s = leading.to_vec();
        s.push(cols * supers);
        s
    };
    let mut st = seed;
    let w: Vec<u32> = (0..rows * kq.weight_cols * supers)
        .map(|_| lcg(&mut st))
        .collect();
    let scales_len = (rows * kq.scales_cols * supers) as usize;
    let scales = if kq.scales_signed {
        let v: Vec<i8> = (0..scales_len)
            .map(|_| (lcg(&mut st) % 17) as i8 - 8)
            .collect();
        MxArray::from_int8(&v, &shape(kq.scales_cols))
    } else {
        let v: Vec<u8> = (0..scales_len).map(|_| (lcg(&mut st) % 64) as u8).collect();
        MxArray::from_uint8(&v, &shape(kq.scales_cols))
    }
    .expect("scales");
    let biases: Vec<u16> = (0..(rows * kq.biases_cols * supers) as usize)
        .map(|i| HALF_SCALES[(i + seed as usize) % HALF_SCALES.len()])
        .collect();
    Weights {
        w: MxArray::from_uint32(&w, &shape(kq.weight_cols)).expect("w"),
        scales,
        biases: MxArray::from_float16(&biases, &shape(kq.biases_cols)).expect("biases"),
    }
}

/// 8-bit integers over 128: exact in all three activation dtypes.
fn activation(shape: &[i64], seed: u32, dtype: DType) -> MxArray {
    let n: i64 = shape.iter().product();
    let mut st = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let values: Vec<f32> = (0..n)
        .map(|_| f32::from((lcg(&mut st) >> 24) as i8) / 128.0)
        .collect();
    match dtype {
        DType::Float32 => MxArray::from_float32(&values, shape),
        DType::Float16 => {
            let bits: Vec<u16> = values
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect();
            MxArray::from_float16(&bits, shape)
        }
        DType::BFloat16 => {
            let bits: Vec<u16> = values.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            MxArray::from_bfloat16(&bits, shape)
        }
        other => panic!("unsupported activation dtype {other:?}"),
    }
    .expect("activation")
}

fn eval(handle: *mut mlx_sys::mlx_array) -> Result<(), String> {
    let mut buf = [0i8; 512];
    let mut h = handle;
    // SAFETY: `handle` is live; `buf` is the error sink.
    let ok = unsafe { mlx_sys::mlx_eval_with_error(&mut h, 1, buf.as_mut_ptr(), buf.len()) };
    if ok {
        return Ok(());
    }
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    Err(String::from_utf8_lossy(&bytes).into_owned())
}

/// Raw output bits; consumes the handle.
fn read_bits(what: &str, h: *mut mlx_sys::mlx_array) -> (DType, Vec<u32>) {
    assert!(!h.is_null(), "{what}: rejected at construction");
    if let Err(e) = eval(h) {
        // SAFETY: owned handle.
        unsafe { mlx_sys::mlx_array_delete(h) };
        panic!("{what}: {e}");
    }
    // SAFETY: `h` is live and evaluated.
    let (len, code) = unsafe { (mlx_sys::mlx_array_size(h), mlx_sys::mlx_array_dtype(h)) };
    let out = if code == DType::Float32 as i32 {
        let mut v = vec![0f32; len];
        // SAFETY: `v` holds `len` elements.
        assert!(unsafe { mlx_sys::mlx_array_to_float32(h, v.as_mut_ptr(), len) });
        (DType::Float32, v.iter().map(|x| x.to_bits()).collect())
    } else {
        let mut v = vec![0u16; len];
        // SAFETY: `v` holds `len` elements.
        assert!(unsafe { mlx_sys::mlx_array_to_uint16(h, v.as_mut_ptr(), len) });
        let dtype = if code == DType::Float16 as i32 {
            DType::Float16
        } else {
            DType::BFloat16
        };
        (dtype, v.into_iter().map(u32::from).collect())
    };
    // SAFETY: owned handle, not used afterwards.
    unsafe { mlx_sys::mlx_array_delete(h) };
    out
}

fn ptr(a: Option<&MxArray>) -> *mut mlx_sys::mlx_array {
    a.map_or(std::ptr::null_mut(), |a| a.as_raw_ptr())
}

#[derive(Default)]
struct Tally {
    cases: usize,
    nonzero: usize,
}

/// Both handles must evaluate to the same dtype and the same bits.
fn assert_identical(
    tally: &mut Tally,
    what: &str,
    ours: *mut mlx_sys::mlx_array,
    fork: *mut mlx_sys::mlx_array,
) {
    let (ours_dtype, ours) = read_bits(&format!("{what} ours"), ours);
    let (fork_dtype, fork) = read_bits(&format!("{what} fork"), fork);
    assert_eq!(ours_dtype, fork_dtype, "{what}: dtype differs");
    assert_eq!(ours.len(), fork.len(), "{what}: length differs");
    if let Some(i) = (0..ours.len()).find(|&i| ours[i] != fork[i]) {
        panic!(
            "{what}: first difference at {i}: ours {:#x}, fork {:#x} ({} of {} differ)",
            ours[i],
            fork[i],
            (0..ours.len()).filter(|&j| ours[j] != fork[j]).count(),
            ours.len()
        );
    }
    tally.cases += 1;
    if ours.iter().any(|&b| b & 0x7fff_ffff != 0) {
        tally.nonzero += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn qmm_pair(tally: &mut Tally, what: &str, x: &MxArray, w: &Weights, transpose: bool, kq: &KQuant) {
    let mode = CString::new(kq.mode).unwrap();
    // SAFETY: every handle outlives the calls.
    let (ours, fork) = unsafe {
        (
            mlx_sys::mlx_quantized_matmul(
                x.as_raw_ptr(),
                w.w.as_raw_ptr(),
                w.scales.as_raw_ptr(),
                w.biases.as_raw_ptr(),
                transpose,
                kq.group_size,
                kq.bits,
                mode.as_ptr(),
            ),
            mlx_sys::mlx_test_fork_kquant_quantized_matmul(
                x.as_raw_ptr(),
                w.w.as_raw_ptr(),
                w.scales.as_raw_ptr(),
                w.biases.as_raw_ptr(),
                transpose,
                kq.group_size,
                kq.bits,
                mode.as_ptr(),
            ),
        )
    };
    assert_identical(tally, what, ours, fork);
}

#[allow(clippy::too_many_arguments)]
fn gather_pair(
    tally: &mut Tally,
    what: &str,
    x: &MxArray,
    w: &Weights,
    lhs: Option<&MxArray>,
    rhs: &MxArray,
    transpose: bool,
    sorted: bool,
    kq: &KQuant,
) {
    let mode = CString::new(kq.mode).unwrap();
    // SAFETY: every handle outlives the calls; null lhs means "derive from x".
    let (ours, fork) = unsafe {
        (
            mlx_sys::mlx_gather_qmm(
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
            ),
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
            ),
        )
    };
    assert_identical(tally, what, ours, fork);
}

fn dequantize_pair(tally: &mut Tally, what: &str, w: &Weights, dtype: DType, kq: &KQuant) {
    let mode = CString::new(kq.mode).unwrap();
    // SAFETY: every handle outlives the calls.
    let (ours, fork) = unsafe {
        (
            mlx_sys::mlx_dequantize(
                w.w.as_raw_ptr(),
                w.scales.as_raw_ptr(),
                w.biases.as_raw_ptr(),
                kq.group_size,
                kq.bits,
                dtype as i32,
                mode.as_ptr(),
            ),
            mlx_sys::mlx_test_fork_kquant_dequantize(
                w.w.as_raw_ptr(),
                w.scales.as_raw_ptr(),
                w.biases.as_raw_ptr(),
                kq.group_size,
                kq.bits,
                dtype as i32,
                mode.as_ptr(),
            ),
        )
    };
    assert_identical(tally, what, ours, fork);
}

fn indices(values: &[u32]) -> MxArray {
    MxArray::from_uint32(values, &[values.len() as i64]).expect("indices")
}

/// The portable prefill tile is a separate kernel layered on top of the op and
/// is gated by its own tests; keep it out of this comparison.
fn disable_portable() {
    // SAFETY: set before any MLX work in this single-threaded test binary.
    unsafe { std::env::set_var("MLX_PORTABLE_KQUANT", "0") };
}

fn run_matmuls(tally: &mut Tally, gpu: bool, ms: &[i64]) {
    let device = if gpu { "gpu" } else { "cpu" };
    for (ki, kq) in KQUANTS.iter().enumerate() {
        let seed = 0x1000 + ki as u32;
        // Transposed: (N, K) = aligned and fast-qmv, then unaligned N with
        // K % 512 != 0, then a batched weight.
        let wide = weights(kq, &[2048], 1024, seed);
        let odd = weights(kq, &[136], 768, seed + 1);
        let batched = weights(kq, &[2, 136], 768, seed + 2);
        let batched_wide = weights(kq, &[2, 256], 1024, seed + 5);
        // Untransposed: the packed axis is the output; K < 1024 is qvm,
        // K >= 1024 is qvm_split_k, M >= 4 is qmm_n.
        let n_short = weights(kq, &[256], 512, seed + 3);
        let n_deep = weights(kq, &[1024], 512, seed + 4);
        for dtype in DTYPES {
            for &m in ms {
                if gpu || m <= 33 {
                    let x = activation(&[m, 1024], seed + m as u32, dtype);
                    qmm_pair(
                        tally,
                        &format!("{device} {} {dtype:?} t M={m} N=2048 K=1024", kq.mode),
                        &x,
                        &wide,
                        true,
                        kq,
                    );
                }
                let x = activation(&[m, 768], seed + 7 + m as u32, dtype);
                qmm_pair(
                    tally,
                    &format!("{device} {} {dtype:?} t M={m} N=136 K=768", kq.mode),
                    &x,
                    &odd,
                    true,
                    kq,
                );
                if m <= 64 {
                    let x = activation(&[2, m, 768], seed + 11 + m as u32, dtype);
                    qmm_pair(
                        tally,
                        &format!("{device} {} {dtype:?} t batched M={m}", kq.mode),
                        &x,
                        &batched,
                        true,
                        kq,
                    );
                    let x = activation(&[2, m, 1024], seed + 23 + m as u32, dtype);
                    qmm_pair(
                        tally,
                        &format!("{device} {} {dtype:?} t batched-aligned M={m}", kq.mode),
                        &x,
                        &batched_wide,
                        true,
                        kq,
                    );
                    let x = activation(&[3, m, 768], seed + 13 + m as u32, dtype);
                    qmm_pair(
                        tally,
                        &format!("{device} {} {dtype:?} t x-batched M={m}", kq.mode),
                        &x,
                        &odd,
                        true,
                        kq,
                    );
                }
                if m <= 64 {
                    let x = activation(&[m, 256], seed + 17 + m as u32, dtype);
                    qmm_pair(
                        tally,
                        &format!("{device} {} {dtype:?} n M={m} K=256", kq.mode),
                        &x,
                        &n_short,
                        false,
                        kq,
                    );
                    let x = activation(&[m, 1024], seed + 19 + m as u32, dtype);
                    qmm_pair(
                        tally,
                        &format!("{device} {} {dtype:?} n M={m} K=1024", kq.mode),
                        &x,
                        &n_deep,
                        false,
                        kq,
                    );
                }
            }
        }
    }
}

fn run_gathers(tally: &mut Tally, gpu: bool) {
    let device = if gpu { "gpu" } else { "cpu" };
    const E: i64 = 8;
    for (ki, kq) in KQUANTS.iter().enumerate() {
        let seed = 0x2000 + ki as u32;
        let experts_t = weights(kq, &[E, 128], 512, seed);
        let experts_n = weights(kq, &[E, 256], 512, seed + 1);
        let experts_odd = weights(kq, &[E, 136], 768, seed + 2);
        let mut st = seed;
        let random: Vec<u32> = (0..32).map(|_| lcg(&mut st) % E as u32).collect();
        let mut sorted_ids = random.clone();
        sorted_ids.sort_unstable();
        let lhs: Vec<u32> = (0..6).map(|_| lcg(&mut st) % 4).collect();
        let rhs6: Vec<u32> = (0..6).map(|_| lcg(&mut st) % E as u32).collect();
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
                    tally,
                    &format!("{device} {} {dtype:?} gather t {tag}", kq.mode),
                    &x,
                    &experts_t,
                    None,
                    &indices(ids),
                    true,
                    sorted,
                    kq,
                );
            }
            let x = activation(&[32, 1, 768], seed + 4, dtype);
            gather_pair(
                tally,
                &format!("{device} {} {dtype:?} gather t odd unsorted", kq.mode),
                &x,
                &experts_odd,
                None,
                &indices(&random),
                true,
                false,
                kq,
            );
            gather_pair(
                tally,
                &format!("{device} {} {dtype:?} gather t odd sorted", kq.mode),
                &x,
                &experts_odd,
                None,
                &indices(&sorted_ids),
                true,
                true,
                kq,
            );
            let xn = activation(&[32, 1, 256], seed + 5, dtype);
            gather_pair(
                tally,
                &format!("{device} {} {dtype:?} gather n sorted", kq.mode),
                &xn,
                &experts_n,
                None,
                &indices(&sorted_ids),
                false,
                true,
                kq,
            );
            gather_pair(
                tally,
                &format!("{device} {} {dtype:?} gather n unsorted", kq.mode),
                &xn,
                &experts_n,
                None,
                &indices(&random),
                false,
                false,
                kq,
            );
            // Both index sets: vector and matrix kernels on either layout.
            for m in [1i64, 4, 33] {
                let x = activation(&[4, m, 512], seed + 7 + m as u32, dtype);
                gather_pair(
                    tally,
                    &format!("{device} {} {dtype:?} gather t lhs M={m}", kq.mode),
                    &x,
                    &experts_t,
                    Some(&indices(&lhs)),
                    &indices(&rhs6),
                    true,
                    false,
                    kq,
                );
                let x = activation(&[4, m, 256], seed + 9 + m as u32, dtype);
                gather_pair(
                    tally,
                    &format!("{device} {} {dtype:?} gather n lhs M={m}", kq.mode),
                    &x,
                    &experts_n,
                    Some(&indices(&lhs)),
                    &indices(&rhs6),
                    false,
                    false,
                    kq,
                );
            }
        }
    }
}

fn run_dequantize(tally: &mut Tally, gpu: bool) {
    let device = if gpu { "gpu" } else { "cpu" };
    for (ki, kq) in KQUANTS.iter().enumerate() {
        let flat = weights(kq, &[96], 1024, 0x3000 + ki as u32);
        let stacked = weights(kq, &[3, 40], 512, 0x3100 + ki as u32);
        for dtype in DTYPES {
            dequantize_pair(
                tally,
                &format!("{device} {} {dtype:?} dequantize 2-D", kq.mode),
                &flat,
                dtype,
                kq,
            );
            dequantize_pair(
                tally,
                &format!("{device} {} {dtype:?} dequantize 3-D", kq.mode),
                &stacked,
                dtype,
                kq,
            );
        }
    }
}

fn report(label: &str, tally: &Tally) {
    println!(
        "{label}: {} cases bit-identical ({} with a nonzero output)",
        tally.cases, tally.nonzero
    );
    assert!(tally.cases > 0);
    assert_eq!(
        tally.nonzero, tally.cases,
        "{label}: a degenerate all-zero case proves nothing"
    );
}

#[test]
fn metal_matmul_matches_fork_bitwise() {
    disable_portable();
    if !select(true) {
        println!("no Metal device; skipped");
        return;
    }
    let mut tally = Tally::default();
    run_matmuls(&mut tally, true, &MS);
    report("metal quantized_matmul", &tally);
}

#[test]
fn metal_gather_and_dequantize_match_fork_bitwise() {
    disable_portable();
    if !select(true) {
        println!("no Metal device; skipped");
        return;
    }
    let mut gather = Tally::default();
    run_gathers(&mut gather, true);
    report("metal gather_qmm", &gather);
    let mut deq = Tally::default();
    run_dequantize(&mut deq, true);
    report("metal dequantize", &deq);
}

#[test]
fn cpu_matches_fork_bitwise() {
    disable_portable();
    assert!(select(false));
    let mut mm = Tally::default();
    run_matmuls(&mut mm, false, &[1, 2, 8, 9, 33]);
    report("cpu quantized_matmul", &mm);
    let mut gather = Tally::default();
    run_gathers(&mut gather, false);
    report("cpu gather_qmm", &gather);
    let mut deq = Tally::default();
    run_dequantize(&mut deq, false);
    report("cpu dequantize", &deq);
}
