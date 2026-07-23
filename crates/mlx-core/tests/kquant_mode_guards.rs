//! K-quant dispatch guards: what must throw, what must work, what must not
//! regress.
//!
//! In-tree port of the fail-loud smoke that shipped with the Phase 2 work. The
//! numeric gate lives in `kquant_ggml_parity.rs`; this file guards the shape of
//! the support matrix around it:
//!
//! * `quantize` throws for every K-quant mode on every device — these formats
//!   are consumed, never produced;
//! * `dequantize` / `quantized_matmul` / `gather_qmm` work on the CPU, and on
//!   Metal they run and agree with the CPU reference;
//! * the validators reject malformed K-quant tensors, so the throws above prove
//!   something about the backends rather than about a missing argument check;
//! * the pre-existing affine / mxfp4 modes still work through the widened
//!   validator.
//!
//! `qqmm` is in the C++ smoke but is not reachable from here: mlx-sys exposes no
//! `mlx_qqmm` binding, so no Rust caller can construct that op.
//!
//! Run:
//!   cargo test -p mlx-core --release --test kquant_mode_guards

use std::ffi::CString;

use mlx_core::array::{DType, MxArray};

/// K = 256, N = 128 — one super-block per row.
const K: i64 = 256;
const N: i64 = 128;

struct KQuant {
    mode: &'static str,
    bits: i32,
    group_size: i32,
    scales_signed: bool,
    weight_cols: i64,
    scales_cols: i64,
    biases_cols: i64,
}

const KQUANTS: [KQuant; 3] = [
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
];

/// 0 = CPU, 1 = GPU, matching `to_device_helper` in
/// `crates/mlx-sys/src/mlx_stream.cpp:28`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Device {
    Cpu,
    Gpu,
}

impl Device {
    fn code(self) -> i32 {
        match self {
            Self::Cpu => 0,
            Self::Gpu => 1,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

/// Pin MLX's process-wide default device and stream. `.cargo/config.toml` pins
/// `RUST_TEST_THREADS=1`, so mutating the global here is safe.
/// Returns false when the device is unavailable (e.g. no Metal).
fn select(device: Device) -> bool {
    // SAFETY: plain global setters in the MLX FFI; both catch internally.
    unsafe {
        mlx_sys::mlx_set_default_device(device.code());
        if mlx_sys::mlx_default_device() != device.code() {
            return false;
        }
        let stream = mlx_sys::mlx_default_stream(device.code());
        mlx_sys::mlx_set_default_stream(stream);
    }
    true
}

fn zeros(shape: &[i64], dtype: DType) -> MxArray {
    MxArray::zeros(shape, Some(dtype)).expect("zeros")
}

/// Force a lazy graph to run and report the error text if it throws.
fn eval(handle: *mut mlx_sys::mlx_array) -> Result<(), String> {
    let mut buf = [0i8; 512];
    let mut h = handle;
    // SAFETY: `handle` is a live array; `buf` is a 512-byte sink for the error.
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

fn drop_array(handle: *mut mlx_sys::mlx_array) {
    if !handle.is_null() {
        // SAFETY: the handle is owned here and not referenced afterwards.
        unsafe { mlx_sys::mlx_array_delete(handle) };
    }
}

/// Result of one op attempt: either it produced values, or it was rejected —
/// at construction (null handle) or at eval (error text).
enum Attempt {
    Ok(Vec<i64>),
    Rejected(String),
}

fn finish(handle: *mut mlx_sys::mlx_array) -> Attempt {
    if handle.is_null() {
        return Attempt::Rejected("rejected at construction".to_string());
    }
    if let Err(e) = eval(handle) {
        drop_array(handle);
        return Attempt::Rejected(e);
    }
    // SAFETY: `handle` is a live, evaluated array.
    let ndim = unsafe { mlx_sys::mlx_array_ndim(handle) };
    let mut shape = vec![0i64; ndim];
    // SAFETY: `shape` has room for `ndim` dimensions.
    unsafe { mlx_sys::mlx_array_shape(handle, shape.as_mut_ptr()) };
    drop_array(handle);
    Attempt::Ok(shape)
}

fn expect_rejected(what: &str, attempt: Attempt) {
    match attempt {
        Attempt::Rejected(why) => println!("  rejects  {what}  ->  {why}"),
        Attempt::Ok(shape) => panic!("{what} was accepted (shape {shape:?}); it must be rejected"),
    }
}

fn expect_shape(what: &str, want: &[i64], attempt: Attempt) {
    match attempt {
        Attempt::Ok(shape) => {
            assert_eq!(shape, want, "{what} produced the wrong shape");
            println!("  works    {what}");
        }
        Attempt::Rejected(why) => panic!("{what} was rejected: {why}"),
    }
}

/// Evaluate an op result and read it back element by element as float32.
///
/// Element-wise rather than `mlx_array_to_float32`, which materialises through
/// `add(arr, zeros)` — the same reason `kquant_ggml_parity.rs` reads this way.
fn expect_f32(what: &str, handle: *mut mlx_sys::mlx_array) -> Vec<f32> {
    assert!(!handle.is_null(), "{what} was rejected at construction");
    if let Err(why) = eval(handle) {
        drop_array(handle);
        panic!("{what} was rejected: {why}");
    }
    // SAFETY: `handle` is a live, evaluated array.
    let len = unsafe { mlx_sys::mlx_array_size(handle) };
    let mut out = vec![0f32; len];
    for (i, slot) in out.iter_mut().enumerate() {
        // SAFETY: i < len == array size.
        let ok = unsafe { mlx_sys::mlx_array_item_at_float32(handle, i, slot) };
        assert!(ok, "{what}: mlx_array_item_at_float32 failed at index {i}");
    }
    drop_array(handle);
    out
}

// ---------------------------------------------------------------------------
// op wrappers
// ---------------------------------------------------------------------------

fn kquant_weights(kq: &KQuant) -> (MxArray, MxArray, MxArray) {
    let scales_dtype = if kq.scales_signed {
        DType::Int8
    } else {
        DType::Uint8
    };
    (
        zeros(&[N, kq.weight_cols], DType::Uint32),
        zeros(&[N, kq.scales_cols], scales_dtype),
        zeros(&[N, kq.biases_cols], DType::Float16),
    )
}

/// Deterministic pseudo-random fill. A plain LCG so both devices see the same
/// bytes and a failure reproduces.
fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// float16 bit patterns for the super-block scale `d` (and `dmin`): small
/// powers of two plus two odd mantissas, so `d * sc` is not trivially exact.
const HALF_SCALES: [u16; 6] = [0x3000, 0x2C00, 0x3400, 0x2E66, 0x3155, 0x2800];

/// K-quant tensors with non-trivial codes and both scale levels populated.
/// `leading` is everything before the packed dimension, so `&[N]` builds a
/// matmul weight and `&[4, N]` an expert stack.
///
/// All three arrays hold a fixed number of entries per 256-value super-block,
/// so the `KQUANTS` column counts — which are stated for `K` — scale linearly.
fn filled_kquant_weights(kq: &KQuant, leading: &[i64], k: i64) -> (MxArray, MxArray, MxArray) {
    assert_eq!(k % K, 0, "k must be a whole number of super-blocks");
    let rows: i64 = leading.iter().product();
    let supers = k / K;
    let (weight_cols, scales_cols, biases_cols) = (
        kq.weight_cols * supers,
        kq.scales_cols * supers,
        kq.biases_cols * supers,
    );
    let shape = |cols: i64| {
        let mut s = leading.to_vec();
        s.push(cols);
        s
    };

    let mut st = 0x5eed_1234u32;
    let weight: Vec<u32> = (0..rows * weight_cols).map(|_| lcg(&mut st)).collect();
    let scales_len = (rows * scales_cols) as usize;
    let scales = if kq.scales_signed {
        // q6k sub-scales are signed.
        let v: Vec<i8> = (0..scales_len)
            .map(|_| (lcg(&mut st) % 17) as i8 - 8)
            .collect();
        MxArray::from_int8(&v, &shape(scales_cols))
    } else {
        // q4k/q5k interleave (sc, m), both 6-bit unsigned.
        let v: Vec<u8> = (0..scales_len).map(|_| (lcg(&mut st) % 64) as u8).collect();
        MxArray::from_uint8(&v, &shape(scales_cols))
    }
    .expect("scales array");
    let biases: Vec<u16> = (0..(rows * biases_cols) as usize)
        .map(|i| HALF_SCALES[i % HALF_SCALES.len()])
        .collect();
    (
        MxArray::from_uint32(&weight, &shape(weight_cols)).expect("weight array"),
        scales,
        MxArray::from_float16(&biases, &shape(biases_cols)).expect("biases array"),
    )
}

/// A deterministic float32 activation in [-1, 1).
fn ramp(shape: &[i64], seed: u32) -> MxArray {
    let n: i64 = shape.iter().product();
    let mut st = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let v: Vec<f32> = (0..n)
        .map(|_| (lcg(&mut st) >> 8) as f32 / 8_388_608.0 - 1.0)
        .collect();
    MxArray::from_float32(&v, shape).expect("activation array")
}

/// The same activation as `ramp`, stored as bfloat16 — the top 16 bits of the
/// float32 pattern. Both devices then see identical bytes.
fn ramp_bf16(shape: &[i64], seed: u32) -> MxArray {
    let n: i64 = shape.iter().product();
    let mut st = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let v: Vec<u16> = (0..n)
        .map(|_| {
            let f = (lcg(&mut st) >> 8) as f32 / 8_388_608.0 - 1.0;
            (f.to_bits() >> 16) as u16
        })
        .collect();
    MxArray::from_bfloat16(&v, shape).expect("activation array")
}

/// Build the same op on the CPU and on the GPU and compare the results.
/// `tol` is relative to the largest CPU magnitude; 0 demands bit equality.
fn compare_devices<F>(what: &str, tol: f32, build: F)
where
    F: Fn() -> *mut mlx_sys::mlx_array,
{
    assert!(select(Device::Cpu), "CPU device is always available");
    let cpu = expect_f32(&format!("{what} cpu"), build());
    assert!(select(Device::Gpu), "the GPU device went away mid-test");
    let gpu = expect_f32(&format!("{what} gpu"), build());

    assert_eq!(cpu.len(), gpu.len(), "{what}: output lengths differ");
    let scale = cpu.iter().fold(1f32, |m, v| m.max(v.abs()));
    let limit = tol * scale;
    let mut worst = 0f32;
    let mut worst_at = 0usize;
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let d = (a - b).abs();
        if d.is_nan() || d > worst {
            worst = d;
            worst_at = i;
        }
    }
    assert!(
        worst <= limit,
        "{what}: Metal and the CPU differ by {worst} at index {worst_at} \
         (cpu {}, gpu {}, limit {limit})",
        cpu[worst_at],
        gpu[worst_at],
    );
    println!("  matches  {what}  (max |cpu - gpu| {worst} <= {limit})");
}

fn quantize(x: &MxArray, group_size: i32, bits: i32, mode: &str) -> Attempt {
    let mode = CString::new(mode).expect("mode");
    let mut q = std::ptr::null_mut();
    let mut s = std::ptr::null_mut();
    let mut b = std::ptr::null_mut();
    // SAFETY: `x` outlives the call; the three out-params are written only on
    // success.
    let ok = unsafe {
        mlx_sys::mlx_quantize(
            x.as_raw_ptr(),
            group_size,
            bits,
            mode.as_ptr(),
            &mut q,
            &mut s,
            &mut b,
        )
    };
    if !ok {
        return Attempt::Rejected("rejected at construction".to_string());
    }
    let out = finish(q);
    drop_array(s);
    drop_array(b);
    out
}

fn dequantize_handle(
    w: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> *mut mlx_sys::mlx_array {
    let mode = CString::new(mode).expect("mode");
    // SAFETY: every handle outlives the call; a null `biases` is the documented
    // "no biases" encoding.
    unsafe {
        mlx_sys::mlx_dequantize(
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.map_or(std::ptr::null_mut(), |b| b.as_raw_ptr()),
            group_size,
            bits,
            0,
            mode.as_ptr(),
        )
    }
}

fn dequantize(
    w: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> Attempt {
    finish(dequantize_handle(w, scales, biases, group_size, bits, mode))
}

#[allow(clippy::too_many_arguments)]
fn quantized_matmul_handle(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> *mut mlx_sys::mlx_array {
    let mode = CString::new(mode).expect("mode");
    // SAFETY: every handle outlives the call.
    unsafe {
        mlx_sys::mlx_quantized_matmul(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.map_or(std::ptr::null_mut(), |b| b.as_raw_ptr()),
            true,
            group_size,
            bits,
            mode.as_ptr(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn quantized_matmul(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> Attempt {
    finish(quantized_matmul_handle(
        x, w, scales, biases, group_size, bits, mode,
    ))
}

#[allow(clippy::too_many_arguments)]
fn gather_qmm_handle(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    lhs: &MxArray,
    rhs: &MxArray,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> *mut mlx_sys::mlx_array {
    let mode = CString::new(mode).expect("mode");
    // SAFETY: every handle outlives the call.
    unsafe {
        mlx_sys::mlx_gather_qmm(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.map_or(std::ptr::null_mut(), |b| b.as_raw_ptr()),
            lhs.as_raw_ptr(),
            rhs.as_raw_ptr(),
            true,
            group_size,
            bits,
            mode.as_ptr(),
            false,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn gather_qmm(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    lhs: &MxArray,
    rhs: &MxArray,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> Attempt {
    finish(gather_qmm_handle(
        x, w, scales, biases, lhs, rhs, group_size, bits, mode,
    ))
}

// ---------------------------------------------------------------------------
// guards
// ---------------------------------------------------------------------------

/// These formats are only ever imported. `quantize` must refuse to produce them
/// on every device, so nothing can silently write a K-quant tensor.
#[test]
fn quantize_rejects_every_kquant_mode_on_every_device() {
    for device in [Device::Cpu, Device::Gpu] {
        if !select(device) {
            eprintln!("skipping {}: device unavailable", device.label());
            continue;
        }
        for kq in &KQUANTS {
            let x = zeros(&[N, K], DType::BFloat16);
            expect_rejected(
                &format!("quantize {} {}", kq.mode, device.label()),
                quantize(&x, kq.group_size, kq.bits, kq.mode),
            );
        }
    }
}

/// The CPU implements the K-quant decode, matmul and gather. If any of these
/// starts throwing, the parity gate next door stops measuring anything.
#[test]
fn cpu_runs_dequantize_matmul_and_gather_for_every_kquant_mode() {
    assert!(select(Device::Cpu), "CPU device is always available");
    for kq in &KQUANTS {
        let (w, scales, biases) = kquant_weights(kq);
        expect_shape(
            &format!("dequantize {} cpu", kq.mode),
            &[N, K],
            dequantize(&w, &scales, Some(&biases), kq.group_size, kq.bits, kq.mode),
        );
        for m in [1i64, 8] {
            let x = zeros(&[m, K], DType::BFloat16);
            expect_shape(
                &format!("quantized_matmul M={m} {} cpu", kq.mode),
                &[m, N],
                quantized_matmul(
                    &x,
                    &w,
                    &scales,
                    Some(&biases),
                    kq.group_size,
                    kq.bits,
                    kq.mode,
                ),
            );
        }
        let scales_dtype = if kq.scales_signed {
            DType::Int8
        } else {
            DType::Uint8
        };
        let we = zeros(&[4, N, kq.weight_cols], DType::Uint32);
        let se = zeros(&[4, N, kq.scales_cols], scales_dtype);
        let be = zeros(&[4, N, kq.biases_cols], DType::Float16);
        let x = zeros(&[2, 8, K], DType::BFloat16);
        let lhs = zeros(&[2], DType::Uint32);
        let rhs = zeros(&[2], DType::Uint32);
        expect_shape(
            &format!("gather_qmm {} cpu", kq.mode),
            &[2, 8, N],
            gather_qmm(
                &x,
                &we,
                &se,
                Some(&be),
                &lhs,
                &rhs,
                kq.group_size,
                kq.bits,
                kq.mode,
            ),
        );
    }
}

/// Metal decodes the two scale levels itself, so the K-quant ops must run
/// there and land on the CPU reference. Zero-filled tensors would only prove a
/// pipeline was found, so the weights, sub-scales and super-scales below are
/// all non-trivial and the outputs are compared value by value.
///
/// The shapes pick six kernels: `qmv` (M = 1), `qmv_fast` (M = 1, K % 512 == 0),
/// `qmv_wide` (M = 8 on gen 15+), `qmm_t_splitk` (M = 64), `qmm_t` (batched)
/// and `gather_qmv`, plus `dequantize`.
#[test]
fn gpu_runs_every_kquant_op_and_matches_cpu() {
    if !select(Device::Gpu) {
        eprintln!("skipping gpu_runs_every_kquant_op_and_matches_cpu: no GPU device");
        return;
    }
    for kq in &KQUANTS {
        let (w, scales, biases) = filled_kquant_weights(kq, &[N], K);
        // dequantize is a pure per-element decode: no accumulation can reorder,
        // so Metal has to reproduce the CPU bits exactly.
        compare_devices(&format!("dequantize {}", kq.mode), 0.0, || {
            dequantize_handle(&w, &scales, Some(&biases), kq.group_size, kq.bits, kq.mode)
        });

        for m in [1i64, 8, 64] {
            let x = ramp(&[m, K], 7 + m as u32);
            compare_devices(&format!("quantized_matmul M={m} {}", kq.mode), 1e-5, || {
                quantized_matmul_handle(
                    &x,
                    &w,
                    &scales,
                    Some(&biases),
                    kq.group_size,
                    kq.bits,
                    kq.mode,
                )
            });
        }

        // N % 8 == 0 and K % 512 == 0 pick qmv_fast, the kernel a real decode
        // step runs; the K = 256 case above picks the tail-handling qmv.
        let (wf, sf, bf) = filled_kquant_weights(kq, &[N], 2 * K);
        let xf = ramp(&[1, 2 * K], 23);
        compare_devices(
            &format!("quantized_matmul qmv_fast {}", kq.mode),
            1e-5,
            || quantized_matmul_handle(&xf, &wf, &sf, Some(&bf), kq.group_size, kq.bits, kq.mode),
        );

        // Batched (B > 1) skips qmm_splitk and lands on qmm, which is where the
        // NAX gate lives. bfloat16 and K % 64 == 0 satisfy the rest of that
        // gate, so without the mode allowlist this asks for a
        // `<mode>_qmm_t_nax_...` kernel that kquant.metal never instantiates.
        let (wb, sb, bb) = filled_kquant_weights(kq, &[3, N], K);
        let xb = ramp_bf16(&[3, 64, K], 3);
        compare_devices(
            &format!("batched quantized_matmul bf16 {}", kq.mode),
            0.02,
            || quantized_matmul_handle(&xb, &wb, &sb, Some(&bb), kq.group_size, kq.bits, kq.mode),
        );

        let (we, se, be) = filled_kquant_weights(kq, &[4, N], K);
        let x = ramp(&[2, 8, K], 11);
        let lhs = MxArray::from_uint32(&[0, 1], &[2]).expect("lhs indices");
        let rhs = MxArray::from_uint32(&[3, 1], &[2]).expect("rhs indices");
        compare_devices(&format!("gather_qmm {}", kq.mode), 1e-5, || {
            gather_qmm_handle(
                &x,
                &we,
                &se,
                Some(&be),
                &lhs,
                &rhs,
                kq.group_size,
                kq.bits,
                kq.mode,
            )
        });
    }
    // leave the process on the CPU for whatever runs next
    select(Device::Cpu);
}

/// Without these, "the GPU throws" would prove nothing — a validator that
/// accepts anything would let the backend throw for the wrong reason.
#[test]
fn validators_reject_malformed_kquant_tensors() {
    assert!(select(Device::Cpu), "CPU device is always available");
    let x = zeros(&[1, K], DType::BFloat16);

    expect_rejected(
        "q6k with uint8 scales",
        quantized_matmul(
            &x,
            &zeros(&[N, 48], DType::Uint32),
            &zeros(&[N, 16], DType::Uint8),
            Some(&zeros(&[N, 1], DType::Float16)),
            16,
            6,
            "q6k",
        ),
    );
    expect_rejected(
        "q4k with missing biases",
        quantized_matmul(
            &x,
            &zeros(&[N, 32], DType::Uint32),
            &zeros(&[N, 16], DType::Uint8),
            None,
            32,
            4,
            "q4k",
        ),
    );
    expect_rejected(
        "q4k with un-interleaved scales",
        quantized_matmul(
            &x,
            &zeros(&[N, 32], DType::Uint32),
            &zeros(&[N, 8], DType::Uint8),
            Some(&zeros(&[N, 2], DType::Float16)),
            32,
            4,
            "q4k",
        ),
    );
    expect_rejected(
        "q6k with non-default group size",
        quantized_matmul(
            &x,
            &zeros(&[N, 48], DType::Uint32),
            &zeros(&[N, 16], DType::Int8),
            Some(&zeros(&[N, 1], DType::Float16)),
            32,
            6,
            "q6k",
        ),
    );
}

/// The K-quant modes widened a validator every other mode goes through. These
/// are the shapes that were working before and must keep working.
#[test]
fn preexisting_quantization_modes_still_work() {
    for device in [Device::Cpu, Device::Gpu] {
        if !select(device) {
            eprintln!("skipping {}: device unavailable", device.label());
            continue;
        }
        let tail = device.label();
        let w = zeros(&[N, 32], DType::Uint32);
        let s = zeros(&[N, 4], DType::BFloat16);
        let b = zeros(&[N, 4], DType::BFloat16);
        for m in [1i64, 8] {
            let x = zeros(&[m, K], DType::BFloat16);
            expect_shape(
                &format!("affine 4/64 M={m} {tail}"),
                &[m, N],
                quantized_matmul(&x, &w, &s, Some(&b), 64, 4, "affine"),
            );
        }
        expect_shape(
            &format!("mxfp4 qmv {tail}"),
            &[1, N],
            quantized_matmul(
                &zeros(&[1, K], DType::BFloat16),
                &w,
                &zeros(&[N, 8], DType::Uint8),
                None,
                32,
                4,
                "mxfp4",
            ),
        );
        expect_shape(
            &format!("affine gather_qmm {tail}"),
            &[2, 8, N],
            gather_qmm(
                &zeros(&[2, 8, K], DType::BFloat16),
                &zeros(&[4, N, 32], DType::Uint32),
                &zeros(&[4, N, 4], DType::BFloat16),
                Some(&zeros(&[4, N, 4], DType::BFloat16)),
                &zeros(&[2], DType::Uint32),
                &zeros(&[2], DType::Uint32),
                64,
                4,
                "affine",
            ),
        );
        expect_shape(
            &format!("affine batched qmm 3d {tail}"),
            &[3, 8, N],
            quantized_matmul(
                &zeros(&[3, 8, K], DType::BFloat16),
                &zeros(&[3, N, 32], DType::Uint32),
                &zeros(&[3, N, 4], DType::BFloat16),
                Some(&zeros(&[3, N, 4], DType::BFloat16)),
                64,
                4,
                "affine",
            ),
        );
    }
    assert!(select(Device::Cpu), "CPU device is always available");
    let x = zeros(&[N, K], DType::BFloat16);
    expect_shape(
        "affine quantize round-trip cpu",
        &[N, K / 8],
        quantize(&x, 64, 4, "affine"),
    );
}
