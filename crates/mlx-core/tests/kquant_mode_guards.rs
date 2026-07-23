//! K-quant dispatch guards: what must throw, what must work, what must not
//! regress.
//!
//! In-tree port of the fail-loud smoke that shipped with the Phase 2 work. The
//! numeric gate lives in `kquant_ggml_parity.rs`; this file guards the shape of
//! the support matrix around it:
//!
//! * `quantize` throws for every K-quant mode on every device — these formats
//!   are consumed, never produced;
//! * `dequantize` / `quantized_matmul` / `gather_qmm` work on the CPU and throw
//!   on the GPU, because Metal and CUDA have no K-quant kernels yet;
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

fn dequantize(
    w: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> Attempt {
    let mode = CString::new(mode).expect("mode");
    // SAFETY: every handle outlives the call; a null `biases` is the documented
    // "no biases" encoding.
    let handle = unsafe {
        mlx_sys::mlx_dequantize(
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.map_or(std::ptr::null_mut(), |b| b.as_raw_ptr()),
            group_size,
            bits,
            0,
            mode.as_ptr(),
        )
    };
    finish(handle)
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
    let mode = CString::new(mode).expect("mode");
    // SAFETY: every handle outlives the call.
    let handle = unsafe {
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
    };
    finish(handle)
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
    let mode = CString::new(mode).expect("mode");
    // SAFETY: every handle outlives the call.
    let handle = unsafe {
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
    };
    finish(handle)
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

/// Metal and CUDA have no K-quant kernels. They must say so out loud rather
/// than fall through to some other mode's kernel and return wrong numbers.
#[test]
fn gpu_rejects_every_kquant_op() {
    if !select(Device::Gpu) {
        eprintln!("skipping gpu_rejects_every_kquant_op: no GPU device");
        return;
    }
    for kq in &KQUANTS {
        let (w, scales, biases) = kquant_weights(kq);
        expect_rejected(
            &format!("dequantize {} gpu", kq.mode),
            dequantize(&w, &scales, Some(&biases), kq.group_size, kq.bits, kq.mode),
        );
        for m in [1i64, 8] {
            let x = zeros(&[m, K], DType::BFloat16);
            expect_rejected(
                &format!("quantized_matmul M={m} {} gpu", kq.mode),
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
        expect_rejected(
            &format!("gather_qmm {} gpu", kq.mode),
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
