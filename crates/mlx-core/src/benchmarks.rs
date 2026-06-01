//! Microbench harness for the production quantized-`qmv` dispatch path.
//!
//! Exposes a single NAPI function — `quantized_qmv_microbench` — that wraps
//! `mlx::core::quantized_matmul` (the production dispatch, including the
//! small-M `qmv_fast_sg4` fast path) in a warmup + median-of-iters wall-clock
//! timing loop. Measurement-only.
//!
//! Why median (not mean): GPU command-buffer setup latency lands on the first
//! few dispatches even after warmup, and a single 2-3× outlier badly skews
//! the mean. Median makes the harness robust to thermal hiccups and JIT
//! recompilation between cold-cache transitions.

use std::time::Instant;

use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::array::{DType, MxArray};

/// Run `mlx::core::quantized_matmul` through the normal production dispatch path.
pub(crate) fn quantized_qmv_matmul(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> Result<MxArray> {
    use std::ffi::CString;

    let mode_c = CString::new(mode)
        .map_err(|_| Error::from_reason("mode must not contain NUL bytes".to_string()))?;
    let biases_ptr = biases.map_or(std::ptr::null_mut(), |b| b.as_raw_ptr());
    let handle = unsafe {
        sys::mlx_quantized_matmul(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases_ptr,
            true,
            group_size,
            bits,
            mode_c.as_ptr(),
        )
    };
    MxArray::from_handle(handle, "quantized_qmv_matmul")
}

// ---------------------------------------------------------------------
// NAPI-facing microbench harness.
// ---------------------------------------------------------------------

/// Microbench result for the production quantized qmv dispatch path.
#[napi(object, js_name = "QmvQuantizedMicrobenchResult")]
#[derive(Clone, Debug)]
pub struct QmvQuantizedMicrobenchResult {
    /// Median `quantized_matmul` wall-clock per call, in nanoseconds.
    pub median_ns: f64,
    /// A tiny materialized checksum of the final output, used to keep the call live.
    pub checksum: f64,
}

/// Run the production quantized-qmv microbench in the current process.
///
/// To compare `MLX_MTP_SMALL_M_QMV=0` versus `1`, call this from separate
/// processes. MLX caches the env-backed dispatch predicate statically.
#[napi]
pub fn quantized_qmv_microbench(
    k: u32,
    n: u32,
    m: u32,
    group_size: u32,
    bits: u32,
    mode: String,
    dtype: DType,
    warmup: Option<u32>,
    iters: Option<u32>,
) -> Result<QmvQuantizedMicrobenchResult> {
    let warmup = warmup.unwrap_or(10) as usize;
    let iters = iters.unwrap_or(50) as usize;
    if iters == 0 {
        return Err(Error::from_reason("iters must be > 0"));
    }
    if k == 0 || n == 0 || m == 0 {
        return Err(Error::from_reason("k, n, and m must be > 0"));
    }
    if m > 16 {
        return Err(Error::from_reason("m must be <= 16 for qmv microbench"));
    }
    if !k.is_multiple_of(512) {
        return Err(Error::from_reason(format!(
            "K={k} must be divisible by 512 for the small-M qmv fast path"
        )));
    }
    if !n.is_multiple_of(16) {
        return Err(Error::from_reason(format!(
            "N={n} must be divisible by 16 for qmv_fast_sg4 tiles"
        )));
    }
    if dtype != DType::BFloat16 && dtype != DType::Float16 {
        return Err(Error::from_reason(
            "dtype must be BFloat16 or Float16 for qmv microbench",
        ));
    }
    if !supported_quantized_qmv_combo(&mode, group_size, bits) {
        return Err(Error::from_reason(format!(
            "unsupported qmv microbench combo: mode={mode} group_size={group_size} bits={bits}"
        )));
    }

    let inputs = make_quantized_microbench_inputs(
        k as i64,
        n as i64,
        m as i64,
        group_size as i32,
        bits as i32,
        &mode,
        dtype,
    )?;
    let QuantizedMicrobenchInputs {
        x,
        w,
        scales,
        biases,
    } = inputs;
    let biases_ref = biases.as_ref();

    for _ in 0..warmup {
        let out = quantized_qmv_matmul(
            &x,
            &w,
            &scales,
            biases_ref,
            group_size as i32,
            bits as i32,
            &mode,
        )?;
        out.eval();
    }

    let mut samples: Vec<u128> = Vec::with_capacity(iters);
    let mut last = None;
    for _ in 0..iters {
        let t0 = Instant::now();
        let out = quantized_qmv_matmul(
            &x,
            &w,
            &scales,
            biases_ref,
            group_size as i32,
            bits as i32,
            &mode,
        )?;
        out.eval();
        samples.push(t0.elapsed().as_nanos());
        last = Some(out);
    }

    let checksum = if let Some(out) = last {
        let sum = out.astype(DType::Float32)?.sum(None, None)?;
        sum.eval();
        sum.to_float32()?[0] as f64
    } else {
        0.0
    };

    Ok(QmvQuantizedMicrobenchResult {
        median_ns: median_ns(&mut samples),
        checksum,
    })
}

/// Median of a sample vector in ns. Sorts in place. Empty input returns 0.
fn median_ns(samples: &mut [u128]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_unstable();
    let mid = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[mid] as f64
    } else {
        // Two-sample average — use f64 to avoid u128 overflow on
        // pathologically large measurements (not expected in practice but
        // defensive).
        (samples[mid - 1] as f64 + samples[mid] as f64) / 2.0
    }
}

pub(crate) struct QuantizedMicrobenchInputs {
    pub x: MxArray,
    pub w: MxArray,
    pub scales: MxArray,
    pub biases: Option<MxArray>,
}

fn supported_quantized_qmv_combo(mode: &str, group_size: u32, bits: u32) -> bool {
    match mode {
        "affine" => matches!(group_size, 32 | 64 | 128) && matches!(bits, 4 | 5 | 6 | 8),
        "nvfp4" => group_size == 16 && bits == 4,
        "mxfp4" => group_size == 32 && bits == 4,
        "mxfp8" => group_size == 32 && bits == 8,
        _ => false,
    }
}

pub(crate) fn make_quantized_microbench_inputs(
    k: i64,
    n: i64,
    m: i64,
    group_size: i32,
    bits: i32,
    mode: &str,
    dtype: DType,
) -> Result<QuantizedMicrobenchInputs> {
    use std::ffi::CString;

    unsafe { sys::mlx_seed(0x5156_4d56) };

    let x = MxArray::random_normal(&[m, k], 0.0, 1.0, Some(dtype))?;
    let w_f32 = MxArray::random_normal(&[n, k], 0.0, 1.0, Some(DType::Float32))?;
    let w_target = w_f32.astype(dtype)?;

    let mode_c = CString::new(mode)
        .map_err(|_| Error::from_reason("mode must not contain NUL bytes".to_string()))?;
    let mut out_q: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_s: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_b: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_quantize(
            w_target.as_raw_ptr(),
            group_size,
            bits,
            mode_c.as_ptr(),
            &mut out_q,
            &mut out_s,
            &mut out_b,
        )
    };
    if !ok {
        return Err(Error::from_reason(format!(
            "mlx_quantize failed building {mode}/{bits}b microbench inputs"
        )));
    }

    let packed = MxArray::from_handle(out_q, "qmv_microbench:packed")?;
    let scales = MxArray::from_handle(out_s, "qmv_microbench:scales")?;
    let biases = if out_b.is_null() {
        None
    } else {
        Some(MxArray::from_handle(out_b, "qmv_microbench:biases")?.astype(dtype)?)
    };
    let scales = if mode == "affine" {
        scales.astype(dtype)?
    } else {
        scales
    };

    x.eval();
    packed.eval();
    scales.eval();
    if let Some(biases) = biases.as_ref() {
        biases.eval();
    }

    Ok(QuantizedMicrobenchInputs {
        x,
        w: packed,
        scales,
        biases,
    })
}
