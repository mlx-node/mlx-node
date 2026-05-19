//! Microbench harness for the W6.30 MTPLX multi3 qmv4 kernel port.
//!
//! Exposes a single NAPI function — `multi3_qmv4_microbench` — that wraps the
//! kernel and the stock `mlx::core::quantized_matmul` baseline in a warmup +
//! median-of-iters wall-clock timing loop. The harness is measurement-only;
//! the kernel itself is not wired into any production path here. Decision-gate
//! follow-up tasks (W6.37 verify hot-path integration, W6.31 NVFP4 variant)
//! consume the numbers this produces.
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

// ---------------------------------------------------------------------
// Safe Rust wrappers around the W6.30 extern "C" FFI surface declared in
// `mlx_qmv_multi3.h`. Keep these private — outside callers should go
// through `multi3_qmv4_microbench` (NAPI) or the test entry points.
// ---------------------------------------------------------------------

/// Run the multi3 qmv4 kernel on the given inputs. Returns the `[3, N]`
/// result array. Falls back internally to stock `quantized_matmul` if the
/// inputs fall outside the kernel's eligibility envelope.
pub(crate) fn multi3_qmv4_matmul(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    group_size: i32,
) -> Result<MxArray> {
    let handle = unsafe {
        sys::mlx_multi3_qmv4_matmul(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
            group_size,
        )
    };
    MxArray::from_handle(handle, "multi3_qmv4_matmul")
}

/// Run stock `mlx::core::quantized_matmul` with affine/4-bit settings on the
/// given inputs. Returns the `[M, N]` result array (M derived from `x.shape[-2]`).
pub(crate) fn stock_qmv4_matmul(
    x: &MxArray,
    w: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    group_size: i32,
) -> Result<MxArray> {
    let handle = unsafe {
        sys::mlx_stock_qmv4_matmul(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
            group_size,
        )
    };
    MxArray::from_handle(handle, "stock_qmv4_matmul")
}

/// Max absolute difference between two arrays, computed element-wise after
/// up-casting both to f32. Both arrays must have the same shape. Used both by
/// the parity tests and by the NAPI microbench's parity probe so a
/// fast-but-wrong kernel cannot be greenlit by a speed-only gate.
///
/// Returns the max diff as `f32`. See `parity_tests::PARITY_TOL_BF16` for the
/// recommended pass/fail threshold at bf16.
pub(crate) fn max_abs_diff(a: &MxArray, b: &MxArray) -> Result<f32> {
    let a_f32 = a.astype(DType::Float32)?;
    let b_f32 = b.astype(DType::Float32)?;
    let diff = a_f32.sub(&b_f32)?.abs()?;
    let m = diff.max(None, None)?;
    m.eval();
    let v = m.to_float32()?;
    Ok(v[0])
}

// ---------------------------------------------------------------------
// NAPI-facing microbench harness.
// ---------------------------------------------------------------------

/// Microbench result for a single (K, N, group_size, dtype) shape.
#[napi(object, js_name = "QmvMulti3MicrobenchResult")]
#[derive(Clone, Debug)]
pub struct QmvMulti3MicrobenchResult {
    /// Median stock `quantized_matmul` wall-clock per call, in nanoseconds.
    pub stock_ns: f64,
    /// Median multi3 qmv4 kernel wall-clock per call, in nanoseconds.
    pub kernel_ns: f64,
    /// `stock_ns / kernel_ns`. Values > 1 mean the kernel is faster than stock.
    pub ratio: f64,
    /// Max absolute difference (element-wise, after f32 up-cast) between stock
    /// and kernel outputs on the same inputs. Sampled once per call. The
    /// kernel reorders per-group accumulation and bf16/fp16 round to limited
    /// mantissa, so non-zero values are expected; the gate is `<= 5e-2`
    /// (matches `parity_tests::PARITY_TOL_BF16`).
    pub max_abs_diff: f64,
}

/// Run the multi3 qmv4 microbench on the given shape.
///
/// Arguments:
///   - `k`, `n`: matmul dimensions. Must satisfy `K % 512 == 0`, `N % 8 == 0`
///     (per the kernel's eligibility envelope).
///   - `group_size`: one of {32, 64, 128}.
///   - `dtype`: bf16 or fp16; scales/biases are quantized to match.
///   - `warmup`: number of untimed dispatches before measurement starts.
///     Recommended ≥ 5 so command-buffer setup latency does not contaminate
///     the timed window. Defaults to 5.
///   - `iters`: number of timed dispatches. Median is reported. Defaults to 20.
///
/// The harness synchronously evaluates each result via `MxArray::eval()` so
/// the wall-clock interval covers the full GPU dispatch (including
/// command-buffer flush) rather than just the lazy graph build. This matches
/// what production verify code observes.
#[napi]
pub fn multi3_qmv4_microbench(
    k: u32,
    n: u32,
    group_size: u32,
    dtype: DType,
    warmup: Option<u32>,
    iters: Option<u32>,
) -> Result<QmvMulti3MicrobenchResult> {
    let warmup = warmup.unwrap_or(5) as usize;
    let iters = iters.unwrap_or(20) as usize;
    if iters == 0 {
        return Err(Error::from_reason("iters must be > 0"));
    }
    if k == 0 || n == 0 {
        return Err(Error::from_reason("k and n must be > 0"));
    }
    if !k.is_multiple_of(512) {
        return Err(Error::from_reason(format!(
            "K={k} must be divisible by 512 (multi3_qmv4 eligibility)"
        )));
    }
    if !n.is_multiple_of(8) {
        return Err(Error::from_reason(format!(
            "N={n} must be divisible by 8 (multi3_qmv4 eligibility)"
        )));
    }
    if group_size != 32 && group_size != 64 && group_size != 128 {
        return Err(Error::from_reason(format!(
            "group_size={group_size} must be one of {{32, 64, 128}}"
        )));
    }
    if dtype != DType::BFloat16 && dtype != DType::Float16 {
        return Err(Error::from_reason(
            "dtype must be BFloat16 or Float16 (multi3_qmv4 eligibility)",
        ));
    }

    // Build inputs: random x[3, K] in target dtype, random w[N, K] f32 then
    // quantize to (uint32, scales, biases) at the requested group_size and
    // bits=4 in affine mode (the only mode multi3_qmv4 supports).
    let inputs = make_microbench_inputs(k as i64, n as i64, group_size as i32, dtype)?;
    let MicrobenchInputs {
        x,
        w,
        scales,
        biases,
    } = inputs;

    // Warmup — first dispatch compiles the kernel and pays cold-cache cost
    // for both paths. We run the *same* number of warmup iterations on both
    // sides so the comparison stays fair: neither pays a one-shot compile
    // tax inside the timed window.
    for _ in 0..warmup {
        let s = stock_qmv4_matmul(&x, &w, &scales, &biases, group_size as i32)?;
        s.eval();
        let kr = multi3_qmv4_matmul(&x, &w, &scales, &biases, group_size as i32)?;
        kr.eval();
    }

    // One-shot parity probe on the same inputs the timing loop will use.
    // Sampled once (deterministic given fixed inputs); a fast-but-wrong
    // kernel must not be greenlit by a speed-only gate.
    let parity_diff = {
        let s = stock_qmv4_matmul(&x, &w, &scales, &biases, group_size as i32)?;
        s.eval();
        let kr = multi3_qmv4_matmul(&x, &w, &scales, &biases, group_size as i32)?;
        kr.eval();
        max_abs_diff(&s, &kr)? as f64
    };

    // Interleave stock and kernel measurements so neither benefits unfairly
    // from a thermally-cooler GPU at the start of the timed window. We sort
    // the per-iter samples post hoc and take the median.
    let mut stock_samples: Vec<u128> = Vec::with_capacity(iters);
    let mut kernel_samples: Vec<u128> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let s = stock_qmv4_matmul(&x, &w, &scales, &biases, group_size as i32)?;
        s.eval();
        stock_samples.push(t0.elapsed().as_nanos());

        let t1 = Instant::now();
        let kr = multi3_qmv4_matmul(&x, &w, &scales, &biases, group_size as i32)?;
        kr.eval();
        kernel_samples.push(t1.elapsed().as_nanos());
    }

    let stock_ns = median_ns(&mut stock_samples);
    let kernel_ns = median_ns(&mut kernel_samples);
    let ratio = if kernel_ns > 0.0 {
        stock_ns / kernel_ns
    } else {
        0.0
    };
    Ok(QmvMulti3MicrobenchResult {
        stock_ns,
        kernel_ns,
        ratio,
        max_abs_diff: parity_diff,
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

/// Random `(x, w, scales, biases)` tuple at the requested shape and dtype.
/// `w/scales/biases` are produced by quantizing a random f32 weight matrix
/// at `(group_size, bits=4, mode="affine")` and then casting scales/biases
/// to `dtype` so they match `x` (the kernel requires dtype-matched
/// scales/biases).
pub(crate) struct MicrobenchInputs {
    pub x: MxArray,
    pub w: MxArray,
    pub scales: MxArray,
    pub biases: MxArray,
}

pub(crate) fn make_microbench_inputs(
    k: i64,
    n: i64,
    group_size: i32,
    dtype: DType,
) -> Result<MicrobenchInputs> {
    use std::ffi::CString;

    // x: [3, K] in target dtype, small std so the int4 quant of the corresponding
    // weight matrix has a non-degenerate dynamic range.
    let x = MxArray::random_normal(&[3, k], 0.0, 1.0, Some(dtype))?;

    // Weight: random f32 [N, K] then quantize. scales/biases come out in f32
    // (or matching dtype depending on the MLX side), so we cast them to the
    // target dtype below to satisfy the multi3_qmv4 eligibility check.
    let w_f32 = MxArray::random_normal(&[n, k], 0.0, 1.0, Some(DType::Float32))?;
    let w_target = w_f32.astype(dtype)?;

    let mode_c = CString::new("affine").unwrap();
    let mut out_q: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_s: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_b: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_quantize(
            w_target.as_raw_ptr(),
            group_size,
            4,
            mode_c.as_ptr(),
            &mut out_q,
            &mut out_s,
            &mut out_b,
        )
    };
    if !ok {
        return Err(Error::from_reason(
            "mlx_quantize failed building microbench inputs",
        ));
    }
    let packed = MxArray::from_handle(out_q, "microbench:packed")?;
    let scales = MxArray::from_handle(out_s, "microbench:scales")?;
    let biases_arr = if out_b.is_null() {
        return Err(Error::from_reason(
            "mlx_quantize returned no biases for affine mode",
        ));
    } else {
        MxArray::from_handle(out_b, "microbench:biases")?
    };

    // scales/biases dtype must match x for the kernel's `T` template
    // substitution. `mlx_quantize` returns them in the input dtype, so the
    // astype is a no-op in the common case — kept defensive in case MLX
    // bumps change that contract.
    let scales = scales.astype(dtype)?;
    let biases_arr = biases_arr.astype(dtype)?;

    // Force materialization once so the timing loop measures the kernel
    // dispatch rather than the lazy quantize graph build.
    x.eval();
    packed.eval();
    scales.eval();
    biases_arr.eval();

    Ok(MicrobenchInputs {
        x,
        w: packed,
        scales,
        biases: biases_arr,
    })
}

// ---------------------------------------------------------------------
// Parity tests — assert the kernel's output matches stock `quantized_matmul`
// to a tight tolerance across three realistic Qwen3.5 shapes. The kernel
// uses a different accumulation order (per-thread fp32 accum + simd_sum
// reduction) than stock qmv, so bit-exact equality is not expected; the goal
// is "no algorithmic bug" — every output element should agree with stock to
// well within bf16's per-element rounding error.
// ---------------------------------------------------------------------

#[cfg(test)]
mod parity_tests {
    use super::*;

    /// (K, N, group_size, label) tuples covering the three Qwen3.5 shape
    /// classes the verify path will hit:
    ///   - Q projection: K=2560, N=2560
    ///   - MLP up:        K=2560, N=9728
    ///   - MLP down:      K=9728, N=2560
    ///
    /// All satisfy `K % 512 == 0` and `N % 8 == 0`.
    const QWEN35_SHAPES: &[(i64, i64, i32, &str)] = &[
        (2560, 2560, 64, "q_proj"),
        (2560, 9728, 64, "mlp_up"),
        (9728, 2560, 64, "mlp_down"),
    ];

    /// Tolerance for stock vs kernel parity at bf16. The kernel reorders the
    /// per-group accumulation across threads (each simd lane owns a stride),
    /// and bf16 round-down to 7 mantissa bits, so a tight-but-not-bit-exact
    /// tolerance is correct. 5e-2 is generous enough to cover the worst
    /// numerical drift across the three shapes we test.
    const PARITY_TOL_BF16: f32 = 5e-2;

    #[test]
    fn parity_against_stock_qmv4_bf16() {
        for &(k, n, gs, label) in QWEN35_SHAPES {
            let inputs = make_microbench_inputs(k, n, gs, DType::BFloat16)
                .unwrap_or_else(|e| panic!("[{label}] make_microbench_inputs failed: {e}"));
            let MicrobenchInputs {
                x,
                w,
                scales,
                biases,
            } = inputs;

            let stock = stock_qmv4_matmul(&x, &w, &scales, &biases, gs)
                .unwrap_or_else(|e| panic!("[{label}] stock dispatch failed: {e}"));
            stock.eval();

            let kernel = multi3_qmv4_matmul(&x, &w, &scales, &biases, gs)
                .unwrap_or_else(|e| panic!("[{label}] kernel dispatch failed: {e}"));
            kernel.eval();

            // Sanity-check shape equality before computing the diff.
            let stock_shape: Vec<i64> = stock.shape().unwrap().to_vec();
            let kernel_shape: Vec<i64> = kernel.shape().unwrap().to_vec();
            assert_eq!(
                stock_shape, kernel_shape,
                "[{label}] shape mismatch: stock={stock_shape:?} kernel={kernel_shape:?}"
            );

            // Stock-output magnitude — guards against the silent-failure mode where
            // both paths produce all-zeros and "agree" trivially. The synthetic
            // inputs are N(0,1) f32 weights quantized to 4-bit, so the dot product
            // of K independent normals should have magnitude on the order of √K
            // and the max abs entry should be well above 0.5 for any K ≥ 64.
            let stock_abs = stock.abs().unwrap();
            let stock_max = stock_abs.max(None, None).unwrap();
            stock_max.eval();
            let stock_max_v = stock_max.to_float32().unwrap()[0];
            assert!(
                stock_max_v > 0.5,
                "[{label}] stock output looks trivial (max abs = {stock_max_v:.4e}) — \
                 inputs probably not materialized correctly",
            );

            let diff = super::max_abs_diff(&stock, &kernel)
                .unwrap_or_else(|e| panic!("[{label}] max_abs_diff failed: {e}"));
            assert!(
                diff <= PARITY_TOL_BF16,
                "[{label}] (K={k}, N={n}, gs={gs}): max abs diff {diff:.4e} > tol {PARITY_TOL_BF16:.4e} \
                 (stock_max_abs={stock_max_v:.4e})",
            );
            eprintln!(
                "[parity] {label} K={k} N={n} gs={gs} stock_max_abs={stock_max_v:.4e} \
                 max_abs_diff={diff:.6e}"
            );
        }
    }
}
