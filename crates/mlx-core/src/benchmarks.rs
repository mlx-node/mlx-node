//! Test-only harness for the production quantized-`qmv` dispatch path.
//!
//! Helpers that wrap `mlx::core::quantized_matmul` (the production dispatch,
//! including the small-M `qmv_fast_sg4` fast path) and build seeded
//! quantized inputs, used by the manual `#[ignore]` CLEF benchmarks below.
//!
//! Why median (not mean): GPU command-buffer setup latency lands on the first
//! few dispatches even after warmup, and a single 2-3× outlier badly skews
//! the mean. Median makes the harness robust to thermal hiccups and JIT
//! recompilation between cold-cache transitions.

use std::time::Instant;

use mlx_sys as sys;
use napi::bindgen_prelude::*;

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

#[cfg(test)]
mod clef_qmm_probe {
    use super::*;

    /// Probe: is a packed QMM or dequantize+dense-GEMM cheaper at the CLEF
    /// prefill shape? Run: cargo test -p mlx-core clef_qmm_probe -- --ignored --nocapture
    #[test]
    #[ignore = "manual benchmark"]
    fn qmm_vs_dequant_gemm_at_clef_shape() {
        let (m, n, k, gs, bits) = (295i64, 4096i64, 4096i64, 32i32, 8i32);
        let QuantizedMicrobenchInputs {
            x,
            w,
            scales,
            biases,
        } = make_quantized_microbench_inputs(k, n, m, gs, bits, "affine", DType::BFloat16).unwrap();

        // warm
        for _ in 0..5 {
            quantized_qmv_matmul(&x, &w, &scales, biases.as_ref(), gs, bits, "affine")
                .unwrap()
                .eval();
        }
        let bench = |f: &mut dyn FnMut() -> Result<MxArray>| -> f64 {
            let mut ts = Vec::new();
            for _ in 0..30 {
                let t = Instant::now();
                f().unwrap().eval();
                ts.push(t.elapsed().as_nanos());
            }
            median_ns(&mut ts)
        };
        let qmm_ms = bench(&mut || {
            quantized_qmv_matmul(&x, &w, &scales, biases.as_ref(), gs, bits, "affine")
        });

        // dequant once (fixed cost) then dense matmul per call
        let w_dense = {
            use std::ffi::CString;
            let mode_c = CString::new("affine").unwrap();
            let handle = unsafe {
                sys::mlx_dequantize(
                    w.as_raw_ptr(),
                    scales.as_raw_ptr(),
                    biases
                        .as_ref()
                        .map_or(std::ptr::null_mut(), |b| b.as_raw_ptr()),
                    gs,
                    bits,
                    -1,
                    mode_c.as_ptr(),
                )
            };
            MxArray::from_handle(handle, "deq").unwrap()
        };
        w_dense.eval();
        let w_t = w_dense.transpose(Some(&[1, 0])).unwrap();
        let gemm_ms = bench(&mut || x.matmul(&w_t));
        let deq_ms = {
            let mut ts = Vec::new();
            for _ in 0..10 {
                use std::ffi::CString;
                let mode_c = CString::new("affine").unwrap();
                let t = Instant::now();
                let handle = unsafe {
                    sys::mlx_dequantize(
                        w.as_raw_ptr(),
                        scales.as_raw_ptr(),
                        biases
                            .as_ref()
                            .map_or(std::ptr::null_mut(), |b| b.as_raw_ptr()),
                        gs,
                        bits,
                        -1,
                        mode_c.as_ptr(),
                    )
                };
                MxArray::from_handle(handle, "deq").unwrap().eval();
                ts.push(t.elapsed().as_nanos());
            }
            median_ns(&mut ts)
        };

        let weight_gb = (n * k) as f64 / 1e9; // q8 ≈ 1 byte/weight + sidecars
        eprintln!("M={m} N={n} K={k} q8g{gs} bf16-x");
        eprintln!(
            "  packed QMM:      {:.3} ms  ({:.0} GB/s effective weight stream)",
            qmm_ms / 1e6,
            weight_gb / (qmm_ms / 1e9)
        );
        eprintln!("  dense GEMM:      {:.3} ms", gemm_ms / 1e6);
        eprintln!("  dequant (once):  {:.3} ms", deq_ms / 1e6);
        eprintln!("  deq+GEMM total:  {:.3} ms", (deq_ms + gemm_ms) / 1e6);
        eprintln!(
            "  GEMM TFLOPS:     {:.1}",
            2.0 * m as f64 * n as f64 * k as f64 / gemm_ms
        );
    }
}

#[cfg(test)]
mod clef_qmm_sweep {
    use super::*;
    use std::ffi::CString;

    /// Dequantize `w` under `mode`, mirroring the probe's helper.
    fn dequant(
        w: &MxArray,
        scales: &MxArray,
        biases: Option<&MxArray>,
        gs: i32,
        bits: i32,
        mode: &str,
    ) -> MxArray {
        let mode_c = CString::new(mode).unwrap();
        let handle = unsafe {
            sys::mlx_dequantize(
                w.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases.map_or(std::ptr::null_mut(), |b| b.as_raw_ptr()),
                gs,
                bits,
                -1,
                mode_c.as_ptr(),
            )
        };
        MxArray::from_handle(handle, "deq").unwrap()
    }

    /// M sweep at K=N=4096 q8g32 affine with BOTH sidecar dtypes:
    /// f32 sidecars = the real GGUF->bf16 compute path (promotes the QMM to
    /// f32, disqualifies qmm_nax); bf16 sidecars = the hypothetical nax path.
    /// Also measures dequant-once + dense GEMM per M.
    #[test]
    #[ignore = "manual benchmark"]
    fn clef_m_sweep() {
        for m in [64i64, 128, 295, 512, 1024, 2048] {
            let (n, k, gs, bits) = (4096i64, 4096i64, 32i32, 8i32);
            let QuantizedMicrobenchInputs {
                x,
                w,
                scales,
                biases,
            } = make_quantized_microbench_inputs(k, n, m, gs, bits, "affine", DType::BFloat16)
                .unwrap();
            // Real path: f32 sidecars (post promote_affine_sidecars_to_f32)
            let scales32 = scales.astype(DType::Float32).unwrap();
            let biases32 = biases.as_ref().map(|b| b.astype(DType::Float32).unwrap());
            scales32.eval();
            if let Some(b) = &biases32 {
                b.eval();
            }
            let bench = |f: &mut dyn FnMut() -> Result<MxArray>| -> f64 {
                let mut ts = Vec::new();
                for _ in 0..30 {
                    let t = Instant::now();
                    f().unwrap().eval();
                    ts.push(t.elapsed().as_nanos());
                }
                median_ns(&mut ts)
            };
            for _ in 0..5 {
                quantized_qmv_matmul(&x, &w, &scales32, biases32.as_ref(), gs, bits, "affine")
                    .unwrap()
                    .eval();
                quantized_qmv_matmul(&x, &w, &scales, biases.as_ref(), gs, bits, "affine")
                    .unwrap()
                    .eval();
            }
            let qmm_f32 = bench(&mut || {
                quantized_qmv_matmul(&x, &w, &scales32, biases32.as_ref(), gs, bits, "affine")
            });
            let qmm_bf16 = bench(&mut || {
                quantized_qmv_matmul(&x, &w, &scales, biases.as_ref(), gs, bits, "affine")
            });
            let w_dense = dequant(&w, &scales32, biases32.as_ref(), gs, bits, "affine")
                .astype(DType::BFloat16)
                .unwrap();
            w_dense.eval();
            let w_t = w_dense.transpose(Some(&[1, 0])).unwrap();
            let gemm = bench(&mut || x.matmul(&w_t));
            eprintln!(
                "M={m:>5} N={n} K={k} q8g{gs}: qmm_f32_sidecars={:.3}ms qmm_bf16_sidecars(nax)={:.3}ms cached_deq+gemm={:.3}ms",
                qmm_f32 / 1e6,
                qmm_bf16 / 1e6,
                gemm / 1e6,
            );
        }
    }

    /// Real clef-flash projection shapes at M=295, f32 sidecars vs nax vs
    /// cached dequant+GEMM. Includes a dequant-cost measurement per shape.
    #[test]
    #[ignore = "manual benchmark"]
    fn clef_shape_sweep() {
        for (n, k) in [
            (2048i64, 4096i64), // kv merged
            (4096, 4096),       // o_proj, down_proj, gdn out_proj
            (8256, 4096),       // gdn qkvz+ba merged
            (8192, 4096),       // q_proj (gate)
            (12288, 4096),      // gate_up
            (12288, 2560),      // (unused, sanity)
            (2560, 4096),       // small-N shape
        ] {
            let (m, gs, bits) = (295i64, 32i32, 8i32);
            let QuantizedMicrobenchInputs {
                x,
                w,
                scales,
                biases,
            } = make_quantized_microbench_inputs(k, n, m, gs, bits, "affine", DType::BFloat16)
                .unwrap();
            let scales32 = scales.astype(DType::Float32).unwrap();
            let biases32 = biases.as_ref().map(|b| b.astype(DType::Float32).unwrap());
            scales32.eval();
            if let Some(b) = &biases32 {
                b.eval();
            }
            let bench = |f: &mut dyn FnMut() -> Result<MxArray>| -> f64 {
                let mut ts = Vec::new();
                for _ in 0..20 {
                    let t = Instant::now();
                    f().unwrap().eval();
                    ts.push(t.elapsed().as_nanos());
                }
                median_ns(&mut ts)
            };
            for _ in 0..5 {
                quantized_qmv_matmul(&x, &w, &scales32, biases32.as_ref(), gs, bits, "affine")
                    .unwrap()
                    .eval();
                quantized_qmv_matmul(&x, &w, &scales, biases.as_ref(), gs, bits, "affine")
                    .unwrap()
                    .eval();
            }
            let qmm_f32 = bench(&mut || {
                quantized_qmv_matmul(&x, &w, &scales32, biases32.as_ref(), gs, bits, "affine")
            });
            let qmm_bf16 = bench(&mut || {
                quantized_qmv_matmul(&x, &w, &scales, biases.as_ref(), gs, bits, "affine")
            });
            let deq = {
                let mut ts = Vec::new();
                for _ in 0..10 {
                    let t = Instant::now();
                    dequant(&w, &scales32, biases32.as_ref(), gs, bits, "affine").eval();
                    ts.push(t.elapsed().as_nanos());
                }
                median_ns(&mut ts)
            };
            let w_dense = dequant(&w, &scales32, biases32.as_ref(), gs, bits, "affine")
                .astype(DType::BFloat16)
                .unwrap();
            w_dense.eval();
            let w_t = w_dense.transpose(Some(&[1, 0])).unwrap();
            let gemm = bench(&mut || x.matmul(&w_t));
            let gb = (n * k) as f64 / 1e9;
            eprintln!(
                "N={n:>6} K={k:>5} M=295: qmm_f32={:.3}ms qmm_nax={:.3}ms deq={:.3}ms gemm={:.3}ms (nax weight-stream {:.0}GB/s)",
                qmm_f32 / 1e6,
                qmm_bf16 / 1e6,
                deq / 1e6,
                gemm / 1e6,
                gb / (qmm_bf16 / 1e9),
            );
        }
    }

    /// Head-relevant: dequant cost of the [vocab, hidden] output table vs
    /// dequant of a gathered subset of rows (packed-row gather then dequant).
    #[test]
    #[ignore = "manual benchmark"]
    fn clef_head_row_dequant() {
        let (vocab, k, gs, bits) = (248320i64, 4096i64, 32i32, 8i32);
        let QuantizedMicrobenchInputs {
            w, scales, biases, ..
        } = make_quantized_microbench_inputs(k, vocab, 1, gs, bits, "affine", DType::BFloat16)
            .unwrap();
        let scales32 = scales.astype(DType::Float32).unwrap();
        let biases32 = biases.as_ref().map(|b| b.astype(DType::Float32).unwrap());
        scales32.eval();
        if let Some(b) = &biases32 {
            b.eval();
        }
        w.eval();
        let t = Instant::now();
        let full = dequant(&w, &scales32, biases32.as_ref(), gs, bits, "affine");
        full.eval();
        let full_ms = t.elapsed().as_nanos() as f64 / 1e6;
        eprintln!("full output-weight dequant [{vocab},{k}]: {full_ms:.2} ms");

        // Gather ~500 packed rows (option tokens) then dequant rows only.
        let n_rows = 500i64;
        let idx_data: Vec<u32> = (0..n_rows as u32).collect();
        let idx = MxArray::from_uint32(&idx_data, &[n_rows]).unwrap();
        let row_packed_words = k * bits as i64 / 32;
        let w_rows = w.take(&idx, 0).unwrap();
        let group_count = k / gs as i64;
        let s_rows = scales32.take(&idx, 0).unwrap();
        let b_rows = biases32.as_ref().map(|b| b.take(&idx, 0).unwrap());
        w_rows.eval();
        s_rows.eval();
        if let Some(b) = &b_rows {
            b.eval();
        }
        let _ = (row_packed_words, group_count);
        let t = Instant::now();
        let rows = dequant(&w_rows, &s_rows, b_rows.as_ref(), gs, bits, "affine");
        rows.eval();
        eprintln!(
            "gather {n_rows} packed rows + dequant: {:.3} ms",
            t.elapsed().as_nanos() as f64 / 1e6
        );
    }
}

#[cfg(test)]
mod clef_qmm_nax_parity {
    use super::*;

    /// Numerics probe: bf16-sidecar (nax-eligible) affine Q8 QMM vs the f32
    /// sidecar production path — max abs/rel output delta at clef shapes.
    /// The only difference is scale/bias storage precision (f32->bf16) and
    /// the kernel's internal math dtype.
    #[test]
    #[ignore = "manual parity probe"]
    fn nax_vs_f32_sidecar_parity() {
        for (n, k) in [(4096i64, 4096i64), (12288, 4096), (8256, 4096)] {
            let (m, gs, bits) = (295i64, 32i32, 8i32);
            let QuantizedMicrobenchInputs {
                x,
                w,
                scales,
                biases,
            } = make_quantized_microbench_inputs(k, n, m, gs, bits, "affine", DType::BFloat16)
                .unwrap();
            let scales32 = scales.astype(DType::Float32).unwrap();
            let biases32 = biases.as_ref().map(|b| b.astype(DType::Float32).unwrap());
            scales32.eval();
            if let Some(b) = &biases32 {
                b.eval();
            }
            let out_f32 =
                quantized_qmv_matmul(&x, &w, &scales32, biases32.as_ref(), gs, bits, "affine")
                    .unwrap();
            let out_bf16 =
                quantized_qmv_matmul(&x, &w, &scales, biases.as_ref(), gs, bits, "affine").unwrap();
            out_f32.eval();
            out_bf16.eval();
            let a = out_f32.astype(DType::Float32).unwrap();
            let b = out_bf16.astype(DType::Float32).unwrap();
            let diff = a.sub(&b).unwrap().abs().unwrap();
            diff.eval();
            let max_d = diff.max(None, None).unwrap();
            max_d.eval();
            let ref_abs = a.abs().unwrap().max(None, None).unwrap();
            ref_abs.eval();
            let md = max_d.to_float32().unwrap()[0];
            let ra = ref_abs.to_float32().unwrap()[0];
            eprintln!(
                "N={n} K={k}: max|dout|={md:.5}  max|out|={ra:.3}  rel={:.4}",
                md / ra
            );
        }
    }
}
