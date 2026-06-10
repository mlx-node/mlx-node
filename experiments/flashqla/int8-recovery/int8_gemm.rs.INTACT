//! NA (Neural Accelerator) int8 W8A8 prefill GEMM — isolated, proven primitive.
//!
//! This is the Stage 1+2 primitive that Stage 3 (MLP wiring) will call. It wraps
//! three C++ FFI ops (see `crates/mlx-sys/src/mlx_na_int8.cpp`):
//!   * [`matmul_int8`]      — int8 `x @ w^T -> int32` (bit-exact integer GEMM)
//!   * [`quantize_weight_int8`] — per-output-channel symmetric int8 weight quant
//!   * [`int8_w8a8_matmul`] — per-token int8 activation quant + GEMM + rescale
//!
//! ## int8 lives entirely C++-side
//! Rust has no `Int8` [`DType`]. The integer GEMM therefore takes bf16/f32
//! [`MxArray`]s holding **integer values in `[-127, 127]`** and casts them to
//! int8 inside C++. The W8A8 path holds the quantized weight as an **opaque**
//! [`MxArray`] handle (int8-typed in MLX) that Rust never introspects.
//!
//! ## Gating (M-threshold / arch)
//! Every op gates internally on **GPU gen >= 17 (M5+)** and **`K % 16 == 0`**,
//! returning `false` on unsupported hardware/shape so the caller can fall back
//! to a bf16 `matmul`. Stage 3 must check eligibility before routing a linear
//! through this path (the NA matmul2d is a *prefill* GEMM — only worth it at
//! `M` large enough to amortize quant; the threshold is a Stage-3 policy knob).

use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// int8 `x @ w^T -> int32 [M, N]`.
///
/// `x` is `[M, K]` and `w` is `[N, K]` (weight rows are output channels), both
/// bf16/f32 [`MxArray`]s holding **exact integer values in `[-127, 127]`**. The
/// returned [`MxArray`] is `Int32 [M, N]`, bit-exact equal to the integer
/// reference `x @ w^T`.
///
/// Returns `Err` when the op is unsupported (gen < 17 or `K % 16 != 0`) or on a
/// kernel/FFI failure — the caller is expected to fall back to a bf16 `matmul`.
pub fn matmul_int8(x: &MxArray, w: &MxArray) -> Result<MxArray> {
    let mut out: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe { sys::mlx_matmul_int8(x.as_raw_ptr(), w.as_raw_ptr(), &mut out) };
    if !ok {
        return Err(Error::from_reason(
            "mlx_matmul_int8 failed (unsupported gen/K or kernel error; see stderr)",
        ));
    }
    MxArray::from_handle(out, "matmul_int8")
}

/// Per-output-channel symmetric int8 weight quantization (load-time; runs once).
///
/// `w` is `[N, K]` bf16/f32. Returns `(w_i8, s_w)` where:
///   * `w_i8` is an **opaque** int8 [`MxArray`] (Rust never reads it). Stage 4b
///     stores it ALREADY in the `[K, N]` kernel layout (transpose+contiguous
///     hoisted here, at load time) so the per-forward GEMM does zero weight
///     reshaping. Rust treats it as opaque, so the stored orientation is
///     invisible to callers — they just hand it back to [`int8_w8a8_matmul`].
///   * `s_w` is `f32 [N]`, the per-output-channel scale `max_k|w[n,k]| / 127`.
///     The scale indexes the OUTPUT channel `N` regardless of weight storage
///     orientation, so it stays correct for the `acc[M,N] * s_x[M] * s_w[N]`
///     rescale.
///
/// Stage 3 holds both handles alongside each quantized linear and passes them to
/// [`int8_w8a8_matmul`] on every forward.
pub fn quantize_weight_int8(w: &MxArray) -> Result<(MxArray, MxArray)> {
    let mut out_w_i8: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_s_w: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe { sys::mlx_quantize_weight_int8(w.as_raw_ptr(), &mut out_w_i8, &mut out_s_w) };
    if !ok {
        return Err(Error::from_reason(
            "mlx_quantize_weight_int8 failed (see stderr)",
        ));
    }
    let w_i8 = MxArray::from_handle(out_w_i8, "quantize_weight_int8:w_i8")?;
    let s_w = MxArray::from_handle(out_s_w, "quantize_weight_int8:s_w")?;
    Ok((w_i8, s_w))
}

/// W8A8 linear: per-token int8 activation quant + int8 GEMM + rescale -> bf16.
///
/// `x` is `[M, K]` bf16 activations; `w_i8` / `s_w` come from
/// [`quantize_weight_int8`] (the `w_i8` handle is opaque and pre-transposed to
/// the `[K, N]` kernel layout). Returns bf16 `[M, N] = x @ w^T`, lossy only by
/// int8 quantization noise (per-row cosine vs the bf16 reference is ≥ 0.999 on
/// real projection shapes — see the parity test below).
///
/// Stage 4b: the returned array is **lazy** — the C++ op no longer force-evals,
/// so the result composes into the surrounding forward graph (downstream swiglu +
/// down-matmul) and MLX keeps async pipelining/fusion across layers. The caller
/// must `eval` at the end of forward (the normal model loop already does).
///
/// The result is narrowed to bf16 **inside C++** before return, so a downstream
/// bf16 residual add is not promoted to f32 by an f32 scale.
pub fn int8_w8a8_matmul(x: &MxArray, w_i8: &MxArray, s_w: &MxArray) -> Result<MxArray> {
    let mut out: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_w8a8_linear(
            x.as_raw_ptr(),
            w_i8.as_raw_ptr(),
            s_w.as_raw_ptr(),
            &mut out,
        )
    };
    if !ok {
        return Err(Error::from_reason(
            "mlx_w8a8_linear failed (unsupported gen/K or kernel error; see stderr)",
        ));
    }
    MxArray::from_handle(out, "int8_w8a8_matmul")
}

/// MEASUREMENT ONLY (profiler/test scope — NOT a production path).
///
/// Pure int8 `x @ w^T -> int32 [M,N]` with a PRE-TRANSPOSED `[K,N]` weight,
/// isolating the GEMM kernel from the per-call `int8_weight_to_kn` transpose
/// that [`matmul_int8`] pays every iteration. `x` is `[M,K]` bf16/f32 holding
/// integers in `[-127,127]`; `w_kn` is the opaque int8 `[K,N]` operand from
/// [`quantize_weight_int8`] (used directly, no transpose/contiguous/quant).
/// This is the apples-to-apples in-engine analogue of the standalone harness.
#[cfg(test)]
pub fn matmul_int8_kn(x: &MxArray, w_kn: &MxArray) -> Result<MxArray> {
    let mut out: *mut sys::mlx_array = std::ptr::null_mut();
    let ok =
        unsafe { sys::mlx_int8_gemm_pretransposed(x.as_raw_ptr(), w_kn.as_raw_ptr(), &mut out) };
    if !ok {
        return Err(Error::from_reason(
            "mlx_int8_gemm_pretransposed failed (unsupported gen/K or kernel error; see stderr)",
        ));
    }
    MxArray::from_handle(out, "matmul_int8_kn")
}

/// MEASUREMENT ONLY. Same as [`matmul_int8_kn`] but the kernel uses
/// `mode::multiply` (overwrite C) with no MLX `init_value`, so MLX skips the
/// per-call full-output zero fill. Used to isolate the fill cost.
#[cfg(test)]
pub fn matmul_int8_kn_nofill(x: &MxArray, w_kn: &MxArray) -> Result<MxArray> {
    let mut out: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_int8_gemm_pretransposed_nofill(x.as_raw_ptr(), w_kn.as_raw_ptr(), &mut out)
    };
    if !ok {
        return Err(Error::from_reason(
            "mlx_int8_gemm_pretransposed_nofill failed (see stderr)",
        ));
    }
    MxArray::from_handle(out, "matmul_int8_kn_nofill")
}

/// MEASUREMENT ONLY (parity test scope). Runs the FUSED v1 activation-quant
/// kernel. `x` is `[M,K]` bf16; returns `(x_i8_as_i32, s_x)` where the int8
/// quant is widened to int32 `[M,K]` (Rust has no Int8 dtype) and `s_x` is f32
/// `[M,1]`.
#[cfg(test)]
pub fn act_quant_fused(x: &MxArray) -> Result<(MxArray, MxArray)> {
    let mut out_i8: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_sx: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe { sys::mlx_int8_act_quant_fused(x.as_raw_ptr(), &mut out_i8, &mut out_sx) };
    if !ok {
        return Err(Error::from_reason("mlx_int8_act_quant_fused failed"));
    }
    Ok((
        MxArray::from_handle(out_i8, "act_quant_fused:i8")?,
        MxArray::from_handle(out_sx, "act_quant_fused:s_x")?,
    ))
}

/// MEASUREMENT ONLY. The LAZY activation-quant chain (parity reference). Same
/// outputs as [`act_quant_fused`].
#[cfg(test)]
pub fn act_quant_lazy(x: &MxArray) -> Result<(MxArray, MxArray)> {
    let mut out_i8: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_sx: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe { sys::mlx_int8_act_quant_lazy(x.as_raw_ptr(), &mut out_i8, &mut out_sx) };
    if !ok {
        return Err(Error::from_reason("mlx_int8_act_quant_lazy failed"));
    }
    Ok((
        MxArray::from_handle(out_i8, "act_quant_lazy:i8")?,
        MxArray::from_handle(out_sx, "act_quant_lazy:s_x")?,
    ))
}

/// MEASUREMENT ONLY (parity test scope). Runs the FUSED v1 rescale kernel.
/// `acc` is `[M,N]` int32, `s_x` is `[M,1]` f32, `s_w` is `[N]` f32. Returns
/// bf16 `[M,N]`.
#[cfg(test)]
pub fn rescale_fused(acc: &MxArray, s_x: &MxArray, s_w: &MxArray) -> Result<MxArray> {
    let mut out: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_int8_rescale_fused(
            acc.as_raw_ptr(),
            s_x.as_raw_ptr(),
            s_w.as_raw_ptr(),
            &mut out,
        )
    };
    if !ok {
        return Err(Error::from_reason("mlx_int8_rescale_fused failed"));
    }
    MxArray::from_handle(out, "rescale_fused")
}

/// MEASUREMENT ONLY. The LAZY rescale (parity reference). Same I/O as
/// [`rescale_fused`].
#[cfg(test)]
pub fn rescale_lazy(acc: &MxArray, s_x: &MxArray, s_w: &MxArray) -> Result<MxArray> {
    let mut out: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_int8_rescale_lazy(
            acc.as_raw_ptr(),
            s_x.as_raw_ptr(),
            s_w.as_raw_ptr(),
            &mut out,
        )
    };
    if !ok {
        return Err(Error::from_reason("mlx_int8_rescale_lazy failed"));
    }
    MxArray::from_handle(out, "rescale_lazy")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;
    use crate::nn::Activations;

    fn gpu_gen() -> i32 {
        unsafe { sys::mlx_gpu_architecture_gen() }
    }

    /// Deterministic pseudo-random integer in `[lo, hi]` from a linear-congruential
    /// state. Kept fully deterministic so a failure reproduces exactly.
    fn next_int(state: &mut u64, lo: i32, hi: i32) -> i32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let span = (hi - lo + 1) as u64;
        lo + ((*state >> 33) % span) as i32
    }

    /// Build an `[rows, cols]` bf16 MxArray holding the given integer values.
    fn int_array_bf16(vals: &[i32], rows: i64, cols: i64) -> MxArray {
        let f: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
        MxArray::from_float32(&f, &[rows, cols])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
    }

    // ============================ STAGE 1 ============================
    // GATE S1: int32 output BIT-EXACT (integer matmul is deterministic).
    // M ∈ {128,256,512}, K ∈ {2560,9216}, N ∈ {a tile multiple, a non-multiple}.
    // Tile is 128x64, so N=2560 is a multiple of 64; N=2570 is a non-multiple
    // (exercises the edge/tail tile + the contiguous w^T transpose path).
    #[test]
    fn s1_int8_gemm_bit_exact() {
        if gpu_gen() < 17 {
            eprintln!(
                "[s1] SKIP: gpu gen {} < 17 (NA matmul2d needs M5+)",
                gpu_gen()
            );
            return;
        }
        let ms = [128usize, 256, 512];
        let ks = [2560usize, 9216];
        let ns = [2560usize, 2570]; // tile-multiple + non-multiple (edge tile)
        let mut state: u64 = 0x1234_5678_9abc_def0;

        for &m in &ms {
            for &k in &ks {
                for &n in &ns {
                    // x[m,k], w[n,k] in [-127,127].
                    let mut xv = vec![0i32; m * k];
                    for v in xv.iter_mut() {
                        *v = next_int(&mut state, -127, 127);
                    }
                    let mut wv = vec![0i32; n * k];
                    for v in wv.iter_mut() {
                        *v = next_int(&mut state, -127, 127);
                    }

                    let x = int_array_bf16(&xv, m as i64, k as i64);
                    let w = int_array_bf16(&wv, n as i64, k as i64);
                    let out = matmul_int8(&x, &w).unwrap();
                    out.eval();

                    assert_eq!(out.dtype().unwrap(), DType::Int32, "output must be int32");
                    let got = out.to_int32().unwrap();
                    let got: &[i32] = &got;
                    assert_eq!(got.len(), m * n, "size m={m} k={k} n={n}");

                    // i32 reference: ref[m,n] = sum_k x[m,k]*w[n,k]. Values in
                    // [-127,127] over k<=9216 fit comfortably in i32
                    // (127*127*9216 ~ 1.49e8 << 2.1e9).
                    let mut bad = 0usize;
                    let mut first: Option<(usize, i32, i32)> = None;
                    for mi in 0..m {
                        for ni in 0..n {
                            let mut acc: i32 = 0;
                            for ki in 0..k {
                                acc += xv[mi * k + ki] * wv[ni * k + ki];
                            }
                            let g = got[mi * n + ni];
                            if g != acc {
                                bad += 1;
                                if first.is_none() {
                                    first = Some((mi * n + ni, g, acc));
                                }
                            }
                        }
                    }
                    assert_eq!(
                        bad, 0,
                        "NOT bit-exact at M={m} K={k} N={n}: {bad} mismatches, first {first:?}"
                    );
                    eprintln!("[s1] BIT-EXACT M={m} K={k} N={n}");
                }
            }
        }
    }

    // ====================== STAGE 1b (DECISIVE) ======================
    // GATE S1b: int32 output BIT-EXACT on PARTIAL tiles — the one open
    // correctness question for the production `mode::multiply` (overwrite, no
    // output zero-fill) GEMM (`int8_gemm_core_nofill`).
    //
    // `mode::multiply` overwrites C with NO MLX init_value fill, so it is only
    // safe if EVERY in-bounds output element is written exactly once — including
    // when the 128x64 tile overhangs M (M%128!=0) AND N (N%64!=0). The S1 test
    // only covers M in {128,256,512} (all %128==0), so the partial-M tile and the
    // DOUBLE-PARTIAL corner tile (M%128!=0 AND N%64!=0 simultaneously) are
    // untested there. A garbage tail would surface here as a non-bit-exact
    // element in the overhang region.
    //
    // M in {300, 1025} (both %128!=0) x N in {2560 (%64==0), 2570 (%64!=0)}
    // x K in {2560, 9216}. M=1025 ^ N=2570 is the double-partial corner.
    // Same deterministic integer reference as S1.
    #[test]
    fn s1b_int8_gemm_partial_tiles() {
        if gpu_gen() < 17 {
            eprintln!(
                "[s1b] SKIP: gpu gen {} < 17 (NA matmul2d needs M5+)",
                gpu_gen()
            );
            return;
        }
        let ms = [300usize, 1025]; // both M%128 != 0 (partial M tile)
        let ks = [2560usize, 9216];
        let ns = [2560usize, 2570]; // %64==0 and %64!=0 (partial N tile)
        let mut state: u64 = 0xdead_1025_0300_2570;

        for &m in &ms {
            for &k in &ks {
                for &n in &ns {
                    // x[m,k], w[n,k] in [-127,127].
                    let mut xv = vec![0i32; m * k];
                    for v in xv.iter_mut() {
                        *v = next_int(&mut state, -127, 127);
                    }
                    let mut wv = vec![0i32; n * k];
                    for v in wv.iter_mut() {
                        *v = next_int(&mut state, -127, 127);
                    }

                    let x = int_array_bf16(&xv, m as i64, k as i64);
                    let w = int_array_bf16(&wv, n as i64, k as i64);
                    // PRODUCTION path: matmul_int8 -> int8_gemm_core_nofill
                    // (mode::multiply, no zero-fill).
                    let out = matmul_int8(&x, &w).unwrap();
                    out.eval();

                    assert_eq!(out.dtype().unwrap(), DType::Int32, "output must be int32");
                    let got = out.to_int32().unwrap();
                    let got: &[i32] = &got;
                    assert_eq!(got.len(), m * n, "size m={m} k={k} n={n}");

                    // SAME integer reference as S1: ref[m,n] = sum_k x[m,k]*w[n,k].
                    // Parallelized over ROW ranges via std::thread::scope (the per-
                    // element integer math is byte-for-byte identical to the serial
                    // S1 loop; only the iteration is split) so the O(M*N*K) debug
                    // reference for the M=1025/K=9216 corner stays in the seconds,
                    // not minutes. Each thread returns its first-mismatch + count.
                    let nthreads = std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4)
                        .min(m);
                    let chunk = m.div_ceil(nthreads);
                    let (xv_r, wv_r) = (&xv, &wv);
                    let results: Vec<(usize, Option<(usize, i32, i32)>)> =
                        std::thread::scope(|scope| {
                            let mut handles = Vec::with_capacity(nthreads);
                            for t in 0..nthreads {
                                let m_lo = t * chunk;
                                let m_hi = ((t + 1) * chunk).min(m);
                                handles.push(scope.spawn(move || {
                                    let mut bad = 0usize;
                                    let mut first: Option<(usize, i32, i32)> = None;
                                    for mi in m_lo..m_hi {
                                        for ni in 0..n {
                                            let mut acc: i32 = 0;
                                            for ki in 0..k {
                                                acc += xv_r[mi * k + ki] * wv_r[ni * k + ki];
                                            }
                                            let g = got[mi * n + ni];
                                            if g != acc {
                                                bad += 1;
                                                if first.is_none() {
                                                    first = Some((mi * n + ni, g, acc));
                                                }
                                            }
                                        }
                                    }
                                    (bad, first)
                                }));
                            }
                            handles.into_iter().map(|h| h.join().unwrap()).collect()
                        });
                    let bad: usize = results.iter().map(|(b, _)| *b).sum();
                    // First mismatch by lowest flat index across all row chunks.
                    let first: Option<(usize, i32, i32)> = results
                        .iter()
                        .filter_map(|(_, f)| *f)
                        .min_by_key(|(idx, _, _)| *idx);
                    if let Some((idx, g, acc)) = first {
                        eprintln!(
                            "[s1b] MISMATCH M={m} K={k} N={n} at flat={idx} \
                             (mi={},ni={}) got={g} want={acc}",
                            idx / n,
                            idx % n
                        );
                    }
                    let corner = if m % 128 != 0 && n % 64 != 0 {
                        " [DOUBLE-PARTIAL CORNER]"
                    } else {
                        ""
                    };
                    assert_eq!(
                        bad, 0,
                        "NOT bit-exact at M={m} K={k} N={n}{corner}: {bad} mismatches, first {first:?}"
                    );
                    eprintln!("[s1b] BIT-EXACT M={m} K={k} N={n}{corner}");
                }
            }
        }
    }

    // ====================== v1 FUSED-QUANT PARITY ======================
    // GATE: the fused activation-quant kernel (v1 kernel 2) must be BIT-IDENTICAL
    // to the lazy MLX chain it replaces — same int8 bytes AND same f32 s_x.
    // Exercised over the S2 shapes + a couple of MLP-real shapes, with realistic
    // bf16 magnitudes (so the per-row absmax / round / clip paths are all hit).
    #[test]
    fn v1_fused_quant_bit_parity() {
        if gpu_gen() < 17 {
            eprintln!("[v1q] SKIP gpu gen {} < 17", gpu_gen());
            return;
        }
        // (M, K). K must be %16==0. Mix of S2 shapes + MLP-real + a tail M.
        let shapes = [
            (512usize, 2560usize),
            (256, 2560),
            (4096, 2560),
            (4096, 9216),
            (300, 5120),
            (4096, 17408),
        ];
        let mut state: u64 = 0x7e57_0f00_d15e_a5e0;
        for &(m, k) in &shapes {
            // Realistic-ish bf16 activations in ~[-0.2,0.2], plus deliberate
            // outliers so the per-row absmax differs from the bulk.
            let mut xf = vec![0f32; m * k];
            for v in xf.iter_mut() {
                *v = next_int(&mut state, -200, 200) as f32 / 1000.0;
            }
            // Inject one large outlier per row to stress absmax + clip.
            for mi in 0..m {
                let col = (next_int(&mut state, 0, (k - 1) as i32)) as usize;
                xf[mi * k + col] = if mi % 2 == 0 { 1.7 } else { -1.3 };
            }
            let x = MxArray::from_float32(&xf, &[m as i64, k as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            x.eval();

            let (qf, sxf) = act_quant_fused(&x).unwrap();
            let (ql, sxl) = act_quant_lazy(&x).unwrap();
            qf.eval();
            ql.eval();
            sxf.eval();
            sxl.eval();

            // int8 (widened to int32) must be BIT-IDENTICAL.
            let a = qf.to_int32().unwrap();
            let a: &[i32] = &a;
            let b = ql.to_int32().unwrap();
            let b: &[i32] = &b;
            assert_eq!(a.len(), m * k);
            assert_eq!(b.len(), m * k);
            let mut bad = 0usize;
            let mut first: Option<(usize, i32, i32)> = None;
            for i in 0..a.len() {
                if a[i] != b[i] {
                    bad += 1;
                    if first.is_none() {
                        first = Some((i, a[i], b[i]));
                    }
                }
            }
            assert_eq!(
                bad, 0,
                "fused-quant int8 NOT bit-identical at M={m} K={k}: {bad} diffs, first {first:?}"
            );

            // s_x must match exactly (same f32 arithmetic).
            let sa = sxf.to_float32().unwrap();
            let sa: &[f32] = &sa;
            let sb = sxl.to_float32().unwrap();
            let sb: &[f32] = &sb;
            assert_eq!(sa.len(), m);
            let mut bad_sx = 0usize;
            for i in 0..m {
                // Exact: both compute max(absmax,1e-12)/127 in f32 over the same
                // bf16-upcast values. Allow a 0-eps but report any drift.
                if (sa[i] - sb[i]).abs() > 0.0 {
                    bad_sx += 1;
                }
            }
            assert_eq!(
                bad_sx, 0,
                "fused-quant s_x NOT exact at M={m} K={k}: {bad_sx} diffs"
            );
            eprintln!("[v1q] BIT-IDENTICAL M={m} K={k} (int8 + s_x exact)");
        }
    }

    // ====================== v1 FUSED-RESCALE PARITY ======================
    // GATE: the fused int32->bf16 rescale kernel (v1 kernel 3) must match the
    // lazy multi-pass rescale to bf16 EPS (ideally bit-identical — both do
    // (acc*s_x)*s_w in f32 then narrow to bf16). Realistic acc/scale magnitudes.
    #[test]
    fn v1_fused_rescale_parity() {
        if gpu_gen() < 17 {
            eprintln!("[v1r] SKIP gpu gen {} < 17", gpu_gen());
            return;
        }
        let shapes = [
            (512usize, 9216usize),
            (256, 2560),
            (4096, 18432),
            (300, 34816),
            // N % 256 != 0 cases: exercise the PARTIAL threadgroup-in-x of the 2D
            // rescale dispatch (grid.x = N, threadgroup.x = 256). The other N here
            // are all multiples of 256, so without these the partial-x tail (where
            // dispatch_threads launches a sub-256 final group) is untested. 2570
            // and 9300 are both %256 != 0; 2570 also crosses M into a non-round M.
            (300, 2570),
            (1025, 9300),
        ];
        let mut state: u64 = 0x0badf00d_deadbeef;
        for &(m, n) in &shapes {
            // acc int32 in a realistic GEMM-accumulator range.
            let mut accv = vec![0i32; m * n];
            for v in accv.iter_mut() {
                *v = next_int(&mut state, -2_000_000, 2_000_000);
            }
            // s_x [M,1] and s_w [N] f32 in the load-time scale range (~absmax/127).
            let mut sxv = vec![0f32; m];
            for v in sxv.iter_mut() {
                *v = (next_int(&mut state, 1, 4000) as f32) / 1e6;
            }
            let mut swv = vec![0f32; n];
            for v in swv.iter_mut() {
                *v = (next_int(&mut state, 1, 4000) as f32) / 1e6;
            }
            let acc = MxArray::from_int32(&accv, &[m as i64, n as i64]).unwrap();
            let s_x = MxArray::from_float32(&sxv, &[m as i64, 1]).unwrap();
            let s_w = MxArray::from_float32(&swv, &[n as i64]).unwrap();
            acc.eval();
            s_x.eval();
            s_w.eval();

            let yf = rescale_fused(&acc, &s_x, &s_w).unwrap();
            let yl = rescale_lazy(&acc, &s_x, &s_w).unwrap();
            yf.eval();
            yl.eval();
            assert_eq!(yf.dtype().unwrap(), DType::BFloat16);
            assert_eq!(yl.dtype().unwrap(), DType::BFloat16);

            // Compare as raw bf16 bits via f32 readback (both narrowed identically).
            let a = yf.astype(DType::Float32).unwrap().to_float32().unwrap();
            let a: &[f32] = &a;
            let b = yl.astype(DType::Float32).unwrap().to_float32().unwrap();
            let b: &[f32] = &b;
            assert_eq!(a.len(), m * n);
            let mut bad = 0usize;
            let mut max_rel = 0.0f64;
            let mut first: Option<(usize, f32, f32)> = None;
            for i in 0..a.len() {
                let da = a[i] as f64;
                let db = b[i] as f64;
                let denom = db.abs().max(1e-6);
                let rel = (da - db).abs() / denom;
                max_rel = max_rel.max(rel);
                // bf16 has ~8 bits mantissa (~1/256 rel); equal narrowing should
                // be bit-identical, so any 1-ULP slip is at most ~1/256.
                if (da - db).abs() > 0.0 {
                    bad += 1;
                    if first.is_none() {
                        first = Some((i, a[i], b[i]));
                    }
                }
            }
            // Gate to bf16 eps (well under 1 ULP). Report exact-mismatch count too.
            assert!(
                max_rel <= 1.0 / 256.0,
                "fused-rescale beyond bf16 eps at M={m} N={n}: max_rel={max_rel:.6} \
                 ({bad} non-identical, first {first:?})"
            );
            eprintln!(
                "[v1r] M={m} N={n}: max_rel={max_rel:.8} non_identical={bad}/{} (<= bf16 eps)",
                m * n
            );
        }
    }

    // ===================== STAGE 4b RESIDUAL PROFILE =====================
    // Localizes the residual ~18-22% prefill regression after the lazy +
    // load-time-transpose fixes. Times, at the real Qwen3.5-4B MLP shapes and
    // M=4096, the pieces of ONE MLP forward so we can attribute the gap:
    //   * bf16 fused-equivalent (2 matmuls + swiglu) — the BASELINE bar
    //   * int8 full W8A8 MLP (quant+gemm+rescale, both projections)
    //   * activation-quant ONLY (absmax/round/clip/astype) for both projections
    //   * int8 GEMM ONLY (pre-quantized acts, no quant, no rescale)
    // Run explicitly:
    //   cargo test -p mlx-core --lib int8_gemm::tests::profile_residual \
    //     -- --ignored --nocapture
    #[test]
    #[ignore = "manual residual profiler; run with --ignored"]
    fn profile_residual() {
        use crate::array::memory::synchronize;
        use std::time::Instant;
        if gpu_gen() < 17 {
            eprintln!("[profile] SKIP gpu gen {} < 17", gpu_gen());
            return;
        }
        let m: i64 = 4096;
        let hidden: i64 = 2560;
        let inter: i64 = 9216;
        let two_inter = 2 * inter;

        // Random bf16 activation + weights at the real shapes.
        let x = MxArray::random_normal(&[m, hidden], 0.0, 0.05, Some(DType::BFloat16)).unwrap();
        // gate_up weight [2*inter, hidden] (N,K); down weight [hidden, inter].
        let w_gu =
            MxArray::random_normal(&[two_inter, hidden], 0.0, 0.02, Some(DType::BFloat16)).unwrap();
        let w_d =
            MxArray::random_normal(&[hidden, inter], 0.0, 0.02, Some(DType::BFloat16)).unwrap();
        // bf16 transposed weights for the matmul baseline.
        let w_gu_t = w_gu.transpose(Some(&[1, 0])).unwrap();
        let w_d_t = w_d.transpose(Some(&[1, 0])).unwrap();
        w_gu_t.eval();
        w_d_t.eval();
        // int8 pre-quantized weights (load-time form).
        let (gu_i8, gu_s) = quantize_weight_int8(&w_gu).unwrap();
        let (d_i8, d_s) = quantize_weight_int8(&w_d).unwrap();
        gu_i8.eval();
        gu_s.eval();
        d_i8.eval();
        d_s.eval();
        x.eval();
        synchronize();

        let iters = 50;
        let warm = 10;

        // ---- (A) bf16 fused-equivalent: 2 matmuls + swiglu (silu*up) ----
        let bf16_mlp = || {
            let gate_up = x.matmul(&w_gu_t).unwrap(); // [M, 2*inter]
            let gate = gate_up.slice(&[0, 0], &[m, inter]).unwrap();
            let up = gate_up.slice(&[0, inter], &[m, two_inter]).unwrap();
            let gated = Activations::silu(&gate).unwrap().mul(&up).unwrap();
            gated.matmul(&w_d_t).unwrap()
        };
        // ---- (B) int8 full W8A8 MLP ----
        let int8_mlp = || {
            let gate_up = int8_w8a8_matmul(&x, &gu_i8, &gu_s).unwrap();
            let gate = gate_up.slice(&[0, 0], &[m, inter]).unwrap();
            let up = gate_up.slice(&[0, inter], &[m, two_inter]).unwrap();
            let gated = Activations::silu(&gate).unwrap().mul(&up).unwrap();
            int8_w8a8_matmul(&gated, &d_i8, &d_s).unwrap()
        };

        let bench = |label: &str, f: &dyn Fn() -> MxArray| {
            for _ in 0..warm {
                let o = f();
                o.eval();
            }
            synchronize();
            let t = Instant::now();
            for _ in 0..iters {
                let o = f();
                o.eval();
            }
            synchronize();
            let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
            eprintln!("[profile] {label:<28} {ms:8.3} ms/iter");
            ms
        };

        eprintln!(
            "[profile] M={m} hidden={hidden} inter={inter} (gate_up N={two_inter}, down N={hidden})"
        );
        // ---- (C) per-token activation quant ONLY, for the gate_up input ----
        // Mirrors mlx_w8a8_linear's quant block: absmax/round/clip/astype.
        // We emulate it in Rust ops over x[M,hidden] so we measure the same
        // arithmetic the C++ op builds into the graph.
        let quant_only_gu = || {
            let xf = x.astype(DType::Float32).unwrap();
            let absmax = xf.abs().unwrap().max(Some(&[1]), Some(true)).unwrap(); // [M,1]
            let sx = absmax.div_scalar(127.0).unwrap();
            let xq = xf.div(&sx).unwrap().round().unwrap();
            let xq = xq.clip(Some(-127.0), Some(127.0)).unwrap();
            xq.astype(DType::BFloat16).unwrap() // proxy for int8 cast (no Int8 dtype)
        };
        // ---- (D) int8 GEMM core ONLY (raw matmul_int8 on pre-int8 acts) ----
        // x already in [-127,127] integer range so values cast cleanly.
        let x_int = x
            .div_scalar(0.05)
            .unwrap()
            .round()
            .unwrap()
            .clip(Some(-127.0), Some(127.0))
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let w_gu_int = w_gu
            .div_scalar(0.02)
            .unwrap()
            .round()
            .unwrap()
            .clip(Some(-127.0), Some(127.0))
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        x_int.eval();
        w_gu_int.eval();
        synchronize();
        let int8_gemm_gu = || matmul_int8(&x_int, &w_gu_int).unwrap();
        // ---- (E) bf16 matmul ONLY at the gate_up shape (the bar the GEMM beats) ----
        let bf16_gemm_gu = || x.matmul(&w_gu_t).unwrap();

        let t_bf16 = bench("A bf16 fused MLP", &bf16_mlp);
        let t_int8 = bench("B int8 W8A8 MLP", &int8_mlp);
        let t_quant = bench("C act-quant only (gate_up)", &quant_only_gu);
        let t_i8gemm = bench("D int8 GEMM only (gate_up)", &int8_gemm_gu);
        let t_bf16gemm = bench("E bf16 GEMM only (gate_up)", &bf16_gemm_gu);
        eprintln!(
            "[profile] int8/bf16 MLP ratio = {:.3} (>1 = int8 slower)",
            t_int8 / t_bf16
        );
        eprintln!(
            "[profile] bf16/int8 prefill-tps-equiv = {:.3} (matches harness prefill ratio)",
            t_bf16 / t_int8
        );
        eprintln!(
            "[profile] gate_up GEMM: int8={t_i8gemm:.3}ms bf16={t_bf16gemm:.3}ms \
             int8/bf16={:.3} (kernel-only; <1 = int8 GEMM faster)",
            t_i8gemm / t_bf16gemm
        );
        eprintln!(
            "[profile] act-quant (gate_up) = {t_quant:.3} ms; as %% of one int8 GEMM = {:.1}%%",
            100.0 * t_quant / t_i8gemm
        );
    }

    // ========================= v1 FUSED MLP PROFILE =========================
    // Reports the v1 fused int8 W8A8 MLP vs bf16 fused MLP wall ratio at M=4096
    // for BOTH the 4B and 27B MLP shapes, with the per-piece breakdown
    // (fused-quant ms / GEMM ms / fused-rescale ms / swiglu ms). The bf16
    // baseline thermally throttles, so we run the int8-vs-bf16 comparison 3x and
    // ratio the (thermally stable) int8 MLP against the COOLEST bf16 sample.
    //
    // Run:
    //   cargo test -p mlx-core --lib int8_gemm::tests::profile_fused \
    //     -- --ignored --nocapture
    #[test]
    #[ignore = "manual v1 fused MLP profiler; run with --ignored"]
    fn profile_fused() {
        use super::{act_quant_fused, matmul_int8_kn_nofill, rescale_fused};
        use crate::array::memory::synchronize;
        use std::time::Instant;
        if gpu_gen() < 17 {
            eprintln!("[fused] SKIP gpu gen {} < 17", gpu_gen());
            return;
        }
        let iters = 50;
        let warm = 12;

        let bench = |f: &dyn Fn() -> MxArray| -> f64 {
            for _ in 0..warm {
                let o = f();
                o.eval();
            }
            synchronize();
            let t = Instant::now();
            for _ in 0..iters {
                let o = f();
                o.eval();
            }
            synchronize();
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        };

        let run = |label: &str, m: i64, hidden: i64, inter: i64| {
            let two_inter = 2 * inter;
            // bf16 activations + weights at realistic magnitudes.
            let x = MxArray::random_normal(&[m, hidden], 0.0, 0.05, Some(DType::BFloat16)).unwrap();
            let w_gu =
                MxArray::random_normal(&[two_inter, hidden], 0.0, 0.02, Some(DType::BFloat16))
                    .unwrap();
            let w_d =
                MxArray::random_normal(&[hidden, inter], 0.0, 0.02, Some(DType::BFloat16)).unwrap();
            let w_gu_t = w_gu.transpose(Some(&[1, 0])).unwrap();
            let w_d_t = w_d.transpose(Some(&[1, 0])).unwrap();
            w_gu_t.eval();
            w_d_t.eval();
            let (gu_i8, gu_s) = quantize_weight_int8(&w_gu).unwrap();
            let (d_i8, d_s) = quantize_weight_int8(&w_d).unwrap();
            gu_i8.eval();
            gu_s.eval();
            d_i8.eval();
            d_s.eval();
            x.eval();
            synchronize();

            // ---- Full MLP: bf16 fused vs int8 W8A8 (production path) ----
            let bf16_mlp = || {
                let gate_up = x.matmul(&w_gu_t).unwrap();
                let gate = gate_up.slice(&[0, 0], &[m, inter]).unwrap();
                let up = gate_up.slice(&[0, inter], &[m, two_inter]).unwrap();
                let gated = Activations::silu(&gate).unwrap().mul(&up).unwrap();
                gated.matmul(&w_d_t).unwrap()
            };
            let int8_mlp = || {
                let gate_up = int8_w8a8_matmul(&x, &gu_i8, &gu_s).unwrap();
                let gate = gate_up.slice(&[0, 0], &[m, inter]).unwrap();
                let up = gate_up.slice(&[0, inter], &[m, two_inter]).unwrap();
                let gated = Activations::silu(&gate).unwrap().mul(&up).unwrap();
                int8_w8a8_matmul(&gated, &d_i8, &d_s).unwrap()
            };

            // 3 runs; int8 is thermally stable, bf16 throttles -> use coolest bf16.
            let mut bf16_runs = [0.0f64; 3];
            let mut int8_runs = [0.0f64; 3];
            for r in 0..3 {
                bf16_runs[r] = bench(&bf16_mlp);
                int8_runs[r] = bench(&int8_mlp);
            }
            let bf16_cool = bf16_runs.iter().cloned().fold(f64::INFINITY, f64::min);
            let int8_med = {
                let mut v = int8_runs;
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[1]
            };
            let int8_cool = int8_runs.iter().cloned().fold(f64::INFINITY, f64::min);

            // ---- Per-piece breakdown (gate_up shape: K=hidden, N=two_inter) ----
            // Fused activation-quant (the [M,hidden] input).
            let t_quant = bench(&|| {
                let (q, _s) = act_quant_fused(&x).unwrap();
                q
            });
            // Build a pre-int8 [M,hidden] operand + the [K,N] weight for the GEMM.
            let (x_i8_i32, sx) = act_quant_fused(&x).unwrap();
            let x_i8_bf16 = x_i8_i32.astype(DType::BFloat16).unwrap(); // int-valued
            x_i8_bf16.eval();
            sx.eval();
            // gu_i8 is the opaque [K,N] int8 kernel operand from load.
            let t_gemm = bench(&|| matmul_int8_kn_nofill(&x_i8_bf16, &gu_i8).unwrap());
            // Fused rescale on an int32 acc of the gate_up shape.
            let acc = matmul_int8_kn_nofill(&x_i8_bf16, &gu_i8).unwrap();
            acc.eval();
            let t_rescale = bench(&|| rescale_fused(&acc, &sx, &gu_s).unwrap());
            // swiglu (silu(gate)*up) over the gate_up output [M, two_inter].
            let gate_up_bf16 =
                MxArray::random_normal(&[m, two_inter], 0.0, 0.1, Some(DType::BFloat16)).unwrap();
            gate_up_bf16.eval();
            let t_swiglu = bench(&|| {
                let gate = gate_up_bf16.slice(&[0, 0], &[m, inter]).unwrap();
                let up = gate_up_bf16.slice(&[0, inter], &[m, two_inter]).unwrap();
                Activations::silu(&gate).unwrap().mul(&up).unwrap()
            });

            eprintln!(
                "[fused] === {label}: M={m} hidden={hidden} inter={inter} \
                 (gate_up N={two_inter} K={hidden}; down N={hidden} K={inter}) ==="
            );
            eprintln!(
                "[fused] bf16 MLP runs (ms): {:.3} {:.3} {:.3}  -> coolest {bf16_cool:.3}",
                bf16_runs[0], bf16_runs[1], bf16_runs[2]
            );
            eprintln!(
                "[fused] int8 MLP runs (ms): {:.3} {:.3} {:.3}  -> median {int8_med:.3} coolest {int8_cool:.3}",
                int8_runs[0], int8_runs[1], int8_runs[2]
            );
            eprintln!(
                "[fused] RATIO int8/bf16 (vs coolest bf16): median={:.3} coolest={:.3}  (<1.0 = int8 FASTER)",
                int8_med / bf16_cool,
                int8_cool / bf16_cool
            );
            eprintln!(
                "[fused] per-piece (gate_up shape): fused-quant={t_quant:.3}ms  \
                 GEMM={t_gemm:.3}ms  fused-rescale={t_rescale:.3}ms  swiglu={t_swiglu:.3}ms"
            );
        };

        eprintln!("[fused] === v1 FUSED int8 W8A8 MLP vs bf16, M=4096 ===");
        // 4B: hidden=2560, inter=9216.
        run("4B", 4096, 2560, 9216);
        // 27B: hidden=5120, inter=17408.
        run("27B", 4096, 5120, 17408);
    }

    // =================== CLEAN PURE-GEMM THROUGHPUT PROFILE ===================
    // DIAGNOSTIC (measurement only). Times the in-engine int8 GEMM with a
    // PRE-TRANSPOSED [K,N] weight (zero per-call transpose) vs bf16 matmul at the
    // real Qwen3.5-4B MLP shapes, M in {512,4096}. Also times the OLD transpose-
    // contaminated matmul_int8 to quantify the contamination delta. Reports
    // absolute TOPS/TFLOPs = 2*M*N*K / sec / 1e12 + ratios.
    //
    // Run:
    //   cargo test -p mlx-core --lib int8_gemm::tests::profile_clean_gemm \
    //     -- --ignored --nocapture
    #[test]
    #[ignore = "manual clean pure-GEMM throughput profiler; run with --ignored"]
    fn profile_clean_gemm() {
        use super::{matmul_int8_kn, matmul_int8_kn_nofill};
        use crate::array::memory::synchronize;
        use std::time::Instant;
        if gpu_gen() < 17 {
            eprintln!("[clean] SKIP gpu gen {} < 17", gpu_gen());
            return;
        }

        // ---- bit-exact cross-check: matmul_int8_kn == matmul_int8 (small case) ----
        // Confirms removing the per-call transpose did not break the math: the
        // pre-transposed [K,N] weight must equal int8_weight_to_kn(w).
        {
            let (m, k, n) = (128usize, 256usize, 128usize);
            let mut state: u64 = 0xabcd_1234_5678_9999;
            let mut xv = vec![0i32; m * k];
            for v in xv.iter_mut() {
                *v = next_int(&mut state, -127, 127);
            }
            let mut wv = vec![0i32; n * k];
            for v in wv.iter_mut() {
                *v = next_int(&mut state, -127, 127);
            }
            let x = int_array_bf16(&xv, m as i64, k as i64); // [M,K]
            let w = int_array_bf16(&wv, n as i64, k as i64); // [N,K]
            // Old contaminated path (transposes w internally).
            let out_old = matmul_int8(&x, &w).unwrap();
            out_old.eval();
            // Pre-transpose w -> [K,N] contiguous via quantize? No — that rescales.
            // Build the [K,N] int-valued operand directly: transpose then force
            // contiguity by a round-trip through from_float32 (guaranteed C-order),
            // so matmul_int8_kn casts a genuinely row-contiguous [K,N] buffer.
            let mut wkn = vec![0f32; k * n]; // [K,N]: wkn[k*N + n] = w[n,k]
            for ni in 0..n {
                for ki in 0..k {
                    wkn[ki * n + ni] = wv[ni * k + ki] as f32;
                }
            }
            let w_kn = MxArray::from_float32(&wkn, &[k as i64, n as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            w_kn.eval();
            let out_new = matmul_int8_kn(&x, &w_kn).unwrap();
            out_new.eval();
            let a = out_old.to_int32().unwrap();
            let a: &[i32] = &a;
            let b = out_new.to_int32().unwrap();
            let b: &[i32] = &b;
            assert_eq!(a.len(), b.len());
            let mut bad = 0usize;
            for i in 0..a.len() {
                if a[i] != b[i] {
                    bad += 1;
                }
            }
            assert_eq!(
                bad, 0,
                "matmul_int8_kn NOT bit-exact vs matmul_int8: {bad} diffs"
            );
            eprintln!(
                "[clean] cross-check BIT-EXACT: matmul_int8_kn == matmul_int8 (M={m} K={k} N={n})"
            );
            // nofill (mode::multiply) must also be bit-exact (overwrite, not accum).
            let out_nf = matmul_int8_kn_nofill(&x, &w_kn).unwrap();
            out_nf.eval();
            let c = out_nf.to_int32().unwrap();
            let c: &[i32] = &c;
            let mut bad2 = 0usize;
            for i in 0..a.len() {
                if a[i] != c[i] {
                    bad2 += 1;
                }
            }
            assert_eq!(bad2, 0, "matmul_int8_kn_nofill NOT bit-exact: {bad2} diffs");
            eprintln!("[clean] cross-check BIT-EXACT: matmul_int8_kn_nofill == matmul_int8");
        }

        let iters = 50;
        let warm = 10;

        // Generic timed comparison for one (M,K,N) shape.
        // Builds: x[M,K] bf16-int, bf16 weight w[N,K] + w^T[K,N], pre-transposed
        // int8-valued [K,N] operand (materialized contiguous, evaled ONCE).
        let run_shape = |label: &str, m: usize, k: usize, n: usize| {
            let mut state: u64 = 0x5151_a7a7_3939_c0c0 ^ ((m as u64) << 40) ^ ((n as u64) << 8);
            // x[M,K] integers in [-127,127] as bf16.
            let mut xv = vec![0i32; m * k];
            for v in xv.iter_mut() {
                *v = next_int(&mut state, -127, 127);
            }
            let x = int_array_bf16(&xv, m as i64, k as i64); // bf16 [M,K]

            // w[N,K] integers as bf16 (for matmul_int8 contaminated path + bf16 ref).
            let mut wv = vec![0i32; n * k];
            for v in wv.iter_mut() {
                *v = next_int(&mut state, -127, 127);
            }
            let w_nk = int_array_bf16(&wv, n as i64, k as i64); // bf16 [N,K]
            let w_t = w_nk.transpose(Some(&[1, 0])).unwrap(); // [K,N] (strided view)
            // Force the bf16 baseline weight contiguous (matches a stored w^T).
            let w_t = w_t.astype(DType::BFloat16).unwrap();

            // Pre-transposed int8-valued [K,N] operand, materialized row-contiguous.
            let mut wkn = vec![0f32; k * n];
            for ni in 0..n {
                for ki in 0..k {
                    wkn[ki * n + ni] = wv[ni * k + ki] as f32;
                }
            }
            let w_kn = MxArray::from_float32(&wkn, &[k as i64, n as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();

            x.eval();
            w_nk.eval();
            w_t.eval();
            w_kn.eval();
            synchronize();

            let bench = |f: &dyn Fn() -> MxArray| -> f64 {
                for _ in 0..warm {
                    let o = f();
                    o.eval();
                }
                synchronize();
                let t = Instant::now();
                for _ in 0..iters {
                    let o = f();
                    o.eval();
                }
                synchronize();
                t.elapsed().as_secs_f64() * 1e3 / iters as f64
            };

            // Clean pure int8 GEMM (pre-transposed weight, no per-call transpose).
            let clean = || matmul_int8_kn(&x, &w_kn).unwrap();
            // Clean int8 GEMM WITHOUT the per-call MLX zero-fill (mode::multiply).
            let nofill = || matmul_int8_kn_nofill(&x, &w_kn).unwrap();
            // bf16 matmul at identical logical shape: [M,K] @ [K,N].
            let bf16 = || x.matmul(&w_t).unwrap();
            // Contaminated: matmul_int8 transposes w[N,K]->[K,N] every call.
            let dirty = || matmul_int8(&x, &w_nk).unwrap();

            let ms_clean = bench(&clean);
            let ms_nofill = bench(&nofill);
            let ms_bf16 = bench(&bf16);
            let ms_dirty = bench(&dirty);

            // TOPS / TFLOPs = 2*M*N*K / sec / 1e12.
            let flop = 2.0 * m as f64 * n as f64 * k as f64;
            let tops_clean = flop / (ms_clean / 1e3) / 1e12;
            let tops_nofill = flop / (ms_nofill / 1e3) / 1e12;
            let tflops_bf16 = flop / (ms_bf16 / 1e3) / 1e12;
            let tops_dirty = flop / (ms_dirty / 1e3) / 1e12;

            eprintln!(
                "[clean] {label:<20} M={m:<5} N={n:<6} K={k:<5} | \
                 int8(clean)={tops_clean:6.1} TOPS ({ms_clean:.3}ms)  \
                 bf16={tflops_bf16:6.1} TF ({ms_bf16:.3}ms)  \
                 ratio(int8/bf16)={:.3}  wall(bf16/int8)={:.3} || \
                 int8(nofill)={tops_nofill:6.1} TOPS ({ms_nofill:.3}ms) \
                 fill_delta={:.3}ms (nofill/bf16={:.3}) || \
                 int8(dirty)={tops_dirty:6.1} TOPS ({ms_dirty:.3}ms) \
                 contam_delta={:.3}ms ({:.2}x)",
                tops_clean / tflops_bf16,
                ms_bf16 / ms_clean,
                ms_clean - ms_nofill,
                tops_nofill / tflops_bf16,
                ms_dirty - ms_clean,
                ms_dirty / ms_clean,
            );
        };

        eprintln!("[clean] === pure-GEMM throughput, M5 NA int8 vs bf16 ===");
        // M=512 first (compare to standalone M=512), then M=4096.
        for &m in &[512usize, 4096usize] {
            // gate_up: x[M,2560] @ w[18432,2560]^T -> N=18432, K=2560.
            run_shape("gate_up", m, 2560, 18432);
            // down: x[M,9216] @ w[2560,9216]^T -> N=2560, K=9216.
            run_shape("down", m, 9216, 2560);
        }
    }

    // ==================== GDN in_proj_qkvz PARITY ====================
    // GATE (qkvz int8 wiring): per-ROW cosine >= 0.999 of the int8 W8A8 qkvz
    // output vs the bf16 `x @ w_qkvz^T` reference, at realistic GDN shapes.
    // qkvz feeds the GDN conv + recurrence so accuracy is load-bearing.
    //
    // Shapes (K=hidden must be %16==0):
    //   * 4B : hidden=2560, qkvz_dim = key_dim*2 + value_dim*2
    //          = (16*128)*2 + (32*128)*2 = 4096 + 8192 = 12288
    //   * 27B: hidden=5120, same head config -> qkvz_dim=12288
    // M=512 (a realistic prefill tile).
    #[test]
    fn qkvz_w8a8_cosine_parity() {
        if gpu_gen() < 17 {
            eprintln!(
                "[qkvz] SKIP: gpu gen {} < 17 (NA matmul2d needs M5+)",
                gpu_gen()
            );
            return;
        }
        // (M, K=hidden, N=qkvz_dim)
        let shapes = [(512usize, 2560usize, 12288usize), (512, 5120, 12288)];
        let mut state: u64 = 0xb16e_5e7e_4242_d00d;

        for &(m, k, n) in &shapes {
            // bf16 activations + weights with realistic small magnitudes.
            let mut xf = vec![0f32; m * k];
            for v in xf.iter_mut() {
                *v = next_int(&mut state, -200, 200) as f32 / 1000.0;
            }
            let mut wf = vec![0f32; n * k];
            for v in wf.iter_mut() {
                *v = next_int(&mut state, -200, 200) as f32 / 1000.0;
            }

            let x = MxArray::from_float32(&xf, &[m as i64, k as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            // w_qkvz is [N=qkvz_dim, K=hidden], exactly quantize_weight_int8's [N,K].
            let w = MxArray::from_float32(&wf, &[n as i64, k as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();

            let (w_i8, s_w) = quantize_weight_int8(&w).unwrap();
            let y = int8_w8a8_matmul(&x, &w_i8, &s_w).unwrap();
            y.eval();
            assert_eq!(
                y.dtype().unwrap(),
                DType::BFloat16,
                "qkvz W8A8 output must be bf16"
            );

            // FP32-accumulated reference: y_ref = x @ w_qkvz^T. A bf16 matmul of
            // this synthetic uniform-random data over large K suffers catastrophic
            // CANCELLATION; the int8 path uses EXACT int32 accumulation, so the
            // bf16 matmul (not int8) is what loses the signal. Upcast to f32 for a
            // faithful gate (int8 matches the f32 reference at ~0.99998).
            let wt = w
                .astype(DType::Float32)
                .unwrap()
                .transpose(Some(&[1, 0]))
                .unwrap();
            let y_ref = x.astype(DType::Float32).unwrap().matmul(&wt).unwrap();
            y_ref.eval();

            let got = y.astype(DType::Float32).unwrap().to_float32().unwrap();
            let got: &[f32] = &got;
            let refv = y_ref.astype(DType::Float32).unwrap().to_float32().unwrap();
            let refv: &[f32] = &refv;
            assert_eq!(got.len(), m * n);
            assert_eq!(refv.len(), m * n);

            let mut min_cos = f64::INFINITY;
            let mut sum_cos = 0.0f64;
            for mi in 0..m {
                let mut dot = 0.0f64;
                let mut na = 0.0f64;
                let mut nb = 0.0f64;
                for ni in 0..n {
                    let a = got[mi * n + ni] as f64;
                    let b = refv[mi * n + ni] as f64;
                    dot += a * b;
                    na += a * a;
                    nb += b * b;
                }
                let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
                let cos = dot / denom;
                min_cos = min_cos.min(cos);
                sum_cos += cos;
            }
            let mean_cos = sum_cos / m as f64;
            eprintln!(
                "[qkvz] hidden={k} qkvz_dim={n} M={m}: min_row_cos={min_cos:.6} mean_row_cos={mean_cos:.6}"
            );
            assert!(
                min_cos >= 0.999,
                "qkvz W8A8 per-row cosine below gate at hidden={k} qkvz_dim={n}: min={min_cos:.6}"
            );
        }
    }

    // ==================== GDN in_proj_qkvz MICROBENCH ====================
    // Reports the int8/bf16 wall ratio of the qkvz projection at prefill M=4096
    // for the 4B (hidden=2560) and 27B (hidden=5120) shapes (qkvz_dim=12288).
    // int8 is thermally stable; bf16 throttles -> ratio int8(median) vs the
    // COOLEST bf16 sample (matches profile_fused's methodology).
    //
    // Run:
    //   cargo test -p mlx-core --lib int8_gemm::tests::profile_qkvz \
    //     -- --ignored --nocapture
    #[test]
    #[ignore = "manual GDN qkvz int8 microbench; run with --ignored"]
    fn profile_qkvz() {
        use crate::array::memory::synchronize;
        use std::time::Instant;
        if gpu_gen() < 17 {
            eprintln!("[qkvz-bench] SKIP gpu gen {} < 17", gpu_gen());
            return;
        }
        let iters = 50;
        let warm = 12;

        let bench = |f: &dyn Fn() -> MxArray| -> f64 {
            for _ in 0..warm {
                let o = f();
                o.eval();
            }
            synchronize();
            let t = Instant::now();
            for _ in 0..iters {
                let o = f();
                o.eval();
            }
            synchronize();
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        };

        let run = |label: &str, m: i64, hidden: i64, qkvz_dim: i64| {
            // bf16 activations + qkvz weight at realistic magnitudes.
            let x = MxArray::random_normal(&[m, hidden], 0.0, 0.05, Some(DType::BFloat16)).unwrap();
            // w_qkvz [N=qkvz_dim, K=hidden].
            let w = MxArray::random_normal(&[qkvz_dim, hidden], 0.0, 0.02, Some(DType::BFloat16))
                .unwrap();
            // bf16 transposed weight [K,N] for the matmul baseline (the E51 stacked
            // path's qkvz matmul; we time qkvz alone since ba is unchanged).
            let w_t = w.transpose(Some(&[1, 0])).unwrap();
            w_t.eval();
            let (qkvz_i8, qkvz_s) = quantize_weight_int8(&w).unwrap();
            qkvz_i8.eval();
            qkvz_s.eval();
            x.eval();
            synchronize();

            let bf16_qkvz = || x.matmul(&w_t).unwrap();
            let int8_qkvz = || int8_w8a8_matmul(&x, &qkvz_i8, &qkvz_s).unwrap();

            // 3 runs; int8 stable, bf16 throttles -> use coolest bf16.
            let mut bf16_runs = [0.0f64; 3];
            let mut int8_runs = [0.0f64; 3];
            for r in 0..3 {
                bf16_runs[r] = bench(&bf16_qkvz);
                int8_runs[r] = bench(&int8_qkvz);
            }
            let bf16_cool = bf16_runs.iter().cloned().fold(f64::INFINITY, f64::min);
            let int8_med = {
                let mut v = int8_runs;
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[1]
            };
            let int8_cool = int8_runs.iter().cloned().fold(f64::INFINITY, f64::min);

            eprintln!(
                "[qkvz-bench] === {label}: M={m} hidden={hidden} qkvz_dim={qkvz_dim} \
                 (N={qkvz_dim} K={hidden}) ==="
            );
            eprintln!(
                "[qkvz-bench] bf16 qkvz runs (ms): {:.3} {:.3} {:.3}  -> coolest {bf16_cool:.3}",
                bf16_runs[0], bf16_runs[1], bf16_runs[2]
            );
            eprintln!(
                "[qkvz-bench] int8 qkvz runs (ms): {:.3} {:.3} {:.3}  -> median {int8_med:.3} coolest {int8_cool:.3}",
                int8_runs[0], int8_runs[1], int8_runs[2]
            );
            eprintln!(
                "[qkvz-bench] RATIO int8/bf16 (vs coolest bf16): median={:.3} coolest={:.3}  (<1.0 = int8 FASTER)",
                int8_med / bf16_cool,
                int8_cool / bf16_cool
            );
        };

        eprintln!("[qkvz-bench] === GDN in_proj_qkvz int8 W8A8 vs bf16, M=4096 ===");
        // 4B: hidden=2560, qkvz_dim=12288.
        run("4B", 4096, 2560, 12288);
        // 27B: hidden=5120, qkvz_dim=12288.
        run("27B", 4096, 5120, 12288);
    }

    // ============================ STAGE 2 ============================
    // GATE S2: per-ROW cosine >= 0.999 on a real projection shape.
    // x[M=512, hidden=2560] @ w[N=intermediate, K=2560]^T, weight quantized once.
    #[test]
    fn s2_w8a8_cosine_parity() {
        if gpu_gen() < 17 {
            eprintln!(
                "[s2] SKIP: gpu gen {} < 17 (NA matmul2d needs M5+)",
                gpu_gen()
            );
            return;
        }
        // Real-ish projection shapes (K must be % 16 == 0).
        // (M, K=hidden, N=intermediate)
        let shapes = [(512usize, 2560usize, 9216usize), (256, 2560, 2560)];
        let mut state: u64 = 0x0fed_cba9_8765_4321;

        for &(m, k, n) in &shapes {
            // bf16 activations and weights with realistic magnitudes (~N(0, 0.05)
            // emulated via small integers / 1000 -> values in ~[-0.127,0.127]).
            let mut xf = vec![0f32; m * k];
            for v in xf.iter_mut() {
                *v = next_int(&mut state, -200, 200) as f32 / 1000.0;
            }
            let mut wf = vec![0f32; n * k];
            for v in wf.iter_mut() {
                *v = next_int(&mut state, -200, 200) as f32 / 1000.0;
            }

            let x = MxArray::from_float32(&xf, &[m as i64, k as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            let w = MxArray::from_float32(&wf, &[n as i64, k as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();

            // Quantize weight ONCE, then run the W8A8 path.
            let (w_i8, s_w) = quantize_weight_int8(&w).unwrap();
            let y = int8_w8a8_matmul(&x, &w_i8, &s_w).unwrap();
            y.eval();
            assert_eq!(
                y.dtype().unwrap(),
                DType::BFloat16,
                "W8A8 output must be bf16"
            );

            // FP32-accumulated reference: y_ref = x @ w^T (x[M,K] @ w^T[K,N]).
            // A *bf16* matmul of this synthetic uniform-random data over large K
            // suffers catastrophic CANCELLATION (the sum is a near-zero residual
            // of large cancelling terms). The int8 path accumulates in EXACT
            // int32 and narrows once, so it is the bf16 matmul — NOT int8 — that
            // loses the signal: a bf16 reference scores cosine ~-0.03 vs ground
            // truth while int8 matches the f32 reference at ~0.99998. Upcasting
            // both operands to f32 gives a faithful gate.
            let wt = w
                .astype(DType::Float32)
                .unwrap()
                .transpose(Some(&[1, 0]))
                .unwrap();
            let y_ref = x.astype(DType::Float32).unwrap().matmul(&wt).unwrap();
            y_ref.eval();

            let got = y.astype(DType::Float32).unwrap().to_float32().unwrap();
            let got: &[f32] = &got;
            let refv = y_ref.astype(DType::Float32).unwrap().to_float32().unwrap();
            let refv: &[f32] = &refv;
            assert_eq!(got.len(), m * n);
            assert_eq!(refv.len(), m * n);

            // Per-row cosine similarity.
            let mut min_cos = f64::INFINITY;
            let mut sum_cos = 0.0f64;
            for mi in 0..m {
                let mut dot = 0.0f64;
                let mut na = 0.0f64;
                let mut nb = 0.0f64;
                for ni in 0..n {
                    let a = got[mi * n + ni] as f64;
                    let b = refv[mi * n + ni] as f64;
                    dot += a * b;
                    na += a * a;
                    nb += b * b;
                }
                let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
                let cos = dot / denom;
                min_cos = min_cos.min(cos);
                sum_cos += cos;
            }
            let mean_cos = sum_cos / m as f64;
            eprintln!(
                "[s2] M={m} K={k} N={n}: min_row_cos={min_cos:.6} mean_row_cos={mean_cos:.6}"
            );
            assert!(
                min_cos >= 0.999,
                "W8A8 per-row cosine below gate at M={m} K={k} N={n}: min={min_cos:.6}"
            );
        }
    }
}
