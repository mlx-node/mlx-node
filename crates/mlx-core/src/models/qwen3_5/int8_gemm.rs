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

/// LOAD-time sym8 kernel-operand builder (runs once per sym8 linear).
///
/// `w_i8_nk` is the STORED checkpoint weight — int8 `[N,K]`, source
/// orientation, as emitted by `mlx convert --q-mode sym8`
/// (`mlx_sym8_quantize_store`). Returns the opaque contiguous `[K,N]` int8
/// kernel operand that [`int8_w8a8_matmul`] / [`int8_w8a8_qmv`] consume —
/// the EXACT transpose+contiguous tail of [`quantize_weight_int8`], minus
/// the quantization. The checkpoint already holds the quantized values, so
/// this is requant-free: bit-exact with what `quantize_weight_int8` would
/// have produced from the original float weight at convert time.
///
/// Fail-loud: `Err` on non-2D / non-int8 input (corrupt sym8 checkpoint) or
/// FFI failure. The result is eval'd C++-side, so the transpose copy is
/// materialized ONCE at load, never per forward.
pub fn sym8_kernel_operand(w_i8_nk: &MxArray) -> Result<MxArray> {
    let mut out: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe { sys::mlx_sym8_kernel_operand(w_i8_nk.as_raw_ptr(), &mut out) };
    if !ok {
        return Err(Error::from_reason(
            "mlx_sym8_kernel_operand failed (expected int8 [N,K] weight; see stderr)",
        ));
    }
    MxArray::from_handle(out, "sym8_kernel_operand")
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

/// sym8 DECODE matvec (QMV): the small-M analogue of [`int8_w8a8_matmul`].
///
/// `x` is `[M,K]` bf16 activations (decode `M` is small, 1..~16); `w_i8` / `s_w`
/// come from [`quantize_weight_int8`] (the `w_i8` handle is the opaque
/// pre-transposed `[K,N]` kernel layout, EXACTLY the operand
/// [`int8_w8a8_matmul`] consumes). Returns bf16 `[M,N] = x @ w^T`.
///
/// Why a separate op: the prefill GEMM ([`int8_w8a8_matmul`]) uses a 128x64 tile
/// that wastes 127/128 rows at `M=1`, so reusing it for sym8 DECODE is a
/// 1.4-1.8x regression vs affine qmv. This op runs a dedicated memory-BW-bound
/// matvec (one thread per output channel, streaming each int8 weight byte once)
/// that reaches parity with affine qmv. It reuses the SAME per-token activation
/// int8 quant (`na_int8_quant`) and applies `s_x[m]*s_w[n]` inline. The forward
/// path picks `qmv` for small `M`, `matmul` (gemm) for large `M`.
///
/// The result is **lazy** (composes into the surrounding forward graph) and
/// narrowed to bf16 inside C++ (so a downstream bf16 residual add is not promoted
/// to f32 by an f32 scale). Returns `Err` when unsupported (gen < 17 or
/// `K % 16 != 0`) or on a kernel/FFI failure — the caller falls back to bf16.
pub fn int8_w8a8_qmv(x: &MxArray, w_i8: &MxArray, s_w: &MxArray) -> Result<MxArray> {
    let mut out: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_int8_qmv(
            x.as_raw_ptr(),
            w_i8.as_raw_ptr(),
            s_w.as_raw_ptr(),
            &mut out,
        )
    };
    if !ok {
        return Err(Error::from_reason(
            "mlx_int8_qmv failed (unsupported gen/K or kernel error; see stderr)",
        ));
    }
    MxArray::from_handle(out, "int8_w8a8_qmv")
}

/// W8A16 sym8 DECODE matvec (QMV): the PRODUCTION decode op (`M <= 2`).
///
/// Unlike [`int8_w8a8_qmv`], the bf16 activations are read DIRECTLY by the
/// kernel — there is NO per-token activation quant (no absmax pass, no int8
/// staging). One pass: `y[m,n] = bf16(s_w[n] * sum_k x[m,k] * w_i8[k,n])`,
/// f32 accumulate. At decode `M` the act-quant passes of the W8A8 qmv are pure
/// overhead (the memory bottleneck is the identical 1-byte/weight stream), so
/// this removes the in-stream cost AND makes decode activation-EXACT — the
/// only remaining quantization error is the weight's.
///
/// Operands: `w_kn` is the opaque pre-transposed `[K,N]` int8 kernel layout
/// (EXACTLY what [`int8_w8a8_qmv`] / [`int8_w8a8_matmul`] consume — used by
/// the 2D-block fallback kernel under `INT8_QMV16_SG=0` and the W8A8
/// reroute); `w_nk` is the `[N,K]` int8 CHECKPOINT tensor (source
/// orientation — consumed by the DEFAULT simd_sum-style decode kernel, which
/// streams `[N,K]` row-major like MLX's affine qmv; the eager layer passes
/// its stored checkpoint weight, so this is buffer-shared, not a copy);
/// `s_w` f32 `[N]`. The result is **lazy** and narrowed to bf16 inside the
/// kernel. `INT8_QMV_W8A16=0` (read inside the shared C++ builder, so eager
/// and compiled stay byte-identical) reroutes back to the W8A8 qmv for
/// same-binary A/B. Returns `Err` when unsupported (gen < 17 or
/// `K % 16 != 0`) or on a kernel/FFI failure.
pub fn int8_w8a16_qmv(
    x: &MxArray,
    w_kn: &MxArray,
    w_nk: &MxArray,
    s_w: &MxArray,
) -> Result<MxArray> {
    let mut out: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_int8_qmv_w8a16(
            x.as_raw_ptr(),
            w_kn.as_raw_ptr(),
            w_nk.as_raw_ptr(),
            s_w.as_raw_ptr(),
            &mut out,
        )
    };
    if !ok {
        return Err(Error::from_reason(
            "mlx_int8_qmv_w8a16 failed (unsupported gen/K or kernel error; see stderr)",
        ));
    }
    MxArray::from_handle(out, "int8_w8a16_qmv")
}

/// MEASUREMENT ONLY (de-risk microbench — NOT a production path).
///
/// Affine-group W8A8 linear directly on the model's **EXACT** affine packed
/// weight (no re-quant). `x` is `[M,K]` bf16; `packed_w` is the MLX affine
/// packed uint32 weight `[N, K/4]` (4 uint8 per word, 8-bit); `scales`/`biases`
/// are f32 `[N, K/group_size]`. Returns bf16 `[M,N] = x @ dequant(packed_w)^T`
/// with per-token int8-quantized activation (identical activation quant to the
/// symmetric [`int8_w8a8_matmul`] path).
///
/// Math (`q` UNSIGNED in `[0,255]`, dequant `w[n,k]=scale[n,g]*q[n,k]+bias[n,g]`,
/// `g=k/group_size`; act per-token symmetric int8 `x_q[m,k]` in `[-127,127]`,
/// `s_x[m]=absmax_k|x[m,k]|/127`):
/// ```text
///   P[m,n,g] = sum_{k in g} x_q[m,k] * q[n,k]
///   S[m,g]   = sum_{k in g} x_q[m,k]
///   y[m,n]   = s_x[m] * sum_g ( scale[n,g]*P[m,n,g] + bias[n,g]*S[m,g] )  -> bf16
/// ```
///
/// Returns `Err` when unsupported (gen<17, `bits != 8`, or `K % group_size != 0`)
/// or on a kernel/FFI failure — the caller is expected to fall back.
pub fn affine_w8a8_matmul(
    x: &MxArray,
    packed_w: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    group_size: i32,
    bits: i32,
) -> Result<MxArray> {
    let mut out: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_affine_w8a8_linear(
            x.as_raw_ptr(),
            packed_w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
            group_size,
            bits,
            &mut out,
        )
    };
    if !ok {
        return Err(Error::from_reason(
            "mlx_affine_w8a8_linear failed (unsupported gen/bits/group_size or kernel error; see stderr)",
        ));
    }
    MxArray::from_handle(out, "affine_w8a8_matmul")
}

/// MEASUREMENT ONLY (de-risk microbench — NOT a production path).
///
/// LOAD-TIME prepare for the TILED affine-group W8A8 GEMM (runs once per
/// quantized linear). Unpacks the affine packed `uint32` weight `[N, K/4]` into
/// the SIGNED int8 `[K,N]` kernel operand (`q - 128`) the tiled `matmul2d`
/// wants, keeps the f32 `scales` `[N, K/group_size]`, and precomputes
/// `bias_adj = 128*scale + bias` `[N, K/group_size]`.
///
/// Returns `(q_s, scale_kept, bias_adj)` opaque handles for
/// [`affine_w8a8_linear_prepared`]. Returns `Err` when unsupported (gen<17,
/// `bits != 8`, or `K % group_size != 0`) or on a kernel/FFI failure.
#[cfg(test)]
pub fn affine_w8a8_prepare(
    packed_w: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    group_size: i32,
    bits: i32,
) -> Result<(MxArray, MxArray, MxArray)> {
    let mut out_q_s: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_scale: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_badj: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_affine_w8a8_prepare(
            packed_w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
            group_size,
            bits,
            &mut out_q_s,
            &mut out_scale,
            &mut out_badj,
        )
    };
    if !ok {
        return Err(Error::from_reason(
            "mlx_affine_w8a8_prepare failed (unsupported gen/bits/group_size or kernel error; see stderr)",
        ));
    }
    Ok((
        MxArray::from_handle(out_q_s, "affine_w8a8_prepare:q_s")?,
        MxArray::from_handle(out_scale, "affine_w8a8_prepare:scale")?,
        MxArray::from_handle(out_badj, "affine_w8a8_prepare:badj")?,
    ))
}

/// MEASUREMENT ONLY (de-risk microbench — NOT a production path).
///
/// Per-FORWARD prepared affine-group W8A8 linear (the TIMED hot path). Per-token
/// int8 activation quant + per-group act-sum `S`, then the TILED grouped
/// `matmul2d` GEMM. `x` is `[M,K]` bf16; `q_s` / `scale` / `bias_adj` come from
/// [`affine_w8a8_prepare`]. Returns bf16 `[M,N]` (lazy).
///
/// Returns `Err` when unsupported (gen<17 or `K % group_size != 0`) or on a
/// kernel/FFI failure.
#[cfg(test)]
pub fn affine_w8a8_linear_prepared(
    x: &MxArray,
    q_s: &MxArray,
    scale: &MxArray,
    bias_adj: &MxArray,
    group_size: i32,
) -> Result<MxArray> {
    let mut out: *mut sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_affine_w8a8_linear_prepared(
            x.as_raw_ptr(),
            q_s.as_raw_ptr(),
            scale.as_raw_ptr(),
            bias_adj.as_raw_ptr(),
            group_size,
            &mut out,
        )
    };
    if !ok {
        return Err(Error::from_reason(
            "mlx_affine_w8a8_linear_prepared failed (unsupported gen/group_size or kernel error; see stderr)",
        ));
    }
    MxArray::from_handle(out, "affine_w8a8_linear_prepared")
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

    // ================= AFFINE-GROUP W8A8 GROUPING OVERHEAD =================
    // DE-RISK MICROBENCH. Isolates the per-group flush + affine bias OVERHEAD of
    // the affine-group W8A8 kernel (`affine_w8a8_matmul`) OVER the proven
    // symmetric W8A8 kernel (`int8_w8a8_matmul`, the realized +39-83% prefill
    // path), at identical dense-MLP shapes. Also reports the actual WIN vs the
    // affine `quantized_matmul` (qmm) baseline on the SAME packed affine weight.
    //
    // For each (M, N, K) it times three ops (warm, many iters, eval+sync between):
    //   (1) AFFINE-GROUP W8A8 : affine_w8a8_matmul on the model's EXACT affine
    //       packed uint32 weight (no re-quant). The kernel under de-risk.
    //   (2) SYMMETRIC  W8A8   : int8_w8a8_matmul. Needs a symmetric int8 weight of
    //       the SAME [N,K] shape — built by DEQUANTIZing the affine packed weight
    //       to bf16 and running quantize_weight_int8 on it.
    //   (3) AFFINE qmm        : mlx_quantized_matmul(transpose=true) on the SAME
    //       packed affine weight/scales/biases — the production quant baseline.
    //
    // Reports median GPU ms/op and the two ratios:
    //   grouped/symmetric  = the OVERHEAD (the de-risk number; >1 = grouping costs)
    //   grouped/qmm        = the actual WIN (<1 = grouped beats qmm).
    //
    // The de-risk question is NOT "is grouped faster than qmm in absolute" (this
    // grouped kernel is a plain int32-accumulator reference, not a tuned matmul2d)
    // — it is how much the per-group flush + bias add OVER the symmetric kernel.
    //
    // Run:
    //   cargo test -p mlx-core --lib int8_gemm::tests::profile_grouping_overhead \
    //     -- --ignored --nocapture
    #[test]
    #[ignore = "manual affine-group W8A8 grouping-overhead microbench; run with --ignored"]
    fn profile_grouping_overhead() {
        use crate::array::memory::synchronize;
        use std::time::Instant;
        if gpu_gen() < 17 {
            eprintln!("[grp] SKIP gpu gen {} < 17 (NA needs M5+)", gpu_gen());
            return;
        }

        let group_size: i32 = 64;
        let bits: i32 = 8;
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

        // Local affine quantize / dequantize / qmm wrappers (test scope).
        let affine_quantize = |w: &MxArray| -> (MxArray, MxArray, MxArray) {
            let mut q: *mut sys::mlx_array = std::ptr::null_mut();
            let mut s: *mut sys::mlx_array = std::ptr::null_mut();
            let mut b: *mut sys::mlx_array = std::ptr::null_mut();
            let ok = unsafe {
                sys::mlx_quantize(
                    w.as_raw_ptr(),
                    group_size,
                    bits,
                    c"affine".as_ptr(),
                    &mut q,
                    &mut s,
                    &mut b,
                )
            };
            assert!(ok, "mlx_quantize(affine) failed");
            (
                MxArray::from_handle(q, "grp:packed_w").unwrap(),
                MxArray::from_handle(s, "grp:scales").unwrap(),
                MxArray::from_handle(b, "grp:biases").unwrap(),
            )
        };
        let affine_dequantize = |q: &MxArray, s: &MxArray, b: &MxArray| -> MxArray {
            let handle = unsafe {
                sys::mlx_dequantize(
                    q.as_raw_ptr(),
                    s.as_raw_ptr(),
                    b.as_raw_ptr(),
                    group_size,
                    bits,
                    3, // bfloat16 (BridgeDType: FLOAT32=0,INT32=1,FLOAT16=2,BFLOAT16=3,UINT32=4,UINT8=5)
                    c"affine".as_ptr(),
                )
            };
            MxArray::from_handle(handle, "grp:dequant").unwrap()
        };
        let affine_qmm = |x: &MxArray, q: &MxArray, s: &MxArray, b: &MxArray| -> MxArray {
            let handle = unsafe {
                sys::mlx_quantized_matmul(
                    x.as_raw_ptr(),
                    q.as_raw_ptr(),
                    s.as_raw_ptr(),
                    b.as_raw_ptr(),
                    true, // transpose: [N,K] weight -> x @ w^T = [M,N]
                    group_size,
                    bits,
                    c"affine".as_ptr(),
                )
            };
            MxArray::from_handle(handle, "grp:qmm").unwrap()
        };

        // One (M,N,K) shape: build the affine packed weight once, the symmetric
        // int8 weight from its dequant, then time the three ops.
        let run = |label: &str, m: i64, n: i64, k: i64| {
            // bf16 activations + weight [N,K] (rows = output channels).
            let x = MxArray::random_normal(&[m, k], 0.0, 0.05, Some(DType::BFloat16)).unwrap();
            let w = MxArray::random_normal(&[n, k], 0.0, 0.02, Some(DType::BFloat16)).unwrap();
            x.eval();
            w.eval();

            // (1) operands: the model's EXACT affine packed weight (no re-quant).
            let (packed_w, scales, biases) = affine_quantize(&w);
            packed_w.eval();
            scales.eval();
            biases.eval();

            // (2) operands: symmetric int8 weight of the SAME [N,K] shape, built
            // from the DEQUANTIZED affine weight (so both paths quantize the SAME
            // numeric weight — apples-to-apples).
            let w_deq = affine_dequantize(&packed_w, &scales, &biases);
            w_deq.eval();
            let (sym_i8, sym_s) = quantize_weight_int8(&w_deq).unwrap();
            sym_i8.eval();
            sym_s.eval();

            // (1') LOAD-TIME prepare for the TILED grouped GEMM — runs ONCE,
            // OUTSIDE the timed loop (mirrors quantize_weight_int8 for the
            // symmetric path). Unpacks the affine weight to the signed int8 [K,N]
            // operand + bias_adj, so the timed closure measures only the GEMM.
            let (grp_q_s, grp_scale, grp_badj) =
                affine_w8a8_prepare(&packed_w, &scales, &biases, group_size, bits).unwrap();
            grp_q_s.eval();
            grp_scale.eval();
            grp_badj.eval();
            synchronize();

            // The three timed closures. `grouped` now times ONLY the prepared
            // per-forward linear (act-quant + S + tiled grouped matmul2d), NOT
            // the one-time weight unpack.
            let grouped = || {
                affine_w8a8_linear_prepared(&x, &grp_q_s, &grp_scale, &grp_badj, group_size)
                    .unwrap()
            };
            let symmetric = || int8_w8a8_matmul(&x, &sym_i8, &sym_s).unwrap();
            let qmm = || affine_qmm(&x, &packed_w, &scales, &biases);

            // 3 runs each, interleaved; report medians (host bench drift ~10-15%).
            let mut g = [0.0f64; 3];
            let mut s = [0.0f64; 3];
            let mut q = [0.0f64; 3];
            for r in 0..3 {
                g[r] = bench(&grouped);
                s[r] = bench(&symmetric);
                q[r] = bench(&qmm);
            }
            let median = |mut v: [f64; 3]| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[1]
            };
            let gm = median(g);
            let sm = median(s);
            let qm = median(q);

            eprintln!(
                "[grp] === {label}: M={m} N={n} K={k} (group_size={group_size}, bits={bits}) ==="
            );
            eprintln!(
                "[grp] grouped (ms): {:.3} {:.3} {:.3} -> median {gm:.3}",
                g[0], g[1], g[2]
            );
            eprintln!(
                "[grp] symmetric(ms): {:.3} {:.3} {:.3} -> median {sm:.3}",
                s[0], s[1], s[2]
            );
            eprintln!(
                "[grp] qmm     (ms): {:.3} {:.3} {:.3} -> median {qm:.3}",
                q[0], q[1], q[2]
            );
            eprintln!(
                "[grp] OVERHEAD grouped/symmetric = {:.3}  (>1 = per-group flush+bias costs)",
                gm / sm
            );
            eprintln!(
                "[grp] WIN      grouped/qmm       = {:.3}  (<1 = grouped beats qmm)",
                gm / qm
            );
        };

        eprintln!("[grp] === affine-group W8A8 grouping overhead, real Qwen3.5-4B MLP shapes ===");
        // Real dense-MLP shapes: K in {2560, 9216}, N in {9216, 2560}, M in
        // {1024, 4096}. 4B: gate_up is K=hidden=2560 -> N=intermediate=9216
        // (intermediate per gate/up branch); down is K=9216 -> N=2560.
        for &m in &[1024i64, 4096] {
            run("gate_up", m, 9216, 2560); // x[M,2560] @ deq(w[9216,2560])^T
            run("down", m, 2560, 9216); // x[M,9216] @ deq(w[2560,9216])^T
        }
    }

    // ===================== PHASE 0: sym8 DECODE de-risk =====================
    // A sym8 (per-channel symmetric int8) checkpoint routes BOTH prefill AND
    // DECODE through int8_w8a8_matmul — a sym8 weight has NO affine packed form to
    // fall back to at M=1. This gate answers the two open DECODE questions BEFORE
    // any convert/loader plumbing (kills the project cheaply if decode is bad):
    //   (1) CORRECTNESS at M=1 / tiny M: does int8_w8a8_matmul produce a faithful
    //       result on a 1-row (partial-tile) activation? (s1b proves bit-exact
    //       partial tiles; this confirms cosine vs an f32 reference at M=1.)
    //   (2) DECODE PERF: is sym8-GEMM-at-M=1 within parity of affine qmv (today's
    //       decode path)? Both stream ~1 byte/weight (BW-bound). A dedicated sym8
    //       qmv (Phase 6) is justified ONLY if this REGRESSES.
    // The weight is quantized from the bf16 SOURCE (quantize_weight_int8 = the true
    // Option-B single-quant path), and the affine baseline affine-quantizes the
    // SAME w (apples-to-apples sym8-decode vs affine-Q8-decode).
    // Run:
    //   cargo test -p mlx-core --lib int8_gemm::tests::profile_sym8_decode \
    //     -- --ignored --nocapture
    #[test]
    #[ignore = "manual sym8 decode de-risk (correctness@M=1 + decode perf vs affine qmv); run with --ignored"]
    fn profile_sym8_decode() {
        use crate::array::memory::synchronize;
        use std::time::Instant;
        if gpu_gen() < 17 {
            eprintln!("[sym8dec] SKIP gpu gen {} < 17 (NA needs M5+)", gpu_gen());
            return;
        }
        let group_size: i32 = 64;
        let bits: i32 = 8;
        let iters = 100;
        let warm = 20;

        let bench = |f: &dyn Fn() -> MxArray| -> f64 {
            for _ in 0..warm {
                f().eval();
            }
            synchronize();
            let t = Instant::now();
            for _ in 0..iters {
                f().eval();
            }
            synchronize();
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        };
        let affine_quantize = |w: &MxArray| -> (MxArray, MxArray, MxArray) {
            let mut q: *mut sys::mlx_array = std::ptr::null_mut();
            let mut s: *mut sys::mlx_array = std::ptr::null_mut();
            let mut b: *mut sys::mlx_array = std::ptr::null_mut();
            let ok = unsafe {
                sys::mlx_quantize(
                    w.as_raw_ptr(),
                    group_size,
                    bits,
                    c"affine".as_ptr(),
                    &mut q,
                    &mut s,
                    &mut b,
                )
            };
            assert!(ok, "mlx_quantize(affine) failed");
            (
                MxArray::from_handle(q, "sym8dec:packed").unwrap(),
                MxArray::from_handle(s, "sym8dec:scales").unwrap(),
                MxArray::from_handle(b, "sym8dec:biases").unwrap(),
            )
        };
        let affine_qmm = |x: &MxArray, q: &MxArray, s: &MxArray, b: &MxArray| -> MxArray {
            let handle = unsafe {
                sys::mlx_quantized_matmul(
                    x.as_raw_ptr(),
                    q.as_raw_ptr(),
                    s.as_raw_ptr(),
                    b.as_raw_ptr(),
                    true,
                    group_size,
                    bits,
                    c"affine".as_ptr(),
                )
            };
            MxArray::from_handle(handle, "sym8dec:qmm").unwrap()
        };

        let run = |label: &str, m: i64, n: i64, k: i64| {
            let x = MxArray::random_normal(&[m, k], 0.0, 0.05, Some(DType::BFloat16)).unwrap();
            let w = MxArray::random_normal(&[n, k], 0.0, 0.02, Some(DType::BFloat16)).unwrap();
            x.eval();
            w.eval();
            // sym8 (Option B) — quantize from bf16 SOURCE (single quant).
            let (w_i8, s_w) = quantize_weight_int8(&w).unwrap();
            w_i8.eval();
            s_w.eval();
            // affine-Q8 baseline of the SAME weight.
            let (packed_w, scales, biases) = affine_quantize(&w);
            packed_w.eval();
            scales.eval();
            biases.eval();

            // (1) CORRECTNESS — per-row cosine vs f32 reference (x_f32 @ w_f32^T).
            let y = int8_w8a8_matmul(&x, &w_i8, &s_w).unwrap();
            y.eval();
            let wt = w
                .astype(DType::Float32)
                .unwrap()
                .transpose(Some(&[1, 0]))
                .unwrap();
            let y_ref = x.astype(DType::Float32).unwrap().matmul(&wt).unwrap();
            y_ref.eval();
            let got = y.astype(DType::Float32).unwrap().to_float32().unwrap();
            let got: &[f32] = &got;
            let refv = y_ref.to_float32().unwrap();
            let refv: &[f32] = &refv;
            let mut min_cos = f64::INFINITY;
            let mut sum_cos = 0.0f64;
            for mi in 0..m as usize {
                let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
                for ni in 0..n as usize {
                    let a = got[mi * n as usize + ni] as f64;
                    let b = refv[mi * n as usize + ni] as f64;
                    dot += a * b;
                    na += a * a;
                    nb += b * b;
                }
                let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
                min_cos = min_cos.min(cos);
                sum_cos += cos;
            }
            let mean_cos = sum_cos / m as f64;

            // (2) DECODE PERF — sym8 GEMM vs affine qmv, same weight.
            let sym = || int8_w8a8_matmul(&x, &w_i8, &s_w).unwrap();
            let qmm = || affine_qmm(&x, &packed_w, &scales, &biases);
            let median = |mut v: [f64; 3]| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[1]
            };
            let mut sv = [0.0f64; 3];
            let mut qv = [0.0f64; 3];
            for r in 0..3 {
                sv[r] = bench(&sym);
                qv[r] = bench(&qmm);
            }
            let sm = median(sv);
            let qm = median(qv);
            eprintln!(
                "[sym8dec] {label} M={m} N={n} K={k}: cos min={min_cos:.6} mean={mean_cos:.6} | sym8={sm:.4}ms qmv={qm:.4}ms  sym8/qmv={:.3} ({})",
                sm / qm,
                if sm / qm <= 1.05 {
                    "PARITY-OK"
                } else {
                    "REGRESSION?"
                }
            );
            assert!(
                min_cos >= 0.999,
                "sym8 decode cosine below gate at {label} M={m}: min={min_cos:.6}"
            );
        };

        eprintln!(
            "[sym8dec] === sym8 decode de-risk: correctness@M=1 + perf vs affine qmv (4B shapes) ==="
        );
        for &m in &[1i64, 4, 8] {
            run("gate_up", m, 9216, 2560);
            run("down", m, 2560, 9216);
            run("o_proj", m, 2560, 2560);
        }
    }

    // ===================== PHASE 6: sym8 DECODE QMV =====================
    // GATE: the DEDICATED sym8 matvec (int8_w8a8_qmv) must be a FAITHFUL decode
    // matvec — per-row cosine vs an f32 reference (x_f32 @ w_f32^T) >= 0.999 at
    // M in {1,4,8} on the three 4B projection shapes {gate_up, down, o_proj}.
    // Mirrors profile_sym8_decode but exercises the qmv path (which Phase-0
    // proved is needed because reusing the 128x64 prefill GEMM at M=1 wastes
    // 127/128 rows -> 1.4-1.8x regression vs affine qmv). The weight is quantized
    // from the bf16 SOURCE (quantize_weight_int8 = the true Option-B single-quant
    // path). This phase's bar is correctness + that it RUNS; the perf verdict (qmv
    // vs affine qmv) is the NEXT phase — we time it here only as a smoke check.
    // Run:
    //   cargo test -p mlx-core --lib int8_gemm::tests::profile_sym8_qmv \
    //     -- --ignored --nocapture
    #[test]
    #[ignore = "manual sym8 decode QMV correctness (cosine@M=1/4/8) + smoke timing; run with --ignored"]
    fn profile_sym8_qmv() {
        use crate::array::memory::synchronize;
        use std::time::Instant;
        if gpu_gen() < 17 {
            eprintln!("[sym8qmv] SKIP gpu gen {} < 17 (NA needs M5+)", gpu_gen());
            return;
        }
        let group_size: i32 = 64;
        let bits: i32 = 8;
        let iters = 100;
        let warm = 20;
        let bench = |f: &dyn Fn() -> MxArray| -> f64 {
            for _ in 0..warm {
                f().eval();
            }
            synchronize();
            let t = Instant::now();
            for _ in 0..iters {
                f().eval();
            }
            synchronize();
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        };
        // affine-Q8 baseline of the SAME weight (the production decode path:
        // mlx_quantized_matmul transpose=true). This is the apples-to-apples
        // affine qmv the dedicated sym8 qmv must reach parity with at M=1.
        let affine_quantize = |w: &MxArray| -> (MxArray, MxArray, MxArray) {
            let mut q: *mut sys::mlx_array = std::ptr::null_mut();
            let mut s: *mut sys::mlx_array = std::ptr::null_mut();
            let mut b: *mut sys::mlx_array = std::ptr::null_mut();
            let ok = unsafe {
                sys::mlx_quantize(
                    w.as_raw_ptr(),
                    group_size,
                    bits,
                    c"affine".as_ptr(),
                    &mut q,
                    &mut s,
                    &mut b,
                )
            };
            assert!(ok, "mlx_quantize(affine) failed");
            (
                MxArray::from_handle(q, "sym8qmv:packed").unwrap(),
                MxArray::from_handle(s, "sym8qmv:scales").unwrap(),
                MxArray::from_handle(b, "sym8qmv:biases").unwrap(),
            )
        };
        let affine_qmm = |x: &MxArray, q: &MxArray, s: &MxArray, b: &MxArray| -> MxArray {
            let handle = unsafe {
                sys::mlx_quantized_matmul(
                    x.as_raw_ptr(),
                    q.as_raw_ptr(),
                    s.as_raw_ptr(),
                    b.as_raw_ptr(),
                    true,
                    group_size,
                    bits,
                    c"affine".as_ptr(),
                )
            };
            MxArray::from_handle(handle, "sym8qmv:qmm").unwrap()
        };

        let run = |label: &str, m: i64, n: i64, k: i64| {
            let x = MxArray::random_normal(&[m, k], 0.0, 0.05, Some(DType::BFloat16)).unwrap();
            let w = MxArray::random_normal(&[n, k], 0.0, 0.02, Some(DType::BFloat16)).unwrap();
            x.eval();
            w.eval();
            // sym8 (Option B) — quantize from bf16 SOURCE (single quant). Same
            // (w_i8 [K,N], s_w [N]) operands int8_w8a8_matmul consumes.
            let (w_i8, s_w) = quantize_weight_int8(&w).unwrap();
            w_i8.eval();
            s_w.eval();
            // affine-Q8 baseline of the SAME weight.
            let (packed_w, scales, biases) = affine_quantize(&w);
            packed_w.eval();
            scales.eval();
            biases.eval();

            // CORRECTNESS — per-row cosine vs f32 reference (x_f32 @ w_f32^T).
            let y = int8_w8a8_qmv(&x, &w_i8, &s_w).unwrap();
            y.eval();
            assert_eq!(
                y.dtype().unwrap(),
                DType::BFloat16,
                "qmv output must be bf16"
            );
            assert_eq!(y.shape_at(0).unwrap(), m, "qmv rows");
            assert_eq!(y.shape_at(1).unwrap(), n, "qmv cols");
            let wt = w
                .astype(DType::Float32)
                .unwrap()
                .transpose(Some(&[1, 0]))
                .unwrap();
            let y_ref = x.astype(DType::Float32).unwrap().matmul(&wt).unwrap();
            y_ref.eval();
            let got = y.astype(DType::Float32).unwrap().to_float32().unwrap();
            let got: &[f32] = &got;
            let refv = y_ref.to_float32().unwrap();
            let refv: &[f32] = &refv;
            let mut min_cos = f64::INFINITY;
            let mut sum_cos = 0.0f64;
            for mi in 0..m as usize {
                let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
                for ni in 0..n as usize {
                    let a = got[mi * n as usize + ni] as f64;
                    let b = refv[mi * n as usize + ni] as f64;
                    dot += a * b;
                    na += a * a;
                    nb += b * b;
                }
                let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
                min_cos = min_cos.min(cos);
                sum_cos += cos;
            }
            let mean_cos = sum_cos / m as f64;

            // DECODE PARITY — dedicated sym8 qmv vs affine qmv (production decode
            // path), same weight. Mirrors profile_sym8_decode's bench structure
            // exactly: warm 20, iters 100, 3 interleaved runs, median, sync
            // between. The gemm column stays as a smoke reference.
            let qmv = || int8_w8a8_qmv(&x, &w_i8, &s_w).unwrap();
            let gemm = || int8_w8a8_matmul(&x, &w_i8, &s_w).unwrap();
            let affqmv = || affine_qmm(&x, &packed_w, &scales, &biases);
            let median = |mut v: [f64; 3]| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[1]
            };
            let mut qv = [0.0f64; 3];
            let mut gv = [0.0f64; 3];
            let mut av = [0.0f64; 3];
            for r in 0..3 {
                qv[r] = bench(&qmv);
                av[r] = bench(&affqmv);
                gv[r] = bench(&gemm);
            }
            let qm = median(qv);
            let gm = median(gv);
            let am = median(av);
            eprintln!(
                "[sym8qmv] {label} M={m} N={n} K={k}: cos min={min_cos:.6} mean={mean_cos:.6} | int8_qmv={qm:.4}ms affine_qmv={am:.4}ms gemm={gm:.4}ms  int8_qmv/affine_qmv={:.3} ({})  qmv/gemm={:.3}",
                qm / am,
                if qm / am <= 1.05 {
                    "PARITY-OK"
                } else {
                    "REGRESSION?"
                },
                qm / gm
            );
            assert!(
                min_cos >= 0.999,
                "sym8 qmv cosine below gate at {label} M={m}: min={min_cos:.6}"
            );
        };

        eprintln!(
            "[sym8qmv] === sym8 decode QMV: correctness@M=1/4/8 + parity vs affine qmv (4B shapes) ==="
        );
        for &m in &[1i64, 4, 8] {
            run("gate_up", m, 9216, 2560);
            run("down", m, 2560, 9216);
            run("o_proj", m, 2560, 2560);
        }
    }

    // ===================== W8A16 sym8 DECODE QMV =====================
    // GATE: the W8A16 decode matvec (int8_w8a16_qmv — bf16 activations read
    // directly, NO act quant, f32 accumulate) must be a FAITHFUL decode matvec:
    // per-row cosine vs an f32 reference (x_f32 @ w_f32^T) >= 0.999 at M in
    // {1,2} (the decode dispatch range) on the three 4B projection shapes.
    // Because the activation is EXACT (only weight-quant error remains), the
    // cosine must also be >= the W8A8 qmv's on the same inputs — asserted
    // directly. PERF (informational — the e2e paired A/B is the real gate):
    // W8A16 qmv vs affine qmv (production affine decode path) vs the old W8A8
    // qmv at M=1, plus an INT8_QMV16_BN/BK geometry sweep.
    // Run:
    //   cargo test -p mlx-core --release int8_gemm::tests::profile_sym8_qmv_w8a16 \
    //     -- --ignored --nocapture --test-threads=1
    #[test]
    #[ignore = "manual W8A16 sym8 decode QMV correctness (cosine@M=1/2) + microbench; run with --ignored"]
    fn profile_sym8_qmv_w8a16() {
        use crate::array::memory::synchronize;
        use std::time::Instant;
        if gpu_gen() < 17 {
            eprintln!("[w8a16qmv] SKIP gpu gen {} < 17 (NA needs M5+)", gpu_gen());
            return;
        }
        let group_size: i32 = 64;
        let bits: i32 = 8;
        let iters = 100;
        let warm = 20;
        let bench = |f: &dyn Fn() -> MxArray| -> f64 {
            for _ in 0..warm {
                f().eval();
            }
            synchronize();
            let t = Instant::now();
            for _ in 0..iters {
                f().eval();
            }
            synchronize();
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        };
        let affine_quantize = |w: &MxArray| -> (MxArray, MxArray, MxArray) {
            let mut q: *mut sys::mlx_array = std::ptr::null_mut();
            let mut s: *mut sys::mlx_array = std::ptr::null_mut();
            let mut b: *mut sys::mlx_array = std::ptr::null_mut();
            let ok = unsafe {
                sys::mlx_quantize(
                    w.as_raw_ptr(),
                    group_size,
                    bits,
                    c"affine".as_ptr(),
                    &mut q,
                    &mut s,
                    &mut b,
                )
            };
            assert!(ok, "mlx_quantize(affine) failed");
            (
                MxArray::from_handle(q, "w8a16qmv:packed").unwrap(),
                MxArray::from_handle(s, "w8a16qmv:scales").unwrap(),
                MxArray::from_handle(b, "w8a16qmv:biases").unwrap(),
            )
        };
        let affine_qmm = |x: &MxArray, q: &MxArray, s: &MxArray, b: &MxArray| -> MxArray {
            let handle = unsafe {
                sys::mlx_quantized_matmul(
                    x.as_raw_ptr(),
                    q.as_raw_ptr(),
                    s.as_raw_ptr(),
                    b.as_raw_ptr(),
                    true,
                    group_size,
                    bits,
                    c"affine".as_ptr(),
                )
            };
            MxArray::from_handle(handle, "w8a16qmv:qmm").unwrap()
        };
        // Per-row cosine of `got` vs `refv`, both [M,N] row-major f32.
        let row_cosines = |got: &[f32], refv: &[f32], m: usize, n: usize| -> (f64, f64) {
            let mut min_cos = f64::INFINITY;
            let mut sum_cos = 0.0f64;
            for mi in 0..m {
                let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
                for ni in 0..n {
                    let a = got[mi * n + ni] as f64;
                    let b = refv[mi * n + ni] as f64;
                    dot += a * b;
                    na += a * a;
                    nb += b * b;
                }
                let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
                min_cos = min_cos.min(cos);
                sum_cos += cos;
            }
            (min_cos, sum_cos / m as f64)
        };

        let run = |label: &str, m: i64, n: i64, k: i64| {
            let x = MxArray::random_normal(&[m, k], 0.0, 0.05, Some(DType::BFloat16)).unwrap();
            let w = MxArray::random_normal(&[n, k], 0.0, 0.02, Some(DType::BFloat16)).unwrap();
            x.eval();
            w.eval();
            // sym8 weight quant from the bf16 SOURCE — the same (w_i8 [K,N],
            // s_w [N]) operands the production forward consumes. The decode
            // matvec ALSO takes the [N,K] checkpoint orientation; rebuild it
            // from the [K,N] operand (sym8_kernel_operand is just a
            // transpose+contiguous — bit-exact with the stored checkpoint).
            let (w_i8, s_w) = quantize_weight_int8(&w).unwrap();
            w_i8.eval();
            s_w.eval();
            let w_nk = sym8_kernel_operand(&w_i8).unwrap();
            w_nk.eval();
            // affine-Q8 baseline of the SAME weight (production affine decode).
            let (packed_w, scales, biases) = affine_quantize(&w);
            packed_w.eval();
            scales.eval();
            biases.eval();

            // CORRECTNESS — per-row cosine vs f32 reference (x_f32 @ w_f32^T).
            let y = int8_w8a16_qmv(&x, &w_i8, &w_nk, &s_w).unwrap();
            y.eval();
            assert_eq!(
                y.dtype().unwrap(),
                DType::BFloat16,
                "w8a16 qmv output must be bf16"
            );
            assert_eq!(y.shape_at(0).unwrap(), m, "w8a16 qmv rows");
            assert_eq!(y.shape_at(1).unwrap(), n, "w8a16 qmv cols");
            let wt = w
                .astype(DType::Float32)
                .unwrap()
                .transpose(Some(&[1, 0]))
                .unwrap();
            let y_ref = x.astype(DType::Float32).unwrap().matmul(&wt).unwrap();
            y_ref.eval();
            let got = y.astype(DType::Float32).unwrap().to_float32().unwrap();
            let refv = y_ref.to_float32().unwrap();
            let (min_cos, mean_cos) = row_cosines(&got, &refv, m as usize, n as usize);
            // W8A8 qmv on the SAME inputs — the W8A16 cosine must be >= it
            // (activation quant error removed; weight error identical).
            let y8 = int8_w8a8_qmv(&x, &w_i8, &s_w).unwrap();
            y8.eval();
            let got8 = y8.astype(DType::Float32).unwrap().to_float32().unwrap();
            let (min_cos8, mean_cos8) = row_cosines(&got8, &refv, m as usize, n as usize);
            eprintln!(
                "[w8a16qmv] {label} M={m} N={n} K={k}: cos min={min_cos:.6} mean={mean_cos:.6} | w8a8 cos min={min_cos8:.6} mean={mean_cos8:.6} ({})",
                if min_cos >= min_cos8 {
                    "W8A16>=W8A8 OK"
                } else {
                    "W8A16 BELOW W8A8?"
                }
            );
            assert!(
                min_cos >= 0.999,
                "w8a16 qmv cosine below gate at {label} M={m}: min={min_cos:.6}"
            );
            assert!(
                mean_cos >= mean_cos8 - 1e-9,
                "w8a16 qmv mean cosine ({mean_cos:.7}) below the W8A8 qmv's \
                 ({mean_cos8:.7}) at {label} M={m} — activation-exact path must \
                 not be LESS accurate"
            );

            // PERF (informational) at this M — W8A16 vs affine qmv vs old W8A8.
            let w8a16 = || int8_w8a16_qmv(&x, &w_i8, &w_nk, &s_w).unwrap();
            let w8a8 = || int8_w8a8_qmv(&x, &w_i8, &s_w).unwrap();
            let affqmv = || affine_qmm(&x, &packed_w, &scales, &biases);
            let median = |mut v: [f64; 3]| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[1]
            };
            let mut nv = [0.0f64; 3];
            let mut ov = [0.0f64; 3];
            let mut av = [0.0f64; 3];
            for r in 0..3 {
                nv[r] = bench(&w8a16);
                av[r] = bench(&affqmv);
                ov[r] = bench(&w8a8);
            }
            let nm = median(nv);
            let am = median(av);
            let om = median(ov);
            eprintln!(
                "[w8a16qmv] {label} M={m} N={n} K={k}: w8a16={nm:.4}ms affine_qmv={am:.4}ms w8a8_qmv={om:.4}ms  w8a16/affine={:.3} ({})  w8a16/w8a8={:.3}",
                nm / am,
                if nm / am <= 1.05 {
                    "PARITY-OK"
                } else {
                    "REGRESSION?"
                },
                nm / om
            );
        };

        eprintln!(
            "[w8a16qmv] === W8A16 sym8 decode QMV: correctness@M=1/2 + perf vs affine/W8A8 qmv (4B shapes) ==="
        );
        for &m in &[1i64, 2] {
            run("gate_up", m, 9216, 2560);
            run("down", m, 2560, 9216);
            run("o_proj", m, 2560, 2560);
            // N=64 GDN gate projection: N % 128 != 0 -> exercises the host's
            // scalar fallback under the shape-aware default geometry.
            run("gdn_gate", m, 64, 2560);
            // Largest real decode shape (qkvz fused): VEC4/BN=128 default path.
            run("qkvz", m, 12288, 2560);
        }

        // GEOMETRY SWEEP (M=1, informational): default BN=32/BK=16 vs
        // alternatives. The env vars are read PER CALL inside
        // int8_qmv_w8a16_core, so set_var between runs re-dispatches. SAFETY:
        // tests in this profile scope run with --test-threads=1 (GPU strictly
        // serial), so the process-global env mutation cannot race.
        let sweep = |bn: &str, bk: &str, stagex: &str| {
            unsafe {
                std::env::set_var("INT8_QMV16_BN", bn);
                std::env::set_var("INT8_QMV16_BK", bk);
                std::env::set_var("INT8_QMV16_STAGEX", stagex);
            }
            for (label, n, k) in [
                ("gate_up", 9216i64, 2560i64),
                ("down", 2560, 9216),
                ("o_proj", 2560, 2560),
            ] {
                let x = MxArray::random_normal(&[1, k], 0.0, 0.05, Some(DType::BFloat16)).unwrap();
                let w = MxArray::random_normal(&[n, k], 0.0, 0.02, Some(DType::BFloat16)).unwrap();
                x.eval();
                w.eval();
                let (w_i8, s_w) = quantize_weight_int8(&w).unwrap();
                w_i8.eval();
                s_w.eval();
                let w_nk = sym8_kernel_operand(&w_i8).unwrap();
                w_nk.eval();
                let f = || int8_w8a16_qmv(&x, &w_i8, &w_nk, &s_w).unwrap();
                let t = bench(&f);
                eprintln!(
                    "[w8a16qmv][sweep] BN={bn} BK={bk} STAGEX={stagex} {label} M=1 N={n} K={k}: {t:.4}ms"
                );
            }
        };
        sweep("32", "16", "0"); // default
        sweep("64", "16", "0");
        sweep("32", "32", "0");
        sweep("32", "16", "1"); // tg-staged x (host gates off where it can't fit)
        unsafe {
            std::env::remove_var("INT8_QMV16_BN");
            std::env::remove_var("INT8_QMV16_BK");
            std::env::remove_var("INT8_QMV16_STAGEX");
        }
    }

    // =============== IN-STREAM decode-shaped attribution bench ===============
    // The e2e paired A/B proved isolated per-call microbenches UNDER-represent
    // the in-stream cost of the sym8 qmv at decode M=1 (W8A8 was
    // isolated-parity yet ~2x slower e2e). This bench reproduces the decode
    // call pattern: a DEPENDENT chain of 150 M=1 projections (50 rounds of
    // gate_up -> down -> o_proj, each consuming the previous output) built
    // LAZILY and eval'd ONCE — exactly how the forward graph streams qmv
    // calls — for three arms: W8A16 qmv, affine quantized_matmul (production
    // baseline), old W8A8 qmv. A second TINY chain (64x64: GPU work ~0)
    // prices the PER-CALL NODE OVERHEAD (graph node + encode + dispatch) of
    // the fast::metal_kernel path vs the native affine primitive — splitting
    // the in-stream delta into "kernel GPU time" vs "custom-kernel node cost".
    // Run:
    //   cargo test -p mlx-core --release int8_gemm::tests::profile_sym8_qmv_instream \
    //     -- --ignored --nocapture --test-threads=1
    #[test]
    #[ignore = "manual in-stream decode-chain attribution bench; run with --ignored"]
    fn profile_sym8_qmv_instream() {
        use crate::array::memory::synchronize;
        use std::time::Instant;
        if gpu_gen() < 17 {
            eprintln!("[instream] SKIP gpu gen {} < 17 (NA needs M5+)", gpu_gen());
            return;
        }
        let group_size: i32 = 64;
        let bits: i32 = 8;
        let affine_quantize = |w: &MxArray| -> (MxArray, MxArray, MxArray) {
            let mut q: *mut sys::mlx_array = std::ptr::null_mut();
            let mut s: *mut sys::mlx_array = std::ptr::null_mut();
            let mut b: *mut sys::mlx_array = std::ptr::null_mut();
            let ok = unsafe {
                sys::mlx_quantize(
                    w.as_raw_ptr(),
                    group_size,
                    bits,
                    c"affine".as_ptr(),
                    &mut q,
                    &mut s,
                    &mut b,
                )
            };
            assert!(ok, "mlx_quantize(affine) failed");
            (
                MxArray::from_handle(q, "instream:packed").unwrap(),
                MxArray::from_handle(s, "instream:scales").unwrap(),
                MxArray::from_handle(b, "instream:biases").unwrap(),
            )
        };
        let affine_qmm = |x: &MxArray, q: &MxArray, s: &MxArray, b: &MxArray| -> MxArray {
            let handle = unsafe {
                sys::mlx_quantized_matmul(
                    x.as_raw_ptr(),
                    q.as_raw_ptr(),
                    s.as_raw_ptr(),
                    b.as_raw_ptr(),
                    true,
                    group_size,
                    bits,
                    c"affine".as_ptr(),
                )
            };
            MxArray::from_handle(handle, "instream:qmm").unwrap()
        };

        // One weight set: (N, K) with std 1/sqrt(K) so the dependent chain's
        // activation magnitude stays ~constant across 50 rounds (no bf16
        // overflow/underflow that would change absmax/quant work mid-chain).
        struct WSet {
            k: i64,
            w_i8: MxArray,
            w_nk: MxArray,
            s_w: MxArray,
            q: MxArray,
            s: MxArray,
            b: MxArray,
        }
        let mk = |n: i64, k: i64| -> WSet {
            let std = 1.0 / (k as f64).sqrt();
            let w = MxArray::random_normal(&[n, k], 0.0, std, Some(DType::BFloat16)).unwrap();
            w.eval();
            let (w_i8, s_w) = quantize_weight_int8(&w).unwrap();
            w_i8.eval();
            s_w.eval();
            // [N,K] checkpoint orientation for the simd_sum decode kernel
            // (transpose+contiguous of the [K,N] operand — bit-exact).
            let w_nk = sym8_kernel_operand(&w_i8).unwrap();
            w_nk.eval();
            let (q, s, b) = affine_quantize(&w);
            q.eval();
            s.eval();
            b.eval();
            WSet {
                k,
                w_i8,
                w_nk,
                s_w,
                q,
                s,
                b,
            }
        };

        // chain(arm, weights, rounds): x0 [1, K0] -> rounds * (each WSet in
        // order) dependent M=1 calls, ONE eval at the end. Returns ms/chain.
        #[derive(Clone, Copy, PartialEq)]
        enum Arm {
            W8a16,
            W8a8,
            Affine,
        }
        let bench_chain = |arm: Arm, sets: &[&WSet], rounds: usize, iters: usize| -> f64 {
            let k0 = sets[0].k;
            let x0 = MxArray::random_normal(&[1, k0], 0.0, 1.0, Some(DType::BFloat16)).unwrap();
            x0.eval();
            synchronize();
            let run_once = || -> MxArray {
                let mut x = x0.clone();
                for _ in 0..rounds {
                    for ws in sets {
                        x = match arm {
                            Arm::W8a16 => int8_w8a16_qmv(&x, &ws.w_i8, &ws.w_nk, &ws.s_w).unwrap(),
                            Arm::W8a8 => int8_w8a8_qmv(&x, &ws.w_i8, &ws.s_w).unwrap(),
                            Arm::Affine => affine_qmm(&x, &ws.q, &ws.s, &ws.b),
                        };
                    }
                }
                x
            };
            // warm
            for _ in 0..3 {
                run_once().eval();
            }
            synchronize();
            let t = Instant::now();
            for _ in 0..iters {
                run_once().eval();
            }
            synchronize();
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        };

        // ---- PRODUCTION-SHAPED chain: 50 rounds x (gate_up, down, o_proj) ----
        let gate_up = mk(9216, 2560);
        let down = mk(2560, 9216);
        let o_proj = mk(2560, 2560);
        let sets: [&WSet; 3] = [&gate_up, &down, &o_proj];
        let rounds = 50usize; // 150 dependent M=1 calls / chain
        let calls = (rounds * sets.len()) as f64;
        let iters = 20usize;
        let median = |mut v: [f64; 3]| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[1]
        };
        let mut a16 = [0.0f64; 3];
        let mut aff = [0.0f64; 3];
        let mut a8 = [0.0f64; 3];
        for r in 0..3 {
            a16[r] = bench_chain(Arm::W8a16, &sets, rounds, iters);
            aff[r] = bench_chain(Arm::Affine, &sets, rounds, iters);
            a8[r] = bench_chain(Arm::W8a8, &sets, rounds, iters);
        }
        let (m16, maff, m8) = (median(a16), median(aff), median(a8));
        eprintln!(
            "[instream] 4B-shaped chain ({} calls): w8a16={m16:.3}ms affine={maff:.3}ms w8a8={m8:.3}ms",
            calls as usize
        );
        eprintln!(
            "[instream]   per-call: w8a16={:.2}us affine={:.2}us w8a8={:.2}us | delta(w8a16-affine)={:.2}us/call ratio={:.3}",
            m16 * 1e3 / calls,
            maff * 1e3 / calls,
            m8 * 1e3 / calls,
            (m16 - maff) * 1e3 / calls,
            m16 / maff
        );

        // ---- TINY chain (64x64): GPU work ~0 -> per-call NODE overhead ----
        let tiny = mk(64, 64);
        let tiny_sets: [&WSet; 1] = [&tiny];
        let tiny_rounds = 150usize;
        let tcalls = tiny_rounds as f64;
        let mut t16 = [0.0f64; 3];
        let mut taff = [0.0f64; 3];
        let mut t8 = [0.0f64; 3];
        for r in 0..3 {
            t16[r] = bench_chain(Arm::W8a16, &tiny_sets, tiny_rounds, iters);
            taff[r] = bench_chain(Arm::Affine, &tiny_sets, tiny_rounds, iters);
            t8[r] = bench_chain(Arm::W8a8, &tiny_sets, tiny_rounds, iters);
        }
        let (tm16, tmaff, tm8) = (median(t16), median(taff), median(t8));
        eprintln!(
            "[instream] tiny 64x64 chain ({} calls, GPU~0): w8a16={tm16:.3}ms affine={tmaff:.3}ms w8a8={tm8:.3}ms",
            tiny_rounds
        );
        eprintln!(
            "[instream]   per-call node overhead: w8a16={:.2}us affine={:.2}us w8a8={:.2}us | delta(w8a16-affine)={:.2}us/call",
            tm16 * 1e3 / tcalls,
            tmaff * 1e3 / tcalls,
            tm8 * 1e3 / tcalls,
            (tm16 - tmaff) * 1e3 / tcalls
        );
        // Attribution split: in-stream per-call delta = node-overhead delta
        // (tiny chain) + kernel GPU-time delta (the remainder).
        let total_delta = (m16 - maff) * 1e3 / calls;
        let node_delta = (tm16 - tmaff) * 1e3 / tcalls;
        eprintln!(
            "[instream] ATTRIBUTION per call: total={total_delta:.2}us node={node_delta:.2}us kernel-gpu={:.2}us ({}% node)",
            total_delta - node_delta,
            if total_delta.abs() > 1e-9 {
                (node_delta / total_delta * 100.0).round()
            } else {
                f64::NAN
            }
        );
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

    // ===================== AFFINE-GROUP W8A8 PARITY =====================
    // GATE (TDD — drives the new affine_w8a8_matmul op): the int8-activation x
    // EXACT-affine-weight kernel must equal the reference
    //   y_ref = x_rec @ dequant(packed_w)^T
    // where x_rec = s_x * x_q is x quantized to int8 EXACTLY the way the kernel
    // does (the SAME per-token symmetric quant as the symmetric W8A8 path — we
    // reuse act_quant_lazy to get the kernel's exact x_q + s_x), and
    // dequant(packed_w) is MLX's own affine dequant of the model's EXACT packed
    // weight (NO re-quantization of the weight). Because both sides apply the
    // identical x_q and the identical affine weight, the only error is bf16/f32
    // narrowing — so the gate is TIGHT: per-row cosine >= 0.9995 AND small max
    // relative error.
    //
    // This test will NOT compile until the Implement phase adds
    // `affine_w8a8_matmul` (TDD). Do not run before then.
    #[test]
    fn affine_w8a8_cosine_parity() {
        if gpu_gen() < 17 {
            eprintln!(
                "[affine] SKIP: gpu gen {} < 17 (NA matmul2d needs M5+)",
                gpu_gen()
            );
            return;
        }

        let group_size: i32 = 64;
        let bits: i32 = 8;
        // N=128, K=256: K % group_size == 0 (4 groups/row); K_packed = K/4 = 64.
        let n: usize = 128;
        let k: usize = 256;
        let m: usize = 256; // realistic prefill tile, >= 256 per spec.
        let gs = group_size as usize;
        let groups_per_row = k / gs; // 4
        // MLX affine bits=8 packs 4 consecutive uint8 q-values per uint32 along K
        // (el_per_int = 32/bits = 4), low element in the low byte (shift 0,8,16,24).
        // See crates/mlx-sys/mlx/mlx/backend/cpu/quantized.cpp `quantize()`:
        //   out_el |= (uint64_t)w_el << (k * bits)  for k in [0,4), bits=8.
        let pack_factor = 32 / bits as usize; // 4
        let k_packed = k / pack_factor; // 64

        let mut state: u64 = 0xa771_4e8e_d00d_b16e;

        // ---- Build the affine-quantized weight DIRECTLY (no re-quant) ----
        // uint8 q[N,K] in [0,255].
        let mut qv = vec![0u8; n * k];
        for v in qv.iter_mut() {
            *v = next_int(&mut state, 0, 255) as u8;
        }
        // Pack to uint32 [N, K_packed]: packed[ni, j] holds q[ni,4j..4j+4],
        // low element in the low byte.
        let mut packed = vec![0u32; n * k_packed];
        for ni in 0..n {
            for j in 0..k_packed {
                let mut w_el: u32 = 0;
                for p in 0..pack_factor {
                    let q = qv[ni * k + j * pack_factor + p] as u32;
                    w_el |= q << (p * bits as usize);
                }
                packed[ni * k_packed + j] = w_el;
            }
        }
        let packed_w = MxArray::from_uint32(&packed, &[n as i64, k_packed as i64]).unwrap();
        packed_w.eval();
        assert_eq!(
            packed_w.dtype().unwrap(),
            DType::Uint32,
            "packed affine weight must be uint32"
        );

        // f32 scales / biases [N, K/group_size]. Realistic affine ranges:
        // scale ~ small positive-ish (can be either sign in MLX affine, but a
        // simple positive small scale is a valid affine weight); bias centers it.
        let mut sc = vec![0f32; n * groups_per_row];
        for v in sc.iter_mut() {
            // ~[0.001, 0.05]
            *v = (next_int(&mut state, 1, 50) as f32) / 1000.0;
        }
        let mut bi = vec![0f32; n * groups_per_row];
        for v in bi.iter_mut() {
            // ~[-0.5, 0.5] so dequant weight ~ bias + scale*q spans a real range.
            *v = (next_int(&mut state, -500, 500) as f32) / 1000.0;
        }
        let scales = MxArray::from_float32(&sc, &[n as i64, groups_per_row as i64]).unwrap();
        let biases = MxArray::from_float32(&bi, &[n as i64, groups_per_row as i64]).unwrap();
        scales.eval();
        biases.eval();

        // ---- Random bf16 activation x [M,K] with realistic magnitudes + outliers ----
        let mut xf = vec![0f32; m * k];
        for v in xf.iter_mut() {
            *v = next_int(&mut state, -200, 200) as f32 / 1000.0;
        }
        for mi in 0..m {
            let col = next_int(&mut state, 0, (k - 1) as i32) as usize;
            xf[mi * k + col] = if mi % 2 == 0 { 1.4 } else { -1.1 };
        }
        let x = MxArray::from_float32(&xf, &[m as i64, k as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        x.eval();

        // ---- REFERENCE: dequant the EXACT affine weight to f32 [N,K] ----
        // out_dtype = Float32 (=0) so the dense table is f32 regardless of the
        // f32 scales/biases dtype. mode = "affine".
        let w_deq_handle = unsafe {
            sys::mlx_dequantize(
                packed_w.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases.as_raw_ptr(),
                group_size,
                bits,
                DType::Float32 as i32, // f32 dense table
                c"affine".as_ptr(),
            )
        };
        assert!(!w_deq_handle.is_null(), "mlx_dequantize(affine) failed");
        let w_deq = MxArray::from_handle(w_deq_handle, "affine_dequant").unwrap();
        w_deq.eval();
        assert_eq!(w_deq.shape_at(0).unwrap(), n as i64);
        assert_eq!(w_deq.shape_at(1).unwrap(), k as i64);

        // ---- REFERENCE: quantize x to int8 the SAME way the kernel does ----
        // act_quant_lazy returns the kernel's EXACT per-token int8 x_q (widened to
        // int32) and s_x f32 [M,1]; reconstruct x_rec = s_x * x_q in f32 so the
        // reference applies the identical activation quantization the kernel uses.
        let (xq_i32, s_x_arr) = act_quant_lazy(&x).unwrap();
        xq_i32.eval();
        s_x_arr.eval();
        let xq = xq_i32.to_int32().unwrap();
        let xq: &[i32] = &xq;
        let sx = s_x_arr.to_float32().unwrap();
        let sx: &[f32] = &sx;
        assert_eq!(xq.len(), m * k);
        assert_eq!(sx.len(), m);
        let mut xrec = vec![0f32; m * k];
        for mi in 0..m {
            let s = sx[mi];
            for ki in 0..k {
                xrec[mi * k + ki] = s * xq[mi * k + ki] as f32;
            }
        }
        let x_rec = MxArray::from_float32(&xrec, &[m as i64, k as i64]).unwrap();
        x_rec.eval();

        // y_ref = x_rec @ w_deq^T  (f32 matmul; w_deq is [N,K] so transpose to [K,N]).
        let wt = w_deq.transpose(Some(&[1, 0])).unwrap();
        let y_ref = x_rec.matmul(&wt).unwrap();
        y_ref.eval();

        // ---- KERNEL: the new affine-group W8A8 op ----
        let y = affine_w8a8_matmul(&x, &packed_w, &scales, &biases, group_size, bits).unwrap();
        y.eval();
        assert_eq!(
            y.dtype().unwrap(),
            DType::BFloat16,
            "affine W8A8 output must be bf16"
        );

        let got = y.astype(DType::Float32).unwrap().to_float32().unwrap();
        let got: &[f32] = &got;
        let refv = y_ref.to_float32().unwrap();
        let refv: &[f32] = &refv;
        assert_eq!(got.len(), m * n);
        assert_eq!(refv.len(), m * n);

        // Per-row cosine + max relative error. Both sides apply the SAME x_q and
        // the SAME affine weight, so they differ only by bf16/f32 narrowing.
        let mut min_cos = f64::INFINITY;
        let mut sum_cos = 0.0f64;
        let mut max_rel = 0.0f64;
        for mi in 0..m {
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            let mut row_absmax = 0.0f64;
            for ni in 0..n {
                let a = got[mi * n + ni] as f64;
                let b = refv[mi * n + ni] as f64;
                dot += a * b;
                na += a * a;
                nb += b * b;
                row_absmax = row_absmax.max(b.abs());
            }
            let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
            let cos = dot / denom;
            min_cos = min_cos.min(cos);
            sum_cos += cos;
            // Relative error normalized by the row's reference scale (robust to
            // near-zero individual elements).
            let rel_denom = row_absmax.max(1e-6);
            for ni in 0..n {
                let a = got[mi * n + ni] as f64;
                let b = refv[mi * n + ni] as f64;
                max_rel = max_rel.max((a - b).abs() / rel_denom);
            }
        }
        let mean_cos = sum_cos / m as f64;
        eprintln!(
            "[affine] N={n} K={k} M={m} gs={group_size}: min_row_cos={min_cos:.6} \
             mean_row_cos={mean_cos:.6} max_rel(row-norm)={max_rel:.6}"
        );
        assert!(
            min_cos >= 0.9995,
            "affine W8A8 per-row cosine below gate: min={min_cos:.6} (N={n} K={k} gs={group_size})"
        );
        // bf16 has ~8 mantissa bits (~1/256 rel); allow a small multiple for the
        // group-accumulated narrowing.
        assert!(
            max_rel <= 0.02,
            "affine W8A8 max relative error too large: {max_rel:.6} (N={n} K={k})"
        );
    }

    // ============== AFFINE-GROUP W8A8 FALLBACK (K % group_size != 0) ==============
    // GATE: the op must return Err cleanly (Rust falls back to bf16) when the shape
    // is unsupported. K % group_size != 0 is the affine-specific unsupported case
    // (group dequant requires K divisible by group_size). Here K=200, group_size=64
    // -> 200 % 64 == 8 != 0, so affine_w8a8_matmul must Err (the C++ op returns
    // false). We build a minimally-valid packed weight of the WRONG K so the only
    // reason to fail is the K%group_size gate.
    #[test]
    fn affine_w8a8_fallback_bad_k() {
        if gpu_gen() < 17 {
            eprintln!("[affine-fb] SKIP: gpu gen {} < 17", gpu_gen());
            return;
        }
        let group_size: i32 = 64;
        let bits: i32 = 8;
        let n: usize = 32;
        let k: usize = 200; // 200 % 64 == 8 != 0  -> unsupported affine shape.
        let m: usize = 256;
        let groups_per_row = k.div_ceil(group_size as usize); // ceil so scales fit
        let pack_factor = 32 / bits as usize; // 4
        // K=200 is divisible by 4, so K_packed is whole; the affine gate (not the
        // pack gate) is what must reject this shape.
        let k_packed = k / pack_factor; // 50

        let mut state: u64 = 0x4444_5555_6666_7777;
        // Pack 4 random uint8 q-values per uint32 (same layout as the parity test);
        // the actual bytes are irrelevant — the op must reject on shape before use.
        let mut packed = vec![0u32; n * k_packed];
        for v in packed.iter_mut() {
            let mut w_el: u32 = 0;
            for p in 0..pack_factor {
                w_el |= (next_int(&mut state, 0, 255) as u32) << (p * bits as usize);
            }
            *v = w_el;
        }
        let packed_w = MxArray::from_uint32(&packed, &[n as i64, k_packed as i64]).unwrap();
        let sc = vec![0.01f32; n * groups_per_row];
        let bi = vec![0.0f32; n * groups_per_row];
        let scales = MxArray::from_float32(&sc, &[n as i64, groups_per_row as i64]).unwrap();
        let biases = MxArray::from_float32(&bi, &[n as i64, groups_per_row as i64]).unwrap();
        let x = MxArray::random_normal(&[m as i64, k as i64], 0.0, 0.05, Some(DType::BFloat16))
            .unwrap();
        packed_w.eval();
        scales.eval();
        biases.eval();
        x.eval();

        let res = affine_w8a8_matmul(&x, &packed_w, &scales, &biases, group_size, bits);
        assert!(
            res.is_err(),
            "affine_w8a8_matmul must Err on K%group_size != 0 (K={k} gs={group_size}) so the \
             caller falls back to bf16; got Ok"
        );
        eprintln!(
            "[affine-fb] OK: K={k} gs={group_size} (K%gs={}) -> Err (fallback)",
            k % group_size as usize
        );
    }
}
