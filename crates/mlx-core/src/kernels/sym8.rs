//! CUDA Sym8 (Hadamard-1024 + per-token/per-channel symmetric INT8 + IMMA GEMM) kernels.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::accelerator::{active_backend, caps, cuda_sym8_available};
use crate::array::{DType, MxArray};

const HADAMARD_BLOCK_SIZE: i64 = 1024;

#[inline]
fn require_cuda_sym8(op: &str) -> Result<()> {
    if !cuda_sym8_available() {
        return Err(Error::from_reason(format!(
            "sym8::{op} requires active CUDA GPU with SM >= 8.0 (backend={:?}, cuda_sm={:?})",
            active_backend(),
            caps().cuda_sm,
        )));
    }
    Ok(())
}

#[inline]
fn is_floating_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::BFloat16 | DType::Float16 | DType::Float32)
}

fn validate_signs(signs: Option<&MxArray>, k: i64, op: &str) -> Result<*mut sys::mlx_array> {
    let Some(s) = signs else {
        return Ok(std::ptr::null_mut());
    };
    let dtype = s.dtype()?;
    if dtype != DType::Float32 {
        return Err(Error::from_reason(format!(
            "sym8::{op}: signs must be Float32, got {dtype:?}"
        )));
    }
    let ndim = s.ndim()?;
    if ndim != 1 {
        return Err(Error::from_reason(format!(
            "sym8::{op}: signs must be 1-D [K={k}], got {ndim}-D"
        )));
    }
    let width = s.shape_at(0)?;
    if width != k {
        return Err(Error::from_reason(format!(
            "sym8::{op}: signs length {width} does not match K={k}"
        )));
    }
    if k % HADAMARD_BLOCK_SIZE != 0 {
        return Err(Error::from_reason(format!(
            "sym8::{op}: Hadamard-1024 rotation requires K={k} divisible by {HADAMARD_BLOCK_SIZE}"
        )));
    }
    Ok(s.as_raw_ptr())
}

/// Optional signed Hadamard-1024 rotation + per-token symmetric INT8 quantization.
///
/// - `x`: floating-point (`BF16`/`FP16`/`FP32`) with shape `[..., K]` (`ndim >= 2`).
/// - `signs`: optional `Float32 [K]`; when `Some`, multiplies by `signs` and runs
///   normalized FWHT-1024 (`K % 1024 == 0`); when `None`, skips FWHT-1024 (any `K >= 1`).
/// - Returns `(x_i8: Int8 [..., K], s_x: Float32 [...])`.
pub fn hadamard_quant_int8(x: &MxArray, signs: Option<&MxArray>) -> Result<(MxArray, MxArray)> {
    require_cuda_sym8("hadamard_quant_int8")?;
    let dtype = x.dtype()?;
    if !is_floating_dtype(dtype) {
        return Err(Error::from_reason(format!(
            "sym8::hadamard_quant_int8: expected floating-point input, got {dtype:?}"
        )));
    }
    let ndim = x.ndim()?;
    if ndim < 2 {
        return Err(Error::from_reason(format!(
            "sym8::hadamard_quant_int8: expected rank >= 2, got {ndim}"
        )));
    }
    let k = x.shape_at(ndim - 1)?;
    if k < 1 || k > i64::from(i32::MAX) {
        return Err(Error::from_reason(format!(
            "sym8::hadamard_quant_int8: invalid K={k}"
        )));
    }
    let m = x.size()? / (k as u64);
    if m < 1 || m > i32::MAX as u64 {
        return Err(Error::from_reason(format!(
            "sym8::hadamard_quant_int8: invalid M={m}"
        )));
    }
    let signs_ptr = validate_signs(signs, k, "hadamard_quant_int8")?;

    let mut out_xq = std::ptr::null_mut();
    let mut out_sx = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_cuda_sym8_hadamard_quant(x.as_raw_ptr(), signs_ptr, &mut out_xq, &mut out_sx)
    };
    if !ok || out_xq.is_null() || out_sx.is_null() {
        return Err(Error::from_reason(
            "sym8::hadamard_quant_int8: CUDA kernel launch failed",
        ));
    }
    Ok((
        MxArray::from_handle(out_xq, "sym8::hadamard_quant_int8:xq")?,
        MxArray::from_handle(out_sx, "sym8::hadamard_quant_int8:sx")?,
    ))
}

/// Fused SwiGLU (`silu(gate) * up`) + optional signed Hadamard-1024 + per-token INT8 quantization.
///
/// - `gate_up`: floating-point (`BF16`/`FP16`/`FP32`) with shape `[..., 2*K]` (`ndim >= 2`),
///   where each row holds `[gate (0..K), up (K..2*K)]`.
/// - `signs`: optional `Float32 [K]`; when `Some`, applies signed FWHT-1024 (`K % 1024 == 0`);
///   when `None`, skips FWHT-1024 (any `K >= 1`).
/// - Returns `(x_i8: Int8 [..., K], s_x: Float32 [...])`.
pub fn swiglu_hadamard_quant_int8(
    gate_up: &MxArray,
    signs: Option<&MxArray>,
) -> Result<(MxArray, MxArray)> {
    require_cuda_sym8("swiglu_hadamard_quant_int8")?;
    let dtype = gate_up.dtype()?;
    if !is_floating_dtype(dtype) {
        return Err(Error::from_reason(format!(
            "sym8::swiglu_hadamard_quant_int8: expected floating-point input, got {dtype:?}"
        )));
    }
    let ndim = gate_up.ndim()?;
    if ndim < 2 {
        return Err(Error::from_reason(format!(
            "sym8::swiglu_hadamard_quant_int8: expected rank >= 2, got {ndim}"
        )));
    }
    let two_k = gate_up.shape_at(ndim - 1)?;
    if two_k < 2 || two_k % 2 != 0 {
        return Err(Error::from_reason(format!(
            "sym8::swiglu_hadamard_quant_int8: last dim {two_k} must be a positive even integer"
        )));
    }
    let k = two_k / 2;
    if k > i64::from(i32::MAX) {
        return Err(Error::from_reason(format!(
            "sym8::swiglu_hadamard_quant_int8: K={k} exceeds i32::MAX"
        )));
    }
    let m = gate_up.size()? / (two_k as u64);
    if m < 1 || m > i32::MAX as u64 {
        return Err(Error::from_reason(format!(
            "sym8::swiglu_hadamard_quant_int8: invalid M={m}"
        )));
    }
    let signs_ptr = validate_signs(signs, k, "swiglu_hadamard_quant_int8")?;

    let mut out_xq = std::ptr::null_mut();
    let mut out_sx = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_cuda_sym8_swiglu_hadamard_quant(
            gate_up.as_raw_ptr(),
            signs_ptr,
            &mut out_xq,
            &mut out_sx,
        )
    };
    if !ok || out_xq.is_null() || out_sx.is_null() {
        return Err(Error::from_reason(
            "sym8::swiglu_hadamard_quant_int8: CUDA kernel launch failed",
        ));
    }
    Ok((
        MxArray::from_handle(out_xq, "sym8::swiglu_hadamard_quant_int8:xq")?,
        MxArray::from_handle(out_sx, "sym8::swiglu_hadamard_quant_int8:sx")?,
    ))
}

/// Fused-epilogue Sym8 INT8 Tensor-Core GEMM: `Y[m, n] = bf16((x_i8[m] . w_i8[n]) * s_x[m] * s_w[n])`.
///
/// - `x_i8`: `Int8 [..., K]` (`ndim >= 2`, `M = size / K >= 1`).
/// - `w_i8`: `Int8 [N, K]` (`ndim == 2`).
/// - `s_x`: `Float32` with `M` elements (e.g. `[...]` or `[M]`).
/// - `s_w`: `Float32 [N]`.
/// - Returns `BFloat16 [..., N]`.
pub fn imma_gemm_bf16(
    x_i8: &MxArray,
    w_i8: &MxArray,
    s_x: &MxArray,
    s_w: &MxArray,
) -> Result<MxArray> {
    require_cuda_sym8("imma_gemm_bf16")?;
    let x_dt = x_i8.dtype()?;
    let w_dt = w_i8.dtype()?;
    let sx_dt = s_x.dtype()?;
    let sw_dt = s_w.dtype()?;
    if x_dt != DType::Int8
        || w_dt != DType::Int8
        || sx_dt != DType::Float32
        || sw_dt != DType::Float32
    {
        return Err(Error::from_reason(format!(
            "sym8::imma_gemm_bf16: expected (Int8, Int8, Float32, Float32), got ({x_dt:?}, {w_dt:?}, {sx_dt:?}, {sw_dt:?})"
        )));
    }
    let x_ndim = x_i8.ndim()?;
    let w_ndim = w_i8.ndim()?;
    let sw_ndim = s_w.ndim()?;
    if x_ndim < 2 || w_ndim != 2 || sw_ndim != 1 {
        return Err(Error::from_reason(format!(
            "sym8::imma_gemm_bf16: invalid ranks x={x_ndim}, w={w_ndim}, s_w={sw_ndim}"
        )));
    }
    let n = w_i8.shape_at(0)?;
    let k = w_i8.shape_at(1)?;
    let x_k = x_i8.shape_at(x_ndim - 1)?;
    if n < 1 || k < 1 || x_k != k || n > i64::from(i32::MAX) || k > i64::from(i32::MAX) {
        return Err(Error::from_reason(format!(
            "sym8::imma_gemm_bf16: incompatible shapes x_k={x_k}, w=[{n}, {k}]"
        )));
    }
    let m = x_i8.size()? / (k as u64);
    if m < 1 || m > i32::MAX as u64 {
        return Err(Error::from_reason(format!(
            "sym8::imma_gemm_bf16: invalid M={m}"
        )));
    }
    if s_x.size()? != m {
        return Err(Error::from_reason(format!(
            "sym8::imma_gemm_bf16: s_x size {} != M={m}",
            s_x.size()?
        )));
    }
    if s_w.shape_at(0)? != n {
        return Err(Error::from_reason(format!(
            "sym8::imma_gemm_bf16: s_w len {} != N={n}",
            s_w.shape_at(0)?
        )));
    }

    let handle = unsafe {
        sys::mlx_cuda_sym8_imma_gemm(
            x_i8.as_raw_ptr(),
            w_i8.as_raw_ptr(),
            s_x.as_raw_ptr(),
            s_w.as_raw_ptr(),
        )
    };
    if handle.is_null() {
        return Err(Error::from_reason(
            "sym8::imma_gemm_bf16: CUDA kernel launch failed",
        ));
    }
    MxArray::from_handle(handle, "sym8::imma_gemm_bf16")
}

/// Bit-exact FP32 weight quantizer from packed affine Q8 (`group_size=32`) to Sym8.
///
/// - `w_u32`: `Uint32 [N, K/4]` (`K % 32 == 0`).
/// - `w_scales`: `Float32`/`BFloat16`/`Float16` `[N, K/32]`.
/// - `w_biases`: optional `Float32`/`BFloat16`/`Float16` `[N, K/32]` (`None` => `bias = 0.0f`).
/// - `signs`: optional `Float32 [K]`; when `Some`, applies signed FWHT-1024 (`K % 1024 == 0`);
///   when `None`, skips FWHT-1024.
/// - Returns `(w_i8: Int8 [N, K], s_w: Float32 [N])`.
pub fn quantize_weight_q8(
    w_u32: &MxArray,
    w_scales: &MxArray,
    w_biases: Option<&MxArray>,
    signs: Option<&MxArray>,
) -> Result<(MxArray, MxArray)> {
    require_cuda_sym8("quantize_weight_q8")?;
    let w_dt = w_u32.dtype()?;
    let sc_dt = w_scales.dtype()?;
    if w_dt != DType::Uint32 || !is_floating_dtype(sc_dt) {
        return Err(Error::from_reason(format!(
            "sym8::quantize_weight_q8: expected Uint32 weights and floating scales, got ({w_dt:?}, {sc_dt:?})"
        )));
    }
    if w_u32.ndim()? != 2 || w_scales.ndim()? != 2 {
        return Err(Error::from_reason(format!(
            "sym8::quantize_weight_q8: expected 2-D weights and scales, got {}-D and {}-D",
            w_u32.ndim()?,
            w_scales.ndim()?
        )));
    }
    let n = w_u32.shape_at(0)?;
    let k_u32 = w_u32.shape_at(1)?;
    let k = k_u32
        .checked_mul(4)
        .ok_or_else(|| Error::from_reason("sym8::quantize_weight_q8: K overflow"))?;
    if n < 1 || n > i64::from(i32::MAX) || k < 32 || k > i64::from(i32::MAX) || k % 32 != 0 {
        return Err(Error::from_reason(format!(
            "sym8::quantize_weight_q8: invalid shape N={n}, K={k} (requires K >= 32, K % 32 == 0)"
        )));
    }
    let groups = k / 32;
    if w_scales.shape_at(0)? != n || w_scales.shape_at(1)? != groups {
        return Err(Error::from_reason(format!(
            "sym8::quantize_weight_q8: w_scales shape [{}, {}] != [{n}, {groups}]",
            w_scales.shape_at(0)?,
            w_scales.shape_at(1)?
        )));
    }
    let biases_ptr = if let Some(b) = w_biases {
        let b_dt = b.dtype()?;
        if !is_floating_dtype(b_dt) || b.ndim()? != 2 {
            return Err(Error::from_reason(format!(
                "sym8::quantize_weight_q8: w_biases must be 2-D floating, got {b_dt:?} {}-D",
                b.ndim()?
            )));
        }
        if b.shape_at(0)? != n || b.shape_at(1)? != groups {
            return Err(Error::from_reason(format!(
                "sym8::quantize_weight_q8: w_biases shape [{}, {}] != [{n}, {groups}]",
                b.shape_at(0)?,
                b.shape_at(1)?
            )));
        }
        b.as_raw_ptr()
    } else {
        std::ptr::null_mut()
    };
    let signs_ptr = validate_signs(signs, k, "quantize_weight_q8")?;

    let mut out_q = std::ptr::null_mut();
    let mut out_s = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_cuda_sym8_quantize_weight_q8(
            w_u32.as_raw_ptr(),
            w_scales.as_raw_ptr(),
            biases_ptr,
            signs_ptr,
            &mut out_q,
            &mut out_s,
        )
    };
    if !ok || out_q.is_null() || out_s.is_null() {
        return Err(Error::from_reason(
            "sym8::quantize_weight_q8: CUDA kernel launch failed",
        ));
    }
    Ok((
        MxArray::from_handle(out_q, "sym8::quantize_weight_q8:q")?,
        MxArray::from_handle(out_s, "sym8::quantize_weight_q8:s")?,
    ))
}

/// Bit-exact FP32 weight quantizer from packed K-quant (`Q4_K` or `Q6_K`,
/// `RowMajor` or `Tiled64`) directly to Sym8.
///
/// - `w`: `Uint32 [N, K/8]` for Q4_K or `Uint32 [N, 3*K/16]` for Q6_K (`K >= 256`, `K % 256 == 0`).
/// - `scales`: `Uint8 [N, K/16]` for Q4_K or `Int8 [N, K/16]` for Q6_K.
/// - `biases`: `Float16 [N, K/128]` for Q4_K or `Float16 [N, K/256]` for Q6_K.
/// - `is_q6k`: `false` for Q4_K, `true` for Q6_K.
/// - `tiled`: `false` for RowMajor, `true` for Tiled64 (requires `N % 64 == 0`).
/// - `signs`: optional `Float32 [K]`; when `Some`, applies signed FWHT-1024 (`K % 1024 == 0`);
///   when `None`, skips FWHT-1024.
/// - Returns `(w_i8: Int8 [N, K], s_w: Float32 [N])`.
pub(crate) fn quantize_weight_kquant(
    w: &MxArray,
    scales: &MxArray,
    biases: &MxArray,
    is_q6k: bool,
    tiled: bool,
    signs: Option<&MxArray>,
) -> Result<(MxArray, MxArray)> {
    require_cuda_sym8("quantize_weight_kquant")?;
    let w_dt = w.dtype()?;
    let sc_dt = scales.dtype()?;
    let bi_dt = biases.dtype()?;
    let expected_sc_dt = if is_q6k { DType::Int8 } else { DType::Uint8 };
    if w_dt != DType::Uint32 || sc_dt != expected_sc_dt || bi_dt != DType::Float16 {
        return Err(Error::from_reason(format!(
            "sym8::quantize_weight_kquant: expected (Uint32, {expected_sc_dt:?}, Float16), got ({w_dt:?}, {sc_dt:?}, {bi_dt:?})"
        )));
    }
    if w.ndim()? != 2 || scales.ndim()? != 2 || biases.ndim()? != 2 {
        return Err(Error::from_reason(format!(
            "sym8::quantize_weight_kquant: expected 2-D weights/scales/biases, got {}-D, {}-D, {}-D",
            w.ndim()?,
            scales.ndim()?,
            biases.ndim()?
        )));
    }
    let n = w.shape_at(0)?;
    let w_cols = w.shape_at(1)?;
    let words_per_sb = if is_q6k { 48i64 } else { 32i64 };
    if n < 1 || n > i64::from(i32::MAX) || w_cols < words_per_sb || w_cols % words_per_sb != 0 {
        return Err(Error::from_reason(format!(
            "sym8::quantize_weight_kquant: invalid weight shape [{n}, {w_cols}] (words_per_sb={words_per_sb})"
        )));
    }
    if tiled && n % 64 != 0 {
        return Err(Error::from_reason(format!(
            "sym8::quantize_weight_kquant: Tiled64 requires N={n} divisible by 64"
        )));
    }
    let num_sb = w_cols / words_per_sb;
    let k = num_sb
        .checked_mul(256)
        .ok_or_else(|| Error::from_reason("sym8::quantize_weight_kquant: K overflow"))?;
    if k > i64::from(i32::MAX) {
        return Err(Error::from_reason(format!(
            "sym8::quantize_weight_kquant: K={k} exceeds i32::MAX"
        )));
    }
    let expected_s_cols = num_sb * 16;
    let expected_b_cols = if is_q6k { num_sb } else { num_sb * 2 };
    if scales.shape_at(0)? != n || scales.shape_at(1)? != expected_s_cols {
        return Err(Error::from_reason(format!(
            "sym8::quantize_weight_kquant: scales shape [{}, {}] != [{n}, {expected_s_cols}]",
            scales.shape_at(0)?,
            scales.shape_at(1)?
        )));
    }
    if biases.shape_at(0)? != n || biases.shape_at(1)? != expected_b_cols {
        return Err(Error::from_reason(format!(
            "sym8::quantize_weight_kquant: biases shape [{}, {}] != [{n}, {expected_b_cols}]",
            biases.shape_at(0)?,
            biases.shape_at(1)?
        )));
    }
    let signs_ptr = validate_signs(signs, k, "quantize_weight_kquant")?;

    let mut out_q = std::ptr::null_mut();
    let mut out_s = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_sym8_quantize_weight_kquant_cuda(
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
            is_q6k,
            tiled,
            signs_ptr,
            &mut out_q,
            &mut out_s,
        )
    };
    if !ok || out_q.is_null() || out_s.is_null() {
        unsafe {
            if !out_q.is_null() {
                sys::mlx_array_delete(out_q);
            }
            if !out_s.is_null() {
                sys::mlx_array_delete(out_s);
            }
        }
        return Err(Error::from_reason(
            "sym8::quantize_weight_kquant: CUDA kernel launch failed",
        ));
    }
    Ok((
        MxArray::from_handle(out_q, "sym8::quantize_weight_kquant:q")?,
        MxArray::from_handle(out_s, "sym8::quantize_weight_kquant:s")?,
    ))
}

/// Bit-exact FP32 weight quantizer from dense `BF16`/`FP16`/`FP32` `[N, K]` to Sym8.
///
/// - `w`: floating-point (`BF16`/`FP16`/`FP32`) `[N, K]`.
/// - `signs`: optional `Float32 [K]`; when `Some`, applies signed FWHT-1024 (`K % 1024 == 0`);
///   when `None`, skips FWHT-1024 (any `K >= 1`).
/// - Returns `(w_i8: Int8 [N, K], s_w: Float32 [N])`.
pub fn quantize_weight_dense(w: &MxArray, signs: Option<&MxArray>) -> Result<(MxArray, MxArray)> {
    require_cuda_sym8("quantize_weight_dense")?;
    let dt = w.dtype()?;
    if !is_floating_dtype(dt) {
        return Err(Error::from_reason(format!(
            "sym8::quantize_weight_dense: expected floating-point weight, got {dt:?}"
        )));
    }
    if w.ndim()? != 2 {
        return Err(Error::from_reason(format!(
            "sym8::quantize_weight_dense: expected 2-D [N, K] weight, got {}-D",
            w.ndim()?
        )));
    }
    let n = w.shape_at(0)?;
    let k = w.shape_at(1)?;
    if n < 1 || n > i64::from(i32::MAX) || k < 1 || k > i64::from(i32::MAX) {
        return Err(Error::from_reason(format!(
            "sym8::quantize_weight_dense: invalid shape [{n}, {k}]"
        )));
    }
    let signs_ptr = validate_signs(signs, k, "quantize_weight_dense")?;

    let mut out_q = std::ptr::null_mut();
    let mut out_s = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_cuda_sym8_quantize_weight_dense(w.as_raw_ptr(), signs_ptr, &mut out_q, &mut out_s)
    };
    if !ok || out_q.is_null() || out_s.is_null() {
        return Err(Error::from_reason(
            "sym8::quantize_weight_dense: CUDA kernel launch failed",
        ));
    }
    Ok((
        MxArray::from_handle(out_q, "sym8::quantize_weight_dense:q")?,
        MxArray::from_handle(out_s, "sym8::quantize_weight_dense:s")?,
    ))
}

/// Outputs of [`add_rmsnorm_hadamard_quant`].
pub struct Sym8NormHadQuantOutput {
    /// Residual sum `bf16(h + delta)` (`BFloat16 [..., H]`), present iff `delta` was `Some`.
    pub h_out: Option<MxArray>,
    /// Post-RMSNorm activation (`BFloat16 [..., H]`), present iff `write_normed` was `true`.
    pub normed_out: Option<MxArray>,
    /// Rotated/quantized activation (`Int8 [..., H]`).
    pub x_i8: MxArray,
    /// Per-token FP32 scales (`Float32 [...]`).
    pub s_x: MxArray,
}

/// Fused CUDA `(optional residual add) + RMSNorm + optional Hadamard-1024 + per-token INT8 quant`.
///
/// Returns `Ok(None)` when the fused contract does not apply (non-CUDA backend
/// or CUDA SM < 8.0, unsupported dtype/shape, `H < 16`, `H > 8192`, `H % 16 != 0`,
/// `signs` with `H % 1024 != 0`, or FFI decline) so callers fall back cleanly to
/// the unfused path.
///
/// - `w_is_f32 == false`: `Native` RMSNorm policy (`w_norm` must be `BFloat16 [H]`).
/// - `w_is_f32 == true`: `PreciseF32` RMSNorm policy (`w_norm` must be `Float32 [H]`).
pub fn add_rmsnorm_hadamard_quant(
    h: &MxArray,
    delta: Option<&MxArray>,
    w_norm: &MxArray,
    eps: f32,
    signs: Option<&MxArray>,
    write_normed: bool,
    w_is_f32: bool,
) -> Result<Option<Sym8NormHadQuantOutput>> {
    if !cuda_sym8_available() || !eps.is_finite() {
        return Ok(None);
    }
    if h.dtype()? != DType::BFloat16 {
        return Ok(None);
    }
    let ndim = h.ndim()?;
    if ndim < 2 {
        return Ok(None);
    }
    let h_dim = h.shape_at(ndim - 1)?;
    if !(16..=8192).contains(&h_dim) || h_dim % 16 != 0 {
        return Ok(None);
    }
    let m = h.size()? / (h_dim as u64);
    if m < 1 || m > i32::MAX as u64 {
        return Ok(None);
    }
    let expected_w_dt = if w_is_f32 {
        DType::Float32
    } else {
        DType::BFloat16
    };
    if w_norm.dtype()? != expected_w_dt || w_norm.ndim()? != 1 || w_norm.shape_at(0)? != h_dim {
        return Ok(None);
    }
    let delta_ptr = if let Some(d) = delta {
        if d.dtype()? != DType::BFloat16 || d.shape()?.as_ref() != h.shape()?.as_ref() {
            return Ok(None);
        }
        d.as_raw_ptr()
    } else {
        std::ptr::null_mut()
    };
    let signs_ptr = if let Some(s) = signs {
        if h_dim % HADAMARD_BLOCK_SIZE != 0
            || s.dtype()? != DType::Float32
            || s.ndim()? != 1
            || s.shape_at(0)? != h_dim
        {
            return Ok(None);
        }
        s.as_raw_ptr()
    } else {
        std::ptr::null_mut()
    };

    let mut out_h = std::ptr::null_mut();
    let mut out_normed = std::ptr::null_mut();
    let mut out_xq = std::ptr::null_mut();
    let mut out_sx = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_cuda_sym8_add_rmsnorm_hadamard_quant(
            h.as_raw_ptr(),
            delta_ptr,
            w_norm.as_raw_ptr(),
            signs_ptr,
            eps,
            write_normed,
            if delta.is_some() {
                &mut out_h
            } else {
                std::ptr::null_mut()
            },
            if write_normed {
                &mut out_normed
            } else {
                std::ptr::null_mut()
            },
            &mut out_xq,
            &mut out_sx,
        )
    };
    if !ok
        || out_xq.is_null()
        || out_sx.is_null()
        || (delta.is_some() && out_h.is_null())
        || (write_normed && out_normed.is_null())
    {
        unsafe {
            if !out_h.is_null() {
                sys::mlx_array_delete(out_h);
            }
            if !out_normed.is_null() {
                sys::mlx_array_delete(out_normed);
            }
            if !out_xq.is_null() {
                sys::mlx_array_delete(out_xq);
            }
            if !out_sx.is_null() {
                sys::mlx_array_delete(out_sx);
            }
        }
        return Ok(None);
    }

    let h_out = if delta.is_some() {
        Some(MxArray::from_handle(
            out_h,
            "sym8::add_rmsnorm_hadamard_quant:h_out",
        )?)
    } else {
        None
    };
    let normed_out = if write_normed {
        Some(MxArray::from_handle(
            out_normed,
            "sym8::add_rmsnorm_hadamard_quant:normed_out",
        )?)
    } else {
        None
    };
    let x_i8 = MxArray::from_handle(out_xq, "sym8::add_rmsnorm_hadamard_quant:xq")?;
    let s_x = MxArray::from_handle(out_sx, "sym8::add_rmsnorm_hadamard_quant:sx")?;
    Ok(Some(Sym8NormHadQuantOutput {
        h_out,
        normed_out,
        x_i8,
        s_x,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::quant_dispatch::kquant_tile_rows;
    use crate::utils::gguf_kquant::{KQuantFormat, KQuantScales, repack_kquant};

    fn next_u64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    fn make_kquant_blocks(format: KQuantFormat, rows: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let num_sb = k / 256;
        let blk_bytes = format.block_bytes();
        let mut blocks = vec![0u8; rows * num_sb * blk_bytes];
        for byte in &mut blocks {
            *byte = (next_u64(&mut rng) >> 24) as u8;
        }
        // Replace raw f16 d / dmin with finite non-zero values so random bit
        // patterns never accidentally inject NaN/Inf into the parity fixture.
        for b in 0..(rows * num_sb) {
            let base = b * blk_bytes;
            let u0 = ((next_u64(&mut rng) & 0xFFFF) as f32) / 65535.0;
            let u1 = ((next_u64(&mut rng) & 0xFFFF) as f32) / 65535.0;
            let sign = if (next_u64(&mut rng) & 1) == 0 {
                1.0f32
            } else {
                -1.0f32
            };
            match format {
                KQuantFormat::Q4K => {
                    let d = half::f16::from_f32(0.005 + 0.15 * u0).to_bits();
                    let dmin = half::f16::from_f32(0.002 + 0.10 * u1).to_bits();
                    blocks[base..base + 2].copy_from_slice(&d.to_le_bytes());
                    blocks[base + 2..base + 4].copy_from_slice(&dmin.to_le_bytes());
                }
                KQuantFormat::Q6K => {
                    let d = half::f16::from_f32(sign * (0.005 + 0.12 * u0)).to_bits();
                    blocks[base + 208..base + 210].copy_from_slice(&d.to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
        blocks
    }

    fn make_rademacher_signs(k: usize, seed: u64) -> Vec<f32> {
        let mut rng = seed.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1;
        (0..k)
            .map(|_| {
                if (next_u64(&mut rng) & 1) == 0 {
                    1.0f32
                } else {
                    -1.0f32
                }
            })
            .collect()
    }

    fn dequant_q4k_ref(
        w: &[u32],
        scales: &[u8],
        biases: &[u16],
        rows: usize,
        k: usize,
    ) -> Vec<f32> {
        let num_sb = k / 256;
        let mut out = Vec::with_capacity(rows * k);
        for r in 0..rows {
            for sb in 0..num_sb {
                let d = half::f16::from_bits(biases[r * num_sb * 2 + sb * 2]).to_f32();
                let dmin = half::f16::from_bits(biases[r * num_sb * 2 + sb * 2 + 1]).to_f32();
                for u in 0..8 {
                    let sc = scales[r * num_sb * 16 + sb * 16 + u * 2] as f32;
                    let m = scales[r * num_sb * 16 + sb * 16 + u * 2 + 1] as f32;
                    let scale = d * sc;
                    let minimum = dmin * m;
                    for q8 in 0..4 {
                        let word = w[r * num_sb * 32 + sb * 32 + u * 4 + q8];
                        for e in 0..8 {
                            let code = ((word >> (4 * e)) & 0xF) as f32;
                            out.push(scale * code - minimum);
                        }
                    }
                }
            }
        }
        out
    }

    fn dequant_q6k_ref(
        w: &[u32],
        scales: &[i8],
        biases: &[u16],
        rows: usize,
        k: usize,
    ) -> Vec<f32> {
        let num_sb = k / 256;
        let mut out = Vec::with_capacity(rows * k);
        for r in 0..rows {
            for sb in 0..num_sb {
                let d = half::f16::from_bits(biases[r * num_sb + sb]).to_f32();
                let sb_w = &w[r * num_sb * 48 + sb * 48..r * num_sb * 48 + (sb + 1) * 48];
                for v in 0..256 {
                    let sc = scales[r * num_sb * 16 + sb * 16 + v / 16] as f32;
                    let bit = v * 6;
                    let wi = bit / 32;
                    let bo = bit % 32;
                    let mut code = (sb_w[wi] >> bo) & 0x3F;
                    if bo + 6 > 32 {
                        code |= (sb_w[wi + 1] << (32 - bo)) & 0x3F;
                    }
                    out.push((d * sc) * ((code as i32 - 32) as f32));
                }
            }
        }
        out
    }

    fn fwht_1024_inplace(buf: &mut [f32]) {
        debug_assert_eq!(buf.len(), 1024);
        let mut h = 1usize;
        while h < 1024 {
            for i in (0..1024).step_by(h * 2) {
                for j in i..i + h {
                    let a = buf[j];
                    let b = buf[j + h];
                    buf[j] = a + b;
                    buf[j + h] = a - b;
                }
            }
            h <<= 1;
        }
        for x in buf.iter_mut() {
            *x *= 0.03125f32;
        }
    }

    fn quantize_rows_ref(
        vals: &[f32],
        signs: Option<&[f32]>,
        rows: usize,
        k: usize,
    ) -> (Vec<i8>, Vec<f32>) {
        let mut q_ref = Vec::with_capacity(rows * k);
        let mut s_ref = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = vals[r * k..(r + 1) * k].to_vec();
            if let Some(s) = signs {
                for (v, &sg) in row.iter_mut().zip(s.iter()) {
                    *v *= sg;
                }
                for chunk in row.as_chunks_mut::<1024>().0 {
                    fwht_1024_inplace(chunk);
                }
            }
            if row.iter().any(|v| !v.is_finite()) {
                s_ref.push(0.0f32);
                q_ref.extend(std::iter::repeat_n(0i8, k));
            } else {
                let amax = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let sw = amax.max(1e-12) / 127.0;
                s_ref.push(sw);
                for &v in &row {
                    let q = (v / sw).round_ties_even().clamp(-127.0, 127.0) as i8;
                    q_ref.push(q);
                }
            }
        }
        (q_ref, s_ref)
    }

    fn assert_sym8_parity(q_got: &[i8], s_got: &[f32], q_ref: &[i8], s_ref: &[f32], label: &str) {
        assert_eq!(q_got.len(), q_ref.len(), "{label}: q length");
        assert_eq!(s_got.len(), s_ref.len(), "{label}: s length");
        let max_q_diff = q_got
            .iter()
            .zip(q_ref.iter())
            .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
            .max()
            .unwrap_or(0);
        assert!(max_q_diff <= 1, "{label}: max int8 diff {max_q_diff} > 1");
        for (r, (&sg, &sr)) in s_got.iter().zip(s_ref.iter()).enumerate() {
            if sr == 0.0 {
                assert_eq!(sg, 0.0, "{label}: row {r} expected 0 scale");
            } else {
                let rel = (sg - sr).abs() / sr.abs().max(1e-12);
                assert!(
                    rel < 1e-4,
                    "{label}: row {r} scale got={sg} ref={sr} rel={rel}"
                );
            }
        }
    }

    fn check_kquant_format_parity(format: KQuantFormat, seed: u64) -> Result<()> {
        let is_q6k = format == KQuantFormat::Q6K;
        let (rows, k) = (128usize, 1024usize);
        let blocks = make_kquant_blocks(format, rows, k, seed);
        let arrays = repack_kquant(format, &blocks, rows, k)?;
        let r_i64 = rows as i64;

        let w_rm = MxArray::from_uint32(&arrays.weight, &[r_i64, format.weight_cols(k) as i64])?;
        let b_rm = MxArray::from_float16(&arrays.biases, &[r_i64, format.biases_cols(k) as i64])?;
        let (s_rm, fp32_ref) = match arrays.scales {
            KQuantScales::Unsigned(ref u) => {
                assert!(!is_q6k);
                let arr = MxArray::from_uint8(u, &[r_i64, format.scales_cols(k) as i64])?;
                let vals = dequant_q4k_ref(&arrays.weight, u, &arrays.biases, rows, k);
                (arr, vals)
            }
            KQuantScales::Signed(ref s) => {
                assert!(is_q6k);
                let arr = MxArray::from_int8(s, &[r_i64, format.scales_cols(k) as i64])?;
                let vals = dequant_q6k_ref(&arrays.weight, s, &arrays.biases, rows, k);
                (arr, vals)
            }
        };

        let w_t64 = kquant_tile_rows(&w_rm, format.bits() as i64)?;
        let s_t64 = kquant_tile_rows(&s_rm, 16)?;
        let b_t64 = kquant_tile_rows(&b_rm, format.bias_entries_per_super_block() as i64)?;
        MxArray::eval_arrays_with_context(&[&w_t64, &s_t64, &b_t64], "sym8 kquant tile")?;

        let signs_vec = make_rademacher_signs(k, seed ^ 0x5A5A);
        let signs_arr = MxArray::from_float32(&signs_vec, &[k as i64])?;

        for with_signs in [false, true] {
            let signs_opt = with_signs.then_some(&signs_arr);
            let (q_ref, s_ref) =
                quantize_rows_ref(&fp32_ref, with_signs.then_some(&signs_vec[..]), rows, k);

            let (q_rm, sw_rm) =
                quantize_weight_kquant(&w_rm, &s_rm, &b_rm, is_q6k, false, signs_opt)?;
            let (q_t64, sw_t64) =
                quantize_weight_kquant(&w_t64, &s_t64, &b_t64, is_q6k, true, signs_opt)?;
            MxArray::eval_arrays_with_context(&[&q_rm, &sw_rm, &q_t64, &sw_t64], "sym8 kquant")?;

            assert_eq!(q_rm.shape()?.to_vec(), vec![r_i64, k as i64]);
            assert_eq!(sw_rm.shape()?.to_vec(), vec![r_i64]);
            assert_eq!(q_rm.dtype()?, DType::Int8);
            assert_eq!(sw_rm.dtype()?, DType::Float32);

            let q_rm_v = q_rm.to_int8()?;
            let sw_rm_v = sw_rm.to_float32()?.to_vec();
            let q_t64_v = q_t64.to_int8()?;
            let sw_t64_v = sw_t64.to_float32()?.to_vec();

            let tag = format!("{format:?} signs={with_signs}");
            assert_sym8_parity(
                &q_rm_v,
                &sw_rm_v,
                &q_ref,
                &s_ref,
                &format!("{tag} RowMajor"),
            );
            assert_sym8_parity(
                &q_t64_v,
                &sw_t64_v,
                &q_ref,
                &s_ref,
                &format!("{tag} Tiled64"),
            );
            assert_eq!(q_rm_v, q_t64_v, "{tag}: Tiled64 q != RowMajor q");
            assert_eq!(sw_rm_v, sw_t64_v, "{tag}: Tiled64 s_w != RowMajor s_w");
        }
        Ok(())
    }

    #[test]
    fn quantize_weight_kquant_q4k_parity() -> Result<()> {
        if !cuda_sym8_available() {
            return Ok(());
        }
        check_kquant_format_parity(KQuantFormat::Q4K, 101)
    }

    #[test]
    fn quantize_weight_kquant_q6k_parity() -> Result<()> {
        if !cuda_sym8_available() {
            return Ok(());
        }
        check_kquant_format_parity(KQuantFormat::Q6K, 202)
    }

    #[test]
    fn quantize_weight_kquant_non_1024_and_edges() -> Result<()> {
        if !cuda_sym8_available() {
            return Ok(());
        }
        // K=768 (3 super-blocks, not a multiple of 4 warps) without signs, plus
        // an all-zero row (row 0) and a NaN-scale row (row 1).
        for format in [KQuantFormat::Q4K, KQuantFormat::Q6K] {
            let is_q6k = format == KQuantFormat::Q6K;
            let (rows, k) = (64usize, 768usize);
            let mut blocks = make_kquant_blocks(format, rows, k, 303 + u64::from(is_q6k));
            let row_bytes = format.row_bytes(k);
            // Row 0: all zeros.
            blocks[0..row_bytes].fill(0);
            // Row 1: inject quiet NaN into first super-block's d.
            let nan_bits = 0x7E00u16.to_le_bytes();
            let d_off = if is_q6k { 208 } else { 0 };
            blocks[row_bytes + d_off..row_bytes + d_off + 2].copy_from_slice(&nan_bits);

            let arrays = repack_kquant(format, &blocks, rows, k)?;
            let r_i64 = rows as i64;
            let w_rm =
                MxArray::from_uint32(&arrays.weight, &[r_i64, format.weight_cols(k) as i64])?;
            let b_rm =
                MxArray::from_float16(&arrays.biases, &[r_i64, format.biases_cols(k) as i64])?;
            let (s_rm, fp32_ref) = match arrays.scales {
                KQuantScales::Unsigned(ref u) => (
                    MxArray::from_uint8(u, &[r_i64, format.scales_cols(k) as i64])?,
                    dequant_q4k_ref(&arrays.weight, u, &arrays.biases, rows, k),
                ),
                KQuantScales::Signed(ref s) => (
                    MxArray::from_int8(s, &[r_i64, format.scales_cols(k) as i64])?,
                    dequant_q6k_ref(&arrays.weight, s, &arrays.biases, rows, k),
                ),
            };
            let w_t64 = kquant_tile_rows(&w_rm, format.bits() as i64)?;
            let s_t64 = kquant_tile_rows(&s_rm, 16)?;
            let b_t64 = kquant_tile_rows(&b_rm, format.bias_entries_per_super_block() as i64)?;
            MxArray::eval_arrays_with_context(&[&w_t64, &s_t64, &b_t64], "sym8 kquant edge tile")?;

            let (q_ref, s_ref) = quantize_rows_ref(&fp32_ref, None, rows, k);
            let (q_rm, sw_rm) = quantize_weight_kquant(&w_rm, &s_rm, &b_rm, is_q6k, false, None)?;
            let (q_t64, sw_t64) =
                quantize_weight_kquant(&w_t64, &s_t64, &b_t64, is_q6k, true, None)?;
            MxArray::eval_arrays_with_context(&[&q_rm, &sw_rm, &q_t64, &sw_t64], "sym8 edge")?;

            let q_rm_v = q_rm.to_int8()?;
            let sw_rm_v = sw_rm.to_float32()?.to_vec();
            let q_t64_v = q_t64.to_int8()?;
            let sw_t64_v = sw_t64.to_float32()?.to_vec();
            assert_sym8_parity(
                &q_rm_v,
                &sw_rm_v,
                &q_ref,
                &s_ref,
                &format!("{format:?} K=768 RM"),
            );
            assert_eq!(q_rm_v, q_t64_v);
            assert_eq!(sw_rm_v, sw_t64_v);
            // Row 1 had NaN -> s_w == 0 and all zeros in q.
            assert_eq!(sw_rm_v[1], 0.0);
            assert!(q_rm_v[k..2 * k].iter().all(|&v| v == 0));

            // Contract rejections: signs with K=768, wrong scales dtype, and Tiled64 with N%64!=0.
            let bad_signs = MxArray::from_float32(&vec![1.0f32; k], &[k as i64])?;
            assert!(
                quantize_weight_kquant(&w_rm, &s_rm, &b_rm, is_q6k, false, Some(&bad_signs))
                    .is_err()
            );
            assert!(quantize_weight_kquant(&w_rm, &s_rm, &b_rm, !is_q6k, false, None).is_err());
            let w_short = w_rm.slice_axis(0, 0, 32)?;
            let s_short = s_rm.slice_axis(0, 0, 32)?;
            let b_short = b_rm.slice_axis(0, 0, 32)?;
            assert!(
                quantize_weight_kquant(&w_short, &s_short, &b_short, is_q6k, true, None).is_err()
            );
        }
        Ok(())
    }
}
