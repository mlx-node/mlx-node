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
