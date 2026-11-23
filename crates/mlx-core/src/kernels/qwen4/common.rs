//! Shared weight descriptors and cross-backend compiled FFI ops for
//! Qwen3.8-Flash-Next (`qwen4_exp`), backed by `common/mlx_qwen4_common.cpp`.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::array::MxArray;

/// Routed expert weight banks for `qwen4_exp` MoE (`H=2560`, `D_FF=640`).
///
/// - `mode_gu`: `0` = RowMajor `Q4_K`, `1` = RowMajor `Q5_K`.
/// - `bits_dn`: `5` = RowMajor `Q5_1` (affine 5-bit g32), `8` = RowMajor `Q8_0Bias` (affine 8-bit g32).
#[allow(dead_code)]
pub(crate) struct Qwen4MoeRoutedWeights<'a> {
    pub gate_w: &'a MxArray,
    pub gate_s: &'a MxArray,
    pub gate_b: &'a MxArray,
    pub up_w: &'a MxArray,
    pub up_s: &'a MxArray,
    pub up_b: &'a MxArray,
    pub down_w: &'a MxArray,
    pub down_s: &'a MxArray,
    pub down_b: &'a MxArray,
    pub mode_gu: i32,
    pub bits_dn: i32,
}

/// Shared expert `Q8_0Bias` weight banks + pre-sigmoid `shared_gate` (`[.., 1]` BF16)
/// fused into `moe_routed_shared_few_cuda`.
#[allow(dead_code)]
pub(crate) struct Qwen4MoeSharedWeights<'a> {
    pub gate_w: &'a MxArray,
    pub gate_s: &'a MxArray,
    pub gate_b: &'a MxArray,
    pub up_w: &'a MxArray,
    pub up_s: &'a MxArray,
    pub up_b: &'a MxArray,
    pub down_w: &'a MxArray,
    pub down_s: &'a MxArray,
    pub down_b: &'a MxArray,
    pub shared_gate: &'a MxArray,
}

/// Optional fused `inject_proj` (`[4, 10240]`) weight descriptor for
/// [`super::cuda::hc_down_act_cuda`].
#[allow(dead_code)]
pub(crate) enum Qwen4HcInjectWeight<'a> {
    /// No fused inject projection (`inj_mode = 0`).
    None,
    /// Quantized `Q8_0Bias` (`[4, 2560]` u32, `[4, 320]` f16, `[4, 320]` f16; `inj_mode = 1`).
    Q8_0Bias {
        weight: &'a MxArray,
        scales: &'a MxArray,
        biases: &'a MxArray,
    },
    /// Dense BF16 `[4, 10240]` (`inj_mode = 2`).
    DenseBf16(&'a MxArray),
    /// Dense FP16 `[4, 10240]` (`inj_mode = 3`).
    DenseFp16(&'a MxArray),
    /// Dense FP32 `[4, 10240]` (`inj_mode = 4`).
    DenseFp32(&'a MxArray),
}

/// Compiled per-group RMSNorm (`mlx_qwen4_norm`).
///
/// Uses specialized Metal norm kernels on Metal when eligible (`group == 2560`
/// and `!centered`), falling back to the cross-backend compiled MLX graph.
#[inline]
pub(crate) fn norm(
    x: &MxArray,
    w: &MxArray,
    group: usize,
    eps: f64,
    centered: bool,
) -> Result<MxArray> {
    let out =
        unsafe { sys::mlx_qwen4_norm(x.as_raw_ptr(), w.as_raw_ptr(), group as i32, eps, centered) };
    MxArray::from_handle(out, "Qwen4 compiled normalization")
}

/// Compiled GDN output RMSNorm + sigmoid gate epilogue (`mlx_qwen4_gdn_epilogue`).
///
/// Returns `Ok(None)` when inputs are outside the kernel contract.
#[inline]
pub(crate) fn gdn_epilogue(
    out: &MxArray,
    z: &MxArray,
    weight: &MxArray,
    eps: f64,
) -> Result<Option<MxArray>> {
    let handle = unsafe {
        sys::mlx_qwen4_gdn_epilogue(out.as_raw_ptr(), z.as_raw_ptr(), weight.as_raw_ptr(), eps)
    };
    if handle.is_null() {
        return Ok(None);
    }
    MxArray::from_handle(handle, "Qwen4 prefill GDN epilogue").map(Some)
}

/// Compiled GDN decay and beta gate computation (`mlx_qwen4_gdn_gates`).
#[inline]
pub(crate) fn gdn_gates(
    a: &MxArray,
    b: &MxArray,
    scale: &MxArray,
    dt: &MxArray,
) -> Result<(MxArray, MxArray)> {
    let (mut decay, mut beta) = (std::ptr::null_mut(), std::ptr::null_mut());
    let ok = unsafe {
        sys::mlx_qwen4_gdn_gates(
            a.as_raw_ptr(),
            b.as_raw_ptr(),
            scale.as_raw_ptr(),
            dt.as_raw_ptr(),
            &mut decay,
            &mut beta,
        )
    };
    if !ok {
        return Err(Error::from_reason("Qwen4 compiled GDN gates failed"));
    }
    Ok((
        MxArray::from_handle(decay, "GDN decay")?,
        MxArray::from_handle(beta, "GDN beta")?,
    ))
}

/// Compiled residual stream injection (`mlx_qwen4_inject`): `x + y * g` across
/// hyper-connection streams.
#[inline]
pub(crate) fn inject(x: &MxArray, y: &MxArray, g: &MxArray) -> Result<MxArray> {
    let handle = unsafe { sys::mlx_qwen4_inject(x.as_raw_ptr(), y.as_raw_ptr(), g.as_raw_ptr()) };
    MxArray::from_handle(handle, "Qwen4 compiled residual injection")
}
