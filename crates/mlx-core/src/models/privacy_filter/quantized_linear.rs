//! Privacy-filter quantized projection helpers.
//!
//! The privacy-filter checkpoint may be quantized with one of four modes
//! (`affine`, `mxfp4`, `mxfp8`, `nvfp4`) via the `mlx convert -m
//! privacy-filter -q --q-mode <mode>` pipeline. The post-quantization
//! layout depends on which axis MLX packs along:
//!
//! - Attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and
//!   the router gate are stored `[out, in]` — exactly the Qwen / HF
//!   convention. `mlx.quantize` packs the last axis, so the post-quant
//!   layout is `[out, in_packed]` and `mlx_quantized_matmul` runs with
//!   `transpose = true` (computes `x @ w.T`).
//! - MoE expert projections (`gate_up_proj`, `down_proj`) are stored
//!   `[E, in, out]` (privacy-filter ships them in this orientation,
//!   which is what `mx.gather_mm` already expects on its right-hand
//!   side). After quantization the OUT axis becomes packed:
//!   `[E, in, out_packed]`. `mlx_gather_qmm` then runs with
//!   `transpose = false` (computes `x @ w` per expert).
//!
//! That second pattern means we cannot reuse Qwen3.5's
//! [`crate::models::qwen3_5_moe::quantized_linear::QuantizedSwitchLinear`]
//! verbatim — it hard-codes `transpose = true`. To avoid touching the
//! Qwen3.5 shipping path we introduce a small wrapper here:
//! [`PrivacyFilterQuantizedSwitchLinear`]. It mirrors the upstream type
//! one-for-one but with a configurable `transpose` flag.
//!
//! ## Loaded projection variants
//!
//! [`LoadedProj`] is the model-side enum that threads through
//! `attention.rs` / `experts.rs` — `Plain` for bf16 tensors and
//! `Quantized` for the four supported quantized modes. Construction
//! happens once in [`super::persistence::load_from_directory`] via
//! [`LoadedProj::from_tensors`].

use std::collections::HashMap;
use std::ffi::CString;

use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

pub use crate::models::qwen3_5::quantized_linear::QuantizedLinear;

/// Per-tensor quantization parameters resolved from the `quantization`
/// block in `config.json`. Constructed lazily by `LoadedProj::from_tensors`.
#[derive(Debug, Clone)]
pub struct TensorQuantParams {
    pub bits: i32,
    pub group_size: i32,
    pub mode: String,
}

/// A linear projection loaded from a privacy-filter checkpoint.
///
/// `Plain` holds a `[out, in]` (2D) or `[E, in, out]` (3D MoE) weight in
/// the model's native dtype (bf16). `Quantized` adds the companion
/// `.scales` / optional `.biases` tensors plus the resolved quantization
/// parameters. `bias` in either branch is the additive *linear* bias
/// (`{q,k,v,o}_proj.bias`, `router.bias`, or `None` for `gate_up_proj` /
/// `down_proj` where the per-expert bias is gathered separately).
pub enum LoadedProj {
    Plain {
        weight: MxArray,
        bias: Option<MxArray>,
    },
    Quantized {
        weight: MxArray,
        scales: MxArray,
        biases: Option<MxArray>,
        bias: Option<MxArray>,
        bits: i32,
        group_size: i32,
        mode: String,
    },
}

impl LoadedProj {
    /// Construct a [`LoadedProj`] from a tensor map.
    ///
    /// - `prefix`: the canonical module path (e.g.
    ///   `"model.layers.0.self_attn.q_proj"` or
    ///   `"model.layers.0.mlp.experts.gate_up_proj"`).
    /// - `bias_key`: explicit additive linear bias key (e.g.
    ///   `"...q_proj.bias"`). MoE expert projections have no linear
    ///   bias (per-expert biases are gathered separately by the
    ///   caller), so they pass `None`.
    /// - `quant`: resolved per-tensor quantization params, or `None` if
    ///   the tensor stays bf16.
    ///
    /// Key resolution:
    /// quantized checkpoints emit `<prefix>.weight` + `<prefix>.scales`
    /// (+ optional `<prefix>.biases`). Unquantized 2D attention
    /// projections were already stored as `<prefix>.weight` in the
    /// source HF checkpoint. Unquantized MoE expert tensors, however,
    /// are stored under the bare `<prefix>` key (no `.weight` suffix) —
    /// both `mlx convert` and the upstream HF dump keep that naming.
    /// We fall back to the bare key when `<prefix>.weight` is absent in
    /// the plain branch so the same call site works for both attention
    /// (`.weight`-suffixed) and MoE (bare).
    pub fn from_tensors(
        tensors: &HashMap<String, MxArray>,
        prefix: &str,
        bias_key: Option<&str>,
        quant: Option<&TensorQuantParams>,
    ) -> Result<Self> {
        let weight_key = format!("{prefix}.weight");
        // For MoE experts the additive linear bias is `None`; for
        // attention / router it's `<prefix>.bias`.
        let bias = bias_key.and_then(|k| tensors.get(k).cloned());
        match quant {
            Some(q) => {
                let weight = tensors.get(&weight_key).cloned().ok_or_else(|| {
                    Error::from_reason(format!(
                        "missing tensor: {weight_key} (quantization config says {prefix} is quantized)"
                    ))
                })?;
                let scales_key = format!("{prefix}.scales");
                let scales = tensors.get(&scales_key).cloned().ok_or_else(|| {
                    Error::from_reason(format!(
                        "missing tensor: {scales_key} (quantization config says {prefix} is quantized)"
                    ))
                })?;
                // `.biases` is present only in affine mode. For mxfp4 /
                // mxfp8 / nvfp4 the bias tensor is absent — we don't
                // fabricate one.
                let biases_key = format!("{prefix}.biases");
                let biases = tensors.get(&biases_key).cloned();
                Ok(LoadedProj::Quantized {
                    weight,
                    scales,
                    biases,
                    bias,
                    bits: q.bits,
                    group_size: q.group_size,
                    mode: q.mode.clone(),
                })
            }
            None => {
                // Try `<prefix>.weight` first (attention / router); fall
                // back to bare `<prefix>` (MoE expert tensors).
                let weight = tensors
                    .get(&weight_key)
                    .or_else(|| tensors.get(prefix))
                    .cloned()
                    .ok_or_else(|| {
                        Error::from_reason(format!(
                            "missing tensor: neither {weight_key} nor {prefix} present"
                        ))
                    })?;
                Ok(LoadedProj::Plain { weight, bias })
            }
        }
    }

    /// True if this projection is quantized.
    pub fn is_quantized(&self) -> bool {
        matches!(self, LoadedProj::Quantized { .. })
    }
}

/// Privacy-filter analog of [`crate::models::qwen3_5_moe::quantized_linear::QuantizedSwitchLinear`]
/// that supports the `[E, in, out_packed]` weight layout via `transpose
/// = false`. Used for both the fused gate-up projection and the down
/// projection, both of which ship in `[E, in, out]` orientation pre-
/// quantization.
pub struct PrivacyFilterQuantizedSwitchLinear {
    weight: MxArray,
    scales: MxArray,
    biases: Option<MxArray>,
    group_size: i32,
    bits: i32,
    mode: String,
    /// Whether the underlying weight is stored `[..., out, in]` (`true`)
    /// or `[..., in, out]` (`false`). For privacy-filter MoE experts
    /// this is always `false`.
    transpose: bool,
}

impl PrivacyFilterQuantizedSwitchLinear {
    pub fn new(
        weight: MxArray,
        scales: MxArray,
        biases: Option<MxArray>,
        group_size: i32,
        bits: i32,
        mode: String,
        transpose: bool,
    ) -> Self {
        Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
            mode,
            transpose,
        }
    }

    /// Forward via `mlx_gather_qmm`. `indices` selects the per-token
    /// expert id (sorted ascending when `sorted = true`). Output shape
    /// follows the unquantized `gather_mm` shape.
    pub fn forward(&self, x: &MxArray, indices: &MxArray, sorted: bool) -> Result<MxArray> {
        let mode_c = CString::new(self.mode.as_str())
            .map_err(|e| Error::from_reason(format!("Invalid mode string: {e}")))?;
        let biases_ptr = self
            .biases
            .as_ref()
            .map_or(std::ptr::null_mut(), |b| b.handle.0);
        let handle = unsafe {
            sys::mlx_gather_qmm(
                x.handle.0,
                self.weight.handle.0,
                self.scales.handle.0,
                biases_ptr,
                std::ptr::null_mut(),
                indices.handle.0,
                self.transpose,
                self.group_size,
                self.bits,
                mode_c.as_ptr(),
                sorted,
            )
        };
        MxArray::from_handle(handle, "privacy_filter gather_qmm")
    }
}

/// Run a 2D linear projection — either a plain matmul (with optional
/// additive bias) or a quantized matmul. Used by attention `q/k/v/o`
/// projections and the router gate.
///
/// For the `Plain` branch we use the same `addmm(bias, weight^T)`
/// fusion as the legacy code path. For the `Quantized` branch the
/// kernel handles the dequantization internally and we add the linear
/// bias afterwards (matches `QuantizedLinear::forward`).
pub fn project_2d(x: &MxArray, proj: &LoadedProj) -> Result<MxArray> {
    match proj {
        LoadedProj::Plain { weight, bias } => {
            let weight_t = weight.transpose(Some(&[1, 0]))?;
            match bias {
                Some(b) => x.addmm(b, &weight_t, None, None),
                None => x.matmul(&weight_t),
            }
        }
        LoadedProj::Quantized {
            weight,
            scales,
            biases,
            bias,
            bits,
            group_size,
            mode,
        } => {
            let mode_c = CString::new(mode.as_str())
                .map_err(|e| Error::from_reason(format!("Invalid mode string: {e}")))?;
            let biases_ptr = biases.as_ref().map_or(std::ptr::null_mut(), |b| b.handle.0);
            let handle = unsafe {
                sys::mlx_quantized_matmul(
                    x.handle.0,
                    weight.handle.0,
                    scales.handle.0,
                    biases_ptr,
                    true, // transpose=true: weight is [out, in_packed]
                    *group_size,
                    *bits,
                    mode_c.as_ptr(),
                )
            };
            let result = MxArray::from_handle(handle, "privacy_filter quantized_matmul")?;
            match bias {
                Some(b) => result.add(b),
                None => Ok(result),
            }
        }
    }
}

/// Run a MoE expert projection — either a plain `gather_mm` or the
/// quantized `gather_qmm` variant. The weight stays in `[E, in, out]`
/// orientation in both branches, so `transpose = false`. Output shape
/// matches the unquantized `gather_mm(x, w, indices, sorted)` shape.
pub fn project_moe(
    x: &MxArray,
    proj: &LoadedProj,
    indices: &MxArray,
    sorted: bool,
) -> Result<MxArray> {
    match proj {
        LoadedProj::Plain { weight, .. } => x.gather_mm(weight, indices, sorted),
        LoadedProj::Quantized {
            weight,
            scales,
            biases,
            bits,
            group_size,
            mode,
            ..
        } => {
            let ql = PrivacyFilterQuantizedSwitchLinear::new(
                weight.clone(),
                scales.clone(),
                biases.clone(),
                *group_size,
                *bits,
                mode.clone(),
                false, // weight is [E, in, out_packed] → transpose=false
            );
            ql.forward(x, indices, sorted)
        }
    }
}
