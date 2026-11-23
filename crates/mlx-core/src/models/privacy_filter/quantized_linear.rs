//! Privacy-filter quantized projection helpers.
//!
//! The privacy-filter checkpoint may be quantized with one of four modes
//! (`affine`, `mxfp4`, `mxfp8`, `nvfp4`) via the `mlx convert -m
//! privacy-filter -q --q-mode <mode>` pipeline. The post-quantization
//! layout depends on which axis MLX packs along:
//!
//! - Attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and
//!   the router gate are stored `[out, in]` — the standard HF convention.
//!   They load directly into canonical [`LinearProj`] (`transpose = true`).
//! - MoE expert projections (`gate_up_proj`, `down_proj`) are stored
//!   `[E, in, out]` (privacy-filter ships them pre-transposed for
//!   `mx.gather_mm`). After quantization the OUT axis becomes packed:
//!   `[E, in, out_packed]`. They load into shared [`SwitchProj`] with
//!   `transpose = false` on both [`SwitchLinear`] and
//!   [`QuantizedSwitchLinear`].

use std::collections::HashMap;

use crate::array::MxArray;
use crate::nn::Linear;
use napi::bindgen_prelude::*;

pub use crate::models::quantized_linear::{LinearProj, QuantizedLinear, QuantizedSwitchLinear};
pub use crate::moe::{SwitchLinear, SwitchProj};

/// Per-tensor quantization parameters resolved from the `quantization`
/// block in `config.json`.
#[derive(Debug, Clone)]
pub struct TensorQuantParams {
    pub bits: i32,
    pub group_size: i32,
    pub mode: String,
}

/// Load a 2D linear projection (`q/k/v/o_proj`, `router`) as a canonical
/// [`LinearProj`].
pub fn load_linear_proj(
    tensors: &HashMap<String, MxArray>,
    prefix: &str,
    bias_key: Option<&str>,
    quant: Option<&TensorQuantParams>,
) -> Result<LinearProj> {
    let weight_key = format!("{prefix}.weight");
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
            let biases_key = format!("{prefix}.biases");
            let biases = tensors.get(&biases_key).cloned();
            Ok(LinearProj::Quantized(
                QuantizedLinear::new(
                    weight,
                    scales,
                    biases,
                    bias,
                    q.group_size,
                    q.bits,
                    q.mode.clone(),
                )
                .with_preserve_promoted_dtype(true),
            ))
        }
        None => {
            let weight = tensors
                .get(&weight_key)
                .ok_or_else(|| Error::from_reason(format!("missing tensor: {weight_key}")))?;
            Ok(LinearProj::Standard(Linear::from_weights(
                weight,
                bias.as_ref(),
            )?))
        }
    }
}

/// Load a 3D MoE expert projection (`gate_up_proj`, `down_proj`) stored in
/// `[E, in, out]` / `[E, in, out_packed]` layout (`transpose = false`) as a
/// shared [`SwitchProj`].
///
/// Unquantized MoE expert tensors in HF/mlx-convert privacy-filter checkpoints
/// are stored under the bare `<prefix>` key (no `.weight` suffix); this falls
/// back to the bare key when `<prefix>.weight` is absent.
pub fn load_moe_proj(
    tensors: &HashMap<String, MxArray>,
    prefix: &str,
    quant: Option<&TensorQuantParams>,
) -> Result<SwitchProj> {
    let weight_key = format!("{prefix}.weight");
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
            let biases_key = format!("{prefix}.biases");
            let biases = tensors.get(&biases_key).cloned();
            Ok(SwitchProj::Quantized(
                QuantizedSwitchLinear::new(
                    weight,
                    scales,
                    biases,
                    q.group_size,
                    q.bits,
                    q.mode.clone(),
                )
                .with_transpose(false)
                .with_preserve_promoted_dtype(true),
            ))
        }
        None => {
            let weight = tensors
                .get(&weight_key)
                .or_else(|| tensors.get(prefix))
                .cloned()
                .ok_or_else(|| {
                    Error::from_reason(format!(
                        "missing tensor: neither {weight_key} nor {prefix} present"
                    ))
                })?;
            Ok(SwitchProj::Standard(
                SwitchLinear::from_weight(weight).with_transpose(false),
            ))
        }
    }
}
