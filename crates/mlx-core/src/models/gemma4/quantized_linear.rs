use std::collections::HashMap;
use std::ffi::CString;

use crate::array::MxArray;
use crate::nn::{Activations, Linear};
use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::mlp::GemmaMLP;

// ---------------------------------------------------------------------------
// QuantizedSwitchLinear — Expert-indexed quantized linear using gather_qmm
// ---------------------------------------------------------------------------

/// QuantizedSwitchLinear: Expert-indexed quantized linear layer using gather_qmm.
///
/// Like the dense `QuantizedLinear`, but batched over experts: the weight tensor
/// has shape `[num_experts, out, in_packed]` and the forward pass dispatches
/// tokens to the correct expert slice via `rhs_indices`.
pub struct QuantizedSwitchLinear {
    weight: MxArray,         // Packed uint32 [num_experts, out, in_packed]
    scales: MxArray,         // Quantization scales [num_experts, out, groups]
    biases: Option<MxArray>, // Quantization biases (for affine mode)
    group_size: i32,
    bits: i32,
    mode: String,
}

impl QuantizedSwitchLinear {
    pub fn new(
        weight: MxArray,
        scales: MxArray,
        biases: Option<MxArray>,
        group_size: i32,
        bits: i32,
        mode: String,
    ) -> Self {
        Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
            mode,
        }
    }

    /// Forward pass using gather_qmm.
    ///
    /// `x`: [N, 1, hidden] — per-token input (already expanded + sorted/unsorted)
    /// `indices`: [N] — expert index for each token
    /// `sorted`: whether indices are pre-sorted for gather efficiency
    pub fn forward(&self, x: &MxArray, indices: &MxArray, sorted: bool) -> Result<MxArray> {
        let mode_c = CString::new(self.mode.as_str())
            .map_err(|e| Error::from_reason(format!("Invalid mode string: {}", e)))?;

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
                std::ptr::null_mut(), // lhs_indices (not used)
                indices.handle.0,
                true,
                self.group_size,
                self.bits,
                mode_c.as_ptr(),
                sorted,
            )
        };
        MxArray::from_handle(handle, "gather_qmm")
    }
}

/// Try to build an affine QuantizedSwitchLinear from weight/scales/biases keys.
pub fn try_build_quantized_switch_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
    group_size: i32,
    bits: i32,
) -> Option<QuantizedSwitchLinear> {
    let weight = params.get(&format!("{}.weight", key_prefix))?;
    let scales = params.get(&format!("{}.scales", key_prefix))?;
    let biases = params.get(&format!("{}.biases", key_prefix)).cloned();
    Some(QuantizedSwitchLinear::new(
        weight.clone(),
        scales.clone(),
        biases,
        group_size,
        bits,
        DEFAULT_QUANT_MODE.to_string(),
    ))
}

/// Try to build an MXFP8 QuantizedSwitchLinear from weight/scales keys.
pub fn try_build_mxfp8_quantized_switch_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
) -> Option<QuantizedSwitchLinear> {
    let weight = params.get(&format!("{}.weight", key_prefix))?;
    let scales = params.get(&format!("{}.scales", key_prefix))?;
    Some(QuantizedSwitchLinear::new(
        weight.clone(),
        scales.clone(),
        None,
        MXFP8_GROUP_SIZE,
        MXFP8_BITS,
        MXFP8_MODE.to_string(),
    ))
}

/// Default quantization parameters for 4-bit models.
pub const DEFAULT_QUANT_BITS: i32 = 4;
pub const DEFAULT_QUANT_GROUP_SIZE: i32 = 64;
pub const DEFAULT_QUANT_MODE: &str = "affine";

/// MXFP8 quantization parameters (for FP8 source checkpoints).
pub const MXFP8_BITS: i32 = 8;
pub const MXFP8_GROUP_SIZE: i32 = 32;
pub const MXFP8_MODE: &str = "mxfp8";

/// A linear projection that can be either standard or quantized.
pub enum LinearProj {
    Standard(Linear),
    Quantized(QuantizedLinear),
}

impl LinearProj {
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        match self {
            LinearProj::Standard(l) => l.forward(x),
            LinearProj::Quantized(l) => l.forward(x),
        }
    }

    pub fn set_weight(&mut self, w: &MxArray, name: &str) -> Result<()> {
        match self {
            LinearProj::Standard(l) => l.set_weight(w),
            LinearProj::Quantized(_) => Err(Error::from_reason(format!(
                "Cannot set weight on quantized {}",
                name
            ))),
        }
    }

    pub fn set_bias(&mut self, b: Option<&MxArray>, name: &str) -> Result<()> {
        match self {
            LinearProj::Standard(l) => l.set_bias(b),
            LinearProj::Quantized(_) => Err(Error::from_reason(format!(
                "Cannot set bias on quantized {}",
                name
            ))),
        }
    }

    pub fn set_quantized(&mut self, ql: QuantizedLinear) {
        *self = LinearProj::Quantized(ql);
    }

    pub fn get_weight(&self) -> MxArray {
        match self {
            LinearProj::Standard(l) => l.get_weight(),
            LinearProj::Quantized(ql) => ql.get_weight().clone(),
        }
    }
}

/// Gemma4 MLP variant: standard (GELU) or quantized.
pub enum Gemma4MLPVariant {
    Standard(GemmaMLP),
    Quantized {
        gate_proj: QuantizedLinear,
        up_proj: QuantizedLinear,
        down_proj: QuantizedLinear,
    },
}

impl Gemma4MLPVariant {
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        match self {
            Gemma4MLPVariant::Standard(mlp) => mlp.forward(x),
            Gemma4MLPVariant::Quantized {
                gate_proj,
                up_proj,
                down_proj,
            } => {
                let gate = gate_proj.forward(x)?;
                let up = up_proj.forward(x)?;
                let activated = Activations::gelu(&gate)?;
                let gated = activated.mul(&up)?;
                down_proj.forward(&gated)
            }
        }
    }

    pub fn set_gate_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        match self {
            Gemma4MLPVariant::Standard(mlp) => mlp.set_gate_proj_weight(w),
            Gemma4MLPVariant::Quantized { .. } => {
                Err(Error::from_reason("Cannot set weight on quantized MLP"))
            }
        }
    }

    pub fn set_up_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        match self {
            Gemma4MLPVariant::Standard(mlp) => mlp.set_up_proj_weight(w),
            Gemma4MLPVariant::Quantized { .. } => {
                Err(Error::from_reason("Cannot set weight on quantized MLP"))
            }
        }
    }

    pub fn set_down_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        match self {
            Gemma4MLPVariant::Standard(mlp) => mlp.set_down_proj_weight(w),
            Gemma4MLPVariant::Quantized { .. } => {
                Err(Error::from_reason("Cannot set weight on quantized MLP"))
            }
        }
    }
}

/// Check if a model checkpoint is quantized by looking for `.scales` keys.
pub fn is_quantized_checkpoint(params: &HashMap<String, MxArray>) -> bool {
    params.keys().any(|k| k.ends_with(".scales"))
}

/// Check if a checkpoint uses MXFP8 quantization (Uint8 scales = E8M0 format).
pub fn is_mxfp8_checkpoint(params: &HashMap<String, MxArray>) -> bool {
    params
        .iter()
        .any(|(k, v)| k.ends_with(".scales") && matches!(v.dtype(), Ok(crate::array::DType::Uint8)))
}

/// Try to build an MXFP8 QuantizedLinear from weight/scales keys in a params map.
pub fn try_build_mxfp8_quantized_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
) -> Option<QuantizedLinear> {
    let weight = params.get(&format!("{}.weight", key_prefix))?;
    let scales = params.get(&format!("{}.scales", key_prefix))?;
    Some(QuantizedLinear::new(
        weight.clone(),
        scales.clone(),
        None,
        None,
        MXFP8_GROUP_SIZE,
        MXFP8_BITS,
        MXFP8_MODE.to_string(),
    ))
}

/// Try to build an affine QuantizedLinear from weight/scales/biases keys.
pub fn try_build_quantized_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
    group_size: i32,
    bits: i32,
) -> Option<QuantizedLinear> {
    let weight = params.get(&format!("{}.weight", key_prefix))?;
    let scales = params.get(&format!("{}.scales", key_prefix))?;
    let biases = params.get(&format!("{}.biases", key_prefix)).cloned();
    Some(QuantizedLinear::new(
        weight.clone(),
        scales.clone(),
        biases,
        None,
        group_size,
        bits,
        DEFAULT_QUANT_MODE.to_string(),
    ))
}

/// QuantizedLinear: Linear layer using quantized_matmul for efficient inference.
pub struct QuantizedLinear {
    weight: MxArray,
    scales: MxArray,
    biases: Option<MxArray>,
    bias: Option<MxArray>,
    group_size: i32,
    bits: i32,
    mode: String,
}

impl QuantizedLinear {
    pub fn new(
        weight: MxArray,
        scales: MxArray,
        biases: Option<MxArray>,
        bias: Option<MxArray>,
        group_size: i32,
        bits: i32,
        mode: String,
    ) -> Self {
        Self {
            weight,
            scales,
            biases,
            bias,
            group_size,
            bits,
            mode,
        }
    }

    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let mode_c = CString::new(self.mode.as_str())
            .map_err(|e| Error::from_reason(format!("Invalid mode string: {}", e)))?;

        let biases_ptr = self
            .biases
            .as_ref()
            .map_or(std::ptr::null_mut(), |b| b.handle.0);

        let handle = unsafe {
            sys::mlx_quantized_matmul(
                x.handle.0,
                self.weight.handle.0,
                self.scales.handle.0,
                biases_ptr,
                true,
                self.group_size,
                self.bits,
                mode_c.as_ptr(),
            )
        };
        let mut result = MxArray::from_handle(handle, "quantized_matmul")?;

        if let Some(ref b) = self.bias {
            result = result.add(b)?;
        }

        Ok(result)
    }

    pub fn get_weight(&self) -> &MxArray {
        &self.weight
    }
}
