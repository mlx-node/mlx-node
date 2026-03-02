use std::collections::HashMap;
use std::ffi::CString;

use crate::array::MxArray;
use crate::nn::{Activations, Linear};
use crate::transformer::MLP;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// Default quantization parameters for 4-bit models.
pub const DEFAULT_QUANT_BITS: i32 = 4;
pub const DEFAULT_QUANT_GROUP_SIZE: i32 = 64;
/// Router gates use higher precision (8-bit affine, group_size=64).
pub const GATE_QUANT_BITS: i32 = 8;
pub const GATE_QUANT_GROUP_SIZE: i32 = 64;
pub const DEFAULT_QUANT_MODE: &str = "affine";

/// MXFP8 quantization parameters (for FP8 source checkpoints).
pub const MXFP8_BITS: i32 = 8;
pub const MXFP8_GROUP_SIZE: i32 = 32;
pub const MXFP8_MODE: &str = "mxfp8";

/// A linear projection that can be either standard or quantized.
///
/// Shared between attention, GatedDeltaNet, and SparseMoeBlock.
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
}

/// An MLP that can be either standard or quantized.
///
/// Shared between decoder_layer and sparse_moe.
pub enum MLPVariant {
    Standard(MLP),
    Quantized {
        gate_proj: QuantizedLinear,
        up_proj: QuantizedLinear,
        down_proj: QuantizedLinear,
    },
}

impl MLPVariant {
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        match self {
            MLPVariant::Standard(mlp) => mlp.forward(x),
            MLPVariant::Quantized {
                gate_proj,
                up_proj,
                down_proj,
            } => {
                let gate = gate_proj.forward(x)?;
                let up = up_proj.forward(x)?;
                let activated = Activations::swiglu(&gate, &up)?;
                down_proj.forward(&activated)
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
/// MXFP8 has no biases (only weight + scales).
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

/// Try to build a QuantizedLinear from weight/scales/biases keys in a params map.
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
///
/// Stores weights in packed uint32 format with separate scales and optional biases.
/// Uses MLX's fused dequantize+matmul Metal kernel for ~4x memory reduction.
pub struct QuantizedLinear {
    weight: MxArray,         // Packed uint32 quantized weights [out, in_packed]
    scales: MxArray,         // Quantization scales
    biases: Option<MxArray>, // Quantization biases (for affine mode)
    bias: Option<MxArray>,   // Linear bias (additive)
    group_size: i32,
    bits: i32,
    mode: String, // "affine" or "none"
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

    /// Forward pass using quantized_matmul.
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
                true, // transpose
                self.group_size,
                self.bits,
                mode_c.as_ptr(),
            )
        };
        let mut result = MxArray::from_handle(handle, "quantized_matmul")?;

        // Add linear bias if present
        if let Some(ref b) = self.bias {
            result = result.add(b)?;
        }

        Ok(result)
    }

    pub fn set_weight(&mut self, weight: MxArray) {
        self.weight = weight;
    }

    pub fn set_scales(&mut self, scales: MxArray) {
        self.scales = scales;
    }

    pub fn set_biases(&mut self, biases: Option<MxArray>) {
        self.biases = biases;
    }

    pub fn set_bias(&mut self, bias: Option<MxArray>) {
        self.bias = bias;
    }

    pub fn get_weight(&self) -> &MxArray {
        &self.weight
    }

    pub fn get_scales(&self) -> &MxArray {
        &self.scales
    }

    pub fn get_biases(&self) -> Option<&MxArray> {
        self.biases.as_ref()
    }
}
