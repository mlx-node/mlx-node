use std::collections::HashMap;
use std::ffi::CString;

use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

// Re-export QuantizedLinear from the dense module so shared types (attention, GatedDeltaNet)
// get the same concrete type.
pub use crate::models::qwen3_5::quantized_linear::QuantizedLinear;
pub use crate::models::qwen3_5::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, DEFAULT_QUANT_MODE, GATE_QUANT_BITS,
    GATE_QUANT_GROUP_SIZE, LinearProj, MLPVariant, MXFP8_BITS, MXFP8_GROUP_SIZE, MXFP8_MODE,
    is_mxfp8_checkpoint, is_quantized_checkpoint, try_build_mxfp8_quantized_linear,
    try_build_quantized_linear,
};

/// QuantizedSwitchLinear: Expert-indexed quantized linear layer using gather_qmm.
///
/// Like SwitchLinear but with quantized weights for ~4x memory reduction.
/// Uses MLX's fused gather_qmm Metal kernel for efficient MoE inference.
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
                std::ptr::null_mut(),
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

    pub fn set_weight(&mut self, weight: MxArray) {
        self.weight = weight;
    }

    pub fn set_scales(&mut self, scales: MxArray) {
        self.scales = scales;
    }

    pub fn set_biases(&mut self, biases: Option<MxArray>) {
        self.biases = biases;
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

impl Clone for QuantizedSwitchLinear {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            scales: self.scales.clone(),
            biases: self.biases.clone(),
            group_size: self.group_size,
            bits: self.bits,
            mode: self.mode.clone(),
        }
    }
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
