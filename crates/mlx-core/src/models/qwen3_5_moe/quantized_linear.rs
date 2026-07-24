use std::collections::HashMap;
use std::ffi::CString;

use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

// Re-export QuantizedLinear from the dense module so shared types (attention, GatedDeltaNet)
// get the same concrete type.
pub use crate::models::qwen3_5::quantized_linear::QuantizedLinear;
pub use crate::models::qwen3_5::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, DEFAULT_QUANT_MODE, FP8_E4M3_BITS,
    FP8_E4M3_GROUP_SIZE, FP8_E4M3_MODE, GATE_QUANT_BITS, GATE_QUANT_GROUP_SIZE, LinearProj,
    MLPVariant, MXFP4_BITS, MXFP4_GROUP_SIZE, MXFP4_MODE, MXFP8_BITS, MXFP8_GROUP_SIZE, MXFP8_MODE,
    NVFP4_BITS, NVFP4_GROUP_SIZE, NVFP4_MODE, PerLayerMode, PerLayerQuant, SYM8_BITS,
    SYM8_GROUP_SIZE, SYM8_MODE, is_mxfp8_checkpoint, is_quantized_checkpoint,
    try_build_fp8_e4m3_quantized_linear, try_build_kquant_quantized_linear,
    try_build_mxfp4_quantized_linear, try_build_mxfp8_quantized_linear,
    try_build_nvfp4_quantized_linear, try_build_quantized_linear, try_build_sym8_quantized_linear,
};

/// Expert-indexed linear backed by a serialized quantized weight format.
///
/// Affine/MX/NVFP modes use fused gather_qmm. Plain `fp8_e4m3` retains raw
/// Uint8 checkpoint storage, reconstructs a BF16 expert stack once at load,
/// and runs ordinary A16 gather_mm; it does not claim native W8A8.
pub struct QuantizedSwitchLinear {
    weight: MxArray,         // Packed uint32 [num_experts, out, in_packed]
    scales: MxArray,         // Quantization scales [num_experts, out, groups]
    biases: Option<MxArray>, // Quantization biases (for affine mode)
    group_size: i32,
    bits: i32,
    mode: String,
    // Pre-transposed BF16 `[E,K,N]` reconstruction for fp8_e4m3 fallback.
    fp8_dequant_weight_t: Option<MxArray>,
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
            fp8_dequant_weight_t: None,
        }
    }

    pub fn new_fp8_e4m3(weight: MxArray, scales: MxArray, dequant_weight_t: MxArray) -> Self {
        Self {
            weight,
            scales,
            biases: None,
            group_size: FP8_E4M3_GROUP_SIZE,
            bits: FP8_E4M3_BITS,
            mode: FP8_E4M3_MODE.to_string(),
            fp8_dequant_weight_t: Some(dequant_weight_t),
        }
    }

    /// Forward pass using gather_qmm.
    pub fn forward(&self, x: &MxArray, indices: &MxArray, sorted: bool) -> Result<MxArray> {
        if self.mode == FP8_E4M3_MODE {
            let weight_t = self.fp8_dequant_weight_t.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "plain FP8 QuantizedSwitchLinear missing load-time BF16 reconstruction",
                )
            })?;
            return x.gather_mm(weight_t, indices, sorted);
        }

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
        if self.mode == FP8_E4M3_MODE {
            self.fp8_dequant_weight_t = None;
        }
    }

    pub fn set_scales(&mut self, scales: MxArray) {
        self.scales = scales;
        if self.mode == FP8_E4M3_MODE {
            self.fp8_dequant_weight_t = None;
        }
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
            fp8_dequant_weight_t: self.fp8_dequant_weight_t.clone(),
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

/// Try to build an MXFP4 QuantizedSwitchLinear from weight/scales keys.
/// MXFP4 has no biases (only weight + uint8 E2M1 scales), fixed at 4 bits / group_size 32.
pub fn try_build_mxfp4_quantized_switch_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
) -> Option<QuantizedSwitchLinear> {
    let weight = params.get(&format!("{}.weight", key_prefix))?;
    let scales = params.get(&format!("{}.scales", key_prefix))?;
    Some(QuantizedSwitchLinear::new(
        weight.clone(),
        scales.clone(),
        None,
        MXFP4_GROUP_SIZE,
        MXFP4_BITS,
        MXFP4_MODE.to_string(),
    ))
}

/// Try to build an NVFP4 QuantizedSwitchLinear from weight/scales keys.
/// NVFP4 has no biases (only weight + uint8 E4M3 scales), fixed at 4 bits / group_size 16.
pub fn try_build_nvfp4_quantized_switch_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
) -> Option<QuantizedSwitchLinear> {
    let weight = params.get(&format!("{}.weight", key_prefix))?;
    let scales = params.get(&format!("{}.scales", key_prefix))?;
    Some(QuantizedSwitchLinear::new(
        weight.clone(),
        scales.clone(),
        None,
        NVFP4_GROUP_SIZE,
        NVFP4_BITS,
        NVFP4_MODE.to_string(),
    ))
}

/// Build a plain E4M3 expert stack and reconstruct it to BF16 once at load.
/// Storage is Uint8 `[E,N,K]` + floating `[E,N,1]`; runtime uses ordinary
/// A16 `gather_mm`, not native W8A8.
pub fn try_build_fp8_e4m3_quantized_switch_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
) -> Result<Option<QuantizedSwitchLinear>> {
    let weight = params.get(&format!("{key_prefix}.weight"));
    let scales = params.get(&format!("{key_prefix}.scales"));
    let (weight, scales) = match (weight, scales) {
        (None, None) => return Ok(None),
        (Some(_), None) => {
            return Err(Error::from_reason(format!(
                "plain FP8 expert layer '{key_prefix}': .weight present but mandatory .scales missing"
            )));
        }
        (None, Some(_)) => {
            return Err(Error::from_reason(format!(
                "plain FP8 expert layer '{key_prefix}': .scales present but .weight missing"
            )));
        }
        (Some(weight), Some(scales)) => (weight, scales),
    };
    if params.contains_key(&format!("{key_prefix}.biases")) {
        return Err(Error::from_reason(format!(
            "plain FP8 expert layer '{key_prefix}': unexpected .biases sidecar"
        )));
    }
    let dequant = crate::quant::fp8_weight::validate_and_dequantize(weight, scales, 3, key_prefix)?;
    let dequant_weight_t = dequant.transpose(Some(&[0, 2, 1]))?;
    Ok(Some(QuantizedSwitchLinear::new_fp8_e4m3(
        weight.clone(),
        scales.clone(),
        dequant_weight_t,
    )))
}

/// Try to build a ggml K-quant expert `QuantizedSwitchLinear` from the stacked
/// `{prefix}.weight` (uint32 `[E,out,in_packed]`), `{prefix}.scales` (int8 for
/// Q6_K / uint8 for Q4_K/Q5_K), and MANDATORY `{prefix}.biases` (float16 ggml
/// `d`). The three stacked companions are assembled by the MoE expert stacker
/// in `persistence.rs` using the SAME `.weight`/`.scales`/`.biases` suffixes as
/// every other quantized mode, so no stacker change is needed.
///
/// Fail-loud contract mirrors [`try_build_kquant_quantized_linear`]: `Ok(None)`
/// only when `.scales` is absent; every partial/malformed group is `Err`.
/// `forward` threads the resolved mode string into `mlx_gather_qmm`.
pub fn try_build_kquant_quantized_switch_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
    mode: PerLayerMode,
    family: &str,
) -> Result<Option<QuantizedSwitchLinear>> {
    let Some(group) =
        crate::models::quant_dispatch::resolve_kquant_group(params, key_prefix, mode, 3, family)?
    else {
        return Ok(None);
    };
    Ok(Some(QuantizedSwitchLinear::new(
        group.weight,
        group.scales,
        Some(group.biases),
        group.group_size,
        group.bits,
        group.mode_str.to_string(),
    )))
}

#[cfg(test)]
mod plain_fp8_expert_tests {
    use super::*;
    use crate::array::DType;

    #[test]
    fn plain_fp8_expert_builder_uses_bf16_gather_mm_fallback() {
        let values = (0..24).map(|i| (i as f32 - 12.0) / 8.0).collect::<Vec<_>>();
        let source = MxArray::from_float32(&values, &[2, 3, 4])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let (weight, scales) =
            crate::quant::fp8_weight::quantize_per_output_channel(&source, "experts").unwrap();
        let params = HashMap::from([
            ("experts.weight".into(), weight.clone()),
            ("experts.scales".into(), scales.clone()),
        ]);
        let qsl = try_build_fp8_e4m3_quantized_switch_linear(&params, "experts")
            .unwrap()
            .unwrap();
        assert_eq!(qsl.get_weight().dtype().unwrap(), DType::Uint8);
        assert_eq!(qsl.get_scales().shape().unwrap().to_vec(), vec![2, 3, 1]);

        let x = MxArray::from_float32(
            &[1.0, -0.5, 0.25, 2.0, -1.0, 0.75, 0.5, 1.25],
            &[2, 1, 1, 4],
        )
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
        let indices = MxArray::from_int32(&[0, 1], &[2, 1]).unwrap();
        let got = qsl.forward(&x, &indices, false).unwrap();
        let dequant =
            crate::quant::fp8_weight::validate_and_dequantize(&weight, &scales, 3, "experts")
                .unwrap();
        let want = x
            .gather_mm(
                &dequant.transpose(Some(&[0, 2, 1])).unwrap(),
                &indices,
                false,
            )
            .unwrap();
        got.eval();
        want.eval();
        assert_eq!(
            got.to_uint16_native().unwrap(),
            want.to_uint16_native().unwrap()
        );
    }

    #[test]
    fn plain_fp8_expert_builder_rejects_malformed_storage() {
        let weight = MxArray::from_uint8(&[0; 24], &[2, 3, 4]).unwrap();
        let scales = MxArray::from_float32(&[1.0, 1.0, 1.0], &[3, 1]).unwrap();
        let params = HashMap::from([
            ("experts.weight".into(), weight),
            ("experts.scales".into(), scales),
        ]);
        assert!(try_build_fp8_e4m3_quantized_switch_linear(&params, "experts").is_err());

        let wrong_weight_dtype = MxArray::from_float32(&[0.0; 24], &[2, 3, 4]).unwrap();
        let valid_scales = MxArray::from_float32(&[1.0; 6], &[2, 3, 1]).unwrap();
        let params = HashMap::from([
            ("experts.weight".into(), wrong_weight_dtype),
            ("experts.scales".into(), valid_scales.clone()),
        ]);
        assert!(try_build_fp8_e4m3_quantized_switch_linear(&params, "experts").is_err());

        let wrong_scale_dtype = MxArray::from_uint8(&[1; 6], &[2, 3, 1]).unwrap();
        let params = HashMap::from([
            (
                "experts.weight".into(),
                MxArray::from_uint8(&[0; 24], &[2, 3, 4]).unwrap(),
            ),
            ("experts.scales".into(), wrong_scale_dtype),
        ]);
        assert!(try_build_fp8_e4m3_quantized_switch_linear(&params, "experts").is_err());

        let params = HashMap::from([(
            "experts.weight".into(),
            MxArray::from_uint8(&[0; 24], &[2, 3, 4]).unwrap(),
        )]);
        assert!(try_build_fp8_e4m3_quantized_switch_linear(&params, "experts").is_err());
    }
}
