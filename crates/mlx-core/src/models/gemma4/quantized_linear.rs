use std::collections::HashMap;
use std::ffi::CString;

use crate::array::{DType, MxArray};
// The int8 W8A8/W8A16 kernels are family-agnostic; importing them from
// qwen3_5 follows the existing cross-family precedent
// (`gemma4::persistence` imports the shared `engine::persistence` helpers).
use crate::models::qwen3_5::int8_gemm;
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
        // Affine gather-QMM promotes mixed activation/sidecar dtypes to FP32.
        // Keep that arithmetic inside the projection, then restore the routed
        // activation dtype before expert outputs reach weighting/residuals.
        // MX/NV/K-quant modes already return `x`'s dtype and stay untouched.
        let activation_dtype = if self.mode == DEFAULT_QUANT_MODE {
            Some(x.dtype()?)
        } else {
            None
        };

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
        let mut result = MxArray::from_handle(handle, "gather_qmm")?;
        if let Some(dtype) = activation_dtype
            && result.dtype()? != dtype
        {
            result = result.astype(dtype)?;
        }
        Ok(result)
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

/// Try to build a ggml K-quant expert `QuantizedSwitchLinear` (gemma4-local
/// type) from the stacked uint32 `.weight`, int8/uint8 `.scales`, and mandatory
/// float16 `.biases`. Validation is delegated to the family-neutral
/// [`resolve_kquant_group`](crate::models::quant_dispatch::resolve_kquant_group)
/// so the gemma4 copy cannot drift from the qwen3_5 reference. `.biases` holds
/// the ggml `d` super-block scale, not an additive bias.
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

/// Default quantization parameters for 4-bit models.
pub const DEFAULT_QUANT_BITS: i32 = 4;
pub const DEFAULT_QUANT_GROUP_SIZE: i32 = 64;
pub const DEFAULT_QUANT_MODE: &str = "affine";

/// MXFP8 quantization parameters (for FP8 source checkpoints).
pub const MXFP8_BITS: i32 = 8;
pub const MXFP8_GROUP_SIZE: i32 = 32;
pub const MXFP8_MODE: &str = "mxfp8";

/// MXFP4 quantization parameters (E2M1 format, fixed bits/group_size).
pub const MXFP4_BITS: i32 = 4;
pub const MXFP4_GROUP_SIZE: i32 = 32;
pub const MXFP4_MODE: &str = "mxfp4";

/// NVFP4 quantization parameters (E2M1 4-bit weights with E4M3 uint8 scales,
/// group_size 16).
pub const NVFP4_BITS: i32 = 4;
pub const NVFP4_GROUP_SIZE: i32 = 16;
pub const NVFP4_MODE: &str = "nvfp4";

// Re-export PerLayerMode/PerLayerQuant from the family-neutral
// `quant_dispatch` module so gemma4 doesn't reach into the qwen3_5 internals
// for these shared types.
pub use crate::models::quant_dispatch::{PerLayerMode, PerLayerQuant};

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

/// Try to build an MXFP4 QuantizedLinear from weight/scales keys in a params map.
/// MXFP4 has no biases (only weight + uint8 E2M1 scales), fixed at 4 bits / group_size 32.
pub fn try_build_mxfp4_quantized_linear(
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
        MXFP4_GROUP_SIZE,
        MXFP4_BITS,
        MXFP4_MODE.to_string(),
    ))
}

/// Try to build an NVFP4 QuantizedLinear from weight/scales keys in a params map.
/// NVFP4 has no biases (only weight + uint8 E4M3 scales), fixed at 4 bits / group_size 16.
pub fn try_build_nvfp4_quantized_linear(
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
        NVFP4_GROUP_SIZE,
        NVFP4_BITS,
        NVFP4_MODE.to_string(),
    ))
}

/// Build the plain per-output E4M3 correctness fallback used by the fixed
/// Gemma4 DGX attention map.
///
/// Storage is strict: Uint8 `[N,K]` E4M3 bytes + floating `[N,1]` dequant
/// scales, with no affine `.biases`. The weight is reconstructed to BF16 once
/// at load and forward uses ordinary A16 matmul. Experts deliberately do not
/// use this path; their upstream class is NVFP4.
pub fn try_build_fp8_e4m3_quantized_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
) -> Result<Option<QuantizedLinear>> {
    let weight_key = format!("{key_prefix}.weight");
    let scales_key = format!("{key_prefix}.scales");
    let weight = params.get(&weight_key);
    let scales = params.get(&scales_key);
    let (weight, scales) = match (weight, scales) {
        (None, None) => return Ok(None),
        (Some(_), None) => {
            return Err(Error::from_reason(format!(
                "plain FP8 layer '{key_prefix}': .weight present but mandatory .scales missing"
            )));
        }
        (None, Some(_)) => {
            return Err(Error::from_reason(format!(
                "plain FP8 layer '{key_prefix}': .scales present but .weight missing"
            )));
        }
        (Some(weight), Some(scales)) => (weight, scales),
    };
    if params.contains_key(&format!("{key_prefix}.biases")) {
        return Err(Error::from_reason(format!(
            "plain FP8 layer '{key_prefix}': unexpected .biases sidecar"
        )));
    }
    let dequant_weight =
        crate::quant::fp8_weight::validate_and_dequantize(weight, scales, 2, key_prefix)?;
    Ok(Some(QuantizedLinear::new_fp8_e4m3(
        weight.clone(),
        scales.clone(),
        dequant_weight,
        None,
    )))
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

/// sym8 quantization parameters (per-output-channel symmetric int8 weights
/// with f32 `[N]` scales; `group_size` is null in the checkpoint and
/// meaningless at runtime — `SYM8_GROUP_SIZE` is a placeholder for the
/// struct field only).
pub const SYM8_BITS: i32 = 8;
pub const SYM8_GROUP_SIZE: i32 = -1;
pub const SYM8_MODE: &str = "sym8";

/// Try to build a sym8 `QuantizedLinear` from `{prefix}.weight` (int8 `[N,K]`)
/// + `{prefix}.scales` (f32 `[N]`) in a params map.
///
/// Returns `Ok(None)` ONLY when `{prefix}.scales` is absent — that is the
/// "this layer is not quantized" signal shared with the other `try_build_*`
/// helpers (a sym8-default checkpoint legitimately carries bf16 layers with
/// no sidecar, e.g. a forced-affine tensor that also failed the K%64 gate).
///
/// Everything else is FAIL-LOUD `Err` (convert should have prevented all of
/// these — assert anyway, a silent fallback would emit garbage):
///   * `.scales` present but `.weight` missing (corrupt checkpoint),
///   * a `.biases` sidecar (sym8 has none by construction),
///   * weight not 2-D int8, scales not 1-D f32, or `scales.len() != N`,
///   * `K % 16 != 0` (kernel contract),
///   * GPU gen < 17 (the int8 kernels need M5+; the convert-side
///     `sym8_eligible` deliberately omits this runtime-only gate).
///
/// The checkpoint-native `[N,K]` tensor is the only resident weight.
///
/// gemma4-local copy of the dense Qwen3.5 reference
/// (`crate::models::qwen3_5::quantized_linear::try_build_sym8_quantized_linear`)
/// — the validation chains must not drift.
pub fn try_build_sym8_quantized_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
) -> Result<Option<QuantizedLinear>> {
    let Some(scales) = params.get(&format!("{}.scales", key_prefix)) else {
        return Ok(None);
    };
    let Some(weight) = params.get(&format!("{}.weight", key_prefix)) else {
        return Err(Error::from_reason(format!(
            "sym8 layer '{}': .scales present but .weight missing (corrupt checkpoint)",
            key_prefix
        )));
    };
    if params.contains_key(&format!("{}.biases", key_prefix)) {
        return Err(Error::from_reason(format!(
            "sym8 layer '{}': unexpected .biases sidecar (sym8 is symmetric — convert never emits one)",
            key_prefix
        )));
    }

    let gpu_gen = unsafe { sys::mlx_gpu_architecture_gen() };
    if gpu_gen < 17 {
        return Err(Error::from_reason(format!(
            "sym8 layer '{}': sym8 checkpoints require an M5+ GPU (gen >= 17), got gen {}. \
             Re-convert the model with an affine quant mode for this host.",
            key_prefix, gpu_gen
        )));
    }

    let w_dtype = weight.dtype()?;
    if w_dtype != crate::array::DType::Int8 {
        return Err(Error::from_reason(format!(
            "sym8 layer '{}': expected int8 .weight, got {:?}",
            key_prefix, w_dtype
        )));
    }
    let w_shape = weight.shape()?;
    if w_shape.len() != 2 {
        return Err(Error::from_reason(format!(
            "sym8 layer '{}': expected 2-D [N,K] .weight, got {:?}",
            key_prefix,
            &w_shape[..]
        )));
    }
    let (n, k) = (w_shape[0], w_shape[1]);
    if k % 16 != 0 {
        return Err(Error::from_reason(format!(
            "sym8 layer '{}': K={} violates the kernel's K % 16 == 0 contract \
             (convert's sym8_eligible gate should have forced this layer to affine)",
            key_prefix, k
        )));
    }
    let s_dtype = scales.dtype()?;
    if s_dtype != crate::array::DType::Float32 {
        return Err(Error::from_reason(format!(
            "sym8 layer '{}': expected f32 .scales, got {:?}",
            key_prefix, s_dtype
        )));
    }
    let s_shape = scales.shape()?;
    if s_shape.len() != 1 || s_shape[0] != n {
        return Err(Error::from_reason(format!(
            "sym8 layer '{}': expected 1-D [N={}] .scales, got {:?}",
            key_prefix,
            n,
            &s_shape[..]
        )));
    }

    Ok(Some(QuantizedLinear::new_sym8(
        weight.clone(),
        scales.clone(),
        None,
    )))
}

/// Try to build a ggml K-quant `QuantizedLinear` (gemma4-local type) from
/// `{prefix}.weight` (uint32), `{prefix}.scales` (int8 for Q6_K / uint8 for
/// Q4_K/Q5_K), and the MANDATORY `{prefix}.biases` (float16 ggml `d`).
///
/// Fail-loud contract mirrors [`try_build_sym8_quantized_linear`] and the
/// qwen3_5 reference: `Ok(None)` only when `.scales` is absent; every partial
/// group is `Err`. Validation is delegated to the family-neutral
/// [`resolve_kquant_group`](crate::models::quant_dispatch::resolve_kquant_group).
pub fn try_build_kquant_quantized_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
    mode: PerLayerMode,
    family: &str,
) -> Result<Option<QuantizedLinear>> {
    let Some(group) =
        crate::models::quant_dispatch::resolve_kquant_group(params, key_prefix, mode, 2, family)?
    else {
        return Ok(None);
    };
    Ok(Some(QuantizedLinear::new(
        group.weight,
        group.scales,
        Some(group.biases),
        None,
        group.group_size,
        group.bits,
        group.mode_str.to_string(),
    )))
}

/// Linear layer backed by a serialized quantized weight format.
///
/// The sym8 surface (`s_w`, `new_sym8`, `forward_sym8`, the
/// `mode == "sym8"` dispatch) is a gemma4-local copy of the dense Qwen3.5
/// reference (`crate::models::qwen3_5::quantized_linear::QuantizedLinear`)
/// calling the same family-agnostic `int8_gemm` kernels. The two copies MUST
/// NOT drift: same M-boundary (M <= 2 -> W8A16 qmv, M >= 3 -> W8A8 gemm),
/// linear bias added AFTER the kernel, result narrowed to bf16 inside C++.
/// Plain `fp8_e4m3` is the non-native exception: it keeps raw Uint8 checkpoint
/// storage, reconstructs BF16 once at load, and uses ordinary A16 matmul.
///
/// The normal load path stores the pre-transposed `[K,N]` graph so decode
/// forwards can pass it directly to matmul, mirroring `nn::Linear::weight_t`.
/// `Source` is a compatibility fallback for the infallible public constructor:
/// if building the transpose graph fails, construction still succeeds and the
/// same error remains deferred until `forward()` as it was before the cache.
enum PlainFp8Weight {
    Transposed(MxArray),
    Source(MxArray),
}

impl PlainFp8Weight {
    #[cfg(test)]
    fn nbytes(&self) -> u64 {
        match self {
            Self::Transposed(weight) | Self::Source(weight) => weight.nbytes() as u64,
        }
    }
}

pub struct QuantizedLinear {
    weight: MxArray,
    scales: MxArray,
    biases: Option<MxArray>,
    bias: Option<MxArray>,
    group_size: i32,
    bits: i32,
    mode: String,
    // Reconstructed BF16 weight for the plain E4M3 correctness fallback.
    // Normally the cached `[K,N]` transpose; `Some` iff mode == fp8_e4m3.
    fp8_dequant_weight: Option<PlainFp8Weight>,
    // sym8 scale: `Some` iff mode == "sym8". Decode/prefill consume the
    // checkpoint-native `self.weight` [N,K] directly.
    s_w: Option<MxArray>,
}

/// Routing observability for the sym8 forward (unit-test scope only):
/// counts how many sym8 forwards took the QMV (decode) vs GEMM (prefill)
/// kernel, so tests can assert the M-dispatch without relying on the two
/// kernels producing different bits.
#[cfg(test)]
pub(crate) static SYM8_QMV_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
pub(crate) static SYM8_GEMM_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `MLX_SYM8_DEBUG=1` prints one line per sym8 forward with the chosen kernel
/// and the (M, K, N) shape — e2e dispatch evidence. Read once per process.
fn sym8_debug_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| match std::env::var("MLX_SYM8_DEBUG") {
        Ok(v) => !v.is_empty() && v != "0" && v != "false",
        Err(_) => false,
    })
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
            fp8_dequant_weight: None,
            s_w: None,
        }
    }

    /// Construct a plain E4M3 storage-backed linear with a load-time BF16
    /// reconstruction. The raw Uint8 tensor remains visible through
    /// `get_weight()` so storage identity is never confused with MXFP8.
    pub fn new_fp8_e4m3(
        weight: MxArray,
        scales: MxArray,
        dequant_weight: MxArray,
        bias: Option<MxArray>,
    ) -> Self {
        // Keep this constructor infallible for API compatibility. Validated
        // checkpoint weights are 2-D, so the load path takes `Transposed`.
        // For an invalid direct-constructor input, preserve the old behavior:
        // retain the source and let `forward()` surface the transpose error.
        let fp8_dequant_weight = if matches!(dequant_weight.ndim(), Ok(2)) {
            match dequant_weight.transpose(Some(&[1, 0])) {
                Ok(weight_t) => PlainFp8Weight::Transposed(weight_t),
                Err(_) => PlainFp8Weight::Source(dequant_weight),
            }
        } else {
            PlainFp8Weight::Source(dequant_weight)
        };
        Self {
            weight,
            scales,
            biases: None,
            bias,
            group_size: crate::quant::fp8_weight::FP8_E4M3_GROUP_SIZE,
            bits: crate::quant::fp8_weight::FP8_E4M3_BITS,
            mode: crate::quant::fp8_weight::FP8_E4M3_MODE.to_string(),
            fp8_dequant_weight: Some(fp8_dequant_weight),
            s_w: None,
        }
    }

    /// Construct a sym8 linear from pre-validated operands (see
    /// [`try_build_sym8_quantized_linear`] for the load-time validation).
    ///
    /// `weight` is the STORED int8 `[N,K]` checkpoint tensor (kept so
    /// `get_weight()` returns the source-layout tensor like every other
    /// mode — it shares the underlying buffer with the params map entry);
    /// `s_w` is the f32 `[N]` scale (doubling as the `scales` field).
    pub fn new_sym8(weight: MxArray, s_w: MxArray, bias: Option<MxArray>) -> Self {
        Self {
            weight,
            scales: s_w.clone(),
            biases: None,
            bias,
            group_size: SYM8_GROUP_SIZE,
            bits: SYM8_BITS,
            mode: SYM8_MODE.to_string(),
            fp8_dequant_weight: None,
            s_w: Some(s_w),
        }
    }

    /// sym8 forward: int8-weight GEMM/QMV + rescale.
    ///
    /// Dispatch rule: `M <= 2` → W8A16 decode (the dedicated
    /// W8A16 decode matvec — bf16 activations read directly, NO act quant,
    /// activation-exact), `M >= 3` → W8A8 prefill. Routing consumes only
    /// `self.weight` `[N,K]`. Keep in lockstep with Qwen3.5.
    fn forward_sym8(&self, x: &MxArray) -> Result<MxArray> {
        let Some(s_w) = self.s_w.as_ref() else {
            return Err(Error::from_reason(
                "sym8 QuantizedLinear missing per-channel scales — \
                 constructed without new_sym8?",
            ));
        };
        let shape = x.shape()?;
        if shape.is_empty() {
            return Err(Error::from_reason("sym8 forward: scalar input"));
        }
        let k = shape[shape.len() - 1];
        let m: i64 = shape[..shape.len() - 1].iter().product();
        let x2d = x.reshape(&[m, k])?;
        let y2d = if m <= 2 {
            #[cfg(test)]
            SYM8_QMV_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            int8_gemm::int8_w8a16_qmv_nk(&x2d, &self.weight, s_w)?
        } else {
            #[cfg(test)]
            SYM8_GEMM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            int8_gemm::int8_w8a8_matmul_nk(&x2d, &self.weight, s_w)?
        };
        let n = y2d.shape_at(1)?;
        if sym8_debug_enabled() {
            eprintln!(
                "[sym8] {} layout=nk M={m} K={k} N={n}",
                if m <= 2 { "qmv" } else { "gemm" }
            );
        }
        let mut out_shape: Vec<i64> = shape[..shape.len() - 1].to_vec();
        out_shape.push(n);
        let mut result = y2d.reshape(&out_shape)?;
        if let Some(ref b) = self.bias {
            result = result.add(b)?;
        }
        Ok(result)
    }

    fn forward_qmm(&self, x: &MxArray, activation_dtype: DType) -> Result<MxArray> {
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

        if result.dtype()? != activation_dtype {
            result = result.astype(activation_dtype)?;
        }

        Ok(result)
    }

    /// Forward pass using quantized_matmul (sym8 routes to the int8 W8A8
    /// kernels instead — `mlx_quantized_matmul` has no sym8 pack and its
    /// legacy no-biases heuristic would misread sym8 as MXFP8).
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        if self.mode == SYM8_MODE {
            return self.forward_sym8(x);
        }

        if self.mode == crate::quant::fp8_weight::FP8_E4M3_MODE {
            let weight = self.fp8_dequant_weight.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "plain FP8 QuantizedLinear missing load-time BF16 reconstruction",
                )
            })?;
            let fallback_weight_t;
            let weight_t = match weight {
                PlainFp8Weight::Transposed(weight_t) => weight_t,
                PlainFp8Weight::Source(weight) => {
                    fallback_weight_t = weight.transpose(Some(&[1, 0]))?;
                    &fallback_weight_t
                }
            };
            let mut result = x.matmul(weight_t)?;
            if let Some(ref b) = self.bias {
                result = result.add(b)?;
            }
            return Ok(result);
        }

        // MLX promotes affine QMM to FP32 when a BF16 activation is paired
        // with GGUF Q4_0's sidecars (including their exact load-time FP32
        // widening). Restore the model's activation dtype at the projection
        // boundary. Otherwise promoted K/V reaches the two-byte paged cache
        // and is rejected, while residual/MLP paths drift to FP32 too.
        let activation_dtype = x.dtype()?;

        self.forward_qmm(x, activation_dtype)
    }

    pub fn get_weight(&self) -> &MxArray {
        &self.weight
    }

    /// Quantization mode discriminator string ("affine", "mxfp8", "mxfp4",
    /// "nvfp4", "fp8_e4m3", or "sym8").
    pub fn mode(&self) -> &str {
        &self.mode
    }

    /// Additional model-owned bytes created by the plain-E4M3 correctness
    /// fallback and not represented by the serialized checkpoint tensors.
    ///
    /// The raw Uint8 weight and BF16 scales are already counted through the
    /// loader's params map; only the reconstructed BF16 weight is extra.
    #[cfg(test)]
    pub(crate) fn reconstructed_fp8_weight_bytes(&self) -> u64 {
        self.fp8_dequant_weight
            .as_ref()
            .map(PlainFp8Weight::nbytes)
            .unwrap_or(0)
    }

    /// Model-owned BF16 reconstruction retained by the plain-E4M3 fallback.
    /// Normally this is the cached `[K,N]` view used directly by forward.
    pub(crate) fn reconstructed_fp8_weight(&self) -> Option<&MxArray> {
        self.fp8_dequant_weight.as_ref().map(|weight| match weight {
            PlainFp8Weight::Transposed(weight) | PlainFp8Weight::Source(weight) => weight,
        })
    }

    /// Whether the plain-E4M3 reconstruction already has its `[K,N]` graph.
    #[cfg(test)]
    fn has_pretransposed_fp8_weight(&self) -> bool {
        matches!(
            self.fp8_dequant_weight.as_ref(),
            Some(PlainFp8Weight::Transposed(_))
        )
    }

    /// Test-scope accessor for the sym8 operands
    /// `(w_nk [N,K] checkpoint, s_w [N])`.
    /// Used by the routing/parity unit tests to call the reference kernels with
    /// the exact operands forward consumes.
    #[cfg(test)]
    pub(crate) fn sym8_operands(&self) -> Option<(&MxArray, &MxArray)> {
        self.s_w.as_ref().map(|s_w| (&self.weight, s_w))
    }
}

#[cfg(test)]
mod plain_fp8_weight_tests {
    use super::*;

    fn params(prefix: &str) -> HashMap<String, MxArray> {
        let source = MxArray::from_float32(
            &[
                0.0, 0.5, -1.0, 2.0, -0.25, 1.5, 0.75, -2.5, 1.0, -0.5, 0.25, 3.0,
            ],
            &[3, 4],
        )
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
        let (weight, scales) =
            crate::quant::fp8_weight::quantize_per_output_channel(&source, prefix).unwrap();
        HashMap::from([
            (format!("{prefix}.weight"), weight),
            (format!("{prefix}.scales"), scales),
        ])
    }

    #[test]
    fn plain_fp8_builder_reconstructs_bf16_and_forward_is_a16_matmul() {
        let p = params("proj");
        let ql = try_build_fp8_e4m3_quantized_linear(&p, "proj")
            .unwrap()
            .unwrap();
        assert_eq!(ql.mode(), crate::quant::fp8_weight::FP8_E4M3_MODE);
        assert_eq!(ql.get_weight().dtype().unwrap(), DType::Uint8);
        assert_eq!(
            ql.get_weight().as_raw_ptr(),
            p.get("proj.weight").unwrap().as_raw_ptr(),
            "get_weight must keep exposing the raw Uint8 checkpoint array"
        );
        assert!(
            ql.has_pretransposed_fp8_weight(),
            "valid plain-FP8 linears must cache the [K,N] graph at load"
        );
        assert_eq!(
            ql.reconstructed_fp8_weight_bytes(),
            3 * 4 * std::mem::size_of::<u16>() as u64,
            "the resident-byte delta must count the reconstructed BF16 weight"
        );

        let x = MxArray::from_float32(&[1.0, -0.5, 0.25, 2.0], &[1, 4])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let got = ql.forward(&x).unwrap();
        let dequant = crate::quant::fp8_weight::validate_and_dequantize(
            p.get("proj.weight").unwrap(),
            p.get("proj.scales").unwrap(),
            2,
            "proj",
        )
        .unwrap();
        let want = x
            .matmul(&dequant.transpose(Some(&[1, 0])).unwrap())
            .unwrap();
        got.eval();
        want.eval();
        assert_eq!(
            got.to_uint16_native().unwrap(),
            want.to_uint16_native().unwrap()
        );
    }

    #[test]
    fn plain_fp8_builder_fails_loud_on_incomplete_or_malformed_storage() {
        let mut missing_scales = params("proj");
        missing_scales.remove("proj.scales");
        assert!(try_build_fp8_e4m3_quantized_linear(&missing_scales, "proj").is_err());

        let mut wrong_weight_dtype = params("proj");
        let bad = wrong_weight_dtype["proj.weight"]
            .from_fp8(DType::BFloat16)
            .unwrap();
        wrong_weight_dtype.insert("proj.weight".into(), bad);
        assert!(try_build_fp8_e4m3_quantized_linear(&wrong_weight_dtype, "proj").is_err());

        let mut wrong_scale_shape = params("proj");
        wrong_scale_shape.insert(
            "proj.scales".into(),
            MxArray::from_float32(&[1.0, 1.0, 1.0], &[3]).unwrap(),
        );
        assert!(try_build_fp8_e4m3_quantized_linear(&wrong_scale_shape, "proj").is_err());
    }

    #[test]
    fn plain_fp8_direct_constructor_defers_malformed_source() {
        let ql = QuantizedLinear::new_fp8_e4m3(
            MxArray::from_uint8(&[0], &[1]).unwrap(),
            MxArray::from_float32(&[1.0], &[1]).unwrap(),
            MxArray::from_float32(&[1.0], &[1]).unwrap(),
            None,
        );
        assert!(
            !ql.has_pretransposed_fp8_weight(),
            "an invalid direct-constructor source must remain on the deferred-error fallback"
        );
    }
}

#[cfg(test)]
mod sym8_tests {
    use super::*;
    use crate::array::DType;
    use std::sync::atomic::Ordering;

    fn gpu_gen() -> i32 {
        unsafe { sys::mlx_gpu_architecture_gen() }
    }

    /// Deterministic pseudo-random integer in `[lo, hi]` (LCG — failures
    /// reproduce exactly). Mirrors the helper in qwen3_5's `sym8_tests`.
    fn next_int(state: &mut u64, lo: i32, hi: i32) -> i32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let span = (hi - lo + 1) as u64;
        lo + ((*state >> 33) % span) as i32
    }

    /// Fabricate a synthetic sym8 checkpoint layer: int8 `[N,K]` weight with
    /// integer values in [-127,127] plus positive f32 `[N]` scales, inserted
    /// under `{prefix}.weight` / `{prefix}.scales`.
    fn synth_sym8_params(prefix: &str, n: i64, k: i64, seed: u64) -> HashMap<String, MxArray> {
        let mut state = seed;
        let q: Vec<f32> = (0..n * k)
            .map(|_| next_int(&mut state, -127, 127) as f32)
            .collect();
        let w_i8 = MxArray::from_float32(&q, &[n, k])
            .unwrap()
            .astype(DType::Int8)
            .unwrap();
        let scales: Vec<f32> = (0..n)
            .map(|_| 0.001 + (next_int(&mut state, 1, 1000) as f32) * 1e-5)
            .collect();
        let s_w = MxArray::from_float32(&scales, &[n]).unwrap();
        let mut params = HashMap::new();
        params.insert(format!("{prefix}.weight"), w_i8);
        params.insert(format!("{prefix}.scales"), s_w);
        params
    }

    /// Random bf16 activations `[shape]` in roughly [-2, 2].
    fn synth_x_bf16(shape: &[i64], seed: u64) -> MxArray {
        let mut state = seed;
        let len: i64 = shape.iter().product();
        let v: Vec<f32> = (0..len)
            .map(|_| next_int(&mut state, -2000, 2000) as f32 / 1000.0)
            .collect();
        MxArray::from_float32(&v, shape)
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
    }

    /// bf16 outputs compared bit-for-bit via the native u16 payload
    /// (no f32 round-trip — see project memory).
    fn assert_bf16_bit_identical(a: &MxArray, b: &MxArray, ctx: &str) {
        a.eval();
        b.eval();
        let av = a.to_uint16_native().unwrap();
        let bv = b.to_uint16_native().unwrap();
        assert_eq!(av.len(), bv.len(), "{ctx}: length mismatch");
        let bad = av.iter().zip(bv.iter()).filter(|(x, y)| x != y).count();
        assert_eq!(bad, 0, "{ctx}: {bad}/{} bf16 words differ", av.len());
    }

    /// M=1 routes the QMV kernel, M=512 routes the GEMM kernel, and each
    /// output is bit-for-bit identical to calling the matching `int8_gemm`
    /// reference op directly with the layer's own operands — the same gate
    /// as qwen3_5's `sym8_forward_routes_qmv_at_m1_gemm_at_m512_bit_exact`.
    #[test]
    fn sym8_forward_routes_qmv_at_m1_gemm_at_m512_bit_exact() {
        if gpu_gen() < 17 {
            eprintln!(
                "[sym8] SKIP: gpu gen {} < 17 (int8 kernels need M5+)",
                gpu_gen()
            );
            return;
        }
        let (n, k) = (48i64, 64i64); // K % 16 == 0
        let params = synth_sym8_params("test_layer", n, k, 0x6E44_0001);
        let ql = try_build_sym8_quantized_linear(&params, "test_layer")
            .expect("builder must succeed on a well-formed sym8 layer")
            .expect("scales present => Some");
        assert_eq!(ql.mode(), SYM8_MODE);
        let (w_nk, s_w) = ql.sym8_operands().expect("sym8 operands present");
        assert_eq!(
            w_nk.as_raw_ptr(),
            params.get("test_layer.weight").unwrap().as_raw_ptr(),
            "builder must retain the checkpoint-native [N,K] allocation",
        );

        // --- M=1 → QMV ---
        let x1 = synth_x_bf16(&[1, k], 0xcccc_0001);
        let qmv_before = SYM8_QMV_CALLS.load(Ordering::Relaxed);
        let gemm_before = SYM8_GEMM_CALLS.load(Ordering::Relaxed);
        let y1 = ql.forward(&x1).unwrap();
        assert_eq!(
            SYM8_QMV_CALLS.load(Ordering::Relaxed),
            qmv_before + 1,
            "M=1 must route the QMV kernel"
        );
        assert_eq!(
            SYM8_GEMM_CALLS.load(Ordering::Relaxed),
            gemm_before,
            "M=1 must NOT route the GEMM kernel"
        );
        let y1_ref = int8_gemm::int8_w8a16_qmv_nk(&x1, w_nk, s_w).unwrap();
        assert_bf16_bit_identical(&y1, &y1_ref, "M=1 qmv parity");

        // --- M=2 still QMV (decode-dispatch upper bound), M=3 first GEMM M ---
        let x2 = synth_x_bf16(&[2, k], 0xcccc_0002);
        let qmv_before = SYM8_QMV_CALLS.load(Ordering::Relaxed);
        ql.forward(&x2).unwrap().eval();
        assert_eq!(SYM8_QMV_CALLS.load(Ordering::Relaxed), qmv_before + 1);
        let x3 = synth_x_bf16(&[3, k], 0xcccc_0003);
        let gemm_before = SYM8_GEMM_CALLS.load(Ordering::Relaxed);
        ql.forward(&x3).unwrap().eval();
        assert_eq!(SYM8_GEMM_CALLS.load(Ordering::Relaxed), gemm_before + 1);

        // --- M=512 (prefill, 3-D input [B, S, K]) → GEMM ---
        let x512 = synth_x_bf16(&[4, 128, k], 0xcccc_0512);
        let qmv_before = SYM8_QMV_CALLS.load(Ordering::Relaxed);
        let gemm_before = SYM8_GEMM_CALLS.load(Ordering::Relaxed);
        let y512 = ql.forward(&x512).unwrap();
        assert_eq!(
            SYM8_GEMM_CALLS.load(Ordering::Relaxed),
            gemm_before + 1,
            "M=512 must route the GEMM kernel"
        );
        assert_eq!(
            SYM8_QMV_CALLS.load(Ordering::Relaxed),
            qmv_before,
            "M=512 must NOT route the QMV kernel"
        );
        assert_eq!(y512.shape().unwrap().to_vec(), vec![4, 128, n]);
        let x512_2d = x512.reshape(&[512, k]).unwrap();
        let y512_ref = int8_gemm::int8_w8a8_matmul_nk(&x512_2d, w_nk, s_w)
            .unwrap()
            .reshape(&[4, 128, n])
            .unwrap();
        assert_bf16_bit_identical(&y512, &y512_ref, "M=512 gemm parity");
    }

    /// Additive linear bias is applied after the int8 kernel.
    #[test]
    fn sym8_forward_applies_linear_bias() {
        if gpu_gen() < 17 {
            eprintln!("[sym8] SKIP: gpu gen {} < 17", gpu_gen());
            return;
        }
        let (n, k) = (32i64, 64i64);
        let params = synth_sym8_params("biased", n, k, 0x6E44_0002);
        let weight = params.get("biased.weight").unwrap().clone();
        let scales = params.get("biased.scales").unwrap().clone();
        let bias = synth_x_bf16(&[n], 0xdddd_0001);
        let ql = QuantizedLinear::new_sym8(weight.clone(), scales.clone(), Some(bias.clone()));
        let x = synth_x_bf16(&[1, k], 0xdddd_0002);
        let y = ql.forward(&x).unwrap();
        let y_ref = int8_gemm::int8_w8a16_qmv_nk(&x, &weight, &scales)
            .unwrap()
            .add(&bias)
            .unwrap();
        assert_bf16_bit_identical(&y, &y_ref, "bias add parity");
    }

    /// Load-time fail-loud contract: every malformed sym8 layer is an `Err`
    /// (never a silent `None` fallback), while a genuinely-absent sidecar is
    /// `Ok(None)`.
    #[test]
    fn sym8_builder_fail_loud_contract() {
        if gpu_gen() < 17 {
            eprintln!(
                "[sym8] SKIP: gpu gen {} < 17 (builder gen-gate untestable)",
                gpu_gen()
            );
            return;
        }
        let (n, k) = (16i64, 32i64);

        // Missing .scales → Ok(None) (bf16-fallback layer in a sym8 checkpoint).
        let mut p = synth_sym8_params("l", n, k, 1);
        p.remove("l.scales");
        assert!(matches!(try_build_sym8_quantized_linear(&p, "l"), Ok(None)));

        // .scales present but .weight missing → Err.
        let mut p = synth_sym8_params("l", n, k, 2);
        p.remove("l.weight");
        assert!(try_build_sym8_quantized_linear(&p, "l").is_err());

        // Unexpected .biases sidecar → Err.
        let mut p = synth_sym8_params("l", n, k, 3);
        let zeros = vec![0.0f32; n as usize];
        p.insert(
            "l.biases".into(),
            MxArray::from_float32(&zeros, &[n]).unwrap(),
        );
        assert!(try_build_sym8_quantized_linear(&p, "l").is_err());

        // Non-int8 weight dtype → Err.
        let mut p = synth_sym8_params("l", n, k, 4);
        let w_f = p.get("l.weight").unwrap().astype(DType::Float32).unwrap();
        p.insert("l.weight".into(), w_f);
        assert!(try_build_sym8_quantized_linear(&p, "l").is_err());

        // K % 16 != 0 → Err.
        let p = synth_sym8_params("l", n, 24, 5);
        assert!(try_build_sym8_quantized_linear(&p, "l").is_err());

        // Non-f32 scales dtype → Err.
        let mut p = synth_sym8_params("l", n, k, 6);
        let s_b = p.get("l.scales").unwrap().astype(DType::BFloat16).unwrap();
        p.insert("l.scales".into(), s_b);
        assert!(try_build_sym8_quantized_linear(&p, "l").is_err());

        // Scales length != N → Err.
        let mut p = synth_sym8_params("l", n, k, 7);
        let short_scales = vec![0.001f32; (n - 1) as usize];
        p.insert(
            "l.scales".into(),
            MxArray::from_float32(&short_scales, &[n - 1]).unwrap(),
        );
        assert!(try_build_sym8_quantized_linear(&p, "l").is_err());
    }
}

#[cfg(test)]
mod quantized_mlp_tests {
    use super::*;

    fn affine_q4(seed: u32) -> QuantizedLinear {
        let packed: Vec<u32> = (0..32 * 4)
            .map(|i| 0x7654_3210u32.rotate_left(((i as u32 + seed) % 8) * 4))
            .collect();
        let scales = vec![half::f16::from_f32(0.0625).to_bits(); 32];
        let biases = vec![half::f16::from_f32(-0.5).to_bits(); 32];
        QuantizedLinear::new(
            MxArray::from_uint32(&packed, &[32, 4]).unwrap(),
            MxArray::from_float16(&scales, &[32, 1]).unwrap(),
            Some(MxArray::from_float16(&biases, &[32, 1]).unwrap()),
            None,
            32,
            4,
            DEFAULT_QUANT_MODE.to_string(),
        )
    }

    fn values(array: &MxArray) -> Vec<f32> {
        array.eval();
        array.to_float32().unwrap().as_ref().to_vec()
    }

    #[test]
    fn separate_q4_projections_preserve_bfloat16() {
        let input_bits: Vec<u16> = (0..64)
            .map(|i| half::bf16::from_f32(((i % 13) as f32 - 6.0) / 8.0).to_bits())
            .collect();
        let input = MxArray::from_bfloat16(&input_bits, &[1, 2, 32]).unwrap();

        let mlp = Gemma4MLPVariant::Quantized {
            gate_proj: affine_q4(2),
            up_proj: affine_q4(7),
            down_proj: affine_q4(11),
        };
        let actual = mlp.forward(&input).unwrap();

        let gate = affine_q4(2).forward(&input).unwrap();
        let up = affine_q4(7).forward(&input).unwrap();
        let gated = Activations::gelu(&gate).unwrap().mul(&up).unwrap();
        let expected = affine_q4(11).forward(&gated).unwrap();

        assert_eq!(actual.dtype().unwrap(), DType::BFloat16);
        assert_eq!(actual.shape().unwrap().to_vec(), vec![1, 2, 32]);
        assert_eq!(values(&actual), values(&expected));
    }
}

#[cfg(test)]
mod affine_dtype_tests {
    use super::*;

    #[test]
    fn affine_q4_forward_preserves_bfloat16_activation_dtype() {
        let input =
            MxArray::from_bfloat16(&[half::bf16::from_f32(0.5).to_bits(); 32], &[1, 32]).unwrap();
        let weight = MxArray::from_uint32(&[0x8888_8888; 4], &[1, 4]).unwrap();
        let scales =
            MxArray::from_float16(&[half::f16::from_f32(0.25).to_bits()], &[1, 1]).unwrap();
        let biases =
            MxArray::from_float16(&[half::f16::from_f32(-2.0).to_bits()], &[1, 1]).unwrap();
        let linear = QuantizedLinear::new(
            weight,
            scales,
            Some(biases),
            None,
            32,
            4,
            DEFAULT_QUANT_MODE.to_string(),
        );

        let output = linear.forward(&input).unwrap();
        assert_eq!(output.dtype().unwrap(), crate::array::DType::BFloat16);
    }

    #[test]
    fn affine_q4_expert_forward_restores_bfloat16_activation_dtype() {
        let input =
            MxArray::from_bfloat16(&[half::bf16::from_f32(0.5).to_bits(); 32], &[1, 1, 1, 32])
                .unwrap();
        let indices = MxArray::from_int32(&[0], &[1, 1]).unwrap();
        let weight = MxArray::from_uint32(&[0x7654_3210; 4], &[1, 1, 4]).unwrap();
        let scales =
            MxArray::from_float16(&[half::f16::from_f32(0.03125).to_bits()], &[1, 1, 1]).unwrap();
        let biases =
            MxArray::from_float16(&[half::f16::from_f32(-0.25).to_bits()], &[1, 1, 1]).unwrap();

        let linear = QuantizedSwitchLinear::new(
            weight,
            scales,
            Some(biases),
            32,
            4,
            DEFAULT_QUANT_MODE.to_string(),
        );
        let actual = linear.forward(&input, &indices, false).unwrap();
        assert_eq!(actual.dtype().unwrap(), DType::BFloat16);
    }
}
