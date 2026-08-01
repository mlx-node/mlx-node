use std::collections::HashMap;
use std::ffi::CString;

use crate::array::MxArray;
use crate::nn::{Activations, Linear};
use crate::transformer::MLP;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::int8_gemm;

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

/// MXFP4 quantization parameters (E2M1 format, fixed bits/group_size).
pub const MXFP4_BITS: i32 = 4;
pub const MXFP4_GROUP_SIZE: i32 = 32;
pub const MXFP4_MODE: &str = "mxfp4";

/// NVFP4 quantization parameters (E2M1 4-bit weights with E4M3 uint8 scales,
/// group_size 16).
pub const NVFP4_BITS: i32 = 4;
pub const NVFP4_GROUP_SIZE: i32 = 16;
pub const NVFP4_MODE: &str = "nvfp4";

pub use crate::quant::fp8_weight::{FP8_E4M3_BITS, FP8_E4M3_GROUP_SIZE, FP8_E4M3_MODE};

// `PerLayerMode` and `PerLayerQuant` are family-neutral types shared with
// `qwen3_5_moe` and `gemma4`; they live in `crate::models::quant_dispatch`
// so the three families don't cross-import from each other.
pub use crate::models::quant_dispatch::{PerLayerMode, PerLayerQuant};

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

    pub fn get_weight(&self) -> MxArray {
        match self {
            LinearProj::Standard(l) => l.get_weight(),
            LinearProj::Quantized(ql) => ql.get_weight().clone(),
        }
    }

    /// Whether this projection holds a quantized backend.
    ///
    /// Used by the dense/bf16-only `save_model_sync` MTP path to refuse
    /// emitting a quantized projection's stale dense `weight` as if it were
    /// a valid bf16 tensor (see `Qwen3_5MTPModule::has_quantized_weights`).
    pub fn is_quantized(&self) -> bool {
        matches!(self, LinearProj::Quantized(_))
    }

    pub(crate) fn has_q_gate_block_layout(&self) -> bool {
        matches!(self, LinearProj::Quantized(ql) if ql.has_q_gate_block_layout())
    }

    /// The per-tensor FP8 activation scale threaded onto the quantized backend
    /// at load time (`None` for a dense projection). Test-only read-back seam
    /// used to prove the loaders thread `PerLayerQuant::input_amax` onto the
    /// built `QuantizedLinear`.
    #[cfg(test)]
    pub(crate) fn input_amax(&self) -> Option<f32> {
        match self {
            LinearProj::Standard(_) => None,
            LinearProj::Quantized(ql) => ql.input_amax(),
        }
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

    pub fn get_gate_proj_weight(&self) -> MxArray {
        match self {
            MLPVariant::Standard(mlp) => mlp.get_gate_proj_weight(),
            MLPVariant::Quantized { gate_proj, .. } => gate_proj.get_weight().clone(),
        }
    }

    pub fn get_up_proj_weight(&self) -> MxArray {
        match self {
            MLPVariant::Standard(mlp) => mlp.get_up_proj_weight(),
            MLPVariant::Quantized { up_proj, .. } => up_proj.get_weight().clone(),
        }
    }

    pub fn get_down_proj_weight(&self) -> MxArray {
        match self {
            MLPVariant::Standard(mlp) => mlp.get_down_proj_weight(),
            MLPVariant::Quantized { down_proj, .. } => down_proj.get_weight().clone(),
        }
    }

    /// Whether this MLP holds a quantized backend.
    ///
    /// Used by the dense/bf16-only `save_model_sync` MTP path to refuse
    /// emitting a quantized MLP's stale dense weights (see
    /// `Qwen3_5MTPModule::has_quantized_weights`).
    pub fn is_quantized(&self) -> bool {
        matches!(self, MLPVariant::Quantized { .. })
    }

    pub fn set_gate_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        match self {
            MLPVariant::Standard(mlp) => mlp.set_gate_proj_weight(w),
            MLPVariant::Quantized { .. } => {
                Err(Error::from_reason("Cannot set weight on quantized MLP"))
            }
        }
    }

    pub fn set_up_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        match self {
            MLPVariant::Standard(mlp) => mlp.set_up_proj_weight(w),
            MLPVariant::Quantized { .. } => {
                Err(Error::from_reason("Cannot set weight on quantized MLP"))
            }
        }
    }

    /// E39: finalize stacked gate+up weight. No-op for quantized variant.
    pub fn finalize_gate_up(&mut self) -> Result<()> {
        match self {
            MLPVariant::Standard(mlp) => mlp.finalize_gate_up(),
            MLPVariant::Quantized { .. } => Ok(()),
        }
    }

    pub fn set_down_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        match self {
            MLPVariant::Standard(mlp) => mlp.set_down_proj_weight(w),
            MLPVariant::Quantized { .. } => {
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

/// Build the plain per-output E4M3 correctness fallback.
///
/// Checkpoint storage is strict: Uint8 `[N,K]` E4M3 bytes + floating `[N,1]`
/// dequant scales, with no affine `.biases`. The weight is reconstructed to
/// BF16 once here and `QuantizedLinear::forward` uses an ordinary A16 matmul.
/// This preserves the DGX artifact's weight format without claiming native
/// W8A8 execution.
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

/// Try to build a ggml K-quant `QuantizedLinear` from `{prefix}.weight` (uint32
/// packed), `{prefix}.scales` (int8 for Q6_K / uint8 for Q4_K/Q5_K), and the
/// MANDATORY `{prefix}.biases` (float16 ggml `d` super-block scale).
///
/// Fail-loud template shared with sym8: `Ok(None)` ONLY when `.scales` is
/// absent (an unquantized bf16 tensor in a mixed-precision UD GGUF); every
/// malformed/partial group is `Err`. Validation is delegated to
/// [`resolve_kquant_group`](crate::models::quant_dispatch::resolve_kquant_group)
/// so the dense, expert, and gemma4 K-quant builders cannot drift. `forward`
/// threads the resolved mode string into `mlx_quantized_matmul`, which does the
/// two-level scale decode from `scales`/`biases`.
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
/// Affine/MX/NVFP modes use packed Uint32 weights and MLX quantized_matmul;
/// sym8 uses dedicated int8 kernels. Plain `fp8_e4m3` is the intentionally
/// non-native exception: it retains raw Uint8 checkpoint storage, reconstructs
/// BF16 once at load, and uses ordinary A16 matmul.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum QuantizedOutputLayout {
    #[default]
    Native,
    QGateBlock,
}

fn q_gate_block_permutation(num_heads: i32, head_dim: i32, output_rows: i64) -> Result<Vec<i32>> {
    if num_heads <= 0 || head_dim <= 0 {
        return Err(Error::from_reason(format!(
            "q/gate block layout requires positive num_heads/head_dim, got {num_heads}/{head_dim}"
        )));
    }
    let h = i64::from(num_heads);
    let d = i64::from(head_dim);
    let expected_rows = h
        .checked_mul(d)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| Error::from_reason("q/gate block row count overflow"))?;
    if output_rows != expected_rows {
        return Err(Error::from_reason(format!(
            "q/gate block layout expected {expected_rows} output rows for H={num_heads}, D={head_dim}, got {output_rows}"
        )));
    }
    if expected_rows > i64::from(i32::MAX) {
        return Err(Error::from_reason(format!(
            "q/gate block layout has too many output rows for int32 indices: {expected_rows}"
        )));
    }

    let mut permutation = Vec::with_capacity(expected_rows as usize);
    for part in 0..2i64 {
        for head in 0..h {
            for dim in 0..d {
                permutation.push((head * 2 * d + part * d + dim) as i32);
            }
        }
    }
    Ok(permutation)
}

pub struct QuantizedLinear {
    weight: MxArray,         // Packed uint32 quantized weights [out, in_packed]
    scales: MxArray,         // Quantization scales
    biases: Option<MxArray>, // Quantization biases (for affine mode)
    bias: Option<MxArray>,   // Linear bias (additive)
    group_size: i32,
    bits: i32,
    mode: String, // "affine", "mxfp8", "mxfp4", "nvfp4", "fp8_e4m3", or "sym8"
    // Reconstructed BF16 `[N,K]` weight for the plain E4M3 correctness
    // fallback. `Some` iff mode == fp8_e4m3; activations remain A16.
    fp8_dequant_weight: Option<MxArray>,
    // sym8 scale: `Some` iff mode == "sym8". Decode/prefill consume the
    // checkpoint-native `self.weight` [N,K] directly.
    s_w: Option<MxArray>,
    // Per-tensor static FP8 (E4M3) activation scale (modelopt MaxCalibrator
    // amax), threaded from the layer's `config.json` quantization override via
    // `PerLayerQuant::input_amax`. `Some` only on calibrated mxfp8 attention/GDN
    // projections; `None` everywhere else. Carried here for a later task to
    // fake-quant activations to E4M3 for W8A8 numeric parity — forward does NOT
    // yet read it, so behaviour is unchanged while `None`.
    input_amax: Option<f32>,
    // The projection's normalized per-layer config key
    // (`normalize_per_layer_key(prefix)`), threaded at load so the
    // activation-amax calibration tap can bucket recorded `max|activation|` by
    // projection. `Some` on every projection built by the two `try_build_ql`
    // loaders (dense + MoE); `None` on test-fabricated / non-loader instances.
    // Read ONLY by the calibration tap in `forward` (gated by
    // `mode == MXFP8_MODE` + collector-enabled), so it never affects normal
    // inference.
    amax_key: Option<String>,
    output_layout: QuantizedOutputLayout,
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
            input_amax: None,
            amax_key: None,
            output_layout: QuantizedOutputLayout::Native,
        }
    }

    /// Construct a plain E4M3 storage-backed linear with a load-time BF16
    /// reconstruction. The raw Uint8 tensor remains available through
    /// `get_weight()` so storage identity is never confused with MXFP8.
    pub fn new_fp8_e4m3(
        weight: MxArray,
        scales: MxArray,
        dequant_weight: MxArray,
        bias: Option<MxArray>,
    ) -> Self {
        Self {
            weight,
            scales,
            biases: None,
            bias,
            group_size: FP8_E4M3_GROUP_SIZE,
            bits: FP8_E4M3_BITS,
            mode: FP8_E4M3_MODE.to_string(),
            fp8_dequant_weight: Some(dequant_weight),
            s_w: None,
            input_amax: None,
            amax_key: None,
            output_layout: QuantizedOutputLayout::Native,
        }
    }

    /// Attach a per-tensor FP8 activation scale (`PerLayerQuant::input_amax`).
    ///
    /// Consuming builder used at the load-time dispatch site to thread the
    /// calibrated amax onto a freshly built projection. `None` is the no-op /
    /// default (bf16 activations, current behaviour). A later task reads this
    /// field in `forward` to fake-quant activations to E4M3.
    pub fn with_input_amax(mut self, input_amax: Option<f32>) -> Self {
        self.input_amax = input_amax;
        self
    }

    /// The calibrated per-tensor FP8 activation scale, if any.
    pub fn input_amax(&self) -> Option<f32> {
        self.input_amax
    }

    /// Attach the projection's normalized config key for the calibration tap.
    ///
    /// Consuming builder used at the load-time dispatch site (next to
    /// [`with_input_amax`](Self::with_input_amax)) so the activation-amax
    /// collector can bucket recorded `max|activation|` by projection. `None` is
    /// the default (no calibration bucket — test-fabricated instances).
    pub fn with_amax_key(mut self, amax_key: Option<String>) -> Self {
        self.amax_key = amax_key;
        self
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
            input_amax: None,
            amax_key: None,
            output_layout: QuantizedOutputLayout::Native,
        }
    }

    /// Reorder an affine q/gate projection's output rows from checkpoint order
    /// `[Q_h0,G_h0,Q_h1,G_h1,...]` to `[Q_all_heads,G_all_heads]`.
    ///
    /// All row-coupled operands are materialized together before replacing the
    /// stored arrays. Once this returns `true`, the original-order arrays are
    /// no longer retained by the projection. Non-affine or incompatible direct
    /// constructor inputs stay untouched and return `false` so attention keeps
    /// using its native per-head split.
    pub(crate) fn finalize_affine_q_gate_block(
        &mut self,
        num_heads: i32,
        head_dim: i32,
    ) -> Result<bool> {
        if self.output_layout == QuantizedOutputLayout::QGateBlock {
            return Ok(true);
        }
        if self.mode != DEFAULT_QUANT_MODE {
            return Ok(false);
        }

        let weight_shape = self.weight.shape()?.to_vec();
        let scales_shape = self.scales.shape()?.to_vec();
        let Some(quant_biases) = self.biases.as_ref() else {
            return Ok(false);
        };
        let quant_biases_shape = quant_biases.shape()?.to_vec();
        let expected_rows = i64::from(num_heads)
            .checked_mul(i64::from(head_dim))
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| Error::from_reason("q/gate block row count overflow"))?;
        let additive_bias_compatible = match self.bias.as_ref() {
            Some(bias) => bias.shape()?.to_vec() == [expected_rows],
            None => true,
        };
        if weight_shape.len() != 2
            || scales_shape.len() != 2
            || quant_biases_shape.len() != 2
            || weight_shape[0] != expected_rows
            || scales_shape[0] != expected_rows
            || quant_biases_shape != scales_shape
            || !additive_bias_compatible
        {
            return Ok(false);
        }

        let permutation = q_gate_block_permutation(num_heads, head_dim, expected_rows)?;
        let indices = MxArray::from_int32(&permutation, &[expected_rows])?;
        let weight = self.weight.take(&indices, 0)?;
        let scales = self.scales.take(&indices, 0)?;
        let quant_biases = quant_biases.take(&indices, 0)?;
        let bias = self
            .bias
            .as_ref()
            .map(|bias| bias.take(&indices, 0))
            .transpose()?;

        let mut arrays = vec![&weight, &scales, &quant_biases];
        if let Some(bias) = bias.as_ref() {
            arrays.push(bias);
        }
        MxArray::eval_arrays(&arrays)?;

        self.weight = weight;
        self.scales = scales;
        self.biases = Some(quant_biases);
        self.bias = bias;
        self.output_layout = QuantizedOutputLayout::QGateBlock;
        Ok(true)
    }

    pub(crate) fn has_q_gate_block_layout(&self) -> bool {
        self.output_layout == QuantizedOutputLayout::QGateBlock
    }

    /// sym8 forward: int8-weight GEMM/QMV + rescale.
    ///
    /// Dispatch rule: `M <= 2` → W8A16 decode (the dedicated
    /// W8A16 decode matvec — bf16 activations read directly, NO act quant,
    /// activation-exact; the W8A8 act-quant passes were pure in-stream
    /// overhead at decode M and the prefill tile wastes 127/128 rows at M=1),
    /// `M >= 3` → W8A8 prefill. Routing consumes only `self.weight` `[N,K]`.
    /// Fail-loud on kernel error (there is no affine pack fallback).
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

    /// Forward pass using quantized_matmul (sym8 routes to the int8 W8A8
    /// kernels instead — `mlx_quantized_matmul` has no sym8 pack).
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        // Whether THIS thread is the calibrating model thread (thread-local, so
        // a concurrently-running inference model on another thread never trips
        // this). Read once and reused for both the tap and the fake-quant
        // suppression below.
        let calibrating =
            crate::calibration::activation_amax::ActivationAmaxCollector::is_calibrating();

        // Activation-amax calibration tap (modelopt MaxCalibrator): while this
        // thread is calibrating, record this projection's raw bf16 activation
        // `max|x|` BEFORE any fake-quant. Gated to mxfp8 sites (the recipe's
        // activation-fp8 attn/GDN projections) via the same mode check the
        // fake-quant gate below uses. The recorded value is the clean bf16 input
        // — the fake-quant is suppressed while calibrating (below), so an
        // already-calibrated model re-calibrates at raw-bf16 MaxCalibrator
        // parity. When not calibrating this is a single thread-local load then
        // skip, so forward is behaviorally unchanged for normal inference.
        if calibrating
            && self.mode == MXFP8_MODE
            && let Some(key) = &self.amax_key
        {
            crate::calibration::activation_amax::ActivationAmaxCollector::record(key, x)?;
        }

        // Calibrated per-tensor FP8 (E4M3) activation fake-quant, matching
        // NVIDIA modelopt's static W8A8 attention/GDN math. SUPPRESSED while
        // this thread is calibrating: the calibration pass must measure the raw
        // bf16 activation regardless of any existing `input_amax` (an upstream
        // fake-quant would perturb downstream activations, breaking raw-bf16
        // MaxCalibrator parity on a re-calibration).
        //
        // Otherwise the gate requires BOTH a positive `input_amax` AND
        // `self.mode == MXFP8_MODE`: in the nvidia recipe the mxfp8 projections
        // are EXACTLY the attn/GDN activation-fp8 sites, so the mode check
        // enforces the invariant at the point of use rather than trusting the
        // loaders. Even if a stale, malformed, or hand-edited config erroneously
        // threads `input_amax` onto a non-mxfp8 projection (mxfp4 FFN, affine
        // gates/in_proj_ba, lm_head, sym8), `x` stays the original bf16 reference
        // there and that forward is byte-identical to before. Apple GPUs have no
        // fp8 matmul hardware — this is numeric parity, not speed.
        let xq_owned;
        let x = if calibrating {
            x
        } else {
            match self.input_amax {
                Some(amax) if amax > 0.0 && self.mode == MXFP8_MODE => {
                    xq_owned = crate::quant::fp8_activation::fp8_fake_quant(x, amax)?;
                    &xq_owned
                }
                _ => x,
            }
        };

        if self.mode == SYM8_MODE {
            return self.forward_sym8(x);
        }

        if self.mode == FP8_E4M3_MODE {
            let weight = self.fp8_dequant_weight.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "plain FP8 QuantizedLinear missing load-time BF16 reconstruction",
                )
            })?;
            let mut result = x.matmul(&weight.transpose(Some(&[1, 0]))?)?;
            if let Some(ref b) = self.bias {
                result = result.add(b)?;
            }
            return Ok(result);
        }

        // Affine QMM promotes mixed activation/sidecar dtypes (for example,
        // BF16 activations with GGUF FP16 scales/biases) to FP32. Keep that
        // arithmetic, including the additive linear bias below, then restore
        // the model activation dtype at the projection boundary. Other modes
        // already define their output dtype from `x` and stay untouched.
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

        if let Some(dtype) = activation_dtype
            && result.dtype()? != dtype
        {
            result = result.astype(dtype)?;
        }

        Ok(result)
    }

    pub fn set_weight(&mut self, weight: MxArray) {
        self.weight = weight;
        if self.mode == FP8_E4M3_MODE {
            // This infallible legacy mutator cannot validate/rebuild the plain
            // FP8 pair. Invalidate the reconstruction so forward fails loud
            // instead of using a stale BF16 weight.
            self.fp8_dequant_weight = None;
        }
    }

    pub fn set_scales(&mut self, scales: MxArray) {
        self.scales = scales;
        if self.mode == FP8_E4M3_MODE {
            self.fp8_dequant_weight = None;
        }
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

    /// Model-owned BF16 reconstruction retained by the plain-E4M3 fallback.
    /// The raw Uint8 weight and scales stay exposed through the ordinary
    /// checkpoint accessors and are accounted separately by the loader.
    pub(crate) fn reconstructed_fp8_weight(&self) -> Option<&MxArray> {
        self.fp8_dequant_weight.as_ref()
    }

    pub fn get_biases(&self) -> Option<&MxArray> {
        self.biases.as_ref()
    }

    /// Quantization mode discriminator string ("affine", "mxfp8", "mxfp4",
    /// "nvfp4", "fp8_e4m3", or "sym8").
    pub fn mode(&self) -> &str {
        &self.mode
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
    use crate::array::DType;

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
        assert_eq!(ql.mode(), FP8_E4M3_MODE);
        assert_eq!(ql.get_weight().dtype().unwrap(), DType::Uint8);
        assert_eq!(ql.get_scales().shape().unwrap().to_vec(), vec![3, 1]);
        assert_eq!(
            ql.reconstructed_fp8_weight().unwrap().nbytes(),
            3 * 4 * std::mem::size_of::<u16>(),
            "loader-visible residency must expose the retained BF16 reconstruction"
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
}

#[cfg(test)]
mod affine_dtype_tests {
    use super::*;
    use crate::array::DType;

    #[test]
    fn affine_q4_forward_restores_bfloat16_after_additive_bias() {
        let input =
            MxArray::from_bfloat16(&[half::bf16::from_f32(0.5).to_bits(); 32], &[1, 32]).unwrap();
        let weight = MxArray::from_uint32(&[0x7654_3210; 4], &[1, 4]).unwrap();
        let scales =
            MxArray::from_float16(&[half::f16::from_f32(0.03125).to_bits()], &[1, 1]).unwrap();
        let biases =
            MxArray::from_float16(&[half::f16::from_f32(-0.25).to_bits()], &[1, 1]).unwrap();
        let additive_bias = MxArray::from_float32(&[0.00390625], &[1]).unwrap();

        let raw_handle = unsafe {
            sys::mlx_quantized_matmul(
                input.as_raw_ptr(),
                weight.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases.as_raw_ptr(),
                true,
                32,
                4,
                c"affine".as_ptr(),
            )
        };
        let raw = MxArray::from_handle(raw_handle, "test_raw_affine_qmm").unwrap();
        assert_eq!(raw.dtype().unwrap(), DType::Float32);
        let expected = raw
            .add(&additive_bias)
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();

        let linear = QuantizedLinear::new(
            weight,
            scales,
            Some(biases),
            Some(additive_bias),
            32,
            4,
            DEFAULT_QUANT_MODE.to_string(),
        );
        let actual = linear.forward(&input).unwrap();
        assert_eq!(actual.dtype().unwrap(), DType::BFloat16);
        actual.eval();
        expected.eval();
        assert_eq!(
            actual.to_uint16_native().unwrap(),
            expected.to_uint16_native().unwrap(),
            "the additive bias must be applied before the projection-boundary cast"
        );
    }
}

#[cfg(test)]
mod q_gate_block_tests {
    use super::*;

    #[test]
    fn q_gate_block_permutation_is_bijective_for_multiple_heads() {
        let permutation = q_gate_block_permutation(3, 2, 12).unwrap();
        assert_eq!(permutation, vec![0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]);
        let mut sorted = permutation.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..12).collect::<Vec<_>>());

        assert!(q_gate_block_permutation(0, 2, 0).is_err());
        assert!(q_gate_block_permutation(3, 0, 0).is_err());
        assert!(q_gate_block_permutation(3, 2, 11).is_err());
    }

    #[test]
    fn affine_q_gate_block_replaces_every_row_coupled_operand() {
        let h = 3;
        let d = 2;
        let rows = 12i64;
        let permutation = q_gate_block_permutation(h, d, rows).unwrap();
        let weight_values = (0..rows)
            .flat_map(|row| [row as u32 * 10, row as u32 * 10 + 1])
            .collect::<Vec<_>>();
        let scales_values = (0..rows).map(|row| row as f32 + 0.25).collect::<Vec<_>>();
        let quant_bias_values = (0..rows).map(|row| row as f32 + 100.0).collect::<Vec<_>>();
        let bias_values = (0..rows).map(|row| row as f32 + 200.0).collect::<Vec<_>>();

        let weight = MxArray::from_uint32(&weight_values, &[rows, 2]).unwrap();
        let scales = MxArray::from_float32(&scales_values, &[rows, 1]).unwrap();
        let quant_biases = MxArray::from_float32(&quant_bias_values, &[rows, 1]).unwrap();
        let bias = MxArray::from_float32(&bias_values, &[rows]).unwrap();
        let original_ptrs = [
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            quant_biases.as_raw_ptr(),
            bias.as_raw_ptr(),
        ];
        let mut linear = QuantizedLinear::new(
            weight,
            scales,
            Some(quant_biases),
            Some(bias),
            32,
            4,
            DEFAULT_QUANT_MODE.to_string(),
        );

        assert!(linear.finalize_affine_q_gate_block(h, d).unwrap());
        assert!(linear.has_q_gate_block_layout());
        assert_ne!(linear.weight.as_raw_ptr(), original_ptrs[0]);
        assert_ne!(linear.scales.as_raw_ptr(), original_ptrs[1]);
        assert_ne!(
            linear.biases.as_ref().unwrap().as_raw_ptr(),
            original_ptrs[2]
        );
        assert_ne!(linear.bias.as_ref().unwrap().as_raw_ptr(), original_ptrs[3]);

        let expected_weight = permutation
            .iter()
            .flat_map(|&row| [row as u32 * 10, row as u32 * 10 + 1])
            .collect::<Vec<_>>();
        let expected_scales = permutation
            .iter()
            .map(|&row| row as f32 + 0.25)
            .collect::<Vec<_>>();
        let expected_quant_biases = permutation
            .iter()
            .map(|&row| row as f32 + 100.0)
            .collect::<Vec<_>>();
        let expected_bias = permutation
            .iter()
            .map(|&row| row as f32 + 200.0)
            .collect::<Vec<_>>();
        assert_eq!(linear.weight.to_uint32().unwrap().to_vec(), expected_weight);
        assert_eq!(
            linear.scales.to_float32().unwrap().to_vec(),
            expected_scales
        );
        assert_eq!(
            linear
                .biases
                .as_ref()
                .unwrap()
                .to_float32()
                .unwrap()
                .to_vec(),
            expected_quant_biases
        );
        assert_eq!(
            linear.bias.as_ref().unwrap().to_float32().unwrap().to_vec(),
            expected_bias
        );

        assert!(
            linear.finalize_affine_q_gate_block(h, d).unwrap(),
            "the finalized layout must be idempotent"
        );
    }

    #[test]
    fn q_gate_block_is_noop_for_non_affine_or_incompatible_operands() {
        let weight = MxArray::from_uint32(&[0; 24], &[12, 2]).unwrap();
        let scales = MxArray::from_float32(&[1.0; 12], &[12, 1]).unwrap();
        let biases = MxArray::from_float32(&[0.0; 12], &[12, 1]).unwrap();

        let mut non_affine = QuantizedLinear::new(
            weight.clone(),
            scales.clone(),
            Some(biases.clone()),
            None,
            32,
            4,
            MXFP4_MODE.to_string(),
        );
        let original = non_affine.weight.as_raw_ptr();
        assert!(!non_affine.finalize_affine_q_gate_block(3, 2).unwrap());
        assert!(!non_affine.has_q_gate_block_layout());
        assert_eq!(non_affine.weight.as_raw_ptr(), original);

        let mut incompatible = QuantizedLinear::new(
            weight,
            MxArray::from_float32(&[1.0; 11], &[11, 1]).unwrap(),
            Some(biases),
            None,
            32,
            4,
            DEFAULT_QUANT_MODE.to_string(),
        );
        let original = incompatible.weight.as_raw_ptr();
        assert!(!incompatible.finalize_affine_q_gate_block(3, 2).unwrap());
        assert!(!incompatible.has_q_gate_block_layout());
        assert_eq!(incompatible.weight.as_raw_ptr(), original);
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
    /// reproduce exactly). Mirrors the helper in `int8_gemm::tests`.
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

    /// GATE (a): M=1 routes the QMV kernel, M=512 routes the GEMM kernel, and
    /// each output is bit-for-bit identical to calling the matching
    /// `int8_gemm` reference op directly with the layer's own operands.
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
        let params = synth_sym8_params("test_layer", n, k, 0x5197_0001);
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
        let x1 = synth_x_bf16(&[1, k], 0xaaaa_0001);
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

        // --- M=2 still QMV (decode-dispatch upper bound) ---
        let x2 = synth_x_bf16(&[2, k], 0xaaaa_0002);
        let qmv_before = SYM8_QMV_CALLS.load(Ordering::Relaxed);
        ql.forward(&x2).unwrap().eval();
        assert_eq!(SYM8_QMV_CALLS.load(Ordering::Relaxed), qmv_before + 1);

        // --- M=3 first GEMM M ---
        let x3 = synth_x_bf16(&[3, k], 0xaaaa_0003);
        let gemm_before = SYM8_GEMM_CALLS.load(Ordering::Relaxed);
        ql.forward(&x3).unwrap().eval();
        assert_eq!(SYM8_GEMM_CALLS.load(Ordering::Relaxed), gemm_before + 1);

        // --- M=512 (prefill, 3-D input [B, S, K]) → GEMM ---
        let x512 = synth_x_bf16(&[4, 128, k], 0xaaaa_0512);
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
        let params = synth_sym8_params("biased", n, k, 0x5197_0002);
        let weight = params.get("biased.weight").unwrap().clone();
        let scales = params.get("biased.scales").unwrap().clone();
        let bias = synth_x_bf16(&[n], 0xbbbb_0001);
        let ql = QuantizedLinear::new_sym8(weight.clone(), scales.clone(), Some(bias.clone()));
        let x = synth_x_bf16(&[1, k], 0xbbbb_0002);
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
mod fp8_activation_tests {
    use super::*;
    use crate::array::DType;
    use crate::quant::fp8_activation::fp8_fake_quant;

    /// Deterministic LCG float in `[-2, 2]` (failures reproduce exactly).
    fn next_f32(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((*state >> 40) & 0xFFFF) as f32 / 65535.0; // [0, 1]
        u * 4.0 - 2.0
    }

    /// MXFP8-quantize a 2D bf16 weight, returning `(packed_weight, uint8 scales)`
    /// (mxfp8 has no biases) — mirrors the embedding-test helper.
    fn quantize_mxfp8(weight: &MxArray) -> (MxArray, MxArray) {
        let mut out_q: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_s: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_b: *mut sys::mlx_array = std::ptr::null_mut();
        let ok = unsafe {
            sys::mlx_quantize(
                weight.as_raw_ptr(),
                MXFP8_GROUP_SIZE,
                MXFP8_BITS,
                c"mxfp8".as_ptr(),
                &mut out_q,
                &mut out_s,
                &mut out_b,
            )
        };
        assert!(ok, "mlx_quantize mxfp8 failed");
        (
            MxArray::from_handle(out_q, "q").expect("q"),
            MxArray::from_handle(out_s, "s").expect("s"),
        )
    }

    /// A fresh mxfp8 `QuantizedLinear` over shared (cloned) packed operands, so
    /// the same weights back every instance in one test (`with_input_amax`
    /// consumes `self`, so each variant needs its own struct).
    fn make_mxfp8_linear(w_q: &MxArray, scales: &MxArray) -> QuantizedLinear {
        QuantizedLinear::new(
            w_q.clone(),
            scales.clone(),
            None,
            None,
            MXFP8_GROUP_SIZE,
            MXFP8_BITS,
            MXFP8_MODE.to_string(),
        )
    }

    /// MXFP4-quantize a 2D bf16 weight, returning `(packed_weight, uint8 E2M1
    /// scales)` (mxfp4 has no biases) — mirrors [`quantize_mxfp8`] with the
    /// mxfp4 mode / group_size / bits. Used to build a NON-mxfp8 projection for
    /// the negative gate test.
    fn quantize_mxfp4(weight: &MxArray) -> (MxArray, MxArray) {
        let mut out_q: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_s: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_b: *mut sys::mlx_array = std::ptr::null_mut();
        let ok = unsafe {
            sys::mlx_quantize(
                weight.as_raw_ptr(),
                MXFP4_GROUP_SIZE,
                MXFP4_BITS,
                c"mxfp4".as_ptr(),
                &mut out_q,
                &mut out_s,
                &mut out_b,
            )
        };
        assert!(ok, "mlx_quantize mxfp4 failed");
        (
            MxArray::from_handle(out_q, "q").expect("q"),
            MxArray::from_handle(out_s, "s").expect("s"),
        )
    }

    /// A fresh mxfp4 `QuantizedLinear` over shared (cloned) packed operands
    /// (`with_input_amax` consumes `self`, so each variant needs its own
    /// struct).
    fn make_mxfp4_linear(w_q: &MxArray, scales: &MxArray) -> QuantizedLinear {
        QuantizedLinear::new(
            w_q.clone(),
            scales.clone(),
            None,
            None,
            MXFP4_GROUP_SIZE,
            MXFP4_BITS,
            MXFP4_MODE.to_string(),
        )
    }

    /// Assert two bf16 outputs are byte-for-byte identical via their native u16
    /// payload (no f32 round-trip — see project memory: an f32 cast can hide a
    /// 1-ULP bf16 divergence).
    fn assert_bf16_bit_identical(a: &MxArray, b: &MxArray, ctx: &str) {
        a.eval();
        b.eval();
        let av = a.to_uint16_native().unwrap();
        let bv = b.to_uint16_native().unwrap();
        assert_eq!(av.len(), bv.len(), "{ctx}: length mismatch");
        let bad = av.iter().zip(bv.iter()).filter(|(x, y)| x != y).count();
        assert_eq!(bad, 0, "{ctx}: {bad}/{} bf16 words differ", av.len());
    }

    fn to_f32(a: &MxArray) -> Vec<f32> {
        a.astype(DType::Float32)
            .expect("astype f32")
            .to_float32()
            .expect("to_float32")
            .to_vec()
    }

    /// Max absolute elementwise difference between two same-shape arrays.
    fn max_abs_diff(a: &MxArray, b: &MxArray) -> f32 {
        let av = to_f32(a);
        let bv = to_f32(b);
        assert_eq!(av.len(), bv.len(), "shape mismatch in max_abs_diff");
        av.iter()
            .zip(&bv)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// The mxfp8 forward fake-quantizes its activation input to per-tensor E4M3
    /// when `input_amax` is set: the output matches a "fake-quant x then the
    /// same matmul" reference, and differs from the `input_amax == None`
    /// (bf16-activation) baseline. Proves Task 4's forward routing.
    #[test]
    fn forward_applies_fp8_fake_quant_when_amax_present() {
        let (n, k) = (32i64, 64i64); // k % MXFP8_GROUP_SIZE == 0
        let mut state = 0x0F80_1234u64;

        // Random bf16 weight [N, K], mxfp8-quantized.
        let wv: Vec<f32> = (0..n * k).map(|_| next_f32(&mut state)).collect();
        let w_bf16 = MxArray::from_float32(&wv, &[n, k])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let (w_q, scales) = quantize_mxfp8(&w_bf16);

        // Random bf16 activations [M, K] spanning [-2, 2] so amax=2.0 pushes
        // magnitudes to the top of the E4M3 grid (meaningful quant error).
        let m = 4i64;
        let xv: Vec<f32> = (0..m * k).map(|_| next_f32(&mut state)).collect();
        let x = MxArray::from_float32(&xv, &[m, k])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();

        // Baseline: input_amax = None (bf16 activations, current behaviour).
        let base = make_mxfp8_linear(&w_q, &scales).forward(&x).unwrap();

        // amax path: forward should fake-quant x internally.
        let got = make_mxfp8_linear(&w_q, &scales)
            .with_input_amax(Some(2.0))
            .forward(&x)
            .unwrap();

        // Reference: fake-quant x explicitly, then the plain (None) matmul.
        let xq = fp8_fake_quant(&x, 2.0).unwrap();
        let want = make_mxfp8_linear(&w_q, &scales).forward(&xq).unwrap();

        let d_got_want = max_abs_diff(&got, &want);
        assert!(
            d_got_want <= 1e-3,
            "amax forward must equal fake-quant-then-matmul reference; max|Δ|={d_got_want}"
        );

        let d_got_base = max_abs_diff(&got, &base);
        assert!(
            d_got_base > 1e-3,
            "amax path must change the output vs the None baseline; max|Δ|={d_got_base}"
        );
    }

    /// Invariant guard: a NON-mxfp8 projection that erroneously carries a
    /// positive `input_amax` (stale / malformed / hand-edited config) must NOT
    /// fake-quant its activations — its forward is byte-identical to the same
    /// projection with `input_amax == None`. Only `mode == MXFP8_MODE` (the
    /// nvidia recipe's attn/GDN activation-fp8 sites) gets FP8 activations;
    /// mxfp4 FFN, affine gates/in_proj_ba, lm_head, and sym8 stay bf16.
    ///
    /// RED on the unfixed gate (before the `&& self.mode == MXFP8_MODE` guard):
    /// this mxfp4 linear WOULD fake-quant `x` and diverge from the baseline, so
    /// the bit-identical assert fails.
    #[test]
    fn forward_ignores_input_amax_on_non_mxfp8() {
        let (n, k) = (32i64, 64i64); // k % MXFP4_GROUP_SIZE == 0
        let mut state = 0x4F40_9AB1u64;

        // Random bf16 weight [N, K], mxfp4-quantized (a NON-mxfp8 projection —
        // stands in for the mxfp4 FFN / affine / lm_head paths).
        let wv: Vec<f32> = (0..n * k).map(|_| next_f32(&mut state)).collect();
        let w_bf16 = MxArray::from_float32(&wv, &[n, k])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let (w_q, scales) = quantize_mxfp4(&w_bf16);

        // Random bf16 activations [M, K] spanning [-2, 2] so an amax=2.0
        // fake-quant WOULD visibly perturb them if it were (wrongly) applied.
        let m = 4i64;
        let xv: Vec<f32> = (0..m * k).map(|_| next_f32(&mut state)).collect();
        let x = MxArray::from_float32(&xv, &[m, k])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();

        // Baseline: input_amax = None (bf16 activations).
        let base = make_mxfp4_linear(&w_q, &scales).forward(&x).unwrap();

        // Erroneous amax on a non-mxfp8 mode: the mode guard must ignore it, so
        // forward stays byte-identical to the None baseline.
        let got = make_mxfp4_linear(&w_q, &scales)
            .with_input_amax(Some(2.0))
            .forward(&x)
            .unwrap();

        assert_bf16_bit_identical(
            &got,
            &base,
            "non-mxfp8 projection with erroneous input_amax must NOT fake-quant",
        );
    }

    /// The calibration tap records the raw `max|x|` for an mxfp8 projection when
    /// the CURRENT thread is armed, bucketed by `amax_key`. A NON-mxfp8
    /// projection records nothing (mode gate), and a disarmed thread records
    /// nothing (arm gate). Proves Task 5's forward tap.
    #[test]
    fn forward_tap_records_max_abs_for_mxfp8() {
        use crate::calibration::activation_amax::{ActivationAmaxCollector, CALIB_TEST_LOCK};

        // Serialize against every other test that records into the shared
        // running-max map (this file's tap tests + the calibration module tests).
        let _g = CALIB_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ActivationAmaxCollector::disarm_current_thread();
        let _ = ActivationAmaxCollector::take();

        let (n, k) = (32i64, 64i64); // k % MXFP8_GROUP_SIZE == 0
        let mut state = 0x7A9_0055u64;

        let wv: Vec<f32> = (0..n * k).map(|_| next_f32(&mut state)).collect();
        let w_bf16 = MxArray::from_float32(&wv, &[n, k])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let (w_q, scales) = quantize_mxfp8(&w_bf16);

        let m = 4i64;
        let xv: Vec<f32> = (0..m * k).map(|_| next_f32(&mut state)).collect();
        let x = MxArray::from_float32(&xv, &[m, k])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        // Expected max|x| computed from the bf16-rounded values (read back via
        // to_f32), independent of the tap's abs->max->item path.
        let expected = to_f32(&x).iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        // --- mxfp8 + armed thread => records max|x| under amax_key ---
        let key = "layers.0.self_attn.q_proj";
        let lin = make_mxfp8_linear(&w_q, &scales).with_amax_key(Some(key.to_string()));
        ActivationAmaxCollector::arm_current_thread();
        let _ = lin.forward(&x).unwrap();
        ActivationAmaxCollector::disarm_current_thread();
        let recorded = ActivationAmaxCollector::take();
        let got = recorded
            .get(key)
            .copied()
            .expect("mxfp8 tap must record under its amax_key");
        assert!(
            (got - expected).abs() <= 1e-4,
            "tapped max|x| {got} vs expected {expected}"
        );
        assert_eq!(recorded.len(), 1, "only the tapped key; got {recorded:?}");

        // --- NON-mxfp8 (mxfp4) armed => records nothing (mode gate) ---
        // --- mxfp8 but thread disarmed => records nothing (arm gate) ---
        let _ = ActivationAmaxCollector::take();
        let (w_q4, scales4) = quantize_mxfp4(&w_bf16);
        let lin4 = make_mxfp4_linear(&w_q4, &scales4)
            .with_amax_key(Some("layers.0.mlp.gate_proj".to_string()));
        ActivationAmaxCollector::arm_current_thread();
        let _ = lin4.forward(&x).unwrap(); // mxfp4 while armed -> skip (mode gate)
        ActivationAmaxCollector::disarm_current_thread();
        let lin8 = make_mxfp8_linear(&w_q, &scales)
            .with_amax_key(Some("layers.0.self_attn.v_proj".to_string()));
        let _ = lin8.forward(&x).unwrap(); // mxfp8 while disarmed -> skip (arm gate)
        let empty = ActivationAmaxCollector::take();
        assert!(
            empty.is_empty(),
            "non-mxfp8 (mode gate) and disarmed-thread (arm gate) must record nothing; got {empty:?}"
        );

        ActivationAmaxCollector::disarm_current_thread();
    }

    /// Task 3 (fake-quant suppression while calibrating). An mxfp8 projection
    /// WITH `input_amax = Some(2.0)`:
    ///   * armed => forward SUPPRESSES the fake-quant, so its output is
    ///     bit-identical to the RAW (`input_amax = None`) baseline AND it
    ///     records `max|x|` under its `amax_key` (raw-bf16 measurement);
    ///   * NOT armed => forward fake-quants (differs from the raw baseline).
    ///
    /// This proves a re-calibration on an already-calibrated model measures raw
    /// bf16 (modelopt parity), and that the calibrated fake-quant is unchanged
    /// for normal inference.
    #[test]
    fn forward_suppresses_fake_quant_while_calibrating() {
        use crate::calibration::activation_amax::{ActivationAmaxCollector, CALIB_TEST_LOCK};

        let _g = CALIB_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ActivationAmaxCollector::disarm_current_thread();
        let _ = ActivationAmaxCollector::take();

        let (n, k) = (32i64, 64i64); // k % MXFP8_GROUP_SIZE == 0
        let mut state = 0x5011_C0DEu64;

        let wv: Vec<f32> = (0..n * k).map(|_| next_f32(&mut state)).collect();
        let w_bf16 = MxArray::from_float32(&wv, &[n, k])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let (w_q, scales) = quantize_mxfp8(&w_bf16);

        let m = 4i64;
        let xv: Vec<f32> = (0..m * k).map(|_| next_f32(&mut state)).collect();
        let x = MxArray::from_float32(&xv, &[m, k])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let expected = to_f32(&x).iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        // Raw baseline: input_amax = None (no fake-quant).
        let base = make_mxfp8_linear(&w_q, &scales).forward(&x).unwrap();

        // input_amax = Some(2.0) but the thread is ARMED -> fake-quant is
        // suppressed, so the output must equal the raw baseline, and the raw
        // max|x| is recorded under amax_key.
        let key = "layers.0.self_attn.o_proj";
        let lin = make_mxfp8_linear(&w_q, &scales)
            .with_input_amax(Some(2.0))
            .with_amax_key(Some(key.to_string()));
        ActivationAmaxCollector::arm_current_thread();
        let got_armed = lin.forward(&x).unwrap();
        ActivationAmaxCollector::disarm_current_thread();
        let recorded = ActivationAmaxCollector::take();

        assert_bf16_bit_identical(
            &got_armed,
            &base,
            "armed forward must suppress fake-quant -> raw bf16 baseline",
        );
        let rec = recorded
            .get(key)
            .copied()
            .expect("armed mxfp8 forward must record raw max|x|");
        assert!(
            (rec - expected).abs() <= 1e-4,
            "recorded raw max|x| {rec} vs expected {expected}"
        );

        // Same input_amax = Some(2.0) but NOT armed -> forward fake-quants, so
        // it must diverge from the raw baseline (proves the suppression above is
        // the arm flag, not a broken amax path).
        let got_unarmed = make_mxfp8_linear(&w_q, &scales)
            .with_input_amax(Some(2.0))
            .forward(&x)
            .unwrap();
        let d = max_abs_diff(&got_unarmed, &base);
        assert!(
            d > 1e-3,
            "unarmed forward with input_amax must fake-quant (differ from baseline); max|Δ|={d}"
        );

        ActivationAmaxCollector::disarm_current_thread();
    }
}

#[cfg(test)]
mod kquant_builder_tests {
    use super::*;
    use crate::array::DType;

    /// A minimal well-typed K-quant group under `{prefix}.*`: uint32 packed
    /// `.weight`, int8 (Q6_K) / uint8 (Q4_K/Q5_K) `.scales`, float16 `.biases`.
    fn kquant_params(prefix: &str, mode: PerLayerMode) -> HashMap<String, MxArray> {
        let scales = match mode {
            PerLayerMode::Q6K => MxArray::from_float32(&[1.0, -1.0, 2.0, -2.0], &[2, 2])
                .unwrap()
                .astype(DType::Int8)
                .unwrap(),
            _ => MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
                .unwrap()
                .astype(DType::Uint8)
                .unwrap(),
        };
        let biases = MxArray::from_float16(
            &[
                half::f16::from_f32(0.5).to_bits(),
                half::f16::from_f32(0.25).to_bits(),
            ],
            &[2, 1],
        )
        .unwrap();
        let weight = MxArray::from_uint32(&[0u32; 2 * 4], &[2, 4]).unwrap();
        HashMap::from([
            (format!("{prefix}.weight"), weight),
            (format!("{prefix}.scales"), scales),
            (format!("{prefix}.biases"), biases),
        ])
    }

    /// The builder installs the mode's fixed (mode string, bits, group_size)
    /// onto the QuantizedLinear — the FFI reads these to drive the two-level
    /// K-quant decode.
    #[test]
    fn kquant_builder_installs_mode_bits_group() {
        for (mode, want_mode, want_bits, want_gs) in [
            (PerLayerMode::Q6K, "q6k", 6, 16),
            (PerLayerMode::Q4K, "q4k", 4, 32),
            (PerLayerMode::Q5K, "q5k", 5, 32),
        ] {
            let p = kquant_params("proj", mode);
            let ql = try_build_kquant_quantized_linear(&p, "proj", mode, "test")
                .expect("well-formed K-quant group builds")
                .expect("scales present => Some");
            assert_eq!(ql.mode(), want_mode);
            assert_eq!(ql.bits, want_bits);
            assert_eq!(ql.group_size, want_gs);
            assert!(
                ql.get_biases().is_some(),
                "K-quant carries mandatory .biases"
            );
        }
    }

    /// Fail-loud contract: absent `.scales` => Ok(None) (a bf16 fallback tensor);
    /// a missing `.weight` or the mandatory `.biases` => Err.
    #[test]
    fn kquant_builder_fail_loud_contract() {
        let mut p = kquant_params("l", PerLayerMode::Q6K);
        p.remove("l.scales");
        assert!(matches!(
            try_build_kquant_quantized_linear(&p, "l", PerLayerMode::Q6K, "test"),
            Ok(None)
        ));

        let mut p = kquant_params("l", PerLayerMode::Q4K);
        p.remove("l.weight");
        assert!(try_build_kquant_quantized_linear(&p, "l", PerLayerMode::Q4K, "test").is_err());

        let mut p = kquant_params("l", PerLayerMode::Q4K);
        p.remove("l.biases");
        assert!(try_build_kquant_quantized_linear(&p, "l", PerLayerMode::Q4K, "test").is_err());
    }
}
