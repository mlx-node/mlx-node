use std::collections::HashMap;
use std::ffi::CString;

use crate::array::MxArray;
use crate::nn::{Activations, Linear};
use crate::quant::prism_hadamard::HadamardTransform;
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

    /// Row-merge two quantized projections into one `[N1 + N2, K]` packed
    /// linear (see [`QuantizedLinear::concat_rows`]). `Ok(None)` when either
    /// side is dense or the packed formats are incompatible — the caller then
    /// keeps the two separate matmuls.
    pub fn concat_rows(&self, other: &LinearProj) -> Result<Option<LinearProj>> {
        match (self, other) {
            (LinearProj::Quantized(a), LinearProj::Quantized(b)) => {
                Ok(a.concat_rows(b)?.map(LinearProj::Quantized))
            }
            _ => Ok(None),
        }
    }

    /// Row-slice view of a quantized projection — see
    /// [`QuantizedLinear::slice_rows`]. `Err` on a dense projection (dense
    /// callers have their own slicing) or a non-plain quantized projection.
    pub fn slice_rows(&self, start: i64, end: i64) -> Result<LinearProj> {
        match self {
            LinearProj::Quantized(ql) => Ok(LinearProj::Quantized(ql.slice_rows(start, end)?)),
            LinearProj::Standard(_) => Err(Error::from_reason(
                "LinearProj::slice_rows requires a quantized projection",
            )),
        }
    }

    /// The calibration-tap config key of the underlying projection, if any.
    /// Used by merge finalization to re-attach a source projection's key onto
    /// its post-merge slice view (slice views are built with `amax_key: None`).
    pub(crate) fn amax_key(&self) -> Option<&str> {
        match self {
            LinearProj::Quantized(ql) => ql.amax_key(),
            LinearProj::Standard(_) => None,
        }
    }

    /// Re-attach a calibration-tap config key — see
    /// [`QuantizedLinear::with_amax_key`]. No-op on a dense projection.
    pub(crate) fn with_amax_key(self, amax_key: Option<String>) -> Self {
        match self {
            LinearProj::Quantized(ql) => LinearProj::Quantized(ql.with_amax_key(amax_key)),
            LinearProj::Standard(_) => self,
        }
    }

    /// Packed weight row count (output features) of the underlying linear.
    pub(crate) fn packed_out_features(&self) -> Result<i64> {
        match self {
            LinearProj::Standard(l) => Ok(l.get_weight().shape()?[0]),
            LinearProj::Quantized(ql) => Ok(ql.get_weight().shape()?[0]),
        }
    }

    pub(crate) fn has_q_gate_block_layout(&self) -> bool {
        matches!(self, LinearProj::Quantized(ql) if ql.has_q_gate_block_layout())
    }

    /// Zero-pad the rows to whole 64-row tiles and tile — see
    /// [`QuantizedLinear::tile_kquant_layout_padded`]. `Ok(false)` on a dense
    /// projection or when the quantized one is not eligible.
    pub(crate) fn tile_kquant_layout_padded(&mut self) -> Result<bool> {
        match self {
            LinearProj::Quantized(ql) => ql.tile_kquant_layout_padded(),
            LinearProj::Standard(_) => Ok(false),
        }
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

    pub(crate) fn has_hadamard(&self) -> bool {
        matches!(self, LinearProj::Quantized(ql) if ql.has_hadamard())
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
        /// Row-merged gate|up projection (`[gate_N + up_N, K]` packed) and
        /// its `gate_N` split point, installed by `finalize_gate_up` when the
        /// two packed formats are mergeable. `gate_proj`/`up_proj` then hold
        /// zero-copy row-slice views so getters stay correct and the merged
        /// buffer is the only resident copy. `None` falls back to the two
        /// original matmuls. Boxed to keep the enum small.
        gate_up: Option<Box<(QuantizedLinear, i64)>>,
    },
}

impl MLPVariant {
    fn quantized_activated(
        x: &MxArray,
        gate_proj: &QuantizedLinear,
        up_proj: &QuantizedLinear,
        gate_up: &Option<Box<(QuantizedLinear, i64)>>,
    ) -> Result<MxArray> {
        if let Some(pair) = gate_up {
            let (merged, split) = &**pair;
            let combined = merged.forward(x)?;
            let last = combined.ndim()? as usize - 1;
            let gu = combined.split_sections(&[*split], last as i32)?;
            return Activations::swiglu_compiled(&gu[0], &gu[1]);
        }
        let gate = gate_proj.forward(x)?;
        let up = up_proj.forward(x)?;
        Activations::swiglu_compiled(&gate, &up)
    }

    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        match self {
            MLPVariant::Standard(mlp) => mlp.forward(x),
            MLPVariant::Quantized {
                gate_proj,
                up_proj,
                down_proj,
                gate_up,
            } => {
                let activated = Self::quantized_activated(x, gate_proj, up_proj, gate_up)?;
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

    #[cfg(test)]
    pub(crate) fn has_hadamard(&self) -> Option<(bool, bool, bool)> {
        match self {
            MLPVariant::Standard(_) => None,
            MLPVariant::Quantized {
                gate_proj,
                up_proj,
                down_proj,
                ..
            } => Some((
                gate_proj.has_hadamard(),
                up_proj.has_hadamard(),
                down_proj.has_hadamard(),
            )),
        }
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

    /// E39: finalize stacked gate+up weight. For the quantized variant this
    /// attempts a packed row-merge (`concat_rows`): on success `gate_up`
    /// holds the single `[gate_N + up_N, K]` projection and `gate_proj` /
    /// `up_proj` are swapped for zero-copy row-slice views of it, so the
    /// originals' storage is released and per-projection getters still work.
    /// Any incompatibility (mixed modes/bits/group sizes, asymmetric sidecars,
    /// special output layouts or transforms) leaves the pair unmerged —
    /// `forward` then takes the two-matmul path exactly as before.
    pub fn finalize_gate_up(&mut self) -> Result<()> {
        match self {
            MLPVariant::Standard(mlp) => mlp.finalize_gate_up(),
            MLPVariant::Quantized {
                gate_proj,
                up_proj,
                gate_up,
                ..
            } => {
                if gate_up.is_some() {
                    return Ok(());
                }
                if let Some(merged) = gate_proj.concat_rows(up_proj)? {
                    let gate_rows = gate_proj.weight.shape()?[0];
                    let up_rows = up_proj.weight.shape()?[0];
                    // Slice views are built keyless; re-attach each source's
                    // calibration key so a view forward still records under
                    // the right bucket.
                    let gate_key = gate_proj.amax_key().map(str::to_owned);
                    let up_key = up_proj.amax_key().map(str::to_owned);
                    *gate_proj = merged.slice_rows(0, gate_rows)?.with_amax_key(gate_key);
                    *up_proj = merged
                        .slice_rows(gate_rows, gate_rows + up_rows)?
                        .with_amax_key(up_key);
                    *gate_up = Some(Box::new((merged, gate_rows)));
                }
                Ok(())
            }
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

/// [`try_build_kquant_quantized_linear`] followed by the Tiled64 repack
/// ([`QuantizedLinear::tile_kquant_layout`]) when
/// [`kquant_tiled_enabled`](crate::models::quant_dispatch::kquant_tiled_enabled)
/// (a Metal host). When the projection did tile,
/// its `key_prefix` is pushed onto `tiled`: the repack evaluates the tiled
/// copies, so the loader's row-major `{key_prefix}.weight/.scales/.biases` in
/// `params` are now dead weight it should drop with
/// [`release_tiled_kquant_sources`] once the layer is installed — otherwise
/// both layouts stay resident until the loader's map goes away and the load
/// peak is twice the model (37 GB for an 18 GB Qwen3.8-27B). Projections that
/// stay row-major are not recorded: those lazily mmapped originals still need
/// the loader's final materialization pass.
pub fn try_build_kquant_quantized_linear_tiled(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
    mode: PerLayerMode,
    family: &str,
    tiled: &mut Vec<String>,
) -> Result<Option<QuantizedLinear>> {
    let Some(mut ql) = try_build_kquant_quantized_linear(params, key_prefix, mode, family)? else {
        return Ok(None);
    };
    if crate::models::quant_dispatch::kquant_tiled_enabled() && ql.tile_kquant_layout()? {
        tiled.push(key_prefix.to_string());
    }
    Ok(Some(ql))
}

/// Drop the row-major `.weight`/`.scales`/`.biases` of every prefix in
/// `tiled` from the loader's `params` (see
/// [`try_build_kquant_quantized_linear_tiled`]); the installed projections own
/// the evaluated Tiled64 copies. Call it only after the layer's dense-fallback
/// lookups are done, so a packed group whose peer is dense still fails loud
/// through `params.get` instead of silently skipping. Clears `tiled`.
pub fn release_tiled_kquant_sources(
    params: &mut HashMap<String, MxArray>,
    tiled: &mut Vec<String>,
) {
    for prefix in tiled.drain(..) {
        for suffix in ["weight", "scales", "biases"] {
            params.remove(&format!("{prefix}.{suffix}"));
        }
    }
}

/// Linear layer backed by a serialized quantized weight format.
///
/// Affine, MX/NVFP, and native GGUF K/IQ modes use packed weights and MLX
/// quantized_matmul; sym8 uses dedicated int8 kernels. Plain `fp8_e4m3` is the
/// intentionally non-native exception: it retains raw Uint8 checkpoint
/// storage, reconstructs BF16 once at load, and uses ordinary A16 matmul.
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
    biases: Option<MxArray>, // Affine biases or native K/IQ floating sidecars
    bias: Option<MxArray>,   // Linear bias (additive)
    group_size: i32,
    bits: i32,
    mode: String, // affine, MX/NVFP, native GGUF K/IQ, fp8_e4m3, or sym8
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
    // The second tuple element is populated only by `concat_rows` when both
    // merged projections were calibration sites: the merge shares one input
    // activation, so `forward` records the same `max|x|` under both keys —
    // exactly what the two unmerged projections would have recorded. Boxed so
    // `MLPVariant::Quantized` stays under clippy's enum-size threshold.
    amax_keys: Option<Box<(String, Option<String>)>>,
    output_layout: QuantizedOutputLayout,
    hadamard: Option<HadamardTransform>,
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
            amax_keys: None,
            output_layout: QuantizedOutputLayout::Native,
            hadamard: None,
        }
    }

    /// Repack a K-quant projection's `.weight`/`.scales`/`.biases` into the
    /// 64-row interleaved `Tiled64` layout (`mlx_kquant.h`) and tag `mode`
    /// with [`KQUANT_TILED_SUFFIX`](crate::models::quant_dispatch::KQUANT_TILED_SUFFIX),
    /// so `forward` reaches the `_t64` Metal kernels: a 64-column
    /// threadgroup then reads one contiguous run per 32-input step instead of
    /// 64 rows K/2 bytes apart (1.18-1.40x on the Qwen3.8 M = 8 verify
    /// shapes). The arrays keep their 2-D shape; the bytes are permuted in
    /// place and the row-major originals are released.
    ///
    /// `Ok(true)` when tiled (or already tiled). `Ok(false)` leaves the
    /// projection untouched: not a K-quant mode, a 3-D / expert weight,
    /// `N % 64 != 0`, `K % 256 != 0`, or a projection with a transform or
    /// non-native output layout (tile AFTER `finalize_packed_q_gate_block`
    /// is not required — that permutes whole tiles when `head_dim % 64 == 0`,
    /// see there).
    ///
    /// Row operations stay valid on a tiled projection only in whole tiles:
    /// [`concat_rows`](Self::concat_rows) (both sides tiled, both `N % 64`)
    /// and [`slice_rows`](Self::slice_rows) at 64-aligned bounds.
    pub fn tile_kquant_layout(&mut self) -> Result<bool> {
        self.tile_kquant_layout_impl(false)
    }

    /// [`tile_kquant_layout`](Self::tile_kquant_layout) for a projection whose
    /// row count is not a whole number of tiles: the rows are first zero-padded
    /// to the next multiple of 64 (`.weight`, `.scales`, `.biases` and any
    /// linear `bias` all get zero rows), then tiled. A zero sub-scale with a
    /// zero super-scale decodes to exactly 0 in every K-quant mode — q4k/q5k
    /// `d*sc*q - dmin*m`, q6k/q3k `d*sc*(q-32)`, iq4xs/iq4nl `d*sc*grid[q]`
    /// (`KQScales` in `mlx_kquant.cpp` / `kquant.h`) — so the padded rows
    /// yield exactly-zero output columns and the real columns are untouched.
    ///
    /// The caller owns the consequence: `forward` returns the PADDED width and
    /// [`packed_out_features`](LinearProj::packed_out_features) reports it, so
    /// the consumer must slice its logical width off the output (the GDN
    /// `in_proj_ba`, N = 2 * num_v_heads = 96 on Qwen3.8, pads to 128 so it
    /// keeps merging with the tiled `in_proj_qkvz`). Same `Ok(false)` cases
    /// as the unpadded form except `N % 64 != 0`.
    pub fn tile_kquant_layout_padded(&mut self) -> Result<bool> {
        self.tile_kquant_layout_impl(true)
    }

    fn tile_kquant_layout_impl(&mut self, pad_rows: bool) -> Result<bool> {
        use crate::models::quant_dispatch::{
            KQUANT_TILE_ROWS, KQUANT_TILED_SUFFIX, is_kquant_mode, kquant_tile_rows,
            kquant_tileable, parse_mode_str, split_kquant_layout,
        };
        let (base, already) = split_kquant_layout(&self.mode);
        if already {
            return Ok(true);
        }
        let Some(mode) = parse_mode_str(Some(base)) else {
            return Ok(false);
        };
        if !is_kquant_mode(mode)
            || self.output_layout != QuantizedOutputLayout::Native
            || self.hadamard.is_some()
            || self.fp8_dequant_weight.is_some()
            || self.s_w.is_some()
        {
            return Ok(false);
        }
        let Some(biases) = self.biases.as_ref() else {
            return Ok(false);
        };
        let shape = self.weight.shape()?;
        if shape.len() != 2 || self.scales.ndim()? != 2 || biases.ndim()? != 2 {
            return Ok(false);
        }
        let n = shape[0];
        let k = shape[1] * 32 / i64::from(self.bits);
        let padded_n = (n + KQUANT_TILE_ROWS - 1) / KQUANT_TILE_ROWS * KQUANT_TILE_ROWS;
        if !kquant_tileable(padded_n, k) || (padded_n != n && !pad_rows) {
            return Ok(false);
        }
        // Codes interleave per 32-value unit (`bits` words); the companions
        // per 256-value super-block: q4k/q5k hold (sc, m) and (d, dmin) pairs,
        // the rest one entry, over `super_ratio` groups (IQ4_NL: one).
        let per_group = if matches!(mode, PerLayerMode::Q4K | PerLayerMode::Q5K) {
            2
        } else {
            1
        };
        let super_ratio = match mode {
            PerLayerMode::Q6K | PerLayerMode::Q3K => 16,
            PerLayerMode::IQ4NL => 1,
            _ => 8,
        };
        // Zero rows appended to a row-major `[n, cols]` array (`padded_n - n`
        // of them); the identity when the row count is already whole tiles.
        let pad = |a: &MxArray| -> Result<MxArray> {
            if padded_n == n {
                return Ok(a.clone());
            }
            let cols = a.shape()?[1];
            let zeros = MxArray::zeros(&[padded_n - n, cols], Some(a.dtype()?))?;
            MxArray::concatenate(a, &zeros, 0)
        };
        let weight = kquant_tile_rows(&pad(&self.weight)?, i64::from(self.bits))?;
        let scales = kquant_tile_rows(&pad(&self.scales)?, super_ratio * per_group)?;
        let biases = kquant_tile_rows(&pad(biases)?, per_group)?;
        let bias = match (&self.bias, padded_n == n) {
            (Some(b), false) => Some(MxArray::concatenate(
                b,
                &MxArray::zeros(&[padded_n - n], Some(b.dtype()?))?,
                0,
            )?),
            (b, _) => b.clone(),
        };
        let mut pending = vec![&weight, &scales, &biases];
        pending.extend(bias.as_ref());
        MxArray::eval_arrays_with_context(&pending, "tile_kquant_layout")?;
        self.weight = weight;
        self.scales = scales;
        self.biases = Some(biases);
        self.bias = bias;
        self.mode = format!("{base}{KQUANT_TILED_SUFFIX}");
        Ok(true)
    }

    /// Whether the packed arrays are in the Tiled64 layout.
    pub fn is_kquant_tiled(&self) -> bool {
        crate::models::quant_dispatch::split_kquant_layout(&self.mode).1
    }

    /// Cast affine-mode `scales`/`biases`/`bias` f16→f32 once at load. With
    /// bf16/f32 activations `quantized_matmul` promotes the whole call to f32
    /// anyway (`promote_types(bf16, f16)`), so the kernel receives f32
    /// sidecars either way — pre-casting is bit-identical while removing two
    /// per-forward `AsType` dispatches (three when a linear bias is present).
    /// Skipped under f16 compute: native f16 sidecars already match the
    /// activation and hoisting would flip the call to the f32 promote path.
    pub fn promote_affine_sidecars_to_f32(
        &mut self,
        compute_dtype: crate::array::DType,
    ) -> Result<()> {
        if self.mode != DEFAULT_QUANT_MODE || compute_dtype == crate::array::DType::Float16 {
            return Ok(());
        }
        if self.scales.dtype()? == crate::array::DType::Float16 {
            self.scales = self.scales.astype(crate::array::DType::Float32)?;
        }
        if let Some(b) = &self.biases
            && b.dtype()? == crate::array::DType::Float16
        {
            self.biases = Some(b.astype(crate::array::DType::Float32)?);
        }
        if let Some(b) = &self.bias
            && b.dtype()? == crate::array::DType::Float16
        {
            self.bias = Some(b.astype(crate::array::DType::Float32)?);
        }
        Ok(())
    }

    /// Row-concatenate two quantized projections into one `[N1+N2, K]`
    /// projection that shares a single quantized-matmul dispatch. Packed
    /// weights, scales, and biases are all row-indexed, so the merge is
    /// value-identical to running the two projections and concatenating their
    /// outputs.
    ///
    /// Returns `Ok(None)` when the pair cannot share a dispatch: different
    /// mode/bits/group_size/K, a one-sided `biases`/`bias` presence, or any
    /// per-projection feature that a merged matrix cannot express (hadamard
    /// input transform, FP8 fallback storage, sym8 scale, unequal
    /// `input_amax`, non-native output layout).
    ///
    /// Calibration taps (`amax_keys`) do NOT block the merge: the merged
    /// projection keeps up to two keys and records the shared input under
    /// each, which reproduces the unmerged recordings exactly.
    pub fn concat_rows(&self, other: &QuantizedLinear) -> Result<Option<QuantizedLinear>> {
        if self.mode != other.mode
            || self.bits != other.bits
            || self.group_size != other.group_size
            || self.output_layout != QuantizedOutputLayout::Native
            || other.output_layout != QuantizedOutputLayout::Native
            || self.input_amax != other.input_amax
            || self.fp8_dequant_weight.is_some()
            || other.fp8_dequant_weight.is_some()
            || self.s_w.is_some()
            || other.s_w.is_some()
            || self.hadamard.is_some()
            || other.hadamard.is_some()
            // An already-merged projection carries a peer key a second merge
            // cannot represent; refuse rather than silently drop it.
            || self.amax_keys.as_ref().is_some_and(|k| k.1.is_some())
            || other.amax_keys.as_ref().is_some_and(|k| k.1.is_some())
        {
            return Ok(None);
        }
        let (w1, w2) = (self.weight.shape()?, other.weight.shape()?);
        if w1.len() != 2 || w2.len() != 2 || w1[1] != w2[1] {
            return Ok(None);
        }
        // Sidecar dtypes must match exactly: a merged matmul sees one scales/
        // biases dtype, and a silent concatenate-promotion would change the
        // dequant arithmetic on half the rows.
        if self.weight.dtype()? != other.weight.dtype()?
            || self.scales.dtype()? != other.scales.dtype()?
            || match (&self.biases, &other.biases) {
                (Some(a), Some(b)) => a.dtype()? != b.dtype()?,
                (None, None) => false,
                _ => true,
            }
            || match (&self.bias, &other.bias) {
                (Some(a), Some(b)) => a.dtype()? != b.dtype()?,
                (None, None) => false,
                _ => true,
            }
        {
            return Ok(None);
        }
        let weight = MxArray::concatenate(&self.weight, &other.weight, 0)?;
        let scales = MxArray::concatenate(&self.scales, &other.scales, 0)?;
        let biases = match (&self.biases, &other.biases) {
            (Some(a), Some(b)) => Some(MxArray::concatenate(a, b, 0)?),
            (None, None) => None,
            _ => return Ok(None),
        };
        let bias = match (&self.bias, &other.bias) {
            (Some(a), Some(b)) => Some(MxArray::concatenate(a, b, 0)?),
            (None, None) => None,
            _ => return Ok(None),
        };
        weight.eval();
        scales.eval();
        if let Some(b) = &biases {
            b.eval();
        }
        if let Some(b) = &bias {
            b.eval();
        }
        // Carry both sources' calibration keys (deduplicated) so a merged
        // mxfp8 site still records under each per-layer config key.
        let a_key = self.amax_keys.as_ref().map(|k| k.0.clone());
        let b_key = other.amax_keys.as_ref().map(|k| k.0.clone());
        let amax_keys = match (a_key, b_key) {
            (Some(a), Some(b)) if a != b => Some(Box::new((a, Some(b)))),
            (a, b) => a.or(b).map(|k| Box::new((k, None))),
        };
        Ok(Some(QuantizedLinear {
            weight,
            scales,
            biases,
            bias,
            group_size: self.group_size,
            bits: self.bits,
            mode: self.mode.clone(),
            fp8_dequant_weight: None,
            s_w: None,
            input_amax: self.input_amax,
            amax_keys,
            output_layout: QuantizedOutputLayout::Native,
            hadamard: None,
        }))
    }

    /// A `[start, end)` row-slice view of this projection's packed weights.
    /// The view shares storage (no copy) and forwards normally — an axis-0
    /// slice of a row-contiguous packed weight is itself row-contiguous.
    /// Intended for keeping per-projection accessors cheap after a
    /// [`concat_rows`](Self::concat_rows) merge replaces the originals.
    ///
    /// Returns `Err` on out-of-range rows or on projections carrying features
    /// a view cannot represent (hadamard, fp8 storage, sym8, non-native output
    /// layout).
    ///
    /// The view always gets `amax_key: None` — slice rows do not know which
    /// source's calibration key applies to them, so callers replacing a
    /// calibration site with a view must re-attach the source's key via
    /// [`with_amax_key`](Self::with_amax_key).
    pub fn slice_rows(&self, start: i64, end: i64) -> Result<QuantizedLinear> {
        if self.fp8_dequant_weight.is_some()
            || self.s_w.is_some()
            || self.hadamard.is_some()
            || self.output_layout != QuantizedOutputLayout::Native
        {
            return Err(Error::from_reason(
                "QuantizedLinear::slice_rows requires a plain projection",
            ));
        }
        let rows = self.weight.shape()?;
        if rows.len() != 2 || start < 0 || end > rows[0] || start >= end {
            return Err(Error::from_reason(format!(
                "QuantizedLinear::slice_rows out of range: [{start},{end}) of {:?}",
                rows.to_vec()
            )));
        }
        // Tiled64 interleaves 64 rows per unit, so only whole tiles are a
        // contiguous (and still tiled) row range.
        if self.is_kquant_tiled() {
            let tile = crate::models::quant_dispatch::KQUANT_TILE_ROWS;
            if start % tile != 0 || end % tile != 0 {
                return Err(Error::from_reason(format!(
                    "QuantizedLinear::slice_rows on a {} projection needs {tile}-aligned bounds, got [{start},{end})",
                    self.mode
                )));
            }
        }
        Ok(QuantizedLinear {
            weight: self.weight.slice_axis(0, start, end)?,
            scales: self.scales.slice_axis(0, start, end)?,
            biases: self
                .biases
                .as_ref()
                .map(|b| b.slice_axis(0, start, end))
                .transpose()?,
            bias: self
                .bias
                .as_ref()
                .map(|b| b.slice_axis(0, start, end))
                .transpose()?,
            group_size: self.group_size,
            bits: self.bits,
            mode: self.mode.clone(),
            fp8_dequant_weight: None,
            s_w: None,
            input_amax: self.input_amax,
            amax_keys: None,
            output_layout: QuantizedOutputLayout::Native,
            hadamard: None,
        })
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
            amax_keys: None,
            output_layout: QuantizedOutputLayout::Native,
            hadamard: None,
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

    /// The projection's calibration-tap config key, if this is an
    /// activation-FP8 calibration site.
    pub(crate) fn amax_key(&self) -> Option<&str> {
        self.amax_keys.as_ref().map(|k| k.0.as_str())
    }

    /// Attach the projection's normalized config key for the calibration tap.
    ///
    /// Consuming builder used at the load-time dispatch site (next to
    /// [`with_input_amax`](Self::with_input_amax)) so the activation-amax
    /// collector can bucket recorded `max|activation|` by projection. `None` is
    /// the default (no calibration bucket — test-fabricated instances).
    pub fn with_amax_key(mut self, amax_key: Option<String>) -> Self {
        self.amax_keys = amax_key.map(|k| Box::new((k, None)));
        self
    }

    pub(crate) fn with_hadamard(mut self, transform: Option<HadamardTransform>) -> Result<Self> {
        if transform.is_some()
            && (self.mode != DEFAULT_QUANT_MODE
                || self.bits != 2
                || self.group_size != 128
                || self.input_amax.is_some())
        {
            return Err(Error::from_reason(format!(
                "prism_hadamard rotations require an affine 2-bit group-size-128 projection without input_amax; '{}' resolved to mode={} bits={} group_size={}",
                self.amax_key().unwrap_or("<unnamed>"),
                self.mode,
                self.bits,
                self.group_size
            )));
        }
        if transform.is_some() && crate::quant::prism_hadamard::hoist_metadata_enabled() {
            self.scales = self.scales.astype(crate::array::DType::Float32)?;
            self.scales.eval();
            self.biases = self
                .biases
                .as_ref()
                .map(|b| b.astype(crate::array::DType::Float32))
                .transpose()?;
            if let Some(biases) = &self.biases {
                biases.eval();
            }
        }
        self.hadamard = transform;
        Ok(self)
    }

    pub(crate) fn has_hadamard(&self) -> bool {
        self.hadamard.is_some()
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
            amax_keys: None,
            output_layout: QuantizedOutputLayout::Native,
            hadamard: None,
        }
    }

    /// Reorder a packed q/gate projection's output rows from checkpoint order
    /// `[Q_h0,G_h0,Q_h1,G_h1,...]` to `[Q_all_heads,G_all_heads]`.
    ///
    /// All row-coupled operands are materialized together before replacing the
    /// stored arrays. Affine and every native GGUF K/IQ mode are row-coupled:
    /// permuting axis 0 of weight/scales/quantization-bias sidecars preserves
    /// the packed code stream and never dequantizes it. Once this returns
    /// `true`, the original-order arrays are no longer retained by the
    /// projection. Other modes or incompatible direct-constructor inputs stay
    /// untouched and return `false` so attention keeps its native per-head
    /// split.
    pub(crate) fn finalize_packed_q_gate_block(
        &mut self,
        num_heads: i32,
        head_dim: i32,
    ) -> Result<bool> {
        if self.output_layout == QuantizedOutputLayout::QGateBlock {
            return Ok(true);
        }
        let (base_mode, tiled) = crate::models::quant_dispatch::split_kquant_layout(&self.mode);
        let mode = crate::models::quant_dispatch::parse_mode_str(Some(base_mode));
        let row_permutable = self.mode == DEFAULT_QUANT_MODE
            || mode.is_some_and(crate::models::quant_dispatch::is_kquant_mode);
        if !row_permutable {
            return Ok(false);
        }
        // On a Tiled64 projection the 2-D rows are tile-major, so a row
        // permutation is only a tile permutation when it moves whole
        // 64-aligned blocks: the q/gate reorder moves `head_dim`-row blocks.
        if tiled && i64::from(head_dim) % crate::models::quant_dispatch::KQUANT_TILE_ROWS != 0 {
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
            || quant_biases_shape[0] != expected_rows
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

    /// String-mode mirror of
    /// [`crate::models::quant_dispatch::admits_static_fp8_activation`] (the
    /// parser-side gate on `input_amax`); the two must agree or a config that
    /// parses cleanly would silently skip the fake-quant.
    ///
    /// Gates the fake-quant ONLY — the calibration tap in `forward` is
    /// deliberately narrower.
    fn admits_static_fp8_activation(&self) -> bool {
        (self.mode == MXFP8_MODE || self.mode == DEFAULT_QUANT_MODE)
            && self.bits == 8
            && self.group_size == 32
    }

    /// Forward pass using quantized_matmul (sym8 routes to the int8 W8A8
    /// kernels instead — `mlx_quantized_matmul` has no sym8 pack).
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let transformed = self
            .hadamard
            .as_ref()
            .map(|t| t.apply(x, false))
            .transpose()?;
        let x = transformed.as_ref().unwrap_or(x);
        // Hoisted FP32 scale/bias metadata requires FP32 projection inputs,
        // but a normal Prism load feeds the transform the 16-bit residual
        // stream (the packed embedding keeps FP16 scales). Promote the
        // transformed activation instead of rejecting the dtype; the
        // transform itself keeps its residual-dtype output contract so
        // same-input projections still share one cached result.
        let xp_owned;
        let x = if self.hadamard.is_some()
            && self.scales.dtype()? == crate::array::DType::Float32
            && x.dtype()? != crate::array::DType::Float32
        {
            xp_owned = x.astype(crate::array::DType::Float32)?;
            &xp_owned
        } else {
            x
        };
        // Whether THIS thread is the calibrating model thread (thread-local, so
        // a concurrently-running inference model on another thread never trips
        // this). Read once and reused for both the tap and the fake-quant
        // suppression below.
        let calibrating =
            crate::calibration::activation_amax::ActivationAmaxCollector::is_calibrating();

        // Activation-amax calibration tap (modelopt MaxCalibrator): record the
        // raw bf16 `max|x|` BEFORE any fake-quant (which is suppressed while
        // calibrating, below, so a re-calibration stays at raw-bf16 parity).
        //
        // The tap PRODUCES an `input_amax`; the fake-quant below CONSUMES one,
        // so this `mode == MXFP8_MODE` gate is deliberately NARROWER than
        // `admits_static_fp8_activation()` and must stay that way. It is the
        // only nvidia-recipe test there is: the qwen3_5 loaders attach an
        // `amax_key` to every attn/GDN site whatever its quant mode, and
        // `--q-recipe unsloth --q-bits 6 --q-group-size 32` packs those sites
        // as affine 8/32. Widening the tap would make `mlx calibrate` rewrite
        // such a checkpoint's config.json with `input_amax` — switching on FP8
        // activation fake-quant for a model that was never a W8A8_FP8 source.
        if calibrating
            && self.mode == MXFP8_MODE
            && let Some(keys) = &self.amax_keys
        {
            // A `concat_rows` merge of two calibration sites carries both keys
            // and records the shared input under each — the same `max|x|` the
            // unmerged pair would have recorded.
            crate::calibration::activation_amax::ActivationAmaxCollector::record(&keys.0, x)?;
            if let Some(peer) = &keys.1 {
                crate::calibration::activation_amax::ActivationAmaxCollector::record(peer, x)?;
            }
        }

        // Calibrated per-tensor FP8 (E4M3) activation fake-quant, matching
        // NVIDIA modelopt's static W8A8 attention/GDN math. SUPPRESSED while
        // this thread is calibrating: the calibration pass must measure the raw
        // bf16 activation regardless of any existing `input_amax` (an upstream
        // fake-quant would perturb downstream activations, breaking raw-bf16
        // MaxCalibrator parity on a re-calibration).
        //
        // Otherwise the gate requires BOTH a positive `input_amax` AND
        // `admits_static_fp8_activation()`, so a stale or hand-edited config
        // cannot fake-quant a projection that was never calibrated — the
        // invariant is enforced here rather than trusted from the loaders.
        // Apple GPUs have no fp8 matmul hardware: numeric parity, not speed.
        let xq_owned;
        let x = if calibrating {
            x
        } else {
            match self.input_amax {
                Some(amax) if amax > 0.0 && self.admits_static_fp8_activation() => {
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

        // BF16 rows against F32 sidecars (Qwen3.5 GGUF affine): one primitive
        // with the promoted path's F32 math and a single BF16 rounding, instead
        // of casting x up and the result back. Small row counts only, so M=1
        // decode and prefill graphs keep their standalone casts.
        if activation_dtype == Some(crate::array::DType::BFloat16)
            && self.bias.is_none()
            && self.scales.dtype()? == crate::array::DType::Float32
            && let Some(biases) = &self.biases
            && biases.dtype()? == crate::array::DType::Float32
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            let k = x.shape_at(x.ndim()?.saturating_sub(1))?;
            let rows = if k > 0 { x.size()? as i64 / k } else { 0 };
            if (2..=8).contains(&rows) {
                let handle = unsafe {
                    sys::mlx_quantized_matmul_affine_bf16(
                        x.handle.0,
                        self.weight.handle.0,
                        self.scales.handle.0,
                        biases.handle.0,
                        self.group_size,
                        self.bits,
                    )
                };
                if !handle.is_null() {
                    return MxArray::from_handle(handle, "quantized_matmul_affine_bf16");
                }
            }
        }

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

    /// Quantization mode discriminator string (affine, MX/NVFP, native GGUF
    /// K/IQ, `fp8_e4m3`, or `sym8`).
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

        // Affine gather-QMM promotes mixed activation/sidecar dtypes to FP32.
        // Preserve its arithmetic, then restore the routed activation dtype at
        // this expert projection boundary. MX/NV/K-quant modes already return
        // `x`'s dtype and remain byte-for-byte unchanged.
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
                std::ptr::null_mut(),
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

    pub fn get_weight(&self) -> &MxArray {
        &self.weight
    }

    pub fn get_scales(&self) -> &MxArray {
        &self.scales
    }

    /// Model-owned pre-transposed BF16 reconstruction retained by the plain
    /// E4M3 expert fallback. Serialized Uint8 storage remains in `weight`.
    pub(crate) fn reconstructed_fp8_weight(&self) -> Option<&MxArray> {
        self.fp8_dequant_weight_t.as_ref()
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

        assert!(linear.finalize_packed_q_gate_block(h, d).unwrap());
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
            linear.finalize_packed_q_gate_block(h, d).unwrap(),
            "the finalized layout must be idempotent"
        );
    }

    #[test]
    fn q4k_q_gate_block_reorders_different_companion_widths_without_dequantizing() {
        let h = 3;
        let d = 2;
        let rows = 12i64;
        let permutation = q_gate_block_permutation(h, d, rows).unwrap();
        let weight_values = (0..rows)
            .flat_map(|row| [row as u32 * 100, row as u32 * 100 + 1])
            .collect::<Vec<_>>();
        let scales_values = (0..rows)
            .flat_map(|row| {
                [
                    row as u8 * 4,
                    row as u8 * 4 + 1,
                    row as u8 * 4 + 2,
                    row as u8 * 4 + 3,
                ]
            })
            .collect::<Vec<_>>();
        let quant_bias_values = (0..rows)
            .flat_map(|row| [0x3000u16 + row as u16, 0x3400u16 + row as u16])
            .collect::<Vec<_>>();

        let mut linear = QuantizedLinear::new(
            MxArray::from_uint32(&weight_values, &[rows, 2]).unwrap(),
            MxArray::from_uint8(&scales_values, &[rows, 4]).unwrap(),
            Some(MxArray::from_float16(&quant_bias_values, &[rows, 2]).unwrap()),
            None,
            32,
            4,
            "q4k".to_string(),
        );

        assert!(linear.finalize_packed_q_gate_block(h, d).unwrap());
        assert!(linear.has_q_gate_block_layout());
        assert_eq!(linear.weight.dtype().unwrap(), crate::array::DType::Uint32);
        assert_eq!(linear.scales.dtype().unwrap(), crate::array::DType::Uint8);
        assert_eq!(
            linear.biases.as_ref().unwrap().dtype().unwrap(),
            crate::array::DType::Float16
        );

        let expected_weight = permutation
            .iter()
            .flat_map(|&row| [row as u32 * 100, row as u32 * 100 + 1])
            .collect::<Vec<_>>();
        let expected_scales = permutation
            .iter()
            .flat_map(|&row| {
                [
                    row as u8 * 4,
                    row as u8 * 4 + 1,
                    row as u8 * 4 + 2,
                    row as u8 * 4 + 3,
                ]
            })
            .collect::<Vec<_>>();
        let expected_biases = permutation
            .iter()
            .flat_map(|&row| [0x3000u16 + row as u16, 0x3400u16 + row as u16])
            .collect::<Vec<_>>();
        assert_eq!(linear.weight.to_uint32().unwrap().to_vec(), expected_weight);
        assert_eq!(linear.scales.to_uint8().unwrap().to_vec(), expected_scales);
        assert_eq!(
            linear
                .biases
                .as_ref()
                .unwrap()
                .to_uint16_native()
                .unwrap()
                .to_vec(),
            expected_biases
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
        assert!(!non_affine.finalize_packed_q_gate_block(3, 2).unwrap());
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
        assert!(!incompatible.finalize_packed_q_gate_block(3, 2).unwrap());
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

    /// Invariant guard: a projection outside the static-FP8 weight shapes that
    /// erroneously carries a positive `input_amax` must NOT fake-quant — its
    /// forward is byte-identical to `input_amax == None`.
    ///
    /// RED without the `&& admits_static_fp8_activation()` guard.
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

    /// Affine-quantize a 2D bf16 weight at 8 bits / group 32, returning
    /// `(packed_weight, scales, biases)`.
    fn quantize_affine_8_32(weight: &MxArray) -> (MxArray, MxArray, MxArray) {
        let mut out_q: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_s: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_b: *mut sys::mlx_array = std::ptr::null_mut();
        let ok = unsafe {
            sys::mlx_quantize(
                weight.as_raw_ptr(),
                32,
                8,
                c"affine".as_ptr(),
                &mut out_q,
                &mut out_s,
                &mut out_b,
            )
        };
        assert!(ok, "mlx_quantize affine 8/32 failed");
        (
            MxArray::from_handle(out_q, "q").expect("q"),
            MxArray::from_handle(out_s, "s").expect("s"),
            MxArray::from_handle(out_b, "b").expect("b"),
        )
    }

    /// A fresh affine 8/32 `QuantizedLinear` per call, since the `with_*`
    /// builders consume `self`.
    fn make_affine_8_32_linear(
        w_q: &MxArray,
        scales: &MxArray,
        biases: &MxArray,
    ) -> QuantizedLinear {
        QuantizedLinear::new(
            w_q.clone(),
            scales.clone(),
            Some(biases.clone()),
            None,
            32,
            8,
            DEFAULT_QUANT_MODE.to_string(),
        )
    }

    /// The calibration tap must stay NARROWER than the fake-quant gate: an
    /// affine 8/32 projection CONSUMES an amax (it fake-quants) but must never
    /// PRODUCE one through the tap, or `mlx calibrate` would start rewriting
    /// config.json for unsloth-recipe qwen3_5 checkpoints.
    ///
    /// MUTATION CAUGHT: the tap using `admits_static_fp8_activation()` instead
    /// of `self.mode == MXFP8_MODE`.
    #[test]
    fn forward_tap_ignores_affine_8_32_site() {
        use crate::calibration::activation_amax::{ActivationAmaxCollector, CALIB_TEST_LOCK};

        let _g = CALIB_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ActivationAmaxCollector::disarm_current_thread();
        let _ = ActivationAmaxCollector::take();

        let (n, k) = (32i64, 64i64); // k % 32 == 0
        let mut state = 0xA1FF_1E32u64;

        let wv: Vec<f32> = (0..n * k).map(|_| next_f32(&mut state)).collect();
        let w_bf16 = MxArray::from_float32(&wv, &[n, k])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let (w_q, scales, biases) = quantize_affine_8_32(&w_bf16);

        let m = 4i64;
        let xv: Vec<f32> = (0..m * k).map(|_| next_f32(&mut state)).collect();
        let x = MxArray::from_float32(&xv, &[m, k])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();

        // An `is_activation_fp8_site` key, as the qwen3_5 loader threads onto
        // an attn projection whatever its quant mode.
        let lin = make_affine_8_32_linear(&w_q, &scales, &biases)
            .with_amax_key(Some("layers.0.self_attn.q_proj".to_string()));
        ActivationAmaxCollector::arm_current_thread();
        let _ = lin.forward(&x).unwrap();
        ActivationAmaxCollector::disarm_current_thread();
        let recorded = ActivationAmaxCollector::take();
        assert!(
            recorded.is_empty(),
            "an affine 8/32 site must not be tapped; got {recorded:?}"
        );

        // Anti-vacuity: this fixture DOES satisfy the wider fake-quant gate, so
        // the empty map above is the tap's mxfp8 test doing the work.
        let base = make_affine_8_32_linear(&w_q, &scales, &biases)
            .forward(&x)
            .unwrap();
        let quantized = make_affine_8_32_linear(&w_q, &scales, &biases)
            .with_input_amax(Some(2.0))
            .forward(&x)
            .unwrap();
        let d = max_abs_diff(&quantized, &base);
        assert!(
            d > 1e-3,
            "affine 8/32 must still CONSUME input_amax (fake-quant); max|Δ|={d}"
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

/// `tile_kquant_layout`: the Tiled64 repack keeps `forward` value-identical
/// (bit-identical where the row-major route is the same kernel), tags the
/// mode, refuses ineligible shapes, and keeps the whole-tile row operations
/// (`concat_rows`, `slice_rows`, the q/gate block reorder) correct.
#[cfg(test)]
mod kquant_tiled_tests {
    use super::*;
    use crate::array::DType;
    use crate::models::quant_dispatch::{KQUANT_TILED_SUFFIX, kquant_untile_rows};

    fn lcg(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *state
    }

    /// A random q4k `[n, k]` projection (uint8 (sc, m) scales, f16 (d, dmin)).
    fn q4k(n: i64, k: i64, seed: u32) -> QuantizedLinear {
        let mut st = seed;
        let words: Vec<u32> = (0..n * k / 8).map(|_| lcg(&mut st)).collect();
        let scales: Vec<u8> = (0..n * k / 16)
            .map(|_| (lcg(&mut st) % 48 + 1) as u8)
            .collect();
        let half_scales = [0x2800u16, 0x2c00, 0x3000, 0x3200];
        let biases: Vec<u16> = (0..n * k / 128)
            .map(|_| half_scales[(lcg(&mut st) as usize) % half_scales.len()])
            .collect();
        QuantizedLinear::new(
            MxArray::from_uint32(&words, &[n, k / 8]).unwrap(),
            MxArray::from_uint8(&scales, &[n, k / 16]).unwrap(),
            Some(MxArray::from_float16(&biases, &[n, k / 128]).unwrap()),
            None,
            32,
            4,
            "q4k".to_string(),
        )
    }

    fn x(m: i64, k: i64, seed: u32) -> MxArray {
        let mut st = seed;
        let bits: Vec<u16> = (0..m * k)
            .map(|_| {
                let v = ((lcg(&mut st) >> 16) as i32 - 32_768) as f32 / 32_768.0;
                (v.to_bits() >> 16) as u16
            })
            .collect();
        MxArray::from_bfloat16(&bits, &[1, m, k]).unwrap()
    }

    fn bits_of(a: &MxArray) -> Vec<u32> {
        a.eval();
        a.astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    fn close(a: &MxArray, b: &MxArray, rel: f32, what: &str) {
        let (a, b) = (bits_of(a), bits_of(b));
        assert_eq!(a.len(), b.len(), "{what}: lengths");
        let peak = b.iter().fold(0f32, |m, &v| m.max(f32::from_bits(v).abs()));
        let worst = a
            .iter()
            .zip(&b)
            .map(|(&p, &q)| (f32::from_bits(p) - f32::from_bits(q)).abs())
            .fold(0f32, f32::max);
        assert!(worst / peak <= rel, "{what}: off by {worst} of peak {peak}");
    }

    fn gpu() -> bool {
        // SAFETY: nullary predicate that catches internally.
        unsafe { mlx_sys::mlx_metal_is_available() }
    }

    #[test]
    fn tile_tags_mode_and_keeps_forward_values() {
        let (n, k) = (256i64, 1024i64);
        let row_major = q4k(n, k, 11);
        let mut tiled = q4k(n, k, 11);
        assert!(tiled.tile_kquant_layout().unwrap());
        assert!(tiled.is_kquant_tiled());
        assert_eq!(tiled.mode(), format!("q4k{KQUANT_TILED_SUFFIX}"));
        assert!(tiled.tile_kquant_layout().unwrap(), "idempotent");
        assert_eq!(
            tiled.get_weight().shape().unwrap().to_vec(),
            row_major.get_weight().shape().unwrap().to_vec(),
            "the 2-D shape is kept"
        );
        // The bytes moved and untile restores them.
        assert_ne!(bits_of(tiled.get_weight()), bits_of(row_major.get_weight()));
        assert_eq!(
            bits_of(&kquant_untile_rows(tiled.get_weight(), 4).unwrap()),
            bits_of(row_major.get_weight())
        );
        assert_eq!(
            bits_of(&kquant_untile_rows(tiled.get_biases().unwrap(), 2).unwrap()),
            bits_of(row_major.get_biases().unwrap())
        );
        assert_eq!(
            bits_of(&kquant_untile_rows(tiled.get_scales(), 16).unwrap()),
            bits_of(row_major.get_scales())
        );
        for m in [1i64, 3, 8, 16, 64] {
            let a = x(m, k, 100 + m as u32);
            let ours = tiled.forward(&a).unwrap();
            let reference = row_major.forward(&a).unwrap();
            // The GPU M=3 (qmv_wide) and M>=16 (qmm) routes are the same
            // kernel in both layouts; M=1 and M=8 change kernels on the
            // GPU, and the CPU reference is always bit-identical.
            let same_kernel = !gpu() || matches!(m, 3 | 16 | 64);
            if same_kernel {
                assert_eq!(
                    bits_of(&ours),
                    bits_of(&reference),
                    "M={m}: not bit-identical"
                );
            } else {
                close(&ours, &reference, 3e-2, &format!("M={m}"));
            }
        }
    }

    #[test]
    fn tile_refuses_ineligible_projections() {
        // N % 64 != 0.
        let mut ql = q4k(96, 512, 1);
        assert!(!ql.tile_kquant_layout().unwrap());
        assert_eq!(ql.mode(), "q4k");
        // (K % 256 != 0 exists only for IQ4_NL's 32-value blocks; the C++
        // validator case is in tests/kquant_tiled.rs.)
        // Not a K-quant mode.
        let mut affine = QuantizedLinear::new(
            MxArray::from_uint32(&[0u32; 128 * 64], &[128, 64]).unwrap(),
            MxArray::from_float16(&[0x3c00u16; 128 * 8], &[128, 8]).unwrap(),
            Some(MxArray::from_float16(&[0u16; 128 * 8], &[128, 8]).unwrap()),
            None,
            64,
            4,
            DEFAULT_QUANT_MODE.to_string(),
        );
        assert!(!affine.tile_kquant_layout().unwrap());
        assert_eq!(affine.mode(), DEFAULT_QUANT_MODE);
    }

    #[test]
    fn tiled_row_operations_stay_whole_tile() {
        let k = 512i64;
        let (mut a, mut b) = (q4k(128, k, 3), q4k(192, k, 4));
        let (a_rm, b_rm) = (q4k(128, k, 3), q4k(192, k, 4));
        assert!(a.tile_kquant_layout().unwrap());
        // Mixed layouts never merge.
        assert!(
            a.concat_rows(&b).unwrap().is_none(),
            "tiled + row-major must not merge"
        );
        assert!(b.tile_kquant_layout().unwrap());
        let merged = a
            .concat_rows(&b)
            .unwrap()
            .expect("two tiled projections merge");
        assert!(merged.is_kquant_tiled());
        let merged_rm = a_rm.concat_rows(&b_rm).unwrap().unwrap();
        let xa = x(3, k, 5);
        assert_eq!(
            bits_of(&merged.forward(&xa).unwrap()),
            bits_of(&merged_rm.forward(&xa).unwrap()),
            "merged tiled forward differs"
        );
        // 64-aligned slices are valid tiled views, unaligned ones are refused.
        let view = merged.slice_rows(128, 320).unwrap();
        assert!(view.is_kquant_tiled());
        assert_eq!(
            bits_of(&view.forward(&xa).unwrap()),
            bits_of(&b_rm.forward(&xa).unwrap()),
            "tiled slice view differs from its source"
        );
        assert!(merged.slice_rows(32, 128).is_err());
        assert!(merged.slice_rows(0, 100).is_err());
    }

    #[test]
    fn tiled_q_gate_block_permutes_whole_tiles() {
        // H=2 heads x D=64 -> 256 rows: the block reorder moves 64-row blocks.
        let (heads, dim, k) = (2, 64, 512i64);
        let n = i64::from(2 * heads * dim);
        let mut tiled = q4k(n, k, 7);
        let mut row_major = q4k(n, k, 7);
        assert!(tiled.tile_kquant_layout().unwrap());
        assert!(tiled.finalize_packed_q_gate_block(heads, dim).unwrap());
        assert!(row_major.finalize_packed_q_gate_block(heads, dim).unwrap());
        let xa = x(3, k, 8);
        assert_eq!(
            bits_of(&tiled.forward(&xa).unwrap()),
            bits_of(&row_major.forward(&xa).unwrap()),
            "q/gate block reorder on a tiled projection differs"
        );
        // D % 64 != 0 cannot be expressed as a tile permutation: left native.
        let mut odd = q4k(i64::from(2 * 4 * 32), k, 9);
        assert!(odd.tile_kquant_layout().unwrap());
        assert!(!odd.finalize_packed_q_gate_block(4, 32).unwrap());
        assert!(!odd.has_q_gate_block_layout());
    }

    /// The GDN `in_proj_ba` case: a 96-row projection cannot tile as is, but
    /// zero-padded to 128 rows it tiles, merges with a tiled 1024-row peer,
    /// and the merged forward reproduces the all-row-major merge on the real
    /// columns while the 32 padding columns are exactly zero — for the M = 1
    /// decode, M = 8 verify and a prefill M.
    #[test]
    fn padded_rows_tile_merge_and_stay_invisible() {
        let (n_qkvz, n_ba, k) = (1024i64, 96i64, 1024i64);
        let (qkvz_rm, ba_rm) = (q4k(n_qkvz, k, 21), q4k(n_ba, k, 22));
        let merged_rm = qkvz_rm.concat_rows(&ba_rm).unwrap().unwrap();

        let (mut qkvz, mut ba) = (q4k(n_qkvz, k, 21), q4k(n_ba, k, 22));
        assert!(qkvz.tile_kquant_layout().unwrap());
        assert!(
            !ba.tile_kquant_layout().unwrap(),
            "96 rows are not whole tiles"
        );
        assert!(ba.tile_kquant_layout_padded().unwrap());
        assert!(ba.is_kquant_tiled());
        assert!(ba.tile_kquant_layout_padded().unwrap(), "idempotent");
        let padded_n = 128i64;
        assert_eq!(ba.get_weight().shape().unwrap().to_vec(), [padded_n, k / 8]);
        assert_eq!(
            ba.get_scales().shape().unwrap().to_vec(),
            [padded_n, k / 16]
        );
        assert_eq!(
            ba.get_biases().unwrap().shape().unwrap().to_vec(),
            [padded_n, k / 128]
        );
        // The appended rows are zero bytes once untiled.
        let untiled = kquant_untile_rows(ba.get_weight(), 4).unwrap();
        let tail = bits_of(&untiled.slice_axis(0, n_ba, padded_n).unwrap());
        assert!(tail.iter().all(|&b| b == 0), "padding codes must be zero");
        assert_eq!(
            bits_of(&untiled.slice_axis(0, 0, n_ba).unwrap()),
            bits_of(ba_rm.get_weight()),
            "the real rows must be untouched"
        );

        let merged = qkvz
            .concat_rows(&ba)
            .unwrap()
            .expect("tiled qkvz + padded tiled ba merge");
        assert!(merged.is_kquant_tiled());
        assert_eq!(
            merged.get_weight().shape().unwrap()[0],
            n_qkvz + padded_n,
            "merged N is the padded width"
        );
        for m in [1i64, 8, 64] {
            let a = x(m, k, 200 + m as u32);
            let out = merged.forward(&a).unwrap();
            assert_eq!(
                out.shape().unwrap().to_vec(),
                [1, m, n_qkvz + padded_n],
                "M={m}: forward returns the padded width"
            );
            let real = out.slice_axis(2, 0, n_qkvz + n_ba).unwrap();
            let pad = out.slice_axis(2, n_qkvz + n_ba, n_qkvz + padded_n).unwrap();
            assert!(
                bits_of(&pad).iter().all(|&b| f32::from_bits(b) == 0.0),
                "M={m}: padding columns must decode to exactly zero"
            );
            let reference = merged_rm.forward(&a).unwrap();
            // Same rule as `tile_tags_mode_and_keeps_forward_values`: the
            // prefill route (and the CPU reference) is the same kernel in
            // both layouts; the GPU M=1 / M=8 routes change kernels.
            if !gpu() || m == 64 {
                assert_eq!(
                    bits_of(&real),
                    bits_of(&reference),
                    "M={m}: padded+tiled merge differs from the row-major merge"
                );
            } else {
                close(&real, &reference, 3e-2, &format!("M={m}"));
            }
            // The standalone padded ba agrees with its row-major self on the
            // same terms, and its own padding is zero too.
            let ba_out = ba.forward(&a).unwrap();
            let ba_pad = ba_out.slice_axis(2, n_ba, padded_n).unwrap();
            assert!(
                bits_of(&ba_pad).iter().all(|&b| f32::from_bits(b) == 0.0),
                "M={m}: standalone padding columns must be zero"
            );
            let ba_real = ba_out.slice_axis(2, 0, n_ba).unwrap();
            let ba_ref = ba_rm.forward(&a).unwrap();
            if !gpu() || m == 64 {
                assert_eq!(bits_of(&ba_real), bits_of(&ba_ref), "M={m}: padded ba");
            } else {
                close(&ba_real, &ba_ref, 3e-2, &format!("M={m}: padded ba"));
            }
        }
        // The padded ba is a whole-tile slice view of the merge.
        let view = merged.slice_rows(n_qkvz, n_qkvz + padded_n).unwrap();
        assert!(view.is_kquant_tiled());
        let a = x(3, k, 300);
        assert_eq!(
            bits_of(&view.forward(&a).unwrap()),
            bits_of(&ba.forward(&a).unwrap()),
            "slice view of the padded rows differs from the padded ba"
        );
    }
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
        assert_eq!(
            qsl.reconstructed_fp8_weight().unwrap().nbytes(),
            2 * 3 * 4 * std::mem::size_of::<u16>(),
            "loader-visible residency must expose the retained BF16 expert reconstruction"
        );

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

#[cfg(test)]
mod switch_affine_dtype_tests {
    use super::*;
    use crate::array::DType;

    #[test]
    fn affine_q4_gather_qmm_restores_bfloat16_activation_dtype() {
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

#[cfg(test)]
mod gate_up_merge_tests {
    use super::*;
    use crate::array::DType;

    /// Affine-quantize a `[N, K]` bf16 weight with `mlx_quantize`, returning the
    /// packed `(weight, scales, biases)` triple `QuantizedLinear::new` expects.
    fn quantize_affine(
        weight: &MxArray,
        group_size: i32,
        bits: i32,
    ) -> (MxArray, MxArray, MxArray) {
        let mut out_q: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_s: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_b: *mut sys::mlx_array = std::ptr::null_mut();
        let ok = unsafe {
            sys::mlx_quantize(
                weight.as_raw_ptr(),
                group_size,
                bits,
                c"affine".as_ptr(),
                &mut out_q,
                &mut out_s,
                &mut out_b,
            )
        };
        assert!(ok, "mlx_quantize affine failed");
        (
            MxArray::from_handle(out_q, "q").expect("q"),
            MxArray::from_handle(out_s, "s").expect("s"),
            MxArray::from_handle(out_b, "b").expect("b"),
        )
    }

    fn bf16_weight(seed: u64, n: i64, k: i64) -> MxArray {
        // Deterministic pseudo-random-ish pattern, no RNG dependency.
        let len = (n * k) as usize;
        let vals: Vec<u16> = (0..len)
            .map(|i| {
                let h = (i as u64).wrapping_mul(0x9E37_79B9).wrapping_add(seed);
                half::bf16::from_f32(((h % 1024) as f32 - 512.0) / 128.0).to_bits()
            })
            .collect();
        MxArray::from_bfloat16(&vals, &[n, k]).unwrap()
    }

    fn make_affine_linear(weight: &MxArray, group_size: i32, bits: i32) -> QuantizedLinear {
        let (w, s, b) = quantize_affine(weight, group_size, bits);
        QuantizedLinear::new(
            w,
            s,
            Some(b),
            None,
            group_size,
            bits,
            DEFAULT_QUANT_MODE.to_string(),
        )
    }

    fn assert_bit_identical(a: &MxArray, b: &MxArray, ctx: &str) {
        a.eval();
        b.eval();
        assert_eq!(
            a.to_uint16_native().unwrap(),
            b.to_uint16_native().unwrap(),
            "{ctx}: outputs differ"
        );
    }

    #[test]
    fn concat_rows_matches_separate_forwards_bit_identical() {
        let (n1, n2, k) = (48i64, 32i64, 128i64);
        let gate = make_affine_linear(&bf16_weight(7, n1, k), 32, 4);
        let up = make_affine_linear(&bf16_weight(11, n2, k), 32, 4);

        let merged = gate
            .concat_rows(&up)
            .unwrap()
            .expect("same-format projections must merge");

        // Different M tile widths exercise qmv / qmv_wide / qmm dispatch.
        for m in [1i64, 2, 4, 6, 8, 16] {
            let x = bf16_weight(3 + m as u64, m, k);
            let combined = merged.forward(&x).unwrap();
            let last = combined.ndim().unwrap() as usize - 1;
            let got_gate = combined.slice_axis(last, 0, n1).unwrap();
            let got_up = combined.slice_axis(last, n1, n1 + n2).unwrap();
            assert_bit_identical(&got_gate, &gate.forward(&x).unwrap(), "gate half");
            assert_bit_identical(&got_up, &up.forward(&x).unwrap(), "up half");
        }
    }

    #[test]
    fn concat_rows_rejects_incompatible_pairs() {
        let w = bf16_weight(5, 32, 128);
        let gate = make_affine_linear(&w, 32, 4);
        let other_group = make_affine_linear(&w, 64, 4);
        let other_bits = make_affine_linear(&w, 32, 8);
        let mxfp4 = {
            let mut out_q: *mut sys::mlx_array = std::ptr::null_mut();
            let mut out_s: *mut sys::mlx_array = std::ptr::null_mut();
            let mut out_b: *mut sys::mlx_array = std::ptr::null_mut();
            assert!(unsafe {
                sys::mlx_quantize(
                    w.as_raw_ptr(),
                    32,
                    4,
                    c"mxfp4".as_ptr(),
                    &mut out_q,
                    &mut out_s,
                    &mut out_b,
                )
            });
            QuantizedLinear::new(
                MxArray::from_handle(out_q, "q").unwrap(),
                MxArray::from_handle(out_s, "s").unwrap(),
                None,
                None,
                32,
                4,
                MXFP4_MODE.to_string(),
            )
        };
        assert!(gate.concat_rows(&other_group).unwrap().is_none());
        assert!(gate.concat_rows(&other_bits).unwrap().is_none());
        assert!(gate.concat_rows(&mxfp4).unwrap().is_none());

        // Mismatched K (input width) cannot share a matmul.
        let narrow = make_affine_linear(&bf16_weight(9, 32, 64), 32, 4);
        assert!(gate.concat_rows(&narrow).unwrap().is_none());
    }

    /// Two keyed mxfp8 calibration sites merge (loaders attach `amax_key` to
    /// every attn/GDN site regardless of recipe): the merged projection
    /// records the shared input's `max|x|` under BOTH keys, and each key's
    /// slice view keeps recording under its own key when the merged path is
    /// not taken.
    #[test]
    fn concat_rows_preserves_calibration_keys() {
        use crate::calibration::activation_amax::{ActivationAmaxCollector, CALIB_TEST_LOCK};

        let _g = CALIB_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ActivationAmaxCollector::disarm_current_thread();
        let _ = ActivationAmaxCollector::take();

        let (n, k) = (32i64, 64i64); // k % MXFP8_GROUP_SIZE == 0
        let w_bf16 = bf16_weight(29, 2 * n, k);
        let mut out_q: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_s: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_b: *mut sys::mlx_array = std::ptr::null_mut();
        assert!(unsafe {
            sys::mlx_quantize(
                w_bf16.as_raw_ptr(),
                MXFP8_GROUP_SIZE,
                MXFP8_BITS,
                c"mxfp8".as_ptr(),
                &mut out_q,
                &mut out_s,
                &mut out_b,
            )
        });
        let w_q = MxArray::from_handle(out_q, "q").unwrap();
        let scales = MxArray::from_handle(out_s, "s").unwrap();
        let make = |w: &MxArray, s: &MxArray, key: &str| {
            QuantizedLinear::new(
                w.clone(),
                s.clone(),
                None,
                None,
                MXFP8_GROUP_SIZE,
                MXFP8_BITS,
                MXFP8_MODE.to_string(),
            )
            .with_amax_key(Some(key.to_string()))
        };
        let key_a = "layers.0.self_attn.k_proj";
        let key_b = "layers.0.self_attn.v_proj";
        let a = make(
            &w_q.slice_axis(0, 0, n).unwrap(),
            &scales.slice_axis(0, 0, n).unwrap(),
            key_a,
        );
        let b = make(
            &w_q.slice_axis(0, n, 2 * n).unwrap(),
            &scales.slice_axis(0, n, 2 * n).unwrap(),
            key_b,
        );
        let merged = a
            .concat_rows(&b)
            .unwrap()
            .expect("two keyed mxfp8 sites must merge");

        let x = bf16_weight(31, 4, k);
        let expected = {
            let xv = x.astype(DType::Float32).unwrap();
            xv.eval();
            xv.to_float32()
                .unwrap()
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max)
        };

        // Merged forward records under both source keys.
        ActivationAmaxCollector::arm_current_thread();
        let _ = merged.forward(&x).unwrap();
        ActivationAmaxCollector::disarm_current_thread();
        let recorded = ActivationAmaxCollector::take();
        for key in [key_a, key_b] {
            let got = recorded
                .get(key)
                .copied()
                .unwrap_or_else(|| panic!("merged projection must record under {key}"));
            assert!(
                (got - expected).abs() <= 1e-4,
                "{key}: recorded {got} vs expected {expected}"
            );
        }

        // A keyless slice view re-keyed to its source still records on its
        // own (the env-killed-merge fallback path).
        let view = merged
            .slice_rows(0, n)
            .unwrap()
            .with_amax_key(Some(key_a.to_string()));
        ActivationAmaxCollector::arm_current_thread();
        let _ = view.forward(&x).unwrap();
        ActivationAmaxCollector::disarm_current_thread();
        let recorded = ActivationAmaxCollector::take();
        assert!(recorded.contains_key(key_a));
        assert!(!recorded.contains_key(key_b));
    }

    #[test]
    fn quantized_mlp_finalize_gate_up_is_bit_identical_and_keeps_getters() {
        let (i, k) = (64i64, 128i64);
        let gate = make_affine_linear(&bf16_weight(13, i, k), 32, 4);
        let up = make_affine_linear(&bf16_weight(17, i, k), 32, 4);
        // down: [K_out=K_hidden, K_in=I] — reuse the affine helper.
        let down = make_affine_linear(&bf16_weight(19, k, i), 32, 4);

        let gate_w = gate.get_weight().clone();
        let up_w = up.get_weight().clone();
        let mut mlp = MLPVariant::Quantized {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
            gate_up: None,
        };

        for m in [1i64, 6, 8] {
            let x = bf16_weight(23 + m as u64, m, k);
            let want = mlp.forward(&x).unwrap();
            // Re-finalize must be a no-op once merged (idempotent).
            mlp.finalize_gate_up().unwrap();
            mlp.finalize_gate_up().unwrap();
            let got = mlp.forward(&x).unwrap();
            assert_bit_identical(&got, &want, "merged vs unmerged MLP");
        }

        // Getters still expose per-projection packed rows (views).
        assert_eq!(
            mlp.get_gate_proj_weight().shape().unwrap().to_vec(),
            gate_w.shape().unwrap().to_vec()
        );
        assert_eq!(
            mlp.get_gate_proj_weight().to_uint32().unwrap().to_vec(),
            gate_w.to_uint32().unwrap().to_vec(),
            "gate rows must be the leading slice of the merged weight"
        );
        assert_eq!(
            mlp.get_up_proj_weight().to_uint32().unwrap().to_vec(),
            up_w.to_uint32().unwrap().to_vec(),
        );
    }
}

#[cfg(test)]
mod affine_qmv_wide_lock_tests {
    use super::*;
    use crate::array::DType;

    fn quantize(weight: &MxArray, group_size: i32, bits: i32) -> (MxArray, MxArray, MxArray) {
        let mut q: *mut sys::mlx_array = std::ptr::null_mut();
        let mut s: *mut sys::mlx_array = std::ptr::null_mut();
        let mut b: *mut sys::mlx_array = std::ptr::null_mut();
        let ok = unsafe {
            sys::mlx_quantize(
                weight.as_raw_ptr(),
                group_size,
                bits,
                c"affine".as_ptr(),
                &mut q,
                &mut s,
                &mut b,
            )
        };
        assert!(ok, "mlx_quantize affine failed");
        (
            MxArray::from_handle(q, "q").unwrap(),
            MxArray::from_handle(s, "s").unwrap(),
            MxArray::from_handle(b, "b").unwrap(),
        )
    }

    /// Deterministic values without an RNG; `fine` leaves bits that BF16
    /// cannot hold, so the F32 variant is a genuine F32 input.
    fn pattern(seed: u64, len: usize, fine: bool) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let h = (i as u64 ^ seed)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .rotate_left(17)
                    .wrapping_add(seed);
                let coarse = ((h % 2048) as f32 - 1024.0) / 256.0;
                if fine {
                    coarse + ((h >> 20) % 1000) as f32 * 1.0e-6
                } else {
                    coarse
                }
            })
            .collect()
    }

    fn fnv(hash: &mut u64, bytes: impl IntoIterator<Item = u8>) {
        for byte in bytes {
            *hash ^= u64::from(byte);
            *hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }

    fn qmm(x: &MxArray, w: &MxArray, s: &MxArray, b: &MxArray, gs: i32, bits: i32) -> MxArray {
        let handle = unsafe {
            sys::mlx_quantized_matmul(
                x.handle.0,
                w.handle.0,
                s.handle.0,
                b.handle.0,
                true,
                gs,
                bits,
                c"affine".as_ptr(),
            )
        };
        let y = MxArray::from_handle(handle, "qmm").unwrap();
        y.eval();
        y
    }

    /// Output bits of the affine `qmv_wide` kernels (M = 2..8, both tile caps)
    /// recorded on an M5 (GPU gen 17) before the kernel gained a separate
    /// scale type. Variants: BF16 x with BF16 sidecars, F32 x with F32
    /// sidecars, and BF16 x with F32 sidecars (promoted to the F32 kernel).
    #[test]
    fn affine_qmv_wide_outputs_hash_locked() {
        if !unsafe { sys::mlx_metal_is_available() } {
            eprintln!("SKIP affine_qmv_wide_outputs_hash_locked: no Metal");
            return;
        }
        let arch = unsafe { sys::mlx_gpu_architecture_gen() };
        if arch != 17 {
            eprintln!(
                "SKIP affine_qmv_wide_outputs_hash_locked: hashes recorded on GPU gen 17, this is {arch}"
            );
            return;
        }
        const EXPECTED: &[(&str, u64)] = &[
            ("gs32_b4_n40_k768_bf16", 0xE959_49D8_9222_7145),
            ("gs32_b4_n40_k768_f32", 0xFE8D_BABC_6F5B_8694),
            ("gs32_b4_n40_k768_bf16x_f32s", 0xF5F3_A62C_0411_87AE),
            ("gs32_b4_n2048_k256_bf16", 0xC223_B96B_B6A3_B1B3),
            ("gs32_b4_n2048_k256_f32", 0x778A_0F43_DA22_BE73),
            ("gs32_b4_n2048_k256_bf16x_f32s", 0xE69D_EF7C_3E8D_1976),
            ("gs32_b8_n40_k768_bf16", 0x3112_1A8E_72EF_DECE),
            ("gs32_b8_n40_k768_f32", 0xC8F3_AB62_6C10_B00D),
            ("gs32_b8_n40_k768_bf16x_f32s", 0x2F08_A8D9_4DCD_E1C9),
            ("gs32_b8_n2048_k256_bf16", 0x1D05_D26E_BC61_1E68),
            ("gs32_b8_n2048_k256_f32", 0xBC58_F0AA_5264_5C74),
            ("gs32_b8_n2048_k256_bf16x_f32s", 0x7C7B_A989_7953_C2FC),
            ("gs64_b4_n40_k768_bf16", 0x4AEC_574E_57B0_B5E8),
            ("gs64_b4_n40_k768_f32", 0x3622_6BBE_2B41_2E20),
            ("gs64_b4_n40_k768_bf16x_f32s", 0x990D_3728_44D1_5BFE),
            ("gs64_b4_n2048_k256_bf16", 0x1C8A_762F_8862_241B),
            ("gs64_b4_n2048_k256_f32", 0x476C_6C5D_C588_994E),
            ("gs64_b4_n2048_k256_bf16x_f32s", 0xBD06_D6D6_979F_8AC8),
            ("gs64_b8_n40_k768_bf16", 0x1A8B_4D7E_5314_2D71),
            ("gs64_b8_n40_k768_f32", 0x1F6E_E45D_4D22_C7E4),
            ("gs64_b8_n40_k768_bf16x_f32s", 0xC5EF_6FB7_CE18_8BBF),
            ("gs64_b8_n2048_k256_bf16", 0x3A53_A1D8_DF2A_1051),
            ("gs64_b8_n2048_k256_f32", 0xC90C_E0A3_BA3E_906D),
            ("gs64_b8_n2048_k256_bf16x_f32s", 0x2AE8_B026_94F3_C2AA),
        ];
        let mut got = Vec::new();
        for (gs, bits) in [(32, 4), (32, 8), (64, 4), (64, 8)] {
            for (n, k) in [(40i64, 768i64), (2048, 256)] {
                let wf = pattern(n as u64 * 31 + k as u64, (n * k) as usize, true);
                let w32 = MxArray::from_float32(&wf, &[n, k]).unwrap();
                let w16 = w32.astype(DType::BFloat16).unwrap();
                let (q16, s16, b16) = quantize(&w16, gs, bits);
                let (q32, s32, b32) = quantize(&w32, gs, bits);
                let (s16p, b16p) = (
                    s16.astype(DType::Float32).unwrap(),
                    b16.astype(DType::Float32).unwrap(),
                );
                for variant in ["bf16", "f32", "bf16x_f32s"] {
                    let mut hash = 0xCBF2_9CE4_8422_2325u64;
                    for m in 2i64..=8 {
                        let xf = pattern(m as u64 * 7 + bits as u64, (m * k) as usize, true);
                        let x32 = MxArray::from_float32(&xf, &[m, k]).unwrap();
                        let x16 = x32.astype(DType::BFloat16).unwrap();
                        let y = match variant {
                            "bf16" => qmm(&x16, &q16, &s16, &b16, gs, bits),
                            "f32" => qmm(&x32, &q32, &s32, &b32, gs, bits),
                            _ => qmm(&x16, &q16, &s16p, &b16p, gs, bits),
                        };
                        if y.dtype().unwrap() == DType::Float32 {
                            let v = y.to_float32().unwrap();
                            fnv(&mut hash, v.iter().flat_map(|f| f.to_bits().to_le_bytes()));
                        } else {
                            let v = y.to_uint16_native().unwrap();
                            fnv(&mut hash, v.iter().flat_map(|h| h.to_le_bytes()));
                        }
                    }
                    got.push((format!("gs{gs}_b{bits}_n{n}_k{k}_{variant}"), hash));
                }
            }
        }
        for (label, hash) in &got {
            eprintln!("        (\"{label}\", 0x{hash:016X}),");
        }
        let expected: Vec<(String, u64)> =
            EXPECTED.iter().map(|(l, h)| (l.to_string(), *h)).collect();
        assert_eq!(got, expected, "affine qmv_wide output bits moved");
    }

    fn bf16_bits(y: &MxArray) -> Vec<u16> {
        y.eval();
        assert_eq!(y.dtype().unwrap(), DType::BFloat16);
        y.to_uint16_native().unwrap()
    }

    /// The promoted path the mixed primitive replaces: F32 x, F32 kernels,
    /// then one cast to BF16.
    fn promoted(
        x: &MxArray,
        w: &MxArray,
        s: &MxArray,
        b: &MxArray,
        gs: i32,
        bits: i32,
    ) -> Vec<u16> {
        bf16_bits(
            &qmm(&x.astype(DType::Float32).unwrap(), w, s, b, gs, bits)
                .astype(DType::BFloat16)
                .unwrap(),
        )
    }

    fn mixed_ffi(
        x: &MxArray,
        w: &MxArray,
        s: &MxArray,
        b: &MxArray,
        gs: i32,
        bits: i32,
    ) -> Option<MxArray> {
        let handle = unsafe {
            sys::mlx_quantized_matmul_affine_bf16(
                x.handle.0, w.handle.0, s.handle.0, b.handle.0, gs, bits,
            )
        };
        (!handle.is_null()).then(|| MxArray::from_handle(handle, "mixed").unwrap())
    }

    /// GGUF-style sidecars: FP16 storage (more mantissa than BF16, so a BF16
    /// rounding of a scale is visible), promoted to F32 at load.
    fn gguf_linear(n: i64, k: i64, gs: i32, bits: i32) -> QuantizedLinear {
        let w = MxArray::random_normal(&[n, k], 0.0, 0.05, Some(DType::Float32)).unwrap();
        let (q, s, b) = quantize(&w, gs, bits);
        let mut linear = QuantizedLinear::new(
            q,
            s.astype(DType::Float16).unwrap(),
            Some(b.astype(DType::Float16).unwrap()),
            None,
            gs,
            bits,
            DEFAULT_QUANT_MODE.to_string(),
        );
        linear
            .promote_affine_sidecars_to_f32(DType::BFloat16)
            .unwrap();
        assert_eq!(linear.get_scales().dtype().unwrap(), DType::Float32);
        linear
    }

    /// BF16 x with F32 affine sidecars: `QuantizedLinear::forward` (the
    /// mixed primitive for 2..=8 rows) and the mixed FFI at every row count
    /// (native qmv_wide or the in-primitive promoted fallback) must equal the
    /// promoted F32 path bit for bit.
    #[test]
    fn affine_q8_bf16_mixed_matches_promoted_path_bitwise() {
        if !unsafe { sys::mlx_metal_is_available() } {
            eprintln!("SKIP affine_q8_bf16_mixed_matches_promoted_path_bitwise: no Metal");
            return;
        }
        let mut cases = 0;
        for (n, k) in [
            (96i64, 5120i64),
            (1024, 5120),
            (5120, 6144),
            (17, 5120),
            (96, 128),
        ] {
            let linear = gguf_linear(n, k, 32, 8);
            let (w, s) = (linear.get_weight(), linear.get_scales());
            let b = linear.get_biases().unwrap();
            let rounded = s
                .astype(DType::BFloat16)
                .unwrap()
                .astype(DType::Float32)
                .unwrap();
            assert_ne!(
                rounded.to_float32().unwrap().to_vec(),
                s.to_float32().unwrap().to_vec(),
                "scales must carry more precision than BF16"
            );
            for seed in 0..3u64 {
                for amp in [1.0e-3f64, 1.0, 30.0] {
                    for m in [1i64, 2, 3, 4, 5, 6, 7, 8, 9, 16, 64] {
                        let x = MxArray::random_normal(&[1, m, k], 0.0, amp, Some(DType::Float32))
                            .unwrap()
                            .mul_scalar(1.0 + seed as f64 * 0.37)
                            .unwrap()
                            .astype(DType::BFloat16)
                            .unwrap();
                        let want = promoted(&x, w, s, b, 32, 8);
                        let ctx = format!("N={n} K={k} M={m} seed={seed} amp={amp}");
                        assert_eq!(
                            bf16_bits(&linear.forward(&x).unwrap()),
                            want,
                            "forward {ctx}"
                        );
                        let direct = mixed_ffi(&x, w, s, b, 32, 8).expect("mixed FFI accepts");
                        assert_eq!(bf16_bits(&direct), want, "mixed FFI {ctx}");
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 5 * 3 * 3 * 11);
    }

    /// Operands the native kernel does not cover (4-bit / group 64, a
    /// transposed x, batched rows) take the in-primitive promoted fallback,
    /// which must also equal the promoted path bit for bit.
    #[test]
    fn affine_mixed_fallback_matches_promoted_path_bitwise() {
        if !unsafe { sys::mlx_metal_is_available() } {
            eprintln!("SKIP affine_mixed_fallback_matches_promoted_path_bitwise: no Metal");
            return;
        }
        let mut cases = 0;
        for (gs, bits) in [(64, 8), (32, 4), (64, 4)] {
            let linear = gguf_linear(96, 768, gs, bits);
            let (w, s) = (linear.get_weight(), linear.get_scales());
            let b = linear.get_biases().unwrap();
            for m in [1i64, 2, 5, 8, 12] {
                let x = MxArray::random_normal(&[m, 768], 0.0, 1.0, Some(DType::BFloat16)).unwrap();
                let want = promoted(&x, w, s, b, gs, bits);
                let got = mixed_ffi(&x, w, s, b, gs, bits).expect("mixed FFI accepts");
                assert_eq!(bf16_bits(&got), want, "gs={gs} bits={bits} M={m}");
                cases += 1;
            }
        }
        let linear = gguf_linear(96, 768, 32, 8);
        let (w, s) = (linear.get_weight(), linear.get_scales());
        let b = linear.get_biases().unwrap();
        let wide = MxArray::random_normal(&[768, 6], 0.0, 1.0, Some(DType::BFloat16)).unwrap();
        let transposed = wide.transpose(Some(&[1, 0])).unwrap();
        let batched =
            MxArray::random_normal(&[3, 4, 768], 0.0, 1.0, Some(DType::BFloat16)).unwrap();
        for x in [&transposed, &batched] {
            let want = promoted(x, w, s, b, 32, 8);
            assert_eq!(bf16_bits(&mixed_ffi(x, w, s, b, 32, 8).unwrap()), want);
            cases += 1;
        }
        assert_eq!(cases, 17);
    }

    #[test]
    fn affine_mixed_ffi_rejects_other_operands() {
        if !unsafe { sys::mlx_metal_is_available() } {
            eprintln!("SKIP affine_mixed_ffi_rejects_other_operands: no Metal");
            return;
        }
        let linear = gguf_linear(96, 256, 32, 8);
        let (w, s) = (linear.get_weight(), linear.get_scales());
        let b = linear.get_biases().unwrap();
        let x = MxArray::random_normal(&[4, 256], 0.0, 1.0, Some(DType::BFloat16)).unwrap();
        assert!(mixed_ffi(&x, w, s, b, 32, 8).is_some());
        let x16 = x.astype(DType::Float16).unwrap();
        assert!(mixed_ffi(&x16, w, s, b, 32, 8).is_none(), "f16 x");
        let x32 = x.astype(DType::Float32).unwrap();
        assert!(mixed_ffi(&x32, w, s, b, 32, 8).is_none(), "f32 x");
        let s16 = s.astype(DType::BFloat16).unwrap();
        let b16 = b.astype(DType::BFloat16).unwrap();
        assert!(
            mixed_ffi(&x, w, &s16, &b16, 32, 8).is_none(),
            "bf16 sidecars"
        );
        assert!(mixed_ffi(&x, w, s, &b16, 32, 8).is_none(), "mixed sidecars");
        assert!(
            mixed_ffi(&x, w, s, b, 64, 8).is_none(),
            "group size vs sidecar shape"
        );
        assert!(
            mixed_ffi(&x, w, s, b, 32, 4).is_none(),
            "bits vs packed width"
        );
        let narrow = MxArray::random_normal(&[4, 128], 0.0, 1.0, Some(DType::BFloat16)).unwrap();
        assert!(mixed_ffi(&narrow, w, s, b, 32, 8).is_none(), "K mismatch");
    }
}
