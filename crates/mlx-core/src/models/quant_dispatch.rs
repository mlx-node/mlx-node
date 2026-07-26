//! Shared per-layer quantization dispatch types and helpers.
//!
//! Mixed-recipe checkpoints (produced by `--q-mxfp` and friends) can carry a
//! different quantization mode for each layer: affine (4/8-bit affine
//! packing), MXFP8 (E8M0 uint8 scales), MXFP4 (E2M1 4-bit format with uint8
//! scales), NVFP4 (E2M1 4-bit format with E4M3 uint8 scales, group_size 16),
//! or plain E4M3 FP8 (raw uint8 weights plus one float dequant scale per output
//! channel). The persistence layer dispatches to the matching `try_build_*`
//! builder per layer based on a `PerLayerQuant` record.
//!
//! This module is family-neutral on purpose: `qwen3_5`, `qwen3_5_moe`, and
//! `gemma4` all import these types from here, instead of cross-importing from
//! one another. That avoids the awkward inter-family coupling that crept in
//! when `gemma4` reached into `qwen3_5::quantized_linear` for the same enum.

use std::collections::HashMap;
use std::path::Path;

use napi::bindgen_prelude::{Error, Result};
use serde_json::Value;
use tracing::warn;

use crate::array::{DType, MxArray};

const SYM8_GROUP_SIZE_SENTINEL: i32 = -1;

/// `config.json` field naming the zero point a symmetric affine group subtracts
/// from every code.
///
/// A ggml Q4_0 block is `w = d * (q - 8)` and a Q8_0 block is `w = d * (q - 128)`:
/// one f16 scale per 32 weights and no stored offset. MLX affine is
/// `w = scale * q + bias`, so the GGUF importer used to write a `.biases` array
/// whose every entry was `-Z * scale` — 0.5 bpw of pure redundancy. It now omits
/// that array and records `Z` here instead;
/// [`expand_symmetric_affine_biases`](crate::engine::persistence::expand_symmetric_affine_biases)
/// rebuilds the companion at load, before any builder runs, so everything
/// downstream sees the historical `(weight, scales, biases)` shape.
///
/// Deliberately a field on `mode: "affine"` rather than a new mode string: an
/// unrecognised mode makes [`parse_mode_str`] return `None`, which sends an
/// older build into a checkpoint-content heuristic that would GUESS a decoder.
/// Keeping the mode string stable instead routes an older build onto MLX's
/// deterministic "Biases must be provided for affine quantization" throw.
pub const SYMMETRIC_ZERO_POINT_KEY: &str = "symmetric_zero_point";

/// Per-layer quantization mode discriminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerLayerMode {
    /// Standard affine packing with separate `bits` / `group_size` / biases.
    Affine,
    /// MXFP8 (E8M0 uint8 scales, 8-bit packed weights, group_size 32).
    Mxfp8,
    /// MXFP4 (E2M1 4-bit format with uint8 scales, group_size 32).
    Mxfp4,
    /// NVFP4 (E2M1 4-bit format with E4M3 uint8 scales, group_size 16).
    Nvfp4,
    /// Plain E4M3 FP8 weight storage: uint8 `[...,N,K]` weight plus floating
    /// `[...,N,1]` per-output-channel dequant scale. Runtime reconstructs BF16
    /// weights and keeps activations A16; this is not MLX MXFP8 or native W8A8.
    Fp8E4m3,
    /// sym8: per-output-channel symmetric int8 (int8 `[N,K]` weight + f32
    /// `[N]` scales, no biases, group_size is null/meaningless). Consumed by
    /// the int8 kernels (`int8_w8a16_qmv` decode / `int8_w8a8_matmul`
    /// prefill), NEVER by `mlx_quantized_matmul` (there is no affine pack).
    /// Dispatched by dense qwen3_5, qwen3_5_moe (non-expert sublayers only —
    /// attention q/k/v/o, GDN linear-attention, shared-expert MLP body; the
    /// router gate and shared_expert_gate are accuracy-forced affine-8 by
    /// `compute_moe_defaults` + convert's `is_router_gate`, so they never
    /// dispatch sym8), lfm2/lfm2_moe, and gemma4 (all via the shared
    /// `try_build_sym8_quantized_linear`). The per-expert `switch_mlp.*`
    /// gather path (qwen3_5_moe, lfm2_moe) has no sym8 kernel and always
    /// resolves to a forced-affine per-layer override instead.
    Sym8,
    /// ggml Q6_K: 6-bit symmetric super-block (SUPER=256, sub group_size=16).
    /// Imported from an Unsloth Dynamic GGUF bit-identically to llama.cpp:
    /// uint32 LSB-first 6-bit `.weight`, int8 (SIGNED ggml `sc`) `.scales`, and
    /// float16 (ggml `d`) `.biases`. `.biases` holds a per-super-block SCALE,
    /// not an additive bias — the K-quant kernel recomputes `scale = d[g>>4]*sc[g]`
    /// and `bias = -32*scale` from the two companions. Consumed by
    /// `mlx_quantized_matmul` / `mlx_gather_qmm` with mode string `"q6k"`.
    Q6K,
    /// ggml Q4_K: 4-bit asymmetric super-block (SUPER=256, sub group_size=32).
    /// uint32 `.weight`, uint8 `.scales` holding interleaved `(sc, m)` pairs,
    /// float16 `.biases` holding interleaved `(d, dmin)` pairs. `.biases` holds
    /// scales, not additive biases. Mode string `"q4k"`.
    Q4K,
    /// ggml Q5_K: identical layout to `Q4K` with a 5-bit weight plane. Mode
    /// string `"q5k"`.
    Q5K,
}

/// Per-layer quantization metadata parsed from `config.json`.
///
/// `bits` and `group_size` are the affine packing parameters; for `Mxfp8`,
/// `Mxfp4`, and `Nvfp4` they are forced to the matching constants by the
/// builders and are kept here only for fallback/reporting. `Fp8E4m3` has no
/// K-axis group and uses `group_size = -1` as an in-memory sentinel.
///
/// `input_amax` is the per-tensor static FP8 (E4M3) activation scale calibrated
/// by NVIDIA modelopt's MaxCalibrator (`max|activation|` over the calib mix),
/// read from an optional `"input_amax"` field on the tensor's `config.json`
/// quantization override. It is `None` for every layer whose config carries no
/// such field (all non-`Mxfp8` attention/GDN sites and unquantized layers).
/// Carrying it here lets the attention/GDN `QuantizedLinear` fake-quant its
/// activations to E4M3 for modelopt W8A8 numeric parity (consumed downstream).
//
// NOTE: `Eq` is intentionally NOT derived — `Option<f32>` is not `Eq` (floats
// have no total equality). `PartialEq` is sufficient for the map-value and
// merge-conflict comparisons this type is used in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerLayerQuant {
    pub bits: i32,
    pub group_size: i32,
    pub mode: PerLayerMode,
    pub input_amax: Option<f32>,
}

/// Decode a `quantization.mode` string into a `PerLayerMode`.
///
/// Returns `None` when the field is missing or holds an unrecognised value,
/// allowing the caller to fall back to a checkpoint-content heuristic such
/// as `is_mxfp8_checkpoint`.
pub fn parse_mode_str(s: Option<&str>) -> Option<PerLayerMode> {
    match s {
        Some("mxfp4") => Some(PerLayerMode::Mxfp4),
        Some("mxfp8") => Some(PerLayerMode::Mxfp8),
        Some("nvfp4") => Some(PerLayerMode::Nvfp4),
        Some("fp8_e4m3") => Some(PerLayerMode::Fp8E4m3),
        Some("affine") => Some(PerLayerMode::Affine),
        Some("sym8") => Some(PerLayerMode::Sym8),
        Some("q6k") => Some(PerLayerMode::Q6K),
        Some("q4k") => Some(PerLayerMode::Q4K),
        Some("q5k") => Some(PerLayerMode::Q5K),
        _ => None,
    }
}

/// Encode a `PerLayerMode` back into its `quantization.mode` string — the exact
/// inverse of [`parse_mode_str`], and the string every mode-aware MLX entry
/// point is handed.
///
/// Three consumers:
///  - `mlx_quantize` (the MTPLX draft-lm-head quantize helper). C++ has no
///    `sym8` mode at all, and rejects the K-quant strings outright ("can be read
///    but not produced", `ops.cpp:5233`), so a nonsensical draft-head spec in
///    either mode fails loud there instead of mis-packing.
///  - `mlx_dequantize` (the MTPLX source-head dequantize helper). `sym8` still
///    fails loud, but the K-quant strings are IMPLEMENTED (`kq_dequantize`,
///    `ops.cpp:5494`), so a K-quant source head dequantizes rather than throwing.
///  - the mode-aware `Embedding` load in the dense and MoE loaders, which threads
///    the on-disk packing mode of `token_embd` / the tied lm_head through to
///    `mlx_quantized_matmul` so a K-quant embedding is not misread as affine.
pub(crate) fn mode_to_str(mode: PerLayerMode) -> &'static str {
    match mode {
        PerLayerMode::Affine => "affine",
        PerLayerMode::Mxfp8 => "mxfp8",
        PerLayerMode::Mxfp4 => "mxfp4",
        PerLayerMode::Nvfp4 => "nvfp4",
        PerLayerMode::Fp8E4m3 => crate::quant::fp8_weight::FP8_E4M3_MODE,
        PerLayerMode::Sym8 => "sym8",
        PerLayerMode::Q6K => "q6k",
        PerLayerMode::Q4K => "q4k",
        PerLayerMode::Q5K => "q5k",
    }
}

/// True when `mode` is one of the ggml K-quant families (Q6_K / Q4_K / Q5_K).
pub fn is_kquant_mode(mode: PerLayerMode) -> bool {
    matches!(
        mode,
        PerLayerMode::Q6K | PerLayerMode::Q4K | PerLayerMode::Q5K
    )
}

/// True when the resolved quantization settings reference sym8 anywhere
/// (top-level default mode OR any per-layer override).
///
/// Used by the loaders with sym8 dispatch (dense qwen3_5, qwen3_5_moe,
/// lfm2/lfm2_moe, gemma4) to scope-gate the checkpoint: qwen3_5 pins flat KV
/// and disables MTP/vision; qwen3_5_moe disables its speculative MTP head and
/// vision encoder the same way (the MTP module's own `try_build_ql`/
/// `try_build_qsl` closures have unwired `Sym8 => None` arms, so a sym8 MTP
/// head cannot load — the loader fail-softs to plain AR decode) while its
/// AR-decode non-expert sublayers keep dispatching sym8; lfm2 is eager-FLAT
/// only v1, so it skips the C++ compiled-forward registration (it stores 2-D
/// `.weight` tensors in the [N,K] checkpoint orientation, which the shared
/// `sym8_linear_proj` fail-loud rejects) and forces the flat decode shape;
/// gemma4 has no compiled registry and keeps its eager paged default.
pub fn has_sym8_mode(
    top_level_mode: Option<PerLayerMode>,
    per_layer: &HashMap<String, PerLayerQuant>,
) -> bool {
    top_level_mode == Some(PerLayerMode::Sym8)
        || per_layer.values().any(|p| p.mode == PerLayerMode::Sym8)
}

/// True when the resolved quantization settings reference a ggml K-quant mode
/// (Q6_K / Q4_K / Q5_K) anywhere — top-level default OR any per-layer override.
///
/// Used by the LM loaders to scope-gate a K-quant checkpoint the same way
/// `has_sym8_mode` scopes sym8: the speculative MTP head is disabled (the MTP
/// builders map every K-quant mode to `None`, and an imported GGUF never ships
/// an MTP head), so the loader fail-softs to plain AR decode. K-quants pack
/// their weights as uint32 like affine, so — unlike sym8 — the paged KV cache
/// and vision encoders stay supported.
pub fn has_kquant_mode(
    top_level_mode: Option<PerLayerMode>,
    per_layer: &HashMap<String, PerLayerQuant>,
) -> bool {
    top_level_mode.is_some_and(is_kquant_mode) || per_layer.values().any(|p| is_kquant_mode(p.mode))
}

/// Fail-loud guard for the DENSE (unquantized) weight fallbacks of
/// QUANTIZABLE projections: a weight reaching a dense `set_weight` route must
/// be floating-point. A truncated sym8 group (int8 `.weight` whose mandatory
/// `.scales` sidecar is missing/stripped) makes every `try_build_*` builder
/// return "not quantized", so without this guard the int8 bytes would flow
/// into a dense bf16 matmul — the shape validates, the dtype does not, and
/// the logits are garbage. Same for a packed `Uint32` affine weight orphaned
/// from its `.scales`.
///
/// Apply ONLY at the dense fallbacks of quantizable projections — norms and
/// additive biases are never quantized and do not need it.
pub fn ensure_dense_weight_floating(key: &str, w: &MxArray) -> Result<()> {
    let dtype = w.dtype()?;
    match dtype {
        DType::Float32 | DType::Float16 | DType::BFloat16 => Ok(()),
        other => Err(Error::from_reason(format!(
            "dense weight '{key}' has non-float dtype {other:?} — int8/non-float storage \
             requires a quantized group (its '.scales' sidecar is missing/stripped from the \
             checkpoint); refusing to load it through the dense route"
        ))),
    }
}

/// Fail-loud guard for metadata-skewed checkpoints: `{base}.weight` stored as
/// int8 (sym8 storage) while the per-layer quant metadata resolves to a
/// NON-sym8 mode. Without this, the int8 tensor flows into the affine/mxfp
/// builders (`mlx_quantized_matmul` would read it as a packed pack — garbage)
/// and, on lfm2, could register with the compiled C++ path as affine
/// quant-info (the compiled gate keys on config metadata — `has_sym8_mode` —
/// only). Call BEFORE dispatching on `plq.mode` in every per-layer builder.
pub fn ensure_int8_storage_resolves_sym8(
    params: &HashMap<String, MxArray>,
    base: &str,
    mode: PerLayerMode,
    family: &str,
) -> Result<()> {
    if mode == PerLayerMode::Sym8 {
        return Ok(());
    }
    if let Some(w) = params.get(&format!("{base}.weight"))
        && w.dtype().ok() == Some(DType::Int8)
    {
        return Err(Error::from_reason(format!(
            "{family}: '{base}.weight' is int8 (sym8 storage) but its per-layer quant mode \
             resolves to {mode:?} — config drift / stale quantization metadata, refusing to load"
        )));
    }
    Ok(())
}

/// Fail-loud guard for plain E4M3 storage whose metadata resolves to another
/// mode. MXFP8 may also store its weight bytes as `Uint8`, so weight dtype alone
/// is not sufficient: plain FP8 is identified by `Uint8` weight plus a real
/// floating `.scales` sidecar (MX/NVFP float modes use `Uint8` encoded scales;
/// affine uses a non-Uint8 packed weight). If that pair's per-layer override is
/// missing or stale, handing it to `mlx_quantized_matmul` would silently decode
/// garbage.
pub fn ensure_plain_fp8_storage_resolves_fp8_e4m3(
    params: &HashMap<String, MxArray>,
    base: &str,
    mode: PerLayerMode,
    family: &str,
) -> Result<()> {
    if mode == PerLayerMode::Fp8E4m3 {
        return Ok(());
    }
    if let (Some(w), Some(scales)) = (
        params.get(&format!("{base}.weight")),
        params.get(&format!("{base}.scales")),
    ) && w.dtype().ok() == Some(DType::Uint8)
        && matches!(
            scales.dtype().ok(),
            Some(DType::Float32 | DType::Float16 | DType::BFloat16)
        )
    {
        return Err(Error::from_reason(format!(
            "{family}: '{base}.weight' is Uint8 (plain E4M3 FP8 storage) but its per-layer \
             quant mode resolves to {mode:?} — config drift / missing fp8_e4m3 override, \
             refusing to load"
        )));
    }
    Ok(())
}

/// Fail-loud guard for K-quant storage whose metadata resolves to another
/// mode. A K-quant group is `uint32` `.weight` + `int8` (Q6_K) / `uint8`
/// (Q4_K/Q5_K) `.scales` + mandatory `float16` `.biases`. `int8` scales are
/// unique to Q6_K, and `uint8` scales paired with a `float16` `.biases` sidecar
/// are unique to Q4_K/Q5_K (MXFP8/NVFP4 also use `uint8` scales but never a
/// `.biases` companion; affine uses `float16` scales). So if that signature is
/// present while the per-layer mode resolves to a NON-K-quant mode, the config
/// is stale/skewed and handing the group to the affine/mxfp builders would
/// misdecode it. Call BEFORE dispatching on `plq.mode` in every per-layer
/// builder, paired with the int8/fp8 storage guards.
pub fn ensure_kquant_storage_resolves_kquant(
    params: &HashMap<String, MxArray>,
    base: &str,
    mode: PerLayerMode,
    family: &str,
) -> Result<()> {
    if is_kquant_mode(mode) {
        return Ok(());
    }
    let Some(scales) = params.get(&format!("{base}.scales")) else {
        return Ok(());
    };
    let scales_dtype = scales.dtype().ok();
    let biases_dtype = params
        .get(&format!("{base}.biases"))
        .and_then(|b| b.dtype().ok());
    let looks_q6k = scales_dtype == Some(DType::Int8);
    let looks_q4k_or_q5k =
        scales_dtype == Some(DType::Uint8) && biases_dtype == Some(DType::Float16);
    if looks_q6k || looks_q4k_or_q5k {
        return Err(Error::from_reason(format!(
            "{family}: '{base}' carries K-quant storage ('{base}.scales' is {scales_dtype:?} with \
             a float16 .biases super-block scale) but its per-layer quant mode resolves to \
             {mode:?} — config drift / stale quantization metadata, refusing to load"
        )));
    }
    Ok(())
}

/// Fail-loud guard for an affine group whose `.biases` companion is missing.
///
/// MLX affine decode is `scale * q + bias` and its `validate_mode_with_type`
/// throws "Biases must be provided for affine quantization" on a null bias, so
/// an affine `.scales` with no `.biases` is never loadable — but the builders
/// take the companion as an `Option` and would carry the `None` all the way to
/// that generic C++ throw, long after the model appeared to load.
///
/// The reachable cause is a checkpoint written by a build that omits the derived
/// companion for symmetric groups (see [`SYMMETRIC_ZERO_POINT_KEY`]) being read
/// through a path that never ran
/// [`expand_symmetric_affine_biases`](crate::engine::persistence::expand_symmetric_affine_biases).
/// Naming the tensor at load beats decoding garbage or throwing anonymously
/// later. Call BEFORE dispatching on `plq.mode`, beside the storage guards.
///
/// Scoped to a group that has both a `.weight` and a `.scales`, which is exactly
/// the shape a symmetric checkpoint has on disk. A `.scales` with no `.weight`
/// is an orphaned sidecar, not a truncated group, and the families already
/// report those by name — this must not preempt that with a vaguer message.
pub fn ensure_affine_biases_present(
    params: &HashMap<String, MxArray>,
    base: &str,
    mode: PerLayerMode,
    family: &str,
) -> Result<()> {
    if mode != PerLayerMode::Affine {
        return Ok(());
    }
    if !params.contains_key(&format!("{base}.weight"))
        || !params.contains_key(&format!("{base}.scales"))
    {
        return Ok(());
    }
    if params.contains_key(&format!("{base}.biases")) {
        return Ok(());
    }
    Err(Error::from_reason(format!(
        "{family}: affine group '{base}' has '{base}.scales' but no '{base}.biases' — affine \
         decode is `scale * q + bias` and cannot run without it. A checkpoint whose \
         config.json declares '{SYMMETRIC_ZERO_POINT_KEY}' stores the companion implicitly and \
         needs a build that rebuilds it at load; otherwise the checkpoint is truncated"
    )))
}

/// A validated ggml K-quant sidecar group ready to hand to `QuantizedLinear` /
/// `QuantizedSwitchLinear`. `biases` holds the float16 ggml `d` super-block
/// SCALE (not an additive bias); the K-quant kernel recombines it with the
/// integer `scales` sub-block codes.
pub struct KQuantGroup {
    pub weight: MxArray,
    pub scales: MxArray,
    pub biases: MxArray,
    pub bits: i32,
    pub group_size: i32,
    pub mode_str: &'static str,
}

/// The `(mode_str, bits, group_size, scales_dtype)` a resolved K-quant mode
/// demands. Mirrors the MLX FFI's `quantization_params_from_mode` +
/// `validate_mode_with_type` (`ops.cpp`) so the Rust load-time validation and
/// the C++ kernel contract cannot drift. Returns `None` for non-K-quant modes.
pub(crate) fn kquant_mode_params(mode: PerLayerMode) -> Option<(&'static str, i32, i32, DType)> {
    match mode {
        PerLayerMode::Q6K => Some(("q6k", 6, 16, DType::Int8)),
        PerLayerMode::Q4K => Some(("q4k", 4, 32, DType::Uint8)),
        PerLayerMode::Q5K => Some(("q5k", 5, 32, DType::Uint8)),
        _ => None,
    }
}

/// Resolve and validate a K-quant `.weight`/`.scales`/`.biases` group under
/// `key_prefix`, the fail-loud template shared by the dense and switch (expert)
/// K-quant builders across all LM families.
///
/// `Ok(None)` ONLY when `{key_prefix}.scales` is absent — the shared "this
/// projection is not quantized" signal (a mixed-precision UD GGUF legitimately
/// carries bf16 tensors with no sidecar). Every other shape is FAIL-LOUD `Err`:
/// a `.weight` or the mandatory `.biases` missing, a wrong companion dtype
/// (`weight` uint32; `scales` int8 for Q6_K / uint8 for Q4_K/Q5_K; `biases`
/// float16), or a rank mismatch (`expected_ndim` is 2 for dense projections, 3
/// for stacked experts). A silent fallback would decode garbage.
pub fn resolve_kquant_group(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
    mode: PerLayerMode,
    expected_ndim: usize,
    family: &str,
) -> Result<Option<KQuantGroup>> {
    let Some((mode_str, bits, group_size, scales_dtype)) = kquant_mode_params(mode) else {
        return Err(Error::from_reason(format!(
            "{family}: K-quant builder called for non-K-quant mode {mode:?} at '{key_prefix}'"
        )));
    };
    let Some(scales) = params.get(&format!("{key_prefix}.scales")) else {
        return Ok(None);
    };
    let Some(weight) = params.get(&format!("{key_prefix}.weight")) else {
        return Err(Error::from_reason(format!(
            "{family} {mode_str} layer '{key_prefix}': .scales present but .weight missing (corrupt checkpoint)"
        )));
    };
    let Some(biases) = params.get(&format!("{key_prefix}.biases")) else {
        return Err(Error::from_reason(format!(
            "{family} {mode_str} layer '{key_prefix}': .biases missing — K-quants store the ggml \
             `d` super-block scale in .biases and it is mandatory"
        )));
    };
    let w_dtype = weight.dtype()?;
    if w_dtype != DType::Uint32 {
        return Err(Error::from_reason(format!(
            "{family} {mode_str} layer '{key_prefix}': expected uint32 packed .weight, got {w_dtype:?}"
        )));
    }
    let s_dtype = scales.dtype()?;
    if s_dtype != scales_dtype {
        return Err(Error::from_reason(format!(
            "{family} {mode_str} layer '{key_prefix}': expected {scales_dtype:?} .scales, got {s_dtype:?}"
        )));
    }
    let b_dtype = biases.dtype()?;
    if b_dtype != DType::Float16 {
        return Err(Error::from_reason(format!(
            "{family} {mode_str} layer '{key_prefix}': expected float16 .biases (ggml `d`), got {b_dtype:?}"
        )));
    }
    let w_ndim = weight.shape()?.len();
    if w_ndim != expected_ndim {
        return Err(Error::from_reason(format!(
            "{family} {mode_str} layer '{key_prefix}': expected {expected_ndim}-D packed .weight, got {w_ndim}-D"
        )));
    }
    Ok(Some(KQuantGroup {
        weight: weight.clone(),
        scales: scales.clone(),
        biases: biases.clone(),
        bits,
        group_size,
        mode_str,
    }))
}

/// Build the fallback `PerLayerQuant` used when no per-layer override exists.
///
/// Honors the top-level `quantization.mode` (passed in as `default_mode`)
/// instead of inferring the mode purely from the checkpoint's scales dtype:
/// MXFP4 scales are also `uint8`, so the older `is_mxfp8` heuristic
/// mis-classifies MXFP4 layers as MXFP8 in mixed checkpoints (e.g. unsloth
/// recipe with some MXFP4 layers + some affine 3-bit layers).
pub fn default_per_layer_quant(
    bits: i32,
    group_size: i32,
    default_mode: PerLayerMode,
) -> PerLayerQuant {
    PerLayerQuant {
        bits,
        group_size,
        mode: default_mode,
        // Fallback default for layers without an explicit override never
        // carries a calibrated activation scale.
        input_amax: None,
    }
}

/// Resolve the default `PerLayerMode` for the fallback path.
///
/// Order of precedence (matches the original design intent):
///  1. Top-level `quantization.mode` (post-MXFP4 checkpoints all carry this).
///  2. The `is_mxfp8` heuristic — kept as a tertiary fallback for very old
///     pre-MXFP4 checkpoints where `config.json` has no `mode` field and
///     uint8 scales unambiguously meant MXFP8 at the time.
///  3. `Affine` otherwise.
pub fn resolve_default_mode(top_level_mode: Option<PerLayerMode>, is_mxfp8: bool) -> PerLayerMode {
    if let Some(m) = top_level_mode {
        return m;
    }
    if is_mxfp8 {
        PerLayerMode::Mxfp8
    } else {
        PerLayerMode::Affine
    }
}

/// Normalize a per-layer override key by stripping common HuggingFace prefixes.
///
/// All three model families (qwen3_5, qwen3_5_moe, gemma4) use the same set of
/// prefixes, so this helper delegates to the authoritative longest-first
/// [`strip_wrapper_prefix`](crate::models::mtp_drafter::strip_wrapper_prefix) —
/// keeping convert + load + quant in lockstep on the exact wrapper list and
/// order. A previous hand-rolled chain omitted the longest
/// `model.language_model.model.` variant, so a triple-wrapped per-layer override
/// key mis-normalized (the shorter `model.language_model.` fired first, leaving
/// `model.layers.*`).
pub fn normalize_per_layer_key(k: &str) -> String {
    crate::models::mtp_drafter::strip_wrapper_prefix(k).to_string()
}

fn parse_explicit_mode(value: Option<&Value>, context: &str) -> Result<Option<PerLayerMode>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let mode = value.as_str().ok_or_else(|| {
        Error::from_reason(format!(
            "Invalid quantization mode at {context}: expected a string, got {value}"
        ))
    })?;
    parse_mode_str(Some(mode)).map(Some).ok_or_else(|| {
        Error::from_reason(format!(
            "Unknown quantization mode '{mode}' at {context}; supported modes are affine, \
             mxfp4, mxfp8, nvfp4, fp8_e4m3, sym8, q6k, q4k, and q5k"
        ))
    })
}

fn quant_object(quant_cfg: Option<&Value>) -> Result<Option<&serde_json::Map<String, Value>>> {
    let Some(value) = quant_cfg else {
        return Ok(None);
    };
    value.as_object().map(Some).ok_or_else(|| {
        Error::from_reason(format!(
            "Invalid quantization metadata: expected quantization block to be an object, got {value}"
        ))
    })
}

/// Select the modern or legacy quantization block without silent alias
/// shadowing. Generated checkpoints duplicate both aliases byte-for-byte; a
/// null/non-object alias or divergent pair is malformed and must fail before
/// any dtype-based legacy mode heuristic runs.
pub fn select_quantization_block(raw: &Value) -> Result<Option<&Value>> {
    let modern = raw.get("quantization");
    let legacy = raw.get("quantization_config");
    for (name, value) in [("quantization", modern), ("quantization_config", legacy)] {
        if let Some(value) = value
            && !value.is_object()
        {
            return Err(Error::from_reason(format!(
                "Invalid {name} alias: expected an object, got {value}"
            )));
        }
    }
    match (modern, legacy) {
        (Some(modern), Some(legacy)) if modern != legacy => Err(Error::from_reason(
            "Conflicting quantization aliases: 'quantization' and 'quantization_config' must be identical when both are present"
                .to_string(),
        )),
        (Some(value), _) | (_, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}

fn parse_i32(value: &Value, context: &str) -> Result<i32> {
    let raw = value.as_i64().ok_or_else(|| {
        Error::from_reason(format!(
            "Invalid {context}: expected an integer, got {value}"
        ))
    })?;
    i32::try_from(raw).map_err(|_| {
        Error::from_reason(format!(
            "Invalid {context}: integer {raw} is outside the supported i32 range"
        ))
    })
}

fn parse_bits(value: &Value, mode: Option<PerLayerMode>, context: &str) -> Result<i32> {
    let bits = parse_i32(value, context)?;
    let valid = match mode {
        Some(PerLayerMode::Mxfp4) | Some(PerLayerMode::Nvfp4) | Some(PerLayerMode::Q4K) => {
            bits == 4
        }
        Some(PerLayerMode::Mxfp8) | Some(PerLayerMode::Fp8E4m3) | Some(PerLayerMode::Sym8) => {
            bits == 8
        }
        Some(PerLayerMode::Q6K) => bits == 6,
        Some(PerLayerMode::Q5K) => bits == 5,
        Some(PerLayerMode::Affine) | None => matches!(bits, 2 | 3 | 4 | 5 | 6 | 8),
    };
    if !valid {
        return Err(Error::from_reason(format!(
            "Invalid {context}={bits} for mode {mode:?}; affine supports bits 2, 3, 4, 5, 6, or 8, \
             while mxfp4/nvfp4/q4k require 4, mxfp8/fp8_e4m3/sym8 require 8, q6k requires 6, and q5k requires 5"
        )));
    }
    Ok(bits)
}

fn parse_group_size(value: &Value, mode: Option<PerLayerMode>, context: &str) -> Result<i32> {
    if value.is_null() {
        return match mode {
            Some(PerLayerMode::Fp8E4m3) => Ok(crate::quant::fp8_weight::FP8_E4M3_GROUP_SIZE),
            Some(PerLayerMode::Sym8) => Ok(SYM8_GROUP_SIZE_SENTINEL),
            _ => Err(Error::from_reason(format!(
                "Invalid {context}=null for mode {mode:?}; only fp8_e4m3 and sym8 have no quantization group"
            ))),
        };
    }

    let group_size = parse_i32(value, context)?;
    let valid = match mode {
        Some(PerLayerMode::Mxfp4) | Some(PerLayerMode::Mxfp8) => group_size == 32,
        Some(PerLayerMode::Nvfp4) => group_size == 16,
        Some(PerLayerMode::Q6K) => group_size == 16,
        Some(PerLayerMode::Q4K) | Some(PerLayerMode::Q5K) => group_size == 32,
        Some(PerLayerMode::Fp8E4m3) | Some(PerLayerMode::Sym8) => false,
        Some(PerLayerMode::Affine) | None => matches!(group_size, 32 | 64 | 128),
    };
    if !valid {
        return Err(Error::from_reason(format!(
            "Invalid {context}={group_size} for mode {mode:?}; affine supports 32, 64, or 128, \
             mxfp4/mxfp8/q4k/q5k require 32, nvfp4/q6k require 16, and fp8_e4m3/sym8 require null"
        )));
    }
    Ok(group_size)
}

fn parse_input_amax(
    value: Option<&Value>,
    mode: PerLayerMode,
    context: &str,
) -> Result<Option<f32>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if mode != PerLayerMode::Mxfp8 {
        return Err(Error::from_reason(format!(
            "Invalid {context}: input_amax is supported only for mxfp8 activation calibration, got mode {mode:?}"
        )));
    }
    let raw = value.as_f64().ok_or_else(|| {
        Error::from_reason(format!(
            "Invalid {context}: expected a positive finite number, got {value}"
        ))
    })?;
    let cast = raw as f32;
    if !raw.is_finite() || raw <= 0.0 || !cast.is_finite() || cast <= 0.0 {
        return Err(Error::from_reason(format!(
            "Invalid {context}: input_amax must be positive, finite, and representable as f32, got {raw}"
        )));
    }
    Ok(Some(cast))
}

/// Validate a `symmetric_zero_point` field and return the zero point it names.
///
/// Legal only on `mode: "affine"`, and only at the one value the algebra
/// permits: a `bits`-wide unsigned code centred at `1 << (bits - 1)`, which is 8
/// for ggml Q4_0 and 128 for Q8_0. Any other value would rebuild every bias in
/// the tensor at the wrong offset — silently, since the shapes still match — so
/// it is rejected rather than clamped.
fn parse_symmetric_zero_point(
    value: Option<&Value>,
    mode: PerLayerMode,
    bits: i32,
    context: &str,
) -> Result<Option<i32>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if mode != PerLayerMode::Affine {
        return Err(Error::from_reason(format!(
            "Invalid {context}: {SYMMETRIC_ZERO_POINT_KEY} describes an affine group whose \
             derived .biases companion was omitted, got mode {mode:?}"
        )));
    }
    let zero_point = parse_i32(value, context)?;
    let expected = 1i32 << (bits - 1);
    if zero_point != expected {
        return Err(Error::from_reason(format!(
            "Invalid {context}={zero_point} for {bits}-bit affine; a symmetric group subtracts \
             {expected}"
        )));
    }
    Ok(Some(zero_point))
}

/// The zero points a checkpoint's `quantization` block declares, resolved the
/// way the loaders resolve any other per-layer setting.
///
/// `per_layer` maps a [`normalize_per_layer_key`]-normalized tensor prefix to
/// its declared zero point; an entry present with value `None` means that
/// tensor has an override which is NOT symmetric, and must therefore shadow the
/// top-level `default` rather than inherit it.
#[derive(Debug, Default, Clone)]
pub struct SymmetricZeroPoints {
    pub default: Option<i32>,
    pub per_layer: HashMap<String, Option<i32>>,
}

impl SymmetricZeroPoints {
    /// The zero point in force for an already-normalized tensor prefix.
    pub fn for_key(&self, normalized_prefix: &str) -> Option<i32> {
        match self.per_layer.get(normalized_prefix) {
            Some(entry) => *entry,
            None => self.default,
        }
    }

    /// True when nothing in the block declares symmetry, so no companion needs
    /// rebuilding and the checkpoint behaves exactly as it did before the field
    /// existed.
    pub fn is_empty(&self) -> bool {
        self.default.is_none() && self.per_layer.values().all(Option::is_none)
    }
}

/// Read the `symmetric_zero_point` declarations out of a `quantization` block.
///
/// Shares [`parse_per_layer_entries`] with the ordinary settings parse, so the
/// two readers cannot disagree about which children are quantization entries or
/// which zero points are legal.
pub fn parse_symmetric_zero_points(
    quant_cfg: Option<&Value>,
    fallback_group_size: i32,
) -> Result<SymmetricZeroPoints> {
    let obj = quant_object(quant_cfg)?;
    let top_level_mode = parse_explicit_mode(
        obj.and_then(|q| q.get("mode")),
        "top-level quantization.mode",
    )?;
    let default = top_level_symmetric_zero_point(obj, top_level_mode)?;
    let (_, per_layer) = parse_per_layer_entries(obj, fallback_group_size)?;
    Ok(SymmetricZeroPoints { default, per_layer })
}

/// Validate the block-level `symmetric_zero_point`, which every tensor without
/// its own override inherits.
fn top_level_symmetric_zero_point(
    obj: Option<&serde_json::Map<String, Value>>,
    top_level_mode: Option<PerLayerMode>,
) -> Result<Option<i32>> {
    let Some(value) = obj.and_then(|q| q.get(SYMMETRIC_ZERO_POINT_KEY)) else {
        return Ok(None);
    };
    let bits_value = obj.and_then(|q| q.get("bits")).ok_or_else(|| {
        Error::from_reason(format!(
            "Invalid top-level quantization.{SYMMETRIC_ZERO_POINT_KEY}: a symmetric group must \
             also declare integer bits"
        ))
    })?;
    let bits = parse_bits(bits_value, top_level_mode, "top-level quantization.bits")?;
    parse_symmetric_zero_point(
        Some(value),
        top_level_mode.unwrap_or(PerLayerMode::Affine),
        bits,
        &format!("top-level quantization.{SYMMETRIC_ZERO_POINT_KEY}"),
    )
}

fn parse_per_layer_overrides(
    obj: Option<&serde_json::Map<String, Value>>,
    fallback_group_size: i32,
) -> Result<HashMap<String, PerLayerQuant>> {
    Ok(parse_per_layer_entries(obj, fallback_group_size)?.0)
}

/// Walk the per-tensor children of a `quantization` block once, yielding both
/// the `PerLayerQuant` overrides and the `symmetric_zero_point` declarations.
#[allow(clippy::type_complexity)]
fn parse_per_layer_entries(
    obj: Option<&serde_json::Map<String, Value>>,
    fallback_group_size: i32,
) -> Result<(HashMap<String, PerLayerQuant>, HashMap<String, Option<i32>>)> {
    let mut per_layer = HashMap::new();
    let mut zero_points = HashMap::new();
    let Some(obj) = obj else {
        return Ok((per_layer, zero_points));
    };

    for (key, value) in obj {
        let Some(child) = value.as_object() else {
            let normalized = normalize_per_layer_key(key);
            let looks_like_tensor_path = normalized.starts_with("layers.")
                || normalized.starts_with("mtp.")
                || normalized == "lm_head"
                || normalized == "embedding"
                || normalized.starts_with("embed_tokens")
                || key.ends_with(".weight");
            if looks_like_tensor_path {
                return Err(Error::from_reason(format!(
                    "Invalid per-layer quantization override '{key}': expected an object, got {value}"
                )));
            }
            continue;
        };
        let looks_quantized = child.contains_key("bits")
            || child.contains_key("group_size")
            || child.contains_key("mode")
            || child.contains_key("input_amax")
            || child.contains_key(SYMMETRIC_ZERO_POINT_KEY);
        if !looks_quantized {
            // Compatibility: unrelated nested metadata objects with no quant
            // schema fields remain outside the per-layer override map.
            continue;
        }

        let context = format!("per-layer quantization override '{key}'");
        let mode = parse_explicit_mode(child.get("mode"), &format!("{context}.mode"))?
            .unwrap_or(PerLayerMode::Affine);
        let bits_value = child.get("bits").ok_or_else(|| {
            Error::from_reason(format!(
                "Invalid {context}: an object containing mode/group_size must also contain integer bits"
            ))
        })?;
        let bits = parse_bits(bits_value, Some(mode), &format!("{context}.bits"))?;
        let group_size = match child.get("group_size") {
            Some(value) => parse_group_size(value, Some(mode), &format!("{context}.group_size"))?,
            None if matches!(mode, PerLayerMode::Mxfp4 | PerLayerMode::Mxfp8) => 32,
            None if mode == PerLayerMode::Nvfp4 => 16,
            None if mode == PerLayerMode::Q6K => 16,
            None if matches!(mode, PerLayerMode::Q4K | PerLayerMode::Q5K) => 32,
            None if mode == PerLayerMode::Fp8E4m3 => crate::quant::fp8_weight::FP8_E4M3_GROUP_SIZE,
            None if mode == PerLayerMode::Sym8 => SYM8_GROUP_SIZE_SENTINEL,
            None => {
                if !matches!(fallback_group_size, 32 | 64 | 128) {
                    return Err(Error::from_reason(format!(
                        "Invalid {context}: affine override omits group_size but would inherit \
                         unsupported top-level group_size {fallback_group_size}; specify 32, 64, or 128"
                    )));
                }
                fallback_group_size
            }
        };
        let input_amax = parse_input_amax(
            child.get("input_amax"),
            mode,
            &format!("{context}.input_amax"),
        )?;
        let zero_point = parse_symmetric_zero_point(
            child.get(SYMMETRIC_ZERO_POINT_KEY),
            mode,
            bits,
            &format!("{context}.{SYMMETRIC_ZERO_POINT_KEY}"),
        )?;
        let normalized = normalize_per_layer_key(key);
        zero_points.insert(normalized.clone(), zero_point);
        per_layer.insert(
            normalized,
            PerLayerQuant {
                bits,
                group_size,
                mode,
                input_amax,
            },
        );
    }
    Ok((per_layer, zero_points))
}

/// Parse the `quantization` (or legacy `quantization_config`) block from a
/// pre-loaded JSON value into a `(top_level_mode, per_layer_overrides)` pair.
///
/// `fallback_group_size` is used for per-layer entries that omit `group_size`
/// (it is the affine packing default for that family, e.g. 64). The
/// `top_level_mode` comes from `quantization.mode` and is what should drive
/// the fallback `PerLayerQuant` for layers without an explicit override; if
/// the field is missing, this returns `None` and the caller should fall back to
/// the legacy `is_mxfp8` heuristic. An explicitly present but unknown or
/// non-string mode is rejected; it must never silently select another decoder.
pub fn parse_quant_block(
    quant_cfg: Option<&Value>,
    fallback_group_size: i32,
) -> Result<(Option<PerLayerMode>, HashMap<String, PerLayerQuant>)> {
    let obj = quant_object(quant_cfg)?;
    let top_level_mode = parse_explicit_mode(
        obj.and_then(|q| q.get("mode")),
        "top-level quantization.mode",
    )?;
    if let Some(value) = obj.and_then(|q| q.get("bits")) {
        parse_bits(value, top_level_mode, "top-level quantization.bits")?;
    }
    if let Some(value) = obj.and_then(|q| q.get("group_size")) {
        parse_group_size(value, top_level_mode, "top-level quantization.group_size")?;
    }
    if obj.is_some_and(|q| q.contains_key("input_amax")) {
        return Err(Error::from_reason(
            "Invalid top-level quantization.input_amax: activation calibration is supported only on per-layer mxfp8 overrides"
                .to_string(),
        ));
    }
    top_level_symmetric_zero_point(obj, top_level_mode)?;
    let per_layer = parse_per_layer_overrides(obj, fallback_group_size)?;
    Ok((top_level_mode, per_layer))
}

/// Extract numeric defaults plus fallible mode metadata from an already
/// selected quantization block. This is shared by the in-memory Qwen loaders
/// and the disk-backed Gemma/LFM loaders so explicit-mode validation cannot
/// diverge by family.
pub fn parse_quant_settings(
    quant_cfg: Option<&Value>,
    default_bits: i32,
    default_group_size: i32,
) -> Result<(
    i32,
    i32,
    Option<PerLayerMode>,
    HashMap<String, PerLayerQuant>,
)> {
    let obj = quant_object(quant_cfg)?;
    let top_level_mode = parse_explicit_mode(
        obj.and_then(|q| q.get("mode")),
        "top-level quantization.mode",
    )?;
    let bits = match obj.and_then(|q| q.get("bits")) {
        Some(value) => parse_bits(value, top_level_mode, "top-level quantization.bits")?,
        None => default_bits,
    };
    let group_size = match obj.and_then(|q| q.get("group_size")) {
        Some(value) => {
            parse_group_size(value, top_level_mode, "top-level quantization.group_size")?
        }
        None => default_group_size,
    };
    if obj.is_some_and(|q| q.contains_key("input_amax")) {
        return Err(Error::from_reason(
            "Invalid top-level quantization.input_amax: activation calibration is supported only on per-layer mxfp8 overrides"
                .to_string(),
        ));
    }
    top_level_symmetric_zero_point(obj, top_level_mode)?;
    let per_layer = parse_per_layer_overrides(obj, group_size)?;
    Ok((bits, group_size, top_level_mode, per_layer))
}

/// Read `config.json` from `model_path` and return the parsed
/// `quantization` (or legacy `quantization_config`) block, along with the
/// extracted `(bits, group_size)` defaults used by the affine packing path.
///
/// Returns `(bits, group_size, top_level_mode, per_layer_overrides)`. When
/// `config.json` is missing/unreadable, returns the supplied
/// `default_bits` / `default_group_size` and empty overrides.
pub fn load_quant_settings_from_disk(
    model_path: &Path,
    default_bits: i32,
    default_group_size: i32,
) -> Result<(
    i32,
    i32,
    Option<PerLayerMode>,
    HashMap<String, PerLayerQuant>,
)> {
    let config_path = model_path.join("config.json");
    let Ok(raw_str) = std::fs::read_to_string(&config_path) else {
        return Ok((default_bits, default_group_size, None, HashMap::new()));
    };
    let Ok(raw) = serde_json::from_str::<Value>(&raw_str) else {
        return Ok((default_bits, default_group_size, None, HashMap::new()));
    };
    let quant_cfg = select_quantization_block(&raw)?;
    parse_quant_settings(quant_cfg, default_bits, default_group_size)
}

/// Resolve the effective `PerLayerQuant` for a sanitized projection prefix.
///
/// This single helper backs both Qwen3.5 dense and MoE persistence so the
/// Rust loaders and the C++ quant-info registry agree on the (mode, bits,
/// group_size) tuple for every quantized projection. Divergence here would
/// corrupt the compiled forward path.
///
/// Resolution order:
///
/// 1. Direct override at `per_layer_quant[prefix]`.
/// 2. The `embedding` prefix aliases to the historical Hugging Face key
///    `embed_tokens` — the Rust loaders' embedding branch consults
///    `per_layer_quant.get("embed_tokens")` even though the sanitized
///    tensor is renamed `embed_tokens.*` -> `embedding.*`. The alias
///    is probed FIRST so the C++ registry sees the same override the
///    loader applied; a direct-key lookup is kept as a defensive
///    fallback for any future config that emits the override under the
///    sanitized key.
/// 3. Merged GDN projections (`*.in_proj_qkvz`, `*.in_proj_ba`) consult
///    the split-side overrides via `merge_per_layer`.
/// 4. Gate-mode prefixes — only meaningful for MoE — fall back to
///    `gate_default` when present; pass `None` from dense callers.
/// 5. Everything else falls back to `default_plq`.
pub fn effective_plq_for(
    prefix: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
    gate_default: Option<PerLayerQuant>,
) -> PerLayerQuant {
    let fallback = match gate_default {
        Some(gp)
            if prefix.ends_with(".mlp.gate") || prefix.ends_with(".mlp.shared_expert_gate") =>
        {
            gp
        }
        _ => default_plq,
    };

    // Mirror the embedding-loader alias: the loaders look up
    // `per_layer_quant.get("embed_tokens")` because the sanitized tensor
    // is renamed `embed_tokens.*` -> `embedding.*`. The C++ registry keys
    // off the sanitized prefix, so we must alias here.
    let direct = if prefix == "embedding" {
        per_layer_quant
            .get("embed_tokens")
            .or_else(|| per_layer_quant.get(prefix))
    } else {
        per_layer_quant.get(prefix)
    };

    direct
        .copied()
        .or_else(|| {
            if let Some(base) = prefix.strip_suffix(".in_proj_qkvz") {
                let qkv = per_layer_quant.get(&format!("{}.in_proj_qkv", base));
                let z = per_layer_quant.get(&format!("{}.in_proj_z", base));
                merge_per_layer(qkv, z, "in_proj_qkvz", "qkv", "z")
            } else if let Some(base) = prefix.strip_suffix(".in_proj_ba") {
                let b_val = per_layer_quant.get(&format!("{}.in_proj_b", base));
                let a_val = per_layer_quant.get(&format!("{}.in_proj_a", base));
                merge_per_layer(b_val, a_val, "in_proj_ba", "b", "a")
            } else {
                None
            }
        })
        .unwrap_or(fallback)
}

/// Merge two per-layer overrides into one for a fused weight.
///
/// Used when the source checkpoint stores quantization metadata under split
/// keys (e.g. `in_proj_qkv` + `in_proj_z`) but our model expects the merged
/// projection (`in_proj_qkvz`). When the two sides disagree we pick the
/// higher-precision side: higher `bits` wins; on equal bits, prefer
/// `Affine` > `Fp8E4m3` > `Sym8` > `Mxfp8` > `Nvfp4` > `Mxfp4`. (Plain FP8
/// has a floating per-output scale and ranks below affine-8 but above sym8;
/// in practice convert emits the same mode on both GDN split sides.)
pub fn merge_per_layer(
    lhs: Option<&PerLayerQuant>,
    rhs: Option<&PerLayerQuant>,
    merged_label: &str,
    lhs_label: &str,
    rhs_label: &str,
) -> Option<PerLayerQuant> {
    fn mode_rank(m: PerLayerMode) -> u8 {
        match m {
            // K-quants are ggml-native lossless imports; rank them above the
            // requantized affine/MX/NV modes. `mode_rank` is only consulted on
            // EQUAL bits, and the two K-quant families never share a bit width
            // (6 / 5 / 4), so the ordering among them is nominal.
            PerLayerMode::Q6K => 8,
            PerLayerMode::Q5K => 7,
            PerLayerMode::Q4K => 6,
            PerLayerMode::Affine => 5,
            PerLayerMode::Fp8E4m3 => 4,
            PerLayerMode::Sym8 => 3,
            PerLayerMode::Mxfp8 => 2,
            PerLayerMode::Nvfp4 => 1,
            PerLayerMode::Mxfp4 => 0,
        }
    }
    fn pick(a: PerLayerQuant, b: PerLayerQuant) -> PerLayerQuant {
        if a.bits != b.bits {
            if a.bits > b.bits { a } else { b }
        } else if mode_rank(a.mode) >= mode_rank(b.mode) {
            a
        } else {
            b
        }
    }
    match (lhs, rhs) {
        (Some(&a), Some(&b)) if a != b => {
            warn!(
                "Merged {} has conflicting overrides: {}={:?}, {}={:?}. Using higher precision.",
                merged_label, lhs_label, a, rhs_label, b
            );
            Some(pick(a, b))
        }
        (Some(&a), _) | (_, Some(&a)) => Some(a),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `normalize_per_layer_key` delegates to the authoritative longest-first
    /// `strip_wrapper_prefix`, so per-layer override keys collapse to the same
    /// bare key at EVERY wrapper depth. The triple-wrap case is the regression:
    /// a prior hand-rolled chain omitted the longest
    /// `model.language_model.model.` variant → stripped only
    /// `model.language_model.`, leaving `model.layers.*`.
    #[test]
    fn normalize_per_layer_key_strips_all_wrapper_depths() {
        let bare = "layers.0.self_attn.q_proj";
        // Triple-wrap regression: was `model.layers.0...` before the fix.
        assert_eq!(
            normalize_per_layer_key("model.language_model.model.layers.0.self_attn.q_proj"),
            bare
        );
        // Double-wrap.
        assert_eq!(
            normalize_per_layer_key("model.language_model.layers.0.self_attn.q_proj"),
            bare
        );
        // `model.`-only.
        assert_eq!(
            normalize_per_layer_key("model.layers.0.self_attn.q_proj"),
            bare
        );
        // Already-bare.
        assert_eq!(normalize_per_layer_key("layers.0.self_attn.q_proj"), bare);
    }

    /// The per-layer parse lifts an optional `"input_amax"` (the calibrated
    /// FP8 activation scale) off each tensor's quantization override — present
    /// → `Some`, absent → `None`. Keys are stored under the normalized bare
    /// form, so we look them up through `normalize_per_layer_key`.
    #[test]
    fn parse_quant_block_reads_input_amax() {
        let cfg = serde_json::json!({
            "mode": "affine",
            "language_model.model.layers.0.self_attn.q_proj": {"bits":8,"group_size":32,"mode":"mxfp8","input_amax":37.5},
            "language_model.model.layers.0.self_attn.k_proj": {"bits":8,"group_size":32,"mode":"mxfp8"}
        });
        let (_mode, per_layer) = parse_quant_block(Some(&cfg), 64).unwrap();
        let q = normalize_per_layer_key("language_model.model.layers.0.self_attn.q_proj");
        let k = normalize_per_layer_key("language_model.model.layers.0.self_attn.k_proj");
        assert_eq!(per_layer[&q].input_amax, Some(37.5));
        assert_eq!(per_layer[&k].input_amax, None);
    }

    #[test]
    fn parse_mode_str_recognises_nvfp4() {
        assert_eq!(parse_mode_str(Some("nvfp4")), Some(PerLayerMode::Nvfp4));
        assert_eq!(
            parse_mode_str(Some("fp8_e4m3")),
            Some(PerLayerMode::Fp8E4m3)
        );
        assert_eq!(parse_mode_str(Some("mxfp4")), Some(PerLayerMode::Mxfp4));
        assert_eq!(parse_mode_str(Some("mxfp8")), Some(PerLayerMode::Mxfp8));
        assert_eq!(parse_mode_str(Some("affine")), Some(PerLayerMode::Affine));
        assert_eq!(parse_mode_str(Some("sym8")), Some(PerLayerMode::Sym8));
        assert_eq!(parse_mode_str(Some("bogus")), None);
        assert_eq!(parse_mode_str(None), None);
    }

    /// `mode_to_str` is the exact inverse of `parse_mode_str`. Nothing else
    /// pins the two tables together, so a mode added to one and not the other
    /// (or spelled differently) would only surface as a checkpoint that writes
    /// a string it cannot read back. Enumerating every variant also makes a new
    /// `PerLayerMode` a compile error here rather than a silent gap.
    #[test]
    fn mode_to_str_round_trips_through_parse_mode_str() {
        for mode in [
            PerLayerMode::Affine,
            PerLayerMode::Mxfp8,
            PerLayerMode::Mxfp4,
            PerLayerMode::Nvfp4,
            PerLayerMode::Fp8E4m3,
            PerLayerMode::Sym8,
            PerLayerMode::Q6K,
            PerLayerMode::Q4K,
            PerLayerMode::Q5K,
        ] {
            let encoded = mode_to_str(mode);
            assert_eq!(
                parse_mode_str(Some(encoded)),
                Some(mode),
                "mode {mode:?} encodes to '{encoded}' which parse_mode_str does not decode back"
            );
        }
    }

    /// The K-quant `(mode_str, bits, group_size, scales_dtype)` table is the
    /// single mirror of the MLX FFI contract, consumed by `resolve_kquant_group`
    /// and by gemma4's packed-embedding arm. Pin it, including that it returns
    /// `None` for every non-K-quant mode, and that its mode string agrees with
    /// `mode_to_str`.
    #[test]
    fn kquant_mode_params_pins_the_ffi_contract() {
        assert_eq!(
            kquant_mode_params(PerLayerMode::Q6K),
            Some(("q6k", 6, 16, DType::Int8))
        );
        assert_eq!(
            kquant_mode_params(PerLayerMode::Q4K),
            Some(("q4k", 4, 32, DType::Uint8))
        );
        assert_eq!(
            kquant_mode_params(PerLayerMode::Q5K),
            Some(("q5k", 5, 32, DType::Uint8))
        );
        for mode in [
            PerLayerMode::Affine,
            PerLayerMode::Mxfp8,
            PerLayerMode::Mxfp4,
            PerLayerMode::Nvfp4,
            PerLayerMode::Fp8E4m3,
            PerLayerMode::Sym8,
        ] {
            assert!(
                kquant_mode_params(mode).is_none(),
                "non-K-quant mode {mode:?} must have no K-quant FFI parameters"
            );
        }
        for mode in [PerLayerMode::Q6K, PerLayerMode::Q4K, PerLayerMode::Q5K] {
            assert_eq!(
                kquant_mode_params(mode).map(|(s, ..)| s),
                Some(mode_to_str(mode)),
                "K-quant FFI mode string must agree with mode_to_str for {mode:?}"
            );
        }
    }

    #[test]
    fn parse_plain_fp8_override_uses_no_group_sentinel() {
        let cfg = serde_json::json!({
            "mode": "nvfp4",
            "language_model.model.layers.0.self_attn.q_proj": {
                "bits": 8,
                "group_size": null,
                "mode": "fp8_e4m3"
            }
        });
        let (top, per_layer) = parse_quant_block(Some(&cfg), 16).unwrap();
        assert_eq!(top, Some(PerLayerMode::Nvfp4));
        let q = &per_layer["layers.0.self_attn.q_proj"];
        assert_eq!(q.mode, PerLayerMode::Fp8E4m3);
        assert_eq!(q.bits, crate::quant::fp8_weight::FP8_E4M3_BITS);
        assert_eq!(q.group_size, crate::quant::fp8_weight::FP8_E4M3_GROUP_SIZE);
    }

    #[test]
    fn plain_fp8_storage_guard_distinguishes_float_scales_from_mxfp8() {
        let weight = MxArray::from_uint8(&[0; 8], &[2, 4]).unwrap();
        let mx_scales = MxArray::from_uint8(&[127, 127], &[2, 1]).unwrap();
        let mx_params = HashMap::from([
            ("proj.weight".to_string(), weight.clone()),
            ("proj.scales".to_string(), mx_scales),
        ]);
        ensure_plain_fp8_storage_resolves_fp8_e4m3(&mx_params, "proj", PerLayerMode::Mxfp8, "test")
            .expect("Uint8 weight + Uint8 scales is MXFP storage, not plain FP8");

        let fp8_params = HashMap::from([
            ("proj.weight".to_string(), weight),
            (
                "proj.scales".to_string(),
                MxArray::from_float32(&[1.0, 1.0], &[2, 1])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap(),
            ),
        ]);
        assert!(
            ensure_plain_fp8_storage_resolves_fp8_e4m3(
                &fp8_params,
                "proj",
                PerLayerMode::Nvfp4,
                "test",
            )
            .is_err(),
            "plain FP8 storage with stale NVFP4 metadata must fail loud"
        );
        ensure_plain_fp8_storage_resolves_fp8_e4m3(
            &fp8_params,
            "proj",
            PerLayerMode::Fp8E4m3,
            "test",
        )
        .expect("plain FP8 storage with fp8_e4m3 metadata must pass");
    }

    #[test]
    fn absent_modes_keep_legacy_fallback_but_explicit_unknown_modes_reject() {
        let legacy = serde_json::json!({
            "bits": 8,
            "group_size": 32,
            "language_model.model.layers.0.self_attn.q_proj": {
                "bits": 4,
                "group_size": 64
            }
        });
        let (bits, group_size, top, per_layer) =
            parse_quant_settings(Some(&legacy), 4, 64).unwrap();
        assert_eq!((bits, group_size), (8, 32));
        assert_eq!(top, None, "missing top-level mode keeps legacy inference");
        assert_eq!(
            resolve_default_mode(top, /* is_mxfp8 */ true),
            PerLayerMode::Mxfp8,
            "legacy Uint8-scale heuristic remains available only when mode is absent"
        );
        assert_eq!(
            per_layer["layers.0.self_attn.q_proj"].mode,
            PerLayerMode::Affine,
            "legacy per-layer records without mode remain affine"
        );

        for typo in ["nvpf4", "fp8_e4m", "definitely_unknown"] {
            let top_typo = serde_json::json!({
                "bits": 4,
                "group_size": 16,
                "mode": typo
            });
            let err = parse_quant_settings(Some(&top_typo), 4, 64).unwrap_err();
            let message = err.reason.to_string();
            assert!(message.contains(typo), "error must name typo: {message}");
            assert!(
                message.contains("top-level"),
                "error must name scope: {message}"
            );

            let layer_typo = serde_json::json!({
                "bits": 4,
                "group_size": 16,
                "mode": "nvfp4",
                "language_model.model.layers.0.self_attn.q_proj": {
                    "bits": 8,
                    "group_size": null,
                    "mode": typo
                }
            });
            let err = parse_quant_settings(Some(&layer_typo), 4, 64).unwrap_err();
            let message = err.reason.to_string();
            assert!(message.contains(typo), "error must name typo: {message}");
            assert!(
                message.contains("per-layer"),
                "error must name scope: {message}"
            );
        }

        let non_string = serde_json::json!({"bits": 4, "group_size": 16, "mode": 7});
        assert!(parse_quant_settings(Some(&non_string), 4, 64).is_err());
    }

    #[test]
    fn malformed_mixed_quant_metadata_cannot_fall_through_to_mxfp8_heuristic() {
        let non_object = serde_json::json!("mxfp4");
        let err = parse_quant_settings(Some(&non_object), 4, 64).unwrap_err();
        assert!(
            err.reason
                .contains("expected quantization block to be an object")
        );

        let missing_bits = serde_json::json!({
            "bits": 8,
            "group_size": 32,
            "mode": "mxfp8",
            "language_model.model.layers.0.mlp.gate_proj": {
                "mode": "mxfp4",
                "group_size": 32
            }
        });
        let err = parse_quant_settings(Some(&missing_bits), 4, 64).unwrap_err();
        let message = err.reason.to_string();
        assert!(message.contains("gate_proj"), "{message}");
        assert!(message.contains("bits"), "{message}");

        let path_scalar = serde_json::json!({
            "bits": 8,
            "group_size": 32,
            "mode": "mxfp8",
            "language_model.model.layers.0.mlp.gate_proj": "mxfp4"
        });
        assert!(parse_quant_settings(Some(&path_scalar), 4, 64).is_err());

        // This is exactly the dangerous fallback malformed metadata used to
        // reach: Uint8 scales make the legacy heuristic choose MXFP8, which
        // would misdecode a skipped MXFP4 override. The parser errors above,
        // so callers never get this default.
        assert_eq!(
            resolve_default_mode(None, /* is_mxfp8_checkpoint */ true),
            PerLayerMode::Mxfp8
        );
    }

    #[test]
    fn quantization_alias_selector_rejects_null_shadow_and_divergent_duplicates() {
        let valid = serde_json::json!({"mode":"mxfp4", "bits":4, "group_size":32});
        let null_shadow = serde_json::json!({
            "quantization": null,
            "quantization_config": valid.clone()
        });
        let err = select_quantization_block(&null_shadow).unwrap_err();
        assert!(err.reason.contains("quantization"), "{}", err.reason);
        assert!(err.reason.contains("expected an object"), "{}", err.reason);

        let divergent = serde_json::json!({
            "quantization": valid.clone(),
            "quantization_config": {"mode":"mxfp8", "bits":8, "group_size":32}
        });
        let err = select_quantization_block(&divergent).unwrap_err();
        assert!(err.reason.contains("Conflicting"), "{}", err.reason);

        let equal = serde_json::json!({
            "quantization": valid.clone(),
            "quantization_config": valid.clone()
        });
        assert_eq!(select_quantization_block(&equal).unwrap(), Some(&valid));

        let legacy_only = serde_json::json!({"quantization_config": valid.clone()});
        assert_eq!(
            select_quantization_block(&legacy_only).unwrap(),
            legacy_only.get("quantization_config")
        );
    }

    #[test]
    fn present_numeric_fields_validate_types_ranges_and_mode_constants() {
        for malformed in [
            serde_json::json!({"mode":"affine", "bits":"4", "group_size":64}),
            serde_json::json!({"mode":"affine", "bits":7, "group_size":64}),
            serde_json::json!({"mode":"affine", "bits":4, "group_size":0}),
            serde_json::json!({"mode":"mxfp4", "bits":8, "group_size":32}),
            serde_json::json!({"mode":"mxfp8", "bits":8, "group_size":64}),
            serde_json::json!({"mode":"nvfp4", "bits":4, "group_size":32}),
            serde_json::json!({"mode":"fp8_e4m3", "bits":8, "group_size":32}),
            serde_json::json!({"mode":"sym8", "bits":8, "group_size":64}),
            serde_json::json!({"mode":"affine", "bits":2147483648_i64, "group_size":64}),
        ] {
            assert!(
                parse_quant_settings(Some(&malformed), 4, 64).is_err(),
                "malformed metadata must reject: {malformed}"
            );
        }

        let valid_ungrouped = serde_json::json!({
            "mode": "sym8",
            "bits": 8,
            "group_size": null,
            "layers.0.self_attn.q_proj": {
                "mode": "fp8_e4m3",
                "bits": 8,
                "group_size": null
            },
            "unrelated_metadata": {"producer": "legacy-tool"}
        });
        let (bits, group_size, mode, per_layer) =
            parse_quant_settings(Some(&valid_ungrouped), 4, 64).unwrap();
        assert_eq!(bits, 8);
        assert_eq!(group_size, SYM8_GROUP_SIZE_SENTINEL);
        assert_eq!(mode, Some(PerLayerMode::Sym8));
        assert_eq!(
            per_layer["layers.0.self_attn.q_proj"].group_size,
            crate::quant::fp8_weight::FP8_E4M3_GROUP_SIZE
        );
        assert_eq!(per_layer.len(), 1, "unrelated nested metadata is ignored");
    }

    #[test]
    fn affine_override_cannot_inherit_non_affine_group_or_ungrouped_sentinel() {
        for raw in [
            serde_json::json!({
                "mode": "nvfp4",
                "bits": 4,
                "group_size": 16,
                "layers.0.mlp.gate_proj": {"mode":"affine", "bits":4}
            }),
            serde_json::json!({
                "mode": "sym8",
                "bits": 8,
                "group_size": null,
                "layers.0.mlp.gate_proj": {"mode":"affine", "bits":4}
            }),
        ] {
            let err = parse_quant_settings(Some(&raw), 4, 64).unwrap_err();
            let message = err.reason.to_string();
            assert!(message.contains("gate_proj"), "{message}");
            assert!(message.contains("group_size"), "{message}");
        }

        let explicit = serde_json::json!({
            "mode": "nvfp4",
            "bits": 4,
            "group_size": 16,
            "layers.0.mlp.gate_proj": {
                "mode":"affine", "bits":4, "group_size":64
            }
        });
        let (_, _, _, overrides) = parse_quant_settings(Some(&explicit), 4, 64).unwrap();
        assert_eq!(overrides["layers.0.mlp.gate_proj"].group_size, 64);
    }

    #[test]
    fn input_amax_requires_positive_finite_f32_mxfp8_override() {
        let valid = serde_json::json!({
            "mode": "mxfp8",
            "bits": 8,
            "group_size": 32,
            "layers.0.self_attn.q_proj": {
                "mode": "mxfp8",
                "bits": 8,
                "group_size": 32,
                "input_amax": 12.5
            }
        });
        let (_, _, _, overrides) = parse_quant_settings(Some(&valid), 4, 64).unwrap();
        assert_eq!(
            overrides["layers.0.self_attn.q_proj"].input_amax,
            Some(12.5)
        );

        for bad in [
            serde_json::json!(0.0),
            serde_json::json!(-1.0),
            serde_json::json!("12.5"),
            serde_json::json!(1.0e300),
        ] {
            let raw = serde_json::json!({
                "mode": "mxfp8",
                "bits": 8,
                "group_size": 32,
                "layers.0.self_attn.q_proj": {
                    "mode": "mxfp8",
                    "bits": 8,
                    "group_size": 32,
                    "input_amax": bad
                }
            });
            assert!(parse_quant_settings(Some(&raw), 4, 64).is_err());
        }

        let wrong_mode = serde_json::json!({
            "mode": "affine",
            "bits": 4,
            "group_size": 64,
            "layers.0.mlp.gate_proj": {
                "mode": "affine",
                "bits": 4,
                "group_size": 64,
                "input_amax": 1.0
            }
        });
        assert!(parse_quant_settings(Some(&wrong_mode), 4, 64).is_err());

        let top_level = serde_json::json!({
            "mode": "mxfp8",
            "bits": 8,
            "group_size": 32,
            "input_amax": 1.0
        });
        assert!(parse_quant_settings(Some(&top_level), 4, 64).is_err());
    }

    /// Gemma4 and LFM2 both consume quantization metadata through the
    /// disk-backed helper. Preserve the same fail-closed behavior there (for
    /// both modern and legacy config aliases), instead of only exercising the
    /// in-memory Qwen parser above.
    #[test]
    fn disk_backed_gemma_lfm_quant_settings_reject_explicit_unknown_modes() {
        use std::sync::atomic::{AtomicU64, Ordering};

        static COUNTER: AtomicU64 = AtomicU64::new(0);
        for (alias, typo) in [
            ("quantization", "nvpf4"),
            ("quantization_config", "fp8_e4m"),
        ] {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "mlx_quant_dispatch_unknown_mode_{}_{}",
                std::process::id(),
                id
            ));
            std::fs::create_dir_all(&dir).expect("create quant settings temp directory");
            let raw = serde_json::json!({
                alias: {
                    "bits": 4,
                    "group_size": 16,
                    "mode": typo
                }
            });
            std::fs::write(
                dir.join("config.json"),
                serde_json::to_vec(&raw).expect("serialize quantization config"),
            )
            .expect("write quantization config");

            let err = load_quant_settings_from_disk(&dir, 4, 64).unwrap_err();
            let message = err.reason.to_string();
            assert!(message.contains(typo), "error must name typo: {message}");
            assert!(
                message.contains("top-level"),
                "error must name scope: {message}"
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// A sym8-default checkpoint carries complete affine override entries for
    /// forced-affine tensors (routers/gates, 3-D experts, K%16!=0 linears,
    /// affine-only keys like lm_head). The override must WIN over the sym8
    /// default so those layers build through the affine path.
    #[test]
    fn sym8_top_level_with_affine_per_layer_override_dispatches_affine() {
        let quant_cfg = serde_json::json!({
            "group_size": null,
            "bits": 8,
            "mode": "sym8",
            "language_model.model.lm_head": {
                "bits": 8,
                "group_size": 64,
                "mode": "affine"
            }
        });
        let (top_level_mode, per_layer) = parse_quant_block(Some(&quant_cfg), 64).unwrap();
        assert_eq!(top_level_mode, Some(PerLayerMode::Sym8));
        assert!(has_sym8_mode(top_level_mode, &per_layer));

        let default_plq = default_per_layer_quant(
            8,
            64,
            resolve_default_mode(top_level_mode, /* is_mxfp8 */ false),
        );
        assert_eq!(default_plq.mode, PerLayerMode::Sym8);

        // The forced-affine override wins for its prefix (key normalized
        // from the `language_model.model.` wrapper)…
        let lm_head = effective_plq_for("lm_head", &per_layer, default_plq, None);
        assert_eq!(lm_head.mode, PerLayerMode::Affine);
        assert_eq!((lm_head.bits, lm_head.group_size), (8, 64));

        // …while un-overridden projections stay on the sym8 default.
        let proj = effective_plq_for("layers.0.mlp.up_proj", &per_layer, default_plq, None);
        assert_eq!(proj.mode, PerLayerMode::Sym8);
    }

    #[test]
    fn has_sym8_mode_detects_top_level_and_per_layer() {
        let empty: HashMap<String, PerLayerQuant> = HashMap::new();
        // Top-level sym8.
        assert!(has_sym8_mode(Some(PerLayerMode::Sym8), &empty));
        // No sym8 anywhere.
        assert!(!has_sym8_mode(Some(PerLayerMode::Affine), &empty));
        assert!(!has_sym8_mode(None, &empty));
        // Per-layer sym8 override under a non-sym8 default.
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        overrides.insert(
            "layers.0.mlp.up_proj".into(),
            PerLayerQuant {
                bits: 8,
                group_size: 64,
                mode: PerLayerMode::Sym8,
                input_amax: None,
            },
        );
        assert!(has_sym8_mode(None, &overrides));
        assert!(has_sym8_mode(Some(PerLayerMode::Affine), &overrides));
    }

    /// Minimal well-typed K-quant sidecar group: uint32 `.weight`, int8 (Q6_K)
    /// or uint8 (Q4_K/Q5_K) `.scales`, and float16 `.biases` (the ggml `d`
    /// super-block scale). Shapes are token-sized — the resolver only checks
    /// dtypes and rank, not the packing arithmetic (that is the FFI's job).
    fn kquant_group_params(prefix: &str, mode: PerLayerMode) -> HashMap<String, MxArray> {
        let scales = match mode {
            PerLayerMode::Q6K => MxArray::from_float32(&[1.0, -1.0], &[1, 2])
                .unwrap()
                .astype(DType::Int8)
                .unwrap(),
            _ => MxArray::from_float32(&[1.0, 2.0], &[1, 2])
                .unwrap()
                .astype(DType::Uint8)
                .unwrap(),
        };
        let biases = MxArray::from_float16(&[half::f16::from_f32(0.5).to_bits()], &[1, 1]).unwrap();
        let weight = MxArray::from_uint32(&[0u32; 4], &[1, 4]).unwrap();
        HashMap::from([
            (format!("{prefix}.weight"), weight),
            (format!("{prefix}.scales"), scales),
            (format!("{prefix}.biases"), biases),
        ])
    }

    #[test]
    fn parse_mode_str_recognises_kquant() {
        assert_eq!(parse_mode_str(Some("q6k")), Some(PerLayerMode::Q6K));
        assert_eq!(parse_mode_str(Some("q4k")), Some(PerLayerMode::Q4K));
        assert_eq!(parse_mode_str(Some("q5k")), Some(PerLayerMode::Q5K));
        assert!(is_kquant_mode(PerLayerMode::Q6K));
        assert!(is_kquant_mode(PerLayerMode::Q4K));
        assert!(is_kquant_mode(PerLayerMode::Q5K));
        assert!(!is_kquant_mode(PerLayerMode::Affine));
    }

    /// The fail-closed `(bits, group_size)` tables accept exactly each K-quant's
    /// width — the W4 gate: a naive `32 / bits` derivation is wrong for
    /// `32/5` and `32/6`, so the parser MUST key on the mode, not arithmetic.
    #[test]
    fn kquant_bits_and_group_size_validate_per_mode() {
        for (m, bits, gs) in [("q6k", 6, 16), ("q4k", 4, 32), ("q5k", 5, 32)] {
            let cfg = serde_json::json!({"mode": m, "bits": bits, "group_size": gs});
            let (b, g, mode, _) = parse_quant_settings(Some(&cfg), 4, 64).unwrap();
            assert_eq!((b, g), (bits, gs), "{m} defaults");
            assert_eq!(mode, parse_mode_str(Some(m)));
        }
        for bad in [
            serde_json::json!({"mode":"q6k","bits":4,"group_size":16}),
            serde_json::json!({"mode":"q4k","bits":6,"group_size":32}),
            serde_json::json!({"mode":"q5k","bits":4,"group_size":32}),
            serde_json::json!({"mode":"q6k","bits":6,"group_size":32}),
            serde_json::json!({"mode":"q4k","bits":4,"group_size":16}),
            serde_json::json!({"mode":"q6k","bits":6,"group_size":null}),
        ] {
            assert!(
                parse_quant_settings(Some(&bad), 4, 64).is_err(),
                "must reject: {bad}"
            );
        }
    }

    #[test]
    fn has_kquant_mode_detects_top_level_and_per_layer() {
        let empty: HashMap<String, PerLayerQuant> = HashMap::new();
        assert!(has_kquant_mode(Some(PerLayerMode::Q6K), &empty));
        assert!(!has_kquant_mode(Some(PerLayerMode::Affine), &empty));
        assert!(!has_kquant_mode(None, &empty));
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        overrides.insert(
            "layers.0.mlp.up_proj".into(),
            PerLayerQuant {
                bits: 4,
                group_size: 32,
                mode: PerLayerMode::Q4K,
                input_amax: None,
            },
        );
        assert!(has_kquant_mode(None, &overrides));
        assert!(has_kquant_mode(Some(PerLayerMode::Affine), &overrides));
        assert!(!has_sym8_mode(None, &overrides));
    }

    #[test]
    fn resolve_kquant_group_fail_loud_contract() {
        // Absent .scales → Ok(None): a bf16 fallback tensor in a mixed GGUF.
        let mut p = kquant_group_params("l", PerLayerMode::Q6K);
        p.remove("l.scales");
        assert!(matches!(
            resolve_kquant_group(&p, "l", PerLayerMode::Q6K, 2, "t"),
            Ok(None)
        ));

        // Well-formed → Some carrying the mode's fixed bits/group/mode string.
        let p = kquant_group_params("l", PerLayerMode::Q4K);
        let g = resolve_kquant_group(&p, "l", PerLayerMode::Q4K, 2, "t")
            .unwrap()
            .unwrap();
        assert_eq!((g.bits, g.group_size, g.mode_str), (4, 32, "q4k"));

        // .scales present but .weight missing → Err.
        let mut p = kquant_group_params("l", PerLayerMode::Q6K);
        p.remove("l.weight");
        assert!(resolve_kquant_group(&p, "l", PerLayerMode::Q6K, 2, "t").is_err());

        // Mandatory .biases missing → Err (K-quants always carry `d`).
        let mut p = kquant_group_params("l", PerLayerMode::Q6K);
        p.remove("l.biases");
        assert!(resolve_kquant_group(&p, "l", PerLayerMode::Q6K, 2, "t").is_err());

        // Wrong scales dtype for the mode (q6k wants int8; hand it uint8) → Err.
        let p = kquant_group_params("l", PerLayerMode::Q4K);
        assert!(resolve_kquant_group(&p, "l", PerLayerMode::Q6K, 2, "t").is_err());

        // Wrong rank (dense expects 2-D, ask for 3-D) → Err.
        let p = kquant_group_params("l", PerLayerMode::Q6K);
        assert!(resolve_kquant_group(&p, "l", PerLayerMode::Q6K, 3, "t").is_err());

        // Non-K-quant mode into the K-quant resolver → Err.
        let p = kquant_group_params("l", PerLayerMode::Q6K);
        assert!(resolve_kquant_group(&p, "l", PerLayerMode::Affine, 2, "t").is_err());
    }

    #[test]
    fn ensure_kquant_storage_guard_catches_skewed_metadata() {
        // Q6_K storage (int8 scales + f16 biases) under a resolved affine mode.
        let p = kquant_group_params("proj", PerLayerMode::Q6K);
        assert!(
            ensure_kquant_storage_resolves_kquant(&p, "proj", PerLayerMode::Affine, "t").is_err()
        );
        // Same storage under the correct K mode passes.
        ensure_kquant_storage_resolves_kquant(&p, "proj", PerLayerMode::Q6K, "t").unwrap();

        // Q4_K storage (uint8 scales + f16 biases) under mxfp8 (which has no
        // biases companion) is skewed → Err.
        let p = kquant_group_params("proj", PerLayerMode::Q4K);
        assert!(
            ensure_kquant_storage_resolves_kquant(&p, "proj", PerLayerMode::Mxfp8, "t").is_err()
        );
        // A genuine mxfp8 group (uint8 scales, NO biases) is NOT K storage → Ok.
        let mut mx = kquant_group_params("proj", PerLayerMode::Q4K);
        mx.remove("proj.biases");
        ensure_kquant_storage_resolves_kquant(&mx, "proj", PerLayerMode::Mxfp8, "t").unwrap();
    }

    /// An affine group in the shape a symmetric checkpoint has on disk:
    /// float16 scales, no `.biases`.
    fn affine_group_without_bias(prefix: &str) -> HashMap<String, MxArray> {
        let scales =
            MxArray::from_float16(&[half::f16::from_f32(0.125).to_bits(); 2], &[1, 2]).unwrap();
        HashMap::from([
            (
                format!("{prefix}.weight"),
                MxArray::from_uint32(&[0u32; 4], &[1, 4]).unwrap(),
            ),
            (format!("{prefix}.scales"), scales),
        ])
    }

    #[test]
    fn ensure_affine_biases_present_names_the_truncated_tensor() {
        // The reachable cause: a checkpoint whose config declares a symmetric
        // zero point, read through a path that never rebuilt the companion.
        // Without this the `None` rides all the way to MLX's anonymous
        // "Biases must be provided for affine quantization" throw.
        let p = affine_group_without_bias("layers.0.mlp.down_proj");
        let err =
            ensure_affine_biases_present(&p, "layers.0.mlp.down_proj", PerLayerMode::Affine, "t")
                .expect_err("an affine group with no .biases must not load");
        assert!(
            err.reason.contains("layers.0.mlp.down_proj")
                && err.reason.contains(SYMMETRIC_ZERO_POINT_KEY),
            "the error must name the tensor and the field: {}",
            err.reason
        );

        // The same group once the companion is present → Ok.
        let mut ok = affine_group_without_bias("layers.0.mlp.down_proj");
        ok.insert(
            "layers.0.mlp.down_proj.biases".to_string(),
            MxArray::from_float16(&[half::f16::from_f32(-1.0).to_bits(); 2], &[1, 2]).unwrap(),
        );
        ensure_affine_biases_present(&ok, "layers.0.mlp.down_proj", PerLayerMode::Affine, "t")
            .unwrap();
    }

    #[test]
    fn ensure_affine_biases_present_leaves_the_biasless_modes_alone() {
        // Weakening this guard into "a missing bias means synthesise one" would
        // hand a fabricated companion to every one-companion mode, so it must
        // stay silent for all of them and for an unquantized prefix.
        let p = affine_group_without_bias("proj");
        for mode in [
            PerLayerMode::Mxfp4,
            PerLayerMode::Mxfp8,
            PerLayerMode::Nvfp4,
            PerLayerMode::Fp8E4m3,
            PerLayerMode::Sym8,
            PerLayerMode::Q4K,
            PerLayerMode::Q5K,
            PerLayerMode::Q6K,
        ] {
            ensure_affine_biases_present(&p, "proj", mode, "t")
                .unwrap_or_else(|e| panic!("{mode:?} must not be judged by the affine rule: {e}"));
        }
        // No `.scales` at all is a dense tensor, not a truncated affine group.
        let dense = HashMap::from([(
            "proj.weight".to_string(),
            MxArray::from_uint32(&[0u32; 4], &[1, 4]).unwrap(),
        )]);
        ensure_affine_biases_present(&dense, "proj", PerLayerMode::Affine, "t").unwrap();

        // A `.scales` with no `.weight` is an orphaned sidecar. The families
        // name those precisely (gemma4's MLP orphan scan, qwen3_5_moe's dense
        // fallback dtype guard); this guard must stay out of the way so the
        // better message is the one the user sees.
        let mut orphan = affine_group_without_bias("proj");
        orphan.remove("proj.weight");
        ensure_affine_biases_present(&orphan, "proj", PerLayerMode::Affine, "t").unwrap();
    }

    #[test]
    fn symmetric_zero_point_is_rejected_outside_its_one_legal_shape() {
        let block = |v: Value| -> Result<()> { parse_quant_settings(Some(&v), 4, 32).map(|_| ()) };
        // The legal pair for each width.
        block(serde_json::json!({
            "bits": 4, "group_size": 32, "mode": "affine", SYMMETRIC_ZERO_POINT_KEY: 8,
        }))
        .expect("4-bit affine subtracts 8");
        block(serde_json::json!({
            "bits": 8, "group_size": 32, "mode": "affine", SYMMETRIC_ZERO_POINT_KEY: 128,
        }))
        .expect("8-bit affine subtracts 128");

        // Off-by-one, cross-width, and a non-affine mode all fail.
        for bad in [
            serde_json::json!({
                "bits": 4, "group_size": 32, "mode": "affine", SYMMETRIC_ZERO_POINT_KEY: 7,
            }),
            serde_json::json!({
                "bits": 4, "group_size": 32, "mode": "affine", SYMMETRIC_ZERO_POINT_KEY: 128,
            }),
            serde_json::json!({
                "bits": 8, "group_size": 32, "mode": "affine", SYMMETRIC_ZERO_POINT_KEY: 8,
            }),
            serde_json::json!({
                "bits": 4, "group_size": 32, "mode": "mxfp4", SYMMETRIC_ZERO_POINT_KEY: 8,
            }),
            serde_json::json!({
                "bits": 6, "group_size": 16, "mode": "q6k", SYMMETRIC_ZERO_POINT_KEY: 32,
            }),
            serde_json::json!({ "group_size": 32, "mode": "affine", SYMMETRIC_ZERO_POINT_KEY: 8 }),
        ] {
            let err = block(bad.clone()).expect_err(&format!("{bad} must be rejected"));
            assert!(
                err.reason.contains(SYMMETRIC_ZERO_POINT_KEY),
                "{bad}: unexpected message {}",
                err.reason
            );
        }

        // A per-tensor override is held to the same rule.
        let err = block(serde_json::json!({
            "bits": 4,
            "group_size": 32,
            "mode": "affine",
            "model.layers.0.mlp.down_proj": {
                "bits": 4, "group_size": 32, "mode": "affine", SYMMETRIC_ZERO_POINT_KEY: 4,
            },
        }))
        .expect_err("an override may not declare its own wrong zero point");
        assert!(
            err.reason.contains(SYMMETRIC_ZERO_POINT_KEY),
            "{}",
            err.reason
        );
    }

    #[test]
    fn symmetric_zero_points_shadow_the_default_per_tensor() {
        let quant = serde_json::json!({
            "bits": 4,
            "group_size": 32,
            "mode": "affine",
            SYMMETRIC_ZERO_POINT_KEY: 8,
            "language_model.model.embed_tokens": { "bits": 6, "group_size": 16, "mode": "q6k" },
            "language_model.model.layers.1.mlp.down_proj": {
                "bits": 4, "group_size": 32, "mode": "affine",
            },
        });
        let z = parse_symmetric_zero_points(Some(&quant), 32).expect("parse");
        assert!(!z.is_empty());
        // Unlisted tensors inherit the block default.
        assert_eq!(z.for_key("layers.0.self_attn.q_proj"), Some(8));
        // A listed tensor takes its own answer, symmetric or not — inheriting
        // here would collide with the q6k super-block `d` stored in `.biases`.
        assert_eq!(z.for_key("embed_tokens"), None);
        assert_eq!(z.for_key("layers.1.mlp.down_proj"), None);

        // No marker anywhere → nothing to rebuild.
        let plain = serde_json::json!({ "bits": 4, "group_size": 32, "mode": "affine" });
        assert!(
            parse_symmetric_zero_points(Some(&plain), 32)
                .expect("parse")
                .is_empty()
        );
    }

    #[test]
    fn default_per_layer_quant_for_nvfp4() {
        // Top-level mode nvfp4 with bits=4, group_size=16 should produce a
        // matching plq the loader can dispatch on directly.
        let plq = default_per_layer_quant(4, 16, PerLayerMode::Nvfp4);
        assert_eq!(plq.bits, 4);
        assert_eq!(plq.group_size, 16);
        assert_eq!(plq.mode, PerLayerMode::Nvfp4);
    }

    // ----- effective_plq_for ---------------------------------------------------
    //
    // Constructors below produce PLQs with distinct (bits, group_size, mode)
    // tuples so each test asserts on the exact override that should win — no
    // accidental collisions with the defaults defined by `effective_defaults()`.

    fn affine_plq(bits: i32, group_size: i32) -> PerLayerQuant {
        PerLayerQuant {
            bits,
            group_size,
            mode: PerLayerMode::Affine,
            input_amax: None,
        }
    }

    fn mxfp8_plq() -> PerLayerQuant {
        PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Mxfp8,
            input_amax: None,
        }
    }

    fn mxfp4_plq() -> PerLayerQuant {
        PerLayerQuant {
            bits: 4,
            group_size: 32,
            mode: PerLayerMode::Mxfp4,
            input_amax: None,
        }
    }

    /// Distinct defaults so we can tell which fallback path was taken.
    /// `default_plq` is Affine 4-bit / gs=64; `default_gate_plq` is Affine 8-bit / gs=64.
    fn effective_defaults() -> (PerLayerQuant, PerLayerQuant) {
        (affine_plq(4, 64), affine_plq(8, 64))
    }

    #[test]
    fn effective_plq_direct_override_hit() {
        let (default_plq, default_gate_plq) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        overrides.insert("layers.3.mlp.up_proj".into(), mxfp4_plq());

        let got = effective_plq_for(
            "layers.3.mlp.up_proj",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(got, mxfp4_plq());
    }

    #[test]
    fn effective_plq_no_override_plain_projection_uses_default() {
        let (default_plq, default_gate_plq) = effective_defaults();
        let overrides: HashMap<String, PerLayerQuant> = HashMap::new();

        let got = effective_plq_for(
            "layers.0.mlp.up_proj",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(got, default_plq);
        // Sanity: must NOT pick the gate default.
        assert_ne!(got, default_gate_plq);
    }

    #[test]
    fn effective_plq_gate_prefix_with_override_returns_override() {
        let (default_plq, default_gate_plq) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        overrides.insert("layers.0.mlp.gate".into(), mxfp8_plq());

        let got = effective_plq_for(
            "layers.0.mlp.gate",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(got, mxfp8_plq());
        // It is NOT the gate default — the override wins.
        assert_ne!(got, default_gate_plq);
    }

    #[test]
    fn effective_plq_gate_prefix_without_override_uses_gate_default() {
        let (default_plq, default_gate_plq) = effective_defaults();
        let overrides: HashMap<String, PerLayerQuant> = HashMap::new();

        let got_gate = effective_plq_for(
            "layers.0.mlp.gate",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(got_gate, default_gate_plq);
        assert_ne!(got_gate, default_plq);

        let got_shared = effective_plq_for(
            "layers.7.mlp.shared_expert_gate",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(got_shared, default_gate_plq);
        assert_ne!(got_shared, default_plq);
    }

    #[test]
    fn effective_plq_gate_prefix_with_no_gate_default_falls_back_to_default_plq() {
        // Dense callers pass `None` for `gate_default`. Even if the prefix
        // looks like a gate, there is no MoE-specific default so we must
        // fall back to `default_plq`.
        let (default_plq, default_gate_plq) = effective_defaults();
        let overrides: HashMap<String, PerLayerQuant> = HashMap::new();

        let got = effective_plq_for("layers.0.mlp.gate", &overrides, default_plq, None);
        assert_eq!(got, default_plq);
        // Sanity: with `None`, the gate default is unreachable.
        assert_ne!(got, default_gate_plq);
    }

    #[test]
    fn effective_plq_qkvz_merges_when_no_direct_override() {
        let (default_plq, default_gate_plq) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        let qkv = mxfp8_plq();
        let z = mxfp4_plq();
        overrides.insert("layers.0.in_proj_qkv".into(), qkv);
        overrides.insert("layers.0.in_proj_z".into(), z);

        let got = effective_plq_for(
            "layers.0.in_proj_qkvz",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        let expected = merge_per_layer(Some(&qkv), Some(&z), "in_proj_qkvz", "qkv", "z")
            .expect("merge_per_layer must yield Some when both sides are present");
        assert_eq!(got, expected);
        // Sanity: must NOT have fallen back to the default.
        assert_ne!(got, default_plq);
    }

    #[test]
    fn effective_plq_qkvz_merges_with_no_gate_default_for_dense_callers() {
        // Dense Qwen3.5 also has GDN merged projections but passes `None` for
        // `gate_default`. The merge logic must still run.
        let (default_plq, _) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        let qkv = mxfp8_plq();
        let z = mxfp4_plq();
        overrides.insert("layers.0.in_proj_qkv".into(), qkv);
        overrides.insert("layers.0.in_proj_z".into(), z);

        let got = effective_plq_for("layers.0.in_proj_qkvz", &overrides, default_plq, None);
        let expected = merge_per_layer(Some(&qkv), Some(&z), "in_proj_qkvz", "qkv", "z")
            .expect("merge_per_layer must yield Some when both sides are present");
        assert_eq!(got, expected);
    }

    #[test]
    fn effective_plq_qkvz_direct_override_beats_merge() {
        let (default_plq, default_gate_plq) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        let direct = affine_plq(6, 128);
        overrides.insert("layers.0.in_proj_qkvz".into(), direct);
        // Splits exist but the direct override must win.
        overrides.insert("layers.0.in_proj_qkv".into(), mxfp8_plq());
        overrides.insert("layers.0.in_proj_z".into(), mxfp4_plq());

        let got = effective_plq_for(
            "layers.0.in_proj_qkvz",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(got, direct);
    }

    #[test]
    fn effective_plq_ba_merges_when_no_direct_override() {
        let (default_plq, default_gate_plq) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        let b = mxfp8_plq();
        let a = mxfp4_plq();
        overrides.insert("layers.2.in_proj_b".into(), b);
        overrides.insert("layers.2.in_proj_a".into(), a);

        let got = effective_plq_for(
            "layers.2.in_proj_ba",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        let expected = merge_per_layer(Some(&b), Some(&a), "in_proj_ba", "b", "a")
            .expect("merge_per_layer must yield Some when both sides are present");
        assert_eq!(got, expected);
        assert_ne!(got, default_plq);
    }

    #[test]
    fn effective_plq_embedding_aliases_embed_tokens_override() {
        // The Rust loaders' embedding branch resolves its PLQ via
        // `per_layer_quant.get("embed_tokens")`; the C++ registry must
        // see the same override under the sanitized prefix `embedding`.
        let (default_plq, default_gate_plq) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        let override_plq = mxfp8_plq();
        overrides.insert("embed_tokens".into(), override_plq);

        let got = effective_plq_for("embedding", &overrides, default_plq, Some(default_gate_plq));
        assert_eq!(got, override_plq);
        // Sanity: must NOT have fallen back to the default.
        assert_ne!(got, default_plq);

        // Same expectation for dense callers (gate_default = None).
        let got_dense = effective_plq_for("embedding", &overrides, default_plq, None);
        assert_eq!(got_dense, override_plq);
    }

    #[test]
    fn effective_plq_embedding_with_no_override_uses_default() {
        // With no `embed_tokens` (or `embedding`) override, the helper must
        // return `default_plq`, not the gate default and not silently None.
        let (default_plq, default_gate_plq) = effective_defaults();
        let overrides: HashMap<String, PerLayerQuant> = HashMap::new();

        let got = effective_plq_for("embedding", &overrides, default_plq, Some(default_gate_plq));
        assert_eq!(got, default_plq);
        assert_ne!(got, default_gate_plq);
    }

    #[test]
    fn effective_plq_embedding_direct_key_also_honored() {
        // Defensive: if a future config emits the override under the
        // sanitized key directly, the helper must still pick it up.
        let (default_plq, default_gate_plq) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        let override_plq = affine_plq(6, 128);
        overrides.insert("embedding".into(), override_plq);

        let got = effective_plq_for("embedding", &overrides, default_plq, Some(default_gate_plq));
        assert_eq!(got, override_plq);
    }

    #[test]
    fn effective_plq_embedding_embed_tokens_wins_over_direct_when_both_present() {
        // The alias lookup is intentionally first (matches the loader's
        // historical lookup order). If both keys are present and conflict,
        // `embed_tokens` wins so the C++ side and Rust loader stay in sync.
        let (default_plq, default_gate_plq) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        let embed_tokens_plq = mxfp4_plq();
        let embedding_plq = affine_plq(6, 128);
        overrides.insert("embed_tokens".into(), embed_tokens_plq);
        overrides.insert("embedding".into(), embedding_plq);

        let got = effective_plq_for("embedding", &overrides, default_plq, Some(default_gate_plq));
        assert_eq!(got, embed_tokens_plq);
    }

    #[test]
    fn effective_plq_gate_proj_is_not_a_gate_prefix() {
        // `*.mlp.gate_proj` ends with `gate_proj`, NOT `gate`. The prefix must
        // be classified as a regular projection and fall back to `default_plq`
        // (not `default_gate_plq`).
        let (default_plq, default_gate_plq) = effective_defaults();
        let overrides: HashMap<String, PerLayerQuant> = HashMap::new();

        let got = effective_plq_for(
            "layers.0.mlp.gate_proj",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(got, default_plq);
        assert_ne!(got, default_gate_plq);
    }
}
