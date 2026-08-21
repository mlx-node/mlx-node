//! NemotronH checkpoint persistence: sanitize, quant dispatch, weight
//! application, and the load-with-thread entry point.
//!
//! Consumes the checkpoint format produced by the convert agent: keys are
//! the HF names with the backbone. wrapper stripped (layers.{i}.mixer.*,
//! embedding.weight, final_norm.weight, lm_head.weight, mtp.*),
//! per-expert projections stacked under mixer.experts.{up,down}_proj, and
//! quantized projections carrying MLX .weight/.scales companions.
//!
//! Quantization dispatch (per-layer, resolved from config.json):
//!   * NVFP4 - experts via try_build_nvfp4_quantized_switch_linear,
//!     shared_experts via try_build_nvfp4_quantized_linear. lm_head is NOT
//!     nvfp4 at runtime: it inherits the top-level nvfp4 mode from config but
//!     convert dequantizes it to dense BF16 (see
//!     `convert::dequant_nvfp4_to_bf16_exact`), so the builder finds no
//!     `.scales` and it takes the dense path. Every NVFP4 prefix additionally
//!     REQUIRES a Float32 `.global_scale` sidecar (`[n_routed_experts]` on the
//!     stacked experts, a scalar on the shared experts): convert emits
//!     NVIDIA's per-group E4M3 `weight_scale` bytes verbatim and carries
//!     `weight_scale_2` here instead of folding it. The lookup is FAIL-CLOSED
//!     — a missing/mis-typed/mis-shaped one is a hard error, because
//!     defaulting it to 1.0 would leave the projection ~1/weight_scale_2
//!     (~1.15e4x) too large with no diagnostic.
//!   * AFFINE 8-bit group-32 - mamba in_proj/out_proj via
//!     try_build_quantized_linear, with the layer's input_amax threaded for
//!     W8A8 numeric parity. NOT mxfp8: MLX's E8M0 block scale rounds
//!     log2(amax/448) to NEAREST and clips ~50% of 32-element groups, costing
//!     5-7% relative RMS on the sequence-mixing backbone (see
//!     convert::recipe::fp8_to_affine8). A projection that still declares
//!     mxfp8 is a pre-fix checkpoint and is REJECTED at load.
//!   * the MoE router pair (mixer.gate.weight,
//!     mixer.gate.e_score_correction_bias) stays at the source F32 - see
//!     convert::recipe::NemotronAction::Router. A BF16 bias there collapses
//!     all 128 experts onto one value and is rejected.
//!   * everything else dense bf16 (attention, conv1d, norms, A_log/D/
//!     dt_bias, embeddings, all mtp.*)

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use serde_json::Value;
use tracing::{info, warn};

use crate::array::{DType, MxArray};
use crate::engine::persistence::{
    load_all_safetensors, parse_generation_defaults, prewarm_checkpoint_pages,
};
use crate::models::quant_dispatch::{
    PerLayerMode, PerLayerQuant, admits_static_fp8_activation, default_per_layer_quant,
    effective_plq_for, parse_quant_block, select_quantization_block,
};
use crate::models::qwen3_5::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, is_quantized_checkpoint,
    try_build_nvfp4_quantized_linear, try_build_quantized_linear,
};
use crate::models::qwen3_5_moe::quantized_linear::{
    LinearProj, QuantizedLinear, QuantizedSwitchLinear, try_build_nvfp4_quantized_switch_linear,
};
use crate::tokenizer::Qwen3Tokenizer;

use super::config::{NemotronHConfig, parse_config};
use super::model::{NemotronHInner, NemotronHModel};
use super::sparse_moe::ExpertProj;

/// Strip the HF backbone. model wrapper from a tensor or quant key.
fn strip_backbone(k: &str) -> &str {
    k.strip_prefix("backbone.").unwrap_or(k)
}

/// Sanitize the checkpoint tensors into the loader's canonical key space:
/// strip backbone., rename embeddings/norm_f, stack per-expert
/// projections, drop modelopt companion keys (weight_scale /
/// weight_scale_2 / input_scale - consumed by convert), and transpose
/// conv1d weights to the MLX [out, kernel, in/groups] orientation.
fn sanitize_weights(
    mut params: HashMap<String, MxArray>,
    config: &NemotronHConfig,
) -> Result<HashMap<String, MxArray>> {
    let mut result: HashMap<String, MxArray> = HashMap::new();
    // Stacked per-expert projections: key -> (expert_idx, array).
    let mut expert_weights: HashMap<String, Vec<(usize, MxArray)>> = HashMap::new();

    for (name, array) in params.drain() {
        if name == "__metadata__" {
            continue;
        }
        let name = strip_backbone(&name).to_string();

        // Drop modelopt companion tensors (raw HF checkpoint layout).
        if name.ends_with(".weight_scale")
            || name.ends_with(".weight_scale_2")
            || name.ends_with(".input_scale")
        {
            continue;
        }

        // Renames.
        let name = if name == "embeddings.weight" {
            "embedding.weight".to_string()
        } else if name == "norm_f.weight" {
            "final_norm.weight".to_string()
        } else {
            name
        };

        // Stack per-expert projections: layers.{i}.mixer.experts.{j}.{proj}.
        if let Some(rest) = name.strip_prefix("layers.")
            && let Some(rest) = rest.strip_suffix(".weight")
            && let Some((layer, exp)) = rest.split_once(".mixer.experts.")
        {
            let parts: Vec<&str> = exp.split('.').collect();
            if parts.len() == 2 {
                let expert_idx: usize = parts[0].parse().map_err(|e| {
                    Error::from_reason(format!(
                        "Failed to parse expert index from weight '{}': {}",
                        name, e
                    ))
                })?;
                let proj = parts[1];
                let key = format!("layers.{}.mixer.experts.{}.weight", layer, proj);
                expert_weights
                    .entry(key)
                    .or_default()
                    .push((expert_idx, array));
                continue;
            }
        }
        // Stack the MTP experts: mtp.layers.{i}.mixer.experts.{j}.{proj}.
        if let Some(rest) = name.strip_prefix("mtp.layers.")
            && let Some(rest) = rest.strip_suffix(".weight")
            && let Some((mtp_layer, exp)) = rest.split_once(".mixer.experts.")
        {
            let parts: Vec<&str> = exp.split('.').collect();
            if parts.len() == 2 {
                let expert_idx: usize = parts[0].parse().map_err(|e| {
                    Error::from_reason(format!(
                        "Failed to parse MTP expert index from weight '{}': {}",
                        name, e
                    ))
                })?;
                let proj = parts[1];
                let key = format!("mtp.layers.{}.mixer.experts.{}.weight", mtp_layer, proj);
                expert_weights
                    .entry(key)
                    .or_default()
                    .push((expert_idx, array));
                continue;
            }
        }

        // conv1d weight: HF [out, in/groups, kernel] -> MLX [out, kernel, in/groups].
        let array = if name.contains(".mixer.conv1d.weight") {
            let shape = array.shape()?;
            if shape.len() == 3 && shape[2] != 1 {
                array.transpose(Some(&[0, 2, 1]))?
            } else {
                array
            }
        } else {
            array
        };

        result.insert(name, array);
    }

    // Stack per-expert weights.
    if !expert_weights.is_empty() {
        for (key, mut experts) in expert_weights {
            experts.sort_by_key(|(idx, _)| *idx);
            let expected = config.n_routed_experts as usize;
            if experts.len() != expected {
                return Err(Error::from_reason(format!(
                    "Expected {} experts for {}, got {}",
                    expected,
                    key,
                    experts.len()
                )));
            }
            let arrays: Vec<&MxArray> = experts.iter().map(|(_, a)| a).collect();
            let stacked = MxArray::stack(arrays, Some(0))?;
            result.insert(key, stacked);
        }
    }

    Ok(result)
}

/// Actionable tail for every `.global_scale` failure: the only way a
/// nemotron NVFP4 checkpoint lacks a well-formed one is that it came out of
/// the superseded folding ingest.
const REGENERATE_HINT: &str = "This checkpoint predates the NVFP4 global-scale split (convert used to fold weight_scale_2 into the per-group E4M3 scales, which costs ~8% relative error). Re-run `mlx convert -m nemotron_h` against the NVIDIA source to regenerate it.";

/// Regeneration hint for the quantization-accuracy pass: the mamba
/// `mixer.{in,out}_proj` moved off MLX's mxfp8 grid onto affine 8-bit
/// group-32 (mxfp8's E8M0 block scale rounds to NEAREST and clips ~50% of
/// groups, costing 5-7% relative RMS), and the MoE router pair is now retained
/// at the source F32 instead of being rounded to BF16 (which collapsed all 128
/// `e_score_correction_bias` entries onto one value).
const REQUANT_REGENERATE_HINT: &str = "This checkpoint predates the Nemotron-H quantization-accuracy pass (mamba in_proj/out_proj mxfp8 -> affine 8-bit group-32, and F32 MoE router retention). Re-run `mlx convert -m nemotron_h` against the NVIDIA source to regenerate it.";

/// Fetch the MANDATORY `[E]` Float32 NVFP4 global scale for a stacked expert
/// projection.
///
/// Fail-closed by design: a missing key must never default to 1.0. The stored
/// weight is `code * decode_e4m3(weight_scale)`, i.e. `1/weight_scale_2`
/// (~1.15e4) times larger than the intended weight, so a silent default would
/// blow the projection up by four orders of magnitude with no diagnostic.
fn require_expert_global_scale(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    n_routed_experts: i64,
) -> Result<MxArray> {
    let key = format!("{prefix}.global_scale");
    let gs = params.get(&key).ok_or_else(|| {
        Error::from_reason(format!(
            "NVFP4 expert projection '{prefix}' is missing the mandatory '{key}' tensor. \
             {REGENERATE_HINT}"
        ))
    })?;
    let dtype = gs.dtype()?;
    if dtype != DType::Float32 {
        return Err(Error::from_reason(format!(
            "'{key}' must be Float32, got {dtype:?} — bf16 would round weight_scale_2 (~8.6e-5) \
             with ~0.4% error and silently re-introduce the scale bug. {REGENERATE_HINT}"
        )));
    }
    let shape = gs.shape()?.to_vec();
    if shape != vec![n_routed_experts] {
        return Err(Error::from_reason(format!(
            "'{key}' must have shape [{n_routed_experts}] (one weight_scale_2 per expert — it \
             VARIES per expert), got {shape:?}. {REGENERATE_HINT}"
        )));
    }
    Ok(gs.clone())
}

/// Fetch the MANDATORY per-tensor NVFP4 global scale for a 2-D projection
/// (the shared experts). Same fail-closed contract as the stacked variant.
/// Returns the validated 1-element Float32 TENSOR, not an `f64`: the runtime
/// must multiply by an f32 ARRAY so MLX promotes `bf16 * f32 -> f32`. An `f64`
/// scalar would go through `mlx_array_mul_scalar`, which builds the scalar in
/// the ARRAY's dtype (mlx-sys/src/mlx_array_ops.cpp:357-362) and truncates the
/// ~8.6e-5 scale to bf16 — the exact error this function's dtype check
/// rejects on disk.
fn require_scalar_global_scale(params: &HashMap<String, MxArray>, prefix: &str) -> Result<MxArray> {
    let key = format!("{prefix}.global_scale");
    let gs = params.get(&key).ok_or_else(|| {
        Error::from_reason(format!(
            "NVFP4 projection '{prefix}' is missing the mandatory '{key}' tensor. \
             {REGENERATE_HINT}"
        ))
    })?;
    let dtype = gs.dtype()?;
    if dtype != DType::Float32 {
        return Err(Error::from_reason(format!(
            "'{key}' must be Float32, got {dtype:?} — bf16 would round weight_scale_2 (~8.6e-5) \
             with ~0.4% error and silently re-introduce the scale bug. {REGENERATE_HINT}"
        )));
    }
    if gs.size()? != 1 {
        return Err(Error::from_reason(format!(
            "'{key}' must be a single per-tensor scalar, got shape {:?}. {REGENERATE_HINT}",
            gs.shape()?.as_ref()
        )));
    }
    let v = gs.item_at_float32(0)? as f64;
    if !v.is_finite() || v <= 0.0 {
        return Err(Error::from_reason(format!(
            "'{key}' must be finite and positive, got {v}. {REGENERATE_HINT}"
        )));
    }
    gs.reshape(&[1])
}

/// Does the RUNTIME have somewhere to apply a 2-D NVFP4 projection's
/// per-tensor `.global_scale`?
///
/// Only the shared experts do: `SparseMoeBlock::shared_experts` owns
/// `{up,down}_global_scale` and multiplies each projection's output by it
/// (sparse_moe.rs). `QuantizedLinear` itself has NO global-scale field, so any
/// OTHER prefix that reaches the nvfp4 arm of `try_build_ql` would build a
/// perfectly valid-looking projection whose weights are `1/weight_scale_2`
/// (~1.15e4x) too large, with no diagnostic — the exact failure the
/// `require_*_global_scale` guards exist to prevent.
///
/// That is not hypothetical. `lm_head` inherits the top-level `nvfp4` mode on
/// every converted checkpoint and enters that arm on every load; it only
/// declines to build because convert dequantizes it to BF16 and leaves no
/// `.scales`. And the mamba `mixer.{in,out}_proj` reached it for real once —
/// see `canonicalize_nemotron_per_layer`, written because their per-layer
/// affine overrides mis-normalized and fell back to the top-level nvfp4
/// default. Convert's own NVFP4 emit arm is not shared-expert-specific either
/// ("shared_experts (and any other non-expert NVFP4 group)"), so it will
/// happily write `.weight` + `.scales` + `.global_scale` for a future source
/// that quantizes some other 2-D projection.
fn nvfp4_global_scale_is_applied(prefix: &str) -> bool {
    prefix.ends_with(".mixer.shared_experts.up_proj")
        || prefix.ends_with(".mixer.shared_experts.down_proj")
}

/// Map a modelopt quant_algo string to a PerLayerMode (with fixed bits).
fn modelopt_mode(algo: &str, key: &str) -> Result<PerLayerQuant> {
    match algo {
        "W4A16_NVFP4" => Ok(PerLayerQuant {
            bits: 4,
            group_size: 16,
            mode: PerLayerMode::Nvfp4,
            input_amax: None,
        }),
        // NVIDIA's per-tensor E4M3 weights are ingested as affine 8-bit
        // group-32, not MLX mxfp8 — see convert::recipe::fp8_to_affine8.
        "FP8" | "W8A8_FP8" => Ok(PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Affine,
            input_amax: None,
        }),
        "MIXED_PRECISION" => Err(Error::from_reason(format!(
            "quantization override '{key}': 'MIXED_PRECISION' is the top-level container, not a layer mode"
        ))),
        other => Err(Error::from_reason(format!(
            "quantization override '{key}': unsupported modelopt quant_algo '{other}'"
        ))),
    }
}

/// Parse the checkpoint's quantization block into (top_level_mode,
/// per_layer overrides), supporting BOTH the mlx schema (mode/bits per
/// layer - produced by convert) and the raw NVIDIA modelopt schema
/// (quantized_layers with quant_algo strings - the HF checkpoint).
fn parse_nemotron_quant_settings(
    quant_cfg: Option<&Value>,
) -> Result<(Option<PerLayerMode>, HashMap<String, PerLayerQuant>)> {
    // mlx schema (mode/bits at top level) - delegate to the shared parser.
    if quant_cfg.is_some_and(|q| q.get("mode").is_some() || q.get("bits").is_some()) {
        let parsed = parse_quant_block(quant_cfg, DEFAULT_QUANT_GROUP_SIZE)?;
        return Ok(canonicalize_nemotron_per_layer(parsed));
    }
    // No quantization block at all.
    if quant_cfg.is_none() || !quant_cfg.is_some_and(|q| q.get("quantized_layers").is_some()) {
        return Ok((None, HashMap::new()));
    }

    let obj = quant_cfg
        .and_then(Value::as_object)
        .ok_or_else(|| Error::from_reason("Invalid quantization metadata: expected an object"))?;
    let layers_obj = obj
        .get("quantized_layers")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            Error::from_reason("modelopt quantization block missing 'quantized_layers' object")
        })?;

    let mut per_layer: HashMap<String, PerLayerQuant> = HashMap::new();
    let mut stacked_aliases: Vec<(String, PerLayerQuant)> = Vec::new();
    for (key, value) in layers_obj {
        let algo = value
            .get("quant_algo")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "quantization override '{key}': missing or invalid quant_algo"
                ))
            })?;
        let mut plq = modelopt_mode(algo, key)?;
        // Optional converted input_amax (activation calibration).
        if let Some(amax) = value.get("input_amax").and_then(Value::as_f64) {
            if !admits_static_fp8_activation(plq.mode, plq.bits, plq.group_size) {
                return Err(Error::from_reason(format!(
                    "quantization override '{key}': input_amax is only valid on static-FP8 layers (mxfp8 8/32 or affine 8/32)"
                )));
            }
            plq.input_amax = Some(amax as f32);
        }
        let normalized = strip_backbone(key).to_string();
        per_layer.insert(normalized.clone(), plq);
        // Register the stacked-expert alias from any single expert entry:
        // layers.{i}.mixer.experts.{j}.{proj} -> layers.{i}.mixer.experts.{proj}
        if let Some(rest) = normalized.strip_prefix("layers.")
            && let Some((layer, exp)) = rest.split_once(".mixer.experts.")
            && let Some((_, proj)) = exp.split_once('.')
        {
            let alias = format!("layers.{}.mixer.experts.{}", layer, proj);
            stacked_aliases.push((alias, plq));
        }
    }
    for (alias, plq) in stacked_aliases {
        per_layer.entry(alias).or_insert(plq);
    }
    // The modelopt block carries no top-level fallback mode.
    Ok(canonicalize_nemotron_per_layer((None, per_layer)))
}

/// Canonicalize per-layer quantization override keys into the sanitized
/// tensor space the loader looks up ("layers.{L}.mixer.{proj}").
///
/// The shared parser (`parse_quant_block`) strips HF wrapper prefixes
/// (`language_model.model.`) but not the Nemotron-specific `backbone.`
/// segment, which [`sanitize_weights`] removes from tensor keys. Without
/// this, converted checkpoints (whose mlx-schema override keys normalize to
/// `backbone.layers...`) silently miss every per-layer override and fall
/// back to the top-level nvfp4 default - which then tries to load the
/// mxfp8-packed mamba projections as nvfp4 and fails the quantized matmul
/// shape check. The modelopt path already strips `backbone.` per key, so
/// this is idempotent there.
fn canonicalize_nemotron_per_layer(
    (top_level, per_layer): (Option<PerLayerMode>, HashMap<String, PerLayerQuant>),
) -> (Option<PerLayerMode>, HashMap<String, PerLayerQuant>) {
    (
        top_level,
        per_layer
            .into_iter()
            .map(|(key, plq)| (strip_backbone(&key).to_string(), plq))
            .collect(),
    )
}

/// A Nemotron-H mamba projection declared `mxfp8` is a pre-affine checkpoint.
///
/// It would still LOAD and RUN — silently 5-7% relative RMS off on the whole
/// 890M-param sequence-mixing backbone, feeding a NON-INVERTIBLE Mamba-2
/// recurrence — because the mxfp8 payload is structurally valid for the old
/// builder. Reject it here, from the CONFIG, before a single tensor is read:
/// the on-disk `.weight` is byte-identical between the two modes, so the
/// payload alone is ambiguous and only the declared mode can tell them apart.
fn reject_legacy_mxfp8_mamba(per_layer: &HashMap<String, PerLayerQuant>) -> Result<()> {
    for (key, plq) in per_layer {
        if plq.mode == PerLayerMode::Mxfp8
            && (key.ends_with(".mixer.in_proj") || key.ends_with(".mixer.out_proj"))
        {
            return Err(Error::from_reason(format!(
                "Nemotron-H projection '{key}' is declared mxfp8. {REQUANT_REGENERATE_HINT}"
            )));
        }
    }
    Ok(())
}

/// The affine-8/32 mamba projections carry floating `.scales` AND a
/// same-shaped floating `.biases`. An mxfp8-era payload has Uint8 `.scales`
/// and no `.biases` at all, and `try_build_quantized_linear` accepts it
/// happily — MLX only rejects it at FORWARD time ("Biases must be provided for
/// affine quantization" / "scales.dtype() == uint8"), deep in the first decode
/// and with no regenerate hint. Catch it at load instead.
fn require_affine_sidecars(params: &HashMap<String, MxArray>, prefix: &str) -> Result<()> {
    let scales = params.get(&format!("{prefix}.scales")).ok_or_else(|| {
        Error::from_reason(format!(
            "affine projection '{prefix}' is missing '.scales'. {REQUANT_REGENERATE_HINT}"
        ))
    })?;
    let scales_dtype = scales.dtype()?;
    if !matches!(
        scales_dtype,
        DType::BFloat16 | DType::Float16 | DType::Float32
    ) {
        return Err(Error::from_reason(format!(
            "affine projection '{prefix}': '.scales' is {scales_dtype:?}, expected a floating \
             dtype — Uint8 scales are the superseded mxfp8 payload. {REQUANT_REGENERATE_HINT}"
        )));
    }
    let biases = params.get(&format!("{prefix}.biases")).ok_or_else(|| {
        Error::from_reason(format!(
            "affine projection '{prefix}' is missing the mandatory '.biases' sidecar. \
             {REQUANT_REGENERATE_HINT}"
        ))
    })?;
    let biases_dtype = biases.dtype()?;
    let biases_shape = biases.shape()?.to_vec();
    let scales_shape = scales.shape()?.to_vec();
    if !matches!(
        biases_dtype,
        DType::BFloat16 | DType::Float16 | DType::Float32
    ) || biases_shape != scales_shape
    {
        return Err(Error::from_reason(format!(
            "affine projection '{prefix}': '.biases' {biases_dtype:?}{biases_shape:?} must match \
             '.scales' {scales_dtype:?}{scales_shape:?}. {REQUANT_REGENERATE_HINT}"
        )));
    }
    Ok(())
}

/// Load quant settings from disk: read the quantization block from
/// config.json, returning the parsed per-layer overrides. Absent block =
/// empty overrides (dense checkpoint).
fn load_quant_settings(
    path: &Path,
) -> Result<(Option<PerLayerMode>, HashMap<String, PerLayerQuant>)> {
    let config_path = path.join("config.json");
    let Ok(raw_str) = fs::read_to_string(&config_path) else {
        return Ok((None, HashMap::new()));
    };
    let Ok(raw) = serde_json::from_str::<Value>(&raw_str) else {
        return Ok((None, HashMap::new()));
    };
    let quant_cfg = select_quantization_block(&raw)?;
    parse_nemotron_quant_settings(quant_cfg)
}

/// Apply the sanitized weights to a constructed inner.
///
/// Per-layer quant resolution mirrors the qwen3.5 families: every
/// quantizable projection resolves its PerLayerQuant through
/// effective_plq_for and dispatches on the mode. Dense (bf16) projections
/// fall through to set_weight with a dtype guard so packed storage can
/// never enter dense math.
fn apply_weights(
    inner: &mut NemotronHInner,
    params: &HashMap<String, MxArray>,
    top_level_mode: Option<PerLayerMode>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
) -> Result<()> {
    let is_quantized = is_quantized_checkpoint(params);
    let default_plq = default_per_layer_quant(
        DEFAULT_QUANT_BITS,
        DEFAULT_QUANT_GROUP_SIZE,
        top_level_mode.unwrap_or(PerLayerMode::Affine),
    );
    let plq_for = |prefix: &str| -> PerLayerQuant {
        effective_plq_for(prefix, per_layer_quant, default_plq, None)
    };

    // Mode-aware linear builder (2-D weights: shared experts, lm_head,
    // mamba in_proj/out_proj, MTP-free 2-D projections).
    let try_build_ql = |params: &HashMap<String, MxArray>,
                        prefix: &str|
     -> Result<Option<QuantizedLinear>> {
        let plq = plq_for(prefix);
        Ok(match plq.mode {
            PerLayerMode::Nvfp4 => {
                let built = try_build_nvfp4_quantized_linear(params, prefix);
                // FAIL-CLOSED, symmetric with the Affine/Mxfp8 arms below.
                // `try_build_nvfp4_quantized_linear` reads only `.weight` and
                // `.scales`; the per-tensor `.global_scale` is applied by the
                // CALLER, and only the shared-expert caller has a slot for it
                // (see `nvfp4_global_scale_is_applied`). Building here for any
                // other prefix would drop `weight_scale_2` silently.
                if built.is_some() && !nvfp4_global_scale_is_applied(prefix) {
                    return Err(Error::from_reason(format!(
                        "NVFP4 projection '{prefix}' is packed nvfp4 (.weight + .scales), but \
                         only the shared experts apply the per-tensor '.global_scale' at \
                         runtime — loading it here would drop weight_scale_2 and run the \
                         projection ~1/weight_scale_2 (~1.15e4x) too large. {REGENERATE_HINT}"
                    )));
                }
                built
            }
            PerLayerMode::Affine => {
                let built = try_build_quantized_linear(params, prefix, plq.group_size, plq.bits);
                if built.is_some() {
                    require_affine_sidecars(params, prefix)?;
                }
                // The mamba projections are W8A8: the static E4M3
                // activation scale rides with the affine weights now, so
                // thread it here as the mxfp8 arm used to.
                built.map(|ql| ql.with_input_amax(plq.input_amax))
            }
            PerLayerMode::Mxfp8 => {
                return Err(Error::from_reason(format!(
                    "Nemotron-H projection '{prefix}' resolves to mxfp8. \
                         {REQUANT_REGENERATE_HINT}"
                )));
            }
            other => {
                // Fp8E4m3 / sym8 / k-quants are not produced for this family.
                let _ = other;
                None
            }
        })
    };
    // Stacked-expert builder (3-D [E,N,K]). An NVFP4 stack is only usable
    // together with its [E] global scale, so the two are built as one unit
    // and a missing sidecar is a hard error, not a dense fallback.
    let n_routed_experts = inner.config.n_routed_experts as i64;
    let try_build_qsl = |params: &HashMap<String, MxArray>,
                         prefix: &str|
     -> Result<Option<(QuantizedSwitchLinear, MxArray)>> {
        let plq = plq_for(prefix);
        if plq.mode != PerLayerMode::Nvfp4 {
            return Ok(None);
        }
        let Some(qsl) = try_build_nvfp4_quantized_switch_linear(params, prefix) else {
            return Ok(None);
        };
        let gs = require_expert_global_scale(params, prefix, n_routed_experts)?;
        Ok(Some((qsl, gs)))
    };

    // Embedding (always dense bf16 for this family).
    let emb = params.get("embedding.weight").ok_or_else(|| {
        Error::from_reason("Checkpoint missing required tensor 'embedding.weight'")
    })?;
    crate::models::quant_dispatch::ensure_dense_weight_floating("embedding.weight", emb)?;
    inner.embedding.set_weight(emb)?;

    // final_norm.
    let final_norm = params.get("final_norm.weight").ok_or_else(|| {
        Error::from_reason("Checkpoint missing required tensor 'final_norm.weight'")
    })?;
    inner.final_norm.set_weight(final_norm)?;

    // lm_head - NVFP4 quantized on the modelopt checkpoint, dense bf16
    // otherwise. The model has an UNTIED lm_head (tie_word_embeddings
    // false), so it is always present.
    if is_quantized {
        if let Some(ql) = try_build_ql(params, "lm_head")? {
            inner.lm_head = Some(LinearProj::Quantized(ql));
        } else if let Some(w) = params.get("lm_head.weight") {
            crate::models::quant_dispatch::ensure_dense_weight_floating("lm_head.weight", w)?;
            let mut head = LinearProj::Standard(crate::nn::Linear::new(
                inner.config.hidden_size as u32,
                inner.config.vocab_size as u32,
                Some(false),
            )?);
            head.set_weight(w, "lm_head")?;
            inner.lm_head = Some(head);
        }
    } else if let Some(w) = params.get("lm_head.weight") {
        crate::models::quant_dispatch::ensure_dense_weight_floating("lm_head.weight", w)?;
        let mut head = LinearProj::Standard(crate::nn::Linear::new(
            inner.config.hidden_size as u32,
            inner.config.vocab_size as u32,
            Some(false),
        )?);
        head.set_weight(w, "lm_head")?;
        inner.lm_head = Some(head);
    }
    if inner.lm_head.is_none() {
        return Err(Error::from_reason(
            "Checkpoint missing lm_head.weight (NemotronH has an untied lm_head)",
        ));
    }

    // Layers.
    for (i, layer) in inner.layers.iter_mut().enumerate() {
        let prefix = format!("layers.{}", i);

        if let Some(m) = layer.mamba_mut() {
            let in_key = format!("{prefix}.mixer.in_proj");
            if let Some(ql) = try_build_ql(params, &in_key)? {
                m.in_proj.set_quantized(ql);
            } else if let Some(w) = params.get(&format!("{in_key}.weight")) {
                crate::models::quant_dispatch::ensure_dense_weight_floating(
                    &format!("{in_key}.weight"),
                    w,
                )?;
                m.in_proj.set_weight(w, "in_proj")?;
            } else {
                return Err(Error::from_reason(format!(
                    "Mamba layer {i}: missing required projection '{in_key}' (quantized or dense)"
                )));
            }
            let out_key = format!("{prefix}.mixer.out_proj");
            if let Some(ql) = try_build_ql(params, &out_key)? {
                m.out_proj.set_quantized(ql);
            } else if let Some(w) = params.get(&format!("{out_key}.weight")) {
                crate::models::quant_dispatch::ensure_dense_weight_floating(
                    &format!("{out_key}.weight"),
                    w,
                )?;
                m.out_proj.set_weight(w, "out_proj")?;
            } else {
                return Err(Error::from_reason(format!(
                    "Mamba layer {i}: missing required projection '{out_key}' (quantized or dense)"
                )));
            }
            for name in [
                "conv1d.weight",
                "conv1d.bias",
                "dt_bias",
                "A_log",
                "D",
                "norm.weight",
            ] {
                let key = format!("{prefix}.mixer.{name}");
                let w = params.get(&key).ok_or_else(|| {
                    Error::from_reason(format!("Mamba layer {i}: missing required tensor '{key}'"))
                })?;
                match name {
                    "conv1d.weight" => m.conv1d.set_weight(w)?,
                    "conv1d.bias" => m.conv1d.set_bias(Some(w))?,
                    "dt_bias" => m.dt_bias = w.astype(DType::Float32)?,
                    "A_log" => m.a_log = w.astype(DType::Float32)?,
                    "D" => m.d = w.astype(DType::Float32)?,
                    _ => m.norm_weight = w.astype(DType::Float32)?,
                }
            }
        }

        if let Some(a) = layer.attention_mut() {
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                let key = format!("{prefix}.mixer.{proj}.weight");
                let w = params.get(&key).ok_or_else(|| {
                    Error::from_reason(format!(
                        "Attention layer {i}: missing required tensor '{key}'"
                    ))
                })?;
                crate::models::quant_dispatch::ensure_dense_weight_floating(&key, w)?;
                match proj {
                    "q_proj" => a.q_proj.set_weight(w, "q_proj")?,
                    "k_proj" => a.k_proj.set_weight(w, "k_proj")?,
                    "v_proj" => a.v_proj.set_weight(w, "v_proj")?,
                    _ => a.o_proj.set_weight(w, "o_proj")?,
                }
            }
        }

        if let Some(m) = layer.moe_mut() {
            let gate_key = format!("{prefix}.mixer.gate.weight");
            let gate_w = params.get(&gate_key).ok_or_else(|| {
                Error::from_reason(format!(
                    "MoE layer {i}: missing required tensor '{gate_key}'"
                ))
            })?;
            m.gate.set_weight(&gate_w.astype(DType::Float32)?, "gate")?;
            let bias_key = format!("{prefix}.mixer.gate.e_score_correction_bias");
            let bias_w = params.get(&bias_key).ok_or_else(|| {
                Error::from_reason(format!(
                    "MoE layer {i}: missing required tensor '{bias_key}'"
                ))
            })?;
            // The router bias is ~40.7 with a spread of only ~0.1 across the
            // 128 experts; BF16's ULP there is 0.25, so a BF16 copy collapses
            // every entry onto one value and DELETES the load-balancing
            // correction (measured on the pre-fix checkpoint: 1 unique value
            // of 128). Convert now retains the source F32, so a BF16 tensor
            // here is a pre-fix checkpoint. Every shipped Nemotron-H source
            // ships this bias as F32 (verified: 46/46 backbone router tensors
            // in 30B-A3B); a future NVIDIA variant shipping BF16 would be
            // rejected spuriously, but loudly and with a one-line fix.
            if bias_w.dtype()? == DType::BFloat16 {
                return Err(Error::from_reason(format!(
                    "MoE layer {i}: '{bias_key}' is BF16. {REQUANT_REGENERATE_HINT}"
                )));
            }
            m.e_score_correction_bias = bias_w.astype(DType::Float32)?;
            // Stacked experts (NVFP4 quantized or dense fallback) — both
            // sides are required for a usable expert layer.
            let up_key = format!("{prefix}.mixer.experts.up_proj");
            let down_key = format!("{prefix}.mixer.experts.down_proj");
            let up = build_expert_stack(params, &up_key, &try_build_qsl)?.ok_or_else(|| {
                Error::from_reason(format!("MoE layer {i}: missing experts.up_proj weight"))
            })?;
            let down = build_expert_stack(params, &down_key, &try_build_qsl)?.ok_or_else(|| {
                Error::from_reason(format!("MoE layer {i}: missing experts.down_proj weight"))
            })?;
            m.experts.set_experts(up, down)?;
            // Shared experts (NVFP4 quantized or dense fallback) — both
            // sides are required.
            for proj in ["up_proj", "down_proj"] {
                let key = format!("{prefix}.mixer.shared_experts.{proj}");
                if let Some(ql) = try_build_ql(params, &key)? {
                    // NVFP4 shared experts carry a per-tensor global scale;
                    // any other quantized mode (none is produced for this
                    // family today) has none. Fail closed on a missing one.
                    let global_scale = if plq_for(&key).mode == PerLayerMode::Nvfp4 {
                        Some(require_scalar_global_scale(params, &key)?)
                    } else {
                        None
                    };
                    match proj {
                        "up_proj" => {
                            m.shared_experts.up_proj.set_quantized(ql);
                            m.shared_experts.up_global_scale = global_scale;
                        }
                        _ => {
                            m.shared_experts.down_proj.set_quantized(ql);
                            m.shared_experts.down_global_scale = global_scale;
                        }
                    }
                } else if let Some(w) = params.get(&format!("{key}.weight")) {
                    crate::models::quant_dispatch::ensure_dense_weight_floating(
                        &format!("{key}.weight"),
                        w,
                    )?;
                    match proj {
                        "up_proj" => m.shared_experts.up_proj.set_weight(w, "shared_up")?,
                        _ => m.shared_experts.down_proj.set_weight(w, "shared_down")?,
                    }
                } else {
                    return Err(Error::from_reason(format!(
                        "MoE layer {i}: missing required projection '{key}' (quantized or dense)"
                    )));
                }
            }
        }

        let norm_key = format!("{prefix}.norm.weight");
        let norm_w = params.get(&norm_key).ok_or_else(|| {
            Error::from_reason(format!("Layer {i}: missing required tensor '{norm_key}'"))
        })?;
        layer.set_norm_weight(norm_w)?;
    }

    /// Build one stacked expert projection (up or down) from the checkpoint,
    /// returning None when the layer is not an expert layer (no tensors).
    fn build_expert_stack<F>(
        params: &HashMap<String, MxArray>,
        key: &str,
        try_build_qsl: &F,
    ) -> Result<Option<super::sparse_moe::ExpertProj>>
    where
        F: Fn(&HashMap<String, MxArray>, &str) -> Result<Option<(QuantizedSwitchLinear, MxArray)>>,
    {
        if let Some((qsl, global_scale)) = try_build_qsl(params, key)? {
            return Ok(Some(super::sparse_moe::ExpertProj::Quantized(
                qsl,
                global_scale,
            )));
        }
        if let Some(w) = params.get(&format!("{}.weight", key)) {
            crate::models::quant_dispatch::ensure_dense_weight_floating(
                &format!("{}.weight", key),
                w,
            )?;
            return Ok(Some(super::sparse_moe::ExpertProj::Dense(w.clone())));
        }
        Ok(None)
    }

    // MTP head - all dense bf16.
    apply_mtp_weights(inner, params)?;

    Ok(())
}

/// Load the dense bf16 mtp.* weights into the MTP module. Fails closed on
/// a PARTIAL MTP weight set: the module stays unloaded (mtp_weights_loaded
/// false) and speculative MTP is disabled with a loud warn.
fn apply_mtp_weights(inner: &mut NemotronHInner, params: &HashMap<String, MxArray>) -> Result<()> {
    let Some(mtp) = inner.mtp.as_mut() else {
        return Ok(());
    };
    let required = [
        "mtp.layers.0.enorm.weight",
        "mtp.layers.0.hnorm.weight",
        "mtp.layers.0.eh_proj.weight",
        "mtp.layers.0.norm.weight",
        "mtp.layers.0.mixer.q_proj.weight",
        "mtp.layers.0.mixer.k_proj.weight",
        "mtp.layers.0.mixer.v_proj.weight",
        "mtp.layers.0.mixer.o_proj.weight",
        "mtp.layers.1.norm.weight",
        "mtp.layers.1.final_layernorm.weight",
        "mtp.layers.1.mixer.gate.weight",
        "mtp.layers.1.mixer.gate.e_score_correction_bias",
        "mtp.layers.1.mixer.experts.up_proj.weight",
        "mtp.layers.1.mixer.experts.down_proj.weight",
        "mtp.layers.1.mixer.shared_experts.up_proj.weight",
        "mtp.layers.1.mixer.shared_experts.down_proj.weight",
    ];
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|k| !params.contains_key(*k))
        .collect();
    if !missing.is_empty() {
        inner.mtp_weights_loaded = false;
        warn!(
            "NemotronH config declares an MTP head (n_mtp_layers={}), but MTP weights are              incomplete; disabling speculative MTP. Missing first entries: {:?} ({} total)",
            inner.config.n_mtp_layers,
            &missing[..missing.len().min(8)],
            missing.len()
        );
        return Ok(());
    }

    // The MTP head runs its OWN attention over its OWN KV cache, so
    // k_proj/v_proj are live weights (they used to be loaded and never
    // read). A mis-shaped projection would only surface as a matmul failure
    // deep inside the first draft, so validate here and take the same
    // fail-closed branch as a missing key.
    let kv_dim = (inner.config.num_key_value_heads * inner.config.head_dim) as i64;
    let hidden = inner.config.hidden_size as i64;
    for key in [
        "mtp.layers.0.mixer.k_proj.weight",
        "mtp.layers.0.mixer.v_proj.weight",
    ] {
        let w = params.get(key).unwrap();
        let shape = w.shape()?.to_vec();
        if shape.len() != 2 || shape[0] != kv_dim || shape[1] != hidden {
            inner.mtp_weights_loaded = false;
            warn!(
                "NemotronH MTP weight {} has shape {:?}, expected [{}, {}]                  (num_key_value_heads * head_dim, hidden_size); disabling speculative MTP",
                key, shape, kv_dim, hidden
            );
            return Ok(());
        }
    }

    // The four MoE-body tensors are installed VERBATIM as dense bf16 — there
    // is no dequant path on the MTP ingest. A packed (U8/U32) or mis-shaped
    // tensor would only surface as garbage logits or a matmul failure deep
    // inside the first draft, so validate dtype AND shape here and take the
    // same fail-closed branch as a missing key. Convert refuses to PRODUCE a
    // quantized MTP head (classify_nemotron_key); this refuses to CONSUME one.
    let n_experts = inner.config.n_routed_experts as i64;
    let inter = inner.config.intermediate_size as i64;
    let shared_inter = inner.config.moe_shared_expert_intermediate_size as i64;
    for (key, want) in [
        (
            "mtp.layers.1.mixer.experts.up_proj.weight",
            vec![n_experts, inter, hidden],
        ),
        (
            "mtp.layers.1.mixer.experts.down_proj.weight",
            vec![n_experts, hidden, inter],
        ),
        (
            "mtp.layers.1.mixer.shared_experts.up_proj.weight",
            vec![shared_inter, hidden],
        ),
        (
            "mtp.layers.1.mixer.shared_experts.down_proj.weight",
            vec![hidden, shared_inter],
        ),
    ] {
        let w = params.get(key).unwrap();
        let dtype = w.dtype()?;
        let shape = w.shape()?.to_vec();
        if !matches!(dtype, DType::Float32 | DType::Float16 | DType::BFloat16) || shape != want {
            inner.mtp_weights_loaded = false;
            warn!(
                "NemotronH MTP weight {} is {:?}{:?}, expected a floating {:?} (the MTP ingest \
                 has no dequant path); disabling speculative MTP",
                key, dtype, shape, want
            );
            return Ok(());
        }
    }

    mtp.enorm
        .set_weight(params.get("mtp.layers.0.enorm.weight").unwrap())?;
    mtp.hnorm
        .set_weight(params.get("mtp.layers.0.hnorm.weight").unwrap())?;
    mtp.eh_proj.set_weight(
        params.get("mtp.layers.0.eh_proj.weight").unwrap(),
        "mtp.eh_proj",
    )?;
    mtp.final_layernorm
        .set_weight(params.get("mtp.layers.1.final_layernorm.weight").unwrap())?;

    {
        let layer = &mut mtp.layers[0];
        layer
            .norm
            .set_weight(params.get("mtp.layers.0.norm.weight").unwrap())?;
        let attn = match &mut layer.mixer {
            super::mtp::NemotronHMtpMixer::Attention(a) => a,
            _ => return Err(Error::from_reason("MTP layer 0 must be attention")),
        };
        attn.q_proj.set_weight(
            params.get("mtp.layers.0.mixer.q_proj.weight").unwrap(),
            "mtp.q_proj",
        )?;
        attn.k_proj.set_weight(
            params.get("mtp.layers.0.mixer.k_proj.weight").unwrap(),
            "mtp.k_proj",
        )?;
        attn.v_proj.set_weight(
            params.get("mtp.layers.0.mixer.v_proj.weight").unwrap(),
            "mtp.v_proj",
        )?;
        attn.o_proj.set_weight(
            params.get("mtp.layers.0.mixer.o_proj.weight").unwrap(),
            "mtp.o_proj",
        )?;
    }

    {
        let layer = &mut mtp.layers[1];
        layer
            .norm
            .set_weight(params.get("mtp.layers.1.norm.weight").unwrap())?;
        let moe = match &mut layer.mixer {
            super::mtp::NemotronHMtpMixer::MoE(m) => m,
            _ => return Err(Error::from_reason("MTP layer 1 must be MoE")),
        };
        moe.gate.set_weight(
            &params
                .get("mtp.layers.1.mixer.gate.weight")
                .unwrap()
                .astype(DType::Float32)?,
            "mtp.gate",
        )?;
        moe.e_score_correction_bias = params
            .get("mtp.layers.1.mixer.gate.e_score_correction_bias")
            .unwrap()
            .astype(DType::Float32)?;
        moe.experts.set_experts(
            ExpertProj::Dense(
                params
                    .get("mtp.layers.1.mixer.experts.up_proj.weight")
                    .unwrap()
                    .clone(),
            ),
            ExpertProj::Dense(
                params
                    .get("mtp.layers.1.mixer.experts.down_proj.weight")
                    .unwrap()
                    .clone(),
            ),
        )?;
        moe.shared_experts.up_proj.set_weight(
            params
                .get("mtp.layers.1.mixer.shared_experts.up_proj.weight")
                .unwrap(),
            "mtp.shared_up",
        )?;
        moe.shared_experts.down_proj.set_weight(
            params
                .get("mtp.layers.1.mixer.shared_experts.down_proj.weight")
                .unwrap(),
            "mtp.shared_down",
        )?;
    }

    inner.mtp_weights_loaded = true;
    info!(
        "NemotronH MTP head loaded ({} layers, dense bf16)",
        mtp.layers.len()
    );
    Ok(())
}

/// Synchronous load of the model weights into a fresh inner. Returns the
/// deterministic weight-byte total for the cache-limit coordinator.
pub(crate) fn load_inner(model_path: &str) -> Result<(NemotronHInner, u64)> {
    let path = Path::new(model_path);
    if !path.exists() {
        return Err(Error::from_reason(format!(
            "Model path does not exist: {}",
            model_path
        )));
    }

    let config_path = path.join("config.json");
    let config_data = fs::read_to_string(&config_path)
        .map_err(|e| Error::from_reason(format!("Failed to read config: {}", e)))?;
    let raw: Value = serde_json::from_str(&config_data)
        .map_err(|e| Error::from_reason(format!("Failed to parse config: {}", e)))?;
    let config = parse_config(&raw)?;

    info!(
        "NemotronH config: {} layers ({} mamba + {} moe + {} attention), hidden={}, experts={}x{}",
        config.num_hidden_layers,
        (0..config.num_hidden_layers as usize)
            .filter(|&i| config.is_mamba_layer(i))
            .count(),
        (0..config.num_hidden_layers as usize)
            .filter(|&i| config.is_moe_layer(i))
            .count(),
        (0..config.num_hidden_layers as usize)
            .filter(|&i| config.is_attention_layer(i))
            .count(),
        config.hidden_size,
        config.n_routed_experts,
        config.num_experts_per_tok,
    );

    // Quantization settings from config.json (mlx or modelopt schema).
    let (top_level_mode, per_layer_quant) = load_quant_settings(path)?;
    reject_legacy_mxfp8_mamba(&per_layer_quant)?;
    if top_level_mode.is_some() || !per_layer_quant.is_empty() {
        info!(
            "NemotronH quantization: {} per-layer overrides",
            per_layer_quant.len()
        );
    }

    // Load all safetensors.
    let mut raw_params = load_all_safetensors(path, false)?;
    prewarm_checkpoint_pages(path);

    let params = sanitize_weights(std::mem::take(&mut raw_params), &config)?;
    info!("NemotronH sanitized to {} tensors", params.len());

    let mut inner = NemotronHInner::new(config.clone())?;
    inner.set_gen_defaults(parse_generation_defaults(path));
    // Quantized projections (NVFP4 experts/lm_head, MXFP8 mamba) dispatch
    // different matmul kernels for M=1 vs M>=2 rows; the batched decode lane
    // must then run those projections per row so its output is bit-identical
    // to the single-row decode (the mamba recurrence amplifies a few ULP into
    // token flips). Dense/bf16 checkpoints keep the fused batched path.
    inner.row_exact_decode_projections = top_level_mode.is_some() || !per_layer_quant.is_empty();
    apply_weights(&mut inner, &params, top_level_mode, &per_layer_quant)?;

    // Materialize the loaded weights (chunked evals avoid Metal command
    // buffer timeouts).
    let weight_refs: Vec<&MxArray> = params.values().collect();
    crate::array::memory::materialize_weights(&weight_refs)?;

    // Tokenizer.
    let tokenizer_path = path.join("tokenizer.json");
    if tokenizer_path.exists() {
        let tokenizer = Qwen3Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::from_reason(format!("Failed to load tokenizer: {}", e)))?;
        inner.set_tokenizer(Arc::new(tokenizer));
        info!("NemotronH tokenizer loaded");
    }

    let weight_bytes: u64 = params
        .values()
        .map(|a| a.nbytes() as u64)
        .fold(0u64, |acc, v| acc.saturating_add(v));

    Ok((inner, weight_bytes))
}

/// Load a pretrained NemotronH model into a dedicated model thread running
/// the engine-owned continuous-batching scheduler (when the block-paged
/// adapter is active; otherwise the thread behaves as the legacy whole-turn
/// loop because the scheduler routes every command to the barrier lane).
pub async fn load_with_thread(model_path: &str) -> Result<NemotronHModel> {
    use super::model::NemotronHSchedulerState;
    let model_path = model_path.to_string();

    let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_scheduler(
        move || {
            let (inner, weight_bytes) = load_inner(&model_path)?;
            let cache_limit_guard = crate::cache_limit::coordinator().register(weight_bytes);
            let mtp_active = inner.has_mtp_weights();
            let paged_active = inner.paged_adapter.is_some();
            let context_limits =
                super::model::NemotronHContextLimits::from_tuple(inner.paged_context_limits());
            let config = inner.config.clone();
            let scheduler = NemotronHSchedulerState::new(inner)?;
            Ok((
                scheduler,
                (
                    config,
                    cache_limit_guard,
                    mtp_active,
                    paged_active,
                    context_limits,
                ),
            ))
        },
        |state, receiver| state.drive(receiver),
    );

    let (config, cache_limit_guard, mtp_active, paged_active, context_limits) = init_rx
        .await
        .map_err(|_| Error::from_reason("Model thread exited during load"))??;

    Ok(NemotronHModel {
        thread,
        config,
        mtp_active,
        paged_active,
        context_limits,
        _cache_limit_guard: cache_limit_guard,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::safetensors::save_safetensors;

    /// Write the BF16 tensor set for the 3-layer (mamba, moe, attention)
    /// synthetic checkpoint into `dir`. Shared so the happy-path load and the
    /// legacy-format rejection gate below differ ONLY in `config.json` — which
    /// is the whole point of the latter, since the on-disk `.weight` bytes are
    /// identical between the mxfp8 and affine encodings.
    fn write_synthetic_bf16_tensors(dir: &std::path::Path) {
        write_synthetic_bf16_tensors_with(dir, Vec::new());
    }

    /// As above, but `extra` is inserted LAST so it can add sidecars or
    /// replace a dense `.weight` with a quantized payload.
    fn write_synthetic_bf16_tensors_with(dir: &std::path::Path, extra: Vec<(String, MxArray)>) {
        let mut tensors: HashMap<String, MxArray> = HashMap::new();

        let put = |name: &str, shape: &[i64], tensors: &mut HashMap<String, MxArray>| {
            let n: usize = shape.iter().map(|&s| s as usize).product();
            let vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013) % 1.0 - 0.5).collect();
            let arr = MxArray::from_float32(&vals, shape)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            tensors.insert(name.to_string(), arr);
        };

        put("embedding.weight", &[32, 8], &mut tensors);
        put("final_norm.weight", &[8], &mut tensors);
        put("lm_head.weight", &[32, 8], &mut tensors);
        // layer 0: mamba
        put("layers.0.norm.weight", &[8], &mut tensors);
        put("layers.0.mixer.in_proj.weight", &[14, 8], &mut tensors);
        put("layers.0.mixer.conv1d.weight", &[8, 4, 1], &mut tensors);
        put("layers.0.mixer.conv1d.bias", &[8], &mut tensors);
        put("layers.0.mixer.dt_bias", &[2], &mut tensors);
        put("layers.0.mixer.A_log", &[2], &mut tensors);
        put("layers.0.mixer.D", &[2], &mut tensors);
        put("layers.0.mixer.norm.weight", &[4], &mut tensors);
        put("layers.0.mixer.out_proj.weight", &[8, 4], &mut tensors);
        // layer 1: moe (2 experts, stacked by the loader)
        put("layers.1.norm.weight", &[8], &mut tensors);
        put("layers.1.mixer.gate.weight", &[2, 8], &mut tensors);
        put(
            "layers.1.mixer.experts.0.up_proj.weight",
            &[6, 8],
            &mut tensors,
        );
        put(
            "layers.1.mixer.experts.0.down_proj.weight",
            &[8, 6],
            &mut tensors,
        );
        put(
            "layers.1.mixer.experts.1.up_proj.weight",
            &[6, 8],
            &mut tensors,
        );
        put(
            "layers.1.mixer.experts.1.down_proj.weight",
            &[8, 6],
            &mut tensors,
        );
        put(
            "layers.1.mixer.shared_experts.up_proj.weight",
            &[8, 8],
            &mut tensors,
        );
        put(
            "layers.1.mixer.shared_experts.down_proj.weight",
            &[8, 8],
            &mut tensors,
        );
        // layer 2: attention
        put("layers.2.norm.weight", &[8], &mut tensors);
        put("layers.2.mixer.q_proj.weight", &[8, 8], &mut tensors);
        put("layers.2.mixer.k_proj.weight", &[4, 8], &mut tensors);
        put("layers.2.mixer.v_proj.weight", &[4, 8], &mut tensors);
        put("layers.2.mixer.o_proj.weight", &[8, 8], &mut tensors);
        // The router bias is retained at F32 by convert (BF16's ULP at the
        // real bias magnitude ~40.7 exceeds the whole 0.06-0.20 inter-expert
        // spread); the loader rejects a BF16 one.
        tensors.insert(
            "layers.1.mixer.gate.e_score_correction_bias".to_string(),
            MxArray::from_float32(&[0.25, -0.5], &[2]).unwrap(),
        );

        for (name, arr) in extra {
            tensors.insert(name, arr);
        }

        save_safetensors(dir.join("model.safetensors"), &mut tensors, None)
            .expect("write safetensors");
    }

    /// Build a tiny synthetic BF16 checkpoint (3 layers: mamba, moe,
    /// attention) in a temp dir, load it with load_inner, and run one flat
    /// forward.
    #[test]
    fn loads_synthetic_bf16_checkpoint_and_forwards() {
        let dir = std::env::temp_dir().join(format!("nemotron_h_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");

        // ---- config.json (converted mlx-schema: no quantization block) ----
        let config_json = r#"{
            "model_type": "nemotron_h",
            "architectures": ["NemotronHForCausalLM"],
            "vocab_size": 32,
            "hidden_size": 8,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 64,
            "layer_norm_epsilon": 1e-5,
            "rope_theta": 10000.0,
            "layers_block_type": ["mamba", "moe", "attention"],
            "mamba_num_heads": 2,
            "mamba_head_dim": 2,
            "ssm_state_size": 2,
            "n_groups": 1,
            "conv_kernel": 4,
            "chunk_size": 4,
            "time_step_min": 0.001,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "routed_scaling_factor": 1.0,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "intermediate_size": 6,
            "moe_shared_expert_intermediate_size": 8,
            "tie_word_embeddings": false,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "num_logits_to_keep": 1,
            "use_block_paged_cache": false
        }"#;
        fs::write(dir.join("config.json"), config_json).expect("write config");
        write_synthetic_bf16_tensors(&dir);

        let (mut inner, weight_bytes) = load_inner(dir.to_str().unwrap()).expect("load_inner");
        assert!(weight_bytes > 0);
        assert!(!inner.has_mtp_weights());

        // Run one flat forward over 3 tokens.
        let ids = MxArray::from_uint32(&[1, 5, 9], &[1, 3]).unwrap();
        let logits = inner.forward(&ids).expect("forward");
        let shape = logits.shape().unwrap().to_vec();
        assert_eq!(shape, vec![1, 3, 32]);

        // ---- CROSS-MODULE SEAM GATE: the flat lane must stay in the paged
        // lane's dtype end to end. ----
        //
        // This fixture's `layers_block_type` is EXACTLY the promotion chain:
        // mamba (layer 0) -> moe (layer 1) -> attention (layer 2), with every
        // weight BF16, so the residual enters layer 2 in whatever the first
        // two mixers hand it. Three separate f32 escapes feed this one
        // observable, and each lives in a different file:
        //   * `mamba2.rs::gated_rmsnorm` — f32 scan output x f32 norm weight;
        //   * `sparse_moe.rs::NemotronHMoE::forward` — the f32 router's
        //     `topk_weights` promoting the routed-expert sum;
        //   * `sparse_moe.rs::NemotronHSharedExpert::forward` — the f32 NVFP4
        //     `global_scale` arrays.
        // No single module's unit test can see the composition, which is
        // precisely why this assertion belongs here: `mamba2.rs`'s mixer tests
        // pass with the MoE escape wide open, and vice versa.
        //
        // Why it matters beyond tidiness: `KVCache::update_and_fetch`
        // allocates its buffer with `keys.dtype()`, so the FLAT cache has no
        // dtype of its own — it stores whatever the residual hands `k_proj`.
        // The PAGED pool is byte-allocated for 2-byte elements and hard-casts
        // to `PAGED_KV_IO_DTYPE`. If the residual promotes, the two lanes hold
        // different precision for the same tokens and stop being comparable.
        let kv = inner.caches[2]
            .as_kv_cache()
            .expect("layers_block_type[2] == \"attention\"");
        assert_eq!(
            kv.keys_ref().expect("prefill wrote keys").dtype().unwrap(),
            crate::models::nemotron_h::attention::PAGED_KV_IO_DTYPE,
            "flat KV keys must hold the dtype the paged pool writes; an f32 \
             here means a mixer promoted the residual stream"
        );
        assert_eq!(
            kv.values_ref()
                .expect("prefill wrote values")
                .dtype()
                .unwrap(),
            crate::models::nemotron_h::attention::PAGED_KV_IO_DTYPE,
            "flat KV values must hold the dtype the paged pool writes"
        );
        assert_eq!(
            logits.dtype().unwrap(),
            crate::models::nemotron_h::attention::PAGED_KV_IO_DTYPE,
            "an all-BF16 checkpoint must produce BF16 logits; f32 here means \
             the residual widened somewhere upstream of lm_head"
        );

        // Second forward (decode-style single token) reuses the caches.
        let ids2 = MxArray::from_uint32(&[17], &[1, 1]).unwrap();
        let logits2 = inner.forward(&ids2).expect("forward decode");
        assert_eq!(logits2.shape().unwrap().to_vec(), vec![1, 1, 32]);
        // Decode appends into the SAME buffer; a promotion that only showed up
        // on the single-token path would silently reallocate it.
        let kv2 = inner.caches[2].as_kv_cache().expect("attention slot");
        assert_eq!(
            kv2.keys_ref().unwrap().dtype().unwrap(),
            crate::models::nemotron_h::attention::PAGED_KV_IO_DTYPE,
            "decode must not widen the flat KV cache either"
        );
        assert_eq!(
            logits2.dtype().unwrap(),
            crate::models::nemotron_h::attention::PAGED_KV_IO_DTYPE,
            "decode logits must stay BF16 too"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// END-TO-END format gate: a pre-affine checkpoint must be rejected by
    /// `load_inner` itself, loudly, with the regenerate hint.
    ///
    /// The unit tests above call `reject_legacy_mxfp8_mamba` and
    /// `require_affine_sidecars` DIRECTLY, which proves the predicates but not
    /// that anything reaches them. This drives a real directory through the
    /// real loader, so a refactor that drops either call site — leaving a
    /// superseded checkpoint to load and silently run 5-7% relative RMS off
    /// across the whole 890M-param sequence-mixing backbone — fails here.
    ///
    /// Both arms reuse `write_synthetic_bf16_tensors`, so the ONLY thing that
    /// varies is `config.json`. That is the crux: the on-disk `.weight` bytes
    /// are byte-identical between the mxfp8 and affine encodings, so the
    /// declared mode is the only thing that can tell them apart.
    #[test]
    fn load_inner_rejects_a_pre_affine_checkpoint_end_to_end() {
        let base_cfg = |quant: &str| {
            format!(
                r#"{{
            "model_type": "nemotron_h",
            "architectures": ["NemotronHForCausalLM"],
            "vocab_size": 32,
            "hidden_size": 8,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 64,
            "layer_norm_epsilon": 1e-5,
            "rope_theta": 10000.0,
            "layers_block_type": ["mamba", "moe", "attention"],
            "mamba_num_heads": 2,
            "mamba_head_dim": 2,
            "ssm_state_size": 2,
            "n_groups": 1,
            "conv_kernel": 4,
            "chunk_size": 4,
            "time_step_min": 0.001,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "routed_scaling_factor": 1.0,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "intermediate_size": 6,
            "moe_shared_expert_intermediate_size": 8,
            "tie_word_embeddings": false,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "num_logits_to_keep": 1,
            "use_block_paged_cache": false,
            "quantization": {quant}
        }}"#
            )
        };

        // The superseded on-disk payload for the two mamba projections:
        // packed uint32 `.weight` + Uint8 E8M0 `.scales`, and NO `.biases`.
        // Identical bytes under both configs below — only the declared mode
        // differs, which is exactly why the mode has to be load-bearing.
        let legacy_mxfp8_payload = || -> Vec<(String, MxArray)> {
            let mut v = Vec::new();
            for (prefix, n) in [
                ("layers.0.mixer.in_proj", 14i64),
                ("layers.0.mixer.out_proj", 8),
            ] {
                v.push((
                    format!("{prefix}.weight"),
                    MxArray::from_uint32(&vec![0u32; (n * 2) as usize], &[n, 2]).unwrap(),
                ));
                v.push((
                    format!("{prefix}.scales"),
                    MxArray::from_uint8(&vec![0x38u8; n as usize], &[n, 1]).unwrap(),
                ));
            }
            v
        };

        let run = |tag: &str, quant: &str, extra: Vec<(String, MxArray)>| -> String {
            let dir = std::env::temp_dir().join(format!(
                "nemotron_h_fmt_{}_{}_{}",
                std::process::id(),
                tag,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("create temp dir");
            fs::write(dir.join("config.json"), base_cfg(quant)).expect("write config");
            write_synthetic_bf16_tensors_with(&dir, extra);
            let err = load_inner(dir.to_str().unwrap())
                .err()
                .unwrap_or_else(|| panic!("{tag}: load_inner must FAIL on this checkpoint"))
                .reason;
            let _ = fs::remove_dir_all(&dir);
            err
        };

        // ---- Guard 1: the dangerous direction. A mamba projection declared
        // mxfp8 is the superseded encoding, and it would otherwise load and
        // run cleanly. Must be refused from the CONFIG, before any tensor is
        // read.
        let legacy = run(
            "mxfp8",
            r#"{
                "bits": 8, "group_size": 32, "mode": "affine",
                "layers.0.mixer.in_proj": {"bits": 8, "group_size": 32, "mode": "mxfp8"},
                "layers.0.mixer.out_proj": {"bits": 8, "group_size": 32, "mode": "mxfp8"}
            }"#,
            legacy_mxfp8_payload(),
        );
        assert!(
            legacy.contains("mxfp8"),
            "Guard 1 must name the superseded mode; got: {legacy}"
        );
        assert!(
            legacy.contains("mixer.in_proj") || legacy.contains("mixer.out_proj"),
            "Guard 1 must name the offending projection; got: {legacy}"
        );
        assert!(
            legacy.contains("mlx convert -m nemotron_h"),
            "a fail-closed format error is only actionable with the regenerate \
             command; got: {legacy}"
        );

        // ---- Guard 2: the same checkpoint re-declared as affine 8/32 gets
        // PAST guard 1 (so guard 1 is mode-specific, not a blanket refusal of
        // every quantized mamba config) and is then caught by the sidecar
        // check, because an mxfp8-era payload carries no `.biases`.
        //
        // Without this, MLX only complains at the FIRST DECODE, deep inside a
        // forward and with no hint about what to do.
        let missing_sidecars = run(
            "affine",
            r#"{
                "bits": 8, "group_size": 32, "mode": "affine",
                "layers.0.mixer.in_proj": {"bits": 8, "group_size": 32, "mode": "affine"},
                "layers.0.mixer.out_proj": {"bits": 8, "group_size": 32, "mode": "affine"}
            }"#,
            legacy_mxfp8_payload(),
        );
        assert!(
            !missing_sidecars.contains("is declared mxfp8"),
            "Guard 1 must not fire on the affine replacement; got: {missing_sidecars}"
        );
        assert!(
            missing_sidecars.contains(".scales") || missing_sidecars.contains(".biases"),
            "Guard 2 must name the missing sidecar; got: {missing_sidecars}"
        );
        assert!(
            missing_sidecars.contains("mlx convert -m nemotron_h"),
            "Guard 2 must carry the regenerate hint too; got: {missing_sidecars}"
        );
    }

    /// A full, correctly shaped MTP weight set for the tiny config below.
    fn mtp_params(kv_dim: i64, hidden: i64) -> HashMap<String, MxArray> {
        let mut params: HashMap<String, MxArray> = HashMap::new();
        let mut put = |name: &str, shape: &[i64]| {
            let n: usize = shape.iter().map(|&s| s as usize).product();
            let vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011) % 1.0 - 0.5).collect();
            let arr = MxArray::from_float32(&vals, shape)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            params.insert(name.to_string(), arr);
        };
        let q_dim = 8i64; // num_attention_heads * head_dim
        put("mtp.layers.0.enorm.weight", &[hidden]);
        put("mtp.layers.0.hnorm.weight", &[hidden]);
        put("mtp.layers.0.eh_proj.weight", &[hidden, 2 * hidden]);
        put("mtp.layers.0.norm.weight", &[hidden]);
        put("mtp.layers.0.mixer.q_proj.weight", &[q_dim, hidden]);
        put("mtp.layers.0.mixer.k_proj.weight", &[kv_dim, hidden]);
        put("mtp.layers.0.mixer.v_proj.weight", &[kv_dim, hidden]);
        put("mtp.layers.0.mixer.o_proj.weight", &[hidden, q_dim]);
        put("mtp.layers.1.norm.weight", &[hidden]);
        put("mtp.layers.1.final_layernorm.weight", &[hidden]);
        put("mtp.layers.1.mixer.gate.weight", &[2, hidden]);
        put("mtp.layers.1.mixer.gate.e_score_correction_bias", &[2]);
        put("mtp.layers.1.mixer.experts.up_proj.weight", &[2, 6, hidden]);
        put(
            "mtp.layers.1.mixer.experts.down_proj.weight",
            &[2, hidden, 6],
        );
        put(
            "mtp.layers.1.mixer.shared_experts.up_proj.weight",
            &[hidden, hidden],
        );
        put(
            "mtp.layers.1.mixer.shared_experts.down_proj.weight",
            &[hidden, hidden],
        );
        params
    }

    fn mtp_test_inner() -> NemotronHInner {
        let raw: Value = serde_json::from_str(
            r#"{
            "model_type": "nemotron_h",
            "vocab_size": 32,
            "hidden_size": 8,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 64,
            "layer_norm_epsilon": 1e-5,
            "layers_block_type": ["mamba", "moe", "attention"],
            "mamba_num_heads": 2,
            "mamba_head_dim": 2,
            "ssm_state_size": 2,
            "n_groups": 1,
            "conv_kernel": 4,
            "chunk_size": 4,
            "time_step_min": 0.001,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "routed_scaling_factor": 1.0,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "intermediate_size": 6,
            "moe_shared_expert_intermediate_size": 8,
            "tie_word_embeddings": false,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "num_logits_to_keep": 1,
            "mtp_layers_block_type": ["attention", "moe"],
            "num_nextn_predict_layers": 1,
            "use_block_paged_cache": false
        }"#,
        )
        .expect("config json");
        let config = parse_config(&raw).expect("parse config");
        NemotronHInner::new(config).expect("inner builds")
    }

    /// The MTP head now RUNS its own k_proj/v_proj (they used to be loaded
    /// and never read), so a mis-shaped KV projection must fail closed at
    /// load time instead of exploding inside the first draft.
    ///
    /// Mutation this catches: dropping the shape gate — the wrong-shaped
    /// load reports `mtp_weights_loaded == true` and the head is armed with
    /// a projection that cannot produce `num_key_value_heads * head_dim`
    /// columns.
    #[test]
    fn mtp_kv_projection_shape_is_gated_at_load() {
        let kv_dim = 4i64; // num_key_value_heads(1) * head_dim(4)
        let hidden = 8i64;

        // Correct shapes load and arm the head.
        let mut inner = mtp_test_inner();
        apply_mtp_weights(&mut inner, &mtp_params(kv_dim, hidden)).expect("apply");
        assert!(
            inner.mtp_weights_loaded,
            "a correctly shaped MTP weight set must arm the head"
        );

        // A k_proj sized for a different KV head count fails closed.
        for bad_key in [
            "mtp.layers.0.mixer.k_proj.weight",
            "mtp.layers.0.mixer.v_proj.weight",
        ] {
            let mut inner = mtp_test_inner();
            let mut params = mtp_params(kv_dim, hidden);
            let bad = MxArray::from_float32(
                &vec![0.1f32; (kv_dim as usize + 1) * hidden as usize],
                &[kv_dim + 1, hidden],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            params.insert(bad_key.to_string(), bad);
            apply_mtp_weights(&mut inner, &params).expect("must not hard-error");
            assert!(
                !inner.mtp_weights_loaded,
                "{bad_key} with a wrong dim0 must disable the MTP head"
            );
            assert!(!inner.has_mtp_weights());
        }
    }

    /// The modelopt quant block parses into per-layer NVFP4/FP8 overrides
    /// with the stacked-expert aliases registered.
    #[test]
    fn parses_modelopt_quantization_block() {
        use serde_json::json;
        let raw = json!({
            "quantization_config": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
                    "backbone.layers.0.mixer.out_proj": {"quant_algo": "FP8"},
                    "backbone.layers.1.mixer.experts.0.up_proj": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
                    "backbone.layers.1.mixer.experts.3.down_proj": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
                    "backbone.layers.1.mixer.shared_experts.up_proj": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
                    "lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16}
                }
            }
        });
        let quant_cfg = select_quantization_block(&raw).unwrap();
        let (top, per_layer) = parse_nemotron_quant_settings(quant_cfg).unwrap();
        assert_eq!(top, None);
        // NVIDIA's per-tensor E4M3 mamba projections are ingested as affine
        // 8-bit group-32, NOT mxfp8 (whose E8M0 block scale clips ~50% of
        // groups). 8/32 also keeps them inside the input_amax allowlist.
        for proj in ["in_proj", "out_proj"] {
            let plq = per_layer[&format!("layers.0.mixer.{proj}")];
            assert_eq!(plq.mode, PerLayerMode::Affine, "{proj}");
            assert_eq!(plq.bits, 8, "{proj}");
            assert_eq!(plq.group_size, 32, "{proj}");
        }
        assert_eq!(
            per_layer["layers.1.mixer.experts.up_proj"].mode,
            PerLayerMode::Nvfp4
        );
        assert_eq!(
            per_layer["layers.1.mixer.experts.down_proj"].mode,
            PerLayerMode::Nvfp4
        );
        assert_eq!(per_layer["lm_head"].mode, PerLayerMode::Nvfp4);
        assert_eq!(per_layer["lm_head"].group_size, 16);
        assert_eq!(
            per_layer["layers.1.mixer.shared_experts.up_proj"].mode,
            PerLayerMode::Nvfp4
        );
    }

    /// A tiny single-MoE-layer NemotronH inner for the NVFP4 loader tests.
    fn nvfp4_moe_inner() -> NemotronHInner {
        let raw: Value = serde_json::from_str(
            r#"{
            "model_type": "nemotron_h",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "layer_norm_epsilon": 1e-5,
            "layers_block_type": ["moe"],
            "mamba_num_heads": 2,
            "mamba_head_dim": 2,
            "ssm_state_size": 2,
            "n_groups": 1,
            "conv_kernel": 4,
            "chunk_size": 4,
            "time_step_min": 0.001,
            "n_routed_experts": 3,
            "num_experts_per_tok": 1,
            "routed_scaling_factor": 1.0,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "intermediate_size": 16,
            "moe_shared_expert_intermediate_size": 16,
            "tie_word_embeddings": false,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "num_logits_to_keep": 1,
            "use_block_paged_cache": false
        }"#,
        )
        .expect("config json");
        let config = parse_config(&raw).expect("parse config");
        NemotronHInner::new(config).expect("inner builds")
    }

    /// A well-formed converted NVFP4 params map for [`nvfp4_moe_inner`]:
    /// packed u32 weights, u8 E4M3 `.scales`, and Float32 `.global_scale`
    /// ([E] on the stacked experts, [1] on the shared experts).
    fn nvfp4_moe_params() -> HashMap<String, MxArray> {
        const E: i64 = 3;
        const H: i64 = 16;
        let mut params: HashMap<String, MxArray> = HashMap::new();
        let put_bf16 = |name: &str, shape: &[i64], params: &mut HashMap<String, MxArray>| {
            let n: usize = shape.iter().map(|&s| s as usize).product();
            let vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017) % 1.0 - 0.5).collect();
            params.insert(
                name.to_string(),
                MxArray::from_float32(&vals, shape)
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap(),
            );
        };
        put_bf16("embedding.weight", &[32, H], &mut params);
        put_bf16("final_norm.weight", &[H], &mut params);
        put_bf16("lm_head.weight", &[32, H], &mut params);
        put_bf16("layers.0.norm.weight", &[H], &mut params);
        put_bf16("layers.0.mixer.gate.weight", &[E, H], &mut params);
        // F32 by construction: convert retains the router pair at source
        // precision and `apply_weights` rejects a BF16 bias.
        params.insert(
            "layers.0.mixer.gate.e_score_correction_bias".to_string(),
            MxArray::from_float32(&[0.25, -0.5, 0.125], &[E]).unwrap(),
        );

        // Stacked experts: [E, N=16, K=16] -> u32 [E,16,2] + u8 [E,16,1].
        let quant = |prefix: &str, shape: &[i64], params: &mut HashMap<String, MxArray>| {
            let words: usize = shape.iter().map(|&s| s as usize).product::<usize>() / 8;
            let w: Vec<u32> = (0..words as u32)
                .map(|i| i.wrapping_mul(0x0104_0301))
                .collect();
            let mut wshape = shape.to_vec();
            *wshape.last_mut().unwrap() = shape[shape.len() - 1] / 8;
            params.insert(
                format!("{prefix}.weight"),
                MxArray::from_uint32(&w, &wshape).unwrap(),
            );
            let groups: usize = shape.iter().map(|&s| s as usize).product::<usize>() / 16;
            let sb: Vec<u8> = (0..groups).map(|i| 0x38u8 + (i % 8) as u8).collect();
            let mut sshape = shape.to_vec();
            *sshape.last_mut().unwrap() = shape[shape.len() - 1] / 16;
            params.insert(
                format!("{prefix}.scales"),
                MxArray::from_uint8(&sb, &sshape).unwrap(),
            );
        };
        for proj in ["up_proj", "down_proj"] {
            let prefix = format!("layers.0.mixer.experts.{proj}");
            quant(&prefix, &[E, H, H], &mut params);
            params.insert(
                format!("{prefix}.global_scale"),
                MxArray::from_float32(&[8.646647e-5, 5.658e-5, 2.124e-4], &[E]).unwrap(),
            );
            let sh = format!("layers.0.mixer.shared_experts.{proj}");
            quant(&sh, &[H, H], &mut params);
            params.insert(
                format!("{sh}.global_scale"),
                MxArray::from_float32(&[1.234e-4], &[1]).unwrap(),
            );
        }
        params
    }

    fn apply_nvfp4(params: &HashMap<String, MxArray>) -> Result<()> {
        let mut inner = nvfp4_moe_inner();
        apply_weights(
            &mut inner,
            params,
            Some(PerLayerMode::Nvfp4),
            &HashMap::new(),
        )
    }

    /// A well-formed NVFP4 checkpoint loads; a stale one (old folding ingest,
    /// no `.global_scale`) is REJECTED with a message that names the missing
    /// key and says to re-convert.
    ///
    /// MUTATION CAUGHT: `unwrap_or(1.0)` / `unwrap_or_default()` on the
    /// lookup. The stored weight is `1/weight_scale_2` (~1.15e4) larger than
    /// the intended one, so a silent default produces a catastrophically wrong
    /// model with zero diagnostics — exactly the failure mode the old folding
    /// checkpoints would hit.
    #[test]
    fn nemotron_nvfp4_experts_require_global_scale() {
        apply_nvfp4(&nvfp4_moe_params()).expect("a complete NVFP4 params map must load");

        for key in [
            "layers.0.mixer.experts.up_proj.global_scale",
            "layers.0.mixer.experts.down_proj.global_scale",
            "layers.0.mixer.shared_experts.up_proj.global_scale",
            "layers.0.mixer.shared_experts.down_proj.global_scale",
        ] {
            let mut params = nvfp4_moe_params();
            params.remove(key);
            let err = apply_nvfp4(&params).expect_err(
                "a missing .global_scale must be a hard load error, never a 1.0 default",
            );
            let msg = err.reason.clone();
            assert!(
                msg.contains(key),
                "error must name the missing key '{key}': {msg}"
            );
            assert!(
                msg.contains("mlx convert"),
                "error must tell the user to regenerate the checkpoint: {msg}"
            );
        }
    }

    /// The `try_build_ql` NVFP4 arm is FAIL-CLOSED outside the shared
    /// experts. `try_build_nvfp4_quantized_linear` reads only `.weight` and
    /// `.scales`, and `QuantizedLinear` has no field to hold a per-tensor
    /// `.global_scale` — only `SparseMoeBlock::shared_experts` does. So a
    /// packed nvfp4 `lm_head` (which inherits the top-level nvfp4 mode on
    /// every converted checkpoint and enters this arm on every load) must be
    /// REJECTED, not built with weight_scale_2 dropped.
    ///
    /// MUTATION CAUGHT: restoring the bare
    /// `PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_linear(params, prefix)`
    /// arm — the checkpoint below then loads clean and runs ~1.15e4x hot.
    #[test]
    fn nemotron_nvfp4_outside_the_shared_experts_is_rejected() {
        // The shipped shape: lm_head is the dense BF16 tensor convert emits,
        // so the same nvfp4 mode resolution is harmless.
        apply_nvfp4(&nvfp4_moe_params()).expect("a dense bf16 lm_head must still load");

        // Repack lm_head [32,16] as an nvfp4 pair: u32 [32,2] + u8 E4M3 [32,1],
        // plus the sidecar convert writes for every non-expert NVFP4 group.
        let mut params = nvfp4_moe_params();
        let words: Vec<u32> = (0..64u32).map(|i| i.wrapping_mul(0x0104_0301)).collect();
        params.insert(
            "lm_head.weight".to_string(),
            MxArray::from_uint32(&words, &[32, 2]).unwrap(),
        );
        let scale_bytes: Vec<u8> = (0..32).map(|i| 0x38u8 + (i % 8) as u8).collect();
        params.insert(
            "lm_head.scales".to_string(),
            MxArray::from_uint8(&scale_bytes, &[32, 1]).unwrap(),
        );
        params.insert(
            "lm_head.global_scale".to_string(),
            MxArray::from_float32(&[8.646647e-5], &[1]).unwrap(),
        );

        let err = apply_nvfp4(&params)
            .expect_err("a packed nvfp4 lm_head has nowhere to apply its global scale");
        let msg = err.reason.clone();
        assert!(msg.contains("lm_head"), "error must name the prefix: {msg}");
        assert!(
            msg.contains("global_scale"),
            "error must name the dropped sidecar: {msg}"
        );
        assert!(
            msg.contains("mlx convert"),
            "error must carry the regenerate hint: {msg}"
        );
    }

    /// `.global_scale` must be Float32 — bf16 rounds weight_scale_2 (~8.6e-5)
    /// with ~0.4% error, a smaller version of the very bug this replaced.
    ///
    /// MUTATION CAUGHT: dropping the dtype guard (or letting a future convert
    /// change cast the key through the BF16 pass).
    #[test]
    fn nemotron_nvfp4_global_scale_must_be_float32() {
        for key in [
            "layers.0.mixer.experts.up_proj.global_scale",
            "layers.0.mixer.shared_experts.up_proj.global_scale",
        ] {
            let mut params = nvfp4_moe_params();
            let bf16 = params[key].astype(DType::BFloat16).unwrap();
            params.insert(key.to_string(), bf16);
            let err = apply_nvfp4(&params).expect_err("bf16 .global_scale must be rejected");
            assert!(err.reason.contains("Float32"), "{}", err.reason);
        }
    }

    /// The stacked-expert `.global_scale` must be `[n_routed_experts]` long —
    /// weight_scale_2 VARIES per expert (up to 4.18x on the real checkpoint),
    /// so a scalar would mis-scale every expert but one.
    ///
    /// MUTATION CAUGHT: accepting a rank-0/[1] scalar and broadcasting it.
    #[test]
    fn nemotron_nvfp4_expert_global_scale_must_be_per_expert() {
        let mut params = nvfp4_moe_params();
        params.insert(
            "layers.0.mixer.experts.up_proj.global_scale".to_string(),
            MxArray::from_float32(&[8.646647e-5], &[1]).unwrap(),
        );
        let err = apply_nvfp4(&params).expect_err("a scalar expert .global_scale must be rejected");
        assert!(err.reason.contains("[3]"), "{}", err.reason);
    }

    /// KEY PLUMBING: `sanitize_weights` must carry `.global_scale` through
    /// untouched. It sits next to `.weight_scale` / `.weight_scale_2`, which
    /// this function DROPS, and next to `.weight`, which it STACKS — so a
    /// sloppier suffix filter on either side would silently swallow it and
    /// leave `apply_weights` looking up a key that no longer exists.
    ///
    /// MUTATION CAUGHT: widening the drop filter to `contains("_scale")` or
    /// `ends_with("scale")`, or routing `.global_scale` into the per-expert
    /// stacker.
    #[test]
    fn sanitize_keeps_the_global_scale_key() {
        let config = nvfp4_moe_inner().config.clone();
        let mut params: HashMap<String, MxArray> = HashMap::new();
        let arr = |n: usize| MxArray::from_float32(&vec![0.5f32; n], &[n as i64]).unwrap();
        // Already-stacked converted keys (convert emits the [E] vector).
        params.insert(
            "backbone.layers.0.mixer.experts.up_proj.global_scale".to_string(),
            arr(3),
        );
        params.insert(
            "backbone.layers.0.mixer.shared_experts.up_proj.global_scale".to_string(),
            arr(1),
        );
        // Sidecars that MUST still be dropped.
        params.insert(
            "backbone.layers.0.mixer.experts.up_proj.weight_scale".to_string(),
            arr(1),
        );
        params.insert(
            "backbone.layers.0.mixer.experts.up_proj.weight_scale_2".to_string(),
            arr(1),
        );

        let out = sanitize_weights(params, &config).expect("sanitize");
        assert!(
            out.contains_key("layers.0.mixer.experts.up_proj.global_scale"),
            "stacked-expert .global_scale must survive sanitize: {:?}",
            out.keys().collect::<Vec<_>>()
        );
        assert!(
            out.contains_key("layers.0.mixer.shared_experts.up_proj.global_scale"),
            "shared-expert .global_scale must survive sanitize"
        );
        assert_eq!(
            out["layers.0.mixer.experts.up_proj.global_scale"]
                .shape()
                .unwrap()
                .to_vec(),
            vec![3],
            ".global_scale must not be re-stacked or reshaped"
        );
        assert!(!out.keys().any(|k| k.ends_with(".weight_scale")));
        assert!(!out.keys().any(|k| k.ends_with(".weight_scale_2")));
    }

    /// GUARD 1: a pre-affine checkpoint is detected from its CONFIG, before a
    /// single tensor is read. The mxfp8 and affine payloads share the same
    /// `.weight` bytes, so the declared mode is the only thing that tells them
    /// apart — and the old payload would otherwise load and run 5-7% off.
    #[test]
    fn legacy_mxfp8_mamba_override_is_rejected_with_regenerate_hint() {
        let mxfp8 = PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Mxfp8,
            input_amax: Some(1.0),
        };
        for proj in ["in_proj", "out_proj"] {
            let mut per_layer = HashMap::new();
            per_layer.insert(format!("layers.7.mixer.{proj}"), mxfp8);
            let err = reject_legacy_mxfp8_mamba(&per_layer)
                .expect_err("an mxfp8 mamba projection must be rejected");
            let msg = err.reason;
            assert!(msg.contains(&format!("layers.7.mixer.{proj}")), "{msg}");
            assert!(msg.contains("mxfp8"), "{msg}");
            assert!(msg.contains("mlx convert -m nemotron_h"), "{msg}");
        }

        // The affine replacement, and an mxfp8 override on a NON-mamba key,
        // both pass — the guard must not fire on the qwen/gemma shape or on a
        // correctly regenerated checkpoint.
        let mut ok = HashMap::new();
        ok.insert(
            "layers.7.mixer.in_proj".to_string(),
            PerLayerQuant {
                mode: PerLayerMode::Affine,
                ..mxfp8
            },
        );
        ok.insert("layers.9.mixer.q_proj".to_string(), mxfp8);
        reject_legacy_mxfp8_mamba(&ok).expect("affine mamba + non-mamba mxfp8 must pass");
    }

    /// GUARD 2: an mxfp8-era PAYLOAD under an affine config (hand-edited or
    /// copied) is caught at LOAD with the regenerate hint, not at the first
    /// forward with an opaque MLX throw.
    #[test]
    fn affine_mamba_projection_requires_floating_scales_and_biases() {
        let prefix = "layers.0.mixer.in_proj";
        let packed = MxArray::from_uint32(&[0u32; 8], &[4, 2]).unwrap();
        let f_scales = MxArray::from_float32(&[0.5f32; 4], &[4, 1])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let biases = MxArray::from_float32(&[-1.0f32; 4], &[4, 1])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();

        // Well-formed affine payload.
        let mut ok: HashMap<String, MxArray> = HashMap::new();
        ok.insert(format!("{prefix}.weight"), packed.clone());
        ok.insert(format!("{prefix}.scales"), f_scales.clone());
        ok.insert(format!("{prefix}.biases"), biases.clone());
        require_affine_sidecars(&ok, prefix).expect("bf16 scales + bf16 biases must pass");

        // The superseded mxfp8 payload: U8 E8M0 scales, no biases at all.
        let mut legacy: HashMap<String, MxArray> = HashMap::new();
        legacy.insert(format!("{prefix}.weight"), packed.clone());
        legacy.insert(
            format!("{prefix}.scales"),
            MxArray::from_uint8(&[0x38u8; 4], &[4, 1]).unwrap(),
        );
        let msg = require_affine_sidecars(&legacy, prefix)
            .expect_err("Uint8 scales are the mxfp8 payload")
            .reason;
        assert!(msg.contains("Uint8"), "{msg}");
        assert!(msg.contains("mlx convert -m nemotron_h"), "{msg}");

        // Floating scales but the mandatory .biases missing.
        let mut no_biases: HashMap<String, MxArray> = HashMap::new();
        no_biases.insert(format!("{prefix}.weight"), packed.clone());
        no_biases.insert(format!("{prefix}.scales"), f_scales.clone());
        let msg = require_affine_sidecars(&no_biases, prefix)
            .expect_err("a missing .biases sidecar must be rejected")
            .reason;
        assert!(msg.contains(".biases"), "{msg}");
        assert!(msg.contains("mlx convert -m nemotron_h"), "{msg}");

        // .biases present but shaped differently from .scales.
        let mut skewed: HashMap<String, MxArray> = HashMap::new();
        skewed.insert(format!("{prefix}.weight"), packed);
        skewed.insert(format!("{prefix}.scales"), f_scales);
        skewed.insert(
            format!("{prefix}.biases"),
            MxArray::from_float32(&[-1.0f32; 2], &[2, 1])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap(),
        );
        let msg = require_affine_sidecars(&skewed, prefix)
            .expect_err("a shape-skewed .biases must be rejected")
            .reason;
        assert!(msg.contains("must match"), "{msg}");
    }

    /// GUARD 3: a BF16 router bias is a pre-fix checkpoint. BF16's ULP at the
    /// real bias magnitude (~40.7) is 0.25, larger than the entire 0.06-0.20
    /// inter-expert spread, so the copy collapses all 128 entries onto one
    /// value and DELETES the load-balancing correction.
    #[test]
    fn bf16_router_correction_bias_is_rejected() {
        let mut params = nvfp4_moe_params();
        // Sanity: the fixture is F32, and F32 loads.
        assert_eq!(
            params["layers.0.mixer.gate.e_score_correction_bias"]
                .dtype()
                .unwrap(),
            DType::Float32
        );
        apply_nvfp4(&params).expect("an F32 router bias loads");

        let bf16 = params["layers.0.mixer.gate.e_score_correction_bias"]
            .astype(DType::BFloat16)
            .unwrap();
        params.insert(
            "layers.0.mixer.gate.e_score_correction_bias".to_string(),
            bf16,
        );
        let msg = apply_nvfp4(&params)
            .expect_err("a BF16 router bias must be rejected")
            .reason;
        assert!(msg.contains("e_score_correction_bias"), "{msg}");
        assert!(msg.contains("BF16"), "{msg}");
        assert!(msg.contains("mlx convert -m nemotron_h"), "{msg}");
    }

    /// The modelopt schema's `input_amax` rides on the affine-8/32 mamba
    /// projections now that FP8 maps to `Affine`. It must still be REJECTED on
    /// every other shape — a stale or hand-edited config cannot fake-quant an
    /// uncalibrated projection.
    #[test]
    fn modelopt_input_amax_rides_affine_8_32_and_nothing_else() {
        use serde_json::json;
        let raw = json!({
            "quantization_config": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8", "input_amax": 112.0}
                }
            }
        });
        let quant_cfg = select_quantization_block(&raw).unwrap();
        let (_, per_layer) = parse_nemotron_quant_settings(quant_cfg).unwrap();
        let plq = per_layer["layers.0.mixer.in_proj"];
        assert_eq!(plq.mode, PerLayerMode::Affine);
        assert_eq!(plq.bits, 8);
        assert_eq!(plq.group_size, 32);
        assert_eq!(plq.input_amax, Some(112.0));

        // NVFP4 (4/16) carries no static activation scale.
        let raw = json!({
            "quantization_config": {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "lm_head": {"quant_algo": "W4A16_NVFP4", "input_amax": 112.0}
                }
            }
        });
        let quant_cfg = select_quantization_block(&raw).unwrap();
        let msg = parse_nemotron_quant_settings(quant_cfg)
            .expect_err("input_amax on nvfp4 must be rejected")
            .reason;
        assert!(msg.contains("input_amax"), "{msg}");

        // And the allowlist's exact boundary, checked directly.
        assert!(admits_static_fp8_activation(PerLayerMode::Affine, 8, 32));
        assert!(admits_static_fp8_activation(PerLayerMode::Mxfp8, 8, 32));
        assert!(!admits_static_fp8_activation(PerLayerMode::Affine, 8, 64));
        assert!(!admits_static_fp8_activation(PerLayerMode::Affine, 4, 32));
        assert!(!admits_static_fp8_activation(PerLayerMode::Nvfp4, 8, 32));
        assert!(!admits_static_fp8_activation(PerLayerMode::Sym8, 8, 32));
    }
}
