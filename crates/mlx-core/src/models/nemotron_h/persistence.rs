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
//!     shared_experts + lm_head via try_build_nvfp4_quantized_linear. Every
//!     NVFP4 prefix additionally REQUIRES a Float32 `.global_scale` sidecar
//!     (`[n_routed_experts]` on the stacked experts, a scalar on the shared
//!     experts): convert emits NVIDIA's per-group E4M3 `weight_scale` bytes
//!     verbatim and carries `weight_scale_2` here instead of folding it. The
//!     lookup is FAIL-CLOSED — a missing/mis-typed/mis-shaped one is a hard
//!     error, because defaulting it to 1.0 would leave the projection
//!     ~1/weight_scale_2 (~1.15e4x) too large with no diagnostic.
//!   * MXFP8 - mamba in_proj/out_proj via try_build_mxfp8_quantized_linear
//!     with the layer's input_amax threaded for W8A8 numeric parity
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
    PerLayerMode, PerLayerQuant, default_per_layer_quant, effective_plq_for, parse_quant_block,
    select_quantization_block,
};
use crate::models::qwen3_5::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, is_quantized_checkpoint,
    try_build_mxfp8_quantized_linear, try_build_nvfp4_quantized_linear, try_build_quantized_linear,
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
fn require_scalar_global_scale(params: &HashMap<String, MxArray>, prefix: &str) -> Result<f64> {
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
    Ok(v)
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
        "FP8" | "W8A8_FP8" => Ok(PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Mxfp8,
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
            if plq.mode != PerLayerMode::Mxfp8 {
                return Err(Error::from_reason(format!(
                    "quantization override '{key}': input_amax is only valid for FP8 layers"
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
    let try_build_ql =
        |params: &HashMap<String, MxArray>, prefix: &str| -> Option<QuantizedLinear> {
            let plq = plq_for(prefix);
            match plq.mode {
                PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_linear(params, prefix),
                PerLayerMode::Mxfp8 => try_build_mxfp8_quantized_linear(params, prefix)
                    .map(|ql| ql.with_input_amax(plq.input_amax)),
                PerLayerMode::Affine => {
                    try_build_quantized_linear(params, prefix, plq.group_size, plq.bits)
                }
                other => {
                    // Fp8E4m3 / sym8 / k-quants are not produced for this family.
                    let _ = other;
                    None
                }
            }
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
        if let Some(ql) = try_build_ql(params, "lm_head") {
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
            if let Some(ql) = try_build_ql(params, &in_key) {
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
            if let Some(ql) = try_build_ql(params, &out_key) {
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
                if let Some(ql) = try_build_ql(params, &key) {
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

        // ---- safetensors: all tensors BF16, deterministic values ----
        let mut tensors: HashMap<String, MxArray> = HashMap::new();
        let mut put = |name: &str, shape: &[i64]| {
            let n: usize = shape.iter().map(|&s| s as usize).product();
            let vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013) % 1.0 - 0.5).collect();
            let arr = MxArray::from_float32(&vals, shape)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            tensors.insert(name.to_string(), arr);
        };

        put("embedding.weight", &[32, 8]);
        put("final_norm.weight", &[8]);
        put("lm_head.weight", &[32, 8]);
        // layer 0: mamba
        put("layers.0.norm.weight", &[8]);
        put("layers.0.mixer.in_proj.weight", &[14, 8]);
        put("layers.0.mixer.conv1d.weight", &[8, 4, 1]);
        put("layers.0.mixer.conv1d.bias", &[8]);
        put("layers.0.mixer.dt_bias", &[2]);
        put("layers.0.mixer.A_log", &[2]);
        put("layers.0.mixer.D", &[2]);
        put("layers.0.mixer.norm.weight", &[4]);
        put("layers.0.mixer.out_proj.weight", &[8, 4]);
        // layer 1: moe (2 experts, stacked by the loader)
        put("layers.1.norm.weight", &[8]);
        put("layers.1.mixer.gate.weight", &[2, 8]);
        put("layers.1.mixer.gate.e_score_correction_bias", &[2]);
        put("layers.1.mixer.experts.0.up_proj.weight", &[6, 8]);
        put("layers.1.mixer.experts.0.down_proj.weight", &[8, 6]);
        put("layers.1.mixer.experts.1.up_proj.weight", &[6, 8]);
        put("layers.1.mixer.experts.1.down_proj.weight", &[8, 6]);
        put("layers.1.mixer.shared_experts.up_proj.weight", &[8, 8]);
        put("layers.1.mixer.shared_experts.down_proj.weight", &[8, 8]);
        // layer 2: attention
        put("layers.2.norm.weight", &[8]);
        put("layers.2.mixer.q_proj.weight", &[8, 8]);
        put("layers.2.mixer.k_proj.weight", &[4, 8]);
        put("layers.2.mixer.v_proj.weight", &[4, 8]);
        put("layers.2.mixer.o_proj.weight", &[8, 8]);

        save_safetensors(dir.join("model.safetensors"), &mut tensors, None)
            .expect("write safetensors");

        let (mut inner, weight_bytes) = load_inner(dir.to_str().unwrap()).expect("load_inner");
        assert!(weight_bytes > 0);
        assert!(!inner.has_mtp_weights());

        // Run one flat forward over 3 tokens.
        let ids = MxArray::from_uint32(&[1, 5, 9], &[1, 3]).unwrap();
        let logits = inner.forward(&ids).expect("forward");
        let shape = logits.shape().unwrap().to_vec();
        assert_eq!(shape, vec![1, 3, 32]);

        // Second forward (decode-style single token) reuses the caches.
        let ids2 = MxArray::from_uint32(&[17], &[1, 1]).unwrap();
        let logits2 = inner.forward(&ids2).expect("forward decode");
        assert_eq!(logits2.shape().unwrap().to_vec(), vec![1, 1, 32]);

        let _ = fs::remove_dir_all(&dir);
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
        assert_eq!(
            per_layer["layers.0.mixer.in_proj"].mode,
            PerLayerMode::Mxfp8
        );
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
        put_bf16(
            "layers.0.mixer.gate.e_score_correction_bias",
            &[E],
            &mut params,
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
}
