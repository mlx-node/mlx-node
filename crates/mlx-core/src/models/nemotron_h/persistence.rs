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
//!     shared_experts + lm_head via try_build_nvfp4_quantized_linear
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
    // Stacked-expert builder (3-D [E,N,K]).
    let try_build_qsl = |params: &HashMap<String, MxArray>, prefix: &str| {
        let plq = plq_for(prefix);
        match plq.mode {
            PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_switch_linear(params, prefix),
            _ => None,
        }
    };

    // Embedding (always dense bf16 for this family).
    if let Some(w) = params.get("embedding.weight") {
        crate::models::quant_dispatch::ensure_dense_weight_floating("embedding.weight", w)?;
        inner.embedding.set_weight(w)?;
    }

    // final_norm.
    if let Some(w) = params.get("final_norm.weight") {
        inner.final_norm.set_weight(w)?;
    }

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

        match layer.mamba_mut() {
            Some(m) => {
                if let Some(ql) = try_build_ql(params, &format!("{}.mixer.in_proj", prefix)) {
                    m.in_proj.set_quantized(ql);
                } else if let Some(w) = params.get(&format!("{}.mixer.in_proj.weight", prefix)) {
                    crate::models::quant_dispatch::ensure_dense_weight_floating(
                        &format!("{}.mixer.in_proj.weight", prefix),
                        w,
                    )?;
                    m.in_proj.set_weight(w, "in_proj")?;
                }
                if let Some(ql) = try_build_ql(params, &format!("{}.mixer.out_proj", prefix)) {
                    m.out_proj.set_quantized(ql);
                } else if let Some(w) = params.get(&format!("{}.mixer.out_proj.weight", prefix)) {
                    crate::models::quant_dispatch::ensure_dense_weight_floating(
                        &format!("{}.mixer.out_proj.weight", prefix),
                        w,
                    )?;
                    m.out_proj.set_weight(w, "out_proj")?;
                }
                if let Some(w) = params.get(&format!("{}.mixer.conv1d.weight", prefix)) {
                    m.conv1d.set_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.mixer.conv1d.bias", prefix)) {
                    m.conv1d.set_bias(Some(w))?;
                }
                if let Some(w) = params.get(&format!("{}.mixer.dt_bias", prefix)) {
                    m.dt_bias = w.astype(DType::Float32)?;
                }
                if let Some(w) = params.get(&format!("{}.mixer.A_log", prefix)) {
                    m.a_log = w.astype(DType::Float32)?;
                }
                if let Some(w) = params.get(&format!("{}.mixer.D", prefix)) {
                    m.d = w.astype(DType::Float32)?;
                }
                if let Some(w) = params.get(&format!("{}.mixer.norm.weight", prefix)) {
                    m.norm_weight = w.astype(DType::Float32)?;
                }
            }
            None => {}
        }

        match layer.attention_mut() {
            Some(a) => {
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    let key = format!("{}.mixer.{}.weight", prefix, proj);
                    if let Some(w) = params.get(&key) {
                        crate::models::quant_dispatch::ensure_dense_weight_floating(&key, w)?;
                        match proj {
                            "q_proj" => a.q_proj.set_weight(w, "q_proj")?,
                            "k_proj" => a.k_proj.set_weight(w, "k_proj")?,
                            "v_proj" => a.v_proj.set_weight(w, "v_proj")?,
                            _ => a.o_proj.set_weight(w, "o_proj")?,
                        }
                    }
                }
            }
            None => {}
        }

        match layer.moe_mut() {
            Some(m) => {
                if let Some(w) = params.get(&format!("{}.mixer.gate.weight", prefix)) {
                    m.gate.set_weight(&w.astype(DType::Float32)?, "gate")?;
                }
                if let Some(w) =
                    params.get(&format!("{}.mixer.gate.e_score_correction_bias", prefix))
                {
                    m.e_score_correction_bias = w.astype(DType::Float32)?;
                }
                // Stacked experts (NVFP4 quantized or dense fallback).
                let up_key = format!("{}.mixer.experts.up_proj", prefix);
                let down_key = format!("{}.mixer.experts.down_proj", prefix);
                let up = build_expert_stack(params, &up_key, &try_build_qsl)?;
                let down = build_expert_stack(params, &down_key, &try_build_qsl)?;
                if up.is_some() || down.is_some() {
                    let up = up.ok_or_else(|| {
                        Error::from_reason(format!("MoE layer {i}: missing experts.up_proj weight"))
                    })?;
                    let down = down.ok_or_else(|| {
                        Error::from_reason(format!(
                            "MoE layer {i}: missing experts.down_proj weight"
                        ))
                    })?;
                    m.experts.set_experts(up, down)?;
                }
                // Shared experts (NVFP4 quantized or dense fallback).
                for proj in ["up_proj", "down_proj"] {
                    let key = format!("{}.mixer.shared_experts.{}", prefix, proj);
                    if let Some(ql) = try_build_ql(params, &key) {
                        match proj {
                            "up_proj" => m.shared_experts.up_proj.set_quantized(ql),
                            _ => m.shared_experts.down_proj.set_quantized(ql),
                        }
                    } else if let Some(w) = params.get(&format!("{}.weight", key)) {
                        crate::models::quant_dispatch::ensure_dense_weight_floating(
                            &format!("{}.weight", key),
                            w,
                        )?;
                        match proj {
                            "up_proj" => m.shared_experts.up_proj.set_weight(w, "shared_up")?,
                            _ => m.shared_experts.down_proj.set_weight(w, "shared_down")?,
                        }
                    }
                }
            }
            None => {}
        }

        if let Some(w) = params.get(&format!("{}.norm.weight", prefix)) {
            layer.set_norm_weight(w)?;
        }
    }

    /// Build one stacked expert projection (up or down) from the checkpoint,
    /// returning None when the layer is not an expert layer (no tensors).
    fn build_expert_stack<F>(
        params: &HashMap<String, MxArray>,
        key: &str,
        try_build_qsl: &F,
    ) -> Result<Option<super::sparse_moe::ExpertProj>>
    where
        F: Fn(&HashMap<String, MxArray>, &str) -> Option<QuantizedSwitchLinear>,
    {
        if let Some(qsl) = try_build_qsl(params, key) {
            return Ok(Some(super::sparse_moe::ExpertProj::Quantized(qsl)));
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
            let config = inner.config.clone();
            let scheduler = NemotronHSchedulerState::new(inner)?;
            Ok((
                scheduler,
                (config, cache_limit_guard, mtp_active, paged_active),
            ))
        },
        |state, receiver| state.drive(receiver),
    );

    let (config, cache_limit_guard, mtp_active, paged_active) = init_rx
        .await
        .map_err(|_| Error::from_reason("Model thread exited during load"))??;

    Ok(NemotronHModel {
        thread,
        config,
        mtp_active,
        paged_active,
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
}
