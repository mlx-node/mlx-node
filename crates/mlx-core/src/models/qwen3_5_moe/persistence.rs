use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;
use tracing::{info, warn};

use crate::array::{DType, MxArray};
use crate::models::qwen3_5::persistence::{load_vision_weights, parse_vision_config};
use crate::models::qwen3_5::persistence_common::{
    dequant_fp8_weights, get_config_bool, get_config_f64, get_config_i32, load_all_safetensors,
};
use crate::models::qwen3_5::processing::Qwen35VLImageProcessor;
use crate::models::qwen3_5::vision::Qwen3_5VisionEncoder;
use crate::tokenizer::Qwen3Tokenizer;

use super::config::Qwen3_5MoeConfig;
use super::decoder_layer::{AttentionType, MLPType};
use super::model::{Qwen3_5MoeModel, Qwen35MoeInner, handle_qwen35_moe_cmd};
use super::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, GATE_QUANT_BITS, GATE_QUANT_GROUP_SIZE,
    MLPVariant, QuantizedSwitchLinear, is_mxfp8_checkpoint, is_quantized_checkpoint,
    try_build_mxfp8_quantized_linear, try_build_mxfp8_quantized_switch_linear,
    try_build_quantized_linear,
};
use super::switch_glu::SwitchGLU;

/// Sanitize weights from HuggingFace format.
fn sanitize_weights(
    mut params: HashMap<String, MxArray>,
    config: &Qwen3_5MoeConfig,
) -> Result<HashMap<String, MxArray>> {
    let mut result: HashMap<String, MxArray> = HashMap::new();

    let has_mtp_weights = params.keys().any(|k| k.contains("mtp."));
    let has_unsanitized_conv1d = params.iter().any(|(name, array)| {
        if !name.contains("conv1d.weight") {
            return false;
        }
        match array.shape() {
            Ok(shape) => shape.len() >= 2 && shape[shape.len() - 1] != 1,
            Err(e) => {
                warn!("Failed to read shape of conv1d weight '{}': {}", name, e);
                false
            }
        }
    });
    let needs_norm_fix = has_mtp_weights || has_unsanitized_conv1d;

    // Detect FP8 source checkpoint before dequantization removes scale_inv keys
    let had_fp8 = params.keys().any(|k| k.ends_with("weight_scale_inv"));

    // FP8 dequantization pass — must run before expert stacking and gate_up_proj splitting
    // because FP8 weights are 2D individual expert weights with paired scale_inv tensors
    dequant_fp8_weights(&mut params, DType::BFloat16)?;
    if had_fp8 {
        crate::array::memory::synchronize_and_clear_cache();
    }

    let has_individual_experts = params.keys().any(|k| {
        k.contains(".mlp.experts.0.up_proj.weight")
            || k.contains("model.layers.0.mlp.experts.0.up_proj.weight")
    });

    let mut expert_weights: HashMap<String, Vec<(usize, MxArray)>> = HashMap::new();

    let norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "final_norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        // NOTE: .linear_attn.norm.weight is intentionally NOT included here.
        // It's stored as f32 with final values (e.g. ~0.87), not as shifted weights.
        // Only standard layer/attention norms need the +1.0 shift for MTP checkpoints.
    ];

    for (name, array) in params.drain() {
        if name.contains("mtp.") {
            continue;
        }
        if name.contains("model.visual") || name.contains("visual_encoder") {
            continue;
        }

        let name = name
            .strip_prefix("model.language_model.")
            .or_else(|| name.strip_prefix("language_model.model."))
            .or_else(|| name.strip_prefix("language_model."))
            .or_else(|| name.strip_prefix("model."))
            .unwrap_or(&name)
            .to_string();

        // Rename special keys (including quantization metadata .scales/.biases)
        let name = if let Some(suffix) = name.strip_prefix("embed_tokens.") {
            format!("embedding.{}", suffix)
        } else if name == "norm.weight" {
            "final_norm.weight".to_string()
        } else {
            name
        };

        if config.tie_word_embeddings && name.starts_with("lm_head.") {
            continue;
        }

        // Handle individually-listed expert weights
        if has_individual_experts && name.contains(".mlp.experts.") {
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() >= 7 {
                let layer = parts[1];
                let expert_idx: usize = parts[4].parse().map_err(|e| {
                    Error::from_reason(format!(
                        "Failed to parse expert index from weight '{}': {}",
                        name, e
                    ))
                })?;
                let proj_name = parts[5];
                let key = format!("layers.{}.mlp.switch_mlp.{}.weight", layer, proj_name);
                expert_weights
                    .entry(key)
                    .or_default()
                    .push((expert_idx, array));
                continue;
            }
        }

        // Handle fused gate_up_proj split
        if name.contains(".mlp.experts.gate_up_proj")
            || name.contains(".mlp.switch_mlp.gate_up_proj")
        {
            let suffix = if name.ends_with(".weight") {
                ".weight"
            } else if name.ends_with(".scales") {
                ".scales"
            } else if name.ends_with(".biases") {
                ".biases"
            } else {
                ".weight"
            };

            let shape = array.shape()?;
            if shape.len() >= 2 {
                let split_axis = shape.len() - 2;
                let mid = shape[split_axis] / 2;
                let gate = array.slice_axis(split_axis, 0, mid)?;
                let up = array.slice_axis(split_axis, mid, shape[split_axis])?;

                let name_stripped = name.strip_suffix(suffix).unwrap_or(&name);
                let base = if name_stripped.contains("switch_mlp") {
                    format!(
                        "{}{}",
                        name_stripped.replace("gate_up_proj", "gate_proj"),
                        suffix
                    )
                } else {
                    format!(
                        "{}{}",
                        name_stripped.replace("experts.gate_up_proj", "switch_mlp.gate_proj"),
                        suffix
                    )
                };
                let up_name = base.replace("gate_proj", "up_proj");

                result.insert(base, gate);
                result.insert(up_name, up);
                continue;
            } else {
                let shape_vec: Vec<i64> = shape.to_vec();
                warn!(
                    "gate_up_proj tensor '{}' has unexpected shape {:?}, skipping split",
                    name, shape_vec
                );
            }
        }

        // Handle experts.down_proj rename
        let name = if name.contains(".mlp.experts.down_proj") {
            let suffix = if name.ends_with(".scales") {
                ".scales"
            } else if name.ends_with(".biases") {
                ".biases"
            } else {
                ".weight"
            };
            let stripped = name.strip_suffix(suffix).unwrap_or(&name);
            format!(
                "{}{}",
                stripped.replace(".mlp.experts.down_proj", ".mlp.switch_mlp.down_proj"),
                suffix
            )
        } else {
            name
        };

        // Fix conv1d weight axis
        let array = if name.contains("conv1d.weight") {
            let shape = array.shape()?;
            if shape.len() == 3 && shape[2] != 1 {
                array.transpose(Some(&[0, 2, 1]))?
            } else {
                array
            }
        } else {
            array
        };

        // Apply norm +1.0 fix
        let array = if needs_norm_fix && norm_suffixes.iter().any(|sfx| name.ends_with(sfx)) {
            let ndim = array.ndim()?;
            if ndim == 1 {
                let one = MxArray::scalar_float(1.0)?.astype(array.dtype()?)?;
                array.add(&one)?
            } else {
                array
            }
        } else {
            array
        };

        result.insert(name, array);
    }

    // Stack individual expert weights
    if !expert_weights.is_empty() {
        let num_experts = config.num_experts as usize;
        for (key, mut experts) in expert_weights {
            experts.sort_by_key(|(idx, _)| *idx);

            if num_experts > 0 && experts.len() != num_experts {
                return Err(Error::from_reason(format!(
                    "Expected {} experts for {}, got {}",
                    num_experts,
                    key,
                    experts.len()
                )));
            }

            let arrays: Vec<&MxArray> = experts.iter().map(|(_, a)| a).collect();
            let stacked = MxArray::stack(arrays, Some(0))?;
            result.insert(key, stacked);
        }
    }

    crate::models::qwen3_5::persistence::merge_split_projections(&mut result)?;

    // For FP8 source checkpoints, keep dequantized bf16 weights as-is.
    // Re-quantizing (FP8→bf16→4bit or →MXFP8) compounds quantization error
    // and produces gibberish. mlx-lm also keeps FP8-dequanted weights as bf16.

    Ok(result)
}

fn try_build_quantized_switch_linear(
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
        "affine".to_string(),
    ))
}

/// Apply weights directly to a Qwen35MoeInner (no locks needed).
///
/// Accesses inner fields directly (no `Arc<RwLock<>>`). Used by
/// `load_with_thread`.
fn apply_weights_moe_inner(
    inner: &mut Qwen35MoeInner,
    params: &HashMap<String, MxArray>,
    config: &Qwen3_5MoeConfig,
    quant_bits: i32,
    quant_group_size: i32,
    per_layer_quant: &HashMap<String, (i32, i32)>,
) -> Result<()> {
    let is_quantized = is_quantized_checkpoint(params);
    let is_mxfp8 = is_mxfp8_checkpoint(params);

    // Helper: try MXFP8 builder first (if applicable), then affine builder.
    let try_build_ql = |params: &HashMap<String, MxArray>, prefix: &str| {
        if is_mxfp8 && let Some(ql) = try_build_mxfp8_quantized_linear(params, prefix) {
            return Some(ql);
        }
        let (bits, gs) = per_layer_quant
            .get(prefix)
            .copied()
            .or_else(|| {
                if prefix.ends_with(".in_proj_qkvz") {
                    let base = prefix.strip_suffix(".in_proj_qkvz").unwrap();
                    let qkv = per_layer_quant.get(&format!("{}.in_proj_qkv", base));
                    let z = per_layer_quant.get(&format!("{}.in_proj_z", base));
                    match (qkv, z) {
                        (Some(&a), Some(&b)) if a != b => {
                            warn!(
                                "Merged in_proj_qkvz has conflicting overrides: \
                                 qkv={:?}, z={:?}. Using higher precision.",
                                a, b
                            );
                            Some(if a.0 > b.0 { a } else { b })
                        }
                        (Some(&a), _) | (_, Some(&a)) => Some(a),
                        _ => None,
                    }
                } else if prefix.ends_with(".in_proj_ba") {
                    let base = prefix.strip_suffix(".in_proj_ba").unwrap();
                    let b_val = per_layer_quant.get(&format!("{}.in_proj_b", base));
                    let a_val = per_layer_quant.get(&format!("{}.in_proj_a", base));
                    match (b_val, a_val) {
                        (Some(&x), Some(&y)) if x != y => {
                            warn!(
                                "Merged in_proj_ba has conflicting overrides: \
                                 b={:?}, a={:?}. Using higher precision.",
                                x, y
                            );
                            Some(if x.0 > y.0 { x } else { y })
                        }
                        (Some(&x), _) | (_, Some(&x)) => Some(x),
                        _ => None,
                    }
                } else {
                    None
                }
            })
            .unwrap_or((quant_bits, quant_group_size));
        try_build_quantized_linear(params, prefix, gs, bits)
    };

    // Router gates: check per-layer overrides first, fallback to 8-bit affine
    let try_build_ql_gate = |params: &HashMap<String, MxArray>, prefix: &str| {
        let (bits, gs) = per_layer_quant
            .get(prefix)
            .copied()
            .unwrap_or((GATE_QUANT_BITS, GATE_QUANT_GROUP_SIZE));
        try_build_quantized_linear(params, prefix, gs, bits)
    };

    let try_build_qsl = |params: &HashMap<String, MxArray>, prefix: &str| {
        if is_mxfp8 && let Some(ql) = try_build_mxfp8_quantized_switch_linear(params, prefix) {
            return Some(ql);
        }
        let (bits, gs) = per_layer_quant
            .get(prefix)
            .copied()
            .unwrap_or((quant_bits, quant_group_size));
        try_build_quantized_switch_linear(params, prefix, gs, bits)
    };

    // Embedding — supports both dense and quantized weights
    if let Some(scales) = params.get("embedding.scales") {
        let weight = params.get("embedding.weight").ok_or_else(|| {
            Error::from_reason("Missing embedding.weight for quantized embedding")
        })?;
        let biases = params.get("embedding.biases");
        let (bits, gs) = per_layer_quant
            .get("embed_tokens")
            .copied()
            .unwrap_or((quant_bits, quant_group_size));
        inner
            .embedding
            .load_quantized(weight, scales, biases, gs, bits)?;
        info!("Loaded quantized embedding ({}-bit)", bits);
    } else if let Some(w) = params.get("embedding.weight") {
        inner.embedding.set_weight(w)?;
    }

    // final_norm — direct access, no lock
    if let Some(w) = params.get("final_norm.weight") {
        inner.final_norm.set_weight(w)?;
    }

    // lm_head — direct access, no lock
    if is_quantized {
        if let Some(ql) = try_build_ql(params, "lm_head") {
            inner.lm_head = Some(super::quantized_linear::LinearProj::Quantized(ql));
        } else if let Some(ref mut head) = inner.lm_head
            && let Some(w) = params.get("lm_head.weight")
        {
            head.set_weight(w, "lm_head")?;
        }
    } else if let Some(ref mut head) = inner.lm_head
        && let Some(w) = params.get("lm_head.weight")
    {
        head.set_weight(w, "lm_head")?;
    }

    // Layers — direct access, no lock
    for (i, layer) in inner.layers.iter_mut().enumerate() {
        let prefix = format!("layers.{}", i);

        // Attention weights
        match &mut layer.attn {
            AttentionType::Linear(gdn) => {
                if is_quantized {
                    if let Some(ql) =
                        try_build_ql(params, &format!("{}.linear_attn.in_proj_qkvz", prefix))
                    {
                        gdn.set_quantized_in_proj_qkvz(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.linear_attn.in_proj_qkvz.weight", prefix))
                    {
                        gdn.set_in_proj_qkvz_weight(w)?;
                    }

                    if let Some(ql) =
                        try_build_ql(params, &format!("{}.linear_attn.in_proj_ba", prefix))
                    {
                        gdn.set_quantized_in_proj_ba(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.linear_attn.in_proj_ba.weight", prefix))
                    {
                        gdn.set_in_proj_ba_weight(w)?;
                    }

                    if let Some(ql) =
                        try_build_ql(params, &format!("{}.linear_attn.out_proj", prefix))
                    {
                        gdn.set_quantized_out_proj(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.linear_attn.out_proj.weight", prefix))
                    {
                        gdn.set_out_proj_weight(w)?;
                    }
                } else {
                    if let Some(w) =
                        params.get(&format!("{}.linear_attn.in_proj_qkvz.weight", prefix))
                    {
                        gdn.set_in_proj_qkvz_weight(w)?;
                    }
                    if let Some(w) =
                        params.get(&format!("{}.linear_attn.in_proj_qkv.weight", prefix))
                    {
                        if let Some(z) =
                            params.get(&format!("{}.linear_attn.in_proj_z.weight", prefix))
                        {
                            let combined = MxArray::concatenate(w, z, 0)?;
                            gdn.set_in_proj_qkvz_weight(&combined)?;
                        } else {
                            return Err(Error::from_reason(format!(
                                "Layer {}: in_proj_qkv found but in_proj_z missing",
                                i
                            )));
                        }
                    }
                    if let Some(w) =
                        params.get(&format!("{}.linear_attn.in_proj_ba.weight", prefix))
                    {
                        gdn.set_in_proj_ba_weight(w)?;
                    }
                    if let Some(b) = params.get(&format!("{}.linear_attn.in_proj_b.weight", prefix))
                        && let Some(a) =
                            params.get(&format!("{}.linear_attn.in_proj_a.weight", prefix))
                    {
                        let combined = MxArray::concatenate(b, a, 0)?;
                        gdn.set_in_proj_ba_weight(&combined)?;
                    }
                    if let Some(w) = params.get(&format!("{}.linear_attn.out_proj.weight", prefix))
                    {
                        gdn.set_out_proj_weight(w)?;
                    }
                }
                if let Some(w) = params.get(&format!("{}.linear_attn.conv1d.weight", prefix)) {
                    gdn.set_conv1d_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.linear_attn.dt_bias", prefix)) {
                    gdn.set_dt_bias(w);
                }
                if let Some(w) = params.get(&format!("{}.linear_attn.norm.weight", prefix)) {
                    gdn.set_norm_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.linear_attn.A_log", prefix)) {
                    gdn.set_a_log(w)?;
                }
            }
            AttentionType::Full(attn) => {
                if is_quantized {
                    if let Some(ql) = try_build_ql(params, &format!("{}.self_attn.q_proj", prefix))
                    {
                        attn.set_quantized_q_proj(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.self_attn.q_proj.weight", prefix))
                    {
                        attn.set_q_proj_weight(w)?;
                    }
                    if let Some(ql) = try_build_ql(params, &format!("{}.self_attn.k_proj", prefix))
                    {
                        attn.set_quantized_k_proj(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.self_attn.k_proj.weight", prefix))
                    {
                        attn.set_k_proj_weight(w)?;
                    }
                    if let Some(ql) = try_build_ql(params, &format!("{}.self_attn.v_proj", prefix))
                    {
                        attn.set_quantized_v_proj(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.self_attn.v_proj.weight", prefix))
                    {
                        attn.set_v_proj_weight(w)?;
                    }
                    if let Some(ql) = try_build_ql(params, &format!("{}.self_attn.o_proj", prefix))
                    {
                        attn.set_quantized_o_proj(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.self_attn.o_proj.weight", prefix))
                    {
                        attn.set_o_proj_weight(w)?;
                    }
                } else {
                    if let Some(w) = params.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
                        attn.set_q_proj_weight(w)?;
                    }
                    if let Some(w) = params.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
                        attn.set_k_proj_weight(w)?;
                    }
                    if let Some(w) = params.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
                        attn.set_v_proj_weight(w)?;
                    }
                    if let Some(w) = params.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
                        attn.set_o_proj_weight(w)?;
                    }
                }
                if let Some(w) = params.get(&format!("{}.self_attn.q_norm.weight", prefix)) {
                    attn.set_q_norm_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.self_attn.k_norm.weight", prefix)) {
                    attn.set_k_norm_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.self_attn.q_proj.bias", prefix)) {
                    attn.set_q_proj_bias(Some(w))?;
                }
                if let Some(w) = params.get(&format!("{}.self_attn.k_proj.bias", prefix)) {
                    attn.set_k_proj_bias(Some(w))?;
                }
                if let Some(w) = params.get(&format!("{}.self_attn.v_proj.bias", prefix)) {
                    attn.set_v_proj_bias(Some(w))?;
                }
                if let Some(w) = params.get(&format!("{}.self_attn.o_proj.bias", prefix)) {
                    attn.set_o_proj_bias(Some(w))?;
                }
            }
        }

        // MLP weights
        match &mut layer.mlp {
            MLPType::Dense(MLPVariant::Standard(mlp)) => {
                if is_quantized {
                    let gate_key = format!("{}.mlp.gate_proj", prefix);
                    let up_key = format!("{}.mlp.up_proj", prefix);
                    let down_key = format!("{}.mlp.down_proj", prefix);

                    let q_gate = try_build_ql(params, &gate_key);
                    let q_up = try_build_ql(params, &up_key);
                    let q_down = try_build_ql(params, &down_key);

                    if let (Some(qg), Some(qu), Some(qd)) = (q_gate, q_up, q_down) {
                        layer.set_quantized_dense_mlp(qg, qu, qd);
                    } else {
                        if let Some(w) = params.get(&format!("{}.weight", gate_key)) {
                            mlp.set_gate_proj_weight(w)?;
                        }
                        if let Some(w) = params.get(&format!("{}.weight", up_key)) {
                            mlp.set_up_proj_weight(w)?;
                        }
                        if let Some(w) = params.get(&format!("{}.weight", down_key)) {
                            mlp.set_down_proj_weight(w)?;
                        }
                    }
                } else {
                    if let Some(w) = params.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
                        mlp.set_gate_proj_weight(w)?;
                    }
                    if let Some(w) = params.get(&format!("{}.mlp.up_proj.weight", prefix)) {
                        mlp.set_up_proj_weight(w)?;
                    }
                    if let Some(w) = params.get(&format!("{}.mlp.down_proj.weight", prefix)) {
                        mlp.set_down_proj_weight(w)?;
                    }
                }
            }
            MLPType::Dense(MLPVariant::Quantized { .. }) => {}
            MLPType::MoE(moe) => {
                if is_quantized {
                    if let Some(ql) = try_build_ql_gate(params, &format!("{}.mlp.gate", prefix)) {
                        moe.set_quantized_gate(ql);
                    } else if let Some(w) = params.get(&format!("{}.mlp.gate.weight", prefix)) {
                        moe.set_gate_weight(w)?;
                    }

                    let gate_proj_key = format!("{}.mlp.switch_mlp.gate_proj", prefix);
                    let up_proj_key = format!("{}.mlp.switch_mlp.up_proj", prefix);
                    let down_proj_key = format!("{}.mlp.switch_mlp.down_proj", prefix);

                    let q_gate = try_build_qsl(params, &gate_proj_key);
                    let q_up = try_build_qsl(params, &up_proj_key);
                    let q_down = try_build_qsl(params, &down_proj_key);

                    if let (Some(qg), Some(qu), Some(qd)) = (q_gate, q_up, q_down) {
                        let quantized_switch = SwitchGLU::new_quantized(qg, qu, qd);
                        moe.set_switch_mlp(quantized_switch);
                    } else {
                        if let Some(w) = params.get(&format!("{}.weight", gate_proj_key)) {
                            moe.set_switch_mlp_gate_proj_weight(w);
                        }
                        if let Some(w) = params.get(&format!("{}.weight", up_proj_key)) {
                            moe.set_switch_mlp_up_proj_weight(w);
                        }
                        if let Some(w) = params.get(&format!("{}.weight", down_proj_key)) {
                            moe.set_switch_mlp_down_proj_weight(w);
                        }
                    }

                    let se_gate_key = format!("{}.mlp.shared_expert.gate_proj", prefix);
                    let se_up_key = format!("{}.mlp.shared_expert.up_proj", prefix);
                    let se_down_key = format!("{}.mlp.shared_expert.down_proj", prefix);

                    let q_se_gate = try_build_ql(params, &se_gate_key);
                    let q_se_up = try_build_ql(params, &se_up_key);
                    let q_se_down = try_build_ql(params, &se_down_key);

                    if let (Some(qg), Some(qu), Some(qd)) = (q_se_gate, q_se_up, q_se_down) {
                        moe.set_quantized_shared_expert(qg, qu, qd);
                    } else {
                        if let Some(w) = params.get(&format!("{}.weight", se_gate_key)) {
                            moe.set_shared_expert_gate_proj_weight(w)?;
                        }
                        if let Some(w) = params.get(&format!("{}.weight", se_up_key)) {
                            moe.set_shared_expert_up_proj_weight(w)?;
                        }
                        if let Some(w) = params.get(&format!("{}.weight", se_down_key)) {
                            moe.set_shared_expert_down_proj_weight(w)?;
                        }
                    }

                    if let Some(ql) =
                        try_build_ql_gate(params, &format!("{}.mlp.shared_expert_gate", prefix))
                    {
                        moe.set_quantized_shared_expert_gate(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.mlp.shared_expert_gate.weight", prefix))
                    {
                        moe.set_shared_expert_gate_weight(w)?;
                    }
                } else {
                    if let Some(w) = params.get(&format!("{}.mlp.gate.weight", prefix)) {
                        moe.set_gate_weight(w)?;
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.switch_mlp.gate_proj.weight", prefix))
                    {
                        moe.set_switch_mlp_gate_proj_weight(w);
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.switch_mlp.up_proj.weight", prefix))
                    {
                        moe.set_switch_mlp_up_proj_weight(w);
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.switch_mlp.down_proj.weight", prefix))
                    {
                        moe.set_switch_mlp_down_proj_weight(w);
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.shared_expert.gate_proj.weight", prefix))
                    {
                        moe.set_shared_expert_gate_proj_weight(w)?;
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.shared_expert.up_proj.weight", prefix))
                    {
                        moe.set_shared_expert_up_proj_weight(w)?;
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.shared_expert.down_proj.weight", prefix))
                    {
                        moe.set_shared_expert_down_proj_weight(w)?;
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.shared_expert_gate.weight", prefix))
                    {
                        moe.set_shared_expert_gate_weight(w)?;
                    }
                }
            }
        }

        // Layer norms
        if let Some(w) = params.get(&format!("{}.input_layernorm.weight", prefix)) {
            layer.set_input_layernorm_weight(w)?;
        }
        if let Some(w) = params.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            layer.set_post_attention_layernorm_weight(w)?;
        }
    }

    // Verify mandatory weights
    let mut missing_mandatory = Vec::new();
    if !params.contains_key("embedding.weight") {
        missing_mandatory.push("embedding.weight".to_string());
    }
    if !params.contains_key("final_norm.weight") {
        missing_mandatory.push("final_norm.weight".to_string());
    }
    if !config.tie_word_embeddings
        && !params.contains_key("lm_head.weight")
        && !params.contains_key("lm_head.scales")
    {
        missing_mandatory.push("lm_head.weight".to_string());
    }

    let num_layers = inner.layers.len();
    let mut layers_missing_attn: Vec<usize> = Vec::new();
    let mut layers_missing_mlp: Vec<usize> = Vec::new();

    for i in 0..num_layers {
        let prefix = format!("layers.{}", i);
        let has_attn = params.contains_key(&format!("{}.self_attn.q_proj.weight", prefix))
            || params.contains_key(&format!("{}.self_attn.q_proj.scales", prefix))
            || params.contains_key(&format!("{}.linear_attn.in_proj_qkvz.weight", prefix))
            || params.contains_key(&format!("{}.linear_attn.in_proj_qkvz.scales", prefix))
            || params.contains_key(&format!("{}.linear_attn.in_proj_qkv.weight", prefix))
            || params.contains_key(&format!("{}.linear_attn.in_proj_qkv.scales", prefix));
        if !has_attn {
            layers_missing_attn.push(i);
        }
        let has_mlp = params.contains_key(&format!("{}.mlp.gate_proj.weight", prefix))
            || params.contains_key(&format!("{}.mlp.gate.weight", prefix))
            || params.contains_key(&format!("{}.mlp.gate.scales", prefix))
            || params.contains_key(&format!("{}.mlp.switch_mlp.gate_proj.weight", prefix))
            || params.contains_key(&format!("{}.mlp.switch_mlp.gate_proj.scales", prefix));
        if !has_mlp {
            layers_missing_mlp.push(i);
        }
    }

    if !layers_missing_attn.is_empty() {
        if layers_missing_attn.len() == num_layers {
            missing_mandatory.push("layers.*.attn weights".to_string());
        } else {
            missing_mandatory.push(format!(
                "attention weights for layers {:?} ({}/{})",
                &layers_missing_attn[..layers_missing_attn.len().min(10)],
                layers_missing_attn.len(),
                num_layers
            ));
        }
    }
    if !layers_missing_mlp.is_empty() {
        if layers_missing_mlp.len() == num_layers {
            missing_mandatory.push("layers.*.mlp weights".to_string());
        } else {
            missing_mandatory.push(format!(
                "MLP weights for layers {:?} ({}/{})",
                &layers_missing_mlp[..layers_missing_mlp.len().min(10)],
                layers_missing_mlp.len(),
                num_layers
            ));
        }
    }

    if !missing_mandatory.is_empty() {
        return Err(Error::from_reason(format!(
            "Checkpoint missing mandatory weights: {:?}",
            missing_mandatory
        )));
    }

    let total_weights = params.len();
    info!(
        "Applied weights to inner from checkpoint: {} total in checkpoint",
        total_weights
    );
    Ok(())
}

/// Load a pretrained Qwen3.5 MoE model into a dedicated model thread.
///
/// All model state lives on the spawned thread. Returns a thin NAPI shell
/// with the thread handle and model configuration.
pub async fn load_with_thread(model_path: &str) -> Result<Qwen3_5MoeModel> {
    let model_path = model_path.to_string();

    let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_init(
        move || {
            let path = Path::new(&model_path);

            if !path.exists() {
                return Err(Error::from_reason(format!(
                    "Model path does not exist: {}",
                    model_path
                )));
            }

            // Load config
            let config_path = path.join("config.json");
            let config_data = fs::read_to_string(&config_path)
                .map_err(|e| Error::from_reason(format!("Failed to read config: {}", e)))?;
            let raw: Value = serde_json::from_str(&config_data)
                .map_err(|e| Error::from_reason(format!("Failed to parse config: {}", e)))?;

            let config = parse_config(&raw)?;

            info!(
                "Qwen3.5 MoE config: {} layers, hidden={}, experts={}x{}",
                config.num_layers,
                config.hidden_size,
                config.num_experts,
                config.num_experts_per_tok
            );

            // Load all weights
            let raw_params = load_all_safetensors(path, false)?;
            info!("Loaded {} raw tensors", raw_params.len());

            // Split vision/text weights
            let has_vision = raw_params
                .keys()
                .any(|k| k.starts_with("vision_tower.") || k.starts_with("visual."));

            let (text_raw_params, vision_params) = if has_vision {
                let mut vision_params: HashMap<String, MxArray> = HashMap::new();
                let mut text_params: HashMap<String, MxArray> = HashMap::new();
                for (name, array) in raw_params {
                    if name.starts_with("vision_tower.") || name.starts_with("visual.") {
                        let vkey = name
                            .strip_prefix("vision_tower.")
                            .or_else(|| name.strip_prefix("visual."))
                            .unwrap_or(&name)
                            .to_string();
                        vision_params.insert(vkey, array);
                    } else {
                        text_params.insert(name, array);
                    }
                }
                info!(
                    "Split: {} vision tensors, {} text tensors",
                    vision_params.len(),
                    text_params.len()
                );
                (text_params, Some(vision_params))
            } else {
                (raw_params, None)
            };

            // Sanitize weights
            let params = sanitize_weights(text_raw_params, &config)?;
            let quantized = is_quantized_checkpoint(&params);
            info!(
                "Sanitized to {} parameters (quantized={})",
                params.len(),
                quantized
            );

            // Parse quantization config
            let quant_cfg = raw
                .get("quantization")
                .or_else(|| raw.get("quantization_config"));
            let quant_bits = quant_cfg
                .and_then(|q| q["bits"].as_i64())
                .unwrap_or(DEFAULT_QUANT_BITS as i64) as i32;
            let quant_group_size = quant_cfg
                .and_then(|q| q["group_size"].as_i64())
                .unwrap_or(DEFAULT_QUANT_GROUP_SIZE as i64)
                as i32;
            let per_layer_quant: HashMap<String, (i32, i32)> = quant_cfg
                .and_then(|q| q.as_object())
                .map(|obj| {
                    obj.iter()
                        .filter(|(_, v)| v.is_object())
                        .filter_map(|(k, v)| {
                            let bits = v["bits"].as_i64()? as i32;
                            let gs =
                                v["group_size"].as_i64().unwrap_or(quant_group_size as i64) as i32;
                            let normalized = k
                                .strip_prefix("model.language_model.")
                                .or_else(|| k.strip_prefix("language_model.model."))
                                .or_else(|| k.strip_prefix("language_model."))
                                .or_else(|| k.strip_prefix("model."))
                                .unwrap_or(k)
                                .to_string();
                            Some((normalized, (bits, gs)))
                        })
                        .collect()
                })
                .unwrap_or_default();

            if quant_cfg.is_some() {
                info!(
                    "Using quantization config: bits={}, group_size={}, per_layer_overrides={}",
                    quant_bits,
                    quant_group_size,
                    per_layer_quant.len()
                );
            }

            // Load tokenizer
            let tokenizer_path = path.join("tokenizer.json");
            let tokenizer = if tokenizer_path.exists() {
                info!("Loading tokenizer from: {}", tokenizer_path.display());
                Some(Qwen3Tokenizer::load_from_file_sync(
                    tokenizer_path.to_str().ok_or_else(|| {
                        Error::from_reason("Tokenizer path contains invalid UTF-8")
                    })?,
                )?)
            } else {
                None
            };

            // Create inner model
            let mut inner = Qwen35MoeInner::new(config.clone())?;

            // Apply weights directly to inner (no locks)
            apply_weights_moe_inner(
                &mut inner,
                &params,
                &config,
                quant_bits,
                quant_group_size,
                &per_layer_quant,
            )?;

            // Register weights with C++ MoE forward pass
            register_moe_weights_with_cpp(&params, inner.model_id);

            // Materialize mmap-backed weights
            {
                let arrays: Vec<&MxArray> = params.values().collect();
                crate::array::memory::materialize_weights(&arrays);
            }

            // Set tokenizer
            if let Some(tok) = tokenizer {
                inner.set_tokenizer(Arc::new(tok));
            }

            // Load vision encoder if present
            if let Some(ref vparams) = vision_params {
                let vision_config = parse_vision_config(&raw);
                info!(
                    "Vision config: {} layers, hidden={}, heads={}, patch={}",
                    vision_config.num_layers,
                    vision_config.hidden_size,
                    vision_config.num_heads,
                    vision_config.patch_size,
                );

                let mut vision_encoder = Qwen3_5VisionEncoder::new(vision_config.clone())?;
                load_vision_weights(&mut vision_encoder, vparams, &vision_config)?;

                inner.init_mrope_layers(
                    vec![11, 11, 10],
                    config.rope_theta,
                    config.max_position_embeddings,
                )?;

                inner.set_vision_encoder(vision_encoder);
                inner.set_image_processor(Qwen35VLImageProcessor::new(None));
                inner.set_spatial_merge_size(vision_config.spatial_merge_size);

                info!("Qwen3.5 MoE-VL model loaded successfully (with vision encoder)");
            } else {
                info!("Qwen3.5 MoE model loaded successfully");
            }

            let model_id = inner.model_id;
            let config_out = inner.config.clone();
            let image_processor = inner.image_processor.as_ref().map(Arc::clone);
            let tokenizer_out = inner.tokenizer.clone();

            Ok((
                inner,
                (config_out, model_id, image_processor, tokenizer_out),
            ))
        },
        handle_qwen35_moe_cmd,
    );

    let (config, model_id, _image_processor, _tokenizer) = init_rx
        .await
        .map_err(|_| Error::from_reason("Model thread exited during load"))??;

    Ok(Qwen3_5MoeModel {
        thread,
        config,
        model_id,
    })
}

/// Parse Qwen3.5 MoE config from JSON.
fn parse_config(raw: &Value) -> Result<Qwen3_5MoeConfig> {
    let text_cfg = raw.get("text_config");

    let gi = |keys: &[&str], default: i32| get_config_i32(raw, text_cfg, keys, default);
    let gf = |keys: &[&str], default: f64| get_config_f64(raw, text_cfg, keys, default);
    let gb = |keys: &[&str], default: bool| get_config_bool(raw, text_cfg, keys, default);

    let hidden_size = gi(&["hidden_size"], 0);
    let num_heads = gi(&["num_attention_heads", "num_heads"], 0);

    let head_dim = text_cfg
        .and_then(|tc| tc["head_dim"].as_i64())
        .or_else(|| raw["head_dim"].as_i64())
        .map(|v| v as i32)
        .unwrap_or_else(|| {
            if num_heads > 0 {
                hidden_size / num_heads
            } else {
                128
            }
        });

    let rope_obj = text_cfg
        .and_then(|tc| tc.get("rope_parameters"))
        .or_else(|| raw.get("rope_parameters"));

    let partial_rotary_factor = rope_obj
        .and_then(|rp| rp["partial_rotary_factor"].as_f64())
        .unwrap_or_else(|| gf(&["partial_rotary_factor"], 0.25));

    let rope_theta = rope_obj
        .and_then(|rp| rp["rope_theta"].as_f64())
        .unwrap_or_else(|| gf(&["rope_theta"], 100_000.0));

    let bos_token_id = gi(&["bos_token_id"], 151643);
    let num_layers = gi(&["num_hidden_layers", "num_layers"], 0);
    let intermediate_size = gi(&["intermediate_size"], 0);
    let num_kv_heads = gi(&["num_key_value_heads", "num_kv_heads"], 8);
    let vocab_size = gi(&["vocab_size"], 151936);

    let num_experts = gi(&["num_experts"], 0);
    let num_experts_per_tok = gi(&["num_experts_per_tok"], 1);
    let moe_i = gi(&["moe_intermediate_size"], 0);

    if hidden_size <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid config: hidden_size must be > 0, got {}",
            hidden_size
        )));
    }
    if num_layers <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid config: num_hidden_layers must be > 0, got {}",
            num_layers
        )));
    }
    if num_experts <= 0 {
        return Err(Error::from_reason(format!(
            "MoE config requires num_experts > 0, got {}",
            num_experts
        )));
    }
    if intermediate_size <= 0 && moe_i <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid config: intermediate_size ({}) or moe_intermediate_size ({}) must be > 0",
            intermediate_size, moe_i
        )));
    }

    Ok(Qwen3_5MoeConfig {
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        intermediate_size,
        rms_norm_eps: gf(&["rms_norm_eps"], 1e-6),
        head_dim,
        tie_word_embeddings: gb(&["tie_word_embeddings"], false),
        attention_bias: gb(&["attention_bias"], false),
        max_position_embeddings: gi(&["max_position_embeddings"], 131072),
        pad_token_id: gi(&["pad_token_id"], bos_token_id),
        eos_token_id: gi(&["eos_token_id"], 151645),
        bos_token_id,

        linear_num_value_heads: gi(&["linear_num_value_heads"], 64),
        linear_num_key_heads: gi(&["linear_num_key_heads"], 16),
        linear_key_head_dim: gi(&["linear_key_head_dim"], 192),
        linear_value_head_dim: gi(&["linear_value_head_dim"], 128),
        linear_conv_kernel_dim: gi(&["linear_conv_kernel_dim"], 4),
        full_attention_interval: gi(&["full_attention_interval"], 4),
        partial_rotary_factor,
        rope_theta,

        num_experts,
        num_experts_per_tok,
        decoder_sparse_step: gi(&["decoder_sparse_step"], 1),
        shared_expert_intermediate_size: {
            let v = gi(&["shared_expert_intermediate_size"], 0);
            if v > 0 { Some(v) } else { None }
        },
        moe_intermediate_size: { if moe_i > 0 { Some(moe_i) } else { None } },
        norm_topk_prob: gb(&["norm_topk_prob"], true),
        mlp_only_layers: text_cfg
            .and_then(|tc| tc["mlp_only_layers"].as_array())
            .or_else(|| raw["mlp_only_layers"].as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64().map(|i| i as i32))
                    .collect()
            }),
    })
}

/// Register all sanitized weights with the C++ MoE forward pass.
/// Uses the same shared g_weights map as the dense path (mlx_store_weight).
/// Sets model_id AFTER all weights are stored.
fn register_moe_weights_with_cpp(params: &HashMap<String, MxArray>, model_id: u64) {
    use mlx_sys as sys;
    use std::ffi::CString;

    // Write-lock the weight RwLock for the entire registration.
    let _guard = crate::models::qwen3_5::model::COMPILED_WEIGHTS_RWLOCK
        .write()
        .unwrap();

    // Clear weights (shared map)
    unsafe { sys::mlx_clear_weights() };

    let store = |name: &str, array: &MxArray| {
        let c_name = CString::new(name).expect("Weight name contains null byte");
        unsafe {
            sys::mlx_store_weight(c_name.as_ptr(), array.as_raw_ptr());
        }
    };

    // Projections are already merged by sanitize_weights → merge_split_projections
    // (handles both bf16 concat and quantized scales/biases concat correctly).
    // Just store all params directly.
    for (name, array) in params {
        store(name, array);
    }

    let count = unsafe { sys::mlx_weight_count() };
    info!("Registered {} weights with C++ MoE forward pass", count);

    // Set model ID AFTER all weights are stored.
    unsafe { sys::mlx_set_model_id(model_id) };
}

/// Create a random-init Qwen3.5 MoE model and save it to disk.
///
/// Spawns a dedicated `ModelThread<Qwen35MoeCmd>` whose init builds a fresh
/// random-weight `Qwen35MoeInner` directly, then dispatches
/// `Qwen35MoeCmd::SaveModel` on that thread. The thread is dropped at the end
/// of the promise, so the in-memory model is released once the checkpoint has
/// been written. Used by TypeScript test fixtures that need an on-disk
/// checkpoint without keeping a NAPI model instance alive.
#[napi]
pub fn create_random_qwen35_moe_checkpoint<'env>(
    env: &'env Env,
    config: Qwen3_5MoeConfig,
    save_path: String,
) -> Result<PromiseRaw<'env, ()>> {
    use super::model::Qwen35MoeCmd;

    let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_init(
        move || {
            let inner = Qwen35MoeInner::new(config)?;
            Ok((inner, ()))
        },
        handle_qwen35_moe_cmd,
    );

    env.spawn_future(async move {
        init_rx
            .await
            .map_err(|_| napi::Error::from_reason("Model thread exited during init"))??;

        let (tx, rx) = tokio::sync::oneshot::channel();
        thread.send(Qwen35MoeCmd::SaveModel {
            save_path,
            reply: tx,
        })?;
        rx.await
            .map_err(|_| napi::Error::from_reason("Model thread exited unexpectedly"))??;

        // Drop the thread explicitly so the dedicated OS thread shuts down
        // now that the checkpoint has been written.
        drop(thread);
        Ok(())
    })
}
