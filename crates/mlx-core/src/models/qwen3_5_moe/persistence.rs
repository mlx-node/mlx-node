use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use serde_json::Value;
use tracing::{info, warn};

use crate::array::{DType, MxArray};
use crate::models::qwen3_5::persistence::{load_vision_weights, parse_vision_config};
use crate::models::qwen3_5::processing::Qwen35VLImageProcessor;
use crate::models::qwen3_5::vision::Qwen3_5VisionEncoder;
use crate::tokenizer::Qwen3Tokenizer;
use crate::utils::safetensors::SafeTensorsFile;

use super::config::Qwen3_5MoeConfig;
use super::decoder_layer::{AttentionType, MLPType};
use super::model::Qwen3_5MoeModel;
use super::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, GATE_QUANT_BITS, GATE_QUANT_GROUP_SIZE,
    MLPVariant, QuantizedSwitchLinear, is_mxfp8_checkpoint, is_quantized_checkpoint,
    try_build_mxfp8_quantized_linear, try_build_mxfp8_quantized_switch_linear,
    try_build_quantized_linear,
};
use super::switch_glu::SwitchGLU;

/// Load all safetensors files from a directory.
fn load_all_safetensors(dir: &Path) -> Result<HashMap<String, MxArray>> {
    let single_path = if dir.join("weights.safetensors").exists() {
        Some(dir.join("weights.safetensors"))
    } else if dir.join("model.safetensors").exists() {
        Some(dir.join("model.safetensors"))
    } else {
        None
    };

    if let Some(path) = single_path {
        info!("Loading weights from: {}", path.display());
        let st_file = SafeTensorsFile::load(&path)?;
        return st_file.load_tensors(&path);
    }

    let mut shard_files: Vec<std::path::PathBuf> = Vec::new();
    let entries = fs::read_dir(dir)
        .map_err(|e| Error::from_reason(format!("Failed to read model directory: {}", e)))?;

    for entry in entries {
        let entry = entry
            .map_err(|e| Error::from_reason(format!("Failed to read directory entry: {}", e)))?;
        let name = entry.file_name().to_string_lossy().to_string();
        let is_shard = (name.starts_with("model-") || name.starts_with("model.safetensors-"))
            && name.ends_with(".safetensors")
            && name.contains("-of-");
        if is_shard {
            shard_files.push(entry.path());
        }
    }

    if shard_files.is_empty() {
        return Err(Error::from_reason(format!(
            "No safetensors files found in {}",
            dir.display()
        )));
    }

    shard_files.sort();
    info!("Loading {} sharded safetensors files", shard_files.len());

    let mut all_params: HashMap<String, MxArray> = HashMap::new();
    for shard_path in &shard_files {
        info!("  Loading shard: {}", shard_path.display());
        let st_file = SafeTensorsFile::load(shard_path)?;
        let shard_params = st_file.load_tensors(shard_path)?;
        all_params.extend(shard_params);
    }

    Ok(all_params)
}

/// FP8 E4M3 block-wise dequantization: weight * scale_inv with block_size=128
///
/// Handles both 2D [out, in] and 1D [n] weights.
/// 1. from_fp8(weight) → target dtype
/// 2. Pad to 128-block alignment
/// 3. Reshape into blocks, multiply by scale_inv
/// 4. Unpad and return as target dtype
fn dequant_fp8(weight: &MxArray, scale_inv: &MxArray, target_dtype: DType) -> Result<MxArray> {
    let weight = weight.from_fp8(target_dtype)?;

    let shape = weight.shape()?;
    let shape_ref = shape.as_ref();

    if shape_ref.len() < 2 {
        // 1D weight (e.g. bias): just scale directly
        return weight.mul(scale_inv)?.astype(target_dtype);
    }

    let m = shape_ref[0] as usize;
    let n = shape_ref[1] as usize;
    let bs: usize = 128;

    let pad_bottom = (bs - (m % bs)) % bs;
    let pad_side = (bs - (n % bs)) % bs;

    let weight = if pad_bottom > 0 || pad_side > 0 {
        weight.pad(&[0, pad_bottom as i32, 0, pad_side as i32], 0.0)?
    } else {
        weight
    };

    let m_padded = m + pad_bottom;
    let n_padded = n + pad_side;
    let weight = weight.reshape(&[
        (m_padded / bs) as i64,
        bs as i64,
        (n_padded / bs) as i64,
        bs as i64,
    ])?;

    let scale = scale_inv.expand_dims(1)?.expand_dims(3)?;
    let weight = weight.mul(&scale)?;

    let weight = weight.reshape(&[m_padded as i64, n_padded as i64])?;
    let weight = if pad_bottom > 0 || pad_side > 0 {
        weight.slice(&[0, 0], &[m as i64, n as i64])?
    } else {
        weight
    };

    weight.astype(target_dtype)
}

/// Dequantize all FP8 weight pairs in-place.
/// Finds all `*weight_scale_inv` keys, dequantizes the corresponding weight,
/// removes scale_inv keys, and replaces weights with dequantized versions.
fn dequant_fp8_weights(params: &mut HashMap<String, MxArray>, target_dtype: DType) -> Result<()> {
    let scale_keys: Vec<String> = params
        .keys()
        .filter(|k| k.ends_with("weight_scale_inv"))
        .cloned()
        .collect();

    if scale_keys.is_empty() {
        return Ok(());
    }

    info!(
        "Dequantizing {} FP8 weight pairs to {:?}",
        scale_keys.len(),
        target_dtype
    );

    for scale_key in scale_keys {
        let weight_key = scale_key.replace("_scale_inv", "");
        let scale_inv = params
            .remove(&scale_key)
            .expect("scale_key must exist in params");
        if let Some(weight) = params.remove(&weight_key) {
            let dequantized = dequant_fp8(&weight, &scale_inv, target_dtype)?;
            // Eval immediately to prevent lazy chain accumulation (OOM with ~31K FP8 pairs)
            dequantized.eval();
            params.insert(weight_key, dequantized);
        }
    }

    Ok(())
}

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

        let name = if name == "embed_tokens.weight" {
            "embedding.weight".to_string()
        } else if name == "norm.weight" {
            "final_norm.weight".to_string()
        } else {
            name
        };

        if config.tie_word_embeddings && name == "lm_head.weight" {
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

/// Apply weights to a Qwen3.5 MoE model.
fn apply_weights(
    model: &mut Qwen3_5MoeModel,
    params: &HashMap<String, MxArray>,
    config: &Qwen3_5MoeConfig,
    quant_bits: i32,
    quant_group_size: i32,
    per_layer_quant: &HashMap<String, (i32, i32)>,
) -> Result<()> {
    let is_quantized = is_quantized_checkpoint(params);
    let is_mxfp8 = is_mxfp8_checkpoint(params);

    // Helper: try MXFP8 builder first (if applicable), then affine builder.
    // Checks per-layer overrides before falling back to global defaults.
    let try_build_ql = |params: &HashMap<String, MxArray>, prefix: &str| {
        if is_mxfp8 && let Some(ql) = try_build_mxfp8_quantized_linear(params, prefix) {
            return Some(ql);
        }
        let (bits, gs) = per_layer_quant
            .get(prefix)
            .copied()
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

    if let Some(w) = params.get("embedding.weight") {
        model.embedding.set_weight(w)?;
    }

    {
        let mut final_norm = model
            .final_norm
            .write()
            .map_err(|_| Error::from_reason("Failed to acquire final_norm write lock"))?;
        if let Some(w) = params.get("final_norm.weight") {
            final_norm.set_weight(w)?;
        }
    }

    {
        let mut lm_head = model
            .lm_head
            .write()
            .map_err(|_| Error::from_reason("Failed to acquire lm_head write lock"))?;
        if is_quantized {
            if let Some(ql) = try_build_ql(params, "lm_head") {
                *lm_head = Some(super::quantized_linear::LinearProj::Quantized(ql));
            } else if let Some(ref mut head) = *lm_head
                && let Some(w) = params.get("lm_head.weight")
            {
                head.set_weight(w, "lm_head")?;
            }
        } else if let Some(ref mut head) = *lm_head
            && let Some(w) = params.get("lm_head.weight")
        {
            head.set_weight(w, "lm_head")?;
        }
    }

    let mut layers = model
        .layers
        .write()
        .map_err(|_| Error::from_reason("Failed to acquire layers write lock"))?;
    for (i, layer) in layers.iter_mut().enumerate() {
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
                // dt_bias must be loaded first (bf16) — it's used as dtype reference
                // for A_log and norm.weight which are stored as f32 in checkpoints
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
                    // Router gate: 8-bit for routing accuracy
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

                    // Shared expert gate: 8-bit for routing accuracy
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

    let num_layers = layers.len();
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
        "Applied weights from checkpoint: {} total in checkpoint",
        total_weights
    );
    Ok(())
}

/// Load a pretrained Qwen3.5 MoE model from a directory.
pub async fn load_pretrained(model_path: &str) -> Result<Qwen3_5MoeModel> {
    let model_path = model_path.to_string();

    napi::tokio::task::spawn_blocking(move || {
        let path = Path::new(&model_path);

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
            "Qwen3.5 MoE config: {} layers, hidden={}, experts={}x{}",
            config.num_layers, config.hidden_size, config.num_experts, config.num_experts_per_tok
        );

        let raw_params = load_all_safetensors(path)?;
        info!("Loaded {} raw tensors", raw_params.len());

        // Check for vision weights and split if present
        let has_vision = raw_params
            .keys()
            .any(|k| k.starts_with("vision_tower.") || k.starts_with("visual."));

        let (text_raw_params, vision_params) = if has_vision {
            let mut vision_params: HashMap<String, MxArray> = HashMap::new();
            let mut text_params: HashMap<String, MxArray> = HashMap::new();

            for (name, array) in raw_params {
                if name.starts_with("vision_tower.") || name.starts_with("visual.") {
                    // Normalize vision key: strip prefix
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

        let params = sanitize_weights(text_raw_params, &config)?;
        let quantized = is_quantized_checkpoint(&params);
        info!(
            "Sanitized to {} parameters (quantized={})",
            params.len(),
            quantized
        );

        // Parse quantization config from config.json (our format or mlx-lm compat)
        let quant_cfg = raw
            .get("quantization")
            .or_else(|| raw.get("quantization_config"));
        let quant_bits = quant_cfg
            .and_then(|q| q["bits"].as_i64())
            .unwrap_or(DEFAULT_QUANT_BITS as i64) as i32;
        let quant_group_size = quant_cfg
            .and_then(|q| q["group_size"].as_i64())
            .unwrap_or(DEFAULT_QUANT_GROUP_SIZE as i64) as i32;
        // Parse per-layer quantization overrides (module path → (bits, group_size)).
        // Normalize keys by stripping model prefixes so they match the sanitized
        // weight keys used in apply_weights (e.g. "layers.0.mlp.gate").
        let per_layer_quant: HashMap<String, (i32, i32)> = quant_cfg
            .and_then(|q| q.as_object())
            .map(|obj| {
                obj.iter()
                    .filter(|(_, v)| v.is_object()) // per-layer entries are objects, globals are scalars
                    .filter_map(|(k, v)| {
                        let bits = v["bits"].as_i64()? as i32;
                        let gs = v["group_size"]
                            .as_i64()
                            .unwrap_or(quant_group_size as i64)
                            as i32;
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
                "Using quantization config from config.json: bits={}, group_size={}, per_layer_overrides={}",
                quant_bits, quant_group_size, per_layer_quant.len()
            );
        }

        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            info!("Loading tokenizer from: {}", tokenizer_path.display());
            Some(Qwen3Tokenizer::load_from_file_sync(
                tokenizer_path
                    .to_str()
                    .ok_or_else(|| Error::from_reason("Tokenizer path contains invalid UTF-8"))?,
            )?)
        } else {
            None
        };

        let mut model = Qwen3_5MoeModel::new(config.clone())?;
        apply_weights(
            &mut model,
            &params,
            &config,
            quant_bits,
            quant_group_size,
            &per_layer_quant,
        )?;

        // Register weights with C++ MoE forward pass.
        // Works for both quantized and unquantized models (C++ detects quantization at init).
        register_moe_weights_with_cpp(&params);

        if let Some(tok) = tokenizer {
            model.tokenizer = Some(Arc::new(tok));
        }

        // If vision weights were found, load vision encoder and configure VLM
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

            // Initialize M-RoPE on all full attention layers
            // mrope_section = [11, 11, 10] for Qwen3.5-VL
            model.init_mrope_layers(
                vec![11, 11, 10],
                config.rope_theta,
                config.max_position_embeddings,
            )?;

            model.set_vision_encoder(vision_encoder);
            model.set_image_processor(Qwen35VLImageProcessor::new(None));
            model.set_spatial_merge_size(vision_config.spatial_merge_size);

            info!("Qwen3.5 MoE-VL model loaded successfully (with vision encoder)");
        } else {
            info!("Qwen3.5 MoE model loaded successfully");
        }

        Ok(model)
    })
    .await
    .map_err(|e| Error::from_reason(format!("Failed to load model: {}", e)))?
}

/// Parse Qwen3.5 MoE config from JSON.
fn parse_config(raw: &Value) -> Result<Qwen3_5MoeConfig> {
    let text_cfg = raw.get("text_config");

    let get_i32 = |keys: &[&str], default: i32| -> i32 {
        for key in keys {
            if let Some(tc) = text_cfg
                && let Some(v) = tc[key].as_i64()
            {
                return v as i32;
            }
            if let Some(v) = raw[key].as_i64() {
                return v as i32;
            }
        }
        default
    };

    let get_f64 = |keys: &[&str], default: f64| -> f64 {
        for key in keys {
            if let Some(tc) = text_cfg
                && let Some(v) = tc[key].as_f64()
            {
                return v;
            }
            if let Some(v) = raw[key].as_f64() {
                return v;
            }
        }
        default
    };

    let get_bool = |keys: &[&str], default: bool| -> bool {
        for key in keys {
            if let Some(tc) = text_cfg
                && let Some(v) = tc[key].as_bool()
            {
                return v;
            }
            if let Some(v) = raw[key].as_bool() {
                return v;
            }
        }
        default
    };

    let hidden_size = get_i32(&["hidden_size"], 0);
    let num_heads = get_i32(&["num_attention_heads", "num_heads"], 0);

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
        .unwrap_or_else(|| get_f64(&["partial_rotary_factor"], 0.25));

    let rope_theta = rope_obj
        .and_then(|rp| rp["rope_theta"].as_f64())
        .unwrap_or_else(|| get_f64(&["rope_theta"], 100_000.0));

    let bos_token_id = get_i32(&["bos_token_id"], 151643);
    let num_layers = get_i32(&["num_hidden_layers", "num_layers"], 0);
    let intermediate_size = get_i32(&["intermediate_size"], 0);
    let num_kv_heads = get_i32(&["num_key_value_heads", "num_kv_heads"], 8);
    let vocab_size = get_i32(&["vocab_size"], 151936);

    let num_experts = get_i32(&["num_experts"], 0);
    let num_experts_per_tok = get_i32(&["num_experts_per_tok"], 1);
    let moe_i = get_i32(&["moe_intermediate_size"], 0);

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
        rms_norm_eps: get_f64(&["rms_norm_eps"], 1e-6),
        head_dim,
        tie_word_embeddings: get_bool(&["tie_word_embeddings"], false),
        attention_bias: get_bool(&["attention_bias"], false),
        max_position_embeddings: get_i32(&["max_position_embeddings"], 131072),
        pad_token_id: get_i32(&["pad_token_id"], bos_token_id),
        eos_token_id: get_i32(&["eos_token_id"], 151645),
        bos_token_id,

        linear_num_value_heads: get_i32(&["linear_num_value_heads"], 64),
        linear_num_key_heads: get_i32(&["linear_num_key_heads"], 16),
        linear_key_head_dim: get_i32(&["linear_key_head_dim"], 192),
        linear_value_head_dim: get_i32(&["linear_value_head_dim"], 128),
        linear_conv_kernel_dim: get_i32(&["linear_conv_kernel_dim"], 4),
        full_attention_interval: get_i32(&["full_attention_interval"], 4),
        partial_rotary_factor,
        rope_theta,

        num_experts,
        num_experts_per_tok,
        decoder_sparse_step: get_i32(&["decoder_sparse_step"], 1),
        shared_expert_intermediate_size: {
            let v = get_i32(&["shared_expert_intermediate_size"], 0);
            if v > 0 { Some(v) } else { None }
        },
        moe_intermediate_size: { if moe_i > 0 { Some(moe_i) } else { None } },
        norm_topk_prob: get_bool(&["norm_topk_prob"], true),
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
/// Uses the same shared g_weights map as the dense path (mlx_qwen35_store_weight).
fn register_moe_weights_with_cpp(params: &HashMap<String, MxArray>) {
    use mlx_sys as sys;
    use std::ffi::CString;

    // Clear weights (shared map)
    unsafe { sys::mlx_qwen35_clear_weights() };

    let store = |name: &str, array: &MxArray| {
        let c_name = CString::new(name).expect("Weight name contains null byte");
        unsafe {
            sys::mlx_qwen35_store_weight(c_name.as_ptr(), array.as_raw_ptr());
        }
    };

    // Projections are already merged by sanitize_weights → merge_split_projections
    // (handles both bf16 concat and quantized scales/biases concat correctly).
    // Just store all params directly.
    for (name, array) in params {
        store(name, array);
    }

    let count = unsafe { sys::mlx_qwen35_weight_count() };
    info!("Registered {} weights with C++ MoE forward pass", count);
}
