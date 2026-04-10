use std::collections::HashMap;
use std::ffi::CString;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;
use tracing::{info, warn};

use crate::array::{DType, MxArray};
use crate::nn::LayerNorm;
use crate::tokenizer::Qwen3Tokenizer;
use crate::vision::encoder::{VisionAttention, VisionEncoderLayer, VisionMLP};
use crate::vision::projector::SpatialProjector;

use super::persistence_common::{
    dequant_fp8_weights, get_config_bool, get_config_f64, get_config_i32, load_all_safetensors,
};

use super::config::Qwen3_5Config;
use super::decoder_layer::AttentionType;
use super::model::{Qwen3_5Model, Qwen35Inner, handle_qwen35_cmd};
use super::processing::Qwen35VLImageProcessor;
use super::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, MLPVariant, is_mxfp8_checkpoint,
    is_quantized_checkpoint, try_build_mxfp8_quantized_linear, try_build_quantized_linear,
};
use super::vision::{Qwen3_5VisionConfig, Qwen3_5VisionEncoder};

/// Sanitize weights from HuggingFace format (dense variant).
///
/// Handles:
/// 1. Strip "model." and "language_model." prefixes
/// 2. Rename embed_tokens → embedding, model.norm → final_norm
/// 3. Remove lm_head.weight when tie_word_embeddings
/// 4. Conv1d weight axis: transpose([0, 2, 1]) when shape[-1] != 1
/// 5. Merge split linear attention projections into combined tensors.
///
/// mlx-vlm/mlx-lm store separate in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
/// but our model expects merged in_proj_qkvz and in_proj_ba.
/// Concatenates .weight, .scales, and .biases along axis 0.
pub(crate) fn merge_split_projections(result: &mut HashMap<String, MxArray>) -> Result<()> {
    // Merge in_proj_qkv + in_proj_z → in_proj_qkvz
    let split_qkv_keys: Vec<String> = result
        .keys()
        .filter(|k| k.ends_with(".linear_attn.in_proj_qkv.weight"))
        .cloned()
        .collect();

    for qkv_key in &split_qkv_keys {
        let prefix = qkv_key.strip_suffix(".in_proj_qkv.weight").unwrap();
        let z_weight_key = format!("{}.in_proj_z.weight", prefix);
        if !result.contains_key(&z_weight_key) {
            continue;
        }

        let qkv_w = result.remove(qkv_key).unwrap();
        let z_w = result.remove(&z_weight_key).unwrap();
        let combined_w = MxArray::concatenate(&qkv_w, &z_w, 0)?;
        result.insert(format!("{}.in_proj_qkvz.weight", prefix), combined_w);

        for suffix in &["scales", "biases"] {
            let qkv_k = format!("{}.in_proj_qkv.{}", prefix, suffix);
            let z_k = format!("{}.in_proj_z.{}", prefix, suffix);
            if let (Some(a), Some(b)) = (result.remove(&qkv_k), result.remove(&z_k)) {
                let combined = MxArray::concatenate(&a, &b, 0)?;
                result.insert(format!("{}.in_proj_qkvz.{}", prefix, suffix), combined);
            }
        }
    }

    // Merge in_proj_b + in_proj_a → in_proj_ba
    let split_b_keys: Vec<String> = result
        .keys()
        .filter(|k| k.ends_with(".linear_attn.in_proj_b.weight"))
        .cloned()
        .collect();

    for b_key in &split_b_keys {
        let prefix = b_key.strip_suffix(".in_proj_b.weight").unwrap();
        let a_weight_key = format!("{}.in_proj_a.weight", prefix);
        if !result.contains_key(&a_weight_key) {
            continue;
        }

        let b_w = result.remove(b_key).unwrap();
        let a_w = result.remove(&a_weight_key).unwrap();
        let combined_w = MxArray::concatenate(&b_w, &a_w, 0)?;
        result.insert(format!("{}.in_proj_ba.weight", prefix), combined_w);

        for suffix in &["scales", "biases"] {
            let b_k = format!("{}.in_proj_b.{}", prefix, suffix);
            let a_k = format!("{}.in_proj_a.{}", prefix, suffix);
            if let (Some(a), Some(b)) = (result.remove(&b_k), result.remove(&a_k)) {
                let combined = MxArray::concatenate(&a, &b, 0)?;
                result.insert(format!("{}.in_proj_ba.{}", prefix, suffix), combined);
            }
        }
    }

    Ok(())
}

/// 5. Norm weight +1.0 adjustment (when unsanitized weights detected)
/// 6. Remove MTP (multi-token prediction) weights
/// 7. FP8 E4M3 dequantization (weight + weight_scale_inv → bf16)
/// 8. 4-bit affine re-quantization (for FP8 source checkpoints)
fn sanitize_weights(
    mut params: HashMap<String, MxArray>,
    config: &Qwen3_5Config,
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

    // FP8 dequantization pass — convert FP8 weights to bf16 before further processing.
    // After all sanitization, FP8 weights are re-quantized to 4-bit affine for memory savings.
    let had_fp8 = params.keys().any(|k| k.ends_with("weight_scale_inv"));
    dequant_fp8_weights(&mut params, DType::BFloat16)?;
    if had_fp8 {
        crate::array::memory::synchronize_and_clear_cache();
    }

    let norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "final_norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        // NOTE: .linear_attn.norm.weight is intentionally NOT included here.
        // It's stored as f32 with final values (e.g. ~0.87), not as shifted weights.
        // Matches mlx-lm, mlx-vlm, and MoE persistence behavior.
    ];

    for (name, array) in params.drain() {
        // Skip MTP weights
        if name.contains("mtp.") {
            continue;
        }

        // Skip visual encoder weights (for VL models)
        if name.contains("model.visual") || name.contains("visual_encoder") {
            continue;
        }

        // Strip prefixes (VL models use model.language_model.*, text-only use model.*)
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

        // Remove lm_head when tie_word_embeddings is set
        if config.tie_word_embeddings && name.starts_with("lm_head.") {
            continue;
        }

        // Fix conv1d weight axis: HF stores [channels, 1, kernel_size],
        // we need [channels, kernel_size, 1] for depthwise conv
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

        // Apply norm +1.0 fix for unsanitized weights
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

    merge_split_projections(&mut result)?;

    // For FP8 source checkpoints, keep dequantized bf16 weights as-is.
    // Re-quantizing (FP8→bf16→4bit or →MXFP8) compounds quantization error
    // and produces gibberish. mlx-lm also keeps FP8-dequanted weights as bf16.

    Ok(result)
}

/// Apply weights directly to a Qwen35Inner (no locks needed).
fn apply_weights_inner(
    inner: &mut Qwen35Inner,
    params: &HashMap<String, MxArray>,
    config: &Qwen3_5Config,
    quant_bits: i32,
    quant_group_size: i32,
    per_layer_quant: &HashMap<String, (i32, i32)>,
) -> Result<()> {
    let is_quantized = is_quantized_checkpoint(params);
    let is_mxfp8 = is_mxfp8_checkpoint(params);

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
                                "Merged in_proj_qkvz has conflicting overrides: qkv={:?}, z={:?}. Using higher precision.",
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
                                "Merged in_proj_ba has conflicting overrides: b={:?}, a={:?}. Using higher precision.",
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

    // Embedding
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
        info!(
            "Loaded quantized embedding ({}-bit, quantized_matmul on forward)",
            bits
        );
    } else if let Some(w) = params.get("embedding.weight") {
        inner.embedding.set_weight(w)?;
    }

    // Final norm
    if let Some(w) = params.get("final_norm.weight") {
        inner.final_norm.set_weight(w)?;
    }

    // LM head
    if let Some(ref mut head) = inner.lm_head {
        if let Some(scales) = params.get("lm_head.scales") {
            let weight = params.get("lm_head.weight").ok_or_else(|| {
                Error::from_reason("Missing lm_head.weight for quantized lm_head")
            })?;
            let biases = params.get("lm_head.biases");
            let (bits, gs) = per_layer_quant
                .get("lm_head")
                .copied()
                .unwrap_or((quant_bits, quant_group_size));
            head.load_quantized(weight, scales, biases, gs, bits)?;
            info!(
                "Loaded quantized lm_head ({}-bit, quantized_matmul on forward)",
                bits
            );
        } else if let Some(w) = params.get("lm_head.weight") {
            head.set_weight(w)?;
        }
    }

    // Per-layer weights
    for (i, layer) in inner.layers.iter_mut().enumerate() {
        let prefix = format!("layers.{}", i);

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

        // Dense MLP weights
        match &mut layer.mlp {
            MLPVariant::Standard(mlp) => {
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
            MLPVariant::Quantized { .. } => {}
        }

        if let Some(w) = params.get(&format!("{}.input_layernorm.weight", prefix)) {
            layer.set_input_layernorm_weight(w)?;
        }
        if let Some(w) = params.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            layer.set_post_attention_layernorm_weight(w)?;
        }
    }

    // Validate mandatory weights
    validate_mandatory_weights(params, config, inner.layers.len())?;

    Ok(())
}

/// Validate mandatory weights presence for `apply_weights_inner`.
fn validate_mandatory_weights(
    params: &HashMap<String, MxArray>,
    config: &Qwen3_5Config,
    num_layers: usize,
) -> Result<()> {
    let mut missing_mandatory = Vec::new();
    if !params.contains_key("embedding.weight") {
        missing_mandatory.push("embedding.weight".to_string());
    }
    if !params.contains_key("final_norm.weight") {
        missing_mandatory.push("final_norm.weight".to_string());
    }
    if !config.tie_word_embeddings && !params.contains_key("lm_head.weight") {
        missing_mandatory.push("lm_head.weight".to_string());
    }

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
            || params.contains_key(&format!("{}.mlp.gate_proj.scales", prefix));
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

    Ok(())
}

/// Load a Qwen3.5 dense model using a dedicated model thread.
///
/// Spawns a `ModelThread<Qwen35Cmd>` that loads all weights inside the init_fn.
/// Returns a `Qwen3_5Model` thin shell with the thread handle.
pub async fn load_with_thread(model_path: &str) -> Result<Qwen3_5Model> {
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
                "Qwen3.5 config: {} layers, hidden={}, heads={}, kv_heads={}",
                config.num_layers, config.hidden_size, config.num_heads, config.num_kv_heads,
            );

            // Load all weights
            let raw_params = load_all_safetensors(path, true)?;
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
            let mut inner = Qwen35Inner::new(config.clone())?;

            // Apply weights
            apply_weights_inner(
                &mut inner,
                &params,
                &config,
                quant_bits,
                quant_group_size,
                &per_layer_quant,
            )?;

            // Register weights with C++
            if !is_quantized_checkpoint(&params) && !is_mxfp8_checkpoint(&params) {
                register_weights_with_cpp(&params, inner.model_id);
            } else {
                info!(
                    "Skipping C++ compiled path for quantized model (using Rust quantized_matmul)"
                );
                let _guard = super::model::COMPILED_WEIGHTS_RWLOCK.write().unwrap();
                unsafe { mlx_sys::mlx_clear_weights() };
            }

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

                info!("Qwen3.5-VL model loaded successfully (with vision encoder)");
            } else {
                info!("Qwen3.5 model loaded successfully");
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
        handle_qwen35_cmd,
    );

    let (config, model_id, _image_processor, _tokenizer) = init_rx
        .await
        .map_err(|_| Error::from_reason("Model thread exited during load"))??;

    Ok(Qwen3_5Model {
        thread,
        config,
        model_id,
    })
}

/// Register all sanitized weights with the C++ fused forward pass.
/// Sets model_id AFTER all weights are stored — ensures no inference sees
/// a partially-populated weight map with the new model's ID.
fn register_weights_with_cpp(params: &HashMap<String, MxArray>, model_id: u64) {
    use mlx_sys as sys;

    // Write-lock the weight RwLock for the entire registration.
    // This blocks any in-flight compiled inference from reading weights
    // until registration is complete and model_id is set.
    let _guard = super::model::COMPILED_WEIGHTS_RWLOCK.write().unwrap();

    unsafe { sys::mlx_clear_weights() };

    let store = |name: &str, array: &MxArray| {
        let c_name = CString::new(name).expect("Weight name contains null byte");
        unsafe {
            sys::mlx_store_weight(c_name.as_ptr(), array.as_raw_ptr());
        }
    };

    let mut handled_splits: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (name, array) in params {
        if name.ends_with(".linear_attn.in_proj_qkv.weight") {
            let prefix = name.strip_suffix(".in_proj_qkv.weight").unwrap();
            let z_key = format!("{}.in_proj_z.weight", prefix);
            if let Some(z_array) = params.get(&z_key)
                && let Ok(combined) = MxArray::concatenate(array, z_array, 0)
            {
                let combined_key = format!("{}.in_proj_qkvz.weight", prefix);
                store(&combined_key, &combined);
                handled_splits.insert(z_key);
                handled_splits.insert(name.clone());
                continue;
            }
        }
        if name.ends_with(".linear_attn.in_proj_z.weight") && handled_splits.contains(name) {
            continue;
        }

        if name.ends_with(".linear_attn.in_proj_b.weight") {
            let prefix = name.strip_suffix(".in_proj_b.weight").unwrap();
            let a_key = format!("{}.in_proj_a.weight", prefix);
            if let Some(a_array) = params.get(&a_key)
                && let Ok(combined) = MxArray::concatenate(array, a_array, 0)
            {
                let combined_key = format!("{}.in_proj_ba.weight", prefix);
                store(&combined_key, &combined);
                handled_splits.insert(a_key);
                handled_splits.insert(name.clone());
                continue;
            }
        }
        if name.ends_with(".linear_attn.in_proj_a.weight") && handled_splits.contains(name) {
            continue;
        }

        if !handled_splits.contains(name) {
            store(name, array);
        }
    }

    let count = unsafe { sys::mlx_weight_count() };
    info!("Registered {} weights with C++ fused forward pass", count);

    // Set model ID AFTER all weights are stored. This ordering ensures no
    // inference sees a partially-populated map with the new model's ID.
    unsafe { sys::mlx_set_model_id(model_id) };
}

/// Parse Qwen3.5 dense config from JSON.
fn parse_config(raw: &Value) -> Result<Qwen3_5Config> {
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

    if hidden_size <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid Qwen3.5 config: hidden_size must be > 0, got {}",
            hidden_size
        )));
    }
    if num_layers <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid Qwen3.5 config: num_hidden_layers must be > 0, got {}",
            num_layers
        )));
    }
    if num_heads <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid Qwen3.5 config: num_attention_heads must be > 0, got {}",
            num_heads
        )));
    }
    if intermediate_size <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid Qwen3.5 config: intermediate_size must be > 0, got {}",
            intermediate_size
        )));
    }
    if num_kv_heads <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid Qwen3.5 config: num_kv_heads must be > 0, got {}",
            num_kv_heads
        )));
    }
    if vocab_size <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid Qwen3.5 config: vocab_size must be > 0, got {}",
            vocab_size
        )));
    }

    Ok(Qwen3_5Config {
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
    })
}

/// Check if weights contain vision encoder tensors.
pub fn has_vision_weights(params: &HashMap<String, MxArray>) -> bool {
    params
        .keys()
        .any(|k| k.starts_with("vision_tower.") || k.starts_with("visual."))
}

/// Parse vision config from JSON.
pub(crate) fn parse_vision_config(raw: &Value) -> Qwen3_5VisionConfig {
    let vision_cfg = raw.get("vision_config");

    let get = |key: &str, default: i32| -> i32 {
        vision_cfg
            .and_then(|v| v[key].as_i64())
            .unwrap_or(default as i64) as i32
    };

    Qwen3_5VisionConfig {
        hidden_size: get("hidden_size", 1152),
        intermediate_size: get("intermediate_size", 4304),
        num_heads: get("num_heads", 16),
        num_layers: vision_cfg
            .and_then(|v| {
                v["depth"]
                    .as_i64()
                    .or_else(|| v["num_hidden_layers"].as_i64())
            })
            .unwrap_or(27) as i32,
        patch_size: get("patch_size", 16),
        spatial_merge_size: get("spatial_merge_size", 2),
        image_size: get("image_size", 768),
        out_hidden_size: get("out_hidden_size", 4096),
    }
}

/// Load vision encoder weights from params.
pub(crate) fn load_vision_weights(
    encoder: &mut Qwen3_5VisionEncoder,
    params: &HashMap<String, MxArray>,
    config: &Qwen3_5VisionConfig,
) -> Result<()> {
    let get = |key: &str| -> Result<&MxArray> {
        params
            .get(key)
            .ok_or_else(|| Error::from_reason(format!("Missing vision weight: {}", key)))
    };

    let get_opt = |key: &str| -> Option<&MxArray> { params.get(key) };

    // Patch embedding: handle both 4D Conv2d [out, kH, kW, in] and
    // 5D Conv3d [out, kD, kH, kW, in] formats. For Conv3d, extract
    // temporal slice 0 for our Conv2d PatchEmbedding.
    if let Some(pe_weight) = get_opt("patch_embed.proj.weight") {
        let ndim = pe_weight.ndim()?;
        if ndim == 5 {
            // Conv3d [out, kD, kH, kW, in] → take slice [:, 0, :, :, :]
            let out_c = pe_weight.shape_at(0)?;
            let kh = pe_weight.shape_at(2)?;
            let kw = pe_weight.shape_at(3)?;
            let in_c = pe_weight.shape_at(4)?;
            let slice0 = pe_weight.slice(&[0, 0, 0, 0, 0], &[out_c, 1, kh, kw, in_c])?;
            let conv2d_weight = slice0.squeeze(Some(&[1]))?;
            encoder.set_patch_embed(&conv2d_weight)?;
        } else {
            encoder.set_patch_embed(pe_weight)?;
        }
    }

    // Position embedding
    if let Some(pos_embed) = get_opt("pos_embed.weight") {
        encoder.set_pos_embed(pos_embed);
    }

    // Encoder layers (blocks.0..blocks.N)
    for layer_idx in 0..config.num_layers {
        let prefix = format!("blocks.{}", layer_idx);

        let qkv_w = get(&format!("{}.attn.qkv.weight", prefix))?;
        let qkv_b = get_opt(&format!("{}.attn.qkv.bias", prefix));
        let proj_w = get(&format!("{}.attn.proj.weight", prefix))?;
        let proj_b = get_opt(&format!("{}.attn.proj.bias", prefix));

        let attn = VisionAttention::new(
            config.hidden_size as u32,
            config.num_heads as u32,
            qkv_w,
            qkv_b,
            proj_w,
            proj_b,
        )?;

        let fc1_w = get(&format!("{}.mlp.linear_fc1.weight", prefix))?;
        let fc1_b = get_opt(&format!("{}.mlp.linear_fc1.bias", prefix));
        let fc2_w = get(&format!("{}.mlp.linear_fc2.weight", prefix))?;
        let fc2_b = get_opt(&format!("{}.mlp.linear_fc2.bias", prefix));

        let mlp = VisionMLP::new(fc1_w, fc1_b, fc2_w, fc2_b)?;

        let norm1_w = get(&format!("{}.norm1.weight", prefix))?;
        let norm1_b = get_opt(&format!("{}.norm1.bias", prefix));
        let norm2_w = get(&format!("{}.norm2.weight", prefix))?;
        let norm2_b = get_opt(&format!("{}.norm2.bias", prefix));

        let ln1 = LayerNorm::from_weights(norm1_w, norm1_b, Some(1e-6))?;
        let ln2 = LayerNorm::from_weights(norm2_w, norm2_b, Some(1e-6))?;

        let layer = VisionEncoderLayer::new(&ln1, &ln2, &attn, &mlp);
        encoder.add_layer(&layer);
    }

    // Merger (spatial projector)
    let ln_q_w = get("merger.norm.weight")?;
    let ln_q_b = get("merger.norm.bias")?;
    let fc1_w = get("merger.linear_fc1.weight")?;
    let fc1_b = get("merger.linear_fc1.bias")?;
    let fc2_w = get("merger.linear_fc2.weight")?;
    let fc2_b = get("merger.linear_fc2.bias")?;

    let merger = SpatialProjector::new(
        config.spatial_merge_size as u32,
        ln_q_w,
        ln_q_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
    )?;
    encoder.set_merger(merger);

    info!(
        "Loaded vision encoder: {} layers, merger ready",
        config.num_layers
    );
    Ok(())
}

/// Create a random-init Qwen3.5 model and save it to disk.
///
/// Spawns a dedicated `ModelThread<Qwen35Cmd>` whose init builds a fresh
/// random-weight `Qwen35Inner` directly, then dispatches `Qwen35Cmd::SaveModel`
/// on that thread. The thread is dropped at the end of the promise, so the
/// in-memory model is released once the checkpoint has been written. Used by
/// TypeScript test fixtures that need an on-disk checkpoint without keeping a
/// NAPI model instance alive.
#[napi]
pub fn create_random_qwen35_checkpoint<'env>(
    env: &'env Env,
    config: Qwen3_5Config,
    save_path: String,
) -> Result<PromiseRaw<'env, ()>> {
    use super::model::Qwen35Cmd;

    let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_init(
        move || {
            let inner = Qwen35Inner::new(config)?;
            Ok((inner, ()))
        },
        handle_qwen35_cmd,
    );

    env.spawn_future(async move {
        init_rx
            .await
            .map_err(|_| napi::Error::from_reason("Model thread exited during init"))??;

        let (tx, rx) = tokio::sync::oneshot::channel();
        thread.send(Qwen35Cmd::SaveModel {
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
