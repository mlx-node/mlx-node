use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use serde_json::Value;
use tracing::info;

use crate::array::MxArray;
use crate::tokenizer::Qwen3Tokenizer;
use crate::utils::safetensors::SafeTensorsFile;

use super::config::Qwen3_5Config;
use super::decoder_layer::{AttentionType, MLPType};
use super::model::Qwen3_5Model;

/// Load all safetensors files from a directory (supports sharded checkpoints).
///
/// Looks for:
/// 1. `weights.safetensors` (single file, our format)
/// 2. `model.safetensors` (single file, HuggingFace format)
/// 3. `model-00001-of-*.safetensors` (sharded HuggingFace format)
fn load_all_safetensors(dir: &Path) -> Result<HashMap<String, MxArray>> {
    // Try single-file formats first
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

    // Try sharded format: model-00001-of-NNNNN.safetensors
    let mut shard_files: Vec<std::path::PathBuf> = Vec::new();
    let entries = fs::read_dir(dir)
        .map_err(|e| Error::from_reason(format!("Failed to read model directory: {}", e)))?;

    for entry in entries {
        let entry = entry.map_err(|e| Error::from_reason(format!("Failed to read directory entry: {}", e)))?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("model-") && name.ends_with(".safetensors") && name.contains("-of-") {
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

/// Sanitize weights from HuggingFace format.
///
/// Handles:
/// 1. Strip "model." and "language_model." prefixes
/// 2. Rename embed_tokens → embedding, model.norm → final_norm
/// 3. Remove lm_head.weight when tie_word_embeddings
/// 4. Conv1d weight axis: moveaxis(2, 1) when shape[-1] != 1
/// 5. Norm weight +1.0 adjustment (when unsanitized weights detected)
/// 6. MoE expert consolidation: stack individual expert weights → switch_mlp
/// 7. MoE gate_up_proj split
/// 8. Remove MTP (multi-token prediction) weights
fn sanitize_weights(
    mut params: HashMap<String, MxArray>,
    config: &Qwen3_5Config,
) -> Result<HashMap<String, MxArray>> {
    let mut result: HashMap<String, MxArray> = HashMap::new();

    // Detect if these are unsanitized HF weights
    let has_mtp_weights = params.keys().any(|k| k.contains("mtp."));
    let has_unsanitized_conv1d = params.iter().any(|(name, array)| {
        if !name.contains("conv1d.weight") {
            return false;
        }
        match array.shape() {
            Ok(shape) => shape.len() >= 2 && shape[shape.len() - 1] != 1,
            Err(_) => false,
        }
    });
    let needs_norm_fix = has_mtp_weights || has_unsanitized_conv1d;

    // Check if these are individually-listed expert weights (qwen3_next format)
    // vs fused gate_up_proj (qwen3_5_moe format)
    let has_individual_experts = params.keys().any(|k| {
        k.contains(".mlp.experts.0.up_proj.weight") || k.contains("model.layers.0.mlp.experts.0.up_proj.weight")
    });

    // MoE expert consolidation: collect individual expert weights for stacking
    let mut expert_weights: HashMap<String, Vec<(usize, MxArray)>> = HashMap::new();

    // Define norm suffixes that get the +1.0 fix
    let norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "final_norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
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

        // Strip prefixes: language_model.model. > language_model. > model.
        let name = name
            .strip_prefix("language_model.model.")
            .or_else(|| name.strip_prefix("language_model."))
            .or_else(|| name.strip_prefix("model."))
            .unwrap_or(&name)
            .to_string();

        // Rename special keys
        let name = if name == "embed_tokens.weight" {
            "embedding.weight".to_string()
        } else if name == "norm.weight" {
            "final_norm.weight".to_string()
        } else {
            name
        };

        // Remove lm_head when tie_word_embeddings is set
        if config.tie_word_embeddings && name == "lm_head.weight" {
            continue;
        }

        // Handle individually-listed expert weights:
        // layers.{l}.mlp.experts.{e}.{proj}.weight → stack later
        if has_individual_experts && name.contains(".mlp.experts.") {
            let parts: Vec<&str> = name.split('.').collect();
            // Expected: layers.{l}.mlp.experts.{e}.{proj}.weight
            // parts:    [0]    [1] [2]  [3]     [4] [5]   [6]
            if parts.len() >= 7 {
                let layer = parts[1];
                let expert_idx: usize = parts[4].parse().unwrap_or(0);
                let proj_name = parts[5]; // gate_proj, up_proj, down_proj
                let key = format!("layers.{}.mlp.switch_mlp.{}.weight", layer, proj_name);
                expert_weights
                    .entry(key)
                    .or_default()
                    .push((expert_idx, array));
                continue;
            }
        }

        // Handle fused gate_up_proj: split into gate_proj + up_proj
        // This occurs in the qwen3_5_moe format where experts have fused projections
        if name.contains(".mlp.experts.gate_up_proj") || name.contains(".mlp.switch_mlp.gate_up_proj") {
            let shape = array.shape()?;
            if shape.len() >= 2 {
                let split_axis = shape.len() - 2;
                let mid = shape[split_axis] / 2;
                let gate = array.slice_axis(split_axis, 0, mid)?;
                let up = array.slice_axis(split_axis, mid, shape[split_axis])?;

                let base = if name.contains("switch_mlp") {
                    name.replace("gate_up_proj", "gate_proj.weight")
                } else {
                    name.replace("experts.gate_up_proj", "switch_mlp.gate_proj.weight")
                };
                let up_name = base.replace("gate_proj", "up_proj");

                result.insert(base, gate);
                result.insert(up_name, up);
                continue;
            }
        }

        // Handle experts.down_proj → switch_mlp.down_proj.weight rename
        let name = if name.contains(".mlp.experts.down_proj") {
            name.replace(".mlp.experts.down_proj", ".mlp.switch_mlp.down_proj.weight")
        } else {
            name
        };

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
        // Python: RMSNorm stores weight as (1 + learned_param), but HF checkpoints
        // may store just learned_param. We detect this via mtp/conv1d indicators.
        let array = if needs_norm_fix && norm_suffixes.iter().any(|sfx| name.ends_with(sfx)) {
            let ndim = array.ndim()?;
            if ndim == 1 {
                let one = MxArray::scalar_float(1.0)?;
                array.add(&one)?
            } else {
                array
            }
        } else {
            array
        };

        result.insert(name, array);
    }

    // Stack individual expert weights into 3D tensors [num_experts, out, in]
    if !expert_weights.is_empty() {
        let num_experts = config.num_experts.unwrap_or(0) as usize;
        for (key, mut experts) in expert_weights {
            experts.sort_by_key(|(idx, _)| *idx);

            if num_experts > 0 && experts.len() != num_experts {
                return Err(Error::from_reason(format!(
                    "Expected {} experts for {}, got {}",
                    num_experts, key, experts.len()
                )));
            }

            let arrays: Vec<&MxArray> = experts.iter().map(|(_, a)| a).collect();
            let stacked = MxArray::stack(arrays, Some(0))?;
            result.insert(key, stacked);
        }
    }

    Ok(result)
}

/// Apply weights to a Qwen3.5 model.
fn apply_weights(model: &mut Qwen3_5Model, params: &HashMap<String, MxArray>) -> Result<()> {
    // Embedding
    if let Some(w) = params.get("embedding.weight") {
        model.embedding.set_weight(w)?;
    }

    // Final norm
    if let Some(w) = params.get("final_norm.weight") {
        model.final_norm.set_weight(w)?;
    }

    // LM head
    if let Some(ref mut head) = model.lm_head {
        if let Some(w) = params.get("lm_head.weight") {
            head.set_weight(w)?;
        }
    }

    // Per-layer weights
    for (i, layer) in model.layers.iter_mut().enumerate() {
        let prefix = format!("layers.{}", i);

        // Attention weights
        match &mut layer.attn {
            AttentionType::Linear(gdn) => {
                // Fused qkvz projection
                if let Some(w) = params.get(&format!("{}.linear_attn.in_proj_qkvz.weight", prefix)) {
                    gdn.set_in_proj_qkvz_weight(w)?;
                }
                // Split qkv + z projections (some checkpoints separate them)
                if let Some(w) = params.get(&format!("{}.linear_attn.in_proj_qkv.weight", prefix)) {
                    if let Some(z) = params.get(&format!("{}.linear_attn.in_proj_z.weight", prefix)) {
                        let combined = MxArray::concatenate(w, z, 0)?;
                        gdn.set_in_proj_qkvz_weight(&combined)?;
                    } else {
                        gdn.set_in_proj_qkvz_weight(w)?;
                    }
                }
                // Fused ba projection
                if let Some(w) = params.get(&format!("{}.linear_attn.in_proj_ba.weight", prefix)) {
                    gdn.set_in_proj_ba_weight(w)?;
                }
                // Split b + a projections
                if let Some(b) = params.get(&format!("{}.linear_attn.in_proj_b.weight", prefix)) {
                    if let Some(a) = params.get(&format!("{}.linear_attn.in_proj_a.weight", prefix)) {
                        let combined = MxArray::concatenate(b, a, 0)?;
                        gdn.set_in_proj_ba_weight(&combined)?;
                    }
                }
                if let Some(w) = params.get(&format!("{}.linear_attn.conv1d.weight", prefix)) {
                    gdn.set_conv1d_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.linear_attn.norm.weight", prefix)) {
                    gdn.set_norm_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.linear_attn.out_proj.weight", prefix)) {
                    gdn.set_out_proj_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.linear_attn.dt_bias", prefix)) {
                    gdn.set_dt_bias(w);
                }
                if let Some(w) = params.get(&format!("{}.linear_attn.A_log", prefix)) {
                    gdn.set_a_log(w);
                }
            }
            AttentionType::Full(attn) => {
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
                if let Some(w) = params.get(&format!("{}.self_attn.q_norm.weight", prefix)) {
                    attn.set_q_norm_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.self_attn.k_norm.weight", prefix)) {
                    attn.set_k_norm_weight(w)?;
                }
                // Biases (optional, depends on attention_bias config)
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
            MLPType::Dense(mlp) => {
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
            MLPType::MoE(moe) => {
                if let Some(w) = params.get(&format!("{}.mlp.gate.weight", prefix)) {
                    moe.set_gate_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.mlp.switch_mlp.gate_proj.weight", prefix)) {
                    moe.set_switch_mlp_gate_proj_weight(w);
                }
                if let Some(w) = params.get(&format!("{}.mlp.switch_mlp.up_proj.weight", prefix)) {
                    moe.set_switch_mlp_up_proj_weight(w);
                }
                if let Some(w) = params.get(&format!("{}.mlp.switch_mlp.down_proj.weight", prefix)) {
                    moe.set_switch_mlp_down_proj_weight(w);
                }
                if let Some(w) = params.get(&format!("{}.mlp.shared_expert.gate_proj.weight", prefix)) {
                    moe.set_shared_expert_gate_proj_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.mlp.shared_expert.up_proj.weight", prefix)) {
                    moe.set_shared_expert_up_proj_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.mlp.shared_expert.down_proj.weight", prefix)) {
                    moe.set_shared_expert_down_proj_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.mlp.shared_expert_gate.weight", prefix)) {
                    moe.set_shared_expert_gate_weight(w)?;
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

    info!("Applied {} weight tensors to Qwen3.5 model", params.len());
    Ok(())
}

/// Load a pretrained Qwen3.5 model from a directory.
pub async fn load_pretrained(model_path: &str) -> Result<Qwen3_5Model> {
    let model_path = model_path.to_string();

    napi::tokio::task::spawn_blocking(move || {
        let path = Path::new(&model_path);

        if !path.exists() {
            return Err(Error::from_reason(format!(
                "Model path does not exist: {}", model_path
            )));
        }

        // Load config
        let config_path = path.join("config.json");
        let config_data = fs::read_to_string(&config_path)
            .map_err(|e| Error::from_reason(format!("Failed to read config: {}", e)))?;
        let raw: Value = serde_json::from_str(&config_data)
            .map_err(|e| Error::from_reason(format!("Failed to parse config: {}", e)))?;

        let config = parse_config(&raw)?;

        info!("Qwen3.5 config: {} layers, hidden={}, heads={}, kv_heads={}, moe={}",
            config.num_layers, config.hidden_size, config.num_heads, config.num_kv_heads,
            config.is_moe());

        // Load all weights (supports single-file and sharded formats)
        let raw_params = load_all_safetensors(path)?;
        info!("Loaded {} raw tensors", raw_params.len());

        // Sanitize weights
        let params = sanitize_weights(raw_params, &config)?;
        info!("Sanitized to {} parameters", params.len());

        // Load tokenizer
        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            info!("Loading tokenizer from: {}", tokenizer_path.display());
            Some(Qwen3Tokenizer::load_from_file_sync(tokenizer_path.to_str().unwrap())?)
        } else {
            None
        };

        // Create model
        let mut model = Qwen3_5Model::new(config)?;

        // Apply weights
        apply_weights(&mut model, &params)?;

        // Set tokenizer
        if let Some(tok) = tokenizer {
            model.tokenizer = Some(Arc::new(tok));
        }

        info!("Qwen3.5 model loaded successfully");
        Ok(model)
    })
    .await
    .map_err(|e| Error::from_reason(format!("Failed to load model: {}", e)))?
}

/// Parse Qwen3.5 config from JSON.
fn parse_config(raw: &Value) -> Result<Qwen3_5Config> {
    let get_i32 = |keys: &[&str], default: i32| -> i32 {
        for key in keys {
            if let Some(v) = raw[key].as_i64() {
                return v as i32;
            }
        }
        default
    };

    let get_f64 = |keys: &[&str], default: f64| -> f64 {
        for key in keys {
            if let Some(v) = raw[key].as_f64() {
                return v;
            }
        }
        default
    };

    let get_bool = |keys: &[&str], default: bool| -> bool {
        for key in keys {
            if let Some(v) = raw[key].as_bool() {
                return v;
            }
        }
        default
    };

    let hidden_size = get_i32(&["hidden_size"], 0);
    let num_heads = get_i32(&["num_attention_heads", "num_heads"], 0);
    let head_dim = if let Some(v) = raw["head_dim"].as_i64() {
        v as i32
    } else {
        hidden_size / num_heads
    };

    // Extract rope parameters from nested config if present
    let partial_rotary_factor = if let Some(rope_params) = raw.get("rope_parameters") {
        rope_params["partial_rotary_factor"].as_f64().unwrap_or(0.25)
    } else {
        raw["partial_rotary_factor"].as_f64().unwrap_or(0.25)
    };

    let rope_theta = if let Some(rope_params) = raw.get("rope_parameters") {
        rope_params["rope_theta"].as_f64().unwrap_or(100_000.0)
    } else {
        raw["rope_theta"].as_f64().unwrap_or(100_000.0)
    };

    let bos_token_id = get_i32(&["bos_token_id"], 151643);

    Ok(Qwen3_5Config {
        vocab_size: get_i32(&["vocab_size"], 151936),
        hidden_size,
        num_layers: get_i32(&["num_hidden_layers", "num_layers"], 0),
        num_heads,
        num_kv_heads: get_i32(&["num_key_value_heads", "num_kv_heads"], 8),
        intermediate_size: get_i32(&["intermediate_size"], 0),
        rms_norm_eps: get_f64(&["rms_norm_eps"], 1e-6),
        head_dim,
        tie_word_embeddings: get_bool(&["tie_word_embeddings"], false),
        attention_bias: get_bool(&["attention_bias"], false),
        max_position_embeddings: get_i32(&["max_position_embeddings"], 131072),
        pad_token_id: get_i32(&["pad_token_id"], bos_token_id),
        eos_token_id: get_i32(&["eos_token_id"], 151645),
        bos_token_id,

        // Linear attention fields
        linear_num_value_heads: get_i32(&["linear_num_value_heads"], 64),
        linear_num_key_heads: get_i32(&["linear_num_key_heads"], 16),
        linear_key_head_dim: get_i32(&["linear_key_head_dim"], 192),
        linear_value_head_dim: get_i32(&["linear_value_head_dim"], 128),
        linear_conv_kernel_dim: get_i32(&["linear_conv_kernel_dim"], 4),
        full_attention_interval: get_i32(&["full_attention_interval"], 4),
        partial_rotary_factor,
        rope_theta,

        // MoE fields
        num_experts: {
            let v = get_i32(&["num_experts"], 0);
            if v > 0 { Some(v) } else { None }
        },
        num_experts_per_tok: {
            let v = get_i32(&["num_experts_per_tok"], 0);
            if v > 0 { Some(v) } else { None }
        },
        decoder_sparse_step: {
            let v = get_i32(&["decoder_sparse_step"], 1);
            Some(v)
        },
        shared_expert_intermediate_size: {
            let v = get_i32(&["shared_expert_intermediate_size"], 0);
            if v > 0 { Some(v) } else { None }
        },
        moe_intermediate_size: {
            let v = get_i32(&["moe_intermediate_size"], 0);
            if v > 0 { Some(v) } else { None }
        },
        norm_topk_prob: Some(get_bool(&["norm_topk_prob"], true)),
        mlp_only_layers: raw["mlp_only_layers"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_i64().map(|i| i as i32)).collect()),
    })
}
