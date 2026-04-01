use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;
use tracing::info;

use crate::array::MxArray;
use crate::models::qwen3_5::persistence_common::load_all_safetensors;
use crate::tokenizer::Qwen3Tokenizer;

use super::{HarrierConfig, HarrierModel};

#[napi]
impl HarrierModel {
    /// Load a Harrier embedding model from a directory.
    ///
    /// Expects the standard HuggingFace layout:
    /// - config.json (model configuration)
    /// - model.safetensors or weights.safetensors (weights)
    /// - tokenizer.json (tokenizer)
    /// - config_sentence_transformers.json (optional, prompt presets)
    #[napi]
    pub fn load<'env>(
        env: &'env Env,
        model_path: String,
    ) -> Result<PromiseRaw<'env, HarrierModel>> {
        env.spawn_future_with_callback(
            async move {
                napi::bindgen_prelude::spawn_blocking(move || load_impl(&model_path))
                    .await
                    .map_err(|e| Error::from_reason(format!("HarrierModel::load failed: {}", e)))?
            },
            |_env, model| Ok(model),
        )
    }
}

fn load_impl(model_path: &str) -> Result<HarrierModel> {
    let path = Path::new(model_path);

    if !path.exists() {
        return Err(Error::from_reason(format!(
            "Model path does not exist: {}",
            model_path
        )));
    }

    let config_path = path.join("config.json");
    if !config_path.exists() {
        return Err(Error::from_reason(format!(
            "Config file not found: {}",
            config_path.display()
        )));
    }

    let config_data = fs::read_to_string(&config_path)?;
    let raw: Value = serde_json::from_str(&config_data)?;
    let config = parse_config(&raw)?;

    info!(
        "HarrierModel config: {} layers, {} hidden, {} heads",
        config.num_layers, config.hidden_size, config.num_heads
    );

    let mut param_map = load_all_safetensors(path, false)?;
    info!("Loaded {} tensors from SafeTensors", param_map.len());

    let mapped_params = map_hf_names(&mut param_map);
    info!("Mapped {} parameters", mapped_params.len());

    let tokenizer_path = path.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(Error::from_reason(format!(
            "Tokenizer file not found: {}",
            tokenizer_path.display()
        )));
    }
    let tokenizer = Qwen3Tokenizer::load_from_file_sync(tokenizer_path.to_str().unwrap())?;

    let prompts = load_prompts(path);
    if !prompts.is_empty() {
        info!(
            "Loaded {} prompt presets: {:?}",
            prompts.len(),
            prompts.keys().collect::<Vec<_>>()
        );
    }

    let mut model = HarrierModel::new(config)?;
    model.load_parameters(&mapped_params)?;
    model.tokenizer = Some(Arc::new(tokenizer));
    model.prompts = prompts;

    for array in mapped_params.values() {
        array.eval();
    }

    info!(
        "HarrierModel loaded successfully ({} parameters)",
        model.num_parameters()
    );
    Ok(model)
}

/// Parse HarrierConfig from raw JSON, supporting both HuggingFace and internal naming.
fn parse_config(raw: &Value) -> Result<HarrierConfig> {
    let hidden_size = get_i32(raw, &["hidden_size", "hiddenSize"])?;
    let num_heads = get_i32(raw, &["num_attention_heads", "num_heads", "numHeads"])?;

    let head_dim = raw["head_dim"]
        .as_i64()
        .or_else(|| raw["headDim"].as_i64())
        .map(|v| v as i32)
        .unwrap_or(hidden_size / num_heads);

    Ok(HarrierConfig {
        hidden_size,
        num_layers: get_i32(raw, &["num_hidden_layers", "num_layers", "numLayers"])?,
        num_heads,
        num_key_value_heads: get_i32(
            raw,
            &[
                "num_key_value_heads",
                "numKeyValueHeads",
                "num_kv_heads",
                "numKvHeads",
            ],
        )?,
        intermediate_size: get_i32(raw, &["intermediate_size", "intermediateSize"])?,
        rms_norm_eps: raw["rms_norm_eps"]
            .as_f64()
            .or_else(|| raw["rmsNormEps"].as_f64())
            .unwrap_or(1e-6),
        rope_theta: raw["rope_theta"]
            .as_f64()
            .or_else(|| raw["ropeTheta"].as_f64())
            .unwrap_or(1_000_000.0),
        max_position_embeddings: raw["max_position_embeddings"]
            .as_i64()
            .or_else(|| raw["maxPositionEmbeddings"].as_i64())
            .unwrap_or(32768) as i32,
        head_dim,
        use_qk_norm: Some(
            raw["use_qk_norm"]
                .as_bool()
                .or_else(|| raw["useQkNorm"].as_bool())
                .unwrap_or(true),
        ),
        vocab_size: get_i32(raw, &["vocab_size", "vocabSize"])?,
    })
}

/// Load prompt presets from config_sentence_transformers.json if present.
///
/// Format: `{ "prompts": { "task_name": "Instruct: ...\nQuery: " } }`
fn load_prompts(model_dir: &Path) -> HashMap<String, String> {
    let prompts_path = model_dir.join("config_sentence_transformers.json");
    let data = match fs::read_to_string(&prompts_path) {
        Ok(d) => d,
        Err(_) => return HashMap::new(),
    };
    let json: Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(_) => return HashMap::new(),
    };

    let mut prompts = HashMap::new();
    if let Some(obj) = json["prompts"].as_object() {
        for (key, val) in obj {
            if let Some(s) = val.as_str() {
                prompts.insert(key.clone(), s.to_string());
            }
        }
    }
    prompts
}

/// Map HuggingFace parameter names to internal names.
fn map_hf_names(params: &mut HashMap<String, MxArray>) -> HashMap<String, MxArray> {
    let mut mapped = HashMap::new();

    for (name, array) in params.drain() {
        let mapped_name = if let Some(stripped) = name.strip_prefix("model.") {
            if stripped == "embed_tokens.weight" {
                "embedding.weight".to_string()
            } else if stripped.starts_with("embed_tokens.") {
                stripped.replace("embed_tokens", "embedding")
            } else if stripped == "norm.weight" {
                "final_norm.weight".to_string()
            } else {
                stripped.to_string()
            }
        } else if name == "lm_head.weight" {
            // Skip lm_head if present — embedding model doesn't use it
            continue;
        } else {
            name
        };
        mapped.insert(mapped_name, array);
    }

    mapped
}

fn get_i32(raw: &Value, keys: &[&str]) -> Result<i32> {
    for key in keys {
        if let Some(v) = raw[key].as_i64() {
            return Ok(v as i32);
        }
    }
    Err(Error::from_reason(format!(
        "Missing required config field: {}",
        keys[0]
    )))
}
