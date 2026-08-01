use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;
use tracing::info;

use crate::array::MxArray;
use crate::engine::persistence::load_all_safetensors;
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
    reject_quantized_checkpoint(&raw, model_path)?;
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

/// Reject quantized Harrier checkpoints before packed tensors reach the dense
/// Qwen3 embedding backbone. Harrier's embedding and projection layers accept
/// only floating weights and do not parse quantization metadata.
///
/// The shared alias selector keeps malformed or divergent
/// `quantization`/`quantization_config` handling aligned with quant-capable
/// families. Empty blocks are harmless compatibility stubs; any non-empty
/// block is an unsupported quantized checkpoint and fails before weight I/O.
fn reject_quantized_checkpoint(raw_config: &Value, model_path: &str) -> Result<()> {
    let Some(block) = crate::models::quant_dispatch::select_quantization_block(raw_config)? else {
        return Ok(());
    };
    if !block.as_object().is_some_and(|o| !o.is_empty()) {
        return Ok(());
    }
    let mode = block
        .get("mode")
        .and_then(|m| m.as_str())
        .unwrap_or("affine");
    Err(Error::from_reason(format!(
        "Model at '{model_path}' carries a '{mode}' quantization config, but the Harrier \
         loader (model_type \"harrier\") has no quantized-weight support — it can only load \
         dense (bf16/f16/f32) checkpoints. Re-convert without quantization."
    )))
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dense_harrier_configs_remain_accepted() {
        let dense = json!({ "model_type": "harrier", "hidden_size": 1024 });
        assert!(reject_quantized_checkpoint(&dense, "/tmp/harrier").is_ok());

        let empty_stub = json!({ "model_type": "harrier", "quantization": {} });
        assert!(reject_quantized_checkpoint(&empty_stub, "/tmp/harrier").is_ok());
    }

    #[test]
    fn harrier_rejects_nonempty_quantization_metadata() {
        for block in [
            json!({ "mode": "affine", "bits": 4, "group_size": 64 }),
            json!({ "mode": "mxfp4", "bits": 4, "group_size": 32 }),
            json!({ "mode": "mxfp8", "bits": 8, "group_size": 32 }),
            json!({ "mode": "nvfp4", "bits": 4, "group_size": 16 }),
            json!({ "mode": "sym8", "bits": 8, "group_size": null }),
        ] {
            let mode = block["mode"].as_str().expect("test mode");
            let config = json!({
                "model_type": "harrier",
                "quantization": block.clone(),
                "quantization_config": block,
            });
            let err = reject_quantized_checkpoint(&config, "/tmp/harrier")
                .expect_err("Harrier's dense runtime must reject quantized metadata");
            assert!(err.reason.contains(mode), "{}", err.reason);
            assert!(
                err.reason.contains("no quantized-weight support"),
                "{}",
                err.reason
            );
        }
    }

    #[test]
    fn harrier_load_rejects_quantization_before_weight_io() {
        let dir = std::env::temp_dir().join(format!(
            "mlx-harrier-quant-reject-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("create Harrier smoke directory");
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(&json!({
                "model_type": "harrier",
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "num_hidden_layers": 1,
                "num_key_value_heads": 2,
                "intermediate_size": 3072,
                "vocab_size": 151936,
                "quantization": {
                    "mode": "mxfp4",
                    "bits": 4,
                    "group_size": 32,
                },
            }))
            .expect("serialize config"),
        )
        .expect("write config.json");

        // There is deliberately no safetensors file: the quantization contract
        // must reject first rather than fall through to weight loading.
        let err = match load_impl(dir.to_str().expect("UTF-8 temp path")) {
            Err(err) => err,
            Ok(_) => panic!("quantized Harrier config must fail before missing weights"),
        };
        assert!(
            err.reason.contains("no quantized-weight support"),
            "{}",
            err.reason
        );

        fs::remove_dir_all(&dir).expect("remove Harrier smoke directory");
    }
}
