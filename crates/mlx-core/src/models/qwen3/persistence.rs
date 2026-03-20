/**
 * Qwen3 Model - Persistence Methods
 *
 * Contains methods for saving and loading model weights and configuration.
 */
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use futures::TryFutureExt;
use napi::{Env, Status, bindgen_prelude::*, tokio};
use napi_derive::napi;
use serde_json::Value;
use tracing::info;

use crate::array::MxArray;
use crate::tokenizer::Qwen3Tokenizer;
use crate::utils::safetensors::{load_safetensors_lazy, save_safetensors};

use super::{Qwen3Config, Qwen3Model};

/// Validate that all required parameters were loaded with correct shapes
///
/// This function verifies:
/// 1. All required weights exist in the parameter map
/// 2. Loaded shapes match expected dimensions based on config
///
/// # Arguments
/// * `params` - HashMap of parameter names to MxArray values
/// * `config` - Qwen3Config specifying model dimensions
///
/// # Returns
/// * Ok(()) if all validations pass
/// * Err with descriptive message if validation fails
fn validate_loaded_parameters(
    params: &HashMap<String, MxArray>,
    config: &Qwen3Config,
) -> Result<()> {
    let num_layers = config.num_layers as usize;
    let hidden_size = config.hidden_size as usize;
    let intermediate_size = config.intermediate_size as usize;
    let num_heads = config.num_heads as usize;
    let num_kv_heads = config.num_kv_heads as usize;
    let head_dim = config.head_dim as usize;
    let vocab_size = config.vocab_size as usize;

    // Collect all required parameters with expected shapes
    let mut required_params: Vec<(String, Vec<usize>)> = Vec::new();

    // Embedding parameters (always required)
    required_params.push((
        "embedding.weight".to_string(),
        vec![vocab_size, hidden_size],
    ));
    required_params.push(("final_norm.weight".to_string(), vec![hidden_size]));

    // Per-layer parameters
    for i in 0..num_layers {
        let prefix = format!("layers.{}", i);

        // Attention weights
        required_params.push((
            format!("{}.self_attn.q_proj.weight", prefix),
            vec![num_heads * head_dim, hidden_size],
        ));
        required_params.push((
            format!("{}.self_attn.k_proj.weight", prefix),
            vec![num_kv_heads * head_dim, hidden_size],
        ));
        required_params.push((
            format!("{}.self_attn.v_proj.weight", prefix),
            vec![num_kv_heads * head_dim, hidden_size],
        ));
        required_params.push((
            format!("{}.self_attn.o_proj.weight", prefix),
            vec![hidden_size, num_heads * head_dim],
        ));

        // MLP weights
        required_params.push((
            format!("{}.mlp.gate_proj.weight", prefix),
            vec![intermediate_size, hidden_size],
        ));
        required_params.push((
            format!("{}.mlp.up_proj.weight", prefix),
            vec![intermediate_size, hidden_size],
        ));
        required_params.push((
            format!("{}.mlp.down_proj.weight", prefix),
            vec![hidden_size, intermediate_size],
        ));

        // Layer norm weights
        required_params.push((
            format!("{}.input_layernorm.weight", prefix),
            vec![hidden_size],
        ));
        required_params.push((
            format!("{}.post_attention_layernorm.weight", prefix),
            vec![hidden_size],
        ));

        // QK norm weights (if enabled)
        if config.use_qk_norm {
            required_params.push((
                format!("{}.self_attn.q_norm.weight", prefix),
                vec![head_dim],
            ));
            required_params.push((
                format!("{}.self_attn.k_norm.weight", prefix),
                vec![head_dim],
            ));
        }
    }

    // LM head (only if not tied to embeddings)
    if !config.tie_word_embeddings {
        required_params.push(("lm_head.weight".to_string(), vec![vocab_size, hidden_size]));
    }

    // Track missing and mismatched parameters for comprehensive error reporting
    let mut missing_params: Vec<String> = Vec::new();
    let mut shape_mismatches: Vec<String> = Vec::new();

    // Validate all required parameters
    for (name, expected_shape) in required_params.iter() {
        match params.get(name) {
            None => {
                missing_params.push(name.clone());
            }
            Some(arr) => {
                let actual_shape_result = arr.shape();
                match actual_shape_result {
                    Ok(shape_data) => {
                        let actual_shape: Vec<usize> =
                            shape_data.as_ref().iter().map(|&x| x as usize).collect();
                        if actual_shape != *expected_shape {
                            shape_mismatches.push(format!(
                                "{}: expected {:?}, got {:?}",
                                name, expected_shape, actual_shape
                            ));
                        }
                    }
                    Err(e) => {
                        shape_mismatches.push(format!("{}: failed to get shape: {}", name, e));
                    }
                }
            }
        }
    }

    // Build comprehensive error message if there are any issues
    if !missing_params.is_empty() || !shape_mismatches.is_empty() {
        let mut error_msg = String::from("Parameter validation failed:\n");

        if !missing_params.is_empty() {
            error_msg.push_str(&format!(
                "\nMissing {} parameter(s):\n",
                missing_params.len()
            ));
            for name in missing_params.iter().take(10) {
                error_msg.push_str(&format!("  - {}\n", name));
            }
            if missing_params.len() > 10 {
                error_msg.push_str(&format!("  ... and {} more\n", missing_params.len() - 10));
            }
        }

        if !shape_mismatches.is_empty() {
            error_msg.push_str(&format!(
                "\nShape mismatch for {} parameter(s):\n",
                shape_mismatches.len()
            ));
            for mismatch in shape_mismatches.iter().take(10) {
                error_msg.push_str(&format!("  - {}\n", mismatch));
            }
            if shape_mismatches.len() > 10 {
                error_msg.push_str(&format!("  ... and {} more\n", shape_mismatches.len() - 10));
            }
        }

        return Err(Error::new(Status::InvalidArg, error_msg));
    }

    info!(
        "✅ Parameter validation passed: {} parameters verified",
        required_params.len()
    );
    Ok(())
}

#[napi]
impl Qwen3Model {
    /// Load a pretrained model from disk
    ///
    /// This loads a model from a directory containing:
    /// - config.json: Model configuration
    /// - weights.mlx (optional): MLX format weights with data arrays
    /// - weights.safetensors (optional): SafeTensors format (not yet supported)
    ///
    /// # Arguments
    /// * `model_path` - Path to the model directory
    ///
    /// # Returns
    /// * A fully initialized Qwen3Model with loaded weights
    #[napi]
    pub fn load<'env>(env: &'env Env, model_path: String) -> Result<PromiseRaw<'env, Qwen3Model>> {
        env.spawn_future_with_callback(async move {
          tokio::task::spawn_blocking(move || {
            let path = Path::new(&model_path);

            // Check if path exists
            if !path.exists() {
                return Err(napi::Error::from_reason(format!(
                    "Model path does not exist: {}",
                    model_path
                )));
            }

            // Load configuration
            let config_path = path.join("config.json");
            if !config_path.exists() {
                return Err(napi::Error::from_reason(format!(
                    "Config file not found: {}",
                    config_path.display()
                )));
            }

            let config_data = fs::read_to_string(&config_path)?;

            let raw_config: Value = serde_json::from_str(&config_data)?;

            // Map HuggingFace config to our format (supporting both snake_case and camelCase)
            // First, parse token IDs so we can use bos_token_id as default for pad_token_id
            let bos_token_id = raw_config["bos_token_id"]
                .as_i64()
                .or_else(|| raw_config["bosTokenId"].as_i64())
                .unwrap_or(151643) as i32; // Qwen3 default

            let config = Qwen3Config {
                vocab_size: raw_config["vocab_size"]
                    .as_i64()
                    .or_else(|| raw_config["vocabSize"].as_i64())
                    .unwrap_or(0) as i32,
                hidden_size: raw_config["hidden_size"]
                    .as_i64()
                    .or_else(|| raw_config["hiddenSize"].as_i64())
                    .unwrap_or(0) as i32,
                // Support both HuggingFace naming (num_hidden_layers) and our naming (num_layers)
                num_layers: raw_config["num_hidden_layers"]
                    .as_i64()
                    .or_else(|| raw_config["num_layers"].as_i64())  // Our checkpoint format
                    .or_else(|| raw_config["numLayers"].as_i64())
                    .unwrap_or(0) as i32,
                num_heads: raw_config["num_attention_heads"]
                    .as_i64()
                    .or_else(|| raw_config["num_heads"].as_i64())   // Our checkpoint format
                    .or_else(|| raw_config["numHeads"].as_i64())
                    .unwrap_or(0) as i32,
                num_kv_heads: raw_config["num_key_value_heads"]
                    .as_i64()
                    .or_else(|| raw_config["num_kv_heads"].as_i64()) // Our checkpoint format
                    .or_else(|| raw_config["numKvHeads"].as_i64())
                    .unwrap_or(0) as i32,
                intermediate_size: raw_config["intermediate_size"]
                    .as_i64()
                    .or_else(|| raw_config["intermediateSize"].as_i64())
                    .unwrap_or(0) as i32,
                rms_norm_eps: raw_config["rms_norm_eps"]
                    .as_f64()
                    .or_else(|| raw_config["rmsNormEps"].as_f64())
                    .unwrap_or(1e-6),
                rope_theta: raw_config["rope_theta"]
                    .as_f64()
                    .or_else(|| raw_config["ropeTheta"].as_f64())
                    .unwrap_or(1000000.0), // Qwen3 uses 1M, not 10K
                max_position_embeddings: raw_config["max_position_embeddings"]
                    .as_i64()
                    .or_else(|| raw_config["maxPositionEmbeddings"].as_i64())
                    .unwrap_or(2048) as i32,
                // Parse head_dim from config (e.g., 128 for Qwen3-0.6B)
                // If not specified, calculate from hidden_size / num_heads
                head_dim: parse_head_dim(&raw_config)?,
                // Qwen3 ALWAYS uses QK normalization (core architectural feature)
                // This is NOT optional - transformers source shows q_norm and k_norm are always initialized
                use_qk_norm: raw_config["use_qk_norm"]
                    .as_bool()
                    .or_else(|| raw_config["useQkNorm"].as_bool())
                    .unwrap_or(true),
                tie_word_embeddings: raw_config["tie_word_embeddings"]
                    .as_bool()
                    .or_else(|| raw_config["tieWordEmbeddings"].as_bool())
                    .unwrap_or(false),
                // Qwen3 uses bos_token_id as pad_token_id (standard practice)
                pad_token_id: raw_config["pad_token_id"]
                    .as_i64()
                    .or_else(|| raw_config["padTokenId"].as_i64())
                    .unwrap_or(bos_token_id as i64) as i32,
                eos_token_id: raw_config["eos_token_id"]
                    .as_i64()
                    .or_else(|| raw_config["eosTokenId"].as_i64())
                    .unwrap_or(151645) as i32, // Qwen3 default
                bos_token_id,

                // Paged attention options (opt-in)
                use_paged_attention: raw_config["use_paged_attention"]
                    .as_bool()
                    .or_else(|| raw_config["usePagedAttention"].as_bool()),
                paged_cache_memory_mb: raw_config["paged_cache_memory_mb"]
                    .as_i64()
                    .or_else(|| raw_config["pagedCacheMemoryMb"].as_i64())
                    .map(|x| x as u32),
                paged_block_size: raw_config["paged_block_size"]
                    .as_i64()
                    .or_else(|| raw_config["pagedBlockSize"].as_i64())
                    .map(|x| x as u32),
                use_fp8_cache: raw_config["use_fp8_cache"]
                    .as_bool()
                    .or_else(|| raw_config["useFp8Cache"].as_bool()),
            };

            // Try to load weights from SafeTensors format first (preferred)
            // Support both weights.safetensors (our format) and model.safetensors (HuggingFace format)
            let safetensors_path = if path.join("weights.safetensors").exists() {
                path.join("weights.safetensors")
            } else {
                path.join("model.safetensors")
            };

            if safetensors_path.exists() {
                info!("📦 Loading model from SafeTensors format: {} (mmap)", safetensors_path.display());

                // Load all tensors via mmap-backed lazy loader
                let mut param_map = load_safetensors_lazy(&safetensors_path)?;

                info!("  Loaded {} tensors", param_map.len());

                // Log ALL top-level (non-layer) tensor names for debugging
                info!("📋 Top-level tensors (not in layers):");
                for (name, array) in param_map.iter() {
                    if !name.contains("layers.") {
                        match array.shape() {
                            Ok(shape) => {
                                let shape_vec: Vec<i64> = shape.as_ref().to_vec();
                                info!("  {}: {:?}", name, shape_vec);
                            }
                            Err(_) => {
                                info!("  {}: <error getting shape>", name);
                            }
                        }
                    }
                }

                // Map HuggingFace parameter names to our naming convention
                // HuggingFace: model.embed_tokens.weight -> Our: embedding.weight
                // HuggingFace: model.layers.X... -> Our: layers.X...
                // HuggingFace: model.norm.weight -> Our: final_norm.weight (but we use 'norm' in persistence)
                let mut mapped_params: HashMap<String, MxArray> = HashMap::new();

                for (name, array) in param_map.drain() {
                    let mapped_name = if let Some(stripped) = name.strip_prefix("model.") {
                        // Strip 'model.' prefix and rename special cases
                        if stripped == "embed_tokens.weight" {
                            "embedding.weight".to_string()
                        } else if stripped.starts_with("embed_tokens.") {
                            stripped.replace("embed_tokens", "embedding")
                        } else if stripped == "norm.weight" {
                            // HuggingFace uses 'model.norm.weight', we use 'final_norm.weight'
                            "final_norm.weight".to_string()
                        } else {
                            stripped.to_string()
                        }
                    } else {
                        // Keep original name (for lm_head, etc.)
                        name
                    };
                    mapped_params.insert(mapped_name, array);
                }

                info!(
                    "✅ Loaded {} parameters from SafeTensors (mapped {} names)",
                    mapped_params.len(),
                    mapped_params.len()
                );

                // Validate all required parameters were loaded with correct shapes
                validate_loaded_parameters(&mapped_params, &config)?;

                // Load tokenizer
                let tokenizer_path = path.join("tokenizer.json");
                if !tokenizer_path.exists() {
                    return Err(Error::new(
                        Status::InvalidArg,
                        format!("Tokenizer file not found: {}", tokenizer_path.display()),
                    ));
                }

                info!("📝 Loading tokenizer from: {}", tokenizer_path.display());
                let tokenizer = Qwen3Tokenizer::load_from_file_sync(tokenizer_path.to_str().unwrap())?;
                info!("✅ Tokenizer loaded successfully");

                return Ok::<_, Error>((config, mapped_params, tokenizer));
            }

            // Fall back to MLX format
            let mlx_weights_path = path.join("weights.mlx");
            if mlx_weights_path.exists() {
                info!("📦 Loading model from MLX format: {}", mlx_weights_path.display());

                // Load MLX weight file
                let weights_data = fs::read_to_string(&mlx_weights_path)?;

                let weights_json: Value = serde_json::from_str(&weights_data)?;

                // Validate format
                if !weights_json["version"].is_string() || !weights_json["weights"].is_object() {
                    return Err(napi::Error::from_reason(
                        "Invalid MLX weight file format".to_string(),
                    ));
                }

                let weights_obj = weights_json["weights"].as_object().unwrap();

                // Convert weights to HashMap<String, MxArray>
                let mut param_map: HashMap<String, MxArray> = HashMap::new();

                for (key, weight_data) in weights_obj.iter() {
                    // Extract shape, dtype, and data
                    let shape_arr = weight_data["shape"].as_array().ok_or_else(|| {
                        napi::Error::from_reason(format!("Missing shape for {}", key))
                    })?;

                    let shape: Vec<i64> = shape_arr.iter().filter_map(|v| v.as_i64()).collect();

                    let dtype = weight_data["dtype"].as_str().ok_or_else(|| {
                        napi::Error::from_reason(format!("Missing dtype for {}", key))
                    })?;

                    let data_arr = weight_data["data"]
                        .as_array()
                        .ok_or_else(|| napi::Error::from_reason(format!("Missing data for {}", key)))?;

                    // Convert to Float32Array
                    let float_data: Vec<f32> = data_arr
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();

                    // Validate size
                    let expected_size: usize = shape.iter().map(|&x| x as usize).product();
                    if float_data.len() != expected_size {
                        return Err(napi::Error::from_reason(format!(
                            "Data size mismatch for {}: expected {}, got {}",
                            key,
                            expected_size,
                            float_data.len()
                        )));
                    }

                    // Create MxArray
                    let mx_array = if dtype == "float32" || dtype == "float16" || dtype == "bfloat16" {
                        MxArray::from_float32(&float_data, &shape)?
                    } else {
                        return Err(napi::Error::from_reason(format!(
                            "Unsupported dtype for {}: {}. Supported: float32, float16, bfloat16",
                            key, dtype
                        )));
                    };

                    param_map.insert(key.clone(), mx_array);
                }

                info!(
                    "✅ Loaded {} parameters from MLX format",
                    weights_obj.len()
                );

                // Validate all required parameters were loaded with correct shapes
                validate_loaded_parameters(&param_map, &config)?;

                // Load tokenizer
                let tokenizer_path = path.join("tokenizer.json");
                if !tokenizer_path.exists() {
                    return Err(Error::new(
                        Status::InvalidArg,
                        format!("Tokenizer file not found: {}", tokenizer_path.display()),
                    ));
                }

                info!("📝 Loading tokenizer from: {}", tokenizer_path.display());
                let tokenizer = Qwen3Tokenizer::load_from_file_sync(tokenizer_path.to_str().unwrap())?;
                info!("✅ Tokenizer loaded successfully");

                return Ok::<_, Error>((config, param_map, tokenizer));
            }

            // No supported weight files found
            Err(Error::new(Status::InvalidArg, format!(
                "No supported weight file found in {}. Expected weights.safetensors or weights.mlx",
                model_path
            )))
          })
            .await
            .map_err(|err| Error::new(Status::GenericFailure, format!("Failed to load model: {err}")))
            .flatten()
        }, |_, (config, param_map, tokenizer)| {
            // Create model with configuration
            let mut model = Qwen3Model::new(config)?;
            model.load_parameters(
                param_map.iter().map(|(k, v)| (k.clone(), v)).collect::<HashMap<_, _>>(),
            )?;

            // Materialize all mmap-backed weight arrays so the first inference
            // prefill timing is not inflated by lazy disk reads.
            {
                let arrays: Vec<&MxArray> = param_map.values().collect();
                crate::array::memory::materialize_weights(&arrays);
            }

            // Set the tokenizer
            model.tokenizer = Some(Arc::new(tokenizer));
            Ok(model)
        })
    }

    /// Save model configuration and weights to disk
    ///
    /// This saves:
    /// - config.json: Model configuration
    /// - weights.safetensors: Full model weights in SafeTensors format
    /// - weights.mlx: Parameter metadata (for reference)
    ///
    /// # Arguments
    /// * `save_path` - Directory to save the model
    #[napi]
    pub fn save_model<'env>(
        &self,
        env: &'env Env,
        save_path: String,
    ) -> Result<PromiseRaw<'env, ()>> {
        // Get all parameters
        let params = self.get_parameters()?;

        // Validate all parameters for NaN/Inf before saving
        // This prevents saving corrupted checkpoints that would fail on resume
        for (name, param) in params.iter() {
            let data = param.to_float32()?;
            let invalid_count = data
                .iter()
                .filter(|v| v.is_nan() || v.is_infinite())
                .count();
            if invalid_count > 0 {
                return Err(napi::Error::new(
                    Status::GenericFailure,
                    format!(
                        "Cannot save model: parameter '{}' contains {} NaN/Inf values. \
                        Model weights are corrupted, likely due to training instability. \
                        Consider reducing learning rate or using an earlier checkpoint.",
                        name, invalid_count
                    ),
                ));
            }
        }

        // Clone parameters for async task
        let params_clone: HashMap<String, MxArray> =
            params.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        // Create weights metadata (for reference)
        let mut weights_metadata = serde_json::Map::new();
        for (key, array) in params.iter() {
            let shape_data = array.shape()?;
            let shape: Vec<i64> = shape_data.as_ref().to_vec();
            let dtype = array.dtype()?;

            let mut param_info = serde_json::Map::new();
            param_info.insert("shape".to_string(), serde_json::json!(shape));
            param_info.insert("dtype".to_string(), serde_json::json!(dtype as i32));

            weights_metadata.insert(key.clone(), serde_json::Value::Object(param_info));
        }

        let config = self.get_config();
        let weights_json = serde_json::json!({
            "version": "1.0",
            "config": config,
            "weights": weights_metadata,
            "note": "Full weights are in weights.safetensors"
        });

        let promise = env.spawn_future(async move {
            tokio::task::spawn_blocking(move || {
                // Create directory if it doesn't exist
                let path = Path::new(&save_path);
                fs::create_dir_all(path)?;

                info!("Saving model to {}", save_path);

                // 1. Save configuration as JSON
                let config_path = path.join("config.json");
                let config_json = serde_json::to_string_pretty(&config)?;
                fs::write(&config_path, config_json)?;
                info!("Saved config.json");

                // 2. Save full weights in SafeTensors format
                let safetensors_path = path.join("weights.safetensors");
                let metadata = Some(serde_json::json!({
                    "format": "mlx-node",
                    "version": "1.0"
                }));
                save_safetensors(&safetensors_path, &params_clone, metadata)?;
                info!("Saved weights.safetensors");

                // 3. Save weights metadata (for reference)
                let weights_str = serde_json::to_string_pretty(&weights_json)?;
                let weights_path = path.join("weights.mlx");
                fs::write(&weights_path, weights_str)?;
                info!("Saved weights.mlx metadata");

                Ok::<_, Error>(())
            })
            .map_err(|err| {
                napi::Error::new(
                    Status::GenericFailure,
                    format!("Failed to save model: {}", err),
                )
            })
            .await
            .flatten()?;
            Ok(())
        })?;

        Ok(promise)
    }

    /// Validate that a set of parameters has all required weights with correct shapes
    ///
    /// This is useful for validating parameters before loading them into a model,
    /// or for checking that saved weights are valid before training.
    ///
    /// # Arguments
    /// * `params` - HashMap of parameter names to MxArray values
    ///
    /// # Returns
    /// * Ok(()) if all validations pass
    /// * Err with descriptive message if validation fails
    #[napi]
    pub fn validate_parameters(&self, params: HashMap<String, &MxArray>) -> Result<()> {
        let owned_params: HashMap<String, MxArray> = params
            .iter()
            .map(|(k, v)| (k.clone(), (*v).clone()))
            .collect();
        let config = self.get_config();
        validate_loaded_parameters(&owned_params, &config)
    }
}

/// Parse head_dim from config, validating that hidden_size is divisible by num_heads
fn parse_head_dim(raw_config: &Value) -> Result<i32> {
    // If head_dim is explicitly specified, use it directly
    if let Some(hd) = raw_config["head_dim"]
        .as_i64()
        .or_else(|| raw_config["headDim"].as_i64())
    {
        return Ok(hd as i32);
    }

    // Otherwise, calculate from hidden_size / num_heads with validation
    let hs = raw_config["hidden_size"]
        .as_i64()
        .or_else(|| raw_config["hiddenSize"].as_i64())
        .unwrap_or(1024) as i32;
    let nh = raw_config["num_attention_heads"]
        .as_i64()
        .or_else(|| raw_config["numAttentionHeads"].as_i64())
        .unwrap_or(16) as i32;

    if nh == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "num_attention_heads cannot be zero",
        ));
    }

    if hs % nh != 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                hs, nh
            ),
        ));
    }

    Ok(hs / nh)
}
