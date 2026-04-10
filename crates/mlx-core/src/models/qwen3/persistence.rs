/**
 * Qwen3 Model - Persistence Methods
 *
 * Contains methods for saving and loading model weights and configuration.
 */
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use napi::{Env, Status, bindgen_prelude::*, tokio};
use napi_derive::napi;
use serde_json::Value;
use tracing::info;

use crate::array::MxArray;
use crate::tokenizer::Qwen3Tokenizer;
use crate::utils::safetensors::load_safetensors_lazy;

use super::model::{Qwen3Cmd, Qwen3Inner, handle_qwen3_cmd};
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
        env.spawn_future(async move { load_with_thread(&model_path).await })
    }

    /// Save model configuration and weights to disk
    ///
    /// This saves:
    /// - config.json: Model configuration
    /// - weights.safetensors: Full model weights in SafeTensors format
    /// - weights.mlx: Parameter metadata (for reference)
    ///
    /// Dispatches to the dedicated model thread — all MxArray reads must
    /// happen on the thread that owns them to avoid the MLX cross-thread
    /// `CommandEncoder` crash.
    ///
    /// # Arguments
    /// * `save_path` - Directory to save the model
    #[napi]
    pub fn save_model<'env>(
        &self,
        env: &'env Env,
        save_path: String,
    ) -> Result<PromiseRaw<'env, ()>> {
        // Dispatch to dedicated model thread so MxArray reads happen on the
        // thread that owns them.
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.thread.send(Qwen3Cmd::SaveModel {
            save_path,
            reply: tx,
        })?;
        let promise = env.spawn_future(async move {
            rx.await
                .map_err(|_| napi::Error::from_reason("Model thread exited unexpectedly"))?
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

/// Create a random-init Qwen3 model and save it to disk.
///
/// Spawns a dedicated `ModelThread<Qwen3Cmd>` whose init builds a fresh
/// random-weight `Qwen3Inner` directly, then dispatches `Qwen3Cmd::SaveModel`
/// on that thread. The thread is dropped at the end of the promise, so the
/// in-memory model is released once the checkpoint has been written. Used by
/// TypeScript test fixtures that need an on-disk checkpoint without keeping a
/// NAPI model instance alive.
#[napi]
pub fn create_random_qwen3_checkpoint<'env>(
    env: &'env Env,
    config: Qwen3Config,
    save_path: String,
) -> Result<PromiseRaw<'env, ()>> {
    let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_init(
        move || {
            let inner = Qwen3Inner::new(config)?;
            Ok((inner, ()))
        },
        handle_qwen3_cmd,
    );

    env.spawn_future(async move {
        init_rx
            .await
            .map_err(|_| napi::Error::from_reason("Model thread exited during init"))??;

        let (tx, rx) = tokio::sync::oneshot::channel();
        thread.send(Qwen3Cmd::SaveModel {
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

/// Parse Qwen3Config from a serde_json::Value (shared between load paths).
fn parse_config(raw_config: &Value) -> Result<Qwen3Config> {
    let bos_token_id = raw_config["bos_token_id"]
        .as_i64()
        .or_else(|| raw_config["bosTokenId"].as_i64())
        .unwrap_or(151643) as i32;

    Ok(Qwen3Config {
        vocab_size: raw_config["vocab_size"]
            .as_i64()
            .or_else(|| raw_config["vocabSize"].as_i64())
            .unwrap_or(0) as i32,
        hidden_size: raw_config["hidden_size"]
            .as_i64()
            .or_else(|| raw_config["hiddenSize"].as_i64())
            .unwrap_or(0) as i32,
        num_layers: raw_config["num_hidden_layers"]
            .as_i64()
            .or_else(|| raw_config["num_layers"].as_i64())
            .or_else(|| raw_config["numLayers"].as_i64())
            .unwrap_or(0) as i32,
        num_heads: raw_config["num_attention_heads"]
            .as_i64()
            .or_else(|| raw_config["num_heads"].as_i64())
            .or_else(|| raw_config["numHeads"].as_i64())
            .unwrap_or(0) as i32,
        num_kv_heads: raw_config["num_key_value_heads"]
            .as_i64()
            .or_else(|| raw_config["num_kv_heads"].as_i64())
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
            .unwrap_or(1000000.0),
        max_position_embeddings: raw_config["max_position_embeddings"]
            .as_i64()
            .or_else(|| raw_config["maxPositionEmbeddings"].as_i64())
            .unwrap_or(2048) as i32,
        head_dim: parse_head_dim(raw_config)?,
        use_qk_norm: raw_config["use_qk_norm"]
            .as_bool()
            .or_else(|| raw_config["useQkNorm"].as_bool())
            .unwrap_or(true),
        tie_word_embeddings: raw_config["tie_word_embeddings"]
            .as_bool()
            .or_else(|| raw_config["tieWordEmbeddings"].as_bool())
            .unwrap_or(false),
        pad_token_id: raw_config["pad_token_id"]
            .as_i64()
            .or_else(|| raw_config["padTokenId"].as_i64())
            .unwrap_or(bos_token_id as i64) as i32,
        eos_token_id: raw_config["eos_token_id"]
            .as_i64()
            .or_else(|| raw_config["eosTokenId"].as_i64())
            .unwrap_or(151645) as i32,
        bos_token_id,
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
    })
}

/// Load weights from SafeTensors, mapping HuggingFace names to our naming convention.
fn load_safetensors_mapped(path: &Path) -> Result<HashMap<String, MxArray>> {
    let safetensors_path = if path.join("weights.safetensors").exists() {
        path.join("weights.safetensors")
    } else {
        path.join("model.safetensors")
    };

    if !safetensors_path.exists() {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "No supported weight file found in {}. Expected weights.safetensors or model.safetensors",
                path.display()
            ),
        ));
    }

    info!(
        "Loading model from SafeTensors format: {} (mmap)",
        safetensors_path.display()
    );

    let mut param_map = load_safetensors_lazy(&safetensors_path)?;
    info!("  Loaded {} tensors", param_map.len());

    let mut mapped_params: HashMap<String, MxArray> = HashMap::new();
    for (name, array) in param_map.drain() {
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
        } else {
            name
        };
        mapped_params.insert(mapped_name, array);
    }

    info!(
        "Loaded {} parameters from SafeTensors (mapped)",
        mapped_params.len()
    );

    Ok(mapped_params)
}

/// Spawn a dedicated model thread, load all weights inside init_fn.
///
/// Returns a thin `Qwen3Model` NAPI shell with the thread handle.
pub async fn load_with_thread(model_path: &str) -> Result<Qwen3Model> {
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
            if !config_path.exists() {
                return Err(Error::from_reason(format!(
                    "Config file not found: {}",
                    config_path.display()
                )));
            }

            let config_data = fs::read_to_string(&config_path)
                .map_err(|e| Error::from_reason(format!("Failed to read config: {}", e)))?;
            let raw_config: Value = serde_json::from_str(&config_data)
                .map_err(|e| Error::from_reason(format!("Failed to parse config: {}", e)))?;

            let config = parse_config(&raw_config)?;

            info!(
                "Qwen3 config: {} layers, hidden={}, heads={}, kv_heads={}",
                config.num_layers, config.hidden_size, config.num_heads, config.num_kv_heads,
            );

            // Load weights
            let mapped_params = load_safetensors_mapped(path)?;

            // Validate parameters
            validate_loaded_parameters(&mapped_params, &config)?;

            // Load tokenizer
            let tokenizer_path = path.join("tokenizer.json");
            if !tokenizer_path.exists() {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("Tokenizer file not found: {}", tokenizer_path.display()),
                ));
            }
            info!("Loading tokenizer from: {}", tokenizer_path.display());
            let tokenizer = Qwen3Tokenizer::load_from_file_sync(tokenizer_path.to_str().unwrap())?;
            info!("Tokenizer loaded successfully");

            // Create Qwen3Inner
            let mut inner = Qwen3Inner::new(config.clone())?;
            inner.set_tokenizer(Arc::new(tokenizer.clone()));

            // Load parameters into inner
            let num_layers = config.num_layers as usize;
            let _head_dim = config.head_dim as usize;

            // Embedding
            if let Some(arr) = mapped_params.get("embedding.weight") {
                inner.embedding.set_weight(arr)?;
            }

            // Final norm
            if let Some(arr) = mapped_params.get("final_norm.weight") {
                inner.final_norm.set_weight(arr)?;
            }

            // LM head (only if not tied)
            if !config.tie_word_embeddings
                && let Some(arr) = mapped_params.get("lm_head.weight")
            {
                inner.lm_head.set_weight(arr)?;
            }

            // Per-layer weights
            for i in 0..num_layers {
                let prefix = format!("layers.{}", i);
                let layer = &mut inner.layers[i];

                if let Some(w) = mapped_params.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
                    layer.self_attn.set_q_proj_weight(w)?;
                }
                if let Some(w) = mapped_params.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
                    layer.self_attn.set_k_proj_weight(w)?;
                }
                if let Some(w) = mapped_params.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
                    layer.self_attn.set_v_proj_weight(w)?;
                }
                if let Some(w) = mapped_params.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
                    layer.self_attn.set_o_proj_weight(w)?;
                }
                if config.use_qk_norm {
                    if let Some(w) =
                        mapped_params.get(&format!("{}.self_attn.q_norm.weight", prefix))
                    {
                        layer.self_attn.set_q_norm_weight(w)?;
                    }
                    if let Some(w) =
                        mapped_params.get(&format!("{}.self_attn.k_norm.weight", prefix))
                    {
                        layer.self_attn.set_k_norm_weight(w)?;
                    }
                }
                if let Some(w) = mapped_params.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
                    layer.mlp.set_gate_proj_weight(w)?;
                }
                if let Some(w) = mapped_params.get(&format!("{}.mlp.up_proj.weight", prefix)) {
                    layer.mlp.set_up_proj_weight(w)?;
                }
                if let Some(w) = mapped_params.get(&format!("{}.mlp.down_proj.weight", prefix)) {
                    layer.mlp.set_down_proj_weight(w)?;
                }
                if let Some(w) = mapped_params.get(&format!("{}.input_layernorm.weight", prefix)) {
                    layer.set_input_layernorm_weight(w)?;
                }
                if let Some(w) =
                    mapped_params.get(&format!("{}.post_attention_layernorm.weight", prefix))
                {
                    layer.set_post_attention_layernorm_weight(w)?;
                }
            }

            // Materialize all mmap-backed weight arrays
            {
                let arrays: Vec<&MxArray> = mapped_params.values().collect();
                crate::array::memory::materialize_weights(&arrays);
            }

            let config_out = config.clone();
            let tokenizer_out = Some(Arc::new(tokenizer));

            Ok((inner, (config_out, tokenizer_out)))
        },
        handle_qwen3_cmd,
    );

    let (config, tokenizer) = init_rx
        .await
        .map_err(|_| Error::from_reason("Model thread exited during load"))??;

    Ok(Qwen3Model {
        thread,
        config: config.clone(),
        tokenizer,
    })
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
