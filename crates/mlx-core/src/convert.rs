/**
 * Model Format Conversion
 *
 * Converts HuggingFace SafeTensors models to MLX float32 format.
 * This is essential for GRPO training which requires full float32 precision.
 * Supports both single-file and sharded models.
 */
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::Deserialize;
use tracing::{info, warn};

use crate::array::{DType, MxArray};
use crate::models::paddleocr_vl::persistence::load_paddleocr_vl_weights;
use crate::utils::safetensors::{SafeTensorsFile, save_safetensors};

/// Structure for parsing model.safetensors.index.json
#[derive(Debug, Deserialize)]
struct ShardedModelIndex {
    /// Maps tensor names to shard filenames
    weight_map: HashMap<String, String>,
}

#[napi(object)]
pub struct ConversionOptions {
    /// Input directory containing model files (config.json, model.safetensors)
    pub input_dir: String,

    /// Output directory for converted model
    pub output_dir: String,

    /// Target dtype for conversion (default: "float32")
    pub dtype: Option<String>,

    /// Whether to verbose logging (default: false)
    pub verbose: Option<bool>,

    /// Model type for model-specific weight sanitization (e.g., "paddleocr-vl")
    pub model_type: Option<String>,
}

#[napi(object)]
pub struct ConversionResult {
    /// Number of tensors converted
    pub num_tensors: i32,

    /// Total number of parameters
    pub num_parameters: i64,

    /// Output model path
    pub output_path: String,

    /// List of converted tensor names
    pub tensor_names: Vec<String>,
}

/// Convert a HuggingFace SafeTensors model to MLX format
///
/// This function:
/// 1. Loads SafeTensors model from input directory
/// 2. Converts all tensors to specified dtype (default: float32)
/// 3. Saves converted model to output directory
/// 4. Copies config.json and tokenizer files
///
/// # Arguments
/// * `options` - Conversion options (input_dir, output_dir, dtype, verbose)
///
/// # Returns
/// * ConversionResult with statistics about the conversion
///
/// # Example
/// ```typescript
/// import { convertModel } from '../../index.cjs';
///
/// const result = await convertModel({
///   inputDir: '.cache/models/qwen3-0.6b',
///   outputDir: '.cache/models/qwen3-0.6b-mlx',
///   dtype: 'float32',
///   verbose: true
/// });
///
/// console.log(`Converted ${result.numTensors} tensors (${result.numParameters} parameters)`);
/// ```
#[napi]
pub async fn convert_model(options: ConversionOptions) -> Result<ConversionResult> {
    let input_dir = PathBuf::from(&options.input_dir);
    let output_dir = PathBuf::from(&options.output_dir);
    let target_dtype = options.dtype.unwrap_or_else(|| "float32".to_string());
    let verbose = options.verbose.unwrap_or(false);
    let model_type = options.model_type;

    // Validate input directory
    if !input_dir.exists() {
        return Err(Error::from_reason(format!(
            "Input directory does not exist: {}",
            input_dir.display()
        )));
    }

    // Check for required files
    let config_path = input_dir.join("config.json");
    if !config_path.exists() {
        return Err(Error::from_reason(format!(
            "config.json not found in input directory: {}",
            input_dir.display()
        )));
    }

    info!("Loading model from: {}", input_dir.display());
    info!("Target dtype: {}", target_dtype);

    // Create output directory
    fs::create_dir_all(&output_dir).map_err(|e| {
        Error::from_reason(format!(
            "Failed to create output directory {}: {}",
            output_dir.display(),
            e
        ))
    })?;

    // Load config to check for tied embeddings
    let config_data = fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_data)?;
    let tie_word_embeddings = config["tie_word_embeddings"].as_bool().unwrap_or(false);

    if tie_word_embeddings && verbose {
        info!("Model uses tied embeddings - will skip lm_head.weight");
    }

    // Load tensors - handle both single file and sharded models
    let tensors: HashMap<String, MxArray>;
    let num_tensors: usize;
    let num_parameters: usize;

    let index_path = input_dir.join("model.safetensors.index.json");
    let single_weights_path = input_dir.join("model.safetensors");
    let alt_weights_path = input_dir.join("weights.safetensors");

    if single_weights_path.exists() {
        // Single file model
        info!(
            "Loading single SafeTensors file: {}",
            single_weights_path.display()
        );
        let st_file = SafeTensorsFile::load(&single_weights_path)?;
        num_tensors = st_file.tensors.len();
        num_parameters = st_file.num_parameters();
        info!(
            "Loaded {} tensors ({} parameters)",
            num_tensors, num_parameters
        );
        tensors = st_file.load_tensors(&single_weights_path)?;
    } else if alt_weights_path.exists() {
        // Alternative single file model
        info!(
            "Loading single SafeTensors file: {}",
            alt_weights_path.display()
        );
        let st_file = SafeTensorsFile::load(&alt_weights_path)?;
        num_tensors = st_file.tensors.len();
        num_parameters = st_file.num_parameters();
        info!(
            "Loaded {} tensors ({} parameters)",
            num_tensors, num_parameters
        );
        tensors = st_file.load_tensors(&alt_weights_path)?;
    } else if index_path.exists() {
        // Sharded model
        info!("Loading sharded model from index: {}", index_path.display());

        // Parse the index file
        let index_data = fs::read_to_string(&index_path)?;
        let index: ShardedModelIndex = serde_json::from_str(&index_data)
            .map_err(|e| Error::from_reason(format!("Failed to parse model index: {}", e)))?;

        // Find unique shard files
        let shard_files: HashSet<String> = index.weight_map.values().cloned().collect();
        info!("Found {} shard files", shard_files.len());

        // Load tensors from each shard
        let mut all_tensors: HashMap<String, MxArray> = HashMap::new();
        let mut total_params = 0usize;

        for shard_name in shard_files.iter() {
            let shard_path = input_dir.join(shard_name);
            if !shard_path.exists() {
                return Err(Error::from_reason(format!(
                    "Shard file not found: {}",
                    shard_path.display()
                )));
            }

            info!("  Loading shard: {}", shard_name);
            let st_file = SafeTensorsFile::load(&shard_path)?;
            total_params += st_file.num_parameters();

            let shard_tensors = st_file.load_tensors(&shard_path)?;
            all_tensors.extend(shard_tensors);
        }

        num_tensors = all_tensors.len();
        num_parameters = total_params;
        tensors = all_tensors;

        info!(
            "Loaded {} tensors ({} parameters) from {} shards",
            num_tensors,
            num_parameters,
            shard_files.len()
        );
    } else {
        return Err(Error::from_reason(format!(
            "No model weights found in input directory.\nExpected: model.safetensors, weights.safetensors, or model.safetensors.index.json\nPath: {}",
            input_dir.display()
        )));
    }

    // Convert tensors to target dtype
    info!("Converting tensors to {}...", target_dtype);

    let mut converted_tensors: HashMap<String, MxArray> = HashMap::new();
    let mut tensor_names = Vec::new();

    for (name, array) in tensors.iter() {
        // Skip lm_head.weight if embeddings are tied
        // When tied, the model should use embed_tokens.weight via as_linear()
        if tie_word_embeddings && name == "lm_head.weight" {
            if verbose {
                info!("  Skipping {} (tied embeddings)", name);
            }
            continue;
        }
        let current_dtype = array.dtype()?;

        if verbose {
            let shape = array.shape()?;
            info!("  {} {:?} {:?}", name, shape.as_ref(), current_dtype);
        }

        // Convert to float32 if needed
        let converted = match target_dtype.as_str() {
            "float32" | "f32" => {
                if current_dtype != DType::Float32 {
                    if verbose {
                        info!("    Converting {:?} -> Float32", current_dtype);
                    }
                    // astype converts to f32
                    array.astype(DType::Float32)?
                } else {
                    array.clone()
                }
            }
            "float16" | "f16" => {
                if current_dtype != DType::Float16 {
                    if verbose {
                        info!("    Converting {:?} -> Float16", current_dtype);
                    }
                    array.astype(DType::Float16)?
                } else {
                    array.clone()
                }
            }
            "bfloat16" | "bf16" => {
                if current_dtype != DType::BFloat16 {
                    if verbose {
                        info!("    Converting {:?} -> BFloat16", current_dtype);
                    }
                    array.astype(DType::BFloat16)?
                } else {
                    array.clone()
                }
            }
            _ => {
                return Err(Error::from_reason(format!(
                    "Unsupported target dtype: {}. Supported: float32, float16, bfloat16",
                    target_dtype
                )));
            }
        };

        converted_tensors.insert(name.clone(), converted);
        tensor_names.push(name.clone());
    }

    // Apply model-specific weight sanitization
    let converted_tensors = match model_type.as_deref() {
        Some("paddleocr-vl") => {
            info!(
                "Applying PaddleOCR-VL weight sanitization (key renaming, Q/K/V merging, conv2d transposition)..."
            );
            load_paddleocr_vl_weights(converted_tensors)?
        }
        Some(other) => {
            return Err(Error::from_reason(format!(
                "Unknown model type: '{}'. Supported: paddleocr-vl",
                other
            )));
        }
        None => converted_tensors,
    };

    // Update tensor names after sanitization
    let mut tensor_names: Vec<String> = converted_tensors.keys().cloned().collect();
    tensor_names.sort();

    // Save converted model
    let output_weights_path = output_dir.join("model.safetensors");
    info!(
        "Saving converted model to: {}",
        output_weights_path.display()
    );

    // Create metadata with dtype info
    let metadata = serde_json::json!({
        "format": "mlx",
        "dtype": target_dtype,
        "converted_from": "huggingface",
        "source": input_dir.file_name().unwrap_or_default().to_string_lossy(),
    });

    save_safetensors(&output_weights_path, &converted_tensors, Some(metadata))?;

    // Copy config.json
    let output_config_path = output_dir.join("config.json");
    info!("Copying config.json to: {}", output_config_path.display());
    fs::copy(&config_path, &output_config_path)
        .map_err(|e| Error::from_reason(format!("Failed to copy config.json: {}", e)))?;

    // Copy tokenizer and model config files if they exist
    let config_files = [
        // Tokenizer files
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        // Generation config
        "generation_config.json",
        // VLM-specific files
        "preprocessor_config.json",
        "processor_config.json",
    ];

    for file_name in config_files.iter() {
        let src = input_dir.join(file_name);
        let dst = output_dir.join(file_name);

        if src.exists() {
            if verbose {
                info!("Copying {}", file_name);
            }
            fs::copy(&src, &dst)
                .map_err(|e| Error::from_reason(format!("Failed to copy {}: {}", file_name, e)))?;
        } else if verbose {
            warn!("Skipping {} (not found)", file_name);
        }
    }

    info!("✓ Conversion complete!");
    info!(
        "  Converted {} tensors ({} parameters)",
        num_tensors, num_parameters
    );
    info!("  Output: {}", output_dir.display());

    Ok(ConversionResult {
        num_tensors: tensor_names.len() as i32,
        num_parameters: num_parameters as i64,
        output_path: output_dir.to_string_lossy().to_string(),
        tensor_names,
    })
}
