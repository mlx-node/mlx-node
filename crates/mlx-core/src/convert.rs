/**
 * Model Format Conversion
 *
 * Converts HuggingFace SafeTensors models to MLX float32 format.
 * This is essential for GRPO training which requires full float32 precision.
 */
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tracing::{info, warn};

use crate::array::{DType, MxArray};
use crate::utils::safetensors::{SafeTensorsFile, save_safetensors};

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

    // Find model weights file (try both common names)
    let weights_path = if input_dir.join("model.safetensors").exists() {
        input_dir.join("model.safetensors")
    } else if input_dir.join("weights.safetensors").exists() {
        input_dir.join("weights.safetensors")
    } else if input_dir.join("model.safetensors.index.json").exists() {
        return Err(Error::from_reason(
            "Sharded models not yet supported. Please provide a single safetensors file."
                .to_string(),
        ));
    } else {
        return Err(Error::from_reason(format!(
            "No model weights found in input directory.\nExpected: model.safetensors or weights.safetensors\nPath: {}",
            input_dir.display()
        )));
    };

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

    // Load SafeTensors file
    info!("Loading SafeTensors from: {}", weights_path.display());
    let st_file = SafeTensorsFile::load(&weights_path)?;

    let num_tensors = st_file.tensors.len();
    let num_parameters = st_file.num_parameters();

    info!(
        "Loaded {} tensors ({} parameters)",
        num_tensors, num_parameters
    );

    // Load config to check for tied embeddings
    let config_data = fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_data)?;
    let tie_word_embeddings = config["tie_word_embeddings"].as_bool().unwrap_or(false);

    if tie_word_embeddings && verbose {
        info!("Model uses tied embeddings - will skip lm_head.weight");
    }

    // Load all tensors and convert to target dtype
    info!("Converting tensors to {}...", target_dtype);
    let tensors = st_file.load_tensors(&weights_path)?;

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

    // Copy tokenizer files if they exist
    let tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ];

    for file_name in tokenizer_files.iter() {
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

    // Sort tensor names for consistent output
    tensor_names.sort();

    Ok(ConversionResult {
        num_tensors: num_tensors as i32,
        num_parameters: num_parameters as i64,
        output_path: output_dir.to_string_lossy().to_string(),
        tensor_names,
    })
}
