/**
 * Model Format Conversion
 *
 * Converts HuggingFace SafeTensors models to MLX format with optional quantization.
 * Supports dtype conversion, FP8 dequantization, model-specific weight sanitization,
 * and offline quantization (4-bit affine or MXFP8).
 * Handles both single-file and sharded models.
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

    /// Enable quantization of converted weights
    pub quantize: Option<bool>,

    /// Quantization bits: 4 (default) or 8
    pub quant_bits: Option<i32>,

    /// Quantization group size (default: 64 for affine, 32 for mxfp8)
    pub quant_group_size: Option<i32>,

    /// Quantization mode: "affine" (default) or "mxfp8"
    pub quant_mode: Option<String>,
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
    let do_quantize = options.quantize.unwrap_or(false);
    let quant_mode = options.quant_mode.unwrap_or_else(|| "affine".to_string());

    // Validate quant_mode before it reaches FFI
    if do_quantize && quant_mode != "affine" && quant_mode != "mxfp8" {
        return Err(Error::from_reason(format!(
            "Invalid quant_mode '{}': must be 'affine' or 'mxfp8'",
            quant_mode
        )));
    }

    let quant_bits = options
        .quant_bits
        .unwrap_or(if quant_mode == "mxfp8" { 8 } else { 4 });
    let quant_group_size = options
        .quant_group_size
        .unwrap_or(if quant_mode == "mxfp8" { 32 } else { 64 });

    if do_quantize && quant_group_size <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid quant_group_size '{}': must be > 0",
            quant_group_size
        )));
    }
    if do_quantize && quant_bits <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid quant_bits '{}': must be > 0",
            quant_bits
        )));
    }

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

    // For models with a sanitizer that handles FP8 dequant + dtype conversion
    // (e.g. qwen3_5_moe), skip the generic dtype conversion and let the sanitizer do it.
    let has_custom_sanitizer = matches!(model_type.as_deref(), Some("qwen3_5_moe" | "qwen3_5"));

    // Convert tensors to target dtype
    info!("Converting tensors to {}...", target_dtype);

    let mut converted_tensors: HashMap<String, MxArray> = HashMap::new();
    let mut tensor_names = Vec::new();

    for (name, array) in tensors.into_iter() {
        // Skip lm_head.weight if embeddings are tied
        // When tied, the model should use embed_tokens.weight via as_linear()
        if tie_word_embeddings && name == "lm_head.weight" {
            if verbose {
                info!("  Skipping {} (tied embeddings)", name);
            }
            continue;
        }

        // If a custom sanitizer handles dtype conversion, pass tensors through as-is
        if has_custom_sanitizer {
            converted_tensors.insert(name.clone(), array);
            tensor_names.push(name);
            continue;
        }

        let current_dtype = array.dtype()?;

        if verbose {
            let shape = array.shape()?;
            info!("  {} {:?} {:?}", name, shape.as_ref(), current_dtype);
        }

        // Convert to target dtype if needed
        let converted = match target_dtype.as_str() {
            "float32" | "f32" => {
                if current_dtype != DType::Float32 {
                    if verbose {
                        info!("    Converting {:?} -> Float32", current_dtype);
                    }
                    array.astype(DType::Float32)?
                } else {
                    array
                }
            }
            "float16" | "f16" => {
                if current_dtype != DType::Float16 {
                    if verbose {
                        info!("    Converting {:?} -> Float16", current_dtype);
                    }
                    array.astype(DType::Float16)?
                } else {
                    array
                }
            }
            "bfloat16" | "bf16" => {
                if current_dtype != DType::BFloat16 {
                    if verbose {
                        info!("    Converting {:?} -> BFloat16", current_dtype);
                    }
                    array.astype(DType::BFloat16)?
                } else {
                    array
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
        tensor_names.push(name);
    }

    // Apply model-specific weight sanitization
    let converted_tensors = match model_type.as_deref() {
        Some("paddleocr-vl") => {
            info!(
                "Applying PaddleOCR-VL weight sanitization (key renaming, Q/K/V merging, conv2d transposition)..."
            );
            load_paddleocr_vl_weights(converted_tensors)?
        }
        Some("qwen3_5_moe" | "qwen3_5") => {
            info!(
                "Applying Qwen3.5 weight sanitization (FP8 dequant, key remapping, expert stacking)..."
            );
            sanitize_qwen35_moe(converted_tensors, &config, &target_dtype)?
        }
        Some(other) => {
            return Err(Error::from_reason(format!(
                "Unknown model type: '{}'. Supported: paddleocr-vl, qwen3_5_moe, qwen3_5",
                other
            )));
        }
        None => converted_tensors,
    };

    // Apply quantization if requested
    let mut converted_tensors = converted_tensors;
    if do_quantize {
        info!(
            "Quantizing weights: bits={}, group_size={}, mode={}",
            quant_bits, quant_group_size, quant_mode
        );
        quantize_weights(
            &mut converted_tensors,
            quant_bits,
            quant_group_size,
            &quant_mode,
        )?;
    }

    // Update tensor names after sanitization/quantization
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

    // Write config.json — inject quantization metadata if quantized
    let output_config_path = output_dir.join("config.json");
    if do_quantize {
        let mut output_config = config.clone();
        output_config["quantization"] = serde_json::json!({
            "group_size": quant_group_size,
            "bits": quant_bits,
            "mode": quant_mode,
        });
        output_config["quantization_config"] = output_config["quantization"].clone();
        let config_str = serde_json::to_string_pretty(&output_config)
            .map_err(|e| Error::from_reason(format!("Failed to serialize config: {}", e)))?;
        fs::write(&output_config_path, config_str)
            .map_err(|e| Error::from_reason(format!("Failed to write config.json: {}", e)))?;
        info!("Wrote config.json with quantization metadata");
    } else {
        info!("Copying config.json to: {}", output_config_path.display());
        fs::copy(&config_path, &output_config_path)
            .map_err(|e| Error::from_reason(format!("Failed to copy config.json: {}", e)))?;
    }

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

/// Determine whether a weight key should be quantized.
fn should_quantize(key: &str) -> bool {
    // Only .weight keys (not .scales, .biases, etc.)
    if !key.ends_with(".weight") {
        return false;
    }

    // Exclude embeddings and lm_head (output projection shares vocab dimension)
    if key.contains("embed_tokens") || key.contains("embedding.") || key.contains("lm_head") {
        return false;
    }

    // Exclude norms (layernorm covers input_layernorm/post_attention_layernorm)
    if key.contains("layernorm") || key.contains("rms_norm") || key.contains("_norm.") {
        return false;
    }

    // Exclude conv1d (not a standard matmul shape)
    if key.contains("conv1d") {
        return false;
    }

    // Exclude A_log and dt_bias (GatedDeltaNet parameters)
    if key.contains("A_log") || key.contains("dt_bias") {
        return false;
    }

    // Exclude in_proj_a, in_proj_b, and in_proj_ba (low-rank projections in GatedDeltaNet)
    if key.contains("in_proj_a.") || key.contains("in_proj_b.") || key.contains("in_proj_ba.") {
        return false;
    }

    true
}

/// Check if a key is a router gate (should be quantized at 8-bit for accuracy).
fn is_router_gate(key: &str) -> bool {
    // Router gates: mlp.gate.weight, shared_expert_gate.weight
    let stripped = key.strip_suffix(".weight").unwrap_or(key);
    stripped.ends_with(".mlp.gate") || stripped.ends_with(".shared_expert_gate")
}

/// Quantize weights in-place using MLX's quantize operation.
///
/// Replaces qualifying `.weight` tensors with quantized (uint32 packed) versions
/// and inserts `.scales` (and `.biases` for affine mode) tensors.
fn quantize_weights(
    weights: &mut HashMap<String, MxArray>,
    bits: i32,
    group_size: i32,
    mode: &str,
) -> Result<()> {
    use std::ffi::CString;

    let mode_c =
        CString::new(mode).map_err(|_| Error::from_reason("Invalid quantize mode string"))?;

    // Gate quantization: 8-bit affine with group_size=64.
    // For MXFP8 mode, router gates are EXCLUDED entirely — MXFP8 quantization
    // of small routing weights destroys expert selection and produces garbage output.
    let gate_mode_c = CString::new("affine").unwrap();
    let gate_bits: i32 = 8;
    let gate_group_size: i32 = 64;
    let is_mxfp8 = mode == "mxfp8";

    // Collect keys to quantize
    let keys_to_quantize: Vec<(String, bool)> = weights
        .keys()
        .filter(|k| should_quantize(k))
        .filter(|k| !(is_mxfp8 && is_router_gate(k))) // Skip gates in MXFP8 mode
        .map(|k| {
            let is_gate = is_router_gate(k);
            (k.clone(), is_gate)
        })
        .collect();

    info!(
        "Quantizing {} weights ({}-bit {}, group_size={})",
        keys_to_quantize.len(),
        bits,
        mode,
        group_size
    );

    let mut count = 0;
    for (key, is_gate) in &keys_to_quantize {
        let array = match weights.remove(key) {
            Some(a) => a,
            None => continue,
        };

        // Check dimensionality — must be 2D+
        let ndim = array.ndim()? as usize;
        if ndim < 2 {
            weights.insert(key.clone(), array);
            continue;
        }

        // Check last dim divisibility
        let last_dim = array.shape_at((ndim - 1) as u32)? as i32;
        let (q_bits, q_gs, q_mode) = if *is_gate {
            (gate_bits, gate_group_size, &gate_mode_c)
        } else {
            (bits, group_size, &mode_c)
        };

        if last_dim % q_gs != 0 {
            weights.insert(key.clone(), array);
            continue;
        }

        // Eval to materialize (prevents lazy graph OOM)
        array.eval();

        // Quantize
        let mut out_quantized: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_scales: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_biases: *mut mlx_sys::mlx_array = std::ptr::null_mut();

        let ok = unsafe {
            mlx_sys::mlx_quantize(
                array.as_raw_ptr(),
                q_gs,
                q_bits,
                q_mode.as_ptr(),
                &mut out_quantized,
                &mut out_scales,
                &mut out_biases,
            )
        };

        if !ok {
            return Err(Error::from_reason(format!(
                "mlx_quantize failed for tensor '{}'",
                key
            )));
        }

        let q_weight = MxArray::from_handle(out_quantized, "quantize_weight")?;
        let q_scales = MxArray::from_handle(out_scales, "quantize_scales")?;

        let prefix = key.strip_suffix(".weight").unwrap_or(key);
        weights.insert(format!("{}.weight", prefix), q_weight);
        weights.insert(format!("{}.scales", prefix), q_scales);

        if !out_biases.is_null() {
            let q_biases = MxArray::from_handle(out_biases, "quantize_biases")?;
            weights.insert(format!("{}.biases", prefix), q_biases);
        }

        count += 1;

        if count % 50 == 0 {
            crate::array::memory::synchronize_and_clear_cache();
            info!(
                "  Quantized {}/{} tensors...",
                count,
                keys_to_quantize.len()
            );
        }
    }

    crate::array::memory::synchronize_and_clear_cache();
    info!(
        "Quantization complete: {} tensors quantized, {} total keys",
        count,
        weights.len()
    );

    Ok(())
}

/// FP8 E4M3 block-wise dequantization: weight * scale_inv with block_size=128
///
/// 1. from_fp8(weight) → target_dtype
/// 2. Pad to 128-block alignment
/// 3. Reshape into blocks, multiply by scale_inv
/// 4. Unpad and return
fn dequant_fp8(weight: &MxArray, scale_inv: &MxArray, target_dtype: DType) -> Result<MxArray> {
    // Step 1: Convert FP8 uint8 → target float type
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

    // Step 2: Pad to block alignment
    let pad_bottom = (bs - (m % bs)) % bs;
    let pad_side = (bs - (n % bs)) % bs;

    let weight = if pad_bottom > 0 || pad_side > 0 {
        weight.pad(&[0, pad_bottom as i32, 0, pad_side as i32], 0.0)?
    } else {
        weight
    };

    // Step 3: Reshape into [m_blocks, bs, n_blocks, bs]
    let m_padded = m + pad_bottom;
    let n_padded = n + pad_side;
    let weight = weight.reshape(&[
        (m_padded / bs) as i64,
        bs as i64,
        (n_padded / bs) as i64,
        bs as i64,
    ])?;

    // Step 4: Multiply by scale_inv [m_blocks, 1, n_blocks, 1] (broadcast)
    let scale = scale_inv.expand_dims(1)?.expand_dims(3)?;
    let weight = weight.mul(&scale)?;

    // Step 5: Reshape back and unpad
    let weight = weight.reshape(&[m_padded as i64, n_padded as i64])?;
    let weight = if pad_bottom > 0 || pad_side > 0 {
        weight.slice(&[0, 0], &[m as i64, n as i64])?
    } else {
        weight
    };

    weight.astype(target_dtype)
}

/// Sanitize Qwen3.5 / Qwen3.5-MoE model weights.
///
/// Output matches mlx-vlm `format: "mlx"` convention (sanitize is skipped on load).
///
/// Handles:
/// 1. VL key prefix remapping to mlx-vlm convention (language_model.model.*, vision_tower.*)
/// 2. Skipping MTP weights
/// 3. FP8 E4M3 dequantization (weight + weight_scale_inv → target dtype)
/// 4. Individual expert stacking (experts.{i}.{proj} → switch_mlp.{proj})
/// 5. mlx-vlm sanitization: norm weight +1.0 shift, conv1d weight transpose
fn sanitize_qwen35_moe(
    weights: HashMap<String, MxArray>,
    config: &serde_json::Value,
    target_dtype_str: &str,
) -> Result<HashMap<String, MxArray>> {
    let target_dtype = match target_dtype_str {
        "float32" | "f32" => DType::Float32,
        "float16" | "f16" => DType::Float16,
        "bfloat16" | "bf16" => DType::BFloat16,
        other => {
            warn!("Unknown target dtype '{}', defaulting to bfloat16", other);
            DType::BFloat16
        }
    };

    // Get num_experts from config (check text_config first, then top-level)
    let num_experts_val = config
        .get("text_config")
        .and_then(|tc| tc.get("num_experts"))
        .or_else(|| config.get("num_experts"))
        .and_then(|v| v.as_u64());
    if num_experts_val.is_none() {
        warn!("num_experts not found in config.json, defaulting to 256");
    }
    let num_experts = num_experts_val.unwrap_or(256) as usize;

    let num_hidden_layers_val = config
        .get("text_config")
        .and_then(|tc| tc.get("num_hidden_layers"))
        .or_else(|| config.get("num_hidden_layers"))
        .and_then(|v| v.as_u64());
    if num_hidden_layers_val.is_none() {
        warn!("num_hidden_layers not found in config.json, defaulting to 40");
    }
    let num_hidden_layers = num_hidden_layers_val.unwrap_or(40) as usize;

    info!(
        "  num_experts={}, num_hidden_layers={}, target_dtype={:?}",
        num_experts, num_hidden_layers, target_dtype
    );

    let has_fp8 = weights.keys().any(|k| k.contains("weight_scale_inv"));
    if has_fp8 {
        info!("  Detected FP8 weights — will dequantize");
    }

    // Step 1: Remap key prefixes, skip MTP
    let mut new_weights: HashMap<String, MxArray> = HashMap::new();
    for (key, value) in weights.into_iter() {
        // Skip MTP (multi-token prediction)
        if key.starts_with("mtp.") || key.starts_with("mtp_") {
            continue;
        }

        // Vision tower: model.visual.* → vision_tower.*, already vision_tower.* stays as-is
        // Skip position_ids (unused in MLX)
        if key.contains("position_ids") {
            continue;
        }
        if key.starts_with("model.visual") {
            let new_key = key.replacen("model.visual", "vision_tower", 1);
            new_weights.insert(new_key, value);
            continue;
        }
        if key.starts_with("vision_tower") {
            new_weights.insert(key, value);
            continue;
        }

        // Language model: strip all known prefixes to bare key
        let bare = key
            .strip_prefix("model.language_model.")
            .or_else(|| key.strip_prefix("language_model.model."))
            .or_else(|| key.strip_prefix("language_model."))
            .or_else(|| key.strip_prefix("model."))
            .unwrap_or(&key);

        // Re-prefix: lm_head directly under language_model., everything else under language_model.model.
        let new_key = if bare.starts_with("lm_head") {
            format!("language_model.{}", bare)
        } else {
            format!("language_model.model.{}", bare)
        };

        new_weights.insert(new_key, value);
    }

    info!("  After key remapping: {} tensors", new_weights.len());

    // Step 2: FP8 dequantization (in-place to avoid extra HashMap allocation)
    if has_fp8 {
        let scale_keys: Vec<String> = new_weights
            .keys()
            .filter(|k| k.contains("weight_scale_inv"))
            .cloned()
            .collect();

        info!("  Dequantizing {} FP8 weight pairs...", scale_keys.len());

        for scale_key in &scale_keys {
            let weight_key = scale_key.replace("_scale_inv", "");
            let scale_inv = new_weights.remove(scale_key).unwrap();
            if let Some(weight) = new_weights.remove(&weight_key) {
                let dequant = dequant_fp8(&weight, &scale_inv, target_dtype)?;
                // Eval immediately to prevent lazy chain accumulation (OOM with many FP8 pairs)
                dequant.eval();
                new_weights.insert(weight_key, dequant);
            } else {
                warn!(
                    "Orphaned FP8 scale_inv key (no matching weight): {}",
                    scale_key
                );
            }
        }

        // Convert remaining non-FP8 weights to target dtype
        let keys: Vec<String> = new_weights.keys().cloned().collect();
        for k in keys {
            let v = new_weights.get(&k).unwrap();
            let current_dtype = v.dtype()?;
            if current_dtype != target_dtype {
                let converted = v.astype(target_dtype)?;
                new_weights.insert(k, converted);
            }
        }

        info!("  After FP8 dequantization: {} tensors", new_weights.len());
    } else {
        // Non-FP8: convert all weights to target dtype
        let keys: Vec<String> = new_weights.keys().cloned().collect();
        for k in keys {
            let v = new_weights.get(&k).unwrap();
            let current_dtype = v.dtype()?;
            if current_dtype != target_dtype {
                let converted = v.astype(target_dtype)?;
                new_weights.insert(k, converted);
            }
        }
    }

    // Step 3: Stack individual expert weights
    for l in 0..num_hidden_layers {
        let prefix = format!("language_model.model.layers.{}.mlp", l);
        let first_expert_key = format!("{}.experts.0.gate_proj.weight", prefix);

        if !new_weights.contains_key(&first_expert_key) {
            continue;
        }

        info!("  Layer {}: stacking {} experts...", l, num_experts);

        for proj in &["gate_proj", "up_proj", "down_proj"] {
            let mut to_stack: Vec<MxArray> = Vec::with_capacity(num_experts);
            for e in 0..num_experts {
                let k = format!("{}.experts.{}.{}.weight", prefix, e, proj);
                match new_weights.remove(&k) {
                    Some(w) => to_stack.push(w),
                    None => {
                        return Err(Error::from_reason(format!("Missing expert weight: {}", k)));
                    }
                }
            }
            let refs: Vec<&MxArray> = to_stack.iter().collect();
            let stacked = MxArray::stack(refs, Some(0))?;
            new_weights.insert(format!("{}.switch_mlp.{}.weight", prefix, proj), stacked);
        }
    }

    // Clean up any remaining individual expert keys (shouldn't be any after stacking)
    let expert_keys: Vec<String> = new_weights
        .keys()
        .filter(|k| k.contains(".mlp.experts.") && k.ends_with(".weight"))
        .cloned()
        .collect();
    for k in expert_keys {
        new_weights.remove(&k);
    }

    info!("  After expert stacking: {} tensors", new_weights.len());

    // Step 4: mlx-vlm sanitization (since format:"mlx" skips sanitize on load)
    // - Norm weights: +1.0 shift (HF stores raw values, MLX RMSNorm expects weight+1)
    // - Conv1d weights: transpose last two dims (HF [out, in/g, k] → MLX [out, k, in/g])
    let norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    ];
    let keys: Vec<String> = new_weights.keys().cloned().collect();
    for k in keys {
        if k.contains("patch_embed.proj.weight") {
            // Conv3d/Conv2d: PyTorch [out, in, t, h, w] → MLX [out, t, h, w, in]
            let v = new_weights.get(&k).unwrap();
            let ndim = v.ndim()? as usize;
            if ndim == 5 {
                let transposed = v.transpose(Some(&[0, 2, 3, 4, 1]))?;
                new_weights.insert(k, transposed);
            }
        } else if k.contains("conv1d.weight") {
            let v = new_weights.get(&k).unwrap();
            let ndim = v.ndim()? as usize;
            if ndim >= 2 {
                let last_dim = v.shape_at((ndim - 1) as u32)?;
                if last_dim != 1 {
                    // moveaxis(2, 1) for 3D: [out, in/g, k] → [out, k, in/g]
                    let transposed = v.transpose(Some(&[0, 2, 1]))?;
                    new_weights.insert(k, transposed);
                }
            }
        } else if norm_suffixes.iter().any(|sfx| k.ends_with(sfx)) {
            let v = new_weights.get(&k).unwrap();
            if v.ndim()? == 1 {
                let shifted = v.add_scalar(1.0)?;
                new_weights.insert(k, shifted);
            }
        }
    }

    info!("  After sanitization: {} tensors", new_weights.len());

    Ok(new_weights)
}
