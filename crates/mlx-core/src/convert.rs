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
use crate::utils::safetensors::{load_safetensors_lazy, save_safetensors};

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

    /// Quantization recipe for per-layer mixed-bit quantization.
    /// Options: mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6, qwen3_5
    pub quant_recipe: Option<String>,

    /// Path to an imatrix GGUF file for AWQ-style pre-scaling.
    /// Improves quantization quality by amplifying important weight channels.
    pub imatrix_path: Option<String>,
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
    let quant_recipe = options.quant_recipe;
    let imatrix_path = options.imatrix_path;

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

    // Validate recipe
    if let Some(ref recipe) = quant_recipe {
        if !do_quantize {
            return Err(Error::from_reason(
                "--q-recipe requires --quantize to be enabled".to_string(),
            ));
        }
        if quant_mode == "mxfp8" {
            return Err(Error::from_reason(
                "--q-recipe is incompatible with --q-mode mxfp8".to_string(),
            ));
        }
        // Validate recipe name early
        let valid = [
            "mixed_2_6",
            "mixed_3_4",
            "mixed_3_6",
            "mixed_4_6",
            "qwen3_5",
            "unsloth",
        ];
        if !valid.contains(&recipe.as_str()) {
            return Err(Error::from_reason(format!(
                "Unknown quantization recipe: '{}'. Available: {}",
                recipe,
                valid.join(", ")
            )));
        }
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
        // Single file model — lazy load
        info!(
            "Loading SafeTensors file (lazy): {}",
            single_weights_path.display()
        );
        tensors = load_safetensors_lazy(&single_weights_path)?;
        num_parameters = tensors
            .values()
            .map(|a| a.size().unwrap_or(0) as usize)
            .sum();
        num_tensors = tensors.len();
        info!(
            "Loaded {} tensors ({} parameters)",
            num_tensors, num_parameters
        );
    } else if alt_weights_path.exists() {
        // Alternative single file model — lazy load
        info!(
            "Loading SafeTensors file (lazy): {}",
            alt_weights_path.display()
        );
        tensors = load_safetensors_lazy(&alt_weights_path)?;
        num_parameters = tensors
            .values()
            .map(|a| a.size().unwrap_or(0) as usize)
            .sum();
        num_tensors = tensors.len();
        info!(
            "Loaded {} tensors ({} parameters)",
            num_tensors, num_parameters
        );
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

        // Load tensors from each shard using MLX's lazy loader (near-zero memory).
        // Tensor data is deferred — read from disk only when eval'd.
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

            info!("  Loading shard (lazy): {}", shard_name);
            let shard_tensors = load_safetensors_lazy(&shard_path)?;
            // Count parameters from shapes (lazy arrays have shape but no data yet)
            for arr in shard_tensors.values() {
                total_params += arr.size()? as usize;
            }
            all_tensors.extend(shard_tensors);
        }

        num_tensors = all_tensors.len();
        num_parameters = total_params;
        tensors = all_tensors;

        info!(
            "Loaded {} tensors ({} parameters) from {} shards (lazy)",
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

    // Apply AWQ pre-scaling if imatrix provided
    let mut converted_tensors = converted_tensors;
    if let Some(ref imatrix_path) = imatrix_path {
        let imatrix = crate::utils::imatrix::parse_imatrix(imatrix_path)?;
        let num_layers = infer_num_layers_from_weights(&converted_tensors);
        apply_awq_prescaling(&mut converted_tensors, &imatrix, 0.5, num_layers)?;
    }

    // Apply quantization if requested
    let mut per_layer_overrides: HashMap<String, serde_json::Value> = HashMap::new();
    if do_quantize {
        info!(
            "Quantizing weights: bits={}, group_size={}, mode={}{}",
            quant_bits,
            quant_group_size,
            quant_mode,
            quant_recipe
                .as_deref()
                .map(|r| format!(", recipe={}", r))
                .unwrap_or_default()
        );

        if let Some(ref recipe) = quant_recipe {
            let weight_keys: Vec<String> = converted_tensors.keys().cloned().collect();
            let predicate =
                build_predicate_for_recipe(recipe, &weight_keys, quant_bits, quant_group_size)
                    .map_err(Error::from_reason)?;
            per_layer_overrides = quantize_weights_with_recipe_pub(
                &mut converted_tensors,
                quant_bits,
                quant_group_size,
                &quant_mode,
                &*predicate,
            )?;
        } else {
            quantize_weights(
                &mut converted_tensors,
                quant_bits,
                quant_group_size,
                &quant_mode,
            )?;
        }
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
        let mut quant_obj = serde_json::json!({
            "group_size": quant_group_size,
            "bits": quant_bits,
            "mode": quant_mode,
        });
        if let Some(obj) = quant_obj.as_object_mut() {
            for (path, override_val) in &per_layer_overrides {
                obj.insert(
                    crate::utils::normalize_override_key(path),
                    override_val.clone(),
                );
            }
        }
        output_config["quantization"] = quant_obj.clone();
        output_config["quantization_config"] = quant_obj;
        let config_str = serde_json::to_string_pretty(&output_config)
            .map_err(|e| Error::from_reason(format!("Failed to serialize config: {}", e)))?;
        fs::write(&output_config_path, config_str)
            .map_err(|e| Error::from_reason(format!("Failed to write config.json: {}", e)))?;
        if per_layer_overrides.is_empty() {
            info!("Wrote config.json with quantization metadata");
        } else {
            info!(
                "Wrote config.json with quantization metadata ({} per-layer overrides)",
                per_layer_overrides.len()
            );
        }
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

    // Exclude vision encoder weights (keep bf16 for quality)
    if key.contains("vision_tower") || key.contains("visual.") {
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

// ── Per-Layer Quantization Recipes ──────────────────────────────────────────

/// Per-weight quantization decision returned by recipe predicates.
#[derive(Debug, Clone)]
pub(crate) enum QuantDecision {
    /// Skip quantization — leave weight as-is (e.g. embeddings, norms)
    Skip,
    /// Use the model's default quantization parameters
    Default,
    /// Use custom bits/group_size/mode for this weight
    Custom {
        bits: i32,
        group_size: i32,
        mode: String,
    },
}

/// Extract the layer index from a weight key like "model.layers.5.self_attn.q_proj.weight" → Some(5).
fn extract_layer_index(key: &str) -> Option<usize> {
    // Look for ".layers.N." or "layers.N."
    let idx = key.find("layers.")?;
    let after = &key[idx + 7..]; // skip "layers."
    let end = after.find('.')?;
    after[..end].parse().ok()
}

/// Infer number of layers from weight keys by finding the max layer index.
fn infer_num_layers(keys: &[String]) -> usize {
    keys.iter()
        .filter_map(|k| extract_layer_index(k))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0)
}

/// Build a recipe predicate matching mlx-lm's mixed-bit quantization recipes.
///
/// Recipes: `mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`
/// Format: `mixed_{low}_{high}` where low/high are bit widths.
///
/// Logic (from mlx-lm):
/// - `use_more_bits`: first 1/8 layers, last 1/8, every 3rd in between
/// - High bits: `v_proj`, `down_proj` in eligible layers + `lm_head`
/// - Low bits: everything else that's quantizable
pub(crate) fn build_recipe_predicate(
    recipe: &str,
    weight_keys: &[String],
    default_group_size: i32,
) -> std::result::Result<Box<dyn Fn(&str) -> QuantDecision + Send + Sync>, String> {
    let (low_bits, high_bits) = match recipe {
        "mixed_2_6" => (2, 6),
        "mixed_3_4" => (3, 4),
        "mixed_3_6" => (3, 6),
        "mixed_4_6" => (4, 6),
        _ => {
            return Err(format!(
                "Unknown mlx-lm recipe: '{recipe}'. Available: mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6"
            ));
        }
    };

    let num_layers = infer_num_layers(weight_keys);
    if num_layers == 0 {
        return Err(
            "Cannot infer num_layers from weight keys — no 'layers.N.' patterns found".into(),
        );
    }

    // Determine which layers get more bits (first 1/8, last 1/8, every 3rd in between)
    let first_boundary = num_layers / 8;
    let last_boundary = num_layers - num_layers / 8;
    let mut use_more_bits = vec![false; num_layers];
    for (i, slot) in use_more_bits.iter_mut().enumerate() {
        if i < first_boundary || i >= last_boundary || (i % 3 == 0) {
            *slot = true;
        }
    }

    let gs = default_group_size;

    Ok(Box::new(move |key: &str| -> QuantDecision {
        // lm_head always gets high bits (checked before should_quantize which
        // would otherwise skip it as a non-standard weight)
        if key.contains("lm_head") && key.ends_with(".weight") {
            return QuantDecision::Custom {
                bits: high_bits,
                group_size: gs,
                mode: "affine".to_string(),
            };
        }

        // Non-quantizable weights are skipped
        if !should_quantize(key) {
            return QuantDecision::Skip;
        }

        // Router gates → 8-bit affine (same as existing behavior)
        if is_router_gate(key) {
            return QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            };
        }

        // Check if this layer is eligible for more bits
        if let Some(layer_idx) = extract_layer_index(key)
            && layer_idx < use_more_bits.len()
            && use_more_bits[layer_idx]
        {
            // In eligible layers: v_proj and down_proj get high bits
            if key.contains("v_proj") || key.contains("down_proj") {
                return QuantDecision::Custom {
                    bits: high_bits,
                    group_size: gs,
                    mode: "affine".to_string(),
                };
            }
        }

        // Everything else gets low bits
        QuantDecision::Custom {
            bits: low_bits,
            group_size: gs,
            mode: "affine".to_string(),
        }
    }))
}

/// Build a Qwen3.5-specific quantization recipe.
///
/// Based on Unsloth GGUF benchmarks (https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks):
///
/// - **ssm_out** (`linear_attn.out_proj`): "dramatically increases KLD and the disk space savings
///   is minuscule" → skip quantization entirely
/// - **attn_\***: "especially sensitive for hybrid architectures" → `min(default_bits + 2, 8)`
/// - **attn_gate** (`linear_attn.in_proj_z`): "performs poorly with MXFP4" → higher bits
/// - **ssm_beta, ssm_alpha** (`in_proj_a/b`): already excluded by `should_quantize()`
/// - **Router gates** → 8-bit affine (standard for MoE routing accuracy)
/// - **FFN expert weights**: "generally ok to quantize to 3-bit" → default bits
/// - **ffn_down_exps**: "slightly more sensitive" → `min(default_bits + 1, 8)`
pub(crate) fn build_qwen35_recipe(
    default_bits: i32,
    default_group_size: i32,
) -> Box<dyn Fn(&str) -> QuantDecision + Send + Sync> {
    let high_bits = (default_bits + 2).min(8);
    let down_proj_bits = (default_bits + 1).min(8);
    let gs = default_group_size;

    Box::new(move |key: &str| -> QuantDecision {
        if !should_quantize(key) {
            return QuantDecision::Skip;
        }

        // Router gates → 8-bit affine
        if is_router_gate(key) {
            return QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            };
        }

        // ssm_out (linear_attn.out_proj): "dramatically increases KLD" — skip entirely.
        // Disk savings are minuscule and quality impact is severe.
        if key.contains("linear_attn.out_proj") {
            return QuantDecision::Skip;
        }

        // Attention projections (q_proj, k_proj, v_proj, o_proj) and
        // remaining SSM-sensitive weights (in_proj_qkv, in_proj_z).
        // Note: in_proj_a/b and A_log/dt_bias are already excluded by should_quantize().
        let is_attn_sensitive = key.contains("self_attn.")
            || key.contains("linear_attn.in_proj_qkv")
            || key.contains("linear_attn.in_proj_z");

        if is_attn_sensitive {
            return QuantDecision::Custom {
                bits: high_bits,
                group_size: gs,
                mode: "affine".to_string(),
            };
        }

        // ffn_down_exps: "slightly more sensitive" than other FFN variants
        if key.contains("down_proj") {
            return QuantDecision::Custom {
                bits: down_proj_bits,
                group_size: gs,
                mode: "affine".to_string(),
            };
        }

        // Everything else (ffn_gate_proj, ffn_up_proj, etc.) → default bits
        QuantDecision::Default
    })
}

/// Build the "unsloth" quantization recipe for Qwen3.5 hybrid models.
///
/// Based on Unsloth's GGUF benchmark findings for Qwen3.5's hybrid
/// GatedDeltaNet (linear attention/SSM) + full attention architecture:
/// (https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)
///
/// Key findings from Unsloth's per-tensor KLD analysis:
///
/// **Most sensitive (skip quantization — keep bf16):**
/// - `ssm_out` (`linear_attn.out_proj`): "dramatically increases KLD and the disk
///   space savings is minuscule" — quantizing even to Q8 degrades quality severely
/// - `attn_*` (`self_attn.*`): "quantizing any attn_* is especially sensitive for
///   hybrid architectures, and so leaving them in higher precision works well"
/// - `attn_gate` (`linear_attn.in_proj_z`): "performs poorly with MXFP4"
/// - `ssm_beta`, `ssm_alpha` (`in_proj_a/b`): degrade significantly with low bits
///   (already excluded by `should_quantize()` since they lack `.weight` suffix)
///
/// **Slightly sensitive (default_bits + 1):**
/// - `ffn_down_exps` (`down_proj`): "slightly more sensitive" than other FFN weights
///
/// **Safe to quantize aggressively (default bits):**
/// - `ffn_up_exps`, `ffn_gate_exps`: "generally ok to quantize to 3-bit"
///
/// **Always 8-bit:**
/// - Router gates: standard for MoE routing accuracy
///
/// This recipe matches Unsloth Dynamic 2.0's approach of "upcasting important
/// layers to 8 or 16-bit" while aggressively quantizing FFN expert weights.
/// Results in larger model size than `qwen3_5` recipe but significantly better
/// quality, particularly for the hybrid attention/SSM architecture.
pub(crate) fn build_unsloth_recipe(
    default_bits: i32,
    default_group_size: i32,
) -> Box<dyn Fn(&str) -> QuantDecision + Send + Sync> {
    let down_proj_bits = (default_bits + 1).min(8);
    let gs = default_group_size;

    Box::new(move |key: &str| -> QuantDecision {
        if !should_quantize(key) {
            return QuantDecision::Skip;
        }

        // Router gates → 8-bit affine
        if is_router_gate(key) {
            return QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            };
        }

        // ssm_out (linear_attn.out_proj): skip entirely — keep bf16
        if key.contains("linear_attn.out_proj") {
            return QuantDecision::Skip;
        }

        // All attention and SSM-sensitive projections: skip entirely — keep bf16
        // This matches Unsloth Dynamic 2.0's "upcasted to 16-bit" approach
        let is_attn_sensitive = key.contains("self_attn.")
            || key.contains("linear_attn.in_proj_qkv")
            || key.contains("linear_attn.in_proj_z");

        if is_attn_sensitive {
            return QuantDecision::Skip;
        }

        // ffn_down_exps: "slightly more sensitive" than other FFN variants
        if key.contains("down_proj") {
            return QuantDecision::Custom {
                bits: down_proj_bits,
                group_size: gs,
                mode: "affine".to_string(),
            };
        }

        // Everything else (ffn_gate_proj, ffn_up_proj, etc.) → default bits
        QuantDecision::Default
    })
}

/// Build a recipe predicate from a recipe name. Returns error for unknown recipes.
/// Supports: mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6, qwen3_5, unsloth
pub(crate) fn build_predicate_for_recipe(
    recipe: &str,
    weight_keys: &[String],
    default_bits: i32,
    default_group_size: i32,
) -> std::result::Result<Box<dyn Fn(&str) -> QuantDecision + Send + Sync>, String> {
    match recipe {
        "mixed_2_6" | "mixed_3_4" | "mixed_3_6" | "mixed_4_6" => {
            build_recipe_predicate(recipe, weight_keys, default_group_size)
        }
        "qwen3_5" => Ok(build_qwen35_recipe(default_bits, default_group_size)),
        "unsloth" => Ok(build_unsloth_recipe(default_bits, default_group_size)),
        _ => Err(format!(
            "Unknown quantization recipe: '{recipe}'. Available: mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6, qwen3_5, unsloth"
        )),
    }
}

/// Quantize weights in-place using MLX's quantize operation.
///
/// Replaces qualifying `.weight` tensors with quantized (uint32 packed) versions
/// and inserts `.scales` (and `.biases` for affine mode) tensors.
///
/// When a `predicate` is provided, it determines per-weight quantization decisions.
/// Otherwise, falls back to the default should_quantize + is_router_gate logic.
///
/// Returns a map of per-layer overrides (module path → {bits, group_size, mode})
/// for any weight that used non-default quantization parameters.
fn quantize_weights_inner(
    weights: &mut HashMap<String, MxArray>,
    default_bits: i32,
    default_group_size: i32,
    default_mode: &str,
    predicate: Option<&(dyn Fn(&str) -> QuantDecision + Send + Sync)>,
) -> Result<HashMap<String, serde_json::Value>> {
    use std::ffi::CString;

    let is_mxfp8 = default_mode == "mxfp8";

    // Gate quantization defaults (used when no predicate)
    let gate_bits: i32 = 8;
    let gate_group_size: i32 = 64;

    // Collect quantization decisions for each weight key
    struct QuantEntry {
        key: String,
        bits: i32,
        group_size: i32,
        mode: String,
    }

    let mut entries: Vec<QuantEntry> = Vec::new();

    for key in weights.keys() {
        if let Some(pred) = predicate {
            match pred(key) {
                QuantDecision::Skip => continue,
                QuantDecision::Default => {
                    if !should_quantize(key) {
                        continue;
                    }
                    entries.push(QuantEntry {
                        key: key.clone(),
                        bits: default_bits,
                        group_size: default_group_size,
                        mode: default_mode.to_string(),
                    });
                }
                QuantDecision::Custom {
                    bits,
                    group_size,
                    mode,
                } => {
                    entries.push(QuantEntry {
                        key: key.clone(),
                        bits,
                        group_size,
                        mode,
                    });
                }
            }
        } else {
            // Legacy path: use should_quantize + is_router_gate
            if !should_quantize(key) {
                continue;
            }
            // Skip gates in MXFP8 mode
            if is_mxfp8 && is_router_gate(key) {
                continue;
            }
            if is_router_gate(key) {
                entries.push(QuantEntry {
                    key: key.clone(),
                    bits: gate_bits,
                    group_size: gate_group_size,
                    mode: "affine".to_string(),
                });
            } else {
                entries.push(QuantEntry {
                    key: key.clone(),
                    bits: default_bits,
                    group_size: default_group_size,
                    mode: default_mode.to_string(),
                });
            }
        }
    }

    info!(
        "Quantizing {} weights ({}-bit {}, group_size={})",
        entries.len(),
        default_bits,
        default_mode,
        default_group_size
    );

    let mut per_layer_overrides: HashMap<String, serde_json::Value> = HashMap::new();
    let mut count = 0;

    for entry in &entries {
        let array = match weights.remove(&entry.key) {
            Some(a) => a,
            None => continue,
        };

        // Check dimensionality — must be 2D+
        let ndim = array.ndim()? as usize;
        if ndim < 2 {
            weights.insert(entry.key.clone(), array);
            continue;
        }

        // Check last dim divisibility
        let last_dim = array.shape_at((ndim - 1) as u32)? as i32;
        if last_dim % entry.group_size != 0 {
            weights.insert(entry.key.clone(), array);
            continue;
        }

        let mode_c = CString::new(entry.mode.as_str())
            .map_err(|_| Error::from_reason("Invalid quantize mode string"))?;

        // Eval to materialize (prevents lazy graph OOM)
        array.eval();

        // Quantize
        let mut out_quantized: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_scales: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_biases: *mut mlx_sys::mlx_array = std::ptr::null_mut();

        let ok = unsafe {
            mlx_sys::mlx_quantize(
                array.as_raw_ptr(),
                entry.group_size,
                entry.bits,
                mode_c.as_ptr(),
                &mut out_quantized,
                &mut out_scales,
                &mut out_biases,
            )
        };

        if !ok {
            return Err(Error::from_reason(format!(
                "mlx_quantize failed for tensor '{}'",
                entry.key
            )));
        }

        let q_weight = MxArray::from_handle(out_quantized, "quantize_weight")?;
        let q_scales = MxArray::from_handle(out_scales, "quantize_scales")?;

        let prefix = entry.key.strip_suffix(".weight").unwrap_or(&entry.key);
        weights.insert(format!("{}.weight", prefix), q_weight);
        weights.insert(format!("{}.scales", prefix), q_scales);

        if !out_biases.is_null() {
            let q_biases = MxArray::from_handle(out_biases, "quantize_biases")?;
            weights.insert(format!("{}.biases", prefix), q_biases);
        }

        // Record per-layer override if this weight uses non-default params.
        // Use the weight key as-is (minus .weight suffix) so that the override
        // key matches the module path in mlx-lm/mlx-vlm's class_predicate.
        // Our own persistence.rs strips prefixes on read, so it handles any format.
        if entry.bits != default_bits
            || entry.group_size != default_group_size
            || entry.mode != default_mode
        {
            per_layer_overrides.insert(
                prefix.to_string(),
                serde_json::json!({
                    "bits": entry.bits,
                    "group_size": entry.group_size,
                    "mode": entry.mode,
                }),
            );
        }

        count += 1;

        if count % 50 == 0 {
            crate::array::memory::synchronize_and_clear_cache();
            info!("  Quantized {}/{} tensors...", count, entries.len());
        }
    }

    crate::array::memory::synchronize_and_clear_cache();
    info!(
        "Quantization complete: {} tensors quantized ({} per-layer overrides), {} total keys",
        count,
        per_layer_overrides.len(),
        weights.len()
    );

    Ok(per_layer_overrides)
}

/// Quantize weights with default behavior (no recipe predicate).
fn quantize_weights(
    weights: &mut HashMap<String, MxArray>,
    bits: i32,
    group_size: i32,
    mode: &str,
) -> Result<()> {
    quantize_weights_inner(weights, bits, group_size, mode, None)?;
    Ok(())
}

/// Public wrapper for quantize_weights, accessible from other crate modules (e.g., GGUF converter)
pub(crate) fn quantize_weights_pub(
    weights: &mut HashMap<String, MxArray>,
    bits: i32,
    group_size: i32,
    mode: &str,
) -> Result<()> {
    quantize_weights(weights, bits, group_size, mode)
}

/// Quantize weights with a recipe predicate, returning per-layer overrides.
/// Used by GGUF converter and convert_model when a recipe is specified.
pub(crate) fn quantize_weights_with_recipe_pub(
    weights: &mut HashMap<String, MxArray>,
    bits: i32,
    group_size: i32,
    mode: &str,
    predicate: &(dyn Fn(&str) -> QuantDecision + Send + Sync),
) -> Result<HashMap<String, serde_json::Value>> {
    quantize_weights_inner(weights, bits, group_size, mode, Some(predicate))
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

    // Step 1b: Dequantize pre-quantized vision weights (MXFP8/affine)
    // Some HuggingFace checkpoints ship vision_tower weights already quantized
    // (U32 packed + U8 scales). Dequantize them to bf16 since our vision encoder
    // uses standard Linear layers, not QuantizedLinear.
    {
        let quant_cfg = config
            .get("quantization")
            .or_else(|| config.get("quantization_config"));
        let quant_mode = quant_cfg
            .and_then(|q| q["mode"].as_str())
            .unwrap_or("affine");
        let quant_bits = quant_cfg.and_then(|q| q["bits"].as_i64()).unwrap_or(8) as i32;
        let quant_group_size = quant_cfg
            .and_then(|q| q["group_size"].as_i64())
            .unwrap_or(32) as i32;

        let vision_scale_keys: Vec<String> = new_weights
            .keys()
            .filter(|k| k.starts_with("vision_tower.") && k.ends_with(".scales"))
            .cloned()
            .collect();

        if !vision_scale_keys.is_empty() {
            info!(
                "  Dequantizing {} pre-quantized vision weights (mode={}, bits={}, group_size={})...",
                vision_scale_keys.len(),
                quant_mode,
                quant_bits,
                quant_group_size
            );

            let mode_cstr = std::ffi::CString::new(quant_mode).unwrap_or_else(|_| c"affine".into());

            for scale_key in &vision_scale_keys {
                let base = scale_key.strip_suffix(".scales").unwrap();
                let weight_key = format!("{}.weight", base);
                let biases_key = format!("{}.biases", base);

                let scales = new_weights.remove(scale_key);
                let weight = new_weights.remove(&weight_key);
                let biases = new_weights.remove(&biases_key);

                if let (Some(w), Some(s)) = (weight, scales) {
                    let biases_ptr = biases
                        .as_ref()
                        .map_or(std::ptr::null_mut(), |b| b.as_raw_ptr());
                    let handle = unsafe {
                        mlx_sys::mlx_dequantize(
                            w.as_raw_ptr(),
                            s.as_raw_ptr(),
                            biases_ptr,
                            quant_group_size,
                            quant_bits,
                            -1, // output dtype from scales
                            mode_cstr.as_ptr(),
                        )
                    };
                    if handle.is_null() {
                        warn!("  Failed to dequantize vision weight: {}", weight_key);
                        // Put originals back
                        new_weights.insert(weight_key, w);
                        new_weights.insert(scale_key.clone(), s);
                    } else {
                        let dequant = MxArray::from_handle(handle, "vision_dequant")?;
                        let dequant = dequant.astype(target_dtype)?;
                        dequant.eval();
                        new_weights.insert(weight_key, dequant);
                        info!("    Dequantized: {}", base);
                    }
                }
            }

            info!(
                "  After vision dequantization: {} tensors",
                new_weights.len()
            );
        }
    }

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
        // Non-FP8: convert non-quantized weights to target dtype.
        // Skip quantized tensor groups (.weight/.scales/.biases with U32/U8 dtypes).
        let quantized_bases: std::collections::HashSet<String> = new_weights
            .keys()
            .filter(|k| k.ends_with(".scales"))
            .map(|k| k.strip_suffix(".scales").unwrap().to_string())
            .collect();
        let keys: Vec<String> = new_weights.keys().cloned().collect();
        for k in keys {
            // Skip quantized tensors: packed weights (U32), scales (U8), biases
            if k.ends_with(".scales") || k.ends_with(".biases") {
                continue;
            }
            if k.ends_with(".weight") {
                let base = k.strip_suffix(".weight").unwrap();
                if quantized_bases.contains(base) {
                    continue; // packed quantized weight
                }
            }
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
    //
    // Detect if model is already in MLX format by checking norm weight values.
    // HF raw norm weights are ~0.0 (unshifted), MLX format is ~1.0 (shifted).
    let already_sanitized = {
        let test_key = new_weights
            .keys()
            .find(|k| k.ends_with(".input_layernorm.weight"))
            .cloned();
        if let Some(ref k) = test_key {
            let v = new_weights.get(k).unwrap();
            // Check first element value: ~0.0 = raw HF, ~1.0 = already shifted
            let f32_v = v.astype(DType::Float32)?;
            f32_v.eval();
            let val = f32_v.item_at_float32(0).unwrap_or(0.0);
            val > 0.5
        } else {
            false
        }
    };
    if already_sanitized {
        info!("  Model already sanitized (norms ~1.0), skipping norm shift + conv transpose");
    }

    let norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    ];
    let keys: Vec<String> = if already_sanitized {
        Vec::new() // skip all sanitization transforms
    } else {
        new_weights.keys().cloned().collect()
    };
    for k in keys {
        if k.contains("patch_embed.proj.weight") {
            // Conv3d/Conv2d: PyTorch [out, in, t, h, w] → MLX [out, t, h, w, in]
            // Skip if already in MLX format (last dim == in_channels, typically 3 for RGB)
            let v = new_weights.get(&k).unwrap();
            let ndim = v.ndim()? as usize;
            if ndim == 5 {
                let last_dim = v.shape_at(4)?;
                let dim1 = v.shape_at(1)?;
                if dim1 == 3 || dim1 == 1 {
                    // PyTorch format: [out, in_c, t, h, w] where in_c is small (3 for RGB)
                    let transposed = v.transpose(Some(&[0, 2, 3, 4, 1]))?;
                    new_weights.insert(k, transposed);
                } else if last_dim == 3 || last_dim == 1 {
                    // Already MLX format: [out, t, h, w, in_c] — skip
                } else {
                    // Ambiguous, assume PyTorch
                    let transposed = v.transpose(Some(&[0, 2, 3, 4, 1]))?;
                    new_weights.insert(k, transposed);
                }
            }
        } else if k.contains("conv1d.weight") {
            // Conv1d: PyTorch [out, in/g, k] → MLX [out, k, in/g]
            // For GatedDeltaNet conv1d, k=4 (linear_conv_kernel_dim)
            // Skip if already in MLX format (last dim == in_channels >> k)
            let v = new_weights.get(&k).unwrap();
            let ndim = v.ndim()? as usize;
            if ndim == 3 {
                let dim2 = v.shape_at(2)?;
                // In PyTorch format dim2 is kernel_size (typically 4)
                // In MLX format dim2 is in_channels (typically 128+)
                // If dim2 is small (≤16), it's likely kernel_size → needs transpose
                if dim2 <= 16 {
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

// ── AWQ Pre-Scaling ─────────────────────────────────────────────────────────

/// Apply AWQ-style pre-scaling using imatrix importance scores.
///
/// For each FFN scale group, amplifies important weight columns and fuses
/// the inverse into the preceding layer. This improves quantization quality
/// without changing model size or inference speed.
///
/// Scale groups per layer:
///   A: post_attention_layernorm → gate_proj, up_proj (input columns)
///   B: up_proj (output rows) → down_proj (input columns)
pub(crate) fn apply_awq_prescaling(
    weights: &mut HashMap<String, MxArray>,
    imatrix: &crate::utils::imatrix::ImatrixData,
    ratio: f32,
    num_layers: usize,
) -> Result<()> {
    info!(
        "Applying AWQ pre-scaling: {} layers, ratio={}, {} imatrix entries",
        num_layers,
        ratio,
        imatrix.importance.len()
    );

    let mut modified = 0usize;

    // Auto-detect key prefix: sanitized VLM models use "language_model.model.layers",
    // standard HF/GGUF models use "model.layers".
    let layer_prefix = if weights
        .keys()
        .any(|k| k.starts_with("language_model.model.layers."))
    {
        "language_model.model.layers"
    } else {
        "model.layers"
    };

    for i in 0..num_layers {
        let prefix = format!("{layer_prefix}.{i}");

        // ── Group A: norm → gate_proj + up_proj ──
        let gate_key = format!("{prefix}.mlp.gate_proj.weight");
        let up_key = format!("{prefix}.mlp.up_proj.weight");
        let norm_key = format!("{prefix}.post_attention_layernorm.weight");

        if let Some(scales) = compute_group_a_scales(imatrix, &gate_key, &up_key, ratio)? {
            // gate_proj.weight *= scales (broadcast over columns: [out, in] * [1, in])
            if let Some(gate) = weights.remove(&gate_key) {
                let scaled = scale_columns(&gate, &scales)?;
                weights.insert(gate_key, scaled);
                modified += 1;
            }
            // up_proj.weight *= scales (broadcast over columns)
            if let Some(up) = weights.remove(&up_key) {
                let scaled = scale_columns(&up, &scales)?;
                weights.insert(up_key.clone(), scaled);
                modified += 1;
            }
            // post_attention_layernorm.weight /= scales
            if let Some(norm) = weights.remove(&norm_key) {
                let inv = invert_scales(&scales)?.astype(norm.dtype()?)?;
                let scaled = norm.mul(&inv)?;
                weights.insert(norm_key, scaled);
                modified += 1;
            }
        }

        // ── Group B: up_proj (rows) → down_proj (columns) ──
        let down_key = format!("{prefix}.mlp.down_proj.weight");

        if let Some(scales) = compute_scales_for_key(imatrix, &down_key, ratio)? {
            // down_proj.weight *= scales (broadcast over columns: [out, in] * [1, in])
            if let Some(down) = weights.remove(&down_key) {
                let scaled = scale_columns(&down, &scales)?;
                weights.insert(down_key, scaled);
                modified += 1;
            }
            // up_proj.weight /= scales (broadcast over rows: [out, in] / [out, 1])
            if let Some(up) = weights.remove(&up_key) {
                let inv = invert_scales(&scales)?;
                let scaled = scale_rows(&up, &inv)?;
                weights.insert(up_key, scaled);
                modified += 1;
            }
        }
    }

    // Eval all modified weights to materialize
    for w in weights.values() {
        w.eval();
    }

    info!(
        "AWQ pre-scaling complete: modified {} weight tensors",
        modified
    );
    Ok(())
}

/// Compute AWQ scales for Group A (norm → gate_proj + up_proj).
/// Takes element-wise max of gate and up importance, then applies ratio.
fn compute_group_a_scales(
    imatrix: &crate::utils::imatrix::ImatrixData,
    gate_key: &str,
    up_key: &str,
    ratio: f32,
) -> Result<Option<MxArray>> {
    let gate_imp = imatrix.importance.get(gate_key);
    let up_imp = imatrix.importance.get(up_key);

    match (gate_imp, up_imp) {
        (Some(g), Some(u)) => {
            // Element-wise max of gate and up importance
            let combined: Vec<f32> = g.iter().zip(u.iter()).map(|(&a, &b)| a.max(b)).collect();
            let scales = compute_normalized_scales(&combined, ratio)?;
            Ok(Some(scales))
        }
        (Some(imp), None) | (None, Some(imp)) => {
            let scales = compute_normalized_scales(imp, ratio)?;
            Ok(Some(scales))
        }
        (None, None) => Ok(None),
    }
}

/// Compute AWQ scales for a single weight key.
fn compute_scales_for_key(
    imatrix: &crate::utils::imatrix::ImatrixData,
    key: &str,
    ratio: f32,
) -> Result<Option<MxArray>> {
    match imatrix.importance.get(key) {
        Some(imp) => {
            let scales = compute_normalized_scales(imp, ratio)?;
            Ok(Some(scales))
        }
        None => Ok(None),
    }
}

/// Compute normalized scales: scales = importance^ratio, then normalize by sqrt(max*min).
fn compute_normalized_scales(importance: &[f32], ratio: f32) -> Result<MxArray> {
    let mut scales: Vec<f32> = importance
        .iter()
        .map(|&x| x.max(1e-8).powf(ratio))
        .collect();

    // Normalize: scales / sqrt(max * min) to keep weights roughly same magnitude
    let max_s = scales.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_s = scales.iter().cloned().fold(f32::INFINITY, f32::min);
    let normalizer = (max_s * min_s).sqrt().max(1e-8);
    for s in &mut scales {
        *s /= normalizer;
    }

    let n = scales.len() as i64;
    MxArray::from_float32(&scales, &[n])
}

/// Cast scales to match weight dtype, then multiply columns: weight[:, j] *= scales[j]
fn scale_columns(weight: &MxArray, scales: &MxArray) -> Result<MxArray> {
    let s = scales.astype(weight.dtype()?)?;
    weight.mul(&s)
}

/// Cast scales to match weight dtype, then multiply rows: weight[i, :] *= scales[i]
fn scale_rows(weight: &MxArray, scales: &MxArray) -> Result<MxArray> {
    let n = scales.shape_at(0)?;
    let s = scales.astype(weight.dtype()?)?.reshape(&[n, 1])?;
    weight.mul(&s)
}

/// Compute 1/scales element-wise
fn invert_scales(scales: &MxArray) -> Result<MxArray> {
    let n = scales.shape_at(0)?;
    let ones_data: Vec<f32> = vec![1.0; n as usize];
    let ones = MxArray::from_float32(&ones_data, &[n])?;
    ones.div(scales)
}

/// Infer the number of model layers from weight keys.
/// Handles both `model.layers.N` and `language_model.model.layers.N` prefixes.
pub(crate) fn infer_num_layers_from_weights(weights: &HashMap<String, MxArray>) -> usize {
    let mut max_layer: Option<usize> = None;
    for key in weights.keys() {
        let rest = key
            .strip_prefix("language_model.model.layers.")
            .or_else(|| key.strip_prefix("model.layers."));
        if let Some(rest) = rest
            && let Some(dot_pos) = rest.find('.')
            && let Ok(n) = rest[..dot_pos].parse::<usize>()
        {
            max_layer = Some(max_layer.map_or(n, |m: usize| m.max(n)));
        }
    }
    max_layer.map_or(0, |m| m + 1)
}
