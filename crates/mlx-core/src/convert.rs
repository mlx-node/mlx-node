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
use crate::models::qianfan_ocr::persistence::load_qianfan_ocr_weights;
use crate::utils::safetensors::load_safetensors_lazy;

/// RAII guard that pins the MLX default device + stream to CPU for one
/// conversion call, then restores the previous values on drop.
///
/// Used by the conversion path to temporarily route every MLX op through
/// CPU for the duration of one `convert_model` /
/// `convert_gguf_to_safetensors` call. Both the default *device* and the
/// default *stream* must be switched: MLX dispatches stream-less ops via
/// `default_stream(default_device())`, so flipping the stream alone is
/// not enough — the device must be CPU too. On drop, the previous
/// device and stream are restored so subsequent inference / training
/// calls keep using the GPU. See the call sites for the rationale.
///
/// MUST be acquired while holding `CONVERT_MUTEX`'s lock — otherwise two
/// overlapping conversions can race on the process-wide MLX defaults and
/// restore each other's `saved_*` fields incorrectly (e.g. both observe
/// the already-flipped CPU device as "original", then both restore to
/// CPU, leaving the process pinned to CPU for the next inference call).
///
/// **Concurrent-inference limitation (intentional):** `convert_mutex`
/// only serializes convert-vs-convert. It does NOT block inference /
/// training entrypoints. If a Node process runs `convert_model` while
/// also serving inference, those inference ops resolve their stream via
/// `default_stream(default_device())` and will be silently routed to
/// CPU until the conversion finishes — typically minutes to hours on
/// large MoE checkpoints, with severe latency degradation. The
/// architecturally correct fix is to plumb explicit `Stream` arguments
/// through every convert-used MLX FFI op so the global default is never
/// touched; that's a substantial refactor outside the scope of this
/// change. For the supported usage today (the `mlx convert` CLI exits
/// after conversion; no other entrypoint in this codebase invokes
/// convert), this is a non-issue. Callers who embed convert inside a
/// long-lived multi-tenant Node process should serialize their own
/// inference against convert externally.
pub(crate) struct CpuConvertGuard {
    saved_device: i32,
    saved_stream: mlx_sys::mlx_stream,
}

impl CpuConvertGuard {
    /// Enter the CPU device + stream. The caller is responsible for holding
    /// `CONVERT_MUTEX` for the lifetime of the returned guard.
    pub(crate) fn enter_cpu() -> Self {
        let saved_device = unsafe { mlx_sys::mlx_default_device() };
        let saved_stream = unsafe { mlx_sys::mlx_default_stream(saved_device) };
        unsafe { mlx_sys::mlx_set_default_device(0) };
        let cpu_stream = unsafe { mlx_sys::mlx_default_stream(0) };
        unsafe { mlx_sys::mlx_set_default_stream(cpu_stream) };
        Self {
            saved_device,
            saved_stream,
        }
    }
}

impl Drop for CpuConvertGuard {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_set_default_stream(self.saved_stream) };
        unsafe { mlx_sys::mlx_set_default_device(self.saved_device) };
    }
}

/// Process-wide async mutex serializing all conversion calls.
///
/// `convert_model` and `convert_gguf_to_safetensors` mutate MLX's
/// process-wide default device + default stream via `CpuConvertGuard`,
/// which is unsafe under concurrency: two overlapping conversions (or a
/// convert during inference that depends on the GPU default) can race on
/// the global state. Both NAPI entrypoints `.await` this mutex before
/// constructing a `CpuConvertGuard`, so only one conversion runs at a
/// time across the entire Node process.
pub(crate) fn convert_mutex() -> &'static tokio::sync::Mutex<()> {
    static CONVERT_MUTEX: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    CONVERT_MUTEX.get_or_init(|| tokio::sync::Mutex::new(()))
}

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

    /// Quantization mode: "affine" (default), "mxfp4", "mxfp8", or "nvfp4"
    pub quant_mode: Option<String>,

    /// Quantization recipe for per-layer mixed-bit quantization.
    /// Options: mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6, qwen3_5
    pub quant_recipe: Option<String>,

    /// Path to an imatrix GGUF file for AWQ-style pre-scaling.
    /// Improves quantization quality by amplifying important weight channels.
    pub imatrix_path: Option<String>,

    /// Upgrade quantization to micro-scaling FP (mxfp4 / mxfp8).
    /// When true, applies after the recipe predicate: any 8-bit affine decision
    /// becomes mxfp8, any 4-bit decision becomes mxfp4. Requires `quant_mode = "affine"`.
    /// Forces `group_size = 32` for upgraded layers.
    pub quant_mxfp: Option<bool>,
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
    let _convert_start = std::time::Instant::now();
    info!(
        target = "mlx_core::convert",
        input_dir = %options.input_dir,
        output_dir = %options.output_dir,
        dtype = ?options.dtype,
        model_type = ?options.model_type,
        quantize = options.quantize.unwrap_or(false),
        quant_mode = ?options.quant_mode,
        quant_recipe = ?options.quant_recipe,
        "convert_model start"
    );
    let result = convert_model_inner(options).await;
    match &result {
        Ok(r) => info!(
            target = "mlx_core::convert",
            total_seconds = _convert_start.elapsed().as_secs_f64(),
            num_tensors = r.num_tensors,
            num_parameters = r.num_parameters,
            output_path = %r.output_path,
            "convert_model finished"
        ),
        Err(e) => tracing::error!(
            target = "mlx_core::convert",
            total_seconds = _convert_start.elapsed().as_secs_f64(),
            error = %e,
            "convert_model failed"
        ),
    }
    result
}

async fn convert_model_inner(options: ConversionOptions) -> Result<ConversionResult> {
    let input_dir = PathBuf::from(&options.input_dir);
    let output_dir = PathBuf::from(&options.output_dir);
    let target_dtype = options.dtype.unwrap_or_else(|| "float32".to_string());
    let verbose = options.verbose.unwrap_or(false);
    let model_type = options.model_type;
    let do_quantize = options.quantize.unwrap_or(false);
    let quant_mode = options.quant_mode.unwrap_or_else(|| "affine".to_string());
    let quant_recipe = options.quant_recipe;
    let imatrix_path = options.imatrix_path;
    let quant_mxfp = options.quant_mxfp.unwrap_or(false);

    // Validate quant_mode before it reaches FFI
    const VALID_QUANT_MODES: &[&str] = &["affine", "mxfp4", "mxfp8", "nvfp4"];
    if do_quantize && !VALID_QUANT_MODES.contains(&quant_mode.as_str()) {
        return Err(Error::from_reason(format!(
            "Invalid quant_mode '{}': must be one of {}",
            quant_mode,
            VALID_QUANT_MODES.join(", ")
        )));
    }

    // Per-mode defaults — match MLX C++ kernel instantiations in
    // mlx/backend/metal/kernels/fp_quantized.metal.
    let (default_bits, default_group_size) = match quant_mode.as_str() {
        "affine" => (4, 64),
        "mxfp4" => (4, 32),
        "mxfp8" => (8, 32),
        "nvfp4" => (4, 16),
        // Unreachable: gated by VALID_QUANT_MODES check above when do_quantize.
        // When !do_quantize, these defaults are unused.
        _ => (4, 64),
    };

    let quant_bits = options.quant_bits.unwrap_or(default_bits);
    let quant_group_size = options.quant_group_size.unwrap_or(default_group_size);

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

    // MXFP modes have strict bits/group_size invariants enforced by the MLX
    // backend. Surface the failure here (with a clear message) rather than
    // letting it bubble up as a confusing FFI error mid-conversion.
    if do_quantize && quant_mode == "mxfp4" && (quant_bits != 4 || quant_group_size != 32) {
        return Err(Error::from_reason(format!(
            "mxfp4 requires bits=4 and group_size=32 (got bits={quant_bits}, group_size={quant_group_size})"
        )));
    }
    if do_quantize && quant_mode == "mxfp8" && (quant_bits != 8 || quant_group_size != 32) {
        return Err(Error::from_reason(format!(
            "mxfp8 requires bits=8 and group_size=32 (got bits={quant_bits}, group_size={quant_group_size})"
        )));
    }
    if do_quantize && quant_mode == "nvfp4" {
        validate_nvfp4_invariants(quant_bits, quant_group_size).map_err(Error::from_reason)?;
    }

    // Validate --q-mxfp orthogonality: requires affine baseline (it then upgrades
    // per-layer affine decisions to mxfp4/mxfp8).
    if quant_mxfp && !do_quantize {
        return Err(Error::from_reason(
            "--q-mxfp requires --quantize to be enabled".to_string(),
        ));
    }
    if quant_mxfp && quant_mode != "affine" {
        return Err(Error::from_reason(format!(
            "--q-mxfp requires --q-mode affine (default), got '{}'. \
             --q-mxfp orthogonally upgrades affine decisions to mxfp4/mxfp8.",
            quant_mode
        )));
    }

    // LFM2 mxfp/nvfp now SUPPORTED for non-MoE linears (fast-follow #1a): the
    // lfm2 loader's attention / conv / dense-MLP projections are mode-aware
    // `LinearProj`/`MLPVariant` backed by `QuantizedLinear`, which threads the
    // resolved mode (affine / mxfp4 / mxfp8 / nvfp4) into `mlx_quantized_matmul`
    // at forward time. The MoE experts/gate already supported all four modes.
    // The EMBEDDING and lm_head remain excluded from quantization (vocab-dim
    // tensors): `should_quantize` skips `embed_tokens`/`lm_head`, so an
    // mxfp8/mxfp4/nvfp4 lfm2 checkpoint ships quantized experts + attn/conv/
    // dense-MLP and a plain bf16 embedding — which the #1a loader can load. A
    // quant-capable embedding lands in #1b; the prior affine-only gate is thus
    // removed.

    // Validate recipe
    if let Some(ref recipe) = quant_recipe {
        if !do_quantize {
            return Err(Error::from_reason(
                "--q-recipe requires --quantize to be enabled".to_string(),
            ));
        }
        if quant_mode != "affine" && quant_mode != "nvfp4" {
            return Err(Error::from_reason(format!(
                "--q-recipe is compatible with --q-mode affine or nvfp4 only; for mxfp4/mxfp8 use --q-mxfp instead. Got '{}'.",
                quant_mode
            )));
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
        // Restrict --q-mode nvfp4 + --q-recipe to recipes that have model-aware
        // tensor-class exclusions for NVFP4-sensitive layers. See
        // [`validate_nvfp4_recipe`] for the full rationale.
        if quant_mode == "nvfp4" {
            validate_nvfp4_recipe(recipe).map_err(Error::from_reason)?;
        }
        // Unsloth recipe requires imatrix for near-lossless attention/SSM quantization
        if recipe == "unsloth" && imatrix_path.is_none() {
            return Err(Error::from_reason(
                "unsloth recipe requires --imatrix-path: imatrix calibration data is needed \
                 for near-lossless quantization of attention/SSM layers"
                    .to_string(),
            ));
        }
    }

    // Validate input directory
    if !input_dir.exists() {
        return Err(Error::from_reason(format!(
            "Input directory does not exist: {}",
            input_dir.display()
        )));
    }

    // Serialize all conversions process-wide before touching MLX's default
    // device + stream — see `convert_mutex` and `CpuConvertGuard` docs for
    // the race this avoids.
    let _convert_lock = convert_mutex().lock().await;

    // Route every MLX op in this conversion through the CPU device + stream.
    //
    // The conversion path is slice / reshape / dtype-cast only — no real math.
    // On GPU, materializing a 1.6 GB sliced view of a fused expert tensor backed
    // by a 250 GB mmap'd source can stall a Metal command buffer past the macOS
    // GPU watchdog (~5 s), surfacing as
    // `kIOGPUCommandBufferCallbackErrorTimeout` mid-shard for large MoE models
    // (e.g. Qwen3.5 122B-A10B with 256 experts × 48 layers). CPU has direct
    // access to the mmap'd pages and is immune to the watchdog. `_stream_guard`
    // restores the prior default device + stream when convert_model returns.
    let _stream_guard = CpuConvertGuard::enter_cpu();

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
    let has_custom_sanitizer = matches!(
        model_type.as_deref(),
        Some("qwen3_5_moe" | "qwen3_5" | "lfm2_moe" | "lfm2")
    );

    // True for models whose sanitizer arm manages quantization itself — the
    // generic quantize block below must skip these to avoid double-quantizing.
    let is_privacy_filter = matches!(model_type.as_deref(), Some("privacy-filter"));

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
        Some("lfm2_moe" | "lfm2") => {
            info!(
                "Applying LFM2 weight sanitization (MLP rename, conv transpose, expert stacking)..."
            );
            sanitize_lfm2_moe(
                converted_tensors,
                &config,
                &target_dtype,
                tie_word_embeddings,
            )?
        }
        Some("qianfan-ocr") => {
            info!(
                "Applying Qianfan-OCR weight sanitization (key renaming, conv2d transposition)..."
            );
            load_qianfan_ocr_weights(converted_tensors)?
        }
        Some("gemma4") => {
            info!(
                "Applying Gemma4 weight sanitization (prefix stripping, vision/audio removal)..."
            );
            sanitize_gemma4_convert(converted_tensors, tie_word_embeddings, verbose)?
        }
        Some("privacy-filter") => {
            // openai/privacy-filter ships with MLX-loadable safetensors already.
            // No tensor renaming, no FP8 dequant, no expert stacking — the
            // generic dtype pass above is the only transformation needed.
            info!("Privacy-filter model: identity pass (no sanitization required).");
            converted_tensors
        }
        // NOTE: privacy-filter quantization is handled below in the dedicated
        // sanitizer-managed quantize block (gated by `is_privacy_filter`),
        // because it needs access to the bits/group_size/mode from the outer
        // scope and we want to suppress the generic quantize pass for it.
        Some(other) => {
            return Err(Error::from_reason(format!(
                "Unknown model type: '{}'. Supported: paddleocr-vl, qwen3_5_moe, qwen3_5, lfm2_moe, lfm2, qianfan-ocr, gemma4, privacy-filter",
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
    // Effective mode/group_size recorded in config.json. The no-recipe path
    // updates these when --q-mxfp upgrades the global mode to mxfp4/mxfp8 so
    // downstream loaders dispatch to the correct builder.
    let mut quant_mode_effective = quant_mode.clone();
    let mut quant_group_size_effective = quant_group_size;
    // lfm2/lfm2_moe opt INTO quantizing the token embedding: their
    // `nn::Embedding` installs a PACKED-quantized backend (gather-dequant
    // lookup + quantized tied-head matmul), so the embedding table can be
    // quantized for real memory savings. Every other family keeps the embedding
    // bf16 (unchanged). A TIED `lm_head` is dropped at sanitize, so this never
    // quantizes an output head.
    let embed_quantizable = matches!(model_type.as_deref(), Some("lfm2") | Some("lfm2_moe"));
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

        if is_privacy_filter {
            // Privacy-filter has a dedicated predicate: quantize attention
            // projections (q/k/v/o) and MoE experts (gate_up_proj, down_proj);
            // quantize routers at 8-bit affine when --q-mode affine; leave
            // embeddings, classifier head, norms, biases, and attention sinks
            // at bf16. Inference path is bf16-only until Phase C lands.
            let preserved_extra = if quant_mode == "affine" {
                "8-bit-affine routers"
            } else {
                "bf16 routers"
            };
            info!(
                "Quantizing privacy-filter (mode={}, bits={}, group_size={}) — projections + \
                 MoE experts only; embeddings, classifier head, norms, biases, sinks preserved \
                 ({}).",
                quant_mode, quant_bits, quant_group_size, preserved_extra
            );
            let predicate =
                build_privacy_filter_predicate(quant_bits, quant_group_size, &quant_mode);
            // Discard any per-tensor overrides emitted by the inner quantizer
            // (it only records when bits/group_size/mode differ from defaults,
            // which is too sparse for our needs); we re-derive a complete
            // override map below from the resulting `.scales` keys.
            let _custom_overrides = quantize_weights_with_recipe_pub(
                &mut converted_tensors,
                quant_bits,
                quant_group_size,
                &quant_mode,
                &*predicate,
                embed_quantizable,
            )?;

            // Build per-layer overrides for ALL quantized tensors so that the
            // downstream loader can discover which tensors are quantized and
            // with which parameters. Unlike Qwen3.5, we want every quantized
            // tensor recorded — not only non-default ones.
            for key in converted_tensors.keys() {
                let Some(prefix) = key.strip_suffix(".scales") else {
                    continue;
                };
                let (bits, group_size, mode) = if key.contains(".mlp.router.") {
                    // Routers are only quantized in affine mode (8-bit, group=quant_group_size)
                    (8, quant_group_size, "affine".to_string())
                } else {
                    (quant_bits, quant_group_size, quant_mode.clone())
                };
                per_layer_overrides.insert(
                    prefix.to_string(),
                    serde_json::json!({
                        "bits": bits,
                        "group_size": group_size,
                        "mode": mode,
                    }),
                );
            }
        } else if let Some(ref recipe) = quant_recipe {
            let weight_keys: Vec<String> = converted_tensors.keys().cloned().collect();
            // Recipes emit affine `Custom` decisions for protected tensors
            // (lm_head, AWQ-corrected attn/SSM projections, etc). Affine
            // quantize only supports group_size ∈ {32, 64, 128}, so when the
            // global mode is nvfp4 (which forces quant_group_size=16) we must
            // pass a recipe-affine-appropriate group_size to the predicate
            // builder. apply_nvfp4_upgrade still sets gs=16 on the 4-bit
            // decisions it promotes, and the top-level config.json still
            // records gs=16/mode=nvfp4 for the default dequantizer.
            let recipe_gs = if quant_mode == "nvfp4" {
                64
            } else {
                quant_group_size
            };
            let predicate = build_predicate_for_recipe(recipe, &weight_keys, quant_bits, recipe_gs)
                .map_err(Error::from_reason)?;
            let predicate: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> = if quant_mxfp {
                apply_mxfp_upgrade(predicate, quant_bits)
            } else if quant_mode == "nvfp4" {
                // Recipe + --q-mode nvfp4: promote 4-bit recipe decisions to
                // NVFP4 (group_size=16). Mutually exclusive with quant_mxfp,
                // since --q-mxfp requires --q-mode affine.
                apply_nvfp4_upgrade(predicate)
            } else {
                predicate
            };
            per_layer_overrides = quantize_weights_with_recipe_pub(
                &mut converted_tensors,
                quant_bits,
                quant_group_size,
                &quant_mode,
                &*predicate,
                embed_quantizable,
            )?;
        } else {
            // No recipe: --q-mxfp overrides the global mode + group_size so the
            // legacy quantize path emits mxfp4/mxfp8 weights. The legacy path
            // STILL emits per-layer overrides for special keys (router-gate
            // upgrades, lm_head/router.proj affine downgrades, embed_tokens
            // affine downgrades), and those overrides MUST be persisted to
            // config.json so the loader dispatches to the correct builder.
            let (effective_mode, effective_gs) = if quant_mxfp {
                match quant_bits {
                    8 => ("mxfp8".to_string(), 32),
                    4 => ("mxfp4".to_string(), 32),
                    _ => {
                        return Err(Error::from_reason(format!(
                            "--q-mxfp without a recipe requires --q-bits 4 or 8 (got {})",
                            quant_bits
                        )));
                    }
                }
            } else {
                (quant_mode.clone(), quant_group_size)
            };
            per_layer_overrides = quantize_weights(
                &mut converted_tensors,
                quant_bits,
                effective_gs,
                &effective_mode,
                embed_quantizable,
            )?;
            if !per_layer_overrides.is_empty() {
                info!(
                    "No-recipe quantization emitted {} per-layer overrides (router-gate / affine-only keys): {:?}",
                    per_layer_overrides.len(),
                    per_layer_overrides.keys().collect::<Vec<_>>()
                );
            }
            quant_mode_effective = effective_mode;
            quant_group_size_effective = effective_gs;
        }
    }

    // Update tensor names after sanitization/quantization
    let mut tensor_names: Vec<String> = converted_tensors.keys().cloned().collect();
    tensor_names.sort();

    // Save converted model — sharded output with index file (mlx-lm/mlx-vlm compatible)
    info!(
        target = "mlx_core::convert",
        output_dir = %output_dir.display(),
        num_tensors = converted_tensors.len(),
        "starting sharded save"
    );

    let save_start = std::time::Instant::now();
    crate::utils::safetensors::save_safetensors_sharded(&output_dir, &mut converted_tensors)?;
    info!(
        target = "mlx_core::convert",
        save_seconds = save_start.elapsed().as_secs_f64(),
        "sharded save complete"
    );

    // Write config.json — clean and sort keys to match mlx-lm/mlx-vlm save_config
    let output_config_path = output_dir.join("config.json");
    let mut output_config = config.clone();

    // Inject quantization metadata if quantized
    if do_quantize {
        let mut quant_obj = serde_json::json!({
            "group_size": quant_group_size_effective,
            "bits": quant_bits,
            "mode": quant_mode_effective,
        });
        if let Some(obj) = quant_obj.as_object_mut() {
            for (path, override_val) in &per_layer_overrides {
                // Privacy-filter uses bare `model.*` keys natively; other models
                // need the `language_model.model.*` prefix expected by mlx-lm.
                let key = if is_privacy_filter {
                    path.clone()
                } else {
                    crate::utils::normalize_override_key(path)
                };
                obj.insert(key, override_val.clone());
            }
        }
        output_config["quantization"] = quant_obj.clone();
        output_config["quantization_config"] = quant_obj;
    }

    // Clean config: remove keys that mlx-lm/mlx-vlm strip
    if let Some(obj) = output_config.as_object_mut() {
        obj.remove("_name_or_path");
    }

    // Sort config keys for readability (matches mlx-lm/mlx-vlm save_config)
    if let Some(obj) = output_config.as_object() {
        let sorted: serde_json::Map<String, serde_json::Value> =
            obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        // BTreeMap for sorted output
        let sorted: std::collections::BTreeMap<String, serde_json::Value> =
            sorted.into_iter().collect();
        let config_str = serde_json::to_string_pretty(&sorted)
            .map_err(|e| Error::from_reason(format!("Failed to serialize config: {}", e)))?;
        fs::write(&output_config_path, config_str)
            .map_err(|e| Error::from_reason(format!("Failed to write config.json: {}", e)))?;
    }
    info!("Wrote config.json");

    // Copy tokenizer, model config, and Python model definition files
    let config_files = [
        // Tokenizer files
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        // Chat template (Gemma4 and other models use external .jinja files)
        "chat_template.jinja",
        // Generation config
        "generation_config.json",
        // VLM-specific files
        "preprocessor_config.json",
        "processor_config.json",
        "viterbi_calibration.json",
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

    // Copy *.py model definition files (mlx-lm/mlx-vlm load model classes from these)
    if let Ok(entries) = fs::read_dir(&input_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("py") {
                let dst = output_dir.join(entry.file_name());
                fs::copy(&path, &dst).map_err(|e| {
                    Error::from_reason(format!(
                        "Failed to copy {}: {}",
                        entry.file_name().to_string_lossy(),
                        e
                    ))
                })?;
                if verbose {
                    info!("Copying {}", entry.file_name().to_string_lossy());
                }
            }
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
///
/// `embed_quantizable` opts the model family INTO quantizing the token
/// embedding (`embed_tokens` / `embedding.`). This is `true` ONLY for
/// lfm2/lfm2_moe, whose `nn::Embedding` now installs a PACKED-quantized backend
/// (`load_quantized_packed`) that gather-then-dequantizes on lookup and runs a
/// quantized matmul for the tied head — so the embedding can be quantized for
/// real memory savings. For every other family (qwen3_5, gemma4, …) it is
/// `false` and the embedding is skipped, preserving the prior behavior. A TIED
/// `lm_head` is always excluded (it is dropped at sanitize time) — the
/// `lm_head` skip below is unconditional.
fn should_quantize(key: &str, embed_quantizable: bool) -> bool {
    // Only .weight keys (not .scales, .biases, etc.)
    if !key.ends_with(".weight") {
        return false;
    }

    // Exclude vision encoder weights (keep bf16 for quality)
    if key.contains("vision_tower") || key.contains("visual.") {
        return false;
    }

    // lm_head (output head) is ALWAYS excluded — for tied models it is dropped
    // at sanitize, for untied models it shares the vocab dimension and loads
    // through an affine-only head loader.
    if key.contains("lm_head") {
        return false;
    }

    // Token embeddings: excluded by default (vocab-dim tensor). lfm2/lfm2_moe
    // opt in via `embed_quantizable` — their packed embedding backend handles
    // a quantized table (gather-dequant lookup + quantized tied-head matmul).
    if !embed_quantizable && (key.contains("embed_tokens") || key.contains("embedding.")) {
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

    // Exclude LFM2's depthwise short conv (`*.conv.conv.weight`, shape
    // [channels, kernel, 1] after the sanitizer transpose). The lfm2 loader
    // NEVER quantizes this tensor — it routes it through the dedicated bf16
    // `set_conv_weight` setter. Belt-and-braces over the `last_dim % group_size`
    // guard below (last dim is 1, which already fails divisibility), so a future
    // group_size of 1 can never sneak it into the affine quantizer.
    if key.ends_with("conv.conv.weight") {
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
    // Router gates: select top-K experts per token. FP4 / low-bit affine
    // quantization noise flips routing decisions and destroys generation
    // quality — these must stay 8-bit affine in every recipe.
    //
    // Naming differs across model families:
    // - Qwen3.x MoE: `.mlp.gate.weight`, `.mlp.shared_expert_gate.weight`
    // - Gemma4 MoE: `.mlp.router.proj.weight` (also affine-only at load)
    // - LFM2 MoE: `.feed_forward.gate.weight` — the per-layer router that
    //   selects top-K experts. The lfm2 loader resolves its quant params via a
    //   direct `feed_forward.gate` override lookup (`build_lfm2_gate_ql`), so it
    //   must receive the explicit 8-bit affine override rather than the low-bit
    //   default; otherwise routing noise destroys generation quality.
    //
    // Note: `router.proj` is *also* listed in `is_affine_only_key` so the
    // load path refuses non-affine modes. Matching it here as well ensures
    // recipes emit an explicit 8-bit Custom override (forcing gs=64),
    // rather than `Default` which would inherit whatever low-bit base the
    // user picked.
    let stripped = key.strip_suffix(".weight").unwrap_or(key);
    stripped.ends_with(".mlp.gate")
        || stripped.ends_with(".shared_expert_gate")
        || stripped.ends_with(".router.proj")
        || stripped.ends_with(".feed_forward.gate")
}

/// Check if a key is loaded through an affine-only dequantization path and
/// must therefore be preserved as affine (never promoted to mxfp4/mxfp8/nvfp4).
///
/// These keys load through affine-only `Linear::load_quantized` /
/// `Embedding::load_quantized` helpers:
/// - `lm_head`: dense Qwen3.5's lm_head loader hardcodes affine dequant.
/// - `router.proj`: Gemma4's MoE router uses affine-only `Linear`.
/// - `embed_tokens` (and `embed_tokens_per_layer`): Gemma4 / others route
///   quantized embeddings through `Embedding::load_quantized`.
/// - `embedding_projection`: Gemma4's `embed_vision.embedding_projection`
///   loads through affine-only `Linear::load_quantized`.
///
/// Emitting MXFP / NVFP weights at these keys would be silently
/// mis-dequantized at load time. Used by `apply_mxfp_upgrade` /
/// `apply_nvfp4_upgrade` to skip the upgrade and by `quantize_weights_inner`
/// to force an affine 8-bit override on the no-recipe path.
fn is_affine_only_key(key: &str) -> bool {
    key.contains("lm_head")
        || key.contains("router.proj")
        || key.contains("embed_tokens")
        || key.contains("embedding_projection")
}

// ── Per-Layer Quantization Recipes ──────────────────────────────────────────

/// Per-weight quantization decision returned by recipe predicates.
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
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
/// Sanitize Gemma4 weights at conversion time.
///
/// Produces output compatible with mlx-lm, mlx-vlm, AND our Rust inference.
/// Matches mlx-lm's gemma4.Model.sanitize() + gemma4_text.Model.sanitize():
///
/// 1. Strip HF prefix, remap to mlx-lm attribute tree
/// 2. Preserve ALL weights (vision/audio/multimodal) — mlx-vlm needs them,
///    and mlx-lm's sanitize() safely ignores them on load
/// 3. Remove rotary_emb, calibration tensors
/// 4. Split fused experts.gate_up_proj into switch_glu.gate_proj + switch_glu.up_proj
/// 5. Rename experts.down_proj to switch_glu.down_proj.weight
/// 6. Drop lm_head.weight when tie_word_embeddings=true
fn sanitize_gemma4_convert(
    weights: HashMap<String, MxArray>,
    tie_word_embeddings: bool,
    verbose: bool,
) -> Result<HashMap<String, MxArray>> {
    let mut sanitized: HashMap<String, MxArray> = HashMap::new();
    let mut skipped = 0usize;

    for (key, array) in weights {
        // Step 1: Strip HF prefix to get the bare key.
        // HF stores as: model.language_model.model.layers.N.* or model.layers.N.*
        let stripped = key
            .strip_prefix("model.language_model.model.")
            .or_else(|| key.strip_prefix("model.language_model."))
            .or_else(|| key.strip_prefix("language_model.model."))
            .or_else(|| key.strip_prefix("language_model."))
            .or_else(|| key.strip_prefix("model."))
            .unwrap_or(&key);

        // Skip rotary_emb keys (precomputed inverse frequencies, unused)
        if stripped.contains("rotary_emb") {
            skipped += 1;
            continue;
        }

        // Skip calibration tensors for language model weights only.
        // Keep them for vision_tower/audio_tower — mlx-vlm's ClippableLinear needs them.
        if stripped.ends_with(".input_max")
            || stripped.ends_with(".input_min")
            || stripped.ends_with(".output_max")
            || stripped.ends_with(".output_min")
        {
            let is_multimodal = stripped.starts_with("vision_tower.")
                || stripped.starts_with("vision_encoder.")
                || stripped.starts_with("audio_tower.")
                || stripped.starts_with("audio_encoder.");
            if !is_multimodal {
                skipped += 1;
                continue;
            }
        }

        // Skip lm_head.weight when tied embeddings
        if tie_word_embeddings && stripped == "lm_head.weight" {
            skipped += 1;
            continue;
        }

        // Multimodal weights: keep with their original (stripped) key prefix.
        // mlx-vlm expects these for vision/audio processing.
        // mlx-lm's sanitize() skips them harmlessly on load.
        if stripped.starts_with("vision_tower.")
            || stripped.starts_with("vision_encoder.")
            || stripped.starts_with("audio_tower.")
            || stripped.starts_with("audio_encoder.")
            || stripped.starts_with("embed_audio.")
            || stripped.starts_with("embed_vision.")
            || stripped.starts_with("multi_modal_projector.")
        {
            // Apply PyTorch→MLX layout conversions for conv weights
            // (matches mlx-vlm's sanitize transforms)
            let ndim = array.ndim()?;
            let array = if stripped.contains("depthwise_conv1d.weight") && ndim == 3 {
                // Conv1d: PyTorch [out, in, kW] → MLX [out, kW, in]
                let transposed = array.transpose(Some(&[0, 2, 1]))?;
                transposed.eval();
                transposed
            } else if stripped.contains("subsample_conv_projection")
                && stripped.contains("conv.weight")
                && ndim == 4
            {
                // Conv2d: PyTorch [out, in, kH, kW] → MLX [out, kH, kW, in]
                let transposed = array.transpose(Some(&[0, 2, 3, 1]))?;
                transposed.eval();
                transposed
            } else {
                array
            };
            sanitized.insert(stripped.to_string(), array);
            continue;
        }

        // Step 2: Apply mlx-lm gemma4_text sanitize transforms.
        // Split fused experts.gate_up_proj and rename experts.down_proj.
        if stripped.ends_with(".experts.gate_up_proj") {
            // Split [num_experts, 2*moe_inter, hidden] along axis -2 into two halves
            let base = stripped.strip_suffix(".gate_up_proj").unwrap();
            let shape = array.shape()?;
            let mid = shape[1] / 2; // split the output dimension in half

            let gate = array.slice_axis(1, 0, mid)?;
            let up = array.slice_axis(1, mid, shape[1])?;

            // Ensure contiguous layout for safetensors (matches Python's mx.contiguous)
            gate.eval();
            up.eval();

            let gate_key = format!("language_model.model.{base}.switch_glu.gate_proj.weight");
            let up_key = format!("language_model.model.{base}.switch_glu.up_proj.weight");
            sanitized.insert(gate_key, gate);
            sanitized.insert(up_key, up);
            continue;
        }

        if stripped.ends_with(".experts.down_proj") {
            let base = stripped.strip_suffix(".down_proj").unwrap();
            let out_key = format!("language_model.model.{base}.switch_glu.down_proj.weight");
            sanitized.insert(out_key, array);
            continue;
        }

        // Step 3: Add the mlx-lm attribute tree prefix.
        // mlx-lm's gemma4.Model has: self.language_model = gemma4_text.Model
        // gemma4_text.Model has: self.model = Gemma4TextModel
        // So all text weights get prefix: language_model.model.
        let out_key = format!("language_model.model.{stripped}");
        sanitized.insert(out_key, array);
    }

    if verbose || skipped > 0 {
        info!(
            "  Gemma4 sanitize: kept {} tensors, skipped {}",
            sanitized.len(),
            skipped
        );
    }

    Ok(sanitized)
}

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
        if !should_quantize(key, /* embed_quantizable */ false) {
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
        if !should_quantize(key, /* embed_quantizable */ false) {
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
/// MLX affine equivalent of Unsloth Dynamic 2.0 (UD) GGUF quantization.
/// Based on Unsloth's per-tensor 99.9% KLD analysis for Qwen3.5's hybrid
/// GatedDeltaNet (linear attention/SSM) + full attention architecture:
/// (https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)
///
/// ## GGUF Equivalence
///
/// | `--q-bits` | GGUF Equivalent    | Size (35B-A3B) |
/// |------------|--------------------|----------------|
/// | 3          | `UD-Q3_K_XL` 16 GB | ~17 GB         |
/// | 4          | `UD-Q4_K_XL` 19 GB | ~20 GB         |
///
/// Size difference (~1 GB) is format-level: GGUF K-quants pack scales within
/// blocks, while MLX affine stores separate scales+biases per group (~2 GB
/// metadata overhead).
///
/// ## Per-Tensor Bit Assignments (default_bits = N)
///
/// | Weight                  | Bits    | GGUF Type  | Rationale                         |
/// |-------------------------|---------|------------|-----------------------------------|
/// | `embed_tokens`          | N+2     | Q5_K/Q6_K  | KLD ~0.15 — very low sensitivity  |
/// | `lm_head`               | N+3     | Q6_K/Q8_0  | KLD ~0.05 — safest tensor         |
/// | `self_attn.q/k/v_proj`  | N+2     | Q5_K/Q6_K  | KLD ~1.5-2.9, AWQ via layernorm   |
/// | `linear_attn.in_proj_*` | N+2     | Q5_K/Q6_K  | KLD ~2.9, AWQ via layernorm       |
/// | `self_attn.o_proj`      | bf16    | bf16       | KLD ~1.5, NOT AWQ-correctable     |
/// | `linear_attn.out_proj`  | bf16    | bf16       | KLD ~6.0, worst tensor by far     |
/// | `down_proj`             | N+1     | Q4_K/Q5_K  | "slightly more sensitive" than FFN |
/// | `gate_proj`, `up_proj`  | N       | Q3_K/Q4_K  | "generally ok" at low bits        |
/// | Router gates            | 8       | Q8_0       | Standard for MoE routing          |
/// | GDN params (A_log, etc) | bf16    | bf16       | Excluded by `should_quantize()`   |
///
/// ## AWQ Pre-Scaling
///
/// imatrix is **required** — attention/SSM weights fed by input_layernorm can
/// be AWQ-corrected (layernorm absorbs inverse scales), but o_proj and out_proj
/// have no preceding norm and must stay bf16.
pub(crate) fn build_unsloth_recipe(
    default_bits: i32,
    default_group_size: i32,
) -> Box<dyn Fn(&str) -> QuantDecision + Send + Sync> {
    // MLX quantize supports: 2, 3, 4, 5, 6, 8 (no 7)
    let snap_bits = |b: i32| -> i32 {
        match b {
            b if b <= 2 => 2,
            3 => 3,
            4 => 4,
            5 => 5,
            6 => 6,
            7 => 8, // 7 not supported, snap up to 8
            _ => 8,
        }
    };
    let down_proj_bits = snap_bits(default_bits + 1);
    let embed_bits = snap_bits(default_bits + 2);
    let lm_head_bits = snap_bits(default_bits + 3);
    let attn_ssm_bits = snap_bits(default_bits + 2);
    let gs = default_group_size;

    Box::new(move |key: &str| -> QuantDecision {
        // Handle embed_tokens and lm_head BEFORE should_quantize (which skips them).
        // These are among the least sensitive tensors per Unsloth's KLD analysis:
        // token_embedding KLD ~0.15, output KLD ~0.05 at q5_k
        if key.contains("embed_tokens") && key.ends_with(".weight") {
            return QuantDecision::Custom {
                bits: embed_bits,
                group_size: gs,
                mode: "affine".to_string(),
            };
        }
        if key.contains("lm_head") && key.ends_with(".weight") {
            return QuantDecision::Custom {
                bits: lm_head_bits,
                group_size: gs,
                mode: "affine".to_string(),
            };
        }

        if !should_quantize(key, /* embed_quantizable */ false) {
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

        // Attention/SSM projections WITH AWQ pre-scaling (Groups C & D):
        // input_layernorm absorbs inverse scales for these.
        let is_awq_corrected_attn = key.contains("self_attn.q_proj")
            || key.contains("self_attn.k_proj")
            || key.contains("self_attn.v_proj")
            || key.contains("linear_attn.in_proj_qkv")
            || key.contains("linear_attn.in_proj_z");

        if is_awq_corrected_attn {
            return QuantDecision::Custom {
                bits: attn_ssm_bits,
                group_size: gs,
                mode: "affine".to_string(),
            };
        }

        // Attention/SSM projections WITHOUT AWQ pre-scaling:
        // o_proj input comes from attention computation (not a norm layer),
        // out_proj input comes from GDN computation.
        // These cannot be AWQ-corrected — keep at bf16 for quality.
        // linear_attn.out_proj: KLD ~6.0 — worst tensor by far.
        // self_attn.o_proj: KLD ~1.5 — sensitive but not catastrophic.
        let is_non_awq_attn =
            key.contains("self_attn.o_proj") || key.contains("linear_attn.out_proj");

        if is_non_awq_attn {
            return QuantDecision::Skip;
        }

        // ffn_down: "slightly more sensitive" than other FFN variants
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

/// Build a quantization predicate for the openai/privacy-filter checkpoint.
///
/// Privacy-filter is a small MoE classifier (8 layers, 33-class head) shipped
/// in bf16. The right tensors to quantize are:
///
/// - **Quantize at default mode/bits**: attention projections
///   (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MoE expert tensors
///   (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`).
/// - **Router (`mlp.router.weight`)**: 8-bit affine **only** when default mode
///   is `affine`. Skipped under `mxfp4`/`mxfp8`/`nvfp4` because FP modes have
///   no biases and the router's small `[128, 640]` shape is not worth a
///   second quantization scheme. This mirrors the Qwen3.5 convention of
///   keeping routers higher-precision than projections.
/// - **Skip everything else**: token embeddings (lookup table — quantizing
///   hurts), classifier head (`score.weight`/`score.bias` — small + sensitive),
///   norms, biases, and attention sinks (f32, shape `[14]`).
pub(crate) fn build_privacy_filter_predicate(
    default_bits: i32,
    default_group_size: i32,
    default_mode: &str,
) -> Box<dyn Fn(&str) -> QuantDecision + Send + Sync> {
    let default_mode = default_mode.to_string();
    Box::new(move |key: &str| -> QuantDecision {
        // Always-quantize tensors: attention projections and MoE experts.
        let is_attn_proj = key.ends_with(".self_attn.q_proj.weight")
            || key.ends_with(".self_attn.k_proj.weight")
            || key.ends_with(".self_attn.v_proj.weight")
            || key.ends_with(".self_attn.o_proj.weight");
        let is_moe_expert =
            key.ends_with(".mlp.experts.gate_up_proj") || key.ends_with(".mlp.experts.down_proj");
        if is_attn_proj || is_moe_expert {
            return QuantDecision::Custom {
                bits: default_bits,
                group_size: default_group_size,
                mode: default_mode.clone(),
            };
        }

        // Router: 8-bit affine ONLY under affine mode; bf16 under FP modes.
        if key.ends_with(".mlp.router.weight") {
            if default_mode == "affine" {
                return QuantDecision::Custom {
                    bits: 8,
                    group_size: default_group_size,
                    mode: "affine".to_string(),
                };
            }
            return QuantDecision::Skip;
        }

        // Everything else (embed_tokens, score.*, layernorms, biases, sinks,
        // router.bias, post_attention_layernorm, model.norm) — leave at bf16.
        QuantDecision::Skip
    })
}

/// Post-transform that upgrades 8-bit affine decisions to MXFP8 and 4-bit
/// to MXFP4. Acts on the output of any recipe predicate. Group size is
/// forced to 32 for upgraded layers (FFI constraint for mxfp* modes).
///
/// Affine-only loader keys are intentionally excluded from the upgrade:
/// - `lm_head`: dense Qwen3.5's lm_head loader uses `Linear::load_quantized`
///   which hardcodes affine dequantization.
/// - `router.proj`: Gemma4's MoE router uses an affine-only `Linear` for
///   `router.proj`.
/// - `embed_tokens` (and `embed_tokens_per_layer`): Gemma4 / others route
///   quantized embeddings through `Embedding::load_quantized`, which calls
///   `mlx_dequantize(..., "affine")` unconditionally.
/// - `embedding_projection`: Gemma4's `embed_vision.embedding_projection`
///   loads through affine-only `Linear::load_quantized`, so MXFP weights
///   here would be silently mis-dequantized.
/// - Qwen3.5 MoE router gates (`mlp.gate`) and `shared_expert_gate`: their
///   loader IS mode-aware so MXFP8 wouldn't crash, but MXFP8's E8M0 per-
///   group power-of-two scales have ~10x the round-trip error of affine
///   8-bit on small-magnitude gate weights. That much routing noise flips
///   top-K expert selection and produces gibberish output. Python mlx-lm's
///   `quant_predicate` in `qwen3_5.py` hardcodes these gates to
///   `{group_size: 64, bits: 8}` affine for exactly this reason.
///
/// MXFP tensors at any of these keys would be silently misinterpreted at
/// load time or destroy routing precision. Supporting MXFP on these keys
/// requires either a LinearProj-style refactor (affine-only loaders) or a
/// quality reason to break parity with Python mlx-lm (router gates).
pub(crate) fn apply_mxfp_upgrade(
    inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync>,
    default_bits: i32,
) -> Box<dyn Fn(&str) -> QuantDecision + Send + Sync> {
    Box::new(move |key: &str| {
        let original = inner(key);
        // Skip mxfp upgrade for affine-only loaders. See the function-level
        // doc for the full rationale. `embed_tokens` also matches
        // `embed_tokens_per_layer` (Gemma4 PLE embedding) via `contains`.
        if is_affine_only_key(key) {
            return original;
        }
        // Router gates and shared_expert_gate: ALWAYS force 8-bit affine,
        // regardless of what the inner predicate returned. MXFP8's coarse
        // E8M0 scales destroy top-K routing precision. We do not preserve
        // `Skip` here either: a recipe that wants to keep gates at full
        // precision would not have a meaningful interaction with `--q-mxfp`
        // since the loader has no path to load an unquantized gate when
        // every other weight is quantized. Forcing affine 8-bit matches
        // Python mlx-lm's `quant_predicate` in `qwen3_5.py`.
        if is_router_gate(key) {
            match original {
                QuantDecision::Skip => return QuantDecision::Skip,
                _ => {
                    return QuantDecision::Custom {
                        bits: 8,
                        group_size: 64,
                        mode: "affine".into(),
                    };
                }
            }
        }
        match original {
            QuantDecision::Skip => QuantDecision::Skip,
            QuantDecision::Default => match default_bits {
                8 => QuantDecision::Custom {
                    bits: 8,
                    group_size: 32,
                    mode: "mxfp8".into(),
                },
                4 => QuantDecision::Custom {
                    bits: 4,
                    group_size: 32,
                    mode: "mxfp4".into(),
                },
                _ => QuantDecision::Default,
            },
            QuantDecision::Custom { bits: 8, .. } => QuantDecision::Custom {
                bits: 8,
                group_size: 32,
                mode: "mxfp8".into(),
            },
            QuantDecision::Custom { bits: 4, .. } => QuantDecision::Custom {
                bits: 4,
                group_size: 32,
                mode: "mxfp4".into(),
            },
            other => other,
        }
    })
}

/// Post-transform that upgrades 4-bit decisions (Default or Custom) to NVFP4.
/// Acts on the output of any recipe predicate. NVFP4 uses `group_size = 16`
/// (FFI / NVIDIA-style FP4 micro-block constraint). Unlike MXFP there is no
/// 8-bit NVFP variant — 8-bit and other bit widths pass through unchanged.
///
/// Affine-only loader keys (`lm_head`, `router.proj`, `embed_tokens*`,
/// `embedding_projection`) are intentionally excluded — see
/// [`apply_mxfp_upgrade`] for the rationale (same set of affine-only
/// dequantization paths apply here).
///
/// Router gates and `shared_expert_gate` are ALWAYS forced to 8-bit affine
/// `group_size = 64` (mirrors [`apply_mxfp_upgrade`]) — even though NVFP4
/// only promotes 4-bit decisions, a Default decision on a router-gate key
/// must still be forced into the safe affine 8-bit slot to preserve top-K
/// routing precision under `--q-mode nvfp4 --q-recipe ...`.
/// Validate NVFP4 bits/group_size invariant. NVFP4 micro-block constraint
/// requires `bits=4, group_size=16` — any other combination would either
/// trigger a confusing FFI error mid-conversion or (worse, when combined
/// with a recipe that produces a top-level bits mismatch) silently write
/// an inconsistent checkpoint with on-disk metadata that disagrees with
/// the per-layer overrides. Returns the formatted error message on failure.
pub(crate) fn validate_nvfp4_invariants(
    bits: i32,
    group_size: i32,
) -> std::result::Result<(), String> {
    if bits != 4 || group_size != 16 {
        return Err(format!(
            "nvfp4 requires bits=4 and group_size=16 (got bits={bits}, group_size={group_size})"
        ));
    }
    Ok(())
}

/// Validate that `--q-mode nvfp4 + --q-recipe` is restricted to recipes that
/// have model-aware tensor-class exclusions for NVFP4-sensitive layers
/// (`unsloth`, `qwen3_5`). The generic `mixed_*` recipes have no sensitivity
/// skip lists, so layers like `linear_attn.out_proj` (KLD ~6.0 — worst tensor
/// in hybrid Qwen3.5/3.6 models) would be promoted to NVFP4 and corrupt the
/// model. Returns the formatted error message on failure.
pub(crate) fn validate_nvfp4_recipe(recipe: &str) -> std::result::Result<(), String> {
    if recipe != "unsloth" && recipe != "qwen3_5" {
        return Err(format!(
            "--q-mode nvfp4 + --q-recipe is currently supported only for 'unsloth' and 'qwen3_5' recipes (got '{}'). Other recipes lack tensor-class exclusions for NVFP4-sensitive layers (e.g. linear_attn.out_proj).",
            recipe
        ));
    }
    Ok(())
}

pub(crate) fn apply_nvfp4_upgrade(
    inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync>,
) -> Box<dyn Fn(&str) -> QuantDecision + Send + Sync> {
    Box::new(move |key: &str| {
        let original = inner(key);
        // Affine-only loader keys must never be upgraded; the loader would
        // silently mis-dequantize NVFP4 packed weights as affine. If the
        // recipe explicitly emitted a Custom/Skip decision, preserve it.
        // If the recipe deferred (Default), we cannot let it fall through
        // to the top-level `mode=nvfp4` because the affine-only loader
        // rejects non-affine modes — emit an explicit 8-bit affine override
        // so the per-layer metadata wins over the global default.
        // (This affects e.g. Gemma4 MoE `router.proj` under the unsloth
        // recipe, which doesn't have a dedicated branch for it.)
        if is_affine_only_key(key) {
            return match original {
                QuantDecision::Default => QuantDecision::Custom {
                    bits: 8,
                    group_size: 64,
                    mode: "affine".into(),
                },
                other => other,
            };
        }
        // Router gates: always 8-bit affine (group_size 64), preserving Skip
        // if the inner predicate explicitly opted out. See
        // `apply_mxfp_upgrade` for the full rationale.
        if is_router_gate(key) {
            match original {
                QuantDecision::Skip => return QuantDecision::Skip,
                _ => {
                    return QuantDecision::Custom {
                        bits: 8,
                        group_size: 64,
                        mode: "affine".into(),
                    };
                }
            }
        }
        match original {
            QuantDecision::Skip => QuantDecision::Skip,
            // Default → only promote when the global default_bits would be
            // 4-bit. The Default arm here is reached when the recipe defers
            // to the global default; under `--q-mode nvfp4` the only valid
            // default_bits is 4 (validated upstream), so promote
            // unconditionally.
            QuantDecision::Default => QuantDecision::Custom {
                bits: 4,
                group_size: 16,
                mode: "nvfp4".into(),
            },
            // Custom 4-bit decisions (e.g. unsloth recipe's q/k/v affine
            // 4-bit) get promoted to NVFP4 with the required group_size=16.
            QuantDecision::Custom { bits: 4, .. } => QuantDecision::Custom {
                bits: 4,
                group_size: 16,
                mode: "nvfp4".into(),
            },
            // All other Custom decisions (3/5/6/8-bit, etc.) pass through:
            // NVFP4 has no 8-bit variant, and other bit widths must keep
            // whatever mode/group_size the recipe chose.
            other => other,
        }
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

/// Number of leading-axis experts per tile when quantizing MoE expert
/// tensors with shape `[num_experts, M, N]`. The Metal quantize kernel
/// dispatches a single command buffer for the entire tensor, so very large
/// expert stacks (e.g. Qwen3.5 MoE with 256 experts) can exceed the macOS
/// GPU watchdog (`kIOGPUCommandBufferCallbackErrorTimeout`, ~5 s). Quantize
/// groups along the LAST axis, so slicing along axis 0 (the expert axis) is
/// bit-exact identical to a single-call quantize and lets us submit several
/// smaller command buffers instead.
const QUANTIZE_TILE_NUM_EXPERTS: i64 = 32;

/// Trigger axis-0 tiling once a 3D tensor's leading dim reaches this many
/// experts. 32 is the chunk size so any tensor at or above 32 experts is
/// guaranteed to split into at least one full tile plus an optional
/// remainder (256 → 8 tiles of 32; 128 → 4; 64 → 2; 32 → 1).
const QUANTIZE_TILE_THRESHOLD_NUM_EXPERTS: i64 = 32;

/// Quantize `array` with optional axis-0 tiling for large 3D MoE expert
/// tensors. For 2D inputs and small 3D inputs this is a direct passthrough
/// to `mlx_quantize`. For 3D inputs with a large leading dim it slices
/// along axis 0 in `QUANTIZE_TILE_NUM_EXPERTS` chunks, calls `mlx_quantize`
/// on each chunk, evals + synchronizes between chunks to commit each
/// command buffer separately, then concatenates the per-chunk outputs
/// along axis 0. Returns `(packed_weight, scales, optional_biases)` — the
/// biases output is None for mxfp4/mxfp8 (which return null biases) and
/// Some for affine.
///
/// Correctness: MLX quantize groups along the last axis (`group_size`
/// slices the innermost dim), so splitting along any non-last axis
/// preserves group alignment. Concatenating `(packed_0, .., packed_k)`
/// along axis 0 reproduces what a single non-tiled quantize would emit,
/// bit-for-bit, for affine / mxfp4 / mxfp8 modes alike.
fn quantize_with_optional_tiling(
    array: &MxArray,
    group_size: i32,
    bits: i32,
    mode_c: &std::ffi::CStr,
    key_for_error: &str,
) -> Result<(MxArray, MxArray, Option<MxArray>)> {
    let ndim = array.ndim()? as usize;
    let leading_dim = if ndim == 3 { array.shape_at(0)? } else { 0 };

    let should_tile = ndim == 3 && leading_dim >= QUANTIZE_TILE_THRESHOLD_NUM_EXPERTS;

    if !should_tile {
        let mut out_quantized: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_scales: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_biases: *mut mlx_sys::mlx_array = std::ptr::null_mut();

        let ok = unsafe {
            mlx_sys::mlx_quantize(
                array.as_raw_ptr(),
                group_size,
                bits,
                mode_c.as_ptr(),
                &mut out_quantized,
                &mut out_scales,
                &mut out_biases,
            )
        };
        if !ok {
            return Err(Error::from_reason(format!(
                "mlx_quantize failed for tensor '{}'",
                key_for_error
            )));
        }

        let q_weight = MxArray::from_handle(out_quantized, "quantize_weight")?;
        let q_scales = MxArray::from_handle(out_scales, "quantize_scales")?;
        let q_biases = if out_biases.is_null() {
            None
        } else {
            Some(MxArray::from_handle(out_biases, "quantize_biases")?)
        };
        return Ok((q_weight, q_scales, q_biases));
    }

    let chunk = QUANTIZE_TILE_NUM_EXPERTS;
    let mut packed_chunks: Vec<MxArray> = Vec::new();
    let mut scale_chunks: Vec<MxArray> = Vec::new();
    let mut bias_chunks: Vec<MxArray> = Vec::new();
    let mut has_biases = false;

    let mut start: i64 = 0;
    while start < leading_dim {
        let end = (start + chunk).min(leading_dim);
        let slice = array.slice_axis(0, start, end)?;

        let mut out_quantized: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_scales: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_biases: *mut mlx_sys::mlx_array = std::ptr::null_mut();

        let ok = unsafe {
            mlx_sys::mlx_quantize(
                slice.as_raw_ptr(),
                group_size,
                bits,
                mode_c.as_ptr(),
                &mut out_quantized,
                &mut out_scales,
                &mut out_biases,
            )
        };
        if !ok {
            return Err(Error::from_reason(format!(
                "mlx_quantize failed for tensor '{}' chunk [{}, {})",
                key_for_error, start, end
            )));
        }

        let q_weight = MxArray::from_handle(out_quantized, "quantize_weight_chunk")?;
        let q_scales = MxArray::from_handle(out_scales, "quantize_scales_chunk")?;
        let q_biases = if out_biases.is_null() {
            None
        } else {
            Some(MxArray::from_handle(out_biases, "quantize_biases_chunk")?)
        };

        // Force this chunk to commit on its own Metal command buffer so the
        // single-kernel dispatch stays well under the GPU watchdog. Without
        // the synchronize the lazy graph re-fuses the chunks back into one
        // monolithic submission and the timeout returns.
        q_weight.eval();
        q_scales.eval();
        if let Some(b) = &q_biases {
            b.eval();
        }
        crate::array::memory::synchronize_and_clear_cache();

        packed_chunks.push(q_weight);
        scale_chunks.push(q_scales);
        // After the unconditional push above, `packed_chunks.len() > 1` means
        // at least one prior chunk was already processed. If this chunk
        // returned biases but earlier chunks didn't (`!has_biases`), the
        // backend disagreed with itself across slices of the same tensor.
        if let Some(b) = q_biases {
            if !has_biases && packed_chunks.len() > 1 {
                return Err(Error::from_reason(format!(
                    "mlx_quantize returned inconsistent biases across chunks for '{}'",
                    key_for_error
                )));
            }
            has_biases = true;
            bias_chunks.push(b);
        } else if has_biases {
            return Err(Error::from_reason(format!(
                "mlx_quantize returned inconsistent biases across chunks for '{}'",
                key_for_error
            )));
        }

        start = end;
    }

    let packed_refs: Vec<&MxArray> = packed_chunks.iter().collect();
    let scale_refs: Vec<&MxArray> = scale_chunks.iter().collect();
    let packed = MxArray::concatenate_many(packed_refs, Some(0))?;
    let scales = MxArray::concatenate_many(scale_refs, Some(0))?;
    let biases = if has_biases {
        let bias_refs: Vec<&MxArray> = bias_chunks.iter().collect();
        Some(MxArray::concatenate_many(bias_refs, Some(0))?)
    } else {
        None
    };

    Ok((packed, scales, biases))
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
    embed_quantizable: bool,
) -> Result<HashMap<String, serde_json::Value>> {
    use std::ffi::CString;

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
        // Guard against re-quantizing an ALREADY-quantized checkpoint. The
        // normal flow is float (bf16/f16/f32) input with no quant sidecars, so
        // both checks below are no-ops there. They only fire when a converted
        // (already-quantized) checkpoint is fed back through `--quantize`.
        //
        // This loop is in its read-only PHASE 1 (collecting `entries`); the map
        // is not mutated until the quantize phase below, so sidecar presence and
        // dtype are tested against the pristine INPUT map.
        if let Some(base) = key.strip_suffix(".weight") {
            // (a) Skip if a quant sidecar already exists for this group. A
            // packed/affine group carries `{base}.scales`; an FP8 group carries
            // `{base}.weight_scale_inv`. Convert has no dequant-then-requant
            // path, so re-quantizing here would double-quantize / corrupt.
            if weights.contains_key(&format!("{base}.scales"))
                || weights.contains_key(&format!("{base}.weight_scale_inv"))
            {
                info!(
                    "skipping quantization of '{}': already quantized (sidecar present)",
                    key
                );
                continue;
            }
            // (b) Skip if the source weight is not floating-point. `mlx_quantize`
            // only accepts float inputs; a packed `Uint32` (affine/mxfp) or FP8
            // `Uint8` weight would crash or be silently corrupted.
            if let Some(array) = weights.get(key)
                && let Ok(dt) = array.dtype()
                && !matches!(dt, DType::Float32 | DType::Float16 | DType::BFloat16)
            {
                info!(
                    "skipping quantization of '{}': non-float dtype {:?}",
                    key, dt
                );
                continue;
            }
        }
        if let Some(pred) = predicate {
            match pred(key) {
                QuantDecision::Skip => continue,
                QuantDecision::Default => {
                    if !should_quantize(key, embed_quantizable) {
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
            if !should_quantize(key, embed_quantizable) {
                continue;
            }
            // Mirror `apply_mxfp_upgrade`'s exclusions for affine-only
            // loader keys: those keys load through affine-only
            // `Linear::load_quantized` / `Embedding::load_quantized` helpers
            // (dense Qwen3.5 lm_head, Gemma4 MoE `router.proj`, Gemma4
            // `embed_tokens` and `embed_tokens_per_layer`, Gemma4
            // `embed_vision.embedding_projection`), so emitting MXFP / NVFP
            // weights for them would be silently mis-dequantized at load
            // time. Force a safe 8-bit affine (group_size 64) override and
            // let the loader pick up the per-layer override via mode-aware
            // dispatch.
            //
            // Note: `lm_head` is already filtered out by `should_quantize`
            // above (the legacy path never quantizes the output head), so
            // in practice this branch fires for `router.proj`,
            // `embed_tokens*`, and `embedding_projection` keys. The
            // `lm_head` arm is kept for defense-in-depth: if a future edit
            // ever relaxes `should_quantize`, the MXFP/NVFP-mode safety net
            // still holds.
            //
            // `embed_tokens` matches both the top-level Gemma4 embedding
            // and the PLE `embed_tokens_per_layer` via substring.
            //
            // EXCEPTION (lfm2/lfm2_moe): when `embed_quantizable`, the lfm2
            // PACKED embedding backend (`load_quantized_packed`) DOES support
            // mxfp4/mxfp8/nvfp4 (mode threaded through gather-dequant +
            // quantized matmul), so the embedding keys must NOT be force-
            // downgraded to affine here — they keep the global non-affine mode.
            let is_non_affine_default =
                default_mode == "mxfp4" || default_mode == "mxfp8" || default_mode == "nvfp4";
            let is_lfm2_packed_embed =
                embed_quantizable && (key.contains("embed_tokens") || key.contains("embedding."));
            if is_non_affine_default && is_affine_only_key(key) && !is_lfm2_packed_embed {
                entries.push(QuantEntry {
                    key: key.clone(),
                    bits: 8,
                    group_size: gate_group_size,
                    mode: "affine".to_string(),
                });
                continue;
            }
            if is_router_gate(key) {
                // Router gates ALWAYS stay at 8-bit affine, regardless of the
                // top-level default mode. MXFP8 (E8M0 per-group power-of-two
                // scales, group_size 32) has ~10x the round-trip error of
                // affine 8-bit on small-magnitude gate weights — too much
                // noise for top-K expert routing. This matches Python
                // mlx-lm's `quant_predicate` in `qwen3_5.py` which hardcodes
                // gates to `{group_size: 64, bits: 8}` affine.
                //
                // See also the matching gate exclusion in
                // `apply_mxfp_upgrade`, which fires for the recipe (`-q
                // --q-mxfp --q-recipe ...`) path; this branch handles the
                // no-recipe legacy path.
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

        let (q_weight, q_scales, q_biases) = quantize_with_optional_tiling(
            &array,
            entry.group_size,
            entry.bits,
            mode_c.as_c_str(),
            &entry.key,
        )?;

        let prefix = entry.key.strip_suffix(".weight").unwrap_or(&entry.key);
        weights.insert(format!("{}.weight", prefix), q_weight);
        weights.insert(format!("{}.scales", prefix), q_scales);

        if let Some(q_biases) = q_biases {
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
///
/// Returns the per-layer override map produced by `quantize_weights_inner`.
/// The no-recipe path still emits non-default entries for special keys —
/// e.g. router gates are upgraded to mxfp8 under a global MXFP mode, and
/// `lm_head` / `router.proj` are forced back to affine — so callers MUST
/// thread the returned map into `config.json["quantization"]` for the
/// loader to dispatch correctly.
fn quantize_weights(
    weights: &mut HashMap<String, MxArray>,
    bits: i32,
    group_size: i32,
    mode: &str,
    embed_quantizable: bool,
) -> Result<HashMap<String, serde_json::Value>> {
    quantize_weights_inner(weights, bits, group_size, mode, None, embed_quantizable)
}

/// Public wrapper for quantize_weights, accessible from other crate modules (e.g., GGUF converter).
/// Returns the per-layer override map; see `quantize_weights` for why this matters.
///
/// `embed_quantizable` gates quantizing the token embedding (lfm2/lfm2_moe only);
/// see `should_quantize`. GGUF/other callers pass `false` to preserve the
/// embedding-skip behavior.
pub(crate) fn quantize_weights_pub(
    weights: &mut HashMap<String, MxArray>,
    bits: i32,
    group_size: i32,
    mode: &str,
    embed_quantizable: bool,
) -> Result<HashMap<String, serde_json::Value>> {
    quantize_weights(weights, bits, group_size, mode, embed_quantizable)
}

/// Quantize weights with a recipe predicate, returning per-layer overrides.
/// Used by GGUF converter and convert_model when a recipe is specified.
///
/// `embed_quantizable` only affects the predicate's `Default` fall-through and
/// the legacy `is_affine_only_key` force (lfm2/lfm2_moe opt-in); a recipe that
/// emits explicit `Custom`/`Skip` decisions for the embedding is unaffected.
pub(crate) fn quantize_weights_with_recipe_pub(
    weights: &mut HashMap<String, MxArray>,
    bits: i32,
    group_size: i32,
    mode: &str,
    predicate: &(dyn Fn(&str) -> QuantDecision + Send + Sync),
    embed_quantizable: bool,
) -> Result<HashMap<String, serde_json::Value>> {
    quantize_weights_inner(
        weights,
        bits,
        group_size,
        mode,
        Some(predicate),
        embed_quantizable,
    )
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

    // Step 3: Stack/normalize expert weights
    //
    // Two source formats:
    // A) Individual experts: experts.{i}.gate_proj.weight, experts.{i}.up_proj.weight, ...
    //    → Stack into 3D [num_experts, out, in] → switch_mlp.{proj}.weight
    // B) Pre-stacked fused: experts.gate_up_proj [E, fused_out, in], experts.down_proj [E, out, in]
    //    → Split gate_up_proj along dim 1, rename → switch_mlp.{proj}.weight
    //
    // Format B comes from HuggingFace models that already fuse gate+up into one tensor
    // and omit the .weight suffix. Without normalization, should_quantize() skips them
    // (requires .weight suffix), leaving 60GB of expert weights unquantized.

    let has_individual_experts = new_weights
        .keys()
        .any(|k| k.contains(".experts.0.gate_proj.weight"));
    let has_prestacked_experts = new_weights
        .keys()
        .any(|k| k.contains(".experts.gate_up_proj") || k.contains(".experts.down_proj"));

    if has_individual_experts && has_prestacked_experts {
        warn!("Model has both individual and pre-stacked expert weights — using individual format");
    }

    if has_individual_experts {
        // Format A: individual experts → stack
        for l in 0..num_hidden_layers {
            let prefix = format!("language_model.model.layers.{}.mlp", l);
            let first_expert_key = format!("{}.experts.0.gate_proj.weight", prefix);

            if !new_weights.contains_key(&first_expert_key) {
                continue;
            }

            info!(
                "  Layer {}: stacking {} individual experts...",
                l, num_experts
            );

            for proj in &["gate_proj", "up_proj", "down_proj"] {
                let mut to_stack: Vec<MxArray> = Vec::with_capacity(num_experts);
                for e in 0..num_experts {
                    let k = format!("{}.experts.{}.{}.weight", prefix, e, proj);
                    match new_weights.remove(&k) {
                        Some(w) => to_stack.push(w),
                        None => {
                            return Err(Error::from_reason(format!(
                                "Missing expert weight: {}",
                                k
                            )));
                        }
                    }
                }
                let refs: Vec<&MxArray> = to_stack.iter().collect();
                let stacked = MxArray::stack(refs, Some(0))?;
                new_weights.insert(format!("{}.switch_mlp.{}.weight", prefix, proj), stacked);
            }
        }

        // Clean up any remaining individual expert keys
        let expert_keys: Vec<String> = new_weights
            .keys()
            .filter(|k| k.contains(".mlp.experts.") && k.ends_with(".weight"))
            .cloned()
            .collect();
        for k in expert_keys {
            new_weights.remove(&k);
        }
    } else if has_prestacked_experts {
        // Format B: pre-stacked fused experts → split gate_up_proj, rename with .weight suffix
        let expert_keys: Vec<String> = new_weights
            .keys()
            .filter(|k| k.contains(".experts.gate_up_proj") || k.contains(".experts.down_proj"))
            .cloned()
            .collect();

        info!(
            "  Normalizing {} pre-stacked expert tensors (split gate_up_proj, add .weight suffix)",
            expert_keys.len()
        );

        for k in expert_keys {
            let array = new_weights.remove(&k).unwrap();

            if k.ends_with(".experts.gate_up_proj") {
                // Split fused [E, gate_dim+up_dim, in] → gate [E, dim, in] + up [E, dim, in]
                let dim1 = array.shape_at(1)?;
                if dim1 % 2 != 0 {
                    return Err(Error::from_reason(format!(
                        "gate_up_proj dim 1 must be even, got {} for '{}'",
                        dim1, k
                    )));
                }
                let half = dim1 / 2;
                let gate = array.slice_axis(1, 0, half)?;
                let up = array.slice_axis(1, half, dim1)?;

                let base = k.strip_suffix(".experts.gate_up_proj").unwrap();
                new_weights.insert(format!("{}.switch_mlp.gate_proj.weight", base), gate);
                new_weights.insert(format!("{}.switch_mlp.up_proj.weight", base), up);
            } else if k.ends_with(".experts.down_proj") {
                let base = k.strip_suffix(".experts.down_proj").unwrap();
                new_weights.insert(format!("{}.switch_mlp.down_proj.weight", base), array);
            }
        }
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

/// Sanitize an LFM2 / LFM2-MoE HuggingFace checkpoint into the exact on-disk
/// layout the lfm2 loader (`models::lfm2::persistence::sanitize_weights`) reads.
///
/// INVERSE-CONSISTENCY with the loader is the contract here. The loader STRIPS
/// the `model.` prefix on read, applies the SAME MLP rename, the SAME conv
/// transpose, and the SAME expert stacking. So this converter must:
///
/// - KEEP the on-disk `model.` prefix (do NOT re-prefix to
///   `language_model.model.*` — that is qwen3_5_moe-specific and would break
///   the lfm2 loader's `strip_prefix("model.")`).
/// - Rename `feed_forward.{w1,w2,w3}` -> `{gate_proj,down_proj,up_proj}` (covers
///   both dense `feed_forward.wN` and per-expert `experts.{e}.wN`).
/// - Transpose the depthwise short conv `*.conv.conv.weight` from
///   `[channels, 1, kernel]` to `[channels, kernel, 1]` (shape[-1] > shape[1]).
/// - Stack per-expert projections into
///   `feed_forward.switch_mlp.{proj}.weight` of shape `[num_experts, out, in]`
///   for every MoE layer (`num_dense_layers..num_hidden_layers`).
/// - Keep `feed_forward.expert_bias` (f32) untouched and EXCLUDE it from the
///   f32->target dtype cast.
/// - Drop `lm_head.weight` when embeddings are tied.
/// - Cast remaining f32 tensors to `target_dtype`, skipping quantized groups
///   (`.weight`/`.scales`/`.biases`) and `expert_bias`.
///
/// Dense `lfm2` (no `num_experts`) takes the same path minus expert stacking —
/// every layer's `feed_forward` is dense `{gate,up,down}_proj` after the rename.
///
/// Note: this runs BEFORE the generic quantize pass. The lfm2 affine-only gate
/// upstream guarantees only affine quantization reaches that pass.
fn sanitize_lfm2_moe(
    weights: HashMap<String, MxArray>,
    config: &serde_json::Value,
    target_dtype_str: &str,
    tie_word_embeddings: bool,
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

    // LFM2 has no `text_config` nesting — every field is top-level.
    let num_hidden_layers = config
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    // `num_experts` absent => dense `lfm2` (no expert stacking).
    let num_experts = config
        .get("num_experts")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let num_dense_layers = config
        .get("num_dense_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    info!(
        "  lfm2 sanitize: num_hidden_layers={}, num_dense_layers={}, num_experts={:?}, target_dtype={:?}",
        num_hidden_layers, num_dense_layers, num_experts, target_dtype
    );

    // Step 1: key rename + conv transpose + lm_head drop. KEEP the `model.`
    // prefix; the loader strips it on read.
    let mut new_weights: HashMap<String, MxArray> = HashMap::new();
    for (key, value) in weights.into_iter() {
        // Drop the tied output head — the loader reuses embed_tokens via
        // `as_linear()`. (Generic pass already drops it, but a non-tied caller
        // may still ship one we must keep; only drop when tied.)
        if key.ends_with("lm_head.weight") && tie_word_embeddings {
            continue;
        }

        // Conv transpose: `*.conv.conv.weight` where shape[-1] > shape[1] is the
        // HF `[channels, 1, kernel]` layout; transpose to `[channels, kernel, 1]`.
        // Mirrors lfm2 loader `sanitize_weights`.
        let value = if key.ends_with("conv.conv.weight") {
            let ndim = value.ndim().unwrap_or(0);
            if ndim == 3 {
                let dim1 = value.shape_at(1).unwrap_or(0);
                let dim2 = value.shape_at(2).unwrap_or(0);
                if dim2 > dim1 {
                    value.transpose(Some(&[0, 2, 1]))?
                } else {
                    value
                }
            } else {
                value
            }
        } else {
            value
        };

        // MLP rename scoped to `feed_forward.*` so it catches both dense
        // (`feed_forward.wN.weight`) and expert (`experts.{e}.wN.weight`) keys
        // without disturbing unrelated tensors. Renames ALL affine-quant group
        // suffixes — `.weight`, `.scales`, AND `.biases` — to mirror the loader's
        // `sanitize_weights`: a pre-quantized affine HF source ships
        // `feed_forward.wN.{scales,biases}` companions that would otherwise be
        // left orphaned under `wN.*` and rejected/misclassified by the loader.
        let new_key = if key.contains("feed_forward") {
            key.replace("w1.weight", "gate_proj.weight")
                .replace("w1.scales", "gate_proj.scales")
                .replace("w1.biases", "gate_proj.biases")
                .replace("w2.weight", "down_proj.weight")
                .replace("w2.scales", "down_proj.scales")
                .replace("w2.biases", "down_proj.biases")
                .replace("w3.weight", "up_proj.weight")
                .replace("w3.scales", "up_proj.scales")
                .replace("w3.biases", "up_proj.biases")
        } else {
            key
        };

        new_weights.insert(new_key, value);
    }

    // Reject pre-quantized per-expert MoE sources (AFFINE *and* FP8): only the
    // per-expert `.weight` is stacked into `switch_mlp.*.weight` (Step 2); the
    // matching quant sidecars are NOT stacked and would be left orphaned under
    // `experts.{e}.*`, producing a non-loadable checkpoint (Step 3's float-only
    // guard correctly skips the non-float packed/FP8 `.weight`, so the output
    // would carry a raw quantized `switch_mlp.*.weight` with orphaned per-expert
    // sidecars → silent corrupted inference). Fail loud instead — this converter
    // takes an UNQUANTIZED checkpoint and quantizes it; per-expert pre-quantized
    // input is unsupported. The sidecar suffixes covered:
    //   * affine: `.scales` / `.biases`
    //   * FP8:    `.weight_scale_inv` (the loader's FP8 scale sidecar; Step-1's
    //             substring rename rewrites `wN.weight_scale_inv` →
    //             `{proj}.weight_scale_inv` because `wN.weight` is a substring).
    // Scoped to `feed_forward.experts.*` so it does NOT reject: (a) unquantized
    // sources (no such sidecars), (b) already-STACKED quantized sources
    // (`switch_mlp.*.{scales,weight_scale_inv}`, no `experts.`), or (c) dense
    // (non-expert) FP8/affine (`feed_forward.{gate,up,down}_proj.*`, no
    // `experts.`).
    if let Some(bad) = new_weights.keys().find(|k| {
        k.contains("feed_forward.experts.")
            && (k.ends_with(".scales")
                || k.ends_with(".biases")
                || k.ends_with(".weight_scale_inv"))
    }) {
        return Err(Error::from_reason(format!(
            "lfm2 convert: pre-quantized per-expert MoE source is unsupported \
             (found '{bad}'); convert from an unquantized checkpoint instead"
        )));
    }

    // Step 2: stack per-expert projections for every MoE layer. Byte-identical
    // to the loader's stacking (`mx.stack` over axis 0 ->
    // `[num_experts, out, in]`). Skipped entirely for dense `lfm2`.
    if let Some(num_experts) = num_experts {
        for l in num_dense_layers..num_hidden_layers {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let key0 = format!("model.layers.{l}.feed_forward.experts.0.{proj}.weight");
                if !new_weights.contains_key(&key0) {
                    continue;
                }
                let mut arrs = Vec::with_capacity(num_experts);
                for e in 0..num_experts {
                    let kk = format!("model.layers.{l}.feed_forward.experts.{e}.{proj}.weight");
                    let a = new_weights.remove(&kk).ok_or_else(|| {
                        Error::from_reason(format!("lfm2: missing expert weight {kk}"))
                    })?;
                    // Root-cause backstop for ALL pre-quantized per-expert sources:
                    // the corruption is *any non-float expert weight reaching the
                    // stack* (it would be packed into `switch_mlp.*.weight` with no
                    // `.scales`, then loaded as plain bf16 → garbage). The name-based
                    // sidecar reject above catches affine/FP8 sources that ship a
                    // recognized sidecar; this dtype guard also catches a packed
                    // weight that arrives WITHOUT any sidecar (e.g. `wN.weight` as
                    // `Uint32`/`Uint8`). A genuine unquantized source is always
                    // float here, so this never rejects a supported input.
                    let dt = a.dtype()?;
                    if !matches!(dt, DType::Float32 | DType::Float16 | DType::BFloat16) {
                        return Err(Error::from_reason(format!(
                            "lfm2 convert: pre-quantized per-expert MoE source is unsupported \
                             (expert weight '{kk}' has non-float dtype {dt:?}); convert from an \
                             unquantized checkpoint instead"
                        )));
                    }
                    arrs.push(a);
                }
                let refs: Vec<&MxArray> = arrs.iter().collect();
                let stacked = MxArray::stack(refs, Some(0))?; // [num_experts, out, in]
                new_weights.insert(
                    format!("model.layers.{l}.feed_forward.switch_mlp.{proj}.weight"),
                    stacked,
                );
            }
        }
    }

    info!(
        "  After lfm2 rename + expert stacking: {} tensors",
        new_weights.len()
    );

    // Step 3: cast every remaining FLOATING-POINT tensor whose dtype differs
    // from the target to `target_dtype` (so a bf16/f16 source still honors
    // `--dtype`, not just f32). The cast is float-precision conversion ONLY: it
    // NEVER touches packed/integer quant data (packed `Uint32` weights, integer
    // tensors) — those are left in place unchanged. EXCLUDE `expert_bias`
    // (loader keeps it f32 per `cast_predicate`) and skip any quantized tensor
    // groups (none exist on this path, but mirror `sanitize_qwen35_moe` for
    // safety against a pre-quantized source).
    let quantized_bases: std::collections::HashSet<String> = new_weights
        .keys()
        .filter(|k| k.ends_with(".scales"))
        .map(|k| k.strip_suffix(".scales").unwrap_or(k.as_str()).to_string())
        .collect();
    let keys: Vec<String> = new_weights.keys().cloned().collect();
    for k in keys {
        if k.ends_with(".expert_bias") {
            continue;
        }
        if k.ends_with(".scales") || k.ends_with(".biases") {
            continue;
        }
        if let Some(base) = k.strip_suffix(".weight")
            && quantized_bases.contains(base)
        {
            continue; // packed quantized weight — leave as-is
        }
        let v = new_weights
            .get(&k)
            .ok_or_else(|| Error::from_reason(format!("lfm2: tensor {k} vanished during cast")))?;
        // Cast ONLY floating-point tensors whose dtype differs from the target.
        // `target_dtype` is always Float32/Float16/BFloat16 (see match above).
        // Non-float tensors (packed `Uint32` quant weights, integer tensors) are
        // never `astype`d — casting them would corrupt the packed bit layout.
        let dt = v.dtype()?;
        if matches!(dt, DType::Float32 | DType::Float16 | DType::BFloat16) && dt != target_dtype {
            let converted = v.astype(target_dtype)?;
            new_weights.insert(k, converted);
        }
    }

    // Final invariant (root-cause backstop): the converter must NEVER emit a
    // non-float `.weight` that the loader would misread. The loader classifies
    // quantization by sidecar presence, so a non-float weight is acceptable ONLY
    // if it is BOTH (a) a quantizable tensor class AND (b) carries its quant
    // sidecar. The earlier per-expert guards (name-based reject + the dtype check
    // in Step 2) already fail loud on per-expert pre-quantized sources; this is
    // the comprehensive backstop for the DENSE (non-expert) analog and any
    // residual.
    //   * (a) Quantizability is base-aware: the depthwise short conv
    //     (`conv.conv.weight`) is the one LFM2 `.weight` the loader NEVER
    //     dequantizes — it is always cloned into a dense `Conv1d` via
    //     `set_conv_weight` (cf. `should_quantize` excludes it; test
    //     `lfm2_depthwise_conv_not_quantized`). A non-float conv weight is
    //     therefore corrupt regardless of any sidecar and must be rejected.
    //   * (b) Every other non-float weight must keep its `{base}.scales`
    //     (affine/MXFP/NVFP) or `{base}.weight_scale_inv` (FP8) companion; a
    //     valid already-quantized tensor passes, a packed weight with no sidecar
    //     is rejected instead of silently corrupting.
    let weight_keys: Vec<String> = new_weights
        .keys()
        .filter(|k| k.ends_with(".weight"))
        .cloned()
        .collect();
    for k in &weight_keys {
        let base = k.strip_suffix(".weight").unwrap_or(k);
        let Some(v) = new_weights.get(k) else {
            continue;
        };
        let dt = v.dtype()?;
        if matches!(dt, DType::Float32 | DType::Float16 | DType::BFloat16) {
            continue;
        }
        // (a) Always-dense tensor classes must never be non-float, sidecar or
        // not — the loader has NO quantized path for them (it always loads a plain
        // float tensor), so a non-float value corrupts regardless of any sidecar.
        // Exhaustively verified against the loader, exactly two classes:
        //   * the depthwise short conv `conv.conv.weight` (loaded dense via
        //     `set_conv_weight`; never quantized);
        //   * EVERY RMSNorm/LayerNorm weight — all end with `norm.weight`
        //     (embedding_norm, the final `norm`, per-layer operator_norm/ffn_norm,
        //     and attn q_layernorm/k_layernorm), loaded via dense norm setters.
        // No quantizable weight ends with either suffix (linears end in
        // `_proj.weight`/`gate.weight`, embeddings in `tokens.weight`, and the
        // affine-capable `lm_head.weight` is handled by (b) via its sidecar), so
        // this never over-rejects a legitimately-quantized tensor.
        if k.ends_with("conv.conv.weight") || k.ends_with("norm.weight") {
            return Err(Error::from_reason(format!(
                "lfm2 convert: non-float weight '{k}' ({dt:?}) on an always-dense \
                 tensor class (depthwise conv / RMSNorm) — these are never \
                 quantized; convert from an unquantized checkpoint instead"
            )));
        }
        // (b) Any other non-float weight must carry its quant sidecar.
        if !new_weights.contains_key(&format!("{base}.scales"))
            && !new_weights.contains_key(&format!("{base}.weight_scale_inv"))
        {
            return Err(Error::from_reason(format!(
                "lfm2 convert: non-float weight '{k}' ({dt:?}) has no quant sidecar \
                 (.scales / .weight_scale_inv) — pre-quantized source is unsupported; \
                 convert from an unquantized checkpoint instead"
            )));
        }
    }

    info!("  After lfm2 sanitization: {} tensors", new_weights.len());

    Ok(new_weights)
}

// ── AWQ Pre-Scaling ─────────────────────────────────────────────────────────

/// Apply AWQ-style pre-scaling using imatrix importance scores.
///
/// For each scale group, amplifies important weight columns and fuses
/// the inverse into the preceding layer. This improves quantization quality
/// without changing model size or inference speed.
///
/// Scale groups per layer:
///   A: post_attention_layernorm → gate_proj, up_proj (input columns)
///   B: up_proj (output rows) → down_proj (input columns)
///   C: input_layernorm → self_attn.q_proj, k_proj, v_proj (full-attention layers)
///   D: input_layernorm → linear_attn.in_proj_qkv, in_proj_z (GatedDeltaNet layers)
///
/// Note: self_attn.o_proj and linear_attn.out_proj are NOT covered — their inputs
/// come from attention/GDN computation, not from a norm layer. These tensors should
/// be kept at bf16 or quantized without AWQ correction.
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

        // ── Group C: input_layernorm → self_attn.q_proj + k_proj + v_proj ──
        // (Only present in full-attention layers, every full_attention_interval-th layer)
        let q_key = format!("{prefix}.self_attn.q_proj.weight");
        let k_key = format!("{prefix}.self_attn.k_proj.weight");
        let v_key = format!("{prefix}.self_attn.v_proj.weight");
        let input_norm_key = format!("{prefix}.input_layernorm.weight");

        // Only apply if this layer has self_attn weights (full attention layer)
        if weights.contains_key(&q_key)
            && let Some(scales) =
                compute_multi_key_scales(imatrix, &[&q_key, &k_key, &v_key], ratio)?
        {
            for proj_key in [&q_key, &k_key, &v_key] {
                if let Some(proj) = weights.remove(proj_key) {
                    let scaled = scale_columns(&proj, &scales)?;
                    weights.insert(proj_key.to_string(), scaled);
                    modified += 1;
                }
            }
            // input_layernorm.weight /= scales
            if let Some(norm) = weights.remove(&input_norm_key) {
                let inv = invert_scales(&scales)?.astype(norm.dtype()?)?;
                let scaled = norm.mul(&inv)?;
                weights.insert(input_norm_key.clone(), scaled);
                modified += 1;
            } else {
                warn!(
                    "AWQ Group C: input_layernorm.weight missing for layer {} — \
                         projection weights were scaled but inverse not fused into norm",
                    i
                );
            }
        }

        // ── Group D: input_layernorm → linear_attn.in_proj_qkv + in_proj_z ──
        // (Only present in GatedDeltaNet layers)
        let qkv_key = format!("{prefix}.linear_attn.in_proj_qkv.weight");
        let z_key = format!("{prefix}.linear_attn.in_proj_z.weight");

        // Only apply if this layer has linear_attn weights (GDN layer)
        if weights.contains_key(&qkv_key)
            && let Some(scales) = compute_multi_key_scales(imatrix, &[&qkv_key, &z_key], ratio)?
        {
            for proj_key in [&qkv_key, &z_key] {
                if let Some(proj) = weights.remove(proj_key) {
                    let scaled = scale_columns(&proj, &scales)?;
                    weights.insert(proj_key.to_string(), scaled);
                    modified += 1;
                }
            }
            // input_layernorm.weight /= scales
            // Groups C and D are mutually exclusive — a layer is either
            // full-attention or GDN, never both — so this norm is only modified once.
            if let Some(norm) = weights.remove(&input_norm_key) {
                let inv = invert_scales(&scales)?.astype(norm.dtype()?)?;
                let scaled = norm.mul(&inv)?;
                weights.insert(input_norm_key, scaled);
                modified += 1;
            } else {
                warn!(
                    "AWQ Group D: input_layernorm.weight missing for layer {} — \
                         projection weights were scaled but inverse not fused into norm",
                    i
                );
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

/// Compute AWQ scales from multiple weight keys (element-wise max of all importances).
fn compute_multi_key_scales(
    imatrix: &crate::utils::imatrix::ImatrixData,
    keys: &[&str],
    ratio: f32,
) -> Result<Option<MxArray>> {
    let importances: Vec<&Vec<f32>> = keys
        .iter()
        .filter_map(|k| imatrix.importance.get(*k))
        .collect();

    if importances.is_empty() {
        return Ok(None);
    }

    // Require ALL keys present — partial AWQ correction is worse than none
    if importances.len() < keys.len() {
        let missing: Vec<&str> = keys
            .iter()
            .filter(|k| !imatrix.importance.contains_key(**k))
            .copied()
            .collect();
        warn!(
            "AWQ: skipping group — imatrix missing {}/{} keys: {}",
            missing.len(),
            keys.len(),
            missing.join(", ")
        );
        return Ok(None);
    }

    // Validate all importance vectors have the same length
    if importances.len() > 1 {
        let expected_len = importances[0].len();
        for (i, imp) in importances.iter().enumerate().skip(1) {
            if imp.len() != expected_len {
                return Err(Error::from_reason(format!(
                    "AWQ imatrix dimension mismatch: key[0] has {} entries but key[{}] has {}",
                    expected_len,
                    i,
                    imp.len()
                )));
            }
        }
    }

    let len = importances[0].len();
    let mut combined = vec![0.0f32; len];
    for imp in &importances {
        for (j, &val) in imp.iter().enumerate() {
            if j < len {
                combined[j] = combined[j].max(val);
            }
        }
    }

    let scales = compute_normalized_scales(&combined, ratio)?;
    Ok(Some(scales))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience: classify a key under a given mode.
    fn classify(
        predicate: &(dyn Fn(&str) -> QuantDecision + Send + Sync),
        key: &str,
    ) -> QuantDecision {
        predicate(key)
    }

    fn assert_skip(predicate: &(dyn Fn(&str) -> QuantDecision + Send + Sync), key: &str) {
        match classify(predicate, key) {
            QuantDecision::Skip => {}
            other => panic!("expected Skip for {key}, got {other:?}"),
        }
    }

    fn assert_custom(
        predicate: &(dyn Fn(&str) -> QuantDecision + Send + Sync),
        key: &str,
        expect_bits: i32,
        expect_group: i32,
        expect_mode: &str,
    ) {
        match classify(predicate, key) {
            QuantDecision::Custom {
                bits,
                group_size,
                mode,
            } => {
                assert_eq!(bits, expect_bits, "bits mismatch for {key}");
                assert_eq!(group_size, expect_group, "group_size mismatch for {key}");
                assert_eq!(mode, expect_mode, "mode mismatch for {key}");
            }
            other => panic!(
                "expected Custom({expect_bits},{expect_group},{expect_mode}) for {key}, got {other:?}"
            ),
        }
    }

    /// Tensor inventory mirroring the shipped privacy-filter checkpoint.
    fn inventory_keys() -> Vec<&'static str> {
        vec![
            // Top-level
            "model.embed_tokens.weight",
            "model.norm.weight",
            "score.weight",
            "score.bias",
            // Per-layer (layer 0; predicate is layer-agnostic)
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.k_proj.bias",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.v_proj.bias",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.o_proj.bias",
            "model.layers.0.self_attn.sinks",
            "model.layers.0.mlp.router.weight",
            "model.layers.0.mlp.router.bias",
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.gate_up_proj_bias",
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.mlp.experts.down_proj_bias",
        ]
    }

    /// Keys we expect the predicate to **always** skip, regardless of mode.
    fn always_skip_keys() -> Vec<&'static str> {
        vec![
            "model.embed_tokens.weight",
            "model.norm.weight",
            "score.weight",
            "score.bias",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.k_proj.bias",
            "model.layers.0.self_attn.v_proj.bias",
            "model.layers.0.self_attn.o_proj.bias",
            "model.layers.0.self_attn.sinks",
            "model.layers.0.mlp.router.bias",
            "model.layers.0.mlp.experts.gate_up_proj_bias",
            "model.layers.0.mlp.experts.down_proj_bias",
        ]
    }

    /// Keys we expect to be quantized at default (bits, group_size, mode) in any mode.
    fn always_quantize_at_default_keys() -> Vec<&'static str> {
        vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.down_proj",
        ]
    }

    #[test]
    fn privacy_filter_predicate_affine_default_recipe() {
        let predicate = build_privacy_filter_predicate(4, 64, "affine");
        // Projections + experts at default
        for key in always_quantize_at_default_keys() {
            assert_custom(&*predicate, key, 4, 64, "affine");
        }
        // Router quantized at 8-bit affine in affine mode
        assert_custom(
            &*predicate,
            "model.layers.0.mlp.router.weight",
            8,
            64,
            "affine",
        );
        // Always-skip set
        for key in always_skip_keys() {
            assert_skip(&*predicate, key);
        }
        // Sanity: every inventory key got a decision (no panic)
        for key in inventory_keys() {
            let _ = classify(&*predicate, key);
        }
    }

    #[test]
    fn privacy_filter_predicate_mxfp4_skips_router() {
        let predicate = build_privacy_filter_predicate(4, 32, "mxfp4");
        for key in always_quantize_at_default_keys() {
            assert_custom(&*predicate, key, 4, 32, "mxfp4");
        }
        // Router skipped under FP modes
        assert_skip(&*predicate, "model.layers.0.mlp.router.weight");
        for key in always_skip_keys() {
            assert_skip(&*predicate, key);
        }
    }

    #[test]
    fn privacy_filter_predicate_mxfp8_skips_router() {
        let predicate = build_privacy_filter_predicate(8, 32, "mxfp8");
        for key in always_quantize_at_default_keys() {
            assert_custom(&*predicate, key, 8, 32, "mxfp8");
        }
        assert_skip(&*predicate, "model.layers.0.mlp.router.weight");
        for key in always_skip_keys() {
            assert_skip(&*predicate, key);
        }
    }

    #[test]
    fn privacy_filter_predicate_nvfp4_skips_router() {
        let predicate = build_privacy_filter_predicate(4, 16, "nvfp4");
        for key in always_quantize_at_default_keys() {
            assert_custom(&*predicate, key, 4, 16, "nvfp4");
        }
        assert_skip(&*predicate, "model.layers.0.mlp.router.weight");
        for key in always_skip_keys() {
            assert_skip(&*predicate, key);
        }
    }

    /// Predicate applies to any layer index, not just layer 0.
    #[test]
    fn privacy_filter_predicate_layer_agnostic() {
        let predicate = build_privacy_filter_predicate(8, 32, "mxfp8");
        for layer in [0_usize, 3, 7] {
            assert_custom(
                &*predicate,
                &format!("model.layers.{layer}.self_attn.q_proj.weight"),
                8,
                32,
                "mxfp8",
            );
            assert_custom(
                &*predicate,
                &format!("model.layers.{layer}.mlp.experts.down_proj"),
                8,
                32,
                "mxfp8",
            );
            assert_skip(
                &*predicate,
                &format!("model.layers.{layer}.mlp.router.weight"),
            );
            assert_skip(
                &*predicate,
                &format!("model.layers.{layer}.input_layernorm.weight"),
            );
        }
    }

    /// Routers can have substring `router` appearing elsewhere — make sure we
    /// only match `.mlp.router.weight` exactly.
    #[test]
    fn privacy_filter_predicate_router_match_is_exact() {
        let predicate = build_privacy_filter_predicate(4, 32, "mxfp4");
        // A hypothetical key containing "router" but not the right suffix.
        assert_skip(&*predicate, "model.layers.0.mlp.router.bias");
    }

    fn const_predicate(
        decision: QuantDecision,
    ) -> Box<dyn Fn(&str) -> QuantDecision + Send + Sync> {
        Box::new(move |_key: &str| decision.clone())
    }

    #[test]
    fn apply_mxfp_upgrade_passes_through_skip() {
        let wrapped = apply_mxfp_upgrade(const_predicate(QuantDecision::Skip), 8);
        assert_eq!(wrapped("model.embed_tokens.weight"), QuantDecision::Skip);
        assert_eq!(
            wrapped("model.layers.0.self_attn.q_proj.weight"),
            QuantDecision::Skip
        );
        assert_eq!(wrapped(""), QuantDecision::Skip);
    }

    #[test]
    fn apply_mxfp_upgrade_promotes_default_with_8_bits() {
        let wrapped = apply_mxfp_upgrade(const_predicate(QuantDecision::Default), 8);
        assert_eq!(
            wrapped("model.layers.0.mlp.up_proj.weight"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 32,
                mode: "mxfp8".to_string(),
            }
        );
    }

    #[test]
    fn apply_mxfp_upgrade_promotes_default_with_4_bits() {
        let wrapped = apply_mxfp_upgrade(const_predicate(QuantDecision::Default), 4);
        assert_eq!(
            wrapped("model.layers.0.mlp.up_proj.weight"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 32,
                mode: "mxfp4".to_string(),
            }
        );
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_default_with_other_bits() {
        for default_bits in [3, 5, 6] {
            let wrapped = apply_mxfp_upgrade(const_predicate(QuantDecision::Default), default_bits);
            assert_eq!(
                wrapped("model.layers.0.mlp.up_proj.weight"),
                QuantDecision::Default,
                "default_bits = {default_bits} should leave Default unchanged",
            );
        }
    }

    #[test]
    fn apply_mxfp_upgrade_upgrades_custom_8bit_to_mxfp8() {
        // default_bits=3 to prove the Custom arm doesn't read default_bits.
        let wrapped = apply_mxfp_upgrade(
            const_predicate(QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            }),
            3,
        );
        assert_eq!(
            wrapped("model.layers.0.self_attn.q_proj.weight"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 32,
                mode: "mxfp8".to_string(),
            }
        );
    }

    #[test]
    fn apply_mxfp_upgrade_upgrades_custom_4bit_to_mxfp4() {
        let wrapped = apply_mxfp_upgrade(
            const_predicate(QuantDecision::Custom {
                bits: 4,
                group_size: 64,
                mode: "affine".to_string(),
            }),
            3,
        );
        assert_eq!(
            wrapped("model.layers.0.self_attn.q_proj.weight"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 32,
                mode: "mxfp4".to_string(),
            }
        );
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_other_custom_bits() {
        for bits in [2, 3, 5, 6, 7] {
            let original = QuantDecision::Custom {
                bits,
                group_size: 64,
                mode: "affine".to_string(),
            };
            let wrapped = apply_mxfp_upgrade(const_predicate(original.clone()), 8);
            assert_eq!(
                wrapped("model.layers.0.mlp.down_proj.weight"),
                original,
                "Custom {{ bits: {bits}, .. }} should pass through unchanged",
            );
        }
    }

    #[test]
    fn apply_mxfp_upgrade_threads_key_through_to_inner_predicate() {
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> = Box::new(|key: &str| {
            if key.contains("q_proj") {
                QuantDecision::Custom {
                    bits: 8,
                    group_size: 64,
                    mode: "affine".to_string(),
                }
            } else if key.contains("gate_proj") {
                QuantDecision::Custom {
                    bits: 4,
                    group_size: 64,
                    mode: "affine".to_string(),
                }
            } else {
                QuantDecision::Default
            }
        });
        let wrapped = apply_mxfp_upgrade(inner, 8);

        assert_eq!(
            wrapped("layer.0.q_proj"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 32,
                mode: "mxfp8".to_string(),
            }
        );
        assert_eq!(
            wrapped("layer.0.gate_proj"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 32,
                mode: "mxfp4".to_string(),
            }
        );
        // Default → default_bits=8 → mxfp8
        assert_eq!(
            wrapped("layer.0.unknown"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 32,
                mode: "mxfp8".to_string(),
            }
        );
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_lm_head_8bit_decision() {
        // Dense Qwen3.5 lm_head loader is affine-only (Linear::load_quantized
        // hardcodes "affine"); the unsloth recipe emits an 8-bit affine
        // decision for lm_head and apply_mxfp_upgrade must NOT promote that to
        // mxfp8, otherwise the on-disk weights are silently mis-dequantized.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_mxfp_upgrade(inner, 8);
        assert_eq!(
            wrapped("lm_head.weight"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            },
            "lm_head must not be upgraded to mxfp8"
        );
        // Also check the language_model-prefixed naming variant.
        assert_eq!(
            wrapped("language_model.lm_head.weight"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            }
        );
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_router_proj_decision() {
        // Gemma4 router.proj uses Linear::load_quantized (affine-only); the
        // upgrade must skip it for the same reason as lm_head.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_mxfp_upgrade(inner, 8);
        assert_eq!(
            wrapped("language_model.model.layers.0.router.proj.weight"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            },
            "router.proj must not be upgraded to mxfp8"
        );
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_embed_tokens_decision() {
        // Gemma4 (and others) load `embed_tokens` via
        // `Embedding::load_quantized`, which calls
        // `mlx_dequantize(..., "affine")` unconditionally. The upgrade must
        // NOT promote these keys to mxfp4/mxfp8.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_mxfp_upgrade(inner, 8);
        for key in [
            "embed_tokens.weight",
            "model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
            // PLE embedding (Gemma4 per-layer-embedding).
            "embed_tokens_per_layer.weight",
            "model.embed_tokens_per_layer.weight",
        ] {
            assert_eq!(
                wrapped(key),
                QuantDecision::Custom {
                    bits: 8,
                    group_size: 64,
                    mode: "affine".to_string(),
                },
                "{key} must not be upgraded to mxfp8"
            );
        }
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_embed_tokens_under_4bit_promotion() {
        // Same check but for the 4-bit -> mxfp4 promotion path.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Default);
        let wrapped = apply_mxfp_upgrade(inner, 4);
        // A non-excluded key DOES get promoted.
        assert_eq!(
            wrapped("layers.0.mlp.up_proj.weight"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 32,
                mode: "mxfp4".to_string(),
            }
        );
        // embed_tokens / embed_tokens_per_layer are passed through (Default
        // remains Default — the no-recipe legacy block downgrades these to
        // affine separately).
        assert_eq!(wrapped("embed_tokens.weight"), QuantDecision::Default);
        assert_eq!(
            wrapped("embed_tokens_per_layer.weight"),
            QuantDecision::Default
        );
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_embedding_projection_decision() {
        // Gemma4's `embed_vision.embedding_projection` loads through
        // affine-only `Linear::load_quantized`, so MXFP weights here would
        // be silently mis-dequantized. The upgrade must NOT promote it.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_mxfp_upgrade(inner, 8);
        for key in [
            "embed_vision.embedding_projection.weight",
            "model.embed_vision.embedding_projection.weight",
            "language_model.model.embed_vision.embedding_projection.weight",
        ] {
            assert_eq!(
                wrapped(key),
                QuantDecision::Custom {
                    bits: 8,
                    group_size: 64,
                    mode: "affine".to_string(),
                },
                "{key} must not be upgraded to mxfp8"
            );
        }
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_embedding_projection_under_4bit_promotion() {
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Default);
        let wrapped = apply_mxfp_upgrade(inner, 4);
        assert_eq!(
            wrapped("embed_vision.embedding_projection.weight"),
            QuantDecision::Default
        );
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_router_gate_decision() {
        // Qwen3.5 MoE router gates (`mlp.gate`) and `shared_expert_gate`
        // must NOT be upgraded to MXFP8. MXFP8's E8M0 power-of-two scales
        // have ~10x the round-trip error of affine 8-bit on small-magnitude
        // gate weights — too much noise for top-K expert routing, which
        // produces gibberish output. Python mlx-lm's `quant_predicate` in
        // `qwen3_5.py` hardcodes these gates to 8-bit affine.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_mxfp_upgrade(inner, 8);
        for key in [
            "model.layers.0.mlp.gate.weight",
            "language_model.model.layers.7.mlp.gate.weight",
            "layers.0.mlp.gate.weight",
            "model.layers.0.mlp.shared_expert_gate.weight",
            "language_model.model.layers.5.mlp.shared_expert_gate.weight",
        ] {
            assert_eq!(
                wrapped(key),
                QuantDecision::Custom {
                    bits: 8,
                    group_size: 64,
                    mode: "affine".to_string(),
                },
                "{key} must not be upgraded to mxfp8"
            );
        }
    }

    #[test]
    fn apply_mxfp_upgrade_forces_router_gate_to_affine_under_default() {
        // When the inner predicate returns Default, router gates would
        // otherwise inherit the global default mode (mxfp8 for --q-mxfp
        // --q-bits 8) via `quantize_weights_inner`'s Default arm. The
        // upgrade wrapper MUST instead force Custom{8, 64, affine} so that
        // top-K routing precision is preserved.
        let wrapped = apply_mxfp_upgrade(const_predicate(QuantDecision::Default), 8);
        assert_eq!(
            wrapped("model.layers.0.mlp.gate.weight"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            },
        );
        assert_eq!(
            wrapped("model.layers.0.mlp.shared_expert_gate.weight"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            },
        );
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_router_gate_skip_decision() {
        // If a future recipe explicitly Skips router gate quantization,
        // the upgrade should preserve that (don't force-quantize a Skip).
        let wrapped = apply_mxfp_upgrade(const_predicate(QuantDecision::Skip), 8);
        assert_eq!(
            wrapped("model.layers.0.mlp.gate.weight"),
            QuantDecision::Skip,
        );
    }

    // ── NVFP4 validator tests ────────────────────────────────────────────────
    //
    // These validators are called from `convert_model` (safetensors path) and
    // `convert_gguf_to_safetensors` (GGUF path). They surface bad combos with
    // a clear message rather than letting them bubble up as confusing FFI
    // errors mid-conversion (or worse, silently writing an inconsistent
    // checkpoint when a 3-bit recipe default is paired with NVFP4's
    // unconditional 4-bit per-layer overrides).

    #[test]
    fn nvfp4_validator_rejects_bits_8() {
        let err = validate_nvfp4_invariants(8, 16).expect_err("bits=8 must be rejected");
        assert!(
            err.contains("requires bits=4"),
            "error must mention 'requires bits=4', got: {err}"
        );
    }

    #[test]
    fn nvfp4_validator_rejects_group_size_32() {
        let err = validate_nvfp4_invariants(4, 32).expect_err("group_size=32 must be rejected");
        assert!(
            err.contains("group_size=16"),
            "error must mention 'group_size=16', got: {err}"
        );
    }

    #[test]
    fn nvfp4_validator_accepts_bits_4_group_size_16() {
        validate_nvfp4_invariants(4, 16).expect("bits=4, group_size=16 must be accepted");
    }

    #[test]
    fn nvfp4_recipe_rejects_mixed_4_6() {
        let err = validate_nvfp4_recipe("mixed_4_6")
            .expect_err("mixed_4_6 must be rejected under --q-mode nvfp4");
        assert!(
            err.contains("supported only for 'unsloth' and 'qwen3_5'"),
            "error must mention restriction to 'unsloth' and 'qwen3_5', got: {err}"
        );
    }

    #[test]
    fn nvfp4_recipe_accepts_unsloth_and_qwen3_5() {
        validate_nvfp4_recipe("unsloth").expect("unsloth recipe must be accepted");
        validate_nvfp4_recipe("qwen3_5").expect("qwen3_5 recipe must be accepted");
    }

    #[test]
    fn nvfp4_recipe_rejects_all_mixed_variants() {
        for recipe in ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"] {
            assert!(
                validate_nvfp4_recipe(recipe).is_err(),
                "{recipe} must be rejected under --q-mode nvfp4"
            );
        }
    }

    // ── apply_nvfp4_upgrade tests ────────────────────────────────────────────
    //
    // NVFP4 only promotes 4-bit decisions and uses `group_size = 16`. The
    // affine-only-key and router-gate exclusions must match `apply_mxfp_upgrade`
    // exactly so future tensors stay consistent across modes.

    #[test]
    fn apply_nvfp4_upgrade_passes_through_skip() {
        let wrapped = apply_nvfp4_upgrade(const_predicate(QuantDecision::Skip));
        assert_eq!(wrapped("model.embed_tokens.weight"), QuantDecision::Skip);
        assert_eq!(
            wrapped("model.layers.0.self_attn.q_proj.weight"),
            QuantDecision::Skip
        );
        assert_eq!(wrapped(""), QuantDecision::Skip);
    }

    #[test]
    fn apply_nvfp4_upgrade_promotes_default_with_4_bits() {
        // Under `--q-mode nvfp4` the global default_bits is validated to be 4
        // upstream, so the Default arm unconditionally promotes to NVFP4.
        let wrapped = apply_nvfp4_upgrade(const_predicate(QuantDecision::Default));
        assert_eq!(
            wrapped("model.layers.0.mlp.up_proj.weight"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 16,
                mode: "nvfp4".to_string(),
            }
        );
    }

    #[test]
    fn apply_nvfp4_upgrade_upgrades_custom_4bit_to_nvfp4() {
        let wrapped = apply_nvfp4_upgrade(const_predicate(QuantDecision::Custom {
            bits: 4,
            group_size: 32,
            mode: "affine".to_string(),
        }));
        assert_eq!(
            wrapped("model.layers.0.self_attn.q_proj.weight"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 16,
                mode: "nvfp4".to_string(),
            }
        );
    }

    #[test]
    fn apply_nvfp4_upgrade_preserves_custom_8bit() {
        // NVFP4 has no 8-bit variant: 8-bit Custom decisions pass through
        // unchanged (e.g. unsloth recipe's lm_head/router-gate 8-bit affine).
        let original = QuantDecision::Custom {
            bits: 8,
            group_size: 64,
            mode: "affine".to_string(),
        };
        let wrapped = apply_nvfp4_upgrade(const_predicate(original.clone()));
        assert_eq!(
            wrapped("model.layers.0.mlp.down_proj.weight"),
            original,
            "Custom 8-bit must pass through under NVFP4 upgrade"
        );
    }

    #[test]
    fn apply_nvfp4_upgrade_preserves_custom_3bit_5bit_6bit() {
        for bits in [2, 3, 5, 6, 7] {
            let original = QuantDecision::Custom {
                bits,
                group_size: 64,
                mode: "affine".to_string(),
            };
            let wrapped = apply_nvfp4_upgrade(const_predicate(original.clone()));
            assert_eq!(
                wrapped("model.layers.0.mlp.down_proj.weight"),
                original,
                "Custom {{ bits: {bits}, .. }} should pass through unchanged under NVFP4",
            );
        }
    }

    #[test]
    fn apply_nvfp4_upgrade_preserves_lm_head_decision() {
        // Dense Qwen3.5 lm_head loader is affine-only — must not be upgraded
        // to NVFP4.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 4,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_nvfp4_upgrade(inner);
        for key in ["lm_head.weight", "language_model.lm_head.weight"] {
            assert_eq!(
                wrapped(key),
                QuantDecision::Custom {
                    bits: 4,
                    group_size: 64,
                    mode: "affine".to_string(),
                },
                "{key} must not be upgraded to nvfp4"
            );
        }
    }

    #[test]
    fn apply_nvfp4_upgrade_preserves_embed_tokens_decision() {
        // Gemma4 / others load `embed_tokens` through `Embedding::load_quantized`
        // which is affine-only. Also covers PLE `embed_tokens_per_layer` via
        // substring match.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 4,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_nvfp4_upgrade(inner);
        for key in [
            "embed_tokens.weight",
            "model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
            "embed_tokens_per_layer.weight",
            "model.embed_tokens_per_layer.weight",
        ] {
            assert_eq!(
                wrapped(key),
                QuantDecision::Custom {
                    bits: 4,
                    group_size: 64,
                    mode: "affine".to_string(),
                },
                "{key} must not be upgraded to nvfp4"
            );
        }
    }

    #[test]
    fn apply_nvfp4_upgrade_preserves_router_proj_decision() {
        // Gemma4 MoE router uses affine-only `Linear::load_quantized`.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 4,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_nvfp4_upgrade(inner);
        assert_eq!(
            wrapped("language_model.model.layers.0.router.proj.weight"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 64,
                mode: "affine".to_string(),
            },
            "router.proj must not be upgraded to nvfp4"
        );
    }

    #[test]
    fn apply_nvfp4_upgrade_forces_affine_on_router_proj_default() {
        // When the recipe defers (Default) for an affine-only key like
        // Gemma4's router.proj, apply_nvfp4_upgrade must emit an explicit
        // 8-bit affine override — preserving Default would let it fall
        // through to the top-level `mode=nvfp4`, which the affine-only
        // loader rejects at load time. Regression test for the Gemma4
        // NVFP4 failure: "router.proj load: Non-affine FP mode Nvfp4 is
        // not supported; affine only".
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Default);
        let wrapped = apply_nvfp4_upgrade(inner);
        for key in [
            "language_model.model.layers.0.router.proj.weight",
            "language_model.lm_head.weight",
            "language_model.model.embed_tokens.weight",
            "embed_vision.embedding_projection.weight",
        ] {
            assert_eq!(
                wrapped(key),
                QuantDecision::Custom {
                    bits: 8,
                    group_size: 64,
                    mode: "affine".to_string(),
                },
                "{} must get explicit 8-bit affine, not Default",
                key
            );
        }
    }

    #[test]
    fn apply_nvfp4_upgrade_preserves_embedding_projection_decision() {
        // Gemma4's `embed_vision.embedding_projection` loads through affine-
        // only `Linear::load_quantized` — must not be promoted.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 4,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_nvfp4_upgrade(inner);
        for key in [
            "embed_vision.embedding_projection.weight",
            "model.embed_vision.embedding_projection.weight",
            "language_model.model.embed_vision.embedding_projection.weight",
        ] {
            assert_eq!(
                wrapped(key),
                QuantDecision::Custom {
                    bits: 4,
                    group_size: 64,
                    mode: "affine".to_string(),
                },
                "{key} must not be upgraded to nvfp4"
            );
        }
    }

    #[test]
    fn apply_nvfp4_upgrade_forces_router_gate_to_affine() {
        // Router gates and `shared_expert_gate` must ALWAYS land at 8-bit
        // affine gs=64, even when the inner predicate returns Default or a
        // 4-bit Custom decision. NVFP4's FP4 micro-block scales would destroy
        // top-K routing precision (same rationale as MXFP8). This mirrors
        // `apply_mxfp_upgrade`'s router-gate forcing.
        for inner_decision in [
            QuantDecision::Default,
            QuantDecision::Custom {
                bits: 4,
                group_size: 32,
                mode: "affine".to_string(),
            },
            QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            },
        ] {
            let wrapped = apply_nvfp4_upgrade(const_predicate(inner_decision.clone()));
            for key in [
                "model.layers.0.mlp.gate.weight",
                "language_model.model.layers.7.mlp.gate.weight",
                "model.layers.0.mlp.shared_expert_gate.weight",
            ] {
                assert_eq!(
                    wrapped(key),
                    QuantDecision::Custom {
                        bits: 8,
                        group_size: 64,
                        mode: "affine".to_string(),
                    },
                    "{key} (inner = {inner_decision:?}) must be forced to 8-bit affine"
                );
            }
        }
    }

    #[test]
    fn apply_nvfp4_upgrade_preserves_router_gate_skip_decision() {
        // If a recipe explicitly Skips router-gate quantization, preserve it
        // (mirrors `apply_mxfp_upgrade`).
        let wrapped = apply_nvfp4_upgrade(const_predicate(QuantDecision::Skip));
        assert_eq!(
            wrapped("model.layers.0.mlp.gate.weight"),
            QuantDecision::Skip,
        );
        assert_eq!(
            wrapped("model.layers.0.mlp.shared_expert_gate.weight"),
            QuantDecision::Skip,
        );
    }

    #[test]
    fn apply_nvfp4_upgrade_threads_key_through_to_inner_predicate() {
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> = Box::new(|key: &str| {
            if key.contains("q_proj") {
                QuantDecision::Custom {
                    bits: 4,
                    group_size: 32,
                    mode: "affine".to_string(),
                }
            } else if key.contains("o_proj") {
                QuantDecision::Skip
            } else if key.contains("down_proj") {
                QuantDecision::Custom {
                    bits: 8,
                    group_size: 64,
                    mode: "affine".to_string(),
                }
            } else {
                QuantDecision::Default
            }
        });
        let wrapped = apply_nvfp4_upgrade(inner);

        // 4-bit Custom → NVFP4.
        assert_eq!(
            wrapped("model.layers.0.self_attn.q_proj.weight"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 16,
                mode: "nvfp4".to_string(),
            }
        );
        // Skip preserved.
        assert_eq!(
            wrapped("model.layers.0.self_attn.o_proj.weight"),
            QuantDecision::Skip
        );
        // 8-bit Custom preserved (no NVFP8 variant).
        assert_eq!(
            wrapped("model.layers.0.mlp.down_proj.weight"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            }
        );
        // Default → NVFP4.
        assert_eq!(
            wrapped("model.layers.0.mlp.up_proj.weight"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 16,
                mode: "nvfp4".to_string(),
            }
        );
    }

    /// Direct single-call `mlx_quantize` baseline for the tiled-quantize bit-
    /// exactness tests. Returns `(packed, scales, biases?)` where packed is
    /// always uint32, scales/biases dtypes depend on `mode`.
    fn quantize_reference(
        array: &MxArray,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> (MxArray, MxArray, Option<MxArray>) {
        use std::ffi::CString;
        let mode_c = CString::new(mode).unwrap();
        let mut out_quantized: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_scales: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_biases: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let ok = unsafe {
            mlx_sys::mlx_quantize(
                array.as_raw_ptr(),
                group_size,
                bits,
                mode_c.as_ptr(),
                &mut out_quantized,
                &mut out_scales,
                &mut out_biases,
            )
        };
        assert!(ok, "mlx_quantize reference call failed for mode {}", mode);
        let q_weight = MxArray::from_handle(out_quantized, "ref_quantize_weight").unwrap();
        let q_scales = MxArray::from_handle(out_scales, "ref_quantize_scales").unwrap();
        let q_biases = if out_biases.is_null() {
            None
        } else {
            Some(MxArray::from_handle(out_biases, "ref_quantize_biases").unwrap())
        };
        (q_weight, q_scales, q_biases)
    }

    fn assert_shape_eq(a: &MxArray, b: &MxArray, label: &str) {
        let sa: Vec<i64> = a.shape().unwrap().to_vec();
        let sb: Vec<i64> = b.shape().unwrap().to_vec();
        assert_eq!(sa, sb, "{label}: shape mismatch");
    }

    fn assert_uint32_bit_exact(a: &MxArray, b: &MxArray, label: &str) {
        assert_shape_eq(a, b, label);
        let va: Vec<u32> = a.to_uint32().unwrap().to_vec();
        let vb: Vec<u32> = b.to_uint32().unwrap().to_vec();
        assert_eq!(va.len(), vb.len(), "{label}: length mismatch");
        for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
            assert_eq!(x, y, "{label}: bit mismatch at index {i}: {x:#x} vs {y:#x}");
        }
    }

    fn assert_uint8_bit_exact(a: &MxArray, b: &MxArray, label: &str) {
        assert_shape_eq(a, b, label);
        let va = a.to_uint8().unwrap();
        let vb = b.to_uint8().unwrap();
        assert_eq!(va.len(), vb.len(), "{label}: length mismatch");
        for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
            assert_eq!(x, y, "{label}: byte mismatch at index {i}: {x} vs {y}");
        }
    }

    fn assert_float32_bit_exact(a: &MxArray, b: &MxArray, label: &str) {
        assert_shape_eq(a, b, label);
        let va: Vec<f32> = a.to_float32().unwrap().to_vec();
        let vb: Vec<f32> = b.to_float32().unwrap().to_vec();
        assert_eq!(va.len(), vb.len(), "{label}: length mismatch");
        for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{label}: f32 bit mismatch at index {i}: {x} vs {y}"
            );
        }
    }

    #[test]
    fn quantize_with_optional_tiling_passthrough_for_2d() {
        use std::ffi::CString;
        // 2D input — must NOT tile, just delegate to mlx_quantize.
        let w = MxArray::random_normal(&[64, 128], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let mode_c = CString::new("affine").unwrap();
        let (packed, scales, biases) =
            quantize_with_optional_tiling(&w, 64, 4, mode_c.as_c_str(), "test.2d.weight").unwrap();
        let (packed_ref, scales_ref, biases_ref) = quantize_reference(&w, 64, 4, "affine");
        assert_uint32_bit_exact(&packed, &packed_ref, "affine-2d packed");
        assert_float32_bit_exact(&scales, &scales_ref, "affine-2d scales");
        assert!(biases.is_some() && biases_ref.is_some());
        assert_float32_bit_exact(
            biases.as_ref().unwrap(),
            biases_ref.as_ref().unwrap(),
            "affine-2d biases",
        );
    }

    #[test]
    fn quantize_with_optional_tiling_passthrough_for_small_3d() {
        use std::ffi::CString;
        // 3D but leading dim below threshold — must NOT tile.
        let w = MxArray::random_normal(&[8, 32, 64], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let mode_c = CString::new("affine").unwrap();
        let (packed, scales, biases) =
            quantize_with_optional_tiling(&w, 64, 4, mode_c.as_c_str(), "test.small3d.weight")
                .unwrap();
        let (packed_ref, scales_ref, biases_ref) = quantize_reference(&w, 64, 4, "affine");
        assert_uint32_bit_exact(&packed, &packed_ref, "affine-small3d packed");
        assert_float32_bit_exact(&scales, &scales_ref, "affine-small3d scales");
        assert_float32_bit_exact(
            biases.as_ref().unwrap(),
            biases_ref.as_ref().unwrap(),
            "affine-small3d biases",
        );
    }

    #[test]
    fn quantize_with_optional_tiling_bit_exact_affine_4bit() {
        use std::ffi::CString;
        // [64, 32, 64] = 131072 elems, leading dim 64 >= threshold 32.
        // Tiles into 2 chunks of 32.
        let w = MxArray::random_normal(&[64, 32, 64], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let mode_c = CString::new("affine").unwrap();
        let (packed, scales, biases) =
            quantize_with_optional_tiling(&w, 64, 4, mode_c.as_c_str(), "test.tile.affine4.weight")
                .unwrap();
        let (packed_ref, scales_ref, biases_ref) = quantize_reference(&w, 64, 4, "affine");
        assert_uint32_bit_exact(&packed, &packed_ref, "tiled affine-4bit packed");
        assert_float32_bit_exact(&scales, &scales_ref, "tiled affine-4bit scales");
        assert!(biases.is_some() && biases_ref.is_some());
        assert_float32_bit_exact(
            biases.as_ref().unwrap(),
            biases_ref.as_ref().unwrap(),
            "tiled affine-4bit biases",
        );
    }

    #[test]
    fn quantize_with_optional_tiling_bit_exact_mxfp8() {
        use std::ffi::CString;
        // mxfp8 requires bits=8 and group_size=32.
        let w = MxArray::random_normal(&[64, 32, 64], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let mode_c = CString::new("mxfp8").unwrap();
        let (packed, scales, biases) =
            quantize_with_optional_tiling(&w, 32, 8, mode_c.as_c_str(), "test.tile.mxfp8.weight")
                .unwrap();
        let (packed_ref, scales_ref, biases_ref) = quantize_reference(&w, 32, 8, "mxfp8");
        assert_uint32_bit_exact(&packed, &packed_ref, "tiled mxfp8 packed");
        assert_uint8_bit_exact(&scales, &scales_ref, "tiled mxfp8 scales");
        assert!(biases.is_none() && biases_ref.is_none());
    }

    #[test]
    fn quantize_with_optional_tiling_bit_exact_mxfp4() {
        use std::ffi::CString;
        // mxfp4 requires bits=4 and group_size=32.
        let w = MxArray::random_normal(&[64, 32, 64], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let mode_c = CString::new("mxfp4").unwrap();
        let (packed, scales, biases) =
            quantize_with_optional_tiling(&w, 32, 4, mode_c.as_c_str(), "test.tile.mxfp4.weight")
                .unwrap();
        let (packed_ref, scales_ref, biases_ref) = quantize_reference(&w, 32, 4, "mxfp4");
        assert_uint32_bit_exact(&packed, &packed_ref, "tiled mxfp4 packed");
        assert_uint8_bit_exact(&scales, &scales_ref, "tiled mxfp4 scales");
        assert!(biases.is_none() && biases_ref.is_none());
    }

    #[test]
    fn quantize_with_optional_tiling_uneven_remainder() {
        use std::ffi::CString;
        // 80 experts → 32 + 32 + 16; remainder chunk must concat correctly.
        let w = MxArray::random_normal(&[80, 16, 64], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let mode_c = CString::new("affine").unwrap();
        let (packed, scales, biases) =
            quantize_with_optional_tiling(&w, 64, 8, mode_c.as_c_str(), "test.tile.uneven.weight")
                .unwrap();
        let (packed_ref, scales_ref, biases_ref) = quantize_reference(&w, 64, 8, "affine");
        assert_uint32_bit_exact(&packed, &packed_ref, "uneven affine packed");
        assert_float32_bit_exact(&scales, &scales_ref, "uneven affine scales");
        assert_float32_bit_exact(
            biases.as_ref().unwrap(),
            biases_ref.as_ref().unwrap(),
            "uneven affine biases",
        );
    }

    #[test]
    fn quantize_skips_already_quantized_group() {
        // Regression: feeding an ALREADY-quantized checkpoint back through
        // `--quantize` must NOT re-quantize the packed `.weight` (would crash in
        // `mlx_quantize` or double-quantize / corrupt). The shared quantizer
        // (`quantize_weights_inner`) must SKIP any `.weight` that already carries
        // a quant sidecar (`{base}.scales` / `{base}.weight_scale_inv`) or whose
        // dtype is non-float, carrying it through UNCHANGED.

        let h = 64i64; // group_size 64 → packed last dim must be divisible by it
        let mut weights: HashMap<String, MxArray> = HashMap::new();

        // (1) An already-quantized affine group: packed `Uint32` `.weight` of
        // shape [out, packed]. Both dims chosen so the shape/divisibility gate
        // would otherwise ACCEPT it — proving the SKIP fires in the new guard,
        // not as an unrelated shape rejection. Distinct values prove identity.
        let out = 8i64;
        let packed = h; // 64, divisible by group_size 64
        let packed_key = "model.layers.1.feed_forward.switch_mlp.gate_proj.weight";
        let scales_key = "model.layers.1.feed_forward.switch_mlp.gate_proj.scales";
        let packed_data: Vec<u32> = (0..(out * packed) as u32).collect();
        weights.insert(
            packed_key.into(),
            MxArray::from_uint32(&packed_data, &[out, packed]).expect("from_uint32 packed"),
        );
        // group_size 64 over last dim 64 → 1 group per row.
        weights.insert(scales_key.into(), lfm2_bf16(&[out, 1], 0.5));

        // Snapshot the packed input bytes for an identity assertion afterwards.
        let input_packed: Vec<u32> = weights
            .get(packed_key)
            .unwrap()
            .to_uint32()
            .unwrap()
            .to_vec();

        // (2) Positive control: a NORMAL float weight in the SAME map with NO
        // sidecar MUST still be quantized — proving the guard is targeted, not a
        // global disable. `should_quantize` accepts this key.
        let float_key = "model.layers.1.feed_forward.switch_mlp.up_proj.weight";
        weights.insert(float_key.into(), lfm2_bf16(&[out, h], 0.02));
        assert!(
            should_quantize(float_key, false),
            "positive-control weight must be quantize-eligible"
        );

        // Drive the actual quantize loop. Must SUCCEED (no crash/error).
        let overrides = quantize_weights(&mut weights, 4, 64, "affine", false)
            .expect("quantize must not crash on an already-quantized group");

        // The already-quantized packed weight is byte/dtype-identical (NOT
        // re-quantized) and its scales sidecar is preserved.
        let out_weight = weights
            .get(packed_key)
            .expect("packed weight must remain in output map");
        assert_eq!(
            out_weight.dtype().unwrap(),
            DType::Uint32,
            "skipped packed weight must stay Uint32 (not re-quantized)"
        );
        let out_packed: Vec<u32> = out_weight.to_uint32().unwrap().to_vec();
        assert_eq!(
            out_packed, input_packed,
            "skipped packed weight must be byte-identical to the input"
        );
        assert!(
            weights.contains_key(scales_key),
            "pre-existing scales sidecar must be preserved"
        );
        // The skip must NOT have inserted a fresh affine `.biases` for this group.
        let gate_biases_key = "model.layers.1.feed_forward.switch_mlp.gate_proj.biases";
        assert!(
            !weights.contains_key(gate_biases_key),
            "skipped group must not gain a new biases sidecar"
        );

        // Positive control: the float weight WAS quantized (now Uint32 packed,
        // with fresh `.scales` companion).
        let q_float = weights
            .get(float_key)
            .expect("float weight must remain in output map");
        assert_eq!(
            q_float.dtype().unwrap(),
            DType::Uint32,
            "float control weight must have been quantized to packed Uint32"
        );
        assert!(
            weights.contains_key("model.layers.1.feed_forward.switch_mlp.up_proj.scales"),
            "float control weight must gain a fresh scales sidecar"
        );

        // The default 4-bit affine control needs no per-layer override.
        let _ = overrides;
    }

    #[test]
    fn nvfp4_quantize_roundtrip_is_close_to_original() {
        // NVFP4 is lossy (4 bits, group_size 16) so use loose tolerance.
        // Round-trip: quantize -> dequantize -> compare to original.
        let w = MxArray::random_normal(&[64, 128], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let original: Vec<f32> = w.to_float32().unwrap().to_vec();

        let (packed, scales, biases) = quantize_reference(&w, 16, 4, "nvfp4");
        assert!(biases.is_none(), "NVFP4 must not emit biases");
        let packed_shape: Vec<i64> = packed.shape().unwrap().to_vec();
        assert_eq!(
            packed_shape,
            vec![64, 16],
            "NVFP4 packed shape: 128 / 8 per uint32 = 16 packs per row"
        );
        let scales_shape: Vec<i64> = scales.shape().unwrap().to_vec();
        assert_eq!(
            scales_shape,
            vec![64, 8],
            "NVFP4 scales shape: 128 / group_size 16 = 8 groups per row"
        );
        assert!(
            matches!(scales.dtype().unwrap(), DType::Uint8),
            "NVFP4 scales must be uint8 (E4M3 packed)"
        );

        let mode_c = std::ffi::CString::new("nvfp4").unwrap();
        let dequant_handle = unsafe {
            mlx_sys::mlx_dequantize(
                packed.as_raw_ptr(),
                scales.as_raw_ptr(),
                std::ptr::null_mut(),
                16,
                4,
                0, // out_dtype = float32
                mode_c.as_ptr(),
            )
        };
        assert!(!dequant_handle.is_null(), "mlx_dequantize for nvfp4 failed");
        let dequant = MxArray::from_handle(dequant_handle, "nvfp4_dequant").unwrap();
        let restored: Vec<f32> = dequant.to_float32().unwrap().to_vec();
        assert_eq!(restored.len(), original.len());
        // Compute relative error tolerance for NVFP4 (4 bits, gs=16 — much
        // higher precision per block than MXFP4 thanks to the finer block).
        // Empirically restored values land within ~0.1 absolute on N(0,
        // 0.02) inputs; we use a generous 0.2 to keep the test stable.
        let max_abs = original
            .iter()
            .zip(&restored)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs < 0.2,
            "NVFP4 round-trip max abs error {} too large",
            max_abs
        );
    }

    /// Compute relative L2 reconstruction error for a quantize/dequantize
    /// round-trip on a given mode. Returns ||W - Q^-1(Q(W))||_2 / ||W||_2.
    fn quantize_roundtrip_rel_l2(w: &MxArray, group_size: i32, bits: i32, mode: &str) -> f32 {
        use std::ffi::CString;
        let (packed, scales, biases) = quantize_reference(w, group_size, bits, mode);
        let mode_c = CString::new(mode).unwrap();
        let biases_ptr = biases
            .as_ref()
            .map_or(std::ptr::null_mut(), |b| b.as_raw_ptr());
        let dequant_handle = unsafe {
            mlx_sys::mlx_dequantize(
                packed.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases_ptr,
                group_size,
                bits,
                0, // out_dtype = float32
                mode_c.as_ptr(),
            )
        };
        assert!(
            !dequant_handle.is_null(),
            "mlx_dequantize failed for {mode}"
        );
        let dequant = MxArray::from_handle(dequant_handle, "roundtrip_dequant").unwrap();
        let original: Vec<f32> = w.to_float32().unwrap().to_vec();
        let restored: Vec<f32> = dequant.to_float32().unwrap().to_vec();
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (a, b) in original.iter().zip(restored.iter()) {
            let d = (*a - *b) as f64;
            num += d * d;
            den += (*a as f64) * (*a as f64);
        }
        (num.sqrt() / den.sqrt().max(1e-12)) as f32
    }

    /// Diagnostic: compare MXFP8 vs affine-8bit round-trip error on a tensor
    /// shaped like a Qwen3.5 MoE router gate ([num_experts=256, hidden=2048]).
    ///
    /// Router gate inputs are post-RMSNorm activations with small magnitudes;
    /// gate weights are correspondingly small (initialized N(0, 0.02)). The
    /// routing softmax + argpartition are extremely sensitive to per-row
    /// noise. The Python mlx-lm `quant_predicate` in `qwen3_5.py` keeps gates
    /// at 8-bit affine regardless of the global quantization mode for this
    /// reason.
    ///
    /// This test documents that MXFP8 (E8M0 per-group power-of-two scale,
    /// group_size 32) has materially worse round-trip error than affine
    /// 8-bit (per-group scale + bias, group_size 64) on the gate-shaped
    /// tensor. The check is loose: we only require MXFP8 error to exceed
    /// affine error by at least 5x. Tightening this further would risk
    /// flakiness across MLX backend changes.
    #[test]
    fn router_gate_shape_mxfp8_vs_affine_error() {
        // Router gate shape from Qwen3.6-35B-A3B MoE: 256 experts, hidden 2048.
        let w = MxArray::random_normal(&[256, 2048], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let err_mxfp8 = quantize_roundtrip_rel_l2(&w, 32, 8, "mxfp8");
        let err_affine = quantize_roundtrip_rel_l2(&w, 64, 8, "affine");
        eprintln!(
            "router_gate_shape err: mxfp8={:.6}  affine8={:.6}  ratio={:.2}x",
            err_mxfp8,
            err_affine,
            err_mxfp8 / err_affine.max(1e-9)
        );
        assert!(
            err_mxfp8 > err_affine * 5.0,
            "expected MXFP8 error to be much larger than affine 8-bit on router-gate-shaped tensor; \
             got mxfp8={err_mxfp8}, affine8={err_affine}"
        );
    }

    // ── LFM2 convert sanitizer ──────────────────────────────────────────────

    /// bf16 array filled with `fill`, shaped as given.
    fn lfm2_bf16(shape: &[i64], fill: f32) -> MxArray {
        let n: i64 = shape.iter().product();
        let data: Vec<f32> = vec![fill; n.max(0) as usize];
        MxArray::from_float32(&data, shape)
            .expect("from_float32")
            .astype(DType::BFloat16)
            .expect("astype bf16")
    }

    /// f32 array filled with `fill` (for `expert_bias`).
    fn lfm2_f32(shape: &[i64], fill: f32) -> MxArray {
        let n: i64 = shape.iter().product();
        let data: Vec<f32> = vec![fill; n.max(0) as usize];
        MxArray::from_float32(&data, shape).expect("from_float32")
    }

    /// Tiny LFM2-MoE config: 3 layers, 1 dense + 2 MoE, 4 experts. Conv layer 0
    /// (dense), MoE layers 1 and 2.
    fn lfm2_moe_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "lfm2_moe",
            "num_hidden_layers": 3,
            "num_dense_layers": 1,
            "num_experts": 4,
            "tie_word_embeddings": true,
        })
    }

    /// Build a small HF-style LFM2-MoE param map (keys carry the `model.` prefix
    /// exactly as on disk). Layer 0 dense, layers 1-2 MoE.
    fn lfm2_moe_hf_params() -> HashMap<String, MxArray> {
        let h = 4i64;
        let inter = 8i64; // dense intermediate
        let moe_inter = 6i64; // expert intermediate
        let experts = 4i64;

        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));
        // Tied output head present on disk — must be dropped.
        p.insert("lm_head.weight".into(), lfm2_bf16(&[16, h], 0.01));

        // Layer 0: dense conv layer.
        p.insert(
            "model.layers.0.operator_norm.weight".into(),
            lfm2_bf16(&[h], 1.0),
        );
        p.insert(
            "model.layers.0.ffn_norm.weight".into(),
            lfm2_bf16(&[h], 1.0),
        );
        // HF conv weight: [channels, 1, kernel] -> must transpose to [channels, kernel, 1].
        p.insert(
            "model.layers.0.conv.conv.weight".into(),
            lfm2_bf16(&[h, 1, 3], 0.5),
        );
        p.insert(
            "model.layers.0.conv.in_proj.weight".into(),
            lfm2_bf16(&[3 * h, h], 0.1),
        );
        p.insert(
            "model.layers.0.conv.out_proj.weight".into(),
            lfm2_bf16(&[h, h], 0.1),
        );
        // Dense feed-forward w1/w2/w3 -> gate/down/up.
        p.insert(
            "model.layers.0.feed_forward.w1.weight".into(),
            lfm2_bf16(&[inter, h], 0.1),
        );
        p.insert(
            "model.layers.0.feed_forward.w2.weight".into(),
            lfm2_bf16(&[h, inter], 0.1),
        );
        p.insert(
            "model.layers.0.feed_forward.w3.weight".into(),
            lfm2_bf16(&[inter, h], 0.1),
        );

        // Layers 1 + 2: MoE (one conv-ish + one attention; operator type is
        // irrelevant to the sanitizer — only the feed_forward matters here).
        for l in 1..=2 {
            let pre = format!("model.layers.{l}");
            p.insert(format!("{pre}.operator_norm.weight"), lfm2_bf16(&[h], 1.0));
            p.insert(format!("{pre}.ffn_norm.weight"), lfm2_bf16(&[h], 1.0));
            // Router gate + expert bias (f32, must stay f32).
            p.insert(
                format!("{pre}.feed_forward.gate.weight"),
                lfm2_bf16(&[experts, h], 0.05),
            );
            p.insert(
                format!("{pre}.feed_forward.expert_bias"),
                lfm2_f32(&[experts], 0.0),
            );
            for e in 0..experts {
                p.insert(
                    format!("{pre}.feed_forward.experts.{e}.w1.weight"),
                    lfm2_bf16(&[moe_inter, h], 0.1),
                );
                p.insert(
                    format!("{pre}.feed_forward.experts.{e}.w2.weight"),
                    lfm2_bf16(&[h, moe_inter], 0.1),
                );
                p.insert(
                    format!("{pre}.feed_forward.experts.{e}.w3.weight"),
                    lfm2_bf16(&[moe_inter, h], 0.1),
                );
            }
        }
        p
    }

    #[test]
    fn lfm2_sanitize_produces_loader_consistent_keys() {
        let cfg = lfm2_moe_config();
        let out = sanitize_lfm2_moe(lfm2_moe_hf_params(), &cfg, "bfloat16", true)
            .expect("sanitize_lfm2_moe");

        // `model.` prefix KEPT (loader strips it on read).
        assert!(out.contains_key("model.embed_tokens.weight"));
        assert!(out.contains_key("model.embedding_norm.weight"));
        // Must NOT re-prefix to `language_model.model.*`.
        assert!(
            !out.keys().any(|k| k.starts_with("language_model.")),
            "lfm2 keys must not be re-prefixed with language_model.*: {:?}",
            out.keys().collect::<Vec<_>>()
        );

        // lm_head dropped (tied).
        assert!(!out.contains_key("lm_head.weight"));

        // Dense layer 0: w1/w2/w3 renamed.
        assert!(out.contains_key("model.layers.0.feed_forward.gate_proj.weight"));
        assert!(out.contains_key("model.layers.0.feed_forward.down_proj.weight"));
        assert!(out.contains_key("model.layers.0.feed_forward.up_proj.weight"));
        assert!(!out.contains_key("model.layers.0.feed_forward.w1.weight"));
        assert!(!out.contains_key("model.layers.0.feed_forward.w2.weight"));
        assert!(!out.contains_key("model.layers.0.feed_forward.w3.weight"));

        // Conv weight transposed [4,1,3] -> [4,3,1].
        let conv = out
            .get("model.layers.0.conv.conv.weight")
            .expect("conv weight present");
        let shape = conv.shape().expect("shape");
        assert_eq!(
            shape.to_vec(),
            vec![4, 3, 1],
            "conv weight must be transposed"
        );

        // MoE layers 1 + 2: experts stacked into switch_mlp.{proj} [E, out, in].
        for l in 1..=2 {
            let pre = format!("model.layers.{l}");
            for (proj, out_dim, in_dim) in [
                ("gate_proj", 6i64, 4i64),
                ("up_proj", 6i64, 4i64),
                ("down_proj", 4i64, 6i64),
            ] {
                let key = format!("{pre}.feed_forward.switch_mlp.{proj}.weight");
                let stacked = out.get(&key).unwrap_or_else(|| panic!("missing {key}"));
                let shape = stacked.shape().expect("shape");
                assert_eq!(
                    shape.to_vec(),
                    vec![4, out_dim, in_dim],
                    "{key} must be [num_experts, out, in]"
                );
            }
            // Individual expert keys consumed.
            assert!(
                !out.keys()
                    .any(|k| k.contains(&format!("layers.{l}.feed_forward.experts."))),
                "individual expert keys for layer {l} must be consumed"
            );
            // Router gate preserved under `feed_forward.gate.weight`.
            assert!(out.contains_key(&format!("{pre}.feed_forward.gate.weight")));
            // expert_bias preserved AND still f32.
            let bias = out
                .get(&format!("{pre}.feed_forward.expert_bias"))
                .expect("expert_bias present");
            assert_eq!(
                bias.dtype().expect("dtype"),
                DType::Float32,
                "expert_bias must stay f32"
            );
        }
    }

    #[test]
    fn lfm2_sanitize_dense_has_no_expert_stacking() {
        // Dense `lfm2` config: no `num_experts` -> no stacking, all feed_forward
        // is dense {gate,up,down}_proj after the rename.
        let cfg = serde_json::json!({
            "model_type": "lfm2",
            "num_hidden_layers": 1,
            "num_dense_layers": 0,
            "tie_word_embeddings": false,
        });
        let h = 4i64;
        let inter = 8i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));
        p.insert(
            "model.layers.0.feed_forward.w1.weight".into(),
            lfm2_bf16(&[inter, h], 0.1),
        );
        p.insert(
            "model.layers.0.feed_forward.w2.weight".into(),
            lfm2_bf16(&[h, inter], 0.1),
        );
        p.insert(
            "model.layers.0.feed_forward.w3.weight".into(),
            lfm2_bf16(&[inter, h], 0.1),
        );

        let out = sanitize_lfm2_moe(p, &cfg, "bfloat16", false).expect("sanitize dense lfm2");
        assert!(out.contains_key("model.layers.0.feed_forward.gate_proj.weight"));
        assert!(out.contains_key("model.layers.0.feed_forward.up_proj.weight"));
        assert!(out.contains_key("model.layers.0.feed_forward.down_proj.weight"));
        assert!(
            !out.keys().any(|k| k.contains("switch_mlp")),
            "dense lfm2 must not produce switch_mlp keys"
        );
    }

    #[test]
    fn sanitize_lfm2_moe_rejects_per_expert_affine_quant_companions() {
        // A PRE-QUANTIZED per-expert AFFINE source ships `.scales`/`.biases`
        // companions next to a packed `Uint32` `.weight`. The converter only
        // stacks `.weight` into `switch_mlp.*.weight`; the companions would be
        // orphaned, so it MUST fail loud — never silently cast (which would
        // corrupt the packed weight) and never produce a non-loadable map.
        let cfg = lfm2_moe_config(); // 3 layers, 1 dense + 2 MoE, 4 experts.

        let h = 4i64;
        let moe_inter = 6i64;
        let experts = 4i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();

        // Minimal non-expert tensors so the map is plausibly a real checkpoint.
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));

        // MoE layer 1: per-expert AFFINE quant companions on `w1` (renamed to
        // `gate_proj`). Packed `.weight` is `Uint32`; `.scales`/`.biases` are
        // small float companions. The reject is key-name based and fires before
        // any cast, so the exact dtypes/shapes here are only for realism.
        let packed = (moe_inter * h / 8).max(1); // arbitrary small packed length
        for e in 0..experts {
            let pre = format!("model.layers.1.feed_forward.experts.{e}");
            let packed_data: Vec<u32> = vec![0u32; packed as usize];
            p.insert(
                format!("{pre}.w1.weight"),
                MxArray::from_uint32(&packed_data, &[packed]).expect("from_uint32 packed weight"),
            );
            p.insert(format!("{pre}.w1.scales"), lfm2_bf16(&[moe_inter, 1], 1.0));
            p.insert(format!("{pre}.w1.biases"), lfm2_bf16(&[moe_inter, 1], 0.0));
        }

        let err = sanitize_lfm2_moe(p, &cfg, "bfloat16", true)
            .err()
            .expect("per-expert affine quant companions must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("per-expert MoE source is unsupported"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn sanitize_lfm2_moe_rejects_per_expert_fp8_companions() {
        // A PRE-QUANTIZED per-expert FP8 source ships a `weight_scale_inv` scale
        // sidecar (the loader's FP8 dequant key) next to a raw FP8/U8 `.weight`.
        // The converter only stacks `.weight` into `switch_mlp.*.weight`; the
        // `weight_scale_inv` companions would be left orphaned under
        // `experts.{e}.*`, and Step 3's float-only guard would skip the non-float
        // weight — producing a raw quantized `switch_mlp.*.weight` with orphaned
        // per-expert scale sidecars (silent corrupted inference). It MUST fail
        // loud. NB Step-1's substring rename rewrites `w1.weight_scale_inv` →
        // `gate_proj.weight_scale_inv` (because `w1.weight` is a substring), so at
        // the reject point the sidecar is `...gate_proj.weight_scale_inv`.
        let cfg = lfm2_moe_config(); // 3 layers, 1 dense + 2 MoE, 4 experts.

        let h = 4i64;
        let moe_inter = 6i64;
        let experts = 4i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();

        // Minimal non-expert tensors so the map is plausibly a real checkpoint.
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));

        // MoE layer 1: per-expert FP8 companions on `w1` (renamed to
        // `gate_proj`). The reject is key-name based and fires before any cast,
        // so the exact dtypes/shapes here are only for realism: a tiny 1-D array
        // whose length matches its element count avoids `from_*` panics.
        let n = moe_inter * h; // weight element count
        for e in 0..experts {
            let pre = format!("model.layers.1.feed_forward.experts.{e}");
            let weight_data: Vec<f32> = vec![0.0f32; n as usize];
            p.insert(
                format!("{pre}.w1.weight"),
                MxArray::from_float32(&weight_data, &[n]).expect("from_float32 fp8 weight"),
            );
            // FP8 scale sidecar (`weight_scale_inv`). Tiny 1-D scale array.
            p.insert(
                format!("{pre}.w1.weight_scale_inv"),
                lfm2_bf16(&[moe_inter], 1.0),
            );
        }

        let err = sanitize_lfm2_moe(p, &cfg, "bfloat16", true)
            .err()
            .expect("per-expert fp8 quant companions must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("per-expert MoE source is unsupported"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn sanitize_lfm2_moe_rejects_per_expert_packed_weight_without_sidecar() {
        // The hardest variant: a PRE-QUANTIZED per-expert source whose `.weight`
        // is packed (`Uint32`/`Uint8`) but ships NO recognized sidecar
        // (`.scales`/`.biases`/`weight_scale_inv`). The name-based reject can't see
        // it, so the dtype guard inside the stacking loop must catch it — otherwise
        // the raw packed weight would be stacked into `switch_mlp.*.weight` with no
        // `.scales`, then loaded as a plain bf16 SwitchGLU weight → garbage.
        let cfg = lfm2_moe_config(); // 3 layers, 1 dense + 2 MoE, 4 experts.

        let h = 4i64;
        let moe_inter = 6i64;
        let experts = 4i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();

        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));

        // MoE layer 1: per-expert packed `w1`/`w2`/`w3` weights (`Uint32`), no
        // sidecars at all. A 1-D array whose length matches its element count
        // avoids `from_*` panics; the dtype guard fires on the first expert.
        let packed = (moe_inter * h / 8).max(1);
        for e in 0..experts {
            let pre = format!("model.layers.1.feed_forward.experts.{e}");
            for w in ["w1", "w2", "w3"] {
                let packed_data: Vec<u32> = vec![0u32; packed as usize];
                p.insert(
                    format!("{pre}.{w}.weight"),
                    MxArray::from_uint32(&packed_data, &[packed])
                        .expect("from_uint32 packed weight"),
                );
            }
        }

        let err = sanitize_lfm2_moe(p, &cfg, "bfloat16", true)
            .err()
            .expect("per-expert packed weight without sidecar must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("per-expert MoE source is unsupported"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn sanitize_lfm2_moe_rejects_dense_packed_weight_without_sidecar() {
        // The DENSE (non-expert) analog of the per-expert no-sidecar case: a dense
        // layer ships a packed `Uint32` `.weight` with no `.scales`. It is not
        // touched by the per-expert guards (no `experts.`), so the final invariant
        // guard must reject it — otherwise it would be saved as a packed weight
        // with no sidecar and loaded as bf16 → garbage.
        let cfg = lfm2_moe_config(); // layer 0 is dense (num_dense_layers = 1).

        let h = 4i64;
        let inter = 8i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));

        // Dense layer 0: packed `w1` (renamed to `gate_proj`), NO sidecar.
        let packed = (inter * h / 8).max(1);
        let packed_data: Vec<u32> = vec![0u32; packed as usize];
        p.insert(
            "model.layers.0.feed_forward.w1.weight".into(),
            MxArray::from_uint32(&packed_data, &[packed]).expect("from_uint32 packed weight"),
        );

        let err = sanitize_lfm2_moe(p, &cfg, "bfloat16", true)
            .err()
            .expect("dense packed weight without sidecar must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("has no quant sidecar"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn sanitize_lfm2_moe_rejects_quantized_depthwise_conv() {
        // The depthwise short conv is NEVER quantized — the loader always clones
        // `conv.conv.weight` into a dense `Conv1d`. A malformed source that ships a
        // non-float conv weight WITH a `.scales` sidecar would satisfy a naive
        // sidecar-presence check yet load as a dense conv → garbage. The final
        // invariant must reject it regardless of the sidecar (base-aware).
        let cfg = lfm2_moe_config();

        let h = 4i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));

        // Layer 0 depthwise conv: packed `Uint32` weight WITH a `.scales` sidecar.
        let packed = 4i64;
        let packed_data: Vec<u32> = vec![0u32; packed as usize];
        p.insert(
            "model.layers.0.conv.conv.weight".into(),
            MxArray::from_uint32(&packed_data, &[packed]).expect("from_uint32 packed weight"),
        );
        p.insert(
            "model.layers.0.conv.conv.scales".into(),
            lfm2_bf16(&[h, 1], 1.0),
        );

        let err = sanitize_lfm2_moe(p, &cfg, "bfloat16", true)
            .err()
            .expect("quantized depthwise conv must be rejected even with a sidecar");
        let msg = err.to_string();
        assert!(
            msg.contains("always-dense"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn sanitize_lfm2_moe_rejects_quantized_norm_weight() {
        // Norm weights (RMSNorm/LayerNorm) are ALWAYS loaded dense — the loader has
        // no quantized path for any `*norm.weight`. A malformed source that ships a
        // non-float norm weight WITH a `.scales` sidecar would satisfy a naive
        // sidecar check yet load as a dense norm → garbage. The base-aware
        // invariant must reject every `*norm.weight` non-float value, sidecar or
        // not. (`embedding_norm` here; per-layer operator_norm/ffn_norm/q_layernorm/
        // k_layernorm and the final `norm` share the `norm.weight` suffix.)
        let cfg = lfm2_moe_config();

        let h = 4i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );

        // embedding_norm: packed `Uint32` weight WITH a `.scales` sidecar.
        let packed = 4i64;
        let packed_data: Vec<u32> = vec![0u32; packed as usize];
        p.insert(
            "model.embedding_norm.weight".into(),
            MxArray::from_uint32(&packed_data, &[packed]).expect("from_uint32 packed weight"),
        );
        p.insert(
            "model.embedding_norm.scales".into(),
            lfm2_bf16(&[h, 1], 1.0),
        );

        let err = sanitize_lfm2_moe(p, &cfg, "bfloat16", true)
            .err()
            .expect("quantized norm weight must be rejected even with a sidecar");
        let msg = err.to_string();
        assert!(
            msg.contains("always-dense"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn sanitize_lfm2_moe_keeps_already_stacked_quant_group() {
        // The final invariant guard must NOT over-reject a legitimately quantized
        // tensor: an already-STACKED affine quant group (`switch_mlp.*.weight`
        // packed `Uint32` + matching `switch_mlp.*.scales`) carries its sidecar, so
        // it passes through untouched (no `experts.` → no stacking; `.scales`
        // present → Step-3 skips the cast; sidecar present → final guard passes).
        let cfg = lfm2_moe_config();

        let h = 4i64;
        let moe_inter = 6i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));

        // MoE layer 1: already-stacked affine quant group with sidecar.
        let packed = (moe_inter * h / 8).max(1);
        let packed_data: Vec<u32> = vec![0u32; packed as usize];
        let base = "model.layers.1.feed_forward.switch_mlp.gate_proj";
        p.insert(
            format!("{base}.weight"),
            MxArray::from_uint32(&packed_data, &[packed]).expect("from_uint32 packed weight"),
        );
        p.insert(format!("{base}.scales"), lfm2_bf16(&[moe_inter, 1], 1.0));

        let out = sanitize_lfm2_moe(p, &cfg, "bfloat16", true)
            .expect("already-stacked quant group must pass");
        assert!(out.contains_key(&format!("{base}.weight")));
        assert!(out.contains_key(&format!("{base}.scales")));
    }

    #[test]
    fn lfm2_router_gate_is_router_gate() {
        // The lfm2 router (`feed_forward.gate`) MUST be treated as a router gate
        // so it routes to the 8-bit affine branch.
        assert!(is_router_gate(
            "language_model.model.layers.5.feed_forward.gate.weight"
        ));
        assert!(is_router_gate("model.layers.5.feed_forward.gate.weight"));
        // Sanity: qwen-style gates still match.
        assert!(is_router_gate("model.layers.0.mlp.gate.weight"));
    }

    #[test]
    fn lfm2_depthwise_conv_not_quantized() {
        // The depthwise short conv must never be quantized.
        assert!(!should_quantize(
            "model.layers.0.conv.conv.weight",
            /* embed_quantizable */ false
        ));
        // But the conv in/out projections (standard matmuls) SHOULD be.
        assert!(should_quantize("model.layers.0.conv.in_proj.weight", false));
        assert!(should_quantize(
            "model.layers.0.conv.out_proj.weight",
            false
        ));
        // And stacked experts SHOULD be quantizable.
        assert!(should_quantize(
            "model.layers.1.feed_forward.switch_mlp.gate_proj.weight",
            false
        ));
    }

    #[test]
    fn embed_tokens_quantized_only_when_embed_quantizable() {
        // Default (non-lfm2): the token embedding is SKIPPED (preserves
        // qwen3_5/gemma4 behavior).
        assert!(!should_quantize(
            "model.embed_tokens.weight",
            /* embed_quantizable */ false
        ));
        assert!(!should_quantize(
            "model.language_model.embedding.weight",
            false
        ));

        // lfm2/lfm2_moe opt-in: the PACKED embedding backend handles a
        // quantized table, so the embedding IS quantizable.
        assert!(should_quantize(
            "model.embed_tokens.weight",
            /* embed_quantizable */ true
        ));

        // A TIED lm_head is ALWAYS excluded, even when embeds are quantizable
        // (it is dropped at sanitize; we never quantize an output head here).
        assert!(!should_quantize("lm_head.weight", true));
        assert!(!should_quantize("lm_head.weight", false));
    }
}
