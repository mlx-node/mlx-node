//! Foreign Weight Format Loaders
//!
//! Native Rust loaders for PyTorch and Paddle weight files, converting them
//! to SafeTensors format with key remapping for our model architectures.
//!
//! Supported formats:
//! - PyTorch `.pt` / `.pkl` / `.pth` (ZIP archive with pickle + raw tensor data)
//! - Paddle `.pdparams` (pickle with inline numpy arrays)

use std::collections::HashMap;
use std::fs;
use std::io::Read as IoRead;
use std::path::{Path, PathBuf};

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::array::MxArray;
use crate::utils::pickle::{PickleValue, unpickle};
use crate::utils::safetensors::save_safetensors;

// ===========================================================================
// Public NAPI types
// ===========================================================================

#[napi(object)]
pub struct ForeignConversionOptions {
    /// Path to the input weights file (.pdparams, .pkl, .pt, .pth)
    pub input_path: String,
    /// Output directory for model.safetensors + config.json
    pub output_dir: String,
    /// Model type: "pp-lcnet-ori" or "uvdoc"
    pub model_type: String,
    /// Enable verbose logging
    pub verbose: Option<bool>,
}

#[napi(object)]
pub struct ForeignConversionResult {
    pub num_tensors: i32,
    pub output_path: String,
    pub tensor_names: Vec<String>,
}

// ===========================================================================
// NAPI entry point
// ===========================================================================

#[napi]
pub fn convert_foreign_weights(
    options: ForeignConversionOptions,
) -> Result<ForeignConversionResult> {
    let input_path = PathBuf::from(&options.input_path);
    let output_dir = PathBuf::from(&options.output_dir);
    let verbose = options.verbose.unwrap_or(false);

    if !input_path.exists() {
        return Err(Error::from_reason(format!(
            "Input file not found: {}",
            input_path.display()
        )));
    }

    fs::create_dir_all(&output_dir).map_err(|e| {
        Error::from_reason(format!(
            "Failed to create output directory {}: {}",
            output_dir.display(),
            e
        ))
    })?;

    let (tensors, config_json) = match options.model_type.as_str() {
        "pp-lcnet-ori" => convert_pp_lcnet_ori(&input_path, verbose)?,
        "uvdoc" => convert_uvdoc(&input_path, verbose)?,
        other => {
            return Err(Error::from_reason(format!(
                "Unknown foreign model type: '{other}'. Supported: pp-lcnet-ori, uvdoc"
            )));
        }
    };

    // Save SafeTensors
    let weights_path = output_dir.join("model.safetensors");
    save_safetensors(&weights_path, &tensors, None)?;

    let mut tensor_names: Vec<String> = tensors.keys().cloned().collect();
    tensor_names.sort();
    let num_tensors = tensor_names.len() as i32;

    if verbose {
        eprintln!(
            "Saved {} tensors to {}",
            num_tensors,
            weights_path.display()
        );
    }

    // Save config.json
    let config_path = output_dir.join("config.json");
    fs::write(&config_path, config_json)
        .map_err(|e| Error::from_reason(format!("Failed to write config.json: {}", e)))?;

    if verbose {
        eprintln!("Saved config to {}", config_path.display());
    }

    Ok(ForeignConversionResult {
        num_tensors,
        output_path: output_dir.to_string_lossy().to_string(),
        tensor_names,
    })
}

// ===========================================================================
// PP-LCNet orientation model (Paddle .pdparams)
// ===========================================================================

fn convert_pp_lcnet_ori(
    input_path: &Path,
    verbose: bool,
) -> Result<(HashMap<String, MxArray>, String)> {
    // Auto-detect format based on input path
    let tensors = if input_path.is_dir() {
        // Directory mode: look for inference.pdiparams + inference.json
        let pdiparams = input_path.join("inference.pdiparams");
        let json = input_path.join("inference.json");
        if pdiparams.exists() && json.exists() {
            let raw = load_paddle_inference_params(&pdiparams, &json, verbose)?;
            remap_inference_keys(raw, verbose)?
        } else {
            // Fall back to looking for .pdparams file in directory
            let pdparams = find_pdparams_in_dir(input_path)?;
            let raw = load_paddle_params(&pdparams, verbose)?;
            remap_training_keys(raw, verbose)?
        }
    } else if input_path.extension().is_some_and(|e| e == "pdiparams") {
        // .pdiparams file: find companion inference.json
        let json = input_path.with_extension("json");
        if !json.exists() {
            return Err(Error::from_reason(format!(
                "Companion inference.json not found at {}",
                json.display()
            )));
        }
        let raw = load_paddle_inference_params(input_path, &json, verbose)?;
        remap_inference_keys(raw, verbose)?
    } else {
        // Legacy .pdparams pickle format
        let raw = load_paddle_params(input_path, verbose)?;
        remap_training_keys(raw, verbose)?
    };

    let config = serde_json::json!({
        "model_type": "pp_lcnet_x1_0_doc_ori",
        "num_classes": 4,
        "labels": ["0", "90", "180", "270"],
        "scale": 1.0,
        "input_size": 224,
    });

    Ok((tensors, serde_json::to_string_pretty(&config).unwrap()))
}

// ===========================================================================
// UVDoc unwarping model (PyTorch .pkl or Paddle .pdiparams)
// ===========================================================================

fn convert_uvdoc(input_path: &Path, verbose: bool) -> Result<(HashMap<String, MxArray>, String)> {
    // Auto-detect format
    let tensors = if input_path.is_dir() {
        // Directory: look for inference.pdiparams + inference.json
        let pdiparams = input_path.join("inference.pdiparams");
        let json = input_path.join("inference.json");
        if pdiparams.exists() && json.exists() {
            let raw = load_paddle_inference_params(&pdiparams, &json, verbose)?;
            remap_uvdoc_inference_keys(raw, verbose)?
        } else {
            // Fall back: look for PyTorch checkpoint files
            let pytorch_exts = ["pkl", "pt", "pth"];
            let pytorch_file = std::fs::read_dir(input_path)
                .map_err(|e| Error::from_reason(format!("Failed to read dir: {e}")))?
                .filter_map(|entry| entry.ok())
                .find(|entry| {
                    entry
                        .path()
                        .extension()
                        .is_some_and(|ext| pytorch_exts.iter().any(|e| ext == *e))
                })
                .map(|entry| entry.path())
                .ok_or_else(|| {
                    Error::from_reason(format!(
                        "No inference.pdiparams/json or .pkl/.pt/.pth found in {}",
                        input_path.display()
                    ))
                })?;
            let raw_tensors = load_pytorch_checkpoint(&pytorch_file, verbose)?;
            convert_uvdoc_pytorch_keys(raw_tensors, verbose)?
        }
    } else if input_path.extension().is_some_and(|e| e == "pdiparams") {
        let json = input_path.with_extension("json");
        if !json.exists() {
            return Err(Error::from_reason(format!(
                "Companion inference.json not found at {}",
                json.display()
            )));
        }
        let raw = load_paddle_inference_params(input_path, &json, verbose)?;
        remap_uvdoc_inference_keys(raw, verbose)?
    } else {
        // PyTorch checkpoint (.pkl / .pt / .pth)
        let raw_tensors = load_pytorch_checkpoint(input_path, verbose)?;
        convert_uvdoc_pytorch_keys(raw_tensors, verbose)?
    };

    let config = serde_json::json!({
        "model_type": "uvdoc",
        "num_filter": 32,
        "kernel_size": 5,
        "img_size": [488, 712],
    });

    Ok((tensors, serde_json::to_string_pretty(&config).unwrap()))
}

/// Process PyTorch UVDoc checkpoint keys.
///
/// Filters out unwanted keys (_metadata, num_batches_tracked, 3D output head)
/// and normalizes block-0 conv keys to include `.0.` prefix for consistency
/// with the persistence loader.
fn convert_uvdoc_pytorch_keys(
    raw_tensors: HashMap<String, MxArray>,
    verbose: bool,
) -> Result<HashMap<String, MxArray>> {
    let mut tensors = HashMap::new();
    for (key, array) in raw_tensors {
        // Skip internal PyTorch metadata
        if key == "_metadata" || key.starts_with("_") {
            continue;
        }
        // Skip batch tracking counters
        if key.ends_with(".num_batches_tracked") {
            continue;
        }
        // Skip 3D output head (we only need 2D displacement)
        if key.starts_with("out_point_positions3D") {
            continue;
        }

        // Normalize block-0 conv keys: PyTorch uses "conv1.weight" for block 0
        // but "conv1.0.weight" for blocks 1+. Our persistence code always
        // expects the ".0." format, so normalize block 0 keys.
        let normalized = normalize_uvdoc_pytorch_key(&key);

        if verbose {
            if normalized != key {
                eprintln!("  {key} -> {normalized}");
            } else {
                eprintln!("  {key}");
            }
        }
        tensors.insert(normalized, array);
    }
    Ok(tensors)
}

/// Normalize PyTorch UVDoc keys to match the naming convention expected by
/// our persistence loader (which uses the Paddle inference convention).
///
/// Two normalizations:
/// 1. Block-0 conv keys: `layer1.0.conv1.weight` → `layer1.0.conv1.0.weight`
///    (Block 0 uses direct Conv2d, blocks 1+ use Sequential(Conv2d))
/// 2. Bridge 1-3 keys: `bridge_N.0.X.suffix` → `bridge_N.0.0.X.suffix`
///    (Paddle has an extra Sequential nesting for single-element bridge branches)
fn normalize_uvdoc_pytorch_key(key: &str) -> String {
    // 1. Normalize block-0 conv keys
    if key.contains("resnet_down.") && key.contains(".0.conv") {
        if let Some(conv_pos) = key.find(".conv1.") {
            let after_conv = &key[conv_pos + 6..];
            if !after_conv.starts_with(".0.") {
                let (prefix, suffix) = key.split_at(conv_pos + 6);
                return format!("{prefix}.0{suffix}");
            }
        }
        if let Some(conv_pos) = key.find(".conv2.") {
            let after_conv = &key[conv_pos + 6..];
            if !after_conv.starts_with(".0.") {
                let (prefix, suffix) = key.split_at(conv_pos + 6);
                return format!("{prefix}.0{suffix}");
            }
        }
    }

    // 2. Normalize bridge_1/2/3 keys: add extra .0 nesting
    // PyTorch: bridge_N.0.{0,1}.suffix (N=1,2,3, single dilated conv)
    // Paddle:  bridge_N.0.0.{0,1}.suffix (extra Sequential wrapping)
    for n in 1..=3 {
        let prefix = format!("bridge_{n}.0.");
        if key.starts_with(&prefix) {
            let rest = &key[prefix.len()..];
            // Only transform if rest starts with 0. or 1. (conv or bn index)
            // Don't transform if already has extra .0. nesting
            if rest.starts_with("0.") || rest.starts_with("1.") {
                return format!("{prefix}0.{rest}");
            }
        }
    }

    key.to_string()
}

// ===========================================================================
// Paddle .pdparams loader
// ===========================================================================

/// Load Paddle `.pdparams` file (pickle containing numpy arrays).
///
/// Paddle's `paddle.save()` converts tensors to numpy arrays before pickling.
/// The pickle contains a dict: {str: numpy.ndarray, ...}
fn load_paddle_params(path: &Path, verbose: bool) -> Result<HashMap<String, MxArray>> {
    let data = fs::read(path)
        .map_err(|e| Error::from_reason(format!("Failed to read {}: {e}", path.display())))?;

    if verbose {
        eprintln!("Loading Paddle params from {}...", path.display());
    }

    let root = unpickle(&data)?;

    let dict = root
        .as_dict()
        .ok_or_else(|| Error::from_reason("Paddle params: root is not a dict".to_string()))?;

    let mut tensors = HashMap::new();
    for (key_val, val) in dict {
        let key = key_val
            .as_string()
            .ok_or_else(|| Error::from_reason("Paddle params: key is not a string".to_string()))?;

        let array = numpy_value_to_mxarray(val, key)?;

        if verbose {
            let shape = array.shape()?;
            eprintln!("  {key}  shape={:?}", shape.as_ref());
        }

        tensors.insert(key.to_string(), array);
    }

    Ok(tensors)
}

// ===========================================================================
// PyTorch checkpoint loader
// ===========================================================================

/// Load a PyTorch checkpoint file (`.pt`, `.pkl`, `.pth`).
///
/// PyTorch's `torch.save()` creates a ZIP archive containing:
/// - `archive/data.pkl` (or `data.pkl`) - pickle with state dict structure
/// - `archive/data/0`, `archive/data/1`, ... - raw tensor data
///
/// The pickle references "storages" by key; actual data lives in the ZIP entries.
fn load_pytorch_checkpoint(path: &Path, verbose: bool) -> Result<HashMap<String, MxArray>> {
    let file_data = fs::read(path)
        .map_err(|e| Error::from_reason(format!("Failed to read {}: {e}", path.display())))?;

    if verbose {
        eprintln!("Loading PyTorch checkpoint from {}...", path.display());
    }

    // PyTorch checkpoints are ZIP archives
    let cursor = std::io::Cursor::new(&file_data);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| Error::from_reason(format!("Not a valid ZIP/PyTorch file: {e}")))?;

    // Find the pickle file inside the archive
    let pickle_name = find_pickle_entry(&archive)?;

    // Read the pickle data
    let pickle_data = {
        let mut entry = archive.by_name(&pickle_name).map_err(|e| {
            Error::from_reason(format!("Failed to read pickle entry '{pickle_name}': {e}"))
        })?;
        let mut buf = Vec::new();
        entry
            .read_to_end(&mut buf)
            .map_err(|e| Error::from_reason(format!("Failed to read pickle data: {e}")))?;
        buf
    };

    // Parse pickle to get the state dict structure
    let root = unpickle(&pickle_data)?;

    // The root might be a dict directly (state_dict) or a checkpoint dict
    // with "model_state" / "state_dict" keys
    let state_dict = extract_state_dict(&root)?;

    // Build storage map: storage_key -> (dtype, raw_data)
    // We lazily read storage data from the ZIP as needed
    let mut tensors = HashMap::new();

    for (key_val, val) in state_dict {
        let key = key_val.as_string().ok_or_else(|| {
            Error::from_reason("PyTorch state_dict: key is not a string".to_string())
        })?;

        // Skip non-tensor entries (e.g., _metadata dict added by PyTorch)
        if key.starts_with('_') || key.ends_with(".num_batches_tracked") {
            if verbose {
                eprintln!("  {key}  [skipped]");
            }
            continue;
        }

        let array = pytorch_value_to_mxarray(val, key, &mut archive)?;

        if verbose {
            let shape = array.shape()?;
            eprintln!("  {key}  shape={:?}", shape.as_ref());
        }

        tensors.insert(key.to_string(), array);
    }

    Ok(tensors)
}

/// Find the pickle entry in a PyTorch ZIP archive.
fn find_pickle_entry<R: IoRead + std::io::Seek>(archive: &zip::ZipArchive<R>) -> Result<String> {
    // Try common paths
    for name in archive.file_names() {
        if name.ends_with("data.pkl") || name.ends_with("/data.pkl") {
            return Ok(name.to_string());
        }
    }
    // Fall back to any .pkl file
    for name in archive.file_names() {
        if name.ends_with(".pkl") {
            return Ok(name.to_string());
        }
    }
    Err(Error::from_reason(
        "No pickle file found in PyTorch archive".to_string(),
    ))
}

/// Extract the state dict from a PyTorch checkpoint.
/// Handles both raw state dicts and wrapped checkpoints.
fn extract_state_dict(root: &PickleValue) -> Result<&[(PickleValue, PickleValue)]> {
    // If root is a dict, check if it has model_state/state_dict keys
    if let Some(dict) = root.as_dict() {
        for (k, v) in dict {
            if let Some(key) = k.as_string()
                && (key == "model_state" || key == "state_dict")
                && let Some(inner) = v.as_dict()
            {
                return Ok(inner);
            }
        }
        // Otherwise the root dict IS the state dict
        return Ok(dict);
    }

    // Handle OrderedDict (collections.OrderedDict)
    if let Some((module, name, args, _state)) = root.as_object()
        && module == "collections"
        && name == "OrderedDict"
    {
        // OrderedDict is constructed from a list of (key, value) tuples
        // or from a dict. Check args.
        if let Some(first_arg) = args.first()
            && let Some(dict) = first_arg.as_dict()
        {
            return Ok(dict);
        }
        // Empty OrderedDict
        return Ok(&[]);
    }

    Err(Error::from_reason(
        "Could not extract state dict from PyTorch checkpoint".to_string(),
    ))
}

// ===========================================================================
// Tensor reconstruction from pickle values
// ===========================================================================

/// Convert a numpy ndarray pickle value to MxArray.
///
/// numpy arrays in pickle are reconstructed via:
///   numpy.core.multiarray._reconstruct(ndarray, (0,), b'b')
/// with BUILD state: (version, shape, dtype_obj, is_fortran, data_bytes)
fn numpy_value_to_mxarray(val: &PickleValue, name: &str) -> Result<MxArray> {
    let (module, class_name, _args, state) = val.as_object().ok_or_else(|| {
        Error::from_reason(format!(
            "Tensor '{name}': expected numpy object, got {:?}",
            std::mem::discriminant(val)
        ))
    })?;

    // Verify this is a numpy reconstruction
    if !module.contains("numpy") && !class_name.contains("reconstruct") && class_name != "ndarray" {
        return Err(Error::from_reason(format!(
            "Tensor '{name}': unexpected object {module}.{class_name}"
        )));
    }

    let state = state.ok_or_else(|| {
        Error::from_reason(format!(
            "Tensor '{name}': numpy object has no state (BUILD)"
        ))
    })?;

    let state_tuple = state.as_tuple().ok_or_else(|| {
        Error::from_reason(format!("Tensor '{name}': numpy state is not a tuple"))
    })?;

    // State tuple: (version, shape, dtype, is_fortran, data_bytes)
    // version is typically 1
    if state_tuple.len() < 5 {
        return Err(Error::from_reason(format!(
            "Tensor '{name}': numpy state has {} elements, expected 5",
            state_tuple.len()
        )));
    }

    let shape = extract_shape(&state_tuple[1], name)?;
    let dtype_str = extract_numpy_dtype(&state_tuple[2], name)?;
    let data = state_tuple[4]
        .as_bytes()
        .ok_or_else(|| Error::from_reason(format!("Tensor '{name}': numpy data is not bytes")))?;

    bytes_to_mxarray(data, &shape, &dtype_str, name)
}

/// Convert a PyTorch tensor pickle value to MxArray.
///
/// PyTorch tensors use: torch._utils._rebuild_tensor_v2(storage, offset, shape, stride, requires_grad, backward_hooks)
/// The storage references a data file in the ZIP archive.
fn pytorch_value_to_mxarray<R: IoRead + std::io::Seek>(
    val: &PickleValue,
    name: &str,
    archive: &mut zip::ZipArchive<R>,
) -> Result<MxArray> {
    let (_module, func, args, _state) = val.as_object().ok_or_else(|| {
        Error::from_reason(format!(
            "Tensor '{name}': expected torch object, got {:?}",
            std::mem::discriminant(val)
        ))
    })?;

    if func == "_rebuild_tensor_v2" {
        // args: [storage, storage_offset, size, stride, requires_grad, backward_hooks]
        if args.len() < 4 {
            return Err(Error::from_reason(format!(
                "Tensor '{name}': _rebuild_tensor_v2 has {} args, expected >= 4",
                args.len()
            )));
        }

        let storage = &args[0];
        let storage_offset = args[1].as_int().unwrap_or(0) as usize;
        let shape = extract_shape(&args[2], name)?;

        // Storage is a reconstructed object:
        //   torch._utils._rebuild_storage(StorageClass, storage_key, location, num_elements)
        // or just the storage object itself
        let (dtype_str, storage_key) = extract_storage_info(storage, name)?;

        // Read raw data from ZIP archive
        let data = read_storage_data(archive, &storage_key)?;

        let elem_size = numpy_dtype_size(&dtype_str);
        let num_elements: usize = shape.iter().map(|&d| d as usize).product();
        let byte_offset = storage_offset * elem_size;
        let byte_len = num_elements * elem_size;

        if byte_offset + byte_len > data.len() {
            return Err(Error::from_reason(format!(
                "Tensor '{name}': data slice out of bounds (offset={byte_offset}, len={byte_len}, total={})",
                data.len()
            )));
        }

        let slice = &data[byte_offset..byte_offset + byte_len];
        bytes_to_mxarray(slice, &shape, &dtype_str, name)
    } else {
        // Try as numpy array (some checkpoints contain numpy arrays directly)
        numpy_value_to_mxarray(val, name)
    }
}

/// Extract storage info (dtype_str, storage_key) from a PyTorch storage object.
fn extract_storage_info(storage: &PickleValue, name: &str) -> Result<(String, String)> {
    match storage {
        PickleValue::Object {
            module: _,
            name: func_name,
            args,
            ..
        } => {
            // _rebuild_storage or _rebuild_tensor_v2's storage arg
            // Common patterns:
            //   torch._utils._rebuild_storage(StorageClass, key, location, num_elements)
            //   or the storage class itself as args[0]

            // Try to find the storage class to determine dtype
            let mut dtype_str = "f32".to_string();
            let mut storage_key = "0".to_string();

            for arg in args {
                match arg {
                    PickleValue::Global {
                        module: _,
                        name: cls_name,
                    }
                    | PickleValue::Object { name: cls_name, .. } => {
                        if let Some(dt) = storage_class_to_dtype(cls_name) {
                            dtype_str = dt.to_string();
                        }
                    }
                    PickleValue::String(s)
                        // Could be storage_key or location
                        // Storage keys are typically numeric strings
                        if s.chars().all(|c| c.is_ascii_digit()) => {
                            storage_key = s.clone();
                        }
                    _ => {}
                }
            }

            // If func_name itself is a storage class
            if let Some(dt) = storage_class_to_dtype(func_name) {
                dtype_str = dt.to_string();
            }

            Ok((dtype_str, storage_key))
        }
        PickleValue::Tuple(t) if t.len() >= 3 => {
            // (storage_class, key, location, num_elements) tuple
            let mut dtype_str = "f32".to_string();
            let mut storage_key = "0".to_string();

            if let Some(cls) = t[0].as_string()
                && let Some(dt) = storage_class_to_dtype(cls)
            {
                dtype_str = dt.to_string();
            }
            if let Some(key) = t[1].as_string() {
                storage_key = key.to_string();
            }

            Ok((dtype_str, storage_key))
        }
        _ => Err(Error::from_reason(format!(
            "Tensor '{name}': cannot extract storage info"
        ))),
    }
}

/// Map PyTorch storage class names to dtype strings
fn storage_class_to_dtype(class_name: &str) -> Option<&'static str> {
    match class_name {
        "FloatStorage" | "float" => Some("f32"),
        "HalfStorage" | "half" => Some("f16"),
        "BFloat16Storage" | "bfloat16" => Some("bf16"),
        "DoubleStorage" | "double" => Some("f64"),
        "IntStorage" | "int" => Some("i32"),
        "LongStorage" | "long" => Some("i64"),
        "ByteStorage" | "byte" | "uint8" => Some("u8"),
        "CharStorage" | "int8" => Some("i8"),
        _ => None,
    }
}

/// Read raw storage data from a PyTorch ZIP archive.
fn read_storage_data<R: IoRead + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    storage_key: &str,
) -> Result<Vec<u8>> {
    // Try common paths for the data file
    let candidates = [
        format!("archive/data/{storage_key}"),
        format!("data/{storage_key}"),
        storage_key.to_string(),
    ];

    for candidate in &candidates {
        if let Ok(mut entry) = archive.by_name(candidate) {
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf).map_err(|e| {
                Error::from_reason(format!("Failed to read storage '{candidate}': {e}"))
            })?;
            return Ok(buf);
        }
    }

    // Try to find by suffix match
    let names: Vec<String> = archive.file_names().map(|s| s.to_string()).collect();
    for name in &names {
        if name.ends_with(&format!("/{storage_key}")) {
            let mut entry = archive
                .by_name(name)
                .map_err(|e| Error::from_reason(format!("Failed to read storage '{name}': {e}")))?;
            let mut buf = Vec::new();
            entry
                .read_to_end(&mut buf)
                .map_err(|e| Error::from_reason(format!("Failed to read storage data: {e}")))?;
            return Ok(buf);
        }
    }

    Err(Error::from_reason(format!(
        "Storage key '{storage_key}' not found in archive. Available: {:?}",
        names
    )))
}

// ===========================================================================
// Helpers
// ===========================================================================

fn extract_shape(val: &PickleValue, name: &str) -> Result<Vec<i64>> {
    let items = val
        .as_tuple()
        .or_else(|| val.as_list())
        .ok_or_else(|| Error::from_reason(format!("Tensor '{name}': shape is not a tuple/list")))?;

    items
        .iter()
        .map(|v| {
            v.as_int()
                .ok_or_else(|| Error::from_reason(format!("Tensor '{name}': shape dim is not int")))
        })
        .collect()
}

/// Extract numpy dtype string from a pickle value.
/// The dtype can appear as:
/// - A Global/Object for numpy.dtype with a string arg like '<f4'
/// - A plain string like '<f4' or 'float32'
fn extract_numpy_dtype(val: &PickleValue, name: &str) -> Result<String> {
    // Direct string
    if let Some(s) = val.as_string() {
        return Ok(normalize_numpy_dtype(s));
    }

    // numpy.dtype object: constructed with a string arg
    if let Some((_module, _cls, args, _state)) = val.as_object() {
        for arg in args {
            if let Some(s) = arg.as_string() {
                return Ok(normalize_numpy_dtype(s));
            }
        }
    }

    Err(Error::from_reason(format!(
        "Tensor '{name}': cannot extract numpy dtype"
    )))
}

fn normalize_numpy_dtype(s: &str) -> String {
    match s {
        "<f4" | "float32" | "f4" => "f32".to_string(),
        "<f2" | "float16" | "f2" => "f16".to_string(),
        "<f8" | "float64" | "f8" => "f64".to_string(),
        "<i4" | "int32" | "i4" => "i32".to_string(),
        "<i8" | "int64" | "i8" => "i64".to_string(),
        "<u1" | "uint8" | "u1" => "u8".to_string(),
        "|b1" | "bool" => "bool".to_string(),
        _ => s.to_string(),
    }
}

fn numpy_dtype_size(dtype: &str) -> usize {
    match dtype {
        "f32" | "i32" => 4,
        "f64" | "i64" => 8,
        "f16" | "bf16" => 2,
        "u8" | "i8" | "bool" => 1,
        _ => 4, // default to 4 bytes
    }
}

/// Convert raw bytes + shape + dtype to MxArray
fn bytes_to_mxarray(data: &[u8], shape: &[i64], dtype: &str, name: &str) -> Result<MxArray> {
    match dtype {
        "f32" => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            MxArray::from_float32(&floats, shape)
        }
        "f64" => {
            // Downcast f64 to f32
            let floats: Vec<f32> = data
                .chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect();
            MxArray::from_float32(&floats, shape)
        }
        "f16" => {
            let u16s: Vec<u16> = data
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            MxArray::from_float16(&u16s, shape)
        }
        "bf16" => {
            let u16s: Vec<u16> = data
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            MxArray::from_bfloat16(&u16s, shape)
        }
        "i32" => {
            let ints: Vec<i32> = data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            MxArray::from_int32(&ints, shape)
        }
        "i64" => {
            let ints: Vec<i64> = data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            MxArray::from_int64(&ints, shape)
        }
        _ => Err(Error::from_reason(format!(
            "Tensor '{name}': unsupported dtype '{dtype}'"
        ))),
    }
}

// ===========================================================================
// Paddle .pdiparams (inference format) loader
// ===========================================================================

/// Load Paddle inference `.pdiparams` file (binary LoDTensor format).
///
/// Requires a companion `inference.json` to map parameter names.
/// The `.pdiparams` stores tensors sequentially as:
///   [version:u32][lod_count:u64][proto_size:u32][TensorDesc proto][raw data]
fn load_paddle_inference_params(
    pdiparams_path: &Path,
    json_path: &Path,
    verbose: bool,
) -> Result<HashMap<String, MxArray>> {
    let data = fs::read(pdiparams_path).map_err(|e| {
        Error::from_reason(format!("Failed to read {}: {e}", pdiparams_path.display()))
    })?;

    if verbose {
        eprintln!(
            "Loading Paddle inference params from {}...",
            pdiparams_path.display()
        );
    }

    // Extract parameter names from inference.json
    let mut param_names = extract_param_names_from_json(json_path)?;

    if verbose {
        eprintln!(
            "  Found {} parameter names in inference.json",
            param_names.len()
        );
    }

    // Parse binary .pdiparams: sequence of LoDTensors
    let mut offset = 0;
    let mut tensors_ordered = Vec::new();

    while offset < data.len() {
        // Need at least 4 bytes for version
        if offset + 4 > data.len() {
            break;
        }

        // u32 version (always 0)
        let _version = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        // u64 lod_count (always 0 for params)
        if offset + 8 > data.len() {
            return Err(Error::from_reason(
                "Unexpected EOF reading lod_count".to_string(),
            ));
        }
        let lod_count = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        // Skip LoD levels if any
        for _ in 0..lod_count {
            if offset + 8 > data.len() {
                return Err(Error::from_reason(
                    "Unexpected EOF reading lod level".to_string(),
                ));
            }
            let level_len = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            offset += 8;
            // Each level entry is a u64
            let skip = (level_len as usize)
                .checked_mul(8)
                .ok_or_else(|| Error::from_reason("LoD level size overflow".to_string()))?;
            offset = offset
                .checked_add(skip)
                .ok_or_else(|| Error::from_reason("LoD offset overflow".to_string()))?;
        }

        // u32 version again (tensor desc version)
        if offset + 4 > data.len() {
            return Err(Error::from_reason(
                "Unexpected EOF reading desc version".to_string(),
            ));
        }
        offset += 4;

        // u32 proto_size
        if offset + 4 > data.len() {
            return Err(Error::from_reason(
                "Unexpected EOF reading proto_size".to_string(),
            ));
        }
        let proto_size = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        // TensorDesc protobuf
        if offset + proto_size > data.len() {
            return Err(Error::from_reason(format!(
                "Unexpected EOF reading TensorDesc (need {} bytes at offset {})",
                proto_size, offset
            )));
        }
        let proto_data = &data[offset..offset + proto_size];
        offset += proto_size;

        let (dtype_id, dims) = parse_tensor_desc(proto_data)?;
        let dtype_str = paddle_dtype_to_str(dtype_id);
        let elem_size = numpy_dtype_size(dtype_str);
        let num_elements: usize = dims.iter().map(|&d| d as usize).product();
        let byte_len = num_elements * elem_size;

        // Raw tensor data
        if offset + byte_len > data.len() {
            return Err(Error::from_reason(format!(
                "Unexpected EOF reading tensor data (need {} bytes at offset {})",
                byte_len, offset
            )));
        }
        let tensor_data = &data[offset..offset + byte_len];
        offset += byte_len;

        let array = bytes_to_mxarray(tensor_data, &dims, dtype_str, "pdiparams")?;
        tensors_ordered.push((dims, array));
    }

    if tensors_ordered.len() != param_names.len() {
        return Err(Error::from_reason(format!(
            "Parameter count mismatch: inference.json has {} names but .pdiparams has {} tensors",
            param_names.len(),
            tensors_ordered.len()
        )));
    }

    // The binary stores tensors in alphabetical order of clean names
    // (with _deepcopy_N suffix stripped). Sort names to match.
    param_names.sort_by_key(|a| strip_deepcopy_suffix(a));

    let mut tensors = HashMap::new();
    for (name, (_dims, array)) in param_names.into_iter().zip(tensors_ordered) {
        if verbose {
            let shape = array.shape()?;
            eprintln!("  {name}  shape={:?}", shape.as_ref());
        }
        tensors.insert(name, array);
    }

    Ok(tensors)
}

/// Extract parameter names from Paddle `inference.json`.
///
/// Structure: `program.regions[0].blocks[0].ops[]`
/// Parameter ops have `"#": "p"` and the parameter name is in `A[3]`.
fn extract_param_names_from_json(json_path: &Path) -> Result<Vec<String>> {
    let json_data = fs::read_to_string(json_path)
        .map_err(|e| Error::from_reason(format!("Failed to read {}: {e}", json_path.display())))?;

    let root: serde_json::Value = serde_json::from_str(&json_data)
        .map_err(|e| Error::from_reason(format!("Failed to parse inference.json: {e}")))?;

    // Navigate: program.regions[0].blocks[0].ops
    let ops = root
        .get("program")
        .and_then(|v| v.get("regions"))
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.get("blocks"))
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.get("ops"))
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            Error::from_reason(
                "inference.json: missing program.regions[0].blocks[0].ops".to_string(),
            )
        })?;

    let mut names = Vec::new();
    for op in ops {
        // Check if this is a parameter op
        let is_param = op.get("#").and_then(|v| v.as_str()) == Some("p");

        if is_param {
            // Parameter name is in A[3]
            if let Some(attrs) = op.get("A").and_then(|v| v.as_array())
                && attrs.len() > 3
                && let Some(name) = attrs[3].as_str()
            {
                names.push(name.to_string());
            }
        }
    }

    if names.is_empty() {
        return Err(Error::from_reason(
            "inference.json: no parameter names found (no ops with '#': 'p')".to_string(),
        ));
    }

    Ok(names)
}

/// Read a protobuf varint from data at the given offset.
/// Returns the decoded u64 value and advances offset.
fn read_varint(data: &[u8], offset: &mut usize) -> u64 {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    loop {
        if *offset >= data.len() || shift >= 70 {
            break;
        }
        let byte = data[*offset];
        *offset += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    result
}

/// Parse a minimal VarType::TensorDesc protobuf.
///
/// Fields:
///   field 1 (varint): data_type enum
///   field 2 (varint, repeated): dims
fn parse_tensor_desc(proto: &[u8]) -> Result<(u32, Vec<i64>)> {
    let mut offset = 0;
    let mut dtype: u32 = 5; // default FP32
    let mut dims = Vec::new();

    while offset < proto.len() {
        let tag = read_varint(proto, &mut offset);
        let field_number = tag >> 3;
        let wire_type = tag & 0x7;

        match (field_number, wire_type) {
            (1, 0) => {
                // data_type varint
                dtype = read_varint(proto, &mut offset) as u32;
            }
            (2, 0) => {
                // single dim varint
                dims.push(read_varint(proto, &mut offset) as i64);
            }
            (2, 2) => {
                // packed repeated varint (length-delimited)
                let len = read_varint(proto, &mut offset) as usize;
                let end = offset + len;
                while offset < end {
                    dims.push(read_varint(proto, &mut offset) as i64);
                }
            }
            (_, 0) => {
                // unknown varint field - skip
                read_varint(proto, &mut offset);
            }
            (_, 2) => {
                // unknown length-delimited field - skip
                let len = read_varint(proto, &mut offset) as usize;
                offset += len;
            }
            _ => {
                // unknown wire type - skip
                break;
            }
        }
    }

    if dims.is_empty() {
        // Scalar tensor
        dims.push(1);
    }

    Ok((dtype, dims))
}

/// Find a `.pdparams` file in a directory.
fn find_pdparams_in_dir(dir: &Path) -> Result<PathBuf> {
    for entry in fs::read_dir(dir).map_err(|e| {
        Error::from_reason(format!("Failed to read directory {}: {e}", dir.display()))
    })? {
        let entry = entry.map_err(|e| Error::from_reason(format!("Dir entry error: {e}")))?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "pdparams") {
            return Ok(path);
        }
    }
    Err(Error::from_reason(format!(
        "No .pdparams file found in {}",
        dir.display()
    )))
}

/// Remap Paddle training (.pdparams) parameter keys to our naming convention.
///
/// Training format uses keys like: `conv1._mean`, `fc_0.w_0`, `blocks2.0.dw_conv.weight`
fn remap_training_keys(
    tensors: HashMap<String, MxArray>,
    verbose: bool,
) -> Result<HashMap<String, MxArray>> {
    let mut result = HashMap::new();
    for (key, array) in tensors {
        // Remap Paddle BatchNorm keys
        let mut new_key = key.replace("._mean", ".running_mean");
        new_key = new_key.replace("._variance", ".running_var");

        // Remap classification head
        if new_key == "fc_0.w_0" || new_key == "fc.weight" {
            new_key = "head.weight".to_string();
        } else if new_key == "fc_0.b_0" || new_key == "fc.bias" {
            new_key = "head.bias".to_string();
        } else {
            new_key = format!("backbone.{new_key}");
        }

        if verbose {
            eprintln!("  {key} -> {new_key}");
        }
        result.insert(new_key, array);
    }
    Ok(result)
}

/// Map Paddle data type enum to our dtype string.
fn paddle_dtype_to_str(dtype: u32) -> &'static str {
    match dtype {
        2 => "i32", // INT32
        3 => "i64", // INT64
        4 => "f16", // FP16
        5 => "f32", // FP32
        6 => "f64", // FP64
        _ => "f32",
    }
}

// ===========================================================================
// PP-LCNet inference key remapping
// ===========================================================================

/// Remap Paddle inference parameter names to our SafeTensors naming convention.
///
/// Paddle inference names: `conv2d_0.w_0_deepcopy_0`, `batch_norm2d_3.w_1_deepcopy_42`
/// Our names: `backbone.conv1.conv.weight`, `backbone.blocks3.0.dw_conv.bn.running_mean`
///
/// Note: Inference format uses `conv2d_N` for ALL convolutions (including depthwise).
/// There is no separate `depthwise_conv2d_N` prefix.
fn remap_inference_keys(
    tensors: HashMap<String, MxArray>,
    verbose: bool,
) -> Result<HashMap<String, MxArray>> {
    // Build the mapping tables from PP-LCNet x1.0 architecture
    let conv2d_map = build_conv2d_map();
    let bn_map = build_batch_norm2d_map();

    let mut result = HashMap::new();

    for (raw_key, array) in tensors {
        // Strip _deepcopy_N suffix
        let key = strip_deepcopy_suffix(&raw_key);

        // Parse: "layer_type_N.param_suffix"
        let new_key = if key.starts_with("conv2d_") {
            remap_conv_key(&key, &conv2d_map)?
        } else if key.starts_with("batch_norm2d_") {
            remap_bn_key(&key, &bn_map)?
        } else if key.starts_with("fc_") || key.starts_with("linear_") {
            remap_fc_key(&key)?
        } else {
            // Unknown key - pass through with backbone prefix
            format!("backbone.{key}")
        };

        if verbose {
            eprintln!("  {raw_key} -> {new_key}");
        }
        result.insert(new_key, array);
    }

    Ok(result)
}

/// Strip `_deepcopy_N` suffix from parameter names.
fn strip_deepcopy_suffix(key: &str) -> String {
    if let Some(pos) = key.find("_deepcopy_") {
        key[..pos].to_string()
    } else {
        key.to_string()
    }
}

/// Parse a layer key like "conv2d_5.w_0" into (index, param_suffix).
fn parse_layer_key(key: &str, prefix: &str) -> Result<(usize, String)> {
    let rest = key
        .strip_prefix(prefix)
        .ok_or_else(|| Error::from_reason(format!("Key '{key}' doesn't start with '{prefix}'")))?;

    // rest is like "5.w_0" or "5.b_0"
    let dot_pos = rest.find('.').ok_or_else(|| {
        Error::from_reason(format!("Key '{key}': missing '.' separator after index"))
    })?;

    let idx: usize = rest[..dot_pos]
        .parse()
        .map_err(|_| Error::from_reason(format!("Key '{key}': invalid layer index")))?;

    let param = rest[dot_pos + 1..].to_string();
    Ok((idx, param))
}

/// Remap a conv2d or depthwise_conv2d key using a lookup table.
fn remap_conv_key(key: &str, map: &[(usize, &str)]) -> Result<String> {
    let prefix = if key.starts_with("depthwise_conv2d_") {
        "depthwise_conv2d_"
    } else {
        "conv2d_"
    };

    let (idx, param) = parse_layer_key(key, prefix)?;

    let target_prefix = map
        .iter()
        .find(|(i, _)| *i == idx)
        .map(|(_, p)| *p)
        .ok_or_else(|| {
            Error::from_reason(format!(
                "Unknown {prefix}{idx} - not in PP-LCNet architecture"
            ))
        })?;

    let param_name = match param.as_str() {
        "w_0" => "weight",
        "b_0" => "bias",
        other => {
            return Err(Error::from_reason(format!(
                "Unknown conv param suffix '{other}' in key '{key}'"
            )));
        }
    };

    Ok(format!("{target_prefix}.{param_name}"))
}

/// Remap a batch_norm2d key.
fn remap_bn_key(key: &str, map: &[(usize, &str)]) -> Result<String> {
    let (idx, param) = parse_layer_key(key, "batch_norm2d_")?;

    let target_prefix = map
        .iter()
        .find(|(i, _)| *i == idx)
        .map(|(_, p)| *p)
        .ok_or_else(|| {
            Error::from_reason(format!(
                "Unknown batch_norm2d_{idx} - not in PP-LCNet architecture"
            ))
        })?;

    let param_name = match param.as_str() {
        "w_0" => "weight",
        "b_0" => "bias",
        "w_1" => "running_mean",
        "w_2" => "running_var",
        other => {
            return Err(Error::from_reason(format!(
                "Unknown batch_norm param suffix '{other}' in key '{key}'"
            )));
        }
    };

    Ok(format!("{target_prefix}.{param_name}"))
}

/// Remap FC/linear key to head.weight / head.bias.
fn remap_fc_key(key: &str) -> Result<String> {
    // fc_0.w_0 or linear_0.w_0 -> head.weight
    if key.contains(".w_0") || key.contains(".weight") {
        Ok("head.weight".to_string())
    } else if key.contains(".b_0") || key.contains(".bias") {
        Ok("head.bias".to_string())
    } else {
        Err(Error::from_reason(format!(
            "Unknown FC param in key '{key}'"
        )))
    }
}

/// Build conv2d index -> target prefix mapping for PP-LCNet x1.0.
///
/// In inference format, ALL convolutions (including depthwise) are numbered
/// sequentially as `conv2d_N`. The architecture order interleaves depthwise
/// and pointwise convolutions:
///   0: stem
///   1-2: blocks2[0] (dw, pw)
///   3-6: blocks3[0-1] (dw, pw each)
///   7-10: blocks4[0-1] (dw, pw each)
///   11-22: blocks5[0-5] (dw, pw each)
///   23-26: blocks6[0] (dw, SE conv1, SE conv2, pw)
///   27-30: blocks6[1] (dw, SE conv1, SE conv2, pw)
///   31: last_conv (512 -> 1280 expansion)
fn build_conv2d_map() -> Vec<(usize, &'static str)> {
    vec![
        (0, "backbone.conv1.conv"),
        // blocks2[0]
        (1, "backbone.blocks2.0.dw_conv.conv"),
        (2, "backbone.blocks2.0.pw_conv.conv"),
        // blocks3[0]
        (3, "backbone.blocks3.0.dw_conv.conv"),
        (4, "backbone.blocks3.0.pw_conv.conv"),
        // blocks3[1]
        (5, "backbone.blocks3.1.dw_conv.conv"),
        (6, "backbone.blocks3.1.pw_conv.conv"),
        // blocks4[0]
        (7, "backbone.blocks4.0.dw_conv.conv"),
        (8, "backbone.blocks4.0.pw_conv.conv"),
        // blocks4[1]
        (9, "backbone.blocks4.1.dw_conv.conv"),
        (10, "backbone.blocks4.1.pw_conv.conv"),
        // blocks5[0]
        (11, "backbone.blocks5.0.dw_conv.conv"),
        (12, "backbone.blocks5.0.pw_conv.conv"),
        // blocks5[1]
        (13, "backbone.blocks5.1.dw_conv.conv"),
        (14, "backbone.blocks5.1.pw_conv.conv"),
        // blocks5[2]
        (15, "backbone.blocks5.2.dw_conv.conv"),
        (16, "backbone.blocks5.2.pw_conv.conv"),
        // blocks5[3]
        (17, "backbone.blocks5.3.dw_conv.conv"),
        (18, "backbone.blocks5.3.pw_conv.conv"),
        // blocks5[4]
        (19, "backbone.blocks5.4.dw_conv.conv"),
        (20, "backbone.blocks5.4.pw_conv.conv"),
        // blocks5[5]
        (21, "backbone.blocks5.5.dw_conv.conv"),
        (22, "backbone.blocks5.5.pw_conv.conv"),
        // blocks6[0]: dw + SE + pw
        (23, "backbone.blocks6.0.dw_conv.conv"),
        (24, "backbone.blocks6.0.se.conv1"),
        (25, "backbone.blocks6.0.se.conv2"),
        (26, "backbone.blocks6.0.pw_conv.conv"),
        // blocks6[1]: dw + SE + pw
        (27, "backbone.blocks6.1.dw_conv.conv"),
        (28, "backbone.blocks6.1.se.conv1"),
        (29, "backbone.blocks6.1.se.conv2"),
        (30, "backbone.blocks6.1.pw_conv.conv"),
        // last_conv: 512 -> 1280 expansion before head
        (31, "backbone.last_conv.conv"),
    ]
}

/// Build batch_norm2d index -> target prefix mapping.
///
/// Two BN layers per block (dw + pw), plus one for the stem (27 total).
fn build_batch_norm2d_map() -> Vec<(usize, &'static str)> {
    vec![
        // stem
        (0, "backbone.conv1.bn"),
        // blocks2[0]
        (1, "backbone.blocks2.0.dw_conv.bn"),
        (2, "backbone.blocks2.0.pw_conv.bn"),
        // blocks3[0]
        (3, "backbone.blocks3.0.dw_conv.bn"),
        (4, "backbone.blocks3.0.pw_conv.bn"),
        // blocks3[1]
        (5, "backbone.blocks3.1.dw_conv.bn"),
        (6, "backbone.blocks3.1.pw_conv.bn"),
        // blocks4[0]
        (7, "backbone.blocks4.0.dw_conv.bn"),
        (8, "backbone.blocks4.0.pw_conv.bn"),
        // blocks4[1]
        (9, "backbone.blocks4.1.dw_conv.bn"),
        (10, "backbone.blocks4.1.pw_conv.bn"),
        // blocks5[0]
        (11, "backbone.blocks5.0.dw_conv.bn"),
        (12, "backbone.blocks5.0.pw_conv.bn"),
        // blocks5[1]
        (13, "backbone.blocks5.1.dw_conv.bn"),
        (14, "backbone.blocks5.1.pw_conv.bn"),
        // blocks5[2]
        (15, "backbone.blocks5.2.dw_conv.bn"),
        (16, "backbone.blocks5.2.pw_conv.bn"),
        // blocks5[3]
        (17, "backbone.blocks5.3.dw_conv.bn"),
        (18, "backbone.blocks5.3.pw_conv.bn"),
        // blocks5[4]
        (19, "backbone.blocks5.4.dw_conv.bn"),
        (20, "backbone.blocks5.4.pw_conv.bn"),
        // blocks5[5]
        (21, "backbone.blocks5.5.dw_conv.bn"),
        (22, "backbone.blocks5.5.pw_conv.bn"),
        // blocks6[0]
        (23, "backbone.blocks6.0.dw_conv.bn"),
        (24, "backbone.blocks6.0.pw_conv.bn"),
        // blocks6[1]
        (25, "backbone.blocks6.1.dw_conv.bn"),
        (26, "backbone.blocks6.1.pw_conv.bn"),
    ]
}

// ===========================================================================
// UVDoc inference key remapping
// ===========================================================================

/// Remap Paddle inference parameter names to UVDoc SafeTensors naming.
///
/// Paddle inference names: `conv2d_N.w_0`, `batch_norm2d_N.w_0`, `p_re_lu_N.w_0`
/// Target names follow the PyTorch UVDoc convention used by persistence.rs:
///   `resnet_head.0.weight`, `resnet_down.layer1.0.conv1.0.weight`, etc.
///
/// Architecture order (47 conv2d, 45 batch_norm2d, 2 PReLU):
///   Head:    2 conv + 2 bn
///   Layer 1: 3 blocks (32→32, stride=1, no downsample)    = 6 conv + 6 bn
///   Layer 2: 4 blocks (32→64, stride=1, block0 downsample) = 9 conv + 9 bn
///   Layer 3: 6 blocks (64→128, stride=2, block0 downsample) = 13 conv + 13 bn
///   Bridge:  6 branches (1+1+1+3+3+3) + concat = 13 conv + 13 bn
///   Out 2D:  2 conv + 1 bn + 1 PReLU
///   Out 3D:  2 conv + 1 bn + 1 PReLU (skipped in final output)
fn remap_uvdoc_inference_keys(
    tensors: HashMap<String, MxArray>,
    verbose: bool,
) -> Result<HashMap<String, MxArray>> {
    let conv_map = build_uvdoc_conv2d_map();
    let bn_map = build_uvdoc_batch_norm2d_map();
    let prelu_map = build_uvdoc_prelu_map();

    let mut result = HashMap::new();

    for (raw_key, array) in tensors {
        let key = strip_deepcopy_suffix(&raw_key);

        let new_key = if key.starts_with("conv2d_") {
            remap_uvdoc_conv_key(&key, &conv_map)?
        } else if key.starts_with("batch_norm2d_") {
            remap_uvdoc_bn_key(&key, &bn_map)?
        } else if key.starts_with("p_re_lu_") {
            remap_uvdoc_prelu_key(&key, &prelu_map)?
        } else {
            key.clone()
        };

        // Skip 3D output head (we only need 2D displacement)
        if new_key.starts_with("out_point_positions3D") {
            if verbose {
                eprintln!("  {raw_key} -> {new_key} [SKIPPED - 3D head]");
            }
            continue;
        }

        if verbose {
            eprintln!("  {raw_key} -> {new_key}");
        }
        result.insert(new_key, array);
    }

    Ok(result)
}

/// Remap a UVDoc conv2d key.
fn remap_uvdoc_conv_key(key: &str, map: &[(usize, &str)]) -> Result<String> {
    let (idx, param) = parse_layer_key(key, "conv2d_")?;

    let target_prefix = map
        .iter()
        .find(|(i, _)| *i == idx)
        .map(|(_, p)| *p)
        .ok_or_else(|| {
            Error::from_reason(format!("Unknown conv2d_{idx} - not in UVDoc architecture"))
        })?;

    let param_name = match param.as_str() {
        "w_0" => "weight",
        "b_0" => "bias",
        other => {
            return Err(Error::from_reason(format!(
                "Unknown conv param suffix '{other}' in key '{key}'"
            )));
        }
    };

    Ok(format!("{target_prefix}.{param_name}"))
}

/// Remap a UVDoc batch_norm2d key.
fn remap_uvdoc_bn_key(key: &str, map: &[(usize, &str)]) -> Result<String> {
    let (idx, param) = parse_layer_key(key, "batch_norm2d_")?;

    let target_prefix = map
        .iter()
        .find(|(i, _)| *i == idx)
        .map(|(_, p)| *p)
        .ok_or_else(|| {
            Error::from_reason(format!(
                "Unknown batch_norm2d_{idx} - not in UVDoc architecture"
            ))
        })?;

    let param_name = match param.as_str() {
        "w_0" => "weight",
        "b_0" => "bias",
        "w_1" => "running_mean",
        "w_2" => "running_var",
        other => {
            return Err(Error::from_reason(format!(
                "Unknown batch_norm param suffix '{other}' in key '{key}'"
            )));
        }
    };

    Ok(format!("{target_prefix}.{param_name}"))
}

/// Remap a UVDoc p_re_lu key.
fn remap_uvdoc_prelu_key(key: &str, map: &[(usize, &str)]) -> Result<String> {
    let (idx, param) = parse_layer_key(key, "p_re_lu_")?;

    let target_prefix = map
        .iter()
        .find(|(i, _)| *i == idx)
        .map(|(_, p)| *p)
        .ok_or_else(|| {
            Error::from_reason(format!("Unknown p_re_lu_{idx} - not in UVDoc architecture"))
        })?;

    if param != "w_0" {
        return Err(Error::from_reason(format!(
            "Unknown PReLU param suffix '{param}' in key '{key}'"
        )));
    }

    Ok(format!("{target_prefix}.weight"))
}

/// Build conv2d index → target prefix mapping for UVDoc.
///
/// Architecture construction order (Paddle inference numbering):
///   0-1:   Head (2 conv, no bias)
///   2-7:   Encoder layer1 (3 blocks × 2 conv = 6, with bias)
///   8-16:  Encoder layer2 (4 blocks, block0 has downsample: 3+2+2+2 = 9, with bias)
///   17-29: Encoder layer3 (6 blocks, block0 has downsample: 3+2+2+2+2+2 = 13, with bias)
///   30-42: Bridge (1+1+1+3+3+3+1 = 13 conv, no bias)
///   43-44: Output 2D head (conv1 no bias, conv2 has bias)
///   45-46: Output 3D head (conv1 no bias, conv2 has bias) [skipped]
fn build_uvdoc_conv2d_map() -> Vec<(usize, &'static str)> {
    vec![
        // Head
        (0, "resnet_head.0"),
        (1, "resnet_head.3"),
        // Layer 1 (3 blocks, 32→32, stride=1, no downsample)
        (2, "resnet_down.layer1.0.conv1.0"),
        (3, "resnet_down.layer1.0.conv2.0"),
        (4, "resnet_down.layer1.1.conv1.0"),
        (5, "resnet_down.layer1.1.conv2.0"),
        (6, "resnet_down.layer1.2.conv1.0"),
        (7, "resnet_down.layer1.2.conv2.0"),
        // Layer 2 (4 blocks, 32→64, stride=1, block0 has downsample for channel change)
        // Block 0: conv1, downsample, conv2
        (8, "resnet_down.layer2.0.conv1.0"),
        (9, "resnet_down.layer2.0.downsample.0"),
        (10, "resnet_down.layer2.0.conv2.0"),
        // Blocks 1-3
        (11, "resnet_down.layer2.1.conv1.0"),
        (12, "resnet_down.layer2.1.conv2.0"),
        (13, "resnet_down.layer2.2.conv1.0"),
        (14, "resnet_down.layer2.2.conv2.0"),
        (15, "resnet_down.layer2.3.conv1.0"),
        (16, "resnet_down.layer2.3.conv2.0"),
        // Layer 3 (6 blocks, 64→128, stride=2, block0 has downsample)
        // Block 0: conv1, downsample, conv2
        (17, "resnet_down.layer3.0.conv1.0"),
        (18, "resnet_down.layer3.0.downsample.0"),
        (19, "resnet_down.layer3.0.conv2.0"),
        // Blocks 1-5
        (20, "resnet_down.layer3.1.conv1.0"),
        (21, "resnet_down.layer3.1.conv2.0"),
        (22, "resnet_down.layer3.2.conv1.0"),
        (23, "resnet_down.layer3.2.conv2.0"),
        (24, "resnet_down.layer3.3.conv1.0"),
        (25, "resnet_down.layer3.3.conv2.0"),
        (26, "resnet_down.layer3.4.conv1.0"),
        (27, "resnet_down.layer3.4.conv2.0"),
        (28, "resnet_down.layer3.5.conv1.0"),
        (29, "resnet_down.layer3.5.conv2.0"),
        // Bridge (bridge_1-3 use prefix "bridge_N.0", bridge_4-6 use "bridge_N")
        (30, "bridge_1.0.0.0"),
        (31, "bridge_2.0.0.0"),
        (32, "bridge_3.0.0.0"),
        (33, "bridge_4.0.0"),
        (34, "bridge_4.1.0"),
        (35, "bridge_4.2.0"),
        (36, "bridge_5.0.0"),
        (37, "bridge_5.1.0"),
        (38, "bridge_5.2.0"),
        (39, "bridge_6.0.0"),
        (40, "bridge_6.1.0"),
        (41, "bridge_6.2.0"),
        (42, "bridge_concat.0"),
        // Output 2D
        (43, "out_point_positions2D.0"),
        (44, "out_point_positions2D.3"),
        // Output 3D (skipped during remap)
        (45, "out_point_positions3D.0"),
        (46, "out_point_positions3D.3"),
    ]
}

/// Build batch_norm2d index → target prefix mapping for UVDoc.
///
/// BN layers follow their paired conv in construction order:
///   0-1:   Head (2 bn)
///   2-7:   Layer1 (3 blocks × 2 bn = 6)
///   8-10:  Layer2 block0 (bn1, downsample.1, bn2 = 3)
///   11-16: Layer2 blocks 1-3 (3 × 2 = 6)
///   17-19: Layer3 block0 (bn1, downsample.1, bn2 = 3)
///   20-29: Layer3 blocks 1-5 (5 × 2 = 10)
///   30-42: Bridge (13 bn)
///   43:    Output 2D bn
///   44:    Output 3D bn [skipped]
fn build_uvdoc_batch_norm2d_map() -> Vec<(usize, &'static str)> {
    vec![
        // Head
        (0, "resnet_head.1"),
        (1, "resnet_head.4"),
        // Layer 1
        (2, "resnet_down.layer1.0.bn1"),
        (3, "resnet_down.layer1.0.bn2"),
        (4, "resnet_down.layer1.1.bn1"),
        (5, "resnet_down.layer1.1.bn2"),
        (6, "resnet_down.layer1.2.bn1"),
        (7, "resnet_down.layer1.2.bn2"),
        // Layer 2 block 0 (with downsample)
        (8, "resnet_down.layer2.0.bn1"),
        (9, "resnet_down.layer2.0.downsample.1"),
        (10, "resnet_down.layer2.0.bn2"),
        // Layer 2 blocks 1-3
        (11, "resnet_down.layer2.1.bn1"),
        (12, "resnet_down.layer2.1.bn2"),
        (13, "resnet_down.layer2.2.bn1"),
        (14, "resnet_down.layer2.2.bn2"),
        (15, "resnet_down.layer2.3.bn1"),
        (16, "resnet_down.layer2.3.bn2"),
        // Layer 3 block 0 (with downsample)
        (17, "resnet_down.layer3.0.bn1"),
        (18, "resnet_down.layer3.0.downsample.1"),
        (19, "resnet_down.layer3.0.bn2"),
        // Layer 3 blocks 1-5
        (20, "resnet_down.layer3.1.bn1"),
        (21, "resnet_down.layer3.1.bn2"),
        (22, "resnet_down.layer3.2.bn1"),
        (23, "resnet_down.layer3.2.bn2"),
        (24, "resnet_down.layer3.3.bn1"),
        (25, "resnet_down.layer3.3.bn2"),
        (26, "resnet_down.layer3.4.bn1"),
        (27, "resnet_down.layer3.4.bn2"),
        (28, "resnet_down.layer3.5.bn1"),
        (29, "resnet_down.layer3.5.bn2"),
        // Bridge (bridge_1-3 use prefix "bridge_N.0", bridge_4-6 use "bridge_N")
        (30, "bridge_1.0.0.1"),
        (31, "bridge_2.0.0.1"),
        (32, "bridge_3.0.0.1"),
        (33, "bridge_4.0.1"),
        (34, "bridge_4.1.1"),
        (35, "bridge_4.2.1"),
        (36, "bridge_5.0.1"),
        (37, "bridge_5.1.1"),
        (38, "bridge_5.2.1"),
        (39, "bridge_6.0.1"),
        (40, "bridge_6.1.1"),
        (41, "bridge_6.2.1"),
        (42, "bridge_concat.1"),
        // Output 2D
        (43, "out_point_positions2D.1"),
        // Output 3D (skipped during remap)
        (44, "out_point_positions3D.1"),
    ]
}

/// Build p_re_lu index → target prefix mapping for UVDoc.
fn build_uvdoc_prelu_map() -> Vec<(usize, &'static str)> {
    vec![
        (0, "out_point_positions2D.2"),
        (1, "out_point_positions3D.2"),
    ]
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_numpy_dtype() {
        assert_eq!(normalize_numpy_dtype("<f4"), "f32");
        assert_eq!(normalize_numpy_dtype("float32"), "f32");
        assert_eq!(normalize_numpy_dtype("<f2"), "f16");
        assert_eq!(normalize_numpy_dtype("<i4"), "i32");
        assert_eq!(normalize_numpy_dtype("<i8"), "i64");
    }

    #[test]
    fn test_storage_class_to_dtype() {
        assert_eq!(storage_class_to_dtype("FloatStorage"), Some("f32"));
        assert_eq!(storage_class_to_dtype("HalfStorage"), Some("f16"));
        assert_eq!(storage_class_to_dtype("BFloat16Storage"), Some("bf16"));
        assert_eq!(storage_class_to_dtype("LongStorage"), Some("i64"));
        assert_eq!(storage_class_to_dtype("Unknown"), None);
    }

    #[test]
    fn test_bytes_to_mxarray_f32() {
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let arr = bytes_to_mxarray(&data, &[2, 2], "f32", "test").unwrap();
        arr.eval();
        let result = arr.to_float32().unwrap();
        assert_eq!(&result[..], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_bytes_to_mxarray_i32() {
        let data: Vec<u8> = [10i32, 20, 30]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let arr = bytes_to_mxarray(&data, &[3], "i32", "test").unwrap();
        arr.eval();
        let result = arr.to_int32().unwrap();
        assert_eq!(&result[..], &[10, 20, 30]);
    }

    // ===================================================================
    // .pdiparams format tests
    // ===================================================================

    #[test]
    fn test_read_varint() {
        // Single byte: 0x05 = 5
        let data = [0x05];
        let mut offset = 0;
        assert_eq!(read_varint(&data, &mut offset), 5);
        assert_eq!(offset, 1);

        // Multi-byte: 150 = 0x96 0x01
        let data = [0x96, 0x01];
        let mut offset = 0;
        assert_eq!(read_varint(&data, &mut offset), 150);
        assert_eq!(offset, 2);

        // Larger: 300 = 0xAC 0x02
        let data = [0xAC, 0x02];
        let mut offset = 0;
        assert_eq!(read_varint(&data, &mut offset), 300);
        assert_eq!(offset, 2);
    }

    #[test]
    fn test_parse_tensor_desc() {
        // Build a minimal protobuf: field 1 = 5 (FP32), field 2 = [3, 4]
        // field 1: tag = (1 << 3) | 0 = 0x08, value = 5
        // field 2: tag = (2 << 3) | 0 = 0x10, value = 3
        // field 2: tag = 0x10, value = 4
        let proto = [0x08, 0x05, 0x10, 0x03, 0x10, 0x04];
        let (dtype, dims) = parse_tensor_desc(&proto).unwrap();
        assert_eq!(dtype, 5); // FP32
        assert_eq!(dims, vec![3, 4]);
    }

    #[test]
    fn test_parse_tensor_desc_packed() {
        // Build protobuf with packed repeated field 2
        // field 1: tag=0x08, value=5 (FP32)
        // field 2 packed: tag = (2 << 3) | 2 = 0x12, len=2, values=[3, 4]
        let proto = [0x08, 0x05, 0x12, 0x02, 0x03, 0x04];
        let (dtype, dims) = parse_tensor_desc(&proto).unwrap();
        assert_eq!(dtype, 5);
        assert_eq!(dims, vec![3, 4]);
    }

    #[test]
    fn test_paddle_dtype_to_str() {
        assert_eq!(paddle_dtype_to_str(2), "i32");
        assert_eq!(paddle_dtype_to_str(3), "i64");
        assert_eq!(paddle_dtype_to_str(4), "f16");
        assert_eq!(paddle_dtype_to_str(5), "f32");
        assert_eq!(paddle_dtype_to_str(6), "f64");
        assert_eq!(paddle_dtype_to_str(99), "f32"); // unknown defaults to f32
    }

    #[test]
    fn test_strip_deepcopy_suffix() {
        assert_eq!(
            strip_deepcopy_suffix("conv2d_0.w_0_deepcopy_0"),
            "conv2d_0.w_0"
        );
        assert_eq!(
            strip_deepcopy_suffix("batch_norm2d_3.w_1_deepcopy_42"),
            "batch_norm2d_3.w_1"
        );
        assert_eq!(strip_deepcopy_suffix("conv2d_0.w_0"), "conv2d_0.w_0");
    }

    #[test]
    fn test_remap_conv_key() {
        let map = build_conv2d_map();

        // Stem
        assert_eq!(
            remap_conv_key("conv2d_0.w_0", &map).unwrap(),
            "backbone.conv1.conv.weight"
        );
        // blocks2[0] depthwise (unified numbering)
        assert_eq!(
            remap_conv_key("conv2d_1.w_0", &map).unwrap(),
            "backbone.blocks2.0.dw_conv.conv.weight"
        );
        // blocks2[0] pointwise
        assert_eq!(
            remap_conv_key("conv2d_2.w_0", &map).unwrap(),
            "backbone.blocks2.0.pw_conv.conv.weight"
        );
        // blocks6[0] SE conv1 (with bias)
        assert_eq!(
            remap_conv_key("conv2d_24.w_0", &map).unwrap(),
            "backbone.blocks6.0.se.conv1.weight"
        );
        assert_eq!(
            remap_conv_key("conv2d_24.b_0", &map).unwrap(),
            "backbone.blocks6.0.se.conv1.bias"
        );
        // last_conv
        assert_eq!(
            remap_conv_key("conv2d_31.w_0", &map).unwrap(),
            "backbone.last_conv.conv.weight"
        );
    }

    #[test]
    fn test_remap_bn_key() {
        let map = build_batch_norm2d_map();

        assert_eq!(
            remap_bn_key("batch_norm2d_0.w_0", &map).unwrap(),
            "backbone.conv1.bn.weight"
        );
        assert_eq!(
            remap_bn_key("batch_norm2d_0.w_1", &map).unwrap(),
            "backbone.conv1.bn.running_mean"
        );
        assert_eq!(
            remap_bn_key("batch_norm2d_0.w_2", &map).unwrap(),
            "backbone.conv1.bn.running_var"
        );
        assert_eq!(
            remap_bn_key("batch_norm2d_1.b_0", &map).unwrap(),
            "backbone.blocks2.0.dw_conv.bn.bias"
        );
        assert_eq!(
            remap_bn_key("batch_norm2d_26.w_0", &map).unwrap(),
            "backbone.blocks6.1.pw_conv.bn.weight"
        );
    }

    #[test]
    fn test_remap_fc_key() {
        assert_eq!(remap_fc_key("fc_0.w_0").unwrap(), "head.weight");
        assert_eq!(remap_fc_key("fc_0.b_0").unwrap(), "head.bias");
        assert_eq!(remap_fc_key("linear_0.w_0").unwrap(), "head.weight");
        assert_eq!(remap_fc_key("linear_0.b_0").unwrap(), "head.bias");
    }

    #[test]
    fn test_mapping_table_sizes() {
        // PP-LCNet x1.0 inference format has exactly:
        // 32 conv2d layers (1 stem + 13 dw + 13 pw + 4 SE + 1 last_conv)
        assert_eq!(build_conv2d_map().len(), 32);
        // 27 batch_norm2d layers (1 stem + 2*13 blocks)
        assert_eq!(build_batch_norm2d_map().len(), 27);
    }

    // ===================================================================
    // UVDoc mapping tests
    // ===================================================================

    #[test]
    fn test_uvdoc_mapping_table_sizes() {
        // UVDoc has 47 conv2d, 45 batch_norm2d, 2 PReLU
        assert_eq!(build_uvdoc_conv2d_map().len(), 47);
        assert_eq!(build_uvdoc_batch_norm2d_map().len(), 45);
        assert_eq!(build_uvdoc_prelu_map().len(), 2);
    }

    #[test]
    fn test_uvdoc_conv_mapping() {
        let map = build_uvdoc_conv2d_map();

        // Head
        assert_eq!(
            remap_uvdoc_conv_key("conv2d_0.w_0", &map).unwrap(),
            "resnet_head.0.weight"
        );
        assert_eq!(
            remap_uvdoc_conv_key("conv2d_1.w_0", &map).unwrap(),
            "resnet_head.3.weight"
        );
        // Layer 1 encoder
        assert_eq!(
            remap_uvdoc_conv_key("conv2d_2.w_0", &map).unwrap(),
            "resnet_down.layer1.0.conv1.0.weight"
        );
        assert_eq!(
            remap_uvdoc_conv_key("conv2d_2.b_0", &map).unwrap(),
            "resnet_down.layer1.0.conv1.0.bias"
        );
        // Layer 2 downsample
        assert_eq!(
            remap_uvdoc_conv_key("conv2d_9.w_0", &map).unwrap(),
            "resnet_down.layer2.0.downsample.0.weight"
        );
        // Bridge (bridge_1 uses extra .0)
        assert_eq!(
            remap_uvdoc_conv_key("conv2d_30.w_0", &map).unwrap(),
            "bridge_1.0.0.0.weight"
        );
        // Bridge (bridge_4 direct)
        assert_eq!(
            remap_uvdoc_conv_key("conv2d_33.w_0", &map).unwrap(),
            "bridge_4.0.0.weight"
        );
        // Bridge concat
        assert_eq!(
            remap_uvdoc_conv_key("conv2d_42.w_0", &map).unwrap(),
            "bridge_concat.0.weight"
        );
        // Output head
        assert_eq!(
            remap_uvdoc_conv_key("conv2d_44.b_0", &map).unwrap(),
            "out_point_positions2D.3.bias"
        );
    }

    #[test]
    fn test_uvdoc_bn_mapping() {
        let map = build_uvdoc_batch_norm2d_map();

        assert_eq!(
            remap_uvdoc_bn_key("batch_norm2d_0.w_0", &map).unwrap(),
            "resnet_head.1.weight"
        );
        assert_eq!(
            remap_uvdoc_bn_key("batch_norm2d_0.w_1", &map).unwrap(),
            "resnet_head.1.running_mean"
        );
        // Downsample BN
        assert_eq!(
            remap_uvdoc_bn_key("batch_norm2d_9.w_0", &map).unwrap(),
            "resnet_down.layer2.0.downsample.1.weight"
        );
        // Bridge BN (bridge_1 extra .0)
        assert_eq!(
            remap_uvdoc_bn_key("batch_norm2d_30.w_0", &map).unwrap(),
            "bridge_1.0.0.1.weight"
        );
    }

    #[test]
    fn test_uvdoc_prelu_mapping() {
        let map = build_uvdoc_prelu_map();

        assert_eq!(
            remap_uvdoc_prelu_key("p_re_lu_0.w_0", &map).unwrap(),
            "out_point_positions2D.2.weight"
        );
        assert_eq!(
            remap_uvdoc_prelu_key("p_re_lu_1.w_0", &map).unwrap(),
            "out_point_positions3D.2.weight"
        );
    }

    #[test]
    fn test_normalize_uvdoc_pytorch_key() {
        // Block 0 conv1 - should add .0.
        assert_eq!(
            normalize_uvdoc_pytorch_key("resnet_down.layer1.0.conv1.weight"),
            "resnet_down.layer1.0.conv1.0.weight"
        );
        // Block 0 conv2 - should add .0.
        assert_eq!(
            normalize_uvdoc_pytorch_key("resnet_down.layer1.0.conv2.bias"),
            "resnet_down.layer1.0.conv2.0.bias"
        );
        // Block 1+ - already has .0., should not double
        assert_eq!(
            normalize_uvdoc_pytorch_key("resnet_down.layer1.1.conv1.0.weight"),
            "resnet_down.layer1.1.conv1.0.weight"
        );
        // Non-encoder keys - should pass through
        assert_eq!(
            normalize_uvdoc_pytorch_key("resnet_head.0.weight"),
            "resnet_head.0.weight"
        );
        // bridge_1/2/3: PyTorch bridge_N.0.X -> bridge_N.0.0.X (extra Sequential nesting)
        assert_eq!(
            normalize_uvdoc_pytorch_key("bridge_1.0.0.weight"),
            "bridge_1.0.0.0.weight"
        );
        assert_eq!(
            normalize_uvdoc_pytorch_key("bridge_2.0.1.weight"),
            "bridge_2.0.0.1.weight"
        );
        // bridge_4+ should pass through (multi-conv branches, already correct)
        assert_eq!(
            normalize_uvdoc_pytorch_key("bridge_4.0.0.weight"),
            "bridge_4.0.0.weight"
        );
    }
}
