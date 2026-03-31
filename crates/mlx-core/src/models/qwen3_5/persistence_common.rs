//! Shared persistence utilities for Qwen3.5 Dense and MoE models.
//!
//! Contains functions that are identical between the two model variants:
//! safetensors loading, FP8 dequantization, and config parsing helpers.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use napi::bindgen_prelude::*;
use serde_json::Value;
use tracing::info;

use crate::array::{DType, MxArray};
use crate::utils::safetensors::load_safetensors_lazy;

/// Load all safetensors files from a directory (supports sharded checkpoints).
/// Uses MLX's native mmap-backed lazy loader — arrays are backed by deferred disk
/// reads and data is only materialized on eval. This makes loading near-instant
/// and memory is only allocated when weights are actually used.
///
/// When `load_vision` is true, also loads `vision.safetensors` if present (for VLM models).
pub(crate) fn load_all_safetensors(
    dir: &Path,
    load_vision: bool,
) -> Result<HashMap<String, MxArray>> {
    let single_path = if dir.join("weights.safetensors").exists() {
        Some(dir.join("weights.safetensors"))
    } else if dir.join("model.safetensors").exists() {
        Some(dir.join("model.safetensors"))
    } else {
        None
    };

    if let Some(path) = single_path {
        info!("Loading weights from: {} (mmap)", path.display());
        let mut params = load_safetensors_lazy(&path)?;

        // Also load vision.safetensors if present (VLM models)
        if load_vision {
            let vision_path = dir.join("vision.safetensors");
            if vision_path.exists() {
                info!(
                    "Loading vision weights from: {} (mmap)",
                    vision_path.display()
                );
                let vision_params = load_safetensors_lazy(&vision_path)?;
                info!("Loaded {} vision tensors", vision_params.len());
                params.extend(vision_params);
            }
        }

        return Ok(params);
    }

    let mut shard_files: Vec<std::path::PathBuf> = Vec::new();
    let entries = fs::read_dir(dir)
        .map_err(|e| Error::from_reason(format!("Failed to read model directory: {}", e)))?;

    for entry in entries {
        let entry = entry
            .map_err(|e| Error::from_reason(format!("Failed to read directory entry: {}", e)))?;
        let name = entry.file_name().to_string_lossy().to_string();
        let is_shard = (name.starts_with("model-") || name.starts_with("model.safetensors-"))
            && name.ends_with(".safetensors")
            && name.contains("-of-");
        if is_shard {
            shard_files.push(entry.path());
        }
    }

    if shard_files.is_empty() {
        return Err(Error::from_reason(format!(
            "No safetensors files found in {}",
            dir.display()
        )));
    }

    shard_files.sort();
    info!(
        "Loading {} sharded safetensors files (mmap)",
        shard_files.len()
    );

    let mut all_params: HashMap<String, MxArray> = HashMap::new();
    for shard_path in &shard_files {
        info!("  Loading shard: {} (mmap)", shard_path.display());
        let shard_params = load_safetensors_lazy(shard_path)?;
        all_params.extend(shard_params);
    }

    Ok(all_params)
}

/// FP8 E4M3 block-wise dequantization: weight * scale_inv with block_size=128
///
/// Handles both 2D [out, in] and 1D [n] weights.
/// 1. from_fp8(weight) → target dtype
/// 2. Pad to 128-block alignment
/// 3. Reshape into blocks, multiply by scale_inv
/// 4. Unpad and return as target dtype
pub(crate) fn dequant_fp8(
    weight: &MxArray,
    scale_inv: &MxArray,
    target_dtype: DType,
) -> Result<MxArray> {
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

    let pad_bottom = (bs - (m % bs)) % bs;
    let pad_side = (bs - (n % bs)) % bs;

    let weight = if pad_bottom > 0 || pad_side > 0 {
        weight.pad(&[0, pad_bottom as i32, 0, pad_side as i32], 0.0)?
    } else {
        weight
    };

    let m_padded = m + pad_bottom;
    let n_padded = n + pad_side;
    let weight = weight.reshape(&[
        (m_padded / bs) as i64,
        bs as i64,
        (n_padded / bs) as i64,
        bs as i64,
    ])?;

    let scale = scale_inv.expand_dims(1)?.expand_dims(3)?;
    let weight = weight.mul(&scale)?;

    let weight = weight.reshape(&[m_padded as i64, n_padded as i64])?;
    let weight = if pad_bottom > 0 || pad_side > 0 {
        weight.slice(&[0, 0], &[m as i64, n as i64])?
    } else {
        weight
    };

    weight.astype(target_dtype)
}

/// Dequantize all FP8 weight pairs in-place.
/// Finds all `*weight_scale_inv` keys, dequantizes the corresponding weight,
/// removes scale_inv keys, and replaces weights with dequantized versions.
pub(crate) fn dequant_fp8_weights(
    params: &mut HashMap<String, MxArray>,
    target_dtype: DType,
) -> Result<()> {
    let scale_keys: Vec<String> = params
        .keys()
        .filter(|k| k.ends_with("weight_scale_inv"))
        .cloned()
        .collect();

    if scale_keys.is_empty() {
        return Ok(());
    }

    info!(
        "Dequantizing {} FP8 weight pairs to {:?}",
        scale_keys.len(),
        target_dtype
    );

    for scale_key in scale_keys {
        let weight_key = scale_key.replace("_scale_inv", "");
        let scale_inv = params
            .remove(&scale_key)
            .expect("scale_key must exist in params");
        if let Some(weight) = params.remove(&weight_key) {
            let dequantized = dequant_fp8(&weight, &scale_inv, target_dtype)?;
            // Eval immediately to prevent lazy chain accumulation (OOM with ~31K FP8 pairs)
            dequantized.eval();
            params.insert(weight_key, dequantized);
        }
    }

    Ok(())
}

/// Helper to read an i32 config value, checking `text_config` first, then root.
/// Tries each key in order, returning the first match or the default.
pub(crate) fn get_config_i32(
    raw: &Value,
    text_cfg: Option<&Value>,
    keys: &[&str],
    default: i32,
) -> i32 {
    for key in keys {
        if let Some(tc) = text_cfg
            && let Some(v) = tc[key].as_i64()
        {
            return v as i32;
        }
        if let Some(v) = raw[key].as_i64() {
            return v as i32;
        }
    }
    default
}

/// Helper to read an f64 config value, checking `text_config` first, then root.
pub(crate) fn get_config_f64(
    raw: &Value,
    text_cfg: Option<&Value>,
    keys: &[&str],
    default: f64,
) -> f64 {
    for key in keys {
        if let Some(tc) = text_cfg
            && let Some(v) = tc[key].as_f64()
        {
            return v;
        }
        if let Some(v) = raw[key].as_f64() {
            return v;
        }
    }
    default
}

/// Helper to read a bool config value, checking `text_config` first, then root.
pub(crate) fn get_config_bool(
    raw: &Value,
    text_cfg: Option<&Value>,
    keys: &[&str],
    default: bool,
) -> bool {
    for key in keys {
        if let Some(tc) = text_cfg
            && let Some(v) = tc[key].as_bool()
        {
            return v;
        }
        if let Some(v) = raw[key].as_bool() {
            return v;
        }
    }
    default
}
