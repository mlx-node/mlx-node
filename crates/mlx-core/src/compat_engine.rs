//! WASM parity shim for the native-only `crate::engine` API surface that the
//! browser model families (Qwen3 / Qwen3.5 dense) still reference on wasm.
//!
//! On native, `crate::engine` is the full model-neutral chat engine
//! (`#[cfg(not(target_family = "wasm"))]`). The browser build compiles a small
//! subset of the model families that reach for a handful of shared, wasm-safe
//! helpers (safetensors loading, FP8 dequant, config parsing, `Top2`,
//! `ModelGenerationDefaults`). This module mirrors exactly those items so the
//! wasm build resolves `crate::engine::{persistence, decode, ModelGenerationDefaults}`.
//!
//! KEEP IN SYNC (task #68 wasm-source parity):
//!   - `ModelGenerationDefaults` ← engine/params.rs
//!   - `persistence::*`          ← engine/persistence.rs
//!   - `decode::Top2`            ← engine/decode.rs
//! The bodies are byte-identical copies of the native definitions; any change
//! to the native originals must be reflected here.

#[derive(Debug, Clone, Default)]
pub struct ModelGenerationDefaults {
    pub temperature: Option<f64>,
    pub top_k: Option<i32>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    /// `do_sample` from `generation_config.json`. `Some(false)` selects
    /// greedy/argmax decoding (HuggingFace transformers semantics: when
    /// `do_sample=False`, `temperature` is ignored), mapped here to
    /// `temperature = 0.0`. `Some(true)` / `None` leave sampling untouched.
    pub do_sample: Option<bool>,
    pub eos_token_ids: Vec<u32>,
}

impl ModelGenerationDefaults {
    /// The effective temperature this checkpoint contributes when a request
    /// omits `temperature`. `do_sample == Some(false)` forces greedy
    /// (`Some(0.0)`), overriding any `temperature` in generation_config.json
    /// (HuggingFace transformers: do_sample=False ignores temperature);
    /// otherwise the file's `temperature`.
    pub(crate) fn effective_temperature(&self) -> Option<f64> {
        if self.do_sample == Some(false) {
            Some(0.0)
        } else {
            self.temperature
        }
    }
}

pub mod persistence {
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    use napi::bindgen_prelude::*;
    use serde_json::Value;
    use tracing::info;

    use super::ModelGenerationDefaults;
    use crate::array::{DType, MxArray};
    use crate::utils::safetensors::load_safetensors_lazy;

    /// Native paged-page prewarm is a Metal/mmap optimization; on wasm the
    /// browser streams weights straight to GPU buffers, so this is a no-op.
    pub(crate) fn prewarm_checkpoint_pages(_dir: &Path) {}

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

pub fn parse_generation_defaults(model_dir: &Path) -> ModelGenerationDefaults {
    let mut defaults = ModelGenerationDefaults::default();

    let gen_config_path = model_dir.join("generation_config.json");
    let Ok(text) = fs::read_to_string(&gen_config_path) else {
        return defaults;
    };
    let Ok(val) = serde_json::from_str::<Value>(&text) else {
        return defaults;
    };

    defaults.temperature = val.get("temperature").and_then(Value::as_f64);
    // `try_from` (not `as`) so a malformed out-of-`i32`-range value is dropped
    // rather than silently wrapping into a bogus negative top_k.
    defaults.top_k = val
        .get("top_k")
        .and_then(Value::as_i64)
        .and_then(|v| i32::try_from(v).ok());
    defaults.top_p = val.get("top_p").and_then(Value::as_f64);
    defaults.min_p = val.get("min_p").and_then(Value::as_f64);
    defaults.repetition_penalty = val.get("repetition_penalty").and_then(Value::as_f64);
    defaults.do_sample = val.get("do_sample").and_then(Value::as_bool);

    if let Some(eos) = val.get("eos_token_id") {
        let mut push_id = |id: i64| {
            // `try_from` drops both negatives (a few checkpoints use -1 as a
            // "no token" sentinel) AND ids above u32::MAX, instead of a lossy
            // `as u32` cast that could wrap into an unrelated stop token.
            if let Ok(id) = u32::try_from(id) {
                defaults.eos_token_ids.push(id);
            }
        };
        match eos {
            Value::Number(_) => {
                if let Some(id) = eos.as_i64() {
                    push_id(id);
                }
            }
            Value::Array(arr) => {
                for item in arr {
                    if let Some(id) = item.as_i64() {
                        push_id(id);
                    }
                }
            }
            _ => {}
        }
    }

    defaults
}
}

pub mod decode {
/// Top-2 entries `(id, logit)` of a logits vector — used by the
/// `MLX_MTP_TRACE_LOGITS` diagnostic.
pub(crate) struct Top2 {
    pub top1_id: i32,
    pub top1_logit: f32,
    pub top2_id: i32,
    pub top2_logit: f32,
}
}
