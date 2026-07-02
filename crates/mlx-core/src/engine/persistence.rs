//! Shared persistence utilities for Qwen3.5 Dense and MoE models.
//!
//! Contains functions that are identical between the two model variants:
//! safetensors loading, FP8 dequantization, and config parsing helpers.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use napi::bindgen_prelude::*;
use serde_json::Value;
use tracing::{info, warn};

use crate::array::{DType, MxArray};
use crate::engine::params::ModelGenerationDefaults;
use crate::utils::safetensors::load_safetensors_lazy;

/// Whether the Metal-only native paths can run on this host.
///
/// The block-paged custom primitives (`paged_kv_write` / `paged_attention`)
/// and the GDN `fast::metal_kernel` kernels require MLX's Metal backend and
/// throw at runtime without it. On the CUDA/Linux build
/// (`mlx_metal_is_available()` is false) the model constructors leave the
/// paged adapter unset and the GDN dispatch takes the ops path, so every
/// forward falls back to the device-agnostic eager Rust path.
///
/// (The name is historical: this probe originally gated the deleted
/// compiled-C++-forward weight registration; today it is a plain
/// Metal-availability check.)
///
/// The probe is cached: `mlx_metal_is_available()` is a constant per process.
pub(crate) fn compiled_forward_backend_available() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| unsafe { mlx_sys::mlx_metal_is_available() })
}

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

/// The directories a model loader may mmap checkpoint shards from, scanned
/// (non-recursively) by the prewarm. Besides the model dir itself, MTP-capable
/// checkpoints keep the speculative-decode head in a `mtp-drafter/` or `mtp/`
/// subdir, or in a sibling `<name>-mtp/` directory — the layouts probed by
/// `detect_drafter_safetensors` and `mtp_sidecar_candidates`. Every entry is
/// best-effort and simply no-ops where absent, so this list is safe for every
/// family (only Qwen3.5 dense/MoE actually populate the MTP locations).
fn standard_checkpoint_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![dir.to_path_buf(), dir.join("mtp-drafter"), dir.join("mtp")];
    if let (Some(parent), Some(name)) = (dir.parent(), dir.file_name()) {
        dirs.push(parent.join(format!("{}-mtp", name.to_string_lossy())));
    }
    dirs
}

/// Collect every `*.safetensors` directly under each of `dirs` (missing dirs
/// skipped), plus any `extra_files` that actually exist. Sorted and de-duped so
/// a file reachable via two layouts (e.g. `mtp.safetensors` both top-level and
/// as a sidecar candidate) is warmed once.
fn collect_safetensors(dirs: &[PathBuf], extra_files: &[PathBuf]) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::new();
    for d in dirs {
        let Ok(read_dir) = fs::read_dir(d) else {
            continue;
        };
        for entry in read_dir.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                files.push(p);
            }
        }
    }
    // `extra_files` may name not-yet-resolved sidecar candidates; keep only the
    // ones present so a non-existent custom path is silently skipped (no warn).
    for f in extra_files {
        if f.is_file() {
            files.push(f.clone());
        }
    }
    files.sort();
    files.dedup();
    files
}

/// Sequentially read each file on the CPU into a throwaway buffer. Best-effort:
/// open/read errors are logged and ignored.
fn prewarm_files(files: &[PathBuf]) {
    use std::io::Read;

    if files.is_empty() {
        return;
    }
    let start = std::time::Instant::now();
    let mut buf = vec![0u8; 32 << 20];
    let mut total: u64 = 0;
    for p in files {
        match fs::File::open(p) {
            Ok(mut f) => loop {
                match f.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => total += n as u64,
                    Err(e) => {
                        warn!("prewarm read error for {}: {}", p.display(), e);
                        break;
                    }
                }
            },
            Err(e) => warn!("prewarm open error for {}: {}", p.display(), e),
        }
    }
    info!(
        "Pre-warmed {} checkpoint shard(s) ({:.1} GB) into the page cache in {:.1}s",
        files.len(),
        total as f64 / (1u64 << 30) as f64,
        start.elapsed().as_secs_f64(),
    );
}

/// Pre-warm the OS page cache for every checkpoint shard a loader may mmap by
/// reading each `*.safetensors` file sequentially on the CPU. Covers the model
/// dir plus the MTP head layouts ([`standard_checkpoint_dirs`]).
///
/// MLX loads weights as lazy mmap-backed arrays. The first GPU op to touch a
/// cold mmap region page-faults inside a Metal command buffer; on slow storage
/// (e.g. a model served off a USB SSD) that stall can exceed the macOS GPU
/// command-buffer watchdog (~5 s) and abort the process uncatchably with
/// `kIOGPUCommandBufferCallbackErrorTimeout`. A plain CPU read is immune to the
/// GPU watchdog and populates the unified buffer cache the mmap shares, so every
/// subsequent eval (FP8 dequant, weight finalize, materialize) hits resident
/// pages — the in-engine equivalent of a manual `cat model.safetensors >/dev/null`.
/// Routing GPU evals via the CPU *stream* does NOT help: the mmap arrays are
/// created GPU-bound during load, so their eval runs on the GPU regardless of
/// the current default stream. Warming the page cache avoids the stall.
///
/// Best-effort: open/read errors are logged and ignored, so load then proceeds
/// exactly as it would have without pre-warming. Shared across every model
/// family that loads via [`load_all_safetensors`].
pub(crate) fn prewarm_checkpoint_pages(dir: &Path) {
    prewarm_files(&collect_safetensors(&standard_checkpoint_dirs(dir), &[]));
}

/// Like [`prewarm_checkpoint_pages`] but also warms `extra_files` — explicit
/// sidecar paths a loader resolves from config (e.g. a non-standard
/// `mlx_lm_extra_tensors.mtp_file`) that the [`standard_checkpoint_dirs`] scan
/// would not reach. Non-existent entries are skipped.
pub(crate) fn prewarm_checkpoint_pages_with(dir: &Path, extra_files: &[PathBuf]) {
    prewarm_files(&collect_safetensors(
        &standard_checkpoint_dirs(dir),
        extra_files,
    ));
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

/// Read a model's `generation_config.json` into a
/// [`ModelGenerationDefaults`].
///
/// The file is optional: a missing or unparseable file yields
/// `ModelGenerationDefaults::default()` (all fields empty), so callers can
/// apply it unconditionally as a no-op. Sampling fields
/// (`temperature`/`top_k`/`top_p`/`min_p`/`repetition_penalty`) are read
/// only when present and well-typed; an absent field stays `None`.
///
/// `do_sample` is read as a boolean; `false` maps to greedy decoding
/// (`temperature = 0`) when a request omits `temperature`, applied in
/// [`crate::engine::params::apply_generation_defaults`].
///
/// `eos_token_id` is read as either a scalar integer (-> one id) or an
/// array of integers (-> each id). Negative values are dropped (a few
/// checkpoints use `-1` as a "no token" sentinel) and the rest are cast to
/// `u32`. Other keys (`bos_token_id`, `pad_token_id`,
/// `transformers_version`, …) are ignored.
///
/// Never panics on malformed input.
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

#[cfg(test)]
mod prewarm_tests {
    use super::*;

    fn touch(p: &Path) {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).expect("mkdir");
        }
        fs::write(p, b"").expect("touch");
    }

    // The set of files we warm MUST cover every safetensors location a loader
    // can later mmap — the model dir AND the MTP head layouts (`mtp-drafter/`,
    // `mtp/`, sibling `<name>-mtp/`) plus an explicit non-standard sidecar
    // passed as an `extra_file`. Missing any of these re-opens the cold-mmap
    // GPU-watchdog hole the prewarm exists to close.
    #[test]
    fn collect_safetensors_covers_mtp_sidecar_and_drafter_layouts() {
        let root = std::env::temp_dir().join(format!("prewarm_cover_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let model = root.join("my-model");

        let top = model.join("model.safetensors");
        let mtp_subdir = model.join("mtp").join("weights.safetensors");
        let drafter = model.join("mtp-drafter").join("model.safetensors");
        let sibling = root.join("my-model-mtp").join("model.safetensors");
        let custom = model.join("custom").join("mtp-sidecar.safetensors");
        for p in [&top, &mtp_subdir, &drafter, &sibling, &custom] {
            touch(p);
        }
        // A non-existent extra candidate (e.g. an unmatched sidecar name) and a
        // non-safetensors file must both be excluded.
        let absent = model.join("nope.safetensors");
        touch(&model.join("config.json"));

        let found = collect_safetensors(
            &standard_checkpoint_dirs(&model),
            &[custom.clone(), absent.clone()],
        );

        for p in [&top, &mtp_subdir, &drafter, &sibling, &custom] {
            assert!(found.contains(p), "prewarm set missing {}", p.display());
        }
        assert!(!found.contains(&absent), "non-existent extra leaked in");
        assert!(
            !found.iter().any(|p| p.ends_with("config.json")),
            "non-safetensors file leaked in"
        );

        // De-dup: a path reachable both via the dir scan and as an explicit
        // extra appears exactly once.
        let with_dup = collect_safetensors(
            &standard_checkpoint_dirs(&model),
            std::slice::from_ref(&top),
        );
        assert_eq!(
            with_dup.iter().filter(|p| **p == top).count(),
            1,
            "top-level shard duplicated"
        );

        let _ = fs::remove_dir_all(&root);
    }
}

#[cfg(test)]
mod generation_defaults_tests {
    use super::*;

    /// Write a `generation_config.json` with the given body into a fresh temp
    /// dir and return the dir (kept alive by the returned `PathBuf` root).
    fn write_gen_config(body: &str) -> PathBuf {
        let mut root = std::env::temp_dir();
        root.push(format!(
            "mlx_gen_defaults_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&root).expect("create temp dir");
        fs::write(root.join("generation_config.json"), body).expect("write gen config");
        root
    }

    #[test]
    fn missing_file_yields_default() {
        let mut root = std::env::temp_dir();
        root.push(format!("mlx_gen_defaults_missing_{}", std::process::id()));
        // Do NOT create the file.
        let d = parse_generation_defaults(&root);
        assert!(d.temperature.is_none());
        assert!(d.top_k.is_none());
        assert!(d.top_p.is_none());
        assert!(d.min_p.is_none());
        assert!(d.repetition_penalty.is_none());
        assert!(d.eos_token_ids.is_empty());
    }

    #[test]
    fn unparseable_file_yields_default() {
        let root = write_gen_config("{ this is not valid json ");
        let d = parse_generation_defaults(&root);
        assert!(d.temperature.is_none());
        assert!(d.eos_token_ids.is_empty());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn scalar_eos_token_id_becomes_single_vec() {
        let root = write_gen_config(r#"{"eos_token_id": 151645}"#);
        let d = parse_generation_defaults(&root);
        assert_eq!(d.eos_token_ids, vec![151645]);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn array_eos_token_id_becomes_vec() {
        let root = write_gen_config(r#"{"eos_token_id": [151645, 151643, 7]}"#);
        let d = parse_generation_defaults(&root);
        assert_eq!(d.eos_token_ids, vec![151645, 151643, 7]);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn out_of_range_ints_are_dropped() {
        // top_k above i32::MAX and an eos id above u32::MAX must be DROPPED
        // (try_from), never wrapped via a lossy cast. In-range values survive.
        let root = write_gen_config(r#"{"top_k": 5000000000, "eos_token_id": [5000000000, 42]}"#);
        let d = parse_generation_defaults(&root);
        assert!(
            d.top_k.is_none(),
            "out-of-i32-range top_k dropped, not wrapped"
        );
        assert_eq!(d.eos_token_ids, vec![42], "out-of-u32-range eos id dropped");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn negative_eos_ids_are_filtered() {
        let root = write_gen_config(r#"{"eos_token_id": [-1, 5, -42, 9]}"#);
        let d = parse_generation_defaults(&root);
        assert_eq!(d.eos_token_ids, vec![5, 9]);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn sampling_fields_parsed_when_present() {
        let root = write_gen_config(
            r#"{"temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.05,
                "repetition_penalty": 1.1, "do_sample": true, "bos_token_id": 1}"#,
        );
        let d = parse_generation_defaults(&root);
        assert_eq!(d.temperature, Some(0.6));
        assert_eq!(d.top_k, Some(20));
        assert_eq!(d.top_p, Some(0.95));
        assert_eq!(d.min_p, Some(0.05));
        assert_eq!(d.repetition_penalty, Some(1.1));
        assert_eq!(d.do_sample, Some(true));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn do_sample_false_is_parsed() {
        let root = write_gen_config(r#"{"do_sample": false, "temperature": 0.7}"#);
        let d = parse_generation_defaults(&root);
        assert_eq!(d.do_sample, Some(false));
        assert_eq!(d.temperature, Some(0.7));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn absent_sampling_field_stays_none() {
        // Only top_p present; the others must stay None.
        let root = write_gen_config(r#"{"top_p": 0.9}"#);
        let d = parse_generation_defaults(&root);
        assert_eq!(d.top_p, Some(0.9));
        assert!(d.temperature.is_none());
        assert!(d.top_k.is_none());
        assert!(d.min_p.is_none());
        assert!(d.repetition_penalty.is_none());
        assert!(d.do_sample.is_none());
        let _ = fs::remove_dir_all(&root);
    }
}
