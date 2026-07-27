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
use crate::models::quant_dispatch::{
    SYMMETRIC_ZERO_POINT_KEY, normalize_per_layer_key, parse_symmetric_zero_points,
    select_quantization_block,
};
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

/// Strip the known Qwen3.5 vision-tower wrappers from a checkpoint key.
///
/// Converted mlx-vlm artifacts commonly use `visual.*` or
/// `vision_tower.*`, while official Hugging Face Qwen3.5/3.6 checkpoints use
/// `model.visual.*`. Vision detection and splitting happen before the text
/// sanitizer removes `model.` wrappers, so all layouts must be recognized at
/// this boundary or an official multimodal checkpoint is silently loaded as
/// text-only.
pub(crate) fn strip_qwen35_vision_weight_prefix(name: &str) -> Option<&str> {
    name.strip_prefix("model.vision_tower.")
        .or_else(|| name.strip_prefix("model.visual."))
        .or_else(|| name.strip_prefix("vision_tower."))
        .or_else(|| name.strip_prefix("visual."))
}

#[cfg(test)]
mod qwen35_vision_weight_prefix_tests {
    use super::strip_qwen35_vision_weight_prefix;

    #[test]
    fn recognizes_official_and_converted_vision_layouts() {
        for (key, expected) in [
            (
                "model.visual.blocks.0.attn.qkv.weight",
                "blocks.0.attn.qkv.weight",
            ),
            (
                "model.vision_tower.blocks.0.attn.qkv.weight",
                "blocks.0.attn.qkv.weight",
            ),
            ("visual.patch_embed.proj.weight", "patch_embed.proj.weight"),
            ("vision_tower.pos_embed.weight", "pos_embed.weight"),
        ] {
            assert_eq!(strip_qwen35_vision_weight_prefix(key), Some(expected));
        }
    }

    #[test]
    fn rejects_language_model_and_substring_false_positives() {
        for key in [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "some.model.visual.blocks.0.attn.qkv.weight",
            "visual_encoder.blocks.0.attn.qkv.weight",
        ] {
            assert_eq!(strip_qwen35_vision_weight_prefix(key), None);
        }
    }
}

/// Load all safetensors files from a directory (supports sharded checkpoints).
/// Uses MLX's native mmap-backed lazy loader — arrays are backed by deferred disk
/// reads and data is only materialized on eval. This makes loading near-instant
/// and memory is only allocated when weights are actually used.
///
/// When `load_vision` is true, also loads `vision.safetensors` if present (for VLM models),
/// regardless of whether the main checkpoint is a single file or sharded.
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
        // Rebuild the derived biases BEFORE the media sidecar joins the map, and
        // keep it that way. `SymmetricZeroPoints::for_key` falls back to the
        // top-level default for any key it has no entry for, so a main model
        // imported from Q4_0 (zero point 8) paired with an mmproj imported from
        // Q8_0 (128) would derive every vision bias at the main model's offset.
        // Running afterwards turns a loud missing-`.biases` failure into silent
        // corruption; running first leaves the sidecar to its own guards.
        expand_symmetric_affine_biases(dir, &mut params)?;
        append_vision_safetensors(dir, load_vision, &mut params)?;
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

    // Same deliberate order as the single-file branch above: derive first, then
    // append the media sidecar, so a main/mmproj pair imported from different
    // symmetric ggml formats cannot inherit the wrong zero point.
    expand_symmetric_affine_biases(dir, &mut all_params)?;
    append_vision_safetensors(dir, load_vision, &mut all_params)?;

    Ok(all_params)
}

/// Append the optional media sidecar emitted by converted VLM checkpoints.
///
/// Keep this outside the single-vs-sharded branch so `load_vision=true` has
/// identical semantics for both checkpoint layouts. The sidecar is appended
/// after the language-model weights, matching the historical single-file
/// behavior when a key is present in both files.
fn append_vision_safetensors(
    dir: &Path,
    load_vision: bool,
    params: &mut HashMap<String, MxArray>,
) -> Result<()> {
    if !load_vision {
        return Ok(());
    }

    let vision_path = dir.join("vision.safetensors");
    if !vision_path.exists() {
        return Ok(());
    }

    info!(
        "Loading vision weights from: {} (mmap)",
        vision_path.display()
    );
    let vision_params = load_safetensors_lazy(&vision_path)?;
    info!("Loaded {} vision tensors", vision_params.len());
    params.extend(vision_params);
    Ok(())
}

/// Rebuild the `.biases` companions a symmetric affine checkpoint leaves off
/// disk, so everything downstream sees the historical
/// `(weight, scales, biases)` shape.
///
/// A ggml Q4_0 block is `w = d * (q - 8)` and a Q8_0 block is `w = d * (q - 128)`.
/// MLX affine is `w = scale * q + bias`, so the GGUF import used to write out an
/// array whose every entry was `-Z * scale` — 0.5 bpw of pure redundancy, 681 MB
/// on Gemma-4-12B-QAT, enough to push the import above the GGUF it came from.
/// The converter now records `Z` in `config.json` (see
/// [`SYMMETRIC_ZERO_POINT_KEY`](crate::models::quant_dispatch::SYMMETRIC_ZERO_POINT_KEY))
/// and this rebuilds the array here.
///
/// `scales * -Z` is a `mul_scalar`, which builds the scalar in the array's own
/// dtype and returns an unevaluated node, so a float16 scales array yields a
/// float16 bias array and nothing is materialized until the weight is first
/// used. The reconstruction is bitwise equal to what the converter used to
/// write: `Z` is a power of two, so the product is exact in float16 for every
/// scale short of overflow, and an overflowing scale reaches infinity on both
/// sides alike.
///
/// A checkpoint whose config declares no zero point is untouched — the map comes
/// back with zero insertions and behaves exactly as it did before the field
/// existed. Returns how many companions were rebuilt.
pub(crate) fn expand_symmetric_affine_biases(
    dir: &Path,
    params: &mut HashMap<String, MxArray>,
) -> Result<usize> {
    let Ok(raw_str) = fs::read_to_string(dir.join("config.json")) else {
        return Ok(0);
    };
    // Every checkpoint written before the field existed must take exactly the
    // path it always took, so the strict block parse below — which can reject a
    // malformed `quantization` block that a given family never read — only runs
    // for a file that literally spells the field out.
    if !raw_str.contains(SYMMETRIC_ZERO_POINT_KEY) {
        return Ok(0);
    }
    let Ok(raw) = serde_json::from_str::<Value>(&raw_str) else {
        return Ok(0);
    };
    let quant_cfg = select_quantization_block(&raw)?;
    // The fallback group size only matters for overrides that omit their own,
    // and an affine override that omits it inherits a value this function never
    // reads; 32 is the group every symmetric ggml block format uses.
    let zero_points = parse_symmetric_zero_points(quant_cfg, 32)?;
    if zero_points.is_empty() {
        return Ok(0);
    }

    let mut pending: Vec<(String, i32)> = Vec::new();
    for key in params.keys() {
        let Some(base) = key.strip_suffix(".scales") else {
            continue;
        };
        let Some(zero_point) = zero_points.for_key(&normalize_per_layer_key(base)) else {
            continue;
        };
        if params.contains_key(&format!("{base}.biases")) {
            return Err(Error::from_reason(format!(
                "'{base}' declares a symmetric zero point but the checkpoint also stores \
                 '{base}.biases' — the companion is either derived or stored, never both; \
                 refusing to load contradictory quantization metadata"
            )));
        }
        pending.push((base.to_string(), zero_point));
    }

    for (base, zero_point) in &pending {
        let scales = params
            .get(&format!("{base}.scales"))
            .expect("scales key was just observed");
        let dtype = scales.dtype()?;
        if !matches!(dtype, DType::Float32 | DType::Float16 | DType::BFloat16) {
            return Err(Error::from_reason(format!(
                "'{base}' declares a symmetric zero point but '{base}.scales' is {dtype:?}; an \
                 affine scale must be floating for the derived bias to be its exact negative \
                 multiple"
            )));
        }
        let biases = scales.mul_scalar(-f64::from(*zero_point))?;
        params.insert(format!("{base}.biases"), biases);
    }

    if !pending.is_empty() {
        info!(
            "Rebuilt {} symmetric affine .biases companions from their scales",
            pending.len()
        );
    }
    Ok(pending.len())
}

#[cfg(test)]
mod safetensors_loading_tests {
    use super::*;
    use crate::utils::safetensors::save_safetensors;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_model_dir(label: &str) -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "load_all_safetensors_{label}_{}_{}",
            std::process::id(),
            id
        ));
        fs::create_dir_all(&dir).expect("create temp model dir");
        dir
    }

    fn save_one(path: &Path, key: &str, value: f32) {
        let mut tensors = HashMap::from([(
            key.to_string(),
            MxArray::from_float32(&[value], &[1]).expect("create test tensor"),
        )]);
        save_safetensors(path, &mut tensors, None).expect("save test safetensors");
    }

    #[test]
    fn single_checkpoint_loads_vision_sidecar_only_when_requested() {
        let dir = temp_model_dir("single_vision");
        save_one(&dir.join("model.safetensors"), "text.weight", 1.0);
        save_one(&dir.join("vision.safetensors"), "vision.weight", 2.0);

        let text_only = load_all_safetensors(&dir, false).expect("load text-only checkpoint");
        assert!(text_only.contains_key("text.weight"));
        assert!(!text_only.contains_key("vision.weight"));

        let with_vision = load_all_safetensors(&dir, true).expect("load vision checkpoint");
        assert!(with_vision.contains_key("text.weight"));
        assert!(with_vision.contains_key("vision.weight"));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn sharded_checkpoint_appends_vision_sidecar() {
        let dir = temp_model_dir("sharded_vision");
        save_one(
            &dir.join("model-00001-of-00002.safetensors"),
            "text.0.weight",
            1.0,
        );
        save_one(
            &dir.join("model-00002-of-00002.safetensors"),
            "text.1.weight",
            2.0,
        );
        save_one(&dir.join("vision.safetensors"), "vision.weight", 3.0);

        let text_only = load_all_safetensors(&dir, false).expect("load sharded text checkpoint");
        assert_eq!(text_only.len(), 2);
        assert!(!text_only.contains_key("vision.weight"));

        let with_vision = load_all_safetensors(&dir, true).expect("load sharded vision checkpoint");
        assert!(with_vision.contains_key("text.0.weight"));
        assert!(with_vision.contains_key("text.1.weight"));
        assert!(with_vision.contains_key("vision.weight"));

        fs::remove_dir_all(dir).ok();
    }
}

#[cfg(test)]
mod symmetric_bias_expansion_tests {
    use super::*;
    use crate::models::quant_dispatch::SYMMETRIC_ZERO_POINT_KEY;
    use crate::utils::gguf::derived_symmetric_bias_bits;
    use crate::utils::safetensors::save_safetensors;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// f16 scales spanning subnormal, normal and the largest magnitude whose
    /// `-8 *` product is still finite, so a reconstruction that only works in
    /// the easy middle of the range cannot pass.
    const SCALE_BITS: [u16; 6] = [0x0001, 0x0200, 0x0400, 0x25c0, 0x3c00, 0x6fff];

    fn temp_dir(label: &str) -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "symmetric_bias_{label}_{}_{}",
            std::process::id(),
            id
        ));
        fs::create_dir_all(&dir).expect("create temp model dir");
        dir
    }

    /// A checkpoint holding one affine group at `key`, plus whatever extra
    /// tensors the caller wants, and the given `quantization` config block.
    fn write_checkpoint(dir: &Path, key: &str, extra: &[(String, MxArray)], quant: Value) {
        let n = SCALE_BITS.len() as i64;
        let mut tensors: HashMap<String, MxArray> = HashMap::from([
            (
                format!("{key}.weight"),
                MxArray::from_uint32(&vec![0x1234_5678u32; SCALE_BITS.len() * 4], &[n, 4])
                    .expect("weight"),
            ),
            (
                format!("{key}.scales"),
                MxArray::from_float16(&SCALE_BITS, &[n, 1]).expect("scales"),
            ),
        ]);
        for (k, v) in extra {
            tensors.insert(k.clone(), v.clone());
        }
        save_safetensors(dir.join("model.safetensors"), &mut tensors, None).expect("save");
        fs::write(
            dir.join("config.json"),
            serde_json::to_string(&serde_json::json!({ "quantization": quant })).unwrap(),
        )
        .expect("write config.json");
    }

    fn q4_0_block() -> Value {
        serde_json::json!({
            "bits": 4,
            "group_size": 32,
            "mode": "affine",
            SYMMETRIC_ZERO_POINT_KEY: 8,
        })
    }

    /// The `.biases` companion the pre-symmetric converter would have written.
    fn historical_bias_bits(zero_point: i32) -> Vec<u16> {
        SCALE_BITS
            .iter()
            .map(|&s| derived_symmetric_bias_bits(s, zero_point))
            .collect()
    }

    /// `MxArray` has no `Debug`, so `expect_err` cannot be used directly.
    fn expect_load_error(dir: &Path, label: &str) -> Error {
        match load_all_safetensors(dir, false) {
            Ok(params) => panic!("{label}: loaded {} tensors instead", params.len()),
            Err(e) => e,
        }
    }

    fn bias_bits(params: &HashMap<String, MxArray>, key: &str) -> Vec<u16> {
        params
            .get(&format!("{key}.biases"))
            .unwrap_or_else(|| panic!("{key}.biases was not rebuilt"))
            .to_uint16_native()
            .expect("read f16 bits")
    }

    #[test]
    fn top_level_marker_rebuilds_the_historical_bias_bitwise() {
        let dir = temp_dir("top_level");
        let key = "model.layers.0.mlp.down_proj";
        write_checkpoint(&dir, key, &[], q4_0_block());

        let params = load_all_safetensors(&dir, false).expect("load");
        assert_eq!(bias_bits(&params, key), historical_bias_bits(8));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn config_names_survive_the_wrapper_prefix_mismatch() {
        // config.json writes `language_model.model.layers.N...` while the
        // safetensors key is `model.layers.N...`; both normalize to the same
        // prefix, and a per-tensor entry has to be found across that gap.
        let dir = temp_dir("prefix");
        let key = "model.layers.3.self_attn.q_proj";
        write_checkpoint(
            &dir,
            key,
            &[],
            serde_json::json!({
                "bits": 4,
                "group_size": 32,
                "mode": "affine",
                "language_model.model.layers.3.self_attn.q_proj": q4_0_block(),
            }),
        );

        let params = load_all_safetensors(&dir, false).expect("load");
        assert_eq!(bias_bits(&params, key), historical_bias_bits(8));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn q8_0_marker_uses_its_own_zero_point() {
        let dir = temp_dir("q8");
        let key = "model.layers.0.mlp.up_proj";
        write_checkpoint(
            &dir,
            key,
            &[],
            serde_json::json!({
                "bits": 8,
                "group_size": 32,
                "mode": "affine",
                SYMMETRIC_ZERO_POINT_KEY: 128,
            }),
        );

        let params = load_all_safetensors(&dir, false).expect("load");
        assert_eq!(bias_bits(&params, key), historical_bias_bits(128));
        assert_ne!(
            historical_bias_bits(128),
            historical_bias_bits(8),
            "the two zero points must produce different arrays or this proves nothing"
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn an_unmarked_checkpoint_is_untouched() {
        // Two shapes an unmarked file legitimately has, both of which a
        // "rebuild whenever a bias is missing" rule would corrupt: an affine
        // group that stores its own bias, and an mxfp4 group that has uint8
        // scales and no bias at all.
        let dir = temp_dir("unmarked");
        let key = "model.layers.0.mlp.down_proj";
        let stored = MxArray::from_float16(&[0x1234u16; 6], &[6, 1]).expect("biases");
        let mxfp_scales = MxArray::from_uint8(&[0x7fu8; 6], &[6, 1]).expect("mxfp scales");
        write_checkpoint(
            &dir,
            key,
            &[
                (format!("{key}.biases"), stored),
                (
                    "model.layers.1.mlp.down_proj.scales".to_string(),
                    mxfp_scales,
                ),
            ],
            serde_json::json!({
                "bits": 4,
                "group_size": 32,
                "mode": "affine",
                "language_model.model.layers.1.mlp.down_proj": {
                    "bits": 4, "group_size": 32, "mode": "mxfp4",
                },
            }),
        );

        let mut params = load_all_safetensors(&dir, false).expect("load");
        assert_eq!(
            expand_symmetric_affine_biases(&dir, &mut params).expect("expand"),
            0,
            "a checkpoint with no zero point must have nothing rebuilt"
        );
        assert_eq!(bias_bits(&params, key), vec![0x1234u16; 6]);
        assert!(
            !params.contains_key("model.layers.1.mlp.down_proj.biases"),
            "an mxfp4 group must never be handed a synthesised bias"
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_marked_group_that_also_stores_its_bias_is_rejected() {
        let dir = temp_dir("contradictory");
        let key = "model.layers.0.mlp.down_proj";
        let stored = MxArray::from_float16(&[0x1234u16; 6], &[6, 1]).expect("biases");
        write_checkpoint(
            &dir,
            key,
            &[(format!("{key}.biases"), stored)],
            q4_0_block(),
        );

        let err = expect_load_error(&dir, "must reject a contradictory checkpoint");
        assert!(
            err.reason.contains(key) && err.reason.contains("derived or stored"),
            "unexpected message: {}",
            err.reason
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_non_symmetric_override_shadows_the_top_level_marker() {
        // A K-quant tensor in an otherwise-symmetric file stores its float16
        // super-block scale in `.biases`. Inheriting the top-level zero point
        // would both collide with that companion and misdescribe the tensor.
        let dir = temp_dir("shadow");
        let key = "model.layers.0.mlp.down_proj";
        let kq_scales = MxArray::from_float16(&SCALE_BITS, &[6, 1]).expect("scales");
        let mxfp_scales = MxArray::from_uint8(&[0x7fu8; 6], &[6, 1]).expect("mxfp scales");
        write_checkpoint(
            &dir,
            key,
            &[
                ("model.embed_tokens.scales".to_string(), kq_scales),
                (
                    "model.layers.1.mlp.down_proj.scales".to_string(),
                    mxfp_scales,
                ),
            ],
            serde_json::json!({
                "bits": 4,
                "group_size": 32,
                "mode": "affine",
                SYMMETRIC_ZERO_POINT_KEY: 8,
                "language_model.model.embed_tokens": {
                    "bits": 6, "group_size": 16, "mode": "q6k",
                },
                "language_model.model.layers.1.mlp.down_proj": {
                    "bits": 4, "group_size": 32, "mode": "mxfp4",
                },
            }),
        );

        let params = load_all_safetensors(&dir, false).expect("load");
        assert_eq!(bias_bits(&params, key), historical_bias_bits(8));
        assert!(
            !params.contains_key("model.embed_tokens.biases"),
            "a q6k override must not inherit the top-level zero point"
        );
        assert!(
            !params.contains_key("model.layers.1.mlp.down_proj.biases"),
            "an mxfp4 override must not inherit the top-level zero point"
        );

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn an_illegal_zero_point_is_rejected() {
        for (label, quant) in [
            (
                "wrong value for the bit width",
                serde_json::json!({
                    "bits": 4, "group_size": 32, "mode": "affine",
                    SYMMETRIC_ZERO_POINT_KEY: 128,
                }),
            ),
            (
                "declared on a non-affine mode",
                serde_json::json!({
                    "bits": 6, "group_size": 16, "mode": "q6k",
                    SYMMETRIC_ZERO_POINT_KEY: 32,
                }),
            ),
        ] {
            let dir = temp_dir("illegal");
            write_checkpoint(&dir, "model.layers.0.mlp.down_proj", &[], quant);
            let err = expect_load_error(&dir, label);
            assert!(
                err.reason.contains(SYMMETRIC_ZERO_POINT_KEY),
                "{label}: unexpected message: {}",
                err.reason
            );
            fs::remove_dir_all(dir).ok();
        }
    }
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
