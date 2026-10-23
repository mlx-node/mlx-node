//! Header-only GGUF import plan: every decision the converter and the in-memory
//! loader make before a tensor byte is read (preflight rejects, the t64 set,
//! the final `config.json`, the runtime asset list), and the writer for the
//! non-weight files of an import.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use napi::bindgen_prelude::*;
use tracing::{info, warn};

use super::gguf::{
    GGUF_RUNTIME_ASSET_FILES, GgufFile, GgufMetaValue, KQUANT_TILED_DISK_PROJECTIONS,
    QWEN35_INLINE_MTP_INDEX_METADATA, SourceQuantProfile, apply_gemma4_attention_geometry,
    apply_qwen35_inline_mtp_config, config_ties_word_embeddings, describe_gemma4_kv_head_counts,
    extract_config, gemma4_config_states, gemma4_kv_counts_cover_both_layer_types,
    gemma4_layer_types_from_missing_v, gguf_name_to_hf_for_metadata, is_clef_main_gguf,
    is_gemma4_main_gguf, is_muse_glimmer_dflash_gguf, is_muse_glimmer_main_gguf,
    is_muse_glimmer_mmproj_gguf, k_quant_output_keys, plain_qwen3_quantized_source,
    prepare_muse_secondary_config, preserved_source_quantization, prism_hadamard_config,
    qwen35_inline_mtp_index, qwen35_kquant_dense_fixup_target, source_quantization_profiles,
    symmetric_zero_point, validate_prism_supplied_config, validate_qwen35_standalone_geometry,
    write_embedded_gpt2_tokenizer,
};
use crate::array::DType;
use crate::models::quant_dispatch::{kquant_mode_params, parse_mode_str};

pub(crate) type SourceQuantProfiles = BTreeMap<String, SourceQuantProfile>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum OutputRole {
    Primary,
    Companion { filename: String },
}

impl OutputRole {
    pub(crate) fn from_output_filename(filename: &str) -> Self {
        if filename == "model.safetensors" {
            Self::Primary
        } else {
            Self::Companion {
                filename: filename.to_string(),
            }
        }
    }

    pub(crate) fn filename(&self) -> &str {
        match self {
            Self::Primary => "model.safetensors",
            Self::Companion { filename } => filename,
        }
    }

    pub(crate) fn is_primary(&self) -> bool {
        matches!(self, Self::Primary)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ImportOptions {
    pub dtype: Option<DType>,
    pub import_k_quants: bool,
    pub native_qwen35_layout: bool,
    pub config_source_dir: Option<PathBuf>,
    /// `--quantize`.
    pub quantize: bool,
    /// Any flag that re-quantizes: `--quantize`, `--q-recipe`, `--q-mxfp`, `--imatrix-path`.
    pub requantize: bool,
    pub vlm_key_prefix: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct ImportPlan {
    pub role: OutputRole,
    pub import_k_quants: bool,
    /// Requested layout, or forced by a prism.hadamard header.
    pub native_qwen35_layout: bool,
    pub prism: bool,
    pub tie_word_embeddings: bool,
    pub preserve_dtype_keys: HashSet<String>,
    pub tiled_gguf_names: BTreeSet<String>,
    pub tiled_prefixes: BTreeSet<String>,
    pub config: Option<serde_json::Value>,
    pub clef_head_config: Option<serde_json::Value>,
    pub asset_copies: Vec<(PathBuf, &'static str)>,
    /// An explicit config source makes a failed asset copy an error.
    pub strict_asset_copies: bool,
    /// A Muse mmproj / DFlash companion merges its quantization and DFlash
    /// blocks into the primary `config.json`.
    pub muse_secondary: bool,
}

/// Record header facts that tensor-name mapping reads from the metadata map.
/// Errors are left to [`plan_gguf_import`], which raises them in order.
pub(crate) fn annotate_gguf_header(gguf: &mut GgufFile) {
    if let Ok(Some(index)) = qwen35_inline_mtp_index(gguf) {
        gguf.metadata.insert(
            QWEN35_INLINE_MTP_INDEX_METADATA.to_string(),
            GgufMetaValue::Uint64(index),
        );
    }
}

fn read_config_json(path: &Path) -> Result<serde_json::Value> {
    let data = fs::read_to_string(path)
        .map_err(|e| Error::from_reason(format!("Failed to read config.json: {e}")))?;
    serde_json::from_str(&data)
        .map_err(|e| Error::from_reason(format!("Failed to parse config.json: {e}")))
}

pub(crate) fn plan_gguf_import(
    gguf: &GgufFile,
    input: &Path,
    opts: &ImportOptions,
    role: OutputRole,
) -> Result<ImportPlan> {
    let prism_contract = prism_hadamard_config(gguf)?;
    if prism_contract.is_some() {
        if opts.requantize {
            return Err(Error::from_reason(
                "prism.hadamard sources carry packed PQ2_0 weights that are losslessly repacked, never re-quantized; --quantize / --q-recipe / --q-mxfp / --imatrix-path cannot be combined with a prism.hadamard GGUF. Rejected from the GGUF header before tensor data was loaded.",
            ));
        }
        if opts.vlm_key_prefix {
            return Err(Error::from_reason(
                "prism.hadamard sources are text-only; --vlm-key-prefix remaps weights the runtime contract names cannot follow. Rejected from the GGUF header before tensor data was loaded.",
            ));
        }
        if !role.is_primary() {
            return Err(Error::from_reason(
                "prism.hadamard sources must convert as the primary 'model.safetensors' output: the rotation contract lives in config.json, which only the primary conversion writes. Rejected from the GGUF header before tensor data was loaded.",
            ));
        }
    }

    let import_k_quants = opts.import_k_quants;
    let native_qwen35_layout = opts.native_qwen35_layout || prism_contract.is_some();
    let qwen35_inline_mtp_index = qwen35_inline_mtp_index(gguf)?;
    if let Some(index) = qwen35_inline_mtp_index
        && gguf
            .metadata
            .get(QWEN35_INLINE_MTP_INDEX_METADATA)
            .and_then(GgufMetaValue::as_u64)
            != Some(index)
    {
        return Err(Error::from_reason(
            "GGUF header carries an inline Qwen3.5 MTP block but was planned without annotate_gguf_header; tensor names would map to the wrong decoder layers",
        ));
    }

    if let Some(tensor) = plain_qwen3_quantized_source(gguf) {
        return Err(Error::from_reason(format!(
            "GGUF tensor '{}' uses {}, but the plain Qwen3 runtime is dense-only and cannot consume preserved quantized weights. Use a floating-point Qwen3 source or a Qwen3.5 hybrid checkpoint. Rejected from the GGUF header before tensor data was loaded.",
            tensor.name,
            tensor.tensor_type.name(),
        )));
    }

    if import_k_quants
        && !native_qwen35_layout
        && let Some((tensor, mapped)) = qwen35_kquant_dense_fixup_target(gguf)
    {
        return Err(Error::from_reason(format!(
            "GGUF tensor '{}' ({}) maps to Qwen3.5 linear-attention tensor '{}', which requires a head-axis reinterleave during conversion. --gguf-kquant cannot safely apply that dense transform to packed codes without also reordering the group's .scales/.biases, so this source cannot currently be imported losslessly. Use a floating-point source or a preconverted checkpoint. Rejected from the GGUF header before tensor data was loaded.",
            tensor.name,
            tensor.tensor_type.name(),
            mapped,
        )));
    }

    if import_k_quants
        && opts.requantize
        && gguf
            .tensors
            .iter()
            .any(|t| t.tensor_type.k_quant_format().is_some())
    {
        return Err(Error::from_reason(
            "K-quant GGUF import cannot be combined with re-quantization \
             (--quantize / --q-recipe / --q-mxfp / --imatrix-path): ggml \
             supported K/IQ blocks are losslessly imported and never \
             dequantized. Import them as-is, or drop --gguf-kquant to convert \
             a dequantized copy.",
        ));
    }

    if opts.quantize
        && let Some(tensor) = gguf
            .tensors
            .iter()
            .find(|t| symmetric_zero_point(t.tensor_type).is_some())
    {
        return Err(Error::from_reason(format!(
            "GGUF tensor '{}' is already quantized ({}); convert without --quantize \
             (--q-mxfp requires it). Symmetric ggml blocks are repacked losslessly to \
             MLX affine groups whose '.biases' companion is derived from the scale \
             instead of being stored, and --quantize skips them and then writes a \
             quantization block describing only the tensors it did quantize — \
             dropping the '{}' that derivation needs.",
            tensor.name,
            tensor.tensor_type.name(),
            crate::models::quant_dispatch::SYMMETRIC_ZERO_POINT_KEY,
        )));
    }

    let is_primary = role.is_primary();
    let is_muse_companion =
        is_muse_glimmer_mmproj_gguf(&gguf.metadata) || is_muse_glimmer_dflash_gguf(&gguf.metadata);
    if !is_primary
        && !is_muse_companion
        && let Some((tensor, profile)) = gguf.tensors.iter().find_map(|t| {
            if !import_k_quants && t.tensor_type.k_quant_format().is_some() {
                return None;
            }
            let profile = SourceQuantProfile::for_gguf_type(t.tensor_type)?;
            (profile.symmetric_zero_point.is_some() || profile.requires_explicit_entry())
                .then_some((t, profile))
        })
    {
        return Err(Error::from_reason(format!(
            "GGUF tensor '{}' is {}, which loads as {} only when config.json says so — but this \
             conversion writes '{}', a secondary output that writes no \
             config.json (the primary 'model.safetensors' conversion owns it). The packed bytes \
             would be written and then decoded with the runtime's default geometry. Convert this \
             file as the primary output, or use a source whose tensors are float or plain \
             group-32 affine.",
            tensor.name,
            tensor.tensor_type.name(),
            profile.describe(),
            role.filename(),
        )));
    }

    let gemma4_layer_types = if is_primary && is_gemma4_main_gguf(&gguf.metadata) {
        gemma4_layer_types_from_missing_v(gguf)?
    } else {
        None
    };

    let gguf_dir = input.parent().unwrap_or(Path::new(".")).to_path_buf();
    let asset_dir = opts.config_source_dir.clone().unwrap_or(gguf_dir);
    let src_config = asset_dir.join("config.json");
    let synthesized_config = !src_config.exists();

    if let Some(contract) = &prism_contract
        && src_config.exists()
    {
        validate_prism_supplied_config(&read_config_json(&src_config)?, contract, gguf)?;
    }

    if is_primary && is_muse_glimmer_main_gguf(&gguf.metadata) && synthesized_config {
        return Err(Error::from_reason(
            "Muse-Glimmer GGUF conversion requires the authoritative base-model config and tokenizer assets. Pass --config-dir pointing to meta-models/Muse-Glimmer-30B (or place config.json beside the GGUF); the GGUF header alone does not contain the multimodal token IDs and complete vision layer table.",
        ));
    }

    if is_primary && native_qwen35_layout && (synthesized_config || prism_contract.is_some()) {
        validate_qwen35_standalone_geometry(&gguf.metadata)?;
    }

    if let Some(types) = &gemma4_layer_types
        && synthesized_config
        && !gemma4_kv_counts_cover_both_layer_types(&gguf.metadata, types)
    {
        return Err(Error::from_reason(format!(
            "Gemma4 GGUF omits 'attn_v' on {} of its {} decoder blocks, so its global layers \
             have a different KV head count from its sliding ones — but \
             'gemma4.attention.head_count_kv' does not state both ({}), and there is no \
             config.json to take them from, so the converted config would describe the global \
             layers with the sliding count. Supply the model's config.json with --config-dir.",
            types.iter().filter(|t| *t == "full_attention").count(),
            types.len(),
            describe_gemma4_kv_head_counts(&gguf.metadata),
        )));
    }

    let preserve_dtype_keys: HashSet<String> = gguf
        .tensors
        .iter()
        .filter(|t| t.tensor_type.k_quant_format().is_some())
        .filter_map(|t| gguf_name_to_hf_for_metadata(&t.name, &gguf.metadata))
        .flat_map(|key| -> Vec<String> {
            if import_k_quants {
                k_quant_output_keys(&key).to_vec()
            } else {
                vec![key]
            }
        })
        .collect();

    let clef_head_config = plan_clef_head_config(gguf)?;

    let supplied_config = if is_primary && src_config.exists() {
        Some(read_config_json(&src_config)?)
    } else {
        None
    };
    let tie_word_embeddings = match &supplied_config {
        Some(supplied) => config_ties_word_embeddings(supplied),
        None => config_ties_word_embeddings(&extract_config(&gguf.metadata)),
    };

    let (tiled_gguf_names, tiled_prefixes) =
        if is_primary && import_k_quants && !opts.quantize && prism_contract.is_none() {
            let profiles = source_quantization_profiles(gguf, import_k_quants)?;
            plan_tiled_kquant(gguf, &profiles, tie_word_embeddings)?
        } else {
            (BTreeSet::new(), BTreeSet::new())
        };

    let mut asset_copies = Vec::new();
    let mut config = None;
    if is_primary {
        if opts.config_source_dir.is_some() && !src_config.exists() {
            return Err(Error::from_reason(format!(
                "Authoritative config.json not found in config source directory: {}",
                asset_dir.display()
            )));
        }

        let mut config_json = match supplied_config {
            Some(supplied) => supplied,
            None => {
                let mut synthesized = extract_config(&gguf.metadata);
                apply_qwen35_inline_mtp_config(&mut synthesized, qwen35_inline_mtp_index);
                synthesized
            }
        };

        if let Some(contract) = &prism_contract
            && let Some(obj) = config_json.as_object_mut()
        {
            let header = extract_config(&gguf.metadata);
            if let Some(header_obj) = header.as_object() {
                for (key, value) in header_obj {
                    obj.entry(key.clone()).or_insert_with(|| value.clone());
                }
            }
            obj.insert(
                "prism_hadamard".to_string(),
                serde_json::to_value(contract).map_err(|e| {
                    Error::from_reason(format!("Failed to serialize prism_hadamard config: {e}"))
                })?,
            );
        }

        if is_gemma4_main_gguf(&gguf.metadata) {
            if let Some(dtype) = opts.dtype
                && let Some(text) = config_json
                    .get_mut("text_config")
                    .and_then(serde_json::Value::as_object_mut)
            {
                text.insert(
                    "dtype".to_string(),
                    serde_json::Value::String(config_dtype_name(dtype)?.to_string()),
                );
            }
            if let Some(types) = &gemma4_layer_types {
                if !gemma4_config_states(&config_json, "attention_k_eq_v") {
                    config_json["attention_k_eq_v"] = serde_json::Value::Bool(true);
                }
                if !gemma4_config_states(&config_json, "layer_types") {
                    config_json["layer_types"] = serde_json::Value::Array(
                        types
                            .iter()
                            .cloned()
                            .map(serde_json::Value::String)
                            .collect(),
                    );
                }
            }
            if synthesized_config && let Some(obj) = config_json.as_object_mut() {
                apply_gemma4_attention_geometry(obj, &gguf.metadata, gemma4_layer_types.as_deref());
            }
        }

        if is_muse_glimmer_main_gguf(&gguf.metadata) {
            config_json["muse_glimmer_gguf_rope_layout"] =
                serde_json::Value::String("interleaved".to_string());
        }

        if native_qwen35_layout {
            config_json["qwen35_gguf_gdn_layout"] = serde_json::Value::String("tiled".to_string());
        }

        // Under --quantize the converter writes the block after quantizing.
        if !opts.quantize
            && let Some(quant_obj) =
                preserved_source_quantization(gguf, import_k_quants, &tiled_prefixes)?
        {
            config_json["quantization"] = quant_obj.clone();
            config_json["quantization_config"] = quant_obj;
        }
        config = Some(config_json);

        for &filename in GGUF_RUNTIME_ASSET_FILES {
            let src = asset_dir.join(filename);
            if src.exists() {
                asset_copies.push((src, filename));
            } else if matches!(filename, "tokenizer.json" | "tokenizer_config.json") {
                warn!(
                    "{filename} not found in asset source directory {}. You may need to download it separately.",
                    asset_dir.display()
                );
            }
        }
    }

    Ok(ImportPlan {
        role,
        import_k_quants,
        native_qwen35_layout,
        prism: prism_contract.is_some(),
        tie_word_embeddings,
        preserve_dtype_keys,
        tiled_gguf_names,
        tiled_prefixes,
        config,
        clef_head_config,
        asset_copies,
        strict_asset_copies: opts.config_source_dir.is_some(),
        muse_secondary: !is_primary && is_muse_companion,
    })
}

fn config_dtype_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::Float32 => Ok("float32"),
        DType::Float16 => Ok("float16"),
        DType::BFloat16 => Ok("bfloat16"),
        other => Err(Error::from_reason(format!(
            "Unsupported dtype: {other:?}. Use float32, float16, or bfloat16"
        ))),
    }
}

/// Parse a `--dtype` spelling into the conversion dtype.
pub(crate) fn parse_import_dtype(dtype: &str) -> Result<DType> {
    match dtype {
        "float32" | "f32" => Ok(DType::Float32),
        "float16" | "f16" => Ok(DType::Float16),
        "bfloat16" | "bf16" => Ok(DType::BFloat16),
        other => Err(Error::from_reason(format!(
            "Unsupported dtype: {other}. Use float32, float16, or bfloat16"
        ))),
    }
}

/// The 2-D K-quant linears the import stores in the Tiled64 layout, decided
/// from the header alone: `(GGUF tensor names, config-normalized HF prefixes)`.
/// The disk tiler's own predicates are not reused, so a change on either side
/// shows up as a set mismatch.
pub(crate) fn plan_tiled_kquant(
    gguf: &GgufFile,
    profiles: &SourceQuantProfiles,
    tie_word_embeddings: bool,
) -> Result<(BTreeSet<String>, BTreeSet<String>)> {
    let mut names = BTreeSet::new();
    let mut prefixes = BTreeSet::new();
    for tensor in &gguf.tensors {
        if tensor.dims.len() != 2 {
            continue;
        }
        let Some(mapped) = gguf_name_to_hf_for_metadata(&tensor.name, &gguf.metadata) else {
            continue;
        };
        let Some(prefix) = mapped.strip_suffix(".weight") else {
            continue;
        };
        let last = prefix.rsplit('.').next().unwrap_or(prefix);
        let candidate = if last == "lm_head" {
            !tie_word_embeddings
        } else {
            KQUANT_TILED_DISK_PROJECTIONS.contains(&last)
        };
        if !candidate {
            continue;
        }
        let normalized = super::normalize_override_key(prefix);
        let Some(profile) = profiles.get(&normalized) else {
            continue;
        };
        if !profile.is_k_quant()
            || parse_mode_str(Some(profile.mode))
                .and_then(kquant_mode_params)
                .is_none()
        {
            continue;
        }
        let shape = tensor.mlx_shape();
        let (n, k) = (shape[0], shape[1]);
        if n <= 0 || k <= 0 || n % 64 != 0 || k % 256 != 0 {
            continue;
        }
        names.insert(tensor.name.clone());
        prefixes.insert(normalized);
    }
    Ok((names, prefixes))
}

/// Panics in debug builds when the header t64 plan and the disk tiler disagree
/// on a conversion where both ran over the same keys, or when two GGUF tensors
/// feed one tiled prefix.
pub(crate) fn debug_assert_tiled_plan_matches(
    checked: bool,
    plan: &ImportPlan,
    tiled: &BTreeSet<String>,
) {
    if checked {
        debug_assert_eq!(
            &plan.tiled_prefixes, tiled,
            "header t64 plan (left) differs from tile_kquant_weights_for_disk (right)"
        );
        debug_assert_eq!(
            plan.tiled_gguf_names.len(),
            plan.tiled_prefixes.len(),
            "tiled GGUF tensors {:?} do not map one-to-one onto prefixes {:?}",
            plan.tiled_gguf_names,
            plan.tiled_prefixes
        );
    }
}

/// `joint_head_config.json` of a CLEF GGUF from its header: the decision-head
/// metadata plus the geometry of `decision.field_norm` and the first head FFN.
/// `None` when a geometry tensor is missing; the tensor transform then reports it.
fn plan_clef_head_config(gguf: &GgufFile) -> Result<Option<serde_json::Value>> {
    if !is_clef_main_gguf(&gguf.metadata) {
        return Ok(None);
    }
    if !gguf.tensors.iter().any(|t| t.name.starts_with("dec.")) {
        return Err(Error::from_reason(
            "CLEF GGUF carries no 'dec.*' decision-head tensors".to_string(),
        ));
    }
    let meta_u64 = |key: &str, what: &str| -> Result<u64> {
        gguf.metadata
            .get(key)
            .and_then(GgufMetaValue::as_u64)
            .ok_or_else(|| Error::from_reason(format!("CLEF GGUF is missing '{key}' ({what})")))
    };
    let routing_layers =
        meta_u64("clef.decision.routing_block_count", "routing layer count")? as usize;
    let head_layers = meta_u64("clef.decision.block_count", "head layer count")? as usize;
    let heads = meta_u64("clef.decision.head_count", "head attention count")? as i64;
    let hidden_size = meta_u64("clef.embedding_length", "backbone hidden size")? as i64;
    if routing_layers + head_layers > 64 {
        return Err(Error::from_reason(
            "CLEF GGUF declares more than 64 decision-head blocks".to_string(),
        ));
    }
    let leading_dim = |name: &str| -> Option<i64> {
        gguf.tensors
            .iter()
            .find(|t| t.name == name)
            .and_then(|t| t.mlx_shape().first().copied())
    };
    let (Some(width), Some(feedforward)) = (
        leading_dim("decision.field_norm.weight"),
        leading_dim("dec.blk.0.ffn_up.weight"),
    ) else {
        return Ok(None);
    };
    Ok(Some(serde_json::json!({
        "hidden_size": hidden_size,
        "width": width,
        "routing_layers": routing_layers,
        "layers": head_layers,
        "heads": heads,
        "feedforward": feedforward,
    })))
}

pub(crate) fn write_clef_head_config(out: &Path, head_config: &serde_json::Value) -> Result<()> {
    let head_config_str = serde_json::to_string_pretty(head_config).map_err(|e| {
        Error::from_reason(format!("Failed to serialize joint_head_config.json: {e}"))
    })?;
    fs::write(out.join("joint_head_config.json"), head_config_str)
        .map_err(|e| Error::from_reason(format!("Failed to write joint_head_config.json: {e}")))
}

/// Write every non-weight file of an import into `out`: `joint_head_config.json`,
/// the primary `config.json`, the runtime assets, the embedded tokenizer
/// fallback, then the Muse companion merge into that `config.json`.
pub(crate) fn write_gguf_assets(
    primary: &ImportPlan,
    companion: Option<(&ImportPlan, &GgufFile)>,
    gguf: &GgufFile,
    out: &Path,
) -> Result<()> {
    if !primary.role.is_primary() {
        return Err(Error::from_reason(format!(
            "write_gguf_assets needs the primary plan, got the plan for '{}'",
            primary.role.filename()
        )));
    }
    if let Some(head_config) = &primary.clef_head_config {
        write_clef_head_config(out, head_config)?;
        info!("Wrote CLEF joint_head_config.json");
    }

    let config = primary
        .config
        .as_ref()
        .ok_or_else(|| Error::from_reason("primary GGUF import plan has no config.json"))?;
    let config_str = serde_json::to_string_pretty(config)
        .map_err(|e| Error::from_reason(format!("Failed to serialize config: {e}")))?;
    fs::write(out.join("config.json"), &config_str)
        .map_err(|e| Error::from_reason(format!("Failed to write config.json: {e}")))?;
    info!("Wrote config.json");

    for (src, filename) in &primary.asset_copies {
        let dst = out.join(filename);
        if *src == dst {
            continue;
        }
        if primary.strict_asset_copies {
            fs::copy(src, &dst)
                .map_err(|e| Error::from_reason(format!("Failed to copy {filename}: {e}")))?;
            info!("Copied {filename}");
        } else if let Err(e) = fs::copy(src, &dst) {
            warn!("Failed to copy {filename}: {e}");
        } else {
            info!("Copied {filename}");
        }
    }
    if !out.join("tokenizer.json").exists() && write_embedded_gpt2_tokenizer(&gguf.metadata, out)? {
        info!("Reconstructed tokenizer.json from embedded GGUF GPT-2 metadata");
    }

    if let Some((plan, companion_gguf)) = companion
        && plan.muse_secondary
    {
        let config_path = out.join("config.json");
        if let Some(merged) =
            prepare_muse_secondary_config(&config_path, companion_gguf, plan.import_k_quants)?
        {
            fs::write(&config_path, merged).map_err(|error| {
                Error::from_reason(format!(
                    "Failed to update {}: {error}",
                    config_path.display()
                ))
            })?;
            info!("Merged Muse-Glimmer companion GGUF metadata into config.json");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::utils::gguf::{
        GgufConversionOptions, GgufTensorType, convert_gguf_to_safetensors, parse_gguf,
    };
    use crate::utils::gguf_test_fixtures::{
        IMPORT_FIXTURE_NAMES, ImportFixture, ImportFixtureFamily, build_minimal_gguf,
        fixture_payload, prepare_import_fixture, safetensors_manifest, sha256_hex, t64_prefixes,
        write_import_fixture,
    };

    fn golden_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/utils/testdata/gguf_import_golden")
    }

    fn work_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("mlx-gguf-plan-{tag}-{}", uuid::Uuid::new_v4()))
    }

    fn read_json(path: &Path) -> serde_json::Value {
        serde_json::from_slice(&fs::read(path).unwrap())
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    /// Writes fixture `name` and checks its bytes against the golden's
    /// `fixture.json`, so a builder change cannot pass for a plan change.
    fn checked_fixture(name: &str, dir: &Path) -> ImportFixture {
        let fixture = write_import_fixture(name, dir);
        let expected = read_json(&golden_root().join(name).join("fixture.json"));
        assert_eq!(
            expected["main"].as_str(),
            Some(sha256_hex(&fs::read(&fixture.main).unwrap()).as_str()),
            "{name}: fixture builder drifted from the golden"
        );
        let companion = fixture
            .companion
            .as_ref()
            .map(|p| sha256_hex(&fs::read(p).unwrap()));
        assert_eq!(
            expected["companion"].as_str(),
            companion.as_deref(),
            "{name}: companion fixture drifted"
        );
        fixture
    }

    struct Planned {
        gguf: GgufFile,
        plan: ImportPlan,
        companion: Option<(GgufFile, ImportPlan)>,
    }

    /// The plan native prepare's converter calls make: bf16, K-quant import,
    /// sibling `config.json` authoritative when present, the companion with
    /// the GGUF parent as its config source.
    fn plan_like_prepare(
        main: &Path,
        native_qwen35_layout: bool,
        companion: Option<(&Path, &str)>,
    ) -> Planned {
        let parent = main.parent().unwrap();
        let opts = ImportOptions {
            dtype: Some(DType::BFloat16),
            import_k_quants: true,
            native_qwen35_layout,
            config_source_dir: parent
                .join("config.json")
                .is_file()
                .then(|| parent.to_path_buf()),
            ..Default::default()
        };
        let mut gguf = parse_gguf(main).unwrap();
        annotate_gguf_header(&mut gguf);
        let plan = plan_gguf_import(&gguf, main, &opts, OutputRole::Primary)
            .unwrap_or_else(|e| panic!("plan {}: {}", main.display(), e.reason));
        let companion = companion.map(|(path, filename)| {
            let opts = ImportOptions {
                dtype: Some(DType::BFloat16),
                import_k_quants: true,
                config_source_dir: Some(parent.to_path_buf()),
                ..Default::default()
            };
            let mut gguf = parse_gguf(path).unwrap();
            annotate_gguf_header(&mut gguf);
            let plan = plan_gguf_import(
                &gguf,
                path,
                &opts,
                OutputRole::Companion {
                    filename: filename.to_string(),
                },
            )
            .unwrap_or_else(|e| panic!("plan {}: {}", path.display(), e.reason));
            (gguf, plan)
        });
        Planned {
            gguf,
            plan,
            companion,
        }
    }

    fn companion_filename(family: ImportFixtureFamily) -> &'static str {
        match family {
            ImportFixtureFamily::MuseGlimmer => "draft.safetensors",
            _ => "vision.safetensors",
        }
    }

    fn plan_fixture(fixture: &ImportFixture) -> Planned {
        let native = matches!(
            fixture.family,
            ImportFixtureFamily::Qwen35 | ImportFixtureFamily::Qwen35Moe
        );
        let filename = companion_filename(fixture.family);
        plan_like_prepare(
            &fixture.main,
            native,
            fixture.companion.as_deref().map(|p| (p, filename)),
        )
    }

    fn write_planned_assets(planned: &Planned, out: &Path) {
        fs::create_dir_all(out).unwrap();
        write_gguf_assets(
            &planned.plan,
            planned.companion.as_ref().map(|(g, p)| (p, g)),
            &planned.gguf,
            out,
        )
        .unwrap_or_else(|e| panic!("write_gguf_assets: {}", e.reason));
    }

    /// Regular, non-hidden file names of `dir` that `keep` accepts.
    fn file_names(dir: &Path, keep: impl Fn(&str) -> bool) -> BTreeSet<String> {
        fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap())
            .filter(|e| e.path().is_file())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| !n.starts_with('.') && keep(n))
            .collect()
    }

    fn is_fixture_golden_asset(name: &str) -> bool {
        name != "fixture.json" && name != "t64.json" && !name.ends_with(".manifest.json")
    }

    /// Byte equality of every asset file, and the same file set on both sides.
    fn assert_assets_equal(
        tag: &str,
        golden: &Path,
        out: &Path,
        golden_keep: impl Fn(&str) -> bool,
    ) {
        let expected = file_names(golden, golden_keep);
        let written = file_names(out, |n| !n.ends_with(".safetensors"));
        assert_eq!(written, expected, "{tag}: asset file set");
        for name in &expected {
            let want = fs::read(golden.join(name)).unwrap();
            let got = fs::read(out.join(name)).unwrap();
            assert!(
                got == want,
                "{tag}: {name} differs ({} vs golden {} bytes)\n--- written\n{}\n--- golden\n{}",
                got.len(),
                want.len(),
                String::from_utf8_lossy(&got),
                String::from_utf8_lossy(&want)
            );
        }
    }

    #[tokio::test]
    async fn header_tiled_set_equals_base_golden() {
        let work = work_dir("tiled");
        for name in IMPORT_FIXTURE_NAMES {
            let fixture = checked_fixture(name, &work.join("src").join(name));
            let planned = plan_fixture(&fixture);
            let golden: BTreeSet<String> =
                serde_json::from_value(read_json(&golden_root().join(name).join("t64.json")))
                    .unwrap();
            assert!(!golden.is_empty(), "{name}: golden t64 set is empty");
            assert_eq!(
                planned.plan.tiled_prefixes, golden,
                "{name}: header t64 set vs base golden"
            );
            assert_eq!(
                planned.plan.tiled_gguf_names.len(),
                golden.len(),
                "{name}: tiled GGUF names {:?}",
                planned.plan.tiled_gguf_names
            );
            // The converter's marks are what tile_kquant_weights_for_disk tiled.
            let entry = prepare_import_fixture(&fixture, &work.join("cache").join(name)).await;
            let live = t64_prefixes(&read_json(&entry.join("config.json")));
            crate::utils::gguf::native_gguf_unlease_in_use(&entry);
            assert_eq!(
                planned.plan.tiled_prefixes, live,
                "{name}: header t64 set vs the live disk tiler"
            );
        }
        fs::remove_dir_all(&work).ok();
    }

    #[test]
    fn planned_assets_equal_base_golden() {
        let work = work_dir("assets");
        for name in IMPORT_FIXTURE_NAMES {
            let fixture = checked_fixture(name, &work.join("src").join(name));
            let planned = plan_fixture(&fixture);
            let out = work.join("out").join(name);
            write_planned_assets(&planned, &out);
            assert_assets_equal(
                name,
                &golden_root().join(name),
                &out,
                is_fixture_golden_asset,
            );
        }
        fs::remove_dir_all(&work).ok();
    }

    #[tokio::test]
    async fn converter_output_equals_base_golden() {
        let work = work_dir("convert");
        for name in IMPORT_FIXTURE_NAMES {
            let fixture = checked_fixture(name, &work.join("src").join(name));
            let entry = prepare_import_fixture(&fixture, &work.join("cache").join(name)).await;
            let golden = golden_root().join(name);
            let manifests = file_names(&golden, |n| n.ends_with(".manifest.json"));
            let written = file_names(&entry, |n| n.ends_with(".safetensors"));
            assert_eq!(
                written
                    .iter()
                    .map(|n| n.replace(".safetensors", ".manifest.json"))
                    .collect::<BTreeSet<_>>(),
                manifests,
                "{name}: safetensors file set"
            );
            for manifest in &manifests {
                let stem = manifest.trim_end_matches(".manifest.json");
                let want: BTreeMap<String, serde_json::Value> =
                    serde_json::from_value(read_json(&golden.join(manifest))).unwrap();
                let got: BTreeMap<String, serde_json::Value> = serde_json::from_value(
                    safetensors_manifest(&entry.join(format!("{stem}.safetensors"))),
                )
                .unwrap();
                assert_eq!(
                    got.keys().collect::<Vec<_>>(),
                    want.keys().collect::<Vec<_>>(),
                    "{name}/{stem}: tensor names"
                );
                for (tensor, want) in &want {
                    assert_eq!(&got[tensor], want, "{name}/{stem}: tensor {tensor}");
                }
            }
            assert_assets_equal(name, &golden, &entry, is_fixture_golden_asset);
            let t64: BTreeSet<String> =
                serde_json::from_value(read_json(&golden.join("t64.json"))).unwrap();
            assert_eq!(
                t64_prefixes(&read_json(&entry.join("config.json"))),
                t64,
                "{name}: converter t64 marks"
            );
            crate::utils::gguf::native_gguf_unlease_in_use(&entry);
        }
        fs::remove_dir_all(&work).ok();
    }

    /// The options native prepare passes to the converter for `fixture`.
    fn convert_options(fixture: &ImportFixture, out: &Path) -> GgufConversionOptions {
        let parent = fixture.main.parent().unwrap();
        GgufConversionOptions {
            input_path: fixture.main.to_string_lossy().into_owned(),
            output_dir: out.to_string_lossy().into_owned(),
            config_source_dir: parent
                .join("config.json")
                .is_file()
                .then(|| parent.to_string_lossy().into_owned()),
            dtype: Some("bfloat16".to_string()),
            verbose: None,
            quantize: Some(false),
            quant_bits: None,
            quant_group_size: None,
            quant_mode: None,
            quant_recipe: None,
            imatrix_path: None,
            output_filename: None,
            vlm_key_prefix: None,
            quant_mxfp: None,
            import_k_quants: Some(true),
            native_qwen35_layout: Some(matches!(
                fixture.family,
                ImportFixtureFamily::Qwen35 | ImportFixtureFamily::Qwen35Moe
            )),
        }
    }

    #[cfg(debug_assertions)]
    #[tokio::test]
    async fn converter_tiler_drift_debug_asserts() {
        let work = work_dir("drift");
        let fixture = checked_fixture("qwen35-dense-untied", &work.join("src/untied"));
        let planned = plan_fixture(&fixture);
        let tiled = planned.plan.tiled_prefixes.clone();
        debug_assert_tiled_plan_matches(true, &planned.plan, &tiled);

        let extra = "language_model.model.layers.7.mlp.gate_proj";
        let mut drifted = planned.plan.clone();
        drifted.tiled_prefixes.insert(extra.to_string());
        drifted
            .tiled_gguf_names
            .insert("blk.7.ffn_gate.weight".to_string());
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            debug_assert_tiled_plan_matches(true, &drifted, &tiled)
        }))
        .expect_err("an extra header prefix must trip the check");
        let message = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(message.contains(extra), "panic names the prefix: {message}");
        debug_assert_tiled_plan_matches(false, &drifted, &tiled);

        // --vlm-key-prefix renames `lm_head.*`, so the tiler skips the lm_head
        // the header plan tiles; --quantize skips both sides. The converter
        // must not run the check for either.
        let out = work.join("out/vlm");
        let mut options = convert_options(&fixture, &out);
        options.vlm_key_prefix = Some(true);
        convert_gguf_to_safetensors(options)
            .await
            .unwrap_or_else(|e| panic!("vlm-key-prefix convert: {}", e.reason));
        assert!(
            planned
                .plan
                .tiled_prefixes
                .contains("language_model.model.lm_head")
                && !t64_prefixes(&read_json(&out.join("config.json")))
                    .contains("language_model.model.lm_head"),
            "the --vlm-key-prefix case no longer separates the plan from the tiler"
        );
        let gemma = checked_fixture("gemma4-mmproj", &work.join("src/gemma"));
        let float_main = gemma.main.with_file_name("gemma4-float.gguf");
        let up = fixture_payload(GgufTensorType::F32, 256 * 64, 7);
        let norm = fixture_payload(GgufTensorType::F32, 256, 8);
        fs::write(
            &float_main,
            build_minimal_gguf(
                &[(
                    "general.architecture",
                    GgufMetaValue::String("gemma4".into()),
                )],
                &[
                    ("blk.0.ffn_up.weight", &[256, 64], GgufTensorType::F32, &up),
                    ("blk.0.attn_norm.weight", &[256], GgufTensorType::F32, &norm),
                ],
            ),
        )
        .unwrap();
        let out = work.join("out/quantize");
        let mut options = convert_options(&gemma, &out);
        options.input_path = float_main.to_string_lossy().into_owned();
        options.quantize = Some(true);
        convert_gguf_to_safetensors(options)
            .await
            .unwrap_or_else(|e| panic!("--quantize convert: {}", e.reason));
        assert!(t64_prefixes(&read_json(&out.join("config.json"))).is_empty());
        fs::remove_dir_all(&work).ok();
    }

    /// `MLX_GATE_GOLDEN=<dir>` (real-file goldens) plus `gate1-files.txt` beside
    /// it (or `MLX_GATE_FILES`); `MLX_GATE_ONLY=a,b` picks golden dirs. The
    /// header-only plan of each real GGUF writes the golden assets byte for byte.
    #[test]
    #[ignore = "reads real GGUF headers; set MLX_GATE_GOLDEN"]
    fn real_header_plan_matches_golden_config() {
        let golden_root = PathBuf::from(
            std::env::var_os("MLX_GATE_GOLDEN").expect("set MLX_GATE_GOLDEN=<golden dir>"),
        );
        let files = std::env::var_os("MLX_GATE_FILES")
            .map(PathBuf::from)
            .unwrap_or_else(|| golden_root.parent().unwrap().join("gate1-files.txt"));
        let only: Option<BTreeSet<String>> = std::env::var("MLX_GATE_ONLY")
            .ok()
            .map(|v| v.split(',').map(|s| s.trim().to_string()).collect());
        let work = work_dir("real");
        let mut checked = 0;
        for line in fs::read_to_string(&files).unwrap().lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let cols: Vec<&str> = line.split('\t').collect();
            let (dir, gguf, class, companion) = (cols[0], cols[1], cols[2], cols[3]);
            if only.as_ref().is_some_and(|o| !o.contains(dir)) {
                continue;
            }
            let native = matches!(class, "Qwen35Model" | "Qwen35MoeModel" | "ClefModel");
            let filename = if class == "MuseGlimmerModel" {
                "draft.safetensors"
            } else {
                "vision.safetensors"
            };
            let companion = (companion != "-").then(|| (Path::new(companion), filename));
            let planned = plan_like_prepare(Path::new(gguf), native, companion);
            let out = work.join(dir);
            write_planned_assets(&planned, &out);
            let golden = golden_root.join(dir);
            assert_assets_equal(dir, &golden, &out, |n| {
                !(n.starts_with("manifest") && n.ends_with(".json"))
            });
            println!(
                "PLAN_OK {dir} t64={} files={}",
                planned.plan.tiled_prefixes.len(),
                file_names(&out, |_| true).len()
            );
            checked += 1;
        }
        fs::remove_dir_all(&work).ok();
        assert!(checked > 0, "no golden rows matched");
    }
}
