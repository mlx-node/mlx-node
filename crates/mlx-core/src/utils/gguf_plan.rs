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
    GGUF_RUNTIME_ASSET_FILES, GgufFile, GgufMetaValue, GgufTensorType,
    KQUANT_TILED_DISK_PROJECTIONS, QWEN35_INLINE_MTP_INDEX_METADATA, SourceQuantProfile,
    apply_gemma4_attention_geometry, apply_qwen35_inline_mtp_config, config_ties_word_embeddings,
    describe_gemma4_kv_head_counts, extract_config, gemma4_config_states,
    gemma4_kv_counts_cover_both_layer_types, gemma4_layer_types_from_missing_v,
    gguf_name_to_hf_for_metadata, is_clef_main_gguf, is_gemma4_main_gguf, is_inkling2_mmproj_gguf,
    is_muse_glimmer_dflash_gguf, is_muse_glimmer_main_gguf, is_muse_glimmer_mmproj_gguf,
    k_quant_output_keys, plain_qwen3_quantized_source, prepare_muse_secondary_config,
    preserved_source_quantization, prism_hadamard_config, qwen35_inline_mtp_index,
    qwen35_kquant_dense_fixup_target, qwen35_mxfp8_dense_fixup_target,
    source_quantization_profiles, symmetric_zero_point, validate_prism_supplied_config,
    validate_qwen35_standalone_geometry, write_embedded_gpt2_tokenizer,
};
use super::gguf_source::reject_split;
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
    /// Dense GGUF tensor the staged read allocates as row shards of this many
    /// rows instead of one buffer (Gemma4 PLE above the shard budget).
    pub ple_shard: Option<(String, usize)>,
    /// Stop ids from the header that join `generation_config.json`
    /// `eos_token_id`: Splash `flint_stop_tokens` and the Qwen `.stopTokens`.
    pub stop_token_ids: Vec<u32>,
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
    reject_split(gguf, input)?;
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

    if !native_qwen35_layout && let Some((tensor, mapped)) = qwen35_mxfp8_dense_fixup_target(gguf) {
        return Err(Error::from_reason(format!(
            "GGUF tensor '{}' ({}) maps to Qwen3.5 linear-attention tensor '{}', which requires a head-axis reinterleave during conversion. Packed MXFP8 import cannot safely apply that dense transform to packed codes without also reordering the group's .scales, so this source cannot currently be imported losslessly. Use a floating-point source or a native-layout checkpoint. Rejected from the GGUF header before tensor data was loaded.",
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

    if opts.quantize
        && let Some(tensor) = gguf.tensors.iter().find(|t| {
            t.tensor_type.is_mlx_mxfp8_quantized() && !super::gguf::is_gather_only_mxfp8(t)
        })
    {
        return Err(Error::from_reason(format!(
            "GGUF tensor '{}' is already quantized ({}); convert without --quantize. \
             MXFP8 tensors are repacked losslessly as packed MLX MXFP8 groups, and --quantize \
             skips already-packed tensors while overwriting config.json's quantization block \
             with only newly quantized tensors — dropping the explicit mxfp8 profile.",
            tensor.name,
            tensor.tensor_type.name(),
        )));
    }

    let is_primary = role.is_primary();
    let is_muse_companion =
        is_muse_glimmer_mmproj_gguf(&gguf.metadata) || is_muse_glimmer_dflash_gguf(&gguf.metadata);
    let is_inkling2_mmproj = is_inkling2_mmproj_gguf(&gguf.metadata);
    if !is_primary && is_inkling2_mmproj {
        // Inkling2 vision projector companions (`clip.vision.projector_type = "inkling2v"`)
        // accept dense F32/F16/BF16 tensors plus 2D+ Q8_0 matrices whose inner dimension is a
        // multiple of 32 (dequantized directly to dense BF16 during import, matching Splash
        // `InklingVision.cpp:226-231` / `VisionPreparation.cpp:33-43`). Reject all other
        // quantized formats or unaligned/1D Q8_0 tensors so callers can fall back to text-only.
        if let Some(tensor) = gguf.tensors.iter().find(|t| match t.tensor_type {
            GgufTensorType::F32 | GgufTensorType::F16 | GgufTensorType::BF16 => false,
            GgufTensorType::Q8_0 => {
                let cols = t.dims.first().copied().unwrap_or(0);
                t.dims.len() < 2 || cols == 0 || !cols.is_multiple_of(32)
            }
            _ => true,
        }) {
            return Err(Error::from_reason(format!(
                "Inkling2 vision projector tensor '{}' uses unsupported type {} with dims {:?}; \
                 only F32, F16, BF16, and 2D+ Q8_0 (inner dim multiple of 32) are supported",
                tensor.name,
                tensor.tensor_type.name(),
                tensor.dims,
            )));
        }
    } else if !is_primary
        && !is_muse_companion
        && let Some((tensor, profile)) = gguf.tensors.iter().find_map(|t| {
            if super::gguf::is_gather_only_mxfp8(t) {
                return None;
            }
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

    let mut preserve_dtype_keys: HashSet<String> = gguf
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
    if is_inkling2_mmproj {
        // Preserve dense F32 tensors in Inkling2 mmproj companions (1D norms, biases,
        // LayerScales, class/position/patch embeddings, and full mmproj-F32.gguf weights)
        // so the companion `--dtype bf16` pass never truncates them before ViT execution.
        for t in &gguf.tensors {
            if t.tensor_type == GgufTensorType::F32
                && let Some(key) = gguf_name_to_hf_for_metadata(&t.name, &gguf.metadata)
            {
                preserve_dtype_keys.insert(key);
            }
        }
    }

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

    let stop_token_ids = if is_primary && is_qwen35_hybrid_gguf(gguf) {
        refuse_unread_qwen35_tensors(gguf, qwen35_inline_mtp_index)?;
        plan_stop_token_ids(gguf)?
    } else {
        Vec::new()
    };

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
        ple_shard: None,
        stop_token_ids,
    })
}

/// Splash `Problems`: each fault named once, counted over the layers it is in.
#[derive(Default)]
struct UnreadTensors {
    faults: Vec<(String, &'static str, u32)>,
}

impl UnreadTensors {
    fn add(&mut self, name: &str, fault: &'static str) {
        let layered = name
            .strip_prefix("blk.")
            .and_then(|rest| rest.find('.').map(|dot| &rest[dot + 1..]));
        let tensor = layered.unwrap_or(name);
        let at = match self
            .faults
            .iter()
            .position(|(t, f, _)| t == tensor && *f == fault)
        {
            Some(at) => at,
            None => {
                self.faults.push((tensor.to_string(), fault, 0));
                self.faults.len() - 1
            }
        };
        self.faults[at].2 += u32::from(layered.is_some());
    }

    fn text(&self) -> String {
        self.faults
            .iter()
            .map(|(tensor, fault, layers)| match layers {
                0 => format!("{tensor} ({fault})"),
                1 => format!("{tensor} ({fault}) in 1 layer"),
                n => format!("{tensor} ({fault}) in {n} layers"),
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Block `N` and the rest of a `blk.N.REST` name (Splash `Problems::inBlocks`).
fn block_of(name: &str) -> Option<(u64, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    let tail = rest[digits..].strip_prefix('.')?;
    if digits == 0 {
        return None;
    }
    Some((rest[..digits].parse().ok()?, tail))
}

const QWEN35_LAYER_TENSORS: [&str; 2] = ["attn_norm.weight", "post_attention_norm.weight"];
const QWEN35_FULL_ATTENTION_TENSORS: [&str; 6] = [
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "attn_output.weight",
];
const QWEN35_GDN_TENSORS: [&str; 9] = [
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_beta.weight",
    "ssm_alpha.weight",
    "ssm_conv1d.weight",
    "ssm_a",
    "ssm_dt.bias",
    "ssm_norm.weight",
    "ssm_out.weight",
];
const QWEN35_DENSE_FFN_TENSORS: [&str; 3] = ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"];
const QWEN35_MOE_FFN_TENSORS: [&str; 8] = [
    "ffn_gate_inp.weight",
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
    "ffn_gate_inp_shexp.weight",
];
const QWEN35_GLOBAL_TENSORS: [&str; 3] =
    ["token_embd.weight", "output.weight", "output_norm.weight"];

/// Splash `familyImages<HybridGguf>` then `Problems::addUnread`: every header
/// tensor is one the qwen35 / qwen35moe loader reads, or one of the
/// `nextn_predict_layers` blocks after the decoder layers; any other tensor
/// (an attention or router bias among them) would change what the model
/// computes, so the GGUF is refused before a weight byte is read.
pub(crate) fn refuse_unread_qwen35_tensors(gguf: &GgufFile, inline_mtp: Option<u64>) -> Result<()> {
    let arch = gguf
        .metadata
        .get("general.architecture")
        .and_then(GgufMetaValue::as_str)
        .unwrap_or_default();
    let key = |name: &str| format!("{arch}.{name}");
    let required = |name: &str| {
        gguf.metadata
            .get(&key(name))
            .and_then(GgufMetaValue::as_u64)
            .ok_or_else(|| Error::from_reason(format!("GGUF metadata '{}' is missing", key(name))))
    };
    let block_count = required("block_count")?;
    let interval = required("full_attention_interval")?;
    let nextn = gguf
        .metadata
        .get(&key("nextn_predict_layers"))
        .and_then(GgufMetaValue::as_u64)
        .unwrap_or(0);
    let consistent = match (nextn, inline_mtp) {
        (0, None) => true,
        (1, Some(index)) => index.checked_add(1) == Some(block_count),
        _ => false,
    };
    if !consistent {
        return Err(Error::from_reason(format!(
            "GGUF '{}' is {nextn} but {}; this loader reads at most one inline MTP block, the \
             last of '{}'",
            key("nextn_predict_layers"),
            match inline_mtp {
                Some(index) => format!("the header holds an inline MTP block at blk.{index}"),
                None => "the header holds no inline MTP block".to_string(),
            },
            key("block_count"),
        )));
    }
    let layers = block_count.checked_sub(nextn).ok_or_else(|| {
        Error::from_reason(format!(
            "GGUF '{}' exceeds '{}'",
            key("nextn_predict_layers"),
            key("block_count")
        ))
    })?;
    let ffn: &[&str] = if arch == "qwen35moe" {
        &QWEN35_MOE_FFN_TENSORS
    } else {
        &QWEN35_DENSE_FFN_TENSORS
    };
    let mut unread = UnreadTensors::default();
    for tensor in &gguf.tensors {
        let read = match block_of(&tensor.name) {
            Some((block, _)) if block >= layers && block < block_count => true,
            Some((block, rest)) if block < layers => {
                let full = interval != 0 && (block + 1) % interval == 0;
                let mixer: &[&str] = if full {
                    &QWEN35_FULL_ATTENTION_TENSORS
                } else {
                    &QWEN35_GDN_TENSORS
                };
                QWEN35_LAYER_TENSORS
                    .iter()
                    .chain(mixer)
                    .chain(ffn)
                    .any(|name| *name == rest)
            }
            Some(_) => false,
            None => QWEN35_GLOBAL_TENSORS.contains(&tensor.name.as_str()),
        };
        if !read {
            unread.add(&tensor.name, "unused");
        }
    }
    if unread.faults.is_empty() {
        Ok(())
    } else {
        Err(Error::from_reason(format!(
            "GGUF tensors this build cannot load: {}",
            unread.text()
        )))
    }
}

fn is_qwen35_hybrid_gguf(gguf: &GgufFile) -> bool {
    matches!(
        gguf.metadata
            .get("general.architecture")
            .and_then(GgufMetaValue::as_str),
        Some("qwen35" | "qwen35moe")
    )
}

/// Splash `flint_stop_tokens`: `tokenizer.ggml.eog_token_ids`, when present,
/// is a non-empty list of ids. Then the Qwen layouts' `.stopTokens` member
/// `<|endoftext|>` by its vocabulary string (`<|im_end|>` is always a stop).
fn plan_stop_token_ids(gguf: &GgufFile) -> Result<Vec<u32>> {
    let invalid = || Error::from_reason("invalid GGUF metadata: tokenizer.ggml.eog_token_ids");
    let mut ids: Vec<u32> = match gguf.metadata.get("tokenizer.ggml.eog_token_ids") {
        None => Vec::new(),
        Some(value) => {
            let wide: Vec<i128> = match value {
                GgufMetaValue::ArrayU32(v) => v.iter().map(|&id| i128::from(id)).collect(),
                GgufMetaValue::ArrayI32(v) => v.iter().map(|&id| i128::from(id)).collect(),
                GgufMetaValue::ArrayU64(v) => v.iter().map(|&id| i128::from(id)).collect(),
                GgufMetaValue::ArrayI64(v) => v.iter().map(|&id| i128::from(id)).collect(),
                _ => return Err(invalid()),
            };
            if wide.is_empty() {
                return Err(invalid());
            }
            wide.into_iter()
                .map(|id| u32::try_from(id).map_err(|_| invalid()))
                .collect::<Result<_>>()?
        }
    };
    if let Some(GgufMetaValue::ArrayString(tokens)) = gguf.metadata.get("tokenizer.ggml.tokens")
        && let Some(id) = tokens.iter().position(|token| token == "<|endoftext|>")
        && let Ok(id) = u32::try_from(id)
        && !ids.contains(&id)
    {
        ids.push(id);
    }
    Ok(ids)
}

/// `generation_config.json` in `out` with `eos_token_id` = its ids, then each
/// of `ids` not already there; every other key kept.
fn merge_stop_token_ids(out: &Path, ids: &[u32]) -> Result<()> {
    let path = out.join("generation_config.json");
    let mut config = match fs::read(&path) {
        Ok(bytes) => serde_json::from_slice::<serde_json::Value>(&bytes)
            .map_err(|e| Error::from_reason(format!("Failed to parse {}: {e}", path.display())))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => serde_json::json!({}),
        Err(e) => {
            return Err(Error::from_reason(format!(
                "Failed to read {}: {e}",
                path.display()
            )));
        }
    };
    let obj = config
        .as_object_mut()
        .ok_or_else(|| Error::from_reason(format!("{} is not a JSON object", path.display())))?;
    let mut eos: Vec<serde_json::Value> = match obj.get("eos_token_id") {
        Some(serde_json::Value::Array(values)) => values.clone(),
        Some(serde_json::Value::Null) | None => Vec::new(),
        Some(value) => vec![value.clone()],
    };
    for &id in ids {
        if !eos
            .iter()
            .any(|value| value.as_u64() == Some(u64::from(id)))
        {
            eos.push(serde_json::Value::from(id));
        }
    }
    obj.insert("eos_token_id".to_string(), serde_json::Value::Array(eos));
    let text = serde_json::to_string_pretty(&config)
        .map_err(|e| Error::from_reason(format!("Failed to serialize {}: {e}", path.display())))?;
    fs::write(&path, text)
        .map_err(|e| Error::from_reason(format!("Failed to write {}: {e}", path.display())))
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
        if prefix.contains(".loop_ctrl.") {
            continue;
        }
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
    // Converting into the GGUF's own folder: the copy source is the user's file.
    let generation_config_is_source = primary
        .asset_copies
        .iter()
        .any(|(src, filename)| *filename == "generation_config.json" && *src == out.join(filename));
    if !primary.stop_token_ids.is_empty() && !generation_config_is_source {
        merge_stop_token_ids(out, &primary.stop_token_ids)?;
    }
    if !out.join("tokenizer.json").exists() && write_embedded_gpt2_tokenizer(&gguf.metadata, out)? {
        info!("Reconstructed tokenizer.json from embedded GGUF GPT-2 metadata");
    }

    if let Some((plan, companion_gguf)) = companion {
        if plan.muse_secondary {
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
        } else if companion_gguf
            .metadata
            .get("clip.projector_type")
            .and_then(|v| v.as_str())
            == Some("qwen3vl_merger")
        {
            let config_path = out.join("config.json");
            let mut root: serde_json::Value =
                serde_json::from_slice(&fs::read(&config_path).map_err(|e| {
                    Error::from_reason(format!("Failed to read {}: {e}", config_path.display()))
                })?)
                .map_err(|e| Error::from_reason(format!("Failed to parse config.json: {e}")))?;
            if let Some(obj) = root.as_object_mut()
                && !obj.contains_key("vision_config")
            {
                let get_u64 = |k: &str| companion_gguf.metadata.get(k).and_then(|v| v.as_u64());
                let mut vc = serde_json::Map::new();
                if let Some(v) = get_u64("clip.vision.embedding_length") {
                    vc.insert("hidden_size".to_string(), serde_json::json!(v));
                }
                if let Some(v) = get_u64("clip.vision.feed_forward_length") {
                    vc.insert("intermediate_size".to_string(), serde_json::json!(v));
                }
                if let Some(v) = get_u64("clip.vision.attention.head_count") {
                    vc.insert("num_heads".to_string(), serde_json::json!(v));
                }
                if let Some(v) = get_u64("clip.vision.block_count") {
                    vc.insert("depth".to_string(), serde_json::json!(v));
                }
                if let Some(v) = get_u64("clip.vision.patch_size") {
                    vc.insert("patch_size".to_string(), serde_json::json!(v));
                }
                if let Some(v) = get_u64("clip.vision.spatial_merge_size") {
                    vc.insert("spatial_merge_size".to_string(), serde_json::json!(v));
                }
                if let Some(v) = get_u64("clip.vision.image_size") {
                    vc.insert("image_size".to_string(), serde_json::json!(v));
                }
                if let Some(v) = get_u64("clip.vision.projection_dim") {
                    vc.insert("out_hidden_size".to_string(), serde_json::json!(v));
                }
                obj.insert("vision_config".to_string(), serde_json::Value::Object(vc));
                let updated = serde_json::to_string_pretty(&root).map_err(|e| {
                    Error::from_reason(format!("Failed to serialize config.json: {e}"))
                })?;
                fs::write(&config_path, updated).map_err(|e| {
                    Error::from_reason(format!("Failed to write {}: {e}", config_path.display()))
                })?;
                info!("Merged Qwen3.5 mmproj vision_config metadata into config.json");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::engine::checkpoint_source::{CheckpointSource, native_import_plans};
    use crate::utils::gguf::parse_gguf;
    #[cfg(debug_assertions)]
    use crate::utils::gguf::{GgufConversionOptions, convert_gguf_to_safetensors};
    use crate::utils::gguf::{GgufTensorType, NativeGgufFamily, preflight_muse_dflash_gguf};
    use crate::utils::gguf_test_fixtures::write_split_gguf;
    use crate::utils::gguf_test_fixtures::{
        IMPORT_FIXTURE_NAMES, ImportFixture, ImportFixtureFamily, convert_import_fixture,
        import_golden_root as golden_root, safetensors_manifest, sha256_hex, t64_prefixes,
        write_import_fixture,
    };
    #[cfg(debug_assertions)]
    use crate::utils::gguf_test_fixtures::{
        build_minimal_gguf, fixture_payload, write_merged_gguf,
    };

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
        if !crate::test_support::in_own_process(
            "utils::gguf_plan::tests::header_tiled_set_equals_base_golden",
        ) {
            return;
        }
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
            let entry = convert_import_fixture(&fixture, &work.join("cache").join(name)).await;
            let live = t64_prefixes(&read_json(&entry.join("config.json")));
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
        if !crate::test_support::in_own_process(
            "utils::gguf_plan::tests::converter_output_equals_base_golden",
        ) {
            return;
        }
        let work = work_dir("convert");
        for name in IMPORT_FIXTURE_NAMES {
            let fixture = checked_fixture(name, &work.join("src").join(name));
            let entry = convert_import_fixture(&fixture, &work.join("cache").join(name)).await;
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
        }
        fs::remove_dir_all(&work).ok();
    }

    /// The options native prepare passes to the converter for `fixture`.
    #[cfg(debug_assertions)]
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
        if !crate::test_support::in_own_process(
            "utils::gguf_plan::tests::converter_tiler_drift_debug_asserts",
        ) {
            return;
        }
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

    const SPLIT_NAMES: [&str; 3] = [
        "tiny-00001-of-00003.gguf",
        "tiny-00002-of-00003.gguf",
        "tiny-00003-of-00003.gguf",
    ];

    fn assert_names_parts(text: &str) {
        for name in SPLIT_NAMES {
            assert!(text.contains(name), "{text}");
        }
    }

    fn tiny_split(dir: &Path, architecture: &str) -> Vec<PathBuf> {
        write_split_gguf(
            dir,
            "tiny",
            &[(
                "general.architecture",
                GgufMetaValue::String(architecture.to_string()),
            )],
            &[("a.weight", &[4]), ("b.weight", &[4]), ("c.weight", &[4])],
            true,
        )
    }

    fn entries(dir: &Path) -> usize {
        fs::read_dir(dir).map(|d| d.count()).unwrap_or(0)
    }

    #[tokio::test]
    async fn plan_refuses_a_lone_split_part() {
        let work = work_dir("split-plan");
        let parts = tiny_split(&work.join("src"), "qwen35");
        let gguf = parse_gguf(&parts[0]).unwrap();
        let Err(e) = plan_gguf_import(
            &gguf,
            &parts[0],
            &ImportOptions::default(),
            OutputRole::Primary,
        ) else {
            panic!("the plan read a lone split part");
        };
        assert_names_parts(&e.reason);
        let Err(e) = native_import_plans(NativeGgufFamily::Qwen35, (&parts[0], &gguf), None) else {
            panic!("native plans read a lone split part");
        };
        assert_names_parts(&e.reason);
        fs::remove_file(&parts[2]).unwrap();
        let root = work.join("cache");
        let Err(e) =
            CheckpointSource::resolve_in(&parts[0], NativeGgufFamily::Qwen35, Some(&root)).await
        else {
            panic!("native prepare read a split GGUF missing a part");
        };
        assert!(
            e.reason.contains(SPLIT_NAMES[2].trim_end_matches(".gguf")),
            "{}",
            e.reason
        );
        assert_eq!(entries(&root), 0, "refused after a cache write");
        fs::remove_dir_all(&work).ok();
    }

    #[tokio::test]
    async fn a_later_part_is_refused_before_the_architecture_check() {
        let work = work_dir("split-family");
        let parts = tiny_split(&work.join("src"), "qwen35moe");
        let root = work.join("cache");
        let Err(e) =
            CheckpointSource::resolve_in(&parts[1], NativeGgufFamily::Qwen35Moe, Some(&root)).await
        else {
            panic!("native prepare read a lone split part");
        };
        assert!(
            e.reason.contains("part 2 of a split GGUF of 3 parts"),
            "{}",
            e.reason
        );
        assert!(
            e.reason.contains("pass its first part") && e.reason.contains(SPLIT_NAMES[0]),
            "{}",
            e.reason
        );
        assert_eq!(entries(&root), 0, "refused after a cache write");

        let Err(e) = preflight_muse_dflash_gguf(
            parts[0].to_string_lossy().into_owned(),
            work.to_string_lossy().into_owned(),
        ) else {
            panic!("the draft preflight read a lone split part");
        };
        assert_names_parts(&e.reason);
        fs::remove_dir_all(&work).ok();
    }

    #[cfg(debug_assertions)]
    #[tokio::test]
    async fn convert_refuses_a_lone_split_part() {
        if !crate::test_support::in_own_process(
            "utils::gguf_plan::tests::convert_refuses_a_lone_split_part",
        ) {
            return;
        }
        let work = work_dir("split-convert");
        let metadata = [(
            "general.architecture",
            GgufMetaValue::String("gemma4".to_string()),
        )];
        let tensors: [(&str, &[u64]); 3] = [
            ("output_norm.weight", &[1]),
            ("blk.0.attn_q.weight", &[32, 1]),
            ("blk.0.attn_norm.weight", &[32]),
        ];
        let options = |input: &Path, out: &Path| GgufConversionOptions {
            input_path: input.to_string_lossy().into_owned(),
            output_dir: out.to_string_lossy().into_owned(),
            config_source_dir: None,
            dtype: None,
            verbose: Some(false),
            quantize: Some(false),
            quant_bits: None,
            quant_group_size: None,
            quant_mode: None,
            quant_recipe: None,
            imatrix_path: None,
            output_filename: None,
            vlm_key_prefix: Some(false),
            quant_mxfp: Some(false),
            import_k_quants: Some(false),
            native_qwen35_layout: None,
        };
        let out = work.join("output");
        fs::create_dir_all(&out).unwrap();
        let sentinel = out.join("model.safetensors");
        fs::write(&sentinel, b"sentinel").unwrap();

        // The first set states no split.tensors.count: split.count alone refuses it.
        for (dir, tensors_count) in [("bare", false), ("split", true)] {
            let parts =
                write_split_gguf(&work.join(dir), "tiny", &metadata, &tensors, tensors_count);
            match convert_gguf_to_safetensors(options(&parts[0], &out)).await {
                Ok(result) => panic!(
                    "{dir}: convert read a lone split part: {} tensors",
                    result.num_tensors
                ),
                Err(e) => assert_names_parts(&e.reason),
            }
            assert_eq!(fs::read(&sentinel).unwrap(), b"sentinel", "{dir}");
        }

        let merged = work.join("merged").join("tiny.gguf");
        fs::create_dir_all(merged.parent().unwrap()).unwrap();
        write_merged_gguf(&merged, &metadata, &tensors);
        let result = convert_gguf_to_safetensors(options(&merged, &work.join("merged-out")))
            .await
            .unwrap_or_else(|e| panic!("the merged file must convert: {}", e.reason));
        assert_eq!(result.num_tensors, 3);
        fs::remove_dir_all(&work).ok();
    }

    fn dense_parts(dir: &Path) -> crate::utils::gguf_test_fixtures::GgufParts {
        let fixture = write_import_fixture("qwen35-dense-untied", dir);
        let parts = crate::utils::gguf_test_fixtures::GgufParts::read(&fixture.main);
        fs::remove_file(&fixture.main).unwrap();
        parts
    }

    fn plan_of(path: &Path) -> Result<ImportPlan> {
        let mut gguf = parse_gguf(path)?;
        annotate_gguf_header(&mut gguf);
        native_import_plans(NativeGgufFamily::Qwen35, (path, &gguf), None).map(|(_, plan, _)| plan)
    }

    fn plan_err(path: &Path) -> String {
        match plan_of(path) {
            Ok(_) => panic!("{}: the plan must refuse", path.display()),
            Err(e) => e.reason,
        }
    }

    fn stop_ids_written(path: &Path, out: &Path) -> Option<Vec<u32>> {
        let plan = plan_of(path).unwrap_or_else(|e| panic!("{}", e.reason));
        let gguf = parse_gguf(path).unwrap();
        fs::create_dir_all(out).unwrap();
        write_gguf_assets(&plan, None, &gguf, out).unwrap();
        out.join("generation_config.json")
            .exists()
            .then(|| crate::engine::persistence::parse_generation_defaults(out).eos_token_ids)
    }

    #[test]
    fn header_eog_ids_join_generation_config() {
        let work = work_dir("eog");
        let src = work.join("src");
        fs::create_dir_all(&src).unwrap();
        let base = dense_parts(&src);
        let path = src.join("m.gguf");

        base.write(&path);
        assert_eq!(stop_ids_written(&path, &work.join("plain")), None);

        let mut eog = dense_parts(&src);
        eog.set(
            "tokenizer.ggml.eog_token_ids",
            GgufMetaValue::ArrayI32(vec![7, 9]),
        );
        eog.write(&path);
        assert_eq!(stop_ids_written(&path, &work.join("eog")), Some(vec![7, 9]));

        fs::write(
            src.join("generation_config.json"),
            r#"{"eos_token_id": 9, "temperature": 0.6}"#,
        )
        .unwrap();
        let out = work.join("merged");
        assert_eq!(stop_ids_written(&path, &out), Some(vec![9, 7]));
        let merged = read_json(&out.join("generation_config.json"));
        assert_eq!(merged["temperature"], serde_json::json!(0.6));
        fs::remove_file(src.join("generation_config.json")).unwrap();

        let mut vocab = dense_parts(&src);
        vocab.set(
            "tokenizer.ggml.tokens",
            GgufMetaValue::ArrayString(vec![
                "a".into(),
                "<|im_end|>".into(),
                "<|endoftext|>".into(),
            ]),
        );
        vocab.write(&path);
        assert_eq!(
            stop_ids_written(&path, &work.join("vocab")),
            Some(vec![2]),
            "Splash Qwen .stopTokens: <|endoftext|> stops decode"
        );

        for bad in [
            GgufMetaValue::ArrayString(vec!["7".into()]),
            GgufMetaValue::ArrayI32(Vec::new()),
            GgufMetaValue::ArrayI32(vec![-1]),
        ] {
            let mut parts = dense_parts(&src);
            parts.set("tokenizer.ggml.eog_token_ids", bad.clone());
            parts.write(&path);
            let text = plan_err(&path);
            assert!(
                text.contains("invalid GGUF metadata: tokenizer.ggml.eog_token_ids"),
                "{bad:?}: {text}"
            );
        }
        fs::remove_dir_all(&work).ok();
    }

    #[tokio::test]
    async fn plan_refuses_unread_qwen35_tensors() {
        use GgufTensorType::*;
        let work = work_dir("unread");
        let src = work.join("src");
        fs::create_dir_all(&src).unwrap();
        let path = src.join("m.gguf");

        dense_parts(&src).write(&path);
        plan_of(&path).unwrap_or_else(|e| panic!("plain fixture: {}", e.reason));

        let mut bias = dense_parts(&src);
        bias.add_tensor("blk.1.attn_q.bias", &[512], F32);
        bias.write(&path);
        let text = plan_err(&path);
        assert!(
            text.contains("GGUF tensors this build cannot load: attn_q.bias (unused) in 1 layer"),
            "{text}"
        );
        let root = work.join("cache");
        let Err(e) =
            CheckpointSource::resolve_in(&path, NativeGgufFamily::Qwen35, Some(&root)).await
        else {
            panic!("native prepare loaded a GGUF with an unread bias");
        };
        assert!(
            e.reason.contains("attn_q.bias (unused) in 1 layer"),
            "{}",
            e.reason
        );
        assert_eq!(
            fs::read_dir(&root)
                .map(|d| d
                    .filter(|e| !e
                        .as_ref()
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .starts_with('.'))
                    .count())
                .unwrap_or(0),
            0,
            "refused after a cache write"
        );

        type Edit = Box<dyn Fn(&mut crate::utils::gguf_test_fixtures::GgufParts)>;
        let cases: Vec<(&str, Edit)> = vec![
            (
                "attn_q.weight (unused) in 1 layer",
                Box::new(|p| p.add_tensor("blk.0.attn_q.weight", &[256, 512], F32)),
            ),
            (
                "rope_freqs.weight (unused)",
                Box::new(|p| p.add_tensor("rope_freqs.weight", &[64], F32)),
            ),
            (
                "attn_norm.weight (unused) in 1 layer",
                Box::new(|p| p.add_tensor("blk.2.attn_norm.weight", &[256], F32)),
            ),
            (
                "is 1 but the header holds no inline MTP block",
                Box::new(|p| p.set("qwen35.nextn_predict_layers", GgufMetaValue::Uint32(1))),
            ),
        ];
        for (want, edit) in &cases {
            let mut parts = dense_parts(&src);
            edit(&mut parts);
            parts.write(&path);
            let text = plan_err(&path);
            assert!(text.contains(want), "{want}: {text}");
        }

        let mtp = |nextn: Option<u32>, extra: Option<&str>| {
            let mut parts = dense_parts(&src);
            parts.set("qwen35.block_count", GgufMetaValue::Uint32(3));
            if let Some(nextn) = nextn {
                parts.set("qwen35.nextn_predict_layers", GgufMetaValue::Uint32(nextn));
            }
            parts.add_tensor("blk.2.nextn.eh_proj.weight", &[512, 256], F32);
            for name in [
                "blk.2.nextn.enorm.weight",
                "blk.2.nextn.hnorm.weight",
                "blk.2.nextn.shared_head_norm.weight",
                "blk.2.attn_norm.weight",
            ] {
                parts.add_tensor(name, &[256], F32);
            }
            parts.add_tensor("blk.2.attn_q.weight", &[256, 512], F32);
            if let Some(extra) = extra {
                parts.add_tensor(extra, &[256], F32);
            }
            parts.write(&path);
        };
        mtp(Some(1), None);
        plan_of(&path).unwrap_or_else(|e| panic!("the MTP block is exempt: {}", e.reason));
        mtp(Some(1), Some("blk.3.attn_norm.weight"));
        let text = plan_err(&path);
        assert!(
            text.contains("attn_norm.weight (unused) in 1 layer"),
            "{text}"
        );
        mtp(None, None);
        let text = plan_err(&path);
        assert!(
            text.contains("is 0 but the header holds an inline MTP block at blk.2"),
            "{text}"
        );

        let moe_dir = work.join("moe");
        let moe = write_import_fixture("qwen35moe-3d-experts", &moe_dir);
        let mut moe_parts = crate::utils::gguf_test_fixtures::GgufParts::read(&moe.main);
        moe_parts.add_tensor("blk.0.ffn_gate_up_exps.weight", &[256, 128, 2], F32);
        moe_parts.write(&moe.main);
        let mut gguf = parse_gguf(&moe.main).unwrap();
        annotate_gguf_header(&mut gguf);
        let Err(e) = native_import_plans(NativeGgufFamily::Qwen35Moe, (&moe.main, &gguf), None)
        else {
            panic!("the MoE plan read a fused gate_up stack");
        };
        assert!(
            e.reason
                .contains("ffn_gate_up_exps.weight (unused) in 1 layer"),
            "{}",
            e.reason
        );
        fs::remove_dir_all(&work).ok();
    }

    #[test]
    fn qwen35_native_mmproj_preflight_and_vision_config_synthesis() {
        use crate::utils::gguf::native_gguf_preflight;

        let work = work_dir("qwen35-mmproj");
        fs::create_dir_all(&work).unwrap();

        // Build a minimal Qwen3.5 dense GGUF from the existing import fixture.
        let src_dir = work.join("src");
        let fixture = write_import_fixture("qwen35-dense-untied", &src_dir);
        let main_path = fixture.main;

        // Without any sibling mmproj-*.gguf, preflight returns companion = None.
        let (parts_no_mm, mm_none) =
            native_gguf_preflight(&main_path, NativeGgufFamily::Qwen35).unwrap();
        assert_eq!(parts_no_mm.len(), 1);
        assert!(mm_none.is_none());

        // Read the fixture's backbone embedding_length so projector width validation matches.
        let main_header = parse_gguf(&main_path).unwrap();
        let backbone_hidden = main_header
            .metadata
            .get("qwen35.embedding_length")
            .and_then(|v| v.as_u64())
            .expect("qwen35-dense-untied must declare qwen35.embedding_length")
            as u32;

        // A mismatched-width projector (e.g. 27B projector beside a 9B backbone) must be ignored.
        let wrong_meta: Vec<(&str, GgufMetaValue)> = vec![
            (
                "general.architecture",
                GgufMetaValue::String("clip".to_string()),
            ),
            ("clip.has_vision_encoder", GgufMetaValue::Bool(true)),
            (
                "clip.projector_type",
                GgufMetaValue::String("qwen3vl_merger".to_string()),
            ),
            (
                "clip.vision.projection_dim",
                GgufMetaValue::Uint32(backbone_hidden + 128),
            ),
        ];
        let mm_wrong = src_dir.join("mmproj-wrong-BF16.gguf");
        fs::write(
            &mm_wrong,
            crate::utils::gguf_test_fixtures::build_minimal_gguf(&wrong_meta, &[]),
        )
        .unwrap();
        let (_, mm_mismatch) = native_gguf_preflight(&main_path, NativeGgufFamily::Qwen35).unwrap();
        assert!(
            mm_mismatch.is_none(),
            "mismatched projection_dim projector must not attach"
        );
        fs::remove_file(&mm_wrong).unwrap();

        // Write two matching Qwen3.5 mmproj companions (F16 and BF16); preflight should
        // pick BF16 (higher dtype tier).
        let mm_meta: Vec<(&str, GgufMetaValue)> = vec![
            (
                "general.architecture",
                GgufMetaValue::String("clip".to_string()),
            ),
            ("clip.has_vision_encoder", GgufMetaValue::Bool(true)),
            (
                "clip.projector_type",
                GgufMetaValue::String("qwen3vl_merger".to_string()),
            ),
            ("clip.vision.embedding_length", GgufMetaValue::Uint32(1152)),
            (
                "clip.vision.feed_forward_length",
                GgufMetaValue::Uint32(4304),
            ),
            (
                "clip.vision.attention.head_count",
                GgufMetaValue::Uint32(16),
            ),
            ("clip.vision.block_count", GgufMetaValue::Uint32(27)),
            ("clip.vision.patch_size", GgufMetaValue::Uint32(16)),
            ("clip.vision.spatial_merge_size", GgufMetaValue::Uint32(2)),
            ("clip.vision.image_size", GgufMetaValue::Uint32(768)),
            (
                "clip.vision.projection_dim",
                GgufMetaValue::Uint32(backbone_hidden),
            ),
        ];
        let mm_f16 = src_dir.join("mmproj-F16.gguf");
        let mm_bf16 = src_dir.join("mmproj-BF16.gguf");
        let mm_bytes = crate::utils::gguf_test_fixtures::build_minimal_gguf(&mm_meta, &[]);
        fs::write(&mm_f16, &mm_bytes).unwrap();
        fs::write(&mm_bf16, &mm_bytes).unwrap();

        let (_, mm_picked) = native_gguf_preflight(&main_path, NativeGgufFamily::Qwen35).unwrap();
        let mm_picked = mm_picked.expect("should discover qwen35 mmproj companion");
        assert_eq!(
            mm_picked.file_name().and_then(|s| s.to_str()),
            Some("mmproj-BF16.gguf")
        );

        // Two matching projectors at the same best dtype tier must fail closed.
        let mm_bf16_dup = src_dir.join("mmproj-alt-BF16.gguf");
        fs::write(&mm_bf16_dup, &mm_bytes).unwrap();
        let err = native_gguf_preflight(&main_path, NativeGgufFamily::Qwen35)
            .expect_err("ambiguous tie at same dtype tier must fail closed");
        assert!(
            err.reason.contains("Ambiguous Qwen3.5 mmproj"),
            "unexpected error: {}",
            err.reason
        );
        fs::remove_file(&mm_bf16_dup).unwrap();

        // Verify write_gguf_assets synthesizes vision_config into config.json.
        let out_dir = work.join("assets");
        fs::create_dir_all(&out_dir).unwrap();
        let planned = plan_like_prepare(&main_path, true, Some((&mm_picked, "mmproj.safetensors")));
        let (comp_gguf, comp_plan) = planned
            .companion
            .as_ref()
            .expect("companion plan must be present");
        write_gguf_assets(
            &planned.plan,
            Some((comp_plan, comp_gguf)),
            &planned.gguf,
            &out_dir,
        )
        .unwrap();

        let cfg = read_json(&out_dir.join("config.json"));
        let vc = cfg
            .get("vision_config")
            .and_then(|v| v.as_object())
            .expect("vision_config must be synthesized from mmproj");
        assert_eq!(vc.get("hidden_size").and_then(|v| v.as_u64()), Some(1152));
        assert_eq!(
            vc.get("intermediate_size").and_then(|v| v.as_u64()),
            Some(4304)
        );
        assert_eq!(vc.get("num_heads").and_then(|v| v.as_u64()), Some(16));
        assert_eq!(vc.get("depth").and_then(|v| v.as_u64()), Some(27));
        assert_eq!(vc.get("patch_size").and_then(|v| v.as_u64()), Some(16));
        assert_eq!(
            vc.get("spatial_merge_size").and_then(|v| v.as_u64()),
            Some(2)
        );
        assert_eq!(vc.get("image_size").and_then(|v| v.as_u64()), Some(768));
        assert_eq!(
            vc.get("out_hidden_size").and_then(|v| v.as_u64()),
            Some(u64::from(backbone_hidden))
        );

        fs::remove_dir_all(&work).ok();
    }

    #[test]
    fn qwen35_non_native_mxfp8_linear_attn_rejected_at_header_preflight() {
        use super::super::gguf::{GgufTensorInfo, GgufTensorType};
        use std::collections::HashMap;

        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufMetaValue::String("qwen35".to_string()),
        );
        let gguf = GgufFile {
            version: 3,
            tensor_count: 1,
            metadata,
            tensors: vec![GgufTensorInfo {
                name: "blk.0.attn_gate.weight".to_string(),
                n_dims: 2,
                dims: vec![64, 64],
                tensor_type: GgufTensorType::MXFP8,
                offset: 0,
            }],
            data_offset: 0,
            alignment: 32,
        };
        // Non-native Qwen3.5 GGUF with MXFP8 linear_attn.in_proj_z.weight MUST fail preflight
        // so dense head reinterleave can never reorder packed codes without reordering scales.
        let err = plan_gguf_import(
            &gguf,
            Path::new("qwen35.gguf"),
            &ImportOptions::default(),
            OutputRole::Primary,
        )
        .unwrap_err();
        assert!(
            err.reason.contains("MXFP8") && err.reason.contains("linear_attn.in_proj_z.weight"),
            "unexpected preflight error: {err}"
        );
        // Native layout bypasses the non-native head-reinterleave fixup.
        assert!(qwen35_mxfp8_dense_fixup_target(&gguf).is_some());

        // Preserved non-engram MXFP8 combined with `--quantize` MUST fail preflight so
        // config.json never drops the explicit `mxfp8` profile for skipped packed tensors.
        let mut attn_q_gguf = gguf;
        attn_q_gguf.tensors[0].name = "blk.0.attn_q.weight".to_string();
        let mut q_opts = ImportOptions::default();
        q_opts.quantize = true;
        q_opts.native_qwen35_layout = true;
        let q_err = plan_gguf_import(
            &attn_q_gguf,
            Path::new("qwen35.gguf"),
            &q_opts,
            OutputRole::Primary,
        )
        .unwrap_err();
        assert!(
            q_err.reason.contains("MXFP8") && q_err.reason.contains("--quantize"),
            "unexpected --quantize preflight error: {q_err}"
        );
    }

    #[tokio::test]
    async fn inkling2_mmproj_q8_0_plan_f32_preservation_and_text_only_fallback() {
        use super::super::gguf::{
            GgufTensorInfo, GgufTensorType, native_gguf_preflight, prepare_gguf_assets_in,
        };
        use std::collections::HashMap;

        let make_mmproj = |tensors: Vec<GgufTensorInfo>| -> GgufFile {
            let mut metadata = HashMap::new();
            metadata.insert(
                "general.architecture".to_string(),
                GgufMetaValue::String("clip".to_string()),
            );
            metadata.insert(
                "clip.vision.projector_type".to_string(),
                GgufMetaValue::String("inkling2v".to_string()),
            );
            GgufFile {
                version: 3,
                tensor_count: tensors.len() as u64,
                metadata,
                tensors,
                data_offset: 0,
                alignment: 32,
            }
        };

        // 1. Valid mmproj-Q8_0: F32 1D norms/biases/patch_embd + 2D Q8_0 matrices with cols % 32 == 0.
        let valid_q8 = make_mmproj(vec![
            GgufTensorInfo {
                name: "v.patch_embd.weight".to_string(),
                n_dims: 4,
                dims: vec![20, 20, 3, 64],
                tensor_type: GgufTensorType::F32,
                offset: 0,
            },
            GgufTensorInfo {
                name: "v.pre_ln.weight".to_string(),
                n_dims: 1,
                dims: vec![64],
                tensor_type: GgufTensorType::F32,
                offset: 0,
            },
            GgufTensorInfo {
                name: "v.blk.0.attn_q.weight".to_string(),
                n_dims: 2,
                dims: vec![64, 64],
                tensor_type: GgufTensorType::Q8_0,
                offset: 0,
            },
            GgufTensorInfo {
                name: "mm.1.weight".to_string(),
                n_dims: 2,
                dims: vec![128, 64],
                tensor_type: GgufTensorType::Q8_0,
                offset: 0,
            },
        ]);
        let opts = ImportOptions {
            dtype: Some(DType::BFloat16),
            import_k_quants: true,
            ..Default::default()
        };
        let plan = plan_gguf_import(
            &valid_q8,
            Path::new("mmproj-Q8_0.gguf"),
            &opts,
            OutputRole::Companion {
                filename: "vision.safetensors".to_string(),
            },
        )
        .expect("Inkling2 mmproj-Q8_0 with aligned 2D Q8_0 matrices must be accepted");
        assert!(
            plan.preserve_dtype_keys
                .contains("vision_tower.patch_embed.weight"),
            "F32 patch_embed must be preserved in preserve_dtype_keys"
        );
        assert!(
            plan.preserve_dtype_keys
                .contains("vision_tower.pre_ln.weight"),
            "F32 pre_ln.weight must be preserved in preserve_dtype_keys"
        );
        assert!(
            !plan
                .preserve_dtype_keys
                .contains("vision_tower.blocks.0.attn.q_proj.weight"),
            "Q8_0 weight is not F32"
        );

        // 2. Unsupported Q4_0, Q4_K, unaligned Q8_0 (cols % 32 != 0), and 1D Q8_0 are all rejected.
        let bad_cases: [(&str, Vec<u64>, GgufTensorType); 4] = [
            ("v.blk.0.attn_q.weight", vec![64, 64], GgufTensorType::Q4_0),
            ("v.blk.0.attn_q.weight", vec![256, 64], GgufTensorType::Q4K),
            ("v.patch_embd.weight", vec![1200, 64], GgufTensorType::Q8_0),
            ("v.pre_ln.weight", vec![64], GgufTensorType::Q8_0),
        ];
        for (name, dims, ty) in bad_cases {
            let bad = make_mmproj(vec![GgufTensorInfo {
                name: name.to_string(),
                n_dims: dims.len() as u32,
                dims,
                tensor_type: ty,
                offset: 0,
            }]);
            let err = plan_gguf_import(
                &bad,
                Path::new("mmproj-bad.gguf"),
                &opts,
                OutputRole::Companion {
                    filename: "vision.safetensors".to_string(),
                },
            )
            .expect_err("unsupported mmproj tensor must fail planning");
            assert!(
                err.reason.contains("Inkling2 vision projector"),
                "unexpected error for {name} ({ty:?}): {err}"
            );
        }

        // 3. End-to-end CheckpointSource / prepare_gguf_assets_in soft fallback on unsupported Q4_0 mmproj
        //    and corrupt/ambiguous sibling mmproj files.
        let work = work_dir("inkling2-mmproj-fallback");
        let src = work.join("src");
        let cache = work.join("cache");
        fs::create_dir_all(&src).unwrap();
        let main_path = src.join("inkling2.gguf");
        let main_bytes = {
            let mut b = Vec::new();
            let main_meta: Vec<(&str, GgufMetaValue)> = vec![
                (
                    "general.architecture",
                    GgufMetaValue::String("inkling2".to_string()),
                ),
                ("inkling2.embedding_length", GgufMetaValue::Uint32(64)),
                ("inkling2.block_count", GgufMetaValue::Uint32(1)),
                ("inkling2.attention.head_count", GgufMetaValue::Uint32(2)),
                ("inkling2.attention.head_count_kv", GgufMetaValue::Uint32(2)),
                ("inkling2.attention.key_length", GgufMetaValue::Uint32(32)),
                ("inkling2.feed_forward_length", GgufMetaValue::Uint32(64)),
            ];
            let mut tmp = Vec::new();
            tmp.extend_from_slice(&super::super::gguf::GGUF_MAGIC.to_le_bytes());
            tmp.extend_from_slice(&super::super::gguf::GGUF_VERSION_3.to_le_bytes());
            tmp.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
            tmp.extend_from_slice(&(main_meta.len() as u64).to_le_bytes());
            for (k, v) in &main_meta {
                tmp.extend_from_slice(&(k.len() as u64).to_le_bytes());
                tmp.extend_from_slice(k.as_bytes());
                match v {
                    GgufMetaValue::String(s) => {
                        tmp.extend_from_slice(&8u32.to_le_bytes());
                        tmp.extend_from_slice(&(s.len() as u64).to_le_bytes());
                        tmp.extend_from_slice(s.as_bytes());
                    }
                    GgufMetaValue::Uint32(n) => {
                        tmp.extend_from_slice(&4u32.to_le_bytes());
                        tmp.extend_from_slice(&n.to_le_bytes());
                    }
                    _ => unreachable!(),
                }
            }
            // token_embd.weight [64, 4] F32
            let tname = "token_embd.weight";
            tmp.extend_from_slice(&(tname.len() as u64).to_le_bytes());
            tmp.extend_from_slice(tname.as_bytes());
            tmp.extend_from_slice(&2u32.to_le_bytes());
            tmp.extend_from_slice(&64u64.to_le_bytes());
            tmp.extend_from_slice(&4u64.to_le_bytes());
            tmp.extend_from_slice(&(GgufTensorType::F32 as u32).to_le_bytes());
            tmp.extend_from_slice(&0u64.to_le_bytes());
            tmp.resize(tmp.len().next_multiple_of(32), 0);
            tmp.extend(std::iter::repeat_n(0u8, 64 * 4 * 4));
            b.extend_from_slice(&tmp);
            b
        };
        fs::write(&main_path, &main_bytes).unwrap();

        // Write an unsupported Q4_0 mmproj companion beside inkling2.gguf.
        let bad_mmproj_path = src.join("mmproj-Q4_0.gguf");
        {
            let mut b = Vec::new();
            let meta = [
                ("general.architecture", "clip"),
                ("clip.vision.projector_type", "inkling2v"),
            ];
            b.extend_from_slice(&super::super::gguf::GGUF_MAGIC.to_le_bytes());
            b.extend_from_slice(&super::super::gguf::GGUF_VERSION_3.to_le_bytes());
            b.extend_from_slice(&1u64.to_le_bytes());
            b.extend_from_slice(&(meta.len() as u64).to_le_bytes());
            for (k, s) in meta {
                b.extend_from_slice(&(k.len() as u64).to_le_bytes());
                b.extend_from_slice(k.as_bytes());
                b.extend_from_slice(&8u32.to_le_bytes());
                b.extend_from_slice(&(s.len() as u64).to_le_bytes());
                b.extend_from_slice(s.as_bytes());
            }
            let tname = "v.blk.0.attn_q.weight";
            b.extend_from_slice(&(tname.len() as u64).to_le_bytes());
            b.extend_from_slice(tname.as_bytes());
            b.extend_from_slice(&2u32.to_le_bytes());
            b.extend_from_slice(&64u64.to_le_bytes());
            b.extend_from_slice(&4u64.to_le_bytes());
            b.extend_from_slice(&(GgufTensorType::Q4_0 as u32).to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
            b.resize(b.len().next_multiple_of(32), 0);
            b.extend(std::iter::repeat_n(0u8, (64 * 4 / 32) * 18));
            fs::write(&bad_mmproj_path, &b).unwrap();
        }
        // Also place a corrupt sibling mmproj file in the same directory; preflight must skip it.
        let corrupt_mmproj_path = src.join("mmproj-corrupt.gguf");
        fs::write(&corrupt_mmproj_path, b"NOT_A_GGUF_FILE").unwrap();

        let ckpt = prepare_gguf_assets_in(&main_path, NativeGgufFamily::Inkling2, &cache)
            .await
            .expect("unsupported Q4_0 mmproj + corrupt sibling must fall back to text-only");
        assert!(
            ckpt.companion.is_none(),
            "unsupported companion must be dropped"
        );
        assert!(
            ckpt.plans.1.is_none(),
            "unsupported companion plan must be None"
        );
        let source_json = read_json(&ckpt.asset_dir.join("gguf-source.json"));
        assert!(
            source_json["companion"].is_null(),
            "gguf-source.json must record null companion after text-only fallback"
        );
        super::super::gguf::native_gguf_unlease_in_use(&ckpt.asset_dir);

        // Ambiguous same-tier mmproj files also degrade cleanly to `None` in preflight.
        fs::remove_file(&bad_mmproj_path).unwrap();
        fs::remove_file(&corrupt_mmproj_path).unwrap();
        let make_valid_mmproj_bytes = || {
            let mut b = Vec::new();
            let meta = [
                ("general.architecture", "clip"),
                ("clip.vision.projector_type", "inkling2v"),
            ];
            b.extend_from_slice(&super::super::gguf::GGUF_MAGIC.to_le_bytes());
            b.extend_from_slice(&super::super::gguf::GGUF_VERSION_3.to_le_bytes());
            b.extend_from_slice(&1u64.to_le_bytes());
            b.extend_from_slice(&(meta.len() as u64).to_le_bytes());
            for (k, s) in meta {
                b.extend_from_slice(&(k.len() as u64).to_le_bytes());
                b.extend_from_slice(k.as_bytes());
                b.extend_from_slice(&8u32.to_le_bytes());
                b.extend_from_slice(&(s.len() as u64).to_le_bytes());
                b.extend_from_slice(s.as_bytes());
            }
            let tname = "v.pre_ln.weight";
            b.extend_from_slice(&(tname.len() as u64).to_le_bytes());
            b.extend_from_slice(tname.as_bytes());
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&16u64.to_le_bytes());
            b.extend_from_slice(&(GgufTensorType::F32 as u32).to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
            b.resize(b.len().next_multiple_of(32), 0);
            b.extend(std::iter::repeat_n(0u8, 16 * 4));
            b
        };
        let mm_bytes = make_valid_mmproj_bytes();
        fs::write(src.join("mmproj-a-BF16.gguf"), &mm_bytes).unwrap();
        fs::write(src.join("mmproj-b-BF16.gguf"), &mm_bytes).unwrap();
        let (_parts, amb_companion) = native_gguf_preflight(&main_path, NativeGgufFamily::Inkling2)
            .expect("ambiguous Inkling2 mmproj must degrade cleanly instead of erroring");
        assert!(amb_companion.is_none(), "ambiguous mmproj must return None");

        fs::remove_dir_all(&work).ok();
    }
}
