use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::cold_tier::{CheckpointLoadGuard, resolve_persist_cold};
use crate::engine::checkpoint_source::CheckpointSource;
use crate::engine::persistence::{
    KeyRule, RenameSpec, apply_rename_spec, load_checkpoint, parse_generation_defaults,
};
use crate::models::gemma4::quantized_linear::{
    LinearProj, try_build_affine_quantized_linear_tiled, try_build_fp8_e4m3_quantized_linear,
    try_build_kquant_quantized_linear_tiled, try_build_mxfp4_quantized_linear,
    try_build_mxfp8_quantized_linear, try_build_nvfp4_quantized_linear,
    try_build_sym8_quantized_linear,
};
use crate::models::quant_dispatch::{
    PerLayerMode, PerLayerQuant, ensure_affine_biases_present, ensure_dense_weight_floating,
    ensure_int8_storage_resolves_sym8, ensure_kquant_storage_resolves_kquant,
    ensure_plain_fp8_storage_resolves_fp8_e4m3, ensure_row_major_kquant_layout, has_kquant_mode,
    load_quant_settings_from_disk, mode_to_str, normalize_per_layer_key,
};
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use crate::transformer::{AttentionKind, KVCacheDType};
use crate::utils::gguf::NativeGgufFamily;
use crate::utils::gguf_import::{GgufPreload, PreloadInput};
use crate::utils::safetensors::{SafeTensorsFile, load_safetensors_lazy};

use super::attention::MuseGlimmerAttention;
use super::config::MuseGlimmerConfig;
use super::decoder_layer::MuseGlimmerDecoderLayer;
use super::dflash::{DFlashAttention, DFlashLayer, DFlashModel};
use super::kv_cache::{
    PagedWindowSlot, WindowCarrier, admit_paged_dispatch_plan, compute_layer_kv_cache_groups,
    compute_layer_kv_cache_specs, group_reserved_blocks,
};
use super::mlp::MuseGlimmerMlp;
use super::model::{MuseGlimmerInner, MuseGlimmerModel, MusePagedRuntime, MuseSchedulerState};
use super::output_parser::ResponseTemplate;

fn required<'a>(params: &'a HashMap<String, MxArray>, key: &str) -> Result<&'a MxArray> {
    params
        .get(key)
        .ok_or_else(|| Error::from_reason(format!("Muse-Glimmer checkpoint is missing '{key}'")))
}

pub(crate) fn quant_lookup_prefix(prefix: &str) -> String {
    normalize_per_layer_key(&crate::utils::normalize_override_key(prefix))
}

pub(crate) fn muse_projection_quant(
    prefix: &str,
    overrides: &HashMap<String, PerLayerQuant>,
    default: PerLayerQuant,
) -> PerLayerQuant {
    let scoped = quant_lookup_prefix(prefix);
    let bare = normalize_per_layer_key(prefix);
    overrides
        .get(&scoped)
        .or_else(|| overrides.get(&bare))
        .copied()
        .unwrap_or(default)
}

pub(super) fn build_projection(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    default: PerLayerQuant,
    overrides: &HashMap<String, PerLayerQuant>,
) -> Result<LinearProj> {
    build_projection_tiled(params, prefix, default, overrides, &mut Vec::new())
}

/// [`build_projection`] that also records, in `tiled`, the prefixes whose
/// K-quant or bf16-companion affine group was repacked into the Tiled64
/// layout here (a `t64` checkpoint tensor is built tiled as-is and not
/// recorded), so the caller can drop their row-major map entries with
/// `release_tiled_kquant_sources` once the owning layer is built. The plain `build_projection` discards the list:
/// its callers (the DFlash draft, tests) hold small maps whose transient is
/// not worth plumbing.
pub(super) fn build_projection_tiled(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    default: PerLayerQuant,
    overrides: &HashMap<String, PerLayerQuant>,
    tiled: &mut Vec<String>,
) -> Result<LinearProj> {
    let plq = muse_projection_quant(prefix, overrides, default);
    let scales_key = format!("{prefix}.scales");
    if params.contains_key(&scales_key) {
        ensure_int8_storage_resolves_sym8(params, prefix, plq.mode, "muse_glimmer")?;
        ensure_plain_fp8_storage_resolves_fp8_e4m3(params, prefix, plq.mode, "muse_glimmer")?;
        ensure_kquant_storage_resolves_kquant(params, prefix, plq.mode, "muse_glimmer")?;
        ensure_affine_biases_present(params, prefix, plq.mode, "muse_glimmer")?;
        let quantized = match plq.mode {
            PerLayerMode::Mxfp4 => try_build_mxfp4_quantized_linear(params, prefix),
            PerLayerMode::Mxfp8 => try_build_mxfp8_quantized_linear(params, prefix),
            PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_linear(params, prefix),
            PerLayerMode::Fp8E4m3 => try_build_fp8_e4m3_quantized_linear(params, prefix)?,
            // Tiled64 into the affine K-quant contract on a Metal host, as
            // the K-quant arm (Muse-Glimmer threads no activation amax).
            PerLayerMode::Affine => try_build_affine_quantized_linear_tiled(
                params,
                prefix,
                plq.group_size,
                plq.bits,
                tiled,
            )?,
            PerLayerMode::Sym8 => try_build_sym8_quantized_linear(params, prefix)?,
            PerLayerMode::Q4K
            | PerLayerMode::Q5K
            | PerLayerMode::Q6K
            | PerLayerMode::Q3K
            | PerLayerMode::Q2K
            | PerLayerMode::IQ4NL
            | PerLayerMode::IQ4XS
            | PerLayerMode::IQ3S
            | PerLayerMode::IQ2XXS
            | PerLayerMode::IQ2XS
            | PerLayerMode::IQ2S
            | PerLayerMode::IQ3XXS
            | PerLayerMode::IQ1S
            | PerLayerMode::IQ1M => try_build_kquant_quantized_linear_tiled(
                params,
                prefix,
                plq.mode,
                plq.layout,
                "muse_glimmer",
                tiled,
            )?,
        }
        .ok_or_else(|| {
            Error::from_reason(format!(
                "Muse-Glimmer projection '{prefix}' has quant sidecars but no complete quant group"
            ))
        })?;
        return Ok(LinearProj::Quantized(quantized));
    }

    let weight_key = format!("{prefix}.weight");
    let weight = required(params, &weight_key)?;
    ensure_dense_weight_floating(&weight_key, weight)?;
    let bias = params.get(&format!("{prefix}.bias"));
    Ok(LinearProj::Standard(Linear::from_weights(weight, bias)?))
}

fn load_embedding(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    vocab: usize,
    hidden: usize,
    default: PerLayerQuant,
    overrides: &HashMap<String, PerLayerQuant>,
) -> Result<Embedding> {
    let mut embedding = Embedding::new_uninitialized(vocab as u32, hidden as u32)?;
    let weight = required(params, &format!("{prefix}.weight"))?;
    if let Some(scales) = params.get(&format!("{prefix}.scales")) {
        let plq = muse_projection_quant(prefix, overrides, default);
        if matches!(plq.mode, PerLayerMode::Sym8 | PerLayerMode::Fp8E4m3) {
            return Err(Error::from_reason(format!(
                "Muse-Glimmer embedding '{prefix}' does not support {:?}",
                plq.mode
            )));
        }
        ensure_kquant_storage_resolves_kquant(params, prefix, plq.mode, "muse_glimmer")?;
        ensure_row_major_kquant_layout(plq, prefix, "muse_glimmer")?;
        embedding.load_quantized_packed(
            weight,
            scales,
            params.get(&format!("{prefix}.biases")),
            plq.group_size,
            plq.bits,
            mode_to_str(plq.mode),
        )?;
    } else {
        ensure_dense_weight_floating(&format!("{prefix}.weight"), weight)?;
        embedding.load_weight(weight)?;
    }
    Ok(embedding)
}

fn load_lm_head(
    params: &HashMap<String, MxArray>,
    tie_word_embeddings: bool,
    default: PerLayerQuant,
    overrides: &HashMap<String, PerLayerQuant>,
) -> Result<Option<LinearProj>> {
    if tie_word_embeddings {
        Ok(None)
    } else {
        build_projection(params, "lm_head", default, overrides).map(Some)
    }
}

pub(super) fn norm(params: &HashMap<String, MxArray>, key: &str, eps: f32) -> Result<RMSNorm> {
    RMSNorm::from_weight(required(params, key)?, Some(eps as f64))
}

/// The DFlash draft's raw weights: `preloaded` (the GGUF companion map,
/// empty without a companion) for a GGUF, else `draft.safetensors` in `path`.
fn draft_params(
    path: &Path,
    preloaded: Option<HashMap<String, MxArray>>,
) -> Result<Option<HashMap<String, MxArray>>> {
    match preloaded {
        Some(draft) => Ok((!draft.is_empty()).then_some(draft)),
        None => {
            let draft_path = path.join("draft.safetensors");
            if draft_path.is_file() {
                load_safetensors_lazy(&draft_path).map(Some)
            } else {
                Ok(None)
            }
        }
    }
}

fn load_dflash(
    config: &MuseGlimmerConfig,
    params: Option<HashMap<String, MxArray>>,
    default: PerLayerQuant,
    overrides: &HashMap<String, PerLayerQuant>,
) -> Result<Option<DFlashModel>> {
    let Some(params) = params else {
        return Ok(None);
    };
    let draft_config = config.dflash_config.clone().ok_or_else(|| {
        Error::from_reason(
            "Muse-Glimmer has a DFlash draft but config.json has no dflash_config; reconvert the companion GGUF with --draft",
        )
    })?;
    let fc = build_projection(&params, "fc", default, overrides)?;
    let hidden_norm = norm(&params, "hidden_norm.weight", draft_config.rms_norm_eps)?;
    let final_norm = norm(&params, "norm.weight", draft_config.rms_norm_eps)?;
    let mut layers = Vec::with_capacity(draft_config.num_hidden_layers);
    for index in 0..draft_config.num_hidden_layers {
        let base = format!("layers.{index}");
        let attention_base = format!("{base}.self_attn");
        let attention = DFlashAttention::new(
            &draft_config,
            build_projection(
                &params,
                &format!("{attention_base}.q_proj"),
                default,
                overrides,
            )?,
            build_projection(
                &params,
                &format!("{attention_base}.k_proj"),
                default,
                overrides,
            )?,
            build_projection(
                &params,
                &format!("{attention_base}.v_proj"),
                default,
                overrides,
            )?,
            build_projection(
                &params,
                &format!("{attention_base}.o_proj"),
                default,
                overrides,
            )?,
            norm(
                &params,
                &format!("{attention_base}.q_norm.weight"),
                draft_config.rms_norm_eps,
            )?,
            norm(
                &params,
                &format!("{attention_base}.k_norm.weight"),
                draft_config.rms_norm_eps,
            )?,
        );
        let mlp_base = format!("{base}.mlp");
        let mlp = MuseGlimmerMlp::new(
            build_projection(
                &params,
                &format!("{mlp_base}.gate_proj"),
                default,
                overrides,
            )?,
            build_projection(&params, &format!("{mlp_base}.up_proj"), default, overrides)?,
            build_projection(
                &params,
                &format!("{mlp_base}.down_proj"),
                default,
                overrides,
            )?,
        );
        layers.push(DFlashLayer::new(
            attention,
            mlp,
            norm(
                &params,
                &format!("{base}.input_layernorm.weight"),
                draft_config.rms_norm_eps,
            )?,
            norm(
                &params,
                &format!("{base}.post_attention_layernorm.weight"),
                draft_config.rms_norm_eps,
            )?,
        ));
    }
    Ok(Some(DFlashModel::from_loaded(
        draft_config,
        fc,
        hidden_norm,
        layers,
        final_norm,
    )))
}

const BYTES_PER_MIB: u64 = 1024 * 1024;
const DEFAULT_PAGED_CACHE_MEMORY_MB: u32 = 2048;
const DEFAULT_PAGED_MAX_SEQUENCES: u32 = 8;

fn build_paged_runtime(config: &MuseGlimmerConfig) -> Result<Option<MusePagedRuntime>> {
    if config.use_block_paged_cache == Some(false)
        || !crate::engine::persistence::compiled_forward_backend_available()
    {
        return Ok(None);
    }
    let block_size = config.paged_block_size.unwrap_or(16);
    let max_chunk = u32::try_from(super::model::PREFILL_STEP_SIZE)
        .map_err(|_| Error::from_reason("positive Muse-Glimmer prefill chunk"))?;
    let specs = compute_layer_kv_cache_specs(config, block_size, KVCacheDType::BFloat16)
        .map_err(Error::from_reason)?;
    let groups =
        compute_layer_kv_cache_groups(config, block_size, KVCacheDType::BFloat16, max_chunk)
            .map_err(Error::from_reason)?;
    let max_seq_len = u32::try_from(config.text_config.max_position_embeddings)
        .map_err(|_| Error::from_reason("Muse-Glimmer max_position_embeddings exceeds u32::MAX"))?;
    let bytes_per_block = groups
        .iter()
        .map(|group| {
            2u64.saturating_mul(u64::from(group.physical_layout.num_kv_heads))
                .saturating_mul(u64::from(group.physical_layout.head_size))
                .saturating_mul(u64::from(block_size))
                .saturating_mul(2)
                .saturating_mul(group.physical_layer_indices.len() as u64)
        })
        .collect::<Vec<_>>();
    let required_bytes_for_width = |width: u32| {
        groups
            .iter()
            .zip(&bytes_per_block)
            .fold(0u64, |sum, (group, bytes)| {
                sum.saturating_add(
                    u64::from(group_reserved_blocks(
                        group.attention_kind,
                        group.max_admission_blocks,
                        width,
                    ))
                    .saturating_mul(*bytes),
                )
            })
    };
    let minimum_bytes = required_bytes_for_width(1);
    let default_memory_mb = minimum_bytes
        .div_ceil(BYTES_PER_MIB)
        .max(u64::from(DEFAULT_PAGED_CACHE_MEMORY_MB));
    let memory_mb = config
        .paged_cache_memory_mb
        .unwrap_or_else(|| u32::try_from(default_memory_mb).unwrap_or(u32::MAX));
    let memory_bytes = u64::from(memory_mb).saturating_mul(BYTES_PER_MIB);
    if memory_bytes < minimum_bytes {
        return Err(Error::from_reason(format!(
            "Muse-Glimmer paged_cache_memory_mb={memory_mb} is below the {} MiB hybrid-cache minimum",
            minimum_bytes.div_ceil(BYTES_PER_MIB)
        )));
    }
    let mut max_sequences = DEFAULT_PAGED_MAX_SEQUENCES.min(32);
    while max_sequences > 1 && required_bytes_for_width(max_sequences) > memory_bytes {
        max_sequences -= 1;
    }
    if required_bytes_for_width(max_sequences) > memory_bytes {
        return Err(Error::from_reason(
            "Muse-Glimmer paged cache cannot admit one sequence",
        ));
    }
    let reserved_bytes = required_bytes_for_width(max_sequences);
    let mut unassigned_bytes = memory_bytes.saturating_sub(reserved_bytes);
    let mut adapters = Vec::with_capacity(groups.len());
    let mut total_blocks = 0u32;
    for (group, bytes) in groups.iter().zip(bytes_per_block.iter().copied()) {
        let mut blocks = group_reserved_blocks(
            group.attention_kind,
            group.max_admission_blocks,
            max_sequences,
        );
        if matches!(group.attention_kind, AttentionKind::Full) {
            let extra = u32::try_from(unassigned_bytes / bytes).unwrap_or(u32::MAX);
            blocks = blocks.saturating_add(extra);
            unassigned_bytes =
                unassigned_bytes.saturating_sub(u64::from(extra).saturating_mul(bytes));
        }
        let group_memory_mb = bytes
            .saturating_mul(u64::from(blocks))
            .div_ceil(BYTES_PER_MIB)
            .max(256);
        let pa_config = mlx_paged_attn::PagedAttentionConfig {
            block_size,
            gpu_memory_mb: u32::try_from(group_memory_mb).unwrap_or(u32::MAX),
            head_size: group.physical_layout.head_size,
            num_kv_heads: group.physical_layout.num_kv_heads,
            num_layers: u32::try_from(group.physical_layer_indices.len()).map_err(|_| {
                Error::from_reason("Muse-Glimmer paged physical layer count exceeds u32::MAX")
            })?,
            use_fp8_cache: Some(false),
            max_seq_len: Some(max_seq_len),
            max_batch_size: Some(max_sequences),
        };
        let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
            blocks, blocks, block_size,
        )));
        let pool = mlx_paged_attn::LayerKVPool::new(
            pa_config,
            blocks,
            blocks,
            mlx_paged_attn::metal::MetalDtype::BFloat16,
        )
        .map_err(|error| {
            Error::from_reason(format!(
                "Muse-Glimmer KV group {} pool construction failed: {error}",
                group.group_id
            ))
        })?;
        let adapter = match group.attention_kind {
            AttentionKind::Full => PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size),
            AttentionKind::SlidingWindow { sliding_window } => PagedKVCacheAdapter::new_sliding(
                allocator,
                Arc::new(pool),
                block_size,
                sliding_window,
                max_seq_len,
            ),
        }
        .map_err(Error::from_reason)?;
        total_blocks = total_blocks.saturating_add(blocks);
        adapters.push(adapter);
    }
    let prefill_windows: Vec<PagedWindowSlot> =
        admit_paged_dispatch_plan(&groups, &vec![WindowCarrier::ExplicitMask; groups.len()])
            .map_err(Error::from_reason)?;
    let decode_windows: Vec<PagedWindowSlot> =
        admit_paged_dispatch_plan(&groups, &vec![WindowCarrier::KernelArgument; groups.len()])
            .map_err(Error::from_reason)?;
    let coordinator = crate::models::gemma4::model::Gemma4KVCacheCoordinator::new(
        &specs,
        groups,
        adapters,
        max_sequences,
    )
    .map_err(Error::from_reason)?;
    let routes = coordinator.routes().to_vec();
    tracing::info!(
        "Muse-Glimmer hybrid paged cache enabled: total_blocks={total_blocks}, block_size={block_size}, memory_mb={memory_mb}, max_sequences={max_sequences}"
    );
    Ok(Some(MusePagedRuntime {
        coordinator,
        routes,
        prefill_windows,
        decode_windows,
    }))
}

fn primary_safetensor_is_mlx(path: &Path) -> Result<bool> {
    let primary = if path.join("weights.safetensors").is_file() {
        path.join("weights.safetensors")
    } else if path.join("model.safetensors").is_file() {
        path.join("model.safetensors")
    } else {
        let mut shards = std::fs::read_dir(path)
            .map_err(|error| {
                Error::from_reason(format!(
                    "Failed to inspect Muse-Glimmer checkpoint directory: {error}"
                ))
            })?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|file| {
                let name = file
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default();
                (name.starts_with("model-") || name.starts_with("model.safetensors-"))
                    && name.ends_with(".safetensors")
                    && name.contains("-of-")
            })
            .collect::<Vec<_>>();
        shards.sort();
        shards.into_iter().next().ok_or_else(|| {
            Error::from_reason(format!(
                "No model SafeTensors file found in {}",
                path.display()
            ))
        })?
    };
    Ok(SafeTensorsFile::load(&primary)?
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("format"))
        .and_then(serde_json::Value::as_str)
        == Some("mlx"))
}

/// Declarative half of `canonicalize_target_weights`: the first-match
/// raw-key rewrites that map the checkpoint's `language_model.*` namespaces
/// into the loader's `model.language_model.*` / `lm_head.*` canonical space,
/// plus the duplicate-canonical-key rejection.
static MUSE_CANONICAL_SPEC: RenameSpec<'static> = RenameSpec {
    raw_rules: &[
        KeyRule::RenamePrefix {
            from: "language_model.model.lm_head.",
            to: "lm_head.",
        },
        KeyRule::RenamePrefix {
            from: "language_model.lm_head.",
            to: "lm_head.",
        },
        KeyRule::RenamePrefix {
            from: "language_model.model.",
            to: "model.language_model.",
        },
    ],
    strip_prefixes: &[],
    rules_require_strip: false,
    rules: &[],
    reject_duplicate_keys_as: Some("Muse-Glimmer"),
};

/// Value hook: centered-sandwich norms in a raw (non-MLX) checkpoint are
/// stored in deviation-from-1 form; shift them +1.0 so the direct-convention
/// RMSNorm kernels see final values.
fn shift_centered_sandwich_norms(canonical: &mut HashMap<String, MxArray>) -> Result<()> {
    for (key, value) in canonical.iter_mut() {
        let is_centered_sandwich_norm = key
            .strip_prefix("model.language_model.layers.")
            .is_some_and(|rest| {
                [
                    ".input_layernorm.weight",
                    ".post_attention_layernorm.weight",
                    ".pre_feedforward_layernorm.weight",
                    ".post_feedforward_layernorm.weight",
                ]
                .iter()
                .any(|suffix| rest.ends_with(suffix))
            });
        if is_centered_sandwich_norm {
            *value = value.add_scalar(1.0)?;
        }
    }
    Ok(())
}

fn canonicalize_target_weights(
    params: HashMap<String, MxArray>,
    norms_are_direct: bool,
) -> Result<HashMap<String, MxArray>> {
    let mut canonical = apply_rename_spec(params, &MUSE_CANONICAL_SPEC)?;
    if !norms_are_direct {
        shift_centered_sandwich_norms(&mut canonical)?;
    }
    Ok(canonical)
}

/// With `preload`, `path` is the GGUF's asset directory (no weight files);
/// the draft companion stays in the preload.
fn load_target_weights(
    path: &Path,
    tie_word_embeddings: bool,
    preload: Option<&mut GgufPreload>,
) -> Result<HashMap<String, MxArray>> {
    // Muse-Glimmer execution is text-only today. Keep the optional
    // vision.safetensors sidecar and any vision tensors embedded in the
    // primary shards out of residency accounting and cold-tier materialization
    // until a vision forward path consumes them. Tied checkpoints may also
    // carry a redundant lm_head quantization group; logits use the embedding
    // table, so exclude the whole unused group before the same accounting.
    // The GGUF import writes direct-convention norms (the converter tags that
    // output "format": "mlx"); shifting them again corrupts every layer.
    let norms_are_direct = match preload {
        Some(_) => true,
        None => primary_safetensor_is_mlx(path)?,
    };
    let mut params =
        canonicalize_target_weights(load_checkpoint(path, false, preload)?, norms_are_direct)?;
    params.retain(|key, _| {
        key.starts_with("model.language_model.")
            || (!tie_word_embeddings && key.starts_with("lm_head."))
    });
    Ok(params)
}

fn requires_row_exact_decode_projections(
    top_mode: Option<PerLayerMode>,
    overrides: &HashMap<String, PerLayerQuant>,
) -> bool {
    has_kquant_mode(top_mode, overrides)
        || top_mode == Some(PerLayerMode::Affine)
        || overrides
            .values()
            .any(|quant| quant.mode == PerLayerMode::Affine)
}

fn preload_gguf(input: &PreloadInput) -> Result<GgufPreload> {
    let preload = GgufPreload::load(input)?;
    let stats = &preload.stats;
    tracing::info!(
        "GGUF preload: {} bytes in {:.2} s (read wait {:.2} s, page cache {}), \
         {} GPU-repacked and {} other tensors",
        stats.bytes,
        stats.preload_secs,
        stats.read_secs,
        if stats.nocache { "off" } else { "on" },
        stats.gpu_tensors,
        stats.cpu_tensors,
    );
    Ok(preload)
}

/// With `preload_input`, `path` is the GGUF's asset directory (or a host
/// config overlay of it). Model thread only.
fn load_inner(path: &Path, preload_input: Option<PreloadInput>) -> Result<(MuseGlimmerInner, u64)> {
    let config = MuseGlimmerConfig::from_path(path)?;
    let persist_env = std::env::var("MLX_PERSIST_PAGED_CACHE").ok();
    let persist_cold = resolve_persist_cold(
        "muse_glimmer",
        persist_env.as_deref(),
        config.persist_paged_cache,
    );
    let mut checkpoint_load = CheckpointLoadGuard::before_mmap(path, persist_cold);
    let from_gguf = preload_input.is_some();
    let _free_released = crate::models::quantized_linear::FreeReleasedSources::enable(from_gguf);
    let mut preload = preload_input
        .map(|input| preload_gguf(&input))
        .transpose()?;
    let mut params = load_target_weights(
        path,
        config.text_config.tie_word_embeddings,
        preload.as_mut(),
    )?;
    let preloaded_draft = preload.map(|mut preload| std::mem::take(&mut preload.companion));
    checkpoint_load.record_mmap();
    if persist_cold {
        crate::engine::persistence::prewarm_checkpoint_pages(path);
    }
    let (bits, group_size, top_mode, overrides) = load_quant_settings_from_disk(path, 4, 64)?;
    let default = PerLayerQuant {
        bits,
        group_size,
        mode: top_mode.unwrap_or(PerLayerMode::Affine),
        input_amax: None,
        layout: Default::default(),
    };
    if params.keys().any(|key| key.ends_with(".scales")) && top_mode.is_none() {
        return Err(Error::from_reason(
            "Muse-Glimmer checkpoint has packed weights but config.json has no quantization mode",
        ));
    }

    let text = &config.text_config;
    let embed_tokens = load_embedding(
        &params,
        "model.language_model.embed_tokens",
        text.vocab_size,
        text.hidden_size,
        default,
        &overrides,
    )?;
    let lm_head = load_lm_head(&params, text.tie_word_embeddings, default, &overrides)?;
    let final_norm = norm(
        &params,
        "model.language_model.norm.weight",
        text.rms_norm_eps,
    )?;

    // Checkpoint byte total BEFORE installation: the layer loop releases the
    // row-major sources of Tiled64 K-quant projections from `params` (the
    // model owns same-sized tiled copies), so the map afterwards under-counts.
    let text_weight_bytes = params
        .values()
        .map(|array| array.nbytes() as u64)
        .fold(0u64, |acc, v| acc.saturating_add(v));
    let mut tiled = Vec::<String>::new();
    let mut layers = Vec::with_capacity(text.num_hidden_layers);
    for index in 0..text.num_hidden_layers {
        let base = format!("model.language_model.layers.{index}");
        let attn = format!("{base}.self_attn");
        let mut proj = |prefix: String| -> Result<LinearProj> {
            build_projection_tiled(&params, &prefix, default, &overrides, &mut tiled)
        };
        let attention = MuseGlimmerAttention::from_projections(
            text,
            index,
            config.rope_traditional,
            proj(format!("{attn}.q_proj"))?,
            proj(format!("{attn}.k_proj"))?,
            proj(format!("{attn}.v_proj"))?,
            proj(format!("{attn}.o_proj"))?,
            proj(format!("{attn}.gate_proj"))?,
        )?;
        let mlp_base = format!("{base}.mlp");
        let mlp = MuseGlimmerMlp::new(
            proj(format!("{mlp_base}.gate_proj"))?,
            proj(format!("{mlp_base}.up_proj"))?,
            proj(format!("{mlp_base}.down_proj"))?,
        );
        // The layer owns its tiled projections now; drop the map's row-major
        // originals (mirrors the dense qwen3_5 loader).
        crate::models::quantized_linear::release_tiled_kquant_sources(&mut params, &mut tiled);
        layers.push(MuseGlimmerDecoderLayer::new(
            attention,
            mlp,
            norm(
                &params,
                &format!("{base}.input_layernorm.weight"),
                text.rms_norm_eps,
            )?,
            norm(
                &params,
                &format!("{base}.post_attention_layernorm.weight"),
                text.post_norm_eps,
            )?,
            norm(
                &params,
                &format!("{base}.pre_feedforward_layernorm.weight"),
                text.rms_norm_eps,
            )?,
            norm(
                &params,
                &format!("{base}.post_feedforward_layernorm.weight"),
                text.post_norm_eps,
            )?,
        ));
    }

    let tokenizer_path = path.join("tokenizer.json");
    let tokenizer = Arc::new(Qwen3Tokenizer::from_file(&tokenizer_path).map_err(|error| {
        Error::from_reason(format!("Failed to load Muse-Glimmer tokenizer: {error}"))
    })?);
    let response_template = ResponseTemplate::from_tokenizer_config(path)?;
    let gen_defaults = parse_generation_defaults(path);
    let draft = draft_params(path, preloaded_draft)?;
    let draft_bytes = match (&draft, from_gguf) {
        (Some(params), true) => params
            .values()
            .map(|array| array.nbytes() as u64)
            .fold(0u64, |acc, v| acc.saturating_add(v)),
        (Some(_), false) => std::fs::metadata(path.join("draft.safetensors"))
            .map(|meta| meta.len())
            .unwrap_or(0),
        (None, _) => 0,
    };
    let dflash = load_dflash(&config, draft, default, &overrides)?;
    let paged = build_paged_runtime(&config)?;
    let weight_bytes = text_weight_bytes.saturating_add(draft_bytes);
    let weights_resident = if persist_cold {
        let arrays = params.values().collect::<Vec<_>>();
        Some(crate::array::memory::materialize_weights(&arrays)?)
    } else {
        None
    };
    let mut inner = MuseGlimmerInner::from_loaded(
        config,
        embed_tokens,
        layers,
        final_norm,
        lm_head,
        tokenizer,
        response_template,
        gen_defaults,
        dflash,
        paged,
    );
    inner.row_exact_decode_projections =
        requires_row_exact_decode_projections(top_mode, &overrides);
    if let Some(weights_resident) = weights_resident.as_ref()
        && let Some(context) =
            inner.build_cold_tier_context(&path.to_string_lossy(), weights_resident)
    {
        if checkpoint_load.stable_after_fingerprint() {
            inner.attach_cold_tier(context, weights_resident);
        } else {
            tracing::warn!(
                "cold-tier persistence disabled for {}: model directory changed during load; KV persistence stays off",
                path.display()
            );
        }
    }
    Ok((inner, weight_bytes))
}

pub(crate) async fn load_with_thread(model_path: &str) -> Result<MuseGlimmerModel> {
    let source =
        CheckpointSource::resolve(Path::new(model_path), NativeGgufFamily::MuseGlimmer).await?;
    let loaded = load_with_source(&source).await;
    if loaded.is_err() {
        source.release_lease();
    }
    loaded
}

/// The caller releases `source`'s lease when this fails.
async fn load_with_source(source: &CheckpointSource) -> Result<MuseGlimmerModel> {
    let model_path = source.asset_dir().to_string_lossy().into_owned();
    let preload_input = match source {
        CheckpointSource::Dir(_) => None,
        CheckpointSource::Gguf(checkpoint) => {
            checkpoint.check_asset_config()?;
            Some(checkpoint.preload_input(true))
        }
    };
    let model_assets_path = model_path.clone();
    let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_scheduler(
        move || {
            let (inner, weight_bytes) = load_inner(Path::new(&model_path), preload_input)?;
            let has_dflash = inner.dflash.is_some();
            let pool_bytes = inner
                .paged
                .as_ref()
                .map(|paged| paged.coordinator.pool_allocated_bytes())
                .transpose()
                .map_err(Error::from_reason)?
                .unwrap_or(0);
            let paged_active = inner.paged.is_some();
            let context_limits = super::model::MuseGlimmerContextLimits::from_inner(&inner);
            let max_concurrent_sequences =
                crate::engine::hybrid_scheduler::scheduler_max_num_seqs_for(
                    inner.paged.as_ref().map_or(1, |paged| {
                        paged.coordinator.max_concurrent_sequences() as usize
                    }),
                ) as u32;
            let guard = crate::cache_limit::coordinator().register(weight_bytes);
            let pool_guard = (pool_bytes != 0)
                .then(|| crate::cache_limit::coordinator().register_pool(pool_bytes));
            Ok((
                MuseSchedulerState::new(inner)?,
                (
                    has_dflash,
                    paged_active,
                    max_concurrent_sequences,
                    context_limits,
                    guard,
                    pool_guard,
                ),
            ))
        },
        |state, receiver| state.drive(receiver),
    );
    let (
        has_dflash,
        paged_active,
        max_concurrent_sequences,
        context_limits,
        cache_limit_guard,
        pool_cache_limit_guard,
    ) = match init_rx.await {
        Ok(Ok(init)) => init,
        Ok(Err(error)) => return Err(error),
        Err(_) => {
            return Err(Error::from_reason(
                "Muse-Glimmer model thread exited during load",
            ));
        }
    };
    Ok(MuseGlimmerModel {
        thread,
        model_assets_path,
        has_dflash,
        paged_active,
        max_concurrent_sequences,
        context_limits,
        _cache_limit_guard: cache_limit_guard,
        _pool_cache_limit_guard: pool_cache_limit_guard,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        canonicalize_target_weights, load_lm_head, load_target_weights, muse_projection_quant,
        primary_safetensor_is_mlx, quant_lookup_prefix, requires_row_exact_decode_projections,
    };
    use crate::array::MxArray;
    use crate::models::gemma4::quantized_linear::{PerLayerMode, PerLayerQuant};
    use crate::utils::safetensors::save_safetensors;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn save_one(path: &std::path::Path, key: &str, value: f32) {
        let mut tensors = HashMap::from([(
            key.to_string(),
            MxArray::from_float32(&[value], &[1]).expect("create test tensor"),
        )]);
        save_safetensors(path, &mut tensors, None).expect("save test safetensors");
    }

    #[test]
    fn target_and_dflash_quantization_namespaces_stay_distinct() {
        assert_eq!(
            quant_lookup_prefix("model.language_model.layers.0.self_attn.q_proj"),
            "language_model.layers.0.self_attn.q_proj"
        );
        assert_eq!(
            quant_lookup_prefix("layers.0.self_attn.q_proj"),
            "layers.0.self_attn.q_proj"
        );
        assert_eq!(
            quant_lookup_prefix("model.language_model.embed_tokens"),
            "language_model.embed_tokens"
        );
        assert_eq!(quant_lookup_prefix("lm_head"), "lm_head");

        let fallback = PerLayerQuant {
            bits: 6,
            group_size: 16,
            mode: PerLayerMode::Q6K,
            input_amax: None,
            layout: Default::default(),
        };
        let overrides = HashMap::from([(
            "layers.0.self_attn.q_proj".to_string(),
            PerLayerQuant {
                bits: 4,
                group_size: 32,
                mode: PerLayerMode::Q4K,
                input_amax: None,
                layout: Default::default(),
            },
        )]);
        assert_eq!(
            muse_projection_quant(
                "model.language_model.layers.0.self_attn.q_proj",
                &overrides,
                fallback
            )
            .mode,
            PerLayerMode::Q4K,
            "SafeTensors-converter overrides use the bare target namespace"
        );
    }

    #[test]
    fn converter_namespace_and_centered_hf_norms_reach_one_runtime_form() {
        let converter = HashMap::from([
            (
                "language_model.model.layers.0.input_layernorm.weight".to_string(),
                MxArray::from_float32(&[1.25], &[1]).expect("direct norm"),
            ),
            (
                "language_model.model.lm_head.weight".to_string(),
                MxArray::from_float32(&[2.0], &[1]).expect("lm head"),
            ),
        ]);
        let converter = canonicalize_target_weights(converter, true).expect("converter namespace");
        assert_eq!(
            converter["model.language_model.layers.0.input_layernorm.weight"]
                .item_at_float32(0)
                .expect("direct value"),
            1.25
        );
        assert!(converter.contains_key("lm_head.weight"));

        let mut raw_hf = HashMap::from([(
            "model.language_model.norm.weight".to_string(),
            MxArray::from_float32(&[0.5], &[1]).expect("final norm"),
        )]);
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            raw_hf.insert(
                format!("model.language_model.layers.0.{norm}.weight"),
                MxArray::from_float32(&[0.25], &[1]).expect("centered norm"),
            );
        }
        let raw_hf = canonicalize_target_weights(raw_hf, false).expect("raw HF namespace");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            assert_eq!(
                raw_hf[&format!("model.language_model.layers.0.{norm}.weight")]
                    .item_at_float32(0)
                    .expect("shifted sandwich norm"),
                1.25
            );
        }
        assert_eq!(
            raw_hf["model.language_model.norm.weight"]
                .item_at_float32(0)
                .expect("unchanged final norm"),
            0.5
        );
    }

    #[test]
    fn mlx_metadata_marks_direct_norm_weights() {
        let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "muse_glimmer_norm_metadata_{}_{}",
            std::process::id(),
            id
        ));
        std::fs::create_dir_all(&dir).expect("create model directory");
        let mut tensors = HashMap::from([(
            "weight".to_string(),
            MxArray::from_float32(&[1.0], &[1]).expect("weight"),
        )]);
        save_safetensors(
            dir.join("model.safetensors"),
            &mut tensors,
            Some(serde_json::json!({"format": "mlx"})),
        )
        .expect("save MLX checkpoint");
        assert!(primary_safetensor_is_mlx(&dir).expect("read metadata"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn text_runtime_excludes_unused_weights_before_residency() {
        let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "muse_glimmer_text_weights_{}_{}",
            std::process::id(),
            id
        ));
        std::fs::create_dir_all(&dir).expect("create model directory");
        let mut primary = HashMap::from([
            (
                "language_model.model.embed_tokens.weight".to_string(),
                MxArray::from_float32(&[1.0], &[1]).expect("create text tensor"),
            ),
            (
                "vision_tower.layers.0.attn.q_proj.weight".to_string(),
                MxArray::from_float32(&[2.0], &[1]).expect("create vision tensor"),
            ),
            (
                "vision_adapter.fc1.weight".to_string(),
                MxArray::from_float32(&[3.0], &[1]).expect("create adapter tensor"),
            ),
            (
                "language_model.model.lm_head.weight".to_string(),
                MxArray::from_float32(&[4.0], &[1]).expect("create lm head weight"),
            ),
            (
                "language_model.model.lm_head.scales".to_string(),
                MxArray::from_float32(&[5.0], &[1]).expect("create lm head scales"),
            ),
            (
                "language_model.model.lm_head.biases".to_string(),
                MxArray::from_float32(&[6.0], &[1]).expect("create lm head biases"),
            ),
        ]);
        save_safetensors(
            dir.join("model.safetensors"),
            &mut primary,
            Some(serde_json::json!({"format": "mlx"})),
        )
        .expect("save mixed Muse safetensors");
        save_one(&dir.join("vision.safetensors"), "vision.weight", 2.0);

        let untied = load_target_weights(&dir, false, None).expect("load untied Muse weights");
        assert!(untied.contains_key("model.language_model.embed_tokens.weight"));
        assert!(untied.contains_key("lm_head.weight"));
        assert!(untied.contains_key("lm_head.scales"));
        assert!(untied.contains_key("lm_head.biases"));
        assert!(!untied.contains_key("vision_tower.layers.0.attn.q_proj.weight"));
        assert!(!untied.contains_key("vision_adapter.fc1.weight"));
        assert!(!untied.contains_key("vision.weight"));

        let tied = load_target_weights(&dir, true, None).expect("load tied Muse weights");
        assert!(tied.contains_key("model.language_model.embed_tokens.weight"));
        assert!(!tied.keys().any(|key| key.starts_with("lm_head.")));
        assert!(!tied.contains_key("vision_tower.layers.0.attn.q_proj.weight"));
        assert!(!tied.contains_key("vision_adapter.fc1.weight"));
        assert!(!tied.contains_key("vision.weight"));

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn packed_checkpoints_enable_row_exact_concurrent_decode() {
        let mut overrides = HashMap::new();
        assert!(requires_row_exact_decode_projections(
            Some(PerLayerMode::Q4K),
            &overrides
        ));
        assert!(requires_row_exact_decode_projections(
            Some(PerLayerMode::Affine),
            &overrides
        ));
        assert!(!requires_row_exact_decode_projections(
            Some(PerLayerMode::Mxfp4),
            &overrides
        ));

        overrides.insert(
            "language_model.layers.0.self_attn.q_proj".to_string(),
            PerLayerQuant {
                bits: 5,
                group_size: 32,
                mode: PerLayerMode::Q5K,
                input_amax: None,
                layout: Default::default(),
            },
        );
        assert!(requires_row_exact_decode_projections(None, &overrides));
    }

    #[test]
    fn tied_embeddings_do_not_require_an_lm_head_tensor() {
        let params = HashMap::<String, MxArray>::new();
        let overrides = HashMap::new();
        let default = PerLayerQuant {
            bits: 4,
            group_size: 64,
            mode: PerLayerMode::Affine,
            input_amax: None,
            layout: Default::default(),
        };

        assert!(
            load_lm_head(&params, true, default, &overrides)
                .expect("tied embedding head")
                .is_none()
        );
        let error = match load_lm_head(&params, false, default, &overrides) {
            Ok(_) => panic!("untied checkpoint must still require lm_head.weight"),
            Err(error) => error,
        };
        assert!(error.reason.contains("lm_head.weight"), "{error}");
    }

    async fn resolve_muse_gguf(
        gguf: &std::path::Path,
        cache: &std::path::Path,
    ) -> crate::engine::checkpoint_source::CheckpointSource {
        use crate::engine::checkpoint_source::CheckpointSource;
        let source = CheckpointSource::resolve_in(
            gguf,
            crate::utils::gguf::NativeGgufFamily::MuseGlimmer,
            Some(cache),
        )
        .await
        .unwrap();
        let CheckpointSource::Gguf(ckpt) = &source else {
            panic!("the Muse GGUF resolved to a plain directory");
        };
        ckpt.check_asset_config().unwrap();
        let weight_files: Vec<_> = std::fs::read_dir(&ckpt.asset_dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".safetensors"))
            .collect();
        assert!(
            weight_files.is_empty(),
            "the asset directory holds weight files: {weight_files:?}"
        );
        source
    }

    fn gguf_checkpoint(
        source: &crate::engine::checkpoint_source::CheckpointSource,
    ) -> &crate::engine::checkpoint_source::GgufCheckpoint {
        match source {
            crate::engine::checkpoint_source::CheckpointSource::Gguf(ckpt) => ckpt,
            crate::engine::checkpoint_source::CheckpointSource::Dir(_) => unreachable!(),
        }
    }

    #[tokio::test]
    async fn muse_norms_direct_for_gguf() {
        use crate::models::muse_glimmer::config::MuseGlimmerConfig;
        use crate::utils::gguf_import::GgufPreload;
        use crate::utils::gguf_import::gate::manifest;
        use crate::utils::gguf_test_fixtures::{prepare_import_fixture, write_import_fixture};

        let work = std::env::temp_dir().join(format!("mlx-muse-norms-{}", uuid::Uuid::new_v4()));
        let fx = write_import_fixture("muse-draft", &work.join("src"));
        let converted = prepare_import_fixture(&fx, &work.join("old-cache")).await;
        let tie = MuseGlimmerConfig::from_path(&converted)
            .unwrap()
            .text_config
            .tie_word_embeddings;
        let want = manifest(&load_target_weights(&converted, tie, None).unwrap());
        assert!(
            want.keys()
                .any(|key| key.starts_with("model.language_model.layers.")
                    && key.ends_with("layernorm.weight")),
            "the fixture carries no sandwich norm: {:?}",
            want.keys()
        );

        let source = resolve_muse_gguf(&fx.main, &work.join("cache")).await;
        let ckpt = gguf_checkpoint(&source);
        let mut preload = GgufPreload::load(&ckpt.preload_input(true)).unwrap();
        let got = manifest(&load_target_weights(&ckpt.asset_dir, tie, Some(&mut preload)).unwrap());
        assert!(
            !preload.companion.is_empty(),
            "the target load consumed the draft companion"
        );
        assert_eq!(
            got.keys().collect::<Vec<_>>(),
            want.keys().collect::<Vec<_>>(),
            "tensor names"
        );
        for (name, entry) in &want {
            assert_eq!(
                got.get(name),
                Some(entry),
                "tensor {name} differs from the converted checkpoint"
            );
        }
        source.release_lease();
        std::fs::remove_dir_all(&work).ok();
    }

    #[tokio::test]
    async fn muse_draft_from_preload() {
        use super::draft_params;
        use crate::models::muse_glimmer::config::MuseGlimmerConfig;
        use crate::utils::gguf_import::GgufPreload;
        use crate::utils::gguf_import::gate::manifest;
        use crate::utils::gguf_test_fixtures::{prepare_import_fixture, write_import_fixture};

        let work = std::env::temp_dir().join(format!("mlx-muse-draft-{}", uuid::Uuid::new_v4()));
        let fx = write_import_fixture("muse-draft", &work.join("src"));
        let converted = prepare_import_fixture(&fx, &work.join("old-cache")).await;
        let want = manifest(
            &draft_params(&converted, None)
                .unwrap()
                .expect("the converted entry has no draft.safetensors"),
        );
        assert!(!want.is_empty());

        let source = resolve_muse_gguf(&fx.main, &work.join("cache")).await;
        let ckpt = gguf_checkpoint(&source);
        assert!(ckpt.companion.is_some(), "the DFlash sibling was not found");
        let tie = MuseGlimmerConfig::from_path(&ckpt.asset_dir)
            .unwrap()
            .text_config
            .tie_word_embeddings;
        let mut preload = GgufPreload::load(&ckpt.preload_input(true)).unwrap();
        load_target_weights(&ckpt.asset_dir, tie, Some(&mut preload)).unwrap();
        let draft = draft_params(
            &ckpt.asset_dir,
            Some(std::mem::take(&mut preload.companion)),
        )
        .unwrap();
        assert!(
            draft.is_some(),
            "the DFlash draft was not taken from the preload companion"
        );
        let got = manifest(&draft.unwrap());
        assert_eq!(
            got.keys().collect::<Vec<_>>(),
            want.keys().collect::<Vec<_>>(),
            "draft tensor names"
        );
        for (name, entry) in &want {
            assert_eq!(got.get(name), Some(entry), "draft tensor {name} differs");
        }
        source.release_lease();

        let lone = write_import_fixture("muse-draft", &work.join("lone"));
        std::fs::remove_file(lone.companion.as_ref().unwrap()).unwrap();
        let source = resolve_muse_gguf(&lone.main, &work.join("cache")).await;
        let ckpt = gguf_checkpoint(&source);
        assert!(ckpt.companion.is_none());
        let mut preload = GgufPreload::load(&ckpt.preload_input(true)).unwrap();
        load_target_weights(&ckpt.asset_dir, tie, Some(&mut preload)).unwrap();
        assert!(
            draft_params(
                &ckpt.asset_dir,
                Some(std::mem::take(&mut preload.companion))
            )
            .unwrap()
            .is_none(),
            "a GGUF without a DFlash companion produced a draft"
        );
        source.release_lease();
        std::fs::remove_dir_all(&work).ok();
    }
}
