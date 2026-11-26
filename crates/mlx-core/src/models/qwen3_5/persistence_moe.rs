//! MoE-specific weight sanitization, residual-dtype alignment, config parsing,
//! and model-thread loaders backed by unified [`Qwen35Inner`].

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use napi::bindgen_prelude::*;
use serde_json::Value;
use tracing::{info, warn};

use crate::array::{DType, MxArray};
use crate::cold_tier::{CheckpointFiles, CheckpointLoadGuard};
use crate::engine::checkpoint_source::CheckpointSource;
use crate::engine::persistence::{
    CheckpointLeaseGuard, GgufPreloadGuard, attach_pool_growth_notifier, get_config_bool,
    get_config_f64, get_config_i32, load_checkpoint, load_gguf_preload, params_weight_bytes,
    prewarm_checkpoint_pages, register_loaded_weights_and_pool, register_thread_memory_waker,
    spawn_loaded_model_thread,
};
use crate::models::mtp_drafter::{DrafterBodyVariant, MTP_MOE_LAYER_LINEAR_SUFFIXES};
use crate::models::paged_config::PagedCacheConfig;
#[cfg(test)]
use crate::models::quant_dispatch::PlainFp8Residency;
use crate::models::quant_dispatch::{
    default_per_layer_quant, defer_plain_fp8_materialization, has_kquant_mode, has_sym8_mode,
    parse_quant_settings, select_quantization_block,
};
use crate::models::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, GATE_QUANT_BITS, GATE_QUANT_GROUP_SIZE,
    PerLayerMode, PerLayerQuant, is_quantized_checkpoint,
};
use crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig;
use crate::models::qwen3_5_moe::model::Qwen3_5MoeModel;
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use crate::utils::gguf_import::PreloadInput;
use crate::vision::qwen::encoder::QwenVisionEncoder;
use crate::vision::qwen::processing::QwenImageProcessor;
use crate::vision::qwen::weights::{load_vision_weights, parse_vision_config};

use super::model::{Qwen35Inner, qwen35_dense_vision_active};
use super::mtp::Qwen3_5MTPModule;
use super::persistence::{
    MTP_LAYER_LINEAR_SUFFIXES, apply_weights_inner_with_residency,
    augment_mtplx_mtp_quantization_with_suffixes, compute_dtype_of, embedding_compute_dtype,
    is_qwen35_mtp_norm_key, is_qwen35_norm_key, merge_split_projections, params_dtype,
    sanitize_qwen35_common, shift_norm_plus_one, split_qwen35_vision_text_params,
    transpose_conv1d_if_needed,
};

/// Sanitize weights from HuggingFace format (MoE variant).
pub(crate) fn sanitize_weights_moe(
    params: HashMap<String, MxArray>,
    config: &Qwen3_5MoeConfig,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
) -> Result<HashMap<String, MxArray>> {
    let (renamed, needs_norm_fix, mtp_norms_need_shift) = sanitize_qwen35_common(
        params,
        config.tie_word_embeddings,
        config.n_mtp_layers,
        "Qwen3.5-MoE",
        false,
    )?;

    let has_individual_experts = renamed
        .keys()
        .any(|k| k.contains(".mlp.experts.0.up_proj.weight"));

    let mut result = normalize_moe_weight_map(
        renamed,
        has_individual_experts,
        needs_norm_fix,
        mtp_norms_need_shift,
        config.num_experts,
    )?;

    merge_split_projections(&mut result, per_layer_quant)?;

    Ok(result)
}

/// Split a MoE expert tensor key into its known suffix and whether that
/// suffix is an outer-scale sidecar (`.scale` / `.global_scale` /
/// `.weight_scale_2` / `.weight_global_scale`).
fn split_moe_expert_suffix(name: &str) -> (&'static str, bool) {
    if name.ends_with(".scales") {
        (".scales", false)
    } else if name.ends_with(".biases") {
        (".biases", false)
    } else if name.ends_with(".scale") {
        (".scale", true)
    } else if name.ends_with(".global_scale") {
        (".global_scale", true)
    } else if name.ends_with(".weight_scale_2") {
        (".weight_scale_2", true)
    } else if name.ends_with(".weight_global_scale") {
        (".weight_global_scale", true)
    } else {
        (".weight", false)
    }
}

/// Pure weight-map transform shared by the checkpoint loader and its unit
/// tests: handles MTP bypass (conv1d + MTP-norm shift only), per-expert weight
/// collection and stacking, fused `gate_up_proj` splitting,
/// `experts.down_proj` → `switch_mlp.down_proj` renames, conv1d axis fix, and
/// the LM-body +1.0 norm shift.
fn normalize_moe_weight_map(
    params: HashMap<String, MxArray>,
    has_individual_experts: bool,
    needs_norm_fix: bool,
    mtp_norms_need_shift: bool,
    num_experts: i32,
) -> Result<HashMap<String, MxArray>> {
    let mut result: HashMap<String, MxArray> = HashMap::new();
    let mut expert_weights: HashMap<String, Vec<(usize, MxArray)>> = HashMap::new();

    for (name, array) in params {
        if name.starts_with("mtp.") && !name.contains(".mlp.experts.") {
            let array = if name.contains("conv1d.weight") {
                transpose_conv1d_if_needed(&name, array)?
            } else if mtp_norms_need_shift && is_qwen35_mtp_norm_key(&name) {
                shift_norm_plus_one(array, false)?
            } else {
                array
            };
            result.insert(name, array);
            continue;
        }

        if let Some((prefix, rest)) = name
            .split_once(".mlp.experts.")
            .filter(|_| has_individual_experts)
        {
            let rest_parts: Vec<&str> = rest.split('.').collect();
            if rest_parts.len() >= 3 {
                let expert_idx: usize = rest_parts[0].parse().map_err(|e| {
                    Error::from_reason(format!(
                        "Failed to parse expert index from weight '{}': {}",
                        name, e
                    ))
                })?;
                let proj_name = rest_parts[1];
                let (suffix_dot, _) = split_moe_expert_suffix(&name);
                let suffix = &suffix_dot[1..];
                let key = format!("{}.mlp.switch_mlp.{}.{}", prefix, proj_name, suffix);
                expert_weights
                    .entry(key)
                    .or_default()
                    .push((expert_idx, array));
                continue;
            }
        }

        if name.contains(".mlp.experts.gate_up_proj")
            || name.contains(".mlp.switch_mlp.gate_up_proj")
        {
            let (suffix, is_outer_scale) = split_moe_expert_suffix(&name);
            let name_stripped = name.strip_suffix(suffix).unwrap_or(&name);
            let base = if name_stripped.contains("switch_mlp") {
                format!(
                    "{}{}",
                    name_stripped.replace("gate_up_proj", "gate_proj"),
                    suffix
                )
            } else {
                format!(
                    "{}{}",
                    name_stripped.replace("experts.gate_up_proj", "switch_mlp.gate_proj"),
                    suffix
                )
            };
            let up_name = base.replace("gate_proj", "up_proj");

            if is_outer_scale {
                // Per-expert / scalar outer scale applies identically to both halves.
                result.insert(base, array.clone());
                result.insert(up_name, array);
                continue;
            }

            let shape = array.shape()?;
            if shape.len() >= 2 {
                let split_axis = shape.len() - 2;
                let mid = shape[split_axis] / 2;
                let gate = array.slice_axis(split_axis, 0, mid)?;
                let up = array.slice_axis(split_axis, mid, shape[split_axis])?;

                result.insert(base, gate);
                result.insert(up_name, up);
                continue;
            } else {
                let shape_vec: Vec<i64> = shape.to_vec();
                warn!(
                    "gate_up_proj tensor '{}' has unexpected shape {:?}, skipping split",
                    name, shape_vec
                );
            }
        }

        let name = if name.contains(".mlp.experts.down_proj") {
            let (suffix, _) = split_moe_expert_suffix(&name);
            let stripped = name.strip_suffix(suffix).unwrap_or(&name);
            format!(
                "{}{}",
                stripped.replace(".mlp.experts.down_proj", ".mlp.switch_mlp.down_proj"),
                suffix
            )
        } else {
            name
        };

        let array = transpose_conv1d_if_needed(&name, array)?;

        let array = if needs_norm_fix && is_qwen35_norm_key(&name) {
            shift_norm_plus_one(array, false)?
        } else {
            array
        };

        result.insert(name, array);
    }

    if !expert_weights.is_empty() {
        let num_experts = num_experts as usize;
        for (key, mut experts) in expert_weights {
            experts.sort_by_key(|(idx, _)| *idx);

            if num_experts > 0 && experts.len() != num_experts {
                return Err(Error::from_reason(format!(
                    "Expected {} experts for {}, got {}",
                    num_experts,
                    key,
                    experts.len()
                )));
            }

            let arrays: Vec<&MxArray> = experts.iter().map(|(_, a)| a).collect();
            let stacked = MxArray::stack(arrays, Some(0))?;
            result.insert(key, stacked);
        }
    }

    Ok(result)
}

/// The keys [`normalize_moe_weight_map`] gives the renamed `mtp.*` key `name`,
/// from the name alone.
pub(crate) fn mtp_expert_keys(name: &str, has_individual_experts: bool) -> Vec<String> {
    let Some((prefix, rest)) = name
        .split_once(".mlp.experts.")
        .filter(|_| name.starts_with("mtp."))
    else {
        return vec![name.to_string()];
    };
    let tensor = |base: &str| {
        let (suffix_dot, _) = split_moe_expert_suffix(name);
        let suffix = &suffix_dot[1..];
        format!("{prefix}.mlp.switch_mlp.{base}.{suffix}")
    };
    let mut parts = rest.split('.');
    if has_individual_experts
        && let (Some(_), Some(proj), Some(_)) = (parts.next(), parts.next(), parts.next())
    {
        return vec![tensor(proj)];
    }
    if rest.starts_with("gate_up_proj") {
        return vec![tensor("gate_proj"), tensor("up_proj")];
    }
    if rest.starts_with("down_proj") {
        return vec![tensor("down_proj")];
    }
    vec![name.to_string()]
}

/// Router gate fallback PLQ for a MoE checkpoint given its resolved default
/// quantization mode.
pub(crate) fn compute_moe_gate_default(default_mode: PerLayerMode) -> PerLayerQuant {
    let default_gate_mode = if matches!(default_mode, PerLayerMode::Mxfp8) {
        PerLayerMode::Mxfp8
    } else {
        PerLayerMode::Affine
    };
    let default_gate_group_size = if matches!(default_gate_mode, PerLayerMode::Mxfp8) {
        32
    } else {
        GATE_QUANT_GROUP_SIZE
    };
    default_per_layer_quant(GATE_QUANT_BITS, default_gate_group_size, default_gate_mode)
}

/// Test helper matching the historical `apply_weights_moe_inner` signature.
#[cfg(test)]
pub(crate) fn apply_weights_moe_inner(
    inner: &mut Qwen35Inner,
    params: &HashMap<String, MxArray>,
    config: &Qwen3_5MoeConfig,
    quant_bits: i32,
    quant_group_size: i32,
    top_level_mode: Option<PerLayerMode>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    has_vision: bool,
) -> Result<()> {
    let mut params = params.clone();
    apply_weights_moe_inner_with_residency(
        inner,
        &mut params,
        config,
        quant_bits,
        quant_group_size,
        top_level_mode,
        per_layer_quant,
        has_vision,
    )
    .map(drop)
}

/// Test helper matching the historical `apply_weights_moe_inner_with_residency`
/// signature: counts pre-release text bytes and installs weights into `inner`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_weights_moe_inner_with_residency(
    inner: &mut Qwen35Inner,
    params: &mut HashMap<String, MxArray>,
    _config: &Qwen3_5MoeConfig,
    quant_bits: i32,
    quant_group_size: i32,
    top_level_mode: Option<PerLayerMode>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    has_vision: bool,
) -> Result<(PlainFp8Residency, u64)> {
    let text_weight_bytes = params_weight_bytes(params);
    let base_config = inner.config.clone();
    let (plain_fp8_residency, _) = apply_weights_inner_with_residency(
        inner,
        params,
        &base_config,
        quant_bits,
        quant_group_size,
        top_level_mode,
        per_layer_quant,
        has_vision,
        None,
        None,
        None,
        false,
    )?;
    Ok((plain_fp8_residency, text_weight_bytes))
}

/// Load the vision encoder onto a MoE `Qwen35Inner` when the checkpoint ships
/// one — unless the checkpoint is sym8.
fn load_vision_encoder_moe(
    inner: &mut Qwen35Inner,
    vision_params: Option<HashMap<String, MxArray>>,
    raw: &Value,
    config: &Qwen3_5MoeConfig,
    top_level_mode: Option<PerLayerMode>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
) -> Result<Option<HashMap<String, MxArray>>> {
    let vision_params = if has_sym8_mode(top_level_mode, per_layer_quant) {
        if vision_params.is_some() {
            warn!(
                "Qwen3.5-MoE: sym8 checkpoint ships a vision tower, but sym8 \
                 v1 is text-only — skipping vision encoder load (image turns \
                 will be rejected)."
            );
        }
        None
    } else {
        vision_params
    };

    if let Some(ref vparams) = vision_params {
        let vision_config = parse_vision_config(raw);
        info!(
            "Vision config: {} layers, hidden={}, heads={}, patch={}",
            vision_config.num_layers,
            vision_config.hidden_size,
            vision_config.num_heads,
            vision_config.patch_size,
        );

        let mut vision_encoder = QwenVisionEncoder::new(vision_config.clone())?;
        load_vision_weights(&mut vision_encoder, vparams, &vision_config)?;

        inner.init_mrope_layers(
            vec![11, 11, 10],
            config.rope_theta,
            config.max_position_embeddings,
        )?;

        inner.set_vision_encoder(vision_encoder)?;
        inner.set_image_processor(QwenImageProcessor::new(None));
        inner.set_spatial_merge_size(vision_config.spatial_merge_size);

        info!("Qwen3.5 MoE-VL model loaded successfully (with vision encoder)");
    } else {
        info!("Qwen3.5 MoE model loaded successfully");
    }

    Ok(vision_params)
}

fn declared_residual_dtype(raw: &Value) -> Option<DType> {
    for level in [raw.get("text_config"), Some(raw)].into_iter().flatten() {
        let Some(declared) = level
            .get("dtype")
            .or_else(|| level.get("torch_dtype"))
            .and_then(Value::as_str)
        else {
            continue;
        };
        match declared.to_ascii_lowercase().as_str() {
            "bfloat16" | "bf16" => return Some(DType::BFloat16),
            "float16" | "f16" | "half" => return Some(DType::Float16),
            _ => {}
        }
    }
    None
}

pub(crate) fn residual_stream_dtype(
    raw: &Value,
    params: &HashMap<String, MxArray>,
) -> Option<DType> {
    residual_dtype_of(raw, &params_dtype(params))
}

fn residual_dtype_of(raw: &Value, dtype_of: &dyn Fn(&str) -> Option<DType>) -> Option<DType> {
    ["layers.0.input_layernorm.weight", "final_norm.weight"]
        .iter()
        .find_map(|key| dtype_of(key))
        .or_else(|| declared_residual_dtype(raw))
}

fn affine_embedding_cast(
    dtype_of: &dyn Fn(&str) -> Option<DType>,
    target: Option<DType>,
) -> Option<DType> {
    let target = target.filter(|t| matches!(t, DType::BFloat16 | DType::Float16))?;
    for suffix in ["scales", "biases"] {
        if dtype_of(&format!("embedding.{suffix}"))
            .is_some_and(|dtype| !matches!(dtype, DType::Float16 | DType::Float32))
        {
            return None;
        }
    }
    (dtype_of("embedding.scales")? != target).then_some(target)
}

/// The compute dtype [`settle_compute_dtype`] leaves the load with, over
/// `dtype_of`, a sanitized key's dtype before alignment.
pub(crate) fn settled_compute_dtype(
    raw: &Value,
    dtype_of: &dyn Fn(&str) -> Option<DType>,
) -> DType {
    affine_embedding_cast(dtype_of, residual_dtype_of(raw, dtype_of))
        .unwrap_or_else(|| compute_dtype_of(dtype_of))
}

pub(crate) fn align_affine_embedding_dtype(
    params: &mut HashMap<String, MxArray>,
    target: Option<DType>,
) -> Result<bool> {
    let Some(target) = affine_embedding_cast(&params_dtype(params), target) else {
        return Ok(false);
    };
    for suffix in ["scales", "biases"] {
        let key = format!("embedding.{suffix}");
        if let Some(sidecar) = params.get(&key).cloned() {
            params.insert(key, sidecar.astype(target)?);
        }
    }
    Ok(true)
}

pub(crate) fn settle_compute_dtype(
    raw: &Value,
    params: &mut HashMap<String, MxArray>,
    config: &mut Qwen3_5MoeConfig,
    requested: Option<crate::transformer::KvFormat>,
) -> Result<crate::transformer::KvFormat> {
    let residual_dtype = residual_stream_dtype(raw, params);
    if align_affine_embedding_dtype(params, residual_dtype)? {
        info!(
            "Aligned the quantized embedding's affine sidecars with the \
             checkpoint's {residual_dtype:?} residual stream"
        );
    }
    Ok(config.pin_kv_format(requested, embedding_compute_dtype(params)))
}

/// Load a pretrained Qwen3.5 MoE model into a dedicated model thread.
pub async fn load_moe_with_thread(
    model_path: &str,
    draft_model_path: Option<String>,
    kv_format: Option<crate::transformer::KvFormat>,
) -> Result<Qwen3_5MoeModel> {
    load_moe_with_source(
        &CheckpointSource::Dir(PathBuf::from(model_path)),
        draft_model_path,
        kv_format,
        crate::models::qwen3_5::load_plan::default_activations(true, false),
        None,
        None,
        crate::engine::scheduler::DEFAULT_DECODE_SHARE,
        crate::disk_tier::DiskTierOptions::default(),
        crate::engine::load_progress::LoadProgressCell::detached(),
    )
    .await
}

/// [`load_moe_with_thread`] for a tool that drives the model directly (`mlx
/// eval`, FP8 calibration): no load warm-up, since no scheduled turn runs.
pub async fn load_moe_for_tool(
    model_path: &str,
    kv_format: Option<crate::transformer::KvFormat>,
    activations: crate::transformer::Activations,
) -> Result<Qwen3_5MoeModel> {
    let lease = CheckpointLeaseGuard::resolve(
        Path::new(model_path),
        crate::utils::gguf::NativeGgufFamily::Qwen35Moe,
    )
    .await?;
    let loaded = load_moe_tool_source(lease.source(), kv_format, activations).await?;
    lease.commit();
    Ok(loaded)
}

/// The caller releases `source`'s lease when this fails.
pub(crate) async fn load_moe_tool_source(
    source: &CheckpointSource,
    kv_format: Option<crate::transformer::KvFormat>,
    activations: crate::transformer::Activations,
) -> Result<Qwen3_5MoeModel> {
    load_moe_with_source_impl(
        source,
        None,
        kv_format,
        activations,
        None,
        None,
        crate::engine::scheduler::DEFAULT_DECODE_SHARE,
        crate::disk_tier::DiskTierOptions::default(),
        false,
        crate::engine::load_progress::LoadProgressCell::detached(),
    )
    .await
}

fn load_moe_weights(
    path: &Path,
    preload_input: Option<&PreloadInput>,
) -> Result<HashMap<String, MxArray>> {
    let mut preload = preload_input.map(load_gguf_preload).transpose()?;
    load_checkpoint(path, false, preload.as_mut())
}

/// The caller releases `source`'s lease when this fails.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn load_moe_with_source(
    source: &CheckpointSource,
    draft_model_path: Option<String>,
    kv_format: Option<crate::transformer::KvFormat>,
    activations: crate::transformer::Activations,
    max_memory_bytes: Option<u64>,
    max_context: Option<u32>,
    decode_share: f64,
    disk: crate::disk_tier::DiskTierOptions,
    progress: Arc<crate::engine::load_progress::LoadProgressCell>,
) -> Result<Qwen3_5MoeModel> {
    load_moe_with_source_impl(
        source,
        draft_model_path,
        kv_format,
        activations,
        max_memory_bytes,
        max_context,
        decode_share,
        disk,
        true,
        progress,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn load_moe_with_source_impl(
    source: &CheckpointSource,
    draft_model_path: Option<String>,
    kv_format: Option<crate::transformer::KvFormat>,
    activations: crate::transformer::Activations,
    max_memory_bytes: Option<u64>,
    max_context: Option<u32>,
    decode_share: f64,
    disk: crate::disk_tier::DiskTierOptions,
    warm: bool,
    progress: Arc<crate::engine::load_progress::LoadProgressCell>,
) -> Result<Qwen3_5MoeModel> {
    let model_assets_path = source.asset_dir().to_string_lossy().into_owned();
    let model_path = model_assets_path.clone();
    let preload_input = match source {
        CheckpointSource::Dir(_) => None,
        CheckpointSource::Gguf(checkpoint) => {
            checkpoint.check_asset_config()?;
            Some(checkpoint.preload_input(false))
        }
    };

    let (
        thread,
        (
            config,
            model_id,
            image_processor,
            spatial_merge_size,
            _tokenizer,
            guards,
            paged_active,
            scheduled_lanes,
            mtp_active,
            vision_active,
            context_limits,
            warmup,
            state_tier,
        ),
    ) = spawn_loaded_model_thread(
        "Model thread exited during load",
        move || {
            let path = Path::new(&model_path);

            if !path.exists() {
                return Err(Error::from_reason(format!(
                    "Model path does not exist: {}",
                    model_path
                )));
            }

            type LoadOutput = (
                Qwen35Inner,
                Qwen3_5MoeConfig,
                u64,
                Option<crate::cache_limit::PoolCacheLimitGuard>,
            );
            let load_result: Result<LoadOutput> = (|| -> Result<LoadOutput> {
                let config_path = path.join("config.json");
                let config_data = fs::read_to_string(&config_path)
                    .map_err(|e| Error::from_reason(format!("Failed to read config: {}", e)))?;
                let raw: Value = serde_json::from_str(&config_data)
                    .map_err(|e| Error::from_reason(format!("Failed to parse config: {}", e)))?;

                let mut config = parse_moe_config(&raw)?;
                config.rope_scaling = Some(crate::nn::rope_scaling::RopeScaling::served_for(
                    "Qwen3.5 MoE",
                    config.max_position_embeddings.max(0) as u32,
                    config.scaling(),
                    max_context,
                )?);
                if config.scaling().yarn() {
                    info!(
                        "Qwen3.5 MoE {}",
                        config
                            .scaling()
                            .startup_line(config.served_context_tokens())
                    );
                }

                crate::transformer::KvFormat::parse(config.kv_format.as_deref())
                    .map_err(|message| Error::from_reason(format!("config.json: {message}")))?;

                info!(
                    "Qwen3.5 MoE config: {} layers, hidden={}, experts={}x{}",
                    config.num_layers,
                    config.hidden_size,
                    config.num_experts,
                    config.num_experts_per_tok
                );

                let reads_identity = disk.reads_identity();
                let gguf_identity = preload_input.as_ref().map(PreloadInput::identity);
                let checkpoint_files = CheckpointFiles::new(path, gguf_identity.as_ref());
                let mut checkpoint_load =
                    CheckpointLoadGuard::before_read(checkpoint_files, reads_identity);

                let _free_released = GgufPreloadGuard::enter(preload_input.is_some());
                let mut raw_params = load_moe_weights(path, preload_input.as_ref())?;

                checkpoint_load.record_read();

                prewarm_checkpoint_pages(path);

                let (base_for_mtp, moe_params_for_mtp) = config.split();
                let mtp_mlp_variant =
                    Qwen3_5MTPModule::mtp_mlp_variant(&base_for_mtp, Some(&moe_params_for_mtp));

                let has_inline_mtp = raw_params.keys().any(|name| name.contains("mtp."));
                if has_inline_mtp {
                    info!("Using inline mtp.* tensors from body shards (drafter merge skipped)");
                } else if let Some(drafter_path) =
                    crate::models::mtp_drafter::detect_drafter_safetensors(path)
                    && let Some(drafter_params) = crate::models::mtp_drafter::load_drafter_tensors(
                        &drafter_path,
                        DrafterBodyVariant::Moe,
                        mtp_mlp_variant,
                        config.n_mtp_layers,
                    )?
                {
                    let drafter_count = drafter_params.len();
                    raw_params.extend(drafter_params);
                    info!(
                        "Merged split MTP drafter tensors: added={} (re-prefixed mtp.*)",
                        drafter_count
                    );
                }
                info!("Loaded {} raw tensors", raw_params.len());

                let (text_raw_params, vision_params) = split_qwen35_vision_text_params(raw_params);
                let has_vision = vision_params.is_some();

                let quant_cfg = select_quantization_block(&raw)?;
                let (quant_bits, quant_group_size, top_level_mode, mut per_layer_quant) =
                    parse_quant_settings(quant_cfg, DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE)?;

                let mut params = sanitize_weights_moe(text_raw_params, &config, &per_layer_quant)?;
                settle_compute_dtype(&raw, &mut params, &mut config, kv_format)?;
                let quantized = is_quantized_checkpoint(&params);
                info!(
                    "Sanitized to {} parameters (quantized={})",
                    params.len(),
                    quantized
                );

                let mtp_linear_suffixes: &[&str] = match mtp_mlp_variant {
                    DrafterBodyVariant::Moe => &MTP_MOE_LAYER_LINEAR_SUFFIXES,
                    DrafterBodyVariant::Dense => &MTP_LAYER_LINEAR_SUFFIXES,
                };
                augment_mtplx_mtp_quantization_with_suffixes(
                    &raw,
                    config.n_mtp_layers,
                    mtp_linear_suffixes,
                    &mut per_layer_quant,
                )?;

                if quant_cfg.is_some() {
                    info!(
                        "Using quantization config: bits={}, group_size={}, top_level_mode={:?}, per_layer_overrides={}",
                        quant_bits,
                        quant_group_size,
                        top_level_mode,
                        per_layer_quant.len()
                    );
                }

                let tokenizer_path = path.join("tokenizer.json");
                let tokenizer = if tokenizer_path.exists() {
                    info!("Loading tokenizer from: {}", tokenizer_path.display());
                    Some(Qwen3Tokenizer::load_from_file_sync(
                        tokenizer_path.to_str().ok_or_else(|| {
                            Error::from_reason("Tokenizer path contains invalid UTF-8")
                        })?,
                    )?)
                } else {
                    None
                };

                let mut inner = Qwen35Inner::new_moe(config.clone())?;
                inner.max_memory_bytes = max_memory_bytes;
                inner.disk_options = disk.clone();
                inner.max_context = max_context;
                inner.decode_share = decode_share;
                inner.activations = activations;
                inner.row_exact_decode_projections =
                    has_kquant_mode(top_level_mode, &per_layer_quant)
                        || top_level_mode == Some(PerLayerMode::Affine)
                        || per_layer_quant
                            .values()
                            .any(|quant| quant.mode == PerLayerMode::Affine);
                inner.set_gen_defaults(crate::engine::persistence::parse_generation_defaults(path));

                let text_weight_bytes = params_weight_bytes(&params);
                let base_config = inner.config.clone();
                let (plain_fp8_residency, _) = apply_weights_inner_with_residency(
                    &mut inner,
                    &mut params,
                    &base_config,
                    quant_bits,
                    quant_group_size,
                    top_level_mode,
                    &per_layer_quant,
                    has_vision,
                    None,
                    None,
                    None,
                    false,
                )?;

                let weights_resident = {
                    let mut arrays: Vec<&MxArray> = params.values().collect();
                    if !defer_plain_fp8_materialization() {
                        arrays.extend(plain_fp8_residency.arrays());
                    }
                    crate::array::memory::materialize_weights(&arrays)?
                };

                let draft_load = draft_model_path.as_deref().map(|draft_path| {
                    crate::models::qwen3_5::dflash2_state::disk::DraftLoadGuard::before_read(
                        Path::new(draft_path),
                        reads_identity,
                    )
                });
                let dflash2_weight_bytes = if let Some(draft_path) = draft_model_path.as_deref() {
                    let (mut draft, bytes) =
                        crate::models::qwen3_5::dflash2::load_dflash2(Path::new(draft_path))?;
                    draft.validate_target(&inner.config)?;
                    draft.configure_moe_execution(&config);
                    inner.dflash2 = Some(draft);
                    info!("Loaded external Qwen3.5 MoE DFlash2 companion from {draft_path}");
                    bytes
                } else {
                    0
                };

                if let Some(tok) = tokenizer {
                    inner.set_tokenizer(Arc::new(tok));
                }

                let vision_params = load_vision_encoder_moe(
                    &mut inner,
                    vision_params,
                    &raw,
                    &config,
                    top_level_mode,
                    &per_layer_quant,
                )?;

                let weights_resident = match vision_params.as_ref() {
                    Some(vparams) => {
                        let arrays: Vec<&MxArray> = vparams.values().collect();
                        weights_resident.and(crate::array::memory::materialize_weights(&arrays)?)
                    }
                    None => weights_resident,
                };
                let mut weight_bytes: u64 = text_weight_bytes;
                if let Some(ref vparams) = vision_params {
                    weight_bytes = weight_bytes.saturating_add(params_weight_bytes(vparams));
                }
                weight_bytes = weight_bytes.saturating_add(plain_fp8_residency.nbytes());
                weight_bytes = weight_bytes.saturating_add(dflash2_weight_bytes);

                drop(params);
                drop(vision_params);
                drop(plain_fp8_residency);
                crate::array::memory::synchronize_and_clear_cache();

                let pool_cache_limit_guard =
                    crate::engine::governor::governor::start_under_growth_lock(
                        crate::engine::governor::governor(),
                        || {
                            inner.initialize_paged_adapter()?;
                            inner.require_dflash2_pool()?;
                            let pool_bytes = inner
                                .paged_adapter
                                .as_ref()
                                .map(PagedKVCacheAdapter::pool_allocated_bytes)
                                .transpose()
                                .map_err(Error::from_reason)?
                                .unwrap_or(0);
                            Ok((pool_bytes != 0).then(|| {
                                crate::cache_limit::coordinator().register_pool(pool_bytes)
                            }))
                        },
                    )?;

                inner.build_disk_tier_context(
                    checkpoint_files,
                    draft_load.as_ref(),
                    &weights_resident,
                    &checkpoint_load,
                );

                Ok((inner, config, weight_bytes, pool_cache_limit_guard))
            })();
            let (mut inner, config_out, weight_bytes, pool_cache_limit_guard) = load_result?;
            let guards = register_loaded_weights_and_pool(weight_bytes, 0, pool_cache_limit_guard);
            if let (Some(guard), Some(adapter)) = (
                guards.pool_cache_limit.as_ref(),
                inner.paged_adapter.as_mut(),
            ) {
                attach_pool_growth_notifier(adapter, guard);
            }

            let model_id = inner.model_id;
            let state_tier = inner.dflash2_states.tier_handle();
            let image_processor = inner.image_processor.as_ref().map(Arc::clone);
            let spatial_merge_size = inner.spatial_merge_size.unwrap_or(2);
            let tokenizer_out = inner.tokenizer.clone();
            let paged_active = inner.paged_adapter.is_some();
            let scheduled_lanes = inner.scheduled_lanes();
            let mtp_active = inner.dflash2.is_some() || inner.has_mtp_weights();
            let vision_active = qwen35_dense_vision_active(
                inner.vision_encoder.is_some(),
                inner.image_processor.is_some(),
                paged_active,
            );
            if let Some(adapter) = inner.paged_adapter.as_ref() {
                adapter
                    .layer_kv_pool()
                    .hold_resident()
                    .map_err(Error::from_reason)?;
            }
            inner.wired = Some(crate::stream::WiredResidency::new(
                crate::stream::WiredResidency::working_set_minus_pools,
            ));
            let context_limits = inner.finish_load_limits()?;
            let (state, warmup) = crate::engine::warmup::serving_scheduler(inner, &progress, warm)?;

            Ok((
                state,
                (
                    config_out,
                    model_id,
                    image_processor,
                    spatial_merge_size,
                    tokenizer_out,
                    guards,
                    paged_active,
                    scheduled_lanes,
                    mtp_active,
                    vision_active,
                    context_limits,
                    warmup,
                    state_tier,
                ),
            ))
        },
        |state, receiver| state.drive(receiver),
    )
    .await?;
    register_thread_memory_waker(model_id, &thread);

    Ok(Qwen3_5MoeModel::from_loaded(
        thread,
        config,
        warmup,
        paged_active,
        scheduled_lanes,
        mtp_active,
        vision_active,
        image_processor,
        spatial_merge_size,
        context_limits,
        model_assets_path,
        state_tier,
        guards.cache_limit,
        guards.pool_cache_limit,
    ))
}

/// Parse Qwen3.5 MoE config from JSON.
pub(crate) fn parse_moe_config(raw: &Value) -> Result<Qwen3_5MoeConfig> {
    let text_cfg = raw.get("text_config");
    let paged = PagedCacheConfig::from_raw_json(raw);

    let gi = |keys: &[&str], default: i32| get_config_i32(raw, text_cfg, keys, default);
    let gf = |keys: &[&str], default: f64| get_config_f64(raw, text_cfg, keys, default);
    let gb = |keys: &[&str], default: bool| get_config_bool(raw, text_cfg, keys, default);

    let hidden_size = gi(&["hidden_size"], 0);
    let num_heads = gi(&["num_attention_heads", "num_heads"], 0);

    let head_dim = text_cfg
        .and_then(|tc| tc["head_dim"].as_i64())
        .or_else(|| raw["head_dim"].as_i64())
        .map(|v| v as i32)
        .unwrap_or_else(|| {
            if num_heads > 0 {
                hidden_size / num_heads
            } else {
                128
            }
        });

    let rope_obj = text_cfg
        .and_then(|tc| tc.get("rope_parameters"))
        .or_else(|| raw.get("rope_parameters"));
    crate::nn::rope_scaling::refuse_scaled_rope("Qwen3.5 MoE", rope_obj)?;

    let partial_rotary_factor = rope_obj
        .and_then(|rp| rp["partial_rotary_factor"].as_f64())
        .unwrap_or_else(|| gf(&["partial_rotary_factor"], 0.25));

    let rope_theta = rope_obj
        .and_then(|rp| rp["rope_theta"].as_f64())
        .unwrap_or_else(|| gf(&["rope_theta"], 100_000.0));

    let bos_token_id = gi(&["bos_token_id"], 151643);
    let num_layers = gi(&["num_hidden_layers", "num_layers"], 0);
    let intermediate_size = gi(&["intermediate_size"], 0);
    let num_kv_heads = gi(&["num_key_value_heads", "num_kv_heads"], 8);
    let vocab_size = gi(&["vocab_size"], 151936);

    let num_experts = gi(&["num_experts"], 0);
    let num_experts_per_tok = gi(&["num_experts_per_tok"], 1);
    let moe_i = gi(&["moe_intermediate_size"], 0);

    if hidden_size <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid config: hidden_size must be > 0, got {}",
            hidden_size
        )));
    }
    if num_layers <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid config: num_hidden_layers must be > 0, got {}",
            num_layers
        )));
    }
    if num_experts <= 0 {
        return Err(Error::from_reason(format!(
            "MoE config requires num_experts > 0, got {}",
            num_experts
        )));
    }
    if intermediate_size <= 0 && moe_i <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid config: intermediate_size ({}) or moe_intermediate_size ({}) must be > 0",
            intermediate_size, moe_i
        )));
    }

    Ok(Qwen3_5MoeConfig {
        rope_scaling: None,
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        intermediate_size,
        rms_norm_eps: gf(&["rms_norm_eps"], 1e-6),
        head_dim,
        tie_word_embeddings: gb(&["tie_word_embeddings"], false),
        attention_bias: gb(&["attention_bias"], false),
        max_position_embeddings: gi(&["max_position_embeddings"], 131072),
        pad_token_id: gi(&["pad_token_id"], bos_token_id),
        eos_token_id: gi(&["eos_token_id"], 151645),
        bos_token_id,

        linear_num_value_heads: gi(&["linear_num_value_heads"], 64),
        linear_num_key_heads: gi(&["linear_num_key_heads"], 16),
        linear_key_head_dim: gi(&["linear_key_head_dim"], 192),
        linear_value_head_dim: gi(&["linear_value_head_dim"], 128),
        linear_conv_kernel_dim: gi(&["linear_conv_kernel_dim"], 4),
        full_attention_interval: gi(&["full_attention_interval"], 4),
        partial_rotary_factor,
        rope_theta,

        num_experts,
        num_experts_per_tok,
        decoder_sparse_step: gi(&["decoder_sparse_step"], 1),
        shared_expert_intermediate_size: {
            let v = gi(&["shared_expert_intermediate_size"], 0);
            if v > 0 { Some(v) } else { None }
        },
        moe_intermediate_size: if moe_i > 0 { Some(moe_i) } else { None },
        norm_topk_prob: gb(&["norm_topk_prob"], true),
        mlp_only_layers: text_cfg
            .and_then(|tc| tc["mlp_only_layers"].as_array())
            .or_else(|| raw["mlp_only_layers"].as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64().map(|i| i as i32))
                    .collect()
            }),
        paged_cache_memory_mb: paged.paged_cache_memory_mb,
        paged_block_size: paged.paged_block_size,
        n_mtp_layers: gi(&["mtp_num_hidden_layers", "num_nextn_predict_layers"], 0),
        qwen35_gguf_gdn_layout: raw
            .get("qwen35_gguf_gdn_layout")
            .and_then(Value::as_str)
            .map(str::to_string),
        kv_format: raw
            .get("kv_format")
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}

/// Create a random-init Qwen3.5 MoE model and save it to `save_path`,
/// synchronously on the calling thread.
pub fn create_random_qwen35_moe_checkpoint_sync(
    config: Qwen3_5MoeConfig,
    save_path: &str,
) -> Result<()> {
    let mut inner = Qwen35Inner::new_moe(config)?;
    if inner.config.n_mtp_layers > 0 {
        inner.mtp_weights_loaded = true;
    }
    inner.save_model_sync(save_path)
}

#[cfg(test)]
#[path = "persistence_moe_tests.rs"]
mod persistence_moe_tests;

#[cfg(test)]
mod outer_scale_tests {
    use super::*;

    #[test]
    fn normalize_moe_weight_map_preserves_outer_scales_on_fused_and_down_proj() {
        let mut params = HashMap::new();
        // Fused gate_up_proj with [E] .scale, .global_scale, .weight_scale_2 + weight [E, 4, 8]
        params.insert(
            "layers.0.mlp.experts.gate_up_proj.weight".to_string(),
            MxArray::zeros(&[2, 4, 8], Some(DType::BFloat16)).unwrap(),
        );
        params.insert(
            "layers.0.mlp.experts.gate_up_proj.scale".to_string(),
            MxArray::from_float32(&[0.5, 1.5], &[2]).unwrap(),
        );
        params.insert(
            "layers.0.mlp.experts.gate_up_proj.global_scale".to_string(),
            MxArray::from_float32(&[10.0, 20.0], &[2]).unwrap(),
        );
        params.insert(
            "layers.0.mlp.experts.down_proj.weight".to_string(),
            MxArray::zeros(&[2, 8, 2], Some(DType::BFloat16)).unwrap(),
        );
        params.insert(
            "layers.0.mlp.experts.down_proj.scale".to_string(),
            MxArray::from_float32(&[0.25, 0.75], &[2]).unwrap(),
        );
        params.insert(
            "layers.0.mlp.experts.down_proj.weight_scale_2".to_string(),
            MxArray::from_float32(&[0.125, 0.375], &[2]).unwrap(),
        );

        let out = normalize_moe_weight_map(params, false, false, false, 2).unwrap();
        assert!(out.contains_key("layers.0.mlp.switch_mlp.gate_proj.weight"));
        assert!(out.contains_key("layers.0.mlp.switch_mlp.up_proj.weight"));
        assert_eq!(
            out["layers.0.mlp.switch_mlp.gate_proj.scale"]
                .to_float32()
                .unwrap()
                .as_ref(),
            &[0.5f32, 1.5]
        );
        assert_eq!(
            out["layers.0.mlp.switch_mlp.up_proj.scale"]
                .to_float32()
                .unwrap()
                .as_ref(),
            &[0.5f32, 1.5]
        );
        assert_eq!(
            out["layers.0.mlp.switch_mlp.gate_proj.global_scale"]
                .to_float32()
                .unwrap()
                .as_ref(),
            &[10.0f32, 20.0]
        );
        assert_eq!(
            out["layers.0.mlp.switch_mlp.up_proj.global_scale"]
                .to_float32()
                .unwrap()
                .as_ref(),
            &[10.0f32, 20.0]
        );
        assert!(out.contains_key("layers.0.mlp.switch_mlp.down_proj.scale"));
        assert!(out.contains_key("layers.0.mlp.switch_mlp.down_proj.weight_scale_2"));
        assert!(!out.contains_key("layers.0.mlp.switch_mlp.down_proj.scale.weight"));
    }

    #[test]
    fn merge_split_projections_keeps_scaled_nvfp4_gdn_projections_split() {
        let mut params = HashMap::new();
        for (name, rows) in [
            ("layers.0.linear_attn.in_proj_qkv", 16i64),
            ("layers.0.linear_attn.in_proj_z", 8i64),
            ("layers.0.linear_attn.in_proj_b", 4i64),
            ("layers.0.linear_attn.in_proj_a", 4i64),
        ] {
            params.insert(
                format!("{name}.weight"),
                MxArray::zeros(&[rows, 8], Some(DType::Uint32)).unwrap(),
            );
            params.insert(
                format!("{name}.scales"),
                MxArray::zeros(&[rows, 4], Some(DType::Uint8)).unwrap(),
            );
        }
        params.insert(
            "layers.0.linear_attn.in_proj_qkv.global_scale".to_string(),
            MxArray::from_float32(&[2688.0], &[1]).unwrap(),
        );
        params.insert(
            "layers.0.linear_attn.in_proj_z.global_scale".to_string(),
            MxArray::from_float32(&[5376.0], &[1]).unwrap(),
        );
        params.insert(
            "layers.0.linear_attn.in_proj_b.weight_scale_2".to_string(),
            MxArray::from_float32(&[0.5], &[1]).unwrap(),
        );

        let plq = HashMap::new();
        super::merge_split_projections(&mut params, &plq).unwrap();

        // Both pairs MUST stay split so neither outer scale is orphaned/dropped!
        assert!(!params.contains_key("layers.0.linear_attn.in_proj_qkvz.weight"));
        assert!(!params.contains_key("layers.0.linear_attn.in_proj_ba.weight"));
        assert!(params.contains_key("layers.0.linear_attn.in_proj_qkv.weight"));
        assert!(params.contains_key("layers.0.linear_attn.in_proj_qkv.global_scale"));
        assert!(params.contains_key("layers.0.linear_attn.in_proj_z.weight"));
        assert!(params.contains_key("layers.0.linear_attn.in_proj_z.global_scale"));
        assert!(params.contains_key("layers.0.linear_attn.in_proj_b.weight"));
        assert!(params.contains_key("layers.0.linear_attn.in_proj_b.weight_scale_2"));
        assert!(params.contains_key("layers.0.linear_attn.in_proj_a.weight"));
    }
}
