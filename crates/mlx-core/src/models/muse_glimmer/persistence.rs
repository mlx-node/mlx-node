use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::cold_tier::{resolve_persist_cold, shard_identities_stable, snapshot_shard_identities};
use crate::engine::persistence::{load_all_safetensors, parse_generation_defaults};
use crate::models::gemma4::quantized_linear::{
    LinearProj, try_build_fp8_e4m3_quantized_linear, try_build_kquant_quantized_linear,
    try_build_mxfp4_quantized_linear, try_build_mxfp8_quantized_linear,
    try_build_nvfp4_quantized_linear, try_build_quantized_linear, try_build_sym8_quantized_linear,
};
use crate::models::quant_dispatch::{
    PerLayerMode, PerLayerQuant, effective_plq_for, ensure_affine_biases_present,
    ensure_dense_weight_floating, ensure_int8_storage_resolves_sym8,
    ensure_kquant_storage_resolves_kquant, ensure_plain_fp8_storage_resolves_fp8_e4m3,
    has_kquant_mode, load_quant_settings_from_disk, mode_to_str, normalize_per_layer_key,
};
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use crate::transformer::{AttentionKind, KVCacheDType};
use crate::utils::safetensors::load_safetensors_lazy;

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

fn quant_lookup_prefix(prefix: &str) -> String {
    normalize_per_layer_key(&crate::utils::normalize_override_key(prefix))
}

pub(super) fn build_projection(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    default: PerLayerQuant,
    overrides: &HashMap<String, PerLayerQuant>,
) -> Result<LinearProj> {
    let quant_prefix = quant_lookup_prefix(prefix);
    let plq = effective_plq_for(&quant_prefix, overrides, default, None);
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
            PerLayerMode::Affine => {
                try_build_quantized_linear(params, prefix, plq.group_size, plq.bits)
            }
            PerLayerMode::Sym8 => try_build_sym8_quantized_linear(params, prefix)?,
            PerLayerMode::Q4K | PerLayerMode::Q5K | PerLayerMode::Q6K => {
                try_build_kquant_quantized_linear(params, prefix, plq.mode, "muse_glimmer")?
            }
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
        let quant_prefix = quant_lookup_prefix(prefix);
        let plq = effective_plq_for(&quant_prefix, overrides, default, None);
        if matches!(plq.mode, PerLayerMode::Sym8 | PerLayerMode::Fp8E4m3) {
            return Err(Error::from_reason(format!(
                "Muse-Glimmer embedding '{prefix}' does not support {:?}",
                plq.mode
            )));
        }
        ensure_kquant_storage_resolves_kquant(params, prefix, plq.mode, "muse_glimmer")?;
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

fn load_dflash(
    path: &Path,
    config: &MuseGlimmerConfig,
    default: PerLayerQuant,
    overrides: &HashMap<String, PerLayerQuant>,
) -> Result<Option<DFlashModel>> {
    let draft_path = path.join("draft.safetensors");
    if !draft_path.is_file() {
        return Ok(None);
    }
    let draft_config = config.dflash_config.clone().ok_or_else(|| {
        Error::from_reason(
            "Muse-Glimmer has draft.safetensors but config.json has no dflash_config; reconvert the companion GGUF with --draft",
        )
    })?;
    let params = load_safetensors_lazy(&draft_path)?;
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
        .expect("positive Muse-Glimmer prefill chunk");
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
            blocks, block_size,
        )));
        let pool = mlx_paged_attn::LayerKVPool::new(
            pa_config,
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

fn load_target_safetensors(path: &Path) -> Result<HashMap<String, MxArray>> {
    // Muse-Glimmer execution is text-only today. Keep the optional
    // vision.safetensors sidecar out of residency accounting and cold-tier
    // materialization until a vision forward path consumes it.
    load_all_safetensors(path, false)
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

fn load_inner(path: &Path) -> Result<(MuseGlimmerInner, u64)> {
    let config = MuseGlimmerConfig::from_path(path)?;
    let persist_env = std::env::var("MLX_PERSIST_PAGED_CACHE").ok();
    let persist_cold = resolve_persist_cold(
        "muse_glimmer",
        persist_env.as_deref(),
        config.persist_paged_cache,
    );
    let shard_snapshot_before_mmap = if persist_cold {
        snapshot_shard_identities(path)
    } else {
        None
    };
    let params = load_target_safetensors(path)?;
    let shard_snapshot_at_mmap = if persist_cold {
        snapshot_shard_identities(path)
    } else {
        None
    };
    if persist_cold {
        crate::engine::persistence::prewarm_checkpoint_pages(path);
    }
    let (bits, group_size, top_mode, overrides) = load_quant_settings_from_disk(path, 4, 64)?;
    let default = PerLayerQuant {
        bits,
        group_size,
        mode: top_mode.unwrap_or(PerLayerMode::Affine),
        input_amax: None,
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

    let mut layers = Vec::with_capacity(text.num_hidden_layers);
    for index in 0..text.num_hidden_layers {
        let base = format!("model.language_model.layers.{index}");
        let attn = format!("{base}.self_attn");
        let attention = MuseGlimmerAttention::from_projections(
            text,
            index,
            config.rope_traditional,
            build_projection(&params, &format!("{attn}.q_proj"), default, &overrides)?,
            build_projection(&params, &format!("{attn}.k_proj"), default, &overrides)?,
            build_projection(&params, &format!("{attn}.v_proj"), default, &overrides)?,
            build_projection(&params, &format!("{attn}.o_proj"), default, &overrides)?,
            build_projection(&params, &format!("{attn}.gate_proj"), default, &overrides)?,
        )?;
        let mlp_base = format!("{base}.mlp");
        let mlp = MuseGlimmerMlp::new(
            build_projection(
                &params,
                &format!("{mlp_base}.gate_proj"),
                default,
                &overrides,
            )?,
            build_projection(&params, &format!("{mlp_base}.up_proj"), default, &overrides)?,
            build_projection(
                &params,
                &format!("{mlp_base}.down_proj"),
                default,
                &overrides,
            )?,
        );
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
    let dflash = load_dflash(path, &config, default, &overrides)?;
    let paged = build_paged_runtime(&config)?;
    let has_dflash = dflash.is_some();
    let weight_bytes = params
        .values()
        .map(|array| array.nbytes() as u64)
        .sum::<u64>()
        .saturating_add(if has_dflash {
            std::fs::metadata(path.join("draft.safetensors"))
                .map(|meta| meta.len())
                .unwrap_or(0)
        } else {
            0
        });
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
        let after_fingerprint = snapshot_shard_identities(path);
        if shard_identities_stable(
            &shard_snapshot_before_mmap,
            &shard_snapshot_at_mmap,
            &after_fingerprint,
        ) {
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
    let model_path = model_path.to_string();
    let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_scheduler(
        move || {
            let (inner, weight_bytes) = load_inner(Path::new(&model_path))?;
            let has_dflash = inner.dflash.is_some();
            let pool_bytes = inner
                .paged
                .as_ref()
                .map(|paged| paged.coordinator.pool_allocated_bytes())
                .transpose()
                .map_err(Error::from_reason)?
                .unwrap_or(0);
            let paged_active = inner.paged.is_some();
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
        cache_limit_guard,
        pool_cache_limit_guard,
    ) = init_rx
        .await
        .map_err(|_| Error::from_reason("Muse-Glimmer model thread exited during load"))??;
    Ok(MuseGlimmerModel {
        thread,
        has_dflash,
        paged_active,
        max_concurrent_sequences,
        _cache_limit_guard: cache_limit_guard,
        _pool_cache_limit_guard: pool_cache_limit_guard,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        load_lm_head, load_target_safetensors, quant_lookup_prefix,
        requires_row_exact_decode_projections,
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
    }

    #[test]
    fn text_runtime_excludes_the_unused_vision_sidecar() {
        let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "muse_glimmer_text_weights_{}_{}",
            std::process::id(),
            id
        ));
        std::fs::create_dir_all(&dir).expect("create model directory");
        save_one(&dir.join("model.safetensors"), "text.weight", 1.0);
        save_one(&dir.join("vision.safetensors"), "vision.weight", 2.0);

        let weights = load_target_safetensors(&dir).expect("load Muse target weights");
        assert!(weights.contains_key("text.weight"));
        assert!(!weights.contains_key("vision.weight"));

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
}
