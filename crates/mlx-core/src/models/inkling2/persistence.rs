use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use mlx_paged_attn::CacheRecency;
use napi::bindgen_prelude::{Error, Result};

use crate::array::memory::{materialize_weights, synchronize_and_clear_cache};
use crate::array::{DType, MxArray};
use crate::cache_limit;
use crate::cold_tier::{CheckpointFiles, CheckpointLoadGuard};
use crate::disk_tier::{DiskTierOptions, TierHandle};
use crate::engine::checkpoint_source::CheckpointSource;
use crate::engine::governor::governor as memory_governor;
use crate::engine::governor::governor::start_under_growth_lock;
use crate::engine::memory_plan::live_memory;
use crate::engine::persistence::{
    CheckpointLeaseGuard, GgufPreloadGuard, load_checkpoint, register_dynamic_pool,
    register_thread_memory_waker, spawn_loaded_model_thread,
};
use crate::models::quant_dispatch::{
    PerLayerMode, PerLayerQuant, QslBuildSpec, QslFp8Policy, QslSym8Policy, build_qsl,
    effective_plq_for, load_embedding_affine_or_bf16, load_linear_proj_quantized_or_bf16,
    load_quant_settings_from_disk,
};
use crate::models::quantized_linear::{LinearProj, MXFP8_MODE, NVFP4_MODE, QuantizedSwitchLinear};
use crate::nn::{Embedding, LayerNorm, Linear};
use crate::stream::WiredResidency;
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::Activations;
use crate::utils::gguf::NativeGgufFamily;
use crate::utils::gguf_import::PreloadInput;

use super::attention::Inkling2Attention;
use super::attn_res::{LoopController, LoopLayerControllers};
use super::config::{Inkling2Config, Inkling2VisionConfig};
use super::dspark::{Inkling2DsparkModel, resolve_inkling2_dspark_candidates};
use super::kv_cache::{Inkling2KVCacheCoordinator, Inkling2PagedConfig};
use super::memory::{
    Inkling2ContextLimits, evaluate_inkling2_plan, inkling2_scheduler_lanes, resolve_activations,
    resolve_decode_share, resolve_kv_format, resolve_max_context, resolve_max_memory,
};
use super::model::{
    AttnInputNorm, Inkling2Inner, Inkling2LoadOptions, Inkling2Model, Inkling2PhysicalLayer,
    default_decode_share, next_inkling2_model_id,
};
use super::moe::{ExpertSwitchProj, Inkling2MoE};
use super::paged_forward::Inkling2PagedRuntime;
use super::scheduled_dspark::Inkling2ScheduledDsparkStates;
use super::scheduler::Inkling2SchedulerState;
use super::state_disk::compute_persistent_identity;
use super::vision::{Inkling2VisionBlock, Inkling2VisionEncoder};

const FAMILY_NAME: &str = "inkling2";

/// Load an Inkling2 / Inkling-2-Mini inner model from a directory or GGUF file path
/// (including optional companion `mmproj-BF16.gguf` vision encoder and embedded tokenizer).
///
/// Automatically initializes the block-paged KV runtime on Metal using **INT8 32-token KTM**
/// by default; use [`load_from_dir_with_paged_config`] or [`Inkling2Inner::enable_paged`] to
/// select `KvFormat::Bf16` or custom pool sizing.
pub async fn load_from_dir(model_path: &str) -> Result<Inkling2Inner> {
    load_from_dir_with_paged_config(model_path, None).await
}

/// Load an Inkling2 / Inkling-2-Mini inner model with an explicit [`Inkling2PagedConfig`] override.
pub async fn load_from_dir_with_paged_config(
    model_path: &str,
    paged_cfg: Option<Inkling2PagedConfig>,
) -> Result<Inkling2Inner> {
    let path = Path::new(model_path);
    let lease = CheckpointLeaseGuard::resolve(path, NativeGgufFamily::Inkling2).await?;
    let source_dir = if path.is_file() {
        path.parent()
            .unwrap_or_else(|| lease.source().asset_dir())
            .to_path_buf()
    } else {
        path.to_path_buf()
    };
    let loaded = load_with_source_dir(lease.source(), &source_dir, paged_cfg).await?;
    lease.commit();
    Ok(loaded)
}

/// Load using an already-resolved [`CheckpointSource`] and model directory.
async fn load_with_source_dir(
    source: &CheckpointSource,
    source_dir: &Path,
    paged_cfg: Option<Inkling2PagedConfig>,
) -> Result<Inkling2Inner> {
    let asset_dir = source.asset_dir().to_path_buf();
    let (preload_input, gguf_vision_meta, is_gguf) = match source {
        CheckpointSource::Dir(_) => (None, None, false),
        CheckpointSource::Gguf(checkpoint) => {
            checkpoint.check_asset_config()?;
            let companion_meta = checkpoint
                .companion
                .as_ref()
                .map(|(_, gguf, _)| gguf.metadata.clone());
            (Some(checkpoint.preload_input(true)), companion_meta, true)
        }
    };

    // Run synchronously on the calling thread so MLX GPU stream affinity matches
    // subsequent forward/decode calls on the returned `Inkling2Inner`.
    load_sync(
        &asset_dir,
        source_dir,
        preload_input,
        gguf_vision_meta.as_ref(),
        is_gguf,
        paged_cfg,
    )
}

/// NAPI entry point backing [`Inkling2Model::load`]: loads weights, evaluates the unified
/// [`crate::engine::memory_plan::MemoryPlan`], registers extent-grown Group 0 + Group 1 KV pools
/// with the unified [`Governor`] ledger, activates [`WiredResidency`] + `PoolResidency`, opens
/// and adopts persistent StateDisk entries, and spawns the continuous-batching
/// [`Inkling2SchedulerState`] on a dedicated OS thread via [`ModelThread::spawn_with_scheduler`].
pub(crate) async fn load_model_with_options(
    model_path: &str,
    options: Inkling2LoadOptions,
) -> Result<Inkling2Model> {
    let explicit_draft_path = options.draft_model_path.clone();
    let auto_load_draft = options.auto_load_draft.unwrap_or(true);
    let draft_is_explicit = explicit_draft_path.is_some();
    let disk_options = options.disk_tier("Inkling2")?;
    let kv_format = resolve_kv_format("Inkling2", options.kv_format.as_deref())?;
    let activations = resolve_activations("Inkling2", options.activations.as_deref())?;
    let max_memory_bytes = resolve_max_memory("Inkling2", options.max_memory_bytes)?;
    let decode_share = resolve_decode_share("Inkling2", options.decode_share)?;
    let max_context_raw = options.max_context_tokens;

    let path = Path::new(model_path);
    let lease = CheckpointLeaseGuard::resolve(path, NativeGgufFamily::Inkling2).await?;
    let source = lease.source();
    let asset_dir = source.asset_dir().to_path_buf();
    let asset_dir_str = asset_dir.display().to_string();
    let source_dir = if path.is_file() {
        path.parent().unwrap_or(&asset_dir).to_path_buf()
    } else {
        path.to_path_buf()
    };
    let (preload_input, gguf_vision_meta, is_gguf) = match source {
        CheckpointSource::Dir(_) => (None, None, false),
        CheckpointSource::Gguf(checkpoint) => {
            checkpoint.check_asset_config()?;
            let companion_meta = checkpoint
                .companion
                .as_ref()
                .map(|(_, gguf, _)| gguf.metadata.clone());
            (Some(checkpoint.preload_input(true)), companion_meta, true)
        }
    };
    let draft_candidates = if draft_is_explicit {
        resolve_inkling2_dspark_candidates(
            &source_dir,
            &asset_dir,
            is_gguf,
            explicit_draft_path.as_deref(),
        )?
    } else if auto_load_draft {
        resolve_inkling2_dspark_candidates(&source_dir, &asset_dir, is_gguf, None)?
    } else {
        Vec::new()
    };

    struct InitMeta {
        model_id: u64,
        config: Inkling2Config,
        vision_config: Option<Inkling2VisionConfig>,
        tokenizer: Option<Arc<Qwen3Tokenizer>>,
        paged_active: bool,
        scheduled_lanes: u32,
        vision_active: bool,
        dspark_active: bool,
        context_limits: Inkling2ContextLimits,
        state_tier: TierHandle,
        weight_guard: cache_limit::CacheLimitGuard,
        pool_guards: Vec<cache_limit::PoolCacheLimitGuard>,
    }

    let max_lanes = inkling2_scheduler_lanes();
    let (thread, meta) = spawn_loaded_model_thread(
        "Inkling2 model thread terminated during initialization",
        move || -> Result<(Inkling2SchedulerState, InitMeta)> {
            let gguf_identity = preload_input.as_ref().map(|input| input.identity());
            let ckpt_files = CheckpointFiles::new(&asset_dir, gguf_identity.as_ref());
            let mut load_guard =
                CheckpointLoadGuard::before_read(ckpt_files, disk_options.reads_identity());

            // 1. Load & materialize backbone + optional DSpark drafter weights.
            let mut inner = load_weights_sync(
                &asset_dir,
                preload_input,
                gguf_vision_meta.as_ref(),
                is_gguf,
            )?;
            if !draft_candidates.is_empty() {
                let mut last_err: Option<Error> = None;
                for draft_path in &draft_candidates {
                    let t_draft = std::time::Instant::now();
                    match Inkling2DsparkModel::load_auto(draft_path, &inner.config, &inner.lm_head)
                    {
                        Ok(draft) => {
                            tracing::info!(
                                "Inkling2 DSpark loaded from {} in {:.2}s (layers={}, block_size={}, markov_rank={}, taps={:?}, weights={:.1} MiB)",
                                draft_path.display(),
                                t_draft.elapsed().as_secs_f64(),
                                draft.config.num_hidden_layers,
                                draft.config.block_size,
                                draft.config.markov_rank,
                                draft.config.target_layer_ids,
                                draft.weight_bytes() as f64 / (1024.0 * 1024.0),
                            );
                            inner.dspark = Some(draft);
                            last_err = None;
                            break;
                        }
                        Err(err) => {
                            tracing::warn!(
                                "Skipping Inkling2 DSpark draft '{}' (trying next candidate if any): {err}",
                                draft_path.display()
                            );
                            last_err = Some(err);
                        }
                    }
                }
                if draft_is_explicit
                    && inner.dspark.is_none()
                    && let Some(err) = last_err
                {
                    return Err(err);
                }
            }
            load_guard.record_read();
            let weight_arrays = inner.weight_arrays();
            let weight_refs: Vec<&MxArray> = weight_arrays.iter().collect();
            let resident = materialize_weights(&weight_refs)?;
            let weight_bytes = inner.weight_bytes();
            synchronize_and_clear_cache();

            let trained_window = u32::try_from(inner.config.max_position_embeddings)
                .unwrap_or(u32::MAX)
                .max(1);
            let max_context = resolve_max_context("Inkling2", max_context_raw, trained_window)?;
            inner.decode_share = decode_share;
            inner.max_context = max_context;
            inner.activations = activations;
            inner.disk_options = disk_options.clone();

            let mut pool_guards: Vec<cache_limit::PoolCacheLimitGuard> = Vec::new();
            let mut state_tier = TierHandle::default();
            let mut scheduled_lanes = 1u32;
            let mut context_limits = Inkling2ContextLimits::from_parts(
                trained_window,
                max_context.unwrap_or(trained_window),
                0,
                super::kv_cache::INKLING2_PAGED_BLOCK_SIZE,
                false,
                kv_format,
                false,
                None,
            );

            // 2. Plan & allocate extent-grown Group 0 + Group 1 KV pools under Governor Ledger.
            if crate::engine::persistence::compiled_forward_backend_available() {
                let model_id = inner.model_id;
                let recency = Arc::clone(&inner.cache_recency);
                let cfg_ref = &inner.config;
                let (pool_plan, guard, mut coordinator) =
                    start_under_growth_lock(memory_governor(), || {
                        let live = live_memory().ok_or_else(|| {
                            Error::from_reason(
                                "Inkling2 memory plan: Metal working set unavailable",
                            )
                        })?;
                        let plan = evaluate_inkling2_plan(
                            "Inkling2",
                            cfg_ref,
                            kv_format,
                            live,
                            max_memory_bytes,
                            max_context,
                            max_lanes,
                            disk_options.cache_disk,
                        )?;
                        let (reg_guard, coord) = Inkling2KVCacheCoordinator::new_with_memory_plan(
                            cfg_ref,
                            &plan,
                            model_id,
                            memory_governor(),
                            Arc::clone(&recency),
                            activations,
                        )?;
                        Ok((plan, reg_guard, coord))
                    })?;

                // Register every group's pool with cache_limit::coordinator() + growth notifier.
                for gid in 0..coordinator.groups().len() {
                    let initial_bytes = coordinator
                        .adapter(gid)
                        .and_then(|a| a.pool_allocated_bytes())
                        .unwrap_or(0);
                    let pool_guard = if let Ok(adapter) = coordinator.adapter_mut(gid) {
                        register_dynamic_pool(adapter, initial_bytes)
                    } else {
                        cache_limit::coordinator().register_pool(initial_bytes)
                    };
                    pool_guards.push(pool_guard);
                }

                let int8_queries = kv_format == crate::transformer::KvFormat::Int8
                    && activations == Activations::Auto;
                scheduled_lanes = pool_plan.lanes;
                context_limits = Inkling2ContextLimits::from_parts(
                    trained_window,
                    pool_plan.effective_window_tokens,
                    pool_plan.global_max_blocks,
                    pool_plan.block_size,
                    true,
                    pool_plan.kv_format,
                    int8_queries,
                    Some(model_id),
                );
                let max_chunk = pool_plan.max_chunk;
                inner.memory_plan = Some(pool_plan.memory_plan);
                inner.memory_registration = Some(guard);
                inner.wired = Some(WiredResidency::new(WiredResidency::working_set_minus_pools));
                inner.paged = Some(Inkling2PagedRuntime::from_coordinator(
                    coordinator,
                    max_chunk,
                    Arc::clone(&inner.cache_recency),
                ));

                // 3. Attach StateDisk SSD tier & adopt persistent entries on restart.
                let geom_opt = inner.paged.as_ref().and_then(|rt| {
                    super::state_disk::Inkling2StateGeometry::from_coordinator(
                        &inner.config,
                        &rt.coordinator,
                    )
                    .ok()
                });
                if let Some(geom) = geom_opt {
                    let persistent_id = if disk_options.reads_identity() {
                        inner.paged.as_ref().and_then(|rt_ref| {
                            compute_persistent_identity(
                                &inner.config,
                                &rt_ref.coordinator,
                                &geom,
                                activations,
                                ckpt_files,
                                &resident,
                                &load_guard,
                            )
                        })
                    } else {
                        None
                    };
                    if let Some(handle) =
                        inner.attach_state_disk(&disk_options, move || persistent_id)
                    {
                        state_tier = handle;
                    }
                }
            }

            let weight_guard = cache_limit::coordinator().register(weight_bytes);
            let meta = InitMeta {
                model_id: inner.model_id,
                config: inner.config.clone(),
                vision_config: inner.config.vision_config.clone(),
                tokenizer: inner.tokenizer.clone(),
                paged_active: inner.paged.is_some(),
                scheduled_lanes,
                vision_active: inner.vision_encoder.is_some(),
                dspark_active: inner.dspark.is_some(),
                context_limits,
                state_tier,
                weight_guard,
                pool_guards,
            };
            let scheduler_state = Inkling2SchedulerState::new(inner)?;
            Ok((scheduler_state, meta))
        },
        |state, receiver| state.drive(receiver),
    )
    .await?;

    register_thread_memory_waker(meta.model_id, &thread);
    lease.commit();

    Ok(Inkling2Model {
        thread,
        config: meta.config,
        vision_config: meta.vision_config,
        tokenizer: meta.tokenizer,
        paged_active: meta.paged_active,
        scheduled_lanes: meta.scheduled_lanes,
        vision_active: meta.vision_active,
        dspark_active: meta.dspark_active,
        context_limits: meta.context_limits,
        model_assets_path: asset_dir_str,
        state_tier: meta.state_tier,
        cache_limit_guards: Mutex::new(Some((meta.weight_guard, meta.pool_guards))),
    })
}

fn load_sync(
    asset_dir: &Path,
    source_dir: &Path,
    preload_input: Option<PreloadInput>,
    gguf_vision_meta: Option<&HashMap<String, crate::utils::gguf::GgufMetaValue>>,
    is_gguf: bool,
    paged_cfg: Option<Inkling2PagedConfig>,
) -> Result<Inkling2Inner> {
    let mut model = load_weights_sync(asset_dir, preload_input, gguf_vision_meta, is_gguf)?;

    // Auto-discover colocated DSpark drafts (`mtp.safetensors` / `dspark-*.gguf`) in target-format
    // priority order, falling through to the next candidate if an earlier draft fails to load.
    let draft_candidates =
        resolve_inkling2_dspark_candidates(source_dir, asset_dir, is_gguf, None)?;
    for draft_path in &draft_candidates {
        match Inkling2DsparkModel::load_auto(draft_path, &model.config, &model.lm_head) {
            Ok(draft) => {
                model.dspark = Some(draft);
                break;
            }
            Err(err) => {
                tracing::warn!(
                    "Skipping auto-discovered Inkling2 DSpark draft '{}' (trying next candidate if any): {err}",
                    draft_path.display()
                );
            }
        }
    }

    // Initialize block-paged hybrid KV runtime on Metal (default INT8 32-token KTM).
    if crate::engine::persistence::compiled_forward_backend_available() {
        let effective_cfg =
            paged_cfg.unwrap_or_else(|| Inkling2PagedConfig::default_for_config(&model.config));
        model.paged = Some(Inkling2PagedRuntime::new_with_recency(
            &model.config,
            effective_cfg,
            Arc::clone(&model.cache_recency),
        )?);
    }

    // Materialize all weights cleanly without Metal watchdog timeout.
    let weight_arrays = model.weight_arrays();
    let weight_refs: Vec<&MxArray> = weight_arrays.iter().collect();
    let _resident = materialize_weights(&weight_refs)?;

    Ok(model)
}

fn load_weights_sync(
    asset_dir: &Path,
    preload_input: Option<PreloadInput>,
    gguf_vision_meta: Option<&HashMap<String, crate::utils::gguf::GgufMetaValue>>,
    is_gguf: bool,
) -> Result<Inkling2Inner> {
    let config_path = asset_dir.join("config.json");
    let config_raw = fs::read_to_string(&config_path).map_err(|e| {
        Error::from_reason(format!(
            "Failed to read Inkling2 config at {}: {e}",
            config_path.display()
        ))
    })?;
    let mut config = Inkling2Config::from_json_str(&config_raw, is_gguf)?;
    if config.vision_config.is_none()
        && let Some(meta) = gguf_vision_meta
    {
        config.vision_config = Some(Inkling2VisionConfig::from_gguf_metadata(
            meta,
            config.hidden_size,
        ));
    }

    let mut gguf = GgufPreloadGuard::load(preload_input.as_ref())?;

    let load_vision = config.vision_config.is_some()
        || gguf
            .preload
            .as_ref()
            .is_some_and(|p| !p.companion.is_empty());
    let raw_params = load_checkpoint(asset_dir, load_vision, gguf.preload.as_mut())?;
    drop(gguf.preload.take());

    // Strip leading `model.` prefix so keys match `load_quant_settings_from_disk`'s normalized
    // `per_layer_quant` namespace (keeping `lm_head.*`, `vision_tower.*`, `vision_projector.*`).
    let mut params: HashMap<String, MxArray> = HashMap::with_capacity(raw_params.len());
    for (k, v) in raw_params {
        let norm_key = k
            .strip_prefix("language_model.model.")
            .or_else(|| k.strip_prefix("model."))
            .map(ToOwned::to_owned)
            .unwrap_or(k);
        params.insert(norm_key, v);
    }

    // Normalize native Thinking Machines HuggingFace SafeTensors checkpoints (`llm.*` / `visual.*`)
    // with ZERO unpacking of NVFP4 (L0-19) and MXFP8 (L20-23) expert weights.
    if !is_gguf && normalize_inkling2_safetensors_params(&mut params, &config)? {
        config.gate_up_interleaved = true;
    }

    let (default_bits, default_group_size, default_mode, per_layer_quant) =
        load_quant_settings_from_disk(asset_dir, 8, 32)?;
    let default_plq = PerLayerQuant {
        bits: default_bits,
        group_size: default_group_size,
        mode: default_mode.unwrap_or(PerLayerMode::Affine),
        layout: Default::default(),
        input_amax: None,
    };

    let hidden_size = config.hidden_size;
    let padded_vocab = config.effective_padded_vocab_size();
    let inv_sqrt_d = 1.0_f64 / (hidden_size as f64).sqrt();

    // 1. Embeddings & embed_norm (keep packed affine embedding sidecars in native F16 so
    // `dequantize` output precision stays bit-identical to GGUF Q8_0; gather-then-dequant
    // only touches T rows and costs ~0.03 ms).
    let mut embed_tokens = Embedding::new(padded_vocab as u32, hidden_size as u32)?;
    load_embedding_affine_or_bf16(
        &mut embed_tokens,
        &params,
        "embed_tokens",
        &per_layer_quant,
        default_plq,
        FAMILY_NAME,
    )?;
    let embed_norm = require_tensor(&params, "embed_norm.weight")?.astype(DType::Float32)?;

    // 2. Output head & output AttnRes query (`[1, D]` row-major & `[D, 1]` transposed FP32)
    let output_res_query = require_tensor(&params, "output_res_query.weight")?
        .astype(DType::Float32)?
        .reshape(&[1, hidden_size as i64])?
        .mul_scalar(inv_sqrt_d)?;
    let output_res_query_t = output_res_query.transpose(None)?;
    let output_norm = require_tensor(&params, "norm.weight")?.astype(DType::Float32)?;
    let mut lm_head = LinearProj::Standard(Linear::new(
        hidden_size as u32,
        padded_vocab as u32,
        Some(false),
    )?);
    load_linear_proj_quantized_or_bf16(
        &mut lm_head,
        &params,
        "lm_head",
        &per_layer_quant,
        default_plq,
        FAMILY_NAME,
    )?;
    // Compact exact symmetric GGUF Q8_0 (`biases == -128 * scales`) to native Float16 scales
    // with zero biases (4x less sidecar DRAM), falling back to F32 hoisting for asymmetric affine.
    lm_head.try_compact_symmetric_q8_or_promote_f32(DType::BFloat16)?;

    // Global cross-pass message queries (Unsloth GGUF format)
    let global_loop_res = params.get("loop_res_query.weight");
    let global_loop_kv_res = params.get("loop_kv_res_query.weight");

    // 3. Build physical layers `0..num_physical_layers`
    let n_phys = config.num_physical_layers();
    let mut layers = Vec::with_capacity(n_phys);
    for phys in 0..n_phys {
        let layer = build_physical_layer(
            phys,
            &config,
            &params,
            &per_layer_quant,
            default_plq,
            inv_sqrt_d,
            global_loop_res,
            global_loop_kv_res,
        )?;
        layers.push(layer);
    }

    // 4. Optional Vision Encoder (`mmproj-BF16.gguf`)
    let vision_encoder = if params.contains_key("vision_tower.patch_embed.weight") {
        let vcfg = config
            .vision_config
            .clone()
            .unwrap_or_else(|| Inkling2VisionConfig {
                projection_dim: hidden_size,
                ..Inkling2VisionConfig::default()
            });
        Some(build_vision_encoder(&vcfg, &params)?)
    } else {
        None
    };

    // 5. Optional Tokenizer
    let tok_path = asset_dir.join("tokenizer.json");
    let tokenizer = if tok_path.is_file() {
        Some(Arc::new(
            Qwen3Tokenizer::from_file(&tok_path).map_err(Error::from_reason)?,
        ))
    } else {
        None
    };

    let model = Inkling2Inner {
        config,
        embed_tokens,
        embed_norm,
        layers,
        output_res_query,
        output_res_query_t,
        output_norm,
        lm_head,
        vision_encoder,
        tokenizer,
        dspark: None,
        dspark_turn_contexts: HashMap::new(),
        scheduled_dspark: Inkling2ScheduledDsparkStates::default(),
        paged: None,
        flat_cache: None,
        cached_token_history: Vec::new(),
        cached_image_key: None,
        turn_cancel: None,
        model_id: next_inkling2_model_id(),
        memory_plan: None,
        memory_registration: None,
        wired: None,
        cache_recency: Arc::new(CacheRecency::new()),
        decode_share: default_decode_share(),
        max_context: None,
        activations: Activations::Auto,
        disk_options: DiskTierOptions::default(),
        prefill_runs: HashMap::new(),
    };

    Ok(model)
}

#[allow(clippy::too_many_arguments)]
fn build_physical_layer(
    phys: usize,
    config: &Inkling2Config,
    params: &HashMap<String, MxArray>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
    inv_sqrt_d: f64,
    global_loop_res: Option<&MxArray>,
    global_loop_kv_res: Option<&MxArray>,
) -> Result<Inkling2PhysicalLayer> {
    let prefix = format!("layers.{phys}");
    let is_local = config.is_local_physical(phys);
    let d = config.hidden_size;
    let n_heads = config.num_attention_heads;
    let n_kv = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let d_rel = config.d_rel;
    let rel_extent = config.rel_extent;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv * head_dim;
    let r_dim = n_heads * d_rel;

    // AttnRes queries (pre-scaled by `1 / sqrt(D)`, both row-major `[Q, D]` & transposed `[D, Q]` FP32)
    let attn_res_query = if let Some(q_raw) = params.get(&format!("{prefix}.attn_res_query.weight"))
    {
        let expected_q = if is_local { 1 } else { 3 };
        Some(
            q_raw
                .astype(DType::Float32)?
                .reshape(&[expected_q, d as i64])?
                .mul_scalar(inv_sqrt_d)?,
        )
    } else if phys == 0 {
        None
    } else {
        return Err(Error::from_reason(format!(
            "Inkling2: missing {prefix}.attn_res_query.weight on physical layer {phys}"
        )));
    };
    let attn_res_query_t = match attn_res_query {
        Some(ref q) => Some(q.transpose(None)?),
        None => None,
    };

    let ffn_res_query = require_tensor(params, &format!("{prefix}.ffn_res_query.weight"))?
        .astype(DType::Float32)?
        .reshape(&[1, d as i64])?
        .mul_scalar(inv_sqrt_d)?;
    let ffn_res_query_t = ffn_res_query.transpose(None)?;

    // Sublayer norms (kept in FP32 so hot-path RMSNorm never re-casts)
    let attn_inp_norm = if is_local {
        AttnInputNorm::Local(
            require_tensor(params, &format!("{prefix}.input_layernorm.weight"))?
                .astype(DType::Float32)?,
        )
    } else {
        let q_norm = require_tensor(params, &format!("{prefix}.self_attn.q_inp_norm.weight"))?
            .astype(DType::Float32)?;
        let k_norm = require_tensor(params, &format!("{prefix}.self_attn.k_inp_norm.weight"))?
            .astype(DType::Float32)?;
        let v_norm = require_tensor(params, &format!("{prefix}.self_attn.v_inp_norm.weight"))?
            .astype(DType::Float32)?;
        let qkv_norm = MxArray::stack(vec![&q_norm, &k_norm, &v_norm], Some(0))?;
        AttnInputNorm::Global {
            q_norm,
            k_norm,
            v_norm,
            qkv_norm,
        }
    };
    let ffn_norm = require_tensor(params, &format!("{prefix}.post_attention_layernorm.weight"))?
        .astype(DType::Float32)?;

    // Attention projections (compact exact symmetric Q8_0 to native F16 scales BEFORE row-concat fusion!)
    let load_proj = |sub: &str, in_f: usize, out_f: usize| -> Result<LinearProj> {
        let base = format!("{prefix}.self_attn.{sub}");
        let mut proj = LinearProj::Standard(Linear::new(in_f as u32, out_f as u32, Some(false))?);
        load_linear_proj_quantized_or_bf16(
            &mut proj,
            params,
            &base,
            per_layer_quant,
            default_plq,
            FAMILY_NAME,
        )?;
        proj.try_compact_symmetric_q8_or_promote_f32(DType::BFloat16)?;
        Ok(proj)
    };

    let q_proj = load_proj("q_proj", d, q_dim)?;
    let k_proj = load_proj("k_proj", d, kv_dim)?;
    let v_proj = load_proj("v_proj", d, kv_dim)?;
    let rel_proj = load_proj("rel_proj", d, r_dim)?;
    let gate_proj = load_proj("gate_proj", d, q_dim)?;
    let sink_proj = load_proj("sink_proj", d, n_heads)?;
    let o_proj = load_proj("o_proj", q_dim, d)?;

    let k_conv = require_tensor(params, &format!("{prefix}.self_attn.k_conv.weight"))?;
    let v_conv = require_tensor(params, &format!("{prefix}.self_attn.v_conv.weight"))?;
    let post_conv = require_tensor(params, &format!("{prefix}.self_attn.post_conv.weight"))?;

    let (k_conv_proj, v_conv_proj) = if !is_local {
        let dyn_out = n_kv * config.sconv_kernel_size;
        (
            Some(load_proj("k_conv_proj", d, dyn_out)?),
            Some(load_proj("v_conv_proj", d, dyn_out)?),
        )
    } else {
        (None, None)
    };

    // Clip weights and relative bias table
    let q_clip = require_tensor(params, &format!("{prefix}.self_attn.q_clip.weight"))?
        .astype(DType::Float32)?
        .reshape(&[1, n_heads as i64, head_dim as i64])?;
    let rel_clip = require_tensor(params, &format!("{prefix}.self_attn.rel_clip.weight"))?
        .astype(DType::Float32)?
        .reshape(&[1, n_heads as i64, d_rel as i64])?;
    let rel_bias_raw = require_tensor(params, &format!("{prefix}.self_attn.rel_bias.weight"))?
        .astype(DType::Float32)?;
    let rb_shape = rel_bias_raw.shape()?;
    // GGUF loads `[rel_extent, d_rel]` (`[1024, 16]`); HF loads `[d_rel, rel_extent]` (`[16, 1024]`).
    let (rel_bias_t, rel_bias_w) = if rb_shape.as_ref() == [rel_extent as i64, d_rel as i64] {
        (rel_bias_raw.transpose(None)?, rel_bias_raw)
    } else if rb_shape.as_ref() == [d_rel as i64, rel_extent as i64] {
        let w = rel_bias_raw.transpose(None)?;
        (rel_bias_raw, w)
    } else {
        return Err(Error::from_reason(format!(
            "Inkling2: unexpected rel_bias shape {:?} on layer {phys} (expected [{rel_extent}, {d_rel}] or [{d_rel}, {rel_extent}])",
            rb_shape.as_ref()
        )));
    };
    // Ensure `rel_bias_w` is row-contiguous in memory (`[rel_extent, d_rel]`) for on-the-fly Metal dot.
    let rel_bias_w = rel_bias_w.copy()?;
    let qk_thresh_scalar = MxArray::scalar_float(config.qk_l2_clip_threshold)?;
    let rel_thresh_scalar = MxArray::scalar_float(config.rel_l2_clip_threshold)?;
    let one_scalar = MxArray::scalar_float(1.0)?;

    let mut self_attn = Inkling2Attention {
        num_heads: n_heads,
        num_kv_heads: n_kv,
        head_dim,
        d_rel,
        rel_extent,
        sliding_window: config.sliding_window,
        is_local,
        qk_l2_clip_threshold: config.qk_l2_clip_threshold,
        rel_l2_clip_threshold: config.rel_l2_clip_threshold,
        log_scaling_n_floor: config.log_scaling_n_floor,
        log_scaling_alpha: config.log_scaling_alpha,
        rms_norm_eps: config.rms_norm_eps as f32,
        q_proj,
        k_proj,
        v_proj,
        rel_proj,
        gate_proj,
        sink_proj,
        o_proj,
        fused_qrg: None,
        fused_qrgkv: None,
        fused_qrg_has_sink: false,
        fused_k_with_conv: None,
        fused_v_with_conv: None,
        k_conv,
        v_conv,
        post_conv,
        k_conv_proj,
        v_conv_proj,
        q_clip,
        rel_clip,
        rel_bias_t,
        rel_bias_w,
        qk_thresh_scalar,
        rel_thresh_scalar,
        one_scalar,
    };
    self_attn.try_fuse_projections()?;

    // MoE block (hoist Q8_0 Float16 sidecars on blk.20..23 MoE experts!)
    let router_weight = require_tensor(params, &format!("{prefix}.mlp.gate.weight"))?;
    let router_bias = params
        .get(&format!("{prefix}.mlp.gate.e_score_correction_bias"))
        .cloned();
    let mut gate_up_proj = load_expert_switch_proj(
        params,
        &format!("{prefix}.mlp.experts.gate_up_proj"),
        per_layer_quant,
        default_plq,
    )?;
    let mut down_proj = load_expert_switch_proj(
        params,
        &format!("{prefix}.mlp.experts.down_proj"),
        per_layer_quant,
        default_plq,
    )?;
    // Compact exact symmetric GGUF Q8_0 (`biases == -128 * scales`) on `blk.20..23` MoE experts
    // to native Float16 scales with zero biases (-255 MiB RAM, 2x less sidecar DRAM, no x_sum ALU),
    // falling back to BF16 sidecar hoisting for non-symmetric affine Q8_0.
    gate_up_proj.try_compact_symmetric_q8_or_promote_bf16()?;
    down_proj.try_compact_symmetric_q8_or_promote_bf16()?;
    let mlp_post_conv = require_tensor(params, &format!("{prefix}.mlp.post_conv.weight"))?;
    let moe = Inkling2MoE::new(
        config.num_experts,
        config.num_experts_per_tok,
        config.effective_moe_intermediate_size(),
        config.route_scale,
        config.mlp_gate_softcap,
        config.mlp_up_softcap,
        config.gate_up_interleaved,
        router_weight,
        router_bias,
        gate_up_proj,
        down_proj,
        mlp_post_conv,
    )?;

    // Optional Loop Controllers for core layers (`6..17`)
    let loop_ctrl = if let Some(core_idx) = config.core_offset(phys) {
        let msg_queries_raw =
            if let Some(per_layer) = params.get(&format!("{prefix}.loop_msg_query.weight")) {
                // Mainline `inkling2_loop`: already `[4, D]` in `[q, k, v, mlp]` order.
                per_layer.astype(DType::Float32)?.reshape(&[4, d as i64])?
            } else if let (Some(lq), Some(lkv)) = (global_loop_res, global_loop_kv_res) {
                // Unsloth `inkling2`: `loop_res_query` `[2*C, D]` (`[2i]=attn, [2i+1]=mlp`)
                // and `loop_kv_res_query` `[2*C, D]` (`[2i]=k, [2i+1]=v`).
                let lq_f32 = lq.astype(DType::Float32)?;
                let lkv_f32 = lkv.astype(DType::Float32)?;
                let row0 = (2 * core_idx) as i64;
                let row1 = row0 + 1;
                let q_attn = lq_f32.slice(&[row0, 0], &[row0 + 1, d as i64])?;
                let q_k = lkv_f32.slice(&[row0, 0], &[row0 + 1, d as i64])?;
                let q_v = lkv_f32.slice(&[row1, 0], &[row1 + 1, d as i64])?;
                let q_mlp = lq_f32.slice(&[row1, 0], &[row1 + 1, d as i64])?;
                MxArray::concatenate_many(vec![&q_attn, &q_k, &q_v, &q_mlp], Some(0))?
            } else {
                return Err(Error::from_reason(format!(
                    "Inkling2: missing loop message queries for core physical layer {phys}"
                )));
            };
        let msg_queries = msg_queries_raw.mul_scalar(inv_sqrt_d)?;

        let load_ctrl = |stream: &str| -> Result<LoopController> {
            let base = format!("{prefix}.loop_ctrl.{stream}");
            LoopController::new(
                require_tensor(params, &format!("{base}.down_proj.weight"))?,
                require_tensor(params, &format!("{base}.gain_proj.weight"))?,
                require_tensor(params, &format!("{base}.forget_proj.weight"))?,
            )
        };
        Some(LoopLayerControllers::new(
            msg_queries,
            load_ctrl("q")?,
            load_ctrl("k")?,
            load_ctrl("v")?,
            load_ctrl("mlp")?,
        )?)
    } else {
        None
    };

    Ok(Inkling2PhysicalLayer {
        phys_idx: phys,
        is_local,
        rms_norm_eps: config.rms_norm_eps as f32,
        attn_res_query,
        attn_res_query_t,
        ffn_res_query,
        ffn_res_query_t,
        attn_inp_norm,
        ffn_norm,
        self_attn,
        moe,
        loop_ctrl,
    })
}

fn load_expert_switch_proj(
    params: &HashMap<String, MxArray>,
    base: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Result<ExpertSwitchProj> {
    if let Some(scales) = params.get(&format!("{base}.scales")) {
        // Native SafeTensors NVFP4 / MXFP8 checkpoints (`quant_method: "modelopt"`) have no
        // per-layer entries in `per_layer_quant`, store Uint8 FP8 scales (E4M3 for NVFP4,
        // UE8M0 for MXFP8), and carry no `.biases`. Detect directly from tensor metadata & shapes
        // so packed weights stay zero-copy!
        let has_override = per_layer_quant.contains_key(base);
        if !has_override
            && scales.dtype()? == DType::Uint8
            && !params.contains_key(&format!("{base}.biases"))
        {
            let w = require_tensor(params, &format!("{base}.weight"))?;
            let w_shape = w.shape()?;
            let s_shape = scales.shape()?;
            if w_shape.len() == 3
                && s_shape.len() == 3
                && w_shape[0] == s_shape[0]
                && w_shape[1] == s_shape[1]
            {
                if let Some(outer) = params.get(&format!("{base}.scale")) {
                    // NVFP4: uint32 packed 4-bit weights (8 elems/u32) + uint8 E4M3 scales (16 elems/scale) -> w_last == s_last * 2
                    if w_shape[2] == s_shape[2] * 2 {
                        let q = QuantizedSwitchLinear::new(
                            w,
                            scales.clone(),
                            None,
                            16,
                            4,
                            NVFP4_MODE.to_string(),
                        )
                        .with_outer_scale(outer.clone());
                        return Ok(ExpertSwitchProj::Quantized(q));
                    }
                } else if w_shape[2] == s_shape[2] * 8 {
                    // MXFP8: uint32 packed FP8 E4M3 weights (4 elems/u32) + uint8 UE8M0 scales (32 elems/scale) -> w_last == s_last * 8
                    let q = QuantizedSwitchLinear::new(
                        w,
                        scales.clone(),
                        None,
                        32,
                        8,
                        MXFP8_MODE.to_string(),
                    );
                    return Ok(ExpertSwitchProj::Quantized(q));
                }
            }
        }

        let plq = effective_plq_for(base, per_layer_quant, default_plq, None);
        let qsl = build_qsl(
            params,
            base,
            plq,
            QslBuildSpec {
                family: "Inkling2",
                validate_storage: false,
                fp8: QslFp8Policy::RejectUnsupported,
                sym8: QslSym8Policy::RejectUnsupported,
                nvfp4_only: false,
                affine_optional_outer_scale: true,
                nvfp4_outer_scale: None,
            },
        )?;
        let qsl = qsl.ok_or_else(|| {
            Error::from_reason(format!(
                "Inkling2: failed to build quantized switch linear for '{base}' (mode={:?})",
                plq.mode
            ))
        })?;
        return Ok(ExpertSwitchProj::Quantized(qsl));
    }

    let w = require_tensor(params, &format!("{base}.weight"))?;
    Ok(ExpertSwitchProj::Dense(w))
}

fn build_vision_encoder(
    config: &Inkling2VisionConfig,
    params: &HashMap<String, MxArray>,
) -> Result<Inkling2VisionEncoder> {
    let p = config.patch_size as i64;
    let d = config.hidden_size as i64;
    let eps = Some(config.layer_norm_eps);

    let class_embedding = require_tensor(params, "vision_tower.class_embedding")?;
    let class_position_embedding =
        require_tensor(params, "vision_tower.class_position_embedding.weight")?;

    // GGUF loads `v.patch_embd.weight` as OIHW `[1024, 3, 20, 20]`.
    // Handle OHWI `[1024, 20, 20, 3]` or pre-flattened `[1024, 1200]` cleanly too.
    let patch_raw = require_tensor(params, "vision_tower.patch_embed.weight")?;
    let p_shape = patch_raw.shape()?;
    let patch_weight_flat = match p_shape.as_ref() {
        [o, 3, ph, pw] if *o == d && *ph == p && *pw == p => patch_raw.reshape(&[d, 3 * p * p])?,
        [o, ph, pw, 3] if *o == d && *ph == p && *pw == p => patch_raw
            .transpose(Some(&[0, 3, 1, 2]))?
            .reshape(&[d, 3 * p * p])?,
        [o, flat] if *o == d && *flat == 3 * p * p => patch_raw,
        other => {
            return Err(Error::from_reason(format!(
                "Inkling2 vision: unexpected patch_embed.weight shape {other:?}"
            )));
        }
    };

    let position_embedding = require_tensor(params, "vision_tower.position_embedding.weight")?;
    let pre_ln = LayerNorm::from_weights(
        &require_tensor(params, "vision_tower.pre_ln.weight")?,
        Some(&require_tensor(params, "vision_tower.pre_ln.bias")?),
        eps,
    )?;

    let mut blocks = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let bp = format!("vision_tower.blocks.{i}");
        let norm1 = LayerNorm::from_weights(
            &require_tensor(params, &format!("{bp}.norm1.weight"))?,
            Some(&require_tensor(params, &format!("{bp}.norm1.bias"))?),
            eps,
        )?;
        let norm2 = LayerNorm::from_weights(
            &require_tensor(params, &format!("{bp}.norm2.weight"))?,
            Some(&require_tensor(params, &format!("{bp}.norm2.bias"))?),
            eps,
        )?;
        let load_lin = |name: &str| -> Result<Linear> {
            Linear::from_weights(
                &require_tensor(params, &format!("{bp}.{name}.weight"))?,
                params.get(&format!("{bp}.{name}.bias")),
            )
        };
        blocks.push(Inkling2VisionBlock {
            norm1,
            q_proj: load_lin("attn.q_proj")?,
            k_proj: load_lin("attn.k_proj")?,
            v_proj: load_lin("attn.v_proj")?,
            o_proj: load_lin("attn.o_proj")?,
            ls1: require_tensor(params, &format!("{bp}.ls1.weight"))?,
            norm2,
            fc1: load_lin("mlp.fc1")?,
            fc2: load_lin("mlp.fc2")?,
            ls2: require_tensor(params, &format!("{bp}.ls2.weight"))?,
        });
    }

    let proj_fc1 = Linear::from_weights(
        &require_tensor(params, "vision_projector.fc1.weight")?,
        params.get("vision_projector.fc1.bias"),
    )?;
    let proj_fc2 = Linear::from_weights(
        &require_tensor(params, "vision_projector.fc2.weight")?,
        params.get("vision_projector.fc2.bias"),
    )?;

    Ok(Inkling2VisionEncoder {
        config: config.clone(),
        class_embedding,
        class_position_embedding,
        patch_weight_flat,
        position_embedding,
        pre_ln,
        blocks,
        proj_fc1,
        proj_fc2,
    })
}

fn require_tensor(params: &HashMap<String, MxArray>, key: &str) -> Result<MxArray> {
    params
        .get(key)
        .cloned()
        .ok_or_else(|| Error::from_reason(format!("Inkling2: missing required tensor '{key}'")))
}

/// Normalize native Thinking Machines HuggingFace SafeTensors keys (`llm.*` / `visual.*`,
/// after leading `model.` strip) into the canonical internal Inkling2 namespace used by
/// `load_weights_sync` with ZERO unpacking of quantized NVFP4 (L0–19) and MXFP8 (L20–23)
/// MoE expert weights.
///
/// Returns `Ok(true)` if the checkpoint used the Thinking Machines SafeTensors layout, or
/// `Ok(false)` (leaving `params` untouched) for GGUF / pre-normalized checkpoints.
fn normalize_inkling2_safetensors_params(
    params: &mut HashMap<String, MxArray>,
    config: &Inkling2Config,
) -> Result<bool> {
    if !params.contains_key("llm.embed.weight")
        && !params.contains_key("llm.layers.0.attn.wq_du.weight")
    {
        return Ok(false);
    }

    let raw = std::mem::take(params);
    let mut out: HashMap<String, MxArray> = HashMap::with_capacity(raw.len());

    // Pair `weight_amax` and `weight_outer_scale_denominator` per `(layer, proj)` for NVFP4 outer scales.
    let mut expert_amax: HashMap<(usize, &'static str), MxArray> = HashMap::new();
    let mut expert_denom: HashMap<(usize, &'static str), MxArray> = HashMap::new();

    // Reverse index `[3, 2, 1, 0]` to flip Global dynamic `static_w` from newest-to-oldest
    // (`[4, C]` in SafeTensors) into oldest-to-newest (`[C, 4]` expected by `dynamic_short_conv`
    // and `inkling2_sconv_k4_kv_prep`).
    let rev_k4 = MxArray::from_int32(&[3, 2, 1, 0], &[4])?;
    let vision_patch = config
        .vision_config
        .as_ref()
        .map_or(20i64, |v| v.patch_size as i64);

    for (key, tensor) in raw {
        // 1. Top-level LLM weights
        match key.as_str() {
            "llm.embed.weight" => {
                out.insert("embed_tokens.weight".to_string(), tensor);
                continue;
            }
            "llm.embed_norm.weight" => {
                out.insert("embed_norm.weight".to_string(), tensor);
                continue;
            }
            "llm.norm.weight" => {
                out.insert("norm.weight".to_string(), tensor);
                continue;
            }
            "llm.unembed.weight" => {
                out.insert("lm_head.weight".to_string(), tensor);
                continue;
            }
            "llm.attn_res.output_query" => {
                out.insert("output_res_query.weight".to_string(), tensor);
                continue;
            }
            "llm.middle_cycle_cross_pass_queries" => {
                out.insert("loop_res_query.weight".to_string(), tensor);
                continue;
            }
            "llm.middle_cycle_cross_pass_kv_queries" => {
                out.insert("loop_kv_res_query.weight".to_string(), tensor);
                continue;
            }
            _ => {}
        }

        // 2. AttnRes queries (`attn_queries.{idx}` -> physical layer `idx + 1`, since phys 0 has none)
        if let Some(rest) = key.strip_prefix("llm.attn_res.attn_queries.") {
            let idx: usize = rest.parse().map_err(|e| {
                Error::from_reason(format!(
                    "Inkling2 ST: invalid attn_queries index '{rest}': {e}"
                ))
            })?;
            out.insert(format!("layers.{}.attn_res_query.weight", idx + 1), tensor);
            continue;
        }
        if let Some(rest) = key.strip_prefix("llm.attn_res.mlp_queries.") {
            let idx: usize = rest.parse().map_err(|e| {
                Error::from_reason(format!(
                    "Inkling2 ST: invalid mlp_queries index '{rest}': {e}"
                ))
            })?;
            out.insert(format!("layers.{idx}.ffn_res_query.weight"), tensor);
            continue;
        }

        // 3. Per-layer LLM weights (`llm.layers.{L}.*`)
        if let Some(rest) = key.strip_prefix("llm.layers.")
            && let Some((layer_str, suffix)) = rest.split_once('.')
        {
            let layer_idx: usize = layer_str.parse().map_err(|e| {
                Error::from_reason(format!(
                    "Inkling2 ST: invalid layer index '{layer_str}': {e}"
                ))
            })?;
            let lp = format!("layers.{layer_idx}");
            match suffix {
                "attn_norm.weight" => {
                    out.insert(format!("{lp}.input_layernorm.weight"), tensor);
                }
                "q_input_norm.weight" => {
                    out.insert(format!("{lp}.self_attn.q_inp_norm.weight"), tensor);
                }
                "k_input_norm.weight" => {
                    out.insert(format!("{lp}.self_attn.k_inp_norm.weight"), tensor);
                }
                "v_input_norm.weight" => {
                    out.insert(format!("{lp}.self_attn.v_inp_norm.weight"), tensor);
                }
                "mlp_norm.weight" => {
                    out.insert(format!("{lp}.post_attention_layernorm.weight"), tensor);
                }
                "attn.wq_du.weight" => {
                    out.insert(format!("{lp}.self_attn.q_proj.weight"), tensor);
                }
                "attn.wk_dv.weight" => {
                    out.insert(format!("{lp}.self_attn.k_proj.weight"), tensor);
                }
                "attn.wv_dv.weight" => {
                    out.insert(format!("{lp}.self_attn.v_proj.weight"), tensor);
                }
                "attn.wr_du.weight" => {
                    out.insert(format!("{lp}.self_attn.rel_proj.weight"), tensor);
                }
                "attn.gate_attn.weight" => {
                    out.insert(format!("{lp}.self_attn.gate_proj.weight"), tensor);
                }
                "attn.attn_sink_proj.weight" => {
                    out.insert(format!("{lp}.self_attn.sink_proj.weight"), tensor);
                }
                "attn.wo_ud.weight" => {
                    out.insert(format!("{lp}.self_attn.o_proj.weight"), tensor);
                }
                "attn.q_l2_clip.scale" => {
                    out.insert(format!("{lp}.self_attn.q_clip.weight"), tensor);
                }
                "attn.rel_l2_clip.scale" => {
                    out.insert(format!("{lp}.self_attn.rel_clip.weight"), tensor);
                }
                "attn.rel_logits_proj.proj" => {
                    out.insert(format!("{lp}.self_attn.rel_bias.weight"), tensor);
                }
                // Static ShortConv (`[C, 1, 4]` oldest-to-newest -> `[C, 4]`)
                "attn.k_sconv.weight" => {
                    let w = if tensor.ndim()? == 3 {
                        tensor.squeeze(Some(&[1]))?
                    } else {
                        tensor
                    };
                    out.insert(format!("{lp}.self_attn.k_conv.weight"), w);
                }
                "attn.v_sconv.weight" => {
                    let w = if tensor.ndim()? == 3 {
                        tensor.squeeze(Some(&[1]))?
                    } else {
                        tensor
                    };
                    out.insert(format!("{lp}.self_attn.v_conv.weight"), w);
                }
                "attn_sconv.weight" => {
                    let w = if tensor.ndim()? == 3 {
                        tensor.squeeze(Some(&[1]))?
                    } else {
                        tensor
                    };
                    out.insert(format!("{lp}.self_attn.post_conv.weight"), w);
                }
                "mlp_sconv.weight" => {
                    let w = if tensor.ndim()? == 3 {
                        tensor.squeeze(Some(&[1]))?
                    } else {
                        tensor
                    };
                    out.insert(format!("{lp}.mlp.post_conv.weight"), w);
                }
                // Dynamic Global ShortConv (`static_w` `[4, C]` newest-to-oldest -> `[C, 4]` oldest-to-newest)
                "attn.k_sconv.static_w" => {
                    let w = tensor.transpose(None)?.take(&rev_k4, 1)?;
                    out.insert(format!("{lp}.self_attn.k_conv.weight"), w);
                }
                "attn.v_sconv.static_w" => {
                    let w = tensor.transpose(None)?.take(&rev_k4, 1)?;
                    out.insert(format!("{lp}.self_attn.v_conv.weight"), w);
                }
                "attn.k_sconv.weight_proj" => {
                    out.insert(format!("{lp}.self_attn.k_conv_proj.weight"), tensor);
                }
                "attn.v_sconv.weight_proj" => {
                    out.insert(format!("{lp}.self_attn.v_conv_proj.weight"), tensor);
                }
                // MoE router
                "mlp.gate.weight" => {
                    out.insert(format!("{lp}.mlp.gate.weight"), tensor);
                }
                "mlp.gate.bias" => {
                    out.insert(format!("{lp}.mlp.gate.e_score_correction_bias"), tensor);
                }
                // MoE experts: ZERO-COPY `view(DType::Uint32)` for NVFP4 (L0–19) & MXFP8 (L20–23)!
                "mlp.experts.w13_weight" => {
                    let w = if tensor.dtype()? == DType::Uint8 {
                        tensor.view(DType::Uint32)?
                    } else {
                        tensor
                    };
                    out.insert(format!("{lp}.mlp.experts.gate_up_proj.weight"), w);
                }
                "mlp.experts.w13_weight_scale" => {
                    out.insert(format!("{lp}.mlp.experts.gate_up_proj.scales"), tensor);
                }
                "mlp.experts.w13_weight_amax" => {
                    expert_amax.insert((layer_idx, "gate_up_proj"), tensor);
                }
                "mlp.experts.w13_weight_outer_scale_denominator" => {
                    expert_denom.insert((layer_idx, "gate_up_proj"), tensor);
                }
                "mlp.experts.w2_weight" => {
                    let w = if tensor.dtype()? == DType::Uint8 {
                        tensor.view(DType::Uint32)?
                    } else {
                        tensor
                    };
                    out.insert(format!("{lp}.mlp.experts.down_proj.weight"), w);
                }
                "mlp.experts.w2_weight_scale" => {
                    out.insert(format!("{lp}.mlp.experts.down_proj.scales"), tensor);
                }
                "mlp.experts.w2_weight_amax" => {
                    expert_amax.insert((layer_idx, "down_proj"), tensor);
                }
                "mlp.experts.w2_weight_outer_scale_denominator" => {
                    expert_denom.insert((layer_idx, "down_proj"), tensor);
                }
                // Loop controllers (`controllers.{attn->q, k->k, v->v, mlp->mlp}.{down,gain,forget}_weight`)
                _ if suffix.starts_with("controllers.") => {
                    if let Some(ctrl_rest) = suffix.strip_prefix("controllers.")
                        && let Some((stream_raw, proj_raw)) = ctrl_rest.split_once('.')
                    {
                        let stream = match stream_raw {
                            "attn" => "q",
                            "k" => "k",
                            "v" => "v",
                            "mlp" => "mlp",
                            other => {
                                return Err(Error::from_reason(format!(
                                    "Inkling2 ST: unknown controller stream '{other}' in '{key}'"
                                )));
                            }
                        };
                        let proj = match proj_raw {
                            "down_weight" => "down_proj.weight",
                            "gain_weight" => "gain_proj.weight",
                            "forget_weight" => "forget_proj.weight",
                            other => {
                                return Err(Error::from_reason(format!(
                                    "Inkling2 ST: unknown controller weight '{other}' in '{key}'"
                                )));
                            }
                        };
                        out.insert(format!("{lp}.loop_ctrl.{stream}.{proj}"), tensor);
                    } else {
                        out.insert(key, tensor);
                    }
                }
                _ => {
                    out.insert(key, tensor);
                }
            }
            continue;
        }

        // 4. Vision tower + projector (`visual.*`)
        if let Some(vrest) = key.strip_prefix("visual.") {
            match vrest {
                "backbone.embeddings.class_embedding" => {
                    out.insert("vision_tower.class_embedding".to_string(), tensor);
                }
                "backbone.embeddings.cls_position_embedding" => {
                    out.insert(
                        "vision_tower.class_position_embedding.weight".to_string(),
                        tensor,
                    );
                }
                "backbone.embeddings.position_embedding" => {
                    out.insert("vision_tower.position_embedding.weight".to_string(), tensor);
                }
                "backbone.embeddings.patch_embedding.weight" => {
                    // SafeTensors stores `[D, P*P*3]` in HWC `(P, P, 3)` flattened order;
                    // `Inkling2VisionEncoder` expects CHW `(3, P, P)` flattened `[D, 3*P*P]`.
                    let shape = tensor.shape()?;
                    let converted =
                        if shape.len() == 2 && shape[1] == 3 * vision_patch * vision_patch {
                            let d_vis = shape[0];
                            tensor
                                .reshape(&[d_vis, vision_patch, vision_patch, 3])?
                                .transpose(Some(&[0, 3, 1, 2]))?
                                .reshape(&[d_vis, 3 * vision_patch * vision_patch])?
                        } else {
                            tensor
                        };
                    out.insert("vision_tower.patch_embed.weight".to_string(), converted);
                }
                "backbone.ln_pre.weight" => {
                    out.insert("vision_tower.pre_ln.weight".to_string(), tensor);
                }
                "backbone.ln_pre.bias" => {
                    out.insert("vision_tower.pre_ln.bias".to_string(), tensor);
                }
                "proj_in.weight" => {
                    out.insert("vision_projector.fc1.weight".to_string(), tensor);
                }
                "proj_in.bias" => {
                    out.insert("vision_projector.fc1.bias".to_string(), tensor);
                }
                "proj_out.weight" => {
                    out.insert("vision_projector.fc2.weight".to_string(), tensor);
                }
                "proj_out.bias" => {
                    out.insert("vision_projector.fc2.bias".to_string(), tensor);
                }
                _ if vrest.starts_with("backbone.layers.") => {
                    if let Some(lrest) = vrest.strip_prefix("backbone.layers.")
                        && let Some((blk_str, sub)) = lrest.split_once('.')
                    {
                        let bp = format!("vision_tower.blocks.{blk_str}");
                        let mapped = match sub {
                            "ln_1.weight" => Some("norm1.weight"),
                            "ln_1.bias" => Some("norm1.bias"),
                            "ln_2.weight" => Some("norm2.weight"),
                            "ln_2.bias" => Some("norm2.bias"),
                            "ls_1.gamma" => Some("ls1.weight"),
                            "ls_2.gamma" => Some("ls2.weight"),
                            "self_attn.q_proj.weight" => Some("attn.q_proj.weight"),
                            "self_attn.q_proj.bias" => Some("attn.q_proj.bias"),
                            "self_attn.k_proj.weight" => Some("attn.k_proj.weight"),
                            "self_attn.k_proj.bias" => Some("attn.k_proj.bias"),
                            "self_attn.v_proj.weight" => Some("attn.v_proj.weight"),
                            "self_attn.v_proj.bias" => Some("attn.v_proj.bias"),
                            "self_attn.out_proj.weight" => Some("attn.o_proj.weight"),
                            "self_attn.out_proj.bias" => Some("attn.o_proj.bias"),
                            "mlp.c_fc.weight" => Some("mlp.fc1.weight"),
                            "mlp.c_fc.bias" => Some("mlp.fc1.bias"),
                            "mlp.c_proj.weight" => Some("mlp.fc2.weight"),
                            "mlp.c_proj.bias" => Some("mlp.fc2.bias"),
                            _ => None,
                        };
                        if let Some(m) = mapped {
                            out.insert(format!("{bp}.{m}"), tensor);
                        } else {
                            out.insert(key, tensor);
                        }
                    } else {
                        out.insert(key, tensor);
                    }
                }
                _ => {
                    out.insert(key, tensor);
                }
            }
            continue;
        }

        out.insert(key, tensor);
    }

    // Materialize per-expert NVFP4 `outer_scale = amax / outer_scale_denominator` (`[E]` F32).
    for ((layer_idx, proj_name), amax) in expert_amax {
        let denom = expert_denom.remove(&(layer_idx, proj_name)).ok_or_else(|| {
            Error::from_reason(format!(
                "Inkling2 ST: missing outer_scale_denominator for layers.{layer_idx}.mlp.experts.{proj_name}"
            ))
        })?;
        let outer = amax
            .astype(DType::Float32)?
            .div(&denom.astype(DType::Float32)?)?;
        out.insert(
            format!("layers.{layer_idx}.mlp.experts.{proj_name}.scale"),
            outer,
        );
    }

    *params = out;
    Ok(true)
}

#[cfg(test)]
pub(crate) fn test_real_model_path() -> String {
    if let Ok(p) =
        std::env::var("INKLING2_MODEL_DIR").or_else(|_| std::env::var("INKLING2_MODEL_PATH"))
        && !p.trim().is_empty()
    {
        return p;
    }
    let xl = "/Users/brooklyn/.mlx-node/models/inkling-2-mini-gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf";
    if Path::new(xl).exists() {
        return xl.to_string();
    }
    "/Users/brooklyn/.mlx-node/models/inkling-2-mini-gguf".to_string()
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::models::inkling2::vision::expand_image_tokens;
    use crate::tools::inkling::{
        CONTENT_MODEL_END_SAMPLING_ID, Inkling2RenderOptions, InklingContentPart, InklingMessage,
        InklingStreamParser, render_inkling2_messages_to_ids,
    };

    #[tokio::test]
    #[ignore = "Requires real Inkling-2-Mini GGUF + mmproj or SafeTensors on disk (~18-20 GB)"]
    async fn inkling2_real_flat_smoke() {
        let model_path = test_real_model_path();
        let model_dir = Path::new(&model_path);
        assert!(
            model_dir.exists(),
            "Real model path missing: {}",
            model_dir.display()
        );

        let t0 = Instant::now();
        let model = load_from_dir(&model_path)
            .await
            .expect("load Inkling-2-Mini");
        let load_secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "[inkling2_real_flat_smoke] Loaded model in {load_secs:.2}s (phys={}, visits={}, vision={})",
            model.config.num_physical_layers(),
            model.config.num_layer_visits(),
            model.vision_encoder.is_some()
        );
        assert!(
            model.vision_encoder.is_some(),
            "Expected mmproj-BF16.gguf vision encoder to auto-load"
        );
        let tokenizer = model
            .tokenizer
            .as_deref()
            .expect("Expected embedded Inkling2 tokenizer");

        // 1. Text greedy generation test ("What is 2 + 2? Answer briefly.")
        let messages = vec![InklingMessage::user("What is 2 + 2? Answer briefly.")];
        let opts = Inkling2RenderOptions {
            reasoning_effort: 0.1,
            ..Default::default()
        };
        let prompt_ids = render_inkling2_messages_to_ids(&messages, tokenizer, &opts)
            .expect("render Inkling2 prompt");
        eprintln!(
            "[inkling2_real_flat_smoke] Text prompt tokens ({}): {:?}",
            prompt_ids.len(),
            prompt_ids
        );

        let t_gen = Instant::now();
        let generated_ids = model
            .generate_greedy(&prompt_ids, None, 128, &[CONTENT_MODEL_END_SAMPLING_ID])
            .expect("text greedy generation");
        let gen_secs = t_gen.elapsed().as_secs_f64();

        let mut parser = InklingStreamParser::from_prompt_tokens(&prompt_ids);
        for &tok in &generated_ids {
            let piece = tokenizer.decode_sync(&[tok], false).unwrap_or_default();
            let _ = parser.push_token(tok, &piece);
        }
        let _ = parser.flush();

        let raw_decoded = tokenizer
            .decode_sync(&generated_ids, false)
            .unwrap_or_default();
        eprintln!(
            "[inkling2_real_flat_smoke] Generated {} tokens in {gen_secs:.2}s\n  raw: {raw_decoded:?}\n  thinking: {:?}\n  content: {:?}",
            generated_ids.len(),
            parser.thinking(),
            parser.content()
        );
        assert!(!generated_ids.is_empty(), "Model produced zero tokens");
        assert!(
            parser.content().contains('4') || raw_decoded.contains('4'),
            "Expected coherent answer containing '4', got content={:?}, raw={:?}",
            parser.content(),
            raw_decoded
        );

        // 2. Vision encoder + multimodal generation smoke test (solid red 80×80 PNG)
        let red_img = image::RgbImage::from_pixel(80, 80, image::Rgb([220, 20, 20]));
        let mut png_bytes = Vec::new();
        {
            let mut cursor = std::io::Cursor::new(&mut png_bytes);
            image::DynamicImage::ImageRgb8(red_img)
                .write_to(&mut cursor, image::ImageFormat::Png)
                .expect("encode synthetic red PNG");
        }

        let vision = model
            .vision_encoder
            .as_ref()
            .expect("vision encoder loaded");
        let t_vis = Instant::now();
        let (img_emb, grid_rows, grid_cols) = vision
            .encode_image_bytes(&png_bytes)
            .expect("encode synthetic PNG");
        img_emb.eval();
        let vis_secs = t_vis.elapsed().as_secs_f64();
        assert_eq!(grid_rows, 2);
        assert_eq!(grid_cols, 2);
        assert_eq!(img_emb.shape().unwrap().as_ref(), &[4, 2304]);
        assert!(
            !img_emb.has_nan_or_inf().unwrap(),
            "Vision embeddings contain NaN or Inf!"
        );
        eprintln!(
            "[inkling2_real_flat_smoke] Encoded 80x80 PNG -> {:?} in {vis_secs:.2}s (finite)",
            img_emb.shape().unwrap().as_ref()
        );

        let mm_messages = vec![InklingMessage::user_parts(vec![
            InklingContentPart::Image,
            InklingContentPart::Text(
                "What solid color is this image? Answer in one word.".to_string(),
            ),
        ])];
        let base_mm_ids = render_inkling2_messages_to_ids(&mm_messages, tokenizer, &opts)
            .expect("render multimodal prompt");
        let mm_prompt_ids =
            expand_image_tokens(&base_mm_ids, &[grid_rows * grid_cols]).expect("expand image pads");

        let mm_gen_ids = model
            .generate_greedy(
                &mm_prompt_ids,
                Some(&img_emb),
                128,
                &[CONTENT_MODEL_END_SAMPLING_ID],
            )
            .expect("multimodal greedy generation");
        let mut mm_parser = InklingStreamParser::from_prompt_tokens(&mm_prompt_ids);
        for &tok in &mm_gen_ids {
            let piece = tokenizer.decode_sync(&[tok], false).unwrap_or_default();
            let _ = mm_parser.push_token(tok, &piece);
        }
        let _ = mm_parser.flush();
        let mm_raw = tokenizer
            .decode_sync(&mm_gen_ids, false)
            .unwrap_or_default();
        eprintln!(
            "[inkling2_real_flat_smoke] Multimodal generated {} tokens\n  raw: {mm_raw:?}\n  thinking: {:?}\n  content: {:?}",
            mm_gen_ids.len(),
            mm_parser.thinking(),
            mm_parser.content()
        );
        let lower = format!("{} {}", mm_parser.content(), mm_raw).to_lowercase();
        assert!(
            lower.contains("red") || lower.contains("crimson") || lower.contains("scarlet"),
            "Expected multimodal model to identify red square, got content={:?}, raw={:?}",
            mm_parser.content(),
            mm_raw
        );
    }
}
