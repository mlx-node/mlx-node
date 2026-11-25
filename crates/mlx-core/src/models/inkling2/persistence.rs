use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use mlx_paged_attn::CacheRecency;
use napi::bindgen_prelude::{Error, Result};

use crate::array::memory::{WeightsResident, materialize_weights, synchronize_and_clear_cache};
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
    effective_plq_for, ensure_dense_weight_floating, ensure_row_major_kquant_layout,
    is_kquant_mode, kquant_mode_params_for_scales, load_embedding_affine_or_bf16,
    load_linear_proj_quantized_or_bf16, load_quant_settings_from_disk,
};
use crate::models::quantized_linear::{
    DEFAULT_QUANT_MODE, LinearProj, MXFP4_BITS, MXFP4_GROUP_SIZE, MXFP4_MODE, MXFP8_BITS,
    MXFP8_GROUP_SIZE, MXFP8_MODE, NVFP4_BITS, NVFP4_GROUP_SIZE, NVFP4_MODE, QuantizedLinear,
    QuantizedSwitchLinear, normalize_scalar_outer_scale, resolve_nvfp4_outer_scale_2d,
    resolve_nvfp4_outer_scale_3d, resolve_optional_outer_scale_3d, row_major_kquant_arrays,
    try_build_fp8_e4m3_quantized_linear,
};
use crate::moe::{SwitchLinear, SwitchProj};
use crate::nn::{Embedding, LayerNorm, Linear};
use crate::stream::WiredResidency;
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::Activations;
use crate::utils::gguf::NativeGgufFamily;
use crate::utils::gguf_import::PreloadInput;

use super::attention::Inkling2Attention;
use super::attn_res::{LoopController, LoopLayerControllers};
use super::config::{Inkling2Config, Inkling2VisionConfig};
use super::dflash2::{discover_inkling2_dflash2_candidates, is_dflash2_dir};
use super::dspark::{Inkling2DsparkModel, resolve_inkling2_dspark_candidates};
use super::kv_cache::{Inkling2KVCacheCoordinator, Inkling2PagedConfig};
use super::memory::{
    Inkling2ContextLimits, evaluate_inkling2_plan, inkling2_scheduler_lanes, resolve_activations,
    resolve_decode_share, resolve_effective_kv_format, resolve_kv_format, resolve_max_context,
    resolve_max_memory,
};
use super::model::{
    AttnInputNorm, Inkling2Inner, Inkling2LoadOptions, Inkling2Model, Inkling2PhysicalLayer,
    default_decode_share, next_inkling2_model_id,
};
use super::moe::{ExpertGateUp, Inkling2MoE};
use super::paged_forward::Inkling2PagedRuntime;
use super::scheduled_dspark::Inkling2ScheduledDsparkStates;
use super::scheduler::Inkling2SchedulerState;
use super::state_disk::compute_persistent_identity;
use super::vision::{Inkling2VisionBlock, Inkling2VisionEncoder, VisionLinear};
use crate::models::qwen3_5::dflash2::{DFlash2TargetShape, load_dflash2, read_dflash2_config};

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
    let auto_draft_path = options.auto_draft_model_path.clone();
    let auto_load_draft = options.auto_load_draft.unwrap_or(true);
    let disk_options = options.disk_tier("Inkling2")?;
    let requested_kv_format = options.kv_format.clone();
    let _ = resolve_kv_format("Inkling2", requested_kv_format.as_deref())?;
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

            // 1. Load & materialize target backbone weights first so an OOM while
            //    materializing an auto-discovered draft cannot abort startup.
            let mut inner = load_weights_sync(
                &asset_dir,
                preload_input,
                gguf_vision_meta.as_ref(),
                is_gguf,
            )?;
            let metal_paged = crate::engine::persistence::compiled_forward_backend_available();
            let cuda_fast = !metal_paged && inner.use_cuda_fast();
            let (kv_format, cuda_kv_format) = resolve_effective_kv_format(
                "Inkling2",
                requested_kv_format.as_deref(),
                metal_paged,
                cuda_fast,
            )?;
            inner.cuda_kv_format = cuda_kv_format;
            // Drop cloned host-backed `weight_arrays` handles BEFORE promoting weights to CUDA
            // device memory (and before step 2 so `inner.lm_head` is already device-resident for
            // DSpark Markov folding).
            let mut resident = materialize_and_promote_weights(&mut inner)?;
            synchronize_and_clear_cache();

            // 2. Load & materialize optional DFlash2 / DSpark drafter weights.
            let (draft_kind, initial_draft_resident) = load_inkling2_draft_into(
                &mut inner,
                &source_dir,
                &asset_dir,
                is_gguf,
                explicit_draft_path.as_deref(),
                auto_draft_path.as_deref(),
                auto_load_draft,
            )?;
            load_guard.record_read();

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

            // 3. Plan & allocate extent-grown Group 0 + Group 1 KV pools under Governor Ledger.
            // If an auto-discovered draft does not fit the KV/memory budget, step down cleanly
            // (`AutoDflash2 -> AutoDspark -> pure AR`) while keeping explicit drafts strict.
            if metal_paged {
                struct KvPlanFallbackCtx<'a> {
                    inner: &'a mut Inkling2Inner,
                    extra_resident: Option<WeightsResident>,
                }

                let model_id = inner.model_id;
                let mut plan_ctx = KvPlanFallbackCtx {
                    inner: &mut inner,
                    extra_resident: initial_draft_resident,
                };
                let ((pool_plan, guard, mut coordinator), _final_draft_kind) =
                    plan_kv_with_auto_draft_fallback(
                        &mut plan_ctx,
                        draft_kind,
                        |ctx, _kind| {
                            let recency = Arc::clone(&ctx.inner.cache_recency);
                            let cfg_ref = &ctx.inner.config;
                            let dflash2_cfg = ctx.inner.dflash2.as_ref().map(|d| &d.config);
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
                                    dflash2_cfg,
                                )?;
                                let (reg_guard, coord) =
                                    Inkling2KVCacheCoordinator::new_with_memory_plan(
                                        cfg_ref,
                                        &plan,
                                        model_id,
                                        memory_governor(),
                                        Arc::clone(&recency),
                                        activations,
                                    )?;
                                Ok((plan, reg_guard, coord))
                            })
                        },
                        |ctx| {
                            ctx.inner.dflash2 = None;
                            ctx.inner.dflash2_identity = None;
                            synchronize_and_clear_cache();
                        },
                        |ctx| {
                            let (loaded, dspark_resident) =
                                load_auto_dspark_into(ctx.inner, &source_dir, &asset_dir, is_gguf);
                            ctx.extra_resident = dspark_resident;
                            loaded
                        },
                        |ctx| {
                            ctx.inner.dspark = None;
                            ctx.extra_resident = None;
                            synchronize_and_clear_cache();
                        },
                    )?;
                let KvPlanFallbackCtx { extra_resident, .. } = plan_ctx;
                if let Some(extra) = extra_resident {
                    resident = resident.and(extra);
                }

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
                let ram_cap = (pool_plan.lanes.max(1) as usize * 4).clamp(8, 32);
                inner.memory_plan = Some(pool_plan.memory_plan);
                inner.memory_registration = Some(guard);
                inner.wired = Some(WiredResidency::new(WiredResidency::working_set_minus_pools));
                inner.paged = Some(Inkling2PagedRuntime::from_coordinator(
                    coordinator,
                    max_chunk,
                    Arc::clone(&inner.cache_recency),
                ));

                // 3. Configure StateStore geometry + RAM cap, attach StateDisk SSD tier & adopt
                //    persistent entries on restart.
                let dflash2_cfg = inner.dflash2.as_ref().map(|d| &d.config);
                let geom_opt = inner.paged.as_mut().and_then(|rt| {
                    rt.state_store.attach_host_budget();
                    rt.state_store.set_ram_cap(ram_cap);
                    let geom =
                        super::state_disk::Inkling2StateGeometry::from_coordinator_with_draft(
                            &inner.config,
                            &rt.coordinator,
                            dflash2_cfg,
                        )
                        .ok()?;
                    rt.state_store.set_geometry(geom.clone());
                    Some(geom)
                });
                if let Some(geom) = geom_opt {
                    let draft_id_ready =
                        inner.dflash2.is_none() || inner.dflash2_identity.is_some();
                    let persistent_id = if disk_options.reads_identity() && draft_id_ready {
                        inner.paged.as_ref().and_then(|rt_ref| {
                            compute_persistent_identity(
                                &inner.config,
                                &rt_ref.coordinator,
                                &geom,
                                activations,
                                ckpt_files,
                                &resident,
                                &load_guard,
                                inner.dflash2_identity.as_deref(),
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

            let weight_bytes = inner.weight_bytes();
            let weight_guard = cache_limit::coordinator().register(weight_bytes);
            let meta = InitMeta {
                model_id: inner.model_id,
                config: inner.config.clone(),
                vision_config: inner.vision_encoder.as_ref().map(|v| v.config.clone()),
                tokenizer: inner.tokenizer.clone(),
                paged_active: inner.paged.is_some(),
                scheduled_lanes,
                vision_active: inner.vision_encoder.is_some(),
                dspark_active: inner.has_mtp_weights(),
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

/// Normalize a candidate path for deduplication: prefer `fs::canonicalize` when
/// the path resolves on disk, falling back to lexically normalized components.
fn normalize_candidate_key(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.components().collect())
}

/// Build the ordered, deduplicated auto-discovery DFlash2 candidate list:
/// 1. Colocated `<modelDir>/draft/` (highest precedence)
/// 2. Caller-supplied auto-discovered external companion (`auto_draft_path`), if present on disk
/// 3. Sibling `<parent(modelDir)>/inkling-2-mini-dflash2/` (and other shape-matching siblings)
fn build_auto_dflash2_candidates(
    source_dir: &Path,
    asset_dir: &Path,
    target_shape: DFlash2TargetShape,
    auto_draft_path: Option<&str>,
) -> Vec<PathBuf> {
    let discovered = discover_inkling2_dflash2_candidates(source_dir, asset_dir, target_shape);
    let mut colocated_keys = HashSet::new();
    for dir in [source_dir, asset_dir] {
        colocated_keys.insert(normalize_candidate_key(&dir.join("draft")));
        if let Ok(canon_dir) = fs::canonicalize(dir) {
            colocated_keys.insert(normalize_candidate_key(&canon_dir.join("draft")));
        }
    }

    let mut colocated = Vec::new();
    let mut siblings = Vec::new();
    for cand in discovered {
        if colocated_keys.contains(&normalize_candidate_key(&cand)) {
            colocated.push(cand);
        } else {
            siblings.push(cand);
        }
    }

    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut push_unique = |cand: PathBuf| {
        if seen.insert(normalize_candidate_key(&cand)) {
            out.push(cand);
        }
    };

    // 1. Colocated `<modelDir>/draft/`.
    for cand in colocated {
        push_unique(cand);
    }

    // 2. Auto-discovered external companion (`auto_draft_model_path`).
    if let Some(raw) = auto_draft_path {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let cand: PathBuf = Path::new(trimmed).components().collect();
            if cand.exists() || cand.symlink_metadata().is_ok() {
                push_unique(cand);
            }
        }
    }

    // 3. Sibling `<parent(modelDir)>/inkling-2-mini-dflash2/` and other compatible siblings.
    for cand in siblings {
        push_unique(cand);
    }

    out
}

/// Provenance of the speculative draft currently attached to an [`Inkling2Inner`].
///
/// Distinguishes user-requested explicit drafts (which remain strict on KV/memory budget
/// failures) from auto-discovered drafts (which step down `AutoDflash2 -> AutoDspark -> None`
/// when startup KV planning or memory reservation fails).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoadedDraftKind {
    None,
    Explicit,
    AutoDflash2,
    AutoDspark,
}

/// Returns `true` when an error from KV memory planning, governor admission, or Metal KV pool
/// allocation indicates a memory-budget / context-fit / allocation-pressure failure that releasing
/// an auto-discovered speculative draft can resolve.
///
/// Structural/configuration errors (e.g. zero global attention visits, unreadable Metal working
/// set, invalid block geometry, or out-of-scope governor logic bugs) return `false` so they
/// propagate immediately without dropping drafts or retrying.
fn is_memory_or_fit_error(err: &Error) -> bool {
    let msg = err.reason.as_str();
    if msg.contains("allocated outside a claim's scope") {
        return false;
    }
    msg.contains("one lane's state and the KV runway need")
        || msg.contains("floors of the memory plan require")
        || (msg.contains("--max-context") && msg.contains("exceeds the"))
        || (msg.contains("paged_cache_memory_mb") && msg.contains("exceeds the"))
        || msg.contains("exceed memory governor limit")
        || msg.contains("macOS is short of memory")
        || msg.contains("the KV pool's runway")
        || msg.contains("engine memory budget exceeded")
        || msg.contains("host memory reserve protected")
        || msg.contains("Metal driver rejected allocation")
        || msg.contains("Metal returned a nil")
        || crate::engine::governor::is_gpu_out_of_memory(msg)
        || msg.to_ascii_lowercase().contains("out of memory")
}

/// Execute startup KV planning & pool allocation with bounded automatic draft step-down on
/// memory/fit failures:
/// - `AutoDflash2` -> release DFlash2 -> try bundled `AutoDspark` (or `None` if absent) -> retry.
/// - `AutoDspark` -> release DSpark -> retry as `None` (pure autoregressive).
/// - `Explicit` / `None` -> single attempt, propagate errors immediately.
/// - Non-memory/structural errors -> propagate immediately without releasing any draft.
fn plan_kv_with_auto_draft_fallback<C, T>(
    ctx: &mut C,
    mut draft_kind: LoadedDraftKind,
    mut plan_fn: impl FnMut(&C, LoadedDraftKind) -> Result<T>,
    mut drop_dflash2_fn: impl FnMut(&mut C),
    mut try_load_dspark_fn: impl FnMut(&mut C) -> LoadedDraftKind,
    mut drop_dspark_fn: impl FnMut(&mut C),
) -> Result<(T, LoadedDraftKind)> {
    loop {
        match plan_fn(ctx, draft_kind) {
            Ok(planned) => return Ok((planned, draft_kind)),
            Err(err) => match draft_kind {
                LoadedDraftKind::AutoDflash2 if is_memory_or_fit_error(&err) => {
                    tracing::warn!(
                        "Auto-discovered Inkling2 DFlash2 draft does not fit memory/KV budget ({err}); releasing DFlash2 and retrying with bundled DSpark / AR fallback"
                    );
                    drop_dflash2_fn(ctx);
                    draft_kind = match try_load_dspark_fn(ctx) {
                        LoadedDraftKind::AutoDspark => LoadedDraftKind::AutoDspark,
                        _ => LoadedDraftKind::None,
                    };
                }
                LoadedDraftKind::AutoDspark if is_memory_or_fit_error(&err) => {
                    tracing::warn!(
                        "Auto-discovered Inkling2 DSpark draft does not fit memory/KV budget ({err}); releasing DSpark and retrying as pure autoregressive"
                    );
                    drop_dspark_fn(ctx);
                    draft_kind = LoadedDraftKind::None;
                }
                _ => return Err(err),
            },
        }
    }
}

/// Auto-discover, load, and materialize a bundled/colocated Inkling2 DSpark draft
/// (`mtp.safetensors` or `dspark-*.gguf`) into `inner`, logging warnings and soft-falling back
/// to `(LoadedDraftKind::None, None)` on any discovery, load, or materialization error.
fn load_auto_dspark_into(
    inner: &mut Inkling2Inner,
    source_dir: &Path,
    asset_dir: &Path,
    is_gguf: bool,
) -> (LoadedDraftKind, Option<WeightsResident>) {
    load_auto_dspark_into_with(inner, source_dir, asset_dir, is_gguf, materialize_weights)
}

fn load_auto_dspark_into_with(
    inner: &mut Inkling2Inner,
    source_dir: &Path,
    asset_dir: &Path,
    is_gguf: bool,
    mut materialize_dspark: impl FnMut(&[&MxArray]) -> Result<WeightsResident>,
) -> (LoadedDraftKind, Option<WeightsResident>) {
    let dspark_candidates =
        match resolve_inkling2_dspark_candidates(source_dir, asset_dir, is_gguf, None) {
            Ok(candidates) => candidates,
            Err(err) => {
                tracing::warn!("Skipping auto-discovered Inkling2 DSpark candidates: {err}");
                return (LoadedDraftKind::None, None);
            }
        };
    for draft_path in &dspark_candidates {
        let t_draft = std::time::Instant::now();
        match Inkling2DsparkModel::load_auto(draft_path, &inner.config, &inner.lm_head) {
            Ok(mut draft) => {
                let mat_res = {
                    let dspark_arrays = draft.weight_arrays();
                    let dspark_refs: Vec<&MxArray> = dspark_arrays.iter().collect();
                    materialize_dspark(&dspark_refs)
                };
                match mat_res.and_then(|res| {
                    draft.ensure_cuda_device()?;
                    Ok(res)
                }) {
                    Ok(dspark_resident) => {
                        synchronize_and_clear_cache();
                        tracing::info!(
                            "Inkling2 DSpark auto-loaded from {} in {:.2}s (layers={}, block_size={}, markov_rank={}, taps={:?}, weights={:.1} MiB)",
                            draft_path.display(),
                            t_draft.elapsed().as_secs_f64(),
                            draft.config.num_hidden_layers,
                            draft.config.block_size,
                            draft.config.markov_rank,
                            draft.config.target_layer_ids,
                            draft.weight_bytes() as f64 / (1024.0 * 1024.0),
                        );
                        inner.dspark = Some(draft);
                        return (LoadedDraftKind::AutoDspark, Some(dspark_resident));
                    }
                    Err(err) => {
                        tracing::warn!(
                            "Failed to materialize auto-discovered Inkling2 DSpark draft '{}' ({err}); releasing DSpark and falling back",
                            draft_path.display()
                        );
                        drop(draft);
                        inner.dspark = None;
                        synchronize_and_clear_cache();
                    }
                }
            }
            Err(err) => {
                tracing::warn!(
                    "Skipping auto-discovered Inkling2 DSpark draft '{}' (trying next candidate if any): {err}",
                    draft_path.display()
                );
            }
        }
    }
    (LoadedDraftKind::None, None)
}

/// Load and materialize an Inkling-2-Mini speculative drafter into `inner` following exact
/// precedence:
/// - Explicit `"none"` / `"off"` / `""` -> disables both DFlash2 and DSpark (ignores `auto_draft_path`).
/// - Explicit path `P` (ignores `auto_draft_path`):
///   - If `is_dflash2_dir(P)` -> load & materialize DFlash2 (hard error on failure).
///   - Else -> resolve, load & materialize DSpark (hard error on failure).
/// - Auto-discovery (`explicit == None && auto_load_draft`):
///   1. Colocated `<modelDir>/draft/` DFlash2 companion
///   2. External auto-discovered companion (`auto_draft_path`), if present
///   3. Sibling `<parent(modelDir)>/inkling-2-mini-dflash2/` (or shape-matching sibling DFlash2 dir)
///   4. Bundled/colocated DSpark (`mtp.safetensors` / `dspark-*.gguf`)
///   5. Pure AR fallback (soft fallback on incompatible/corrupt/OOM auto-discovered drafts).
///
/// Returns `(LoadedDraftKind, Option<WeightsResident>)`, where the `WeightsResident` carries
/// DSpark's materialization witness when DSpark is attached so callers can fold it into the
/// backbone witness or drop it if DSpark is later released during KV-fit fallback. (`load_dflash2`
/// already materializes DFlash2 weights internally.)
fn load_inkling2_draft_into(
    inner: &mut Inkling2Inner,
    source_dir: &Path,
    asset_dir: &Path,
    is_gguf: bool,
    explicit: Option<&str>,
    auto_draft_path: Option<&str>,
    auto_load_draft: bool,
) -> Result<(LoadedDraftKind, Option<WeightsResident>)> {
    load_inkling2_draft_into_with(
        inner,
        source_dir,
        asset_dir,
        is_gguf,
        explicit,
        auto_draft_path,
        auto_load_draft,
        load_dflash2,
        materialize_weights,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_inkling2_draft_into_with(
    inner: &mut Inkling2Inner,
    source_dir: &Path,
    asset_dir: &Path,
    is_gguf: bool,
    explicit: Option<&str>,
    auto_draft_path: Option<&str>,
    auto_load_draft: bool,
    mut load_dflash2_fn: impl FnMut(
        &Path,
    ) -> Result<(crate::models::qwen3_5::dflash2::DFlash2Model, u64)>,
    mut materialize_dspark: impl FnMut(&[&MxArray]) -> Result<WeightsResident>,
) -> Result<(LoadedDraftKind, Option<WeightsResident>)> {
    let target_shape = inner.draft_target_shape();

    if let Some(raw) = explicit {
        let trimmed = raw.trim();
        if trimmed.is_empty()
            || trimmed.eq_ignore_ascii_case("none")
            || trimmed.eq_ignore_ascii_case("off")
        {
            return Ok((LoadedDraftKind::None, None));
        }
        let explicit_path = Path::new(trimmed);
        if is_dflash2_dir(explicit_path) {
            let t_draft = std::time::Instant::now();
            read_dflash2_config(explicit_path)?.validate_target(target_shape)?;
            let (mut draft, bytes) = match load_dflash2_fn(explicit_path) {
                Ok(loaded) => loaded,
                Err(err) => {
                    synchronize_and_clear_cache();
                    return Err(err);
                }
            };
            draft.validate_target(target_shape)?;
            draft.configure_inkling2_execution(target_shape);
            if crate::accelerator::cuda_ops_available() {
                draft.configure_dense_execution();
            }
            synchronize_and_clear_cache();
            let verified_identity = match draft.context_fusion_fingerprint() {
                Ok(fp) => Some(fp),
                Err(err) => {
                    tracing::warn!(
                        "Inkling2 DFlash2 failed to fingerprint loaded fusion weights from '{}': {err}; disabling persistent StateDisk identity",
                        explicit_path.display()
                    );
                    None
                }
            };
            tracing::info!(
                "Inkling2 DFlash2 loaded from {} in {:.2}s (block_size={}, taps={:?}, softcap={:?}, weights={:.1} MiB)",
                explicit_path.display(),
                t_draft.elapsed().as_secs_f64(),
                draft.config.block_size,
                draft.config.target_layers,
                draft.config.final_logit_softcapping,
                bytes as f64 / (1024.0 * 1024.0),
            );
            inner.dflash2 = Some(draft);
            inner.dflash2_identity = verified_identity;
            return Ok((LoadedDraftKind::Explicit, None));
        }

        let candidates =
            resolve_inkling2_dspark_candidates(source_dir, asset_dir, is_gguf, Some(trimmed))?;
        let mut last_err: Option<Error> = None;
        for draft_path in &candidates {
            let t_draft = std::time::Instant::now();
            match Inkling2DsparkModel::load_auto(draft_path, &inner.config, &inner.lm_head) {
                Ok(mut draft) => {
                    let mat_res = {
                        let dspark_arrays = draft.weight_arrays();
                        let dspark_refs: Vec<&MxArray> = dspark_arrays.iter().collect();
                        materialize_dspark(&dspark_refs)
                    };
                    let dspark_resident = match mat_res.and_then(|res| {
                        draft.ensure_cuda_device()?;
                        Ok(res)
                    }) {
                        Ok(res) => res,
                        Err(err) => {
                            drop(draft);
                            synchronize_and_clear_cache();
                            return Err(err);
                        }
                    };
                    synchronize_and_clear_cache();
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
                    return Ok((LoadedDraftKind::Explicit, Some(dspark_resident)));
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
        if let Some(err) = last_err {
            return Err(err);
        }
        return Ok((LoadedDraftKind::None, None));
    }

    if !auto_load_draft {
        return Ok((LoadedDraftKind::None, None));
    }

    // 1, 2 & 3. Auto-discover DFlash2 (`<modelDir>/draft/` -> `auto_draft_path` ->
    // `<parent>/inkling-2-mini-dflash2/` and compatible siblings) ahead of bundled DSpark,
    // with soft fallback on any config/shape/weight/materialization error.
    for dflash_path in
        build_auto_dflash2_candidates(source_dir, asset_dir, target_shape, auto_draft_path)
    {
        let t_draft = std::time::Instant::now();
        match read_dflash2_config(&dflash_path)
            .and_then(|cfg| cfg.validate_target(target_shape))
            .and_then(|()| load_dflash2_fn(&dflash_path))
            .and_then(|(mut draft, bytes)| {
                draft.validate_target(target_shape)?;
                draft.configure_inkling2_execution(target_shape);
                if crate::accelerator::cuda_ops_available() {
                    draft.configure_dense_execution();
                }
                Ok((draft, bytes))
            }) {
            Ok((draft, bytes)) => {
                synchronize_and_clear_cache();
                let verified_identity = match draft.context_fusion_fingerprint() {
                    Ok(fp) => Some(fp),
                    Err(err) => {
                        tracing::warn!(
                            "Inkling2 DFlash2 failed to fingerprint auto-loaded fusion weights from '{}': {err}; disabling persistent StateDisk identity",
                            dflash_path.display()
                        );
                        None
                    }
                };
                tracing::info!(
                    "Inkling2 DFlash2 auto-loaded from {} in {:.2}s (block_size={}, taps={:?}, softcap={:?}, weights={:.1} MiB)",
                    dflash_path.display(),
                    t_draft.elapsed().as_secs_f64(),
                    draft.config.block_size,
                    draft.config.target_layers,
                    draft.config.final_logit_softcapping,
                    bytes as f64 / (1024.0 * 1024.0),
                );
                inner.dflash2 = Some(draft);
                inner.dflash2_identity = verified_identity;
                return Ok((LoadedDraftKind::AutoDflash2, None));
            }
            Err(err) => {
                synchronize_and_clear_cache();
                tracing::warn!(
                    "Skipping auto-discovered Inkling2 DFlash2 draft '{}' (trying next candidate): {err}",
                    dflash_path.display()
                );
            }
        }
    }

    // 4. Fall back to bundled/colocated DSpark (`mtp.safetensors` / `dspark-*.gguf`).
    Ok(load_auto_dspark_into_with(
        inner,
        source_dir,
        asset_dir,
        is_gguf,
        materialize_dspark,
    ))
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

    // Materialize target backbone weights cleanly without Metal watchdog timeout, dropping
    // cloned host-backed handles before promoting weights to CUDA device memory, before
    // attaching any auto-discovered draft.
    let _resident = materialize_and_promote_weights(&mut model)?;
    synchronize_and_clear_cache();

    // Auto-discover DFlash2 (`draft/` -> sibling `inkling-2-mini-dflash2/`) before bundled DSpark
    // (`mtp.safetensors` / `dspark-*.gguf`), falling back cleanly to pure AR if none load.
    let (_draft_kind, _draft_resident) =
        load_inkling2_draft_into(&mut model, source_dir, asset_dir, is_gguf, None, None, true)?;

    // Initialize block-paged hybrid KV runtime on Metal (default INT8 32-token KTM).
    if crate::engine::persistence::compiled_forward_backend_available() {
        let effective_cfg =
            paged_cfg.unwrap_or_else(|| Inkling2PagedConfig::default_for_config(&model.config));
        let mut rt = Inkling2PagedRuntime::new_with_recency(
            &model.config,
            effective_cfg,
            Arc::clone(&model.cache_recency),
        )?;
        let geom = super::state_disk::Inkling2StateGeometry::from_coordinator_with_draft(
            &model.config,
            &rt.coordinator,
            model.dflash2.as_ref().map(|d| &d.config),
        )?;
        rt.state_store.set_geometry(geom);
        model.paged = Some(rt);
    }
    if !model.use_cuda_fast() {
        model.cuda_kv_format = super::cuda_kv::Inkling2CudaKvFormat::Bf16;
    }

    Ok(model)
}

/// Materialize all resident weights and immediately drop the temporary cloned
/// `MxArray` handles before calling `ensure_cuda_device_weights()` so old
/// host-backed buffers can be freed incrementally during promotion instead of
/// holding both host and device copies simultaneously (~2x peak load RAM).
fn materialize_and_promote_weights(inner: &mut Inkling2Inner) -> Result<WeightsResident> {
    let resident = {
        let weight_arrays = inner.weight_arrays();
        let weight_refs: Vec<&MxArray> = weight_arrays.iter().collect();
        materialize_weights(&weight_refs)?
    };
    inner.ensure_cuda_device_weights()?;
    Ok(resident)
}

/// Opt-in Splash-style CUDA load-time quantization of dense BF16 attention projections
/// (`q/k/v/rel/gate/o_proj`) and `lm_head` to exact symmetric `Q8_0` (`SymQ8`, Float16
/// scales, zero biases).
///
/// Active ONLY when CUDA is available AND `MLX_INKLING2_BF16_PROJ_Q8` is set to
/// `1|true|on|yes` (default OFF -> 100% lossless BF16). Never touches macOS Metal.
#[inline]
fn bf16_proj_q8_cuda_enabled() -> bool {
    if !crate::accelerator::cuda_ops_available() {
        return false;
    }
    std::env::var("MLX_INKLING2_BF16_PROJ_Q8").is_ok_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

/// Quantize a bias-free dense BF16 [`Linear`] `[N, K]` (`N >= 256`, `K >= 256`, `K % 32 == 0`)
/// into exact symmetric `Q8_0` (`QuantizedLinear` with `bits=8, group_size=32, mode="affine"`,
/// `Float16` scales, `biases=None`, `is_symmetric_q8() == true`), evaluating in row chunks of at
/// most `chunk_rows` to cap transient VRAM on large matrices such as `lm_head` (`[201024, 2304]`).
///
/// Byte encoding matches CUDA `SymQ8` (`affine_q8_few.cu.inc`, `inkling2_lm_head.cu.inc`):
/// `byte = (clamp(round(w / scale), -127, 127) + 128) as u8` packed little-endian into `Uint32`
/// (`w_i8 = (int8_t)(byte ^ 0x80)`, `val = scale * w_i8`). Returns `Ok(None)` when ineligible
/// (tiny projections like `sink_proj`/`k_conv_proj`/`v_conv_proj`, biased, non-BF16, or scales
/// outside exact F16 symmetric range).
fn quantize_bf16_linear_to_symq8_with_chunk(
    lin: &Linear,
    chunk_rows: i64,
) -> Result<Option<QuantizedLinear>> {
    if lin.get_bias().is_some() || chunk_rows <= 0 {
        return Ok(None);
    }
    let w = lin.get_weight();
    if w.dtype()? != DType::BFloat16 {
        return Ok(None);
    }
    let shape = w.shape()?;
    if shape.len() != 2 {
        return Ok(None);
    }
    let n = shape[0];
    let k = shape[1];
    if n < 256 || k < 256 || k % 32 != 0 {
        return Ok(None);
    }

    const GROUP_SIZE: i64 = 32;
    let groups = k / GROUP_SIZE;
    let zero_f32 = MxArray::scalar_float(0.0)?;
    let one_f32 = MxArray::scalar_float(1.0)?;

    let quantize_chunk = |w_chunk: &MxArray, rows: i64| -> Result<(MxArray, MxArray)> {
        let w_g = w_chunk
            .astype(DType::Float32)?
            .reshape(&[rows, groups, GROUP_SIZE])?;
        let amax = w_g.abs()?.max(Some(&[-1]), Some(true))?;
        let scale_f16_3d = amax.div_scalar(127.0)?.astype(DType::Float16)?;
        let scale_f32 = scale_f16_3d.astype(DType::Float32)?;
        let has_scale = scale_f32.greater(&zero_f32)?;
        let safe_scale = has_scale.where_(&scale_f32, &one_f32)?;
        let q_scaled = has_scale.where_(&w_g.div(&safe_scale)?, &zero_f32)?;
        let q_u8 = q_scaled
            .round()?
            .clip(Some(-127.0), Some(127.0))?
            .add_scalar(128.0)?
            .astype(DType::Uint8)?
            .reshape(&[rows, k])?;
        let w_u32 = q_u8.view(DType::Uint32)?;
        let scales_2d = scale_f16_3d.reshape(&[rows, groups])?;
        Ok((w_u32, scales_2d))
    };

    let (weight, scales) = if n <= chunk_rows {
        let (w_u32, s_f16) = quantize_chunk(&w, n)?;
        MxArray::eval_arrays(&[&w_u32, &s_f16])?;
        (w_u32, s_f16)
    } else {
        let n_chunks = (n as usize).div_ceil(chunk_rows as usize);
        let mut w_chunks = Vec::with_capacity(n_chunks);
        let mut s_chunks = Vec::with_capacity(n_chunks);
        let mut start = 0i64;
        while start < n {
            let end = (start + chunk_rows).min(n);
            let sub = w.slice_axis(0, start, end)?;
            let (w_c, s_c) = quantize_chunk(&sub, end - start)?;
            MxArray::eval_arrays(&[&w_c, &s_c])?;
            w_chunks.push(w_c);
            s_chunks.push(s_c);
            start = end;
        }
        let w_refs: Vec<&MxArray> = w_chunks.iter().collect();
        let s_refs: Vec<&MxArray> = s_chunks.iter().collect();
        let w_cat = MxArray::concatenate_many(w_refs, Some(0))?;
        let s_cat = MxArray::concatenate_many(s_refs, Some(0))?;
        MxArray::eval_arrays(&[&w_cat, &s_cat])?;
        (w_cat, s_cat)
    };

    let biases = scales
        .astype(DType::Float32)?
        .mul_scalar(-128.0)?
        .astype(DType::Float16)?;
    let mut ql = QuantizedLinear::new(
        weight,
        scales,
        Some(biases),
        None,
        32,
        8,
        DEFAULT_QUANT_MODE.to_string(),
    );
    if !ql.try_compact_symmetric_q8_f16()? {
        return Ok(None);
    }
    Ok(Some(ql))
}

/// Quantize a bias-free dense BF16 [`Linear`] `[N, K]` (`N >= 256`, `K >= 256`, `K % 32 == 0`)
/// into exact symmetric `Q8_0` (`SymQ8`) using a 16384-row chunk cap.
fn quantize_bf16_linear_to_symq8(lin: &Linear) -> Result<Option<QuantizedLinear>> {
    quantize_bf16_linear_to_symq8_with_chunk(lin, 16384)
}

/// On CUDA with `MLX_INKLING2_BF16_PROJ_Q8=1`, quantize eligible dense BF16 `LinearProj`
/// instances (`N >= 256`, `K >= 256`, `K % 32 == 0`, bias-free) to exact `SymQ8` at load time.
/// No-op off CUDA, on macOS Metal, when unset (default), on already-quantized weights, or on
/// tiny projections (`sink_proj`, `k_conv_proj`, `v_conv_proj`).
fn maybe_quantize_bf16_proj_to_symq8_cuda(proj: &mut LinearProj) -> Result<()> {
    if !bf16_proj_q8_cuda_enabled() {
        return Ok(());
    }
    let LinearProj::Standard(lin) = proj else {
        return Ok(());
    };
    if let Some(ql) = quantize_bf16_linear_to_symq8(lin)? {
        *proj = LinearProj::Quantized(ql);
    }
    Ok(())
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

    // Detect SafeTensors converter layout on RAW keys BEFORE prefix stripping.
    let layout = if is_gguf {
        Inkling2CheckpointLayout::Canonical
    } else {
        detect_inkling2_safetensors_layout(&raw_params)
    };

    // Strip leading wrapper prefixes (`model.language_model.model.`, `model.language_model.`,
    // `language_model.model.`, `language_model.`, `model.`) while keeping vision keys cleanly
    // addressable for `normalize_inkling2_vision_params`.
    let mut params = strip_inkling2_wrapper_prefixes(raw_params);

    // Normalize SafeTensors text layouts (`Release`, `oMLX`, `mlx-vlm`, `mlx-lm`) into the
    // canonical Inkling2 namespace and set authoritative `gate_up_interleaved` from layout.
    let st_norm = if !is_gguf {
        let norm = normalize_inkling2_safetensors_params(&mut params, layout, &config)?;
        if let Some(interleaved) = norm.gate_up_interleaved {
            config.gate_up_interleaved = interleaved;
        }
        // Normalize vision tower keys independently of text layout; on quantized or
        // unsupported vision weights, log a warning and fall back to text-only mode.
        if let Err(err) = normalize_inkling2_vision_params(&mut params, &config) {
            tracing::warn!(
                "Inkling2: vision tower unavailable ({err}); continuing in text-only mode"
            );
            params.retain(|k, _| {
                !k.starts_with("visual.")
                    && !k.starts_with("vision_tower.")
                    && !k.starts_with("vision_projector.")
            });
            config.vision_config = None;
        }
        norm
    } else {
        Inkling2StNormResult::default()
    };

    let (default_bits, default_group_size, default_mode, mut per_layer_quant) =
        load_quant_settings_from_disk(asset_dir, 8, 32)?;
    if !is_gguf {
        remap_inkling2_per_layer_quant(&mut per_layer_quant, layout, &st_norm, &config)?;
        infer_unregistered_2d_fp_quant_overrides(&params, &mut per_layer_quant)?;
    }
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

    // 1. Embeddings & embed_norm (keep packed embedding sidecars intact; supports Affine,
    // MXFP4/MXFP8/NVFP4, K-quants such as Q6_K, and dense floating weights).
    let mut embed_tokens = Embedding::new(padded_vocab as u32, hidden_size as u32)?;
    load_inkling2_embedding(&mut embed_tokens, &params, &per_layer_quant, default_plq)?;
    let embed_norm =
        load_dense_or_dequant_f32(&params, "embed_norm", &per_layer_quant, default_plq)?;

    // 2. Output head & output AttnRes query (`[1, D]` row-major & `[D, 1]` transposed FP32)
    let output_res_query =
        load_dense_or_dequant_f32(&params, "output_res_query", &per_layer_quant, default_plq)?
            .reshape(&[1, hidden_size as i64])?
            .mul_scalar(inv_sqrt_d)?;
    let output_res_query_t = output_res_query.transpose(None)?;
    let output_norm = load_dense_or_dequant_f32(&params, "norm", &per_layer_quant, default_plq)?;
    let mut lm_head = LinearProj::Standard(Linear::new(
        hidden_size as u32,
        padded_vocab as u32,
        Some(false),
    )?);
    load_inkling2_linear_proj(
        &mut lm_head,
        &params,
        "lm_head",
        &per_layer_quant,
        default_plq,
    )?;
    // Compact exact symmetric GGUF Q8_0 (`biases == -128 * scales`) to native Float16 scales
    // with zero biases (4x less sidecar DRAM), falling back to F32 hoisting for asymmetric affine.
    lm_head.try_compact_symmetric_q8_or_promote_f32(DType::BFloat16)?;
    maybe_quantize_bf16_proj_to_symq8_cuda(&mut lm_head)?;
    let lm_head_bf16_unpadded = if let LinearProj::Standard(ref lin) = lm_head
        && lin.get_bias().is_none()
    {
        let w = lin.get_weight();
        let unpadded = config.effective_unpadded_vocab_size();
        let vocab_cols = w.shape_at(0)? as usize;
        if w.dtype()? == DType::BFloat16 && unpadded < vocab_cols {
            Some(w.slice_axis(0, 0, unpadded as i64)?)
        } else {
            None
        }
    } else {
        None
    };

    // Global cross-pass message queries (Unsloth GGUF & Release SafeTensors format) —
    // dequantize ONCE before the physical layer loop.
    let global_loop_res =
        try_load_dense_or_dequant_f32(&params, "loop_res_query", &per_layer_quant, default_plq)?;
    let global_loop_kv_res =
        try_load_dense_or_dequant_f32(&params, "loop_kv_res_query", &per_layer_quant, default_plq)?;

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
            global_loop_res.as_ref(),
            global_loop_kv_res.as_ref(),
        )?;
        layers.push(layer);
    }

    // 4. Optional Vision Encoder (`mmproj-BF16/F32/Q8_0.gguf` or SafeTensors)
    let vision_encoder = if params.contains_key("vision_tower.patch_embed.weight") {
        let vcfg = config
            .vision_config
            .clone()
            .unwrap_or_else(|| Inkling2VisionConfig {
                projection_dim: hidden_size,
                ..Inkling2VisionConfig::default()
            });
        match build_vision_encoder(&vcfg, &params, hidden_size) {
            Ok(encoder) => Some(encoder),
            Err(err) => {
                tracing::warn!(
                    "Inkling2: vision tower unavailable ({err}); continuing in text-only mode"
                );
                config.vision_config = None;
                None
            }
        }
    } else {
        config.vision_config = None;
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

    let cuda_kv_format = if super::cuda_forward::cuda_fast_supported_for(&config) {
        super::cuda_kv::Inkling2CudaKvFormat::resolve(None)
    } else {
        super::cuda_kv::Inkling2CudaKvFormat::Bf16
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
        lm_head_bf16_unpadded,
        vision_encoder,
        tokenizer,
        dspark: None,
        dflash2: None,
        dflash2_identity: None,
        dspark_turn_contexts: HashMap::new(),
        cuda_dflash2_ctx: None,
        cuda_verify_budget: crate::models::qwen3_5::verify_tree::VerifyBudget::new(),
        cuda_lookup_acceptance: crate::models::qwen3_5::verify_tree::LookupAcceptance::new(),
        cuda_dspark_ctx: None,
        scheduled_dspark: Inkling2ScheduledDsparkStates::default(),
        paged: None,
        flat_cache: None,
        cuda_cache: None,
        cuda_kv_format,
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
    let attn_res_query = if let Some(q_raw) = try_load_dense_or_dequant_f32(
        params,
        &format!("{prefix}.attn_res_query"),
        per_layer_quant,
        default_plq,
    )? {
        let expected_q = if is_local { 1 } else { 3 };
        Some(
            q_raw
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

    let ffn_res_query = load_dense_or_dequant_f32(
        params,
        &format!("{prefix}.ffn_res_query"),
        per_layer_quant,
        default_plq,
    )?
    .reshape(&[1, d as i64])?
    .mul_scalar(inv_sqrt_d)?;
    let ffn_res_query_t = ffn_res_query.transpose(None)?;

    // Sublayer norms (kept in FP32 so hot-path RMSNorm never re-casts)
    let attn_inp_norm = if is_local {
        AttnInputNorm::Local(load_dense_or_dequant_f32(
            params,
            &format!("{prefix}.input_layernorm"),
            per_layer_quant,
            default_plq,
        )?)
    } else {
        let q_norm = load_dense_or_dequant_f32(
            params,
            &format!("{prefix}.self_attn.q_inp_norm"),
            per_layer_quant,
            default_plq,
        )?;
        let k_norm = load_dense_or_dequant_f32(
            params,
            &format!("{prefix}.self_attn.k_inp_norm"),
            per_layer_quant,
            default_plq,
        )?;
        let v_norm = load_dense_or_dequant_f32(
            params,
            &format!("{prefix}.self_attn.v_inp_norm"),
            per_layer_quant,
            default_plq,
        )?;
        let qkv_norm = MxArray::stack(vec![&q_norm, &k_norm, &v_norm], Some(0))?;
        AttnInputNorm::Global {
            q_norm,
            k_norm,
            v_norm,
            qkv_norm,
        }
    };
    let ffn_norm = load_dense_or_dequant_f32(
        params,
        &format!("{prefix}.post_attention_layernorm"),
        per_layer_quant,
        default_plq,
    )?;

    // Attention projections (compact exact symmetric Q8_0 to native F16 scales BEFORE row-concat fusion!)
    let load_proj = |sub: &str, in_f: usize, out_f: usize| -> Result<LinearProj> {
        let base = format!("{prefix}.self_attn.{sub}");
        let mut proj = LinearProj::Standard(Linear::new(in_f as u32, out_f as u32, Some(false))?);
        load_inkling2_linear_proj(&mut proj, params, &base, per_layer_quant, default_plq)?;
        proj.try_compact_symmetric_q8_or_promote_f32(DType::BFloat16)?;
        maybe_quantize_bf16_proj_to_symq8_cuda(&mut proj)?;
        Ok(proj)
    };

    let q_proj = load_proj("q_proj", d, q_dim)?;
    let k_proj = load_proj("k_proj", d, kv_dim)?;
    let v_proj = load_proj("v_proj", d, kv_dim)?;
    let rel_proj = load_proj("rel_proj", d, r_dim)?;
    let gate_proj = load_proj("gate_proj", d, q_dim)?;
    let sink_proj = load_proj("sink_proj", d, n_heads)?;
    let o_proj = load_proj("o_proj", q_dim, d)?;

    // ShortConv weights: dequantize if quantized and normalize F16 -> F32 so Metal
    // `mlx_inkling2_sconv_k4*` (`is_bf16_or_f32`) never crashes on F16 checkpoints.
    let k_conv = load_dense_or_dequant_f32(
        params,
        &format!("{prefix}.self_attn.k_conv"),
        per_layer_quant,
        default_plq,
    )?;
    let v_conv = load_dense_or_dequant_f32(
        params,
        &format!("{prefix}.self_attn.v_conv"),
        per_layer_quant,
        default_plq,
    )?;
    let post_conv = load_dense_or_dequant_f32(
        params,
        &format!("{prefix}.self_attn.post_conv"),
        per_layer_quant,
        default_plq,
    )?;

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
    let q_clip = load_dense_or_dequant_f32(
        params,
        &format!("{prefix}.self_attn.q_clip"),
        per_layer_quant,
        default_plq,
    )?
    .reshape(&[1, n_heads as i64, head_dim as i64])?;
    let rel_clip = load_dense_or_dequant_f32(
        params,
        &format!("{prefix}.self_attn.rel_clip"),
        per_layer_quant,
        default_plq,
    )?
    .reshape(&[1, n_heads as i64, d_rel as i64])?;
    let rel_bias_raw = load_dense_or_dequant_f32(
        params,
        &format!("{prefix}.self_attn.rel_bias"),
        per_layer_quant,
        default_plq,
    )?;
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

    // MoE block (supports fused `gate_up_proj` and split `gate_proj` + `up_proj`,
    // fusing compatible split pairs BEFORE symmetric-Q8 compaction / BF16 promotion!)
    let router_weight = load_dense_or_dequant_f32(
        params,
        &format!("{prefix}.mlp.gate"),
        per_layer_quant,
        default_plq,
    )?;
    let router_bias = match params.get(&format!("{prefix}.mlp.gate.e_score_correction_bias")) {
        Some(b) => {
            ensure_dense_weight_floating(&format!("{prefix}.mlp.gate.e_score_correction_bias"), b)?;
            Some(if b.dtype()? == DType::Float32 {
                b.clone()
            } else {
                b.astype(DType::Float32)?
            })
        }
        None => None,
    };
    let (mut gate_up_proj, effective_gate_up_interleaved) = load_expert_gate_up(
        params,
        &prefix,
        per_layer_quant,
        default_plq,
        config.gate_up_interleaved,
    )?;
    let mut down_proj = load_expert_switch_proj(
        params,
        &format!("{prefix}.mlp.experts.down_proj"),
        per_layer_quant,
        default_plq,
    )?;
    down_proj.try_compact_symmetric_q8_or_promote_bf16()?;
    gate_up_proj.promote_outer_scale_to_f32()?;
    down_proj.promote_outer_scale_to_f32()?;
    let mlp_post_conv = load_dense_or_dequant_f32(
        params,
        &format!("{prefix}.mlp.post_conv"),
        per_layer_quant,
        default_plq,
    )?;
    let moe = Inkling2MoE::new(
        config.num_experts,
        config.num_experts_per_tok,
        config.effective_moe_intermediate_size(),
        config.route_scale,
        config.mlp_gate_softcap,
        config.mlp_up_softcap,
        effective_gate_up_interleaved,
        router_weight,
        router_bias,
        gate_up_proj,
        down_proj,
        mlp_post_conv,
    )?;

    // Optional Loop Controllers for core layers (`6..17`)
    let loop_ctrl = if let Some(core_idx) = config.core_offset(phys) {
        let msg_queries_raw = if let Some(per_layer) = try_load_dense_or_dequant_f32(
            params,
            &format!("{prefix}.loop_msg_query"),
            per_layer_quant,
            default_plq,
        )? {
            // Mainline `inkling2_loop`, oMLX, mlx-vlm, mlx-lm: already `[4, D]` in `[q, k, v, mlp]` order.
            per_layer.reshape(&[4, d as i64])?
        } else if let (Some(lq), Some(lkv)) = (global_loop_res, global_loop_kv_res) {
            // Unsloth `inkling2` / Release: `loop_res_query` `[2*C, D]` (`[2i]=attn, [2i+1]=mlp`)
            // and `loop_kv_res_query` `[2*C, D]` (`[2i]=k, [2i+1]=v`).
            let row0 = (2 * core_idx) as i64;
            let row1 = row0 + 1;
            let q_attn = lq.slice(&[row0, 0], &[row0 + 1, d as i64])?;
            let q_k = lkv.slice(&[row0, 0], &[row0 + 1, d as i64])?;
            let q_v = lkv.slice(&[row1, 0], &[row1 + 1, d as i64])?;
            let q_mlp = lq.slice(&[row1, 0], &[row1 + 1, d as i64])?;
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
                load_dense_or_dequant_tensor(
                    params,
                    &format!("{base}.down_proj"),
                    per_layer_quant,
                    default_plq,
                    true,
                )?,
                load_dense_or_dequant_tensor(
                    params,
                    &format!("{base}.gain_proj"),
                    per_layer_quant,
                    default_plq,
                    true,
                )?,
                load_dense_or_dequant_tensor(
                    params,
                    &format!("{base}.forget_proj"),
                    per_layer_quant,
                    default_plq,
                    true,
                )?,
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

/// Load `embed_tokens` supporting Affine, MXFP4, MXFP8, NVFP4, all 14 GGUF K-quant/IQ
/// modes (e.g. Q6_K), optional outer scales (dequantized with outer scale applied because
/// `Embedding` packed storage has no `outer_scale` field), and dense floating weights.
fn load_inkling2_embedding(
    embed_tokens: &mut Embedding,
    params: &HashMap<String, MxArray>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Result<()> {
    let has_outer_scale = params.contains_key("embed_tokens.scale")
        || params.contains_key("embed_tokens.global_scale")
        || params.contains_key("embed_tokens.weight_scale_2")
        || params.contains_key("embed_tokens.weight_global_scale");
    if has_outer_scale {
        let w_f32 = load_dense_or_dequant_tensor(
            params,
            "embed_tokens",
            per_layer_quant,
            default_plq,
            false,
        )?;
        let w_bf16 = if w_f32.dtype()? == DType::BFloat16 {
            w_f32
        } else {
            w_f32.astype(DType::BFloat16)?
        };
        return embed_tokens.load_weight(&w_bf16);
    }
    if params.contains_key("embed_tokens.scales") {
        let embed_plq = effective_plq_for("embed_tokens", per_layer_quant, default_plq, None);
        if is_kquant_mode(embed_plq.mode) {
            ensure_row_major_kquant_layout(embed_plq, "embed_tokens", FAMILY_NAME)?;
            let w = require_tensor(params, "embed_tokens.weight")?;
            let s = require_tensor(params, "embed_tokens.scales")?;
            let b = params.get("embed_tokens.biases").cloned();
            if w.dtype()? != DType::Uint32 {
                return Err(Error::from_reason(format!(
                    "Inkling2: K-quant embed_tokens.weight must be Uint32, got {:?}",
                    w.dtype()?
                )));
            }
            let s_dtype = s.dtype()?;
            let kq = kquant_mode_params_for_scales(embed_plq.mode, s_dtype).ok_or_else(|| {
                Error::from_reason(format!(
                    "Inkling2: embed_tokens K-quant mode {:?} has unsupported scales dtype {s_dtype:?}",
                    embed_plq.mode
                ))
            })?;
            if s_dtype != kq.scales_dtype {
                return Err(Error::from_reason(format!(
                    "Inkling2: embed_tokens K-quant mode {:?} expects {:?} scales, got {s_dtype:?}",
                    embed_plq.mode, kq.scales_dtype
                )));
            }
            let Some(ref b_arr) = b else {
                return Err(Error::from_reason(format!(
                    "Inkling2: K-quant embed_tokens ({:?}) missing mandatory .biases",
                    embed_plq.mode
                )));
            };
            if b_arr.dtype()? != DType::Float16 {
                return Err(Error::from_reason(format!(
                    "Inkling2: K-quant embed_tokens ({:?}) biases must be Float16, got {:?}",
                    embed_plq.mode,
                    b_arr.dtype()?
                )));
            }
            return embed_tokens.load_quantized_packed(
                &w,
                &s,
                b.as_ref(),
                kq.group_size,
                kq.bits,
                kq.mode_str,
            );
        }
    }
    load_embedding_affine_or_bf16(
        embed_tokens,
        params,
        "embed_tokens",
        per_layer_quant,
        default_plq,
        FAMILY_NAME,
    )
}

/// Dequantize a quantized tensor (1-D or 2-D, Affine, K-quant RowMajor or Tiled64,
/// MXFP4/MXFP8/NVFP4, FP8 E4M3, Sym8) into `Float32`, or load a dense floating tensor
/// (optionally preserving native `BFloat16` when `preserve_bf16 == true`, otherwise casting
/// to `Float32`).
///
/// Enforces all MLX dequantize invariants:
/// 1. 1-D quantized tensors `expand_dims(0)` before `mlx_dequantize` and `squeeze(0)` after.
/// 2. Affine `scales` and `biases` cast to `Float32` BEFORE `mlx_dequantize` (avoiding F16
///    intermediate arithmetic inside `mlx::core::affine_dequantize`).
/// 3. K-quant `Tiled64` untiles first via `row_major_kquant_arrays`.
/// 4. Dense path rejects orphaned packed `Uint32`/`Int8`/`Uint8` via `ensure_dense_weight_floating`.
fn load_dense_or_dequant_tensor(
    params: &HashMap<String, MxArray>,
    base: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
    preserve_bf16: bool,
) -> Result<MxArray> {
    if params.contains_key(&format!("{base}.scales")) {
        let plq = effective_plq_for(base, per_layer_quant, default_plq, None);
        if plq.mode == PerLayerMode::Sym8 {
            let w = require_tensor(params, &format!("{base}.weight"))?.astype(DType::Float32)?;
            let mut s =
                require_tensor(params, &format!("{base}.scales"))?.astype(DType::Float32)?;
            if w.ndim()? == 2 && s.ndim()? == 1 {
                s = s.expand_dims(-1)?;
            }
            return w.mul(&s);
        }
        if plq.mode == PerLayerMode::Fp8E4m3 {
            let w_raw = require_tensor(params, &format!("{base}.weight"))?;
            let s_raw = require_tensor(params, &format!("{base}.scales"))?;
            let was_1d = w_raw.ndim()? == 1;
            let mut tmp = HashMap::new();
            tmp.insert(
                format!("{base}.weight"),
                if was_1d { w_raw.expand_dims(0)? } else { w_raw },
            );
            tmp.insert(
                format!("{base}.scales"),
                if was_1d && s_raw.ndim()? == 1 {
                    s_raw.expand_dims(0)?
                } else {
                    s_raw
                },
            );
            let ql = try_build_fp8_e4m3_quantized_linear(&tmp, base)?.ok_or_else(|| {
                Error::from_reason(format!("Inkling2: failed to build fp8_e4m3 for '{base}'"))
            })?;
            let mut out = ql.dequantized_weight()?.astype(DType::Float32)?;
            if was_1d {
                out = out.squeeze(Some(&[0]))?;
            }
            return Ok(out);
        }

        let (mut w, mut s, mut b, mode_str, group_size, bits) = if is_kquant_mode(plq.mode) {
            let Some((w, s, b)) = row_major_kquant_arrays(params, base, plq.mode, plq.layout)?
            else {
                return Err(Error::from_reason(format!(
                    "Inkling2: missing K-quant weight/scales for '{base}' (mode={:?})",
                    plq.mode
                )));
            };
            let s_dtype = s.dtype()?;
            let kq = kquant_mode_params_for_scales(plq.mode, s_dtype).ok_or_else(|| {
                Error::from_reason(format!(
                    "Inkling2: unsupported K-quant scales dtype {s_dtype:?} on '{base}' ({:?})",
                    plq.mode
                ))
            })?;
            (w, s, b, kq.mode_str, kq.group_size, kq.bits)
        } else {
            let w = require_tensor(params, &format!("{base}.weight"))?;
            let s = require_tensor(params, &format!("{base}.scales"))?;
            match plq.mode {
                PerLayerMode::Affine => {
                    // CRITICAL: cast affine scales & biases to Float32 BEFORE mlx_dequantize
                    // because `mlx::core::affine_dequantize` computes `w * s + b` in scales.dtype().
                    let s_f32 = if s.dtype()? == DType::Float32 {
                        s
                    } else {
                        s.astype(DType::Float32)?
                    };
                    let b_f32 = match params.get(&format!("{base}.biases")) {
                        Some(b) => Some(if b.dtype()? == DType::Float32 {
                            b.clone()
                        } else {
                            b.astype(DType::Float32)?
                        }),
                        None => None,
                    };
                    (w, s_f32, b_f32, "affine", plq.group_size, plq.bits)
                }
                PerLayerMode::Mxfp4 => (w, s, None, MXFP4_MODE, MXFP4_GROUP_SIZE, MXFP4_BITS),
                PerLayerMode::Mxfp8 => (w, s, None, MXFP8_MODE, MXFP8_GROUP_SIZE, MXFP8_BITS),
                PerLayerMode::Nvfp4 => (w, s, None, NVFP4_MODE, NVFP4_GROUP_SIZE, NVFP4_BITS),
                other => {
                    return Err(Error::from_reason(format!(
                        "Inkling2: unsupported quantized small-tensor mode {other:?} on '{base}'"
                    )));
                }
            }
        };

        let was_1d = w.ndim()? == 1;
        if was_1d {
            w = w.expand_dims(0)?;
            if s.ndim()? == 1 {
                s = s.expand_dims(0)?;
            }
            if let Some(ref b_arr) = b
                && b_arr.ndim()? == 1
            {
                b = Some(b_arr.expand_dims(0)?);
            }
        }

        let mode_c = CString::new(mode_str)
            .map_err(|e| Error::from_reason(format!("Invalid quant mode '{mode_str}': {e}")))?;
        let biases_ptr = b
            .as_ref()
            .map_or(std::ptr::null_mut(), |arr| arr.as_raw_ptr());
        let handle = unsafe {
            mlx_sys::mlx_dequantize(
                w.as_raw_ptr(),
                s.as_raw_ptr(),
                biases_ptr,
                group_size,
                bits,
                DType::Float32 as i32,
                mode_c.as_ptr(),
            )
        };
        if handle.is_null() {
            return Err(Error::from_reason(format!(
                "Inkling2: mlx_dequantize failed for '{base}' (mode={mode_str}, gs={group_size}, bits={bits})"
            )));
        }
        let mut out = MxArray::from_handle(handle, "inkling2_small_tensor_dequant")?;
        if was_1d {
            out = out.squeeze(Some(&[0]))?;
        }
        if out.dtype()? != DType::Float32 {
            out = out.astype(DType::Float32)?;
        }
        if plq.mode == PerLayerMode::Nvfp4 {
            if let Some(outer) = resolve_nvfp4_outer_scale_2d(params, base)? {
                out = out.mul(&outer)?;
            }
        } else if let Some(raw_scale) = params.get(&format!("{base}.scale")) {
            let outer = normalize_scalar_outer_scale(raw_scale, &format!("{base}.scale"))?;
            out = out.mul(&outer)?;
        }
        return Ok(out);
    }

    let key = format!("{base}.weight");
    let w = require_tensor(params, &key)?;
    ensure_dense_weight_floating(&key, &w)?;
    let dt = w.dtype()?;
    if (preserve_bf16 && dt == DType::BFloat16) || dt == DType::Float32 {
        Ok(w)
    } else {
        w.astype(DType::Float32)
    }
}

/// Load an Inkling2 2-D linear projection (`self_attn.*` or `lm_head`) and preserve any scalar
/// `{base}.scale` outer scale on non-NVFP4 projections (such as oMLX MXFP4/MXFP8 fused
/// attention splits). NVFP4 projections already resolve `.scale`/`.global_scale`/`.weight_scale_2`/
/// `.weight_global_scale` inside [`load_linear_proj_quantized_or_bf16`]; NVFP4-only outer-scale
/// sidecars on non-NVFP4 projections fail loudly instead of being silently ignored.
fn load_inkling2_linear_proj(
    proj: &mut LinearProj,
    params: &HashMap<String, MxArray>,
    base: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Result<()> {
    load_linear_proj_quantized_or_bf16(
        proj,
        params,
        base,
        per_layer_quant,
        default_plq,
        FAMILY_NAME,
    )?;

    let is_nvfp4_quant = matches!(proj, LinearProj::Quantized(ql) if ql.mode() == NVFP4_MODE);
    if is_nvfp4_quant {
        return Ok(());
    }

    for bad_sfx in [".global_scale", ".weight_scale_2", ".weight_global_scale"] {
        let bad_key = format!("{base}{bad_sfx}");
        if params.contains_key(&bad_key) {
            return Err(Error::from_reason(format!(
                "Inkling2: unsupported NVFP4 outer scale tensor '{bad_key}' on non-NVFP4 projection '{base}'"
            )));
        }
    }

    let scale_key = format!("{base}.scale");
    let Some(raw_scale) = params.get(&scale_key) else {
        return Ok(());
    };
    let outer = normalize_scalar_outer_scale(raw_scale, &scale_key)?;
    match proj {
        LinearProj::Quantized(ql) => {
            if ql.outer_scale().is_none() {
                let placeholder = Linear::new(1, 1, Some(false))?;
                let owned = std::mem::replace(proj, LinearProj::Standard(placeholder));
                if let LinearProj::Quantized(ql_owned) = owned {
                    *proj = LinearProj::Quantized(ql_owned.with_outer_scale(outer));
                }
            }
        }
        LinearProj::Standard(lin) => {
            let w = lin.get_weight();
            let dt = w.dtype()?;
            let w_f32 = if dt == DType::Float32 {
                w
            } else {
                w.astype(DType::Float32)?
            };
            let scaled = w_f32.mul(&outer)?;
            let scaled = if dt == DType::Float32 {
                scaled
            } else {
                scaled.astype(dt)?
            };
            let bias = lin.get_bias();
            *proj = LinearProj::Standard(Linear::from_weights(&scaled, bias.as_ref())?);
        }
    }
    Ok(())
}

fn load_dense_or_dequant_f32(
    params: &HashMap<String, MxArray>,
    base: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Result<MxArray> {
    load_dense_or_dequant_tensor(params, base, per_layer_quant, default_plq, false)
}

fn try_load_dense_or_dequant_f32(
    params: &HashMap<String, MxArray>,
    base: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Result<Option<MxArray>> {
    if params.contains_key(&format!("{base}.weight"))
        || params.contains_key(&format!("{base}.scales"))
    {
        Ok(Some(load_dense_or_dequant_f32(
            params,
            base,
            per_layer_quant,
            default_plq,
        )?))
    } else {
        Ok(None)
    }
}

fn outer_scales_bitwise_compatible(
    a: Option<&MxArray>,
    b: Option<&MxArray>,
) -> Result<Option<Option<MxArray>>> {
    match (a, b) {
        (None, None) => Ok(Some(None)),
        (Some(sa), Some(sb)) => {
            if sa.shape()?.as_ref() != sb.shape()?.as_ref() {
                return Ok(None);
            }
            let va = if sa.dtype()? == DType::Float32 {
                sa.to_float32()?
            } else {
                sa.astype(DType::Float32)?.to_float32()?
            };
            let vb = if sb.dtype()? == DType::Float32 {
                sb.to_float32()?
            } else {
                sb.astype(DType::Float32)?.to_float32()?
            };
            if va.len() == vb.len()
                && va
                    .iter()
                    .zip(vb.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits())
            {
                Ok(Some(Some(sa.clone())))
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

fn try_fuse_split_expert_gate_up(gate: &SwitchProj, up: &SwitchProj) -> Result<Option<SwitchProj>> {
    match (gate, up) {
        (SwitchProj::Standard(g), SwitchProj::Standard(u)) => {
            let Some(shared_scale) =
                outer_scales_bitwise_compatible(g.outer_scale(), u.outer_scale())?
            else {
                return Ok(None);
            };
            let gw = g.weight_ref();
            let uw = u.weight_ref();
            let gs = gw.shape()?;
            let us = uw.shape()?;
            if gw.dtype()? != uw.dtype()?
                || gs.len() != 3
                || us.len() != 3
                || gs[0] != us[0]
                || gs[1] != us[1]
                || gs[2] != us[2]
            {
                return Ok(None);
            }
            let w = MxArray::concatenate(gw, uw, 1)?;
            let mut sl = SwitchLinear::from_weight(w);
            if let Some(scale) = shared_scale {
                sl = sl.with_outer_scale(scale);
            }
            Ok(Some(SwitchProj::Standard(sl)))
        }
        (SwitchProj::Quantized(g), SwitchProj::Quantized(u)) => {
            if g.mode() != u.mode()
                || g.bits() != u.bits()
                || g.group_size() != u.group_size()
                || !g.transpose()
                || !u.transpose()
                || crate::models::quant_dispatch::split_kquant_layout(g.mode()).1
                || g.is_symmetric_q8()
                || u.is_symmetric_q8()
                || g.reconstructed_fp8_weight().is_some()
                || u.reconstructed_fp8_weight().is_some()
            {
                return Ok(None);
            }
            let Some(shared_scale) =
                outer_scales_bitwise_compatible(g.outer_scale(), u.outer_scale())?
            else {
                return Ok(None);
            };
            let (gw, uw) = (g.get_weight(), u.get_weight());
            let (gs, us) = (g.get_scales(), u.get_scales());
            if gw.dtype()? != uw.dtype()? || gs.dtype()? != us.dtype()? {
                return Ok(None);
            }
            let (gws, uws) = (gw.shape()?, uw.shape()?);
            let (gss, uss) = (gs.shape()?, us.shape()?);
            if gws.len() != 3
                || uws.len() != 3
                || gss.len() != 3
                || uss.len() != 3
                || gws[0] != uws[0]
                || gws[1] != uws[1]
                || gws[2] != uws[2]
                || gss[0] != uss[0]
                || gss[1] != uss[1]
                || gss[2] != uss[2]
            {
                return Ok(None);
            }
            let biases = match (g.get_biases(), u.get_biases()) {
                (Some(gb), Some(ub)) => {
                    let (gbs, ubs) = (gb.shape()?, ub.shape()?);
                    if gb.dtype()? != ub.dtype()?
                        || gbs.len() != 3
                        || ubs.len() != 3
                        || gbs[0] != ubs[0]
                        || gbs[1] != ubs[1]
                        || gbs[2] != ubs[2]
                    {
                        return Ok(None);
                    }
                    Some(MxArray::concatenate(gb, ub, 1)?)
                }
                (None, None) => None,
                _ => return Ok(None),
            };
            let weight = MxArray::concatenate(gw, uw, 1)?;
            let scales = MxArray::concatenate(gs, us, 1)?;
            let mut qsl = QuantizedSwitchLinear::new(
                weight,
                scales,
                biases,
                g.group_size(),
                g.bits(),
                g.mode().to_string(),
            );
            if let Some(scale) = shared_scale {
                qsl = qsl.with_outer_scale(scale);
            }
            Ok(Some(SwitchProj::Quantized(qsl)))
        }
        _ => Ok(None),
    }
}

/// Load MoE expert `gate_up_proj`, handling both fused `gate_up_proj` and split
/// `gate_proj` + `up_proj` (GGUF `ffn_gate_exps`+`ffn_up_exps` or `mlx-lm`
/// `switch_mlp.{gate,up}_proj`).
///
/// Compatible split pairs are fused along axis 1 BEFORE `try_compact_symmetric_q8_or_promote_bf16()`,
/// and split layers always return `effective_gate_up_interleaved = false`.
fn load_expert_gate_up(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
    config_gate_up_interleaved: bool,
) -> Result<(ExpertGateUp, bool)> {
    let gu_base = format!("{prefix}.mlp.experts.gate_up_proj");
    if params.contains_key(&format!("{gu_base}.weight"))
        || params.contains_key(&format!("{gu_base}.scales"))
    {
        let mut gu = load_expert_switch_proj(params, &gu_base, per_layer_quant, default_plq)?;
        gu.try_compact_symmetric_q8_or_promote_bf16()?;
        return Ok((ExpertGateUp::Fused(gu), config_gate_up_interleaved));
    }

    let gate_base = format!("{prefix}.mlp.experts.gate_proj");
    let up_base = format!("{prefix}.mlp.experts.up_proj");
    let has_gate = params.contains_key(&format!("{gate_base}.weight"))
        || params.contains_key(&format!("{gate_base}.scales"));
    let has_up = params.contains_key(&format!("{up_base}.weight"))
        || params.contains_key(&format!("{up_base}.scales"));
    if has_gate && has_up {
        let mut gate = load_expert_switch_proj(params, &gate_base, per_layer_quant, default_plq)?;
        let mut up = load_expert_switch_proj(params, &up_base, per_layer_quant, default_plq)?;
        if let Some(mut fused) = try_fuse_split_expert_gate_up(&gate, &up)? {
            fused.try_compact_symmetric_q8_or_promote_bf16()?;
            return Ok((ExpertGateUp::Fused(fused), false));
        }
        gate.try_compact_symmetric_q8_or_promote_bf16()?;
        up.try_compact_symmetric_q8_or_promote_bf16()?;
        return Ok((ExpertGateUp::Split { gate, up }, false));
    }

    Err(Error::from_reason(format!(
        "Inkling2: missing required MoE expert projection '{gu_base}.weight' (or split '{gate_base}.weight' + '{up_base}.weight')"
    )))
}

fn load_expert_switch_proj(
    params: &HashMap<String, MxArray>,
    base: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Result<SwitchProj> {
    if let Some(scales) = params.get(&format!("{base}.scales")) {
        // Native SafeTensors NVFP4 / MXFP8 / MXFP4 checkpoints (`quant_method: "modelopt"` or
        // flat MLX) have no per-layer override, store Uint8 FP8/E8M0 scales, and carry no `.biases`.
        // Match shape ratio FIRST (`* 2` => NVFP4, `* 8` => MXFP8, `* 4` => MXFP4) so neither
        // unscaled NVFP4 nor MXFP8-with-`.scale` falls through to Affine.
        let has_override = per_layer_quant.contains_key(base);
        if !has_override
            && scales.dtype()? == DType::Uint8
            && !params.contains_key(&format!("{base}.biases"))
        {
            let w = require_tensor(params, &format!("{base}.weight"))?;
            let w_shape = w.shape()?;
            let s_shape = scales.shape()?;
            if w.dtype()? == DType::Uint32
                && w_shape.len() == 3
                && s_shape.len() == 3
                && w_shape[0] == s_shape[0]
                && w_shape[1] == s_shape[1]
            {
                let num_experts = w_shape[0];
                if w_shape[2] == s_shape[2] * 2 {
                    // NVFP4: uint32 packed 4-bit (8 elems/u32) + uint8 E4M3 scales (16 elems/scale)
                    let mut q = QuantizedSwitchLinear::new(
                        w,
                        scales.clone(),
                        None,
                        NVFP4_GROUP_SIZE,
                        NVFP4_BITS,
                        NVFP4_MODE.to_string(),
                    );
                    if let Some(outer) = resolve_nvfp4_outer_scale_3d(params, base, num_experts)? {
                        q = q.with_outer_scale(outer);
                    }
                    return Ok(SwitchProj::Quantized(q));
                } else if w_shape[2] == s_shape[2] * 8 {
                    // MXFP8: uint32 packed FP8 E4M3 (4 elems/u32) + uint8 UE8M0 scales (32 elems/scale)
                    let mut q = QuantizedSwitchLinear::new(
                        w,
                        scales.clone(),
                        None,
                        MXFP8_GROUP_SIZE,
                        MXFP8_BITS,
                        MXFP8_MODE.to_string(),
                    );
                    if let Some(outer) = resolve_optional_outer_scale_3d(params, base, num_experts)?
                    {
                        q = q.with_outer_scale(outer);
                    }
                    return Ok(SwitchProj::Quantized(q));
                } else if w_shape[2] == s_shape[2] * 4 {
                    // MXFP4: uint32 packed 4-bit (8 elems/u32) + uint8 UE8M0 scales (32 elems/scale)
                    let mut q = QuantizedSwitchLinear::new(
                        w,
                        scales.clone(),
                        None,
                        MXFP4_GROUP_SIZE,
                        MXFP4_BITS,
                        MXFP4_MODE.to_string(),
                    );
                    if let Some(outer) = resolve_optional_outer_scale_3d(params, base, num_experts)?
                    {
                        q = q.with_outer_scale(outer);
                    }
                    return Ok(SwitchProj::Quantized(q));
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
        return Ok(SwitchProj::Quantized(qsl));
    }

    let weight_key = format!("{base}.weight");
    let mut w = require_tensor(params, &weight_key)?;
    ensure_dense_weight_floating(&weight_key, &w)?;
    // Normalize non-BF16 dense expert weights (e.g. SafeTensors/unconverted Float16) to BFloat16
    // once at load time: prevents MLX `gather_mm(BF16, F16)` from promoting BOTH operands to
    // Float32 (~4.08 GB/visit) and keeps Stage 2 `inkling2_situ`/`inkling2_moe_combine` compatible.
    if w.dtype()? != DType::BFloat16 {
        w = w.astype(DType::BFloat16)?;
    }
    let num_experts = w.shape_at(0)?;
    let mut sl = SwitchLinear::from_weight(w);
    if let Some(outer) = resolve_optional_outer_scale_3d(params, base, num_experts)? {
        sl = sl.with_outer_scale(outer);
    }
    Ok(SwitchProj::Standard(sl))
}

fn build_vision_encoder(
    config: &Inkling2VisionConfig,
    params: &HashMap<String, MxArray>,
    expected_projection_dim: usize,
) -> Result<Inkling2VisionEncoder> {
    if config.projection_dim != expected_projection_dim {
        return Err(Error::from_reason(format!(
            "Inkling2 vision: projection_dim ({}) does not match text hidden_size ({expected_projection_dim})",
            config.projection_dim
        )));
    }
    if config.patch_size == 0
        || config.hidden_size == 0
        || config.num_attention_heads == 0
        || !config
            .hidden_size
            .is_multiple_of(config.num_attention_heads)
        || config.spatial_merge_size == 0
        || config.posemb_grid_size() == 0
    {
        return Err(Error::from_reason(format!(
            "Inkling2 vision: invalid vision config {config:?}"
        )));
    }

    let p = config.patch_size as i64;
    let d = config.hidden_size as i64;
    let m = config.spatial_merge_size as i64;
    let grid = config.posemb_grid_size() as i64;
    let grid_sq = grid * grid;
    let eps = Some(config.layer_norm_eps);

    let require_float = |key: &str| -> Result<MxArray> {
        let t = require_tensor(params, key)?;
        ensure_dense_weight_floating(key, &t)?;
        Ok(t)
    };
    let normalize_vec_d = |t: MxArray, key: &str| -> Result<MxArray> {
        let shape = t.shape()?;
        match shape.as_ref() {
            [dim] if *dim == d => Ok(t),
            [1, dim] if *dim == d => t.squeeze(Some(&[0])),
            other => Err(Error::from_reason(format!(
                "Inkling2 vision: unexpected {key} shape {other:?} (expected [{d}] or [1, {d}])"
            ))),
        }
    };

    let class_embedding = normalize_vec_d(
        require_float("vision_tower.class_embedding")?,
        "vision_tower.class_embedding",
    )?;
    let class_position_embedding = normalize_vec_d(
        require_float("vision_tower.class_position_embedding.weight")?,
        "vision_tower.class_position_embedding.weight",
    )?;

    // GGUF loads `v.patch_embd.weight` as OIHW `[1024, 3, 20, 20]`.
    // Handle OHWI `[1024, 20, 20, 3]` or pre-flattened `[1024, 1200]` cleanly too.
    let patch_raw = require_float("vision_tower.patch_embed.weight")?;
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

    let pos_raw = require_float("vision_tower.position_embedding.weight")?;
    let pos_shape = pos_raw.shape()?;
    let position_embedding = match pos_shape.as_ref() {
        [n, dim] if *n == grid_sq && *dim == d => pos_raw,
        [1, n, dim] if *n == grid_sq && *dim == d => pos_raw.squeeze(Some(&[0]))?,
        [gh, gw, dim] if *gh == grid && *gw == grid && *dim == d => {
            pos_raw.reshape(&[grid_sq, d])?
        }
        other => {
            return Err(Error::from_reason(format!(
                "Inkling2 vision: unexpected position_embedding shape {other:?} (expected [{grid_sq}, {d}], [1, {grid_sq}, {d}], or [{grid}, {grid}, {d}])"
            )));
        }
    };

    let pre_ln = LayerNorm::from_weights(
        &require_float("vision_tower.pre_ln.weight")?,
        Some(&require_float("vision_tower.pre_ln.bias")?),
        eps,
    )?;

    let load_f32_vec = |key: &str| -> Result<MxArray> {
        let t = normalize_vec_d(require_float(key)?, key)?;
        if t.dtype()? == DType::Float32 {
            Ok(t)
        } else {
            t.astype(DType::Float32)
        }
    };
    let load_vision_linear = |base: &str| -> Result<VisionLinear> {
        let w_key = format!("{base}.weight");
        let w = require_float(&w_key)?;
        let b_key = format!("{base}.bias");
        let b = match params.get(&b_key) {
            Some(bias) => {
                ensure_dense_weight_floating(&b_key, bias)?;
                Some(bias)
            }
            None => None,
        };
        VisionLinear::from_weights(&w, b)
    };

    let mut blocks = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let bp = format!("vision_tower.blocks.{i}");
        let norm1 = LayerNorm::from_weights(
            &require_float(&format!("{bp}.norm1.weight"))?,
            Some(&require_float(&format!("{bp}.norm1.bias"))?),
            eps,
        )?;
        let norm2 = LayerNorm::from_weights(
            &require_float(&format!("{bp}.norm2.weight"))?,
            Some(&require_float(&format!("{bp}.norm2.bias"))?),
            eps,
        )?;
        blocks.push(Inkling2VisionBlock {
            norm1,
            q_proj: load_vision_linear(&format!("{bp}.attn.q_proj"))?,
            k_proj: load_vision_linear(&format!("{bp}.attn.k_proj"))?,
            v_proj: load_vision_linear(&format!("{bp}.attn.v_proj"))?,
            o_proj: load_vision_linear(&format!("{bp}.attn.o_proj"))?,
            ls1: load_f32_vec(&format!("{bp}.ls1.weight"))?,
            norm2,
            fc1: load_vision_linear(&format!("{bp}.mlp.fc1"))?,
            fc2: load_vision_linear(&format!("{bp}.mlp.fc2"))?,
            ls2: load_f32_vec(&format!("{bp}.ls2.weight"))?,
        });
    }

    let proj_fc1 = load_vision_linear("vision_projector.fc1")?;
    let proj_fc2 = load_vision_linear("vision_projector.fc2")?;
    let merged_dim = m * m * d;
    let fc1_shape = proj_fc1.linear.get_weight().shape()?;
    if fc1_shape.len() != 2 || fc1_shape[1] != merged_dim {
        return Err(Error::from_reason(format!(
            "Inkling2 vision: vision_projector.fc1.weight shape {:?} incompatible with merged dim {merged_dim}",
            fc1_shape.as_ref()
        )));
    }
    let fc2_shape = proj_fc2.linear.get_weight().shape()?;
    if fc2_shape.len() != 2
        || fc2_shape[0] != expected_projection_dim as i64
        || fc2_shape[1] != fc1_shape[0]
    {
        return Err(Error::from_reason(format!(
            "Inkling2 vision: vision_projector.fc2.weight shape {:?} incompatible with fc1 out {} and projection_dim {expected_projection_dim}",
            fc2_shape.as_ref(),
            fc1_shape[0]
        )));
    }

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

/// Detected SafeTensors checkpoint naming layout (probed on raw keys BEFORE prefix stripping).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Inkling2CheckpointLayout {
    /// Native Thinking Machines HF SafeTensors (`model.llm.embed.weight`, interleaved `w13`).
    Release,
    /// oMLX converted SafeTensors (`model.embed.weight`, fused `attn.in_proj`, interleaved `w13`).
    OMlx,
    /// mlx-vlm converted SafeTensors (`model.language_model.embed_tokens.weight`, interleaved `gate_up_proj`).
    MlxVlm,
    /// mlx-lm converted SafeTensors (`language_model.model.embed_tokens.weight`, split `switch_mlp.{gate,up}_proj`).
    MlxLm,
    /// GGUF or already-canonical Inkling2 layout.
    Canonical,
}

const ST_QUANT_SUFFIXES: [&str; 8] = [
    ".weight",
    ".scales",
    ".weight_scale",
    ".biases",
    ".scale",
    ".global_scale",
    ".weight_scale_2",
    ".weight_global_scale",
];

fn split_st_quant_suffix(key: &str) -> Option<(&str, &'static str)> {
    for suffix in ST_QUANT_SUFFIXES {
        if let Some(base) = key.strip_suffix(suffix) {
            return Some((base, suffix));
        }
    }
    None
}

fn canonical_st_quant_suffix(suffix: &'static str) -> &'static str {
    if suffix == ".weight_scale" {
        ".scales"
    } else {
        suffix
    }
}

fn is_scalar_outer_scale_suffix(suffix: &str) -> bool {
    matches!(
        suffix,
        ".scale" | ".global_scale" | ".weight_scale_2" | ".weight_global_scale"
    )
}

fn maybe_view_u8_to_u32(tensor: MxArray) -> Result<MxArray> {
    if tensor.dtype()? == DType::Uint8 {
        tensor.view(DType::Uint32)
    } else {
        Ok(tensor)
    }
}

fn expand_global_loop_msg_queries(
    out: &mut HashMap<String, MxArray>,
    tensor: MxArray,
    config: &Inkling2Config,
    key: &str,
) -> Result<()> {
    let lc = config.loop_config.as_ref().ok_or_else(|| {
        Error::from_reason(format!(
            "Inkling2 ST: unexpected top-level loop query '{key}' on model without loop_config"
        ))
    })?;
    let expected_core = lc.core_end.saturating_sub(lc.core_start) as i64;
    let d = config.hidden_size as i64;
    let shape = tensor.shape()?;
    if shape.as_ref() != [expected_core, 4, d] {
        return Err(Error::from_reason(format!(
            "Inkling2 ST: top-level loop query '{key}' has shape {:?}, expected [{expected_core}, 4, {d}]",
            shape.as_ref()
        )));
    }
    for ci in 0..expected_core {
        let q = tensor
            .slice(&[ci, 0, 0], &[ci + 1, 4, d])?
            .squeeze(Some(&[0]))?;
        // Deterministic precedence: per-layer `layers.{L}.loop_msg_query.weight` wins if already present.
        out.entry(format!(
            "layers.{}.loop_msg_query.weight",
            lc.core_start + ci as usize
        ))
        .or_insert(q);
    }
    Ok(())
}

fn normalize_sconv_weight(tensor: MxArray, key: &str) -> Result<MxArray> {
    match tensor.ndim()? {
        2 => Ok(tensor),
        3 => {
            let shape = tensor.shape()?;
            if shape[1] == 1 {
                tensor.squeeze(Some(&[1]))
            } else if shape[2] == 1 {
                tensor.squeeze(Some(&[2]))
            } else {
                Err(Error::from_reason(format!(
                    "Inkling2 ST: unexpected 3-D sconv weight shape {:?} for '{key}' (expected [C, 1, K] or [C, K, 1])",
                    shape.as_ref()
                )))
            }
        }
        other => Err(Error::from_reason(format!(
            "Inkling2 ST: unexpected {other}-D sconv weight for '{key}'"
        ))),
    }
}

fn normalize_dynamic_static_conv(tensor: MxArray, rev_k: &MxArray) -> Result<MxArray> {
    tensor.transpose(None)?.take(rev_k, 1)
}

fn detect_inkling2_safetensors_layout(raw: &HashMap<String, MxArray>) -> Inkling2CheckpointLayout {
    if raw.contains_key("model.llm.embed.weight")
        || raw.contains_key("model.llm.embed.scales")
        || raw.contains_key("llm.embed.weight")
        || raw.contains_key("llm.embed.scales")
        || raw.contains_key("model.llm.layers.0.attn.wq_du.weight")
        || raw.contains_key("llm.layers.0.attn.wq_du.weight")
    {
        return Inkling2CheckpointLayout::Release;
    }
    if raw.contains_key("model.embed.weight")
        || raw.contains_key("model.embed.scales")
        || raw.contains_key("embed.weight")
        || raw.contains_key("embed.scales")
        || raw.keys().any(|k| {
            k.contains(".attn.in_proj.")
                || k.ends_with(".mlp.router_weight")
                || k.contains(".mlp.experts.w13.")
                || k.ends_with(".loop_down")
        })
    {
        return Inkling2CheckpointLayout::OMlx;
    }
    // Check `MlxVlm`-exclusive structural markers BEFORE `MlxLm`-exclusive markers so a
    // partially converted `mlx-vlm` checkpoint with both fused and split layers is still
    // detected as `MlxVlm` (preserving interleaved order for its fused layers while split
    // layers force non-interleaved per-layer inside `load_expert_gate_up`).
    if raw.keys().any(|k| {
        k.contains(".post_attention_convolution.")
            || k.contains(".post_mlp_convolution.")
            || k.ends_with(".self_attn.q_l2_clip.scale")
            || k.contains(".self_attn.rel_logits_proj.proj")
    }) {
        return Inkling2CheckpointLayout::MlxVlm;
    }
    if raw.keys().any(|k| {
        k.contains(".mlp.switch_mlp.")
            || k.ends_with(".mlp.gate_weight")
            || k.ends_with(".self_attn.q_scale")
            || k.contains(".attn_sconv.conv.")
    }) {
        return Inkling2CheckpointLayout::MlxLm;
    }
    // Shared `mlx-vlm` / `mlx-lm` markers: disambiguate by wrapper prefix when present.
    if raw.keys().any(|k| {
        k.contains(".q_message_gate.")
            || k.ends_with(".attention_input.query")
            || k.contains(".self_attn.r_proj.")
    }) {
        if raw.keys().any(|k| {
            k.starts_with("language_model.model.") || k.starts_with("language_model.lm_head.")
        }) {
            return Inkling2CheckpointLayout::MlxLm;
        }
        return Inkling2CheckpointLayout::MlxVlm;
    }
    // Recognize canonical tensor names BEFORE wrapper-only fallbacks so an already-canonical
    // checkpoint nested under `language_model.model.*` keeps `Canonical` and preserves explicit
    // `config.gate_up_interleaved`.
    if raw.keys().any(|k| {
        k.contains(".self_attn.k_conv.")
            || k.contains(".self_attn.v_conv.")
            || k.contains(".self_attn.post_conv.")
            || k.contains(".mlp.post_conv.")
            || k.contains(".loop_ctrl.")
            || k.contains(".attn_res_query.")
            || k.contains(".ffn_res_query.")
            || k.contains(".loop_msg_query.")
            || k.contains(".self_attn.q_clip.")
            || k.contains(".self_attn.rel_bias.")
    }) {
        return Inkling2CheckpointLayout::Canonical;
    }
    if raw
        .keys()
        .any(|k| k.starts_with("language_model.model.") || k.starts_with("language_model.lm_head."))
    {
        return Inkling2CheckpointLayout::MlxLm;
    }
    if raw.keys().any(|k| k.starts_with("model.language_model.")) {
        return Inkling2CheckpointLayout::MlxVlm;
    }
    Inkling2CheckpointLayout::Canonical
}

fn strip_inkling2_wrapper_prefixes(
    raw_params: HashMap<String, MxArray>,
) -> HashMap<String, MxArray> {
    let mut params = HashMap::with_capacity(raw_params.len());
    for (k, v) in raw_params {
        let norm_key = if let Some(rest) = k.strip_prefix("model.language_model.model.") {
            rest.to_owned()
        } else if let Some(rest) = k.strip_prefix("model.language_model.") {
            rest.to_owned()
        } else if let Some(rest) = k.strip_prefix("language_model.model.") {
            rest.to_owned()
        } else if let Some(rest) = k.strip_prefix("language_model.") {
            rest.to_owned()
        } else if let Some(rest) = k.strip_prefix("model.") {
            rest.to_owned()
        } else {
            k
        };
        params.insert(norm_key, v);
    }
    params
}

fn slice_fused_projection_group(
    out: &mut HashMap<String, MxArray>,
    lp: &str,
    suffix: &str,
    tensor: MxArray,
    segments: &[(&str, i64)],
    source_key: &str,
) -> Result<()> {
    if is_scalar_outer_scale_suffix(suffix) && tensor.size()? == 1 {
        for &(target_proj, _) in segments {
            out.insert(
                format!("{lp}.self_attn.{target_proj}{suffix}"),
                tensor.clone(),
            );
        }
        return Ok(());
    }
    let total_expected: i64 = segments.iter().map(|(_, rows)| *rows).sum();
    let total_rows = tensor.shape_at(0)?;
    if total_rows != total_expected {
        return Err(Error::from_reason(format!(
            "Inkling2 oMLX: fused projection '{source_key}' has {total_rows} rows, expected {total_expected}"
        )));
    }
    let mut offset = 0i64;
    for &(target_proj, rows) in segments {
        let sliced = tensor.slice_axis(0, offset, offset + rows)?;
        out.insert(format!("{lp}.self_attn.{target_proj}{suffix}"), sliced);
        offset += rows;
    }
    Ok(())
}

/// Shape-derived normalization metadata returned by [`normalize_inkling2_safetensors_params`]
/// so [`remap_inkling2_per_layer_quant`] follows the exact same oMLX split plan as the tensors.
#[derive(Default, Debug, Clone)]
struct Inkling2StNormResult {
    gate_up_interleaved: Option<bool>,
    omlx_fused_in_proj: HashSet<usize>,
    omlx_fused_q: HashSet<usize>,
    omlx_fused_k: HashSet<usize>,
    omlx_fused_v: HashSet<usize>,
}

/// Normalize SafeTensors text layouts (`Release`, `oMLX`, `mlx-vlm`, `mlx-lm`) into the
/// canonical internal Inkling2 namespace used by `load_weights_sync`.
fn normalize_inkling2_safetensors_params(
    params: &mut HashMap<String, MxArray>,
    layout: Inkling2CheckpointLayout,
    config: &Inkling2Config,
) -> Result<Inkling2StNormResult> {
    if layout == Inkling2CheckpointLayout::Canonical {
        // Even when per-layer tensors are already canonical, expand a top-level
        // `[C, 4, D]` loop message query tensor if present (per-layer queries take precedence).
        for top_key in [
            "message_input.query",
            "message_queries",
            "loop_message.query",
        ] {
            if let Some(tensor) = params.remove(top_key) {
                expand_global_loop_msg_queries(params, tensor, config, top_key)?;
            }
        }
        return Ok(Inkling2StNormResult::default());
    }

    let raw = std::mem::take(params);
    let mut out: HashMap<String, MxArray> = HashMap::with_capacity(raw.len());

    let d = config.hidden_size as i64;
    let n_heads = config.num_attention_heads as i64;
    let n_kv = config.num_key_value_heads as i64;
    let head_dim = config.head_dim as i64;
    let d_rel = config.d_rel as i64;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv * head_dim;
    let rel_dim = n_heads * d_rel;
    let taps = n_kv * config.sconv_kernel_size as i64;
    let gate_rank = config
        .loop_config
        .as_ref()
        .map_or(64i64, |lc| lc.gate_rank as i64);

    let k_len = config.sconv_kernel_size as i32;
    let rev_vals: Vec<i32> = (0..k_len).rev().collect();
    let rev_k = MxArray::from_int32(&rev_vals, &[k_len as i64])?;

    // Release layout pairs `weight_amax` and `weight_outer_scale_denominator` per `(layer, proj)`.
    let mut expert_amax: HashMap<(usize, &'static str), MxArray> = HashMap::new();
    let mut expert_denom: HashMap<(usize, &'static str), MxArray> = HashMap::new();

    // Pre-scan which raw prefixes carry a quantization scale companion (`.scales` / `.weight_scale` /
    // Release `*_weight_scale`) so packed Uint8 ModelOpt weights are reinterpreted as Uint32 ONLY
    // when paired with scales (leaving orphaned Uint8 dense weights caught by `ensure_dense_weight_floating`).
    let mut bases_with_scales: HashSet<String> = HashSet::new();
    for k in raw.keys() {
        if let Some(base) = k
            .strip_suffix(".scales")
            .or_else(|| k.strip_suffix(".weight_scale"))
        {
            bases_with_scales.insert(base.to_string());
        } else if let Some(base) = k.strip_suffix("_weight_scale") {
            bases_with_scales.insert(format!("{base}_weight"));
        }
    }

    // Pre-scan oMLX fused projection shapes so `.scales`/`.biases`/`.global_scale` and
    // `per_layer_quant` overrides fan out consistently from the exact same split plan.
    let mut omlx_fused_in_proj: HashSet<usize> = HashSet::new();
    let mut omlx_fused_q: HashSet<usize> = HashSet::new();
    let mut omlx_fused_k: HashSet<usize> = HashSet::new();
    let mut omlx_fused_v: HashSet<usize> = HashSet::new();
    if layout == Inkling2CheckpointLayout::OMlx {
        let full_q_rows = 2 * q_dim + rel_dim + n_heads;
        let full_kv_rows = kv_dim + taps;
        for (key, tensor) in &raw {
            if let Some(rest) = key.strip_prefix("layers.")
                && let Some((layer_str, suffix)) = rest.split_once('.')
                && let Ok(layer_idx) = layer_str.parse::<usize>()
            {
                if suffix == "attn.in_proj.weight" {
                    omlx_fused_in_proj.insert(layer_idx);
                } else if suffix == "attn.q_proj.weight"
                    && tensor.shape_at(0).ok() == Some(full_q_rows)
                {
                    omlx_fused_q.insert(layer_idx);
                } else if suffix == "attn.k_proj.weight"
                    && tensor.shape_at(0).ok() == Some(full_kv_rows)
                {
                    omlx_fused_k.insert(layer_idx);
                } else if suffix == "attn.v_proj.weight"
                    && tensor.shape_at(0).ok() == Some(full_kv_rows)
                {
                    omlx_fused_v.insert(layer_idx);
                }
            }
        }
    }

    for (key, tensor) in raw {
        // Leave vision keys untouched for `normalize_inkling2_vision_params`.
        if key.starts_with("visual.")
            || key.starts_with("vision_tower.")
            || key.starts_with("vision_projector.")
        {
            out.insert(key, tensor);
            continue;
        }

        match layout {
            Inkling2CheckpointLayout::Release => {
                // 1. Top-level Release LLM weights
                if let Some((base, q_sfx)) = split_st_quant_suffix(&key) {
                    let dst_base = match base {
                        "llm.embed" => Some("embed_tokens"),
                        "llm.embed_norm" => Some("embed_norm"),
                        "llm.norm" => Some("norm"),
                        "llm.unembed" => Some("lm_head"),
                        _ => None,
                    };
                    if let Some(dst) = dst_base {
                        let out_sfx = canonical_st_quant_suffix(q_sfx);
                        let t = if q_sfx == ".weight" && bases_with_scales.contains(base) {
                            maybe_view_u8_to_u32(tensor)?
                        } else {
                            tensor
                        };
                        out.insert(format!("{dst}{out_sfx}"), t);
                        continue;
                    }
                }
                match key.as_str() {
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

                // 2. AttnRes queries (`attn_queries.{idx}` -> physical layer `idx + 1`)
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

                // 3. Per-layer Release weights (`llm.layers.{L}.*`)
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
                        "attn.q_l2_clip.scale" => {
                            out.insert(format!("{lp}.self_attn.q_clip.weight"), tensor);
                        }
                        "attn.rel_l2_clip.scale" => {
                            out.insert(format!("{lp}.self_attn.rel_clip.weight"), tensor);
                        }
                        "attn.rel_logits_proj.proj" => {
                            out.insert(format!("{lp}.self_attn.rel_bias.weight"), tensor);
                        }
                        "attn.k_sconv.weight" => {
                            let w = normalize_sconv_weight(tensor, &key)?;
                            out.insert(format!("{lp}.self_attn.k_conv.weight"), w);
                        }
                        "attn.v_sconv.weight" => {
                            let w = normalize_sconv_weight(tensor, &key)?;
                            out.insert(format!("{lp}.self_attn.v_conv.weight"), w);
                        }
                        "attn_sconv.weight" => {
                            let w = normalize_sconv_weight(tensor, &key)?;
                            out.insert(format!("{lp}.self_attn.post_conv.weight"), w);
                        }
                        "mlp_sconv.weight" => {
                            let w = normalize_sconv_weight(tensor, &key)?;
                            out.insert(format!("{lp}.mlp.post_conv.weight"), w);
                        }
                        "attn.k_sconv.static_w" => {
                            let w = normalize_dynamic_static_conv(tensor, &rev_k)?;
                            out.insert(format!("{lp}.self_attn.k_conv.weight"), w);
                        }
                        "attn.v_sconv.static_w" => {
                            let w = normalize_dynamic_static_conv(tensor, &rev_k)?;
                            out.insert(format!("{lp}.self_attn.v_conv.weight"), w);
                        }
                        "attn.k_sconv.weight_proj" => {
                            out.insert(format!("{lp}.self_attn.k_conv_proj.weight"), tensor);
                        }
                        "attn.v_sconv.weight_proj" => {
                            out.insert(format!("{lp}.self_attn.v_conv_proj.weight"), tensor);
                        }
                        "mlp.gate.bias" => {
                            out.insert(format!("{lp}.mlp.gate.e_score_correction_bias"), tensor);
                        }
                        "mlp.experts.w13_weight" => {
                            let w = if bases_with_scales.contains(&key) {
                                maybe_view_u8_to_u32(tensor)?
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
                            let w = if bases_with_scales.contains(&key) {
                                maybe_view_u8_to_u32(tensor)?
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
                            if let Some((base, q_sfx)) = split_st_quant_suffix(suffix) {
                                let mapped = match base {
                                    "attn_norm" => Some("input_layernorm"),
                                    "q_input_norm" => Some("self_attn.q_inp_norm"),
                                    "k_input_norm" => Some("self_attn.k_inp_norm"),
                                    "v_input_norm" => Some("self_attn.v_inp_norm"),
                                    "mlp_norm" => Some("post_attention_layernorm"),
                                    "attn.wq_du" => Some("self_attn.q_proj"),
                                    "attn.wk_dv" => Some("self_attn.k_proj"),
                                    "attn.wv_dv" => Some("self_attn.v_proj"),
                                    "attn.wr_du" => Some("self_attn.rel_proj"),
                                    "attn.gate_attn" => Some("self_attn.gate_proj"),
                                    "attn.attn_sink_proj" => Some("self_attn.sink_proj"),
                                    "attn.wo_ud" => Some("self_attn.o_proj"),
                                    "attn.k_sconv.weight_proj" => Some("self_attn.k_conv_proj"),
                                    "attn.v_sconv.weight_proj" => Some("self_attn.v_conv_proj"),
                                    "mlp.gate" => Some("mlp.gate"),
                                    _ => None,
                                };
                                if let Some(dst) = mapped {
                                    let raw_base = key.strip_suffix(q_sfx).unwrap_or(&key);
                                    let out_sfx = canonical_st_quant_suffix(q_sfx);
                                    let t = if q_sfx == ".weight"
                                        && bases_with_scales.contains(raw_base)
                                    {
                                        maybe_view_u8_to_u32(tensor)?
                                    } else {
                                        tensor
                                    };
                                    out.insert(format!("{lp}.{dst}{out_sfx}"), t);
                                    continue;
                                }
                            }
                            out.insert(key, tensor);
                        }
                    }
                    continue;
                }
                out.insert(key, tensor);
            }

            Inkling2CheckpointLayout::OMlx => {
                // 1. Top-level oMLX weights
                if let Some((base, q_sfx)) = split_st_quant_suffix(&key) {
                    match base {
                        "embed" | "embed_tokens" => {
                            out.insert(format!("embed_tokens{q_sfx}"), tensor);
                            continue;
                        }
                        "unembed" | "lm_head" => {
                            out.insert(format!("lm_head{q_sfx}"), tensor);
                            continue;
                        }
                        "embed_norm" => {
                            out.insert(format!("embed_norm{q_sfx}"), tensor);
                            continue;
                        }
                        "norm" => {
                            out.insert(format!("norm{q_sfx}"), tensor);
                            continue;
                        }
                        _ => {}
                    }
                }
                match key.as_str() {
                    "final_input.query" | "attn_res.output_query" => {
                        out.insert("output_res_query.weight".to_string(), tensor);
                        continue;
                    }
                    "message_queries" | "message_input.query" | "loop_message.query" => {
                        expand_global_loop_msg_queries(&mut out, tensor, config, &key)?;
                        continue;
                    }
                    _ => {}
                }

                // 2. Per-layer oMLX weights (`layers.{L}.*`)
                if let Some(rest) = key.strip_prefix("layers.")
                    && let Some((layer_str, suffix)) = rest.split_once('.')
                {
                    let layer_idx: usize = layer_str.parse().map_err(|e| {
                        Error::from_reason(format!(
                            "Inkling2 oMLX: invalid layer index '{layer_str}': {e}"
                        ))
                    })?;
                    let lp = format!("layers.{layer_idx}");

                    // Bare (non-quant-suffixed) exact names first so `attn.rel_proj` never collides.
                    match suffix {
                        "attn.q_scale" => {
                            out.insert(format!("{lp}.self_attn.q_clip.weight"), tensor);
                            continue;
                        }
                        "attn.r_scale" | "attn.rel_scale" => {
                            out.insert(format!("{lp}.self_attn.rel_clip.weight"), tensor);
                            continue;
                        }
                        "attn.rel_proj" => {
                            out.insert(format!("{lp}.self_attn.rel_bias.weight"), tensor);
                            continue;
                        }
                        "attn.k_conv" | "attn.k_conv.weight" => {
                            let w = normalize_sconv_weight(tensor, &key)?;
                            out.insert(format!("{lp}.self_attn.k_conv.weight"), w);
                            continue;
                        }
                        "attn.v_conv" | "attn.v_conv.weight" => {
                            let w = normalize_sconv_weight(tensor, &key)?;
                            out.insert(format!("{lp}.self_attn.v_conv.weight"), w);
                            continue;
                        }
                        "attn.k_static" | "attn.k_static.weight" => {
                            let w = normalize_dynamic_static_conv(tensor, &rev_k)?;
                            out.insert(format!("{lp}.self_attn.k_conv.weight"), w);
                            continue;
                        }
                        "attn.v_static" | "attn.v_static.weight" => {
                            let w = normalize_dynamic_static_conv(tensor, &rev_k)?;
                            out.insert(format!("{lp}.self_attn.v_conv.weight"), w);
                            continue;
                        }
                        "attn_conv" | "attn_conv.weight" => {
                            let w = normalize_sconv_weight(tensor, &key)?;
                            out.insert(format!("{lp}.self_attn.post_conv.weight"), w);
                            continue;
                        }
                        "mlp_conv" | "mlp_conv.weight" => {
                            let w = normalize_sconv_weight(tensor, &key)?;
                            out.insert(format!("{lp}.mlp.post_conv.weight"), w);
                            continue;
                        }
                        "attn_input.query" | "attention_input.query" => {
                            out.insert(format!("{lp}.attn_res_query.weight"), tensor);
                            continue;
                        }
                        "mlp_input.query" => {
                            out.insert(format!("{lp}.ffn_res_query.weight"), tensor);
                            continue;
                        }
                        "mlp.router_weight" => {
                            out.insert(format!("{lp}.mlp.gate.weight"), tensor);
                            continue;
                        }
                        "mlp.router_bias" | "mlp.gate.bias" => {
                            out.insert(format!("{lp}.mlp.gate.e_score_correction_bias"), tensor);
                            continue;
                        }
                        "loop_message.query" | "message_input.query" => {
                            out.insert(format!("{lp}.loop_msg_query.weight"), tensor);
                            continue;
                        }
                        "loop_down" | "loop_down.weight" => {
                            // Packed `[4, 2*D, rank]` -> per-stream `[rank, 2*D]`.
                            let reshaped = tensor.reshape(&[4, 2 * d, gate_rank])?;
                            for (g, stream) in [(0i64, "q"), (1, "k"), (2, "v"), (3, "mlp")] {
                                let w = reshaped
                                    .slice(&[g, 0, 0], &[g + 1, 2 * d, gate_rank])?
                                    .squeeze(Some(&[0]))?
                                    .transpose(None)?;
                                out.insert(format!("{lp}.loop_ctrl.{stream}.down_proj.weight"), w);
                            }
                            continue;
                        }
                        "loop_gain_forget" | "loop_gain_forget.weight" => {
                            // Packed `[4, rank, 2*D]` -> cols `[0..D]` gain^T, cols `[D..2*D]` forget^T.
                            let reshaped = tensor.reshape(&[4, gate_rank, 2 * d])?;
                            for (g, stream) in [(0i64, "q"), (1, "k"), (2, "v"), (3, "mlp")] {
                                let gf = reshaped
                                    .slice(&[g, 0, 0], &[g + 1, gate_rank, 2 * d])?
                                    .squeeze(Some(&[0]))?;
                                let gain = gf.slice(&[0, 0], &[gate_rank, d])?.transpose(None)?;
                                let forget =
                                    gf.slice(&[0, d], &[gate_rank, 2 * d])?.transpose(None)?;
                                out.insert(
                                    format!("{lp}.loop_ctrl.{stream}.gain_proj.weight"),
                                    gain,
                                );
                                out.insert(
                                    format!("{lp}.loop_ctrl.{stream}.forget_proj.weight"),
                                    forget,
                                );
                            }
                            continue;
                        }
                        _ => {}
                    }

                    if let Some((base, q_sfx)) = split_st_quant_suffix(suffix) {
                        let raw_base = key.strip_suffix(q_sfx).unwrap_or(&key);
                        let out_sfx = canonical_st_quant_suffix(q_sfx);
                        match base {
                            "attn.in_proj" => {
                                let t =
                                    if q_sfx == ".weight" && bases_with_scales.contains(raw_base) {
                                        maybe_view_u8_to_u32(tensor)?
                                    } else {
                                        tensor
                                    };
                                slice_fused_projection_group(
                                    &mut out,
                                    &lp,
                                    out_sfx,
                                    t,
                                    &[
                                        ("q_proj", q_dim),
                                        ("k_proj", kv_dim),
                                        ("v_proj", kv_dim),
                                        ("rel_proj", rel_dim),
                                        ("gate_proj", q_dim),
                                        ("sink_proj", n_heads),
                                    ],
                                    &key,
                                )?;
                                continue;
                            }
                            "attn.q_proj" if omlx_fused_q.contains(&layer_idx) => {
                                let t =
                                    if q_sfx == ".weight" && bases_with_scales.contains(raw_base) {
                                        maybe_view_u8_to_u32(tensor)?
                                    } else {
                                        tensor
                                    };
                                slice_fused_projection_group(
                                    &mut out,
                                    &lp,
                                    out_sfx,
                                    t,
                                    &[
                                        ("q_proj", q_dim),
                                        ("rel_proj", rel_dim),
                                        ("gate_proj", q_dim),
                                        ("sink_proj", n_heads),
                                    ],
                                    &key,
                                )?;
                                continue;
                            }
                            "attn.k_proj" if omlx_fused_k.contains(&layer_idx) => {
                                let t =
                                    if q_sfx == ".weight" && bases_with_scales.contains(raw_base) {
                                        maybe_view_u8_to_u32(tensor)?
                                    } else {
                                        tensor
                                    };
                                slice_fused_projection_group(
                                    &mut out,
                                    &lp,
                                    out_sfx,
                                    t,
                                    &[("k_proj", kv_dim), ("k_conv_proj", taps)],
                                    &key,
                                )?;
                                continue;
                            }
                            "attn.v_proj" if omlx_fused_v.contains(&layer_idx) => {
                                let t =
                                    if q_sfx == ".weight" && bases_with_scales.contains(raw_base) {
                                        maybe_view_u8_to_u32(tensor)?
                                    } else {
                                        tensor
                                    };
                                slice_fused_projection_group(
                                    &mut out,
                                    &lp,
                                    out_sfx,
                                    t,
                                    &[("v_proj", kv_dim), ("v_conv_proj", taps)],
                                    &key,
                                )?;
                                continue;
                            }
                            "mlp.experts.w13" => {
                                let raw_base = key.strip_suffix(q_sfx).unwrap_or(&key);
                                let out_sfx = canonical_st_quant_suffix(q_sfx);
                                let t =
                                    if q_sfx == ".weight" && bases_with_scales.contains(raw_base) {
                                        maybe_view_u8_to_u32(tensor)?
                                    } else {
                                        tensor
                                    };
                                out.insert(format!("{lp}.mlp.experts.gate_up_proj{out_sfx}"), t);
                                continue;
                            }
                            "mlp.experts.w2" => {
                                let raw_base = key.strip_suffix(q_sfx).unwrap_or(&key);
                                let out_sfx = canonical_st_quant_suffix(q_sfx);
                                let t =
                                    if q_sfx == ".weight" && bases_with_scales.contains(raw_base) {
                                        maybe_view_u8_to_u32(tensor)?
                                    } else {
                                        tensor
                                    };
                                out.insert(format!("{lp}.mlp.experts.down_proj{out_sfx}"), t);
                                continue;
                            }
                            _ => {
                                let mapped = match base {
                                    "attn_norm" | "input_layernorm" => Some("input_layernorm"),
                                    "q_input_norm"
                                    | "q_input_layernorm"
                                    | "attn.q_input_norm"
                                    | "self_attn.q_inp_norm" => Some("self_attn.q_inp_norm"),
                                    "k_input_norm"
                                    | "k_input_layernorm"
                                    | "attn.k_input_norm"
                                    | "self_attn.k_inp_norm" => Some("self_attn.k_inp_norm"),
                                    "v_input_norm"
                                    | "v_input_layernorm"
                                    | "attn.v_input_norm"
                                    | "self_attn.v_inp_norm" => Some("self_attn.v_inp_norm"),
                                    "mlp_norm" | "post_attention_layernorm" => {
                                        Some("post_attention_layernorm")
                                    }
                                    "attn.q_proj" => Some("self_attn.q_proj"),
                                    "attn.k_proj" => Some("self_attn.k_proj"),
                                    "attn.v_proj" => Some("self_attn.v_proj"),
                                    "attn.r_proj" => Some("self_attn.rel_proj"),
                                    "attn.gate_proj" => Some("self_attn.gate_proj"),
                                    "attn.sink_proj" => Some("self_attn.sink_proj"),
                                    "attn.o_proj" => Some("self_attn.o_proj"),
                                    "mlp.router" | "mlp.gate" => Some("mlp.gate"),
                                    _ => None,
                                };
                                if let Some(dst) = mapped {
                                    let raw_base = key.strip_suffix(q_sfx).unwrap_or(&key);
                                    let out_sfx = canonical_st_quant_suffix(q_sfx);
                                    let t = if q_sfx == ".weight"
                                        && bases_with_scales.contains(raw_base)
                                    {
                                        maybe_view_u8_to_u32(tensor)?
                                    } else {
                                        tensor
                                    };
                                    out.insert(format!("{lp}.{dst}{out_sfx}"), t);
                                    continue;
                                }
                            }
                        }
                    }
                    out.insert(key, tensor);
                    continue;
                }
                out.insert(key, tensor);
            }

            Inkling2CheckpointLayout::MlxVlm | Inkling2CheckpointLayout::MlxLm => {
                // 1. Top-level mlx-vlm / mlx-lm weights
                match key.as_str() {
                    "final_input.query" | "attn_res.output_query" => {
                        out.insert("output_res_query.weight".to_string(), tensor);
                        continue;
                    }
                    "message_queries" | "message_input.query" | "loop_message.query" => {
                        expand_global_loop_msg_queries(&mut out, tensor, config, &key)?;
                        continue;
                    }
                    _ => {}
                }

                // 2. Per-layer mlx-vlm / mlx-lm weights (`layers.{L}.*`)
                if let Some(rest) = key.strip_prefix("layers.")
                    && let Some((layer_str, suffix)) = rest.split_once('.')
                {
                    let layer_idx: usize = layer_str.parse().map_err(|e| {
                        Error::from_reason(format!(
                            "Inkling2 MLX: invalid layer index '{layer_str}': {e}"
                        ))
                    })?;
                    let lp = format!("layers.{layer_idx}");

                    // Bare / exact keys first (guards `self_attn.rel_proj` vs `self_attn.rel_proj.*` collision!)
                    match suffix {
                        "self_attn.rel_proj" | "self_attn.rel_logits_proj.proj" => {
                            out.insert(format!("{lp}.self_attn.rel_bias.weight"), tensor);
                            continue;
                        }
                        "self_attn.q_scale" | "self_attn.q_l2_clip.scale" => {
                            out.insert(format!("{lp}.self_attn.q_clip.weight"), tensor);
                            continue;
                        }
                        "self_attn.rel_scale"
                        | "self_attn.r_scale"
                        | "self_attn.rel_l2_clip.scale" => {
                            out.insert(format!("{lp}.self_attn.rel_clip.weight"), tensor);
                            continue;
                        }
                        "self_attn.k_sconv.conv1d.weight"
                        | "self_attn.k_sconv.conv.weight"
                        | "self_attn.k_sconv.weight" => {
                            let w = normalize_sconv_weight(tensor, &key)?;
                            out.insert(format!("{lp}.self_attn.k_conv.weight"), w);
                            continue;
                        }
                        "self_attn.v_sconv.conv1d.weight"
                        | "self_attn.v_sconv.conv.weight"
                        | "self_attn.v_sconv.weight" => {
                            let w = normalize_sconv_weight(tensor, &key)?;
                            out.insert(format!("{lp}.self_attn.v_conv.weight"), w);
                            continue;
                        }
                        "self_attn.k_sconv.static_weight" | "self_attn.k_sconv.static_w" => {
                            let w = normalize_dynamic_static_conv(tensor, &rev_k)?;
                            out.insert(format!("{lp}.self_attn.k_conv.weight"), w);
                            continue;
                        }
                        "self_attn.v_sconv.static_weight" | "self_attn.v_sconv.static_w" => {
                            let w = normalize_dynamic_static_conv(tensor, &rev_k)?;
                            out.insert(format!("{lp}.self_attn.v_conv.weight"), w);
                            continue;
                        }
                        "self_attn.k_sconv.weight_proj" => {
                            out.insert(format!("{lp}.self_attn.k_conv_proj.weight"), tensor);
                            continue;
                        }
                        "self_attn.v_sconv.weight_proj" => {
                            out.insert(format!("{lp}.self_attn.v_conv_proj.weight"), tensor);
                            continue;
                        }
                        "post_attention_convolution.conv1d.weight"
                        | "attn_sconv.conv.weight"
                        | "attn_sconv.weight" => {
                            let w = normalize_sconv_weight(tensor, &key)?;
                            out.insert(format!("{lp}.self_attn.post_conv.weight"), w);
                            continue;
                        }
                        "post_mlp_convolution.conv1d.weight"
                        | "mlp_sconv.conv.weight"
                        | "mlp_sconv.weight" => {
                            let w = normalize_sconv_weight(tensor, &key)?;
                            out.insert(format!("{lp}.mlp.post_conv.weight"), w);
                            continue;
                        }
                        "attention_input.query" | "attn_input.query" => {
                            out.insert(format!("{lp}.attn_res_query.weight"), tensor);
                            continue;
                        }
                        "mlp_input.query" => {
                            out.insert(format!("{lp}.ffn_res_query.weight"), tensor);
                            continue;
                        }
                        "message_input.query" => {
                            out.insert(format!("{lp}.loop_msg_query.weight"), tensor);
                            continue;
                        }
                        "mlp.gate_weight" => {
                            out.insert(format!("{lp}.mlp.gate.weight"), tensor);
                            continue;
                        }
                        "mlp.e_score_correction_bias"
                        | "mlp.gate.e_score_correction_bias"
                        | "mlp.gate.bias" => {
                            out.insert(format!("{lp}.mlp.gate.e_score_correction_bias"), tensor);
                            continue;
                        }
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
                                            "Inkling2 MLX: unknown controller stream '{other}' in '{key}'"
                                        )));
                                    }
                                };
                                let proj = match proj_raw {
                                    "down_weight" => "down_proj.weight",
                                    "gain_weight" => "gain_proj.weight",
                                    "forget_weight" => "forget_proj.weight",
                                    other => {
                                        return Err(Error::from_reason(format!(
                                            "Inkling2 MLX: unknown controller weight '{other}' in '{key}'"
                                        )));
                                    }
                                };
                                out.insert(format!("{lp}.loop_ctrl.{stream}.{proj}"), tensor);
                                continue;
                            }
                        }
                        _ => {}
                    }

                    if let Some((base, q_sfx)) = split_st_quant_suffix(suffix) {
                        // Message gate projections (`{q,k,v,mlp}_message_gate.*`)
                        if let Some((gate_prefix, gate_proj)) = base.split_once("_message_gate.")
                            && matches!(gate_prefix, "q" | "k" | "v" | "mlp")
                        {
                            let mapped_proj = match gate_proj {
                                "down_proj" => Some("down_proj"),
                                "message_gate_proj" => Some("gain_proj"),
                                "forget_gate_proj" => Some("forget_proj"),
                                _ => None,
                            };
                            if let Some(p) = mapped_proj {
                                out.insert(
                                    format!("{lp}.loop_ctrl.{gate_prefix}.{p}{q_sfx}"),
                                    tensor,
                                );
                                continue;
                            }
                        }

                        // Experts (`mlp.experts.{gate_up_proj,down_proj}` or `mlp.switch_mlp.{gate,up,down}_proj`)
                        let expert_dst = match base {
                            "mlp.experts.gate_up_proj" => Some("mlp.experts.gate_up_proj"),
                            "mlp.experts.down_proj" | "mlp.switch_mlp.down_proj" => {
                                Some("mlp.experts.down_proj")
                            }
                            "mlp.switch_mlp.gate_proj" => Some("mlp.experts.gate_proj"),
                            "mlp.switch_mlp.up_proj" => Some("mlp.experts.up_proj"),
                            _ => None,
                        };
                        if let Some(dst) = expert_dst {
                            let raw_base = key.strip_suffix(q_sfx).unwrap_or(&key);
                            let out_sfx = canonical_st_quant_suffix(q_sfx);
                            let t = if q_sfx == ".weight" && bases_with_scales.contains(raw_base) {
                                maybe_view_u8_to_u32(tensor)?
                            } else {
                                tensor
                            };
                            out.insert(format!("{lp}.{dst}{out_sfx}"), t);
                            continue;
                        }

                        // Norms & attention projections
                        let mapped = match base {
                            "q_input_layernorm"
                            | "q_input_norm"
                            | "self_attn.q_input_layernorm"
                            | "self_attn.q_inp_norm" => Some("self_attn.q_inp_norm"),
                            "k_input_layernorm"
                            | "k_input_norm"
                            | "self_attn.k_input_layernorm"
                            | "self_attn.k_inp_norm" => Some("self_attn.k_inp_norm"),
                            "v_input_layernorm"
                            | "v_input_norm"
                            | "self_attn.v_input_layernorm"
                            | "self_attn.v_inp_norm" => Some("self_attn.v_inp_norm"),
                            "self_attn.r_proj" => Some("self_attn.rel_proj"),
                            "self_attn.k_sconv.weight_proj" => Some("self_attn.k_conv_proj"),
                            "self_attn.v_sconv.weight_proj" => Some("self_attn.v_conv_proj"),
                            _ => None,
                        };
                        if let Some(dst) = mapped {
                            let raw_base = key.strip_suffix(q_sfx).unwrap_or(&key);
                            let out_sfx = canonical_st_quant_suffix(q_sfx);
                            let t = if q_sfx == ".weight" && bases_with_scales.contains(raw_base) {
                                maybe_view_u8_to_u32(tensor)?
                            } else {
                                tensor
                            };
                            out.insert(format!("{lp}.{dst}{out_sfx}"), t);
                            continue;
                        }
                    }

                    out.insert(key, tensor);
                    continue;
                }

                out.insert(key, tensor);
            }

            Inkling2CheckpointLayout::Canonical => unreachable!(),
        }
    }

    // Materialize per-expert Release NVFP4 `outer_scale = amax / outer_scale_denominator` (`[E]` F32).
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
    if let Some(((layer_idx, proj_name), _)) = expert_denom.into_iter().next() {
        return Err(Error::from_reason(format!(
            "Inkling2 ST: orphaned outer_scale_denominator without matching weight_amax for layers.{layer_idx}.mlp.experts.{proj_name}"
        )));
    }

    let has_split_experts = out.keys().any(|k| {
        k.ends_with(".mlp.experts.gate_proj.weight") || k.ends_with(".mlp.experts.gate_proj.scales")
    });
    let has_fused_experts = out.keys().any(|k| {
        k.ends_with(".mlp.experts.gate_up_proj.weight")
            || k.ends_with(".mlp.experts.gate_up_proj.scales")
    });
    *params = out;
    let gate_up_interleaved = match layout {
        Inkling2CheckpointLayout::Release | Inkling2CheckpointLayout::OMlx => Some(true),
        Inkling2CheckpointLayout::MlxVlm => {
            // Preserve interleaved order for any stored fused layers; split layers always
            // force `effective_gate_up_interleaved = false` per-layer inside `load_expert_gate_up`.
            if has_fused_experts {
                Some(true)
            } else if has_split_experts {
                Some(false)
            } else {
                None
            }
        }
        Inkling2CheckpointLayout::MlxLm => {
            // Only force non-interleaved when all layers are split; if fused layers coexist,
            // leave `config.gate_up_interleaved` untouched while split layers force `false` per-layer.
            if has_split_experts && !has_fused_experts {
                Some(false)
            } else {
                None
            }
        }
        Inkling2CheckpointLayout::Canonical => None,
    };
    Ok(Inkling2StNormResult {
        gate_up_interleaved,
        omlx_fused_in_proj,
        omlx_fused_q,
        omlx_fused_k,
        omlx_fused_v,
    })
}

/// Infer 2-D NVFP4 / MXFP8 / MXFP4 per-layer overrides from validated storage geometry
/// (`Uint32` `.weight` + `Uint8` `.scales` + no `.biases`) for checkpoints whose config carries
/// no MLX `quantization` entries (e.g. Release ModelOpt NVFP4 SafeTensors).
fn infer_unregistered_2d_fp_quant_overrides(
    params: &HashMap<String, MxArray>,
    per_layer_quant: &mut HashMap<String, PerLayerQuant>,
) -> Result<()> {
    for (key, scales) in params {
        let Some(base) = key.strip_suffix(".scales") else {
            continue;
        };
        let norm_key = crate::models::quant_dispatch::normalize_per_layer_key(base);
        if per_layer_quant.contains_key(base) || per_layer_quant.contains_key(&norm_key) {
            continue;
        }
        if scales.dtype()? != DType::Uint8 || params.contains_key(&format!("{base}.biases")) {
            continue;
        }
        let Some(w) = params.get(&format!("{base}.weight")) else {
            continue;
        };
        if w.dtype()? != DType::Uint32 {
            continue;
        }
        let w_shape = w.shape()?;
        let s_shape = scales.shape()?;
        if w_shape.len() != 2 || s_shape.len() != 2 || w_shape[0] != s_shape[0] {
            continue;
        }
        let inferred = if w_shape[1] == s_shape[1] * 2 {
            Some(PerLayerQuant {
                bits: NVFP4_BITS,
                group_size: NVFP4_GROUP_SIZE,
                mode: PerLayerMode::Nvfp4,
                layout: Default::default(),
                input_amax: None,
            })
        } else if w_shape[1] == s_shape[1] * 8 {
            Some(PerLayerQuant {
                bits: MXFP8_BITS,
                group_size: MXFP8_GROUP_SIZE,
                mode: PerLayerMode::Mxfp8,
                layout: Default::default(),
                input_amax: None,
            })
        } else if w_shape[1] == s_shape[1] * 4 {
            Some(PerLayerQuant {
                bits: MXFP4_BITS,
                group_size: MXFP4_GROUP_SIZE,
                mode: PerLayerMode::Mxfp4,
                layout: Default::default(),
                input_amax: None,
            })
        } else {
            None
        };
        if let Some(plq) = inferred {
            per_layer_quant.insert(norm_key, plq);
        }
    }
    Ok(())
}

/// Normalize SafeTensors vision tower keys independently of text layout across all 4
/// verified Inkling2 SafeTensors layouts (matching Splash `InklingVision.cpp:139-186`):
/// 1. Transformers / MLX (`model.vision_tower.*` -> `vision_tower.*` + `merger.linear_{1,2}`)
/// 2. Release HF (`model.visual.backbone.*` -> `visual.backbone.*` + `visual.proj_{in,out}`)
/// 3. oMLX (`vision_tower.backbone.*` + `vision_tower.proj_{in,out}`)
/// 4. thoddnn NVFP4 (`vision_tower.*` + `vision_tower.proj_{in,out}`)
///
/// Transposes 2-D flattened HWC `embeddings.patch_embedding.weight` (`[D, P*P*3]`) to CHW once
/// (leaving 4-D OIHW/OHWI or already-canonical `vision_tower.patch_embed.weight` untouched),
/// and rejects quantized vision weights (`.scales` / `.weight_scale`) so `load_weights_sync`
/// cleanly falls back to text-only mode.
fn normalize_inkling2_vision_params(
    params: &mut HashMap<String, MxArray>,
    config: &Inkling2Config,
) -> Result<()> {
    // Reject quantized vision weights before any remap so fallback engages cleanly.
    if let Some(qkey) = params.keys().find(|k| {
        (k.starts_with("visual.")
            || k.starts_with("vision_tower.")
            || k.starts_with("vision_projector."))
            && (k.ends_with(".scales") || k.ends_with(".weight_scale"))
    }) {
        return Err(Error::from_reason(format!(
            "vision tower is quantized (found '{qkey}'); quantized vision weights are not supported"
        )));
    }

    // Probe the 4 SafeTensors vision layouts after `strip_inkling2_wrapper_prefixes`:
    // `(owner_prefix, backbone_prefix)`.
    let detected = if params.contains_key("visual.backbone.embeddings.patch_embedding.weight") {
        // Layout 2: Release HF (`model.visual.backbone.*` + `model.visual.proj_{in,out}.*`)
        Some(("visual.", "visual.backbone."))
    } else if params.contains_key("vision_tower.backbone.embeddings.patch_embedding.weight") {
        // Layout 3: oMLX (`vision_tower.backbone.*` + `vision_tower.proj_{in,out}.*`)
        Some(("vision_tower.", "vision_tower.backbone."))
    } else if params.contains_key("vision_tower.embeddings.patch_embedding.weight") {
        // Layout 1 (Transformers / MLX: `vision_tower.*` + `vision_tower.merger.linear_{1,2}.*`)
        // & Layout 4 (thoddnn NVFP4: `vision_tower.*` + `vision_tower.proj_{in,out}.*`)
        Some(("vision_tower.", "vision_tower."))
    } else {
        None
    };

    let Some((owner_prefix, backbone_prefix)) = detected else {
        return Ok(());
    };

    let vision_patch = config
        .vision_config
        .as_ref()
        .map_or(20i64, |v| v.patch_size as i64);

    let keys_to_move: Vec<String> = params
        .keys()
        .filter(|k| k.starts_with(owner_prefix))
        .cloned()
        .collect();
    let mut moved: Vec<(String, MxArray)> = Vec::with_capacity(keys_to_move.len());
    for key in keys_to_move {
        if let Some(tensor) = params.remove(&key) {
            moved.push((key, tensor));
        }
    }

    for (key, tensor) in moved {
        // 1. Backbone tensors (`{backbone_prefix}*`)
        if let Some(brest) = key.strip_prefix(backbone_prefix) {
            match brest {
                "embeddings.class_embedding" => {
                    params.insert("vision_tower.class_embedding".to_string(), tensor);
                    continue;
                }
                "embeddings.cls_position_embedding" => {
                    params.insert(
                        "vision_tower.class_position_embedding.weight".to_string(),
                        tensor,
                    );
                    continue;
                }
                "embeddings.position_embedding" | "embeddings.position_embedding.weight" => {
                    params.insert("vision_tower.position_embedding.weight".to_string(), tensor);
                    continue;
                }
                "embeddings.patch_embedding.weight" => {
                    // All 4 SafeTensors layouts store 2-D `[D, P*P*3]` in HWC `(P, P, 3)` order;
                    // `Inkling2VisionEncoder` expects CHW `(3, P, P)` flattened `[D, 3*P*P]`.
                    // 4-D OIHW/OHWI weights are left untouched for `build_vision_encoder`.
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
                    params.insert("vision_tower.patch_embed.weight".to_string(), converted);
                    continue;
                }
                "ln_pre.weight" | "pre_layernorm.weight" => {
                    params.insert("vision_tower.pre_ln.weight".to_string(), tensor);
                    continue;
                }
                "ln_pre.bias" | "pre_layernorm.bias" => {
                    params.insert("vision_tower.pre_ln.bias".to_string(), tensor);
                    continue;
                }
                _ if brest.starts_with("layers.") => {
                    if let Some(lrest) = brest.strip_prefix("layers.")
                        && let Some((blk_str, sub)) = lrest.split_once('.')
                    {
                        let bp = format!("vision_tower.blocks.{blk_str}");
                        let mapped = match sub {
                            "ln_1.weight" | "layernorm_before.weight" => Some("norm1.weight"),
                            "ln_1.bias" | "layernorm_before.bias" => Some("norm1.bias"),
                            "ln_2.weight" | "layernorm_after.weight" => Some("norm2.weight"),
                            "ln_2.bias" | "layernorm_after.bias" => Some("norm2.bias"),
                            "ls_1.gamma" | "lambda_1" | "lambda_1.weight" => Some("ls1.weight"),
                            "ls_2.gamma" | "lambda_2" | "lambda_2.weight" => Some("ls2.weight"),
                            "self_attn.q_proj.weight" => Some("attn.q_proj.weight"),
                            "self_attn.q_proj.bias" => Some("attn.q_proj.bias"),
                            "self_attn.k_proj.weight" => Some("attn.k_proj.weight"),
                            "self_attn.k_proj.bias" => Some("attn.k_proj.bias"),
                            "self_attn.v_proj.weight" => Some("attn.v_proj.weight"),
                            "self_attn.v_proj.bias" => Some("attn.v_proj.bias"),
                            "self_attn.out_proj.weight" => Some("attn.o_proj.weight"),
                            "self_attn.out_proj.bias" => Some("attn.o_proj.bias"),
                            "mlp.c_fc.weight" | "mlp.fc1.weight" => Some("mlp.fc1.weight"),
                            "mlp.c_fc.bias" | "mlp.fc1.bias" => Some("mlp.fc1.bias"),
                            "mlp.c_proj.weight" | "mlp.fc2.weight" => Some("mlp.fc2.weight"),
                            "mlp.c_proj.bias" | "mlp.fc2.bias" => Some("mlp.fc2.bias"),
                            _ => None,
                        };
                        if let Some(m) = mapped {
                            params.insert(format!("{bp}.{m}"), tensor);
                            continue;
                        }
                    }
                }
                _ => {}
            }
        }

        // 2. Projector tensors (`{owner_prefix}proj_{in,out}.*` or `{owner_prefix}merger.linear_{1,2}.*`)
        if let Some(orest) = key.strip_prefix(owner_prefix) {
            let proj_dst = match orest {
                "proj_in.weight" | "merger.linear_1.weight" => Some("vision_projector.fc1.weight"),
                "proj_in.bias" | "merger.linear_1.bias" => Some("vision_projector.fc1.bias"),
                "proj_out.weight" | "merger.linear_2.weight" => Some("vision_projector.fc2.weight"),
                "proj_out.bias" | "merger.linear_2.bias" => Some("vision_projector.fc2.bias"),
                _ => None,
            };
            if let Some(dst) = proj_dst {
                params.insert(dst.to_string(), tensor);
                continue;
            }
        }

        params.insert(key, tensor);
    }
    Ok(())
}

/// Remap converter-specific `per_layer_quant` keys into the canonical Inkling2 namespace,
/// fanning out oMLX fused `attn.in_proj` / `attn.{q,k,v}_proj` overrides using the exact
/// shape-derived split plan from `norm`.
fn remap_inkling2_per_layer_quant(
    per_layer_quant: &mut HashMap<String, PerLayerQuant>,
    layout: Inkling2CheckpointLayout,
    norm: &Inkling2StNormResult,
    config: &Inkling2Config,
) -> Result<()> {
    if per_layer_quant.is_empty() || layout == Inkling2CheckpointLayout::Canonical {
        return Ok(());
    }
    let n_phys = config.num_physical_layers();
    let old = std::mem::take(per_layer_quant);
    let mut remapped: HashMap<String, PerLayerQuant> = HashMap::with_capacity(old.len() * 2);

    for (raw_key, plq) in old {
        // Preserve already-normalized key as fallback, then add canonical target(s).
        let stripped = raw_key
            .strip_prefix("language_model.")
            .or_else(|| raw_key.strip_prefix("llm."))
            .unwrap_or(&raw_key);

        match stripped {
            "embed" => {
                remapped.insert("embed_tokens".to_string(), plq);
                continue;
            }
            "unembed" => {
                remapped.insert("lm_head".to_string(), plq);
                continue;
            }
            _ => {}
        }

        if let Some(rest) = stripped.strip_prefix("layers.")
            && let Some((layer_str, sub)) = rest.split_once('.')
            && let Ok(layer_idx) = layer_str.parse::<usize>()
        {
            if layer_idx >= n_phys {
                return Err(Error::from_reason(format!(
                    "Inkling2: quantization override '{raw_key}' references out-of-range physical layer {layer_idx} (num_physical_layers={n_phys})"
                )));
            }
            let lp = format!("layers.{layer_idx}");
            match sub {
                // oMLX fused attention projections -> fan out using exact shape-derived split plan
                "attn.in_proj"
                    if layout == Inkling2CheckpointLayout::OMlx
                        && norm.omlx_fused_in_proj.contains(&layer_idx) =>
                {
                    for dst in [
                        "self_attn.q_proj",
                        "self_attn.k_proj",
                        "self_attn.v_proj",
                        "self_attn.rel_proj",
                        "self_attn.gate_proj",
                        "self_attn.sink_proj",
                    ] {
                        remapped.entry(format!("{lp}.{dst}")).or_insert(plq);
                    }
                    continue;
                }
                // Ignore stale `attn.in_proj` metadata when no fused `attn.in_proj` tensor exists
                // so independent projections retain their own overrides / file-wide defaults.
                "attn.in_proj" => {
                    continue;
                }
                "attn.q_proj"
                    if layout == Inkling2CheckpointLayout::OMlx
                        && norm.omlx_fused_q.contains(&layer_idx) =>
                {
                    for dst in [
                        "self_attn.q_proj",
                        "self_attn.rel_proj",
                        "self_attn.gate_proj",
                        "self_attn.sink_proj",
                    ] {
                        remapped.entry(format!("{lp}.{dst}")).or_insert(plq);
                    }
                    continue;
                }
                "attn.k_proj"
                    if layout == Inkling2CheckpointLayout::OMlx
                        && norm.omlx_fused_k.contains(&layer_idx) =>
                {
                    for dst in ["self_attn.k_proj", "self_attn.k_conv_proj"] {
                        remapped.entry(format!("{lp}.{dst}")).or_insert(plq);
                    }
                    continue;
                }
                "attn.v_proj"
                    if layout == Inkling2CheckpointLayout::OMlx
                        && norm.omlx_fused_v.contains(&layer_idx) =>
                {
                    for dst in ["self_attn.v_proj", "self_attn.v_conv_proj"] {
                        remapped.entry(format!("{lp}.{dst}")).or_insert(plq);
                    }
                    continue;
                }
                // Release / oMLX / mlx-vlm / mlx-lm single-target aliases (including renamed norms!)
                "attn_norm" => {
                    remapped.insert(format!("{lp}.input_layernorm"), plq);
                    continue;
                }
                "mlp_norm" => {
                    remapped.insert(format!("{lp}.post_attention_layernorm"), plq);
                    continue;
                }
                "q_input_norm"
                | "q_input_layernorm"
                | "attn.q_input_norm"
                | "self_attn.q_input_layernorm" => {
                    remapped.insert(format!("{lp}.self_attn.q_inp_norm"), plq);
                    continue;
                }
                "k_input_norm"
                | "k_input_layernorm"
                | "attn.k_input_norm"
                | "self_attn.k_input_layernorm" => {
                    remapped.insert(format!("{lp}.self_attn.k_inp_norm"), plq);
                    continue;
                }
                "v_input_norm"
                | "v_input_layernorm"
                | "attn.v_input_norm"
                | "self_attn.v_input_layernorm" => {
                    remapped.insert(format!("{lp}.self_attn.v_inp_norm"), plq);
                    continue;
                }
                "attn.wq_du" | "attn.q_proj" => {
                    remapped.insert(format!("{lp}.self_attn.q_proj"), plq);
                    continue;
                }
                "attn.wk_dv" | "attn.k_proj" => {
                    remapped.insert(format!("{lp}.self_attn.k_proj"), plq);
                    continue;
                }
                "attn.wv_dv" | "attn.v_proj" => {
                    remapped.insert(format!("{lp}.self_attn.v_proj"), plq);
                    continue;
                }
                "attn.wr_du" | "attn.r_proj" | "self_attn.r_proj" => {
                    remapped.insert(format!("{lp}.self_attn.rel_proj"), plq);
                    continue;
                }
                "attn.gate_attn" | "attn.gate_proj" => {
                    remapped.insert(format!("{lp}.self_attn.gate_proj"), plq);
                    continue;
                }
                "attn.attn_sink_proj" | "attn.sink_proj" => {
                    remapped.insert(format!("{lp}.self_attn.sink_proj"), plq);
                    continue;
                }
                "attn.wo_ud" | "attn.o_proj" => {
                    remapped.insert(format!("{lp}.self_attn.o_proj"), plq);
                    continue;
                }
                "attn.k_sconv.weight_proj" | "self_attn.k_sconv.weight_proj" => {
                    remapped.insert(format!("{lp}.self_attn.k_conv_proj"), plq);
                    continue;
                }
                "attn.v_sconv.weight_proj" | "self_attn.v_sconv.weight_proj" => {
                    remapped.insert(format!("{lp}.self_attn.v_conv_proj"), plq);
                    continue;
                }
                "mlp.router" => {
                    remapped.insert(format!("{lp}.mlp.gate"), plq);
                    continue;
                }
                "mlp.experts.w13" => {
                    remapped.insert(format!("{lp}.mlp.experts.gate_up_proj"), plq);
                    continue;
                }
                "mlp.experts.w2" | "mlp.switch_mlp.down_proj" => {
                    remapped.insert(format!("{lp}.mlp.experts.down_proj"), plq);
                    continue;
                }
                "mlp.switch_mlp.gate_proj" => {
                    remapped.insert(format!("{lp}.mlp.experts.gate_proj"), plq);
                    continue;
                }
                "mlp.switch_mlp.up_proj" => {
                    remapped.insert(format!("{lp}.mlp.experts.up_proj"), plq);
                    continue;
                }
                _ => {
                    if let Some((gate_prefix, gate_proj)) = sub.split_once("_message_gate.")
                        && matches!(gate_prefix, "q" | "k" | "v" | "mlp")
                    {
                        let mapped_proj = match gate_proj {
                            "down_proj" => Some("down_proj"),
                            "message_gate_proj" => Some("gain_proj"),
                            "forget_gate_proj" => Some("forget_proj"),
                            _ => None,
                        };
                        if let Some(p) = mapped_proj {
                            remapped.insert(format!("{lp}.loop_ctrl.{gate_prefix}.{p}"), plq);
                            continue;
                        }
                    }
                }
            }
        }

        remapped.insert(stripped.to_string(), plq);
    }

    *per_layer_quant = remapped;
    Ok(())
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

    #[test]
    fn materialize_and_promote_weights_drops_host_clones_and_promotes_to_device() {
        let hidden = 16usize;
        let vocab = 32usize;
        let cfg_raw = serde_json::json!({
            "model_type": "inkling2",
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_physical_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": vocab,
            "sliding_window": 16,
            "n_layers_per_attn_res_block": 1
        })
        .to_string();
        let config = Inkling2Config::from_json_str(&cfg_raw, true).unwrap();

        let (embed_w, _, _) =
            MxArray::alloc_uninit(&[vocab as i64, hidden as i64], DType::BFloat16).unwrap();
        let (embed_norm, _, _) = MxArray::alloc_uninit(&[hidden as i64], DType::Float32).unwrap();
        let (output_res_query, _, _) =
            MxArray::alloc_uninit(&[1, hidden as i64], DType::Float32).unwrap();
        let output_res_query_t = output_res_query.transpose(None).unwrap();
        let (output_norm, _, _) = MxArray::alloc_uninit(&[hidden as i64], DType::Float32).unwrap();
        let (lm_w, _, _) =
            MxArray::alloc_uninit(&[vocab as i64, hidden as i64], DType::BFloat16).unwrap();

        let mut inner = Inkling2Inner {
            config,
            embed_tokens: Embedding::from_weight(&embed_w).unwrap(),
            embed_norm,
            layers: Vec::new(),
            output_res_query,
            output_res_query_t,
            output_norm,
            lm_head: LinearProj::Standard(Linear::from_weights(&lm_w, None).unwrap()),
            lm_head_bf16_unpadded: None,
            vision_encoder: None,
            tokenizer: None,
            dspark: None,
            dflash2: None,
            dflash2_identity: None,
            dspark_turn_contexts: HashMap::new(),
            cuda_dflash2_ctx: None,
            cuda_verify_budget: crate::models::qwen3_5::verify_tree::VerifyBudget::new(),
            cuda_lookup_acceptance: crate::models::qwen3_5::verify_tree::LookupAcceptance::new(),
            cuda_dspark_ctx: None,
            scheduled_dspark: Inkling2ScheduledDsparkStates::default(),
            paged: None,
            flat_cache: None,
            cuda_cache: None,
            cuda_kv_format: super::super::cuda_kv::Inkling2CudaKvFormat::Bf16,
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
        drop(embed_w);
        drop(lm_w);

        let resident = materialize_and_promote_weights(&mut inner).unwrap();
        assert!(resident.arrays() > 0);
        assert!(resident.bytes() > 0);
        if crate::accelerator::cuda_ops_available() {
            assert_eq!(inner.embed_norm.cuda_host_bytes(), 0);
            assert_eq!(inner.output_res_query.cuda_host_bytes(), 0);
            assert_eq!(inner.output_norm.cuda_host_bytes(), 0);
            assert_eq!(inner.lm_head.get_weight().cuda_host_bytes(), 0);
        }
    }

    struct TempDirGuard(PathBuf);

    impl TempDirGuard {
        fn new(tag: &str) -> Self {
            static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "mlx-inkling2-draft-test-{tag}-{}-{id}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("create temp test dir");
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn build_synthetic_draft_target_inner() -> Inkling2Inner {
        let d = 256usize;
        let vocab = 64usize;
        let unpadded_vocab = 60usize;
        let config = Inkling2Config {
            model_type: "inkling2".to_string(),
            hidden_size: d,
            num_hidden_layers: 2,
            num_physical_layers: Some(2),
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 128,
            value_head_dim: None,
            vocab_size: vocab,
            padded_vocab_size: Some(vocab),
            unpadded_vocab_size: Some(unpadded_vocab),
            rms_norm_eps: 1e-6,
            max_position_embeddings: 256,
            sliding_window: 64,
            sliding_window_pattern: vec![true, false],
            local_layer_ids: vec![0],
            sconv_kernel_size: 4,
            n_layers_per_attn_res_block: 2,
            d_rel: 16,
            rel_extent: 64,
            qk_l2_clip_threshold: 16.0,
            rel_l2_clip_threshold: 8.0,
            log_scaling_n_floor: 4,
            log_scaling_alpha: 0.25,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: Some(64),
            intermediate_size: Some(64),
            route_scale: 1.0,
            mlp_gate_softcap: 15.0,
            mlp_up_softcap: 15.0,
            final_logit_softcapping: 30.0,
            loop_config: None,
            vision_config: None,
            dspark_config: None,
            eos_token_id: 2,
            pad_token_id: 0,
            gate_up_interleaved: false,
        };
        let emb_w = MxArray::zeros(&[vocab as i64, d as i64], Some(DType::BFloat16)).expect("emb");
        let embed_tokens = Embedding::from_weight(&emb_w).expect("Embedding::from_weight");
        let embed_norm = MxArray::ones(&[d as i64], Some(DType::Float32)).expect("embed_norm");
        let output_res_query = MxArray::zeros(&[1, d as i64], Some(DType::Float32)).expect("out_q");
        let output_res_query_t = output_res_query.transpose(None).expect("out_q_t");
        let output_norm = MxArray::ones(&[d as i64], Some(DType::Float32)).expect("out_norm");
        let lm_w = MxArray::zeros(&[vocab as i64, d as i64], Some(DType::BFloat16)).expect("lm_w");
        let lm_head = LinearProj::Standard(Linear::from_weights(&lm_w, None).expect("lm_head"));

        Inkling2Inner {
            config,
            embed_tokens,
            embed_norm,
            layers: Vec::new(),
            output_res_query,
            output_res_query_t,
            output_norm,
            lm_head,
            lm_head_bf16_unpadded: None,
            vision_encoder: None,
            tokenizer: None,
            dspark: None,
            dflash2: None,
            dflash2_identity: None,
            dspark_turn_contexts: HashMap::new(),
            cuda_dflash2_ctx: None,
            cuda_verify_budget: crate::models::qwen3_5::verify_tree::VerifyBudget::new(),
            cuda_lookup_acceptance: crate::models::qwen3_5::verify_tree::LookupAcceptance::new(),
            cuda_dspark_ctx: None,
            scheduled_dspark: Inkling2ScheduledDsparkStates::default(),
            paged: None,
            flat_cache: None,
            cuda_cache: None,
            cuda_kv_format: super::super::cuda_kv::Inkling2CudaKvFormat::Bf16,
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
        }
    }

    fn dflash2_config_json(
        hidden_size: usize,
        num_target_layers: usize,
        vocab_size: usize,
        valid_vocab_size: usize,
    ) -> serde_json::Value {
        serde_json::json!({
            "architectures": ["DFlash2DraftModel"],
            "hidden_size": hidden_size,
            "intermediate_size": hidden_size,
            "num_hidden_layers": 1,
            "num_target_layers": num_target_layers,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "vocab_size": vocab_size,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 256,
            "sliding_window": 16,
            "layer_types": ["sliding_attention"],
            "is_causal": false,
            "rope_theta": 10000.0,
            "dflash_config": {
                "block_size": 8,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "mask_token_id": 0,
                "selector_rank": 64,
                "selector_top_k": 4,
                "target_layer_ids": [0, 1],
                "final_logit_softcapping": 30.0,
                "selector_softcap_mode": "after_correction",
                "valid_vocab_size": valid_vocab_size,
                "embedding_normalization": "folded_rmsnorm",
                "capture_final_norm": true,
                "target_hidden_reduction": "none"
            }
        })
    }

    #[test]
    fn auto_draft_path_soft_fallback_on_corrupt_mismatched_and_missing() {
        let root = TempDirGuard::new("soft-fallback");
        let model_dir = root.path().join("inkling-2-mini");
        let external_store = root.path().join("external-store");
        fs::create_dir_all(&model_dir).expect("create model_dir");
        fs::create_dir_all(&external_store).expect("create external_store");

        // 1. Corrupt/truncated `model.safetensors` with valid matching `config.json`.
        let corrupt_weights = external_store.join("external-corrupt-weights");
        fs::create_dir_all(&corrupt_weights).expect("mkdir corrupt_weights");
        fs::write(
            corrupt_weights.join("config.json"),
            dflash2_config_json(256, 2, 64, 60).to_string(),
        )
        .expect("write config.json");
        fs::write(
            corrupt_weights.join("model.safetensors"),
            b"not-safetensors",
        )
        .expect("write truncated safetensors");

        let mut inner = build_synthetic_draft_target_inner();
        load_inkling2_draft_into(
            &mut inner,
            &model_dir,
            &model_dir,
            false,
            None,
            Some(&corrupt_weights.to_string_lossy()),
            true,
        )
        .expect("corrupt auto_draft_model_path must soft-fallback, never Err");
        assert!(inner.dflash2.is_none());
        assert!(inner.dspark.is_none());

        // 2. Corrupt `config.json` (malformed JSON).
        let corrupt_config = external_store.join("external-corrupt-config");
        fs::create_dir_all(&corrupt_config).expect("mkdir corrupt_config");
        fs::write(corrupt_config.join("config.json"), "{ invalid json").expect("write bad json");
        fs::write(corrupt_config.join("model.safetensors"), b"truncated").expect("write stub");

        let mut inner = build_synthetic_draft_target_inner();
        load_inkling2_draft_into(
            &mut inner,
            &model_dir,
            &model_dir,
            false,
            None,
            Some(&corrupt_config.to_string_lossy()),
            true,
        )
        .expect("corrupt config.json in auto_draft_model_path must soft-fallback");
        assert!(inner.dflash2.is_none());

        // 3. Mismatched-shape `config.json` (wrong hidden_size / num_target_layers / vocab_size).
        let mismatched = external_store.join("external-mismatched-shape");
        fs::create_dir_all(&mismatched).expect("mkdir mismatched");
        fs::write(
            mismatched.join("config.json"),
            dflash2_config_json(512, 48, 128, 120).to_string(),
        )
        .expect("write mismatched config");
        fs::write(mismatched.join("model.safetensors"), b"stub").expect("write stub");

        let mut inner = build_synthetic_draft_target_inner();
        load_inkling2_draft_into(
            &mut inner,
            &model_dir,
            &model_dir,
            false,
            None,
            Some(&mismatched.to_string_lossy()),
            true,
        )
        .expect("mismatched auto_draft_model_path must soft-fallback");
        assert!(inner.dflash2.is_none());

        // 4. Nonexistent `auto_draft_model_path`.
        let missing = external_store.join("does-not-exist-companion");
        let mut inner = build_synthetic_draft_target_inner();
        load_inkling2_draft_into(
            &mut inner,
            &model_dir,
            &model_dir,
            false,
            None,
            Some(&missing.to_string_lossy()),
            true,
        )
        .expect("nonexistent auto_draft_model_path must soft-fallback");
        assert!(inner.dflash2.is_none());

        // 5. Corrupt `auto_draft_model_path` falls through cleanly to a valid sibling DFlash2 dir.
        let sibling_valid = root.path().join("inkling-2-mini-dflash2");
        crate::models::qwen3_5::dflash2::write_test_checkpoint(
            &sibling_valid,
            &dflash2_config_json(256, 2, 64, 60),
        )
        .expect("write valid sibling DFlash2 checkpoint");

        let mut inner = build_synthetic_draft_target_inner();
        load_inkling2_draft_into(
            &mut inner,
            &model_dir,
            &model_dir,
            false,
            None,
            Some(&corrupt_weights.to_string_lossy()),
            true,
        )
        .expect("corrupt auto_draft_model_path must fall through to valid sibling");
        assert!(
            inner.dflash2.is_some(),
            "expected valid sibling DFlash2 to load after corrupt auto_draft_model_path"
        );
    }

    #[test]
    fn explicit_draft_path_remains_strict_and_ignores_auto_draft_path() {
        let root = TempDirGuard::new("explicit-strict");
        let model_dir = root.path().join("inkling-2-mini");
        let external_store = root.path().join("external-store");
        fs::create_dir_all(&model_dir).expect("create model_dir");
        fs::create_dir_all(&external_store).expect("create external_store");

        let valid_auto = external_store.join("valid-external-dflash2");
        crate::models::qwen3_5::dflash2::write_test_checkpoint(
            &valid_auto,
            &dflash2_config_json(256, 2, 64, 60),
        )
        .expect("write valid external checkpoint");

        // Sanity check: `valid_auto` genuinely loads via `auto_draft_path` when auto-load is on.
        let mut sanity_inner = build_synthetic_draft_target_inner();
        load_inkling2_draft_into(
            &mut sanity_inner,
            &model_dir,
            &model_dir,
            false,
            None,
            Some(&valid_auto.to_string_lossy()),
            true,
        )
        .expect("valid external auto_draft_model_path must load");
        assert!(sanity_inner.dflash2.is_some());

        let corrupt_weights = external_store.join("corrupt-explicit-weights");
        fs::create_dir_all(&corrupt_weights).expect("mkdir corrupt_weights");
        fs::write(
            corrupt_weights.join("config.json"),
            dflash2_config_json(256, 2, 64, 60).to_string(),
        )
        .expect("write config.json");
        fs::write(corrupt_weights.join("model.safetensors"), b"corrupt-bytes")
            .expect("write corrupt safetensors");

        // 1. Explicit corrupt DFlash2 directory MUST return Err even when `auto_draft_path` is valid.
        let mut inner = build_synthetic_draft_target_inner();
        let err = load_inkling2_draft_into(
            &mut inner,
            &model_dir,
            &model_dir,
            false,
            Some(&corrupt_weights.to_string_lossy()),
            Some(&valid_auto.to_string_lossy()),
            true,
        )
        .map(|(k, _)| k)
        .expect_err("explicit draft_model_path with corrupt safetensors must fail hard");
        assert!(!err.reason.is_empty());
        assert!(inner.dflash2.is_none());

        // 2. Explicit mismatched-shape DFlash2 directory MUST return Err.
        let mismatched = external_store.join("mismatched-explicit");
        fs::create_dir_all(&mismatched).expect("mkdir mismatched");
        fs::write(
            mismatched.join("config.json"),
            dflash2_config_json(512, 48, 128, 120).to_string(),
        )
        .expect("write mismatched config");
        fs::write(mismatched.join("model.safetensors"), b"stub").expect("write stub");

        let mut inner = build_synthetic_draft_target_inner();
        let err = load_inkling2_draft_into(
            &mut inner,
            &model_dir,
            &model_dir,
            false,
            Some(&mismatched.to_string_lossy()),
            Some(&valid_auto.to_string_lossy()),
            true,
        )
        .map(|(k, _)| k)
        .expect_err("explicit draft_model_path with mismatched shape must fail hard");
        assert!(
            err.reason.contains("mismatch"),
            "expected shape mismatch error, got: {}",
            err.reason
        );
        assert!(inner.dflash2.is_none());

        // 3. Explicit `"none"` / `"off"` / `""` disables all drafts and ignores `auto_draft_path`.
        for disable in ["none", "off", ""] {
            let mut inner = build_synthetic_draft_target_inner();
            load_inkling2_draft_into(
                &mut inner,
                &model_dir,
                &model_dir,
                false,
                Some(disable),
                Some(&valid_auto.to_string_lossy()),
                true,
            )
            .expect("explicit disable sentinel must succeed");
            assert!(
                inner.dflash2.is_none() && inner.dspark.is_none(),
                "explicit '{disable}' must disable all drafts"
            );
        }

        // 4. `auto_load_draft = false` ignores `auto_draft_path`.
        let mut inner = build_synthetic_draft_target_inner();
        load_inkling2_draft_into(
            &mut inner,
            &model_dir,
            &model_dir,
            false,
            None,
            Some(&valid_auto.to_string_lossy()),
            false,
        )
        .expect("auto_load_draft=false must succeed without loading");
        assert!(inner.dflash2.is_none());
        assert!(inner.dspark.is_none());
    }

    #[test]
    fn auto_dflash2_candidate_precedence_and_dedup() {
        let root = TempDirGuard::new("precedence");
        let model_dir = root.path().join("inkling-2-mini");
        let colocated = model_dir.join("draft");
        let sibling = root.path().join("inkling-2-mini-dflash2");
        let external_parent = root.path().join("external-store");
        let external = external_parent.join("cached-dflash2");
        fs::create_dir_all(&colocated).expect("mkdir colocated");
        fs::create_dir_all(&sibling).expect("mkdir sibling");
        fs::create_dir_all(&external).expect("mkdir external");

        let cfg = dflash2_config_json(256, 2, 64, 60).to_string();
        for dir in [&colocated, &sibling, &external] {
            fs::write(dir.join("config.json"), &cfg).expect("write config");
            fs::write(dir.join("model.safetensors"), b"stub").expect("write stub");
        }

        let inner = build_synthetic_draft_target_inner();
        let target_shape = inner.draft_target_shape();

        // Exact order: 1. colocated `<modelDir>/draft/`, 2. `auto_draft_path`, 3. sibling `inkling-2-mini-dflash2/`.
        let ordered = build_auto_dflash2_candidates(
            &model_dir,
            &model_dir,
            target_shape,
            Some(&external.to_string_lossy()),
        );
        let ordered_canon: Vec<PathBuf> =
            ordered.iter().map(|p| normalize_candidate_key(p)).collect();
        assert_eq!(
            ordered_canon,
            vec![
                normalize_candidate_key(&colocated),
                normalize_candidate_key(&external),
                normalize_candidate_key(&sibling),
            ]
        );

        // Deduplication when `auto_draft_path` aliases colocated `<modelDir>/draft/` via `./`.
        let colocated_alias = model_dir.join(".").join("draft");
        let dedup_colocated = build_auto_dflash2_candidates(
            &model_dir,
            &model_dir,
            target_shape,
            Some(&colocated_alias.to_string_lossy()),
        );
        let dedup_colocated_canon: Vec<PathBuf> = dedup_colocated
            .iter()
            .map(|p| normalize_candidate_key(p))
            .collect();
        assert_eq!(
            dedup_colocated_canon,
            vec![
                normalize_candidate_key(&colocated),
                normalize_candidate_key(&sibling),
            ]
        );

        // Deduplication when `auto_draft_path` aliases sibling `inkling-2-mini-dflash2/`.
        let dedup_sibling = build_auto_dflash2_candidates(
            &model_dir,
            &model_dir,
            target_shape,
            Some(&sibling.to_string_lossy()),
        );
        let dedup_sibling_canon: Vec<PathBuf> = dedup_sibling
            .iter()
            .map(|p| normalize_candidate_key(p))
            .collect();
        assert_eq!(
            dedup_sibling_canon,
            vec![
                normalize_candidate_key(&colocated),
                normalize_candidate_key(&sibling),
            ]
        );
    }

    #[test]
    fn loaded_draft_kind_distinguishes_explicit_auto_and_disabled() {
        let root = TempDirGuard::new("draft-kind");
        let model_dir = root.path().join("inkling-2-mini");
        let external_dir = root.path().join("external-dflash2");
        fs::create_dir_all(&model_dir).expect("create model_dir");
        crate::models::qwen3_5::dflash2::write_test_checkpoint(
            &external_dir,
            &dflash2_config_json(256, 2, 64, 60),
        )
        .expect("write valid external DFlash2 checkpoint");

        // Explicit DFlash2 path -> LoadedDraftKind::Explicit.
        let mut inner = build_synthetic_draft_target_inner();
        let (kind, resident) = load_inkling2_draft_into(
            &mut inner,
            &model_dir,
            &model_dir,
            false,
            Some(&external_dir.to_string_lossy()),
            None,
            true,
        )
        .expect("explicit DFlash2 should load");
        assert_eq!(kind, LoadedDraftKind::Explicit);
        assert!(resident.is_none());
        assert!(inner.dflash2.is_some());

        // Auto-discovered DFlash2 -> LoadedDraftKind::AutoDflash2.
        let mut inner = build_synthetic_draft_target_inner();
        let (kind, resident) = load_inkling2_draft_into(
            &mut inner,
            &model_dir,
            &model_dir,
            false,
            None,
            Some(&external_dir.to_string_lossy()),
            true,
        )
        .expect("auto DFlash2 should load");
        assert_eq!(kind, LoadedDraftKind::AutoDflash2);
        assert!(resident.is_none());
        assert!(inner.dflash2.is_some());

        // Disabled via "none" or auto_load_draft=false -> LoadedDraftKind::None.
        let mut inner = build_synthetic_draft_target_inner();
        let (kind, resident) = load_inkling2_draft_into(
            &mut inner,
            &model_dir,
            &model_dir,
            false,
            Some("none"),
            Some(&external_dir.to_string_lossy()),
            true,
        )
        .expect("explicit none should succeed");
        assert_eq!(kind, LoadedDraftKind::None);
        assert!(resident.is_none());
        assert!(!inner.has_speculative_draft());

        let mut inner = build_synthetic_draft_target_inner();
        let (kind, resident) = load_inkling2_draft_into(
            &mut inner,
            &model_dir,
            &model_dir,
            false,
            None,
            Some(&external_dir.to_string_lossy()),
            false,
        )
        .expect("auto_load_draft=false should succeed");
        assert_eq!(kind, LoadedDraftKind::None);
        assert!(resident.is_none());
        assert!(!inner.has_speculative_draft());
    }

    fn write_synthetic_dspark_checkpoint(dir: &Path) {
        fs::create_dir_all(dir).expect("mkdir dspark checkpoint dir");
        let cfg = serde_json::json!({
            "model_type": "inkling2",
            "mtp_config": {
                "hidden_size": 256,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 128,
                "block_size": 8,
                "markov_rank": 32,
                "target_layer_ids": [0, 1],
                "local_layer_ids": [0]
            }
        });
        fs::write(dir.join("config.json"), cfg.to_string()).expect("write dspark config.json");

        let d = 256i64;
        let q_dim = 256i64;
        let kv_dim = 128i64;
        let head_dim = 128i64;
        let ffn = 64i64;
        let vocab = 64i64;
        let rank = 32i64;

        let mut tensors: HashMap<String, MxArray> = HashMap::new();
        let mut insert_zeros = |name: &str, shape: &[i64], dtype: DType| {
            let arr = MxArray::zeros(shape, Some(dtype)).expect("zeros tensor");
            tensors.insert(name.to_string(), arr);
        };

        insert_zeros("fuse_weight", &[2, 2], DType::BFloat16);
        insert_zeros(
            "layers.0.self_attn.q_proj.weight",
            &[q_dim, d],
            DType::BFloat16,
        );
        insert_zeros(
            "layers.0.self_attn.k_proj.weight",
            &[kv_dim, d],
            DType::BFloat16,
        );
        insert_zeros(
            "layers.0.self_attn.v_proj.weight",
            &[kv_dim, d],
            DType::BFloat16,
        );
        insert_zeros(
            "layers.0.self_attn.k_proj_ctx.weight",
            &[kv_dim, d],
            DType::BFloat16,
        );
        insert_zeros(
            "layers.0.self_attn.v_proj_ctx.weight",
            &[kv_dim, d],
            DType::BFloat16,
        );
        insert_zeros(
            "layers.0.self_attn.o_proj.weight",
            &[d, q_dim],
            DType::BFloat16,
        );
        insert_zeros("layers.0.mlp.gate_proj.weight", &[ffn, d], DType::BFloat16);
        insert_zeros("layers.0.mlp.up_proj.weight", &[ffn, d], DType::BFloat16);
        insert_zeros("layers.0.mlp.down_proj.weight", &[d, ffn], DType::BFloat16);
        insert_zeros("layers.0.input_layernorm.weight", &[d], DType::Float32);
        insert_zeros(
            "layers.0.post_attention_layernorm.weight",
            &[d],
            DType::Float32,
        );
        insert_zeros("layers.0.dflare_norm.weight", &[d], DType::Float32);
        insert_zeros(
            "layers.0.self_attn.q_norm.weight",
            &[head_dim],
            DType::Float32,
        );
        insert_zeros(
            "layers.0.self_attn.k_norm.weight",
            &[head_dim],
            DType::Float32,
        );
        insert_zeros(
            "layers.0.self_attn.k_norm_ctx.weight",
            &[head_dim],
            DType::Float32,
        );
        insert_zeros("norm.weight", &[d], DType::Float32);
        insert_zeros("mask_embed", &[d], DType::BFloat16);
        insert_zeros("markov_head.dflare_norm.weight", &[d], DType::Float32);
        insert_zeros(
            "markov_head.markov_w1.weight",
            &[vocab, rank],
            DType::BFloat16,
        );
        insert_zeros(
            "markov_head.rnn_proj.weight",
            &[3 * rank, 2 * rank + 2 * d],
            DType::BFloat16,
        );
        insert_zeros("markov_head.rnn_proj.bias", &[3 * rank], DType::Float32);
        insert_zeros("markov_head.hidden_out.weight", &[d, rank], DType::BFloat16);
        insert_zeros("markov_head.hidden_out.bias", &[d], DType::BFloat16);

        crate::utils::safetensors::save_safetensors(
            dir.join("mtp.safetensors"),
            &mut tensors,
            None,
        )
        .expect("save synthetic mtp.safetensors");
    }

    #[test]
    fn auto_draft_materialization_failure_falls_through_while_explicit_fails_strictly() {
        let root = TempDirGuard::new("draft-mat-fallback");
        let model_dir = root.path().join("inkling-2-mini");
        let dflash_dir = model_dir.join("draft");
        write_synthetic_dspark_checkpoint(&model_dir);
        crate::models::qwen3_5::dflash2::write_test_checkpoint(
            &dflash_dir,
            &dflash2_config_json(256, 2, 64, 60),
        )
        .expect("write valid colocated DFlash2 checkpoint");

        let oom_msg = "Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)";

        // 1. Auto DFlash2 materialization failure falls through to bundled Auto DSpark.
        let mut inner = build_synthetic_draft_target_inner();
        let (kind, resident) = {
            let _oom = crate::array::fail_native_call_at("weight_materialize_chunk", oom_msg);
            load_inkling2_draft_into(&mut inner, &model_dir, &model_dir, false, None, None, true)
                .expect("auto DFlash2 OOM must soft-fallback to auto DSpark")
        };
        assert_eq!(kind, LoadedDraftKind::AutoDspark);
        assert!(
            resident.as_ref().is_some_and(|w| w.arrays() > 0),
            "kept AutoDspark must return its non-empty WeightsResident witness"
        );
        assert!(inner.dflash2.is_none());
        assert!(inner.dspark.is_some());

        // 2. Auto DSpark materialization failure (when DFlash2 is absent) falls through to pure AR.
        let dspark_only_dir = root.path().join("inkling-2-mini-dspark-only");
        write_synthetic_dspark_checkpoint(&dspark_only_dir);
        let mut inner = build_synthetic_draft_target_inner();
        let (kind, resident) = {
            let _oom = crate::array::fail_native_call_at("weight_materialize_chunk", oom_msg);
            load_inkling2_draft_into(
                &mut inner,
                &dspark_only_dir,
                &dspark_only_dir,
                false,
                None,
                None,
                true,
            )
            .expect("auto DSpark OOM must soft-fallback to pure AR")
        };
        assert_eq!(kind, LoadedDraftKind::None);
        assert!(resident.is_none());
        assert!(inner.dflash2.is_none());
        assert!(inner.dspark.is_none());

        // 3. Full single-call cascade: AutoDflash2 materialization fails -> tries AutoDspark ->
        //    AutoDspark materialization ALSO fails -> falls through cleanly to None (pure AR).
        let mut inner = build_synthetic_draft_target_inner();
        let attempted: std::cell::RefCell<Vec<&'static str>> = std::cell::RefCell::new(Vec::new());
        let (kind, resident) = load_inkling2_draft_into_with(
            &mut inner,
            &model_dir,
            &model_dir,
            false,
            None,
            None,
            true,
            |_path| {
                attempted.borrow_mut().push("dflash2");
                Err(Error::from_reason(oom_msg))
            },
            |arrays| {
                assert!(!arrays.is_empty());
                attempted.borrow_mut().push("dspark");
                Err(Error::from_reason(oom_msg))
            },
        )
        .expect("AutoDflash2 -> AutoDspark -> None materialization cascade must not abort");
        assert_eq!(*attempted.borrow(), vec!["dflash2", "dspark"]);
        assert_eq!(kind, LoadedDraftKind::None);
        assert!(resident.is_none());
        assert!(inner.dflash2.is_none());
        assert!(inner.dspark.is_none());

        // 4. Explicit DFlash2 materialization failure propagates Err strictly (never falls back
        //    to colocated DSpark).
        let mut inner = build_synthetic_draft_target_inner();
        let err = {
            let _oom = crate::array::fail_native_call_at("weight_materialize_chunk", oom_msg);
            load_inkling2_draft_into(
                &mut inner,
                &model_dir,
                &model_dir,
                false,
                Some(&dflash_dir.to_string_lossy()),
                None,
                true,
            )
            .map(|(k, _)| k)
            .expect_err("explicit DFlash2 materialization failure must fail hard")
        };
        assert!(
            err.reason.contains("Insufficient Memory"),
            "expected OOM error to propagate, got: {}",
            err.reason
        );
        assert!(inner.dflash2.is_none());
        assert!(inner.dspark.is_none());

        // 5. Explicit DSpark materialization failure propagates Err strictly and leaves
        //    `inner.dspark` unset.
        let mtp_path = model_dir.join("mtp.safetensors");
        let mut inner = build_synthetic_draft_target_inner();
        let err = {
            let _oom = crate::array::fail_native_call_at("weight_materialize_chunk", oom_msg);
            load_inkling2_draft_into(
                &mut inner,
                &model_dir,
                &model_dir,
                false,
                Some(&mtp_path.to_string_lossy()),
                None,
                true,
            )
            .map(|(k, _)| k)
            .expect_err("explicit DSpark materialization failure must fail hard")
        };
        assert!(
            err.reason.contains("Insufficient Memory"),
            "expected OOM error to propagate, got: {}",
            err.reason
        );
        assert!(inner.dspark.is_none());
        assert!(inner.dflash2.is_none());
    }

    #[test]
    fn memory_or_fit_error_classifies_real_plan_and_governor_errors() {
        use crate::engine::governor::governor::Failure;
        use crate::engine::memory_plan::{MemoryBound, PlanError};

        let fit_errors = vec![
            Error::from_reason(
                PlanError::KvPoolDoesNotFit {
                    required: 22u64 << 30,
                    budget: 20u64 << 30,
                    bound: MemoryBound::DeviceMemory,
                }
                .message("Inkling2"),
            ),
            Error::from_reason(
                PlanError::KvPoolDoesNotFit {
                    required: 22u64 << 30,
                    budget: 20u64 << 30,
                    bound: MemoryBound::MaxMemory,
                }
                .message("Inkling2"),
            ),
            Error::from_reason(
                PlanError::FloorsDoNotFit {
                    required: 21u64 << 30,
                    budget: 20u64 << 30,
                    bound: MemoryBound::DeviceMemory,
                }
                .message("Inkling2"),
            ),
            Error::from_reason(
                PlanError::ContextAbovePlan {
                    requested: 131_072,
                    plan: 65_536,
                    bound: MemoryBound::MaxMemory,
                }
                .message("Inkling2"),
            ),
            Error::from_reason(
                PlanError::PoolAbovePlan {
                    requested_mib: 4096,
                    plan_mib: 2048,
                    bound: MemoryBound::DeviceMemory,
                }
                .message("Inkling2"),
            ),
            Error::from_reason(
                "memory plan: existing Metal allocations exceed memory governor limit (20480 MiB charged, 18432 MiB limit)",
            ),
            Error::from_reason(
                "memory plan: macOS is short of memory (512 MiB available, 2048 MiB kept for macOS); close memory-heavy applications",
            ),
            Error::from_reason(format!(
                "memory plan: the KV pool's runway (128 MiB) could not be allocated: {}",
                Failure::name(Some(Failure::EngineBudget))
            )),
            Error::from_reason(format!(
                "memory plan: the KV pool's runway (128 MiB) could not be allocated: {}",
                Failure::name(Some(Failure::HostPressure))
            )),
            Error::from_reason(format!(
                "memory plan: the KV pool's runway (128 MiB) could not be allocated: {}",
                Failure::name(Some(Failure::DriverRejected))
            )),
            Error::from_reason(
                "Failed to construct Inkling2 KV group 0 pool: LayerKVPool::new: Metal returned a nil Key buffer",
            ),
            Error::from_reason(
                "[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).",
            ),
        ];

        for err in &fit_errors {
            assert!(
                is_memory_or_fit_error(err),
                "expected memory/fit error to trigger fallback: {}",
                err.reason
            );
        }

        let structural_errors = vec![
            Error::from_reason("Inkling2 memory plan: config has zero global attention visits"),
            Error::from_reason("Inkling2 memory plan: Metal working set unavailable"),
            Error::from_reason("Inkling2 memory plan: produced no KV pool plan"),
            Error::from_reason(
                PlanError::WorkingSetTooSmall {
                    working_set: 512u64 << 20,
                    margin: 1u64 << 30,
                }
                .message("Inkling2"),
            ),
            Error::from_reason(
                PlanError::ContextWithoutPlan {
                    requested: 2_000_000,
                    native: 1_048_576,
                }
                .message("Inkling2"),
            ),
            Error::from_reason("Inkling2 KV cache block_size must be > 0"),
            Error::from_reason(
                "Failed to construct Inkling2 KV group 0 pool: LayerKVPool::new: num_blocks must be > 0",
            ),
            Error::from_reason(format!(
                "memory plan: the KV pool's runway (128 MiB) could not be allocated: {}",
                Failure::name(Some(Failure::OutsideScope))
            )),
        ];

        for err in &structural_errors {
            assert!(
                !is_memory_or_fit_error(err),
                "structural/non-memory error must NOT trigger draft fallback: {}",
                err.reason
            );
        }
    }

    #[test]
    fn kv_plan_auto_draft_fallback_transitions() {
        use crate::engine::memory_plan::{MemoryBound, PlanError};

        #[derive(Default)]
        struct FakeState {
            dflash2_loaded: bool,
            dspark_loaded: bool,
            has_bundled_dspark: bool,
            dropped_dflash2: usize,
            tried_load_dspark: usize,
            dropped_dspark: usize,
        }

        let kv_fit_err = || {
            Error::from_reason(
                PlanError::KvPoolDoesNotFit {
                    required: 22u64 << 30,
                    budget: 20u64 << 30,
                    bound: MemoryBound::DeviceMemory,
                }
                .message("Inkling2"),
            )
        };
        let context_fit_err = || {
            Error::from_reason(
                PlanError::ContextAbovePlan {
                    requested: 131_072,
                    plan: 65_536,
                    bound: MemoryBound::MaxMemory,
                }
                .message("Inkling2"),
            )
        };

        // (a) AutoDflash2 fails with fit error -> releases DFlash2 -> loads AutoDspark -> succeeds.
        let mut st = FakeState {
            dflash2_loaded: true,
            has_bundled_dspark: true,
            ..Default::default()
        };
        let mut attempts = Vec::new();
        let (res, final_kind) = plan_kv_with_auto_draft_fallback(
            &mut st,
            LoadedDraftKind::AutoDflash2,
            |s, kind| {
                attempts.push(kind);
                // Verify weights were actually released/loaded before re-planning.
                match kind {
                    LoadedDraftKind::AutoDflash2 => {
                        assert!(s.dflash2_loaded && !s.dspark_loaded);
                        Err(kv_fit_err())
                    }
                    LoadedDraftKind::AutoDspark => {
                        assert!(!s.dflash2_loaded && s.dspark_loaded);
                        Ok("planned-with-dspark")
                    }
                    other => panic!("unexpected plan attempt kind: {other:?}"),
                }
            },
            |s| {
                s.dflash2_loaded = false;
                s.dropped_dflash2 += 1;
            },
            |s| {
                s.tried_load_dspark += 1;
                if s.has_bundled_dspark {
                    s.dspark_loaded = true;
                    LoadedDraftKind::AutoDspark
                } else {
                    LoadedDraftKind::None
                }
            },
            |s| {
                s.dspark_loaded = false;
                s.dropped_dspark += 1;
            },
        )
        .expect("AutoDflash2 -> AutoDspark fallback must succeed");
        assert_eq!(res, "planned-with-dspark");
        assert_eq!(final_kind, LoadedDraftKind::AutoDspark);
        assert_eq!(
            attempts,
            vec![LoadedDraftKind::AutoDflash2, LoadedDraftKind::AutoDspark]
        );
        assert!(!st.dflash2_loaded && st.dspark_loaded);
        assert_eq!(st.dropped_dflash2, 1);
        assert_eq!(st.tried_load_dspark, 1);
        assert_eq!(st.dropped_dspark, 0);

        // (b) AutoDflash2 fails with fit error -> AutoDspark ALSO fails with fit error ->
        //     releases DSpark -> succeeds as None (pure AR).
        let mut st = FakeState {
            dflash2_loaded: true,
            has_bundled_dspark: true,
            ..Default::default()
        };
        let mut attempts = Vec::new();
        let (res, final_kind) = plan_kv_with_auto_draft_fallback(
            &mut st,
            LoadedDraftKind::AutoDflash2,
            |s, kind| {
                attempts.push(kind);
                match kind {
                    LoadedDraftKind::AutoDflash2 => {
                        assert!(s.dflash2_loaded && !s.dspark_loaded);
                        Err(kv_fit_err())
                    }
                    LoadedDraftKind::AutoDspark => {
                        assert!(!s.dflash2_loaded && s.dspark_loaded);
                        Err(context_fit_err())
                    }
                    LoadedDraftKind::None => {
                        assert!(!s.dflash2_loaded && !s.dspark_loaded);
                        Ok("planned-pure-ar")
                    }
                    LoadedDraftKind::Explicit => panic!("unexpected Explicit"),
                }
            },
            |s| {
                s.dflash2_loaded = false;
                s.dropped_dflash2 += 1;
            },
            |s| {
                s.tried_load_dspark += 1;
                s.dspark_loaded = true;
                LoadedDraftKind::AutoDspark
            },
            |s| {
                s.dspark_loaded = false;
                s.dropped_dspark += 1;
            },
        )
        .expect("AutoDflash2 -> AutoDspark -> pure AR fallback must succeed");
        assert_eq!(res, "planned-pure-ar");
        assert_eq!(final_kind, LoadedDraftKind::None);
        assert_eq!(
            attempts,
            vec![
                LoadedDraftKind::AutoDflash2,
                LoadedDraftKind::AutoDspark,
                LoadedDraftKind::None,
            ]
        );
        assert!(!st.dflash2_loaded && !st.dspark_loaded);
        assert_eq!(st.dropped_dflash2, 1);
        assert_eq!(st.tried_load_dspark, 1);
        assert_eq!(st.dropped_dspark, 1);

        // (c) AutoDflash2 fails with fit error -> no bundled DSpark exists -> retries directly
        //     as None (pure AR) and succeeds in 2 attempts.
        let mut st = FakeState {
            dflash2_loaded: true,
            has_bundled_dspark: false,
            ..Default::default()
        };
        let mut attempts = Vec::new();
        let (res, final_kind) = plan_kv_with_auto_draft_fallback(
            &mut st,
            LoadedDraftKind::AutoDflash2,
            |s, kind| {
                attempts.push(kind);
                match kind {
                    LoadedDraftKind::AutoDflash2 => Err(kv_fit_err()),
                    LoadedDraftKind::None => {
                        assert!(!s.dflash2_loaded && !s.dspark_loaded);
                        Ok("planned-direct-ar")
                    }
                    other => panic!("unexpected attempt kind: {other:?}"),
                }
            },
            |s| {
                s.dflash2_loaded = false;
                s.dropped_dflash2 += 1;
            },
            |s| {
                s.tried_load_dspark += 1;
                LoadedDraftKind::None
            },
            |s| {
                s.dropped_dspark += 1;
            },
        )
        .expect("AutoDflash2 -> None direct fallback must succeed");
        assert_eq!(res, "planned-direct-ar");
        assert_eq!(final_kind, LoadedDraftKind::None);
        assert_eq!(
            attempts,
            vec![LoadedDraftKind::AutoDflash2, LoadedDraftKind::None]
        );
        assert_eq!(st.dropped_dflash2, 1);
        assert_eq!(st.tried_load_dspark, 1);
        assert_eq!(st.dropped_dspark, 0);

        // (d) Explicit draft fails with fit error -> immediately propagates Err without dropping
        //     the explicit draft or retrying.
        let mut st = FakeState {
            dflash2_loaded: true,
            has_bundled_dspark: true,
            ..Default::default()
        };
        let mut attempts = Vec::new();
        let err = plan_kv_with_auto_draft_fallback(
            &mut st,
            LoadedDraftKind::Explicit,
            |_s, kind| -> Result<()> {
                attempts.push(kind);
                Err(kv_fit_err())
            },
            |s| {
                s.dflash2_loaded = false;
                s.dropped_dflash2 += 1;
            },
            |s| {
                s.tried_load_dspark += 1;
                LoadedDraftKind::AutoDspark
            },
            |s| {
                s.dropped_dspark += 1;
            },
        )
        .expect_err("Explicit draft must propagate fit error without fallback");
        assert!(is_memory_or_fit_error(&err));
        assert_eq!(attempts, vec![LoadedDraftKind::Explicit]);
        assert!(st.dflash2_loaded, "explicit draft must not be dropped");
        assert_eq!(st.dropped_dflash2, 0);
        assert_eq!(st.tried_load_dspark, 0);
        assert_eq!(st.dropped_dspark, 0);

        // (e) Non-memory structural error propagates immediately without dropping auto draft.
        let mut st = FakeState {
            dflash2_loaded: true,
            has_bundled_dspark: true,
            ..Default::default()
        };
        let mut attempts = Vec::new();
        let err = plan_kv_with_auto_draft_fallback(
            &mut st,
            LoadedDraftKind::AutoDflash2,
            |_s, kind| -> Result<()> {
                attempts.push(kind);
                Err(Error::from_reason(
                    "Inkling2 memory plan: config has zero global attention visits",
                ))
            },
            |s| {
                s.dflash2_loaded = false;
                s.dropped_dflash2 += 1;
            },
            |s| {
                s.tried_load_dspark += 1;
                LoadedDraftKind::AutoDspark
            },
            |s| {
                s.dropped_dspark += 1;
            },
        )
        .expect_err("structural error must propagate immediately");
        assert!(err.reason.contains("zero global attention visits"));
        assert_eq!(attempts, vec![LoadedDraftKind::AutoDflash2]);
        assert!(
            st.dflash2_loaded,
            "structural error must not drop auto draft"
        );
        assert_eq!(st.dropped_dflash2, 0);
        assert_eq!(st.tried_load_dspark, 0);
        assert_eq!(st.dropped_dspark, 0);

        // (f) Initial AutoDspark fails with fit error -> drops DSpark -> succeeds as None.
        let mut st = FakeState {
            dspark_loaded: true,
            ..Default::default()
        };
        let mut attempts = Vec::new();
        let (res, final_kind) = plan_kv_with_auto_draft_fallback(
            &mut st,
            LoadedDraftKind::AutoDspark,
            |s, kind| {
                attempts.push(kind);
                match kind {
                    LoadedDraftKind::AutoDspark => Err(kv_fit_err()),
                    LoadedDraftKind::None => {
                        assert!(!s.dspark_loaded);
                        Ok("planned-after-dspark-drop")
                    }
                    other => panic!("unexpected attempt kind: {other:?}"),
                }
            },
            |s| {
                s.dropped_dflash2 += 1;
            },
            |s| {
                s.tried_load_dspark += 1;
                LoadedDraftKind::None
            },
            |s| {
                s.dspark_loaded = false;
                s.dropped_dspark += 1;
            },
        )
        .expect("AutoDspark -> None fallback must succeed");
        assert_eq!(res, "planned-after-dspark-drop");
        assert_eq!(final_kind, LoadedDraftKind::None);
        assert_eq!(
            attempts,
            vec![LoadedDraftKind::AutoDspark, LoadedDraftKind::None]
        );
        assert_eq!(st.dropped_dflash2, 0);
        assert_eq!(st.tried_load_dspark, 0);
        assert_eq!(st.dropped_dspark, 1);

        // (g) Even pure AR fails with fit error -> terminates cleanly after 3 attempts and returns
        //     the final pure-AR error.
        let mut st = FakeState {
            dflash2_loaded: true,
            has_bundled_dspark: true,
            ..Default::default()
        };
        let mut attempts = Vec::new();
        let err = plan_kv_with_auto_draft_fallback(
            &mut st,
            LoadedDraftKind::AutoDflash2,
            |_s, kind| -> Result<()> {
                attempts.push(kind);
                match kind {
                    LoadedDraftKind::AutoDflash2 => Err(kv_fit_err()),
                    LoadedDraftKind::AutoDspark => Err(kv_fit_err()),
                    LoadedDraftKind::None => Err(Error::from_reason(
                        PlanError::FloorsDoNotFit {
                            required: 19u64 << 30,
                            budget: 16u64 << 30,
                            bound: MemoryBound::DeviceMemory,
                        }
                        .message("Inkling2"),
                    )),
                    LoadedDraftKind::Explicit => panic!("unexpected Explicit"),
                }
            },
            |s| {
                s.dflash2_loaded = false;
                s.dropped_dflash2 += 1;
            },
            |s| {
                s.tried_load_dspark += 1;
                s.dspark_loaded = true;
                LoadedDraftKind::AutoDspark
            },
            |s| {
                s.dspark_loaded = false;
                s.dropped_dspark += 1;
            },
        )
        .expect_err("pure AR fit failure must return final error");
        assert!(err.reason.contains("floors of the memory plan require"));
        assert_eq!(
            attempts,
            vec![
                LoadedDraftKind::AutoDflash2,
                LoadedDraftKind::AutoDspark,
                LoadedDraftKind::None,
            ]
        );
        assert!(!st.dflash2_loaded && !st.dspark_loaded);
        assert_eq!(st.dropped_dflash2, 1);
        assert_eq!(st.tried_load_dspark, 1);
        assert_eq!(st.dropped_dspark, 1);
    }

    #[test]
    fn dflash2_identity_binds_to_loaded_fusion_weights_across_same_size_same_mtime_edits() {
        let root = TempDirGuard::new("dflash2-fusion-fp");
        let model_dir = root.path().join("inkling-2-mini");
        fs::create_dir_all(&model_dir).expect("create model_dir");

        // Helper: clone `src_dir` into `dst_dir`, mutate one named BF16 tensor in
        // `model.safetensors` in-place while preserving exact file size and exact file mtime.
        let copy_and_mutate_tensor = |src_dir: &Path, dst_dir: &Path, tensor_name: &str| {
            fs::create_dir_all(dst_dir).expect("create dst_dir");
            for file in ["config.json", "model.safetensors"] {
                fs::copy(src_dir.join(file), dst_dir.join(file)).expect("copy checkpoint file");
            }
            let st_path = dst_dir.join("model.safetensors");
            let orig_meta = fs::metadata(src_dir.join("model.safetensors")).expect("src meta");
            let orig_mtime = orig_meta.modified().expect("src mtime");
            let cfg_mtime = fs::metadata(src_dir.join("config.json"))
                .expect("cfg meta")
                .modified()
                .expect("cfg mtime");

            let mut bytes = fs::read(&st_path).expect("read safetensors");
            let hdr_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
            let header: serde_json::Value =
                serde_json::from_slice(&bytes[8..8 + hdr_len]).expect("parse safetensors header");
            let offsets = header
                .get(tensor_name)
                .and_then(|v| v.get("data_offsets"))
                .and_then(serde_json::Value::as_array)
                .expect("tensor data_offsets");
            let start = 8 + hdr_len + offsets[0].as_u64().unwrap() as usize;
            let end = 8 + hdr_len + offsets[1].as_u64().unwrap() as usize;
            assert!(end > start + 16, "tensor {tensor_name} payload too small");
            // Replace the first 32 BF16 values with large distinct finite values (`+3.0` / `-3.0`)
            // so both BF16 weights (`hidden_norm`) and affine Q4-quantized weights (`fc`) change.
            let stride_end = (start + 64).min(end);
            for (idx, chunk) in bytes[start..stride_end]
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .enumerate()
            {
                let val = if idx % 2 == 0 { 3.5f32 } else { -3.5f32 };
                let bits = half::bf16::from_f32(val).to_bits();
                chunk.copy_from_slice(&bits.to_le_bytes());
            }
            fs::write(&st_path, &bytes).expect("write mutated safetensors");

            // Restore exact file mtimes on both `model.safetensors` and `config.json`.
            fs::OpenOptions::new()
                .write(true)
                .open(&st_path)
                .expect("open mutated safetensors")
                .set_modified(orig_mtime)
                .expect("set safetensors mtime");
            fs::OpenOptions::new()
                .write(true)
                .open(dst_dir.join("config.json"))
                .expect("open config.json")
                .set_modified(cfg_mtime)
                .expect("set config mtime");

            // Verify exact file size AND exact mtime match the original directory.
            let mut_meta = fs::metadata(&st_path).expect("mut meta");
            assert_eq!(mut_meta.len(), orig_meta.len());
            assert_eq!(mut_meta.modified().expect("mut mtime"), orig_mtime);
        };

        let dir_base = root.path().join("dflash2-base");
        let cfg_json = dflash2_config_json(256, 2, 64, 60);
        crate::models::qwen3_5::dflash2::write_test_checkpoint(&dir_base, &cfg_json)
            .expect("write baseline DFlash2 checkpoint");

        let dir_fc_mut = root.path().join("dflash2-fc-mut");
        copy_and_mutate_tensor(&dir_base, &dir_fc_mut, "fc.weight");

        let dir_norm_mut = root.path().join("dflash2-norm-mut");
        copy_and_mutate_tensor(&dir_base, &dir_norm_mut, "hidden_norm.weight");

        // Load baseline on BOTH explicit and auto-discovered paths.
        let mut base_explicit = build_synthetic_draft_target_inner();
        let (kind_base_exp, _) = load_inkling2_draft_into_with(
            &mut base_explicit,
            &model_dir,
            &model_dir,
            false,
            Some(&dir_base.to_string_lossy()),
            None,
            true,
            load_dflash2,
            materialize_weights,
        )
        .expect("baseline explicit load");
        assert_eq!(kind_base_exp, LoadedDraftKind::Explicit);
        let fp_base_exp = base_explicit
            .dflash2_identity
            .clone()
            .expect("baseline explicit identity");
        assert_eq!(
            fp_base_exp,
            base_explicit
                .dflash2
                .as_ref()
                .unwrap()
                .context_fusion_fingerprint()
                .expect("baseline context_fusion_fingerprint"),
            "dflash2_identity must match loaded DFlash2Model::context_fusion_fingerprint()"
        );

        let mut base_auto = build_synthetic_draft_target_inner();
        let (kind_base_auto, _) = load_inkling2_draft_into_with(
            &mut base_auto,
            &model_dir,
            &model_dir,
            false,
            None,
            Some(&dir_base.to_string_lossy()),
            true,
            load_dflash2,
            materialize_weights,
        )
        .expect("baseline auto load");
        assert_eq!(kind_base_auto, LoadedDraftKind::AutoDflash2);
        assert_eq!(
            base_auto.dflash2_identity.as_deref(),
            Some(fp_base_exp.as_slice()),
            "explicit and auto-discovered loads of identical weights must produce identical identity"
        );

        // Load same-size, same-mtime `fc.weight`-mutated checkpoint (explicit + auto).
        let mut fc_explicit = build_synthetic_draft_target_inner();
        load_inkling2_draft_into_with(
            &mut fc_explicit,
            &model_dir,
            &model_dir,
            false,
            Some(&dir_fc_mut.to_string_lossy()),
            None,
            true,
            load_dflash2,
            materialize_weights,
        )
        .expect("fc-mutated explicit load");
        let fp_fc_exp = fc_explicit
            .dflash2_identity
            .clone()
            .expect("fc-mutated explicit identity");
        assert_ne!(
            fp_fc_exp, fp_base_exp,
            "same-size/same-mtime fc.weight mutation MUST change context_fusion_fingerprint / dflash2_identity"
        );

        let mut fc_auto = build_synthetic_draft_target_inner();
        load_inkling2_draft_into_with(
            &mut fc_auto,
            &model_dir,
            &model_dir,
            false,
            None,
            Some(&dir_fc_mut.to_string_lossy()),
            true,
            load_dflash2,
            materialize_weights,
        )
        .expect("fc-mutated auto load");
        assert_eq!(
            fc_auto.dflash2_identity.as_deref(),
            Some(fp_fc_exp.as_slice())
        );

        // Load same-size, same-mtime `hidden_norm.weight`-mutated checkpoint.
        let mut norm_explicit = build_synthetic_draft_target_inner();
        load_inkling2_draft_into_with(
            &mut norm_explicit,
            &model_dir,
            &model_dir,
            false,
            Some(&dir_norm_mut.to_string_lossy()),
            None,
            true,
            load_dflash2,
            materialize_weights,
        )
        .expect("norm-mutated explicit load");
        let fp_norm_exp = norm_explicit
            .dflash2_identity
            .clone()
            .expect("norm-mutated explicit identity");
        assert_ne!(
            fp_norm_exp, fp_base_exp,
            "same-size/same-mtime hidden_norm.weight mutation MUST change context_fusion_fingerprint / dflash2_identity"
        );
        assert_ne!(fp_norm_exp, fp_fc_exp);
    }

    #[test]
    fn quantize_bf16_linear_to_symq8_parity() {
        let mut seed: u64 = 0x51A5_8001;
        let mut next_f32 = |scale: f32| -> f32 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((seed >> 40) as f32) / 16_777_216.0;
            (2.0 * u - 1.0) * scale
        };

        let m = 4usize;
        let n = 256usize;
        let k = 2304usize;

        // 1. Ineligible shapes/dtypes/biased linears must decline cleanly (`Ok(None)`).
        let tiny_w = MxArray::zeros(&[24, k as i64], Some(DType::BFloat16)).unwrap();
        let tiny_lin = Linear::from_weights(&tiny_w, None).unwrap();
        assert!(
            quantize_bf16_linear_to_symq8(&tiny_lin).unwrap().is_none(),
            "tiny projection (N=24, e.g. sink_proj) must stay dense BF16"
        );
        let conv_w = MxArray::zeros(&[32, k as i64], Some(DType::BFloat16)).unwrap();
        let conv_lin = Linear::from_weights(&conv_w, None).unwrap();
        assert!(
            quantize_bf16_linear_to_symq8(&conv_lin).unwrap().is_none(),
            "tiny projection (N=32, e.g. k_conv_proj/v_conv_proj) must stay dense BF16"
        );
        let f32_w = MxArray::zeros(&[n as i64, k as i64], Some(DType::Float32)).unwrap();
        let f32_lin = Linear::from_weights(&f32_w, None).unwrap();
        assert!(
            quantize_bf16_linear_to_symq8(&f32_lin).unwrap().is_none(),
            "non-BF16 projection must decline"
        );
        let biased_w = MxArray::zeros(&[n as i64, k as i64], Some(DType::BFloat16)).unwrap();
        let bias = MxArray::zeros(&[n as i64], Some(DType::BFloat16)).unwrap();
        let biased_lin = Linear::from_weights(&biased_w, Some(&bias)).unwrap();
        assert!(
            quantize_bf16_linear_to_symq8(&biased_lin)
                .unwrap()
                .is_none(),
            "biased projection must decline"
        );

        // 2. Eligible BF16 `[256, 2304]` linear (including an all-zero 32-element group).
        let mut w_vals: Vec<f32> = (0..n * k).map(|_| next_f32(0.04)).collect();
        for elem in &mut w_vals[0..32] {
            *elem = 0.0;
        }
        let w_bits: Vec<u16> = w_vals
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_bits())
            .collect();
        let w_bf16 = MxArray::from_bfloat16(&w_bits, &[n as i64, k as i64]).unwrap();
        let lin = Linear::from_weights(&w_bf16, None).unwrap();

        let ql = quantize_bf16_linear_to_symq8(&lin)
            .unwrap()
            .expect("eligible [256, 2304] BF16 linear must quantize to SymQ8");
        assert!(ql.is_symmetric_q8(), "expected is_symmetric_q8() == true");
        assert!(ql.get_biases().is_none(), "expected biases == None");
        assert_eq!(ql.get_scales().dtype().unwrap(), DType::Float16);
        assert_eq!(ql.get_weight().dtype().unwrap(), DType::Uint32);
        assert_eq!(
            ql.get_weight().shape().unwrap().as_ref(),
            &[n as i64, (k / 4) as i64]
        );
        assert_eq!(
            ql.get_scales().shape().unwrap().as_ref(),
            &[n as i64, (k / 32) as i64]
        );

        // Chunked path (non-divisible chunk size 100 -> 100 + 100 + 56) must be bit-identical.
        let ql_chunked = quantize_bf16_linear_to_symq8_with_chunk(&lin, 100)
            .unwrap()
            .expect("chunked [256, 2304] SymQ8 quantization must succeed");
        assert_eq!(
            ql_chunked.get_weight().to_uint32().unwrap().as_ref(),
            ql.get_weight().to_uint32().unwrap().as_ref(),
            "chunked weight packing must be bit-identical to single-shot"
        );
        assert_eq!(
            ql_chunked
                .get_scales()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .as_ref(),
            ql.get_scales()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .as_ref(),
            "chunked F16 scales must be bit-identical to single-shot"
        );

        // 3. Parity against FP32 `X @ W^T` on `[4, 2304] @ [256, 2304]^T`.
        let x_vals: Vec<f32> = (0..m * k).map(|_| next_f32(0.35)).collect();
        let x_bits: Vec<u16> = x_vals
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_bits())
            .collect();
        let x_bf16 = MxArray::from_bfloat16(&x_bits, &[m as i64, k as i64]).unwrap();

        let ref_f32 = x_bf16
            .astype(DType::Float32)
            .unwrap()
            .matmul(
                &w_bf16
                    .astype(DType::Float32)
                    .unwrap()
                    .transpose(None)
                    .unwrap(),
            )
            .unwrap()
            .to_float32()
            .unwrap();

        let got_fwd = ql
            .forward(&x_bf16)
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        let max_abs_fwd = got_fwd
            .as_ref()
            .iter()
            .zip(ref_f32.as_ref().iter())
            .map(|(&g, &w)| {
                assert!(g.is_finite(), "non-finite SymQ8 forward output: {g}");
                (g - w).abs()
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_fwd < 2e-2,
            "SymQ8 forward max abs diff {max_abs_fwd:.6} >= 2e-2"
        );

        if crate::kernels::inkling2::sm80_available() {
            let got_lm = crate::kernels::inkling2::lm_head_softcap_cuda(
                &x_bf16,
                ql.get_weight(),
                Some(ql.get_scales()),
                ql.get_biases(),
                n as i32,
                0.0,
                crate::kernels::inkling2::LmHeadOutMode::LogitsF32,
            )
            .unwrap()
            .expect("lm_head_softcap_cuda must accept SymQ8 projection on SM >= 8.0");
            let got_lm_f = got_lm.to_float32().unwrap();
            let max_abs_lm = got_lm_f
                .as_ref()
                .iter()
                .zip(ref_f32.as_ref().iter())
                .map(|(&g, &w)| {
                    assert!(g.is_finite(), "non-finite SymQ8 lm_head output: {g}");
                    (g - w).abs()
                })
                .fold(0.0f32, f32::max);
            assert!(
                max_abs_lm < 2e-2,
                "SymQ8 lm_head_softcap_cuda (uncapped) max abs diff {max_abs_lm:.6} >= 2e-2"
            );

            let softcap = 30.0f32;
            let got_cap = crate::kernels::inkling2::lm_head_softcap_cuda(
                &x_bf16,
                ql.get_weight(),
                Some(ql.get_scales()),
                ql.get_biases(),
                n as i32,
                softcap,
                crate::kernels::inkling2::LmHeadOutMode::LogitsF32,
            )
            .unwrap()
            .expect("lm_head_softcap_cuda (softcap=30) must accept SymQ8 projection");
            let got_cap_f = got_cap.to_float32().unwrap();
            let max_abs_cap = got_cap_f
                .as_ref()
                .iter()
                .zip(ref_f32.as_ref().iter())
                .map(|(&g, &w)| {
                    let want = softcap * (w / softcap).tanh();
                    (g - want).abs()
                })
                .fold(0.0f32, f32::max);
            assert!(
                max_abs_cap < 2e-2,
                "SymQ8 lm_head_softcap_cuda (softcap=30) max abs diff {max_abs_cap:.6} >= 2e-2"
            );
        }

        crate::array::clear_cache();
    }

    fn quantize_affine_for_test(
        w_f32: &MxArray,
        group_size: i32,
        bits: i32,
        f16_sidecars: bool,
    ) -> (MxArray, MxArray, MxArray) {
        let mut q = std::ptr::null_mut();
        let mut s = std::ptr::null_mut();
        let mut b = std::ptr::null_mut();
        let mode_c = CString::new("affine").unwrap();
        unsafe {
            mlx_sys::mlx_quantize(
                w_f32.as_raw_ptr(),
                group_size,
                bits,
                mode_c.as_ptr(),
                &mut q,
                &mut s,
                &mut b,
            );
        }
        let wq = MxArray::from_handle(q, "test_q").unwrap();
        let mut sq = MxArray::from_handle(s, "test_s").unwrap();
        let mut bq = MxArray::from_handle(b, "test_b").unwrap();
        if f16_sidecars {
            sq = sq.astype(DType::Float16).unwrap();
            bq = bq.astype(DType::Float16).unwrap();
        }
        (wq, sq, bq)
    }

    #[test]
    fn small_tensors_and_message_gates_dequantize_1d_2d_affine_mxfp_nvfp4_kquant_and_reject_raw_u32()
     {
        use crate::models::quant_dispatch::{KQuantLayout, kquant_tile_rows};
        use crate::utils::gguf_kquant::{KQuantFormat, KQuantRepacker, KQuantScales};

        let default_plq = PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Affine,
            layout: KQuantLayout::RowMajor,
            input_amax: None,
        };

        // 1. 1-D quantized norm & clip (Affine with F16 scales/biases -> must expand_dims(0), cast s/b to F32, squeeze(0))
        let norm_ref: Vec<f32> = (0..64).map(|i| 0.5 + (i as f32) * 0.015).collect();
        let norm_2d = MxArray::from_float32(&norm_ref, &[1, 64]).unwrap();
        let (nw, ns, nb) = quantize_affine_for_test(&norm_2d, 32, 8, true);
        let mut params = HashMap::new();
        params.insert(
            "embed_norm.weight".to_string(),
            nw.squeeze(Some(&[0])).unwrap(),
        );
        params.insert(
            "embed_norm.scales".to_string(),
            ns.squeeze(Some(&[0])).unwrap(),
        );
        params.insert(
            "embed_norm.biases".to_string(),
            nb.squeeze(Some(&[0])).unwrap(),
        );

        let got_norm =
            load_dense_or_dequant_f32(&params, "embed_norm", &HashMap::new(), default_plq).unwrap();
        assert_eq!(got_norm.dtype().unwrap(), DType::Float32);
        assert_eq!(got_norm.shape().unwrap().as_ref(), &[64]);
        let got_norm_v = got_norm.to_float32().unwrap();
        let max_err = got_norm_v
            .iter()
            .zip(norm_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-2, "1-D affine dequant max_err={max_err}");

        // 2. Quantized 2-D message gates across Affine (4/5/8-bit), MXFP4, MXFP8, NVFP4 (+ .scale / .global_scale),
        //    and K-quants (RowMajor + Tiled64 Q4K, Q5K, Q6K, Q3K, Q2K, IQ4NL).
        let (rows, cols) = (64usize, 256usize);
        let gate_ref: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.07).sin() * 0.5)
            .collect();
        let gate_arr = MxArray::from_float32(&gate_ref, &[rows as i64, cols as i64]).unwrap();

        for (bits, gs) in [(4, 32), (5, 32), (8, 32)] {
            let (wq, sq, bq) = quantize_affine_for_test(&gate_arr, gs, bits, true);
            let mut p = HashMap::new();
            p.insert("g.weight".to_string(), wq);
            p.insert("g.scales".to_string(), sq);
            p.insert("g.biases".to_string(), bq);
            let mut plqs = HashMap::new();
            plqs.insert(
                "g".to_string(),
                PerLayerQuant {
                    bits,
                    group_size: gs,
                    mode: PerLayerMode::Affine,
                    layout: KQuantLayout::RowMajor,
                    input_amax: None,
                },
            );
            let out = load_dense_or_dequant_tensor(&p, "g", &plqs, default_plq, true).unwrap();
            assert_eq!(out.dtype().unwrap(), DType::Float32);
            assert_eq!(out.shape().unwrap().as_ref(), &[rows as i64, cols as i64]);
        }

        // MXFP4, MXFP8, NVFP4 (with .global_scale = 5376.0 -> outer_scale = 2.0)
        for (mode, gs, bits, w_last, s_last, w_word, s_byte) in [
            (
                PerLayerMode::Mxfp4,
                32,
                4,
                cols / 8,
                cols / 32,
                0x2222_2222u32,
                127u8,
            ),
            (
                PerLayerMode::Mxfp8,
                32,
                8,
                cols / 4,
                cols / 32,
                0x3838_3838u32,
                127u8,
            ),
            (
                PerLayerMode::Nvfp4,
                16,
                4,
                cols / 8,
                cols / 16,
                0x2222_2222u32,
                0x38u8,
            ),
        ] {
            let mut p = HashMap::new();
            p.insert(
                "g.weight".to_string(),
                MxArray::from_uint32(&vec![w_word; rows * w_last], &[rows as i64, w_last as i64])
                    .unwrap(),
            );
            p.insert(
                "g.scales".to_string(),
                MxArray::from_uint8(&vec![s_byte; rows * s_last], &[rows as i64, s_last as i64])
                    .unwrap(),
            );
            if mode == PerLayerMode::Nvfp4 {
                p.insert(
                    "g.global_scale".to_string(),
                    MxArray::from_float32(&[5376.0], &[1]).unwrap(), // / 2688 = 2.0
                );
            }
            let mut plqs = HashMap::new();
            plqs.insert(
                "g".to_string(),
                PerLayerQuant {
                    bits,
                    group_size: gs,
                    mode,
                    layout: KQuantLayout::RowMajor,
                    input_amax: None,
                },
            );
            let out = load_dense_or_dequant_tensor(&p, "g", &plqs, default_plq, true).unwrap();
            assert_eq!(out.dtype().unwrap(), DType::Float32);
            let vals = out.to_float32().unwrap();
            let want = if mode == PerLayerMode::Nvfp4 {
                2.0
            } else {
                1.0
            };
            assert!(
                vals.iter().all(|&v| (v - want).abs() < 1e-3),
                "{mode:?}: expected {want}, got {}",
                vals[0]
            );
        }

        // K-quants: RowMajor AND pre-saved Tiled64 (catches K-quant Tiled64 trap on loop_ctrl.down_proj!)
        for (fmt, plq_mode) in [
            (KQuantFormat::Q4K, PerLayerMode::Q4K),
            (KQuantFormat::Q5K, PerLayerMode::Q5K),
            (KQuantFormat::Q6K, PerLayerMode::Q6K),
            (KQuantFormat::Q3K, PerLayerMode::Q3K),
            (KQuantFormat::Q2K, PerLayerMode::Q2K),
            (KQuantFormat::IQ4NL, PerLayerMode::IQ4NL),
        ] {
            let blocks_per_row = cols / fmt.block_size();
            let mut raw = vec![0u8; rows * blocks_per_row * fmt.block_bytes()];
            for (idx, byte) in raw.iter_mut().enumerate() {
                // Keep all bytes < 0x58 so FP16 superblock scale headers (d/dmin) stay strictly finite.
                *byte = ((idx * 37 + 13) % 0x58) as u8;
            }
            let mut repacker = KQuantRepacker::new(fmt, cols, rows).unwrap();
            repacker.push_rows(&raw, rows).unwrap();
            let arrays = repacker.finish();
            let w_rm =
                MxArray::from_uint32(&arrays.weight, &[rows as i64, fmt.weight_cols(cols) as i64])
                    .unwrap();
            let s_rm = match &arrays.scales {
                KQuantScales::Unsigned(v) => {
                    MxArray::from_uint8(v, &[rows as i64, fmt.scales_cols(cols) as i64]).unwrap()
                }
                KQuantScales::Signed(v) => {
                    MxArray::from_int8(v, &[rows as i64, fmt.scales_cols(cols) as i64]).unwrap()
                }
            };
            let b_rm =
                MxArray::from_float16(&arrays.biases, &[rows as i64, fmt.biases_cols(cols) as i64])
                    .unwrap();

            let mut p_rm = HashMap::new();
            p_rm.insert("g.weight".to_string(), w_rm.clone());
            p_rm.insert("g.scales".to_string(), s_rm.clone());
            p_rm.insert("g.biases".to_string(), b_rm.clone());
            let mut plqs_rm = HashMap::new();
            plqs_rm.insert(
                "g".to_string(),
                PerLayerQuant {
                    bits: fmt.bits() as i32,
                    group_size: fmt.group_size() as i32,
                    mode: plq_mode,
                    layout: KQuantLayout::RowMajor,
                    input_amax: None,
                },
            );
            let out_rm = load_dense_or_dequant_tensor(&p_rm, "g", &plqs_rm, default_plq, true)
                .unwrap()
                .to_float32()
                .unwrap();

            // Now tile to Tiled64 and verify load_dense_or_dequant_tensor untiles bit-for-bit!
            let kq = kquant_mode_params_for_scales(plq_mode, s_rm.dtype().unwrap()).unwrap();
            let w_t64 = kquant_tile_rows(&w_rm, i64::from(kq.bits)).unwrap();
            let s_t64 =
                kquant_tile_rows(&s_rm, i64::from(kq.scale_entries_per_super_block())).unwrap();
            let b_t64 =
                kquant_tile_rows(&b_rm, i64::from(kq.bias_entries_per_super_block)).unwrap();
            let mut p_t64 = HashMap::new();
            p_t64.insert("g.weight".to_string(), w_t64);
            p_t64.insert("g.scales".to_string(), s_t64);
            p_t64.insert("g.biases".to_string(), b_t64);
            let mut plqs_t64 = HashMap::new();
            plqs_t64.insert(
                "g".to_string(),
                PerLayerQuant {
                    bits: fmt.bits() as i32,
                    group_size: fmt.group_size() as i32,
                    mode: plq_mode,
                    layout: KQuantLayout::Tiled64,
                    input_amax: None,
                },
            );
            let out_t64 = load_dense_or_dequant_tensor(&p_t64, "g", &plqs_t64, default_plq, true)
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(
                out_rm.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                out_t64.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{fmt:?}: Tiled64 untile must match RowMajor bit-for-bit"
            );
        }

        // 3. Mixed Q8_0 down_proj + BF16 gain/forget widens cleanly to Float32 via Bug 1
        let d_small = 32i64;
        let r_small = 8i64;
        let down_f32 = MxArray::ones(&[r_small, 2 * d_small], Some(DType::Float32)).unwrap();
        let (dw, ds, db) = quantize_affine_for_test(&down_f32, 32, 8, true);
        let mut p_mix = HashMap::new();
        p_mix.insert("c.down_proj.weight".to_string(), dw);
        p_mix.insert("c.down_proj.scales".to_string(), ds);
        p_mix.insert("c.down_proj.biases".to_string(), db);
        p_mix.insert(
            "c.gain_proj.weight".to_string(),
            MxArray::ones(&[d_small, r_small], Some(DType::BFloat16)).unwrap(),
        );
        p_mix.insert(
            "c.forget_proj.weight".to_string(),
            MxArray::ones(&[d_small, r_small], Some(DType::BFloat16)).unwrap(),
        );
        let ctrl = LoopController::new(
            load_dense_or_dequant_tensor(&p_mix, "c.down_proj", &HashMap::new(), default_plq, true)
                .unwrap(),
            load_dense_or_dequant_tensor(&p_mix, "c.gain_proj", &HashMap::new(), default_plq, true)
                .unwrap(),
            load_dense_or_dequant_tensor(
                &p_mix,
                "c.forget_proj",
                &HashMap::new(),
                default_plq,
                true,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(ctrl.down_proj_w.dtype().unwrap(), DType::Float32);
        assert_eq!(ctrl.gain_forget_w.dtype().unwrap(), DType::Float32);

        // 4. Dense F16 conv normalizes to Float32 (avoiding Metal sconv_k4 crash)
        let mut p_f16 = HashMap::new();
        p_f16.insert(
            "conv.weight".to_string(),
            MxArray::ones(&[16, 4], Some(DType::Float16)).unwrap(),
        );
        let conv_f32 =
            load_dense_or_dequant_f32(&p_f16, "conv", &HashMap::new(), default_plq).unwrap();
        assert_eq!(conv_f32.dtype().unwrap(), DType::Float32);

        // 5. Orphaned raw Uint32 without `.scales` fails loud via ensure_dense_weight_floating
        let mut p_bad = HashMap::new();
        p_bad.insert(
            "orphan.weight".to_string(),
            MxArray::zeros(&[8, 8], Some(DType::Uint32)).unwrap(),
        );
        assert!(
            load_dense_or_dequant_f32(&p_bad, "orphan", &HashMap::new(), default_plq).is_err(),
            "raw Uint32 without .scales must fail loud"
        );
    }

    #[test]
    fn load_expert_switch_proj_and_split_gate_up_fusion_matrix() {
        use crate::models::quant_dispatch::KQuantLayout;
        use crate::utils::gguf_kquant::{KQuantFormat, KQuantRepacker, KQuantScales};

        let e = 2i64;
        let inter = 16i64;
        let d = 64i64;
        let default_plq = PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Affine,
            layout: KQuantLayout::RowMajor,
            input_amax: None,
        };
        let empty_plq = HashMap::new();

        // 1. load_expert_switch_proj: Dense F16 + .scale normalizes to BF16 + keeps outer_scale
        {
            let mut p = HashMap::new();
            p.insert(
                "l.mlp.experts.down_proj.weight".to_string(),
                MxArray::ones(&[e, d, inter], Some(DType::Float16)).unwrap(),
            );
            p.insert(
                "l.mlp.experts.down_proj.scale".to_string(),
                MxArray::from_float32(&[0.5, 2.0], &[e, 1]).unwrap(),
            );
            let proj =
                load_expert_switch_proj(&p, "l.mlp.experts.down_proj", &empty_plq, default_plq)
                    .unwrap();
            assert!(!proj.is_quantized());
            assert_eq!(proj.weight_ref().dtype().unwrap(), DType::BFloat16);
            assert_eq!(
                proj.outer_scale().unwrap().to_float32().unwrap().as_ref(),
                &[0.5f32, 2.0]
            );
        }

        // 2. load_expert_switch_proj: MXFP8 + .scale, MXFP4 + .scale, unscaled NVFP4, NVFP4 + .global_scale, NVFP4 + .weight_scale_2
        {
            // MXFP8 + .scale (previously fell through to Affine!)
            let mut p = HashMap::new();
            p.insert(
                "x.weight".to_string(),
                MxArray::from_uint32(
                    &vec![0x3838_3838u32; (e * inter * (d / 4)) as usize],
                    &[e, inter, d / 4],
                )
                .unwrap(),
            );
            p.insert(
                "x.scales".to_string(),
                MxArray::from_uint8(
                    &vec![127u8; (e * inter * (d / 32)) as usize],
                    &[e, inter, d / 32],
                )
                .unwrap(),
            );
            p.insert(
                "x.scale".to_string(),
                MxArray::from_float32(&[1.25, 0.75], &[e]).unwrap(),
            );
            let proj = load_expert_switch_proj(&p, "x", &empty_plq, default_plq).unwrap();
            let SwitchProj::Quantized(q) = proj else {
                panic!("expected quantized")
            };
            assert_eq!(q.mode(), MXFP8_MODE);
            assert_eq!(
                q.outer_scale().unwrap().to_float32().unwrap().as_ref(),
                &[1.25f32, 0.75]
            );

            // Unscaled NVFP4 (previously fell through to Affine!)
            let mut p_nv = HashMap::new();
            p_nv.insert(
                "x.weight".to_string(),
                MxArray::from_uint32(
                    &vec![0x2222_2222u32; (e * inter * (d / 8)) as usize],
                    &[e, inter, d / 8],
                )
                .unwrap(),
            );
            p_nv.insert(
                "x.scales".to_string(),
                MxArray::from_uint8(
                    &vec![0x38u8; (e * inter * (d / 16)) as usize],
                    &[e, inter, d / 16],
                )
                .unwrap(),
            );
            let proj_unscaled =
                load_expert_switch_proj(&p_nv, "x", &empty_plq, default_plq).unwrap();
            let SwitchProj::Quantized(q_un) = proj_unscaled else {
                panic!("expected quantized")
            };
            assert_eq!(q_un.mode(), NVFP4_MODE);
            assert!(q_un.outer_scale().is_none());

            // NVFP4 + .global_scale (/ 2688.0)
            let mut p_gs = p_nv.clone();
            p_gs.insert(
                "x.global_scale".to_string(),
                MxArray::from_float32(&[2688.0, 5376.0], &[e]).unwrap(),
            );
            let SwitchProj::Quantized(q_gs) =
                load_expert_switch_proj(&p_gs, "x", &empty_plq, default_plq).unwrap()
            else {
                panic!("expected quantized")
            };
            assert_eq!(q_gs.mode(), NVFP4_MODE);
            assert_eq!(
                q_gs.outer_scale().unwrap().to_float32().unwrap().as_ref(),
                &[1.0f32, 2.0]
            );

            // NVFP4 + .weight_scale_2 (direct, NO / 2688.0)
            let mut p_ws2 = p_nv.clone();
            p_ws2.insert(
                "x.weight_scale_2".to_string(),
                MxArray::from_float32(&[0.25, 0.5], &[e]).unwrap(),
            );
            let SwitchProj::Quantized(q_ws2) =
                load_expert_switch_proj(&p_ws2, "x", &empty_plq, default_plq).unwrap()
            else {
                panic!("expected quantized")
            };
            assert_eq!(
                q_ws2.outer_scale().unwrap().to_float32().unwrap().as_ref(),
                &[0.25f32, 0.5]
            );
        }

        // 3. Split gate_proj + up_proj FUSIBLE cases -> ExpertGateUp::Fused + interleaved == false (overriding config true!)
        // (a) Dense F16 split -> fused BF16
        {
            let mut p = HashMap::new();
            p.insert(
                "layers.0.mlp.experts.gate_proj.weight".to_string(),
                MxArray::ones(&[e, inter, d], Some(DType::Float16)).unwrap(),
            );
            p.insert(
                "layers.0.mlp.experts.up_proj.weight".to_string(),
                MxArray::ones(&[e, inter, d], Some(DType::Float16)).unwrap(),
            );
            let (gu, interleaved) =
                load_expert_gate_up(&p, "layers.0", &empty_plq, default_plq, true).unwrap();
            assert!(
                !interleaved,
                "split gate+up must override interleaved to false"
            );
            let ExpertGateUp::Fused(SwitchProj::Standard(sl)) = gu else {
                panic!("expected fused dense")
            };
            assert_eq!(sl.weight_ref().dtype().unwrap(), DType::BFloat16);
            assert_eq!(
                sl.weight_ref().shape().unwrap().as_ref(),
                &[e, 2 * inter, d]
            );
        }

        // (b) Symmetric Q8_0 split -> fuses BEFORE compaction, then compacts to symmetric_q8!
        {
            let make_sym_q8 = |p: &mut HashMap<String, MxArray>, name: &str| {
                let s = MxArray::full(
                    &[e, inter, d / 32],
                    napi::Either::A(0.25f64),
                    Some(DType::Float16),
                )
                .unwrap();
                let b = s
                    .astype(DType::Float32)
                    .unwrap()
                    .mul_scalar(-128.0)
                    .unwrap()
                    .astype(DType::Float16)
                    .unwrap();
                p.insert(
                    format!("{name}.weight"),
                    MxArray::from_uint32(
                        &vec![0x8080_8080u32; (e * inter * (d / 4)) as usize],
                        &[e, inter, d / 4],
                    )
                    .unwrap(),
                );
                p.insert(format!("{name}.scales"), s);
                p.insert(format!("{name}.biases"), b);
            };
            let mut p = HashMap::new();
            make_sym_q8(&mut p, "layers.0.mlp.experts.gate_proj");
            make_sym_q8(&mut p, "layers.0.mlp.experts.up_proj");
            let (gu, interleaved) =
                load_expert_gate_up(&p, "layers.0", &empty_plq, default_plq, true).unwrap();
            assert!(!interleaved);
            let ExpertGateUp::Fused(SwitchProj::Quantized(q)) = gu else {
                panic!("expected fused sym Q8")
            };
            assert!(
                q.is_symmetric_q8(),
                "must compact to symmetric_q8 after fusion"
            );
            assert_eq!(
                q.get_weight().shape().unwrap().as_ref(),
                &[e, 2 * inter, d / 4]
            );
        }

        // (c) NVFP4 equal `.scale` -> Fused; distinct `.scale` -> Split!
        {
            let mut p = HashMap::new();
            for proj_name in ["gate_proj", "up_proj"] {
                p.insert(
                    format!("layers.0.mlp.experts.{proj_name}.weight"),
                    MxArray::from_uint32(
                        &vec![0x2222_2222u32; (e * inter * (d / 8)) as usize],
                        &[e, inter, d / 8],
                    )
                    .unwrap(),
                );
                p.insert(
                    format!("layers.0.mlp.experts.{proj_name}.scales"),
                    MxArray::from_uint8(
                        &vec![0x38u8; (e * inter * (d / 16)) as usize],
                        &[e, inter, d / 16],
                    )
                    .unwrap(),
                );
                p.insert(
                    format!("layers.0.mlp.experts.{proj_name}.scale"),
                    MxArray::from_float32(&[0.5, 1.5], &[e]).unwrap(),
                );
            }
            let (gu_eq, il_eq) =
                load_expert_gate_up(&p, "layers.0", &empty_plq, default_plq, true).unwrap();
            assert!(!il_eq);
            assert!(gu_eq.is_fused(), "equal NVFP4 .scale must fuse");

            // Change up_proj.scale so gate != up -> MUST stay Split!
            p.insert(
                "layers.0.mlp.experts.up_proj.scale".to_string(),
                MxArray::from_float32(&[0.5, 2.5], &[e]).unwrap(),
            );
            let (gu_diff, il_diff) =
                load_expert_gate_up(&p, "layers.0", &empty_plq, default_plq, true).unwrap();
            assert!(!il_diff);
            assert!(!gu_diff.is_fused(), "distinct NVFP4 .scale must stay Split");
        }

        // (d) Mixed K-quant Q5_K gate + Q4_K up -> MUST stay Split!
        {
            let k_kq = 256usize;
            let make_kq = |fmt: KQuantFormat| {
                let rows = (e * inter) as usize;
                let raw = vec![0u8; rows * (k_kq / fmt.block_size()) * fmt.block_bytes()];
                let mut rep = KQuantRepacker::new(fmt, k_kq, rows).unwrap();
                rep.push_rows(&raw, rows).unwrap();
                let a = rep.finish();
                let w = MxArray::from_uint32(&a.weight, &[e, inter, fmt.weight_cols(k_kq) as i64])
                    .unwrap();
                let s = match a.scales {
                    KQuantScales::Unsigned(v) => {
                        MxArray::from_uint8(&v, &[e, inter, fmt.scales_cols(k_kq) as i64]).unwrap()
                    }
                    KQuantScales::Signed(v) => {
                        MxArray::from_int8(&v, &[e, inter, fmt.scales_cols(k_kq) as i64]).unwrap()
                    }
                };
                let b = MxArray::from_float16(&a.biases, &[e, inter, fmt.biases_cols(k_kq) as i64])
                    .unwrap();
                (w, s, b)
            };
            let (gw, gs, gb) = make_kq(KQuantFormat::Q5K);
            let (uw, us, ub) = make_kq(KQuantFormat::Q4K);
            let mut p = HashMap::new();
            p.insert("layers.0.mlp.experts.gate_proj.weight".to_string(), gw);
            p.insert("layers.0.mlp.experts.gate_proj.scales".to_string(), gs);
            p.insert("layers.0.mlp.experts.gate_proj.biases".to_string(), gb);
            p.insert("layers.0.mlp.experts.up_proj.weight".to_string(), uw);
            p.insert("layers.0.mlp.experts.up_proj.scales".to_string(), us);
            p.insert("layers.0.mlp.experts.up_proj.biases".to_string(), ub);
            let mut plqs = HashMap::new();
            plqs.insert(
                "layers.0.mlp.experts.gate_proj".to_string(),
                PerLayerQuant {
                    bits: 5,
                    group_size: 32,
                    mode: PerLayerMode::Q5K,
                    layout: KQuantLayout::RowMajor,
                    input_amax: None,
                },
            );
            plqs.insert(
                "layers.0.mlp.experts.up_proj".to_string(),
                PerLayerQuant {
                    bits: 4,
                    group_size: 32,
                    mode: PerLayerMode::Q4K,
                    layout: KQuantLayout::RowMajor,
                    input_amax: None,
                },
            );
            let (gu, il) = load_expert_gate_up(&p, "layers.0", &plqs, default_plq, true).unwrap();
            assert!(!il);
            assert!(!gu.is_fused(), "mixed Q5K+Q4K must stay Split");
        }
    }

    #[test]
    fn cross_layout_safetensors_equivalence_release_omlx_mlxvlm_mlxlm() {
        use crate::models::inkling2::config::Inkling2LoopConfig;
        use crate::models::quant_dispatch::KQuantLayout;

        // Tiny synthetic Inkling2: 2 physical layers (layer 0 = local SWA, layer 1 = global + core loop),
        // D=16, n_heads=2, n_kv=1, head_dim=8, d_rel=4, rel_extent=8, K_conv=4, E=2, top_k=1, inter=8, rank=4.
        let d = 16usize;
        let n_heads = 2usize;
        let n_kv = 1usize;
        let head_dim = 8usize;
        let d_rel = 4usize;
        let rel_extent = 8usize;
        let k_conv = 4usize;
        let e = 2usize;
        let inter = 8usize;
        let rank = 4usize;
        let vocab = 32usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv * head_dim;
        let r_dim = n_heads * d_rel;
        let taps = n_kv * k_conv;

        let config = Inkling2Config {
            model_type: "inkling2_loop".to_string(),
            hidden_size: d,
            num_hidden_layers: 3,
            num_physical_layers: Some(2),
            num_attention_heads: n_heads,
            num_key_value_heads: n_kv,
            head_dim,
            value_head_dim: None,
            vocab_size: vocab,
            padded_vocab_size: Some(vocab),
            unpadded_vocab_size: Some(vocab),
            rms_norm_eps: 1e-6,
            max_position_embeddings: 64,
            sliding_window: 16,
            sliding_window_pattern: vec![true, false],
            local_layer_ids: vec![0],
            sconv_kernel_size: k_conv,
            n_layers_per_attn_res_block: 1,
            d_rel,
            rel_extent,
            qk_l2_clip_threshold: 22.6,
            rel_l2_clip_threshold: 8.0,
            log_scaling_n_floor: 8192,
            log_scaling_alpha: 0.1,
            num_experts: e,
            num_experts_per_tok: 1,
            moe_intermediate_size: Some(inter),
            intermediate_size: Some(inter),
            route_scale: 1.0,
            mlp_gate_softcap: 4.0,
            mlp_up_softcap: 25.0,
            final_logit_softcapping: 30.0,
            loop_config: Some(Inkling2LoopConfig {
                core_start: 1,
                core_end: 2,
                num_passes: 2,
                gate_rank: rank,
                is_varloop: false,
            }),
            vision_config: None,
            dspark_config: None,
            eos_token_id: 2,
            pad_token_id: 0,
            gate_up_interleaved: false,
        };

        let mk = |seed: usize, shape: &[i64], dt: DType| -> MxArray {
            let n: usize = shape.iter().map(|&v| v as usize).product();
            let vals: Vec<f32> = (0..n)
                .map(|i| (((i + seed * 17) as f32) * 0.13).sin() * 0.25)
                .collect();
            let arr = MxArray::from_float32(&vals, shape).unwrap();
            if dt == DType::Float32 {
                arr
            } else {
                arr.astype(dt).unwrap()
            }
        };

        // Shared ground-truth canonical tensors (BF16 projections, F32 norms/queries/convs/clips)
        let embed_w = mk(1, &[vocab as i64, d as i64], DType::BFloat16);
        let lm_head_w = mk(2, &[vocab as i64, d as i64], DType::BFloat16);
        let embed_norm_w = mk(3, &[d as i64], DType::Float32);
        let final_norm_w = mk(4, &[d as i64], DType::Float32);
        let out_res_q = mk(5, &[d as i64], DType::Float32);

        // Layer 0 (local) & Layer 1 (global + loop core 0)
        let l0_inp_norm = mk(10, &[d as i64], DType::Float32);
        let l0_ffn_norm = mk(11, &[d as i64], DType::Float32);
        let l0_ffn_q = mk(12, &[d as i64], DType::Float32);
        let l0_q = mk(13, &[q_dim as i64, d as i64], DType::BFloat16);
        let l0_k = mk(14, &[kv_dim as i64, d as i64], DType::BFloat16);
        let l0_v = mk(15, &[kv_dim as i64, d as i64], DType::BFloat16);
        let l0_r = mk(16, &[r_dim as i64, d as i64], DType::BFloat16);
        let l0_g = mk(17, &[q_dim as i64, d as i64], DType::BFloat16);
        let l0_s = mk(18, &[n_heads as i64, d as i64], DType::BFloat16);
        let l0_o = mk(19, &[d as i64, q_dim as i64], DType::BFloat16);
        let l0_qc = mk(20, &[n_heads as i64, head_dim as i64], DType::Float32);
        let l0_rc = mk(21, &[n_heads as i64, d_rel as i64], DType::Float32);
        let l0_rb = mk(22, &[d_rel as i64, rel_extent as i64], DType::Float32);
        let l0_kc = mk(23, &[kv_dim as i64, k_conv as i64], DType::Float32);
        let l0_vc = mk(24, &[kv_dim as i64, k_conv as i64], DType::Float32);
        let l0_pc = mk(25, &[d as i64, k_conv as i64], DType::Float32);
        let l0_mpc = mk(26, &[d as i64, k_conv as i64], DType::Float32);
        let l0_router = mk(27, &[e as i64, d as i64], DType::Float32);
        let l0_rbias = mk(28, &[e as i64], DType::Float32);
        let l0_e_gate = mk(29, &[e as i64, inter as i64, d as i64], DType::BFloat16);
        let l0_e_up = mk(30, &[e as i64, inter as i64, d as i64], DType::BFloat16);
        let l0_e_down = mk(31, &[e as i64, d as i64, inter as i64], DType::BFloat16);
        // Interleaved `[E, 2*I, D]` matching `(g0, u0, g1, u1, ...)`
        let interleave = |g: &MxArray, u: &MxArray| -> MxArray {
            MxArray::stack(vec![g, u], Some(2))
                .unwrap()
                .reshape(&[e as i64, (2 * inter) as i64, d as i64])
                .unwrap()
        };
        let l0_e_gu_il = interleave(&l0_e_gate, &l0_e_up);

        // Layer 1 (global + core 0)
        let l1_attn_q = mk(40, &[3, d as i64], DType::Float32);
        let l1_ffn_q = mk(41, &[d as i64], DType::Float32);
        let l1_qn = mk(42, &[d as i64], DType::Float32);
        let l1_kn = mk(43, &[d as i64], DType::Float32);
        let l1_vn = mk(44, &[d as i64], DType::Float32);
        let l1_ffn_norm = mk(45, &[d as i64], DType::Float32);
        let l1_q = mk(46, &[q_dim as i64, d as i64], DType::BFloat16);
        let l1_k = mk(47, &[kv_dim as i64, d as i64], DType::BFloat16);
        let l1_v = mk(48, &[kv_dim as i64, d as i64], DType::BFloat16);
        let l1_r = mk(49, &[r_dim as i64, d as i64], DType::BFloat16);
        let l1_g = mk(50, &[q_dim as i64, d as i64], DType::BFloat16);
        let l1_s = mk(51, &[n_heads as i64, d as i64], DType::BFloat16);
        let l1_o = mk(52, &[d as i64, q_dim as i64], DType::BFloat16);
        let l1_kcp = mk(53, &[taps as i64, d as i64], DType::BFloat16);
        let l1_vcp = mk(54, &[taps as i64, d as i64], DType::BFloat16);
        let l1_qc = mk(55, &[n_heads as i64, head_dim as i64], DType::Float32);
        let l1_rc = mk(56, &[n_heads as i64, d_rel as i64], DType::Float32);
        let l1_rb = mk(57, &[d_rel as i64, rel_extent as i64], DType::Float32);
        // Stored as `[K, C]` newest-to-oldest in SafeTensors!
        let l1_k_static_kc = mk(58, &[k_conv as i64, kv_dim as i64], DType::Float32);
        let l1_v_static_kc = mk(59, &[k_conv as i64, kv_dim as i64], DType::Float32);
        let l1_pc = mk(60, &[d as i64, k_conv as i64], DType::Float32);
        let l1_mpc = mk(61, &[d as i64, k_conv as i64], DType::Float32);
        let l1_router = mk(62, &[e as i64, d as i64], DType::Float32);
        let l1_rbias = mk(63, &[e as i64], DType::Float32);
        let l1_e_gate = mk(64, &[e as i64, inter as i64, d as i64], DType::BFloat16);
        let l1_e_up = mk(65, &[e as i64, inter as i64, d as i64], DType::BFloat16);
        let l1_e_down = mk(66, &[e as i64, d as i64, inter as i64], DType::BFloat16);
        let l1_e_gu_il = interleave(&l1_e_gate, &l1_e_up);

        // Loop message queries `[4, D]` ([q, k, v, mlp]) and 4 controllers (mixed BF16 down + F32 gain/forget!)
        let msg_q4 = mk(70, &[4, d as i64], DType::Float32);
        let ctrl_down: Vec<MxArray> = (0..4)
            .map(|g| mk(80 + g, &[rank as i64, (2 * d) as i64], DType::BFloat16))
            .collect();
        let ctrl_gain: Vec<MxArray> = (0..4)
            .map(|g| mk(90 + g, &[d as i64, rank as i64], DType::Float32))
            .collect();
        let ctrl_forget: Vec<MxArray> = (0..4)
            .map(|g| mk(100 + g, &[d as i64, rank as i64], DType::Float32))
            .collect();

        // Build all 4 raw SafeTensors maps
        // (A) Release (`model.llm.*`)
        let mut raw_release = HashMap::new();
        raw_release.insert("model.llm.embed.weight".into(), embed_w.clone());
        raw_release.insert("model.llm.unembed.weight".into(), lm_head_w.clone());
        raw_release.insert("model.llm.embed_norm.weight".into(), embed_norm_w.clone());
        raw_release.insert("model.llm.norm.weight".into(), final_norm_w.clone());
        raw_release.insert("model.llm.attn_res.output_query".into(), out_res_q.clone());
        raw_release.insert("model.llm.attn_res.mlp_queries.0".into(), l0_ffn_q.clone());
        raw_release.insert(
            "model.llm.attn_res.attn_queries.0".into(),
            l1_attn_q.clone(),
        );
        raw_release.insert("model.llm.attn_res.mlp_queries.1".into(), l1_ffn_q.clone());
        // Release global cross-pass queries: `[2*C, D]` ([q, mlp]) and `[2*C, D]` ([k, v])
        let rel_lq = MxArray::concatenate(
            &msg_q4.slice(&[0, 0], &[1, d as i64]).unwrap(),
            &msg_q4.slice(&[3, 0], &[4, d as i64]).unwrap(),
            0,
        )
        .unwrap();
        let rel_lkv = MxArray::concatenate(
            &msg_q4.slice(&[1, 0], &[2, d as i64]).unwrap(),
            &msg_q4.slice(&[2, 0], &[3, d as i64]).unwrap(),
            0,
        )
        .unwrap();
        raw_release.insert("model.llm.middle_cycle_cross_pass_queries".into(), rel_lq);
        raw_release.insert(
            "model.llm.middle_cycle_cross_pass_kv_queries".into(),
            rel_lkv,
        );
        raw_release.insert(
            "model.llm.layers.0.attn_norm.weight".into(),
            l0_inp_norm.clone(),
        );
        raw_release.insert(
            "model.llm.layers.0.mlp_norm.weight".into(),
            l0_ffn_norm.clone(),
        );
        raw_release.insert("model.llm.layers.0.attn.wq_du.weight".into(), l0_q.clone());
        raw_release.insert("model.llm.layers.0.attn.wk_dv.weight".into(), l0_k.clone());
        raw_release.insert("model.llm.layers.0.attn.wv_dv.weight".into(), l0_v.clone());
        raw_release.insert("model.llm.layers.0.attn.wr_du.weight".into(), l0_r.clone());
        raw_release.insert(
            "model.llm.layers.0.attn.gate_attn.weight".into(),
            l0_g.clone(),
        );
        raw_release.insert(
            "model.llm.layers.0.attn.attn_sink_proj.weight".into(),
            l0_s.clone(),
        );
        raw_release.insert("model.llm.layers.0.attn.wo_ud.weight".into(), l0_o.clone());
        raw_release.insert(
            "model.llm.layers.0.attn.q_l2_clip.scale".into(),
            l0_qc.clone(),
        );
        raw_release.insert(
            "model.llm.layers.0.attn.rel_l2_clip.scale".into(),
            l0_rc.clone(),
        );
        raw_release.insert(
            "model.llm.layers.0.attn.rel_logits_proj.proj".into(),
            l0_rb.clone(),
        );
        raw_release.insert(
            "model.llm.layers.0.attn.k_sconv.weight".into(),
            l0_kc.expand_dims(1).unwrap(),
        );
        raw_release.insert(
            "model.llm.layers.0.attn.v_sconv.weight".into(),
            l0_vc.expand_dims(1).unwrap(),
        );
        raw_release.insert(
            "model.llm.layers.0.attn_sconv.weight".into(),
            l0_pc.expand_dims(1).unwrap(),
        );
        raw_release.insert(
            "model.llm.layers.0.mlp_sconv.weight".into(),
            l0_mpc.expand_dims(1).unwrap(),
        );
        raw_release.insert(
            "model.llm.layers.0.mlp.gate.weight".into(),
            l0_router.clone(),
        );
        raw_release.insert("model.llm.layers.0.mlp.gate.bias".into(), l0_rbias.clone());
        raw_release.insert(
            "model.llm.layers.0.mlp.experts.w13_weight".into(),
            l0_e_gu_il.clone(),
        );
        raw_release.insert(
            "model.llm.layers.0.mlp.experts.w2_weight".into(),
            l0_e_down.clone(),
        );
        raw_release.insert(
            "model.llm.layers.1.q_input_norm.weight".into(),
            l1_qn.clone(),
        );
        raw_release.insert(
            "model.llm.layers.1.k_input_norm.weight".into(),
            l1_kn.clone(),
        );
        raw_release.insert(
            "model.llm.layers.1.v_input_norm.weight".into(),
            l1_vn.clone(),
        );
        raw_release.insert(
            "model.llm.layers.1.mlp_norm.weight".into(),
            l1_ffn_norm.clone(),
        );
        raw_release.insert("model.llm.layers.1.attn.wq_du.weight".into(), l1_q.clone());
        raw_release.insert("model.llm.layers.1.attn.wk_dv.weight".into(), l1_k.clone());
        raw_release.insert("model.llm.layers.1.attn.wv_dv.weight".into(), l1_v.clone());
        raw_release.insert("model.llm.layers.1.attn.wr_du.weight".into(), l1_r.clone());
        raw_release.insert(
            "model.llm.layers.1.attn.gate_attn.weight".into(),
            l1_g.clone(),
        );
        raw_release.insert(
            "model.llm.layers.1.attn.attn_sink_proj.weight".into(),
            l1_s.clone(),
        );
        raw_release.insert("model.llm.layers.1.attn.wo_ud.weight".into(), l1_o.clone());
        raw_release.insert(
            "model.llm.layers.1.attn.k_sconv.weight_proj".into(),
            l1_kcp.clone(),
        );
        raw_release.insert(
            "model.llm.layers.1.attn.v_sconv.weight_proj".into(),
            l1_vcp.clone(),
        );
        raw_release.insert(
            "model.llm.layers.1.attn.q_l2_clip.scale".into(),
            l1_qc.clone(),
        );
        raw_release.insert(
            "model.llm.layers.1.attn.rel_l2_clip.scale".into(),
            l1_rc.clone(),
        );
        raw_release.insert(
            "model.llm.layers.1.attn.rel_logits_proj.proj".into(),
            l1_rb.clone(),
        );
        raw_release.insert(
            "model.llm.layers.1.attn.k_sconv.static_w".into(),
            l1_k_static_kc.clone(),
        );
        raw_release.insert(
            "model.llm.layers.1.attn.v_sconv.static_w".into(),
            l1_v_static_kc.clone(),
        );
        raw_release.insert(
            "model.llm.layers.1.attn_sconv.weight".into(),
            l1_pc.expand_dims(1).unwrap(),
        );
        raw_release.insert(
            "model.llm.layers.1.mlp_sconv.weight".into(),
            l1_mpc.expand_dims(1).unwrap(),
        );
        raw_release.insert(
            "model.llm.layers.1.mlp.gate.weight".into(),
            l1_router.clone(),
        );
        raw_release.insert("model.llm.layers.1.mlp.gate.bias".into(), l1_rbias.clone());
        raw_release.insert(
            "model.llm.layers.1.mlp.experts.w13_weight".into(),
            l1_e_gu_il.clone(),
        );
        raw_release.insert(
            "model.llm.layers.1.mlp.experts.w2_weight".into(),
            l1_e_down.clone(),
        );
        for (i, s) in ["attn", "k", "v", "mlp"].iter().enumerate() {
            raw_release.insert(
                format!("model.llm.layers.1.controllers.{s}.down_weight"),
                ctrl_down[i].clone(),
            );
            raw_release.insert(
                format!("model.llm.layers.1.controllers.{s}.gain_weight"),
                ctrl_gain[i].clone(),
            );
            raw_release.insert(
                format!("model.llm.layers.1.controllers.{s}.forget_weight"),
                ctrl_forget[i].clone(),
            );
        }

        // (B) oMLX (`model.embed.weight`, fused `attn.in_proj`, fused `attn.{q,k,v}_proj`, packed `loop_down`/`loop_gain_forget`)
        let mut raw_omlx = HashMap::new();
        raw_omlx.insert("model.embed.weight".into(), embed_w.clone());
        raw_omlx.insert("model.lm_head.weight".into(), lm_head_w.clone());
        raw_omlx.insert("model.embed_norm.weight".into(), embed_norm_w.clone());
        raw_omlx.insert("model.norm.weight".into(), final_norm_w.clone());
        raw_omlx.insert("model.final_input.query".into(), out_res_q.clone());
        raw_omlx.insert(
            "model.layers.0.attn_norm.weight".into(),
            l0_inp_norm.clone(),
        );
        raw_omlx.insert("model.layers.0.mlp_norm.weight".into(), l0_ffn_norm.clone());
        raw_omlx.insert("model.layers.0.mlp_input.query".into(), l0_ffn_q.clone());
        let l0_in_proj =
            MxArray::concatenate_many(vec![&l0_q, &l0_k, &l0_v, &l0_r, &l0_g, &l0_s], Some(0))
                .unwrap();
        raw_omlx.insert("model.layers.0.attn.in_proj.weight".into(), l0_in_proj);
        raw_omlx.insert("model.layers.0.attn.o_proj.weight".into(), l0_o.clone());
        raw_omlx.insert("model.layers.0.attn.q_scale".into(), l0_qc.clone());
        raw_omlx.insert("model.layers.0.attn.r_scale".into(), l0_rc.clone());
        raw_omlx.insert("model.layers.0.attn.rel_proj".into(), l0_rb.clone());
        raw_omlx.insert("model.layers.0.attn.k_conv".into(), l0_kc.clone());
        raw_omlx.insert("model.layers.0.attn.v_conv".into(), l0_vc.clone());
        raw_omlx.insert("model.layers.0.attn_conv".into(), l0_pc.clone());
        raw_omlx.insert("model.layers.0.mlp_conv".into(), l0_mpc.clone());
        raw_omlx.insert("model.layers.0.mlp.router_weight".into(), l0_router.clone());
        raw_omlx.insert("model.layers.0.mlp.router_bias".into(), l0_rbias.clone());
        raw_omlx.insert(
            "model.layers.0.mlp.experts.w13.weight".into(),
            l0_e_gu_il.clone(),
        );
        raw_omlx.insert(
            "model.layers.0.mlp.experts.w2.weight".into(),
            l0_e_down.clone(),
        );
        raw_omlx.insert("model.layers.1.attn_input.query".into(), l1_attn_q.clone());
        raw_omlx.insert("model.layers.1.mlp_input.query".into(), l1_ffn_q.clone());
        raw_omlx.insert("model.layers.1.q_input_norm.weight".into(), l1_qn.clone());
        raw_omlx.insert("model.layers.1.k_input_norm.weight".into(), l1_kn.clone());
        raw_omlx.insert("model.layers.1.v_input_norm.weight".into(), l1_vn.clone());
        raw_omlx.insert("model.layers.1.mlp_norm.weight".into(), l1_ffn_norm.clone());
        let l1_q_fused =
            MxArray::concatenate_many(vec![&l1_q, &l1_r, &l1_g, &l1_s], Some(0)).unwrap();
        let l1_k_fused = MxArray::concatenate(&l1_k, &l1_kcp, 0).unwrap();
        let l1_v_fused = MxArray::concatenate(&l1_v, &l1_vcp, 0).unwrap();
        raw_omlx.insert("model.layers.1.attn.q_proj.weight".into(), l1_q_fused);
        raw_omlx.insert("model.layers.1.attn.k_proj.weight".into(), l1_k_fused);
        raw_omlx.insert("model.layers.1.attn.v_proj.weight".into(), l1_v_fused);
        raw_omlx.insert("model.layers.1.attn.o_proj.weight".into(), l1_o.clone());
        raw_omlx.insert("model.layers.1.attn.q_scale".into(), l1_qc.clone());
        raw_omlx.insert("model.layers.1.attn.r_scale".into(), l1_rc.clone());
        raw_omlx.insert("model.layers.1.attn.rel_proj".into(), l1_rb.clone());
        raw_omlx.insert(
            "model.layers.1.attn.k_static".into(),
            l1_k_static_kc.clone(),
        );
        raw_omlx.insert(
            "model.layers.1.attn.v_static".into(),
            l1_v_static_kc.clone(),
        );
        raw_omlx.insert("model.layers.1.attn_conv".into(), l1_pc.clone());
        raw_omlx.insert("model.layers.1.mlp_conv".into(), l1_mpc.clone());
        raw_omlx.insert("model.layers.1.mlp.router_weight".into(), l1_router.clone());
        raw_omlx.insert("model.layers.1.mlp.router_bias".into(), l1_rbias.clone());
        raw_omlx.insert(
            "model.layers.1.mlp.experts.w13.weight".into(),
            l1_e_gu_il.clone(),
        );
        raw_omlx.insert(
            "model.layers.1.mlp.experts.w2.weight".into(),
            l1_e_down.clone(),
        );
        raw_omlx.insert("model.layers.1.loop_message.query".into(), msg_q4.clone());
        let omlx_down_t: Vec<MxArray> = ctrl_down
            .iter()
            .map(|a| a.transpose(None).unwrap())
            .collect();
        raw_omlx.insert(
            "model.layers.1.loop_down".into(),
            MxArray::stack(omlx_down_t.iter().collect(), Some(0)).unwrap(),
        );
        let omlx_gf: Vec<MxArray> = (0..4)
            .map(|i| {
                MxArray::concatenate(
                    &ctrl_gain[i].transpose(None).unwrap(),
                    &ctrl_forget[i].transpose(None).unwrap(),
                    1,
                )
                .unwrap()
            })
            .collect();
        raw_omlx.insert(
            "model.layers.1.loop_gain_forget".into(),
            MxArray::stack(omlx_gf.iter().collect(), Some(0)).unwrap(),
        );

        // (C) mlx-vlm (`model.language_model.*`, `[C, K, 1]` convs, `{q,k,v,mlp}_message_gate.*`)
        let mut raw_mlxvlm = HashMap::new();
        raw_mlxvlm.insert(
            "model.language_model.embed_tokens.weight".into(),
            embed_w.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.lm_head.weight".into(),
            lm_head_w.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.embed_norm.weight".into(),
            embed_norm_w.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.norm.weight".into(),
            final_norm_w.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.final_input.query".into(),
            out_res_q.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.input_layernorm.weight".into(),
            l0_inp_norm.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.post_attention_layernorm.weight".into(),
            l0_ffn_norm.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.mlp_input.query".into(),
            l0_ffn_q.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.self_attn.q_proj.weight".into(),
            l0_q.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.self_attn.k_proj.weight".into(),
            l0_k.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.self_attn.v_proj.weight".into(),
            l0_v.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.self_attn.r_proj.weight".into(),
            l0_r.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.self_attn.gate_proj.weight".into(),
            l0_g.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.self_attn.sink_proj.weight".into(),
            l0_s.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.self_attn.o_proj.weight".into(),
            l0_o.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.self_attn.q_l2_clip.scale".into(),
            l0_qc.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.self_attn.rel_l2_clip.scale".into(),
            l0_rc.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.self_attn.rel_logits_proj.proj".into(),
            l0_rb.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.self_attn.k_sconv.conv1d.weight".into(),
            l0_kc.expand_dims(2).unwrap(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.self_attn.v_sconv.conv1d.weight".into(),
            l0_vc.expand_dims(2).unwrap(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.post_attention_convolution.conv1d.weight".into(),
            l0_pc.expand_dims(2).unwrap(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.post_mlp_convolution.conv1d.weight".into(),
            l0_mpc.expand_dims(2).unwrap(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.mlp.gate.weight".into(),
            l0_router.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.mlp.gate.e_score_correction_bias".into(),
            l0_rbias.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.mlp.experts.gate_up_proj.weight".into(),
            l0_e_gu_il.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.0.mlp.experts.down_proj.weight".into(),
            l0_e_down.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.attention_input.query".into(),
            l1_attn_q.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.mlp_input.query".into(),
            l1_ffn_q.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.q_input_layernorm.weight".into(),
            l1_qn.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.k_input_layernorm.weight".into(),
            l1_kn.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.v_input_layernorm.weight".into(),
            l1_vn.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.post_attention_layernorm.weight".into(),
            l1_ffn_norm.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.self_attn.q_proj.weight".into(),
            l1_q.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.self_attn.k_proj.weight".into(),
            l1_k.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.self_attn.v_proj.weight".into(),
            l1_v.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.self_attn.r_proj.weight".into(),
            l1_r.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.self_attn.gate_proj.weight".into(),
            l1_g.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.self_attn.sink_proj.weight".into(),
            l1_s.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.self_attn.o_proj.weight".into(),
            l1_o.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.self_attn.k_sconv.weight_proj.weight".into(),
            l1_kcp.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.self_attn.v_sconv.weight_proj.weight".into(),
            l1_vcp.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.self_attn.q_l2_clip.scale".into(),
            l1_qc.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.self_attn.rel_l2_clip.scale".into(),
            l1_rc.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.self_attn.rel_logits_proj.proj".into(),
            l1_rb.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.self_attn.k_sconv.static_weight".into(),
            l1_k_static_kc.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.self_attn.v_sconv.static_weight".into(),
            l1_v_static_kc.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.post_attention_convolution.conv1d.weight".into(),
            l1_pc.expand_dims(2).unwrap(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.post_mlp_convolution.conv1d.weight".into(),
            l1_mpc.expand_dims(2).unwrap(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.mlp.gate.weight".into(),
            l1_router.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.mlp.gate.e_score_correction_bias".into(),
            l1_rbias.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.mlp.experts.gate_up_proj.weight".into(),
            l1_e_gu_il.clone(),
        );
        raw_mlxvlm.insert(
            "model.language_model.layers.1.mlp.experts.down_proj.weight".into(),
            l1_e_down.clone(),
        );
        // Top-level `[C, 4, D]` `message_input.query` (Splash mlx-vlm layout)
        raw_mlxvlm.insert(
            "model.language_model.message_input.query".into(),
            msg_q4.expand_dims(0).unwrap(),
        );
        for (i, s) in ["q", "k", "v", "mlp"].iter().enumerate() {
            raw_mlxvlm.insert(
                format!("model.language_model.layers.1.{s}_message_gate.down_proj.weight"),
                ctrl_down[i].clone(),
            );
            raw_mlxvlm.insert(
                format!("model.language_model.layers.1.{s}_message_gate.message_gate_proj.weight"),
                ctrl_gain[i].clone(),
            );
            raw_mlxvlm.insert(
                format!("model.language_model.layers.1.{s}_message_gate.forget_gate_proj.weight"),
                ctrl_forget[i].clone(),
            );
        }

        // (D) mlx-lm (`language_model.model.*`, `language_model.lm_head.*`, bare `self_attn.rel_proj` collision,
        //     split `switch_mlp.{gate,up,down}_proj`, top-level `message_input.query` [C, 4, D])
        let mut raw_mlxlm = HashMap::new();
        raw_mlxlm.insert(
            "language_model.model.embed_tokens.weight".into(),
            embed_w.clone(),
        );
        raw_mlxlm.insert("language_model.lm_head.weight".into(), lm_head_w.clone());
        raw_mlxlm.insert(
            "language_model.model.embed_norm.weight".into(),
            embed_norm_w.clone(),
        );
        raw_mlxlm.insert(
            "language_model.model.norm.weight".into(),
            final_norm_w.clone(),
        );
        raw_mlxlm.insert(
            "language_model.model.final_input.query".into(),
            out_res_q.clone(),
        );
        raw_mlxlm.insert(
            "language_model.model.message_input.query".into(),
            msg_q4.expand_dims(0).unwrap(),
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.input_layernorm.weight".into(),
            l0_inp_norm,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.post_attention_layernorm.weight".into(),
            l0_ffn_norm,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.mlp_input.query".into(),
            l0_ffn_q,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.self_attn.q_proj.weight".into(),
            l0_q,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.self_attn.k_proj.weight".into(),
            l0_k,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.self_attn.v_proj.weight".into(),
            l0_v,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.self_attn.r_proj.weight".into(),
            l0_r,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.self_attn.gate_proj.weight".into(),
            l0_g,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.self_attn.sink_proj.weight".into(),
            l0_s,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.self_attn.o_proj.weight".into(),
            l0_o,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.self_attn.q_scale".into(),
            l0_qc,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.self_attn.rel_scale".into(),
            l0_rc,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.self_attn.rel_proj".into(),
            l0_rb,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.self_attn.k_sconv.conv.weight".into(),
            l0_kc.expand_dims(2).unwrap(),
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.self_attn.v_sconv.conv.weight".into(),
            l0_vc.expand_dims(2).unwrap(),
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.attn_sconv.conv.weight".into(),
            l0_pc.expand_dims(2).unwrap(),
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.mlp_sconv.conv.weight".into(),
            l0_mpc.expand_dims(2).unwrap(),
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.mlp.gate_weight".into(),
            l0_router,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.mlp.e_score_correction_bias".into(),
            l0_rbias,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight".into(),
            l0_e_gate,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight".into(),
            l0_e_up,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight".into(),
            l0_e_down,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.attention_input.query".into(),
            l1_attn_q,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.mlp_input.query".into(),
            l1_ffn_q,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.q_input_layernorm.weight".into(),
            l1_qn,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.k_input_layernorm.weight".into(),
            l1_kn,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.v_input_layernorm.weight".into(),
            l1_vn,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.post_attention_layernorm.weight".into(),
            l1_ffn_norm,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.self_attn.q_proj.weight".into(),
            l1_q,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.self_attn.k_proj.weight".into(),
            l1_k,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.self_attn.v_proj.weight".into(),
            l1_v,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.self_attn.r_proj.weight".into(),
            l1_r,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.self_attn.gate_proj.weight".into(),
            l1_g,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.self_attn.sink_proj.weight".into(),
            l1_s,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.self_attn.o_proj.weight".into(),
            l1_o,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.self_attn.k_sconv.weight_proj.weight".into(),
            l1_kcp,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.self_attn.v_sconv.weight_proj.weight".into(),
            l1_vcp,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.self_attn.q_scale".into(),
            l1_qc,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.self_attn.rel_scale".into(),
            l1_rc,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.self_attn.rel_proj".into(),
            l1_rb,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.self_attn.k_sconv.static_weight".into(),
            l1_k_static_kc,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.self_attn.v_sconv.static_weight".into(),
            l1_v_static_kc,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.attn_sconv.conv.weight".into(),
            l1_pc.expand_dims(2).unwrap(),
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.mlp_sconv.conv.weight".into(),
            l1_mpc.expand_dims(2).unwrap(),
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.mlp.gate_weight".into(),
            l1_router,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.mlp.e_score_correction_bias".into(),
            l1_rbias,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight".into(),
            l1_e_gate,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.mlp.switch_mlp.up_proj.weight".into(),
            l1_e_up,
        );
        raw_mlxlm.insert(
            "language_model.model.layers.1.mlp.switch_mlp.down_proj.weight".into(),
            l1_e_down,
        );
        for (i, s) in ["q", "k", "v", "mlp"].iter().enumerate() {
            raw_mlxlm.insert(
                format!("language_model.model.layers.1.{s}_message_gate.down_proj.weight"),
                ctrl_down[i].clone(),
            );
            raw_mlxlm.insert(
                format!("language_model.model.layers.1.{s}_message_gate.message_gate_proj.weight"),
                ctrl_gain[i].clone(),
            );
            raw_mlxlm.insert(
                format!("language_model.model.layers.1.{s}_message_gate.forget_gate_proj.weight"),
                ctrl_forget[i].clone(),
            );
        }

        let default_plq = PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Affine,
            layout: KQuantLayout::RowMajor,
            input_amax: None,
        };
        let empty_plq = HashMap::new();
        let inv_sqrt_d = 1.0 / (d as f64).sqrt();

        // Normalize & build physical layers for all 4 layouts and compare forward outputs!
        let mut layer_outputs: Vec<(&str, Vec<f32>)> = Vec::new();
        let x_in = mk(200, &[2, d as i64], DType::Float32);

        let mut omlx_norm_captured = Inkling2StNormResult::default();
        for (tag, raw_map, want_layout, want_il) in [
            (
                "Release",
                raw_release,
                Inkling2CheckpointLayout::Release,
                Some(true),
            ),
            ("oMLX", raw_omlx, Inkling2CheckpointLayout::OMlx, Some(true)),
            (
                "mlx-vlm",
                raw_mlxvlm,
                Inkling2CheckpointLayout::MlxVlm,
                Some(true),
            ),
            (
                "mlx-lm",
                raw_mlxlm,
                Inkling2CheckpointLayout::MlxLm,
                Some(false),
            ),
        ] {
            let detected = detect_inkling2_safetensors_layout(&raw_map);
            assert_eq!(detected, want_layout, "{tag}: layout detection mismatch");
            let mut params = strip_inkling2_wrapper_prefixes(raw_map);
            let norm =
                normalize_inkling2_safetensors_params(&mut params, detected, &config).unwrap();
            assert_eq!(
                norm.gate_up_interleaved, want_il,
                "{tag}: gate_up_interleaved mismatch"
            );
            if detected == Inkling2CheckpointLayout::OMlx {
                omlx_norm_captured = norm.clone();
            }
            assert!(params.contains_key("embed_tokens.weight"), "{tag}");
            assert!(params.contains_key("lm_head.weight"), "{tag}");

            let mut cfg = config.clone();
            cfg.gate_up_interleaved = norm.gate_up_interleaved.unwrap();

            let glr =
                try_load_dense_or_dequant_f32(&params, "loop_res_query", &empty_plq, default_plq)
                    .unwrap();
            let glkv = try_load_dense_or_dequant_f32(
                &params,
                "loop_kv_res_query",
                &empty_plq,
                default_plq,
            )
            .unwrap();
            let l0 = build_physical_layer(
                0,
                &cfg,
                &params,
                &empty_plq,
                default_plq,
                inv_sqrt_d,
                glr.as_ref(),
                glkv.as_ref(),
            )
            .unwrap();
            let l1 = build_physical_layer(
                1,
                &cfg,
                &params,
                &empty_plq,
                default_plq,
                inv_sqrt_d,
                glr.as_ref(),
                glkv.as_ref(),
            )
            .unwrap();

            // Verify mixed BF16/F32 loop controllers widened cleanly to F32 across all 4 layouts
            let ctrl = l1.loop_ctrl.as_ref().unwrap();
            assert_eq!(ctrl.qkv_down_w.dtype().unwrap(), DType::Float32, "{tag}");
            assert_eq!(
                ctrl.mlp_ctrl.down_proj_w.dtype().unwrap(),
                DType::Float32,
                "{tag}"
            );

            // Run attention o_proj + both MoE blocks + loop controller to prove numerical equivalence across all 4 layouts!
            let o_out = l0
                .self_attn
                .o_proj
                .forward(&x_in.astype(DType::BFloat16).unwrap())
                .unwrap();
            let mut s0 = crate::models::inkling2::short_conv::VisitShortConvState::zeros(
                3,
                1,
                d,
                DType::BFloat16,
            )
            .unwrap();
            let y0 = l0.moe.forward(&o_out, &mut s0).unwrap();
            let mut s1 = crate::models::inkling2::short_conv::VisitShortConvState::zeros(
                3,
                1,
                d,
                DType::BFloat16,
            )
            .unwrap();
            let y1 = l1.moe.forward(&y0, &mut s1).unwrap();
            let msg_out = ctrl
                .mlp_ctrl
                .forward(&x_in, &y1.astype(DType::Float32).unwrap())
                .unwrap();
            layer_outputs.push((tag, msg_out.to_float32().unwrap().to_vec()));
        }

        let (ref_tag, ref_vals) = &layer_outputs[0];
        for (tag, vals) in &layer_outputs[1..] {
            let max_diff = vals
                .iter()
                .zip(ref_vals.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-4,
                "layout {tag} diverged from {ref_tag}: max_diff={max_diff}"
            );
        }

        // Verify per_layer_quant remapping including shape-driven oMLX fused fan-out!
        let mut plq_omlx = HashMap::new();
        let q4 = PerLayerQuant {
            bits: 4,
            group_size: 64,
            mode: PerLayerMode::Affine,
            layout: KQuantLayout::RowMajor,
            input_amax: None,
        };
        let q8 = PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Affine,
            layout: KQuantLayout::RowMajor,
            input_amax: None,
        };
        plq_omlx.insert("embed".to_string(), q8);
        plq_omlx.insert("layers.0.attn.in_proj".to_string(), q4);
        plq_omlx.insert("layers.1.attn.q_proj".to_string(), q4);
        plq_omlx.insert("layers.1.attn.k_proj".to_string(), q4);
        plq_omlx.insert("layers.1.mlp.experts.w13".to_string(), q8);
        remap_inkling2_per_layer_quant(
            &mut plq_omlx,
            Inkling2CheckpointLayout::OMlx,
            &omlx_norm_captured,
            &config,
        )
        .unwrap();
        assert_eq!(plq_omlx.get("embed_tokens"), Some(&q8));
        for p in [
            "q_proj",
            "k_proj",
            "v_proj",
            "rel_proj",
            "gate_proj",
            "sink_proj",
        ] {
            assert_eq!(plq_omlx.get(&format!("layers.0.self_attn.{p}")), Some(&q4));
        }
        for p in ["q_proj", "rel_proj", "gate_proj", "sink_proj"] {
            assert_eq!(plq_omlx.get(&format!("layers.1.self_attn.{p}")), Some(&q4));
        }
        assert_eq!(plq_omlx.get("layers.1.self_attn.k_proj"), Some(&q4));
        assert_eq!(plq_omlx.get("layers.1.self_attn.k_conv_proj"), Some(&q4));
        assert_eq!(plq_omlx.get("layers.1.mlp.experts.gate_up_proj"), Some(&q8));

        // Verify unfused oMLX global layer does NOT fan out `attn.q_proj` or stale `attn.in_proj`,
        // AND renamed attention norms (`q_input_norm`) remap to `self_attn.q_inp_norm`!
        let mut plq_omlx_unfused = HashMap::new();
        plq_omlx_unfused.insert("layers.0.attn.in_proj".to_string(), q4);
        plq_omlx_unfused.insert("layers.1.attn.q_proj".to_string(), q4);
        plq_omlx_unfused.insert("layers.1.q_input_norm".to_string(), q8);
        remap_inkling2_per_layer_quant(
            &mut plq_omlx_unfused,
            Inkling2CheckpointLayout::OMlx,
            &Inkling2StNormResult::default(),
            &config,
        )
        .unwrap();
        assert_eq!(plq_omlx_unfused.get("layers.0.self_attn.q_proj"), None);
        assert_eq!(plq_omlx_unfused.get("layers.1.self_attn.q_proj"), Some(&q4));
        assert_eq!(plq_omlx_unfused.get("layers.1.self_attn.rel_proj"), None);
        assert_eq!(
            plq_omlx_unfused.get("layers.1.self_attn.q_inp_norm"),
            Some(&q8)
        );

        // Stale out-of-range `layers.99` override returns descriptive Err instead of panicking!
        let mut plq_stale = HashMap::new();
        plq_stale.insert("layers.99.attn.q_proj".to_string(), q4);
        let stale_err = remap_inkling2_per_layer_quant(
            &mut plq_stale,
            Inkling2CheckpointLayout::OMlx,
            &Inkling2StNormResult::default(),
            &config,
        )
        .unwrap_err();
        assert!(
            stale_err.reason.contains("layers.99.attn.q_proj"),
            "{stale_err}"
        );

        let mut plq_mlxlm = HashMap::new();
        plq_mlxlm.insert("layers.0.self_attn.r_proj".to_string(), q4);
        plq_mlxlm.insert("layers.0.mlp.switch_mlp.gate_proj".to_string(), q8);
        plq_mlxlm.insert("layers.1.q_message_gate.message_gate_proj".to_string(), q8);
        remap_inkling2_per_layer_quant(
            &mut plq_mlxlm,
            Inkling2CheckpointLayout::MlxLm,
            &Inkling2StNormResult::default(),
            &config,
        )
        .unwrap();
        assert_eq!(plq_mlxlm.get("layers.0.self_attn.rel_proj"), Some(&q4));
        assert_eq!(plq_mlxlm.get("layers.0.mlp.experts.gate_proj"), Some(&q8));
        assert_eq!(plq_mlxlm.get("layers.1.loop_ctrl.q.gain_proj"), Some(&q8));
    }

    #[test]
    fn release_modelopt_2d_nvfp4_and_wrapped_canonical_interleaved_and_loop_query_validation() {
        let cfg = Inkling2Config::from_json_str(
            &serde_json::json!({
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "d_rel": 8,
                "rel_extent": 16,
                "sconv_kernel_size": 4,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 16,
                "n_layers_per_attn_res_block": 1,
                "layer_types": ["sliding_attention", "full_attention"],
                "loop_config": { "core_start": 1, "core_end": 2, "num_passes": 2, "gate_rank": 4 },
                "gate_up_interleaved": true
            })
            .to_string(),
            false,
        )
        .unwrap();

        // 1. Release ModelOpt 2-D NVFP4 (`Uint8` .weight + `Uint8` .weight_scale + .weight_scale_2)
        // must normalize `.weight` -> Uint32, `.weight_scale` -> `.scales`, infer PerLayerMode::Nvfp4
        // without explicit MLX quantization config, and apply `.weight_scale_2` directly (NO / 2688).
        let mut raw_release = HashMap::new();
        raw_release.insert(
            "model.llm.embed.weight".to_string(),
            MxArray::zeros(&[16, 32], Some(DType::BFloat16)).unwrap(),
        );
        // 32x32 FP4 = 32x16 U8 bytes (0x22 = two 1.0 FP4 values), 32x2 U8 scales (0x38 = 1.0 E4M3)
        raw_release.insert(
            "model.llm.layers.0.attn.wo_ud.weight".to_string(),
            MxArray::from_uint8(&vec![0x22u8; 32 * 16], &[32, 16]).unwrap(),
        );
        raw_release.insert(
            "model.llm.layers.0.attn.wo_ud.weight_scale".to_string(),
            MxArray::from_uint8(&[0x38u8; 32 * 2], &[32, 2]).unwrap(),
        );
        raw_release.insert(
            "model.llm.layers.0.attn.wo_ud.weight_scale_2".to_string(),
            MxArray::from_float32(&[0.25], &[1]).unwrap(),
        );
        assert_eq!(
            detect_inkling2_safetensors_layout(&raw_release),
            Inkling2CheckpointLayout::Release
        );
        let mut params = strip_inkling2_wrapper_prefixes(raw_release);
        normalize_inkling2_safetensors_params(&mut params, Inkling2CheckpointLayout::Release, &cfg)
            .unwrap();
        assert_eq!(
            params["layers.0.self_attn.o_proj.weight"].dtype().unwrap(),
            DType::Uint32
        );
        assert!(params.contains_key("layers.0.self_attn.o_proj.scales"));
        assert!(params.contains_key("layers.0.self_attn.o_proj.weight_scale_2"));

        let mut plq = HashMap::new();
        infer_unregistered_2d_fp_quant_overrides(&params, &mut plq).unwrap();
        assert_eq!(
            plq.get("layers.0.self_attn.o_proj").map(|p| p.mode),
            Some(PerLayerMode::Nvfp4)
        );
        let default_plq = PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Affine,
            layout: Default::default(),
            input_amax: None,
        };
        let mut o_proj = LinearProj::Standard(Linear::new(32, 32, Some(false)).unwrap());
        load_linear_proj_quantized_or_bf16(
            &mut o_proj,
            &params,
            "layers.0.self_attn.o_proj",
            &plq,
            default_plq,
            FAMILY_NAME,
        )
        .unwrap();
        let x = MxArray::ones(&[1, 32], Some(DType::Float32)).unwrap();
        let y = o_proj.forward(&x).unwrap().to_float32().unwrap();
        // 32 * 1.0 * 1.0 * 0.25 = 8.0 (proves weight_scale_2 applied directly without / 2688.0!)
        assert!(
            y.iter().all(|&v| (v - 8.0).abs() < 1e-3),
            "Release ModelOpt NVFP4 2D o_proj got {}",
            y[0]
        );

        // 2. Canonical checkpoint wrapped under `language_model.model.*` with fused interleaved
        // `gate_up_proj` must NOT be misclassified as MlxLm or overwrite explicit `gate_up_interleaved = true`.
        let mut raw_wrapped_canon = HashMap::new();
        raw_wrapped_canon.insert(
            "language_model.model.layers.0.self_attn.k_conv.weight".to_string(),
            MxArray::zeros(&[32, 4], Some(DType::Float32)).unwrap(),
        );
        raw_wrapped_canon.insert(
            "language_model.model.layers.0.mlp.experts.gate_up_proj.weight".to_string(),
            MxArray::zeros(&[2, 32, 32], Some(DType::BFloat16)).unwrap(),
        );
        assert_eq!(
            detect_inkling2_safetensors_layout(&raw_wrapped_canon),
            Inkling2CheckpointLayout::Canonical
        );
        let mut p_canon = strip_inkling2_wrapper_prefixes(raw_wrapped_canon);
        let norm_canon = normalize_inkling2_safetensors_params(
            &mut p_canon,
            Inkling2CheckpointLayout::Canonical,
            &cfg,
        )
        .unwrap();
        assert_eq!(norm_canon.gate_up_interleaved, None);

        // Even wrapper-only `language_model.model.*` without split experts returns None (preserves config).
        let mut raw_wrapper_only = HashMap::new();
        raw_wrapper_only.insert(
            "language_model.model.layers.0.mlp.experts.gate_up_proj.weight".to_string(),
            MxArray::zeros(&[2, 32, 32], Some(DType::BFloat16)).unwrap(),
        );
        let det_wo = detect_inkling2_safetensors_layout(&raw_wrapper_only);
        let mut p_wo = strip_inkling2_wrapper_prefixes(raw_wrapper_only);
        let norm_wo = normalize_inkling2_safetensors_params(&mut p_wo, det_wo, &cfg).unwrap();
        assert_eq!(norm_wo.gate_up_interleaved, None);

        // Mixed `MlxVlm` checkpoint (layer 0 fused interleaved + layer 1 split `switch_mlp`):
        // checkpoint-level `gate_up_interleaved` stays `Some(true)` for fused layer 0, while
        // `load_expert_gate_up` resolves split layer 1 to `effective_gate_up_interleaved == false`!
        let mut raw_mixed_vlm = HashMap::new();
        raw_mixed_vlm.insert(
            "model.language_model.layers.0.post_attention_convolution.conv1d.weight".to_string(),
            MxArray::zeros(&[32, 4, 1], Some(DType::Float32)).unwrap(),
        );
        raw_mixed_vlm.insert(
            "model.language_model.layers.0.mlp.experts.gate_up_proj.weight".to_string(),
            MxArray::zeros(&[2, 32, 32], Some(DType::BFloat16)).unwrap(),
        );
        raw_mixed_vlm.insert(
            "model.language_model.layers.1.mlp.switch_mlp.gate_proj.weight".to_string(),
            MxArray::zeros(&[2, 16, 32], Some(DType::BFloat16)).unwrap(),
        );
        raw_mixed_vlm.insert(
            "model.language_model.layers.1.mlp.switch_mlp.up_proj.weight".to_string(),
            MxArray::zeros(&[2, 16, 32], Some(DType::BFloat16)).unwrap(),
        );
        let det_mixed = detect_inkling2_safetensors_layout(&raw_mixed_vlm);
        assert_eq!(
            det_mixed,
            Inkling2CheckpointLayout::MlxVlm,
            "mixed MlxVlm checkpoint must be detected as MlxVlm, not MlxLm"
        );
        let mut p_mixed = strip_inkling2_wrapper_prefixes(raw_mixed_vlm);
        let norm_mixed =
            normalize_inkling2_safetensors_params(&mut p_mixed, det_mixed, &cfg).unwrap();
        assert_eq!(norm_mixed.gate_up_interleaved, Some(true));
        let empty_plq = HashMap::new();
        let (_, il_l0) =
            load_expert_gate_up(&p_mixed, "layers.0", &empty_plq, default_plq, true).unwrap();
        let (_, il_l1) =
            load_expert_gate_up(&p_mixed, "layers.1", &empty_plq, default_plq, true).unwrap();
        assert!(il_l0, "fused layer 0 must preserve interleaved=true");
        assert!(!il_l1, "split layer 1 must force interleaved=false");

        // Fused oMLX ModelOpt `.weight_scale` + packed `Uint8` normalizes to `.scales` + `Uint32`.
        let mut raw_omlx_fused_q = HashMap::new();
        raw_omlx_fused_q.insert(
            "model.embed.weight".to_string(),
            MxArray::zeros(&[16, 32], Some(DType::BFloat16)).unwrap(),
        );
        // k_dim (32) + taps (8) = 40 rows; K=32 -> U8 [40, 16], weight_scale [40, 2]
        raw_omlx_fused_q.insert(
            "model.layers.1.attn.k_proj.weight".to_string(),
            MxArray::from_uint8(&vec![0x22u8; 40 * 16], &[40, 16]).unwrap(),
        );
        raw_omlx_fused_q.insert(
            "model.layers.1.attn.k_proj.weight_scale".to_string(),
            MxArray::from_uint8(&[0x38u8; 40 * 2], &[40, 2]).unwrap(),
        );
        let mut p_omlx_fq = strip_inkling2_wrapper_prefixes(raw_omlx_fused_q);
        normalize_inkling2_safetensors_params(&mut p_omlx_fq, Inkling2CheckpointLayout::OMlx, &cfg)
            .unwrap();
        assert_eq!(
            p_omlx_fq["layers.1.self_attn.k_proj.weight"]
                .dtype()
                .unwrap(),
            DType::Uint32
        );
        assert!(p_omlx_fq.contains_key("layers.1.self_attn.k_proj.scales"));
        assert!(p_omlx_fq.contains_key("layers.1.self_attn.k_conv_proj.scales"));

        // 3. Top-level `message_input.query` with wrong core count fails loudly.
        let mut bad_q = HashMap::new();
        bad_q.insert(
            "language_model.model.message_input.query".to_string(),
            MxArray::zeros(&[3, 4, 32], Some(DType::Float32)).unwrap(),
        );
        let mut bad_p = strip_inkling2_wrapper_prefixes(bad_q);
        let err = normalize_inkling2_safetensors_params(
            &mut bad_p,
            Inkling2CheckpointLayout::MlxLm,
            &cfg,
        )
        .unwrap_err();
        assert!(err.reason.contains("expected [1, 4, 32]"), "{err}");

        // 4. Canonical layout with top-level `message_input.query` still expands into core layers,
        // and explicit per-layer `loop_msg_query.weight` wins deterministically when both coexist.
        let mut canon_top_q = HashMap::new();
        canon_top_q.insert(
            "layers.0.self_attn.k_conv.weight".to_string(),
            MxArray::zeros(&[32, 4], Some(DType::Float32)).unwrap(),
        );
        canon_top_q.insert(
            "message_input.query".to_string(),
            MxArray::from_float32(&vec![2.0f32; 4 * 32], &[1, 4, 32]).unwrap(),
        );
        assert_eq!(
            detect_inkling2_safetensors_layout(&canon_top_q),
            Inkling2CheckpointLayout::Canonical
        );
        normalize_inkling2_safetensors_params(
            &mut canon_top_q,
            Inkling2CheckpointLayout::Canonical,
            &cfg,
        )
        .unwrap();
        assert!(canon_top_q.contains_key("layers.1.loop_msg_query.weight"));
        assert_eq!(
            canon_top_q["layers.1.loop_msg_query.weight"]
                .to_float32()
                .unwrap()[0],
            2.0
        );

        // Coexistence: explicit per-layer `layers.1.loop_msg_query.weight` (5.0) wins over global (2.0)!
        let mut canon_coexist = HashMap::new();
        canon_coexist.insert(
            "layers.0.self_attn.k_conv.weight".to_string(),
            MxArray::zeros(&[32, 4], Some(DType::Float32)).unwrap(),
        );
        canon_coexist.insert(
            "layers.1.loop_msg_query.weight".to_string(),
            MxArray::from_float32(&vec![5.0f32; 4 * 32], &[4, 32]).unwrap(),
        );
        canon_coexist.insert(
            "message_input.query".to_string(),
            MxArray::from_float32(&vec![2.0f32; 4 * 32], &[1, 4, 32]).unwrap(),
        );
        normalize_inkling2_safetensors_params(
            &mut canon_coexist,
            Inkling2CheckpointLayout::Canonical,
            &cfg,
        )
        .unwrap();
        assert_eq!(
            canon_coexist["layers.1.loop_msg_query.weight"]
                .to_float32()
                .unwrap()[0],
            5.0
        );

        // 5. Release expert group with orphaned `weight_outer_scale_denominator` (missing `weight_amax`) fails loudly.
        let mut orphan_rel = HashMap::new();
        orphan_rel.insert(
            "model.llm.embed.weight".to_string(),
            MxArray::zeros(&[16, 32], Some(DType::BFloat16)).unwrap(),
        );
        orphan_rel.insert(
            "model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator".to_string(),
            MxArray::from_float32(&[2688.0, 2688.0], &[2]).unwrap(),
        );
        let mut orphan_p = strip_inkling2_wrapper_prefixes(orphan_rel);
        let orphan_err = normalize_inkling2_safetensors_params(
            &mut orphan_p,
            Inkling2CheckpointLayout::Release,
            &cfg,
        )
        .unwrap_err();
        assert!(
            orphan_err
                .reason
                .contains("orphaned outer_scale_denominator"),
            "{orphan_err}"
        );
    }

    #[test]
    fn safetensors_all_4_vision_layouts_match_gguf_and_guard_quantized_or_corrupt() {
        use crate::models::inkling2::vision::preprocess_dynamic_image;

        // Tiny synthetic 1-block vision tower: hidden=8, heads=2 (head_dim=4), patch=2, merge=2,
        // image_size=4 (posemb_grid=2 -> 4 pos rows), projection_dim=12.
        let vcfg = Inkling2VisionConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            image_size: 4,
            patch_size: 2,
            spatial_merge_size: 2,
            projection_dim: 12,
            layer_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        };
        let cfg = Inkling2Config {
            hidden_size: 12,
            num_hidden_layers: 1,
            num_physical_layers: Some(1),
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 4,
            vocab_size: 16,
            vision_config: Some(vcfg.clone()),
            ..Inkling2Config::from_json_str(
                r#"{"hidden_size":12,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"vocab_size":16}"#,
                false,
            )
            .unwrap()
        };

        let fill = |shape: &[i64], seed: f32, dt: DType| -> MxArray {
            let n: usize = shape.iter().map(|&x| x as usize).product();
            let vals: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed) * 0.13).sin() * 0.25)
                .collect();
            let arr = MxArray::from_float32(&vals, shape).unwrap();
            if dt == DType::Float32 {
                arr
            } else {
                arr.astype(dt).unwrap()
            }
        };

        // Patch weight in OIHW `[8, 3, 2, 2]` (GGUF) and HWC 2D `[8, 12]` (SafeTensors).
        let patch_oihw = fill(&[8, 3, 2, 2], 1.0, DType::Float32);
        let patch_hwc_2d = patch_oihw
            .transpose(Some(&[0, 2, 3, 1]))
            .unwrap()
            .reshape(&[8, 12])
            .unwrap();
        let class_emb = fill(&[8], 2.0, DType::Float32);
        let class_pos = fill(&[8], 3.0, DType::Float32);
        let pos_emb = fill(&[4, 8], 4.0, DType::Float32);
        let pre_ln_w = fill(&[8], 5.0, DType::Float32);
        let pre_ln_b = fill(&[8], 6.0, DType::Float32);
        let n1_w = fill(&[8], 7.0, DType::Float32);
        let n1_b = fill(&[8], 8.0, DType::Float32);
        let n2_w = fill(&[8], 9.0, DType::Float32);
        let n2_b = fill(&[8], 10.0, DType::Float32);
        let ls1 = fill(&[8], 11.0, DType::Float32);
        let ls2 = fill(&[8], 12.0, DType::Float32);
        let q_w = fill(&[8, 8], 13.0, DType::BFloat16);
        let q_b = fill(&[8], 14.0, DType::Float32);
        let k_w = fill(&[8, 8], 15.0, DType::BFloat16);
        let k_b = fill(&[8], 16.0, DType::Float32);
        let v_w = fill(&[8, 8], 17.0, DType::BFloat16);
        let v_b = fill(&[8], 18.0, DType::Float32);
        let o_w = fill(&[8, 8], 19.0, DType::BFloat16);
        let o_b = fill(&[8], 20.0, DType::Float32);
        let fc1_w = fill(&[16, 8], 21.0, DType::BFloat16);
        let fc1_b = fill(&[16], 22.0, DType::Float32);
        let fc2_w = fill(&[8, 16], 23.0, DType::BFloat16);
        let fc2_b = fill(&[8], 24.0, DType::Float32);
        let p1_w = fill(&[32, 32], 25.0, DType::BFloat16);
        let p2_w = fill(&[12, 32], 26.0, DType::BFloat16);

        // Reference canonical GGUF params
        let mut gguf_params = HashMap::new();
        gguf_params.insert("vision_tower.class_embedding".into(), class_emb.clone());
        gguf_params.insert(
            "vision_tower.class_position_embedding.weight".into(),
            class_pos.clone(),
        );
        gguf_params.insert("vision_tower.patch_embed.weight".into(), patch_oihw.clone());
        gguf_params.insert(
            "vision_tower.position_embedding.weight".into(),
            pos_emb.clone(),
        );
        gguf_params.insert("vision_tower.pre_ln.weight".into(), pre_ln_w.clone());
        gguf_params.insert("vision_tower.pre_ln.bias".into(), pre_ln_b.clone());
        gguf_params.insert("vision_tower.blocks.0.norm1.weight".into(), n1_w.clone());
        gguf_params.insert("vision_tower.blocks.0.norm1.bias".into(), n1_b.clone());
        gguf_params.insert("vision_tower.blocks.0.norm2.weight".into(), n2_w.clone());
        gguf_params.insert("vision_tower.blocks.0.norm2.bias".into(), n2_b.clone());
        gguf_params.insert("vision_tower.blocks.0.ls1.weight".into(), ls1.clone());
        gguf_params.insert("vision_tower.blocks.0.ls2.weight".into(), ls2.clone());
        gguf_params.insert(
            "vision_tower.blocks.0.attn.q_proj.weight".into(),
            q_w.clone(),
        );
        gguf_params.insert("vision_tower.blocks.0.attn.q_proj.bias".into(), q_b.clone());
        gguf_params.insert(
            "vision_tower.blocks.0.attn.k_proj.weight".into(),
            k_w.clone(),
        );
        gguf_params.insert("vision_tower.blocks.0.attn.k_proj.bias".into(), k_b.clone());
        gguf_params.insert(
            "vision_tower.blocks.0.attn.v_proj.weight".into(),
            v_w.clone(),
        );
        gguf_params.insert("vision_tower.blocks.0.attn.v_proj.bias".into(), v_b.clone());
        gguf_params.insert(
            "vision_tower.blocks.0.attn.o_proj.weight".into(),
            o_w.clone(),
        );
        gguf_params.insert("vision_tower.blocks.0.attn.o_proj.bias".into(), o_b.clone());
        gguf_params.insert("vision_tower.blocks.0.mlp.fc1.weight".into(), fc1_w.clone());
        gguf_params.insert("vision_tower.blocks.0.mlp.fc1.bias".into(), fc1_b.clone());
        gguf_params.insert("vision_tower.blocks.0.mlp.fc2.weight".into(), fc2_w.clone());
        gguf_params.insert("vision_tower.blocks.0.mlp.fc2.bias".into(), fc2_b.clone());
        gguf_params.insert("vision_projector.fc1.weight".into(), p1_w.clone());
        gguf_params.insert("vision_projector.fc2.weight".into(), p2_w.clone());

        let ref_enc = build_vision_encoder(&vcfg, &gguf_params, 12).expect("build ref GGUF vision");
        let mut test_img = image::RgbImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                test_img.put_pixel(
                    x,
                    y,
                    image::Rgb([(x * 60) as u8, (y * 60) as u8, ((x + y) * 30) as u8]),
                );
            }
        }
        let prep = preprocess_dynamic_image(&image::DynamicImage::ImageRgb8(test_img), &vcfg)
            .expect("preprocess");
        let ref_out = ref_enc
            .encode(&prep)
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();

        // Build all 4 SafeTensors vision layouts and verify exact parity with GGUF reference!
        let build_layout = |kind: &str| -> HashMap<String, MxArray> {
            let mut m = HashMap::new();
            let (owner, backbone, transformers_blocks, pos_has_weight, merger_proj) = match kind {
                "transformers" => (
                    "model.vision_tower.",
                    "model.vision_tower.",
                    true,
                    true,
                    true,
                ),
                "release" => (
                    "model.visual.",
                    "model.visual.backbone.",
                    false,
                    false,
                    false,
                ),
                "omlx" => (
                    "vision_tower.",
                    "vision_tower.backbone.",
                    false,
                    false,
                    false,
                ),
                "nvfp4" => ("vision_tower.", "vision_tower.", true, false, false),
                _ => unreachable!(),
            };
            m.insert(
                format!("{backbone}embeddings.class_embedding"),
                class_emb.clone(),
            );
            m.insert(
                format!("{backbone}embeddings.cls_position_embedding"),
                class_pos.clone(),
            );
            m.insert(
                if pos_has_weight {
                    format!("{backbone}embeddings.position_embedding.weight")
                } else {
                    format!("{backbone}embeddings.position_embedding")
                },
                pos_emb.clone(),
            );
            m.insert(
                format!("{backbone}embeddings.patch_embedding.weight"),
                patch_hwc_2d.clone(),
            );
            let pre_ln = if transformers_blocks {
                "pre_layernorm"
            } else {
                "ln_pre"
            };
            m.insert(format!("{backbone}{pre_ln}.weight"), pre_ln_w.clone());
            m.insert(format!("{backbone}{pre_ln}.bias"), pre_ln_b.clone());
            let (ln1, ln2, l1, l2, f1, f2) = if transformers_blocks {
                (
                    "layernorm_before",
                    "layernorm_after",
                    "lambda_1",
                    "lambda_2",
                    "fc1",
                    "fc2",
                )
            } else {
                ("ln_1", "ln_2", "ls_1.gamma", "ls_2.gamma", "c_fc", "c_proj")
            };
            m.insert(format!("{backbone}layers.0.{ln1}.weight"), n1_w.clone());
            m.insert(format!("{backbone}layers.0.{ln1}.bias"), n1_b.clone());
            m.insert(format!("{backbone}layers.0.{ln2}.weight"), n2_w.clone());
            m.insert(format!("{backbone}layers.0.{ln2}.bias"), n2_b.clone());
            m.insert(format!("{backbone}layers.0.{l1}"), ls1.clone());
            m.insert(format!("{backbone}layers.0.{l2}"), ls2.clone());
            m.insert(
                format!("{backbone}layers.0.self_attn.q_proj.weight"),
                q_w.clone(),
            );
            m.insert(
                format!("{backbone}layers.0.self_attn.q_proj.bias"),
                q_b.clone(),
            );
            m.insert(
                format!("{backbone}layers.0.self_attn.k_proj.weight"),
                k_w.clone(),
            );
            m.insert(
                format!("{backbone}layers.0.self_attn.k_proj.bias"),
                k_b.clone(),
            );
            m.insert(
                format!("{backbone}layers.0.self_attn.v_proj.weight"),
                v_w.clone(),
            );
            m.insert(
                format!("{backbone}layers.0.self_attn.v_proj.bias"),
                v_b.clone(),
            );
            m.insert(
                format!("{backbone}layers.0.self_attn.out_proj.weight"),
                o_w.clone(),
            );
            m.insert(
                format!("{backbone}layers.0.self_attn.out_proj.bias"),
                o_b.clone(),
            );
            m.insert(format!("{backbone}layers.0.mlp.{f1}.weight"), fc1_w.clone());
            m.insert(format!("{backbone}layers.0.mlp.{f1}.bias"), fc1_b.clone());
            m.insert(format!("{backbone}layers.0.mlp.{f2}.weight"), fc2_w.clone());
            m.insert(format!("{backbone}layers.0.mlp.{f2}.bias"), fc2_b.clone());
            if merger_proj {
                m.insert(format!("{owner}merger.linear_1.weight"), p1_w.clone());
                m.insert(format!("{owner}merger.linear_2.weight"), p2_w.clone());
            } else {
                m.insert(format!("{owner}proj_in.weight"), p1_w.clone());
                m.insert(format!("{owner}proj_out.weight"), p2_w.clone());
            }
            m
        };

        for layout_name in ["transformers", "release", "omlx", "nvfp4"] {
            let raw = build_layout(layout_name);
            let mut stripped = strip_inkling2_wrapper_prefixes(raw);
            normalize_inkling2_vision_params(&mut stripped, &cfg)
                .unwrap_or_else(|e| panic!("normalize {layout_name}: {e}"));
            let enc = build_vision_encoder(&vcfg, &stripped, 12)
                .unwrap_or_else(|e| panic!("build {layout_name}: {e}"));
            let got = enc
                .encode(&prep)
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(got.len(), ref_out.len(), "{layout_name}: output len");
            for (i, (&g, &r)) in got.iter().zip(ref_out.iter()).enumerate() {
                assert!(
                    (g - r).abs() < 1e-4,
                    "{layout_name}: mismatch at {i}: got {g}, ref {r}"
                );
            }
        }

        // Quantized vision guard rejects `.scales` and `.weight_scale` across vision prefixes.
        for qkey in [
            "model.vision_tower.layers.0.self_attn.q_proj.scales",
            "vision_tower.backbone.layers.0.mlp.c_fc.weight_scale",
            "model.visual.proj_in.scales",
        ] {
            let mut raw = build_layout("transformers");
            raw.insert(
                qkey.to_string(),
                MxArray::zeros(&[8, 1], Some(DType::Float16)).unwrap(),
            );
            let mut stripped = strip_inkling2_wrapper_prefixes(raw);
            let err = normalize_inkling2_vision_params(&mut stripped, &cfg)
                .expect_err("quantized vision must be rejected");
            assert!(
                err.reason.contains("quantized"),
                "expected quantized error for {qkey}, got {err}"
            );
        }

        // End-to-end `load_weights_sync` graceful text-only fallback on quantized & corrupt vision.
        let write_tiny_st_dir = |tag: &str, mut extra: HashMap<String, MxArray>| -> TempDirGuard {
            let dir = TempDirGuard::new(tag);
            fs::create_dir_all(&dir.0).unwrap();
            let config_json = serde_json::json!({
                "model_type": "inkling2",
                "hidden_size": 12,
                "num_hidden_layers": 1,
                "num_physical_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "vocab_size": 16,
                "unpadded_vocab_size": 16,
                "d_rel": 4,
                "rel_extent": 8,
                "sconv_kernel_size": 4,
                "n_layers_per_attn_res_block": 2,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 8,
                "vision_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "image_size": 4,
                    "patch_size": 2,
                    "spatial_merge_size": 2,
                    "projection_dim": 12
                }
            });
            fs::write(
                dir.0.join("config.json"),
                serde_json::to_string_pretty(&config_json).unwrap(),
            )
            .unwrap();

            let mut w: HashMap<String, MxArray> = HashMap::new();
            w.insert(
                "embed_tokens.weight".into(),
                MxArray::zeros(&[16, 12], Some(DType::BFloat16)).unwrap(),
            );
            w.insert(
                "embed_norm.weight".into(),
                MxArray::ones(&[12], Some(DType::Float32)).unwrap(),
            );
            w.insert(
                "output_res_query.weight".into(),
                MxArray::zeros(&[1, 12], Some(DType::Float32)).unwrap(),
            );
            w.insert(
                "norm.weight".into(),
                MxArray::ones(&[12], Some(DType::Float32)).unwrap(),
            );
            w.insert(
                "lm_head.weight".into(),
                MxArray::zeros(&[16, 12], Some(DType::BFloat16)).unwrap(),
            );
            // Layer 0 (local: n_layers_per_attn_res_block=2 -> phys 0 is local)
            w.insert(
                "layers.0.ffn_res_query.weight".into(),
                MxArray::zeros(&[1, 12], Some(DType::Float32)).unwrap(),
            );
            w.insert(
                "layers.0.input_layernorm.weight".into(),
                MxArray::ones(&[12], Some(DType::Float32)).unwrap(),
            );
            w.insert(
                "layers.0.post_attention_layernorm.weight".into(),
                MxArray::ones(&[12], Some(DType::Float32)).unwrap(),
            );
            w.insert(
                "layers.0.self_attn.q_proj.weight".into(),
                MxArray::zeros(&[8, 12], Some(DType::BFloat16)).unwrap(),
            );
            w.insert(
                "layers.0.self_attn.k_proj.weight".into(),
                MxArray::zeros(&[8, 12], Some(DType::BFloat16)).unwrap(),
            );
            w.insert(
                "layers.0.self_attn.v_proj.weight".into(),
                MxArray::zeros(&[8, 12], Some(DType::BFloat16)).unwrap(),
            );
            w.insert(
                "layers.0.self_attn.rel_proj.weight".into(),
                MxArray::zeros(&[8, 12], Some(DType::BFloat16)).unwrap(),
            );
            w.insert(
                "layers.0.self_attn.gate_proj.weight".into(),
                MxArray::zeros(&[8, 12], Some(DType::BFloat16)).unwrap(),
            );
            w.insert(
                "layers.0.self_attn.sink_proj.weight".into(),
                MxArray::zeros(&[2, 12], Some(DType::BFloat16)).unwrap(),
            );
            w.insert(
                "layers.0.self_attn.o_proj.weight".into(),
                MxArray::zeros(&[12, 8], Some(DType::BFloat16)).unwrap(),
            );
            w.insert(
                "layers.0.self_attn.k_conv.weight".into(),
                MxArray::zeros(&[8, 4], Some(DType::Float32)).unwrap(),
            );
            w.insert(
                "layers.0.self_attn.v_conv.weight".into(),
                MxArray::zeros(&[8, 4], Some(DType::Float32)).unwrap(),
            );
            w.insert(
                "layers.0.self_attn.post_conv.weight".into(),
                MxArray::zeros(&[12, 4], Some(DType::Float32)).unwrap(),
            );
            w.insert(
                "layers.0.self_attn.q_clip.weight".into(),
                MxArray::ones(&[8], Some(DType::Float32)).unwrap(),
            );
            w.insert(
                "layers.0.self_attn.rel_clip.weight".into(),
                MxArray::ones(&[8], Some(DType::Float32)).unwrap(),
            );
            w.insert(
                "layers.0.self_attn.rel_bias.weight".into(),
                MxArray::zeros(&[8, 4], Some(DType::Float32)).unwrap(),
            );
            w.insert(
                "layers.0.mlp.gate.weight".into(),
                MxArray::zeros(&[2, 12], Some(DType::Float32)).unwrap(),
            );
            w.insert(
                "layers.0.mlp.experts.gate_up_proj.weight".into(),
                MxArray::zeros(&[2, 16, 12], Some(DType::BFloat16)).unwrap(),
            );
            w.insert(
                "layers.0.mlp.experts.down_proj.weight".into(),
                MxArray::zeros(&[2, 12, 8], Some(DType::BFloat16)).unwrap(),
            );
            w.insert(
                "layers.0.mlp.post_conv.weight".into(),
                MxArray::zeros(&[12, 4], Some(DType::Float32)).unwrap(),
            );
            for (k, v) in extra.drain() {
                w.insert(k, v);
            }
            crate::utils::safetensors::save_safetensors(
                dir.0.join("model.safetensors"),
                &mut w,
                None,
            )
            .unwrap();
            dir
        };

        // Case A: quantized vision tower -> falls back cleanly to text-only (`vision_encoder == None`, `vision_config == None`).
        let mut q_vis = build_layout("transformers");
        q_vis.insert(
            "model.vision_tower.layers.0.self_attn.q_proj.scales".into(),
            MxArray::ones(&[8, 1], Some(DType::Float16)).unwrap(),
        );
        let dir_q = write_tiny_st_dir("vis-quant-fallback", q_vis);
        let loaded_q = load_weights_sync(&dir_q.0, None, None, false)
            .expect("quantized vision must fall back to text-only");
        assert!(loaded_q.vision_encoder.is_none());
        assert!(loaded_q.config.vision_config.is_none());

        // Case B: missing vision tensor -> falls back cleanly to text-only.
        let mut broken_vis = build_layout("omlx");
        broken_vis.remove("vision_tower.proj_out.weight");
        let dir_broken = write_tiny_st_dir("vis-broken-fallback", broken_vis);
        let loaded_broken = load_weights_sync(&dir_broken.0, None, None, false)
            .expect("incomplete vision tower must fall back to text-only");
        assert!(loaded_broken.vision_encoder.is_none());
        assert!(loaded_broken.config.vision_config.is_none());

        // Case C: valid Transformers vision tower loads end-to-end with `vision_encoder.is_some()`.
        let dir_ok = write_tiny_st_dir("vis-ok", build_layout("transformers"));
        let loaded_ok = load_weights_sync(&dir_ok.0, None, None, false)
            .expect("valid Transformers vision tower must load");
        assert!(loaded_ok.vision_encoder.is_some());
        assert!(loaded_ok.config.vision_config.is_some());
    }

    #[test]
    fn omlx_split_mxfp_attention_projections_preserve_outer_scale_and_reject_nvfp4_sidecars() {
        use crate::models::quant_dispatch::KQuantLayout;

        // Tiny 1-layer local Inkling2 config with hidden_size=32 (aligned to MXFP group_size=32):
        // q_dim = 2*8 = 16, kv_dim = 1*8 = 8, rel_dim = 2*4 = 8, sink = 2 -> fused attn.in_proj rows = 16+8+8+8+16+2 = 58.
        let cfg = Inkling2Config::from_json_str(
            &serde_json::json!({
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "d_rel": 4,
                "rel_extent": 16,
                "sconv_kernel_size": 4,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 16,
                "n_layers_per_attn_res_block": 1,
                "layer_types": ["sliding_attention"]
            })
            .to_string(),
            false,
        )
        .unwrap();

        let d = 32usize;
        let in_proj_rows = 16 + 8 + 8 + 8 + 16 + 2; // 58
        let outer_val = 2.5f32;
        let x = MxArray::ones(&[1, d as i64], Some(DType::BFloat16)).unwrap();
        let default_plq = PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Affine,
            layout: KQuantLayout::RowMajor,
            input_amax: None,
        };

        for (mode, gs, bits, w_last, w_word) in [
            (PerLayerMode::Mxfp4, 32, 4, d / 8, 0x2222_2222u32), // all nibbles = 0x2 (+1.0)
            (PerLayerMode::Mxfp8, 32, 8, d / 4, 0x3838_3838u32), // all bytes = 0x38 (+1.0)
        ] {
            let mut raw = HashMap::new();
            raw.insert(
                "model.embed.weight".to_string(),
                MxArray::zeros(&[16, d as i64], Some(DType::BFloat16)).unwrap(),
            );
            raw.insert(
                "model.layers.0.attn.in_proj.weight".to_string(),
                MxArray::from_uint32(
                    &vec![w_word; in_proj_rows * w_last],
                    &[in_proj_rows as i64, w_last as i64],
                )
                .unwrap(),
            );
            raw.insert(
                "model.layers.0.attn.in_proj.scales".to_string(),
                MxArray::from_uint8(
                    &vec![127u8; in_proj_rows * (d / 32)],
                    &[in_proj_rows as i64, (d / 32) as i64],
                )
                .unwrap(),
            );
            raw.insert(
                "model.layers.0.attn.in_proj.scale".to_string(),
                MxArray::from_float32(&[outer_val], &[1]).unwrap(),
            );

            assert_eq!(
                detect_inkling2_safetensors_layout(&raw),
                Inkling2CheckpointLayout::OMlx
            );
            let mut params = strip_inkling2_wrapper_prefixes(raw);
            let norm = normalize_inkling2_safetensors_params(
                &mut params,
                Inkling2CheckpointLayout::OMlx,
                &cfg,
            )
            .unwrap();
            assert!(norm.omlx_fused_in_proj.contains(&0));

            let plq = PerLayerQuant {
                bits,
                group_size: gs,
                mode,
                layout: KQuantLayout::RowMajor,
                input_amax: None,
            };
            let mut per_layer_quant = HashMap::new();
            per_layer_quant.insert("layers.0.attn.in_proj".to_string(), plq);
            remap_inkling2_per_layer_quant(
                &mut per_layer_quant,
                Inkling2CheckpointLayout::OMlx,
                &norm,
                &cfg,
            )
            .unwrap();

            for (sub, out_f) in [
                ("q_proj", 16usize),
                ("k_proj", 8),
                ("v_proj", 8),
                ("rel_proj", 8),
                ("gate_proj", 16),
                ("sink_proj", 2),
            ] {
                let base = format!("layers.0.self_attn.{sub}");
                let mut proj =
                    LinearProj::Standard(Linear::new(d as u32, out_f as u32, Some(false)).unwrap());
                load_inkling2_linear_proj(&mut proj, &params, &base, &per_layer_quant, default_plq)
                    .unwrap();
                proj.try_compact_symmetric_q8_or_promote_f32(DType::BFloat16)
                    .unwrap();

                match &proj {
                    LinearProj::Quantized(ql) => {
                        let s = ql
                            .outer_scale()
                            .unwrap_or_else(|| panic!("{mode:?} {sub} lost outer_scale"))
                            .to_float32()
                            .unwrap();
                        assert!(
                            (s[0] - outer_val).abs() < 1e-5,
                            "{mode:?} {sub}: expected outer_scale {outer_val}, got {}",
                            s[0]
                        );
                    }
                    LinearProj::Standard(_) => panic!("{mode:?} {sub} must stay Quantized"),
                }

                // Weight is all +1.0, x is all +1.0 (len=32), outer_scale=2.5 -> output = 32 * 2.5 = 80.0.
                let y = proj.forward(&x).unwrap();
                let vals = y.to_float32().unwrap();
                assert_eq!(vals.len(), out_f);
                for (idx, &v) in vals.iter().enumerate() {
                    assert!(
                        (v - 80.0).abs() < 1e-2,
                        "{mode:?} {sub}[{idx}]: expected 80.0 (32 * 2.5), got {v}"
                    );
                }
            }

            // Confirm QuantizedLinear::concat_rows refuses to fuse projections with outer_scale
            // (so Inkling2Attention::try_fuse_projections can never silently drop outer_scale).
            let mut q_p = LinearProj::Standard(Linear::new(d as u32, 16, Some(false)).unwrap());
            let mut k_p = LinearProj::Standard(Linear::new(d as u32, 8, Some(false)).unwrap());
            load_inkling2_linear_proj(
                &mut q_p,
                &params,
                "layers.0.self_attn.q_proj",
                &per_layer_quant,
                default_plq,
            )
            .unwrap();
            load_inkling2_linear_proj(
                &mut k_p,
                &params,
                "layers.0.self_attn.k_proj",
                &per_layer_quant,
                default_plq,
            )
            .unwrap();
            if let (LinearProj::Quantized(q_ql), LinearProj::Quantized(k_ql)) = (&q_p, &k_p) {
                assert!(
                    q_ql.concat_rows(k_ql).unwrap().is_none(),
                    "{mode:?}: concat_rows must decline fusion when outer_scale is present"
                );
            }

            // Unsupported NVFP4-only `.global_scale` on MXFP4/MXFP8 must fail loudly.
            let mut bad_params = params.clone();
            bad_params.insert(
                "layers.0.self_attn.q_proj.global_scale".to_string(),
                MxArray::from_float32(&[5376.0], &[1]).unwrap(),
            );
            let mut bad_proj =
                LinearProj::Standard(Linear::new(d as u32, 16, Some(false)).unwrap());
            let err = load_inkling2_linear_proj(
                &mut bad_proj,
                &bad_params,
                "layers.0.self_attn.q_proj",
                &per_layer_quant,
                default_plq,
            )
            .expect_err("NVFP4-only global_scale on MXFP projection must fail loudly");
            assert!(
                err.reason.contains("unsupported NVFP4 outer scale tensor"),
                "unexpected error: {}",
                err.reason
            );
        }

        // Dense projection with scalar `.scale` also folds the scale instead of dropping it.
        let mut dense_params = HashMap::new();
        dense_params.insert(
            "layers.0.self_attn.o_proj.weight".to_string(),
            MxArray::ones(&[8, d as i64], Some(DType::BFloat16)).unwrap(),
        );
        dense_params.insert(
            "layers.0.self_attn.o_proj.scale".to_string(),
            MxArray::from_float32(&[outer_val], &[1]).unwrap(),
        );
        let mut dense_proj = LinearProj::Standard(Linear::new(d as u32, 8, Some(false)).unwrap());
        load_inkling2_linear_proj(
            &mut dense_proj,
            &dense_params,
            "layers.0.self_attn.o_proj",
            &HashMap::new(),
            default_plq,
        )
        .unwrap();
        let dense_y = dense_proj.forward(&x).unwrap().to_float32().unwrap();
        assert!(
            dense_y.iter().all(|&v| (v - 80.0).abs() < 1e-1),
            "dense o_proj with .scale=2.5 must produce 80.0, got {:?}",
            dense_y.as_ref()
        );
    }
}
