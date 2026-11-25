use std::collections::{HashMap, HashSet};
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
use super::moe::{ExpertSwitchProj, Inkling2MoE};
use super::paged_forward::Inkling2PagedRuntime;
use super::scheduled_dspark::Inkling2ScheduledDsparkStates;
use super::scheduler::Inkling2SchedulerState;
use super::state_disk::compute_persistent_identity;
use super::vision::{Inkling2VisionBlock, Inkling2VisionEncoder};
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
                vision_config: inner.config.vision_config.clone(),
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

    let k_conv = require_tensor(params, &format!("{prefix}.self_attn.k_conv.weight"))?
        .astype(DType::Float32)?;
    let v_conv = require_tensor(params, &format!("{prefix}.self_attn.v_conv.weight"))?
        .astype(DType::Float32)?;
    let post_conv = require_tensor(params, &format!("{prefix}.self_attn.post_conv.weight"))?
        .astype(DType::Float32)?;

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
    gate_up_proj.promote_outer_scale_to_f32()?;
    down_proj.promote_outer_scale_to_f32()?;
    let mlp_post_conv = require_tensor(params, &format!("{prefix}.mlp.post_conv.weight"))?
        .astype(DType::Float32)?;
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
}
