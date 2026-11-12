//! The memory plan a dense or MoE Qwen3.5 load serves its lanes from.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::array::DType;
use crate::engine::checkpoint_source::CheckpointSource;
use crate::engine::memory_plan::{
    ContextBounds, FlatInput, KvLayout, LaneShape, LiveMemory, MeasuredMemory, MemoryFootprint,
    MemoryPlan, PlanError, PoolInput, SPECULATIVE_SCRATCH_TOKENS, check_max_context,
    dflash2_lane_state_bytes, live_memory,
};
use crate::models::mtp_drafter::{
    DrafterBodyVariant, detect_drafter_safetensors, missing_mtp_keys, reprefix_drafter_key,
};
use crate::models::quant_dispatch::{
    has_kquant_mode, has_sym8_mode, parse_quant_settings, select_quantization_block,
};
use crate::models::quantized_linear::{DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE};
use crate::nn::rope_scaling::RopeScaling;
use crate::transformer::KvFormat;
use crate::utils::gguf::NativeGgufFamily;
use crate::utils::gguf_import::imported_dtypes;
use crate::utils::safetensors::{SafeTensorDType, SafeTensorsFile, checkpoint_safetensors_files};

use super::config::Qwen3_5Config;
use super::dflash2::DFlash2Config;
use super::dflash2_decode::{flat_lane_transient, pool_lane_transient};
use super::dflash2_turn::DFlash2Target;
use super::persistence::{
    compute_dtype_of, mtp_sidecar_candidates, normalize_mtp_weight_key, qwen35_text_key,
};
use crate::models::qwen3_5_moe::persistence::{mtp_expert_keys, settled_compute_dtype};

/// The pool's element type for `format`. BF16 pages hold any compute dtype
/// cast to bf16; int8 pages need BF16 rows. Either format exists only where
/// mlx-node has the in-place prefill attention kernel: a query group of 4,
/// 6, 8 or 16, head 128 or 256, 16- or 32-token blocks. Any other load
/// fails, as Splash's start fails when `compiledAttention` has "no
/// paged-attention kernel for layout". Every GPU family loads: Splash's
/// `kMinimumAppleGpuFamily` is an engine-wide floor, not this kernel's.
pub(crate) fn qwen35_pool_cache_dtype(
    label: &str,
    format: KvFormat,
    heads: PoolHeads,
    block_size: u32,
    compute: DType,
) -> Result<mlx_paged_attn::metal::MetalDtype> {
    use mlx_paged_attn::metal::MetalDtype;
    let group = if heads.kv > 0 && heads.query.is_multiple_of(heads.kv) {
        heads.query / heads.kv
    } else {
        0
    };
    if !(matches!(group, 4 | 6 | 8 | 16)
        && matches!(heads.dim, 128 | 256)
        && matches!(block_size, 16 | 32))
    {
        return Err(Error::from_reason(format!(
            "{label}: no paged prefill attention kernel for {}/{} heads, head_dim {}, block size \
             {block_size}; load with useBlockPagedCache false",
            heads.query, heads.kv, heads.dim
        )));
    }
    match format {
        KvFormat::Bf16 => Ok(MetalDtype::BFloat16),
        KvFormat::Int8 if compute == DType::BFloat16 => Ok(MetalDtype::Int8),
        KvFormat::Int8 => Err(Error::from_reason(format!(
            "{label}: int8 pages need BF16 rows; this load computes in {compute:?}; load with \
             kvFormat 'bf16'"
        ))),
    }
}

/// Query heads, KV heads and head size of a pool's full-attention layers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PoolHeads {
    pub query: u32,
    pub kv: u32,
    pub dim: u32,
}

impl PoolHeads {
    pub(crate) fn of(config: &Qwen3_5Config) -> Self {
        Self {
            query: config.num_heads.max(0) as u32,
            kv: config.num_kv_heads.max(0) as u32,
            dim: config.head_dim.max(0) as u32,
        }
    }
}

/// The blocks a pool starts with: `layout`'s runway pages (Splash
/// `kvRunwayPages(minimumExtentPages)`) and one request's recurrent state,
/// which the scheduler charges inside the pool.
pub(crate) fn qwen35_runway_blocks(
    block_size: u32,
    bytes_per_block: u64,
    recurrent_state_bytes: u64,
    layout: &KvLayout,
) -> u32 {
    let tokens = layout.runway_pages() * crate::engine::memory_plan::KV_PAGE_TOKENS;
    tokens
        .div_ceil(u64::from(block_size.max(1)))
        .saturating_add(recurrent_state_bytes.div_ceil(bytes_per_block.max(1)))
        .min(u64::from(u32::MAX)) as u32
}

/// The paged pool a load builds: blocks of `format` over the full-attention
/// layers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PoolGeometry {
    pub format: KvFormat,
    pub bytes_per_block: u64,
    pub block_size: u32,
    pub layers: u32,
    /// `paged_cache_memory_mb`.
    pub explicit_mib: Option<u32>,
}

/// What a DFlash2 draft adds to a plan.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DraftTerms {
    pub tap_layers: usize,
    pub ring_bytes: u64,
    pub weight_bytes: u64,
}

impl DraftTerms {
    pub(crate) fn of(config: &DFlash2Config, weight_bytes: u64) -> Self {
        Self {
            tap_layers: config.target_layers.len(),
            ring_bytes: config.kv_ring_bytes(),
            weight_bytes,
        }
    }
}

/// The lanes one load prices.
pub(crate) struct LoadLanes<'a> {
    pub config: &'a Qwen3_5Config,
    pub recurrent_state_bytes: u64,
    pub pool: Option<PoolGeometry>,
    pub draft: Option<DraftTerms>,
    /// Format and dtype of the flat autoregressive lane's K/V (a load
    /// without a pool).
    pub format: KvFormat,
    pub dtype: DType,
    /// The load can run an MTP turn on the pool lane (MTP heads, no
    /// DFlash2 draft): the lane holds its draft state at the window.
    pub mtp: bool,
    /// The scheduler books that state inside the pool as well.
    pub mtp_state_in_pool: bool,
}

/// Whether the hybrid scheduler serves this process's Qwen3.5 turns: it
/// books lane state inside the pool.
pub(crate) fn qwen35_scheduler_books_state_in_pool() -> bool {
    super::model::Qwen35SchedulerState::continuous_batching_enabled()
}

/// Splash `evaluateEngineMemoryPlan` over `lanes` on the device `live`.
/// Every load prices one lane: the pool lane, whose state carries a DFlash2
/// draft's (Splash's one window for every turn), or the flat lane of a load
/// without a pool.
pub(crate) fn evaluate_qwen35_plan(
    label: &str,
    lanes: &LoadLanes<'_>,
    live: LiveMemory,
    max_memory_bytes: Option<u64>,
) -> Result<MemoryPlan> {
    let config = lanes.config;
    let draft = lanes.draft;
    let footprint = MemoryFootprint {
        working_set_bytes: live.working_set_bytes,
        max_memory_bytes,
        active_bytes: live.active_bytes,
        other_pool_bytes: live.registered_pool_bytes,
        staging_bytes: if lanes.pool.is_some() {
            mlx_paged_attn::RESTORE_STAGING_BYTES
        } else {
            0
        },
        draft_weights_bytes: draft.map_or(0, |draft| draft.weight_bytes),
        held_back_bytes: 0,
    };
    let recurrent_state_bytes = lanes.recurrent_state_bytes;
    let trained = config.served_context_tokens();
    let fa_layers = config.full_attention_layer_count();
    let draft_taps = draft.map(|draft| draft.tap_layers);
    let pool_transient = |window| pool_lane_transient(config, draft_taps, window);
    let mtp = lanes.mtp;
    // A DFlash2 turn runs alone on the Barrier lane, so its state is never
    // booked in the pool.
    let draft_state =
        draft.map(|draft| dflash2_lane_state_bytes(recurrent_state_bytes, draft.ring_bytes));
    let lane_state = |window: u32| match draft_state {
        Some(bytes) => bytes,
        None if mtp => super::scheduled_mtp::ScheduledMtpState::held_bytes(config, window),
        None => 0,
    };
    let pool_input = lanes.pool.map(|geometry| {
        let layout = KvLayout::of(
            fa_layers,
            config.num_kv_heads,
            config.head_dim,
            DType::BFloat16,
            geometry.format,
        );
        PoolInput {
            shape: LaneShape {
                trained,
                lane_bytes: recurrent_state_bytes,
                layout,
                scratch_rows: SPECULATIVE_SCRATCH_TOKENS,
                transient: &pool_transient,
            },
            bytes_per_block: geometry.bytes_per_block,
            layers: u64::from(geometry.layers),
            runway_blocks: qwen35_runway_blocks(
                geometry.block_size,
                geometry.bytes_per_block,
                recurrent_state_bytes,
                &layout,
            ),
            explicit_mib: geometry.explicit_mib,
            lane_in_pool: true,
            state: &lane_state,
            state_in_pool: mtp && lanes.mtp_state_in_pool,
        }
    });
    let format = lanes.format;
    let autoregressive_transient = |window| flat_lane_transient(config, format, window);
    let flat = lanes.pool.is_none().then(|| FlatInput {
        shape: LaneShape {
            trained,
            lane_bytes: recurrent_state_bytes,
            layout: KvLayout::of(
                fa_layers,
                config.num_kv_heads,
                config.head_dim,
                lanes.dtype,
                format,
            ),
            scratch_rows: 0,
            transient: &autoregressive_transient,
        },
        copy_layers: Some(fa_layers as u64),
    });
    MemoryPlan::evaluate(footprint, pool_input, flat)
        .map_err(|error| Error::from_reason(error.message(label)))
}

/// Every turn's window: the pool lane's, else the flat lane's of a load
/// without a pool.
fn turn_window(plan: &MemoryPlan) -> Option<ContextBounds> {
    plan.pool
        .as_ref()
        .map(|pool| pool.lane.window)
        .or(plan.flat.as_ref().map(|lane| lane.window))
}

/// Splash `ServingRuntime::start`'s `--max-context` refusal over `plan`.
pub(crate) fn check_qwen35_max_context(
    label: &str,
    plan: &MemoryPlan,
    max_context: Option<u32>,
) -> Result<()> {
    match turn_window(plan) {
        Some(window) => check_max_context(window, plan.bound, max_context)
            .map_err(|error| Error::from_reason(error.message(label))),
        None => Ok(()),
    }
}

/// A load without a plan (device memory unreadable) serves at most its
/// native window.
fn check_unplanned_max_context(
    label: &str,
    config: &Qwen3_5Config,
    max_context: Option<u32>,
) -> Result<()> {
    let native = config.max_position_embeddings.max(0) as u32;
    match max_context {
        Some(requested) if requested > native => Err(Error::from_reason(
            PlanError::ContextWithoutPlan { requested, native }.message(label),
        )),
        _ => Ok(()),
    }
}

/// The plan of `target`'s load from what the device holds now: the pool lane
/// when `pool` is set, else the flat lane. `None` when the working set is
/// unreadable.
pub(crate) fn plan_qwen35_load<T: DFlash2Target>(
    label: &str,
    target: &T,
    recurrent_state_bytes: u64,
    mtp_weights: bool,
    pool: Option<PoolGeometry>,
    max_memory_bytes: Option<u64>,
    max_context: Option<u32>,
) -> Result<Option<MemoryPlan>> {
    let draft = target.dflash2();
    let config = target.dense_config();
    let Some(live) = live_memory() else {
        check_unplanned_max_context(label, &config, max_context)?;
        return Ok(None);
    };
    let lanes = LoadLanes {
        config: &config,
        recurrent_state_bytes,
        pool,
        draft: draft.map(|draft| DraftTerms::of(&draft.config, draft.weight_bytes)),
        format: config.kv_format(),
        dtype: target.embedding_dtype()?,
        mtp: mtp_weights && draft.is_none(),
        mtp_state_in_pool: qwen35_scheduler_books_state_in_pool(),
    };
    let plan = evaluate_qwen35_plan(label, &lanes, live, max_memory_bytes)?;
    check_qwen35_max_context(label, &plan, max_context)?;
    if let Some(line) = plan.max_memory_ignored() {
        tracing::info!("{label} {line}");
    }
    tracing::info!("{label} memory plan:\n{}", plan.describe());
    Ok(Some(plan))
}

/// Splash `auditActualMemory` at the end of a load: what is resident now
/// against `plan`. `pool_bytes` is this load's pool, already registered.
pub(crate) fn audit_qwen35_load(label: &str, plan: &MemoryPlan, pool_bytes: u64) -> Result<()> {
    let Some(live) = live_memory() else {
        return Ok(());
    };
    crate::engine::memory_plan::audit(plan, measured_after_load(plan, live, pool_bytes)).map_err(
        |error| {
            tracing::error!("{label} memory plan:\n{}", plan.describe());
            Error::from_reason(format!("{label} memory plan audit: {error}"))
        },
    )
}

/// The audit's view of `live` once this load's pool of `pool_bytes` is
/// registered: the coordinator holds that pool with its staging, which the
/// audit charges from the plan, so neither is another pool.
fn measured_after_load(plan: &MemoryPlan, live: LiveMemory, pool_bytes: u64) -> MeasuredMemory {
    let own = match pool_bytes {
        0 => 0,
        bytes => bytes.saturating_add(plan.footprint.staging_bytes),
    };
    MeasuredMemory {
        active_bytes: live.active_bytes,
        pool_bytes,
        other_pool_bytes: live.registered_pool_bytes.saturating_sub(own),
    }
}

/// The `maxMemoryBytes` load option: a positive byte count.
pub(crate) fn max_memory_option(label: &str, value: Option<i64>) -> Result<Option<u64>> {
    match value {
        None => Ok(None),
        Some(bytes) if bytes > 0 => Ok(Some(bytes as u64)),
        Some(_) => Err(Error::from_reason(format!(
            "{label} load: maxMemoryBytes must be a positive byte count"
        ))),
    }
}

/// What a load of a Qwen3.5 checkpoint would serve, priced before it loads.
#[napi(object)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ContextEstimate {
    /// Tokens every turn may hold: the pool lane's (with a DFlash2 draft's
    /// state priced), the flat lane's without a pool.
    pub window_tokens: u32,
    /// `'memory'` when the plan binds the window, `'model'` when the trained
    /// window does.
    pub bound: String,
}

/// The load-free estimate behind `/v1/models`: the plan a load of
/// `model_path` would make, with the weights' bytes taken from their files
/// (Splash sizes weights the same way before they load). `None` when the
/// Metal working set is unreadable. `model_type` is `qwen3_5` or
/// `qwen3_5_moe`; `options` are the load's.
#[napi]
pub async fn estimate_qwen35_context_window(
    model_path: String,
    model_type: String,
    options: Option<crate::models::qwen3_5::model::Qwen35LoadOptions>,
) -> Result<Option<Qwen35ContextEstimate>> {
    let (moe, family, label) = match model_type.as_str() {
        "qwen3_5" => (false, NativeGgufFamily::Qwen35, "Qwen3.5"),
        "qwen3_5_moe" => (true, NativeGgufFamily::Qwen35Moe, "Qwen3.5 MoE"),
        other => {
            return Err(Error::from_reason(format!(
                "Qwen3.5 estimate: model_type {other:?} is not qwen3_5 or qwen3_5_moe"
            )));
        }
    };
    let options = options.unwrap_or_default();
    let kv_format = options
        .kv_format
        .as_deref()
        .map(|value| KvFormat::parse(Some(value)))
        .transpose()
        .map_err(|message| Error::from_reason(format!("{label} estimate: {message}")))?;
    let max_memory_bytes = max_memory_option(label, options.max_memory_bytes)?;
    let max_context =
        crate::nn::rope_scaling::max_context_option(label, options.max_context_tokens)?;
    let source = CheckpointSource::resolve(Path::new(&model_path), family).await?;
    let estimate = checkpoint_weight_bytes(&source).and_then(|weight_bytes| {
        estimate_qwen35_load(
            label,
            &EstimateInput {
                asset_dir: source.asset_dir(),
                moe,
                weight_bytes,
                header: checkpoint_header(&source, moe)?,
                draft_dir: options.draft_model_path.as_deref().map(Path::new),
                kv_format,
                max_memory_bytes,
                max_context,
            },
        )
    });
    source.release_lease();
    estimate
}

/// Bytes of a checkpoint's weight files: a directory's top-level
/// safetensors, or the GGUF and its companion.
pub(crate) fn checkpoint_weight_bytes(source: &CheckpointSource) -> Result<u64> {
    let stat = |path: &Path| {
        std::fs::metadata(path)
            .map(|metadata| metadata.len())
            .map_err(|error| {
                Error::from_reason(format!("Failed to stat {}: {error}", path.display()))
            })
    };
    match source {
        CheckpointSource::Dir(dir) => {
            let entries = std::fs::read_dir(dir).map_err(|error| {
                Error::from_reason(format!("Failed to list {}: {error}", dir.display()))
            })?;
            let mut total = 0u64;
            for entry in entries {
                let path = entry
                    .map_err(|error| {
                        Error::from_reason(format!("Failed to list {}: {error}", dir.display()))
                    })?
                    .path();
                let safetensors = path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"));
                if safetensors && path.is_file() {
                    total = total.saturating_add(stat(&path)?);
                }
            }
            Ok(total)
        }
        CheckpointSource::Gguf(checkpoint) => {
            let companion = checkpoint
                .companion
                .as_ref()
                .map_or(Ok(0), |(path, _, _)| stat(path))?;
            Ok(stat(&checkpoint.main.0)?.saturating_add(companion))
        }
    }
}

/// What a load of a checkpoint reads from its weights, taken from the
/// headers alone.
pub(crate) struct CheckpointHeader {
    /// The compute dtype the load settles on.
    pub compute: DType,
    /// The sanitized `mtp.*` keys the load's MTP gate sees, with their dtypes.
    pub mtp: HashMap<String, DType>,
}

/// Each tensor's dtype in the safetensors files `paths`, an fp8 weight with
/// a `_scale_inv` as the BF16 `dequant_fp8_weights` makes of it.
fn safetensors_dtypes(paths: &[PathBuf]) -> Result<Vec<(String, DType)>> {
    let mut dtypes = Vec::new();
    let mut fp8_scaled = HashSet::new();
    for path in paths {
        for (name, info) in SafeTensorsFile::load(path)?.tensors {
            if let Some(weight) = name.strip_suffix("_scale_inv") {
                fp8_scaled.insert(weight.to_string());
            }
            let dtype = match info.dtype {
                SafeTensorDType::F32 => DType::Float32,
                SafeTensorDType::F16 => DType::Float16,
                SafeTensorDType::BF16 => DType::BFloat16,
                SafeTensorDType::I32 => DType::Int32,
                SafeTensorDType::U32 => DType::Uint32,
                SafeTensorDType::I8 => DType::Int8,
                SafeTensorDType::U8 | SafeTensorDType::F8_E4M3 => DType::Uint8,
                SafeTensorDType::I64 | SafeTensorDType::F64 | SafeTensorDType::BOOL => continue,
            };
            dtypes.push((name, dtype));
        }
    }
    for (name, dtype) in &mut dtypes {
        if fp8_scaled.contains(name.as_str()) {
            *dtype = DType::BFloat16;
        }
    }
    Ok(dtypes)
}

/// The loaders' rule over the dtypes the weights' headers give the
/// sanitized keys (the GGUF ones are the import's): the compute dtype, and
/// the `mtp.*` keys from the body, else (no inline MTP) the dense loader's
/// sidecar, else the split drafter.
pub(crate) fn checkpoint_header(source: &CheckpointSource, moe: bool) -> Result<CheckpointHeader> {
    let mut raw_dtypes = match source {
        CheckpointSource::Dir(dir) => safetensors_dtypes(&checkpoint_safetensors_files(dir))?,
        CheckpointSource::Gguf(checkpoint) => {
            imported_dtypes(&checkpoint.main.1, &checkpoint.plans.0, &checkpoint.opts)
        }
    };
    let asset_dir = source.asset_dir();
    let raw = read_config_json(asset_dir)?;
    let inline_mtp = raw_dtypes.iter().any(|(name, _)| {
        if moe {
            name.contains("mtp.")
        } else {
            normalize_mtp_weight_key(name).is_some()
        }
    });
    if !inline_mtp {
        let mut sidecar = Vec::new();
        if !moe {
            for candidate in mtp_sidecar_candidates(asset_dir, &raw) {
                if !candidate.exists() {
                    continue;
                }
                sidecar = safetensors_dtypes(&[candidate])?
                    .into_iter()
                    .filter_map(|(name, dtype)| Some((normalize_mtp_weight_key(&name)?, dtype)))
                    .collect();
                if !sidecar.is_empty() {
                    break;
                }
            }
        }
        if sidecar.is_empty()
            && let Some(drafter) = detect_drafter_safetensors(asset_dir)
        {
            sidecar = safetensors_dtypes(&[drafter])?
                .into_iter()
                .map(|(name, dtype)| (reprefix_drafter_key(&name), dtype))
                .collect();
        }
        raw_dtypes.extend(sidecar);
    }
    let has_individual_experts = raw_dtypes
        .iter()
        .any(|(name, _)| name.contains(".mlp.experts.0.up_proj.weight"));
    let dtypes: HashMap<String, DType> = raw_dtypes
        .into_iter()
        .filter_map(|(name, dtype)| Some((qwen35_text_key(&name)?, dtype)))
        .collect();
    let mut mtp = HashMap::new();
    for (name, dtype) in &dtypes {
        if !name.starts_with("mtp.") {
            continue;
        }
        if moe {
            for key in mtp_expert_keys(name, has_individual_experts) {
                mtp.insert(key, *dtype);
            }
        } else {
            mtp.insert(name.clone(), *dtype);
        }
    }
    let dtype_of = |key: &str| dtypes.get(key).copied();
    let compute = if moe {
        settled_compute_dtype(&raw, &dtype_of)
    } else {
        compute_dtype_of(&dtype_of)
    };
    Ok(CheckpointHeader { compute, mtp })
}

fn read_config_json(dir: &Path) -> Result<serde_json::Value> {
    let config_path = dir.join("config.json");
    std::fs::read_to_string(&config_path)
        .map_err(|error| {
            Error::from_reason(format!("Failed to read {}: {error}", config_path.display()))
        })
        .and_then(|data| {
            serde_json::from_str(&data).map_err(|error| {
                Error::from_reason(format!(
                    "Failed to parse {}: {error}",
                    config_path.display()
                ))
            })
        })
}

pub(crate) struct EstimateInput<'a> {
    /// The directory holding the checkpoint's `config.json`.
    pub asset_dir: &'a Path,
    pub moe: bool,
    pub weight_bytes: u64,
    pub header: CheckpointHeader,
    pub draft_dir: Option<&'a Path>,
    pub kv_format: Option<KvFormat>,
    pub max_memory_bytes: Option<u64>,
    pub max_context: Option<u32>,
}

/// The plan a load of `input` would make on the device now, with the
/// target's and the draft's weight files counted as active bytes.
pub(crate) fn estimate_qwen35_load(
    label: &str,
    input: &EstimateInput<'_>,
) -> Result<Option<Qwen35ContextEstimate>> {
    let raw = read_config_json(input.asset_dir)?;
    let (mut config, mtp_body) = if input.moe {
        let moe = crate::models::qwen3_5_moe::persistence::parse_config(&raw)?;
        let body = crate::models::qwen3_5_moe::mtp::Qwen3_5MoeMTPModule::mtp_mlp_variant(&moe);
        (moe.to_dense_config(), body)
    } else {
        (
            crate::models::qwen3_5::persistence::parse_config(&raw)?,
            DrafterBodyVariant::Dense,
        )
    };
    config.rope_scaling = Some(RopeScaling::served_for(
        label,
        config.max_position_embeddings.max(0) as u32,
        config.scaling(),
        input.max_context,
    )?);
    let (_, _, top_level_mode, per_layer_quant) = parse_quant_settings(
        select_quantization_block(&raw)?,
        DEFAULT_QUANT_BITS,
        DEFAULT_QUANT_GROUP_SIZE,
    )?;
    let sym8 = has_sym8_mode(top_level_mode, &per_layer_quant);
    // Both loaders pin a sym8 checkpoint to flat KV after reading its quant block.
    if sym8 && config.use_block_paged_cache == Some(true) {
        config.use_block_paged_cache = Some(false);
    }
    // The loaders turn the MTP head off for sym8 (both) and K-quant (MoE)
    // checkpoints, and for an incomplete `mtp.*` set.
    let mtp_head = config.n_mtp_layers > 0
        && !sym8
        && !(input.moe && has_kquant_mode(top_level_mode, &per_layer_quant))
        && missing_mtp_keys(
            &|key| {
                input
                    .header
                    .mtp
                    .get(key)
                    .map(|dtype| *dtype == DType::Uint32)
            },
            mtp_body,
            config.n_mtp_layers,
        )
        .is_empty();
    KvFormat::parse(config.kv_format.as_deref())
        .map_err(|message| Error::from_reason(format!("config.json: {message}")))?;
    let compute = input.header.compute;
    let format = config.pin_kv_format(input.kv_format, compute);
    let draft = input
        .draft_dir
        .map(|dir| {
            let draft = super::dflash2::read_dflash2_config(dir)?;
            Ok::<_, Error>(DraftTerms::of(
                &draft,
                super::dflash2::dflash2_file_bytes(dir)?,
            ))
        })
        .transpose()?;
    let pool = if config.use_block_paged_cache.unwrap_or(true)
        && crate::engine::persistence::compiled_forward_backend_available()
    {
        let layers = config.full_attention_layer_count() as u32;
        let block_size = config.paged_block_size();
        let cache_dtype =
            qwen35_pool_cache_dtype(label, format, PoolHeads::of(&config), block_size, compute)?;
        let bytes_per_block = mlx_paged_attn::profile::bytes_per_block(
            layers,
            config.num_kv_heads as u32,
            config.head_dim as u32,
            block_size,
            cache_dtype,
        )
        .map_err(|error| Error::from_reason(format!("{label} block-paged adapter: {error}")))?;
        Some(PoolGeometry {
            format,
            bytes_per_block,
            block_size,
            layers,
            explicit_mib: config.paged_cache_memory_mb,
        })
    } else {
        None
    };
    if draft.is_some() && pool.is_none() {
        return Err(Error::from_reason(format!(
            "{label} DFlash2 runs on the block-paged KV pool and this load has none; load \
             without the companion (autoLoadDraft: false) or with the paged cache"
        )));
    }
    let served = config.served_context_tokens();
    let cap = input.max_context.unwrap_or(served);
    let Some(mut live) = live_memory() else {
        check_unplanned_max_context(label, &config, input.max_context)?;
        return Ok(None);
    };
    live.active_bytes = live
        .active_bytes
        .saturating_add(input.weight_bytes)
        .saturating_add(draft.map_or(0, |draft| draft.weight_bytes));
    let lanes = LoadLanes {
        config: &config,
        recurrent_state_bytes: config.recurrent_state_bytes(),
        pool,
        draft,
        format,
        dtype: compute,
        mtp: mtp_head && draft.is_none(),
        mtp_state_in_pool: qwen35_scheduler_books_state_in_pool(),
    };
    let plan = evaluate_qwen35_plan(label, &lanes, live, input.max_memory_bytes)?;
    check_qwen35_max_context(label, &plan, input.max_context)?;
    let window = turn_window(&plan);
    Ok(Some(Qwen35ContextEstimate {
        window_tokens: window.map_or(served, |window| window.maximum()).min(cap),
        bound: window.map_or("model", |window| window.bound()).to_string(),
    }))
}

#[cfg(test)]
mod tests;
