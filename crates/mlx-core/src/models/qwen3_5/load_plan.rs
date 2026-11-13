//! The memory plan a dense or MoE Qwen3.5 load serves its lanes from.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::array::DType;
use crate::engine::checkpoint_source::CheckpointSource;
use crate::engine::memory_plan::{
    ContextBounds, KvLayout, LaneShape, LiveMemory, MeasuredMemory, MemoryFootprint, MemoryPlan,
    PlanError, PoolInput, SPECULATIVE_SCRATCH_TOKENS, check_max_context, dflash2_lane_state_bytes,
    live_memory,
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
use super::dflash2_decode::{pool_decode_step_bytes, pool_lane_transient};
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
            "{label}: no paged-attention kernel for layout: {}/{} heads, head_dim {}, block \
             size {block_size} (query groups 4, 6, 8, 16; head 128, 256; blocks 16, 32)",
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

#[cfg(test)]
thread_local! {
    /// Loads on this thread see no Metal device.
    pub(crate) static NO_METAL_POOL: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

fn metal_device() -> bool {
    #[cfg(test)]
    if NO_METAL_POOL.with(std::cell::Cell::get) {
        return false;
    }
    crate::engine::persistence::compiled_forward_backend_available()
}

/// Whether a Qwen3.5 load builds the block-paged pool: Splash runs only on
/// Metal. A load off Metal (the CUDA preview) has no pool: training, eval
/// and calibration run on it, and its chat turns take the flat lane
/// ([`qwen35_flat_lane_check`]).
pub(crate) fn qwen35_pool_backend() -> bool {
    metal_device()
}

/// The flat autoregressive lane serves only a load off Metal (the paged
/// kernels exist only on Metal). On Metal every turn runs on the pool.
pub(crate) fn qwen35_flat_lane_check(label: &str) -> Result<()> {
    if !metal_device() {
        return Ok(());
    }
    Err(Error::from_reason(format!(
        "{label}: the flat KV lane serves only loads off Metal; this load serves every turn \
         from the block-paged pool"
    )))
}

/// A load without a pool keeps its K/V in bf16: int8 K/V lives only in the
/// pool (Splash `RuntimeResourcesConfig::kvFormat`).
pub(crate) fn qwen35_no_pool_format(label: &str, format: KvFormat) -> Result<()> {
    match format {
        KvFormat::Bf16 => Ok(()),
        KvFormat::Int8 => Err(Error::from_reason(format!(
            "{label}: int8 K/V lives in the block-paged pool, which this load has none of (it \
             needs Metal); load with kvFormat 'bf16'"
        ))),
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
    /// The lane's draft ring and context window.
    pub lane_bytes: u64,
    pub weight_bytes: u64,
}

impl DraftTerms {
    pub(crate) fn of(config: &DFlash2Config, weight_bytes: u64) -> Self {
        Self {
            tap_layers: config.target_layers.len(),
            lane_bytes: config.lane_bytes(),
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
    /// The load can run an MTP turn on the pool lane (MTP heads, no
    /// DFlash2 draft): the lane holds its draft state at the window.
    pub mtp: bool,
    /// The most decode lanes the scheduler runs on this load
    /// ([`qwen35_scheduler_lanes`]); the plan grants at most this many.
    pub lanes: u32,
    /// Bytes each decode row's MoE layers allocate past the dense view
    /// (`Qwen3_5MoeConfig::decode_row_expert_bytes`); 0 for a dense model.
    pub expert_row_bytes: u64,
    /// The load keeps GDN ladder rungs a claim refused until the turn's
    /// cold-tier capture ([`qwen35_keeps_refused_rungs`]).
    pub refused_rungs: bool,
}

/// Whether a load keeps refused GDN ladder rungs: its paged cache persists
/// to the cold tier (`persist_cold`, [`crate::cold_tier::resolve_persist_cold`])
/// with a GDN sidecar, the case
/// [`super::paged_forward::gdn_cold_sidecar_ladder_wanted`] reads at run time.
pub(crate) fn qwen35_keeps_refused_rungs(
    config: &Qwen3_5Config,
    activation: DType,
    persist_cold: bool,
) -> bool {
    persist_cold
        && super::gdn_sidecar::policy(config, activation)
            .is_some_and(|policy| policy.group() == mlx_paged_attn::ColdGroup::GdnState)
}

/// Physical memory from which a Qwen3.5 load may run the full hybrid
/// live-state cap. A 256 GB Mac reports 256 GiB and a 192 GB Mac 192 GiB;
/// 256 * 10^9 bytes lies between them.
const ALL_LANES_PHYSICAL_BYTES: u64 = 256_000_000_000;

/// Lane ceiling of every smaller machine, or of one whose memory is
/// unreadable.
const SMALL_MACHINE_LANES: usize = 4;

/// The most decode lanes a Qwen3.5 load runs on a machine of
/// `physical_bytes`: the hybrid live-state cap from 256 GB, 4 below.
pub(crate) fn qwen35_lane_ceiling(physical_bytes: Option<u64>) -> usize {
    let all = crate::engine::recurrent_state::HYBRID_LIVE_STATE_UNITS;
    match physical_bytes {
        Some(bytes) if bytes >= ALL_LANES_PHYSICAL_BYTES => all,
        _ => SMALL_MACHINE_LANES.min(all),
    }
}

/// The decode lanes the scheduler can run on a Qwen3.5 load: this machine's
/// lane ceiling (`MLX_SCHED_MAX_NUM_SEQS` only lowers it), or one when every
/// turn runs whole-turn.
pub(crate) fn qwen35_scheduler_lanes() -> u32 {
    if super::model::Qwen35SchedulerState::force_serial() {
        1
    } else {
        crate::engine::hybrid_scheduler::scheduler_max_num_seqs_for(qwen35_lane_ceiling(
            crate::utils::host::physical_memory(),
        )) as u32
    }
}

/// Target rows of one decode step over `lanes` lanes, each at its widest
/// step: one AR row, or the widest MTP verify the scheduler reserves
/// (`MAX_DRAFTS` drafts and the anchor). A one-lane plan keeps the
/// transient [`pool_lane_transient`] prices.
fn decode_step_rows(lanes: u32, mtp: bool) -> u64 {
    let per_lane = if mtp {
        super::scheduled_mtp::MAX_DRAFTS as u64 + 1
    } else {
        1
    };
    u64::from(lanes.max(1)) * per_lane
}

/// Splash `evaluateEngineMemoryPlan` over `lanes` on the device `live`.
/// A pool load prices the first lane's state, which carries a DFlash2
/// draft's (Splash's one window for every turn); the plan then grants the
/// most decode lanes, up to `lanes.lanes`, whose states and joint decode
/// step keep the one-lane window. Lanes give way first: the window never
/// shrinks for a lane, and a load the one-lane plan admits always loads. A
/// load without a pool prices its footprint only.
pub(crate) fn evaluate_qwen35_plan(
    label: &str,
    lanes: &LoadLanes<'_>,
    live: LiveMemory,
    max_memory_bytes: Option<u64>,
) -> Result<MemoryPlan> {
    let single = evaluate_qwen35_lanes(lanes, live, max_memory_bytes, 1)
        .map_err(|error| Error::from_reason(error.message(label)))?;
    let Some(window) = single.pool.map(|pool| pool.lane.window.maximum()) else {
        return Ok(single);
    };
    for count in (2..=lanes.lanes).rev() {
        if let Ok(plan) = evaluate_qwen35_lanes(lanes, live, max_memory_bytes, count)
            && plan
                .pool
                .is_some_and(|pool| pool.lane.window.maximum() >= window)
        {
            return Ok(plan);
        }
    }
    Ok(single)
}

/// The plan of `lanes` with the decode step of `count` lanes.
fn evaluate_qwen35_lanes(
    lanes: &LoadLanes<'_>,
    live: LiveMemory,
    max_memory_bytes: Option<u64>,
    count: u32,
) -> std::result::Result<MemoryPlan, PlanError> {
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
        margin: crate::engine::memory_plan::Margin::Ledger,
    };
    let recurrent_state_bytes = lanes.recurrent_state_bytes;
    let trained = config.served_context_tokens();
    let fa_layers = config.full_attention_layer_count();
    let draft_taps = draft.map(|draft| draft.tap_layers);
    let mtp = lanes.mtp;
    let step_rows = decode_step_rows(count, mtp);
    let pool_transient = |window| {
        let one_lane = pool_lane_transient(config, draft_taps, window);
        if count > 1 {
            one_lane.max(
                pool_decode_step_bytes(config, step_rows, window)
                    .saturating_add(step_rows.saturating_mul(lanes.expert_row_bytes)),
            )
        } else {
            one_lane
        }
    };
    let draft_state =
        draft.map(|draft| dflash2_lane_state_bytes(recurrent_state_bytes, draft.lane_bytes));
    let mtp_state = |window: u32| {
        if mtp {
            super::scheduled_mtp::ScheduledMtpState::held_bytes(config, window)
        } else {
            0
        }
    };
    // Each lane past the first holds its own GDN state and, on an MTP load,
    // its own draft state at the full window. On a load that keeps refused
    // GDN ladder rungs, every lane may hold them beside its state until its
    // turn ends.
    let rungs = if lanes.refused_rungs {
        super::gdn_checkpoint_store::refused_rung_bytes(recurrent_state_bytes)
    } else {
        0
    };
    let lane_state = |window: u32| {
        let first = draft_state
            .unwrap_or_else(|| mtp_state(window))
            .saturating_add(rungs);
        let other = recurrent_state_bytes
            .saturating_add(mtp_state(window))
            .saturating_add(rungs);
        first.saturating_add(other.saturating_mul(u64::from(count.max(1) - 1)))
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
        }
    });
    let mut plan = MemoryPlan::evaluate(footprint, pool_input)?;
    if let Some(pool) = plan.pool.as_mut() {
        pool.lanes = count;
    }
    Ok(plan)
}

/// Every turn's window: the pool lane's.
fn turn_window(plan: &MemoryPlan) -> Option<ContextBounds> {
    plan.pool.as_ref().map(|pool| pool.lane.window)
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
/// when `pool` is set, else the footprint. `None` when the working set is
/// unreadable.
pub(crate) fn plan_qwen35_load<T: DFlash2Target>(
    label: &str,
    target: &T,
    recurrent_state_bytes: u64,
    mtp_weights: bool,
    pool: Option<PoolGeometry>,
    persist_cold: bool,
    max_memory_bytes: Option<u64>,
    max_context: Option<u32>,
) -> Result<Option<MemoryPlan>> {
    let draft = target.dflash2();
    let config = target.dense_config();
    let refused_rungs = pool.is_some()
        && qwen35_keeps_refused_rungs(&config, target.embedding_dtype()?, persist_cold);
    let Some(live) = live_memory() else {
        check_unplanned_max_context(label, &config, max_context)?;
        return Ok(None);
    };
    let lanes = LoadLanes {
        config: &config,
        recurrent_state_bytes,
        pool,
        draft: draft.map(|draft| DraftTerms::of(&draft.config, draft.weight_bytes)),
        mtp: mtp_weights && draft.is_none(),
        lanes: qwen35_scheduler_lanes(),
        expert_row_bytes: target.decode_row_expert_bytes(),
        refused_rungs,
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

/// The `decodeShare` load option (Splash `Engine::Engine`): nonnegative
/// and finite; unset is Splash's 0.5.
pub(crate) fn decode_share_option(label: &str, value: Option<f64>) -> Result<f64> {
    match value {
        None => Ok(crate::engine::scheduler::DEFAULT_DECODE_SHARE),
        Some(share) if share.is_finite() && share >= 0.0 => Ok(share),
        Some(_) => Err(Error::from_reason(format!(
            "{label} load: decode share must be nonnegative and finite"
        ))),
    }
}

/// What a load of a Qwen3.5 checkpoint would serve, priced before it loads.
#[napi(object)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35ContextEstimate {
    /// Tokens every turn may hold: the pool lane's (with a DFlash2 draft's
    /// state priced); the served window without a pool.
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
            let mut total = companion;
            for (path, _, _) in &checkpoint.main.parts {
                total = total.saturating_add(stat(path)?);
            }
            Ok(total)
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
        CheckpointSource::Gguf(checkpoint) => imported_dtypes(
            &checkpoint.main.header,
            &checkpoint.plans.0,
            &checkpoint.opts,
        ),
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

/// The loader's cold-tier decision from `config.json` and the environment
/// (a host's config overlay is not on disk).
fn estimate_persist_cold(model_type: &str, config_flag: Option<bool>) -> bool {
    let env = std::env::var("MLX_PERSIST_PAGED_CACHE").ok();
    crate::cold_tier::resolve_persist_cold(model_type, env.as_deref(), config_flag)
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
    let (mut config, mtp_body, expert_row_bytes, persist_cold) = if input.moe {
        let moe = crate::models::qwen3_5_moe::persistence::parse_config(&raw)?;
        let body = crate::models::qwen3_5_moe::mtp::Qwen3_5MoeMTPModule::mtp_mlp_variant(&moe);
        let persist_cold = estimate_persist_cold("qwen3_5_moe", moe.persist_paged_cache);
        (
            moe.to_dense_config(),
            body,
            moe.decode_row_expert_bytes(),
            persist_cold,
        )
    } else {
        let config = crate::models::qwen3_5::persistence::parse_config(&raw)?;
        let persist_cold = estimate_persist_cold("qwen3_5", config.persist_paged_cache);
        (config, DrafterBodyVariant::Dense, 0, persist_cold)
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
    let pool = if qwen35_pool_backend() {
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
        qwen35_no_pool_format(label, format)?;
        None
    };
    if draft.is_some() && pool.is_none() {
        return Err(Error::from_reason(format!(
            "{label} DFlash2 runs on the block-paged KV pool, which this load has none of (it \
             needs Metal); load without the companion (autoLoadDraft: false)"
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
        mtp: mtp_head && draft.is_none(),
        lanes: qwen35_scheduler_lanes(),
        expert_row_bytes,
        refused_rungs: pool.is_some() && qwen35_keeps_refused_rungs(&config, compute, persist_cold),
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
