//! The memory plan a dense or MoE Qwen3.5 load serves its lanes from.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::array::DType;
use crate::disk_tier::CacheDisk;
use crate::engine::checkpoint_source::CheckpointSource;
use crate::engine::memory_plan::{
    ContextBounds, KvLayout, LaneShape, LiveMemory, MeasuredMemory, MemoryFootprint, MemoryPlan,
    PlanError, PoolInput, SPECULATIVE_SCRATCH_TOKENS, check_max_context, dflash2_lane_state_bytes,
    lane_ceiling_for_physical_memory, live_memory, select_max_lanes_preserving_window,
};
pub(crate) use crate::engine::memory_plan::{decode_share_option, max_memory_option};
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
    compute_dtype_of, mtp_expert_keys, mtp_sidecar_candidates, normalize_mtp_weight_key,
    qwen35_text_key, settled_compute_dtype,
};

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
    /// The 4-bit context window a stored state keeps.
    pub window_bytes: u64,
    pub weight_bytes: u64,
}

impl DraftTerms {
    pub(crate) fn of(config: &DFlash2Config, weight_bytes: u64) -> Self {
        Self {
            tap_layers: config.target_layers.len(),
            lane_bytes: config.lane_bytes(),
            window_bytes: config.context_window_bytes(),
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
    /// The scheduler runs `draft`'s turns as DFlash2 decode lanes: every
    /// lane, not only the first, holds the draft's lane state and its
    /// pending GDN tape.
    pub draft_lanes: bool,
    /// The most decode lanes the scheduler runs on this load
    /// ([`qwen35_scheduler_lanes`]); the plan grants at most this many.
    pub lanes: u32,
    /// Bytes each decode row's MoE layers allocate past the dense view
    /// (`Qwen3_5MoeConfig::decode_row_expert_bytes`); 0 for a dense model.
    pub expert_row_bytes: u64,
    /// The SSD tier's state staging the plan sets aside.
    pub state_staging_bytes: u64,
}

/// The most decode lanes a Qwen3.5 load runs on a machine of
/// `physical_bytes`: the hybrid live-state cap from 256 GB, 4 below.
pub(crate) fn qwen35_lane_ceiling(physical_bytes: Option<u64>) -> usize {
    lane_ceiling_for_physical_memory(
        physical_bytes,
        crate::engine::recurrent_state::HYBRID_LIVE_STATE_UNITS,
    )
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
    let (plan, _) = select_max_lanes_preserving_window(
        lanes.lanes,
        |count| evaluate_qwen35_lanes(lanes, live, max_memory_bytes, count),
        |plan| plan.pool.map(|pool| pool.lane.window.maximum()),
    )
    .map_err(|error| Error::from_reason(error.message(label)))?;
    Ok(plan)
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
        state_staging_bytes: lanes.state_staging_bytes,
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
        let verifying = if lanes.draft_lanes { count } else { 1 };
        let verify = pool_lane_transient(config, draft_taps, window, verifying);
        if count > 1 {
            verify.max(
                pool_decode_step_bytes(config, step_rows, window)
                    .saturating_add(step_rows.saturating_mul(lanes.expert_row_bytes)),
            )
        } else {
            verify
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
    let draft_lane = draft.filter(|_| lanes.draft_lanes).map(|draft| {
        super::scheduled_dflash2::lane_state_bytes(config, recurrent_state_bytes, draft.lane_bytes)
    });
    // Each lane past the first holds its own GDN state and, on an MTP load,
    // its own draft state at the full window; a DFlash2 lane its draft
    // state and its pending GDN tape (the first lane's tape is in the
    // one-lane transient).
    let lane_state = |window: u32| {
        let first = draft_state.unwrap_or_else(|| mtp_state(window));
        let other =
            recurrent_state_bytes.saturating_add(draft_lane.unwrap_or_else(|| mtp_state(window)));
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

/// Bytes of one stored state of `lanes`' load (Splash
/// `CompositeStateLayout::cachedBytes`: the GDN cell and the draft window,
/// the cell alone without a draft); 0 without a pool.
fn lanes_state_bytes(lanes: &LoadLanes<'_>) -> u64 {
    match (lanes.draft, lanes.pool) {
        (Some(draft), Some(_)) => lanes
            .recurrent_state_bytes
            .saturating_add(draft.window_bytes),
        (None, Some(_)) => lanes.recurrent_state_bytes,
        _ => 0,
    }
}

/// Splash `RuntimeResources` staging rules over `evaluate_qwen35_plan`: a
/// draft load whose SSD tier quota holds one state sets that state's staging
/// aside; the automatic tier turns off (`diskOff`) when the plan cannot hold
/// the model beside it, or `--max-context` beside it while it can without.
/// The plan's `state_staging_bytes` says whether the tier may open.
pub(crate) fn plan_with_disk(
    label: &str,
    lanes: &mut LoadLanes<'_>,
    live: LiveMemory,
    max_memory_bytes: Option<u64>,
    max_context: Option<u32>,
    cache_disk: CacheDisk,
) -> Result<MemoryPlan> {
    let state = lanes_state_bytes(lanes);
    lanes.state_staging_bytes = 0;
    if state > 0 && cache_disk != CacheDisk::Off {
        if cache_disk.nominal_bytes() >= state {
            lanes.state_staging_bytes = state;
        } else {
            tracing::info!(
                "{label} SSD cache off: its quota holds no state of {} MiB; no state staging is \
                 set aside.",
                state.div_ceil(1 << 20)
            );
        }
    }
    if lanes.state_staging_bytes > 0 && cache_disk == CacheDisk::Auto {
        let off = match evaluate_qwen35_plan(label, lanes, live, max_memory_bytes) {
            Err(_) => Some("the memory plan cannot hold the model beside its state staging"),
            Ok(plan) if check_qwen35_max_context(label, &plan, max_context).is_err() => {
                let staging = std::mem::take(&mut lanes.state_staging_bytes);
                let without = evaluate_qwen35_plan(label, lanes, live, max_memory_bytes)
                    .is_ok_and(|plan| check_qwen35_max_context(label, &plan, max_context).is_ok());
                lanes.state_staging_bytes = staging;
                without.then_some(
                    "the memory plan cannot hold the requested context beside its state staging",
                )
            }
            Ok(_) => None,
        };
        if let Some(reason) = off {
            tracing::info!("{label} SSD cache off: {reason}; no state staging is set aside.");
            lanes.state_staging_bytes = 0;
        }
    }
    let plan = evaluate_qwen35_plan(label, lanes, live, max_memory_bytes)?;
    check_qwen35_max_context(label, &plan, max_context)?;
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
/// when `pool` is set, else the footprint; `draft_lanes` when the scheduler
/// runs the draft's turns as decode lanes. `None` when the working set is
/// unreadable.
pub(crate) fn plan_qwen35_load<T: DFlash2Target>(
    label: &str,
    target: &T,
    recurrent_state_bytes: u64,
    mtp_weights: bool,
    draft_lanes: bool,
    pool: Option<PoolGeometry>,
    max_memory_bytes: Option<u64>,
    max_context: Option<u32>,
    cache_disk: CacheDisk,
) -> Result<Option<MemoryPlan>> {
    let Some(plan) = plan_qwen35_load_freed(
        label,
        target,
        recurrent_state_bytes,
        mtp_weights,
        draft_lanes,
        pool,
        max_memory_bytes,
        max_context,
        cache_disk,
        0,
    )?
    else {
        return Ok(None);
    };
    if let Some(line) = plan.max_memory_ignored() {
        tracing::info!("{label} {line}");
    }
    tracing::info!("{label} memory plan:\n{}", plan.describe());
    Ok(Some(plan))
}

/// [`plan_qwen35_load`] as if `freed_bytes` of what the device holds now
/// were free, with no log line.
pub(crate) fn plan_qwen35_load_freed<T: DFlash2Target>(
    label: &str,
    target: &T,
    recurrent_state_bytes: u64,
    mtp_weights: bool,
    draft_lanes: bool,
    pool: Option<PoolGeometry>,
    max_memory_bytes: Option<u64>,
    max_context: Option<u32>,
    cache_disk: CacheDisk,
    freed_bytes: u64,
) -> Result<Option<MemoryPlan>> {
    let draft = target.dflash2();
    let config = target.dense_config();
    let Some(mut live) = live_memory() else {
        check_unplanned_max_context(label, &config, max_context)?;
        return Ok(None);
    };
    live.active_bytes = live.active_bytes.saturating_sub(freed_bytes);
    let mut lanes = LoadLanes {
        config: &config,
        recurrent_state_bytes,
        pool,
        draft: draft.map(|draft| DraftTerms::of(&draft.config, draft.weight_bytes)),
        mtp: mtp_weights && draft.is_none(),
        draft_lanes,
        lanes: qwen35_scheduler_lanes(),
        expert_row_bytes: target.decode_row_expert_bytes(),
        state_staging_bytes: 0,
    };
    plan_with_disk(
        label,
        &mut lanes,
        live,
        max_memory_bytes,
        max_context,
        cache_disk,
    )
    .map(Some)
}

/// The most of the pool's blocks a load's GGUF planes may take.
pub(crate) const PLANES_POOL_LOSS_LIMIT: f64 = 0.15;

/// Whether a load keeps its GGUF planes of `planes_bytes`: `with` is its plan
/// with the planes resident, `without` the plan with those bytes free. The
/// planes stay only when both plans grant the same lanes and window and the
/// pool loses at most [`PLANES_POOL_LOSS_LIMIT`] of its blocks. `Ok` holds
/// the ledger line, `Err` the reason they leave.
pub(crate) fn planes_ledger(
    planes_bytes: u64,
    with: &MemoryPlan,
    without: &MemoryPlan,
) -> std::result::Result<String, String> {
    let (Some(a), Some(b)) = (with.pool.as_ref(), without.pool.as_ref()) else {
        return Err("the load plans no pool".to_string());
    };
    if a.lanes != b.lanes {
        return Err(format!("{} lanes, {} without them", a.lanes, b.lanes));
    }
    let (wa, wb) = (a.lane.window.maximum(), b.lane.window.maximum());
    if wa != wb {
        return Err(format!("a window of {wa} tokens, {wb} without them"));
    }
    let lost = b.max_blocks.saturating_sub(a.max_blocks);
    let share = f64::from(lost) / f64::from(b.max_blocks.max(1));
    let line = format!(
        "planes {planes_bytes} B in the weights: pool {} of {} max blocks ({:.1} % less), lanes {}, \
         window {wa}",
        a.max_blocks,
        b.max_blocks,
        share * 100.0,
        a.lanes
    );
    if share > PLANES_POOL_LOSS_LIMIT {
        return Err(format!(
            "{line}; more than {:.0} % of the pool",
            PLANES_POOL_LOSS_LIMIT * 100.0
        ));
    }
    Ok(line)
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

/// The activations of a load whose options name none: `Auto` for a dense
/// load with a DFlash2 draft (Splash's default, `RuntimeResources`: int8
/// queries from KV position 7,168 where the device scores them), `Bf16` for
/// a dense load with no draft (its autoregressive decode reads slower with
/// int8 queries) and for MoE.
pub(crate) fn default_activations(moe: bool, draft: bool) -> crate::transformer::Activations {
    if !moe && draft {
        crate::transformer::Activations::Auto
    } else {
        crate::transformer::Activations::Bf16
    }
}

/// The page of a dense Qwen3.5 pool: its block and its key order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PoolPage {
    pub block_size: u32,
    /// Keys token-major, values dimension-major
    /// (`mlx_paged_attn::PageLayout::keys_token_major`).
    pub keys_token_major: bool,
}

/// What [`pool_page`] reads of a load.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PageInputs {
    pub heads: PoolHeads,
    pub cache_dtype: mlx_paged_attn::metal::MetalDtype,
    /// The load has a DFlash2 draft. Such a load plans no native MTP turn
    /// (its speculative plan is the draft's), so MTP weights it ships are not
    /// read on the pool.
    pub draft: bool,
    /// `paged_block_size` in the checkpoint's config.
    pub explicit_block: Option<u32>,
    /// The GPU builds the in-place prefill attention pipelines.
    pub tile: bool,
}

/// The page a dense Qwen3.5 pool takes (Splash's attention page: 32-token
/// blocks, keys token-major, `paged_attention_tile.h`):
///
/// | load | page |
/// |---|---|
/// | `paged_block_size` set | that block, vLLM page |
/// | heads 24/4, head 256, int8 pool, DFlash2 draft, tile GPU | 32, keys token-major |
/// | any other | 16, vLLM page |
///
/// A store a 16-token load wrote moves into the new page at the first load
/// (`dflash2_state::migrate`).
pub(crate) fn pool_page(load: &PageInputs) -> PoolPage {
    let vllm_page = PoolPage {
        block_size: load.explicit_block.unwrap_or(16),
        keys_token_major: false,
    };
    let shape = load.heads.query == 24 && load.heads.kv == 4 && load.heads.dim == 256;
    if load.explicit_block.is_some()
        || !shape
        || load.cache_dtype != mlx_paged_attn::metal::MetalDtype::Int8
        || !load.draft
        || !load.tile
    {
        return vllm_page;
    }
    PoolPage {
        block_size: 32,
        keys_token_major: true,
    }
}

/// Whether this GPU builds the in-place prefill attention pipelines.
pub(crate) fn device_builds_tile() -> bool {
    #[cfg(target_os = "macos")]
    {
        crate::transformer::paged_kv_cache_adapter::device_prefill_mpp_builds()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
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
    let cache_disk = options.disk_tier(label)?.cache_disk;
    let source = CheckpointSource::resolve(Path::new(&model_path), family).await?;
    let estimate = checkpoint_weight_bytes(&source).and_then(|weight_bytes| {
        estimate_qwen35_load(
            label,
            &EstimateInput {
                asset_dir: source.asset_dir(),
                moe,
                weight_bytes,
                planes_bytes: planes_estimate_bytes(
                    &source,
                    moe,
                    options.draft_model_path.is_some(),
                    max_memory_bytes,
                ),
                header: checkpoint_header(&source, moe)?,
                draft_dir: options.draft_model_path.as_deref().map(Path::new),
                kv_format,
                max_memory_bytes,
                max_context,
                cache_disk,
            },
        )
    });
    source.release_lease();
    estimate
}

/// The bytes of the GGUF planes a dense load of `source` would write: the
/// plan a load makes from the header, the device and the load's draft and
/// memory cap. A load that cannot read its planes in whole 64-row bands or
/// whose device refuses a pipeline writes fewer.
pub(crate) fn planes_estimate_bytes(
    source: &CheckpointSource,
    moe: bool,
    draft: bool,
    max_memory_bytes: Option<u64>,
) -> u64 {
    let CheckpointSource::Gguf(checkpoint) = source else {
        return 0;
    };
    if moe {
        return 0;
    }
    crate::utils::gguf_planes::plan_for_load(
        &checkpoint.main.header,
        &crate::utils::gguf_planes::PlanesDevice::this(max_memory_bytes),
        &crate::utils::gguf_planes::PlanesLoad::of_load(draft, false, checkpoint.main.parts.len()),
    )
    .bytes()
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

pub(crate) struct EstimateInput<'a> {
    /// The directory holding the checkpoint's `config.json`.
    pub asset_dir: &'a Path,
    pub moe: bool,
    pub weight_bytes: u64,
    /// The GGUF planes a load would write ([`planes_estimate_bytes`]).
    pub planes_bytes: u64,
    pub header: CheckpointHeader,
    pub draft_dir: Option<&'a Path>,
    pub kv_format: Option<KvFormat>,
    pub max_memory_bytes: Option<u64>,
    pub max_context: Option<u32>,
    pub cache_disk: CacheDisk,
}

/// The plan a load of `input` would make on the device now, with the
/// target's and the draft's weight files counted as active bytes; with its
/// planes when [`planes_ledger`] keeps them.
pub(crate) fn estimate_qwen35_load(
    label: &str,
    input: &EstimateInput<'_>,
) -> Result<Option<Qwen35ContextEstimate>> {
    let Some(plans) = estimate_plans(label, input)? else {
        return Ok(None);
    };
    let plan = match plans.with {
        Some(Ok(with)) if planes_ledger(input.planes_bytes, &with, &plans.without).is_ok() => with,
        _ => plans.without,
    };
    let window = turn_window(&plan);
    Ok(Some(Qwen35ContextEstimate {
        window_tokens: window
            .map_or(plans.served, |window| window.maximum())
            .min(plans.cap),
        bound: window.map_or("model", |window| window.bound()).to_string(),
    }))
}

/// Whether a load of `input` may write its `input.planes_bytes` of GGUF
/// planes, judged before its import from the device now with the target's
/// and the draft's weight files counted as active bytes: [`planes_ledger`]
/// of the plans with and without them. A load that would lose a lane, its
/// window or too much of its pool to them never allocates them. `Ok` holds
/// the ledger line, `Err` why they leave.
pub(crate) fn admit_planes(
    label: &str,
    input: &EstimateInput<'_>,
) -> std::result::Result<String, String> {
    let plans = estimate_plans(label, input)
        .map_err(|error| error.reason)?
        .ok_or_else(|| "the working set is unreadable".to_string())?;
    let with = plans
        .with
        .ok_or_else(|| "no planes to price".to_string())?
        .map_err(|error| error.reason)?;
    planes_ledger(input.planes_bytes, &with, &plans.without)
}

/// The plans of [`estimate_qwen35_load`]: `with` holds the plan with the
/// planes resident when `input` has planes.
struct EstimatePlans {
    without: MemoryPlan,
    with: Option<Result<MemoryPlan>>,
    served: u32,
    cap: u32,
}

fn estimate_plans(label: &str, input: &EstimateInput<'_>) -> Result<Option<EstimatePlans>> {
    let raw = read_config_json(input.asset_dir)?;
    let (mut config, mtp_body, expert_row_bytes) = if input.moe {
        let moe = super::persistence::parse_moe_config(&raw)?;
        let (base, moe_params) = moe.split();
        let body = super::mtp::Qwen3_5MTPModule::mtp_mlp_variant(&base, Some(&moe_params));
        let expert_bytes = moe_params.decode_row_expert_bytes(&base);
        (base, body, expert_bytes)
    } else {
        (
            crate::models::qwen3_5::persistence::parse_config(&raw)?,
            DrafterBodyVariant::Dense,
            0,
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
        let heads = PoolHeads::of(&config);
        let config_block = config.paged_block_size();
        let cache_dtype = qwen35_pool_cache_dtype(label, format, heads, config_block, compute)?;
        let block_size = if input.moe {
            config_block
        } else {
            pool_page(&PageInputs {
                heads,
                cache_dtype,
                draft: draft.is_some(),
                explicit_block: config.paged_block_size,
                tile: device_builds_tile(),
            })
            .block_size
        };
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
    if input.moe && draft.is_some() && pool.is_none() {
        return Err(Error::from_reason(format!(
            "{label} MoE DFlash2 runs on the block-paged KV pool, which this load has none of (it \
             needs Metal); load without the companion (autoLoadDraft: false)"
        )));
    }
    // Dense off-Metal loads use the eager flat DFlash2 lane and plan only the
    // resident target and draft weights. MoE keeps its Metal pool requirement.
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
    let mut lanes = LoadLanes {
        config: &config,
        recurrent_state_bytes: config.recurrent_state_bytes(),
        pool,
        draft,
        mtp: mtp_head && draft.is_none(),
        draft_lanes: !input.moe,
        lanes: qwen35_scheduler_lanes(),
        expert_row_bytes,
        state_staging_bytes: 0,
    };
    let without = plan_with_disk(
        label,
        &mut lanes,
        live,
        input.max_memory_bytes,
        input.max_context,
        input.cache_disk,
    )?;
    let with = (input.planes_bytes > 0).then(|| {
        let mut with_live = live;
        with_live.active_bytes = with_live.active_bytes.saturating_add(input.planes_bytes);
        plan_with_disk(
            label,
            &mut lanes,
            with_live,
            input.max_memory_bytes,
            input.max_context,
            input.cache_disk,
        )
    });
    Ok(Some(EstimatePlans {
        without,
        with,
        served,
        cap,
    }))
}

#[cfg(test)]
mod tests;
