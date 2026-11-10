//! The memory plan a dense or MoE Qwen3.5 load serves its lanes from.

use std::path::Path;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::array::DType;
use crate::engine::checkpoint_source::CheckpointSource;
use crate::engine::memory_plan::{
    ContextBounds, FlatInput, KvLayout, LaneShape, LiveMemory, MeasuredMemory, MemoryFootprint,
    MemoryPlan, PlanError, PoolInput, SPECULATIVE_SCRATCH_TOKENS, check_max_context, live_memory,
    speculative_lane_bytes,
};
use crate::models::quant_dispatch::{
    has_sym8_mode, parse_quant_settings, select_quantization_block,
};
use crate::models::quantized_linear::{DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE};
use crate::nn::rope_scaling::RopeScaling;
use crate::transformer::KvFormat;
use crate::utils::gguf::NativeGgufFamily;

use super::config::Qwen3_5Config;
use super::dflash2::DFlash2Config;
use super::dflash2_decode::{
    PrefillSdpaKernels, dflash2_lane_transient, flat_lane_transient, pool_lane_transient,
};
use super::dflash2_turn::{DFlash2Target, dflash2_target_kv_dtype, dflash2_target_kv_format};

/// Tokens of the pool every load starts with (Splash `warmupKvPages`).
const RUNWAY_TOKENS: u32 = 2048;

/// The blocks a pool starts with: the runway's tokens and one request's
/// recurrent state.
pub(crate) fn qwen35_runway_blocks(
    block_size: u32,
    bytes_per_block: u64,
    recurrent_state_bytes: u64,
) -> u32 {
    RUNWAY_TOKENS.div_ceil(block_size.max(1)).saturating_add(
        recurrent_state_bytes
            .div_ceil(bytes_per_block.max(1))
            .min(u64::from(u32::MAX)) as u32,
    )
}

/// The paged pool a load builds: BF16 blocks over the full-attention layers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PoolGeometry {
    pub bytes_per_block: u64,
    pub block_size: u32,
    pub layers: u32,
    /// `paged_cache_memory_mb`.
    pub explicit_mib: Option<u32>,
}

/// What a DFlash2 draft adds to a plan.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DraftTerms {
    pub block_size: usize,
    pub tap_layers: usize,
    pub ring_bytes: u64,
    pub weight_bytes: u64,
}

impl DraftTerms {
    pub(crate) fn of(config: &DFlash2Config, weight_bytes: u64) -> Self {
        Self {
            block_size: config.block_size,
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
    /// Format and dtype of the flat K/V (the DFlash2 lane's, or the flat
    /// lane's without a pool and a draft).
    pub format: KvFormat,
    pub dtype: DType,
}

/// Splash `evaluateEngineMemoryPlan` over `lanes` on the device `live`.
/// Every load prices a lane: one with neither pool nor draft serves its
/// turns from flat K/V, priced as the flat lane.
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
    let kernels = PrefillSdpaKernels::probe(config.head_dim);
    let pool_transient = |window| pool_lane_transient(config, DType::BFloat16, kernels, window);
    let pool_input = lanes.pool.map(|geometry| PoolInput {
        shape: LaneShape {
            trained,
            lane_bytes: recurrent_state_bytes,
            layout: KvLayout::of(
                fa_layers,
                config.num_kv_heads,
                config.head_dim,
                DType::BFloat16,
                KvFormat::Bf16,
            ),
            scratch_rows: SPECULATIVE_SCRATCH_TOKENS,
            transient: &pool_transient,
        },
        bytes_per_block: geometry.bytes_per_block,
        layers: u64::from(geometry.layers),
        runway_blocks: qwen35_runway_blocks(
            geometry.block_size,
            geometry.bytes_per_block,
            recurrent_state_bytes,
        ),
        explicit_mib: geometry.explicit_mib,
        lane_in_pool: true,
    });
    let (block_size, tap_layers) =
        draft.map_or((0, 0), |draft| (draft.block_size, draft.tap_layers));
    let format = lanes.format;
    let flat_transient =
        |window| dflash2_lane_transient(config, block_size, format, tap_layers, window);
    let speculative = draft.map(|draft| FlatInput {
        shape: LaneShape {
            trained,
            lane_bytes: speculative_lane_bytes(recurrent_state_bytes, draft.ring_bytes),
            layout: KvLayout::of(
                fa_layers,
                config.num_kv_heads,
                config.head_dim,
                lanes.dtype,
                format,
            ),
            scratch_rows: draft.block_size as u64,
            transient: &flat_transient,
        },
        copy_layers: Some(fa_layers as u64),
    });
    let autoregressive_transient = |window| flat_lane_transient(config, format, window);
    let flat = (lanes.pool.is_none() && draft.is_none()).then(|| FlatInput {
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
    MemoryPlan::evaluate(footprint, pool_input, speculative, flat)
        .map_err(|error| Error::from_reason(error.message(label)))
}

/// The lane a default turn takes: DFlash2 with a draft, else the pool,
/// else the flat lane of a load with neither.
fn default_turn_window(plan: &MemoryPlan) -> Option<ContextBounds> {
    plan.speculative
        .as_ref()
        .map(|lane| lane.window)
        .or(plan.pool.as_ref().map(|pool| pool.lane.window))
        .or(plan.flat.as_ref().map(|lane| lane.window))
}

/// Splash `ServingRuntime::start`'s `--max-context` refusal over `plan`.
pub(crate) fn check_qwen35_max_context(
    label: &str,
    plan: &MemoryPlan,
    max_context: Option<u32>,
) -> Result<()> {
    match default_turn_window(plan) {
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
/// when `pool` is set, the DFlash2 lane with a draft, the flat lane with
/// neither. `None` when the working set is unreadable.
pub(crate) fn plan_qwen35_load<T: DFlash2Target>(
    label: &str,
    target: &T,
    recurrent_state_bytes: u64,
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
        format: dflash2_target_kv_format(target),
        dtype: dflash2_target_kv_dtype(target)?,
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
    /// Tokens a non-DFlash2 turn may hold: the pool lane's, the flat lane's
    /// without a pool and a draft, the trained window with a draft and no
    /// pool.
    pub window_tokens: u32,
    /// Tokens a DFlash2 turn may hold; set only with a draft.
    pub speculative_window_tokens: Option<u32>,
    /// `'memory'` when the plan binds the window a default turn takes,
    /// `'model'` when the trained window does.
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

pub(crate) struct EstimateInput<'a> {
    /// The directory holding the checkpoint's `config.json`.
    pub asset_dir: &'a Path,
    pub moe: bool,
    pub weight_bytes: u64,
    pub draft_dir: Option<&'a Path>,
    pub kv_format: Option<KvFormat>,
    pub max_memory_bytes: Option<u64>,
    pub max_context: Option<u32>,
}

/// The plan a load of `input` would make on the device now, with the
/// target's and the draft's weight files counted as active bytes. The
/// compute dtype is taken as BF16: the weights that decide it are not read.
pub(crate) fn estimate_qwen35_load(
    label: &str,
    input: &EstimateInput<'_>,
) -> Result<Option<Qwen35ContextEstimate>> {
    let config_path = input.asset_dir.join("config.json");
    let raw: serde_json::Value = std::fs::read_to_string(&config_path)
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
        })?;
    let mut config = if input.moe {
        crate::models::qwen3_5_moe::persistence::parse_config(&raw)?.to_dense_config()
    } else {
        crate::models::qwen3_5::persistence::parse_config(&raw)?
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
    // Both loaders pin a sym8 checkpoint to flat KV after reading its quant block.
    if has_sym8_mode(top_level_mode, &per_layer_quant) && config.use_block_paged_cache == Some(true)
    {
        config.use_block_paged_cache = Some(false);
    }
    KvFormat::parse(config.kv_format.as_deref())
        .map_err(|message| Error::from_reason(format!("config.json: {message}")))?;
    let format = config.pin_kv_format(input.kv_format, DType::BFloat16);
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
        let bytes_per_block = mlx_paged_attn::profile::bytes_per_block(
            layers,
            config.num_kv_heads as u32,
            config.head_dim as u32,
            block_size,
            mlx_paged_attn::metal::MetalDtype::BFloat16,
        )
        .map_err(|error| Error::from_reason(format!("{label} block-paged adapter: {error}")))?;
        Some(PoolGeometry {
            bytes_per_block,
            block_size,
            layers,
            explicit_mib: config.paged_cache_memory_mb,
        })
    } else {
        None
    };
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
        dtype: DType::BFloat16,
    };
    let plan = evaluate_qwen35_plan(label, &lanes, live, input.max_memory_bytes)?;
    check_qwen35_max_context(label, &plan, input.max_context)?;
    let window = plan
        .pool
        .as_ref()
        .map(|pool| pool.lane)
        .or(plan.flat)
        .map(|lane| lane.window);
    let speculative_window = plan.speculative.as_ref().map(|lane| lane.window);
    let capped = |tokens: u32| tokens.min(cap);
    Ok(Some(Qwen35ContextEstimate {
        window_tokens: capped(window.map_or(served, |window| window.maximum())),
        speculative_window_tokens: speculative_window.map(|window| capped(window.maximum())),
        bound: default_turn_window(&plan)
            .map_or("model", |window| window.bound())
            .to_string(),
    }))
}

#[cfg(test)]
mod tests;
