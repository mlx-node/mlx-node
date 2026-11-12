//! The memory plan a Flash Next load serves its lanes from: the stores are
//! planned before any payload read (Splash `planStreamedMemory`), the lanes
//! after the load from what is resident (Splash `evaluateEngineMemoryPlan`).

use std::path::Path;

use napi::{Error, Result};
use napi_derive::napi;

use crate::array::DType;
use crate::engine::memory_plan::{
    KvLayout, LaneShape, LiveMemory, MemoryBound, MemoryFootprint, MemoryPlan, PoolInput,
    RESIDENT_ALLOWANCE_TOKENS, SPECULATIVE_SCRATCH_TOKENS, StorePlan, StreamedInputs, StreamedPart,
    allowance_dynamic_bytes, check_max_context, hard_budget_bytes, plan_streamed_store,
    table_share_bytes,
};
use crate::transformer::KvFormat;

use super::config::Config;
use super::memory::{draft_state_bytes, prefill_scratch, row_state_bytes};

/// The Flash Next KV page (Splash `QwenHybridLayout::kvLayout`): K/V of the
/// attention layers in the load's format and the indexer's bf16 block keys
/// beside them.
pub(super) fn flash_next_layout(c: &Config) -> KvLayout {
    KvLayout::of(
        full_attention_layers(c),
        c.num_key_value_heads as i32,
        c.head_dim as i32,
        DType::BFloat16,
        c.kv_format,
    )
    .with_index(c.indexer_head_dim as u64)
}

/// The page format of a load: `requested` (`kvFormat`), else int8 where the
/// int8 page store and dequantize serve the head (128 or 256 wide), else
/// bf16. Splash `RuntimeResources::create` takes `kvFormat` (int8 by default)
/// for every target and `storeSuffix` refuses a layout it has no kernel for.
pub(super) fn resolve_kv_format(
    label: &str,
    requested: Option<&str>,
    c: &Config,
) -> Result<KvFormat> {
    let served = matches!(c.head_dim, 128 | 256);
    let Some(requested) = requested else {
        return Ok(if served {
            KvFormat::Int8
        } else {
            KvFormat::Bf16
        });
    };
    let format = KvFormat::parse(Some(requested))
        .map_err(|e| Error::from_reason(format!("{label}: {e}")))?;
    if format == KvFormat::Int8 && !served {
        return Err(Error::from_reason(format!(
            "{label}: no int8 page kernels for head_dim {}; load with kvFormat 'bf16'",
            c.head_dim
        )));
    }
    Ok(format)
}

pub(super) fn full_attention_layers(c: &Config) -> usize {
    (0..c.num_hidden_layers).filter(|&i| !c.linear(i)).count()
}

/// One lane's transient for a `window`-token turn: one prefill chunk's
/// scratch at that context.
pub(super) fn transient(c: &Config, window: u32) -> u64 {
    prefill_scratch(c, super::MAX_PREFILL_CHUNK, window as usize)
}

/// Splash `servedFor` on `c`: YaRN past the native window, with the
/// startup line.
pub(super) fn serve_context(label: &str, c: &mut Config, max_context: Option<u32>) -> Result<()> {
    c.rope_scaling = crate::nn::rope_scaling::RopeScaling::served_for(
        label,
        c.max_position_embeddings.min(u32::MAX as usize) as u32,
        c.rope_scaling,
        max_context,
    )?;
    if c.rope_scaling.yarn() {
        tracing::info!(
            "{label} {}",
            c.rope_scaling.startup_line(c.served_context_tokens())
        );
    }
    Ok(())
}

/// Splash `ServingRuntime::start` on the AR lane, the default turn.
pub(super) fn check_context(
    label: &str,
    plan: &MemoryPlan,
    max_context: Option<u32>,
) -> Result<()> {
    match plan.pool.as_ref() {
        Some(pool) => check_max_context(pool.lane.window, plan.bound, max_context)
            .map_err(|error| Error::from_reason(error.message(label))),
        None => Ok(()),
    }
}

/// The pool lane: one cell and `state` held beside the pool, priced pages
/// from the runway, the served window.
pub(super) fn pool_input<'a>(
    c: &Config,
    transient: &'a dyn Fn(u32) -> u64,
    state: &'a dyn Fn(u32) -> u64,
) -> PoolInput<'a> {
    let layout = flash_next_layout(c);
    PoolInput {
        shape: LaneShape {
            trained: c.served_context_tokens(),
            lane_bytes: row_state_bytes(c),
            layout,
            scratch_rows: SPECULATIVE_SCRATCH_TOKENS,
            transient,
        },
        bytes_per_block: layout.page_bytes(),
        layers: full_attention_layers(c) as u64,
        runway_blocks: layout.runway_pages().min(u64::from(u32::MAX)) as u32,
        explicit_mib: None,
        lane_in_pool: false,
        state,
        state_in_pool: false,
    }
}

/// The lane's state besides the cell: with MTP weights, the MTP draft state
/// of a `W`-token turn, which grows with `W`. The lane is one and charged
/// once (Splash `evaluateEngineMemoryPlan` `availableForOneRequestKv`).
pub(super) fn mtp_state(c: &Config, has_mtp: bool) -> impl Fn(u32) -> u64 + '_ {
    move |window| {
        if has_mtp {
            draft_state_bytes(c, u64::from(window))
        } else {
            0
        }
    }
}

/// The rows a decoded-row cache holds for a 128K-token prompt (Splash
/// `tableShareBytes`): the PLE ids of every token and the token rows, decoded
/// BF16, beside the read staging.
pub(super) fn row_cache_share(c: &Config, read_staging: u64) -> StreamedPart {
    let ple = if c.ple_layer_ids.is_empty() {
        0
    } else {
        let ids = ((c.ngram_size - 1) * c.heads_per_ngram) as u64;
        let row_bytes = (c.ple_embed_dim as u64 / ids.max(1)) * 2;
        table_share_bytes(ids, c.ngram_vocab_size_base, row_bytes)
    };
    let token = table_share_bytes(1, c.vocab_size as u64, c.hidden_size as u64 * 2);
    StreamedPart {
        all: ple.saturating_add(token).saturating_add(read_staging),
        minimum: read_staging,
    }
}

/// What a load holds before any payload read, from the descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LoadInventory {
    /// The store's weights besides the experts and rows: the hot set less
    /// the routed experts, and the MTP draft banks.
    pub store_fixed_bytes: u64,
    /// The vision tower, decoded on its first image (Splash
    /// `visionWeightsBytes`).
    pub vision_bytes: u64,
    pub experts: StreamedPart,
    pub rows: StreamedPart,
}

/// The store plan and the dynamic budget whose lanes hold the allowance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StoreDecision {
    pub store: StorePlan,
    pub allowance: u64,
}

/// Splash `planStreamedMemory` for a target that cannot spill: the KV
/// allowance of the requested context, else `min(trained,
/// kResidentAllowanceTokens)`, on the load's one lane (with MTP weights it
/// holds the MTP draft state) comes before the experts.
pub(super) fn plan_store(
    label: &str,
    c: &Config,
    inventory: &LoadInventory,
    live: LiveMemory,
    max_memory: Option<u64>,
    max_context: Option<u32>,
    has_mtp: bool,
) -> Result<StoreDecision> {
    let transient = |window: u32| transient(c, window);
    let state = mtp_state(c, has_mtp);
    let pool = pool_input(c, &transient, &state);
    let lane = |dynamic: u64| pool.plan_at(dynamic).lane.window.maximum();
    let automatic = pool.shape.trained.min(RESIDENT_ALLOWANCE_TOKENS);
    let allowance = allowance_dynamic_bytes(&[&lane], max_context.unwrap_or(automatic));
    let minimum_dynamic = pool.plan_at(0).lane.minimum_bytes;
    let hard = hard_budget_bytes(live.working_set_bytes, max_memory);
    let fixed = live
        .active_bytes
        .saturating_add(live.registered_pool_bytes)
        .saturating_add(inventory.store_fixed_bytes)
        .saturating_add(inventory.vision_bytes);
    let store = plan_streamed_store(&StreamedInputs {
        hard,
        bound: MemoryBound::of(max_memory, hard),
        fixed,
        allowance,
        minimum_dynamic,
        experts: inventory.experts,
        rows: inventory.rows,
    })
    .map_err(|error| Error::from_reason(error.message(label)))?;
    Ok(StoreDecision { store, allowance })
}

/// Splash `evaluateEngineMemoryPlan` after the load: what is resident now,
/// with `held_back` bytes of the stores and the vision tower above it.
pub(super) fn plan_engine(
    label: &str,
    c: &Config,
    live: LiveMemory,
    held_back: u64,
    max_memory: Option<u64>,
    has_mtp: bool,
) -> Result<MemoryPlan> {
    let transient = |window: u32| transient(c, window);
    let state = mtp_state(c, has_mtp);
    let footprint = MemoryFootprint {
        working_set_bytes: live.working_set_bytes,
        max_memory_bytes: max_memory,
        active_bytes: live.active_bytes,
        other_pool_bytes: live.registered_pool_bytes,
        staging_bytes: 0,
        draft_weights_bytes: 0,
        held_back_bytes: held_back,
    };
    MemoryPlan::evaluate(
        footprint,
        Some(pool_input(c, &transient, &state)),
        None,
        None,
    )
    .map_err(|error| Error::from_reason(error.message(label)))
}

/// Splash plans the stores again with their loaded sizes (`storesPlan`): an
/// engine plan that leaves the lanes less than the allowance takes the
/// shortfall back from the store's held-back cap, which `lower` gives up
/// before any of it is allocated.
pub(super) fn settle(
    engine: &dyn Fn(u64) -> Result<MemoryPlan>,
    held_back: u64,
    allowance: u64,
    lower: &mut dyn FnMut(u64) -> Result<bool>,
) -> Result<MemoryPlan> {
    let plan = engine(held_back)?;
    let shortfall = allowance.saturating_sub(plan.dynamic_budget_bytes);
    if shortfall == 0 || shortfall > held_back || !lower(shortfall)? {
        return Ok(plan);
    }
    engine(held_back - shortfall)
}

/// The checkpoint's config: the GGUF's metadata, or `config.json` beside
/// the safetensors.
pub(super) fn checkpoint_config(weights: &super::weights::Store, path: &Path) -> Result<Config> {
    if weights.gguf {
        return super::gguf::config(weights);
    }
    let raw =
        std::fs::read(path.join("config.json")).map_err(|e| Error::from_reason(e.to_string()))?;
    let raw = serde_json::from_slice(&raw).map_err(|e| Error::from_reason(e.to_string()))?;
    Config::parse(&raw)
}

/// Attach the auxiliary checkpoint's descriptors (MTP and vision a GGUF
/// omits) and return the raw config the vision tower is read from.
pub(super) fn attach_auxiliary(
    weights: &mut super::weights::Store,
    config: &mut Config,
    path: &Path,
    auxiliary: Option<&str>,
) -> Result<serde_json::Value> {
    if let Some(aux) = auxiliary {
        return super::auxiliary::attach(weights, config, Path::new(aux));
    }
    if weights.gguf {
        return Ok(serde_json::Value::Null);
    }
    Ok(serde_json::from_slice(&std::fs::read(
        path.join("config.json"),
    )?)?)
}

/// What a load of a Flash Next checkpoint would serve, priced before it
/// loads.
#[napi(object)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4ExpContextEstimate {
    /// Tokens any turn (AR, image, MTP) may hold.
    pub window_tokens: u32,
    /// `'memory'` when the plan binds the window, `'model'` when the
    /// trained window does.
    pub bound: String,
}

/// The plan a load makes, from the descriptors: the store plan, then the
/// engine plan with the fixed weights loaded and the stores at their cap.
pub(super) fn estimate(
    label: &str,
    c: &Config,
    inventory: &LoadInventory,
    live: LiveMemory,
    max_memory: Option<u64>,
    max_context: Option<u32>,
    has_mtp: bool,
) -> Result<Qwen4ExpContextEstimate> {
    let decision = plan_store(label, c, inventory, live, max_memory, max_context, has_mtp)?;
    let loaded = LiveMemory {
        active_bytes: live
            .active_bytes
            .saturating_add(inventory.store_fixed_bytes),
        ..live
    };
    let held_back = decision
        .store
        .expert_cap_bytes
        .saturating_add(decision.store.row_cache_bytes)
        .saturating_add(inventory.vision_bytes);
    let plan = plan_engine(label, c, loaded, held_back, max_memory, has_mtp)?;
    check_context(label, &plan, max_context)?;
    let pool = plan.pool.as_ref().ok_or_else(|| {
        Error::from_reason(format!("{label} memory plan: the load planned no pool"))
    })?;
    let cap = max_context.unwrap_or_else(|| c.served_context_tokens());
    Ok(Qwen4ExpContextEstimate {
        window_tokens: pool.lane.window.maximum().min(cap),
        bound: pool.lane.window.bound().to_string(),
    })
}

/// The load-free estimate behind `/v1/models`: the plan a load of
/// `model_path` with `options` would make, from the tensors' descriptors
/// (no payload read, no pack sweep, no asset written). `None` when the Metal
/// working set is unreadable.
#[napi]
pub async fn estimate_qwen4_exp_context_window(
    model_path: String,
    options: Option<super::Qwen4ExpLoadOptions>,
) -> Result<Option<Qwen4ExpContextEstimate>> {
    const LABEL: &str = "Qwen4Exp";
    let options = options.unwrap_or_default();
    let max_memory =
        crate::models::qwen3_5::load_plan::max_memory_option(LABEL, options.max_memory_bytes)?;
    let max_context =
        crate::nn::rope_scaling::max_context_option(LABEL, options.max_context_tokens)?;
    napi::bindgen_prelude::spawn_blocking(move || {
        let Some(live) = crate::engine::memory_plan::live_memory() else {
            return Ok(None);
        };
        let path = Path::new(&model_path);
        let mut weights = super::weights::Store::open_descriptors(path)?;
        let mut config = checkpoint_config(&weights, path)?;
        let raw = attach_auxiliary(
            &mut weights,
            &mut config,
            path,
            options.auxiliary_model_path.as_deref(),
        )?;
        serve_context(LABEL, &mut config, max_context)?;
        config.kv_format = resolve_kv_format(LABEL, options.kv_format.as_deref(), &config)?;
        let has_mtp = super::auxiliary::validate_mtp(&weights, &config)?;
        let vision = super::media::Vision::metadata(&raw, &weights, &config)?.available();
        let inventory = weights.load_inventory(&config, vision)?;
        estimate(
            LABEL,
            &config,
            &inventory,
            live,
            max_memory,
            max_context,
            has_mtp,
        )
        .map(Some)
    })
    .await
    .map_err(|error| Error::from_reason(format!("{LABEL} estimate: {error}")))?
}

#[cfg(test)]
pub(super) mod tests;
