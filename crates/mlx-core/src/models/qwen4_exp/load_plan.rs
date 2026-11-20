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
use super::memory::{
    draft_hidden_row_bytes, draft_index_bytes_per_token, draft_state_bytes, prefill_scratch,
    row_state_bytes, state_bytes,
};
use crate::models::qwen3_5::dflash2::{DFlash2Config, dflash2_file_bytes, read_dflash2_config};
use crate::models::qwen3_5::dflash2_state::store::lineage_bytes;

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
    }
}

/// What an attached DFlash2 draft adds to a plan. Its weights are resident
/// before a load reads the process, so the plan finds them in the active
/// bytes and `weights` is only reported (an estimate adds them itself).
/// The lane holds one context (the draft follows one sequence), the layer
/// outputs a tapped pass keeps until it ends (a drafted turn's passes are
/// one prefill window at most: it starts only on filled banks), and the
/// evaluated taps kept between cycles (the ring's rows and the last pass).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct DraftPrice {
    pub weights: u64,
    pub ring: u64,
    pub pass_taps: u64,
    pub kept_taps: u64,
}

impl DraftPrice {
    pub(super) fn of(c: &Config, draft: &DFlash2Config, weights: u64) -> Self {
        let taps = draft.target_layers.len() as u64;
        let pass = super::MAX_PREFILL_CHUNK as u64;
        let ring_rows = draft.sliding_window.saturating_sub(1) as u64;
        Self {
            weights,
            ring: draft.kv_ring_bytes(),
            pass_taps: taps * pass * draft_hidden_row_bytes(c),
            kept_taps: taps * (ring_rows + pass) * c.hidden_size as u64 * 2,
        }
    }

    /// The draft's bytes in the one lane, whatever the window.
    pub(super) fn lane(&self) -> u64 {
        self.ring
            .saturating_add(self.pass_taps)
            .saturating_add(self.kept_taps)
    }

    /// The price of the draft in `dir` for a target of config `c`, before
    /// it loads: its checkpoint file stands for its weights.
    pub(super) fn of_checkpoint(label: &str, c: &Config, dir: &Path) -> Result<Self> {
        let refused = |reason: String| {
            Error::from_reason(format!(
                "{label}: the DFlash2 draft at {} does not serve this model: {reason}",
                dir.display()
            ))
        };
        let draft = read_dflash2_config(dir).map_err(|error| refused(error.reason))?;
        draft
            .validate_target(super::decoder::draft_target_shape_of(c))
            .map_err(|error| refused(error.reason))?;
        let weights = dflash2_file_bytes(dir).map_err(|error| refused(error.reason))?;
        Ok(Self::of(c, &draft, weights))
    }
}

/// A load serves one speculation source: the MTP head of its checkpoint or
/// a DFlash2 draft.
pub(super) fn refuse_head_with_draft(
    label: &str,
    has_mtp: bool,
    draft: Option<&str>,
) -> Result<()> {
    match draft {
        Some(dir) if has_mtp => Err(Error::from_reason(format!(
            "{label}: the DFlash2 draft at {dir} cannot load beside the MTP head of this checkpoint: \
             drop draftModelPath to use the head, or load the model without the head to use the draft"
        ))),
        _ => Ok(()),
    }
}

/// The lane's state besides the cell: with MTP weights, the draft history
/// every prefill keeps for a `W`-token turn (draft K/V and indexer keys of
/// every row, a full prefill window of hidden rows waiting) and an MTP
/// turn's verify snapshots, which grows with `W`; with a DFlash2 draft,
/// its lane. The lane is one and charged once (Splash
/// `evaluateEngineMemoryPlan` `availableForOneRequestKv`).
pub(super) fn lane_state(c: &Config, has_mtp: bool, draft: DraftPrice) -> impl Fn(u32) -> u64 + '_ {
    move |window| {
        let mtp = if has_mtp {
            draft_state_bytes(c, u64::from(window))
                + u64::from(window) * draft_index_bytes_per_token(c)
                + super::MAX_PREFILL_CHUNK as u64 * draft_hidden_row_bytes(c)
        } else {
            0
        };
        mtp.saturating_add(draft.lane())
    }
}

/// The disk tier's host staging on a load with it on
/// (`MemoryFootprint::state_staging_bytes`): fixed, whatever the window.
pub(super) fn state_staging(c: &Config, has_mtp: bool, disk: bool) -> u64 {
    if disk {
        super::memory::disk_staging_bytes(c, has_mtp)
    } else {
        0
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

/// The store plan and the dynamic budget whose lanes hold the allowance of
/// `allowance_tokens` and, beside the pool, `states` bytes of stored prefix
/// states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StoreDecision {
    pub store: StorePlan,
    pub allowance: u64,
    pub allowance_tokens: u32,
    pub states: u64,
}

/// The stored states a store plan leaves room for: the one lane's
/// resume state (in use until its turn ends), the replay point the turn
/// publishes, and the one state the disk tier writes at a time. A publication
/// can give up neither of the first and the third; any other state is its
/// victim.
pub(super) const STORED_STATES: u64 = 3;

/// [`STORED_STATES`] states at `tokens` rows, priced as a publication checks
/// one (`Inner::room_for_state`): the cell, its draft part with the MTP head
/// at the larger of its sizes over the prefill windows (linear in the rows
/// the window drains, so at one of its ends), and its lineage.
pub(super) fn states_reserve(c: &Config, has_mtp: bool, tokens: u32) -> u64 {
    let rows = tokens as usize;
    let state = |chunk: usize| state_bytes(c, rows as u64, chunk as u64, has_mtp);
    let one = state(super::MAX_PREFILL_CHUNK)
        .max(state(1))
        .saturating_add(lineage_bytes(rows, rows / super::prefix_state::BLOCK_ROWS));
    STORED_STATES.saturating_mul(one)
}

/// Splash `planStreamedMemory` for a target that cannot spill: the KV
/// allowance of the requested context, else `min(trained,
/// kResidentAllowanceTokens)`, on the load's one lane (with MTP weights it
/// holds the MTP draft state) comes before the experts. The allowance holds
/// [`states_reserve`] of stored prefix states beside the pool as well: a
/// streamed store takes everything the allowance leaves, and a whole store's
/// pool grows into all its slack, so without the reserve neither leaves a
/// state room. A whole store whose slack is under the reserve streams.
pub(super) fn plan_store(
    label: &str,
    c: &Config,
    inventory: &LoadInventory,
    live: LiveMemory,
    max_memory: Option<u64>,
    max_context: Option<u32>,
    has_mtp: bool,
    disk: bool,
    draft: DraftPrice,
) -> Result<StoreDecision> {
    let plan = |states: u64| {
        plan_lanes(
            label,
            c,
            inventory,
            live,
            max_memory,
            max_context,
            has_mtp,
            disk,
            draft,
            states,
        )
    };
    let base = plan(0)?;
    let above = base
        .store
        .expert_cap_bytes
        .saturating_sub(inventory.experts.minimum);
    let states = states_reserve(c, has_mtp, base.allowance_tokens).min(above);
    if states == 0 {
        return Ok(base);
    }
    plan(states)
}

/// [`plan_store`] with `states` bytes beside the pool in the allowance.
fn plan_lanes(
    label: &str,
    c: &Config,
    inventory: &LoadInventory,
    live: LiveMemory,
    max_memory: Option<u64>,
    max_context: Option<u32>,
    has_mtp: bool,
    disk: bool,
    draft: DraftPrice,
    states: u64,
) -> Result<StoreDecision> {
    let transient = |window: u32| transient(c, window);
    let state = lane_state(c, has_mtp, draft);
    let pool = pool_input(c, &transient, &state);
    let lane = |dynamic: u64| pool.plan_at(dynamic).lane.window.maximum();
    let automatic = pool.shape.trained.min(RESIDENT_ALLOWANCE_TOKENS);
    let allowance_tokens = max_context.unwrap_or(automatic);
    let allowance = allowance_dynamic_bytes(&[&lane], allowance_tokens).saturating_add(states);
    let minimum_dynamic = pool.plan_at(0).lane.minimum_bytes;
    let hard = hard_budget_bytes(
        live.working_set_bytes,
        max_memory,
        crate::engine::memory_plan::Margin::Unguarded,
    );
    let fixed = live
        .active_bytes
        .saturating_add(live.registered_pool_bytes)
        .saturating_add(inventory.store_fixed_bytes)
        .saturating_add(inventory.vision_bytes)
        .saturating_add(state_staging(c, has_mtp, disk));
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
    Ok(StoreDecision {
        store,
        allowance,
        allowance_tokens,
        states,
    })
}

/// Splash `evaluateEngineMemoryPlan` after the load: what is resident now,
/// with `held_back` bytes of the stores and the vision tower above it, and
/// the store decision's `states` held beside the pool, which the pool never
/// grows into.
pub(super) fn plan_engine(
    label: &str,
    c: &Config,
    live: LiveMemory,
    held_back: u64,
    max_memory: Option<u64>,
    has_mtp: bool,
    disk: bool,
    draft: DraftPrice,
    states: u64,
) -> Result<MemoryPlan> {
    let transient = |window: u32| transient(c, window);
    let lane = lane_state(c, has_mtp, draft);
    let state = |window: u32| lane(window).saturating_add(states);
    let footprint = MemoryFootprint {
        working_set_bytes: live.working_set_bytes,
        max_memory_bytes: max_memory,
        active_bytes: live.active_bytes,
        other_pool_bytes: live.registered_pool_bytes,
        staging_bytes: 0,
        state_staging_bytes: state_staging(c, has_mtp, disk),
        draft_weights_bytes: draft.weights,
        held_back_bytes: held_back,
        margin: crate::engine::memory_plan::Margin::Unguarded,
    };
    MemoryPlan::evaluate(footprint, Some(pool_input(c, &transient, &state)))
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
    disk: bool,
    draft: DraftPrice,
) -> Result<Qwen4ExpContextEstimate> {
    let live = LiveMemory {
        active_bytes: live.active_bytes.saturating_add(draft.weights),
        ..live
    };
    let decision = plan_store(
        label,
        c,
        inventory,
        live,
        max_memory,
        max_context,
        has_mtp,
        disk,
        draft,
    )?;
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
    let plan = plan_engine(
        label,
        c,
        loaded,
        held_back,
        max_memory,
        has_mtp,
        disk,
        draft,
        decision.states,
    )?;
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
        refuse_head_with_draft(LABEL, has_mtp, options.draft_model_path.as_deref())?;
        let vision = super::media::Vision::metadata(&raw, &weights, &config)?.available();
        let inventory = weights.load_inventory(&config, vision)?;
        let env = std::env::var(crate::cold_tier::PERSIST_PAGED_CACHE_ENV).ok();
        let disk = super::disk_plan(LABEL, &options, env.as_deref())?.1;
        let draft = match options.draft_model_path.as_deref() {
            Some(dir) => DraftPrice::of_checkpoint(LABEL, &config, Path::new(dir))?,
            None => DraftPrice::default(),
        };
        estimate(
            LABEL,
            &config,
            &inventory,
            live,
            max_memory,
            max_context,
            has_mtp,
            disk,
            draft,
        )
        .map(Some)
    })
    .await
    .map_err(|error| Error::from_reason(format!("{LABEL} estimate: {error}")))?
}

#[cfg(test)]
pub(super) mod tests;
