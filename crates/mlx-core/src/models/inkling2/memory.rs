//! Unified [`MemoryPlan`] + [`Governor`] ledger + [`WiredResidency`] sizing for Inkling-2-Mini.
//!
//! Inkling-2-Mini executes 24 physical layers across 48 logical visits split into two KV groups:
//! - Group 0: **8 Global visits** (`5, 11, 17, 23, 29, 35, 41, 47`), elastic extent-grown pool
//!   registered with the unified [`Governor`] ledger.
//! - Group 1: **40 SWA visits** (`sliding_window = 1024`), bounded extent-grown pool sized for
//!   `lanes` concurrent sliding windows + one prefill chunk + one reserved null sentinel block.
//!
//! Defaults to **INT8 32-token KTM** on Metal with `bf16` fallback (`--kv-format bf16`), and
//! sizes up to 4 concurrent lanes below 256 GB physical RAM or 8 lanes on >=256 GB machines.

use napi::bindgen_prelude::{Error, Result};
use napi_derive::napi;

use crate::array::DType;
use crate::disk_tier::CacheDisk;
use crate::engine::memory_plan::{
    KvLayout, LaneShape, LiveMemory, Margin, MemoryFootprint, MemoryPlan, PlanError, PoolInput,
    SPECULATIVE_SCRATCH_TOKENS, check_max_context, decode_share_option,
    lane_ceiling_for_physical_memory, max_memory_option,
    resolve_kv_format as resolve_kv_format_shared, select_max_lanes_preserving_window,
};
use crate::transformer::{Activations, KvFormat};

use super::config::Inkling2Config;
use super::kv_cache::{
    INKLING2_PAGED_BLOCK_SIZE, INKLING2_PAGED_PREFILL_MAX_CHUNK, default_inkling2_kv_format,
    kv_format_metal_layout,
};

/// Concurrent decode lane cap at or above 256 GB physical RAM.
const LARGE_MACHINE_LANES: usize = 8;

/// Target bytes per extent growth step for Inkling2's extent-grown Global KV pool (128 MiB).
pub(crate) const INKLING2_EXTENT_TARGET_BYTES: u64 = 128 << 20;

/// Trained and physically available active-context limits for a loaded Inkling-2-Mini model.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Inkling2ContextLimits {
    pub trained_window_tokens: u32,
    pub effective_window_tokens: u32,
    pub paged_block_capacity: u32,
    pub paged_block_size: u32,
    pub rest_of_window_output_budget: bool,
    pub omitted_budget_is_rest_of_window: bool,
    #[napi(ts_type = "'int8' | 'bf16'")]
    pub kv_format: String,
    #[napi(ts_type = "'int8' | 'bf16'")]
    pub attention_queries: String,
    pub governor_model_id: Option<f64>,
}

impl Inkling2ContextLimits {
    pub(crate) fn from_parts(
        trained_window_tokens: u32,
        effective_window_tokens: u32,
        paged_block_capacity: u32,
        paged_block_size: u32,
        memory_window: bool,
        kv_format: KvFormat,
        int8_queries: bool,
        governor_model_id: Option<u64>,
    ) -> Self {
        Self {
            trained_window_tokens,
            effective_window_tokens,
            paged_block_capacity,
            paged_block_size,
            rest_of_window_output_budget: true,
            omitted_budget_is_rest_of_window: memory_window,
            kv_format: kv_format.as_str().to_string(),
            attention_queries: if int8_queries { "int8" } else { "bf16" }.to_string(),
            governor_model_id: governor_model_id.map(|id| id as f64),
        }
    }
}

/// Resolved Metal/KV/scheduler plan derived from [`MemoryPlan`] after weight materialization.
#[derive(Debug, Clone)]
pub(crate) struct Inkling2PoolPlan {
    pub memory_plan: MemoryPlan,
    pub kv_format: KvFormat,
    pub block_size: u32,
    pub max_chunk: u32,
    pub lanes: u32,
    /// Global (Group 0, 8 visits) initial runway blocks and max growth ceiling.
    pub global_runway_blocks: u32,
    pub global_max_blocks: u32,
    pub global_extent_blocks: u32,
    pub global_bytes_per_block: u64,
    /// SWA (Group 1, 40 visits) initial runway blocks and max growth ceiling.
    pub swa_runway_blocks: u32,
    pub swa_max_blocks: u32,
    pub swa_extent_blocks: u32,
    pub swa_bytes_per_block: u64,
    /// Effective maximum context tokens per turn.
    pub effective_window_tokens: u32,
    /// Host state staging bytes reserved for StateDisk (0 when SSD cache is off).
    #[cfg_attr(not(test), allow(dead_code))]
    pub state_staging_bytes: u64,
}

/// Number of Global (full-attention) logical visits (`8` on Inkling-2-Mini).
pub(crate) fn global_visit_count(config: &Inkling2Config) -> usize {
    config.visits().into_iter().filter(|v| !v.is_local).count()
}

/// Number of SWA (sliding-window) logical visits (`40` on Inkling-2-Mini).
pub(crate) fn swa_visit_count(config: &Inkling2Config) -> usize {
    config.visits().into_iter().filter(|v| v.is_local).count()
}

/// Exact RAM bytes of one sequence's 48-visit ShortConv(`k=4`) rolling state in BF16.
///
/// Per visit: `k` and `v` are `[k-1, kv_dim]`, `attn_post` and `mlp_post` are `[k-1, hidden_size]`.
pub(crate) fn short_conv_state_bytes(config: &Inkling2Config) -> u64 {
    let visits = config.num_layer_visits() as u64;
    let hist = config.sconv_kernel_size.saturating_sub(1) as u64;
    let kv_dim = (config.num_key_value_heads as u64).saturating_mul(config.head_dim as u64);
    let hidden = config.hidden_size as u64;
    let per_visit_elems = hist.saturating_mul(
        kv_dim
            .saturating_mul(2)
            .saturating_add(hidden.saturating_mul(2)),
    );
    // BF16 = 2 bytes per element.
    visits.saturating_mul(per_visit_elems).saturating_mul(2)
}

/// Maximum number of live physical SWA blocks retained at any block-aligned boundary.
///
/// For `sliding_window = 1024, block_size = 32`, this is exactly `32` blocks.
pub(crate) fn swa_window_blocks(config: &Inkling2Config, block_size: u32) -> u32 {
    let bs = block_size.max(1);
    let window = u32::try_from(config.sliding_window).unwrap_or(u32::MAX);
    if window == 0 {
        return 0;
    }
    if window.is_multiple_of(bs) {
        window / bs
    } else {
        window.div_ceil(bs).saturating_add(1)
    }
}

/// Exact bytes of one physical KV block across all Global visits (`8`) or SWA visits (`40`).
pub(crate) fn group_bytes_per_block(
    config: &Inkling2Config,
    num_layers: usize,
    kv_format: KvFormat,
    block_size: u32,
) -> Result<u64> {
    if num_layers == 0 {
        return Ok(0);
    }
    let (metal_dtype, _, _) = kv_format_metal_layout(kv_format);
    let layers = u32::try_from(num_layers)
        .map_err(|_| Error::from_reason("Inkling2 KV group layer count overflows u32"))?;
    let kv_heads = u32::try_from(config.num_key_value_heads)
        .map_err(|_| Error::from_reason("Inkling2 num_key_value_heads overflows u32"))?;
    let head_dim = u32::try_from(config.head_dim)
        .map_err(|_| Error::from_reason("Inkling2 head_dim overflows u32"))?;
    mlx_paged_attn::profile::bytes_per_block(layers, kv_heads, head_dim, block_size, metal_dtype)
        .map_err(|e| Error::from_reason(format!("Inkling2 bytes_per_block failed: {e}")))
}

/// Exact byte size of one full SWA trailing window (`max_swa_blocks` blocks across 40 SWA visits).
///
/// On Inkling-2-Mini INT8 KTM (`block_size = 32`), this is `32 * 40 * 67,584 = 86,507,520` B (~82.5 MiB).
pub(crate) fn swa_window_blob_bytes(
    config: &Inkling2Config,
    kv_format: KvFormat,
    block_size: u32,
) -> Result<u64> {
    let swa_layers = swa_visit_count(config);
    let swa_bpb = group_bytes_per_block(config, swa_layers, kv_format, block_size)?;
    let blocks = u64::from(swa_window_blocks(config, block_size));
    Ok(blocks.saturating_mul(swa_bpb))
}

/// Exact byte size of one content-keyed Inkling2 state blob:
/// `[40 SWA(1024) trailing KV window] + [48 ShortConv(k=4) ring states]`.
///
/// Constant regardless of context length (~84.3 MiB INT8 KTM on Inkling-2-Mini).
pub(crate) fn state_blob_bytes(
    config: &Inkling2Config,
    kv_format: KvFormat,
    block_size: u32,
) -> Result<u64> {
    let swa = swa_window_blob_bytes(config, kv_format, block_size)?;
    let conv = short_conv_state_bytes(config);
    Ok(swa.saturating_add(conv))
}

/// Compute the initial runway blocks and max ceiling blocks for Group 1 (40 SWA visits).
///
/// Group 1 holds at most one prefill chunk (`max_chunk`) plus `lanes` live sliding windows,
/// a small speculative/verify cushion per lane, plus 1 reserved null sentinel block.
pub(crate) fn swa_pool_block_bounds(
    config: &Inkling2Config,
    block_size: u32,
    max_chunk: u32,
    lanes: u32,
) -> (u32, u32) {
    let swa_layers = swa_visit_count(config);
    if swa_layers == 0 {
        return (0, 0);
    }
    let bs = block_size.max(1);
    let win_blocks = swa_window_blocks(config, bs).max(1);
    let chunk_blocks = max_chunk.max(bs).div_ceil(bs);
    let lanes = lanes.max(1);
    // Runway fits one full active sequence prefilling a chunk + 1 null sentinel + cushion.
    let runway = 1u32
        .saturating_add(win_blocks)
        .saturating_add(chunk_blocks)
        .saturating_add(4);
    // Max fits `lanes` concurrent sequences + 1 active prefill chunk + null sentinel + lookahead.
    let max_blocks = 1u32
        .saturating_add(win_blocks.saturating_mul(lanes))
        .saturating_add(chunk_blocks)
        .saturating_add(lanes.saturating_mul(4))
        .max(runway);
    (runway, max_blocks)
}

/// Lane ceiling for a machine with `physical_bytes` of RAM:
/// up to 4 lanes below 256 GB, up to 8 lanes at or above 256 GB.
pub(crate) fn inkling2_lane_ceiling(physical_bytes: Option<u64>) -> usize {
    let max_units =
        crate::engine::recurrent_state::HYBRID_LIVE_STATE_UNITS.max(LARGE_MACHINE_LANES);
    lane_ceiling_for_physical_memory(physical_bytes, LARGE_MACHINE_LANES.min(max_units))
}

/// Effective scheduler lane ceiling after honoring `MLX_SCHED_MAX_NUM_SEQS` and `force_serial()`.
pub(crate) fn inkling2_scheduler_lanes() -> u32 {
    if super::scheduler::Inkling2SchedulerState::force_serial() {
        1
    } else {
        crate::engine::hybrid_scheduler::scheduler_max_num_seqs_for(inkling2_lane_ceiling(
            crate::utils::host::physical_memory(),
        )) as u32
    }
}

/// Extent growth quantum (in blocks) for a KV pool with `bytes_per_block`.
pub(crate) fn extent_growth_blocks(bytes_per_block: u64) -> u32 {
    if bytes_per_block == 0 {
        return 64;
    }
    INKLING2_EXTENT_TARGET_BYTES
        .div_ceil(bytes_per_block)
        .clamp(16, u64::from(u32::MAX)) as u32
}

/// Conservative transient activation scratch (in bytes) for one prefill chunk of up to `max_chunk`
/// rows at context length `window`.
pub(crate) fn prefill_transient_bytes(
    config: &Inkling2Config,
    max_chunk: u32,
    _window: u32,
) -> u64 {
    let rows = u64::from(max_chunk.max(1));
    let d = config.hidden_size as u64;
    let q_dim = (config.num_attention_heads as u64).saturating_mul(config.head_dim as u64);
    let moe_dim = config.effective_moe_intermediate_size() as u64;
    let top_k = config.num_experts_per_tok.max(1) as u64;
    // AttnRes block stack (~9 BF16 blocks) + Q/K/V/R/gate + MoE top-k scratch + MPP FP32 workspace.
    let attn_res = 10u64
        .saturating_mul(rows)
        .saturating_mul(d)
        .saturating_mul(2);
    let qkv_scratch = rows.saturating_mul(q_dim).saturating_mul(8);
    let moe_scratch = rows
        .saturating_mul(top_k)
        .saturating_mul(moe_dim)
        .saturating_mul(4);
    let mpp_ws = rows
        .saturating_mul(config.num_attention_heads as u64)
        .saturating_mul(config.head_dim as u64)
        .saturating_mul(8);
    attn_res
        .saturating_add(qkv_scratch)
        .saturating_add(moe_scratch)
        .saturating_add(mpp_ws)
        .max(64 << 20)
}

/// Conservative transient scratch for a batched `[lanes, 1]` decode step.
pub(crate) fn decode_transient_bytes(config: &Inkling2Config, lanes: u32) -> u64 {
    let rows = u64::from(lanes.max(1));
    let d = config.hidden_size as u64;
    let moe_dim = config.effective_moe_intermediate_size() as u64;
    let top_k = config.num_experts_per_tok.max(1) as u64;
    rows.saturating_mul(
        d.saturating_mul(32)
            .saturating_add(top_k.saturating_mul(moe_dim).saturating_mul(6)),
    )
    .max(16 << 20)
}

/// Parse and validate `kvFormat` load option (defaults to INT8 on Metal, BF16 elsewhere).
pub(crate) fn resolve_kv_format(label: &str, requested: Option<&str>) -> Result<KvFormat> {
    resolve_kv_format_shared(
        &format!("{label} load"),
        requested,
        default_inkling2_kv_format(),
        crate::engine::persistence::compiled_forward_backend_available(),
        || "int8 K/V requires the Metal paged backend; load with kvFormat 'bf16'".to_string(),
    )
}

/// Parse and validate `activations` load option (`'auto'` default on Metal, or `'bf16'`).
pub(crate) fn resolve_activations(label: &str, requested: Option<&str>) -> Result<Activations> {
    Activations::parse(requested, Activations::Auto)
        .map_err(|e| Error::from_reason(format!("{label} load: {e}")))
}

/// Parse and validate `maxMemoryBytes`.
pub(crate) fn resolve_max_memory(label: &str, raw: Option<i64>) -> Result<Option<u64>> {
    max_memory_option(label, raw)
}

/// Parse and validate `maxContextTokens`.
pub(crate) fn resolve_max_context(
    label: &str,
    raw: Option<i64>,
    trained: u32,
) -> Result<Option<u32>> {
    match raw {
        None => Ok(None),
        Some(v) if v <= 0 => Err(Error::from_reason(format!(
            "{label} load: maxContextTokens must be >= 1, got {v}"
        ))),
        Some(v) => {
            let requested = u32::try_from(v).map_err(|_| {
                Error::from_reason(format!("{label} load: maxContextTokens {v} overflows u32"))
            })?;
            if requested > trained {
                return Err(Error::from_reason(format!(
                    "{label} load: maxContextTokens {requested} exceeds trained context window {trained}"
                )));
            }
            Ok(Some(requested))
        }
    }
}

/// Parse and validate `decodeShare` (non-negative finite float, default `0.5`).
pub(crate) fn resolve_decode_share(label: &str, raw: Option<f64>) -> Result<f64> {
    decode_share_option(label, raw)
}

/// Evaluate Inkling-2-Mini's unified [`MemoryPlan`] after weights are materialized.
///
/// Sizes:
/// - Group 0 (8 Global visits): elastic extent-grown pool up to the dynamic budget remainder.
/// - Group 1 (40 SWA visits): bounded extent-grown pool sized for `lanes` concurrent windows.
/// - Concurrent lanes: tries from `max_lanes` down to `1`, keeping the one-lane context window
///   (lanes give way before the context window shrinks, matching Qwen3.5).
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_inkling2_plan(
    label: &str,
    config: &Inkling2Config,
    kv_format: KvFormat,
    live: LiveMemory,
    max_memory_bytes: Option<u64>,
    max_context_tokens: Option<u32>,
    max_lanes: u32,
    cache_disk: CacheDisk,
) -> Result<Inkling2PoolPlan> {
    let block_size = INKLING2_PAGED_BLOCK_SIZE;
    let max_chunk = INKLING2_PAGED_PREFILL_MAX_CHUNK;
    let global_layers = global_visit_count(config);
    let swa_layers = swa_visit_count(config);
    if global_layers == 0 {
        return Err(Error::from_reason(format!(
            "{label} memory plan: config has zero global attention visits"
        )));
    }
    let global_bpb = group_bytes_per_block(config, global_layers, kv_format, block_size)?;
    let swa_bpb = group_bytes_per_block(config, swa_layers, kv_format, block_size)?;
    let conv_bytes = short_conv_state_bytes(config);
    let state_staging_bytes = if cache_disk != CacheDisk::Off {
        state_blob_bytes(config, kv_format, block_size)?
    } else {
        0
    };
    let trained = u32::try_from(config.max_position_embeddings)
        .unwrap_or(u32::MAX)
        .max(block_size);
    let effective_trained = max_context_tokens.map_or(trained, |req| req.min(trained));

    let eval_for_lanes = |lanes: u32| -> std::result::Result<(MemoryPlan, u32, u32), PlanError> {
        let (swa_runway, swa_max) = swa_pool_block_bounds(config, block_size, max_chunk, lanes);
        let swa_max_bytes = u64::from(swa_max).saturating_mul(swa_bpb);
        let lanes_conv_bytes = u64::from(lanes).saturating_mul(conv_bytes);

        let footprint = MemoryFootprint {
            working_set_bytes: live.working_set_bytes,
            max_memory_bytes,
            active_bytes: live.active_bytes,
            other_pool_bytes: live.registered_pool_bytes,
            staging_bytes: mlx_paged_attn::RESTORE_STAGING_BYTES,
            state_staging_bytes,
            draft_weights_bytes: 0,
            held_back_bytes: 0,
            margin: Margin::Ledger,
        };
        let layout = KvLayout::of(
            global_layers,
            config.num_key_value_heads as i32,
            config.head_dim as i32,
            DType::BFloat16,
            kv_format,
        );
        let runway_blocks = layout.runway_pages().max(64).min(u64::from(u32::MAX)) as u32;
        let transient_fn = |window: u32| {
            prefill_transient_bytes(config, max_chunk, window)
                .max(decode_transient_bytes(config, lanes))
        };
        // SWA pool ceiling + per-lane ShortConv live outside the Global pool.
        let non_global_state = swa_max_bytes.saturating_add(lanes_conv_bytes);
        let state_fn = |_window: u32| non_global_state;
        let pool_input = PoolInput {
            shape: LaneShape {
                trained: effective_trained,
                lane_bytes: conv_bytes,
                layout,
                scratch_rows: SPECULATIVE_SCRATCH_TOKENS,
                transient: &transient_fn,
            },
            bytes_per_block: global_bpb,
            layers: global_layers as u64,
            runway_blocks,
            explicit_mib: None,
            lane_in_pool: false,
            state: &state_fn,
        };
        let mut plan = MemoryPlan::evaluate(footprint, Some(pool_input))?;
        if let Some(ref mut pool) = plan.pool {
            pool.lanes = lanes;
        }
        Ok((plan, swa_runway, swa_max))
    };

    let ((memory_plan, swa_runway_blocks, swa_max_blocks), lanes) =
        select_max_lanes_preserving_window(max_lanes, eval_for_lanes, |(plan, _, _)| {
            plan.pool.as_ref().map(|p| p.lane.window.maximum())
        })
        .map_err(|e| Error::from_reason(e.message(label)))?;
    let pool = memory_plan.pool.as_ref().ok_or_else(|| {
        Error::from_reason(format!("{label} memory plan: produced no KV pool plan"))
    })?;
    check_max_context(pool.lane.window, memory_plan.bound, max_context_tokens)
        .map_err(|e| Error::from_reason(e.message(label)))?;

    let effective_window_tokens = max_context_tokens
        .map_or(pool.lane.window.maximum(), |req| {
            req.min(pool.lane.window.maximum())
        })
        .max(block_size);

    Ok(Inkling2PoolPlan {
        global_runway_blocks: pool.runway_blocks.max(1),
        global_max_blocks: pool.max_blocks.max(pool.runway_blocks).max(1),
        global_extent_blocks: extent_growth_blocks(global_bpb),
        global_bytes_per_block: global_bpb,
        swa_runway_blocks,
        swa_max_blocks,
        swa_extent_blocks: extent_growth_blocks(swa_bpb).min(swa_max_blocks.max(1)),
        swa_bytes_per_block: swa_bpb,
        effective_window_tokens,
        state_staging_bytes,
        memory_plan,
        kv_format,
        block_size,
        max_chunk,
        lanes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inkling2_memory_plan_sizes_global_swa_and_lanes() {
        let cfg = Inkling2Config::from_json_str(
            r#"{
                "model_type": "inkling2",
                "hidden_size": 2304,
                "num_hidden_layers": 24,
                "num_attention_heads": 24,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 1048576,
                "sliding_window": 1024,
                "sliding_window_pattern": [
                    true,true,true,true,true,false,
                    true,true,true,true,true,false,
                    true,true,true,true,true,false,
                    true,true,true,true,true,false
                ],
                "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3}
            }"#,
            true,
        )
        .expect("mini config");
        assert_eq!(global_visit_count(&cfg), 8);
        assert_eq!(swa_visit_count(&cfg), 40);
        assert_eq!(swa_window_blocks(&cfg, 32), 32);

        // State blob in INT8 KTM should be ~84.3 MiB (82.5 MiB SWA + 1.83 MiB ShortConv).
        let blob = state_blob_bytes(&cfg, KvFormat::Int8, 32).expect("state blob bytes");
        let conv = short_conv_state_bytes(&cfg);
        assert!(conv > 1_800_000 && conv < 2_100_000, "conv={conv}");
        assert!(blob > 80_000_000 && blob < 92_000_000, "blob={blob}");

        // Lane ceiling: 4 on <256 GB, 8 on >=256 GB.
        assert_eq!(inkling2_lane_ceiling(Some(128u64 << 30)), 4);
        assert_eq!(inkling2_lane_ceiling(Some(192u64 << 30)), 4);
        assert_eq!(inkling2_lane_ceiling(Some(256u64 << 30)), 8);

        // Evaluate plan on a 96 GiB working set with ~18 GiB weights.
        let live = LiveMemory {
            working_set_bytes: 96u64 << 30,
            active_bytes: 18u64 << 30,
            registered_pool_bytes: 0,
        };
        let plan = evaluate_inkling2_plan(
            "Inkling2",
            &cfg,
            KvFormat::Int8,
            live,
            None,
            None,
            4,
            CacheDisk::Auto,
        )
        .expect("plan on 96 GiB");
        assert_eq!(plan.lanes, 4);
        assert!(plan.global_max_blocks > plan.global_runway_blocks);
        assert!(plan.swa_max_blocks >= plan.swa_runway_blocks);
        assert!(plan.effective_window_tokens >= 131_072);
        assert_eq!(plan.state_staging_bytes, blob);

        // BF16 fallback halves global block capacity vs INT8.
        let bf16_plan = evaluate_inkling2_plan(
            "Inkling2",
            &cfg,
            KvFormat::Bf16,
            live,
            None,
            None,
            4,
            CacheDisk::Off,
        )
        .expect("bf16 plan");
        assert!(bf16_plan.global_max_blocks < plan.global_max_blocks);
        assert_eq!(bf16_plan.state_staging_bytes, 0);
    }
}
