use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::array::mask::create_causal_mask;
use crate::array::{DType, MxArray};
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::nn::{Linear, RMSNorm, RoPE};
use crate::transformer::paged_kv_cache_adapter::{
    PagedAttentionV2Layout, PagedKVCacheAdapter, PagedPrefillMemorySnapshot,
    paged_attention_v2_aux_fits, paged_attention_v2_partition_upper_bound,
};
use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::config::Gemma4Config;
use super::layer_cache::Gemma4LayerCache;
use super::quantized_linear::{LinearProj, QuantizedLinear};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CacheHitPrefillMode {
    /// Keep the physical paged pool authoritative and choose the compute
    /// operator from live memory headroom.
    Auto,
    /// Gather paged K/V inside the MLX graph and run matrix SDPA.
    ForceSdpa,
    /// Run compact varlen PagedAttention directly over the physical pool.
    ForceVarlen,
    /// Preserve the former duplicated-block-row PagedAttention bridge.
    ForceLegacy,
    /// Materialize K/V through the synchronous host-read fallback.
    ForceHostRead,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CacheHitPrefillPath {
    PagedPoolSdpa,
    PagedVarlen,
    PagedLegacy,
    HostRead,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CacheHitPrefillPlan {
    path: CacheHitPrefillPath,
    estimated_sdpa_bytes: u64,
    estimated_varlen_bytes: u64,
    live_headroom_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LivePrefillHeadroom {
    selected_bytes: Option<u64>,
    allocator_available_bytes: Option<u64>,
    metal_available_bytes: Option<u64>,
}

const PREFILL_FIXED_OVERHEAD_BYTES: u64 = 64 * 1024 * 1024;
const PREFILL_HEADROOM_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Maximum compact K/V gathered for one full-attention layer by the automatic
/// single-token decode route. Gemma4-12B has eight D=512/Hkv=1 global layers,
/// so this caps the total lazy graph payload at roughly 512 MiB while keeping
/// the physical paged pool authoritative. The estimate includes both selected
/// block views and their contiguous copies. Larger contexts stay on the
/// memory-bounded PagedAttention operator.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PagedDecodeMode {
    Auto,
    ForceSdpa,
    ForcePagedAttention,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PagedDecodePath {
    PagedPoolSdpa,
    PagedAttention,
}

fn parse_paged_decode_mode(route: Option<&str>) -> PagedDecodeMode {
    match route.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        None | Some("") | Some("auto") => PagedDecodeMode::Auto,
        Some("sdpa") | Some("paged_pool_sdpa") => PagedDecodeMode::ForceSdpa,
        Some("paged") | Some("paged_attention") => PagedDecodeMode::ForcePagedAttention,
        Some(_) => PagedDecodeMode::Auto,
    }
}

fn resolve_paged_decode_mode(route: Option<&str>, grouped_gemma4: Option<&str>) -> PagedDecodeMode {
    let explicit = parse_paged_decode_mode(route);
    let grouped_requested = parse_grouped_gemma4_selector(grouped_gemma4) != "off";
    if grouped_requested && matches!(explicit, PagedDecodeMode::Auto) {
        PagedDecodeMode::ForcePagedAttention
    } else {
        explicit
    }
}

/// Parse the selector exactly as both paged-attention dispatchers do. Keeping
/// this deliberately case-sensitive and whitespace-sensitive prevents Rust
/// route diagnostics from claiming that the grouped kernel is enabled while
/// the C++/Rust Metal dispatcher treats the same process environment as off.
fn parse_grouped_gemma4_selector(value: Option<&str>) -> &'static str {
    match value {
        Some("1" | "on" | "auto" | "true") => "auto",
        Some("force") => "force",
        _ => "off",
    }
}

fn grouped_gemma4_diagnostic_config() -> (&'static str, Option<u32>) {
    static CONFIG: OnceLock<(String, Option<u32>)> = OnceLock::new();
    let config = CONFIG.get_or_init(|| {
        let selector_env = std::env::var("MLX_PAGED_GROUPED_GEMMA4").ok();
        let selector = parse_grouped_gemma4_selector(selector_env.as_deref());
        let stripes = std::env::var("MLX_PAGED_GROUPED_GEMMA4_STRIPES")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| matches!(value, 4 | 8 | 16 | 32 | 64 | 128 | 256));
        (selector.to_string(), stripes)
    });
    (config.0.as_str(), config.1)
}

fn grouped_gemma4_planned_stripes(
    selector: &str,
    override_stripes: Option<u32>,
    total_context: u32,
) -> Option<u32> {
    let eligible = total_context > 512
        && (selector == "force"
            || (selector == "auto" && (3_072..=16_384).contains(&total_context)));
    if !eligible {
        return None;
    }
    override_stripes.or(Some(match total_context {
        0..=4_096 => 32,
        4_097..=8_192 => 64,
        _ => 128,
    }))
}

#[allow(clippy::too_many_arguments)]
fn grouped_gemma4_kernel_candidate(
    selector: &str,
    override_stripes: Option<u32>,
    total_context: u32,
    query_dtype: DType,
    cache_dtype: Option<DType>,
    block_size: u32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
) -> (&'static str, Option<u32>) {
    let grouped_stripes = grouped_gemma4_planned_stripes(selector, override_stripes, total_context);
    let exact_shape = query_dtype == DType::BFloat16
        && cache_dtype == Some(DType::BFloat16)
        && block_size == 16
        && num_heads == 16
        && num_kv_heads == 1
        && head_dim == 512;
    if exact_shape && grouped_stripes.is_some() {
        ("grouped_gemma4_d512_staged", grouped_stripes)
    } else {
        ("generic_v2", None)
    }
}

fn paged_decode_mode() -> PagedDecodeMode {
    static MODE: OnceLock<PagedDecodeMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let route = std::env::var("MLX_GEMMA4_PAGED_DECODE_ROUTE").ok();
        let grouped = std::env::var("MLX_PAGED_GROUPED_GEMMA4").ok();
        resolve_paged_decode_mode(route.as_deref(), grouped.as_deref())
    })
}

fn estimate_decode_sdpa_gather_bytes(
    total_context: u64,
    num_kv_heads: u64,
    head_dim: u64,
    dtype_bytes: u64,
) -> u64 {
    total_context
        .saturating_mul(num_kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(dtype_bytes)
        // K + V, plus the contiguous copy of each selected block view.
        .saturating_mul(4)
}

fn select_paged_decode_path(
    mode: PagedDecodeMode,
    query_dtype: DType,
    cache_dtype: Option<DType>,
    total_context: u64,
    num_kv_heads: u64,
    head_dim: u64,
    graph_backend_available: bool,
) -> (PagedDecodePath, u64) {
    let effective_dtype = prefill_sdpa_effective_dtype(query_dtype, cache_dtype);
    let dtype_bytes = match effective_dtype {
        Some(DType::Float16 | DType::BFloat16) => 2,
        Some(DType::Float32) => 4,
        _ => 0,
    };
    let estimated_gather_bytes =
        estimate_decode_sdpa_gather_bytes(total_context, num_kv_heads, head_dim, dtype_bytes);
    let sdpa_available = graph_backend_available && dtype_bytes == 2;

    let path = match mode {
        PagedDecodeMode::ForceSdpa if sdpa_available => PagedDecodePath::PagedPoolSdpa,
        PagedDecodeMode::Auto
        | PagedDecodeMode::ForceSdpa
        | PagedDecodeMode::ForcePagedAttention => PagedDecodePath::PagedAttention,
    };
    (path, estimated_gather_bytes)
}

fn should_report_paged_decode_path(path: PagedDecodePath) -> bool {
    static SDPA_REPORTED: AtomicBool = AtomicBool::new(false);
    static PAGED_REPORTED: AtomicBool = AtomicBool::new(false);
    match path {
        PagedDecodePath::PagedPoolSdpa => !SDPA_REPORTED.swap(true, Ordering::Relaxed),
        PagedDecodePath::PagedAttention => !PAGED_REPORTED.swap(true, Ordering::Relaxed),
    }
}

fn should_report_paged_decode_fallback() -> bool {
    static REPORTED: AtomicBool = AtomicBool::new(false);
    !REPORTED.swap(true, Ordering::Relaxed)
}

fn parse_cache_hit_prefill_mode(
    route: Option<&str>,
    legacy_paged_attention: Option<&str>,
) -> CacheHitPrefillMode {
    if let Some(route) = route.map(str::trim) {
        return match route.to_ascii_lowercase().as_str() {
            "" | "auto" => CacheHitPrefillMode::Auto,
            "sdpa" | "paged_pool_sdpa" => CacheHitPrefillMode::ForceSdpa,
            "varlen" | "paged_varlen" => CacheHitPrefillMode::ForceVarlen,
            "legacy" | "paged_legacy" => CacheHitPrefillMode::ForceLegacy,
            "host" | "host_read" => CacheHitPrefillMode::ForceHostRead,
            _ => CacheHitPrefillMode::Auto,
        };
    }

    // Backward compatibility: this switch previously selected between the
    // duplicated-row paged bridge and host materialization. Disabling it must
    // continue to force the diagnostic host-read path; enabled/unset now opts
    // into the faster adaptive paged-storage policy.
    match legacy_paged_attention {
        Some(value) if !crate::inference_trace::env_flag_value_enabled(value) => {
            CacheHitPrefillMode::ForceHostRead
        }
        _ => CacheHitPrefillMode::Auto,
    }
}

fn cache_hit_prefill_mode() -> CacheHitPrefillMode {
    static MODE: OnceLock<CacheHitPrefillMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let route = std::env::var("MLX_GEMMA4_PAGED_PREFILL_ROUTE").ok();
        let legacy = std::env::var("MLX_GEMMA4_PAGED_PREFILL_PAGED_ATTENTION").ok();
        parse_cache_hit_prefill_mode(route.as_deref(), legacy.as_deref())
    })
}

/// Static part of the cache-hit prefill policy used before a chunk graph is
/// built. The chunk planner must not sample live allocator state: that would
/// make chunk boundaries depend on a transient probe that the per-layer route
/// selection may observe differently later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Gemma4PagedPrefillRoutePolicy {
    Auto,
    ForceVarlen,
    ForceLegacy,
    NonV2,
}

fn gemma4_paged_prefill_route_policy_for_mode(
    mode: CacheHitPrefillMode,
) -> Gemma4PagedPrefillRoutePolicy {
    match mode {
        CacheHitPrefillMode::Auto => Gemma4PagedPrefillRoutePolicy::Auto,
        CacheHitPrefillMode::ForceVarlen => Gemma4PagedPrefillRoutePolicy::ForceVarlen,
        CacheHitPrefillMode::ForceLegacy => Gemma4PagedPrefillRoutePolicy::ForceLegacy,
        CacheHitPrefillMode::ForceSdpa | CacheHitPrefillMode::ForceHostRead => {
            Gemma4PagedPrefillRoutePolicy::NonV2
        }
    }
}

pub(crate) fn gemma4_paged_prefill_route_policy() -> Gemma4PagedPrefillRoutePolicy {
    gemma4_paged_prefill_route_policy_for_mode(cache_hit_prefill_mode())
}

/// Pre-plan the V2 layout, if any, for a candidate compute chunk.
///
/// Auto mode deliberately reuses the normal route planner with no live-memory
/// sample. This is sufficient for the safety decision: an oversized varlen
/// auxiliary layout estimates as `u64::MAX`, so both the pre-plan and the
/// later live-memory plan must select SDPA. Gemma4's paged cache is BF16, hence
/// the two-byte estimate matches the runtime cache-hit path.
pub(crate) fn gemma4_paged_prefill_v2_layout_for_chunk(
    policy: Gemma4PagedPrefillRoutePolicy,
    query_tokens: u32,
    total_context: u32,
    num_query_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Option<PagedAttentionV2Layout> {
    match policy {
        Gemma4PagedPrefillRoutePolicy::ForceVarlen => Some(PagedAttentionV2Layout::Varlen),
        Gemma4PagedPrefillRoutePolicy::ForceLegacy => Some(PagedAttentionV2Layout::SingleRowBatch),
        Gemma4PagedPrefillRoutePolicy::NonV2 => None,
        Gemma4PagedPrefillRoutePolicy::Auto => {
            let estimated_sdpa_bytes = estimate_paged_pool_sdpa_bytes(
                query_tokens as u64,
                total_context as u64,
                num_query_heads as u64,
                num_kv_heads as u64,
                head_dim as u64,
                2,
            );
            let estimated_varlen_bytes = estimate_varlen_paged_attention_bytes(
                query_tokens as u64,
                total_context as u64,
                num_query_heads as u64,
                num_kv_heads as u64,
                head_dim as u64,
                2,
            );
            let plan = select_cache_hit_prefill_plan(
                CacheHitPrefillMode::Auto,
                query_tokens as u64,
                estimated_sdpa_bytes,
                estimated_varlen_bytes,
                None,
                true,
            );
            match plan.path {
                CacheHitPrefillPath::PagedVarlen => Some(PagedAttentionV2Layout::Varlen),
                CacheHitPrefillPath::PagedLegacy => Some(PagedAttentionV2Layout::SingleRowBatch),
                CacheHitPrefillPath::PagedPoolSdpa | CacheHitPrefillPath::HostRead => None,
            }
        }
    }
}

fn prefill_sdpa_effective_dtype(query: DType, cache: Option<DType>) -> Option<DType> {
    let cache = cache?;
    match (query, cache) {
        (DType::Float16, DType::Float16) => Some(DType::Float16),
        (DType::BFloat16, DType::BFloat16) => Some(DType::BFloat16),
        (DType::Float32, DType::Float16 | DType::BFloat16 | DType::Float32)
        | (DType::Float16 | DType::BFloat16, DType::Float32)
        | (DType::Float16, DType::BFloat16)
        | (DType::BFloat16, DType::Float16) => Some(DType::Float32),
        _ => None,
    }
}

fn mlx_sdpa_uses_fused_kernel(
    query_tokens: u64,
    num_query_heads: u64,
    num_kv_heads: u64,
    head_dim: u64,
) -> bool {
    if num_kv_heads == 0 || !num_query_heads.is_multiple_of(num_kv_heads) {
        return false;
    }
    if query_tokens <= 8 {
        let supported = matches!(head_dim, 64 | 96 | 128 | 256);
        return supported && query_tokens.saturating_mul(num_query_heads / num_kv_heads) <= 32;
    }
    // The pinned MLX dispatcher has fused full-attention kernels for these
    // widths. Gemma4's D=512 global attention deliberately stays on the
    // conservative score-matrix estimate below.
    matches!(head_dim, 64 | 80 | 128)
}

fn estimate_paged_pool_sdpa_bytes(
    query_tokens: u64,
    total_context: u64,
    num_query_heads: u64,
    num_kv_heads: u64,
    head_dim: u64,
    dtype_bytes: u64,
) -> u64 {
    let one_kv = total_context
        .saturating_mul(num_kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(dtype_bytes);
    let one_query = query_tokens
        .saturating_mul(num_query_heads)
        .saturating_mul(head_dim)
        .saturating_mul(dtype_bytes);
    let gathered = one_kv
        .saturating_mul(4)
        .saturating_add(one_query.saturating_mul(2))
        .saturating_add(PREFILL_FIXED_OVERHEAD_BYTES);
    if mlx_sdpa_uses_fused_kernel(query_tokens, num_query_heads, num_kv_heads, head_dim) {
        return gathered;
    }
    let scores = num_query_heads
        .saturating_mul(query_tokens)
        .saturating_mul(total_context)
        .saturating_mul(dtype_bytes);
    let fp32_output = num_query_heads
        .saturating_mul(query_tokens)
        .saturating_mul(head_dim)
        .saturating_mul(4);
    gathered.saturating_add(scores).saturating_add(fp32_output)
}

fn estimate_varlen_paged_attention_bytes(
    query_tokens: u64,
    total_context: u64,
    num_query_heads: u64,
    num_kv_heads: u64,
    head_dim: u64,
    dtype_bytes: u64,
) -> u64 {
    let output = query_tokens
        .saturating_mul(num_query_heads)
        .saturating_mul(head_dim)
        .saturating_mul(dtype_bytes);
    if total_context <= 512 {
        return output.saturating_add(PREFILL_FIXED_OVERHEAD_BYTES);
    }
    let Ok(query_tokens_u32) = u32::try_from(query_tokens) else {
        return u64::MAX;
    };
    let Ok(total_context_u32) = u32::try_from(total_context) else {
        return u64::MAX;
    };
    let Ok(num_query_heads_u32) = u32::try_from(num_query_heads) else {
        return u64::MAX;
    };
    let Ok(num_kv_heads_u32) = u32::try_from(num_kv_heads) else {
        return u64::MAX;
    };
    let Ok(head_dim_u32) = u32::try_from(head_dim) else {
        return u64::MAX;
    };
    if !paged_attention_v2_aux_fits(
        PagedAttentionV2Layout::Varlen,
        query_tokens_u32,
        num_query_heads_u32,
        num_kv_heads_u32,
        total_context_u32,
        head_dim_u32,
    ) {
        return u64::MAX;
    }
    let partitions = paged_attention_v2_partition_upper_bound(
        PagedAttentionV2Layout::Varlen,
        query_tokens_u32,
        num_query_heads_u32,
        num_kv_heads_u32,
        total_context_u32,
        head_dim_u32,
    );
    let rows = query_tokens
        .saturating_mul(num_query_heads)
        .saturating_mul(partitions);
    let partial_output = rows.saturating_mul(head_dim).saturating_mul(dtype_bytes);
    let softmax_state = rows.saturating_mul(2).saturating_mul(4);
    output
        .saturating_add(partial_output)
        .saturating_add(softmax_state)
        .saturating_add(PREFILL_FIXED_OVERHEAD_BYTES)
}

fn select_live_prefill_headroom(
    allocator_available_bytes: Option<u64>,
    metal_available_bytes: Option<u64>,
) -> Option<u64> {
    match (allocator_available_bytes, metal_available_bytes) {
        (Some(allocator), Some(metal)) => Some(allocator.min(metal)),
        (Some(allocator), None) => Some(allocator),
        (None, Some(metal)) => Some(metal),
        (None, None) => None,
    }
}

fn live_prefill_headroom(snapshot: PagedPrefillMemorySnapshot) -> LivePrefillHeadroom {
    let allocator_ceiling_bytes = snapshot.allocator_limit_bytes.map(|limit| {
        snapshot
            .metal_recommended_working_set_bytes
            .map(|recommended| limit.min(recommended.saturating_mul(95) / 100))
            .unwrap_or(limit)
    });
    let mut allocator_available_bytes = allocator_ceiling_bytes
        .zip(snapshot.allocator_active_bytes)
        .map(|(ceiling, active)| ceiling.saturating_sub(active));
    let metal_available_bytes = snapshot
        .metal_recommended_working_set_bytes
        .zip(snapshot.metal_current_allocated_bytes)
        .map(|(recommended, current)| {
            let reclaimable_cache = snapshot.allocator_cached_bytes.unwrap_or(0).min(current);
            recommended.saturating_sub(current.saturating_sub(reclaimable_cache))
        });
    if metal_available_bytes.is_none() {
        allocator_available_bytes = allocator_available_bytes.map(|available| {
            available.saturating_sub(snapshot.paged_pool_allocated_bytes.unwrap_or(0))
        });
    }
    LivePrefillHeadroom {
        selected_bytes: select_live_prefill_headroom(
            allocator_available_bytes,
            metal_available_bytes,
        ),
        allocator_available_bytes,
        metal_available_bytes,
    }
}

fn select_cache_hit_prefill_plan(
    mode: CacheHitPrefillMode,
    query_tokens: u64,
    estimated_sdpa_bytes: u64,
    estimated_varlen_bytes: u64,
    live_headroom_bytes: Option<u64>,
    graph_backend_available: bool,
) -> CacheHitPrefillPlan {
    let path = match mode {
        CacheHitPrefillMode::ForceSdpa if graph_backend_available => {
            CacheHitPrefillPath::PagedPoolSdpa
        }
        CacheHitPrefillMode::ForceVarlen if graph_backend_available => {
            CacheHitPrefillPath::PagedVarlen
        }
        CacheHitPrefillMode::ForceLegacy => CacheHitPrefillPath::PagedLegacy,
        CacheHitPrefillMode::ForceHostRead => CacheHitPrefillPath::HostRead,
        CacheHitPrefillMode::Auto if graph_backend_available => match live_headroom_bytes {
            Some(headroom) => {
                let budget = headroom
                    .saturating_sub(PREFILL_HEADROOM_RESERVE_BYTES)
                    .saturating_mul(9)
                    / 10;
                if query_tokens <= 8 {
                    CacheHitPrefillPath::PagedVarlen
                } else if estimated_sdpa_bytes <= budget {
                    CacheHitPrefillPath::PagedPoolSdpa
                } else if estimated_varlen_bytes <= budget
                    || estimated_varlen_bytes <= estimated_sdpa_bytes
                {
                    CacheHitPrefillPath::PagedVarlen
                } else {
                    CacheHitPrefillPath::PagedPoolSdpa
                }
            }
            None => {
                if query_tokens > 8 && estimated_sdpa_bytes < estimated_varlen_bytes {
                    CacheHitPrefillPath::PagedPoolSdpa
                } else {
                    CacheHitPrefillPath::PagedVarlen
                }
            }
        },
        CacheHitPrefillMode::Auto
        | CacheHitPrefillMode::ForceSdpa
        | CacheHitPrefillMode::ForceVarlen => CacheHitPrefillPath::HostRead,
    };
    CacheHitPrefillPlan {
        path,
        estimated_sdpa_bytes,
        estimated_varlen_bytes,
        live_headroom_bytes,
    }
}

fn native_kv_write_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::inference_trace::env_flag_enabled_or_default("MLX_GEMMA4_NATIVE_KV_WRITE", true)
    })
}

/// Trim mask to match K/V sequence length (e.g. after RotatingKVCache eviction).
fn trim_mask(mask: Option<&MxArray>, kv_len: i64) -> Result<Option<MxArray>> {
    match mask {
        Some(m) => {
            let mask_len = m.shape_at(3)?;
            if mask_len == kv_len {
                Ok(Some(m.clone()))
            } else if mask_len > kv_len {
                Ok(Some(m.slice_axis(3, mask_len - kv_len, mask_len)?))
            } else {
                Err(Error::from_reason(format!(
                    "Gemma4 attention mask is shorter than K/V: mask_len={mask_len}, kv_len={kv_len}"
                )))
            }
        }
        None => Ok(None),
    }
}

// ============================================
// Gemma4 Proportional RoPE (global layers)
// ============================================

/// Gemma4 proportional RoPE for global attention layers.
///
/// 1:1 port of mlx-lm `ProportionalRoPE` (rope_utils.py).
/// Uses inf-padded frequencies with a SINGLE `mx.fast.rope` call.
/// Non-rotated dimensions get `inf` frequency → no rotation (identity).
///
/// Key insight: exponent denominator = full `dims` (head_size), not `rotated_dims`.
/// Only `partial_rotary_factor` fraction of dims are actually rotated.
pub(crate) struct Gemma4ProportionalRoPE {
    /// Pre-computed frequencies for `mx.fast.rope`, shape [dims/2].
    /// First rotated_dims/2 entries: factor * base^(2i / dims)
    /// Remaining entries: inf (causes no rotation in mx.fast.rope)
    freqs: MxArray,
    /// Full head dimension (e.g. 512)
    dims: i32,
}

impl Gemma4ProportionalRoPE {
    /// Create proportional RoPE for global attention.
    ///
    /// Matches mlx-lm rope_utils.py:ProportionalRoPE.__init__
    ///
    /// # Arguments
    /// * `dims` - Full head dimension (e.g. 512)
    /// * `partial_rotary_factor` - Fraction of dims to rotate (e.g. 0.25)
    /// * `base` - RoPE theta (e.g. 1_000_000.0)
    pub(crate) fn new(dims: i32, partial_rotary_factor: f64, base: f64) -> Result<Self> {
        // rotated_dims = int(dims * partial_rotary_factor)
        let rotated_dims = (dims as f64 * partial_rotary_factor) as i32;
        let half_rotated = (rotated_dims / 2) as usize;
        let half_dims = (dims / 2) as usize;
        let nope_dims = half_dims - half_rotated;

        // freqs = concat([base^(arange(0,rotated_dims,2)/dims), full(inf, nope_dims)])
        let mut freqs_data: Vec<f32> = Vec::with_capacity(half_dims);
        for i in 0..half_rotated {
            let exponent = (2 * i) as f64 / dims as f64;
            freqs_data.push(base.powf(exponent) as f32);
        }
        // Pad with inf for non-rotated dimensions (identity rotation)
        freqs_data.extend(std::iter::repeat_n(f32::INFINITY, nope_dims));

        let freqs = MxArray::from_float32(&freqs_data, &[half_dims as i64])?;

        Ok(Self { freqs, dims })
    }

    /// Apply proportional RoPE to tensor in [B, H, T, D] format.
    ///
    /// Single fused `mx.fast.rope` call with inf-padded frequencies.
    /// No split/scatter needed — the kernel handles everything.
    pub(crate) fn forward(&self, x: &MxArray, offset: i32) -> Result<MxArray> {
        let offset_arr = MxArray::from_int32(&[offset], &[1])?;
        let handle = unsafe {
            sys::mlx_fast_rope_with_freqs(
                x.handle.0,
                self.dims, // full head dimension
                false,     // traditional=False (neox-style)
                0.0,       // base ignored when freqs provided
                1.0,       // scale=1.0
                offset_arr.handle.0,
                self.freqs.handle.0,
            )
        };
        MxArray::from_handle(handle, "proportional_rope")
    }
}

// ============================================
// Gemma4 RoPE dispatch (sliding vs global)
// ============================================

/// RoPE variant for Gemma4 attention layers.
enum Gemma4RoPE {
    /// Standard RoPE for sliding (local) attention layers.
    /// Uses `fast.rope(dims=head_dim, base=10K)` — correct because dims == head_size.
    Standard(RoPE),
    /// Proportional RoPE for global (full) attention layers.
    /// Uses mx.fast.rope with precomputed freqs on only the rotated dims.
    Proportional(Gemma4ProportionalRoPE),
}

impl Gemma4RoPE {
    fn forward(&self, x: &MxArray, offset: i32) -> Result<MxArray> {
        match self {
            Self::Standard(rope) => rope.forward(x, Some(offset)),
            Self::Proportional(rope) => rope.forward(x, offset),
        }
    }
}

// ============================================
// Gemma4 Attention
// ============================================

/// Gemma4 multi-head attention with QKV normalization and dual RoPE.
///
/// Key differences from Qwen3.5 attention:
/// 1. No gating (standard attention, not gated)
/// 2. Sliding layers: full RoPE rotation with theta=10K
/// 3. Global layers: proportional RoPE rotation with theta=1M (head_size denominator)
/// 4. Different head dimensions per layer type (sliding vs global)
/// 5. Optional K=V sharing (keys and values share projection weights)
/// 6. Values are also RMS-normalized (scale-free, no learnable weight)
/// 7. Attention scale = 1.0 (QK norm handles scaling; no query_pre_attn_scalar)
pub struct Gemma4Attention {
    q_proj: LinearProj,
    k_proj: LinearProj,
    v_proj: Option<LinearProj>, // None when attention_k_eq_v=true
    o_proj: LinearProj,

    q_norm: RMSNorm,
    k_norm: RMSNorm,
    /// V norm epsilon (scale-free: passes weight=None to rms_norm, matching Python RMSNormNoScale)
    v_norm_eps: f32,

    rope: Gemma4RoPE,

    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    k_is_v: bool,
}

impl Gemma4Attention {
    pub fn new(config: &Gemma4Config, layer_idx: usize) -> Result<Self> {
        let is_sliding = config.is_sliding_layer(layer_idx);
        let is_global = !is_sliding;

        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.effective_kv_heads(is_global);
        let head_dim = config.effective_head_dim(is_global);
        let has_bias = config.attention_bias;

        // K=V sharing only applies to global (full attention) layers.
        // vLLM: use_k_eq_v = self.is_full_attention and config.attention_k_eq_v
        let k_is_v = is_global && config.attention_k_eq_v;

        let q_proj = Linear::new(
            hidden_size as u32,
            (num_heads * head_dim) as u32,
            Some(has_bias),
        )?;
        let k_proj = Linear::new(
            hidden_size as u32,
            (num_kv_heads * head_dim) as u32,
            Some(has_bias),
        )?;

        // When k_is_v, we skip v_proj entirely and reuse k_proj output
        let v_proj = if k_is_v {
            None
        } else {
            Some(LinearProj::Standard(Linear::new(
                hidden_size as u32,
                (num_kv_heads * head_dim) as u32,
                Some(has_bias),
            )?))
        };

        let o_proj = Linear::new(
            (num_heads * head_dim) as u32,
            hidden_size as u32,
            Some(has_bias),
        )?;

        let q_norm = RMSNorm::new(head_dim as u32, Some(config.rms_norm_eps))?;
        let k_norm = RMSNorm::new(head_dim as u32, Some(config.rms_norm_eps))?;
        // V norm is scale-free: passes weight=None to rms_norm
        // Matches Python's RMSNormNoScale: mx.fast.rms_norm(x, None, eps)
        let v_norm_eps = config.rms_norm_eps as f32;

        // RoPE: sliding uses standard RoPE (theta=10K, dims=head_dim).
        // Global uses proportional RoPE (theta=1M, partial rotation via mx.fast.rope).
        let rope = if is_sliding {
            Gemma4RoPE::Standard(RoPE::new(
                config.rope_dims_sliding(),
                Some(false),
                Some(config.rope_local_base_freq),
                None,
            ))
        } else {
            Gemma4RoPE::Proportional(Gemma4ProportionalRoPE::new(
                head_dim,                     // full head dimension (e.g. 512)
                config.partial_rotary_factor, // fraction of dims to rotate (e.g. 0.25)
                config.rope_theta,            // 1M
            )?)
        };

        Ok(Self {
            q_proj: LinearProj::Standard(q_proj),
            k_proj: LinearProj::Standard(k_proj),
            v_proj,
            o_proj: LinearProj::Standard(o_proj),
            q_norm,
            k_norm,
            v_norm_eps,
            rope,
            num_heads,
            num_kv_heads,
            head_dim,
            k_is_v,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input [B, T, hidden_size]
    /// * `mask` - Attention mask
    /// * `cache` - Layer cache (KVCache for global, RotatingKVCache for sliding)
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Gemma4LayerCache>,
        needs_stash: bool,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;
        let trace_enabled = inference_trace_enabled();

        // Q/K/V projections.
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = if self.k_is_v {
            keys.clone()
        } else {
            self.v_proj.as_ref().unwrap().forward(x)?
        };

        // Reshape to [B, T, H, D]
        let queries =
            queries.reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?;
        let keys = keys.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let values = values.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;

        // QKV normalization
        let queries = self.q_norm.forward(&queries)?;
        let keys = self.k_norm.forward(&keys)?;
        // V norm: scale-free (weight=None), matching Python's RMSNormNoScale
        let values = {
            let handle = unsafe {
                sys::mlx_fast_rms_norm(values.handle.0, std::ptr::null_mut(), self.v_norm_eps)
            };
            MxArray::from_handle(handle, "v_norm")?
        };

        // Transpose to [B, H, T, D] BEFORE RoPE
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;

        // Apply RoPE with cache offset
        let offset = cache.as_ref().map_or(0, |c| c.get_offset());
        let queries = self.rope.forward(&queries, offset)?;
        let keys = self.rope.forward(&keys, offset)?;

        // Update cache
        let (keys, values) = if let Some(c) = cache {
            if needs_stash {
                c.update_and_fetch_stash(&keys, &values)?
            } else {
                c.update_and_fetch(&keys, &values)?
            }
        } else {
            (keys, values)
        };

        let mask = trim_mask(mask, keys.shape_at(2)?)?;
        if trace_enabled && offset > 0 && seq_len > 1 {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 attention_flat_kv_ready offset_before={} seq_len={} kv_len={} mask_len={} needs_stash={}",
                offset,
                seq_len,
                keys.shape_at(2).unwrap_or(-1),
                mask.as_ref().and_then(|m| m.shape_at(3).ok()).unwrap_or(0),
                needs_stash
            ));
        }

        if trace_enabled && offset > 0 && seq_len > 1 {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 attention_flat_sdpa_start offset_before={} seq_len={} q_heads={} kv_heads={} kv_len={} mask={}",
                offset,
                seq_len,
                queries.shape_at(1).unwrap_or(-1),
                keys.shape_at(1).unwrap_or(-1),
                keys.shape_at(2).unwrap_or(-1),
                if mask.is_some() { "explicit" } else { "causal" }
            ));
        }

        // Scaled dot-product attention with scale=1.0
        let output = if let Some(ref m) = mask {
            scaled_dot_product_attention(&queries, &keys, &values, 1.0, Some(m))?
        } else if seq_len > 1 {
            scaled_dot_product_attention_causal(&queries, &keys, &values, 1.0)?
        } else {
            scaled_dot_product_attention(&queries, &keys, &values, 1.0, None)?
        };
        if trace_enabled && offset > 0 && seq_len > 1 {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 attention_flat_sdpa_done offset_before={} seq_len={} kv_len={}",
                offset,
                seq_len,
                keys.shape_at(2).unwrap_or(-1)
            ));
        }

        // Transpose back [B, H, T, D] → [B, T, H*D]
        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        // Output projection
        self.o_proj.forward(&output)
    }

    /// Forward pass for KV-shared layers.
    ///
    /// Only computes queries; keys and values come from the anchor layer's cache.
    /// The anchor's K/V already have RoPE applied and are in [B, H, T, D] format.
    ///
    /// # Arguments
    /// * `x` - Input [B, T, hidden_size]
    /// * `mask` - Attention mask (may need to be adjusted for anchor's sequence length)
    /// * `shared_keys` - [B, H_kv, T_anchor, D] from anchor layer's cache (RoPE applied)
    /// * `shared_values` - [B, H_kv, T_anchor, D] from anchor layer's cache
    /// * `cache_offset` - RoPE offset for queries (total tokens seen so far, from anchor cache)
    pub fn forward_shared(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        shared_keys: &MxArray,
        shared_values: &MxArray,
        cache_offset: i32,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // Only compute queries
        let queries = self.q_proj.forward(x)?;
        let queries =
            queries.reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?;
        let queries = self.q_norm.forward(&queries)?;

        // Transpose to [B, H, T, D] before RoPE
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;

        // Apply RoPE to queries using the anchor's cache offset
        let queries = self.rope.forward(&queries, cache_offset)?;

        let mask = trim_mask(mask, shared_keys.shape_at(2)?)?;

        // Use shared K/V directly (already [B, H_kv, T, D] with RoPE applied)
        let output = if let Some(ref m) = mask {
            scaled_dot_product_attention(&queries, shared_keys, shared_values, 1.0, Some(m))?
        } else if seq_len > 1 {
            scaled_dot_product_attention_causal(&queries, shared_keys, shared_values, 1.0)?
        } else {
            scaled_dot_product_attention(&queries, shared_keys, shared_values, 1.0, None)?
        };

        // Transpose back [B, H, T, D] -> [B, T, H*D]
        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        // Output projection
        self.o_proj.forward(&output)
    }

    /// Compute one decode token against the authoritative physical paged K/V
    /// pool. The storage policy never changes: the SDPA route gathers compact
    /// graph views of the active request, while the PagedAttention route reads
    /// the same pool in place.
    fn forward_paged_single_token_attention(
        &self,
        x: &MxArray,
        queries_bhtd: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        paged_idx: u32,
        total_ctx: u32,
    ) -> Result<MxArray> {
        let mode = paged_decode_mode();
        let query_dtype = queries_bhtd.dtype()?;
        let cache_dtype = adapter.prefill_sdpa_cache_dtype();
        let (planned_path, estimated_gather_bytes) = select_paged_decode_path(
            mode,
            query_dtype,
            cache_dtype,
            total_ctx as u64,
            self.num_kv_heads as u64,
            self.head_dim as u64,
            crate::engine::persistence::compiled_forward_backend_available(),
        );
        let configured_mode = match mode {
            PagedDecodeMode::Auto => "auto",
            PagedDecodeMode::ForceSdpa => "sdpa",
            PagedDecodeMode::ForcePagedAttention => "paged_attention",
        };
        let (grouped_selector, grouped_stripe_override) = grouped_gemma4_diagnostic_config();
        // The custom primitive is lazy, so this is the exact shape/env
        // candidate rather than a claim that Metal pipeline-capability checks
        // have already succeeded. The dispatcher can still fall back to V2 at
        // evaluation time when the specialized pipeline is unavailable.
        let (paged_kernel_candidate, grouped_stripes) = grouped_gemma4_kernel_candidate(
            grouped_selector,
            grouped_stripe_override,
            total_ctx,
            query_dtype,
            cache_dtype,
            adapter.block_size(),
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        );

        if planned_path == PagedDecodePath::PagedPoolSdpa {
            match adapter.gather_kv_for_prefill_sdpa(paged_idx, total_ctx) {
                Ok((keys, values)) => {
                    match scaled_dot_product_attention(queries_bhtd, &keys, &values, 1.0, None) {
                        Ok(output) => {
                            if paged_idx == 0
                                && should_report_paged_decode_path(PagedDecodePath::PagedPoolSdpa)
                            {
                                tracing::info!(
                                    target: "mlx_core::inference",
                                    event = "gemma4_paged_decode_route",
                                    configured_mode,
                                    path = "paged_pool_sdpa",
                                    physical_pool_authoritative = true,
                                    paged_kernel_candidate,
                                    grouped_selector,
                                    grouped_stripes,
                                    total_context_tokens = total_ctx,
                                    num_query_heads = self.num_heads,
                                    num_kv_heads = self.num_kv_heads,
                                    head_dim = self.head_dim,
                                    estimated_gather_mib = estimated_gather_bytes as f64
                                        / (1024.0 * 1024.0),
                                    "Gemma4 paged decode attention route selected"
                                );
                            }
                            return output.astype(x.dtype()?);
                        }
                        Err(err) => {
                            if paged_idx == 0 && should_report_paged_decode_fallback() {
                                tracing::warn!(
                                    target: "mlx_core::inference",
                                    event = "gemma4_paged_decode_fallback",
                                    configured_mode,
                                    failed_path = "paged_pool_sdpa",
                                    stage = "sdpa",
                                    error = %err,
                                    "Gemma4 paged decode SDPA construction failed"
                                );
                            }
                        }
                    }
                }
                Err(err) => {
                    if paged_idx == 0 && should_report_paged_decode_fallback() {
                        tracing::warn!(
                            target: "mlx_core::inference",
                            event = "gemma4_paged_decode_fallback",
                            configured_mode,
                            failed_path = "paged_pool_sdpa",
                            stage = "paged_pool_gather",
                            error = %err,
                            "Gemma4 paged decode compact K/V gather failed"
                        );
                    }
                }
            }
        }

        let queries_3d = queries_bhtd.squeeze(Some(&[2]))?.reshape(&[
            1,
            self.num_heads as i64,
            self.head_dim as i64,
        ])?;
        let attn_3d = match adapter.gather_kv_for_decode_graph(paged_idx, &queries_3d, 1.0, 1.0) {
            Ok(output) => output,
            Err(err) => {
                if paged_idx == 0 && should_report_paged_decode_fallback() {
                    tracing::warn!(
                        target: "mlx_core::inference",
                        event = "gemma4_paged_decode_fallback",
                        configured_mode,
                        failed_path = "paged_attention_graph",
                        stage = "graph_construction",
                        error = %err,
                        "Gemma4 graph-native paged decode attention failed"
                    );
                }
                adapter
                    .gather_kv_for_decode(paged_idx, &queries_3d, 1.0, 1.0)
                    .map_err(napi::Error::from_reason)?
            }
        };
        if paged_idx == 0 && should_report_paged_decode_path(PagedDecodePath::PagedAttention) {
            tracing::info!(
                target: "mlx_core::inference",
                event = "gemma4_paged_decode_route",
                configured_mode,
                path = "paged_attention",
                physical_pool_authoritative = true,
                paged_kernel_candidate,
                grouped_selector,
                grouped_stripes,
                total_context_tokens = total_ctx,
                num_query_heads = self.num_heads,
                num_kv_heads = self.num_kv_heads,
                head_dim = self.head_dim,
                estimated_gather_mib = estimated_gather_bytes as f64 / (1024.0 * 1024.0),
                "Gemma4 paged decode attention route selected"
            );
        }
        let attn_3d = attn_3d.astype(x.dtype()?)?;
        attn_3d.reshape(&[1, self.num_heads as i64, 1, self.head_dim as i64])
    }

    /// Compute a multi-token cache-hit prefill while keeping the physical
    /// paged K/V pool authoritative.
    ///
    /// The operator policy is intentionally independent from storage:
    /// matrix SDPA gathers K/V through graph-native pool views, compact
    /// varlen attention consumes one block-table row for the request, and the
    /// former per-query duplicated-row bridge remains a diagnostic fallback.
    fn forward_paged_cache_hit_prefill(
        &self,
        x: &MxArray,
        queries_bhtd: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        paged_idx: u32,
        cached_prefix_len: u32,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;
        let total_ctx = cached_prefix_len
            .checked_add(seq_len as u32)
            .ok_or_else(|| Error::from_reason("Gemma4 cache-hit prefill context overflow"))?;
        // Every graph-native paged bridge currently accepts one request. The
        // former call sites applied this guard individually; keep it in the
        // shared policy so a future batched caller cannot squeeze/broadcast
        // the wrong request shape.
        let graph_backend_available =
            batch == 1 && crate::engine::persistence::compiled_forward_backend_available();
        let mode = cache_hit_prefill_mode();
        let effective_sdpa_dtype =
            prefill_sdpa_effective_dtype(queries_bhtd.dtype()?, adapter.prefill_sdpa_cache_dtype());
        let dtype_bytes = match effective_sdpa_dtype {
            Some(DType::Float16 | DType::BFloat16) => 2,
            Some(DType::Float32) | None => 4,
            Some(_) => 4,
        };
        let estimated_sdpa_bytes = estimate_paged_pool_sdpa_bytes(
            seq_len as u64,
            total_ctx as u64,
            self.num_heads as u64,
            self.num_kv_heads as u64,
            self.head_dim as u64,
            dtype_bytes,
        );
        let estimated_varlen_bytes = estimate_varlen_paged_attention_bytes(
            seq_len as u64,
            total_ctx as u64,
            self.num_heads as u64,
            self.num_kv_heads as u64,
            self.head_dim as u64,
            dtype_bytes,
        );
        let memory_probe_performed =
            mode == CacheHitPrefillMode::Auto && seq_len > 8 && graph_backend_available;
        let live_headroom = if memory_probe_performed {
            live_prefill_headroom(adapter.prefill_memory_snapshot())
        } else {
            LivePrefillHeadroom::default()
        };
        let plan = select_cache_hit_prefill_plan(
            mode,
            seq_len as u64,
            estimated_sdpa_bytes,
            estimated_varlen_bytes,
            live_headroom.selected_bytes,
            graph_backend_available,
        );
        let configured_mode = match mode {
            CacheHitPrefillMode::Auto => "auto",
            CacheHitPrefillMode::ForceSdpa => "sdpa",
            CacheHitPrefillMode::ForceVarlen => "varlen",
            CacheHitPrefillMode::ForceLegacy => "legacy",
            CacheHitPrefillMode::ForceHostRead => "host_read",
        };
        let planned_path = match plan.path {
            CacheHitPrefillPath::PagedPoolSdpa => "paged_pool_sdpa",
            CacheHitPrefillPath::PagedVarlen => "paged_attention_varlen",
            CacheHitPrefillPath::PagedLegacy => "paged_attention_legacy",
            CacheHitPrefillPath::HostRead => "host_read",
        };
        let report_route = paged_idx == 0 && seq_len > 8;
        if report_route {
            tracing::info!(
                target: "mlx_core::inference",
                event = "gemma4_cache_hit_prefill_plan",
                layer = paged_idx,
                suffix_tokens = seq_len,
                cached_prefix_tokens = cached_prefix_len,
                total_context_tokens = total_ctx,
                configured_mode,
                planned_path,
                graph_backend_available,
                effective_sdpa_dtype = ?effective_sdpa_dtype,
                estimated_sdpa_mib = plan.estimated_sdpa_bytes as f64 / (1024.0 * 1024.0),
                estimated_varlen_mib = plan.estimated_varlen_bytes as f64 / (1024.0 * 1024.0),
                memory_probe_performed,
                live_headroom_reported = plan.live_headroom_bytes.is_some(),
                live_headroom_mib = plan.live_headroom_bytes.unwrap_or(0) as f64
                    / (1024.0 * 1024.0),
                allocator_headroom_mib = live_headroom.allocator_available_bytes.unwrap_or(0)
                    as f64
                    / (1024.0 * 1024.0),
                metal_headroom_mib = live_headroom.metal_available_bytes.unwrap_or(0) as f64
                    / (1024.0 * 1024.0),
                "Gemma4 cache-hit prefill route selected"
            );
        }

        let mut attention = None;
        if plan.path == CacheHitPrefillPath::PagedPoolSdpa && graph_backend_available {
            let started = std::time::Instant::now();
            match adapter.gather_kv_for_prefill_sdpa(paged_idx, total_ctx) {
                Ok((keys, values)) => {
                    match scaled_dot_product_attention_causal(queries_bhtd, &keys, &values, 1.0) {
                        Ok(output) => {
                            if report_route {
                                tracing::info!(
                                    target: "mlx_core::inference",
                                    event = "gemma4_cache_hit_prefill_route",
                                    layer = paged_idx,
                                    path = "paged_pool_sdpa",
                                    elapsed_ms = elapsed_ms(started),
                                    "Gemma4 cache-hit prefill graph constructed"
                                );
                            }
                            attention = Some(output);
                        }
                        Err(err) => tracing::warn!(
                            target: "mlx_core::inference",
                            event = "gemma4_cache_hit_prefill_fallback",
                            layer = paged_idx,
                            failed_path = "paged_pool_sdpa",
                            stage = "sdpa",
                            error = %err,
                            "Gemma4 cache-hit SDPA construction failed"
                        ),
                    }
                }
                Err(err) => tracing::warn!(
                    target: "mlx_core::inference",
                    event = "gemma4_cache_hit_prefill_fallback",
                    layer = paged_idx,
                    failed_path = "paged_pool_sdpa",
                    stage = "paged_pool_gather",
                    error = %err,
                    "Gemma4 cache-hit paged-pool gather failed"
                ),
            }
        }

        let try_varlen = graph_backend_available
            && attention.is_none()
            && matches!(
                mode,
                CacheHitPrefillMode::Auto | CacheHitPrefillMode::ForceVarlen
            )
            && matches!(
                plan.path,
                CacheHitPrefillPath::PagedPoolSdpa | CacheHitPrefillPath::PagedVarlen
            );
        if try_varlen {
            let started = std::time::Instant::now();
            let queries_3d = queries_bhtd
                .squeeze(Some(&[0]))?
                .transpose(Some(&[1, 0, 2]))?;
            match adapter.gather_kv_for_prefill_chunk_varlen(
                paged_idx,
                &queries_3d,
                cached_prefix_len,
                1.0,
            ) {
                Ok(output) => {
                    let output = output.astype(x.dtype()?)?;
                    let output = output.transpose(Some(&[1, 0, 2]))?.reshape(&[
                        batch,
                        self.num_heads as i64,
                        seq_len,
                        self.head_dim as i64,
                    ])?;
                    if report_route {
                        tracing::info!(
                            target: "mlx_core::inference",
                            event = "gemma4_cache_hit_prefill_route",
                            layer = paged_idx,
                            path = "paged_attention_varlen",
                            elapsed_ms = elapsed_ms(started),
                            "Gemma4 cache-hit prefill graph constructed"
                        );
                    }
                    attention = Some(output);
                }
                Err(err) => tracing::warn!(
                    target: "mlx_core::inference",
                    event = "gemma4_cache_hit_prefill_fallback",
                    layer = paged_idx,
                    failed_path = "paged_attention_varlen",
                    stage = "graph_construction",
                    error = %err,
                    "Gemma4 cache-hit compact varlen construction failed"
                ),
            }
        }

        // The duplicated-block-row bridge is retained only as an explicit
        // regression/compatibility escape hatch. Production Auto never falls
        // back into its O(query_tokens * block_count) metadata layout.
        let try_legacy =
            attention.is_none() && batch == 1 && mode == CacheHitPrefillMode::ForceLegacy;
        if try_legacy {
            let started = std::time::Instant::now();
            let queries_3d = queries_bhtd
                .squeeze(Some(&[0]))?
                .transpose(Some(&[1, 0, 2]))?;
            match adapter.gather_kv_for_prefill_chunk(
                paged_idx,
                &queries_3d,
                cached_prefix_len,
                1.0,
            ) {
                Ok(output) => {
                    let output = output.astype(x.dtype()?)?;
                    let output = output.transpose(Some(&[1, 0, 2]))?.reshape(&[
                        batch,
                        self.num_heads as i64,
                        seq_len,
                        self.head_dim as i64,
                    ])?;
                    if report_route {
                        tracing::info!(
                            target: "mlx_core::inference",
                            event = "gemma4_cache_hit_prefill_route",
                            layer = paged_idx,
                            path = "paged_attention_legacy",
                            elapsed_ms = elapsed_ms(started),
                            "Gemma4 cache-hit prefill graph constructed"
                        );
                    }
                    attention = Some(output);
                }
                Err(err) => tracing::warn!(
                    target: "mlx_core::inference",
                    event = "gemma4_cache_hit_prefill_fallback",
                    layer = paged_idx,
                    failed_path = "paged_attention_legacy",
                    stage = "graph_construction",
                    error = %err,
                    "Gemma4 cache-hit legacy paged construction failed"
                ),
            }
        }

        if let Some(output) = attention {
            return Ok(output);
        }

        // Last-resort compatibility path. It synchronously materializes the
        // paged K/V pool through the host and therefore must never be the
        // adaptive default.
        let started = std::time::Instant::now();
        let (keys, values) = adapter
            .read_kv_range(paged_idx, 0, total_ctx)
            .map_err(napi::Error::from_reason)?;
        let mask = create_causal_mask(seq_len as i32, Some(cached_prefix_len as i32), None)?;
        let output = scaled_dot_product_attention(queries_bhtd, &keys, &values, 1.0, Some(&mask))?;
        tracing::warn!(
            target: "mlx_core::inference",
            event = "gemma4_cache_hit_prefill_route",
            layer = paged_idx,
            path = "host_read_fallback",
            elapsed_ms = elapsed_ms(started),
            "Gemma4 cache-hit prefill used synchronous host materialization"
        );
        Ok(output)
    }

    /// Forward pass driven by `PagedKVCacheAdapter` for global Gemma4
    /// attention layers.
    ///
    /// Mirrors `Lfm2Attention::forward_paged` but adapted to Gemma4's
    /// quirks:
    /// * Q/K/V are reshaped to `[B, T, H, D]` BEFORE per-head RMSNorm
    ///   (matches `forward`).
    /// * V receives a SCALE-FREE RMSNorm (`mlx_fast_rms_norm` with
    ///   `weight=null`), not a learned-scale norm.
    /// * RoPE dispatches between `Standard` (sliding) and
    ///   `Proportional` (global) — only global layers should call this
    ///   method, but the dispatch is uniform.
    /// * Optional K=V sharing collapses the V projection.
    /// * Attention scale is `1.0` (not `head_dim^-0.5`).
    ///
    /// Caller responsibilities (mirrors LFM2 / Qwen3 helper contracts):
    /// 1. `adapter.record_tokens(suffix)` BEFORE this call so the
    ///    cursor is advanced; `update_keys_values` enforces alignment.
    /// 2. `paged_idx` is the GLOBAL-LAYER ORDINAL into the adapter's
    ///    `LayerKVPool` (NOT the absolute decoder index). The pool is
    ///    sized for the global layer count.
    /// 3. `first_logical_position` is the first token's logical index
    ///    in the FULL request — used both as the RoPE offset and the
    ///    `update_keys_values` write position.
    /// 4. The decoder layer's `input_layernorm` is applied OUTSIDE this
    ///    method so `x` here is already pre-normalized (matches the
    ///    flat path's call site).
    ///
    /// Output: `[1, seq_len, hidden_size]` so the decoder layer's
    /// `apply_ffn_ple_scalar` tail can consume it unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_paged(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        paged_idx: u32,
        first_logical_position: u32,
        cached_prefix_len: u32,
        is_prefill: bool,
        explicit_prefill_mask: Option<&MxArray>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;
        let trace_enabled = inference_trace_enabled();

        // 1. Q/K/V projections (matches `forward`).
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = if self.k_is_v {
            keys.clone()
        } else {
            self.v_proj.as_ref().unwrap().forward(x)?
        };

        // 2. Reshape to [B, T, H, D] BEFORE per-head norm (matches `forward`).
        let queries =
            queries.reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?;
        let keys = keys.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let values = values.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;

        // 3. QKV normalization (Q/K learned-scale, V scale-free).
        let queries = self.q_norm.forward(&queries)?;
        let keys = self.k_norm.forward(&keys)?;
        let values = {
            let handle = unsafe {
                sys::mlx_fast_rms_norm(values.handle.0, std::ptr::null_mut(), self.v_norm_eps)
            };
            MxArray::from_handle(handle, "v_norm")?
        };

        // 4. Transpose to [B, H, T, D] BEFORE RoPE (matches `forward`).
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;

        // 5. Apply RoPE using the request's logical offset.
        let rope_offset = first_logical_position as i32;
        let queries_bhtd = self.rope.forward(&queries, rope_offset)?;
        let keys_bhtd = self.rope.forward(&keys, rope_offset)?;
        let values_bhtd = values;

        // 6. Convert K/V into the paged layout `[num_tokens, n_kv_heads,
        //    head_dim]` expected by `update_keys_values`. Currently
        //    batch=1 so num_tokens = batch * seq_len = seq_len.
        //    [B, H_kv, T, D] -> [B, T, H_kv, D] -> [B*T, H_kv, D]
        let keys_paged = keys_bhtd.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
            batch * seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let values_paged = values_bhtd.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
            batch * seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;

        let write_trace_start = trace_enabled.then(std::time::Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 attention_paged_kv_write_start paged_idx={} first_position={} cached_prefix={} seq_len={} batch={} q_heads={} kv_heads={} head_dim={} input_dtype={:?} current_tokens={} blocks={}",
                paged_idx,
                first_logical_position,
                cached_prefix_len,
                seq_len,
                batch,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                x.dtype().ok(),
                adapter.current_token_count(),
                adapter.num_allocated_blocks()
            ));
        }
        let write_path = if native_kv_write_enabled() {
            match adapter.update_keys_values_native(
                paged_idx,
                &keys_paged,
                &values_paged,
                first_logical_position,
            ) {
                Ok(()) => "native",
                Err(err) => {
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] gemma4 attention_paged_kv_write_fallback paged_idx={} first_position={} seq_len={} error={}",
                            paged_idx, first_logical_position, seq_len, err
                        ));
                    }
                    adapter
                        .update_keys_values(
                            paged_idx,
                            &keys_paged,
                            &values_paged,
                            first_logical_position,
                        )
                        .map_err(napi::Error::from_reason)?;
                    "legacy"
                }
            }
        } else {
            adapter
                .update_keys_values(
                    paged_idx,
                    &keys_paged,
                    &values_paged,
                    first_logical_position,
                )
                .map_err(napi::Error::from_reason)?;
            "legacy"
        };
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 attention_paged_kv_write_done paged_idx={} first_position={} seq_len={} path={} elapsed_ms={:.1}",
                paged_idx,
                first_logical_position,
                seq_len,
                write_path,
                write_trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }

        // 7. Compute attention output. Gemma4's attention scale is 1.0
        //    (the QK norm handles scaling).
        //
        // Single-token query path (`seq_len == 1`) ALWAYS takes the
        // mask=None branch regardless of `is_prefill` /
        // `cached_prefix_len`. The query is at logical position
        // `first_logical_position` and every key in
        // `[0, first_logical_position + 1)` is at a strictly-earlier
        // (or equal) position, so a causal mask never filters anything
        // out. But MLX dispatches `scaled_dot_product_attention` to
        // different kernels with vs. without an explicit mask, and the
        // mask-bearing kernel uses a different BF16 reduction order.
        // For Gemma4 paged-vs-flat parity at the prefill→decode
        // boundary (split-prefill pass 2 and every decode step) we
        // need the SAME kernel as flat decode (`scaled_dot_product_
        // attention(..., None)` at attention.rs:319) — without this,
        // the K/V written for the last prompt position diverges from
        // the flat cache by a few ULP per layer in BF16, which
        // compounds into an argmax flip on the first decode step.
        let attn_bhtd = if is_prefill && seq_len > 1 {
            if cached_prefix_len == 0 {
                // Fresh multi-token prefill: SDPA over in-flight Q/K/V with
                // internal causal mask — UNLESS an explicit prefill mask is
                // supplied (Gemma 4 unified-vision bidirectional overlay), in
                // which case the boolean keep-mask is applied directly. The
                // explicit-mask kernel differs from the causal kernel in BF16
                // reduction order, so it is taken only when an overlay is
                // actually present (text/SigLIP/31B keep the causal fast path).
                let sdpa_trace_start = trace_enabled.then(std::time::Instant::now);
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 attention_paged_sdpa_causal_start paged_idx={} seq_len={} cached_prefix=0 explicit_mask={}",
                        paged_idx,
                        seq_len,
                        explicit_prefill_mask.is_some()
                    ));
                }
                let out = if let Some(mask) = explicit_prefill_mask {
                    scaled_dot_product_attention(
                        &queries_bhtd,
                        &keys_bhtd,
                        &values_bhtd,
                        1.0,
                        Some(mask),
                    )?
                } else {
                    scaled_dot_product_attention_causal(
                        &queries_bhtd,
                        &keys_bhtd,
                        &values_bhtd,
                        1.0,
                    )?
                };
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 attention_paged_sdpa_causal_done paged_idx={} seq_len={} elapsed_ms={:.1}",
                        paged_idx,
                        seq_len,
                        sdpa_trace_start.map(elapsed_ms).unwrap_or(0.0)
                    ));
                }
                out
            } else {
                self.forward_paged_cache_hit_prefill(
                    x,
                    &queries_bhtd,
                    adapter,
                    paged_idx,
                    cached_prefix_len,
                )?
            }
        } else {
            let total_ctx = adapter.current_token_count();
            self.forward_paged_single_token_attention(
                x,
                &queries_bhtd,
                adapter,
                paged_idx,
                total_ctx,
            )?
        };

        // 8. Output: [B, H, T, D] -> [B, T, H*D] -> projection.
        let output = attn_bhtd.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;
        self.o_proj.forward(&output)
    }

    // ========== Weight setters ==========

    /// Forward pass for KV-shared layers whose anchor is a global layer
    /// routed through the paged adapter.
    ///
    /// Only Q is computed; K and V are consumed directly from the anchor's
    /// paged slot (already RoPE-applied since the anchor wrote them
    /// post-RoPE during its own `forward_paged` call).
    ///
    /// Caller responsibilities:
    /// 1. `cache_offset` is the RoPE offset for the queries — equal to
    ///    the anchor's logical position when the anchor processed the
    ///    same chunk (i.e. `first_logical_position` for prefill,
    ///    `current_token_count - 1` for decode).
    /// 2. `total_ctx` is the number of K/V tokens available in the
    ///    anchor's paged slot. For prefill of a fresh suffix this is
    ///    `cached_prefix_len + seq_len`; for decode it is the live
    ///    token count.
    ///
    /// Output: `[1, seq_len, hidden_size]`, ready for the decoder
    /// layer's `apply_ffn_ple_scalar` tail.
    pub fn forward_paged_shared(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        anchor_paged_idx: u32,
        cache_offset: i32,
        total_ctx: u32,
        is_prefill: bool,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;
        // Q-only path (mirrors flat `forward_shared`).
        let queries = self.q_proj.forward(x)?;
        let queries =
            queries.reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?;
        let queries = self.q_norm.forward(&queries)?;
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let queries_bhtd = self.rope.forward(&queries, cache_offset)?;

        // SDPA. Same scale=1.0 as `forward_paged`. For cache-hit suffix
        // prefill, prefer the on-GPU paged prefill helper and fall back to
        // materialized K/V only if the bridge rejects this shape/request. For
        // decode (seq_len == 1), dispatch the on-GPU decode helper — every
        // cached key is at a strictly earlier position, so mask=None is
        // implicit.
        let attn_bhtd = if is_prefill && seq_len > 1 {
            let cached_prefix_len = total_ctx
                .checked_sub(seq_len as u32)
                .ok_or_else(|| Error::from_reason("forward_paged_shared: total_ctx < seq_len"))?;
            self.forward_paged_cache_hit_prefill(
                x,
                &queries_bhtd,
                adapter,
                anchor_paged_idx,
                cached_prefix_len,
            )?
        } else {
            self.forward_paged_single_token_attention(
                x,
                &queries_bhtd,
                adapter,
                anchor_paged_idx,
                total_ctx,
            )?
        };

        // Output projection.
        let output = attn_bhtd.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;
        self.o_proj.forward(&output)
    }

    // ========== Test-only weight getters ==========
    #[cfg(test)]
    pub(crate) fn q_proj_weight(&self) -> MxArray {
        self.q_proj.get_weight()
    }
    #[cfg(test)]
    pub(crate) fn k_proj_weight(&self) -> MxArray {
        self.k_proj.get_weight()
    }
    #[cfg(test)]
    pub(crate) fn v_proj_weight_opt(&self) -> Option<MxArray> {
        self.v_proj.as_ref().map(|p| p.get_weight())
    }
    #[cfg(test)]
    pub(crate) fn o_proj_weight(&self) -> MxArray {
        self.o_proj.get_weight()
    }
    #[cfg(test)]
    pub(crate) fn q_norm_weight(&self) -> MxArray {
        self.q_norm.get_weight()
    }
    #[cfg(test)]
    pub(crate) fn k_norm_weight(&self) -> MxArray {
        self.k_norm.get_weight()
    }

    pub fn set_q_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.q_proj.set_weight(w, "q_proj")
    }
    pub fn set_k_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.k_proj.set_weight(w, "k_proj")
    }
    pub fn set_v_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        if let Some(ref mut vp) = self.v_proj {
            vp.set_weight(w, "v_proj")
        } else {
            // k_is_v mode: v_proj doesn't exist, ignore silently
            Ok(())
        }
    }
    pub fn set_o_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.o_proj.set_weight(w, "o_proj")
    }
    pub fn set_q_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.q_proj.set_bias(b, "q_proj")
    }
    pub fn set_k_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.k_proj.set_bias(b, "k_proj")
    }
    pub fn set_v_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        if let Some(ref mut vp) = self.v_proj {
            vp.set_bias(b, "v_proj")
        } else {
            Ok(())
        }
    }
    pub fn set_o_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.o_proj.set_bias(b, "o_proj")
    }
    pub fn set_q_norm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.q_norm.set_weight(w)
    }
    pub fn set_k_norm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.k_norm.set_weight(w)
    }

    // ========== Quantized setters ==========

    pub fn set_quantized_q_proj(&mut self, ql: QuantizedLinear) {
        self.q_proj.set_quantized(ql);
    }
    pub fn set_quantized_k_proj(&mut self, ql: QuantizedLinear) {
        self.k_proj.set_quantized(ql);
    }
    pub fn set_quantized_v_proj(&mut self, ql: QuantizedLinear) {
        if let Some(ref mut vp) = self.v_proj {
            vp.set_quantized(ql);
        }
    }
    pub fn set_quantized_o_proj(&mut self, ql: QuantizedLinear) {
        self.o_proj.set_quantized(ql);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_paged_decode_route_keeps_storage_paged_and_bounds_sdpa_gather() {
        assert_eq!(parse_paged_decode_mode(None), PagedDecodeMode::Auto);
        assert_eq!(
            parse_paged_decode_mode(Some("sdpa")),
            PagedDecodeMode::ForceSdpa
        );
        assert_eq!(
            parse_paged_decode_mode(Some("paged")),
            PagedDecodeMode::ForcePagedAttention
        );
        assert_eq!(
            parse_paged_decode_mode(Some("unknown")),
            PagedDecodeMode::Auto
        );
        assert_eq!(
            resolve_paged_decode_mode(None, Some("1")),
            PagedDecodeMode::ForcePagedAttention,
            "the grouped-kernel diagnostic selector must reach physical paging"
        );
        assert_eq!(
            resolve_paged_decode_mode(Some("auto"), Some("force")),
            PagedDecodeMode::ForcePagedAttention
        );
        assert_eq!(
            resolve_paged_decode_mode(Some("sdpa"), Some("1")),
            PagedDecodeMode::ForceSdpa,
            "an explicit decode-route override wins over the grouped selector"
        );
        assert_eq!(
            resolve_paged_decode_mode(None, Some("0")),
            PagedDecodeMode::Auto
        );
        assert_eq!(parse_grouped_gemma4_selector(Some("force")), "force");
        assert_eq!(parse_grouped_gemma4_selector(Some("auto")), "auto");
        assert_eq!(
            parse_grouped_gemma4_selector(Some(" FORCE ")),
            "off",
            "route diagnostics must use the dispatcher's exact env syntax"
        );

        assert_eq!(
            grouped_gemma4_kernel_candidate(
                "force",
                Some(16),
                3_417,
                DType::BFloat16,
                Some(DType::BFloat16),
                16,
                16,
                1,
                512,
            ),
            ("grouped_gemma4_d512_staged", Some(16))
        );
        assert_eq!(
            grouped_gemma4_kernel_candidate(
                "off",
                Some(16),
                3_417,
                DType::BFloat16,
                Some(DType::BFloat16),
                16,
                16,
                1,
                512,
            ),
            ("generic_v2", None),
            "a stripes override alone must not enable the grouped kernel"
        );
        assert_eq!(
            grouped_gemma4_kernel_candidate(
                "force",
                Some(16),
                3_417,
                DType::Float16,
                Some(DType::Float16),
                16,
                16,
                1,
                512,
            ),
            ("generic_v2", None),
            "the diagnostic candidate must enforce the dispatcher's BF16 guard"
        );

        let short = select_paged_decode_path(
            PagedDecodeMode::Auto,
            DType::BFloat16,
            Some(DType::BFloat16),
            3417,
            1,
            512,
            true,
        );
        assert_eq!(short, (PagedDecodePath::PagedAttention, 13_996_032));

        assert_eq!(
            select_paged_decode_path(
                PagedDecodeMode::Auto,
                DType::BFloat16,
                Some(DType::BFloat16),
                16_384,
                1,
                512,
                true,
            )
            .0,
            PagedDecodePath::PagedAttention,
            "automatic decode must consume the physical paged pool directly"
        );
        assert_eq!(
            select_paged_decode_path(
                PagedDecodeMode::Auto,
                DType::BFloat16,
                Some(DType::BFloat16),
                16_385,
                1,
                512,
                true,
            )
            .0,
            PagedDecodePath::PagedAttention,
            "long contexts must retain the custom paged operator"
        );
        for (cache_dtype, graph_backend_available) in [(None, true), (Some(DType::BFloat16), false)]
        {
            assert_eq!(
                select_paged_decode_path(
                    PagedDecodeMode::Auto,
                    DType::BFloat16,
                    cache_dtype,
                    3417,
                    1,
                    512,
                    graph_backend_available,
                )
                .0,
                PagedDecodePath::PagedAttention,
                "FP8/unsupported cache layouts and non-Metal callers stay on paged attention"
            );
        }
        assert_eq!(
            select_paged_decode_path(
                PagedDecodeMode::ForcePagedAttention,
                DType::BFloat16,
                Some(DType::BFloat16),
                3417,
                1,
                512,
                true,
            )
            .0,
            PagedDecodePath::PagedAttention
        );
        assert_eq!(
            select_paged_decode_path(
                PagedDecodeMode::ForceSdpa,
                DType::BFloat16,
                Some(DType::BFloat16),
                131_072,
                1,
                512,
                true,
            )
            .0,
            PagedDecodePath::PagedPoolSdpa,
            "the explicit diagnostic override may exceed the automatic memory budget"
        );
    }

    #[test]
    fn cache_hit_prefill_mode_preserves_compatibility_and_explicit_routes() {
        assert_eq!(
            parse_cache_hit_prefill_mode(None, None),
            CacheHitPrefillMode::Auto
        );
        assert_eq!(
            parse_cache_hit_prefill_mode(None, Some("0")),
            CacheHitPrefillMode::ForceHostRead,
            "the former disable switch must retain its host-read diagnostic semantics"
        );
        assert_eq!(
            parse_cache_hit_prefill_mode(None, Some("1")),
            CacheHitPrefillMode::Auto
        );
        for (route, expected) in [
            ("auto", CacheHitPrefillMode::Auto),
            ("sdpa", CacheHitPrefillMode::ForceSdpa),
            ("varlen", CacheHitPrefillMode::ForceVarlen),
            ("legacy", CacheHitPrefillMode::ForceLegacy),
            ("host", CacheHitPrefillMode::ForceHostRead),
        ] {
            assert_eq!(
                parse_cache_hit_prefill_mode(Some(route), Some("0")),
                expected,
                "the explicit route must take precedence over the compatibility switch"
            );
        }
        assert_eq!(
            gemma4_paged_prefill_route_policy_for_mode(CacheHitPrefillMode::ForceSdpa),
            Gemma4PagedPrefillRoutePolicy::NonV2,
            "forced SDPA must never inherit a V2-only auxiliary cap"
        );
        assert_eq!(
            gemma4_paged_prefill_route_policy_for_mode(CacheHitPrefillMode::ForceHostRead),
            Gemma4PagedPrefillRoutePolicy::NonV2
        );
    }

    #[test]
    fn gemma4_d512_prefill_plan_is_memory_gated_and_keeps_legacy_explicit() {
        // Shipped Gemma4 global-attention geometries: 12B, 26B-A4B, 31B.
        // (8/1/512 belongs to E2B, not the 12B model.)
        for (query_heads, kv_heads, expected_sdpa, expected_varlen) in [
            (16, 1, 513_138_688, 405_012_480),
            (16, 2, 531_480_576, 405_012_480),
            (32, 4, 995_852_288, 742_916_096),
        ] {
            assert!(!mlx_sdpa_uses_fused_kernel(
                2048,
                query_heads,
                kv_heads,
                512
            ));
            assert_eq!(
                estimate_paged_pool_sdpa_bytes(2048, 4478, query_heads, kv_heads, 512, 2),
                expected_sdpa
            );
            assert_eq!(
                estimate_varlen_paged_attention_bytes(2048, 4478, query_heads, kv_heads, 512, 2),
                expected_varlen
            );
        }
        let sdpa = estimate_paged_pool_sdpa_bytes(2048, 4478, 16, 1, 512, 2);
        let varlen = estimate_varlen_paged_attention_bytes(2048, 4478, 16, 1, 512, 2);
        assert_eq!(sdpa, 513_138_688);
        assert_eq!(varlen, 405_012_480);

        assert_eq!(
            select_cache_hit_prefill_plan(
                CacheHitPrefillMode::Auto,
                2048,
                sdpa,
                varlen,
                Some(16 * 1024 * 1024 * 1024),
                true,
            )
            .path,
            CacheHitPrefillPath::PagedPoolSdpa,
            "healthy headroom should match mlx-lm's matrix-prefill compute"
        );
        assert_eq!(
            select_cache_hit_prefill_plan(
                CacheHitPrefillMode::Auto,
                2048,
                sdpa,
                varlen,
                Some(2560 * 1024 * 1024),
                true,
            )
            .path,
            CacheHitPrefillPath::PagedVarlen,
            "compact varlen must win when the unfused score matrix misses the reserve"
        );
        assert_eq!(
            select_cache_hit_prefill_plan(
                CacheHitPrefillMode::ForceLegacy,
                2048,
                sdpa,
                varlen,
                None,
                true,
            )
            .path,
            CacheHitPrefillPath::PagedLegacy
        );
        assert_eq!(
            select_cache_hit_prefill_plan(
                CacheHitPrefillMode::Auto,
                2048,
                sdpa,
                varlen,
                None,
                false,
            )
            .path,
            CacheHitPrefillPath::HostRead,
            "a non-Metal/batched caller must not enter a rank-3 paged bridge"
        );
    }

    #[test]
    fn gemma4_prefill_aux_preplan_only_selects_a_v2_layout_when_safe() {
        assert_eq!(
            gemma4_paged_prefill_v2_layout_for_chunk(
                Gemma4PagedPrefillRoutePolicy::NonV2,
                2048,
                2064,
                16,
                1,
                512,
            ),
            None,
            "a forced 2K SDPA chunk must remain at the configured compute size"
        );
        assert_eq!(
            gemma4_paged_prefill_v2_layout_for_chunk(
                Gemma4PagedPrefillRoutePolicy::Auto,
                8192,
                27_954,
                16,
                1,
                512,
            ),
            None,
            "auto must pre-plan SDPA when the varlen V2 auxiliary layout is unsafe"
        );
        assert_eq!(
            gemma4_paged_prefill_v2_layout_for_chunk(
                Gemma4PagedPrefillRoutePolicy::ForceVarlen,
                8192,
                27_954,
                16,
                1,
                512,
            ),
            Some(PagedAttentionV2Layout::Varlen),
            "an explicit varlen route remains V2-capped"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gemma4_cache_hit_prefill_operators_preserve_causal_values() -> Result<()> {
        use std::sync::{Arc, Mutex};

        use half::bf16;
        use mlx_paged_attn::{BlockAllocator, LayerKVPool, PagedAttentionConfig};

        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping Gemma4 paged-prefill operator parity without Metal");
            return Ok(());
        }

        let cfg = PagedAttentionConfig {
            block_size: 8,
            num_kv_heads: 1,
            head_size: 512,
            num_layers: 1,
            gpu_memory_mb: 256,
            use_fp8_cache: Some(false),
            max_seq_len: Some(64),
            max_batch_size: Some(2),
        };
        let pool = match LayerKVPool::new(cfg, 4, mlx_paged_attn::metal::MetalDtype::BFloat16) {
            Ok(pool) => Arc::new(pool),
            Err(err) => {
                eprintln!("skipping Gemma4 paged-prefill operator parity: {err}");
                return Ok(());
            }
        };
        let allocator = Arc::new(Mutex::new(BlockAllocator::new(4, 8)));
        let mut adapter = PagedKVCacheAdapter::new(allocator, pool, 8).expect("adapter");
        adapter.reset_for_new_request(7).unwrap();
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4]).unwrap();

        let keys = MxArray::zeros(&[4, 1, 512], Some(DType::BFloat16)).unwrap();
        let mut value_bits = Vec::with_capacity(4 * 512);
        for token_idx in 0..4 {
            value_bits.extend(std::iter::repeat_n(
                bf16::from_f32((token_idx + 1) as f32).to_bits(),
                512,
            ));
        }
        let values = MxArray::from_bfloat16(&value_bits, &[4, 1, 512]).unwrap();
        match adapter.update_keys_values_native(0, &keys, &values, 0) {
            Ok(()) => {}
            Err(err) if err.contains("Metal GPU not available") => {
                eprintln!("skipping Gemma4 paged-prefill operator parity: {err}");
                return Ok(());
            }
            Err(err) => panic!("unexpected native paged write failure: {err}"),
        }

        let queries = MxArray::zeros(&[2, 2, 512], Some(DType::BFloat16)).unwrap();
        let legacy = adapter
            .gather_kv_for_prefill_chunk(0, &queries, 2, 1.0)
            .expect("legacy paged prefill");
        let varlen = adapter
            .gather_kv_for_prefill_chunk_varlen(0, &queries, 2, 1.0)
            .expect("compact varlen prefill");
        let (gathered_keys, gathered_values) = adapter
            .gather_kv_for_prefill_sdpa(0, 4)
            .expect("graph-native paged gather");
        let queries_bhtd = queries
            .transpose(Some(&[1, 0, 2]))
            .unwrap()
            .reshape(&[1, 2, 2, 512])
            .unwrap();
        let sdpa = scaled_dot_product_attention_causal(
            &queries_bhtd,
            &gathered_keys,
            &gathered_values,
            1.0,
        )?
        .transpose(Some(&[0, 2, 1, 3]))?
        .reshape(&[2, 2, 512])?;

        let outputs = [
            ("paged_pool_sdpa", sdpa),
            ("paged_attention_varlen", varlen),
            ("paged_attention_legacy", legacy),
        ];
        for (path, output) in outputs {
            let values = output.to_float32().unwrap();
            for (token_idx, expected) in [2.0_f32, 2.5].into_iter().enumerate() {
                for head_idx in 0..2 {
                    let actual = values[(token_idx * 2 + head_idx) * 512];
                    assert!(
                        (actual - expected).abs() < 0.06,
                        "{path} token={token_idx} head={head_idx}: got {actual}, expected {expected}"
                    );
                }
            }
        }

        // Single-token decode route parity over the same physical pool. Both
        // operators consume every recorded key, so zero queries must yield the
        // same uniform average of the four value rows (2.5) for every head.
        let decode_queries = MxArray::zeros(&[1, 2, 512], Some(DType::BFloat16)).unwrap();
        let paged_decode = adapter
            .gather_kv_for_decode_graph(0, &decode_queries, 1.0, 1.0)
            .expect("graph-native paged decode");
        let (decode_keys, decode_values) = adapter
            .gather_kv_for_prefill_sdpa(0, 4)
            .expect("graph-native compact decode gather");
        let decode_queries_bhtd = decode_queries.reshape(&[1, 2, 1, 512]).unwrap();
        let sdpa_decode = scaled_dot_product_attention(
            &decode_queries_bhtd,
            &decode_keys,
            &decode_values,
            1.0,
            None,
        )?
        .reshape(&[1, 2, 512])?;
        let paged_values = paged_decode.to_float32().unwrap();
        let sdpa_values = sdpa_decode.to_float32().unwrap();
        assert_eq!(paged_values.len(), sdpa_values.len());
        for (idx, (paged, sdpa)) in paged_values.iter().zip(sdpa_values.iter()).enumerate() {
            assert!(
                (paged - sdpa).abs() < 0.06,
                "decode operator mismatch at element {idx}: paged={paged}, sdpa={sdpa}"
            );
            assert!(
                (sdpa - 2.5).abs() < 0.06,
                "decode SDPA value mismatch at element {idx}: got {sdpa}, expected 2.5"
            );
        }
        Ok(())
    }

    #[test]
    fn proportional_rope_preserves_bfloat16_input_dtype() {
        let rope = Gemma4ProportionalRoPE::new(256, 0.25, 1_000_000.0).unwrap();
        let values = vec![half::bf16::from_f32(0.25).to_bits(); 256];
        let input = MxArray::from_bfloat16(&values, &[1, 1, 1, 256]).unwrap();
        let output = rope.forward(&input, 0).unwrap();

        assert_eq!(output.dtype().unwrap(), crate::array::DType::BFloat16);
    }

    #[test]
    fn trim_mask_rejects_mask_shorter_than_kv() {
        let mask = MxArray::zeros(&[1, 1, 2, 3], None).unwrap();
        let err = match trim_mask(Some(&mask), 4) {
            Ok(_) => panic!("expected trim_mask to reject a short mask"),
            Err(err) => err,
        };
        assert!(
            err.reason.contains("mask is shorter than K/V"),
            "unexpected error: {}",
            err.reason
        );
    }

    #[test]
    fn trim_mask_trims_longer_mask_to_kv_len() {
        let mask = MxArray::zeros(&[1, 1, 2, 5], None).unwrap();
        let trimmed = trim_mask(Some(&mask), 3).unwrap().unwrap();
        assert_eq!(trimmed.shape_at(0).unwrap(), 1);
        assert_eq!(trimmed.shape_at(1).unwrap(), 1);
        assert_eq!(trimmed.shape_at(2).unwrap(), 2);
        assert_eq!(trimmed.shape_at(3).unwrap(), 3);
    }
}
