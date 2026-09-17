//! Shared paged-KV execution policy for every family driving
//! `PagedKVCacheAdapter`.
//!
//! The try-native-then-fallback ladders and their telemetry lived once per
//! attention implementation; they differ only in whether a regression is
//! observable. Kernel, mask, scale, and dtype choices stay with each family
//! (they are parity-load-bearing) — only the dispatch ladder and the
//! memory-estimate helpers that steer route selection live here.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use crate::array::{DType, MxArray};
use crate::transformer::paged_flags::{graph_decode_gather_enabled, native_kv_write_enabled};
use crate::transformer::paged_kv_cache_adapter::{
    PagedAttentionV2Layout, PagedKVCacheAdapter, PagedPrefillMemorySnapshot, SeqId,
    paged_attention_v2_aux_fits, paged_attention_v2_partition_upper_bound,
};

/// Report a synchronous paged-KV fallback once per `(family, site)` per
/// process: these paths force `eval_pending_pool_write_for_layer` plus a
/// host-side gather, so a silently-hot fallback is a real regression that
/// used to be invisible in the families that swallowed the `Err`.
pub(crate) fn warn_once_on_sync_fallback(
    family: &'static str,
    site: &'static str,
    layer_idx: u32,
    err: &str,
) {
    static REPORTED: OnceLock<Mutex<HashSet<(&'static str, &'static str)>>> = OnceLock::new();
    let reported = REPORTED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut reported = reported.lock().unwrap_or_else(|e| e.into_inner());
    if reported.insert((family, site)) {
        tracing::warn!(
            target: "mlx_core::inference",
            family,
            site,
            layer = layer_idx,
            error = %err,
            "graph-native paged path failed; using synchronous fallback (reported once per site)"
        );
    }
}

/// Write one layer's paged-layout K/V chunk: graph-native
/// `update_keys_values_native` when `MLX_PAGED_NATIVE_KV_WRITE` is on,
/// synchronous `update_keys_values` otherwise or after a native-write error
/// (a failed native write leaves the pool untouched, so this is not a
/// double-write). Callers that must know which path ran — the shared
/// transformer block couples its decode-gather gate to the write outcome —
/// keep their own inline ladder rather than forcing the result through
/// this signature.
pub(crate) fn write_kv_chunk(
    adapter: &mut PagedKVCacheAdapter,
    layer_idx: u32,
    keys_paged: &MxArray,
    values_paged: &MxArray,
    first_logical_position: u32,
    family: &'static str,
) -> Result<(), String> {
    let native_written = native_kv_write_enabled()
        && match adapter.update_keys_values_native(
            layer_idx,
            keys_paged,
            values_paged,
            first_logical_position,
        ) {
            Ok(()) => true,
            Err(err) => {
                warn_once_on_sync_fallback(family, "kv_write", layer_idx, &err);
                false
            }
        };
    if native_written {
        return Ok(());
    }
    adapter
        .update_keys_values(layer_idx, keys_paged, values_paged, first_logical_position)
        .map(|_| ())
}

/// Gather historical K/V for one decode step: graph-native
/// `gather_kv_for_decode_graph` when `MLX_PAGED_GRAPH_DECODE_GATHER` allows,
/// synchronous `gather_kv_for_decode` otherwise or after a graph-gather
/// error.
///
/// Callers that need an extra gate (e.g. the shared transformer block
/// couples the read path to its KV write path) keep their own condition
/// rather than forcing every gate into this signature.
pub(crate) fn gather_kv_for_decode_with_fallback(
    adapter: &mut PagedKVCacheAdapter,
    layer_idx: u32,
    queries: &MxArray,
    scale: f32,
    softcap: f32,
    family: &'static str,
) -> Result<MxArray, String> {
    if graph_decode_gather_enabled() {
        match adapter.gather_kv_for_decode_graph(layer_idx, queries, scale, softcap) {
            Ok(attn) => return Ok(attn),
            Err(err) => warn_once_on_sync_fallback(family, "decode_gather", layer_idx, &err),
        }
    }
    adapter.gather_kv_for_decode(layer_idx, queries, scale, softcap)
}

/// Reverse-order unwind of the rows a partially-recorded wave already
/// committed: `activate_request` then `rollback_last_tokens(n)` per row,
/// newest first. Used by [`record_decode_wave`] and by the ragged
/// multi-token wave variants, which carry a per-row token count.
pub(crate) fn unwind_recorded_rows(
    adapter: &mut PagedKVCacheAdapter,
    recorded: &[(SeqId, u32)],
) -> Result<(), String> {
    for &(seq_id, n_tokens) in recorded.iter().rev() {
        adapter
            .activate_request(seq_id)
            .and_then(|_| adapter.rollback_last_tokens(n_tokens))
            .map_err(|e| format!("rollback for sequence {seq_id} failed: {e}"))?;
    }
    Ok(())
}

/// Phase-1 of every batched decode wave, shared by all families: reject
/// duplicate or unknown sequences, snapshot each row's pre-record write
/// position, then record every token all-or-nothing — a recording failure
/// unwinds the rows recorded so far (newest first) before the error
/// propagates, so a half-recorded wave can be rescheduled without a
/// survivor having folded a token into non-invertible recurrent state.
///
/// Returns each row's `(seq_id, position)` in input order for the forward
/// pass; the caller then runs its family-specific forward body.
///
/// Gemma4's batched decode carries the same contract over its own
/// `Gemma4KVCacheCoordinator` (KV-sharing groups, `*_all` multi-request
/// methods) rather than `PagedKVCacheAdapter` — one caller over a different
/// interface does not justify a generic shim.
pub(crate) fn record_decode_wave(
    adapter: &mut PagedKVCacheAdapter,
    rows: &[(SeqId, u32)],
    family: &'static str,
) -> Result<Vec<(SeqId, u32)>, String> {
    let mut seen = HashSet::with_capacity(rows.len());
    let mut planned_rows = Vec::with_capacity(rows.len());
    for &(seq_id, _) in rows {
        if !seen.insert(seq_id) {
            return Err(format!(
                "{family} batched decode received duplicate sequence {seq_id}"
            ));
        }
        let position = adapter
            .current_token_count_for(seq_id)
            .ok_or_else(|| format!("{family} batched decode received unknown sequence {seq_id}"))?;
        planned_rows.push((seq_id, position));
    }

    let mut recorded: Vec<(SeqId, u32)> = Vec::with_capacity(rows.len());
    for &(seq_id, token) in rows {
        if let Err(error) = adapter.record_token_for(seq_id, token) {
            unwind_recorded_rows(adapter, &recorded)
                .map_err(|rollback| {
                    format!(
                        "{family} batched decode failed to record sequence {seq_id}: {error}; {rollback}"
                    )
                })?;
            return Err(format!(
                "{family} batched decode failed to record sequence {seq_id}: {error}"
            ));
        }
        recorded.push((seq_id, 1));
    }
    Ok(planned_rows)
}

/// Effective dtype a gathered-K/V SDPA prefill must run at: mixed
/// query/cache precisions promote to f32, identical precisions keep theirs,
/// and `None` (no cache read yet — a cold prefill) means the caller decides.
pub(crate) fn prefill_sdpa_effective_dtype(query: DType, cache: Option<DType>) -> Option<DType> {
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

/// Fixed per-call MLX dispatch overhead used in every prefill byte estimate.
pub(crate) const PREFILL_ESTIMATE_FIXED_OVERHEAD_BYTES: u64 = 64 * 1024 * 1024;

/// Live memory headroom a cache-hit prefill route may consume, computed from
/// one [`PagedPrefillMemorySnapshot`]. `selected_bytes` is the route
/// decision input; the remaining fields are kept for family telemetry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LivePrefillHeadroom {
    pub(crate) selected_bytes: Option<u64>,
    pub(crate) allocator_available_bytes: Option<u64>,
    pub(crate) metal_available_bytes: Option<u64>,
    pub(crate) allocator_active_bytes: Option<u64>,
    pub(crate) allocator_cached_bytes: Option<u64>,
    pub(crate) allocator_limit_bytes: Option<u64>,
    pub(crate) allocator_ceiling_bytes: Option<u64>,
    pub(crate) metal_recommended_working_set_bytes: Option<u64>,
    pub(crate) metal_current_allocated_bytes: Option<u64>,
    pub(crate) paged_pool_allocated_bytes: Option<u64>,
}

/// Whether MLX's pinned dispatcher will pick a fused full-attention kernel
/// for this geometry — decides whether the score-matrix fallback estimate
/// applies. `d256_full_sdpa_available` gates the D=256 NAX path; families
/// without it (Gemma4's D=512 global attention) pass `false`.
pub(crate) fn mlx_sdpa_uses_fused_kernel(
    query_tokens: u64,
    num_query_heads: u64,
    num_kv_heads: u64,
    head_dim: u64,
    d256_full_sdpa_available: bool,
) -> bool {
    if num_kv_heads == 0 || !num_query_heads.is_multiple_of(num_kv_heads) {
        return false;
    }
    if query_tokens <= 8 {
        let supported = matches!(head_dim, 64 | 96 | 128 | 256);
        return supported && query_tokens.saturating_mul(num_query_heads / num_kv_heads) <= 32;
    }
    matches!(head_dim, 64 | 80 | 128)
        || (head_dim == 256 && query_tokens >= 1_024 && d256_full_sdpa_available)
}

/// Conservative peak for gathering one paged layer into contiguous K/V and
/// running MLX causal SDPA. The gather can transiently hold both the selected
/// block tensors and their unpacked contiguous copies, hence four K/V-sized
/// tensors. When MLX cannot use its fused kernel (including Qwen3.6-27B
/// D=256 residual chunks below 1,024 tokens, non-NAX hosts, or an explicit
/// rollback), include the materialized score matrix and fp32 output using the
/// same shape gate as MLX's Metal dispatcher.
pub(crate) fn estimate_paged_pool_sdpa_bytes(
    query_tokens: u64,
    total_context: u64,
    num_query_heads: u64,
    num_kv_heads: u64,
    head_dim: u64,
    dtype_bytes: u64,
    d256_full_sdpa_available: bool,
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
        .saturating_add(PREFILL_ESTIMATE_FIXED_OVERHEAD_BYTES);
    if mlx_sdpa_uses_fused_kernel(
        query_tokens,
        num_query_heads,
        num_kv_heads,
        head_dim,
        d256_full_sdpa_available,
    ) {
        // The D=256 NAX kernel deliberately pads ragged sequence dimensions
        // so every block can use its aligned pipeline. Those buffers coexist
        // with the original gathered K/V and Q/output until the command
        // encoder completes, so include them in the live-headroom estimate.
        if head_dim == 256 {
            let kv_padding = if total_context.is_multiple_of(32) {
                0
            } else {
                total_context
                    .div_ceil(32)
                    .saturating_mul(32)
                    .saturating_mul(num_kv_heads)
                    .saturating_mul(head_dim)
                    .saturating_mul(dtype_bytes)
                    .saturating_mul(2)
            };
            let query_padding = if query_tokens.is_multiple_of(64) {
                0
            } else {
                query_tokens
                    .div_ceil(64)
                    .saturating_mul(64)
                    .saturating_mul(num_query_heads)
                    .saturating_mul(head_dim)
                    .saturating_mul(dtype_bytes)
                    .saturating_mul(2)
            };
            return gathered
                .saturating_add(kv_padding)
                .saturating_add(query_padding);
        }
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

/// Peak auxiliary storage used by the varlen paged kernel. Above one 512-token
/// partition, V2 keeps per-query/head/partition softmax state and a partial
/// head-sized output. For long multi-token chunks this can be larger than the
/// contiguous K/V needed by fused SDPA. For unfused head_dim=256 SDPA it is
/// still the O(L) safety path when the faster score-matrix route will not fit.
pub(crate) fn estimate_varlen_paged_attention_bytes(
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
        return output.saturating_add(PREFILL_ESTIMATE_FIXED_OVERHEAD_BYTES);
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
    // Share the layout-aware conservative partition upper bound and
    // signed-32-bit auxiliary-buffer guard with the runtime adapter.
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
        .saturating_add(PREFILL_ESTIMATE_FIXED_OVERHEAD_BYTES)
}

pub(crate) fn select_live_prefill_headroom(
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

pub(crate) fn live_prefill_headroom(snapshot: PagedPrefillMemorySnapshot) -> LivePrefillHeadroom {
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
            // MLX's cache is reclaimable at its GC threshold. Add back only
            // bytes known to be part of the device's current allocation.
            let reclaimable_cache = snapshot.allocator_cached_bytes.unwrap_or(0).min(current);
            recommended.saturating_sub(current.saturating_sub(reclaimable_cache))
        });

    if metal_available_bytes.is_none() {
        // A missing Metal snapshot should be rare once a paged adapter exists.
        // Keep the allocator fallback conservative by subtracting the known
        // external K/V pool that MLX active-memory accounting omits.
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
        allocator_active_bytes: snapshot.allocator_active_bytes,
        allocator_cached_bytes: snapshot.allocator_cached_bytes,
        allocator_limit_bytes: snapshot.allocator_limit_bytes,
        allocator_ceiling_bytes,
        metal_recommended_working_set_bytes: snapshot.metal_recommended_working_set_bytes,
        metal_current_allocated_bytes: snapshot.metal_current_allocated_bytes,
        paged_pool_allocated_bytes: snapshot.paged_pool_allocated_bytes,
    }
}
