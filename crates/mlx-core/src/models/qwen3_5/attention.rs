use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::array::{DType, MxArray};
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::models::paddleocr_vl::language::{
    MultimodalRoPE, apply_interleaved_rotary, apply_multimodal_rotary_pos_emb_interleaved,
    select_interleaved_cos_sin,
};
use crate::nn::{Activations, Linear, RMSNorm, RoPE};
use crate::transformer::KVCache;
use crate::transformer::paged_kv_cache_adapter::{
    PagedAttentionV2Layout, PagedKVCacheAdapter, PagedPrefillMemorySnapshot,
    paged_attention_v2_aux_fits, paged_attention_v2_partition_upper_bound,
};
use napi::bindgen_prelude::*;

use super::config::Qwen3_5Config;
use super::quantized_linear::{LinearProj, QuantizedLinear};

/// Qwen3.5 full attention with gating and partial RoPE.
///
/// Key differences from standard Qwen3 attention:
/// 1. q_proj outputs 2x width → split into queries + gate
/// 2. Partial RoPE: only rotates `head_dim * partial_rotary_factor` dimensions
/// 3. Output is gated: `o_proj(sdpa_output * sigmoid(gate))`
pub struct Qwen3_5Attention {
    q_proj: LinearProj, // hidden → num_heads * head_dim * 2 (queries + gate)
    k_proj: LinearProj, // hidden → num_kv_heads * head_dim
    v_proj: LinearProj, // hidden → num_kv_heads * head_dim
    o_proj: LinearProj, // num_heads * head_dim → hidden

    q_norm: RMSNorm, // [head_dim]
    k_norm: RMSNorm, // [head_dim]

    rope: RoPE,
    /// Optional M-RoPE for VLM mode (3D position encoding: temporal, height, width)
    mrope: Option<MultimodalRoPE>,

    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,

    /// Pre-transposed, OUTPUT-reordered `[hidden, 2*num_heads*head_dim]`
    /// q_proj weight: block order `[Q_h0..Q_h{H-1}, G_h0..G_h{H-1}]` instead
    /// of the checkpoint's per-head-interleaved order
    /// `[Q_h0,G_h0,Q_h1,G_h1,...]`. Populated once by
    /// `finalize_q_gate_block()` after `q_proj` is loaded; invalidated back
    /// to `None` by any `q_proj` setter.
    ///
    /// When present, `project_q_gate` slices queries/gate as two flat,
    /// row-contiguous halves of one matmul output instead of reshaping to
    /// `[B,T,H,2D]` and slicing per head. The per-head split's `gate` slice
    /// has a `2*head_dim` stride between heads, so `reshape([B,T,H*D])`
    /// fails MLX's `prepare_reshape` free-view check and dispatches a real
    /// strided `copy_gpu_inplace` (`CopyType::General`) Metal kernel on
    /// every call — this cache makes that copy a one-time load-time cost
    /// instead of a per-forward one. `None` (falls back to the unfused
    /// per-head path) when `q_proj` is quantized, mirroring
    /// `GatedDeltaNet::finalize_in_proj`.
    q_gate_block_t: Option<MxArray>,
    /// Reordered `[2*num_heads*head_dim]` q_proj bias matching
    /// `q_gate_block_t`'s column order. `None` when q_proj has no bias.
    q_gate_block_bias: Option<MxArray>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CacheHitPrefillMode {
    /// Keep the paged pool authoritative and select the compute kernel from
    /// live memory headroom.
    Auto,
    /// Force compact varlen PagedAttention for cache-hit prefill.
    ForcePaged,
    /// Force graph-native pool gather + MLX causal SDPA.
    ForceSdpa,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CacheHitPrefillPath {
    PagedVarlen,
    PagedPoolSdpa,
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
    allocator_active_bytes: Option<u64>,
    allocator_cached_bytes: Option<u64>,
    allocator_limit_bytes: Option<u64>,
    allocator_ceiling_bytes: Option<u64>,
    metal_recommended_working_set_bytes: Option<u64>,
    metal_current_allocated_bytes: Option<u64>,
    paged_pool_allocated_bytes: Option<u64>,
}

const FAST_SDPA_FIXED_OVERHEAD_BYTES: u64 = 64 * 1024 * 1024;
const FAST_SDPA_HEADROOM_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
static NATIVE_KV_FALLBACK_REPORTED: AtomicBool = AtomicBool::new(false);
static DECODE_GATHER_FALLBACK_REPORTED: AtomicBool = AtomicBool::new(false);

fn parse_cache_hit_prefill_mode(value: Option<&str>) -> CacheHitPrefillMode {
    match value.map(str::trim) {
        Some(value) if crate::inference_trace::env_flag_value_enabled(value) => {
            CacheHitPrefillMode::ForcePaged
        }
        Some(_) => CacheHitPrefillMode::ForceSdpa,
        None => CacheHitPrefillMode::Auto,
    }
}

fn cache_hit_prefill_mode() -> CacheHitPrefillMode {
    static MODE: OnceLock<CacheHitPrefillMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        parse_cache_hit_prefill_mode(
            std::env::var("MLX_PAGED_PREFILL_PAGED_ATTENTION")
                .ok()
                .as_deref(),
        )
    })
}

fn should_probe_cache_hit_prefill_memory(
    mode: CacheHitPrefillMode,
    query_tokens: i64,
    graph_backend_available: bool,
) -> bool {
    mode == CacheHitPrefillMode::Auto && query_tokens > 8 && graph_backend_available
}

fn should_try_varlen_after_sdpa(mode: CacheHitPrefillMode, sdpa_constructed: bool) -> bool {
    !sdpa_constructed && mode != CacheHitPrefillMode::ForceSdpa
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

fn d256_full_sdpa_available(effective_dtype_is_float32: bool) -> bool {
    static LOW_PRECISION_AVAILABLE: OnceLock<bool> = OnceLock::new();
    static FLOAT32_AVAILABLE: OnceLock<bool> = OnceLock::new();
    let available = if effective_dtype_is_float32 {
        &FLOAT32_AVAILABLE
    } else {
        &LOW_PRECISION_AVAILABLE
    };
    *available.get_or_init(|| {
        let mut supported = false;
        let status = unsafe {
            mlx_sys::mlx_metal_d256_full_sdpa_available(effective_dtype_is_float32, &mut supported)
        };
        status == 0 && supported
    })
}

fn mlx_sdpa_uses_fused_kernel(
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
fn estimate_paged_pool_sdpa_bytes(
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
        .saturating_add(FAST_SDPA_FIXED_OVERHEAD_BYTES);
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
        return output.saturating_add(FAST_SDPA_FIXED_OVERHEAD_BYTES);
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
    let partial_output_elements = rows.saturating_mul(head_dim);
    let softmax_state = rows.saturating_mul(2).saturating_mul(4);
    let partial_output = partial_output_elements.saturating_mul(dtype_bytes);
    output
        .saturating_add(softmax_state)
        .saturating_add(partial_output)
        .saturating_add(FAST_SDPA_FIXED_OVERHEAD_BYTES)
}

fn select_cache_hit_prefill_plan(
    mode: CacheHitPrefillMode,
    query_tokens: u64,
    estimated_sdpa_bytes: u64,
    estimated_varlen_bytes: u64,
    live_headroom_bytes: Option<u64>,
) -> CacheHitPrefillPlan {
    let path = match mode {
        CacheHitPrefillMode::ForcePaged => CacheHitPrefillPath::PagedVarlen,
        CacheHitPrefillMode::ForceSdpa => CacheHitPrefillPath::PagedPoolSdpa,
        CacheHitPrefillMode::Auto => match live_headroom_bytes {
            Some(headroom) => {
                // Keep a fixed process reserve plus a 10% cushion for the
                // model's MLP/quantized-matmul transients. Multi-token SDPA is
                // the fast path (including fused D=256 full attention on NAX)
                // whenever its full transient fits. If that
                // misses the budget, prefer compact varlen paging when it fits;
                // if neither fits, choose the smaller estimated transient.
                let budget = headroom
                    .saturating_sub(FAST_SDPA_HEADROOM_RESERVE_BYTES)
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
    };
    CacheHitPrefillPlan {
        path,
        estimated_sdpa_bytes,
        estimated_varlen_bytes,
        live_headroom_bytes,
    }
}

/// Return the tightest live allocation allowance reported by Metal/MLX.
///
/// `MTLDevice.currentAllocatedSize` is process-local and includes the private
/// paged-pool buffers that bypass MLX's allocator. MLX's active/cache counters
/// distinguish live graph allocations from reclaimable cache, while its
/// effective GC ceiling is bounded by 95% of Metal's recommended working set.
/// Use the tighter of those independently useful allowances.
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

fn native_kv_write_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MLX_QWEN35_NATIVE_KV_WRITE")
            .or_else(|_| std::env::var("MLX_NATIVE_KV_WRITE"))
            .map(|value| crate::inference_trace::env_flag_value_enabled(&value))
            .unwrap_or(true)
    })
}

impl Qwen3_5Attention {
    pub fn new(config: &Qwen3_5Config) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let has_bias = config.attention_bias;

        // q_proj outputs 2x for gating: queries + gate
        let q_proj = Linear::new(
            hidden_size as u32,
            (num_heads * head_dim * 2) as u32,
            Some(has_bias),
        )?;
        let k_proj = Linear::new(
            hidden_size as u32,
            (num_kv_heads * head_dim) as u32,
            Some(has_bias),
        )?;
        let v_proj = Linear::new(
            hidden_size as u32,
            (num_kv_heads * head_dim) as u32,
            Some(has_bias),
        )?;
        let o_proj = Linear::new(
            (num_heads * head_dim) as u32,
            hidden_size as u32,
            Some(has_bias),
        )?;

        let q_norm = RMSNorm::new(head_dim as u32, Some(config.rms_norm_eps))?;
        let k_norm = RMSNorm::new(head_dim as u32, Some(config.rms_norm_eps))?;

        // Partial RoPE: only rotate a fraction of dimensions
        let rope_dims = config.rope_dims();
        let rope = RoPE::new(rope_dims, Some(false), Some(config.rope_theta), None);

        let scale = (head_dim as f32).powf(-0.5);

        Ok(Self {
            q_proj: LinearProj::Standard(q_proj),
            k_proj: LinearProj::Standard(k_proj),
            v_proj: LinearProj::Standard(v_proj),
            o_proj: LinearProj::Standard(o_proj),
            q_norm,
            k_norm,
            rope,
            mrope: None,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            q_gate_block_t: None,
            q_gate_block_bias: None,
        })
    }

    /// Project queries + gate, returning `(queries [B,T,H,D], gate
    /// [B,T,H*D])`.
    ///
    /// Fast path (`q_gate_block_t` present, i.e. `q_proj` is non-quantized
    /// and `finalize_q_gate_block()` has run): one matmul against the
    /// block-ordered weight, then two flat `slice_axis` calls — both
    /// already row-contiguous, so `queries`'s subsequent `[B,T,H,D]`
    /// reshape is a free view and `gate` needs no reshape at all.
    /// `MLX_DISABLE_QGATE_BLOCK_SPLIT=1` forces the fallback below (for
    /// same-binary A/B benchmarking), mirroring
    /// `MLX_DISABLE_E51_STACKED_GDN_IN_PROJ`.
    ///
    /// Fallback path (quantized `q_proj`, or the env override above):
    /// the original per-head reshape+slice, unchanged from before this
    /// split existed. `gate`'s reshape here pays a strided
    /// `copy_gpu_inplace` every call — see `q_gate_block_t`'s doc comment.
    fn project_q_gate(&self, x: &MxArray, batch: i64, seq_len: i64) -> Result<(MxArray, MxArray)> {
        let hd = (self.num_heads * self.head_dim) as i64;
        if let Some(w_block_t) = &self.q_gate_block_t
            && std::env::var("MLX_DISABLE_QGATE_BLOCK_SPLIT").is_err()
        {
            let flat = match &self.q_gate_block_bias {
                Some(bias) => x.addmm(bias, w_block_t, None, None)?,
                None => x.matmul(w_block_t)?,
            };
            let queries_flat = flat.slice_axis(2, 0, hd)?;
            let gate = flat.slice_axis(2, hd, 2 * hd)?;
            let queries = queries_flat.reshape(&[
                batch,
                seq_len,
                self.num_heads as i64,
                self.head_dim as i64,
            ])?;
            Ok((queries, gate))
        } else {
            // Project queries (2x width for gating)
            let q_proj_output = self.q_proj.forward(x)?;

            // Split into queries and gate PER-HEAD (not flat):
            //   reshape to [B, T, num_heads, head_dim*2]
            //   split on last axis → queries [B,T,H,D] and gate [B,T,H,D]
            let q_per_head = q_proj_output.reshape(&[
                batch,
                seq_len,
                self.num_heads as i64,
                (self.head_dim * 2) as i64,
            ])?;
            let queries = q_per_head.slice_axis(3, 0, self.head_dim as i64)?;
            let gate =
                q_per_head.slice_axis(3, self.head_dim as i64, (self.head_dim * 2) as i64)?;
            // Flatten gate for later: [B, T, H, D] → [B, T, H*D]
            let gate = gate.reshape(&[batch, seq_len, hd])?;
            Ok((queries, gate))
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input [B, T, hidden_size]
    /// * `mask` - Attention mask (causal)
    /// * `cache` - Optional KVCache for incremental generation
    /// * `position_ids` - Optional [3, B, T] M-RoPE positions for VLM mode.
    ///   When None, uses scalar offset from KVCache (standard text-only behavior).
    ///
    /// # Returns
    /// Output [B, T, hidden_size]
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
        position_ids: Option<&MxArray>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // Project queries (2x width for gating), split into per-head
        // queries [B,T,H,D] and flat gate [B,T,H*D]. See `project_q_gate`.
        let (queries, gate) = self.project_q_gate(x, batch, seq_len)?;

        // Project keys and values
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape to head format: [B, T, H, D]
        // queries already in [B, T, H, D] from per-head split above
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

        // Apply QK normalization (operates on last dim)
        let queries = self.q_norm.forward(&queries)?;
        let keys = self.k_norm.forward(&keys)?;

        // Apply RoPE: either M-RoPE (VLM) or standard scalar offset (text-only)
        let (queries, keys) = if let (Some(pos_ids), Some(mrope)) = (position_ids, &self.mrope) {
            // M-RoPE: compute cos/sin from 3D position IDs [3, B, T].
            // Qwen3.5-VL uses the INTERLEAVED (stride-3) per-frequency axis
            // selector, NOT PaddleOCR-VL's contiguous-chunk (sectioned) one.
            let (cos, sin) = mrope.forward(&queries, pos_ids)?;
            // Transpose to [B, H, T, D] for the rotary apply.
            let q_t = queries.transpose(Some(&[0, 2, 1, 3]))?;
            let k_t = keys.transpose(Some(&[0, 2, 1, 3]))?;
            let (q_out, k_out) = apply_multimodal_rotary_pos_emb_interleaved(
                &q_t,
                &k_t,
                &cos,
                &sin,
                mrope.mrope_section_arr().to_vec(),
            )?;
            // Transpose back to [B, T, H, D]
            let q_out = q_out.transpose(Some(&[0, 2, 1, 3]))?;
            let k_out = k_out.transpose(Some(&[0, 2, 1, 3]))?;
            (q_out, k_out)
        } else {
            // Standard scalar-offset RoPE (text-only path).
            //
            // `fast::rope` varies the rotation position along axis -2 of its
            // input, so it must see the [B, H, T, D] layout (token axis at
            // -2) — matching mlx-lm's `self.rope(x.transpose(0, 2, 1, 3),
            // offset)`. Applying it on [B, T, H, D] rotates along the HEAD
            // axis instead: every token in a multi-token forward gets the
            // same angle (offset + head_index), collapsing per-token
            // positions. Transpose in, rotate, transpose back (the extra
            // transposes are views; qwen3.5's partial rotary
            // (rope_dims < head_dim) takes the rope kernel's copying
            // `dims_ < D` branch either way, so the transposed input costs a
            // strided rather than vector copy — the same price mlx-lm pays).
            let offset = cache.as_ref().map_or(0, |c| c.get_offset());
            let q_t = queries.transpose(Some(&[0, 2, 1, 3]))?;
            let k_t = keys.transpose(Some(&[0, 2, 1, 3]))?;
            let q_rot = self.rope.forward(&q_t, Some(offset))?;
            let k_rot = self.rope.forward(&k_t, Some(offset))?;
            (
                q_rot.transpose(Some(&[0, 2, 1, 3]))?,
                k_rot.transpose(Some(&[0, 2, 1, 3]))?,
            )
        };

        // Transpose to [B, H, T, D] for KVCache and SDPA
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;

        // Update KV cache (expects [B, H, T, D])
        let (keys, values) = if let Some(c) = cache {
            c.update_and_fetch(&keys, &values)?
        } else {
            (keys, values)
        };

        // Scaled dot-product attention using fast kernel.
        // When no explicit mask is provided:
        //   - seq_len > 1 (prefill): use "causal" mode — MLX's fused Metal kernel handles
        //     causal masking internally without materializing an O(N²) mask array.
        //     This matches Python mlx-lm's `create_attention_mask` returning "causal".
        //   - seq_len == 1 (decode): no mask needed (single token only attends to past).
        // When an explicit mask is provided (e.g., sliding window): use it directly.
        let output = if let Some(m) = mask {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale as f64, Some(m))?
        } else if seq_len > 1 {
            scaled_dot_product_attention_causal(&queries, &keys, &values, self.scale as f64)?
        } else {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale as f64, None)?
        };

        // Transpose back: [B, H, T, D] → [B, T, H, D] → flatten to [B, T, H*D]
        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        // Apply gate: output * sigmoid(gate)
        // gate is already [B, T, H*D] from the per-head split above
        let gate_sigmoid = Activations::sigmoid(&gate)?;
        let gated_output = output.mul(&gate_sigmoid)?;

        // Output projection
        self.o_proj.forward(&gated_output)
    }

    /// Forward pass routed through the block-paged KV adapter.
    ///
    /// Mirrors [`Self::forward`] (Q-gating, partial RoPE, Q/K layernorm)
    /// but writes K/V into the paged pool instead of a flat `KVCache`
    /// and reads attention K/V back via either an explicit
    /// `read_kv_range` (cache-hit prefill) or a host-side
    /// `read_kv_range` followed by SDPA (decode). Decode uses
    /// `read_kv_range` instead of `gather_kv_for_decode` to keep BF16
    /// reduction order bit-equal to the flat path's SDPA — matches
    /// Qwen3 / Gemma4's paged decode strategy.
    ///
    /// **Caller contract** (mirrors LFM2 / Gemma4):
    /// 1. `adapter.record_tokens(&[...suffix])` BEFORE this call so the
    ///    adapter cursor is advanced by the chunk; `update_keys_values`
    ///    enforces alignment.
    /// 2. `attn_layer_idx` is the FULL-ATTENTION ORDINAL into the
    ///    adapter pool (NOT the absolute decoder index). Pool was sized
    ///    by `Qwen3_5Config::full_attention_layer_count()`.
    /// 3. RoPE selection mirrors [`Self::forward`]: when `position_ids`
    ///    is `Some` and this is a VLM checkpoint (`self.mrope` set), apply
    ///    3-row M-RoPE over those positions (the image-bearing prefill
    ///    path); otherwise use standard scalar-offset `self.rope` from
    ///    `first_logical_position` (the text-only path). The text-only
    ///    `position_ids = None` branch is byte-identical to the flat path's
    ///    `position_ids = None` behaviour.
    ///
    /// Returns `[B, T, hidden_size]` (post-output-projection,
    /// post-gate) so the layer's residual `h = x + r` matches the flat
    /// path.
    /// `mrope_cache` is a per-forward-pass scratch slot for the M-RoPE arm:
    /// every full-attention layer in one Qwen3.5-VL forward pass shares
    /// byte-identical `position_ids`/`mrope_section`/dtype
    /// (`init_mrope_layers` seeds every layer from the same config), so the
    /// FIRST layer to see `Some(position_ids)` computes the selected cos/sin
    /// and stores it here; every later layer in the same forward pass reuses
    /// it instead of recomputing the cos/sin table + `take_along_axis`
    /// gather. Callers outside the per-layer VLM prefill loop (decode / MTP
    /// steps, which always pass `position_ids = None`) can pass `&mut None`
    /// — it is never touched on that path.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_paged(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        attn_layer_idx: u32,
        first_logical_position: u32,
        cached_prefix_len: u32,
        is_prefill: bool,
        position_ids: Option<&MxArray>,
        rope_position_offset: i32,
        mrope_cache: &mut Option<(MxArray, MxArray)>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // Project queries (2x width for gating), split into per-head
        // queries / flat gate (matches forward(); see `project_q_gate`).
        let (queries, gate) = self.project_q_gate(x, batch, seq_len)?;

        // K/V projections + reshape to per-head layout.
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;
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

        // QK normalization on the last dim.
        let queries = self.q_norm.forward(&queries)?;
        let keys = self.k_norm.forward(&keys)?;

        // RoPE: 3-row M-RoPE over `position_ids` for image-bearing prefill,
        // standard scalar offset otherwise. The M-RoPE arm reproduces the flat
        // path's layout and transpose order exactly ([B,T,H,D] -> [B,H,T,D] ->
        // rotate -> [B,T,H,D]) so the rotation is bf16-bit-identical to flat;
        // the `None` (text-only) arm matches the flat path's scalar-offset
        // behaviour.
        let (queries, keys) = if let (Some(pos_ids), Some(mrope)) = (position_ids, &self.mrope) {
            // Qwen3.5-VL uses the INTERLEAVED (stride-3) per-frequency axis
            // selector, NOT PaddleOCR-VL's contiguous-chunk (sectioned) one.
            //
            // Every full-attention layer in one forward pass shares
            // byte-identical `pos_ids`/`mrope_section`/dtype, so the cos/sin
            // table build + axis-selector gather only needs to run once per
            // forward pass (see `mrope_cache`'s doc comment above), not once
            // per full-attention layer.
            let (cos_final, sin_final) = match mrope_cache {
                Some(cached) => cached.clone(),
                None => {
                    let (cos, sin) = mrope.forward(&queries, pos_ids)?;
                    let selected =
                        select_interleaved_cos_sin(&cos, &sin, mrope.mrope_section_arr())?;
                    *mrope_cache = Some(selected.clone());
                    selected
                }
            };
            let q_t = queries.transpose(Some(&[0, 2, 1, 3]))?;
            let k_t = keys.transpose(Some(&[0, 2, 1, 3]))?;
            let (q_out, k_out) = apply_interleaved_rotary(&q_t, &k_t, &cos_final, &sin_final)?;
            let q_out = q_out.transpose(Some(&[0, 2, 1, 3]))?;
            let k_out = k_out.transpose(Some(&[0, 2, 1, 3]))?;
            (q_out, k_out)
        } else {
            // Scalar-offset RoPE. `rope_position_offset` decouples the
            // rotation position from the physical KV slot: a turn that
            // warm-continues an image prefill rotates at the compressed
            // M-RoPE position (physical slot + a negative cross-turn delta)
            // while K/V still writes at the physical slot below. Text turns
            // pass `rope_position_offset == first_logical_position as i32`.
            //
            // `fast::rope` varies the rotation position along axis -2 of its
            // input, so it must see the [B, H, T, D] layout (token axis at
            // -2) — matching mlx-lm and the flat `forward` above. Applying
            // it on [B, T, H, D] rotates along the HEAD axis, collapsing
            // per-token positions within any multi-token chunk.
            let rope_offset = rope_position_offset;
            let q_t = queries.transpose(Some(&[0, 2, 1, 3]))?;
            let k_t = keys.transpose(Some(&[0, 2, 1, 3]))?;
            let q_rot = self.rope.forward(&q_t, Some(rope_offset))?;
            let k_rot = self.rope.forward(&k_t, Some(rope_offset))?;
            (
                q_rot.transpose(Some(&[0, 2, 1, 3]))?,
                k_rot.transpose(Some(&[0, 2, 1, 3]))?,
            )
        };

        // Transpose to [B, H, T, D] for SDPA.
        let queries_bhtd = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys_bhtd = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values_bhtd = values.transpose(Some(&[0, 2, 1, 3]))?;

        // Paged-pool layout: `[num_tokens, num_kv_heads, head_dim]`.
        // [B, H_kv, T, D] -> [B, T, H_kv, D] -> [B*T, H_kv, D].
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

        let trace_enabled = inference_trace_enabled();
        let inference_info_enabled =
            tracing::enabled!(target: "mlx_core::inference", tracing::Level::INFO);
        let inference_debug_enabled =
            tracing::enabled!(target: "mlx_core::inference", tracing::Level::DEBUG);
        let write_trace_start = (trace_enabled || inference_debug_enabled).then(Instant::now);
        let write_info_start =
            (inference_info_enabled && is_prefill && attn_layer_idx == 0 && seq_len > 8)
                .then(Instant::now);
        let write_path = if native_kv_write_enabled() {
            match adapter.update_keys_values_native(
                attn_layer_idx,
                &keys_paged,
                &values_paged,
                first_logical_position,
            ) {
                Ok(()) => "native",
                Err(err) => {
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] qwen3.5-attn paged_kv_write_fallback \
                             layer={} first_position={} seq_len={} error={}",
                            attn_layer_idx, first_logical_position, seq_len, err
                        ));
                    }
                    if inference_info_enabled && attn_layer_idx == 0 {
                        let first_report =
                            !NATIVE_KV_FALLBACK_REPORTED.swap(true, Ordering::Relaxed);
                        if is_prefill || first_report || first_logical_position.is_multiple_of(32) {
                            tracing::warn!(
                                target: "mlx_core::inference",
                                event = "paged_kv_write_fallback",
                                layer = attn_layer_idx,
                                first_position = first_logical_position,
                                sequence_tokens = seq_len,
                                error = %err,
                                "native paged KV write failed; using legacy write path"
                            );
                        }
                    }
                    adapter
                        .update_keys_values(
                            attn_layer_idx,
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
                    attn_layer_idx,
                    &keys_paged,
                    &values_paged,
                    first_logical_position,
                )
                .map_err(napi::Error::from_reason)?;
            "legacy"
        };
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-attn paged_kv_write_done \
                 layer={} first_position={} seq_len={} path={} elapsed_ms={:.1}",
                attn_layer_idx,
                first_logical_position,
                seq_len,
                write_path,
                write_trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        if inference_debug_enabled {
            tracing::debug!(
                target: "mlx_core::inference",
                event = "paged_kv_write_done",
                layer = attn_layer_idx,
                first_position = first_logical_position,
                sequence_tokens = seq_len,
                path = write_path,
                elapsed_ms = write_trace_start.map(elapsed_ms).unwrap_or(0.0),
                "paged KV layer write completed"
            );
        }
        if inference_info_enabled && is_prefill && attn_layer_idx == 0 && seq_len > 8 {
            tracing::info!(
                target: "mlx_core::inference",
                event = "paged_kv_write_done",
                layer = attn_layer_idx,
                first_position = first_logical_position,
                sequence_tokens = seq_len,
                path = write_path,
                elapsed_ms = write_info_start.map(elapsed_ms).unwrap_or(0.0),
                "paged prefill KV write completed"
            );
        }

        // Compute attention output.
        let attn_bhtd = if is_prefill {
            if cached_prefix_len == 0 {
                // Fresh prefill: SDPA over in-flight Q/K/V with internal
                // causal mask.
                if seq_len > 1 {
                    scaled_dot_product_attention_causal(
                        &queries_bhtd,
                        &keys_bhtd,
                        &values_bhtd,
                        self.scale as f64,
                    )?
                } else {
                    scaled_dot_product_attention(
                        &queries_bhtd,
                        &keys_bhtd,
                        &values_bhtd,
                        self.scale as f64,
                        None,
                    )?
                }
            } else {
                // Cache-hit prefill keeps the paged pool authoritative. With
                // sufficient live headroom, gather this layer's blocks inside
                // the MLX graph and use MLX causal SDPA. Under
                // pressure, use compact varlen PagedAttention directly over
                // the pool. Decode remains paged regardless of this choice.
                let total_ctx = cached_prefix_len + (seq_len as u32);
                let graph_backend_available =
                    crate::engine::persistence::compiled_forward_backend_available();
                let query_dtype = queries.dtype()?;
                let effective_sdpa_dtype =
                    prefill_sdpa_effective_dtype(query_dtype, adapter.prefill_sdpa_cache_dtype());
                let dtype_bytes = match effective_sdpa_dtype {
                    Some(DType::Float16 | DType::BFloat16) => 2,
                    Some(DType::Float32) | None => 4,
                    Some(_) => 4,
                };
                let d256_full_sdpa_available = effective_sdpa_dtype
                    .map(|dtype| d256_full_sdpa_available(dtype == DType::Float32))
                    .unwrap_or(false);
                let estimated_sdpa_bytes = estimate_paged_pool_sdpa_bytes(
                    seq_len as u64,
                    total_ctx as u64,
                    self.num_heads as u64,
                    self.num_kv_heads as u64,
                    self.head_dim as u64,
                    dtype_bytes,
                    d256_full_sdpa_available,
                );
                let estimated_varlen_bytes = estimate_varlen_paged_attention_bytes(
                    seq_len as u64,
                    total_ctx as u64,
                    self.num_heads as u64,
                    self.num_kv_heads as u64,
                    self.head_dim as u64,
                    dtype_bytes,
                );
                let varlen_aux_fits = estimated_varlen_bytes != u64::MAX;
                let prefill_mode = cache_hit_prefill_mode();
                // Tiny MTP verification prefills are hard-routed to varlen,
                // and explicit overrides ignore memory heuristics. Avoid all
                // live probes on those hot paths. The adapter caches the
                // snapshot for real prefills, so every full-attention layer in
                // one chunk uses a single, consistent route decision.
                let memory_probe_performed = should_probe_cache_hit_prefill_memory(
                    prefill_mode,
                    seq_len,
                    graph_backend_available,
                );
                let live_headroom = if memory_probe_performed {
                    live_prefill_headroom(adapter.prefill_memory_snapshot())
                } else {
                    LivePrefillHeadroom::default()
                };
                let plan = select_cache_hit_prefill_plan(
                    prefill_mode,
                    seq_len as u64,
                    estimated_sdpa_bytes,
                    estimated_varlen_bytes,
                    live_headroom.selected_bytes,
                );
                let planned_path = match plan.path {
                    CacheHitPrefillPath::PagedVarlen => "paged_attention_varlen",
                    CacheHitPrefillPath::PagedPoolSdpa => "paged_pool_sdpa",
                };
                let configured_mode = match prefill_mode {
                    CacheHitPrefillMode::Auto => "auto",
                    CacheHitPrefillMode::ForcePaged => "force_paged",
                    CacheHitPrefillMode::ForceSdpa => "force_sdpa",
                };
                let report_prefill_route =
                    inference_info_enabled && attn_layer_idx == 0 && seq_len > 8;
                if report_prefill_route {
                    tracing::info!(
                        target: "mlx_core::inference",
                        event = "cache_hit_prefill_plan",
                        layer = attn_layer_idx,
                        suffix_tokens = seq_len,
                        cached_prefix_tokens = cached_prefix_len,
                        total_context_tokens = total_ctx,
                        configured_mode,
                        planned_path,
                        graph_backend_available,
                        effective_sdpa_dtype = ?effective_sdpa_dtype,
                        d256_full_sdpa_available,
                        varlen_aux_fits,
                        estimated_sdpa_mib = plan.estimated_sdpa_bytes as f64
                            / (1024.0 * 1024.0),
                        estimated_varlen_mib = plan.estimated_varlen_bytes as f64
                            / (1024.0 * 1024.0),
                        memory_probe_performed,
                        allocator_headroom_reported = live_headroom
                            .allocator_available_bytes
                            .is_some(),
                        allocator_headroom_mib = live_headroom
                            .allocator_available_bytes
                            .unwrap_or(0) as f64
                            / (1024.0 * 1024.0),
                        allocator_active_mib = live_headroom
                            .allocator_active_bytes
                            .unwrap_or(0) as f64
                            / (1024.0 * 1024.0),
                        allocator_cached_mib = live_headroom
                            .allocator_cached_bytes
                            .unwrap_or(0) as f64
                            / (1024.0 * 1024.0),
                        allocator_limit_mib = live_headroom
                            .allocator_limit_bytes
                            .unwrap_or(0) as f64
                            / (1024.0 * 1024.0),
                        allocator_ceiling_mib = live_headroom
                            .allocator_ceiling_bytes
                            .unwrap_or(0) as f64
                            / (1024.0 * 1024.0),
                        metal_headroom_reported = live_headroom.metal_available_bytes.is_some(),
                        metal_headroom_mib = live_headroom
                            .metal_available_bytes
                            .unwrap_or(0) as f64
                            / (1024.0 * 1024.0),
                        metal_recommended_working_set_mib = live_headroom
                            .metal_recommended_working_set_bytes
                            .unwrap_or(0) as f64
                            / (1024.0 * 1024.0),
                        metal_current_allocated_mib = live_headroom
                            .metal_current_allocated_bytes
                            .unwrap_or(0) as f64
                            / (1024.0 * 1024.0),
                        paged_pool_allocated_mib = live_headroom
                            .paged_pool_allocated_bytes
                            .unwrap_or(0) as f64
                            / (1024.0 * 1024.0),
                        live_headroom_reported = plan.live_headroom_bytes.is_some(),
                        live_headroom_mib = plan.live_headroom_bytes.unwrap_or(0) as f64
                            / (1024.0 * 1024.0),
                        "cache-hit prefill route selected"
                    );
                }

                let maybe_sdpa = if batch == 1
                    && graph_backend_available
                    && plan.path == CacheHitPrefillPath::PagedPoolSdpa
                {
                    let sdpa_trace_start =
                        (trace_enabled || report_prefill_route).then(Instant::now);
                    match adapter.gather_kv_for_prefill_sdpa(attn_layer_idx, total_ctx) {
                        Ok((k_full, v_full)) => match scaled_dot_product_attention_causal(
                            &queries_bhtd,
                            &k_full,
                            &v_full,
                            self.scale as f64,
                        ) {
                            Ok(attn) => {
                                if trace_enabled {
                                    write_inference_trace(format_args!(
                                        "[MLX_TRACE] qwen3.5-attn cache_hit_prefill \
                                         layer={} suffix_tokens={} cached_prefix_tokens={} total_ctx={} \
                                         path=paged_pool_sdpa estimated_sdpa_mib={:.1} \
                                         estimated_varlen_mib={:.1} \
                                         live_headroom_mib={:.1} elapsed_ms={:.1}",
                                        attn_layer_idx,
                                        seq_len,
                                        cached_prefix_len,
                                        total_ctx,
                                        plan.estimated_sdpa_bytes as f64 / (1024.0 * 1024.0),
                                        plan.estimated_varlen_bytes as f64 / (1024.0 * 1024.0),
                                        plan.live_headroom_bytes.unwrap_or(0) as f64
                                            / (1024.0 * 1024.0),
                                        sdpa_trace_start.map(elapsed_ms).unwrap_or(0.0)
                                    ));
                                }
                                if report_prefill_route {
                                    tracing::info!(
                                        target: "mlx_core::inference",
                                        event = "cache_hit_prefill_route",
                                        layer = attn_layer_idx,
                                        suffix_tokens = seq_len,
                                        cached_prefix_tokens = cached_prefix_len,
                                        total_context_tokens = total_ctx,
                                        path = "paged_pool_sdpa",
                                        elapsed_ms = sdpa_trace_start
                                            .map(elapsed_ms)
                                            .unwrap_or(0.0),
                                        "cache-hit prefill attention graph constructed"
                                    );
                                }
                                Some(attn)
                            }
                            Err(err) => {
                                if trace_enabled {
                                    write_inference_trace(format_args!(
                                        "[MLX_TRACE] qwen3.5-attn cache_hit_prefill_sdpa_construction_fallback \
                                         layer={} suffix_tokens={} cached_prefix_tokens={} total_ctx={} \
                                         stage=sdpa error={}",
                                        attn_layer_idx, seq_len, cached_prefix_len, total_ctx, err
                                    ));
                                }
                                tracing::warn!(
                                    target: "mlx_core::inference",
                                    event = "cache_hit_prefill_fallback",
                                    layer = attn_layer_idx,
                                    suffix_tokens = seq_len,
                                    cached_prefix_tokens = cached_prefix_len,
                                    total_context_tokens = total_ctx,
                                    failed_path = "paged_pool_sdpa",
                                    stage = "sdpa",
                                    error = %err,
                                    "cache-hit prefill SDPA construction failed"
                                );
                                None
                            }
                        },
                        Err(err) => {
                            if trace_enabled {
                                write_inference_trace(format_args!(
                                    "[MLX_TRACE] qwen3.5-attn cache_hit_prefill_sdpa_construction_fallback \
                                     layer={} suffix_tokens={} cached_prefix_tokens={} total_ctx={} \
                                     stage=paged_pool_gather error={}",
                                    attn_layer_idx, seq_len, cached_prefix_len, total_ctx, err
                                ));
                            }
                            tracing::warn!(
                                target: "mlx_core::inference",
                                event = "cache_hit_prefill_fallback",
                                layer = attn_layer_idx,
                                suffix_tokens = seq_len,
                                cached_prefix_tokens = cached_prefix_len,
                                total_context_tokens = total_ctx,
                                failed_path = "paged_pool_sdpa",
                                stage = "paged_pool_gather",
                                error = %err,
                                "cache-hit prefill paged-pool gather failed"
                            );
                            None
                        }
                    }
                } else {
                    None
                };

                let maybe_paged_attn = if should_try_varlen_after_sdpa(
                    prefill_mode,
                    maybe_sdpa.is_some(),
                ) && batch == 1
                    && graph_backend_available
                {
                    let paged_trace_start =
                        (trace_enabled || report_prefill_route).then(Instant::now);
                    let queries_paged =
                        queries.reshape(&[seq_len, self.num_heads as i64, self.head_dim as i64])?;
                    match adapter.gather_kv_for_prefill_chunk_varlen(
                        attn_layer_idx,
                        &queries_paged,
                        cached_prefix_len,
                        self.scale,
                    ) {
                        Ok(attn_t_h_d) => {
                            let target_dtype = x.dtype()?;
                            let attn_t_h_d = attn_t_h_d.astype(target_dtype)?;
                            let attn = attn_t_h_d.reshape(&[
                                batch,
                                seq_len,
                                self.num_heads as i64,
                                self.head_dim as i64,
                            ])?;
                            let attn = attn.transpose(Some(&[0, 2, 1, 3]))?;
                            if trace_enabled {
                                write_inference_trace(format_args!(
                                    "[MLX_TRACE] qwen3.5-attn cache_hit_prefill \
                                     layer={} suffix_tokens={} cached_prefix_tokens={} total_ctx={} \
                                     path=paged_attention_varlen bridge_ms={:.1} \
                                     estimated_sdpa_mib={:.1} estimated_varlen_mib={:.1} \
                                     live_headroom_mib={:.1}",
                                    attn_layer_idx,
                                    seq_len,
                                    cached_prefix_len,
                                    total_ctx,
                                    paged_trace_start.map(elapsed_ms).unwrap_or(0.0),
                                    plan.estimated_sdpa_bytes as f64 / (1024.0 * 1024.0),
                                    plan.estimated_varlen_bytes as f64 / (1024.0 * 1024.0),
                                    plan.live_headroom_bytes.unwrap_or(0) as f64
                                        / (1024.0 * 1024.0)
                                ));
                            }
                            if report_prefill_route {
                                tracing::info!(
                                    target: "mlx_core::inference",
                                    event = "cache_hit_prefill_route",
                                    layer = attn_layer_idx,
                                    suffix_tokens = seq_len,
                                    cached_prefix_tokens = cached_prefix_len,
                                    total_context_tokens = total_ctx,
                                    path = "paged_attention_varlen",
                                    graph_build_ms = paged_trace_start
                                        .map(elapsed_ms)
                                        .unwrap_or(0.0),
                                    "cache-hit prefill attention graph constructed"
                                );
                            }
                            Some(attn)
                        }
                        Err(err) => {
                            if trace_enabled {
                                write_inference_trace(format_args!(
                                    "[MLX_TRACE] qwen3.5-attn cache_hit_prefill_paged_construction_fallback \
                                     layer={} suffix_tokens={} cached_prefix_tokens={} total_ctx={} \
                                     error={}",
                                    attn_layer_idx, seq_len, cached_prefix_len, total_ctx, err
                                ));
                            }
                            tracing::warn!(
                                target: "mlx_core::inference",
                                event = "cache_hit_prefill_fallback",
                                layer = attn_layer_idx,
                                suffix_tokens = seq_len,
                                cached_prefix_tokens = cached_prefix_len,
                                total_context_tokens = total_ctx,
                                failed_path = "paged_attention_varlen",
                                stage = "graph_construction",
                                error = %err,
                                "cache-hit varlen PagedAttention construction failed"
                            );
                            None
                        }
                    }
                } else {
                    None
                };

                match maybe_sdpa.or(maybe_paged_attn) {
                    Some(attn) => attn,
                    None => {
                        // Last-resort graph-construction path. Metal dispatch
                        // errors surface later when the lazy graph evaluates.
                        // This synchronously reads K/V through the host, so it
                        // is intentionally never the normal route.
                        let read_trace_start =
                            (trace_enabled || inference_info_enabled).then(Instant::now);
                        let (k_full, v_full) = adapter
                            .read_kv_range(attn_layer_idx, 0, total_ctx)
                            .map_err(napi::Error::from_reason)?;
                        let read_kv_range_ms = read_trace_start.map(elapsed_ms);
                        let sdpa_trace_start =
                            (trace_enabled || inference_info_enabled).then(Instant::now);
                        let attn = scaled_dot_product_attention_causal(
                            &queries_bhtd,
                            &k_full,
                            &v_full,
                            self.scale as f64,
                        )?;
                        if trace_enabled {
                            write_inference_trace(format_args!(
                                "[MLX_TRACE] qwen3.5-attn cache_hit_prefill \
                                 layer={} suffix_tokens={} cached_prefix_tokens={} total_ctx={} \
                                 path=host_read_fallback read_kv_range_ms={:.1} \
                                 sdpa_mode=causal sdpa_graph_ms={:.1}",
                                attn_layer_idx,
                                seq_len,
                                cached_prefix_len,
                                total_ctx,
                                read_kv_range_ms.unwrap_or(0.0),
                                sdpa_trace_start.map(elapsed_ms).unwrap_or(0.0)
                            ));
                        }
                        tracing::warn!(
                            target: "mlx_core::inference",
                            event = "cache_hit_prefill_route",
                            layer = attn_layer_idx,
                            suffix_tokens = seq_len,
                            cached_prefix_tokens = cached_prefix_len,
                            total_context_tokens = total_ctx,
                            path = "host_read_fallback",
                            read_kv_range_ms = read_kv_range_ms.unwrap_or(0.0),
                            sdpa_graph_ms = sdpa_trace_start.map(elapsed_ms).unwrap_or(0.0),
                            "cache-hit prefill used synchronous host-read fallback"
                        );
                        attn
                    }
                }
            }
        } else {
            // Decode: prefer graph-native paged attention so native K/V
            // writes and attention reads remain in one MLX dependency graph.
            let queries_3d = queries_bhtd.squeeze(Some(&[2]))?.reshape(&[
                1,
                self.num_heads as i64,
                self.head_dim as i64,
            ])?;
            let gather_trace_start = (trace_enabled || inference_debug_enabled).then(Instant::now);
            let attn_3d = match adapter.gather_kv_for_decode_graph(
                attn_layer_idx,
                &queries_3d,
                self.scale,
                /* softcap */ 1.0,
            ) {
                Ok(attn_3d) => {
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] qwen3.5-attn decode_gather_done \
                             layer={} path=graph total_ctx={} elapsed_ms={:.1}",
                            attn_layer_idx,
                            adapter.current_token_count(),
                            gather_trace_start.map(elapsed_ms).unwrap_or(0.0)
                        ));
                    }
                    if inference_debug_enabled {
                        tracing::debug!(
                            target: "mlx_core::inference",
                            event = "paged_attention_gather_done",
                            layer = attn_layer_idx,
                            path = "graph",
                            context_tokens = adapter.current_token_count(),
                            elapsed_ms = gather_trace_start.map(elapsed_ms).unwrap_or(0.0),
                            "paged attention layer gather completed"
                        );
                    }
                    attn_3d
                }
                Err(err) => {
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] qwen3.5-attn decode_gather_fallback \
                             layer={} path=raw total_ctx={} error={}",
                            attn_layer_idx,
                            adapter.current_token_count(),
                            err
                        ));
                    }
                    if inference_info_enabled && attn_layer_idx == 0 {
                        let context_tokens = adapter.current_token_count();
                        let first_report =
                            !DECODE_GATHER_FALLBACK_REPORTED.swap(true, Ordering::Relaxed);
                        if first_report || context_tokens.is_multiple_of(32) {
                            tracing::warn!(
                                target: "mlx_core::inference",
                                event = "paged_attention_gather_fallback",
                                layer = attn_layer_idx,
                                context_tokens,
                                failed_path = "graph",
                                fallback_path = "raw",
                                error = %err,
                                "graph paged-attention gather failed; using raw gather"
                            );
                        }
                    }
                    let attn_3d = adapter
                        .gather_kv_for_decode(
                            attn_layer_idx,
                            &queries_3d,
                            self.scale,
                            /* softcap */ 1.0,
                        )
                        .map_err(napi::Error::from_reason)?;
                    if inference_debug_enabled {
                        tracing::debug!(
                            target: "mlx_core::inference",
                            event = "paged_attention_gather_done",
                            layer = attn_layer_idx,
                            path = "raw_fallback",
                            context_tokens = adapter.current_token_count(),
                            elapsed_ms = gather_trace_start.map(elapsed_ms).unwrap_or(0.0),
                            "paged attention layer gather completed"
                        );
                    }
                    attn_3d
                }
            };
            let target_dtype = x.dtype()?;
            let attn_3d = attn_3d.astype(target_dtype)?;
            attn_3d.reshape(&[1, self.num_heads as i64, 1, self.head_dim as i64])?
        };

        // Transpose back: [B, H, T, D] -> [B, T, H*D].
        let output = attn_bhtd.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        // Apply gate: output * sigmoid(gate).
        let gate_sigmoid = Activations::sigmoid(&gate)?;
        let gated_output = output.mul(&gate_sigmoid)?;

        // Output projection.
        self.o_proj.forward(&gated_output)
    }

    /// Initialize M-RoPE for VLM mode.
    pub fn init_mrope(
        &mut self,
        mrope_section: Vec<i32>,
        rope_theta: f64,
        max_position_embeddings: i32,
        rope_dims: i32,
    ) -> Result<()> {
        // Use rope_dims (head_dim * partial_rotary_factor), not full head_dim
        self.mrope = Some(MultimodalRoPE::new(
            rope_dims,
            max_position_embeddings,
            Some(rope_theta),
            mrope_section,
        )?);
        Ok(())
    }

    // ========== Weight accessors (standard mode) ==========

    pub fn set_q_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.q_gate_block_t = None; // invalidate block-order cache
        self.q_gate_block_bias = None;
        self.q_proj.set_weight(w, "q_proj")
    }
    pub fn set_k_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.k_proj.set_weight(w, "k_proj")
    }
    pub fn set_v_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.v_proj.set_weight(w, "v_proj")
    }
    pub fn set_o_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.o_proj.set_weight(w, "o_proj")
    }
    pub fn set_q_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.q_gate_block_t = None;
        self.q_gate_block_bias = None;
        self.q_proj.set_bias(b, "q_proj")
    }
    pub fn set_k_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.k_proj.set_bias(b, "k_proj")
    }
    pub fn set_v_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.v_proj.set_bias(b, "v_proj")
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

    /// Precompute the block-ordered `[hidden, 2*H*D]` q_proj weight (queries
    /// flat | gate flat) so `project_q_gate` can split queries vs. gate as
    /// two flat, row-contiguous slices instead of reshaping to
    /// `[B,T,H,2D]` and slicing per head. See `q_gate_block_t`'s doc
    /// comment for why the checkpoint's native per-head-interleaved column
    /// order forces a strided copy on every call.
    ///
    /// Safe to call repeatedly (idempotent). No-op when `q_proj` is
    /// quantized — mirrors `GatedDeltaNet::finalize_in_proj` (quantized
    /// checkpoints stay on the unfused path).
    pub fn finalize_q_gate_block(&mut self) -> Result<()> {
        let LinearProj::Standard(q_lin) = &self.q_proj else {
            return Ok(());
        };
        let h = self.num_heads as i64;
        let d = self.head_dim as i64;

        let weight = q_lin.get_weight(); // [2*H*D, hidden], per-head-interleaved
        let hidden = weight.shape_at(1)?;
        let w_per_head = weight.reshape(&[h, 2 * d, hidden])?;
        let w_q = w_per_head.slice_axis(1, 0, d)?.reshape(&[h * d, hidden])?;
        let w_g = w_per_head
            .slice_axis(1, d, 2 * d)?
            .reshape(&[h * d, hidden])?;
        let w_block_t = MxArray::concatenate(&w_q, &w_g, 0)?.transpose(Some(&[1, 0]))?;
        w_block_t.eval();
        self.q_gate_block_t = Some(w_block_t);

        self.q_gate_block_bias = match q_lin.get_bias() {
            Some(b) => {
                let b_per_head = b.reshape(&[h, 2 * d])?;
                let b_q = b_per_head.slice_axis(1, 0, d)?.reshape(&[h * d])?;
                let b_g = b_per_head.slice_axis(1, d, 2 * d)?.reshape(&[h * d])?;
                let b_block = MxArray::concatenate(&b_q, &b_g, 0)?;
                b_block.eval();
                Some(b_block)
            }
            None => None,
        };
        Ok(())
    }

    // ========== Quantized setters ==========

    pub fn set_quantized_q_proj(&mut self, ql: QuantizedLinear) {
        self.q_gate_block_t = None;
        self.q_gate_block_bias = None;
        self.q_proj.set_quantized(ql);
    }
    pub fn set_quantized_k_proj(&mut self, ql: QuantizedLinear) {
        self.k_proj.set_quantized(ql);
    }
    pub fn set_quantized_v_proj(&mut self, ql: QuantizedLinear) {
        self.v_proj.set_quantized(ql);
    }
    pub fn set_quantized_o_proj(&mut self, ql: QuantizedLinear) {
        self.o_proj.set_quantized(ql);
    }

    // ========== Weight getters (for training parameter extraction) ==========

    pub fn get_q_proj_weight(&self) -> MxArray {
        self.q_proj.get_weight()
    }
    pub fn get_k_proj_weight(&self) -> MxArray {
        self.k_proj.get_weight()
    }
    pub fn get_v_proj_weight(&self) -> MxArray {
        self.v_proj.get_weight()
    }
    pub fn get_o_proj_weight(&self) -> MxArray {
        self.o_proj.get_weight()
    }
    pub fn get_q_norm_weight(&self) -> MxArray {
        self.q_norm.get_weight()
    }
    pub fn get_k_norm_weight(&self) -> MxArray {
        self.k_norm.get_weight()
    }

    /// Whether any of the q/k/v/o projections hold quantized weights.
    ///
    /// Used by the dense/bf16-only MTP save path to detect a quantized MTP
    /// head (loaded from a `--q-mtp all`/`cyankiwi` checkpoint) and refuse
    /// to serialize stale dense weights (see
    /// `Qwen3_5MTPModule::has_quantized_weights`).
    pub fn is_quantized(&self) -> bool {
        self.q_proj.is_quantized()
            || self.k_proj.is_quantized()
            || self.v_proj.is_quantized()
            || self.o_proj.is_quantized()
    }

    /// The per-tensor FP8 activation scale threaded onto the q_proj quantized
    /// backend at load time. Test-only read-back seam: proves a loader carried
    /// `PerLayerQuant::input_amax` from config through to the built
    /// `QuantizedLinear` on this attention block.
    #[cfg(test)]
    pub(crate) fn q_proj_input_amax(&self) -> Option<f32> {
        self.q_proj.input_amax()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_hit_prefill_mode_preserves_explicit_override_semantics() {
        assert_eq!(
            parse_cache_hit_prefill_mode(None),
            CacheHitPrefillMode::Auto
        );
        for value in ["1", "true", " yes ", "ON"] {
            assert_eq!(
                parse_cache_hit_prefill_mode(Some(value)),
                CacheHitPrefillMode::ForcePaged,
                "{value:?} should force paged prefill"
            );
        }
        for value in ["0", "false", "off", "", "invalid"] {
            assert_eq!(
                parse_cache_hit_prefill_mode(Some(value)),
                CacheHitPrefillMode::ForceSdpa,
                "{value:?} should preserve the prior disabled-paged override"
            );
        }
    }

    #[test]
    fn force_sdpa_never_falls_through_to_varlen_paged_attention() {
        assert!(!should_try_varlen_after_sdpa(
            CacheHitPrefillMode::ForceSdpa,
            false,
        ));
        assert!(should_try_varlen_after_sdpa(
            CacheHitPrefillMode::Auto,
            false,
        ));
        assert!(should_try_varlen_after_sdpa(
            CacheHitPrefillMode::ForcePaged,
            false,
        ));
        assert!(!should_try_varlen_after_sdpa(
            CacheHitPrefillMode::Auto,
            true,
        ));
    }

    #[test]
    fn mlx_sdpa_fused_shape_gate_matches_metal_dispatcher() {
        assert!(mlx_sdpa_uses_fused_kernel(2_048, 24, 4, 128, false));
        assert!(!mlx_sdpa_uses_fused_kernel(2_048, 24, 4, 256, false));
        assert!(mlx_sdpa_uses_fused_kernel(2_048, 24, 4, 256, true));
        assert!(!mlx_sdpa_uses_fused_kernel(1_023, 24, 4, 256, true));
        assert!(mlx_sdpa_uses_fused_kernel(1_024, 24, 4, 256, true));
        assert!(mlx_sdpa_uses_fused_kernel(1, 24, 4, 256, false));
        assert!(!mlx_sdpa_uses_fused_kernel(8, 24, 4, 256, true));
        assert!(!mlx_sdpa_uses_fused_kernel(2_048, 24, 5, 128, true));
    }

    #[test]
    fn paged_prefill_accounts_for_mlx_sdpa_dtype_promotion() {
        assert_eq!(
            prefill_sdpa_effective_dtype(DType::BFloat16, Some(DType::BFloat16)),
            Some(DType::BFloat16)
        );
        assert_eq!(
            prefill_sdpa_effective_dtype(DType::Float16, Some(DType::Float16)),
            Some(DType::Float16)
        );
        assert_eq!(
            prefill_sdpa_effective_dtype(DType::Float16, Some(DType::BFloat16)),
            Some(DType::Float32)
        );
        assert_eq!(
            prefill_sdpa_effective_dtype(DType::BFloat16, Some(DType::Float32)),
            Some(DType::Float32)
        );
        assert_eq!(
            prefill_sdpa_effective_dtype(DType::BFloat16, None),
            None,
            "FP8 paged caches cannot feed graph-native SDPA without dequantization"
        );
    }

    #[test]
    fn qwen27b_long_prefill_fits_fast_sdpa_with_healthy_headroom() {
        // Exact Qwen3.6-27B attention shape from the local checkpoint:
        // 24 query heads, 4 KV heads, head_dim 256, bf16 activations.
        let estimate = estimate_paged_pool_sdpa_bytes(2_048, 64_754, 24, 4, 256, 2, false);
        let varlen_estimate = estimate_varlen_paged_attention_bytes(2_048, 64_754, 24, 4, 256, 2);
        assert_eq!(estimate, 7_063_814_144);
        assert_eq!(varlen_estimate, 3_338_272_768);
        let plan = select_cache_hit_prefill_plan(
            CacheHitPrefillMode::Auto,
            2_048,
            estimate,
            varlen_estimate,
            Some(16 * 1024 * 1024 * 1024),
        );
        assert_eq!(plan.path, CacheHitPrefillPath::PagedPoolSdpa);
    }

    #[test]
    fn qwen27b_fused_d256_estimate_drops_scores_only_at_supported_boundary() {
        let unfused = estimate_paged_pool_sdpa_bytes(2_048, 64_754, 24, 4, 256, 2, false);
        let fused = estimate_paged_pool_sdpa_bytes(2_048, 64_754, 24, 4, 256, 2, true);
        assert!(fused < unfused / 4, "fused={fused} unfused={unfused}");

        let one_kv = 64_754_u64 * 4 * 256 * 2;
        let one_query = 2_048_u64 * 24 * 256 * 2;
        let padded_kv = 64_768_u64 * 4 * 256 * 2;
        assert_eq!(
            fused,
            one_kv * 4 + one_query * 2 + FAST_SDPA_FIXED_OVERHEAD_BYTES + padded_kv * 2,
            "ragged K/V padding must remain in the fused peak estimate"
        );

        let ragged_query = estimate_paged_pool_sdpa_bytes(1_031, 4_129, 24, 4, 256, 2, true);
        let ragged_one_kv = 4_129_u64 * 4 * 256 * 2;
        let ragged_one_query = 1_031_u64 * 24 * 256 * 2;
        let ragged_padded_kv = 4_160_u64 * 4 * 256 * 2;
        let ragged_padded_query = 1_088_u64 * 24 * 256 * 2;
        assert_eq!(
            ragged_query,
            ragged_one_kv * 4
                + ragged_one_query * 2
                + FAST_SDPA_FIXED_OVERHEAD_BYTES
                + ragged_padded_kv * 2
                + ragged_padded_query * 2,
            "ragged Q and K/V padding must both remain in the fused peak estimate"
        );

        // Residual chunks below the upstream q_len=1024 routing boundary
        // still use the primitives fallback and must retain score storage.
        assert_eq!(
            estimate_paged_pool_sdpa_bytes(1_023, 64_754, 24, 4, 256, 2, true),
            estimate_paged_pool_sdpa_bytes(1_023, 64_754, 24, 4, 256, 2, false),
        );

        let varlen = estimate_varlen_paged_attention_bytes(2_048, 64_754, 24, 4, 256, 2);
        let headroom = Some(4 * 1024 * 1024 * 1024);
        assert_eq!(
            select_cache_hit_prefill_plan(
                CacheHitPrefillMode::Auto,
                2_048,
                unfused,
                varlen,
                headroom,
            )
            .path,
            CacheHitPrefillPath::PagedVarlen,
        );
        assert_eq!(
            select_cache_hit_prefill_plan(
                CacheHitPrefillMode::Auto,
                2_048,
                fused,
                varlen,
                headroom,
            )
            .path,
            CacheHitPrefillPath::PagedPoolSdpa,
        );
    }

    #[test]
    fn automatic_prefill_uses_varlen_when_sdpa_score_matrix_will_not_fit() {
        let estimate = estimate_paged_pool_sdpa_bytes(2_048, 64_754, 24, 4, 256, 2, false);
        let varlen_estimate = estimate_varlen_paged_attention_bytes(2_048, 64_754, 24, 4, 256, 2);
        assert_eq!(
            select_cache_hit_prefill_plan(
                CacheHitPrefillMode::Auto,
                2_048,
                estimate,
                varlen_estimate,
                None,
            )
            .path,
            CacheHitPrefillPath::PagedVarlen
        );
        assert_eq!(
            select_cache_hit_prefill_plan(
                CacheHitPrefillMode::Auto,
                2_048,
                estimate,
                varlen_estimate,
                Some(8 * 1024 * 1024 * 1024),
            )
            .path,
            CacheHitPrefillPath::PagedVarlen
        );

        // Decode-shaped reuse prefills stay directly on varlen paging instead
        // of gathering the full contiguous K/V for MLX's vector SDPA.
        assert_eq!(
            select_cache_hit_prefill_plan(CacheHitPrefillMode::Auto, 1, 1, 1, Some(u64::MAX),).path,
            CacheHitPrefillPath::PagedVarlen
        );
    }

    #[test]
    fn qwen27b_varlen_estimate_respects_query_layout() {
        // q_len=2, 24Q/4KV, D256, BF16 at >64K can select 1,024 stripes.
        // Aux state is 49,152 rows * (two f32 stats + 256 BF16 values),
        // plus final output and the planner's fixed 64 MiB headroom.
        assert_eq!(
            estimate_varlen_paged_attention_bytes(2, 114_688, 24, 4, 256, 2),
            92_692_480
        );
        assert_eq!(
            estimate_varlen_paged_attention_bytes(1, 114_688, 24, 4, 256, 2),
            69_916_672,
            "one varlen row uses generic 512-token partitions"
        );
    }

    #[test]
    fn automatic_prefill_rejects_varlen_beyond_metal_aux_element_limit() {
        // At 114,688 tokens the generic V2 route has 224 partitions. With
        // 24 heads and D=256, q=1,560 is the last shape whose partial-output
        // tensor fits the bridge's signed 32-bit element count.
        assert_ne!(
            estimate_varlen_paged_attention_bytes(1_560, 114_688, 24, 4, 256, 2),
            u64::MAX
        );
        assert_eq!(
            estimate_varlen_paged_attention_bytes(1_561, 114_688, 24, 4, 256, 2),
            u64::MAX
        );

        let sdpa = estimate_paged_pool_sdpa_bytes(2_048, 114_688, 24, 4, 256, 2, false);
        let varlen = estimate_varlen_paged_attention_bytes(2_048, 114_688, 24, 4, 256, 2);
        assert_eq!(varlen, u64::MAX);
        assert_eq!(
            select_cache_hit_prefill_plan(
                CacheHitPrefillMode::Auto,
                2_048,
                sdpa,
                varlen,
                Some(4 * 1024 * 1024 * 1024),
            )
            .path,
            CacheHitPrefillPath::PagedPoolSdpa,
            "auto mode must not select a varlen graph the Metal bridge rejects"
        );
        assert_eq!(
            select_cache_hit_prefill_plan(CacheHitPrefillMode::Auto, 2_048, sdpa, varlen, None,)
                .path,
            CacheHitPrefillPath::PagedPoolSdpa,
            "the no-probe planner branch must reject the same invalid graph"
        );
        assert_eq!(
            select_cache_hit_prefill_plan(
                CacheHitPrefillMode::ForcePaged,
                2_048,
                sdpa,
                varlen,
                None,
            )
            .path,
            CacheHitPrefillPath::PagedVarlen,
            "an explicit diagnostic override retains its existing semantics"
        );
    }

    #[test]
    fn automatic_prefill_chooses_smaller_transient_when_neither_path_fits() {
        let headroom = Some(2 * 1024 * 1024 * 1024);
        assert_eq!(
            select_cache_hit_prefill_plan(
                CacheHitPrefillMode::Auto,
                256,
                622_739_456,
                1_727_660_032,
                headroom,
            )
            .path,
            CacheHitPrefillPath::PagedPoolSdpa
        );
        assert_eq!(
            select_cache_hit_prefill_plan(
                CacheHitPrefillMode::Auto,
                2_048,
                7_063_814_144,
                3_338_272_768,
                headroom,
            )
            .path,
            CacheHitPrefillPath::PagedVarlen
        );
    }

    #[test]
    fn automatic_prefill_uses_smaller_estimate_without_memory_probe() {
        assert_eq!(
            select_cache_hit_prefill_plan(
                CacheHitPrefillMode::Auto,
                256,
                622_739_456,
                1_727_660_032,
                None,
            )
            .path,
            CacheHitPrefillPath::PagedPoolSdpa
        );
    }

    #[test]
    fn allocator_and_metal_headroom_are_independent_bounds() {
        assert_eq!(
            select_live_prefill_headroom(None, Some(16 * 1024 * 1024 * 1024)),
            Some(16 * 1024 * 1024 * 1024)
        );
        assert_eq!(select_live_prefill_headroom(Some(12), Some(8)), Some(8));
        assert_eq!(select_live_prefill_headroom(Some(12), None), Some(12));
        assert_eq!(select_live_prefill_headroom(None, None), None);
    }

    #[test]
    fn live_headroom_includes_external_metal_pool_and_reclaimable_cache() {
        let gib = 1024 * 1024 * 1024;
        let headroom = live_prefill_headroom(PagedPrefillMemorySnapshot {
            allocator_active_bytes: Some(20 * gib),
            allocator_cached_bytes: Some(4 * gib),
            allocator_limit_bytes: Some(120 * gib),
            metal_recommended_working_set_bytes: Some(100 * gib),
            metal_current_allocated_bytes: Some(40 * gib),
            paged_pool_allocated_bytes: Some(16 * gib),
        });

        // MLX GC ceiling: min(120 GiB, 95% of 100 GiB) - 20 GiB active.
        assert_eq!(headroom.allocator_ceiling_bytes, Some(95 * gib));
        assert_eq!(headroom.allocator_available_bytes, Some(75 * gib));
        // Metal sees the external 16 GiB pool in currentAllocatedSize; only
        // the 4 GiB MLX cache is reclaimable.
        assert_eq!(headroom.metal_available_bytes, Some(64 * gib));
        assert_eq!(headroom.selected_bytes, Some(64 * gib));
    }

    #[test]
    fn missing_metal_probe_subtracts_known_external_pool_from_allocator() {
        let gib = 1024 * 1024 * 1024;
        let headroom = live_prefill_headroom(PagedPrefillMemorySnapshot {
            allocator_active_bytes: Some(70 * gib),
            allocator_cached_bytes: Some(2 * gib),
            allocator_limit_bytes: Some(100 * gib),
            paged_pool_allocated_bytes: Some(16 * gib),
            ..PagedPrefillMemorySnapshot::default()
        });
        assert_eq!(headroom.allocator_available_bytes, Some(14 * gib));
        assert_eq!(headroom.selected_bytes, Some(14 * gib));

        let exhausted = live_prefill_headroom(PagedPrefillMemorySnapshot {
            allocator_active_bytes: Some(95 * gib),
            allocator_limit_bytes: Some(100 * gib),
            paged_pool_allocated_bytes: Some(16 * gib),
            ..PagedPrefillMemorySnapshot::default()
        });
        assert_eq!(exhausted.selected_bytes, Some(0));
    }

    #[test]
    fn memory_probe_is_skipped_for_mtp_and_explicit_routes() {
        assert!(!should_probe_cache_hit_prefill_memory(
            CacheHitPrefillMode::Auto,
            2,
            true,
        ));
        assert!(!should_probe_cache_hit_prefill_memory(
            CacheHitPrefillMode::ForcePaged,
            2_048,
            true,
        ));
        assert!(!should_probe_cache_hit_prefill_memory(
            CacheHitPrefillMode::ForceSdpa,
            2_048,
            true,
        ));
        assert!(!should_probe_cache_hit_prefill_memory(
            CacheHitPrefillMode::Auto,
            2_048,
            false,
        ));
        assert!(should_probe_cache_hit_prefill_memory(
            CacheHitPrefillMode::Auto,
            2_048,
            true,
        ));
    }

    #[test]
    fn explicit_prefill_override_wins_over_memory_heuristic() {
        assert_eq!(
            select_cache_hit_prefill_plan(
                CacheHitPrefillMode::ForcePaged,
                1,
                1,
                u64::MAX,
                Some(u64::MAX),
            )
            .path,
            CacheHitPrefillPath::PagedVarlen
        );
        assert_eq!(
            select_cache_hit_prefill_plan(CacheHitPrefillMode::ForceSdpa, 1, u64::MAX, 1, Some(1),)
                .path,
            CacheHitPrefillPath::PagedPoolSdpa
        );
    }

    fn tiny_cfg() -> Qwen3_5Config {
        Qwen3_5Config {
            vocab_size: 32,
            hidden_size: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 64,
            rms_norm_eps: 1e-6,
            head_dim: 8,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 128,
            pad_token_id: 0,
            eos_token_id: 0,
            bos_token_id: 0,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            partial_rotary_factor: 0.5,
            rope_theta: 100_000.0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: None,
            persist_paged_cache: None,
            n_mtp_layers: 0,
        }
    }

    /// RoPE token-axis regression test. Prefills one KV cache with a single
    /// 4-token forward (chunk) and another with 4 single-token forwards
    /// (stepwise), then runs the same probe token through both caches and
    /// compares the outputs.
    ///
    /// `fast::rope` varies the rotation position along axis -2 of its
    /// input. The scalar-offset arm used to rope on `[B, T, H, D]`, which
    /// rotates along the HEAD axis: in the 4-token chunk every token got
    /// the same angle (`offset + head_index`), while the stepwise path got
    /// per-token angles — so the two caches held O(1)-different keys and
    /// the probe outputs diverged (observed max_abs_diff 0.053 with this
    /// setup, vs 1.4e-4 with the fix). With the rotation on `[B, H, T, D]`
    /// the caches agree and the probe outputs match to f32-kernel noise.
    ///
    /// Everything is f32 on purpose: f32 matmuls take the non-NAX Metal
    /// path on gen-17 GPUs, so this test isolates rope-layout semantics
    /// from the half-precision NAX GEMM issues that poison bf16
    /// chunk-vs-stepwise comparisons on M5 hosts (see cleanup-G report).
    #[test]
    fn scalar_rope_rotates_along_token_axis() -> Result<()> {
        let cfg = tiny_cfg();
        let mut attn = Qwen3_5Attention::new(&cfg)?;

        let h = cfg.num_heads as i64;
        let d = cfg.head_dim as i64;
        let hidden = cfg.hidden_size as i64;
        let kv = cfg.num_kv_heads as i64;

        // Deterministic weights, scaled small so multi-layer products stay
        // O(1) in f32.
        let q_w: Vec<f32> = (0..(2 * h * d * hidden))
            .map(|i| ((i as f32) * 0.7391).sin() * 0.2)
            .collect();
        let k_w: Vec<f32> = (0..(kv * d * hidden))
            .map(|i| ((i as f32) * 0.5711 + 1.0).sin() * 0.2)
            .collect();
        let v_w: Vec<f32> = (0..(kv * d * hidden))
            .map(|i| ((i as f32) * 0.9173 + 2.0).sin() * 0.2)
            .collect();
        let o_w: Vec<f32> = (0..(hidden * h * d))
            .map(|i| ((i as f32) * 0.6133 + 3.0).sin() * 0.2)
            .collect();
        attn.set_q_proj_weight(&MxArray::from_float32(&q_w, &[2 * h * d, hidden])?)?;
        attn.set_k_proj_weight(&MxArray::from_float32(&k_w, &[kv * d, hidden])?)?;
        attn.set_v_proj_weight(&MxArray::from_float32(&v_w, &[kv * d, hidden])?)?;
        attn.set_o_proj_weight(&MxArray::from_float32(&o_w, &[hidden, h * d])?)?;

        let x_vals: Vec<f32> = (0..(4 * hidden))
            .map(|i| ((i as f32) * 0.8317).sin())
            .collect();
        let probe_vals: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32) * 0.3719 + 5.0).sin())
            .collect();
        let probe = MxArray::from_float32(&probe_vals, &[1, 1, hidden])?;

        // Chunk prefill: one 4-token forward.
        let mut cache_chunk = KVCache::new();
        let x_full = MxArray::from_float32(&x_vals, &[1, 4, hidden])?;
        let _ = attn.forward(&x_full, None, Some(&mut cache_chunk), None)?;
        assert_eq!(cache_chunk.get_offset(), 4);

        // Stepwise prefill: four 1-token forwards.
        let mut cache_step = KVCache::new();
        for t in 0..4usize {
            let x_t = MxArray::from_float32(
                &x_vals[t * hidden as usize..(t + 1) * hidden as usize],
                &[1, 1, hidden],
            )?;
            let _ = attn.forward(&x_t, None, Some(&mut cache_step), None)?;
        }
        assert_eq!(cache_step.get_offset(), 4);

        // Same probe token through both caches.
        let out_chunk = attn.forward(&probe, None, Some(&mut cache_chunk), None)?;
        let out_step = attn.forward(&probe, None, Some(&mut cache_step), None)?;

        let a = out_chunk.to_float32()?;
        let b = out_step.to_float32()?;
        assert_eq!(a.len(), b.len());
        let mut max_diff = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        // Observed ~1.4e-4 with the fix (chunk vs stepwise runs different
        // f32 GEMM/GEMV kernels and softmax reduction orders); the broken
        // head-axis rotation produced ~0.9. 1e-3 sits three orders of
        // magnitude below the failure signal.
        assert!(
            max_diff < 1e-3,
            "chunk-prefilled and stepwise-prefilled caches disagree \
             (max_abs_diff={max_diff}); scalar RoPE is not rotating along \
             the token axis"
        );
        Ok(())
    }

    /// Builds two `Qwen3_5Attention`s from byte-identical q/k/v/o weights:
    /// one with `finalize_q_gate_block()` called (block-order fast path in
    /// `project_q_gate`) and one without (the pre-fix per-head
    /// reshape+slice fallback). Asserts `forward()` produces numerically
    /// identical output on both — proving the q_proj row reorder is a pure
    /// layout change with no effect on the computed queries/gate values.
    #[test]
    fn q_gate_block_split_matches_unfused_fallback() -> Result<()> {
        let cfg = tiny_cfg();
        let mut fast = Qwen3_5Attention::new(&cfg)?;
        let mut slow = Qwen3_5Attention::new(&cfg)?;

        let h = cfg.num_heads as i64;
        let d = cfg.head_dim as i64;
        let hidden = cfg.hidden_size as i64;
        let kv = cfg.num_kv_heads as i64;

        // Deterministic, distinct-per-element weights (iota-derived) so any
        // column-reorder bug shows up as a numeric mismatch rather than
        // hiding behind a symmetric weight matrix.
        let q_w: Vec<f32> = (0..(2 * h * d * hidden))
            .map(|i| (i as f32) * 0.001)
            .collect();
        let k_w: Vec<f32> = (0..(kv * d * hidden))
            .map(|i| (i as f32) * 0.001 + 1.0)
            .collect();
        let v_w: Vec<f32> = (0..(kv * d * hidden))
            .map(|i| (i as f32) * 0.001 + 2.0)
            .collect();
        let o_w: Vec<f32> = (0..(hidden * h * d))
            .map(|i| (i as f32) * 0.001 + 3.0)
            .collect();

        let q_weight = MxArray::from_float32(&q_w, &[2 * h * d, hidden])?;
        let k_weight = MxArray::from_float32(&k_w, &[kv * d, hidden])?;
        let v_weight = MxArray::from_float32(&v_w, &[kv * d, hidden])?;
        let o_weight = MxArray::from_float32(&o_w, &[hidden, h * d])?;

        for attn in [&mut fast, &mut slow] {
            attn.set_q_proj_weight(&q_weight)?;
            attn.set_k_proj_weight(&k_weight)?;
            attn.set_v_proj_weight(&v_weight)?;
            attn.set_o_proj_weight(&o_weight)?;
        }
        fast.finalize_q_gate_block()?;
        // `slow` intentionally left un-finalized: `q_gate_block_t` stays
        // `None`, exercising the pre-fix per-head reshape+slice path.
        assert!(slow.q_gate_block_t.is_none());
        assert!(fast.q_gate_block_t.is_some());

        let x_data: Vec<f32> = (0..(2 * hidden))
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let x = MxArray::from_float32(&x_data, &[1, 2, hidden])?;

        let out_fast = fast.forward(&x, None, None, None)?;
        let out_slow = slow.forward(&x, None, None, None)?;

        let got = out_fast.to_float32()?;
        let want = out_slow.to_float32()?;
        assert_eq!(got.len(), want.len());
        // Empirically bit-identical (both paths compute the same per-column
        // dot products, just via differently-ordered matmul calls); keep a
        // tight-but-nonzero epsilon so the test isn't brittle to a future
        // MLX GEMM version choosing different tiling.
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-6,
                "mismatch at element {i}: fast={g} slow={w}"
            );
        }
        Ok(())
    }

    /// Same parity check as `q_gate_block_split_matches_unfused_fallback`,
    /// but WITH a q_proj bias loaded. This exercises the two bias-only
    /// branches the no-bias test above never reaches: the
    /// `q_lin.get_bias() => Some(b)` bias reorder in `finalize_q_gate_block`
    /// and the `Some(bias) => x.addmm(bias, ...)` fast path in
    /// `project_q_gate`. `Linear::set_bias` accepts a bias regardless of
    /// `attention_bias`, so the same tiny config is reused.
    #[test]
    fn q_gate_block_split_matches_unfused_fallback_with_bias() -> Result<()> {
        let cfg = tiny_cfg();
        let mut fast = Qwen3_5Attention::new(&cfg)?;
        let mut slow = Qwen3_5Attention::new(&cfg)?;

        let h = cfg.num_heads as i64;
        let d = cfg.head_dim as i64;
        let hidden = cfg.hidden_size as i64;
        let kv = cfg.num_kv_heads as i64;

        // Byte-identical iota-derived weights, same as the no-bias test.
        let q_w: Vec<f32> = (0..(2 * h * d * hidden))
            .map(|i| (i as f32) * 0.001)
            .collect();
        let k_w: Vec<f32> = (0..(kv * d * hidden))
            .map(|i| (i as f32) * 0.001 + 1.0)
            .collect();
        let v_w: Vec<f32> = (0..(kv * d * hidden))
            .map(|i| (i as f32) * 0.001 + 2.0)
            .collect();
        let o_w: Vec<f32> = (0..(hidden * h * d))
            .map(|i| (i as f32) * 0.001 + 3.0)
            .collect();
        // Distinct iota-derived q_proj bias, per-head-interleaved `[2*H*D]`
        // to match the checkpoint column order `finalize_q_gate_block`
        // reorders. Nonzero + distinct-per-element so a bias-reorder bug
        // surfaces as a numeric mismatch.
        let q_b: Vec<f32> = (0..(2 * h * d)).map(|i| (i as f32) * 0.01 + 4.0).collect();

        let q_weight = MxArray::from_float32(&q_w, &[2 * h * d, hidden])?;
        let k_weight = MxArray::from_float32(&k_w, &[kv * d, hidden])?;
        let v_weight = MxArray::from_float32(&v_w, &[kv * d, hidden])?;
        let o_weight = MxArray::from_float32(&o_w, &[hidden, h * d])?;
        let q_bias = MxArray::from_float32(&q_b, &[2 * h * d])?;

        for attn in [&mut fast, &mut slow] {
            attn.set_q_proj_weight(&q_weight)?;
            attn.set_k_proj_weight(&k_weight)?;
            attn.set_v_proj_weight(&v_weight)?;
            attn.set_o_proj_weight(&o_weight)?;
            // Load the q_proj bias BEFORE finalize: every q_proj setter
            // invalidates the block cache to `None`, so `finalize` must run
            // last to snapshot both the weight and the bias (matches the
            // production load order).
            attn.set_q_proj_bias(Some(&q_bias))?;
        }
        fast.finalize_q_gate_block()?;
        // `slow` intentionally left un-finalized: exercises the per-head
        // reshape+slice fallback (with `Linear::forward`'s own bias add).
        assert!(slow.q_gate_block_t.is_none());
        assert!(fast.q_gate_block_t.is_some());
        // Proves the bias-reorder branch actually ran (vs. silently taking
        // the `None` arm): `q_gate_block_bias` is populated only when
        // `q_proj` has a bias to reorder.
        assert!(
            fast.q_gate_block_bias.is_some(),
            "finalize_q_gate_block should have reordered the q_proj bias"
        );

        let x_data: Vec<f32> = (0..(2 * hidden))
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let x = MxArray::from_float32(&x_data, &[1, 2, hidden])?;

        let out_fast = fast.forward(&x, None, None, None)?;
        let out_slow = slow.forward(&x, None, None, None)?;

        let got = out_fast.to_float32()?;
        let want = out_slow.to_float32()?;
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-6,
                "mismatch at element {i}: fast={g} slow={w}"
            );
        }
        Ok(())
    }
}
