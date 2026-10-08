use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::array::kv_int8::{Int8KvRows, segmented_sdpa_int8};
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
use crate::transformer::KvFormat;
use crate::transformer::paged_flags::{graph_decode_gather_enabled, native_kv_write_enabled};
#[cfg(test)]
use crate::transformer::paged_kv_cache_adapter::PagedPrefillMemorySnapshot;
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};
use crate::transformer::paged_policy::{
    LivePrefillHeadroom, estimate_paged_pool_sdpa_bytes_with_portable,
    estimate_varlen_paged_attention_bytes, live_prefill_headroom, prefill_sdpa_effective_dtype,
};
#[cfg(test)]
use crate::transformer::paged_policy::{
    PREFILL_ESTIMATE_FIXED_OVERHEAD_BYTES, estimate_paged_pool_sdpa_bytes,
    mlx_sdpa_uses_fused_kernel, select_live_prefill_headroom,
};
use napi::bindgen_prelude::*;

use super::config::Qwen3_5Config;
use crate::models::quantized_linear::{LinearProj, QuantizedLinear};

/// Widest segmented SDPA query chunk this device supports for `gqa`; 0 when
/// segmented SDPA is not supported here. Errors when its prebuilt kernels are
/// missing from `paged_attn.metallib` (a packaging error, details on stderr),
/// so a stale metallib cannot silently fall back to concatenated K/V.
fn segmented_max_query_length(gqa: i64) -> Result<i64> {
    let max_q = unsafe { mlx_sys::mlx_segmented_sdpa_max_query_length(gqa as i32) };
    if max_q < 0 {
        return Err(Error::from_reason(
            "segmented SDPA kernels are missing from paged_attn.metallib (stale or incomplete \
             metallib; rebuild it with `yarn build:native`)",
        ));
    }
    Ok(i64::from(max_q))
}

fn segmented_verify_sdpa(
    queries: &MxArray,
    prefix_keys: &MxArray,
    prefix_values: &MxArray,
    new_keys: &MxArray,
    new_values: &MxArray,
    scale: f32,
    causal: bool,
) -> Result<MxArray> {
    let handle = unsafe {
        mlx_sys::mlx_segmented_sdpa_forward(
            queries.as_raw_ptr(),
            prefix_keys.as_raw_ptr(),
            prefix_values.as_raw_ptr(),
            new_keys.as_raw_ptr(),
            new_values.as_raw_ptr(),
            scale,
            causal,
        )
    };
    MxArray::from_handle(handle, "segmented_verify_sdpa")
}

fn verify_sdpa_without_kv_concat(
    queries: &MxArray,
    prefix_keys: &MxArray,
    prefix_values: &MxArray,
    new_keys: &MxArray,
    new_values: &MxArray,
    scale: f32,
    causal: bool,
) -> Result<MxArray> {
    let q_heads = queries.shape_at(1)?;
    let kv_heads = prefix_keys.shape_at(1)?;
    let gqa = if kv_heads > 0 { q_heads / kv_heads } else { 0 };
    let fallback = || {
        let keys = MxArray::concatenate(prefix_keys, new_keys, 2)?;
        let values = MxArray::concatenate(prefix_values, new_values, 2)?;
        if causal {
            scaled_dot_product_attention_causal(queries, &keys, &values, scale as f64)
        } else {
            scaled_dot_product_attention(queries, &keys, &values, scale as f64, None)
        }
    };
    // Empty prefixes have no buffer to bind. They are uncommon after the
    // first verify cycle and retain the established contiguous path.
    if !unsafe { mlx_sys::mlx_metal_is_available() }
        || unsafe { mlx_sys::mlx_default_device() } != 1
        || prefix_keys.shape_at(2)? == 0
        || queries.shape_at(3)? != 256
        || queries.dtype()? != DType::BFloat16
        || kv_heads < 1
        || q_heads < 1
        || q_heads % kv_heads != 0
        || q_heads / kv_heads > 32
    {
        return fallback();
    }
    let max_q = segmented_max_query_length(gqa)?;
    match segmented_verify_head_len(queries.shape_at(2)?, max_q) {
        Some(0) => {}
        Some(_) if causal => {}
        _ => return fallback(),
    }
    segmented_verify_sdpa(
        queries,
        prefix_keys,
        prefix_values,
        new_keys,
        new_values,
        scale,
        causal,
    )
}

/// Rows of the leading chunk when `seq_len` exceeds the widest supported
/// segmented query chunk (`Some(0)`: one chunk). `None` when two chunks
/// cannot cover the block. The segmented primitive serves such a block in one
/// call and splits internally only where the chunks' reductions differ.
fn segmented_verify_head_len(seq_len: i64, max_q: i64) -> Option<i64> {
    if max_q < 1 || !(1..=8).contains(&seq_len) {
        return None;
    }
    if seq_len <= max_q {
        return Some(0);
    }
    let head = seq_len - max_q.min(seq_len - 1);
    (head <= max_q).then_some(head)
}

fn verify_shapeless_geometry(seq_len: i64, gqa: i64, head_dim: i32, device_max_q: i64) -> bool {
    if !(1..=8).contains(&seq_len)
        || !(1..=32).contains(&gqa)
        || !matches!(head_dim, 64 | 96 | 128 | 256)
    {
        return false;
    }
    // This is the pinned vector kernel's reduction contract, not a device
    // performance threshold. A detected segmented limit may narrow a chunk.
    let vector_max_q = 32 / gqa;
    if seq_len <= vector_max_q {
        return true;
    }
    let max_q = if device_max_q > 0 {
        device_max_q.min(vector_max_q)
    } else {
        vector_max_q
    };
    let tail = max_q.min(seq_len - 1);
    // Use the smaller bound for both dtype routes: BF16 may use segmented
    // attention; other types use the ordinary vector partition.
    tail >= 1 && seq_len - tail <= vector_max_q
}

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
    is_prism_model: bool,

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
    /// instead of a per-forward one. Affine and native GGUF K/IQ q_proj store
    /// the same block order directly in their packed operands instead of using
    /// this duplicate dense cache; other quantization modes retain the fallback.
    q_gate_block_t: Option<MxArray>,
    /// Reordered `[2*num_heads*head_dim]` q_proj bias matching
    /// `q_gate_block_t`'s column order. `None` when q_proj has no bias.
    q_gate_block_bias: Option<MxArray>,

    /// Packed row-merged `[w_k; w_v]` quantized projection plus the k-row
    /// split point, installed by `finalize_kv_proj()` when both projections
    /// are mergeable quantized linears. `k_proj`/`v_proj` are then swapped
    /// for zero-copy row-slice views (getters/set_weight unaffected), so the
    /// merged buffer is the only resident copy.
    kv_proj: Option<(LinearProj, i64)>,
}

/// The K/V prefix a compiled verify forward reads: views into the flat
/// `KVCache` buffers in the cache's format.
pub(crate) enum VerifyPrefix<'a> {
    /// BF16 `[B, Hkv, P, D]` keys and values.
    Bf16 {
        keys: &'a MxArray,
        values: &'a MxArray,
    },
    /// int8 `[B, Hkv, P, D]` keys and values with their fp32 `[B, Hkv, P]`
    /// scales (`KVCache::int8_view`).
    Int8 {
        keys: &'a MxArray,
        values: &'a MxArray,
        key_scales: &'a MxArray,
        value_scales: &'a MxArray,
    },
}

/// The post-RoPE K/V block a compiled verify forward emits, in the layout the
/// cache stores: BF16 rows, or int8 rows plus scales (quantized inside the
/// graph, so the verify attention reads the new rows in the same format as
/// the prefix and the cache write is a plain slice assignment).
pub(crate) enum VerifyKvOut {
    Bf16(MxArray, MxArray),
    Int8(Int8KvRows),
}

/// IO bundle for [`Qwen3_5Attention::forward_verify`] (the compiled DFlash2
/// verify path): the graph reads position and the K/V prefix as array inputs
/// and returns the post-RoPE block through `out_kv`, so nothing host-baked —
/// cache offsets, prefix lengths, write bounds — enters the traced region.
pub(crate) struct AttentionVerifyIo<'a> {
    /// Live K/V prefix views into the flat KVCache buffers.
    pub prefix: VerifyPrefix<'a>,
    /// Per-batch RoPE position of this block's first row, `[B] int32`.
    pub rope_offsets: &'a MxArray,
    /// Receives the post-RoPE `[B, Hkv, T, D]` block in the cache's format —
    /// what `KVCache::update_and_fetch` / `append_quantized` store.
    pub out_kv: &'a mut Option<VerifyKvOut>,
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

impl Qwen3_5Attention {
    /// Unfused causal SDPA constructs its mask from host-side prefix lengths.
    /// Such a trace cannot be reused as the prefix grows. Keep
    /// shapeless replay only when every query chunk uses the fused primitive.
    pub(crate) fn verify_can_be_shapeless(&self, seq_len: i64) -> Result<bool> {
        if !unsafe { mlx_sys::mlx_metal_is_available() }
            || unsafe { mlx_sys::mlx_default_device() } != 1
            || self.num_kv_heads <= 0
            || self.num_heads % self.num_kv_heads != 0
        {
            return Ok(false);
        }
        let gqa = i64::from(self.num_heads / self.num_kv_heads);
        let device_max_q = if self.head_dim == 256 {
            segmented_max_query_length(gqa)?
        } else {
            0
        };
        Ok(verify_shapeless_geometry(
            seq_len,
            gqa,
            self.head_dim,
            device_max_q,
        ))
    }

    pub(super) fn paged_attention_operand(
        &self,
        x: &MxArray,
        cache_dtype: Option<DType>,
    ) -> Result<MxArray> {
        if (self.is_prism_model || self.q_proj.has_hadamard()) && x.dtype()? == DType::Float32 {
            let dtype = cache_dtype
                .filter(|dtype| matches!(dtype, DType::Float16 | DType::BFloat16))
                .ok_or_else(|| {
                    Error::from_reason("prism_hadamard paged attention requires a 16-bit KV cache")
                })?;
            x.astype(dtype)
        } else {
            Ok(x.clone())
        }
    }

    pub(super) fn set_prism_model(&mut self, enabled: bool) {
        self.is_prism_model = enabled;
    }

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
            is_prism_model: false,
            q_gate_block_t: None,
            q_gate_block_bias: None,
            kv_proj: None,
        })
    }

    /// Project queries + gate, returning `(queries [B,T,H,D], gate
    /// [B,T,H*D])`.
    ///
    /// Fast path (`finalize_q_gate_block()` installed either the dense cache or
    /// a packed quantized block layout): one projection followed by a single
    /// `split_sections` — zero-copy views instead of two `slice_axis` copy
    /// dispatches. Queries' `[B,T,H,D]` reshape stays a strided view and
    /// materializes only if a downstream kernel requires contiguous input;
    /// gate feeds elementwise ops directly.
    /// Fallback path (unsupported packed formats or an unfinalized projection):
    /// the original per-head reshape+slice, unchanged from before this
    /// split existed. `gate`'s reshape here pays a strided
    /// `copy_gpu_inplace` every call — see `q_gate_block_t`'s doc comment.
    fn project_q_gate(&self, x: &MxArray, batch: i64, seq_len: i64) -> Result<(MxArray, MxArray)> {
        let hd = (self.num_heads * self.head_dim) as i64;
        if self.q_gate_block_t.is_some() || self.q_proj.has_q_gate_block_layout() {
            let flat = match (&self.q_gate_block_t, &self.q_gate_block_bias) {
                (Some(w_block_t), Some(bias)) => x.addmm(bias, w_block_t, None, None)?,
                (Some(w_block_t), None) => x.matmul(w_block_t)?,
                (None, _) => self.q_proj.forward(x)?,
            };
            let qg = flat.split_sections(&[hd], 2)?;
            let queries_flat = &qg[0];
            let gate = qg[1].clone();
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

    /// Project keys and values. When `finalize_kv_proj` merged the two
    /// quantized projections this is ONE packed matmul plus two axis-2 slices;
    /// otherwise the original two matmuls for incompatible projections.
    fn project_kv(&self, x: &MxArray) -> Result<(MxArray, MxArray)> {
        if let Some((merged, k_rows)) = &self.kv_proj {
            let kv = merged.forward(x)?; // [B, T, k_dim + v_dim]
            let last = kv.ndim()? as usize - 1;
            let kv_split = kv.split_sections(&[*k_rows], last as i32)?;
            return Ok((kv_split[0].clone(), kv_split[1].clone()));
        }
        Ok((self.k_proj.forward(x)?, self.v_proj.forward(x)?))
    }

    /// Merge `k_proj` + `v_proj` into one packed quantized projection when both
    /// are merge-compatible (`LinearProj::concat_rows`). The originals are
    /// replaced by row-slice views of the merged buffer, so per-projection
    /// getters and the unfused forwards stay correct. Dense or mixed-format
    /// pairs keep the two-matmul path. Idempotent.
    pub fn finalize_kv_proj(&mut self) -> Result<()> {
        if self.kv_proj.is_some() {
            return Ok(());
        }
        if let Some(merged) = self.k_proj.concat_rows(&self.v_proj)? {
            let k_rows = self.k_proj.packed_out_features()?;
            let v_rows = self.v_proj.packed_out_features()?;
            // Preserve calibration keys for consumers of individual views.
            let k_key = self.k_proj.amax_key().map(str::to_owned);
            let v_key = self.v_proj.amax_key().map(str::to_owned);
            self.k_proj = merged.slice_rows(0, k_rows)?.with_amax_key(k_key);
            self.v_proj = merged
                .slice_rows(k_rows, k_rows + v_rows)?
                .with_amax_key(v_key);
            self.kv_proj = Some((merged, k_rows));
        }
        Ok(())
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

        // Project keys and values (one merged matmul when `finalize_kv_proj`
        // packed the two quantized projections into `kv_proj`).
        let (keys, values) = self.project_kv(x)?;

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

        // An int8 cache quantizes the block on write and reads the int8 rows
        // through its own kernels; see `forward_int8_cached`.
        if cache
            .as_deref()
            .is_some_and(|c| c.format() == KvFormat::Int8)
        {
            let c = cache.ok_or_else(|| Error::from_reason("int8 KV cache vanished"))?;
            let output = self.forward_int8_cached(&queries, &keys, &values, mask, c, seq_len)?;
            let output = output.transpose(Some(&[0, 2, 1, 3]))?;
            let output =
                output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;
            let gated_output = Activations::sigmoid_mul_compiled(&gate, &output)?;
            return self.o_proj.forward(&gated_output);
        }

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
            // The fused vector kernel launches 32×gqa×qL threads in its
            // 2-pass variant — qL·gqa > 32 exceeds the threadgroup limit and
            // MLX falls back to a ~15-op unfused graph (expand + scores matmul
            // + arange mask + where + softmax + matmul). Speculative verify
            // blocks land just over the bound (e.g. qL=7, gqa=6 → 42).
            // Splitting the query block keeps each call inside the fused
            // kernel: causal alignment follows from qL_off = kL − qL, so the
            // head chunk runs against keys truncated to kL − tail while the
            // tail chunk sees the full cache.
            let gqa = (self.num_heads / self.num_kv_heads.max(1)) as i64;
            let vector_dims = matches!(self.head_dim, 64 | 96 | 128 | 256);
            let tail = if (1..=32).contains(&gqa) {
                (32 / gqa).min(seq_len - 1)
            } else {
                0
            };
            if vector_dims && tail >= 1 && seq_len <= 8 && seq_len * gqa > 32 {
                let head_len = seq_len - tail;
                let kv_len = keys.shape_at(2)?;
                let q_parts = queries.split_sections(&[head_len], 2)?;
                let kv_split = keys.split_sections(&[kv_len - tail], 2)?;
                let vv_split = values.split_sections(&[kv_len - tail], 2)?;
                let out_head = if head_len > 1 {
                    scaled_dot_product_attention_causal(
                        &q_parts[0],
                        &kv_split[0],
                        &vv_split[0],
                        self.scale as f64,
                    )?
                } else {
                    // Single-row head: causal is a no-op, it attends to the
                    // whole truncated prefix.
                    scaled_dot_product_attention(
                        &q_parts[0],
                        &kv_split[0],
                        &vv_split[0],
                        self.scale as f64,
                        None,
                    )?
                };
                let out_tail = scaled_dot_product_attention_causal(
                    &q_parts[1],
                    &keys,
                    &values,
                    self.scale as f64,
                )?;
                MxArray::concatenate(&out_head, &out_tail, 2)?
            } else {
                scaled_dot_product_attention_causal(&queries, &keys, &values, self.scale as f64)?
            }
        } else {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale as f64, None)?
        };

        // Transpose back: [B, H, T, D] → [B, T, H, D] → flatten to [B, T, H*D]
        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        // Apply gate: output * sigmoid(gate) — one compiled fusion instead
        // of a Sigmoid + Multiply pair.
        // gate is already [B, T, H*D] from the per-head split above
        let gated_output = Activations::sigmoid_mul_compiled(&gate, &output)?;

        // Output projection
        self.o_proj.forward(&gated_output)
    }

    /// SDPA of `queries` `[B, Hq, T, D]` over an int8 `KVCache` after
    /// appending the BF16 block `keys` / `values` `[B, Hkv, T, D]` to it
    /// (quantized on write). Returns `[B, Hq, T, D]`.
    ///
    /// Readers by block shape:
    ///   * `T == 1` (AR decode, MTP draft step) and `2..=8` without a mask
    ///     (eager MTP / DFlash2 verify): the segmented int8 kernels read the
    ///     int8 rows directly — prefix `[0, P)` and the block's own rows as
    ///     the "new" segment, causal for a block.
    ///   * anything else (prefill chunks, explicit masks): the cached prefix
    ///     is dequantized to a BF16 temporary (one pass per layer per chunk)
    ///     and MLX's fused SDPA runs over `concat(prefix, fresh block)`; the
    ///     block's own rows stay BF16 there.
    ///
    /// An empty prefix has no int8 segment to bind, so it also takes the
    /// BF16 route (the first prefill chunk, a one-token prompt's first step).
    fn forward_int8_cached(
        &self,
        queries: &MxArray,
        keys: &MxArray,
        values: &MxArray,
        mask: Option<&MxArray>,
        cache: &mut KVCache,
        seq_len: i64,
    ) -> Result<MxArray> {
        let prefix_len = cache.get_offset() as i64;
        let all = cache.update_and_fetch_int8(keys, values)?;
        let gqa = (self.num_heads / self.num_kv_heads.max(1)) as i64;
        let segmented = mask.is_none()
            && prefix_len > 0
            && (1..=8).contains(&seq_len)
            && (1..=32).contains(&gqa)
            && self.head_dim == 256
            && queries.dtype()? == DType::BFloat16
            && unsafe { mlx_sys::mlx_metal_is_available() }
            && unsafe { mlx_sys::mlx_default_device() } == 1;
        if segmented {
            let prefix = all.slice_tokens(0, prefix_len)?;
            let new = all.slice_tokens(prefix_len, prefix_len + seq_len)?;
            return segmented_sdpa_int8(queries, &prefix, &new, self.scale, seq_len > 1);
        }
        let (k, v) = if prefix_len > 0 {
            let (pk, pv) = all.slice_tokens(0, prefix_len)?.dequantize()?;
            (
                MxArray::concatenate(&pk, keys, 2)?,
                MxArray::concatenate(&pv, values, 2)?,
            )
        } else {
            (keys.clone(), values.clone())
        };
        if let Some(m) = mask {
            scaled_dot_product_attention(queries, &k, &v, self.scale as f64, Some(m))
        } else if seq_len > 1 {
            scaled_dot_product_attention_causal(queries, &k, &v, self.scale as f64)
        } else {
            scaled_dot_product_attention(queries, &k, &v, self.scale as f64, None)
        }
    }

    /// `rope(q_norm(q))`, `rope(k_norm(k))` in one Metal dispatch
    /// (`mlx_qk_norm_rope`), from `[B, T, H, D]` inputs (any strides) to the
    /// `[B, H, T, D]` layout the verify attention reads. Bit-identical to
    /// `fast::rms_norm` + `fast::rope` (gated by `fused_qk_norm_rope_eq`).
    /// `None` on a contract miss (non-Metal, M-RoPE checkpoint, traditional
    /// rope, differing eps, unsupported dtype/dims); callers keep the four-op
    /// chain.
    pub(crate) fn fused_qk_norm_rope(
        &self,
        queries: &MxArray,
        keys: &MxArray,
        offsets: &MxArray,
    ) -> Option<(MxArray, MxArray)> {
        static METAL: OnceLock<bool> = OnceLock::new();
        let metal = *METAL.get_or_init(|| unsafe { mlx_sys::mlx_metal_is_available() });
        if !metal
            || self.mrope.is_some()
            || self.rope.traditional
            || self.q_norm.eps_f32() != self.k_norm.eps_f32()
        {
            return None;
        }
        let mut out_q = std::ptr::null_mut();
        let mut out_k = std::ptr::null_mut();
        // SAFETY: every handle is a live array for the call; the outputs are
        // owned handles or stay null when the call reports false.
        let ok = unsafe {
            mlx_sys::mlx_qk_norm_rope(
                queries.handle.0,
                keys.handle.0,
                self.q_norm.weight().handle.0,
                self.k_norm.weight().handle.0,
                offsets.handle.0,
                self.q_norm.eps_f32(),
                self.rope.base,
                self.rope.scale,
                self.rope.dims,
                &mut out_q,
                &mut out_k,
            )
        };
        if !ok {
            return None;
        }
        let q = MxArray::from_handle(out_q, "qk_norm_rope:q").ok()?;
        let k = MxArray::from_handle(out_k, "qk_norm_rope:k").ok()?;
        Some((q, k))
    }

    /// Compiled-verify forward for the DFlash2 flat-cache path.
    ///
    /// Identical math to [`Self::forward`] with `mask = None`,
    /// `position_ids = None` and a `KVCache`, but the position base and the
    /// K/V prefix arrive as graph INPUTS and the post-RoPE block leaves
    /// through `io.out_kv` — the traced region performs no host-int cache
    /// reads and no cache writes, so the recorded tape stays valid as the
    /// prefix length varies under `shapeless` compile:
    ///
    ///   * RoPE uses `forward_with_offsets` — `offset[b] + t` equals the
    ///     scalar path's `offset + t` bit-for-bit, but the base is a runtime
    ///     array instead of a baked host int.
    ///   * Segmented vector SDPA reads prefix K/V and new K/V as two logical
    ///     spans, so no host-bound prefix slice or full-prefix concat enters
    ///     the tape. The whole verify block is one call: the primitive picks
    ///     its dispatch from the real prefix length at eval time.
    ///   * Without segmented attention (other dtypes or head dims) the block
    ///     is split so each piece keeps `qL·gqa <= 32`; the truncated head
    ///     reads `new[..head_len]`, a slice on the constant block axis. The
    ///     fused SDPA primitive derives `qL_off = kL - qL` from the real input
    ///     shapes at eval time (see
    ///     `backend/metal/scaled_dot_product_attention.cpp`), so the causal
    ///     alignment stays correct for any prefix length, and each piece stays
    ///     on the fused vector kernel — never the unfused fallback, which
    ///     would bake `kL - qL` into `arange` nodes at trace time.
    ///
    /// The caller persists `out_kv` after invoke via `KVCache::update_and_fetch`.
    pub(crate) fn forward_verify(
        &self,
        x: &MxArray,
        io: &mut AttentionVerifyIo<'_>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        let (queries, gate) = self.project_q_gate(x, batch, seq_len)?;
        let (keys, values) = self.project_kv(x)?;
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
        // One dispatch for q_norm + k_norm + rope(q) + rope(k), bit-identical
        // to the four-op chain below (its fallback).
        let (queries, new_keys) = match self.fused_qk_norm_rope(&queries, &keys, io.rope_offsets) {
            Some(fused) => fused,
            None => {
                let queries = self.q_norm.forward(&queries)?;
                let keys = self.k_norm.forward(&keys)?;

                // RoPE rotates along axis -2: feed [B, H, T, D] directly
                // and keep that layout — the eager path's round-trip back
                // to [B, T, H, D] only exists to satisfy the KVCache write
                // order, which lives in the caller here.
                let queries = self.rope.forward_with_offsets(
                    &queries.transpose(Some(&[0, 2, 1, 3]))?,
                    io.rope_offsets,
                )?;
                let new_keys = self
                    .rope
                    .forward_with_offsets(&keys.transpose(Some(&[0, 2, 1, 3]))?, io.rope_offsets)?;
                (queries, new_keys)
            }
        };
        let new_values = values.transpose(Some(&[0, 2, 1, 3]))?;

        let (prefix_keys, prefix_values) = match io.prefix {
            VerifyPrefix::Bf16 { keys, values } => (keys, values),
            VerifyPrefix::Int8 {
                keys,
                values,
                key_scales,
                value_scales,
            } => {
                // Quantize the block inside the graph: the cache stores
                // exactly these rows, and the verify attention reads them
                // in the same int8 form as the prefix (one operand type per
                // kernel, as Splash's store-then-attend order).
                let new = Int8KvRows::quantize(&new_keys, &new_values)?;
                let prefix = Int8KvRows {
                    keys: keys.clone(),
                    values: values.clone(),
                    key_scales: key_scales.clone(),
                    value_scales: value_scales.clone(),
                };
                let output = segmented_sdpa_int8(&queries, &prefix, &new, self.scale, true)?;
                *io.out_kv = Some(VerifyKvOut::Int8(new));
                let output = output.transpose(Some(&[0, 2, 1, 3]))?;
                let output =
                    output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;
                let gated_output = Activations::sigmoid_mul_compiled(&gate, &output)?;
                return self.o_proj.forward(&gated_output);
            }
        };
        *io.out_kv = Some(VerifyKvOut::Bf16(new_keys.clone(), new_values.clone()));

        let output = if seq_len > 1 {
            let gqa = (self.num_heads / self.num_kv_heads.max(1)) as i64;
            let vector_dims = matches!(self.head_dim, 64 | 96 | 128 | 256);
            let segmented_enabled = unsafe { mlx_sys::mlx_metal_is_available() }
                && unsafe { mlx_sys::mlx_default_device() } == 1
                && self.head_dim == 256
                && queries.dtype()? == DType::BFloat16;
            let device_max_q = if segmented_enabled {
                segmented_max_query_length(gqa)?
            } else {
                0
            };
            // Segmented attention serves the whole block in one call.
            let one_call = segmented_enabled
                && prefix_keys.shape_at(2)? > 0
                && segmented_verify_head_len(seq_len, device_max_q).is_some();
            // Otherwise the same qL·gqa verify-split as `forward` — see its
            // comment for the threadgroup-limit rationale — but the head
            // chunk's truncated K/V is `concat(prefix, new[..head_len])` so no
            // slice bound depends on the (shape-varying) prefix length. Query
            // partitioning is constrained by the selected pipeline's SIMD
            // width and maximum threads; keep the established 32-thread
            // arithmetic when the capability probe is absent.
            let max_q = if device_max_q > 0 {
                device_max_q
            } else if (1..=32).contains(&gqa) {
                32 / gqa
            } else {
                0
            };
            let tail = if (1..=32).contains(&gqa) {
                max_q.min(seq_len - 1)
            } else {
                0
            };
            if !one_call && vector_dims && tail >= 1 && seq_len <= 8 && seq_len > max_q {
                let head_len = seq_len - tail;
                let q_parts = queries.split_sections(&[head_len], 2)?;
                let head_new_k = new_keys.slice_axis(2, 0, head_len)?;
                let head_new_v = new_values.slice_axis(2, 0, head_len)?;
                let out_head = verify_sdpa_without_kv_concat(
                    &q_parts[0],
                    prefix_keys,
                    prefix_values,
                    &head_new_k,
                    &head_new_v,
                    self.scale,
                    head_len > 1,
                )?;
                let out_tail = verify_sdpa_without_kv_concat(
                    &q_parts[1],
                    prefix_keys,
                    prefix_values,
                    &new_keys,
                    &new_values,
                    self.scale,
                    true,
                )?;
                MxArray::concatenate(&out_head, &out_tail, 2)?
            } else {
                verify_sdpa_without_kv_concat(
                    &queries,
                    prefix_keys,
                    prefix_values,
                    &new_keys,
                    &new_values,
                    self.scale,
                    true,
                )?
            }
        } else {
            verify_sdpa_without_kv_concat(
                &queries,
                prefix_keys,
                prefix_values,
                &new_keys,
                &new_values,
                self.scale,
                false,
            )?
        };

        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;
        let gated_output = Activations::sigmoid_mul_compiled(&gate, &output)?;
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
        let (keys, values) = self.project_kv(x)?;
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

        let cache_dtype = adapter.prefill_sdpa_cache_dtype();
        let queries = self.paged_attention_operand(&queries, cache_dtype)?;
        let keys = self.paged_attention_operand(&keys, cache_dtype)?;
        let values = self.paged_attention_operand(&values, cache_dtype)?;

        // Transpose to [B, H, T, D] for SDPA.
        let queries_bhtd = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys_bhtd = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values_bhtd = values.transpose(Some(&[0, 2, 1, 3]))?;

        // Paged-pool layout: `[num_tokens, num_kv_heads, head_dim]`.
        // [B, H_kv, T, D] -> [B, T, H_kv, D] -> [B*T, H_kv, D].
        let (keys_paged, values_paged) = crate::models::attention_core::paged_kv_layout(
            &keys_bhtd,
            &values_bhtd,
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        )?;

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
                let portable_d256_available = self.head_dim == 256
                    && dtype_bytes == 2
                    && seq_len > 8
                    && unsafe { mlx_sys::mlx_metal_portable_d256_sdpa_available() };
                let estimated_sdpa_bytes = estimate_paged_pool_sdpa_bytes_with_portable(
                    seq_len as u64,
                    total_ctx as u64,
                    self.num_heads as u64,
                    self.num_kv_heads as u64,
                    self.head_dim as u64,
                    dtype_bytes,
                    d256_full_sdpa_available,
                    portable_d256_available,
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
                        portable_d256_available,
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
            let attn_3d = if !graph_decode_gather_enabled() {
                let attn_3d = adapter
                    .gather_kv_for_decode(
                        attn_layer_idx,
                        &queries_3d,
                        self.scale,
                        /* softcap */ 1.0,
                    )
                    .map_err(napi::Error::from_reason)?;
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] qwen3.5-attn decode_gather_done \
                         layer={} path=legacy total_ctx={} elapsed_ms={:.1}",
                        attn_layer_idx,
                        adapter.current_token_count(),
                        gather_trace_start.map(elapsed_ms).unwrap_or(0.0)
                    ));
                }
                attn_3d
            } else {
                match adapter.gather_kv_for_decode_graph(
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
                }
            };
            let target_dtype = x.dtype()?;
            let attn_3d = attn_3d.astype(target_dtype)?;
            attn_3d.reshape(&[1, self.num_heads as i64, 1, self.head_dim as i64])?
        };

        // Transpose back: [B, H, T, D] -> [B, T, H*D].
        let output = attn_bhtd.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        // Apply gate: output * sigmoid(gate) — one compiled fusion.
        let gated_output = Activations::sigmoid_mul_compiled(&gate, &output)?;

        // Output projection.
        self.o_proj.forward(&gated_output)
    }

    /// Uniform-width hybrid decode and MTP verification. GDN keeps [N,T,H];
    /// only attention packs the time axis into per-request paged query rows.
    pub(crate) fn forward_paged_batched(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        attn_layer_idx: u32,
        rows: &[(SeqId, u32)],
        preserve_singleton_projection_graphs: bool,
    ) -> Result<MxArray> {
        let shape = x.shape()?;
        if rows.is_empty()
            || shape.as_ref().len() != 3
            || shape[0] != rows.len() as i64
            || shape[1] <= 0
        {
            return Err(Error::from_reason(format!(
                "Qwen3_5Attention::forward_paged_batched expects [N,T,H] for {} rows, got {:?}",
                rows.len(),
                shape.as_ref()
            )));
        }
        if !native_kv_write_enabled() {
            return Err(Error::from_reason(
                "Qwen3.5 batched decode requires native K/V writes",
            ));
        }

        let batch = rows.len() as i64;
        let seq_len = shape[1];
        let offsets = rows
            .iter()
            .map(|&(seq_id, position)| {
                i32::try_from(position).map_err(|_| {
                    Error::from_reason(format!(
                        "Qwen3.5 batched decode sequence {seq_id} position {position} exceeds i32::MAX"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = MxArray::from_int32(&offsets, &[batch])?;
        let seq_ids = rows.iter().map(|&(seq_id, _)| seq_id).collect::<Vec<_>>();

        // K-quant projections retain a separate graph per owner.
        // Packed kernels may select a different reduction path for `B > 1`,
        // which can change greedy tokens even though the paged attention
        // operation itself is row-independent. Projection rows are cheap to
        // concatenate and K/V attention below remains one genuine batch.
        let (queries, gate, keys, values) = if preserve_singleton_projection_graphs {
            let mut query_rows = Vec::with_capacity(rows.len());
            let mut gate_rows = Vec::with_capacity(rows.len());
            let mut key_rows = Vec::with_capacity(rows.len());
            let mut value_rows = Vec::with_capacity(rows.len());
            for row in 0..rows.len() {
                let x_row = x.slice_axis(0, row as i64, row as i64 + 1)?;
                let (query, gate) = self.project_q_gate(&x_row, 1, seq_len)?;
                query_rows.push(self.q_norm.forward(&query)?);
                gate_rows.push(gate);
                let (key_raw, value_raw) = self.project_kv(&x_row)?;
                let key = key_raw.reshape(&[
                    1,
                    seq_len,
                    self.num_kv_heads as i64,
                    self.head_dim as i64,
                ])?;
                key_rows.push(self.k_norm.forward(&key)?);
                value_rows.push(value_raw.reshape(&[
                    1,
                    seq_len,
                    self.num_kv_heads as i64,
                    self.head_dim as i64,
                ])?);
            }
            (
                MxArray::concatenate_many(query_rows.iter().collect(), Some(0))?,
                MxArray::concatenate_many(gate_rows.iter().collect(), Some(0))?,
                MxArray::concatenate_many(key_rows.iter().collect(), Some(0))?,
                MxArray::concatenate_many(value_rows.iter().collect(), Some(0))?,
            )
        } else {
            let (queries, gate) = self.project_q_gate(x, batch, seq_len)?;
            let queries = self.q_norm.forward(&queries)?;
            let (key_raw, value_raw) = self.project_kv(x)?;
            let keys = self.k_norm.forward(&key_raw.reshape(&[
                batch,
                seq_len,
                self.num_kv_heads as i64,
                self.head_dim as i64,
            ])?)?;
            let values = value_raw.reshape(&[
                batch,
                seq_len,
                self.num_kv_heads as i64,
                self.head_dim as i64,
            ])?;
            (queries, gate, keys, values)
        };
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let queries = self.rope.forward_with_offsets(&queries, &offsets)?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = self.rope.forward_with_offsets(&keys, &offsets)?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;
        let cache_dtype = adapter.prefill_sdpa_cache_dtype();
        let queries = self.paged_attention_operand(&queries, cache_dtype)?;
        let keys = self.paged_attention_operand(&keys, cache_dtype)?;
        let values = self.paged_attention_operand(&values, cache_dtype)?;

        let output = if seq_len == 1 {
            let queries = queries.squeeze(Some(&[2]))?;
            let keys = keys.squeeze(Some(&[2]))?;
            let values = values.squeeze(Some(&[2]))?;
            adapter
                .update_keys_values_native_batched(attn_layer_idx, &keys, &values, rows)
                .map_err(Error::from_reason)?;
            adapter
                .gather_kv_for_decode_graph_batched(
                    attn_layer_idx,
                    &queries,
                    &seq_ids,
                    self.scale,
                    1.0,
                )
                .map_err(Error::from_reason)?
        } else {
            let packed = |array: &MxArray, heads: i32| -> Result<MxArray> {
                array.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
                    batch * seq_len,
                    heads as i64,
                    self.head_dim as i64,
                ])
            };
            let queries = packed(&queries, self.num_heads)?;
            let keys = packed(&keys, self.num_kv_heads)?;
            let values = packed(&values, self.num_kv_heads)?;
            let ragged = rows
                .iter()
                .map(|&(seq_id, position)| {
                    crate::transformer::paged_kv_cache_adapter::PagedRaggedRow {
                        seq_id,
                        first_logical_position: position,
                        query_len: seq_len as u32,
                    }
                })
                .collect::<Vec<_>>();
            adapter
                .update_keys_values_native_ragged(attn_layer_idx, &keys, &values, &ragged)
                .map_err(Error::from_reason)?;
            adapter
                .gather_kv_for_ragged_graph(attn_layer_idx, &queries, &ragged, self.scale, 1.0)
                .map_err(Error::from_reason)?
        }
        .astype(x.dtype()?)?
        .reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;
        let output = Activations::sigmoid_mul_compiled(&gate, &output)?;
        if preserve_singleton_projection_graphs {
            let projected = (0..rows.len())
                .map(|row| {
                    let row = output.slice_axis(0, row as i64, row as i64 + 1)?;
                    self.o_proj.forward(&row)
                })
                .collect::<Result<Vec<_>>>()?;
            MxArray::concatenate_many(projected.iter().collect(), Some(0))
        } else {
            self.o_proj.forward(&output)
        }
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
        self.kv_proj = None;
        self.k_proj.set_weight(w, "k_proj")
    }
    pub fn set_v_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        self.kv_proj = None;
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
        self.kv_proj = None;
        self.k_proj.set_bias(b, "k_proj")
    }
    pub fn set_v_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.kv_proj = None;
        self.v_proj.set_bias(b, "v_proj")
    }
    pub fn set_o_proj_bias(&mut self, b: Option<&MxArray>) -> Result<()> {
        self.o_proj.set_bias(b, "o_proj")
    }
    pub fn set_q_norm_weight(&mut self, w: &MxArray, compute_dtype: DType) -> Result<()> {
        self.q_norm
            .set_weight(&super::sidecar_to_compute_dtype(w, compute_dtype)?)
    }
    pub fn set_k_norm_weight(&mut self, w: &MxArray, compute_dtype: DType) -> Result<()> {
        self.k_norm
            .set_weight(&super::sidecar_to_compute_dtype(w, compute_dtype)?)
    }

    /// Precompute the block-ordered `[hidden, 2*H*D]` q_proj weight (queries
    /// flat | gate flat) so `project_q_gate` can split queries vs. gate as
    /// two flat, row-contiguous slices instead of reshaping to
    /// `[B,T,H,2D]` and slicing per head. See `q_gate_block_t`'s doc
    /// comment for why the checkpoint's native per-head-interleaved column
    /// order forces a strided copy on every call.
    ///
    /// Safe to call repeatedly (idempotent). Affine and native GGUF K/IQ
    /// q_proj consume their native-order packed operands and replace them with
    /// materialized block-order arrays. Other quantization modes remain on the
    /// fallback.
    pub fn finalize_q_gate_block(&mut self) -> Result<()> {
        let h = self.num_heads as i64;
        let d = self.head_dim as i64;

        if let LinearProj::Quantized(q_lin) = &mut self.q_proj {
            q_lin.finalize_packed_q_gate_block(self.num_heads, self.head_dim)?;
            return Ok(());
        }

        let LinearProj::Standard(q_lin) = &self.q_proj else {
            return Err(Error::from_reason(
                "Qwen3.5 q_proj changed variants while finalizing the block layout",
            ));
        };

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
        self.kv_proj = None;
        self.k_proj.set_quantized(ql);
    }
    pub fn set_quantized_v_proj(&mut self, ql: QuantizedLinear) {
        self.kv_proj = None;
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

    /// Test-only: the q_proj quantization mode (`None` when dense) and
    /// whether its q/gate block was merged (packed reorder or dense cache).
    #[cfg(test)]
    pub(crate) fn q_proj_mode_and_block(&self) -> (Option<String>, bool) {
        let mode = match &self.q_proj {
            LinearProj::Quantized(ql) => Some(ql.mode().to_string()),
            LinearProj::Standard(_) => None,
        };
        (
            mode,
            self.q_gate_block_t.is_some() || self.q_proj.has_q_gate_block_layout(),
        )
    }

    #[cfg(test)]
    pub(crate) fn prism_hadamard_sites(&self) -> (bool, bool, bool, bool) {
        (
            self.q_proj.has_hadamard(),
            self.k_proj.has_hadamard(),
            self.v_proj.has_hadamard(),
            self.o_proj.has_hadamard(),
        )
    }
}

#[cfg(test)]
mod kv_int8_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapeless_verify_requires_dynamic_prefix_safe_attention() {
        assert!(!verify_shapeless_geometry(8, 1, 32, 0));
        assert!(verify_shapeless_geometry(8, 1, 64, 0));
        assert!(verify_shapeless_geometry(8, 6, 256, 5));
        assert!(!verify_shapeless_geometry(8, 6, 256, 2));
        assert!(!verify_shapeless_geometry(8, 32, 256, 1));
        assert!(!verify_shapeless_geometry(9, 1, 64, 0));
        assert!(!verify_shapeless_geometry(8, 0, 64, 0));
    }

    fn synthetic_segmented_plan(
        q_len: i32,
        gqa: i32,
        two_pass: bool,
        partitions: i32,
        stage1_width: usize,
        stage1_max_threads: usize,
        stage1_static_memory: usize,
        device_max_memory: usize,
        stage2_width: usize,
        stage2_max_threads: usize,
        stage2_static_memory: usize,
    ) -> (bool, u32, u32) {
        let mut stage1_threads = 0;
        let mut stage2_threads = 0;
        let supported = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_plan(
                q_len,
                gqa,
                two_pass,
                partitions,
                stage1_width,
                stage1_max_threads,
                stage1_static_memory,
                device_max_memory,
                stage2_width,
                stage2_max_threads,
                stage2_static_memory,
                &mut stage1_threads,
                &mut stage2_threads,
            )
        } == 1;
        (supported, stage1_threads, stage2_threads)
    }

    #[test]
    fn segmented_sdpa_capability_planner_accepts_only_supported_profiles() {
        let valid =
            synthetic_segmented_plan(5, 6, true, 128, 32, 1024, 4096, 32768, 32, 1024, 4096);
        assert_eq!(valid, (true, 960, 1024));

        // SIMD width 32 belongs to the reduction algorithm. Device capacities
        // are runtime inputs and must independently reject an unsafe launch.
        assert!(
            !synthetic_segmented_plan(5, 6, true, 128, 16, 1024, 4096, 32768, 32, 1024, 4096).0
        );
        assert!(!synthetic_segmented_plan(5, 6, true, 128, 32, 512, 4096, 32768, 32, 1024, 4096).0);
        assert!(
            !synthetic_segmented_plan(5, 6, true, 128, 32, 1024, 65536, 32768, 32, 1024, 4096).0
        );
        assert!(!synthetic_segmented_plan(5, 6, true, 128, 32, 1024, 4096, 32768, 32, 512, 4096).0);

        let one_pass = synthetic_segmented_plan(8, 6, false, 32, 32, 1024, 4096, 32768, 0, 0, 0);
        assert_eq!(one_pass, (true, 1024, 0));
    }

    #[cfg(target_os = "macos")]
    fn deterministic_bf16(len: usize, salt: u32) -> Vec<u16> {
        (0..len)
            .map(|i| {
                let x = (i as u32).wrapping_mul(0x9e37_79b9).wrapping_add(salt);
                let sign = ((x >> 23) as u16 & 1) << 15;
                sign | 0x3f00 | ((x >> 16) as u16 & 0x7f)
            })
            .collect()
    }

    #[cfg(target_os = "macos")]
    fn concat_sdpa_for_test(
        q: &MxArray,
        pk: &MxArray,
        pv: &MxArray,
        nk: &MxArray,
        nv: &MxArray,
        causal: bool,
    ) -> Result<MxArray> {
        let k = MxArray::concatenate(pk, nk, 2)?;
        let v = MxArray::concatenate(pv, nv, 2)?;
        if causal {
            scaled_dot_product_attention_causal(q, &k, &v, 0.0625)
        } else {
            scaled_dot_product_attention(q, &k, &v, 0.0625, None)
        }
    }

    #[cfg(target_os = "macos")]
    fn segmented_or_concat_split_for_test(
        q: &MxArray,
        pk: &MxArray,
        pv: &MxArray,
        nk: &MxArray,
        nv: &MxArray,
        segmented: bool,
    ) -> Result<MxArray> {
        let q_len = q.shape_at(2)?;
        let gqa = q.shape_at(1)? / pk.shape_at(1)?;
        let probed_max =
            (unsafe { mlx_sys::mlx_segmented_sdpa_max_query_length(gqa as i32) }) as i64;
        let max_q = if probed_max > 0 { probed_max } else { 32 / gqa };
        let call =
            |query: &MxArray, new_k: &MxArray, new_v: &MxArray, causal: bool| -> Result<MxArray> {
                if segmented && pk.shape_at(2)? > 0 {
                    let handle = unsafe {
                        mlx_sys::mlx_segmented_sdpa_test_forward(
                            query.as_raw_ptr(),
                            pk.as_raw_ptr(),
                            pv.as_raw_ptr(),
                            new_k.as_raw_ptr(),
                            new_v.as_raw_ptr(),
                            0.0625,
                            causal,
                        )
                    };
                    MxArray::from_handle(handle, "strict segmented SDPA test")
                } else {
                    concat_sdpa_for_test(query, pk, pv, new_k, new_v, causal)
                }
            };
        if q_len <= max_q {
            return call(q, nk, nv, q_len > 1);
        }
        let tail = max_q.min(q_len - 1);
        let head_len = q_len - tail;
        let q_parts = q.split_sections(&[head_len], 2)?;
        let head_k = nk.slice_axis(2, 0, head_len)?;
        let head_v = nv.slice_axis(2, 0, head_len)?;
        let head = call(&q_parts[0], &head_k, &head_v, head_len > 1)?;
        let tail = call(&q_parts[1], nk, nv, true)?;
        MxArray::concatenate(&head, &tail, 2)
    }

    /// This is intentionally an explicit GPU test: it covers every required
    /// verify width and prefix boundary, including the capability-derived
    /// split at q=8 (3/5 on the measured pipeline). Comparing f32 views is
    /// bit-exact for BF16 because every BF16 value has a unique f32
    /// representation.
    #[test]
    #[ignore = "requires coordinated Metal GPU validation"]
    #[cfg(target_os = "macos")]
    fn segmented_sdpa_matches_concat_bf16_exactly() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        // The dedicated test entry rejects unsupported pipelines instead of
        // falling back to concat SDPA, so eligible parity cases must exercise
        // SegmentedSdpa without changing process-global environment state.
        let max_query_length = unsafe { mlx_sys::mlx_segmented_sdpa_max_query_length(6) };
        eprintln!("segmented SDPA GQA=6 max_query_length={max_query_length}");
        if max_query_length < 1 {
            eprintln!(
                "SKIP segmented SDPA parity: selected Metal pipelines do not support any query width"
            );
            return Ok(());
        }
        // Production uses at most two chunks. A smaller legal pipeline can
        // support some widths without supporting the full eight-row split.
        let max_split_width = i64::from(max_query_length) * 2;
        if max_split_width < 8 {
            eprintln!(
                "SKIP segmented SDPA parity query widths {}..=8: production \
                 two-chunk route supports at most {max_split_width} rows",
                max_split_width + 1
            );
        }
        const D: i64 = 256;
        // Include segment and partition boundaries on either side of the
        // split traversal, including transitions crossed by an eight-row tail.
        for prefix in [
            0_i64, 1, 31, 32, 87, 1015, 1016, 1023, 1024, 4095, 4096, 6143, 6144, 6145, 32759,
            32760, 32767, 32768, 32769,
        ] {
            let prefix_elements = (prefix * D) as usize;
            let pk = MxArray::from_bfloat16(
                &deterministic_bf16(prefix_elements, 0x1234_5678),
                &[1, 1, prefix, D],
            )?;
            let pv = MxArray::from_bfloat16(
                &deterministic_bf16(prefix_elements, 0x8765_4321),
                &[1, 1, prefix, D],
            )?;
            for q_len in 1_i64..=8 {
                if q_len > max_split_width {
                    continue;
                }
                let q_elements = (6 * q_len * D) as usize;
                let kv_elements = (q_len * D) as usize;
                let q = MxArray::from_bfloat16(
                    &deterministic_bf16(q_elements, 0x1357_9bdf),
                    &[1, 6, q_len, D],
                )?;
                let nk = MxArray::from_bfloat16(
                    &deterministic_bf16(kv_elements, 0x2468_ace0),
                    &[1, 1, q_len, D],
                )?;
                let nv = MxArray::from_bfloat16(
                    &deterministic_bf16(kv_elements, 0xfdb9_7531),
                    &[1, 1, q_len, D],
                )?;
                let got = segmented_or_concat_split_for_test(&q, &pk, &pv, &nk, &nv, true)?
                    .to_float32()?;
                let expected = segmented_or_concat_split_for_test(&q, &pk, &pv, &nk, &nv, false)?
                    .to_float32()?;
                assert_eq!(
                    got.as_ref(),
                    expected.as_ref(),
                    "BF16 mismatch at prefix={prefix}, query_length={q_len}"
                );
            }
        }

        // Production layout coverage: projection outputs start as [B,T,H,D]
        // and are transposed into [B,H,T,D] views. Exercise non-unit batch,
        // multiple KV heads, the q=8 split, and both vector-SDPA routes.
        if max_split_width < 8 {
            eprintln!(
                "SKIP segmented SDPA eight-row strided cases: unsupported \
                 production split; supported contiguous widths were checked"
            );
            return Ok(());
        }
        for prefix in [87_i64, 4096] {
            let pk = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 2 * prefix * D) as usize, 0x1020_3040),
                &[2, 2, prefix, D],
            )?;
            let pv = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 2 * prefix * D) as usize, 0x5060_7080),
                &[2, 2, prefix, D],
            )?;
            let q_source = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 8 * 12 * D) as usize, 0x90a0_b0c0),
                &[2, 8, 12, D],
            )?;
            let nk_source = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 8 * 2 * D) as usize, 0xd0e0_f001),
                &[2, 8, 2, D],
            )?;
            let nv_source = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 8 * 2 * D) as usize, 0x1234_abcd),
                &[2, 8, 2, D],
            )?;
            let q = q_source.transpose(Some(&[0, 2, 1, 3]))?;
            let nk = nk_source.transpose(Some(&[0, 2, 1, 3]))?;
            let nv = nv_source.transpose(Some(&[0, 2, 1, 3]))?;
            let got =
                segmented_or_concat_split_for_test(&q, &pk, &pv, &nk, &nv, true)?.to_float32()?;
            let expected =
                segmented_or_concat_split_for_test(&q, &pk, &pv, &nk, &nv, false)?.to_float32()?;
            assert_eq!(
                got.as_ref(),
                expected.as_ref(),
                "production-layout BF16 mismatch at prefix={prefix}, query_length=8"
            );
        }
        Ok(())
    }

    /// Segmented SDPA equals concatenated K/V through MLX's vector SDPA only
    /// while it copies MLX's two-pass route and partition count
    /// (`sdpa_vector_uses_two_pass` / `sdpa_vector_partition_count` in
    /// `mlx_segmented_sdpa_plan.h`), because either one changes the reduction
    /// order. Compares one single-chunk block per `(q_heads, kv_heads, rows)`
    /// and total K/V length, against `[1, kv, 65600, 256]` cache slices.
    #[cfg(target_os = "macos")]
    fn segmented_policy_cases(layouts: &[(i64, i64, i64)], totals: &[i64]) -> Result<usize> {
        const D: i64 = 256;
        const CAPACITY: i64 = 65_600;
        let kv_max = layouts.iter().map(|l| l.1).max().unwrap_or(1);
        let base_k = MxArray::from_bfloat16(
            &deterministic_bf16((kv_max * CAPACITY * D) as usize, 0x0f1e_2d3c),
            &[1, kv_max, CAPACITY, D],
        )?;
        let base_v = MxArray::from_bfloat16(
            &deterministic_bf16((kv_max * CAPACITY * D) as usize, 0x4b5a_6978),
            &[1, kv_max, CAPACITY, D],
        )?;
        base_k.eval();
        base_v.eval();
        let mut checked = 0usize;
        for &(q_heads, kv_heads, rows) in layouts {
            let gqa = q_heads / kv_heads;
            let max_q =
                i64::from(unsafe { mlx_sys::mlx_segmented_sdpa_max_query_length(gqa as i32) });
            assert!(max_q >= 1, "gqa {gqa}: no supported segmented query width");
            assert!(
                segmented_verify_head_len(rows, max_q).is_some(),
                "gqa {gqa} rows {rows}: max_query_length {max_q} cannot cover the block"
            );
            let q = MxArray::from_bfloat16(
                &deterministic_bf16((q_heads * rows * D) as usize, 0x1357_9bdf),
                &[1, q_heads, rows, D],
            )?;
            let nk = MxArray::from_bfloat16(
                &deterministic_bf16((kv_heads * rows * D) as usize, 0x2468_ace0),
                &[1, kv_heads, rows, D],
            )?;
            let nv = MxArray::from_bfloat16(
                &deterministic_bf16((kv_heads * rows * D) as usize, 0xfdb9_7531),
                &[1, kv_heads, rows, D],
            )?;
            for &total in totals {
                let prefix = total - rows;
                let pk = base_k.slice(&[0, 0, 0, 0], &[1, kv_heads, prefix, D])?;
                let pv = base_v.slice(&[0, 0, 0, 0], &[1, kv_heads, prefix, D])?;
                let got = segmented_or_concat_split_for_test(&q, &pk, &pv, &nk, &nv, true)?
                    .to_float32()?;
                let expected = segmented_or_concat_split_for_test(&q, &pk, &pv, &nk, &nv, false)?
                    .to_float32()?;
                assert_eq!(
                    got.as_ref(),
                    expected.as_ref(),
                    "gqa {gqa} ({q_heads}/{kv_heads}) rows {rows} total {total}"
                );
                checked += 1;
            }
        }
        Ok(checked)
    }

    /// Every exact boundary of every device class's reduction policy, at
    /// active simdgroups 1..8 (gqa x rows), plus the one-call verify routes.
    /// Runs by default. Other device classes need a fresh process:
    ///
    /// ```text
    /// MLX_METAL_GPU_ARCH=applegpu_g17d cargo test -p mlx-core --release --lib -- segmented_sdpa_policy_boundaries
    /// MLX_METAL_GPU_ARCH=applegpu_g17g cargo test -p mlx-core --release --lib -- segmented_sdpa_policy_boundaries
    /// ```
    #[test]
    #[cfg(target_os = "macos")]
    fn segmented_sdpa_policy_boundaries_match_mlx_vector_sdpa() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            assert!(
                !crate::test_support::metal_required(),
                "MLX_TEST_REQUIRE_METAL=1 but no Metal device"
            );
            eprintln!("SKIP segmented SDPA policy boundaries: no Metal device");
            return Ok(());
        }
        // (q heads, kv heads, rows): active simdgroups 1, 2, 3, 4, 5, 6, 8;
        // gqa 1 never takes the GQA two-pass route, the others do from 4096.
        let layouts = [
            (1_i64, 1_i64, 1_i64),
            (1, 1, 2),
            (1, 1, 3),
            (2, 1, 2),
            (1, 1, 5),
            (6, 1, 1),
            (8, 2, 2),
        ];
        let totals = [
            1023_i64, 1024, 1025, 4095, 4096, 8192, 8193, 16383, 16384, 32768, 32769, 65535, 65536,
            65537,
        ];
        let checked = segmented_policy_cases(&layouts, &totals)?;
        assert_eq!(checked, layouts.len() * totals.len());

        // One-call verify: 8 rows of the 27B layout (24 q / 4 kv heads). The
        // head and tail chunks straddle each class's steps: 1024 ('s', 'd'
        // two-pass), 4096 (base-class GQA two-pass), 8192 ('s' 128 -> 256) and
        // 16384 ('d' 128 -> 512), so every class reaches every route its GPU
        // can launch.
        const D: i64 = 256;
        const HQ: i64 = 24;
        const HKV: i64 = 4;
        let max_q = i64::from(unsafe { mlx_sys::mlx_segmented_sdpa_max_query_length(6) });
        assert!(max_q >= 1, "gqa 6: no supported segmented query width");
        let head_len = segmented_verify_head_len(8, max_q)
            .ok_or_else(|| Error::from_reason("8 verify rows cannot be covered"))?;
        let prefixes = [1000_i64, 1016, 1020, 4090, 6219, 8186, 8190, 16378];
        const CAPACITY: i64 = 16_400;
        let base_k = MxArray::from_bfloat16(
            &deterministic_bf16((HKV * CAPACITY * D) as usize, 0x1234_5678),
            &[1, HKV, CAPACITY, D],
        )?;
        let base_v = MxArray::from_bfloat16(
            &deterministic_bf16((HKV * CAPACITY * D) as usize, 0x8765_4321),
            &[1, HKV, CAPACITY, D],
        )?;
        let q = MxArray::from_bfloat16(
            &deterministic_bf16((HQ * 8 * D) as usize, 0x1357_9bdf),
            &[1, HQ, 8, D],
        )?;
        let nk = MxArray::from_bfloat16(
            &deterministic_bf16((HKV * 8 * D) as usize, 0x2468_ace0),
            &[1, HKV, 8, D],
        )?;
        let nv = MxArray::from_bfloat16(
            &deterministic_bf16((HKV * 8 * D) as usize, 0xfdb9_7531),
            &[1, HKV, 8, D],
        )?;
        let mut seen = [0usize; 4];
        let mut unified_eligible = 0usize;
        let mut unified_launchable = 0usize;
        let mut class = 0u8;
        for prefix in prefixes {
            let (route, device_class) = device_verify_route(HQ, HKV, 8, prefix);
            class = device_class;
            let (expected, eligible) = predicted_verify_route(class, HQ, HKV, 8, prefix, max_q);
            unified_eligible += usize::from(eligible);
            unified_launchable += usize::from(expected == VERIFY_UNIFIED);
            assert_eq!(
                route, expected,
                "class '{}' prefix {prefix}: verify route",
                class as char
            );
            seen[route as usize] += 1;
            let pk = nan_tailed_prefix(&base_k, prefix)?;
            let pv = nan_tailed_prefix(&base_v, prefix)?;
            let (got, dispatched) = strict_segmented_routed(&q, &pk, &pv, &nk, &nv)?;
            assert_eq!(
                dispatched, route,
                "class '{}' prefix {prefix}: dispatched verify route",
                class as char
            );
            let concat =
                segmented_or_concat_split_for_test(&q, &pk, &pv, &nk, &nv, false)?.to_float32()?;
            assert_eq!(
                got.as_ref(),
                concat.as_ref(),
                "one-call verify prefix {prefix} route {route}"
            );
        }
        eprintln!(
            "class '{}': {checked} policy boundaries + {} verify blocks equal MLX's vector \
             SDPA; verify routes one_pass={} unified={} split={}",
            class as char,
            prefixes.len(),
            seen[1],
            seen[2],
            seen[3]
        );
        assert!(
            head_len > 0,
            "8 rows fit one chunk (max_query_length {max_q})"
        );
        assert!(
            unified_eligible > 0,
            "class '{}': no prefix reaches the one-call route",
            class as char
        );
        assert_eq!(
            seen[VERIFY_UNIFIED as usize], unified_launchable,
            "class '{}': one-call verify dispatches",
            class as char
        );
        if unified_launchable == 0 {
            eprintln!(
                "class '{}': this GPU cannot launch the one-call verify kernel; \
                 {unified_eligible} eligible blocks took the split route",
                class as char
            );
        }
        for route in [VERIFY_ONE_PASS, VERIFY_SPLIT] {
            assert!(
                seen[route as usize] > 0,
                "class '{}': verify route {route} never ran",
                class as char
            );
        }
        Ok(())
    }

    /// The exhaustive version: totals on both sides of every boundary for GQA
    /// 1/2/4/6/8 at every coverable row count (1152 blocks). Run it under
    /// `MLX_METAL_GPU_ARCH` with a 'd' and a base-class architecture too.
    #[test]
    #[ignore = "requires coordinated Metal GPU validation"]
    #[cfg(target_os = "macos")]
    fn segmented_sdpa_reduction_policy_matches_mlx_vector_sdpa() -> Result<()> {
        assert!(
            unsafe { mlx_sys::mlx_metal_is_available() },
            "this GPU validation needs a Metal device"
        );
        let mut layouts = Vec::new();
        for (q_heads, kv_heads) in [(1_i64, 1_i64), (2, 1), (4, 1), (6, 1), (8, 2), (16, 2)] {
            for rows in 1_i64..=8 {
                layouts.push((q_heads, kv_heads, rows));
            }
        }
        let mut totals = Vec::new();
        for boundary in [
            1024_i64, 1025, 4096, 4097, 8192, 8193, 16384, 16385, 32768, 32769, 65536, 65537,
        ] {
            totals.extend([boundary - 1, boundary]);
        }
        let checked = segmented_policy_cases(&layouts, &totals)?;
        assert!(checked >= 1152, "only {checked} policy cases ran");
        eprintln!("{checked} policy-boundary cases equal MLX's vector SDPA");
        Ok(())
    }

    const VERIFY_SINGLE: i32 = 0;
    const VERIFY_ONE_PASS: i32 = 1;
    const VERIFY_UNIFIED: i32 = 2;
    const VERIFY_SPLIT: i32 = 3;

    fn verify_route(
        rows: i64,
        max_q: i64,
        head: (bool, i64),
        tail: (bool, i64),
        unified_supported: bool,
    ) -> i32 {
        unsafe {
            mlx_sys::mlx_segmented_sdpa_test_verify_route(
                rows as i32,
                max_q as i32,
                head.0,
                head.1 as i32,
                tail.0,
                tail.1 as i32,
                unified_supported,
            )
        }
    }

    fn verify_plan(
        rows: i32,
        gqa: i32,
        partitions: i32,
        stage1: (usize, usize, usize),
        stage2: (usize, usize, usize),
    ) -> (bool, u32) {
        let mut threads = 0;
        let supported = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_verify_plan(
                rows,
                gqa,
                partitions,
                stage1.0,
                stage1.1,
                stage1.2,
                32768,
                stage2.0,
                stage2.1,
                stage2.2,
                &mut threads,
            )
        } == 1;
        (supported, threads)
    }

    #[test]
    fn segmented_verify_route_planner_is_exact_contract() {
        // The Rust eligibility rule must agree with the C++ chunk rule.
        for rows in 0_i64..=9 {
            for max_q in 0_i64..=9 {
                let route = verify_route(rows, max_q, (false, 32), (false, 32), false);
                let expected = match segmented_verify_head_len(rows, max_q) {
                    None => -1,
                    Some(0) => VERIFY_SINGLE,
                    Some(_) => VERIFY_ONE_PASS,
                };
                assert_eq!(route, expected, "rows={rows}, max_q={max_q}");
            }
        }
        assert_eq!(segmented_verify_head_len(8, 5), Some(3));
        assert_eq!(segmented_verify_head_len(7, 5), Some(2));
        assert_eq!(segmented_verify_head_len(6, 5), Some(1));
        assert_eq!(segmented_verify_head_len(8, 4), Some(4));
        assert_eq!(segmented_verify_head_len(8, 3), None);
        assert_eq!(segmented_verify_head_len(5, 5), Some(0));
        assert_eq!(segmented_verify_head_len(8, 8), Some(0));
        assert_eq!(segmented_verify_head_len(8, 0), None);

        // One dispatch is exact only when both chunks reduce the same way.
        let one = (false, 32);
        let p256 = (true, 256);
        let p512 = (true, 512);
        assert_eq!(verify_route(5, 5, p256, p512, true), VERIFY_SINGLE);
        assert_eq!(verify_route(8, 5, one, one, true), VERIFY_ONE_PASS);
        assert_eq!(verify_route(8, 5, p256, p256, true), VERIFY_UNIFIED);
        assert_eq!(verify_route(8, 5, p256, p512, true), VERIFY_SPLIT);
        assert_eq!(verify_route(8, 5, one, p256, true), VERIFY_SPLIT);
        assert_eq!(verify_route(8, 5, p256, one, true), VERIFY_SPLIT);
        assert_eq!(verify_route(8, 5, p256, p256, false), VERIFY_SPLIT);
        assert_eq!(
            verify_route(6, 5, (true, 128), (true, 128), true),
            VERIFY_UNIFIED
        );

        // Two pairs per simdgroup: 32 * gqa * rows / 2 threads, with the same
        // partition and device-capability rules as the chunked launch.
        let caps = (32, 1024, 0);
        assert_eq!(verify_plan(8, 6, 256, caps, caps), (true, 768));
        assert_eq!(verify_plan(7, 6, 128, caps, caps), (true, 672));
        assert_eq!(verify_plan(6, 6, 64, caps, caps), (true, 576));
        assert!(!verify_plan(3, 3, 256, caps, caps).0, "odd pair count");
        assert!(
            !verify_plan(1, 6, 256, caps, caps).0,
            "one row is not a block"
        );
        assert!(!verify_plan(8, 6, 256, (32, 512, 0), caps).0);
        assert!(!verify_plan(8, 6, 256, (16, 1024, 0), caps).0);
        assert!(!verify_plan(8, 6, 256, (32, 1024, 65536), caps).0);
        assert!(!verify_plan(8, 6, 256, caps, (32, 512, 0)).0);
        assert!(!verify_plan(8, 6, 256, caps, (16, 1024, 0)).0);
        assert!(!verify_plan(8, 6, 48, caps, caps).0, "partitions % 32");
        assert!(!verify_plan(8, 6, 0, caps, caps).0, "partitions < 32");
    }

    /// Tile planner over synthetic limits: (supported, tile keys, threads,
    /// threadgroup bytes, partitions for `total` keys).
    fn tile_plan(
        rows: i32,
        gqa: i32,
        total: i32,
        blocks_override: i32,
        stage1: (usize, usize, usize),
        device_memory: usize,
        stage2: (usize, usize, usize),
    ) -> (bool, u32, u32, u32, u32) {
        let mut out = [0u32; 4];
        let supported = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_verify_tile_plan(
                rows,
                gqa,
                total,
                blocks_override,
                stage1.0,
                stage1.1,
                stage1.2,
                device_memory,
                stage2.0,
                stage2.1,
                stage2.2,
                out.as_mut_ptr(),
            )
        } == 1;
        (supported, out[0], out[1], out[2], out[3])
    }

    /// The simdgroup-matrix verify kernel: 2 simdgroups per 8 queries of
    /// M = gqa x rows, a K/V tile of `tile_n x (256 + 8)` BF16 within three
    /// quarters of the device's threadgroup memory limit plus the score
    /// exchange (simdgroups x 32 x tile_n / 4 floats), and contiguous
    /// partitions of about 8 tiles (32..=1024, multiples of 32).
    #[test]
    fn segmented_verify_tile_planner_follows_device_limits() {
        let caps = (32, 1024, 0);
        let kib = 1024;
        // 24 q heads / 4 kv heads, 8 rows: M = 48 -> 12 simdgroups.
        assert_eq!(
            tile_plan(8, 6, 32776, 0, caps, 32 * kib, caps),
            (true, 32, 384, 32 * 264 * 2 + 12 * 32 * 8 * 4, 160)
        );
        assert_eq!(tile_plan(8, 6, 95, 0, caps, 32 * kib, caps).4, 32);
        assert_eq!(tile_plan(8, 6, 6227, 0, caps, 32 * kib, caps).4, 32);
        assert_eq!(tile_plan(8, 6, 16392, 0, caps, 32 * kib, caps).4, 96);
        assert_eq!(tile_plan(8, 6, 1 << 20, 0, caps, 32 * kib, caps).4, 1024);
        // MLX_SDPA_BLOCKS rounds up to 32, as the vector policy.
        assert_eq!(tile_plan(8, 6, 32776, 100, caps, 32 * kib, caps).4, 128);
        assert_eq!(tile_plan(8, 6, 32776, 4096, caps, 32 * kib, caps).4, 1024);
        // The tile follows the memory limit: 16 KiB -> 16 keys; 32 keys is
        // the ceiling (64 measured slower); 8 KiB fits nothing.
        assert_eq!(tile_plan(8, 6, 32776, 0, caps, 16 * kib, caps).1, 16);
        assert_eq!(tile_plan(8, 6, 32776, 0, caps, 64 * kib, caps).1, 32);
        assert!(!tile_plan(8, 6, 32776, 0, caps, 8 * kib, caps).0);
        // M must be a multiple of 8 and the block at least 2 rows.
        assert!(tile_plan(4, 6, 1000, 0, caps, 32 * kib, caps).0);
        assert!(!tile_plan(5, 6, 1000, 0, caps, 32 * kib, caps).0);
        assert!(!tile_plan(1, 8, 1000, 0, caps, 32 * kib, caps).0);
        assert!(tile_plan(2, 8, 1000, 0, caps, 32 * kib, caps).0);
        // 32 x 8 = 256 queries need 64 simdgroups: over 1024 threads. Wider
        // exchanges shrink the tile: 16 simdgroups (gqa 8) and 32 (gqa 16)
        // take 16 keys.
        assert!(!tile_plan(8, 32, 1000, 0, caps, 32 * kib, caps).0);
        assert_eq!(
            tile_plan(8, 16, 1000, 0, caps, 32 * kib, caps),
            (true, 16, 1024, 16 * 264 * 2 + 32 * 32 * 4 * 4, 32)
        );
        assert_eq!(tile_plan(8, 8, 1000, 0, caps, 32 * kib, caps).1, 16);
        assert_eq!(tile_plan(8, 8, 1000, 0, caps, 32 * kib, caps).2, 512);
        // The pipeline's own limits rule: width, threads, static memory.
        assert!(!tile_plan(8, 6, 1000, 0, (16, 1024, 0), 32 * kib, caps).0);
        assert!(!tile_plan(8, 6, 1000, 0, (32, 256, 0), 32 * kib, caps).0);
        assert_eq!(
            tile_plan(8, 6, 1000, 0, (32, 1024, 12 * kib), 32 * kib, caps).1,
            16
        );
        assert!(!tile_plan(8, 6, 1000, 0, (32, 1024, 26 * kib), 32 * kib, caps).0);
        assert!(!tile_plan(8, 6, 1000, 0, caps, 32 * kib, (32, 512, 0)).0);
    }

    /// Tensor-op planner over synthetic limits: (supported, M, tile keys,
    /// threads, threadgroup bytes, partitions for `total` keys).
    fn nax_plan(
        rows: i32,
        gqa: i32,
        total: i32,
        blocks_override: i32,
        stage1: (usize, usize, usize),
        device_memory: usize,
        stage2: (usize, usize, usize),
    ) -> (bool, u32, u32, u32, u32, u32) {
        let mut out = [0u32; 5];
        let supported = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_verify_nax_plan(
                rows,
                gqa,
                total,
                blocks_override,
                stage1.0,
                stage1.1,
                stage1.2,
                device_memory,
                stage2.0,
                stage2.1,
                stage2.2,
                out.as_mut_ptr(),
            )
        } == 1;
        (supported, out[0], out[1], out[2], out[3], out[4])
    }

    /// The tensor-op verify kernel: 256 threads whatever M, M = gqa x rows a
    /// compiled multiple of 8 up to 64, the scratch (fp32 scores + BF16
    /// probabilities [M][tile_n], fp32 scale [M], two flags) within three
    /// quarters of the device's threadgroup memory limit choosing 64 or 32
    /// keys, and the tile kernel's partition policy at that tile size.
    #[test]
    fn segmented_verify_nax_planner_follows_device_limits() {
        // The tensor-op pipeline reports 256 threads; MLX's reduction 1024.
        let caps = (32, 256, 0);
        let red = (32, 1024, 0);
        let kib = 1024;
        let scratch = |m: u32, n: u32| m * n * 6 + m * 4 + 8;
        // 24 q heads / 4 kv heads, 8 rows: M = 48 -> 64 keys, 512 keys per
        // partition.
        assert_eq!(
            nax_plan(8, 6, 32776, 0, caps, 32 * kib, red),
            (true, 48, 64, 256, scratch(48, 64), 96)
        );
        assert_eq!(nax_plan(8, 6, 95, 0, caps, 32 * kib, red).5, 32);
        assert_eq!(nax_plan(8, 6, 16392, 0, caps, 32 * kib, red).5, 64);
        assert_eq!(nax_plan(8, 6, 1 << 20, 0, caps, 32 * kib, red).5, 1024);
        assert_eq!(nax_plan(8, 6, 32776, 100, caps, 32 * kib, red).5, 128);
        // M = 64 (gqa 8) exceeds the 64-key budget at 32 KiB: 32 keys and
        // 256 keys per partition; M = 56 still fits 64 keys.
        assert_eq!(
            nax_plan(8, 8, 32776, 0, caps, 32 * kib, red),
            (true, 64, 32, 256, scratch(64, 32), 160)
        );
        assert_eq!(nax_plan(7, 8, 1000, 0, caps, 32 * kib, red).2, 64);
        assert_eq!(nax_plan(8, 8, 1000, 0, caps, 64 * kib, red).2, 64);
        // 16 KiB: 64 keys up to M = 24, 32 keys to M = 56, nothing at 64;
        // 8 KiB: 32 keys up to M = 24.
        assert_eq!(nax_plan(4, 6, 1000, 0, caps, 16 * kib, red).2, 64);
        assert_eq!(nax_plan(8, 6, 1000, 0, caps, 16 * kib, red).2, 32);
        assert_eq!(nax_plan(7, 8, 1000, 0, caps, 16 * kib, red).2, 32);
        assert!(!nax_plan(8, 8, 1000, 0, caps, 16 * kib, red).0);
        assert!(!nax_plan(8, 6, 1000, 0, caps, 8 * kib, red).0);
        assert_eq!(nax_plan(4, 6, 1000, 0, caps, 8 * kib, red).2, 32);
        // Every compiled M: multiples of 8 from 8 (gqa 4 x 2) to 64.
        for rows in 2..=8 {
            for gqa in 1..=32 {
                let m = rows * gqa;
                let plan = nax_plan(rows, gqa, 1000, 0, caps, 32 * kib, red);
                assert_eq!(plan.0, m % 8 == 0 && m <= 64, "rows={rows} gqa={gqa}");
                if plan.0 {
                    assert_eq!(plan.1, m as u32);
                }
            }
        }
        assert!(!nax_plan(1, 8, 1000, 0, caps, 32 * kib, red).0, "rows < 2");
        // The pipeline's own limits rule: width, 256 threads, static memory.
        assert!(!nax_plan(8, 6, 1000, 0, (16, 256, 0), 32 * kib, red).0);
        assert!(!nax_plan(8, 6, 1000, 0, (32, 128, 0), 32 * kib, red).0);
        assert!(nax_plan(8, 6, 1000, 0, (32, 1024, 0), 32 * kib, red).0);
        assert_eq!(
            nax_plan(8, 6, 1000, 0, (32, 256, 8 * kib), 32 * kib, red).2,
            32
        );
        assert!(!nax_plan(8, 6, 1000, 0, (32, 256, 26 * kib), 32 * kib, red).0);
        assert!(!nax_plan(8, 6, 1000, 0, caps, 32 * kib, (32, 512, 0)).0);
    }

    /// `[B, H, prefix, D]` inputs for a verify block of `rows` over `prefix`
    /// keys from `[B, H, capacity, D]` caches with NaN past the prefix.
    #[cfg(target_os = "macos")]
    struct TileCase {
        q: MxArray,
        pk: MxArray,
        pv: MxArray,
        nk: MxArray,
        nv: MxArray,
    }

    #[cfg(target_os = "macos")]
    fn tile_case(
        base_k: &MxArray,
        base_v: &MxArray,
        q_heads: i64,
        rows: i64,
        prefix: i64,
        salt: u32,
        exponent_shift: u16,
    ) -> Result<TileCase> {
        const D: i64 = 256;
        let kv_heads = base_k.shape_at(1)?;
        let mut q_bits = deterministic_bf16((q_heads * rows * D) as usize, salt);
        // Adding to the exponent scales exactly: sharper softmax, larger
        // score magnitudes and more running-max changes between tiles.
        q_bits.iter_mut().for_each(|b| *b += exponent_shift << 7);
        Ok(TileCase {
            q: MxArray::from_bfloat16(&q_bits, &[1, q_heads, rows, D])?,
            pk: nan_tailed_prefix(base_k, prefix)?,
            pv: nan_tailed_prefix(base_v, prefix)?,
            nk: MxArray::from_bfloat16(
                &deterministic_bf16((kv_heads * rows * D) as usize, salt ^ 0x5a5a),
                &[1, kv_heads, rows, D],
            )?,
            nv: MxArray::from_bfloat16(
                &deterministic_bf16((kv_heads * rows * D) as usize, salt ^ 0xa5a5),
                &[1, kv_heads, rows, D],
            )?,
        })
    }

    #[cfg(target_os = "macos")]
    fn strict_tile_for_test(c: &TileCase) -> Result<MxArray> {
        let handle = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_forward_tile(
                c.q.as_raw_ptr(),
                c.pk.as_raw_ptr(),
                c.pv.as_raw_ptr(),
                c.nk.as_raw_ptr(),
                c.nv.as_raw_ptr(),
                0.0625,
            )
        };
        MxArray::from_handle(handle, "strict tile segmented SDPA test")
    }

    #[cfg(target_os = "macos")]
    fn strict_nax_for_test(c: &TileCase) -> Result<MxArray> {
        let handle = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_forward_nax(
                c.q.as_raw_ptr(),
                c.pk.as_raw_ptr(),
                c.pv.as_raw_ptr(),
                c.nk.as_raw_ptr(),
                c.nv.as_raw_ptr(),
                0.0625,
            )
        };
        MxArray::from_handle(handle, "strict tensor-op segmented SDPA test")
    }

    /// Whether this device plans the tensor-op kernel for a verify block of
    /// `rows` over `total` keys at `q_heads` / `kv_heads`; `plan` receives
    /// M, tile keys, threads, threadgroup bytes, partitions, pipeline max
    /// threads.
    #[cfg(target_os = "macos")]
    fn nax_supported(
        q_heads: i64,
        kv_heads: i64,
        rows: i64,
        total: i64,
        plan: &mut [u32; 6],
    ) -> bool {
        let supported = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_nax_plan(
                q_heads as i32,
                kv_heads as i32,
                rows as i32,
                total as i32,
                plan.as_mut_ptr(),
            )
        };
        supported == 1
    }

    /// BF16 spacing at `x` (subnormals share the smallest normal spacing).
    fn bf16_ulp(x: f32) -> f32 {
        let exponent = x.abs().max(f32::from_bits(0x0080_0000)).log2().floor();
        2f32.powf(exponent - 7.0)
    }

    /// Max and mean |diff| in BF16 ulps of the output's magnitude (its
    /// largest |expected|). The tile route rounds P to BF16, so its error
    /// is absolute at that scale rather than relative per element.
    fn tile_error(got: &[f32], expected: &[f32]) -> (f32, f64) {
        assert_eq!(got.len(), expected.len());
        let magnitude = expected.iter().fold(0f32, |m, e| m.max(e.abs()));
        let ulp = bf16_ulp(magnitude);
        let mut max_diff = 0f32;
        let mut sum = 0f64;
        for (&g, &e) in got.iter().zip(expected) {
            assert!(g.is_finite(), "non-finite tile output {g}");
            let diff = (g - e).abs();
            max_diff = max_diff.max(diff);
            sum += f64::from(diff);
        }
        (max_diff / ulp, sum / got.len() as f64 / f64::from(ulp))
    }

    /// The block routes sum in another order (fp32 MMAs, BF16 P), so they
    /// are not bit-identical to the vector route; each must stay within 2
    /// BF16 ulps of it over every prefix boundary and block width, with
    /// random and large-magnitude (peaked softmax) scores, on NaN-tailed
    /// caches and the production [B, T, H, D] layout. The tensor-op (NAX)
    /// route is checked on the same blocks wherever this device plans it
    /// (gen-17+, M = gqa x rows <= 64) and must refuse a Q whose heads are
    /// not contiguous with its rows. Runs by default on Metal.
    #[test]
    #[cfg(target_os = "macos")]
    fn segmented_verify_tile_matches_vector_route_within_tolerance() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            assert!(
                !crate::test_support::metal_required(),
                "MLX_TEST_REQUIRE_METAL=1 but no Metal device"
            );
            eprintln!("SKIP tile verify tolerance: no Metal device");
            return Ok(());
        }
        const D: i64 = 256;
        const HKV: i64 = 4;
        const CAPACITY: i64 = 32_768;
        let mut plan = [0u32; 5];
        let supported = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_tile_plan(24, HKV as i32, 8, 32_776, plan.as_mut_ptr())
        };
        if supported != 1 {
            assert!(
                !crate::test_support::metal_required(),
                "MLX_TEST_REQUIRE_METAL=1 but the tile route is unsupported ({supported})"
            );
            eprintln!("SKIP tile verify tolerance: this device cannot launch the tile kernel");
            return Ok(());
        }
        eprintln!(
            "tile plan 24/4 rows 8: tile_n={} threads={} tg_bytes={} partitions@32776={} \
             pipeline_max_threads={}",
            plan[0], plan[1], plan[2], plan[3], plan[4]
        );
        let base_k = MxArray::from_bfloat16(
            &deterministic_bf16((HKV * CAPACITY * D) as usize, 0x1234_5678),
            &[1, HKV, CAPACITY, D],
        )?;
        let base_v = MxArray::from_bfloat16(
            &deterministic_bf16((HKV * CAPACITY * D) as usize, 0x8765_4321),
            &[1, HKV, CAPACITY, D],
        )?;
        base_k.eval();
        base_v.eval();
        // The segmented primitive rejects an empty prefix (the Rust caller
        // concatenates there), so prefix 0 has no tile route to compare.
        let prefixes = [
            1_i64, 7, 31, 32, 33, 87, 1023, 1024, 4095, 4096, 6219, 32768,
        ];
        // gqa 6 tiles rows 4 and 8 (M % 8), gqa 8 every width 2..=8.
        let layouts: [(i64, Vec<i64>); 2] = [(24, vec![4, 8]), (32, (2..=8).collect())];
        let mut nax_plan = [0u32; 6];
        let nax_any = nax_supported(24, HKV, 8, 32_776, &mut nax_plan);
        if nax_any {
            eprintln!(
                "nax plan 24/4 rows 8: m={} tile_n={} threads={} tg_bytes={} partitions@32776={} \
                 pipeline_max_threads={}",
                nax_plan[0], nax_plan[1], nax_plan[2], nax_plan[3], nax_plan[4], nax_plan[5]
            );
        } else {
            eprintln!("nax route unsupported on this device; checking the tile route only");
        }
        let mut worst = (0f32, 0f64);
        let mut worst_nax = (0f32, 0f64);
        let mut checked = 0usize;
        let mut checked_nax = 0usize;
        for (q_heads, rows_list) in &layouts {
            for &rows in rows_list {
                for &prefix in &prefixes {
                    let nax_here =
                        nax_any && nax_supported(*q_heads, HKV, rows, prefix + rows, &mut nax_plan);
                    // Random, sharp (x8) and adversarial (x64: scores in
                    // the tens, a few keys own the softmax).
                    for (set, shift) in [0u16, 3, 6].into_iter().enumerate() {
                        let salt =
                            0x1357_9bdf ^ (prefix as u32 * 31) ^ (rows as u32) ^ (set as u32);
                        let c = tile_case(&base_k, &base_v, *q_heads, rows, prefix, salt, shift)?;
                        let got = strict_tile_for_test(&c)?.to_float32()?;
                        let expected = segmented_or_concat_split_for_test(
                            &c.q, &c.pk, &c.pv, &c.nk, &c.nv, true,
                        )?
                        .to_float32()?;
                        let (max_ulps, mean_ulps) = tile_error(got.as_ref(), expected.as_ref());
                        assert!(
                            max_ulps <= 2.0 && mean_ulps <= 0.25,
                            "q_heads={q_heads} rows={rows} prefix={prefix} set={set}: \
                             max {max_ulps} / mean {mean_ulps:.4} BF16 ulps of the output magnitude"
                        );
                        worst.0 = worst.0.max(max_ulps);
                        worst.1 = worst.1.max(mean_ulps);
                        checked += 1;
                        if nax_here {
                            let nax = strict_nax_for_test(&c)?.to_float32()?;
                            let (max_ulps, mean_ulps) = tile_error(nax.as_ref(), expected.as_ref());
                            assert!(
                                max_ulps <= 2.0 && mean_ulps <= 0.25,
                                "nax q_heads={q_heads} rows={rows} prefix={prefix} set={set}: \
                                 max {max_ulps} / mean {mean_ulps:.4} BF16 ulps of the output \
                                 magnitude"
                            );
                            worst_nax.0 = worst_nax.0.max(max_ulps);
                            worst_nax.1 = worst_nax.1.max(mean_ulps);
                            checked_nax += 1;
                        }
                    }
                }
            }
        }
        // Production layout: [B, T, H, D] projections transposed into
        // [B, H, T, D] views, batch 2, the prefix from a [B, P, H, D] cache.
        for prefix in [87_i64, 4096, 32766] {
            let q = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 8 * 32 * D) as usize, 0x90a0_b0c0),
                &[2, 8, 32, D],
            )?
            .transpose(Some(&[0, 2, 1, 3]))?;
            let nk = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 8 * 4 * D) as usize, 0xd0e0_f001),
                &[2, 8, 4, D],
            )?
            .transpose(Some(&[0, 2, 1, 3]))?;
            let nv = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 8 * 4 * D) as usize, 0x1234_abcd),
                &[2, 8, 4, D],
            )?
            .transpose(Some(&[0, 2, 1, 3]))?;
            let pk = MxArray::from_bfloat16(
                &deterministic_bf16((2 * prefix * 4 * D) as usize, 0x1020_3040),
                &[2, prefix, 4, D],
            )?
            .transpose(Some(&[0, 2, 1, 3]))?;
            let pv = MxArray::from_bfloat16(
                &deterministic_bf16((2 * prefix * 4 * D) as usize, 0x5060_7080),
                &[2, prefix, 4, D],
            )?
            .transpose(Some(&[0, 2, 1, 3]))?;
            let c = TileCase { q, pk, pv, nk, nv };
            let got = strict_tile_for_test(&c)?.to_float32()?;
            let expected =
                segmented_or_concat_split_for_test(&c.q, &c.pk, &c.pv, &c.nk, &c.nv, true)?
                    .to_float32()?;
            let (max_ulps, mean_ulps) = tile_error(got.as_ref(), expected.as_ref());
            assert!(
                max_ulps <= 2.0 && mean_ulps <= 0.25,
                "production layout prefix={prefix}: max {max_ulps} / mean {mean_ulps:.4} BF16 ulps"
            );
            worst.0 = worst.0.max(max_ulps);
            worst.1 = worst.1.max(mean_ulps);
            checked += 1;
            if nax_any {
                // A [B, T, H, D] view interleaves the heads of a row: the
                // tensor-op route refuses it (the production entry takes the
                // tile kernel there) ...
                assert!(
                    strict_nax_for_test(&c).is_err(),
                    "prefix={prefix}: the nax route accepted a Q with interleaved heads"
                );
                // ... and takes the contiguous [B, H, T, D] Q that RoPE
                // produces in the verify path, with the same K/V views.
                let q = MxArray::from_bfloat16(
                    &deterministic_bf16((2 * 32 * 8 * D) as usize, 0x90a0_b0c0),
                    &[2, 32, 8, D],
                )?;
                let c = TileCase {
                    q,
                    pk: c.pk,
                    pv: c.pv,
                    nk: c.nk,
                    nv: c.nv,
                };
                let nax = strict_nax_for_test(&c)?.to_float32()?;
                let expected =
                    segmented_or_concat_split_for_test(&c.q, &c.pk, &c.pv, &c.nk, &c.nv, true)?
                        .to_float32()?;
                let (max_ulps, mean_ulps) = tile_error(nax.as_ref(), expected.as_ref());
                assert!(
                    max_ulps <= 2.0 && mean_ulps <= 0.25,
                    "nax production K/V layout prefix={prefix}: max {max_ulps} / mean \
                     {mean_ulps:.4} BF16 ulps"
                );
                worst_nax.0 = worst_nax.0.max(max_ulps);
                worst_nax.1 = worst_nax.1.max(mean_ulps);
                checked_nax += 1;
            }
        }
        eprintln!(
            "tile verify: {checked} blocks within tolerance; worst max {} / mean {:.4} BF16 ulps \
             of the output magnitude",
            worst.0, worst.1
        );
        if nax_any {
            assert!(checked_nax > 0, "nax supported but no block took it");
            eprintln!(
                "nax verify: {checked_nax} blocks within tolerance; worst max {} / mean {:.4} \
                 BF16 ulps of the output magnitude",
                worst_nax.0, worst_nax.1
            );
        }
        Ok(())
    }

    /// The strict vector entry never dispatches a block kernel; the tile and
    /// tensor-op entries each dispatch their own kernel exactly once per call
    /// (plus MLX's reduction) and never the other's.
    #[test]
    #[cfg(target_os = "macos")]
    fn segmented_verify_tile_route_is_counted() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        const D: i64 = 256;
        let mut plan = [0u32; 5];
        if unsafe { mlx_sys::mlx_segmented_sdpa_test_tile_plan(24, 4, 8, 1008, plan.as_mut_ptr()) }
            != 1
        {
            eprintln!("SKIP tile route counters: this device cannot launch the tile kernel");
            return Ok(());
        }
        let base_k = MxArray::from_bfloat16(
            &deterministic_bf16((4 * 1100 * D) as usize, 1),
            &[1, 4, 1100, D],
        )?;
        let base_v = MxArray::from_bfloat16(
            &deterministic_bf16((4 * 1100 * D) as usize, 2),
            &[1, 4, 1100, D],
        )?;
        let c = tile_case(&base_k, &base_v, 24, 8, 1000, 3, 0)?;
        let count = |family: &std::ffi::CStr| unsafe {
            mlx_sys::mlx_test_kquant_family_count(family.as_ptr())
        };
        // Enabling the counters clears them.
        unsafe { mlx_sys::mlx_test_kquant_counting(true) };
        let tile = strict_tile_for_test(&c).and_then(|o| o.to_float32());
        let tile_counts = [
            count(c"segmented_sdpa_route_tile"),
            count(c"segmented_sdpa_verify_tile_2pass_1"),
            count(c"segmented_sdpa_2pass_2"),
        ];
        let tile_nax_counts = [
            count(c"segmented_sdpa_route_nax"),
            count(c"segmented_sdpa_verify_nax_2pass_1"),
        ];
        unsafe { mlx_sys::mlx_test_kquant_counting(false) };
        tile?;
        assert_eq!(tile_counts, [1, 1, 1], "tile route counters");
        assert_eq!(tile_nax_counts, [0, 0], "tile entry touched the nax route");
        let mut nax_plan = [0u32; 6];
        if nax_supported(24, 4, 8, 1008, &mut nax_plan) {
            unsafe { mlx_sys::mlx_test_kquant_counting(true) };
            let nax = strict_nax_for_test(&c).and_then(|o| o.to_float32());
            let nax_counts = [
                count(c"segmented_sdpa_route_nax"),
                count(c"segmented_sdpa_verify_nax_2pass_1"),
                count(c"segmented_sdpa_2pass_2"),
                count(c"segmented_sdpa_route_tile"),
                count(c"segmented_sdpa_verify_tile_2pass_1"),
            ];
            unsafe { mlx_sys::mlx_test_kquant_counting(false) };
            nax?;
            assert_eq!(nax_counts, [1, 1, 1, 0, 0], "nax route counters");
        } else {
            eprintln!("nax route unsupported on this device; tile counters only");
        }
        // Its own counting window: the vector entry never reaches a block
        // kernel.
        let (_, route) = strict_segmented_routed(&c.q, &c.pk, &c.pv, &c.nk, &c.nv)?;
        let after_vector = [
            count(c"segmented_sdpa_route_tile"),
            count(c"segmented_sdpa_verify_tile_2pass_1"),
            count(c"segmented_sdpa_route_nax"),
            count(c"segmented_sdpa_verify_nax_2pass_1"),
        ];
        assert_eq!(
            after_vector,
            [0, 0, 0, 0],
            "vector entry (route {route}) touched a block route"
        );
        Ok(())
    }

    /// The crossover selector over synthetic timings (pure, no GPU): the
    /// smallest count from which the block route stays at least as fast
    /// through the largest count; one past the largest when it never is; a
    /// win followed by a loss is not a crossover; non-positive or NaN
    /// timings lose; the result is clamped to [256, 8192].
    #[test]
    fn segmented_block_min_keys_selection_follows_timings() {
        fn select(keys: &[i32], vector: &[f64], block: &[f64]) -> i32 {
            assert_eq!(keys.len(), vector.len());
            assert_eq!(keys.len(), block.len());
            unsafe {
                mlx_sys::mlx_segmented_sdpa_test_select_block_min_keys(
                    keys.as_ptr(),
                    vector.as_ptr(),
                    block.as_ptr(),
                    keys.len(),
                )
            }
        }
        let keys = [256, 512, 1024, 2048, 4096];
        let vector = [10.0, 12.0, 20.0, 40.0, 80.0];
        // Equal at 1024, block ahead above: 2048 (a tie keeps the exact
        // vector route; the block route must win by kSegmentedBlockMinGain).
        assert_eq!(
            select(&keys, &vector, &[13.0, 14.0, 20.0, 30.0, 60.0]),
            2048
        );
        // Ahead at 1024 by less than the required gain: still 2048.
        assert_eq!(
            select(&keys, &vector, &[13.0, 14.0, 19.5, 30.0, 60.0]),
            2048
        );
        // Block ahead by the gain everywhere: the floor.
        assert_eq!(select(&keys, &vector, &[9.0, 11.0, 19.0, 30.0, 60.0]), 256);
        // Never ahead: one past the largest count.
        assert_eq!(
            select(&keys, &vector, &[11.0, 13.0, 21.0, 41.0, 81.0]),
            4097
        );
        // Ahead at 512, behind at 1024: 2048 (a win followed by a loss).
        assert_eq!(
            select(&keys, &vector, &[11.0, 11.0, 21.0, 30.0, 60.0]),
            2048
        );
        // A NaN or zero block sample is a loss at that point.
        assert_eq!(
            select(&keys, &vector, &[9.0, 11.0, f64::NAN, 30.0, 60.0]),
            2048
        );
        assert_eq!(select(&keys, &vector, &[9.0, 11.0, 19.0, 30.0, 0.0]), 4097);
        // So is a zero vector sample.
        assert_eq!(
            select(
                &keys,
                &[10.0, 12.0, 20.0, 0.0, 80.0],
                &[9.0, 11.0, 19.0, 30.0, 60.0]
            ),
            4096
        );
        // Clamp below the floor and above the ceiling.
        assert_eq!(select(&[64, 128], &[1.0, 2.0], &[0.5, 1.0]), 256);
        assert_eq!(select(&[8192, 16384], &[1.0, 2.0], &[2.0, 3.0]), 8192);
        assert_eq!(select(&[16384], &[1.0], &[0.5]), 8192);
        assert_eq!(select(&[], &[], &[]), 8192);
        assert_eq!(
            unsafe {
                mlx_sys::mlx_segmented_sdpa_test_select_block_min_keys(
                    std::ptr::null(),
                    vector.as_ptr(),
                    vector.as_ptr(),
                    keys.len(),
                )
            },
            -1
        );
    }

    /// This process calibrates the block-kernel crossover once on a
    /// production-shaped verify block; the result lies in the clamp range
    /// and the production entry switches exactly there: a rows-8 block over
    /// `min_keys - 1` keys takes a vector route, over `min_keys` keys a block
    /// kernel. Runs by default on Metal.
    #[test]
    #[cfg(target_os = "macos")]
    fn segmented_block_crossover_is_calibrated_and_routes() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        const D: i64 = 256;
        let mut keys = [0i32; 8];
        let mut vector_s = [0f64; 8];
        let mut block_s = [0f64; 8];
        let (mut elapsed_ms, mut kernel, mut result) = (0f64, 0i32, 0i32);
        let points = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_block_calibration(
                keys.as_mut_ptr(),
                vector_s.as_mut_ptr(),
                block_s.as_mut_ptr(),
                keys.len(),
                &mut elapsed_ms,
                &mut kernel,
                &mut result,
            )
        };
        assert!(points >= 0, "calibration query failed");
        let kernel_name = ["none", "tile", "nax"][kernel as usize];
        eprintln!(
            "block crossover: {result} keys ({kernel_name}), calibrated in {elapsed_ms:.3} ms"
        );
        for i in 0..points as usize {
            eprintln!(
                "  {:>5} keys: vector {:>8.2} us, block {:>8.2} us",
                keys[i],
                vector_s[i] * 1e6,
                block_s[i] * 1e6
            );
        }
        assert!(
            (256..=8192).contains(&result),
            "calibrated crossover {result} outside the clamp range"
        );
        assert!(
            points == 0 || elapsed_ms < 100.0,
            "calibration took {elapsed_ms:.3} ms"
        );
        let min_keys = i64::from(unsafe { mlx_sys::mlx_segmented_sdpa_test_block_min_keys() });
        assert_eq!(
            min_keys,
            i64::from(result),
            "effective crossover is the calibrated one"
        );
        let mut plan = [0u32; 5];
        if unsafe {
            mlx_sys::mlx_segmented_sdpa_test_tile_plan(24, 4, 8, min_keys as i32, plan.as_mut_ptr())
        } != 1
        {
            eprintln!("SKIP route switch: this device cannot launch a block kernel");
            return Ok(());
        }
        assert!(points > 0, "a device with a block kernel must calibrate");
        let base_k = MxArray::from_bfloat16(
            &deterministic_bf16((4 * min_keys * D) as usize, 0x7a7a),
            &[1, 4, min_keys, D],
        )?;
        let base_v = MxArray::from_bfloat16(
            &deterministic_bf16((4 * min_keys * D) as usize, 0x7b7b),
            &[1, 4, min_keys, D],
        )?;
        let count = |family: &std::ffi::CStr| unsafe {
            mlx_sys::mlx_test_kquant_family_count(family.as_ptr())
        };
        for (prefix, expect_block) in [(min_keys - 9, false), (min_keys - 8, true)] {
            let c = tile_case(&base_k, &base_v, 24, 8, prefix, 11, 0)?;
            unsafe { mlx_sys::mlx_test_kquant_counting(true) };
            let out = segmented_verify_sdpa(&c.q, &c.pk, &c.pv, &c.nk, &c.nv, 0.0625, true)
                .and_then(|o| o.to_float32());
            let block = count(c"segmented_sdpa_route_tile") + count(c"segmented_sdpa_route_nax");
            let vector = count(c"segmented_sdpa_route_single")
                + count(c"segmented_sdpa_route_one_pass")
                + count(c"segmented_sdpa_route_unified")
                + count(c"segmented_sdpa_route_split");
            unsafe { mlx_sys::mlx_test_kquant_counting(false) };
            out?;
            assert_eq!(
                (block, vector),
                (u64::from(expect_block), u64::from(!expect_block)),
                "routes at {} keys (crossover {min_keys})",
                prefix + 8
            );
        }
        Ok(())
    }

    /// MLX's vector-SDPA reduction policy per device class
    /// (`mlx_segmented_sdpa_plan.h`), used to predict the verify route
    /// independently of the C++ planner: (two-pass, partitions).
    #[cfg(target_os = "macos")]
    fn class_reduction(class: u8, total: i64, gqa: i64, active_simdgroups: i64) -> (bool, i64) {
        let large = class == b'd' || class == b's';
        if !((large && total >= 1024) || (gqa > 1 && total >= 4096)) {
            return (false, 32);
        }
        let partitions = match class {
            b's' if total > 1024 && active_simdgroups > 4 => match total {
                ..=8192 => 128,
                8193..=32768 => 256,
                32769..=65536 => 512,
                _ => 1024,
            },
            b's' => 64,
            b'd' if active_simdgroups <= 2 && total > 8192 => 256,
            b'd' if active_simdgroups >= 6 && (16384..65536).contains(&total) => 512,
            b'd' if active_simdgroups >= 6 && total >= 65536 => 1024,
            b'd' => 128,
            _ if active_simdgroups >= 4 => 64,
            _ => 32,
        };
        (true, partitions)
    }

    #[cfg(target_os = "macos")]
    fn device_verify_route(q_heads: i64, kv_heads: i64, rows: i64, prefix: i64) -> (i32, u8) {
        let mut class: std::ffi::c_char = 0;
        let route = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_device_verify_route(
                q_heads as i32,
                kv_heads as i32,
                rows as i32,
                prefix as i32,
                &mut class,
            )
        };
        (route, class as u8)
    }

    /// The verify route MLX's policy and this GPU's raw pipeline limits
    /// predict for a causal block, and whether both chunks reduce alike.
    #[cfg(target_os = "macos")]
    fn predicted_verify_route(
        class: u8,
        q_heads: i64,
        kv_heads: i64,
        rows: i64,
        prefix: i64,
        max_q: i64,
    ) -> (i32, bool) {
        let gqa = q_heads / kv_heads;
        let head_len = segmented_verify_head_len(rows, max_q)
            .unwrap_or_else(|| panic!("{rows} rows cannot be covered at max_q {max_q}"));
        if head_len == 0 {
            return (VERIFY_SINGLE, false);
        }
        let head = class_reduction(class, prefix + head_len, gqa, gqa * head_len);
        let tail = class_reduction(class, prefix + rows, gqa, gqa * (rows - head_len));
        let eligible = head.0 && tail.0 && head.1 == tail.1;
        let launchable = eligible && verify_unified_launchable(gqa, rows, tail.1);
        (verify_route(rows, max_q, head, tail, launchable), eligible)
    }

    /// Whether this GPU can serve a verify block in one dispatch, from the raw
    /// limits of its pipelines rather than the C++ launch planner. The verify
    /// kernel runs two (head, row) pairs per simdgroup (`PAIRS` in
    /// sdpa_segmented.metal), so 32 * gqa * rows / 2 threads; MLX's reduction
    /// kernel runs 1024.
    #[cfg(target_os = "macos")]
    fn verify_unified_launchable(gqa: i64, rows: i64, partitions: i64) -> bool {
        const PAIRS_PER_SIMDGROUP: i64 = 2;
        let mut limits = [0u64; 7];
        let status = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_verify_pipeline_limits(
                gqa as i32,
                rows as i32,
                partitions as i32,
                limits.as_mut_ptr(),
            )
        };
        assert_eq!(
            status, 0,
            "verify pipeline limits gqa {gqa} rows {rows} partitions {partitions}"
        );
        let [
            width,
            max_threads,
            static_memory,
            reduce_width,
            reduce_max_threads,
            reduce_static_memory,
            device_memory,
        ] = limits;
        let pairs = gqa * rows;
        let threads = 32 * (pairs / PAIRS_PER_SIMDGROUP) as u64;
        (2..=8).contains(&rows)
            && (1..=32).contains(&gqa)
            && pairs % PAIRS_PER_SIMDGROUP == 0
            && partitions >= 32
            && partitions % 32 == 0
            && width == 32
            && threads <= max_threads
            && static_memory <= device_memory
            && reduce_width == 32
            && reduce_max_threads >= 1024
            && reduce_static_memory <= device_memory
    }

    /// `strict_segmented_for_test` evaluated, and the verify route its
    /// `eval_gpu` dispatched, from the bridge kernel counters.
    #[cfg(target_os = "macos")]
    fn strict_segmented_routed(
        q: &MxArray,
        pk: &MxArray,
        pv: &MxArray,
        nk: &MxArray,
        nv: &MxArray,
    ) -> Result<(napi::bindgen_prelude::Float32Array, i32)> {
        let count = |family: &std::ffi::CStr| unsafe {
            mlx_sys::mlx_test_kquant_family_count(family.as_ptr())
        };
        unsafe { mlx_sys::mlx_test_kquant_counting(true) };
        let out = strict_segmented_for_test(q, pk, pv, nk, nv).and_then(|o| o.to_float32());
        let routes = [
            c"segmented_sdpa_route_single",
            c"segmented_sdpa_route_one_pass",
            c"segmented_sdpa_route_unified",
            c"segmented_sdpa_route_split",
        ]
        .map(count);
        let verify_kernels = count(c"segmented_sdpa_verify_2pass_1");
        unsafe { mlx_sys::mlx_test_kquant_counting(false) };
        let out = out?;
        assert_eq!(
            routes.iter().sum::<u64>(),
            1,
            "one verify dispatch, routes {routes:?}"
        );
        let route = routes.iter().position(|&n| n == 1).unwrap_or(0) as i32;
        assert_eq!(
            verify_kernels,
            u64::from(route == VERIFY_UNIFIED),
            "one-call verify kernel dispatches on route {route}"
        );
        Ok((out, route))
    }

    #[cfg(target_os = "macos")]
    fn strict_segmented_for_test(
        q: &MxArray,
        pk: &MxArray,
        pv: &MxArray,
        nk: &MxArray,
        nv: &MxArray,
    ) -> Result<MxArray> {
        let handle = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_forward(
                q.as_raw_ptr(),
                pk.as_raw_ptr(),
                pv.as_raw_ptr(),
                nk.as_raw_ptr(),
                nv.as_raw_ptr(),
                0.0625,
                true,
            )
        };
        MxArray::from_handle(handle, "strict one-call segmented SDPA test")
    }

    /// `[B, H, prefix, D]` view of a `[B, H, prefix + 64, D]` cache whose rows
    /// past the prefix are NaN, as the compiled verify slices its KV cache.
    #[cfg(target_os = "macos")]
    fn nan_tailed_prefix(base: &MxArray, prefix: i64) -> Result<MxArray> {
        let batch = base.shape_at(0)?;
        let heads = base.shape_at(1)?;
        let d = base.shape_at(3)?;
        let nan = MxArray::from_bfloat16(
            &vec![0x7fc0_u16; (batch * heads * 64 * d) as usize],
            &[batch, heads, 64, d],
        )?;
        let cache = MxArray::concatenate(&base.slice_axis(2, 0, prefix)?, &nan, 2)?;
        cache.eval();
        cache.slice_axis(2, 0, prefix)
    }

    /// One call over the whole verify block must equal both the per-chunk
    /// segmented calls and the independent concat + MLX vector SDPA chunks,
    /// for the request-tail widths 6 and 7 and the full width 8, on every
    /// route. On class 's' the chosen route is asserted per (rows, prefix) so
    /// the one-call kernel cannot pass vacuously.
    #[test]
    #[ignore = "requires coordinated Metal GPU validation"]
    #[cfg(target_os = "macos")]
    fn segmented_verify_one_call_matches_split_bf16_exactly() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        const D: i64 = 256;
        const HQ: i64 = 24;
        const HKV: i64 = 4;
        let gqa = HQ / HKV;
        let max_q = i64::from(unsafe { mlx_sys::mlx_segmented_sdpa_max_query_length(gqa as i32) });
        eprintln!("segmented SDPA GQA={gqa} max_query_length={max_q}");
        let prefixes = [
            1_i64, 87, 1000, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 4095,
            4096, 6219, 8184, 8185, 8186, 8187, 8189, 8190, 8191, 8192, 16384, 32488, 32760, 32761,
            32762, 32763, 32765, 32766, 32767, 32768, 33000,
        ];
        let max_prefix = *prefixes.iter().max().unwrap_or(&1);
        let kv_elements = (HKV * max_prefix * D) as usize;
        let base_k = MxArray::from_bfloat16(
            &deterministic_bf16(kv_elements, 0x1234_5678),
            &[1, HKV, max_prefix, D],
        )?;
        let base_v = MxArray::from_bfloat16(
            &deterministic_bf16(kv_elements, 0x8765_4321),
            &[1, HKV, max_prefix, D],
        )?;
        let mut checked = 0usize;
        for rows in [6_i64, 7, 8] {
            if segmented_verify_head_len(rows, max_q).is_none() {
                eprintln!("SKIP rows={rows}: max_query_length={max_q} cannot cover the block");
                continue;
            }
            let q_bits = deterministic_bf16((HQ * rows * D) as usize, 0x1357_9bdf);
            // x8 adds 3 to every exponent: exact in BF16, sharper softmax.
            let q_sharp_bits: Vec<u16> = q_bits.iter().map(|&b| b + 0x0180).collect();
            let queries = [
                MxArray::from_bfloat16(&q_bits, &[1, HQ, rows, D])?,
                MxArray::from_bfloat16(&q_sharp_bits, &[1, HQ, rows, D])?,
            ];
            let nk = MxArray::from_bfloat16(
                &deterministic_bf16((HKV * rows * D) as usize, 0x2468_ace0),
                &[1, HKV, rows, D],
            )?;
            let nv = MxArray::from_bfloat16(
                &deterministic_bf16((HKV * rows * D) as usize, 0xfdb9_7531),
                &[1, HKV, rows, D],
            )?;
            let mut seen = [0usize; 4];
            let mut class_s = false;
            let mut unified_eligible = 0usize;
            for prefix in prefixes {
                let (route, class) = device_verify_route(HQ, HKV, rows, prefix);
                assert!(
                    route >= 0,
                    "no segmented route for rows={rows}, prefix={prefix}"
                );
                seen[route as usize] += 1;
                if class == b's' {
                    class_s = true;
                    let (expected, eligible) =
                        predicted_verify_route(b's', HQ, HKV, rows, prefix, max_q);
                    unified_eligible += usize::from(eligible);
                    assert_eq!(route, expected, "route for rows={rows}, prefix={prefix}");
                }
                let pk = nan_tailed_prefix(&base_k, prefix)?;
                let pv = nan_tailed_prefix(&base_v, prefix)?;
                for (set, q) in queries.iter().enumerate() {
                    let (got, dispatched) = strict_segmented_routed(q, &pk, &pv, &nk, &nv)?;
                    assert_eq!(
                        dispatched, route,
                        "dispatched route for rows={rows}, prefix={prefix}"
                    );
                    let chunks = segmented_or_concat_split_for_test(q, &pk, &pv, &nk, &nv, true)?
                        .to_float32()?;
                    let concat = segmented_or_concat_split_for_test(q, &pk, &pv, &nk, &nv, false)?
                        .to_float32()?;
                    assert!(got.iter().all(|x| x.is_finite()), "non-finite output");
                    assert_eq!(
                        got.as_ref(),
                        chunks.as_ref(),
                        "one call vs segmented chunks: rows={rows}, prefix={prefix}, set={set}, route={route}"
                    );
                    assert_eq!(
                        got.as_ref(),
                        concat.as_ref(),
                        "one call vs concat SDPA chunks: rows={rows}, prefix={prefix}, set={set}, route={route}"
                    );
                    checked += 1;
                }
            }
            eprintln!(
                "rows={rows} routes single={} one_pass={} unified={} split={}",
                seen[0], seen[1], seen[2], seen[3]
            );
            if class_s {
                assert!(unified_eligible > 0, "rows={rows}: no one-call prefix");
                for route in [VERIFY_ONE_PASS, VERIFY_SPLIT] {
                    assert!(
                        seen[route as usize] > 0,
                        "rows={rows} never took route {route}"
                    );
                }
            }
        }
        eprintln!("checked {checked} one-call cases");

        // Production layout: projections leave [B,T,H,D] and are transposed
        // into [B,H,T,D] views; non-unit batch and several KV heads.
        if segmented_verify_head_len(8, max_q).is_none() {
            return Ok(());
        }
        for prefix in [87_i64, 4096, 32766] {
            let pk = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 2 * prefix * D) as usize, 0x1020_3040),
                &[2, 2, prefix, D],
            )?;
            let pv = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 2 * prefix * D) as usize, 0x5060_7080),
                &[2, 2, prefix, D],
            )?;
            let q = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 8 * 12 * D) as usize, 0x90a0_b0c0),
                &[2, 8, 12, D],
            )?
            .transpose(Some(&[0, 2, 1, 3]))?;
            let nk = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 8 * 2 * D) as usize, 0xd0e0_f001),
                &[2, 8, 2, D],
            )?
            .transpose(Some(&[0, 2, 1, 3]))?;
            let nv = MxArray::from_bfloat16(
                &deterministic_bf16((2 * 8 * 2 * D) as usize, 0x1234_abcd),
                &[2, 8, 2, D],
            )?
            .transpose(Some(&[0, 2, 1, 3]))?;
            let (route, _) = device_verify_route(12, 2, 8, prefix);
            let got = strict_segmented_for_test(&q, &pk, &pv, &nk, &nv)?.to_float32()?;
            let chunks =
                segmented_or_concat_split_for_test(&q, &pk, &pv, &nk, &nv, true)?.to_float32()?;
            let concat =
                segmented_or_concat_split_for_test(&q, &pk, &pv, &nk, &nv, false)?.to_float32()?;
            assert_eq!(
                got.as_ref(),
                chunks.as_ref(),
                "production layout prefix={prefix} route={route}"
            );
            assert_eq!(
                got.as_ref(),
                concat.as_ref(),
                "production layout prefix={prefix} route={route}"
            );
        }
        Ok(())
    }

    /// The route is chosen from the real prefix at eval time, so one shapeless
    /// trace must stay exact while replays cross every route boundary. The
    /// production entry takes a block kernel (tensor-op where NAX exists,
    /// else the simdgroup tile) from the key count this process calibrated
    /// (`mlx_segmented_sdpa_test_block_min_keys`; rows 8 here; rows 6 and 7
    /// are not multiples of 8 queries): the replay must equal the eager
    /// production call bit for bit on either side, and the vector chunks bit
    /// for bit on the vector routes or within the tile tolerance on a block
    /// route.
    #[test]
    #[ignore = "requires coordinated Metal GPU validation"]
    #[cfg(target_os = "macos")]
    fn segmented_verify_shapeless_replay_crosses_routes() -> Result<()> {
        use crate::compiled_graph::invoke_compiled_graph;
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        const D: i64 = 256;
        const HQ: i64 = 24;
        const HKV: i64 = 4;
        let max_q = i64::from(unsafe { mlx_sys::mlx_segmented_sdpa_max_query_length(6) });
        // Calibrates before any counting window opens below.
        let block_min_keys =
            i64::from(unsafe { mlx_sys::mlx_segmented_sdpa_test_block_min_keys() });
        assert!(block_min_keys >= 0, "block crossover query failed");
        // Rows-8 prefixes straddling the crossover (totals min_keys - 10 ..=
        // min_keys + 10), then the vector policy's own 8192 / 32768 steps.
        let crossover_prefixes: Vec<i64> = if block_min_keys > 0 {
            ((block_min_keys - 8 - 10).max(1)..=block_min_keys - 8 + 10).collect()
        } else {
            (1010..=1030).collect()
        };
        let prefixes: Vec<i64> = crossover_prefixes
            .iter()
            .copied()
            .chain(8180..=8195)
            .chain(32755..=32770)
            .collect();
        let max_prefix = 32770;
        let kv_elements = (HKV * max_prefix * D) as usize;
        let base_k = MxArray::from_bfloat16(
            &deterministic_bf16(kv_elements, 0x0bad_cafe),
            &[1, HKV, max_prefix, D],
        )?;
        let base_v = MxArray::from_bfloat16(
            &deterministic_bf16(kv_elements, 0x0dea_dbee),
            &[1, HKV, max_prefix, D],
        )?;
        for rows in [6_i64, 7, 8] {
            if segmented_verify_head_len(rows, max_q).is_none() {
                eprintln!("SKIP rows={rows}: max_query_length={max_q}");
                continue;
            }
            let q = MxArray::from_bfloat16(
                &deterministic_bf16((HQ * rows * D) as usize, 0x2233_4455),
                &[1, HQ, rows, D],
            )?;
            let nk = MxArray::from_bfloat16(
                &deterministic_bf16((HKV * rows * D) as usize, 0x6677_8899),
                &[1, HKV, rows, D],
            )?;
            let nv = MxArray::from_bfloat16(
                &deterministic_bf16((HKV * rows * D) as usize, 0xaabb_ccdd),
                &[1, HKV, rows, D],
            )?;
            let fn_id = 0xDFC5_0000_0000_0000_u64 | rows as u64;
            let builds = std::cell::Cell::new(0usize);
            let mut builder = |inputs: &[MxArray]| -> Result<Vec<MxArray>> {
                builds.set(builds.get() + 1);
                Ok(vec![segmented_verify_sdpa(
                    &inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], 0.0625, true,
                )?])
            };
            let mut seen = [0usize; 4];
            let mut seen_tile = 0usize;
            for &prefix in &prefixes {
                let pk = nan_tailed_prefix(&base_k, prefix)?;
                let pv = nan_tailed_prefix(&base_v, prefix)?;
                // Enabling the counters clears them; the replay evaluates on
                // this thread.
                unsafe { mlx_sys::mlx_test_kquant_counting(true) };
                let compiled =
                    invoke_compiled_graph(fn_id, &[&q, &pk, &pv, &nk, &nv], 1, true, &mut builder)
                        .and_then(|out| {
                            out.ok_or_else(|| Error::from_reason("compiled invoke failed"))
                        })
                        .and_then(|out| out[0].to_float32());
                // Either block kernel (tile or tensor-op) is held to the
                // tile tolerance.
                let tile_route = unsafe {
                    mlx_sys::mlx_test_kquant_family_count(c"segmented_sdpa_route_tile".as_ptr())
                        + mlx_sys::mlx_test_kquant_family_count(
                            c"segmented_sdpa_route_nax".as_ptr(),
                        )
                } > 0;
                unsafe { mlx_sys::mlx_test_kquant_counting(false) };
                let compiled = compiled?;
                let eager =
                    segmented_verify_sdpa(&q, &pk, &pv, &nk, &nv, 0.0625, true)?.to_float32()?;
                let chunks = segmented_or_concat_split_for_test(&q, &pk, &pv, &nk, &nv, true)?
                    .to_float32()?;
                let (route, _) = device_verify_route(HQ, HKV, rows, prefix);
                assert_eq!(
                    compiled.as_ref(),
                    eager.as_ref(),
                    "compiled vs eager: rows={rows}, prefix={prefix}, tile={tile_route}"
                );
                if tile_route {
                    seen_tile += 1;
                    let (max_ulps, mean_ulps) = tile_error(compiled.as_ref(), chunks.as_ref());
                    assert!(
                        max_ulps <= 2.0 && mean_ulps <= 0.25,
                        "rows={rows}, prefix={prefix}: tile replay max {max_ulps} / mean \
                         {mean_ulps:.4} ulps"
                    );
                } else {
                    seen[route.max(0) as usize] += 1;
                    assert_eq!(
                        compiled.as_ref(),
                        chunks.as_ref(),
                        "rows={rows}, prefix={prefix}, route={route}"
                    );
                }
            }
            assert_eq!(
                builds.get(),
                1,
                "rows={rows}: the prefix must stay out of the trace key"
            );
            eprintln!(
                "rows={rows} replay routes one_pass={} unified={} split={} tile={seen_tile}",
                seen[1], seen[2], seen[3]
            );
            let mut plan = [0u32; 5];
            let tile_supported = unsafe {
                mlx_sys::mlx_segmented_sdpa_test_tile_plan(
                    HQ as i32,
                    HKV as i32,
                    rows as i32,
                    2048,
                    plan.as_mut_ptr(),
                )
            } == 1;
            if tile_supported && block_min_keys > 0 {
                // The crossover prefixes straddle `block_min_keys`.
                assert!(seen_tile > 0, "rows={rows}: no replay took the tile route");
                assert!(
                    seen[1] + seen[2] + seen[3] > 0,
                    "rows={rows}: no replay took a vector route"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "manual rotating-prefix SDPA benchmark; run without other GPU work"]
    #[cfg(target_os = "macos")]
    fn segmented_sdpa_rotating_prefix_benchmark() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            eprintln!("SKIP segmented benchmark: Metal backend unavailable");
            return Ok(());
        }
        let max_query_length =
            i64::from(unsafe { mlx_sys::mlx_segmented_sdpa_max_query_length(6) });
        if segmented_verify_head_len(8, max_query_length).is_none() {
            eprintln!(
                "SKIP segmented benchmark: q=8 requires two supported chunks; \
                 GQA=6 pipeline supports at most {max_query_length} rows per chunk"
            );
            return Ok(());
        }
        const D: i64 = 256;
        const SETS: usize = 4;
        const SAMPLES: usize = 24;
        // Split chunks, the one-call vector route, the simdgroup-matrix tile
        // route and the tensor-op (NAX) route (each block route skipped where
        // this device cannot launch it).
        const ROUTES: [&str; 4] = ["split", "one_call", "tile", "nax"];
        let mut plan = [0u32; 5];
        let tile_supported = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_tile_plan(24, 4, 8, 32_776, plan.as_mut_ptr())
        } == 1;
        if tile_supported {
            eprintln!("tile pipeline maxTotalThreadsPerThreadgroup={}", plan[4]);
        } else {
            eprintln!("tile route unsupported on this device; timing the vector routes only");
        }
        let mut nax_plan = [0u32; 6];
        let nax_supported_here = tile_supported && nax_supported(24, 4, 8, 32_776, &mut nax_plan);
        if nax_supported_here {
            eprintln!(
                "nax pipeline m={} tile_n={} maxTotalThreadsPerThreadgroup={}",
                nax_plan[0], nax_plan[1], nax_plan[5]
            );
        } else {
            eprintln!("nax route unsupported on this device");
        }
        let routes = 2 + usize::from(tile_supported) + usize::from(nax_supported_here);
        for prefix in [87_i64, 1024, 6219, 16384, 32768] {
            let mut cases = Vec::with_capacity(SETS);
            for i in 0..SETS {
                let salt = 0x1234_5678u32.wrapping_add(i as u32 * 29);
                let q = MxArray::from_bfloat16(
                    &deterministic_bf16((24 * 8 * D) as usize, salt),
                    &[1, 24, 8, D],
                )?;
                let pk = MxArray::from_bfloat16(
                    &deterministic_bf16((4 * prefix * D) as usize, salt ^ 0x12),
                    &[1, 4, prefix, D],
                )?;
                let pv = MxArray::from_bfloat16(
                    &deterministic_bf16((4 * prefix * D) as usize, salt ^ 0x34),
                    &[1, 4, prefix, D],
                )?;
                let nk = MxArray::from_bfloat16(
                    &deterministic_bf16((4 * 8 * D) as usize, salt ^ 0x56),
                    &[1, 4, 8, D],
                )?;
                let nv = MxArray::from_bfloat16(
                    &deterministic_bf16((4 * 8 * D) as usize, salt ^ 0x78),
                    &[1, 4, 8, D],
                )?;
                cases.push(TileCase { q, pk, pv, nk, nv });
            }
            let run = |i: usize, route: usize| -> Result<MxArray> {
                let c = &cases[i % SETS];
                let out = match route {
                    0 => {
                        segmented_or_concat_split_for_test(&c.q, &c.pk, &c.pv, &c.nk, &c.nv, true)?
                    }
                    1 => strict_segmented_for_test(&c.q, &c.pk, &c.pv, &c.nk, &c.nv)?,
                    2 => strict_tile_for_test(c)?,
                    _ => strict_nax_for_test(c)?,
                };
                MxArray::eval_arrays(&[&out])?;
                Ok(out)
            };
            for i in 0..SETS * 2 {
                let before = run(i, 0)?.to_float32()?;
                let after = run(i, 1)?.to_float32()?;
                assert_eq!(before.as_ref(), after.as_ref(), "prefix={prefix}, set={i}");
                for route in 2..routes {
                    let block = run(i, route)?.to_float32()?;
                    let (max_ulps, mean_ulps) = tile_error(block.as_ref(), before.as_ref());
                    assert!(
                        max_ulps <= 2.0 && mean_ulps <= 0.25,
                        "prefix={prefix}, set={i}: {} max {max_ulps} / mean {mean_ulps:.4} ulps",
                        ROUTES[route]
                    );
                }
            }
            let mut ms = vec![Vec::with_capacity(SAMPLES); routes];
            let mut tile_plan = [0u32; 5];
            if tile_supported {
                unsafe {
                    mlx_sys::mlx_segmented_sdpa_test_tile_plan(
                        24,
                        4,
                        8,
                        (prefix + 8) as i32,
                        tile_plan.as_mut_ptr(),
                    )
                };
            }
            if nax_supported_here {
                nax_supported(24, 4, 8, prefix + 8, &mut nax_plan);
            }
            for i in 0..SAMPLES {
                // Rotate the order so no route is always the second reader of
                // a just-warmed prefix. Rotate four independent KV sets.
                for k in 0..routes {
                    let route = (k + i) % routes;
                    let started = Instant::now();
                    let out = run(i, route)?;
                    ms[route].push(started.elapsed().as_secs_f64() * 1e3);
                    drop(out);
                }
            }
            let mut line =
                format!("segmented prefix={prefix} q=8 hq=24 hkv=4 sets={SETS} samples={SAMPLES}");
            for (route, samples) in ms.iter_mut().enumerate() {
                samples.sort_by(f64::total_cmp);
                line += &format!(
                    " {0}_min_ms={1:.4} {0}_median_ms={2:.4}",
                    ROUTES[route],
                    samples[0],
                    samples[SAMPLES / 2]
                );
            }
            if tile_supported {
                line += &format!(" tile_partitions={}", tile_plan[3]);
            }
            if nax_supported_here {
                line += &format!(" nax_partitions={}", nax_plan[4]);
            }
            eprintln!("{line}");
        }
        Ok(())
    }

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
    fn m3_pro_recorded_continuation_uses_portable_sdpa_within_headroom() {
        // Supplied M3 Pro trace: 1,012-token materialized chunk following
        // 78,012 cached tokens; 3,318.9 MiB of Metal headroom. The old route
        // estimated 4,389.7 MiB for SDPA and chose 1,942.8 MiB varlen scratch.
        let portable =
            estimate_paged_pool_sdpa_bytes_with_portable(1_012, 79_024, 24, 4, 256, 2, false, true);
        let old = estimate_paged_pool_sdpa_bytes(1_012, 79_024, 24, 4, 256, 2, false);
        let varlen = estimate_varlen_paged_attention_bytes(1_012, 79_024, 24, 4, 256, 2);
        let headroom = Some((3318.9 * 1024.0 * 1024.0) as u64);
        assert!(portable < 710 * 1024 * 1024);
        assert!(old > 4 * 1024 * 1024 * 1024);
        assert_eq!(
            select_cache_hit_prefill_plan(CacheHitPrefillMode::Auto, 1_012, old, varlen, headroom)
                .path,
            CacheHitPrefillPath::PagedVarlen
        );
        assert_eq!(
            select_cache_hit_prefill_plan(
                CacheHitPrefillMode::Auto,
                1_012,
                portable,
                varlen,
                headroom
            )
            .path,
            CacheHitPrefillPath::PagedPoolSdpa
        );
    }

    #[test]
    fn portable_estimate_covers_ragged_continuations() {
        for query in [9_u64, 31, 32, 33, 531, 1_012, 1_024, 2_049] {
            let total = 85_501 + query;
            let estimate = estimate_paged_pool_sdpa_bytes_with_portable(
                query, total, 24, 4, 256, 2, false, true,
            );
            let expected = total * 4 * 256 * 2 * 4
                + query * 24 * 256 * 2 * 2
                + PREFILL_ESTIMATE_FIXED_OVERHEAD_BYTES;
            assert_eq!(estimate, expected);
        }
        for (query, dim, dtype) in [(8, 256, 2), (33, 256, 4), (33, 512, 2)] {
            assert_eq!(
                estimate_paged_pool_sdpa_bytes_with_portable(
                    query, 4097, 24, 4, dim, dtype, false, true
                ),
                estimate_paged_pool_sdpa_bytes(query, 4097, 24, 4, dim, dtype, false),
            );
        }
    }

    #[test]
    fn qwen27b_fused_d256_estimate_drops_scores_only_at_supported_boundary() {
        let unfused = estimate_paged_pool_sdpa_bytes(2_048, 64_754, 24, 4, 256, 2, false);
        let fused = estimate_paged_pool_sdpa_bytes(2_048, 64_754, 24, 4, 256, 2, true);
        assert!(fused < unfused / 4, "fused={fused} unfused={unfused}");

        let one_kv = 64_754_u64 * 4 * 256 * 2;
        let one_query = 2_048_u64 * 24 * 256 * 2;
        assert_eq!(
            fused,
            one_kv * 4 + one_query * 2 + PREFILL_ESTIMATE_FIXED_OVERHEAD_BYTES
        );

        // MLX's NAX D=256 kernel reads ragged tiles in place: no padded copies.
        let ragged_query = estimate_paged_pool_sdpa_bytes(1_031, 4_129, 24, 4, 256, 2, true);
        assert_eq!(
            ragged_query,
            4_129_u64 * 4 * 256 * 2 * 4
                + 1_031_u64 * 24 * 256 * 2 * 2
                + PREFILL_ESTIMATE_FIXED_OVERHEAD_BYTES
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

    #[test]
    fn prism_hadamard_paged_operands_cast_only_rotated_fp32() -> Result<()> {
        use crate::quant::prism_hadamard::HadamardTransform;
        let cfg = Qwen3_5Config {
            hidden_size: 1024,
            ..tiny_cfg()
        };
        let mut attention = Qwen3_5Attention::new(&cfg)?;
        let x = MxArray::from_float32(&[0.1, -0.2, 0.3, 0.4, -0.5, 0.6], &[1, 2, 3])?;
        assert_eq!(
            attention
                .paged_attention_operand(&x, Some(DType::BFloat16))?
                .dtype()?,
            DType::Float32
        );
        let projection = QuantizedLinear::new(
            MxArray::zeros(&[64, 64], Some(DType::Uint32))?,
            MxArray::from_float16(&vec![0x3800; 64 * 8], &[64, 8])?,
            Some(MxArray::from_float16(&vec![0xb800; 64 * 8], &[64, 8])?),
            None,
            128,
            2,
            "affine".to_string(),
        )
        .with_hadamard(Some(HadamardTransform {
            signs: MxArray::from_float32(&vec![1.0; 1024], &[1024])?,
            block_size: 1024,
            gdn_permutation: None,
        }))?;
        attention.set_quantized_q_proj(projection);
        for dtype in [DType::Float16, DType::BFloat16] {
            let actual = attention.paged_attention_operand(&x, Some(dtype))?;
            assert_eq!(actual.dtype()?, dtype);
            assert_eq!(actual.shape()?.as_ref(), x.shape()?.as_ref());
            assert_eq!(
                actual.astype(DType::Float32)?.to_float32()?.as_ref(),
                x.astype(dtype)?
                    .astype(DType::Float32)?
                    .to_float32()?
                    .as_ref()
            );
        }
        let half = x.astype(DType::Float16)?;
        assert_eq!(
            attention
                .paged_attention_operand(&half, Some(DType::BFloat16))?
                .dtype()?,
            DType::Float16
        );
        assert!(attention.paged_attention_operand(&x, None).is_err());
        assert!(
            attention
                .paged_attention_operand(&x, Some(DType::Float32))
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn prism_hadamard_paged_operands_cast_for_dense_attention() -> Result<()> {
        let mut attention = Qwen3_5Attention::new(&tiny_cfg())?;
        let x = MxArray::from_float32(&[0.1, -0.2, 0.3, 0.4], &[1, 1, 4])?;
        assert_eq!(
            attention.prism_hadamard_sites(),
            (false, false, false, false)
        );
        attention.set_prism_model(true);
        for dtype in [DType::Float16, DType::BFloat16] {
            let actual = attention.paged_attention_operand(&x, Some(dtype))?;
            assert_eq!(actual.dtype()?, dtype);
            assert_eq!(actual.shape()?.as_ref(), x.shape()?.as_ref());
            assert_eq!(
                actual.astype(DType::Float32)?.to_float32()?.as_ref(),
                x.astype(dtype)?
                    .astype(DType::Float32)?
                    .to_float32()?
                    .as_ref()
            );
            let half = x.astype(dtype)?;
            assert_eq!(
                attention
                    .paged_attention_operand(&half, Some(dtype))?
                    .as_raw_ptr(),
                half.as_raw_ptr()
            );
        }
        assert!(attention.paged_attention_operand(&x, None).is_err());
        assert!(
            attention
                .paged_attention_operand(&x, Some(DType::Float32))
                .is_err()
        );
        attention.set_prism_model(false);
        assert_eq!(
            attention
                .paged_attention_operand(&x, Some(DType::BFloat16))?
                .as_raw_ptr(),
            x.as_raw_ptr()
        );
        Ok(())
    }

    pub(super) fn tiny_cfg() -> Qwen3_5Config {
        Qwen3_5Config {
            qwen35_gguf_gdn_layout: None,
            kv_format: None,
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
            paged_cache_initial_memory_mb: None,
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

    fn quantize_affine(weight: &MxArray) -> Result<(MxArray, MxArray, MxArray)> {
        let mut out_q: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_s: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_b: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let ok = unsafe {
            mlx_sys::mlx_quantize(
                weight.as_raw_ptr(),
                32,
                4,
                c"affine".as_ptr(),
                &mut out_q,
                &mut out_s,
                &mut out_b,
            )
        };
        if !ok {
            return Err(Error::from_reason("mlx_quantize affine failed"));
        }
        Ok((
            MxArray::from_handle(out_q, "q_gate_test_weight")?,
            MxArray::from_handle(out_s, "q_gate_test_scales")?,
            MxArray::from_handle(out_b, "q_gate_test_biases")?,
        ))
    }

    #[test]
    fn affine_quantized_q_gate_block_matches_native_h4_with_bias() -> Result<()> {
        let cfg = tiny_cfg();
        assert!(
            cfg.num_heads > 1,
            "fixture must cover the multi-head permutation"
        );
        let h = cfg.num_heads as i64;
        let d = cfg.head_dim as i64;
        let hidden = cfg.hidden_size as i64;
        let rows = 2 * h * d;

        let dense_values = (0..rows * hidden)
            .map(|i| ((i as f32) * 0.017 + 0.3).sin() * 0.5)
            .collect::<Vec<_>>();
        let dense =
            MxArray::from_float32(&dense_values, &[rows, hidden])?.astype(DType::BFloat16)?;
        let (weight, scales, quant_biases) = quantize_affine(&dense)?;
        let additive_values = (0..rows)
            .map(|i| ((i as f32) * 0.031 - 0.2).cos() * 0.1)
            .collect::<Vec<_>>();
        let additive_bias =
            MxArray::from_float32(&additive_values, &[rows])?.astype(DType::BFloat16)?;

        let make_q_proj = || {
            QuantizedLinear::new(
                weight.clone(),
                scales.clone(),
                Some(quant_biases.clone()),
                Some(additive_bias.clone()),
                32,
                4,
                crate::models::quantized_linear::DEFAULT_QUANT_MODE.to_string(),
            )
        };
        let mut block = Qwen3_5Attention::new(&cfg)?;
        let mut native = Qwen3_5Attention::new(&cfg)?;
        block.set_quantized_q_proj(make_q_proj());
        native.set_quantized_q_proj(make_q_proj());
        let block_original_weight = block.get_q_proj_weight().as_raw_ptr();
        let native_original_weight = native.get_q_proj_weight().as_raw_ptr();

        block.finalize_q_gate_block()?;
        // The independent reference keeps its original unfinalized layout.
        assert!(block.q_proj.has_q_gate_block_layout());
        assert!(!native.q_proj.has_q_gate_block_layout());
        assert!(
            block.q_gate_block_t.is_none(),
            "quantized block order must replace operands, not retain a dense cache"
        );
        assert_ne!(
            block.get_q_proj_weight().as_raw_ptr(),
            block_original_weight
        );
        assert_eq!(
            native.get_q_proj_weight().as_raw_ptr(),
            native_original_weight
        );

        let x_values = (0..2 * hidden)
            .map(|i| ((i as f32) * 0.023 + 0.7).sin())
            .collect::<Vec<_>>();
        let x = MxArray::from_float32(&x_values, &[1, 2, hidden])?.astype(DType::BFloat16)?;
        let (block_q, block_gate) = block.project_q_gate(&x, 1, 2)?;
        let (native_q, native_gate) = native.project_q_gate(&x, 1, 2)?;
        MxArray::eval_arrays(&[&block_q, &block_gate, &native_q, &native_gate])?;
        assert_eq!(
            block_q.to_uint16_native()?,
            native_q.to_uint16_native()?,
            "block-order affine queries must be bit-identical to native per-head splitting"
        );
        assert_eq!(
            block_gate.to_uint16_native()?,
            native_gate.to_uint16_native()?,
            "block-order affine gates must be bit-identical to native per-head splitting"
        );
        Ok(())
    }

    #[test]
    fn q4k_quantized_q_gate_block_matches_native_h4_with_bias() -> Result<()> {
        use crate::utils::gguf_kquant::{KQuantFormat, KQuantScales, repack_kquant};

        let mut cfg = tiny_cfg();
        cfg.hidden_size = 256;
        cfg.head_dim = 32;
        cfg.num_heads = 4;
        cfg.num_kv_heads = 2;
        cfg.intermediate_size = 512;
        let h = cfg.num_heads as i64;
        let d = cfg.head_dim as i64;
        let hidden = cfg.hidden_size as i64;
        let rows = 2 * h * d;
        let format = KQuantFormat::Q4K;

        // One deterministic ggml Q4_K block per output row. Repack through the
        // same importer as a UD GGUF; keep d/dmin small and finite so the M=2
        // qmv_wide comparison is numerically well-conditioned.
        let mut state = 0x5147_4b21u64;
        let mut blocks = (0..rows as usize * format.block_bytes())
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 33) as u8
            })
            .collect::<Vec<_>>();
        for row in 0..rows as usize {
            let base = row * format.block_bytes();
            blocks[base] = 0x00;
            blocks[base + 1] = 0x24; // d = f16 0.015625
            blocks[base + 2] = 0x00;
            blocks[base + 3] = 0x20; // dmin = f16 0.0078125
        }
        let packed = repack_kquant(format, &blocks, rows as usize, hidden as usize)?;
        let weight = MxArray::from_uint32(
            &packed.weight,
            &[rows, format.weight_cols(hidden as usize) as i64],
        )?;
        let scales = match &packed.scales {
            KQuantScales::Unsigned(values) => {
                MxArray::from_uint8(values, &[rows, format.scales_cols(hidden as usize) as i64])?
            }
            KQuantScales::Signed(_) => panic!("Q4_K scales must be unsigned"),
        };
        let quant_biases = MxArray::from_float16(
            &packed.biases,
            &[rows, format.biases_cols(hidden as usize) as i64],
        )?;
        let additive_values = (0..rows)
            .map(|i| ((i as f32) * 0.031 - 0.2).cos() * 0.1)
            .collect::<Vec<_>>();
        let additive_bias =
            MxArray::from_float32(&additive_values, &[rows])?.astype(DType::BFloat16)?;

        let make_q_proj = || {
            QuantizedLinear::new(
                weight.clone(),
                scales.clone(),
                Some(quant_biases.clone()),
                Some(additive_bias.clone()),
                32,
                4,
                "q4k".to_string(),
            )
        };
        let mut block = Qwen3_5Attention::new(&cfg)?;
        let mut native = Qwen3_5Attention::new(&cfg)?;
        block.set_quantized_q_proj(make_q_proj());
        native.set_quantized_q_proj(make_q_proj());
        block.finalize_q_gate_block()?;
        // The independent reference keeps its original unfinalized layout.
        assert!(block.q_proj.has_q_gate_block_layout());
        assert!(!native.q_proj.has_q_gate_block_layout());
        assert!(block.q_gate_block_t.is_none());

        let x_values = (0..2 * hidden)
            .map(|i| ((i as f32) * 0.023 + 0.7).sin())
            .collect::<Vec<_>>();
        let x = MxArray::from_float32(&x_values, &[1, 2, hidden])?.astype(DType::BFloat16)?;
        let (block_q, block_gate) = block.project_q_gate(&x, 1, 2)?;
        let (native_q, native_gate) = native.project_q_gate(&x, 1, 2)?;
        MxArray::eval_arrays(&[&block_q, &block_gate, &native_q, &native_gate])?;
        assert_eq!(
            block_q.to_uint16_native()?,
            native_q.to_uint16_native()?,
            "block-order Q4_K queries must be bit-identical to native per-head splitting"
        );
        assert_eq!(
            block_gate.to_uint16_native()?,
            native_gate.to_uint16_native()?,
            "block-order Q4_K gates must be bit-identical to native per-head splitting"
        );
        Ok(())
    }

    /// Gate for `fused_qk_norm_rope`: the one-dispatch kernel must equal
    /// `fast::rms_norm` -> transpose -> `fast::rope(offsets)` bit-for-bit on
    /// random BF16 data, over block rows 1..=8, several position offsets,
    /// the Qwen3.8 head counts (24 q / 4 kv) and a smaller pair, with the
    /// strided `[B, T, H, D]` views the verify path hands it (q is a split of
    /// the q/gate projection, k a split of the merged k/v projection).
    #[test]
    fn fused_qk_norm_rope_eq() -> Result<()> {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return Ok(());
        }
        let d = 256i64;
        let rot = 64;
        for (hq, hk) in [(24i64, 4i64), (4, 2)] {
            let mut cfg = tiny_cfg();
            cfg.head_dim = d as i32;
            cfg.num_heads = hq as i32;
            cfg.num_kv_heads = hk as i32;
            cfg.partial_rotary_factor = rot as f64 / d as f64;
            cfg.rope_theta = 10_000_000.0;
            let mut attn = Qwen3_5Attention::new(&cfg)?;
            assert_eq!(attn.rope.dims, rot);
            let wq = MxArray::random_normal(&[d], 1.0, 0.25, Some(DType::BFloat16))?;
            let wk = MxArray::random_normal(&[d], 1.0, 0.25, Some(DType::BFloat16))?;
            attn.q_norm.set_weight(&wq)?;
            attn.k_norm.set_weight(&wk)?;
            for t in 1..=8i64 {
                for offset in [0i32, 1, 7, 4096, 32_480] {
                    let offsets = MxArray::from_int32(&[offset], &[1])?;
                    // Strided views: q = first half of a [B, T, 2*hq*d] row,
                    // k = first hk*d of a [B, T, 2*hk*d] row.
                    let qg = MxArray::random_normal(
                        &[1, t, 2 * hq * d],
                        0.0,
                        2.0,
                        Some(DType::BFloat16),
                    )?;
                    let q = qg.split_sections(&[hq * d], 2)?[0].reshape(&[1, t, hq, d])?;
                    let kv = MxArray::random_normal(
                        &[1, t, 2 * hk * d],
                        0.0,
                        2.0,
                        Some(DType::BFloat16),
                    )?;
                    let k = kv.split_sections(&[hk * d], 2)?[0].reshape(&[1, t, hk, d])?;

                    let (fq, fk) = attn
                        .fused_qk_norm_rope(&q, &k, &offsets)
                        .expect("fused qk norm+rope must take this contract");
                    let rq = attn.rope.forward_with_offsets(
                        &attn.q_norm.forward(&q)?.transpose(Some(&[0, 2, 1, 3]))?,
                        &offsets,
                    )?;
                    let rk = attn.rope.forward_with_offsets(
                        &attn.k_norm.forward(&k)?.transpose(Some(&[0, 2, 1, 3]))?,
                        &offsets,
                    )?;
                    for (name, fused, reference) in [("q", &fq, &rq), ("k", &fk, &rk)] {
                        assert_eq!(fused.dtype()?, DType::BFloat16);
                        assert_eq!(fused.shape()?.as_ref(), reference.shape()?.as_ref());
                        let a = fused.astype(DType::Float32)?.to_float32()?;
                        let b = reference.astype(DType::Float32)?.to_float32()?;
                        let mismatches = a
                            .as_ref()
                            .iter()
                            .zip(b.as_ref().iter())
                            .filter(|(x, y)| x.to_bits() != y.to_bits())
                            .count();
                        assert_eq!(
                            mismatches,
                            0,
                            "{name}: hq={hq} hk={hk} t={t} offset={offset}: {mismatches} of {} elements differ",
                            a.len()
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
