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
    let max_q = (unsafe { mlx_sys::mlx_segmented_sdpa_max_query_length(gqa as i32) }) as i64;
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

/// IO bundle for [`Qwen3_5Attention::forward_verify`] (the compiled DFlash2
/// verify path): the graph reads position and the K/V prefix as array inputs
/// and returns the post-RoPE block through `out_kv`, so nothing host-baked —
/// cache offsets, prefix lengths, write bounds — enters the traced region.
pub(crate) struct AttentionVerifyIo<'a> {
    /// Live K prefix `[B, Hkv, P, D]` — a view into the flat KVCache buffer.
    pub prefix_keys: &'a MxArray,
    /// Live V prefix `[B, Hkv, P, D]`.
    pub prefix_values: &'a MxArray,
    /// Per-batch RoPE position of this block's first row, `[B] int32`.
    pub rope_offsets: &'a MxArray,
    /// Receives `(new_k, new_v)` `[B, Hkv, T, D]` post-RoPE — the layout
    /// `KVCache::update_and_fetch` would store.
    pub out_kv: &'a mut Option<(MxArray, MxArray)>,
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
    pub(crate) fn verify_can_be_shapeless(&self, seq_len: i64) -> bool {
        if !unsafe { mlx_sys::mlx_metal_is_available() }
            || unsafe { mlx_sys::mlx_default_device() } != 1
            || self.num_kv_heads <= 0
            || self.num_heads % self.num_kv_heads != 0
        {
            return false;
        }
        let gqa = i64::from(self.num_heads / self.num_kv_heads);
        let device_max_q = if self.head_dim == 256 {
            (unsafe { mlx_sys::mlx_segmented_sdpa_max_query_length(gqa as i32) }) as i64
        } else {
            0
        };
        verify_shapeless_geometry(seq_len, gqa, self.head_dim, device_max_q)
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
        let queries = self.q_norm.forward(&queries)?;
        let keys = self.k_norm.forward(&keys)?;

        // RoPE rotates along axis -2: feed [B, H, T, D] directly and keep
        // that layout — the eager path's round-trip back to [B, T, H, D]
        // only exists to satisfy the KVCache write order, which lives in
        // the caller here.
        let queries = self
            .rope
            .forward_with_offsets(&queries.transpose(Some(&[0, 2, 1, 3]))?, io.rope_offsets)?;
        let new_keys = self
            .rope
            .forward_with_offsets(&keys.transpose(Some(&[0, 2, 1, 3]))?, io.rope_offsets)?;
        let new_values = values.transpose(Some(&[0, 2, 1, 3]))?;
        *io.out_kv = Some((new_keys.clone(), new_values.clone()));

        let output = if seq_len > 1 {
            let gqa = (self.num_heads / self.num_kv_heads.max(1)) as i64;
            let vector_dims = matches!(self.head_dim, 64 | 96 | 128 | 256);
            let segmented_enabled = unsafe { mlx_sys::mlx_metal_is_available() }
                && unsafe { mlx_sys::mlx_default_device() } == 1
                && self.head_dim == 256
                && queries.dtype()? == DType::BFloat16;
            let device_max_q = if segmented_enabled {
                (unsafe { mlx_sys::mlx_segmented_sdpa_max_query_length(gqa as i32) }) as i64
            } else {
                0
            };
            // Segmented attention serves the whole block in one call.
            let one_call = segmented_enabled
                && io.prefix_keys.shape_at(2)? > 0
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
                    io.prefix_keys,
                    io.prefix_values,
                    &head_new_k,
                    &head_new_v,
                    self.scale,
                    head_len > 1,
                )?;
                let out_tail = verify_sdpa_without_kv_concat(
                    &q_parts[1],
                    io.prefix_keys,
                    io.prefix_values,
                    &new_keys,
                    &new_values,
                    self.scale,
                    true,
                )?;
                MxArray::concatenate(&out_head, &out_tail, 2)?
            } else {
                verify_sdpa_without_kv_concat(
                    &queries,
                    io.prefix_keys,
                    io.prefix_values,
                    &new_keys,
                    &new_values,
                    self.scale,
                    true,
                )?
            }
        } else {
            verify_sdpa_without_kv_concat(
                &queries,
                io.prefix_keys,
                io.prefix_values,
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
        // 16384 ('d' 128 -> 512), so every class reaches every route.
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
        let gqa = HQ / HKV;
        let tail_len = 8 - head_len;
        let mut seen = [0usize; 4];
        let mut class = 0u8;
        for prefix in prefixes {
            let (route, device_class) = device_verify_route(HQ, HKV, 8, prefix);
            class = device_class;
            let expected = if head_len == 0 {
                VERIFY_SINGLE
            } else {
                verify_route(
                    8,
                    max_q,
                    class_reduction(class, prefix + head_len, gqa, gqa * head_len),
                    class_reduction(class, prefix + 8, gqa, gqa * tail_len),
                    true,
                )
            };
            assert_eq!(
                route, expected,
                "class '{}' prefix {prefix}: verify route",
                class as char
            );
            seen[route as usize] += 1;
            let pk = nan_tailed_prefix(&base_k, prefix)?;
            let pv = nan_tailed_prefix(&base_v, prefix)?;
            let got = strict_segmented_for_test(&q, &pk, &pv, &nk, &nv)?.to_float32()?;
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
        for route in [VERIFY_ONE_PASS, VERIFY_UNIFIED, VERIFY_SPLIT] {
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
            let Some(head_len) = segmented_verify_head_len(rows, max_q) else {
                eprintln!("SKIP rows={rows}: max_query_length={max_q} cannot cover the block");
                continue;
            };
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
            for prefix in prefixes {
                let (route, class) = device_verify_route(HQ, HKV, rows, prefix);
                assert!(
                    route >= 0,
                    "no segmented route for rows={rows}, prefix={prefix}"
                );
                seen[route as usize] += 1;
                if class == b's' {
                    class_s = true;
                    let expected = if head_len == 0 {
                        VERIFY_SINGLE
                    } else {
                        verify_route(
                            rows,
                            max_q,
                            class_reduction(b's', prefix + head_len, gqa, gqa * head_len),
                            class_reduction(b's', prefix + rows, gqa, gqa * (rows - head_len)),
                            true,
                        )
                    };
                    assert_eq!(route, expected, "route for rows={rows}, prefix={prefix}");
                }
                let pk = nan_tailed_prefix(&base_k, prefix)?;
                let pv = nan_tailed_prefix(&base_v, prefix)?;
                for (set, q) in queries.iter().enumerate() {
                    let got = strict_segmented_for_test(q, &pk, &pv, &nk, &nv)?.to_float32()?;
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
                for route in [VERIFY_ONE_PASS, VERIFY_UNIFIED, VERIFY_SPLIT] {
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
    /// trace must stay exact while replays cross every route boundary.
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
        let prefixes: Vec<i64> = (1010..=1030)
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
            for &prefix in &prefixes {
                let pk = nan_tailed_prefix(&base_k, prefix)?;
                let pv = nan_tailed_prefix(&base_v, prefix)?;
                let compiled =
                    invoke_compiled_graph(fn_id, &[&q, &pk, &pv, &nk, &nv], 1, true, &mut builder)?
                        .ok_or_else(|| Error::from_reason("compiled invoke failed"))?;
                let compiled = compiled[0].to_float32()?;
                let eager = strict_segmented_for_test(&q, &pk, &pv, &nk, &nv)?.to_float32()?;
                let chunks = segmented_or_concat_split_for_test(&q, &pk, &pv, &nk, &nv, true)?
                    .to_float32()?;
                let (route, _) = device_verify_route(HQ, HKV, rows, prefix);
                seen[route.max(0) as usize] += 1;
                assert_eq!(
                    compiled.as_ref(),
                    eager.as_ref(),
                    "rows={rows}, prefix={prefix}, route={route}"
                );
                assert_eq!(
                    compiled.as_ref(),
                    chunks.as_ref(),
                    "rows={rows}, prefix={prefix}, route={route}"
                );
            }
            assert_eq!(
                builds.get(),
                1,
                "rows={rows}: the prefix must stay out of the trace key"
            );
            eprintln!(
                "rows={rows} replay routes one_pass={} unified={} split={}",
                seen[1], seen[2], seen[3]
            );
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
        for prefix in [87_i64, 6219, 32768] {
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
                cases.push((q, pk, pv, nk, nv));
            }
            let run = |i: usize, one_call: bool| -> Result<MxArray> {
                let (q, pk, pv, nk, nv) = &cases[i % SETS];
                let out = if one_call {
                    strict_segmented_for_test(q, pk, pv, nk, nv)?
                } else {
                    segmented_or_concat_split_for_test(q, pk, pv, nk, nv, true)?
                };
                MxArray::eval_arrays(&[&out])?;
                Ok(out)
            };
            for i in 0..SETS * 2 {
                let before = run(i, false)?.to_float32()?;
                let after = run(i, true)?.to_float32()?;
                assert_eq!(before.as_ref(), after.as_ref(), "prefix={prefix}, set={i}");
            }
            let mut split_ms = Vec::with_capacity(SAMPLES);
            let mut one_call_ms = Vec::with_capacity(SAMPLES);
            for i in 0..SAMPLES {
                // Alternate order so the candidate is not always the second
                // reader of a just-warmed prefix. Rotate four independent KV sets.
                for one_call in if i % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let started = Instant::now();
                    let out = run(i, one_call)?;
                    let ms = started.elapsed().as_secs_f64() * 1e3;
                    if one_call {
                        one_call_ms.push(ms);
                    } else {
                        split_ms.push(ms);
                    }
                    drop(out);
                }
            }
            split_ms.sort_by(f64::total_cmp);
            one_call_ms.sort_by(f64::total_cmp);
            eprintln!(
                "segmented prefix={prefix} q=8 hq=24 hkv=4 sets={SETS} samples={SAMPLES} split_min_ms={:.4} split_median_ms={:.4} one_call_min_ms={:.4} one_call_median_ms={:.4}",
                split_ms[0],
                split_ms[SAMPLES / 2],
                one_call_ms[0],
                one_call_ms[SAMPLES / 2]
            );
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

    fn tiny_cfg() -> Qwen3_5Config {
        Qwen3_5Config {
            qwen35_gguf_gdn_layout: None,
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
}
