use std::sync::OnceLock;

use crate::array::MxArray;
use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::models::attention_core::{
    BatchedDecodeLabels, CacheHitPrefillRoute, PagedAttentionCore,
};
use crate::models::quantized_linear::LinearProj;
use crate::nn::{Linear, RMSNorm, RoPE};
use crate::transformer::KVCache;
use crate::transformer::paged_kv_cache_adapter::{
    PagedDecodeRouteHint, PagedKVCacheAdapter, SeqId,
};
use napi::bindgen_prelude::*;

/// When enabled (opt-in; default OFF), cache-hit prefill (`cached_prefix_len > 0`,
/// i.e. every multi-turn chat continuation) first tries the MLX graph-native
/// paged-attention bridge (`PagedKVCacheAdapter::gather_kv_for_prefill_chunk`),
/// which reads the K/V pool through MLX graph dependencies with no forced host
/// sync. When disabled (the default), or when the bridge is unavailable for the
/// inputs (non-Metal backend, batch > 1, an unsupported cache dtype, or an
/// oversized auxiliary buffer), `gather_kv_for_prefill_sdpa` gathers the dense
/// `[0, total_ctx)` K/V in-graph for an explicit-mask SDPA instead; the
/// synchronous `read_kv_range` host read remains only as a last resort for
/// cache dtypes the dense gather cannot serve (FP8).
///
/// The bridge reads the SAME physical KV bytes as `read_kv_range`; only the
/// attention kernel differs (fused paged-attn vs explicit-mask SDPA), the
/// accepted ~1-ULP class already shipped default-on for paged DECODE. It is held
/// opt-in here — unlike `qwen3_5`/`gemma4`, which default it on — only because
/// the divergence has no green automated parity gate on the one available LFM2
/// checkpoint (greedy decode there is repeat-loop-degenerate, so byte-identical
/// text parity is an unreliable oracle). Flip to default-on once a gemma4-style
/// paged-vs-flat gate exists on a stable checkpoint.
fn paged_prefill_paged_attention_enabled() -> bool {
    // Without a Metal backend (CUDA/Linux build) the C++ paged-attention
    // kernel throws, so a cache-hit prefill must NOT dispatch it. Hard-close
    // the path here so reuse-turn prefills stay on the device-agnostic SDPA
    // fallback (`read_kv_range` + explicit mask).
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return false;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::inference_trace::env_flag_enabled_or_default(
            "MLX_LFM2_PAGED_PREFILL_PAGED_ATTENTION",
            false,
        )
    })
}

/// LFM2 multi-head attention with QK RMSNorm and RoPE.
///
/// Follows `lfm2.py:53-109` (Attention class).
///
/// Key features:
/// - GQA: 32 query heads, 8 KV heads (head_dim=64)
/// - Per-head RMSNorm on Q and K (not V)
/// - Standard RoPE (neox-style, base=1M)
/// - No bias on any projection
pub struct Lfm2Attention {
    pub(crate) q_proj: LinearProj,
    pub(crate) k_proj: LinearProj,
    pub(crate) v_proj: LinearProj,
    pub(crate) out_proj: LinearProj,
    pub(crate) q_layernorm: RMSNorm,
    pub(crate) k_layernorm: RMSNorm,
    rope: RoPE,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f64,
}

impl Lfm2Attention {
    /// Create a new LFM2 attention layer.
    pub fn new(
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        norm_eps: f64,
        rope_theta: f64,
    ) -> Result<Self> {
        let h = hidden_size as u32;
        let q_dim = (num_heads * head_dim) as u32;
        let kv_dim = (num_kv_heads * head_dim) as u32;

        let q_proj = LinearProj::Standard(Linear::new(h, q_dim, Some(false))?);
        let k_proj = LinearProj::Standard(Linear::new(h, kv_dim, Some(false))?);
        let v_proj = LinearProj::Standard(Linear::new(h, kv_dim, Some(false))?);
        let out_proj = LinearProj::Standard(Linear::new(q_dim, h, Some(false))?);

        let q_layernorm = RMSNorm::new(head_dim as u32, Some(norm_eps))?;
        let k_layernorm = RMSNorm::new(head_dim as u32, Some(norm_eps))?;

        let rope = RoPE::new(
            head_dim,
            Some(false), // traditional=False (neox-style)
            Some(rope_theta),
            None,
        );

        let scale = (head_dim as f64).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            q_layernorm,
            k_layernorm,
            rope,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input [B, T, hidden_size]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional KVCache for incremental decoding
    ///
    /// # Returns
    /// Output tensor [B, T, hidden_size]
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // Q/K/V projections
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape to [B, T, num_heads, head_dim] and apply per-head layernorm
        let queries =
            queries.reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?;
        let queries = self.q_layernorm.forward(&queries)?;
        // Transpose to [B, num_heads, T, head_dim]
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;

        let keys = keys.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let keys = self.k_layernorm.forward(&keys)?;
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;

        // V: reshape + transpose (no layernorm on V)
        let values = values.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let values = values.transpose(Some(&[0, 2, 1, 3]))?;

        // Apply RoPE with cache offset
        let offset = cache.as_ref().map_or(0, |c| c.get_offset());
        let queries = self.rope.forward(&queries, Some(offset))?;
        let keys = self.rope.forward(&keys, Some(offset))?;

        // Update KV cache
        let (keys, values) = if let Some(c) = cache {
            c.update_and_fetch(&keys, &values)?
        } else {
            (keys, values)
        };

        // Scaled dot-product attention
        let output = if let Some(m) = mask {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale, Some(m))?
        } else if seq_len > 1 {
            scaled_dot_product_attention_causal(&queries, &keys, &values, self.scale)?
        } else {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale, None)?
        };

        // Transpose back [B, H, T, D] -> [B, T, H*D]
        let output = output.transpose(Some(&[0, 2, 1, 3]))?;
        let output = output.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;

        // Output projection
        self.out_proj.forward(&output)
    }

    /// The shared paged skeleton with LFM2's parameterization: per-head
    /// Q/K RMSNorm on `[B,T,H,D]` before the transpose (V has no norm),
    /// scalar-offset RoPE, `Auto` decode route, and the opt-in
    /// `MLX_LFM2_PAGED_PREFILL_PAGED_ATTENTION` bridge for cache-hit
    /// prefill (`gather_kv_for_prefill_chunk`, batch-1 only, falling back
    /// to the `gather_kv_for_prefill_sdpa` graph-SDPA gather then
    /// `read_kv_range` + explicit-mask SDPA).
    fn paged_core(&self) -> PagedAttentionCore<'_> {
        PagedAttentionCore {
            q_proj: &self.q_proj,
            k_proj: &self.k_proj,
            v_proj: &self.v_proj,
            o_proj: &self.out_proj,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            scale: self.scale,
            qk_norm: Some((&self.q_layernorm, &self.k_layernorm)),
            rope: Some(&self.rope),
            kv_io_dtype: None,
            decode_route_hint: PagedDecodeRouteHint::Auto,
            cache_hit_prefill: CacheHitPrefillRoute::BridgeIfBatch1ThenGraphSdpa {
                gate: paged_prefill_paged_attention_enabled,
            },
            family: "lfm2",
        }
    }

    /// Forward pass driven by `PagedKVCacheAdapter` for full-attention
    /// LFM2 layers.
    ///
    /// Mirrors `TransformerBlock::forward_paged_adapter` (Qwen3) but
    /// adapted for LFM2's attention layout (Q/K layernorm AFTER reshape,
    /// V has no layernorm, no Q gating). The decoder layer's
    /// pre-attention `operator_norm` is applied OUTSIDE this method to
    /// match the existing flat-path call site, so `x` here is already
    /// pre-normalized.
    ///
    /// Caller responsibilities (mirrors Qwen3 helper contract):
    /// 1. `adapter.record_tokens(&[suffix])` BEFORE this call so the
    ///    cursor is advanced by the chunk; `update_keys_values` enforces
    ///    alignment.
    /// 2. `attn_layer_idx` is the ATTENTION-LAYER ORDINAL into the
    ///    adapter's `LayerKVPool`, NOT the absolute decoder index. The
    ///    pool is sized for `config.full_attn_idxs().len()` slots.
    ///
    /// Output: `[1, seq_len, hidden_size]` so the residual `h = x + r`
    /// in the decoder layer stays the same.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_paged(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        attn_layer_idx: u32,
        first_logical_position: u32,
        cached_prefix_len: u32,
        is_prefill: bool,
        prefill_mask: Option<&MxArray>,
    ) -> Result<MxArray> {
        self.paged_core().forward_paged(
            x,
            adapter,
            attn_layer_idx,
            first_logical_position,
            cached_prefix_len,
            is_prefill,
            prefill_mask,
        )
    }

    /// Uniform batched paged decode with one request-specific RoPE offset and
    /// one token per row.
    ///
    /// This deliberately requires the graph-native batched K/V write and
    /// attention gather. Falling back to N serial rows would make scheduler
    /// occupancy look healthy while forfeiting the shared weight stream.
    pub(crate) fn forward_paged_batched(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        attn_layer_idx: u32,
        rows: &[(SeqId, u32)],
    ) -> Result<MxArray> {
        self.paged_core().forward_paged_batched(
            x,
            adapter,
            attn_layer_idx,
            rows,
            &BatchedDecodeLabels {
                type_name: "Lfm2Attention",
                family: "LFM2",
            },
        )
    }

    // ========== Weight setters ==========
    //
    // NOTE: q/k/v/out_proj weights are loaded via the `*_proj_mut()` accessors
    // below, which expose the mode-aware `LinearProj`. The persistence layer
    // either installs a `QuantizedLinear` backend (any of affine / mxfp4 /
    // mxfp8 / nvfp4) via `LinearProj::set_quantized`, or sets a dense bf16
    // weight via `LinearProj::set_weight`. The `forward` path dispatches
    // quantized vs dense transparently. q/k_layernorm are never quantized.

    pub fn set_q_layernorm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.q_layernorm.set_weight(w)
    }

    pub fn set_k_layernorm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.k_layernorm.set_weight(w)
    }

    // ========== Mutable projection accessors ==========
    //
    // Expose the mode-aware `LinearProj`s so the persistence layer can install
    // a quantized backend (affine / mxfp4 / mxfp8 / nvfp4) via
    // `set_quantized`, or a plain bf16 weight via `set_weight`, uniformly for a
    // fully quantized checkpoint. The `forward` path dispatches quantized vs
    // dense transparently.

    pub fn q_proj_mut(&mut self) -> &mut LinearProj {
        &mut self.q_proj
    }

    pub fn k_proj_mut(&mut self) -> &mut LinearProj {
        &mut self.k_proj
    }

    pub fn v_proj_mut(&mut self) -> &mut LinearProj {
        &mut self.v_proj
    }

    pub fn out_proj_mut(&mut self) -> &mut LinearProj {
        &mut self.out_proj
    }
}
