//! K2-Horizon grouped-query attention.
//!
//! Mirrors `lfm2/attention.rs` structurally (LinearProj projections,
//! PagedKVCacheAdapter plumbing, batched decode) minus the per-head Q/K
//! RMSNorm — K2's `query_key_norm` is `false`, so Q/K go straight from
//! projection → reshape → RoPE. Verified against the HF remote code:
//! `q_proj/k_proj/v_proj/o_proj` (all bias-free), full RoPE on `head_dim`
//! with `rope_theta` from `rope_parameters`.
//!
//! GQA layout (7B checkpoint): 32 query heads, 8 KV heads, head_dim=128,
//! scale = head_dim^-0.5.

use crate::array::MxArray;
use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::models::attention_core::{
    BatchedDecodeLabels, CacheHitPrefillRoute, PagedAttentionCore,
};
use crate::models::quantized_linear::LinearProj;
use crate::nn::{Linear, RoPE};
use crate::transformer::KVCache;
use crate::transformer::paged_kv_cache_adapter::{
    PagedDecodeRouteHint, PagedKVCacheAdapter, SeqId,
};
use napi::bindgen_prelude::*;

/// K2-Horizon attention block (dense-or-quantized projections).
///
/// Projections are `LinearProj` so the persistence layer can install an
/// MXFP8 quantized backend post-load; forward dispatches transparently.
pub struct K2Attention {
    pub(crate) q_proj: LinearProj,
    pub(crate) k_proj: LinearProj,
    pub(crate) v_proj: LinearProj,
    pub(crate) o_proj: LinearProj,
    rope: RoPE,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f64,
}

impl K2Attention {
    pub fn new(
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rope_theta: f64,
    ) -> Result<Self> {
        let h = hidden_size as u32;
        let q_dim = (num_heads * head_dim) as u32;
        let kv_dim = (num_kv_heads * head_dim) as u32;

        let q_proj = LinearProj::Standard(Linear::new(h, q_dim, Some(false))?);
        let k_proj = LinearProj::Standard(Linear::new(h, kv_dim, Some(false))?);
        let v_proj = LinearProj::Standard(Linear::new(h, kv_dim, Some(false))?);
        let o_proj = LinearProj::Standard(Linear::new(q_dim, h, Some(false))?);

        // Full RoPE on head_dim, neox-style (rotate-half). K2 ships
        // `rope_head_dim == head_dim` and `rope_type: "default"`.
        let rope = RoPE::new(head_dim, Some(false), Some(rope_theta), None);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// Flat forward — x: `[B, T, hidden_size]` → `[B, T, hidden_size]`.
    ///
    /// `mask` is the caller's causal mask (chunked prefill); `cache`
    /// appends K/V and returns the full sequence.
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let queries = queries
            .reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let keys = keys
            .reshape(&[
                batch,
                seq_len,
                self.num_kv_heads as i64,
                self.head_dim as i64,
            ])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let values = values
            .reshape(&[
                batch,
                seq_len,
                self.num_kv_heads as i64,
                self.head_dim as i64,
            ])?
            .transpose(Some(&[0, 2, 1, 3]))?;

        let offset = cache.as_ref().map_or(0, |c| c.get_offset());
        let queries = self.rope.forward(&queries, Some(offset))?;
        let keys = self.rope.forward(&keys, Some(offset))?;

        let (keys, values) = if let Some(c) = cache {
            c.update_and_fetch(&keys, &values)?
        } else {
            (keys, values)
        };

        let output = if let Some(m) = mask {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale, Some(m))?
        } else if seq_len > 1 {
            scaled_dot_product_attention_causal(&queries, &keys, &values, self.scale)?
        } else {
            scaled_dot_product_attention(&queries, &keys, &values, self.scale, None)?
        };

        let output = output.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
            batch,
            seq_len,
            (self.num_heads * self.head_dim) as i64,
        ])?;
        self.o_proj.forward(&output)
    }

    /// The shared paged skeleton with K2's parameterization: no Q/K norm
    /// (`query_key_norm = false`), full RoPE on `head_dim`, `ForceD128`
    /// decode route (K2's 32q/8kv/hs128 geometry is eligible for the
    /// grouped striped kernel; the hint degrades to generic V2 whenever
    /// the dispatch predicate or pipeline check fails), and cache-hit
    /// prefill gathers K/V through the MLX graph and uses the existing
    /// explicit-mask SDPA; a failed gather retains the synchronous host
    /// fallback.
    fn paged_core(&self) -> PagedAttentionCore<'_> {
        PagedAttentionCore {
            q_proj: &self.q_proj,
            k_proj: &self.k_proj,
            v_proj: &self.v_proj,
            o_proj: &self.o_proj,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            scale: self.scale,
            qk_norm: None,
            rope: Some(&self.rope),
            kv_io_dtype: None,
            decode_route_hint: PagedDecodeRouteHint::ForceD128,
            cache_hit_prefill: CacheHitPrefillRoute::GraphSdpa,
            family: "k2_horizon",
        }
    }

    /// Paged forward driven by `PagedKVCacheAdapter`.
    ///
    /// Caller contract (same as LFM2/Qwen3 paged layers):
    /// 1. `adapter.record_tokens(&[suffix])` BEFORE this call so the
    ///    cursor is advanced by the chunk.
    /// 2. `attn_layer_idx` is the decoder-layer ordinal (every K2 layer is
    ///    full-attention, so it equals the absolute layer index).
    /// 3. `x` is already pre-normalized (the decoder layer applies the
    ///    grouped input layernorm outside).
    ///
    /// Returns `[B, seq_len, hidden_size]`.
    pub fn forward_paged(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        attn_layer_idx: u32,
        first_logical_position: u32,
        cached_prefix_len: u32,
        is_prefill: bool,
    ) -> Result<MxArray> {
        self.paged_core().forward_paged(
            x,
            adapter,
            attn_layer_idx,
            first_logical_position,
            cached_prefix_len,
            is_prefill,
            None,
        )
    }

    /// Batched paged decode: `[N, 1, hidden]` rows, per-row RoPE offsets.
    ///
    /// Requires the graph-native batched K/V write + attention gather —
    /// scheduler occupancy without them would forfeit the shared weight
    /// stream (same contract as `Lfm2Attention::forward_paged_batched`).
    ///
    /// Single-owner decode (num_seqs == 1) takes the grouped striped
    /// kernel via ForceD128; multi-owner waves fall back to generic V2
    /// inside the dispatch predicate.
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
                type_name: "K2Attention",
                family: "K2",
            },
        )
    }

    // ========== Mutable projection accessors ==========
    //
    // The persistence layer installs a `QuantizedLinear` backend (mxfp8)
    // via `LinearProj::set_quantized`, or a dense weight via
    // `LinearProj::set_weight`. Forward dispatches transparently.

    pub fn q_proj_mut(&mut self) -> &mut LinearProj {
        &mut self.q_proj
    }

    pub fn k_proj_mut(&mut self) -> &mut LinearProj {
        &mut self.k_proj
    }

    pub fn v_proj_mut(&mut self) -> &mut LinearProj {
        &mut self.v_proj
    }

    pub fn o_proj_mut(&mut self) -> &mut LinearProj {
        &mut self.o_proj
    }
}
