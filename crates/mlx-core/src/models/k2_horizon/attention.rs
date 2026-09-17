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
use crate::array::mask::create_causal_mask;
use crate::models::qwen3_5_moe::quantized_linear::LinearProj;
use crate::nn::{Linear, RoPE};
use crate::transformer::KVCache;
use crate::transformer::paged_flags::{graph_decode_gather_enabled, native_kv_write_enabled};
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
            .reshape(&[batch, seq_len, self.num_kv_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let values = values
            .reshape(&[batch, seq_len, self.num_kv_heads as i64, self.head_dim as i64])?
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

        let output = output
            .transpose(Some(&[0, 2, 1, 3]))?
            .reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;
        self.o_proj.forward(&output)
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
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let queries_bhtd = queries
            .reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let keys_bhtd = keys
            .reshape(&[batch, seq_len, self.num_kv_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let values_bhtd = values
            .reshape(&[batch, seq_len, self.num_kv_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;

        let rope_offset = first_logical_position as i32;
        let queries_bhtd = self.rope.forward(&queries_bhtd, Some(rope_offset))?;
        let keys_bhtd = self.rope.forward(&keys_bhtd, Some(rope_offset))?;

        // [B, H_kv, T, D] → [B*T, H_kv, D] for the adapter's paged layout.
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

        let native_written = native_kv_write_enabled()
            && adapter
                .update_keys_values_native(
                    attn_layer_idx,
                    &keys_paged,
                    &values_paged,
                    first_logical_position,
                )
                .is_ok();
        if !native_written {
            adapter
                .update_keys_values(
                    attn_layer_idx,
                    &keys_paged,
                    &values_paged,
                    first_logical_position,
                )
                .map_err(napi::Error::from_reason)?;
        }

        let attn_bhtd = if is_prefill {
            if cached_prefix_len == 0 {
                // Fresh prefill: in-flight SDPA with internal causal mask.
                if seq_len > 1 {
                    scaled_dot_product_attention_causal(
                        &queries_bhtd,
                        &keys_bhtd,
                        &values_bhtd,
                        self.scale,
                    )?
                } else {
                    scaled_dot_product_attention(
                        &queries_bhtd,
                        &keys_bhtd,
                        &values_bhtd,
                        self.scale,
                        None,
                    )?
                }
            } else {
                // Cache-hit prefill: read the pool for [0, total_ctx) and
                // apply an explicit causal mask with the prefix offset.
                // The graph-native gather_kv_for_prefill_chunk bridge stays
                // opt-out here — K2's parity gate is fresh; the explicit
                // path is the verified one.
                let total_ctx = cached_prefix_len + (seq_len as u32);
                let (k_full, v_full) = adapter
                    .read_kv_range(attn_layer_idx, 0, total_ctx)
                    .map_err(napi::Error::from_reason)?;
                let mask =
                    create_causal_mask(seq_len as i32, Some(cached_prefix_len as i32), None)?;
                scaled_dot_product_attention(
                    &queries_bhtd,
                    &k_full,
                    &v_full,
                    self.scale,
                    Some(&mask),
                )?
            }
        } else {
            // Decode: gather full historical K/V via the paged kernel.
            let queries_3d = queries_bhtd.squeeze(Some(&[2]))?.reshape(&[
                1,
                self.num_heads as i64,
                self.head_dim as i64,
            ])?;
            let attn_3d = if graph_decode_gather_enabled() {
                // K2's 32q/8kv/hs128 geometry is eligible for the grouped
                // striped paged kernel; ForceD128 degrades to generic V2
                // whenever the dispatch predicate or pipeline check fails.
                match adapter.gather_kv_for_decode_graph_with_route(
                    attn_layer_idx,
                    &queries_3d,
                    self.scale as f32,
                    /* softcap */ 1.0,
                    PagedDecodeRouteHint::ForceD128,
                ) {
                    Ok(attn_3d) => attn_3d,
                    Err(_) => adapter
                        .gather_kv_for_decode(
                            attn_layer_idx,
                            &queries_3d,
                            self.scale as f32,
                            /* softcap */ 1.0,
                        )
                        .map_err(napi::Error::from_reason)?,
                }
            } else {
                adapter
                    .gather_kv_for_decode(
                        attn_layer_idx,
                        &queries_3d,
                        self.scale as f32,
                        /* softcap */ 1.0,
                    )
                    .map_err(napi::Error::from_reason)?
            };
            let attn_3d = attn_3d.astype(x.dtype()?)?;
            attn_3d.reshape(&[1, self.num_heads as i64, 1, self.head_dim as i64])?
        };

        let output = attn_bhtd
            .transpose(Some(&[0, 2, 1, 3]))?
            .reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i64])?;
        self.o_proj.forward(&output)
    }

    /// Batched paged decode: `[N, 1, hidden]` rows, per-row RoPE offsets.
    ///
    /// Requires the graph-native batched K/V write + attention gather —
    /// scheduler occupancy without them would forfeit the shared weight
    /// stream (same contract as `Lfm2Attention::forward_paged_batched`).
    pub(crate) fn forward_paged_batched(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        attn_layer_idx: u32,
        rows: &[(SeqId, u32)],
    ) -> Result<MxArray> {
        let shape = x.shape()?;
        if rows.is_empty()
            || shape.as_ref().len() != 3
            || shape[0] != rows.len() as i64
            || shape[1] != 1
        {
            return Err(Error::from_reason(format!(
                "K2Attention::forward_paged_batched expects [N,1,H] for {} rows, got {:?}",
                rows.len(),
                shape.as_ref()
            )));
        }
        if !native_kv_write_enabled() || !graph_decode_gather_enabled() {
            return Err(Error::from_reason(
                "K2 batched decode requires native K/V writes and graph decode gather",
            ));
        }

        let batch = rows.len() as i64;
        let offsets = rows
            .iter()
            .map(|&(seq_id, position)| {
                i32::try_from(position).map_err(|_| {
                    Error::from_reason(format!(
                        "K2 batched decode sequence {seq_id} position {position} exceeds i32::MAX"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = MxArray::from_int32(&offsets, &[batch])?;
        let seq_ids = rows.iter().map(|&(seq_id, _)| seq_id).collect::<Vec<_>>();

        let queries = self
            .q_proj
            .forward(x)?
            .reshape(&[batch, 1, self.num_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let queries = self.rope.forward_with_offsets(&queries, &offsets)?;

        let keys = self
            .k_proj
            .forward(x)?
            .reshape(&[batch, 1, self.num_kv_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let keys = self.rope.forward_with_offsets(&keys, &offsets)?;

        let values = self
            .v_proj
            .forward(x)?
            .reshape(&[batch, 1, self.num_kv_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;

        let queries = queries.squeeze(Some(&[2]))?;
        let keys = keys.squeeze(Some(&[2]))?;
        let values = values.squeeze(Some(&[2]))?;
        adapter
            .update_keys_values_native_batched(attn_layer_idx, &keys, &values, rows)
            .map_err(Error::from_reason)?;
        // Single-owner decode (num_seqs == 1) takes the grouped striped
        // kernel via ForceD128; multi-owner waves fall back to generic V2
        // inside the dispatch predicate. Stripes 0 asks the adapter for the
        // context-table default clamped by device limits.
        let attended = adapter
            .gather_kv_for_decode_graph_batched_with_plan(
                attn_layer_idx,
                &queries,
                &seq_ids,
                self.scale as f32,
                1.0,
                PagedDecodeRouteHint::ForceD128,
                0,
            )
            .map_err(Error::from_reason)?
            .astype(x.dtype()?)?
            .reshape(&[batch, 1, (self.num_heads * self.head_dim) as i64])?;
        self.o_proj.forward(&attended)
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
