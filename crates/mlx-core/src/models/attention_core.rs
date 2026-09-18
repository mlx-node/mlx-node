//! Shared paged-attention core for the `LinearProj`-based families
//! (K2-Horizon, LFM2, Nemotron-H).
//!
//! Each family's `forward_paged` used to hand-copy the same sequence:
//! `LinearProj` q/k/v projections → reshape to per-head layout → optional
//! per-head Q/K RMSNorm → optional scalar-offset RoPE → `[B,H,T,D]` →
//! `[B*T,H_kv,D]` paged write → gather/SDPA dispatch → o_proj. The only
//! degrees of freedom across the migrated families are the fields of
//! [`PagedAttentionCore`]:
//!
//! - `qk_norm` — per-head Q/K RMSNorm on `[B,T,H,D]` before the transpose
//!   (LFM2) or none (K2, Nemotron-H).
//! - `rope` — scalar-offset RoPE on `[B,H,T,D]` (K2, LFM2) or none
//!   (Nemotron-H is NoPE).
//! - `kv_io_dtype` — optional cast of the paged K/V chunk and the decode
//!   query at the pool boundary (Nemotron-H's `PAGED_KV_IO_DTYPE`).
//! - `decode_route_hint` — the single-token decode compute route (K2
//!   requests `ForceD128` grouped stripes; the others `Auto`).
//! - `cache_hit_prefill` — how a cache-hit prefill attends once the suffix
//!   is in the pool; see [`CacheHitPrefillRoute`].
//!
//! Families whose skeleton diverges further — sliding windows, gated
//! attention, M-RoPE, decode route policies (muse_glimmer, gemma4,
//! qwen3_5, and the fused-QKV `TransformerBlock` path) — keep their own
//! `forward_paged`; this core is only the step-for-step common skeleton.
//! They still share [`paged_kv_layout`] for the `[B,H,T,D]` →
//! `[B*T,H_kv,D]` conversion, which is byte-identical everywhere.
//!
//! Lives in `models/` (not `transformer/`) because it is parameterized by
//! `LinearProj` (`models::quantized_linear`); `transformer/` has no
//! `models` dependency today and every consumer already imports
//! `transformer::paged_kv_cache_adapter`.

use crate::array::attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
use crate::array::mask::create_causal_mask;
use crate::array::{DType, MxArray};
use crate::models::quantized_linear::LinearProj;
use crate::nn::{RMSNorm, RoPE};
use crate::transformer::paged_flags::{graph_decode_gather_enabled, native_kv_write_enabled};
use crate::transformer::paged_kv_cache_adapter::{
    PagedDecodeRouteHint, PagedKVCacheAdapter, SeqId,
};
use crate::transformer::paged_policy::{
    gather_kv_for_decode_with_route_and_fallback, warn_once_on_sync_fallback, write_kv_chunk,
};
use napi::bindgen_prelude::*;

/// How a cache-hit prefill (`is_prefill && cached_prefix_len > 0`) computes
/// attention once the suffix chunk has been written to the pool. Bridge routes
/// may use paged attention; their SDPA fallback arms share the existing explicit
/// mask.
#[derive(Clone, Copy)]
pub(crate) enum CacheHitPrefillRoute {
    /// Gather K/V in the MLX graph for explicit-mask SDPA, falling back to
    /// the synchronous host read only when graph gathering fails.
    GraphSdpa,
    /// Attempt `gather_kv_for_prefill_chunk` only when `batch == 1` and
    /// the family's runtime `gate` holds; on gate-false or bridge error,
    /// try `gather_kv_for_prefill_sdpa` for graph-SDPA before the
    /// `read_kv_range` host fallback. The gate runs lazily
    /// on this arm only (LFM2's opt-in
    /// `MLX_LFM2_PAGED_PREFILL_PAGED_ATTENTION` flag).
    BridgeIfBatch1ThenGraphSdpa { gate: fn() -> bool },
    /// Attempt the bridge unconditionally: the `[B,H,T,D]` → `[T,H,D]`
    /// squeeze is NOT batch-gated, so `batch != 1` propagates the squeeze
    /// error exactly like the hand-rolled body (Nemotron-H).
    /// Host-read SDPA on bridge error only.
    BridgeUnconditional,
}

/// Error-message labels for [`PagedAttentionCore::forward_paged_batched`]
/// so each family's strings stay byte-identical to its hand-rolled body.
pub(crate) struct BatchedDecodeLabels {
    /// `"{type_name}::forward_paged_batched expects [N,1,H] ..."`.
    pub type_name: &'static str,
    /// `"{family} batched decode ..."` prefix for the flag/position errors.
    pub family: &'static str,
}

/// The shared paged-attention skeleton. Construct per call — all fields
/// are borrows or `Copy` scalars, so this is free.
///
/// Caller contract (unchanged from the hand-rolled bodies):
/// 1. `adapter.record_tokens(&[suffix])` BEFORE `forward_paged` so the
///    cursor is advanced by the chunk; `update_keys_values` enforces
///    alignment.
/// 2. `attn_layer_idx` is the ATTENTION-LAYER ORDINAL into the adapter's
///    `LayerKVPool`, not the absolute decoder index.
/// 3. `x` is already pre-normalized.
pub(crate) struct PagedAttentionCore<'a> {
    pub q_proj: &'a LinearProj,
    pub k_proj: &'a LinearProj,
    pub v_proj: &'a LinearProj,
    pub o_proj: &'a LinearProj,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    /// SDPA scale (`head_dim^-0.5` for every migrated family).
    pub scale: f64,
    /// Per-head Q/K RMSNorm applied on `[B,T,H,D]` before the transpose
    /// (LFM2). `None` for K2 (`query_key_norm = false`) and Nemotron-H.
    pub qk_norm: Option<(&'a RMSNorm, &'a RMSNorm)>,
    /// Scalar-offset RoPE applied on `[B,H,T,D]` (K2, LFM2). `None` for
    /// NoPE families (Nemotron-H).
    pub rope: Option<&'a RoPE>,
    /// Optional cast applied to the paged K/V chunk before the pool write
    /// and to the decode query before the gather (Nemotron-H's
    /// `PAGED_KV_IO_DTYPE`). `None` = no cast.
    pub kv_io_dtype: Option<DType>,
    /// Single-token / batched decode compute route (`ForceD128` for K2's
    /// grouped stripes, `Auto` elsewhere — `Auto` is exactly what
    /// `gather_kv_for_decode_graph[_batched]` pass internally).
    pub decode_route_hint: PagedDecodeRouteHint,
    /// Cache-hit prefill route; see [`CacheHitPrefillRoute`].
    pub cache_hit_prefill: CacheHitPrefillRoute,
    pub family: &'static str,
}

/// `[B, H_kv, T, D]` → `[B*T, H_kv, D]` for K and V — the pool layout
/// `update_keys_values*` consume. Byte-identical in every family's
/// hand-rolled body (muse_glimmer and qwen3_5 included), so it is the one
/// piece also shared by the families that keep their own `forward_paged`.
pub(crate) fn paged_kv_layout(
    keys_bhtd: &MxArray,
    values_bhtd: &MxArray,
    batch: i64,
    seq_len: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> Result<(MxArray, MxArray)> {
    let keys_paged = keys_bhtd.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
        batch * seq_len,
        num_kv_heads,
        head_dim,
    ])?;
    let values_paged = values_bhtd.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
        batch * seq_len,
        num_kv_heads,
        head_dim,
    ])?;
    Ok((keys_paged, values_paged))
}

/// Write one paged-layout K/V chunk: prefer the graph-native lazy write so
/// the same-step attention read depends on it through MLX's graph (no
/// per-layer host sync), and fall back to the synchronous write when the
/// flag is off or the native kernel could not place the K/V (a failed
/// native write leaves the pool untouched, so this is not a double-write).
fn paged_kv_write(
    adapter: &mut PagedKVCacheAdapter,
    attn_layer_idx: u32,
    keys_paged: &MxArray,
    values_paged: &MxArray,
    first_logical_position: u32,
    family: &'static str,
) -> Result<()> {
    write_kv_chunk(
        adapter,
        attn_layer_idx,
        keys_paged,
        values_paged,
        first_logical_position,
        family,
    )
    .map_err(Error::from_reason)
}

impl PagedAttentionCore<'_> {
    /// Single-sequence paged forward: `x: [B, T, hidden]` → `[B, T, hidden]`.
    ///
    /// Reproduces the hand-rolled K2/LFM2/Nemotron bodies op-for-op:
    /// projections → `[B,T,H,D]` reshape → optional Q/K norm → `[B,H,T,D]`
    /// transpose → optional RoPE → paged write → prefill SDPA / cache-hit
    /// route / decode gather → o_proj.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_paged(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        attn_layer_idx: u32,
        first_logical_position: u32,
        cached_prefix_len: u32,
        is_prefill: bool,
        prefill_mask: Option<&MxArray>,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        // 1. Q/K/V projections.
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // 2. [B,T,H,D] reshape, optional per-head Q/K norm, [B,H,T,D]
        //    transpose.
        let queries =
            queries.reshape(&[batch, seq_len, self.num_heads as i64, self.head_dim as i64])?;
        let queries = match self.qk_norm {
            Some((q_norm, _)) => q_norm.forward(&queries)?,
            None => queries,
        };
        let queries_bhtd = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = keys.reshape(&[
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let keys = match self.qk_norm {
            Some((_, k_norm)) => k_norm.forward(&keys)?,
            None => keys,
        };
        let keys_bhtd = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let values_bhtd = values
            .reshape(&[
                batch,
                seq_len,
                self.num_kv_heads as i64,
                self.head_dim as i64,
            ])?
            .transpose(Some(&[0, 2, 1, 3]))?;

        // 3. Optional scalar-offset RoPE at the request's logical position
        //    (the adapter's pre-record offset).
        let (queries_bhtd, keys_bhtd) = match self.rope {
            Some(rope) => {
                let rope_offset = first_logical_position as i32;
                (
                    rope.forward(&queries_bhtd, Some(rope_offset))?,
                    rope.forward(&keys_bhtd, Some(rope_offset))?,
                )
            }
            None => (queries_bhtd, keys_bhtd),
        };

        // 4. [B, H_kv, T, D] → [B*T, H_kv, D] for the paged write, with the
        //    optional pool-boundary dtype cast (Nemotron-H).
        let (keys_paged, values_paged) = paged_kv_layout(
            &keys_bhtd,
            &values_bhtd,
            batch,
            seq_len,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        )?;
        let (keys_paged, values_paged) = match self.kv_io_dtype {
            Some(dtype) => (keys_paged.astype(dtype)?, values_paged.astype(dtype)?),
            None => (keys_paged, values_paged),
        };
        paged_kv_write(
            adapter,
            attn_layer_idx,
            &keys_paged,
            &values_paged,
            first_logical_position,
            self.family,
        )?;

        // 5. Attention output, still in [B, H, T, D].
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
                self.cache_hit_prefill_attention(
                    x,
                    adapter,
                    attn_layer_idx,
                    &queries_bhtd,
                    batch,
                    seq_len,
                    cached_prefix_len,
                    prefill_mask,
                )?
            }
        } else {
            // Decode: gather full historical K/V via the paged kernel. Both
            // gather variants expect `[1, num_query_heads, head_size]`
            // queries, so reshape from [1, H, 1, D].
            let queries_3d = queries_bhtd.squeeze(Some(&[2]))?.reshape(&[
                1,
                self.num_heads as i64,
                self.head_dim as i64,
            ])?;
            let queries_3d = match self.kv_io_dtype {
                Some(dtype) => queries_3d.astype(dtype)?,
                None => queries_3d,
            };
            // Prefer the graph-native gather (reads the lazy pool arrays
            // through graph dependencies — no per-layer host eval). Fall
            // back to the synchronous gather when it is disabled or
            // unavailable for these inputs.
            let attn_3d = gather_kv_for_decode_with_route_and_fallback(
                adapter,
                attn_layer_idx,
                &queries_3d,
                self.scale as f32,
                1.0,
                self.decode_route_hint,
                self.family,
            )
            .map_err(Error::from_reason)?;
            // Cast back to x's dtype so the residual stays homogeneous.
            let attn_3d = attn_3d.astype(x.dtype()?)?;
            // [1, H, D] -> [1, H, 1, D] for the standard [B,H,T,D] tail.
            attn_3d.reshape(&[1, self.num_heads as i64, 1, self.head_dim as i64])?
        };

        // 6. [B, H, T, D] -> [B, T, H*D] -> output projection.
        let output = attn_bhtd.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
            batch,
            seq_len,
            (self.num_heads * self.head_dim) as i64,
        ])?;
        self.o_proj.forward(&output)
    }

    /// Uniform batched paged decode: `[N, 1, hidden]` rows, one token per
    /// row, one batched native K/V write and one batched gather. No serial
    /// fallback — a genuine N-row wave must share the weight stream, so
    /// this requires the graph-native write and gather flags.
    ///
    /// `labels` keeps each family's error strings byte-identical.
    pub(crate) fn forward_paged_batched(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        attn_layer_idx: u32,
        rows: &[(SeqId, u32)],
        labels: &BatchedDecodeLabels,
    ) -> Result<MxArray> {
        let shape = x.shape()?;
        if rows.is_empty()
            || shape.as_ref().len() != 3
            || shape[0] != rows.len() as i64
            || shape[1] != 1
        {
            return Err(Error::from_reason(format!(
                "{}::forward_paged_batched expects [N,1,H] for {} rows, got {:?}",
                labels.type_name,
                rows.len(),
                shape.as_ref()
            )));
        }
        if !native_kv_write_enabled() || !graph_decode_gather_enabled() {
            return Err(Error::from_reason(format!(
                "{} batched decode requires native K/V writes and graph decode gather",
                labels.family
            )));
        }

        let batch = rows.len() as i64;
        // Per-row RoPE offsets are only needed when the family rotates —
        // NoPE families (Nemotron-H) never build the array, matching the
        // hand-rolled bodies.
        let offsets = match self.rope {
            Some(_) => {
                let offsets = rows
                    .iter()
                    .map(|&(seq_id, position)| {
                        i32::try_from(position).map_err(|_| {
                            Error::from_reason(format!(
                                "{} batched decode sequence {seq_id} position {position} \
                                 exceeds i32::MAX",
                                labels.family
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Some(MxArray::from_int32(&offsets, &[batch])?)
            }
            None => None,
        };
        let seq_ids = rows.iter().map(|&(seq_id, _)| seq_id).collect::<Vec<_>>();

        // Per-head chain: proj -> [N,1,H,D] reshape -> optional Q/K norm ->
        // [N,H,1,D] transpose -> optional per-row RoPE.
        let queries = self.q_proj.forward(x)?.reshape(&[
            batch,
            1,
            self.num_heads as i64,
            self.head_dim as i64,
        ])?;
        let queries = match self.qk_norm {
            Some((q_norm, _)) => q_norm.forward(&queries)?,
            None => queries,
        };
        let queries = queries.transpose(Some(&[0, 2, 1, 3]))?;
        let queries = match (&self.rope, offsets.as_ref()) {
            (Some(rope), Some(offsets)) => rope.forward_with_offsets(&queries, offsets)?,
            _ => queries,
        };

        let keys = self.k_proj.forward(x)?.reshape(&[
            batch,
            1,
            self.num_kv_heads as i64,
            self.head_dim as i64,
        ])?;
        let keys = match self.qk_norm {
            Some((_, k_norm)) => k_norm.forward(&keys)?,
            None => keys,
        };
        let keys = keys.transpose(Some(&[0, 2, 1, 3]))?;
        let keys = match (&self.rope, offsets.as_ref()) {
            (Some(rope), Some(offsets)) => rope.forward_with_offsets(&keys, offsets)?,
            _ => keys,
        };

        let values = self
            .v_proj
            .forward(x)?
            .reshape(&[batch, 1, self.num_kv_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[0, 2, 1, 3]))?;

        // [N, H, 1, D] -> [N, H, D] for the batched paged write/gather;
        // Nemotron-H also casts to the pool IO dtype here.
        let squeeze_io = |t: &MxArray| -> Result<MxArray> {
            let t = t.squeeze(Some(&[2]))?;
            match self.kv_io_dtype {
                Some(dtype) => t.astype(dtype),
                None => Ok(t),
            }
        };
        let queries = squeeze_io(&queries)?;
        let keys = squeeze_io(&keys)?;
        let values = squeeze_io(&values)?;

        adapter
            .update_keys_values_native_batched(attn_layer_idx, &keys, &values, rows)
            .map_err(Error::from_reason)?;
        let attended = adapter
            .gather_kv_for_decode_graph_batched_with_plan(
                attn_layer_idx,
                &queries,
                &seq_ids,
                self.scale as f32,
                /* softcap */ 1.0,
                self.decode_route_hint,
                /* grouped_stripes */ 0,
            )
            .map_err(Error::from_reason)?
            .astype(x.dtype()?)?
            .reshape(&[batch, 1, (self.num_heads * self.head_dim) as i64])?;
        self.o_proj.forward(&attended)
    }

    /// Cache-hit prefill arm: the suffix was just written above, so the
    /// pool's `[0, total_ctx)` covers the full attention context. The
    /// bridge attempt (when routed) reads the pool through MLX graph
    /// dependencies; the fallback gathers K/V in-graph for SDPA
    /// (`gather_kv_for_prefill_sdpa`, `BridgeIfBatch1ThenGraphSdpa` arm
    /// only) before resorting to a synchronous host read + explicit
    /// causal mask + SDPA.
    #[allow(clippy::too_many_arguments)]
    fn cache_hit_prefill_attention(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        attn_layer_idx: u32,
        queries_bhtd: &MxArray,
        batch: i64,
        seq_len: i64,
        cached_prefix_len: u32,
        prefill_mask: Option<&MxArray>,
    ) -> Result<MxArray> {
        let total_ctx = cached_prefix_len + (seq_len as u32);
        let try_bridge = match self.cache_hit_prefill {
            CacheHitPrefillRoute::GraphSdpa => false,
            CacheHitPrefillRoute::BridgeIfBatch1ThenGraphSdpa { gate } => batch == 1 && gate(),
            CacheHitPrefillRoute::BridgeUnconditional => true,
        };
        let maybe_paged_attn = if try_bridge {
            // [B, H, T, D] -> [H, T, D] -> [T, H, D], matching
            // `PagedKVCacheAdapter::gather_kv_for_prefill_chunk`.
            let queries_paged = queries_bhtd
                .squeeze(Some(&[0]))?
                .transpose(Some(&[1, 0, 2]))?;
            adapter
                .gather_kv_for_prefill_chunk(
                    attn_layer_idx,
                    &queries_paged,
                    cached_prefix_len,
                    self.scale as f32,
                )
                .inspect_err(|err| {
                    if matches!(
                        self.cache_hit_prefill,
                        CacheHitPrefillRoute::BridgeIfBatch1ThenGraphSdpa { .. }
                    ) {
                        warn_once_on_sync_fallback(
                            self.family,
                            "prefill_paged_attention",
                            attn_layer_idx,
                            err,
                        );
                    }
                })
                .ok()
                .map(|attn_t_h_d| {
                    let target_dtype = x.dtype()?;
                    let attn_t_h_d = attn_t_h_d.astype(target_dtype)?;
                    // [T, H, D] -> [H, T, D] -> [B, H, T, D]
                    attn_t_h_d.transpose(Some(&[1, 0, 2]))?.reshape(&[
                        batch,
                        self.num_heads as i64,
                        seq_len,
                        self.head_dim as i64,
                    ])
                })
                .transpose()?
        } else {
            None
        };

        match maybe_paged_attn {
            Some(attn) => Ok(attn),
            None => {
                let (k_full, v_full) = if matches!(
                    self.cache_hit_prefill,
                    CacheHitPrefillRoute::GraphSdpa
                        | CacheHitPrefillRoute::BridgeIfBatch1ThenGraphSdpa { .. }
                ) {
                    adapter
                        .gather_kv_for_prefill_sdpa(attn_layer_idx, total_ctx)
                        .or_else(|err| {
                            warn_once_on_sync_fallback(
                                self.family,
                                "prefill_sdpa_gather",
                                attn_layer_idx,
                                &err,
                            );
                            adapter.read_kv_range(attn_layer_idx, 0, total_ctx)
                        })
                } else {
                    adapter.read_kv_range(attn_layer_idx, 0, total_ctx)
                }
                .map_err(Error::from_reason)?;
                let mask = match prefill_mask {
                    Some(mask) => mask.clone(),
                    None => {
                        create_causal_mask(seq_len as i32, Some(cached_prefix_len as i32), None)?
                    }
                };
                scaled_dot_product_attention(
                    queries_bhtd,
                    &k_full,
                    &v_full,
                    self.scale,
                    Some(&mask),
                )
            }
        }
    }
}
