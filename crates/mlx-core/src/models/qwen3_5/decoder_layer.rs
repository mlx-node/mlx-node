use crate::array::{DType, MxArray};
use crate::nn::RMSNorm;
use crate::transformer::MLP;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use crate::transformer::paged_kv_cache_adapter::SeqId;
use napi::bindgen_prelude::*;

use super::attention::{PagedPass, PagedWaveLane, Qwen3_5Attention};
use super::config::Qwen3_5Config;
use super::gated_delta_net::GatedDeltaNet;
use super::gdn_blob::{GdnWaveCells, gdn_wave_layer};
use super::layer_cache::Qwen3_5LayerCache;
use crate::models::quantized_linear::{MLPVariant, QuantizedLinear};

/// Per-layer routing kind for Qwen3.5's paged dispatch.
///
/// Mirrors `Lfm2LayerKind` / `Gemma4LayerKind` but reflects Qwen3.5's
/// hybrid layout (linear attention via GatedDeltaNet + full attention).
/// Only full-attention layers route through the paged adapter; linear
/// (GDN) layers continue to use `Qwen3_5LayerCache::Linear(ArraysCache)`
/// with no cross-request prefix reuse — vLLM's `MambaManager`-style "no
/// prefix reuse for recurrent layers" stance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen3_5LayerKind {
    /// GDN linear-attention layer that stays on the existing flat
    /// `Qwen3_5LayerCache::Linear` path.
    Linear,
    /// Full-attention layer routed through the paged adapter.
    /// `paged_idx` is the FULL-ATTENTION ORDINAL into the adapter's
    /// `LayerKVPool` (NOT the absolute decoder index). The pool is
    /// sized for `Qwen3_5Config::full_attention_layer_count()` slots.
    FullAttentionPaged { paged_idx: u32 },
}

/// Attention type for a decoder layer.
///
/// GDN carries native split packed-projection state and is the common hot path
/// in this hybrid model. Keep it inline rather than adding a pointer chase to
/// every linear-attention forward merely to equalize enum variant sizes.
#[allow(clippy::large_enum_variant)]
pub enum AttentionType {
    Linear(GatedDeltaNet),
    Full(Qwen3_5Attention),
}

/// Build the per-layer routing list for Qwen3.5 (dense or MoE).
///
/// Walks `is_linear_layer(i)` to assign:
/// * `Linear` for GDN layers,
/// * `FullAttentionPaged { paged_idx }` for full-attention layers
///   where `paged_idx` is the ordinal index counting only
///   full-attention layers (i.e. the index into the paged adapter's
///   `LayerKVPool`).
pub(crate) fn compute_layer_kinds(
    num_layers: usize,
    is_linear: impl Fn(usize) -> bool,
) -> Vec<Qwen3_5LayerKind> {
    let mut kinds = Vec::with_capacity(num_layers);
    let mut paged_idx: u32 = 0;
    for i in 0..num_layers {
        if is_linear(i) {
            kinds.push(Qwen3_5LayerKind::Linear);
        } else {
            kinds.push(Qwen3_5LayerKind::FullAttentionPaged { paged_idx });
            paged_idx += 1;
        }
    }
    kinds
}

/// Input to a decoder layer when cross-layer `AddRMSNorm` fusion is active:
/// either a materialized hidden state (layer 0 after embedding) or a deferred
/// `(h, delta)` pair from the previous layer's MLP whose sum is fused with
/// this layer's `input_layernorm` (or `final_norm` after the last layer).
pub(crate) enum LayerResidual {
    Hidden(MxArray),
    Pending { h: MxArray, delta: MxArray },
}

impl LayerResidual {
    pub(crate) fn seq_len(&self) -> Result<i64> {
        match self {
            Self::Hidden(x) => x.shape_at(1),
            Self::Pending { h, .. } => h.shape_at(1),
        }
    }

    pub(crate) fn into_hidden(self) -> Result<MxArray> {
        match self {
            Self::Hidden(x) => Ok(x),
            Self::Pending { h, delta } => h.add(&delta),
        }
    }
}

/// A single decoder layer in the Qwen3.5 dense model.
///
/// Each layer has:
/// - Either linear attention (GatedDeltaNet) or full attention (Qwen3NextAttention)
/// - Dense MLP (standard or quantized)
/// - Pre-norm architecture with residual connections
pub struct DecoderLayer {
    pub attn: AttentionType,
    pub mlp: MLPVariant,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    pub(crate) fn clear_cuda_dense_prefill(&self) {
        match &self.attn {
            AttentionType::Linear(gdn) => gdn.clear_cuda_dense_prefill(),
            AttentionType::Full(attention) => attention.clear_cuda_dense_prefill(),
        }
        self.mlp.clear_cuda_dense_prefill();
    }

    /// `h = x + res` + `normed = norm(h)` — one Metal dispatch via the fused
    /// add_rmsnorm kernel when the contract holds, else the separate ops.
    /// The fused kernel is bit-identical to `x.add(res)` + `norm.forward(h)`
    /// (same element mapping as `rms_single_row`/`rms_looped`), so this is
    /// safe on every path — eager, paged, prefill and compiled verify alike.
    pub(crate) fn add_residual_norm(
        norm: &RMSNorm,
        x: &MxArray,
        res: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        if let Some(pair) = norm.forward_residual_add(x, res) {
            return Ok(pair);
        }
        let h = x.add(res)?;
        let normed = norm.forward(&h)?;
        Ok((h, normed))
    }

    /// Whether this layer uses linear attention (derived from attention type).
    pub fn is_linear(&self) -> bool {
        matches!(self.attn, AttentionType::Linear(_))
    }

    /// Prefill tail pruning for the final layer when it is a FullAttention layer
    /// past the last tap, preserving cross-layer `AddRMSNorm` fusion: resolves
    /// `input` through `input_layernorm` (returning the completed previous
    /// layer hidden state as the first tuple element), updates the KV cache for
    /// all rows, and either skips Q/SDPA/O/MLP (`keep_last_row == false`) or
    /// computes them only on the last row, returning `LayerResidual::Pending`
    /// so `final_norm` fuses the last MLP residual too.
    pub(crate) fn forward_final_prefill_pruned_pending(
        &mut self,
        input: &LayerResidual,
        cache: Option<&mut Qwen3_5LayerCache>,
        keep_last_row: bool,
    ) -> Result<Option<(MxArray, LayerResidual)>> {
        let AttentionType::Full(attn) = &self.attn else {
            return Ok(None);
        };
        let kv_cache = match cache {
            Some(Qwen3_5LayerCache::FullAttention(kv)) => Some(kv),
            None => None,
            Some(Qwen3_5LayerCache::Linear(_)) => return Ok(None),
        };
        let seq_len = input.seq_len()?;
        if seq_len <= 1 {
            return Ok(None);
        }
        let (x, normed) = match input {
            LayerResidual::Hidden(x) => (x.clone(), self.input_layernorm.forward(x)?),
            LayerResidual::Pending { h, delta } => {
                Self::add_residual_norm(&self.input_layernorm, h, delta)?
            }
        };
        if !keep_last_row {
            attn.forward_kv_only(&normed, kv_cache)?;
            let last = x.slice_axis(1, seq_len - 1, seq_len)?;
            return Ok(Some((x, LayerResidual::Hidden(last))));
        }
        let attn_last = attn.forward_last_row_prefill(&normed, kv_cache)?;
        let x_last = x.slice_axis(1, seq_len - 1, seq_len)?;
        let (h, post_normed) =
            Self::add_residual_norm(&self.post_attention_layernorm, &x_last, &attn_last)?;
        let mlp_out = self.mlp.forward(&post_normed)?;
        Ok(Some((x, LayerResidual::Pending { h, delta: mlp_out })))
    }

    pub fn new(config: &Qwen3_5Config, layer_idx: usize) -> Result<Self> {
        let is_linear = config.is_linear_layer(layer_idx);

        let attn = if is_linear {
            AttentionType::Linear(GatedDeltaNet::new(config)?)
        } else {
            AttentionType::Full(Qwen3_5Attention::new(config)?)
        };

        let mlp = MLPVariant::Standard(MLP::new(
            config.hidden_size as u32,
            config.intermediate_size as u32,
        )?);

        let input_layernorm = RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps))?;
        let post_attention_layernorm =
            RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps))?;

        Ok(Self {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input [B, T, hidden_size]
    /// * `mask` - Attention mask (causal)
    /// * `cache` - Optional layer cache
    /// * `position_ids` - Optional [3, B, T] M-RoPE positions (VLM only, full attention layers)
    pub fn forward(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
    ) -> Result<MxArray> {
        self.forward_inner(x, mask, cache, position_ids, use_kernel)
    }

    fn forward_inner(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
    ) -> Result<MxArray> {
        self.forward_inner_with_tape(x, mask, cache, position_ids, use_kernel, None)
    }

    /// Like [`DecoderLayer::forward`], but threads an eager-MTP tape sink into
    /// the GDN (`Linear`) sublayer so the verify forward can record the
    /// per-step recurrence inputs for the rollback replay. Full-attention
    /// layers ignore the sink (it stays `None` for their tape slot). When
    /// `tape_sink` is `None`, behavior is byte-identical to `forward`.
    pub(crate) fn forward_with_tape(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
    ) -> Result<MxArray> {
        self.forward_inner_with_tape(x, mask, cache, position_ids, use_kernel, tape_sink)
    }

    fn forward_inner_with_tape(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
    ) -> Result<MxArray> {
        // Pre-norm + attention (always on full T for cache fidelity)
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = match &mut self.attn {
            AttentionType::Linear(gdn) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                gdn.forward_with_tape(&normed, mask, ac, use_kernel, tape_sink)?
            }
            AttentionType::Full(attn) => {
                // Full-attention layers do not record a GDN tape.
                let _ = tape_sink;
                let kvc = cache.and_then(|c| c.as_kv_cache_mut());
                attn.forward(&normed, mask, kvc, position_ids)?
            }
        };
        // Residual + post-attention norm (fused add_rmsnorm when eligible)
        let (h, normed) = Self::add_residual_norm(&self.post_attention_layernorm, x, &attn_out)?;
        h.add(&self.mlp.forward(&normed)?)
    }

    /// Cross-layer `AddRMSNorm` variant of [`Self::forward_inner_with_tape`]:
    /// resolves `input` through `input_layernorm` (fusing the previous layer's
    /// MLP residual add into `input_layernorm` when `input` is
    /// [`LayerResidual::Pending`]), runs the mixer (with optional CUDA deferred
    /// GDN fold) and MLP, and returns `(completed_x, LayerResidual::Pending { h, delta })`.
    /// `completed_x` is the previous layer's exact post-MLP hidden state.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn forward_inner_with_tape_pending(
        &mut self,
        input: LayerResidual,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
        cuda_deferred: Option<Option<&super::layer_cache::CudaPendingGdn>>,
    ) -> Result<(MxArray, LayerResidual)> {
        self.forward_inner_with_tape_pending_tree(
            input,
            mask,
            cache,
            position_ids,
            use_kernel,
            tape_sink,
            cuda_deferred,
            None,
        )
    }

    /// Tree-verification variant of [`Self::forward_inner_with_tape_pending`]:
    /// threads optional [`super::verify_tree::TreeDeviceViews`] into both
    /// linear (GDN) and full-attention layers while preserving cross-layer
    /// `AddRMSNorm` fusion (`tree_len <= 16`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_inner_with_tape_pending_tree(
        &mut self,
        input: LayerResidual,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
        cuda_deferred: Option<Option<&super::layer_cache::CudaPendingGdn>>,
        tree: Option<&super::verify_tree::TreeDeviceViews>,
    ) -> Result<(MxArray, LayerResidual)> {
        let (x, normed) = match input {
            LayerResidual::Hidden(x) => {
                let normed = self.input_layernorm.forward(&x)?;
                (x, normed)
            }
            LayerResidual::Pending { h, delta } => {
                Self::add_residual_norm(&self.input_layernorm, &h, &delta)?
            }
        };
        let attn_out = match &mut self.attn {
            AttentionType::Linear(gdn) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                gdn.forward_with_tape_deferred_tree(
                    &normed,
                    mask,
                    ac,
                    use_kernel,
                    tape_sink,
                    cuda_deferred,
                    tree,
                )?
            }
            AttentionType::Full(attn) => {
                let _ = tape_sink;
                let _ = cuda_deferred;
                let kvc = cache.and_then(|c| c.as_kv_cache_mut());
                attn.forward_tree(&normed, mask, kvc, position_ids, tree)?
            }
        };
        let (h, normed) = Self::add_residual_norm(&self.post_attention_layernorm, &x, &attn_out)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok((x, LayerResidual::Pending { h, delta: mlp_out }))
    }

    /// Forward pass with paged-or-flat dispatch for Qwen3.5.
    ///
    /// Branches on `kind`:
    /// * `Linear` — calls the existing flat `forward(...)` over the
    ///   per-layer `Qwen3_5LayerCache::Linear(ArraysCache)` slot. GDN
    ///   layers do NOT participate in cross-request prefix reuse on
    ///   this path (mirrors LFM2's conv-layer treatment and vLLM's
    ///   `MambaManager` default behavior).
    /// * `FullAttentionPaged { paged_idx }` — pre-norms via
    ///   `input_layernorm`, calls `Qwen3_5Attention::forward_paged` to
    ///   write K/V into the pool and compute attention, then runs the
    ///   shared MLP/residual tail.
    ///
    /// `flat_cache` is required for the `Linear` branch (it must point
    /// to the per-layer `Qwen3_5LayerCache::Linear` slot). For the
    /// `FullAttentionPaged` branch it is ignored — the adapter owns
    /// K/V.
    ///
    /// `use_kernel` is forwarded only to the flat `forward(...)` path.
    /// `position_ids` is forwarded to both: the flat `forward(...)` and the
    /// `FullAttentionPaged` branch's `forward_paged`, which applies 3-row
    /// M-RoPE over those positions for an image-bearing prefill and the
    /// scalar partial-RoPE when they are `None` (text-only).
    ///
    /// **Layer-kind / operator coherence**: an attention-type
    /// mismatch (e.g. `FullAttentionPaged` on a Linear operator)
    /// returns a descriptive error rather than panicking — matches
    /// the LFM2 dispatch pattern.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_paged_or_flat(
        &mut self,
        x: &MxArray,
        kind: Qwen3_5LayerKind,
        adapter: &mut PagedKVCacheAdapter,
        first_logical_position: u32,
        pass: PagedPass,
        mask: Option<&MxArray>,
        flat_cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
        rope_position_offset: i32,
        mrope_cache: &mut Option<(MxArray, MxArray)>,
    ) -> Result<MxArray> {
        match kind {
            Qwen3_5LayerKind::Linear => {
                // Linear (GDN) layer takes the flat path unchanged.
                let _ = adapter;
                let _ = first_logical_position;
                let _ = pass;
                let _ = rope_position_offset;
                if !matches!(self.attn, AttentionType::Linear(_)) {
                    return Err(Error::from_reason(
                        "Qwen3_5DecoderLayer::forward_paged_or_flat: kind=Linear applied to a \
                         FullAttention operator",
                    ));
                }
                self.forward(x, mask, flat_cache, position_ids, use_kernel)
            }
            Qwen3_5LayerKind::FullAttentionPaged { paged_idx } => {
                let _ = flat_cache; // adapter owns K/V for paged layers
                let _ = use_kernel;
                let _ = mask; // paged path uses internal causal mask
                let attn = match &self.attn {
                    AttentionType::Full(a) => a,
                    AttentionType::Linear(_) => {
                        return Err(Error::from_reason(
                            "Qwen3_5DecoderLayer::forward_paged_or_flat: kind=FullAttentionPaged \
                             applied to a Linear (GDN) operator",
                        ));
                    }
                };
                // Pre-norm + paged attention. `position_ids` carries M-RoPE
                // positions for the image-bearing prefill; `None` keeps the
                // scalar-offset text path.
                let normed = self.input_layernorm.forward(x)?;
                let attn_out = attn.forward_paged(
                    &normed,
                    adapter,
                    paged_idx,
                    first_logical_position,
                    pass,
                    position_ids,
                    rope_position_offset,
                    mrope_cache,
                )?;
                // Residual + post-attention norm (fused when eligible).
                let (h, normed) =
                    Self::add_residual_norm(&self.post_attention_layernorm, x, &attn_out)?;
                h.add(&self.mlp.forward(&normed)?)
            }
        }
    }

    /// [`Self::forward_paged_or_flat`] for a paged prefill chunk
    /// (`PagedPass::Prefill`, no mask, kernels on): the mixer output
    /// projection adds the residual in its GEMM store when its shape takes
    /// the epilogue GEMM (else a separate add + post-attention norm), and
    /// the MLP runs its SwiGLU and residual add in the GEMM stores where they
    /// apply. The bits of [`Self::forward_paged_or_flat`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_paged_prefill(
        &mut self,
        x: &MxArray,
        kind: Qwen3_5LayerKind,
        adapter: &mut PagedKVCacheAdapter,
        first_logical_position: u32,
        flat_cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        rope_position_offset: i32,
        mrope_cache: &mut Option<(MxArray, MxArray)>,
    ) -> Result<MxArray> {
        /// The mixer's output: with the residual, or the delta to add.
        enum Mixed {
            Hidden(MxArray),
            Delta(MxArray),
        }
        let rows = x.size()? as i64 / x.shape_at(x.ndim()?.saturating_sub(1))?.max(1);
        let normed = self.input_layernorm.forward(x)?;
        let mixed = match (kind, &mut self.attn) {
            (Qwen3_5LayerKind::Linear, AttentionType::Linear(gdn)) => {
                let ac = flat_cache.and_then(|c| c.as_arrays_cache_mut());
                if gdn.out_proj_store_route(rows)? {
                    Mixed::Hidden(gdn.forward_store_residual(&normed, ac, true, x)?)
                } else {
                    Mixed::Delta(gdn.forward_with_tape(&normed, None, ac, true, None)?)
                }
            }
            (Qwen3_5LayerKind::FullAttentionPaged { paged_idx }, AttentionType::Full(attn)) => {
                if attn.o_proj_store_route(rows)? {
                    Mixed::Hidden(attn.forward_paged_store_residual(
                        &normed,
                        adapter,
                        paged_idx,
                        first_logical_position,
                        PagedPass::Prefill,
                        position_ids,
                        rope_position_offset,
                        mrope_cache,
                        x,
                    )?)
                } else {
                    Mixed::Delta(attn.forward_paged(
                        &normed,
                        adapter,
                        paged_idx,
                        first_logical_position,
                        PagedPass::Prefill,
                        position_ids,
                        rope_position_offset,
                        mrope_cache,
                    )?)
                }
            }
            (Qwen3_5LayerKind::Linear, AttentionType::Full(_)) => {
                return Err(Error::from_reason(
                    "Qwen3_5DecoderLayer::forward_paged_prefill: kind=Linear applied to a \
                     FullAttention operator",
                ));
            }
            (Qwen3_5LayerKind::FullAttentionPaged { .. }, AttentionType::Linear(_)) => {
                return Err(Error::from_reason(
                    "Qwen3_5DecoderLayer::forward_paged_prefill: kind=FullAttentionPaged \
                     applied to a Linear (GDN) operator",
                ));
            }
        };
        let (h, normed) = match mixed {
            Mixed::Hidden(h) => {
                let normed = self.post_attention_layernorm.forward(&h)?;
                (h, normed)
            }
            Mixed::Delta(attn_out) => {
                Self::add_residual_norm(&self.post_attention_layernorm, x, &attn_out)?
            }
        };
        self.mlp.forward_store_residual(&normed, &h)
    }

    /// A text prefill chunk through a full-attention layer whose output no
    /// one reads: input norm and the attention's K/V store, no output.
    pub(crate) fn store_paged_kv(
        &self,
        x: &MxArray,
        paged_idx: u32,
        adapter: &mut PagedKVCacheAdapter,
        first_logical_position: u32,
        rope_position_offset: i32,
    ) -> Result<()> {
        let AttentionType::Full(attn) = &self.attn else {
            return Err(Error::from_reason(
                "Qwen3_5DecoderLayer::store_paged_kv: a Linear (GDN) operator stores no K/V",
            ));
        };
        let normed = self.input_layernorm.forward(x)?;
        attn.store_paged_kv(
            &normed,
            adapter,
            paged_idx,
            first_logical_position,
            rope_position_offset,
        )
    }

    /// One uniform decode forward over independent recurrent-state rows.
    pub(crate) fn forward_paged_batched(
        &mut self,
        x: &MxArray,
        kind: Qwen3_5LayerKind,
        adapter: &mut PagedKVCacheAdapter,
        rows: &[(SeqId, u32)],
        flat_cache: Option<&mut Qwen3_5LayerCache>,
    ) -> Result<MxArray> {
        self.forward_paged_batched_with_tape(x, kind, adapter, rows, flat_cache, None)
    }

    pub(crate) fn forward_paged_batched_with_tape(
        &mut self,
        x: &MxArray,
        kind: Qwen3_5LayerKind,
        adapter: &mut PagedKVCacheAdapter,
        rows: &[(SeqId, u32)],
        flat_cache: Option<&mut Qwen3_5LayerCache>,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
    ) -> Result<MxArray> {
        match kind {
            Qwen3_5LayerKind::Linear => {
                if !matches!(self.attn, AttentionType::Linear(_)) {
                    return Err(Error::from_reason(
                        "Qwen3_5DecoderLayer::forward_paged_batched: Linear kind/operator mismatch",
                    ));
                }
                self.forward_with_tape(x, None, flat_cache, None, true, tape_sink)
            }
            Qwen3_5LayerKind::FullAttentionPaged { paged_idx } => {
                if let Some(tape) = tape_sink {
                    *tape = None;
                }
                let attn = match &self.attn {
                    AttentionType::Full(attn) => attn,
                    AttentionType::Linear(_) => {
                        return Err(Error::from_reason(
                            "Qwen3_5DecoderLayer::forward_paged_batched: FullAttention kind/operator mismatch",
                        ));
                    }
                };
                let normed = self.input_layernorm.forward(x)?;
                let attn_out =
                    attn.forward_paged_batched(&normed, adapter, paged_idx, rows, false)?;
                let (h, normed) =
                    Self::add_residual_norm(&self.post_attention_layernorm, x, &attn_out)?;
                h.add(&self.mlp.forward(&normed)?)
            }
        }
    }

    /// Tape-recording variant of [`Self::forward_paged_or_flat`] for the eager
    /// paged MTP verify forward.
    ///
    /// Identical to `forward_paged_or_flat` except the `Linear` (GDN) branch
    /// records a per-layer [`GdnLayerTape`] into `tape_sink` (the rollback
    /// replay keystone), exactly like the flat
    /// [`Self::forward_with_tape`]. The `FullAttentionPaged` branch never
    /// records a tape (full-attention K/V lives in the paged pool and is
    /// rewound by `PagedKVCacheAdapter::rollback_last_tokens`); it drains
    /// `tape_sink` to `None` so the absolute-layer-indexed tape stays `None`
    /// for full-attention slots, matching `forward_inner_with_tape`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_paged_or_flat_with_tape(
        &mut self,
        x: &MxArray,
        kind: Qwen3_5LayerKind,
        adapter: &mut PagedKVCacheAdapter,
        first_logical_position: u32,
        pass: PagedPass,
        flat_cache: Option<&mut Qwen3_5LayerCache>,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
        rope_position_offset: i32,
    ) -> Result<MxArray> {
        match kind {
            Qwen3_5LayerKind::Linear => {
                let _ = adapter;
                let _ = first_logical_position;
                let _ = pass;
                let _ = rope_position_offset;
                if !matches!(self.attn, AttentionType::Linear(_)) {
                    return Err(Error::from_reason(
                        "Qwen3_5DecoderLayer::forward_paged_or_flat_with_tape: kind=Linear applied \
                         to a FullAttention operator",
                    ));
                }
                self.forward_with_tape(x, None, flat_cache, None, true, tape_sink)
            }
            Qwen3_5LayerKind::FullAttentionPaged { paged_idx } => {
                let _ = flat_cache; // adapter owns K/V for paged layers
                // Full-attention layers do not record a GDN tape.
                if let Some(slot) = tape_sink {
                    *slot = None;
                }
                let attn = match &self.attn {
                    AttentionType::Full(a) => a,
                    AttentionType::Linear(_) => {
                        return Err(Error::from_reason(
                            "Qwen3_5DecoderLayer::forward_paged_or_flat_with_tape: \
                             kind=FullAttentionPaged applied to a Linear (GDN) operator",
                        ));
                    }
                };
                let normed = self.input_layernorm.forward(x)?;
                // MTP tape forwards always carry M-RoPE positions as `None`
                // (the K+1 verify ids are re-embedded, not image features), so
                // RoPE takes the scalar-offset path. `rope_position_offset`
                // still carries any cross-turn delta: a text turn that
                // warm-continues an image prefill runs paged MTP and must
                // rotate at the compressed position, not the physical slot.
                let attn_out = attn.forward_paged(
                    &normed,
                    adapter,
                    paged_idx,
                    first_logical_position,
                    pass,
                    None,
                    rope_position_offset,
                    &mut None,
                )?;
                let (h, normed) =
                    Self::add_residual_norm(&self.post_attention_layernorm, x, &attn_out)?;
                h.add(&self.mlp.forward(&normed)?)
            }
        }
    }

    /// [`Self::forward_paged_or_flat_with_tape`] for a DFlash2 verify
    /// (`PagedPass::Verify`): a mixer output projection with planes adds the
    /// residual in its `_r` store and the post-attention norm runs on that
    /// hidden state alone; the MLP runs [`MLPVariant::forward_verify`]. A
    /// projection without planes runs as in
    /// [`Self::forward_paged_or_flat_with_tape`], with its bits.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_verify_with_tape(
        &mut self,
        x: &MxArray,
        kind: Qwen3_5LayerKind,
        adapter: &mut PagedKVCacheAdapter,
        first_logical_position: u32,
        flat_cache: Option<&mut Qwen3_5LayerCache>,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
        rope_position_offset: i32,
    ) -> Result<MxArray> {
        /// The mixer's output: with the residual, or the delta to add.
        enum Mixed {
            Hidden(MxArray),
            Delta(MxArray),
        }
        let rows = x.size()? as i64 / x.shape_at(x.ndim()?.saturating_sub(1))?.max(1);
        let mixed = match (kind, &mut self.attn) {
            (Qwen3_5LayerKind::Linear, AttentionType::Linear(gdn)) => {
                let normed = self.input_layernorm.forward(x)?;
                let ac = flat_cache.and_then(|c| c.as_arrays_cache_mut());
                if gdn.out_proj_verify_route(rows) {
                    Mixed::Hidden(gdn.forward_verify_residual_with_tape(&normed, ac, tape_sink, x)?)
                } else {
                    Mixed::Delta(gdn.forward_with_tape(&normed, None, ac, true, tape_sink)?)
                }
            }
            (Qwen3_5LayerKind::FullAttentionPaged { paged_idx }, AttentionType::Full(attn)) => {
                // Full-attention layers do not record a GDN tape.
                if let Some(slot) = tape_sink {
                    *slot = None;
                }
                let normed = self.input_layernorm.forward(x)?;
                if attn.o_proj_verify_route(rows) {
                    Mixed::Hidden(attn.forward_paged_verify_residual(
                        &normed,
                        adapter,
                        paged_idx,
                        first_logical_position,
                        rope_position_offset,
                        x,
                    )?)
                } else {
                    Mixed::Delta(attn.forward_paged(
                        &normed,
                        adapter,
                        paged_idx,
                        first_logical_position,
                        PagedPass::Verify,
                        None,
                        rope_position_offset,
                        &mut None,
                    )?)
                }
            }
            (Qwen3_5LayerKind::Linear, AttentionType::Full(_)) => {
                return Err(Error::from_reason(
                    "Qwen3_5DecoderLayer::forward_verify_with_tape: kind=Linear applied to a \
                     FullAttention operator",
                ));
            }
            (Qwen3_5LayerKind::FullAttentionPaged { .. }, AttentionType::Linear(_)) => {
                return Err(Error::from_reason(
                    "Qwen3_5DecoderLayer::forward_verify_with_tape: kind=FullAttentionPaged \
                     applied to a Linear (GDN) operator",
                ));
            }
        };
        let (h, normed) = match mixed {
            Mixed::Hidden(h) => {
                let normed = self.post_attention_layernorm.forward(&h)?;
                (h, normed)
            }
            Mixed::Delta(attn_out) => {
                Self::add_residual_norm(&self.post_attention_layernorm, x, &attn_out)?
            }
        };
        self.mlp.forward_verify(&normed, &h)
    }

    /// One layer of a verify wave, `x` `[N, T, H]` (lane `i` = row `i`):
    /// norms, residuals and SwiGLU stacked, the mixer through
    /// [`gdn_wave_layer`] (blob row `row`) or
    /// [`Qwen3_5Attention::forward_paged_wave_residual`], the MLP through
    /// [`MLPVariant::forward_verify_lanes`]. A mixer output projection with
    /// planes adds the residual in its `_r` store over the lanes and the
    /// post-attention norm runs on that hidden state alone. Each lane's rows
    /// carry the bits [`Self::forward_verify_with_tape`] gives that lane
    /// alone.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_wave(
        &self,
        x: &MxArray,
        kind: Qwen3_5LayerKind,
        layer_idx: usize,
        row: usize,
        adapter: &mut PagedKVCacheAdapter,
        attn_lanes: &[PagedWaveLane],
        gdn_lanes: &mut [GdnWaveCells<'_>],
    ) -> Result<MxArray> {
        let normed = self.input_layernorm.forward(x)?;
        let lanes = x.shape_at(0)?;
        let (residual, mixed) = match (kind, &self.attn) {
            (Qwen3_5LayerKind::Linear, AttentionType::Linear(gdn)) => {
                let residual = gdn.out_proj_verify_lanes_route(lanes).then_some(x);
                let out = gdn_wave_layer(gdn, &normed, layer_idx, row, gdn_lanes, residual)?;
                (residual, out)
            }
            (Qwen3_5LayerKind::FullAttentionPaged { paged_idx }, AttentionType::Full(attn)) => {
                let residual = attn.o_proj_verify_lanes_route(lanes).then_some(x);
                let out = attn.forward_paged_wave_residual(
                    &normed, adapter, paged_idx, attn_lanes, residual,
                )?;
                (residual, out)
            }
            _ => {
                return Err(Error::from_reason(format!(
                    "Qwen3_5DecoderLayer::forward_wave: layer {layer_idx} kind/operator mismatch"
                )));
            }
        };
        let (h, normed) = match residual {
            Some(_) => {
                let normed = self.post_attention_layernorm.forward(&mixed)?;
                (mixed, normed)
            }
            None => Self::add_residual_norm(&self.post_attention_layernorm, x, &mixed)?,
        };
        self.mlp.forward_verify_lanes(&normed, &h)
    }

    // ========== Weight accessors ==========

    pub fn set_input_layernorm_weight(&mut self, w: &MxArray, compute_dtype: DType) -> Result<()> {
        self.input_layernorm
            .set_weight(&super::sidecar_to_compute_dtype(w, compute_dtype)?)
    }

    pub fn set_post_attention_layernorm_weight(
        &mut self,
        w: &MxArray,
        compute_dtype: DType,
    ) -> Result<()> {
        self.post_attention_layernorm
            .set_weight(&super::sidecar_to_compute_dtype(w, compute_dtype)?)
    }

    // ========== Weight getters (for training parameter extraction) ==========

    pub fn get_input_layernorm_weight(&self) -> MxArray {
        self.input_layernorm.get_weight()
    }

    pub fn get_post_attention_layernorm_weight(&self) -> MxArray {
        self.post_attention_layernorm.get_weight()
    }

    /// Replace the dense MLP with a quantized version. Attempts the packed
    /// gate|up row-merge (`finalize_gate_up`); incompatible pairs simply stay
    /// unmerged, but an internal merge error fails the load loud.
    pub fn set_quantized_dense_mlp(
        &mut self,
        gate_proj: QuantizedLinear,
        up_proj: QuantizedLinear,
        down_proj: QuantizedLinear,
    ) -> Result<()> {
        self.mlp = MLPVariant::Quantized {
            gate_proj,
            up_proj,
            down_proj,
            gate_up: None,
        };
        self.mlp.finalize_gate_up()
    }

    /// Whether any main-model projection in this decoder layer is quantized.
    pub fn is_quantized(&self) -> bool {
        let attention = match &self.attn {
            AttentionType::Linear(gdn) => gdn.is_quantized(),
            AttentionType::Full(attn) => attn.is_quantized(),
        };
        attention || self.mlp.is_quantized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_layer_kinds_dense_qwen35_default_interval() {
        // Default `full_attention_interval = 4` makes layer 3, 7, 11,
        // ... full-attention; everything else is GDN.
        let n = 12usize;
        let kinds = compute_layer_kinds(n, |i| (i + 1) % 4 != 0);

        let mut paged_idx = 0u32;
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let expected_full = (i + 1).is_multiple_of(4);
            match kinds[i] {
                Qwen3_5LayerKind::Linear => {
                    assert!(!expected_full, "layer {i} should be Linear");
                }
                Qwen3_5LayerKind::FullAttentionPaged { paged_idx: pidx } => {
                    assert!(expected_full, "layer {i} should be FullAttentionPaged");
                    assert_eq!(pidx, paged_idx, "paged_idx mismatch at layer {i}");
                    paged_idx += 1;
                }
            }
        }
        // 12/4 = 3 full attention layers expected
        assert_eq!(paged_idx, 3);
    }

    #[test]
    fn test_compute_layer_kinds_all_linear() {
        // full_attention_interval <= 0 → all linear
        let n = 5usize;
        let kinds = compute_layer_kinds(n, |_| true);
        for k in &kinds {
            assert!(matches!(k, Qwen3_5LayerKind::Linear));
        }
    }

    #[test]
    fn test_compute_layer_kinds_all_full_attention() {
        // is_linear always false → all paged
        let n = 4usize;
        let kinds = compute_layer_kinds(n, |_| false);
        for (i, k) in kinds.iter().enumerate() {
            match k {
                Qwen3_5LayerKind::FullAttentionPaged { paged_idx } => {
                    assert_eq!(*paged_idx, i as u32);
                }
                _ => panic!("expected FullAttentionPaged at {i}"),
            }
        }
    }
}
