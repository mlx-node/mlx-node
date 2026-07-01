use crate::array::MxArray;
use crate::nn::RMSNorm;
use crate::transformer::MLP;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use napi::bindgen_prelude::*;

use super::attention::Qwen3_5Attention;
use super::config::Qwen3_5Config;
use super::gated_delta_net::GatedDeltaNet;
use super::layer_cache::Qwen3_5LayerCache;
use super::quantized_linear::{MLPVariant, QuantizedLinear};

/// Per-layer routing kind for Qwen3.5's paged dispatch.
///
/// Mirrors `Lfm2LayerKind` / `Gemma4LayerKind` but reflects Qwen3.5's
/// hybrid layout (linear attention via GatedDeltaNet + full attention).
/// Only full-attention layers route through the paged adapter; linear
/// (GDN) layers continue to use `Qwen3_5LayerCache::Linear(ArraysCache)`
/// with no cross-request prefix reuse — vLLM's `MambaManager`-style "no
/// prefix reuse for recurrent layers" stance.
#[derive(Debug, Clone, Copy)]
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
    /// Whether this layer uses linear attention (derived from attention type).
    pub fn is_linear(&self) -> bool {
        matches!(self.attn, AttentionType::Linear(_))
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
        self.forward_inner(x, mask, cache, position_ids, use_kernel, false)
    }

    /// Like `forward`, but if `slice_to_last_before_mlp` is true, slices the
    /// post-attention activation to just the last token before running the
    /// MLP + final residual. Used by the LAST decoder layer during prefill
    /// when only the last token's hidden state will be consumed downstream
    /// (lm_head only samples from the last token).
    ///
    /// The attention sublayer always runs on the full T input so the
    /// K/V (or GDN recurrent) caches are updated for every prefill token.
    pub fn forward_with_optional_last_slice(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
        slice_to_last_before_mlp: bool,
    ) -> Result<MxArray> {
        self.forward_inner(
            x,
            mask,
            cache,
            position_ids,
            use_kernel,
            slice_to_last_before_mlp,
        )
    }

    fn forward_inner(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
        slice_to_last_before_mlp: bool,
    ) -> Result<MxArray> {
        self.forward_inner_with_tape(
            x,
            mask,
            cache,
            position_ids,
            use_kernel,
            slice_to_last_before_mlp,
            None,
        )
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
        self.forward_inner_with_tape(x, mask, cache, position_ids, use_kernel, false, tape_sink)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner_with_tape(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
        slice_to_last_before_mlp: bool,
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

        // Residual connection
        let h = x.add(&attn_out)?;

        // E38: slice h to last token before MLP if asked (last-layer prefill
        // optimization). Saves ~120 GFLOPs of MLP work at T=860.
        let h = if slice_to_last_before_mlp {
            let seq_len = h.shape_at(1)?;
            h.slice_axis(1, seq_len - 1, seq_len)?
        } else {
            h
        };

        // Pre-norm + MLP
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;

        // Residual connection
        h.add(&mlp_out)
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
        cached_prefix_len: u32,
        is_prefill: bool,
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
                let _ = cached_prefix_len;
                let _ = is_prefill;
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
                    cached_prefix_len,
                    is_prefill,
                    position_ids,
                    rope_position_offset,
                    mrope_cache,
                )?;
                // Residual.
                let h = x.add(&attn_out)?;
                // Pre-norm + MLP.
                let normed = self.post_attention_layernorm.forward(&h)?;
                let mlp_out = self.mlp.forward(&normed)?;
                h.add(&mlp_out)
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
        cached_prefix_len: u32,
        is_prefill: bool,
        flat_cache: Option<&mut Qwen3_5LayerCache>,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
        rope_position_offset: i32,
    ) -> Result<MxArray> {
        match kind {
            Qwen3_5LayerKind::Linear => {
                let _ = adapter;
                let _ = first_logical_position;
                let _ = cached_prefix_len;
                let _ = is_prefill;
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
                    cached_prefix_len,
                    is_prefill,
                    None,
                    rope_position_offset,
                    &mut None,
                )?;
                let h = x.add(&attn_out)?;
                let normed = self.post_attention_layernorm.forward(&h)?;
                let mlp_out = self.mlp.forward(&normed)?;
                h.add(&mlp_out)
            }
        }
    }

    // ========== Weight accessors ==========

    pub fn set_input_layernorm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.input_layernorm.set_weight(w)
    }

    pub fn set_post_attention_layernorm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.post_attention_layernorm.set_weight(w)
    }

    // ========== Weight getters (for training parameter extraction) ==========

    pub fn get_input_layernorm_weight(&self) -> MxArray {
        self.input_layernorm.get_weight()
    }

    pub fn get_post_attention_layernorm_weight(&self) -> MxArray {
        self.post_attention_layernorm.get_weight()
    }

    /// Replace the dense MLP with a quantized version.
    pub fn set_quantized_dense_mlp(
        &mut self,
        gate_proj: QuantizedLinear,
        up_proj: QuantizedLinear,
        down_proj: QuantizedLinear,
    ) {
        self.mlp = MLPVariant::Quantized {
            gate_proj,
            up_proj,
            down_proj,
        };
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
