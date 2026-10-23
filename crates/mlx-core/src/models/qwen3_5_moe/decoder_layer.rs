use crate::array::MxArray;
use crate::nn::RMSNorm;
use crate::transformer::MLP;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use crate::transformer::paged_kv_cache_adapter::SeqId;
use napi::bindgen_prelude::*;

use super::attention::Qwen3_5Attention;
use super::config::Qwen3_5MoeConfig;
use super::gated_delta_net::GatedDeltaNet;
use super::layer_cache::Qwen3_5LayerCache;
use super::sparse_moe::SparseMoeBlock;
use crate::models::quantized_linear::{MLPVariant, QuantizedLinear};
// Reuse the dense layer-kind enum for MoE; the routing semantics
// (Linear vs FullAttentionPaged) are identical between dense and MoE.
pub(crate) use crate::models::qwen3_5::decoder_layer::{
    LayerVerifyIo, Qwen3_5LayerKind, VerifyResidual,
};

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

/// MLP type for a decoder layer.
///
/// The `Dense` variant carries an inline `MLPVariant` (which embeds a dense
/// `MLP`); the `MoE` variant is already boxed. The `MLP` struct gained the
/// Stage-3 int8 W8A8 prefill quant fields (`gate_up_w_i8`/`s_w`,
/// `down_w_i8`/`s_w`), pushing the inline `Dense` payload just past clippy's
/// 200-byte `large_enum_variant` threshold. Boxing `Dense` would ripple through
/// ~37 MoE match sites for no runtime benefit — this enum is constructed once
/// per layer at load, never in a hot loop — so we allow the size asymmetry.
#[allow(clippy::large_enum_variant)]
pub enum MLPType {
    Dense(MLPVariant),
    MoE(Box<SparseMoeBlock>),
}

impl MLPType {
    /// Forward `finalize_gate_up` to the dense variant (MoE blocks merge
    /// their shared expert via `set_quantized_shared_expert`).
    pub fn finalize_gate_up(&mut self) -> Result<()> {
        match self {
            MLPType::Dense(mlp) => mlp.finalize_gate_up(),
            MLPType::MoE(_) => Ok(()),
        }
    }
}

/// Owned counterpart of VerifyResidual; meaning is fixed by the graph output.
/// Source combine returns Hidden, while ordinary FFN stays Pending.
pub(crate) enum VerifyLayerOutput {
    Hidden(MxArray),
    Pending { h: MxArray, delta: MxArray },
}
impl VerifyLayerOutput {
    pub(crate) fn as_input(&self) -> VerifyResidual<'_> {
        match self {
            Self::Hidden(h) => VerifyResidual::Hidden(h),
            Self::Pending { h, delta } => VerifyResidual::Pending { h, delta },
        }
    }
    pub(crate) fn normalize(&self, norm: &RMSNorm) -> Result<(MxArray, MxArray)> {
        match self {
            Self::Hidden(h) => Ok((h.clone(), norm.forward(h)?)),
            Self::Pending { h, delta } => DecoderLayer::add_residual_norm(norm, h, delta),
        }
    }
}

/// A single decoder layer in the Qwen3.5 MoE model.
pub struct DecoderLayer {
    pub attn: AttentionType,
    pub mlp: MLPType,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    source_mixer: bool,
}

impl DecoderLayer {
    pub(crate) fn configure_source_mixer(&mut self, enabled: bool) {
        self.source_mixer = enabled;
    }

    fn source_mixer_rows(&self, x: &MxArray) -> Result<bool> {
        Ok(self.source_mixer
            && x.shape()?.as_ref() == [1, 8, 2048]
            && x.dtype()? == crate::array::DType::BFloat16)
    }

    fn finish_mixer(
        norm: &RMSNorm,
        x: &MxArray,
        output: MxArray,
        includes_residual: bool,
    ) -> Result<(MxArray, MxArray)> {
        if includes_residual {
            let normalized = norm.forward(&output)?;
            Ok((output, normalized))
        } else {
            Self::add_residual_norm(norm, x, &output)
        }
    }

    pub(crate) fn add_residual_norm(
        norm: &RMSNorm,
        x: &MxArray,
        residual: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        crate::models::qwen3_5::decoder_layer::DecoderLayer::add_residual_norm(norm, x, residual)
    }

    pub(crate) fn forward_verify(
        &mut self,
        input: VerifyResidual<'_>,
        io: &mut LayerVerifyIo<'_>,
        use_kernel: bool,
        tape_sink: Option<&mut Option<super::gated_delta_net::GdnLayerTape>>,
        row_budget: Option<&MxArray>,
    ) -> Result<(MxArray, VerifyLayerOutput)> {
        let (x, normed) = match input {
            VerifyResidual::Hidden(x) => (x.clone(), self.input_layernorm.forward(x)?),
            VerifyResidual::Pending { h, delta } => {
                Self::add_residual_norm(&self.input_layernorm, h, delta)?
            }
        };
        let includes_residual = self.source_mixer_rows(&x)?;
        let attn_out = match (&mut self.attn, io) {
            (AttentionType::Linear(gdn), LayerVerifyIo::Linear(ac)) => {
                if includes_residual {
                    gdn.forward_source_residual_with_tape(
                        &normed,
                        None,
                        Some(&mut **ac),
                        use_kernel,
                        tape_sink,
                        &x,
                    )?
                } else {
                    gdn.forward_with_tape(&normed, None, Some(&mut **ac), use_kernel, tape_sink)?
                }
            }
            (AttentionType::Full(attn), LayerVerifyIo::FullAttention(aio)) => {
                if includes_residual {
                    attn.forward_verify_source_residual(&normed, aio, &x)?
                } else {
                    attn.forward_verify(&normed, aio)?
                }
            }
            (AttentionType::Linear(_), _) | (AttentionType::Full(_), _) => {
                return Err(Error::from_reason(
                    "Qwen3.5 MoE compiled verify: layer/cache io kind mismatch",
                ));
            }
        };

        let (h, normed) = Self::finish_mixer(
            &self.post_attention_layernorm,
            &x,
            attn_out,
            includes_residual,
        )?;
        if let MLPType::MoE(moe) = &self.mlp {
            if let Some(hidden) = moe.forward_source_residual(&normed, &h, row_budget)? {
                return Ok((x, VerifyLayerOutput::Hidden(hidden)));
            }
        }
        let mlp_out = match &self.mlp {
            MLPType::Dense(mlp) => mlp.forward(&normed)?,
            MLPType::MoE(moe) => moe.forward_with_budget(&normed, row_budget)?,
        };
        Ok((x, VerifyLayerOutput::Pending { h, delta: mlp_out }))
    }

    fn forward_mlp_decode_rows(&self, x: &MxArray) -> Result<MxArray> {
        let batch = usize::try_from(x.shape_at(0)?)
            .map_err(|_| Error::from_reason("Qwen3.5 MoE decode MLP received a negative batch"))?;
        if batch == 0 {
            return Err(Error::from_reason(
                "Qwen3.5 MoE decode MLP requires at least one row",
            ));
        }
        let rows = (0..batch)
            .map(|row| {
                let row = x.slice_axis(0, row as i64, row as i64 + 1)?;
                match &self.mlp {
                    MLPType::Dense(mlp) => mlp.forward(&row),
                    MLPType::MoE(moe) => moe.forward(&row),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        MxArray::concatenate_many(rows.iter().collect(), Some(0))
    }

    pub fn is_linear(&self) -> bool {
        matches!(self.attn, AttentionType::Linear(_))
    }

    pub fn new(config: &Qwen3_5MoeConfig, layer_idx: usize) -> Result<Self> {
        let is_linear = config.is_linear_layer(layer_idx);

        // Attention and GatedDeltaNet need a Qwen3_5Config-compatible interface.
        // We create a temporary Qwen3_5Config from the MoE config for shared types.
        let dense_config = config.to_dense_config();

        let attn = if is_linear {
            AttentionType::Linear(GatedDeltaNet::new(&dense_config)?)
        } else {
            AttentionType::Full(Qwen3_5Attention::new(&dense_config)?)
        };

        let is_moe = config.is_moe_layer(layer_idx);
        let mlp = if is_moe {
            MLPType::MoE(Box::new(SparseMoeBlock::new(config)?))
        } else {
            MLPType::Dense(MLPVariant::Standard(MLP::new(
                config.hidden_size as u32,
                config.intermediate_size as u32,
            )?))
        };

        let input_layernorm = RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps))?;
        let post_attention_layernorm =
            RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps))?;

        Ok(Self {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            source_mixer: false,
        })
    }

    pub fn forward(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
    ) -> Result<MxArray> {
        let normed = self.input_layernorm.forward(x)?;
        let includes_residual = self.source_mixer_rows(x)?;
        let attn_out = match &mut self.attn {
            AttentionType::Linear(gdn) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                if includes_residual {
                    gdn.forward_source_residual_with_tape(&normed, mask, ac, use_kernel, None, x)?
                } else {
                    gdn.forward(&normed, mask, ac, use_kernel)?
                }
            }
            AttentionType::Full(attn) => {
                let kvc = cache.and_then(|c| c.as_kv_cache_mut());
                if includes_residual {
                    attn.forward_source_residual(&normed, mask, kvc, position_ids, x)?
                } else {
                    attn.forward(&normed, mask, kvc, position_ids)?
                }
            }
        };

        let (h, normed) = Self::finish_mixer(
            &self.post_attention_layernorm,
            x,
            attn_out,
            includes_residual,
        )?;
        if let MLPType::MoE(moe) = &self.mlp {
            if let Some(hidden) = moe.forward_source_residual(&normed, &h, None)? {
                return Ok(hidden);
            }
        }
        let mlp_out = match &self.mlp {
            MLPType::Dense(mlp) => mlp.forward(&normed)?,
            MLPType::MoE(moe) => moe.forward(&normed)?,
        };

        h.add(&mlp_out)
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
        let normed = self.input_layernorm.forward(x)?;
        let includes_residual = self.source_mixer_rows(x)?;
        let attn_out = match &mut self.attn {
            AttentionType::Linear(gdn) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                if includes_residual {
                    gdn.forward_source_residual_with_tape(
                        &normed, mask, ac, use_kernel, tape_sink, x,
                    )?
                } else {
                    gdn.forward_with_tape(&normed, mask, ac, use_kernel, tape_sink)?
                }
            }
            AttentionType::Full(attn) => {
                let _ = tape_sink;
                let kvc = cache.and_then(|c| c.as_kv_cache_mut());
                if includes_residual {
                    attn.forward_source_residual(&normed, mask, kvc, position_ids, x)?
                } else {
                    attn.forward(&normed, mask, kvc, position_ids)?
                }
            }
        };

        let (h, normed) = Self::finish_mixer(
            &self.post_attention_layernorm,
            x,
            attn_out,
            includes_residual,
        )?;
        if let MLPType::MoE(moe) = &self.mlp {
            if let Some(hidden) = moe.forward_source_residual(&normed, &h, None)? {
                return Ok(hidden);
            }
        }
        let mlp_out = match &self.mlp {
            MLPType::Dense(mlp) => mlp.forward(&normed)?,
            MLPType::MoE(moe) => moe.forward(&normed)?,
        };

        h.add(&mlp_out)
    }

    /// Paged-or-flat dispatch for the MoE decoder layer.
    ///
    /// Same shape as the dense `DecoderLayer::forward_paged_or_flat`
    /// (see that method's rustdoc) but threaded through the MoE
    /// MLP/expert tail. `Linear` layers stay on the flat
    /// `Qwen3_5LayerCache::Linear` path; `FullAttentionPaged` layers
    /// route through `Qwen3_5Attention::forward_paged`.
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
                let _ = adapter;
                let _ = first_logical_position;
                let _ = cached_prefix_len;
                let _ = is_prefill;
                let _ = rope_position_offset;
                if !matches!(self.attn, AttentionType::Linear(_)) {
                    return Err(Error::from_reason(
                        "Qwen3_5MoeDecoderLayer::forward_paged_or_flat: kind=Linear applied to a \
                         FullAttention operator",
                    ));
                }
                self.forward(x, mask, flat_cache, position_ids, use_kernel)
            }
            Qwen3_5LayerKind::FullAttentionPaged { paged_idx } => {
                let _ = flat_cache;
                let _ = use_kernel;
                let _ = mask;
                let attn = match &self.attn {
                    AttentionType::Full(a) => a,
                    AttentionType::Linear(_) => {
                        return Err(Error::from_reason(
                            "Qwen3_5MoeDecoderLayer::forward_paged_or_flat: \
                             kind=FullAttentionPaged applied to a Linear (GDN) operator",
                        ));
                    }
                };
                let normed = self.input_layernorm.forward(x)?;
                // `position_ids` carries M-RoPE positions for an image-bearing
                // prefill; `None` keeps the scalar-offset text path.
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
                let (h, normed) =
                    Self::add_residual_norm(&self.post_attention_layernorm, x, &attn_out)?;
                let mlp_out = match &self.mlp {
                    MLPType::Dense(mlp) => mlp.forward(&normed)?,
                    MLPType::MoE(moe) => moe.forward(&normed)?,
                };
                h.add(&mlp_out)
            }
        }
    }

    /// Tape-recording twin of [`DecoderLayer::forward_paged_or_flat`] for the
    /// MoE eager paged MTP verify forward.
    ///
    /// `Linear` layers stay on the flat `Qwen3_5LayerCache::Linear` path and
    /// record their per-step recurrence inputs into `tape_sink` (the replay
    /// keystone the rollback consumes); `FullAttentionPaged` layers route
    /// through the adapter and clear their tape slot. Sparse expert routing
    /// already flattens `[B, T, H]`, so the `[1, K+1, H]` verify block runs
    /// the same expert tail as a prefill chunk.
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
                        "Qwen3_5MoeDecoderLayer::forward_paged_or_flat_with_tape: kind=Linear \
                         applied to a FullAttention operator",
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
                            "Qwen3_5MoeDecoderLayer::forward_paged_or_flat_with_tape: \
                             kind=FullAttentionPaged applied to a Linear (GDN) operator",
                        ));
                    }
                };
                let normed = self.input_layernorm.forward(x)?;
                // MTP tape forwards always carry M-RoPE positions as `None`
                // (the K+1 verify ids are re-embedded tokens, not image
                // features), so RoPE takes the scalar-offset path.
                // `rope_position_offset` still carries any cross-turn delta: a
                // text turn that warm-continues an image prefill runs paged MTP
                // and must rotate at the compressed position.
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
                let (h, normed) =
                    Self::add_residual_norm(&self.post_attention_layernorm, x, &attn_out)?;
                let mlp_out = match &self.mlp {
                    MLPType::Dense(mlp) => mlp.forward(&normed)?,
                    MLPType::MoE(moe) => moe.forward(&normed)?,
                };
                h.add(&mlp_out)
            }
        }
    }

    /// One text decode forward over independent paged/GDN request rows.
    /// Sparse expert routing already flattens `[B,T,H]`, so the MoE tail uses
    /// the same batched tensor as attention without a serial fallback.
    pub(crate) fn forward_paged_batched(
        &mut self,
        x: &MxArray,
        kind: Qwen3_5LayerKind,
        adapter: &mut PagedKVCacheAdapter,
        rows: &[(SeqId, u32)],
        flat_cache: Option<&mut Qwen3_5LayerCache>,
        preserve_singleton_projection_graphs: bool,
    ) -> Result<MxArray> {
        self.forward_paged_batched_with_tape(
            x,
            kind,
            adapter,
            rows,
            flat_cache,
            preserve_singleton_projection_graphs,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_paged_batched_with_tape(
        &mut self,
        x: &MxArray,
        kind: Qwen3_5LayerKind,
        adapter: &mut PagedKVCacheAdapter,
        rows: &[(SeqId, u32)],
        flat_cache: Option<&mut Qwen3_5LayerCache>,
        preserve_singleton_projection_graphs: bool,
        tape_sink: Option<&mut Option<crate::models::qwen3_5::gated_delta_net::GdnLayerTape>>,
    ) -> Result<MxArray> {
        match kind {
            Qwen3_5LayerKind::Linear => {
                if !matches!(self.attn, AttentionType::Linear(_)) {
                    return Err(Error::from_reason(
                        "Qwen3_5MoeDecoderLayer::forward_paged_batched: Linear kind/operator mismatch",
                    ));
                }
                let normed = self.input_layernorm.forward(x)?;
                let attn_out = match &mut self.attn {
                    AttentionType::Linear(gdn) => {
                        let cache = flat_cache
                            .and_then(|cache| cache.as_arrays_cache_mut())
                            .ok_or_else(|| {
                                Error::from_reason(
                                    "Qwen3.5 MoE batched GDN layer has no recurrent cache",
                                )
                            })?;
                        if preserve_singleton_projection_graphs {
                            let capture = tape_sink.is_some();
                            let mut tapes = Vec::new();
                            let (output, next_cache) =
                                crate::models::qwen3_5::arrays_cache::forward_rows_independently(
                                    &normed,
                                    cache,
                                    |row, row_cache| {
                                        let mut tape = None;
                                        let output = gdn.forward_with_tape(
                                            row,
                                            None,
                                            Some(row_cache),
                                            true,
                                            capture.then_some(&mut tape),
                                        )?;
                                        if capture {
                                            tapes.push(tape.ok_or_else(|| {
                                                Error::from_reason(
                                                    "GDN verify did not record an owner tape",
                                                )
                                            })?);
                                        }
                                        Ok(output)
                                    },
                                )?;
                            if let Some(sink) = tape_sink {
                                *sink = Some(crate::models::qwen3_5::gated_delta_net::GdnLayerTape::stack_rows(&tapes)?);
                            }
                            *cache = next_cache;
                            output
                        } else {
                            gdn.forward_with_tape(&normed, None, Some(cache), true, tape_sink)?
                        }
                    }
                    AttentionType::Full(_) => {
                        return Err(Error::from_reason(
                            "Qwen3_5MoeDecoderLayer::forward_paged_batched: Linear operator disappeared",
                        ));
                    }
                };
                let (h, normed) =
                    Self::add_residual_norm(&self.post_attention_layernorm, x, &attn_out)?;
                let mlp_out = if preserve_singleton_projection_graphs {
                    self.forward_mlp_decode_rows(&normed)?
                } else {
                    match &self.mlp {
                        MLPType::Dense(mlp) => mlp.forward(&normed)?,
                        MLPType::MoE(moe) => moe.forward(&normed)?,
                    }
                };
                h.add(&mlp_out)
            }
            Qwen3_5LayerKind::FullAttentionPaged { paged_idx } => {
                if let Some(tape) = tape_sink {
                    *tape = None;
                }
                let attn = match &self.attn {
                    AttentionType::Full(attn) => attn,
                    AttentionType::Linear(_) => {
                        return Err(Error::from_reason(
                            "Qwen3_5MoeDecoderLayer::forward_paged_batched: FullAttention kind/operator mismatch",
                        ));
                    }
                };
                let normed = self.input_layernorm.forward(x)?;
                let attn_out = attn.forward_paged_batched(
                    &normed,
                    adapter,
                    paged_idx,
                    rows,
                    preserve_singleton_projection_graphs,
                )?;
                let (h, normed) =
                    Self::add_residual_norm(&self.post_attention_layernorm, x, &attn_out)?;
                let mlp_out = if preserve_singleton_projection_graphs {
                    self.forward_mlp_decode_rows(&normed)?
                } else {
                    match &self.mlp {
                        MLPType::Dense(mlp) => mlp.forward(&normed)?,
                        MLPType::MoE(moe) => moe.forward(&normed)?,
                    }
                };
                h.add(&mlp_out)
            }
        }
    }

    pub(crate) fn enable_source_residual_norms(&mut self, source_reduction: bool) -> Result<()> {
        if source_reduction {
            self.input_layernorm.enable_source_f32_splash()?;
            self.post_attention_layernorm.enable_source_f32_splash()
        } else {
            self.input_layernorm.enable_source_f32_reference()?;
            self.post_attention_layernorm.enable_source_f32_reference()
        }
    }

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

    pub fn set_quantized_dense_mlp(
        &mut self,
        gate_proj: QuantizedLinear,
        up_proj: QuantizedLinear,
        down_proj: QuantizedLinear,
    ) -> Result<()> {
        self.mlp = MLPType::Dense(MLPVariant::Quantized {
            gate_proj,
            up_proj,
            down_proj,
            gate_up: None,
        });
        self.mlp.finalize_gate_up()
    }

    /// Whether any main-model projection in this decoder layer is quantized,
    /// including routed experts and shared-expert projections.
    pub fn is_quantized(&self) -> bool {
        let attention = match &self.attn {
            AttentionType::Linear(gdn) => gdn.is_quantized(),
            AttentionType::Full(attn) => attn.is_quantized(),
        };
        let mlp = match &self.mlp {
            MLPType::Dense(mlp) => mlp.is_quantized(),
            MLPType::MoE(moe) => moe.is_quantized(),
        };
        attention || mlp
    }
}

#[cfg(test)]
#[path = "source_residual_tests.rs"]
mod source_residual_tests;

#[cfg(test)]
#[path = "source_mixer_tests.rs"]
mod source_mixer_tests;
