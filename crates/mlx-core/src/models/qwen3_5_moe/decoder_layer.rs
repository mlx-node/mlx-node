use crate::array::MxArray;
use crate::nn::RMSNorm;
use crate::transformer::MLP;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use napi::bindgen_prelude::*;

use super::attention::Qwen3_5Attention;
use super::config::Qwen3_5MoeConfig;
use super::gated_delta_net::GatedDeltaNet;
use super::layer_cache::Qwen3_5LayerCache;
use super::quantized_linear::{MLPVariant, QuantizedLinear};
use super::sparse_moe::SparseMoeBlock;
// Reuse the dense layer-kind enum for MoE; the routing semantics
// (Linear vs FullAttentionPaged) are identical between dense and MoE.
pub(crate) use crate::models::qwen3_5::decoder_layer::Qwen3_5LayerKind;

/// Attention type for a decoder layer.
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

/// A single decoder layer in the Qwen3.5 MoE model.
pub struct DecoderLayer {
    pub attn: AttentionType,
    pub mlp: MLPType,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
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
        let attn_out = match &mut self.attn {
            AttentionType::Linear(gdn) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                gdn.forward(&normed, mask, ac, use_kernel)?
            }
            AttentionType::Full(attn) => {
                let kvc = cache.and_then(|c| c.as_kv_cache_mut());
                attn.forward(&normed, mask, kvc, position_ids)?
            }
        };

        let h = x.add(&attn_out)?;

        let normed = self.post_attention_layernorm.forward(&h)?;
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
        let attn_out = match &mut self.attn {
            AttentionType::Linear(gdn) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                gdn.forward_with_tape(&normed, mask, ac, use_kernel, tape_sink)?
            }
            AttentionType::Full(attn) => {
                let _ = tape_sink;
                let kvc = cache.and_then(|c| c.as_kv_cache_mut());
                attn.forward(&normed, mask, kvc, position_ids)?
            }
        };

        let h = x.add(&attn_out)?;

        let normed = self.post_attention_layernorm.forward(&h)?;
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
                let h = x.add(&attn_out)?;
                let normed = self.post_attention_layernorm.forward(&h)?;
                let mlp_out = match &self.mlp {
                    MLPType::Dense(mlp) => mlp.forward(&normed)?,
                    MLPType::MoE(moe) => moe.forward(&normed)?,
                };
                h.add(&mlp_out)
            }
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
    ) {
        self.mlp = MLPType::Dense(MLPVariant::Quantized {
            gate_proj,
            up_proj,
            down_proj,
        });
    }
}
