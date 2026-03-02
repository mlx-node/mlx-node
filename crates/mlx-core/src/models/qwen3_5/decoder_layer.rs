use crate::array::MxArray;
use crate::nn::RMSNorm;
use crate::transformer::MLP;
use napi::bindgen_prelude::*;

use super::attention::Qwen3_5Attention;
use super::config::Qwen3_5Config;
use super::gated_delta_net::GatedDeltaNet;
use super::layer_cache::Qwen3_5LayerCache;
use super::quantized_linear::{MLPVariant, QuantizedLinear};

/// Attention type for a decoder layer.
pub enum AttentionType {
    Linear(GatedDeltaNet),
    Full(Qwen3_5Attention),
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
    pub fn forward(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
    ) -> Result<MxArray> {
        // Pre-norm + attention
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = match &mut self.attn {
            AttentionType::Linear(gdn) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                gdn.forward(&normed, mask, ac)?
            }
            AttentionType::Full(attn) => {
                let kvc = cache.and_then(|c| c.as_kv_cache_mut());
                attn.forward(&normed, mask, kvc)?
            }
        };

        // Residual connection
        let h = x.add(&attn_out)?;

        // Pre-norm + MLP
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;

        // Residual connection
        h.add(&mlp_out)
    }

    // ========== Weight accessors ==========

    pub fn set_input_layernorm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.input_layernorm.set_weight(w)
    }

    pub fn set_post_attention_layernorm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.post_attention_layernorm.set_weight(w)
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
