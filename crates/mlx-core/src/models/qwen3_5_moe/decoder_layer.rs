use crate::array::MxArray;
use crate::nn::RMSNorm;
use crate::transformer::MLP;
use napi::bindgen_prelude::*;

use super::attention::Qwen3_5Attention;
use super::config::Qwen3_5MoeConfig;
use super::gated_delta_net::GatedDeltaNet;
use super::layer_cache::Qwen3_5LayerCache;
use super::quantized_linear::{MLPVariant, QuantizedLinear};
use super::sparse_moe::SparseMoeBlock;

/// Attention type for a decoder layer.
pub enum AttentionType {
    Linear(GatedDeltaNet),
    Full(Qwen3_5Attention),
}

/// MLP type for a decoder layer.
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

    pub fn is_moe(&self) -> bool {
        matches!(self.mlp, MLPType::MoE(_))
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
    ) -> Result<MxArray> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = match &mut self.attn {
            AttentionType::Linear(gdn) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                gdn.forward(&normed, mask, ac)?
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

    pub fn set_input_layernorm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.input_layernorm.set_weight(w)
    }

    pub fn set_post_attention_layernorm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.post_attention_layernorm.set_weight(w)
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
