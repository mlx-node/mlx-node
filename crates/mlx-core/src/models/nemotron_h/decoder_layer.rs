//! NemotronH decoder layer: pre-RMSNorm + ONE mixer + residual.
//!
//! Each layer has exactly ONE norm and one mixer — not the classic
//! attention+FFN two-norm block.

use crate::array::MxArray;
use crate::nn::RMSNorm;
use napi::bindgen_prelude::*;

use super::attention::NemotronHAttention;
use super::config::NemotronHConfig;
use super::layer_cache::NemotronHLayerCache;
use super::mamba2::{NemotronHMamba2Mixer, new_mamba_mixer};
use super::sparse_moe::NemotronHMoE;

pub enum NemotronHMixer {
    Mamba(NemotronHMamba2Mixer),
    Attention(NemotronHAttention),
    MoE(NemotronHMoE),
}

pub struct NemotronHDecoderLayer {
    pub(crate) norm: RMSNorm,
    pub(crate) mixer: NemotronHMixer,
}

impl NemotronHDecoderLayer {
    /// Build the layer for layer_idx, selecting the mixer from the
    /// config's block type. Unsupported kinds ("mlp") fail closed.
    pub fn new(config: &NemotronHConfig, layer_idx: usize) -> Result<Self> {
        let kind = config.layer_kind(layer_idx);
        let mixer = match kind {
            "linear_attention" => NemotronHMixer::Mamba(new_mamba_mixer(config)?),
            "full_attention" => NemotronHMixer::Attention(NemotronHAttention::new(config)?),
            "moe" => NemotronHMixer::MoE(NemotronHMoE::new(config, false)?),
            other => {
                return Err(Error::from_reason(format!(
                    "NemotronH layer {layer_idx}: unsupported mixer kind '{other}'"
                )));
            }
        };
        let norm = RMSNorm::new(config.hidden_size as u32, Some(config.layer_norm_epsilon))?;
        Ok(Self { norm, mixer })
    }
}

impl NemotronHDecoderLayer {
    /// The layer's mixer kind string (remapped block type).
    pub fn kind(&self) -> &'static str {
        match &self.mixer {
            NemotronHMixer::Mamba(_) => "linear_attention",
            NemotronHMixer::Attention(_) => "full_attention",
            NemotronHMixer::MoE(_) => "moe",
        }
    }

    /// Forward: norm -> mixer -> residual add.
    pub fn forward(&self, x: &MxArray, cache: Option<&mut NemotronHLayerCache>) -> Result<MxArray> {
        let normed = self.norm.forward(x)?;
        let mixer_out = match (&self.mixer, cache) {
            (NemotronHMixer::Mamba(m), cache) => {
                let state = match cache {
                    Some(NemotronHLayerCache::Mamba(s)) => Some(s),
                    Some(_) => {
                        return Err(Error::from_reason(
                            "NemotronH layer cache mismatch: Mamba mixer needs a Mamba cache",
                        ));
                    }
                    None => None,
                };
                m.forward(&normed, state)?
            }
            (NemotronHMixer::Attention(a), cache) => {
                let kvc = match cache {
                    Some(NemotronHLayerCache::Attention(c)) => Some(c),
                    Some(_) => {
                        return Err(Error::from_reason(
                            "NemotronH layer cache mismatch: Attention mixer needs an Attention cache",
                        ));
                    }
                    None => None,
                };
                a.forward(&normed, None, kvc)?
            }
            (NemotronHMixer::MoE(m), _) => m.forward(&normed)?,
        };
        x.add(&mixer_out)
    }

    /// Install the pre-mixer RMSNorm weight (the persistence loader's only
    /// write path for it). No matching reader: unlike qwen3_5's GDN layer,
    /// NemotronH has no compiled forward to re-upload it into.
    pub fn set_norm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.norm.set_weight(w)
    }

    /// Mutable accessors for the persistence layer.
    pub fn mamba_mut(&mut self) -> Option<&mut NemotronHMamba2Mixer> {
        match &mut self.mixer {
            NemotronHMixer::Mamba(m) => Some(m),
            _ => None,
        }
    }

    pub fn attention_mut(&mut self) -> Option<&mut NemotronHAttention> {
        match &mut self.mixer {
            NemotronHMixer::Attention(a) => Some(a),
            _ => None,
        }
    }

    pub fn moe_mut(&mut self) -> Option<&mut NemotronHMoE> {
        match &mut self.mixer {
            NemotronHMixer::MoE(m) => Some(m),
            _ => None,
        }
    }

    /// Whether any mixer projection holds a quantized backend.
    pub fn is_quantized(&self) -> bool {
        match &self.mixer {
            NemotronHMixer::Mamba(m) => m.is_quantized(),
            NemotronHMixer::Attention(a) => a.is_quantized(),
            NemotronHMixer::MoE(m) => m.is_quantized(),
        }
    }
}
