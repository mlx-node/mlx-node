use crate::array::MxArray;
use crate::nn::RMSNorm;
use crate::transformer::MLP;
use napi::bindgen_prelude::*;

use super::attention::Lfm2Attention;
use super::config::Lfm2Config;
use super::layer_cache::Lfm2LayerCache;
use super::short_conv::ShortConv;

/// Operator type: either a conv layer or an attention layer.
pub(crate) enum OperatorType {
    Conv(ShortConv),
    Attention(Lfm2Attention),
}

/// A single decoder layer in the LFM2 model.
///
/// Follows `lfm2.py:197-234` (Lfm2DecoderLayer class).
///
/// Forward:
///   r = operator(operator_norm(x), ...)   // conv or attention
///   h = x + r
///   out = h + feed_forward(ffn_norm(h))
pub struct Lfm2DecoderLayer {
    pub(crate) operator: OperatorType,
    pub(crate) feed_forward: MLP,
    operator_norm: RMSNorm,
    ffn_norm: RMSNorm,
}

impl Lfm2DecoderLayer {
    /// Create a new decoder layer.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `layer_idx` - Layer index (determines conv vs attention)
    pub fn new(config: &Lfm2Config, layer_idx: usize) -> Result<Self> {
        let is_attention = config.is_attention_layer(layer_idx);
        let h = config.hidden_size;

        let operator = if is_attention {
            OperatorType::Attention(Lfm2Attention::new(
                h,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim(),
                config.norm_eps,
                config.rope_theta,
            )?)
        } else {
            OperatorType::Conv(ShortConv::new(h, config.conv_l_cache, config.conv_bias)?)
        };

        // MLP uses the auto-adjusted ff_dim
        let ff_dim = config.computed_ff_dim();
        let feed_forward = MLP::new(h as u32, ff_dim as u32)?;

        let eps = Some(config.norm_eps);
        let operator_norm = RMSNorm::new(h as u32, eps)?;
        let ffn_norm = RMSNorm::new(h as u32, eps)?;

        Ok(Self {
            operator,
            feed_forward,
            operator_norm,
            ffn_norm,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input [B, T, hidden_size]
    /// * `mask` - Attention mask (used only for attention layers)
    /// * `cache` - Layer cache (KVCache for attention, ArraysCache for conv)
    ///
    /// # Returns
    /// Output tensor [B, T, hidden_size]
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Lfm2LayerCache>,
    ) -> Result<MxArray> {
        // Pre-norm
        let normed = self.operator_norm.forward(x)?;

        // Operator dispatch
        let r = match &self.operator {
            OperatorType::Conv(conv) => {
                let conv_cache = cache.and_then(|c| c.as_conv_cache_mut());
                conv.forward(&normed, conv_cache)?
            }
            OperatorType::Attention(attn) => {
                let attn_cache = cache.and_then(|c| c.as_kv_cache_mut());
                attn.forward(&normed, mask, attn_cache)?
            }
        };

        // Residual connection
        let h = x.add(&r)?;

        // FFN with pre-norm + residual
        let ffn_normed = self.ffn_norm.forward(&h)?;
        let ffn_out = self.feed_forward.forward(&ffn_normed)?;
        h.add(&ffn_out)
    }

    /// Whether this layer uses attention (vs conv).
    pub fn is_attention_layer(&self) -> bool {
        matches!(&self.operator, OperatorType::Attention(_))
    }

    // ========== Norm weight setters ==========

    pub fn set_operator_norm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.operator_norm.set_weight(w)
    }

    pub fn set_ffn_norm_weight(&mut self, w: &MxArray) -> Result<()> {
        self.ffn_norm.set_weight(w)
    }

    // ========== Operator weight setters (delegate) ==========

    /// Get a mutable reference to the attention operator.
    /// Returns None if this layer is a conv layer.
    pub fn attention_mut(&mut self) -> Option<&mut Lfm2Attention> {
        match &mut self.operator {
            OperatorType::Attention(attn) => Some(attn),
            OperatorType::Conv(_) => None,
        }
    }

    /// Get a mutable reference to the conv operator.
    /// Returns None if this layer is an attention layer.
    pub fn conv_mut(&mut self) -> Option<&mut ShortConv> {
        match &mut self.operator {
            OperatorType::Conv(conv) => Some(conv),
            OperatorType::Attention(_) => None,
        }
    }

    /// Get a mutable reference to the feed-forward MLP.
    pub fn feed_forward_mut(&mut self) -> &mut MLP {
        &mut self.feed_forward
    }
}
