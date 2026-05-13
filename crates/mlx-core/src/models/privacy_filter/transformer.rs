//! One transformer block of the OpenAI Privacy Filter.
//!
//! Pre-norm architecture: RMSNorm → attention → residual → RMSNorm → MoE → residual.
//! Matches `transformers/src/transformers/models/gpt_oss/modeling_gpt_oss.py`
//! and `mlx-lm/mlx_lm/models/gpt_oss.py` block layout.
//!
//! ```text
//!   hidden
//!     │
//!     ├── x = RMSNorm(hidden, input_layernorm)
//!     ├── attn_out = AttentionLayer(x)
//!     ├── hidden = hidden + attn_out
//!     │
//!     ├── x = RMSNorm(hidden, post_attention_layernorm)
//!     ├── mlp_out = GptOssMlp(x)
//!     └── hidden = hidden + mlp_out
//! ```
//!
//! Both RMSNorm layers share the same `rms_norm_eps` from the config and
//! reuse the existing `mx.fast.rms_norm` Metal kernel via
//! [`crate::nn::RMSNorm::from_weight`].
//!
//! The block is a pure borrow-only view: it owns no MLX state and is
//! constructed fresh on every forward pass alongside its
//! [`AttentionLayer`] and [`GptOssMlp`] components. KV caches are not
//! applicable to the bidirectional token-classification setting.

use crate::array::MxArray;
use crate::nn::RMSNorm;
use napi::bindgen_prelude::*;

use super::attention::AttentionLayer;
use super::config::PrivacyFilterConfig;
use super::experts::GptOssMlp;
use super::persistence::LayerWeights;

/// One pre-norm transformer block of the privacy-filter model.
pub struct Block<'a> {
    pub weights: &'a LayerWeights,
    pub config: &'a PrivacyFilterConfig,
    /// Shared YaRN frequencies (`[head_dim / 2]`, f32). Built once at
    /// model load and threaded through every block — see
    /// [`super::yarn::compute_yarn_freqs`].
    pub yarn_freqs: &'a MxArray,
    /// Index of this block within the stack (`0..num_hidden_layers`).
    /// Drives the per-layer attention type (sliding vs full) via
    /// [`PrivacyFilterConfig::band_for_layer`] — gpt-oss alternates
    /// sliding/full attention by default.
    pub layer_idx: usize,
}

impl<'a> Block<'a> {
    /// Run one block. Input shape `[B, T, hidden_size]`, output shape
    /// `[B, T, hidden_size]`. Dtype is preserved (bf16 for the shipped
    /// checkpoint).
    pub fn forward(&self, hidden: &MxArray) -> Result<MxArray> {
        let eps = self.config.rms_norm_eps as f64;

        // 1. Pre-attention RMSNorm.
        let pre_attn_norm = RMSNorm::from_weight(&self.weights.input_layernorm, Some(eps))?;
        let attn_in = pre_attn_norm.forward(hidden)?;

        // 2. Self-attention with banded mask + YaRN RoPE + sinks. The
        //    band depends on whether this layer is `sliding_attention`
        //    or `full_attention` per the gpt-oss default alternation.
        let attn = AttentionLayer {
            weights: &self.weights.self_attn,
            config: self.config,
            yarn_freqs: self.yarn_freqs,
        };
        let band = self.config.band_for_layer(self.layer_idx);
        let attn_out = attn.forward(&attn_in, band)?;

        // 3. First residual.
        let hidden = hidden.add(&attn_out)?;

        // 4. Pre-MLP RMSNorm.
        let post_attn_norm =
            RMSNorm::from_weight(&self.weights.post_attention_layernorm, Some(eps))?;
        let mlp_in = post_attn_norm.forward(&hidden)?;

        // 5. Sparse MoE FFN (gpt-oss-style clamped SwiGLU).
        let mlp = GptOssMlp {
            weights: &self.weights.mlp,
            config: self.config,
        };
        let mlp_out = mlp.forward(&mlp_in)?;

        // 6. Second residual.
        hidden.add(&mlp_out)
    }
}
