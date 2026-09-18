//! K2-Horizon decoder layer.
//!
//! Dense pre-norm transformer block (HF remote code `K2HorizonDecoderLayer`):
//!
//! ```text
//! h = x + attention(input_layernorm(x))
//! out = h + mlp(post_attention_layernorm(h))
//! ```
//!
//! Both norms are K2's grouped RMSNorm; attention is GQA; the MLP is a
//! dense SwiGLU (`MLPVariant` so persistence can install mxfp8 quantized
//! projections in place). No conv, no MoE, no sliding window — every layer
//! is identical, so there is no per-layer kind dispatch (unlike LFM2).

use crate::array::MxArray;
use crate::models::quantized_linear::MLPVariant;
use crate::nn::GroupedRMSNorm;
use crate::transformer::KVCache;
use crate::transformer::MLP;
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};
use napi::bindgen_prelude::*;

use super::attention::K2Attention;
use super::config::K2HorizonConfig;

/// One K2-Horizon decoder layer.
pub struct K2DecoderLayer {
    pub(crate) attention: K2Attention,
    pub(crate) mlp: MLPVariant,
    /// `model.layers.N.input_layernorm.weight`
    pub(crate) input_layernorm: GroupedRMSNorm,
    /// `model.layers.N.post_attention_layernorm.weight`
    pub(crate) post_attention_layernorm: GroupedRMSNorm,
}

impl K2DecoderLayer {
    pub fn new(config: &K2HorizonConfig) -> Result<Self> {
        let h = config.hidden_size;
        let attention = K2Attention::new(
            h,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim(),
            config.rope_theta(),
        )?;
        let mlp = MLPVariant::Standard(MLP::new(h as u32, config.intermediate_size as u32)?);
        let input_layernorm = GroupedRMSNorm::new(
            h as i64,
            config.layernorm_num_groups as i64,
            config.norm_eps,
        )?;
        let post_attention_layernorm = GroupedRMSNorm::new(
            h as i64,
            config.layernorm_num_groups as i64,
            config.norm_eps,
        )?;
        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Flat forward — x: `[B, T, hidden]` → `[B, T, hidden]`.
    pub fn forward(
        &self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut KVCache>,
    ) -> Result<MxArray> {
        let normed = self.input_layernorm.forward(x)?;
        let r = self.attention.forward(&normed, mask, cache)?;
        let h = x.add(&r)?;

        let ffn_normed = self.post_attention_layernorm.forward(&h)?;
        let ffn_out = self.mlp.forward(&ffn_normed)?;
        h.add(&ffn_out)
    }

    /// Paged forward — every K2 layer routes through the adapter (all
    /// layers are full attention, so `layer_idx` IS the adapter ordinal).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_paged(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        layer_idx: u32,
        first_logical_position: u32,
        cached_prefix_len: u32,
        is_prefill: bool,
    ) -> Result<MxArray> {
        let normed = self.input_layernorm.forward(x)?;
        let r = self.attention.forward_paged(
            &normed,
            adapter,
            layer_idx,
            first_logical_position,
            cached_prefix_len,
            is_prefill,
        )?;
        let h = x.add(&r)?;

        let ffn_normed = self.post_attention_layernorm.forward(&h)?;
        let ffn_out = self.mlp.forward(&ffn_normed)?;
        h.add(&ffn_out)
    }

    /// Batched paged decode — `[N,1,H]` rows, one token each.
    pub(crate) fn forward_paged_batched(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        layer_idx: u32,
        rows: &[(SeqId, u32)],
    ) -> Result<MxArray> {
        let normed = self.input_layernorm.forward(x)?;
        let r = self
            .attention
            .forward_paged_batched(&normed, adapter, layer_idx, rows)?;
        let h = x.add(&r)?;

        let ffn_normed = self.post_attention_layernorm.forward(&h)?;
        let ffn_out = self.mlp.forward(&ffn_normed)?;
        h.add(&ffn_out)
    }
}
