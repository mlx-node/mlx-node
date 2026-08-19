use crate::array::MxArray;
use crate::models::gemma4::layer_cache::Gemma4LayerCache;
use crate::nn::RMSNorm;
use crate::transformer::paged_kv_cache_adapter::{PagedKVCacheAdapter, SeqId};
use napi::bindgen_prelude::*;

use super::attention::MuseGlimmerAttention;
use super::kv_cache::PagedWindowSlot;
use super::mlp::MuseGlimmerMlp;

pub struct MuseGlimmerDecoderLayer {
    pub attention: MuseGlimmerAttention,
    pub mlp: MuseGlimmerMlp,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    pre_feedforward_layernorm: RMSNorm,
    post_feedforward_layernorm: RMSNorm,
}

impl MuseGlimmerDecoderLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        attention: MuseGlimmerAttention,
        mlp: MuseGlimmerMlp,
        input_layernorm: RMSNorm,
        post_attention_layernorm: RMSNorm,
        pre_feedforward_layernorm: RMSNorm,
        post_feedforward_layernorm: RMSNorm,
    ) -> Self {
        Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        }
    }

    pub fn forward(&self, x: &MxArray, cache: &mut Gemma4LayerCache) -> Result<MxArray> {
        let attn = self
            .attention
            .forward(&self.input_layernorm.forward(x)?, cache)?;
        let h = x.add(&self.post_attention_layernorm.forward(&attn)?)?;
        let ffn = self
            .mlp
            .forward(&self.pre_feedforward_layernorm.forward(&h)?)?;
        h.add(&self.post_feedforward_layernorm.forward(&ffn)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_paged(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        paged_idx: u32,
        first_logical_position: u32,
        cached_prefix_len: u32,
        is_prefill: bool,
        window: PagedWindowSlot,
    ) -> Result<MxArray> {
        let attn = self.attention.forward_paged(
            &self.input_layernorm.forward(x)?,
            adapter,
            paged_idx,
            first_logical_position,
            cached_prefix_len,
            is_prefill,
            window,
        )?;
        let h = x.add(&self.post_attention_layernorm.forward(&attn)?)?;
        let ffn = self
            .mlp
            .forward(&self.pre_feedforward_layernorm.forward(&h)?)?;
        h.add(&self.post_feedforward_layernorm.forward(&ffn)?)
    }

    pub(crate) fn forward_paged_batched(
        &self,
        x: &MxArray,
        adapter: &mut PagedKVCacheAdapter,
        paged_idx: u32,
        rows: &[(SeqId, u32)],
        window: PagedWindowSlot,
        preserve_singleton_projection_graphs: bool,
    ) -> Result<MxArray> {
        let attn = self.attention.forward_paged_batched(
            &self.input_layernorm.forward(x)?,
            adapter,
            paged_idx,
            rows,
            window,
            preserve_singleton_projection_graphs,
        )?;
        let h = x.add(&self.post_attention_layernorm.forward(&attn)?)?;
        let normed = self.pre_feedforward_layernorm.forward(&h)?;
        let ffn = if preserve_singleton_projection_graphs {
            super::row_exact::forward_rows_independently(&normed, |row| self.mlp.forward(row))?
        } else {
            self.mlp.forward(&normed)?
        };
        h.add(&self.post_feedforward_layernorm.forward(&ffn)?)
    }
}
