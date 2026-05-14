use crate::array::MxArray;
use crate::nn::RMSNorm;
use crate::transformer::MLP;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use napi::bindgen_prelude::*;

use super::attention::Qwen3_5Attention;
use super::config::Qwen3_5Config;
use super::debug::log_tensor_stats;
use super::gated_delta_net::GatedDeltaNet;
use super::layer_cache::Qwen3_5LayerCache;
use super::quantized_linear::{MLPVariant, QuantizedLinear};

/// Per-layer routing kind for Qwen3.5's paged dispatch.
///
/// Only full-attention layers route through the paged adapter; GDN layers
/// continue to use their flat `Qwen3_5LayerCache::Linear` state.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Qwen3_5LayerKind {
    Linear,
    FullAttentionPaged { paged_idx: u32 },
}

/// Build the per-layer routing list for Qwen3.5 dense/MoE paged dispatch.
pub(crate) fn compute_layer_kinds(
    num_layers: usize,
    is_linear: impl Fn(usize) -> bool,
) -> Vec<Qwen3_5LayerKind> {
    let mut kinds = Vec::with_capacity(num_layers);
    let mut paged_idx = 0u32;
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
    layer_idx: usize,
}

impl DecoderLayer {
    /// Whether this layer uses linear attention (derived from attention type).
    pub fn is_linear(&self) -> bool {
        matches!(self.attn, AttentionType::Linear(_))
    }

    pub fn new(config: &Qwen3_5Config, layer_idx: usize) -> Result<Self> {
        let is_linear = config.is_linear_layer(layer_idx);

        let attn = if is_linear {
            AttentionType::Linear(GatedDeltaNet::new(config, layer_idx)?)
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
            layer_idx,
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
        self.forward_with_rope_delta(x, mask, cache, position_ids, use_kernel, 0)
    }

    pub fn forward_with_rope_delta(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: Option<&MxArray>,
        use_kernel: bool,
        rope_offset_delta: i32,
    ) -> Result<MxArray> {
        // Pre-norm + attention
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = match &mut self.attn {
            AttentionType::Linear(gdn) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                // Linear attention layers don't use explicit position IDs
                gdn.forward(&normed, mask, ac, use_kernel)?
            }
            AttentionType::Full(attn) => {
                let kvc = cache.and_then(|c| c.as_kv_cache_mut());
                attn.forward(&normed, mask, kvc, position_ids, rope_offset_delta)?
            }
        };

        log_tensor_stats(
            &format!("layer.{:02}.decoder.attn_out", self.layer_idx),
            &attn_out,
        );
        // Residual connection
        let h = x.add(&attn_out)?;
        log_tensor_stats(
            &format!("layer.{:02}.decoder.post_attn_residual", self.layer_idx),
            &h,
        );

        // Pre-norm + MLP
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        log_tensor_stats(
            &format!("layer.{:02}.decoder.mlp_out", self.layer_idx),
            &mlp_out,
        );

        // Residual connection
        let out = h.add(&mlp_out)?;
        log_tensor_stats(&format!("layer.{:02}.decoder.out", self.layer_idx), &out);
        Ok(out)
    }

    /// Forward pass with paged-or-flat dispatch for Qwen3.5.
    ///
    /// GDN layers keep the existing flat cache path. Full-attention layers
    /// route through `Qwen3_5Attention::forward_paged`, using the compact
    /// full-attention ordinal (`paged_idx`) as the adapter pool index.
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
    ) -> Result<MxArray> {
        match kind {
            Qwen3_5LayerKind::Linear => {
                let _ = adapter;
                let _ = first_logical_position;
                let _ = cached_prefix_len;
                let _ = is_prefill;
                if !matches!(self.attn, AttentionType::Linear(_)) {
                    return Err(Error::from_reason(
                        "Qwen3_5DecoderLayer::forward_paged_or_flat: kind=Linear applied to a \
                         FullAttention operator",
                    ));
                }
                self.forward(x, mask, flat_cache, position_ids, use_kernel)
            }
            Qwen3_5LayerKind::FullAttentionPaged { paged_idx } => {
                let _ = flat_cache;
                let _ = position_ids;
                let _ = use_kernel;
                let _ = mask;
                let attn = match &self.attn {
                    AttentionType::Full(a) => a,
                    AttentionType::Linear(_) => {
                        return Err(Error::from_reason(
                            "Qwen3_5DecoderLayer::forward_paged_or_flat: \
                             kind=FullAttentionPaged applied to a Linear (GDN) operator",
                        ));
                    }
                };

                let normed = self.input_layernorm.forward(x)?;
                let attn_out = attn.forward_paged(
                    &normed,
                    adapter,
                    paged_idx,
                    first_logical_position,
                    cached_prefix_len,
                    is_prefill,
                )?;
                log_tensor_stats(
                    &format!("layer.{:02}.decoder.attn_out", self.layer_idx),
                    &attn_out,
                );

                let h = x.add(&attn_out)?;
                log_tensor_stats(
                    &format!("layer.{:02}.decoder.post_attn_residual", self.layer_idx),
                    &h,
                );

                let normed = self.post_attention_layernorm.forward(&h)?;
                let mlp_out = self.mlp.forward(&normed)?;
                log_tensor_stats(
                    &format!("layer.{:02}.decoder.mlp_out", self.layer_idx),
                    &mlp_out,
                );

                let out = h.add(&mlp_out)?;
                log_tensor_stats(&format!("layer.{:02}.decoder.out", self.layer_idx), &out);
                Ok(out)
            }
        }
    }

    #[cfg(target_family = "wasm")]
    pub fn forward_vlm_prefill(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Qwen3_5LayerCache>,
        position_ids: &MxArray,
        use_kernel: bool,
    ) -> Result<MxArray> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = match &mut self.attn {
            AttentionType::Linear(gdn) => {
                let ac = cache.and_then(|c| c.as_arrays_cache_mut());
                gdn.forward(&normed, mask, ac, use_kernel)?
            }
            AttentionType::Full(attn) => {
                let (attn_out, keys, values) =
                    attn.forward_vlm_prefill(&normed, mask, position_ids)?;
                if let Some(Qwen3_5LayerCache::FullAttention(kvc)) = cache {
                    kvc.set_keys(keys);
                    kvc.set_values(values);
                    kvc.set_offset(x.shape_at(1)? as i32);
                }
                attn_out
            }
        };

        let h = x.add(&attn_out)?;

        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;

        h.add(&mlp_out)
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

    /// Whether any main-model projection in this decoder layer is quantized.
    pub fn is_quantized(&self) -> bool {
        let attention = match &self.attn {
            AttentionType::Linear(gdn) => gdn.is_quantized(),
            AttentionType::Full(attn) => attn.is_quantized(),
        };
        attention || self.mlp.is_quantized()
    }
}
