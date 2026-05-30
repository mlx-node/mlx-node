use crate::array::MxArray;
use crate::models::qwen3_5_moe::quantized_linear::MLPVariant;
use crate::nn::RMSNorm;
use crate::transformer::MLP;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use napi::bindgen_prelude::*;

use super::attention::Lfm2Attention;
use super::config::Lfm2Config;
use super::layer_cache::Lfm2LayerCache;
use super::short_conv::ShortConv;
use super::sparse_moe::Lfm2SparseMoeBlock;

/// Per-layer routing kind for the paged dispatch.
///
/// Mirrors vLLM's hybrid coordinator pattern (single shared block pool +
/// per-layer-type managers) but keeps the pragmatic "attention-only paged,
/// flat fallback" stance described in the messages-kv-reuse docs:
/// only `FullAttention` layers go through the `PagedKVCacheAdapter`; conv
/// layers continue to use `Lfm2LayerCache::Conv(ArraysCache)`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Lfm2LayerKind {
    /// Full-attention layer routed through the paged adapter.
    /// `paged_idx` is the ATTENTION-LAYER ORDINAL into the adapter's
    /// `LayerKVPool` (NOT the absolute decoder index). The pool is sized
    /// for `config.full_attn_idxs().len()` slots.
    FullAttention { paged_idx: u32 },
    /// Conv layer that stays on the existing `Lfm2LayerCache::Conv` path.
    Conv,
}

/// Operator type: either a conv layer or an attention layer.
pub(crate) enum OperatorType {
    Conv(ShortConv),
    Attention(Lfm2Attention),
}

/// Per-layer feed-forward block.
///
/// Dense LFM2 checkpoints (and the first `num_dense_layers` layers of a MoE
/// checkpoint) use a standard SwiGLU MLP; the remaining MoE-checkpoint
/// layers use a sparse top-k `Lfm2SparseMoeBlock`. Both expose the same
/// `forward(&MxArray) -> Result<MxArray>` contract so the residual path in
/// the decoder layer is identical.
///
/// The dense arm holds an `MLPVariant` (shared with qwen3_5) so its gate/up/
/// down projections can each be quantized in ANY mode (affine / mxfp4 / mxfp8
/// / nvfp4) — the quantized arm runs three `QuantizedLinear::forward` + swiglu
/// with no dense `get_weight()` materialization, while the `Standard` arm is
/// the eager-dense `MLP` (default at construction).
pub(crate) enum FeedForward {
    Dense(MLPVariant),
    Moe(Lfm2SparseMoeBlock),
}

impl FeedForward {
    fn forward(&self, x: &MxArray) -> Result<MxArray> {
        match self {
            FeedForward::Dense(mlp) => mlp.forward(x),
            FeedForward::Moe(moe) => moe.forward(x),
        }
    }
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
    pub(crate) feed_forward: FeedForward,
    pub(crate) operator_norm: RMSNorm,
    pub(crate) ffn_norm: RMSNorm,
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

        // Per-layer feed-forward: sparse MoE for MoE-checkpoint layers
        // (`layer_idx >= num_dense_layers`), dense SwiGLU otherwise.
        let feed_forward = if config.is_moe_layer(layer_idx) {
            FeedForward::Moe(Lfm2SparseMoeBlock::new(config)?)
        } else {
            // Dense-in-MoE layers use `intermediate_size` DIRECTLY (no 2/3
            // `computed_ff_dim()` shrink — see lfm2_moe.py MLP). Pure-dense
            // checkpoints keep the existing `computed_ff_dim()` path.
            let ff_dim = if config.is_moe() {
                config.intermediate_size.ok_or_else(|| {
                    Error::from_reason(
                        "lfm2_moe dense layer requires intermediate_size in config.json",
                    )
                })?
            } else {
                config.computed_ff_dim()
            };
            FeedForward::Dense(MLPVariant::Standard(MLP::new(h as u32, ff_dim as u32)?))
        };

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

    /// Forward pass with paged-or-flat dispatch.
    ///
    /// Branches on `kind`:
    /// * `FullAttention { paged_idx }` — routes through the
    ///   `PagedKVCacheAdapter` (writes K/V into the shared pool, runs
    ///   SDPA over `read_kv_range` for cache-hit prefill or
    ///   `gather_kv_for_decode` for decode).
    /// * `Conv` — forwards through the existing flat `ShortConv` path
    ///   using the layer's `Lfm2LayerCache::Conv(ArraysCache)` slot. Conv
    ///   layers do NOT participate in cross-request prefix reuse on this
    ///   commit (mirrors vLLM's `MambaManager` default behavior of "no
    ///   prefix reuse for recurrent layers").
    ///
    /// `cache` is required for the `Conv` branch; it must be the per-layer
    /// `Lfm2LayerCache::Conv` slot. For `FullAttention` `cache` is
    /// ignored (the adapter is the source of truth).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_paged_or_flat(
        &self,
        x: &MxArray,
        kind: Lfm2LayerKind,
        adapter: &mut PagedKVCacheAdapter,
        first_logical_position: u32,
        cached_prefix_len: u32,
        is_prefill: bool,
        conv_cache: Option<&mut Lfm2LayerCache>,
    ) -> Result<MxArray> {
        // Pre-norm
        let normed = self.operator_norm.forward(x)?;

        // Operator dispatch
        let r = match (kind, &self.operator) {
            (Lfm2LayerKind::FullAttention { paged_idx }, OperatorType::Attention(attn)) => attn
                .forward_paged(
                    &normed,
                    adapter,
                    paged_idx,
                    first_logical_position,
                    cached_prefix_len,
                    is_prefill,
                )?,
            (Lfm2LayerKind::Conv, OperatorType::Conv(conv)) => {
                let conv_cache_slot = conv_cache.and_then(|c| c.as_conv_cache_mut());
                conv.forward(&normed, conv_cache_slot)?
            }
            (Lfm2LayerKind::FullAttention { .. }, OperatorType::Conv(_)) => {
                return Err(Error::from_reason(
                    "Lfm2DecoderLayer::forward_paged_or_flat: layer kind \
                     FullAttention applied to a Conv operator",
                ));
            }
            (Lfm2LayerKind::Conv, OperatorType::Attention(_)) => {
                return Err(Error::from_reason(
                    "Lfm2DecoderLayer::forward_paged_or_flat: layer kind \
                     Conv applied to an Attention operator",
                ));
            }
        };

        // Residual connection
        let h = x.add(&r)?;

        // FFN with pre-norm + residual
        let ffn_normed = self.ffn_norm.forward(&h)?;
        let ffn_out = self.feed_forward.forward(&ffn_normed)?;
        h.add(&ffn_out)
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

    /// Get a mutable reference to the dense feed-forward `MLPVariant`.
    /// Returns `None` if this layer uses a sparse MoE block.
    ///
    /// The returned `MLPVariant` is `Standard` at construction; the
    /// persistence layer swaps it to `Quantized` in place when the dense-MLP
    /// projections ship `.scales`.
    pub fn dense_mlp_mut(&mut self) -> Option<&mut MLPVariant> {
        match &mut self.feed_forward {
            FeedForward::Dense(m) => Some(m),
            FeedForward::Moe(_) => None,
        }
    }

    /// Get a mutable reference to the sparse MoE feed-forward block.
    /// Returns `None` if this layer uses a dense MLP.
    pub fn moe_mut(&mut self) -> Option<&mut Lfm2SparseMoeBlock> {
        match &mut self.feed_forward {
            FeedForward::Moe(m) => Some(m),
            FeedForward::Dense(_) => None,
        }
    }

    /// Whether this layer's feed-forward is a sparse MoE block.
    pub fn is_moe_layer(&self) -> bool {
        matches!(self.feed_forward, FeedForward::Moe(_))
    }
}
