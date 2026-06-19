//! Model-neutral vision merge contract.
//!
//! The qwen3.5 dense and MoE families share one image->text merge pipeline:
//! the vision encoder produces image features, those features are scattered
//! into the text token embeddings at the image-placeholder positions, and
//! M-RoPE position ids are computed for the mixed image+text sequence. The
//! product of that pipeline is [`VisionMerge`] — the embeddings + positions
//! that feed the language-model prefill.
//!
//! Both families then run the SAME prefill layer loop over that merge; the
//! only thing that differs is the attention-mask policy (dense relies on the
//! attention layer's internal causal SDPA and passes no mask; MoE builds an
//! explicit causal mask). [`run_vlm_prefill_layers`] is that shared loop,
//! generic over the family's layer + cache types via [`VlmPrefillLayer`].

use napi::Result;

use crate::array::MxArray;

/// The output of the vision encode + image-feature merge + M-RoPE position
/// computation: everything the language-model prefill needs for an
/// image-bearing turn.
pub(crate) struct VisionMerge {
    /// Token embeddings with image features scattered into the image
    /// placeholder slots. Shape `[1, seq_len, hidden]`, text-embedding dtype.
    pub inputs_embeds: MxArray,
    /// M-RoPE position ids for the mixed image+text sequence. Shape
    /// `[3, 1, seq_len]` (the temporal/height/width axes), i32.
    pub position_ids: MxArray,
    /// `max_position + 1 - seq_len`, the scalar offset a follow-up turn would
    /// re-anchor from. Persisted by the caller but inert on the eager path:
    /// decode re-derives positions from the KV-cache scalar offset.
    pub rope_deltas: i64,
}

/// A decoder layer that can participate in the shared VLM prefill loop. Each
/// family implements this for its own `DecoderLayer` (and names its own
/// layer-cache type via [`Self::Cache`]), so [`run_vlm_prefill_layers`] stays
/// model-neutral.
pub(crate) trait VlmPrefillLayer {
    /// The family's per-layer KV / recurrent cache type.
    type Cache;

    /// Whether this is a (GDN) linear-attention layer. Linear layers take no
    /// attention mask and no M-RoPE position ids.
    fn vlm_is_linear(&self) -> bool;

    /// Run the layer's prefill forward (kernel path enabled).
    fn vlm_forward(
        &mut self,
        x: &MxArray,
        mask: Option<&MxArray>,
        cache: Option<&mut Self::Cache>,
        position_ids: Option<&MxArray>,
    ) -> Result<MxArray>;
}

/// Run the shared VLM prefill layer loop and return the final hidden state
/// (pre-final-norm). The caller supplies the attention-mask policy: dense
/// passes `None` (internal causal SDPA), MoE passes an explicit causal mask.
///
/// Linear layers receive neither the mask nor the M-RoPE positions; full
/// attention layers receive both — the invariant both families must share.
pub(crate) fn run_vlm_prefill_layers<L: VlmPrefillLayer>(
    layers: &mut [L],
    caches: &mut Option<Vec<L::Cache>>,
    merge: &VisionMerge,
    mask: Option<&MxArray>,
) -> Result<MxArray> {
    let mut h = merge.inputs_embeds.clone();
    for i in 0..layers.len() {
        let is_linear = layers[i].vlm_is_linear();
        let layer_mask = if is_linear { None } else { mask };
        let layer_pos = if is_linear {
            None
        } else {
            Some(&merge.position_ids)
        };
        let cache = caches.as_mut().map(|c| &mut c[i]);
        h = layers[i].vlm_forward(&h, layer_mask, cache, layer_pos)?;
    }
    Ok(h)
}
