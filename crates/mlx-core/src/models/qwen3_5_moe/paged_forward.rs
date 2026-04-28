//! Block-paged forward dispatch helpers for Qwen3.5 MoE.
//!
//! Mirrors `crate::models::qwen3_5::paged_forward` but threads through
//! the MoE `DecoderLayer` (which holds an MoE/dense MLP variant) and
//! its own `forward_paged_or_flat` method.

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::nn::{Embedding, RMSNorm};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

use super::decoder_layer::{DecoderLayer, Qwen3_5LayerKind};
use super::layer_cache::Qwen3_5LayerCache;
use super::quantized_linear::LinearProj;

/// Forward the cached-prefix tokens through GDN (linear-attention)
/// layers ONLY. Same pattern as the dense helper.
pub(crate) fn run_gdn_only_prefill(
    prefix_tokens: &[u32],
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
) -> Result<()> {
    if prefix_tokens.is_empty() {
        return Ok(());
    }
    let input_ids = MxArray::from_uint32(prefix_tokens, &[1, prefix_tokens.len() as i64])?;
    let mut hidden_states = embed.forward(&input_ids)?;

    let num_layers = layers.len();
    #[allow(clippy::needless_range_loop)]
    for layer_idx in 0..num_layers {
        if !layers[layer_idx].is_linear() {
            continue;
        }
        let cache_slot = unsafe {
            let ptr = caches.as_mut_ptr().add(layer_idx);
            &mut *ptr
        };
        hidden_states =
            layers[layer_idx].forward(&hidden_states, None, Some(cache_slot), None, true)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_prefill_chunk(
    full_tokens: &[u32],
    suffix_tokens: &[u32],
    cached_prefix_len: u32,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
) -> Result<MxArray> {
    if suffix_tokens.is_empty() {
        return Err(Error::from_reason(
            "MoE run_paged_prefill_chunk called with empty suffix",
        ));
    }

    paged_adapter
        .record_tokens(suffix_tokens)
        .map_err(Error::from_reason)?;

    if cached_prefix_len > 0 {
        let prefix = &full_tokens[..(cached_prefix_len as usize)];
        run_gdn_only_prefill(prefix, embed, layers, caches)?;
    }

    let suffix_len = suffix_tokens.len() as i64;
    let input_ids = MxArray::from_uint32(suffix_tokens, &[1, suffix_len])?;
    let mut hidden_states = embed.forward(&input_ids)?;

    let num_layers = layers.len();
    let first_logical_position = cached_prefix_len;

    #[allow(clippy::needless_range_loop)]
    for layer_idx in 0..num_layers {
        let kind = layer_kinds[layer_idx];
        let layer = unsafe {
            let ptr = layers.as_mut_ptr().add(layer_idx);
            &mut *ptr
        };
        let cache_slot = unsafe {
            let ptr = caches.as_mut_ptr().add(layer_idx);
            &mut *ptr
        };
        hidden_states = layer.forward_paged_or_flat(
            &hidden_states,
            kind,
            paged_adapter,
            first_logical_position,
            cached_prefix_len,
            true,
            None,
            Some(cache_slot),
            None,
            true,
        )?;
        // Smooth the prefill memory peak: every K layers, materialize the
        // residual stream so MLX can release the upstream graph nodes
        // (embedding + every prior layer's attention/MLP intermediates)
        // from the cache pool. Without this the in-flight lazy graph
        // accumulates ~50 GB on long contexts before the post-prefill
        // sync fires. Cadence is `MLX_PAGED_PREFILL_EVAL_INTERVAL` (default 8).
        crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states);
    }

    let h = final_norm.forward(&hidden_states)?;
    let logits = if let Some(head) = lm_head {
        head.forward(&h)?
    } else {
        let weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        h.matmul(&weight_t)?
    };

    let seq_len = logits.shape_at(1)?;
    let last = logits
        .slice_axis(1, seq_len - 1, seq_len)?
        .squeeze(Some(&[0, 1]))?;
    Ok(last)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_decode_step(
    token_id: u32,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
) -> Result<MxArray> {
    let first_logical_position = paged_adapter.current_token_count();
    paged_adapter
        .record_tokens(&[token_id])
        .map_err(Error::from_reason)?;

    let input_ids = MxArray::from_uint32(&[token_id], &[1, 1])?;
    let mut hidden_states = embed.forward(&input_ids)?;

    let num_layers = layers.len();
    #[allow(clippy::needless_range_loop)]
    for layer_idx in 0..num_layers {
        let kind = layer_kinds[layer_idx];
        let layer = unsafe {
            let ptr = layers.as_mut_ptr().add(layer_idx);
            &mut *ptr
        };
        let cache_slot = unsafe {
            let ptr = caches.as_mut_ptr().add(layer_idx);
            &mut *ptr
        };
        hidden_states = layer.forward_paged_or_flat(
            &hidden_states,
            kind,
            paged_adapter,
            first_logical_position,
            0,
            false,
            None,
            Some(cache_slot),
            None,
            true,
        )?;
    }

    let h = final_norm.forward(&hidden_states)?;
    let logits = if let Some(head) = lm_head {
        head.forward(&h)?
    } else {
        let weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        h.matmul(&weight_t)?
    };
    Ok(logits)
}
