//! Block-paged forward dispatch helpers for Qwen3.5 (dense + MoE).
//!
//! These helpers implement the same two-pass prefill / per-step decode
//! pattern as LFM2's paged path, but adapted for Qwen3.5's hybrid layer
//! mix (GDN linear-attention layers + Qwen3_5 full-attention layers).
//!
//! Pass 1: GDN-only prefill over the cached prefix tokens (when
//! `cached_prefix_len > 0`) — brings GDN recurrent state up to position
//! `cached_prefix_len`. Attention layers are skipped on this pass; the
//! adapter pool already holds the prefix K/V from a prior request.
//!
//! Pass 2: full forward (GDN + attention) over the SUFFIX tokens.
//! Attention layers attend over `read_kv_range(0, total_ctx)` to recover
//! cached + new context.
//!
//! The decode step is a single-token forward through every layer,
//! gathering K/V from the paged pool for attention layers.
//!
//! Strategy notes (mirrors LFM2/Qwen3.5-MoE):
//! * Full-attention layers reuse K/V through the paged adapter. GDN
//!   layers can only skip prefix replay when the caller has restored a
//!   matching sidecar checkpoint (`gdn_prefix_already_primed=true`);
//!   otherwise this helper replays the cached prefix through GDN.
//! * The two-pass scheme is approximate for GDN over the cached
//!   prefix: the prefix's GDN forward sees a hidden-state stream
//!   produced by passing through ALL layers (including attention)
//!   in pass 1, but the attention layers can't run during pass 1
//!   without their K/V reaching back into the pool — so pass 1 is
//!   GDN-only, with attention layers acting as identity passthroughs
//!   (their MLP / residual contribution is approximated). This is
//!   the same limitation LFM2 documents as P1 — pure-cache-hit
//!   dispatch is not bit-equal to a fresh prefill on hybrid models.
//!   For the **no-cache** case (cached_prefix_len = 0), pass 1 is
//!   skipped entirely and the result is exact.

use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

use super::decoder_layer::{DecoderLayer, Qwen3_5LayerKind};
use super::layer_cache::Qwen3_5LayerCache;

fn bytes_to_mib(bytes: f64) -> f64 {
    bytes / (1024.0 * 1024.0)
}

fn trace_memory_mib() -> (f64, f64, f64) {
    (
        bytes_to_mib(crate::array::get_active_memory()),
        bytes_to_mib(crate::array::get_cache_memory()),
        bytes_to_mib(crate::array::get_peak_memory()),
    )
}

/// Forward the cached-prefix tokens through GDN layers ONLY. Used as
/// "pass 1" of the paged prefill when there is a non-zero cached
/// prefix.
///
/// Skips full-attention layers — their state is reconstructed from the
/// paged pool's prefix cache during pass 2's `read_kv_range`. The
/// hidden_states stream produced by pass 1 is therefore an
/// approximation that omits attention layers' MLP/residual
/// contribution; this is the same trade-off LFM2 makes (see module
/// rustdoc).
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
            // Skip attention layers — pass 2 reads their state from
            // the paged pool. Identity-passthrough on hidden_states.
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

/// Run a paged prefill over the suffix tokens. Returns the last
/// position's logits squeezed to `[vocab]`.
///
/// `cached_prefix_len` is how many tokens the paged adapter has
/// already cached for this request (0 on a fresh prefill). The full
/// prompt is `tokens` (used for the GDN pass-1 prefill of the prefix);
/// the suffix `&tokens[cached_prefix_len..]` is what gets recorded
/// into the paged adapter and fed through the full forward pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_prefill_chunk(
    full_tokens: &[u32],
    suffix_tokens: &[u32],
    cached_prefix_len: u32,
    gdn_prefix_already_primed: bool,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<Linear>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
) -> Result<MxArray> {
    let chunk_size = crate::array::paged_prefill_chunk_size();
    run_paged_prefill_chunk_with_size(
        full_tokens,
        suffix_tokens,
        cached_prefix_len,
        gdn_prefix_already_primed,
        embed,
        layers,
        caches,
        final_norm,
        lm_head,
        embedding_weight,
        layer_kinds,
        paged_adapter,
        chunk_size,
    )
}

/// Chunk-size-parameterized worker for `run_paged_prefill_chunk`.
///
/// `chunk_size <= 0` keeps the single-shot path. Positive chunk sizes split
/// only the uncached suffix. Each chunk writes its K/V into the paged adapter,
/// attends over the cumulative cached range, then clears MLX's transient graph
/// before the next chunk. This matches the Qwen3/Qwen3.5 MoE driver shape and
/// keeps dense Qwen from building one giant prefill graph for 30k+ suffixes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_prefill_chunk_with_size(
    full_tokens: &[u32],
    suffix_tokens: &[u32],
    cached_prefix_len: u32,
    gdn_prefix_already_primed: bool,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<Linear>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    chunk_size: i32,
) -> Result<MxArray> {
    if suffix_tokens.is_empty() {
        return Err(Error::from_reason(
            "run_paged_prefill_chunk called with empty suffix",
        ));
    }

    if chunk_size <= 0 || suffix_tokens.len() <= chunk_size as usize {
        return run_paged_prefill_single_shot(
            full_tokens,
            suffix_tokens,
            cached_prefix_len,
            gdn_prefix_already_primed,
            embed,
            layers,
            caches,
            final_norm,
            lm_head,
            embedding_weight,
            layer_kinds,
            paged_adapter,
        );
    }

    let trace_enabled = inference_trace_enabled();
    let chunk_size_usize = chunk_size as usize;

    // Pass 1: GDN-only prefill over the cached prefix. This runs once before
    // suffix chunking; GDN recurrent state then advances in-place across chunks.
    if cached_prefix_len > 0 && !gdn_prefix_already_primed {
        let gdn_trace_start = trace_enabled.then(Instant::now);
        let prefix = &full_tokens[..(cached_prefix_len as usize)];
        run_gdn_only_prefill(prefix, embed, layers, caches)?;
        if let Some(start) = gdn_trace_start {
            let (active_mib, cache_mib, peak_mib) = trace_memory_mib();
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-dense paged_prefill_gdn_prefix_done \
                 prefix_tokens={} elapsed_ms={:.1} active_mib={:.1} cache_mib={:.1} peak_mib={:.1}",
                cached_prefix_len,
                elapsed_ms(start),
                active_mib,
                cache_mib,
                peak_mib
            ));
        }
    }

    let total_chunks = suffix_tokens.len().div_ceil(chunk_size_usize);
    let mut last_logits: Option<MxArray> = None;
    let mut chunk_start_position = cached_prefix_len;

    for (chunk_idx, chunk) in suffix_tokens.chunks(chunk_size_usize).enumerate() {
        let is_last_chunk = chunk_idx + 1 == total_chunks;
        let chunk_trace_start = trace_enabled.then(Instant::now);

        paged_adapter
            .record_tokens(chunk)
            .map_err(Error::from_reason)?;

        let hidden_states = run_paged_prefill_one_chunk(
            chunk,
            chunk_start_position,
            embed,
            layers,
            caches,
            layer_kinds,
            paged_adapter,
        )?;

        if is_last_chunk {
            last_logits = Some(project_last_token_logits(
                &hidden_states,
                final_norm,
                lm_head,
                embedding_weight,
            )?);
            if let Some(start) = chunk_trace_start {
                let chunk_elapsed_ms = elapsed_ms(start);
                let (active_mib, cache_mib, peak_mib) = trace_memory_mib();
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-dense paged_prefill_chunk_final_graph_built \
                     chunk_index={} total_chunks={} chunk_tokens={} context_before={} context_after={} \
                     elapsed_ms={:.1} active_mib={:.1} cache_mib={:.1} peak_mib={:.1}",
                    chunk_idx + 1,
                    total_chunks,
                    chunk.len(),
                    chunk_start_position,
                    chunk_start_position + chunk.len() as u32,
                    chunk_elapsed_ms,
                    active_mib,
                    cache_mib,
                    peak_mib
                ));
            }
        } else {
            hidden_states.eval();
            crate::array::synchronize_and_clear_cache();
            if let Some(start) = chunk_trace_start {
                let chunk_elapsed_ms = elapsed_ms(start);
                let chunk_tok_s = if chunk_elapsed_ms > 0.0 {
                    chunk.len() as f64 / (chunk_elapsed_ms / 1000.0)
                } else {
                    0.0
                };
                let (active_mib, cache_mib, peak_mib) = trace_memory_mib();
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-dense paged_prefill_chunk_done \
                     chunk_index={} total_chunks={} chunk_tokens={} context_before={} context_after={} \
                     elapsed_ms={:.1} tok_s={:.2} active_mib={:.1} cache_mib={:.1} peak_mib={:.1}",
                    chunk_idx + 1,
                    total_chunks,
                    chunk.len(),
                    chunk_start_position,
                    chunk_start_position + chunk.len() as u32,
                    chunk_elapsed_ms,
                    chunk_tok_s,
                    active_mib,
                    cache_mib,
                    peak_mib
                ));
            }
        }

        chunk_start_position += chunk.len() as u32;
    }

    last_logits.ok_or_else(|| {
        Error::from_reason(
            "chunked prefill produced no last chunk (unreachable for non-empty suffix)",
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn run_paged_prefill_single_shot(
    full_tokens: &[u32],
    suffix_tokens: &[u32],
    cached_prefix_len: u32,
    gdn_prefix_already_primed: bool,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<Linear>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
) -> Result<MxArray> {
    paged_adapter
        .record_tokens(suffix_tokens)
        .map_err(Error::from_reason)?;

    if cached_prefix_len > 0 && !gdn_prefix_already_primed {
        let prefix = &full_tokens[..(cached_prefix_len as usize)];
        run_gdn_only_prefill(prefix, embed, layers, caches)?;
    }

    let hidden_states = run_paged_prefill_one_chunk(
        suffix_tokens,
        cached_prefix_len,
        embed,
        layers,
        caches,
        layer_kinds,
        paged_adapter,
    )?;

    project_last_token_logits(&hidden_states, final_norm, lm_head, embedding_weight)
}

/// Phase 4b B3.1 — paged prefill variant that ALSO returns the
/// post-`final_norm` hidden state for every prompt token,
/// concatenated along the time axis to `[1, prompt_len, hidden]`.
///
/// Mirror of `chunked_prefill_with_hidden` (dense / flat path). The
/// paged-MTP gate inside `chat_sync_core_paged_inner` consumes this so
/// `prefill_mtp_commit` can seed `g_mtp_committed_len = N` before the
/// first MTP cycle — without it the MTP draft attends over a
/// prompt-less context and parity vs the AR run breaks.
///
/// Caller MUST gate on `cached_prefix_len == 0` (the dense gate uses
/// the same `want_prompt_hidden` predicate). On a cache-reuse turn the
/// prefill only processes the suffix, so the captured hidden would not
/// cover the full prompt and `prefill_mtp_commit` cannot use it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_prefill_chunk_with_hidden(
    full_tokens: &[u32],
    suffix_tokens: &[u32],
    cached_prefix_len: u32,
    gdn_prefix_already_primed: bool,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<Linear>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    keep_last_hidden: Option<usize>,
) -> Result<(MxArray, MxArray)> {
    let chunk_size = crate::array::paged_prefill_chunk_size();
    run_paged_prefill_chunk_with_hidden_with_size(
        full_tokens,
        suffix_tokens,
        cached_prefix_len,
        gdn_prefix_already_primed,
        embed,
        layers,
        caches,
        final_norm,
        lm_head,
        embedding_weight,
        layer_kinds,
        paged_adapter,
        chunk_size,
        keep_last_hidden,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_paged_prefill_chunk_with_hidden_with_size(
    full_tokens: &[u32],
    suffix_tokens: &[u32],
    cached_prefix_len: u32,
    gdn_prefix_already_primed: bool,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<Linear>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    chunk_size: i32,
    keep_last_hidden: Option<usize>,
) -> Result<(MxArray, MxArray)> {
    if suffix_tokens.is_empty() {
        return Err(Error::from_reason(
            "run_paged_prefill_chunk_with_hidden called with empty suffix",
        ));
    }

    if chunk_size <= 0 || suffix_tokens.len() <= chunk_size as usize {
        return run_paged_prefill_single_shot_with_hidden(
            full_tokens,
            suffix_tokens,
            cached_prefix_len,
            gdn_prefix_already_primed,
            embed,
            layers,
            caches,
            final_norm,
            lm_head,
            embedding_weight,
            layer_kinds,
            paged_adapter,
            keep_last_hidden,
        );
    }

    let chunk_size_usize = chunk_size as usize;

    if cached_prefix_len > 0 && !gdn_prefix_already_primed {
        let prefix = &full_tokens[..(cached_prefix_len as usize)];
        run_gdn_only_prefill(prefix, embed, layers, caches)?;
    }

    let total_chunks = suffix_tokens.len().div_ceil(chunk_size_usize);
    let mut last_logits: Option<MxArray> = None;
    let mut hidden_chunks: Vec<MxArray> = Vec::with_capacity(total_chunks);
    let total_suffix_len = suffix_tokens.len();
    let keep_start = keep_last_hidden
        .map(|keep| total_suffix_len.saturating_sub(keep.max(1)))
        .unwrap_or(0);
    let mut chunk_start_position = cached_prefix_len;
    let mut suffix_offset = 0usize;

    for (chunk_idx, chunk) in suffix_tokens.chunks(chunk_size_usize).enumerate() {
        let is_last_chunk = chunk_idx + 1 == total_chunks;
        let chunk_start = suffix_offset;
        let chunk_end = chunk_start + chunk.len();
        let overlaps_kept_tail = chunk_end > keep_start;

        paged_adapter
            .record_tokens(chunk)
            .map_err(Error::from_reason)?;

        let hidden_states = run_paged_prefill_one_chunk(
            chunk,
            chunk_start_position,
            embed,
            layers,
            caches,
            layer_kinds,
            paged_adapter,
        )?;

        let chunk_hidden = if overlaps_kept_tail || is_last_chunk {
            Some(final_norm.forward(&hidden_states)?)
        } else {
            None
        };

        if is_last_chunk {
            // Reuse the already-normed last chunk to project last-token
            // logits — `forward` on a final_norm output is idempotent
            // would be wasteful; slice directly instead.
            let chunk_hidden = chunk_hidden.as_ref().ok_or_else(|| {
                Error::from_reason("run_paged_prefill_chunk_with_hidden: missing last hidden")
            })?;
            let chunk_len = chunk_hidden.shape_at(1)?;
            let last_hidden = chunk_hidden.slice_axis(1, chunk_len - 1, chunk_len)?;
            let logits = if let Some(head) = lm_head {
                head.forward(&last_hidden)?
            } else {
                let weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
                last_hidden.matmul(&weight_t)?
            };
            last_logits = Some(logits.squeeze(Some(&[0, 1]))?);
        }

        if let Some(chunk_hidden) = chunk_hidden
            && overlaps_kept_tail
        {
            let keep_from = keep_start.max(chunk_start);
            let kept_hidden = if keep_from > chunk_start {
                chunk_hidden.slice_axis(
                    1,
                    (keep_from - chunk_start) as i64,
                    (chunk_end - chunk_start) as i64,
                )?
            } else {
                chunk_hidden
            };
            // Materialize hidden BEFORE clear_cache; the hidden is a lazy
            // handle into graph nodes that the per-layer cache eviction
            // would otherwise free between chunks.
            kept_hidden.eval();
            hidden_chunks.push(kept_hidden);
        }
        if !is_last_chunk {
            crate::array::synchronize_and_clear_cache();
        }
        chunk_start_position += chunk.len() as u32;
        suffix_offset = chunk_end;
    }

    let last_logits = last_logits.ok_or_else(|| {
        Error::from_reason(
            "chunked prefill (with-hidden) produced no last chunk (unreachable for non-empty suffix)",
        )
    })?;

    let prompt_hidden = if hidden_chunks.len() == 1 {
        hidden_chunks.into_iter().next().ok_or_else(|| {
            Error::from_reason("run_paged_prefill_chunk_with_hidden: empty hidden chunks")
        })?
    } else {
        let mut acc = hidden_chunks[0].clone();
        for chunk in &hidden_chunks[1..] {
            acc = MxArray::concatenate(&acc, chunk, 1)?;
        }
        acc
    };

    Ok((last_logits, prompt_hidden))
}

#[allow(clippy::too_many_arguments)]
fn run_paged_prefill_single_shot_with_hidden(
    full_tokens: &[u32],
    suffix_tokens: &[u32],
    cached_prefix_len: u32,
    gdn_prefix_already_primed: bool,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<Linear>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    keep_last_hidden: Option<usize>,
) -> Result<(MxArray, MxArray)> {
    paged_adapter
        .record_tokens(suffix_tokens)
        .map_err(Error::from_reason)?;

    if cached_prefix_len > 0 && !gdn_prefix_already_primed {
        let prefix = &full_tokens[..(cached_prefix_len as usize)];
        run_gdn_only_prefill(prefix, embed, layers, caches)?;
    }

    let hidden_states = run_paged_prefill_one_chunk(
        suffix_tokens,
        cached_prefix_len,
        embed,
        layers,
        caches,
        layer_kinds,
        paged_adapter,
    )?;

    project_last_token_logits_with_full_hidden(
        &hidden_states,
        final_norm,
        lm_head,
        embedding_weight,
        keep_last_hidden,
    )
}

fn run_paged_prefill_one_chunk(
    chunk_tokens: &[u32],
    chunk_first_position: u32,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
) -> Result<MxArray> {
    debug_assert_eq!(layers.len(), caches.len());
    debug_assert_eq!(layers.len(), layer_kinds.len());

    let chunk_len = chunk_tokens.len() as i64;
    let input_ids = MxArray::from_uint32(chunk_tokens, &[1, chunk_len])?;
    let mut hidden_states = embed.forward(&input_ids)?;

    for (layer_idx, ((layer, cache_slot), kind)) in layers
        .iter_mut()
        .zip(caches.iter_mut())
        .zip(layer_kinds.iter().copied())
        .enumerate()
    {
        hidden_states = layer.forward_paged_or_flat(
            &hidden_states,
            kind,
            paged_adapter,
            chunk_first_position,
            chunk_first_position,
            /* is_prefill */ true,
            /* mask */ None,
            Some(cache_slot),
            /* position_ids */ None,
            /* use_kernel */ true,
        )?;
        crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states)?;
    }
    Ok(hidden_states)
}

fn project_last_token_logits(
    hidden_states: &MxArray,
    final_norm: &RMSNorm,
    lm_head: &Option<Linear>,
    embedding_weight: &MxArray,
) -> Result<MxArray> {
    let seq_len = hidden_states.shape_at(1)?;
    let last_hidden = hidden_states.slice_axis(1, seq_len - 1, seq_len)?;

    let h = final_norm.forward(&last_hidden)?;
    let logits = if let Some(head) = lm_head {
        head.forward(&h)?
    } else {
        let weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        h.matmul(&weight_t)?
    };

    logits.squeeze(Some(&[0, 1]))
}

/// Project the FULL pre-norm hidden chunk through `final_norm` and the LM
/// head, returning `(last_token_logits[vocab], full_chunk_hidden[1, T, hidden])`.
///
/// Phase 4b B3.1: the paged prefill variant needs every chunk's
/// post-`final_norm` hidden so the MTP committed-history prefill seed
/// (`prefill_mtp_commit`) gets a contiguous `[1, prompt_len, hidden]`
/// tensor — mirrors `chunked_prefill_with_hidden` on the dense path.
fn project_last_token_logits_with_full_hidden(
    hidden_states: &MxArray,
    final_norm: &RMSNorm,
    lm_head: &Option<Linear>,
    embedding_weight: &MxArray,
    keep_last_hidden: Option<usize>,
) -> Result<(MxArray, MxArray)> {
    let prompt_len = hidden_states.shape_at(1)?;
    let hidden_dim = hidden_states.shape_at(2)?;
    let full_hidden = final_norm.forward(hidden_states)?;
    let last_hidden = full_hidden.slice_axis(1, prompt_len - 1, prompt_len)?;
    let logits = if let Some(head) = lm_head {
        head.forward(&last_hidden)?
    } else {
        let weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        last_hidden.matmul(&weight_t)?
    };

    let keep_start = keep_last_hidden
        .map(|keep| prompt_len.saturating_sub(keep.max(1) as i64))
        .unwrap_or(0);
    let kept_hidden = if keep_start > 0 {
        full_hidden.slice_axis(1, keep_start, prompt_len)?
    } else {
        full_hidden
    };

    // Phase 4b B4 fix #4 — the caller runs `synchronize_and_clear_cache()`
    // before `prefill_mtp_commit`, which would otherwise free the lazy graph
    // nodes backing the kept hidden. Materialise before return.
    kept_hidden.eval();
    debug_assert_eq!(kept_hidden.shape_at(0)?, 1);
    debug_assert!(kept_hidden.shape_at(1)? >= 1);
    debug_assert_eq!(kept_hidden.shape_at(2)?, hidden_dim);
    debug_assert_eq!(kept_hidden.dtype()?, crate::array::DType::BFloat16);

    let logits = logits.squeeze(Some(&[0, 1]))?;
    Ok((logits, kept_hidden))
}

/// Run one paged decode step: feed `[token_id]` through the model.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_decode_step(
    token_id: u32,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<Linear>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
) -> Result<MxArray> {
    // Capture logical position BEFORE record_tokens advances the
    // cursor.
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
            /* cached_prefix_len */ 0,
            /* is_prefill */ false,
            /* mask */ None,
            Some(cache_slot),
            /* position_ids */ None,
            /* use_kernel */ true,
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
