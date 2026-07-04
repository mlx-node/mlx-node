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
//!   the same limitation LFM2 has — pure-cache-hit dispatch is not
//!   bit-equal to a fresh prefill on hybrid models.
//!   For the **no-cache** case (cached_prefix_len = 0), pass 1 is
//!   skipped entirely and the result is exact.

use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::engine::vision::VisionMerge;
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::nn::{Embedding, RMSNorm};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

use super::decoder_layer::{DecoderLayer, Qwen3_5LayerKind};
use super::layer_cache::Qwen3_5LayerCache;
use super::quantized_linear::LinearProj;

fn bytes_to_mib(bytes: f64) -> f64 {
    bytes / (1024.0 * 1024.0)
}

/// Compute the scalar RoPE rotation offset for a paged forward step, decoupling
/// the rotation position from the physical KV slot.
///
/// Image turns compress their ~hundreds of placeholder tokens into far fewer
/// M-RoPE positions, so the running M-RoPE position trails the physical token
/// count by `cached_rope_deltas` (negative). The paged pool still writes K/V at
/// the PHYSICAL slot, but the query/key rotation must use the compressed
/// position so a warm-continuation query lines up with the image-compressed
/// keys it attends over. Text-only turns carry `cached_rope_deltas == 0`, so
/// the result is the physical position unchanged (byte-identical to the prior
/// behaviour).
///
/// `physical_position` is cast to `i32` BEFORE adding the (possibly negative)
/// delta so the arithmetic never underflows a `u32`.
pub(crate) fn paged_rope_offset(physical_position: u32, cached_rope_deltas: i32) -> i32 {
    physical_position as i32 + cached_rope_deltas
}

/// Decide the cross-turn M-RoPE delta to carry into the next paged turn.
///
/// `cached_rope_deltas` is shared model state: an image prefill bakes in a
/// compressed-position delta (negative) that only aligns with the image's
/// physically-resident K/V. The delta is meaningful for exactly ONE outcome of
/// the paged turn planner — `continued_live_prefix`, where the live image
/// sequence is being extended and its K/V is re-attended. Every other outcome
/// must drop it:
/// * a cold/fresh turn carries no cross-turn delta;
/// * a NON-live prefix-cache hit (`cached_prefix_len > 0` but
///   `continued_live_prefix == false`) can only restore pure-text prefix blocks
///   — image requests prefill with `skip_lookup` and never publish a hashable
///   text stream that collides with their expanded-placeholder blocks — so the
///   suffix must rotate at the raw physical slot (delta 0), not at the stale
///   negative delta a prior image turn left on the shared model.
///
/// Keying the reset on `cached_prefix_len == 0` is therefore too weak: it leaks
/// a stale image delta into unrelated text requests that merely share a cached
/// text prefix.
pub(crate) fn rope_delta_for_paged_turn(
    cached_rope_deltas: Option<i32>,
    continued_live_prefix: bool,
) -> Option<i32> {
    if continued_live_prefix {
        cached_rope_deltas
    } else {
        None
    }
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
    lm_head: &Option<LinearProj>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    cached_rope_deltas: i32,
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
        cached_rope_deltas,
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
    lm_head: &Option<LinearProj>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    chunk_size: i32,
    cached_rope_deltas: i32,
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
            cached_rope_deltas,
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
            /* inputs_embeds */ None,
            /* position_ids */ None,
            cached_rope_deltas,
        )?;

        if is_last_chunk {
            last_logits = Some(project_last_token_logits(
                &hidden_states,
                final_norm,
                lm_head,
                embed,
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
    lm_head: &Option<LinearProj>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    cached_rope_deltas: i32,
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
        /* inputs_embeds */ None,
        /* position_ids */ None,
        cached_rope_deltas,
    )?;

    project_last_token_logits(&hidden_states, final_norm, lm_head, embed, embedding_weight)
}

/// Single-turn image-bearing paged prefill.
///
/// Feeds the vision encoder's image-merged token embeddings
/// (`merge.inputs_embeds`) through the paged adapter and applies 3-row M-RoPE
/// over `merge.position_ids` on the full-attention layers, while GDN/linear
/// layers run with neither mask nor positions.
///
/// `expanded_tokens` are the placeholder-expanded prompt tokens (one entry per
/// embedding row). They drive `record_tokens` / the physical slot cursor only;
/// the forward itself consumes the merged embeddings, not re-embedded ids.
///
/// SINGLE-TURN ONLY: runs on a fresh prefill (`cached_prefix_len == 0`); there
/// is no GDN prefix replay and no cache-hit read-back. The forward is run in
/// one shot over the whole sequence so the GDN recurrent-state accumulation
/// and M-RoPE positions match the flat VLM prefill exactly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_vlm_prefill(
    expanded_tokens: &[u32],
    merge: &VisionMerge,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
) -> Result<MxArray> {
    if expanded_tokens.is_empty() {
        return Err(Error::from_reason(
            "run_paged_vlm_prefill called with empty prompt",
        ));
    }

    paged_adapter
        .record_tokens(expanded_tokens)
        .map_err(Error::from_reason)?;

    let hidden_states = run_paged_prefill_one_chunk(
        expanded_tokens,
        /* chunk_first_position */ 0,
        embed,
        layers,
        caches,
        layer_kinds,
        paged_adapter,
        Some(&merge.inputs_embeds),
        Some(&merge.position_ids),
        // Image prefill drives full-attention layers through the 3-row M-RoPE
        // arm (`position_ids` is `Some`), so the scalar offset is unused here.
        /* cached_rope_deltas */
        0,
    )?;

    project_last_token_logits(&hidden_states, final_norm, lm_head, embed, embedding_weight)
}

/// Paged prefill variant that ALSO returns the post-`final_norm` hidden
/// state for every prompt token, concatenated along the time axis to
/// `[1, prompt_len, hidden]`.
///
/// Mirror of `chunked_prefill_with_hidden` (dense / flat path). The
/// paged-MTP gate inside `paged_turn_sync_core_inner` consumes this so
/// `begin_mtp_decode`'s prompt-prefix seed can commit the full prompt
/// (advancing the stepper's `committed_len` to N) before the
/// first MTP cycle — without it the MTP draft attends over a
/// prompt-less context and parity vs the AR run breaks.
///
/// Caller MUST gate on `cached_prefix_len == 0` (the dense gate uses
/// the same `want_prompt_hidden` predicate). On a cache-reuse turn the
/// prefill only processes the suffix, so the captured hidden would not
/// cover the full prompt and the prompt-prefix seed cannot use it.
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
    lm_head: &Option<LinearProj>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    keep_last_hidden: Option<usize>,
    cached_rope_deltas: i32,
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
        cached_rope_deltas,
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
    lm_head: &Option<LinearProj>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    chunk_size: i32,
    keep_last_hidden: Option<usize>,
    cached_rope_deltas: i32,
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
            cached_rope_deltas,
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
            /* inputs_embeds */ None,
            /* position_ids */ None,
            cached_rope_deltas,
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
            } else if embed.is_packed_quantized() {
                embed.as_linear(&last_hidden)?
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
    lm_head: &Option<LinearProj>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    keep_last_hidden: Option<usize>,
    cached_rope_deltas: i32,
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
        /* inputs_embeds */ None,
        /* position_ids */ None,
        cached_rope_deltas,
    )?;

    project_last_token_logits_with_full_hidden(
        &hidden_states,
        final_norm,
        lm_head,
        embed,
        embedding_weight,
        keep_last_hidden,
    )
}

/// Forward one paged prefill chunk through every layer.
///
/// `inputs_embeds` is the image-merged token embeddings `[1, T, hidden]` for an
/// image-bearing prefill; when `Some` it replaces `embed.forward(chunk_tokens)`,
/// while `chunk_tokens` still drives `record_tokens` / the slot cursor upstream.
/// `position_ids` is the per-chunk M-RoPE slice `[3, 1, T]` (full-attention
/// layers only); both are `None` on the text-only path, which is byte-identical
/// to the prior behaviour.
#[allow(clippy::too_many_arguments)]
fn run_paged_prefill_one_chunk(
    chunk_tokens: &[u32],
    chunk_first_position: u32,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    inputs_embeds: Option<&MxArray>,
    position_ids: Option<&MxArray>,
    cached_rope_deltas: i32,
) -> Result<MxArray> {
    debug_assert_eq!(layers.len(), caches.len());
    debug_assert_eq!(layers.len(), layer_kinds.len());

    let mut hidden_states = match inputs_embeds {
        Some(embeds) => embeds.clone(),
        None => {
            let chunk_len = chunk_tokens.len() as i64;
            let input_ids = MxArray::from_uint32(chunk_tokens, &[1, chunk_len])?;
            embed.forward(&input_ids)?
        }
    };

    // Scalar-offset RoPE position for this chunk's queries/keys. For a text
    // suffix that warm-continues an image prefill, the rotation must trail the
    // physical slot by the negative cross-turn delta so the suffix keys stay
    // consistent with the immutable compressed-M-RoPE image keys. Text-only
    // prefill carries `cached_rope_deltas == 0` (offset == physical position),
    // and image prefill uses the M-RoPE arm so this is ignored there.
    let rope_position_offset = paged_rope_offset(chunk_first_position, cached_rope_deltas);

    // Shared per-forward-pass scratch slot for the M-RoPE cos/sin precompute
    // (see `Qwen3_5Attention::forward_paged`'s `mrope_cache` doc comment).
    // Every `FullAttentionPaged` layer in this loop shares one `position_ids`
    // array, so the first such layer computes the selected cos/sin and every
    // later one reuses it instead of recomputing the cos/sin table +
    // `take_along_axis` gather. Stays `None` (untouched) on the text-only
    // path where `position_ids` is `None`.
    let mut mrope_cache: Option<(MxArray, MxArray)> = None;

    for (layer_idx, ((layer, cache_slot), kind)) in layers
        .iter_mut()
        .zip(caches.iter_mut())
        .zip(layer_kinds.iter().copied())
        .enumerate()
    {
        // M-RoPE positions feed full-attention layers only; GDN/linear layers
        // take none (matches the flat VLM prefill policy).
        let layer_positions = match kind {
            Qwen3_5LayerKind::FullAttentionPaged { .. } => position_ids,
            Qwen3_5LayerKind::Linear => None,
        };
        hidden_states = layer.forward_paged_or_flat(
            &hidden_states,
            kind,
            paged_adapter,
            chunk_first_position,
            chunk_first_position,
            /* is_prefill */ true,
            /* mask */ None,
            Some(cache_slot),
            layer_positions,
            /* use_kernel */ true,
            rope_position_offset,
            &mut mrope_cache,
        )?;
        crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states)?;
    }
    Ok(hidden_states)
}

fn project_last_token_logits(
    hidden_states: &MxArray,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embed: &Embedding,
    embedding_weight: &MxArray,
) -> Result<MxArray> {
    let seq_len = hidden_states.shape_at(1)?;
    let last_hidden = hidden_states.slice_axis(1, seq_len - 1, seq_len)?;

    let h = final_norm.forward(&last_hidden)?;
    let logits = if let Some(head) = lm_head {
        head.forward(&h)?
    } else if embed.is_packed_quantized() {
        // Tied + packed-quantized embedding: route through the packed
        // `quantized_matmul` instead of a dense `[vocab, hidden]` transpose
        // + matmul (the `embedding_weight` fallback below reads a fully
        // pre-dequantized/on-demand-dequantized dense copy).
        embed.as_linear(&h)?
    } else {
        let weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        h.matmul(&weight_t)?
    };

    logits.squeeze(Some(&[0, 1]))
}

/// Project the FULL pre-norm hidden chunk through `final_norm` and the LM
/// head, returning `(last_token_logits[vocab], full_chunk_hidden[1, T, hidden])`.
///
/// The paged prefill variant needs every chunk's post-`final_norm` hidden so
/// the MTP committed-history prompt seed (`prompt_hidden`, consumed by
/// `begin_mtp_decode`) gets a contiguous `[1, prompt_len, hidden]` tensor —
/// mirrors `chunked_prefill_with_hidden` on the dense path.
fn project_last_token_logits_with_full_hidden(
    hidden_states: &MxArray,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embed: &Embedding,
    embedding_weight: &MxArray,
    keep_last_hidden: Option<usize>,
) -> Result<(MxArray, MxArray)> {
    let prompt_len = hidden_states.shape_at(1)?;
    let hidden_dim = hidden_states.shape_at(2)?;
    let full_hidden = final_norm.forward(hidden_states)?;
    let last_hidden = full_hidden.slice_axis(1, prompt_len - 1, prompt_len)?;
    let logits = if let Some(head) = lm_head {
        head.forward(&last_hidden)?
    } else if embed.is_packed_quantized() {
        embed.as_linear(&last_hidden)?
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

    // The caller runs `synchronize_and_clear_cache()` after prefill, before
    // `begin_mtp_decode` consumes the kept hidden as its prompt-prefix seed
    // — that sweep would otherwise free the lazy graph nodes backing the
    // kept hidden. Materialise before return.
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
    lm_head: &Option<LinearProj>,
    embedding_weight: &MxArray,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    cached_rope_deltas: i32,
) -> Result<MxArray> {
    // Capture logical position BEFORE record_tokens advances the
    // cursor.
    let first_logical_position = paged_adapter.current_token_count();
    paged_adapter
        .record_tokens(&[token_id])
        .map_err(Error::from_reason)?;

    let input_ids = MxArray::from_uint32(&[token_id], &[1, 1])?;
    let mut hidden_states = embed.forward(&input_ids)?;

    // Decode rotates the query at the physical slot plus the cross-turn M-RoPE
    // delta (0 for text turns) while K/V still writes at the physical slot.
    let rope_position_offset = paged_rope_offset(first_logical_position, cached_rope_deltas);

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
            rope_position_offset,
            &mut None,
        )?;
    }

    let h = final_norm.forward(&hidden_states)?;
    let logits = if let Some(head) = lm_head {
        head.forward(&h)?
    } else if embed.is_packed_quantized() {
        embed.as_linear(&h)?
    } else {
        let weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        h.matmul(&weight_t)?
    };
    Ok(logits)
}

/// Eager paged MTP Step-A forward: a single `[1, 1]` paged forward returning
/// both the verifier logits and the post-`final_norm` hidden.
///
/// Routes full-attention layers through the paged adapter (writing one new K/V
/// slot into the pool, attending over `read_kv_range(0, total_ctx)`) and GDN
/// layers through the flat `Qwen3_5LayerCache::Linear` slots in `caches`, the
/// same split `run_paged_decode_step` uses. The eager analogue of the deleted
/// compiled `forward_with_hidden` closure that called `forward_dense_cpp_paged`
/// + `export_last_hidden_paged`.
///
/// Returns `(logits [1, 1, vocab], hidden [1, hidden])`. The hidden is squeezed
/// on the time axis to match the eager-flat MTP `forward_with_hidden` contract
/// (`needs_squeeze = true`); the caller reshapes it back to `[1, 1, hidden]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_step_with_hidden(
    token_id: u32,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embedding_weight: &MxArray,
    embedding_weight_t: Option<&MxArray>,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    cached_rope_deltas: i32,
) -> Result<(MxArray, MxArray)> {
    let first_logical_position = paged_adapter.current_token_count();
    paged_adapter
        .record_tokens(&[token_id])
        .map_err(Error::from_reason)?;

    let input_ids = MxArray::from_uint32(&[token_id], &[1, 1])?;
    let mut hidden_states = embed.forward(&input_ids)?;

    // Same cross-turn delta as `run_paged_decode_step`: a text MTP Step-A
    // forward that warm-continues an image prefill must rotate at the
    // compressed position.
    let rope_position_offset = paged_rope_offset(first_logical_position, cached_rope_deltas);

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
            rope_position_offset,
            &mut None,
        )?;
    }

    let h3 = final_norm.forward(&hidden_states)?;
    let logits = if let Some(head) = lm_head {
        head.forward(&h3)?
    } else if embed.is_packed_quantized() {
        embed.as_linear(&h3)?
    } else {
        match embedding_weight_t {
            Some(wt) => h3.matmul(wt)?,
            None => {
                let wt = embedding_weight.transpose(Some(&[1, 0]))?;
                h3.matmul(&wt)?
            }
        }
    };
    let hidden = h3.squeeze(Some(&[1]))?;
    Ok((logits, hidden))
}

/// Eager paged MTP batched verify forward: a single `[1, K+1]` paged forward
/// returning the verifier target distribution and the post-`final_norm` hidden
/// at every verify position, recording the per-layer GDN tape for the rollback
/// replay.
///
/// The eager analogue of the deleted compiled `forward_mtp_verify_paged` FFI. The
/// `verify_ids` (`[1, K+1]` int32) are recorded into the adapter in ONE
/// `record_tokens` call (so the new K/V land at logical positions
/// `[ctx, ctx+K]`), then run through every layer: full-attention via the paged
/// adapter (with `is_prefill = true` so the internal causal mask covers all
/// K+1 query positions over the full context), GDN via the flat `Linear`
/// slots while recording a [`GdnLayerTape`] (the bit-exactness keystone the
/// rollback replay consumes).
///
/// Returns `MtpVerifyOutput::logits_only(logits [1, K+1, vocab],
/// hiddens [1, K+1, hidden])`. The `tape` is pre-sized / cleared by this
/// function to `layers.len()` (`Some` for GDN layers, `None` for full-attn).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_verify_step(
    verify_ids: &MxArray,
    embed: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut [Qwen3_5LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embedding_weight: &MxArray,
    embedding_weight_t: Option<&MxArray>,
    layer_kinds: &[Qwen3_5LayerKind],
    paged_adapter: &mut PagedKVCacheAdapter,
    tape: &mut Vec<Option<super::gated_delta_net::GdnLayerTape>>,
    cached_rope_deltas: i32,
) -> Result<super::mtp_decode::MtpVerifyOutput> {
    debug_assert_eq!(layers.len(), caches.len());
    debug_assert_eq!(layers.len(), layer_kinds.len());

    // Materialise the verify ids on host so the slot mapping records the exact
    // K+1 tokens, then feed the same array back through the embedding graph.
    let id_window = verify_ids.to_int32().map_err(|e| {
        Error::from_reason(format!(
            "run_paged_verify_step: verify_ids to_int32: {}",
            e.reason
        ))
    })?;
    let verify_len = id_window.len();
    if verify_len == 0 {
        return Err(Error::from_reason(
            "run_paged_verify_step: verify_ids must have at least one token",
        ));
    }
    let verify_u32: Vec<u32> = id_window.iter().map(|&v| v as u32).collect();

    let chunk_first_position = paged_adapter.current_token_count();
    paged_adapter
        .record_tokens(&verify_u32)
        .map_err(Error::from_reason)?;

    let input_ids = MxArray::from_uint32(&verify_u32, &[1, verify_len as i64])?;
    let mut hidden_states = embed.forward(&input_ids)?;

    // The K+1 verify ids rotate at the physical context start plus the
    // cross-turn M-RoPE delta (0 for text turns), matching the Step-A forward.
    let rope_position_offset = paged_rope_offset(chunk_first_position, cached_rope_deltas);

    let num_layers = layers.len();
    tape.clear();
    tape.resize(num_layers, None);
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
        let mut slot: Option<super::gated_delta_net::GdnLayerTape> = None;
        hidden_states = layer.forward_paged_or_flat_with_tape(
            &hidden_states,
            kind,
            paged_adapter,
            chunk_first_position,
            chunk_first_position,
            /* is_prefill */ true,
            Some(cache_slot),
            Some(&mut slot),
            rope_position_offset,
        )?;
        tape[layer_idx] = slot;
    }

    let hiddens = final_norm.forward(&hidden_states)?;
    let logits = if let Some(head) = lm_head {
        head.forward(&hiddens)?
    } else if embed.is_packed_quantized() {
        embed.as_linear(&hiddens)?
    } else {
        match embedding_weight_t {
            Some(wt) => hiddens.matmul(wt)?,
            None => {
                let wt = embedding_weight.transpose(Some(&[1, 0]))?;
                hiddens.matmul(&wt)?
            }
        }
    };
    Ok(super::mtp_decode::MtpVerifyOutput::logits_only(
        logits, hiddens,
    ))
}

#[cfg(test)]
mod rope_offset_tests {
    //! Model-free coverage for the paged scalar-offset RoPE position helper.
    //!
    //! The cross-turn image-decode fix decouples the rotation position from the
    //! physical KV slot: image prefill compresses ~hundreds of placeholder
    //! tokens into far fewer M-RoPE positions, so a warm-continuation turn must
    //! rotate its queries at `physical_slot + cached_rope_deltas` (the delta is
    //! negative) while K/V still writes at the physical slot. These tests pin
    //! that arithmetic — including the cast-before-add type-safety guard — and
    //! the text-turn identity (`delta == 0`) that keeps text decode
    //! byte-identical. They construct no model, so they run on any host.

    use super::{paged_rope_offset, rope_delta_for_paged_turn};

    #[test]
    fn live_continuation_preserves_image_delta() {
        // A live continuation re-attends the image request's physically-resident
        // compressed-position K/V, so the negative delta MUST survive to keep the
        // text suffix rotating at the compressed position.
        assert_eq!(rope_delta_for_paged_turn(Some(-726), true), Some(-726));
        // Suffix at physical slot 754 then rotates at the compressed position 28.
        let delta = rope_delta_for_paged_turn(Some(-726), true).unwrap_or(0);
        assert_eq!(paged_rope_offset(754, delta), 28);
    }

    #[test]
    fn cold_start_clears_delta() {
        // A fresh/miss turn (no reused prefix) carries no cross-turn delta.
        assert_eq!(rope_delta_for_paged_turn(Some(-726), false), None);
        assert_eq!(rope_delta_for_paged_turn(None, false), None);
    }

    #[test]
    fn non_live_prefix_cache_hit_clears_stale_image_delta() {
        // Regression: a prior image turn leaves a stale negative delta on the
        // shared model. A later TEXT request that merely HITS the cross-request
        // prefix cache (cached_prefix_len > 0) is NOT a live image continuation
        // (continued_live_prefix == false) — its restored blocks can only be the
        // pure-text prefix. The stale delta must be dropped so text rotates at
        // the raw physical slot, NOT at physical + stale_negative_delta.
        let stale_image_delta = Some(-726);
        let after_text_hit = rope_delta_for_paged_turn(stale_image_delta, false);
        assert_eq!(after_text_hit, None);
        assert_eq!(paged_rope_offset(42, after_text_hit.unwrap_or(0)), 42);
    }

    #[test]
    fn text_turn_zero_delta_is_identity() {
        // Text-only turns store delta 0 -> the rotation offset equals the
        // physical KV slot exactly, keeping text decode byte-identical.
        assert_eq!(paged_rope_offset(0, 0), 0);
        assert_eq!(paged_rope_offset(42, 0), 42);
        assert_eq!(paged_rope_offset(1_000_000, 0), 1_000_000);
    }

    #[test]
    fn image_turn_negative_delta_shifts_offset_down() {
        // An image turn compressing ~754 placeholder tokens to ~28 M-RoPE
        // positions stores delta = 28 - 754 = -726. Decode at physical slot
        // 754 must rotate at the compressed position 28, NOT 754; the next
        // physical slot (755) rotates at 29.
        let delta = -726;
        assert_eq!(paged_rope_offset(754, delta), 28);
        assert_eq!(paged_rope_offset(755, delta), 29);
    }

    #[test]
    fn offset_casts_to_i32_before_adding_negative_delta() {
        // Type-safety guard: the cast to i32 happens BEFORE the add, so a small
        // physical position with a large negative delta yields a negative i32
        // rather than wrapping a u32 subtraction. In practice the physical
        // position always exceeds |delta| on a warm continuation, but the
        // helper must not underflow if it ever did not.
        assert_eq!(paged_rope_offset(10, -726), -716);
        // Physical position equal to |delta| collapses to exactly 0.
        assert_eq!(paged_rope_offset(726, -726), 0);
    }

    #[test]
    fn resetting_delta_to_zero_restores_physical_offset() {
        // Round-trip of the stored cross-turn delta: applying a negative delta
        // shifts the offset, and clearing it back to 0 (a fresh text turn /
        // `Option::unwrap_or(0)`) restores the physical position unchanged.
        let physical = 800;
        let with_image_delta = paged_rope_offset(physical, -726);
        assert_eq!(with_image_delta, 74);
        let after_reset = paged_rope_offset(physical, 0);
        assert_eq!(after_reset, physical as i32);
    }
}
