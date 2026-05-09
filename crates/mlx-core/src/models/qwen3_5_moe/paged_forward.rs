//! Block-paged forward dispatch helpers for Qwen3.5 MoE.
//!
//! Mirrors `crate::models::qwen3_5::paged_forward` but threads through
//! the MoE `DecoderLayer` (which holds an MoE/dense MLP variant) and
//! its own `forward_paged_or_flat` method.

use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
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

fn trace_memory_mib() -> (f64, f64, f64) {
    (
        bytes_to_mib(crate::array::get_active_memory()),
        bytes_to_mib(crate::array::get_cache_memory()),
        bytes_to_mib(crate::array::get_peak_memory()),
    )
}

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

/// Public entry point for paged prefill of a (cached_prefix + suffix) pair.
///
/// Reads `MLX_PAGED_PREFILL_CHUNK_SIZE` once and forwards into the
/// chunk-size-parameterized worker. When `chunk_size > 0` AND
/// `suffix_tokens.len() > chunk_size`, the suffix is sliced into
/// `chunk_size`-token chunks; each chunk runs through every layer
/// (GDN linear-attention layers' recurrent state propagates in-place
/// across chunks; full-attention layers write K/V into the paged
/// pool). The hidden state is materialized + the MLX cache cleared
/// between chunks so the lazy graph + caching allocator do not
/// accumulate the entire suffix's intermediates simultaneously.
/// `final_norm` + `lm_head` only run on the LAST chunk (vocab
/// projection on intermediate chunks is throwaway work — matches
/// vLLM's `is_prefill_chunk` skip).
///
/// Memory peak is then bounded by `chunk_size * hidden_dim` instead
/// of `suffix_len * hidden_dim`. The 28K-token `Qwen3.6-35b-a3b`
/// cold-prefill scenario peaks at 117 GB / 39 GB swap today; with
/// `MLX_PAGED_PREFILL_CHUNK_SIZE=1024` the per-chunk working set
/// drops to ~1024 * hidden_dim, dramatically reducing peak memory.
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
/// `chunk_size <= 0` OR `suffix_tokens.len() <= chunk_size` takes the
/// legacy single-shot path (`run_paged_prefill_single_shot`). Anything
/// else loops over `suffix_tokens.chunks(chunk_size)`.
///
/// Critical correctness notes:
///
/// 1. **GDN pre-pass runs ONCE, before any chunking.** The GDN
///    linear-attention layers consume the cached prefix in one shot
///    (this is the existing approximation — orthogonal to suffix
///    chunking). Live continuation callers can pass
///    `gdn_prefix_already_primed = true` when `caches` already contain
///    the linear-attention state for `cached_prefix_len`, avoiding this
///    replay.
///
/// 2. **GDN state propagates in-place across chunks.** When the layer
///    loop calls `layer.forward_paged_or_flat` with a Linear-kind
///    layer, it mutates the `Qwen3_5LayerCache::Linear(ArraysCache)`
///    slot. After chunk N's layer loop completes, chunk N+1's layer
///    loop sees the same cache slot with state advanced by chunk N's
///    tokens. We do NOT reset between chunks — that's the entire
///    point of vLLM-aligned chunking.
///
/// 3. **MoE expert routing** is per-token within a layer. Chunking
///    gives smaller batches; each chunk's tokens are routed
///    independently. No global state spans chunks.
///
/// 4. **Position arguments**: For chunk N starting at cumulative
///    position P (= `cached_prefix_len` + sum of prior chunk lens):
///    - `first_logical_position = P` (where Q starts).
///    - `cached_prefix_len` argument to the layer = P (K/V[0..P] is
///      what's cumulatively cached at this point).
///    - `record_tokens(chunk)` happens BEFORE the layer loop, so the
///      adapter's cursor is `P + chunk.len()` after; the layer's K/V
///      write goes to slots `[P, P+chunk.len())`.
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
) -> Result<MxArray> {
    if suffix_tokens.is_empty() {
        return Err(Error::from_reason(
            "MoE run_paged_prefill_chunk called with empty suffix",
        ));
    }

    let trace_enabled = inference_trace_enabled();

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

    let chunk_size_usize = chunk_size as usize;

    // GDN pre-pass over the cached prefix runs ONCE, before any suffix
    // chunking. The GDN linear-attention layers consume the prefix in
    // one shot (existing approximation; orthogonal to chunking).
    if cached_prefix_len > 0 && !gdn_prefix_already_primed {
        let gdn_trace_start = trace_enabled.then(Instant::now);
        let prefix = &full_tokens[..(cached_prefix_len as usize)];
        run_gdn_only_prefill(prefix, embed, layers, caches)?;
        if let Some(start) = gdn_trace_start {
            let (active_mib, cache_mib, peak_mib) = trace_memory_mib();
            write_inference_trace(format_args!(
                "[MLX_TRACE] qwen3.5-moe paged_prefill_gdn_prefix_done \
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
    let mut chunk_start_position: u32 = cached_prefix_len;

    for (chunk_idx, chunk) in suffix_tokens.chunks(chunk_size_usize).enumerate() {
        let is_last_chunk = chunk_idx + 1 == total_chunks;
        let chunk_trace_start = trace_enabled.then(Instant::now);

        // 1. Advance cursor + grow blocks for this chunk. Must happen
        //    BEFORE calling the per-chunk helper so `update_keys_values`
        //    inside the layer loop aligns at
        //    `[chunk_start_position, chunk_start_position+chunk.len())`.
        paged_adapter
            .record_tokens(chunk)
            .map_err(Error::from_reason)?;

        // 2. Run the per-chunk forward (embed → layer loop). For chunk N
        //    starting at cumulative position `chunk_start_position`:
        //    * Q has chunk.len() positions starting at chunk_start_position
        //    * K/V is cumulative [0..chunk_start_position+chunk.len())
        //    * `forward_paged_or_flat` for FullAttentionPaged layers
        //      builds `create_causal_mask(num_tokens=chunk.len(),
        //      offset=chunk_start_position)` to align Q within K.
        //    * Linear (GDN) layers' recurrent state mutates in-place
        //      across chunks via `Qwen3_5LayerCache::Linear(ArraysCache)`.
        let hidden = run_paged_prefill_one_chunk_moe(
            chunk,
            chunk_start_position,
            embed,
            layers,
            caches,
            layer_kinds,
            paged_adapter,
        )?;

        if is_last_chunk {
            // Last chunk: project final_norm + lm_head and extract
            // last-token logits.
            last_logits = Some(project_last_token_logits_moe(
                &hidden,
                final_norm,
                lm_head,
                embedding_weight,
            )?);
            if let Some(start) = chunk_trace_start {
                let chunk_elapsed_ms = elapsed_ms(start);
                let (active_mib, cache_mib, peak_mib) = trace_memory_mib();
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-moe paged_prefill_chunk_final_graph_built \
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
            // Force materialize the residual stream so MLX can release
            // the upstream graph nodes (embedding + every prior layer's
            // attention/MLP intermediates) for this chunk before we
            // start building the next chunk's graph. Without this the
            // lazy DAG accumulates across chunks and defeats the entire
            // memory-bounding purpose. Skipping `final_norm`/`lm_head`
            // on intermediate chunks matches vLLM's `is_prefill_chunk`
            // skip — those projections would be discarded anyway.
            hidden.eval();
            crate::array::synchronize_and_clear_cache();
            if let Some(start) = chunk_trace_start {
                let chunk_elapsed_ms = elapsed_ms(start);
                let (active_mib, cache_mib, peak_mib) = trace_memory_mib();
                let chunk_tok_s = if chunk_elapsed_ms > 0.0 {
                    chunk.len() as f64 / (chunk_elapsed_ms / 1000.0)
                } else {
                    0.0
                };
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-moe paged_prefill_chunk_done \
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
            "MoE chunked prefill produced no last chunk (unreachable for non-empty suffix)",
        )
    })
}

/// Single-shot prefill: feed the entire suffix through every layer in
/// one forward pass. Identical to the pre-chunking implementation.
/// Used both by the legacy code path (chunk_size <= 0) and the
/// chunked driver's "small enough to skip chunking" fast path.
///
/// The empty-suffix check is performed by the caller
/// (`run_paged_prefill_chunk_with_size`); this helper trusts its input.
///
/// Thin wrapper over `run_paged_prefill_one_chunk_moe` +
/// `project_last_token_logits_moe`. Kept as a named helper because
/// callers (and the chunked driver's fast-path branch) reference it
/// by name and the GDN pre-pass / `record_tokens` ordering matches
/// the single-shot semantics we want to preserve byte-for-byte.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_paged_prefill_single_shot(
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
) -> Result<MxArray> {
    paged_adapter
        .record_tokens(suffix_tokens)
        .map_err(Error::from_reason)?;

    if cached_prefix_len > 0 && !gdn_prefix_already_primed {
        let prefix = &full_tokens[..(cached_prefix_len as usize)];
        run_gdn_only_prefill(prefix, embed, layers, caches)?;
    }

    let hidden_states = run_paged_prefill_one_chunk_moe(
        suffix_tokens,
        cached_prefix_len,
        embed,
        layers,
        caches,
        layer_kinds,
        paged_adapter,
    )?;
    project_last_token_logits_moe(&hidden_states, final_norm, lm_head, embedding_weight)
}

/// Run a single prefill chunk through `embed → layer loop`. Returns
/// the post-last-layer residual stream (NOT logits — caller decides
/// whether to project to vocab).
///
/// This is the per-chunk inner shared between
/// `run_paged_prefill_single_shot` and the chunked driver. It must
/// NOT touch `final_norm` / `lm_head` so the chunked path can skip
/// those on intermediate chunks (matches vLLM's `is_prefill_chunk`
/// skip — non-final-chunk vocab projections would be discarded
/// anyway).
///
/// Caller contract:
///
/// * `paged_adapter.record_tokens(chunk_tokens)` MUST have been
///   called BEFORE this function so `update_keys_values` inside the
///   FullAttention layer aligns at
///   `[chunk_first_position, chunk_first_position+chunk.len())`.
/// * For the cached-prefix GDN pre-pass (when
///   `chunk_first_position > 0` on the first chunk), the caller is
///   responsible for running `run_gdn_only_prefill` on the prefix
///   tokens BEFORE the first chunk. This helper only handles
///   per-chunk forward — no cached-prefix replay.
/// * `cached_prefix_len` (the layer's K/V coverage at chunk start)
///   equals `chunk_first_position`: every token already in the paged
///   pool — be it from a prior cache hit OR from a prior chunk
///   written by an earlier iteration of the chunked driver — lives
///   at logical positions `[0, chunk_first_position)`.
fn run_paged_prefill_one_chunk_moe(
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
    let mut hidden = embed.forward(&input_ids)?;

    // Layer loop. Safe-by-construction via `iter_mut().zip(...)` —
    // each iteration takes disjoint `&mut DecoderLayer` and `&mut
    // Qwen3_5LayerCache` references, with `kind` consumed by-value
    // (`Qwen3_5LayerKind: Copy`).
    for (layer_idx, ((layer, cache_slot), kind)) in layers
        .iter_mut()
        .zip(caches.iter_mut())
        .zip(layer_kinds.iter().copied())
        .enumerate()
    {
        hidden = layer.forward_paged_or_flat(
            &hidden,
            kind,
            paged_adapter,
            chunk_first_position,
            chunk_first_position,
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
        crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden)?;
    }
    Ok(hidden)
}

/// Project the per-token residual stream through `final_norm` + the
/// LM head and slice the last position's logits down to `[vocab]`.
/// Shared between the single-shot path and the chunked path's
/// final-chunk return path. Hidden state shape on entry: `[1,
/// chunk_len, hidden]`.
///
/// `lm_head = None` corresponds to the tied-word-embeddings case;
/// we matmul against `embedding_weight.T` instead of going through a
/// dedicated `LinearProj`.
fn project_last_token_logits_moe(
    hidden_states: &MxArray,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embedding_weight: &MxArray,
) -> Result<MxArray> {
    let h = final_norm.forward(hidden_states)?;
    let logits = if let Some(head) = lm_head {
        head.forward(&h)?
    } else {
        let weight_t = embedding_weight.transpose(Some(&[1, 0]))?;
        h.matmul(&weight_t)?
    };
    // Slice last token: logits shape [1, chunk_len, vocab] -> [vocab].
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

#[cfg(test)]
mod tests {
    //! Inline tests for the chunked paged-prefill driver in this module.
    //!
    //! These mirror the Phase B (Qwen3) parity tests in
    //! `crates/mlx-core/src/models/qwen3/model.rs`: a synthetic
    //! `Qwen3_5MoeConfig` is built with a tiny vocab + a tiny number of
    //! layers spanning both `Linear` (GDN) and `FullAttentionPaged`
    //! kinds, weights are cast to BFloat16 to match the paged pool's
    //! K/V dtype, and the chunked path is compared against the
    //! single-shot path on a same-prompt fresh-adapter run.
    //!
    //! Each parity test pins MLX's PRNG via `mlx_sys::mlx_seed` before
    //! constructing the model so weight init is reproducible across
    //! `cargo test` invocations. Without that, weight init varies per
    //! run and the chunked-vs-single-shot drift magnitude wanders from
    //! `~5e-2` to `~2e-1` with occasional argmax flips, which would
    //! turn the parity assertion into a flaky check. With a fixed
    //! seed both `max_abs_diff` and the argmax index are bit-stable.
    //!
    //! **All tests are `#[ignore]`-marked.** They require a Metal
    //! GPU; on machines without Metal, `Qwen35MoeInner::new` can
    //! throw a foreign C++ exception BEFORE Rust receives an `Err`,
    //! aborting the test process with `fatal runtime error: Rust
    //! cannot catch foreign exceptions, aborting`. String-matching
    //! the no-Metal error message is therefore unsafe in CI/sandboxes,
    //! so we gate the tests behind `--ignored` instead. Run with:
    //!
    //! ```bash
    //! cargo test -p mlx-core --lib qwen3_5_moe -- --ignored
    //! ```
    //!
    //! The default `cargo test` run does NOT execute these. They also
    //! skip when the env var `MLX_PAGED_PREFILL_CHUNK_SIZE` is set to
    //! a non-zero value for the default-path test (process-global
    //! OnceLock pollution).

    use super::*;
    use crate::array::DType;
    use crate::models::qwen3_5::decoder_layer::compute_layer_kinds;
    use crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig;
    use crate::models::qwen3_5_moe::model::Qwen35MoeInner;

    /// Build a tiny MoE config that exercises both Linear (GDN) and
    /// FullAttentionPaged layer kinds. With `num_layers=8` and
    /// `full_attention_interval=4`, layers 3 and 7 are full-attention
    /// (the default Qwen3.5 placement); the rest are GDN. This gives
    /// us 2 full-attention layers and 6 GDN layers — enough to verify
    /// state propagation across chunks for both kinds.
    fn moe_paged_tiny_config() -> Qwen3_5MoeConfig {
        // `head_dim = 32` chosen to satisfy the paged Metal kernel's
        // valid-head-size whitelist `[32, 64, 80, 96, 112, 120, 128,
        // 192, 256, 512]`. `hidden_size = num_heads * head_dim = 128`
        // keeps q_proj output dimension consistent. Linear (GDN) head
        // dims also bumped to 32 so all attention paths share a
        // dtype-compatible layout.
        Qwen3_5MoeConfig {
            vocab_size: 128,
            hidden_size: 128,
            num_layers: 8,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 32,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 256,
            pad_token_id: 0,
            eos_token_id: 0,
            bos_token_id: 0,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 32,
            linear_value_head_dim: 32,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            partial_rotary_factor: 0.25,
            rope_theta: 100_000.0,
            num_experts: 4,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            shared_expert_intermediate_size: None,
            moe_intermediate_size: None,
            norm_topk_prob: true,
            mlp_only_layers: None,
            paged_cache_memory_mb: Some(256),
            paged_block_size: Some(16),
            use_block_paged_cache: Some(true),
        }
    }

    /// Recursively cast every weight in the freshly-constructed
    /// `Qwen35MoeInner` to BFloat16 so the K/V the full-attention layers
    /// produce matches the paged pool's `MetalDtype::BFloat16`. Without
    /// this `update_keys_values` rejects the F32 K/V the random-init
    /// weights would otherwise produce.
    ///
    /// We walk every named-MxArray weight surface reachable through
    /// public setters: embedding, final norm, layer norms, the
    /// full-attention `Qwen3_5Attention` projections (q/k/v/o + qn/kn),
    /// the GDN `GatedDeltaNet` projections (in_proj_qkvz, in_proj_ba,
    /// conv1d, norm, out_proj, dt_bias), the dense MLP (gate/up/down)
    /// for non-MoE layers, and the SparseMoeBlock surfaces (gate,
    /// shared_expert_gate, shared expert gate/up/down, switch_mlp
    /// gate/up/down) for MoE layers.
    ///
    /// **NOT cast** (deliberately): GDN `a_log` — a documented
    /// pattern, mlx-lm's `cast_predicate` excludes `A_log` from dtype
    /// casting (it stays f32 and is cast on-the-fly inside
    /// `compute_g`). Casting it would diverge from mlx-lm semantics.
    fn cast_moe_inner_weights_bf16(inner: &mut Qwen35MoeInner) {
        use crate::models::qwen3_5_moe::decoder_layer::{AttentionType, MLPType};
        use crate::models::qwen3_5_moe::quantized_linear::MLPVariant;
        let cast = |a: &MxArray| -> MxArray { a.astype(DType::BFloat16).expect("astype BFloat16") };

        // Embedding.
        let w = inner.embedding.get_weight();
        inner.embedding.set_weight(&cast(&w)).expect("set embed");

        // Final norm.
        let w = inner.final_norm.get_weight();
        inner
            .final_norm
            .set_weight(&cast(&w))
            .expect("set final_norm");

        // LM head: tied to embedding, no separate weights to cast.

        // Per-layer.
        for layer in inner.layers.iter_mut() {
            // Layer norms.
            let w = layer.get_input_layernorm_weight();
            layer
                .set_input_layernorm_weight(&cast(&w))
                .expect("set in ln");
            let w = layer.get_post_attention_layernorm_weight();
            layer
                .set_post_attention_layernorm_weight(&cast(&w))
                .expect("set post ln");

            // Attention.
            match &mut layer.attn {
                AttentionType::Full(attn) => {
                    let w = attn.get_q_proj_weight();
                    attn.set_q_proj_weight(&cast(&w)).expect("set q_proj");
                    let w = attn.get_k_proj_weight();
                    attn.set_k_proj_weight(&cast(&w)).expect("set k_proj");
                    let w = attn.get_v_proj_weight();
                    attn.set_v_proj_weight(&cast(&w)).expect("set v_proj");
                    let w = attn.get_o_proj_weight();
                    attn.set_o_proj_weight(&cast(&w)).expect("set o_proj");
                    let w = attn.get_q_norm_weight();
                    attn.set_q_norm_weight(&cast(&w)).expect("set q_norm");
                    let w = attn.get_k_norm_weight();
                    attn.set_k_norm_weight(&cast(&w)).expect("set k_norm");
                }
                AttentionType::Linear(gdn) => {
                    // dt_bias must be cast FIRST: `set_norm_weight` derives
                    // its target dtype from `self.dt_bias.dtype()`, so a
                    // late dt_bias cast leaves norm.weight at the wrong
                    // dtype.
                    let w = gdn.get_dt_bias();
                    gdn.set_dt_bias(&cast(&w));
                    let w = gdn.get_in_proj_qkvz_weight();
                    gdn.set_in_proj_qkvz_weight(&cast(&w))
                        .expect("set in_proj_qkvz");
                    let w = gdn.get_in_proj_ba_weight();
                    gdn.set_in_proj_ba_weight(&cast(&w))
                        .expect("set in_proj_ba");
                    let w = gdn.get_conv1d_weight();
                    gdn.set_conv1d_weight(&cast(&w)).expect("set conv1d");
                    let w = gdn.get_norm_weight();
                    gdn.set_norm_weight(&cast(&w)).expect("set norm");
                    let w = gdn.get_out_proj_weight();
                    gdn.set_out_proj_weight(&cast(&w)).expect("set out_proj");
                    // a_log stays f32 (mlx-lm cast_predicate excludes it).
                }
            }

            // MLP / MoE.
            match &mut layer.mlp {
                MLPType::Dense(mlp_variant) => {
                    if let MLPVariant::Standard(mlp) = mlp_variant {
                        let w = mlp.get_gate_proj_weight();
                        mlp.set_gate_proj_weight(&cast(&w)).expect("set gate_proj");
                        let w = mlp.get_up_proj_weight();
                        mlp.set_up_proj_weight(&cast(&w)).expect("set up_proj");
                        let w = mlp.get_down_proj_weight();
                        mlp.set_down_proj_weight(&cast(&w)).expect("set down_proj");
                    }
                    // MLPVariant::Quantized: weights are pre-quantized,
                    // not directly cast-able. Synthetic configs build
                    // Standard variants only.
                }
                MLPType::MoE(moe) => {
                    // Router gate.
                    let w = moe.get_gate_weight();
                    moe.set_gate_weight(&cast(&w)).expect("set moe gate");
                    // Shared expert.
                    let w = moe.get_shared_expert_gate_proj_weight();
                    moe.set_shared_expert_gate_proj_weight(&cast(&w))
                        .expect("set shared_expert gate_proj");
                    let w = moe.get_shared_expert_up_proj_weight();
                    moe.set_shared_expert_up_proj_weight(&cast(&w))
                        .expect("set shared_expert up_proj");
                    let w = moe.get_shared_expert_down_proj_weight();
                    moe.set_shared_expert_down_proj_weight(&cast(&w))
                        .expect("set shared_expert down_proj");
                    let w = moe.get_shared_expert_gate_weight();
                    moe.set_shared_expert_gate_weight(&cast(&w))
                        .expect("set shared_expert gate");
                    // Switch (per-expert) projections.
                    let w = moe.get_switch_mlp().get_gate_proj_weight();
                    moe.set_switch_mlp_gate_proj_weight(&cast(&w));
                    let w = moe.get_switch_mlp().get_up_proj_weight();
                    moe.set_switch_mlp_up_proj_weight(&cast(&w));
                    let w = moe.get_switch_mlp().get_down_proj_weight();
                    moe.set_switch_mlp_down_proj_weight(&cast(&w));
                }
            }
        }
    }

    /// Read the full contents of a 1-D `[vocab]` `MxArray` to a host
    /// `Vec<f32>`. Goes via `astype(F32)` + per-element `item_at_float32`
    /// so it works on bf16 logits as well.
    fn logits_to_f32_vec(logits: &MxArray) -> Vec<f32> {
        let f32_arr = logits.astype(DType::Float32).expect("astype f32");
        f32_arr.eval();
        let n = f32_arr.shape_at(0).expect("shape_at(0)") as usize;
        (0..n)
            .map(|i| f32_arr.item_at_float32(i).expect("item_at_float32"))
            .collect()
    }

    /// Run the prefill against a freshly-reset adapter via the public
    /// `run_paged_prefill_chunk_with_size` helper. Encapsulates the
    /// boilerplate (init caches, reset adapter, allocate suffix) so
    /// each parity test stays focused on the comparison.
    ///
    /// Returns `Ok(Some(logits_vec))` on success, `Ok(None)` if the
    /// run hit a "no Metal device" / "Metal GPU not available" error
    /// (in which case the caller should skip the test), and `Err`
    /// for any other failure.
    fn run_one(
        inner: &mut Qwen35MoeInner,
        prompt: &[u32],
        chunk_size: i32,
    ) -> Result<Option<Vec<f32>>> {
        // Init caches if not yet initialized.
        if inner.caches.is_none() {
            inner.init_caches_sync()?;
        } else {
            inner.reset_caches_sync()?;
            inner.init_caches_sync()?;
        }

        let layer_kinds = compute_layer_kinds(inner.config.num_layers as usize, |i| {
            inner.config.is_linear_layer(i)
        });
        let num_layers = inner.layers.len();
        let _ = num_layers;

        // Reset adapter and allocate suffix blocks.
        {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            adapter.reset_for_new_request(0).expect("reset");
            let prefix = adapter
                .find_cached_prefix(prompt, &[], 0, false)
                .expect("find_cached_prefix");
            assert_eq!(prefix.cached_token_count, 0);
            adapter
                .allocate_suffix_blocks(prompt.len() as u32)
                .expect("allocate_suffix_blocks");
        }

        let logits = {
            let embed = inner.embedding.clone();
            let embedding_weight = embed.get_weight();
            let caches_ref = inner.caches.as_mut().expect("caches");
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            match super::run_paged_prefill_chunk_with_size(
                prompt,
                prompt,
                0,
                false,
                &embed,
                &mut inner.layers,
                caches_ref,
                &inner.final_norm,
                &inner.lm_head,
                &embedding_weight,
                &layer_kinds,
                adapter,
                chunk_size,
            ) {
                Ok(l) => l,
                Err(e) => {
                    let msg = e.reason.to_string();
                    if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                        return Ok(None);
                    }
                    return Err(e);
                }
            }
        };

        let v = logits_to_f32_vec(&logits);

        // Cleanup. We deliberately DO NOT call
        // `register_full_blocks_for_reuse` so a subsequent run on the
        // same prompt sees `cached_token_count = 0` from
        // `find_cached_prefix`. This keeps single-shot vs chunked
        // comparisons apples-to-apples (same fresh-cold-cache start).
        {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            adapter.release_request().expect("release_request");
        }
        Ok(Some(v))
    }

    /// **Parity test (multi-chunk)**: 96-token prompt with chunk_size=16
    /// (6 chunks) must produce the same final-token logits as the
    /// single-shot path (chunk_size=0). This is the primary correctness
    /// guarantee: if any chunk-boundary bug exists in either the
    /// full-attention K/V write path OR the GDN linear-attention
    /// recurrent-state propagation, this test fails. Asserts via
    /// `assert_logit_parity_relaxed` (tight `max_abs_diff <= 0.25` +
    /// argmax-must-match — see helper rustdoc for tolerance rationale).
    ///
    /// `mlx_sys::mlx_seed(0xC0DEC0DE)` pins MLX's random init so the
    /// chunked-vs-single-shot drift is reproducible across runs;
    /// observed `max_abs_diff = 0.0693`, argmax stable at idx=69.
    /// Requires Metal GPU; run with `--ignored`.
    /// `Qwen35MoeInner::new` can throw a foreign C++ exception on
    /// machines without Metal, which aborts the test process before
    /// Rust can catch the failure.
    #[test]
    #[ignore = "requires Metal GPU; run with --ignored"]
    fn test_chunked_prefill_qwen3_5_moe_matches_single_shot_logits() {
        unsafe {
            mlx_sys::mlx_seed(0xC0DEC0DE);
        }
        let cfg = moe_paged_tiny_config();
        let mut inner = match Qwen35MoeInner::new(cfg.clone()) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") || msg.contains("No Metal device") {
                    eprintln!(
                        "skipping test_chunked_prefill_qwen3_5_moe_matches_single_shot_logits \
                         (no Metal): {msg}"
                    );
                    return;
                }
                panic!("unexpected Qwen35MoeInner::new failure: {msg}");
            }
        };
        cast_moe_inner_weights_bf16(&mut inner);

        // 96 tokens — vocab_size=128, modulo to stay in range.
        let prompt: Vec<u32> = (0u32..96).map(|i| (i * 7 + 3) % 128).collect();

        // ---- Run 1: single-shot ----
        let single_vec = match run_one(&mut inner, &prompt, /* chunk_size */ 0) {
            Ok(Some(v)) => v,
            Ok(None) => {
                eprintln!(
                    "skipping test_chunked_prefill_qwen3_5_moe_matches_single_shot_logits: \
                     no Metal device available"
                );
                return;
            }
            Err(e) => panic!("unexpected single-shot prefill failure: {}", e.reason),
        };
        assert_eq!(single_vec.len(), cfg.vocab_size as usize);
        for (i, v) in single_vec.iter().enumerate() {
            assert!(v.is_finite(), "single-shot logits[{i}] not finite: {v}");
        }

        // ---- Run 2: chunked, chunk_size = 16 (6 chunks) ----
        let chunked_vec = match run_one(&mut inner, &prompt, /* chunk_size */ 16) {
            Ok(Some(v)) => v,
            Ok(None) => {
                eprintln!(
                    "skipping test_chunked_prefill_qwen3_5_moe_matches_single_shot_logits \
                     (Metal failure on second run)"
                );
                return;
            }
            Err(e) => panic!("unexpected chunked prefill failure: {}", e.reason),
        };
        assert_eq!(chunked_vec.len(), cfg.vocab_size as usize);

        assert_logit_parity_relaxed(&single_vec, &chunked_vec, "MoE chunked-vs-single-shot");
    }

    /// **Uneven-tail parity test**: 97-token prompt with chunk_size=16
    /// produces 6 full chunks of 16 + 1 trailing chunk of 1 token. This
    /// is the worst case for off-by-one bugs at chunk boundaries — the
    /// trailing 1-token chunk's `chunk_start_position = 96` is not
    /// aligned to a block boundary, and the explicit causal mask in
    /// the full-attention path must be built with `num_tokens=1,
    /// offset=96`. Compared against single-shot for the same 97-token
    /// prompt. Asserts via `assert_logit_parity_relaxed` (tight
    /// `max_abs_diff <= 0.25` + argmax-must-match — see helper rustdoc).
    ///
    /// `mlx_sys::mlx_seed(0xC0DEC0DE)` pins MLX's random init so the
    /// chunked-vs-single-shot drift is reproducible across runs;
    /// observed `max_abs_diff = 0.1199`, argmax stable at idx=80.
    /// Requires Metal GPU; run with `--ignored`.
    /// `Qwen35MoeInner::new` can throw a foreign C++ exception on
    /// machines without Metal, which aborts the test process before
    /// Rust can catch the failure.
    #[test]
    #[ignore = "requires Metal GPU; run with --ignored"]
    fn test_chunked_prefill_qwen3_5_moe_uneven_tail() {
        unsafe {
            mlx_sys::mlx_seed(0xC0DEC0DE);
        }
        let cfg = moe_paged_tiny_config();
        let mut inner = match Qwen35MoeInner::new(cfg.clone()) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") || msg.contains("No Metal device") {
                    eprintln!(
                        "skipping test_chunked_prefill_qwen3_5_moe_uneven_tail (no Metal): {msg}"
                    );
                    return;
                }
                panic!("unexpected Qwen35MoeInner::new failure: {msg}");
            }
        };
        cast_moe_inner_weights_bf16(&mut inner);

        // 97 tokens — 6*16 full chunks + 1 leftover. vocab_size=128 so
        // the modulo stays in range.
        let prompt: Vec<u32> = (0u32..97).map(|i| (i * 13 + 7) % 128).collect();

        let single_vec = match run_one(&mut inner, &prompt, /* chunk_size */ 0) {
            Ok(Some(v)) => v,
            Ok(None) => {
                eprintln!("skipping test_chunked_prefill_qwen3_5_moe_uneven_tail: no Metal");
                return;
            }
            Err(e) => panic!("unexpected single-shot prefill failure: {}", e.reason),
        };
        assert_eq!(single_vec.len(), cfg.vocab_size as usize);

        let chunked_vec = match run_one(&mut inner, &prompt, /* chunk_size */ 16) {
            Ok(Some(v)) => v,
            Ok(None) => {
                eprintln!(
                    "skipping test_chunked_prefill_qwen3_5_moe_uneven_tail (Metal failure on second run)"
                );
                return;
            }
            Err(e) => panic!("unexpected chunked prefill failure: {}", e.reason),
        };
        assert_eq!(chunked_vec.len(), cfg.vocab_size as usize);

        assert_logit_parity_relaxed(
            &single_vec,
            &chunked_vec,
            "MoE uneven-tail chunked-vs-single-shot",
        );
    }

    /// Real parity assertion for MoE chunked-vs-single-shot.
    ///
    /// **Determinism gate**: callers must seed `mlx_sys::mlx_seed` to a
    /// fixed value at the top of the test before constructing
    /// `Qwen35MoeInner`. With a fixed seed, MLX's random init produces
    /// byte-equal weights across `cargo test` invocations, and the
    /// downstream chunked-vs-single-shot logit drift is reproducible
    /// (verified bit-stable across 5 consecutive runs with seed
    /// `0xC0DEC0DE`). Without a seed, weight init varies per run and
    /// the drift magnitude can range from `~5e-2` to `~2e-1` with
    /// occasional argmax flips, which is what made the original
    /// finiteness-only assertion structurally weak — it would also
    /// pass for a deterministic-but-wrong chunking path.
    ///
    /// **Tolerance choice**: bf16 paged K/V (the pool's hard
    /// requirement) + GDN linear-attention's recurrent state mutating
    /// in chunk-sized batches + MoE softmax+top-k under bf16 noise add
    /// up to `~1e-1` of element-wise drift between chunked and
    /// single-shot at this synthetic config size. The dense Qwen3
    /// `5e-3` budget would falsely flag this as a regression, so we
    /// allow `max_abs_diff <= 0.25` — comfortably above the observed
    /// `0.07` (matches test) / `0.12` (uneven-tail test) drift but
    /// far below the `O(1)` divergence any real chunking bug would
    /// produce (e.g. GDN state reset between chunks, K/V write to
    /// wrong slot, position off-by-one — all of those amplify through
    /// 8 layers + softmax to either NaN/Inf or a complete logit
    /// reshuffle). Argmax must match — a "deterministic finite
    /// wrong" chunked path is overwhelmingly likely to flip the
    /// argmax under any real bug, since the small bf16 noise we
    /// tolerate is weight-driven and can't move argmax under fixed
    /// weights.
    ///
    /// **Why not byte-equal**: The dense Qwen3 Phase B inline parity
    /// tests (in `crates/mlx-core/src/models/qwen3/model.rs`) hit
    /// byte-equal `max_abs_diff = 0` because that model has 2 layers,
    /// no GDN, no MoE — bf16 fma orderings happen to align across
    /// chunk boundaries at that scale. Adding GDN's recurrent fma
    /// chain and MoE's per-token softmax routing breaks bit-equality
    /// without breaking semantic equality.
    ///
    /// Numerical parity on **real Qwen3.5 MoE checkpoint weights** is
    /// gated separately by the integration test in
    /// `crates/mlx-core/tests/qwen3_5_moe_chunked_prefill.rs` (which
    /// asserts byte-identical greedy-decoded token streams under a
    /// real prompt). That test runs only when a checkpoint is on
    /// disk; this synthetic test runs always and gates against the
    /// regressions any real chunking bug would produce.
    fn assert_logit_parity_relaxed(single_vec: &[f32], chunked_vec: &[f32], label: &str) {
        assert_eq!(
            single_vec.len(),
            chunked_vec.len(),
            "{label}: vector length mismatch"
        );

        // Finiteness on both paths. NaN/Inf would short-circuit the
        // numeric checks below, so assert separately first.
        for (i, v) in single_vec.iter().enumerate() {
            assert!(
                v.is_finite(),
                "{label}: single-shot logits[{i}] not finite: {v}"
            );
        }
        for (i, v) in chunked_vec.iter().enumerate() {
            assert!(
                v.is_finite(),
                "{label}: chunked logits[{i}] not finite: {v}"
            );
        }

        let argmax = |v: &[f32]| -> (usize, f32) {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                })
        };
        let (idx_single, val_single) = argmax(single_vec);
        let (idx_chunked, val_chunked) = argmax(chunked_vec);
        let mut max_abs_diff = 0.0f32;
        let mut argmax_diff_idx = 0usize;
        for (i, (a, b)) in single_vec.iter().zip(chunked_vec.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
                argmax_diff_idx = i;
            }
        }

        eprintln!(
            "{label}: max_abs_diff={max_abs_diff} (at idx={argmax_diff_idx}), single-shot \
             argmax=(idx={idx_single}, val={val_single}), chunked argmax=(idx={idx_chunked}, \
             val={val_chunked}) over {} elements",
            single_vec.len()
        );

        // Hard parity bounds — the real assertions a chunking bug must
        // surface against. See the rustdoc above for tolerance rationale.
        const ATOL: f32 = 0.25;
        assert!(
            max_abs_diff <= ATOL,
            "{label}: max_abs_diff {max_abs_diff} exceeds tolerance {ATOL} \
             (at idx={argmax_diff_idx}: single={}, chunked={}). A real chunking \
             regression — GDN state reset between chunks, position off-by-one in \
             the layer loop, mask misalignment, or paged-pool K/V write to wrong \
             slots — would surface here. Re-seed only after investigating.",
            single_vec[argmax_diff_idx],
            chunked_vec[argmax_diff_idx]
        );
        assert_eq!(
            idx_single, idx_chunked,
            "{label}: argmax flipped (single={idx_single} val={val_single}, \
             chunked={idx_chunked} val={val_chunked}). Under a fixed mlx_seed, \
             argmax flipping is the canonical signature of a chunking bug — bf16 \
             noise alone cannot move argmax under fixed weights."
        );
    }

    /// **Default-path guard**: the public entry point
    /// `run_paged_prefill_chunk` (which reads `MLX_PAGED_PREFILL_CHUNK_SIZE`
    /// once via OnceLock) must be byte-equivalent to the explicit
    /// single-shot path (`run_paged_prefill_chunk_with_size(..., 0)`)
    /// when the env knob is unset. Skips if the env knob is set
    /// (process-global OnceLock pollution would route through the
    /// chunked branch and invalidate the comparison).
    /// Requires Metal GPU; run with `--ignored`.
    /// `Qwen35MoeInner::new` can throw a foreign C++ exception on
    /// machines without Metal, which aborts the test process before
    /// Rust can catch the failure.
    #[test]
    #[ignore = "requires Metal GPU; run with --ignored"]
    fn test_run_paged_prefill_chunk_default_matches_single_shot_qwen3_5_moe() {
        if crate::array::memory::paged_prefill_chunk_size() != 0 {
            eprintln!(
                "skipping test_run_paged_prefill_chunk_default_matches_single_shot_qwen3_5_moe: \
                 MLX_PAGED_PREFILL_CHUNK_SIZE is set, default-path coverage is environment-dependent"
            );
            return;
        }

        let cfg = moe_paged_tiny_config();
        let mut inner = match Qwen35MoeInner::new(cfg.clone()) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") || msg.contains("No Metal device") {
                    eprintln!(
                        "skipping test_run_paged_prefill_chunk_default_matches_single_shot_qwen3_5_moe \
                         (no Metal): {msg}"
                    );
                    return;
                }
                panic!("unexpected Qwen35MoeInner::new failure: {msg}");
            }
        };
        cast_moe_inner_weights_bf16(&mut inner);

        // Short 8-token prompt: small enough that any positive
        // chunk_size value would also take the single-shot fast path.
        let prompt: Vec<u32> = vec![5, 11, 21, 33, 47, 60, 71, 83];

        // Init caches.
        if inner.caches.is_none() {
            inner.init_caches_sync().expect("init_caches");
        }

        let layer_kinds = compute_layer_kinds(inner.config.num_layers as usize, |i| {
            inner.config.is_linear_layer(i)
        });

        // Run 1: public default-path entry point.
        {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            adapter.reset_for_new_request(0).expect("reset");
            let prefix = adapter
                .find_cached_prefix(&prompt, &[], 0, false)
                .expect("find_cached_prefix");
            assert_eq!(prefix.cached_token_count, 0);
            adapter
                .allocate_suffix_blocks(prompt.len() as u32)
                .expect("allocate_suffix_blocks");
        }
        let logits_default = {
            let embed = inner.embedding.clone();
            let embedding_weight = embed.get_weight();
            let caches_ref = inner.caches.as_mut().expect("caches");
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            match super::run_paged_prefill_chunk(
                &prompt,
                &prompt,
                0,
                false,
                &embed,
                &mut inner.layers,
                caches_ref,
                &inner.final_norm,
                &inner.lm_head,
                &embedding_weight,
                &layer_kinds,
                adapter,
            ) {
                Ok(l) => l,
                Err(e) => {
                    let msg = e.reason.to_string();
                    if msg.contains("Metal GPU not available") || msg.contains("No Metal device") {
                        eprintln!(
                            "skipping test_run_paged_prefill_chunk_default_matches_single_shot_qwen3_5_moe: \
                             {msg}"
                        );
                        return;
                    }
                    panic!("unexpected default-path prefill failure: {msg}");
                }
            }
        };
        let default_vec = logits_to_f32_vec(&logits_default);
        assert_eq!(default_vec.len(), cfg.vocab_size as usize);
        for (i, v) in default_vec.iter().enumerate() {
            assert!(v.is_finite(), "default-path logits[{i}] not finite: {v}");
        }

        // Cleanup before second run.
        {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            let _ = adapter.register_full_blocks_for_reuse(&[], 0);
            adapter.release_request().expect("release_request");
        }
        inner.reset_caches_sync().expect("reset_caches");
        inner.init_caches_sync().expect("re-init_caches");

        // Run 2: explicit chunk_size=0.
        {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            adapter.reset_for_new_request(0).expect("reset");
            let prefix = adapter
                .find_cached_prefix(&prompt, &[], 0, false)
                .expect("find_cached_prefix");
            assert_eq!(prefix.cached_token_count, 0);
            adapter
                .allocate_suffix_blocks(prompt.len() as u32)
                .expect("allocate_suffix_blocks");
        }
        let logits_explicit = {
            let embed = inner.embedding.clone();
            let embedding_weight = embed.get_weight();
            let caches_ref = inner.caches.as_mut().expect("caches");
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            super::run_paged_prefill_chunk_with_size(
                &prompt,
                &prompt,
                0,
                false,
                &embed,
                &mut inner.layers,
                caches_ref,
                &inner.final_norm,
                &inner.lm_head,
                &embedding_weight,
                &layer_kinds,
                adapter,
                0,
            )
            .expect("explicit chunk_size=0 prefill")
        };
        let explicit_vec = logits_to_f32_vec(&logits_explicit);

        // Bytewise equality: both paths run the same code (single-shot
        // helper). Any divergence here would be a wrapper bug.
        for (i, (a, b)) in default_vec.iter().zip(explicit_vec.iter()).enumerate() {
            let abs_diff = (a - b).abs();
            assert!(
                abs_diff <= 1e-6,
                "MoE default path diverged from explicit chunk_size=0 at index {i}: \
                 default={a}, explicit={b}, abs_diff={abs_diff}"
            );
        }

        // Cleanup.
        {
            let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
            let _ = adapter.register_full_blocks_for_reuse(&[], 0);
            adapter.release_request().expect("release_request");
        }
    }
}
