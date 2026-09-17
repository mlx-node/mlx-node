//! Shared prefill/forward numeric core for Qwen3.5 MoE.

use super::*;

use crate::models::forward as fwd;

/// Run the MoE eager layer stack over `[1, T]` ids and return the
/// pre-final-norm hidden `[1, T, hidden]`.
///
/// This is the shared eager-MTP primitive: it advances `caches` (the flat
/// per-layer caches: `Linear` GDN slots + `FullAttention` KV slots) by `T`
/// and returns the full per-position hidden, exactly mirroring the dense
/// `forward_pre_norm_inner`. No explicit mask is ever built: `Linear` (GDN)
/// layers run mask-free, and full-attention layers pass `mask: None` too, so
/// `Qwen3_5Attention::forward` picks its fused "causal" SDPA kernel whenever
/// `seq_len > 1` — covering both prefill and the `[1, K+1]` eager-MTP verify
/// shape this helper backs.
pub(super) fn forward_pre_norm_inner(
    input_ids: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    _fa_idx: usize,
) -> Result<MxArray> {
    fwd::forward_pre_norm_with(
        input_ids,
        |ids| embedding.forward(ids),
        layers,
        caches,
        |layer, h, caches, i| {
            layer.forward(h, None, caches.as_mut().map(|c| &mut c[i]), None, true)
        },
        None,
    )
}

/// Like [`forward_pre_norm_inner`], but records a per-layer GDN tape for the
/// eager-MTP rollback replay. `tape` must be pre-sized to `layers.len()`;
/// each GDN (`Linear`) layer writes `Some(GdnLayerTape)` into its slot and
/// full-attention layers leave it `None` — the exact indexing the rollback
/// replay relies on. The forward output is byte-identical to the non-tape
/// variant (the tape is a side-channel clone of the kernel inputs). No
/// explicit mask is built here either — see `forward_pre_norm_inner`.
fn forward_pre_norm_inner_with_tape(
    input_ids: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    _fa_idx: usize,
    tape: &mut [Option<crate::models::qwen3_5_moe::gated_delta_net::GdnLayerTape>],
) -> Result<MxArray> {
    debug_assert_eq!(
        tape.len(),
        layers.len(),
        "forward_pre_norm_inner_with_tape: tape length must equal layer count"
    );
    fwd::forward_pre_norm_with(
        input_ids,
        |ids| embedding.forward(ids),
        layers,
        caches,
        |layer, h, caches, i| {
            let mut slot = None;
            let out = layer.forward_with_tape(
                h,
                None,
                caches.as_mut().map(|c| &mut c[i]),
                None,
                true,
                Some(&mut slot),
            )?;
            tape[i] = slot;
            Ok(out)
        },
        None,
    )
}

/// Project a pre-/post-final-norm hidden to logits, preserving the leading
/// dims (`[*, hidden] -> [*, vocab]`). Uses the explicit `lm_head` when
/// present, else the tied embedding's linear projection. Mirrors the dense
/// `project_logits_from_hidden` and keeps packed embeddings packed-resident.
pub(super) fn project_logits_from_hidden(
    hidden: &MxArray,
    lm_head: &Option<LinearProj>,
    embedding: &Embedding,
) -> Result<MxArray> {
    fwd::project_logits(hidden, lm_head.as_ref(), embedding)
}

/// Batched eager verify: run the `[1, K+1]` verify ids through the MoE main
/// stack (recording the GDN tape when `tape` is `Some`), apply the final norm,
/// and project logits. Advances `caches` by `K+1`. Returns the dense logits
/// `[1, K+1, vocab]` plus the post-final-norm hiddens `[1, K+1, hidden]`
/// (`MtpVerifyOutput::logits_only`). Mirrors the dense `eager_verify_step`.
#[allow(clippy::too_many_arguments)]
pub(super) fn eager_verify_step(
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    fa_idx: usize,
    verify_ids: &MxArray,
    embedding: &Embedding,
    tape: Option<&mut Vec<Option<crate::models::qwen3_5_moe::gated_delta_net::GdnLayerTape>>>,
) -> Result<mtp_decode::MtpVerifyOutput> {
    let pre = match tape {
        Some(tape) => {
            tape.clear();
            tape.resize(layers.len(), None);
            forward_pre_norm_inner_with_tape(verify_ids, embedding, layers, caches, fa_idx, tape)?
        }
        None => forward_pre_norm_inner(verify_ids, embedding, layers, caches, fa_idx)?,
    };
    let hiddens = final_norm.forward(&pre)?;
    let logits = project_logits_from_hidden(&hiddens, lm_head, embedding)?;
    Ok(mtp_decode::MtpVerifyOutput::logits_only(logits, hiddens))
}

/// Forward pass using already-acquired lock guards (no lock overhead).
///
/// Used by generate/chat to avoid re-acquiring locks on every decode step.
pub(super) fn forward_inner(
    input_ids: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    _fa_idx: usize,
) -> Result<MxArray> {
    // No explicit mask is ever built. Full-attention layers pass `mask:
    // None`, so `Qwen3_5Attention::forward` picks its fused "causal" SDPA
    // kernel whenever `seq_len > 1` (prefill and the `[1, K+1]` eager-MTP
    // verify shape alike) instead of a materialized boolean-mask array.
    // Linear (GDN) layers already ran mask-free — mlx-vlm never creates one
    // for `ArraysCache`, and an all-ones mask would just be a no-op that
    // adds graph nodes and Metal overhead. Mirrors the dense
    // `forward_pre_norm_inner`.
    let h = fwd::forward_body_normed(
        input_ids,
        embedding,
        layers,
        caches,
        final_norm,
        |layer, h, caches, i| {
            layer.forward(h, None, caches.as_mut().map(|c| &mut c[i]), None, true)
        },
    )?;
    fwd::project_logits(&h, lm_head.as_ref(), embedding)
}

/// Default prefill chunk size (tokens per chunk).
///
/// Matches the Qwen3.5 Dense path and Python mlx-lm's `prefill_step_size`
/// default of 2048. Chunking bounds the per-layer transient peak at
/// `chunk × hidden_dim` and inserts a cache-eval +
/// `clear_cache` barrier between chunks so the transient allocator state
/// does not accumulate across chunks.
pub(crate) const PREFILL_STEP_SIZE: i64 = 2048;

/// Chunked prefill for Qwen3.5 MoE.
///
/// Processes `prompt` (shape `[1, seq_len]`) in chunks of `PREFILL_STEP_SIZE`
/// tokens, evaluating all KV-cache arrays and clearing the MLX compute cache
/// between chunks to bound peak GPU activation memory. Returns the logits
/// from the **final** chunk, which share the same shape contract as a
/// single-shot `forward_inner` call: `[1, last_chunk_len, vocab_size]`.
///
/// Invariants vs. single-shot `forward_inner`:
/// - Identical numerical output at full precision (the KV caches thread
///   through chunk N into chunk N+1 just like they would through
///   successive `forward_inner(full_prompt)` calls during regular decode).
/// - The linear-attention recurrent state advances chunk-by-chunk. This is
///   the same forward direction as a single-shot call — chunking is a
///   memory-only transformation, not a semantic one.
/// - The decode KV caches are seeded in-place: `caches` is advanced
///   chunk-by-chunk through `&mut`, so when this returns the per-layer
///   `Qwen3_5LayerCache` entries (and the GDN recurrent state) already
///   reflect the full prompt. There is no separate post-prefill seeding
///   step for the caller to run.
///
/// Small prompts (<= `PREFILL_STEP_SIZE` tokens) hit exactly one loop
/// iteration and behave identically to a single `forward_inner` call — no
/// extra evals, no extra cache clears.
#[allow(clippy::too_many_arguments)]
pub(super) fn chunked_prefill(
    prompt: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    fa_idx: usize,
    generation_stream: Stream,
    turn_cancel: Option<&AtomicBool>,
) -> Result<MxArray> {
    chunked_prefill_with_size(
        prompt,
        embedding,
        layers,
        caches,
        final_norm,
        lm_head,
        fa_idx,
        generation_stream,
        PREFILL_STEP_SIZE,
        turn_cancel,
    )
}

/// Explicit-size variant of `chunked_prefill`.
///
/// Same semantics as `chunked_prefill` but the chunk size is an explicit
/// parameter. Primarily used by tests to compare chunked vs single-shot
/// (by passing a chunk size >= prompt length) without plumbing a config
/// knob through every caller. Production callers should use
/// `chunked_prefill` which hardcodes `PREFILL_STEP_SIZE`.
#[allow(clippy::too_many_arguments)]
fn chunked_prefill_with_size(
    prompt: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    fa_idx: usize,
    generation_stream: Stream,
    chunk_size: i64,
    turn_cancel: Option<&AtomicBool>,
) -> Result<MxArray> {
    debug_assert!(chunk_size > 0, "chunk_size must be positive");
    let mut ctx = (embedding, layers, caches, final_norm, lm_head, fa_idx);
    fwd::chunked_prefill(
        &mut ctx,
        prompt,
        generation_stream,
        chunk_size,
        true,
        move |_| turn_cancel,
        |ctx, chunk, _is_final| {
            // Every chunk (looped and final) runs the full forward_inner —
            // the looped logits are discarded by the driver.
            forward_inner(chunk, ctx.0, ctx.1, ctx.2, ctx.3, ctx.4, ctx.5)
        },
        |ctx| {
            // Materialize all cache arrays on GPU so the next chunk doesn't
            // extend a giant lazy graph rooted at the prior chunk's inputs.
            fwd::eval_caches_and_clear(&*ctx.2)
        },
    )
}
