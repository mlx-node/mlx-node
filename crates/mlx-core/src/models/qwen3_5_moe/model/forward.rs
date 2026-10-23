//! Shared prefill/forward numeric core for Qwen3.5 MoE.

use super::*;

use crate::models::forward as fwd;

// Distinct family tag; per-instance ids and format/destination bits bound every
// graph variant. Lifecycle erases these tapes before weights change or unload.
pub(crate) const COMPILED_VERIFY_TAG: u64 = 0xD51B_3500_0000_0000;
const COMPILED_VERIFY_INT8_BIT: u64 = 1 << 32;
const COMPILED_VERIFY_DST_BIT: u64 = 1 << 33;
const COMPILED_VERIFY_BUDGET_BIT: u64 = 1 << 34;
const COMPILED_VERIFY_DEFERRED_GDN_BIT: u64 = 1 << 36;
pub(crate) const COMPILED_VERIFY_ERASE_MASK: u64 = 0xFFFF_FFE0_FFFF_FF00;
#[cfg(test)]
thread_local! {
    static DFLASH2_COMPILED_TEST_COUNTS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
    static DFLASH2_INJECT_STORE_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
pub(crate) fn take_dflash2_compiled_test_counts() -> (usize, usize) {
    DFLASH2_COMPILED_TEST_COUNTS.with(|counts| counts.replace((0, 0)))
}

#[cfg(test)]
pub(crate) fn inject_dflash2_store_failure() {
    DFLASH2_INJECT_STORE_FAILURE.with(|fault| fault.set(true));
}

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
    match lm_head {
        Some(head) => head.forward_logits(hidden),
        None => embedding.as_linear(hidden),
    }
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

/// How much of the sequence an LM-head projection must cover for a DFlash2
/// target forward.
///
/// Verify blocks need every row (each verify position's next-token
/// distribution feeds acceptance). Prefill and single-token materialization
/// callers keep only the final row's logits, so projecting the head over the
/// whole chunk would run a full `M x vocab` GEMM for one surviving row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DFlash2LogitsSpan {
    All,
    LastRow,
}

/// Capture residual taps and recurrent rollback tape for the MoE DFlash2 verifier.
pub(crate) fn forward_dflash2_with_taps(
    inner: &mut Qwen35MoeInner,
    input_ids: &MxArray,
    tap_layers: &[usize],
    record_tape: bool,
    logits_span: DFlash2LogitsSpan,
    gdn_dst: Option<&crate::models::qwen3_5::gdn_blob::GdnDestinations>,
) -> Result<(
    MxArray,
    Vec<MxArray>,
    Vec<Option<crate::models::qwen3_5::gated_delta_net::GdnLayerTape>>,
)> {
    forward_dflash2_with_budget(
        inner,
        input_ids,
        tap_layers,
        record_tape,
        logits_span,
        gdn_dst,
        None,
    )
}

/// Device budget changes routed-expert work only; physical cache writes and
/// rollback tape keep the original block width. Eager fallback verifies all rows.
pub(crate) fn forward_dflash2_with_budget(
    inner: &mut Qwen35MoeInner,
    input_ids: &MxArray,
    tap_layers: &[usize],
    record_tape: bool,
    logits_span: DFlash2LogitsSpan,
    gdn_dst: Option<&crate::models::qwen3_5::gdn_blob::GdnDestinations>,
    row_budget: Option<&MxArray>,
) -> Result<(
    MxArray,
    Vec<MxArray>,
    Vec<Option<crate::models::qwen3_5::gated_delta_net::GdnLayerTape>>,
)> {
    if record_tape && matches!(logits_span, DFlash2LogitsSpan::All) {
        match forward_dflash2_compiled(inner, input_ids, tap_layers, gdn_dst, row_budget, None) {
            Ok(Some(out)) => return Ok(out),
            Ok(None) => {}
            Err(error) => {
                inner.dflash2_compiled_verify_disabled = true;
                // Store construction may have published some lazy KV writes.
                // The turn driver resets the whole target/draft lane on this
                // error; only the NEXT request may use the eager fallback.
                tracing::warn!(%error, "MoE DFlash2 compiled verify failed; resetting the turn and disabling future compiled verification");
                return Err(error);
            }
        }
    }
    if tap_layers.is_empty() || tap_layers.iter().any(|&layer| layer >= inner.layers.len()) {
        return Err(Error::from_reason(format!(
            "DFlash2 target tap layers are invalid: {tap_layers:?} for {} layers",
            inner.layers.len()
        )));
    }
    let mut hidden = inner.embedding.forward(input_ids)?;
    let mut taps: Vec<Option<MxArray>> = vec![None; tap_layers.len()];
    let mut tape = std::iter::repeat_with(|| None)
        .take(inner.layers.len())
        .collect::<Vec<_>>();
    for index in 0..inner.layers.len() {
        let cache = inner.caches.as_mut().map(|caches| &mut caches[index]);
        hidden = if record_tape {
            let mut slot = None;
            let hidden = inner.layers[index].forward_with_tape(
                &hidden,
                None,
                cache,
                None,
                true,
                Some(&mut slot),
            )?;
            tape[index] = slot;
            hidden
        } else {
            inner.layers[index].forward(&hidden, None, cache, None, true)?
        };
        if record_tape {
            crate::models::qwen3_5::paged_forward::maybe_submit_paged_decode_layer(
                index,
                inner.layers.len(),
                &hidden,
            );
        }
        for (slot, &tap_layer) in tap_layers.iter().enumerate() {
            if tap_layer == index {
                taps[slot] = Some(hidden.clone());
            }
        }
    }
    let taps = taps
        .into_iter()
        .enumerate()
        .map(|(slot, tap)| {
            tap.ok_or_else(|| {
                Error::from_reason(format!(
                    "DFlash2 target tap {} at layer {} was not captured",
                    slot, tap_layers[slot]
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    // Norm and the LM head are per-row ops: slicing the hidden tail first is
    // value-identical to projecting every row and slicing the logits after,
    // but skips the full-width head GEMM for callers that keep one row.
    let hidden = match logits_span {
        DFlash2LogitsSpan::All => hidden,
        DFlash2LogitsSpan::LastRow => {
            let rows = hidden.shape_at(1)?;
            hidden.slice_axis(1, rows - 1, rows)?
        }
    };
    let normalized = inner.final_norm.forward(&hidden)?;
    let logits = project_logits_from_hidden(&normalized, &inner.lm_head, &inner.embedding)?;
    Ok((logits, taps, tape))
}

/// Deferred recurrent state is private to the exclusive stepper. There is no
/// eager fallback after entering this graph: ordinary consumers require settlement.
pub(crate) fn forward_dflash2_deferred(
    inner: &mut Qwen35MoeInner,
    input_ids: &MxArray,
    tap_layers: &[usize],
    gdn_dst: &crate::models::qwen3_5::gdn_blob::GdnDestinations,
    row_budget: Option<&MxArray>,
    pending: &crate::models::qwen3_5::gdn_blob::GdnDeferredInput<'_>,
) -> Result<(
    MxArray,
    Vec<MxArray>,
    Vec<Option<crate::models::qwen3_5::gated_delta_net::GdnLayerTape>>,
)> {
    if input_ids.shape()?.as_ref() != [1, 8] {
        return Err(Error::from_reason(
            "deferred GDN verifier requires eight physical rows",
        ));
    }
    match forward_dflash2_compiled(
        inner,
        input_ids,
        tap_layers,
        Some(gdn_dst),
        row_budget,
        Some(pending),
    ) {
        Ok(Some(out)) => Ok(out),
        result => {
            inner.dflash2_compiled_verify_disabled = true;
            match result {
                Err(error) => Err(error),
                _ => Err(Error::from_reason(
                    "deferred GDN compiled verifier declined",
                )),
            }
        }
    }
}

/// Compiled-graph variant of the DFlash2 verify forward (the
/// `record_tape=true`, `DFlash2LogitsSpan::All`, batch-1 shape).
///
/// The entire verify forward — embedding lookup, every decoder layer, final
/// norm, LM head — is traced once per `(model, verify length L)` and replayed
/// through `mlx::core::compile`'s fused tape: the ~2400 per-cycle Rust op
/// constructions collapse into one FFI call plus `compile_replace`, and MLX's
/// fusion pass merges the elementwise/layout soup the eager builder emits.
///
/// # Graph contract
///
/// Inputs, in order:
///   `input_ids` `[1, L]` i32, `rope_offsets` `[1]` i32, then per layer:
///     Linear        — conv_state `[1, K-1, conv_dim]`,
///                     recurrent `[1, Hv, Dv, Dk]`
///     FullAttention — live K prefix `[1, Hkv, P, D]`, V prefix `[1, Hkv, P, D]`
///   then, with `gdn_dst`, per linear layer a state destination
///   `[1, Hv, Dv, Dk]` and a conv-history destination `[K-1, conv_dim]`
///   (the fused GDN kernel writes its window-final state there instead of
///   fresh buffers; `COMPILED_VERIFY_DST_BIT` in the id).
/// Outputs, in order:
///   `logits` `[1, L, V]`, tap hiddens (in `tap_layers` order), then per layer:
///     Linear        — kernel tape `q, k, v, g, beta` + post-mask `qkv` +
///                     the window-final recurrent state and conv history (8)
///     FullAttention — post-RoPE `new_k, new_v` `[1, Hkv, L, D]` (2)
///
/// `shapeless` lifts the prefix length `P` — which changes every cycle — out
/// of the compile cache key. Every host-int branch inside the builder depends
/// only on `L` and model constants, both folded into the `fn_id`, so the
/// traced tape is consistent on replay. `P` flows only through
/// shape-polymorphic ops: the K/V prefix concat (axis 2) and the fused SDPA
/// primitive, which derives `qL_off = kL - qL` from real input shapes at eval
/// time.
///
/// After a successful invoke the caller stores every layer's emitted KV block
/// in one `KVCache::store_blocks` dispatch, with the same offset advance as
/// the eager path. Commit's `trim(snapshot_offset + steps)` rollback is
/// unchanged.
///
/// Returns `Ok(None)` when the caches are not in the flat-verify shape the
/// graph expects (missing state slots, variant mismatch, divergent
/// full-attention offsets) or an FFI invoke declines before cache publication;
/// the caller then uses the eager path. Builder or store errors surface as
/// `Err` and fail the turn closed, because a store may already have published
/// some layer writes. A later request can use the eager path after reset.
fn forward_dflash2_compiled(
    inner: &mut Qwen35MoeInner,
    input_ids: &MxArray,
    tap_layers: &[usize],
    gdn_dst: Option<&crate::models::qwen3_5::gdn_blob::GdnDestinations>,
    row_budget: Option<&MxArray>,
    deferred: Option<&crate::models::qwen3_5::gdn_blob::GdnDeferredInput<'_>>,
) -> Result<
    Option<(
        MxArray,
        Vec<MxArray>,
        Vec<Option<crate::models::qwen3_5::gated_delta_net::GdnLayerTape>>,
    )>,
> {
    use crate::models::qwen3_5::attention::{VerifyKvOut, VerifyPrefix};
    use crate::models::qwen3_5::gated_delta::GdnKernelTape;
    use crate::models::qwen3_5::gated_delta_net::GdnLayerTape;
    use crate::models::qwen3_5_moe::decoder_layer::{LayerVerifyIo, VerifyResidual};
    use crate::transformer::KvFormat;

    if inner.dflash2_compiled_verify_disabled
        || std::env::var("MLX_MOE_DFLASH2_COMPILE").is_ok_and(|v| v == "0")
    {
        return Ok(None);
    }
    if tap_layers.is_empty() || tap_layers.iter().any(|&l| l >= inner.layers.len()) {
        return Err(Error::from_reason(format!(
            "DFlash2 target tap layers are invalid: {tap_layers:?} for {} layers",
            inner.layers.len()
        )));
    }
    let batch = input_ids.shape_at(0)?;
    let seq_len = input_ids.shape_at(1)?;
    if batch != 1 || !(1..=255).contains(&seq_len) {
        return Ok(None);
    }
    let Some(caches) = inner.caches.as_ref() else {
        return Ok(None);
    };
    if caches.len() != inner.layers.len() {
        return Ok(None);
    }

    // Gather graph inputs in the contract order. The shared RoPE base is the
    // full-attention offset — the flat verify invariant (asserted by
    // `flat_attention_frontier`) is that every FA cache agrees on it. Every
    // FA cache must also share one KV format: BF16 slots pass 2 arrays, int8
    // slots 4 (rows + scales); the format is part of the tape id.
    let mut rope_base: Option<i32> = None;
    let mut kv_format: Option<KvFormat> = None;
    let mut per_layer_state: Vec<MxArray> = Vec::with_capacity(4 * caches.len());
    let mut fa_prefix_offsets: Vec<Option<i32>> = Vec::with_capacity(caches.len());
    for (layer, cache) in inner.layers.iter().zip(caches.iter()) {
        match (&layer.attn, cache) {
            (
                crate::models::qwen3_5_moe::decoder_layer::AttentionType::Linear(_),
                Qwen3_5LayerCache::Linear(ac),
            ) => {
                let (Some(conv), Some(rec)) = (ac.get(0), ac.get(1)) else {
                    return Ok(None);
                };
                per_layer_state.push(conv.clone());
                per_layer_state.push(rec.clone());
                fa_prefix_offsets.push(None);
            }
            (
                crate::models::qwen3_5_moe::decoder_layer::AttentionType::Full(_),
                Qwen3_5LayerCache::FullAttention(kvc),
            ) => {
                let offset = kvc.get_offset();
                let (Some(keys), Some(values)) = (kvc.keys_ref(), kvc.values_ref()) else {
                    return Ok(None);
                };
                if offset <= 0 || offset as i64 > keys.shape_at(2)? {
                    return Ok(None);
                }
                match rope_base {
                    Some(base) if base != offset => return Ok(None),
                    None => rope_base = Some(offset),
                    _ => {}
                }
                match kv_format {
                    Some(format) if format != kvc.format() => return Ok(None),
                    None => kv_format = Some(kvc.format()),
                    _ => {}
                }
                match kvc.format() {
                    KvFormat::Bf16 => {
                        per_layer_state.push(keys.slice_axis(2, 0, offset as i64)?);
                        per_layer_state.push(values.slice_axis(2, 0, offset as i64)?);
                    }
                    KvFormat::Int8 => {
                        let Some(view) = kvc.int8_view() else {
                            return Ok(None);
                        };
                        per_layer_state.extend(view.arrays().into_iter().cloned());
                    }
                }
                fa_prefix_offsets.push(Some(offset));
            }
            _ => return Ok(None),
        }
    }
    let kv_format = kv_format.unwrap_or_default();
    let fa_arrays: usize = match kv_format {
        KvFormat::Bf16 => 2,
        KvFormat::Int8 => 4,
    };
    let n_linear = inner.layers.iter().filter(|l| l.is_linear()).count();
    let n_fa = inner.layers.len() - n_linear;
    // The destinations are a per-cycle input (parity flips every commit),
    // so the graph binds them as inputs; their count fixes the arity.
    let dst = gdn_dst.filter(|dst| dst.state.len() == n_linear && dst.conv.len() == n_linear);
    let has_dst = dst.is_some();
    let rope_offsets = MxArray::from_int32(&[rope_base.unwrap_or(0)], &[1])?;
    let mut inputs: Vec<MxArray> =
        Vec::with_capacity(2 + per_layer_state.len() + if has_dst { 2 * n_linear } else { 0 });
    inputs.push(input_ids.clone());
    inputs.push(rope_offsets);
    inputs.extend(per_layer_state);
    let dst_base = inputs.len();
    if let Some(dst) = dst {
        for (state, conv) in dst.state.iter().zip(&dst.conv) {
            inputs.extend([state.clone(), conv.clone()]);
        }
    }
    let deferred_base = inputs.len();
    let deferred_keep_index = if let Some(prior) = deferred {
        if !has_dst
            || prior.tape.len() != inner.layers.len()
            || prior.keep.shape()?.as_ref() != [1]
            || prior.keep.dtype()? != crate::array::DType::Int32
        {
            return Err(Error::from_reason("deferred GDN graph inputs are invalid"));
        }
        for (index, layer) in inner.layers.iter().enumerate() {
            if layer.is_linear() {
                let tape = prior.tape[index]
                    .as_ref()
                    .ok_or_else(|| Error::from_reason("deferred GDN prior tape is missing"))?;
                if tape.qkv.shape_at(1)? != 8 {
                    return Err(Error::from_reason("deferred GDN prior width changed"));
                }
                inputs.extend([
                    tape.kernel.k.clone(),
                    tape.kernel.v.clone(),
                    tape.kernel.g.clone(),
                    tape.kernel.beta.clone(),
                    tape.qkv.clone(),
                ]);
            }
        }
        let index = inputs.len();
        inputs.push(prior.keep.clone());
        Some(index)
    } else {
        None
    };
    let budget_index = row_budget.map(|budget| {
        let index = inputs.len();
        inputs.push(budget.clone());
        index
    });
    let n_outputs = 1 + tap_layers.len() + 8 * n_linear + fa_arrays * n_fa;
    // Per-(model, L) id: the verify length decides every host-int branch in
    // the builder (SDPA verify-split geometry), while the prefix length stays
    // shapeless. The high tag namespaces these ids away from the C++-side
    // pointer-derived ids and the test range.
    let mut shapeless_verify = true;
    for layer in &inner.layers {
        if let crate::models::qwen3_5_moe::decoder_layer::AttentionType::Full(attention) =
            &layer.attn
            && !attention.verify_can_be_shapeless(seq_len)?
        {
            shapeless_verify = false;
            break;
        }
    }
    if !shapeless_verify {
        // Unfused causal attention bakes prefix-sized mask constants. Reusing
        // that graph as the prefix grows is invalid; specializing every prefix
        // would retain an unbounded set of full-model traces. Use eager verify.
        return Ok(None);
    }
    let fn_id = COMPILED_VERIFY_TAG
        | ((inner.model_id & 0x00FF_FFFF) << 8)
        | (seq_len as u64 & 0xFF)
        | match kv_format {
            KvFormat::Bf16 => 0,
            KvFormat::Int8 => COMPILED_VERIFY_INT8_BIT,
        }
        | if has_dst { COMPILED_VERIFY_DST_BIT } else { 0 }
        | if row_budget.is_some() {
            COMPILED_VERIFY_BUDGET_BIT
        } else {
            0
        }
        | if deferred.is_some() {
            COMPILED_VERIFY_DEFERRED_GDN_BIT
        } else {
            0
        };

    let layers = &mut inner.layers;
    let embedding = &inner.embedding;
    let final_norm = &inner.final_norm;
    let lm_head = &inner.lm_head;
    let mut builder = move |graph_inputs: &[MxArray]| -> Result<Vec<MxArray>> {
        #[cfg(test)]
        DFLASH2_COMPILED_TEST_COUNTS.with(|counts| {
            let (completed, builds) = counts.get();
            counts.set((completed, builds + 1));
        });
        let ids = &graph_inputs[0];
        let rope_offsets = &graph_inputs[1];
        let mut cursor = 2usize;
        let row_budget = budget_index.map(|index| &graph_inputs[index]);
        let embedded = embedding.forward(ids)?;
        // Pending FFN+residual sums materialize at the next norm; source
        // Hidden outputs are already summed. Both expose the same raw tap,
        // and final normalization occurs exactly once.
        let mut pending: Option<super::super::decoder_layer::VerifyLayerOutput> = None;
        let mut taps: Vec<Option<MxArray>> = vec![None; tap_layers.len()];
        let mut capture = |layer: usize, hidden: &MxArray| {
            for (slot, &tap_layer) in tap_layers.iter().enumerate() {
                if tap_layer == layer {
                    taps[slot] = Some(hidden.clone());
                }
            }
        };
        let mut extras: Vec<MxArray> = Vec::with_capacity(8 * n_linear + fa_arrays * n_fa);
        let mut linear_ordinal = 0usize;
        for (index, layer) in layers.iter_mut().enumerate() {
            let input = match &pending {
                Some(output) => output.as_input(),
                None => VerifyResidual::Hidden(&embedded),
            };
            let mut tape_slot: Option<GdnLayerTape> = None;
            if layer.is_linear() {
                // Detached cache seeded with the graph-input states: the
                // verify-time writes land in it and are exported below;
                // commit replays the accepted prefix from the snapshot.
                // Slots 2 and 3 are the window-final state's and history's
                // destinations (the spare blob rows), which a full accept
                // adopts.
                let mut detached = crate::models::qwen3_5::arrays_cache::ArraysCache::new(
                    if deferred_keep_index.is_some() { 10 } else { 4 },
                );
                detached.set(0, graph_inputs[cursor].clone())?;
                detached.set(1, graph_inputs[cursor + 1].clone())?;
                if has_dst {
                    detached.set(2, graph_inputs[dst_base + 2 * linear_ordinal].clone())?;
                    detached.set(3, graph_inputs[dst_base + 2 * linear_ordinal + 1].clone())?;
                }
                if let Some(keep_index) = deferred_keep_index {
                    for slot in 0..5 {
                        detached.set(
                            4 + slot,
                            graph_inputs[deferred_base + 5 * linear_ordinal + slot].clone(),
                        )?;
                    }
                    detached.set(9, graph_inputs[keep_index].clone())?;
                }
                linear_ordinal += 1;
                let mut io = LayerVerifyIo::Linear(&mut detached);
                let (x, output) =
                    layer.forward_verify(input, &mut io, true, Some(&mut tape_slot), row_budget)?;
                if index > 0 {
                    capture(index - 1, &x);
                }
                pending = Some(output);
                let tape = tape_slot.ok_or_else(|| {
                    Error::from_reason("compiled verify: GDN layer produced no tape")
                })?;
                let (Some(cache_conv), Some(final_state)) =
                    (detached.get(0).cloned(), detached.get(1).cloned())
                else {
                    return Err(Error::from_reason(
                        "compiled verify: GDN layer produced no state",
                    ));
                };
                let GdnLayerTape {
                    kernel,
                    qkv,
                    final_conv,
                    ..
                } = tape;
                let GdnKernelTape { q, k, v, g, beta } = kernel;
                // The kernel's own history output when the layer recorded
                // one (adoptable in place); otherwise the cache view.
                extras.extend([
                    q,
                    k,
                    v,
                    g,
                    beta,
                    qkv,
                    final_state,
                    final_conv.unwrap_or(cache_conv),
                ]);
                cursor += 2;
            } else {
                let mut out_kv = None;
                {
                    let prefix = match kv_format {
                        KvFormat::Bf16 => VerifyPrefix::Bf16 {
                            keys: &graph_inputs[cursor],
                            values: &graph_inputs[cursor + 1],
                        },
                        KvFormat::Int8 => VerifyPrefix::Int8 {
                            keys: &graph_inputs[cursor],
                            values: &graph_inputs[cursor + 1],
                            key_scales: &graph_inputs[cursor + 2],
                            value_scales: &graph_inputs[cursor + 3],
                        },
                    };
                    let mut io = LayerVerifyIo::FullAttention(
                        crate::models::qwen3_5::attention::AttentionVerifyIo {
                            prefix,
                            rope_offsets,
                            out_kv: &mut out_kv,
                        },
                    );
                    let (x, output) = layer.forward_verify(
                        input,
                        &mut io,
                        true,
                        Some(&mut tape_slot),
                        row_budget,
                    )?;
                    if index > 0 {
                        capture(index - 1, &x);
                    }
                    pending = Some(output);
                }
                match out_kv {
                    Some(VerifyKvOut::Bf16(new_k, new_v)) if kv_format == KvFormat::Bf16 => {
                        extras.push(new_k);
                        extras.push(new_v);
                    }
                    Some(VerifyKvOut::Int8(rows)) if kv_format == KvFormat::Int8 => {
                        extras.extend(rows.arrays().into_iter().cloned());
                    }
                    _ => {
                        return Err(Error::from_reason(
                            "compiled verify: attention layer produced no kv block in the \
                             cache's format",
                        ));
                    }
                }
                cursor += fa_arrays;
            }
        }
        let output =
            pending.ok_or_else(|| Error::from_reason("compiled verify: model has no layers"))?;
        let (hidden, normalized) = output.normalize(final_norm)?;
        capture(layers.len() - 1, &hidden);
        let logits = project_logits_from_hidden(&normalized, lm_head, embedding)?;
        let mut outputs = Vec::with_capacity(n_outputs);
        outputs.push(logits);
        for (slot, tap) in taps.into_iter().enumerate() {
            outputs.push(tap.ok_or_else(|| {
                Error::from_reason(format!(
                    "compiled verify: tap {slot} at layer {} was not captured",
                    tap_layers[slot]
                ))
            })?);
        }
        outputs.extend(extras);
        Ok(outputs)
    };

    let input_refs: Vec<&MxArray> = inputs.iter().collect();
    let Some(outputs) = crate::compiled_graph::invoke_compiled_graph(
        fn_id,
        &input_refs,
        n_outputs,
        true,
        &mut builder,
    )?
    else {
        return Ok(None);
    };

    // Unpack the contract: logits, taps, then per-layer extras in layer order.
    // Any failure after the invoke rewinds published FA offsets before the
    // error escapes. Store construction may already have published handles,
    // so the turn driver resets the whole lane instead of retrying in place.
    let unpack = |outputs: &[MxArray],
                  caches: &mut [Qwen3_5LayerCache],
                  tape: &mut [Option<GdnLayerTape>]|
     -> Result<()> {
        let mut cursor = 1 + tap_layers.len();
        let mut blocks = Vec::new();
        for (index, layer) in inner.layers.iter().enumerate() {
            if layer.is_linear() {
                let kd = match &layer.attn {
                    crate::models::qwen3_5_moe::decoder_layer::AttentionType::Linear(gdn) => {
                        gdn.conv_kernel_dim()
                    }
                    _ => {
                        return Err(Error::from_reason(
                            "compiled verify: linear layer kind mismatch",
                        ));
                    }
                };
                let mut take = |outputs: &[MxArray]| -> Result<MxArray> {
                    let a = outputs.get(cursor).ok_or_else(|| {
                        Error::from_reason("compiled verify: output arity shortfall")
                    })?;
                    cursor += 1;
                    Ok(a.clone())
                };
                tape[index] = Some(GdnLayerTape {
                    kernel: GdnKernelTape {
                        q: take(outputs)?,
                        k: take(outputs)?,
                        v: take(outputs)?,
                        g: take(outputs)?,
                        beta: take(outputs)?,
                    },
                    qkv: take(outputs)?,
                    conv_kernel_dim: kd,
                    // Only a destination-written state is adoptable; a fresh
                    // one is just the dropped verify write.
                    final_state: {
                        let state = take(outputs)?;
                        has_dst.then_some(state)
                    },
                    final_conv: {
                        let conv = take(outputs)?;
                        has_dst.then_some(conv)
                    },
                });
            } else {
                let kvc = caches[index].as_kv_cache_mut().ok_or_else(|| {
                    Error::from_reason("compiled verify: attention cache kind mismatch")
                })?;
                if kvc.format() != kv_format {
                    return Err(Error::from_reason(
                        "compiled verify: attention cache format changed mid-invoke",
                    ));
                }
                let Some(block) = outputs.get(cursor..cursor + fa_arrays) else {
                    return Err(Error::from_reason(
                        "compiled verify: output arity shortfall",
                    ));
                };
                cursor += fa_arrays;
                blocks.push((
                    index,
                    match kv_format {
                        KvFormat::Bf16 => crate::transformer::kv_cache::KvBlock::Bf16 {
                            keys: block[0].clone(),
                            values: block[1].clone(),
                        },
                        KvFormat::Int8 => crate::transformer::kv_cache::KvBlock::Int8(
                            crate::array::kv_int8::Int8KvRows {
                                keys: block[0].clone(),
                                values: block[1].clone(),
                                key_scales: block[2].clone(),
                                value_scales: block[3].clone(),
                            },
                        ),
                    },
                ));
            }
        }
        // One store dispatch for every layer's rows (+ scales), in place.
        let mut blocks = blocks.into_iter().peekable();
        let mut entries = Vec::with_capacity(blocks.len());
        for (index, cache) in caches.iter_mut().enumerate() {
            if blocks.peek().is_some_and(|(at, _)| *at == index)
                && let Some((_, block)) = blocks.next()
            {
                let kvc = cache.as_kv_cache_mut().ok_or_else(|| {
                    Error::from_reason("compiled verify: attention cache kind mismatch")
                })?;
                entries.push((kvc, block));
            }
        }
        #[cfg(test)]
        if DFLASH2_INJECT_STORE_FAILURE.with(|fault| fault.replace(false)) {
            if let Some((cache, block)) = entries.first_mut() {
                cache.store_block(block)?;
            }
            return Err(Error::from_reason("injected compiled MoE KV store failure"));
        }
        crate::transformer::kv_cache::KVCache::store_blocks(&mut entries)
    };
    let mut tape: Vec<Option<GdnLayerTape>> = std::iter::repeat_with(|| None)
        .take(inner.layers.len())
        .collect();
    let logits = outputs[0].clone();
    let taps: Vec<MxArray> = outputs[1..1 + tap_layers.len()].to_vec();
    {
        let caches = inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("compiled verify: caches dropped mid-invoke"))?;
        if let Err(e) = unpack(&outputs, caches, &mut tape) {
            for (index, prefix) in fa_prefix_offsets.iter().enumerate() {
                if let (Some(prefix), Some(kvc)) = (*prefix, caches[index].as_kv_cache_mut()) {
                    kvc.trim(prefix);
                }
            }
            return Err(e);
        }
    }
    #[cfg(test)]
    DFLASH2_COMPILED_TEST_COUNTS.with(|counts| {
        let (completed, builds) = counts.get();
        counts.set((completed + 1, builds));
    });
    Ok(Some((logits, taps, tape)))
}

#[cfg(test)]
mod capture_erase_tests {
    use super::*;

    #[test]
    fn capture_and_cache_format_variants_erase_with_model() -> Result<()> {
        if !crate::engine::persistence::compiled_forward_backend_available()
            || unsafe { mlx_sys::mlx_default_device() } != 1
        {
            return Ok(());
        }
        let base = COMPILED_VERIFY_TAG | (0x00A1_A101 << 8);
        let other = COMPILED_VERIFY_TAG | (0x00A1_A102 << 8) | 8;
        let ids: Vec<u64> = (0..32u64).map(|flags| base | (flags << 32) | 8).collect();
        for &id in ids.iter().chain(std::iter::once(&other)) {
            crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, id);
        }
        let input = MxArray::from_float32(&[1.0], &[1])?;
        let mut builds = std::collections::HashMap::<u64, usize>::new();
        let mut invoke = |id: u64| -> Result<()> {
            let mut builder = |inputs: &[MxArray]| -> Result<Vec<MxArray>> {
                *builds.entry(id).or_default() += 1;
                Ok(vec![inputs[0].add(&inputs[0])?])
            };
            let output =
                crate::compiled_graph::invoke_compiled_graph(id, &[&input], 1, true, &mut builder)?
                    .ok_or_else(|| Error::from_reason("capture erase fixture compile declined"))?;
            assert_eq!(output[0].to_float32()?.as_ref(), &[2.0]);
            Ok(())
        };
        for &id in ids.iter().chain(std::iter::once(&other)) {
            invoke(id)?;
        }
        // The real lifecycle mask must erase every capture/KV/budget/destination
        // combination while preserving a different model's compiled tape.
        crate::compiled_graph::erase_compiled_graphs_matching(COMPILED_VERIFY_ERASE_MASK, base);
        for &id in ids.iter().chain(std::iter::once(&other)) {
            invoke(id)?;
        }
        drop(invoke);
        for id in ids {
            assert_eq!(builds[&id], 2, "variant {id:x} survived model erasure");
        }
        assert_eq!(builds[&other], 1, "different model was incorrectly erased");
        crate::compiled_graph::erase_compiled_graphs_matching(COMPILED_VERIFY_ERASE_MASK, base);
        crate::compiled_graph::erase_compiled_graphs_matching(u64::MAX, other);
        Ok(())
    }
}
