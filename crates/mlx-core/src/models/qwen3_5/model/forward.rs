//! Shared prefill/forward numeric core.

use super::*;

/// Default prefill chunk size (tokens per chunk).
/// Matches Python mlx-lm's `prefill_step_size` default of 2048.
///
/// E55: bumped 1024 → 2048 after benching against mlx-lm at 20k prompt:
/// chunk=1024 incurred 20 chunk boundaries vs mlx-lm's 10 (mlx-lm uses
/// 2048 by default); the doubled per-chunk overhead cost ~14% at 20k.
/// At 1024-prompt single-chunk the value is irrelevant — the loop is
/// guarded by `total_len - offset > PREFILL_STEP_SIZE` so any T < step
/// goes through the single `remaining` branch unchanged.
pub(crate) const PREFILL_STEP_SIZE: i64 = 2048;

/// Evaluate all cache arrays across all layers to materialize them on GPU.
/// Must be called between prefill chunks to break lazy dependency chains.
pub(crate) fn eval_layer_caches(caches: &Option<Vec<Qwen3_5LayerCache>>) -> Result<()> {
    if let Some(caches) = caches {
        let mut arrays: Vec<&MxArray> = Vec::new();
        for cache in caches.iter() {
            cache.collect_arrays(&mut arrays);
        }
        MxArray::eval_arrays(&arrays)?;
    }
    Ok(())
}

/// Async variant of `eval_layer_caches`: kicks GPU on cache materialization
/// but does NOT block the CPU. Used between prefill chunks so the CPU can
/// start building the next chunk's graph while the previous chunk's cache
/// writes are still in flight.
pub(crate) fn async_eval_layer_caches(caches: &Option<Vec<Qwen3_5LayerCache>>) {
    if let Some(caches) = caches {
        let mut arrays: Vec<&MxArray> = Vec::new();
        for cache in caches.iter() {
            cache.collect_arrays(&mut arrays);
        }
        MxArray::async_eval_arrays(&arrays);
    }
}

/// Chunked prefill: process prompt in chunks of `PREFILL_STEP_SIZE`, evaluating
/// caches and clearing compute cache between chunks to bound peak memory.
///
/// Accepts `&MxArray` shaped `[1, seq_len]`. Slices on GPU — no data roundtrip.
/// For `&[u32]` inputs (from tokenizer), callers convert with `MxArray::from_uint32` first.
#[allow(clippy::too_many_arguments)]
pub(super) fn chunked_prefill(
    prompt: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    generation_stream: crate::stream::Stream,
    turn_cancel: Option<&AtomicBool>,
) -> Result<MxArray> {
    chunked_prefill_with_size(
        prompt,
        embedding,
        layers,
        caches,
        final_norm,
        lm_head,
        generation_stream,
        PREFILL_STEP_SIZE,
        turn_cancel,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn chunked_prefill_with_size(
    prompt: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    generation_stream: crate::stream::Stream,
    chunk_size: i64,
    turn_cancel: Option<&AtomicBool>,
) -> Result<MxArray> {
    let total_len = prompt.shape_at(1)?;
    if total_len <= 0 {
        return Err(Error::from_reason("chunked_prefill: empty prompt"));
    }
    let chunk_size = if chunk_size <= 0 {
        total_len
    } else {
        chunk_size
    };
    let mut offset: i64 = 0;

    // E28: env-var toggle for A/B. Default: async between chunks. When set,
    // falls back to synchronous eval_layer_caches (the prior behavior).
    let chunk_async = std::env::var("MLX_PREFILL_SYNC_BETWEEN_CHUNKS").is_err();
    while total_len - offset > chunk_size {
        // Cooperative-cancel checkpoint (H1b): abort at the chunk
        // boundary. The Err rides the flat engine's
        // `fail_closed_flat_turn` arm — no `save_cache_state`, the
        // session is invalidated, so the partially-advanced caches never
        // become a live prefix.
        if turn_cancel.is_some_and(|f| f.load(Ordering::Relaxed)) {
            return Err(Error::from_reason("prefill cancelled"));
        }
        let chunk = prompt.slice_axis(1, offset, offset + chunk_size)?;
        {
            let _stream_ctx = StreamContext::new(generation_stream);
            let _hidden = forward_pre_norm_inner(&chunk, embedding, layers, caches)?;
        }
        if chunk_async {
            async_eval_layer_caches(caches);
        } else {
            eval_layer_caches(caches)?;
        }
        crate::array::clear_cache();
        offset += chunk_size;
    }

    // The final remainder is a chunk boundary too once at least one looped
    // chunk ran: poll before forwarding it so a cancel landing during the
    // last looped chunk aborts instead of riding through the remainder.
    // `offset == 0` means the whole prompt fits in one forward — single-shot
    // prefills stay uncancellable by design.
    if offset > 0 && turn_cancel.is_some_and(|f| f.load(Ordering::Relaxed)) {
        return Err(Error::from_reason("prefill cancelled"));
    }
    let remaining = prompt.slice_axis(1, offset, total_len)?;
    let last_logits = {
        let _stream_ctx = StreamContext::new(generation_stream);
        let hidden = forward_pre_norm_inner(&remaining, embedding, layers, caches)?;
        project_last_logits_from_pre_norm_hidden(&hidden, final_norm, lm_head, embedding)?
    };
    Ok(last_logits)
}

/// `chunked_prefill` variant that ALSO returns the post-final-norm hidden
/// state for the prompt tail needed by MTP, concatenated along the time axis
/// -> `[1, kept_len, hidden]`.
///
/// Used only when MTP is active for the turn: the prompt hiddens flow
/// through `ChatDecodeInputs::prompt_hidden` into `begin_mtp_decode`'s
/// prompt-prefix seed, which commits the prompt prefix into the MTP
/// committed-history caches. Logits-only callers keep the cheaper
/// `chunked_prefill`. The per-chunk forward op sequence is identical for
/// chunks whose hidden is kept; chunks before the requested tail use the
/// logits-only path and discard hidden to avoid materializing prompt history
/// MTPLX would not seed.
#[allow(clippy::too_many_arguments)]
pub(super) fn chunked_prefill_with_hidden(
    prompt: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    generation_stream: crate::stream::Stream,
    keep_last_hidden: Option<usize>,
    turn_cancel: Option<&AtomicBool>,
) -> Result<(MxArray, MxArray)> {
    chunked_prefill_with_hidden_with_size(
        prompt,
        embedding,
        layers,
        caches,
        final_norm,
        lm_head,
        generation_stream,
        keep_last_hidden,
        PREFILL_STEP_SIZE,
        turn_cancel,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn chunked_prefill_with_hidden_with_size(
    prompt: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    generation_stream: crate::stream::Stream,
    keep_last_hidden: Option<usize>,
    chunk_size: i64,
    turn_cancel: Option<&AtomicBool>,
) -> Result<(MxArray, MxArray)> {
    let total_len = prompt.shape_at(1)?;
    if total_len <= 0 {
        return Err(Error::from_reason(
            "chunked_prefill_with_hidden: empty prompt",
        ));
    }
    let chunk_size = if chunk_size <= 0 {
        total_len
    } else {
        chunk_size
    };
    let mut offset: i64 = 0;
    let mut hidden_chunks: Vec<MxArray> = Vec::new();
    let keep_start = keep_last_hidden
        .map(|keep| total_len.saturating_sub(keep.max(1) as i64))
        .unwrap_or(0);

    while total_len - offset > chunk_size {
        // Cooperative-cancel checkpoint (H1b): abort at the chunk boundary,
        // same contract as `chunked_prefill_with_size`.
        if turn_cancel.is_some_and(|f| f.load(Ordering::Relaxed)) {
            return Err(Error::from_reason("prefill cancelled"));
        }
        let end = offset + chunk_size;
        let chunk = prompt.slice_axis(1, offset, end)?;
        let overlaps_kept_tail = end > keep_start;
        let kept_hidden = if overlaps_kept_tail {
            let _stream_ctx = StreamContext::new(generation_stream);
            let hidden = forward_pre_norm_inner(&chunk, embedding, layers, caches)?;
            let keep_from = keep_start.max(offset);
            let hidden = if keep_from > offset {
                hidden.slice_axis(1, keep_from - offset, end - offset)?
            } else {
                hidden
            };
            Some(final_norm.forward(&hidden)?)
        } else {
            let _stream_ctx = StreamContext::new(generation_stream);
            let _hidden = forward_pre_norm_inner(&chunk, embedding, layers, caches)?;
            None
        };
        eval_layer_caches(caches)?;
        if let Some(kept_hidden) = kept_hidden {
            // Materialize the kept hidden before clearing the MLX cache — it
            // is a lazy handle referencing graph nodes that `clear_cache`
            // would otherwise free.
            kept_hidden.eval();
            hidden_chunks.push(kept_hidden);
        }
        crate::array::clear_cache();
        offset = end;
    }

    // Final-remainder boundary poll, mirroring `chunked_prefill_with_size`:
    // single-shot (`offset == 0`) stays uncancellable by design.
    if offset > 0 && turn_cancel.is_some_and(|f| f.load(Ordering::Relaxed)) {
        return Err(Error::from_reason("prefill cancelled"));
    }
    let remaining = prompt.slice_axis(1, offset, total_len)?;
    let (last_logits, last_hidden) = {
        let _stream_ctx = StreamContext::new(generation_stream);
        let hidden = forward_pre_norm_inner(&remaining, embedding, layers, caches)?;
        let logits =
            project_last_logits_from_pre_norm_hidden(&hidden, final_norm, lm_head, embedding)?;
        let keep_from = keep_start.max(offset);
        let hidden = if keep_from > offset {
            hidden.slice_axis(1, keep_from - offset, total_len - offset)?
        } else {
            hidden
        };
        (logits, final_norm.forward(&hidden)?)
    };
    hidden_chunks.push(last_hidden);

    // Concatenate every kept `[1, chunk, hidden]` along axis 1 →
    // `[1, kept_len, hidden]`.
    let prompt_hidden = if hidden_chunks.len() == 1 {
        hidden_chunks
            .into_iter()
            .next()
            .ok_or_else(|| Error::from_reason("chunked_prefill_with_hidden: empty hidden chunks"))?
    } else {
        let mut acc = hidden_chunks[0].clone();
        for chunk in &hidden_chunks[1..] {
            acc = MxArray::concatenate(&acc, chunk, 1)?;
        }
        acc
    };
    Ok((last_logits, prompt_hidden))
}

/// Lock-free forward pass through all layers.
/// Attention layer handles causal masking internally via "causal" SDPA mode.
/// Format an `MxArray`'s shape for logging. Returns `[d0, d1, ...]`
/// or `"<unavailable>"` if `ndim()` fails.
fn shape_dbg(arr: &MxArray) -> String {
    let ndim = match arr.ndim() {
        Ok(n) => n,
        Err(_) => return "<unavailable>".to_string(),
    };
    let mut dims: Vec<i64> = Vec::with_capacity(ndim as usize);
    for axis in 0..ndim {
        match arr.shape_at(axis) {
            Ok(d) => dims.push(d),
            Err(_) => return "<unavailable>".to_string(),
        }
    }
    format!("{:?}", dims)
}

pub(super) fn forward_inner(
    input_ids: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
) -> Result<MxArray> {
    let hidden = forward_pre_norm_inner(input_ids, embedding, layers, caches)?;
    let hidden = final_norm.forward(&hidden)?;
    project_logits_from_hidden(&hidden, lm_head, embedding)
}

pub(super) fn forward_pre_norm_inner(
    input_ids: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
) -> Result<MxArray> {
    let hidden_states = embedding.forward(input_ids)?;
    let mut h = hidden_states.clone();

    debug!(
        "Qwen3.5 forward_inner: input_ids_shape={} post_embed_shape={}",
        shape_dbg(input_ids),
        shape_dbg(&h),
    );

    let num_layers = layers.len();
    // Plain layer loop. In-loop async_eval was tested (every 8 layers,
    // including h + all cache arrays) and found neutral-to-negative at
    // single-chunk prefill on M3 (back-to-back A/B in the same binary
    // showed deltas inside the run-to-run noise band). The CPU/GPU
    // overlap benefit only materializes across the inter-chunk barrier
    // in chunked_prefill, which now uses async_eval_layer_caches.
    //
    // This is the SHARED pre-norm primitive: it MUST return the full
    // per-position hidden. The MTP prompt-hidden path
    // (`chunked_prefill_with_hidden_with_size`) keeps the result and
    // re-slices it by chunk length, so a last-token slice here would
    // corrupt it. The logits-only callers get the equivalent of the
    // upstream E37 last-token optimization from
    // `project_last_logits_from_pre_norm_hidden` (which slices before
    // `final_norm` + `lm_head`), so the slice deliberately does NOT
    // live in this loop.
    for i in 0..num_layers {
        let cache = caches.as_mut().map(|c| &mut c[i]);
        h = layers[i].forward(&h, None, cache, None, true)?;
        if i == 0 || i + 1 == num_layers {
            debug!(
                "Qwen3.5 forward_inner: post_layer[{}/{}] shape={}",
                i,
                num_layers,
                shape_dbg(&h),
            );
        }
    }

    Ok(h)
}

/// Tape-recording variant of [`forward_pre_norm_inner`] for the eager MTP
/// verify forward.
///
/// Identical to `forward_pre_norm_inner` except it records a per-layer
/// [`GdnLayerTape`] for every GDN (`Linear`) layer into `tape`, indexed by
/// ABSOLUTE layer index (`tape[i]` is `Some` for GDN layers, stays `None` for
/// full-attention layers). `tape` is pre-sized to `layers.len()` by the caller.
/// Recording is by lazy `.clone()` (no eval), so it stays inside the fused MLX
/// graph that `eval_step`/`async_eval_layer_caches` materializes.
fn forward_pre_norm_inner_with_tape(
    input_ids: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    tape: &mut [Option<crate::models::qwen3_5::gated_delta_net::GdnLayerTape>],
) -> Result<MxArray> {
    let hidden_states = embedding.forward(input_ids)?;
    let mut h = hidden_states.clone();

    let num_layers = layers.len();
    debug_assert_eq!(
        tape.len(),
        num_layers,
        "forward_pre_norm_inner_with_tape: tape length must equal layer count"
    );
    for i in 0..num_layers {
        let cache = caches.as_mut().map(|c| &mut c[i]);
        let mut slot: Option<crate::models::qwen3_5::gated_delta_net::GdnLayerTape> = None;
        h = layers[i].forward_with_tape(&h, None, cache, None, true, Some(&mut slot))?;
        tape[i] = slot;
    }

    Ok(h)
}

/// Target forward used by the external DFlash2 stepper. Captures post-layer
/// residuals in the companion checkpoint's declared order and optionally
/// records the GDN recurrence tape needed to roll a speculative verify block
/// back to its accepted prefix.
pub(crate) fn forward_dflash2_with_taps(
    inner: &mut Qwen35Inner,
    input_ids: &MxArray,
    tap_layers: &[usize],
    record_tape: bool,
) -> Result<(
    MxArray,
    Vec<MxArray>,
    Vec<Option<crate::models::qwen3_5::gated_delta_net::GdnLayerTape>>,
)> {
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
    let normalized = inner.final_norm.forward(&hidden)?;
    let logits = project_logits_from_hidden(&normalized, &inner.lm_head, &inner.embedding)?;
    Ok((logits, taps, tape))
}

pub(super) fn project_logits_from_hidden(
    hidden: &MxArray,
    lm_head: &Option<LinearProj>,
    embedding: &Embedding,
) -> Result<MxArray> {
    match lm_head {
        Some(head) => head.forward(hidden),
        None => embedding.as_linear(hidden),
    }
}

/// Eager (pure-Rust) MTP verify step.
///
/// Translation of the deleted compiled `forward_mtp_verify_compiled_with_hidden`
/// FFI: runs the `verify_ids` (`[1, K+1]` int32) through the SAME main-model
/// stack the AR path uses (`forward_pre_norm_inner` + `final_norm` +
/// `project_logits_from_hidden`), advancing `inner.caches` by `K+1` positions.
///
/// Returns `MtpVerifyOutput::logits_only(logits, hiddens)` where:
///   * `logits` is `[1, K+1, vocab]` (the verifier target distribution at
///     every verify position),
///   * `hiddens` is `[1, K+1, hidden]` — the post-final-norm hidden at every
///     verify position (the chained-seed and commit context).
///
/// `embedding` owns the lookup and tied-head projection backends. Packed
/// quantized tables therefore stay packed for both operations.
#[allow(clippy::too_many_arguments)]
pub(super) fn eager_verify_step(
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    verify_ids: &MxArray,
    embedding: &Embedding,
    tape: Option<&mut Vec<Option<crate::models::qwen3_5::gated_delta_net::GdnLayerTape>>>,
) -> Result<mtp_decode::MtpVerifyOutput> {
    let pre = match tape {
        Some(tape) => {
            // Record a per-layer GDN tape during the verify forward so the
            // rollback replay can reconstruct the AR-exact carried state.
            tape.clear();
            tape.resize(layers.len(), None);
            forward_pre_norm_inner_with_tape(verify_ids, embedding, layers, caches, tape)?
        }
        None => forward_pre_norm_inner(verify_ids, embedding, layers, caches)?,
    };
    let hiddens = final_norm.forward(&pre)?;
    let logits = project_logits_from_hidden(&hiddens, lm_head, embedding)?;
    Ok(mtp_decode::MtpVerifyOutput::logits_only(logits, hiddens))
}

fn project_last_logits_from_pre_norm_hidden(
    hidden: &MxArray,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embedding: &Embedding,
) -> Result<MxArray> {
    let seq_len = hidden.shape_at(1)?;
    let last_hidden = hidden.slice_axis(1, seq_len - 1, seq_len)?;
    let last_hidden = final_norm.forward(&last_hidden)?;
    let logits = project_logits_from_hidden(&last_hidden, lm_head, embedding)?;
    logits.squeeze(Some(&[1]))
}

/// Partition `total` committed tokens into chunk sizes all within the
/// commit graph's `M in [1, 7]` window.
///
/// Strategy: greedily take size-6 chunks. The final remainder `r` is
/// `total % 6`:
///   - `r == 0`           → all chunks are size 6.
///   - `r >= 2`           → append one chunk of size `r`.
///   - `r == 1`           -> append one chunk of size 1.
///
/// Precondition: `total >= 1`. For `total in {1..7}` the single chunk is
/// `total` itself.
///
/// `pub(crate)`: also used by `MoeMtpStepper::begin_mtp_decode`'s
/// committed-history v2 prompt-prefix seed
/// (`crate::models::qwen3_5_moe::model`), which mirrors this dense chunking.
pub(crate) fn partition_prefill_chunks(total: usize) -> Vec<usize> {
    debug_assert!(total >= 1, "partition_prefill_chunks: total must be >= 1");
    const CHUNK: usize = 6;
    if total == 1 {
        return vec![1];
    }
    if total <= 7 {
        // A single chunk in [1, 7] covers it directly.
        return vec![total];
    }
    let mut chunks: Vec<usize> = Vec::new();
    let mut remaining = total;
    while remaining > 7 {
        chunks.push(CHUNK);
        remaining -= CHUNK;
    }
    // `remaining` is now in [1, 7]. Push it directly.
    debug_assert!(
        (1..=7).contains(&remaining),
        "partition_prefill_chunks: remainder {remaining} out of [1, 7]"
    );
    chunks.push(remaining);
    chunks
}
