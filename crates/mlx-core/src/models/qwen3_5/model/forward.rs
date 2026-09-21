//! Shared prefill/forward numeric core.

use super::*;

use crate::models::forward as fwd;

/// Default prefill chunk size (tokens per chunk).
/// Matches Python mlx-lm's `prefill_step_size` default of 2048.
pub(crate) const PREFILL_STEP_SIZE: i64 = 2048;

/// High tag on compiled-verify `fn_id`s. The low byte carries the verify
/// length; bits 8..31 carry the per-instance `model_id`. `Qwen35Inner`'s Drop
/// erases `COMPILED_VERIFY_TAG | model_id << 8` under mask `...FF00` so an
/// unload releases the tapes' captured weights.
pub(crate) const COMPILED_VERIFY_TAG: u64 = 0xD51A_3500_0000_0000;

/// Evaluate all cache arrays across all layers to materialize them on GPU.
/// Must be called between prefill chunks to break lazy dependency chains.
pub(crate) fn eval_layer_caches(caches: &Option<Vec<Qwen3_5LayerCache>>) -> Result<()> {
    fwd::eval_layer_caches(caches)
}

/// Async variant of `eval_layer_caches`: kicks GPU on cache materialization
/// but does NOT block the CPU. Used between prefill chunks so the CPU can
/// start building the next chunk's graph while the previous chunk's cache
/// writes are still in flight.
pub(crate) fn async_eval_layer_caches(caches: &Option<Vec<Qwen3_5LayerCache>>) {
    fwd::async_eval_layer_caches(caches);
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
    // `MLX_PREFILL_SYNC_BETWEEN_CHUNKS` forces synchronous `eval_layer_caches`
    // between chunks instead of the async default.
    let chunk_async = std::env::var("MLX_PREFILL_SYNC_BETWEEN_CHUNKS").is_err();
    let mut ctx = (embedding, layers, caches, final_norm, lm_head);
    fwd::chunked_prefill(
        &mut ctx,
        prompt,
        generation_stream,
        chunk_size,
        true,
        move |_| turn_cancel,
        |ctx, chunk, is_final| {
            let hidden = forward_pre_norm_inner(chunk, ctx.0, ctx.1, ctx.2)?;
            if is_final {
                project_last_logits_from_pre_norm_hidden(&hidden, ctx.3, ctx.4, ctx.0)
            } else {
                Ok(hidden)
            }
        },
        |ctx| {
            if chunk_async {
                fwd::async_eval_layer_caches(&*ctx.2);
            } else {
                fwd::eval_layer_caches(&*ctx.2)?;
            }
            crate::array::clear_cache();
            Ok(())
        },
    )
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
        // Cooperative-cancel checkpoint: abort at the chunk boundary,
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

pub(super) fn forward_pre_norm_inner(
    input_ids: &MxArray,
    embedding: &Embedding,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
) -> Result<MxArray> {
    let num_layers = layers.len();
    // Plain layer loop.
    //
    // This is the SHARED pre-norm primitive: it MUST return the full
    // per-position hidden. The MTP prompt-hidden path
    // (`chunked_prefill_with_hidden_with_size`) keeps the result and
    // re-slices it by chunk length, so a last-token slice here would
    // corrupt it. The logits-only callers get the equivalent of the
    // upstream last-token optimization from
    // `project_last_logits_from_pre_norm_hidden` (which slices before
    // `final_norm` + `lm_head`), so the slice deliberately does NOT
    // live in this loop.
    fwd::forward_pre_norm_with(
        input_ids,
        |ids| {
            let h = embedding.forward(ids)?;
            debug!(
                "Qwen3.5 forward_inner: input_ids_shape={} post_embed_shape={}",
                shape_dbg(input_ids),
                shape_dbg(&h),
            );
            Ok(h)
        },
        layers,
        caches,
        |layer, h, caches, i| {
            let out = layer.forward(h, None, caches.as_mut().map(|c| &mut c[i]), None, true)?;
            if i == 0 || i + 1 == num_layers {
                debug!(
                    "Qwen3.5 forward_inner: post_layer[{}/{}] shape={}",
                    i,
                    num_layers,
                    shape_dbg(&out),
                );
            }
            Ok(out)
        },
        None,
    )
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

/// Target forward used by the external DFlash2 stepper. Captures post-layer
/// residuals in the companion checkpoint's declared order and optionally
/// records the GDN recurrence tape needed to roll a speculative verify block
/// back to its accepted prefix.
pub(crate) fn forward_dflash2_with_taps(
    inner: &mut Qwen35Inner,
    input_ids: &MxArray,
    tap_layers: &[usize],
    record_tape: bool,
    logits_span: DFlash2LogitsSpan,
) -> Result<(
    MxArray,
    Vec<MxArray>,
    Vec<Option<crate::models::qwen3_5::gated_delta_net::GdnLayerTape>>,
)> {
    // The compiled verify path replays the whole decode forward as one fused
    // MLX tape — see `forward_dflash2_compiled` for the contract. It only
    // covers the speculative-verify shape (B=1, tape recorded, all-rows
    // logits); anything else stays on the eager path below.
    if record_tape && matches!(logits_span, DFlash2LogitsSpan::All) {
        match forward_dflash2_compiled(inner, input_ids, tap_layers) {
            Ok(Some(out)) => return Ok(out),
            Ok(None) => {}
            Err(e) => {
                // Trace/replay failures leave the caches untouched (all cache
                // writes happen after a successful invoke), so the eager path
                // is a safe retry — but a builder-level failure is structural,
                // not transient, so disable the compiled path for this model.
                inner.dflash2_compiled_verify_disabled = true;
                eprintln!("[dflash2] compiled verify disabled: {e}");
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
    let layer_probe = std::env::var("MLX_DFLASH2_VERIFY_LAYERS").is_ok();
    let mut probe_t = std::time::Instant::now();
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
        if layer_probe && (index + 1) % 15 == 0 {
            MxArray::eval_arrays(&[&hidden])?;
            let now = std::time::Instant::now();
            eprintln!(
                "[verify-layers] layers {:>2}-{:>2}: {:?}",
                index + 1 - 14,
                index + 1,
                now.duration_since(probe_t)
            );
            probe_t = now;
        }
    }
    let _ = probe_t;
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
/// Outputs, in order:
///   `logits` `[1, L, V]`, tap hiddens (in `tap_layers` order), then per layer:
///     Linear        — kernel tape `q, k, v, g, beta` + post-mask `qkv` (6)
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
/// After a successful invoke the caller writes each emitted `(new_k, new_v)`
/// into the real `KVCache` via `update_and_fetch` — the same grow +
/// in-place-write + offset-bump semantics the eager path uses — so commit's
/// `trim(snapshot_offset + steps)` rollback is unchanged.
///
/// Returns `Ok(None)` when the caches are not in the flat-verify shape the
/// graph expects (missing state slots, variant mismatch, divergent
/// full-attention offsets) or the FFI invoke fails — the caller falls back
/// to the eager path. Builder errors surface as `Err`.
fn forward_dflash2_compiled(
    inner: &mut Qwen35Inner,
    input_ids: &MxArray,
    tap_layers: &[usize],
) -> Result<
    Option<(
        MxArray,
        Vec<MxArray>,
        Vec<Option<crate::models::qwen3_5::gated_delta_net::GdnLayerTape>>,
    )>,
> {
    use crate::models::qwen3_5::decoder_layer::LayerVerifyIo;
    use crate::models::qwen3_5::gated_delta::GdnKernelTape;
    use crate::models::qwen3_5::gated_delta_net::GdnLayerTape;

    if inner.dflash2_compiled_verify_disabled
        || std::env::var("MLX_DISABLE_DFLASH2_COMPILED_VERIFY").is_ok()
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
    // `flat_attention_frontier`) is that every FA cache agrees on it.
    let mut rope_base: Option<i32> = None;
    let mut per_layer_state: Vec<MxArray> = Vec::with_capacity(2 * caches.len());
    let mut fa_prefix_offsets: Vec<Option<i32>> = Vec::with_capacity(caches.len());
    for (layer, cache) in inner.layers.iter().zip(caches.iter()) {
        match (&layer.attn, cache) {
            (
                crate::models::qwen3_5::decoder_layer::AttentionType::Linear(_),
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
                crate::models::qwen3_5::decoder_layer::AttentionType::Full(_),
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
                per_layer_state.push(keys.slice_axis(2, 0, offset as i64)?);
                per_layer_state.push(values.slice_axis(2, 0, offset as i64)?);
                fa_prefix_offsets.push(Some(offset));
            }
            _ => return Ok(None),
        }
    }
    let rope_offsets = MxArray::from_int32(&[rope_base.unwrap_or(0)], &[1])?;
    let mut inputs: Vec<MxArray> = Vec::with_capacity(2 + per_layer_state.len());
    inputs.push(input_ids.clone());
    inputs.push(rope_offsets);
    inputs.extend(per_layer_state);

    let n_linear = inner.layers.iter().filter(|l| l.is_linear()).count();
    let n_fa = inner.layers.len() - n_linear;
    let n_outputs = 1 + tap_layers.len() + 6 * n_linear + 2 * n_fa;
    // Per-(model, L) id: the verify length decides every host-int branch in
    // the builder (SDPA verify-split geometry), while the prefix length stays
    // shapeless. The high tag namespaces these ids away from the C++-side
    // pointer-derived ids and the test range.
    let fn_id =
        COMPILED_VERIFY_TAG | ((inner.model_id & 0x00FF_FFFF) << 8) | (seq_len as u64 & 0xFF);

    let layers = &mut inner.layers;
    let embedding = &inner.embedding;
    let final_norm = &inner.final_norm;
    let lm_head = &inner.lm_head;
    let mut builder = move |graph_inputs: &[MxArray]| -> Result<Vec<MxArray>> {
        let ids = &graph_inputs[0];
        let rope_offsets = &graph_inputs[1];
        let mut cursor = 2usize;
        let mut hidden = embedding.forward(ids)?;
        let mut taps: Vec<Option<MxArray>> = vec![None; tap_layers.len()];
        let mut extras: Vec<MxArray> = Vec::with_capacity(6 * n_linear + 2 * n_fa);
        for (index, layer) in layers.iter_mut().enumerate() {
            let mut tape_slot: Option<GdnLayerTape> = None;
            if layer.is_linear() {
                // Detached cache seeded with the graph-input states: the
                // verify-time writes land in it and are dropped; commit
                // replays the accepted prefix from the snapshot.
                let mut detached = crate::models::qwen3_5::arrays_cache::ArraysCache::new(2);
                detached.set(0, graph_inputs[cursor].clone())?;
                detached.set(1, graph_inputs[cursor + 1].clone())?;
                let mut io = LayerVerifyIo::Linear(&mut detached);
                hidden = layer.forward_verify(&hidden, &mut io, true, Some(&mut tape_slot))?;
                let tape = tape_slot.ok_or_else(|| {
                    Error::from_reason("compiled verify: GDN layer produced no tape")
                })?;
                let GdnLayerTape { kernel, qkv, .. } = tape;
                let GdnKernelTape { q, k, v, g, beta } = kernel;
                extras.extend([q, k, v, g, beta, qkv]);
            } else {
                let mut out_kv = None;
                {
                    let mut io = LayerVerifyIo::FullAttention(
                        crate::models::qwen3_5::attention::AttentionVerifyIo {
                            prefix_keys: &graph_inputs[cursor],
                            prefix_values: &graph_inputs[cursor + 1],
                            rope_offsets,
                            out_kv: &mut out_kv,
                        },
                    );
                    hidden = layer.forward_verify(&hidden, &mut io, true, Some(&mut tape_slot))?;
                }
                let (new_k, new_v) = out_kv.ok_or_else(|| {
                    Error::from_reason("compiled verify: attention layer produced no kv block")
                })?;
                extras.push(new_k);
                extras.push(new_v);
            }
            cursor += 2;
            for (slot, &tap_layer) in tap_layers.iter().enumerate() {
                if tap_layer == index {
                    taps[slot] = Some(hidden.clone());
                }
            }
        }
        let normalized = final_norm.forward(&hidden)?;
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
    // Any failure after the invoke must leave the live FA caches at their
    // prefix offsets — the caller falls back to the eager forward, which
    // appends through `update_and_fetch` again.
    let unpack = |outputs: &[MxArray],
                  caches: &mut [Qwen3_5LayerCache],
                  tape: &mut [Option<GdnLayerTape>]|
     -> Result<()> {
        let mut cursor = 1 + tap_layers.len();
        for (index, layer) in inner.layers.iter().enumerate() {
            if layer.is_linear() {
                let kd = match &layer.attn {
                    crate::models::qwen3_5::decoder_layer::AttentionType::Linear(gdn) => {
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
                });
            } else {
                let kvc = caches[index].as_kv_cache_mut().ok_or_else(|| {
                    Error::from_reason("compiled verify: attention cache kind mismatch")
                })?;
                let (Some(new_k), Some(new_v)) = (outputs.get(cursor), outputs.get(cursor + 1))
                else {
                    return Err(Error::from_reason(
                        "compiled verify: output arity shortfall",
                    ));
                };
                cursor += 2;
                kvc.update_and_fetch(new_k, new_v)?;
            }
        }
        Ok(())
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
    Ok(Some((logits, taps, tape)))
}

pub(super) fn project_logits_from_hidden(
    hidden: &MxArray,
    lm_head: &Option<LinearProj>,
    embedding: &Embedding,
) -> Result<MxArray> {
    fwd::project_logits(hidden, lm_head.as_ref(), embedding)
}

/// Eager (pure-Rust) MTP verify step.
///
/// Runs the `verify_ids` (`[1, K+1]` int32) through the SAME main-model
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
    fwd::project_last_hidden_logits(hidden, final_norm, |h| {
        fwd::project_logits(h, lm_head.as_ref(), embedding)
    })
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
