//! Shared prefill/forward numeric core.

use super::*;

use crate::models::forward as fwd;

/// Default prefill chunk size (tokens per chunk).
/// Matches Python mlx-lm's `prefill_step_size` default of 2048.
pub(crate) const PREFILL_STEP_SIZE: i64 = 2048;

/// Metal allocator pool size above which a decide drains the free pool.
/// Below this the pool's own reuse is cheaper than a sync+wipe, and the
/// CacheLimitCoordinator cap already bounds steady-state growth.
const CLEF_CLEAR_CACHE_THRESHOLD_BYTES: u64 = 512 * 1024 * 1024;

/// High tag on the compiled CLEF decide-backbone `fn_id`: bits 8..31 carry
/// the per-instance `model_id`, the rest of the id is fixed — `shapeless =
/// false` puts each prompt length on its own tape inside the entry's compile
/// cache. `Qwen35Inner`'s Drop erases `COMPILED_DECIDE_TAG | model_id << 8`
/// under `...FFFF_FF00` so an unload releases the captured weights.
pub(crate) const COMPILED_DECIDE_TAG: u64 = 0xC1EF_3500_0000_0000;
pub(crate) const COMPILED_DECIDE_ERASE_MASK: u64 = 0xFFFF_FFFF_FFFF_FF00;

/// Cap on distinct prompt lengths a model may give a compiled decide tape.
/// `shapeless = false` keeps one tape per T in the process-global compile
/// cache with no eviction; beyond this many shapes a new length runs the
/// eager backbone instead of growing the registry without bound. Decide
/// prompts re-use a small set of lengths, so the cap leaves every realistic
/// workload compiled while pinning worst-case tape memory.
const CLEF_COMPILED_DECIDE_MAX_TAPES: usize = 32;

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
            fwd::async_eval_layer_caches(&*ctx.2);
            crate::array::clear_cache();
            Ok(fwd::ChunkBarrier::Submitted)
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
        crate::engine::progress::with_turn_progress(|p| p.prefill_advanced(end as u32));
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

impl Qwen35Inner {
    /// Compiled decide backbone: embedding -> all layers -> `final_norm` as
    /// ONE traced graph replayed per call. Graph inputs are `[ids, zero
    /// conv-history, zero recurrent-state]` — each linear layer seeds a
    /// detached `ArraysCache` with the two zeros, so the per-layer zero-fills
    /// stay outside the tape, and full-attention layers run `cache = None`
    /// (value-identical to the fresh KVCache the eager path allocates per
    /// decide). `shapeless = false`: the prompt length keys the inner
    /// compile cache, so each distinct T traces once (~5-15 ms) and replays
    /// thereafter.
    ///
    /// Returns `Ok(None)` when the compiled path is latched off (a previous
    /// trace/replay failed), this prompt length exceeds the tape cap, or the
    /// FFI invoke declines — the caller falls back to the eager layer loop.
    /// Builder errors also latch the model off: a trace failure is
    /// structural and the failed trace keeps its registry entry.
    fn decide_backbone_compiled(&mut self, ids: &MxArray) -> Result<Option<MxArray>> {
        if self.clef_compiled_decide_disabled || std::env::var_os("MLX_CLEF_NO_COMPILED").is_some()
        {
            return Ok(None);
        }
        let prompt_len = ids.shape_at(1)?;
        if self.clef_compiled_decide_shapes.len() >= CLEF_COMPILED_DECIDE_MAX_TAPES
            && !self.clef_compiled_decide_shapes.contains(&prompt_len)
        {
            return Ok(None);
        }
        self.clef_compiled_decide_shapes.insert(prompt_len);
        let model_id = self.model_id;
        let conv_zero = MxArray::zeros(
            &[
                1,
                (self.config.linear_conv_kernel_dim - 1) as i64,
                self.config.linear_conv_dim() as i64,
            ],
            Some(crate::array::DType::BFloat16),
        )?;
        let state_zero = MxArray::zeros(
            &[
                1,
                self.config.linear_num_value_heads as i64,
                self.config.linear_value_head_dim as i64,
                self.config.linear_key_head_dim as i64,
            ],
            Some(crate::array::DType::Float32),
        )?;
        MxArray::eval_arrays(&[&conv_zero, &state_zero])?;

        // The embedding lookup runs OUTSIDE the tape: on packed (GGUF)
        // checkpoints its gather+dequantize chain replays badly inside the
        // compiled graph (measured +45ms/replay on the affine Q8_0 table),
        // while eagerly it is sub-ms. The hidden row block becomes a graph
        // input instead of the token ids.
        let mut hidden_in = self.embedding.forward(ids)?;
        if hidden_in.dtype()? != crate::array::DType::BFloat16 {
            hidden_in = hidden_in.astype(crate::array::DType::BFloat16)?;
        }

        let layers = &mut self.layers;
        let final_norm = &self.final_norm;
        let mut builder = move |graph_inputs: &[MxArray]| -> Result<Vec<MxArray>> {
            let mut hidden = graph_inputs[0].clone();
            for layer in layers.iter_mut() {
                let mut cache_slot = if layer.is_linear() {
                    let mut detached = crate::models::qwen3_5::arrays_cache::ArraysCache::new(2);
                    detached.set(0, graph_inputs[1].clone())?;
                    detached.set(1, graph_inputs[2].clone())?;
                    Some(Qwen3_5LayerCache::Linear(detached))
                } else {
                    None
                };
                hidden = layer.forward(&hidden, None, cache_slot.as_mut(), None, true)?;
            }
            Ok(vec![final_norm.forward(&hidden)?])
        };

        let fn_id = COMPILED_DECIDE_TAG | ((model_id & 0x00FF_FFFF) << 8);
        match crate::compiled_graph::invoke_compiled_graph(
            fn_id,
            &[&hidden_in, &conv_zero, &state_zero],
            1,
            false,
            &mut builder,
        ) {
            Ok(Some(mut outputs)) => Ok(Some(outputs.remove(0))),
            Ok(None) => Ok(None),
            Err(e) => {
                eprintln!("[clef] compiled backbone failed, latching eager: {e}");
                self.clef_compiled_decide_disabled = true;
                Ok(None)
            }
        }
    }

    pub(crate) fn decide_clef(
        &mut self,
        request: &crate::models::clef::encoding::Request,
        cancelled: &AtomicBool,
    ) -> Result<String> {
        use crate::models::clef::{check_cancel, encoding};
        let timing = std::env::var_os("CLEF_DEBUG_TIMING").is_some();
        let phase_start = std::time::Instant::now();
        check_cancel(cancelled)?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("CLEF tokenizer is missing"))?;
        let record = encoding::encode(
            request,
            |frags| tokenizer.encode_batch_sync(frags, Some(false)),
            16384.min(self.config.max_position_embeddings as usize),
        )?;
        if timing {
            eprintln!(
                "[clef] encode: {:?} ({} tokens)",
                phase_start.elapsed(),
                record.input_ids.len()
            );
        }
        // One unpadded sequence; all recurrent and attention state belongs to
        // this request. Nothing can be reused by the next decision, even on error.
        let ids = MxArray::from_uint32(&record.input_ids, &[1, record.input_ids.len() as i64])?;
        // Inference stream, matching the chat prefill's StreamContext. The
        // compiled graph keys on the default stream — it must be active
        // before the first invoke traces.
        let _stream_ctx = crate::stream::StreamContext::new(crate::stream::Stream::generation());
        // The compiled replay cannot check the flag between layers (the
        // whole backbone is one tape), so gate the one point we still
        // control before entering it.
        check_cancel(cancelled)?;
        let mut compiled_backbone = false;
        let hidden = match self.decide_backbone_compiled(&ids)? {
            Some(hidden) => {
                compiled_backbone = true;
                hidden
            }
            None => {
                let mut caches = Some(fresh_dense_layer_caches(&self.config));
                let mut hidden = self.embedding.forward(&ids)?;
                // A packed affine (GGUF) embedding dequantizes to f16; the
                // first bf16 norm would promote the whole residual stream to
                // f32 and route every affine QMM to its tf32-rate kernel.
                // Pin the bf16 activation contract the dense checkpoint
                // computes in; the clef affine sidecars are also loaded bf16
                // (persistence) so the bf16 nax kernels serve the prefill.
                if hidden.dtype()? != crate::array::DType::BFloat16 {
                    hidden = hidden.astype(crate::array::DType::BFloat16)?;
                }
                // eval() drains the GPU pipeline synchronously, so chunked
                // evals serialize each 4-layer block behind a full drain
                // plus the next block's CPU graph build while the GPU idles
                // — measured ~x1.4-1.8 slower than a single eval at decide
                // sizes. Sync every four layers only for long prompts, where
                // an unevaluated whole-backbone graph would hold tens of GB
                // of lazy intermediates.
                let layer_count = self.layers.len();
                let long_prompt = ids.shape_at(1)? > 4096;
                for (i, layer) in self.layers.iter_mut().enumerate() {
                    check_cancel(cancelled)?;
                    let layer_start = timing.then(std::time::Instant::now);
                    hidden = layer.forward(
                        &hidden,
                        None,
                        caches.as_mut().map(|c| &mut c[i]),
                        None,
                        true,
                    )?;
                    if i + 1 == layer_count || (long_prompt && (i + 1) % 4 == 0) {
                        MxArray::eval_arrays(&[&hidden])?;
                    }
                    if let Some(t) = layer_start {
                        eprintln!(
                            "[clef] layer {i} ({}) build+eval: {:?}",
                            if layer.is_linear() { "gdn" } else { "full" },
                            t.elapsed()
                        );
                    }
                }
                self.final_norm.forward(&hidden)?
            }
        };
        check_cancel(cancelled)?;
        MxArray::eval_arrays(&[&hidden])?;
        if timing {
            eprintln!(
                "[clef] backbone: {:?} total ({})",
                phase_start.elapsed(),
                if compiled_backbone {
                    "compiled"
                } else {
                    "eager"
                }
            );
        }
        // The joint head gathers only option-token rows of the output table;
        // on a packed (GGUF-imported) lm_head `take_dequantized_rows`
        // dequantizes just those rows instead of the whole `[vocab, hidden]`
        // table per request. Fetched after the backbone: the compiled path
        // needs &mut self, so these borrows cannot outlive it.
        let head = self
            .clef_head
            .as_ref()
            .ok_or_else(|| Error::from_reason("CLEF head is not loaded"))?;
        let output = self
            .lm_head
            .as_ref()
            .ok_or_else(|| Error::from_reason("CLEF output embeddings are missing"))?;
        let head_start = std::time::Instant::now();
        let probabilities = head.forward(&hidden, &record, output)?;
        if timing {
            eprintln!("[clef] head: {:?}", head_start.elapsed());
        }
        // Per-decide clear_cache was a 4-6ms GPU sync + pool drain that only
        // released freed transients: layer caches are already rebuilt per
        // decide and clef registers a CacheLimitCoordinator cap that
        // auto-evicts the Metal pool. Keep a size gate for hosts where the
        // coordinator cap is loose.
        let cc_start = std::time::Instant::now();
        let cache_bytes = crate::array::get_cache_memory();
        if cache_bytes > CLEF_CLEAR_CACHE_THRESHOLD_BYTES as f64 {
            crate::array::clear_cache();
            if timing {
                eprintln!(
                    "[clef] clear_cache: {:?} (pool {cache_bytes:.0}B > threshold)",
                    cc_start.elapsed()
                );
            }
        }
        check_cancel(cancelled)?;
        let rows: Vec<_> = record.questions.iter().zip(probabilities).map(|(q,p)| serde_json::json!({"id":q.id,"type":q.kind,"options":q.option_ids,"probabilities":p})).collect();
        serde_json::to_string(
            &serde_json::json!({"questions":rows,"input_tokens":record.input_ids.len()}),
        )
        .map_err(|e| Error::from_reason(e.to_string()))
    }
}
