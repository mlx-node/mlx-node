//! Shared flat-forward numeric core.
//!
//! The flat-cache families (k2_horizon, lfm2, nemotron_h, muse_glimmer,
//! qwen3, gemma4, qwen3_5 dense/MoE) each hand-copied the same skeleton:
//!
//! ```text
//! ids ──► embed ──► per-layer forward ──► final norm ──► logits head
//!                        │
//!                        └─ chunked prefill: slice axis 1 at a fixed
//!                           stride, cancel-poll at each boundary, eval
//!                           + clear cache between chunks
//! ```
//!
//! plus the matching tails (last-token logits projections), the bulk
//! cache-materialization evals, and the `save_cache_state` history
//! commit (`history = save_tokens ++ trim(generated)` on reuse, else
//! reset). This module holds those pieces once.
//!
//! Every helper preserves the families' exact op order — mask
//! construction, cache-eval timing, slicing/squeeze axes, and the
//! output-multiplier → softcap ordering. Family hooks (muse_glimmer's
//! unscaled RMS embed, nemotron_h's chunk-aligned prefill grid,
//! k2_horizon/qwen3's exact-match rewind) stay in the family files and
//! plug in through the closure parameters.

use std::sync::atomic::{AtomicBool, Ordering};

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::engine::paged_epilogue::{FinalTokenPolicy, save_paged_token_history};
use crate::nn::{Embedding, GroupedRMSNorm, Linear, RMSNorm};
use crate::stream::{Stream, StreamContext};
use crate::transformer::KVCache;

/// Default prefill chunk size (tokens per chunk). Matches Python
/// mlx-lm's `prefill_step_size` default of 2048 — k2_horizon, lfm2,
/// nemotron_h and qwen3_5 share it (muse_glimmer overrides with 512;
/// qwen3's generate threads a config value through).
pub(crate) const PREFILL_STEP_SIZE: i64 = 2048;

/// The cooperative-cancel reason every chunked-prefill boundary emits.
/// One literal so the engine's fail-closed arms and the
/// `err.reason == "prefill cancelled"` tests match on a single string.
pub(crate) const PREFILL_CANCELLED: &str = "prefill cancelled";

// ============================ final norms ============================

/// `forward(&MxArray) -> Result<MxArray>` over the final-norm types the
/// flat-forward families use (`RMSNorm`, `GroupedRMSNorm`).
pub(crate) trait NormForward {
    fn forward(&self, hidden: &MxArray) -> Result<MxArray>;
}

impl NormForward for RMSNorm {
    fn forward(&self, hidden: &MxArray) -> Result<MxArray> {
        RMSNorm::forward(self, hidden)
    }
}

impl NormForward for GroupedRMSNorm {
    fn forward(&self, hidden: &MxArray) -> Result<MxArray> {
        GroupedRMSNorm::forward(self, hidden)
    }
}

// ============================ logits heads ============================

/// An untied logits head — anything with `forward(&MxArray) -> Result`.
/// Implemented for `nn::Linear` and the quantized-or-dense `LinearProj`
/// enums the families ship; tied checkpoints take [`project_logits`]'s
/// `None` arm (`Embedding::as_linear`) instead.
pub(crate) trait LogitsHead {
    fn project(&self, hidden: &MxArray) -> Result<MxArray>;
}

impl LogitsHead for Linear {
    fn project(&self, hidden: &MxArray) -> Result<MxArray> {
        self.forward(hidden)
    }
}

impl LogitsHead for crate::models::quantized_linear::LinearProj {
    fn project(&self, hidden: &MxArray) -> Result<MxArray> {
        self.forward(hidden)
    }
}

impl LogitsHead for crate::models::gemma4::quantized_linear::LinearProj {
    fn project(&self, hidden: &MxArray) -> Result<MxArray> {
        self.forward(hidden)
    }
}

// ============================ cache arrays ============================

/// Per-layer cache exposing its live arrays for a bulk materialization
/// eval — the `collect_arrays` convention every family's cache enum
/// grew. Implemented for the family cache types AND their containers
/// (`Vec`, `Option<Vec>`, slices), so `eval_layer_caches(&self.caches)`
/// works whether `caches` is a `Vec<KVCache>` or an
/// `Option<Vec<Qwen3_5LayerCache>>`.
pub(crate) trait CollectCacheArrays {
    fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>);
}

impl CollectCacheArrays for KVCache {
    fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        if let Some(k) = self.keys_ref() {
            out.push(k);
        }
        if let Some(v) = self.values_ref() {
            out.push(v);
        }
    }
}

impl CollectCacheArrays for crate::models::lfm2::layer_cache::Lfm2LayerCache {
    fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        self.collect_arrays(out);
    }
}

impl CollectCacheArrays for crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache {
    fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        self.collect_arrays(out);
    }
}

impl CollectCacheArrays for crate::models::nemotron_h::layer_cache::NemotronHLayerCache {
    fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        self.collect_arrays(out);
    }
}

impl CollectCacheArrays for crate::models::gemma4::layer_cache::Gemma4LayerCache {
    fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        self.collect_cache_arrays(out);
    }
}

impl<C: CollectCacheArrays> CollectCacheArrays for Vec<C> {
    fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        for cache in self {
            cache.collect_arrays(out);
        }
    }
}

impl<C: CollectCacheArrays> CollectCacheArrays for [C] {
    fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        for cache in self {
            cache.collect_arrays(out);
        }
    }
}

impl<C: CollectCacheArrays> CollectCacheArrays for Option<Vec<C>> {
    fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        if let Some(caches) = self {
            for cache in caches {
                cache.collect_arrays(out);
            }
        }
    }
}

/// Collect every live cache array and materialize them in one eval —
/// the post-prefill/between-chunk barrier every flat family hand-copied
/// (breaks the lazy dependency chains so the next chunk doesn't extend
/// a giant graph rooted at the prior chunk's inputs).
pub(crate) fn eval_layer_caches<C: CollectCacheArrays + ?Sized>(caches: &C) -> Result<()> {
    let mut arrays: Vec<&MxArray> = Vec::new();
    caches.collect_arrays(&mut arrays);
    if !arrays.is_empty() {
        MxArray::eval_arrays(&arrays)?;
    }
    Ok(())
}

/// Async variant of [`eval_layer_caches`]: kicks the GPU materialization
/// but does NOT block the CPU — used between prefill chunks so the CPU
/// can build the next chunk's graph while the previous chunk's cache
/// writes are still in flight.
pub(crate) fn async_eval_layer_caches<C: CollectCacheArrays + ?Sized>(caches: &C) {
    let mut arrays: Vec<&MxArray> = Vec::new();
    caches.collect_arrays(&mut arrays);
    if !arrays.is_empty() {
        MxArray::async_eval_arrays(&arrays);
    }
}

/// The between-chunk barrier the fixed-stride prefills share: bulk
/// cache eval, then `crate::array::clear_cache()` to release the
/// transient allocator state the chunk's graph built.
pub(crate) fn eval_caches_and_clear<C: CollectCacheArrays + ?Sized>(caches: &C) -> Result<()> {
    eval_layer_caches(caches)?;
    crate::array::clear_cache();
    Ok(())
}

// ============================ forward skeleton ============================

/// `embed → per-layer forward` returning the PRE-final-norm hidden
/// `[B, T, hidden]` — the loop every flat family hand-copied.
///
/// `layer_forward` receives the layer, the current hidden, the whole
/// cache container, and the layer index — it owns the family's mask
/// args and cache-slot shape (`Some(&mut caches[i])` vs
/// `caches.as_mut().map(|c| &mut c[i])` vs paged adapters), which is
/// what keeps this generic without a giant trait bound.
pub(crate) fn forward_pre_norm<L, C>(
    input_ids: &MxArray,
    embedding: &Embedding,
    layers: &mut [L],
    caches: &mut C,
    layer_forward: impl FnMut(&mut L, &MxArray, &mut C, usize) -> Result<MxArray>,
) -> Result<MxArray> {
    forward_pre_norm_with(
        input_ids,
        |ids| embedding.forward(ids),
        layers,
        caches,
        layer_forward,
        None,
    )
}

/// [`forward_pre_norm`] parameterized on the embed step and optional
/// post-layer residual taps.
///
/// `embed` lets families keep their own embed op order (muse_glimmer's
/// `rms_norm_unscaled(embedding.forward(ids))`, gemma4's
/// `sqrt(hidden_size)` scaling). `taps = Some((layer_ids, out))` pushes
/// `h.clone()` after every layer whose index is in `layer_ids` —
/// muse_glimmer's `forward_with_taps` capture point (post residual
/// add, pre final norm).
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_pre_norm_with<L, C>(
    input_ids: &MxArray,
    embed: impl FnOnce(&MxArray) -> Result<MxArray>,
    layers: &mut [L],
    caches: &mut C,
    mut layer_forward: impl FnMut(&mut L, &MxArray, &mut C, usize) -> Result<MxArray>,
    taps: Option<(&[usize], &mut Vec<MxArray>)>,
) -> Result<MxArray> {
    let mut h = embed(input_ids)?;
    let (tap_ids, mut tap_out) = taps.unzip();
    for (i, layer) in layers.iter_mut().enumerate() {
        h = layer_forward(layer, &h, caches, i)?;
        if let (Some(ids), Some(out)) = (tap_ids, tap_out.as_mut())
            && ids.contains(&i)
        {
            out.push(h.clone());
        }
    }
    Ok(h)
}

/// `embed → layer loop → final_norm` — the normed-hidden tail the flat
/// forwards share. `layer_forward` is the same per-layer hook as
/// [`forward_pre_norm`].
pub(crate) fn forward_body_normed<L, C, N: NormForward>(
    input_ids: &MxArray,
    embedding: &Embedding,
    layers: &mut [L],
    caches: &mut C,
    final_norm: &N,
    layer_forward: impl FnMut(&mut L, &MxArray, &mut C, usize) -> Result<MxArray>,
) -> Result<MxArray> {
    let h = forward_pre_norm(input_ids, embedding, layers, caches, layer_forward)?;
    final_norm.forward(&h)
}

// ============================ logits projections ============================

/// `head(hidden)` when the family ships an untied logits head, else
/// `embedding.as_linear(hidden)` — the tied-head path (dense and
/// packed-quantized embeddings alike). The exact tail of the
/// lfm2/nemotron_h/muse_glimmer/qwen3_5 flat forwards.
pub(crate) fn project_logits<H: LogitsHead>(
    hidden: &MxArray,
    lm_head: Option<&H>,
    embedding: &Embedding,
) -> Result<MxArray> {
    match lm_head {
        Some(head) => head.project(hidden),
        None => embedding.as_linear(hidden),
    }
}

/// Post-head logit shaping: `output_multiplier` then
/// `cap * tanh(logits / cap)` softcap — muse_glimmer's
/// `output_multiplier`/`final_logit_softcapping` pair, each optional.
/// `None`/`None` is identity.
pub(crate) fn shape_logits(
    logits: &MxArray,
    output_multiplier: Option<f64>,
    final_logit_softcapping: Option<f64>,
) -> Result<MxArray> {
    let mut out = logits.clone();
    if let Some(multiplier) = output_multiplier {
        out = out.mul_scalar(multiplier)?;
    }
    if let Some(cap) = final_logit_softcapping {
        out = out.div_scalar(cap)?.tanh()?.mul_scalar(cap)?;
    }
    Ok(out)
}

/// `norm(hidden[1,T]) → head → keep row T-1 → squeeze [0,1]` → `[vocab]`.
///
/// Projects EVERY row then keeps the last — the qwen3/k2/gemma4
/// paged-tail convention (`head` runs the family's full head pipeline
/// over the normed input: untied/tied projection plus any softcap or
/// shaping).
pub(crate) fn project_last_token_logits<N: NormForward>(
    hidden: &MxArray,
    final_norm: &N,
    head: impl FnOnce(&MxArray) -> Result<MxArray>,
) -> Result<MxArray> {
    let normed = final_norm.forward(hidden)?;
    let logits = head(&normed)?;
    let seq_len = logits.shape_at(1)?;
    logits
        .slice_axis(1, seq_len - 1, seq_len)?
        .squeeze(Some(&[0, 1]))
}

/// `hidden[:, T-1] → norm → head → squeeze [1]` → `[1, vocab]`.
///
/// Slices BEFORE the norm so norm+head run on a single row — the
/// qwen3_5 chunked-prefill tail convention.
pub(crate) fn project_last_hidden_logits<N: NormForward>(
    hidden: &MxArray,
    final_norm: &N,
    head: impl FnOnce(&MxArray) -> Result<MxArray>,
) -> Result<MxArray> {
    let seq_len = hidden.shape_at(1)?;
    let last_hidden = hidden.slice_axis(1, seq_len - 1, seq_len)?;
    let normed = final_norm.forward(&last_hidden)?;
    head(&normed)?.squeeze(Some(&[1]))
}

/// `logits[:, T-1]` kept as `[1, vocab]` — the flat
/// `ChatBackend::prefill` tail in k2/lfm2/nemotron_h (squeeze axis 1
/// only so the shape flows through the shared penalty + sampling
/// pipeline).
pub(crate) fn slice_last_logits_keep_batch(logits: &MxArray) -> Result<MxArray> {
    let seq_len = logits.shape_at(1)?;
    logits
        .slice_axis(1, seq_len - 1, seq_len)?
        .squeeze(Some(&[1]))
}

/// `logits[:, T-1]` squeezed to `[vocab]` — the generate-path tail
/// (`squeeze [0, 1]`, matching [`project_last_token_logits`]'s output
/// convention).
pub(crate) fn slice_last_logits_to_vocab(logits: &MxArray) -> Result<MxArray> {
    let seq_len = logits.shape_at(1)?;
    logits
        .slice_axis(1, seq_len - 1, seq_len)?
        .squeeze(Some(&[0, 1]))
}

// ============================ chunked prefill ============================

/// Fixed-stride chunked prefill driver.
///
/// Polls `turn_cancel` at every looped-chunk boundary and — when
/// `poll_before_final` and at least one looped chunk ran — before the
/// final remainder, so a cancel landing during the last looped chunk
/// still aborts (single-shot prefills stay uncancellable by design;
/// the Err rides the flat engine's `fail_closed_flat_turn` arm — no
/// `save_cache_state`, the session is invalidated, so the
/// partially-advanced caches never become a live prefix).
///
/// `forward(ctx, start, end, is_final)` runs the family's forward over
/// the `[start, end)` token range — the closure owns slicing and stream
/// placement so families whose chunk isn't a plain `prompt.slice_axis(1)`
/// (gemma4's precomputed embeds + PLE slice) still share the loop. Its
/// output is dropped for non-final chunks and returned for the final
/// remainder. `between_chunks(ctx)` is the cache-eval + `clear_cache`
/// barrier after each non-final chunk.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chunked_prefill_ranges<C>(
    ctx: &mut C,
    total_len: i64,
    chunk_size: i64,
    poll_before_final: bool,
    turn_cancel: impl Fn(&C) -> Option<&AtomicBool>,
    mut forward: impl FnMut(&mut C, i64, i64, bool) -> Result<MxArray>,
    mut between_chunks: impl FnMut(&mut C) -> Result<()>,
) -> Result<MxArray> {
    if total_len <= 0 {
        return Err(Error::from_reason("chunked_prefill: empty prompt"));
    }
    let chunk_size = if chunk_size <= 0 {
        total_len
    } else {
        chunk_size
    };
    let mut offset: i64 = 0;
    while total_len - offset > chunk_size {
        if turn_cancel(&*ctx).is_some_and(|f| f.load(Ordering::Relaxed)) {
            return Err(Error::from_reason(PREFILL_CANCELLED));
        }
        let _ = forward(ctx, offset, offset + chunk_size, false)?;
        between_chunks(ctx)?;
        offset += chunk_size;
    }
    if poll_before_final
        && offset > 0
        && turn_cancel(&*ctx).is_some_and(|f| f.load(Ordering::Relaxed))
    {
        return Err(Error::from_reason(PREFILL_CANCELLED));
    }
    forward(ctx, offset, total_len, true)
}

/// [`chunked_prefill_ranges`] for the common `[1, T] ->
/// slice_axis(1, s, e)` chunk shape: slices `prompt` on the host stream,
/// then wraps `forward` in `generation_stream`'s `StreamContext` —
/// the exact ordering every family's loop used.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chunked_prefill<C>(
    ctx: &mut C,
    prompt: &MxArray,
    generation_stream: Stream,
    chunk_size: i64,
    poll_before_final: bool,
    turn_cancel: impl Fn(&C) -> Option<&AtomicBool>,
    mut forward: impl FnMut(&mut C, &MxArray, bool) -> Result<MxArray>,
    between_chunks: impl FnMut(&mut C) -> Result<()>,
) -> Result<MxArray> {
    chunked_prefill_ranges(
        ctx,
        prompt.shape_at(1)?,
        chunk_size,
        poll_before_final,
        turn_cancel,
        |ctx, start, end, is_final| {
            let chunk = prompt.slice_axis(1, start, end)?;
            let _stream_ctx = StreamContext::new(generation_stream);
            forward(ctx, &chunk, is_final)
        },
        between_chunks,
    )
}

/// Explicit-boundary chunked prefill: forwards `slices` in order,
/// polling `turn_cancel` before EVERY slice — nemotron_h's
/// chunk-aligned grid (the Mamba-2 chunk scan pads the last chunk of
/// each forward, so intermediate boundaries must land on the configured
/// chunk grid or the recurrence's reduction grouping changes).
///
/// Returns the LAST slice's `forward` output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chunked_prefill_slices<C>(
    ctx: &mut C,
    prompt: &MxArray,
    slices: &[(i64, i64)],
    generation_stream: Stream,
    turn_cancel: impl Fn(&C) -> Option<&AtomicBool>,
    mut forward: impl FnMut(&mut C, &MxArray) -> Result<MxArray>,
    mut between_chunks: impl FnMut(&mut C) -> Result<()>,
) -> Result<MxArray> {
    let last_idx = slices.len().saturating_sub(1);
    let mut last = None;
    for (idx, &(start, end)) in slices.iter().enumerate() {
        if turn_cancel(&*ctx).is_some_and(|f| f.load(Ordering::Relaxed)) {
            return Err(Error::from_reason(PREFILL_CANCELLED));
        }
        let chunk = prompt.slice_axis(1, start, end)?;
        last = Some({
            let _stream_ctx = StreamContext::new(generation_stream);
            forward(ctx, &chunk)?
        });
        if idx != last_idx {
            between_chunks(ctx)?;
        }
    }
    last.ok_or_else(|| Error::from_reason("chunked_prefill produced no chunks"))
}

// ============================ history save ============================

/// The flat-path `save_cache_state` history arithmetic every flat
/// family inlined (`reuse → history = save_tokens ++ trim(generated)`;
/// the `else` reset stays family-side). Delegates to the paged
/// epilogue's `save_paged_token_history` so the flat and paged saves
/// can never disagree: `policy` + `keep_all` are the same
/// `FinalTokenPolicy` polarity the paged save uses — pure-KV stacks
/// pass `KeepAllOnLength` with `keep_all = last_token_in_cache`,
/// conv/GDN/mamba stacks compute `keep_all` themselves (nemotron_h's
/// MTP arm) or pass `AlwaysDrop`.
///
/// Returns `true` on the reuse arm (history written); `false` means
/// the caller runs its own reset path (the helper already cleared
/// `history`, mirroring `save_paged_token_history`).
pub(crate) fn save_flat_token_history(
    save_tokens: &[u32],
    generated_tokens: &[u32],
    keep_all: bool,
    reuse_cache: bool,
    policy: FinalTokenPolicy,
    cached_token_history: &mut Vec<u32>,
) -> bool {
    save_paged_token_history(
        save_tokens,
        generated_tokens,
        keep_all,
        reuse_cache,
        policy,
        cached_token_history,
    );
    reuse_cache
}
