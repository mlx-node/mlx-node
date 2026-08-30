//! Dtype-agnostic forward core: PLE, the prefill/decode forward bodies, sampling, and the KV-layout derivation.

use super::*;

/// PLE (Per-Layer Embeddings) model-level components.
///
/// Provides per-layer token-level information to each decoder layer.
/// Present in E2B (2.3B) and E4B (4.5B) models.
pub(crate) struct PleComponents {
    /// Embedding table: [vocab_size_per_layer_input, num_layers * ple_dim]
    pub embed_tokens_per_layer: Embedding,
    /// Projection: [hidden_size, num_layers * ple_dim]
    pub per_layer_model_projection: LinearProj,
    /// Norm applied per ple_dim slice: weight shape [ple_dim]
    pub per_layer_projection_norm: RMSNorm,
    /// Scale factor: 2.0^(-0.5) = 1/sqrt(2) for per_layer_input_scale
    pub per_layer_input_scale: f64,
    /// Scale factor: hidden_size^(-0.5) for per_layer_model_projection_scale
    pub per_layer_model_projection_scale: f64,
    /// Dimension of per-layer embeddings
    pub ple_dim: i32,
    /// Number of layers
    pub num_layers: i32,
    /// PLE vocab size (may be smaller than main vocab_size)
    pub vocab_size_per_layer_input: i32,
}

/// How many layers to batch per eval during warmup.
///
/// Larger GPUs can handle bigger Metal command buffers before timing out,
/// but the timeout is nondeterministic (thermal state, system load).
/// Uses `max_recommended_working_set_size` (GPU memory) as proxy:
///   ≤128 GB → 1  (base / Pro / Max)
///   ≤384 GB → 2  (Ultra variants)
///   >384 GB → 4  (future hardware)
fn warmup_layer_batch_size() -> usize {
    let gb = crate::stream::WiredLimitContext::get_max_working_set_size() / (1 << 30);
    match gb {
        0..=128 => 1,
        129..=384 => 2,
        _ => 4,
    }
}

/// Single-token forward pass to trigger Metal shader compilation at load time.
/// Layers are eval'd in batches (sized by GPU capability) to keep Metal
/// command buffers under the timeout limit on cold shader cache.
pub(crate) fn warmup_forward(inner: &Gemma4Inner) -> Result<()> {
    let config = &inner.config;
    let batch = warmup_layer_batch_size();
    let mem_before = crate::array::get_active_memory();
    info!(
        "[warmup] layer batch size: {} (GPU mem: query complete)",
        batch
    );

    {
        let mut caches = init_caches_for_config(config);
        let dummy = MxArray::from_int32(&[1i32], &[1, 1])?;

        let mut h = inner.embed_tokens.forward(&dummy)?;
        h = h.mul_scalar((config.hidden_size as f64).sqrt())?;
        h.eval();

        for (i, layer) in inner.layers.iter().enumerate() {
            h = layer.forward(&h, None, Some(&mut caches[i]), None, false)?;
            if (i + 1) % batch == 0 || i + 1 == inner.layers.len() {
                h.eval();
            }
        }

        h = inner.final_norm.forward(&h)?;
        let logits = if let Some(ref head) = inner.lm_head {
            head.forward(&h)?
        } else if inner.embed_tokens.is_packed_quantized() {
            // Packed tied lm_head: project through the quantized matmul without
            // materializing the dense table.
            inner.embed_tokens.as_linear(&h)?
        } else if let Some(ref w_t) = inner.embed_weight_t {
            h.matmul(w_t)?
        } else {
            let weight = inner.embed_tokens.get_weight();
            let weight_t = weight.transpose(Some(&[1, 0]))?;
            h.matmul(&weight_t)?
        };
        logits.eval();
    }

    crate::array::synchronize_and_clear_cache();
    let mem_after = crate::array::get_active_memory();
    info!(
        "[warmup] memory: {:.2} GB → {:.2} GB (delta: {:.2} GB)",
        mem_before / 1e9,
        mem_after / 1e9,
        (mem_after - mem_before) / 1e9
    );

    Ok(())
}

/// Build throwaway KV caches for a Gemma4 config.
///
/// Used by `warmup_forward` to run a single dummy token through the
/// full layer stack at load time (triggering Metal shader compilation)
/// without touching the persistent `self.caches` on `Gemma4Inner`. The
/// persistent path initializes its caches via `init_caches_sync` from
/// the engine's miss-path `reset_caches(ResetScope::PrefixMiss)` (or
/// defensively inside `ChatBackend::prefill` / the vision cores).
pub(super) fn init_caches_for_config(config: &Gemma4Config) -> Vec<Gemma4LayerCache> {
    let num_layers = config.num_hidden_layers as usize;
    let mut caches = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        if config.is_global_layer(i) {
            caches.push(Gemma4LayerCache::new_global());
        } else {
            caches.push(Gemma4LayerCache::new_sliding(config.sliding_window));
        }
    }
    caches
}

/// Check whether `token` should terminate decoding.
///
/// The config-level `eos_token_ids` are always honored. The caller-supplied
/// `eos_token_id` is treated as an additional stop token — it does NOT
/// replace the config list. Session-start callers get their clean boundary
/// token (for Gemma4 that is `<turn|>`) while still respecting the
/// underlying model's intrinsic eos set.
#[inline]
pub(super) fn is_eos_token(token: u32, eos_ids: &[i32], eos_token_id: u32) -> bool {
    if eos_ids.contains(&(token as i32)) {
        return true;
    }
    eos_token_id == token
}

#[derive(Clone, Copy)]
pub(super) struct Gemma4RepetitionCutoff {
    max_consecutive_tokens: i32,
    max_ngram_repeats: i32,
    ngram_size: i32,
}

pub(super) fn repetition_cutoff_from_config(config: &ChatConfig) -> Gemma4RepetitionCutoff {
    Gemma4RepetitionCutoff {
        max_consecutive_tokens: config
            .max_consecutive_tokens
            .unwrap_or(crate::sampling::DEFAULT_MAX_CONSECUTIVE_TOKENS),
        max_ngram_repeats: config
            .max_ngram_repeats
            .unwrap_or(crate::sampling::DEFAULT_MAX_NGRAM_REPEATS),
        ngram_size: config
            .ngram_size
            .unwrap_or(crate::sampling::DEFAULT_NGRAM_SIZE),
    }
}

pub(super) fn check_gemma4_repetition_cutoff(
    generated_tokens: &[u32],
    cutoff: Gemma4RepetitionCutoff,
) -> Option<&'static str> {
    crate::sampling::check_repetition_cutoff(
        generated_tokens,
        cutoff.max_consecutive_tokens,
        cutoff.max_ngram_repeats,
        cutoff.ngram_size,
    )
}

pub(super) fn make_sampling_config(
    config: &ChatConfig,
    model_config: &Gemma4Config,
) -> Option<SamplingConfig> {
    let temp = config
        .temperature
        .or(model_config.default_temperature)
        .unwrap_or(0.0);
    if temp <= 0.0 {
        // Greedy: use a near-zero temperature for argmax-like behavior.
        // Cannot pass None because sample() defaults to temperature=1.0.
        return Some(SamplingConfig {
            temperature: Some(0.0),
            top_k: None,
            top_p: None,
            min_p: None,
        });
    }
    Some(SamplingConfig {
        temperature: Some(temp),
        top_k: config.top_k.or(model_config.default_top_k),
        top_p: config.top_p.or(model_config.default_top_p),
        min_p: config.min_p,
    })
}

pub(super) fn sample_next_token(
    logits: &MxArray,
    config: Option<SamplingConfig>,
) -> Result<MxArray> {
    if is_greedy_sampling(config) {
        return logits.argmax(-1, Some(false));
    }
    sample(logits, config)
}

fn is_greedy_sampling(config: Option<SamplingConfig>) -> bool {
    config.is_some_and(|cfg| {
        cfg.temperature.unwrap_or(1.0) <= 0.0
            && cfg.top_k.is_none()
            && cfg.top_p.is_none()
            && cfg.min_p.is_none()
    })
}

/// Transformer body: embedding through decoder layers and final norm.
///
/// Matches mlx-vlm `Gemma4TextModel.__call__`. Does NOT run lm_head or softcap.
/// Used by chunked prefill for intermediate chunks and by the full forward.
///
/// When `inputs_embeds` is provided, uses it directly (skipping embedding lookup).
/// When `per_layer_inputs` is provided, uses it directly (skipping PLE computation).
///
/// `layer_ids` order; the compute graph is otherwise unchanged.
pub(crate) fn forward_body(
    input_ids: Option<&MxArray>,
    inputs_embeds: Option<MxArray>,
    embedding: &Embedding,
    layers: &[Gemma4DecoderLayer],
    caches: &mut [Gemma4LayerCache],
    final_norm: &RMSNorm,
    ple: Option<&PleComponents>,
    per_layer_inputs: Option<&MxArray>,
    config: &Gemma4Config,
) -> Result<MxArray> {
    // Step 1: Embedding (or use pre-computed embeddings)
    let mut h = if let Some(embeds) = inputs_embeds {
        embeds
    } else {
        let ids = input_ids.ok_or_else(|| {
            Error::from_reason("forward_body: either input_ids or inputs_embeds must be provided")
        })?;
        let emb = embedding.forward(ids)?;
        emb.mul_scalar((config.hidden_size as f64).sqrt())?
    };

    let seq_len = h.shape_at(1)?;

    // Step 2: PLE (per-layer embeddings) — compute or reuse
    let owned_ple: Option<MxArray>;
    let effective_ple: Option<&MxArray> = if let Some(ple_inputs) = per_layer_inputs {
        // Pre-computed: might need to slice for chunked prefill
        if ple_inputs.shape_at(1)? != seq_len {
            // Slice to match current chunk (chunked prefill)
            let cache_offset = caches
                .iter()
                .find_map(|c| {
                    let off = c.get_offset();
                    if off > 0 { Some(off as i64) } else { None }
                })
                .unwrap_or(0);
            let max_start = ple_inputs.shape_at(1)? - seq_len;
            let start = cache_offset.min(max_start);
            owned_ple = Some(ple_inputs.slice_axis(1, start, start + seq_len)?);
            owned_ple.as_ref()
        } else {
            Some(ple_inputs)
        }
    } else if let Some(ple) = ple {
        if let Some(ids) = input_ids {
            owned_ple = Some(compute_ple(ids, &h, ple, seq_len)?);
            owned_ple.as_ref()
        } else {
            None
        }
    } else {
        None
    };

    // Step 3: Project PLE if we have per-layer inputs
    // Matches mlx-vlm project_per_layer_inputs: projects h and combines with token PLEs
    let projected_ple: Option<MxArray> = if let Some(ple_data) = effective_ple {
        if let Some(ple) = ple {
            Some(project_per_layer_inputs(&h, ple_data, ple)?)
        } else {
            None
        }
    } else {
        None
    };

    // Step 4: Build masks
    // Global layers: None during prefill → triggers fused causal SDPA kernel
    // Sliding layers: explicit windowed mask during prefill
    // Decode (seq_len == 1): None for both
    //
    // Matches mlx-vlm create_attention_mask behavior:
    //   global → "causal" string → fused kernel
    //   sliding → explicit mask with window constraint
    // Sliding mask: only needed when the previous rotating-cache view plus the
    // current chunk exceeds the window. Matches mlx-lm RotatingKVCache.make_mask.
    let sliding_window = config.sliding_window as i64;
    let sliding_mask_offset = if seq_len > 1 {
        let sliding_idx = (0..config.num_hidden_layers as usize)
            .find(|&i| config.is_sliding_layer(i))
            .unwrap_or(0);
        let offset = if sliding_idx < caches.len() {
            caches[sliding_idx].get_offset()
        } else {
            0
        };
        sliding_mask_offset_for_chunk(seq_len, offset, sliding_window)
    } else {
        None
    };
    let sliding_mask = sliding_mask_offset
        .map(|offset| create_sliding_mask(seq_len, offset, sliding_window))
        .transpose()?;

    // Step 5: Forward through layers with KV cache sharing
    let has_kv_sharing = config.num_kv_shared_layers.is_some_and(|n| n > 0);
    let mut shared_kv: HashMap<usize, (MxArray, MxArray)> = HashMap::new();

    crate::models::gemma4::diagnostic::set_path("flat");

    for (i, layer) in layers.iter().enumerate() {
        crate::models::gemma4::diagnostic::set_layer(i);
        let is_global = config.is_global_layer(i);

        // Global layers: None mask → attention module uses causal SDPA or no-mask path
        // Sliding layers: explicit windowed mask
        let mask: Option<&MxArray> = if is_global {
            None
        } else {
            sliding_mask.as_ref()
        };

        let ple_input = projected_ple.as_ref().map(|p| {
            // projected_ple shape: [B, T, num_layers, ple_dim], extract layer i
            p.slice_axis(2, i as i64, i as i64 + 1)
                .and_then(|s| s.squeeze(Some(&[2])))
        });
        let ple_input_ref = match &ple_input {
            Some(Ok(arr)) => Some(arr),
            _ => None,
        };

        if has_kv_sharing && config.is_kv_shared_layer(i) {
            let anchor_idx = config.kv_shared_anchor(i).ok_or_else(|| {
                Error::from_reason(format!(
                    "Layer {} is shared but has no anchor (missing layer type match)",
                    i
                ))
            })?;

            let (shared_keys, shared_values) = shared_kv.get(&anchor_idx).ok_or_else(|| {
                Error::from_reason(format!(
                    "Anchor layer {} K/V not found for shared layer {}",
                    anchor_idx, i
                ))
            })?;

            // Shared layer uses anchor's cache offset.
            // Subtract seq_len to get pre-update offset (queries need same positions as anchor).
            let cache_offset = caches[anchor_idx].get_offset() - seq_len as i32;

            h = layer.forward_shared(
                &h,
                mask,
                shared_keys,
                shared_values,
                cache_offset,
                ple_input_ref,
            )?;
        } else {
            let needs_stash = has_kv_sharing && config.should_store_shared_kv(i);
            h = layer.forward(&h, mask, Some(&mut caches[i]), ple_input_ref, needs_stash)?;

            if has_kv_sharing
                && config.should_store_shared_kv(i)
                && let Some((keys, values)) = caches[i].take_stashed_kv()
            {
                shared_kv.insert(i, (keys, values));
            }
        }
    }

    // Final norm
    final_norm.forward(&h)
}

/// Full forward pass: transformer body + lm_head + logit softcapping.
///
/// Used for the final prefill chunk and for each decode step.
pub(crate) fn forward_inner(
    input_ids: &MxArray,
    embedding: &Embedding,
    layers: &[Gemma4DecoderLayer],
    caches: &mut [Gemma4LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embed_weight_t: Option<&MxArray>,
    ple: Option<&PleComponents>,
    config: &Gemma4Config,
) -> Result<MxArray> {
    let h = forward_body(
        Some(input_ids),
        None,
        embedding,
        layers,
        caches,
        final_norm,
        ple,
        None,
        config,
    )?;
    lm_head_logits(&h, embedding, lm_head, embed_weight_t, config)
}

/// LM head + logit softcapping over a post-final-norm hidden state — the
/// tail `forward_inner` runs after `forward_body`.
///
/// Projects through the explicit lm_head when present, otherwise through the
/// tied embedding table (packed-quantized, pre-transposed, or dense
/// transpose fallback), then applies `final_logit_softcapping` when the
/// config sets it.
pub(crate) fn lm_head_logits(
    h: &MxArray,
    embedding: &Embedding,
    lm_head: &Option<LinearProj>,
    embed_weight_t: Option<&MxArray>,
    config: &Gemma4Config,
) -> Result<MxArray> {
    crate::models::gemma4::diagnostic::dump_norm(0, "post_final_norm", h, None);

    // LM head or tied embeddings
    let logits = if let Some(head) = lm_head {
        head.forward(h)?
    } else if embedding.is_packed_quantized() {
        // Packed tied lm_head: project through the quantized matmul without
        // materializing the dense table.
        embedding.as_linear(h)?
    } else if let Some(w_t) = embed_weight_t {
        h.matmul(w_t)?
    } else {
        let weight = embedding.get_weight();
        let weight_t = weight.transpose(Some(&[1, 0]))?;
        h.matmul(&weight_t)?
    };
    crate::models::gemma4::diagnostic::dump_logits("pre_softcap", &logits);

    // Logit softcapping — compiled fused kernel (matches Python's mx.compile logit_softcap)
    if let Some(cap) = config.final_logit_softcapping {
        let cap_arr = MxArray::scalar_float_like(cap, &logits)?;
        let handle = unsafe { mlx_sys::mlx_logit_softcap(logits.handle.0, cap_arr.handle.0) };
        let capped = MxArray::from_handle(handle, "logit_softcap")?;
        crate::models::gemma4::diagnostic::dump_logits("post_softcap", &capped);
        Ok(capped)
    } else {
        crate::models::gemma4::diagnostic::dump_logits("post_softcap", &logits);
        Ok(logits)
    }
}

/// Draft-side accumulator a TAPPED paged prefill fills.
///
/// `layer_ids` are the drafter's target decoder indices (strictly ascending);
/// `ctx` is the drafter's fused-context cache, which gains one row per
/// prefilled token at its absolute sequence position.
pub(crate) struct Gemma4DsparkPrefillTap<'a> {
    pub(crate) layer_ids: &'a [usize],
    pub(crate) ctx: &'a mut DsparkContextCache,
}

/// Final-norm + LM-head + softcap projection over a paged residual
/// `[1, T, hidden]` — the tail every paged forward runs after its layer loop.
///
/// `last_only == false` keeps every row (`[1, T, vocab]`): the speculative
/// verify shape, one logit row per drafted position, reusing the decode
/// tail's head/softcap idiom unchanged over T rows. `last_only == true`
/// reproduces the paged prefill tails' last-token projection exactly —
/// project all rows, then slice row T-1 and squeeze to `[vocab]` — so the
/// two modes differ only in the final slice.
pub(crate) fn project_paged_hidden_rows(
    hidden_states: &MxArray,
    final_norm: &RMSNorm,
    embedding: &Embedding,
    lm_head: &Option<LinearProj>,
    embed_weight_t: Option<&MxArray>,
    config: &Gemma4Config,
    last_only: bool,
) -> Result<MxArray> {
    let hidden = final_norm.forward(hidden_states)?;
    let logits = lm_head_logits(&hidden, embedding, lm_head, embed_weight_t, config)?;
    if !last_only {
        return Ok(logits);
    }
    let last_seq_len = logits.shape_at(1)?;
    logits
        .slice_axis(1, last_seq_len - 1, last_seq_len)?
        .squeeze(Some(&[0, 1]))
}

/// The paged layer loops' twin of `forward_body`'s tap validation: capture
/// order is push-in-loop, so `layer_ids` must be strictly ascending decoder
/// indices below the layer count.
pub(super) fn validate_paged_tap_layer_ids(
    tap: Option<&DsparkTap<'_>>,
    num_layers: usize,
) -> Result<()> {
    let Some(t) = tap else {
        return Ok(());
    };
    let mut previous: Option<usize> = None;
    for &id in t.layer_ids {
        if id >= num_layers || previous.is_some_and(|prev| id <= prev) {
            return Err(Error::from_reason(format!(
                "paged layer loop: tap layer_ids {:?} must be strictly ascending decoder \
                 indices below {num_layers}",
                t.layer_ids
            )));
        }
        previous = Some(id);
    }
    Ok(())
}

/// Run the target over a `[1, T]` verify block at the current cache offset
/// and return the `[1, T, vocab]` softcapped logits together with the
/// `[1, T, hidden]` post-final-norm hidden state (the assistant draft chains
/// its next round's `h_prev` from the hidden at the last kept slot).
///
/// It does not sample and touches no history bookkeeping; caches advance by
/// T. Callers pair it with `snapshot_before_verify` / `commit_after_verify`
/// for rollback.
pub(crate) fn assistant_verify_forward(
    block_ids: &MxArray,
    embedding: &Embedding,
    layers: &[Gemma4DecoderLayer],
    caches: &mut [Gemma4LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embed_weight_t: Option<&MxArray>,
    ple: Option<&PleComponents>,
    config: &Gemma4Config,
) -> Result<(MxArray, MxArray)> {
    if block_ids.ndim()? != 2 || block_ids.shape_at(0)? != 1 || block_ids.shape_at(1)? < 1 {
        return Err(Error::from_reason(format!(
            "assistant_verify_forward expects block_ids shaped [1, T] with T >= 1, got {:?}",
            block_ids.shape()?.as_ref()
        )));
    }
    let hidden = forward_body(
        Some(block_ids),
        None,
        embedding,
        layers,
        caches,
        final_norm,
        ple,
        None,
        config,
    )?;
    let logits = lm_head_logits(&hidden, embedding, lm_head, embed_weight_t, config)?;
    Ok((logits, hidden))
}

/// Shared-slot mask for `snapshot_before_verify`, index-aligned with the
/// per-layer caches vec: entry i is true iff decoder layer i is KV-shared.
/// Shared layers read their anchor layer's cache; their own vec entry is
/// never written by a forward pass.
pub(crate) fn dspark_shared_slot_mask(config: &Gemma4Config) -> Vec<bool> {
    (0..config.num_hidden_layers as usize)
        .map(|i| config.is_kv_shared_layer(i))
        .collect()
}

/// Target-layer indices whose KV caches the assistant draft reads: one
/// source per attention type, index-aligned with the per-layer caches vec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AssistantKvSources {
    pub sliding: usize,
    pub full: usize,
}

/// Resolve the assistant draft's K/V source layers: for each attention type,
/// the LAST non-KV-shared target layer of that type — the max index
/// `i < config.first_kv_shared_layer()` whose `layer_types` entry equals the
/// type string exactly, the same matching `should_store_shared_kv` uses to
/// mark anchors. Layers with a missing or unrecognized `layer_types` entry
/// match neither type. With KV sharing enabled these are exactly the anchor
/// layers `should_store_shared_kv` marks; without sharing they are simply
/// the last layer of each type. Errors when the non-shared prefix lacks
/// either attention type — the draft needs one K/V source per type.
pub(crate) fn assistant_kv_source_indices(config: &Gemma4Config) -> Result<AssistantKvSources> {
    let first_shared = config.first_kv_shared_layer();
    let last_below_boundary = |layer_type: &str| {
        (0..first_shared).rfind(|&i| config.layer_types.get(i).is_some_and(|t| t == layer_type))
    };
    let sliding = last_below_boundary("sliding_attention").ok_or_else(|| {
        Error::from_reason(format!(
            "assistant KV source mapping: no non-KV-shared sliding_attention layer in layers 0..{first_shared}"
        ))
    })?;
    let full = last_below_boundary("full_attention").ok_or_else(|| {
        Error::from_reason(format!(
            "assistant KV source mapping: no non-KV-shared full_attention layer in layers 0..{first_shared}"
        ))
    })?;
    Ok(AssistantKvSources { sliding, full })
}

/// Compute PLE (per-layer embeddings) from input_ids.
/// Returns shape [B, T, num_layers, ple_dim].
pub(crate) fn compute_ple(
    input_ids: &MxArray,
    h: &MxArray,
    ple: &PleComponents,
    seq_len: i64,
) -> Result<MxArray> {
    let ple_dim = ple.ple_dim as i64;
    let num_layers = ple.num_layers as i64;

    // Mask OOV token IDs to 0 for PLE embedding
    let ple_vocab = MxArray::scalar_int(ple.vocab_size_per_layer_input)?;
    let zero = MxArray::scalar_int(0)?;
    let valid_mask = input_ids
        .greater_equal(&zero)?
        .logical_and(&input_ids.less(&ple_vocab)?)?;
    let masked_ids = valid_mask.where_(input_ids, &zero)?;

    // per_layer_embeds: [B, T, num_layers * ple_dim]
    let per_layer_embeds = ple.embed_tokens_per_layer.forward(&masked_ids)?;
    let per_layer_embeds = per_layer_embeds.mul_scalar((ple.ple_dim as f64).sqrt())?;
    let batch = per_layer_embeds.shape_at(0)?;
    let per_layer_embeds = per_layer_embeds.reshape(&[batch, seq_len, num_layers, ple_dim])?;

    // Project from main hidden state
    let projected = ple.per_layer_model_projection.forward(h)?;
    let projected = projected.mul_scalar(ple.per_layer_model_projection_scale)?;
    let projected = projected.reshape(&[batch, seq_len, num_layers, ple_dim])?;

    let projected = ple.per_layer_projection_norm.forward(&projected)?;

    // Combine: (normed_projection + per_layer_embeds) * 1/sqrt(2)
    let combined = projected.add(&per_layer_embeds)?;
    combined.mul_scalar(ple.per_layer_input_scale)
}

/// Project per-layer inputs: combine PLE data with hidden state projection.
/// Returns shape [B, T, num_layers, ple_dim].
fn project_per_layer_inputs(
    _h: &MxArray,
    per_layer_data: &MxArray,
    _ple: &PleComponents,
) -> Result<MxArray> {
    // PLE data is already fully computed (combined projection + token embeddings)
    Ok(per_layer_data.clone())
}

#[cfg(test)]
pub(crate) fn compute_layer_kinds_from_kv_cache_specs(
    config: &Gemma4Config,
) -> std::result::Result<Vec<Gemma4LayerKind>, String> {
    let n = config.num_hidden_layers as usize;
    let block_size = config.paged_block_size.unwrap_or(16);
    let specs = compute_layer_kv_cache_specs(config, block_size, KVCacheDType::BFloat16)?;
    let max_model_len = u32::try_from(config.max_position_embeddings).map_err(|_| {
        format!(
            "Gemma4 layer kind routes: invalid max_position_embeddings {}",
            config.max_position_embeddings
        )
    })?;
    let routes = crate::transformer::derive_layer_kv_cache_routes(
        &specs,
        max_model_len,
        gemma4_paged_prefill_group_max_chunk(),
    )
    .map_err(|e| format!("Gemma4 layer kind route derivation failed: {e}"))?;

    layer_kinds_from_routes(&routes, n)
}

pub(super) fn layer_kinds_from_routes(
    routes: &[LayerKVCacheRoute],
    n: usize,
) -> std::result::Result<Vec<Gemma4LayerKind>, String> {
    let mut kinds = vec![None; n];
    for route in routes {
        if route.layer_index >= n {
            return Err(format!(
                "Gemma4 layer kind route derivation produced out-of-range layer {} for {n} layers",
                route.layer_index
            ));
        }
        let physical_ordinal = u32::try_from(route.physical_layer_ordinal).map_err(|_| {
            format!(
                "Gemma4 layer kind route ordinal {} does not fit u32",
                route.physical_layer_ordinal
            )
        })?;
        let kind = match (route.shared_kv_anchor, route.attention_kind) {
            (Some(_), AttentionKind::Full) => Gemma4LayerKind::SharedOnGlobal {
                group_id: route.group_id,
                anchor_paged_idx: physical_ordinal,
            },
            (Some(_), AttentionKind::SlidingWindow { .. }) => Gemma4LayerKind::SharedOnSliding {
                group_id: route.group_id,
                anchor_paged_idx: physical_ordinal,
            },
            (None, AttentionKind::Full) => Gemma4LayerKind::GlobalPaged {
                group_id: route.group_id,
                paged_idx: physical_ordinal,
            },
            (None, AttentionKind::SlidingWindow { .. }) => Gemma4LayerKind::SlidingPaged {
                group_id: route.group_id,
                paged_idx: physical_ordinal,
            },
        };
        kinds[route.layer_index] = Some(kind);
    }

    kinds
        .into_iter()
        .enumerate()
        .map(|(layer_index, kind)| {
            kind.ok_or_else(|| {
                format!("Gemma4 layer kind route derivation missed layer {layer_index}")
            })
        })
        .collect()
}

/// Build Gemma4's model-independent KV-cache specs.
///
/// The specs are the long-term source of truth for the paged/sliding cache
/// architecture: models declare attention/cache requirements, and common
/// transformer infrastructure groups layers and owns block tables. The current
/// Gemma4 runtime still routes through `Gemma4LayerKind`, but both helpers must
/// agree on physical storage ownership: KV-shared layers are aliases and do not
/// allocate separate cache slots.
pub(crate) fn compute_layer_kv_cache_specs(
    config: &Gemma4Config,
    block_size: u32,
    cache_dtype: KVCacheDType,
) -> std::result::Result<Vec<LayerKVCacheSpec>, String> {
    if block_size == 0 {
        return Err("Gemma4 KV cache specs require block_size > 0".to_string());
    }
    if config.sliding_window <= 0 {
        return Err(format!(
            "Gemma4 KV cache specs require sliding_window > 0, got {}",
            config.sliding_window
        ));
    }

    let n = config.num_hidden_layers as usize;
    let mut specs = Vec::with_capacity(n);
    for layer_index in 0..n {
        let is_global = config.is_global_layer(layer_index);
        let head_size = u32::try_from(config.effective_head_dim(is_global)).map_err(|_| {
            format!(
                "Gemma4 KV cache specs: layer {layer_index} has invalid head_dim {}",
                config.effective_head_dim(is_global)
            )
        })?;
        let num_kv_heads = u32::try_from(config.effective_kv_heads(is_global)).map_err(|_| {
            format!(
                "Gemma4 KV cache specs: layer {layer_index} has invalid num_kv_heads {}",
                config.effective_kv_heads(is_global)
            )
        })?;
        let layout = KVCachePhysicalLayout::new(block_size, num_kv_heads, head_size, cache_dtype);
        if !layout.is_valid() {
            return Err(format!(
                "Gemma4 KV cache specs: layer {layer_index} has invalid physical layout \
                 block_size={block_size}, num_kv_heads={num_kv_heads}, head_size={head_size}"
            ));
        }

        let attention_kind = if is_global {
            AttentionKind::Full
        } else {
            AttentionKind::SlidingWindow {
                sliding_window: config.sliding_window as u32,
            }
        };
        let mut spec = LayerKVCacheSpec::new(layer_index, attention_kind, layout);
        if config.is_kv_shared_layer(layer_index) {
            let anchor = config.kv_shared_anchor(layer_index).ok_or_else(|| {
                format!(
                    "Gemma4 KV cache specs: layer {layer_index} is KV-shared but has no \
                     resolvable anchor"
                )
            })?;
            spec = spec.shared_with_anchor(anchor);
        }
        specs.push(spec);
    }

    crate::transformer::validate_layer_kv_cache_specs(&specs)
        .map_err(|e| format!("Gemma4 KV cache specs failed validation: {e}"))?;
    Ok(specs)
}

pub(crate) fn compute_layer_kv_cache_groups(
    config: &Gemma4Config,
    block_size: u32,
    cache_dtype: KVCacheDType,
    max_chunk: u32,
) -> std::result::Result<Vec<KVCacheGroup>, String> {
    let specs = compute_layer_kv_cache_specs(config, block_size, cache_dtype)?;
    let max_model_len = u32::try_from(config.max_position_embeddings).map_err(|_| {
        format!(
            "Gemma4 KV cache groups: invalid max_position_embeddings {}",
            config.max_position_embeddings
        )
    })?;
    group_layer_kv_cache_specs(&specs, max_model_len, max_chunk)
        .map_err(|e| format!("Gemma4 KV cache grouping failed: {e}"))
}

pub(super) fn gemma4_group_reserved_blocks(
    attention_kind: AttentionKind,
    max_admission_blocks: u32,
    scheduler_width: u32,
) -> u32 {
    match attention_kind {
        AttentionKind::Full => max_admission_blocks,
        AttentionKind::SlidingWindow { .. } => {
            max_admission_blocks.max(scheduler_width).saturating_add(1)
        }
    }
}

#[cfg(test)]
pub(super) fn physical_full_attention_layer_count(specs: &[LayerKVCacheSpec]) -> usize {
    specs
        .iter()
        .filter(|spec| {
            spec.shared_kv_anchor.is_none() && matches!(spec.attention_kind, AttentionKind::Full)
        })
        .count()
}

#[cfg(test)]
pub(super) fn gemma4_default_paged_cache_memory_mb(
    max_seq_len: u32,
    block_size: u32,
    head_size: u32,
    num_kv_heads: u32,
    num_layers: u32,
) -> u32 {
    if max_seq_len == 0 || block_size == 0 || head_size == 0 || num_kv_heads == 0 || num_layers == 0
    {
        return GEMMA4_PAGED_CACHE_MIN_DEFAULT_MEMORY_MB;
    }

    let max_blocks = u64::from(max_seq_len.div_ceil(block_size));
    let bytes_per_block = 2u64
        .saturating_mul(u64::from(num_kv_heads))
        .saturating_mul(u64::from(head_size))
        .saturating_mul(u64::from(block_size))
        .saturating_mul(2)
        .saturating_mul(u64::from(num_layers));
    let required_mb = bytes_per_block
        .saturating_mul(max_blocks)
        .div_ceil(BYTES_PER_MIB)
        .max(u64::from(GEMMA4_PAGED_CACHE_MIN_DEFAULT_MEMORY_MB));
    u32::try_from(required_mb).unwrap_or(u32::MAX)
}
