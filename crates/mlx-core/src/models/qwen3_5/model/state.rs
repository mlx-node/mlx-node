//! Family state helpers: layer caches, GDN checkpoint records, media plans, decode inputs.

use super::*;

pub(super) fn fresh_dense_layer_caches(config: &Qwen3_5Config) -> Vec<Qwen3_5LayerCache> {
    (0..config.num_layers as usize)
        .map(|i| {
            if config.is_linear_layer(i) {
                Qwen3_5LayerCache::new_linear()
            } else {
                Qwen3_5LayerCache::new_full_attention()
            }
        })
        .collect()
}

/// Create the profiler owned by a paged-MTP turn before its prefill starts.
///
/// Paged AR intentionally keeps its existing terminal-only metrics path and
/// therefore does not allocate a decode profiler here. This helper starts the
/// prefill timer immediately before the actual paged prefill; the caller ends
/// it immediately afterward, then carries this same instance into
/// `run_mtp_turn` for decode and acceptance accounting.
pub(super) fn configure_paged_mtp_profiler(
    mut profiler: crate::decode_profiler::DecodeProfiler,
    fresh_suffix_tokens: u32,
) -> crate::decode_profiler::DecodeProfiler {
    profiler.set_prompt_tokens(fresh_suffix_tokens);
    profiler.snapshot_memory_before();
    profiler.begin_prefill();
    profiler
}

pub(super) fn begin_paged_mtp_profiler(
    eager_mtp_paged: bool,
    fresh_suffix_tokens: u32,
) -> Option<crate::decode_profiler::DecodeProfiler> {
    eager_mtp_paged.then(|| {
        configure_paged_mtp_profiler(
            crate::decode_profiler::DecodeProfiler::new("chat_paged_mtp_eager", "qwen3_5"),
            fresh_suffix_tokens,
        )
    })
}

/// Enforce one fixed paged-pool token ceiling against already-resolved chat
/// parameters. Shared by dense and MoE so every native paged executor uses the
/// same prompt rejection marker and the same last-sampled-token accounting.
pub(crate) fn constrain_paged_context_params(
    family: &str,
    prompt_tokens: usize,
    capacity: u32,
    params: &mut engine::ChatParams,
) -> Result<()> {
    let prompt = u32::try_from(prompt_tokens).unwrap_or(u32::MAX);
    if prompt > capacity {
        return Err(Error::from_reason(format!(
            "context_length_exceeded: rendered prompt has {prompt} tokens, effective active \
             context is {capacity} tokens"
        )));
    }
    // The final sampled token is returned without another model forward, so N
    // generated tokens consume only N-1 additional KV positions.
    let max_output = capacity.saturating_sub(prompt).saturating_add(1);
    let requested = params.max_new_tokens.max(0) as u32;
    if requested > max_output {
        warn!(
            "{family} clamping max_new_tokens from {} to {} for effective context {} \
             (prompt_tokens={})",
            requested, max_output, capacity, prompt
        );
        params.max_new_tokens = max_output as i32;
    }
    Ok(())
}

pub(super) struct DenseGdnPrefixCheckpoint {
    pub(super) owner_id: String,
    pub(super) prefix_len: u32,
    pub(super) block_size: u32,
    pub(super) final_block_hash: u64,
    pub(super) block_hashes: Vec<u64>,
    pub(super) tokens: Vec<u32>,
    pub(super) caches: Vec<Qwen3_5LayerCache>,
}

impl GdnCheckpointLineage for DenseGdnPrefixCheckpoint {
    fn owner_id(&self) -> &str {
        &self.owner_id
    }

    fn prefix_len(&self) -> u32 {
        self.prefix_len
    }

    fn block_size(&self) -> u32 {
        self.block_size
    }

    fn final_block_hash(&self) -> u64 {
        self.final_block_hash
    }

    fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    fn block_hashes(&self) -> &[u64] {
        &self.block_hashes
    }
}

pub(super) struct DenseGdnHistoryCheckpoint {
    pub(super) owner_id: String,
    pub(super) image_key: Option<u64>,
    pub(super) tokens: Vec<u32>,
    pub(super) caches: Vec<Qwen3_5LayerCache>,
}

pub(super) struct DenseGdnPrefixPreparation {
    pub(super) state: &'static str,
    pub(super) already_primed: bool,
    pub(super) restored_prefix_tokens: u32,
    pub(super) replayed_prefix_tokens: u32,
}

#[derive(Default)]
pub(super) struct DenseGdnCheckpointStoreTrace {
    pub(super) stored: bool,
    pub(super) eval_ms: f64,
    pub(super) clone_ms: f64,
    pub(super) token_clone_ms: f64,
    pub(super) update_ms: f64,
    pub(super) total_ms: f64,
}

impl DenseGdnCheckpointStoreTrace {
    pub(super) fn finish(mut self, start: Option<std::time::Instant>) -> Self {
        self.total_ms = start.map(elapsed_ms).unwrap_or(0.0);
        self
    }
}

#[derive(Clone, Copy)]
pub(super) struct TokenPrefixMismatchTrace {
    pub(super) index: i64,
    pub(super) prompt_token: i64,
    pub(super) cached_token: i64,
}

impl Default for TokenPrefixMismatchTrace {
    fn default() -> Self {
        Self {
            index: -1,
            prompt_token: -1,
            cached_token: -1,
        }
    }
}

pub(super) fn token_prefix_mismatch_trace(
    prompt: &[u32],
    cached: &[u32],
) -> TokenPrefixMismatchTrace {
    let common_len = prompt.len().min(cached.len());
    for i in 0..common_len {
        if prompt[i] != cached[i] {
            return TokenPrefixMismatchTrace {
                index: i as i64,
                prompt_token: prompt[i] as i64,
                cached_token: cached[i] as i64,
            };
        }
    }

    TokenPrefixMismatchTrace {
        index: common_len as i64,
        prompt_token: prompt.get(common_len).map_or(-1, |token| *token as i64),
        cached_token: cached.get(common_len).map_or(-1, |token| *token as i64),
    }
}

pub(super) fn dense_paged_linear_caches_ready(
    config: &Qwen3_5Config,
    caches: Option<&[Qwen3_5LayerCache]>,
) -> bool {
    let Some(caches) = caches else {
        return false;
    };
    if caches.len() != config.num_layers as usize {
        return false;
    }
    for (i, cache) in caches.iter().enumerate() {
        if !config.is_linear_layer(i) {
            continue;
        }
        let Qwen3_5LayerCache::Linear(arrays) = cache else {
            return false;
        };
        if arrays.get(0).is_none() || arrays.get(1).is_none() {
            return false;
        }
    }
    true
}

/// Signed skew between the adapter's recorded token count and the drop-last
/// token history a dense paged epilogue is about to persist, `None` when they
/// agree. Both sides drop the SAME unforwarded final token (the paged decode
/// loop never forwards the last sampled token, and the saved history drops it
/// too), so agreement is STRICT equality — any ±1 tolerance here would either
/// mask a real one-token skew or arm the refuse-to-persist latch forever.
pub(super) fn dense_paged_frontier_skew(
    adapter_recorded_len: usize,
    history_len: usize,
) -> Option<i64> {
    let skew = adapter_recorded_len as i64 - history_len as i64;
    (skew != 0).then_some(skew)
}

/// Exact bit-level equality of two arrays (test oracle only). 16-bit floats
/// compare on their raw bit patterns via the native extraction; everything
/// else round-trips through f32 and compares `to_bits`, so `-0.0 != 0.0` and
/// differing NaN payloads count as differences — exactly what a
/// state-equals-its-key audit needs.
pub(crate) fn arrays_bits_equal_for_test(a: &MxArray, b: &MxArray) -> Result<bool> {
    a.eval();
    b.eval();
    let (da, db) = (a.dtype()?, b.dtype()?);
    if da != db {
        return Ok(false);
    }
    match da {
        crate::array::DType::BFloat16 | crate::array::DType::Float16 => {
            Ok(a.to_uint16_native()? == b.to_uint16_native()?)
        }
        _ => {
            let av = a.to_float32()?;
            let bv = b.to_float32()?;
            if av.len() != bv.len() {
                return Ok(false);
            }
            Ok(av
                .iter()
                .zip(bv.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()))
        }
    }
}

pub(super) fn clone_dense_linear_layer_caches(
    config: &Qwen3_5Config,
    caches: &[Qwen3_5LayerCache],
) -> Option<Vec<Qwen3_5LayerCache>> {
    if !dense_paged_linear_caches_ready(config, Some(caches)) {
        return None;
    }

    let mut cloned = fresh_dense_layer_caches(config);
    for i in 0..config.num_layers as usize {
        if !config.is_linear_layer(i) {
            continue;
        }
        let Qwen3_5LayerCache::Linear(arrays) = &caches[i] else {
            return None;
        };
        cloned[i] = Qwen3_5LayerCache::Linear(arrays.clone());
    }
    Some(cloned)
}

/// Build the dense media admission contract from the components wired into
/// one loaded Qwen3.5 instance.
///
/// Image execution needs all three pieces. Missing combinations remain
/// backend-validated so the family core preserves its established diagnostic
/// instead of the engine replacing it with a generic unsupported-media error.
pub(crate) const fn qwen35_dense_vision_active(
    has_vision_encoder: bool,
    has_image_processor: bool,
    has_paged_adapter: bool,
) -> bool {
    has_vision_encoder && has_image_processor && has_paged_adapter
}

pub(super) const fn qwen35_dense_media_plan(
    has_vision_encoder: bool,
    has_image_processor: bool,
    has_paged_adapter: bool,
) -> MediaPlan {
    let images_available =
        qwen35_dense_vision_active(has_vision_encoder, has_image_processor, has_paged_adapter);
    MediaPlan::with_backend_validation(
        MediaCapabilities {
            images: images_available,
            audio: false,
        },
        MediaCapabilities::IMAGES,
    )
}

/// Media represented by the live dense session state.
///
/// The content hash identifies a freshly saved image turn. Generic paged
/// continuations intentionally clear that key, but retain the M-RoPE delta
/// while they keep extending the same live image-derived KV prefix. Treat
/// either signal as image context so request planning stays truthful across
/// every successive text continuation.
pub(super) const fn qwen35_dense_session_media(
    has_cached_image_key: bool,
    has_cached_rope_delta: bool,
) -> MediaCapabilities {
    if has_cached_image_key || has_cached_rope_delta {
        MediaCapabilities::IMAGES
    } else {
        MediaCapabilities::NONE
    }
}

pub(super) fn qwen35_dense_session_media_matches_payloads(
    cached_image_key: Option<u64>,
    images: &[Vec<u8>],
    audio: &[Vec<u8>],
) -> bool {
    audio.is_empty()
        && !images.is_empty()
        && cached_image_key == Some(engine::compute_image_cache_key(images))
}

/// Make the resolved decoder authoritative for legacy Qwen3.5 whole-turn
/// cores, which still derive their local `ChatParams` from a `ChatConfig`.
pub(super) fn apply_qwen35_dense_planned_decoder(
    config: &mut ChatConfig,
    decoder: DecoderPlan,
) -> bool {
    let planned_mtp = matches!(
        decoder,
        DecoderPlan::Speculative(SpeculativeKind::NativeMtp)
    );
    config.enable_mtp = Some(planned_mtp);
    planned_mtp
}

/// Input bundle for [`Qwen35Inner::chat_with_caches_inner`].
///
/// Packs every value the shared post-prefill pipeline needs into a single
/// named struct so callers don't have to thread 20+ positional arguments.
/// Constructed by the prefill-side of [`Qwen35Inner::vision_mtp_whole_turn_core`] and
/// [`Qwen35Inner::chat_tokens_delta_sync`].
///
/// The caller is responsible for:
///   - constructing a `WiredLimitContext` tied to `generation_stream` for
///     the lifetime of the call,
///   - running prefill and packaging the resulting `last_logits` and
///     `seq_len`.
pub(crate) struct ChatDecodeInputs {
    /// Logits for the last position of the prefill chunk. Penalties and
    /// sampling run against this to produce the first decoded token.
    pub last_logits: MxArray,
    /// Total context length after prefill (cached + newly-prefilled).
    pub seq_len: i64,
    /// `true` when this invocation is a session DELTA continuation
    /// (text-only append on top of the live KV cache). Drives the
    /// post-decode save pathway: deltas keep `cached_image_key` sticky
    /// so image attention state baked into the KV caches by a prior
    /// prefill stays addressable; prefills (re)set the key based on
    /// the fresh turn's `has_images`.
    pub is_delta: bool,

    /// `true` when the current turn carries images.
    pub has_images: bool,

    /// Full pre-decode token sequence. Seeds the decode loop's running
    /// history (mutated in place) and the penalty context.
    pub token_history_init: Vec<u32>,
    /// Token snapshot handed to `save_cache_state_direct`. For text-only
    /// this equals `token_history_init`; for VLM it's the pre-expansion
    /// tokens.
    pub save_tokens: Vec<u32>,
    /// Expanded token sequence (with image placeholders expanded) used by
    /// the VLM save path. `None` for text-only.
    pub save_expanded_tokens: Option<Vec<u32>>,
    /// Image cache key for the current turn. 0 for text-only.
    pub save_image_cache_key: u64,

    pub tokenizer: Arc<Qwen3Tokenizer>,
    pub think_end_id: Option<u32>,
    pub think_end_str: Option<String>,
    /// Resolved thinking-mode state for the turn — the single source of
    /// truth, threaded from `WholeTurnArgs::thinking` so the cores share
    /// one `resolve_enable_thinking` result.
    pub thinking: ThinkingSetup,
    /// End-of-sequence token id for the decode loop. For `vision_mtp_whole_turn_core` this
    /// is `config.eos_token_id`; for the session delta path it's
    /// `<|im_end|>` so cache boundaries stay clean.
    pub eos_id: u32,

    pub profiler: crate::decode_profiler::DecodeProfiler,
    pub generation_start: Option<std::time::Instant>,
    pub first_token_instant: Option<std::time::Instant>,
    /// Number of tokens actually prefilled this turn (for throughput math).
    pub prefill_tokens_len: usize,
    /// Prompt token count reported on the `ChatResult`.
    pub prompt_tokens_for_result: u32,
    /// Length of the reused cached prefix to report on `ChatResult.cached_tokens`.
    ///
    /// For fresh prefills this is `cached_prefix_len` (0 on a miss, full
    /// cached length on an exact-append hit). For the session delta path
    /// this is the full prior-history length because the delta is
    /// appended on top of the existing caches — we skip the `cached_prefix_len`
    /// driver (which only gates the VLM rope-delta replay branch) while
    /// still reporting the reused prefix accurately for observability.
    pub cached_tokens_for_result: u32,

    pub embedding: Embedding,
    pub generation_stream: Stream,
    pub params: crate::engine::ChatParams,

    /// Post-final-norm hidden state for every prefilled prompt token,
    /// `[1, prefill_len, hidden]`. `Some` only when MTP is active for this
    /// turn (`params.enable_mtp && has_mtp_weights`) and the prefill ran
    /// the hidden-emitting `chunked_prefill_with_hidden`. Consumed once,
    /// by `begin_mtp_decode`'s prompt-prefix seed, to commit the prompt
    /// prefix into the MTP committed-history cache.
    /// `None` for non-MTP turns and for the streaming/delta paths.
    pub prompt_hidden: Option<MxArray>,
    /// The exact prompt token ids whose hiddens `prompt_hidden` holds —
    /// i.e. the `prefill_tokens` slice the hidden-emitting prefill
    /// forwarded. `prompt_hidden.shape(1) == prompt_hidden_ids.len()`.
    /// `Some` iff `prompt_hidden` is `Some`.
    pub prompt_hidden_ids: Option<Vec<u32>>,
}
