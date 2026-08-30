//! Stream glue for the paged/vision cores plus the small per-turn state values: the chunk-sink adapter, the channel dispatch latch, the emitter, media plans and the draft handle.

use super::*;

/// Adapter giving the paged/vision streaming cores a `cb.call(result, mode)`
/// shape over the engine's [`ChunkSink`].
///
/// The engine owns the channel and hands the probes/emitter a `&dyn
/// ChunkSink`, so the wrapper forwards `.call()` to [`ChunkSink::send`].
/// The call mode is meaningless on the mpsc path and is dropped.
pub(super) struct StreamSender<'a>(pub(super) &'a dyn ChunkSink);

impl StreamSender<'_> {
    pub(super) fn call(&self, result: Result<ChatStreamChunk>, _mode: ThreadsafeFunctionCallMode) {
        self.0.send(result);
    }
}

fn emit_stream_delta(text: String, is_reasoning: bool, cb: &StreamSender<'_>) {
    if text.is_empty() {
        return;
    }
    cb.call(
        Ok(ChatStreamChunk {
            text,
            done: false,
            finish_reason: None,
            tool_calls: None,
            thinking: None,
            thinking_enabled: None,
            num_tokens: None,
            prompt_tokens: None,
            reasoning_tokens: None,
            raw_text: None,
            public_raw_text: None,
            text_authoritative: None,
            cached_tokens: None,
            performance: None,
            is_reasoning: Some(is_reasoning),
        }),
        ThreadsafeFunctionCallMode::NonBlocking,
    );
}

/// Gemma4 marks both hidden reasoning and some answer-only turns with
/// `<|channel>thought\n...<channel|>`. Once a reasoning delta has been
/// streamed to Anthropic SSE we cannot re-label that content as visible
/// text, so keep leading channel bytes pending until a visible text/tool
/// segment proves the channel was real reasoning. If an ambiguous,
/// model-opened channel ends with only that pending body, surface it as
/// normal text; a prompt-seeded channel is known reasoning even when
/// generation truncates before its close marker.
#[derive(Default)]
pub(super) struct Gemma4StreamDispatchState {
    pending_reasoning: String,
    visible_text_emitted: bool,
    tool_call_seen: bool,
    starts_in_prompted_channel: bool,
}

impl Gemma4StreamDispatchState {
    pub(super) fn new(starts_in_prompted_channel: bool) -> Self {
        Self {
            starts_in_prompted_channel,
            ..Self::default()
        }
    }

    pub(super) fn dispatch_segments(
        &mut self,
        segments: Vec<crate::models::gemma4::output_parser::StreamSegment>,
        cb: &StreamSender<'_>,
    ) {
        use crate::models::gemma4::output_parser::StreamSegment;
        for seg in segments {
            match seg {
                StreamSegment::Text(text) => {
                    if text.is_empty() {
                        continue;
                    }
                    self.flush_pending_reasoning(cb);
                    self.visible_text_emitted = true;
                    emit_stream_delta(text, false, cb);
                }
                StreamSegment::Reasoning(text) => {
                    if text.is_empty() {
                        continue;
                    }
                    if self.visible_text_emitted || self.tool_call_seen {
                        emit_stream_delta(text, true, cb);
                    } else {
                        self.pending_reasoning.push_str(&text);
                    }
                }
                StreamSegment::ToolCall => {
                    self.tool_call_seen = true;
                    self.flush_pending_reasoning(cb);
                    // Accumulated on `parser.tool_calls()` for the terminal chunk.
                }
            }
        }
    }

    pub(super) fn finish(&mut self, cb: &StreamSender<'_>) {
        if self.pending_reasoning.is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.pending_reasoning);
        if self.visible_text_emitted || self.tool_call_seen || self.starts_in_prompted_channel {
            emit_stream_delta(text, true, cb);
        } else {
            self.visible_text_emitted = true;
            emit_stream_delta(text, false, cb);
        }
    }

    fn flush_pending_reasoning(&mut self, cb: &StreamSender<'_>) {
        if self.pending_reasoning.is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.pending_reasoning);
        emit_stream_delta(text, true, cb);
    }
}

pub(super) fn promote_channel_only_output(
    parsed: &mut crate::models::gemma4::output_parser::Gemma4ParsedOutput,
    starts_in_prompted_channel: bool,
) {
    if !starts_in_prompted_channel
        && parsed.text.trim().is_empty()
        && parsed.tool_calls.is_empty()
        && parsed
            .thinking
            .as_deref()
            .is_some_and(|thinking| !thinking.trim().is_empty())
    {
        parsed.text = parsed.thinking.take().unwrap_or_default();
    }
}

/// Gemma4's [`StreamEmitter`]: routes every committed token's raw
/// (special-token-preserving — [`ChatBackend::stream_skip_special_tokens`]
/// returns `false`) text through [`Gemma4StreamParser`] +
/// [`Gemma4StreamDispatchState`]: channel/tool-call segmentation,
/// pending-reasoning buffering, channel-only promotion, empty-chunk
/// filtering. `is_reasoning` / `include_reasoning` are deliberately
/// ignored — Gemma4's reasoning labeling comes from the parser's channel
/// markers, not the engine's `<think>`-token tracker. Selectable thinking
/// is enabled by the prompt's `<|think|>` capability token; the tracker
/// stays disabled because Gemma4 closes reasoning with `<channel|>`, not
/// a `</think>` token.
pub(super) struct Gemma4Emitter {
    parser: crate::models::gemma4::output_parser::Gemma4StreamParser,
    dispatch: Gemma4StreamDispatchState,
}

impl Gemma4Emitter {
    pub(super) fn new(starts_in_open_channel: bool) -> Self {
        Self {
            parser: crate::models::gemma4::output_parser::Gemma4StreamParser::new_with_open_channel(
                starts_in_open_channel,
            ),
            dispatch: Gemma4StreamDispatchState::new(starts_in_open_channel),
        }
    }
}

impl StreamEmitter for Gemma4Emitter {
    fn on_token_text(
        &mut self,
        token_text: &str,
        _is_reasoning: bool,
        _include_reasoning: bool,
        sink: &dyn ChunkSink,
    ) {
        let cb = StreamSender(sink);
        let segments = self.parser.feed(token_text);
        self.dispatch.dispatch_segments(segments, &cb);
    }

    fn on_residual(
        &mut self,
        residual: &str,
        _is_reasoning: bool,
        _include_reasoning: bool,
        sink: &dyn ChunkSink,
    ) {
        // Residual flush: feed the leftover bytes through the same parser.
        // The trailing `flush()` lives in `finish` below (the engine calls
        // `finish` unconditionally, so the flush happens whether or not a
        // residual existed — identical segment sequence either way since
        // `dispatch_segments` is stateful-sequential).
        let cb = StreamSender(sink);
        let segments = self.parser.feed(residual);
        self.dispatch.dispatch_segments(segments, &cb);
    }

    fn finish(&mut self, result: &ChatResult, sink: &dyn ChunkSink) {
        let cb = StreamSender(sink);
        let tail = self.parser.flush();
        self.dispatch.dispatch_segments(tail, &cb);
        self.dispatch.finish(&cb);

        // Terminal chunk: text stays empty (segments already streamed);
        // tool_calls/thinking come from the stream parser
        // (`parser.tool_calls()` / `.thinking()`); everything else from the
        // finalized result. `result.finish_reason` already carries the
        // tool_calls promotion from `finalize_turn`, which parses the same
        // raw text the parser does.
        let parsed_tool_calls = self.parser.tool_calls();
        let parsed_thinking = self.parser.thinking();
        cb.call(
            Ok(ChatStreamChunk {
                text: String::new(),
                done: true,
                finish_reason: Some(result.finish_reason.clone()),
                tool_calls: Some(parsed_tool_calls),
                thinking: parsed_thinking,
                thinking_enabled: Some(result.thinking_enabled),
                num_tokens: Some(result.num_tokens),
                prompt_tokens: Some(result.prompt_tokens),
                reasoning_tokens: Some(result.reasoning_tokens),
                raw_text: Some(result.raw_text.clone()),
                public_raw_text: None,
                text_authoritative: Some(false),
                cached_tokens: Some(result.cached_tokens),
                performance: result.performance.clone(),
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );
    }
}

/// Describe Gemma's actually wired media paths separately from inputs that
/// must enter the family backend only to preserve a specific compatibility
/// error.
pub(super) const fn gemma4_media_plan(
    image_components_loaded: bool,
    audio_embedder_loaded: bool,
    paged_adapter_loaded: bool,
) -> MediaPlan {
    let images_available = image_components_loaded && paged_adapter_loaded;
    let audio_available = audio_embedder_loaded && paged_adapter_loaded;
    MediaPlan::with_backend_validation(
        MediaCapabilities {
            images: images_available,
            audio: audio_available,
        },
        MediaCapabilities {
            // Images are admitted unconditionally so the Gemma core — not the engine —
            // raises the "no vision" / "no paged execution" error.
            images: true,
            // Audio is admitted only when its embedder exists; without one the
            // engine rejects it before render.
            audio: audio_embedder_loaded,
        },
    )
}

pub(super) const fn gemma4_image_path_loaded(
    image_processor_loaded: bool,
    vision_projection_loaded: bool,
    standard_vision_tower_loaded: bool,
    unified_vision_embedder_loaded: bool,
    paged_adapter_loaded: bool,
) -> bool {
    image_processor_loaded
        && vision_projection_loaded
        && (standard_vision_tower_loaded || unified_vision_embedder_loaded)
        && paged_adapter_loaded
}

pub(super) const fn gemma4_media_continuable(has_image: bool, has_audio: bool) -> bool {
    has_image && !has_audio
}

pub(super) fn gemma4_session_media_matches_payloads(
    media_session_continuable: bool,
    cached_image_key: Option<u64>,
    cached_audio_key: Option<u64>,
    images: &[Vec<u8>],
    audio: &[Vec<u8>],
) -> bool {
    let image_key = (!images.is_empty()).then(|| engine::compute_image_cache_key(images));
    let audio_key = (!audio.is_empty()).then(|| engine::compute_image_cache_key(audio));
    media_session_continuable
        && !images.is_empty()
        && audio.is_empty()
        && cached_image_key == image_key
        && cached_audio_key == audio_key
}

pub(super) fn gemma4_carries_image_lineage(
    context_media: MediaCapabilities,
    cached_image_key: Option<u64>,
    cached_image_token_positions: &[(u32, u64)],
    cached_token_history: &[u32],
    tokens: &[u32],
) -> bool {
    context_media.images
        && cached_image_key.is_some()
        && !cached_image_token_positions.is_empty()
        && !cached_token_history.is_empty()
        && tokens.starts_with(cached_token_history)
}

/// Request-local metadata parked by the Gemma scheduler beside each owner's
/// physical cache lane. KV arrays remain in the coordinator (paged lane) or
/// `Gemma4SchedulerOwnerState::flat_caches` (speculative lane); this value is
/// intentionally cheap to clone when the scheduler changes the active row.
#[derive(Clone, Default)]
pub(crate) struct Gemma4OwnerMetadata {
    pub(crate) cached_token_history: Vec<u32>,
    pub(super) cached_image_key: Option<u64>,
    pub(super) cached_audio_key: Option<u64>,
    pub(super) cached_paged_image_token_positions: Vec<(u32, u64)>,
    pub(super) media_session_context: MediaCapabilities,
    pub(super) media_session_continuable: bool,
}

#[derive(Default)]
pub(crate) struct Gemma4SchedulerOwnerState {
    pub(crate) metadata: Gemma4OwnerMetadata,
    pub(crate) flat_caches: Option<Vec<Gemma4LayerCache>>,
}

/// Draft-model variant loaded alongside the target for speculative decoding
/// (`Gemma4LoadOptions::draft_model_path`). The kind probe in
/// `persistence.rs` picks the variant from the draft checkpoint's
/// config.json identity fields, then hands the directory to that variant's
/// strict loader.
pub(crate) enum Gemma4Draft {
    /// DeepSpec DSpark external draft: 5-layer cross-attending transformer
    /// drafting whole masked blocks over a fused target-hidden context
    /// ([`crate::models::gemma4::dspark`]).
    Dspark(crate::models::gemma4::dspark::DsparkDraftModel),
    /// Google assistant checkpoint draft: Q-only transformer drafting by
    /// chained single-token AR steps over the target's committed KV caches
    /// ([`crate::models::gemma4::assistant`]).
    Assistant(crate::models::gemma4::assistant::AssistantDraftModel),
}

impl Gemma4Draft {
    /// Checkpoint tensor bytes for cache-limit accounting (see the variant
    /// loaders' `weight_bytes` docs for the measurement contract).
    pub(crate) fn weight_bytes(&self) -> u64 {
        match self {
            Self::Dspark(draft) => draft.weight_bytes(),
            Self::Assistant(draft) => draft.weight_bytes(),
        }
    }

    /// Every checkpoint-backed tensor the draft owns, for the post-load
    /// materialization pass (cheap array-handle clones covering exactly the
    /// applied checkpoint set — byte-coverage pinned per variant).
    pub(crate) fn collect_weight_arrays(&self) -> Vec<MxArray> {
        match self {
            Self::Dspark(draft) => draft.collect_weight_arrays(),
            Self::Assistant(draft) => draft.collect_weight_arrays(),
        }
    }
}
