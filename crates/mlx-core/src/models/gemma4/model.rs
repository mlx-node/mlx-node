use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;

use crate::array::mask::create_causal_mask;
use crate::array::{DType, MxArray};
use crate::engine::backend::{
    ChatBackend, ChunkSink, DecodeStep, FinalizeArgs, PagedBackend, PagedPrefix, PagedTurnSetup,
    ResetScope, SaveStateArgs, StreamEmitter, TurnOutput, TurnSetup, WholeTurnArgs,
};
use crate::engine::cmd::ChatCmd;
use crate::engine::params::ChatParams;
use crate::engine::plan::{
    DecoderPlan, ExecutionPlan, MediaCapabilities, MediaPlan, PagedAttentionPlan, SpeculativeKind,
    SpeculativePlan,
};
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};
use crate::models::gemma4::quantized_linear::LinearProj;
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{SamplingConfig, sample};
use crate::stream::{DeviceType, Stream, StreamContext};
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};
use crate::transformer::paged_kv_cache_adapter::{
    ColdTierContext, PagedKVCacheAdapter, paged_attention_v2_aux_fits,
};
use crate::transformer::rotating_kv_cache::RotatingKVCacheSnapshot;
use crate::transformer::{
    AttentionKind, KVCacheDType, KVCacheGroup, KVCachePhysicalLayout, LayerKVCacheSpec,
    derive_layer_kv_cache_routes, group_layer_kv_cache_specs,
};

use super::image_processor::{Gemma4ImageProcessor, ProcessedGemma4Image};
use super::vision::{Gemma4MultimodalEmbedder, Gemma4VisionModel};
use super::vision_embedder::Gemma4UnifiedVisionEmbedder;
use super::vision_mask::apply_bidirectional_vision_overlay;

use super::attention::{
    Gemma4PagedPrefillRoutePolicy, gemma4_paged_prefill_route_policy,
    gemma4_paged_prefill_v2_layout_for_chunk,
};
use super::config::Gemma4Config;
use super::decoder_layer::{Gemma4DecoderLayer, Gemma4LayerKind};
use super::dspark::DsparkTap;
use super::layer_cache::Gemma4LayerCache;
use super::sliding_sidecar;
use crate::engine;
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use tracing::info;

/// PLE (Per-Layer Embeddings) model-level components.
///
/// Provides per-layer token-level information to each decoder layer.
/// Present in E2B (2.3B) and E4B (4.5B) models.
pub(crate) struct PleComponents {
    /// Embedding table: [vocab_size_per_layer_input, num_layers * ple_dim]
    pub embed_tokens_per_layer: Embedding,
    /// Projection: [hidden_size, num_layers * ple_dim]
    pub per_layer_model_projection: Linear,
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

/// Adapter giving the paged/vision streaming cores a `cb.call(result, mode)`
/// shape over the engine's [`ChunkSink`].
///
/// The engine owns the channel and hands the probes/emitter a `&dyn
/// ChunkSink`, so the wrapper forwards `.call()` to [`ChunkSink::send`].
/// The call mode is meaningless on the mpsc path and is dropped.
struct StreamSender<'a>(&'a dyn ChunkSink);

impl StreamSender<'_> {
    fn call(&self, result: Result<ChatStreamChunk>, _mode: ThreadsafeFunctionCallMode) {
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
struct Gemma4StreamDispatchState {
    pending_reasoning: String,
    visible_text_emitted: bool,
    tool_call_seen: bool,
    starts_in_prompted_channel: bool,
}

impl Gemma4StreamDispatchState {
    fn new(starts_in_prompted_channel: bool) -> Self {
        Self {
            starts_in_prompted_channel,
            ..Self::default()
        }
    }

    fn dispatch_segments(
        &mut self,
        segments: Vec<super::output_parser::StreamSegment>,
        cb: &StreamSender<'_>,
    ) {
        use super::output_parser::StreamSegment;
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
                StreamSegment::ToolCall(_) => {
                    self.tool_call_seen = true;
                    self.flush_pending_reasoning(cb);
                    // Accumulated on `parser.tool_calls()` for the terminal chunk.
                }
            }
        }
    }

    fn finish(&mut self, cb: &StreamSender<'_>) {
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

fn promote_channel_only_output(
    parsed: &mut super::output_parser::Gemma4ParsedOutput,
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
struct Gemma4Emitter {
    parser: super::output_parser::Gemma4StreamParser,
    dispatch: Gemma4StreamDispatchState,
}

impl Gemma4Emitter {
    fn new(starts_in_open_channel: bool) -> Self {
        Self {
            parser: super::output_parser::Gemma4StreamParser::new_with_open_channel(
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

/// Internal model state owned exclusively by the dedicated model thread.
///
/// No `Arc<RwLock<>>` — the model thread has sole ownership.
pub(crate) struct Gemma4Inner {
    pub(crate) config: Gemma4Config,
    pub(crate) embed_tokens: Embedding,
    pub(crate) layers: Vec<Gemma4DecoderLayer>,
    pub(crate) final_norm: RMSNorm,
    pub(crate) lm_head: Option<LinearProj>,
    /// Pre-transposed embedding weight for tied lm_head: [hidden_size, vocab_size].
    /// Only populated when tie_word_embeddings=true.
    pub(crate) embed_weight_t: Option<MxArray>,
    pub(crate) ple: Option<PleComponents>,
    // Vision components (None for text-only models)
    pub(crate) vision_tower: Option<Gemma4VisionModel>,
    /// Encoder-free unified vision embedder. `Some` only for the unified
    /// multimodal checkpoint (`unified_vision_config.is_some()`); mutually
    /// exclusive with `vision_tower` (the SigLIP path).
    pub(crate) unified_vision_embedder: Option<Gemma4UnifiedVisionEmbedder>,
    pub(crate) embed_vision: Option<Gemma4MultimodalEmbedder>,
    /// Encoder-free unified AUDIO embedder. `Some` only when the checkpoint
    /// declares an `audio_config` (`config.has_audio`). Structurally identical
    /// to `embed_vision` (RMSNormNoScale + Linear), but projects raw
    /// 640-sample audio windows (`audio_embed_dim` → `hidden_size`).
    pub(crate) embed_audio: Option<Gemma4MultimodalEmbedder>,
    pub(crate) image_processor: Option<Gemma4ImageProcessor>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    /// Lazily-initialized KV caches that persist across chat turns.
    ///
    /// `None` after construction and after `reset_caches_sync`. Populated
    /// by `init_caches_sync`, triggered on the first turn of a session by
    /// the engine's miss-path `reset_caches(ResetScope::PrefixMiss)` (or
    /// defensively inside [`ChatBackend::prefill`] / the vision cores).
    /// Shared across turns by the session API.
    pub(crate) caches: Option<Vec<Gemma4LayerCache>>,
    /// Tokens (post image-expansion) whose KV state is currently live in
    /// `caches`. Maintained in parallel with `caches` for prefix-reuse
    /// verification in Step 5c. Empty when no session is active.
    pub(crate) cached_token_history: Vec<u32>,
    /// Content hash of the image set associated with the live cache. Used
    /// in Step 5c to detect mid-session image changes (which require a
    /// full session restart). Preserved with the ordered image-position
    /// sidecar across successful warm text saves so subsequent registrations
    /// retain the exact image-aware cache lineage.
    pub(crate) cached_image_key: Option<u64>,
    /// Content hash of the audio set associated with the live cache. Audio
    /// counterpart of `cached_image_key`: set after an audio prefill so a
    /// follow-up text delta is rejected (the continue path is text-only) and
    /// a follow-up audio turn cold-restarts. Like the image key, this is
    /// cleared after a warm text save even though the live media KV remains;
    /// `media_session_context` is the persistent source of truth.
    pub(crate) cached_audio_key: Option<u64>,
    /// Ordered absolute image-placeholder positions paired with all four words
    /// of their SHA-256 image digest for the media lineage currently represented
    /// by the live/persisted paged request. Text continuations preserve this
    /// sidecar so every later registration uses the same image-aware per-block
    /// keys instead of republishing image K/V under token-only hashes.
    cached_paged_image_token_positions: Vec<(u32, u64)>,
    /// Block-paged KV adapter (vLLM-style refcounted prefix cache).
    ///
    /// **Opt-in via `Gemma4Config::use_block_paged_cache`**. Gemma4's
    /// hybrid sliding+global attention, K=V sharing, KV-shared layers
    /// (`forward_shared`), MoE/PLE branches, and per-layer-type head
    /// dimensions are all handled by
    /// `Gemma4DecoderLayer::forward_paged_or_flat`, which routes only
    /// global attention layers through this adapter. Defaults to `None`
    /// when the config flag is unset, in which case the model falls
    /// back to the flat `Gemma4LayerCache` path.
    pub(crate) paged_adapter: Option<PagedKVCacheAdapter>,
    /// Draft model for speculative decoding (`Gemma4LoadOptions::
    /// draft_model_path`), either [`Gemma4Draft`] variant. Mutually
    /// exclusive with `paged_adapter`: the load path hard-errors on an
    /// explicit `use_block_paged_cache: true` conflict and forces the unset
    /// default to flat, so `draft.is_some()` implies
    /// `paged_adapter.is_none()`.
    pub(crate) draft: Option<Gemma4Draft>,
    /// Per-turn draft handoff: the whole-turn core builds the variant's
    /// prefill-derived state (DSpark fused-context cache / assistant
    /// last-prompt hidden) during prefill and stashes it here;
    /// `DsparkBackend::begin_dspark_decode` TAKES it into the turn's
    /// stepper (the engine's `DsparkTurnSetup` carries only turn constants,
    /// so prefill-derived state travels through this seam). Always `None`
    /// outside a live draft whole-turn.
    pub(crate) draft_turn_state: Option<super::dspark_decode::Gemma4DraftTurnState>,
    /// Cached result of `compute_layer_kinds_from_kv_cache_specs(&config)`,
    /// computed once here in `Gemma4Inner::new` instead of re-derived
    /// (BTreeMap/BTreeSet grouping + a sort) on every paged prefill-chunk /
    /// decode-step call. Pure function of the immutable `config`, so it
    /// never changes for the lifetime of this instance. Empty when
    /// `paged_adapter` is `None`: every paged-only call site that reads it
    /// errors out on a `None` adapter before consuming the value.
    pub(crate) layer_kinds: Vec<Gemma4LayerKind>,
    sliding_prefix_checkpoints: VecDeque<Gemma4SlidingPrefixCheckpoint>,
    sliding_prompt_boundary_checkpoint: Option<Gemma4SlidingPrefixCheckpoint>,
    /// Sliding state at [`gemma4_cold_restore_reachable_boundary`] for the
    /// prompt of the turn currently in flight, when that boundary differs from
    /// `sliding_prompt_boundary_checkpoint`'s — i.e. exactly when the prompt
    /// length is a multiple of the block size and the prompt boundary sits one
    /// block past anything a restore can name.
    ///
    /// Deliberately NOT a member of `sliding_prefix_checkpoints`, and read by
    /// nothing but [`Gemma4Inner::find_gemma4_sliding_capture_checkpoints`]:
    ///
    ///  * the retained SET decides which checkpoint a later warm turn resumes
    ///    from, and that is observable in the emitted tokens, so a persist turn
    ///    must retain exactly what a persistence-OFF turn retains. Keeping this
    ///    outside the deque means the ladder, its limit and its eviction order
    ///    are bit-for-bit what they were;
    ///  * the ladder's eviction rule takes the oldest NON-anchor first, and this
    ///    boundary is not on the `block_size * 4^k` grid, so an entry in the
    ///    deque would be the preferred victim of the very next push — while it
    ///    is the one boundary the turn's own capture most wants.
    ///
    /// Written only on a turn whose caps say [`Gemma4SlidingRetentionCaps::
    /// wants_ladder`] (a persistence-OFF turn never allocates it), cleared at
    /// the start of every prefill body that could publish it, so at most one
    /// window of sliding K/V is held for it.
    sliding_cold_restore_tail_checkpoint: Option<Gemma4SlidingPrefixCheckpoint>,
    /// Token length of the prompt the paged text turn in flight planned, i.e.
    /// `PagedBackend::prime_prefix_state`'s `plan.len()`.
    ///
    /// The capture at finalize runs over `adapter.request_tokens()`, which is
    /// the prompt PLUS everything generated, and so cannot tell where the
    /// prompt ended. It needs to, because the bound it must respect is
    /// [`gemma4_cold_restore_reachable_boundary`] of the PROMPT — a later
    /// restore of this same prompt looks up `prompt[..prompt_len - 1]`, and no
    /// amount of generated tail widens that.
    ///
    /// `0` outside a paged text turn, which fails the capture closed (no
    /// reachable boundary, so no sidecar) rather than open.
    paged_turn_prompt_len: u32,
    sliding_last_history_checkpoint: Option<Gemma4SlidingHistoryCheckpoint>,
    /// Media kinds causally represented by the current session's live/persisted
    /// prefix. This survives every successful warm text continuation because
    /// those turns extend — rather than replace — the media-derived KV. Cleared
    /// when that session is reset, invalidated, or successfully replaced.
    media_session_context: MediaCapabilities,
    /// Context handed to the currently executing generic paged text turn.
    /// `run_paged_turn` snapshots `TurnPlan::context_media` here so
    /// `save_paged_history` can distinguish a warm media continuation from a
    /// fresh text replacement without widening the model-neutral trait.
    paged_text_turn_context: MediaCapabilities,
    /// True only while a pure image turn left its
    /// global paged KV live AND a sliding history checkpoint remembered at the
    /// full kept-live prefix, so a full-history continuation can reuse
    /// the live media KV causally. Set exclusively by
    /// `finalize_vision_turn_media_state` on the continuable branch; reset to
    /// `false` at every non-continuable point (`clear_reuse_state`, both vision
    /// prefill-start blocks, the non-continuable finalize).
    media_session_continuable: bool,
    /// `PagedBackend::finalize_paged_turn` is infallible at the trait seam, but
    /// Gemma's per-block registration is not. Latch a failure here so the
    /// immediately-following fallible `save_paged_history` refuses to publish
    /// token/sliding history and lets the engine reset the failed session.
    paged_finalize_failed: bool,
    /// True when this turn's rendered prompt ends inside
    /// `<|channel>thought\n`. The generated suffix then begins at the
    /// reasoning body, so both sync and streaming output parsers must start
    /// in `Channel` rather than `Message`. Every render entry point overwrites
    /// the latch before decode; the dedicated model thread serializes turns.
    output_starts_in_reasoning_channel: AtomicBool,
    pub(crate) model_id: u64,
}

/// Describe Gemma's actually wired media paths separately from inputs that
/// must enter the family backend only to preserve a specific compatibility
/// error.
const fn gemma4_media_plan(
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
            // Image input was historically admitted unconditionally so the
            // Gemma core could distinguish missing vision from missing paged
            // execution. Keep that family-owned diagnostic.
            images: true,
            // Audio was historically admitted only when its embedder existed.
            // With no paged adapter, the family core owns the compatibility
            // error; with no embedder, the engine rejects it before render.
            audio: audio_embedder_loaded,
        },
    )
}

const fn gemma4_image_path_loaded(
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

const fn gemma4_media_continuable(has_image: bool, has_audio: bool) -> bool {
    has_image && !has_audio
}

const fn gemma4_vlm_prefix_checkpoint_eligible(
    has_image: bool,
    has_audio: bool,
    reuse_cache: bool,
) -> bool {
    has_image && !has_audio && reuse_cache
}

fn gemma4_carries_image_lineage(
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

/// Draft-model variant loaded alongside the target for speculative decoding
/// (`Gemma4LoadOptions::draft_model_path`). The kind probe in
/// `persistence.rs` picks the variant from the draft checkpoint's
/// config.json identity fields, then hands the directory to that variant's
/// strict loader.
pub(crate) enum Gemma4Draft {
    /// DeepSpec DSpark external draft: 5-layer cross-attending transformer
    /// drafting whole masked blocks over a fused target-hidden context
    /// ([`super::dspark`]).
    Dspark(super::dspark::DsparkDraftModel),
    /// Google assistant checkpoint draft: Q-only transformer drafting by
    /// chained single-token AR steps over the target's committed KV caches
    /// ([`super::assistant`]).
    Assistant(super::assistant::AssistantDraftModel),
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

/// Gemma 4 dense language model.
///
/// Supports E2B (2.3B), E4B (4.5B), and 31B variants.
/// Features: hybrid attention (sliding + global), GeGLU MLP, logit softcapping,
/// embedding scaling, and optional per-layer embeddings.
///
/// All model state lives on a dedicated OS thread. NAPI methods dispatch
/// commands via channels and await responses.
#[napi]
pub struct Gemma4Model {
    /// Dedicated model thread owning `Gemma4Inner`. `None` when the model
    /// was constructed via `new(config)` without loading weights — in that
    /// uninitialized state every session method returns an error and
    /// only `isInitialized` is meaningful. Mirrors the same `Option<..>`
    /// gate used by the OCR models (`VLModel`, `QianfanOCRModel`).
    ///
    /// Gemma4 is chat-only (no training/generate variants), so the
    /// thread dispatches the model-neutral [`ChatCmd`] directly via
    /// `engine::cmd::handle_chat_cmd::<Gemma4Inner>` — no per-family
    /// command enum.
    pub(crate) thread: Option<crate::model_thread::ModelThread<ChatCmd>>,
    pub(crate) model_id: u64,
    /// Whether the loaded config includes `vision_config`. Mirrored here so
    /// the NAPI side can fail fast on image inputs to a text-only model
    /// without round-tripping to the model thread. The actual image
    /// processor lives on `Gemma4Inner` and runs on the model thread.
    pub(crate) has_vision: bool,
    /// Whether the loaded config declares an `audio_config` (unified Gemma 4
    /// audio support, `Gemma4Config::has_audio`). Mirrored here so the NAPI
    /// image-guard can fail fast on audio inputs to a model with no audio
    /// support without round-tripping to the model thread.
    pub(crate) has_audio: bool,
    /// Whether the model was loaded with real weights. `false` for
    /// `new Gemma4Model(config)` calls that never called `load()`.
    /// Session methods check this and refuse to dispatch when false,
    /// since the coordinator was never told about this model's delta
    /// (its guard is `None`) — running inference on that stub would
    /// under-cap the allocator.
    pub(crate) initialized: bool,
    /// Snapshot of `Gemma4Inner::paged_adapter.is_some()` captured at
    /// construction time. Default-OFF on Gemma4 (parity-blocked — see
    /// `Gemma4Config::use_block_paged_cache` and the WIP per-layer
    /// numerical-diff tracker), so this is `false` for the entire matrix
    /// of currently-shipping configs. Stubs from `new(config)` always
    /// report `false` because no inner was constructed. Surfaced through
    /// the `hasBlockPagedCache()` NAPI method so server endpoints can
    /// branch on it without round-tripping through the model thread.
    pub(crate) paged_active: bool,
    /// RAII: unregisters this model's delta from the cache-limit
    /// coordinator on drop. `None` for instances constructed via the
    /// synchronous `new(config)` path that never loaded weights.
    pub(crate) _cache_limit_guard: Option<crate::cache_limit::CacheLimitGuard>,
    /// Snapshot of `Gemma4Inner::draft.is_some()` captured at load time
    /// (same mirroring pattern as `paged_active`): whether a draft model —
    /// either [`Gemma4Draft`] variant — was loaded via
    /// `Gemma4LoadOptions::draft_model_path` or discovered in the target's
    /// `draft/` directory. Surfaced through the
    /// `hasMtpWeights()` NAPI method (named for parity with the Qwen3.5
    /// surface) so server endpoints can branch without a model-thread
    /// roundtrip. Stubs from `new(config)` always report `false`.
    pub(crate) draft_active: bool,
}

/// Optional load-time settings for [`Gemma4Model::load`].
#[napi(object)]
#[derive(Debug, Clone, Default)]
pub struct Gemma4LoadOptions {
    /// Directory of a draft checkpoint (config.json + safetensors) to load
    /// alongside the target model for speculative decoding — either a
    /// DSpark draft or a Google assistant draft; the kind is probed from
    /// the draft config.json. When omitted, `<model_path>/draft/` is loaded
    /// automatically when present. Draft decoding runs only on the flat
    /// KV-cache path: setting this while the model config explicitly enables
    /// `use_block_paged_cache` is a hard load error, and an unset
    /// `use_block_paged_cache` is forced to `false`.
    pub draft_model_path: Option<String>,
}

static MODEL_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Classification of the prefix-cache decision made from a
/// [`Gemma4Inner::verify_cache_prefix`] return value plus the incoming
/// token count.
///
/// Test-only mirror of the reset-or-reuse branch the engine session
/// core (`engine::session::chat_turn_core`) takes from this backend's
/// `verify_cache_prefix` return — separating the decision
/// logic from the native state mutation so the "exact-match routes to
/// miss" invariant can be pinned by pure-logic unit tests that do not
/// require a loaded Gemma4 model. Production code keeps the inlined
/// form for zero-overhead dispatch; this enum exists solely to drive
/// `prefix_cache_decision_tests`'s four-case coverage (empty cache,
/// strict-extend hit, divergence miss, exact-match miss). Any change
/// to the inlined production branch MUST be mirrored here or the test
/// ceases to guard the real code.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum PrefixCacheDecision {
    /// Strict-extend hit: the new prompt begins with the cached prefix
    /// and carries additional delta tokens. Warm-reuse safe: skip the
    /// cached prefix and prefill only the tail.
    StrictExtendHit,
    /// Cache miss — covers three sub-cases that all dispatch through
    /// the same `reset_caches_sync` + `init_caches_sync` + full-prefill
    /// branch:
    /// * `cached_prefix_len == 0` (no prior cache or verifier rejected
    ///   the prefix overlap for any reason).
    /// * `cached_prefix_len == tokens_len` (exact-match) — routed to
    ///   miss because Gemma4 has no snapshot of final-step logits and
    ///   no cheap rewind primitive for its sliding-window cache.
    Miss,
}

const GEMMA4_SLIDING_PREFIX_CHECKPOINT_MIN_LIMIT: usize = 16;
const GEMMA4_SLIDING_PREFIX_CHECKPOINT_WINDOW_MULTIPLIER: usize = 2;
const GEMMA4_SLIDING_PREFIX_CHECKPOINT_MAX_DEFAULT_LIMIT: usize = 128;
const GEMMA4_IMAGE_PREFIX_CHECKPOINT_LIMIT: usize = 2;
const GEMMA4_SLIDING_CHECKPOINT_MEMORY_BUDGET_BYTES: u64 = 1024 * 1024 * 1024;
const GEMMA4_PAGED_CACHE_MIN_DEFAULT_MEMORY_MB: u32 = 256;
const BYTES_PER_MIB: u64 = 1024 * 1024;

/// Spacing of the cold-sidecar anchor rungs, as a multiple of the paged block
/// size: `block_size * RATIO^k`, ascending.
///
/// Anchored at ZERO, not at the prompt end — this is the one place gemma4 must
/// differ from qwen3.5's `gdn_prefill_checkpoint_boundaries`, which walks
/// `deepest / 4^k` DOWN from the prompt. A rung's cold key is the block chain
/// over `tokens[0..b]`, so a grid pinned to 0 makes the SAME sidecar object
/// reusable by every later turn (and every later process) whose prompt shares
/// that prefix. A prompt-anchored ladder would land on 112/496/2032 for one
/// prompt and 128/512/2048 for the next and never dedup — and `mlx agent`,
/// which is what this whole path exists for, sends a slightly different prompt
/// every turn.
const GEMMA4_SLIDING_ANCHOR_RATIO: u32 = 4;

/// How many anchor rungs a ladder may hold, before the byte budget below
/// trims it further. Mirrors `GDN_CHECKPOINT_LADDER_RUNGS`.
///
/// With `block_size = 16` this is `{64, 256, 1024, 4096}` — two rungs BELOW
/// gemma4's 1024-token window (where a payload is `min(b, window)` rows and so
/// nearly free) and two at or above it (a full window each). Without the cap a
/// small checkpoint whose full-window payload is a few MiB would keep admitting
/// rungs until the budget ran out, hundreds of them.
const GEMMA4_SLIDING_ANCHOR_MAX_RUNGS: usize = 4;

/// Byte ceiling for the WHOLE retained set on the ladder arm — the anchors plus
/// the pre-ladder reserve — at the same conservative 4 bytes/element
/// [`gemma4_sliding_checkpoint_estimated_bytes`] uses.
///
/// Measured against the checkpoint that found this bug
/// (`Gemma-4-12B-IT-nvidia-mxfp-mlx`: 40 physical sliding layers, window 1024,
/// 8 kv heads, head_dim 256, `block_size` 16):
///
/// ```text
///   full window            671.1 MB    pre-ladder reserve (2 slots)  1342.2 MB
///   rung   64   41.9 MB    rung  256   167.8 MB
///   rung 1024  671.1 MB    rung 4096   671.1 MB   (payload capped at a window)
///   -> anchors 1551.9 MB + reserve 1342.2 MB = 2894.1 MB  <= 3072 MB
///   -> a fifth rung at 16384 would need 3565.2 MB and is refused
/// ```
///
/// Actual bf16 residency is half of that (~1.4 GB), and only on a turn that
/// writes a sidecar. The 4 bytes/element figure is kept because the snapshot
/// type still promises no dtype; see the estimator's own comment.
const GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES: u64 = 3 * 1024 * 1024 * 1024;

/// Which entry an over-limit [`trim_gemma4_sliding_prefix_checkpoints`] evicts.
///
/// Both arms move the same observable — the depth a later turn resumes from —
/// so both answer to the same `want_ladder` predicate that decides whether
/// anchor rungs are published at all
/// ([`Gemma4Inner::gemma4_sliding_cold_ladder_wanted`]). Retention and the
/// published rung set must not disagree about whether this is a persist turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma4SlidingRetentionPolicy {
    /// The victim order this store used before anchor rungs existed: the first
    /// entry that is not an image-protected prompt boundary, i.e. the OLDEST
    /// (shallowest) text checkpoint.
    ///
    /// Restored verbatim for a turn that publishes no ladder. That is a
    /// compatibility contract, not an optimization: which checkpoint a later
    /// warm turn lands on decides whether
    /// `prepare_gemma4_sliding_prefix` takes its `prefix_checkpoint` arm (install
    /// a snapshot) or its `replay` arm (re-forward the whole cached prefix
    /// through `run_sliding_only_prefill`). Those are different spans of
    /// arithmetic in a different order, so they can emit different tokens. A
    /// persistence-OFF request gets nothing back for that risk, so it must not
    /// take it — the same lesson `GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER`
    /// records, where the divergence was measured at character 56.
    PreLadder,
    /// Ladder-aware: evict non-anchors first, oldest first, so the SHALLOW
    /// rungs survive a prefill that keeps ratcheting deeper cadence
    /// checkpoints in behind them. Those shallow rungs are the only entries a
    /// cold capture can anchor on while the persisted K/V chain still lags the
    /// prompt, and the chain advances only one writer-queue's worth of blocks
    /// per turn.
    ///
    /// Anchors are deferred, never permanently protected: once no non-anchor is
    /// left, the first anchor that is NOT an ancestor of the newest entry goes,
    /// which is what stops a finished conversation's rungs from squatting after
    /// a lineage switch.
    Ladder,
}

/// The anchor rung grid, inline so [`Gemma4SlidingRetentionCaps`] stays `Copy`
/// and every consumer reads the SAME grid without an allocation or a borrow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Gemma4SlidingAnchorRungs {
    rungs: [u32; GEMMA4_SLIDING_ANCHOR_MAX_RUNGS],
    len: usize,
}

impl Gemma4SlidingAnchorRungs {
    fn from_slice(rungs: &[u32]) -> Self {
        let mut inline = Self::default();
        for &rung in rungs.iter().take(GEMMA4_SLIDING_ANCHOR_MAX_RUNGS) {
            inline.rungs[inline.len] = rung;
            inline.len += 1;
        }
        inline
    }

    fn as_slice(&self) -> &[u32] {
        &self.rungs[..self.len]
    }

    fn contains(&self, boundary: u32) -> bool {
        self.as_slice().contains(&boundary)
    }
}

/// What one retained checkpoint costs, as a function of its boundary — inline
/// so [`Gemma4SlidingRetentionCaps`] stays `Copy`.
///
/// A checkpoint at `boundary` holds `min(boundary, window)` token rows, so the
/// cost of a retained SET is NOT `count * full_window`: across the rungs this
/// ladder publishes it varies 16x on the 12B geometry (41.9 MB at 64 tokens,
/// 671.1 MB at 1024). That is precisely why the entry COUNT
/// [`gemma4_sliding_retention_caps_for_override`] derives from
/// [`GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES`] is not a cap on bytes: it is
/// derived from a PLANNED mix of shallow rungs and deep entries, and nothing
/// forces the retained set to be that mix. Once the cursor is deep every
/// retained entry is a full window, and six of those are 4026 MB against a
/// declared 3072 MB ceiling.
///
/// Overrunning here is not "a cache tier degrades". MLX targets unified memory
/// (see `docs/architecture.md`): weights, the paged KV pool and these
/// checkpoints draw on ONE physical budget, so the extra gigabyte comes
/// straight out of the pool and the weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Gemma4SlidingCheckpointBytes {
    full_window_bytes: u64,
    window_tokens: u32,
}

impl Gemma4SlidingCheckpointBytes {
    fn for_config(config: &Gemma4Config) -> Self {
        Self {
            full_window_bytes: gemma4_sliding_checkpoint_estimated_bytes(config),
            window_tokens: config.sliding_window.max(0) as u32,
        }
    }

    /// Conservative bytes a checkpoint at `boundary_tokens` occupies. Zero for
    /// a geometry with no sliding state at all (all-global, or window 0), which
    /// is also what disables the byte cap in
    /// [`trim_gemma4_sliding_prefix_checkpoints`] — there is nothing to bound.
    fn at(&self, boundary_tokens: u32) -> u64 {
        if self.window_tokens == 0 {
            return 0;
        }
        let window = u64::from(self.window_tokens);
        let rows = u64::from(boundary_tokens).min(window);
        self.full_window_bytes / window * rows
    }

    /// Total for a retained set. Saturating: a bogus geometry must not wrap
    /// the sum into "fits".
    fn total<'a>(
        &self,
        checkpoints: impl IntoIterator<Item = &'a Gemma4SlidingPrefixCheckpoint>,
    ) -> u64 {
        checkpoints.into_iter().fold(0u64, |sum, checkpoint| {
            sum.saturating_add(self.at(checkpoint.prefix_len))
        })
    }
}

/// Everything one prefill's checkpoint bookkeeping answers to: how many entries
/// survive, which one an eviction takes, and where the anchor rungs are.
///
/// ```text
///   want_ladder  ->  limit                          policy     anchors
///   false            gemma4_sliding_prefix_checkpoint_limit_for_override
///                    (unchanged; 2 on a 12B)        PreLadder  none
///   true             that + anchor rung count
///                    (6 on a 12B)                   Ladder     {64,256,1024,4096}
/// ```
///
/// All three travel together on purpose, and the rungs live here rather than at
/// the prefill call site for a specific reason: whether a boundary is PUBLISHED
/// ([`gemma4_sliding_chunk_checkpoint_boundaries`]), whether the entry it
/// produces is MARKED an anchor, and whether retention then defers it must be
/// three readings of one fact. Threading them separately is how the qwen3.5
/// ladder shipped with a prefill body that published rungs nothing retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Gemma4SlidingRetentionCaps {
    limit: usize,
    policy: Gemma4SlidingRetentionPolicy,
    anchors: Gemma4SlidingAnchorRungs,
    /// Per-entry byte cost, so retention can bound the set in BYTES and not
    /// only in entries.
    ///
    /// Carried on BOTH arms, and deliberately so. A persistence-OFF turn must
    /// retain exactly what it retained before the ladder existed — a byte cap
    /// that fired there would move which checkpoint a later warm turn resumes
    /// from, and that changes emitted tokens. What stops it is `policy`, the
    /// same single predicate that decides everything else about this turn. If
    /// this field were zeroed on the `PreLadder` arm instead, the guard would
    /// have a second, silent reason to hold and no test could tell whether the
    /// real one still works.
    bytes: Gemma4SlidingCheckpointBytes,
}

impl Gemma4SlidingRetentionCaps {
    fn pre_ladder(limit: usize, bytes: Gemma4SlidingCheckpointBytes) -> Self {
        Self {
            limit,
            policy: Gemma4SlidingRetentionPolicy::PreLadder,
            anchors: Gemma4SlidingAnchorRungs::default(),
            bytes,
        }
    }

    fn ladder(
        limit: usize,
        anchors: Gemma4SlidingAnchorRungs,
        bytes: Gemma4SlidingCheckpointBytes,
    ) -> Self {
        Self {
            limit,
            policy: Gemma4SlidingRetentionPolicy::Ladder,
            anchors,
            bytes,
        }
    }

    /// Whether this turn publishes and defers anchor rungs — the one predicate,
    /// read by the publish seam and the retention seam alike.
    fn wants_ladder(&self) -> bool {
        self.policy == Gemma4SlidingRetentionPolicy::Ladder
    }
}

/// A decode cursor that publishes a sliding checkpoint, and everything the
/// publishing step needs to know about it. Produced only by
/// [`gemma4_sliding_decode_boundary_plan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Gemma4SlidingDecodeBoundary {
    prefix_len: u32,
    block_size: u32,
    checkpoint_interval: u32,
    /// Trace-only: whether this boundary is one of the turn's anchor rungs.
    ///
    /// Named differently from the stored entry's `cold_anchor_rung` on purpose.
    /// The stored flag has exactly one writer,
    /// [`Gemma4SlidingPrefixCheckpointDraft::into_checkpoint`] — pinned by
    /// `gemma4_sliding_anchor_flag_has_exactly_one_writer` — and a second field
    /// spelled the same way would both blunt that guard and invite someone to
    /// route this value into the store. This one only ever reaches a trace line.
    on_anchor_rung: bool,
}

#[derive(Clone)]
struct Gemma4SlidingPrefixCheckpoint {
    prefix_len: u32,
    block_size: u32,
    final_block_hash: u64,
    protected_image_prompt_boundary: bool,
    /// This entry sits on a [`gemma4_sliding_cold_anchor_rungs`] rung, i.e. it
    /// was published FOR the cold sidecar rather than for the warm in-process
    /// path. Retention under [`Gemma4SlidingRetentionPolicy::Ladder`] evicts
    /// non-anchors first, because an anchor is the only kind of entry a cold
    /// capture can use while the persisted K/V chain still lags the prompt.
    ///
    /// Always `false` on the pre-ladder arm: nothing publishes a rung there,
    /// and `PreLadder` never reads this flag.
    cold_anchor_rung: bool,
    tokens: Vec<u32>,
    snapshots: Vec<Option<RotatingKVCacheSnapshot>>,
}

/// Everything a publishing site honestly knows about a checkpoint it just
/// produced. `cold_anchor_rung` is deliberately NOT among those fields.
///
/// d134ab3e claimed the flag was derived "from the same caps that gate
/// publishing, so a publishing site cannot forget to mark a rung". That was
/// false: two of the four publish sites open-coded the deque push and hard-coded
/// the flag `false`, and two more passed a dead `false` through the upsert. Four
/// identical literals, two load-bearing, and a genuine rung born with the flag
/// clear is the ladder's PREFERRED eviction victim — the exact "born then
/// evicted" failure the ladder exists to fix.
///
/// A draft cannot be pushed into the store, and
/// [`Gemma4SlidingPrefixCheckpointDraft::into_checkpoint`] is the only place in
/// the file that writes the flag, so the claim is now structural rather than
/// aspirational.
#[derive(Clone)]
struct Gemma4SlidingPrefixCheckpointDraft {
    prefix_len: u32,
    block_size: u32,
    final_block_hash: u64,
    protected_image_prompt_boundary: bool,
    tokens: Vec<u32>,
    snapshots: Vec<Option<RotatingKVCacheSnapshot>>,
}

/// Where a checkpoint captured inside a prefill compute chunk is filed.
///
/// The distinction is not bookkeeping: `PrefixStore` is the RETAINED set, and
/// which entry a later warm turn resumes from decides the tokens it emits, so a
/// persist turn may not put anything there a persistence-OFF turn would not.
/// `ColdRestoreTail` is read by the cold sidecar capture and by nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma4SlidingCapturedCheckpointSink {
    /// `sliding_prefix_checkpoints`, under this turn's retention caps.
    PrefixStore,
    /// `sliding_cold_restore_tail_checkpoint`, outside the retained set.
    ColdRestoreTail,
}

impl Gemma4SlidingPrefixCheckpointDraft {
    /// Derive the stored entry. The grid that decided this boundary was
    /// published is the same grid that decides retention defers it, so the two
    /// readings come from one `caps` value.
    fn into_checkpoint(self, caps: Gemma4SlidingRetentionCaps) -> Gemma4SlidingPrefixCheckpoint {
        Gemma4SlidingPrefixCheckpoint {
            cold_anchor_rung: caps.wants_ladder() && caps.anchors.contains(self.prefix_len),
            prefix_len: self.prefix_len,
            block_size: self.block_size,
            final_block_hash: self.final_block_hash,
            protected_image_prompt_boundary: self.protected_image_prompt_boundary,
            tokens: self.tokens,
            snapshots: self.snapshots,
        }
    }
}

struct Gemma4SlidingHistoryCheckpoint {
    tokens: Vec<u32>,
    image_token_positions: Vec<(u32, u64)>,
    snapshots: Vec<Option<RotatingKVCacheSnapshot>>,
}

struct Gemma4SlidingPrefixCheckpointHit {
    prefix_len: u32,
    caches: Vec<Gemma4LayerCache>,
}

#[derive(Default)]
struct Gemma4SlidingCheckpointStoreTrace {
    stored: bool,
    eval_ms: f64,
    snapshot_ms: f64,
    token_clone_ms: f64,
    update_ms: f64,
    total_ms: f64,
}

impl Gemma4SlidingCheckpointStoreTrace {
    fn finish(mut self, start: Option<std::time::Instant>) -> Self {
        self.total_ms = start.map(elapsed_ms).unwrap_or(0.0);
        self
    }
}

struct Gemma4SlidingPrefixPreparation {
    state: &'static str,
    primed_prefix_len: u32,
}

struct Gemma4VlmTurnPreparation {
    cached_prefix_len: u32,
    suffix_embeds: MxArray,
    layer_kinds: Vec<Gemma4LayerKind>,
    extra_keys_per_block: Vec<Vec<u64>>,
    publish_prefix_checkpoints: bool,
}

/// Explicit capture identity for Gemma4's out-of-pool sliding state.
///
/// Text turns still source their prompt length from the generic paged lifecycle,
/// but VLM turns bypass that lifecycle entirely. Carrying the VLM prompt length
/// and ordered image positions here prevents media capture from accidentally
/// reading the text-only `paged_turn_prompt_len` ambient field (which is zero on
/// this path) or stale image lineage from a prior turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Gemma4SlidingColdCaptureContext<'a> {
    prompt_len: u32,
    image_token_positions: &'a [(u32, u64)],
    media: Gemma4SlidingColdCaptureMedia,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma4SlidingColdCaptureMedia {
    Text,
    PureImage,
}

impl<'a> Gemma4SlidingColdCaptureContext<'a> {
    fn text(prompt_len: u32, image_token_positions: &'a [(u32, u64)]) -> Self {
        Self {
            prompt_len,
            image_token_positions,
            media: Gemma4SlidingColdCaptureMedia::Text,
        }
    }

    fn pure_image(prompt_len: u32, image_token_positions: &'a [(u32, u64)]) -> Self {
        Self {
            prompt_len,
            image_token_positions,
            media: Gemma4SlidingColdCaptureMedia::PureImage,
        }
    }

    /// First boundary this capture mode may persist.
    ///
    /// Text behavior stays byte-for-byte conservative: a generic text turn that
    /// still carries image lineage remains unsupported, matching the old blanket
    /// media guard. A native pure-image turn must anchor strictly after every
    /// expanded image placeholder. `checked_add` makes an unrepresentable
    /// exclusive endpoint fail closed.
    fn minimum_safe_boundary(self) -> Option<u32> {
        match self.media {
            Gemma4SlidingColdCaptureMedia::Text => {
                self.image_token_positions.is_empty().then_some(0)
            }
            Gemma4SlidingColdCaptureMedia::PureImage => self
                .image_token_positions
                .iter()
                .map(|(position, _)| *position)
                .max()?
                .checked_add(1),
        }
    }
}

const fn gemma4_sliding_cold_sidecar_matches_prefix(boundary: u32, cached_prefix_len: u32) -> bool {
    boundary > 0 && boundary == cached_prefix_len
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Gemma4VlmPrefixPolicy {
    unified_boundary_safe: bool,
    require_exact_checkpoint: bool,
    may_replay_leading_text: bool,
}

fn gemma4_vlm_prefix_policy(
    candidate_cached_prefix_len: u32,
    first_image_position: Option<u32>,
    unified_overlay_last_image_exclusive: Option<u32>,
) -> Gemma4VlmPrefixPolicy {
    let unified_boundary_safe =
        unified_overlay_last_image_exclusive.is_none_or(|last_image_exclusive| {
            candidate_cached_prefix_len == 0 || candidate_cached_prefix_len >= last_image_exclusive
        });
    let candidate_crosses_image = first_image_position
        .is_some_and(|first_image_position| candidate_cached_prefix_len > first_image_position);
    let require_exact_checkpoint =
        unified_overlay_last_image_exclusive.is_some() || candidate_crosses_image;
    Gemma4VlmPrefixPolicy {
        unified_boundary_safe,
        require_exact_checkpoint,
        may_replay_leading_text: unified_boundary_safe
            && !require_exact_checkpoint
            && candidate_cached_prefix_len > 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn gemma4_vlm_prefill_chunk_end(
    pass1_position: u32,
    pass1_end: u32,
    configured_chunk_size: i32,
    overlay_active: bool,
    leading_text_checkpoint_boundary: u32,
    prompt_checkpoint_boundary: u32,
    last_image_exclusive: Option<u32>,
) -> u32 {
    let first_overlay_chunk = overlay_active && pass1_position == 0;
    let default_chunk_end = if first_overlay_chunk {
        let safe_boundary = prompt_checkpoint_boundary;
        if safe_boundary >= last_image_exclusive.unwrap_or(u32::MAX) && safe_boundary < pass1_end {
            safe_boundary
        } else {
            pass1_end
        }
    } else if configured_chunk_size > 0 {
        pass1_position
            .saturating_add(configured_chunk_size as u32)
            .min(pass1_end)
    } else {
        pass1_end
    };

    let mut chunk_end = default_chunk_end;
    for boundary in [leading_text_checkpoint_boundary, prompt_checkpoint_boundary] {
        if first_overlay_chunk && boundary < last_image_exclusive.unwrap_or(u32::MAX) {
            continue;
        }
        if boundary > pass1_position && boundary < chunk_end {
            chunk_end = boundary;
        }
    }
    chunk_end
}

struct Gemma4PagedTurnPreparation {
    cached_prefix_len: u32,
    suffix_len: u32,
    sliding_primed_prefix_len: u32,
}

#[cfg(test)]
fn compute_gemma4_paged_prefix_block_hash(
    tokens: &[u32],
    prefix_len: u32,
    block_size: u32,
    cache_salt: u64,
) -> Option<u64> {
    let empty_extra_keys = vec![Vec::new(); (prefix_len / block_size.max(1)) as usize];
    compute_gemma4_paged_prefix_block_hash_with_keys(
        tokens,
        prefix_len,
        block_size,
        &empty_extra_keys,
        cache_salt,
    )
}

fn compute_gemma4_paged_prefix_block_hash_with_keys(
    tokens: &[u32],
    prefix_len: u32,
    block_size: u32,
    extra_keys_per_block: &[Vec<u64>],
    cache_salt: u64,
) -> Option<u64> {
    if prefix_len == 0 || block_size == 0 || !prefix_len.is_multiple_of(block_size) {
        return None;
    }

    let prefix_len = prefix_len as usize;
    let block_size = block_size as usize;
    if prefix_len > tokens.len() {
        return None;
    }

    let num_blocks = prefix_len / block_size;
    let mut parent_hash = 0;
    for block_idx in 0..num_blocks {
        let extra_keys = extra_keys_per_block.get(block_idx)?;
        let start = block_idx * block_size;
        let end = start + block_size;
        parent_hash = if block_idx == 0 && cache_salt != 0 {
            let mut salted_keys = Vec::with_capacity(extra_keys.len() + 1);
            salted_keys.extend_from_slice(extra_keys);
            salted_keys.push(cache_salt);
            mlx_paged_attn::hash_tokens(&tokens[start..end], parent_hash, &salted_keys)
        } else {
            mlx_paged_attn::hash_tokens(&tokens[start..end], parent_hash, extra_keys)
        };
    }

    Some(parent_hash)
}

fn gemma4_prefix_uses_media_keys(
    prefix_len: u32,
    block_size: u32,
    extra_keys_per_block: &[Vec<u64>],
) -> bool {
    if block_size == 0 {
        return false;
    }
    extra_keys_per_block
        .iter()
        .take((prefix_len / block_size) as usize)
        .any(|keys| !keys.is_empty())
}

fn gemma4_sliding_caches_ready_at(
    config: &Gemma4Config,
    caches: Option<&[Gemma4LayerCache]>,
    offset: u32,
) -> Result<bool> {
    let Some(caches) = caches else {
        return Ok(false);
    };
    if caches.len() != config.num_hidden_layers as usize {
        return Ok(false);
    }
    for (layer_idx, cache) in caches.iter().enumerate() {
        // KV-shared layers are aliases: SharedOnSliding consumes its physical
        // anchor's stash and never advances the alias slot itself. Requiring an
        // offset on that empty slot makes every E2B checkpoint impossible.
        if !config.is_sliding_layer(layer_idx) || config.is_kv_shared_layer(layer_idx) {
            continue;
        }
        if !cache.sliding_offset_matches(offset as i32)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn snapshot_gemma4_sliding_caches(
    config: &Gemma4Config,
    caches: &[Gemma4LayerCache],
    expected_offset: u32,
) -> Result<Option<Vec<Option<RotatingKVCacheSnapshot>>>> {
    if !gemma4_sliding_caches_ready_at(config, Some(caches), expected_offset)? {
        return Ok(None);
    }

    let mut snapshots = Vec::with_capacity(caches.len());
    for (layer_idx, cache) in caches.iter().enumerate() {
        if config.is_sliding_layer(layer_idx) && !config.is_kv_shared_layer(layer_idx) {
            let Some(snapshot) = cache.snapshot_sliding()? else {
                return Ok(None);
            };
            snapshots.push(Some(snapshot));
        } else {
            snapshots.push(None);
        }
    }
    Ok(Some(snapshots))
}

fn gemma4_sliding_snapshots_ready_at(
    config: &Gemma4Config,
    snapshots: &[Option<RotatingKVCacheSnapshot>],
    expected_offset: u32,
) -> bool {
    if snapshots.len() != config.num_hidden_layers as usize {
        return false;
    }
    snapshots.iter().enumerate().all(|(layer_idx, snapshot)| {
        if config.is_sliding_layer(layer_idx) && !config.is_kv_shared_layer(layer_idx) {
            snapshot.as_ref().is_some_and(|snapshot| {
                snapshot.offset == expected_offset as i32
                    && snapshot.max_size == config.sliding_window
            })
        } else {
            snapshot.is_none()
        }
    })
}

fn prepare_gemma4_sliding_checkpoint_captures(
    config: &Gemma4Config,
    caches: &mut [Gemma4LayerCache],
    boundaries: &[u32],
) -> Result<()> {
    if caches.len() != config.num_hidden_layers as usize {
        return Err(Error::from_reason(format!(
            "Gemma4 sliding checkpoint capture cache count mismatch: caches={} layers={}",
            caches.len(),
            config.num_hidden_layers
        )));
    }
    for (layer_idx, cache) in caches.iter_mut().enumerate() {
        if config.is_sliding_layer(layer_idx) && !config.is_kv_shared_layer(layer_idx) {
            cache.prepare_sliding_checkpoint_capture(boundaries)?;
        } else {
            cache.prepare_sliding_checkpoint_capture(&[])?;
        }
    }
    Ok(())
}

fn take_gemma4_sliding_checkpoint_captures(
    config: &Gemma4Config,
    caches: &mut [Gemma4LayerCache],
    boundaries: &[u32],
) -> Result<Vec<Vec<Option<RotatingKVCacheSnapshot>>>> {
    let mut captures = vec![vec![None; caches.len()]; boundaries.len()];
    for (layer_idx, cache) in caches.iter_mut().enumerate() {
        let layer_captures = cache.take_sliding_checkpoint_captures();
        if !config.is_sliding_layer(layer_idx) || config.is_kv_shared_layer(layer_idx) {
            if !layer_captures.is_empty() {
                return Err(Error::from_reason(format!(
                    "Gemma4 non-physical sliding layer {layer_idx} produced checkpoint captures"
                )));
            }
            continue;
        }
        if layer_captures.len() != boundaries.len() {
            return Err(Error::from_reason(format!(
                "Gemma4 sliding layer {layer_idx} captured {} checkpoints for {} boundaries",
                layer_captures.len(),
                boundaries.len()
            )));
        }
        for (boundary_idx, (snapshot, &boundary)) in layer_captures
            .into_iter()
            .zip(boundaries.iter())
            .enumerate()
        {
            if snapshot.offset != boundary as i32 {
                return Err(Error::from_reason(format!(
                    "Gemma4 sliding layer {layer_idx} checkpoint offset {} != boundary {boundary}",
                    snapshot.offset
                )));
            }
            captures[boundary_idx][layer_idx] = Some(snapshot);
        }
    }
    Ok(captures)
}

fn materialize_gemma4_sliding_snapshots(
    snapshots: &mut [Option<RotatingKVCacheSnapshot>],
) -> Result<()> {
    for snapshot in snapshots
        .iter_mut()
        .filter_map(|snapshot| snapshot.as_mut())
    {
        snapshot.keys = snapshot.keys.copy()?;
        snapshot.values = snapshot.values.copy()?;
    }

    let mut arrays: Vec<&MxArray> = Vec::new();
    for snapshot in snapshots.iter().filter_map(|snapshot| snapshot.as_ref()) {
        arrays.push(&snapshot.keys);
        arrays.push(&snapshot.values);
    }
    MxArray::eval_arrays(&arrays)
}

fn restore_gemma4_sliding_caches(
    config: &Gemma4Config,
    snapshots: &[Option<RotatingKVCacheSnapshot>],
    expected_offset: u32,
) -> Result<Option<Vec<Gemma4LayerCache>>> {
    if snapshots.len() != config.num_hidden_layers as usize {
        return Ok(None);
    }

    let mut caches = init_caches_for_config(config);
    for (layer_idx, cache) in caches
        .iter_mut()
        .enumerate()
        .take(config.num_hidden_layers as usize)
    {
        if !config.is_sliding_layer(layer_idx) || config.is_kv_shared_layer(layer_idx) {
            continue;
        }
        let Some(snapshot) = snapshots.get(layer_idx).and_then(|s| s.as_ref()) else {
            return Ok(None);
        };
        if snapshot.offset != expected_offset as i32 {
            return Ok(None);
        }
        cache.restore_sliding_snapshot(snapshot)?;
    }

    if !gemma4_sliding_caches_ready_at(config, Some(&caches), expected_offset)? {
        return Ok(None);
    }

    Ok(Some(caches))
}

/// Test-only helper: decide what to do given the verifier's answer and
/// the incoming prompt length. Exact-match (`cached_prefix_len ==
/// tokens_len`) and zero-length prefix both route to
/// [`PrefixCacheDecision::Miss`].
///
/// Mirrors the engine session core's reset-or-reuse branch over this
/// backend's `verify_cache_prefix` return
/// (`engine::session::chat_turn_core`); lifting it out keeps the
/// invariant pinnable without loading a real Gemma4 model.
#[cfg(test)]
#[inline]
pub(crate) fn classify_prefix_cache_decision(
    cached_prefix_len: usize,
    tokens_len: usize,
) -> PrefixCacheDecision {
    if cached_prefix_len > 0 && cached_prefix_len < tokens_len {
        PrefixCacheDecision::StrictExtendHit
    } else {
        PrefixCacheDecision::Miss
    }
}

impl Gemma4Inner {
    /// Create a new Gemma4Inner with empty (uninitialized) weights.
    pub(crate) fn new(config: Gemma4Config) -> Result<Self> {
        let num_layers = config.num_hidden_layers as usize;
        let hidden_size = config.hidden_size as u32;
        let vocab_size = config.vocab_size as u32;

        let embed_tokens = Embedding::new(vocab_size, hidden_size)?;
        let final_norm = RMSNorm::new(hidden_size, Some(config.rms_norm_eps))?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(LinearProj::Standard(Linear::new(
                hidden_size,
                vocab_size,
                Some(false),
            )?))
        };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(Gemma4DecoderLayer::new(&config, i)?);
        }

        // Initialize PLE model-level components if enabled
        let ple = if config.per_layer_input_embeds {
            let ple_dim = config.ple_dim();
            let vocab_ple = config.vocab_size_per_layer_input.unwrap_or(0);
            if ple_dim > 0 && vocab_ple > 0 {
                let total_ple_dim = (num_layers as i32) * ple_dim;
                Some(PleComponents {
                    embed_tokens_per_layer: Embedding::new(vocab_ple as u32, total_ple_dim as u32)?,
                    per_layer_model_projection: Linear::new(
                        hidden_size,
                        total_ple_dim as u32,
                        Some(false),
                    )?,
                    per_layer_projection_norm: RMSNorm::new(
                        ple_dim as u32,
                        Some(config.rms_norm_eps),
                    )?,
                    per_layer_input_scale: 2.0_f64.powf(-0.5),
                    per_layer_model_projection_scale: (config.hidden_size as f64).powf(-0.5),
                    ple_dim,
                    num_layers: num_layers as i32,
                    vocab_size_per_layer_input: vocab_ple,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Initialize vision components. Two disjoint paths:
        //  - SigLIP vision tower (dense gemma4 family), driven by `vision_config`.
        //  - Encoder-free unified embedder, driven by `unified_vision_config`.
        let (vision_tower, unified_vision_embedder, embed_vision, image_processor) =
            if let Some(ref vc) = config.vision_config {
                let vt = Gemma4VisionModel::new(vc)?;
                let ev = Gemma4MultimodalEmbedder::new(
                    vc.hidden_size,
                    config.hidden_size,
                    vc.rms_norm_eps,
                )?;
                let ip = Gemma4ImageProcessor::new(
                    vc.patch_size,
                    vc.default_output_length,
                    vc.pooling_kernel_size,
                );
                (Some(vt), None, Some(ev), Some(ip))
            } else if let Some(ref uvc) = config.unified_vision_config {
                let embedder = Gemma4UnifiedVisionEmbedder::new(uvc)?;
                let ev = Gemma4MultimodalEmbedder::new(
                    uvc.output_proj_dims,
                    config.hidden_size,
                    uvc.rms_norm_eps,
                )?;
                let ip = Gemma4ImageProcessor::new_unified(
                    uvc.patch_size,
                    uvc.num_soft_tokens,
                    uvc.pooling_kernel_size,
                    uvc.model_patch_size,
                );
                (None, Some(embedder), Some(ev), Some(ip))
            } else {
                (None, None, None, None)
            };

        // Encoder-free unified audio embedder. Built only when the checkpoint
        // declares an `audio_config` (`has_audio`). The raw-window projection is
        // Linear(audio_samples_per_token → hidden_size); the embedder's
        // `set_weight` later validates the [hidden, in] shape against the loaded
        // [3840, 640] tensor.
        let embed_audio = if config.has_audio {
            let in_dim = config.audio_samples_per_token.unwrap_or(640);
            Some(Gemma4MultimodalEmbedder::new(
                in_dim,
                config.hidden_size,
                config.rms_norm_eps,
            )?)
        } else {
            None
        };

        let model_id = MODEL_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Block-paged KV adapter — default-on; opt out via
        // `use_block_paged_cache: false`.
        //
        // The long-term source of truth is the model-independent
        // LayerKVCacheSpec plan: Gemma4 declares full/sliding/shared KV
        // requirements, common transformer code groups those specs, and model
        // dispatch should consume opaque group metadata. Runtime still uses the
        // existing single PagedKVCacheAdapter for full-attention groups while
        // sliding-window groups stay on RotatingKVCache until true paged
        // sliding eviction is wired.
        //
        // Cache dtype: BFloat16 (Gemma4's production dtype). KV-shared layers
        // are aliases and do not consume physical pool slots; they resolve to
        // their anchor's group ordinal through `compute_layer_kinds`.
        // The block-paged KV path uses Metal-only kernels; on a non-Metal
        // backend (the CUDA/Linux build) its write/gather methods are throwing
        // stubs. Force flat eager there by leaving the adapter None, so the
        // `paged_adapter.is_some()` routing falls through to the flat path.
        // macOS is unaffected — the probe is always true, so the default wins.
        let paged_adapter = if config.use_block_paged_cache.unwrap_or(true)
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            let block_size = config.paged_block_size.unwrap_or(16);
            let kv_cache_specs =
                compute_layer_kv_cache_specs(&config, block_size, KVCacheDType::BFloat16).map_err(
                    |e| {
                        Error::from_reason(format!(
                            "Gemma4 block-paged adapter: failed to build KV cache specs: {e}"
                        ))
                    },
                )?;
            let kv_cache_groups = compute_layer_kv_cache_groups(
                &config,
                block_size,
                KVCacheDType::BFloat16,
                gemma4_paged_prefill_group_max_chunk(),
            )
            .map_err(|e| {
                Error::from_reason(format!(
                    "Gemma4 block-paged adapter: failed to group KV cache specs: {e}"
                ))
            })?;
            let full_groups: Vec<&KVCacheGroup> = kv_cache_groups
                .iter()
                .filter(|group| matches!(group.attention_kind, AttentionKind::Full))
                .collect();
            if full_groups.len() > 1 {
                return Err(Error::from_reason(format!(
                    "Gemma4 block-paged adapter currently supports one full-attention KV group, \
                     but spec grouping produced {} groups. This model needs the grouped \
                     HybridKVCacheManager path.",
                    full_groups.len()
                )));
            }
            let Some(full_group) = full_groups.first().copied() else {
                return Err(napi::Error::from_reason(
                    "Gemma4 block-paged adapter: config has no full_attention KV group; \
                     paged KV cache requires at least one global attention layer",
                ));
            };
            let num_global_layers = physical_full_attention_layer_count(&kv_cache_specs) as u32;
            if num_global_layers == 0 {
                return Err(napi::Error::from_reason(
                    "Gemma4 block-paged adapter: config has no full_attention layers; \
                     paged KV cache requires at least one global attention layer",
                ));
            }

            let head_size = full_group.physical_layout.head_size;
            let num_kv_heads = full_group.physical_layout.num_kv_heads;
            let max_seq_len = u32::try_from(config.max_position_embeddings).map_err(|_| {
                napi::Error::from_reason(format!(
                    "Gemma4 block-paged adapter: invalid max_position_embeddings={}",
                    config.max_position_embeddings
                ))
            })?;
            if max_seq_len == 0 {
                return Err(napi::Error::from_reason(
                    "Gemma4 block-paged adapter: max_position_embeddings must be > 0",
                ));
            }
            let default_gpu_memory_mb = gemma4_default_paged_cache_memory_mb(
                max_seq_len,
                block_size,
                head_size,
                num_kv_heads,
                num_global_layers,
            );
            let (gpu_memory_mb, paged_cache_memory_source) =
                if let Some(configured_memory_mb) = config.paged_cache_memory_mb {
                    (configured_memory_mb, "config")
                } else {
                    (default_gpu_memory_mb, "auto_full_context")
                };

            let pa_config = mlx_paged_attn::PagedAttentionConfig {
                block_size,
                gpu_memory_mb,
                head_size,
                num_kv_heads,
                // Pool covers only physical full-attention layers. KV-shared
                // aliases reuse their anchor's slot and do not allocate.
                num_layers: num_global_layers,
                use_fp8_cache: Some(false),
                max_seq_len: Some(max_seq_len),
                max_batch_size: Some(32),
            };

            let num_blocks = pa_config.calculate_num_blocks();
            if num_blocks == 0 {
                return Err(napi::Error::from_reason(format!(
                    "Gemma4 block-paged adapter: gpu_memory_mb={gpu_memory_mb} too small \
                     (head_size={head_size}, num_kv_heads={num_kv_heads}, \
                     block_size={block_size}, num_global_layers={num_global_layers})",
                )));
            }

            let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
                num_blocks, block_size,
            )));

            let cache_dtype = mlx_paged_attn::metal::MetalDtype::BFloat16;
            let pool = mlx_paged_attn::LayerKVPool::new(pa_config, num_blocks, cache_dtype)
                .map_err(|e| {
                    napi::Error::from_reason(format!(
                        "Failed to construct LayerKVPool for Gemma4 block-paged adapter: {e}"
                    ))
                })?;

            let adapter =
                PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size).map_err(|e| {
                    napi::Error::from_reason(format!(
                        "Failed to construct Gemma4 PagedKVCacheAdapter: {e}"
                    ))
                })?;

            tracing::info!(
                "Gemma4 block-paged adapter enabled: num_blocks={num_blocks}, \
                 block_size={block_size}, gpu_memory_mb={gpu_memory_mb}, \
                 paged_cache_memory_source={paged_cache_memory_source}, \
                 max_seq_len={max_seq_len}, max_cached_tokens={}, \
                 physical_full_layers={num_global_layers}, kv_groups={}, \
                 full_group_max_admission_blocks={}, cache_dtype=BFloat16",
                num_blocks.saturating_mul(block_size),
                kv_cache_groups.len(),
                full_group.max_admission_blocks
            );
            Some(adapter)
        } else {
            None
        };

        // Derive the per-layer paged-routing classification once here
        // instead of on every paged prefill-chunk / decode-step call (see
        // `compute_layer_kinds_from_kv_cache_specs`'s BTreeMap/BTreeSet
        // grouping + sort). It's a pure function of `config`, which is
        // immutable for the lifetime of this `Gemma4Inner`. Only meaningful
        // when `paged_adapter` is `Some` — every caller first errors out on
        // a `None` adapter before reading the result, so `Vec::new()` below
        // is never read in that case. Guaranteed to succeed whenever
        // `paged_adapter` built above, since that already validated a
        // strictly stronger constraint (a single full-attention group) over
        // the same specs.
        let layer_kinds = if paged_adapter.is_some() {
            compute_layer_kinds_from_kv_cache_specs(&config).map_err(|e| {
                Error::from_reason(format!(
                    "Gemma4Inner::new: failed to derive cached layer-kind routes: {e}"
                ))
            })?
        } else {
            Vec::new()
        };

        Ok(Self {
            config,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            embed_weight_t: None,
            ple,
            vision_tower,
            unified_vision_embedder,
            embed_vision,
            embed_audio,
            image_processor,
            tokenizer: None,
            caches: None,
            cached_token_history: Vec::new(),
            cached_image_key: None,
            cached_audio_key: None,
            cached_paged_image_token_positions: Vec::new(),
            paged_adapter,
            draft: None,
            draft_turn_state: None,
            layer_kinds,
            sliding_prefix_checkpoints: VecDeque::new(),
            sliding_prompt_boundary_checkpoint: None,
            sliding_cold_restore_tail_checkpoint: None,
            paged_turn_prompt_len: 0,
            sliding_last_history_checkpoint: None,
            media_session_context: MediaCapabilities::NONE,
            paged_text_turn_context: MediaCapabilities::NONE,
            media_session_continuable: false,
            paged_finalize_failed: false,
            output_starts_in_reasoning_channel: AtomicBool::new(false),
            model_id,
        })
    }

    /// Whether the complete physical image execution path is loaded.
    ///
    /// This is the single authority for both `ExecutionPlan.media.images` and
    /// the NAPI `supportsImages()` snapshot. A config declaration or lone image
    /// processor is insufficient: inference also needs one vision stack, its
    /// projection, and the paged adapter used by Gemma's multimodal executor.
    pub(crate) fn image_path_loaded(&self) -> bool {
        gemma4_image_path_loaded(
            self.image_processor.is_some(),
            self.embed_vision.is_some(),
            self.vision_tower.is_some(),
            self.unified_vision_embedder.is_some(),
            self.paged_adapter.is_some(),
        )
    }

    /// The loaded DSpark draft, when the draft variant is DSpark.
    pub(crate) fn dspark_draft(&self) -> Option<&super::dspark::DsparkDraftModel> {
        match self.draft.as_ref() {
            Some(Gemma4Draft::Dspark(draft)) => Some(draft),
            _ => None,
        }
    }

    /// The loaded assistant draft, when the draft variant is assistant.
    pub(crate) fn assistant_draft(&self) -> Option<&super::assistant::AssistantDraftModel> {
        match self.draft.as_ref() {
            Some(Gemma4Draft::Assistant(draft)) => Some(draft),
            _ => None,
        }
    }

    /// Whether ANY draft variant is loaded (the speculative whole-turn
    /// gate; see `mtp_turn`).
    pub(crate) fn has_draft(&self) -> bool {
        self.draft.is_some()
    }

    /// Initialize the per-turn KV caches in-place.
    ///
    /// Called on the first turn of a session by the engine's miss-path
    /// `reset_caches(ResetScope::PrefixMiss)` and the vision cores (or
    /// defensively whenever `self.caches` is `None` because a previous
    /// `reset_caches_sync` wiped them). Subsequent turns reuse the
    /// already-populated cache in-place.
    ///
    /// Layer-type routing mirrors the free `init_caches_for_config` used
    /// by `warmup_forward`: global layers get `KVCache`, sliding layers get
    /// `RotatingKVCache` with `config.sliding_window`.
    pub(crate) fn init_caches_sync(&mut self) -> Result<()> {
        let caches = (0..self.config.num_hidden_layers as usize)
            .map(|i| {
                if self.config.is_global_layer(i) {
                    Gemma4LayerCache::new_global()
                } else {
                    Gemma4LayerCache::new_sliding(self.config.sliding_window)
                }
            })
            .collect();
        self.caches = Some(caches);
        self.clear_reuse_state();
        Ok(())
    }

    /// Return the per-layer routing list for the paged dispatch.
    ///
    /// Cheap clone of `self.layer_kinds`, cached once in `Gemma4Inner::new`
    /// instead of being re-derived (BTreeMap/BTreeSet grouping + a sort —
    /// see [`compute_layer_kinds`] (free helper) and
    /// `compute_layer_kinds_from_kv_cache_specs`) on every call. It's a pure
    /// function of the immutable `Gemma4Config`, so recomputing it from
    /// scratch on every paged prefill-chunk / decode-step call was pure
    /// waste.
    pub(crate) fn compute_layer_kinds(&self) -> Result<Vec<Gemma4LayerKind>> {
        Ok(self.layer_kinds.clone())
    }

    /// Drop the live KV caches and clear reuse-tracking state.
    ///
    /// `Gemma4LayerCache` has no `reset()` (the inner `KVCache` /
    /// `RotatingKVCache` don't expose one here), so this simply takes the
    /// Vec and lets the next `init_caches_sync` rebuild. Cleared reuse
    /// state ensures a subsequent chat turn can't mistakenly claim a cache
    /// prefix hit against stale history.
    ///
    /// Called by the session API's reset path
    /// (`ChatBackend::reset_caches`) so that a fresh turn starts from an
    /// empty cache. The prefill/decode primitives never call it directly
    /// — they trust their caller's cache-management.
    pub(crate) fn reset_caches_sync(&mut self) -> Result<()> {
        self.caches = None;
        self.clear_reuse_state();
        Ok(())
    }

    /// Clear cached token history and media identity/context. Called from both
    /// `init_caches_sync` and `reset_caches_sync`.
    fn clear_reuse_state(&mut self) {
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_audio_key = None;
        self.cached_paged_image_token_positions.clear();
        self.media_session_context = MediaCapabilities::NONE;
        self.paged_text_turn_context = MediaCapabilities::NONE;
        self.sliding_prefix_checkpoints.clear();
        self.sliding_prompt_boundary_checkpoint = None;
        self.sliding_cold_restore_tail_checkpoint = None;
        self.paged_turn_prompt_len = 0;
        self.sliding_last_history_checkpoint = None;
        // Covers both reset paths (init_caches_sync + reset_caches_sync): a
        // session that just dropped its media KV can no longer warm-continue.
        self.media_session_continuable = false;
        self.paged_finalize_failed = false;
    }

    /// Publish the raw media identity and the persistent causal context for a
    /// successfully finalized multimodal turn.
    fn publish_media_session_context(
        &mut self,
        new_image_key: Option<u64>,
        new_audio_key: Option<u64>,
    ) {
        self.cached_image_key = new_image_key;
        self.cached_audio_key = new_audio_key;
        self.media_session_context = MediaCapabilities {
            images: new_image_key.is_some(),
            audio: new_audio_key.is_some(),
        };
    }

    fn find_gemma4_sliding_history_checkpoint(
        &self,
        tokens: &[u32],
        prefix_len: u32,
        image_token_positions: &[(u32, u64)],
    ) -> Result<Option<Vec<Gemma4LayerCache>>> {
        let Some(prefix_tokens) = tokens.get(..prefix_len as usize) else {
            return Ok(None);
        };
        let Some(checkpoint) = self.sliding_last_history_checkpoint.as_ref() else {
            return Ok(None);
        };
        if checkpoint.tokens.as_slice() != prefix_tokens
            || checkpoint.image_token_positions.as_slice() != image_token_positions
        {
            return Ok(None);
        }
        restore_gemma4_sliding_caches(&self.config, &checkpoint.snapshots, prefix_len)
    }

    fn remember_gemma4_sliding_history_checkpoint(
        &mut self,
        history_tokens: &[u32],
    ) -> Result<Gemma4SlidingCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = Gemma4SlidingCheckpointStoreTrace::default();
        if history_tokens.is_empty() {
            self.sliding_last_history_checkpoint = None;
            return Ok(trace.finish(total_start));
        }

        let expected_offset = history_tokens.len() as u32;
        if !gemma4_sliding_caches_ready_at(&self.config, self.caches.as_deref(), expected_offset)? {
            self.sliding_last_history_checkpoint = None;
            return Ok(trace.finish(total_start));
        }

        let eval_start = trace_enabled.then(std::time::Instant::now);
        eval_gemma4_caches(
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 sliding checkpoint caches missing"))?,
        )?;
        trace.eval_ms = eval_start.map(elapsed_ms).unwrap_or(0.0);

        let snapshot_start = trace_enabled.then(std::time::Instant::now);
        let Some(snapshots) = snapshot_gemma4_sliding_caches(
            &self.config,
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 sliding checkpoint caches missing"))?,
            expected_offset,
        )?
        else {
            self.sliding_last_history_checkpoint = None;
            trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);
            return Ok(trace.finish(total_start));
        };
        trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);

        let token_clone_start = trace_enabled.then(std::time::Instant::now);
        let tokens = history_tokens.to_vec();
        trace.token_clone_ms = token_clone_start.map(elapsed_ms).unwrap_or(0.0);

        let update_start = trace_enabled.then(std::time::Instant::now);
        self.sliding_last_history_checkpoint = Some(Gemma4SlidingHistoryCheckpoint {
            tokens,
            image_token_positions: self.cached_paged_image_token_positions.clone(),
            snapshots,
        });
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;
        Ok(trace.finish(total_start))
    }

    /// Retention caps for this turn.
    ///
    /// The whole decision — including whether this turn wants a ladder at all —
    /// lives in [`gemma4_sliding_retention_caps_for_cold_tier`], a free function
    /// of `(config, cold tier, block size)`. All this method contributes is the
    /// borrow of the adapter's cold-tier context, so the interesting half is
    /// reachable from a unit test without a GPU or a loaded checkpoint. Before
    /// that split, `gemma4_sliding_cold_ladder_wanted` was an untested master
    /// switch: flipping it to `false` made the cold tier inert with every test
    /// still green.
    fn gemma4_sliding_retention_caps_for_turn(
        &self,
        block_size: u32,
    ) -> Gemma4SlidingRetentionCaps {
        gemma4_sliding_retention_caps_for_cold_tier(
            &self.config,
            self.paged_adapter
                .as_ref()
                .and_then(|adapter| adapter.cold_tier()),
            block_size,
        )
    }

    #[cfg(test)]
    fn find_gemma4_sliding_prefix_checkpoint(
        &self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        cache_salt: u64,
    ) -> Result<Option<Gemma4SlidingPrefixCheckpointHit>> {
        let extra_keys_per_block = engine::build_paged_extra_keys(tokens.len(), block_size, &[]);
        self.find_gemma4_sliding_prefix_checkpoint_with_keys(
            tokens,
            prefix_len,
            block_size,
            &extra_keys_per_block,
            cache_salt,
        )
    }

    fn find_gemma4_sliding_prefix_checkpoint_with_keys(
        &self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Result<Option<Gemma4SlidingPrefixCheckpointHit>> {
        fn try_restore_checkpoint(
            config: &Gemma4Config,
            checkpoint: &Gemma4SlidingPrefixCheckpoint,
            tokens: &[u32],
            target_prefix_len: u32,
            block_size: u32,
            extra_keys_per_block: &[Vec<u64>],
            cache_salt: u64,
        ) -> Result<Option<Gemma4SlidingPrefixCheckpointHit>> {
            if checkpoint.prefix_len > target_prefix_len || checkpoint.block_size != block_size {
                return Ok(None);
            }
            let Some(prefix_tokens) = tokens.get(..checkpoint.prefix_len as usize) else {
                return Ok(None);
            };
            if checkpoint.tokens.as_slice() != prefix_tokens {
                return Ok(None);
            }
            let Some(final_block_hash) = compute_gemma4_paged_prefix_block_hash_with_keys(
                tokens,
                checkpoint.prefix_len,
                block_size,
                extra_keys_per_block,
                cache_salt,
            ) else {
                return Ok(None);
            };
            if checkpoint.final_block_hash != final_block_hash {
                return Ok(None);
            }
            let Some(caches) = restore_gemma4_sliding_caches(
                config,
                &checkpoint.snapshots,
                checkpoint.prefix_len,
            )?
            else {
                return Ok(None);
            };
            Ok(Some(Gemma4SlidingPrefixCheckpointHit {
                prefix_len: checkpoint.prefix_len,
                caches,
            }))
        }

        let mut best_hit: Option<Gemma4SlidingPrefixCheckpointHit> = None;
        if let Some(checkpoint) = self.sliding_prompt_boundary_checkpoint.as_ref()
            && let Some(hit) = try_restore_checkpoint(
                &self.config,
                checkpoint,
                tokens,
                prefix_len,
                block_size,
                extra_keys_per_block,
                cache_salt,
            )?
        {
            best_hit = Some(hit);
        }

        for checkpoint in self.sliding_prefix_checkpoints.iter().rev() {
            if best_hit
                .as_ref()
                .is_some_and(|hit| hit.prefix_len >= checkpoint.prefix_len)
            {
                continue;
            }
            if let Some(hit) = try_restore_checkpoint(
                &self.config,
                checkpoint,
                tokens,
                prefix_len,
                block_size,
                extra_keys_per_block,
                cache_salt,
            )? {
                if hit.prefix_len == prefix_len {
                    return Ok(Some(hit));
                }
                best_hit = Some(hit);
            }
        }

        Ok(best_hit)
    }

    fn remember_gemma4_sliding_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        cache_salt: u64,
    ) -> Result<Gemma4SlidingCheckpointStoreTrace> {
        let extra_keys_per_block = engine::build_paged_extra_keys(
            tokens.len(),
            block_size,
            &self.cached_paged_image_token_positions,
        );
        self.remember_gemma4_sliding_prefix_checkpoint_with_keys(
            tokens,
            prefix_len,
            block_size,
            &extra_keys_per_block,
            cache_salt,
        )
    }

    fn remember_gemma4_sliding_prefix_checkpoint_with_keys(
        &mut self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Result<Gemma4SlidingCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = Gemma4SlidingCheckpointStoreTrace::default();
        let Some(final_block_hash) = compute_gemma4_paged_prefix_block_hash_with_keys(
            tokens,
            prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        ) else {
            return Ok(trace.finish(total_start));
        };
        let Some(prefix_tokens) = tokens.get(..prefix_len as usize) else {
            return Ok(trace.finish(total_start));
        };
        if !gemma4_sliding_caches_ready_at(&self.config, self.caches.as_deref(), prefix_len)? {
            return Ok(trace.finish(total_start));
        }

        let eval_start = trace_enabled.then(std::time::Instant::now);
        eval_gemma4_caches(
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 sliding prefix caches missing"))?,
        )?;
        trace.eval_ms = eval_start.map(elapsed_ms).unwrap_or(0.0);

        let snapshot_start = trace_enabled.then(std::time::Instant::now);
        let Some(snapshots) = snapshot_gemma4_sliding_caches(
            &self.config,
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 sliding prefix caches missing"))?,
            prefix_len,
        )?
        else {
            trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);
            return Ok(trace.finish(total_start));
        };
        trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);

        let token_clone_start = trace_enabled.then(std::time::Instant::now);
        let prefix_tokens = prefix_tokens.to_vec();
        trace.token_clone_ms = token_clone_start.map(elapsed_ms).unwrap_or(0.0);

        let update_start = trace_enabled.then(std::time::Instant::now);
        // Hoisted before the `&mut` borrow of the store, as the captured path
        // already does.
        let caps = self.gemma4_sliding_retention_caps_for_turn(block_size);
        upsert_gemma4_sliding_prefix_checkpoint(
            &mut self.sliding_prefix_checkpoints,
            Gemma4SlidingPrefixCheckpointDraft {
                prefix_len,
                block_size,
                final_block_hash,
                protected_image_prompt_boundary: false,
                tokens: prefix_tokens,
                snapshots,
            },
            caps,
            trace_enabled,
        );
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;
        Ok(trace.finish(total_start))
    }

    fn remember_gemma4_sliding_materialized_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        cache_salt: u64,
    ) -> Result<Gemma4SlidingCheckpointStoreTrace> {
        let extra_keys_per_block = engine::build_paged_extra_keys(
            tokens.len(),
            block_size,
            &self.cached_paged_image_token_positions,
        );
        self.remember_gemma4_sliding_materialized_prefix_checkpoint_with_keys(
            tokens,
            prefix_len,
            block_size,
            &extra_keys_per_block,
            cache_salt,
        )
    }

    fn remember_gemma4_sliding_materialized_prefix_checkpoint_with_keys(
        &mut self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Result<Gemma4SlidingCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = Gemma4SlidingCheckpointStoreTrace::default();
        let Some(final_block_hash) = compute_gemma4_paged_prefix_block_hash_with_keys(
            tokens,
            prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        ) else {
            return Ok(trace.finish(total_start));
        };
        let Some(prefix_tokens) = tokens.get(..prefix_len as usize) else {
            return Ok(trace.finish(total_start));
        };
        if !gemma4_sliding_caches_ready_at(&self.config, self.caches.as_deref(), prefix_len)? {
            return Ok(trace.finish(total_start));
        }

        let snapshot_start = trace_enabled.then(std::time::Instant::now);
        let Some(mut snapshots) = snapshot_gemma4_sliding_caches(
            &self.config,
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 sliding prefix caches missing"))?,
            prefix_len,
        )?
        else {
            trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);
            return Ok(trace.finish(total_start));
        };
        trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);

        let eval_start = trace_enabled.then(std::time::Instant::now);
        materialize_gemma4_sliding_snapshots(&mut snapshots)?;
        trace.eval_ms = eval_start.map(elapsed_ms).unwrap_or(0.0);

        let token_clone_start = trace_enabled.then(std::time::Instant::now);
        let prefix_tokens = prefix_tokens.to_vec();
        trace.token_clone_ms = token_clone_start.map(elapsed_ms).unwrap_or(0.0);

        let update_start = trace_enabled.then(std::time::Instant::now);
        let caps = self.gemma4_sliding_retention_caps_for_turn(block_size);
        upsert_gemma4_sliding_prefix_checkpoint(
            &mut self.sliding_prefix_checkpoints,
            Gemma4SlidingPrefixCheckpointDraft {
                prefix_len,
                block_size,
                final_block_hash,
                protected_image_prompt_boundary: false,
                tokens: prefix_tokens,
                snapshots,
            },
            caps,
            trace_enabled,
        );
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;
        Ok(trace.finish(total_start))
    }

    /// Store a prefix checkpoint captured from inside a larger prefill compute
    /// chunk. Unlike the live-cache path above, these snapshots describe an
    /// earlier logical offset and therefore must not be re-read from
    /// `self.caches`, which has already advanced to the chunk end.
    fn remember_gemma4_sliding_captured_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        cache_salt: u64,
        mut snapshots: Vec<Option<RotatingKVCacheSnapshot>>,
        sink: Gemma4SlidingCapturedCheckpointSink,
    ) -> Result<Gemma4SlidingCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = Gemma4SlidingCheckpointStoreTrace::default();
        let extra_keys_per_block = engine::build_paged_extra_keys(
            tokens.len(),
            block_size,
            &self.cached_paged_image_token_positions,
        );
        let Some(final_block_hash) = compute_gemma4_paged_prefix_block_hash_with_keys(
            tokens,
            prefix_len,
            block_size,
            &extra_keys_per_block,
            cache_salt,
        ) else {
            return Ok(trace.finish(total_start));
        };
        let Some(prefix_tokens) = tokens.get(..prefix_len as usize) else {
            return Ok(trace.finish(total_start));
        };
        if !gemma4_sliding_snapshots_ready_at(&self.config, &snapshots, prefix_len) {
            return Err(Error::from_reason(format!(
                "Gemma4 captured sliding snapshots are incomplete at offset {prefix_len}"
            )));
        }

        let eval_start = trace_enabled.then(std::time::Instant::now);
        materialize_gemma4_sliding_snapshots(&mut snapshots)?;
        trace.eval_ms = eval_start.map(elapsed_ms).unwrap_or(0.0);

        let token_clone_start = trace_enabled.then(std::time::Instant::now);
        let prefix_tokens = prefix_tokens.to_vec();
        trace.token_clone_ms = token_clone_start.map(elapsed_ms).unwrap_or(0.0);

        let update_start = trace_enabled.then(std::time::Instant::now);
        let caps = self.gemma4_sliding_retention_caps_for_turn(block_size);
        let draft = Gemma4SlidingPrefixCheckpointDraft {
            prefix_len,
            block_size,
            final_block_hash,
            protected_image_prompt_boundary: false,
            tokens: prefix_tokens,
            snapshots,
        };
        match sink {
            Gemma4SlidingCapturedCheckpointSink::PrefixStore => {
                upsert_gemma4_sliding_prefix_checkpoint(
                    &mut self.sliding_prefix_checkpoints,
                    draft,
                    caps,
                    trace_enabled,
                );
            }
            Gemma4SlidingCapturedCheckpointSink::ColdRestoreTail => {
                self.sliding_cold_restore_tail_checkpoint = Some(draft.into_checkpoint(caps));
            }
        }
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;
        Ok(trace.finish(total_start))
    }

    fn remember_gemma4_sliding_materialized_prompt_boundary_checkpoint(
        &mut self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        cache_salt: u64,
    ) -> Result<Gemma4SlidingCheckpointStoreTrace> {
        let extra_keys_per_block = engine::build_paged_extra_keys(
            tokens.len(),
            block_size,
            &self.cached_paged_image_token_positions,
        );
        self.remember_gemma4_sliding_materialized_prompt_boundary_checkpoint_with_keys(
            tokens,
            prefix_len,
            block_size,
            &extra_keys_per_block,
            cache_salt,
            false,
        )
    }

    fn remember_gemma4_sliding_materialized_prompt_boundary_checkpoint_with_keys(
        &mut self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        protect_image_prompt_boundary: bool,
    ) -> Result<Gemma4SlidingCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = Gemma4SlidingCheckpointStoreTrace::default();
        let Some(final_block_hash) = compute_gemma4_paged_prefix_block_hash_with_keys(
            tokens,
            prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        ) else {
            self.sliding_prompt_boundary_checkpoint = None;
            return Ok(trace.finish(total_start));
        };
        let Some(prefix_tokens) = tokens.get(..prefix_len as usize) else {
            self.sliding_prompt_boundary_checkpoint = None;
            return Ok(trace.finish(total_start));
        };
        if !gemma4_sliding_caches_ready_at(&self.config, self.caches.as_deref(), prefix_len)? {
            self.sliding_prompt_boundary_checkpoint = None;
            return Ok(trace.finish(total_start));
        }

        let snapshot_start = trace_enabled.then(std::time::Instant::now);
        let Some(mut snapshots) = snapshot_gemma4_sliding_caches(
            &self.config,
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 sliding prefix caches missing"))?,
            prefix_len,
        )?
        else {
            self.sliding_prompt_boundary_checkpoint = None;
            trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);
            return Ok(trace.finish(total_start));
        };
        trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);

        let eval_start = trace_enabled.then(std::time::Instant::now);
        materialize_gemma4_sliding_snapshots(&mut snapshots)?;
        trace.eval_ms = eval_start.map(elapsed_ms).unwrap_or(0.0);

        let token_clone_start = trace_enabled.then(std::time::Instant::now);
        let prefix_tokens = prefix_tokens.to_vec();
        trace.token_clone_ms = token_clone_start.map(elapsed_ms).unwrap_or(0.0);

        let update_start = trace_enabled.then(std::time::Instant::now);
        let draft = Gemma4SlidingPrefixCheckpointDraft {
            prefix_len,
            block_size,
            final_block_hash,
            protected_image_prompt_boundary: protect_image_prompt_boundary
                && gemma4_prefix_uses_media_keys(prefix_len, block_size, extra_keys_per_block),
            tokens: prefix_tokens,
            snapshots,
        };
        let caps = self.gemma4_sliding_retention_caps_for_turn(block_size);
        // The singleton is set first so the store's `&mut` borrow does not span
        // the assignment; the clone is the same one this path always paid.
        self.sliding_prompt_boundary_checkpoint = Some(draft.clone().into_checkpoint(caps));
        upsert_gemma4_sliding_prefix_checkpoint(
            &mut self.sliding_prefix_checkpoints,
            draft,
            caps,
            trace_enabled,
        );
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;
        Ok(trace.finish(total_start))
    }

    /// Publish this decode step's sliding checkpoint, if the cursor sits on a
    /// boundary this turn publishes.
    ///
    /// Everything that DECIDES is in [`gemma4_sliding_decode_boundary_plan`];
    /// this body only reads the adapter's three facts and does the I/O. That
    /// split exists because the decision is the part that can silently
    /// disconnect the cold tier (`want_ladder` hard-coded `false` here reverted
    /// decode to cadence-only while every unit test stayed green), and it is
    /// the part a test can execute without a GPU or a loaded checkpoint.
    fn maybe_remember_gemma4_sliding_decode_boundary_checkpoint(
        &mut self,
        trace_label: &str,
        trace_enabled: bool,
    ) -> Result<()> {
        let Some(adapter) = self.paged_adapter.as_ref() else {
            return Ok(());
        };
        let Some(boundary) = gemma4_sliding_decode_boundary_plan(
            &self.config,
            adapter.cold_tier(),
            adapter.block_size(),
            adapter.current_token_count(),
        ) else {
            return Ok(());
        };
        let request_tokens = adapter.request_tokens().to_vec();

        let store_trace = self.remember_gemma4_sliding_materialized_prefix_checkpoint(
            &request_tokens,
            boundary.prefix_len,
            boundary.block_size,
            0,
        )?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 {trace_label}_sliding_block_checkpoint boundary_tokens={} block_size={} checkpoint_interval={} cold_anchor_rung={} stored={} materialize_ms={:.1} snapshot_ms={:.1} token_clone_ms={:.1} update_ms={:.1} total_ms={:.1}",
                boundary.prefix_len,
                boundary.block_size,
                boundary.checkpoint_interval,
                boundary.on_anchor_rung,
                store_trace.stored,
                store_trace.eval_ms,
                store_trace.snapshot_ms,
                store_trace.token_clone_ms,
                store_trace.update_ms,
                store_trace.total_ms
            ));
        }
        Ok(())
    }

    fn maybe_remember_gemma4_sliding_prompt_boundary_checkpoint(
        &mut self,
        trace_label: &str,
        tokens: &[u32],
        boundary_len: u32,
        trace_enabled: bool,
    ) -> Result<()> {
        let Some(adapter) = self.paged_adapter.as_ref() else {
            return Ok(());
        };
        let block_size = adapter.block_size();
        if boundary_len == 0 || block_size == 0 || !boundary_len.is_multiple_of(block_size) {
            return Ok(());
        }

        let store_trace = self.remember_gemma4_sliding_materialized_prompt_boundary_checkpoint(
            tokens,
            boundary_len,
            block_size,
            0,
        )?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 {trace_label}_sliding_prompt_checkpoint boundary_tokens={} block_size={} stored={} materialize_ms={:.1} snapshot_ms={:.1} token_clone_ms={:.1} update_ms={:.1} total_ms={:.1}",
                boundary_len,
                block_size,
                store_trace.stored,
                store_trace.eval_ms,
                store_trace.snapshot_ms,
                store_trace.token_clone_ms,
                store_trace.update_ms,
                store_trace.total_ms
            ));
        }
        Ok(())
    }

    fn prepare_gemma4_sliding_prefix_state_with_keys(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        continued_live_prefix: bool,
        extra_keys_per_block: &[Vec<u64>],
        image_token_positions: &[(u32, u64)],
        require_exact_checkpoint: bool,
    ) -> Result<Gemma4SlidingPrefixPreparation> {
        let trace_enabled = inference_trace_enabled();
        let prepare_start = trace_enabled.then(std::time::Instant::now);

        if cached_prefix_len == 0 {
            self.caches = Some(init_caches_for_config(&self.config));
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=fresh cached_prefix_tokens=0 elapsed_ms={:.1}",
                    prepare_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            return Ok(Gemma4SlidingPrefixPreparation {
                state: "fresh",
                primed_prefix_len: 0,
            });
        }

        let image_identity_matches =
            self.cached_paged_image_token_positions.as_slice() == image_token_positions;
        if continued_live_prefix
            && image_identity_matches
            && gemma4_sliding_caches_ready_at(
                &self.config,
                self.caches.as_deref(),
                cached_prefix_len,
            )?
        {
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=live cached_prefix_tokens={} elapsed_ms={:.1}",
                    cached_prefix_len,
                    prepare_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            return Ok(Gemma4SlidingPrefixPreparation {
                state: "live",
                primed_prefix_len: cached_prefix_len,
            });
        }

        let matches_live_history = image_identity_matches
            && self.cached_token_history.len() == cached_prefix_len as usize
            && tokens.starts_with(&self.cached_token_history);
        if matches_live_history
            && gemma4_sliding_caches_ready_at(
                &self.config,
                self.caches.as_deref(),
                cached_prefix_len,
            )?
        {
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=last_history cached_prefix_tokens={} elapsed_ms={:.1}",
                    cached_prefix_len,
                    prepare_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            return Ok(Gemma4SlidingPrefixPreparation {
                state: "last_history",
                primed_prefix_len: cached_prefix_len,
            });
        }

        let history_lookup_start = trace_enabled.then(std::time::Instant::now);
        if let Some(caches) = self.find_gemma4_sliding_history_checkpoint(
            tokens,
            cached_prefix_len,
            image_token_positions,
        )? {
            self.caches = Some(caches);
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=last_history_checkpoint cached_prefix_tokens={} history_lookup_ms={:.1} elapsed_ms={:.1}",
                    cached_prefix_len,
                    history_lookup_start.map(elapsed_ms).unwrap_or(0.0),
                    prepare_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            return Ok(Gemma4SlidingPrefixPreparation {
                state: "last_history_checkpoint",
                primed_prefix_len: cached_prefix_len,
            });
        }

        let block_size = self
            .paged_adapter
            .as_ref()
            .map(|adapter| adapter.block_size())
            .unwrap_or(0);
        let prefix_lookup_start = trace_enabled.then(std::time::Instant::now);
        if let Some(hit) = self.find_gemma4_sliding_prefix_checkpoint_with_keys(
            tokens,
            cached_prefix_len,
            block_size,
            extra_keys_per_block,
            0,
        )? {
            let hit_prefix_len = hit.prefix_len;
            if require_exact_checkpoint && hit_prefix_len != cached_prefix_len {
                // A partial in-memory checkpoint cannot back image K/V, but it
                // must not hide an exact sidecar the adapter just restored from
                // SSD. Reset the partial state and continue to the cold-sidecar
                // probe below; if that also misses, the VLM resolver restarts
                // the whole prepared request cold.
                self.caches = Some(init_caches_for_config(&self.config));
            } else {
                self.caches = Some(hit.caches);
                let state = if hit_prefix_len == cached_prefix_len {
                    "prefix_checkpoint"
                } else {
                    "partial_prefix_checkpoint"
                };
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state={} cached_prefix_tokens={} primed_prefix_tokens={} replay_delta_tokens={} prefix_lookup_ms={:.1} elapsed_ms={:.1}",
                        state,
                        cached_prefix_len,
                        hit_prefix_len,
                        cached_prefix_len.saturating_sub(hit_prefix_len),
                        prefix_lookup_start.map(elapsed_ms).unwrap_or(0.0),
                        prepare_start.map(elapsed_ms).unwrap_or(0.0)
                    ));
                }
                return Ok(Gemma4SlidingPrefixPreparation {
                    state,
                    primed_prefix_len: hit_prefix_len,
                });
            }
        }

        // Every in-memory source has missed. Before paying a full decoder
        // replay over the reused prefix, install the sliding state the SSD
        // cold tier restored alongside this turn's paged K/V — if it restored
        // any. `install_gemma4_sliding_cold_sidecar` accepts only a sidecar at
        // EXACTLY `cached_prefix_len`, so it is also a valid exact checkpoint
        // for an image-lineage turn. A missing/misaligned image sidecar falls
        // through with `primed_prefix_len == 0`; the VLM resolver then discards
        // the global-only hit and restarts cold rather than replaying image
        // placeholder ids.
        if let Some(preparation) = self.install_gemma4_sliding_cold_sidecar(cached_prefix_len)? {
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state={} cached_prefix_tokens={} primed_prefix_tokens={} replay_delta_tokens={} elapsed_ms={:.1}",
                    preparation.state,
                    cached_prefix_len,
                    preparation.primed_prefix_len,
                    cached_prefix_len.saturating_sub(preparation.primed_prefix_len),
                    prepare_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            return Ok(preparation);
        }

        self.caches = Some(init_caches_for_config(&self.config));
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=replay cached_prefix_tokens={} history_lookup_ms={:.1} prefix_lookup_ms={:.1} elapsed_ms={:.1}",
                cached_prefix_len,
                history_lookup_start.map(elapsed_ms).unwrap_or(0.0),
                prefix_lookup_start.map(elapsed_ms).unwrap_or(0.0),
                prepare_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        Ok(Gemma4SlidingPrefixPreparation {
            state: "replay",
            primed_prefix_len: 0,
        })
    }

    /// Install auxiliary sliding-window state the SSD cold tier restored
    /// alongside this turn's paged K/V prefix.
    ///
    /// This is the cold-tier twin of the in-memory checkpoint lookups above:
    /// same destination (`self.caches` at a known offset), different source
    /// (an on-disk [`mlx_paged_attn::ColdSidecar`] instead of a live
    /// `RotatingKVCacheSnapshot`). It is consulted only after every in-memory
    /// source has missed, because those are already materialized and cost no
    /// decode.
    ///
    /// `ColdTierWalk::restore_extend` guarantees the sidecar backs EXACTLY the
    /// prefix the adapter reported, so no boundary re-derivation is needed —
    /// but every structural precondition is re-checked here anyway (group,
    /// layout equality against this config's geometry, boundary equal to the
    /// reported prefix). A contract slip must degrade to a MISS, i.e. a return
    /// of `None` that falls through to the caller's full replay, never to
    /// state installed at the wrong offset.
    ///
    /// Returns `None` when there is no sidecar, or when anything about it
    /// fails to line up. Taking the sidecar is unconditional so a rejected one
    /// cannot be reconsidered later in the same turn.
    fn install_gemma4_sliding_cold_sidecar(
        &mut self,
        cached_prefix_len: u32,
    ) -> Result<Option<Gemma4SlidingPrefixPreparation>> {
        let Some(adapter) = self.paged_adapter.as_mut() else {
            return Ok(None);
        };
        let Some(sidecar) = adapter.take_restored_sidecar() else {
            return Ok(None);
        };
        if sidecar.layout.group != mlx_paged_attn::ColdGroup::SlidingWindow {
            return Ok(None);
        }
        let boundary = sidecar.layout.boundary_tokens;
        // The walk reconciles the prefix and the state together, so the two
        // boundaries must be identical. Accepting a shallower sidecar would
        // create a global/sliding split-brain state; on an image turn it would
        // also invite replay across real vision embeddings. Refuse every
        // mismatch and let the caller restart cold.
        if !gemma4_sliding_cold_sidecar_matches_prefix(boundary, cached_prefix_len) {
            return Ok(None);
        }
        let Some(geometry) = sliding_sidecar::geometry(&self.config) else {
            return Ok(None);
        };
        // `load_sidecar` already compared the layout to the policy's template;
        // comparing again against the geometry derived from the LOADED config
        // makes the install independent of that earlier check.
        if sliding_sidecar::layout_at(&geometry, boundary) != sidecar.layout {
            return Ok(None);
        }
        let Some(snapshots) =
            sliding_sidecar::decode_snapshots(&self.config, &geometry, &sidecar.tensors, boundary)?
        else {
            return Ok(None);
        };
        let Some(caches) = restore_gemma4_sliding_caches(&self.config, &snapshots, boundary)?
        else {
            return Ok(None);
        };
        self.caches = Some(caches);
        // The one observable that separates "the tier restored the sliding half"
        // from "the tier read it and every arm above declined". Both produce
        // identical text, identical `num_tokens` and identical `cached_tokens`;
        // only the second re-forwards the whole prefix.
        crate::cold_tier::cold_sidecar_counters().record_install();
        Ok(Some(Gemma4SlidingPrefixPreparation {
            state: "cold_sidecar",
            primed_prefix_len: boundary,
        }))
    }

    /// Persist this turn's sliding-window state to the SSD cold tier, so a
    /// later process can resume from the paged prefix WITHOUT replaying the
    /// decoder over it (`run_sliding_only_prefill`).
    ///
    /// Best-effort and infallible by construction — every failure path is a
    /// silent skip. A missing sidecar is never a correctness problem: the
    /// restore walk simply reconciles the candidate prefix down past that
    /// boundary and the state is recomputed exactly as it is today.
    ///
    /// The boundary is the DEEPEST `B` that satisfies all of:
    ///
    ///  * `B` is a positive multiple of the paged block size (see
    ///    `sliding_sidecar::boundary_is_representable` — there is no window
    ///    floor: the payload carries `min(B, window)` rows, exactly what a live
    ///    rotating cache holds at that offset);
    ///  * the persisted K/V chain reaches `B` (`cold_captured_blocks`) — a
    ///    sidecar past the chain's break can never be selected, so writing one
    ///    would only burn quota;
    ///  * an already-materialized in-memory checkpoint sits exactly at `B` and
    ///    matches this request's tokens AND its per-block cache identity, so
    ///    the payload costs no extra forward and no extra `eval`.
    ///
    /// At most ONE sidecar per turn: the payload is `physical sliding layers ×
    /// 2 × min(B, window) × kv_heads × head_dim` elements — hundreds of MiB on
    /// a real checkpoint — and the writer queue is bounded.
    ///
    /// Pure-image turns use the same payload and image-aware key chain as their
    /// global K/V blocks, but apply one additional conservative rule: `B` must
    /// be at or after the complete expanded image run. This is stricter than the
    /// causal E2B warm-path policy (which can use an exact checkpoint inside an
    /// image run), deliberately: the first durable media implementation shares
    /// one fail-closed rule with unified bidirectional vision and never resumes
    /// from a half-image boundary.
    fn capture_gemma4_sliding_cold_sidecar(&self, context: Gemma4SlidingColdCaptureContext<'_>) {
        crate::cold_tier::cold_sidecar_counters().record_capture_reached();
        let media = match context.media {
            Gemma4SlidingColdCaptureMedia::Text => "text",
            Gemma4SlidingColdCaptureMedia::PureImage => "image",
        };
        let Some(minimum_safe_boundary) = context.minimum_safe_boundary() else {
            crate::cold_tier::cold_sidecar_counters().record_boundary_skip();
            if inference_trace_enabled() {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_cold_sidecar_capture_skipped reason=unsupported_media_capture_context media={} prompt_tokens={} image_tokens={}",
                    media,
                    context.prompt_len,
                    context.image_token_positions.len(),
                ));
            }
            return;
        };
        let Some(adapter) = self.paged_adapter.as_ref() else {
            return;
        };
        let Some(cold) = adapter.cold_tier() else {
            return;
        };
        let Some(policy) = cold.sidecar_policy.as_ref() else {
            return;
        };
        if policy.group() != mlx_paged_attn::ColdGroup::SlidingWindow {
            return;
        }
        let Some(geometry) = sliding_sidecar::geometry(&self.config) else {
            return;
        };
        let block_size = adapter.block_size();
        // The K/V capture walk that just ran spends its budget waiting for
        // writer-queue slots, so it hands this sidecar a queue it may well have
        // filled microseconds ago. A non-blocking offer here loses that race,
        // and a dropped sidecar is strictly worse than a dropped block: the
        // restore reconciles down to the deepest boundary a VALIDATED sidecar
        // backs, so losing it makes the turn's entire persisted K/V chain
        // unusable. Wait out the same budget the walk did.
        let sidecar_wait = adapter.cold_capture_budget().max_walk;
        if block_size == 0 {
            return;
        }
        let request_tokens = adapter.request_tokens();
        // Ceiling: whole blocks of this request that the persisted K/V chain
        // actually covers.
        let full_blocks = request_tokens.len() / block_size as usize;
        // ...and no deeper than a restore of this prompt could ever ASK for.
        //
        // The two sides count different sequences. This capture runs at
        // finalize, so `request_tokens` is the prompt plus everything the turn
        // generated; the restore that has to find this object runs at prepare,
        // over `prompt[..prompt_len - 1]`. Generated tokens do not widen that —
        // a later turn resends a prompt, not a prompt plus its own completion —
        // so the honest bound is the deepest boundary a restore of a prompt
        // ENDING HERE could probe, which is
        // `gemma4_cold_restore_reachable_boundary(prompt_len)`, not
        // `request_tokens.len()` rounded down.
        //
        // The bound is conservative in the one direction that matters. A
        // growing conversation (turn N+1's prompt = this turn's prompt plus its
        // completion plus new user text) could reach one block deeper, and gives
        // that block up here; a REPLAY of this exact prompt — which is what a
        // cold tier exists for, since its whole point is a later process — can
        // reach no further, and under the unclamped ceiling got back nothing at
        // all whenever `prompt_len` was block-aligned.
        let reachable_blocks =
            gemma4_cold_restore_reachable_boundary(context.prompt_len, block_size) as usize
                / block_size as usize;
        let chain_blocks = gemma4_sliding_cold_capture_ceiling_blocks(
            adapter.cold_captured_blocks(),
            request_tokens.len(),
            context.prompt_len,
            block_size,
        );
        if chain_blocks == 0 {
            // A different diagnosis from the checkpoint miss below: the
            // persisted chain covers no whole block of this request, so no
            // checkpoint at any depth could have been used. This is the arm a
            // genuinely cold process hits on its first turns, while the bounded
            // writer queue is still ratcheting the chain forward.
            crate::cold_tier::cold_sidecar_counters().record_chain_empty();
            if inference_trace_enabled() {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_cold_sidecar_capture_skipped reason=persisted_chain_covers_no_whole_block cold_captured_blocks={} full_blocks={} restore_reachable_blocks={} prompt_tokens={} block_size={} request_tokens={}",
                    adapter.cold_captured_blocks(),
                    full_blocks,
                    reachable_blocks,
                    context.prompt_len,
                    block_size,
                    request_tokens.len()
                ));
            }
            return;
        }
        let extra_keys_per_block = engine::build_paged_extra_keys(
            request_tokens.len(),
            block_size,
            context.image_token_positions,
        );

        // Every representable boundary an in-memory checkpoint backs, deepest
        // first. A LIST rather than the single deepest, because the deepest one
        // may already be on disk and this walk has to be able to keep going —
        // see the descent below.
        let candidates = self.find_gemma4_sliding_capture_checkpoints(
            &geometry,
            request_tokens,
            block_size,
            chain_blocks,
            &extra_keys_per_block,
            minimum_safe_boundary,
        );
        if candidates.is_empty() {
            // The one silent way this whole feature stays inert: a capture needs
            // an already-materialized checkpoint sitting exactly on a block
            // boundary AT OR BELOW the chain's reach, and the chain only
            // advances one writer-queue's worth of blocks per turn.
            //
            // The cadence alone cannot supply one. It fires every
            // `sliding_window` tokens, so its shallowest entry is one whole
            // window, and `Gemma4SlidingRetentionPolicy::PreLadder` evicts
            // oldest-first, so a prompt several windows long ends the prefill
            // holding only its DEEPEST couple of entries. Measured on
            // Gemma-4-12B-IT-nvidia-mxfp (window 1024, `limit` 2, 8140-token
            // prompt): the store finished at `{7168, 8128}` while the chain
            // reached 1136 — and the entry at 1024 had been born and evicted.
            //
            // `gemma4_sliding_cold_anchor_rungs` is what closes that gap on a
            // persist turn: a fixed `block_size * 4^k` grid, published by the
            // prefill and deferred by `Ladder` retention. This branch then
            // means the chain has not yet reached even the SHALLOWEST rung
            // (turn 1 of a cold process, before the queue has drained), or the
            // ladder is off. Trace it so that stays visible under MLX_TRACE
            // instead of looking like a working cache.
            crate::cold_tier::cold_sidecar_counters().record_boundary_skip();
            if inference_trace_enabled() {
                let reason = if context.media == Gemma4SlidingColdCaptureMedia::PureImage {
                    "no_exact_checkpoint_after_complete_image"
                } else {
                    "no_representable_checkpoint_at_or_below_chain_reach"
                };
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_cold_sidecar_capture_skipped reason={} media={} minimum_safe_boundary={} chain_reach_tokens={} chain_blocks={} block_size={} window={} request_tokens={} prompt_boundary={} prefix_checkpoints={} retained={:?} anchor_rungs={:?}",
                    reason,
                    media,
                    minimum_safe_boundary,
                    chain_blocks as u64 * block_size as u64,
                    chain_blocks,
                    block_size,
                    geometry.window,
                    request_tokens.len(),
                    self.sliding_prompt_boundary_checkpoint
                        .as_ref()
                        .map_or(0, |checkpoint| checkpoint.prefix_len),
                    self.sliding_prefix_checkpoints.len(),
                    self.sliding_prefix_checkpoints
                        .iter()
                        .map(|checkpoint| checkpoint.prefix_len)
                        .collect::<Vec<_>>(),
                    self.sliding_prefix_checkpoints
                        .iter()
                        .filter(|checkpoint| checkpoint.cold_anchor_rung)
                        .map(|checkpoint| checkpoint.prefix_len)
                        .collect::<Vec<_>>()
                ));
            }
            return;
        }
        // Descend the candidates, deepest first, past every boundary already on
        // disk, and capture the first one that is not. See
        // `gemma4_select_cold_capture_candidate` for why stopping at the first
        // already-persisted one is an absorbing state.
        //
        // The sidecar chain is the KV chain recomputed under the
        // `SlidingWindow` domain tag: identical per-block arguments, different
        // group (vLLM's `BlockHashWithGroupId`). `ColdTierWalk::
        // deepest_backed_boundary` derives the identical chain on restore.
        //
        // Derived BEFORE the payload is built so the dedup can skip the whole
        // encode: reading this state back off the GPU is hundreds of MiB on a
        // real checkpoint, and every later turn on the same prompt would
        // otherwise redo it and rewrite an object already on disk.
        let selection = gemma4_select_cold_capture_candidate(candidates, |(boundary, _)| {
            let Some(key) = gemma4_sliding_cold_sidecar_chain_key(
                cold.fingerprint,
                request_tokens,
                &extra_keys_per_block,
                block_size,
                *boundary,
            ) else {
                return Gemma4ColdCaptureProbe::Underivable;
            };
            if cold
                .manager
                .contains_in(&key, mlx_paged_attn::ColdGroup::SlidingWindow)
            {
                Gemma4ColdCaptureProbe::Persisted
            } else {
                Gemma4ColdCaptureProbe::Missing(key)
            }
        });
        let ((boundary, snapshots), key) = match selection {
            Gemma4ColdCaptureSelection::Capture { candidate, key, .. } => (candidate, key),
            // Every candidate is already on disk: nothing to do, and nothing
            // wrong. Mirrors `ColdTierWalk::capture_chain`'s `contains` dedup,
            // and `contains_in` is explicitly side-effect free (no hit/miss
            // accounting), so this arm must do its own — an unrecorded exit here
            // reads downstream as `enqueued=0`, which is also what a collapsed
            // rung ladder produces.
            Gemma4ColdCaptureSelection::AllPersisted { skipped_persisted } => {
                crate::cold_tier::cold_sidecar_counters().record_already_persisted();
                if inference_trace_enabled() {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 sliding_cold_sidecar_capture_skipped reason=every_candidate_already_persisted already_persisted={} chain_reach_tokens={} block_size={}",
                        skipped_persisted,
                        chain_blocks as u64 * block_size as u64,
                        block_size
                    ));
                }
                return;
            }
            // No candidate's chain derived, so the tier holds nothing here and
            // `already_persisted` would be a lie. Same diagnosis as an empty
            // candidate list above: no usable boundary under the chain's reach.
            Gemma4ColdCaptureSelection::NoChainDerived => {
                crate::cold_tier::cold_sidecar_counters().record_boundary_skip();
                if inference_trace_enabled() {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 sliding_cold_sidecar_capture_skipped reason=no_candidate_chain_derives chain_reach_tokens={} block_size={}",
                        chain_blocks as u64 * block_size as u64,
                        block_size
                    ));
                }
                return;
            }
        };

        let Ok(Some(tensors)) =
            sliding_sidecar::encode_tensors(&self.config, &geometry, snapshots, boundary)
        else {
            return;
        };
        let sidecar = mlx_paged_attn::ColdSidecar {
            key,
            fingerprint: cold.fingerprint,
            layout: sliding_sidecar::layout_at(&geometry, boundary),
            tensors,
        };
        match cold
            .manager
            .enqueue_sidecar_before(sidecar, std::time::Instant::now() + sidecar_wait)
        {
            Ok(true) => {
                crate::cold_tier::cold_sidecar_counters().record_enqueued();
                if context.media == Gemma4SlidingColdCaptureMedia::PureImage
                    && inference_trace_enabled()
                {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 sliding_cold_sidecar_capture_enqueued media=image boundary_tokens={} last_image_exclusive={}",
                        boundary, minimum_safe_boundary,
                    ));
                }
            }
            // The bounded writer queue stayed full for the whole capture
            // budget. Nothing is written and nothing failed, so this turn is
            // otherwise indistinguishable from a successful capture.
            Ok(false) => {
                crate::cold_tier::cold_sidecar_counters().record_queue_drop();
                tracing::debug!(
                    target: "mlx_core::gemma4::paged",
                    "Gemma4 sliding sidecar dropped at boundary {boundary}: cold-cache writer queue full"
                );
            }
            Err(error) => tracing::debug!(
                target: "mlx_core::gemma4::paged",
                "Gemma4 sliding sidecar enqueue failed at boundary {boundary}: {error}"
            ),
        }
    }

    /// Every already-materialized sliding checkpoint that can anchor a cold
    /// sidecar for `request_tokens`, with its snapshots, DEEPEST FIRST and
    /// deduplicated by boundary.
    ///
    /// Candidates must sit at a boundary this layout can express
    /// (`boundary_is_representable`), be covered by the persisted K/V chain
    /// (`<= chain_blocks * block_size`), and carry BOTH the exact token prefix
    /// and the exact per-block cache identity — the same `final_block_hash`
    /// the in-memory lookup path checks, so a checkpoint recorded under
    /// different image keys can never anchor a text sidecar.
    ///
    /// `sliding_cold_restore_tail_checkpoint` leads the chain because it is the
    /// deepest boundary a restore of this turn's prompt can name; the prompt
    /// boundary follows, and it is the SAME boundary except on a block-aligned
    /// prompt, where it sits one block past the tail and the ceiling below
    /// filters it out.
    fn find_gemma4_sliding_capture_checkpoints<'a>(
        &'a self,
        geometry: &sliding_sidecar::SlidingSidecarGeometry,
        request_tokens: &[u32],
        block_size: u32,
        chain_blocks: usize,
        extra_keys_per_block: &[Vec<u64>],
        minimum_safe_boundary: u32,
    ) -> Vec<(u32, &'a [Option<RotatingKVCacheSnapshot>])> {
        let ceiling = (chain_blocks as u64).saturating_mul(block_size as u64);
        let ceiling = u32::try_from(ceiling).unwrap_or(u32::MAX);
        let mut found: Vec<(u32, &[Option<RotatingKVCacheSnapshot>])> = Vec::new();
        let candidates = self
            .sliding_cold_restore_tail_checkpoint
            .iter()
            .chain(self.sliding_prompt_boundary_checkpoint.iter())
            .chain(self.sliding_prefix_checkpoints.iter());
        for checkpoint in candidates {
            let boundary = checkpoint.prefix_len;
            if boundary < minimum_safe_boundary
                || boundary > ceiling
                || checkpoint.block_size != block_size
                || !sliding_sidecar::boundary_is_representable(geometry, boundary, block_size)
                || found.iter().any(|(seen, _)| *seen == boundary)
            {
                continue;
            }
            if request_tokens.get(..boundary as usize) != Some(checkpoint.tokens.as_slice()) {
                continue;
            }
            let Some(final_block_hash) = compute_gemma4_paged_prefix_block_hash_with_keys(
                request_tokens,
                boundary,
                block_size,
                extra_keys_per_block,
                0,
            ) else {
                continue;
            };
            if checkpoint.final_block_hash != final_block_hash {
                continue;
            }
            if !gemma4_sliding_snapshots_ready_at(&self.config, &checkpoint.snapshots, boundary) {
                continue;
            }
            found.push((boundary, checkpoint.snapshots.as_slice()));
        }
        found.sort_unstable_by(|(left, _), (right, _)| right.cmp(left));
        found
    }

    /// Build the process-global SSD cold-tier context (manager + COMPLETE
    /// content fingerprint) for `model_path` WITHOUT attaching it, mirroring
    /// `Qwen3Inner::build_cold_tier_context` — see its doc for how the weight
    /// identity is established and why the caller brackets the load around it.
    ///
    /// The gemma4 difference is the [`mlx_paged_attn::ColdSidecarPolicy`]:
    /// gemma4's pool covers the FULL-ATTENTION layers only, so a K/V-only
    /// restore would resume from sliding-window state the pool never held. The
    /// policy turns the restore walk into vLLM's reconcile-down — the candidate
    /// prefix is reduced to the deepest boundary a validated sidecar backs, and
    /// a boundary nothing backs restores nothing.
    ///
    /// The sliding geometry is folded into the fingerprint explicitly:
    /// [`crate::cold_tier::ColdTierGeometry`] describes the POOL, which here
    /// covers only the global layers, so two configs differing ONLY in window
    /// size or sliding/global split would otherwise share a pool geometry.
    ///
    /// Returns `None` (fail-open) when the paged adapter is absent, the tier
    /// cannot be opened, this checkpoint has no sliding layers to persist, or a
    /// complete content fingerprint cannot be established.
    ///
    /// The `weights` witness pins this call after the loader's
    /// `materialize_weights` pass: MLX preads shard bytes lazily, so an identity
    /// read before that pass can describe bytes the model never runs.
    pub(crate) fn build_cold_tier_context(
        &self,
        model_path: &str,
        weights: &crate::array::memory::WeightsResident,
    ) -> Option<crate::transformer::paged_kv_cache_adapter::ColdTierContext> {
        let adapter = self.paged_adapter.as_ref()?;
        let manager = crate::cold_tier::global_cold_cache()?;
        // No sliding layers means no out-of-pool state — but it also means this
        // is not the hybrid gemma4 the sidecar work validated, so stay off
        // rather than silently behaving like a dense family.
        let geometry = sliding_sidecar::geometry(&self.config)?;
        let sidecar_policy = sliding_sidecar::policy(&self.config)?;
        let mut config_json = serde_json::to_vec(&self.config).ok()?;
        config_json.extend_from_slice(&geometry.fingerprint_component());
        let pool = adapter.layer_kv_pool();
        let pool_geometry = crate::cold_tier::ColdTierGeometry {
            block_size: pool.block_size() as u64,
            num_layers: pool.num_layers() as u64,
            num_kv_heads: pool.config().num_kv_heads as u64,
            head_size: pool.config().head_size as u64,
            cache_dtype: format!("{:?}", pool.cache_dtype()),
        };
        match crate::cold_tier::build_model_fingerprint(
            "gemma4",
            model_path,
            Some(&config_json),
            &pool_geometry,
            weights,
        ) {
            Some(fingerprint) => Some(
                crate::transformer::paged_kv_cache_adapter::ColdTierContext {
                    manager,
                    fingerprint,
                    sidecar_policy: Some(sidecar_policy),
                },
            ),
            None => {
                tracing::warn!(
                    "cold-tier persistence disabled for {model_path}: could not establish a \
                     content fingerprint (unreadable or missing weight shard)"
                );
                None
            }
        }
    }

    /// Attach a previously-built cold-tier context to the paged adapter. A
    /// no-op (fail-open) when the paged adapter is absent. Split from
    /// [`Self::build_cold_tier_context`] so the caller can verify shard
    /// identity is still stable AFTER the fingerprint read and BEFORE the cold
    /// tier is committed.
    ///
    /// Takes the same `materialize_weights` witness as the build step so the
    /// COMMIT point, not just the identity read, is compiler-pinned below
    /// materialization.
    pub(crate) fn attach_cold_tier(
        &mut self,
        ctx: crate::transformer::paged_kv_cache_adapter::ColdTierContext,
        _weights: &crate::array::memory::WeightsResident,
    ) {
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter.set_cold_tier(ctx);
        }
    }

    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    /// Decode + resize + patch raw image bytes and expand the rendered
    /// prompt's per-image `<|image|>` placeholders.
    ///
    /// The engine session core owns message-side image extraction
    /// (`engine::session::extract_images_from_messages`) and prompt
    /// rendering; the raw bytes arrive via [`WholeTurnArgs::media`].
    /// The "no vision support" rejection surfaces from INSIDE the vision
    /// turn (after render).
    fn prepare_vision_tokens(
        &self,
        rendered_tokens: &[u32],
        raw_images: &[Vec<u8>],
    ) -> Result<(
        Vec<u32>,
        Vec<ProcessedGemma4Image>,
        Option<u64>,
        Vec<(u32, u64)>,
    )> {
        let ip = self.image_processor.as_ref().ok_or_else(|| {
            Error::from_reason(
                "Images provided but model has no vision support (no vision_config in config.json)",
            )
        })?;
        let mut processed_images = Vec::with_capacity(raw_images.len());
        for bytes in raw_images {
            processed_images.push(ip.process_bytes(bytes)?);
        }

        // Compute the image cache key BEFORE the prefill so it can be
        // recorded on `self.cached_image_key` after the decode loop.
        // Session callers inspect this field to decide whether a
        // session-continue delta is allowed (text-only) or requires
        // a fresh `chat_session_start`.
        let (combined_image_key, per_image_hashes) = engine::compute_image_cache_keys(raw_images);
        let new_image_key = Some(combined_image_key);

        // Expand image tokens. Gemma4 uses: <|image>  (BOI) +
        // <|image|> × num_soft_tokens + <image|> (EOI). The chat template
        // inserts a single <|image|> per image; we expand it here.
        let image_token_id = self.config.image_token_id.unwrap_or(258880) as u32;
        let boi_token_id = self.config.boi_token_id.unwrap_or(255999) as u32;
        let eoi_token_id = self.config.eoi_token_id.unwrap_or(258882) as u32;
        let expanded = expand_image_tokens(
            rendered_tokens,
            &processed_images,
            image_token_id,
            boi_token_id,
            eoi_token_id,
        )?;

        let per_image_token_counts = processed_images
            .iter()
            .map(|image| image.num_soft_tokens as usize)
            .collect::<Vec<_>>();
        let image_token_positions = engine::map_expanded_image_token_positions(
            &expanded,
            image_token_id,
            &per_image_token_counts,
            &per_image_hashes,
        )
        .map_err(Error::from_reason)?;

        Ok((
            expanded,
            processed_images,
            new_image_key,
            image_token_positions,
        ))
    }

    /// Decode raw (encoded) audio bytes and expand the rendered prompt's
    /// per-clip `<|audio|>` placeholders into `boa + audio×n_frames + eoa`
    /// spans. The audio counterpart of [`Self::prepare_vision_tokens`].
    ///
    /// Each clip is decoded (`decode_wav_to_pcm`) into a mono 16 kHz f32
    /// waveform and framed (`frames_from_pcm`) into `[n_frames, 640]` raw
    /// windows; the per-clip frame counts drive `expand_audio_tokens`. All
    /// clips' frames are concatenated (axis 0) into a single
    /// `[total_frames, 640]` tensor so the merge scatter feeds them in order.
    /// `tokens` is the (possibly image-expanded) token stream; the audio
    /// expansion runs on top of it, leaving image spans untouched.
    fn prepare_audio_tokens(
        &self,
        tokens: &[u32],
        raw_audio: &[Vec<u8>],
    ) -> Result<(Vec<u32>, MxArray, Option<u64>)> {
        let spt = self.config.audio_samples_per_token.unwrap_or(640) as usize;
        let audio_token_id = self.config.audio_token_id.unwrap_or(258881) as u32;
        let boa_token_id = self.config.boa_token_id.unwrap_or(256000) as u32;
        let eoa_token_id = self.config.eoa_token_id.unwrap_or(258883) as u32;

        let mut per_clip_frames: Vec<MxArray> = Vec::with_capacity(raw_audio.len());
        let mut n_frames_per_clip: Vec<usize> = Vec::with_capacity(raw_audio.len());
        for bytes in raw_audio {
            let pcm = super::audio_processor::decode_wav_to_pcm(bytes)?;
            let frames = super::audio_processor::frames_from_pcm(&pcm, spt)?;
            let n = frames.shape_at(0)? as usize;
            n_frames_per_clip.push(n);
            per_clip_frames.push(frames);
        }

        let audio_frames = if per_clip_frames.len() == 1 {
            per_clip_frames.remove(0)
        } else {
            let refs: Vec<&MxArray> = per_clip_frames.iter().collect();
            MxArray::concatenate_many(refs, Some(0))?
        };

        let expanded = super::audio_processor::expand_audio_tokens(
            tokens,
            &n_frames_per_clip,
            audio_token_id,
            boa_token_id,
            eoa_token_id,
        )?;

        // Audio uses the same byte-identity cache key as images so an
        // audio-change cold-restarts the session server-side.
        let new_audio_key = Some(engine::compute_image_cache_key(raw_audio));

        Ok((expanded, audio_frames, new_audio_key))
    }

    /// Build the merged multimodal+text input embeddings for a prefill.
    ///
    /// Scatters image features (`@image_token_id`) AND audio features
    /// (`@audio_token_id`) into the SAME `sqrt(hidden)`-scaled text stream
    /// via chained `masked_scatter`s. Image-only turns skip the audio scatter
    /// (the image scatter math matches the prior vision-only prefill exactly);
    /// audio-only turns skip the image scatter. Returns `None` only when
    /// neither modality contributes features (text-only fallback).
    fn build_gemma4_multimodal_embeds(
        &self,
        prompt: &MxArray,
        processed_images: &[ProcessedGemma4Image],
        audio_frames: Option<&MxArray>,
    ) -> Result<Option<MxArray>> {
        let has_image_features = !processed_images.is_empty() && self.embed_vision.is_some();
        let has_audio_features = audio_frames.is_some() && self.embed_audio.is_some();
        if !has_image_features && !has_audio_features {
            return Ok(None);
        }

        // Base scaled text stream (built once; both scatters write into it).
        let text_embeds = self.embed_tokens.forward(prompt)?;
        let mut merged = text_embeds.mul_scalar((self.config.hidden_size as f64).sqrt())?;
        let embed_dtype = merged.dtype()?;

        // Image scatter @ image_token_id.
        if has_image_features {
            let ev = self.embed_vision.as_ref().unwrap();
            let image_token_id = self.config.image_token_id.unwrap_or(258880);
            let mut all_features: Vec<MxArray> = Vec::new();
            for proc in processed_images {
                let features = if let Some(vt) = self.vision_tower.as_ref() {
                    vt.forward(&proc.pixel_values)?
                } else if let Some(ve) = self.unified_vision_embedder.as_ref() {
                    let positions = proc.position_ids.as_ref().ok_or_else(|| {
                        Error::from_reason(
                            "Unified vision embedder requires per-patch position ids, but none \
                             were produced by the image processor.",
                        )
                    })?;
                    ve.forward(&proc.pixel_values, positions)?.expand_dims(0)?
                } else {
                    return Err(Error::from_reason(
                        "Image features requested but no vision tower / unified embedder present",
                    ));
                };
                all_features.push(ev.forward(&features)?);
            }
            let image_features = if all_features.len() == 1 {
                all_features.remove(0)
            } else {
                let refs: Vec<&MxArray> = all_features.iter().collect();
                MxArray::concatenate_many(refs, Some(1))?
            };
            let image_features = image_features.astype(embed_dtype)?;

            let image_token = MxArray::scalar_int(image_token_id)?;
            let image_mask = prompt.equal(&image_token)?;
            let mask_count_arr = image_mask.astype(DType::Int32)?.sum(None, None)?;
            mask_count_arr.eval();
            let mask_count = mask_count_arr.item_at_int32(0)? as i64;
            let feature_count = image_features.shape_at(1)?;
            if mask_count != feature_count {
                return Err(Error::new(
                    Status::GenericFailure,
                    format!(
                        "Image token count ({mask_count}) does not match vision feature count ({feature_count}). \
                         Check that image token expansion produced the correct number of tokens."
                    ),
                ));
            }
            let image_mask_expanded = image_mask.expand_dims(-1)?.broadcast_to(&merged.shape()?)?;
            merged = masked_scatter(&merged, &image_mask_expanded, &image_features)?;
        }

        // Audio scatter @ audio_token_id (CAUSAL; audio features unscaled).
        if has_audio_features {
            let ea = self.embed_audio.as_ref().unwrap();
            let audio_token_id = self.config.audio_token_id.unwrap_or(258881);
            let audio_features = ea.forward(audio_frames.unwrap())?.astype(embed_dtype)?;

            let audio_token = MxArray::scalar_int(audio_token_id)?;
            let audio_mask = prompt.equal(&audio_token)?;
            let mask_count_arr = audio_mask.astype(DType::Int32)?.sum(None, None)?;
            mask_count_arr.eval();
            let mask_count = mask_count_arr.item_at_int32(0)? as i64;
            let feature_count = audio_features.shape_at(0)?;
            if mask_count != feature_count {
                return Err(Error::new(
                    Status::GenericFailure,
                    format!(
                        "Audio token count ({mask_count}) does not match audio frame count ({feature_count}). \
                         Check that audio token expansion produced the correct number of frames."
                    ),
                ));
            }
            // Zero-frame audio has no scatter targets; leave the stream as-is
            // (a `masked_scatter` over an empty source would divide by zero).
            if feature_count > 0 {
                let audio_mask_expanded =
                    audio_mask.expand_dims(-1)?.broadcast_to(&merged.shape()?)?;
                merged = masked_scatter(&merged, &audio_mask_expanded, &audio_features)?;
            }
        }

        Ok(Some(merged))
    }

    /// Build only the embeddings the effective paged suffix will forward.
    /// When an image-aware hit already covers the complete image span, the
    /// suffix is text-only: avoid running SigLIP/unified vision entirely and
    /// embed just that suffix. Otherwise build the faithful full multimodal
    /// stream once and slice it at the effective cache boundary.
    fn build_gemma4_multimodal_suffix_embeds(
        &self,
        prompt: &MxArray,
        processed_images: &[ProcessedGemma4Image],
        audio_frames: Option<&MxArray>,
        cached_prefix_len: u32,
        last_image_exclusive: Option<u32>,
    ) -> Result<MxArray> {
        let prompt_len = u32::try_from(prompt.shape_at(1)?)
            .map_err(|_| Error::from_reason("Gemma4 prompt length exceeds u32"))?;
        if cached_prefix_len >= prompt_len {
            return Err(Error::from_reason(format!(
                "Gemma4 multimodal suffix is empty: cached_prefix_len={cached_prefix_len}, prompt_len={prompt_len}"
            )));
        }

        let image_span_fully_cached = audio_frames.is_none()
            && !processed_images.is_empty()
            && last_image_exclusive
                .is_some_and(|last_image_exclusive| cached_prefix_len >= last_image_exclusive);
        if image_span_fully_cached {
            let last_image_exclusive =
                last_image_exclusive.expect("fully cached image span has an endpoint");
            tracing::info!(
                target: "mlx_core::inference",
                event = "vlm_vision_tower_skip",
                model = "gemma4",
                cached_prefix_tokens = cached_prefix_len,
                last_image_exclusive,
                suffix_tokens = prompt_len - cached_prefix_len,
                "skipping Gemma4 vision tower because the image span is fully cached"
            );
            if inference_trace_enabled() {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 vlm_vision_tower_skip cached_prefix_tokens={} last_image_exclusive={} suffix_tokens={}",
                    cached_prefix_len,
                    last_image_exclusive,
                    prompt_len - cached_prefix_len,
                ));
            }
            let suffix = prompt.slice_axis(1, cached_prefix_len as i64, prompt_len as i64)?;
            return self
                .embed_tokens
                .forward(&suffix)?
                .mul_scalar((self.config.hidden_size as f64).sqrt());
        }

        let merged =
            match self.build_gemma4_multimodal_embeds(prompt, processed_images, audio_frames)? {
                Some(merged) => merged,
                None => self
                    .embed_tokens
                    .forward(prompt)?
                    .mul_scalar((self.config.hidden_size as f64).sqrt())?,
            };
        merged.slice_axis(1, cached_prefix_len as i64, prompt_len as i64)
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_gemma4_multimodal_paged_turn(
        &mut self,
        tokens: &[u32],
        prompt: &MxArray,
        processed_images: &[ProcessedGemma4Image],
        audio_frames: Option<&MxArray>,
        new_image_key: Option<u64>,
        new_audio_key: Option<u64>,
        image_token_positions: &[(u32, u64)],
        reuse_cache: bool,
    ) -> Result<Gemma4VlmTurnPreparation> {
        let layer_kinds = self.compute_layer_kinds()?;
        let total_budget = u32::try_from(tokens.len())
            .map_err(|_| Error::from_reason("Gemma4 multimodal prompt exceeds u32"))?;
        let block_size = self
            .paged_adapter
            .as_ref()
            .ok_or_else(|| {
                Error::from_reason("prepare_gemma4_multimodal_paged_turn: paged_adapter is None")
            })?
            .block_size();
        let image_only = new_image_key.is_some() && new_audio_key.is_none();
        let extra_keys_per_block = engine::build_paged_extra_keys(
            tokens.len(),
            block_size,
            if image_only {
                image_token_positions
            } else {
                &[]
            },
        );
        let last_image_exclusive = image_token_positions
            .last()
            .map(|(position, _)| position.saturating_add(1));

        let cached_prefix_len = if image_only {
            let image_token_id = self.config.image_token_id.unwrap_or(258880) as u32;
            let overlay_active = super::vision_mask::vision_overlay_active(
                self.config.is_unified,
                self.config.use_bidirectional_attention.as_deref() == Some("vision"),
                !image_token_positions.is_empty(),
                false,
                tokens.len(),
            );
            let allow_live_continue = reuse_cache
                && self.media_session_continuable
                && self.cached_image_key == new_image_key
                && self.cached_audio_key.is_none()
                && self.cached_paged_image_token_positions == image_token_positions;
            self.media_session_continuable = false;
            let resolution = self.prepare_gemma4_vlm_paged_prefix(
                tokens,
                total_budget,
                block_size,
                &extra_keys_per_block,
                image_token_positions,
                reuse_cache,
                allow_live_continue,
                if overlay_active {
                    last_image_exclusive
                } else {
                    None
                },
            )?;
            // The image placeholder is load-bearing for the image-only branch;
            // keep this check close to planning so a malformed expansion cannot
            // accidentally take the text-only tower-skip path.
            if !tokens.contains(&image_token_id) {
                self.invalidate_gemma4_hybrid_session(
                    "VLM image metadata had no expanded image placeholder",
                );
                return Err(Error::from_reason(
                    "Gemma4 image prompt contains no expanded image tokens",
                ));
            }
            resolution.effective_plan.cached_prefix_len
        } else {
            // Audio and mixed-media identity is not represented in the paged
            // block chain yet. Keep that path deliberately cold and do not
            // publish reusable prefix entries from its finalizer.
            self.media_session_continuable = false;
            let cold_plan = match self.paged_adapter.as_mut() {
                Some(adapter) => adapter
                    .prepare_turn_per_block_with_max_cache_hit_tokens(
                        0,
                        tokens,
                        total_budget,
                        false,
                        &extra_keys_per_block,
                        0,
                        true,
                        0,
                    )
                    .map_err(Error::from_reason),
                None => Err(Error::from_reason(
                    "prepare_gemma4_multimodal_paged_turn: paged_adapter is None",
                )),
            };
            if let Err(error) = cold_plan {
                self.invalidate_gemma4_hybrid_session(
                    "audio/mixed VLM cold paged preparation failure",
                );
                return Err(error);
            }
            self.caches = Some(init_caches_for_config(&self.config));
            self.cached_token_history.clear();
            self.cached_image_key = None;
            self.cached_audio_key = None;
            self.cached_paged_image_token_positions.clear();
            self.media_session_context = MediaCapabilities::NONE;
            self.paged_text_turn_context = MediaCapabilities::NONE;
            self.sliding_last_history_checkpoint = None;
            0
        };

        let suffix_embeds = match self.build_gemma4_multimodal_suffix_embeds(
            prompt,
            processed_images,
            audio_frames,
            cached_prefix_len,
            last_image_exclusive,
        ) {
            Ok(embeds) => embeds,
            Err(error) => {
                self.invalidate_gemma4_hybrid_session("VLM suffix embedding preparation failure");
                return Err(error);
            }
        };

        Ok(Gemma4VlmTurnPreparation {
            cached_prefix_len,
            suffix_embeds,
            layer_kinds,
            extra_keys_per_block,
            publish_prefix_checkpoints: gemma4_vlm_prefix_checkpoint_eligible(
                new_image_key.is_some(),
                new_audio_key.is_some(),
                reuse_cache,
            ),
        })
    }

    /// Prepare the merged multimodal prompt for a paged prefill: expand audio
    /// placeholders (when audio present) then image placeholders (when images
    /// present) on the rendered token stream, and decode/frame the audio.
    ///
    /// Each expansion replaces only the placeholder emitted by the checkpoint
    /// template. Missing or extra placeholders fail closed.
    ///
    /// Returns `(tokens, processed_images, audio_frames, new_image_key,
    /// new_audio_key, image_token_positions)`. Image-only turns never touch the audio path and leave
    /// `audio_frames`/`new_audio_key` as `None` (byte-identical to the old
    /// vision-only flow); audio-only turns never run the image processor and
    /// leave `processed_images` empty + `new_image_key` `None`.
    #[allow(clippy::type_complexity)]
    fn prepare_multimodal_tokens(
        &self,
        rendered_tokens: &[u32],
        raw_images: &[Vec<u8>],
        raw_audio: &[Vec<u8>],
    ) -> Result<(
        Vec<u32>,
        Vec<ProcessedGemma4Image>,
        Option<MxArray>,
        Option<u64>,
        Option<u64>,
        Vec<(u32, u64)>,
    )> {
        // Audio expansion first (only when audio present — keeps image-only
        // turns off the audio path and leaves `new_audio_key` None).
        let mut audio_frames: Option<MxArray> = None;
        let mut new_audio_key: Option<u64> = None;
        let tokens_after_audio = if raw_audio.is_empty() {
            rendered_tokens.to_vec()
        } else {
            let (expanded, frames, audio_key) =
                self.prepare_audio_tokens(rendered_tokens, raw_audio)?;
            audio_frames = Some(frames);
            new_audio_key = audio_key;
            expanded
        };

        // Image expansion only touches `<|image|>` ids, so audio spans are
        // inert to it.
        let (tokens, processed_images, new_image_key, image_token_positions) =
            if raw_images.is_empty() {
                (tokens_after_audio, Vec::new(), None, Vec::new())
            } else {
                self.prepare_vision_tokens(&tokens_after_audio, raw_images)?
            };

        Ok((
            tokens,
            processed_images,
            audio_frames,
            new_image_key,
            new_audio_key,
            image_token_positions,
        ))
    }

    /// Terminal media-state finalize shared by both vision cores (sync +
    /// stream), so the two stay byte-identical. Resolves the session into
    /// exactly ONE of two states, never partial:
    ///
    /// - **Continuable** (when `media_continuable` — currently a pure image
    ///   turn, including the unified bidirectional-vision image — AND
    ///   `reuse_cache`, AND `finalize_turn_keep_live_per_block` succeeds, AND the
    ///   sliding-history checkpoint actually `stored`, AND the adapter is
    ///   `live_for_continue`): the global paged KV is kept live (full blocks
    ///   registered for content-addressed reuse) and the marker is set so
    ///   the next full-history text continuation may reuse the global prefix
    ///   IN-PLACE (`continue_turn` keeps the block
    ///   table, `cachedTokens > 0`, only the new suffix is forwarded — it is NOT
    ///   re-walked) and the sliding caches resolve to `state="live"`
    ///   (`continued_live_prefix && gemma4_sliding_caches_ready_at`), so
    ///   `run_sliding_only_prefill` is skipped and no media position is ever
    ///   re-embedded from a raw `<|image|>`/`<|audio|>` id. Mirrors the
    ///   qwen3_5_moe two-state finalize.
    /// - **Non-continuable** (`reuse_cache=false`, a keep-live failure, or the
    ///   sliding checkpoint did not store / the adapter is not
    ///   `live_for_continue`): `release_request` only, keep history + media keys
    ///   live (marker stays false), so the next full-history continuation's
    ///   prefix verification forces a cold restart. The vision core does NOT
    ///   `reset_caches_sync` here, unlike the text/MoE path.
    ///
    /// ## Why `stored && live_for_continue` is the faithfulness gate
    /// `gemma4_sliding_caches_ready_at` requires every PHYSICAL non-shared
    /// sliding anchor cache to be populated. KV-shared alias slots (E2B's
    /// `SharedOnSliding`) intentionally hold no flat K/V and are skipped; their
    /// real anchor snapshots carry the state for both layers.
    /// A warm media→text continue is only numerically faithful when the media
    /// positions' sliding K/V can be reused IN PLACE: a text token's true
    /// embedding IS `embed_tokens.forward(id)` (replay-safe), but a media
    /// position's is a scattered SigLIP/audio feature that replay CANNOT rebuild
    /// from the raw special-token id. So the marker is armed ONLY when
    /// `stored && live_for_continue`: both non-shared checkpoints and E2B's
    /// physical anchors can store real K/V and warm-continue via `state="live"`.
    /// Missing/misaligned physical anchors still fail closed.
    ///
    /// ## R1 sliding-offset reconciliation (the length-finish materialize)
    /// The vision decode loop never forwards the final sampled token, so after
    /// the loop the live (non-shared) sliding caches AND the global paged KV sit
    /// at offset `prefill_len + G - 1`. The drop-last history rule yields
    /// `cached_token_history.len() == prefill_len + G - 1` on
    /// stop/repetition/cancelled (offsets MATCH) but `prefill_len + G` on a
    /// `"length"` finish (one short). On the continuable+`"length"` path we
    /// forward that final token once via `run_paged_decode_step` — exactly what
    /// the text path's `materialize_final` does (`paged_turn.rs` length gate →
    /// `Gemma4PagedDecode::materialize_final` → `run_paged_decode_step`) —
    /// advancing both caches to `prefill_len + G` so the kept-live global KV
    /// content-addresses against the saved history for the next delta's live
    /// restore. (Verified byte-exact by the non-unified-image warm==cold golden.)
    #[allow(clippy::too_many_arguments)]
    fn finalize_vision_turn_media_state(
        &mut self,
        expanded_tokens: &[u32],
        generated_tokens: &[u32],
        finish_reason: &str,
        new_image_key: Option<u64>,
        new_audio_key: Option<u64>,
        image_token_positions: &[(u32, u64)],
        media_continuable: bool,
        reuse_cache: bool,
    ) -> Result<()> {
        let continuable_eligible = reuse_cache && media_continuable;
        let is_length = finish_reason == "length";

        // Drop-last history (mirrors the non-continuable save the vision cores
        // do today and the text path's `save_paged_history`): keep all tokens on
        // a `"length"` finish, otherwise drop the terminal token.
        let history_tokens: &[u32] = if !is_length && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            generated_tokens
        };
        let mut full_history = Vec::with_capacity(expanded_tokens.len() + history_tokens.len());
        full_history.extend_from_slice(expanded_tokens);
        full_history.extend_from_slice(history_tokens);

        if continuable_eligible {
            // R1: align the sliding caches with the keep-all history before any
            // checkpoint. On a `"length"` finish the loop left the final token
            // unforwarded (offset == history.len() - 1); forward it now so both
            // the global paged KV and the sliding caches reach history.len().
            if is_length && let Some(&last_token) = generated_tokens.last() {
                // Forwards the token through the paged adapter + sliding caches.
                // A failure here aborts the turn before any state is published
                // (the request is still live; the caller's Err path releases it).
                let _logits = self.run_paged_decode_step(last_token)?;
            }

            let (keep_live_ok, live_for_continue) = match self.paged_adapter.as_mut() {
                Some(adapter) => {
                    let total = adapter.request_tokens().len();
                    let bs = adapter.block_size();
                    let extra = engine::build_paged_extra_keys(total, bs, image_token_positions);
                    let ok = match adapter.finalize_turn_keep_live_per_block(&extra, 0) {
                        Ok(_) => true,
                        Err(error) => {
                            tracing::warn!(
                                target: "mlx_core::gemma4::paged",
                                "Gemma4 image per-block finalize failed: {error}"
                            );
                            false
                        }
                    };
                    (ok, adapter.is_live_for_continue())
                }
                None => (false, false),
            };

            if keep_live_ok {
                // `finalize_turn_keep_live_per_block` has now published and
                // offered the image-aware GLOBAL K/V chain to the SSD writer.
                // Only after that succeeds may the out-of-pool sliding half be
                // offered, keyed from the same tokens/image positions. Carry the
                // VLM prompt length explicitly: this path bypasses the generic
                // text backend's `paged_turn_prompt_len` writer.
                if new_image_key.is_some()
                    && new_audio_key.is_none()
                    && !image_token_positions.is_empty()
                {
                    let prompt_len = u32::try_from(expanded_tokens.len()).map_err(|_| {
                        Error::from_reason("Gemma4 VLM prompt length exceeds u32 at finalize")
                    })?;
                    self.capture_gemma4_sliding_cold_sidecar(
                        Gemma4SlidingColdCaptureContext::pure_image(
                            prompt_len,
                            image_token_positions,
                        ),
                    );
                }

                // Publish history FIRST: the checkpoint reads its length, and
                // the next delta's prefix restore matches against it.
                self.cached_token_history = full_history;
                self.publish_media_session_context(new_image_key, new_audio_key);
                self.cached_paged_image_token_positions = image_token_positions.to_vec();
                let history_for_ckpt = self.cached_token_history.clone();
                let stored =
                    match self.remember_gemma4_sliding_history_checkpoint(&history_for_ckpt) {
                        Ok(trace) => trace.stored,
                        Err(error) => {
                            self.invalidate_gemma4_hybrid_session(
                                "VLM sliding-history checkpoint failure",
                            );
                            return Err(error);
                        }
                    };
                // Warm continuation is only faithful when the sliding state is
                // restorable from a stored checkpoint (or the in-place live
                // caches it implies). A text position's true embedding IS
                // `embed_tokens.forward(id)`, so REPLAY rebuilds it exactly. A
                // MEDIA position's true embedding is a scattered SigLIP/audio
                // feature that replay cannot reconstruct from the raw
                // `<|image|>`/`<|audio|>` special-token id. KV-shared alias
                // slots hold no flat K/V, so checkpoint readiness/snapshotting
                // intentionally skips them and persists only their physical
                // sliding anchors. If any physical anchor is unavailable,
                // `stored == false` and the turn downgrades to a clean
                // non-continuable state rather than replaying media ids.
                //
                // `live_for_continue` guards a second gap: a keep-live with zero
                // FULL blocks (a media turn shorter than `block_size`) returns
                // Ok without registering the request, so the next delta could
                // not take the live-continue path and would re-prefill the media
                // placeholders as text. Unreachable on shipped configs (media
                // turns far exceed the 16-token block), but cheap to gate.
                if stored && live_for_continue {
                    self.media_session_continuable = true;
                    return Ok(());
                }
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    let _ = adapter.release_request();
                }
                self.media_session_continuable = false;
                return Ok(());
            }
            // keep-live failed: fall through to the non-continuable teardown.
            self.invalidate_gemma4_hybrid_session("VLM per-block finalize failure");
            return Err(Error::from_reason(
                "Gemma4 image paged finalize failed; reusable state was invalidated",
            ));
        }

        // Non-continuable: release the global KV but keep history + media keys
        // so full-history prefix verification forces the next continuation to
        // cold-restart (marker is false).
        if let Some(adapter) = self.paged_adapter.as_mut() {
            let _ = adapter.release_request();
        }
        self.cached_token_history = full_history;
        self.publish_media_session_context(new_image_key, new_audio_key);
        self.cached_paged_image_token_positions.clear();
        self.media_session_continuable = false;
        Ok(())
    }

    /// Vision (VLM) whole-turn core over the BLOCK-PAGED backend,
    /// non-streaming.
    ///
    /// Shared multimodal prep (`prepare_multimodal_tokens` to expand
    /// `<|image|>` / `<|audio|>` placeholders, `build_gemma4_multimodal_embeds`
    /// to `masked_scatter` image+audio features into the residual) writes
    /// full-attention K/V into the paged adapter pool. Sliding layers still use
    /// the flat rotating caches.
    ///
    /// Image-only prompts use image-aware per-block keys plus exact sliding
    /// checkpoints. Audio and mixed-media prompts remain deliberately cold.
    fn vision_paged_turn_sync_core(
        &mut self,
        rendered_tokens: &[u32],
        raw_images: &[Vec<u8>],
        raw_audio: &[Vec<u8>],
        tokenizer: &Arc<Qwen3Tokenizer>,
        config: &ChatConfig,
        eos_token_id: u32,
    ) -> Result<ChatResult> {
        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
        let (
            tokens,
            processed_images,
            audio_frames,
            new_image_key,
            new_audio_key,
            image_token_positions,
        ) = self.prepare_multimodal_tokens(rendered_tokens, raw_images, raw_audio)?;
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        let sampling_config = make_sampling_config(config, &self.config);
        let repetition_cutoff = repetition_cutoff_from_config(config);
        let eos_ids = self.config.eos_token_ids.clone();

        let prefill_slice: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let prefill_len = prefill_slice.len();
        let prompt = MxArray::from_int32(&prefill_slice, &[1, prefill_len as i64])?;
        let prompt_token_count = tokens.len();

        let generation_stream = Stream::new(DeviceType::Gpu);
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        let generation_start = std::time::Instant::now();
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let turn = self.prepare_gemma4_multimodal_paged_turn(
            &tokens,
            &prompt,
            &processed_images,
            audio_frames.as_ref(),
            new_image_key,
            new_audio_key,
            &image_token_positions,
            reuse_cache,
        )?;
        let cached_prefix_len = turn.cached_prefix_len;

        let forward_result = (|| -> Result<(Vec<u32>, String)> {
            let last_logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                crate::models::gemma4::diagnostic::set_step(-1);
                self.run_paged_vlm_prefill(
                    &tokens,
                    &turn.suffix_embeds,
                    &turn.layer_kinds,
                    cached_prefix_len,
                    &turn.extra_keys_per_block,
                    &image_token_positions,
                    turn.publish_prefix_checkpoints,
                )?
            };

            crate::array::synchronize_and_clear_cache();

            let mut y = sample_next_token(&last_logits, sampling_config)?;
            y.eval();

            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut finish_reason = String::from("length");

            for step in 0..max_new_tokens {
                let token_id = y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);

                if is_eos_token(token_id, &eos_ids, eos_token_id) {
                    finish_reason = String::from("stop");
                    break;
                }
                if let Some(reason) =
                    check_gemma4_repetition_cutoff(&generated_tokens, repetition_cutoff)
                {
                    finish_reason = reason.to_string();
                    break;
                }
                if step + 1 >= max_new_tokens {
                    break;
                }

                let next_logits = {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    crate::models::gemma4::diagnostic::set_step(step);
                    self.run_paged_decode_step(token_id)?
                };
                let next_logits = next_logits.squeeze(Some(&[1]))?;
                y = sample_next_token(&next_logits, sampling_config)?;
                y.eval();

                crate::array::maybe_clear_cache_for_paged_step(step);
            }

            Ok((generated_tokens, finish_reason))
        })();

        // The Ok branch does NOT release the request here — the media-state
        // finalize decides between keep-live (continuable) and release
        // (non-continuable). The Err branch still releases fully.
        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => t,
            Err(e) => {
                self.invalidate_gemma4_hybrid_session("VLM sync forward/decode failure");
                return Err(e);
            }
        };

        let first_token_instant = std::time::Instant::now();

        let raw_text = match tokenizer.decode_sync(&generated_tokens, false) {
            Ok(text) => text,
            Err(error) => {
                self.invalidate_gemma4_hybrid_session("VLM sync tokenizer decode failure");
                return Err(error);
            }
        };

        // Two-state media finalize: keep the global paged KV live + remember a
        // sliding history checkpoint when this is a pure image turn under
        // reuse, so a follow-up text delta
        // warm-continues; otherwise release + keep history/keys so the guard
        // rejects (single-shot, as today). A finalize Err means the live
        // request must be released before returning.
        let media_continuable =
            gemma4_media_continuable(new_image_key.is_some(), new_audio_key.is_some());
        if let Err(e) = self.finalize_vision_turn_media_state(
            &tokens,
            &generated_tokens,
            &finish_reason,
            new_image_key,
            new_audio_key,
            &image_token_positions,
            media_continuable,
            reuse_cache,
        ) {
            self.invalidate_gemma4_hybrid_session("VLM sync finalize failure");
            return Err(e);
        }

        let generation_end = std::time::Instant::now();
        let ttft_ms = first_token_instant
            .duration_since(generation_start)
            .as_secs_f64()
            * 1000.0;
        let decode_ms = generation_end
            .duration_since(first_token_instant)
            .as_secs_f64()
            * 1000.0;
        let gen_toks = generated_tokens.len() as f64;

        let performance = Some(crate::profiling::PerformanceMetrics {
            ttft_ms,
            prefill_tokens_per_second: if ttft_ms > 0.0 {
                (prefill_len.saturating_sub(cached_prefix_len as usize)) as f64 / (ttft_ms / 1000.0)
            } else {
                0.0
            },
            decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                (gen_toks - 1.0) / (decode_ms / 1000.0)
            } else {
                0.0
            },
            mtp_mean_accepted_tokens: None,
            mtp_mean_accepted_tokens_total: None,
            mtp_acceptance_by_position: None,
            mtp_cycles: None,
            mtp_mean_depth: None,
            profile_phases: None,
        });

        let starts_in_prompted_channel = self.output_starts_in_reasoning_channel();
        let mut parsed = super::output_parser::parse_gemma4_output_with_open_channel(
            &raw_text,
            starts_in_prompted_channel,
        );
        promote_channel_only_output(&mut parsed, starts_in_prompted_channel);
        let finish_reason = if parsed.tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        Ok(ChatResult {
            text: parsed.text,
            tool_calls: parsed.tool_calls,
            thinking: parsed.thinking,
            thinking_enabled: engine::resolve_enable_thinking(config).unwrap_or(true),
            num_tokens: generated_tokens.len() as u32,
            prompt_tokens: prompt_token_count as u32,
            reasoning_tokens: 0,
            finish_reason,
            raw_text,
            public_raw_text: None,
            cached_tokens: cached_prefix_len,
            performance,
        })
    }

    /// Streaming twin of [`Self::vision_paged_turn_sync_core`]. Same paged
    /// prefill + decode spine; streams parser segments and emits the terminal
    /// chunk itself.
    #[allow(clippy::too_many_arguments)]
    fn vision_paged_turn_stream_core(
        &mut self,
        rendered_tokens: &[u32],
        raw_images: &[Vec<u8>],
        raw_audio: &[Vec<u8>],
        tokenizer: &Arc<Qwen3Tokenizer>,
        config: &ChatConfig,
        eos_token_id: u32,
        sink: &dyn ChunkSink,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        let cb = StreamSender(sink);
        let max_new_tokens = config.max_new_tokens.unwrap_or(2048);
        let (
            tokens,
            processed_images,
            audio_frames,
            new_image_key,
            new_audio_key,
            image_token_positions,
        ) = self.prepare_multimodal_tokens(rendered_tokens, raw_images, raw_audio)?;
        if tokens.is_empty() {
            return Err(Error::from_reason("Empty prompt"));
        }
        let sampling_config = make_sampling_config(config, &self.config);
        let repetition_cutoff = repetition_cutoff_from_config(config);
        let eos_ids = self.config.eos_token_ids.clone();

        let prefill_slice: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let prefill_len = prefill_slice.len();
        let prompt = MxArray::from_int32(&prefill_slice, &[1, prefill_len as i64])?;
        let prompt_token_count = tokens.len();

        let generation_stream = Stream::new(DeviceType::Gpu);
        let _wired_ctx = crate::stream::WiredLimitContext::new(usize::MAX, vec![generation_stream]);

        let generation_start = std::time::Instant::now();
        let reuse_cache = config.reuse_cache.unwrap_or(true);
        let turn = self.prepare_gemma4_multimodal_paged_turn(
            &tokens,
            &prompt,
            &processed_images,
            audio_frames.as_ref(),
            new_image_key,
            new_audio_key,
            &image_token_positions,
            reuse_cache,
        )?;
        let cached_prefix_len = turn.cached_prefix_len;

        let mut decode_stream = tokenizer.inner().decode_stream(false);
        let mut streamed_text_len = 0;
        let starts_in_prompted_channel = self.output_starts_in_reasoning_channel();
        let mut stream_parser = super::output_parser::Gemma4StreamParser::new_with_open_channel(
            starts_in_prompted_channel,
        );
        let mut stream_dispatch = Gemma4StreamDispatchState::new(starts_in_prompted_channel);

        let forward_result = (|| -> Result<(Vec<u32>, String)> {
            let last_logits = {
                let _stream_ctx = StreamContext::new(generation_stream);
                crate::models::gemma4::diagnostic::set_step(-1);
                self.run_paged_vlm_prefill(
                    &tokens,
                    &turn.suffix_embeds,
                    &turn.layer_kinds,
                    cached_prefix_len,
                    &turn.extra_keys_per_block,
                    &image_token_positions,
                    turn.publish_prefix_checkpoints,
                )?
            };

            crate::array::synchronize_and_clear_cache();

            let mut y = sample_next_token(&last_logits, sampling_config)?;
            y.eval();

            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut finish_reason = String::from("length");

            for step in 0..max_new_tokens {
                let token_id = y.item_at_int32(0)? as u32;
                generated_tokens.push(token_id);

                if cancelled.load(Ordering::Relaxed) {
                    finish_reason = "cancelled".to_string();
                    break;
                }

                let token_text = Qwen3Tokenizer::step_decode_stream(
                    &mut decode_stream,
                    tokenizer.inner(),
                    token_id,
                    &generated_tokens,
                    streamed_text_len,
                );
                streamed_text_len += token_text.len();
                let segments = stream_parser.feed(&token_text);
                stream_dispatch.dispatch_segments(segments, &cb);

                if is_eos_token(token_id, &eos_ids, eos_token_id) {
                    finish_reason = "stop".to_string();
                    break;
                }
                if let Some(reason) =
                    check_gemma4_repetition_cutoff(&generated_tokens, repetition_cutoff)
                {
                    finish_reason = reason.to_string();
                    break;
                }
                if step + 1 >= max_new_tokens {
                    break;
                }

                let next_logits = {
                    let _stream_ctx = StreamContext::new(generation_stream);
                    crate::models::gemma4::diagnostic::set_step(step);
                    self.run_paged_decode_step(token_id)?
                };
                let next_logits = next_logits.squeeze(Some(&[1]))?;
                y = sample_next_token(&next_logits, sampling_config)?;
                y.eval();

                crate::array::maybe_clear_cache_for_paged_step(step);
            }

            Ok((generated_tokens, finish_reason))
        })();

        // The Ok branch does NOT release the request here — the media-state
        // finalize decides between keep-live (continuable) and release
        // (non-continuable). The Err branch still releases fully.
        let (generated_tokens, finish_reason) = match forward_result {
            Ok(t) => t,
            Err(e) => {
                self.invalidate_gemma4_hybrid_session("VLM stream forward/decode failure");
                return Err(e);
            }
        };

        let first_token_instant = std::time::Instant::now();

        let raw_text = match tokenizer.decode_sync(&generated_tokens, false) {
            Ok(text) => text,
            Err(error) => {
                self.invalidate_gemma4_hybrid_session("VLM stream tokenizer decode failure");
                return Err(error);
            }
        };

        // Flush residual bytes through the stream parser.
        if raw_text.len() > streamed_text_len {
            let residual = raw_text[streamed_text_len..].to_string();
            let mut segments = stream_parser.feed(&residual);
            segments.extend(stream_parser.flush());
            stream_dispatch.dispatch_segments(segments, &cb);
        } else {
            let tail = stream_parser.flush();
            stream_dispatch.dispatch_segments(tail, &cb);
        }
        stream_dispatch.finish(&cb);

        // Two-state media finalize (identical to the sync core via the shared
        // helper): keep-live + sliding checkpoint for a continuable pure-causal
        // media turn, else release + keep history/keys so the guard rejects.
        let media_continuable =
            gemma4_media_continuable(new_image_key.is_some(), new_audio_key.is_some());
        if let Err(e) = self.finalize_vision_turn_media_state(
            &tokens,
            &generated_tokens,
            &finish_reason,
            new_image_key,
            new_audio_key,
            &image_token_positions,
            media_continuable,
            reuse_cache,
        ) {
            self.invalidate_gemma4_hybrid_session("VLM stream finalize failure");
            return Err(e);
        }

        let generation_end = std::time::Instant::now();
        let ttft_ms = first_token_instant
            .duration_since(generation_start)
            .as_secs_f64()
            * 1000.0;
        let decode_ms = generation_end
            .duration_since(first_token_instant)
            .as_secs_f64()
            * 1000.0;
        let gen_toks = generated_tokens.len() as f64;

        let performance = Some(crate::profiling::PerformanceMetrics {
            ttft_ms,
            prefill_tokens_per_second: if ttft_ms > 0.0 {
                (prefill_len.saturating_sub(cached_prefix_len as usize)) as f64 / (ttft_ms / 1000.0)
            } else {
                0.0
            },
            decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
                (gen_toks - 1.0) / (decode_ms / 1000.0)
            } else {
                0.0
            },
            mtp_mean_accepted_tokens: None,
            mtp_mean_accepted_tokens_total: None,
            mtp_acceptance_by_position: None,
            mtp_cycles: None,
            mtp_mean_depth: None,
            profile_phases: None,
        });

        let parsed_tool_calls = stream_parser.tool_calls();
        let parsed_thinking = stream_parser.thinking();
        let finish_reason = if parsed_tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            finish_reason
        };

        cb.call(
            Ok(ChatStreamChunk {
                text: String::new(),
                done: true,
                finish_reason: Some(finish_reason),
                tool_calls: Some(parsed_tool_calls),
                thinking: parsed_thinking,
                thinking_enabled: Some(engine::resolve_enable_thinking(config).unwrap_or(true)),
                num_tokens: Some(generated_tokens.len() as u32),
                prompt_tokens: Some(prompt_token_count as u32),
                reasoning_tokens: Some(0),
                raw_text: Some(raw_text),
                public_raw_text: None,
                text_authoritative: Some(false),
                cached_tokens: Some(cached_prefix_len),
                performance,
                is_reasoning: None,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );

        Ok(())
    }

    // =================================================================
    // Block-paged dispatch (paged_turn_sync_core + helpers).
    //
    // Mirrors Qwen3's `paged_turn_sync_core` and LFM2's `forward_paged_or_flat`
    // pattern — sliding layers continue to use the existing flat
    // `Gemma4LayerCache::Sliding` path while global layers route through
    // `PagedKVCacheAdapter`. KV-shared layers are routed through their
    // anchor's physical KV slot/stash using routes derived from
    // `LayerKVCacheSpec`.
    //
    // Lifecycle (mirrors Qwen3 / LFM2):
    // 1. Adapter cold-start (or warm-continue when previous turn
    //    finalize_turn_keep_live'd a strict-prefix request).
    // 2. Sliding caches are restored from the live turn/checkpoint when
    //    available; otherwise the cached prefix is replayed through sliding
    //    layers before suffix prefill.
    // 3. Prefill via `run_paged_prefill_chunk` over the suffix.
    // 4. Decode loop via `run_paged_decode_step`.
    // 5. End-of-turn (success): `finalize_turn_keep_live` so the next
    //    turn's `continue_turn` can build on top of the partial trailing
    //    block's K/V (same partial-block carry trick as Qwen3 / LFM2).
    //
    // Caveats / scope:
    // * Text-only — vision turns dispatch through the flat path.
    // * Sliding layers still use flat rotating caches; true paged sliding
    //   storage is a separate kernel/storage step.
    // * Exact prefix hits are capped at `prompt_len - 1` so the final
    //   prompt token is always recomputed to produce logits.
    // =================================================================

    fn suppress_large_sliding_prefix_reuse_if_needed(
        &mut self,
        trace_label: &str,
        tokens: &[u32],
        total_budget: u32,
        seq_id: u32,
        restore_tokens: u32,
        trace_enabled: bool,
    ) -> Result<bool> {
        let block_size = self
            .paged_adapter
            .as_ref()
            .map(|adapter| adapter.block_size())
            .unwrap_or(0);
        let Some(suppression) = gemma4_large_sliding_restore_suppression_limit(
            &self.config,
            block_size,
            restore_tokens,
        ) else {
            return Ok(false);
        };

        // Sliding layers are recursive: without a close sliding checkpoint,
        // restoring a large paged-prefix hit can be slower than simply
        // recomputing the prompt. Keep prefix reuse only when the missing
        // sliding delta fits within the normal checkpoint interval; otherwise
        // fall back to a coherent cold prefill. Operators can set
        // MLX_GEMMA4_MAX_SLIDING_RESTORE_TOKENS=off for debugging.
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 {}_prefix_reuse_suppressed reason=large_sliding_restore_limit restore_tokens={} limit={} limit_source={} block_size={}",
                trace_label, restore_tokens, suppression.limit, suppression.source, block_size
            ));
        }
        let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
            Error::from_reason(format!(
                "{}: paged_adapter is None while suppressing large sliding restore",
                trace_label
            ))
        })?;
        let _ = adapter.release_request();
        adapter
            .reset_for_new_request(seq_id)
            .map_err(Error::from_reason)?;
        let prefix = adapter
            .find_cached_prefix(tokens, &[], 0, true)
            .map_err(Error::from_reason)?;
        let allocated = adapter
            .allocate_suffix_blocks(total_budget)
            .map_err(Error::from_reason)?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 {}_adapter_reset_done reason=large_sliding_restore_suppressed cached_prefix_tokens={} cached_blocks={} allocated_blocks={} request_tokens={} blocks={}",
                trace_label,
                prefix.cached_token_count,
                prefix.blocks.len(),
                allocated,
                adapter.current_token_count(),
                adapter.num_allocated_blocks()
            ));
        }
        // Counted at the point of no return, not at the decision: everything
        // above between here and the `return Ok(false)` can fail the turn with
        // `?`, and a suppression that errored out is not a suppression that
        // happened. The trace above fires earlier on purpose — it is the
        // diagnosis, and it has to survive the failure it might be explaining.
        //
        // The counter exists because the trace alone cannot answer the
        // question. `MLX_TRACE` is opt-in and per-process, so without a number
        // a suppression firing on every turn is indistinguishable from a cache
        // that is simply cold: both end with the whole prompt recomputed, both
        // leave text and token counts identical, and this one is preceded by
        // real `hits` that make it look like reuse worked.
        crate::cold_tier::cold_sidecar_counters().record_restore_suppressed();
        Ok(true)
    }

    fn invalidate_gemma4_hybrid_session(&mut self, reason: &'static str) {
        tracing::warn!(
            target: "mlx_core::gemma4::paged",
            reason,
            "invalidating Gemma4 hybrid paged/sliding session"
        );
        if let Some(adapter) = self.paged_adapter.as_mut() {
            let _ = adapter.release_request();
        }
        self.caches = None;
        self.clear_reuse_state();
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_gemma4_vlm_paged_prefix(
        &mut self,
        tokens: &[u32],
        total_budget: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        image_token_positions: &[(u32, u64)],
        reuse_cache: bool,
        allow_live_continue: bool,
        unified_overlay_last_image_exclusive: Option<u32>,
    ) -> Result<engine::VlmPagedPrefixResolution> {
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let candidate_plan_result = match self.paged_adapter.as_mut() {
            Some(adapter) => adapter
                .prepare_turn_per_block_with_max_cache_hit_tokens(
                    0,
                    tokens,
                    total_budget,
                    allow_live_continue,
                    extra_keys_per_block,
                    0,
                    !reuse_cache,
                    max_cache_hit_tokens,
                )
                .map_err(Error::from_reason),
            None => Err(Error::from_reason(
                "prepare_gemma4_vlm_paged_prefix: paged_adapter is None",
            )),
        };
        let candidate_plan = match candidate_plan_result {
            Ok(plan) => plan,
            Err(error) => {
                self.invalidate_gemma4_hybrid_session("VLM paged-prefix preparation failure");
                return Err(error);
            }
        };

        // Unified image K/V encodes the bidirectional overlay. The existing
        // attention API can consume an already-materialized overlay prefix only
        // when the cached boundary is after the complete expanded image run.
        // A candidate before/inside that run must be discarded, even if its
        // token/image block hash matches.
        let first_image_position = image_token_positions.first().map(|(position, _)| *position);
        let prefix_policy = gemma4_vlm_prefix_policy(
            candidate_plan.cached_prefix_len,
            first_image_position,
            unified_overlay_last_image_exclusive,
        );
        let sliding_preparation = if prefix_policy.unified_boundary_safe {
            self.prepare_gemma4_sliding_prefix_state_with_keys(
                tokens,
                candidate_plan.cached_prefix_len,
                candidate_plan.continued_live_prefix,
                extra_keys_per_block,
                image_token_positions,
                prefix_policy.require_exact_checkpoint,
            )
        } else {
            self.caches = Some(init_caches_for_config(&self.config));
            Ok(Gemma4SlidingPrefixPreparation {
                state: "unified_image_boundary_unsafe",
                primed_prefix_len: 0,
            })
        };
        let mut sliding_preparation = match sliding_preparation {
            Ok(preparation) => preparation,
            Err(error) => {
                self.invalidate_gemma4_hybrid_session("VLM sliding-prefix preparation failure");
                return Err(error);
            }
        };
        // A causal E2B hit ending before the first image token is pure text.
        // Reconstruct only that missing physical sliding prefix from token
        // embeddings; once a candidate includes an image position, replay is
        // forbidden because the placeholder embedding is not the vision feature.
        if prefix_policy.may_replay_leading_text
            && sliding_preparation.primed_prefix_len < candidate_plan.cached_prefix_len
        {
            let replay_result = (|| -> Result<()> {
                let replay = tokens
                    .get(
                        sliding_preparation.primed_prefix_len as usize
                            ..candidate_plan.cached_prefix_len as usize,
                    )
                    .ok_or_else(|| {
                        Error::from_reason("Gemma4 leading-text sliding replay range is invalid")
                    })?;
                let layer_kinds = self.compute_layer_kinds()?;
                self.run_sliding_only_prefill(
                    replay,
                    sliding_preparation.primed_prefix_len,
                    &layer_kinds,
                )?;
                self.remember_gemma4_sliding_materialized_prefix_checkpoint_with_keys(
                    tokens,
                    candidate_plan.cached_prefix_len,
                    block_size,
                    extra_keys_per_block,
                    0,
                )?;
                Ok(())
            })();
            if let Err(error) = replay_result {
                self.invalidate_gemma4_hybrid_session("VLM leading-text replay failure");
                return Err(error);
            }
            sliding_preparation.primed_prefix_len = candidate_plan.cached_prefix_len;
            sliding_preparation.state = "leading_text_replay";
        }
        let sliding_prefix_exact = prefix_policy.unified_boundary_safe
            && sliding_preparation.primed_prefix_len == candidate_plan.cached_prefix_len;

        let resolution =
            engine::resolve_vlm_paged_prefix(candidate_plan, sliding_prefix_exact, || {
                self.paged_adapter
                    .as_mut()
                    .ok_or_else(|| {
                        "prepare_gemma4_vlm_paged_prefix: adapter dropped before cold restart"
                            .to_string()
                    })?
                    .restart_prepared_turn_cold_per_block(
                        0,
                        tokens,
                        total_budget,
                        extra_keys_per_block,
                        0,
                    )
            });
        let resolution = match resolution {
            Ok(resolution) => resolution,
            Err(error) => {
                self.invalidate_gemma4_hybrid_session("VLM paged cold-restart failure");
                return Err(Error::from_reason(error));
            }
        };

        // The prepared request now owns the new turn. Clear only live/history
        // state; retain the bounded image-aware prefix checkpoints so A -> B -> A
        // can restore A after B displaced the live request.
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_audio_key = None;
        self.cached_paged_image_token_positions = image_token_positions.to_vec();
        self.media_session_context = MediaCapabilities::NONE;
        self.paged_text_turn_context = MediaCapabilities::NONE;
        self.sliding_last_history_checkpoint = None;
        self.media_session_continuable = false;

        tracing::info!(
            target: "mlx_core::inference",
            event = "vlm_prefix_plan",
            model = "gemma4",
            prompt_tokens = tokens.len(),
            image_tokens = image_token_positions.len(),
            candidate_cached_prefix_tokens = resolution.candidate_cached_prefix_len,
            effective_cached_prefix_tokens = resolution.effective_plan.cached_prefix_len,
            continued_live_prefix = resolution.effective_plan.continued_live_prefix,
            sliding_prefix_exact,
            unified_boundary_safe = prefix_policy.unified_boundary_safe,
            downgraded_to_cold = resolution.downgraded_to_cold,
            "image-aware Gemma4 paged prefix planned"
        );

        Ok(resolution)
    }

    fn prepare_gemma4_paged_turn(
        &mut self,
        trace_label: &str,
        tokens: &[u32],
        reuse_cache: bool,
        total_budget: u32,
        seq_id: u32,
        trace_enabled: bool,
    ) -> Result<Gemma4PagedTurnPreparation> {
        let image_token_id = self.config.image_token_id.unwrap_or(258880) as u32;
        let audio_token_id = self.config.audio_token_id.unwrap_or(258881) as u32;
        let prompt_holds_media =
            prompt_holds_media_placeholders(tokens, image_token_id, audio_token_id);
        let block_size = self
            .paged_adapter
            .as_ref()
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "{trace_label}: paged_adapter is None while preparing paged turn"
                ))
            })?
            .block_size();
        let carries_image_lineage = gemma4_carries_image_lineage(
            self.paged_text_turn_context,
            self.cached_image_key,
            &self.cached_paged_image_token_positions,
            &self.cached_token_history,
            tokens,
        );
        let image_token_positions = if carries_image_lineage {
            self.cached_paged_image_token_positions.clone()
        } else {
            Vec::new()
        };
        let extra_keys_per_block =
            engine::build_paged_extra_keys(tokens.len(), block_size, &image_token_positions);
        let plan = {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason(format!(
                    "{trace_label}: paged_adapter is None while preparing paged turn"
                ))
            })?;
            let adapter_live = adapter.is_live_for_continue();
            let adapter_request_tokens = adapter.request_tokens().len();
            let adapter_common_prefix = tokens
                .iter()
                .zip(adapter.request_tokens().iter())
                .take_while(|(a, b)| a == b)
                .count();
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 {trace_label}_adapter_prepare live={} request_tokens={} common_prefix={} total_budget={} reuse_cache={}",
                    adapter_live,
                    adapter_request_tokens,
                    adapter_common_prefix,
                    total_budget,
                    reuse_cache
                ));
            }
            let max_cache_hit_tokens = total_budget.saturating_sub(1);
            // Unknown media placeholders remain lookup-disabled. A warm text
            // continuation carrying an exact persisted image lineage uses the
            // same per-block keys as the original image turn, so neither lookup
            // nor finalize can republish those blocks token-only.
            let skip_lookup = prompt_holds_media && !carries_image_lineage;
            let plan = adapter
                .prepare_turn_per_block_with_max_cache_hit_tokens(
                    seq_id,
                    tokens,
                    total_budget,
                    reuse_cache && (!prompt_holds_media || carries_image_lineage),
                    &extra_keys_per_block,
                    0,
                    skip_lookup,
                    max_cache_hit_tokens,
                )
                .map_err(Error::from_reason)?;
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 {trace_label}_adapter_prepare_done reason={:?} cached_prefix_tokens={} cached_blocks={} allocated_blocks={} request_tokens={} blocks={} continued_live={}",
                    plan.reason,
                    plan.cached_prefix_len,
                    plan.cached_blocks,
                    plan.allocated_blocks,
                    adapter.current_token_count(),
                    adapter.num_allocated_blocks(),
                    plan.continued_live_prefix
                ));
            }
            plan
        };

        let mut cached_prefix_len = plan.cached_prefix_len;
        let mut sliding_preparation = self.prepare_gemma4_sliding_prefix_state_with_keys(
            tokens,
            cached_prefix_len,
            plan.continued_live_prefix,
            &extra_keys_per_block,
            &image_token_positions,
            carries_image_lineage,
        )?;
        if carries_image_lineage && sliding_preparation.primed_prefix_len < cached_prefix_len {
            if let Some(adapter) = self.paged_adapter.as_mut() {
                let _ = adapter.release_request();
            }
            return Err(Error::from_reason(format!(
                "{}{trace_label} lost the exact image-aware sliding checkpoint",
                engine::IMAGE_CHANGE_RESTART_PREFIX
            )));
        }
        if sliding_preparation.primed_prefix_len < cached_prefix_len {
            let suppressed = self.suppress_large_sliding_prefix_reuse_if_needed(
                trace_label,
                tokens,
                total_budget,
                seq_id,
                cached_prefix_len.saturating_sub(sliding_preparation.primed_prefix_len),
                trace_enabled,
            )?;
            if suppressed {
                let previous_cached_prefix_len = cached_prefix_len;
                cached_prefix_len = 0;
                sliding_preparation = self.prepare_gemma4_sliding_prefix_state_with_keys(
                    tokens,
                    cached_prefix_len,
                    false,
                    &extra_keys_per_block,
                    &image_token_positions,
                    false,
                )?;
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 {trace_label}_cached_prefix_reset previous_cached_prefix_tokens={} reason=sliding_restore_limit",
                        previous_cached_prefix_len
                    ));
                }
            }
        }

        let suffix_len = total_budget.checked_sub(cached_prefix_len).ok_or_else(|| {
            Error::from_reason(format!(
                "{trace_label}: cached_prefix_len {cached_prefix_len} exceeds total_budget \
                 {total_budget}"
            ))
        })?;
        if trace_enabled {
            let already_primed = sliding_preparation.primed_prefix_len == cached_prefix_len;
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 {trace_label}_sliding_prefix_state state={} cached_prefix_tokens={} sliding_primed_prefix_tokens={} replay_delta_tokens={} suffix_tokens={} already_primed={} continued_live={}",
                sliding_preparation.state,
                cached_prefix_len,
                sliding_preparation.primed_prefix_len,
                cached_prefix_len.saturating_sub(sliding_preparation.primed_prefix_len),
                suffix_len,
                already_primed,
                plan.continued_live_prefix
            ));
        }

        if !carries_image_lineage {
            self.cached_image_key = None;
            self.cached_paged_image_token_positions.clear();
        }

        Ok(Gemma4PagedTurnPreparation {
            cached_prefix_len,
            suffix_len,
            sliding_primed_prefix_len: sliding_preparation.primed_prefix_len,
        })
    }

    /// Run a paged-attention prefill over the full prompt, dispatching
    /// per-layer between the adapter (global layers) and the existing
    /// flat path (sliding layers).
    ///
    /// `full_tokens` is the entire prompt (sliding layers re-prefill
    /// from token 0). `suffix_tokens` is the new portion beyond the
    /// paged prefix-cache hit (used by `record_tokens` +
    /// `update_keys_values` for global layers). `cached_prefix_len`
    /// is the paged-cache hit length.
    ///
    /// Returns the last position's logits squeezed to `[vocab]`.
    ///
    /// ## Prefill split (parity with the flat path)
    ///
    /// The flat path's `prefill_body_gemma4` processes tokens
    /// `[0..N-1]` through `forward_body`, then the caller runs a
    /// SECOND, single-token `forward_inner` for the final token. That
    /// second dispatch is load-bearing — see the doc-comment on
    /// `prefill_body_gemma4`: "SDPA computes slightly different
    /// numerical results for multi-token causal attention vs
    /// single-token attention with cached K/V. These small differences
    /// compound through layers, causing divergent logits if the last
    /// prompt token is processed in the same batch as the rest."
    ///
    /// This function mirrors that split for the paged path so the
    /// K/V-cache reduction order at the prefill→decode boundary
    /// matches between flat and paged. Without the split, BF16 SDPA
    /// drift on the last layer's hidden state at step 0 (~1%) flips
    /// argmax to a nearby zero-embedding `<unused>` token, causing the
    /// `<turn|>` stop signal to be missed and the decoder to fall into
    /// the all-zero-input cycle (`mean(V)` attention output → `id+1`
    /// counting cascade).
    fn run_paged_prefill_chunk(
        &mut self,
        full_tokens: &[u32],
        suffix_tokens: &[u32],
        cached_prefix_len: u32,
        sliding_primed_prefix_len: u32,
    ) -> Result<MxArray> {
        if suffix_tokens.is_empty() {
            return Err(Error::from_reason(
                "run_paged_prefill_chunk called with empty suffix",
            ));
        }
        if sliding_primed_prefix_len > cached_prefix_len {
            return Err(Error::from_reason(format!(
                "Gemma4 paged prefill sliding_primed_prefix_len {} exceeds cached_prefix_len {}",
                sliding_primed_prefix_len, cached_prefix_len
            )));
        }

        let suffix_len = suffix_tokens.len() as u32;
        let layer_kinds = self.compute_layer_kinds()?;
        let trace_enabled = inference_trace_enabled();
        let trace_start = trace_enabled.then(std::time::Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_start full_tokens={} cached_prefix_tokens={} suffix_tokens={}",
                full_tokens.len(),
                cached_prefix_len,
                suffix_tokens.len()
            ));
        }

        // For sliding layers we need state at position cached_prefix_len.
        // Sliding layers are restored each turn via reset_caches_sync, so
        // we need to reprefill any unprimed cached-prefix delta through
        // them BEFORE the suffix can attend. When a sparse checkpoint hits,
        // this is only the delta from that checkpoint to cached_prefix_len.
        if sliding_primed_prefix_len < cached_prefix_len {
            let prefix =
                &full_tokens[(sliding_primed_prefix_len as usize)..(cached_prefix_len as usize)];
            let sliding_trace_start = trace_enabled.then(std::time::Instant::now);
            self.run_sliding_only_prefill(prefix, sliding_primed_prefix_len, &layer_kinds)?;
            let block_size = self
                .paged_adapter
                .as_ref()
                .map(|adapter| adapter.block_size())
                .unwrap_or(0);
            let store_trace = self.remember_gemma4_sliding_prefix_checkpoint(
                full_tokens,
                cached_prefix_len,
                block_size,
                0,
            )?;
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 paged_prefill_sliding_prefix_done cached_prefix_tokens={} restored_prefix_tokens={} replay_tokens={} checkpoint_stored={} store_eval_ms={:.1} store_snapshot_ms={:.1} store_token_clone_ms={:.1} store_update_ms={:.1} store_ms={:.1} elapsed_ms={:.1}",
                    cached_prefix_len,
                    sliding_primed_prefix_len,
                    prefix.len(),
                    store_trace.stored,
                    store_trace.eval_ms,
                    store_trace.snapshot_ms,
                    store_trace.token_clone_ms,
                    store_trace.update_ms,
                    store_trace.total_ms,
                    sliding_trace_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
        } else if cached_prefix_len > 0 && trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_sliding_prefix_skipped cached_prefix_tokens={} sliding_primed_prefix_tokens={} reason=already_primed",
                cached_prefix_len, sliding_primed_prefix_len
            ));
        }
        // Sliding-window state now covers the whole cached prefix (either it
        // already did, or the replay above just extended it to
        // `cached_prefix_len`). Discharge the adapter's auxiliary-state
        // obligation before the first `record_tokens` of the turn.
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter
                .confirm_aux_prefix_primed(cached_prefix_len)
                .map_err(Error::from_reason)?;
        }

        crate::models::gemma4::diagnostic::set_path("paged");
        crate::models::gemma4::diagnostic::set_step(-1);

        // Two-pass split (mirrors flat `prefill_body_gemma4 →
        // forward_inner`):
        //   Pass 1: tokens `[..suffix_len-1]` (no-op if suffix_len == 1).
        //           Run this body in bounded chunks so long-context paged
        //           prefill does not build a single enormous lazy graph before
        //           the first cache materialization.
        //   Pass 2: the FINAL token (length 1). Now
        //           `cached_prefix_len_for_chunk = cached_prefix_len +
        //           suffix_len - 1`, which is > 0, so global layers
        //           take the graph-native single-token paged-attention branch
        //           used by decode. This aligns the reduction order with the
        //           paged `forward_inner` dispatch.
        let configured_chunk_size = crate::array::paged_prefill_chunk_size();
        let mut pass2_first_position = cached_prefix_len;
        if suffix_len > 1 {
            // --- Pass 1: all-but-last suffix tokens, chunked. ---
            let pass1_tokens = &suffix_tokens[..(suffix_len as usize - 1)];
            let num_query_heads = u32::try_from(self.config.num_attention_heads).map_err(|_| {
                Error::from_reason(format!(
                    "Gemma4 paged prefill invalid num_attention_heads={}",
                    self.config.num_attention_heads
                ))
            })?;
            let global_head_size =
                u32::try_from(self.config.effective_head_dim(true)).map_err(|_| {
                    Error::from_reason(format!(
                        "Gemma4 paged prefill invalid global head_dim={}",
                        self.config.effective_head_dim(true)
                    ))
                })?;
            let num_kv_heads =
                u32::try_from(self.config.effective_kv_heads(true)).map_err(|_| {
                    Error::from_reason(format!(
                        "Gemma4 paged prefill invalid global num_kv_heads={}",
                        self.config.effective_kv_heads(true)
                    ))
                })?;
            let route_policy = gemma4_paged_prefill_route_policy();
            let block_size = self
                .paged_adapter
                .as_ref()
                .map(|adapter| adapter.block_size())
                .unwrap_or(0);
            let full_tokens_len = u32::try_from(full_tokens.len())
                .map_err(|_| Error::from_reason("Gemma4 paged prefill token count exceeds u32"))?;
            let prompt_checkpoint_boundary_len = full_tokens_len
                .checked_div(block_size)
                .map(|blocks| blocks.saturating_mul(block_size))
                .unwrap_or(0);
            // Anchor rungs for the cold sidecar. Derived once for the whole
            // prefill (they are a pure function of the config and the block
            // size) and, deliberately, NOT fed to
            // `gemma4_split_body_chunk_plan_at_position` below: a rung is
            // snapshotted from the temporal K/V view the chunk already
            // produced, so publishing one is numerically transparent, while
            // splitting the chunk at it would change every downstream GEMM's
            // `M` and with it the accumulation order.
            let sliding_caps = self.gemma4_sliding_retention_caps_for_turn(block_size);
            let cold_restore_tail_boundary =
                gemma4_cold_restore_tail_publish(full_tokens_len, block_size, sliding_caps);
            // Strictly a per-turn artifact: it costs one window of sliding K/V
            // and only this turn's capture can use it.
            self.sliding_cold_restore_tail_checkpoint = None;
            // Compute chunks follow the configured prefill step directly, as
            // in the authoritative mlx-lm generator. Sliding checkpoints are
            // captured from temporal K/V views inside a chunk; they must not
            // split a 2K matrix-prefill into 1K work.
            let mut body_chunk_plan = gemma4_paged_prefill_body_chunk_plan(
                configured_chunk_size,
                pass1_tokens.len(),
                pass2_first_position,
                num_query_heads,
                num_kv_heads,
                global_head_size,
                route_policy,
            )?;
            gemma4_split_body_chunk_plan_at_position(
                &mut body_chunk_plan,
                prompt_checkpoint_boundary_len,
            );
            let total_body_chunks = body_chunk_plan.len();
            let first_body_chunk_size = body_chunk_plan.first().map(|chunk| chunk.len).unwrap_or(0);
            let min_body_chunk_size = body_chunk_plan
                .iter()
                .map(|chunk| chunk.len)
                .min()
                .unwrap_or(0);
            let max_body_chunk_size = body_chunk_plan
                .iter()
                .map(|chunk| chunk.len)
                .max()
                .unwrap_or(0);
            let dynamic_v2_aux_caps = body_chunk_plan
                .iter()
                .filter(|chunk| chunk.capped_by_v2_aux_limit)
                .count();
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 paged_prefill_body_chunking body_tokens={} chunk_size={} configured_chunk_size={} chunks={} min_chunk_size={} max_chunk_size={} dynamic_v2_aux_caps={} route_policy={:?}",
                    pass1_tokens.len(),
                    first_body_chunk_size,
                    configured_chunk_size,
                    total_body_chunks,
                    min_body_chunk_size,
                    max_body_chunk_size,
                    dynamic_v2_aux_caps,
                    route_policy
                ));
            }
            for (chunk_idx, chunk_plan) in body_chunk_plan.iter().enumerate() {
                let chunk_end = chunk_plan
                    .start
                    .checked_add(chunk_plan.len)
                    .ok_or_else(|| Error::from_reason("Gemma4 paged prefill chunk end overflow"))?;
                let chunk = pass1_tokens
                    .get(chunk_plan.start..chunk_end)
                    .ok_or_else(|| {
                        Error::from_reason("Gemma4 paged prefill chunk plan out of range")
                    })?;
                let chunk_first_position = chunk_plan.first_position;
                debug_assert_eq!(chunk_first_position, pass2_first_position);
                let chunk_end_position = chunk_first_position
                    .checked_add(chunk.len() as u32)
                    .ok_or_else(|| {
                        Error::from_reason("Gemma4 paged prefill chunk position overflow")
                    })?;
                let checkpoint_interval =
                    gemma4_sliding_decode_checkpoint_interval(&self.config, block_size);
                let mut checkpoint_boundaries = gemma4_sliding_chunk_checkpoint_boundaries(
                    chunk_first_position,
                    chunk_end_position,
                    checkpoint_interval,
                    sliding_caps,
                );
                // The prompt boundary is already a real compute endpoint and
                // is stored by the dedicated protected/prompt checkpoint path.
                // It leaves the list here, which is why the tail below cannot
                // use `already_published` to notice that it coincides with the
                // prompt boundary — `gemma4_cold_restore_tail_publish` screens
                // that case out instead.
                checkpoint_boundaries
                    .retain(|&boundary| boundary != prompt_checkpoint_boundary_len);
                let chunk_cold_restore_tail = gemma4_chunk_cold_restore_tail(
                    cold_restore_tail_boundary,
                    chunk_first_position,
                    chunk_end_position,
                    &checkpoint_boundaries,
                );
                if let Some(boundary) = chunk_cold_restore_tail {
                    // `prepare_sliding_checkpoint_capture` rejects offsets that
                    // are not strictly increasing.
                    checkpoint_boundaries.push(boundary);
                    checkpoint_boundaries.sort_unstable();
                }
                if !checkpoint_boundaries.is_empty() {
                    let caches = self.caches.as_mut().ok_or_else(|| {
                        Error::from_reason("Gemma4 paged prefill sliding checkpoint caches missing")
                    })?;
                    prepare_gemma4_sliding_checkpoint_captures(
                        &self.config,
                        caches,
                        &checkpoint_boundaries,
                    )?;
                }
                let chunk_trace_start = trace_enabled.then(std::time::Instant::now);
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 paged_prefill_body_chunk_start chunk={}/{} first_position={} tokens={} capped_by_v2_aux_limit={} checkpoint_interval={} captured_checkpoint_boundaries={:?} cold_ladder={} anchor_rungs={:?}",
                        chunk_idx + 1,
                        total_body_chunks,
                        chunk_first_position,
                        chunk.len(),
                        chunk_plan.capped_by_v2_aux_limit,
                        checkpoint_interval,
                        checkpoint_boundaries,
                        sliding_caps.wants_ladder(),
                        sliding_caps.anchors.as_slice()
                    ));
                }
                {
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason("run_paged_prefill_chunk: paged_adapter is None")
                    })?;
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] gemma4 paged_prefill_record_tokens_start chunk={}/{} first_position={} tokens={} current_tokens_before={} blocks_before={}",
                            chunk_idx + 1,
                            total_body_chunks,
                            chunk_first_position,
                            chunk.len(),
                            adapter.current_token_count(),
                            adapter.num_allocated_blocks()
                        ));
                    }
                    adapter.record_tokens(chunk).map_err(Error::from_reason)?;
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] gemma4 paged_prefill_record_tokens_done chunk={}/{} current_tokens_after={} blocks_after={}",
                            chunk_idx + 1,
                            total_body_chunks,
                            adapter.current_token_count(),
                            adapter.num_allocated_blocks()
                        ));
                    }
                }
                let layer_loop_start = trace_enabled.then(std::time::Instant::now);
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 paged_prefill_layer_loop_start chunk={}/{} first_position={} cached_prefix_for_chunk={} tokens={}",
                        chunk_idx + 1,
                        total_body_chunks,
                        chunk_first_position,
                        chunk_first_position,
                        chunk.len()
                    ));
                }
                let _hidden_pass1 = self.run_paged_prefill_layer_loop(
                    chunk,
                    chunk_first_position,
                    chunk_first_position,
                    &layer_kinds,
                )?;
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    adapter
                        .eval_pending_pool_writes()
                        .map_err(Error::from_reason)?;
                }
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 paged_prefill_layer_loop_done chunk={}/{} first_position={} tokens={} elapsed_ms={:.1}",
                        chunk_idx + 1,
                        total_body_chunks,
                        chunk_first_position,
                        chunk.len(),
                        layer_loop_start.map(elapsed_ms).unwrap_or(0.0)
                    ));
                }

                // Materialize writes from this body chunk before the next
                // chunk reads through them. Native paged writes are lazy graph
                // nodes; sliding flat caches are lazy too.
                if let Some(caches) = self.caches.as_ref() {
                    eval_gemma4_caches(caches)?;
                }
                let captured_checkpoints = if checkpoint_boundaries.is_empty() {
                    Vec::new()
                } else {
                    let caches = self.caches.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "Gemma4 paged prefill sliding checkpoint caches missing post-forward",
                        )
                    })?;
                    take_gemma4_sliding_checkpoint_captures(
                        &self.config,
                        caches,
                        &checkpoint_boundaries,
                    )?
                };
                for (&boundary, snapshots) in checkpoint_boundaries.iter().zip(captured_checkpoints)
                {
                    let is_anchor_rung = sliding_caps.anchors.contains(boundary);
                    let sink = if chunk_cold_restore_tail == Some(boundary) {
                        Gemma4SlidingCapturedCheckpointSink::ColdRestoreTail
                    } else {
                        Gemma4SlidingCapturedCheckpointSink::PrefixStore
                    };
                    let store_trace = self.remember_gemma4_sliding_captured_prefix_checkpoint(
                        full_tokens,
                        boundary,
                        block_size,
                        0,
                        snapshots,
                        sink,
                    )?;
                    if trace_enabled {
                        write_inference_trace(format_args!(
                            "[MLX_TRACE] gemma4 paged_prefill_sliding_captured_checkpoint boundary_tokens={} block_size={} checkpoint_interval={} cold_anchor_rung={} cold_restore_tail={} stored={} materialize_ms={:.1} token_clone_ms={:.1} update_ms={:.1} total_ms={:.1}",
                            boundary,
                            block_size,
                            checkpoint_interval,
                            is_anchor_rung,
                            sink == Gemma4SlidingCapturedCheckpointSink::ColdRestoreTail,
                            store_trace.stored,
                            store_trace.eval_ms,
                            store_trace.token_clone_ms,
                            store_trace.update_ms,
                            store_trace.total_ms
                        ));
                    }
                }
                crate::array::clear_cache();
                pass2_first_position = pass2_first_position
                    .checked_add(chunk.len() as u32)
                    .ok_or_else(|| {
                        Error::from_reason("Gemma4 paged prefill token position overflow")
                    })?;
                if pass2_first_position == prompt_checkpoint_boundary_len {
                    self.maybe_remember_gemma4_sliding_prompt_boundary_checkpoint(
                        "paged_prefill",
                        full_tokens,
                        prompt_checkpoint_boundary_len,
                        trace_enabled,
                    )?;
                }
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 paged_prefill_body_chunk_done chunk={}/{} next_position={} elapsed_ms={:.1}",
                        chunk_idx + 1,
                        total_body_chunks,
                        pass2_first_position,
                        chunk_trace_start.map(elapsed_ms).unwrap_or(0.0)
                    ));
                }
            }
        }

        // --- Pass 2: the FINAL suffix token (length 1). ---
        let pass2_tokens = &suffix_tokens[(suffix_len as usize - 1)..];
        {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_prefill_chunk: paged_adapter is None")
            })?;
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 paged_prefill_final_record_tokens_start first_position={} tokens={} current_tokens_before={} blocks_before={}",
                    pass2_first_position,
                    pass2_tokens.len(),
                    adapter.current_token_count(),
                    adapter.num_allocated_blocks()
                ));
            }
            adapter
                .record_tokens(pass2_tokens)
                .map_err(Error::from_reason)?;
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 paged_prefill_final_record_tokens_done current_tokens_after={} blocks_after={}",
                    adapter.current_token_count(),
                    adapter.num_allocated_blocks()
                ));
            }
        }
        let pass2_cached_prefix_len = pass2_first_position;
        let pass2_layer_loop_start = trace_enabled.then(std::time::Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_final_layer_loop_start first_position={} cached_prefix_for_chunk={} tokens={}",
                pass2_first_position,
                pass2_cached_prefix_len,
                pass2_tokens.len()
            ));
        }
        let mut hidden_states = self.run_paged_prefill_layer_loop(
            pass2_tokens,
            pass2_first_position,
            pass2_cached_prefix_len,
            &layer_kinds,
        )?;
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter
                .eval_pending_pool_writes()
                .map_err(Error::from_reason)?;
        }
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_final_layer_loop_done first_position={} tokens={} elapsed_ms={:.1}",
                pass2_first_position,
                pass2_tokens.len(),
                pass2_layer_loop_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }

        self.maybe_remember_gemma4_sliding_prompt_boundary_checkpoint(
            "paged_prefill",
            full_tokens,
            pass2_first_position + pass2_tokens.len() as u32,
            trace_enabled,
        )?;

        // Final norm + lm_head + softcap (only for the final token).
        hidden_states = self.final_norm.forward(&hidden_states)?;
        crate::models::gemma4::diagnostic::dump_norm(0, "post_final_norm", &hidden_states, None);
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&hidden_states)?
        } else if self.embed_tokens.is_packed_quantized() {
            // Packed tied lm_head: project through the quantized matmul without
            // materializing the dense table.
            self.embed_tokens.as_linear(&hidden_states)?
        } else if let Some(ref w_t) = self.embed_weight_t {
            hidden_states.matmul(w_t)?
        } else {
            let weight = self.embed_tokens.get_weight();
            let weight_t = weight.transpose(Some(&[1, 0]))?;
            hidden_states.matmul(&weight_t)?
        };
        crate::models::gemma4::diagnostic::dump_logits("pre_softcap", &logits);
        let logits = if let Some(cap) = self.config.final_logit_softcapping {
            let cap_arr = MxArray::scalar_float_like(cap, &logits)?;
            let handle = unsafe { mlx_sys::mlx_logit_softcap(logits.handle.0, cap_arr.handle.0) };
            let capped = MxArray::from_handle(handle, "logit_softcap")?;
            crate::models::gemma4::diagnostic::dump_logits("post_softcap", &capped);
            capped
        } else {
            crate::models::gemma4::diagnostic::dump_logits("post_softcap", &logits);
            logits
        };

        let last_seq_len = logits.shape_at(1)?;
        let last = logits
            .slice_axis(1, last_seq_len - 1, last_seq_len)?
            .squeeze(Some(&[0, 1]))?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_done suffix_tokens={} elapsed_ms={:.1}",
                suffix_tokens.len(),
                trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        Ok(last)
    }

    /// One forward pass through the embed → PLE → layer-loop pipeline
    /// for a single contiguous chunk of tokens. Returns the chunk's
    /// post-final-layer hidden state (NO final norm / lm_head / softcap
    /// — the caller decides whether to apply those).
    ///
    /// `chunk_tokens` is the slice being processed THIS call.
    /// `first_logical_position` is the absolute logical position of
    /// `chunk_tokens[0]` in the request (used as the RoPE offset and
    /// the slot-mapping anchor). `cached_prefix_len_for_chunk` is the
    /// number of K/V tokens already in the paged pool BEFORE this
    /// chunk's writes — when this is > 0 global attention adaptively chooses
    /// graph-native pool gather + SDPA or compact varlen PagedAttention while
    /// retaining the same physical paged storage. `layer_kinds` is the
    /// per-layer routing classification (Sliding / GlobalPaged /
    /// SharedOnGlobal / SharedOnSliding).
    ///
    /// Caller must have already called `record_tokens(chunk_tokens)`
    /// on the paged adapter so `update_keys_values`'s alignment check
    /// (`first_logical_position == current_token_count - chunk.len()`)
    /// passes.
    fn run_paged_prefill_layer_loop(
        &mut self,
        chunk_tokens: &[u32],
        first_logical_position: u32,
        cached_prefix_len_for_chunk: u32,
        layer_kinds: &[Gemma4LayerKind],
    ) -> Result<MxArray> {
        let chunk_len = chunk_tokens.len() as u32;
        if chunk_len == 0 {
            return Err(Error::from_reason(
                "run_paged_prefill_layer_loop: chunk_tokens must be non-empty",
            ));
        }
        let trace_enabled = inference_trace_enabled();
        let trace_start = trace_enabled.then(std::time::Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_layer_loop_enter first_position={} cached_prefix_for_chunk={} tokens={} layers={}",
                first_logical_position,
                cached_prefix_len_for_chunk,
                chunk_len,
                self.layers.len()
            ));
        }

        let input_ids = MxArray::from_uint32(chunk_tokens, &[1, chunk_len as i64])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;
        // Apply Gemma4 embedding scaling (sqrt(hidden_size)).
        hidden_states = hidden_states.mul_scalar((self.config.hidden_size as f64).sqrt())?;

        // Compute PLE (per-layer embeddings) for the chunk's tokens.
        // Mirrors `forward_body`: PLE feeds an additive residual inside
        // every layer's `apply_ffn_ple_scalar` tail. For Gemma4 E2B/E4B
        // this is load-bearing — dropping it produces nonsense logits
        // because each layer is missing a critical residual
        // contribution. Sliding-only re-prefill of any cached prefix
        // doesn't propagate PLE through the global layers we'll touch
        // here (their stored K/V already accounts for it).
        let projected_ple: Option<MxArray> = if let Some(ref ple) = self.ple {
            let pre_layer_h = hidden_states.clone();
            Some(compute_ple(
                &input_ids,
                &pre_layer_h,
                ple,
                chunk_len as i64,
            )?)
        } else {
            None
        };

        // Build sliding masks against the bounded rotating-cache attention view,
        // not the absolute prompt offset. This mirrors mlx-lm's
        // RotatingKVCache.make_mask behavior and avoids huge long-context masks.
        let seq_len = chunk_len as i64;
        let sliding_offset = self
            .caches
            .as_ref()
            .and_then(|caches| {
                caches
                    .iter()
                    .enumerate()
                    .find(|(i, _)| self.config.is_sliding_layer(*i))
                    .map(|(_, c)| c.get_offset())
            })
            .unwrap_or(0);
        let sliding_window = self.config.sliding_window as i64;
        let sliding_mask_offset =
            sliding_mask_offset_for_chunk(seq_len, sliding_offset, sliding_window);
        if trace_enabled && (sliding_offset > 0 || sliding_mask_offset.is_some()) {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_sliding_mask seq_len={} cache_offset={} mask_offset={} window={} explicit_mask={}",
                seq_len,
                sliding_offset,
                sliding_mask_offset.unwrap_or(0),
                sliding_window,
                sliding_mask_offset.is_some()
            ));
        }
        let sliding_mask = sliding_mask_offset
            .map(|offset| create_sliding_mask(seq_len, offset, sliding_window))
            .transpose()?;

        let has_kv_sharing = self.config.num_kv_shared_layers.is_some_and(|n| n > 0);
        let num_layers = self.layers.len();
        // Stash for sliding-anchor K/V reused by SharedOnSliding layers.
        let mut sliding_shared_kv: HashMap<u32, (MxArray, MxArray)> = HashMap::new();

        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            crate::models::gemma4::diagnostic::set_layer(layer_idx);
            let kind = layer_kinds[layer_idx];
            let layer_trace_start = trace_enabled.then(std::time::Instant::now);
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 paged_prefill_layer_start layer={} kind={:?} first_position={} cached_prefix_for_chunk={} tokens={}",
                    layer_idx, kind, first_logical_position, cached_prefix_len_for_chunk, chunk_len
                ));
            }
            let layer: &Gemma4DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            let mask: Option<&MxArray> = if matches!(kind, Gemma4LayerKind::Sliding) {
                sliding_mask.as_ref()
            } else {
                None
            };

            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason(
                    "run_paged_prefill_layer_loop: paged_adapter dropped mid-forward",
                )
            })?;
            let flat_cache: Option<&mut Gemma4LayerCache> =
                if matches!(kind, Gemma4LayerKind::Sliding) {
                    let caches = unsafe {
                        let raw = self.caches.as_mut().ok_or_else(|| {
                            Error::from_reason(
                                "run_paged_prefill_layer_loop: sliding cache slot missing",
                            )
                        })? as *mut Vec<Gemma4LayerCache>;
                        &mut *raw
                    };
                    Some(&mut caches[layer_idx])
                } else {
                    None
                };

            // Build SharedKvInputs for shared layer kinds.
            let shared_inputs = match kind {
                Gemma4LayerKind::SharedOnGlobal { .. } => {
                    // Anchor's pool currently holds
                    // `cached_prefix_len_for_chunk + chunk_len` tokens
                    // for this layer (the anchor wrote its part of
                    // this chunk earlier in the same loop).
                    let total_ctx = cached_prefix_len_for_chunk + chunk_len;
                    Some(super::decoder_layer::SharedKvInputs {
                        cache_offset: first_logical_position as i32,
                        total_ctx,
                        keys: None,
                        values: None,
                    })
                }
                Gemma4LayerKind::SharedOnSliding { anchor_layer_idx } => {
                    let (k, v) = sliding_shared_kv.get(&anchor_layer_idx).ok_or_else(|| {
                        Error::from_reason(format!(
                            "run_paged_prefill_layer_loop: SharedOnSliding anchor {} stash \
                             missing",
                            anchor_layer_idx
                        ))
                    })?;
                    let cache_offset =
                        (first_logical_position as i32 + chunk_len as i32) - seq_len as i32;
                    Some(super::decoder_layer::SharedKvInputs {
                        cache_offset,
                        total_ctx: 0, // unused for SharedOnSliding
                        keys: Some(k),
                        values: Some(v),
                    })
                }
                _ => None,
            };

            // For Sliding layers that anchor a SharedOnSliding chain,
            // request the stash so the shared layer can pull K/V.
            let needs_stash = has_kv_sharing
                && matches!(kind, Gemma4LayerKind::Sliding)
                && self.config.should_store_shared_kv(layer_idx);

            // Slice the per-layer PLE input ([B, T, num_layers, ple_dim] →
            // [B, T, ple_dim]). Mirrors `forward_body`'s per-layer slice.
            let ple_input = projected_ple.as_ref().map(|p| {
                p.slice_axis(2, layer_idx as i64, layer_idx as i64 + 1)
                    .and_then(|s| s.squeeze(Some(&[2])))
            });
            let ple_input_ref = match &ple_input {
                Some(Ok(arr)) => Some(arr),
                _ => None,
            };

            let next_hidden_states = layer.forward_paged_or_flat(
                &hidden_states,
                kind,
                adapter,
                first_logical_position,
                cached_prefix_len_for_chunk,
                /* is_prefill */ true,
                mask,
                flat_cache,
                ple_input_ref,
                needs_stash,
                shared_inputs,
            )?;
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 paged_prefill_layer_done layer={} kind={:?} elapsed_ms={:.1}",
                    layer_idx,
                    kind,
                    layer_trace_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            hidden_states = next_hidden_states;

            // After a Sliding anchor's forward, capture its stash so
            // downstream SharedOnSliding layers can attend over it.
            if needs_stash {
                let caches = unsafe {
                    let raw = self.caches.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "run_paged_prefill_layer_loop: sliding cache slot missing \
                             post-forward",
                        )
                    })? as *mut Vec<Gemma4LayerCache>;
                    &mut *raw
                };
                if let Some((k, v)) = caches[layer_idx].take_stashed_kv() {
                    sliding_shared_kv.insert(layer_idx as u32, (k, v));
                }
            }
            // Smooth the prefill memory peak: every K layers, materialize the
            // residual stream so MLX can release the upstream graph nodes
            // (embedding + every prior layer's attention/MLP/PLE intermediates)
            // from the cache pool. Without this the in-flight lazy graph
            // accumulates on long contexts before the post-prefill sync fires.
            // Cadence is `MLX_PAGED_PREFILL_EVAL_INTERVAL` (default 8).
            crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states)?;
        }

        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 paged_prefill_layer_loop_exit first_position={} tokens={} elapsed_ms={:.1}",
                first_logical_position,
                chunk_len,
                trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        Ok(hidden_states)
    }

    /// Vision variant of [`Self::run_paged_prefill_layer_loop`]: drives one
    /// contiguous chunk of the merged image+text embeddings through the hybrid
    /// paged dispatch (global → adapter, sliding → flat rotating cache,
    /// KV-shared → anchor stash).
    ///
    /// Identical layer routing to the text loop, with two image-aware seams:
    ///   * the residual stream is seeded from the supplied `chunk_embeds`
    ///     (the `masked_scatter` output for this chunk, ALREADY scaled by
    ///     `sqrt(hidden_size)` by the caller) instead of
    ///     `embed_tokens.forward(token_ids)`;
    ///   * PLE per-layer embeddings zero the image-token positions in
    ///     `chunk_token_ids` before `compute_ple`, because the image positions
    ///     carry vision features in the residual, not token PLE residuals.
    ///
    /// `chunk_token_ids` is the expanded token slice for this chunk (drives the
    /// PLE image mask and the sliding-mask sequence length).
    /// `chunk_embeds` is `[1, chunk_len, hidden]`.
    #[allow(clippy::too_many_arguments)]
    fn run_paged_vlm_prefill_layer_loop(
        &mut self,
        chunk_token_ids: &[u32],
        chunk_embeds: &MxArray,
        first_logical_position: u32,
        cached_prefix_len_for_chunk: u32,
        layer_kinds: &[Gemma4LayerKind],
        overlay_type_ids: Option<&MxArray>,
    ) -> Result<MxArray> {
        let chunk_len = chunk_token_ids.len() as u32;
        if chunk_len == 0 {
            return Err(Error::from_reason(
                "run_paged_vlm_prefill_layer_loop: chunk_token_ids must be non-empty",
            ));
        }

        let input_ids = MxArray::from_uint32(chunk_token_ids, &[1, chunk_len as i64])?;
        let mut hidden_states = chunk_embeds.clone();

        // PLE over media-masked token ids: image AND audio positions hold
        // projected media features (not token embeddings), so their PLE
        // residual must be zero.
        let projected_ple: Option<MxArray> = if let Some(ref ple) = self.ple {
            let image_token_id = self.config.image_token_id.unwrap_or(258880);
            let image_token = MxArray::scalar_int(image_token_id)?;
            let mut media_mask = input_ids.equal(&image_token)?;
            if let Some(audio_token_id) = self.config.audio_token_id {
                let audio_token = MxArray::scalar_int(audio_token_id)?;
                let audio_mask = input_ids.equal(&audio_token)?;
                media_mask = media_mask.logical_or(&audio_mask)?;
            }
            let zero = MxArray::scalar_int(0)?;
            // Media positions (image and audio) are excluded from the PLE
            // residual because their embedding is the projected media feature,
            // not a learned token.
            let masked_ids = media_mask.where_(&zero, &input_ids)?;
            let pre_layer_h = hidden_states.clone();
            Some(compute_ple(
                &masked_ids,
                &pre_layer_h,
                ple,
                chunk_len as i64,
            )?)
        } else {
            None
        };

        // Sliding mask against the bounded rotating-cache attention view —
        // identical derivation to the text paged loop.
        let seq_len = chunk_len as i64;
        let sliding_offset = self
            .caches
            .as_ref()
            .and_then(|caches| {
                caches
                    .iter()
                    .enumerate()
                    .find(|(i, _)| self.config.is_sliding_layer(*i))
                    .map(|(_, c)| c.get_offset())
            })
            .unwrap_or(0);
        let sliding_window = self.config.sliding_window as i64;
        let sliding_mask_offset =
            sliding_mask_offset_for_chunk(seq_len, sliding_offset, sliding_window);
        let mut sliding_mask = sliding_mask_offset
            .map(|offset| create_sliding_mask(seq_len, offset, sliding_window))
            .transpose()?;

        // Unified-vision bidirectional overlay. Active only on the cold-start
        // single-chunk prefill (`overlay_type_ids` is Some and
        // `cached_prefix_len_for_chunk == 0`), where every mask key dimension
        // equals `seq_len`. Both layer types get an EXPLICIT materialized
        // boolean keep-mask (true=keep): the global layer's normal None/causal
        // fast path and the sliding layer's possibly-None window mask are
        // replaced by `base | same_image_block`.
        let overlay_active = overlay_type_ids.is_some() && cached_prefix_len_for_chunk == 0;
        let overlay_global_mask: Option<MxArray> = if overlay_active {
            let type_ids = overlay_type_ids.unwrap();
            let base = create_causal_mask(seq_len as i32, None, None)?;
            let base = base.reshape(&[1, 1, seq_len, seq_len])?;
            Some(apply_bidirectional_vision_overlay(&base, type_ids)?)
        } else {
            None
        };
        if overlay_active {
            let type_ids = overlay_type_ids.unwrap();
            let base = create_causal_mask(seq_len as i32, None, Some(sliding_window as i32))?;
            let base = base.reshape(&[1, 1, seq_len, seq_len])?;
            sliding_mask = Some(apply_bidirectional_vision_overlay(&base, type_ids)?);
        }

        let has_kv_sharing = self.config.num_kv_shared_layers.is_some_and(|n| n > 0);
        let num_layers = self.layers.len();
        let mut sliding_shared_kv: HashMap<u32, (MxArray, MxArray)> = HashMap::new();

        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            crate::models::gemma4::diagnostic::set_layer(layer_idx);
            let kind = layer_kinds[layer_idx];
            let layer: &Gemma4DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            let mask: Option<&MxArray> = if matches!(kind, Gemma4LayerKind::Sliding) {
                sliding_mask.as_ref()
            } else {
                // Global/full layers normally pass None (internal causal). When
                // the overlay is active they receive the explicit bidirectional
                // keep-mask, which `forward_paged` applies in the fresh-prefill
                // branch.
                overlay_global_mask.as_ref()
            };

            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason(
                    "run_paged_vlm_prefill_layer_loop: paged_adapter dropped mid-forward",
                )
            })?;
            let flat_cache: Option<&mut Gemma4LayerCache> =
                if matches!(kind, Gemma4LayerKind::Sliding) {
                    let caches = unsafe {
                        let raw = self.caches.as_mut().ok_or_else(|| {
                            Error::from_reason(
                                "run_paged_vlm_prefill_layer_loop: sliding cache slot missing",
                            )
                        })? as *mut Vec<Gemma4LayerCache>;
                        &mut *raw
                    };
                    Some(&mut caches[layer_idx])
                } else {
                    None
                };

            let shared_inputs = match kind {
                Gemma4LayerKind::SharedOnGlobal { .. } => {
                    let total_ctx = cached_prefix_len_for_chunk + chunk_len;
                    Some(super::decoder_layer::SharedKvInputs {
                        cache_offset: first_logical_position as i32,
                        total_ctx,
                        keys: None,
                        values: None,
                    })
                }
                Gemma4LayerKind::SharedOnSliding { anchor_layer_idx } => {
                    let (k, v) = sliding_shared_kv.get(&anchor_layer_idx).ok_or_else(|| {
                        Error::from_reason(format!(
                            "run_paged_vlm_prefill_layer_loop: SharedOnSliding anchor {} stash \
                             missing",
                            anchor_layer_idx
                        ))
                    })?;
                    let cache_offset =
                        (first_logical_position as i32 + chunk_len as i32) - seq_len as i32;
                    Some(super::decoder_layer::SharedKvInputs {
                        cache_offset,
                        total_ctx: 0,
                        keys: Some(k),
                        values: Some(v),
                    })
                }
                _ => None,
            };

            let needs_stash = has_kv_sharing
                && matches!(kind, Gemma4LayerKind::Sliding)
                && self.config.should_store_shared_kv(layer_idx);

            let ple_input = projected_ple.as_ref().map(|p| {
                p.slice_axis(2, layer_idx as i64, layer_idx as i64 + 1)
                    .and_then(|s| s.squeeze(Some(&[2])))
            });
            let ple_input_ref = match &ple_input {
                Some(Ok(arr)) => Some(arr),
                _ => None,
            };

            let next_hidden_states = layer.forward_paged_or_flat(
                &hidden_states,
                kind,
                adapter,
                first_logical_position,
                cached_prefix_len_for_chunk,
                /* is_prefill */ true,
                mask,
                flat_cache,
                ple_input_ref,
                needs_stash,
                shared_inputs,
            )?;
            hidden_states = next_hidden_states;

            if needs_stash {
                let caches = unsafe {
                    let raw = self.caches.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "run_paged_vlm_prefill_layer_loop: sliding cache slot missing \
                             post-forward",
                        )
                    })? as *mut Vec<Gemma4LayerCache>;
                    &mut *raw
                };
                if let Some((k, v)) = caches[layer_idx].take_stashed_kv() {
                    sliding_shared_kv.insert(layer_idx as u32, (k, v));
                }
            }
            crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states)?;
        }

        Ok(hidden_states)
    }

    /// Cold-start paged prefill over the merged image+text embeddings.
    ///
    /// Single-shot only: the adapter holds zero tokens and the sliding flat
    /// caches were freshly built, so `cached_prefix_len == 0` and there is no
    /// prefix-cache restore. Splits the merged-embedding body prefill from a
    /// last-token `forward_inner`, a split that is load-bearing — see
    /// [`Self::run_paged_prefill_chunk`] for why
    /// the final prompt token must run through the cache-hit branch separately
    /// (BF16 SDPA drift otherwise flips argmax to a zero-embedding `<unused>`
    /// token and the `<turn|>` stop is missed).
    ///
    /// `expanded_tokens` is the full `BOI + N×image + EOI` expanded sequence.
    /// `inputs_embeds` is `[1, prompt_len, hidden]`, ALREADY scaled by
    /// `sqrt(hidden_size)` and with vision features scattered at the image
    /// positions. Returns the final token's logits squeezed to `[vocab]`.
    fn run_paged_vlm_prefill(
        &mut self,
        expanded_tokens: &[u32],
        suffix_embeds: &MxArray,
        layer_kinds: &[Gemma4LayerKind],
        cached_prefix_len: u32,
        extra_keys_per_block: &[Vec<u64>],
        image_token_positions: &[(u32, u64)],
        publish_prefix_checkpoints: bool,
    ) -> Result<MxArray> {
        if expanded_tokens.is_empty() {
            return Err(Error::from_reason(
                "run_paged_vlm_prefill called with empty prompt",
            ));
        }
        let prompt_len = expanded_tokens.len() as u32;
        if cached_prefix_len >= prompt_len {
            return Err(Error::from_reason(format!(
                "run_paged_vlm_prefill requires a non-empty suffix: cached_prefix_len={cached_prefix_len}, prompt_len={prompt_len}"
            )));
        }
        let suffix_len = prompt_len - cached_prefix_len;
        if suffix_embeds.shape_at(1)? != suffix_len as i64 {
            return Err(Error::from_reason(format!(
                "run_paged_vlm_prefill suffix embedding length {} does not match suffix token length {suffix_len}",
                suffix_embeds.shape_at(1)?
            )));
        }

        // Sliding-window state covers the whole cached prefix by construction
        // on this path: `resolve_vlm_paged_prefix` either kept a candidate
        // whose `sliding_prefix_exact` was true, or restarted the turn cold
        // (`cached_prefix_len == 0`). Discharge the adapter's auxiliary-state
        // obligation before the first `record_tokens` of the turn — the same
        // ack `run_paged_prefill_chunk` makes for the text path.
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter
                .confirm_aux_prefix_primed(cached_prefix_len)
                .map_err(Error::from_reason)?;
        }

        crate::models::gemma4::diagnostic::set_path("paged");
        crate::models::gemma4::diagnostic::set_step(-1);

        // Unified-vision bidirectional overlay gate: is_unified +
        // use_bidirectional_attention=="vision" + image tokens present + no audio
        // tokens + prefill (seq_len>1). Mixed image+audio prompts stay causal
        // (audio wins) — see `vision_overlay_active`. When active, the whole image
        // block must live in ONE prefill chunk so bidirectionality is not severed
        // by chunk boundaries.
        let image_token_id = self.config.image_token_id.unwrap_or(258880) as u32;
        let audio_token_id = self.config.audio_token_id.unwrap_or(258881) as u32;
        let has_image = expanded_tokens.contains(&image_token_id);
        let has_audio = expanded_tokens.contains(&audio_token_id);
        let overlay_full_type_ids: Option<MxArray> = if cached_prefix_len == 0
            && super::vision_mask::vision_overlay_active(
                self.config.is_unified,
                self.config.use_bidirectional_attention.as_deref() == Some("vision"),
                has_image,
                has_audio,
                prompt_len as usize,
            ) {
            Some(super::vision_mask::build_image_token_type_ids(
                expanded_tokens,
                image_token_id,
            )?)
        } else {
            None
        };
        let overlay_active = overlay_full_type_ids.is_some();
        // The overlay only reaches GlobalPaged/Sliding layers. KV-shared layers
        // (SharedOnGlobal/SharedOnSliding) run forward_paged_shared, which takes
        // no mask and would silently stay causal — a half-applied overlay across
        // the stack. The 12B unified checkpoint has num_kv_shared_layers==0, so
        // this never fires; fail loudly rather than corrupt attention if a shared
        // unified checkpoint is ever loaded.
        if overlay_active && self.config.num_kv_shared_layers.is_some_and(|n| n > 0) {
            return Err(Error::from_reason(
                "Gemma4 unified-vision bidirectional overlay is unsupported with KV-shared layers \
                 (num_kv_shared_layers > 0): forward_paged_shared does not carry the overlay mask",
            ));
        }

        let block_size = self
            .paged_adapter
            .as_ref()
            .ok_or_else(|| Error::from_reason("run_paged_vlm_prefill: paged_adapter is None"))?
            .block_size();
        let prompt_checkpoint_boundary = prompt_len
            .saturating_sub(1)
            .checked_div(block_size)
            .map(|blocks| blocks.saturating_mul(block_size))
            .unwrap_or(0);
        let first_image_position = image_token_positions.first().map(|(position, _)| *position);
        let last_image_exclusive = image_token_positions
            .last()
            .map(|(position, _)| position.saturating_add(1));
        // SigLIP/E2B is causal, so a changed image may still reuse complete
        // leading-text blocks. Unified overlay cannot split before its image.
        let leading_text_checkpoint_boundary = if overlay_active {
            0
        } else {
            first_image_position
                .and_then(|position| position.checked_div(block_size))
                .map(|blocks| blocks.saturating_mul(block_size))
                .unwrap_or(0)
        };

        // Pass 1: uncached suffix except its final token. Pass 2: the final
        // token alone, preserving the prefill/decode reduction boundary.
        let pass1_end = prompt_len - 1;
        let mut pass1_position = cached_prefix_len;
        if pass1_position < pass1_end {
            let configured_chunk_size = crate::array::paged_prefill_chunk_size();
            while pass1_position < pass1_end {
                // The first unified chunk must include the complete image
                // overlay. Boundaries before the end of that span are ignored;
                // otherwise the later chunk would receive no overlay ids and
                // silently run half of the image causally.
                let chunk_end = gemma4_vlm_prefill_chunk_end(
                    pass1_position,
                    pass1_end,
                    configured_chunk_size,
                    overlay_active,
                    leading_text_checkpoint_boundary,
                    prompt_checkpoint_boundary,
                    last_image_exclusive,
                );
                let chunk_start = pass1_position as usize;
                let chunk_end_usize = chunk_end as usize;
                let chunk_tokens = &expanded_tokens[chunk_start..chunk_end_usize];
                let relative_start = pass1_position - cached_prefix_len;
                let relative_end = chunk_end - cached_prefix_len;
                let chunk_embeds =
                    suffix_embeds.slice_axis(1, relative_start as i64, relative_end as i64)?;
                let chunk_type_ids: Option<MxArray> = match &overlay_full_type_ids {
                    Some(ids) if pass1_position == 0 => {
                        Some(ids.slice_axis(1, 0, chunk_end as i64)?)
                    }
                    _ => None,
                };
                {
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason("run_paged_vlm_prefill: paged_adapter is None")
                    })?;
                    adapter
                        .record_tokens(chunk_tokens)
                        .map_err(Error::from_reason)?;
                }
                let _hidden = self.run_paged_vlm_prefill_layer_loop(
                    chunk_tokens,
                    &chunk_embeds,
                    pass1_position,
                    pass1_position,
                    layer_kinds,
                    chunk_type_ids.as_ref(),
                )?;
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    adapter
                        .eval_pending_pool_writes()
                        .map_err(Error::from_reason)?;
                }
                if let Some(caches) = self.caches.as_ref() {
                    eval_gemma4_caches(caches)?;
                }
                if publish_prefix_checkpoints && chunk_end == prompt_checkpoint_boundary {
                    self.remember_gemma4_sliding_materialized_prompt_boundary_checkpoint_with_keys(
                        expanded_tokens,
                        chunk_end,
                        block_size,
                        extra_keys_per_block,
                        0,
                        true,
                    )?;
                } else if publish_prefix_checkpoints
                    && chunk_end == leading_text_checkpoint_boundary
                {
                    self.remember_gemma4_sliding_materialized_prefix_checkpoint_with_keys(
                        expanded_tokens,
                        chunk_end,
                        block_size,
                        extra_keys_per_block,
                        0,
                    )?;
                }
                crate::array::clear_cache();
                pass1_position = chunk_end;
            }
        }

        // Pass 2: the FINAL token (length 1).
        let last_idx = (prompt_len - 1) as usize;
        let pass2_tokens = &expanded_tokens[last_idx..];
        let pass2_relative_idx = last_idx - cached_prefix_len as usize;
        let pass2_embeds = suffix_embeds.slice_axis(
            1,
            pass2_relative_idx as i64,
            pass2_relative_idx as i64 + 1,
        )?;
        {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_vlm_prefill: paged_adapter is None")
            })?;
            adapter
                .record_tokens(pass2_tokens)
                .map_err(Error::from_reason)?;
        }
        let mut hidden_states = self.run_paged_vlm_prefill_layer_loop(
            pass2_tokens,
            &pass2_embeds,
            pass1_position,
            pass1_position,
            layer_kinds,
            // Pass 2 is the single final token (seq_len==1); the overlay never
            // applies to a single-token query.
            None,
        )?;
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter
                .eval_pending_pool_writes()
                .map_err(Error::from_reason)?;
        }

        hidden_states = self.final_norm.forward(&hidden_states)?;
        crate::models::gemma4::diagnostic::dump_norm(0, "post_final_norm", &hidden_states, None);
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&hidden_states)?
        } else if self.embed_tokens.is_packed_quantized() {
            // Packed tied lm_head: project through the quantized matmul without
            // materializing the dense table.
            self.embed_tokens.as_linear(&hidden_states)?
        } else if let Some(ref w_t) = self.embed_weight_t {
            hidden_states.matmul(w_t)?
        } else {
            let weight = self.embed_tokens.get_weight();
            let weight_t = weight.transpose(Some(&[1, 0]))?;
            hidden_states.matmul(&weight_t)?
        };
        crate::models::gemma4::diagnostic::dump_logits("pre_softcap", &logits);
        let logits = if let Some(cap) = self.config.final_logit_softcapping {
            let cap_arr = MxArray::scalar_float_like(cap, &logits)?;
            let handle = unsafe { mlx_sys::mlx_logit_softcap(logits.handle.0, cap_arr.handle.0) };
            let capped = MxArray::from_handle(handle, "logit_softcap")?;
            crate::models::gemma4::diagnostic::dump_logits("post_softcap", &capped);
            capped
        } else {
            crate::models::gemma4::diagnostic::dump_logits("post_softcap", &logits);
            logits
        };

        let last_seq_len = logits.shape_at(1)?;
        logits
            .slice_axis(1, last_seq_len - 1, last_seq_len)?
            .squeeze(Some(&[0, 1]))
    }

    /// Run one paged decode step: feed `[token_id]` through the model.
    fn run_paged_decode_step(&mut self, token_id: u32) -> Result<MxArray> {
        let first_logical_position = {
            let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
                Error::from_reason("run_paged_decode_step: paged_adapter is None")
            })?;
            adapter.current_token_count()
        };
        {
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_decode_step: paged_adapter dropped")
            })?;
            adapter
                .record_tokens(&[token_id])
                .map_err(Error::from_reason)?;
        }

        let layer_kinds = self.compute_layer_kinds()?;

        let input_ids = MxArray::from_uint32(&[token_id], &[1, 1])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;
        hidden_states = hidden_states.mul_scalar((self.config.hidden_size as f64).sqrt())?;

        // Compute PLE for the single decode token. Same load-bearing
        // residual contribution as the prefill path — see the comment in
        // `run_paged_prefill_chunk` for why dropping this destroys logits
        // on Gemma4 E2B/E4B.
        let projected_ple_step: Option<MxArray> = if let Some(ref ple) = self.ple {
            let pre_layer_h = hidden_states.clone();
            Some(compute_ple(&input_ids, &pre_layer_h, ple, 1)?)
        } else {
            None
        };

        let has_kv_sharing = self.config.num_kv_shared_layers.is_some_and(|n| n > 0);
        let num_layers = self.layers.len();
        let mut sliding_shared_kv: HashMap<u32, (MxArray, MxArray)> = HashMap::new();
        crate::models::gemma4::diagnostic::set_path("paged");
        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            crate::models::gemma4::diagnostic::set_layer(layer_idx);
            let kind = layer_kinds[layer_idx];
            let layer: &Gemma4DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("run_paged_decode_step: paged_adapter dropped mid-forward")
            })?;
            let flat_cache: Option<&mut Gemma4LayerCache> =
                if matches!(kind, Gemma4LayerKind::Sliding) {
                    let caches = unsafe {
                        let raw = self.caches.as_mut().ok_or_else(|| {
                            Error::from_reason("run_paged_decode_step: sliding cache slot missing")
                        })? as *mut Vec<Gemma4LayerCache>;
                        &mut *raw
                    };
                    Some(&mut caches[layer_idx])
                } else {
                    None
                };

            let shared_inputs = match kind {
                Gemma4LayerKind::SharedOnGlobal { .. } => {
                    // Anchor's slot already has the new token (it ran
                    // its own forward_paged earlier in this loop, which
                    // wrote K/V via update_keys_values). Read full ctx.
                    let total_ctx = first_logical_position + 1;
                    Some(super::decoder_layer::SharedKvInputs {
                        cache_offset: first_logical_position as i32,
                        total_ctx,
                        keys: None,
                        values: None,
                    })
                }
                Gemma4LayerKind::SharedOnSliding { anchor_layer_idx } => {
                    let (k, v) = sliding_shared_kv.get(&anchor_layer_idx).ok_or_else(|| {
                        Error::from_reason(format!(
                            "run_paged_decode_step: SharedOnSliding anchor {} stash missing",
                            anchor_layer_idx
                        ))
                    })?;
                    let cache_offset = first_logical_position as i32;
                    Some(super::decoder_layer::SharedKvInputs {
                        cache_offset,
                        total_ctx: 0,
                        keys: Some(k),
                        values: Some(v),
                    })
                }
                _ => None,
            };

            let needs_stash = has_kv_sharing
                && matches!(kind, Gemma4LayerKind::Sliding)
                && self.config.should_store_shared_kv(layer_idx);

            // Slice the per-layer PLE input ([B, T, num_layers, ple_dim] →
            // [B, T, ple_dim]).
            let ple_input = projected_ple_step.as_ref().map(|p| {
                p.slice_axis(2, layer_idx as i64, layer_idx as i64 + 1)
                    .and_then(|s| s.squeeze(Some(&[2])))
            });
            let ple_input_ref = match &ple_input {
                Some(Ok(arr)) => Some(arr),
                _ => None,
            };

            hidden_states = layer.forward_paged_or_flat(
                &hidden_states,
                kind,
                adapter,
                first_logical_position,
                /* cached_prefix_len */ 0,
                /* is_prefill */ false,
                /* mask */ None,
                flat_cache,
                ple_input_ref,
                needs_stash,
                shared_inputs,
            )?;

            if needs_stash {
                let caches = unsafe {
                    let raw = self.caches.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "run_paged_decode_step: sliding cache slot missing post-forward",
                        )
                    })? as *mut Vec<Gemma4LayerCache>;
                    &mut *raw
                };
                if let Some((k, v)) = caches[layer_idx].take_stashed_kv() {
                    sliding_shared_kv.insert(layer_idx as u32, (k, v));
                }
            }
        }

        hidden_states = self.final_norm.forward(&hidden_states)?;
        crate::models::gemma4::diagnostic::dump_norm(0, "post_final_norm", &hidden_states, None);
        let logits = if let Some(ref head) = self.lm_head {
            head.forward(&hidden_states)?
        } else if self.embed_tokens.is_packed_quantized() {
            // Packed tied lm_head: project through the quantized matmul without
            // materializing the dense table.
            self.embed_tokens.as_linear(&hidden_states)?
        } else if let Some(ref w_t) = self.embed_weight_t {
            hidden_states.matmul(w_t)?
        } else {
            let weight = self.embed_tokens.get_weight();
            let weight_t = weight.transpose(Some(&[1, 0]))?;
            hidden_states.matmul(&weight_t)?
        };
        crate::models::gemma4::diagnostic::dump_logits("pre_softcap", &logits);
        let logits = if let Some(cap) = self.config.final_logit_softcapping {
            let cap_arr = MxArray::scalar_float_like(cap, &logits)?;
            let handle = unsafe { mlx_sys::mlx_logit_softcap(logits.handle.0, cap_arr.handle.0) };
            let capped = MxArray::from_handle(handle, "logit_softcap")?;
            crate::models::gemma4::diagnostic::dump_logits("post_softcap", &capped);
            capped
        } else {
            crate::models::gemma4::diagnostic::dump_logits("post_softcap", &logits);
            logits
        };
        Ok(logits)
    }

    /// Replay cached prefix tokens to reconstruct the flat sliding caches.
    /// Used to bring sliding-layer state up to the paged cache's
    /// `cached_prefix_len` boundary before the main `run_paged_prefill_chunk`
    /// continues with the suffix.
    ///
    /// Global layers run as read-only Q projections against their existing
    /// paged K/V. That keeps hidden states flowing into later sliding layers
    /// without rebuilding throwaway global K/V for the cached prefix.
    ///
    /// This body publishes ONE checkpoint (at `cached_prefix_len`, by its
    /// caller) and deliberately no [`gemma4_sliding_cold_anchor_rungs`], unlike
    /// `run_paged_prefill_chunk`'s pass-1 loop. Every rung it would cross is
    /// already in the store, because reaching this replay at all means the
    /// prefix below `first_logical_position` was already reconstructed from a
    /// source that published them:
    ///
    /// ```text
    ///   cold_sidecar arm   -> primed == cached, this body does not run
    ///   prefix_checkpoint  -> the store holds that entry AND every shallower
    ///                         rung of the same lineage (Ladder defers them)
    ///   replay arm         -> primed == 0. The pass-1 loop crosses the whole
    ///                         grid itself only when cached_prefix_len == 0 too;
    ///                         see below for the case where it is not.
    /// ```
    ///
    /// On the replay arm with a NON-empty cached prefix the pass-1 loop starts
    /// at `cached_prefix_len`, and `gemma4_sliding_chunk_checkpoint_boundaries`
    /// filters `rung > start_offset`, so the rungs below that offset are not
    /// republished by this turn. That is bounded rather than open-ended:
    /// [`Self::suppress_large_sliding_prefix_reuse_if_needed`] forces
    /// `cached_prefix_len := 0` whenever the restore exceeds
    /// `gemma4_default_sliding_restore_limit` (1024 on the 12B), which sends the
    /// turn down the fresh arm and republishes the whole ladder from 0. The
    /// residual is a cached prefix at or below that limit, where the shallow
    /// rungs the cold tier needs are `<= 1024` anyway and the next turn that
    /// misses restores them. Do not "fix" this by publishing rungs here: this
    /// body re-forwards a span the caller already accounted for, and adding
    /// checkpoints changes the retained set, which changes emitted tokens.
    ///
    /// Adding a ladder here would therefore be code no scenario needs. If
    /// one is ever found, the seam is the same as pass-1's: this loop drives
    /// `update_and_fetch`, so `prepare_gemma4_sliding_checkpoint_captures` /
    /// `take_gemma4_sliding_checkpoint_captures` work unchanged.
    fn run_sliding_only_prefill(
        &mut self,
        prefix_tokens: &[u32],
        first_logical_position: u32,
        layer_kinds: &[Gemma4LayerKind],
    ) -> Result<()> {
        if prefix_tokens.is_empty() {
            return Ok(());
        }
        let configured_chunk_size = crate::array::paged_prefill_chunk_size();
        let num_query_heads = u32::try_from(self.config.num_attention_heads).map_err(|_| {
            Error::from_reason(format!(
                "Gemma4 sliding restore invalid num_attention_heads={}",
                self.config.num_attention_heads
            ))
        })?;
        let global_head_size =
            u32::try_from(self.config.effective_head_dim(true)).map_err(|_| {
                Error::from_reason(format!(
                    "Gemma4 sliding restore invalid global head_dim={}",
                    self.config.effective_head_dim(true)
                ))
            })?;
        let num_kv_heads = u32::try_from(self.config.effective_kv_heads(true)).map_err(|_| {
            Error::from_reason(format!(
                "Gemma4 sliding restore invalid global num_kv_heads={}",
                self.config.effective_kv_heads(true)
            ))
        })?;
        let route_policy = gemma4_paged_prefill_route_policy();
        let mut chunk_plan = gemma4_paged_prefill_body_chunk_plan(
            configured_chunk_size,
            prefix_tokens.len(),
            first_logical_position,
            num_query_heads,
            num_kv_heads,
            global_head_size,
            route_policy,
        )?;
        gemma4_coalesce_single_token_restore_chunks(&mut chunk_plan);

        let trace_enabled = inference_trace_enabled();
        let total_trace_start = trace_enabled.then(std::time::Instant::now);
        if trace_enabled {
            let first_chunk_size = chunk_plan.first().map(|chunk| chunk.len).unwrap_or(0);
            let min_chunk_size = chunk_plan.iter().map(|chunk| chunk.len).min().unwrap_or(0);
            let max_chunk_size = chunk_plan.iter().map(|chunk| chunk.len).max().unwrap_or(0);
            let aux_caps = chunk_plan
                .iter()
                .filter(|chunk| chunk.capped_by_v2_aux_limit)
                .count();
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 sliding_prefix_restore_start first_position={} prefix_tokens={} chunks={} chunk_size={} min_chunk_size={} max_chunk_size={} configured_chunk_size={} dynamic_v2_aux_caps={} path=paged_global_readonly",
                first_logical_position,
                prefix_tokens.len(),
                chunk_plan.len(),
                first_chunk_size,
                min_chunk_size,
                max_chunk_size,
                configured_chunk_size,
                aux_caps
            ));
        }

        let total_chunks = chunk_plan.len();
        for (chunk_idx, chunk_plan) in chunk_plan.iter().enumerate() {
            let chunk_end = chunk_plan
                .start
                .checked_add(chunk_plan.len)
                .ok_or_else(|| Error::from_reason("Gemma4 sliding restore chunk end overflow"))?;
            let chunk = prefix_tokens
                .get(chunk_plan.start..chunk_end)
                .ok_or_else(|| {
                    Error::from_reason("Gemma4 sliding restore chunk plan out of range")
                })?;
            let chunk_trace_start = trace_enabled.then(std::time::Instant::now);
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_restore_chunk_start chunk={}/{} first_position={} tokens={} capped_by_v2_aux_limit={}",
                    chunk_idx + 1,
                    total_chunks,
                    chunk_plan.first_position,
                    chunk.len(),
                    chunk_plan.capped_by_v2_aux_limit
                ));
            }

            self.run_sliding_prefix_restore_layer_loop(
                chunk,
                chunk_plan.first_position,
                layer_kinds,
            )?;

            if let Some(caches) = self.caches.as_ref() {
                eval_gemma4_caches(caches)?;
            }
            crate::array::clear_cache();

            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_restore_chunk_done chunk={}/{} next_position={} elapsed_ms={:.1}",
                    chunk_idx + 1,
                    total_chunks,
                    chunk_plan.first_position + chunk.len() as u32,
                    chunk_trace_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
        }

        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 sliding_prefix_restore_done first_position={} prefix_tokens={} chunks={} elapsed_ms={:.1}",
                first_logical_position,
                prefix_tokens.len(),
                total_chunks,
                total_trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        Ok(())
    }

    fn run_sliding_prefix_restore_layer_loop(
        &mut self,
        chunk_tokens: &[u32],
        first_logical_position: u32,
        layer_kinds: &[Gemma4LayerKind],
    ) -> Result<()> {
        let chunk_len = chunk_tokens.len() as u32;
        if chunk_len == 0 {
            return Ok(());
        }

        let input_ids = MxArray::from_uint32(chunk_tokens, &[1, chunk_len as i64])?;
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;
        hidden_states = hidden_states.mul_scalar((self.config.hidden_size as f64).sqrt())?;

        let projected_ple_prefix: Option<MxArray> = if let Some(ref ple) = self.ple {
            let pre_layer_h = hidden_states.clone();
            Some(compute_ple(
                &input_ids,
                &pre_layer_h,
                ple,
                chunk_len as i64,
            )?)
        } else {
            None
        };

        let sliding_offset = self
            .caches
            .as_ref()
            .and_then(|caches| {
                caches
                    .iter()
                    .enumerate()
                    .find(|(i, _)| self.config.is_sliding_layer(*i))
                    .map(|(_, c)| c.get_offset())
            })
            .unwrap_or(0);
        if sliding_offset != first_logical_position as i32 {
            return Err(Error::from_reason(format!(
                "Gemma4 sliding restore cache offset mismatch: expected {} got {}",
                first_logical_position, sliding_offset
            )));
        }

        let seq_len = chunk_len as i64;
        let sliding_window = self.config.sliding_window as i64;
        let sliding_mask_offset =
            sliding_mask_offset_for_chunk(seq_len, sliding_offset, sliding_window);
        let sliding_mask = sliding_mask_offset
            .map(|offset| create_sliding_mask(seq_len, offset, sliding_window))
            .transpose()?;

        let has_kv_sharing = self.config.num_kv_shared_layers.is_some_and(|n| n > 0);
        let total_ctx = first_logical_position
            .checked_add(chunk_len)
            .ok_or_else(|| Error::from_reason("Gemma4 sliding restore total_ctx overflow"))?;
        let mut sliding_shared_kv: HashMap<u32, (MxArray, MxArray)> = HashMap::new();

        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..self.layers.len() {
            crate::models::gemma4::diagnostic::set_layer(layer_idx);
            let kind = layer_kinds[layer_idx];
            let layer: &Gemma4DecoderLayer = unsafe {
                let ptr = self.layers.as_ptr().add(layer_idx);
                &*ptr
            };
            let ple_input = projected_ple_prefix.as_ref().map(|p| {
                p.slice_axis(2, layer_idx as i64, layer_idx as i64 + 1)
                    .and_then(|s| s.squeeze(Some(&[2])))
            });
            let ple_input_ref = match &ple_input {
                Some(Ok(arr)) => Some(arr),
                _ => None,
            };

            let needs_stash = has_kv_sharing
                && matches!(kind, Gemma4LayerKind::Sliding)
                && self.config.should_store_shared_kv(layer_idx);

            match kind {
                Gemma4LayerKind::Sliding => {
                    let caches = unsafe {
                        let raw = self.caches.as_mut().ok_or_else(|| {
                            Error::from_reason(
                                "run_sliding_prefix_restore_layer_loop: sliding cache missing",
                            )
                        })? as *mut Vec<Gemma4LayerCache>;
                        &mut *raw
                    };
                    hidden_states = layer.forward(
                        &hidden_states,
                        sliding_mask.as_ref(),
                        Some(&mut caches[layer_idx]),
                        ple_input_ref,
                        needs_stash,
                    )?;
                    if needs_stash && let Some((k, v)) = caches[layer_idx].take_stashed_kv() {
                        sliding_shared_kv.insert(layer_idx as u32, (k, v));
                    }
                }
                Gemma4LayerKind::GlobalPaged { paged_idx } => {
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "run_sliding_prefix_restore_layer_loop: paged_adapter missing",
                        )
                    })?;
                    hidden_states = layer.forward_paged_or_flat(
                        &hidden_states,
                        Gemma4LayerKind::SharedOnGlobal {
                            anchor_paged_idx: paged_idx,
                        },
                        adapter,
                        first_logical_position,
                        first_logical_position,
                        /* is_prefill */ true,
                        /* mask */ None,
                        /* flat_cache */ None,
                        ple_input_ref,
                        /* needs_stash */ false,
                        Some(super::decoder_layer::SharedKvInputs {
                            cache_offset: first_logical_position as i32,
                            total_ctx,
                            keys: None,
                            values: None,
                        }),
                    )?;
                }
                Gemma4LayerKind::SharedOnGlobal { .. } => {
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "run_sliding_prefix_restore_layer_loop: paged_adapter missing",
                        )
                    })?;
                    hidden_states = layer.forward_paged_or_flat(
                        &hidden_states,
                        kind,
                        adapter,
                        first_logical_position,
                        first_logical_position,
                        /* is_prefill */ true,
                        /* mask */ None,
                        /* flat_cache */ None,
                        ple_input_ref,
                        /* needs_stash */ false,
                        Some(super::decoder_layer::SharedKvInputs {
                            cache_offset: first_logical_position as i32,
                            total_ctx,
                            keys: None,
                            values: None,
                        }),
                    )?;
                }
                Gemma4LayerKind::SharedOnSliding { anchor_layer_idx } => {
                    let (k, v) = sliding_shared_kv.get(&anchor_layer_idx).ok_or_else(|| {
                        Error::from_reason(format!(
                            "run_sliding_prefix_restore_layer_loop: SharedOnSliding anchor {} stash missing",
                            anchor_layer_idx
                        ))
                    })?;
                    hidden_states = layer.forward_paged_or_flat(
                        &hidden_states,
                        kind,
                        self.paged_adapter.as_mut().ok_or_else(|| {
                            Error::from_reason(
                                "run_sliding_prefix_restore_layer_loop: paged_adapter missing",
                            )
                        })?,
                        first_logical_position,
                        first_logical_position,
                        /* is_prefill */ true,
                        /* mask */ None,
                        /* flat_cache */ None,
                        ple_input_ref,
                        /* needs_stash */ false,
                        Some(super::decoder_layer::SharedKvInputs {
                            cache_offset: first_logical_position as i32,
                            total_ctx: 0,
                            keys: Some(k),
                            values: Some(v),
                        }),
                    )?;
                }
            }

            crate::array::maybe_eval_clear_for_paged_prefill_layer(layer_idx, &hidden_states)?;
        }

        Ok(())
    }

    // =================================================================
    // Session-specific stop-token resolution. Conversation structure is
    // rendered exclusively by the checkpoint-provided chat template in the
    // shared engine; this family supplies only its terminal token policy.
    // =================================================================

    /// Resolve the token id for Gemma4's `<turn|>` turn terminator.
    ///
    /// Used as the `eos_token_id` in the session-start path so the
    /// decode loop stops at the model's turn boundary. Computed on demand
    /// rather than cached — encoding a special token is O(1) and the cost is
    /// trivial relative to a chat turn.
    pub(crate) fn turn_end_id(&self) -> Result<u32> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))?;
        let ids = tokenizer.encode_sync("<turn|>", Some(false))?;
        if ids.is_empty() {
            return Err(Error::from_reason(
                "Tokenizer encoded <turn|> to empty id vector",
            ));
        }
        if ids.len() != 1 {
            return Err(Error::from_reason(format!(
                "Tokenizer encoded <turn|> to {} tokens; expected 1",
                ids.len()
            )));
        }
        Ok(ids[0])
    }

    /// Multimodal whole-turn dispatch for the engine's
    /// [`ChatBackend::run_multimodal_turn`] handler. A complete structured
    /// continuation may include historical media, so the paged cores verify
    /// the rendered token prefix and media identity before deciding whether
    /// to reuse live state or cold-prefill.
    ///
    /// Image turns run ONLY on the block-paged KV backend. A model with
    /// no paged adapter (explicit `use_block_paged_cache: false`, a
    /// non-Metal build, or paged init failure) has no vision path and
    /// returns an error instead of silently falling back.
    fn multimodal_chat_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        if self.paged_adapter.is_none() {
            return Err(Error::from_reason(
                "gemma4 image turns require the block-paged KV backend; the model was loaded \
                 without a paged adapter (use_block_paged_cache=false, non-Metal build, or paged \
                 init failed)",
            ));
        }
        let tokenizer = args.tokenizer.clone();
        match (args.sink, args.cancelled) {
            (Some(sink), Some(cancelled)) => {
                self.vision_paged_turn_stream_core(
                    args.tokens,
                    args.media.images,
                    args.media.audio,
                    &tokenizer,
                    args.config,
                    args.eos_id,
                    sink,
                    cancelled,
                )?;
                Ok(TurnOutput::Streamed)
            }
            _ => {
                let result = self.vision_paged_turn_sync_core(
                    args.tokens,
                    args.media.images,
                    args.media.audio,
                    &tokenizer,
                    args.config,
                    args.eos_id,
                )?;
                Ok(TurnOutput::Complete(Box::new(result)))
            }
        }
    }
}

/// Eager flat decode stepper for one gemma4 turn
/// ([`ChatBackend::begin_decode`]). Runs the flat decode-loop step body:
/// `diagnostic::set_step(step)` before every forward (the
/// `MLX_DEBUG_GEMMA4_DUMP` per-step dump), `forward_inner` over the live
/// session caches, async-eval of the sampled token only (gemma4 never
/// async-evals the logits).
pub(crate) struct Gemma4Decode<'a> {
    inner: &'a mut Gemma4Inner,
    /// Diagnostic step counter. The engine loop has no step index in the
    /// `DecodeStep` seam, so the stepper carries its own 0-based sequence
    /// to feed `set_step`.
    step: i32,
}

impl DecodeStep for Gemma4Decode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        let inner = &mut *self.inner;
        let caches = inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 decode: caches missing"))?;
        crate::models::gemma4::diagnostic::set_step(self.step);
        self.step += 1;
        let logits = forward_inner(
            input_ids,
            &inner.embed_tokens,
            &inner.layers,
            caches,
            &inner.final_norm,
            &inner.lm_head,
            inner.embed_weight_t.as_ref(),
            inner.ple.as_ref(),
            &inner.config,
            None,
        )?;
        // `true` requests the engine's `squeeze(Some(&[1]))`: the eager
        // forward returns `[1, 1, vocab]`.
        Ok((logits, true))
    }

    fn eval_step(&mut self, next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {
        MxArray::async_eval_arrays(&[next_token]);
    }

    fn materialize_final(&mut self, token_id: u32) -> Result<()> {
        // LENGTH-exit only (the engine gates the call): run ONE more
        // `forward_inner` for the final committed token so its K/V lands in
        // the live session caches, then DISCARD the logits. This makes the
        // per-layer cache offsets equal the keep-all-on-length saved
        // history. No sample / push / emit. Like the paged override, this
        // deliberately does NOT fire a sliding decode-boundary checkpoint.
        let inner = &mut *self.inner;
        let caches = inner
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 materialize_final: caches missing"))?;
        let input_ids = MxArray::from_int32(&[token_id as i32], &[1, 1])?;
        crate::models::gemma4::diagnostic::set_step(self.step);
        self.step += 1;
        let _logits = forward_inner(
            &input_ids,
            &inner.embed_tokens,
            &inner.layers,
            caches,
            &inner.final_norm,
            &inner.lm_head,
            inner.embed_weight_t.as_ref(),
            inner.ple.as_ref(),
            &inner.config,
            None,
        )?;
        Ok(())
    }
}

/// Paged decode stepper for gemma4 (pure-eager — no compiled path, so no
/// lifecycle/reset guard fields). Drives
/// [`crate::engine::decode::run_decode_loop`] through
/// [`Gemma4Inner::run_paged_decode_step`], advancing the per-instance
/// sliding-window KV checkpoint machinery as a side effect of each
/// committed decode step.
pub(crate) struct Gemma4PagedDecode<'a> {
    /// Diagnostic step counter, fed to `set_step` before every paged
    /// forward. The engine loop has no step index in the `DecodeStep`
    /// seam, so the stepper carries its own.
    step: i32,
    inner: &'a mut Gemma4Inner,
}

impl DecodeStep for Gemma4PagedDecode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        // The loop hands the already-extracted token via
        // `forward_with_token`; recover it here from the `[1, 1]` input for
        // the bare `forward` contract (idempotent eval with the loop-top
        // `y.eval()`).
        let token_id = input_ids.item_at_int32(0)? as u32;
        self.forward_with_token(input_ids, token_id)
    }

    fn forward_with_token(
        &mut self,
        _input_ids: &MxArray,
        token_id: u32,
    ) -> Result<(MxArray, bool)> {
        crate::models::gemma4::diagnostic::set_step(self.step);
        self.step += 1;
        let trace_enabled = inference_trace_enabled();
        // `run_paged_decode_step` records the token in the adapter at its
        // top (BEFORE the forward), then returns `[1, 1, vocab]`.
        let logits = self.inner.run_paged_decode_step(token_id)?;
        // The sliding-window decode-boundary checkpoint runs RIGHT AFTER
        // the forward, reading the adapter's post-record cursor. It must
        // NOT move to `maintain_cache` (which runs at the loop TOP, before
        // this forward, so it would read a stale cursor) — see the engine
        // loop ordering. Fallible: a checkpoint/eval error aborts the turn.
        self.inner
            .maybe_remember_gemma4_sliding_decode_boundary_checkpoint("paged", trace_enabled)?;
        // `run_paged_decode_step` returns `[1, 1, vocab]`; `true` requests
        // the engine's squeeze of axis 1 (the eager convention).
        Ok((logits, true))
    }

    fn eval_step(&mut self, next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {
        // Async-eval the sampled token only (gemma4 never async-evals the
        // logits); the loop-top `y.eval()` forces materialization next
        // iteration.
        MxArray::async_eval_arrays(&[next_token]);
    }

    fn maintain_cache(&mut self, step: i32) {
        // Paged cadence — the per-step
        // `maybe_clear_cache_for_paged_step(step)`.
        crate::array::maybe_clear_cache_for_paged_step(step);
    }

    fn materialize_final(&mut self, token_id: u32) -> Result<()> {
        // LENGTH-exit only (the engine gates the call): run ONE more
        // `run_paged_decode_step` for the final committed token so its K/V
        // lands in the paged adapter, then DISCARD the logits. The adapter's
        // `request_tokens()` / cursor advances by exactly 1 to equal the
        // saved keep-all history.
        //
        // Deliberately does NOT fire the sliding decode-boundary checkpoint:
        // the final length-exit token is not checkpointed at the boundary.
        // The history checkpoint (in `save_paged_history`) covers the kept
        // history instead.
        let _logits = self.inner.run_paged_decode_step(token_id)?;
        Ok(())
    }
    // end_decode → default Ok(()).
}

/// gemma4 paged prefix state — the effective prefix/suffix split from
/// `prepare_gemma4_paged_turn`. `effective_cached_prefix_len` is the
/// POST-suppression length (the prepare may zero the plan's cached_len
/// when a large sliding-prefix reuse is suppressed). `full_tokens`
/// carries the entire prompt: the engine hands `paged_prefill` only the
/// suffix, but `run_paged_prefill_chunk` re-prefills the sliding layers
/// from the prompt start, and `sliding_primed_prefix_len` tells it how
/// much of the cached prefix the sliding caches already hold.
pub(crate) struct Gemma4PrefixState {
    effective_cached_prefix_len: usize,
    suffix_len: usize,
    sliding_primed_prefix_len: u32,
    full_tokens: Vec<u32>,
}

impl PagedPrefix for Gemma4PrefixState {
    fn effective_cached_prefix_len(&self) -> usize {
        self.effective_cached_prefix_len
    }
    fn suffix_len(&self) -> usize {
        self.suffix_len
    }
}

impl PagedBackend for Gemma4Inner {
    type PagedDecode<'a>
        = Gemma4PagedDecode<'a>
    where
        Self: 'a;
    type PrefixState = Gemma4PrefixState;

    fn prime_prefix_state(
        &mut self,
        plan: &[u32],
        reuse_cache: bool,
        _block_size: usize,
        _extra_keys: &[u64],
        _cache_salt: u64,
    ) -> Result<Self::PrefixState> {
        let trace_enabled = inference_trace_enabled();
        let total_budget = plan.len() as u32;
        // The one writer of this field. `finalize_paged_turn` runs the cold
        // sidecar capture over `request_tokens` = prompt + generated, and the
        // boundary it may anchor at is bounded by the PROMPT
        // (`gemma4_cold_restore_reachable_boundary`), so the prompt length has
        // to survive the decode. `engine::paged_turn::run_paged_turn` calls this
        // exactly once per turn and always before that finalize.
        self.paged_turn_prompt_len = total_budget;
        // Per-turn seq_id: the adapter is single-request and the prepare's
        // warm-continue / cold-reset arms make the previous seq_id
        // irrelevant.
        let seq_id: u32 = 0;
        // The prepare runs the adapter's warm-continue / cold-reset arms,
        // applies the vLLM `max_cache_hit_tokens = total_budget - 1` cap,
        // and may ZERO the cached prefix mid-prepare when a large
        // sliding-prefix reuse is suppressed — so the EFFECTIVE
        // post-suppression length surfaces here (never the plan's raw
        // cached_len).
        let prep = self.prepare_gemma4_paged_turn(
            "paged",
            plan,
            reuse_cache,
            total_budget,
            seq_id,
            trace_enabled,
        )?;
        Ok(Gemma4PrefixState {
            effective_cached_prefix_len: prep.cached_prefix_len as usize,
            suffix_len: prep.suffix_len as usize,
            sliding_primed_prefix_len: prep.sliding_primed_prefix_len,
            // Sliding-layer re-prefill needs the FULL prompt, not just the
            // suffix the engine passes to `paged_prefill`.
            full_tokens: plan.to_vec(),
        })
    }

    fn paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        _stream: Stream,
    ) -> Result<MxArray> {
        // Mark the diagnostic step as -1 (prefill) before the forward
        // (diagnostic-only). The engine fires the post-prefill
        // `synchronize_and_clear_cache` AFTER this returns.
        crate::models::gemma4::diagnostic::set_step(-1);
        self.run_paged_prefill_chunk(
            &prefix.full_tokens,
            suffix_tokens,
            prefix.effective_cached_prefix_len as u32,
            prefix.sliding_primed_prefix_len,
        )
    }

    fn begin_paged_decode(&mut self, _setup: &PagedTurnSetup<'_>) -> Result<Self::PagedDecode<'_>> {
        Ok(Gemma4PagedDecode {
            step: 0,
            inner: self,
        })
    }

    fn finalize_paged_turn(&mut self, reuse_cache: bool) {
        // Terminal lifecycle for the paged turn. Success: keep the request
        // live across turns when reuse is on so the next turn builds on the
        // partial trailing block's live K/V; otherwise register full blocks
        // for reuse + release. Infallible (`let _ =` every call — a teardown
        // failure must not mask the turn result).
        self.paged_finalize_failed = false;
        // The non-reuse branch defers its `release_request` past the sidecar
        // capture below: releasing clears `request_tokens` and the cold-chain
        // capture depth, and the sidecar is keyed off exactly those.
        let mut release_pending = false;
        let mut finalize_error = match self.paged_adapter.as_mut() {
            Some(adapter) => {
                let total_tokens = adapter.request_tokens().len();
                let block_size = adapter.block_size();
                let extra_keys = engine::build_paged_extra_keys(
                    total_tokens,
                    block_size,
                    &self.cached_paged_image_token_positions,
                );
                if reuse_cache {
                    adapter
                        .finalize_turn_keep_live_per_block(&extra_keys, 0)
                        .err()
                } else {
                    release_pending = true;
                    adapter
                        .register_full_blocks_for_reuse_per_block(&extra_keys, 0)
                        .err()
                }
            }
            None => Some("Gemma4 paged adapter missing during finalize".to_string()),
        };
        // Persist the out-of-pool sliding state for the SAME chain the adapter
        // just captured, so a later process can resume from the restored K/V
        // prefix instead of replaying the decoder over it. Skipped when the
        // finalize failed: the K/V chain the sidecar would anchor on was not
        // published, so nothing could ever select it.
        if finalize_error.is_none() {
            self.capture_gemma4_sliding_cold_sidecar(Gemma4SlidingColdCaptureContext::text(
                self.paged_turn_prompt_len,
                &self.cached_paged_image_token_positions,
            ));
        }
        if release_pending && let Some(adapter) = self.paged_adapter.as_mut() {
            finalize_error = finalize_error.or(adapter.release_request().err());
        }
        if let Some(error) = finalize_error {
            tracing::warn!(
                target: "mlx_core::gemma4::paged",
                "Gemma4 paged finalize failed: {error}"
            );
            self.paged_finalize_failed = true;
            if let Some(adapter) = self.paged_adapter.as_mut() {
                let _ = adapter.release_request();
            }
            self.media_session_continuable = false;
        }
    }

    fn abort_paged_turn(&mut self) {
        // Error-path teardown: release the request fully — partial
        // block_table state is unsafe to keep around. Infallible (`let _ =`).
        if let Some(adapter) = self.paged_adapter.as_mut() {
            let _ = adapter.release_request();
        }
        self.caches = None;
        self.clear_reuse_state();
    }

    fn save_paged_history(
        &mut self,
        save_tokens: &[u32],
        generated: &[u32],
        keep_all: bool,
        reuse_cache: bool,
    ) -> Result<()> {
        if self.paged_finalize_failed {
            self.cached_token_history.clear();
            self.sliding_last_history_checkpoint = None;
            self.media_session_continuable = false;
            return Err(Error::from_reason(
                "Gemma4 paged finalize failed; refusing to publish reusable history",
            ));
        }
        // `run_paged_turn` snapshots the request planner's context here for
        // the duration of the executor. Empty means this text turn is a fresh
        // replacement; non-empty means it extended a media-derived session.
        let continued_media_context = self.paged_text_turn_context;
        // Save token history ONLY — the adapter's pool owns the K/V.
        // `keep_all` is the flat rule (engine: `finish_reason ==
        // "length"`); when it is false the terminal stop token is dropped
        // (DROP-LAST trim). The engine reconciles `request_tokens()` to this
        // same trimmed history via `reconcile_paged_request_tokens` BEFORE
        // finalize, so the adapter and the saved history stay aligned for
        // the next turn's warm-continue.
        if reuse_cache {
            let mut full_history = save_tokens.to_vec();
            let history_tokens = if keep_all || generated.is_empty() {
                generated
            } else {
                &generated[..generated.len() - 1]
            };
            full_history.extend_from_slice(history_tokens);
            self.cached_token_history = full_history;
            if continued_media_context.is_empty() {
                // A successful fresh text turn replaced any previous media
                // session. Its saved/live KV is now genuinely text-only.
                self.cached_image_key = None;
                self.cached_audio_key = None;
                self.cached_paged_image_token_positions.clear();
                self.media_session_context = MediaCapabilities::NONE;
                self.media_session_continuable = false;
            } else {
                // A warm text delta extended the same live media prefix.
                // Preserve the exact image key and ordered placeholder sidecar:
                // subsequent text blocks must keep registering under the same
                // image-aware lineage, and live continuation needs raw identity.
                self.cached_audio_key = None;
                debug_assert!(self.media_session_continuable);
                self.media_session_context = continued_media_context;
            }
            // Sliding-window warm-continue checkpoint keyed on the freshly
            // set history (post-reconcile `request_tokens()` == the trimmed
            // history). Fallible: a checkpoint/eval error aborts the turn so
            // reusable state is never published without a materialized
            // checkpoint.
            let history_for_checkpoint = self.cached_token_history.clone();
            let _store_trace =
                self.remember_gemma4_sliding_history_checkpoint(&history_for_checkpoint)?;
        } else {
            self.cached_token_history.clear();
            self.sliding_last_history_checkpoint = None;
            // Fresh paged start: a text turn holds no media, so clear any media
            // key a prior turn on this reused model left set (mirrors the flat
            // `save_cache_state` fresh-turn clear). Without the audio clear a
            // text-only start over a model whose last turn was audio would leave
            // `cached_audio_key` stale and the delta image guard would wrongly
            // force an "audio state" restart on the text-only session.
            self.cached_image_key = None;
            self.cached_audio_key = None;
            self.cached_paged_image_token_positions.clear();
            self.media_session_context = MediaCapabilities::NONE;
            self.media_session_continuable = false;
        }
        Ok(())
    }

    fn reconcile_paged_request_tokens(
        &mut self,
        prompt_len: usize,
        generated: &[u32],
        keep_all: bool,
    ) -> bool {
        // Perf-parity warm-continue restore (see the trait doc). The
        // pipelined decode loop records the stop token into the adapter
        // (its forward ran at the loop top BEFORE the stop-check), but the
        // saved history DROPS it on a non-length exit. Roll the adapter back
        // to the to-be-saved history length so `request_tokens()` matches
        // the persisted history. `history_len` uses the EXACT same trim as
        // `save_paged_history`; `saturating_sub` makes it a no-op on a length
        // exit (`materialize_final` already recorded the final token) and on
        // a final-step stop (forward never ran).
        let Some(adapter) = self.paged_adapter.as_mut() else {
            return true;
        };
        let history_len = if keep_all || generated.is_empty() {
            generated.len()
        } else {
            generated.len() - 1
        };
        let target_len = prompt_len + history_len;
        let surplus = adapter.request_tokens().len().saturating_sub(target_len);
        if surplus > 0
            && let Err(e) = adapter.rollback_last_tokens(surplus as u32)
        {
            tracing::warn!(
                target: "mlx_core::gemma4::paged",
                "reconcile_paged_request_tokens: rollback_last_tokens({surplus}) failed \
                 (finalize releases the request; next turn cold-prefills): {e}",
            );
            return false;
        }
        true
    }
}

impl Gemma4Inner {
    fn record_output_parser_prompt_state(
        &self,
        tok: &Qwen3Tokenizer,
        rendered_tokens: &[u32],
    ) -> Result<()> {
        let open_channel = tok.encode_sync("<|channel>thought\n", Some(false))?;
        self.output_starts_in_reasoning_channel.store(
            !open_channel.is_empty() && rendered_tokens.ends_with(&open_channel),
            Ordering::Relaxed,
        );
        Ok(())
    }

    fn output_starts_in_reasoning_channel(&self) -> bool {
        self.output_starts_in_reasoning_channel
            .load(Ordering::Relaxed)
    }
}

impl ChatBackend for Gemma4Inner {
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        self.tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Tokenizer not loaded"))
    }

    fn family_name(&self) -> &'static str {
        "gemma4"
    }

    fn session_eos_id(&self, _tok: &Qwen3Tokenizer) -> Result<u32> {
        // Gemma4 stops on its `<turn|>` turn terminator, not `<|im_end|>`.
        self.turn_end_id()
    }

    fn policy(&self) -> engine::ThinkingPolicy {
        // Gemma4's selectable mode is a PROMPT capability (`<|think|>` in
        // the first system turn), not a Qwen-style `<think>...</think>`
        // decode region. Keep the generic tracker disabled: it has no
        // `<channel|>` end-token support and enabling it would incorrectly
        // classify every generated token as reasoning. Segmentation remains
        // downstream in `parse_gemma4_output` / `Gemma4StreamParser`, keyed
        // on `<|channel>` markers. Consequently Gemma4 still has no generic
        // think-budget forcing and reports `reasoning_tokens: 0`.
        engine::ThinkingPolicy::None
    }

    fn resolve_params(&self, config: &ChatConfig) -> ChatParams {
        let mut p = engine::extract_chat_params(config);
        // Fold the MODEL-config sampling defaults in; unset → T=0 greedy.
        // The engine's `sampling::sample` argmax fast path at T=0 is the
        // greedy argmax.
        p.sampling_config = make_sampling_config(config, &self.config);
        // gemma4 treats the penalty fields as no-ops. Neutralize so the
        // engine's `apply_all_penalties` skips all penalty work
        // structurally.
        p.repetition_penalty = 1.0;
        p.presence_penalty = 0.0;
        p.frequency_penalty = 0.0;
        // gemma4 ALWAYS returns Some(PerformanceMetrics), regardless of
        // `config.report_performance`.
        p.report_performance = true;
        // gemma4 never suppresses reasoning deltas at the loop level
        // (`include_reasoning` is a no-op here; the stream parser routes
        // channel segments itself). Defensive: pin `true` so the engine's
        // emitter gate can never suppress.
        p.include_reasoning = true;
        // Draft depth: with a draft model loaded, `mtpDepth` resolves per
        // variant — a family-local post-edit of the engine's central
        // `[1, 5]` clamp (an MTP-head contract that does not apply to
        // external drafts), always clamping from the RAW config value.
        //   * DSpark: unset runs full draft blocks (`block_size`, 7 on the
        //     v1 checkpoint) with the measured target-AR break-even guard;
        //     explicit depth pins the cap and disables that guard unless
        //     `mtpAdaptiveDepth: true` explicitly opts it back in.
        //   * Assistant: chained AR drafting has no checkpoint-pinned block
        //     size — unset resolves to `ASSISTANT_DEFAULT_DEPTH`; explicit
        //     values clamp to `[1, ASSISTANT_MAX_DEPTH]`.
        match self.draft.as_ref() {
            Some(Gemma4Draft::Dspark(draft)) => {
                let block_size = draft.config.block_size;
                p.mtp_depth = match config.mtp_depth {
                    Some(d) => (d.max(1) as usize).min(block_size),
                    None => block_size,
                };
                p.mtp_adaptive_depth = match config.mtp_adaptive_depth {
                    Some(enabled) => enabled,
                    None => config.mtp_depth.is_none(),
                };
            }
            Some(Gemma4Draft::Assistant(_)) => {
                p.mtp_depth = match config.mtp_depth {
                    Some(d) => (d.max(1) as usize).min(super::assistant::ASSISTANT_MAX_DEPTH),
                    None => super::assistant::ASSISTANT_DEFAULT_DEPTH,
                };
            }
            None => {}
        }
        p
    }

    /// Render every turn from the checkpoint-provided template. Gemma's
    /// stream parser also needs to know whether that template left the
    /// generation prompt inside an open thought channel.
    fn render_prompt(
        &self,
        tok: &Qwen3Tokenizer,
        messages: &[ChatMessage],
        config: &ChatConfig,
    ) -> Result<Vec<u32>> {
        let tokens = tok.apply_chat_template_sync(
            messages,
            Some(true),
            config.tools.as_deref(),
            engine::resolve_enable_thinking(config),
        )?;
        self.record_output_parser_prompt_state(tok, &tokens)?;
        Ok(tokens)
    }

    fn cached_token_history(&self) -> &[u32] {
        &self.cached_token_history
    }

    fn reset_caches(&mut self, scope: ResetScope) -> Result<()> {
        // Legacy miss branch ran `reset_caches_sync()? +
        // init_caches_sync()?` back-to-back (the flat prefill needs live
        // caches); the explicit command reset only cleared (caches stay
        // `None` until the next turn's lazy init).
        self.reset_caches_sync()?;
        if scope == ResetScope::PrefixMiss {
            self.init_caches_sync()?;
        }
        // The EXPLICIT command reset must restore a fully cold state.
        // gemma4's flat reset path (`reset_caches_sync`) never touches the
        // paged adapter, so a prior turn's request stays live AND its full
        // blocks stay content-addressed in the per-instance BlockAllocator's
        // prefix cache. A reset-then-rerun of the same prompt would then take
        // the prefix-hit suffix-prefill path (via `find_longest_cache_hit`
        // inside `prepare_gemma4_paged_turn`) — a different bf16 reduction
        // order than the cold full prefill, enough to flip a greedy
        // near-tie.
        // `release_request_and_purge_prefix_cache` releases the live request
        // (the release gemma4's reset otherwise skips) AND purges every
        // prefix-cache entry. The turn-internal `PrefixMiss` reset keeps the
        // prefix cache (cross-request block reuse after a history miss is the
        // paged design's entire point).
        if scope == ResetScope::Command
            && let Some(adapter) = self.paged_adapter.as_mut()
        {
            adapter
                .release_request_and_purge_prefix_cache()
                .map_err(|e| {
                    Error::from_reason(format!(
                        "gemma4 reset_caches: paged prefix-cache purge failed: {e}"
                    ))
                })?;
        }
        Ok(())
    }

    /// Prefix-reuse check. The engine routes every media-bearing turn
    /// through the multimodal executor BEFORE this check, so only the
    /// session-side media gate (`session_media()` non-empty → miss) is needed
    /// here; there is no `has_images` parameter.
    ///
    /// All-or-nothing: returns `0` or `cached.len()` (exact-match falls
    /// through the `hit == tokens.len()` branch in the session core to
    /// the miss/reset path — gemma4's sliding-window cache has no "rewind
    /// by one" primitive).
    fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool) -> usize {
        if !reuse_cache {
            return 0;
        }
        // Text-only prefix reuse: force a miss whenever the cached
        // session holds image or audio state UNLESS the media turn is
        // continuable (kept-live + sliding checkpoint at the full prefix). This
        // keeps prefix reuse strictly aligned with text-only sessions and
        // sidesteps the media-key coordination the Qwen3.5 shared helper
        // handles, while letting a continuable media session reuse an
        // exactly-cached prefix. Held state is `session_media()` (raw keys ∪
        // persistent `media_session_context`), not the raw keys alone: after a
        // failed media prepare on a warm-continued session only the context
        // survives, and the media-expanded cached history must not seed a
        // text-only prefix hit.
        if !self.session_media().is_empty() && !self.media_session_continuable {
            return 0;
        }
        // The live KV caches must exist — `cached_token_history` can
        // carry stale content after a prior `reset_caches_sync` if any
        // caller forgot to also clear it, so both must line up.
        if self.caches.is_none() {
            return 0;
        }
        let cached = &self.cached_token_history;
        if cached.is_empty() {
            return 0;
        }
        if tokens.len() < cached.len() {
            return 0;
        }
        if tokens[..cached.len()] != cached[..] {
            return 0;
        }
        cached.len()
    }

    fn save_cache_state(&mut self, args: SaveStateArgs<'_>) {
        // Flat save (identical on the fresh and delta paths): persist
        // `prompt + generated`, dropping the terminal turn-boundary token
        // when the decode terminated on stop so the cached history ends on
        // the `<turn|>` boundary the next delta re-renders itself.
        // Unconditional — there is no `reuse_cache` branch here (only the
        // paged core has one, and paged turns never reach this hook), and
        // the engine's session_start guard rejects `reuse_cache=Some(false)`
        // anyway.
        let history_tokens: &[u32] =
            if args.finish_reason != "length" && !args.generated_tokens.is_empty() {
                &args.generated_tokens[..args.generated_tokens.len() - 1]
            } else {
                args.generated_tokens
            };
        let mut new_history = Vec::with_capacity(args.save_tokens.len() + history_tokens.len());
        new_history.extend_from_slice(args.save_tokens);
        new_history.extend_from_slice(history_tokens);
        self.cached_token_history = new_history;
        if !args.is_delta {
            // Fresh text-only turn: clear any stale image/audio key (a
            // text-only turn has no multimodal key to set). Delta turns leave
            // them untouched — text-only by the delta image guard, so they are
            // structurally `None`.
            self.cached_image_key = None;
            self.cached_audio_key = None;
            self.media_session_context = MediaCapabilities::NONE;
            self.media_session_continuable = false;
        }
    }

    fn eval_caches(&self) -> Result<()> {
        // Materialize the prefill KV before entering the decode loop.
        eval_gemma4_caches(
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 eval_caches: caches missing"))?,
        )
    }

    /// Flat prefill for the engine's generic flow. `prefill_body_gemma4`
    /// processes `tokens[0 .. N-1]` through the body (a no-op when
    /// `N == 1`), the per-layer KV evals materialize, then the last
    /// token runs the full forward for sampling-ready `[1, vocab]`
    /// logits. Serves the fresh path (full prompt or strict-extend
    /// tail) and the session-delta path identically.
    ///
    /// `diagnostic::set_step(-1)` marks the prefill forward for
    /// `MLX_DEBUG_GEMMA4_DUMP`, uniformly across entry points.
    fn prefill(&mut self, prompt_tokens: &[u32], stream: Stream) -> Result<MxArray> {
        // Defensive: caches must be live before the prefill runs. The
        // engine's miss-reset re-inits, and verify/`has_live_session`
        // check liveness — but if somebody cleared the caches
        // out-of-band between turns, re-init here.
        if self.caches.is_none() {
            self.init_caches_sync()?;
        }

        let prefill_slice: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
        let prefill_len = prefill_slice.len();
        let prompt = MxArray::from_int32(&prefill_slice, &[1, prefill_len as i64])?;

        {
            let _stream_ctx = StreamContext::new(stream);
            let caches = self
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("Gemma4 prefill: caches missing"))?;
            prefill_body_gemma4(
                &prompt,
                &self.embed_tokens,
                &self.layers,
                caches,
                &self.final_norm,
                self.ple.as_ref(),
                &self.config,
                None,
            )?;
        }
        eval_gemma4_caches(
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 prefill: caches missing"))?,
        )?;

        // Last token → logits. `prefill_body_gemma4` processed
        // `[0 .. prefill_len - 1]` and left the final token for us.
        let last_token = prompt.slice_axis(1, prefill_len as i64 - 1, prefill_len as i64)?;
        let logits = {
            let _stream_ctx = StreamContext::new(stream);
            let caches = self
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("Gemma4 prefill: caches missing"))?;
            crate::models::gemma4::diagnostic::set_step(-1);
            forward_inner(
                &last_token,
                &self.embed_tokens,
                &self.layers,
                caches,
                &self.final_norm,
                &self.lm_head,
                self.embed_weight_t.as_ref(),
                self.ple.as_ref(),
                &self.config,
                None,
            )?
        };
        logits.squeeze(Some(&[1]))
    }

    type Decode<'a>
        = Gemma4Decode<'a>
    where
        Self: 'a;

    fn begin_decode(&mut self, _turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
        // No compiled path, no turn-constant captures: gemma4's eager
        // decode threads everything through the live session caches.
        Ok(Gemma4Decode {
            inner: self,
            step: 0,
        })
    }

    /// Gemma4 output finalization: raw decode (`skip_special_tokens =
    /// false` so the channel/tool-call DSL markers survive) →
    /// `parse_gemma4_output` → `promote_channel_only_output` →
    /// tool-calls finish-reason promotion. `reasoning_tokens` arrives as
    /// 0 (thinking disabled) and `prompt_tokens` / `performance` are
    /// passed through unchanged. `cached_tokens` is overwritten by the
    /// session core.
    fn finalize_turn(&self, args: FinalizeArgs<'_>) -> Result<ChatResult> {
        let raw_text = args.tokenizer.decode_sync(args.generated_tokens, false)?;
        let starts_in_prompted_channel = self.output_starts_in_reasoning_channel();
        let mut parsed = super::output_parser::parse_gemma4_output_with_open_channel(
            &raw_text,
            starts_in_prompted_channel,
        );
        promote_channel_only_output(&mut parsed, starts_in_prompted_channel);
        let finish_reason = if parsed.tool_calls.iter().any(|tc| tc.status == "ok") {
            "tool_calls".to_string()
        } else {
            args.finish_reason
        };
        Ok(ChatResult {
            text: parsed.text,
            tool_calls: parsed.tool_calls,
            thinking: parsed.thinking,
            // Generic and paged session cores overwrite this with the
            // effective Jinja kwarg after family finalization.
            thinking_enabled: args.thinking_enabled,
            num_tokens: args.generated_tokens.len() as u32,
            prompt_tokens: args.prompt_tokens,
            reasoning_tokens: args.reasoning_tokens,
            finish_reason,
            raw_text,
            public_raw_text: None,
            cached_tokens: 0,
            performance: args.performance,
        })
    }

    fn execution_plan(&self) -> ExecutionPlan {
        let paged_available = self.paged_adapter.is_some();
        let image_components_loaded = self.image_path_loaded();
        let audio_embedder_loaded = self.embed_audio.is_some();
        ExecutionPlan {
            media: gemma4_media_plan(
                image_components_loaded,
                audio_embedder_loaded,
                paged_available,
            ),
            paged_attention: self.paged_adapter.as_ref().map(|_| PagedAttentionPlan {
                supports_delta: true,
            }),
            speculative: self.has_draft().then_some(SpeculativePlan {
                kind: SpeculativeKind::DraftModel,
                supported_input_media: MediaCapabilities::NONE,
                supported_context_media: MediaCapabilities::NONE,
                supports_paged_attention: false,
            }),
        }
    }

    fn extra_eos_ids(&self) -> Vec<u32> {
        // The MODEL-config eos list (`<eos>` / `<end_of_turn>`) honored
        // alongside the session `<turn|>` id. A negative config id can
        // never equal a `u32`-cast token, so filter those out instead of
        // wrapping.
        self.config
            .eos_token_ids
            .iter()
            .filter(|&&id| id >= 0)
            .map(|&id| id as u32)
            .collect()
    }

    fn stream_skip_special_tokens(&self) -> bool {
        // `decode_stream(false)`: the stream parser must see the
        // `<|channel>` / `<|tool_call>` markers. The residual flush then
        // decodes with the same flag (engine guarantee), keeping
        // `streamed_text_len` accounting consistent.
        false
    }

    fn stream_emitter(&self) -> Box<dyn StreamEmitter> {
        Box::new(Gemma4Emitter::new(
            self.output_starts_in_reasoning_channel(),
        ))
    }

    // `augment_performance` deliberately NOT overridden: the default
    // (`profiler.fill_mtp_acceptance`) fills the `mtp_*` acceptance fields
    // after a DSpark turn (and copies `profile_phases` when profiling is
    // enabled). AR turns record no MTP cycle, so their acceptance fields
    // stay `None` as before.

    fn has_live_session(&self) -> bool {
        // Requires an initialized session: a non-empty
        // `cached_token_history` AND live `caches`.
        !self.cached_token_history.is_empty() && self.caches.is_some()
    }

    fn session_media(&self) -> MediaCapabilities {
        // Keys cover a just-finalized media turn and direct/transitional test
        // states. `media_session_context` remains authoritative after warm
        // text saves clear those keys while preserving the same live media KV.
        self.media_session_context.union(MediaCapabilities {
            images: self.cached_image_key.is_some(),
            audio: self.cached_audio_key.is_some(),
        })
    }

    fn run_paged_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        // The execution plan admits every text turn shape (fresh + delta,
        // sync + streaming) when the adapter is loaded. The generic paged
        // engine drives the lifecycle via [`PagedBackend`].
        debug_assert!(args.plan.use_paged_attention);
        debug_assert!(self.paged_adapter.is_some());
        debug_assert!(matches!(args.plan.decoder, DecoderPlan::Autoregressive));
        debug_assert!(self.paged_text_turn_context.is_empty());
        self.paged_text_turn_context = args.plan.context_media;
        let result = crate::engine::paged_turn::run_paged_turn(self, args);
        self.paged_text_turn_context = MediaCapabilities::NONE;
        result
    }

    /// Draft speculative-decode whole-turn path (either [`Gemma4Draft`]
    /// variant). The execution plan admits this handler only after request
    /// opt-in, with a loaded draft, flat KV, and text-only input.
    fn run_speculative_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        debug_assert!(args.media.is_empty());
        debug_assert!(matches!(
            args.plan.decoder,
            DecoderPlan::Speculative(SpeculativeKind::DraftModel)
        ));
        self.draft_chat_turn(args)
    }

    fn run_multimodal_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        debug_assert!(!args.media.is_empty());
        self.multimodal_chat_turn(args)
    }
}

#[napi]
impl Gemma4Model {
    /// Create an uninitialized `Gemma4Model` stub from a config.
    ///
    /// **Prefer [`Gemma4Model::load`]** for any real usage — `new(config)`
    /// is a config-only stub that matches the OCR-model pattern
    /// (`VLModel::new(config)`, `QianfanOCRModel::new(config)`) and is
    /// intentionally NOT runnable. It was introduced in the cache-limit
    /// coordinator work so that the coordinator's per-model delta is
    /// registered exclusively on the `load()` path, eliminating a
    /// baseline-registration gap where a no-op `new(config)` would have
    /// leaked an empty guard into the coordinator.
    ///
    /// This path does NOT spawn a model thread, NOT materialize any
    /// weights, and NOT register with the cache-limit coordinator. The
    /// returned instance is only useful for config inspection — every
    /// session method (`chatSessionStart` / `chatSessionContinue` /
    /// `chatSessionContinueTool` and their streaming variants) rejects
    /// with a `napi::Error` whose message is exactly
    /// `"Model not initialized. Call Gemma4Model.load() first."` until
    /// `load()` runs and installs the underlying model thread. The
    /// synchronous `resetCaches()` call is a silent no-op on the stub
    /// to keep `ChatSession.reset()` idempotent across both runnable
    /// and stub instances.
    ///
    /// A runnable model requires `await Gemma4Model.load(path)`. The
    /// constructor signature is fixed by NAPI-RS; the stub-only behavior is
    /// covered by the regression tests in
    /// `__test__/models/model-loader-gemma4.test.ts`.
    #[napi(constructor)]
    pub fn new(config: Gemma4Config) -> Self {
        let has_vision = config.vision_config.is_some() || config.unified_vision_config.is_some();
        let has_audio = config.has_audio;
        Self {
            thread: None,
            model_id: 0,
            has_vision,
            has_audio,
            initialized: false,
            paged_active: false,
            _cache_limit_guard: None,
            draft_active: false,
        }
    }

    /// Returns true if weights have been loaded via `load()`.
    #[napi(getter)]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Whether the block-paged KV cache adapter is active on this model
    /// instance.
    ///
    /// `true` iff `Gemma4Inner::paged_adapter` was successfully
    /// constructed at load time (driven by
    /// `Gemma4Config::use_block_paged_cache`). The
    /// `gemma4_paged_vs_flat_parity` integration test pins greedy
    /// byte-equal at BF16 against real Gemma-4-E2B-IT weights. Stubs
    /// constructed via `new(config)` always return `false`. Surfaced
    /// through this NAPI method so server endpoints can branch on it
    /// without a model-thread roundtrip.
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged_active
    }

    /// Whether this loaded instance can execute image-bearing chat turns.
    /// Config-only stubs and incomplete/non-paged physical paths return false.
    #[napi]
    pub fn supports_images(&self) -> bool {
        self.initialized && self.paged_active && self.has_vision
    }

    #[napi]
    pub fn model_id(&self) -> u32 {
        self.model_id as u32
    }

    /// Whether a draft model — DSpark or Google assistant — is loaded on
    /// this instance (via `Gemma4LoadOptions::draft_model_path`), enabling
    /// the speculative-decode whole-turn path.
    ///
    /// Note: this only reports draft availability. Whether speculative
    /// decoding actually runs on a given call also requires the per-request
    /// `enableMtp` flag. Named `hasMtpWeights` for parity with the Qwen3.5
    /// surface, but it reports an external draft model (either variant),
    /// not in-checkpoint MTP heads. Stubs from `new(config)` always return
    /// `false`.
    #[napi]
    pub fn has_mtp_weights(&self) -> bool {
        self.draft_active
    }

    /// Load a Gemma4 model from a directory.
    #[napi]
    pub async fn load(
        model_path: String,
        options: Option<Gemma4LoadOptions>,
    ) -> Result<Gemma4Model> {
        Self::load_from_dir(&model_path, options).await
    }

    /// Test-only entry point that dispatches `ChatCmd::StreamSessionStart`
    /// and returns the raw mpsc receiver the model thread writes into, so a
    /// pure-Rust integration test can exercise the streaming path without a
    /// NAPI host (same pattern as `Qwen3_5Model::chat_stream_session_start_for_test`).
    #[doc(hidden)]
    pub fn chat_stream_session_start_for_test(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<(
        crate::engine::types::ChatStreamHandle,
        tokio::sync::mpsc::UnboundedReceiver<Result<ChatStreamChunk>>,
    )> {
        let thread = self.thread.as_ref().ok_or_else(|| {
            Error::from_reason("Model not initialized. Call Gemma4Model.load() first.")
        })?;
        let config = config.unwrap_or_default();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_inner = cancelled.clone();
        let (stream_tx, stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatStreamChunk>>();
        thread.send(ChatCmd::StreamSessionStart {
            messages,
            config,
            stream_tx,
            cancelled: cancelled_inner,
        })?;
        Ok((
            crate::engine::types::ChatStreamHandle { cancelled },
            stream_rx,
        ))
    }
}

crate::models::chat_napi::chat_napi_surface! {
    class: Gemma4Model,
    thread_cmd: crate::engine::cmd::ChatCmd,
    thread: { option: "Model not initialized. Call Gemma4Model.load() first." },
    image_guard: { vision: has_vision, audio: has_audio },
    ts_stream_start: "messages: ChatMessage[], config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue: "messages: ChatMessage[], config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue_tool: "messages: ChatMessage[], config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
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
fn init_caches_for_config(config: &Gemma4Config) -> Vec<Gemma4LayerCache> {
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
fn is_eos_token(token: u32, eos_ids: &[i32], eos_token_id: u32) -> bool {
    if eos_ids.contains(&(token as i32)) {
        return true;
    }
    eos_token_id == token
}

#[derive(Clone, Copy)]
struct Gemma4RepetitionCutoff {
    max_consecutive_tokens: i32,
    max_ngram_repeats: i32,
    ngram_size: i32,
}

fn repetition_cutoff_from_config(config: &ChatConfig) -> Gemma4RepetitionCutoff {
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

fn check_gemma4_repetition_cutoff(
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

fn make_sampling_config(
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

fn sample_next_token(logits: &MxArray, config: Option<SamplingConfig>) -> Result<MxArray> {
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
/// When `tap` is provided, the residual-stream hidden of each tapped layer
/// (post residual add, PRE final-norm) is pushed onto `tap.captured` in
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
    mut tap: Option<&mut DsparkTap<'_>>,
) -> Result<MxArray> {
    if let Some(t) = tap.as_deref() {
        let mut previous: Option<usize> = None;
        for &id in t.layer_ids {
            if id >= layers.len() || previous.is_some_and(|prev| id <= prev) {
                return Err(Error::from_reason(format!(
                    "forward_body: tap layer_ids {:?} must be strictly ascending decoder indices below {}",
                    t.layer_ids,
                    layers.len()
                )));
            }
            previous = Some(id);
        }
    }

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

        // Residual-stream hidden of layer i (post residual add, pre
        // final-norm — HF `hidden_states[i + 1]`), captured for both the
        // regular and the KV-shared branch.
        if let Some(t) = tap.as_deref_mut()
            && t.layer_ids.contains(&i)
        {
            t.captured.push(h.clone());
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
    tap: Option<&mut DsparkTap<'_>>,
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
        tap,
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

/// Run the target over a `[1, T]` verify block at the current cache offset,
/// capturing the tapped hidden states, and return the `[1, T, vocab]` logits.
///
/// This is exactly the existing T>1-at-offset forward (`forward_inner`, with
/// the same masks/rope the chunked prefill uses). It does not sample and
/// touches no history bookkeeping; caches advance by T. Callers pair it with
/// `snapshot_before_verify` / `commit_after_verify` for rollback.
pub(crate) fn dspark_verify_forward(
    block_ids: &MxArray,
    embedding: &Embedding,
    layers: &[Gemma4DecoderLayer],
    caches: &mut [Gemma4LayerCache],
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    embed_weight_t: Option<&MxArray>,
    ple: Option<&PleComponents>,
    config: &Gemma4Config,
    tap: &mut DsparkTap<'_>,
) -> Result<MxArray> {
    if block_ids.ndim()? != 2 || block_ids.shape_at(0)? != 1 || block_ids.shape_at(1)? < 1 {
        return Err(Error::from_reason(format!(
            "dspark_verify_forward expects block_ids shaped [1, T] with T >= 1, got {:?}",
            block_ids.shape()?.as_ref()
        )));
    }
    forward_inner(
        block_ids,
        embedding,
        layers,
        caches,
        final_norm,
        lm_head,
        embed_weight_t,
        ple,
        config,
        Some(tap),
    )
}

/// Run the target over a `[1, T]` verify block at the current cache offset
/// and return the `[1, T, vocab]` softcapped logits together with the
/// `[1, T, hidden]` post-final-norm hidden state (the assistant draft chains
/// its next round's `h_prev` from the hidden at the last kept slot).
///
/// Same forward as [`dspark_verify_forward`] minus the residual-stream tap:
/// it does not sample and touches no history bookkeeping; caches advance by
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
        None,
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

/// Build the per-layer routing list for the paged dispatch (pure
/// function over a `Gemma4Config`).
///
/// Returns `Vec<Gemma4LayerKind>` of length `config.num_hidden_layers`
/// where each entry classifies a layer as:
/// * `Sliding` — stays on the flat `Gemma4LayerCache::Sliding` path.
/// * `GlobalPaged { paged_idx }` — routes through the paged adapter
///   at the given global-layer ordinal.
/// * `SharedOnGlobal { anchor_paged_idx }` — KV-shared layer whose
///   anchor is a global layer; reads K/V via the adapter.
/// * `SharedOnSliding { anchor_layer_idx }` — KV-shared layer whose
///   anchor is a sliding layer; reads K/V from the anchor's flat
///   cache stash.
///
/// `paged_idx` counts only physical non-shared `full_attention` layers in
/// their original decoder order — matches the `LayerKVPool` slot count from
/// `Gemma4Inner::new`. KV-shared layers do NOT consume a paged slot (they
/// reuse the anchor's K/V); the shared variants carry the anchor's index so
/// the shared forward path can resolve it.
///
/// Lifted to a free helper so unit tests can drive it without owning a
/// `Gemma4Inner` (which requires loaded weights). Mirrors LFM2's
/// `compute_layer_kinds` pattern.
#[cfg(test)]
pub(crate) fn compute_layer_kinds(config: &Gemma4Config) -> Vec<Gemma4LayerKind> {
    compute_layer_kinds_from_kv_cache_specs(config)
        .expect("Gemma4 layer kinds must derive from valid KV cache specs")
}

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
    let routes = derive_layer_kv_cache_routes(
        &specs,
        max_model_len,
        gemma4_paged_prefill_group_max_chunk(),
    )
    .map_err(|e| format!("Gemma4 layer kind route derivation failed: {e}"))?;

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
                anchor_paged_idx: physical_ordinal,
            },
            (Some(anchor), AttentionKind::SlidingWindow { .. }) => {
                let anchor_layer_idx = u32::try_from(anchor).map_err(|_| {
                    format!("Gemma4 shared sliding anchor layer {anchor} does not fit u32")
                })?;
                Gemma4LayerKind::SharedOnSliding { anchor_layer_idx }
            }
            (None, AttentionKind::Full) => Gemma4LayerKind::GlobalPaged {
                paged_idx: physical_ordinal,
            },
            (None, AttentionKind::SlidingWindow { .. }) => Gemma4LayerKind::Sliding,
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

fn physical_full_attention_layer_count(specs: &[LayerKVCacheSpec]) -> usize {
    specs
        .iter()
        .filter(|spec| {
            spec.shared_kv_anchor.is_none() && matches!(spec.attention_kind, AttentionKind::Full)
        })
        .count()
}

fn gemma4_default_paged_cache_memory_mb(
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

/// Default prefill chunk size (tokens per chunk).
/// Note: mlx-lm uses 2048 but the first eval triggers Metal shader compilation
/// which can GPU-timeout with very large graphs. Using 512 keeps individual
/// command buffers under Metal's timeout limit.
pub(crate) const GEMMA4_PREFILL_STEP_SIZE: i64 = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma4SlidingRestoreLimitOverride {
    Cap(u32),
    Uncapped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Gemma4SlidingRestoreSuppression {
    limit: u32,
    source: &'static str,
}

fn parse_gemma4_sliding_restore_limit(value: &str) -> Option<Gemma4SlidingRestoreLimitOverride> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    if matches!(
        value.to_ascii_lowercase().as_str(),
        "off" | "none" | "false" | "no" | "unlimited" | "uncapped"
    ) {
        return Some(Gemma4SlidingRestoreLimitOverride::Uncapped);
    }
    value
        .parse::<u32>()
        .ok()
        .map(Gemma4SlidingRestoreLimitOverride::Cap)
}

fn gemma4_sliding_restore_limit_override() -> Option<Gemma4SlidingRestoreLimitOverride> {
    static OVERRIDE: OnceLock<Option<Gemma4SlidingRestoreLimitOverride>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("MLX_GEMMA4_MAX_SLIDING_RESTORE_TOKENS")
            .ok()
            .and_then(|value| parse_gemma4_sliding_restore_limit(&value))
    })
}

fn gemma4_default_sliding_restore_limit(config: &Gemma4Config, block_size: u32) -> Option<u32> {
    let interval = gemma4_sliding_decode_checkpoint_interval(config, block_size);
    (interval > 0).then_some(interval)
}

fn gemma4_large_sliding_restore_suppression_limit_for_override(
    config: &Gemma4Config,
    block_size: u32,
    override_limit: Option<Gemma4SlidingRestoreLimitOverride>,
    restore_tokens: u32,
) -> Option<Gemma4SlidingRestoreSuppression> {
    let (limit, source) = match override_limit {
        Some(Gemma4SlidingRestoreLimitOverride::Uncapped) => return None,
        Some(Gemma4SlidingRestoreLimitOverride::Cap(limit)) => (limit, "env"),
        None => (
            gemma4_default_sliding_restore_limit(config, block_size)?,
            "default",
        ),
    };
    (restore_tokens > limit).then_some(Gemma4SlidingRestoreSuppression { limit, source })
}

fn gemma4_large_sliding_restore_suppression_limit(
    config: &Gemma4Config,
    block_size: u32,
    restore_tokens: u32,
) -> Option<Gemma4SlidingRestoreSuppression> {
    gemma4_large_sliding_restore_suppression_limit_for_override(
        config,
        block_size,
        gemma4_sliding_restore_limit_override(),
        restore_tokens,
    )
}

fn parse_gemma4_sliding_checkpoint_limit(value: &str) -> Option<usize> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    value.parse::<usize>().ok().filter(|limit| *limit > 0)
}

fn gemma4_sliding_checkpoint_limit_override() -> Option<usize> {
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("MLX_GEMMA4_SLIDING_CHECKPOINT_LIMIT")
            .ok()
            .and_then(|value| parse_gemma4_sliding_checkpoint_limit(&value))
    })
}

fn gemma4_sliding_prefix_checkpoint_limit_for_override(
    config: &Gemma4Config,
    block_size: u32,
    override_limit: Option<usize>,
) -> usize {
    if let Some(limit) = override_limit {
        return limit;
    }
    let sliding_window = config.sliding_window.max(0) as usize;
    let block_size = block_size as usize;
    if sliding_window == 0 || block_size == 0 {
        return GEMMA4_SLIDING_PREFIX_CHECKPOINT_MIN_LIMIT;
    }
    let logical_limit = sliding_window
        .div_ceil(block_size)
        .saturating_mul(GEMMA4_SLIDING_PREFIX_CHECKPOINT_WINDOW_MULTIPLIER)
        .clamp(
            GEMMA4_SLIDING_PREFIX_CHECKPOINT_MIN_LIMIT,
            GEMMA4_SLIDING_PREFIX_CHECKPOINT_MAX_DEFAULT_LIMIT,
        );
    let checkpoint_bytes = gemma4_sliding_checkpoint_estimated_bytes(config);
    if checkpoint_bytes == 0 {
        return logical_limit;
    }
    let memory_limit = (GEMMA4_SLIDING_CHECKPOINT_MEMORY_BUDGET_BYTES / checkpoint_bytes)
        .max(GEMMA4_IMAGE_PREFIX_CHECKPOINT_LIMIT as u64);
    logical_limit.min(usize::try_from(memory_limit).unwrap_or(usize::MAX))
}

fn gemma4_sliding_checkpoint_estimated_bytes(config: &Gemma4Config) -> u64 {
    let physical_sliding_layers = (0..config.num_hidden_layers.max(0) as usize)
        .filter(|&layer_idx| {
            config.is_sliding_layer(layer_idx) && !config.is_kv_shared_layer(layer_idx)
        })
        .count() as u64;
    if physical_sliding_layers == 0 {
        return 0;
    }
    // Conservatively budget four bytes per element. Most shipped checkpoints
    // use BF16 caches, but the snapshot type does not promise that dtype and an
    // f32 load must not multiply a 128-entry logical limit into an OOM.
    physical_sliding_layers
        .saturating_mul(config.sliding_window.max(0) as u64)
        .saturating_mul(config.num_key_value_heads.max(0) as u64)
        .saturating_mul(config.head_dim.max(0) as u64)
        .saturating_mul(2) // K + V
        .saturating_mul(4) // conservative bytes per element
}

/// Conservative bytes one checkpoint at `boundary_tokens` occupies.
///
/// The payload a sliding checkpoint holds is `min(boundary, window)` token
/// rows — exactly what a live `RotatingKVCache` holds at that offset, and what
/// `sliding_sidecar::payload_tokens` writes. Sizing every entry at a FULL
/// window (which is what [`gemma4_sliding_checkpoint_estimated_bytes`] does, and
/// all any pre-ladder caller needed) is what makes a sub-window rung look as
/// expensive as a deep one: on the 12B geometry a rung at 64 tokens costs
/// 41.9 MB, not the 671.1 MB the flat estimate charges it.
fn gemma4_sliding_checkpoint_estimated_bytes_at(
    config: &Gemma4Config,
    boundary_tokens: u32,
) -> u64 {
    Gemma4SlidingCheckpointBytes::for_config(config).at(boundary_tokens)
}

/// Anchor rungs this config publishes for the cold sidecar, ascending.
///
/// `block_size * RATIO^k` for `k = 1..`, capped by
/// [`GEMMA4_SLIDING_ANCHOR_MAX_RUNGS`] and by
/// [`GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES`] minus the reserve the
/// pre-ladder limit already claims. Empty when there is no sliding state, no
/// block size, or the reserve alone already fills the budget — in which case
/// the ladder degenerates to today's behaviour rather than overrunning memory.
///
/// Pure function of `(config, block_size, base_limit)`: the same grid every
/// turn and every process, which is the whole point (see
/// [`GEMMA4_SLIDING_ANCHOR_RATIO`]).
fn gemma4_sliding_cold_anchor_rungs(
    config: &Gemma4Config,
    block_size: u32,
    base_limit: usize,
) -> Vec<u32> {
    let full_window_bytes = gemma4_sliding_checkpoint_estimated_bytes(config);
    if block_size == 0 || full_window_bytes == 0 {
        return Vec::new();
    }
    let reserve = full_window_bytes.saturating_mul(base_limit as u64);
    let mut budget = GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES.saturating_sub(reserve);
    let mut rungs = Vec::with_capacity(GEMMA4_SLIDING_ANCHOR_MAX_RUNGS);
    let mut rung = block_size;
    for _ in 0..GEMMA4_SLIDING_ANCHOR_MAX_RUNGS {
        let Some(next) = rung.checked_mul(GEMMA4_SLIDING_ANCHOR_RATIO) else {
            break;
        };
        rung = next;
        let cost = gemma4_sliding_checkpoint_estimated_bytes_at(config, rung);
        if cost > budget {
            break;
        }
        budget -= cost;
        rungs.push(rung);
    }
    rungs
}

fn gemma4_sliding_retention_caps_for_override(
    config: &Gemma4Config,
    block_size: u32,
    want_ladder: bool,
    override_limit: Option<usize>,
) -> Gemma4SlidingRetentionCaps {
    let base_limit =
        gemma4_sliding_prefix_checkpoint_limit_for_override(config, block_size, override_limit);
    let bytes = Gemma4SlidingCheckpointBytes::for_config(config);
    if !want_ladder {
        return Gemma4SlidingRetentionCaps::pre_ladder(base_limit, bytes);
    }
    let anchors = Gemma4SlidingAnchorRungs::from_slice(&gemma4_sliding_cold_anchor_rungs(
        config, block_size, base_limit,
    ));
    // An explicit override is the operator's final word on how many entries fit
    // in memory; widening it behind their back would defeat the knob.
    if override_limit.is_some() {
        return Gemma4SlidingRetentionCaps::ladder(base_limit, anchors, bytes);
    }
    Gemma4SlidingRetentionCaps::ladder(base_limit.saturating_add(anchors.len), anchors, bytes)
}

/// Retention for this turn. `want_ladder` is
/// [`gemma4_sliding_cold_ladder_wanted`] — the SAME predicate that decides
/// whether anchor rungs are published, so the two cannot disagree.
fn gemma4_sliding_retention_caps(
    config: &Gemma4Config,
    block_size: u32,
    want_ladder: bool,
) -> Gemma4SlidingRetentionCaps {
    gemma4_sliding_retention_caps_for_override(
        config,
        block_size,
        want_ladder,
        gemma4_sliding_checkpoint_limit_override(),
    )
}

/// Whether this turn's cold tier will actually consume a checkpoint ladder.
///
/// The anchor rungs exist for ONE consumer:
/// [`Gemma4Inner::capture_gemma4_sliding_cold_sidecar`], which can only anchor a
/// sidecar where the persisted K/V chain already reaches. With no
/// `SlidingWindow` sidecar policy installed nothing can ever read them, so
/// publishing them is pure cost — extra `RotatingKVCacheSnapshot`s held
/// resident, and, worse, a different retained SET, which moves the depth a later
/// warm turn resumes from and therefore the tokens it emits.
///
/// Single source of truth for the published rung set
/// ([`gemma4_sliding_chunk_checkpoint_boundaries`]), retention
/// ([`gemma4_sliding_retention_caps`]), the decode publish union
/// ([`gemma4_sliding_decode_publishes_checkpoint`]) and the ladder byte cap
/// ([`trim_gemma4_sliding_prefix_checkpoints`]), so the four cannot disagree
/// about whether this is a persist turn. Mirrors
/// `qwen3_5::paged_forward::gdn_cold_sidecar_ladder_wanted`.
///
/// A free function of the cold-tier context rather than a `&Gemma4Inner` method
/// on purpose: it is the master switch for all four behaviours, and as a method
/// it was reachable only from a loaded checkpoint on a GPU, i.e. from no test at
/// all. The one thing left on the `Gemma4Inner` side is the borrow
/// `paged_adapter -> cold_tier()`, which
/// `paged_kv_cache_adapter::tests::cold_tier_defaults_none_and_holds_context_across_resets`
/// already pins.
fn gemma4_sliding_cold_ladder_wanted(cold: Option<&ColdTierContext>) -> bool {
    cold.and_then(|cold| cold.sidecar_policy.as_ref())
        .is_some_and(|policy| policy.group() == mlx_paged_attn::ColdGroup::SlidingWindow)
}

/// Retention caps for a turn whose adapter carries `cold`. THE production
/// derivation: every publish and retention seam reads its caps from here (via
/// [`Gemma4Inner::gemma4_sliding_retention_caps_for_turn`]), and nothing else in
/// production chooses the `want_ladder` boolean.
fn gemma4_sliding_retention_caps_for_cold_tier(
    config: &Gemma4Config,
    cold: Option<&ColdTierContext>,
    block_size: u32,
) -> Gemma4SlidingRetentionCaps {
    gemma4_sliding_retention_caps(config, block_size, gemma4_sliding_cold_ladder_wanted(cold))
}

fn gemma4_sliding_decode_checkpoint_interval(config: &Gemma4Config, block_size: u32) -> u32 {
    if block_size == 0 {
        return 0;
    }
    let sliding_window = config.sliding_window.max(0) as u32;
    let target = sliding_window.max(block_size);
    target.div_ceil(block_size).saturating_mul(block_size)
}

/// Whether a decode cursor sitting at `prefix_len` publishes a sliding
/// checkpoint: the cadence UNION this turn's anchor rungs.
///
/// The union is not a nicety. `gemma4_sliding_decode_checkpoint_interval` is
/// `max(window, block).div_ceil(block) * block` = 1024 on the 12B, and
/// `window / block_size = 64 = 4^3`, so EVERY rung with `k >= 3` is also a
/// cadence boundary and every rung below the window is not. The only other
/// publisher is `gemma4_sliding_chunk_checkpoint_boundaries`, whose filter is
/// strict (`rung > start_offset`). So for the shape `mlx agent` actually sends
/// — a short prompt and a long generation — the rung at 256 was published by
/// nothing at all:
///
/// ```text
///   turn 1 prefill 0..199   publishes {64}
///   turn 1 decode  200..N   cadence only: 1024, 2048, ...   256 never fires
///   turn 2 prefill starts at 200+generated  ->  rung > start refuses 256
/// ```
///
/// Gated on `caps.wants_ladder()`: a persistence-OFF turn keeps the bare
/// cadence, because publishing an extra checkpoint changes the retained set and
/// therefore the depth a later warm turn resumes from, and that is observable
/// in the emitted tokens.
fn gemma4_sliding_decode_publishes_checkpoint(
    prefix_len: u32,
    checkpoint_interval: u32,
    caps: Gemma4SlidingRetentionCaps,
) -> bool {
    if prefix_len == 0 {
        return false;
    }
    let on_cadence = checkpoint_interval != 0 && prefix_len.is_multiple_of(checkpoint_interval);
    on_cadence || (caps.wants_ladder() && caps.anchors.contains(prefix_len))
}

/// Whether `prefix_len` sits on the `block_size * RATIO^k` GRID the anchor
/// rungs are drawn from — ignoring the byte budget that may have truncated the
/// published set, so this is a strict SUPERSET of `caps.anchors.contains`.
///
/// Exists purely so the decode hot path can skip deriving `caps` on the steps
/// that cannot possibly publish. It is a handful of integer ops; `caps` walks
/// `num_hidden_layers` doing `String == "full_attention"` three times over (96
/// string compares on the 12B) plus an env-var `OnceLock` read. HEAD returned
/// early on every non-cadence decode step and this restores that, for the
/// ladder arm and the persistence-OFF arm alike.
fn gemma4_sliding_prefix_len_is_on_the_anchor_grid(prefix_len: u32, block_size: u32) -> bool {
    if block_size == 0 || prefix_len == 0 {
        return false;
    }
    let mut rung = block_size;
    for _ in 0..GEMMA4_SLIDING_ANCHOR_MAX_RUNGS {
        let Some(next) = rung.checked_mul(GEMMA4_SLIDING_ANCHOR_RATIO) else {
            return false;
        };
        rung = next;
        if rung == prefix_len {
            return true;
        }
        if rung > prefix_len {
            return false;
        }
    }
    false
}

/// What a decode step at `prefix_len` publishes, or `None` for the overwhelming
/// majority of steps that publish nothing.
///
/// This is the whole DECISION half of
/// [`Gemma4Inner::maybe_remember_gemma4_sliding_decode_boundary_checkpoint`],
/// pulled out as a free function of `(config, cold tier, block size, cursor)`.
/// Its call site contributes nothing but three adapter reads, which is the
/// point: hard-coding `want_ladder` to `false` at that call site used to revert
/// decode to cadence-only — the exact defect the rung union fixed — while both
/// decode tests passed, because they called
/// [`gemma4_sliding_decode_publishes_checkpoint`] directly and never reached
/// production's caps derivation.
///
/// Ordering is deliberate, and it is what keeps this off the decode hot path:
///
/// ```text
///   prefix_len == 0                        one compare
///   cadence multiple                       one modulo
///   is there a SlidingWindow sidecar       two pointer derefs + an enum compare
///   block_size * 4^k for some k <= 4       at most four multiplies
///   ---- only now ----
///   derive caps                            walks num_hidden_layers at least
///                                          three times, and up to seven on the
///                                          ladder arm (the limit, the cost
///                                          model, and one per admitted rung),
///                                          each layer a `String ==
///                                          "full_attention"` compare — 144+
///                                          string compares on the 12B — plus
///                                          an env-var OnceLock read
/// ```
///
/// A persistence-OFF turn therefore pays exactly what it paid before the ladder
/// existed: the cold-tier probe fails and it returns on the same non-boundary
/// steps HEAD returned on, having derived no caps at all. A persist turn pays
/// the derivation on the cadence boundaries plus at most four cursors a turn.
fn gemma4_sliding_decode_boundary_plan(
    config: &Gemma4Config,
    cold: Option<&ColdTierContext>,
    block_size: u32,
    prefix_len: u32,
) -> Option<Gemma4SlidingDecodeBoundary> {
    if prefix_len == 0 {
        return None;
    }
    let checkpoint_interval = gemma4_sliding_decode_checkpoint_interval(config, block_size);
    let on_cadence = checkpoint_interval != 0 && prefix_len.is_multiple_of(checkpoint_interval);
    // Exact, not approximate: `caps.anchors` is always a SUBSET of the
    // `block_size * 4^k` grid (the byte budget can only truncate it), and
    // `caps.wants_ladder()` is `gemma4_sliding_cold_ladder_wanted(cold)` by
    // construction, so the two screens together can only skip cursors the full
    // predicate below would have rejected anyway.
    if !on_cadence
        && !(gemma4_sliding_cold_ladder_wanted(cold)
            && gemma4_sliding_prefix_len_is_on_the_anchor_grid(prefix_len, block_size))
    {
        return None;
    }
    let caps = gemma4_sliding_retention_caps_for_cold_tier(config, cold, block_size);
    if !gemma4_sliding_decode_publishes_checkpoint(prefix_len, checkpoint_interval, caps) {
        return None;
    }
    Some(Gemma4SlidingDecodeBoundary {
        prefix_len,
        block_size,
        checkpoint_interval,
        on_anchor_rung: caps.wants_ladder() && caps.anchors.contains(prefix_len),
    })
}

/// What the cold tier already holds at one capture candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma4ColdCaptureProbe<K> {
    /// The chain derives and nothing is on disk under it: capture here.
    Missing(K),
    /// The chain derives and the object is already on disk.
    Persisted,
    /// The chain cannot be derived at this boundary at all, so neither side can
    /// name it. Not a skip — nothing was ever written here to skip.
    Underivable,
}

/// How a descent over the capture candidates ended.
///
/// Three outcomes, not two, because the two that capture nothing are different
/// states of the tier and the counters must not read them as one: a descent
/// that found everything already written is a healthy saturated ladder, while a
/// descent that could not derive a single chain has nothing on disk at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma4ColdCaptureSelection<C, K> {
    /// Capture here. `skipped_persisted` deeper candidates were stepped past
    /// because the tier already holds them.
    Capture {
        candidate: C,
        key: K,
        skipped_persisted: usize,
    },
    /// Every candidate whose chain derives is already on disk.
    AllPersisted { skipped_persisted: usize },
    /// Not one candidate named a chain, so nothing was ever written at any of
    /// them to skip.
    NoChainDerived,
}

/// The deepest capture candidate the cold tier does not already hold.
///
/// The DECISION half of `Gemma4Inner::capture_gemma4_sliding_cold_sidecar`'s
/// descent, split out for the reason the file splits every such decision: the
/// body around it needs a loaded checkpoint, a paged adapter and an open cold
/// root, and this rule needs none of those.
///
/// Stopping at the first `Persisted` — what this used to do inline — makes the
/// on-disk state ABSORBING. The next turn on the same prompt recomputes the
/// same key, sees it present again, and exits before it can try anything
/// shallower; a root that once acquired an unreachable object keeps it forever.
/// It also stalls the anchor-rung ladder at its top, when the whole point of the
/// ladder is to give a lagging K/V chain a SHALLOW boundary to reconcile down
/// to. Only one candidate is ever captured per turn either way — the skips cost
/// an index probe each.
///
/// The return is [`Gemma4ColdCaptureSelection`] rather than an
/// `(Option<_>, usize)` pair so that an all-`Underivable` descent cannot reach
/// the caller wearing an already-persisted count of zero: that pair shape is
/// what let the capture record `already_persisted` for a tier that holds
/// nothing.
fn gemma4_select_cold_capture_candidate<C, K>(
    candidates: impl IntoIterator<Item = C>,
    mut probe: impl FnMut(&C) -> Gemma4ColdCaptureProbe<K>,
) -> Gemma4ColdCaptureSelection<C, K> {
    let mut skipped_persisted = 0usize;
    for candidate in candidates {
        match probe(&candidate) {
            Gemma4ColdCaptureProbe::Missing(key) => {
                return Gemma4ColdCaptureSelection::Capture {
                    candidate,
                    key,
                    skipped_persisted,
                };
            }
            Gemma4ColdCaptureProbe::Persisted => skipped_persisted += 1,
            Gemma4ColdCaptureProbe::Underivable => {}
        }
    }
    if skipped_persisted == 0 {
        return Gemma4ColdCaptureSelection::NoChainDerived;
    }
    Gemma4ColdCaptureSelection::AllPersisted { skipped_persisted }
}

/// The `SlidingWindow`-group chain key that names the sidecar at `boundary`.
///
/// `None` when the chain cannot be derived — `boundary` is not a whole number of
/// blocks of `request_tokens`, or a block has no `extra_keys` — which is the
/// same break-at-the-first-underivable-block rule the restore's
/// `deepest_backed_boundary` applies, so the two sides agree on which
/// boundaries exist at all.
fn gemma4_sliding_cold_sidecar_chain_key(
    fingerprint: mlx_paged_attn::ColdCacheFingerprint,
    request_tokens: &[u32],
    extra_keys_per_block: &[Vec<u64>],
    block_size: u32,
    boundary: u32,
) -> Option<mlx_paged_attn::ColdCacheKey> {
    if block_size == 0 {
        return None;
    }
    let blocks = boundary as usize / block_size as usize;
    let mut parent: Option<mlx_paged_attn::ColdCacheKey> = None;
    for index in 0..blocks {
        let extra_keys = extra_keys_per_block.get(index)?;
        let tokens =
            request_tokens.get(index * block_size as usize..(index + 1) * block_size as usize)?;
        parent = Some(mlx_paged_attn::ColdCacheKey::chain(
            mlx_paged_attn::ColdGroup::SlidingWindow,
            fingerprint,
            parent,
            tokens,
            extra_keys,
            0,
            index,
        ));
    }
    parent
}

/// The deepest block boundary a later restore of a `prompt_len`-token prompt
/// can ever probe.
///
/// The whole reason this is not just "the last full block of the prompt": the
/// two sides of the cold tier measure different sequences.
///
/// ```text
///   capture (finalize)                  restore (a later turn on this prompt)
///   request_tokens = prompt + generated lookup = prompt[..prompt_len - 1]
///   ceiling = chain_blocks * bs         full_blocks = (prompt_len - 1) / bs
///   anchors the DEEPEST candidate       probes counts full_blocks .. 1
/// ```
///
/// `prompt_len - 1` is vLLM's `max_cache_hit_tokens` rule
/// (`PagedKVCacheAdapter::find_cached_prefix_per_block_with_max_tokens`): a
/// prefill needs at least one suffix token to forward, so the lookup never sees
/// the last prompt token. `ColdTierWalk::deepest_backed_boundary` therefore
/// enumerates `(prompt_len - 1) / bs` blocks and the deepest boundary it can
/// name is that count times the block size — one block SHALLOWER than the
/// prompt's own end whenever `prompt_len` is an exact multiple of `bs`.
///
/// A sidecar anchored past this line is unreachable by construction: nothing
/// the restore derives ever spells its key. It is also self-locking, because
/// the next capture recomputes the same key, `contains_in` reports it present,
/// and the capture returns without trying anything shallower.
///
/// Same rule, same reason as `qwen3_5::paged_forward::gdn_checkpoint_target`,
/// which the GDN ladder has always used; gemma4's sliding prompt boundary is
/// the one publisher that rounded the other way.
fn gemma4_cold_restore_reachable_boundary(prompt_len: u32, block_size: u32) -> u32 {
    if block_size == 0 {
        return 0;
    }
    prompt_len.saturating_sub(1) / block_size * block_size
}

/// How many whole blocks the cold sidecar capture may anchor within, given the
/// three facts it has at finalize.
///
/// ```text
///   cold_captured_blocks  how far the persisted K/V chain reached
///   request_tokens_len    prompt + everything generated
///   prompt_len            where the PROMPT ended
/// ```
///
/// The first two were always here; the third is the one that makes the answer
/// reachable. A sidecar is selected by a restore that derives its key from
/// `prompt[..prompt_len - 1]`, so a boundary past
/// [`gemma4_cold_restore_reachable_boundary`] is one nothing on the read side
/// can spell — see that function.
///
/// The `prompt_len` bound applies to the WHOLE capture, not only to the aligned
/// prompt boundary that motivated it, and that is deliberate. It also drops the
/// candidates the decode published inside the generated region, which turn N+1
/// of a growing conversation really could name. One sidecar is written per
/// turn, so this is a priority rule, and it spends that write on the deepest
/// boundary a restore of THIS prompt can name — the replay a cold tier exists
/// for, and the case that measured zero reuse — rather than on a deeper
/// boundary that pays off only if the conversation continues with exactly these
/// tokens. The give-up is one turn deep: turn N+1's own ceiling covers
/// everything turn N discarded. See
/// `the_capture_ceiling_gives_up_this_turns_generated_region_and_the_next_turn_covers_it`.
///
/// A free function, and the reason is the same one
/// [`gemma4_sliding_decode_boundary_plan`] gives: this is the DECISION, its
/// caller contributes three adapter reads, and as a method on `Gemma4Inner` it
/// would be reachable only from a loaded checkpoint on a GPU, i.e. from no test
/// at all.
fn gemma4_sliding_cold_capture_ceiling_blocks(
    cold_captured_blocks: u32,
    request_tokens_len: usize,
    prompt_len: u32,
    block_size: u32,
) -> usize {
    if block_size == 0 {
        return 0;
    }
    let full_blocks = request_tokens_len / block_size as usize;
    let reachable_blocks =
        (gemma4_cold_restore_reachable_boundary(prompt_len, block_size) / block_size) as usize;
    (cold_captured_blocks as usize)
        .min(full_blocks)
        .min(reachable_blocks)
}

/// The cold-restore tail boundary a prefill over `prompt_len` tokens publishes,
/// or `None`.
///
/// [`gemma4_cold_restore_reachable_boundary`] on a persist turn over a
/// BLOCK-ALIGNED prompt, nothing otherwise, and the alignment screen is as
/// load-bearing as the persistence one.
///
/// The reachable boundary equals the prompt boundary
/// (`prompt_checkpoint_boundary_len`) except when `prompt_len` is an exact
/// multiple of `block_size`, where it is one block shallower — and that one
/// block is the whole defect. On an aligned prompt the prompt boundary is the
/// only tail checkpoint the turn has, it sits past `max_cache_hit_tokens =
/// prompt_len - 1`, and the capture anchors a sidecar there that no restore can
/// ever ask for. Everywhere else — 15 prompt lengths in 16 — the two COINCIDE,
/// `maybe_remember_gemma4_sliding_prompt_boundary_checkpoint` already
/// snapshots that offset (`gemma4_split_body_chunk_plan_at_position` splits the
/// plan there, so a chunk always ends on it whenever the tail is in range), and
/// `find_gemma4_sliding_capture_checkpoints` dedups the pair by boundary. A
/// second snapshot of one offset is one sliding window of pure cost with no
/// reader.
///
/// The chunk walk cannot make that call for us:
/// `gemma4_chunk_cold_restore_tail`'s `already_published` argument is the
/// chunk's boundary list AFTER the prompt boundary has been retained out of it,
/// so its containment test is blind to precisely the coinciding case.
///
/// Two properties this must keep, and both are why it is a function rather than
/// an expression at the prefill call site (the `want_ladder` hard-coded `false`
/// that reverted decode to cadence-only is the precedent — see
/// [`gemma4_sliding_decode_boundary_plan`]):
///
///  * gated on `caps.wants_ladder()`, like every other publisher here. A
///    persistence-OFF turn must snapshot exactly what it snapshotted before the
///    cold tier existed;
///  * the boundary is CAPTURED from the temporal K/V view a chunk already
///    produced, never reached by splitting the chunk plan. Splitting would
///    change every downstream GEMM's `M` and with it the tokens the turn emits
///    — on the persist side of a parity gate that compares persist against
///    no-persist, that is a failure either way.
fn gemma4_cold_restore_tail_publish(
    prompt_len: u32,
    block_size: u32,
    caps: Gemma4SlidingRetentionCaps,
) -> Option<u32> {
    if !caps.wants_ladder() || block_size == 0 || !prompt_len.is_multiple_of(block_size) {
        return None;
    }
    let boundary = gemma4_cold_restore_reachable_boundary(prompt_len, block_size);
    (boundary > 0).then_some(boundary)
}

/// Where, if anywhere, one compute chunk captures the cold-restore tail.
///
/// `already_published` is what this chunk already snapshots — the decode cadence
/// union this turn's anchor rungs, minus the prompt boundary. The tail is taken
/// IN ADDITION to that set and never INSTEAD of a member of it, which is what
/// keeps it inert for everything but the cold capture: the boundaries that reach
/// `remember_gemma4_sliding_captured_prefix_checkpoint`'s retained store, and so
/// the checkpoint a later warm turn resumes from, stay exactly the set a
/// persistence-OFF turn produces. Only the extra snapshot, parked in a singleton
/// outside the deque, is new.
///
/// `(start, end]` matches `gemma4_sliding_chunk_checkpoint_boundaries`'s rung
/// filter: a boundary at or below where this chunk began was already passed.
fn gemma4_chunk_cold_restore_tail(
    tail: Option<u32>,
    chunk_start: u32,
    chunk_end: u32,
    already_published: &[u32],
) -> Option<u32> {
    tail.filter(|boundary| {
        *boundary > chunk_start && *boundary <= chunk_end && !already_published.contains(boundary)
    })
}

fn gemma4_sliding_checkpoint_boundaries_crossed(
    start_offset: u32,
    end_offset: u32,
    checkpoint_interval: u32,
) -> Vec<u32> {
    if checkpoint_interval == 0 || end_offset <= start_offset {
        return Vec::new();
    }
    let Some(mut boundary) = start_offset
        .checked_div(checkpoint_interval)
        .and_then(|bucket| bucket.checked_add(1))
        .and_then(|bucket| bucket.checked_mul(checkpoint_interval))
    else {
        return Vec::new();
    };
    let mut boundaries = Vec::new();
    while boundary <= end_offset {
        boundaries.push(boundary);
        let Some(next) = boundary.checked_add(checkpoint_interval) else {
            break;
        };
        boundary = next;
    }
    boundaries
}

/// Boundaries this compute chunk snapshots at, ascending and deduped.
///
/// ```text
///   PreLadder  ->  gemma4_sliding_checkpoint_boundaries_crossed
///                  (the decode cadence, unchanged)
///   Ladder     ->  that UNION `caps.anchors` inside the chunk
/// ```
///
/// The `PreLadder` arm is the compatibility contract, and the reason this is one
/// function rather than an `if` at the call site. Capturing an extra boundary
/// is numerically transparent — `RotatingKVCache::snapshot_from_attention_view`
/// slices the attention view the chunk already produced, and the chunk plan is
/// NOT split at a rung — but the extra entries it puts in the store change
/// which checkpoint a later warm turn resumes from, and that is observable in
/// the emitted tokens. A persistence-OFF request must publish exactly what it
/// published before anchor rungs existed.
fn gemma4_sliding_chunk_checkpoint_boundaries(
    start_offset: u32,
    end_offset: u32,
    checkpoint_interval: u32,
    caps: Gemma4SlidingRetentionCaps,
) -> Vec<u32> {
    let mut boundaries =
        gemma4_sliding_checkpoint_boundaries_crossed(start_offset, end_offset, checkpoint_interval);
    if !caps.wants_ladder() {
        return boundaries;
    }
    boundaries.extend(
        caps.anchors
            .as_slice()
            .iter()
            .copied()
            .filter(|rung| *rung > start_offset && *rung <= end_offset),
    );
    // `prepare_sliding_checkpoint_capture` rejects offsets that are not
    // strictly increasing, so the union must be normalized here, not hoped for.
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
}

/// Whether `ancestor` describes a strict token prefix of `descendant` under the
/// same block size.
///
/// Used only to pick an eviction VICTIM, never to authorize a restore: every
/// lookup path re-derives `final_block_hash` before it installs anything, so a
/// token-prefix match that is not a real cache-identity match can at worst
/// retain a useless entry one push longer.
fn gemma4_sliding_checkpoint_is_strict_ancestor(
    ancestor: &Gemma4SlidingPrefixCheckpoint,
    descendant: &Gemma4SlidingPrefixCheckpoint,
) -> bool {
    ancestor.block_size == descendant.block_size
        && ancestor.tokens.len() < descendant.tokens.len()
        && descendant.tokens.starts_with(&ancestor.tokens)
}

/// Index the ladder policy evicts, given the store is over its limit (or over
/// its byte budget).
///
/// ```text
///   1. oldest non-anchor                                  (excluding the entry just pushed)
///   2. oldest anchor that is NOT an ancestor of the newest (excluding it too)  <- lineage switch
///   3. DEEPEST remaining anchor                           (excluding it too)
///   4. oldest non-image-protected   (the pre-ladder rule, as a floor)
///   5. index 0
/// ```
///
/// Steps 1-3 skip the last slot so a push can never evict itself while an older
/// entry is eligible; step 4 does not, because that is exactly what the
/// pre-ladder rule does and it is the floor this must never fall below.
///
/// Step 3 is what stops this function from undoing the ladder. Once steps 1 and
/// 2 come up empty, every eligible entry below the newest is an anchor that IS
/// an ancestor of the newest — all useful, one has to go — and the pre-ladder
/// floor at step 4 is `position(|c| !protected)`, i.e. the SHALLOWEST entry,
/// which is precisely the rung a lagging persisted chain can reach. gemma4's
/// chain advances ~34 blocks (~544 tokens) a turn, so the shallow rungs are the
/// only reachable ones for the first several turns and the deep ones are dead
/// weight until then; giving up the deepest costs the least. Without step 3 the
/// byte loop below re-creates the "born, then evicted" failure the anchor flag
/// exists to prevent, inside the fix for the byte budget.
///
/// Reachable, and not only through the byte loop: two image-protected prompt
/// boundaries (`GEMMA4_IMAGE_PREFIX_CHECKPOINT_LIMIT`) are never eligible, so a
/// store of `{image, image, rung, rung, rung, deep}` — a VLM turn followed by a
/// fresh text turn, which clears `cached_paged_image_token_positions` but leaves
/// the protected entries in the store — has nothing for steps 1 or 2 to take.
fn gemma4_sliding_ladder_victim(checkpoints: &VecDeque<Gemma4SlidingPrefixCheckpoint>) -> usize {
    let Some(newest) = checkpoints.back() else {
        return 0;
    };
    let head = checkpoints.len().saturating_sub(1);
    let eligible = |index: usize| -> bool {
        checkpoints
            .get(index)
            .is_some_and(|checkpoint| !checkpoint.protected_image_prompt_boundary)
    };
    (0..head)
        .find(|&index| {
            eligible(index) && checkpoints.get(index).is_some_and(|c| !c.cold_anchor_rung)
        })
        .or_else(|| {
            (0..head).find(|&index| {
                eligible(index)
                    && checkpoints
                        .get(index)
                        .is_some_and(|c| !gemma4_sliding_checkpoint_is_strict_ancestor(c, newest))
            })
        })
        .or_else(|| {
            (0..head)
                .filter(|&index| eligible(index))
                .max_by_key(|&index| {
                    checkpoints
                        .get(index)
                        .map(|checkpoint| checkpoint.prefix_len)
                        .unwrap_or(0)
                })
        })
        .or_else(|| {
            checkpoints
                .iter()
                .position(|checkpoint| !checkpoint.protected_image_prompt_boundary)
        })
        .unwrap_or(0)
}

fn trim_gemma4_sliding_prefix_checkpoints(
    checkpoints: &mut VecDeque<Gemma4SlidingPrefixCheckpoint>,
    caps: Gemma4SlidingRetentionCaps,
    trace_enabled: bool,
) {
    let limit = caps.limit;
    if limit == 0 {
        return;
    }
    let mut evicted = 0usize;
    let mut first_prefix_len = None;
    let mut last_prefix_len = None;

    while checkpoints
        .iter()
        .filter(|checkpoint| checkpoint.protected_image_prompt_boundary)
        .count()
        > GEMMA4_IMAGE_PREFIX_CHECKPOINT_LIMIT
    {
        let Some(index) = checkpoints
            .iter()
            .position(|checkpoint| checkpoint.protected_image_prompt_boundary)
        else {
            break;
        };
        if let Some(checkpoint) = checkpoints.remove(index) {
            first_prefix_len.get_or_insert(checkpoint.prefix_len);
            last_prefix_len = Some(checkpoint.prefix_len);
            evicted += 1;
        }
    }
    while checkpoints.len() > limit {
        // Decode/text checkpoints are reproducible from token embeddings. Keep
        // the two most recent image-aware prompt boundaries preferentially so
        // an A -> B -> A branch can restore A without retaining every image.
        let removable = match caps.policy {
            Gemma4SlidingRetentionPolicy::PreLadder => checkpoints
                .iter()
                .position(|checkpoint| !checkpoint.protected_image_prompt_boundary)
                .unwrap_or(0),
            Gemma4SlidingRetentionPolicy::Ladder => gemma4_sliding_ladder_victim(checkpoints),
        };
        if let Some(checkpoint) = checkpoints.remove(removable) {
            first_prefix_len.get_or_insert(checkpoint.prefix_len);
            last_prefix_len = Some(checkpoint.prefix_len);
            evicted += 1;
        }
    }
    // The count above is DERIVED from a byte budget, on the assumption that the
    // slots the ladder added hold cheap sub-window rungs
    // (`gemma4_sliding_cold_anchor_rungs` prices a 64-token rung at 41.9 MB, not
    // 671.1 MB — which is the only reason a fourth rung fit). Nothing forces the
    // retained set to BE that mix: once the cursor is past one window every
    // retained entry is a full window, and `base_limit + anchors.len` = 6 of
    // those is 4026 MB against a declared 3072 MB ceiling. So the budget has to
    // be enforced where it is actually spent, in bytes, over the entries that
    // are actually here.
    //
    // Ladder-only, and after the count loop rather than instead of it: a
    // persistence-OFF turn must evict exactly what it evicted before the ladder
    // existed, and `PreLadder` retains at most `base_limit` full windows, which
    // is what the budget reserved for it in the first place.
    if caps.wants_ladder() {
        while checkpoints.len() > 1
            && caps.bytes.total(checkpoints.iter()) > GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES
        {
            let removable = gemma4_sliding_ladder_victim(checkpoints);
            let Some(checkpoint) = checkpoints.remove(removable) else {
                break;
            };
            first_prefix_len.get_or_insert(checkpoint.prefix_len);
            last_prefix_len = Some(checkpoint.prefix_len);
            evicted += 1;
        }
    }
    // `retained_bytes` is emitted on BOTH arms, deliberately. It is the one
    // number that says whether an eviction was a count decision or a byte
    // decision, and `caps.bytes` is populated on the `PreLadder` arm too (see
    // `Gemma4SlidingRetentionCaps::bytes`), so it is just as meaningful there —
    // a persistence-OFF turn showing 5120 MB retained under a 3072 MB ladder
    // ceiling is exactly the fact an operator reading this line needs. Hiding it
    // behind `wants_ladder()` would make the arm that does NOT enforce the
    // budget the arm you cannot see the budget for. It costs one integer fold
    // over at most `limit` entries, and only when `MLX_TRACE` is on AND
    // something was actually evicted. Nothing in the repo parses this line
    // (grep: it appears only here), so widening it is a diagnostic change, not
    // an interface change.
    if trace_enabled && evicted > 0 {
        write_inference_trace(format_args!(
            "[MLX_TRACE] gemma4 sliding_prefix_checkpoint_evict evicted={} limit={} policy={:?} remaining={} retained_bytes={} first_prefix_tokens={} last_prefix_tokens={} retained={:?}",
            evicted,
            limit,
            caps.policy,
            checkpoints.len(),
            caps.bytes.total(checkpoints.iter()),
            first_prefix_len.unwrap_or(0),
            last_prefix_len.unwrap_or(0),
            checkpoints
                .iter()
                .map(|checkpoint| checkpoint.prefix_len)
                .collect::<Vec<_>>()
        ));
    }
}

/// The ONE way an entry enters the sliding-prefix store: derive the anchor
/// flag, replace an identical entry, push, trim to `caps`.
///
/// It takes a [`Gemma4SlidingPrefixCheckpointDraft`] rather than a finished
/// checkpoint so the flag cannot be supplied by a caller. All four publish
/// sites — decode cadence, warm text continuation, prefill capture, prompt
/// boundary — go through here.
fn upsert_gemma4_sliding_prefix_checkpoint(
    checkpoints: &mut VecDeque<Gemma4SlidingPrefixCheckpoint>,
    draft: Gemma4SlidingPrefixCheckpointDraft,
    caps: Gemma4SlidingRetentionCaps,
    trace_enabled: bool,
) {
    let checkpoint = draft.into_checkpoint(caps);
    checkpoints.retain(|existing| {
        !(existing.prefix_len == checkpoint.prefix_len
            && existing.block_size == checkpoint.block_size
            && existing.final_block_hash == checkpoint.final_block_hash
            && existing.tokens == checkpoint.tokens)
    });
    checkpoints.push_back(checkpoint);
    trim_gemma4_sliding_prefix_checkpoints(checkpoints, caps, trace_enabled);
}

fn gemma4_paged_prefill_group_max_chunk() -> u32 {
    let configured_chunk_size = crate::array::paged_prefill_chunk_size();
    if configured_chunk_size > 0 {
        configured_chunk_size as u32
    } else {
        GEMMA4_PREFILL_STEP_SIZE as u32
    }
}

fn gemma4_paged_prefill_body_chunk_size(configured_chunk_size: i32, body_tokens: usize) -> usize {
    if configured_chunk_size > 0 {
        configured_chunk_size as usize
    } else {
        body_tokens.min(GEMMA4_PREFILL_STEP_SIZE as usize)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Gemma4PagedPrefillBodyChunk {
    start: usize,
    len: usize,
    first_position: u32,
    capped_by_v2_aux_limit: bool,
}

fn gemma4_coalesce_single_token_restore_chunks(chunks: &mut Vec<Gemma4PagedPrefillBodyChunk>) {
    if chunks.len() < 2 || chunks.iter().all(|chunk| chunk.len > 1) {
        return;
    }

    let mut merged = Vec::with_capacity(chunks.len());
    let mut idx = 0usize;
    while idx < chunks.len() {
        let mut chunk = chunks[idx].clone();
        if chunk.len == 1 && idx + 1 < chunks.len() {
            let next = &chunks[idx + 1];
            chunk.len += next.len;
            chunk.capped_by_v2_aux_limit |= next.capped_by_v2_aux_limit;
            merged.push(chunk);
            idx += 2;
            continue;
        }
        if chunk.len == 1
            && let Some(previous) = merged.last_mut()
        {
            previous.len += 1;
            previous.capped_by_v2_aux_limit |= chunk.capped_by_v2_aux_limit;
        } else {
            merged.push(chunk);
        }
        idx += 1;
    }
    *chunks = merged;
}

fn gemma4_split_body_chunk_plan_at_position(
    chunks: &mut Vec<Gemma4PagedPrefillBodyChunk>,
    boundary_position: u32,
) {
    if boundary_position == 0 {
        return;
    }

    let Some(idx) = chunks.iter().position(|chunk| {
        let first = chunk.first_position as u64;
        let end = first + chunk.len as u64;
        boundary_position as u64 > first && (boundary_position as u64) < end
    }) else {
        return;
    };

    let chunk = &mut chunks[idx];
    let before_len = (boundary_position - chunk.first_position) as usize;
    let after_len = chunk.len - before_len;
    let after_chunk = Gemma4PagedPrefillBodyChunk {
        start: chunk.start + before_len,
        len: after_len,
        first_position: boundary_position,
        capped_by_v2_aux_limit: chunk.capped_by_v2_aux_limit,
    };
    chunk.len = before_len;
    chunks.insert(idx + 1, after_chunk);
}

fn gemma4_paged_prefill_chunk_route_is_aux_safe(
    num_new_tokens: usize,
    first_position: u32,
    num_query_heads: u32,
    num_kv_heads: u32,
    head_size: u32,
    route_policy: Gemma4PagedPrefillRoutePolicy,
) -> bool {
    if num_new_tokens == 0 || num_query_heads == 0 || head_size == 0 {
        return false;
    }
    let Ok(num_new_tokens) = u32::try_from(num_new_tokens) else {
        return false;
    };
    let Some(total_context) = first_position.checked_add(num_new_tokens) else {
        return false;
    };
    let Some(layout) = gemma4_paged_prefill_v2_layout_for_chunk(
        route_policy,
        num_new_tokens,
        total_context,
        num_query_heads,
        num_kv_heads,
        head_size,
    ) else {
        // SDPA and host-read routes do not allocate V2 auxiliary buffers.
        return true;
    };
    paged_attention_v2_aux_fits(
        layout,
        num_new_tokens,
        num_query_heads,
        num_kv_heads,
        total_context,
        head_size,
    )
}

fn gemma4_paged_prefill_aux_limited_chunk_size(
    configured_chunk_size: i32,
    remaining_tokens: usize,
    first_position: u32,
    num_query_heads: u32,
    num_kv_heads: u32,
    head_size: u32,
    route_policy: Gemma4PagedPrefillRoutePolicy,
) -> (usize, bool) {
    let base = gemma4_paged_prefill_body_chunk_size(configured_chunk_size, remaining_tokens)
        .min(remaining_tokens)
        .max(1);

    if gemma4_paged_prefill_chunk_route_is_aux_safe(
        base,
        first_position,
        num_query_heads,
        num_kv_heads,
        head_size,
        route_policy,
    ) {
        return (base, false);
    }

    let mut lo = 1usize;
    let mut hi = base.saturating_sub(1).max(1);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if gemma4_paged_prefill_chunk_route_is_aux_safe(
            mid,
            first_position,
            num_query_heads,
            num_kv_heads,
            head_size,
            route_policy,
        ) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    (lo.max(1), true)
}

fn gemma4_paged_prefill_body_chunk_plan(
    configured_chunk_size: i32,
    body_tokens: usize,
    first_position: u32,
    num_query_heads: u32,
    num_kv_heads: u32,
    head_size: u32,
    route_policy: Gemma4PagedPrefillRoutePolicy,
) -> Result<Vec<Gemma4PagedPrefillBodyChunk>> {
    gemma4_paged_prefill_body_chunk_plan_inner(
        configured_chunk_size,
        body_tokens,
        first_position,
        num_query_heads,
        num_kv_heads,
        head_size,
        route_policy,
    )
}

fn gemma4_paged_prefill_body_chunk_plan_inner(
    configured_chunk_size: i32,
    body_tokens: usize,
    first_position: u32,
    num_query_heads: u32,
    num_kv_heads: u32,
    head_size: u32,
    route_policy: Gemma4PagedPrefillRoutePolicy,
) -> Result<Vec<Gemma4PagedPrefillBodyChunk>> {
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut position = first_position;
    while start < body_tokens {
        let remaining = body_tokens - start;
        let (len, capped_by_v2_aux_limit) = gemma4_paged_prefill_aux_limited_chunk_size(
            configured_chunk_size,
            remaining,
            position,
            num_query_heads,
            num_kv_heads,
            head_size,
            route_policy,
        );
        if len == 0 {
            return Err(Error::from_reason(
                "Gemma4 paged prefill dynamic chunking produced an empty chunk",
            ));
        }
        chunks.push(Gemma4PagedPrefillBodyChunk {
            start,
            len,
            first_position: position,
            capped_by_v2_aux_limit,
        });
        start = start
            .checked_add(len)
            .ok_or_else(|| Error::from_reason("Gemma4 paged prefill chunk start overflow"))?;
        position = position
            .checked_add(len as u32)
            .ok_or_else(|| Error::from_reason("Gemma4 paged prefill token position overflow"))?;
    }
    Ok(chunks)
}

/// Evaluate all Gemma4 cache arrays to materialize them on GPU.
/// Must be called between prefill chunks to break lazy dependency chains.
pub(crate) fn eval_gemma4_caches(caches: &[Gemma4LayerCache]) -> Result<()> {
    let mut arrays: Vec<&MxArray> = Vec::new();
    for cache in caches {
        cache.collect_cache_arrays(&mut arrays);
    }
    if !arrays.is_empty() {
        let trace_enabled = inference_trace_enabled();
        let trace_start = trace_enabled.then(std::time::Instant::now);
        MxArray::eval_arrays(&arrays)?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 eval_caches arrays={} elapsed_ms={:.1}",
                arrays.len(),
                trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
    }
    Ok(())
}

/// Chunked prefill: process all tokens EXCEPT the last one.
///
/// Matches mlx-lm generate.py generate_step prefill pattern:
/// - The prefill loop processes tokens [0:N-1] (all but the last)
/// - The last token is processed by the caller via `forward_inner`, which
///   also produces the logits used to sample the first output token
///
/// This is CRITICAL for correctness: SDPA computes slightly different numerical
/// results for multi-token causal attention vs single-token attention with cached
/// K/V. These small differences compound through layers, causing divergent logits
/// if the last prompt token is processed in the same batch as the rest.
///
/// 1. Embed ALL tokens once upfront (including PLE if enabled)
/// 2. Run only the transformer body for each chunk (no lm_head)
/// 3. Stop BEFORE the last token — the caller handles it via forward_inner
fn prefill_body_gemma4(
    prompt: &MxArray,
    embedding: &Embedding,
    layers: &[Gemma4DecoderLayer],
    caches: &mut [Gemma4LayerCache],
    final_norm: &RMSNorm,
    ple: Option<&PleComponents>,
    config: &Gemma4Config,
    mut tap: Option<&mut DsparkTap<'_>>,
) -> Result<()> {
    let total_len = prompt.shape_at(1)?;

    // Must have at least 2 tokens (1 for prefill, 1 for caller to process)
    if total_len <= 1 {
        return Ok(());
    }

    // Process tokens [0:N-1] — leave last token for the caller
    let prefill_len = total_len - 1;

    // Step 1: Embed tokens [0:N-1]
    let prefill_ids = prompt.slice_axis(1, 0, prefill_len)?;
    let all_embeds = {
        let emb = embedding.forward(&prefill_ids)?;
        emb.mul_scalar((config.hidden_size as f64).sqrt())?
    };

    // Step 2: Compute PLE for prefill tokens (if enabled)
    let all_ple: Option<MxArray> = if let Some(ple) = ple {
        Some(compute_ple(&prefill_ids, &all_embeds, ple, prefill_len)?)
    } else {
        None
    };

    let mut offset: i64 = 0;

    // Process in chunks
    while prefill_len - offset > GEMMA4_PREFILL_STEP_SIZE {
        let chunk_embeds = all_embeds.slice_axis(1, offset, offset + GEMMA4_PREFILL_STEP_SIZE)?;
        let chunk_ple = all_ple
            .as_ref()
            .map(|p| p.slice_axis(1, offset, offset + GEMMA4_PREFILL_STEP_SIZE))
            .transpose()?;

        let _hidden = forward_body(
            None,
            Some(chunk_embeds),
            embedding,
            layers,
            caches,
            final_norm,
            ple,
            chunk_ple.as_ref(),
            config,
            tap.as_deref_mut(),
        )?;
        eval_gemma4_caches(caches)?;
        crate::array::clear_cache();
        offset += GEMMA4_PREFILL_STEP_SIZE;
    }

    // Final chunk (still body only — no lm_head needed)
    if offset < prefill_len {
        let remaining_embeds = all_embeds.slice_axis(1, offset, prefill_len)?;
        let remaining_ple = all_ple
            .as_ref()
            .map(|p| p.slice_axis(1, offset, prefill_len))
            .transpose()?;

        let _hidden = forward_body(
            None,
            Some(remaining_embeds),
            embedding,
            layers,
            caches,
            final_norm,
            ple,
            remaining_ple.as_ref(),
            config,
            tap,
        )?;
    }

    Ok(())
}

fn create_sliding_mask(seq_len: i64, offset: i32, window_size: i64) -> Result<MxArray> {
    let total_len = seq_len + offset as i64;
    let rows = MxArray::arange(offset as f64, (offset as i64 + seq_len) as f64, None, None)?;
    let cols = MxArray::arange(0.0, total_len as f64, None, None)?;
    let rows = rows.reshape(&[seq_len, 1])?;
    let cols = cols.reshape(&[1, total_len])?;
    let distance = rows.sub(&cols)?;

    let zero = MxArray::scalar_int(0)?;
    let window = MxArray::scalar_int(window_size as i32)?;
    let causal = distance.greater_equal(&zero)?;
    let in_window = distance.less(&window)?;
    let valid = causal.logical_and(&in_window)?;

    // MLX bool mask semantics are `true = keep`. Returning bool here keeps the
    // mask dtype independent of Gemma4's BF16 residual stream; an additive
    // float32 mask is rejected by `mx.fast.scaled_dot_product_attention` for
    // BF16 Q/K/V because it would promote the output away from BF16.
    valid.reshape(&[1, 1, seq_len, total_len])
}

fn sliding_mask_offset_for_chunk(seq_len: i64, cache_offset: i32, window_size: i64) -> Option<i32> {
    if seq_len <= 1 || window_size <= 0 {
        return None;
    }

    let prior_len = (cache_offset.max(0) as i64).min(window_size);
    if prior_len + seq_len > window_size {
        Some(prior_len as i32)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Vision helpers
// ---------------------------------------------------------------------------

/// Expand image tokens in a token sequence.
///
/// The chat template inserts a single `<|image|>` per image. This function
/// replaces each occurrence with: `boi_token + image_token × num_soft_tokens + eoi_token`.
///
/// The placeholder count must exactly match the processed image count. Prompt
/// structure belongs to the checkpoint template, so this processor never
/// inserts a fallback span at an invented position.
fn expand_image_tokens(
    tokens: &[u32],
    processed_images: &[super::image_processor::ProcessedGemma4Image],
    image_token_id: u32,
    boi_token_id: u32,
    eoi_token_id: u32,
) -> Result<Vec<u32>> {
    let image_count = tokens.iter().filter(|&&t| t == image_token_id).count();
    if image_count != processed_images.len() {
        return Err(Error::from_reason(format!(
            "expand_image_tokens: {} image placeholder(s) but {} image(s) supplied",
            image_count,
            processed_images.len()
        )));
    }

    // Replace each <|image|> with the expanded BOI + N×image_token + EOI sequence
    let mut result = Vec::with_capacity(tokens.len() * 2);
    let mut img_idx = 0;
    for &t in tokens {
        if t == image_token_id && img_idx < processed_images.len() {
            let num_soft = processed_images[img_idx].num_soft_tokens;
            result.push(boi_token_id);
            for _ in 0..num_soft {
                result.push(image_token_id);
            }
            result.push(eoi_token_id);
            img_idx += 1;
        } else {
            result.push(t);
        }
    }
    Ok(result)
}

/// masked_scatter: replace positions where mask=true with values from source.
///
/// Matches Python: `mx.where(mask_flat, aligned, input_flat).reshape(input.shape)`
/// where `aligned = source.flatten()[(cumsum(mask_flat) - 1) % source.size]`
fn masked_scatter(input: &MxArray, mask: &MxArray, source: &MxArray) -> Result<MxArray> {
    let input_shape = input.shape()?;
    let mask_flat = mask.reshape(&[-1])?.astype(DType::Int32)?;
    let input_flat = input.reshape(&[-1])?;

    let source_flat = source.reshape(&[-1])?;
    let source_size = source_flat.shape_at(0)?;

    // cumsum of mask gives 1-based indices into source; subtract 1 for 0-based
    let indices = mask_flat.cumsum(0)?.sub(&MxArray::scalar_int(1)?)?;
    // Modulo source_size to handle wrap-around safely
    let source_size_arr = MxArray::scalar_int(source_size as i32)?;
    let safe_indices = indices.remainder(&source_size_arr)?;
    let aligned = source_flat.take(&safe_indices, 0)?;

    // where mask=1 use aligned (source), else keep input
    let result = mask_flat.where_(&aligned, &input_flat)?;
    result.reshape(&input_shape)
}

/// Reports whether `tokens` carry an image or audio placeholder id.
///
/// Used to decide whether a paged text turn may run a content-address prefix
/// lookup. Per-block prefix-cache hashes cover only token ids, not media
/// feature K/V, so a prompt that still holds media placeholders must skip the
/// lookup: otherwise a continue-turn-failure fallback could match the
/// token-only hash of media blocks registered by another session and reuse
/// that session's stale media K/V.
fn prompt_holds_media_placeholders(
    tokens: &[u32],
    image_token_id: u32,
    audio_token_id: u32,
) -> bool {
    tokens.contains(&image_token_id) || tokens.contains(&audio_token_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::plan::{TurnPath, TurnPlan, TurnRequest};
    use crate::models::gemma4::output_parser::{StreamSegment, parse_gemma4_output};

    #[test]
    fn prompt_holds_media_placeholders_detects_image_audio_and_text() {
        let image_token_id = 258880u32;
        let audio_token_id = 258881u32;

        let image_prompt = [1u32, 2, image_token_id, 3];
        assert!(prompt_holds_media_placeholders(
            &image_prompt,
            image_token_id,
            audio_token_id
        ));

        let audio_prompt = [4u32, audio_token_id, 5];
        assert!(prompt_holds_media_placeholders(
            &audio_prompt,
            image_token_id,
            audio_token_id
        ));

        let text_prompt = [6u32, 7, 8, 9];
        assert!(!prompt_holds_media_placeholders(
            &text_prompt,
            image_token_id,
            audio_token_id
        ));
    }

    #[test]
    fn gemma4_media_plan_separates_availability_from_backend_validation() {
        let text_only_flat = gemma4_media_plan(false, false, false);
        assert_eq!(text_only_flat.available, MediaCapabilities::NONE);
        assert_eq!(text_only_flat.backend_validated, MediaCapabilities::IMAGES);

        let image_flat = gemma4_media_plan(true, false, false);
        assert_eq!(image_flat.available, MediaCapabilities::NONE);
        assert_eq!(image_flat.backend_validated, MediaCapabilities::IMAGES);

        let audio_flat = gemma4_media_plan(false, true, false);
        assert_eq!(audio_flat.available, MediaCapabilities::NONE);
        assert_eq!(
            audio_flat.backend_validated,
            MediaCapabilities::IMAGES_AND_AUDIO
        );

        let media_paged = gemma4_media_plan(true, true, true);
        assert_eq!(media_paged.available, MediaCapabilities::IMAGES_AND_AUDIO);
        assert_eq!(media_paged.backend_validated, MediaCapabilities::NONE);

        let missing_image_components_paged = gemma4_media_plan(false, true, true);
        assert_eq!(
            missing_image_components_paged.available,
            MediaCapabilities {
                images: false,
                audio: true,
            }
        );
        assert_eq!(
            missing_image_components_paged.backend_validated,
            MediaCapabilities::IMAGES
        );
    }

    #[test]
    fn gemma4_image_capability_requires_one_complete_paged_path() {
        assert!(gemma4_image_path_loaded(true, true, true, false, true));
        assert!(gemma4_image_path_loaded(true, true, false, true, true));

        assert!(!gemma4_image_path_loaded(false, true, true, false, true));
        assert!(!gemma4_image_path_loaded(true, false, true, false, true));
        assert!(!gemma4_image_path_loaded(true, true, false, false, true));
        assert!(!gemma4_image_path_loaded(true, true, true, false, false));
    }

    #[test]
    fn gemma4_vlm_checkpoint_publication_is_image_only_and_opt_in() {
        assert!(gemma4_vlm_prefix_checkpoint_eligible(true, false, true));
        assert!(!gemma4_vlm_prefix_checkpoint_eligible(true, false, false));
        assert!(!gemma4_vlm_prefix_checkpoint_eligible(false, true, true));
        assert!(!gemma4_vlm_prefix_checkpoint_eligible(true, true, true));
        assert!(!gemma4_vlm_prefix_checkpoint_eligible(false, false, true));
    }

    #[test]
    fn gemma4_image_lineage_requires_declared_media_context() {
        let history = [1, 2, 3, 4];
        let extended = [1, 2, 3, 4, 5];
        let image_positions = [(1, 0xAAAA)];

        assert!(gemma4_carries_image_lineage(
            MediaCapabilities::IMAGES,
            Some(0xAAAA),
            &image_positions,
            &history,
            &extended,
        ));
        assert!(!gemma4_carries_image_lineage(
            MediaCapabilities::NONE,
            Some(0xAAAA),
            &image_positions,
            &history,
            &extended,
        ));
        assert!(!gemma4_carries_image_lineage(
            MediaCapabilities::IMAGES,
            Some(0xAAAA),
            &image_positions,
            &history,
            &[1, 2, 9],
        ));
    }

    #[test]
    fn gemma4_causal_leading_text_hit_replays_only_before_image() {
        let before_image = gemma4_vlm_prefix_policy(16, Some(32), None);
        assert!(before_image.unified_boundary_safe);
        assert!(!before_image.require_exact_checkpoint);
        assert!(before_image.may_replay_leading_text);

        let at_image_boundary = gemma4_vlm_prefix_policy(32, Some(32), None);
        assert!(!at_image_boundary.require_exact_checkpoint);
        assert!(at_image_boundary.may_replay_leading_text);

        let crosses_image = gemma4_vlm_prefix_policy(48, Some(32), None);
        assert!(crosses_image.require_exact_checkpoint);
        assert!(!crosses_image.may_replay_leading_text);

        let unified_inside_image = gemma4_vlm_prefix_policy(48, Some(32), Some(80));
        assert!(!unified_inside_image.unified_boundary_safe);
        assert!(unified_inside_image.require_exact_checkpoint);
        assert!(!unified_inside_image.may_replay_leading_text);

        let unified_after_image = gemma4_vlm_prefix_policy(80, Some(32), Some(80));
        assert!(unified_after_image.unified_boundary_safe);
        assert!(unified_after_image.require_exact_checkpoint);
        assert!(!unified_after_image.may_replay_leading_text);
    }

    #[test]
    fn gemma4_sliding_cold_capture_context_is_fail_closed_for_media() {
        let image_positions = [(47, 0xAAAA), (32, 0xBBBB), (79, 0xAAAA)];

        assert_eq!(
            Gemma4SlidingColdCaptureContext::text(128, &[]).minimum_safe_boundary(),
            Some(0),
            "the existing text-only capture has no media floor"
        );
        assert_eq!(
            Gemma4SlidingColdCaptureContext::text(128, &image_positions).minimum_safe_boundary(),
            None,
            "a generic text turn carrying image lineage must remain unsupported"
        );
        assert_eq!(
            Gemma4SlidingColdCaptureContext::pure_image(128, &[]).minimum_safe_boundary(),
            None,
            "a pure-image label without image positions must not capture"
        );
        assert_eq!(
            Gemma4SlidingColdCaptureContext::pure_image(128, &image_positions)
                .minimum_safe_boundary(),
            Some(80),
            "the floor must sit after the complete image run even if positions arrive unsorted"
        );
        assert_eq!(
            Gemma4SlidingColdCaptureContext::pure_image(u32::MAX, &[(u32::MAX, 0xAAAA)],)
                .minimum_safe_boundary(),
            None,
            "an unrepresentable exclusive image endpoint must fail closed"
        );
    }

    #[test]
    fn gemma4_restored_sliding_sidecar_must_match_the_effective_prefix_exactly() {
        assert!(!gemma4_sliding_cold_sidecar_matches_prefix(0, 0));
        assert!(!gemma4_sliding_cold_sidecar_matches_prefix(16, 32));
        assert!(gemma4_sliding_cold_sidecar_matches_prefix(32, 32));
        assert!(!gemma4_sliding_cold_sidecar_matches_prefix(48, 32));
    }

    #[test]
    fn gemma4_unified_first_chunk_never_splits_inside_image_overlay() {
        assert_eq!(
            gemma4_vlm_prefill_chunk_end(0, 128, 32, true, 0, 48, Some(80)),
            128,
            "an inside-image prompt checkpoint must be ignored"
        );
        assert_eq!(
            gemma4_vlm_prefill_chunk_end(0, 128, 32, true, 0, 96, Some(80)),
            96,
            "a checkpoint after the complete image span is safe"
        );
        assert_eq!(
            gemma4_vlm_prefill_chunk_end(0, 128, 32, false, 16, 48, Some(80)),
            16,
            "causal E2B may still split at a leading-text checkpoint"
        );
    }

    #[test]
    fn gemma4_large_sliding_snapshots_are_memory_bounded() {
        let mut config = paged_tiny_config(None);
        config.num_hidden_layers = 40;
        config.layer_types = vec!["sliding_attention".to_string(); 40];
        config.num_kv_shared_layers = None;
        config.sliding_window = 1024;
        config.num_key_value_heads = 8;
        config.head_dim = 256;

        assert_eq!(
            gemma4_sliding_checkpoint_estimated_bytes(&config),
            40 * 1024 * 8 * 256 * 2 * 4
        );
        assert_eq!(
            gemma4_sliding_prefix_checkpoint_limit_for_override(&config, 16, None),
            2,
            "the default byte budget must not retain 128 huge unified snapshots"
        );
        assert_eq!(
            gemma4_sliding_retention_caps_for_override(&config, 16, false, None),
            Gemma4SlidingRetentionCaps::pre_ladder(
                2,
                Gemma4SlidingCheckpointBytes::for_config(&config)
            ),
            "a persistence-OFF turn must keep the pre-ladder cap verbatim"
        );
        assert_eq!(
            gemma4_sliding_retention_caps_for_override(&config, 16, true, None),
            Gemma4SlidingRetentionCaps::ladder(
                6,
                Gemma4SlidingAnchorRungs::from_slice(&[64, 256, 1024, 4096]),
                Gemma4SlidingCheckpointBytes::for_config(&config)
            ),
            "a persist turn widens by exactly the anchor rung count"
        );
    }

    /// A hybrid geometry with every sixth layer global, by the four axes that
    /// move the checkpoint byte arithmetic.
    fn sliding_config(
        num_hidden_layers: i32,
        sliding_window: i32,
        num_key_value_heads: i32,
        head_dim: i32,
        num_kv_shared_layers: Option<i32>,
    ) -> super::Gemma4Config {
        let mut config = paged_tiny_config(None);
        config.num_hidden_layers = num_hidden_layers;
        config.layer_types = (0..num_hidden_layers)
            .map(|index| {
                if (index + 1) % 6 == 0 {
                    "full_attention".to_string()
                } else {
                    "sliding_attention".to_string()
                }
            })
            .collect();
        config.num_kv_shared_layers = num_kv_shared_layers;
        config.sliding_window = sliding_window;
        config.num_key_value_heads = num_key_value_heads;
        config.head_dim = head_dim;
        config
    }

    /// The geometry that produced this bug on real weights:
    /// `Gemma-4-12B-IT-nvidia-mxfp-mlx` — 48 decoder layers, every sixth
    /// global, so 40 physical sliding layers; window 1024; 8 kv heads;
    /// head_dim 256; no KV sharing.
    fn twelve_b_sliding_config() -> super::Gemma4Config {
        sliding_config(48, 1024, 8, 256, None)
    }

    /// Geometries the byte cap must hold on besides the 12B.
    ///
    /// These are NOT claims about `Gemma-4-26B-A4B` or `Gemma-4-E2B`
    /// specifically: this repo carries no config for either and neither was
    /// available locally, so encoding one under that name would be a guess
    /// dressed as a fixture. What they do encode are the AXES a second geometry
    /// moves — KV sharing (which turns trailing sliding layers into aliases and
    /// so shrinks a checkpoint), a narrower window with fewer/smaller heads
    /// (which makes checkpoints cheap enough that the COUNT cap already fits the
    /// budget), and an all-global stack (no sliding state at all, where the byte
    /// cap must be an inert no-op rather than a divide-by-zero). Pinning the
    /// invariant across all four is the point; pinning it on one shape is how
    /// the count cap came to be treated as a byte cap in the first place.
    fn kv_shared_sliding_config() -> super::Gemma4Config {
        sliding_config(48, 1024, 8, 256, Some(4))
    }

    fn narrow_window_sliding_config() -> super::Gemma4Config {
        sliding_config(30, 512, 4, 128, None)
    }

    fn all_global_config() -> super::Gemma4Config {
        let mut config = sliding_config(30, 512, 4, 128, None);
        config.layer_types = vec!["full_attention".to_string(); 30];
        config
    }

    /// A draft, not a checkpoint: the anchor flag is not a thing a publish site
    /// (or a test standing in for one) can set. `into_checkpoint` derives it
    /// from the caps, and a test that set it by hand would be testing its own
    /// bookkeeping instead of the seam the prefill actually goes through.
    fn sliding_checkpoint_at(
        prefix_len: u32,
        block_size: u32,
        tokens: &[u32],
    ) -> Gemma4SlidingPrefixCheckpointDraft {
        Gemma4SlidingPrefixCheckpointDraft {
            prefix_len,
            block_size,
            final_block_hash: u64::from(prefix_len),
            protected_image_prompt_boundary: false,
            tokens: tokens[..prefix_len as usize].to_vec(),
            snapshots: Vec::new(),
        }
    }

    /// The same draft, flagged as a VLM prompt boundary — what
    /// `remember_gemma4_sliding_materialized_prompt_boundary_checkpoint_with_keys`
    /// stores on an image turn. Never an eviction candidate for steps 1 or 2 of
    /// the ladder victim rule, which is what makes the deep fallback reachable.
    fn image_prompt_checkpoint_at(
        prefix_len: u32,
        block_size: u32,
        tokens: &[u32],
    ) -> Gemma4SlidingPrefixCheckpointDraft {
        Gemma4SlidingPrefixCheckpointDraft {
            protected_image_prompt_boundary: true,
            ..sliding_checkpoint_at(prefix_len, block_size, tokens)
        }
    }

    /// Replay one prefill's checkpoint pushes through the real publish +
    /// retention seams, into a store that may already hold an earlier turn's
    /// entries. Returns the boundaries the chunk loop snapshotted at; what is
    /// left in `retained` afterwards is the state
    /// `capture_gemma4_sliding_cold_sidecar` runs against.
    ///
    /// The chunk walk mirrors `run_paged_prefill_chunk`'s pass-1 loop: the body
    /// is forwarded in `chunk_tokens`-sized pieces, each piece publishes
    /// `gemma4_sliding_chunk_checkpoint_boundaries` minus the prompt boundary,
    /// and the prompt boundary is stored last by its own path.
    ///
    /// The rung list is handed over on BOTH arms, unlike the call site (which
    /// does not bother computing it when the ladder is off). Refusing to
    /// publish is then the `want_ladder` parameter's own job, which is what a
    /// future refactor hoisting the rung computation out of its `if` must not
    /// be able to break silently.
    ///
    /// `start_offset` is the pass-1 loop's `chunk_first_position`, i.e. the
    /// `cached_prefix_len` a WARM turn resumes from. It is not cosmetic: the
    /// rung filter is strict (`rung > start_offset`), so a warm turn republishes
    /// none of the rungs below where it resumed, and the store it inherits is
    /// the only thing that still holds them.
    fn replay_prefill_into(
        retained: &mut VecDeque<Gemma4SlidingPrefixCheckpoint>,
        config: &super::Gemma4Config,
        block_size: u32,
        start_offset: u32,
        prompt_boundary: u32,
        chunk_tokens: u32,
        want_ladder: bool,
    ) -> Vec<u32> {
        let caps =
            gemma4_sliding_retention_caps_for_override(config, block_size, want_ladder, None);
        let interval = gemma4_sliding_decode_checkpoint_interval(config, block_size);
        let tokens: Vec<u32> = (0..prompt_boundary).collect();

        let mut published: Vec<u32> = Vec::new();
        let mut start = start_offset;
        while start < prompt_boundary {
            let end = (start + chunk_tokens).min(prompt_boundary);
            let mut boundaries =
                gemma4_sliding_chunk_checkpoint_boundaries(start, end, interval, caps);
            boundaries.retain(|boundary| *boundary != prompt_boundary);
            assert!(
                boundaries.windows(2).all(|pair| pair[0] < pair[1]),
                "prepare_sliding_checkpoint_capture rejects a non-ascending set: {boundaries:?}"
            );
            for boundary in boundaries {
                published.push(boundary);
                upsert_gemma4_sliding_prefix_checkpoint(
                    retained,
                    sliding_checkpoint_at(boundary, block_size, &tokens),
                    caps,
                    false,
                );
            }
            start = end;
        }
        published.push(prompt_boundary);
        upsert_gemma4_sliding_prefix_checkpoint(
            retained,
            sliding_checkpoint_at(prompt_boundary, block_size, &tokens),
            caps,
            false,
        );
        published
    }

    fn retained_boundaries(retained: &VecDeque<Gemma4SlidingPrefixCheckpoint>) -> Vec<u32> {
        retained
            .iter()
            .map(|checkpoint| checkpoint.prefix_len)
            .collect()
    }

    /// One COLD prefill into an empty store, the shape every offset-0 test wants.
    fn replay_prefill_checkpoints(
        config: &super::Gemma4Config,
        block_size: u32,
        prompt_boundary: u32,
        chunk_tokens: u32,
        want_ladder: bool,
    ) -> (Vec<u32>, Vec<u32>) {
        let mut retained: VecDeque<Gemma4SlidingPrefixCheckpoint> = VecDeque::new();
        let published = replay_prefill_into(
            &mut retained,
            config,
            block_size,
            0,
            prompt_boundary,
            chunk_tokens,
            want_ladder,
        );
        (published, retained_boundaries(&retained))
    }

    /// Deepest retained boundary a cold capture could anchor on, given the
    /// persisted K/V chain only reaches `chain_reach_tokens`. This is the
    /// selection `find_gemma4_sliding_capture_checkpoints` performs.
    fn deepest_reachable(retained: &[u32], chain_reach_tokens: u32) -> Option<u32> {
        retained
            .iter()
            .copied()
            .filter(|boundary| *boundary <= chain_reach_tokens)
            .max()
    }

    /// Every boundary a later restore of a `prompt_len`-token prompt would
    /// probe, ascending.
    ///
    /// A restated COPY of the READ path, never a call into the capture-side
    /// helpers it exists to pin — the same discipline
    /// `cold_tier_parity_harness::expected_checkpoint_ladder` follows. A test
    /// that derived its expectation from
    /// `gemma4_cold_restore_reachable_boundary` would move with that function
    /// and could never fail, which is exactly how the one-block gap survived
    /// three green suites.
    ///
    /// ```text
    ///   Gemma4Inner::prepare_gemma4_paged_turn
    ///       max_cache_hit_tokens = total_budget - 1
    ///   PagedKVCacheAdapter::find_cached_prefix_per_block_inner
    ///       lookup_len    = min(max_cache_hit_tokens, prompt_tokens.len())
    ///       lookup_tokens = &prompt_tokens[..lookup_len]
    ///   ColdTierWalk::restore_extend
    ///       full_blocks = lookup_tokens.len() / block_size
    ///   ColdTierWalk::deepest_backed_boundary
    ///       for count in (floor + 1..=keys.len()).rev()
    ///           boundary = count * block_size
    /// ```
    fn restore_probeable_boundaries(prompt_len: u32, block_size: u32) -> Vec<u32> {
        if block_size == 0 {
            return Vec::new();
        }
        let lookup_len = prompt_len.saturating_sub(1);
        let full_blocks = lookup_len / block_size;
        (1..=full_blocks).map(|count| count * block_size).collect()
    }

    /// The capture may never anchor where the restore cannot look.
    ///
    /// With the persisted chain unbounded, the ceiling this pins is the ONLY
    /// thing standing between the capture and a boundary no key on the read
    /// side ever spells. `request_tokens` is deliberately swept past the prompt:
    /// the capture runs at finalize, so it sees the completion too, and the
    /// defect was precisely that it measured its ceiling against that longer
    /// sequence.
    #[test]
    fn the_capture_ceiling_is_exactly_the_deepest_boundary_a_restore_can_probe() {
        for block_size in [1u32, 8, 16, 32, 64] {
            for prompt_len in 0..=400u32 {
                let probeable = restore_probeable_boundaries(prompt_len, block_size);
                for generated in [0usize, 1, 15, 512] {
                    let request_tokens_len = prompt_len as usize + generated;
                    let ceiling_blocks = gemma4_sliding_cold_capture_ceiling_blocks(
                        u32::MAX,
                        request_tokens_len,
                        prompt_len,
                        block_size,
                    );
                    let ceiling_tokens = ceiling_blocks as u32 * block_size;
                    match probeable.last() {
                        None => assert_eq!(
                            ceiling_tokens, 0,
                            "block_size={block_size} prompt_len={prompt_len} \
                             generated={generated}: a restore of this prompt can probe NO \
                             boundary, so a capture that names {ceiling_tokens} writes an \
                             object nothing can ask for"
                        ),
                        Some(&deepest) => assert_eq!(
                            ceiling_tokens, deepest,
                            "block_size={block_size} prompt_len={prompt_len} \
                             generated={generated}: the restore probes {probeable:?}; the \
                             capture ceiling must be its deepest member, not \
                             {ceiling_tokens}"
                        ),
                    }
                }
            }
        }
    }

    /// The one-block gap, with the numbers it was measured at.
    ///
    /// A 4-token A/B on Gemma-4-26B-A4B-IT-UD-Q4_K_XL-mlx, everything else held
    /// constant: the 6572-token prompt restored 6560 of 6572 tokens, the
    /// 6576-token one restored ZERO. The only difference is that 6576 is a
    /// multiple of 16, which puts the prompt-boundary checkpoint one block above
    /// `max_cache_hit_tokens`.
    #[test]
    fn a_block_aligned_prompt_is_the_only_case_the_prompt_boundary_outruns_the_restore() {
        const BS: u32 = 16;
        for (prompt_len, prompt_boundary, reachable) in
            [(6572u32, 6560u32, 6560u32), (6576, 6576, 6560)]
        {
            // What the prompt-boundary publisher aims at
            // (`prompt_checkpoint_boundary_len` in `run_paged_prefill_chunk`).
            assert_eq!(prompt_len / BS * BS, prompt_boundary);
            assert_eq!(
                gemma4_cold_restore_reachable_boundary(prompt_len, BS),
                reachable
            );
            let probeable = restore_probeable_boundaries(prompt_len, BS);
            assert!(
                probeable.contains(&reachable),
                "prompt_len={prompt_len}: the reachable boundary must be one the restore \
                 actually enumerates"
            );
            assert_eq!(
                probeable.contains(&prompt_boundary),
                prompt_boundary == reachable,
                "prompt_len={prompt_len}: the prompt boundary {prompt_boundary} is probeable \
                 if and only if it IS the reachable one; when it is not, a sidecar anchored \
                 there is dead on arrival and self-locking"
            );

            // And the ceiling the capture actually runs with, for the turn as
            // it happened: a 6576-token prompt with a 40-token completion and a
            // chain that covered every block of it.
            let ceiling_blocks = gemma4_sliding_cold_capture_ceiling_blocks(
                u32::MAX,
                prompt_len as usize + 40,
                prompt_len,
                BS,
            );
            assert_eq!(
                ceiling_blocks as u32 * BS,
                reachable,
                "prompt_len={prompt_len}: the capture must stop at {reachable}, the deepest \
                 boundary a restore of this prompt enumerates. It measured \
                 {} instead, which is what wrote 209.7 MB a restore could never name.",
                ceiling_blocks as u32 * BS
            );
        }
    }

    /// What the reachability clamp COSTS a growing conversation, and the bound
    /// on that cost.
    ///
    /// The clamp is scoped to the whole capture, not just to the aligned prompt
    /// boundary it was written for, so it also drops every candidate the DECODE
    /// published — `maybe_remember_gemma4_sliding_decode_boundary_checkpoint`
    /// publishes over `request_tokens` = prompt + generated, so those
    /// candidates are real, and turn N+1 of a growing conversation, whose
    /// prompt contains them, really could name them.
    ///
    /// It is kept broad anyway, and this test is the ledger for that choice.
    /// One sidecar is written per turn, so the clamp is a PRIORITY rule: it
    /// spends the turn's single write on the deepest boundary a restore of THIS
    /// prompt can name — the replay a cold tier exists for, and the case that
    /// measured zero reuse — instead of on a deeper boundary that pays off only
    /// if the conversation continues with exactly these tokens. The give-up is
    /// bounded by one turn: turn N+1's own ceiling covers everything turn N
    /// discarded, so the deeper boundaries are lost only to a process that dies
    /// between the two finalizes.
    #[test]
    fn the_capture_ceiling_gives_up_this_turns_generated_region_and_the_next_turn_covers_it() {
        const BS: u32 = 16;
        // Turn N: an aligned 6576-token prompt and a 2048-token completion,
        // long enough that the decode cadence published inside it.
        let prompt_n = 6576u32;
        let request_n = prompt_n as usize + 2048;
        let unclamped_n = (request_n as u32 / BS) * BS;
        let ceiling_n =
            gemma4_sliding_cold_capture_ceiling_blocks(u32::MAX, request_n, prompt_n, BS) as u32
                * BS;
        assert_eq!(
            ceiling_n, 6560,
            "the capture stops at the deepest boundary a restore of THIS prompt enumerates"
        );
        assert_eq!(
            unclamped_n, 8624,
            "…while the chain and the request alone would have allowed 8624"
        );
        assert_eq!(
            unclamped_n - ceiling_n,
            2064,
            "so the clamp gives up 2064 tokens' worth of generated-region candidates"
        );

        // Turn N+1 of the same conversation: its prompt is turn N's whole
        // request plus new user text, so its own ceiling sits at or past
        // everything turn N gave up.
        for new_user_tokens in [1u32, 17, 512] {
            let prompt_next = request_n as u32 + new_user_tokens;
            let ceiling_next = gemma4_sliding_cold_capture_ceiling_blocks(
                u32::MAX,
                prompt_next as usize + 8,
                prompt_next,
                BS,
            ) as u32
                * BS;
            assert!(
                ceiling_next >= unclamped_n,
                "new_user_tokens={new_user_tokens}: turn N+1 must be able to name every \
                 boundary turn N discarded ({ceiling_next} < {unclamped_n}), or the clamp \
                 loses them for good instead of deferring them by one turn"
            );
        }
    }

    /// Which prompts need the extra cold-restore tail checkpoint at all, and
    /// where it must sit.
    ///
    /// Exactly the block-aligned ones, exactly one block below the prompt
    /// boundary. Everywhere else the two coincide and the prefill publishes
    /// nothing extra — which is what keeps the added snapshot from being a cost
    /// every turn pays.
    #[test]
    fn the_cold_tail_checkpoint_is_needed_exactly_when_the_prompt_is_block_aligned() {
        for block_size in [1u32, 8, 16, 32] {
            for prompt_len in 1..=300u32 {
                let prompt_boundary = prompt_len / block_size * block_size;
                let tail = gemma4_cold_restore_reachable_boundary(prompt_len, block_size);
                if prompt_len.is_multiple_of(block_size) {
                    assert_eq!(
                        tail + block_size,
                        prompt_boundary,
                        "block_size={block_size} prompt_len={prompt_len}: an aligned prompt \
                         needs a tail one block below its own boundary"
                    );
                } else {
                    assert_eq!(
                        tail, prompt_boundary,
                        "block_size={block_size} prompt_len={prompt_len}: a ragged prompt's \
                         boundary is already reachable, so nothing extra may be published"
                    );
                }
            }
        }
    }

    /// THE persistence-OFF transparency claim for this change, as a test.
    ///
    /// Chunk length is the GEMM's `M`, and the retained checkpoint set decides
    /// which one a later warm turn resumes from, so both are observable in the
    /// emitted tokens. A turn with no `SlidingWindow` sidecar policy must
    /// therefore snapshot exactly what it snapshotted before the cold tier
    /// existed: nothing extra, at any prompt length, at any block size.
    #[test]
    fn a_persistence_off_turn_publishes_no_cold_restore_tail_at_all() {
        let config = twelve_b_sliding_config();
        for block_size in [1u32, 8, 16, 32] {
            let off = gemma4_sliding_retention_caps_for_override(&config, block_size, false, None);
            let on = gemma4_sliding_retention_caps_for_override(&config, block_size, true, None);
            assert!(!off.wants_ladder() && on.wants_ladder());
            for prompt_len in 0..=300u32 {
                assert_eq!(
                    gemma4_cold_restore_tail_publish(prompt_len, block_size, off),
                    None,
                    "block_size={block_size} prompt_len={prompt_len}: a persistence-OFF turn \
                     that snapshots one extra boundary changes the retained set, and with it \
                     the depth a later warm turn resumes from"
                );
                let reachable = gemma4_cold_restore_reachable_boundary(prompt_len, block_size);
                assert_eq!(
                    gemma4_cold_restore_tail_publish(prompt_len, block_size, on),
                    (prompt_len.is_multiple_of(block_size) && reachable > 0).then_some(reachable),
                    "block_size={block_size} prompt_len={prompt_len}"
                );
            }
        }
    }

    /// Where the tail lands inside the prefill's chunk walk, and — the half that
    /// keeps it numerically inert — that it never displaces a boundary the
    /// chunk was already going to snapshot.
    #[test]
    fn the_cold_restore_tail_is_captured_beside_the_chunks_own_boundaries_never_instead() {
        // A 6576-token aligned prompt: the body runs to position 6575, so the
        // tail at 6560 falls inside the final chunk.
        let tail = gemma4_cold_restore_tail_publish(
            6576,
            16,
            gemma4_sliding_retention_caps_for_override(&twelve_b_sliding_config(), 16, true, None),
        );
        assert_eq!(tail, Some(6560));

        // Earlier chunks pass it by.
        assert_eq!(gemma4_chunk_cold_restore_tail(tail, 0, 2048, &[1024]), None);
        assert_eq!(
            gemma4_chunk_cold_restore_tail(tail, 2048, 4096, &[3072]),
            None
        );
        // The chunk that crosses it takes it.
        assert_eq!(
            gemma4_chunk_cold_restore_tail(tail, 4096, 6575, &[5120, 6144]),
            Some(6560)
        );
        // `(start, end]`, so a chunk that merely STARTS there does not re-take
        // it — `prepare_sliding_checkpoint_capture` needs strictly increasing
        // offsets and a duplicate would be rejected.
        assert_eq!(
            gemma4_chunk_cold_restore_tail(tail, 6560, 6575, &[]),
            None,
            "a boundary at or below where the chunk began was already passed"
        );
        // And when the cadence or a rung already lands on it, the tail adds
        // nothing: the retained set must stay byte-for-byte the persist-off one.
        assert_eq!(
            gemma4_chunk_cold_restore_tail(tail, 4096, 6575, &[5120, 6144, 6560]),
            None,
            "the tail must never be routed to the singleton INSTEAD of the store when the \
             chunk was already publishing that boundary — that would silently remove an entry \
             a persistence-OFF turn retains"
        );
    }

    /// A RAGGED prompt must publish no tail at all, and the chunk walk cannot
    /// be the thing that enforces it.
    ///
    /// On 15 prompts out of 16 the reachable boundary and the prompt boundary
    /// are the same number, and the prompt boundary is already snapshotted by
    /// `maybe_remember_gemma4_sliding_prompt_boundary_checkpoint` — which the
    /// chunk plan is split at, so that path always fires when the tail would
    /// have. Publishing the tail as well takes a SECOND full sliding-window
    /// snapshot of one offset, and `find_gemma4_sliding_capture_checkpoints`
    /// then dedups the pair back down to one candidate. Cost with no reader.
    ///
    /// The chunk walk cannot catch it: `run_paged_prefill_chunk` strips the
    /// prompt boundary out of `checkpoint_boundaries` one line before handing
    /// that same list in as `already_published`, so
    /// `gemma4_chunk_cold_restore_tail`'s containment test is blind to exactly
    /// the coinciding case. The publish gate is where it has to be decided.
    #[test]
    fn a_ragged_prompt_publishes_no_tail_beside_the_prompt_boundary_it_coincides_with() {
        const BS: u32 = 16;
        let caps =
            gemma4_sliding_retention_caps_for_override(&twelve_b_sliding_config(), BS, true, None);

        // The measured pair: 6572 ragged, 6576 aligned, same block size.
        assert_eq!(
            6572 / BS * BS,
            gemma4_cold_restore_reachable_boundary(6572, BS)
        );
        assert_eq!(
            gemma4_cold_restore_tail_publish(6572, BS, caps),
            None,
            "a ragged prompt's own boundary IS the reachable one and is already \
             snapshotted; a tail here is a duplicate window nothing reads"
        );
        assert_eq!(
            gemma4_cold_restore_tail_publish(6576, BS, caps),
            Some(6560),
            "the aligned prompt is the one case the tail exists for and must keep it"
        );

        // What the prefill would do with it. The chunk that ends on the ragged
        // prompt boundary is the same chunk the prompt-boundary path fires
        // after, and the stripped `already_published` cannot say so.
        let ragged_tail = gemma4_cold_restore_tail_publish(6572, BS, caps);
        assert_eq!(
            gemma4_chunk_cold_restore_tail(ragged_tail, 4096, 6560, &[5120, 6144]),
            None,
            "the chunk that ends on the prompt boundary must publish nothing extra"
        );

        // And it is not one fixture: exactly the aligned prompts publish.
        for block_size in [1u32, 8, 16, 32] {
            let caps = gemma4_sliding_retention_caps_for_override(
                &twelve_b_sliding_config(),
                block_size,
                true,
                None,
            );
            for prompt_len in 1..=300u32 {
                let published =
                    gemma4_cold_restore_tail_publish(prompt_len, block_size, caps).is_some();
                let reachable = gemma4_cold_restore_reachable_boundary(prompt_len, block_size);
                assert_eq!(
                    published,
                    prompt_len.is_multiple_of(block_size) && reachable > 0,
                    "block_size={block_size} prompt_len={prompt_len}: the tail exists only \
                     where the prompt boundary outruns the restore, i.e. only on an aligned \
                     prompt"
                );
            }
        }
    }

    /// The descent must step PAST what is already on disk, or a poisoned root
    /// never heals.
    ///
    /// The scenario is the one users already have: a pre-fix run anchored a
    /// sidecar at the aligned prompt boundary 6576, which no restore can name.
    /// Everything shallower is still missing. A walk that stops at the first
    /// already-persisted candidate writes nothing, this turn and every turn
    /// after it, because the key it recomputes is the same key.
    #[test]
    fn the_capture_descends_past_boundaries_already_on_disk() {
        // Deepest first, exactly as `find_gemma4_sliding_capture_checkpoints`
        // hands them over.
        let candidates = [6576u32, 6560, 4096, 1024];
        let on_disk = [6576u32];

        assert_eq!(
            gemma4_select_cold_capture_candidate(candidates, |boundary| {
                if on_disk.contains(boundary) {
                    Gemma4ColdCaptureProbe::Persisted
                } else {
                    Gemma4ColdCaptureProbe::Missing(*boundary)
                }
            }),
            Gemma4ColdCaptureSelection::Capture {
                candidate: 6560,
                key: 6560,
                skipped_persisted: 1,
            },
            "the deepest candidate is the useless one already on disk; the capture must go on \
             to 6560, the boundary a restore of this prompt actually enumerates, and account \
             for the one it passed over"
        );

        // Steady state: everything reachable is already written, so the turn
        // does nothing and says so.
        assert_eq!(
            gemma4_select_cold_capture_candidate(candidates, |_| {
                Gemma4ColdCaptureProbe::<u32>::Persisted
            }),
            Gemma4ColdCaptureSelection::AllPersisted {
                skipped_persisted: candidates.len(),
            },
            "a fully populated ladder must write nothing, not re-enqueue its shallowest rung"
        );

        // A boundary whose chain cannot be derived is not a skip: nothing was
        // ever written there to skip, and counting it as one would make a
        // healthy short turn wear the signature of a saturated ladder.
        assert_eq!(
            gemma4_select_cold_capture_candidate(candidates, |boundary| match *boundary {
                6576 => Gemma4ColdCaptureProbe::Underivable,
                6560 => Gemma4ColdCaptureProbe::Persisted,
                other => Gemma4ColdCaptureProbe::Missing(other),
            }),
            Gemma4ColdCaptureSelection::Capture {
                candidate: 4096,
                key: 4096,
                skipped_persisted: 1,
            }
        );
    }

    /// A descent that derived NO chain at all must not reach the counters
    /// wearing the saturated ladder's signature.
    ///
    /// `Underivable` and `Persisted` are opposite states of the tier — one
    /// means nothing was ever written at that boundary, the other means
    /// something was — and the capture records a different counter for each.
    /// When the walk returned `(None, 0)` for both, the all-`Underivable` turn
    /// bumped `already_persisted`, so a root holding nothing reported itself
    /// full.
    #[test]
    fn an_all_underivable_descent_is_not_an_already_persisted_one() {
        let candidates = [6576u32, 6560, 4096, 1024];

        assert_eq!(
            gemma4_select_cold_capture_candidate(candidates, |_| {
                Gemma4ColdCaptureProbe::<u32>::Underivable
            }),
            Gemma4ColdCaptureSelection::NoChainDerived,
            "not one chain derived, so the tier holds nothing here — reporting this as an \
             already-persisted descent makes an empty root read as a saturated one"
        );

        // Mixed, and still not a persistence claim: one derivable boundary that
        // IS on disk is what separates the two outcomes.
        assert_eq!(
            gemma4_select_cold_capture_candidate(candidates, |boundary| match *boundary {
                1024 => Gemma4ColdCaptureProbe::Persisted,
                _ => Gemma4ColdCaptureProbe::<u32>::Underivable,
            }),
            Gemma4ColdCaptureSelection::AllPersisted {
                skipped_persisted: 1,
            }
        );

        // An empty candidate list derives nothing either.
        assert_eq!(
            gemma4_select_cold_capture_candidate([0u32; 0], |_| {
                Gemma4ColdCaptureProbe::<u32>::Underivable
            }),
            Gemma4ColdCaptureSelection::NoChainDerived
        );
    }

    /// The reachability clamp is an EXTRA bound, not a replacement: the
    /// persisted chain and the request still cap the capture, and a turn whose
    /// prompt length was never recorded captures nothing rather than guessing.
    #[test]
    fn the_capture_ceiling_still_honours_the_chain_the_request_and_a_missing_prompt() {
        assert_eq!(
            gemma4_sliding_cold_capture_ceiling_blocks(3, 6616, 6576, 16),
            3,
            "a chain that reached 3 blocks bounds the capture at 3 blocks"
        );
        assert_eq!(
            gemma4_sliding_cold_capture_ceiling_blocks(u32::MAX, 32, 6576, 16),
            2,
            "a request holding 2 whole blocks cannot anchor deeper than 2"
        );
        assert_eq!(
            gemma4_sliding_cold_capture_ceiling_blocks(u32::MAX, 6576, 0, 16),
            0,
            "no recorded prompt length must fail CLOSED, never fall back to the request"
        );
        assert_eq!(
            gemma4_sliding_cold_capture_ceiling_blocks(u32::MAX, 6576, 6576, 0),
            0,
            "block_size 0 folds to a no-op instead of dividing by zero"
        );
    }

    #[test]
    fn gemma4_sliding_anchor_rungs_are_powers_of_four_from_the_block_size() {
        let config = twelve_b_sliding_config();
        assert_eq!(
            gemma4_sliding_cold_anchor_rungs(&config, 16, 2),
            vec![64, 256, 1024, 4096],
            "the grid is block_size * 4^k, pinned to zero so the same rung is \
             reusable by every later turn sharing the prefix"
        );

        // Why the fourth rung fits at all: a rung's payload is min(b, window)
        // rows, so the two sub-window rungs are nearly free. Charging every
        // entry a full window - what `gemma4_sliding_checkpoint_estimated_bytes`
        // does, and all any pre-ladder caller needed - does not fit.
        let full_window = gemma4_sliding_checkpoint_estimated_bytes(&config);
        let reserve = full_window * 2;
        let actual: u64 = [64u32, 256, 1024, 4096]
            .iter()
            .map(|rung| gemma4_sliding_checkpoint_estimated_bytes_at(&config, *rung))
            .sum();
        assert!(
            actual + reserve <= GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES,
            "boundary-scaled: {} + {} > {}",
            actual,
            reserve,
            GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES
        );
        assert!(
            full_window * 4 + reserve > GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES,
            "flat full-window sizing would have to refuse a rung"
        );
    }

    #[test]
    fn gemma4_sliding_checkpoint_bytes_scale_with_min_boundary_window() {
        let config = twelve_b_sliding_config();
        let full_window = gemma4_sliding_checkpoint_estimated_bytes(&config);
        assert_eq!(
            gemma4_sliding_checkpoint_estimated_bytes_at(&config, 64),
            full_window / 16,
            "a 64-token rung carries 64 of the window's 1024 rows"
        );
        assert_eq!(
            gemma4_sliding_checkpoint_estimated_bytes_at(&config, 1024),
            full_window
        );
        assert_eq!(
            gemma4_sliding_checkpoint_estimated_bytes_at(&config, 4096),
            full_window,
            "past the window a payload stops growing"
        );
    }

    /// The headline gate for the gemma4 cold-tier ladder.
    ///
    /// Reproduced twice on real weights before the fix
    /// (`Gemma-4-12B-IT-nvidia-mxfp-mlx`, 8140-token prompt, `mlx agent`):
    ///
    /// ```text
    ///   W1 cold     chain reach  576 tok (36 blk)   0 sliding_window sidecars
    ///   W2 restart  chain reach 1136 tok (71 blk)   0 sliding_window sidecars
    ///   trace: sliding_cold_sidecar_capture_skipped
    ///          reason=no_representable_checkpoint_at_or_below_chain_reach
    /// ```
    ///
    /// The store finished at `{7168, 8128}` both times: the cadence fires every
    /// window, `limit` is 2 on this geometry, and the pre-ladder victim is the
    /// oldest entry — so the rung at 1024 was born and then evicted, and
    /// nothing at or below the chain's reach was left.
    #[test]
    fn gemma4_sliding_ladder_retains_a_rung_the_lagging_chain_can_reach() {
        let config = twelve_b_sliding_config();
        let (published, retained) = replay_prefill_checkpoints(&config, 16, 8128, 2048, true);
        assert_eq!(
            published,
            vec![64, 256, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8128],
            "the cadence, plus the anchor rungs, in one ascending set per chunk"
        );
        assert_eq!(
            retained,
            vec![64, 256, 1024, 4096, 7168, 8128],
            "the anchors must survive the cadence ratcheting deeper entries in behind them"
        );
        assert_eq!(
            deepest_reachable(&retained, 576),
            Some(256),
            "turn 1: the chain reached 36 blocks, so only a sub-window rung can anchor"
        );
        assert_eq!(
            deepest_reachable(&retained, 1136),
            Some(1024),
            "turn 2: the chain reached 71 blocks and the rung at 1024 must still be there"
        );
        assert_eq!(
            deepest_reachable(&retained, 4200),
            Some(4096),
            "later turns must keep deepening rather than sticking at one window"
        );
    }

    /// Lesson (a) from qwen3.5's GDN ladder, which shipped broken twice: a
    /// request with no cold tier must retain exactly what it retained before
    /// the ladder existed. Which checkpoint a later warm turn lands on decides
    /// whether `prepare_gemma4_sliding_prefix` installs a snapshot or replays
    /// the whole cached prefix, and those emit different tokens.
    #[test]
    fn gemma4_persistence_off_retains_exactly_the_pre_ladder_set() {
        let config = twelve_b_sliding_config();
        let caps = gemma4_sliding_retention_caps_for_override(&config, 16, false, None);
        assert_eq!(
            caps,
            Gemma4SlidingRetentionCaps::pre_ladder(
                gemma4_sliding_prefix_checkpoint_limit_for_override(&config, 16, None),
                Gemma4SlidingCheckpointBytes::for_config(&config)
            )
        );
        let (published, retained) = replay_prefill_checkpoints(&config, 16, 8128, 2048, false);
        assert_eq!(
            published,
            vec![1024, 2048, 3072, 4096, 5120, 6144, 7168, 8128],
            "no cold tier means the bare cadence: no rung may be snapshotted"
        );
        assert_eq!(
            retained,
            vec![7168, 8128],
            "and it is trimmed oldest-first to the pre-ladder cap, as it always was"
        );
    }

    #[test]
    fn gemma4_sliding_published_boundaries_are_unchanged_when_the_ladder_is_off() {
        let config = twelve_b_sliding_config();
        // Same rungs on both arms, so refusing to publish them is the POLICY's
        // job. A `PreLadder` turn does not even compute a grid at the call
        // site; handing it one here is what makes this discriminating.
        let anchors =
            Gemma4SlidingAnchorRungs::from_slice(&gemma4_sliding_cold_anchor_rungs(&config, 16, 2));
        assert!(anchors.len > 0);
        let off = Gemma4SlidingRetentionCaps {
            anchors,
            ..gemma4_sliding_retention_caps_for_override(&config, 16, false, None)
        };
        let on = gemma4_sliding_retention_caps_for_override(&config, 16, true, None);
        for (start, end) in [(0u32, 2048u32), (2048, 4096), (4096, 6144), (6144, 8128)] {
            assert_eq!(
                gemma4_sliding_chunk_checkpoint_boundaries(start, end, 1024, off),
                gemma4_sliding_checkpoint_boundaries_crossed(start, end, 1024),
                "chunk ({start}, {end}] must publish the bare cadence with the ladder off"
            );
        }
        assert_eq!(
            gemma4_sliding_chunk_checkpoint_boundaries(0, 2048, 1024, on),
            vec![64, 256, 1024, 2048],
            "with the ladder on, a rung that coincides with the cadence is published once"
        );
    }

    /// Every cursor in `first..=last` at which decode publishes a checkpoint.
    /// `first` is the token count the prefill left behind, since decode only
    /// ever walks forward from there.
    fn decode_published_boundaries(
        config: &super::Gemma4Config,
        block_size: u32,
        caps: Gemma4SlidingRetentionCaps,
        first: u32,
        last: u32,
    ) -> Vec<u32> {
        let interval = gemma4_sliding_decode_checkpoint_interval(config, block_size);
        (first..=last)
            .filter(|&cursor| gemma4_sliding_decode_publishes_checkpoint(cursor, interval, caps))
            .collect()
    }

    /// Decode is the ONLY publisher for the shape `mlx agent` actually sends —
    /// a short prompt and a long generation — and before this it fired on the
    /// cadence alone.
    ///
    /// The cadence is `max(window, block).div_ceil(block) * block` = 1024 here,
    /// and `window / block_size = 64 = 4^3`, so the rung ladder and the cadence
    /// COLLIDE at every rung with `k >= 3` and miss each other everywhere below:
    ///
    /// ```text
    ///   rungs    64   256   1024   4096
    ///   cadence              1024   4096  (…every 1024)
    ///   union    64   256   1024   4096
    /// ```
    ///
    /// A 200-token prompt publishes {64} at prefill and then generates. Without
    /// the union, 256 is published by nothing: the cadence skips it, and the
    /// next turn's prefill starts past it and its rung filter is strict
    /// (`rung > start_offset`). So the chain — which advances ~34 blocks
    /// (~544 tokens) per turn — has nothing at or below its reach to anchor on,
    /// which is exactly the inert cold tier the ladder exists to fix.
    #[test]
    fn gemma4_sliding_decode_publishes_the_rungs_the_cadence_skips() {
        let config = twelve_b_sliding_config();
        let caps = gemma4_sliding_retention_caps_for_override(&config, 16, true, None);
        assert_eq!(
            gemma4_sliding_decode_checkpoint_interval(&config, 16),
            1024,
            "the cadence is a whole window, which is ABOVE two of the four rungs"
        );
        assert_eq!(caps.anchors.as_slice(), &[64, 256, 1024, 4096]);
        assert_eq!(
            decode_published_boundaries(&config, 16, caps, 1, 1200),
            vec![64, 256, 1024],
            "the cadence UNION the rungs — the two sub-window rungs are the whole point"
        );
        assert!(
            !gemma4_sliding_decode_publishes_checkpoint(0, 1024, caps),
            "an empty request publishes nothing"
        );
    }

    /// Defect A: a checkpoint that genuinely sits on a rung but was born with
    /// `cold_anchor_rung` clear is the ladder's PREFERRED eviction victim, so
    /// the rung the decode path just published is the FIRST thing thrown away.
    ///
    /// This drives the decode publisher (short prompt, long generation) through
    /// the same store seam production uses, and asserts the rungs outlive the
    /// deeper cadence entries that ratchet in behind them.
    #[test]
    fn gemma4_sliding_decode_rungs_survive_the_cadence_ratcheting_past_them() {
        let config = twelve_b_sliding_config();
        let block_size = 16;
        let caps = gemma4_sliding_retention_caps_for_override(&config, block_size, true, None);
        let tokens: Vec<u32> = (0..6000).collect();
        let mut retained: VecDeque<Gemma4SlidingPrefixCheckpoint> = VecDeque::new();

        // Turn 1: a 200-token prompt. Only the shallowest rung is crossed.
        let published = replay_prefill_into(&mut retained, &config, block_size, 0, 200, 2048, true);
        assert_eq!(published, vec![64, 200]);

        // Then generate to 6000. Every cursor the decode predicate accepts is
        // stored, exactly as `maybe_remember_gemma4_sliding_decode_boundary_checkpoint`
        // does.
        let decode_boundaries = decode_published_boundaries(&config, block_size, caps, 201, 6000);
        assert_eq!(
            decode_boundaries,
            vec![256, 1024, 2048, 3072, 4096, 5120],
            "256 is published by decode or by nothing at all"
        );
        for boundary in &decode_boundaries {
            upsert_gemma4_sliding_prefix_checkpoint(
                &mut retained,
                sliding_checkpoint_at(*boundary, block_size, &tokens),
                caps,
                false,
            );
        }

        let survivors = retained_boundaries(&retained);
        assert_eq!(
            survivors,
            vec![64, 256, 1024, 3072, 4096, 5120],
            "the rungs are deferred; the plain cadence entries are what gets evicted"
        );
        let flagged: Vec<u32> = retained
            .iter()
            .filter(|checkpoint| checkpoint.cold_anchor_rung)
            .map(|checkpoint| checkpoint.prefix_len)
            .collect();
        assert_eq!(
            flagged,
            vec![64, 256, 1024, 4096],
            "a rung published by DECODE must carry the flag too, not only one \
             published by the prefill capture path"
        );
        assert_eq!(
            deepest_reachable(&survivors, 544),
            Some(256),
            "the chain advances ~34 blocks a turn; only a sub-window rung is in reach"
        );
    }

    /// Defect B's other half, and the axis `replay_prefill_checkpoints` could
    /// not see while it hard-coded `start = 0`: a WARM turn resumes at
    /// `cached_prefix_len`, and `gemma4_sliding_chunk_checkpoint_boundaries`
    /// filters `rung > start_offset`, so it republishes none of the rungs below
    /// where it resumed. The inherited store is the only thing that still holds
    /// them, and `Ladder` retention is the only reason it still does.
    #[test]
    fn gemma4_sliding_warm_turn_keeps_the_rungs_it_cannot_republish() {
        let config = twelve_b_sliding_config();
        let block_size = 16;
        let mut retained: VecDeque<Gemma4SlidingPrefixCheckpoint> = VecDeque::new();

        let turn1 = replay_prefill_into(&mut retained, &config, block_size, 0, 512, 2048, true);
        assert_eq!(turn1, vec![64, 256, 512]);

        let turn2 = replay_prefill_into(&mut retained, &config, block_size, 512, 8128, 2048, true);
        assert_eq!(
            turn2,
            vec![1024, 2048, 3072, 4096, 5120, 6144, 7168, 8128],
            "resuming at 512 republishes no rung below 512"
        );

        let survivors = retained_boundaries(&retained);
        assert_eq!(
            survivors,
            vec![64, 256, 1024, 4096, 7168, 8128],
            "the shallow rungs turn 2 could not republish must survive it"
        );
        assert_eq!(
            deepest_reachable(&survivors, 544),
            Some(256),
            "otherwise a warm turn silently loses everything the chain can reach"
        );
    }

    /// Defect C: the ladder's `limit` is a COUNT derived from a byte budget on
    /// the assumption that the extra slots hold cheap sub-window rungs —
    /// `gemma4_sliding_cold_anchor_rungs` prices a 64-token rung at 41.9 MB
    /// rather than 671.1 MB, which is the only reason a fourth rung fit. Nothing
    /// forces the retained set to BE that mix. Once the cursor is past one
    /// window every retained entry costs a full window:
    ///
    /// ```text
    ///   6 x 671.1 MB = 4026 MB   vs   budget 3072 MB    (+31%)
    /// ```
    ///
    /// On unified memory that gigabyte is not taken from a spare tier; it comes
    /// out of the weights and the paged pool (see `docs/architecture.md`), and
    /// an oversized pool separately costs ~10x on long-context decode.
    #[test]
    fn gemma4_sliding_ladder_bounds_the_retained_set_in_bytes_not_entries() {
        let block_size = 16u32;
        let mut geometries_that_overran_the_count_cap = 0usize;
        for (label, config) in [
            ("12B", twelve_b_sliding_config()),
            ("kv-shared", kv_shared_sliding_config()),
            ("narrow-window", narrow_window_sliding_config()),
            ("all-global", all_global_config()),
        ] {
            let caps = gemma4_sliding_retention_caps_for_override(&config, block_size, true, None);
            let window = config.sliding_window.max(0) as u32;
            let full_window = gemma4_sliding_checkpoint_estimated_bytes(&config);
            let tokens: Vec<u32> = (0..u32::from(u16::MAX)).collect();

            // Every entry sits PAST the window, so each costs a full window, and
            // none lands on a rung — the count cap alone would keep all of them.
            let deep: Vec<u32> = (1..=caps.limit as u32)
                .map(|index| window + block_size * index)
                .collect();
            assert!(
                deep.iter()
                    .all(|boundary| !caps.anchors.contains(*boundary)),
                "{label}: the scenario must be plain deep entries, not rungs"
            );

            let mut retained: VecDeque<Gemma4SlidingPrefixCheckpoint> = VecDeque::new();
            for boundary in &deep {
                upsert_gemma4_sliding_prefix_checkpoint(
                    &mut retained,
                    sliding_checkpoint_at(*boundary, block_size, &tokens),
                    caps,
                    false,
                );
            }

            let retained_bytes = caps.bytes.total(retained.iter());
            assert!(
                retained_bytes <= GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES,
                "{label}: retained {} entries for {} bytes, over the declared {} ceiling",
                retained.len(),
                retained_bytes,
                GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES
            );
            assert!(
                !retained.is_empty(),
                "{label}: the byte cap must bound the set, not empty it"
            );

            let count_only_bytes = caps.limit as u64 * full_window;
            if count_only_bytes > GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES {
                geometries_that_overran_the_count_cap += 1;
                assert!(
                    retained.len() < caps.limit,
                    "{label}: {} full-window entries are {} bytes, so the byte cap had to evict",
                    caps.limit,
                    count_only_bytes
                );
            }
        }
        assert!(
            geometries_that_overran_the_count_cap > 0,
            "no geometry actually exercised the overrun; the assertion above would be vacuous"
        );
    }

    /// The byte budget must not be paid for out of the one rung the chain can
    /// reach.
    ///
    /// `gemma4_sliding_ladder_victim` skips anchors at step 1 and skips
    /// ancestor-anchors at step 2, but its pre-ladder FLOOR is
    /// `position(|c| !protected_image_prompt_boundary)` — the SHALLOWEST entry,
    /// anchor or not. Reaching the floor is not exotic: the two
    /// `GEMMA4_IMAGE_PREFIX_CHECKPOINT_LIMIT` slots are never eligible for
    /// steps 1 or 2, so a store of `{image, image, rung, rung, rung, deep}` has
    /// nothing for either step to take. That store is what a VLM turn followed
    /// by a fresh text turn leaves behind: `save_paged_history` clears
    /// `cached_paged_image_token_positions` on a fresh text turn, so the media
    /// refusal in `capture_gemma4_sliding_cold_sidecar` lifts, while the
    /// protected entries stay in the store.
    ///
    /// ```text
    ///   store            img@2048  img@3072   256    1024   4096   deep@5120
    ///   bytes (MB)          671.1     671.1  167.8   671.1  671.1      671.1
    ///   total 3523.2 MB  >  3072 MB ceiling  ->  the byte loop must evict
    ///
    ///   shallowest-first   evicts 256 then 1024   ->  chain@544 reaches NOTHING
    ///   deepest-anchor     evicts 4096            ->  chain@544 reaches 256
    /// ```
    ///
    /// Evicting the deepest anchor is the cheap answer as well as the right
    /// one: one eviction clears the overrun where the shallow rungs take two.
    #[test]
    fn gemma4_sliding_ladder_byte_budget_never_evicts_the_shallowest_reachable_rung() {
        let config = twelve_b_sliding_config();
        let block_size = 16u32;
        let caps = gemma4_sliding_retention_caps_for_override(&config, block_size, true, None);
        assert_eq!(caps.limit, 6);
        assert_eq!(caps.anchors.as_slice(), &[64, 256, 1024, 4096]);
        let tokens: Vec<u32> = (0..8192).collect();

        let mut retained: VecDeque<Gemma4SlidingPrefixCheckpoint> = VecDeque::new();
        for draft in [
            image_prompt_checkpoint_at(2048, block_size, &tokens),
            image_prompt_checkpoint_at(3072, block_size, &tokens),
            sliding_checkpoint_at(256, block_size, &tokens),
            sliding_checkpoint_at(1024, block_size, &tokens),
            sliding_checkpoint_at(4096, block_size, &tokens),
            sliding_checkpoint_at(5120, block_size, &tokens),
        ] {
            upsert_gemma4_sliding_prefix_checkpoint(&mut retained, draft, caps, false);
        }

        // The scenario has to be the byte loop and nothing else: the count loop
        // never fires (six entries against a limit of six), and the six of them
        // genuinely overrun the ceiling.
        assert!(
            caps.bytes.at(2048) * 5 + caps.bytes.at(256)
                > GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES,
            "the pushed set must exceed the ceiling, or this proves nothing"
        );

        let survivors: Vec<(u32, bool, bool)> = retained
            .iter()
            .map(|checkpoint| {
                (
                    checkpoint.prefix_len,
                    checkpoint.cold_anchor_rung,
                    checkpoint.protected_image_prompt_boundary,
                )
            })
            .collect();
        assert_eq!(
            survivors,
            vec![
                (2048, false, true),
                (3072, false, true),
                (256, true, false),
                (1024, true, false),
                (5120, false, false),
            ],
            "the byte loop must take the DEEPEST anchor (4096). Taking the shallowest \
             re-creates the very failure the anchor flag exists to prevent, and taking \
             index 0 throws away a protected image boundary the count loop is required \
             to keep"
        );
        assert!(
            caps.bytes.total(retained.iter()) <= GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES,
            "and it must actually get under the ceiling"
        );

        let boundaries = retained_boundaries(&retained);
        assert_eq!(
            deepest_reachable(&boundaries, 544),
            Some(256),
            "the persisted chain advances ~34 blocks (~544 tokens) a turn; the shallow \
             rung is the only thing it can anchor on"
        );
    }

    /// Persistence-OFF, the byte axis. `PreLadder` carries the SAME per-entry
    /// cost model as `Ladder` (see `Gemma4SlidingRetentionCaps::bytes`), so the
    /// only thing keeping the byte cap off a persistence-OFF turn is `policy`.
    /// An override of 8 on the 12B geometry is 5120 MB — well over the ladder's
    /// 3072 MB ceiling — and it must still retain all 8, because a smaller
    /// retained set moves which checkpoint a later warm turn resumes from and
    /// that changes emitted tokens.
    #[test]
    fn gemma4_persistence_off_is_never_trimmed_by_the_ladder_byte_budget() {
        let config = twelve_b_sliding_config();
        let block_size = 16u32;
        let caps = gemma4_sliding_retention_caps_for_override(&config, block_size, false, Some(8));
        assert_eq!(caps.limit, 8);
        assert!(!caps.wants_ladder());
        assert!(
            caps.bytes.full_window_bytes > 0,
            "the cost model must be populated on this arm, or the guard below \
             would hold for a second, silent reason"
        );

        let tokens: Vec<u32> = (0..u32::from(u16::MAX)).collect();
        let mut retained: VecDeque<Gemma4SlidingPrefixCheckpoint> = VecDeque::new();
        let deep: Vec<u32> = (1..=8u32).map(|index| 1024 + block_size * index).collect();
        for boundary in &deep {
            upsert_gemma4_sliding_prefix_checkpoint(
                &mut retained,
                sliding_checkpoint_at(*boundary, block_size, &tokens),
                caps,
                false,
            );
        }
        assert!(
            caps.bytes.total(retained.iter()) > GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES,
            "the scenario must actually exceed the ladder budget, or this proves nothing"
        );
        assert_eq!(
            retained_boundaries(&retained),
            deep,
            "a persistence-OFF turn retains exactly what it retained before the ladder existed"
        );
    }

    /// Persistence-OFF, the decode-publish axis. Same construction as
    /// `gemma4_sliding_published_boundaries_are_unchanged_when_the_ladder_is_off`:
    /// the OFF caps are handed the real rung grid, so refusing to publish is the
    /// policy's job and not an accident of an empty list.
    #[test]
    fn gemma4_persistence_off_decode_publishes_only_the_cadence() {
        let config = twelve_b_sliding_config();
        let anchors =
            Gemma4SlidingAnchorRungs::from_slice(&gemma4_sliding_cold_anchor_rungs(&config, 16, 2));
        assert_eq!(anchors.as_slice(), &[64, 256, 1024, 4096]);
        let off = Gemma4SlidingRetentionCaps {
            anchors,
            ..gemma4_sliding_retention_caps_for_override(&config, 16, false, None)
        };
        assert_eq!(
            decode_published_boundaries(&config, 16, off, 1, 1200),
            vec![1024],
            "with no cold tier the decode cadence is untouched: no rung may fire"
        );
    }

    /// A real [`ColdTierContext`] carrying `sidecar_policy`, plus the temp root
    /// it owns so the caller can remove it.
    ///
    /// Deliberately the real type with a real manager rather than a stand-in:
    /// the thing under test is production's own derivation, and a stand-in for
    /// `ColdTierContext` would be a second implementation of the fact these
    /// tests exist to pin. Opening the manager touches a directory and nothing
    /// else — no block is ever written.
    fn cold_tier_context_with(
        label: &str,
        sidecar_policy: Option<mlx_paged_attn::ColdSidecarPolicy>,
    ) -> (ColdTierContext, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "mlx-gemma4-sliding-ladder-{}-{label}",
            std::process::id()
        ));
        let manager = mlx_paged_attn::ColdCacheManager::open_default_at(root.clone())
            .expect("temp-dir cold cache must open");
        (
            ColdTierContext {
                manager: Arc::new(manager),
                fingerprint: mlx_paged_attn::ColdCacheFingerprint::from_components([
                    b"gemma4-sliding-ladder-test".as_slice(),
                ]),
                sidecar_policy,
            },
            root,
        )
    }

    /// A cold tier belonging to a DIFFERENT hybrid family (qwen3.5's GDN
    /// recurrent state). Present so "wants a ladder" cannot be satisfied by
    /// "some sidecar policy exists": gemma4's rungs are readable only by
    /// gemma4's own sliding capture.
    fn gdn_sidecar_policy() -> mlx_paged_attn::ColdSidecarPolicy {
        mlx_paged_attn::ColdSidecarPolicy::new(mlx_paged_attn::ColdSidecarLayout {
            group: mlx_paged_attn::ColdGroup::GdnState,
            boundary_tokens: 0,
            num_layers: 4,
            tensors_per_layer: 2,
            dtype: "BFloat16".to_string(),
            dims: vec![1, 8, 128],
            bytes_per_tensor: 8 * 128 * 2,
        })
        .expect("a GdnState sidecar policy must validate")
    }

    /// The master switch, executed for real.
    ///
    /// `gemma4_sliding_cold_ladder_wanted` decides FOUR things — whether the
    /// prefill publishes rungs, whether a stored entry is FLAGGED a rung,
    /// whether decode publishes off-cadence, and whether the ladder byte cap
    /// runs — and until now nothing ran it. Every test built its caps by handing
    /// `gemma4_sliding_retention_caps_for_override` an explicit boolean, so
    /// making this predicate return `false` unconditionally left the cold tier
    /// completely inert with the whole suite green.
    ///
    /// What this cannot reach is the `paged_adapter -> cold_tier()` borrow in
    /// `Gemma4Inner::gemma4_sliding_retention_caps_for_turn`, which needs a
    /// constructed adapter (Metal) and a loaded checkpoint;
    /// `paged_kv_cache_adapter::tests::cold_tier_defaults_none_and_holds_context_across_resets`
    /// pins that accessor, and
    /// `gemma4_sliding_ladder_intent_has_one_production_source` pins that the
    /// borrow is the ONLY thing that call site contributes.
    #[test]
    fn gemma4_sliding_cold_ladder_wants_a_ladder_only_for_a_sliding_window_sidecar() {
        let config = twelve_b_sliding_config();
        let block_size = 16u32;

        assert!(
            !gemma4_sliding_cold_ladder_wanted(None),
            "no cold tier: nothing could ever read a rung"
        );

        let (no_policy, root_no_policy) = cold_tier_context_with("no-policy", None);
        assert!(
            !gemma4_sliding_cold_ladder_wanted(Some(&no_policy)),
            "a family whose whole per-token state lives in the paged pool (dense qwen3) \
             installs no sidecar policy, so it has no auxiliary state to anchor"
        );

        let (gdn, root_gdn) = cold_tier_context_with("gdn", Some(gdn_sidecar_policy()));
        assert!(
            !gemma4_sliding_cold_ladder_wanted(Some(&gdn)),
            "another family's sidecar group must not switch gemma4's ladder on"
        );

        let policy = sliding_sidecar::policy(&config).expect("the 12B geometry has a policy");
        assert_eq!(policy.group(), mlx_paged_attn::ColdGroup::SlidingWindow);
        let (sliding, root_sliding) = cold_tier_context_with("sliding", Some(policy));
        assert!(
            gemma4_sliding_cold_ladder_wanted(Some(&sliding)),
            "THE persist turn: a SlidingWindow sidecar policy is exactly what \
             capture_gemma4_sliding_cold_sidecar needs a rung for"
        );

        // And the caps every publish/retention seam reads follow it, with no
        // second opinion about whether this is a persist turn.
        assert_eq!(
            gemma4_sliding_retention_caps_for_cold_tier(&config, Some(&sliding), block_size),
            gemma4_sliding_retention_caps(&config, block_size, true),
            "a SlidingWindow cold tier must produce the Ladder arm"
        );
        assert_eq!(
            gemma4_sliding_retention_caps_for_cold_tier(&config, None, block_size),
            gemma4_sliding_retention_caps(&config, block_size, false),
            "no cold tier must produce the pre-ladder arm verbatim"
        );
        assert!(
            gemma4_sliding_retention_caps_for_cold_tier(&config, Some(&sliding), block_size)
                .wants_ladder()
        );
        assert!(
            !gemma4_sliding_retention_caps_for_cold_tier(&config, Some(&gdn), block_size)
                .wants_ladder()
        );

        for root in [root_no_policy, root_gdn, root_sliding] {
            let _ = std::fs::remove_dir_all(&root);
        }
    }

    /// The decode publisher's DECISION, driven through production's own caps
    /// derivation rather than through the free predicate.
    ///
    /// Both decode tests above call `gemma4_sliding_decode_publishes_checkpoint`
    /// directly with hand-built caps, so hard-coding `want_ladder` to `false`
    /// inside the decode publisher reverted it to cadence-only — defect B fully
    /// un-fixed in production — with both of them still green. This one starts
    /// from a `ColdTierContext`, which is what the adapter actually hands over.
    #[test]
    fn gemma4_sliding_decode_boundary_plan_reads_the_turns_real_cold_tier() {
        let config = twelve_b_sliding_config();
        let block_size = 16u32;
        let policy = sliding_sidecar::policy(&config).expect("the 12B geometry has a policy");
        let (persisting, root) = cold_tier_context_with("decode-plan", Some(policy));

        // 256 is a rung and NOT a cadence multiple (the cadence is one whole
        // window, 1024). It is published by decode or by nothing at all.
        assert_eq!(
            gemma4_sliding_decode_boundary_plan(&config, Some(&persisting), block_size, 256),
            Some(Gemma4SlidingDecodeBoundary {
                prefix_len: 256,
                block_size,
                checkpoint_interval: 1024,
                on_anchor_rung: true,
            }),
            "a persist turn must publish the sub-window rung the cadence skips"
        );
        assert_eq!(
            gemma4_sliding_decode_boundary_plan(&config, None, block_size, 256),
            None,
            "persistence-OFF keeps the bare cadence: the same cursor publishes nothing"
        );

        for (label, cold) in [("persist", Some(&persisting)), ("off", None)] {
            assert_eq!(
                gemma4_sliding_decode_boundary_plan(&config, cold, block_size, 1024),
                Some(Gemma4SlidingDecodeBoundary {
                    prefix_len: 1024,
                    block_size,
                    checkpoint_interval: 1024,
                    on_anchor_rung: cold.is_some(),
                }),
                "{label}: the cadence fires on both arms; only the rung FLAG differs"
            );
            assert_eq!(
                gemma4_sliding_decode_boundary_plan(&config, cold, block_size, 300),
                None,
                "{label}: an ordinary decode step publishes nothing"
            );
            assert_eq!(
                gemma4_sliding_decode_boundary_plan(&config, cold, block_size, 0),
                None,
                "{label}: an empty request publishes nothing"
            );
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    /// The decode publisher now tests the cheap cadence/grid arithmetic BEFORE
    /// deriving caps, because HEAD returned early on every non-boundary step
    /// and deriving caps walks `num_hidden_layers` three times over (96
    /// `String == "full_attention"` compares on the 12B) plus an env `OnceLock`
    /// read — per decode token, on every gemma4 paged turn, persistence-OFF
    /// included.
    ///
    /// A short-circuit that changes WHICH cursors publish would change emitted
    /// tokens, so pin the two against each other on every cursor across four
    /// windows, on both arms.
    #[test]
    fn gemma4_sliding_decode_plan_matches_the_publish_predicate_on_every_cursor() {
        let config = twelve_b_sliding_config();
        let block_size = 16u32;
        let policy = sliding_sidecar::policy(&config).expect("the 12B geometry has a policy");
        let (persisting, root) = cold_tier_context_with("decode-equiv", Some(policy));
        let interval = gemma4_sliding_decode_checkpoint_interval(&config, block_size);

        for (label, cold, want_ladder) in
            [("persist", Some(&persisting), true), ("off", None, false)]
        {
            let caps = gemma4_sliding_retention_caps(&config, block_size, want_ladder);
            let mut published = Vec::new();
            for cursor in 0..=4200u32 {
                let planned =
                    gemma4_sliding_decode_boundary_plan(&config, cold, block_size, cursor);
                assert_eq!(
                    planned.is_some(),
                    gemma4_sliding_decode_publishes_checkpoint(cursor, interval, caps),
                    "{label}: cursor {cursor} disagrees with the publish predicate"
                );
                if planned.is_some() {
                    published.push(cursor);
                }
            }
            let expected: Vec<u32> = if want_ladder {
                vec![64, 256, 1024, 2048, 3072, 4096]
            } else {
                vec![1024, 2048, 3072, 4096]
            };
            assert_eq!(
                published, expected,
                "{label}: the short-circuit must not move the published set"
            );
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    /// The anchor-grid pre-test is only allowed to be CHEAP, never selective:
    /// it must accept every boundary any published rung set could contain, or
    /// the decode publisher silently drops rungs on some geometry.
    #[test]
    fn gemma4_sliding_anchor_grid_pretest_is_a_superset_of_every_published_rung() {
        for (label, config) in [
            ("12B", twelve_b_sliding_config()),
            ("kv-shared", kv_shared_sliding_config()),
            ("narrow-window", narrow_window_sliding_config()),
            ("all-global", all_global_config()),
        ] {
            for block_size in [8u32, 16, 32] {
                for base_limit in [1usize, 2, 6] {
                    for rung in gemma4_sliding_cold_anchor_rungs(&config, block_size, base_limit) {
                        assert!(
                            gemma4_sliding_prefix_len_is_on_the_anchor_grid(rung, block_size),
                            "{label}: published rung {rung} (block {block_size}, base limit \
                             {base_limit}) is not on the grid the decode fast path screens with"
                        );
                    }
                }
            }
        }
        // ...and it is genuinely a screen, not `true`.
        assert!(!gemma4_sliding_prefix_len_is_on_the_anchor_grid(300, 16));
        assert!(!gemma4_sliding_prefix_len_is_on_the_anchor_grid(16, 16));
        assert!(
            !gemma4_sliding_prefix_len_is_on_the_anchor_grid(16384, 16),
            "the grid stops at GEMMA4_SLIDING_ANCHOR_MAX_RUNGS"
        );
        assert!(!gemma4_sliding_prefix_len_is_on_the_anchor_grid(64, 0));
        assert!(!gemma4_sliding_prefix_len_is_on_the_anchor_grid(0, 16));
    }

    /// Production text of this file: everything above the unit-test module.
    fn production_source() -> &'static str {
        include_str!("model.rs")
            .split_once("#[cfg(test)]\nmod tests {")
            .expect("model.rs must contain its unit-test module")
            .0
    }

    /// Production lines, trimmed, with line comments and `[MLX_TRACE]` format
    /// strings dropped. Doc comments quote these identifiers constantly and a
    /// trace line prints `cold_anchor_rung={}` / names the caps helpers in
    /// prose; neither is code, and counting them makes every guard below a
    /// comment-editing tripwire instead of a structural one.
    fn production_code_lines() -> Vec<&'static str> {
        production_source()
            .lines()
            .map(str::trim)
            .filter(|line| !line.starts_with("//") && !line.contains("[MLX_TRACE]"))
            .collect()
    }

    /// Every production line that could WRITE `cold_anchor_rung`: the field name
    /// followed, after any spacing, by `:` (a field declaration or a struct
    /// literal) or by a LONE `=` (an assignment). `==` and `!=` are reads, and
    /// so is a bare `checkpoint.cold_anchor_rung` in a predicate.
    ///
    /// The `=` arm is the whole reason this is not a substring count. The guard
    /// this replaces was `matches("cold_anchor_rung:").count() == 2`, which sees
    /// struct-literal syntax only — so the shortest way to restore the defect it
    /// exists to prevent slipped straight past it:
    ///
    /// ```text
    ///   if let Some(last) = self.sliding_prefix_checkpoints.back_mut() {
    ///       last.cold_anchor_rung = false;      // no colon: count stays 2
    ///   }
    /// ```
    fn cold_anchor_rung_write_sites() -> Vec<&'static str> {
        production_code_lines()
            .into_iter()
            .filter(|line| {
                line.match_indices("cold_anchor_rung")
                    .any(|(index, needle)| {
                        // A `&mut` borrow hands the field to someone else to
                        // write — `mem::take`/`replace`/`swap` set it without
                        // ever naming an operator here. The borrow reaches the
                        // field through a path (`&mut last.cold_anchor_rung`),
                        // so accept anything between the `&mut` and the field
                        // that is still part of that path.
                        let before = &line[..index];
                        if let Some(borrow) = before.rfind("&mut") {
                            let path = &before[borrow + "&mut".len()..];
                            if path
                                .chars()
                                .all(|c| c.is_alphanumeric() || matches!(c, '_' | '.' | ':' | ' '))
                            {
                                return true;
                            }
                        }
                        let mut rest = line[index + needle.len()..].trim_start().chars();
                        match rest.next() {
                            Some(':') => true,
                            // Plain assignment, but NOT the `==` comparison.
                            Some('=') => rest.next() != Some('='),
                            // Compound assignment. `x &= flag` is the worst of
                            // these: with a value that is false on a normal
                            // turn it clears the flag on every stored entry and
                            // restores the defect this guard exists to prevent,
                            // while naming no `=` of its own.
                            Some('&' | '|' | '^' | '+' | '-' | '*' | '/' | '%') => {
                                rest.next() == Some('=')
                            }
                            _ => false,
                        }
                    })
            })
            .collect()
    }

    /// `cold_anchor_rung` is the ladder's whole eviction ordering, and d134ab3e
    /// shipped four visually identical `cold_anchor_rung: false` literals of
    /// which two were load-bearing and two were dead. A test cannot execute the
    /// publish sites without a GPU and real caches, so what is pinned here is
    /// the structure that makes the derivation unforgeable: production WRITES
    /// the field in exactly two places — the declaration, and the single
    /// derivation in `Gemma4SlidingPrefixCheckpointDraft::into_checkpoint`.
    /// Every publish site hands over a draft, which has no such field.
    ///
    /// Two things this cannot see, both covered elsewhere:
    ///   * a publish site handing `into_checkpoint` the WRONG caps (a literal
    ///     `Gemma4SlidingRetentionCaps::pre_ladder(..)` instead of the turn's) —
    ///     `gemma4_sliding_ladder_intent_has_one_production_source` pins the
    ///     construction sites of the caps themselves;
    ///   * a rung that is never PUBLISHED at all, which is
    ///     `gemma4_sliding_decode_boundary_plan`'s job and is tested against the
    ///     real cold-tier derivation.
    #[test]
    fn gemma4_sliding_anchor_flag_has_exactly_one_writer() {
        assert_eq!(
            cold_anchor_rung_write_sites(),
            vec![
                "cold_anchor_rung: bool,",
                "cold_anchor_rung: caps.wants_ladder() && caps.anchors.contains(self.prefix_len),",
            ],
            "expected the field declaration plus `into_checkpoint`'s derivation and \
             nothing else. A publish site that sets the flag by hand — in struct-literal \
             OR assignment form — can clear it on a real rung, and an unflagged rung is \
             the ladder's PREFERRED eviction victim, i.e. born then immediately evicted"
        );
    }

    /// The production wiring the ladder hangs off, pinned as text because the
    /// two call sites that consume it are only reachable with a GPU and a
    /// loaded checkpoint.
    ///
    /// Both of these mutations disconnect the feature in production and neither
    /// changed a single behavioural test:
    ///
    /// ```text
    ///   prefill orchestrator:  self.gemma4_sliding_retention_caps_for_turn(block_size)
    ///                       -> gemma4_sliding_retention_caps(&config, block_size, false)
    ///   decode publisher:      the same substitution
    /// ```
    ///
    /// Both work by introducing a SECOND place that picks the `want_ladder`
    /// boolean. So what is pinned is that production picks it once: the
    /// bool-taking constructors are called exactly where they are defined to be
    /// called, and the only producer of the boolean is
    /// `gemma4_sliding_cold_ladder_wanted`, whose behaviour
    /// `gemma4_sliding_cold_ladder_wants_a_ladder_only_for_a_sliding_window_sidecar`
    /// tests for real.
    ///
    /// Counts include each function's own definition, so "2" reads as
    /// "defined once, called once".
    #[test]
    fn gemma4_sliding_ladder_intent_has_one_production_source() {
        let code = production_code_lines().join("\n");
        for (needle, expected, why) in [
            (
                "gemma4_sliding_cold_ladder_wanted(",
                3usize,
                "defined once; called from `gemma4_sliding_retention_caps_for_cold_tier` (the \
                 derivation) and from `gemma4_sliding_decode_boundary_plan`'s hot-path screen. \
                 The screen is covered behaviourally by \
                 `gemma4_sliding_decode_boundary_plan_reads_the_turns_real_cold_tier`",
            ),
            (
                "gemma4_sliding_retention_caps(",
                2,
                "defined once, called once (from `gemma4_sliding_retention_caps_for_cold_tier`); \
                 a second call is how both disconnect mutations spell themselves",
            ),
            (
                "gemma4_sliding_retention_caps_for_override(",
                2,
                "defined once, called once (from `gemma4_sliding_retention_caps`); production \
                 must not reach past the env override",
            ),
            (
                "gemma4_sliding_retention_caps_for_cold_tier(",
                3,
                "defined once, called from `gemma4_sliding_retention_caps_for_turn` and from \
                 `gemma4_sliding_decode_boundary_plan`",
            ),
            (
                "Gemma4SlidingRetentionCaps::pre_ladder(",
                1,
                "only `gemma4_sliding_retention_caps_for_override` may build the OFF arm; a \
                 publish site building one by hand would hand `into_checkpoint` caps that \
                 clear the anchor flag on a genuine rung",
            ),
            (
                "Gemma4SlidingRetentionCaps::ladder(",
                2,
                "only `gemma4_sliding_retention_caps_for_override`'s two return sites (the \
                 operator-override arm and the widened arm)",
            ),
            (
                "cold_tier()",
                3,
                "the borrow is the ONLY thing the two GPU-only call sites contribute — \
                 `gemma4_sliding_retention_caps_for_turn` and the decode publisher — plus \
                 `capture_gemma4_sliding_cold_sidecar`, the rungs' one consumer. Passing a \
                 literal `None` in place of one of them disconnects the feature just as \
                 completely as hard-coding `want_ladder`, and leaves the derivation itself \
                 untouched, so no behavioural test can see it",
            ),
        ] {
            assert_eq!(
                code.matches(needle).count(),
                expected,
                "production mentions `{needle}` {} time(s), expected {expected}: {why}",
                code.matches(needle).count()
            );
        }
    }

    /// A speculative / native-MTP gemma4 turn publishes no sliding
    /// decode-boundary checkpoint — not a rung, not even a cadence entry. That
    /// is harmless only because a draft turn has no paged adapter, and so no
    /// cold tier for a rung to serve:
    ///
    /// ```text
    ///   load_from_dir sees a draft  ->  use_block_paged_cache = Some(false)
    ///                               ->  Gemma4Inner::new builds no adapter
    ///                               ->  build_cold_tier_context returns None
    ///                               ->  capture_gemma4_sliding_cold_sidecar returns at line 1
    /// ```
    ///
    /// The load-path half is pinned by
    /// `persistence::tests::{dspark,embedded}_draft_conflicts_with_explicit_paged_cache`.
    /// This is the other half, and the tripwire: the day a draft turn gains a
    /// paged adapter, the sliding decode publisher has to be wired into the
    /// accept loop — and that is NOT a copy of the AR call, because speculative
    /// decode accepts a variable number of tokens per cycle, so the cursor can
    /// step from below a rung to above it without ever landing on it. The
    /// predicate would have to be "a boundary lies in `(previous, current]`",
    /// the same `gemma4_sliding_checkpoint_boundaries_crossed` shape the prefill
    /// chunk walk uses, and the snapshot would have to be sliced back to that
    /// boundary rather than taken at the cursor.
    #[test]
    fn gemma4_draft_decode_paths_never_touch_the_paged_adapter() {
        for (label, source) in [
            ("dspark_decode.rs", include_str!("dspark_decode.rs")),
            ("assistant_decode.rs", include_str!("assistant_decode.rs")),
        ] {
            // Comments name the field to explain why it is absent; code must not.
            let code = source
                .lines()
                .map(str::trim)
                .filter(|line| !line.starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                !code.contains("paged_adapter"),
                "{label} mentions `paged_adapter`. A draft turn used to be flat by \
                 construction, which is the only reason it is safe for the draft decode \
                 loops not to publish a sliding decode-boundary checkpoint. If that has \
                 changed, wire the publisher in — with a crossed-boundary predicate, not \
                 a landed-on-boundary one: a variable accept count can jump the cursor \
                 straight over a rung"
            );
        }
    }

    /// A draft turn's real, production-derived plan: no paged adapter, so the
    /// turn reaches the speculative handler.
    ///
    /// `tiny_inner_with_draft` is the in-crate stand-in for a real draft load —
    /// its config sets `use_block_paged_cache: false`, which is exactly what
    /// `persistence::resolve_gemma4_draft_paged_cache` forces on every draft
    /// checkpoint. This half of the pair asserts the arrangement that works;
    /// [`gemma4_paged_plus_draft_silently_drops_the_draft`] asserts what the
    /// other arrangement does.
    #[test]
    fn gemma4_flat_draft_plan_routes_to_the_speculative_handler() {
        let inner = crate::models::gemma4::dspark_decode::tests::tiny_inner_with_draft();
        let execution = inner.execution_plan();

        assert!(
            execution.paged_attention.is_none(),
            "a draft-carrying Gemma4Inner must build no paged adapter — \
             `use_block_paged_cache: false` in the config is the load-path forcing that \
             keeps it that way, and it is the ONLY reason the draft actually runs"
        );
        let speculative = match execution.speculative {
            Some(s) => s,
            None => panic!("tiny_inner_with_draft carries a draft; the plan must advertise it"),
        };
        assert!(
            !speculative.supports_paged_attention,
            "gemma4 draft proposal/verification is implemented against flat KV only"
        );

        let plan = TurnPlan::resolve(
            execution,
            TurnRequest {
                is_delta: false,
                input_media: MediaCapabilities::NONE,
                context_media: MediaCapabilities::NONE,
                speculative_requested: true,
            },
        );
        assert_eq!(
            plan.decoder,
            DecoderPlan::Speculative(SpeculativeKind::DraftModel),
            "flat + draft + opt-in must select the draft decoder"
        );
        assert_eq!(
            plan.path(),
            TurnPath::Speculative,
            "the draft decoder only runs when `path()` also lands on the speculative \
             handler; `run_paged_turn` has no speculative branch to fall back on"
        );
    }

    /// Paged and draft cannot coexist on a gemma4 turn, and the failure is
    /// SILENT — no error, no log, just an autoregressive decode with a fully
    /// loaded draft that is never stepped. This test states that in executable
    /// form so the invariant is not carried by prose alone.
    ///
    /// ```text
    ///   paged ON + supports_paged_attention:false  -> decoder downgraded to AR
    ///   paged ON + supports_paged_attention:TRUE   -> decoder Speculative, but
    ///                                                 path() tests paged FIRST
    ///   ...both land on TurnPath::Paged -> engine::paged_turn, which never
    ///      reads `plan.decoder` (the only mentions in that file are its own
    ///      test fixture and a JSON key). The `debug_assert!` in
    ///      `run_paged_turn` is compiled out of release.
    /// ```
    ///
    /// The second row is the point. It is the shape a future change produces
    /// when someone reads "draft turns are flat by construction", decides to
    /// lift the restriction, and flips `supports_paged_attention` to `true`
    /// here without touching the load path — and it still does not run. Paged
    /// speculative decode needs a branch inside `engine::paged_turn`, not a
    /// flag flip.
    #[test]
    fn gemma4_paged_plus_draft_silently_drops_the_draft() {
        let inner = crate::models::gemma4::dspark_decode::tests::tiny_inner_with_draft();
        let flat = inner.execution_plan();
        let speculative = match flat.speculative {
            Some(s) => s,
            None => panic!("tiny_inner_with_draft carries a draft; the plan must advertise it"),
        };

        // Exactly what dropping the load-path forcing produces: the same draft
        // plan, plus the paged adapter `unwrap_or(true)` would have built.
        let paged_and_draft = ExecutionPlan {
            paged_attention: Some(PagedAttentionPlan {
                supports_delta: true,
            }),
            ..flat
        };

        let request = TurnRequest {
            is_delta: false,
            input_media: MediaCapabilities::NONE,
            context_media: MediaCapabilities::NONE,
            speculative_requested: true,
        };

        let plan = TurnPlan::resolve(paged_and_draft, request);
        assert_eq!(
            plan.decoder,
            DecoderPlan::Autoregressive,
            "paged ON + a paged-incapable proposer must downgrade to plain AR — the draft \
             is loaded and resident, and not one draft forward runs"
        );
        assert_eq!(
            plan.path(),
            TurnPath::Paged,
            "`path()` checks paged BEFORE speculative, so the turn goes to \
             `engine::paged_turn`, which never reads `plan.decoder`. Silent loss of the \
             measured draft speedup — the exact outcome the explicit-`true` config is \
             hard-errored for. Keep the load path forcing paged OFF whenever a draft \
             resolves"
        );

        // Flipping the capability flag alone does NOT fix it.
        let flag_flipped = ExecutionPlan {
            speculative: Some(SpeculativePlan {
                supports_paged_attention: true,
                ..speculative
            }),
            ..paged_and_draft
        };
        let flipped_plan = TurnPlan::resolve(flag_flipped, request);
        assert_eq!(
            flipped_plan.decoder,
            DecoderPlan::Speculative(SpeculativeKind::DraftModel),
            "with the flag set, resolve() no longer downgrades the decoder"
        );
        assert_eq!(
            flipped_plan.path(),
            TurnPath::Paged,
            "...and the turn STILL routes to `engine::paged_turn`, which has no \
             speculative branch, so the draft is still never stepped — now without even \
             the AR downgrade to make the plan honest. Setting \
             `supports_paged_attention: true` is not the missing piece; a speculative \
             branch inside `engine::paged_turn` (plus a CROSSED-boundary sliding \
             checkpoint predicate, since a variable accept count jumps rungs) is"
        );
    }

    /// A finished conversation's anchors must not squat. `Ladder` defers
    /// anchors, it does not protect them: once no non-anchor is left, the
    /// first anchor that is NOT an ancestor of the newest entry goes.
    ///
    /// The interleaving is what makes this discriminating, and it is the
    /// interleaving several conversations multiplexed over one model actually
    /// produce. `B@64` is pushed FIRST, so a victim rule that only walks the
    /// deque in publish order takes it — even though it is a strict ancestor
    /// of the entry being published right now, i.e. the single most reusable
    /// thing in the store. Only the ancestor test skips past it to `A@64`.
    ///
    /// ```text
    ///   store (oldest first)   B@64   A@64   A@256   <- push B@256, limit 3
    ///   with the ancestor test        ----          evicted: A@64
    ///   publish order only     ----                 evicted: B@64   (wrong)
    /// ```
    #[test]
    fn gemma4_sliding_ladder_evicts_a_stale_lineage_anchor() {
        let block_size = 16;
        let caps = Gemma4SlidingRetentionCaps::ladder(
            3,
            Gemma4SlidingAnchorRungs::from_slice(&[64, 256, 1024]),
            Gemma4SlidingCheckpointBytes::for_config(&twelve_b_sliding_config()),
        );
        let lineage_a: Vec<u32> = (0..4096).collect();
        let lineage_b: Vec<u32> = (0..4096).map(|token| token + 90_000).collect();
        let from_b = |checkpoint: &Gemma4SlidingPrefixCheckpoint| {
            checkpoint.tokens.first().copied().unwrap_or(0) >= 90_000
        };

        let mut retained: VecDeque<Gemma4SlidingPrefixCheckpoint> = VecDeque::new();
        for (rung, tokens) in [
            (64u32, &lineage_b),
            (64, &lineage_a),
            (256, &lineage_a),
            (256, &lineage_b),
        ] {
            upsert_gemma4_sliding_prefix_checkpoint(
                &mut retained,
                sliding_checkpoint_at(rung, block_size, tokens),
                caps,
                false,
            );
        }

        let survivors: Vec<(u32, bool)> = retained
            .iter()
            .map(|checkpoint| (checkpoint.prefix_len, from_b(checkpoint)))
            .collect();
        assert_eq!(
            survivors,
            vec![(64, true), (256, false), (256, true)],
            "the stale lineage's anchor must go, not the newest entry's own ancestor"
        );
    }

    #[test]
    fn gemma4_prompt_boundary_retains_a_across_a_b_a_image_identity() {
        let tokens: Vec<u32> = (1..=12).collect();
        let block_size = 4;
        let prefix_len = 8;
        let a_keys = engine::build_paged_extra_keys(tokens.len(), block_size, &[(4, 0xAAAA)]);
        let b_keys = engine::build_paged_extra_keys(tokens.len(), block_size, &[(4, 0xBBBB)]);
        let a_hash = compute_gemma4_paged_prefix_block_hash_with_keys(
            &tokens, prefix_len, block_size, &a_keys, 0,
        )
        .expect("A image-aware prefix hash");
        let b_hash = compute_gemma4_paged_prefix_block_hash_with_keys(
            &tokens, prefix_len, block_size, &b_keys, 0,
        )
        .expect("B image-aware prefix hash");
        assert_ne!(a_hash, b_hash);

        let checkpoint = |final_block_hash| Gemma4SlidingPrefixCheckpointDraft {
            prefix_len,
            block_size,
            final_block_hash,
            protected_image_prompt_boundary: true,
            tokens: tokens[..prefix_len as usize].to_vec(),
            snapshots: Vec::new(),
        };
        let bytes = Gemma4SlidingCheckpointBytes::for_config(&twelve_b_sliding_config());
        let mut retained = VecDeque::new();
        upsert_gemma4_sliding_prefix_checkpoint(
            &mut retained,
            checkpoint(a_hash),
            Gemma4SlidingRetentionCaps::pre_ladder(8, bytes),
            false,
        );
        let mut latest_prompt_boundary = checkpoint(a_hash);
        assert_eq!(latest_prompt_boundary.final_block_hash, a_hash);
        upsert_gemma4_sliding_prefix_checkpoint(
            &mut retained,
            checkpoint(b_hash),
            Gemma4SlidingRetentionCaps::pre_ladder(8, bytes),
            false,
        );
        latest_prompt_boundary = checkpoint(b_hash);

        assert_eq!(latest_prompt_boundary.final_block_hash, b_hash);
        assert!(
            retained
                .iter()
                .any(|entry| entry.final_block_hash == a_hash)
        );
        assert!(
            retained
                .iter()
                .any(|entry| entry.final_block_hash == b_hash)
        );
        upsert_gemma4_sliding_prefix_checkpoint(
            &mut retained,
            Gemma4SlidingPrefixCheckpointDraft {
                prefix_len,
                block_size,
                final_block_hash: 0xDEC0DE,
                protected_image_prompt_boundary: false,
                tokens: tokens[..prefix_len as usize].to_vec(),
                snapshots: Vec::new(),
            },
            Gemma4SlidingRetentionCaps::pre_ladder(2, bytes),
            false,
        );
        assert_eq!(retained.len(), 2);
        assert!(
            retained
                .iter()
                .any(|entry| entry.final_block_hash == a_hash),
            "decode checkpoints must not evict protected image A"
        );
        assert!(
            retained
                .iter()
                .any(|entry| entry.final_block_hash == b_hash),
            "decode checkpoints must not evict protected image B"
        );
        let restored_a = retained.iter().rev().find(|entry| {
            entry.prefix_len == prefix_len
                && entry.tokens == tokens[..prefix_len as usize]
                && entry.final_block_hash == a_hash
        });
        assert!(
            restored_a.is_some(),
            "A must remain restorable after B replaces the latest singleton boundary"
        );
    }

    #[test]
    fn image_expansion_requires_template_placeholder() {
        let image_token_id = 258880u32;
        let boi = 255999u32;
        let eoi = 258882u32;
        let image = ProcessedGemma4Image {
            pixel_values: MxArray::zeros(&[1, 1], Some(DType::Float32)).unwrap(),
            num_soft_tokens: 3,
            position_ids: None,
        };
        let error = expand_image_tokens(
            &[2, 9],
            std::slice::from_ref(&image),
            image_token_id,
            boi,
            eoi,
        )
        .expect_err("missing checkpoint-template image placeholder must fail");
        assert!(error.to_string().contains("0 image placeholder(s)"));
    }

    #[test]
    fn stream_dispatch_promotes_channel_only_output_to_visible_text() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sender = StreamSender(&tx);
        let mut state = Gemma4StreamDispatchState::default();

        state.dispatch_segments(
            vec![StreamSegment::Reasoning("final answer".into())],
            &sender,
        );
        assert!(rx.try_recv().is_err());

        state.finish(&sender);
        let chunk = rx.try_recv().unwrap().unwrap();
        assert_eq!(chunk.text, "final answer");
        assert_eq!(chunk.is_reasoning, Some(false));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn stream_dispatch_keeps_truncated_prompted_channel_as_reasoning() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sender = StreamSender(&tx);
        let mut state = Gemma4StreamDispatchState::new(true);

        state.dispatch_segments(
            vec![StreamSegment::Reasoning("unfinished plan".into())],
            &sender,
        );
        assert!(rx.try_recv().is_err());

        state.finish(&sender);
        let chunk = rx.try_recv().unwrap().unwrap();
        assert_eq!(chunk.text, "unfinished plan");
        assert_eq!(chunk.is_reasoning, Some(true));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn stream_dispatch_keeps_reasoning_when_visible_text_follows() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sender = StreamSender(&tx);
        let mut state = Gemma4StreamDispatchState::default();

        state.dispatch_segments(
            vec![
                StreamSegment::Reasoning("scratch".into()),
                StreamSegment::Text("answer".into()),
            ],
            &sender,
        );
        state.finish(&sender);

        let reasoning = rx.try_recv().unwrap().unwrap();
        assert_eq!(reasoning.text, "scratch");
        assert_eq!(reasoning.is_reasoning, Some(true));

        let text = rx.try_recv().unwrap().unwrap();
        assert_eq!(text.text, "answer");
        assert_eq!(text.is_reasoning, Some(false));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn stream_dispatch_keeps_reasoning_when_tool_call_follows() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sender = StreamSender(&tx);
        let mut state = Gemma4StreamDispatchState::default();

        state.dispatch_segments(
            vec![
                StreamSegment::Reasoning("scratch".into()),
                StreamSegment::ToolCall(crate::tools::ToolCallResult::ok(
                    "tool".into(),
                    serde_json::json!({}),
                    String::new(),
                )),
            ],
            &sender,
        );
        state.finish(&sender);

        let reasoning = rx.try_recv().unwrap().unwrap();
        assert_eq!(reasoning.text, "scratch");
        assert_eq!(reasoning.is_reasoning, Some(true));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn promote_channel_only_output_moves_thinking_to_text() {
        let mut parsed = parse_gemma4_output("<|channel>thought\nvisible answer<channel|>");
        promote_channel_only_output(&mut parsed, false);

        assert_eq!(parsed.text, "visible answer");
        assert!(parsed.thinking.is_none());
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn seeded_channel_truncation_is_not_promoted_to_visible_text() {
        let mut parsed =
            crate::models::gemma4::output_parser::parse_gemma4_output_with_open_channel(
                "unfinished plan",
                true,
            );
        promote_channel_only_output(&mut parsed, true);

        assert!(parsed.text.is_empty());
        assert_eq!(parsed.thinking.as_deref(), Some("unfinished plan"));
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn sliding_mask_is_valid_for_bf16_gqa_attention() {
        let q = MxArray::zeros(&[1, 4, 4, 16], Some(DType::BFloat16)).unwrap();
        let k = MxArray::zeros(&[1, 1, 6, 16], Some(DType::BFloat16)).unwrap();
        let v = MxArray::zeros(&[1, 1, 6, 16], Some(DType::BFloat16)).unwrap();
        let mask = create_sliding_mask(4, 2, 3).unwrap();

        assert_eq!(mask.shape_at(0).unwrap(), 1);
        assert_eq!(mask.shape_at(1).unwrap(), 1);
        assert_eq!(mask.shape_at(2).unwrap(), 4);
        assert_eq!(mask.shape_at(3).unwrap(), 6);

        let out = crate::array::scaled_dot_product_attention(&q, &k, &v, 1.0, Some(&mask)).unwrap();
        let values = out.to_float32().unwrap();
        assert_eq!(values.len(), 4 * 4 * 16);
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sliding_mask_offset_uses_rotating_window_view() {
        assert_eq!(sliding_mask_offset_for_chunk(512, 16, 1024), None);
        assert_eq!(sliding_mask_offset_for_chunk(512, 528, 1024), Some(528));
        assert_eq!(sliding_mask_offset_for_chunk(512, 43_688, 1024), Some(1024));
        assert_eq!(sliding_mask_offset_for_chunk(2048, 0, 1024), Some(0));
        assert_eq!(sliding_mask_offset_for_chunk(1, 4096, 1024), None);
    }

    #[test]
    fn test_gemma4_paged_prefill_body_chunk_size_honors_configured_size() {
        assert_eq!(
            super::gemma4_paged_prefill_body_chunk_size(4096, 27_938),
            4096
        );
        assert_eq!(
            super::gemma4_paged_prefill_body_chunk_size(512, 27_938),
            512
        );
        assert_eq!(
            super::gemma4_paged_prefill_body_chunk_size(0, 27_938),
            super::GEMMA4_PREFILL_STEP_SIZE as usize
        );
        assert_eq!(
            super::gemma4_paged_prefill_body_chunk_size(0, 127),
            127,
            "the default bound must not pad a short final chunk"
        );
    }

    #[test]
    fn test_gemma4_paged_prefill_body_chunk_plan_caps_v2_aux() {
        let plan = super::gemma4_paged_prefill_body_chunk_plan(
            8192,
            27_938,
            16,
            16,
            1,
            512,
            super::Gemma4PagedPrefillRoutePolicy::ForceVarlen,
        )
        .unwrap();
        assert_eq!(plan.first().unwrap().len, 8192);
        assert!(plan.iter().any(|chunk| chunk.capped_by_v2_aux_limit));

        let mut expected_start = 0usize;
        let mut expected_position = 16u32;
        for chunk in &plan {
            assert_eq!(chunk.start, expected_start);
            assert_eq!(chunk.first_position, expected_position);
            assert!(super::gemma4_paged_prefill_chunk_route_is_aux_safe(
                chunk.len,
                chunk.first_position,
                16,
                1,
                512,
                super::Gemma4PagedPrefillRoutePolicy::ForceVarlen,
            ));
            expected_start += chunk.len;
            expected_position += chunk.len as u32;
        }
        assert_eq!(expected_start, 27_938);

        let forced_sdpa = super::gemma4_paged_prefill_body_chunk_plan(
            8192,
            27_938,
            16,
            16,
            1,
            512,
            super::Gemma4PagedPrefillRoutePolicy::NonV2,
        )
        .unwrap();
        assert_eq!(forced_sdpa.len(), 4);
        assert_eq!(forced_sdpa[0].len, 8192);
        assert!(
            forced_sdpa
                .iter()
                .all(|chunk| !chunk.capped_by_v2_aux_limit)
        );

        let auto = super::gemma4_paged_prefill_body_chunk_plan(
            8192,
            27_938,
            16,
            16,
            1,
            512,
            super::Gemma4PagedPrefillRoutePolicy::Auto,
        )
        .unwrap();
        assert_eq!(auto.len(), 4);
        assert!(
            auto.iter().all(|chunk| !chunk.capped_by_v2_aux_limit),
            "auto must keep full compute chunks when its safe pre-plan selects SDPA"
        );
    }

    #[test]
    fn test_gemma4_sliding_restore_chunk_plan_avoids_singletons() {
        let mut plan = super::gemma4_paged_prefill_body_chunk_plan(
            4,
            9,
            0,
            16,
            1,
            512,
            super::Gemma4PagedPrefillRoutePolicy::NonV2,
        )
        .unwrap();
        assert_eq!(
            plan.iter().map(|chunk| chunk.len).collect::<Vec<_>>(),
            vec![4, 4, 1]
        );

        super::gemma4_coalesce_single_token_restore_chunks(&mut plan);
        assert_eq!(
            plan.iter().map(|chunk| chunk.len).collect::<Vec<_>>(),
            vec![4, 5]
        );
        assert_eq!(plan[1].start, 4);
        assert_eq!(plan[1].first_position, 4);

        let mut one_token_chunks = super::gemma4_paged_prefill_body_chunk_plan(
            1,
            5,
            0,
            16,
            1,
            512,
            super::Gemma4PagedPrefillRoutePolicy::NonV2,
        )
        .unwrap();
        super::gemma4_coalesce_single_token_restore_chunks(&mut one_token_chunks);
        assert_eq!(
            one_token_chunks
                .iter()
                .map(|chunk| chunk.len)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert_eq!(one_token_chunks[1].first_position, 2);
    }

    #[test]
    fn test_gemma4_paged_prefill_chunk_plan_splits_prompt_cache_boundary() {
        let mut plan = super::gemma4_paged_prefill_body_chunk_plan(
            1024,
            1432,
            44_320,
            16,
            1,
            512,
            super::Gemma4PagedPrefillRoutePolicy::NonV2,
        )
        .unwrap();
        super::gemma4_split_body_chunk_plan_at_position(&mut plan, 45_744);
        assert_eq!(
            plan.iter().map(|chunk| chunk.len).collect::<Vec<_>>(),
            vec![1024, 400, 8]
        );
        assert_eq!(
            plan.iter()
                .map(|chunk| chunk.first_position)
                .collect::<Vec<_>>(),
            vec![44_320, 45_344, 45_744]
        );
        assert_eq!(
            plan.iter().map(|chunk| chunk.start).collect::<Vec<_>>(),
            vec![0, 1024, 1424]
        );

        let mut unchanged = plan.clone();
        super::gemma4_split_body_chunk_plan_at_position(&mut unchanged, 45_344);
        assert_eq!(unchanged, plan);
    }

    #[test]
    fn test_gemma4_paged_prefill_chunk_plan_is_independent_of_checkpoint_cadence() {
        let plan = super::gemma4_paged_prefill_body_chunk_plan(
            2048,
            3000,
            16,
            16,
            1,
            512,
            super::Gemma4PagedPrefillRoutePolicy::NonV2,
        )
        .unwrap();

        assert_eq!(
            plan.iter().map(|chunk| chunk.len).collect::<Vec<_>>(),
            vec![2048, 952]
        );
        assert_eq!(
            plan.iter()
                .map(|chunk| chunk.first_position)
                .collect::<Vec<_>>(),
            vec![16, 2064]
        );
        assert_eq!(
            plan.iter().map(|chunk| chunk.start).collect::<Vec<_>>(),
            vec![0, 2048]
        );

        let capped = super::gemma4_paged_prefill_body_chunk_plan(
            512,
            1600,
            768,
            16,
            1,
            512,
            super::Gemma4PagedPrefillRoutePolicy::NonV2,
        )
        .unwrap();
        assert_eq!(
            capped.iter().map(|chunk| chunk.len).collect::<Vec<_>>(),
            vec![512, 512, 512, 64]
        );
        assert!(capped.iter().all(|chunk| chunk.len <= 512));
    }

    #[test]
    fn test_gemma4_sliding_checkpoint_cadence_crosses_unaligned_compute_chunk() {
        assert_eq!(
            super::gemma4_sliding_checkpoint_boundaries_crossed(16, 2064, 1024),
            vec![1024, 2048],
            "a cache hit at 16 followed by one 2K compute chunk must publish both cadence points"
        );
        assert_eq!(
            super::gemma4_sliding_checkpoint_boundaries_crossed(2064, 3016, 1024),
            Vec::<u32>::new()
        );
        assert_eq!(
            super::gemma4_sliding_checkpoint_boundaries_crossed(2064, 4096, 1024),
            vec![3072, 4096]
        );
    }

    #[test]
    fn test_gemma4_sliding_restore_default_is_checkpoint_bounded() {
        let cfg = super::Gemma4Config {
            sliding_window: 1024,
            ..paged_tiny_config(Some(true))
        };

        assert_eq!(
            super::gemma4_large_sliding_restore_suppression_limit_for_override(
                &cfg, 16, None, 1024
            ),
            None
        );
        assert_eq!(
            super::gemma4_large_sliding_restore_suppression_limit_for_override(
                &cfg, 16, None, 24_336
            ),
            Some(super::Gemma4SlidingRestoreSuppression {
                limit: 1024,
                source: "default"
            })
        );
    }

    #[test]
    fn test_gemma4_sliding_restore_env_limit_overrides_default() {
        let cfg = super::Gemma4Config {
            sliding_window: 1024,
            ..paged_tiny_config(Some(true))
        };

        assert_eq!(
            super::parse_gemma4_sliding_restore_limit("32768"),
            Some(super::Gemma4SlidingRestoreLimitOverride::Cap(32_768))
        );
        assert_eq!(
            super::parse_gemma4_sliding_restore_limit(" 44512 "),
            Some(super::Gemma4SlidingRestoreLimitOverride::Cap(44_512))
        );
        assert_eq!(super::parse_gemma4_sliding_restore_limit(""), None);
        assert_eq!(
            super::parse_gemma4_sliding_restore_limit("off"),
            Some(super::Gemma4SlidingRestoreLimitOverride::Uncapped)
        );

        assert_eq!(
            super::gemma4_large_sliding_restore_suppression_limit_for_override(
                &cfg,
                16,
                Some(super::Gemma4SlidingRestoreLimitOverride::Cap(32_768)),
                32_768
            ),
            None
        );
        assert_eq!(
            super::gemma4_large_sliding_restore_suppression_limit_for_override(
                &cfg,
                16,
                Some(super::Gemma4SlidingRestoreLimitOverride::Cap(32_768)),
                44_512
            ),
            Some(super::Gemma4SlidingRestoreSuppression {
                limit: 32_768,
                source: "env"
            })
        );
        assert_eq!(
            super::gemma4_large_sliding_restore_suppression_limit_for_override(
                &cfg,
                16,
                Some(super::Gemma4SlidingRestoreLimitOverride::Uncapped),
                1_000_000
            ),
            None
        );
    }

    #[test]
    fn test_ple_oov_masking() {
        // Simulate token IDs where some exceed PLE vocab or are negative
        let input_ids = MxArray::from_int32(&[5, 100, 262143, 0, -1], &[1, 5]).unwrap();
        let ple_vocab = 262144i32; // PLE vocab size

        let ple_vocab_arr = MxArray::scalar_int(ple_vocab).unwrap();
        let zero = MxArray::scalar_int(0).unwrap();
        let valid_mask = input_ids
            .greater_equal(&zero)
            .unwrap()
            .logical_and(&input_ids.less(&ple_vocab_arr).unwrap())
            .unwrap();
        let masked_ids = valid_mask.where_(&input_ids, &zero).unwrap();

        masked_ids.eval();
        // IDs within range: unchanged. IDs out of range (negative): mapped to 0.
        assert_eq!(masked_ids.item_at_int32(0).unwrap(), 5); // in range
        assert_eq!(masked_ids.item_at_int32(1).unwrap(), 100); // in range
        // 262143 < 262144, so it's valid
        assert_eq!(masked_ids.item_at_int32(2).unwrap(), 262143);
        assert_eq!(masked_ids.item_at_int32(3).unwrap(), 0); // in range (0 is valid)
        assert_eq!(masked_ids.item_at_int32(4).unwrap(), 0); // -1 is OOV, mapped to 0
    }

    /// Tiny Gemma4 config compatible with `LayerKVPool`'s validate
    /// constraints (head_size in {32, 64, 96, 128, 256}, FP8 off, etc.).
    /// `head_dim = 32`, num_kv_heads = 2, no PLE/MoE/vision/sharing.
    #[cfg(test)]
    fn paged_tiny_config(use_block_paged: Option<bool>) -> super::Gemma4Config {
        super::Gemma4Config {
            persist_paged_cache: None,
            vocab_size: 100,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 32,
            intermediate_size: 64,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            max_position_embeddings: 128,
            sliding_window: 128,
            // All-global so the uniform paged pool's head_dim choice
            // matches every layer trivially.
            layer_types: vec!["full_attention".to_string(), "full_attention".to_string()],
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            partial_rotary_factor: 0.25,
            global_num_key_value_heads: None,
            global_head_dim: None,
            attention_k_eq_v: false,
            is_unified: false,
            use_bidirectional_attention: None,
            final_logit_softcapping: None,
            per_layer_input_embeds: false,
            hidden_size_per_layer_input: None,
            vocab_size_per_layer_input: None,
            pad_token_id: 0,
            eos_token_ids: vec![1],
            bos_token_id: 2,
            attention_bias: false,
            use_double_wide_mlp: false,
            num_kv_shared_layers: None,
            default_temperature: None,
            default_top_k: None,
            default_top_p: None,
            enable_moe_block: false,
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
            vision_config: None,
            unified_vision_config: None,
            image_token_id: None,
            boi_token_id: None,
            eoi_token_id: None,
            vision_soft_tokens_per_image: None,
            has_audio: false,
            audio_token_id: None,
            boa_token_id: None,
            eoa_token_id: None,
            audio_samples_per_token: None,
            paged_cache_memory_mb: Some(256),
            paged_block_size: Some(16),
            use_block_paged_cache: use_block_paged,
        }
    }

    /// `use_block_paged_cache` defaults to `None` when absent from the
    /// JSON config — guards against silently switching the storage
    /// backend on existing Gemma4 checkpoints.
    ///
    /// Pure-CPU; no MLX runtime needed.
    #[test]
    fn test_use_block_paged_cache_defaults_to_none_via_serde() {
        let json = serde_json::json!({
            "vocab_size": 0,
            "hidden_size": 0,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "intermediate_size": 1,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "max_position_embeddings": 2048,
        });
        let cfg: super::Gemma4Config =
            serde_json::from_value(json).expect("deserialize Gemma4Config");
        assert_eq!(
            cfg.use_block_paged_cache, None,
            "use_block_paged_cache must default to None on JSON without the key"
        );
        assert_eq!(cfg.paged_block_size, None);
        assert_eq!(cfg.paged_cache_memory_mb, None);
    }

    /// `use_block_paged_cache: true` round-trips through serde.
    #[test]
    fn test_use_block_paged_cache_round_trips_true() {
        let json = serde_json::json!({
            "vocab_size": 0,
            "hidden_size": 0,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "intermediate_size": 1,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "max_position_embeddings": 2048,
            "use_block_paged_cache": true,
        });
        let cfg: super::Gemma4Config =
            serde_json::from_value(json).expect("deserialize Gemma4Config");
        assert_eq!(cfg.use_block_paged_cache, Some(true));
    }

    #[test]
    fn test_default_paged_cache_memory_covers_gemma4_full_context() {
        let memory_mb = super::gemma4_default_paged_cache_memory_mb(131_072, 16, 512, 2, 5);
        assert_eq!(
            memory_mb, 2560,
            "Gemma4 26B-A4B global KV cache needs 2560MiB to cover 128k tokens"
        );

        let cfg = mlx_paged_attn::PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb: memory_mb,
            head_size: 512,
            num_kv_heads: 2,
            num_layers: 5,
            use_fp8_cache: Some(false),
            max_seq_len: Some(131_072),
            max_batch_size: Some(32),
        };
        assert_eq!(cfg.calculate_num_blocks(), 8192);
        assert_eq!(cfg.max_cached_tokens(), 131_072);

        let undersized_cfg = mlx_paged_attn::PagedAttentionConfig {
            gpu_memory_mb: 2048,
            ..cfg
        };
        assert!(
            undersized_cfg.max_cached_tokens() < 124_920,
            "the previous fixed 2048MiB default cannot hold the failed 124,920-token prompt"
        );
    }

    #[test]
    fn test_default_paged_cache_memory_respects_minimum() {
        assert_eq!(
            super::gemma4_default_paged_cache_memory_mb(128, 16, 32, 2, 2),
            256
        );
    }

    /// Explicit opt-out (`Some(false)`) must NOT allocate the block-paged
    /// adapter. The previous "None means no adapter" assertion was removed
    /// when the default flipped from `unwrap_or(false)` to `unwrap_or(true)`
    /// — the explicit-false path is the new "no adapter" guarantee.
    #[test]
    fn test_gemma4_inner_no_paged_adapter_when_flag_is_explicit_false() {
        let cfg = paged_tiny_config(Some(false));
        let inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };
        assert!(
            inner.paged_adapter.is_none(),
            "paged_adapter must be None when use_block_paged_cache is Some(false)"
        );
    }

    /// Default-flag construction (`None`) must allocate the block-paged
    /// adapter under the new default-on policy (`unwrap_or(true)`).
    /// Allocates a `LayerKVPool`, so requires Metal — gracefully skips
    /// on no-Metal sandboxes.
    #[test]
    fn test_gemma4_inner_paged_adapter_when_flag_is_none_default_on_macos() {
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
        let cfg = paged_tiny_config(None);
        match super::Gemma4Inner::new(cfg) {
            Ok(inner) => {
                assert!(
                    inner.paged_adapter.is_some(),
                    "paged_adapter must be Some when use_block_paged_cache is None \
                     (new default-on policy: unwrap_or(true))"
                );
            }
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        }
    }

    /// Construction with `use_block_paged_cache: Some(true)` must populate
    /// `paged_adapter`. Allocates a `LayerKVPool`, so requires Metal —
    /// gracefully skips on no-Metal sandboxes.
    #[test]
    fn test_gemma4_inner_constructs_paged_adapter_when_flag_is_true() {
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
        let cfg = paged_tiny_config(Some(true));
        match super::Gemma4Inner::new(cfg) {
            Ok(inner) => {
                assert!(
                    inner.paged_adapter.is_some(),
                    "paged_adapter must be Some when use_block_paged_cache = Some(true)"
                );
            }
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        }
    }

    /// Paged/flat parity: a fresh (non-reuse) text-only `save_paged_history`
    /// must clear `cached_audio_key`, exactly as the flat `save_cache_state`
    /// does on a fresh turn. Without that clear, a text-only paged start over a
    /// reused model whose prior turn was audio would leave stale media state
    /// attached to the replacement session.
    #[test]
    fn test_text_only_paged_save_clears_stale_audio_key() {
        let cfg = paged_tiny_config(Some(false));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        // Simulate a completed audio turn that left the audio key set, then a
        // fresh text-only paged START (no `reset()`): image key already None,
        // session not continuable.
        inner.cached_audio_key = Some(7);
        inner.cached_image_key = None;
        inner.media_session_continuable = false;

        // Fresh (non-reuse, non-delta) text-only paged save — the same shape
        // the engine uses to persist a fresh text turn's history.
        let save_tokens: Vec<u32> = vec![10, 11, 12];
        let generated: Vec<u32> = vec![20, 21];
        inner
            .save_paged_history(&save_tokens, &generated, false, false)
            .expect("text-only paged save must succeed");

        // The fix: the stale audio key is cleared on the text-only save.
        assert!(
            inner.cached_audio_key.is_none(),
            "text-only paged save must clear the stale audio key"
        );

        assert_eq!(inner.session_media(), MediaCapabilities::NONE);
    }

    /// A warm text save over a pure-image session extends the same live
    /// media-derived KV. It must preserve the raw image identity and ordered
    /// placeholder sidecar so later text blocks keep the same image-aware block
    /// keys. Audio and mixed turns are deliberately cold/non-continuable and do
    /// not enter this path.
    #[test]
    fn test_media_context_survives_repeated_warm_text_saves() {
        let cfg = paged_tiny_config(Some(false));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let image_key = 11;
        let image_positions = vec![(1, image_key)];
        inner.publish_media_session_context(Some(image_key), None);
        inner.cached_paged_image_token_positions = image_positions.clone();
        inner.media_session_continuable = true;
        assert_eq!(inner.session_media(), MediaCapabilities::IMAGES);

        for turn in 0..2u32 {
            // Mirrors `run_paged_turn` handing the planner's prior context
            // to `save_paged_history` for a successful warm text delta.
            inner.paged_text_turn_context = inner.session_media();
            inner
                .save_paged_history(&[10, 11, turn], &[20, 21], false, true)
                .expect("warm text paged save must succeed");

            assert_eq!(inner.cached_image_key, Some(image_key));
            assert!(inner.cached_audio_key.is_none());
            assert_eq!(
                inner.cached_paged_image_token_positions, image_positions,
                "turn {turn} must preserve exact image cache lineage"
            );
            assert_eq!(
                inner.session_media(),
                MediaCapabilities::IMAGES,
                "turn {turn} must preserve the image-derived context"
            );
            assert!(inner.media_session_continuable);
        }
    }

    /// A failed paged media prepare must fail CLOSED. The vision core disarms
    /// `media_session_continuable` before the fallible adapter prepare, and all
    /// subsequent prepare failures call `invalidate_gemma4_hybrid_session`,
    /// which releases the request and clears both global and sliding reuse
    /// state. No media-derived history may survive as a token-only prefix hit.
    ///
    /// The state is built with the real transition functions
    /// (`publish_media_session_context` → warm `save_paged_history` → marker
    /// disarm → `invalidate_gemma4_hybrid_session`); driving the complete
    /// multimodal core to the failing prepare needs a real tokenizer file,
    /// which unit tests do not have.
    #[test]
    fn test_failed_media_prepare_fails_closed_after_warm_continuation() {
        let cfg = paged_tiny_config(Some(false));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        // Live caches so the prefix check below can only miss via the media
        // gate (not via `caches.is_none()`).
        inner
            .init_caches_sync()
            .expect("init_caches_sync must succeed");

        // Warm-continued pure-image session: finalize published exact image
        // identity + context and armed the marker; the warm text save preserves
        // that image lineage for later image-aware block registration.
        inner.publish_media_session_context(Some(11), None);
        inner.cached_paged_image_token_positions = vec![(1, 11)];
        inner.media_session_continuable = true;
        inner.paged_text_turn_context = inner.session_media();
        inner
            .save_paged_history(&[100, 101, 102], &[103, 104], false, true)
            .expect("warm text paged save must succeed");
        // Mirrors `run_paged_turn` resetting the turn-scoped snapshot.
        inner.paged_text_turn_context = MediaCapabilities::NONE;
        assert_eq!(inner.cached_image_key, Some(11));
        assert!(inner.cached_audio_key.is_none());
        assert_eq!(inner.cached_paged_image_token_positions, vec![(1, 11)]);
        assert_eq!(inner.session_media(), MediaCapabilities::IMAGES);

        // `keep_all = false` dropped the trailing stop token 104.
        assert_eq!(inner.cached_token_history, vec![100, 101, 102, 103]);
        let delta_tokens: Vec<u32> = vec![100, 101, 102, 103, 200];

        // While the continuation is armed the media gate does not force a
        // prefix miss (warm reuse stays possible).
        assert_eq!(
            inner.verify_cache_prefix(&delta_tokens, true),
            inner.cached_token_history.len(),
            "an armed continuation must not be forced to miss"
        );

        // The next media turn's failure path: the vision core disarms the
        // marker, then its prepare helper invalidates the complete hybrid
        // session before returning the error.
        inner.media_session_continuable = false;
        inner.invalidate_gemma4_hybrid_session("unit-test media prepare failure");

        assert!(inner.caches.is_none());
        assert!(inner.cached_token_history.is_empty());
        assert!(inner.cached_image_key.is_none());
        assert!(inner.cached_audio_key.is_none());
        assert!(inner.cached_paged_image_token_positions.is_empty());
        assert_eq!(inner.session_media(), MediaCapabilities::NONE);
        assert!(!inner.media_session_continuable);
        assert_eq!(
            inner.verify_cache_prefix(&delta_tokens, true),
            0,
            "invalidated media history must not seed a text-only prefix hit"
        );
    }

    #[test]
    fn test_fresh_text_save_replaces_persistent_media_context() {
        let cfg = paged_tiny_config(Some(false));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        inner.publish_media_session_context(Some(11), Some(22));
        inner.media_session_continuable = true;
        // Fresh plans carry no prior context. A successful text save therefore
        // replaces, rather than extends, the old media session.
        inner.paged_text_turn_context = MediaCapabilities::NONE;
        inner
            .save_paged_history(&[1, 2, 3], &[4, 5], false, true)
            .expect("fresh text paged save must succeed");

        assert_eq!(inner.session_media(), MediaCapabilities::NONE);
        assert!(!inner.media_session_continuable);
        assert!(inner.cached_image_key.is_none());
        assert!(inner.cached_audio_key.is_none());
    }

    /// Image/audio symmetry in `verify_cache_prefix`: a non-continuable session
    /// that still holds a cached AUDIO key must MISS (return `0`), exactly as it
    /// already does for a cached IMAGE key, so stale media KV is reset instead
    /// of being reused as a token-id prefix hit. With an otherwise-hitting
    /// prefix (live caches + matching `cached_token_history`), the audio guard
    /// must override the would-be hit. A continuable audio session (warm-
    /// continue) must NOT be forced to miss by this guard.
    ///
    /// Pre-fix (image-only guard) this would return `cached.len()` for the
    /// non-continuable audio case — a HIT — because the audio key was ignored,
    /// so the first assertion below would fail.
    #[test]
    fn test_verify_cache_prefix_audio_key_forces_miss() {
        let cfg = paged_tiny_config(Some(false));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        // Build an otherwise-hitting state: live caches + a non-empty cached
        // history that the incoming tokens match as a prefix. `init_caches_sync`
        // also clears reuse state, so the keys/marker/history are set AFTER.
        inner
            .init_caches_sync()
            .expect("init_caches_sync must succeed");
        inner.cached_token_history = vec![100, 101, 102];
        let tokens: Vec<u32> = vec![100, 101, 102, 103];

        // Non-continuable session holding only an AUDIO key: must MISS.
        inner.cached_image_key = None;
        inner.cached_audio_key = Some(7);
        inner.media_session_continuable = false;
        assert_eq!(
            inner.verify_cache_prefix(&tokens, true),
            0,
            "a non-continuable session holding audio state must force a cache miss"
        );

        // Continuable audio session (warm-continue): the guard must NOT force a
        // miss, so the otherwise-hitting prefix returns `cached.len()`.
        inner.media_session_continuable = true;
        assert_eq!(
            inner.verify_cache_prefix(&tokens, true),
            inner.cached_token_history.len(),
            "a continuable audio session must not be forced to miss by the media guard"
        );

        // Parity check: the same shape with an IMAGE key (already guarded) also
        // misses when non-continuable — the audio branch mirrors it exactly.
        inner.cached_image_key = Some(42);
        inner.cached_audio_key = None;
        inner.media_session_continuable = false;
        assert_eq!(
            inner.verify_cache_prefix(&tokens, true),
            0,
            "a non-continuable session holding image state must force a cache miss"
        );
    }

    /// Marker reset matrix: `media_session_continuable` must return to `false`
    /// at every session-reset entry point so a dropped-media session can never
    /// wrongly warm-continue. Covers `clear_reuse_state` and `reset_caches_sync`
    /// (both clear via `clear_reuse_state`).
    #[test]
    fn test_media_session_continuable_reset_matrix() {
        let cfg = paged_tiny_config(Some(false));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        // Fresh construction: marker defaults to false.
        assert!(
            !inner.media_session_continuable,
            "marker must default to false on construction"
        );

        // clear_reuse_state resets the marker and both persistent/transient
        // media context sources.
        inner.publish_media_session_context(Some(7), Some(8));
        inner.paged_text_turn_context = MediaCapabilities::IMAGES_AND_AUDIO;
        inner.media_session_continuable = true;
        inner.clear_reuse_state();
        assert!(
            !inner.media_session_continuable,
            "clear_reuse_state must reset the continuable marker"
        );
        assert_eq!(inner.session_media(), MediaCapabilities::NONE);
        assert_eq!(
            inner.paged_text_turn_context,
            MediaCapabilities::NONE,
            "clear_reuse_state must clear transient turn context"
        );

        // reset_caches_sync (which calls clear_reuse_state) resets the marker
        // AND nulls caches → has_live_session() false → continuation is rejected.
        inner.publish_media_session_context(None, Some(9));
        inner.paged_text_turn_context = MediaCapabilities::AUDIO;
        inner.media_session_continuable = true;
        inner
            .reset_caches_sync()
            .expect("reset_caches_sync must succeed");
        assert!(
            !inner.media_session_continuable,
            "reset_caches_sync must reset the continuable marker"
        );
        assert!(
            inner.cached_audio_key.is_none(),
            "reset_caches_sync must clear the media key"
        );
        assert_eq!(inner.session_media(), MediaCapabilities::NONE);
        assert!(!inner.has_live_session());
    }

    /// Only pure image turns currently publish image-aware per-block keys.
    /// Audio and mixed-media turns stay cold until their non-token identity is
    /// represented in the same cache chain.
    #[test]
    fn test_gemma4_media_continuable_gate() {
        assert!(!gemma4_media_continuable(false, false));
        assert!(gemma4_media_continuable(true, false));
        assert!(!gemma4_media_continuable(false, true));
        assert!(!gemma4_media_continuable(true, true));
    }

    /// All-global config: every layer must route through `GlobalPaged`
    /// with paged_idx == absolute index, no shared layers.
    #[test]
    fn test_compute_layer_kinds_all_global() {
        let cfg = super::Gemma4Config {
            num_hidden_layers: 4,
            layer_types: vec!["full_attention".to_string(); 4],
            ..paged_tiny_config(None)
        };
        let kinds = super::compute_layer_kinds(&cfg);
        assert_eq!(kinds.len(), 4);
        for (i, k) in kinds.iter().enumerate() {
            match k {
                super::Gemma4LayerKind::GlobalPaged { paged_idx } => {
                    assert_eq!(*paged_idx as usize, i, "layer {i} paged_idx mismatch");
                }
                other => panic!("layer {i}: expected GlobalPaged, got {other:?}"),
            }
        }
    }

    /// Hybrid sliding+global with no sharing: paged_idx counts only
    /// global layers in original order; sliding layers map to `Sliding`.
    #[test]
    fn test_compute_layer_kinds_hybrid_no_sharing() {
        // 5-layer cycle: 4 sliding + 1 global, repeated for 10 layers.
        let cycle = ["sliding_attention"; 4]
            .iter()
            .map(|s| s.to_string())
            .chain(std::iter::once("full_attention".to_string()))
            .collect::<Vec<_>>();
        let layer_types: Vec<String> = (0..10).map(|i| cycle[i % 5].clone()).collect();
        let cfg = super::Gemma4Config {
            num_hidden_layers: 10,
            layer_types,
            ..paged_tiny_config(None)
        };
        let kinds = super::compute_layer_kinds(&cfg);
        // Global layers at indices 4 and 9 -> paged_idx 0, 1.
        for (i, k) in kinds.iter().enumerate() {
            if i == 4 {
                assert!(
                    matches!(k, super::Gemma4LayerKind::GlobalPaged { paged_idx: 0 }),
                    "layer 4 must be GlobalPaged{{0}}, got {k:?}"
                );
            } else if i == 9 {
                assert!(
                    matches!(k, super::Gemma4LayerKind::GlobalPaged { paged_idx: 1 }),
                    "layer 9 must be GlobalPaged{{1}}, got {k:?}"
                );
            } else {
                assert!(
                    matches!(k, super::Gemma4LayerKind::Sliding),
                    "layer {i} must be Sliding, got {k:?}"
                );
            }
        }
    }

    /// Smoke test for `paged_turn_sync_core` via direct helper drives.
    ///
    /// Random-init weights cast to BF16 (the paged pool's expected
    /// dtype). Validates the adapter lifecycle (reset →
    /// find_cached_prefix → allocate_suffix → record_tokens →
    /// forward_paged_or_flat) and that produced logits have the
    /// expected shape, without asserting numerical equivalence to the
    /// flat path (random weights). Gracefully skipped on no-Metal.
    #[test]
    fn test_run_paged_prefill_decode_smoke() {
        // Block-paged needs the Metal backend; on a non-Metal build the
        // adapter is gated off (None) and there is nothing to exercise.
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }
        use crate::array::{DType, MxArray};

        let cfg = paged_tiny_config(Some(true));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };
        assert!(inner.paged_adapter.is_some());
        if let Err(e) = inner.init_caches_sync() {
            eprintln!("init_caches_sync skipped: {}", e.reason);
            return;
        }

        // Cast all weights to BF16 to match the pool dtype. Mirrors
        // LFM2's smoke-test cast pattern.
        let cast = |a: &MxArray| -> MxArray { a.astype(DType::BFloat16).expect("astype BFloat16") };
        let w = inner.embed_tokens.get_weight();
        inner.embed_tokens.set_weight(&cast(&w)).expect("embed");
        let w = inner.final_norm.get_weight();
        inner.final_norm.set_weight(&cast(&w)).expect("final_norm");
        if let Some(ref mut head) = inner.lm_head {
            let w = head.get_weight();
            head.set_weight(&cast(&w), "lm_head").expect("lm_head");
        }
        for layer in inner.layers.iter_mut() {
            // Norms.
            layer
                .set_input_layernorm_weight(&cast(&layer.input_layernorm_weight().clone()))
                .ok();
            layer
                .set_post_attention_layernorm_weight(&cast(
                    &layer.post_attention_layernorm_weight().clone(),
                ))
                .ok();
            layer
                .set_pre_feedforward_layernorm_weight(&cast(
                    &layer.pre_feedforward_layernorm_weight().clone(),
                ))
                .ok();
            layer
                .set_post_feedforward_layernorm_weight(&cast(
                    &layer.post_feedforward_layernorm_weight().clone(),
                ))
                .ok();
            // Attention projections + norms.
            let attn = &mut layer.self_attn;
            let w = attn.q_proj_weight();
            attn.set_q_proj_weight(&cast(&w)).expect("q");
            let w = attn.k_proj_weight();
            attn.set_k_proj_weight(&cast(&w)).expect("k");
            if let Some(w) = attn.v_proj_weight_opt() {
                attn.set_v_proj_weight(&cast(&w)).expect("v");
            }
            let w = attn.o_proj_weight();
            attn.set_o_proj_weight(&cast(&w)).expect("o");
            let w = attn.q_norm_weight();
            attn.set_q_norm_weight(&cast(&w)).expect("qn");
            let w = attn.k_norm_weight();
            attn.set_k_norm_weight(&cast(&w)).expect("kn");
            // MLP.
            if let crate::models::gemma4::quantized_linear::Gemma4MLPVariant::Standard(
                ref mut mlp,
            ) = layer.mlp
            {
                let w = mlp.gate_proj_weight();
                mlp.set_gate_proj_weight(&cast(&w)).expect("gate");
                let w = mlp.up_proj_weight();
                mlp.set_up_proj_weight(&cast(&w)).expect("up");
                let w = mlp.down_proj_weight();
                mlp.set_down_proj_weight(&cast(&w)).expect("down");
            }
        }

        // Adapter lifecycle.
        let prompt: Vec<u32> = vec![1, 2, 3, 4];
        if let Some(adapter) = inner.paged_adapter.as_mut() {
            if let Err(e) = adapter.reset_for_new_request(0) {
                eprintln!("skipping (adapter reset failed): {e}");
                return;
            }
            if let Err(e) = adapter.find_cached_prefix(&prompt, &[], 0, false) {
                eprintln!("skipping (find_cached_prefix failed): {e}");
                return;
            }
            if let Err(e) = adapter.allocate_suffix_blocks(16) {
                eprintln!("skipping (allocate_suffix_blocks failed): {e}");
                return;
            }
        }

        let last_logits = match inner.run_paged_prefill_chunk(&prompt, &prompt, 0, 0) {
            Ok(l) => l,
            Err(e) => {
                let msg = e.reason.to_string();
                if msg.contains("No Metal device found") || msg.contains("not supported") {
                    eprintln!("skipping smoke: {msg}");
                    return;
                }
                panic!("run_paged_prefill_chunk failed: {msg}");
            }
        };
        let vocab = last_logits.shape_at(0).expect("shape");
        assert_eq!(vocab, 100, "vocab_size from paged_tiny_config");

        let mut next_token: u32 = 5;
        for _ in 0..4 {
            match inner.run_paged_decode_step(next_token) {
                Ok(logits) => {
                    assert_eq!(logits.shape_at(0).expect("shape"), 1);
                    assert_eq!(logits.shape_at(1).expect("shape"), 1);
                    assert_eq!(logits.shape_at(2).expect("shape"), 100);
                }
                Err(e) => {
                    let msg = e.reason.to_string();
                    if msg.contains("No Metal device found") {
                        eprintln!("skipping decode (no Metal): {msg}");
                        return;
                    }
                    panic!("run_paged_decode_step failed: {msg}");
                }
            }
            next_token = next_token.wrapping_add(1);
        }

        if let Some(adapter) = inner.paged_adapter.as_mut() {
            let _ = adapter.release_request();
        }
    }

    /// Hybrid sliding/global config whose sidecar geometry is well defined:
    /// two PHYSICAL sliding layers, two global layers to carry the paged pool.
    #[cfg(test)]
    fn sliding_capture_config() -> super::Gemma4Config {
        super::Gemma4Config {
            num_hidden_layers: 4,
            layer_types: vec![
                "sliding_attention".to_string(),
                "full_attention".to_string(),
                "sliding_attention".to_string(),
                "full_attention".to_string(),
            ],
            ..paged_tiny_config(Some(true))
        }
    }

    /// Snapshots a live rotating cache would hold at `boundary`: one per
    /// PHYSICAL sliding layer, `None` everywhere else, with
    /// `cached_tokens = min(boundary, window)` rows — a genuine PRE-WRAP state
    /// when `boundary < window`, not a padded full-window one.
    #[cfg(test)]
    fn sliding_capture_snapshots(
        config: &super::Gemma4Config,
        boundary: u32,
    ) -> Vec<Option<RotatingKVCacheSnapshot>> {
        let geometry = sliding_sidecar::geometry(config).expect("geometry");
        let cached = boundary.min(geometry.window);
        let shape = [
            1i64,
            geometry.kv_heads as i64,
            cached as i64,
            geometry.head_dim as i64,
        ];
        let elements =
            (geometry.kv_heads as usize) * (cached as usize) * (geometry.head_dim as usize);
        let mut snapshots: Vec<Option<RotatingKVCacheSnapshot>> = (0..config.num_hidden_layers
            as usize)
            .map(|_| None)
            .collect();
        for (ordinal, &layer) in sliding_sidecar::physical_sliding_layers(config)
            .iter()
            .enumerate()
        {
            let make = |tag: u16| -> MxArray {
                let raw: Vec<u16> = (0..elements)
                    .map(|i| (i as u16).wrapping_mul(31).wrapping_add(tag))
                    .collect();
                MxArray::from_bfloat16(&raw, &shape).expect("bf16 snapshot array")
            };
            snapshots[layer] = Some(RotatingKVCacheSnapshot {
                keys: make(ordinal as u16 * 2),
                values: make(ordinal as u16 * 2 + 1),
                offset: boundary as i32,
                max_size: config.sliding_window,
                keep: 0,
                cached_tokens: cached as i32,
            });
        }
        snapshots
    }

    /// A checkpoint BELOW one sliding window is a legal capture anchor.
    ///
    /// This is the capture half of Track A. The payload carries
    /// `min(boundary, window)` rows, so a 32-token boundary under a 128-token
    /// window describes exactly what a live `RotatingKVCache` holds there. The
    /// old `boundary >= window` rule made `boundary_is_representable` refuse
    /// it, `find_gemma4_sliding_capture_checkpoints` come back empty, and gemma4's
    /// sidecar inert for every typical chat prompt.
    ///
    /// Drives the REAL selector rather than `boundary_is_representable` alone,
    /// so the token-prefix, block-hash and snapshot-readiness gates in front of
    /// it are all exercised at a sub-window boundary too.
    #[test]
    fn test_gemma4_capture_checkpoint_selects_sub_window_boundary() {
        let cfg = sliding_capture_config();
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(inner) => inner,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let geometry = sliding_sidecar::geometry(&inner.config).expect("hybrid config geometry");
        let block_size = 16u32;
        let boundary = 32u32;
        assert!(
            boundary < geometry.window,
            "fixture must be SUB-window: boundary={boundary} window={}",
            geometry.window
        );

        // 4 full blocks of prompt; the persisted K/V chain reaches all of them.
        let request_tokens: Vec<u32> = (7000..7064).collect();
        let extra_keys_per_block =
            engine::build_paged_extra_keys(request_tokens.len(), block_size, &[]);
        let final_block_hash = super::compute_gemma4_paged_prefix_block_hash_with_keys(
            &request_tokens,
            boundary,
            block_size,
            &extra_keys_per_block,
            0,
        )
        .expect("sub-window prefix hash");

        inner.sliding_prompt_boundary_checkpoint = Some(super::Gemma4SlidingPrefixCheckpoint {
            prefix_len: boundary,
            block_size,
            final_block_hash,
            protected_image_prompt_boundary: false,
            cold_anchor_rung: false,
            tokens: request_tokens[..boundary as usize].to_vec(),
            snapshots: sliding_capture_snapshots(&inner.config, boundary),
        });

        let selected = inner.find_gemma4_sliding_capture_checkpoints(
            &geometry,
            &request_tokens,
            block_size,
            4,
            &extra_keys_per_block,
            0,
        );
        let &(selected_boundary, snapshots) = selected
            .first()
            .expect("a sub-window checkpoint must now anchor a capture");
        assert_eq!(selected_boundary, boundary);
        assert_eq!(snapshots.len(), inner.config.num_hidden_layers as usize);
        for (layer, snapshot) in snapshots.iter().enumerate() {
            if inner.config.is_sliding_layer(layer) && !inner.config.is_kv_shared_layer(layer) {
                let snapshot = snapshot.as_ref().expect("physical sliding layer snapshot");
                assert_eq!(snapshot.offset, boundary as i32);
                assert_eq!(snapshot.cached_tokens, boundary as i32);
                assert_eq!(snapshot.max_size, inner.config.sliding_window);
            } else {
                assert!(snapshot.is_none(), "layer {layer} must carry no snapshot");
            }
        }

        // And the payload the capture would write is well formed at this
        // sub-window boundary — the format follows the boundary, it does not
        // dictate it.
        let layout = sliding_sidecar::layout_at(&geometry, boundary);
        assert_eq!(layout.boundary_tokens, boundary);
        assert_eq!(
            layout.dims,
            vec![1u32, geometry.kv_heads, boundary, geometry.head_dim]
        );
        let tensors =
            sliding_sidecar::encode_tensors(&inner.config, &geometry, snapshots, boundary)
                .expect("encode must not error")
                .expect("sub-window snapshots must encode");
        assert_eq!(tensors.len(), layout.tensor_count().expect("tensor count"));
        assert!(tensors.iter().all(|t| t.len() == layout.bytes_per_tensor));

        // Fail-closed the other way: a checkpoint DEEPER than the persisted
        // K/V chain is still refused. A sidecar past the chain's break could
        // never be selected on restore, so writing one would only burn quota.
        assert!(
            inner
                .find_gemma4_sliding_capture_checkpoints(
                    &geometry,
                    &request_tokens,
                    block_size,
                    1,
                    &extra_keys_per_block,
                    0,
                )
                .is_empty(),
            "boundary 32 must not be selected when the chain reaches only 16 tokens"
        );
    }

    /// The durable image path is deliberately stricter than same-process E2B
    /// reuse: the selected checkpoint must be after the complete expanded image
    /// run and carry the exact image-aware block hash.
    #[test]
    fn test_gemma4_image_capture_requires_after_image_exact_checkpoint() {
        let cfg = sliding_capture_config();
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(inner) => inner,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let geometry = sliding_sidecar::geometry(&inner.config).expect("hybrid config geometry");
        let block_size = 16u32;
        let boundary = 48u32;
        let request_tokens: Vec<u32> = (8000..8064).collect();
        let image_positions: Vec<(u32, u64)> =
            (20..40).map(|position| (position, 0xAAAA)).collect();
        let last_image_exclusive = 40u32;
        let extra_keys_per_block =
            engine::build_paged_extra_keys(request_tokens.len(), block_size, &image_positions);
        let final_block_hash = super::compute_gemma4_paged_prefix_block_hash_with_keys(
            &request_tokens,
            boundary,
            block_size,
            &extra_keys_per_block,
            0,
        )
        .expect("image-aware prefix hash");

        inner.sliding_prompt_boundary_checkpoint = Some(super::Gemma4SlidingPrefixCheckpoint {
            prefix_len: boundary,
            block_size,
            final_block_hash,
            protected_image_prompt_boundary: true,
            cold_anchor_rung: false,
            tokens: request_tokens[..boundary as usize].to_vec(),
            snapshots: sliding_capture_snapshots(&inner.config, boundary),
        });

        let selected = inner.find_gemma4_sliding_capture_checkpoints(
            &geometry,
            &request_tokens,
            block_size,
            4,
            &extra_keys_per_block,
            last_image_exclusive,
        );
        assert_eq!(
            selected
                .iter()
                .map(|(selected_boundary, _)| *selected_boundary)
                .collect::<Vec<_>>(),
            vec![boundary],
            "an exact checkpoint after the complete image run must be selectable"
        );

        assert!(
            inner
                .find_gemma4_sliding_capture_checkpoints(
                    &geometry,
                    &request_tokens,
                    block_size,
                    4,
                    &extra_keys_per_block,
                    boundary + 1,
                )
                .is_empty(),
            "a checkpoint before the conservative image floor must be refused"
        );

        let changed_image_positions: Vec<(u32, u64)> =
            (20..40).map(|position| (position, 0xBBBB)).collect();
        let changed_extra_keys = engine::build_paged_extra_keys(
            request_tokens.len(),
            block_size,
            &changed_image_positions,
        );
        assert!(
            inner
                .find_gemma4_sliding_capture_checkpoints(
                    &geometry,
                    &request_tokens,
                    block_size,
                    4,
                    &changed_extra_keys,
                    last_image_exclusive,
                )
                .is_empty(),
            "the same tokens with a different image hash must not select the checkpoint"
        );
    }

    /// The aligned-prompt case end to end through the SELECTOR: with the prompt
    /// boundary out of reach, the cold-restore tail is what the capture anchors
    /// on — and it only gets that chance because it is a candidate at all.
    ///
    /// A 64-token prompt (4 whole blocks of 16) plus an 8-token completion. The
    /// prompt boundary lands at 64; a restore of this prompt looks up
    /// `prompt[..63]`, whose deepest block boundary is 48. Under the old
    /// ceiling the selector took 64 — 209.7 MB of structurally valid sidecar at
    /// an address no restore can name, rewritten every session.
    #[test]
    fn test_gemma4_capture_prefers_the_cold_tail_over_an_unreachable_prompt_boundary() {
        let cfg = sliding_capture_config();
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(inner) => inner,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let geometry = sliding_sidecar::geometry(&inner.config).expect("hybrid config geometry");
        let block_size = 16u32;
        let prompt_len = 64u32;
        assert!(
            prompt_len.is_multiple_of(block_size),
            "the fixture's whole point is a block-ALIGNED prompt"
        );
        let prompt_boundary = prompt_len / block_size * block_size;
        let tail_boundary = super::gemma4_cold_restore_reachable_boundary(prompt_len, block_size);
        assert_eq!((prompt_boundary, tail_boundary), (64, 48));

        // The capture sees the completion too, which is what used to widen its
        // ceiling past the prompt.
        let request_tokens: Vec<u32> = (7000..7072).collect();
        let extra_keys_per_block =
            engine::build_paged_extra_keys(request_tokens.len(), block_size, &[]);
        let checkpoint_at = |boundary: u32| super::Gemma4SlidingPrefixCheckpoint {
            prefix_len: boundary,
            block_size,
            final_block_hash: super::compute_gemma4_paged_prefix_block_hash_with_keys(
                &request_tokens,
                boundary,
                block_size,
                &extra_keys_per_block,
                0,
            )
            .expect("prefix hash"),
            protected_image_prompt_boundary: false,
            cold_anchor_rung: false,
            tokens: request_tokens[..boundary as usize].to_vec(),
            snapshots: sliding_capture_snapshots(&inner.config, boundary),
        };
        inner.sliding_prompt_boundary_checkpoint = Some(checkpoint_at(prompt_boundary));
        inner.sliding_cold_restore_tail_checkpoint = Some(checkpoint_at(tail_boundary));

        // The chain covered every block of the request; only reachability binds.
        let chain_blocks = super::gemma4_sliding_cold_capture_ceiling_blocks(
            u32::MAX,
            request_tokens.len(),
            prompt_len,
            block_size,
        );
        assert_eq!(chain_blocks, 3, "48 tokens, not the request's 72");

        let candidates = inner.find_gemma4_sliding_capture_checkpoints(
            &geometry,
            &request_tokens,
            block_size,
            chain_blocks,
            &extra_keys_per_block,
            0,
        );
        let boundaries: Vec<u32> = candidates.iter().map(|(boundary, _)| *boundary).collect();
        assert_eq!(
            boundaries,
            vec![tail_boundary],
            "the tail must be offered and the unreachable prompt boundary must not be"
        );

        // And it really is a usable anchor, not just a boundary number: the
        // payload the capture would write encodes at it.
        let (_, snapshots) = candidates[0];
        let layout = sliding_sidecar::layout_at(&geometry, tail_boundary);
        assert_eq!(layout.boundary_tokens, tail_boundary);
        let tensors =
            sliding_sidecar::encode_tensors(&inner.config, &geometry, snapshots, tail_boundary)
                .expect("encode must not error")
                .expect("the cold tail must encode");
        assert_eq!(tensors.len(), layout.tensor_count().expect("tensor count"));
        assert!(tensors.iter().all(|t| t.len() == layout.bytes_per_tensor));
    }

    #[test]
    fn test_gemma4_prompt_boundary_checkpoint_survives_decode_checkpoint_eviction() {
        let cfg = paged_tiny_config(Some(true));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let block_size = 16;
        let prompt: Vec<u32> = (10..26).collect();
        let prompt_hash = super::compute_gemma4_paged_prefix_block_hash(
            &prompt,
            prompt.len() as u32,
            block_size,
            0,
        )
        .expect("prompt hash");
        inner.sliding_prompt_boundary_checkpoint = Some(super::Gemma4SlidingPrefixCheckpoint {
            prefix_len: prompt.len() as u32,
            block_size,
            final_block_hash: prompt_hash,
            protected_image_prompt_boundary: false,
            cold_anchor_rung: false,
            tokens: prompt.clone(),
            snapshots: vec![None; inner.config.num_hidden_layers as usize],
        });

        let checkpoint_limit = super::gemma4_sliding_prefix_checkpoint_limit_for_override(
            &inner.config,
            block_size,
            None,
        );
        for i in 0..(checkpoint_limit + 3) {
            let tokens: Vec<u32> = (0..16).map(|token| 100 + i as u32 + token).collect();
            inner
                .sliding_prefix_checkpoints
                .push_back(super::Gemma4SlidingPrefixCheckpoint {
                    prefix_len: tokens.len() as u32,
                    block_size,
                    final_block_hash: i as u64 + 1,
                    protected_image_prompt_boundary: false,
                    cold_anchor_rung: false,
                    tokens,
                    snapshots: vec![None; inner.config.num_hidden_layers as usize],
                });
            while inner.sliding_prefix_checkpoints.len() > checkpoint_limit {
                inner.sliding_prefix_checkpoints.pop_front();
            }
        }
        assert_eq!(inner.sliding_prefix_checkpoints.len(), checkpoint_limit);

        let restored = inner
            .find_gemma4_sliding_prefix_checkpoint(&prompt, prompt.len() as u32, block_size, 0)
            .expect("prefix lookup");
        assert!(
            restored.is_some(),
            "prompt-boundary checkpoint must not be evicted by decode-boundary checkpoints"
        );
    }

    #[test]
    fn test_gemma4_decode_checkpoint_retains_recent_retokenization_drift() {
        let cfg = paged_tiny_config(Some(true));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let block_size = 16;
        let target_tokens: Vec<u32> = (1000..1016).collect();
        let target_hash = super::compute_gemma4_paged_prefix_block_hash(
            &target_tokens,
            target_tokens.len() as u32,
            block_size,
            0,
        )
        .expect("target hash");
        inner
            .sliding_prefix_checkpoints
            .push_back(super::Gemma4SlidingPrefixCheckpoint {
                prefix_len: target_tokens.len() as u32,
                block_size,
                final_block_hash: target_hash,
                protected_image_prompt_boundary: false,
                cold_anchor_rung: false,
                tokens: target_tokens.clone(),
                snapshots: vec![None; inner.config.num_hidden_layers as usize],
            });

        // The observed Gemma4 tool-call retokenization drift needed the
        // checkpoint five block boundaries behind the final decode state:
        // 46272 was requested after 46288, 46304, 46320, and 46336 had
        // also been checkpointed.
        for i in 0..4 {
            let tokens: Vec<u32> = (0..16).map(|token| 2000 + i as u32 + token).collect();
            let hash = super::compute_gemma4_paged_prefix_block_hash(
                &tokens,
                tokens.len() as u32,
                block_size,
                0,
            )
            .expect("newer hash");
            inner
                .sliding_prefix_checkpoints
                .push_back(super::Gemma4SlidingPrefixCheckpoint {
                    prefix_len: tokens.len() as u32,
                    block_size,
                    final_block_hash: hash,
                    protected_image_prompt_boundary: false,
                    cold_anchor_rung: false,
                    tokens,
                    snapshots: vec![None; inner.config.num_hidden_layers as usize],
                });
            let checkpoint_limit = super::gemma4_sliding_prefix_checkpoint_limit_for_override(
                &inner.config,
                block_size,
                None,
            );
            while inner.sliding_prefix_checkpoints.len() > checkpoint_limit {
                inner.sliding_prefix_checkpoints.pop_front();
            }
        }

        let restored = inner
            .find_gemma4_sliding_prefix_checkpoint(
                &target_tokens,
                target_tokens.len() as u32,
                block_size,
                0,
            )
            .expect("prefix lookup");
        assert!(
            restored.is_some(),
            "decode checkpoints must retain the block needed after modest retokenization drift"
        );
    }

    #[test]
    fn test_gemma4_decode_checkpoint_retains_sliding_window_drift() {
        let mut cfg = paged_tiny_config(Some(true));
        cfg.sliding_window = 512;
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let block_size = 16;
        let checkpoint_limit = super::gemma4_sliding_prefix_checkpoint_limit_for_override(
            &inner.config,
            block_size,
            None,
        );
        assert_eq!(
            checkpoint_limit, 64,
            "512-token sliding window with 16-token blocks should retain two windows of decode checkpoints"
        );
        let target_tokens: Vec<u32> = (3000..3016).collect();
        let target_hash = super::compute_gemma4_paged_prefix_block_hash(
            &target_tokens,
            target_tokens.len() as u32,
            block_size,
            0,
        )
        .expect("target hash");
        inner
            .sliding_prefix_checkpoints
            .push_back(super::Gemma4SlidingPrefixCheckpoint {
                prefix_len: target_tokens.len() as u32,
                block_size,
                final_block_hash: target_hash,
                protected_image_prompt_boundary: false,
                cold_anchor_rung: false,
                tokens: target_tokens.clone(),
                snapshots: vec![None; inner.config.num_hidden_layers as usize],
            });

        // The live 2026-05-09 Gemma4 trace needed a checkpoint eighteen
        // block boundaries behind the final decode state (57072 requested
        // after decode reached 57360). A one-window default retains that
        // level of retokenization drift instead of forcing a full replay.
        for i in 0..18 {
            let token_base = 4000 + (i as u32 * block_size);
            let tokens: Vec<u32> = (0..block_size).map(|token| token_base + token).collect();
            let hash = super::compute_gemma4_paged_prefix_block_hash(
                &tokens,
                tokens.len() as u32,
                block_size,
                0,
            )
            .expect("newer hash");
            inner
                .sliding_prefix_checkpoints
                .push_back(super::Gemma4SlidingPrefixCheckpoint {
                    prefix_len: tokens.len() as u32,
                    block_size,
                    final_block_hash: hash,
                    protected_image_prompt_boundary: false,
                    cold_anchor_rung: false,
                    tokens,
                    snapshots: vec![None; inner.config.num_hidden_layers as usize],
                });
            while inner.sliding_prefix_checkpoints.len() > checkpoint_limit {
                inner.sliding_prefix_checkpoints.pop_front();
            }
        }

        let restored = inner
            .find_gemma4_sliding_prefix_checkpoint(
                &target_tokens,
                target_tokens.len() as u32,
                block_size,
                0,
            )
            .expect("prefix lookup");
        assert!(
            restored.is_some(),
            "decode checkpoints must retain one sliding-window worth of retokenization drift"
        );
    }

    #[test]
    fn test_gemma4_decode_checkpoint_retains_auxiliary_branch_interleaving() {
        let mut cfg = paged_tiny_config(Some(true));
        cfg.sliding_window = 1024;
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let block_size = 16;
        let checkpoint_limit = super::gemma4_sliding_prefix_checkpoint_limit_for_override(
            &inner.config,
            block_size,
            None,
        );
        assert_eq!(
            checkpoint_limit, 128,
            "1024-token sliding window with 16-token blocks should retain two windows"
        );
        let target_tokens: Vec<u32> = (10_000..10_016).collect();
        let target_hash = super::compute_gemma4_paged_prefix_block_hash(
            &target_tokens,
            target_tokens.len() as u32,
            block_size,
            0,
        )
        .expect("target hash");
        inner
            .sliding_prefix_checkpoints
            .push_back(super::Gemma4SlidingPrefixCheckpoint {
                prefix_len: target_tokens.len() as u32,
                block_size,
                final_block_hash: target_hash,
                protected_image_prompt_boundary: false,
                cold_anchor_rung: false,
                tokens: target_tokens.clone(),
                snapshots: vec![None; inner.config.num_hidden_layers as usize],
            });

        // The 2026-05-09 live trace stored the needed 48,416-token
        // checkpoint, then 93 checkpoints from auxiliary 29k/33k branches
        // before the main branch asked for 48,416 again. A one-window FIFO
        // cap evicted it; two windows retains it without unbounded growth.
        for i in 0..93 {
            let token_base = 20_000 + (i as u32 * block_size);
            let tokens: Vec<u32> = (0..block_size).map(|token| token_base + token).collect();
            let hash = super::compute_gemma4_paged_prefix_block_hash(
                &tokens,
                tokens.len() as u32,
                block_size,
                0,
            )
            .expect("newer hash");
            inner
                .sliding_prefix_checkpoints
                .push_back(super::Gemma4SlidingPrefixCheckpoint {
                    prefix_len: tokens.len() as u32,
                    block_size,
                    final_block_hash: hash,
                    protected_image_prompt_boundary: false,
                    cold_anchor_rung: false,
                    tokens,
                    snapshots: vec![None; inner.config.num_hidden_layers as usize],
                });
            super::trim_gemma4_sliding_prefix_checkpoints(
                &mut inner.sliding_prefix_checkpoints,
                super::Gemma4SlidingRetentionCaps::pre_ladder(
                    checkpoint_limit,
                    super::Gemma4SlidingCheckpointBytes::for_config(&inner.config),
                ),
                false,
            );
        }

        let restored = inner
            .find_gemma4_sliding_prefix_checkpoint(
                &target_tokens,
                target_tokens.len() as u32,
                block_size,
                0,
            )
            .expect("prefix lookup");
        assert!(
            restored.is_some(),
            "decode checkpoints must survive auxiliary branch interleaving seen in live sessions"
        );
    }

    #[test]
    fn test_gemma4_sliding_decode_checkpoint_interval_uses_window_stride() {
        let mut cfg = paged_tiny_config(Some(true));
        cfg.sliding_window = 1024;
        assert_eq!(
            super::gemma4_sliding_decode_checkpoint_interval(&cfg, 16),
            1024
        );

        cfg.sliding_window = 1000;
        assert_eq!(
            super::gemma4_sliding_decode_checkpoint_interval(&cfg, 16),
            1008,
            "checkpoint interval should stay aligned to paged block boundaries"
        );
    }

    #[test]
    fn test_gemma4_sliding_prefix_checkpoint_restores_nearest_prefix() {
        let cfg = paged_tiny_config(Some(true));
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let block_size = 16;
        let tokens: Vec<u32> = (0..1280).map(|token| 50_000 + token).collect();
        let checkpoint_len = 1024;
        let checkpoint_hash =
            super::compute_gemma4_paged_prefix_block_hash(&tokens, checkpoint_len, block_size, 0)
                .expect("checkpoint hash");
        inner
            .sliding_prefix_checkpoints
            .push_back(super::Gemma4SlidingPrefixCheckpoint {
                prefix_len: checkpoint_len,
                block_size,
                final_block_hash: checkpoint_hash,
                protected_image_prompt_boundary: false,
                cold_anchor_rung: false,
                tokens: tokens[..checkpoint_len as usize].to_vec(),
                snapshots: vec![None; inner.config.num_hidden_layers as usize],
            });

        let hit = inner
            .find_gemma4_sliding_prefix_checkpoint(&tokens, tokens.len() as u32, block_size, 0)
            .expect("prefix lookup")
            .expect("nearest checkpoint hit");
        assert_eq!(hit.prefix_len, checkpoint_len);
    }

    #[test]
    fn test_gemma4_mid_prompt_prefix_hit_uses_near_prefill_checkpoint() {
        let mut cfg = paged_tiny_config(Some(true));
        cfg.sliding_window = 1024;
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let block_size = 16;
        let cached_prefix_len = 24_352;
        let checkpoint_len = 23_552;
        let tokens: Vec<u32> = (0..cached_prefix_len).map(|token| 90_000 + token).collect();
        let checkpoint_hash =
            super::compute_gemma4_paged_prefix_block_hash(&tokens, checkpoint_len, block_size, 0)
                .expect("checkpoint hash");
        inner
            .sliding_prefix_checkpoints
            .push_back(super::Gemma4SlidingPrefixCheckpoint {
                prefix_len: checkpoint_len,
                block_size,
                final_block_hash: checkpoint_hash,
                protected_image_prompt_boundary: false,
                cold_anchor_rung: false,
                tokens: tokens[..checkpoint_len as usize].to_vec(),
                snapshots: vec![None; inner.config.num_hidden_layers as usize],
            });

        let hit = inner
            .find_gemma4_sliding_prefix_checkpoint(&tokens, cached_prefix_len, block_size, 0)
            .expect("prefix lookup")
            .expect("near checkpoint hit");
        assert_eq!(hit.prefix_len, checkpoint_len);
        assert_eq!(cached_prefix_len - hit.prefix_len, 800);
        assert_eq!(
            super::gemma4_large_sliding_restore_suppression_limit(
                &inner.config,
                block_size,
                cached_prefix_len - hit.prefix_len
            ),
            None,
            "a one-window prefill checkpoint should prevent cold-prefill suppression"
        );
    }

    #[test]
    fn test_gemma4_window_stride_checkpoints_retain_old_branch_prefix() {
        let mut cfg = paged_tiny_config(Some(true));
        cfg.sliding_window = 1024;
        let mut inner = match super::Gemma4Inner::new(cfg) {
            Ok(i) => i,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("No Metal device found") {
                    eprintln!("skipping (no Metal device): {msg}");
                    return;
                }
                panic!("unexpected Gemma4Inner::new failure: {msg}");
            }
        };

        let block_size = 16;
        let target_len = 36_096;
        let target_tokens: Vec<u32> = (0..target_len).map(|token| 70_000 + token).collect();
        let target_hash = super::compute_gemma4_paged_prefix_block_hash(
            &target_tokens,
            target_len,
            block_size,
            0,
        )
        .expect("target hash");
        inner
            .sliding_prefix_checkpoints
            .push_back(super::Gemma4SlidingPrefixCheckpoint {
                prefix_len: target_len,
                block_size,
                final_block_hash: target_hash,
                protected_image_prompt_boundary: false,
                cold_anchor_rung: false,
                tokens: target_tokens.clone(),
                snapshots: vec![None; inner.config.num_hidden_layers as usize],
            });

        let checkpoint_limit = super::gemma4_sliding_prefix_checkpoint_limit_for_override(
            &inner.config,
            block_size,
            None,
        );
        let interval = super::gemma4_sliding_decode_checkpoint_interval(&inner.config, block_size);
        assert_eq!(interval, 1024);
        assert_eq!(checkpoint_limit, 128);

        for i in 0..96 {
            let prefix_len = 80_000 + i as u32 * interval;
            let tokens: Vec<u32> = (0..prefix_len).map(|token| 200_000 + token).collect();
            let hash =
                super::compute_gemma4_paged_prefix_block_hash(&tokens, prefix_len, block_size, 0)
                    .expect("newer hash");
            inner
                .sliding_prefix_checkpoints
                .push_back(super::Gemma4SlidingPrefixCheckpoint {
                    prefix_len,
                    block_size,
                    final_block_hash: hash,
                    protected_image_prompt_boundary: false,
                    cold_anchor_rung: false,
                    tokens,
                    snapshots: vec![None; inner.config.num_hidden_layers as usize],
                });
            super::trim_gemma4_sliding_prefix_checkpoints(
                &mut inner.sliding_prefix_checkpoints,
                super::Gemma4SlidingRetentionCaps::pre_ladder(
                    checkpoint_limit,
                    super::Gemma4SlidingCheckpointBytes::for_config(&inner.config),
                ),
                false,
            );
        }

        let hit = inner
            .find_gemma4_sliding_prefix_checkpoint(
                &target_tokens,
                target_tokens.len() as u32,
                block_size,
                0,
            )
            .expect("prefix lookup")
            .expect("old branch checkpoint hit");
        assert_eq!(hit.prefix_len, target_len);
    }

    /// KV-shared layers must resolve their anchor's pool slot
    /// (SharedOnGlobal) or absolute index (SharedOnSliding).
    #[test]
    fn test_compute_layer_kinds_kv_sharing_resolves_anchors() {
        // 8 layers: pattern S G S G S G S G (4 global @ 1, 3, 5, 7).
        // num_kv_shared_layers = 4 → last 4 (indices 4, 5, 6, 7) reuse anchors.
        // Anchor for shared global at i=5 should be the last non-shared
        // global before first_kv_shared_layer (=4): that's i=3 → paged_idx=1.
        // Anchor for shared sliding at i=4 should be sliding at i=2.
        let layer_types: Vec<String> = (0..8)
            .map(|i| {
                if i % 2 == 1 {
                    "full_attention".to_string()
                } else {
                    "sliding_attention".to_string()
                }
            })
            .collect();
        let cfg = super::Gemma4Config {
            num_hidden_layers: 8,
            layer_types,
            num_kv_shared_layers: Some(4),
            ..paged_tiny_config(None)
        };
        let kinds = super::compute_layer_kinds(&cfg);
        // Non-shared layers: 0=Sliding, 1=GlobalPaged{0}, 2=Sliding, 3=GlobalPaged{1}.
        assert!(matches!(kinds[0], super::Gemma4LayerKind::Sliding));
        assert!(matches!(
            kinds[1],
            super::Gemma4LayerKind::GlobalPaged { paged_idx: 0 }
        ));
        assert!(matches!(kinds[2], super::Gemma4LayerKind::Sliding));
        assert!(matches!(
            kinds[3],
            super::Gemma4LayerKind::GlobalPaged { paged_idx: 1 }
        ));
        // Shared layers 4..8 are aliases. They do not consume paged slots;
        // SharedOnGlobal carries the ANCHOR's pool slot, and
        // SharedOnSliding carries the anchor's absolute layer index.
        match kinds[4] {
            super::Gemma4LayerKind::SharedOnSliding { anchor_layer_idx } => {
                assert_eq!(anchor_layer_idx, 2, "anchor for sliding-shared layer 4");
            }
            ref other => panic!("layer 4: expected SharedOnSliding, got {other:?}"),
        }
        match kinds[5] {
            super::Gemma4LayerKind::SharedOnGlobal { anchor_paged_idx } => {
                // Anchor at layer 3 → paged_idx 1.
                assert_eq!(anchor_paged_idx, 1, "anchor paged_idx for global-shared 5");
            }
            ref other => panic!("layer 5: expected SharedOnGlobal, got {other:?}"),
        }
        match kinds[6] {
            super::Gemma4LayerKind::SharedOnSliding { anchor_layer_idx } => {
                assert_eq!(anchor_layer_idx, 2, "anchor for sliding-shared layer 6");
            }
            ref other => panic!("layer 6: expected SharedOnSliding, got {other:?}"),
        }
        match kinds[7] {
            super::Gemma4LayerKind::SharedOnGlobal { anchor_paged_idx } => {
                assert_eq!(anchor_paged_idx, 1, "anchor paged_idx for global-shared 7");
            }
            ref other => panic!("layer 7: expected SharedOnGlobal, got {other:?}"),
        }
    }

    /// Element-wise comparison for `Gemma4LayerKind`, which intentionally
    /// does not derive `PartialEq` (mirrors `Lfm2LayerKind`, which doesn't
    /// either — this codebase's existing tests compare these routing enums
    /// via `matches!`/`match`, not `assert_eq!`).
    fn layer_kind_matches(a: &super::Gemma4LayerKind, b: &super::Gemma4LayerKind) -> bool {
        use super::Gemma4LayerKind::*;
        match (a, b) {
            (Sliding, Sliding) => true,
            (GlobalPaged { paged_idx: x }, GlobalPaged { paged_idx: y }) => x == y,
            (
                SharedOnGlobal {
                    anchor_paged_idx: x,
                },
                SharedOnGlobal {
                    anchor_paged_idx: y,
                },
            ) => x == y,
            (
                SharedOnSliding {
                    anchor_layer_idx: x,
                },
                SharedOnSliding {
                    anchor_layer_idx: y,
                },
            ) => x == y,
            _ => false,
        }
    }

    /// `Gemma4Inner::new` must cache `layer_kinds` once instead of
    /// re-deriving it (BTreeMap/BTreeSet grouping + a sort, see
    /// `compute_layer_kinds_from_kv_cache_specs`) on every paged
    /// prefill-chunk / decode-step call. The cached field must always equal
    /// a fresh from-scratch computation over the same config — covers
    /// all-global, hybrid sliding+global, and KV-shared layouts (mirrors
    /// the three `test_compute_layer_kinds_*` cases above).
    #[test]
    fn test_gemma4_inner_caches_layer_kinds_matching_fresh_compute() {
        if !crate::engine::persistence::compiled_forward_backend_available() {
            eprintln!("skipping (paged backend unavailable without Metal)");
            return;
        }

        let all_global = super::Gemma4Config {
            num_hidden_layers: 4,
            layer_types: vec!["full_attention".to_string(); 4],
            ..paged_tiny_config(Some(true))
        };

        let cycle = ["sliding_attention"; 4]
            .iter()
            .map(|s| s.to_string())
            .chain(std::iter::once("full_attention".to_string()))
            .collect::<Vec<_>>();
        let hybrid = super::Gemma4Config {
            num_hidden_layers: 10,
            layer_types: (0..10).map(|i| cycle[i % 5].clone()).collect(),
            ..paged_tiny_config(Some(true))
        };

        let shared_layer_types: Vec<String> = (0..8)
            .map(|i| {
                if i % 2 == 1 {
                    "full_attention".to_string()
                } else {
                    "sliding_attention".to_string()
                }
            })
            .collect();
        let kv_shared = super::Gemma4Config {
            num_hidden_layers: 8,
            layer_types: shared_layer_types,
            num_kv_shared_layers: Some(4),
            ..paged_tiny_config(Some(true))
        };

        for cfg in [all_global, hybrid, kv_shared] {
            let expected = super::compute_layer_kinds_from_kv_cache_specs(&cfg)
                .expect("fresh layer-kind computation must succeed for a valid paged config");
            let inner = match super::Gemma4Inner::new(cfg) {
                Ok(inner) => inner,
                Err(err) => {
                    let msg = err.reason.to_string();
                    if msg.contains("No Metal device found") {
                        eprintln!("skipping (no Metal device): {msg}");
                        return;
                    }
                    panic!("unexpected Gemma4Inner::new failure: {msg}");
                }
            };
            assert!(
                inner.paged_adapter.is_some(),
                "test configs force use_block_paged_cache=true"
            );
            assert_eq!(
                inner.layer_kinds.len(),
                expected.len(),
                "cached layer_kinds length must match a fresh compute"
            );
            for (i, (got, want)) in inner.layer_kinds.iter().zip(expected.iter()).enumerate() {
                assert!(
                    layer_kind_matches(got, want),
                    "layer {i}: cached layer_kinds diverged from fresh compute: \
                     got {got:?}, want {want:?}"
                );
            }
        }
    }

    /// Manual timing probe (not a correctness gate — `#[ignore]`d so it
    /// never runs in CI). Measures the per-call cost this task eliminates:
    /// re-deriving the routing table from scratch (BTreeMap/BTreeSet + sort)
    /// vs. the cached `Vec::clone`. Pure CPU, no GPU/model weights, immune
    /// to thermal throttling. Run with:
    /// `cargo test -p mlx-core --release --lib -- --ignored --nocapture \
    ///  bench_layer_kinds_manual`
    #[test]
    #[ignore]
    fn bench_layer_kinds_manual() {
        // Scaled to ~48 layers with a realistic 5:1 sliding:global cycle
        // and KV-sharing, so the BTreeMap/BTreeSet grouping + sort has a
        // realistic amount of work to do.
        let cycle = ["sliding_attention"; 4]
            .iter()
            .map(|s| s.to_string())
            .chain(std::iter::once("full_attention".to_string()))
            .collect::<Vec<_>>();
        let cfg = super::Gemma4Config {
            num_hidden_layers: 48,
            layer_types: (0..48).map(|i| cycle[i % 5].clone()).collect(),
            num_kv_shared_layers: Some(8),
            ..paged_tiny_config(Some(true))
        };

        let n: u32 = 200_000;

        let start = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(
                super::compute_layer_kinds_from_kv_cache_specs(std::hint::black_box(&cfg)).unwrap(),
            );
        }
        eprintln!("recompute: {:?}/call", start.elapsed() / n);

        let cached = super::compute_layer_kinds_from_kv_cache_specs(&cfg).unwrap();
        let start = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(std::hint::black_box(&cached).clone());
        }
        eprintln!("cached clone: {:?}/call", start.elapsed() / n);
    }

    #[test]
    fn test_compute_layer_kv_cache_specs_group_full_sliding_and_shared_aliases() {
        let layer_types: Vec<String> = (0..8)
            .map(|i| {
                if i % 2 == 1 {
                    "full_attention".to_string()
                } else {
                    "sliding_attention".to_string()
                }
            })
            .collect();
        let cfg = super::Gemma4Config {
            num_hidden_layers: 8,
            layer_types,
            num_kv_shared_layers: Some(4),
            sliding_window: 17,
            max_position_embeddings: 128,
            ..paged_tiny_config(None)
        };

        let specs =
            super::compute_layer_kv_cache_specs(&cfg, 8, super::KVCacheDType::BFloat16).unwrap();
        assert_eq!(specs.len(), 8);
        assert_eq!(specs[4].shared_kv_anchor, Some(2));
        assert_eq!(specs[5].shared_kv_anchor, Some(3));
        assert_eq!(super::physical_full_attention_layer_count(&specs), 2);

        let groups =
            super::compute_layer_kv_cache_groups(&cfg, 8, super::KVCacheDType::BFloat16, 32)
                .unwrap();
        let full_group = groups
            .iter()
            .find(|group| matches!(group.attention_kind, super::AttentionKind::Full))
            .expect("full group");
        assert_eq!(full_group.layer_indices, vec![1, 3, 5, 7]);
        assert_eq!(full_group.physical_layer_indices, vec![1, 3]);

        let sliding_group = groups
            .iter()
            .find(|group| {
                matches!(
                    group.attention_kind,
                    super::AttentionKind::SlidingWindow { sliding_window: 17 }
                )
            })
            .expect("sliding group");
        assert_eq!(sliding_group.layer_indices, vec![0, 2, 4, 6]);
        assert_eq!(sliding_group.physical_layer_indices, vec![0, 2]);
        assert_eq!(
            sliding_group.max_admission_blocks, 7,
            "ceil((17 - 1 + 32) / 8) + one partial block"
        );
    }
}

#[cfg(test)]
mod prefix_cache_reuse_integration_tests {
    //! End-to-end tests for the prefix KV cache reuse refactor on Gemma4.
    //! These verify that `chat_session_start_sync` no longer
    //! unconditionally wipes the cache — stateless agent clients that
    //! resend the full transcript on every turn should hit the
    //! `verify_cache_prefix` exact-append path and skip redundant
    //! prefill work.
    //!
    //! The Gemma4 variant additionally locks in the exact-match policy:
    //! when the new prompt equals the cached one
    //! (`cached_prefix_len == tokens.len()`), we fall through to the
    //! miss branch and do a full reset + re-prefill. Gemma4 has no
    //! snapshot of final-step logits and no safe rewind-by-1 primitive
    //! over its sliding-window cache; reprefilling the last cached token
    //! on top of the live caches would advance cache state to
    //! `prompt + last_token` (duplicated) while the history write-back
    //! block only persists `tokens + generated`, corrupting the next
    //! warm-hit turn.
    //!
    //! These tests are `#[ignore]`-marked because they require loading a
    //! real Gemma4 model file and a tokenizer. Run them with:
    //!
    //!     cargo test -p mlx-core --test '*' -- --ignored prefix_cache_reuse_integration
    //!
    //! with `MLX_NODE_GEMMA4_MODEL_DIR` set to a local Gemma4 model dir.

    /// Append hit: two back-to-back session-start calls where the second
    /// extends the first by exactly one user turn. Must report
    /// `cached_tokens > 0` and only prefill the delta.
    #[ignore = "requires a real Gemma4 model directory; run with --ignored"]
    #[test]
    fn append_hit_reuses_cached_prefix() {
        // Pseudocode (same shape as the Qwen3.5 Dense stubs):
        //
        //   let p = vec![ChatMessage::user("Hi")];
        //   let r1 = model.chat_session_start_sync(p.clone(), cfg())?;
        //   let mut p2 = p.clone();
        //   p2.push(ChatMessage::assistant(&r1.text));
        //   p2.push(ChatMessage::user("Follow-up"));
        //   let r2 = model.chat_session_start_sync(p2, cfg())?;
        //   assert!(r2.cached_tokens > 0);
    }

    /// Divergence miss: second call's history is unrelated. Must report
    /// `cached_tokens == 0` and do a full-history prefill.
    #[ignore = "requires a real Gemma4 model directory; run with --ignored"]
    #[test]
    fn divergence_miss_resets_and_full_prefills() {
        // Pseudocode:
        //
        //   let p1 = vec![ChatMessage::user("Ping")];
        //   let p2 = vec![ChatMessage::user("Totally unrelated")];
        //   let _ = model.chat_session_start_sync(p1, cfg())?;
        //   let r2 = model.chat_session_start_sync(p2, cfg())?;
        //   assert_eq!(r2.cached_tokens, 0);
    }

    /// Exact-match: the new prompt is byte-equal to the cached one.
    /// With the exact-match-as-miss fix, the second call must report
    /// `cached_tokens == 0` (full reset + full re-prefill). A subsequent
    /// strict-extension must then hit the warm path.
    #[ignore = "requires a real Gemma4 model directory; run with --ignored"]
    #[test]
    fn exact_match_falls_through_to_cache_miss() {
        // Pseudocode:
        //
        //   let p = vec![ChatMessage::user("Ping")];
        //   let _ = model.chat_session_start_sync(p.clone(), cfg())?;
        //   let r2 = model.chat_session_start_sync(p.clone(), cfg())?;
        //   assert_eq!(r2.cached_tokens, 0); // miss, not exact-match reuse
        //
        //   // After the miss, the caches represent `p` cleanly. A strict
        //   // extension should warm-hit against that fresh state.
        //   let prompt_token_count_p = r2.prompt_token_count;
        //   let mut p3 = p.clone();
        //   p3.push(ChatMessage::assistant(&r2.text));
        //   p3.push(ChatMessage::user("Follow-up"));
        //   let r3 = model.chat_session_start_sync(p3, cfg())?;
        //   assert!(r3.cached_tokens >= prompt_token_count_p);
    }
}

#[cfg(test)]
mod prefix_cache_decision_tests {
    //! Pure-logic coverage of the prefix-cache decision tree — no model
    //! load required. The verifier `Gemma4Inner::verify_cache_prefix`
    //! returns either `0` (miss) or `cached_token_history.len()` (exact
    //! prefix relation). The engine session core
    //! (`engine::session::chat_turn_core`) then classifies that
    //! value plus the incoming prompt length into
    //! [`PrefixCacheDecision::StrictExtendHit`] (warm-reuse, skip the
    //! cached prefix, prefill only the tail) vs
    //! [`PrefixCacheDecision::Miss`] (reset caches + re-init + full
    //! prefill).
    //!
    //! The four cases covered below pin the invariant: exact-match MUST
    //! route to `Miss`, not to `StrictExtendHit`. Treating exact-match as a
    //! shortcut would corrupt the next warm-hit turn by advancing cache
    //! state to `prompt + last_token` while the history write-back only
    //! persists `tokens + generated`. The `#[ignore]`-gated integration
    //! tests above exercise the end-to-end behaviour against a loaded
    //! Gemma4 model; this module guarantees the decision logic stays
    //! correct in every CI run without a model dependency.

    use super::{PrefixCacheDecision, classify_prefix_cache_decision};

    #[test]
    fn empty_cache_is_miss() {
        // verify_cache_prefix returned 0 (cached_token_history empty,
        // reuse_cache disabled, has_images guard, or prefix mismatch).
        // Regardless of tokens.len(), the classifier routes to Miss so
        // the caller runs reset_caches_sync + init_caches_sync + full
        // prefill.
        assert_eq!(
            classify_prefix_cache_decision(0, 0),
            PrefixCacheDecision::Miss,
            "empty cache + empty tokens must be Miss"
        );
        assert_eq!(
            classify_prefix_cache_decision(0, 10),
            PrefixCacheDecision::Miss,
            "empty cache + non-empty tokens must be Miss"
        );
    }

    #[test]
    fn strict_extend_is_hit() {
        // verify_cache_prefix returned cached_token_history.len() AND
        // tokens.len() > cached_token_history.len() — the new prompt
        // strictly extends the cached one. This is the only case that
        // takes the warm-reuse path: prefill_offset = cached_prefix_len,
        // so only the tail delta is prefilled.
        assert_eq!(
            classify_prefix_cache_decision(5, 8),
            PrefixCacheDecision::StrictExtendHit,
            "cached.len() < tokens.len() must be StrictExtendHit"
        );
        assert_eq!(
            classify_prefix_cache_decision(1, 2),
            PrefixCacheDecision::StrictExtendHit,
            "cached.len() = 1, tokens.len() = 2 must be StrictExtendHit (smallest hit)"
        );
    }

    #[test]
    fn divergence_is_miss() {
        // verify_cache_prefix returned 0 because tokens[..cached.len()]
        // != cached[..] — semantically a divergence even though we only
        // observe the 0 return here. Same code path as `empty_cache_is_miss`
        // — both flavours of Miss fall into the same branch.
        assert_eq!(
            classify_prefix_cache_decision(0, 20),
            PrefixCacheDecision::Miss,
            "divergence (verifier returned 0) must be Miss"
        );
    }

    #[test]
    fn exact_match_is_miss() {
        // verify_cache_prefix returned cached_token_history.len() AND
        // tokens.len() == cached_token_history.len() — byte-equal
        // prompt. The classifier routes to Miss because Gemma4 has no
        // snapshot of final-step logits and no safe "rewind by 1"
        // primitive over the sliding-window cache. Reprefilling the
        // last cached token over the live caches would advance cache
        // state to `prompt + last_token` (duplicated) while the
        // history write-back persists `tokens + generated`, desyncing
        // cache and history for the next warm-hit turn.
        //
        // This invariant guards against silently corrupting multi-turn
        // correctness.
        assert_eq!(
            classify_prefix_cache_decision(5, 5),
            PrefixCacheDecision::Miss,
            "exact-match (cached.len() == tokens.len()) must be Miss, not StrictExtendHit"
        );
        assert_eq!(
            classify_prefix_cache_decision(1, 1),
            PrefixCacheDecision::Miss,
            "exact-match single token must be Miss"
        );
        assert_eq!(
            classify_prefix_cache_decision(1000, 1000),
            PrefixCacheDecision::Miss,
            "exact-match long prompts must still be Miss"
        );
    }

    #[test]
    fn invariant_cached_len_never_exceeds_tokens_len_in_hit() {
        // Belt-and-braces: the verifier itself returns 0 when
        // tokens.len() < cached.len() (no partial-cache reuse), so
        // `cached_prefix_len > tokens_len` should never be observed by
        // the classifier in practice. But if it ever was, the branch
        // routes it to Miss (cached_prefix_len < tokens_len is false),
        // which is the safe fallthrough.
        assert_eq!(
            classify_prefix_cache_decision(10, 5),
            PrefixCacheDecision::Miss,
            "cached_prefix_len > tokens_len must be Miss (defensive fallthrough)"
        );
    }
}

#[cfg(test)]
mod dspark_tap_tests {
    //! Tap purity: threading a `DsparkTap` through the Gemma4 forward
    //! paths must leave the compute graph byte-identical to a tap-less
    //! run, while capturing the residual-stream hiddens of the tapped
    //! layers. Runs a tiny random-weight Gemma4 (4 layers, hybrid
    //! sliding/global types, one KV-shared layer) through the REAL
    //! `forward_body` / `forward_inner` / `dspark_verify_forward` paths.

    use super::*;
    use crate::models::gemma4::dspark::DsparkTap;

    fn tiny_config() -> Gemma4Config {
        serde_json::from_value(serde_json::json!({
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "intermediate_size": 64,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "max_position_embeddings": 128,
            "sliding_window": 8,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ],
            "num_kv_shared_layers": 1,
        }))
        .expect("tiny Gemma4 config must deserialize")
    }

    pub(super) fn tiny_model(
        config: &Gemma4Config,
    ) -> (Embedding, Vec<Gemma4DecoderLayer>, RMSNorm) {
        let embedding =
            Embedding::new(config.vocab_size as u32, config.hidden_size as u32).unwrap();
        let layers: Vec<Gemma4DecoderLayer> = (0..config.num_hidden_layers as usize)
            .map(|i| Gemma4DecoderLayer::new(config, i).unwrap())
            .collect();
        let final_norm =
            RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps)).unwrap();
        (embedding, layers, final_norm)
    }

    pub(super) fn assert_bitwise_eq(a: &MxArray, b: &MxArray, ctx: &str) {
        a.eval();
        b.eval();
        assert_eq!(
            a.shape().unwrap().to_vec(),
            b.shape().unwrap().to_vec(),
            "{ctx}: shape"
        );
        let a_bits: Vec<u32> = a
            .to_float32()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let b_bits: Vec<u32> = b
            .to_float32()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(a_bits, b_bits, "{ctx}: bits");
    }

    #[test]
    fn dspark_tap_purity_and_verify_forward() {
        let config = tiny_config();
        let (embedding, layers, final_norm) = tiny_model(&config);

        // 6-token prefill then a 3-token verify block: the block runs
        // T>1 at offset 6, which also crosses the sliding window (6+3 > 8)
        // so the windowed-mask path is exercised.
        let prefill_ids = MxArray::from_int32(&[3, 9, 17, 25, 33, 41], &[1, 6]).unwrap();
        let block_ids = MxArray::from_int32(&[7, 11, 13], &[1, 3]).unwrap();

        // Pass A: no tap.
        let mut caches_a = init_caches_for_config(&config);
        let hidden_a = forward_body(
            Some(&prefill_ids),
            None,
            &embedding,
            &layers,
            &mut caches_a,
            &final_norm,
            None,
            None,
            &config,
            None,
        )
        .unwrap();
        let logits_a = forward_inner(
            &block_ids,
            &embedding,
            &layers,
            &mut caches_a,
            &final_norm,
            &None,
            None,
            None,
            &config,
            None,
        )
        .unwrap();

        // Pass B: tapped, including the KV-shared layer 3 (anchor = layer 1),
        // with the real snapshot → verify → commit flow around the verify.
        let layer_ids = [0usize, 2, 3];
        let shared_slots = dspark_shared_slot_mask(&config);
        assert_eq!(
            shared_slots,
            vec![false, false, false, true],
            "config-derived shared-slot mask"
        );
        let mut caches_b = init_caches_for_config(&config);
        let mut prefill_tap = DsparkTap::new(&layer_ids);
        let hidden_b = forward_body(
            Some(&prefill_ids),
            None,
            &embedding,
            &layers,
            &mut caches_b,
            &final_norm,
            None,
            None,
            &config,
            Some(&mut prefill_tap),
        )
        .unwrap();
        let rollback = super::super::layer_cache::snapshot_before_verify(
            &caches_b,
            block_ids.shape_at(1).unwrap() as usize,
            &shared_slots,
        )
        .unwrap();
        let mut verify_tap = DsparkTap::new(&layer_ids);
        let logits_b = dspark_verify_forward(
            &block_ids,
            &embedding,
            &layers,
            &mut caches_b,
            &final_norm,
            &None,
            None,
            None,
            &config,
            &mut verify_tap,
        )
        .unwrap();

        // Tap must not perturb the compute graph.
        assert_bitwise_eq(&hidden_a, &hidden_b, "prefill hidden");
        assert_bitwise_eq(&logits_a, &logits_b, "verify logits");
        assert_eq!(logits_b.shape().unwrap().to_vec(), vec![1, 3, 64]);

        // One [B, T, hidden] capture per tapped layer, per forward call.
        assert_eq!(prefill_tap.captured.len(), layer_ids.len());
        for arr in &prefill_tap.captured {
            assert_eq!(arr.shape().unwrap().to_vec(), vec![1, 6, 32]);
        }
        assert_eq!(verify_tap.captured.len(), layer_ids.len());
        for arr in &verify_tap.captured {
            assert_eq!(arr.shape().unwrap().to_vec(), vec![1, 3, 32]);
        }

        // Different layers must yield different hiddens (real per-layer
        // captures, not one array pushed repeatedly).
        let first = verify_tap.captured[0].to_float32().unwrap().to_vec();
        let second = verify_tap.captured[1].to_float32().unwrap().to_vec();
        assert_ne!(first, second, "captures must differ across layers");

        // Caches advance by T on both passes; the KV-shared layer's own
        // vec entry is never written (it reads its anchor's cache).
        for (idx, cache) in caches_b.iter().enumerate().take(3) {
            assert_eq!(cache.get_offset(), 9, "cache {idx} offset");
            assert_eq!(caches_a[idx].get_offset(), 9, "cache {idx} tapless offset");
        }
        assert_eq!(
            caches_b[3].get_offset(),
            0,
            "KV-shared layer's cache entry must stay untouched"
        );

        // Partial-keep commit on the real model: active caches land at
        // prefill + keep, the shared slot stays untouched.
        super::super::layer_cache::commit_after_verify(&mut caches_b, &rollback, 1).unwrap();
        for (idx, cache) in caches_b.iter().enumerate().take(3) {
            assert_eq!(cache.get_offset(), 7, "cache {idx} post-commit offset");
        }
        assert_eq!(
            caches_b[3].get_offset(),
            0,
            "KV-shared layer's cache entry must stay untouched after commit"
        );
    }

    #[test]
    fn dspark_tap_rejects_unsorted_or_out_of_range_layer_ids() {
        let config = tiny_config();
        let (embedding, layers, final_norm) = tiny_model(&config);
        let ids = MxArray::from_int32(&[3, 9], &[1, 2]).unwrap();

        for bad in [vec![2usize, 0], vec![1, 1], vec![7]] {
            let mut caches = init_caches_for_config(&config);
            let mut tap = DsparkTap::new(&bad);
            let result = forward_body(
                Some(&ids),
                None,
                &embedding,
                &layers,
                &mut caches,
                &final_norm,
                None,
                None,
                &config,
                Some(&mut tap),
            );
            assert!(result.is_err(), "layer_ids {bad:?} must be rejected");
        }
    }

    #[test]
    fn dspark_verify_forward_rejects_bad_block_shape() {
        let config = tiny_config();
        let (embedding, layers, final_norm) = tiny_model(&config);
        let layer_ids = [0usize];

        // Batch > 1 is rejected.
        let batch2 = MxArray::from_int32(&[1, 2], &[2, 1]).unwrap();
        let mut caches = init_caches_for_config(&config);
        let mut tap = DsparkTap::new(&layer_ids);
        assert!(
            dspark_verify_forward(
                &batch2,
                &embedding,
                &layers,
                &mut caches,
                &final_norm,
                &None,
                None,
                None,
                &config,
                &mut tap,
            )
            .is_err()
        );

        // 1-D input is rejected.
        let flat = MxArray::from_int32(&[1, 2], &[2]).unwrap();
        let mut caches = init_caches_for_config(&config);
        let mut tap = DsparkTap::new(&layer_ids);
        assert!(
            dspark_verify_forward(
                &flat,
                &embedding,
                &layers,
                &mut caches,
                &final_norm,
                &None,
                None,
                None,
                &config,
                &mut tap,
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod assistant_seam_tests {
    //! Target-side seams for the assistant draft model: the K/V source
    //! mapping (which target caches the draft reads), the extracted
    //! `lm_head_logits` tail, and `assistant_verify_forward` (verify logits
    //! plus the post-final-norm hidden the draft chains from). Runs a tiny
    //! random-weight Gemma4 (4 hybrid layers, one KV-shared) through the
    //! REAL forward paths.

    use super::dspark_tap_tests::{assert_bitwise_eq, tiny_model};
    use super::*;
    use crate::models::gemma4::dspark::DsparkTap;

    /// Tiny flat-path Gemma4 config (mirrors the DSpark decode tests):
    /// 4 hybrid layers, one KV-shared.
    fn tiny_target_config() -> Gemma4Config {
        serde_json::from_value(tiny_target_config_value())
            .expect("tiny Gemma4 config must deserialize")
    }

    fn tiny_target_config_value() -> serde_json::Value {
        serde_json::json!({
            "vocab_size": 16,
            "hidden_size": 8,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "intermediate_size": 16,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "max_position_embeddings": 128,
            "sliding_window": 8,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ],
            "num_kv_shared_layers": 1,
            "use_block_paged_cache": false,
            "eos_token_ids": []
        })
    }

    /// [`tiny_target_config`] with overridden layer types and KV sharing
    /// (the two inputs `assistant_kv_source_indices` reads).
    fn hybrid_config(layer_types: &[&str], num_kv_shared_layers: Option<i32>) -> Gemma4Config {
        let mut v = tiny_target_config_value();
        v["layer_types"] = serde_json::json!(layer_types);
        v["num_hidden_layers"] = serde_json::json!(layer_types.len());
        match num_kv_shared_layers {
            Some(n) => v["num_kv_shared_layers"] = serde_json::json!(n),
            None => {
                v.as_object_mut()
                    .expect("tiny config value is an object")
                    .remove("num_kv_shared_layers");
            }
        }
        serde_json::from_value(v).expect("tiny Gemma4 config must deserialize")
    }

    // ── K/V source mapping ─────────────────────────────────────────────

    /// With one KV-shared layer the non-shared prefix is [s, f, s]: the
    /// draft reads the last sliding layer (2) and the last full layer (1)
    /// — exactly the anchors `should_store_shared_kv` marks.
    #[test]
    fn kv_source_indices_pick_last_non_shared_layer_of_each_type() {
        let config = hybrid_config(
            &[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            Some(1),
        );
        let sources = assistant_kv_source_indices(&config).expect("mapping must resolve");
        assert_eq!(
            sources,
            AssistantKvSources {
                sliding: 2,
                full: 1
            }
        );
    }

    /// Without KV sharing the boundary is num_hidden_layers, so the mapping
    /// is simply the last layer of each type.
    #[test]
    fn kv_source_indices_without_sharing_pick_last_layer_of_each_type() {
        let config = hybrid_config(
            &[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            None,
        );
        let sources = assistant_kv_source_indices(&config).expect("mapping must resolve");
        assert_eq!(
            sources,
            AssistantKvSources {
                sliding: 2,
                full: 3
            }
        );
    }

    /// A non-shared prefix lacking either attention type is a hard error —
    /// the draft needs one K/V source per type.
    #[test]
    fn kv_source_indices_error_when_type_missing_below_boundary() {
        // Prefix [sliding, sliding] has no full_attention layer.
        let config = hybrid_config(
            &[
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "full_attention",
            ],
            Some(2),
        );
        let err = assistant_kv_source_indices(&config).expect_err("missing full layer must error");
        assert!(err.reason.contains("full_attention"), "got: {}", err.reason);

        // Prefix [full, full] has no sliding_attention layer.
        let config = hybrid_config(
            &[
                "full_attention",
                "full_attention",
                "sliding_attention",
                "sliding_attention",
            ],
            Some(2),
        );
        let err =
            assistant_kv_source_indices(&config).expect_err("missing sliding layer must error");
        assert!(
            err.reason.contains("sliding_attention"),
            "got: {}",
            err.reason
        );
    }

    /// A truncated `layer_types` vec leaves trailing layers without an
    /// entry. Such layers match neither attention type: the mapping resolves
    /// only exact `layer_types` entries (like `should_store_shared_kv`) and
    /// errors when a type has no exact entry below the boundary, instead of
    /// treating the missing entries as full attention.
    #[test]
    fn kv_source_indices_ignore_layers_with_missing_layer_types_entry() {
        // 4 layers, `layer_types` truncated to 2 entries, no KV sharing.
        let truncated = |layer_types: &[&str]| -> Gemma4Config {
            let mut v = tiny_target_config_value();
            v["layer_types"] = serde_json::json!(layer_types);
            v.as_object_mut()
                .expect("tiny config value is an object")
                .remove("num_kv_shared_layers");
            serde_json::from_value(v).expect("tiny Gemma4 config must deserialize")
        };

        // Both types have exact entries: layers 2/3 (no entry) must not be
        // selected even though the full-attention fallback would claim them.
        let sources =
            assistant_kv_source_indices(&truncated(&["sliding_attention", "full_attention"]))
                .expect("mapping must resolve from the exact entries");
        assert_eq!(
            sources,
            AssistantKvSources {
                sliding: 0,
                full: 1
            }
        );

        // No exact full_attention entry anywhere: hard error, not index 3.
        let err =
            assistant_kv_source_indices(&truncated(&["sliding_attention", "sliding_attention"]))
                .expect_err("missing full_attention entry must error");
        assert!(err.reason.contains("full_attention"), "got: {}", err.reason);
    }

    // ── lm_head tail extraction ────────────────────────────────────────

    /// `forward_body` + `lm_head_logits` composed by hand must reproduce
    /// `forward_inner` bitwise, with and without logit softcapping.
    #[test]
    fn lm_head_logits_matches_forward_inner() {
        let mut capped = tiny_target_config_value();
        capped["final_logit_softcapping"] = serde_json::json!(30.0);
        let configs: [Gemma4Config; 2] = [
            tiny_target_config(),
            serde_json::from_value(capped).expect("tiny Gemma4 config must deserialize"),
        ];
        for config in &configs {
            let (embedding, layers, final_norm) = tiny_model(config);
            let ids = MxArray::from_int32(&[3, 9, 1, 5], &[1, 4]).unwrap();

            let mut caches_a = init_caches_for_config(config);
            let logits_a = forward_inner(
                &ids,
                &embedding,
                &layers,
                &mut caches_a,
                &final_norm,
                &None,
                None,
                None,
                config,
                None,
            )
            .unwrap();

            let mut caches_b = init_caches_for_config(config);
            let hidden = forward_body(
                Some(&ids),
                None,
                &embedding,
                &layers,
                &mut caches_b,
                &final_norm,
                None,
                None,
                config,
                None,
            )
            .unwrap();
            let logits_b = lm_head_logits(&hidden, &embedding, &None, None, config).unwrap();

            let ctx = format!(
                "lm_head tail (softcap {:?})",
                config.final_logit_softcapping
            );
            assert_bitwise_eq(&logits_a, &logits_b, &ctx);
        }
    }

    // ── assistant verify forward ───────────────────────────────────────

    /// Same forward as `dspark_verify_forward` (bitwise-equal logits against
    /// an empty-tap run on equivalent fresh caches), plus the post-final-norm
    /// hidden as the second tuple element; caches advance by T and bad block
    /// shapes are rejected.
    #[test]
    fn assistant_verify_forward_returns_hidden_and_logits() {
        let config = tiny_target_config();
        let (embedding, layers, final_norm) = tiny_model(&config);

        // 6-token prefill then a 3-token verify block: the block runs T>1
        // at offset 6 and crosses the sliding window (6+3 > 8).
        let prefill_ids = MxArray::from_int32(&[3, 9, 1, 5, 2, 8], &[1, 6]).unwrap();
        let block_ids = MxArray::from_int32(&[7, 11, 13], &[1, 3]).unwrap();
        let prefill = |caches: &mut [Gemma4LayerCache]| {
            forward_body(
                Some(&prefill_ids),
                None,
                &embedding,
                &layers,
                caches,
                &final_norm,
                None,
                None,
                &config,
                None,
            )
            .unwrap()
        };

        // Reference: dspark_verify_forward with an EMPTY tap.
        let mut caches_a = init_caches_for_config(&config);
        prefill(&mut caches_a);
        let mut tap = DsparkTap::new(&[]);
        let logits_a = dspark_verify_forward(
            &block_ids,
            &embedding,
            &layers,
            &mut caches_a,
            &final_norm,
            &None,
            None,
            None,
            &config,
            &mut tap,
        )
        .unwrap();
        assert!(tap.captured.is_empty(), "empty tap must capture nothing");

        // Assistant seam on equivalent fresh caches.
        let mut caches_b = init_caches_for_config(&config);
        prefill(&mut caches_b);
        let (logits_b, hidden) = assistant_verify_forward(
            &block_ids,
            &embedding,
            &layers,
            &mut caches_b,
            &final_norm,
            &None,
            None,
            None,
            &config,
        )
        .unwrap();

        assert_eq!(logits_b.shape().unwrap().to_vec(), vec![1, 3, 16]);
        assert_eq!(hidden.shape().unwrap().to_vec(), vec![1, 3, 8]);
        assert_bitwise_eq(&logits_a, &logits_b, "verify logits");

        // The hidden is the post-final-norm state of the same block forward.
        let mut caches_c = init_caches_for_config(&config);
        prefill(&mut caches_c);
        let hidden_ref = forward_body(
            Some(&block_ids),
            None,
            &embedding,
            &layers,
            &mut caches_c,
            &final_norm,
            None,
            None,
            &config,
            None,
        )
        .unwrap();
        assert_bitwise_eq(&hidden_ref, &hidden, "post-final-norm hidden");

        // Caches advance by T; the KV-shared layer's own vec entry is never
        // written (it reads its anchor's cache).
        for (idx, cache) in caches_b.iter().enumerate().take(3) {
            assert_eq!(cache.get_offset(), 9, "cache {idx} offset");
        }
        assert_eq!(
            caches_b[3].get_offset(),
            0,
            "KV-shared layer's cache entry must stay untouched"
        );

        // Bad block shapes are rejected: batch > 1 and 1-D input.
        for bad in [
            MxArray::from_int32(&[1, 2], &[2, 1]).unwrap(),
            MxArray::from_int32(&[1, 2], &[2]).unwrap(),
        ] {
            let mut caches = init_caches_for_config(&config);
            assert!(
                assistant_verify_forward(
                    &bad,
                    &embedding,
                    &layers,
                    &mut caches,
                    &final_norm,
                    &None,
                    None,
                    None,
                    &config,
                )
                .is_err(),
                "block shape {:?} must be rejected",
                bad.shape().unwrap().as_ref()
            );
        }
    }
}
