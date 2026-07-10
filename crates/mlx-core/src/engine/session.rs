//! Generic session-turn cores driving [`ChatBackend`].
//!
//! One private [`chat_turn_core`] runs the session skeleton for every
//! family:
//!
//! ```text
//! reuse_cache guard → tokenizer → resolve_params
//!   → pre-render image guard → render_prompt
//!   → resolve TurnPlan → optional multimodal/paged/speculative executor
//!   → verify_cache_prefix → reset-or-delta split → prefill
//!   → first-token sample (apply_all_penalties + sampling::sample)
//!   → eval_caches → begin_decode → run_decode_loop → end_decode
//!   → save_cache_state → finalize_turn (+ cached_tokens overwrite)
//! ```
//!
//! Everywhere families genuinely differ, the difference is a
//! [`ChatBackend`] hook (documented on the trait), never a branch on
//! family. The 8 public entry points (4 sync + 4 streaming twins) are
//! thin guard wrappers around the core.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{
    ChatBackend, ChunkSink, DecodeStep, FinalizeArgs, ResetScope, SaveStateArgs, StreamEmitter,
    TurnOutput, TurnSetup, WholeTurnArgs,
};
use crate::engine::cache::IMAGE_CHANGE_RESTART_PREFIX;
use crate::engine::decode::{DecodeLoopArgs, StreamingCtx, run_decode_loop};
use crate::engine::finalize::compute_performance_metrics;
use crate::engine::params::generated_capacity_hint;
use crate::engine::penalties::{ReasoningTracker, apply_all_penalties};
use crate::engine::plan::{MediaCapabilities, MediaInputs, TurnPath, TurnPlan, TurnRequest};
use crate::engine::types::{ChatConfig, ChatResult};
use crate::stream::{DeviceType, Stream};
use crate::tokenizer::ChatMessage;

/// What kind of turn the core is running.
enum TurnInput {
    /// Fresh prompt rendered from messages (session-start). Runs the
    /// full verify-prefix / reset-or-delta split.
    Fresh { messages: Vec<ChatMessage> },
    /// Pre-tokenized delta appended on top of the live caches
    /// (session-continue / tool / raw tokens-delta). Strict extension
    /// by construction — skips prefix verification.
    Delta { delta_tokens: Vec<u32> },
}

/// Streaming context handed to [`chat_turn_core`] by the streaming
/// twins: the chunk sink plus the cancel flag. Only `.load(Relaxed)` is
/// used on the flag, so a plain `&AtomicBool` suffices; `Arc` derefs at
/// the call sites.
struct StreamingHooks<'a> {
    sink: &'a dyn ChunkSink,
    cancelled: &'a AtomicBool,
}

// =====================================================================
// Sync entry points
// =====================================================================

/// Generic `chat_session_start_sync`: full-prompt session turn with
/// `<|im_end|>`-style session EOS and internal prefix-cache reuse.
///
/// Rejects an explicit `reuse_cache=false` up front (the session API
/// only makes sense with cache reuse; accepting it would let the
/// post-decode save path wipe the caches the next continue call depends
/// on), then delegates to the core. NOTE: no unconditional reset here —
/// prefix-reuse support requires the core to decide reset-vs-reuse from
/// `verify_cache_prefix` (stateless-agent clients resend the full
/// transcript every turn).
pub(crate) fn session_start<B: ChatBackend>(
    backend: &mut B,
    messages: Vec<ChatMessage>,
    config: ChatConfig,
) -> Result<ChatResult> {
    if config.reuse_cache == Some(false) {
        return Err(Error::from_reason(
            "chat_session_start requires reuse_cache=true (pass ChatConfig { reuse_cache: Some(true), .. } or leave as None). The session API only makes sense with cache reuse enabled.",
        ));
    }
    expect_sync_result(chat_turn_core(
        backend,
        TurnInput::Fresh { messages },
        config,
        None,
    ))
}

/// Generic `chat_session_continue_sync`: build the family's
/// continue-delta via [`ChatBackend::render_continue_delta`] and run it
/// through the delta path.
///
/// `images` / `audio` are the opt-in guard parameters shared by every
/// family: non-empty input is rejected with the
/// `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` prefix so the TS
/// `ChatSession` layer can route an image/audio change back through a
/// fresh session start (the continue/delta path is text-only).
pub(crate) fn session_continue<B: ChatBackend>(
    backend: &mut B,
    user_message: String,
    images: Option<Vec<Uint8Array>>,
    audio: Option<Vec<Uint8Array>>,
    config: ChatConfig,
) -> Result<ChatResult> {
    if images.as_ref().is_some_and(|v| !v.is_empty()) {
        return Err(Error::from_reason(format!(
            "{IMAGE_CHANGE_RESTART_PREFIX} chat_session_continue is text-only; start a new session with chat_session_start to change the image",
        )));
    }
    if audio.as_ref().is_some_and(|v| !v.is_empty()) {
        return Err(Error::from_reason(format!(
            "{IMAGE_CHANGE_RESTART_PREFIX} chat_session_continue is text-only; start a new session with chat_session_start to change the audio",
        )));
    }
    let tokenizer = backend.tokenizer()?;
    let delta_tokens = backend.render_continue_delta(&tokenizer, &user_message, &config)?;
    tokens_delta(backend, delta_tokens, config)
}

/// Generic `chat_session_continue_tool_sync`: build the family's
/// tool-result delta via [`ChatBackend::render_tool_delta`] and run it
/// through the delta path.
pub(crate) fn session_continue_tool<B: ChatBackend>(
    backend: &mut B,
    tool_call_id: String,
    content: String,
    is_error: Option<bool>,
    config: ChatConfig,
) -> Result<ChatResult> {
    let tokenizer = backend.tokenizer()?;
    let delta_tokens =
        backend.render_tool_delta(&tokenizer, &tool_call_id, &content, is_error, &config)?;
    tokens_delta(backend, delta_tokens, config)
}

/// Generic `chat_tokens_delta_sync`: prefill a pre-tokenized delta on
/// top of the live caches and decode the reply.
pub(crate) fn tokens_delta<B: ChatBackend>(
    backend: &mut B,
    delta_tokens: Vec<u32>,
    config: ChatConfig,
) -> Result<ChatResult> {
    delta_guards(backend, &delta_tokens, &config, DeltaEntry::Sync)?;
    expect_sync_result(chat_turn_core(
        backend,
        TurnInput::Delta { delta_tokens },
        config,
        None,
    ))
}

// =====================================================================
// Streaming twins
// =====================================================================

/// Streaming twin of [`session_start`]. Guard failures and errors are
/// delivered through the sink as `Err` items (see
/// [`crate::engine::finalize::send_stream_error`]'s rustdoc for why
/// they must NOT be fake done-chunks), never returned.
pub(crate) fn session_start_stream<B: ChatBackend>(
    backend: &mut B,
    messages: Vec<ChatMessage>,
    config: ChatConfig,
    sink: &dyn ChunkSink,
    cancelled: &AtomicBool,
) {
    if cancelled.load(Ordering::Relaxed) {
        sink.send(Err(Error::from_reason(
            "chat_stream_session_start cancelled before start",
        )));
        return;
    }
    if config.reuse_cache == Some(false) {
        sink.send(Err(Error::from_reason(
            "chat_stream_session_start requires reuse_cache=true (leave as None or set to true). \
             The session API only makes sense with cache reuse enabled.",
        )));
        return;
    }
    if let Err(e) = chat_turn_core(
        backend,
        TurnInput::Fresh { messages },
        config,
        Some(StreamingHooks { sink, cancelled }),
    ) {
        sink.send(Err(e));
    }
}

/// Streaming twin of [`session_continue`].
pub(crate) fn session_continue_stream<B: ChatBackend>(
    backend: &mut B,
    user_message: String,
    images: Option<Vec<Uint8Array>>,
    audio: Option<Vec<Uint8Array>>,
    config: ChatConfig,
    sink: &dyn ChunkSink,
    cancelled: &AtomicBool,
) {
    if cancelled.load(Ordering::Relaxed) {
        sink.send(Err(Error::from_reason(
            "chat_stream_session_continue cancelled before start",
        )));
        return;
    }
    if images.as_ref().is_some_and(|v| !v.is_empty()) {
        sink.send(Err(Error::from_reason(format!(
            "{IMAGE_CHANGE_RESTART_PREFIX} chat_stream_session_continue is text-only; start a new session with chat_stream_session_start to change the image",
        ))));
        return;
    }
    if audio.as_ref().is_some_and(|v| !v.is_empty()) {
        sink.send(Err(Error::from_reason(format!(
            "{IMAGE_CHANGE_RESTART_PREFIX} chat_stream_session_continue is text-only; start a new session with chat_stream_session_start to change the audio",
        ))));
        return;
    }
    let tokenizer = match backend.tokenizer() {
        Ok(t) => t,
        Err(e) => {
            sink.send(Err(e));
            return;
        }
    };
    let delta_tokens = match backend.render_continue_delta(&tokenizer, &user_message, &config) {
        Ok(t) => t,
        Err(e) => {
            sink.send(Err(e));
            return;
        }
    };
    tokens_delta_stream(backend, delta_tokens, config, sink, cancelled);
}

/// Streaming twin of [`session_continue_tool`].
pub(crate) fn session_continue_tool_stream<B: ChatBackend>(
    backend: &mut B,
    tool_call_id: String,
    content: String,
    is_error: Option<bool>,
    config: ChatConfig,
    sink: &dyn ChunkSink,
    cancelled: &AtomicBool,
) {
    if cancelled.load(Ordering::Relaxed) {
        sink.send(Err(Error::from_reason(
            "chat_stream_session_continue_tool cancelled before start",
        )));
        return;
    }
    let tokenizer = match backend.tokenizer() {
        Ok(t) => t,
        Err(e) => {
            sink.send(Err(e));
            return;
        }
    };
    let delta_tokens =
        match backend.render_tool_delta(&tokenizer, &tool_call_id, &content, is_error, &config) {
            Ok(t) => t,
            Err(e) => {
                sink.send(Err(e));
                return;
            }
        };
    tokens_delta_stream(backend, delta_tokens, config, sink, cancelled);
}

/// Streaming twin of [`tokens_delta`].
pub(crate) fn tokens_delta_stream<B: ChatBackend>(
    backend: &mut B,
    delta_tokens: Vec<u32>,
    config: ChatConfig,
    sink: &dyn ChunkSink,
    cancelled: &AtomicBool,
) {
    if cancelled.load(Ordering::Relaxed) {
        sink.send(Err(Error::from_reason(
            "chat_stream_tokens_delta cancelled before start",
        )));
        return;
    }
    if let Err(e) = delta_guards(backend, &delta_tokens, &config, DeltaEntry::Stream) {
        sink.send(Err(e));
        return;
    }
    if let Err(e) = chat_turn_core(
        backend,
        TurnInput::Delta { delta_tokens },
        config,
        Some(StreamingHooks { sink, cancelled }),
    ) {
        sink.send(Err(e));
    }
}

// =====================================================================
// Shared guards / helpers
// =====================================================================

/// Which NAPI entry point invoked the delta path — selects the guard
/// message wording. The sync and streaming delta twins emit DIFFERENT
/// strings (sync names `chat_tokens_delta_sync` / `chatSessionStart`,
/// streaming names `chat_stream_tokens_delta` / `chatStreamSessionStart`);
/// this selector reproduces both.
#[derive(Clone, Copy)]
enum DeltaEntry {
    Sync,
    Stream,
}

impl DeltaEntry {
    /// The entry point's wire name as it appears in guard messages.
    fn fn_name(self) -> &'static str {
        match self {
            DeltaEntry::Sync => "chat_tokens_delta_sync",
            DeltaEntry::Stream => "chat_stream_tokens_delta",
        }
    }

    /// The session-start method named in the "no live session" guard.
    fn start_name(self) -> &'static str {
        match self {
            DeltaEntry::Sync => "chatSessionStart",
            DeltaEntry::Stream => "chatStreamSessionStart",
        }
    }
}

/// The four delta-path guards shared by `chat_tokens_delta_sync` /
/// `chat_stream_tokens_delta_sync` on every family; wording per
/// [`DeltaEntry`].
///
/// gemma4's two distinct live-session checks (empty history vs
/// `caches.is_none()`, the latter AFTER the image guard) fold into the
/// single `has_live_session` check here — see the
/// [`ChatBackend::has_live_session`] rustdoc.
fn delta_guards<B: ChatBackend>(
    backend: &B,
    delta_tokens: &[u32],
    config: &ChatConfig,
    entry: DeltaEntry,
) -> Result<()> {
    if config.reuse_cache == Some(false) {
        return Err(Error::from_reason(format!(
            "{} requires reuse_cache to be enabled; \
             the delta path operates on session state by construction",
            entry.fn_name(),
        )));
    }
    if !backend.has_live_session() {
        return Err(Error::from_reason(format!(
            "{} requires an initialized session (call {} first)",
            entry.fn_name(),
            entry.start_name(),
        )));
    }
    if delta_tokens.is_empty() {
        return Err(Error::from_reason(format!(
            "{} requires a non-empty delta",
            entry.fn_name(),
        )));
    }
    // Family policy hook. Default: text-only families reject deltas while
    // the session holds image state (lfm2's defensive guard);
    // image-capable families accept text deltas on image sessions by
    // design (qwen3.5's sticky image-key contract — see
    // `save_cache_state_after_delta`). Gemma4 overrides to REJECT with
    // the typed restart prefix despite being image-capable.
    if let Some(message) = backend.text_delta_media_guard(entry.fn_name()) {
        return Err(Error::from_reason(message));
    }
    Ok(())
}

/// Unwrap the core's sync-path result. `Ok(None)` means a whole-turn
/// executor returned [`TurnOutput::Streamed`] with no sink attached —
/// a family-impl contract violation, surfaced as an error rather than
/// a panic.
fn expect_sync_result(out: Result<Option<ChatResult>>) -> Result<ChatResult> {
    out?.ok_or_else(|| {
        Error::from_reason(
            "specialized executor returned TurnOutput::Streamed on the sync (sink-less) path",
        )
    })
}

/// Map a specialized executor's outcome into the core's return shape.
///
/// `is_streaming` is the turn's sink presence. The streaming contract
/// (documented on [`TurnOutput`] and the specialized executors): an
/// executor running with a sink attached MUST deliver every chunk —
/// including the terminal done-chunk — through the sink and return
/// [`TurnOutput::Streamed`]. A `Complete` outcome under streaming would
/// otherwise pass through silently and close the stream with NO chunks,
/// NO terminal done-chunk, and NO error (JS consumers hang or treat the
/// empty stream as success). It is rejected here with a loud `Err` that
/// the streaming entry wrappers deliver through the sink exactly like
/// every other streaming error path — deliberately NOT auto-emitted via
/// the emitter, which would mask family bugs. The mirror violation
/// (`Streamed` on the sync, sink-less path) is rejected by
/// [`expect_sync_result`].
fn whole_turn_outcome(out: Result<TurnOutput>, is_streaming: bool) -> Result<Option<ChatResult>> {
    match out? {
        TurnOutput::Complete(result) => {
            if is_streaming {
                return Err(Error::from_reason(
                    "specialized executor returned TurnOutput::Complete on a streaming \
                     (sink-bearing) turn; streaming executors must deliver all output \
                     (including the terminal done-chunk) through the sink and return \
                     TurnOutput::Streamed",
                ));
            }
            Ok(Some(*result))
        }
        TurnOutput::Streamed => Ok(None),
    }
}

/// Collect every image payload from the turn's messages, in order.
fn extract_images_from_messages(messages: &[ChatMessage]) -> Vec<Vec<u8>> {
    let mut all_images: Vec<Vec<u8>> = Vec::new();
    for msg in messages {
        if let Some(ref images) = msg.images {
            for img in images {
                all_images.push(img.to_vec());
            }
        }
    }
    all_images
}

/// Collect every audio payload from the turn's messages, in order. Mirrors
/// [`extract_images_from_messages`] for the unified Gemma 4 audio path.
fn extract_audio_from_messages(messages: &[ChatMessage]) -> Vec<Vec<u8>> {
    let mut all_audio: Vec<Vec<u8>> = Vec::new();
    for msg in messages {
        if let Some(ref audio) = msg.audio {
            for clip in audio {
                all_audio.push(clip.to_vec());
            }
        }
    }
    all_audio
}

// =====================================================================
// The turn core
// =====================================================================

/// One chat turn, generic over the backend.
///
/// Returns `Ok(Some(result))` for sync callers; `Ok(None)` when the
/// turn's output was fully delivered through the streaming sink (the
/// generic streaming flow emits the terminal chunk itself and still
/// returns `Ok(None)`; specialized executors signal the same via
/// [`TurnOutput::Streamed`]).
fn chat_turn_core<B: ChatBackend>(
    backend: &mut B,
    input: TurnInput,
    config: ChatConfig,
    streaming: Option<StreamingHooks<'_>>,
) -> Result<Option<ChatResult>> {
    // --- tokenizer + session EOS + thinking state ---
    let tokenizer = backend.tokenizer()?;
    let eos_id = backend.session_eos_id(&tokenizer)?;
    let think_end_id = tokenizer.think_end_id();
    let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());

    // Family-resolved params: the default is the config-only
    // `extract_chat_params`; gemma4 folds its model-config sampling
    // defaults (unset → greedy argmax), neutralizes penalties, and forces
    // report_performance. Everything below reads the RESOLVED params,
    // never raw config.
    let p = backend.resolve_params(&config);
    let reuse_cache = p.reuse_cache;
    let report_perf = p.report_performance;
    let max_new_tokens = p.max_new_tokens;
    let thinking = backend.thinking_setup(&config);
    // Immutable load-time capabilities, read once for this turn. The hot
    // decode loop consumes the resolved `TurnPlan` and never probes them.
    let execution = backend.execution_plan();
    // `backend_validated` does not claim an encoder exists. It only admits
    // the request through this generic boundary so the family's multimodal
    // handler can retain its more precise validation/error contract.
    let admitted_media = execution.media.admitted();

    // --- template/render: full prompt tokens for this turn ---
    // Fresh: family render hook (default = the jinja chat-template path,
    // gemma4 adds its manual `<|turn>` fallback). Delta: cached history +
    // delta (the delta paths skip the template entirely — the caller owns
    // cache coherence by construction).
    let (tokens, images, audio, is_delta, prior_cached_len) = match &input {
        TurnInput::Fresh { messages } => {
            // Pre-render image guard — TS `ChatSession` restart-routing
            // contract: a text-only backend MUST reject an image-bearing
            // fresh turn with the typed
            // `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`-prefixed error,
            // and that rejection MUST happen BEFORE `render_prompt`.
            // `serialize_message_for_jinja` represents image user
            // content as an array, so a text-only family's chat
            // template could otherwise fail with an UNTYPED template
            // error first, breaking the prefix routing. Vision-capable
            // backends with image capability — including
            // gemma4's unconditional policy, whose "no vision support"
            // error surfaces from inside the multimodal executor) skip the
            // rejection, render normally, and route through the
            // multimodal executor below with these exact extracted
            // images (single extraction — no drift).
            let images = extract_images_from_messages(messages);
            if !images.is_empty() && !admitted_media.images {
                return Err(Error::from_reason(format!(
                    "{IMAGE_CHANGE_RESTART_PREFIX} this model is text-only; image messages are not supported",
                )));
            }
            // Audio mirrors the image guard: a fresh audio-bearing turn against
            // a model with no audio support is rejected with the typed
            // restart prefix before `render_prompt`. Every non-audio family
            // rejects here and image-only / text-only flows stay byte-identical.
            let audio = extract_audio_from_messages(messages);
            if !audio.is_empty() && !admitted_media.audio {
                return Err(Error::from_reason(format!(
                    "{IMAGE_CHANGE_RESTART_PREFIX} this model has no audio support; audio messages are not supported",
                )));
            }
            let tokens = backend.render_prompt(&tokenizer, messages, &config)?;
            (tokens, images, audio, false, 0usize)
        }
        TurnInput::Delta { delta_tokens } => {
            let prior = backend.cached_token_history().len();
            let mut full = backend.cached_token_history().to_vec();
            full.extend_from_slice(delta_tokens);
            (full, Vec::new(), Vec::new(), true, prior)
        }
    };

    let media = MediaInputs {
        images: &images,
        audio: &audio,
    };
    let input_media = media.capabilities();
    // A fresh request fully describes its own context; stale state from a
    // previous session must not constrain the new plan. A delta has no raw
    // media input but may continue over media already encoded in live KV, so
    // ask the backend for the exact media kinds represented there.
    let context_media = if is_delta {
        backend.session_media()
    } else {
        MediaCapabilities::NONE
    };
    let turn_plan = TurnPlan::resolve(
        execution,
        TurnRequest {
            is_delta,
            input_media,
            context_media,
            speculative_requested: p.enable_mtp,
        },
    );

    // --- specialized whole-turn execution ---
    {
        let mut wt_args = WholeTurnArgs {
            tokens: &tokens,
            tokenizer: &tokenizer,
            eos_id,
            config: &config,
            params: &p,
            thinking,
            plan: turn_plan,
            sink: streaming.as_ref().map(|s| s.sink),
            cancelled: streaming.as_ref().map(|s| s.cancelled),
            media,
        };

        // `TurnPlan` keeps current input media, live-context media,
        // paged-attention eligibility, and decoder strategy as independent
        // data. The outer multimodal path depends only on current input; the
        // path is derived only here so a
        // supported combination such as dense Qwen3.5 paged+MTP reaches the
        // paged executor with its speculative decoder choice intact.
        let specialized = match turn_plan.path() {
            TurnPath::Multimodal => Some(backend.run_multimodal_turn(&mut wt_args)),
            TurnPath::Paged => Some(backend.run_paged_turn(&mut wt_args)),
            TurnPath::Speculative => Some(backend.run_speculative_turn(&mut wt_args)),
            TurnPath::Generic => None,
        };
        if let Some(out) = specialized {
            return whole_turn_outcome(out, streaming.is_some());
        }
    }

    // --- generic text-only flow ---
    let generation_start = if report_perf {
        Some(Instant::now())
    } else {
        None
    };
    let mut first_token_instant: Option<Instant> = None;

    // verify_cache_prefix → reset-or-delta split.
    //
    // Fresh turns: `verify_cache_prefix` returns 0 or the full cached
    // length (all-or-nothing contract — see the trait rustdoc). A
    // strict-extend hit (`0 < hit < tokens.len()`) prefills only the
    // tail; everything else — miss AND exact-match — resets and
    // re-prefills the full prompt. Exact-match-as-miss is deliberate:
    // lfm2's short-conv state and qwen3.5's GDN recurrent state have no
    // safe "rewind by one" primitive (lfm2 routes it via its strict
    // `< tokens.len()` check; qwen3.5 via its zero-delta guard).
    //
    // Delta turns: strict extension by construction — the live caches
    // already hold the prior history; prefill exactly the delta tail.
    //
    // A prior eager-MTP turn that stopped mid-cycle leaves the flat trunk
    // advanced past `cached_token_history` (`flat_caches_desynced`). The
    // GDN recurrent layers can't rewind, so neither a fresh prefix-reuse
    // nor a delta-extend onto that trunk is safe — both heal by discarding
    // the trunk and re-prefilling the whole conversation (`tokens` already
    // == saved history ++ any delta).
    let desynced = backend.flat_caches_desynced();
    let (prefill_tokens, cached_prefix_len) = match &input {
        TurnInput::Fresh { .. } => {
            let hit = if desynced {
                0
            } else {
                backend.verify_cache_prefix(&tokens, reuse_cache)
            };
            if hit > 0 && hit < tokens.len() {
                tracing::info!(
                    "Cache reuse: {} cached tokens, {} new tokens to prefill",
                    hit,
                    tokens.len() - hit,
                );
                (tokens[hit..].to_vec(), hit)
            } else {
                backend.reset_caches(ResetScope::PrefixMiss)?;
                (tokens.clone(), 0)
            }
        }
        // `cached_prefix_len` stays 0 on the delta path: the delta tail
        // is the only freshly-prefilled span, so there is no separately
        // reused prefix to report here. The REPORTED reuse is
        // `prior_cached_len` — see `cached_tokens_for_result`.
        TurnInput::Delta { delta_tokens } => {
            if desynced {
                backend.reset_caches(ResetScope::PrefixMiss)?;
                (tokens.clone(), 0usize)
            } else {
                (delta_tokens.clone(), 0usize)
            }
        }
    };
    // A desynced delta turn re-prefilled the whole conversation, reusing
    // nothing — report 0 cached, not the stale `prior_cached_len`.
    let cached_tokens_for_result: u32 = if is_delta && !desynced {
        prior_cached_len as u32
    } else {
        cached_prefix_len as u32
    };

    let prompt_token_count = tokens.len();
    let mut token_history: Vec<u32> = tokens.clone();
    let mut generated_tokens: Vec<u32> =
        Vec::with_capacity(generated_capacity_hint(max_new_tokens));
    let mut finish_reason = String::from("length");

    let generation_stream = Stream::new(DeviceType::Gpu);
    // `None` skips the WiredLimitContext ENTIRELY (qwen3 creates none —
    // see the `wired_limit_bytes` rustdoc); `Some(bytes)` wires the
    // family's byte budget for the turn.
    let _wired_ctx = backend
        .wired_limit_bytes()
        .map(|bytes| crate::stream::WiredLimitContext::new(bytes, vec![generation_stream]));

    let mut profiler = DecodeProfiler::new(
        backend.profiler_label(is_delta, streaming.is_some()),
        backend.family_name(),
    );
    profiler.set_prompt_tokens(prefill_tokens.len() as u32);
    profiler.snapshot_memory_before();

    let mut reasoning_tracker = ReasoningTracker::from_setup(&thinking, think_end_id);

    // Stop set + streaming-order knob, resolved ONCE per turn.
    let extra_eos_ids = backend.extra_eos_ids();
    let eos_before_emit = backend.eos_before_emit();

    // Streaming decode state. The detokenizer's skip-special flag is a
    // family hook (ChatML cores stream `decode_stream(true)`; gemma4
    // streams `false` so its parser sees the channel/tool-call
    // markers). Created unconditionally (cheap) so the borrow structure
    // is identical on both paths; only the streaming branch reads it.
    let stream_skip_special = backend.stream_skip_special_tokens();
    let mut decode_stream = tokenizer.inner().decode_stream(stream_skip_special);
    let mut streamed_text_len = 0usize;
    let mut last_is_reasoning = thinking.enabled;
    // Per-family chunk emitter. Built once per streaming turn, BEFORE
    // `begin_decode` takes the long &mut borrow of the backend.
    let mut emitter: Option<Box<dyn StreamEmitter>> =
        streaming.as_ref().map(|_| backend.stream_emitter());

    // --- prefill ---
    profiler.begin_prefill();
    let last_logits = backend.prefill(&prefill_tokens, generation_stream)?;
    profiler.end_prefill();

    // --- first-token sample ---
    let last_logits = apply_all_penalties(last_logits, &token_history, &p)?;
    let y = crate::sampling::sample(&last_logits, p.sampling_config)?;
    y.eval();

    // --- eval caches post-prefill ---
    backend.eval_caches()?;

    if report_perf {
        first_token_instant = Some(Instant::now());
    }

    // --- decode ---
    // The stepper mutably borrows the backend for the whole loop; scope
    // it so `save_cache_state` below can borrow again.
    {
        let turn_setup = TurnSetup {
            params: &p,
            is_delta,
            // The generic flow is text-only; image turns routed through
            // the multimodal executor above.
            has_images: false,
            total_seq_len: tokens.len(),
        };
        let mut step = backend.begin_decode(&turn_setup)?;
        // Decode-path relabel — see
        // `DecodeStep::profiler_relabel`.
        if let Some(label) = step.profiler_relabel() {
            profiler.set_label(label);
        }
        let streaming_ctx = match (streaming.as_ref(), emitter.as_mut()) {
            (Some(s), Some(em)) => Some(StreamingCtx {
                callback: s.sink,
                cancelled: s.cancelled,
                decode_stream: &mut decode_stream,
                tokenizer: tokenizer.inner(),
                streamed_text_len: &mut streamed_text_len,
                last_is_reasoning: &mut last_is_reasoning,
                emitter: em.as_mut(),
            }),
            _ => None,
        };
        run_decode_loop(
            &mut step,
            DecodeLoopArgs {
                y,
                params: &p,
                reasoning_tracker: &mut reasoning_tracker,
                profiler: &mut profiler,
                max_new_tokens,
                eos_id,
                extra_eos_ids: &extra_eos_ids,
                eos_before_emit,
                generated_tokens: &mut generated_tokens,
                token_history: &mut token_history,
                finish_reason: &mut finish_reason,
                first_token_instant: &mut first_token_instant,
                report_perf,
                generation_stream,
            },
            streaming_ctx,
        )?;
        // Record the final committed token's K/V on a LENGTH exit. The
        // shared decode loop's forward gate (`step_idx + 1 < max_new_tokens
        // && !is_terminal`) skips the last token's forward, so a pure-KV
        // flat stepper (qwen3 / gemma4) ends one token SHORTER than the
        // keep-all-on-length history its `save_cache_state` persists; one
        // extra discard-logits forward closes that gap so the saved cache
        // equals the saved history. LENGTH exits ONLY — an EOS / cancel /
        // repetition final token is a boundary marker the next delta
        // re-renders, and `save_cache_state` drops it. Default no-op for
        // lfm2 (conv state can't re-run a forward — it drops-last instead)
        // and any family whose flat stepper doesn't override it; the MTP
        // cores bypass this flow and the paged flow materializes in
        // `run_paged_turn`.
        if finish_reason == "length"
            && let Some(&last_token) = generated_tokens.last()
        {
            step.materialize_final(last_token)?;
        }
        // Fallible post-loop hook (currently a no-op for every family —
        // see `DecodeStep::end_decode`). Runs while the
        // stepper (and any guards it holds) is still alive, BEFORE
        // `save_cache_state` below. On Err the turn aborts here: the
        // stepper drops (its guards fire) and NO session state is
        // saved.
        step.end_decode()?;
    }

    // --- save cache state ---
    backend.save_cache_state(SaveStateArgs {
        reuse_cache,
        is_delta,
        has_images: false,
        generated_tokens: &generated_tokens,
        finish_reason: &finish_reason,
        save_tokens: &tokens,
        save_expanded_tokens: None,
        image_cache_key: 0,
    });

    // Drop the desync flag only now that `save_cache_state` has committed
    // `cached_token_history` to match the healed trunk. Clearing earlier
    // (right after the heal prefill) would lose the flag if `begin_decode`,
    // `run_decode_loop`, or `end_decode` returned `Err` — leaving the trunk
    // holding the uncommitted healed prompt while history stayed stale, so
    // the next delta would diverge again. Keeping it set across an abort
    // lets the next turn re-heal. The generic flow is AR-only and never
    // re-sets the flag, so this is the turn's single, final clear.
    if desynced {
        backend.clear_flat_caches_desynced();
    }

    // --- finalize ---
    let performance = if report_perf {
        compute_performance_metrics(
            generation_start,
            first_token_instant,
            prefill_tokens.len(),
            generated_tokens.len(),
        )
        .map(|mut m| {
            // Family perf augmentation (default = fill_mtp_acceptance:
            // MTP acceptance fields + profile_phases when profiling is
            // on).
            backend.augment_performance(&profiler, &mut m);
            m
        })
    } else {
        None
    };
    let reasoning_tokens = reasoning_tracker.reasoning_token_count();

    if let (Some(s), Some(em)) = (streaming.as_ref(), emitter.as_mut()) {
        // Flush residual buffered bytes from the incremental decode
        // stream (multi-token grapheme tails the DecodeStream held
        // back) through the emitter. The default emitter suppresses when
        // the residual is reasoning text and include_reasoning is off.
        // The decode here uses the SAME skip-special flag as the in-loop
        // DecodeStream so `streamed_text_len` accounting stays consistent
        // (gemma4 decodes raw).
        let full_text = tokenizer
            .decode_sync(&generated_tokens, stream_skip_special)
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });
        if full_text.len() > streamed_text_len {
            let residual = &full_text[streamed_text_len..];
            em.on_residual(residual, last_is_reasoning, p.include_reasoning, s.sink);
        }
    }

    // Streaming delta turns report the family's `prompt_tokens` choice
    // on the terminal chunk: qwen3_5/MoE report the delta count,
    // lfm2/qwen3 override to the full length. Sync results always carry
    // the full history+delta length — no family diverges there.
    let reported_prompt_tokens: u32 = if is_delta && streaming.is_some() {
        let delta_len = prompt_token_count - prior_cached_len;
        backend.stream_delta_prompt_tokens(prompt_token_count, delta_len)
    } else {
        prompt_token_count as u32
    };

    // Family finalize hook. Default = the ChatML `finalize_chat_result`
    // pipeline; gemma4 overrides with its raw decode + output_parser
    // pipeline.
    let mut result = backend.finalize_turn(FinalizeArgs {
        tokenizer: &tokenizer,
        generated_tokens: &generated_tokens,
        finish_reason,
        think_end_id,
        think_end_str: think_end_str.as_deref(),
        performance,
        include_reasoning: p.include_reasoning,
        thinking_enabled: thinking.enabled,
        prompt_tokens: reported_prompt_tokens,
        reasoning_tokens,
    })?;
    // cached_tokens overwrite stays in the session core (AFTER the
    // finalize hook — overrides must not fill it): fresh turns report
    // the matched prefix length from `verify_cache_prefix`; delta turns
    // report the full prior history length reused by construction.
    result.cached_tokens = cached_tokens_for_result;

    if let (Some(s), Some(em)) = (streaming.as_ref(), emitter.as_mut()) {
        // Terminal done-chunk via the emitter. Family emitters (gemma4)
        // build their own terminal chunk from the finalized result.
        em.finish(&result, s.sink);
        return Ok(None);
    }

    Ok(Some(result))
}
