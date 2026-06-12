//! Generic session-turn cores driving [`ChatBackend`].
//!
//! One private [`chat_turn_core`] reproduces the exact per-family
//! session skeleton (reference implementations:
//! `models/lfm2/model.rs::chat_sync_core` / `chat_stream_sync_core` /
//! `chat_tokens_delta_sync` and the qwen3.5 equivalents in
//! `models/qwen3_5/model.rs`):
//!
//! ```text
//! reuse_cache guard → tokenizer → template/render → extract_chat_params
//!   → vision_turn probe → paged_turn probe → mtp_turn probe
//!   → verify_cache_prefix → reset-or-delta split → prefill
//!   → first-token sample (apply_all_penalties + sampling::sample)
//!   → eval_caches → begin_decode → run_decode_loop
//!   → save_cache_state → finalize_chat_result (+ cached_tokens overwrite)
//! ```
//!
//! Everywhere lfm2 and qwen3.5 genuinely differ, the difference is a
//! [`ChatBackend`] hook (documented on the trait), never a branch on
//! family. The 8 public entry points (4 sync + 4 streaming twins) are
//! thin guard wrappers around the core, mirroring the per-family
//! `chat_session_*_sync` / `chat_stream_session_*_sync` entry points.

// consumed from S7 family migrations; remove in S12
#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{
    ChatBackend, ChunkSink, SaveStateArgs, TurnOutput, TurnSetup, WholeTurnArgs,
};
use crate::engine::cache::IMAGE_CHANGE_RESTART_PREFIX;
use crate::engine::decode::{DecodeLoopArgs, StreamingCtx, run_decode_loop};
use crate::engine::finalize::{compute_performance_metrics, finalize_chat_result};
use crate::engine::params::{
    extract_chat_params, generated_capacity_hint, resolve_enable_thinking,
};
use crate::engine::penalties::{ReasoningTracker, apply_all_penalties};
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
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
/// twins. Mirrors the `(cb, cancelled)` pair the per-family streaming
/// cores take (`chat_stream_sync_core(.., cb: &StreamSender, cancelled:
/// &Arc<AtomicBool>)` — only `.load(Relaxed)` is used, so a plain
/// `&AtomicBool` suffices; `Arc` derefs at the call sites).
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
/// Mirrors `Lfm2Inner::chat_session_start_sync` /
/// `Qwen35Inner::chat_session_start_sync`: rejects an explicit
/// `reuse_cache=false` up front (the session API only makes sense with
/// cache reuse; accepting it would let the post-decode save path wipe
/// the caches the next continue call depends on), then delegates to the
/// core. NOTE: no unconditional reset here — prefix-reuse support
/// requires the core to decide reset-vs-reuse from
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
/// `images` is the opt-in guard parameter shared by every family:
/// non-empty input is rejected with the
/// `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` prefix so the TS
/// `ChatSession` layer can route image changes back through a fresh
/// session start.
pub(crate) fn session_continue<B: ChatBackend>(
    backend: &mut B,
    user_message: String,
    images: Option<Vec<Uint8Array>>,
    config: ChatConfig,
) -> Result<ChatResult> {
    if images.as_ref().is_some_and(|v| !v.is_empty()) {
        return Err(Error::from_reason(format!(
            "{IMAGE_CHANGE_RESTART_PREFIX} chat_session_continue is text-only; start a new session with chat_session_start to change the image",
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
    delta_guards(backend, &delta_tokens, &config)?;
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

/// Streaming twin of [`session_start`]. Mirrors the per-family
/// `chat_stream_session_*_sync` contract: guard failures and errors are
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
    if let Err(e) = delta_guards(backend, &delta_tokens, &config) {
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

/// The four delta-path guards shared by `chat_tokens_delta_sync` /
/// `chat_stream_tokens_delta_sync` on every family. Order preserved
/// from the reference implementations.
fn delta_guards<B: ChatBackend>(
    backend: &B,
    delta_tokens: &[u32],
    config: &ChatConfig,
) -> Result<()> {
    if config.reuse_cache == Some(false) {
        return Err(Error::from_reason(
            "chat_tokens_delta_sync requires reuse_cache to be enabled; \
             the delta path operates on session state by construction",
        ));
    }
    if !backend.has_live_session() {
        return Err(Error::from_reason(
            "chat_tokens_delta_sync requires an initialized session (call chatSessionStart first)",
        ));
    }
    if delta_tokens.is_empty() {
        return Err(Error::from_reason(
            "chat_tokens_delta_sync requires a non-empty delta",
        ));
    }
    // Text-only families reject deltas while the session holds image
    // state (lfm2's defensive guard). Image-capable families accept
    // text deltas on image sessions by design (qwen3.5's sticky
    // image-key contract — see `save_cache_state_after_delta`).
    if !backend.supports_images() && backend.session_holds_images() {
        return Err(Error::from_reason(
            "chat_tokens_delta_sync is text-only; session currently holds image state",
        ));
    }
    Ok(())
}

/// Unwrap the core's sync-path result. `Ok(None)` means a whole-turn
/// override returned [`TurnOutput::Streamed`] with no sink attached —
/// a family-impl contract violation, surfaced as an error rather than
/// a panic.
fn expect_sync_result(out: Result<Option<ChatResult>>) -> Result<ChatResult> {
    out?.ok_or_else(|| {
        Error::from_reason(
            "whole-turn override returned TurnOutput::Streamed on the sync (sink-less) path",
        )
    })
}

/// Map a whole-turn override's outcome into the core's return shape.
fn whole_turn_outcome(out: Result<TurnOutput>) -> Result<Option<ChatResult>> {
    match out? {
        TurnOutput::Complete(result) => Ok(Some(*result)),
        TurnOutput::Streamed => Ok(None),
    }
}

/// Collect every image payload from the turn's messages, in order. ==
/// `models/qwen3_5/model.rs::extract_images_from_messages` (S7+ deletes
/// the per-family copies in favor of this one).
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

// =====================================================================
// The turn core
// =====================================================================

/// One chat turn, generic over the backend.
///
/// Returns `Ok(Some(result))` for sync callers; `Ok(None)` when the
/// turn's output was fully delivered through the streaming sink (the
/// generic streaming flow emits the terminal chunk itself and still
/// returns `Ok(None)`; whole-turn overrides signal the same via
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

    let p = extract_chat_params(&config);
    let reuse_cache = p.reuse_cache;
    let report_perf = p.report_performance;
    let max_new_tokens = p.max_new_tokens;
    let thinking = backend.thinking_setup(&config);

    // --- template/render: full prompt tokens for this turn ---
    // Fresh: jinja-render the messages. Delta: cached history + delta
    // (the delta paths skip the template entirely — the caller owns
    // cache coherence by construction).
    let (tokens, images, is_delta, prior_cached_len) = match &input {
        TurnInput::Fresh { messages } => {
            let tool_defs = config.tools.as_deref();
            let enable_thinking = resolve_enable_thinking(&config);
            let tokens = tokenizer.apply_chat_template_sync(
                messages,
                Some(true),
                tool_defs,
                enable_thinking,
            )?;
            let images = extract_images_from_messages(messages);
            (tokens, images, false, 0usize)
        }
        TurnInput::Delta { delta_tokens } => {
            let prior = backend.cached_token_history().len();
            let mut full = backend.cached_token_history().to_vec();
            full.extend_from_slice(delta_tokens);
            (full, Vec::new(), true, prior)
        }
    };

    // --- whole-turn overrides: vision → paged → mtp ---
    {
        let mut wt_args = WholeTurnArgs {
            tokens: &tokens,
            tokenizer: &tokenizer,
            eos_id,
            config: &config,
            params: &p,
            is_delta,
            sink: streaming.as_ref().map(|s| s.sink),
            cancelled: streaming.as_ref().map(|s| s.cancelled),
            images: &images,
        };

        // Vision probe. Text-only families reject image-bearing turns
        // with the typed restart prefix the TS `ChatSession` layer
        // matches on; image-capable families MUST take the override
        // (the generic flow below is text-only, so silently falling
        // through would drop the images).
        if !images.is_empty() {
            if !backend.supports_images() {
                return Err(Error::from_reason(format!(
                    "{IMAGE_CHANGE_RESTART_PREFIX} this model is text-only; image messages are not supported",
                )));
            }
            return match backend.vision_turn(&mut wt_args) {
                Some(out) => whole_turn_outcome(out),
                None => Err(Error::from_reason(
                    "model reports supports_images() but provided no vision_turn override",
                )),
            };
        }

        // Paged probe. == the `self.paged_adapter.is_some()` dispatch
        // right after tokenization in every family core. A family whose
        // MTP takes precedence over paged (qwen3.5's
        // `mtp_takes_dense_path`) returns `None` here for those turns
        // so the `mtp_turn` probe below picks them up.
        if backend.has_paged_adapter()
            && let Some(out) = backend.paged_turn(&mut wt_args)
        {
            return whole_turn_outcome(out);
        }

        // MTP probe. == the `p.enable_mtp && has_mtp_weights` branch in
        // the qwen3.5 dense/MoE cores.
        if let Some(out) = backend.mtp_turn(&mut wt_args) {
            return whole_turn_outcome(out);
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
    // re-prefills the full prompt. Exact-match-as-miss is deliberate on
    // BOTH reference families: lfm2's short-conv state and qwen3.5's
    // GDN recurrent state have no safe "rewind by one" primitive (lfm2
    // routes it via its strict `< tokens.len()` check; qwen3.5 via its
    // zero-delta guard).
    //
    // Delta turns: strict extension by construction — the live caches
    // already hold the prior history; prefill exactly the delta tail.
    let (prefill_tokens, cached_prefix_len) = match &input {
        TurnInput::Fresh { .. } => {
            let hit = backend.verify_cache_prefix(&tokens, reuse_cache);
            if hit > 0 && hit < tokens.len() {
                tracing::info!(
                    "Cache reuse: {} cached tokens, {} new tokens to prefill",
                    hit,
                    tokens.len() - hit,
                );
                (tokens[hit..].to_vec(), hit)
            } else {
                backend.reset_caches()?;
                (tokens.clone(), 0)
            }
        }
        // `cached_prefix_len` stays 0 on the delta path (it feeds the
        // VLM rope-delta decisions keyed on fresh-prefill reuse;
        // `is_delta` drives the delta-side decisions). The REPORTED
        // reuse is `prior_cached_len` — see `cached_tokens_for_result`.
        TurnInput::Delta { delta_tokens } => (delta_tokens.clone(), 0usize),
    };
    let cached_tokens_for_result: u32 = if is_delta {
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
    let _wired_ctx =
        crate::stream::WiredLimitContext::new(backend.wired_limit_bytes(), vec![generation_stream]);

    let mut profiler = DecodeProfiler::new(
        if is_delta { "chat_delta" } else { "chat" },
        backend.family_name(),
    );
    profiler.set_prompt_tokens(prefill_tokens.len() as u32);
    profiler.snapshot_memory_before();

    let mut reasoning_tracker =
        ReasoningTracker::new(thinking.enabled, thinking.budget, think_end_id);

    // Streaming decode state. Created unconditionally (cheap) so the
    // borrow structure is identical on both paths; only the streaming
    // branch reads it.
    let mut decode_stream = tokenizer.inner().decode_stream(true);
    let mut streamed_text_len = 0usize;
    let mut last_is_reasoning = thinking.enabled;

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
            max_new_tokens,
            is_delta,
            // The generic flow is text-only; image turns routed through
            // `vision_turn` above.
            has_images: false,
            cached_prefix_len,
            total_seq_len: tokens.len(),
        };
        let mut step = backend.begin_decode(&turn_setup)?;
        let streaming_ctx = streaming.as_ref().map(|s| StreamingCtx {
            callback: s.sink,
            cancelled: s.cancelled,
            decode_stream: &mut decode_stream,
            tokenizer: tokenizer.inner(),
            streamed_text_len: &mut streamed_text_len,
            last_is_reasoning: &mut last_is_reasoning,
        });
        run_decode_loop(
            &mut step,
            DecodeLoopArgs {
                y,
                params: &p,
                reasoning_tracker: &mut reasoning_tracker,
                profiler: &mut profiler,
                max_new_tokens,
                eos_id,
                generated_tokens: &mut generated_tokens,
                token_history: &mut token_history,
                finish_reason: &mut finish_reason,
                first_token_instant: &mut first_token_instant,
                report_perf,
                generation_stream,
            },
            streaming_ctx,
        )?;
    }

    // lfm2's `last_token_in_cache` derivation: the loop builds the
    // pipelined forward (which writes the just-sampled token into the
    // caches) iff `step + 1 < max_new_tokens`, so at EVERY exit the
    // final pushed token is in the caches exactly when the loop stopped
    // short of the budget. Families without the trim semantics ignore
    // the field (see `SaveStateArgs::last_token_in_cache`).
    let last_token_in_cache = (generated_tokens.len() as i64) < max_new_tokens as i64;

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
        last_token_in_cache,
    });

    // --- finalize ---
    let performance = if report_perf {
        compute_performance_metrics(
            generation_start,
            first_token_instant,
            prefill_tokens.len(),
            generated_tokens.len(),
        )
    } else {
        None
    };
    let reasoning_tokens = reasoning_tracker.reasoning_token_count();

    if let Some(s) = streaming.as_ref() {
        // Flush residual buffered bytes from the incremental decode
        // stream (multi-token grapheme tails the DecodeStream held
        // back). Mirrors the per-family streaming cores: suppress when
        // the residual is reasoning text and include_reasoning is off.
        let full_text = tokenizer
            .decode_sync(&generated_tokens, true)
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });
        if full_text.len() > streamed_text_len {
            let residual = full_text[streamed_text_len..].to_string();
            if p.include_reasoning || !last_is_reasoning {
                s.sink.send(Ok(ChatStreamChunk {
                    text: residual,
                    done: false,
                    finish_reason: None,
                    tool_calls: None,
                    thinking: None,
                    num_tokens: None,
                    prompt_tokens: None,
                    reasoning_tokens: None,
                    raw_text: None,
                    cached_tokens: None,
                    performance: None,
                    is_reasoning: Some(last_is_reasoning),
                }));
            }
        }
    }

    let mut result = finalize_chat_result(
        &tokenizer,
        &generated_tokens,
        finish_reason,
        think_end_id,
        think_end_str.as_deref(),
        performance,
        p.include_reasoning,
        thinking.enabled,
        prompt_token_count as u32,
        reasoning_tokens,
    )?;
    // cached_tokens overwrite: fresh turns report the matched prefix
    // length from `verify_cache_prefix`; delta turns report the full
    // prior history length reused by construction.
    result.cached_tokens = cached_tokens_for_result;

    if let Some(s) = streaming.as_ref() {
        // Terminal done-chunk, built from the finalized result —
        // byte-identical field mapping to the per-family streaming
        // cores' final `cb.call`.
        s.sink.send(Ok(ChatStreamChunk {
            text: result.text.clone(),
            done: true,
            finish_reason: Some(result.finish_reason.clone()),
            tool_calls: Some(result.tool_calls.clone()),
            thinking: result.thinking.clone(),
            num_tokens: Some(result.num_tokens),
            prompt_tokens: Some(result.prompt_tokens),
            reasoning_tokens: Some(result.reasoning_tokens),
            raw_text: Some(result.raw_text.clone()),
            cached_tokens: Some(result.cached_tokens),
            performance: result.performance.clone(),
            is_reasoning: None,
        }));
        return Ok(None);
    }

    Ok(Some(result))
}
