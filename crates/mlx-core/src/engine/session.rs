//! Generic session-turn cores driving [`ChatBackend`].
//!
//! One private [`chat_turn_core`] runs the session skeleton for every
//! family:
//!
//! ```text
//! reuse_cache guard → tokenizer → resolve_params
//!   → pre-render image guard → render_prompt
//!   → resolve TurnPlan → optional multimodal/paged/speculative executor
//!   → verify_cache_prefix → reset-or-reuse split → prefill
//!   → first-token sample (apply_all_penalties + sampling::sample)
//!   → eval_caches → begin_decode → run_decode_loop → end_decode
//!   → save_cache_state → finalize_turn (+ cached_tokens overwrite)
//! ```
//!
//! Everywhere families genuinely differ, the difference is a
//! [`ChatBackend`] hook (documented on the trait), never a branch on
//! family. The 6 public entry points (3 sync + 3 streaming twins) are
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
    expect_sync_result(chat_turn_core(backend, messages, config, None))
}

/// Generic role-aware session continuation.
///
/// The caller supplies the complete structured conversation, including
/// the pending user message. The model-provided chat template renders that
/// full history, then the normal fresh-turn prefix verifier either reuses
/// the exact cached token prefix and prefills only the suffix, or resets and
/// safely replays the full prompt. No Rust-side wire-format delta is
/// synthesized.
pub(crate) fn session_continue<B: ChatBackend>(
    backend: &mut B,
    messages: Vec<ChatMessage>,
    config: ChatConfig,
) -> Result<ChatResult> {
    if config.reuse_cache == Some(false) {
        return Err(Error::from_reason(
            "chat_session_continue requires reuse_cache=true (leave as None or set to true). \
             The session API only makes sense with cache reuse enabled.",
        ));
    }
    if !backend.has_live_session() {
        return Err(Error::from_reason(
            "chat_session_continue requires an initialized session (call chatSessionStart first)",
        ));
    }
    expect_sync_result(chat_turn_core(backend, messages, config, None))
}

/// Generic role-aware tool-result continuation.
///
/// Like [`session_continue`], the caller supplies the complete history,
/// now ending in the pending tool-role message. The loaded chat template is
/// the sole authority for tool-call/result layout.
pub(crate) fn session_continue_tool<B: ChatBackend>(
    backend: &mut B,
    messages: Vec<ChatMessage>,
    config: ChatConfig,
) -> Result<ChatResult> {
    if config.reuse_cache == Some(false) {
        return Err(Error::from_reason(
            "chat_session_continue_tool requires reuse_cache=true (leave as None or set to true). \
             The session API only makes sense with cache reuse enabled.",
        ));
    }
    if !backend.has_live_session() {
        return Err(Error::from_reason(
            "chat_session_continue_tool requires an initialized session (call chatSessionStart first)",
        ));
    }
    expect_sync_result(chat_turn_core(backend, messages, config, None))
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
        messages,
        config,
        Some(StreamingHooks { sink, cancelled }),
    ) {
        sink.send(Err(e));
    }
}

/// Streaming twin of [`session_continue`].
pub(crate) fn session_continue_stream<B: ChatBackend>(
    backend: &mut B,
    messages: Vec<ChatMessage>,
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
    if config.reuse_cache == Some(false) {
        sink.send(Err(Error::from_reason(
            "chat_stream_session_continue requires reuse_cache=true (leave as None or set to true). \
             The session API only makes sense with cache reuse enabled.",
        )));
        return;
    }
    if !backend.has_live_session() {
        sink.send(Err(Error::from_reason(
            "chat_stream_session_continue requires an initialized session (call chatStreamSessionStart first)",
        )));
        return;
    }
    if let Err(e) = chat_turn_core(
        backend,
        messages,
        config,
        Some(StreamingHooks { sink, cancelled }),
    ) {
        sink.send(Err(e));
    }
}

/// Streaming twin of [`session_continue_tool`].
pub(crate) fn session_continue_tool_stream<B: ChatBackend>(
    backend: &mut B,
    messages: Vec<ChatMessage>,
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
    if config.reuse_cache == Some(false) {
        sink.send(Err(Error::from_reason(
            "chat_stream_session_continue_tool requires reuse_cache=true (leave as None or set to true). \
             The session API only makes sense with cache reuse enabled.",
        )));
        return;
    }
    if !backend.has_live_session() {
        sink.send(Err(Error::from_reason(
            "chat_stream_session_continue_tool requires an initialized session (call chatStreamSessionStart first)",
        )));
        return;
    }
    if let Err(e) = chat_turn_core(
        backend,
        messages,
        config,
        Some(StreamingHooks { sink, cancelled }),
    ) {
        sink.send(Err(e));
    }
}

// =====================================================================
// Shared guards / helpers
// =====================================================================

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

/// Invalidate every live-session signal after the generic flat executor has
/// started mutating physical cache state and then fails.
///
/// `PrefixMiss` is intentionally insufficient here: Qwen3.5 dense/MoE preserve
/// their committed token history across that turn-internal reset so a
/// successful full re-prefill can overwrite it at commit time. On an aborted
/// re-prefill, retaining that history would let the next full-history request
/// prefix-hit empty or partially advanced physical caches. The explicit
/// command reset is the shared fail-closed contract.
fn fail_closed_flat_turn<B: ChatBackend, T>(backend: &mut B, turn_error: Error) -> Result<T> {
    if let Err(reset_error) = backend.reset_caches(ResetScope::Command) {
        tracing::error!(
            "generic flat turn failed ({}) and session invalidation also failed ({})",
            turn_error.reason,
            reset_error.reason,
        );
    }
    Err(turn_error)
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
    messages: Vec<ChatMessage>,
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
    // Replay provenance is the effective boolean handed to the model's
    // Jinja template, not the family's decode-time ThinkingSetup. Those
    // differ for Gemma4 (`ThinkingPolicy::None`) and LFM2
    // (`AlwaysOnBudgetFromEffort`).
    let template_thinking_enabled =
        crate::engine::params::resolve_enable_thinking(&config).unwrap_or(true);
    backend.set_cache_owner_id(&p.cache_owner_id, p.cache_root_owner_id.as_deref());
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
    // Every public session entry receives the complete structured
    // transcript and renders it through the checkpoint-provided Jinja
    // template. Rust never synthesizes a model-specific wire-format delta.
    //
    // Pre-render image guard — TS `ChatSession` restart-routing contract:
    // a text-only backend MUST reject an image-bearing turn with the typed
    // `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`-prefixed error, and that
    // rejection MUST happen BEFORE `render_prompt`.
    // `serialize_message_for_jinja` represents image user content as an
    // array, so a text-only family's chat template could otherwise fail
    // with an UNTYPED template error first, breaking the prefix routing.
    // Vision-capable backends with image capability — including gemma4's
    // unconditional policy, whose "no vision support" error surfaces from
    // inside the multimodal executor — skip the rejection, render normally,
    // and route through the multimodal executor below with these exact
    // extracted images (single extraction — no drift).
    let images = extract_images_from_messages(&messages);
    if !images.is_empty() && !admitted_media.images {
        return Err(Error::from_reason(format!(
            "{IMAGE_CHANGE_RESTART_PREFIX} this model is text-only; image messages are not supported",
        )));
    }
    // Audio mirrors the image guard: an audio-bearing turn against a model
    // with no audio support is rejected with the typed restart prefix before
    // `render_prompt`. Every non-audio family rejects here and image-only /
    // text-only flows stay byte-identical.
    let audio = extract_audio_from_messages(&messages);
    if !audio.is_empty() && !admitted_media.audio {
        return Err(Error::from_reason(format!(
            "{IMAGE_CHANGE_RESTART_PREFIX} this model has no audio support; audio messages are not supported",
        )));
    }
    let tokens = backend.render_prompt(&tokenizer, &messages, &config)?;

    let media = MediaInputs {
        images: &images,
        audio: &audio,
    };
    let input_media = media.capabilities();
    // A full-history request describes its own context; stale state from a
    // previous session must not constrain the new plan.
    let context_media = MediaCapabilities::NONE;
    let turn_plan = TurnPlan::resolve(
        execution,
        TurnRequest {
            is_delta: false,
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

    // verify_cache_prefix → reset-or-reuse split.
    //
    // `verify_cache_prefix` returns 0 or the full cached length
    // (all-or-nothing contract — see the trait rustdoc). A strict-extend
    // hit (`0 < hit < tokens.len()`) prefills only the tail; everything
    // else — miss AND exact-match — resets and re-prefills the full prompt.
    // Exact-match-as-miss is deliberate: lfm2's short-conv state and
    // qwen3.5's GDN recurrent state have no safe "rewind by one" primitive
    // (lfm2 routes it via its strict `< tokens.len()` check; qwen3.5 via its
    // zero-delta guard).
    //
    // A prior eager-MTP turn that stopped mid-cycle leaves the flat trunk
    // advanced past `cached_token_history` (`flat_caches_desynced`). The
    // GDN recurrent layers can't rewind, so prefix reuse onto that trunk is
    // unsafe; heal by discarding it and re-prefilling the rendered history.
    let desynced = backend.flat_caches_desynced();
    let hit = if desynced {
        0
    } else {
        backend.verify_cache_prefix(&tokens, reuse_cache)
    };
    let (prefill_tokens, cached_prefix_len) = if hit > 0 && hit < tokens.len() {
        tracing::info!(
            "Cache reuse: {} cached tokens, {} new tokens to prefill",
            hit,
            tokens.len() - hit,
        );
        (tokens[hit..].to_vec(), hit)
    } else {
        if let Err(error) = backend.reset_caches(ResetScope::PrefixMiss) {
            return fail_closed_flat_turn(backend, error);
        }
        (tokens.clone(), 0)
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
        backend.profiler_label(false, streaming.is_some()),
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

    // From prefill onward, every fallible operation runs inside one error
    // boundary. The temporary closure ensures a decode stepper borrowing the
    // backend is dropped before fail-closed invalidation borrows it again.
    let flat_turn_result: Result<()> = (|| {
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
                is_delta: false,
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
            // Fallible post-loop hook. Runs while the stepper (and any guards
            // it holds) is still alive, before `save_cache_state` below.
            step.end_decode()?;
        }
        Ok(())
    })();
    if let Err(error) = flat_turn_result {
        return fail_closed_flat_turn(backend, error);
    }

    // --- save cache state ---
    backend.save_cache_state(SaveStateArgs {
        reuse_cache,
        is_delta: false,
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

    // Family finalize hook. Default = the ChatML `finalize_chat_result`
    // pipeline; gemma4 overrides with its raw decode + output_parser
    // pipeline.
    let finalized = backend.finalize_turn(FinalizeArgs {
        tokenizer: &tokenizer,
        generated_tokens: &generated_tokens,
        finish_reason,
        think_end_id,
        think_end_str: think_end_str.as_deref(),
        performance,
        include_reasoning: p.include_reasoning,
        thinking_enabled: thinking.enabled,
        prompt_tokens: prompt_token_count as u32,
        reasoning_tokens,
    });
    let mut result = match finalized {
        Ok(result) => result,
        Err(error) => return fail_closed_flat_turn(backend, error),
    };
    result.thinking_enabled = template_thinking_enabled;
    // cached_tokens overwrite stays in the session core (AFTER the
    // finalize hook — overrides must not fill it): report the matched
    // prefix length from `verify_cache_prefix`.
    result.cached_tokens = cached_prefix_len as u32;

    if let (Some(s), Some(em)) = (streaming.as_ref(), emitter.as_mut()) {
        // Terminal done-chunk via the emitter. Family emitters (gemma4)
        // build their own terminal chunk from the finalized result.
        em.finish(&result, s.sink);
        return Ok(None);
    }

    Ok(Some(result))
}
