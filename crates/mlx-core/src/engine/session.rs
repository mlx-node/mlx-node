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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TurnKind {
    Start,
    Continue,
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
        messages,
        config,
        TurnKind::Start,
        None,
    ))
}

/// Generic role-aware session continuation.
///
/// The caller supplies the complete structured conversation, including
/// the pending user message. The model-provided chat template renders that
/// full history. When the completed structured history exactly describes the
/// live cache, the core preserves the committed token IDs and appends only
/// the template-authored remainder; otherwise it resets and safely replays the
/// full prompt.
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
    expect_sync_result(chat_turn_core(
        backend,
        messages,
        config,
        TurnKind::Continue,
        None,
    ))
}

/// Generic role-aware tool-result continuation.
///
/// Like [`session_continue`], the caller supplies the complete history,
/// now ending in the pending tool-role message. The loaded chat template is
/// the sole authority for the appended tool-call/result layout.
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
    expect_sync_result(chat_turn_core(
        backend,
        messages,
        config,
        TurnKind::Continue,
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
        messages,
        config,
        TurnKind::Start,
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
        TurnKind::Continue,
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
        TurnKind::Continue,
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

fn media_capabilities_from_messages(messages: &[ChatMessage]) -> MediaCapabilities {
    MediaCapabilities {
        images: messages.iter().any(|message| {
            message
                .images
                .as_ref()
                .is_some_and(|images| !images.is_empty())
        }),
        audio: messages.iter().any(|message| {
            message
                .audio
                .as_ref()
                .is_some_and(|clips| !clips.is_empty())
        }),
    }
}

/// Verify that the caller-supplied completed transcript describes the exact
/// media payloads encoded by the live cache.
///
/// Matching only modality presence is insufficient because chat templates
/// render different image/audio byte payloads to the same placeholder tokens.
/// Text-only histories need no payload proof; media histories fail closed
/// unless the backend can match its stored cache key or digest.
pub(super) fn live_history_media_matches<B: ChatBackend>(
    backend: &B,
    completed_history: &[ChatMessage],
) -> bool {
    let history_media = media_capabilities_from_messages(completed_history);
    if backend.session_media() != history_media {
        return false;
    }
    if history_media.is_empty() {
        return true;
    }

    let images = extract_images_from_messages(completed_history);
    let audio = extract_audio_from_messages(completed_history);
    backend.session_media_matches_payloads(&images, &audio)
}

const THINK_END: &str = "</think>";

#[derive(Debug, PartialEq, Eq)]
struct StructuredReasoningBoundary {
    ordinal: usize,
    reasoning: String,
    content: String,
}

fn copy_message_with_reasoning(
    message: &ChatMessage,
    reasoning_content: Option<String>,
) -> ChatMessage {
    ChatMessage {
        role: message.role.clone(),
        content: message.content.clone(),
        tool_calls: message.tool_calls.clone(),
        tool_call_id: message.tool_call_id.clone(),
        is_error: message.is_error,
        reasoning_content,
        thinking_enabled: message.thinking_enabled,
        images: message.images.as_ref().map(|images| {
            images
                .iter()
                .map(|image| Uint8Array::with_data_copied(image.as_ref()))
                .collect()
        }),
        audio: message.audio.as_ref().map(|clips| {
            clips
                .iter()
                .map(|clip| Uint8Array::with_data_copied(clip.as_ref()))
                .collect()
        }),
    }
}

/// Locate close tags that were emitted from structured assistant reasoning.
///
/// The shadow render replaces each non-empty `reasoning_content` field with a
/// unique sentinel. If the original and shadow renders differ only at those
/// exact sentinel spans, the following `</think>` tags have role/template
/// provenance. A template that transforms the reasoning field in any other way
/// fails closed instead of authorizing a looser history comparison.
fn structured_reasoning_boundary_ordinals(
    tokenizer: &crate::tokenizer::Qwen3Tokenizer,
    completed_history: &[ChatMessage],
    config: &ChatConfig,
) -> Result<Option<Vec<StructuredReasoningBoundary>>> {
    let completed_template = tokenizer.render_chat_template_sync(
        completed_history,
        Some(false),
        config.tools.as_deref(),
        crate::engine::params::resolve_enable_thinking(config),
    )?;
    let salt = (0usize..)
        .find(|salt| !completed_template.contains(&format!("__MLX_REASONING_PROVENANCE_{salt}_")))
        .expect("an unbounded numeric salt must produce a unique sentinel");

    let mut replacements = Vec::new();
    let mut shadow_history = Vec::with_capacity(completed_history.len());
    for (message_index, message) in completed_history.iter().enumerate() {
        let replacement = message
            .reasoning_content
            .as_ref()
            .filter(|reasoning| {
                !reasoning.is_empty() && message.role.trim().eq_ignore_ascii_case("assistant")
            })
            .map(|reasoning| {
                let sentinel = format!("__MLX_REASONING_PROVENANCE_{salt}_{message_index}__");
                replacements.push((sentinel.clone(), reasoning.clone(), message.content.clone()));
                sentinel
            });
        shadow_history.push(copy_message_with_reasoning(
            message,
            replacement.or_else(|| message.reasoning_content.clone()),
        ));
    }
    if replacements.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let shadow_template = tokenizer.render_chat_template_sync(
        &shadow_history,
        Some(false),
        config.tools.as_deref(),
        crate::engine::params::resolve_enable_thinking(config),
    )?;
    let mut rendered_replacements = Vec::new();
    for replacement in replacements {
        match shadow_template.matches(&replacement.0).count() {
            0 => {}
            1 => rendered_replacements.push(replacement),
            _ => return Ok(None),
        }
    }

    let Some(boundary_positions) = locate_structured_reasoning_boundaries(
        &completed_template,
        &shadow_template,
        &rendered_replacements,
    ) else {
        return Ok(None);
    };
    let all_tag_positions = completed_template
        .match_indices(THINK_END)
        .map(|(position, _)| position)
        .collect::<Vec<_>>();
    let mut boundaries = Vec::with_capacity(boundary_positions.len());
    for (position, (_, reasoning, content)) in
        boundary_positions.into_iter().zip(rendered_replacements)
    {
        let Ok(ordinal) = all_tag_positions.binary_search(&position) else {
            return Ok(None);
        };
        boundaries.push(StructuredReasoningBoundary {
            ordinal,
            reasoning,
            content,
        });
    }
    Ok(Some(boundaries))
}

fn locate_structured_reasoning_boundaries(
    original: &str,
    shadow: &str,
    replacements: &[(String, String, String)],
) -> Option<Vec<usize>> {
    let mut original_cursor = 0usize;
    let mut shadow_cursor = 0usize;
    let mut boundaries = Vec::with_capacity(replacements.len());
    for (sentinel, reasoning, _) in replacements {
        let sentinel_offset = shadow[shadow_cursor..].find(sentinel)?;
        let sentinel_start = shadow_cursor + sentinel_offset;
        let common_prefix = &shadow[shadow_cursor..sentinel_start];
        if !original[original_cursor..].starts_with(common_prefix) {
            return None;
        }
        original_cursor += common_prefix.len();
        shadow_cursor = sentinel_start + sentinel.len();

        if !original[original_cursor..].starts_with(reasoning) {
            return None;
        }
        original_cursor += reasoning.len();

        let close_offset = shadow[shadow_cursor..].find(THINK_END)?;
        let before_close = &shadow[shadow_cursor..shadow_cursor + close_offset];
        if !original[original_cursor..].starts_with(before_close) {
            return None;
        }
        original_cursor += before_close.len();
        shadow_cursor += close_offset;
        if !original[original_cursor..].starts_with(THINK_END)
            || !shadow[shadow_cursor..].starts_with(THINK_END)
        {
            return None;
        }
        boundaries.push(original_cursor);
        original_cursor += THINK_END.len();
        shadow_cursor += THINK_END.len();
    }
    (original[original_cursor..] == shadow[shadow_cursor..]).then_some(boundaries)
}

/// Verify caller-owned reasoning/content bytes against the committed cache
/// before any template-separator normalization.
///
/// A cached prefix may end before a later reasoning close (length exit), in
/// which case the ordinary prefix comparison remains authoritative. Once a
/// proven close tag is present, however, the structured message fields must
/// match exactly. Leading assistant-content whitespace is rejected rather than
/// ambiguously treating it as separator whitespace.
fn cached_structured_reasoning_matches(
    cached_text: &str,
    boundaries: &[StructuredReasoningBoundary],
) -> bool {
    let tag_positions = cached_text
        .match_indices(THINK_END)
        .map(|(position, _)| position)
        .collect::<Vec<_>>();
    for boundary in boundaries {
        let Some(&tag_start) = tag_positions.get(boundary.ordinal) else {
            continue;
        };
        if boundary
            .reasoning
            .ends_with(|character: char| character.is_ascii_whitespace())
        {
            return false;
        }
        let before_tag = cached_text[..tag_start]
            .trim_end_matches(|character: char| character.is_ascii_whitespace());
        if !before_tag.ends_with(&boundary.reasoning) {
            return false;
        }
        if !boundary.content.is_empty() {
            let after_tag = &cached_text[tag_start + THINK_END.len()..];
            let after_separator =
                after_tag.trim_start_matches(|character: char| character.is_ascii_whitespace());
            if boundary
                .content
                .starts_with(|character: char| character.is_ascii_whitespace())
            {
                return false;
            }
            let content_matches = if after_separator.len() >= boundary.content.len() {
                after_separator.starts_with(&boundary.content)
            } else {
                boundary.content.starts_with(after_separator)
            };
            if !content_matches {
                return false;
            }
        }
    }
    true
}

/// Normalize whitespace only around reasoning-close tags whose structured
/// assistant/template provenance was established above, retaining a map from
/// normalized byte boundaries back to the original text.
///
/// Checkpoint templates commonly render `\n</think>\n\n`, while the committed
/// generated tokens can carry `</think>\n`. Those byte layouts describe the
/// same reasoning boundary but are not prefix-equal. Literal tags in user or
/// ordinary assistant content are never selected.
fn normalize_reasoning_boundaries(
    text: &str,
    reasoning_boundary_ordinals: &[usize],
    require_all_boundaries: bool,
) -> Option<(String, Vec<usize>)> {
    let bytes = text.as_bytes();
    let mut omitted = vec![false; bytes.len()];
    let selected = reasoning_boundary_ordinals
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let mut found = 0usize;
    for (ordinal, (tag_start, _)) in text.match_indices(THINK_END).enumerate() {
        if !selected.contains(&ordinal) {
            continue;
        }
        found += 1;
        let mut before = tag_start;
        while before > 0 && bytes[before - 1].is_ascii_whitespace() {
            before -= 1;
            omitted[before] = true;
        }

        let mut after = tag_start + THINK_END.len();
        while after < bytes.len() && bytes[after].is_ascii_whitespace() {
            omitted[after] = true;
            after += 1;
        }
    }
    if require_all_boundaries && found != selected.len() {
        return None;
    }

    let mut normalized = Vec::with_capacity(bytes.len());
    let mut source_boundaries = Vec::with_capacity(bytes.len() + 1);
    source_boundaries.push(0);
    for (index, &byte) in bytes.iter().enumerate() {
        if omitted[index] {
            *source_boundaries
                .last_mut()
                .expect("the initial boundary is always present") = index + 1;
        } else {
            normalized.push(byte);
            source_boundaries.push(index + 1);
        }
    }

    Some((
        String::from_utf8(normalized).expect("removing ASCII whitespace preserves valid UTF-8"),
        source_boundaries,
    ))
}

/// Reconstruct a live continuation from the exact committed token IDs plus
/// the suffix authored by the checkpoint template.
///
/// Generated text is not a lossless token-history format: decoding and then
/// re-encoding the same bytes may choose a different BPE segmentation, and a
/// length-truncated reasoning turn has no closing tag for a template to
/// reproduce. A full structured re-render can therefore describe the same
/// conversation while failing the exact token-prefix check required by live
/// KV reuse.
///
/// To keep the template authoritative, render both the completed history and
/// the full history including the pending user/tool message. Compare the live
/// history in the template's logical placeholder form and normalize only
/// template-owned reasoning-boundary whitespace, then append the mapped
/// template remainder to the original cached token IDs. Any edited or
/// incompatible history fails these checks and returns `None`, making the
/// caller safely cold-replay the complete render.
fn render_live_continuation<B: ChatBackend>(
    backend: &B,
    tokenizer: &crate::tokenizer::Qwen3Tokenizer,
    messages: &[ChatMessage],
    config: &ChatConfig,
    full_tokens: &[u32],
) -> Result<Option<Vec<u32>>> {
    let Some((_pending, completed_history)) = messages.split_last() else {
        return Ok(None);
    };
    let cached_tokens = backend.cached_token_history();
    if completed_history.is_empty() || cached_tokens.is_empty() {
        return Ok(None);
    }
    if !live_history_media_matches(backend, completed_history) {
        return Ok(None);
    }

    let completed_tokens = tokenizer.apply_chat_template_sync(
        completed_history,
        Some(false),
        config.tools.as_deref(),
        crate::engine::params::resolve_enable_thinking(config),
    )?;
    let cached_comparison_tokens = backend.template_history_comparison_tokens(cached_tokens);
    let cached_text = tokenizer.decode_sync(&cached_comparison_tokens, false)?;
    let completed_text = tokenizer.decode_sync(&completed_tokens, false)?;
    let full_text = tokenizer.decode_sync(full_tokens, false)?;
    let Some(reasoning_boundaries) =
        structured_reasoning_boundary_ordinals(tokenizer, completed_history, config)?
    else {
        return Ok(None);
    };
    if !cached_structured_reasoning_matches(&cached_text, &reasoning_boundaries) {
        return Ok(None);
    }
    let reasoning_boundary_ordinals = reasoning_boundaries
        .iter()
        .map(|boundary| boundary.ordinal)
        .collect::<Vec<_>>();
    let Some((cached_comparison_text, _)) =
        normalize_reasoning_boundaries(&cached_text, &reasoning_boundary_ordinals, false)
    else {
        return Ok(None);
    };
    let Some((completed_comparison_text, _)) =
        normalize_reasoning_boundaries(&completed_text, &reasoning_boundary_ordinals, true)
    else {
        return Ok(None);
    };
    let Some((full_comparison_text, full_source_boundaries)) =
        normalize_reasoning_boundaries(&full_text, &reasoning_boundary_ordinals, true)
    else {
        return Ok(None);
    };

    if !completed_comparison_text.starts_with(&cached_comparison_text)
        || !full_comparison_text.starts_with(&completed_comparison_text)
    {
        return Ok(None);
    }

    let suffix_start = full_source_boundaries[cached_comparison_text.len()];
    let suffix_text = &full_text[suffix_start..];
    let suffix_tokens = tokenizer.encode_sync(suffix_text, Some(false))?;
    let mut continuation = Vec::with_capacity(cached_tokens.len() + suffix_tokens.len());
    continuation.extend_from_slice(cached_tokens);
    continuation.extend_from_slice(&suffix_tokens);

    // Tokenizer decoders can normalize byte sequences. Only use the splice
    // when the combined IDs still decode to the same logical placeholder and
    // reasoning-boundary form as the full template render; otherwise fall
    // back to the ordinary cold replay.
    let continuation_comparison_tokens = backend.template_history_comparison_tokens(&continuation);
    let continuation_text = tokenizer.decode_sync(&continuation_comparison_tokens, false)?;
    let Some((continuation_comparison_text, _)) =
        normalize_reasoning_boundaries(&continuation_text, &reasoning_boundary_ordinals, true)
    else {
        return Ok(None);
    };
    if continuation_comparison_text != full_comparison_text {
        return Ok(None);
    }

    Ok(Some(continuation))
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
    turn_kind: TurnKind,
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
    let pending_messages = match turn_kind {
        TurnKind::Start => messages.as_slice(),
        TurnKind::Continue => messages.last().map(std::slice::from_ref).unwrap_or(&[]),
    };
    let pending_media = media_capabilities_from_messages(pending_messages);
    if pending_media.images && turn_kind == TurnKind::Continue {
        return Err(Error::from_reason(format!(
            "{IMAGE_CHANGE_RESTART_PREFIX} continuation messages cannot replace the media held by the live session",
        )));
    }
    if pending_media.images && !admitted_media.images {
        return Err(Error::from_reason(format!(
            "{IMAGE_CHANGE_RESTART_PREFIX} this model is text-only; image messages are not supported",
        )));
    }
    if pending_media.audio && turn_kind == TurnKind::Continue {
        return Err(Error::from_reason(format!(
            "{IMAGE_CHANGE_RESTART_PREFIX} continuation messages cannot replace the media held by the live session",
        )));
    }
    if pending_media.audio && !admitted_media.audio {
        return Err(Error::from_reason(format!(
            "{IMAGE_CHANGE_RESTART_PREFIX} this model has no audio support; audio messages are not supported",
        )));
    }

    let full_tokens = backend.render_prompt(&tokenizer, &messages, &config)?;
    let live_continuation = if turn_kind == TurnKind::Continue {
        render_live_continuation(backend, &tokenizer, &messages, &config, &full_tokens)?
    } else {
        None
    };
    let is_delta = live_continuation.is_some();
    let tokens = live_continuation.unwrap_or(full_tokens);

    // A successful live splice carries only a text suffix on top of existing
    // session state. A cold replay must feed every historical media payload
    // back through the model.
    let images = if is_delta {
        Vec::new()
    } else {
        extract_images_from_messages(&messages)
    };
    if !images.is_empty() && !admitted_media.images {
        return Err(Error::from_reason(format!(
            "{IMAGE_CHANGE_RESTART_PREFIX} this model is text-only; image messages are not supported",
        )));
    }
    // Audio mirrors the image guard: an audio-bearing turn against a model
    // with no audio support is rejected with the typed restart prefix before
    // `render_prompt`. Every non-audio family rejects here and image-only /
    // text-only flows stay byte-identical.
    let audio = if is_delta {
        Vec::new()
    } else {
        extract_audio_from_messages(&messages)
    };
    if !audio.is_empty() && !admitted_media.audio {
        return Err(Error::from_reason(format!(
            "{IMAGE_CHANGE_RESTART_PREFIX} this model has no audio support; audio messages are not supported",
        )));
    }
    let media = MediaInputs {
        images: &images,
        audio: &audio,
    };
    let input_media = media.capabilities();
    // A live continuation adds no new media but may reuse image/audio state
    // already encoded in the held caches. A cold full-history replay owns all
    // current media itself.
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

#[cfg(test)]
mod continuation_comparison_tests {
    use super::{
        StructuredReasoningBoundary, cached_structured_reasoning_matches,
        locate_structured_reasoning_boundaries, normalize_reasoning_boundaries,
    };

    #[test]
    fn reasoning_boundary_normalization_maps_back_to_the_exact_suffix() {
        let cached = "a</think>\n</think>\n\nbody for";
        let full = "a\n</think>\n\n</think>\n\nbody for the<|im_end|>\n";
        let (cached_normalized, _) =
            normalize_reasoning_boundaries(cached, &[0], true).expect("cached boundary exists");
        let (full_normalized, full_boundaries) =
            normalize_reasoning_boundaries(full, &[0], true).expect("full boundary exists");

        assert!(full_normalized.starts_with(&cached_normalized));
        let suffix_start = full_boundaries[cached_normalized.len()];
        assert_eq!(&full[suffix_start..], " the<|im_end|>\n");
    }

    #[test]
    fn reasoning_boundary_normalization_preserves_ordinary_whitespace() {
        let (single_space, _) = normalize_reasoning_boundaries("hello world", &[], true)
            .expect("no boundaries required");
        let (double_space, _) = normalize_reasoning_boundaries("hello  world", &[], true)
            .expect("no boundaries required");
        assert_ne!(single_space, double_space);
    }

    #[test]
    fn reasoning_boundary_normalization_preserves_literal_tags() {
        let text = "user: a </think> b\nassistant: thought \n</think>\n\nanswer";
        let (normalized, _) =
            normalize_reasoning_boundaries(text, &[1], true).expect("assistant boundary exists");

        assert!(
            normalized.starts_with("user: a </think> b\n"),
            "the unselected literal user tag and its whitespace must remain exact"
        );
        assert!(normalized.ends_with("assistant: thought</think>answer"));
    }

    #[test]
    fn cached_prefix_may_end_before_a_proven_reasoning_boundary() {
        assert_eq!(
            normalize_reasoning_boundaries("assistant: partial thought", &[0], false)
                .map(|(text, _)| text),
            Some("assistant: partial thought".to_string())
        );
        assert!(
            normalize_reasoning_boundaries("assistant: partial thought", &[0], true).is_none(),
            "a complete template render must contain every proven boundary"
        );
    }

    #[test]
    fn structured_reasoning_locator_rejects_unexplained_render_changes() {
        let replacements = vec![(
            "__SENTINEL__".to_string(),
            "private".to_string(),
            "answer".to_string(),
        )];
        assert_eq!(
            locate_structured_reasoning_boundaries(
                "user literal </think>\nassistant private\n</think>\nanswer",
                "user literal </think>\nassistant __SENTINEL__\n</think>\nanswer",
                &replacements,
            ),
            Some(vec![40])
        );
        assert_eq!(
            locate_structured_reasoning_boundaries(
                "user edited </think>\nassistant private\n</think>\nanswer",
                "user literal </think>\nassistant __SENTINEL__\n</think>\nanswer",
                &replacements,
            ),
            None,
            "any difference outside the structured reasoning substitution fails closed"
        );
    }

    #[test]
    fn cached_reasoning_identity_rejects_message_owned_boundary_whitespace_edits() {
        let cached = "assistant thought</think>\nanswer";
        let exact = StructuredReasoningBoundary {
            ordinal: 0,
            reasoning: "thought".to_string(),
            content: "answer continues beyond the cached prefix".to_string(),
        };
        assert!(cached_structured_reasoning_matches(cached, &[exact]));

        let edited_reasoning = StructuredReasoningBoundary {
            ordinal: 0,
            reasoning: "thought ".to_string(),
            content: "answer".to_string(),
        };
        assert!(!cached_structured_reasoning_matches(
            cached,
            &[edited_reasoning]
        ));

        let edited_content = StructuredReasoningBoundary {
            ordinal: 0,
            reasoning: "thought".to_string(),
            content: " answer".to_string(),
        };
        assert!(!cached_structured_reasoning_matches(
            cached,
            &[edited_content]
        ));
    }
}
