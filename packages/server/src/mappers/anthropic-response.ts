/** ChatResult / ChatStreamEvent → Anthropic Messages API output. */

import type { ChatResult, ChatStreamProgress } from '@mlx-node/core';

import { sanitizeJsonString } from '../strict-json.js';
import { mergeTimingUsageExtensions, type PerformanceMetricsForUsage, type ServerTimingForUsage } from '../timing.js';
import type {
  AnthropicContentBlockDeltaEvent,
  AnthropicContentBlockStartEvent,
  AnthropicContentBlockStopEvent,
  AnthropicDelta,
  AnthropicMessageDeltaEvent,
  AnthropicMessageStartEvent,
  AnthropicMessageStopEvent,
  AnthropicMessagesRequest,
  AnthropicMessagesResponse,
  AnthropicPingEvent,
  AnthropicResponseContent,
  AnthropicStopReason,
  MlxProgress,
} from '../types-anthropic.js';
import { genId } from './response.js';

function parseArguments(args: Record<string, unknown> | string): Record<string, unknown> {
  if (typeof args === 'string') {
    // Model-generated arguments are verbatim JSON text: a raw control
    // character inside one of its string literals makes a bare JSON.parse
    // throw and sinks the whole turn into a 500. Sanitizing first is
    // byte-identical for compliant input and repairs the raw bytes to
    // proper escapes.
    try {
      const parsed: unknown = JSON.parse(sanitizeJsonString(args));
      // `tool_use.input` is a JSON OBJECT on the wire: malformed arguments
      // (structurally broken even after sanitizing) or valid JSON that is not
      // an object must not throw or emit a schema-invalid input. Fall back to
      // the same `{}` the streaming tool_use block starts with.
      if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
      return {};
    } catch {
      return {};
    }
  }
  return args;
}

/**
 * Translate a native tool-call id (minted as `call_<uuid>` by the Rust
 * parser, which keeps the OpenAI Responses convention) to the Anthropic
 * Messages wire convention (`toolu_<uuid>`). The uuid body is preserved
 * verbatim so the inverse `anthropicToolUseIdToInternal` round-trips
 * losslessly when clients echo the id back via `tool_result.tool_use_id`.
 *
 * Defensive: ids that do not have the expected `call_` prefix (e.g. a
 * legacy caller, an in-process driver constructing its own id, or a
 * future variant) pass through unchanged so this function never
 * synthesizes a wrong id.
 */
export function internalToolCallIdToAnthropic(id: string): string {
  return id.startsWith('call_') ? `toolu_${id.slice('call_'.length)}` : id;
}

/**
 * Inverse of `internalToolCallIdToAnthropic`: translate an incoming
 * Anthropic `tool_use_id` (`toolu_<uuid>`) back to the internal `call_*`
 * shape so historical assistant turns and the tool_call_id lookup in
 * the native session store keep matching. Ids without the expected
 * `toolu_` prefix pass through unchanged for the same defensive reason
 * (some legacy callers send raw `call_*` directly).
 */
export function anthropicToolUseIdToInternal(id: string): string {
  return id.startsWith('toolu_') ? `call_${id.slice('toolu_'.length)}` : id;
}

export function mapStopReason(
  finishReason: string,
  hasToolCalls: boolean,
  matchedStopSequence?: string | null,
  outputClampedToContext = false,
): AnthropicStopReason {
  if (matchedStopSequence) {
    return 'stop_sequence';
  }
  if (finishReason === 'length') {
    // Splash `stop_reason`: the context window, not `max_tokens`, ended it.
    return outputClampedToContext ? 'model_context_window_exceeded' : 'max_tokens';
  }
  if (hasToolCalls) {
    return 'tool_use';
  }
  return 'end_turn';
}

/**
 * The answer text a response shows: the parsed text, or the recovered markup
 * of tool calls the request did not allow, with the raw block of each call
 * that names no function read as text where the model wrote it (Splash
 * `StreamingToolCallProjector._begin_call`). Splash cuts the raw output at a
 * stop (`CallbackStreamer._emit`) before the projector reads it, so the block
 * must keep its place for the stop search to cut what follows it.
 */
export function visibleAnswerText(
  result: Pick<ChatResult, 'text' | 'rawText' | 'toolCalls'>,
  allowToolUse: boolean,
): string {
  const text =
    !allowToolUse &&
    result.text.length === 0 &&
    result.toolCalls.some((t) => t.status === 'ok') &&
    containsToolCallMarkup(result.rawText)
      ? recoverSuppressedToolCallText(result.rawText)
      : result.text;
  if (!result.toolCalls.some((call) => call.status === 'missing_name')) return text;
  const placed = withNamelessCallsInPlace(text, result.rawText, result.toolCalls);
  if (placed !== null) return placed;
  // The blocks cannot be placed in `rawText`: they go after the text.
  let nameless = '';
  for (const call of result.toolCalls) {
    if (call.status === 'missing_name') nameless += call.rawContent;
  }
  return text + nameless;
}

function isWhitespace(ch: string): boolean {
  return ch.trim().length === 0;
}

/**
 * `text` with each nameless call's block put back at its place in `rawText`,
 * or null when `text` is not `rawText` without the `<tool_call>` blocks,
 * trimmed and cut after the reasoning (the native finalize), or a nameless
 * block falls before the text starts.
 */
function withNamelessCallsInPlace(text: string, rawText: string, calls: ChatResult['toolCalls']): string | null {
  let stripped = '';
  let kept = '';
  const namelessAt: number[] = [];
  let cursor = 0;
  for (const call of calls) {
    if (call.rawContent.length === 0) return null;
    const start = rawText.indexOf(call.rawContent, cursor);
    if (start < 0) return null;
    const before = withoutUnlistedToolCallBlocks(rawText.slice(cursor, start));
    stripped += before;
    kept += before;
    if (call.status === 'missing_name') {
      namelessAt.push(stripped.length);
      kept += call.rawContent;
    }
    cursor = start + call.rawContent.length;
  }
  const tail = withoutUnlistedToolCallBlocks(rawText.slice(cursor));
  stripped += tail;
  kept += tail;

  const visible = text.trim();
  const head = stripped.trimEnd();
  if (!head.endsWith(visible)) return null;
  let start = head.length - visible.length;
  while (start > 0 && isWhitespace(stripped[start - 1]!)) start--;
  if (namelessAt.some((at) => at < start)) return null;
  // No nameless block precedes `start`, so it is the same offset in `kept`.
  return kept.slice(start).trim();
}

const TOOL_CALL_OPEN = '<tool_call>';
const TOOL_CALL_CLOSE = '</tool_call>';

/**
 * `segment` without the closed `<tool_call>` blocks the native parser strips
 * from the text but does not list as calls; the scan stops at an unclosed
 * open tag like the native `extract_tag_blocks`.
 */
function withoutUnlistedToolCallBlocks(segment: string): string {
  let out = '';
  let from = 0;
  for (;;) {
    const open = segment.indexOf(TOOL_CALL_OPEN, from);
    if (open < 0) break;
    const close = segment.indexOf(TOOL_CALL_CLOSE, open + TOOL_CALL_OPEN.length);
    if (close < 0) break;
    out += segment.slice(from, open);
    from = close + TOOL_CALL_CLOSE.length;
  }
  return out + segment.slice(from);
}

export function containsToolCallMarkup(rawText: string): boolean {
  return (
    rawText.includes('<tool_call') ||
    rawText.includes('</tool_call') ||
    rawText.includes('<|tool_call') ||
    rawText.includes('<tool_call|>')
  );
}

export function recoverSuppressedToolCallText(rawText: string): string {
  return (
    rawText
      .replace(/<\|channel>[\s\S]*?(?:<channel\|>|$)/g, '')
      .replace(/<channel\|>/g, '')
      .replace(/<\|tool_call>[\s\S]*?(?:<tool_call\|>|$)/g, '')
      .replace(/<tool_call>[\s\S]*?(?:<\/tool_call>|$)/g, '')
      .replace(/<\|tool_response>[\s\S]*?(?:<tool_response\|>|$)/g, '')
      .replace(/<\|tool>[\s\S]*?(?:<tool\|>|$)/g, '')
      // A complete LFM2 block plus any echoed call bodies that follow it —
      // the model may repeat `[f()]<|tool_call_end|>` without a start
      // sentinel. The optional tempered body can't cross another
      // `<|tool_call_start|>`, so real later blocks (and prose before
      // them) survive; a bare orphan end sentinel is still dropped below.
      .replace(
        /<\|tool_call_start\|>[\s\S]*?<\|tool_call_end\|>(?:(?:(?!<\|tool_call_start\|>)[\s\S])*<\|tool_call_end\|>)?|<\|tool_call_start\|>[\s\S]*$/g,
        '',
      )
      .replace(/<\|tool_call_end\|>/g, '')
      .replace(/<\|turn>[^\n]*(?:\n|$)/g, '')
      .replace(/<turn\|>/g, '')
  );
}

export function buildAnthropicContent(
  result: ChatResult,
  allowToolUse = true,
  stopMatched = false,
): AnthropicResponseContent[] {
  const content: AnthropicResponseContent[] = [];

  if (result.thinking) {
    // Local models have no encrypted-reasoning service; the empty placeholder
    // keeps strict Anthropic-schema consumers (grok fork, Claude Code) happy.
    content.push({ type: 'thinking', thinking: result.thinking, signature: '' });
  }

  const parsedToolCalls = result.toolCalls.filter((t) => t.status === 'ok');
  // A matched stop sequence halts generation at its position, so any tool call
  // whose tag would have followed the stop boundary is dropped and the
  // truncated visible text (`result.text`, already cut at the stop) is emitted
  // verbatim — the suppressed-markup recovery is skipped because it would
  // re-introduce text that lived after the stop.
  const okToolCalls = stopMatched || !allowToolUse ? [] : parsedToolCalls;
  const text = stopMatched ? result.text : visibleAnswerText(result, allowToolUse);

  // Emit a text block unless tool calls exist and there is no text.
  if (text || okToolCalls.length === 0) {
    content.push({ type: 'text', text });
  }

  for (const tc of okToolCalls) {
    // Translate native `call_<uuid>` to Anthropic `toolu_<uuid>` at the
    // wire boundary. The fallback `genId('toolu_')` covers the case where
    // the native parser did not mint an id (an in-process driver or a
    // legacy bridge — present-day Rust paths always populate it).
    content.push({
      type: 'tool_use',
      id: tc.id != null ? internalToolCallIdToAnthropic(tc.id) : genId('toolu_'),
      name: tc.name,
      input: parseArguments(tc.arguments),
    });
  }

  return content;
}

export function buildAnthropicResponse(
  result: ChatResult,
  req: AnthropicMessagesRequest,
  messageId: string,
  performance?: PerformanceMetricsForUsage,
  allowToolUse = true,
  serverTiming?: ServerTimingForUsage,
  matchedStopSequence?: string | null,
  outputClampedToContext = false,
): AnthropicMessagesResponse {
  // A matched stop sequence takes precedence over tool_use: suppress the tool
  // calls so `stop_reason: 'stop_sequence'` is never emitted alongside a
  // tool_use block whose tag followed the stop boundary.
  const stopMatched = Boolean(matchedStopSequence);
  const okToolCalls = allowToolUse && !stopMatched ? result.toolCalls.filter((t) => t.status === 'ok') : [];
  const hasToolCalls = okToolCalls.length > 0;

  // Cache accounting (Anthropic Messages API spec):
  //   * On a cache HIT (`cachedTokens > 0`) the wire MUST emit
  //     `cache_read_input_tokens: cachedTokens` and reduce
  //     `input_tokens` to the unsuffixed remainder
  //     `promptTokens - cachedTokens` — Claude Code (and other
  //     Anthropic-compatible UIs) read this directly for cost /
  //     billing display, and a wire that left `input_tokens` at the
  //     full prompt count would silently double-bill the cached
  //     prefix.
  //   * On a cache MISS (`cachedTokens === 0`) the cache fields are
  //     OMITTED — they are optional in the spec and other
  //     Anthropic-compatible servers elide them on misses.
  //   * `cache_creation_input_tokens` stays unset: this server's KV
  //     reuse is implicit (no `cache_control` breakpoints), so a
  //     client that did not request explicit caching should never
  //     see a non-zero creation count.
  const cachedTokens = result.cachedTokens;
  const usage: AnthropicMessagesResponse['usage'] =
    cachedTokens > 0
      ? {
          input_tokens: result.promptTokens - cachedTokens,
          output_tokens: result.numTokens,
          cache_read_input_tokens: cachedTokens,
        }
      : {
          input_tokens: result.promptTokens,
          output_tokens: result.numTokens,
        };

  // Server-extension perf fields. Same gating pattern as
  // `cache_read_input_tokens`: only land on the wire when the native
  // dispatch produced a finite, positive value — `undefined` /
  // `NaN` / `0` is elided so the launcher's verbose log can read
  // absence as "not plumbed" instead of treating zero as a real
  // measurement. Cache-context fields make the prefill rate explicit:
  // on cached-prefix turns the denominator is the uncached suffix, not
  // the full logical prompt.
  mergeTimingUsageExtensions(usage, performance, result.promptTokens, result.numTokens, cachedTokens, serverTiming);

  return {
    id: messageId,
    type: 'message',
    role: 'assistant',
    model: req.model,
    content: buildAnthropicContent(result, allowToolUse, stopMatched),
    stop_reason: mapStopReason(result.finishReason, hasToolCalls, matchedStopSequence, outputClampedToContext),
    stop_sequence: matchedStopSequence ?? null,
    usage,
  };
}

// Streaming helpers

/** Embedded message has empty content and zero output_tokens at start. */
export function buildMessageStartEvent(
  req: AnthropicMessagesRequest,
  messageId: string,
  inputTokens: number,
): AnthropicMessageStartEvent {
  return {
    type: 'message_start',
    message: {
      id: messageId,
      type: 'message',
      role: 'assistant',
      model: req.model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: {
        input_tokens: inputTokens,
        output_tokens: 0,
      },
    },
  };
}

export function buildContentBlockStart(
  index: number,
  block: AnthropicResponseContent,
): AnthropicContentBlockStartEvent {
  return {
    type: 'content_block_start',
    index,
    content_block: block,
  };
}

export function buildContentBlockDelta(index: number, delta: AnthropicDelta): AnthropicContentBlockDeltaEvent {
  return {
    type: 'content_block_delta',
    index,
    delta,
  };
}

export function buildContentBlockStop(index: number): AnthropicContentBlockStopEvent {
  return {
    type: 'content_block_stop',
    index,
  };
}

export function buildMessageDelta(
  stopReason: string,
  outputTokens: number,
  inputTokens?: number,
  cachedTokens?: number,
  performance?: PerformanceMetricsForUsage,
  serverTiming?: ServerTimingForUsage,
  stopSequence?: string | null,
): AnthropicMessageDeltaEvent {
  // Streaming `message_delta` mirrors the non-streaming response's
  // cache accounting: when `cachedTokens > 0` we emit
  // `cache_read_input_tokens: cachedTokens` AND subtract that count
  // from `input_tokens`. On a cache miss (or when `cachedTokens` is
  // omitted by an in-process driver / mock) the cache fields stay
  // off the wire. See the matching block on `buildAnthropicResponse`
  // and the field-level docstrings on `AnthropicUsage`.
  const usage: AnthropicMessageDeltaEvent['usage'] = {
    output_tokens: outputTokens,
  };
  if (cachedTokens != null && cachedTokens > 0) {
    if (inputTokens != null) {
      usage.input_tokens = inputTokens - cachedTokens;
    }
    usage.cache_read_input_tokens = cachedTokens;
  } else if (inputTokens != null) {
    usage.input_tokens = inputTokens;
  }
  // Server-extension perf fields — same gating pattern as the
  // cache-field block above. See `buildAnthropicResponse` for the
  // matching non-streaming branch and the docstring on
  // `AnthropicUsage` for the wire-format rationale.
  mergeTimingUsageExtensions(usage, performance, inputTokens, outputTokens, cachedTokens, serverTiming);
  return {
    type: 'message_delta',
    delta: {
      stop_reason: stopReason,
      stop_sequence: stopSequence ?? null,
    },
    usage,
  };
}

export function buildMessageStop(): AnthropicMessageStopEvent {
  return {
    type: 'message_stop',
  };
}

/** Tokens per second over a window; `undefined` unless both deltas are positive. */
function windowRate(tokens: number, ms: number): number | undefined {
  if (!(tokens > 0) || !(ms > 0)) return undefined;
  return Math.round((tokens * 10_000) / ms) / 10;
}

/**
 * Rate of `count` over `elapsed`, measured from the previous sample when that
 * sample is from the same phase, otherwise from the phase start.
 */
function phaseRate(
  count: number | undefined,
  elapsed: number | undefined,
  prevCount: number | undefined,
  prevElapsed: number | undefined,
): number | undefined {
  if (count === undefined || elapsed === undefined) return undefined;
  if (prevCount === undefined || prevElapsed === undefined) return windowRate(count, elapsed);
  return windowRate(count - prevCount, elapsed - prevElapsed);
}

/**
 * Build the `ping` frame for one progress tick. `previous` is the snapshot
 * the last emitted ping was built from. Returns `null` once the native turn
 * is done: its terminal frames are about to follow.
 */
export function buildProgressPing(
  snapshot: ChatStreamProgress,
  previous: ChatStreamProgress | null,
  elapsedMs: number,
): AnthropicPingEvent | null {
  if (snapshot.phase === 'done') return null;
  const progress: MlxProgress = {
    phase: snapshot.phase,
    elapsed_ms: Math.max(0, Math.round(elapsedMs)),
  };
  if (snapshot.prefillTotal !== undefined) progress.prefill_total = snapshot.prefillTotal;
  if (snapshot.prefillDone !== undefined) progress.prefill_done = snapshot.prefillDone;
  if (snapshot.cachedTokens !== undefined) progress.cached_tokens = snapshot.cachedTokens;
  if (snapshot.phase === 'prefill') {
    const samePhase = previous?.phase === 'prefill';
    const rate = phaseRate(
      snapshot.prefillDone,
      snapshot.prefillElapsedMs,
      samePhase ? previous.prefillDone : undefined,
      samePhase ? previous.prefillElapsedMs : undefined,
    );
    if (rate !== undefined) progress.prefill_tokens_per_second = rate;
  }
  // Cores that do not count kept tokens (gemma4 paged/vision, deepseek_v41,
  // qianfan_ocr) report 0 for the whole decode: that 0 means unknown.
  if (snapshot.generatedTokens !== undefined && snapshot.generatedTokens > 0) {
    progress.generated_tokens = snapshot.generatedTokens;
  }
  if (snapshot.phase === 'decode') {
    const samePhase = previous?.phase === 'decode';
    const rate = phaseRate(
      snapshot.generatedTokens,
      snapshot.decodeElapsedMs,
      samePhase ? previous.generatedTokens : undefined,
      samePhase ? previous.decodeElapsedMs : undefined,
    );
    if (rate !== undefined) progress.decode_tokens_per_second = rate;
  }
  return { type: 'ping', mlx_progress: progress };
}
