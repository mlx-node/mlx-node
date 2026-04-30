/** ChatResult / ChatStreamEvent → Anthropic Messages API output. */

import type { ChatResult } from '@mlx-node/core';

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
  AnthropicResponseContent,
} from '../types-anthropic.js';
import { genId } from './response.js';

/**
 * Subset of `@mlx-node/core`'s `PerformanceMetrics` consumed by the
 * Anthropic mapper. Defined locally to keep the mapper independent of
 * the native package's exported shape — only the three fields we
 * forward onto the wire are required, and each one is gated through
 * `Number.isFinite(...) && > 0` before emission so a partially-plumbed
 * driver (or a future shape that adds fields) cannot leak invalid
 * values through.
 */
export interface PerformanceMetricsForUsage {
  ttftMs?: number;
  prefillTokensPerSecond?: number;
  decodeTokensPerSecond?: number;
}

/**
 * Mirror the cache-field gating pattern: emit `key` on `usage` only
 * when `value` is a finite, strictly-positive number. Missing,
 * non-finite, zero, or negative metrics are elided (the launcher's
 * verbose log treats absence as "unknown / not plumbed", which is
 * the correct read for a turn whose native dispatch did not produce
 * the metric).
 */
function assignFinitePositive<T extends Record<string, number | undefined>>(
  usage: T,
  key: keyof T & string,
  value: number | undefined,
): void {
  if (value != null && Number.isFinite(value) && value > 0) {
    (usage as Record<string, number | undefined>)[key] = value;
  }
}

function mergePerformanceIntoUsage(
  usage: { time_to_first_token_ms?: number; prefill_tokens_per_second?: number; decode_tokens_per_second?: number },
  performance: PerformanceMetricsForUsage | undefined,
): void {
  if (performance == null) return;
  assignFinitePositive(usage, 'time_to_first_token_ms', performance.ttftMs);
  assignFinitePositive(usage, 'prefill_tokens_per_second', performance.prefillTokensPerSecond);
  assignFinitePositive(usage, 'decode_tokens_per_second', performance.decodeTokensPerSecond);
}

function parseArguments(args: Record<string, unknown> | string): Record<string, unknown> {
  if (typeof args === 'string') {
    return JSON.parse(args) as Record<string, unknown>;
  }
  return args;
}

export function mapStopReason(finishReason: string, hasToolCalls: boolean): 'end_turn' | 'max_tokens' | 'tool_use' {
  if (finishReason === 'length') {
    return 'max_tokens';
  }
  if (hasToolCalls) {
    return 'tool_use';
  }
  return 'end_turn';
}

export function buildAnthropicContent(result: ChatResult): AnthropicResponseContent[] {
  const content: AnthropicResponseContent[] = [];

  if (result.thinking) {
    content.push({ type: 'thinking', thinking: result.thinking });
  }

  const okToolCalls = result.toolCalls.filter((t) => t.status === 'ok');

  // Emit a text block unless tool calls exist and there is no text.
  if (result.text || okToolCalls.length === 0) {
    content.push({ type: 'text', text: result.text });
  }

  for (const tc of okToolCalls) {
    content.push({
      type: 'tool_use',
      id: tc.id ?? genId('toolu_'),
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
): AnthropicMessagesResponse {
  const okToolCalls = result.toolCalls.filter((t) => t.status === 'ok');
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
  // measurement.
  mergePerformanceIntoUsage(usage, performance);

  return {
    id: messageId,
    type: 'message',
    role: 'assistant',
    model: req.model,
    content: buildAnthropicContent(result),
    stop_reason: mapStopReason(result.finishReason, hasToolCalls),
    stop_sequence: null,
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
): AnthropicMessageDeltaEvent {
  // Streaming `message_delta` mirrors the non-streaming response's
  // cache accounting: when `cachedTokens > 0` we emit
  // `cache_read_input_tokens: cachedTokens` AND subtract that count
  // from `input_tokens`. On a cache miss (or when `cachedTokens` is
  // omitted by an in-process driver / mock) the cache fields stay
  // off the wire. See the matching block on `buildAnthropicResponse`
  // and the field-level docstrings on `AnthropicUsage`.
  const usage: AnthropicMessageDeltaEvent['usage'] = { output_tokens: outputTokens };
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
  mergePerformanceIntoUsage(usage, performance);
  return {
    type: 'message_delta',
    delta: {
      stop_reason: stopReason,
      stop_sequence: null,
    },
    usage,
  };
}

export function buildMessageStop(): AnthropicMessageStopEvent {
  return {
    type: 'message_stop',
  };
}
