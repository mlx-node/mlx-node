/**
 * Maps ChatResult / ChatStreamEvent to Anthropic Messages API output.
 */

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
 * Parse tool call arguments into an object.
 * If the arguments are already an object, use them directly.
 * If they are a string, JSON.parse them.
 */
function parseArguments(args: Record<string, unknown> | string): Record<string, unknown> {
  if (typeof args === 'string') {
    return JSON.parse(args) as Record<string, unknown>;
  }
  return args;
}

/**
 * Map the internal finishReason to the Anthropic stop_reason.
 */
export function mapStopReason(finishReason: string, hasToolCalls: boolean): 'end_turn' | 'max_tokens' | 'tool_use' {
  if (finishReason === 'length') {
    return 'max_tokens';
  }
  if (hasToolCalls) {
    return 'tool_use';
  }
  return 'end_turn';
}

/**
 * Build the content array from a ChatResult for the Anthropic Messages API.
 */
export function buildAnthropicContent(result: ChatResult): AnthropicResponseContent[] {
  const content: AnthropicResponseContent[] = [];

  // Thinking block
  if (result.thinking) {
    content.push({ type: 'thinking', thinking: result.thinking });
  }

  // Tool use blocks (only successfully parsed tool calls)
  const okToolCalls = result.toolCalls.filter((t) => t.status === 'ok');

  // Text block (present if text is non-empty, or if there are no ok tool calls)
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

/**
 * Build a full AnthropicMessagesResponse from a ChatResult and the original request.
 */
export function buildAnthropicResponse(
  result: ChatResult,
  req: AnthropicMessagesRequest,
  messageId: string,
): AnthropicMessagesResponse {
  const okToolCalls = result.toolCalls.filter((t) => t.status === 'ok');
  const hasToolCalls = okToolCalls.length > 0;

  return {
    id: messageId,
    type: 'message',
    role: 'assistant',
    model: req.model,
    content: buildAnthropicContent(result),
    stop_reason: mapStopReason(result.finishReason, hasToolCalls),
    stop_sequence: null,
    usage: {
      input_tokens: result.promptTokens,
      output_tokens: result.numTokens,
    },
  };
}

// ---------------------------------------------------------------------------
// Streaming helpers
// ---------------------------------------------------------------------------

/**
 * Build a message_start event.
 * The embedded message has empty content and zero output_tokens at this point.
 */
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

/**
 * Build a content_block_start event.
 */
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

/**
 * Build a content_block_delta event.
 */
export function buildContentBlockDelta(index: number, delta: AnthropicDelta): AnthropicContentBlockDeltaEvent {
  return {
    type: 'content_block_delta',
    index,
    delta,
  };
}

/**
 * Build a content_block_stop event.
 */
export function buildContentBlockStop(index: number): AnthropicContentBlockStopEvent {
  return {
    type: 'content_block_stop',
    index,
  };
}

/**
 * Build a message_delta event with the final stop_reason and token count.
 */
export function buildMessageDelta(
  stopReason: string,
  outputTokens: number,
  inputTokens?: number,
): AnthropicMessageDeltaEvent {
  return {
    type: 'message_delta',
    delta: {
      stop_reason: stopReason,
      stop_sequence: null,
    },
    usage: {
      ...(inputTokens != null ? { input_tokens: inputTokens } : {}),
      output_tokens: outputTokens,
    },
  };
}

/**
 * Build a message_stop event.
 */
export function buildMessageStop(): AnthropicMessageStopEvent {
  return {
    type: 'message_stop',
  };
}
