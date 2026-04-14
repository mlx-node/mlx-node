/**
 * POST /v1/messages endpoint
 *
 * Implements the Anthropic Messages API, dispatching to loaded models
 * via the ModelRegistry. Supports both streaming (SSE) and non-streaming
 * (JSON) response modes.
 *
 * The Anthropic Messages API is stateless — every request carries the
 * full conversation in `req.messages`, and no response store is
 * consulted. We therefore allocate a brand-new `ChatSession` for each
 * request (via `SessionRegistry.getOrCreate(null)`), prime it with the
 * mapped messages, and run `startFromHistory` / `startFromHistoryStream`.
 * No `adopt()` / `drop()` — the session's lifetime is this single call.
 */

import type { ServerResponse } from 'node:http';

import type { ChatConfig, ChatMessage, ChatResult } from '@mlx-node/core';
import type { ChatSession, ChatStreamEvent, SessionCapableModel } from '@mlx-node/lm';

import { sendAnthropicBadRequest, sendAnthropicInternalError, sendAnthropicNotFound } from '../errors.js';
import { mapAnthropicRequest } from '../mappers/anthropic-request.js';
import {
  buildAnthropicResponse,
  buildContentBlockDelta,
  buildContentBlockStart,
  buildContentBlockStop,
  buildMessageDelta,
  buildMessageStartEvent,
  buildMessageStop,
  mapStopReason,
} from '../mappers/anthropic-response.js';
import { genId } from '../mappers/response.js';
import type { ModelRegistry } from '../registry.js';
import type { SessionRegistry } from '../session-registry.js';
import { beginSSE, endSSE, writeSSEEvent } from '../streaming.js';
import { ToolCallTagBuffer } from '../tool-call-buffer.js';
import type { AnthropicMessagesRequest } from '../types-anthropic.js';

// ---------------------------------------------------------------------------
// Non-streaming path
// ---------------------------------------------------------------------------

async function handleNonStreaming(
  res: ServerResponse,
  result: ChatResult,
  body: AnthropicMessagesRequest,
): Promise<void> {
  const messageId = genId('msg_');
  const response = buildAnthropicResponse(result, body, messageId);
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(response));
}

// ---------------------------------------------------------------------------
// Streaming path
// ---------------------------------------------------------------------------

async function handleStreamingNative(
  res: ServerResponse,
  chatStream: AsyncGenerator<ChatStreamEvent>,
  body: AnthropicMessagesRequest,
): Promise<void> {
  const messageId = genId('msg_');
  beginSSE(res);

  writeSSEEvent(res, 'message_start', buildMessageStartEvent(body, messageId, 0));

  let contentBlockIndex = 0;
  let hasEmittedThinking = false;
  let hasEmittedText = false;
  let emittedTextLength = 0;
  const tagBuffer = new ToolCallTagBuffer();

  for await (const event of chatStream) {
    if (event.done) {
      // Final event

      // Flush any remaining pending text
      const remainingText = tagBuffer.flush();
      if (!tagBuffer.suppressed && remainingText) {
        if (!hasEmittedText) {
          // Close thinking block if open
          if (hasEmittedThinking) {
            writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex - 1));
          }
          hasEmittedText = true;
          writeSSEEvent(
            res,
            'content_block_start',
            buildContentBlockStart(contentBlockIndex, { type: 'text', text: '' }),
          );
        }
        emittedTextLength += remainingText.length;
        writeSSEEvent(
          res,
          'content_block_delta',
          buildContentBlockDelta(contentBlockIndex, { type: 'text_delta', text: remainingText }),
        );
      }

      // Close thinking block if open and text was never emitted
      if (hasEmittedThinking && !hasEmittedText) {
        writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex - 1));
      }

      // Handle final text
      const finalText = event.text;
      const okToolCalls = event.toolCalls.filter((t) => t.status === 'ok');
      const hasToolCalls = okToolCalls.length > 0;

      // Recovery: if tool-call suppression was triggered but no tool calls were parsed,
      // create a text block from the final event text (no text was streamed before suppression)
      if (tagBuffer.suppressed && !hasToolCalls && finalText && !hasEmittedText) {
        if (hasEmittedThinking) {
          // Thinking block already closed above
        }
        hasEmittedText = true;
        writeSSEEvent(
          res,
          'content_block_start',
          buildContentBlockStart(contentBlockIndex, { type: 'text', text: '' }),
        );
        emittedTextLength += finalText.length;
        writeSSEEvent(
          res,
          'content_block_delta',
          buildContentBlockDelta(contentBlockIndex, { type: 'text_delta', text: finalText }),
        );
      } else if (tagBuffer.suppressed && !hasToolCalls && finalText && hasEmittedText) {
        // Recovery: text was already being streamed but got cut off by a false-alarm <tool_call>
        // tag. Emit the portion of the final text that was never sent as a delta.
        const unsent = finalText.slice(emittedTextLength);
        if (unsent) {
          emittedTextLength += unsent.length;
          writeSSEEvent(
            res,
            'content_block_delta',
            buildContentBlockDelta(contentBlockIndex, { type: 'text_delta', text: unsent }),
          );
        }
      }

      // Emit any unsent suffix when final text is longer than what was streamed
      if (hasEmittedText && finalText && finalText.length > emittedTextLength) {
        const unsent = finalText.slice(emittedTextLength);
        emittedTextLength += unsent.length;
        writeSSEEvent(
          res,
          'content_block_delta',
          buildContentBlockDelta(contentBlockIndex, { type: 'text_delta', text: unsent }),
        );
      }

      if (hasEmittedText) {
        writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex));
        contentBlockIndex++;
      } else if (!finalText && hasToolCalls) {
        // No text at all and tool calls present -- skip text block entirely
      } else if (finalText) {
        // Text was never emitted during streaming but final has text
        // (possible if all text arrived in the final event somehow)
        writeSSEEvent(
          res,
          'content_block_start',
          buildContentBlockStart(contentBlockIndex, { type: 'text', text: '' }),
        );
        emittedTextLength += finalText.length;
        writeSSEEvent(
          res,
          'content_block_delta',
          buildContentBlockDelta(contentBlockIndex, { type: 'text_delta', text: finalText }),
        );
        writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex));
        contentBlockIndex++;
      }

      // Emit tool_use blocks
      for (const tc of okToolCalls) {
        const toolId = tc.id ?? genId('toolu_');
        const parsedInput =
          typeof tc.arguments === 'string'
            ? (JSON.parse(tc.arguments) as Record<string, unknown>)
            : (tc.arguments as Record<string, unknown>);

        writeSSEEvent(
          res,
          'content_block_start',
          buildContentBlockStart(contentBlockIndex, { type: 'tool_use', id: toolId, name: tc.name, input: {} }),
        );
        writeSSEEvent(
          res,
          'content_block_delta',
          buildContentBlockDelta(contentBlockIndex, {
            type: 'input_json_delta',
            partial_json: JSON.stringify(parsedInput),
          }),
        );
        writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex));
        contentBlockIndex++;
      }

      // Emit message_delta and message_stop
      const stopReason = mapStopReason(event.finishReason, hasToolCalls);
      writeSSEEvent(res, 'message_delta', buildMessageDelta(stopReason, event.numTokens, event.promptTokens));
      writeSSEEvent(res, 'message_stop', buildMessageStop());

      endSSE(res);
      return;
    }

    // Delta event
    if (event.isReasoning) {
      // Filter out </think> tag
      const deltaText = event.text.replace(/<\/think>/g, '');
      if (!deltaText) continue;

      if (!hasEmittedThinking) {
        hasEmittedThinking = true;
        writeSSEEvent(
          res,
          'content_block_start',
          buildContentBlockStart(contentBlockIndex, { type: 'thinking', thinking: '' }),
        );
        contentBlockIndex++;
      }
      writeSSEEvent(
        res,
        'content_block_delta',
        buildContentBlockDelta(contentBlockIndex - 1, { type: 'thinking_delta', thinking: deltaText }),
      );
    } else {
      // Text delta with tool_call tag buffering
      const { safeText, tagFound, cleanPrefix } = tagBuffer.push(event.text);
      if (tagFound) {
        if (cleanPrefix.trim()) {
          if (!hasEmittedText) {
            if (hasEmittedThinking) {
              writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex - 1));
            }
            hasEmittedText = true;
            writeSSEEvent(
              res,
              'content_block_start',
              buildContentBlockStart(contentBlockIndex, { type: 'text', text: '' }),
            );
          }
          emittedTextLength += cleanPrefix.length;
          writeSSEEvent(
            res,
            'content_block_delta',
            buildContentBlockDelta(contentBlockIndex, { type: 'text_delta', text: cleanPrefix }),
          );
        }
      } else if (safeText) {
        if (!hasEmittedText) {
          if (hasEmittedThinking) {
            writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex - 1));
          }
          hasEmittedText = true;
          writeSSEEvent(
            res,
            'content_block_start',
            buildContentBlockStart(contentBlockIndex, { type: 'text', text: '' }),
          );
        }
        emittedTextLength += safeText.length;
        writeSSEEvent(
          res,
          'content_block_delta',
          buildContentBlockDelta(contentBlockIndex, { type: 'text_delta', text: safeText }),
        );
      }
    }
  }

  // Safety net: if the async iterator exhausted without a done event,
  // emit terminal events so clients don't see a dangling stream.
  if (hasEmittedThinking && !hasEmittedText) {
    // Thinking block was opened but text block was never started — close thinking
    writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex - 1));
  }
  if (hasEmittedText) {
    writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex));
  }
  writeSSEEvent(res, 'message_delta', buildMessageDelta('end_turn', 0));
  writeSSEEvent(res, 'message_stop', buildMessageStop());
  endSSE(res);
}

// ---------------------------------------------------------------------------
// Session routing
// ---------------------------------------------------------------------------

/**
 * Run a stateless Anthropic request through a fresh `ChatSession`.
 *
 * The Anthropic API carries the full conversation in `req.messages`,
 * so there's no cache hit path. Every request primes a new session
 * and runs `startFromHistory`.
 */
async function runSessionNonStreaming(
  session: ChatSession<SessionCapableModel>,
  messages: ChatMessage[],
  config: ChatConfig,
): Promise<ChatResult> {
  session.primeHistory(messages);
  return await session.startFromHistory(config);
}

function runSessionStreaming(
  session: ChatSession<SessionCapableModel>,
  messages: ChatMessage[],
  config: ChatConfig,
): AsyncGenerator<ChatStreamEvent> {
  session.primeHistory(messages);
  return session.startFromHistoryStream(config);
}

// ---------------------------------------------------------------------------
// Public handler
// ---------------------------------------------------------------------------

export async function handleCreateMessage(
  res: ServerResponse,
  body: AnthropicMessagesRequest,
  registry: ModelRegistry,
): Promise<void> {
  // Validate required fields
  if (body == null || typeof body !== 'object') {
    sendAnthropicBadRequest(res, 'Request body must be a JSON object');
    return;
  }
  if (!body.model) {
    sendAnthropicBadRequest(res, 'Missing required field: model');
    return;
  }
  if (!body.messages || !Array.isArray(body.messages) || body.messages.length === 0) {
    sendAnthropicBadRequest(res, 'Missing required field: messages');
    return;
  }
  if (body.max_tokens == null || !Number.isInteger(body.max_tokens) || body.max_tokens <= 0) {
    sendAnthropicBadRequest(res, 'Missing required field: max_tokens');
    return;
  }

  // Validate message items are non-null objects
  for (const msg of body.messages) {
    if (msg == null || typeof msg !== 'object') {
      sendAnthropicBadRequest(res, 'Each message must be a non-null object');
      return;
    }
  }

  // Look up model
  const model = registry.get(body.model);
  if (!model) {
    sendAnthropicNotFound(res, `Model "${body.model}" not found`);
    return;
  }

  // Fetch the per-model session registry. The Anthropic API is
  // stateless — we allocate a fresh session for every request via
  // `getOrCreate(null)` and never adopt it into the cache.
  const sessionReg: SessionRegistry | undefined = registry.getSessionRegistry(body.model);
  if (!sessionReg) {
    sendAnthropicInternalError(res, 'session registry missing for registered model');
    return;
  }

  // Map request
  let messages: ChatMessage[];
  let config: ChatConfig;
  try {
    ({ messages, config } = mapAnthropicRequest(body));
  } catch (err) {
    sendAnthropicBadRequest(res, err instanceof Error ? err.message : 'Invalid request');
    return;
  }

  const session = sessionReg.getOrCreate(null);

  try {
    if (body.stream === true) {
      const chatStream = runSessionStreaming(session, messages, config);
      await handleStreamingNative(res, chatStream, body);
    } else {
      const result = await runSessionNonStreaming(session, messages, config);
      await handleNonStreaming(res, result, body);
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error during inference';
    if (!res.headersSent) {
      sendAnthropicInternalError(res, message);
    } else {
      // Headers already sent (streaming) -- best effort: write error event and close
      writeSSEEvent(res, 'error', { error: { type: 'api_error', message } });
      endSSE(res);
    }
  }
}
