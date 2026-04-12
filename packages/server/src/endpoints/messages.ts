/**
 * POST /v1/messages endpoint
 *
 * Implements the Anthropic Messages API, dispatching to loaded models
 * via the ModelRegistry. Supports both native streaming (SSE), simulated
 * streaming (from chat()), and non-streaming (JSON) response modes.
 */

import type { ServerResponse } from 'node:http';

import type { ChatConfig, ChatMessage, ChatResult } from '@mlx-node/core';
import type { ChatStreamEvent } from '@mlx-node/lm';

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
import type { ModelRegistry, ServableModel } from '../registry.js';
import { beginSSE, endSSE, writeSSEEvent } from '../streaming.js';
import { ToolCallTagBuffer } from '../tool-call-buffer.js';
import type { AnthropicMessagesRequest } from '../types-anthropic.js';

// ---------------------------------------------------------------------------
// Non-streaming path
// ---------------------------------------------------------------------------

async function handleNonStreaming(
  res: ServerResponse,
  model: ServableModel,
  messages: ChatMessage[],
  config: ChatConfig,
  body: AnthropicMessagesRequest,
): Promise<void> {
  const result = (await model.chat(messages, config)) as ChatResult;
  const messageId = genId('msg_');
  const response = buildAnthropicResponse(result, body, messageId);
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(response));
}

// ---------------------------------------------------------------------------
// Streaming path -- model supports chatStream()
// ---------------------------------------------------------------------------

async function handleStreamingNative(
  res: ServerResponse,
  model: ServableModel,
  messages: ChatMessage[],
  config: ChatConfig,
  body: AnthropicMessagesRequest,
): Promise<void> {
  const chatStream = (
    model as unknown as {
      chatStream(m: ChatMessage[], c: unknown): AsyncGenerator<ChatStreamEvent>;
    }
  ).chatStream(messages, config);

  if (!chatStream || typeof (chatStream as unknown as Record<symbol, unknown>)[Symbol.asyncIterator] !== 'function') {
    // chatStream did not return an async iterable — fall back to simulated streaming
    return handleStreamingSimulated(res, model, messages, config, body);
  }

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
// Streaming path -- simulated from chat()
// ---------------------------------------------------------------------------

async function handleStreamingSimulated(
  res: ServerResponse,
  model: ServableModel,
  messages: ChatMessage[],
  config: ChatConfig,
  body: AnthropicMessagesRequest,
): Promise<void> {
  const messageId = genId('msg_');

  beginSSE(res);
  writeSSEEvent(res, 'message_start', buildMessageStartEvent(body, messageId, 0));

  const result = (await model.chat(messages, config)) as ChatResult;
  const okToolCalls = result.toolCalls.filter((t) => t.status === 'ok');
  const hasToolCalls = okToolCalls.length > 0;

  let contentBlockIndex = 0;

  // Thinking block
  if (result.thinking) {
    writeSSEEvent(
      res,
      'content_block_start',
      buildContentBlockStart(contentBlockIndex, { type: 'thinking', thinking: '' }),
    );
    writeSSEEvent(
      res,
      'content_block_delta',
      buildContentBlockDelta(contentBlockIndex, { type: 'thinking_delta', thinking: result.thinking }),
    );
    writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex));
    contentBlockIndex++;
  }

  // Text block
  if (result.text || !hasToolCalls) {
    writeSSEEvent(res, 'content_block_start', buildContentBlockStart(contentBlockIndex, { type: 'text', text: '' }));
    writeSSEEvent(
      res,
      'content_block_delta',
      buildContentBlockDelta(contentBlockIndex, { type: 'text_delta', text: result.text }),
    );
    writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex));
    contentBlockIndex++;
  }

  // Tool use blocks
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

  // message_delta + message_stop
  const stopReason = mapStopReason(result.finishReason, hasToolCalls);
  writeSSEEvent(res, 'message_delta', buildMessageDelta(stopReason, result.numTokens, result.promptTokens));
  writeSSEEvent(res, 'message_stop', buildMessageStop());

  endSSE(res);
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

  // Map request
  let messages: ChatMessage[];
  let config: ChatConfig;
  try {
    ({ messages, config } = mapAnthropicRequest(body));
  } catch (err) {
    sendAnthropicBadRequest(res, err instanceof Error ? err.message : 'Invalid request');
    return;
  }

  try {
    if (body.stream === true) {
      if (registry.hasStreamSupport(model)) {
        await handleStreamingNative(res, model, messages, config, body);
      } else {
        await handleStreamingSimulated(res, model, messages, config, body);
      }
    } else {
      await handleNonStreaming(res, model, messages, config, body);
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
