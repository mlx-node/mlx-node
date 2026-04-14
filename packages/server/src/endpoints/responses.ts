/**
 * POST /v1/responses endpoint
 *
 * Implements the OpenAI Responses API, dispatching to loaded models
 * via the ModelRegistry. Supports both streaming (SSE) and non-streaming
 * (JSON) response modes.
 *
 * All inference goes through a per-model `ChatSession` looked up or
 * allocated via the model's `SessionRegistry`. Sessions are keyed by
 * `previous_response_id`: on a cache hit the session's live KV cache
 * is reused via `session.send()` / `sendStream()` / `sendToolResult()`.
 * On a cache miss (no prior response, eviction, or restart) the full
 * conversation is reconstructed from the `ResponseStore`, primed into
 * a fresh session via `primeHistory()`, and replayed through
 * `startFromHistory()` / `startFromHistoryStream()`.
 */

import type { ServerResponse } from 'node:http';

import type { ChatConfig, ChatMessage, ChatResult, ResponseStore, StoredResponseRecord } from '@mlx-node/core';
import type { ChatSession, ChatStreamEvent, SessionCapableModel } from '@mlx-node/lm';

import { sendBadRequest, sendInternalError, sendNotFound } from '../errors.js';
import { mapRequest, reconstructMessagesFromChain } from '../mappers/request.js';
import {
  buildPartialResponse,
  buildResponseObject,
  computeOutputText,
  genId,
  mapFinishReasonToStatus,
} from '../mappers/response.js';
import type { ModelRegistry } from '../registry.js';
import type { SessionRegistry } from '../session-registry.js';
import { beginSSE, endSSE, writeSSEEvent } from '../streaming.js';
import { ToolCallTagBuffer } from '../tool-call-buffer.js';
import type {
  FunctionCallOutputItem,
  MessageOutputItem,
  OutputItem,
  ReasoningOutputItem,
  ResponseObject,
  ResponsesAPIRequest,
} from '../types.js';

/** How long stored responses live (seconds). */
const RESPONSE_TTL_SECONDS = 1800; // 30 minutes

// ---------------------------------------------------------------------------
// Non-streaming path
// ---------------------------------------------------------------------------

async function handleNonStreaming(
  res: ServerResponse,
  result: ChatResult,
  req: ResponsesAPIRequest,
  responseId: string,
  previousResponseId: string | undefined,
  store: ResponseStore | null,
  newInputMessages: ChatMessage[],
): Promise<void> {
  const response = buildResponseObject(result, req, responseId, previousResponseId);

  // Persist only the new input messages (not the full expanded conversation)
  if (store && req.store !== false) {
    await persistResponse(store, response, newInputMessages, previousResponseId);
  }

  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(response));
}

// ---------------------------------------------------------------------------
// Streaming path
// ---------------------------------------------------------------------------

async function handleStreamingNative(
  res: ServerResponse,
  chatStream: AsyncGenerator<ChatStreamEvent>,
  req: ResponsesAPIRequest,
  responseId: string,
  previousResponseId: string | undefined,
  store: ResponseStore | null,
  newInputMessages: ChatMessage[],
): Promise<void> {
  beginSSE(res);

  const partial = buildPartialResponse(req, responseId, previousResponseId);
  writeSSEEvent(res, 'response.created', { response: partial });
  writeSSEEvent(res, 'response.in_progress', { response: partial });

  const outputItems: OutputItem[] = [];
  let outputIndex = 0;

  // State tracking for streaming
  let reasoningItemId: string | null = null;
  let reasoningText = '';
  let messageItemId: string | null = null;
  let messageText = '';
  let hasEmittedMessage = false;
  let hasEmittedReasoning = false;
  let suppressedMessageIndex = -1;
  const tagBuffer = new ToolCallTagBuffer();

  for await (const event of chatStream) {
    if (event.done) {
      // Final event -- close open items and emit completed

      // Flush any remaining pending text (no tool call tag was found)
      const remainingText = tagBuffer.flush();
      if (!tagBuffer.suppressed && remainingText) {
        if (!hasEmittedMessage) {
          hasEmittedMessage = true;
          messageItemId = genId('msg_');
          const messageItem: MessageOutputItem = {
            id: messageItemId,
            type: 'message',
            role: 'assistant',
            status: 'in_progress',
            content: [],
          };
          const miIndex = outputItems.length;
          outputItems.push(messageItem);
          outputIndex = miIndex;
          writeSSEEvent(res, 'response.output_item.added', { output_index: miIndex, item: messageItem });
          const textPart = { type: 'output_text' as const, text: '', annotations: [] as never[] };
          writeSSEEvent(res, 'response.content_part.added', {
            item_id: messageItemId,
            output_index: miIndex,
            content_index: 0,
            part: textPart,
          });
        }
        messageText += remainingText;
        writeSSEEvent(res, 'response.output_text.delta', {
          item_id: messageItemId,
          output_index: outputItems.findIndex((i) => i.id === messageItemId),
          content_index: 0,
          delta: remainingText,
        });
      }

      // Close reasoning item if open
      if (hasEmittedReasoning && reasoningItemId) {
        writeSSEEvent(res, 'response.reasoning_summary_text.done', {
          item_id: reasoningItemId,
          output_index: outputItems.length - (hasEmittedMessage ? 1 : 0) - 1,
          summary_index: 0,
          text: event.thinking ?? reasoningText,
        });
        const reasoningItem: ReasoningOutputItem = {
          id: reasoningItemId,
          type: 'reasoning',
          summary: [{ type: 'summary_text', text: event.thinking ?? reasoningText }],
        };
        const riIndex = outputItems.findIndex((i) => i.id === reasoningItemId);
        if (riIndex >= 0) {
          outputItems[riIndex] = reasoningItem;
        }
        writeSSEEvent(res, 'response.output_item.done', {
          output_index: riIndex >= 0 ? riIndex : 0,
          item: reasoningItem,
        });
      }

      // Close message item if open.
      // Use the final event's parsed text (markup-stripped) as the authoritative content.
      // If the parsed text is empty and there are tool calls, skip the message item entirely
      // (matching the non-streaming buildOutputItems behavior).
      const finalText = event.text;
      const hasToolCalls = event.toolCalls.some((t) => t.status === 'ok');
      const skipMessageItem = !finalText && hasToolCalls;

      // Recovery: if tool-call suppression was triggered but the final event has no
      // parsed tool calls (false alarm — e.g., literal "<tool_call>" in model output),
      // create a message item using the final parsed text.
      if (tagBuffer.suppressed && !hasToolCalls && finalText && !hasEmittedMessage) {
        hasEmittedMessage = true;
        messageItemId = genId('msg_');
        const messageItem: MessageOutputItem = {
          id: messageItemId,
          type: 'message',
          role: 'assistant',
          status: 'in_progress',
          content: [],
        };
        const miIndex = outputItems.length;
        outputItems.push(messageItem);
        outputIndex = miIndex;
        writeSSEEvent(res, 'response.output_item.added', { output_index: miIndex, item: messageItem });
        const textPart = { type: 'output_text' as const, text: '', annotations: [] as never[] };
        writeSSEEvent(res, 'response.content_part.added', {
          item_id: messageItemId,
          output_index: miIndex,
          content_index: 0,
          part: textPart,
        });
        messageText = finalText;
        writeSSEEvent(res, 'response.output_text.delta', {
          item_id: messageItemId,
          output_index: miIndex,
          content_index: 0,
          delta: finalText,
        });
      } else if (tagBuffer.suppressed && !hasToolCalls && finalText && hasEmittedMessage) {
        // Recovery: text was already being streamed but got cut off by a false-alarm
        // <tool_call> tag. Emit the unsent portion as a delta.
        const unsent = finalText.slice(messageText.length);
        if (unsent) {
          messageText += unsent;
          writeSSEEvent(res, 'response.output_text.delta', {
            item_id: messageItemId,
            output_index: outputItems.findIndex((i) => i.id === messageItemId),
            content_index: 0,
            delta: unsent,
          });
        }
      }

      // Emit any unsent suffix when final text is longer than what was streamed
      if (hasEmittedMessage && finalText && finalText.length > messageText.length && !tagBuffer.suppressed) {
        const unsent = finalText.slice(messageText.length);
        messageText += unsent;
        writeSSEEvent(res, 'response.output_text.delta', {
          item_id: messageItemId,
          output_index: outputItems.findIndex((i) => i.id === messageItemId),
          content_index: 0,
          delta: unsent,
        });
      }

      // Recovery: text was never emitted during streaming but final has text
      // (possible if all text arrived in the final event only)
      if (!hasEmittedMessage && finalText && !skipMessageItem) {
        hasEmittedMessage = true;
        messageItemId = genId('msg_');
        const messageItem: MessageOutputItem = {
          id: messageItemId,
          type: 'message',
          role: 'assistant',
          status: 'in_progress',
          content: [],
        };
        const miIndex = outputItems.length;
        outputItems.push(messageItem);
        outputIndex = miIndex;
        writeSSEEvent(res, 'response.output_item.added', { output_index: miIndex, item: messageItem });
        const textPart = { type: 'output_text' as const, text: '', annotations: [] as never[] };
        writeSSEEvent(res, 'response.content_part.added', {
          item_id: messageItemId,
          output_index: miIndex,
          content_index: 0,
          part: textPart,
        });
        messageText = finalText;
        writeSSEEvent(res, 'response.output_text.delta', {
          item_id: messageItemId,
          output_index: miIndex,
          content_index: 0,
          delta: finalText,
        });
      }

      if (hasEmittedMessage && messageItemId && !skipMessageItem) {
        const miIndex = outputItems.findIndex((i) => i.id === messageItemId);
        const contentIndex = 0;

        writeSSEEvent(res, 'response.output_text.done', {
          item_id: messageItemId,
          output_index: miIndex >= 0 ? miIndex : outputIndex,
          content_index: contentIndex,
          text: finalText,
        });

        const textPart = { type: 'output_text' as const, text: finalText, annotations: [] as never[] };
        writeSSEEvent(res, 'response.content_part.done', {
          item_id: messageItemId,
          output_index: miIndex >= 0 ? miIndex : outputIndex,
          content_index: contentIndex,
          part: textPart,
        });

        const messageItem: MessageOutputItem = {
          id: messageItemId,
          type: 'message',
          role: 'assistant',
          status: mapFinishReasonToStatus(event.finishReason),
          content: [textPart],
        };
        if (miIndex >= 0) {
          outputItems[miIndex] = messageItem;
        }
        writeSSEEvent(res, 'response.output_item.done', {
          output_index: miIndex >= 0 ? miIndex : outputIndex,
          item: messageItem,
        });
      } else if (hasEmittedMessage && messageItemId && skipMessageItem) {
        // A message item was started (output_item.added / content_part.added events already
        // sent to the client) but we now know it should be suppressed because the final
        // text is empty and there are tool calls.  Send proper done events to close out
        // the item gracefully so clients do not see a dangling in-progress item, then
        // remove it from outputItems so it does not appear in the completed response.
        const miIndex = outputItems.findIndex((i) => i.id === messageItemId);
        const miOutputIndex = miIndex >= 0 ? miIndex : outputIndex;

        writeSSEEvent(res, 'response.output_text.done', {
          item_id: messageItemId,
          output_index: miOutputIndex,
          content_index: 0,
          text: '',
        });

        const emptyTextPart = { type: 'output_text' as const, text: '', annotations: [] as never[] };
        writeSSEEvent(res, 'response.content_part.done', {
          item_id: messageItemId,
          output_index: miOutputIndex,
          content_index: 0,
          part: emptyTextPart,
        });

        const closedMessageItem: MessageOutputItem = {
          id: messageItemId,
          type: 'message',
          role: 'assistant',
          status: 'completed',
          content: [],
        };
        writeSSEEvent(res, 'response.output_item.done', {
          output_index: miOutputIndex,
          item: closedMessageItem,
        });

        // Track suppressed index for exclusion from final response
        // but keep in array so subsequent output_index values remain unique.
        if (miIndex >= 0) {
          suppressedMessageIndex = miIndex;
        }
      }

      // Emit function call items
      for (const tc of event.toolCalls.filter((t) => t.status === 'ok')) {
        const callId = tc.id ?? genId('call_');
        const fcItem: FunctionCallOutputItem = {
          id: genId('fc_'),
          type: 'function_call',
          call_id: callId,
          name: tc.name,
          arguments: typeof tc.arguments === 'string' ? tc.arguments : JSON.stringify(tc.arguments),
          status: 'completed',
        };
        const fcIndex = outputItems.length;
        outputItems.push(fcItem);

        writeSSEEvent(res, 'response.output_item.added', { output_index: fcIndex, item: fcItem });

        const argsStr = fcItem.arguments;
        writeSSEEvent(res, 'response.function_call_arguments.delta', {
          item_id: fcItem.id,
          output_index: fcIndex,
          delta: argsStr,
        });
        writeSSEEvent(res, 'response.function_call_arguments.done', {
          item_id: fcItem.id,
          output_index: fcIndex,
          arguments: argsStr,
        });

        writeSSEEvent(res, 'response.output_item.done', { output_index: fcIndex, item: fcItem });
      }

      // Build completed response
      const promptTokens = event.promptTokens ?? 0;
      const reasoningTokens = event.reasoningTokens ?? 0;
      const usage = {
        input_tokens: promptTokens,
        output_tokens: event.numTokens,
        output_tokens_details: { reasoning_tokens: reasoningTokens },
        total_tokens: promptTokens + event.numTokens,
      };

      const finalOutput = outputItems.filter((_, idx) => idx !== suppressedMessageIndex);
      const completedResponse: ResponseObject = {
        ...partial,
        status: mapFinishReasonToStatus(event.finishReason),
        output: finalOutput,
        output_text: computeOutputText(finalOutput),
        incomplete_details: event.finishReason === 'length' ? { reason: 'max_output_tokens' } : null,
        usage,
      };

      // Persist only the new input messages
      if (store && req.store !== false) {
        await persistResponse(store, completedResponse, newInputMessages, previousResponseId);
      }

      writeSSEEvent(res, 'response.completed', { response: completedResponse });

      endSSE(res);
      return;
    }

    // Delta event
    if (event.isReasoning) {
      // Filter out </think> tag from reasoning deltas
      const deltaText = event.text.replace(/<\/think>/g, '');
      if (!deltaText) continue; // Skip empty deltas (e.g., just the </think> token)

      if (!hasEmittedReasoning) {
        // First reasoning chunk -- add reasoning item
        hasEmittedReasoning = true;
        reasoningItemId = genId('rs_');
        const reasoningItem: ReasoningOutputItem = {
          id: reasoningItemId,
          type: 'reasoning',
          summary: [],
        };
        const riIndex = outputItems.length;
        outputItems.push(reasoningItem);

        writeSSEEvent(res, 'response.output_item.added', { output_index: riIndex, item: reasoningItem });
      }
      reasoningText += deltaText;
      writeSSEEvent(res, 'response.reasoning_summary_text.delta', {
        item_id: reasoningItemId,
        output_index: outputItems.findIndex((i) => i.id === reasoningItemId),
        summary_index: 0,
        delta: deltaText,
      });
    } else {
      // Text delta with tool_call tag buffering
      const { safeText, tagFound, cleanPrefix } = tagBuffer.push(event.text);
      if (tagFound) {
        // Emit any clean text before the tag.
        // Trim whitespace-only prefixes: whitespace immediately before <tool_call>
        // is always markup-related (e.g. "\n<tool_call>"), not user-visible content.
        // Emitting it would create a dangling message item that needs special-casing
        // at finalization when skipMessageItem is true.
        if (cleanPrefix.trim()) {
          if (!hasEmittedMessage) {
            hasEmittedMessage = true;
            messageItemId = genId('msg_');
            const messageItem: MessageOutputItem = {
              id: messageItemId,
              type: 'message',
              role: 'assistant',
              status: 'in_progress',
              content: [],
            };
            const miIndex = outputItems.length;
            outputItems.push(messageItem);
            outputIndex = miIndex;
            writeSSEEvent(res, 'response.output_item.added', { output_index: miIndex, item: messageItem });
            const textPart = { type: 'output_text' as const, text: '', annotations: [] as never[] };
            writeSSEEvent(res, 'response.content_part.added', {
              item_id: messageItemId,
              output_index: miIndex,
              content_index: 0,
              part: textPart,
            });
          }
          messageText += cleanPrefix;
          writeSSEEvent(res, 'response.output_text.delta', {
            item_id: messageItemId,
            output_index: outputItems.findIndex((i) => i.id === messageItemId),
            content_index: 0,
            delta: cleanPrefix,
          });
        }
      } else if (safeText) {
        if (!hasEmittedMessage) {
          hasEmittedMessage = true;
          messageItemId = genId('msg_');
          const messageItem: MessageOutputItem = {
            id: messageItemId,
            type: 'message',
            role: 'assistant',
            status: 'in_progress',
            content: [],
          };
          const miIndex = outputItems.length;
          outputItems.push(messageItem);
          outputIndex = miIndex;
          writeSSEEvent(res, 'response.output_item.added', { output_index: miIndex, item: messageItem });
          const textPart = { type: 'output_text' as const, text: '', annotations: [] as never[] };
          writeSSEEvent(res, 'response.content_part.added', {
            item_id: messageItemId,
            output_index: miIndex,
            content_index: 0,
            part: textPart,
          });
        }
        messageText += safeText;
        writeSSEEvent(res, 'response.output_text.delta', {
          item_id: messageItemId,
          output_index: outputItems.findIndex((i) => i.id === messageItemId),
          content_index: 0,
          delta: safeText,
        });
      }
    }
  }

  // Safety net: if the async iterator exhausted without a done event,
  // emit a completed response with whatever partial state we have so
  // clients and previous_response_id chaining don't see a dangling stream.
  const fallbackOutput = outputItems.filter((_, idx) => idx !== suppressedMessageIndex);
  const fallbackResponse: ResponseObject = {
    ...partial,
    status: 'incomplete',
    output: fallbackOutput,
    output_text: computeOutputText(fallbackOutput),
    incomplete_details: { reason: 'max_output_tokens' },
    usage: { input_tokens: 0, output_tokens: 0, output_tokens_details: { reasoning_tokens: 0 }, total_tokens: 0 },
  };

  if (store && req.store !== false) {
    await persistResponse(store, fallbackResponse, newInputMessages, previousResponseId);
  }

  writeSSEEvent(res, 'response.completed', { response: fallbackResponse });
  endSSE(res);
}

// ---------------------------------------------------------------------------
// Session routing
// ---------------------------------------------------------------------------

/**
 * Route a non-streaming request through a `ChatSession`.
 *
 * Cold path (fresh session): prime with the full mapped history and
 * run `startFromHistory`. Hot path (cached session with a live KV
 * cache): send only the last new input message via `send` or
 * `sendToolResult`. Multi-message hot-path requests fall back to a
 * reset + cold re-prime.
 */
async function runSessionNonStreaming(
  session: ChatSession<SessionCapableModel>,
  messages: ChatMessage[],
  newInputMessages: ChatMessage[],
  config: ChatConfig,
): Promise<ChatResult> {
  if (session.turns === 0) {
    session.primeHistory(messages);
    return await session.startFromHistory(config);
  }

  // Hot path — session's KV cache is already warmed for this chain.
  if (newInputMessages.length === 1) {
    const last = newInputMessages[0]!;
    if (last.role === 'user') {
      const images = last.images ?? undefined;
      return await session.send(last.content, images ? { images, config } : { config });
    }
    if (last.role === 'tool') {
      if (!last.toolCallId) {
        throw new Error('tool message missing toolCallId');
      }
      return await session.sendToolResult(last.toolCallId, last.content, { config });
    }
    throw new Error(`unsupported last message role on hot path: ${last.role}`);
  }

  // Multi-message hot-path input: drop the cached session state and
  // re-run as a cold path. Correct but pays the full prefill cost.
  await session.reset();
  session.primeHistory(messages);
  return await session.startFromHistory(config);
}

/**
 * Streaming counterpart to {@link runSessionNonStreaming}. Returns the
 * session's underlying async generator so the SSE loop can consume it
 * directly.
 */
async function runSessionStreaming(
  session: ChatSession<SessionCapableModel>,
  messages: ChatMessage[],
  newInputMessages: ChatMessage[],
  config: ChatConfig,
): Promise<AsyncGenerator<ChatStreamEvent>> {
  if (session.turns === 0) {
    session.primeHistory(messages);
    return session.startFromHistoryStream(config);
  }

  if (newInputMessages.length === 1) {
    const last = newInputMessages[0]!;
    if (last.role === 'user') {
      const images = last.images ?? undefined;
      return session.sendStream(last.content, images ? { images, config } : { config });
    }
    if (last.role === 'tool') {
      if (!last.toolCallId) {
        throw new Error('tool message missing toolCallId');
      }
      return session.sendToolResultStream(last.toolCallId, last.content, { config });
    }
    throw new Error(`unsupported last message role on hot path: ${last.role}`);
  }

  await session.reset();
  session.primeHistory(messages);
  return session.startFromHistoryStream(config);
}

// ---------------------------------------------------------------------------
// Storage helper
// ---------------------------------------------------------------------------

async function persistResponse(
  store: ResponseStore,
  response: ResponseObject,
  newInputMessages: ChatMessage[],
  previousResponseId: string | undefined,
): Promise<void> {
  // Store only the NEW input messages from this request, not the full
  // expanded conversation. Chain reconstruction re-derives the full history
  // by following previous_response_id links.
  const record: StoredResponseRecord = {
    id: response.id,
    createdAt: response.created_at,
    model: response.model,
    status: response.status,
    instructions: response.instructions ?? undefined,
    inputJson: JSON.stringify(newInputMessages),
    outputJson: JSON.stringify(response.output),
    outputText: response.output_text,
    usageJson: JSON.stringify(response.usage),
    previousResponseId: previousResponseId ?? undefined,
    configJson: JSON.stringify({
      temperature: response.temperature,
      top_p: response.top_p,
      max_output_tokens: response.max_output_tokens,
      tools: response.tools,
      reasoning: response.reasoning,
    }),
    expiresAt: Math.floor(Date.now() / 1000) + RESPONSE_TTL_SECONDS,
  };
  await store.store(record);
}

// ---------------------------------------------------------------------------
// Public handler
// ---------------------------------------------------------------------------

export async function handleCreateResponse(
  res: ServerResponse,
  body: ResponsesAPIRequest,
  registry: ModelRegistry,
  store: ResponseStore | null,
): Promise<void> {
  // Validate required fields
  if (body == null || typeof body !== 'object') {
    sendBadRequest(res, 'Request body must be a JSON object', 'body');
    return;
  }
  if (!body.model) {
    sendBadRequest(res, 'Missing required field: model', 'model');
    return;
  }
  if (body.input == null) {
    sendBadRequest(res, 'Missing required field: input', 'input');
    return;
  }
  if (typeof body.input !== 'string' && !Array.isArray(body.input)) {
    sendBadRequest(res, 'Field "input" must be a string or an array', 'input');
    return;
  }

  // Look up model
  const model = registry.get(body.model);
  if (!model) {
    sendNotFound(
      res,
      `Model "${body.model}" not found. Available models: ${registry
        .list()
        .map((m) => m.id)
        .join(', ')}`,
    );
    return;
  }

  // Fetch the per-model session registry. `model` was just resolved
  // from the same `ModelRegistry`, so this should always succeed — but
  // guard anyway to surface a clear error rather than crashing.
  const sessionReg: SessionRegistry | undefined = registry.getSessionRegistry(body.model);
  if (!sessionReg) {
    sendInternalError(res, 'session registry missing for registered model');
    return;
  }

  const responseId = genId('resp_');

  // Resolve previous_response_id chain
  let priorMessages: ChatMessage[] | undefined;
  let previousResponseId: string | undefined;

  if (body.previous_response_id && store) {
    try {
      const chain = await store.getChain(body.previous_response_id);
      if (chain.length === 0) {
        sendNotFound(res, `Previous response "${body.previous_response_id}" not found`);
        return;
      }
      priorMessages = reconstructMessagesFromChain(chain);
      previousResponseId = body.previous_response_id;
    } catch (err) {
      const msg = err instanceof Error ? err.message : '';
      if (/not found/i.test(msg)) {
        sendNotFound(res, `Previous response "${body.previous_response_id}" not found or expired`);
      } else {
        sendInternalError(res, `Failed to retrieve previous response: ${msg || 'unknown error'}`);
      }
      return;
    }
  } else if (body.previous_response_id && !store) {
    sendBadRequest(res, 'previous_response_id requires a response store to be configured');
    return;
  }

  // Map request — full messages include prior + new input
  let messages: ChatMessage[];
  let config: ChatConfig;
  try {
    ({ messages, config } = mapRequest(body, priorMessages));
  } catch (err) {
    sendBadRequest(res, err instanceof Error ? err.message : 'Invalid request input', 'input');
    return;
  }

  // Compute the new-only messages (what this request added, excluding prior history
  // and instructions). Instructions are stored separately and should not be persisted
  // as input messages — otherwise chained calls replay stale system messages.
  const instructionsOffset = body.instructions ? 1 : 0;
  const priorOffset = instructionsOffset + (priorMessages?.length ?? 0);
  const newInputMessages = messages.slice(priorOffset);

  // Route the request through a `ChatSession` looked up by the prior
  // response id. A miss returns a fresh session; the hot path reuses
  // an already-warmed KV cache when the caller passes the id we
  // allocated on the previous turn.
  const session = sessionReg.getOrCreate(previousResponseId ?? null);

  try {
    if (body.stream) {
      const chatStream = await runSessionStreaming(session, messages, newInputMessages, config);
      await handleStreamingNative(res, chatStream, body, responseId, previousResponseId, store, newInputMessages);
    } else {
      const result = await runSessionNonStreaming(session, messages, newInputMessages, config);
      await handleNonStreaming(res, result, body, responseId, previousResponseId, store, newInputMessages);
    }

    // Success — re-key the (possibly just-primed) session under the
    // newly allocated response id. Drop the prior id so the cache
    // doesn't hold two entries for the same logical thread.
    if (previousResponseId) {
      sessionReg.drop(previousResponseId);
    }
    sessionReg.adopt(responseId, session);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error during inference';
    // If headers haven't been sent yet, send a proper error response
    if (!res.headersSent) {
      sendInternalError(res, message);
    } else {
      // Headers already sent (streaming) -- best effort: write error event and close
      writeSSEEvent(res, 'error', { error_type: 'server_error', message });
      endSSE(res);
    }
  }
}
