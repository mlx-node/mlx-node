/**
 * POST /v1/responses endpoint
 *
 * Implements the OpenAI Responses API, dispatching to loaded models
 * via the ModelRegistry. Supports both streaming (SSE) and non-streaming
 * (JSON) response modes.
 */

import type { ServerResponse } from 'node:http';

import type { ChatConfig, ChatMessage, ChatResult, ResponseStore, StoredResponseRecord } from '@mlx-node/core';
import type { ChatStreamEvent } from '@mlx-node/lm';

import { sendBadRequest, sendInternalError, sendNotFound } from '../errors.js';
import { mapRequest, reconstructMessagesFromChain } from '../mappers/request.js';
import {
  buildOutputItems,
  buildPartialResponse,
  buildResponseObject,
  computeOutputText,
  genId,
  mapFinishReasonToStatus,
} from '../mappers/response.js';
import type { ModelRegistry, ServableModel } from '../registry.js';
import { beginSSE, endSSE, writeSSEEvent } from '../streaming.js';
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
  model: ServableModel,
  messages: ChatMessage[],
  config: ChatConfig,
  req: ResponsesAPIRequest,
  responseId: string,
  previousResponseId: string | undefined,
  store: ResponseStore | null,
  newInputMessages: ChatMessage[],
): Promise<void> {
  const result = (await model.chat(messages, config)) as ChatResult;
  const response = buildResponseObject(result, req, responseId, previousResponseId);

  // Persist only the new input messages (not the full expanded conversation)
  if (store && req.store !== false) {
    await persistResponse(store, response, newInputMessages, previousResponseId);
  }

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

  const TOOL_CALL_TAG = '<tool_call>';

  // State tracking for streaming
  let reasoningItemId: string | null = null;
  let reasoningText = '';
  let messageItemId: string | null = null;
  let messageText = '';
  let pendingText = '';
  let hasEmittedMessage = false;
  let hasEmittedReasoning = false;
  let suppressTextDeltas = false;

  const chatStream = (
    model as unknown as {
      chatStream(m: ChatMessage[], c: unknown): AsyncGenerator<ChatStreamEvent>;
    }
  ).chatStream(messages, config);

  for await (const event of chatStream) {
    if (event.done) {
      // Final event -- close open items and emit completed

      // Flush any remaining pending text (no tool call tag was found)
      if (!suppressTextDeltas && pendingText) {
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
        messageText += pendingText;
        writeSSEEvent(res, 'response.output_text.delta', {
          item_id: messageItemId,
          output_index: outputItems.findIndex((i) => i.id === messageItemId),
          content_index: 0,
          delta: pendingText,
        });
        pendingText = '';
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
      const hasToolCalls = event.toolCalls.length > 0;
      const skipMessageItem = !finalText && hasToolCalls;

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
        // Remove the message placeholder from outputItems since we're skipping it
        const miIndex = outputItems.findIndex((i) => i.id === messageItemId);
        if (miIndex >= 0) {
          outputItems.splice(miIndex, 1);
        }
      }

      // Emit function call items
      for (const tc of event.toolCalls) {
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

      const completedResponse: ResponseObject = {
        ...partial,
        status: mapFinishReasonToStatus(event.finishReason),
        output: outputItems,
        output_text: computeOutputText(outputItems),
        incomplete_details: event.finishReason === 'length' ? { reason: 'max_output_tokens' } : null,
        usage,
      };

      writeSSEEvent(res, 'response.completed', { response: completedResponse });

      // Persist only the new input messages
      if (store && req.store !== false) {
        await persistResponse(store, completedResponse, newInputMessages, previousResponseId);
      }

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
      if (!suppressTextDeltas) {
        pendingText += event.text;

        // Check for full tool-call tag
        const tagIdx = pendingText.indexOf(TOOL_CALL_TAG);
        if (tagIdx >= 0) {
          // Emit any clean text before the tag
          const cleanPrefix = pendingText.slice(0, tagIdx);
          if (cleanPrefix) {
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
          suppressTextDeltas = true;
          pendingText = '';
        } else {
          // Check if pendingText ends with a partial prefix of '<tool_call>'
          let safeLen = pendingText.length;
          for (let i = 1; i <= Math.min(pendingText.length, TOOL_CALL_TAG.length - 1); i++) {
            const suffix = pendingText.slice(-i);
            if (TOOL_CALL_TAG.startsWith(suffix)) {
              safeLen = pendingText.length - i;
              break;
            }
          }

          const safeText = pendingText.slice(0, safeLen);
          pendingText = pendingText.slice(safeLen);

          if (safeText) {
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
    }
  }
}

// ---------------------------------------------------------------------------
// Streaming path -- model does NOT support chatStream() (simulate from chat)
// ---------------------------------------------------------------------------

async function handleStreamingSimulated(
  res: ServerResponse,
  model: ServableModel,
  messages: ChatMessage[],
  config: ChatConfig,
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

  // Run chat() to completion
  const result = (await model.chat(messages, config)) as ChatResult;
  const outputItems = buildOutputItems(result);

  let outputIndex = 0;

  for (const item of outputItems) {
    const idx = outputIndex++;

    if (item.type === 'reasoning') {
      writeSSEEvent(res, 'response.output_item.added', { output_index: idx, item });
      for (const s of item.summary) {
        writeSSEEvent(res, 'response.reasoning_summary_text.delta', {
          item_id: item.id,
          output_index: idx,
          summary_index: 0,
          delta: s.text,
        });
        writeSSEEvent(res, 'response.reasoning_summary_text.done', {
          item_id: item.id,
          output_index: idx,
          summary_index: 0,
          text: s.text,
        });
      }
      writeSSEEvent(res, 'response.output_item.done', { output_index: idx, item });
    } else if (item.type === 'message') {
      writeSSEEvent(res, 'response.output_item.added', { output_index: idx, item: { ...item, content: [] } });
      for (let ci = 0; ci < item.content.length; ci++) {
        const part = item.content[ci];
        writeSSEEvent(res, 'response.content_part.added', {
          item_id: item.id,
          output_index: idx,
          content_index: ci,
          part: { type: 'output_text', text: '', annotations: [] },
        });
        writeSSEEvent(res, 'response.output_text.delta', {
          item_id: item.id,
          output_index: idx,
          content_index: ci,
          delta: part.text,
        });
        writeSSEEvent(res, 'response.output_text.done', {
          item_id: item.id,
          output_index: idx,
          content_index: ci,
          text: part.text,
        });
        writeSSEEvent(res, 'response.content_part.done', {
          item_id: item.id,
          output_index: idx,
          content_index: ci,
          part,
        });
      }
      writeSSEEvent(res, 'response.output_item.done', { output_index: idx, item });
    } else if (item.type === 'function_call') {
      writeSSEEvent(res, 'response.output_item.added', { output_index: idx, item });
      writeSSEEvent(res, 'response.function_call_arguments.delta', {
        item_id: item.id,
        output_index: idx,
        delta: item.arguments,
      });
      writeSSEEvent(res, 'response.function_call_arguments.done', {
        item_id: item.id,
        output_index: idx,
        arguments: item.arguments,
      });
      writeSSEEvent(res, 'response.output_item.done', { output_index: idx, item });
    }
  }

  // Completed response
  const response = buildResponseObject(result, req, responseId, previousResponseId);
  writeSSEEvent(res, 'response.completed', { response });

  // Persist only the new input messages
  if (store && req.store !== false) {
    await persistResponse(store, response, newInputMessages, previousResponseId);
  }

  endSSE(res);
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
  if (!body.model) {
    sendBadRequest(res, 'Missing required field: model', 'model');
    return;
  }
  if (body.input == null) {
    sendBadRequest(res, 'Missing required field: input', 'input');
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
    } catch {
      sendNotFound(res, `Previous response "${body.previous_response_id}" not found or expired`);
      return;
    }
  } else if (body.previous_response_id && !store) {
    sendBadRequest(res, 'previous_response_id requires a response store to be configured');
    return;
  }

  // Map request — full messages include prior + new input
  const { messages, config } = mapRequest(body, priorMessages);

  // Compute the new-only messages (what this request added, excluding prior history).
  // Instructions (if any) are placed first, then prior messages, then new input.
  const priorOffset = (priorMessages?.length ?? 0) + (body.instructions ? 1 : 0);
  const newInputMessages = priorMessages ? messages.slice(priorOffset) : messages;

  try {
    if (body.stream) {
      if (registry.hasStreamSupport(model)) {
        await handleStreamingNative(
          res,
          model,
          messages,
          config,
          body,
          responseId,
          previousResponseId,
          store,
          newInputMessages,
        );
      } else {
        await handleStreamingSimulated(
          res,
          model,
          messages,
          config,
          body,
          responseId,
          previousResponseId,
          store,
          newInputMessages,
        );
      }
    } else {
      await handleNonStreaming(
        res,
        model,
        messages,
        config,
        body,
        responseId,
        previousResponseId,
        store,
        newInputMessages,
      );
    }
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
