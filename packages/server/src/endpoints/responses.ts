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
 * Walk a mapped message list backward to the most recent assistant
 * turn and, when that turn fanned out to more than one named tool
 * call, return the array of sibling call ids. Returns `null`
 * otherwise. The caller uses this set as the authoritative "pending
 * outstanding tool calls" to validate a submitted continuation
 * against — comparing exact ids instead of just counts catches
 * duplicate / wrong / partial replays that would otherwise satisfy
 * a count-only check.
 *
 * Should be invoked on the STORED prior chain (via
 * `reconstructMessagesFromChain`) when available, never on the
 * already-augmented `messages` list — otherwise a caller that
 * echoes `function_call` items in the new input could overwrite the
 * trailing assistant with a forged single-call turn and slip past
 * the guard.
 */
function extractOutstandingToolCallIds(messages: ChatMessage[]): string[] | null {
  let lastAssistantWithCallsIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg?.role === 'assistant') {
      const tcs = msg.toolCalls ?? [];
      if (tcs.length > 0) {
        lastAssistantWithCallsIdx = i;
      }
      break;
    }
  }
  if (lastAssistantWithCallsIdx === -1) {
    return null;
  }
  const trailingAssistant = messages[lastAssistantWithCallsIdx]!;
  const orderedIds: string[] = [];
  for (const tc of trailingAssistant.toolCalls ?? []) {
    if (typeof tc.id === 'string' && tc.id.length > 0) {
      orderedIds.push(tc.id);
    }
  }
  if (orderedIds.length === 0) {
    return null;
  }
  const outstanding = new Set(orderedIds);
  for (let j = lastAssistantWithCallsIdx + 1; j < messages.length; j++) {
    const m = messages[j];
    if (m?.role === 'tool' && typeof m.toolCallId === 'string' && m.toolCallId.length > 0) {
      outstanding.delete(m.toolCallId);
    }
  }
  if (outstanding.size === 0) {
    return null;
  }
  return orderedIds.filter((id) => outstanding.has(id));
}

/**
 * Build a set of `call_id`s owned by the trailing assistant turn's
 * tool calls. Used to authenticate echoed `function_call` items in a
 * `previous_response_id` continuation against the stored authoritative
 * state: a client that round-trips `response.output` into the next
 * request will re-send its tool calls verbatim, and the server needs
 * to distinguish that legitimate shape from a forgery attempt.
 *
 * Ownership check only — `name` and `arguments` are NOT compared
 * against the stored payload. A client that parses and reserializes
 * its own prior arguments (different JSON whitespace, key order,
 * number formatting) would otherwise fail continuation even though
 * the server never consumes the echoed payload. Any `call_id` absent
 * from the returned set is still rejected as an unambiguous forgery
 * attempt by the caller.
 *
 * Returns `null` when the trailing message is not an assistant turn
 * with any tool calls — callers treat `null` the same as "no echoed
 * function_call allowed" because there is no stored call to own the
 * echo.
 */
function buildTrailingAssistantToolCallIds(messages: ChatMessage[]): Set<string> | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg?.role === 'assistant') {
      const ids = new Set<string>();
      for (const tc of msg.toolCalls ?? []) {
        if (typeof tc.id === 'string' && tc.id.length > 0) {
          ids.add(tc.id);
        }
      }
      return ids.size > 0 ? ids : null;
    }
  }
  return null;
}

/**
 * Reorder the tool messages in `messages` starting at `startOffset`
 * so their relative positions match `expectedOrder`.
 *
 * Replay correctness for a multi-tool-call fan-out depends on POSITION,
 * not `tool_call_id` — several backends drop the id on the wire and
 * pair tool responses to the trailing assistant calls by sibling index.
 * A caller that submits `function_call_output` items in the wrong order
 * would therefore silently bind results to the wrong calls even after
 * the id-set gate passes. This helper canonicalizes the submitted
 * ordering to the stored sibling order before the replay runs.
 *
 * Assumes the call-id SET has already been validated against
 * `expectedOrder`; this helper is a no-op when any precondition does
 * not hold (missing id, count mismatch, etc.) so callers are safe to
 * invoke it unconditionally after the gate passes.
 */
function canonicalizeToolMessageOrder(
  messages: ChatMessage[],
  startOffset: number,
  expectedOrder: readonly string[],
): void {
  const toolPositions: number[] = [];
  const byId = new Map<string, ChatMessage>();
  for (let i = startOffset; i < messages.length; i++) {
    const m = messages[i]!;
    if (m.role === 'tool' && typeof m.toolCallId === 'string' && m.toolCallId.length > 0) {
      toolPositions.push(i);
      byId.set(m.toolCallId, m);
    }
  }
  if (toolPositions.length !== expectedOrder.length) return;
  for (const id of expectedOrder) {
    if (!byId.has(id)) return;
  }
  let alreadyOrdered = true;
  for (let k = 0; k < toolPositions.length; k++) {
    if (messages[toolPositions[k]!]!.toolCallId !== expectedOrder[k]) {
      alreadyOrdered = false;
      break;
    }
  }
  if (alreadyOrdered) return;
  for (let k = 0; k < toolPositions.length; k++) {
    messages[toolPositions[k]!] = byId.get(expectedOrder[k]!)!;
  }
}

/**
 * Outcome of a non-streaming session dispatch. `committed` is the
 * honest "did the session actually advance" signal, accounting for
 * any internal `session.reset()` the helper may have performed
 * before dispatch.
 */
interface NonStreamingOutcome {
  result: ChatResult;
  /**
   * `true` if the session's turn counter advanced past its
   * post-helper-reset baseline. The endpoint uses this to decide
   * whether to adopt the session under the freshly allocated
   * response id — uncommitted dispatches must NOT be adopted
   * because their in-memory KV state is out of sync with whatever
   * the endpoint layer persists.
   */
  committed: boolean;
}

/**
 * Outcome of a streaming session dispatch. `wasCommitted()` is a
 * closure that reports the commit signal AFTER the stream has been
 * consumed by the SSE writer — it compares `session.turns` against
 * the baseline the helper captured AFTER any internal
 * `session.reset()`, so the signal is honest regardless of which
 * dispatch path ran.
 */
interface StreamingOutcome {
  stream: AsyncGenerator<ChatStreamEvent>;
  wasCommitted(): boolean;
}

/**
 * Route a non-streaming request through a `ChatSession`.
 *
 * Cold path (fresh session): prime with the full mapped history and
 * run `startFromHistory`. Hot path (cached session with a live KV
 * cache): send only the last new input message via `send` or
 * `sendToolResult`. Multi-message hot-path requests fall back to a
 * reset + cold re-prime.
 *
 * The caller is responsible for rejecting partial tool-result
 * submissions against a session whose prior assistant turn fanned
 * out to multiple tool calls — see `handleCreateResponse` for the
 * `pendingUnresolvedToolCallCount` gate that guards against this.
 *
 * Returns an explicit `{ result, committed }` so the endpoint's
 * `sessionReg.adopt()` step can honor `ChatSession`'s commit
 * semantics even across the multi-message reset-and-restart branch
 * (where a pre-helper snapshot of `session.turns` would be stale).
 */
async function runSessionNonStreaming(
  session: ChatSession<SessionCapableModel>,
  messages: ChatMessage[],
  newInputMessages: ChatMessage[],
  config: ChatConfig,
): Promise<NonStreamingOutcome> {
  if (session.turns === 0) {
    session.primeHistory(messages);
    const initialTurns = session.turns;
    const result = await session.startFromHistory(config);
    return { result, committed: session.turns > initialTurns };
  }

  // Hot path — session's KV cache is already warmed for this chain.
  if (newInputMessages.length === 1) {
    const last = newInputMessages[0]!;
    const initialTurns = session.turns;
    if (last.role === 'user') {
      const images = last.images ?? undefined;
      const result = await session.send(last.content, images ? { images, config } : { config });
      return { result, committed: session.turns > initialTurns };
    }
    if (last.role === 'tool') {
      if (!last.toolCallId) {
        throw new Error('tool message missing toolCallId');
      }
      const result = await session.sendToolResult(last.toolCallId, last.content, { config });
      return { result, committed: session.turns > initialTurns };
    }
    throw new Error(`unsupported last message role on hot path: ${last.role}`);
  }

  // Multi-message hot-path input: drop the cached session state and
  // re-run as a cold path. Correct but pays the full prefill cost.
  // The caller re-keys this session under the newly allocated response
  // id on success, so subsequent turns will resume from the cache that
  // `startFromHistory` just warmed — the reset is amortized.
  //
  // NOTE: the commit baseline MUST be captured AFTER `session.reset()`
  // (which zeroes `turns`), otherwise a pre-reset snapshot — taken e.g.
  // by the endpoint before calling this helper — would read as
  // "post > pre" only if the old turn count happened to be zero.
  await session.reset();
  session.primeHistory(messages);
  const initialTurns = session.turns;
  const result = await session.startFromHistory(config);
  return { result, committed: session.turns > initialTurns };
}

/**
 * Streaming counterpart to {@link runSessionNonStreaming}. Returns
 * the session's underlying async generator plus a `wasCommitted()`
 * closure that the endpoint calls after the SSE writer has finished
 * consuming the stream. The closure compares `session.turns` against
 * a baseline captured AFTER any internal `session.reset()`, so the
 * signal is honest for the reset-and-cold-restart branch as well.
 */
async function runSessionStreaming(
  session: ChatSession<SessionCapableModel>,
  messages: ChatMessage[],
  newInputMessages: ChatMessage[],
  config: ChatConfig,
): Promise<StreamingOutcome> {
  if (session.turns === 0) {
    session.primeHistory(messages);
    const initialTurns = session.turns;
    return {
      stream: session.startFromHistoryStream(config),
      wasCommitted: () => session.turns > initialTurns,
    };
  }

  if (newInputMessages.length === 1) {
    const last = newInputMessages[0]!;
    const initialTurns = session.turns;
    if (last.role === 'user') {
      const images = last.images ?? undefined;
      return {
        stream: session.sendStream(last.content, images ? { images, config } : { config }),
        wasCommitted: () => session.turns > initialTurns,
      };
    }
    if (last.role === 'tool') {
      if (!last.toolCallId) {
        throw new Error('tool message missing toolCallId');
      }
      return {
        stream: session.sendToolResultStream(last.toolCallId, last.content, { config }),
        wasCommitted: () => session.turns > initialTurns,
      };
    }
    throw new Error(`unsupported last message role on hot path: ${last.role}`);
  }

  // Multi-message hot-path input: same reset-and-cold-restart as the
  // non-streaming variant. See `runSessionNonStreaming` for the
  // reasoning behind the post-success re-keying — and why the
  // initialTurns snapshot lives AFTER the reset.
  await session.reset();
  session.primeHistory(messages);
  const initialTurns = session.turns;
  return {
    stream: session.startFromHistoryStream(config),
    wasCommitted: () => session.turns > initialTurns,
  };
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

  // Echoed `function_call` items on a `previous_response_id` continuation
  // are validated for ownership (call_id must belong to the stored
  // trailing assistant turn) and then stripped unconditionally.
  //
  // Motivation: the common "round-trip `response.output` into the next
  // `input`" shape sends the prior assistant's `function_call` items
  // back alongside the new `function_call_output` results, which is a
  // legitimate client pattern. But `mapRequest` rebuilds each echoed
  // item into a synthetic assistant message at the tail of the
  // augmented `messages` list, which would both duplicate stored state
  // and (crucially) let a forged echo rewrite the trailing assistant
  // turn — poisoning `primeHistory()` and bypassing the multi-tool gate
  // below. Since `priorMessages` is the authoritative copy, the
  // correct response is to verify ownership by `call_id`, then drop
  // the echo so the stored view is used downstream.
  //
  // Name/arguments are NOT compared against stored — a client that
  // parses and reserializes its own prior arguments (different JSON
  // whitespace, key order, number formatting) would otherwise fail
  // continuation even though the server never consumes the echoed
  // payload. Any `call_id` absent from the stored index is still
  // rejected as an unambiguous forgery attempt.
  let effectiveInput = body.input;
  if (previousResponseId && priorMessages && Array.isArray(body.input)) {
    const storedCallIds = buildTrailingAssistantToolCallIds(priorMessages);
    const filtered: typeof body.input = [];
    for (const item of body.input) {
      if (item != null && typeof item === 'object' && (item as { type?: string }).type === 'function_call') {
        const fc = item as { call_id?: unknown };
        const callId = typeof fc.call_id === 'string' ? fc.call_id : null;
        if (!callId || !storedCallIds || !storedCallIds.has(callId)) {
          sendBadRequest(
            res,
            `echoed function_call item references an unknown call_id "${callId ?? ''}" — the stored ` +
              `trailing assistant turn is the authoritative copy, and any echoed function_call must ` +
              `reference one of its outstanding tool calls. Drop the echoed item or resolve the ` +
              `continuation against the correct previous_response_id.`,
            'input',
          );
          return;
        }
        // Stored state is authoritative — drop the echo regardless of
        // whether the client's `name`/`arguments` match byte-for-byte.
        continue;
      }
      filtered.push(item);
    }
    effectiveInput = filtered;
  }

  // Map request — full messages include prior + new input.
  // Feed mapRequest the echo-stripped input so no forged function_call
  // item can sneak through into the augmented trailing assistant turn.
  let messages: ChatMessage[];
  let config: ChatConfig;
  try {
    const mappedBody = effectiveInput === body.input ? body : { ...body, input: effectiveInput };
    ({ messages, config } = mapRequest(mappedBody, priorMessages));
  } catch (err) {
    sendBadRequest(res, err instanceof Error ? err.message : 'Invalid request input', 'input');
    return;
  }

  // Compute the new-only messages (what this request added, excluding prior history
  // and instructions). Instructions are stored separately and should not be persisted
  // as input messages — otherwise chained calls replay stale system messages.
  const instructionsOffset = body.instructions ? 1 : 0;
  const priorOffset = instructionsOffset + (priorMessages?.length ?? 0);
  let newInputMessages = messages.slice(priorOffset);

  // Client-shape validation: every tool message in the continuation delta
  // must carry a non-empty `tool_call_id`. Catching this up front gives a
  // clean 400 instead of letting `runSession*()` throw and be mapped to a
  // generic 500, but the real reason is correctness: the multi-tool-call
  // fan-out gate below authenticates submitted tool outputs against the
  // stored outstanding call-id set, and `submittedIds` / the set gate
  // silently ignores any tool message whose id is missing or empty. A
  // malicious client can otherwise submit `[tool(call_a), tool(call_b),
  // tool(/* anonymous */)]` against an outstanding pair `{call_a, call_b}`
  // — the id-set check would pass because both expected ids are present,
  // canonicalizeToolMessageOrder would also ignore the anonymous entry,
  // and the extra tool turn would slip through into native dispatch /
  // cold replay / persistence. Several native session backends identify
  // tool responses positionally or drop the id on the wire, so the extra
  // turn reopens tool-response injection despite the id-set gate. Reject
  // every anonymous tool message here so the gate can safely assume
  // every `role === 'tool'` item in `newInputMessages` carries a
  // well-formed id.
  for (const m of newInputMessages) {
    if (m.role === 'tool' && (typeof m.toolCallId !== 'string' || m.toolCallId.length === 0)) {
      sendBadRequest(res, 'tool message missing tool_call_id', 'input');
      return;
    }
  }

  // Extract the caller's current `instructions` (prefix/system state).
  // The session registry uses this to detect mid-chain changes — a hot
  // hit against a session warmed with different instructions would
  // silently keep using the stale system context, so we pass it to
  // `getOrCreate` and let the registry force a cold replay on
  // mismatch. Non-string values (including the field's absence) are
  // normalized to `null` so the equality check against stored entries
  // is byte-for-byte.
  const requestedInstructions: string | null = typeof body.instructions === 'string' ? body.instructions : null;

  // Route the request through a `ChatSession` looked up by the prior
  // response id. A miss (null id, unknown id, expired entry, or
  // prefix-state mismatch) returns a fresh session; a hit leases the
  // cached session out of the registry (single-use — the entry is
  // removed on hit so overlapping requests against the same prior id
  // cannot race on the same single-flight ChatSession).
  const session = sessionReg.getOrCreate(previousResponseId ?? null, requestedInstructions);

  // Multi-tool-call fan-out gate.
  //
  // The chat-session API cannot interleave tool results for a
  // multi-call fan-out turn (each `sendToolResult` dispatch re-opens
  // the assistant turn, so responding to the siblings would weave new
  // assistant replies between the results — see
  // `ChatSession.pendingUnresolvedToolCallCount`). The only valid forward
  // progress from such a turn is an atomic replay that resolves every
  // sibling call in one cold-restart, so we reject any continuation
  // whose submitted `function_call_output` set does not exactly match
  // the outstanding call ids.
  //
  // The gate only runs for `previous_response_id` continuations, where
  // the STORED prior chain (`priorMessages`, reconstructed via
  // `reconstructMessagesFromChain`) is the authoritative view of the
  // trailing assistant turn and `newInputMessages` contains only the
  // caller's continuation delta. Stateless requests (no
  // `previous_response_id`) carry a full self-contained history in
  // `input`, and historical tool outputs for prior resolved turns
  // would otherwise be misclassified against the latest assistant's
  // outstanding id set — leave cold-start histories to the jinja
  // template / chat-session prefill to handle as-is.
  const expectedOutstandingIds = priorMessages ? extractOutstandingToolCallIds(priorMessages) : null;

  // Forged-tool-output guard. A `previous_response_id` continuation that
  // submits any `function_call_output` when the stored prior chain has
  // ZERO outstanding tool calls is structurally invalid: there is no
  // assistant tool call for the result to resolve, so dispatching it
  // would inject a synthetic `<tool_response>` delta into a thread the
  // model never asked to call. Native backends do not authenticate
  // `tool_call_id` against prior state — several just append the
  // delta verbatim — so the gate must live here. Stateless requests
  // (no `previous_response_id`) carry a full self-contained history
  // and are left to the jinja template / chat-session prefill.
  if (previousResponseId && expectedOutstandingIds === null) {
    for (const m of newInputMessages) {
      if (m.role === 'tool') {
        sendBadRequest(
          res,
          `function_call_output submitted against a thread with no outstanding tool call. ` +
            `The prior assistant turn either never emitted a tool call or every sibling call has ` +
            `already been resolved, so there is nothing for this function_call_output to answer. ` +
            `Dispatching it anyway would synthesize a tool-response delta for a call the model ` +
            `never made and corrupt the conversation structure. Drop the function_call_output, ` +
            `or start a new chain without previous_response_id.`,
          'input',
        );
        return;
      }
    }
  }

  if (expectedOutstandingIds !== null) {
    // Contiguous-prefix guard: function_call_output items must appear
    // as an unbroken prefix of the continuation delta, before any
    // user/assistant/system message. A shape like
    // `[tool(call_a), user(hi), tool(call_b)]` would otherwise pass
    // every id-set check below (both outstanding ids present, no
    // duplicates, no stale ids) while still orphaning the fan-out,
    // because the interleaved user turn re-opens the assistant turn
    // between the two tool results. Reject early so the caller cannot
    // smuggle a user turn into the middle of a resolved fan-out.
    let seenNonTool = false;
    for (const m of newInputMessages) {
      if (m.role === 'tool') {
        if (seenNonTool) {
          sendBadRequest(
            res,
            `function_call_output items must appear as a contiguous prefix of the continuation ` +
              `before any user, assistant, or system message. Interleaving a non-tool message ` +
              `between sibling function_call_output items orphans the fan-out by weaving a new ` +
              `assistant turn between the tool results. Reorder the submission so every ` +
              `function_call_output precedes any subsequent message, or start a new chain ` +
              `without previous_response_id.`,
            'input',
          );
          return;
        }
      } else {
        seenNonTool = true;
      }
    }

    const submittedIds: string[] = [];
    for (const m of newInputMessages) {
      if (m.role === 'tool' && typeof m.toolCallId === 'string' && m.toolCallId.length > 0) {
        submittedIds.push(m.toolCallId);
      }
    }

    // Short-circuit: a plain user continuation (zero tool results)
    // would orphan the outstanding call(s) just as surely as a
    // partial tool-result submission. Reject both paths with the
    // same 400.
    const plural = expectedOutstandingIds.length > 1;
    if (submittedIds.length === 0) {
      sendBadRequest(
        res,
        `Previous assistant turn has ${expectedOutstandingIds.length} unresolved tool call${plural ? 's' : ''} ` +
          `(${expectedOutstandingIds.join(', ')}); the chat-session API requires every outstanding ` +
          `function_call_output to be submitted before the thread can advance. A plain user turn ` +
          `would orphan the unresolved call${plural ? 's' : ''}. Submit function_call_output items for ` +
          `every outstanding id, or start a new chain without previous_response_id.`,
        'input',
      );
      return;
    }

    const expectedSet = new Set(expectedOutstandingIds);
    const seen = new Set<string>();
    for (const id of submittedIds) {
      if (seen.has(id)) {
        sendBadRequest(
          res,
          `Duplicate function_call_output call_id "${id}" — each outstanding tool call must be answered exactly once.`,
          'input',
        );
        return;
      }
      seen.add(id);
      if (!expectedSet.has(id)) {
        sendBadRequest(
          res,
          `Unexpected function_call_output call_id "${id}"; the outstanding multi-tool-call set is ` +
            `${expectedOutstandingIds.join(', ')}. Submitting an unrelated or stale call_id would advance ` +
            `the chain past an unresolved turn.`,
          'input',
        );
        return;
      }
    }
    if (seen.size !== expectedSet.size) {
      const missing: string[] = [];
      for (const id of expectedOutstandingIds) {
        if (!seen.has(id)) missing.push(id);
      }
      sendBadRequest(
        res,
        `Missing function_call_output items for outstanding tool calls: ${missing.join(', ')}. ` +
          `Partial submissions would orphan the sibling tool calls and advance the chain past an ` +
          `unresolved turn. Resubmit with every sibling output, or start a new chain without ` +
          `previous_response_id.`,
        'input',
      );
      return;
    }

    // All outstanding ids are accounted for. Canonicalize the submitted
    // tool-message order to the stored sibling order before the replay
    // runs — both `messages` (primed into the fresh session on the cold
    // path) and `newInputMessages` (persisted verbatim into the store
    // for future chain reconstruction) must reflect the canonical
    // order, otherwise a caller can swap outputs and silently poison
    // replay even after the id-set gate passes.
    canonicalizeToolMessageOrder(messages, priorOffset, expectedOutstandingIds);
    newInputMessages = messages.slice(priorOffset);
  }

  try {
    // `runSession*` plumbs an honest commit signal out of the helper:
    // `ChatSession` only advances `turns` on a successful non-error
    // final chunk (streaming) or a resolved native promise
    // (non-streaming). The streaming safety-net path (generator
    // exhausts without a `done` event, see `handleStreamingNative`
    // fallback) and the `finishReason === 'error'` final chunk both
    // leave `turns` unchanged. The helper captures its baseline
    // AFTER any internal `session.reset()` on the multi-message
    // reset-and-cold-restart branch, so the signal is honest there
    // too — a pre-helper snapshot would be stale.
    let committed: boolean;
    if (body.stream) {
      const outcome = await runSessionStreaming(session, messages, newInputMessages, config);
      await handleStreamingNative(res, outcome.stream, body, responseId, previousResponseId, store, newInputMessages);
      committed = outcome.wasCommitted();
    } else {
      const outcome = await runSessionNonStreaming(session, messages, newInputMessages, config);
      await handleNonStreaming(res, outcome.result, body, responseId, previousResponseId, store, newInputMessages);
      committed = outcome.committed;
    }

    // Belt-and-braces: the prior id was already leased out of the
    // registry at `getOrCreate` time (single-use lease semantics), so
    // this drop is a no-op in the common case. It stays because a
    // defensive `drop` keeps the bookkeeping readable and guards
    // against future refactors that might re-introduce a non-lease
    // hit path.
    //
    // Only adopt under the new id when the session actually committed
    // — otherwise future chained requests must fall through to the
    // cold-replay path, which reconstructs from `ResponseStore` on a
    // fresh session, so the in-memory KV cache cannot diverge from
    // the persisted chain.
    if (previousResponseId) {
      sessionReg.drop(previousResponseId);
    }
    if (committed) {
      sessionReg.adopt(responseId, session, requestedInstructions);
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
