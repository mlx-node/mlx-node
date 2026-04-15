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
  modelInstanceId: number | undefined,
): Promise<void> {
  const response = buildResponseObject(result, req, responseId, previousResponseId);

  // Persist only the new input messages (not the full expanded conversation)
  if (store && req.store !== false) {
    await persistResponse(store, response, newInputMessages, previousResponseId, modelInstanceId);
  }

  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(response));
}

// ---------------------------------------------------------------------------
// Streaming path
// ---------------------------------------------------------------------------

/**
 * Stream a chat session's events to the SSE writer, gated on the
 * session's commit signal.
 *
 * `wasCommitted` is a closure that reads `session.turns` at call
 * time. On the streaming path the session only advances `turns` on a
 * successful non-error final chunk (see `ChatSession.sendStream`'s
 * `sawFinal` gate), so this closure returns `false` when the native
 * stream emits `done: true, finishReason: 'error'`, when the async
 * iterator exhausts without a `done` event, or when a mid-decode
 * throw propagates through. In every non-committed case we MUST skip
 * `persistResponse()` and emit `response.failed` instead of
 * `response.completed`, otherwise a later `previous_response_id`
 * continuation would cold-replay a turn the session never committed —
 * silently resurrecting failed or partial output as authoritative
 * history.
 *
 * The closure is called AFTER the `for await` loop has fully drained
 * (either via a `break` inside the done branch or because the
 * iterator exhausted). Draining is load-bearing: `ChatSession`
 * increments `turns` in the generator's `finally` block, which only
 * runs once the consumer's `.return()` / natural-exhaust cascade
 * reaches the outer generator. A pre-drain `wasCommitted()` would
 * read a stale baseline and falsely report "not committed" even on a
 * successful turn. The `runSessionStreaming` helper captures its
 * baseline AFTER any internal `session.reset()` too, so the signal is
 * honest for the multi-message reset-and-cold-restart branch as well.
 */
async function handleStreamingNative(
  res: ServerResponse,
  chatStream: AsyncGenerator<ChatStreamEvent>,
  req: ResponsesAPIRequest,
  responseId: string,
  previousResponseId: string | undefined,
  store: ResponseStore | null,
  newInputMessages: ChatMessage[],
  wasCommitted: () => boolean,
  modelInstanceId: number | undefined,
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

  // Terminal response captured inside the done branch (or synthesized
  // in the fallback after the loop if the iterator exhausted). The
  // actual `response.completed` / `response.failed` emission is
  // deferred until AFTER the loop drains so `wasCommitted()` can read
  // an authoritative `session.turns` — otherwise we would emit the
  // terminal event while the producer's finally has not yet run.
  let completedResponse: ResponseObject | null = null;
  let sawDone = false;

  for await (const event of chatStream) {
    if (event.done) {
      sawDone = true;
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

      // Build the terminal response object but do NOT persist or emit
      // `response.completed` yet — both actions are gated on the
      // session's commit signal, which only becomes authoritative
      // after the outer generator's finally has run. We `break` out
      // of the loop so the for-await's cleanup runs the producer's
      // finally (setting `turnCount` if the session committed), then
      // defer persistence + emission to the post-loop block below.
      const promptTokens = event.promptTokens ?? 0;
      const reasoningTokens = event.reasoningTokens ?? 0;
      const usage = {
        input_tokens: promptTokens,
        output_tokens: event.numTokens,
        output_tokens_details: { reasoning_tokens: reasoningTokens },
        total_tokens: promptTokens + event.numTokens,
      };

      const finalOutput = outputItems.filter((_, idx) => idx !== suppressedMessageIndex);
      completedResponse = {
        ...partial,
        status: mapFinishReasonToStatus(event.finishReason),
        output: finalOutput,
        output_text: computeOutputText(finalOutput),
        incomplete_details: event.finishReason === 'length' ? { reason: 'max_output_tokens' } : null,
        usage,
      };
      break;
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

  // Post-loop terminal emission.
  //
  // The producer's finally has now run (either via the `break` after
  // a done event or via natural iterator exhaustion), so
  // `wasCommitted()` reads an authoritative `session.turns` baseline.
  // Three cases:
  //
  //  1. sawDone && committed: happy path. Persist the terminal
  //     response and emit `response.completed`. Future
  //     `previous_response_id` continuations can hot-resume through
  //     the registry or cold-replay from the store.
  //  2. sawDone && !committed: the final chunk carried
  //     `finishReason: 'error'` (the ChatSession gates `turnCount` on
  //     a non-error final chunk, so the session never committed). We
  //     skip persistence and emit `response.failed` so clients can't
  //     chain off of output the session never accepted as history.
  //  3. !sawDone: the iterator exhausted before a terminal chunk
  //     arrived. The session also never committed in this path, so
  //     we synthesize an incomplete fallback, skip persistence, and
  //     emit `response.failed`.
  //
  // In all non-committed paths the registry-level `adopt()` gate in
  // `handleCreateResponse` already skipped caching this session, so
  // the in-memory and persisted views agree: there is no authoritative
  // record of this turn anywhere.
  const committed = wasCommitted();

  if (!sawDone) {
    const fallbackOutput = outputItems.filter((_, idx) => idx !== suppressedMessageIndex);
    completedResponse = {
      ...partial,
      status: 'incomplete',
      output: fallbackOutput,
      output_text: computeOutputText(fallbackOutput),
      incomplete_details: { reason: 'max_output_tokens' },
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        output_tokens_details: { reasoning_tokens: 0 },
        total_tokens: 0,
      },
    };
  }

  // `completedResponse` is non-null at this point: either the done
  // branch set it, or the `!sawDone` fallback above set it. Assert
  // for the type checker.
  const terminal = completedResponse!;

  if (committed) {
    if (store && req.store !== false) {
      await persistResponse(store, terminal, newInputMessages, previousResponseId, modelInstanceId);
    }
    writeSSEEvent(res, 'response.completed', { response: terminal });
  } else {
    // Non-committed terminal. Do NOT persist — the ChatSession refused
    // to advance its history, so the store must agree. Re-label the
    // response status as `failed` so a client round-tripping the
    // response into `previous_response_id` cannot accidentally
    // resurrect partial output.
    const failedResponse: ResponseObject = { ...terminal, status: 'failed' };
    writeSSEEvent(res, 'response.failed', { response: failedResponse });
  }
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
 * Reorder the tool messages in `messages` across the half-open range
 * `[startOffset, blockEnd)` so their relative positions match
 * `expectedOrder`.
 *
 * Replay correctness for a multi-tool-call fan-out depends on POSITION,
 * not `tool_call_id` — several backends drop the id on the wire and
 * pair tool responses to the trailing assistant calls by sibling index.
 * A caller that submits `function_call_output` items in the wrong order
 * would therefore silently bind results to the wrong calls even after
 * the id-set gate passes. This helper canonicalizes the submitted
 * ordering to the stored sibling order before the replay runs.
 *
 * The `blockEnd` bound is load-bearing: callers MUST size it to a
 * single contiguous tool block (i.e. the run of `role === 'tool'`
 * messages that immediately follow one assistant fan-out). A history
 * with multiple resolved fan-outs has several such blocks, and
 * walking past the first block's end would pull in tool messages from
 * a later, unrelated fan-out — the id-set gate below would then bail
 * on `toolPositions.length !== expectedOrder.length` without
 * reordering anything, silently leaving the first block misordered.
 * The full-history walker at `validateAndCanonicalizeHistoryToolOrder`
 * computes a `blockEnd` per fan-out and invokes this helper once per
 * block; the `previous_response_id` continuation path computes its
 * own `blockEnd` by scanning forward while the next message is a
 * `tool` turn.
 *
 * Assumes the call-id SET has already been validated against
 * `expectedOrder`; this helper is a no-op when any precondition does
 * not hold (missing id, count mismatch, etc.) so callers are safe to
 * invoke it unconditionally after the gate passes.
 */
function canonicalizeToolMessageOrder(
  messages: ChatMessage[],
  startOffset: number,
  blockEnd: number,
  expectedOrder: readonly string[],
): void {
  const toolPositions: number[] = [];
  const byId = new Map<string, ChatMessage>();
  for (let i = startOffset; i < blockEnd; i++) {
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
 * Walk `messages` and canonicalize every assistant fan-out's
 * trailing tool-result block against the assistant's declared
 * `toolCalls` order. Mutates `messages` in place.
 *
 * The existing `canonicalizeToolMessageOrder` only handles a single
 * contiguous tool block at a known offset against a precomputed
 * `expectedOrder` — it was built for the `previous_response_id`
 * continuation path, where the stored prior chain supplies the
 * trailing assistant's outstanding ids and only the caller's new
 * delta needs to be reordered. This helper, by contrast, walks the
 * FULL history and canonicalizes EVERY fan-out block in it, so it
 * can be invoked on stateless cold-start histories (no
 * `previous_response_id`) and on the Anthropic `/v1/messages`
 * endpoint, both of which feed caller-supplied tool-message order
 * straight into `primeHistory()` without the continuation gate
 * running.
 *
 * Validation rules (checked BEFORE any reorder):
 *
 *   - Every `role === 'tool'` message in the history must appear
 *     inside a contiguous block immediately following an assistant
 *     fan-out turn. An orphan tool message (no preceding assistant,
 *     or the preceding assistant has no `toolCalls`) is a violation.
 *   - Inside a fan-out's tool block, every submitted `toolCallId`
 *     must appear in the assistant's declared sibling-id set.
 *   - The tool block must contain exactly one message per declared
 *     sibling id — no missing ids, no extras, no duplicates.
 *   - The final assistant turn in the history is not allowed to be
 *     an unresolved fan-out: if the last assistant carries tool
 *     calls and no resolutions follow it, the caller is submitting
 *     a self-contained history whose trailing turn the chat-session
 *     API cannot express as a continuation seed. Reject the request
 *     rather than silently advancing into the model. (The
 *     continuation path has its own gate for this shape — we do NOT
 *     run the helper on the previous_response_id branch's delta,
 *     see the call site for the invocation condition.)
 *
 * Canonicalization only runs once every precondition passes. The
 * reorder is in place: `messages[i]` entries are swapped to match
 * the sibling order, nothing is inserted or deleted.
 *
 * @param apiSurface Controls the vocabulary used in error
 *   strings. Defaults to `'openai'` so the `/v1/responses`
 *   endpoint returns `function_call_output` / `call_id`
 *   wording. Pass `'anthropic'` from the `/v1/messages` endpoint
 *   so callers who posted `tool_result` / `tool_use_id` get
 *   remediation advice in their own request vocabulary (iter-23
 *   finding 4). The validation logic and canonicalization are
 *   identical between surfaces — only the error text differs.
 *
 * @returns `null` on success, or a human-readable error string
 *   describing the first violation. Callers send the string back as
 *   a 400 `invalid_request_error`.
 */
export function validateAndCanonicalizeHistoryToolOrder(
  messages: ChatMessage[],
  apiSurface: 'openai' | 'anthropic' = 'openai',
): string | null {
  // Map surface-specific names so every error string below reads
  // in the caller's own vocabulary. The OpenAI responses surface
  // uses `function_call_output` / `call_id` / "assistant fan-out";
  // the Anthropic messages surface uses `tool_result` /
  // `tool_use_id` / "assistant turn with tool_use blocks".
  const vocab =
    apiSurface === 'anthropic'
      ? {
          toolResult: 'tool_result',
          toolCallId: 'tool_use_id',
          fanOut: 'assistant turn with tool_use blocks',
        }
      : {
          toolResult: 'function_call_output',
          toolCallId: 'call_id',
          fanOut: 'assistant fan-out',
        };

  // Walk forward. When we see an assistant fan-out, read the
  // contiguous tool block that follows and canonicalize it.
  // When we see a tool message outside such a block, that's an
  // orphan and we reject.
  let i = 0;
  while (i < messages.length) {
    const m = messages[i]!;
    if (m.role === 'tool') {
      return (
        `tool message at index ${i} (${vocab.toolCallId} "${m.toolCallId ?? ''}") is not preceded by an ` +
        `${vocab.fanOut}. Every ${vocab.toolResult} must immediately follow the assistant turn whose ` +
        `tool calls include its ${vocab.toolCallId}.`
      );
    }
    if (m.role !== 'assistant' || !m.toolCalls || m.toolCalls.length === 0) {
      i++;
      continue;
    }

    // Assistant fan-out. Collect declared sibling ids.
    const declaredIds: string[] = [];
    const declaredSet = new Set<string>();
    for (const tc of m.toolCalls) {
      const id = typeof tc.id === 'string' ? tc.id : null;
      if (id === null || id.length === 0) {
        // Assistant tool call without an id — the server should never
        // have produced one, but be defensive. Skip canonicalization
        // for this fan-out; without an id we cannot reorder
        // positionally by id.
        return (
          `${vocab.fanOut} at index ${i} declares a tool call with no id, which cannot be paired ` +
          `with its ${vocab.toolResult} positionally.`
        );
      }
      if (declaredSet.has(id)) {
        return (
          `${vocab.fanOut} at index ${i} declares duplicate ${vocab.toolCallId} "${id}". Each sibling ` +
          `call must have a unique ${vocab.toolCallId}.`
        );
      }
      declaredIds.push(id);
      declaredSet.add(id);
    }

    // Read the contiguous tool block following the fan-out.
    const blockStart = i + 1;
    let blockEnd = blockStart;
    const seenInBlock = new Set<string>();
    while (blockEnd < messages.length && messages[blockEnd]!.role === 'tool') {
      const tool = messages[blockEnd]!;
      const id = typeof tool.toolCallId === 'string' ? tool.toolCallId : null;
      if (id === null || id.length === 0) {
        return (
          `tool message at index ${blockEnd} is missing ${vocab.toolCallId}. Every ${vocab.toolResult} ` +
          `in an ${vocab.fanOut}'s resolution block must carry the ${vocab.toolCallId} it resolves.`
        );
      }
      if (!declaredSet.has(id)) {
        return (
          `tool message at index ${blockEnd} references ${vocab.toolCallId} "${id}", which is not ` +
          `declared by the preceding ${vocab.fanOut} at index ${i}. Submitting a ${vocab.toolResult} ` +
          `for an undeclared ${vocab.toolCallId} would silently bind output to the wrong sibling.`
        );
      }
      if (seenInBlock.has(id)) {
        return (
          `duplicate tool message for ${vocab.toolCallId} "${id}" inside the ${vocab.fanOut}'s ` +
          `resolution block (index ${blockEnd}). Each outstanding sibling must be resolved exactly once.`
        );
      }
      seenInBlock.add(id);
      blockEnd++;
    }

    const blockLength = blockEnd - blockStart;
    if (blockLength === 0) {
      // No resolutions at all. Allowed ONLY when the fan-out is the
      // trailing assistant turn AND the caller intends to submit
      // tool results in a follow-up request. In a self-contained
      // stateless history (which is what this helper is invoked
      // against) the chain cannot end with an unresolved fan-out —
      // the chat-session API would have nothing to continue from.
      if (blockEnd === messages.length) {
        return (
          `${vocab.fanOut} at index ${i} is the trailing turn of the history but has no ` +
          `${vocab.toolResult} resolutions. A stateless cold-start history cannot end on an ` +
          `unresolved tool-call fan-out because there is nothing for the model to continue from.`
        );
      }
      // Mid-history assistant fan-out followed directly by another
      // assistant/user/system message. This shape orphans the fan-out.
      return (
        `${vocab.fanOut} at index ${i} declares ${declaredIds.length} tool call${declaredIds.length === 1 ? '' : 's'} ` +
        `but the next message at index ${blockEnd} is a ${messages[blockEnd]!.role} turn. Every fan-out ` +
        `must be fully resolved by ${vocab.toolResult} messages before the next assistant/user/system turn.`
      );
    }
    if (blockLength < declaredIds.length) {
      const missing = declaredIds.filter((id) => !seenInBlock.has(id));
      return (
        `${vocab.fanOut} at index ${i} has unresolved sibling tool calls: ${missing.join(', ')}. ` +
        `Every declared tool call must be answered by a ${vocab.toolResult} before the next turn.`
      );
    }
    // blockLength > declaredIds.length is impossible: every entry in
    // the block must have an id in declaredSet, and seenInBlock
    // deduplicates by id, so seen.size == blockLength ≤ declaredIds.length.

    // Canonicalize. The existing canonicalizeToolMessageOrder handles
    // a single block cleanly — reuse it so the reorder logic lives
    // in one place. Pass `blockEnd` so the helper only inspects THIS
    // fan-out's contiguous tool block and doesn't accidentally scan
    // into a later fan-out's tool messages (which would cause the
    // helper's count gate to bail without reordering anything).
    canonicalizeToolMessageOrder(messages, blockStart, blockEnd, declaredIds);

    // Advance past the resolved block.
    i = blockEnd;
  }

  return null;
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
  modelInstanceId: number | undefined,
): Promise<void> {
  // Store only the NEW input messages from this request, not the full
  // expanded conversation. Chain reconstruction re-derives the full history
  // by following previous_response_id links.
  //
  // `modelInstanceId` is the monotonic id `ModelRegistry` assigned to
  // the model object that serviced this request. It is stashed inside
  // the `configJson` blob so the Rust-side schema stays untouched; on
  // a later `previous_response_id` continuation the responses endpoint
  // reads it back out of the trailing chain record and compares it
  // against the live id for `body.model`. See the endpoint's
  // `readStoredModelIdentity` helper and the guard block in
  // `handleCreateResponse` — records without this field are rejected
  // outright per iter-23 finding 1 (the iter-22 friendly-name
  // compat fallback silently reopened same-name hot-swap corruption
  // and has been removed).
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
      modelInstanceId,
    }),
    expiresAt: Math.floor(Date.now() / 1000) + RESPONSE_TTL_SECONDS,
  };
  await store.store(record);
}

/**
 * Identity signal extracted from a stored chain record's
 * `configJson` blob:
 *
 *   - `{ kind: 'present', instanceId }` — the record carries an
 *     explicit `modelInstanceId`. The caller runs the strict
 *     instance-id comparison and rejects any mismatch as a
 *     hot-swap / rebind.
 *   - `{ kind: 'absent' }` — the record either has no `configJson`,
 *     has a malformed blob, or has a blob that doesn't carry a
 *     well-formed `modelInstanceId` field. Per iter-23 finding 1
 *     the caller rejects these outright with a 400 explaining the
 *     upgrade boundary — the iter-22 friendly-name compat fallback
 *     silently reopened same-name hot-swap corruption and has
 *     been removed. The discriminated return is kept so the call
 *     site explicitly handles the absent case rather than
 *     implicitly allowing a legacy record to slip through.
 */
type StoredModelIdentity = { kind: 'present'; instanceId: number } | { kind: 'absent' };

function readStoredModelIdentity(record: StoredResponseRecord): StoredModelIdentity {
  if (record.configJson == null) return { kind: 'absent' };
  try {
    const parsed = JSON.parse(record.configJson) as { modelInstanceId?: unknown };
    if (typeof parsed.modelInstanceId === 'number' && Number.isFinite(parsed.modelInstanceId)) {
      return { kind: 'present', instanceId: parsed.modelInstanceId };
    }
    return { kind: 'absent' };
  } catch {
    return { kind: 'absent' };
  }
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

  // Capture an initial snapshot of the live binding for `body.model`.
  // These values are the INITIAL observation — on a
  // `previous_response_id` continuation we re-read them after
  // `await store.getChain(...)` and reject the request if the
  // binding moved under us (see the hot-swap race guard below).
  // Stateless requests never hit the store so the re-read is a
  // no-op for them.
  const initialSessionReg: SessionRegistry | undefined = registry.getSessionRegistry(body.model);
  if (!initialSessionReg) {
    sendInternalError(res, 'session registry missing for registered model');
    return;
  }
  const initialInstanceId = registry.getInstanceId(body.model);

  // Mutable handles for the registry binding that actually gets
  // used for dispatch / persistence. For stateless requests these
  // stay equal to the initial snapshot. For a `previous_response_id`
  // continuation they are re-read after `await store.getChain()`
  // and, if they match the initial snapshot, are used as the
  // canonical current-binding values from that point forward.
  let sessionReg: SessionRegistry = initialSessionReg;
  let currentInstanceId: number | undefined = initialInstanceId;

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

      // Hot-swap race guard (iter-22 finding 3).
      //
      // Between the pre-await snapshot above and this point a
      // concurrent `registry.register(body.model, differentModel)`
      // can re-point the friendly name at a new object. If we
      // kept using `initialSessionReg` / `initialInstanceId` the
      // request would dispatch through the stale session
      // registry, compare the stored identity against the dead
      // instance id, and persist the new record under the old
      // binding — even though `body.model` now resolves to a
      // different live model. Re-read the current binding and
      // reject the request when anything changed so the caller
      // can retry against the new identity.
      const refreshedSessionReg = registry.getSessionRegistry(body.model);
      const refreshedInstanceId = registry.getInstanceId(body.model);
      if (
        refreshedSessionReg === undefined ||
        refreshedInstanceId === undefined ||
        refreshedSessionReg !== initialSessionReg ||
        refreshedInstanceId !== initialInstanceId
      ) {
        sendBadRequest(
          res,
          `Model "${body.model}" binding changed while the request was resolving its previous_response_id ` +
            `chain. A concurrent register() re-pointed the name at a different model instance (or released ` +
            `it entirely) during the store lookup, so the session registry and instance id captured before ` +
            `the await no longer match the live binding. Dispatching anyway would replay the stored chain ` +
            `through the wrong model. Retry the request — if the swap was intentional, the new binding will ` +
            `service the retry cleanly.`,
          'model',
        );
        return;
      }
      sessionReg = refreshedSessionReg;
      currentInstanceId = refreshedInstanceId;

      // Cross-model continuation guard, keyed on MODEL-INSTANCE
      // IDENTITY, not friendly name. The stored trailing record
      // carries a monotonic `modelInstanceId` assigned by
      // `ModelRegistry` when the model object that serviced the
      // original turn was first registered; we compare that id
      // against the CURRENT instance id for `body.model`.
      //
      // A plain string comparison on the friendly name is not
      // sufficient:
      //
      //  * `ModelRegistry.register(name, model)` explicitly supports
      //    replacing the object bound to a name. A chain produced
      //    by the OLD instance of `foo` would still pass a name
      //    check after `foo` is hot-swapped to a different model,
      //    and the continuation would silently replay through the
      //    wrong tokenizer / chat template / KV layout.
      //  * Conversely, two names aliasing the SAME model instance
      //    are already safe because iter-19's per-instance
      //    `SessionRegistry` sharing routes them through one
      //    binding — but a name check would spuriously reject
      //    `body.model = "beta"` against a chain stored under
      //    `"alpha"`.
      //
      // Strict instance-id policy (iter-23 finding 1). Every stored
      // record must carry an explicit `modelInstanceId`. Records
      // that lack the field — either because they predate iter-21
      // or because a tool rewrote `configJson` without preserving
      // unknown keys — are rejected outright. The iter-22 compat
      // path that fell back to friendly-name comparison silently
      // reopened the same-name hot-swap corruption window: any
      // stored row without identity could be replayed through a
      // different tokenizer / chat template / KV layout as long as
      // the friendly name matched. This branch is an explicit
      // breaking change in chain semantics — no production rows
      // exist without identity on `feat/qwen35-chat-session`, so
      // strict-reject is safe to land without a migration.
      const trailingRecord = chain[chain.length - 1]!;
      const storedIdentity = readStoredModelIdentity(trailingRecord);
      if (storedIdentity.kind === 'absent') {
        sendBadRequest(
          res,
          `previous_response_id "${body.previous_response_id}" belongs to a stored chain whose trailing ` +
            `record does not carry a modelInstanceId. Such records predate the iter-21 identity scheme ` +
            `(or were rewritten without preserving the identity field) and are not eligible for ` +
            `continuation: a friendly-name comparison would silently reopen same-name hot-swap corruption, ` +
            `replaying the chain through a potentially different tokenizer, chat template, or KV layout. ` +
            `Start a new chain without previous_response_id.`,
          'model',
        );
        return;
      }
      if (currentInstanceId === undefined || storedIdentity.instanceId !== currentInstanceId) {
        sendBadRequest(
          res,
          `previous_response_id "${body.previous_response_id}" belongs to a chain produced by a different ` +
            `model instance than the one currently bound to "${body.model}". This happens when the named ` +
            `model has been hot-swapped to a different underlying object since the chain was stored or ` +
            `when the original binding has been released entirely. Continuations cannot cross model ` +
            `boundaries — a stored chain is tied to the tokenizer, chat template, and KV layout of the ` +
            `exact model object that produced it, and replaying it through a different model would ` +
            `silently corrupt the conversation. Start a new chain without previous_response_id.`,
          'model',
        );
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

  // Per-model execution mutex. Every dispatch through this endpoint
  // serializes with every dispatch through `/v1/messages` for the
  // same model binding. The native model is a single mutable
  // resource — one `cached_token_history` / one `caches` vector per
  // `SessionCapableModel` instance — so two concurrent `primeHistory`
  // / `send*` calls would clobber each other's KV state even though
  // `getOrCreate` hands out distinct `ChatSession` wrappers. The
  // mutex restores correctness by making the entire
  // `getOrCreate → dispatch → adopt/drop` span exclusive for this
  // model, and the `finally` inside `withExclusive` releases the
  // lock on both success and failure so a rejected dispatch cannot
  // leave the next waiter stuck.
  //
  // Validation inside the exclusive block runs synchronously before
  // any native work begins, so a 400 early return under the lock
  // releases it immediately for the next waiter — the fan-out
  // gate's `return` statements exit the closure without calling
  // any native decode entry points.
  await sessionReg.withExclusive(async () => {
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
      //
      // Compute the tool block's end as the contiguous-prefix run of
      // `role === 'tool'` messages starting at `priorOffset`. The
      // contiguous-prefix guard above already rejected any shape that
      // interleaves a non-tool message inside the delta's tool block,
      // so this simple forward scan matches the exact block the gate
      // just authenticated. Passing an explicit `blockEnd` keeps the
      // helper from accidentally walking into any later turn that
      // `mapRequest` may have appended to `messages`.
      let deltaBlockEnd = priorOffset;
      while (deltaBlockEnd < messages.length && messages[deltaBlockEnd]!.role === 'tool') {
        deltaBlockEnd++;
      }
      canonicalizeToolMessageOrder(messages, priorOffset, deltaBlockEnd, expectedOutstandingIds);
      newInputMessages = messages.slice(priorOffset);
    }

    // Walk the full merged history and canonicalize every assistant
    // fan-out's trailing tool block against its declared sibling order.
    //
    // The multi-tool-call gate above only fires on `previous_response_id`
    // continuations, and even there it only handles the caller's delta
    // block against the STORED prior chain's trailing assistant. That
    // leaves two cases uncovered:
    //
    //   1. Stateless cold-start histories (no `previous_response_id`).
    //      The caller ships a full self-contained conversation through
    //      `input`; the gate is skipped entirely and the caller-supplied
    //      tool-message order flows straight into `primeHistory()`. A
    //      caller can reverse two sibling tool outputs, and since
    //      several native session backends pair tool results to
    //      fan-out calls POSITIONALLY (not by id), each result binds
    //      to the wrong sibling call.
    //   2. Earlier fan-outs embedded inside the stored prior history
    //      on a continuation. Those came from the server's own store
    //      so they should already be canonical, but defense in depth
    //      is cheap — a single full-history walk covers every shape.
    //
    // Malformed histories (missing/duplicate/unknown ids, orphan tool
    // messages, unresolved trailing fan-out in a stateless request)
    // are rejected with a clear 400 instead of silently rewritten.
    const historyError = validateAndCanonicalizeHistoryToolOrder(messages);
    if (historyError !== null) {
      sendBadRequest(res, historyError, 'input');
      return;
    }
    // Canonicalization may have reordered tool messages inside the
    // continuation delta (on the stateless-history walk over the
    // post-priorOffset portion), so recompute `newInputMessages` from
    // the now-canonical `messages`.
    newInputMessages = messages.slice(priorOffset);

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
        const streamingWasCommitted = () => outcome.wasCommitted();
        await handleStreamingNative(
          res,
          outcome.stream,
          body,
          responseId,
          previousResponseId,
          store,
          newInputMessages,
          streamingWasCommitted,
          currentInstanceId,
        );
        committed = streamingWasCommitted();
      } else {
        const outcome = await runSessionNonStreaming(session, messages, newInputMessages, config);
        await handleNonStreaming(
          res,
          outcome.result,
          body,
          responseId,
          previousResponseId,
          store,
          newInputMessages,
          currentInstanceId,
        );
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
  });
}
