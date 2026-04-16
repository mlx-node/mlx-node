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

import type { IncomingMessage, ServerResponse } from 'node:http';

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

/**
 * Mutable visibility flags shared between the per-handler
 * (non-streaming / streaming) path and the outer
 * `handleCreateResponse` error block. Distinct from `res.headersSent`
 * because Node's `ServerResponse.writeHead()` flips `headersSent`
 * synchronously BEFORE any body bytes leave the buffer. A handler
 * that throws inside `res.end()` after `writeHead()` has a
 * `headersSent === true` but the client never actually observed the
 * response — adopting the committed session under that unseen
 * responseId would leak a warm session into the registry that no
 * caller can ever reach.
 *
 * Each handler flips the flag that corresponds to "the client has
 * actually seen a terminal artefact for this responseId" on its own
 * success path:
 *
 *   * `responseBodyWritten` — non-streaming: set ONLY after
 *     `res.end(JSON.stringify(response))` completes without throwing.
 *   * `terminalEmitted` — streaming: set ONLY after a terminal SSE
 *     event (`response.completed` on the success path, or
 *     `response.failed` on the failure epilogue) has been written to
 *     the wire without throwing.
 *
 * The outer catch computes `safeToSuppress = responseBodyWritten ||
 * terminalEmitted`. A committed turn whose handler threw WITHOUT
 * setting either flag is NOT adopted and the handler error is
 * rethrown (so the outer `sendInternalError` path sends a 500 the
 * client can parse, and the registry stays in sync with the client's
 * view of the world).
 */
interface ResponseVisibility {
  /** Non-streaming: `res.end(body)` returned without throwing. */
  responseBodyWritten: boolean;
  /** Streaming: a terminal SSE event was written to the wire. */
  terminalEmitted: boolean;
}

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
  visibility: ResponseVisibility,
): Promise<void> {
  const response = buildResponseObject(result, req, responseId, previousResponseId);

  // Persist only the new input messages (not the full expanded conversation).
  // Persistence is best-effort: the turn is already committed in the
  // session's KV cache, so a store failure must not prevent the client
  // from receiving the response (with its responseId for hot-resume).
  if (store && req.store !== false) {
    try {
      await persistResponse(store, response, newInputMessages, previousResponseId, modelInstanceId);
    } catch (err) {
      console.error('[responses] post-commit persistence failed, response will still be sent:', err);
    }
  }

  // `writeHead` flips `res.headersSent` synchronously, but the client
  // has not actually received the JSON body yet — if `res.end()`
  // below throws, `res.headersSent` alone cannot tell the outer
  // handler whether the response made it to the wire. Only flip
  // `responseBodyWritten` AFTER `res.end()` returns cleanly, so the
  // adopt-on-visibility gate treats an end-time crash the same way
  // as a writeHead-time crash (no adopt, rethrow).
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(response));
  visibility.responseBodyWritten = true;
}

// ---------------------------------------------------------------------------
// Streaming path
// ---------------------------------------------------------------------------

/**
 * Build a dedicated failure terminal ResponseObject from an
 * in-progress partial + the deltas captured so far. The returned
 * object has:
 *
 *   * `status: 'failed'`
 *   * `incomplete_details: { reason }` — the string passed by the
 *     caller (`error`, `client_abort`, `stream_exhausted`, etc.).
 *   * Every nested output item whose `status` is still `in_progress`
 *     or `completed` normalized to `incomplete`, so a client that
 *     inspects `response.output` on `response.failed` cannot see a
 *     success-shaped item inside a failed envelope. Iter-28
 *     finding 3: the previous implementation did `{ ...terminal,
 *     status: 'failed' }`, which left nested messages marked
 *     `completed` (on the finishReason=error path where the done
 *     branch finalized them) or `in_progress` (on the exhaust path
 *     where no item-closing ran at all). Both shapes contradicted
 *     the top-level failure status.
 *
 * `ReasoningOutputItem` has no `status` field and is left alone.
 * `FunctionCallOutputItem` items whose `status` is `completed` or
 * `in_progress` are also downgraded to `incomplete` — iter-29
 * finding 1 concluded that the previous exemption (leaving
 * function_call items untouched because the type was narrow) was
 * incorrect: streaming tool_call items can now be collected into
 * `outputItems` before the commit gate passes, and a failed
 * terminal that reports them as `completed` contradicts the
 * top-level `status: 'failed'` envelope.
 */
function buildFailedTerminal(
  partial: ResponseObject,
  outputItems: OutputItem[],
  reason: string,
  usage: ResponseObject['usage'],
): ResponseObject {
  const normalized: OutputItem[] = outputItems.map((item) => {
    if (item.type === 'message') {
      const prev = item.status;
      if (prev === 'in_progress' || prev === 'completed') {
        return { ...item, status: 'incomplete' };
      }
      return item;
    }
    if (item.type === 'function_call') {
      if (item.status === 'completed' || item.status === 'incomplete') {
        return { ...item, status: 'incomplete' as const };
      }
      return item;
    }
    return item;
  });
  return {
    ...partial,
    status: 'failed',
    output: normalized,
    output_text: computeOutputText(normalized),
    incomplete_details: { reason },
    usage,
  };
}

/**
 * Stream a chat session's events to the SSE writer, gated on the
 * session's commit signal.
 *
 * `wasCommitted` is a closure that reads `session.turns` at call
 * time. On the streaming path the session only advances `turns` on a
 * successful non-error final chunk (see `ChatSession.sendStream`'s
 * `sawFinal` gate), so this closure returns `false` when the native
 * stream emits `done: true, finishReason: 'error'`, when the async
 * iterator exhausts without a `done` event, when a mid-decode throw
 * propagates through (caught by the try/catch added in iter-28
 * finding 2), or when the client disconnect flag fires mid-iteration.
 * In every non-committed case we MUST skip `persistResponse()` and
 * emit `response.failed` instead of `response.completed`, otherwise
 * a later `previous_response_id` continuation would cold-replay a
 * turn the session never committed — silently resurrecting failed
 * or partial output as authoritative history.
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
 *
 * Iter-28 finding 2 — fault plumbing:
 *
 *   1. The `for await` loop is wrapped in try/catch/finally so a
 *      mid-decode throw from the underlying generator no longer
 *      escapes out into the outer handler's generic error catch.
 *      Instead control reaches the post-loop block with a sticky
 *      `thrownError` flag; the block routes the request through the
 *      same failure epilogue that handles finishReason=error and
 *      iterator exhaustion, so the session is NEVER adopted via
 *      `wasCommitted()` on a faulted stream.
 *   2. When the caller passes `httpReq`, we install `close`/`error`
 *      listeners that flip a `clientAborted` flag checked at the
 *      top of every loop iteration. The underlying
 *      `chatStreamSessionStart` does not yet accept an AbortSignal,
 *      so we cannot cancel the native decode in-flight — but we
 *      CAN stop consuming deltas and route to the failure
 *      epilogue, which prevents a disconnected client from keeping
 *      the session under the adopt gate's happy path. Once the
 *      native generator exposes an AbortSignal surface this hook
 *      can be upgraded to plumb the controller through; until
 *      then the flag-based opt-out is sufficient to keep the
 *      registry and store in agreement with the client's view.
 *   3. A single `buildFailedTerminal` helper normalizes every
 *      failure path's payload so clients see a consistent envelope:
 *      top-level status=failed, nested items with `in_progress` or
 *      `completed` flipped to `incomplete`, and `incomplete_details`
 *      populated with the specific reason (`error`, `client_abort`,
 *      `stream_exhausted`, `finish_reason_error`, `not_committed`).
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
  httpReq: IncomingMessage | undefined,
  visibility: ResponseVisibility,
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

  // Iter-28 finding 2: fault state. `thrownError` sticks when the
  // underlying async generator throws; `clientAborted` sticks when
  // the HTTP request emits `close`/`error` while we're mid-iteration.
  // Either one diverts the post-loop block to the failure epilogue.
  let thrownError: Error | null = null;
  let clientAborted = false;
  const onClientClose = () => {
    clientAborted = true;
  };
  const onClientError = (_err: unknown) => {
    clientAborted = true;
  };
  if (httpReq) {
    httpReq.once('close', onClientClose);
    httpReq.once('error', onClientError);
  }

  try {
    for await (const event of chatStream) {
      // Iter-28 finding 2: honor a client disconnect at loop-top. The
      // native generator does not yet accept an AbortSignal, so we
      // cannot cancel in-flight decode; the best we can do is stop
      // consuming deltas so the writer does not emit content to a
      // dead socket and the post-loop failure epilogue runs instead
      // of the commit/adopt path. Dropping the generator reference
      // via `break` also triggers the producer's `finally`, which
      // releases any per-model locks and lets the next dispatch in
      // the mutex queue proceed.
      if (clientAborted) break;
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

        // Collect function call items but defer SSE emission until
        // after the commit gate — emitting them inside the done
        // branch would let clients see completed tool calls from a
        // turn the session later refuses to commit (iter-29 finding 1).
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
          outputItems.push(fcItem);
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
  } catch (err: unknown) {
    // Iter-28 finding 2: a mid-decode throw from the underlying async
    // generator (native model crash, tool-call parse throw, etc.)
    // used to escape out into the outer generic handler catch,
    // which sent a JSON error *after* SSE headers had been flushed
    // — producing a partially-streamed response with no terminal
    // event. Capture the error into a sticky flag so the post-loop
    // block below routes the request through the failure epilogue
    // and emits a proper `response.failed` terminal, and so the
    // registry-level `adopt()` gate never sees a committed state
    // for this session.
    thrownError = err instanceof Error ? err : new Error(String(err));
  } finally {
    if (httpReq) {
      httpReq.off('close', onClientClose);
      httpReq.off('error', onClientError);
    }
  }

  // Post-loop terminal emission.
  //
  // The producer's finally has now run (either via the `break` after
  // a done event, via natural iterator exhaustion, via a mid-decode
  // throw surfaced through the try/catch above, or via a client
  // disconnect that flipped `clientAborted`), so `wasCommitted()`
  // reads an authoritative `session.turns` baseline. Four cases:
  //
  //  1. sawDone && committed && !thrownError && !clientAborted:
  //     happy path. Persist the terminal response and emit
  //     `response.completed`. Future `previous_response_id`
  //     continuations can hot-resume through the registry or
  //     cold-replay from the store.
  //  2. sawDone && !committed: the final chunk carried
  //     `finishReason: 'error'` (the ChatSession gates `turnCount`
  //     on a non-error final chunk, so the session never
  //     committed). Route through the failure epilogue with reason
  //     `finish_reason_error`.
  //  3. thrownError != null: the underlying generator threw. Route
  //     through the failure epilogue with reason `error`.
  //  4. clientAborted: HTTP request emitted `close`/`error` mid
  //     stream. Route through the failure epilogue with reason
  //     `client_abort`. We still emit `response.failed` so a tee /
  //     proxy that remains connected sees a terminal event rather
  //     than a hung stream.
  //  5. !sawDone && none of the above: the iterator exhausted
  //     before a terminal chunk arrived. Reason `stream_exhausted`.
  //
  // In all non-committed paths the registry-level `adopt()` gate in
  // `handleCreateResponse` already skipped caching this session, so
  // the in-memory and persisted views agree: there is no authoritative
  // record of this turn anywhere.
  const committed = wasCommitted();
  const successful = sawDone && committed && thrownError == null && !clientAborted;

  if (successful) {
    // `completedResponse` is non-null on the success path (the done
    // branch set it before breaking out of the loop). Assert for the
    // type checker.
    const terminal = completedResponse!;

    // Emit deferred function_call item events. These were collected
    // in the done branch but their SSE emission was held until the
    // commit gate passed, so clients never see completed tool calls
    // from an uncommitted turn (iter-29 finding 1).
    for (const item of terminal.output) {
      if (item.type === 'function_call') {
        const fcIndex = outputItems.indexOf(item);
        writeSSEEvent(res, 'response.output_item.added', { output_index: fcIndex, item });
        const argsStr = item.arguments;
        writeSSEEvent(res, 'response.function_call_arguments.delta', {
          item_id: item.id,
          output_index: fcIndex,
          delta: argsStr,
        });
        writeSSEEvent(res, 'response.function_call_arguments.done', {
          item_id: item.id,
          output_index: fcIndex,
          arguments: argsStr,
        });
        writeSSEEvent(res, 'response.output_item.done', { output_index: fcIndex, item });
      }
    }

    // Persistence is best-effort: the turn is already committed in the
    // session's KV cache, so a store failure must not prevent the client
    // from receiving the terminal `response.completed` event (with its
    // responseId for hot-resume).
    if (store && req.store !== false) {
      try {
        await persistResponse(store, terminal, newInputMessages, previousResponseId, modelInstanceId);
      } catch (err) {
        console.error('[responses] post-commit persistence failed (streaming), terminal will still be emitted:', err);
      }
    }
    writeSSEEvent(res, 'response.completed', { response: terminal });
    // Flip only AFTER the terminal event successfully left the wire.
    // `writeSSEEvent` forwards to `res.write`, which throws
    // synchronously on a dead socket / aborted stream; if the throw
    // escapes here, the outer handler catch must still see
    // `terminalEmitted === false` so it does NOT adopt a session the
    // client never got a terminal event for.
    visibility.terminalEmitted = true;
    endSSE(res);
    return;
  }

  // Failure epilogue.
  //
  // Build the failure terminal through `buildFailedTerminal` so
  // every nested message item is normalized to `status: 'incomplete'`
  // (iter-28 finding 3 — the previous code did `{ ...terminal,
  // status: 'failed' }`, which left nested items marked
  // `completed`/`in_progress` inside a `failed` envelope).
  //
  // Emit `response.output_item.done` for any nested message items
  // that are still dangling (the producer threw before the done
  // branch closed them), so clients that track output_index state
  // see a matching close for each open item BEFORE the terminal
  // `response.failed`. Function-call items are NOT emitted on the
  // failure path — their SSE emission is deferred to the post-commit
  // success path (iter-29 finding 1), so on failure they only exist
  // in outputItems for the terminal payload, normalized to
  // `incomplete` by `buildFailedTerminal`. Reasoning items have no
  // `status` field so they are left untouched.
  const reason: string = thrownError
    ? 'error'
    : clientAborted
      ? 'client_abort'
      : sawDone
        ? 'finish_reason_error'
        : 'stream_exhausted';

  // Build a synthetic usage block when we never reached a done
  // event: no token counts are available. When we DID reach a done
  // event but the session refused to commit, prefer the captured
  // `completedResponse.usage` so clients still see what was spent.
  const usage: ResponseObject['usage'] = completedResponse?.usage ?? {
    input_tokens: 0,
    output_tokens: 0,
    output_tokens_details: { reasoning_tokens: 0 },
    total_tokens: 0,
  };

  const finalOutput = outputItems.filter((_, idx) => idx !== suppressedMessageIndex);

  // Flush still-open message items before the terminal event. A
  // message item is considered still-open if it was started
  // (`hasEmittedMessage && messageItemId != null`) but the done
  // branch never ran (sawDone === false, or sawDone === true but
  // the done branch broke out before emitting the item's close
  // events on the finishReason=error path). We only emit the
  // closing events on the non-sawDone path because the done branch
  // already emits matching closes on the sawDone path before
  // `break` fires.
  if (!sawDone && hasEmittedMessage && messageItemId != null) {
    const miIndex = outputItems.findIndex((i) => i.id === messageItemId);
    writeSSEEvent(res, 'response.output_text.done', {
      item_id: messageItemId,
      output_index: miIndex >= 0 ? miIndex : outputIndex,
      content_index: 0,
      text: messageText,
    });
    const textPart = { type: 'output_text' as const, text: messageText, annotations: [] as never[] };
    writeSSEEvent(res, 'response.content_part.done', {
      item_id: messageItemId,
      output_index: miIndex >= 0 ? miIndex : outputIndex,
      content_index: 0,
      part: textPart,
    });
    const closedMessageItem: MessageOutputItem = {
      id: messageItemId,
      type: 'message',
      role: 'assistant',
      status: 'incomplete',
      content: messageText ? [textPart] : [],
    };
    if (miIndex >= 0) {
      outputItems[miIndex] = closedMessageItem;
      finalOutput[miIndex] = closedMessageItem;
    }
    writeSSEEvent(res, 'response.output_item.done', {
      output_index: miIndex >= 0 ? miIndex : outputIndex,
      item: closedMessageItem,
    });
  }
  if (!sawDone && hasEmittedReasoning && reasoningItemId != null) {
    // Reasoning items have no `status` field; just emit the closing
    // events so output_index bookkeeping stays consistent on the
    // client side. The reasoning item shape is preserved verbatim.
    writeSSEEvent(res, 'response.reasoning_summary_text.done', {
      item_id: reasoningItemId,
      output_index: outputItems.findIndex((i) => i.id === reasoningItemId),
      summary_index: 0,
      text: reasoningText,
    });
    const riIndex = outputItems.findIndex((i) => i.id === reasoningItemId);
    if (riIndex >= 0) {
      const reasoningItem: ReasoningOutputItem = {
        id: reasoningItemId,
        type: 'reasoning',
        summary: [{ type: 'summary_text', text: reasoningText }],
      };
      outputItems[riIndex] = reasoningItem;
      finalOutput[riIndex] = reasoningItem;
      writeSSEEvent(res, 'response.output_item.done', { output_index: riIndex, item: reasoningItem });
    }
  }

  const failedTerminal = buildFailedTerminal(partial, finalOutput, reason, usage);
  writeSSEEvent(res, 'response.failed', { response: failedTerminal });
  // Flip only AFTER `response.failed` successfully left the wire.
  // An SSE write that throws against a dead socket must NOT count
  // as a terminal the client saw; in that case the outer catch
  // rethrows so the request fails loudly instead of silently
  // suppressing an uncommitted error under a responseId no one saw.
  visibility.terminalEmitted = true;
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
 *   - `{ kind: 'absent' }` — the record has a parseable (or empty)
 *     `configJson` blob that simply does not carry a well-formed
 *     `modelInstanceId` field. This is the LEGACY shape written by
 *     branches before iter-21 stamped an explicit instance id into
 *     every row. Iter-28 finding 1: the caller services this shape
 *     by cold-replaying under a narrow "trust on first use"
 *     window — but ONLY when the stored `record.model` friendly
 *     name exactly matches the incoming `body.model`, so a caller
 *     cannot redirect a legacy chain through an unrelated model.
 *     A legacy row whose friendly name differs from the incoming
 *     request is rejected outright.
 *   - `{ kind: 'malformed' }` — the `configJson` blob failed to
 *     JSON-parse. Iter-28 finding 1: the iter-27 legacy compat
 *     path silently classified malformed blobs as `absent`, which
 *     meant the narrow friendly-name-equality check below would
 *     happily cold-replay through a row whose stored config state
 *     we cannot verify at all. Surface the parse failure as a
 *     distinct variant so the caller can reject it with a clean
 *     400 without opening the legacy window.
 */
type StoredModelIdentity = { kind: 'present'; instanceId: number } | { kind: 'absent' } | { kind: 'malformed' };

function readStoredModelIdentity(record: StoredResponseRecord): StoredModelIdentity {
  if (record.configJson == null) return { kind: 'absent' };
  let parsed: { modelInstanceId?: unknown };
  try {
    parsed = JSON.parse(record.configJson) as { modelInstanceId?: unknown };
  } catch {
    return { kind: 'malformed' };
  }
  if (typeof parsed.modelInstanceId === 'number' && Number.isFinite(parsed.modelInstanceId)) {
    return { kind: 'present', instanceId: parsed.modelInstanceId };
  }
  return { kind: 'absent' };
}

// ---------------------------------------------------------------------------
// Public handler
// ---------------------------------------------------------------------------

export async function handleCreateResponse(
  res: ServerResponse,
  body: ResponsesAPIRequest,
  registry: ModelRegistry,
  store: ResponseStore | null,
  httpReq?: IncomingMessage,
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

  // Acquire a dispatch lease on `body.model`'s session-registry
  // binding. The lease keeps the binding (and its FIFO `execLock`
  // mutex chain) alive across every await in this handler — crucial
  // because a concurrent `unregister()` + `register(sameModel)`
  // sequence would otherwise tear the old `SessionRegistry` down and
  // allocate a fresh one, and the new request's `withExclusive`
  // would race against this in-flight dispatch on one shared native
  // model with two independent mutex chains. The lease MUST be
  // released in a `finally` below so the binding's teardown (if
  // deferred by `releaseBinding`) completes once the last dispatch
  // finishes.
  const lease = registry.acquireDispatchLease(body.model);
  if (!lease) {
    sendInternalError(res, 'session registry missing for registered model');
    return;
  }
  const leaseModel = lease.model;
  try {
    // Capture an initial snapshot of the live binding for `body.model`.
    // These values are the INITIAL observation — on a
    // `previous_response_id` continuation we re-read them after
    // `await store.getChain(...)` and reject the request if the
    // binding moved under us (see the hot-swap race guard below).
    // Stateless requests never hit the store so the re-read is a
    // no-op for them.
    const initialSessionReg: SessionRegistry = lease.registry;
    const initialInstanceId: number = lease.instanceId;

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
    // Inherited instructions from the trailing stored chain record —
    // see Finding 4. Null when either the caller supplied their own
    // `body.instructions`, when the continuation has no stored chain,
    // or when the trailing record did not carry an instructions field.
    let inheritedInstructions: string | null = null;

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
        // Legacy-row handling (iter-29 finding 2). Stored rows that
        // lack an explicit `modelInstanceId` are the pre-iter-21
        // shape. The iter-27 compat code serviced them via cold
        // replay gated on friendly-name equality, and iter-28
        // narrowed the window further — but the iter-29 review
        // concluded that friendly-name equality is insufficient:
        // an operator who hot-swaps the same friendly name to a
        // different model during the TTL window can still silently
        // replay through the wrong tokenizer, chat template, or KV
        // layout. Legacy rows are now rejected outright. The
        // 30-minute TTL migration window from iter-27 has expired
        // in any production deployment by now; any remaining legacy
        // rows will flush naturally on TTL expiry.
        const trailingRecord = chain[chain.length - 1]!;
        const storedIdentity = readStoredModelIdentity(trailingRecord);
        if (
          storedIdentity.kind === 'present' &&
          (currentInstanceId === undefined || storedIdentity.instanceId !== currentInstanceId)
        ) {
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
        // Iter-28 finding 1 — malformed configJson.
        //
        // `readStoredModelIdentity` now distinguishes a row with a
        // parseable-but-instance-id-less `configJson` (legacy shape,
        // kind=absent) from a row whose `configJson` failed to
        // JSON-parse (kind=malformed). The iter-27 compat code
        // silently folded both into the `absent` bucket and routed
        // them through the legacy cold-replay path — but a row whose
        // stored config state we cannot even parse has no trustable
        // fields at all. Any cold replay would rebuild the chain
        // against an unreadable prior turn, so reject the request
        // outright with a clean 400 instead of opening the legacy
        // window on a row we cannot verify. An admin tool can purge
        // malformed rows on its own schedule; the endpoint layer
        // does not assume one exists.
        if (storedIdentity.kind === 'malformed') {
          sendBadRequest(
            res,
            `previous_response_id "${body.previous_response_id}" points at a stored record whose ` +
              `configJson blob failed to parse — the server cannot verify the model identity or prior ` +
              `config state it was produced under, so continuing the chain through any model would ` +
              `silently replay against an unreadable prior turn. Start a new chain without ` +
              `previous_response_id.`,
            'previous_response_id',
          );
          return;
        }
        // Iter-29 finding 2 — reject ALL legacy (absent-identity) rows.
        //
        // The iter-28 gate narrowed legacy rows to friendly-name
        // equality, but iter-29 concluded that is still insufficient:
        // an operator hot-swapping the same friendly name to a
        // different model during the TTL window silently replays
        // through the wrong model. Reject outright so the caller
        // must start a fresh chain.
        if (storedIdentity.kind === 'absent') {
          sendBadRequest(
            res,
            `previous_response_id "${body.previous_response_id}" points at a legacy stored record ` +
              `that does not carry a modelInstanceId — the server cannot verify which model instance ` +
              `produced the chain, so continuing it through any model risks silently replaying ` +
              `under the wrong tokenizer, chat template, or KV layout. Start a new chain without ` +
              `previous_response_id.`,
            'previous_response_id',
          );
          return;
        }
        priorMessages = reconstructMessagesFromChain(chain);
        previousResponseId = body.previous_response_id;
        // Inherit the trailing stored record's `instructions` when
        // the continuation request does NOT override it. Finding 4:
        // the iter-25 cold-replay path dropped stored instructions
        // entirely — the caller who originally sent the first turn
        // with `instructions: "You are a pirate"` would see the
        // pirate persona disappear on any cold-replay continuation
        // (TTL expiry, process restart, lease-on-hit miss), because
        // `reconstructMessagesFromChain()` only walked inputJson /
        // outputJson and the endpoint re-read `body.instructions`
        // from the fresh request. An `undefined` body.instructions
        // means "keep the existing system context", not "forget it".
        //
        // The trailing record carries the effective instructions
        // that were in force for that turn (either the caller's
        // original value or a previously inherited one), so reading
        // from it gives us the full prefix state without walking the
        // whole chain. We apply the inheritance only when
        // `body.instructions` is absent — any explicit value (even
        // an empty string) means the caller is deliberately
        // overriding the prefix state, and we surface that change to
        // the `SessionRegistry` cache key below so a hot hit under
        // the stale system context forces a cold replay.
        if (typeof body.instructions !== 'string') {
          const storedInstructions = chain[chain.length - 1]!.instructions;
          if (typeof storedInstructions === 'string' && storedInstructions.length > 0) {
            inheritedInstructions = storedInstructions;
          }
        }
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

    // Compute the effective instructions for this turn. The caller's
    // explicit `body.instructions` wins; otherwise we inherit the
    // trailing stored record's value (Finding 4). The effective
    // value is then used for mapping (prepends the system message
    // via `mapRequest`'s existing logic), for the session-registry
    // cache key (so a hot hit under the stale prefix still matches),
    // for `buildResponseObject` (so the new response roundtrips the
    // prefix), and for persistence (so the next cold replay can
    // re-inherit).
    //
    // We fold the inherited value into a fresh mapped body rather
    // than mutating `body` so the mutation cannot leak to any other
    // code path that still holds the original reference.
    const effectiveInstructions: string | null =
      typeof body.instructions === 'string' ? body.instructions : inheritedInstructions;

    // Map request — full messages include prior + new input.
    // Feed mapRequest the echo-stripped input so no forged function_call
    // item can sneak through into the augmented trailing assistant turn.
    let messages: ChatMessage[];
    let config: ChatConfig;
    const mappedBody: ResponsesAPIRequest =
      effectiveInput === body.input && effectiveInstructions === (body.instructions ?? null)
        ? body
        : {
            ...body,
            input: effectiveInput,
            instructions: effectiveInstructions ?? undefined,
          };
    try {
      ({ messages, config } = mapRequest(mappedBody, priorMessages));
    } catch (err) {
      sendBadRequest(res, err instanceof Error ? err.message : 'Invalid request input', 'input');
      return;
    }

    // Compute the new-only messages (what this request added, excluding prior history
    // and instructions). Instructions are stored separately and should not be persisted
    // as input messages — otherwise chained calls replay stale system messages.
    //
    // Use `mappedBody.instructions` (not `body.instructions`) so an
    // inherited system message also contributes one offset — the
    // reconstruction path prepended it via `mapRequest` above.
    // Mirror `mapRequest`'s truthy check (an empty-string override
    // does NOT push a system message and therefore contributes
    // zero offset, matching the mapper's behavior byte-for-byte).
    const instructionsOffset = mappedBody.instructions ? 1 : 0;
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

    // Extract the EFFECTIVE `instructions` (caller-supplied OR
    // inherited from the trailing stored record; see the block at
    // `effectiveInstructions` above). The session registry uses this
    // as its prefix/system state cache key — a hot hit against a
    // session warmed with different instructions would silently keep
    // using the stale system context, so we pass the effective value
    // to `getOrCreate` and let the registry force a cold replay on
    // mismatch. Inheriting the stored value on a continuation means a
    // cold replay and a warm hit both converge on the SAME prefix
    // state as the original turn, matching what the caller expects
    // when they omit `instructions` on a follow-up request.
    const requestedInstructions: string | null = effectiveInstructions;

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
    // Snapshot the pre-lock binding state. For stateless requests these
    // are `initialSessionReg` / `initialInstanceId` (never updated). For
    // `previous_response_id` continuations they were refreshed by the
    // iter-22 re-read that fires after `store.getChain()`. The in-lock
    // re-check compares against THIS snapshot so the guard catches a
    // hot-swap that lands strictly between the pre-lock read and the
    // moment this waiter wins the mutex.
    const preLockSessionReg = sessionReg;
    const preLockInstanceId = currentInstanceId;

    await sessionReg.withExclusive(async () => {
      // Hot-swap race guard inside the mutex.
      //
      // `withExclusive` can park this waiter behind a long-running
      // dispatch on the same model, and `ModelRegistry.register()` is
      // NOT coordinated with that lock — a concurrent
      // `registry.register(body.model, newModel)` can re-point the
      // friendly name while we are parked. Without this in-lock re-read
      // the closure would still lease a session out of the already-
      // captured `preLockSessionReg`, adopt under the dead
      // `preLockInstanceId`, and persist the new chain under a binding
      // that `body.model` no longer resolves to. The iter-22 pre-lock
      // re-read only covered the `store.getChain()` await window; the
      // mutex-wait window is strictly later and equally unsafe.
      //
      // Compare the live binding to the pre-lock snapshot (captured
      // just before entering the mutex — already iter-22-refreshed on
      // the continuation path, identical to the handler-top snapshot
      // on the stateless path). Any drift — nullable or value — is
      // fatal and rejected with the same 400 envelope the iter-22
      // guard uses, so clients see a consistent "binding changed"
      // error regardless of which await window caught the race.
      const lockedSessionReg = registry.getSessionRegistry(body.model);
      const lockedInstanceId = registry.getInstanceId(body.model);
      if (
        lockedSessionReg === undefined ||
        lockedInstanceId === undefined ||
        lockedSessionReg !== preLockSessionReg ||
        lockedInstanceId !== preLockInstanceId
      ) {
        sendBadRequest(
          res,
          `Model "${body.model}" binding changed while the request was queued behind the per-model ` +
            `execution mutex. A concurrent register() re-pointed the name at a different model instance ` +
            `(or released it entirely) while this waiter was parked, so the session registry and instance ` +
            `id captured before the mutex wait no longer match the live binding. Dispatching anyway would ` +
            `route the request through the wrong model — priming, decoding, and persisting under a dead ` +
            `binding. Retry the request — if the swap was intentional, the new binding will service the ` +
            `retry cleanly.`,
          'model',
        );
        return;
      }

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
        // Pass `mappedBody` (not the raw `body`) so the response
        // object and the persisted record carry the EFFECTIVE
        // instructions, including any value inherited from the
        // trailing stored record via Finding 4. Using `body` here
        // would re-drop the inherited value on the wire — the
        // client's response would report `instructions: null` even
        // though the turn was run against the inherited system
        // context, and the next cold replay would have nothing to
        // re-inherit from.
        // Wrap the handler call in its own try/catch so that a
        // post-commit persistence failure does not prevent adopt.
        // Post-commit store failures are caught inside the handlers
        // themselves (handleNonStreaming / handleStreamingNative) and
        // demoted to log-only. A handlerError at this level therefore
        // comes from non-persistence failures (response construction,
        // SSE write, res.writeHead/end crash).
        //
        // Iter-32 finding 1 & 2: the adopt/rethrow decision used to
        // key on `res.headersSent`, which is a LIE for "the client
        // received the response". Node's `ServerResponse.writeHead()`
        // flips `headersSent = true` synchronously before any body
        // bytes leave the buffer, so a throw from `res.end()` on the
        // non-streaming path (or a throw from `writeSSEEvent` before
        // any terminal SSE event on the streaming path) looked like
        // the happy "already on the wire" case under the old gate
        // and silently adopted / swallowed the error. The handlers
        // now flip explicit visibility flags (`responseBodyWritten`
        // / `terminalEmitted`) strictly AFTER the terminal artefact
        // the client depends on is known to have left the wire, and
        // the gate below keys on those flags instead.
        let handlerError: Error | null = null;
        const visibility: ResponseVisibility = {
          responseBodyWritten: false,
          terminalEmitted: false,
        };

        if (mappedBody.stream) {
          const outcome = await runSessionStreaming(session, messages, newInputMessages, config);
          const streamingWasCommitted = () => outcome.wasCommitted();
          try {
            await handleStreamingNative(
              res,
              outcome.stream,
              mappedBody,
              responseId,
              previousResponseId,
              store,
              newInputMessages,
              streamingWasCommitted,
              currentInstanceId,
              httpReq,
              visibility,
            );
          } catch (err) {
            handlerError = err instanceof Error ? err : new Error(String(err));
          }
          committed = streamingWasCommitted();
        } else {
          const outcome = await runSessionNonStreaming(session, messages, newInputMessages, config);
          try {
            await handleNonStreaming(
              res,
              outcome.result,
              mappedBody,
              responseId,
              previousResponseId,
              store,
              newInputMessages,
              currentInstanceId,
              visibility,
            );
          } catch (err) {
            handlerError = err instanceof Error ? err : new Error(String(err));
          }
          committed = outcome.committed;
        }

        // "Safe to suppress" collapses to: did the client observe a
        // terminal artefact for this responseId? On the non-
        // streaming path that is the JSON body landing cleanly on
        // the wire; on the streaming path it is a terminal SSE
        // event (`response.completed` or `response.failed`) landing
        // cleanly on the wire. In either case the client can see
        // the responseId and knows the turn is over, so adopting
        // the committed session under that id is safe and
        // swallowing the (already-surfaced-via-failed-event)
        // handler error is the only option that does not produce a
        // malformed double-response.
        const safeToSuppress = visibility.responseBodyWritten || visibility.terminalEmitted;

        if (previousResponseId) {
          sessionReg.drop(previousResponseId);
        }
        // Only adopt if the turn committed AND either the handler
        // succeeded or a terminal artefact is already on the wire.
        // A committed turn whose handler threw before the client
        // saw anything it can chain off of must NOT be adopted —
        // the responseId is unreachable from the client, so caching
        // the session under it creates a permanently dangling warm
        // session.
        if (committed && (handlerError == null || safeToSuppress)) {
          sessionReg.adopt(responseId, session, requestedInstructions);
        }

        // Rethrow handler errors when the client hasn't seen a
        // terminal yet, regardless of commit state. The outer
        // catch will send a proper 500 (non-streaming) or a last-
        // ditch SSE `error` event (streaming, after `beginSSE` but
        // before any terminal). Without this the request would
        // hang from the client's perspective.
        if (handlerError && !safeToSuppress) {
          throw handlerError;
        }
        // If a terminal is on the wire but the handler still
        // threw: log only. Rethrowing would produce a malformed
        // double-response; the client already has a terminal event
        // it can parse.
        if (handlerError) {
          console.error('[responses] handler error after terminal response already delivered:', handlerError);
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
  } finally {
    // Release the dispatch lease on the ORIGINAL model object the
    // lease was acquired against (not a re-read of `body.model`,
    // which may have been hot-swapped while we held the mutex). A
    // pending teardown — `releaseBinding()` called concurrently
    // while this dispatch held the lease — finalises here once the
    // in-flight counter drops to zero.
    registry.releaseDispatchLease(leaseModel);
  }
}
