/**
 * POST /v1/responses — OpenAI Responses API, streaming (SSE) and non-streaming (JSON).
 *
 * Dispatches to loaded models via `ModelRegistry`. Inference goes through a per-model
 * `ChatSession` looked up by `previous_response_id` in the model's `SessionRegistry`: a
 * hit reuses the live KV cache (`send` / `sendStream` / `sendToolResult`); a miss
 * reconstructs the full conversation from `ResponseStore` and cold-replays via
 * `primeHistory` + `startFromHistory[Stream]`.
 */

import { randomUUID } from 'node:crypto';
import type { IncomingMessage, ServerResponse } from 'node:http';

import type { ChatConfig, ChatMessage, ChatResult, ResponseStore, StoredResponseRecord } from '@mlx-node/core';
import type { ChatSession, ChatStreamEvent, SessionCapableModel } from '@mlx-node/lm';

import { sendBadRequest, sendInternalError, sendNotFound, sendRateLimit, sendStorageTimeout } from '../errors.js';
import { mapRequest, reconstructMessagesFromChain } from '../mappers/request.js';
import {
  buildPartialResponse,
  buildResponseObject,
  computeOutputText,
  genId,
  mapFinishReasonToStatus,
} from '../mappers/response.js';
import { getPendingWritesFor } from '../pending-writes.js';
import type { ModelRegistry } from '../registry.js';
import { QueueFullError, type SessionRegistry } from '../session-registry.js';
import { beginSSE, endSSE, writeSSEEvent } from '../streaming.js';
import { ToolCallTagBuffer } from '../tool-call-buffer.js';
import {
  createVisibility,
  endJson,
  flushTerminalSSE,
  markSSEMode,
  type TransportVisibility,
  writeFallbackErrorSSE,
} from '../transport-visibility.js';
import type {
  FunctionCallOutputItem,
  MessageOutputItem,
  OutputItem,
  ReasoningOutputItem,
  ResponseObject,
  ResponsesAPIRequest,
} from '../types.js';

/**
 * Fallback retention for stored response rows when no explicit
 * `responseRetentionSec` is threaded in. Production wires retention via
 * `ServerConfig.responseRetentionSec` (default 7 days, see `server.ts`);
 * this 30-minute fallback is only used by legacy direct-invocation callers.
 */
const RESPONSE_TTL_SECONDS = 1800;

/**
 * Value of the `X-Session-Cache` response header emitted on every
 * `/v1/responses` and `/v1/messages` response. Advertises whether the
 * request warm-hit the per-model `SessionRegistry` (`hit`), missed and
 * cold-replayed from the stored chain (`cold_replay`), or started a
 * fresh session without a `previous_response_id` (`fresh`). The literal
 * string values are load-bearing — clients and operator tooling pin on
 * them.
 */
export type SessionCacheStatus = 'hit' | 'cold_replay' | 'fresh';

/**
 * Upper bound (ms) on how long the recovery path waits for an in-flight
 * `store.store(...)` to land. On timeout we re-probe `getChain` once to
 * catch a late-landing write, then surface HTTP 503 (retryable) rather
 * than 404 (permanent). Default 2000ms — short enough to fail fast on a
 * wedged backend, long enough that healthy SQLite writes complete well
 * within it. Override via `MLX_CHAIN_WRITE_WAIT_TIMEOUT_MS`.
 */
function getChainWriteWaitTimeoutMs(): number {
  const raw = process.env.MLX_CHAIN_WRITE_WAIT_TIMEOUT_MS;
  if (raw == null || raw === '') return 2000;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return 2000;
  return parsed;
}

/**
 * Soft timeout (ms) on how long the outer handler awaits the off-lock
 * `store.store(...)` before detaching and letting the write run in the
 * background. The pending-writes tracker still holds a reference so
 * chained continuations can observe it. Default 5000ms (larger than the
 * chain-write wait because this bound is not client-facing — the client
 * already has its terminal response). Override via
 * `MLX_POST_COMMIT_PERSIST_TIMEOUT_MS`.
 */
function getPostCommitPersistTimeoutMs(): number {
  const raw = process.env.MLX_POST_COMMIT_PERSIST_TIMEOUT_MS;
  if (raw == null || raw === '') return 5000;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return 5000;
  return parsed;
}

/**
 * Hard timeout (ms) for the off-lock post-commit persist — the
 * second-stage breaker that force-releases the `retainBinding` paired
 * with `initiatePersist` when the write is truly wedged (never settles).
 *
 * The soft persist timeout above only detaches the handler; the retain
 * stays pinned so a slow-but-eventual write still lands against the
 * live `modelInstanceId`. This hard breaker bounds the leak for a
 * genuinely wedged promise at this value instead of process lifetime.
 * On fire, it also retires the instance id via a refcounted tombstone
 * so a same-object re-registration inherits the id and the late write
 * remains chainable — a true hot-swap to a different object still
 * mints a fresh id and correctly fails stale chains with 400.
 *
 * Default 60000ms — well past the soft timeout so slow-but-eventual
 * writes are unaffected. Override via `MLX_POST_COMMIT_PERSIST_HARD_TIMEOUT_MS`:
 * empty/whitespace-only falls back to default (so a config-templating
 * typo cannot silently disable the breaker); `'0'` explicitly disables;
 * non-numeric garbage falls back to default. Exported for unit tests.
 */
export function getPostCommitPersistHardTimeoutMs(): number {
  const raw = process.env.MLX_POST_COMMIT_PERSIST_HARD_TIMEOUT_MS;
  const normalized = raw?.trim();
  if (normalized == null || normalized === '') return 60_000;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 60_000;
}

/**
 * TTL (ms) for hard-timed-out markers in the per-store pending-writes
 * tracker. See `pending-writes.ts` for the full lifetime model.
 *
 * An independent TTL with lazy expiry on read bounds marker memory at
 * O(requestRate × TTL) even when the underlying wedged writes never
 * settle (and their `.finally(...)` cleanup therefore never fires).
 * Default 300000ms (5 min) — past this, the best-effort persist
 * contract has long since failed and permanent 404 is the correct
 * eventual outcome. Override via `MLX_HARD_TIMEOUT_MARKER_TTL_MS`
 * (same parse semantics as the hard-timeout env var above). Exported
 * for unit tests.
 */
export function getHardTimedOutMarkerTtlMs(): number {
  const raw = process.env.MLX_HARD_TIMEOUT_MARKER_TTL_MS;
  const normalized = raw?.trim();
  if (normalized == null || normalized === '') return 300_000;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 300_000;
}

/**
 * Per-process boot id stamped into every stored response row's
 * `configJson` alongside `modelInstanceId`. The pair enables
 * restart-safe chain continuation while preserving the in-process
 * hot-swap guard:
 *
 *   - stored `serverBootId` == live boot id AND `modelInstanceId`
 *     matches live  → strict hit (in-process hot-swap protection).
 *   - stored `serverBootId` != live boot id (or missing, i.e. rows
 *     written before this field existed)  → cross-restart. The
 *     stored `modelInstanceId` belongs to a dead process and is
 *     meaningless, so the instance-id check is skipped and the
 *     continuation falls back to name-based resume through whatever
 *     model is currently bound to the requested name.
 *   - stored `configJson` malformed  → reject.
 *   - stored row has neither `serverBootId` NOR `modelInstanceId`
 *     (truly legacy, pre-instance-id)  → reject.
 *
 * `getServerBootId()` resolves lazily on every call so tests can
 * install a deterministic boot id via `__setServerBootIdForTesting`
 * before exercising either the persistence or validation path.
 */
let serverBootId: string = randomUUID();

export function getServerBootId(): string {
  return serverBootId;
}

export function __setServerBootIdForTesting(id: string): void {
  serverBootId = id;
}

// ---------------------------------------------------------------------------
// Non-streaming path
// ---------------------------------------------------------------------------

/**
 * Outcome of the non-streaming handler. The outer handler persists
 * `response` AFTER releasing the per-model mutex — keeping persistence
 * off the critical path so a slow store does not pin the next waiter.
 */
interface NonStreamingHandlerOutcome {
  response: ResponseObject;
}

async function handleNonStreaming(
  res: ServerResponse,
  result: ChatResult,
  req: ResponsesAPIRequest,
  responseId: string,
  previousResponseId: string | undefined,
  visibility: TransportVisibility,
): Promise<NonStreamingHandlerOutcome> {
  const response = buildResponseObject(result, req, responseId, previousResponseId);

  // `chatSession*` has no AbortSignal surface yet, so a mid-decode
  // client disconnect still burns the full decode budget — peer loss
  // is only observable when native decode resolves. Disconnect
  // detection is delegated to `endJson`'s `isSocketGone(res)` check:
  // on a dead peer it rejects AFTER committing `responseMode = 'json'`
  // so the outer catch routes to the JSON error / socket-destroy
  // shape; `responseBodyWritten` flips only from `res.end`'s write
  // callback (proving the kernel accepted the chunk) so the adopt
  // gate refuses to cache the session under an unreachable responseId.
  await endJson(res, JSON.stringify(response), visibility);
  return { response };
}

// ---------------------------------------------------------------------------
// Streaming path
// ---------------------------------------------------------------------------

/**
 * Build a failure terminal `ResponseObject`: `status: 'failed'`,
 * `incomplete_details: { reason }`, and every nested message /
 * function_call item with `status` `in_progress` or `completed`
 * normalized to `incomplete` so a client inspecting `response.output`
 * on a failed envelope cannot see success-shaped items inside it.
 * `ReasoningOutputItem` has no `status` field and is left alone.
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
 * Outcome of the streaming handler.
 *
 * `terminalToPersist` is non-null only on the committed-success path
 * (the outer handler writes it to the `ResponseStore` after releasing
 * the per-model mutex). Every failure path leaves it null — the turn
 * never committed, so there is nothing authoritative to persist or
 * cold-replay.
 *
 * `failureMode` carries the reason out to the adopt gate, which must
 * refuse to cache under an unreachable responseId — in particular, a
 * `res.close` that fires AFTER the final chunk can commit the session
 * while the client will never chain off that id, so the gate keys on
 * `failureMode === null`, not on `committed` alone.
 */
interface StreamingHandlerOutcome {
  terminalToPersist: ResponseObject | null;
  failureMode: 'client_abort' | 'error' | 'finish_reason_error' | 'stream_exhausted' | null;
}

async function handleStreamingNative(
  res: ServerResponse,
  chatStream: AsyncGenerator<ChatStreamEvent>,
  req: ResponsesAPIRequest,
  responseId: string,
  previousResponseId: string | undefined,
  wasCommitted: () => boolean,
  httpReq: IncomingMessage | undefined,
  visibility: TransportVisibility,
): Promise<StreamingHandlerOutcome> {
  beginSSE(res);
  // Commit to SSE wire format synchronously so the outer catch
  // branches on `responseMode` (not `headersSent`) and routes an
  // early `writeSSEEvent` failure to the streaming error epilogue
  // instead of corrupting the JSON path.
  markSSEMode(visibility);

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

  // Terminal response is captured in the done branch but emitted AFTER
  // the loop drains — `wasCommitted()` only reads authoritative
  // `session.turns` once the producer's finally has run.
  let completedResponse: ResponseObject | null = null;
  let sawDone = false;

  // Fault state. `thrownError` sticks on a generator throw;
  // `clientAborted` sticks on any `close`/`error` from `httpReq`, `res`,
  // or `res.socket`. Either flips the post-loop block to the failure
  // epilogue. Listening on `res` and `res.socket` matters because
  // non-terminal SSE writes can silently "succeed" on a dead socket.
  let thrownError: Error | null = null;
  let clientAborted = false;
  const onClientClose = () => {
    clientAborted = true;
  };
  const onClientError = (_err: unknown) => {
    clientAborted = true;
  };
  const onResClose = () => {
    clientAborted = true;
  };
  const onResError = (_err: unknown) => {
    clientAborted = true;
  };
  const resSocketForAbort = res.socket;
  if (httpReq) {
    httpReq.once('close', onClientClose);
    httpReq.once('error', onClientError);
  }
  res.once('close', onResClose);
  res.once('error', onResError);
  if (resSocketForAbort != null) {
    resSocketForAbort.once('close', onResClose);
  }

  try {
    for await (const event of chatStream) {
      // Honor client disconnect at loop-top. Native decode has no
      // AbortSignal yet; `break` drops the generator reference so
      // the producer's `finally` releases per-model locks and the
      // post-loop block routes to the failure epilogue.
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

        // Collect function_call items but defer SSE emission until
        // after the commit gate — otherwise clients can see completed
        // tool calls from a turn the session later refuses to commit.
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

        // Build the terminal but do NOT emit `response.completed` yet:
        // commit signal only becomes authoritative after the producer's
        // finally runs. Break so for-await cleanup triggers that finally,
        // then the post-loop block handles emission + persistence.
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
    // Capture mid-decode throws so the post-loop block routes to the
    // failure epilogue and emits `response.failed` — otherwise the
    // error would escape into the outer JSON error path with SSE
    // headers already on the wire.
    thrownError = err instanceof Error ? err : new Error(String(err));
  } finally {
    if (httpReq) {
      httpReq.off('close', onClientClose);
      httpReq.off('error', onClientError);
    }
    res.off('close', onResClose);
    res.off('error', onResError);
    if (resSocketForAbort != null) {
      resSocketForAbort.off('close', onResClose);
    }
  }

  // Post-loop terminal emission. The producer's finally has run so
  // `wasCommitted()` reads an authoritative baseline. On success emit
  // `response.completed`; otherwise route through the failure epilogue
  // with one of `finish_reason_error` / `error` / `client_abort` /
  // `stream_exhausted`. `response.failed` is emitted even on
  // `client_abort` so a tee/proxy that stays connected sees a terminal.
  const committed = wasCommitted();
  const successful = sawDone && committed && thrownError == null && !clientAborted;

  if (successful) {
    const terminal = completedResponse!;

    // Emit deferred function_call events now that the commit gate
    // passed — held until here so clients never see completed tool
    // calls from an uncommitted turn.
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

    // The terminal SSE flushes inside the per-model mutex (client
    // expects it ordered against prior deltas); the `ResponseStore`
    // write is deferred to the outer handler so a slow SQLite write
    // does not pin the next waiter. `flushTerminalSSE` flips
    // `terminalEmitted` only once the kernel acks the frame — a
    // callback-reported error rejects so the outer catch refuses to
    // adopt under an unseen responseId.
    await flushTerminalSSE(res, 'response.completed', { response: terminal }, visibility);
    endSSE(res);
    return { terminalToPersist: terminal, failureMode: null };
  }

  // Failure epilogue. Close any dangling message items BEFORE the
  // terminal so clients tracking `output_index` see matching closes.
  // Function_call items are never emitted on failure (their SSE is
  // deferred to the success path); reasoning items have no `status`.
  const reason: 'error' | 'client_abort' | 'finish_reason_error' | 'stream_exhausted' = thrownError
    ? 'error'
    : clientAborted
      ? 'client_abort'
      : sawDone
        ? 'finish_reason_error'
        : 'stream_exhausted';

  // Prefer captured usage on a finish_reason_error path so clients
  // still see what was spent; synthesize zero-usage only when no done
  // event was ever observed.
  const usage: ResponseObject['usage'] = completedResponse?.usage ?? {
    input_tokens: 0,
    output_tokens: 0,
    output_tokens_details: { reasoning_tokens: 0 },
    total_tokens: 0,
  };

  const finalOutput = outputItems.filter((_, idx) => idx !== suppressedMessageIndex);

  // Flush still-open message items before the terminal. Only on the
  // non-sawDone path — the done branch emits its own closes before
  // breaking out.
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
    // No `status` field on reasoning items — just emit closes so
    // client-side output_index bookkeeping stays consistent.
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
  await flushTerminalSSE(res, 'response.failed', { response: failedTerminal }, visibility);
  endSSE(res);
  // No terminalToPersist on an uncommitted turn: a later continuation
  // that cold-replayed this record would silently resurrect failed
  // output as authoritative history.
  return { terminalToPersist: null, failureMode: reason };
}

// ---------------------------------------------------------------------------
// Session routing
// ---------------------------------------------------------------------------

/**
 * Return the ordered sibling call ids for the trailing assistant
 * fan-out (if any calls remain unresolved), else `null`. MUST be
 * invoked on the STORED prior chain, never on the augmented `messages`
 * list — otherwise an echoed `function_call` could overwrite the
 * trailing assistant with a forged single-call turn.
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
 * Set of `call_id`s owned by the trailing assistant turn, used to
 * authenticate echoed `function_call` items in a `previous_response_id`
 * continuation. Ownership check only — `name` / `arguments` are not
 * compared against the stored payload (clients commonly reserialize
 * their own arguments with different whitespace). Returns `null` when
 * the trailing message is not an assistant fan-out.
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
 * Reorder tool messages in `messages[startOffset, blockEnd)` to match
 * `expectedOrder`. Replay correctness for a multi-call fan-out depends
 * on POSITION — several native backends drop the id on the wire and
 * pair results to calls by sibling index, so a reordered submission
 * would silently bind results to the wrong calls even after the
 * id-set gate passes.
 *
 * `blockEnd` MUST be sized to a single contiguous tool block; the
 * full-history walker computes one per fan-out. No-op when any
 * precondition fails.
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
 * Walk the full `messages` history, validate each assistant fan-out's
 * tool-result block, and canonicalize each block to sibling order in
 * place. Invoked on stateless cold-start histories and on the
 * Anthropic `/v1/messages` endpoint (both feed caller-supplied tool
 * order straight into `primeHistory()` without the continuation gate).
 *
 * Validation rejects: orphan tool messages, unknown `toolCallId`s,
 * missing/duplicate resolutions, and a trailing unresolved fan-out in
 * a stateless history. Returns `null` on success or a human-readable
 * error string (sent as 400 `invalid_request_error`).
 *
 * @param apiSurface controls error-string vocabulary (`openai` default
 *   uses `function_call_output` / `call_id`; `anthropic` uses
 *   `tool_result` / `tool_use_id`). Validation logic is identical.
 */
export function validateAndCanonicalizeHistoryToolOrder(
  messages: ChatMessage[],
  apiSurface: 'openai' | 'anthropic' = 'openai',
): string | null {
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
      // Trailing unresolved fan-out is rejected — a stateless history
      // has nothing for the model to continue from. Mid-history the
      // next non-tool turn orphans the fan-out.
      if (blockEnd === messages.length) {
        return (
          `${vocab.fanOut} at index ${i} is the trailing turn of the history but has no ` +
          `${vocab.toolResult} resolutions. A stateless cold-start history cannot end on an ` +
          `unresolved tool-call fan-out because there is nothing for the model to continue from.`
        );
      }
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
    // blockLength > declaredIds.length is impossible (every id is in
    // declaredSet and seenInBlock dedupes).

    canonicalizeToolMessageOrder(messages, blockStart, blockEnd, declaredIds);
    i = blockEnd;
  }

  return null;
}

/**
 * Non-streaming dispatch outcome. `committed` is measured against a
 * baseline captured AFTER any internal `session.reset()` so it is
 * honest across the multi-message reset-and-restart branch — a
 * pre-helper snapshot would be stale. Uncommitted dispatches must
 * never be adopted: their KV state is out of sync with persistence.
 */
interface NonStreamingOutcome {
  result: ChatResult;
  committed: boolean;
}

/** Streaming dispatch outcome. `wasCommitted()` is valid only AFTER
 *  the SSE writer has drained the stream. */
interface StreamingOutcome {
  stream: AsyncGenerator<ChatStreamEvent>;
  wasCommitted(): boolean;
}

/**
 * Route a non-streaming request through `ChatSession`. Cold path
 * (fresh session) runs `primeHistory` + `startFromHistory`; hot path
 * uses `send` / `sendToolResult` for a single new message, or falls
 * back to reset + cold re-prime on multi-message input. The caller
 * is responsible for rejecting partial tool-result submissions
 * against a fan-out (`handleCreateResponse` fan-out gate).
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
  // Single-message continuations whose role is `user` or `tool` take
  // the cheap delta paths (`send` / `sendToolResult`). Any other single
  // role (`assistant`, `system`) is still accepted by `mapRequest` —
  // `reconstructMessagesFromChain` + `primeHistory` tolerate a tail of
  // either — but the chat-session delta API has no entry point for
  // them, so fall through to reset + cold re-prime against the fully
  // rebuilt history. Returning 500 here would regress the pre-session-
  // API full-history path, making valid continuation payloads fail
  // nondeterministically based on cache state.
  if (newInputMessages.length === 1) {
    const last = newInputMessages[0]!;
    if (last.role === 'user') {
      const initialTurns = session.turns;
      const images = last.images ?? undefined;
      const result = await session.send(last.content, images ? { images, config } : { config });
      return { result, committed: session.turns > initialTurns };
    }
    if (last.role === 'tool') {
      if (!last.toolCallId) {
        throw new Error('tool message missing toolCallId');
      }
      const initialTurns = session.turns;
      const result = await session.sendToolResult(last.toolCallId, last.content, { config });
      return { result, committed: session.turns > initialTurns };
    }
    // Non-user / non-tool single-message continuation (assistant /
    // system) falls through to the multi-message reset + cold re-prime
    // branch below.
  }

  // Multi-message (or single non-user/non-tool) hot path: reset + cold
  // re-prime. `initialTurns` MUST be captured AFTER `session.reset()`
  // zeroes `turns`, otherwise the committed check reads stale.
  // Amortized: the caller re-keys this session under the new
  // responseId on success.
  await session.reset();
  session.primeHistory(messages);
  const initialTurns = session.turns;
  const result = await session.startFromHistory(config);
  return { result, committed: session.turns > initialTurns };
}

/** Streaming counterpart to {@link runSessionNonStreaming}. */
async function runSessionStreaming(
  session: ChatSession<SessionCapableModel>,
  messages: ChatMessage[],
  newInputMessages: ChatMessage[],
  config: ChatConfig,
  signal: AbortSignal | undefined,
): Promise<StreamingOutcome> {
  if (session.turns === 0) {
    session.primeHistory(messages);
    const initialTurns = session.turns;
    return {
      stream: session.startFromHistoryStream(config, signal),
      wasCommitted: () => session.turns > initialTurns,
    };
  }

  // See {@link runSessionNonStreaming} for the routing contract. A
  // single assistant/system continuation falls through to the
  // multi-message reset + cold re-prime branch below rather than
  // crashing with 500.
  if (newInputMessages.length === 1) {
    const last = newInputMessages[0]!;
    if (last.role === 'user') {
      const initialTurns = session.turns;
      const images = last.images ?? undefined;
      return {
        stream: session.sendStream(last.content, images ? { images, config, signal } : { config, signal }),
        wasCommitted: () => session.turns > initialTurns,
      };
    }
    if (last.role === 'tool') {
      if (!last.toolCallId) {
        throw new Error('tool message missing toolCallId');
      }
      const initialTurns = session.turns;
      return {
        stream: session.sendToolResultStream(last.toolCallId, last.content, { config, signal }),
        wasCommitted: () => session.turns > initialTurns,
      };
    }
    // Non-user / non-tool single-message continuation falls through to
    // the reset + cold re-prime branch below.
  }

  // Multi-message (or single non-user/non-tool) hot path: same reset +
  // cold re-prime as the non-streaming variant. `initialTurns` must be
  // captured AFTER reset.
  await session.reset();
  session.primeHistory(messages);
  const initialTurns = session.turns;
  return {
    stream: session.startFromHistoryStream(config, signal),
    wasCommitted: () => session.turns > initialTurns,
  };
}

// ---------------------------------------------------------------------------
// Storage helper
// ---------------------------------------------------------------------------

/**
 * Build the `StoredResponseRecord` for a committed response. Pure
 * function, split out from `initiatePersist` so the caller can build
 * the record synchronously inside `withExclusive`, register the
 * in-flight write in the tracker before the mutex releases, and await
 * off-lock purely for error logging. See `pending-writes.ts` for the
 * tracker contract.
 *
 * Only NEW input messages are stored — chain reconstruction re-derives
 * full history via `previous_response_id` links. `modelInstanceId` is
 * stashed in `configJson` (leaving the Rust-side schema untouched)
 * alongside `serverBootId` so `readStoredModelIdentity` can distinguish
 * a live in-process hot-swap (strict instance-id guard) from a cross-
 * restart resume (skip the instance-id guard, fall back to name-based
 * resume against whatever model is currently bound).
 */
function buildResponseRecord(
  response: ResponseObject,
  newInputMessages: ChatMessage[],
  previousResponseId: string | undefined,
  modelInstanceId: number | undefined,
  retentionSec?: number,
): StoredResponseRecord {
  // Retention is decoupled from the warm `SessionRegistry` TTL (30 min
  // KV cache) — the row must outlive the session so a later cold
  // replay can rebuild from SQLite. Default 7 days via `createServer`.
  const effectiveRetention =
    retentionSec != null && Number.isFinite(retentionSec) && retentionSec > 0 ? retentionSec : RESPONSE_TTL_SECONDS;
  return {
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
      serverBootId: getServerBootId(),
    }),
    expiresAt: Math.floor(Date.now() / 1000) + effectiveRetention,
  };
}

/**
 * Kick off an off-lock `store.store(record)` write and register it in
 * the per-store pending-write tracker. MUST be called synchronously
 * inside `withExclusive` so the tracker registration happens before
 * the mutex releases — a back-to-back continuation that slips in
 * observes the in-flight write via `awaitPending(previous_response_id)`
 * and retries `getChain` rather than 404-ing on a fresh responseId.
 *
 * `absoluteExpiresAtMs` = min(record expiry, chain earliest expiry) —
 * once crossed, the `awaitPending` path can short-circuit to 404
 * rather than keep emitting retryable 503 for an unrecoverable chain.
 *
 * Caller awaits the returned promise off-lock purely for error logging.
 */
function initiatePersist(
  store: ResponseStore,
  record: StoredResponseRecord,
  absoluteExpiresAtMs?: number,
): Promise<void> {
  const writePromise = store.store(record);
  getPendingWritesFor(store).track(record.id, writePromise, absoluteExpiresAtMs);
  return writePromise;
}

/**
 * Identity signal from a stored record's `configJson` blob:
 *   - `present` (well-formed `modelInstanceId`; `bootId` may be
 *     `undefined` for rows written before `serverBootId` was added)
 *   - `absent` (truly legacy row with neither `modelInstanceId` nor
 *     `serverBootId` — rejected outright, cannot verify anything)
 *   - `malformed` (blob failed to JSON-parse — rejected with 400)
 *
 * The boot id is threaded through so the validation block can
 * distinguish same-process hot-swap (strict instance-id check) from
 * cross-restart resume (skip instance-id check, name-based only).
 */
type StoredModelIdentity =
  | { kind: 'present'; instanceId: number; bootId: string | undefined }
  | { kind: 'absent' }
  | { kind: 'malformed' };

function readStoredModelIdentity(record: StoredResponseRecord): StoredModelIdentity {
  if (record.configJson == null) return { kind: 'absent' };
  let parsed: { modelInstanceId?: unknown; serverBootId?: unknown };
  try {
    parsed = JSON.parse(record.configJson) as { modelInstanceId?: unknown; serverBootId?: unknown };
  } catch {
    return { kind: 'malformed' };
  }
  const bootId =
    typeof parsed.serverBootId === 'string' && parsed.serverBootId.length > 0 ? parsed.serverBootId : undefined;
  if (typeof parsed.modelInstanceId === 'number' && Number.isFinite(parsed.modelInstanceId)) {
    return { kind: 'present', instanceId: parsed.modelInstanceId, bootId };
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
  responseRetentionSec?: number,
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

  // Per-request retention override: `metadata.retention_seconds` lets a
  // client pin a single row to a longer (VIP / onboarding) or shorter
  // (one-shot PII) lifetime than the server-wide default. Bounds
  // `[60, 90 * 86400]` cap runaway retention and bound operator disk
  // use; `null` / `undefined` / missing → fall through to
  // `responseRetentionSec`. The error message is exact — clients parse
  // on it.
  let requestedRetentionSec: number | undefined;
  if (body.metadata != null && typeof body.metadata === 'object') {
    const raw = (body.metadata as { retention_seconds?: unknown }).retention_seconds;
    if (raw != null) {
      const RETENTION_MIN = 60;
      const RETENTION_MAX = 90 * 86400; // 7_776_000
      if (
        typeof raw !== 'number' ||
        !Number.isFinite(raw) ||
        !Number.isInteger(raw) ||
        raw < RETENTION_MIN ||
        raw > RETENTION_MAX
      ) {
        sendBadRequest(
          res,
          'metadata.retention_seconds must be an integer in [60, 7776000]',
          'metadata.retention_seconds',
        );
        return;
      }
      requestedRetentionSec = raw;
    }
  }
  const effectiveRetentionSec = requestedRetentionSec ?? responseRetentionSec;

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

  // Dispatch lease keeps the binding (and its FIFO `execLock` chain)
  // alive across every await in this handler — required because a
  // concurrent `unregister()` + `register(sameModel)` would otherwise
  // allocate a fresh `SessionRegistry` and race two independent mutex
  // chains against one native model. Released in `finally` below.
  const lease = registry.acquireDispatchLease(body.model);
  if (!lease) {
    sendInternalError(res, 'session registry missing for registered model');
    return;
  }
  const leaseModel = lease.model;
  // AbortController wired to disconnect events, declared at handler
  // scope so the outer `finally` can always detach even on early
  // return. Listeners attach only after the pre-lock validation gates
  // pass; `abortListenersAttached` guards the detach.
  const abortController = new AbortController();
  const abortSocket = res.socket;
  const onAbortClose = (): void => {
    abortController.abort();
  };
  const onAbortError = (_err: unknown): void => {
    abortController.abort();
  };
  let abortListenersAttached = false;
  // `runPostDispatchCleanup` runs eagerly after `withExclusive` returns
  // (so a wedged post-commit persist does not pin abort listeners or
  // the lease) and also idempotently from the outer `finally` for the
  // early-return path. These flags keep it a no-op when already run.
  let cleanupPerformed = false;
  let leaseReleased = false;
  try {
    // Initial snapshot of the live binding. On a continuation we
    // re-read after `await store.getChain()` and reject if the
    // binding moved (hot-swap race guard below). Stateless requests
    // keep the snapshot unchanged.
    const initialSessionReg: SessionRegistry = lease.registry;
    const initialInstanceId: number = lease.instanceId;

    let sessionReg: SessionRegistry = initialSessionReg;
    let currentInstanceId: number | undefined = initialInstanceId;

    const responseId = genId('resp_');

    let priorMessages: ChatMessage[] | undefined;
    let previousResponseId: string | undefined;
    // Trailing-record inherited instructions, applied when the caller
    // omits `body.instructions` (empty string still counts as an
    // explicit override). Keeps `instructions: "You are a pirate"`
    // alive across cold replays.
    let inheritedInstructions: string | null = null;
    // Precomputed scalar = earliest wall-clock expiry across the
    // resolved chain (epoch-ms). `ResponseStore.getChain()` aborts on
    // the first expired ancestor (see
    // `crates/mlx-db/src/response_store/reader.rs:44-59`), so once
    // this bound is crossed the chain is unrecoverable and we can
    // short-circuit the retryable-503 path to permanent 404. Threading
    // only the scalar (not the record array) keeps background
    // hard-timeout closures O(1) per pending continuation.
    let chainEarliestExpiresAtMs: number | undefined = undefined;

    if (body.previous_response_id && store) {
      try {
        // Persist-before-getChain race: a client firing back-to-back
        // `previous_response_id: A` can reach `getChain(A)` before
        // the producer's off-lock `store.store(A)` has landed. The
        // pending-writes tracker is registered synchronously inside
        // `withExclusive` (see `initiatePersist`) so we can observe
        // the in-flight write and retry.
        //
        // Native mlx-db throws `"Response not found: <id>"` on miss
        // (`crates/mlx-db/src/response_store/reader.rs`); in-memory
        // mocks return `[]`. Handle both — the lenient /not found/
        // match routes both into the retry path while letting real
        // infrastructure errors bubble to the outer catch.
        let chain: StoredResponseRecord[];
        let firstAttemptError: unknown = null;
        try {
          chain = await store.getChain(body.previous_response_id);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          if (!/not found/i.test(msg)) {
            throw err;
          }
          firstAttemptError = err;
          chain = [];
        }

        if (chain.length === 0) {
          const pending = getPendingWritesFor(store).awaitPending(body.previous_response_id);
          if (pending !== undefined) {
            // Bound the wait — `awaitPending` returns the raw
            // `store.store` promise which can hang indefinitely on a
            // wedged backend. On timeout fall through to the
            // last-probe branch below.
            type PendingOutcome = 'landed' | 'timeout';
            const chainWriteWaitTimeoutMs = getChainWriteWaitTimeoutMs();
            let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
            const timeoutPromise = new Promise<PendingOutcome>((resolve) => {
              timeoutHandle = setTimeout(() => {
                resolve('timeout');
              }, chainWriteWaitTimeoutMs);
            });
            const pendingOutcome: Promise<PendingOutcome> = pending.then(() => 'landed' as const);
            let timedOut = false;
            try {
              const outcome = await Promise.race([pendingOutcome, timeoutPromise]);
              timedOut = outcome === 'timeout';
            } catch {
              // Write rejection is the producer's problem; the
              // tracker's .finally() already cleared the entry so
              // the retry below sees the true post-failure state.
            } finally {
              if (timeoutHandle !== undefined) {
                clearTimeout(timeoutHandle);
              }
            }
            if (timedOut) {
              // Last-probe race closer: a write landing at
              // (timeout + epsilon) would have succeeded but 404-ing
              // here is non-retryable and permanently poisons the
              // client's chain. If the probe misses too, surface 503
              // storage_timeout (retryable) instead of 404.
              let probed: StoredResponseRecord[] | null = null;
              try {
                probed = await store.getChain(body.previous_response_id);
              } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                if (!/not found/i.test(msg)) {
                  throw err;
                }
                probed = null;
              }
              if (probed !== null && probed.length > 0) {
                // Log the wedged-writer condition even on a
                // successful probe so operators see the slow path
                // fired.
                console.warn(
                  `[responses] pending store write for previous_response_id "${body.previous_response_id}" did ` +
                    `not settle within ${chainWriteWaitTimeoutMs}ms, but a last-probe getChain found the record. ` +
                    `Continuing with the probed chain — likely a slow SQLite writer that landed just after the ` +
                    `timeout fired.`,
                );
                chain = probed;
              } else {
                // Once the chain's earliest-recoverable expiry has
                // passed, `getChain()` can never succeed (the reader
                // aborts on the first expired ancestor, see
                // `reader.rs:44-59`). Short-circuit to permanent 404
                // rather than loop the client on retryable 503 for an
                // unrecoverable chain.
                const earliestMs = getPendingWritesFor(store).getEarliestExpiresAtMs(body.previous_response_id);
                if (earliestMs !== undefined && Date.now() >= earliestMs) {
                  console.warn(
                    `[responses] timed out after ${chainWriteWaitTimeoutMs}ms waiting for pending store write ` +
                      `for previous_response_id "${body.previous_response_id}"; last-probe getChain still missed. ` +
                      `Earliest recoverable expiry (${earliestMs}ms) already crossed — returning 404 NotFound ` +
                      `rather than retryable 503 because getChain() can no longer succeed for this chain.`,
                  );
                  sendNotFound(res, `Previous response "${body.previous_response_id}" not found`);
                  return;
                }
                console.warn(
                  `[responses] timed out after ${chainWriteWaitTimeoutMs}ms waiting for pending store write ` +
                    `for previous_response_id "${body.previous_response_id}"; last-probe getChain still missed. ` +
                    `Returning 503 storage_timeout — the underlying store.store(...) promise did not settle in time, ` +
                    `likely a wedged SQLite writer or stuck native backend. The client may retry with the same ` +
                    `previous_response_id.`,
                );
                sendStorageTimeout(
                  res,
                  `Storage write for "${body.previous_response_id}" did not settle within ${chainWriteWaitTimeoutMs}ms. ` +
                    `This is a transient backend condition — retry the request with the same previous_response_id.`,
                );
                return;
              }
            } else {
              try {
                chain = await store.getChain(body.previous_response_id);
              } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                if (!/not found/i.test(msg)) {
                  throw err;
                }
                chain = [];
              }
            }
          } else if (firstAttemptError !== null) {
            // Hard-timed-out marker path: the post-commit persist
            // hit the hard breaker but the raw write may still land.
            // Classify as retryable 503 (not 404) so clients keep
            // the chain alive; re-probe once first to catch a write
            // that slipped in between marker-set and now.
            if (getPendingWritesFor(store).isHardTimedOut(body.previous_response_id)) {
              let lastChance: StoredResponseRecord[] | null = null;
              try {
                lastChance = await store.getChain(body.previous_response_id);
              } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                if (!/not found/i.test(msg)) {
                  throw err;
                }
                lastChance = null;
              }
              if (lastChance !== null && lastChance.length > 0) {
                console.warn(
                  `[responses] previous_response_id "${body.previous_response_id}" missing on first lookup and ` +
                    `its post-commit persist crossed the hard-timeout breaker, but a last-probe getChain found ` +
                    `the record. Continuing with the probed chain — likely a wedged SQLite writer that landed ` +
                    `just after the marker was set.`,
                );
                chain = lastChance;
              } else {
                console.warn(
                  `[responses] previous_response_id "${body.previous_response_id}" missing from store, but its ` +
                    `post-commit persist crossed the hard-timeout breaker and is still unresolved (last-probe ` +
                    `getChain still missed). Returning 503 storage_timeout so the client retries with the same ` +
                    `id rather than discarding the chain as permanently invalid.`,
                );
                sendStorageTimeout(
                  res,
                  `Storage write for "${body.previous_response_id}" crossed the post-commit persist hard-timeout ` +
                    `breaker and has not yet settled. This is a transient backend condition — retry the request ` +
                    `with the same previous_response_id.`,
                );
                return;
              }
            } else {
              // Genuine 404: first call missed, no pending write, no
              // hard-timed-out marker. Rethrow so outer catch emits 404.
              throw firstAttemptError;
            }
          }
          if (chain.length === 0) {
            // Mirror the rethrow branch: a mock-compatible store that
            // returned `[]` rather than throwing still needs the
            // hard-timed-out marker retryable-503 classification.
            if (getPendingWritesFor(store).isHardTimedOut(body.previous_response_id)) {
              let lastChance: StoredResponseRecord[] | null = null;
              try {
                lastChance = await store.getChain(body.previous_response_id);
              } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                if (!/not found/i.test(msg)) {
                  throw err;
                }
                lastChance = null;
              }
              if (lastChance !== null && lastChance.length > 0) {
                console.warn(
                  `[responses] previous_response_id "${body.previous_response_id}" missing on first lookup and ` +
                    `its post-commit persist crossed the hard-timeout breaker, but a last-probe getChain found ` +
                    `the record. Continuing with the probed chain — likely a wedged SQLite writer that landed ` +
                    `just after the marker was set.`,
                );
                chain = lastChance;
              } else {
                console.warn(
                  `[responses] previous_response_id "${body.previous_response_id}" missing from store, but its ` +
                    `post-commit persist crossed the hard-timeout breaker and is still unresolved (last-probe ` +
                    `getChain still missed). Returning 503 storage_timeout so the client retries with the same ` +
                    `id rather than discarding the chain as permanently invalid.`,
                );
                sendStorageTimeout(
                  res,
                  `Storage write for "${body.previous_response_id}" crossed the post-commit persist hard-timeout ` +
                    `breaker and has not yet settled. This is a transient backend condition — retry the request ` +
                    `with the same previous_response_id.`,
                );
                return;
              }
            } else {
              sendNotFound(res, `Previous response "${body.previous_response_id}" not found`);
              return;
            }
          }
        }

        // Hot-swap race guard (getChain await window): re-read the
        // binding and reject if `registry.register(body.model, …)`
        // re-pointed the name while we awaited. The in-lock guard
        // below covers the mutex-wait window; this one covers the
        // getChain window.
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

        // Cross-model continuation guard keyed on MODEL-INSTANCE
        // IDENTITY (not friendly name): friendly-name equality would
        // accept a chain produced by the pre-hot-swap instance and
        // silently replay through a different tokenizer / chat
        // template / KV layout. Aliases to the same instance are
        // handled transparently by shared `SessionRegistry` routing.
        //
        // Restart safety: the instance-id comparison is only
        // meaningful WITHIN a single process lifetime. The stored
        // row's `serverBootId` gates the comparison — when it does
        // NOT match the live boot id (or is absent on a row written
        // before this field was added) the stored `modelInstanceId`
        // belongs to a dead process and is meaningless, so the
        // strict guard is skipped and the continuation falls back
        // to name-based resume against whatever model is currently
        // bound to `body.model`. Truly legacy rows that carry
        // NEITHER `modelInstanceId` NOR `serverBootId` are still
        // rejected outright (cannot verify anything).
        const trailingRecord = chain[chain.length - 1]!;
        const storedIdentity = readStoredModelIdentity(trailingRecord);
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
        // kind === 'present': apply the strict instance-id guard ONLY when
        // the stored row carries a boot id that matches the live process.
        // A missing or non-matching stored boot id means the row was
        // produced by a prior process (or pre-bootId rollout), so the
        // stored instance id cannot be compared against anything live.
        const liveBootId = getServerBootId();
        const sameProcess = storedIdentity.bootId !== undefined && storedIdentity.bootId === liveBootId;
        if (sameProcess && (currentInstanceId === undefined || storedIdentity.instanceId !== currentInstanceId)) {
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
        // Fold chain expiries (epoch-seconds → ms) into a single
        // scalar. The full chain is NOT retained on outer scope, so
        // the hard-timeout closure below does not capture ancestor
        // JSON payloads. Rows with missing/malformed `expiresAt` are
        // skipped; all-missing chains leave the scalar `undefined`.
        const chainExpirySeconds =
          chain.length > 0
            ? Math.min(...chain.map((r) => r.expiresAt).filter((v): v is number => v != null && Number.isFinite(v)))
            : Number.POSITIVE_INFINITY;
        chainEarliestExpiresAtMs = Number.isFinite(chainExpirySeconds) ? chainExpirySeconds * 1000 : undefined;
        // Inherit the trailing record's `instructions` when the
        // request omits `body.instructions` (empty string still counts
        // as explicit override). The trailing record carries the
        // effective instructions in force for that turn, so no
        // full-chain walk is required. The effective value is also
        // threaded into the `SessionRegistry` cache key so a hot hit
        // under stale system context forces a cold replay.
        //
        // Empty-string stored instructions MUST be inherited as `""`,
        // not dropped to `null`: a chain that intentionally cleared
        // instructions with an explicit empty string was adopted
        // against the registry under `requestedInstructions = ""`, so
        // resolving a later no-instructions turn to `null` would
        // silently flip the byte-for-byte comparison in
        // `SessionRegistry.getOrCreate` and force a cold replay on
        // every follow-up. Gate on `typeof === 'string'` so only a
        // genuinely absent stored value (legacy rows, or rows whose
        // turn had no instructions in force) short-circuits inheritance.
        if (typeof body.instructions !== 'string') {
          const storedInstructions = chain[chain.length - 1]!.instructions;
          if (typeof storedInstructions === 'string') {
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

    // Echoed `function_call` items on a continuation are validated
    // for ownership (call_id in stored trailing assistant turn) then
    // stripped. `mapRequest` would otherwise rebuild each echo into a
    // synthetic assistant message at the tail of `messages`, letting
    // a forged echo rewrite the trailing assistant turn and bypass
    // the fan-out gate. `priorMessages` is the authoritative copy.
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
          // Stored state is authoritative — drop the echo regardless
          // of whether `name`/`arguments` match byte-for-byte.
          continue;
        }
        filtered.push(item);
      }
      effectiveInput = filtered;
    }

    // Effective instructions = caller's explicit `body.instructions`
    // or the trailing record's inherited value. Threaded through
    // `mapRequest` (prepends system msg), the registry cache key,
    // `buildResponseObject`, and persistence. Applied via a fresh
    // mapped body rather than mutating `body`.
    const effectiveInstructions: string | null =
      typeof body.instructions === 'string' ? body.instructions : inheritedInstructions;

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

    // New-only messages (what this request added). Instructions are
    // stored separately — persisting them as input messages would
    // replay stale system messages on cold chain. Mirror `mapRequest`'s
    // truthy check (empty string contributes zero offset).
    const instructionsOffset = mappedBody.instructions ? 1 : 0;
    const priorOffset = instructionsOffset + (priorMessages?.length ?? 0);
    let newInputMessages = messages.slice(priorOffset);

    // Every tool message in the continuation delta must carry a
    // non-empty `tool_call_id`. Correctness-critical: the id-set gate
    // below silently ignores anonymous tool messages, and native
    // backends that pair results positionally would bind the
    // anonymous entry to the wrong call.
    for (const m of newInputMessages) {
      if (m.role === 'tool' && (typeof m.toolCallId !== 'string' || m.toolCallId.length === 0)) {
        sendBadRequest(res, 'tool message missing tool_call_id', 'input');
        return;
      }
    }

    // `SessionRegistry` cache key — passing the effective value lets
    // the registry force a cold replay on instructions mismatch.
    const requestedInstructions: string | null = effectiveInstructions;

    // The native model is a single mutable resource (one
    // `cached_token_history`, one `caches` vector) so every dispatch
    // through `/v1/responses` and `/v1/messages` for the same binding
    // serializes through `sessionReg.withExclusive`. The mutex spans
    // `getOrCreate → dispatch → adopt/drop`.
    const preLockSessionReg = sessionReg;
    const preLockInstanceId = currentInstanceId;

    // Arm the abort listeners. `@mlx-node/lm`'s streaming wrappers
    // plumb the signal into `_runChatStream`, which calls
    // `handle.cancel()` on the native stream handle AND pushes a
    // synthetic marker to unblock the next `waitForItem()`. Attached
    // here (not at function entry) so early-return validation gates
    // above don't need paired detach calls.
    res.once('close', onAbortClose);
    res.once('error', onAbortError);
    if (abortSocket != null) {
      abortSocket.once('close', onAbortClose);
    }
    if (httpReq) {
      httpReq.once('close', onAbortClose);
      httpReq.once('error', onAbortError);
    }
    abortListenersAttached = true;
    const streamSignal: AbortSignal = abortController.signal;

    // Persistence is a two-step dance.
    //
    //   (1) INSIDE the per-model mutex (on the happy path only):
    //       synchronously kick off `store.store(record)` via
    //       `initiatePersist` — which registers the in-flight
    //       promise in a per-store pending-write tracker keyed on
    //       the response id. The mutex releases BEFORE the write
    //       lands in SQLite.
    //
    //   (2) AFTER the mutex releases: await the in-flight promise
    //       just to surface errors to the log. The write is
    //       already on its way; the caller waits purely for
    //       logging completeness.
    //
    // A back-to-back `previous_response_id` continuation that fires
    // between mutex release and SQLite land observes the pending
    // write through the tracker (see the `getChain`-empty retry at
    // the top of this handler) and awaits it before falling
    // through to the 404 epilogue. This closes the race where a
    // fresh response id on the wire could transiently 404 under
    // `getChain`.
    //
    // `pendingPersistOuter` is the in-flight promise captured
    // inside the lock; the out-of-lock awaiter just catches errors
    // and logs them. `persistMode` is populated alongside so the
    // log line keeps the streaming / non-streaming discrimination.
    let pendingPersistOuter: Promise<void> | null = null;
    let persistMode: 'streaming' | 'non-streaming' | null = null;
    // Structural scaffolding for the binding retain paired with the
    // in-flight persist. The persist's `.finally(...)` calls this
    // closure on settlement to balance the `retainBinding` taken at
    // dispatch time — the closure's idempotency flag matters only
    // to that one call site today.
    //
    // The box shape is kept deliberately so a future iteration can
    // reintroduce a surgical "split teardown" (e.g. release heavy
    // resources on timeout while keeping identity pinned until
    // settlement) without rewiring the retain wrappers in both
    // dispatch branches. Do NOT force-release on post-commit
    // timeout: a slow-but-eventual persist can still land after
    // the timer fires, and releasing the retain before the write
    // settles lets an intervening same-object `unregister()` +
    // `register()` finalise the old binding and mint a fresh
    // instance id, causing the late write to record a stale id and
    // break the next `previous_response_id` continuation.
    //
    // Held in a box because TypeScript's control-flow analysis
    // otherwise narrows the in-closure assignment to `never`
    // across the intervening `await` / try-catch boundaries.
    const persistRetainBox: { release: (() => void) | null } = { release: null };
    // `failureMode` carries the streaming failure-epilogue reason
    // from `handleStreamingNative` out to the outer adopt gate.
    // A final-chunk commit followed by a post-terminal `res.close`
    // takes the `client_abort` branch and flushes `response.failed`
    // successfully, which would otherwise flip `safeToSuppress =
    // true` and let the adopt gate cache a session under a response
    // id the client will never chain off of. The gate refuses to
    // adopt when `failureMode === 'client_abort'` regardless of how
    // `committed` / `safeToSuppress` landed.
    let streamFailureMode: StreamingHandlerOutcome['failureMode'] = null;

    try {
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
        // that `body.model` no longer resolves to. The pre-lock
        // re-read only covered the `store.getChain()` await window; the
        // mutex-wait window is strictly later and equally unsafe.
        //
        // Compare the live binding to the pre-lock snapshot (captured
        // just before entering the mutex — already refreshed on the
        // continuation path, identical to the handler-top snapshot
        // on the stateless path). Any drift — nullable or value — is
        // fatal and rejected with the same 400 envelope the pre-lock
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
        //
        // Hot-path eligibility gate: the chat-session delta API only
        // serves a SINGLE `user` or `tool` continuation message — the
        // `send` / `sendToolResult` entry points cover exactly that
        // shape. A single `assistant` / `system` continuation cannot
        // be advanced incrementally against the warm KV cache and
        // must be handled via reset + cold re-prime. Consuming a warm
        // lease only to immediately `session.reset()` would destroy
        // the cached prefix for no benefit (and mislabel the turn as
        // `hit` when the client actually paid a full cold-replay
        // prefill), so detect the case up front and force a fresh
        // session lookup by passing `null` into `getOrCreate`. The
        // subsequent `ResponseStore` reconstruction + `primeHistory` +
        // `startFromHistory*` path below handles the rebuild. Multi-
        // message continuations are left to the existing reset + cold
        // re-prime fall-through inside `runSession*` (see comments
        // there).
        const hotPathIneligible =
          previousResponseId != null &&
          newInputMessages.length === 1 &&
          newInputMessages[0]!.role !== 'user' &&
          newInputMessages[0]!.role !== 'tool';
        const lookup = hotPathIneligible
          ? sessionReg.getOrCreate(null, requestedInstructions)
          : sessionReg.getOrCreate(previousResponseId ?? null, requestedInstructions);
        const session = lookup.session;
        // `X-Session-Cache` observability header: classify this turn as
        // `fresh` (no `previous_response_id` on the request), `hit`
        // (warm-cache lease consumed), or `cold_replay` (request carried
        // `previous_response_id` but the warm entry was missing / expired
        // / instructions-mismatched / already leased, OR the request
        // shape is ineligible for the hot path — the endpoint will
        // rebuild the session from the `ResponseStore` below). Set
        // before any `writeHead` / SSE `beginSSE` so both JSON and SSE
        // responses carry it. See `endpoints/messages.ts` for the
        // matching always-`fresh` emission on `/v1/messages`.
        const sessionCacheStatus: SessionCacheStatus =
          previousResponseId == null ? 'fresh' : lookup.hit && !hotPathIneligible ? 'hit' : 'cold_replay';
        res.setHeader('X-Session-Cache', sessionCacheStatus);

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

        // Visibility / wire-format tracker shared between the handler
        // body and the outer catch. Declared outside the `try` so the
        // catch can branch on `responseMode` (JSON vs SSE) and know
        // whether a terminal artefact already landed — both signals
        // are authoritative, unlike `res.headersSent`.
        const visibility = createVisibility();

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
          // trailing stored record via instruction inheritance.
          // Using `body` here
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
          // `res.headersSent` is NOT a reliable proxy for "the client
          // received the response": Node's `writeHead` flips
          // `headersSent = true` synchronously before any body bytes
          // leave the buffer, and the sync return of `res.end()` /
          // `writeSSEEvent` only proves the bytes were queued — an
          // async socket failure after the queue could still leave
          // the client with no terminal. Picking JSON-vs-SSE fallback
          // from `res.headersSent` is also unsafe because a
          // `writeHead(200, 'application/json')` → `res.end()` crash
          // would otherwise emit SSE frames into a JSON-declared
          // response.
          //
          // The `TransportVisibility` record instead tracks both the
          // wire format the handler committed to (`responseMode`)
          // AND whether the client observed a terminal artefact
          // (`responseBodyWritten` / `terminalEmitted`). Both flags
          // are flipped only from the kernel-ack callback of the
          // underlying `res.end` / `res.write` — synchronous return
          // is NOT treated as proof of visibility. The outer catch
          // branches on `responseMode` to choose the clean-up shape
          // (JSON error, SSE `error` frame, or socket destroy).
          let handlerError: Error | null = null;

          if (mappedBody.stream) {
            const outcome = await runSessionStreaming(session, messages, newInputMessages, config, streamSignal);
            const streamingWasCommitted = () => outcome.wasCommitted();
            try {
              const handlerOutcome = await handleStreamingNative(
                res,
                outcome.stream,
                mappedBody,
                responseId,
                previousResponseId,
                streamingWasCommitted,
                httpReq,
                visibility,
              );
              streamFailureMode = handlerOutcome.failureMode;
              if (handlerOutcome.terminalToPersist != null && store && body.store !== false) {
                // Initiate the write SYNCHRONOUSLY inside the mutex so
                // the pending-write tracker observes it before the
                // mutex releases. The promise is awaited off-lock in
                // the outer finally block.
                const record = buildResponseRecord(
                  handlerOutcome.terminalToPersist,
                  newInputMessages,
                  previousResponseId,
                  currentInstanceId,
                  effectiveRetentionSec,
                );
                // Pair a `retainBinding` against the persist promise
                // so the binding's `modelInstanceId` survives a
                // concurrent same-model unregister + re-register that
                // races the post-commit write. `releaseBinding` runs
                // in the persist's `.finally(...)` regardless of
                // outcome, so the retention counter stays balanced
                // whether the write fulfils or rejects.
                //
                // Leaving the retain pinned forever on a wedged write
                // would make the binding unreclaimable until process
                // restart, so an INDEPENDENT hard-timeout timer is
                // armed alongside the persist (see
                // `getPostCommitPersistHardTimeoutMs` for the default).
                // If the persist settles naturally the timer is
                // cancelled via `clearTimeout` inside the same
                // `.finally(...)` — slow-but-eventual writes are
                // unaffected. If the persist is still wedged past the
                // hard bound, the timer fires and force-releases the
                // retain via the idempotent `persistRetainBox`. The
                // hard timer is armed off the handler's await path, so
                // the response is never delayed by it.
                //
                // Before the hard timeout force-releases the retain
                // (which unblocks binding teardown), it calls
                // `registry.retireInstanceIdForForceRelease(leaseModel)`
                // to tombstone the binding's current instance id on
                // the model object. A subsequent `register()` of the
                // SAME model object inherits that retired id rather
                // than minting fresh — so the late-landing persist's
                // record (stamped with the retired id) still matches
                // the live binding and stays chainable through
                // `previous_response_id`. Only a true hot-swap
                // (re-register with a DIFFERENT model object) mints a
                // fresh id, and the 400 instance-mismatch that results
                // is the correct semantic outcome because the new
                // model is semantically different from the one that
                // produced the stored record. Retirement MUST happen
                // BEFORE release so `instanceIds.get(model)` still
                // returns the live id the record carries.
                //
                // The tombstone's lifetime is scoped to the pending
                // persists that installed it — the `.finally(...)`
                // calls `registry.releaseTombstone(leaseModel)` so
                // that when the late write eventually settles
                // (fulfills or rejects), the shared refcount drops
                // and, once every outstanding persist has released,
                // any subsequent re-registration correctly mints a
                // fresh id. Without this scoping, a past hard-timeout
                // event would permanently re-enable id inheritance
                // across unrelated later lifecycles — reopening
                // stale-chain replay across what should be logically
                // dead bindings. The refcounted single-entry layout
                // handles OVERLAPPING hard-timeouts on the same live
                // instance id in bounded space: every breaker targets
                // the SAME retired id (the register-inherit path
                // keeps using it while the tombstone is alive) so one
                // shared refcount safely collapses every in-flight
                // retire, and memory stays O(1) per model even under
                // a truly wedged store that never settles.
                registry.retainBinding(leaseModel);
                let persistRetainReleased = false;
                persistRetainBox.release = () => {
                  if (persistRetainReleased) return;
                  persistRetainReleased = true;
                  registry.releaseBinding(leaseModel);
                };
                const streamingPersistMode = 'streaming' as const;
                const streamingHardTimeoutMs = getPostCommitPersistHardTimeoutMs();
                let retiredTombstone: { instanceId: number } | undefined;
                // Compute the scalar `absoluteExpiresAtMs` ONCE up
                // front — the MINIMUM of the newly produced record's
                // own row expiry and the earliest expiry across any
                // resolved ancestor chain. This value is threaded
                // into both the pending-write tracker at
                // `initiatePersist()` time (so the pre-breaker
                // `awaitPending` path can short-circuit to 404 once
                // the bound is crossed) AND the hard-timeout marker
                // at breaker-fire time (absolute cap). The
                // hard-timeout closure captures ONLY this scalar —
                // NOT the full resolved chain — so the closure's
                // retained heap stays O(1) under sustained pending
                // continuations against a degraded backend.
                //
                // `record.expiresAt` is epoch-seconds (see
                // `buildResponseRecord` — it adds
                // `RESPONSE_TTL_SECONDS` to `Math.floor(Date.now() /
                // 1000)`); convert to ms at this boundary. If both
                // the record and the chain lack a finite expiry
                // (legacy rows), fall back to
                // `Number.POSITIVE_INFINITY` at the marker call site
                // so TTL-only bounding still holds.
                const recordExpiresAtMs =
                  record.expiresAt != null && Number.isFinite(record.expiresAt) ? record.expiresAt * 1000 : undefined;
                const absoluteExpiresAtMs =
                  recordExpiresAtMs !== undefined && chainEarliestExpiresAtMs !== undefined
                    ? Math.min(recordExpiresAtMs, chainEarliestExpiresAtMs)
                    : (recordExpiresAtMs ?? chainEarliestExpiresAtMs);
                const streamingHardTimeoutHandle: ReturnType<typeof setTimeout> | null =
                  streamingHardTimeoutMs > 0
                    ? setTimeout(() => {
                        if (persistRetainReleased) return;
                        console.error(
                          `[responses] post-commit persist HARD timeout (${streamingHardTimeoutMs}ms, ` +
                            `${streamingPersistMode}): underlying store.store(...) has not settled; assuming ` +
                            `wedged backend, force-releasing the binding retain so the binding can be torn ` +
                            `down. Retiring the current instance id via tombstone so a same-object ` +
                            `re-registration inherits it and a late-landing persist remains chainable; a ` +
                            `hot-swap to a DIFFERENT model object will mint a fresh id and the stale chain ` +
                            `will correctly fail with 400 instance-mismatch.`,
                        );
                        // Move the pending-write tracker entry into
                        // the hard-timed-out marker state for this
                        // response id. The pending entry is dropped
                        // so a wedged store.store(...) does not pin
                        // one promise closure + tracker entry per
                        // hard-timed-out request, AND the id is added
                        // to the `hardTimedOut` marker so a concurrent
                        // `previous_response_id` continuation can
                        // tell the difference between a permanent
                        // 404 and a slow-but-eventual persist that
                        // crossed the hard timeout. The continuation
                        // path consults `isHardTimedOut(id)` before
                        // falling through to `sendNotFound(...)` and
                        // returns retryable 503 `storage_timeout`
                        // instead, so clients keep retrying rather
                        // than discarding the chain. The marker has
                        // two cleanup paths: (1) fast — the underlying
                        // store promise's `.finally(...)` inside
                        // `track()` fires when the wedged store
                        // unwedges; (2) slow — an independent TTL
                        // (`MLX_HARD_TIMEOUT_MARKER_TTL_MS`, default
                        // 300s) bounds memory at O(requestRate × TTL)
                        // even against a truly wedged store that
                        // NEVER settles. Marker lifetime =
                        // min(settlement, TTL expiry).
                        //
                        // Pass the record's absolute row expiry as a
                        // hard cap on the marker. The record's
                        // `expiresAt` field is epoch-seconds (see
                        // `buildResponseRecord` — it adds
                        // `RESPONSE_TTL_SECONDS` to `Math.floor(Date.now()
                        // / 1000)`), so convert to ms for the marker
                        // map. Once the absolute bound passes,
                        // `ResponseStore.getChain()` hides the row and
                        // the retryable-503 classification is factually
                        // wrong — the marker must flip to 404 regardless
                        // of ongoing client retries.
                        //
                        // Capture ONLY the precomputed scalar
                        // `absoluteExpiresAtMs` in this closure — NOT
                        // the full resolved chain. The scalar is
                        // `min(record.expiresAt * 1000,
                        // chainEarliestExpiresAtMs)`, computed once
                        // when the hard-timeout handle was armed
                        // above. `ResponseStore.getChain()` walks
                        // ancestors and aborts on the first expired
                        // link (see
                        // `crates/mlx-db/src/response_store/reader.rs:44-59`),
                        // so clamping the marker at whichever link
                        // would disappear from `getChain()` first is
                        // the authoritative bound. Capturing only
                        // the scalar means background pending
                        // continuations under a degraded store do
                        // not retain ancestor transcripts —
                        // heap growth stays O(1) per hard-timed-out
                        // persist regardless of chain length.
                        getPendingWritesFor(store).markHardTimedOut(
                          record.id,
                          getHardTimedOutMarkerTtlMs(),
                          absoluteExpiresAtMs ?? Number.POSITIVE_INFINITY,
                        );
                        // Retire the id FIRST (binding is still alive
                        // here — retirement reads the live id) then
                        // drop the retain, which may trigger the
                        // deferred teardown. Capture the retired id so
                        // the persist's `.finally(...)` can release
                        // the tombstone once the late write eventually
                        // settles. The registry stores one refcounted
                        // tombstone per model regardless of how many
                        // hard-timeouts overlap — each retire
                        // increments the shared counter and each
                        // release decrements it — so the returned
                        // `{ instanceId }` is captured as a presence
                        // flag and `releaseTombstone(leaseModel)` is
                        // called in the persist's `.finally(...)`.
                        retiredTombstone = registry.retireInstanceIdForForceRelease(leaseModel);
                        persistRetainBox.release?.();
                      }, streamingHardTimeoutMs)
                    : null;
                pendingPersistOuter = initiatePersist(store, record, absoluteExpiresAtMs).finally(() => {
                  if (streamingHardTimeoutHandle !== null) {
                    clearTimeout(streamingHardTimeoutHandle);
                  }
                  // If the hard-timeout breaker fired and installed a
                  // tombstone on `leaseModel`, decrement the shared
                  // refcount now that this persist has settled. The
                  // single-entry refcount layout means overlapping
                  // breakers share one slot — releasing one balances
                  // one retire, and the entry survives until the
                  // last outstanding persist releases.
                  if (retiredTombstone !== undefined) {
                    registry.releaseTombstone(leaseModel);
                  }
                  persistRetainBox.release?.();
                });
                persistMode = streamingPersistMode;
              }
            } catch (err) {
              handlerError = err instanceof Error ? err : new Error(String(err));
            }
            committed = streamingWasCommitted();
          } else {
            // The non-streaming native path has NO AbortSignal surface
            // (plain `chatSession*` returns a Promise, no cancel), so a
            // client that disconnects mid-generation still burns the
            // full decode budget under this mutex. TODO: native
            // cancellation for `chatSession*` — until then the best we
            // can do is the disconnect-aware skip inside
            // `handleNonStreaming` (short-circuits `endJson` and
            // signals the outer persist gate) plus this documented
            // limitation.
            const outcome = await runSessionNonStreaming(session, messages, newInputMessages, config);
            try {
              const handlerOutcome = await handleNonStreaming(
                res,
                outcome.result,
                mappedBody,
                responseId,
                previousResponseId,
                visibility,
              );
              if (store && body.store !== false) {
                // Same in-lock-initiate / off-lock-await split as the
                // streaming branch. The non-streaming handler only
                // returns when the JSON body's `res.end()` callback
                // has fired, so reaching this point means the client
                // observed the turn — the pending-write tracker
                // protects a back-to-back continuation from a
                // transient 404.
                const record = buildResponseRecord(
                  handlerOutcome.response,
                  newInputMessages,
                  previousResponseId,
                  currentInstanceId,
                  effectiveRetentionSec,
                );
                // See the streaming branch for the retain/release
                // rationale — a same-model unregister + re-register
                // during the slow persist must not mint a fresh
                // `modelInstanceId` that invalidates the row this
                // write is about to land. The idempotent-release
                // scaffolding is a structural hook for a future split
                // teardown; the post-commit SOFT timeout arm does not
                // force-fire it.
                //
                // A wedged persist would otherwise leak the binding
                // retain for the lifetime of the process, so the
                // hard-timeout timer is armed here in the same shape
                // as the streaming branch, cancelled from the
                // persist's own `.finally(...)` when the write settles
                // naturally, and fires a force-release through the
                // idempotent `persistRetainBox` otherwise. Default
                // 60s, override via
                // `MLX_POST_COMMIT_PERSIST_HARD_TIMEOUT_MS`, `'0'`
                // disables. Empty string is treated as unset (falls
                // back to the 60000ms default) so a config-templating
                // typo cannot silently disable the breaker.
                //
                // The force-release path also calls
                // `registry.retireInstanceIdForForceRelease(leaseModel)`
                // BEFORE releasing the retain so a same-object
                // re-registration AFTER teardown inherits the retired
                // instance id from the tombstone — a late-landing
                // persist against the retired id stays chainable. A
                // hot-swap to a DIFFERENT model object mints a fresh
                // id and the 400 instance-mismatch is correct.
                //
                // The tombstone's lifetime is scoped to the pending
                // persists that installed it — the `.finally(...)`
                // calls `registry.releaseTombstone(leaseModel)` so
                // that when the late write eventually settles, the
                // shared refcount drops and, once every outstanding
                // persist has released, any subsequent
                // re-registration correctly mints a fresh id. Without
                // this scoping, a past hard-timeout event would
                // permanently re-enable id inheritance across
                // unrelated later lifecycles — reopening stale-chain
                // replay across what should be logically dead
                // bindings. The refcounted single-entry layout
                // handles OVERLAPPING hard-timeouts on the same live
                // instance id in bounded space: every breaker targets
                // the SAME retired id (the register-inherit path
                // keeps using it while the tombstone is alive) so one
                // shared refcount safely collapses every in-flight
                // retire, and memory stays O(1) per model even under
                // a truly wedged store that never settles.
                registry.retainBinding(leaseModel);
                let persistRetainReleased = false;
                persistRetainBox.release = () => {
                  if (persistRetainReleased) return;
                  persistRetainReleased = true;
                  registry.releaseBinding(leaseModel);
                };
                const nonStreamingPersistMode = 'non-streaming' as const;
                const nonStreamingHardTimeoutMs = getPostCommitPersistHardTimeoutMs();
                let retiredTombstone: { instanceId: number } | undefined;
                // See the matching streaming-path comment above —
                // precompute the scalar `absoluteExpiresAtMs`
                // (`min(record.expiresAt * 1000,
                // chainEarliestExpiresAtMs)`) ONCE, thread it into
                // the tracker at `initiatePersist()` time, and capture
                // ONLY this scalar in the hard-timeout closure.
                const recordExpiresAtMs =
                  record.expiresAt != null && Number.isFinite(record.expiresAt) ? record.expiresAt * 1000 : undefined;
                const absoluteExpiresAtMs =
                  recordExpiresAtMs !== undefined && chainEarliestExpiresAtMs !== undefined
                    ? Math.min(recordExpiresAtMs, chainEarliestExpiresAtMs)
                    : (recordExpiresAtMs ?? chainEarliestExpiresAtMs);
                const nonStreamingHardTimeoutHandle: ReturnType<typeof setTimeout> | null =
                  nonStreamingHardTimeoutMs > 0
                    ? setTimeout(() => {
                        if (persistRetainReleased) return;
                        console.error(
                          `[responses] post-commit persist HARD timeout (${nonStreamingHardTimeoutMs}ms, ` +
                            `${nonStreamingPersistMode}): underlying store.store(...) has not settled; ` +
                            `assuming wedged backend, force-releasing the binding retain so the binding can ` +
                            `be torn down. Retiring the current instance id via tombstone so a same-object ` +
                            `re-registration inherits it and a late-landing persist remains chainable; a ` +
                            `hot-swap to a DIFFERENT model object will mint a fresh id and the stale chain ` +
                            `will correctly fail with 400 instance-mismatch.`,
                        );
                        // Move the pending-write tracker entry into
                        // the hard-timed-out marker state for this
                        // response id. The pending entry is dropped
                        // so a wedged store.store(...) does not pin
                        // one promise closure + tracker entry per
                        // hard-timed-out request, AND the id is added
                        // to the `hardTimedOut` marker so a concurrent
                        // `previous_response_id` continuation can
                        // tell the difference between a permanent
                        // 404 and a slow-but-eventual persist that
                        // crossed the hard timeout. The continuation
                        // path consults `isHardTimedOut(id)` before
                        // falling through to `sendNotFound(...)` and
                        // returns retryable 503 `storage_timeout`
                        // instead, so clients keep retrying rather
                        // than discarding the chain. The marker has
                        // two cleanup paths: (1) fast — the
                        // underlying store promise's `.finally(...)`
                        // inside `track()` fires when the wedged
                        // store unwedges; (2) slow — an independent
                        // TTL (`MLX_HARD_TIMEOUT_MARKER_TTL_MS`,
                        // default 300s) bounds memory at
                        // O(requestRate × TTL) even against a truly
                        // wedged store that NEVER settles. Marker
                        // lifetime = min(settlement, TTL expiry).
                        //
                        // Pass the record's absolute row expiry as a
                        // hard cap on the marker. The record's
                        // `expiresAt` field is epoch-seconds (see
                        // `buildResponseRecord` — it adds
                        // `RESPONSE_TTL_SECONDS` to `Math.floor(Date.now()
                        // / 1000)`), so convert to ms for the marker
                        // map. Once the absolute bound passes,
                        // `ResponseStore.getChain()` hides the row and
                        // the retryable-503 classification is factually
                        // wrong — the marker must flip to 404 regardless
                        // of ongoing client retries.
                        //
                        // Capture ONLY the precomputed scalar
                        // `absoluteExpiresAtMs` in this closure — see
                        // the matching streaming-path comment for the
                        // full rationale. The scalar was computed
                        // above when the hard-timeout handle was
                        // armed.
                        getPendingWritesFor(store).markHardTimedOut(
                          record.id,
                          getHardTimedOutMarkerTtlMs(),
                          absoluteExpiresAtMs ?? Number.POSITIVE_INFINITY,
                        );
                        // Retire the id FIRST (binding is still alive
                        // here — retirement reads the live id) then
                        // drop the retain, which may trigger the
                        // deferred teardown. Capture the retired id so
                        // the persist's `.finally(...)` can release
                        // the tombstone once the late write eventually
                        // settles. The registry stores one refcounted
                        // tombstone per model regardless of how many
                        // hard-timeouts overlap — each retire
                        // increments the shared counter and each
                        // release decrements it — so the returned
                        // `{ instanceId }` is captured as a presence
                        // flag and `releaseTombstone(leaseModel)` is
                        // called in the persist's `.finally(...)`.
                        retiredTombstone = registry.retireInstanceIdForForceRelease(leaseModel);
                        persistRetainBox.release?.();
                      }, nonStreamingHardTimeoutMs)
                    : null;
                pendingPersistOuter = initiatePersist(store, record, absoluteExpiresAtMs).finally(() => {
                  if (nonStreamingHardTimeoutHandle !== null) {
                    clearTimeout(nonStreamingHardTimeoutHandle);
                  }
                  // If the hard-timeout breaker fired and installed a
                  // tombstone on `leaseModel`, decrement the shared
                  // refcount now that this persist has settled. The
                  // single-entry refcount layout means overlapping
                  // breakers share one slot — releasing one balances
                  // one retire, and the entry survives until the
                  // last outstanding persist releases.
                  if (retiredTombstone !== undefined) {
                    registry.releaseTombstone(leaseModel);
                  }
                  persistRetainBox.release?.();
                });
                persistMode = nonStreamingPersistMode;
              }
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
          //
          // Refuse to adopt whenever the streaming handler took ANY
          // failure epilogue, not just `client_abort`. The streaming
          // handler writes `failureMode` for every path that does
          // not produce a clean `response.completed`:
          //
          //   * `'client_abort'`  — client dropped the socket after
          //     the decode loop committed but before the success
          //     terminal was flushed; `response.failed` goes on the
          //     wire under a responseId the client has abandoned.
          //
          //   * `'error'`         — post-final teardown threw in
          //     the stream adapter's `finally` after the decode
          //     loop had already committed; `terminalToPersist` is
          //     null and the client saw `response.failed`, so the
          //     responseId is not a chainable artefact from the
          //     client's perspective.
          //
          //   * `'finish_reason_error'` / `'stream_exhausted'` —
          //     terminal derived from a non-clean end of stream.
          //     Same reasoning: `response.failed` on the wire, no
          //     chainable success terminal.
          //
          // In every non-null `failureMode` case the session
          // committed at the native level but the observable wire
          // state is a failure, so adopting the session under the
          // responseId would evict the last good hot session for
          // this model under the single-warm invariant even
          // though the adopted slot is unreachable.
          //
          // `failureMode === null` is the sole signal that the
          // stream path completed cleanly and the adopted session
          // is genuinely reachable via the responseId.
          if (committed && (handlerError == null || safeToSuppress) && streamFailureMode === null) {
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
          // Branch on `responseMode` (the wire format the handler
          // committed to), NOT `res.headersSent`
          // (which flips synchronously in `writeHead` and lies about
          // which format the client is consuming). Each branch
          // produces output that matches the Content-Type the client
          // already received — or no output at all if the terminal
          // already landed.
          if (visibility.responseMode === null) {
            // Headers never went out. Safe to emit a clean 500 JSON
            // error.
            sendInternalError(res, message);
          } else if (visibility.responseMode === 'json') {
            // We already wrote `Content-Type: application/json` and
            // possibly some body bytes; emitting an SSE frame here
            // would corrupt the response. Best we can do is destroy
            // the socket so the client sees a truncated JSON
            // response instead of a malformed document with an
            // unexpected MIME type. If the body was fully written
            // (`responseBodyWritten === true`) the outcome gate
            // above already returned without rethrowing, so reaching
            // this branch means the JSON never fully landed.
            try {
              res.destroy(err instanceof Error ? err : new Error(message));
            } catch {
              // Socket may already be gone; nothing more we can do.
            }
          } else {
            // `responseMode === 'sse'`: headers advertise SSE and
            // some (or all) of the stream already went out. If a
            // terminal event already landed, emitting another frame
            // is a no-op from the client's perspective but we still
            // close the stream cleanly. If no terminal landed (early
            // `writeSSEEvent` crash before `response.created`), emit
            // a best-effort streaming `error` frame so the client
            // sees SOMETHING it can parse.
            if (!visibility.terminalEmitted) {
              writeFallbackErrorSSE(res, 'error', { error_type: 'server_error', message });
            }
            try {
              endSSE(res);
            } catch {
              // Already closed / destroyed.
            }
          }
        }
      });
    } catch (err) {
      // Admission-control rejection from the per-model queue cap
      // (`SessionRegistry.withExclusive` threw before chaining into
      // the FIFO). Emit HTTP 429 so clients back off instead of
      // silently piling up more waiters. Post-dispatch cleanup below
      // still runs via the idempotent `finally` — abort listeners
      // were never fully armed for a never-dispatched request, and
      // the dispatch lease MUST be released exactly once against the
      // originally captured `leaseModel`.
      //
      // Any other error continues to propagate up to the handler's
      // outer try/catch so existing failure-epilogue behaviour is
      // preserved untouched.
      if (err instanceof QueueFullError) {
        if (!res.headersSent) {
          sendRateLimit(res, `Model queue full: ${err.queuedCount} waiting (limit ${err.limit}). Retry after 1s.`);
        }
      } else {
        throw err;
      }
    }

    // RELEASE the dispatch lease and DETACH the abort listeners
    // IMMEDIATELY now that `withExclusive` returned
    // and the terminal bytes have either been flushed or the outer
    // catch has emitted its error frame. The post-commit persist
    // wait that follows must NOT pin the request's lifecycle — a
    // wedged `store.store(...)` would otherwise leak socket/abort
    // listeners, keep the binding's `inFlight` counter elevated,
    // and block teardown after a hot-swap for the lifetime of the
    // wedged write.
    //
    // The binding's `modelInstanceId` still needs to survive until
    // the post-commit write has actually landed — otherwise a
    // same-model unregister + re-register sequence during a slow
    // persist would mint a fresh id, and the row (when it finally
    // lands) would reference a dead id that the very next
    // `previous_response_id` continuation would reject. That
    // lifetime is covered by the ORTHOGONAL `retainBinding` /
    // `releaseBinding` retention counter paired around
    // `initiatePersist` below, so the eager dispatch-lease release
    // here stays lossless.
    //
    // The outer `finally` below re-runs both cleanups idempotently
    // so an early-return validation failure (before the
    // `withExclusive` site) still cleans up; `cleanupPerformed` is
    // the guard.
    cleanupPerformed = runPostDispatchCleanup();

    // The persist write was INITIATED synchronously inside
    // `withExclusive` via `initiatePersist` — which registers the
    // in-flight promise in the per-store pending-write tracker
    // BEFORE the mutex releases. The SQLite flush is already on
    // its way; a back-to-back continuation observing the tracker
    // will block on the same promise instead of spuriously
    // returning 404 under `getChain` (see the `getChain`-empty
    // retry at the top of this handler).
    //
    // BOUND the wait on the persist promise with
    // `POST_COMMIT_PERSIST_TIMEOUT_MS`. A wedged native backend
    // can return a promise that never settles, and an
    // unconditional `await` would pin this handler forever —
    // leaking abort listeners and the dispatch lease (handled
    // above by running cleanup before this wait). On timeout we
    // leave the promise running in the background: the
    // pending-writes tracker still holds its reference so chained
    // continuations can still observe it, and its `.finally(...)`
    // handler will clear the tracker entry whenever the write
    // eventually settles (or stays wedged until the process exits).
    //
    // Persistence is best-effort — a failed write demotes to a
    // log line. The pending-write tracker's `.finally(...)`
    // handler removes the entry regardless of fulfill / reject,
    // so a rejected write correctly leaves the store empty AND
    // clears the tracker, and a subsequent `getChain()` then
    // returns empty legitimately. A `.catch(...)` is attached
    // synchronously so an eventual rejection from the
    // backgrounded promise does not trigger an
    // unhandled-rejection diagnostic after this handler returns.
    if (pendingPersistOuter != null) {
      // The local narrowed reference convinces the type-aware
      // lint that we're awaiting a real Promise; assigning
      // through `let` loses that narrowing because the closure
      // above could (in principle) reassign it.
      const promise: Promise<void> = pendingPersistOuter;
      // Attach terminal error handling FIRST. The tracker's own
      // `.finally(...)` is already attached and surfaces nothing
      // to Node's unhandled-rejection detector; this catch arm
      // logs the rejection and suppresses it locally so the
      // raced-against `Promise.race` sees a plain fulfillment
      // (`'settled' | 'timeout'`) rather than a rejection that
      // would otherwise require per-branch handling below.
      const capturedMode = persistMode;
      const settled: Promise<'settled'> = promise
        .then(() => 'settled' as const)
        .catch((err: unknown) => {
          console.error(`[responses] post-commit persistence failed (${capturedMode ?? 'unknown'}, off-lock):`, err);
          return 'settled' as const;
        });
      const postCommitPersistTimeoutMs = getPostCommitPersistTimeoutMs();
      let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
      const timeoutPromise: Promise<'timeout'> = new Promise<'timeout'>((resolve) => {
        timeoutHandle = setTimeout(() => {
          resolve('timeout');
        }, postCommitPersistTimeoutMs);
      });
      try {
        const outcome = await Promise.race([settled, timeoutPromise]);
        if (outcome === 'timeout') {
          console.warn(
            `[responses] post-commit persistence did not settle within ${postCommitPersistTimeoutMs}ms ` +
              `(${capturedMode ?? 'unknown'}, off-lock); detaching the handler and leaving the write in the ` +
              `background. The pending-writes tracker still holds a reference so chained continuations can ` +
              `observe the in-flight write, and the binding retain stays live until the write truly ` +
              `settles so the binding's modelInstanceId cannot be recycled under the late write. This ` +
              `condition usually signals a wedged SQLite writer or stuck native backend.`,
          );
          // Do NOT force-release the `retainBinding` here on the
          // soft timeout. `Promise.race` treats any write that
          // EXCEEDS the timeout as "safe to unpin", but most
          // timeouts in practice are slow-but-eventual writes —
          // the promise still fulfils later, and the retain
          // invariant has to hold for the entire interval until
          // it does. If a same-object unregister + re-register
          // happens in the window between timeout and actual
          // settlement, force-releasing the retain lets
          // `pendingPersists` drop to 0, the binding fully tears
          // down, the re-register mints a fresh
          // `modelInstanceId`, and the late write lands with the
          // stale id that `buildResponseRecord` stamped into
          // `configJson` — exactly the chain-break the retain
          // was introduced to prevent.
          //
          // We accept the bounded cost of a TRULY wedged persist
          // leaking one binding (counters + registry reference)
          // until process exit. A wedged SQLite writer already
          // means the server is compromised, and one lingering
          // binding is much smaller than a user-visible 400
          // instance-mismatch on the next continuation. The
          // idempotent `release` stays wired from the persist's
          // own `.finally(...)`, so the moment the slow write
          // actually settles — even minutes later — the retain
          // drops and teardown proceeds normally. The
          // independent hard-timeout breaker (armed at
          // `initiatePersist` time) bounds the truly-wedged case
          // via tombstoned id retirement.
          //
          // The pending-writes tracker keeps its own reference
          // to the detached promise, so chained continuations
          // can still observe the in-flight write via the
          // cold-replay path.
        }
      } finally {
        if (timeoutHandle !== undefined) {
          clearTimeout(timeoutHandle);
        }
      }
    }
  } finally {
    // Idempotent fallback: if the post-dispatch cleanup above
    // never ran (early-return validation failure, or an exception
    // raised inside the outer `try` block between lease
    // acquisition and the `withExclusive` call), make sure the
    // abort listeners are detached and the dispatch lease is
    // released here. `runPostDispatchCleanup` is safe to re-invoke
    // — the `abortListenersAttached` check and
    // `releaseDispatchLease`'s `inFlight < 0` floor make it a
    // no-op when the happy-path already fired it.
    if (!cleanupPerformed) {
      runPostDispatchCleanup();
    }
  }

  function runPostDispatchCleanup(): true {
    // Drop the AbortController's socket/request listeners so they
    // do not keep the request object alive past
    // the handler's return. Only detach when listeners were actually
    // installed — early-return validation failures exit the outer
    // try before the installation site, so an unconditional detach
    // would pull listeners that were never attached.
    if (abortListenersAttached) {
      res.removeListener('close', onAbortClose);
      res.removeListener('error', onAbortError);
      if (abortSocket != null) {
        abortSocket.removeListener('close', onAbortClose);
      }
      if (httpReq) {
        httpReq.removeListener('close', onAbortClose);
        httpReq.removeListener('error', onAbortError);
      }
      abortListenersAttached = false;
    }
    // Release the dispatch lease on the ORIGINAL model object the
    // lease was acquired against (not a re-read of `body.model`,
    // which may have been hot-swapped while we held the mutex). A
    // pending teardown — `unregister()` called concurrently while
    // this dispatch held the lease — finalises here once the
    // in-flight counter drops to zero AND the post-commit persist
    // retention has also released (see `retainBinding` below).
    //
    // This runs BEFORE the post-commit persist wait, not after, so
    // a wedged `store.store(...)` no longer pins the lease.
    // Teardown of a same-model unregister is still deferred by the
    // `retainBinding` counter so the binding's `modelInstanceId`
    // survives until the pending write has stamped its row
    // durably — see `initiatePersist`.
    if (!leaseReleased) {
      leaseReleased = true;
      registry.releaseDispatchLease(leaseModel);
    }
    return true;
  }
}
