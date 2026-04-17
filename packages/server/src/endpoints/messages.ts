/**
 * POST /v1/messages — stateless Anthropic Messages API.
 *
 * Every request carries the full conversation in `req.messages`. We allocate
 * a fresh `ChatSession` per request via `SessionRegistry.getOrCreate(null)`,
 * prime with the mapped history, and run `startFromHistory[Stream]`. No
 * adopt/drop: the session's lifetime is this single call.
 */

import type { IncomingMessage, ServerResponse } from 'node:http';

import type { ChatConfig, ChatMessage, ChatResult } from '@mlx-node/core';
import type { ChatSession, ChatStreamEvent, SessionCapableModel } from '@mlx-node/lm';

import {
  sendAnthropicBadRequest,
  sendAnthropicInternalError,
  sendAnthropicNotFound,
  sendAnthropicRateLimit,
} from '../errors.js';
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
import type { AnthropicMessagesRequest } from '../types-anthropic.js';
import { validateAndCanonicalizeHistoryToolOrder } from './responses.js';

// Non-streaming path

async function handleNonStreaming(
  res: ServerResponse,
  result: ChatResult,
  body: AnthropicMessagesRequest,
  visibility: TransportVisibility,
): Promise<void> {
  const messageId = genId('msg_');
  const response = buildAnthropicResponse(result, body, messageId);

  // Native `chatSession*` has no AbortSignal surface yet, so a client that
  // disconnects mid-decode still burns every remaining token under the
  // per-model mutex. Disconnect handling is delegated to `endJson`'s
  // pre-entry destroyed check, which rejects synchronously after `responseMode`
  // has been committed to 'json' — the outer catch then destroys the socket.
  await endJson(res, JSON.stringify(response), visibility);
}

// Streaming path

async function handleStreamingNative(
  res: ServerResponse,
  chatStream: AsyncGenerator<ChatStreamEvent>,
  body: AnthropicMessagesRequest,
  wasCommitted: () => boolean,
  httpReq: IncomingMessage | undefined,
  visibility: TransportVisibility,
): Promise<void> {
  const messageId = genId('msg_');
  beginSSE(res);
  // Commit SSE wire format now so any throw before the terminal event routes
  // to the streaming error epilogue instead of corrupting the JSON path.
  markSSEMode(visibility);

  writeSSEEvent(res, 'message_start', buildMessageStartEvent(body, messageId, 0));

  let contentBlockIndex = 0;
  let hasEmittedThinking = false;
  let hasEmittedText = false;
  let emittedTextLength = 0;
  const tagBuffer = new ToolCallTagBuffer();

  // Terminal emission is deferred until after the loop drains so `wasCommitted()`
  // reads an authoritative `session.turns`. On a committed done chunk we emit
  // `message_delta` + `message_stop`; on an uncommitted terminal (finishReason=error,
  // mid-decode throw, client abort, iterator exhaustion) we emit a single streaming
  // `error` event and withhold `message_stop`.
  let sawDone = false;
  let terminalStopReason: string | null = null;
  let terminalNumTokens = 0;
  let terminalPromptTokens: number | undefined;
  let terminalErrorMessage: string | null = null;

  // `thrownError` sticks on a generator throw; `clientAborted` sticks on
  // HTTP `close`/`error` on req, res, or res.socket. Either one routes the
  // post-loop block to the failure epilogue. Native decode has no
  // AbortSignal yet, so on a client disconnect we can only stop consuming
  // deltas — the native decode still runs to completion under the mutex.
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
      if (clientAborted) break;
      if (event.done) {
        sawDone = true;

        // An error terminal must NOT flush content blocks — doing so would race
        // with the post-loop close and advertise a clean fan-out that the
        // session rolled back.
        if (event.finishReason === 'error') {
          terminalErrorMessage = 'model reported finishReason=error';
          break;
        }

        const remainingText = tagBuffer.flush();
        if (!tagBuffer.suppressed && remainingText) {
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
          emittedTextLength += remainingText.length;
          writeSSEEvent(
            res,
            'content_block_delta',
            buildContentBlockDelta(contentBlockIndex, { type: 'text_delta', text: remainingText }),
          );
        }

        if (hasEmittedThinking && !hasEmittedText) {
          writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex - 1));
        }

        const finalText = event.text;
        const okToolCalls = event.toolCalls.filter((t) => t.status === 'ok');
        const hasToolCalls = okToolCalls.length > 0;

        // Recovery: suppression triggered but no tool calls parsed — emit final text as a text block.
        if (tagBuffer.suppressed && !hasToolCalls && finalText && !hasEmittedText) {
          // Thinking block (if any) was already closed above.
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
          // Recovery: streaming text was cut off by a false-alarm `<tool_call>` tag. Emit the unsent suffix.
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

        // Emit any unsent suffix when final text is longer than what was streamed.
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
          // Pure tool-call turn — no text block.
        } else if (finalText) {
          // All text arrived in the final event; emit it as a single block.
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

        // Capture terminal state and break — actual `message_delta` / `message_stop` /
        // `error` emission is deferred until after the loop so `wasCommitted()` reads
        // an authoritative `session.turns` (the producer's finally runs on break).
        terminalStopReason = mapStopReason(event.finishReason, hasToolCalls);
        terminalNumTokens = event.numTokens;
        terminalPromptTokens = event.promptTokens;
        break;
      }

      // Delta event
      if (event.isReasoning) {
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
        // Text delta with `<tool_call>` buffering.
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
  } catch (err: unknown) {
    // Capture into a sticky flag so the post-loop block routes through the failure
    // epilogue (single streaming `error` event, no `message_stop`).
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

  // Success requires ALL of: sawDone, wasCommitted, no thrown error, no client abort.
  // Every failure path emits a streaming `error` and withholds `message_stop`.
  const committed = wasCommitted();
  const successful = sawDone && committed && thrownError == null && !clientAborted;

  if (successful) {
    const stopReason = terminalStopReason ?? 'end_turn';
    writeSSEEvent(res, 'message_delta', buildMessageDelta(stopReason, terminalNumTokens, terminalPromptTokens));
    await flushTerminalSSE(res, 'message_stop', buildMessageStop(), visibility);
  } else {
    // Close any dangling content block so the error frame lands at a clean state,
    // then emit the streaming error. Never emit `message_stop` here — pairing it
    // with an error would tell the client the turn completed cleanly.
    if (hasEmittedThinking && !hasEmittedText) {
      writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex - 1));
    } else if (hasEmittedText) {
      writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex));
    }
    let message: string;
    if (thrownError != null) {
      message = thrownError.message;
    } else if (clientAborted) {
      message = 'client disconnected before the stream completed';
    } else if (terminalErrorMessage != null) {
      message = terminalErrorMessage;
    } else if (sawDone) {
      message = 'model refused to commit the turn';
    } else {
      message = 'stream ended without a done event';
    }
    // The streaming `error` event is the Anthropic terminal on the failure path.
    await flushTerminalSSE(res, 'error', { type: 'error', error: { type: 'api_error', message } }, visibility);
  }
  endSSE(res);
}

// Session routing

/** Prime a fresh session with the full history and run a single turn. */
async function runSessionNonStreaming(
  session: ChatSession<SessionCapableModel>,
  messages: ChatMessage[],
  config: ChatConfig,
): Promise<ChatResult> {
  session.primeHistory(messages);
  return await session.startFromHistory(config);
}

/**
 * Streaming dispatch outcome. `wasCommitted()` compares `session.turns` against
 * the baseline captured AFTER `primeHistory`, matching the `/v1/responses`
 * streaming commit gate. Called post-drain by the SSE writer to pick the
 * terminal event (success → `message_stop`, failure → streaming `error`).
 */
interface MessagesStreamingOutcome {
  stream: AsyncGenerator<ChatStreamEvent>;
  wasCommitted: () => boolean;
}

function runSessionStreaming(
  session: ChatSession<SessionCapableModel>,
  messages: ChatMessage[],
  config: ChatConfig,
  signal: AbortSignal | undefined,
): MessagesStreamingOutcome {
  session.primeHistory(messages);
  const initialTurns = session.turns;
  return {
    stream: session.startFromHistoryStream(config, signal),
    wasCommitted: () => session.turns > initialTurns,
  };
}

// Public handler

export async function handleCreateMessage(
  res: ServerResponse,
  body: AnthropicMessagesRequest,
  registry: ModelRegistry,
  httpReq?: IncomingMessage,
): Promise<void> {
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

  for (const msg of body.messages) {
    if (msg == null || typeof msg !== 'object') {
      sendAnthropicBadRequest(res, 'Each message must be a non-null object');
      return;
    }
  }

  const model = registry.get(body.model);
  if (!model) {
    sendAnthropicNotFound(res, `Model "${body.model}" not found`);
    return;
  }

  // The lease keeps the binding's FIFO `execLock` chain alive across every
  // await — a concurrent `unregister()` + `register(sameModel)` would otherwise
  // tear down the old `SessionRegistry` and race two independent mutex chains
  // against one shared native model. Must be released in the `finally` below.
  const lease = registry.acquireDispatchLease(body.model);
  if (!lease) {
    sendAnthropicInternalError(res, 'session registry missing for registered model');
    return;
  }
  const leaseModel = lease.model;
  // AbortController wired to disconnect events. Declared at function scope
  // so the outer `finally` can detach listeners on early returns; the
  // `abortListenersAttached` flag gates the detach so pre-validation exits
  // skip it safely.
  const abortController = new AbortController();
  const abortSocket = res.socket;
  const onAbortClose = (): void => {
    abortController.abort();
  };
  const onAbortError = (_err: unknown): void => {
    abortController.abort();
  };
  let abortListenersAttached = false;
  try {
    const sessionReg: SessionRegistry = lease.registry;
    // Snapshot the monotonic instance id so the in-mutex re-read can detect a
    // hot-swap that lands between lease acquisition and mutex entry. Unlike
    // `/v1/responses`, the Anthropic handler has no stored-identity check
    // downstream to catch the race later.
    const preLockInstanceId: number = lease.instanceId;

    let messages: ChatMessage[];
    let config: ChatConfig;
    try {
      ({ messages, config } = mapAnthropicRequest(body));
    } catch (err) {
      sendAnthropicBadRequest(res, err instanceof Error ? err.message : 'Invalid request');
      return;
    }

    // Canonicalize every assistant fan-out's trailing tool block against its
    // declared sibling order. Several native session backends pair tool results
    // to fan-out calls POSITIONALLY (not by id), so caller-reversed sibling
    // results would silently bind to the wrong call. `'anthropic'` selects
    // error-message vocabulary (`tool_result` / `tool_use_id`).
    const historyError = validateAndCanonicalizeHistoryToolOrder(messages, 'anthropic');
    if (historyError !== null) {
      sendAnthropicBadRequest(res, historyError);
      return;
    }

    // The system prompt is baked into `messages` and replayed via `startFromHistory`,
    // so it cannot leak across requests. We still pass a canonicalized form to
    // `getOrCreate` to keep the registry API uniform with `/v1/responses`. Arrays
    // are JSON-stringified; plain strings pass through.
    let requestedSystem: string | null;
    if (typeof body.system === 'string') {
      requestedSystem = body.system;
    } else if (body.system != null) {
      requestedSystem = JSON.stringify(body.system);
    } else {
      requestedSystem = null;
    }

    // Per-model execution mutex. Every dispatch through `/v1/messages` serializes
    // with every dispatch through `/v1/responses` for the same model binding.
    // The native `SessionCapableModel` is a single mutable resource (shared
    // `cached_token_history` / `caches`), so two concurrent `primeHistory` +
    // `startFromHistory` would clobber each other's KV state.
    //
    // Arm the AbortController now — past all validation gates, so the
    // matching detach in the outer `finally` is guarded by
    // `abortListenersAttached`. Streaming wrappers in `@mlx-node/lm` plumb
    // this signal through `_runChatStream` to cancel the native
    // `ChatStreamHandle` and unblock the pending `waitForItem()` on
    // disconnect.
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

    try {
      await sessionReg.withExclusive(async () => {
        // Hot-swap race guard. `ModelRegistry.register()` is not coordinated with
        // `withExclusive`, so a concurrent re-register of the same friendly name
        // could silently dispatch this request through a stale model. Any drift
        // from the pre-lock snapshot is fatal.
        const lockedSessionReg = registry.getSessionRegistry(body.model);
        const lockedInstanceId = registry.getInstanceId(body.model);
        if (
          lockedSessionReg === undefined ||
          lockedInstanceId === undefined ||
          lockedSessionReg !== sessionReg ||
          lockedInstanceId !== preLockInstanceId
        ) {
          sendAnthropicBadRequest(
            res,
            `Model "${body.model}" binding changed while the request was queued behind the per-model ` +
              `execution mutex. A concurrent register() re-pointed the name at a different model instance ` +
              `(or released it entirely) while this waiter was parked, so the session registry and instance ` +
              `id captured before the mutex wait no longer match the live binding. Dispatching anyway would ` +
              `service this request through a stale model object — a silent cross-model handoff. Retry the ` +
              `request — if the swap was intentional, the new binding will service the retry cleanly.`,
          );
          return;
        }

        const session = sessionReg.getOrCreate(null, requestedSystem).session;

        // `X-Session-Cache` observability header: `/v1/messages` is
        // stateless — every request allocates a fresh `ChatSession` via
        // `getOrCreate(null, …)` — so the status is always `fresh`. Emit
        // it anyway to keep the header contract uniform with
        // `/v1/responses`, and set it before any `writeHead` / SSE
        // `beginSSE` so it lands on both JSON and SSE responses.
        res.setHeader('X-Session-Cache', 'fresh');

        // Outer catch branches on `responseMode` (not `res.headersSent`, which
        // flips in `writeHead` before the body lands) so a crash after
        // `writeHead(application/json)` cannot leak SSE frames into a JSON body.
        const visibility = createVisibility();

        try {
          if (body.stream === true) {
            const outcome = runSessionStreaming(session, messages, config, streamSignal);
            await handleStreamingNative(res, outcome.stream, body, outcome.wasCommitted, httpReq, visibility);
          } else {
            // Native `chatSessionStart` has no AbortSignal yet — disconnect handling
            // lives inside `handleNonStreaming` / `endJson`.
            const result = await runSessionNonStreaming(session, messages, config);
            await handleNonStreaming(res, result, body, visibility);
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : 'Unknown error during inference';
          if (visibility.responseMode === null) {
            sendAnthropicInternalError(res, message);
          } else if (visibility.responseMode === 'json') {
            // Already committed to JSON — destroy the socket rather than corrupt the body.
            try {
              res.destroy(err instanceof Error ? err : new Error(message));
            } catch {
              // Socket may already be gone.
            }
          } else {
            // SSE: best-effort streaming `error`, but only if no terminal landed
            // (a double terminal would confuse the client state machine).
            if (!visibility.terminalEmitted) {
              writeFallbackErrorSSE(res, 'error', { error: { type: 'api_error', message } });
            }
            try {
              endSSE(res);
            } catch {
              // Already closed.
            }
          }
        }
      });
    } catch (err) {
      // Admission-control rejection from the per-model queue cap
      // (`SessionRegistry.withExclusive` threw before chaining into
      // the FIFO). Emit Anthropic-shape HTTP 429 so clients back off
      // instead of silently piling up more waiters. The outer
      // `finally` below still detaches abort listeners and releases
      // the dispatch lease, so no per-request resources are leaked.
      //
      // Any other error continues to propagate so an abnormal failure
      // still routes through the handler's existing error paths.
      if (err instanceof QueueFullError) {
        if (!res.headersSent) {
          sendAnthropicRateLimit(
            res,
            `Model queue full: ${err.queuedCount} waiting (limit ${err.limit}). Retry after 1s.`,
          );
        }
      } else {
        throw err;
      }
    }
  } finally {
    // Drop disconnect listeners so they don't pin the request past handler
    // return. Only detach if we actually attached (gated by the flag).
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
    }
    // Release against the ORIGINAL lease model — re-reading `body.model`
    // would resolve to a possibly hot-swapped binding. A concurrent
    // `unregister()` held against this lease finalises its teardown here
    // when the in-flight counter drops to zero.
    registry.releaseDispatchLease(leaseModel);
  }
}
