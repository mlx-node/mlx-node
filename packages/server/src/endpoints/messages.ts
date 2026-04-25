/**
 * POST /v1/messages — stateless Anthropic Messages API.
 *
 * Every request carries the full conversation in `req.messages`. We allocate
 * a fresh `ChatSession` per request via `SessionRegistry.getOrCreate(null,
 * …, null)`, prime with the mapped history, and run `startFromHistory[Stream]`.
 * No adopt/drop on that path — the session's lifetime is the single call.
 *
 * The `prompt_cache_key` prefix-reuse feature is NOT exposed on this endpoint
 * — the field has been removed from `AnthropicMessagesRequest` so clients
 * are not misled into believing they are getting prefix reuse. The tier-2
 * lookup will be re-enabled (together with advertising the request field)
 * once native KV can be preserved across `reset() + primeHistory()` on the
 * stateless per-turn dispatch this endpoint performs. For /v1/responses-
 * style prefix reuse today, clients should use `prompt_cache_key` on
 * `/v1/responses` instead.
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
import type { IdleSweeper } from '../idle-sweeper.js';
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
  // Tier-2 lookup is currently disabled on `/v1/messages` (see the
  // block comment around `sessionReg.getOrCreate` in the handler),
  // so `session` is always a fresh `turns === 0` wrapper from
  // `newSession()` and the `turns > 0` branch below never fires
  // today. The `turns === 0` branch still needs an explicit
  // `reset()` because a fresh JS session does NOT imply a fresh
  // native cache — the underlying `SessionCapableModel` is shared
  // across `ChatSession` lifetimes, and its native
  // `cached_token_history` persists across requests. After the
  // native refactor moved the unconditional wipe out of
  // `chat_session_start_sync` into the miss branch of
  // `verify_cache_prefix_direct`, a fresh-session cold replay that
  // did not explicitly reset would silently reuse whatever prefix
  // happened to overlap with the previous request — a cross-request
  // cache-affinity side channel. Only registry HITS are authorized
  // for cache reuse, and this endpoint never produces one, so every
  // cold replay must wipe.
  await session.reset();
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

async function runSessionStreaming(
  session: ChatSession<SessionCapableModel>,
  messages: ChatMessage[],
  config: ChatConfig,
  signal: AbortSignal | undefined,
): Promise<MessagesStreamingOutcome> {
  // See `runSessionNonStreaming` for the full rationale. Always
  // `reset()` before `primeHistory()` to wipe the shared native
  // model's `cached_token_history` — a fresh JS session does not
  // imply a fresh native cache.
  await session.reset();
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
  idleSweeper?: IdleSweeper | null,
  resolveModel?: (name: string) => Promise<void>,
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

  // Run the Anthropic→internal mapping BEFORE the lazy-load hook.
  //
  // Background: in `mlx launch claude` mode `resolveModel` may load a
  // 27GB model from disk (~30s) on first sight of an unknown name. If
  // we then fail mapping (unsupported role, malformed tool block, etc.)
  // we've burned a load — and possibly evicted the currently-resident
  // model — just to return 400 a moment later. Mapping is a pure
  // transform with no side effects, so it's safe to hoist above
  // resolveModel and use as a cheap pre-flight gate.
  let mappedMessages: ChatMessage[];
  let mappedConfig: ChatConfig;
  try {
    ({ messages: mappedMessages, config: mappedConfig } = mapAnthropicRequest(body));
  } catch (err) {
    sendAnthropicBadRequest(res, err instanceof Error ? err.message : 'Invalid request');
    return;
  }

  // Lazy-load hook: give the host a chance to register the requested
  // model before we look it up. Errors bubble up to the handler's
  // top-level catch which returns 500.
  //
  // The load is bracketed by `idleSweeper.withSuspendedDrains` so the
  // post-request drain timer armed by the PREVIOUS request's
  // `endRequest()` cannot fire mid-load. In `mlx launch claude` mode
  // `resolveModel` may invoke a 30s `loadModel()` on first sight of an
  // unknown name; if the prior request's matching `endRequest()`
  // armed the default 30s drain immediately before this load began,
  // the timer would otherwise call `clearCache()` while weight
  // materialization was still allocating through the Metal free pool —
  // exactly the hot-load race `withSuspendedDrains` exists to prevent.
  // The wrapper handles try/finally itself and is a pass-through on
  // the disabled sweeper, so the bracket is unconditional whenever
  // a sweeper is supplied.
  if (resolveModel) {
    // A throw here (bad model path, corrupt weights, native loader failure)
    // would otherwise bubble up to the outer `createHandler` catch which
    // emits the OpenAI-shape `{ error: ... }` envelope via `sendInternalError`.
    // This endpoint is Anthropic; clients parse the
    // `{ type: 'error', error: { type, message } }` shape, so we must
    // serialize the failure through `sendAnthropicInternalError` here. Mirrors
    // the `mapAnthropicRequest` try/catch above.
    try {
      if (idleSweeper) {
        await idleSweeper.withSuspendedDrains(() => resolveModel(body.model));
      } else {
        await resolveModel(body.model);
      }
    } catch (err) {
      sendAnthropicInternalError(res, err instanceof Error ? err.message : 'Failed to resolve model');
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
  // Idle-sweeper bracket flags — hoisted so the outer `finally` can
  // observe whether the `beginRequest()` bump ever happened. Early
  // validation-failure returns skip the bump and therefore also skip
  // the matching `endRequest()`. `idleRequestEnded` is the `done`
  // flag that guarantees the decrement fires exactly once regardless
  // of which finalize path — outer `finally`, `finish`, `close`,
  // `error` — wins the race.
  //
  // Listeners are attached EAGERLY at `beginRequest()` time, not
  // lazily from the outer `finally`. The round-4 review surfaced a
  // leak where a terminal socket event fired before the outer
  // `finally` ran: the lazy attach saw `writableEnded === false` at
  // check time, attached listeners on a socket whose terminal event
  // had already been emitted, and `endRequest()` then never fired,
  // leaving `inFlight` pinned above zero and the sweeper permanently
  // armed.
  let idleRequestStarted = false;
  let idleRequestEnded = false;
  let idleListenersAttached = false;
  const finalizeIdleRequest = (): void => {
    if (!idleRequestStarted) return;
    if (idleRequestEnded) return;
    idleRequestEnded = true;
    idleSweeper?.endRequest();
  };
  const onFinalizeEvent = (): void => {
    finalizeIdleRequest();
  };
  try {
    const sessionReg: SessionRegistry = lease.registry;
    // Snapshot the monotonic instance id so the in-mutex re-read can detect a
    // hot-swap that lands between lease acquisition and mutex entry. Unlike
    // `/v1/responses`, the Anthropic handler has no stored-identity check
    // downstream to catch the race later.
    const preLockInstanceId: number = lease.instanceId;

    // `mapAnthropicRequest` already ran (and succeeded) above as a cheap
    // pre-flight gate before `resolveModel` so a malformed request can't
    // trigger a multi-second model load just to 400 a moment later.
    const messages: ChatMessage[] = mappedMessages;
    const config: ChatConfig = mappedConfig;

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

    // Bracket the native-model dispatch with the idle sweeper.
    // Scoped here (past validation, before any native prefill /
    // decode) so purely observational endpoints and pre-validation
    // rejections do not push the sweeper's pending-drain timer out.
    //
    // Attach the terminal-event listeners BEFORE any `await` — the
    // round-4 fix for a leak where a fast terminal event fired
    // before the outer `finally` attached its listeners, leaving
    // `inFlight` pinned above zero. `finalizeIdleRequest` is
    // idempotent (guarded by `idleRequestEnded`) so whichever path
    // wins — listeners, outer `finally`, or a pre-dispatch early
    // return — the decrement fires exactly once.
    idleSweeper?.beginRequest();
    idleRequestStarted = true;
    res.once('finish', onFinalizeEvent);
    res.once('close', onFinalizeEvent);
    res.once('error', onFinalizeEvent);
    idleListenersAttached = true;

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

        // Tier-2 lookup disabled on /v1/messages until native KV can be
        // preserved across `reset() + primeHistory()`.
        //
        // The endpoint's `runSession*` helpers still `session.reset()`
        // on every turn (since `primeHistory` refuses to overwrite a
        // `turns > 0` session), and `ChatSession.reset()` wipes BOTH
        // JS-side state AND the underlying model's native KV cache.
        // Passing any `prompt_cache_key` into `sessionReg.getOrCreate`
        // would therefore:
        //   1. On a tier-2 hit, lease out the one warm session from the
        //      single-warm registry (entries.clear() runs inside
        //      getOrCreate), immediately reset() it — wiping the very
        //      KV state the tier-2 hit exists to preserve — and NEVER
        //      re-adopt(), because this endpoint does not assign
        //      response ids or call `sessionReg.adopt`. Net effect: a
        //      /v1/messages request with a matching `prompt_cache_key`
        //      silently STEALS and destroys the /v1/responses
        //      endpoint's warm session while gaining no reuse itself.
        //   2. On a tier-2 miss, `entries.clear()` in the fallthrough
        //      branch still evicts any entry keyed to a different
        //      prompt_cache_key — same destructive side effect.
        // Passing `null` here keeps the registry untouched for other
        // clients and makes this endpoint a pure cold-start path.
        //
        // The `prompt_cache_key` field has been removed from the
        // `AnthropicMessagesRequest` public type — advertising a
        // field the handler silently ignores misled clients into
        // believing they were getting prefix reuse. Re-adding both
        // the field and this lookup becomes a single-PR change once
        // native KV can survive a `reset()` + `primeHistory()`
        // round-trip on this endpoint.
        const lookup = sessionReg.getOrCreate(null, requestedSystem, null);
        const session = lookup.session;
        res.setHeader('X-Session-Cache', 'fresh');

        // Outer catch branches on `responseMode` (not `res.headersSent`, which
        // flips in `writeHead` before the body lands) so a crash after
        // `writeHead(application/json)` cannot leak SSE frames into a JSON body.
        const visibility = createVisibility();

        try {
          if (body.stream === true) {
            const outcome = await runSessionStreaming(session, messages, config, streamSignal);
            await handleStreamingNative(res, outcome.stream, body, outcome.wasCommitted, httpReq, visibility);
          } else {
            // Native `chatSessionStart` has no AbortSignal yet — disconnect handling
            // lives inside `handleNonStreaming` / `endJson`.
            const result = await runSessionNonStreaming(session, messages, config);
            // X-Cached-Tokens intentionally not emitted here: the
            // tier-2 lookup is disabled above, the session is reset()
            // before every turn, and the runSession* helpers never
            // resume an existing conversation — so `result.cachedTokens`
            // is always 0 on this endpoint. Flip this back on (together
            // with the tier-2 lookup) once native KV can survive the
            // reset() + primeHistory() round-trip.
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
    // Belt-and-suspenders: call `finalize()` unconditionally here.
    // The eagerly-attached `finish`/`close`/`error` listeners almost
    // always win the race, but we still fire here to cover
    // pathological cases where the terminal event never arrives —
    // e.g. a synthetic mock, or a pre-dispatch early return that
    // skipped the attach entirely. `finalizeIdleRequest` is
    // idempotent (guarded by `idleRequestEnded`) so the double-fire
    // is a no-op. Detach afterwards so the listeners don't pin the
    // handler scope past return.
    finalizeIdleRequest();
    if (idleListenersAttached) {
      res.removeListener('finish', onFinalizeEvent);
      res.removeListener('close', onFinalizeEvent);
      res.removeListener('error', onFinalizeEvent);
      idleListenersAttached = false;
    }
  }
}
