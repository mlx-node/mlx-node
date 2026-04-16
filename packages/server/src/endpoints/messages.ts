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

import type { IncomingMessage, ServerResponse } from 'node:http';

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

// ---------------------------------------------------------------------------
// Non-streaming path
// ---------------------------------------------------------------------------

async function handleNonStreaming(
  res: ServerResponse,
  result: ChatResult,
  body: AnthropicMessagesRequest,
  visibility: TransportVisibility,
): Promise<void> {
  const messageId = genId('msg_');
  const response = buildAnthropicResponse(result, body, messageId);

  // Iter-35 finding 2 (part a): the non-streaming Anthropic path
  // has no AbortSignal surface on `chatSession*`, so a client that
  // disconnects mid-decode still burns every token under the per-
  // model mutex — we only learn about the peer loss once native
  // decode resolves. TODO(iter35): add a native cancellation
  // surface for `chatSession*` so the dispatch can bail at the
  // next safepoint rather than running to completion.
  //
  // Disconnect-aware handling is delegated to `endJson`'s own
  // pre-entry destroyed check (iter-34): on a dead peer it rejects
  // synchronously after `writeHead` has already flipped
  // `responseMode = 'json'`, which is exactly the shape the outer
  // catch expects — the JSON-mode branch destroys the socket
  // cleanly. Short-circuiting here with an early return would
  // leave `responseMode === null`, routing the error through the
  // clean-500 path instead, which tests (and production
  // expectations) rely on destroying the socket.

  // `endJson` commits `responseMode = 'json'` synchronously and
  // only flips `responseBodyWritten` once the kernel has
  // acknowledged the final chunk. The outer catch branches on
  // `responseMode`, so a crash between `writeHead` and the end-
  // callback routes to the JSON error path (or a socket destroy)
  // rather than emitting SSE frames into a JSON-declared response.
  await endJson(res, JSON.stringify(response), visibility);
}

// ---------------------------------------------------------------------------
// Streaming path
// ---------------------------------------------------------------------------

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
  // Commit to SSE wire format synchronously. The outer catch branches
  // on `responseMode` — any throw from `writeSSEEvent` between here
  // and the terminal event routes to the streaming `error` epilogue
  // instead of corrupting the JSON path.
  markSSEMode(visibility);

  writeSSEEvent(res, 'message_start', buildMessageStartEvent(body, messageId, 0));

  let contentBlockIndex = 0;
  let hasEmittedThinking = false;
  let hasEmittedText = false;
  let emittedTextLength = 0;
  const tagBuffer = new ToolCallTagBuffer();

  // Terminal state captured inside the done branch (or left null if
  // the iterator exhausts without a done event). The actual
  // `message_stop` / streaming `error` emission is deferred until
  // AFTER the loop drains so `wasCommitted()` can read an
  // authoritative `session.turns` — otherwise we would emit a
  // terminal event while the producer's finally has not yet run.
  //
  // Iter-27 finding 3: the previous implementation emitted
  // `message_stop` unconditionally the moment a `done` event arrived,
  // even when the final chunk carried `finishReason: 'error'`. That
  // reported a failed generation as a successful one to Anthropic
  // clients: no `error` SSE event, no way to distinguish a real
  // completion from a mid-decode failure. Mirror the
  // `/v1/responses` commit gate — on a committed (non-error) done
  // chunk we emit `message_delta` + `message_stop`; on an
  // uncommitted terminal (error finish or iterator exhaustion) we
  // emit a single streaming `error` event in the Anthropic shape.
  let sawDone = false;
  let terminalStopReason: string | null = null;
  let terminalNumTokens = 0;
  let terminalPromptTokens: number | undefined;
  let terminalErrorMessage: string | null = null;

  // Iter-28 finding 2: fault state. A mid-decode throw from the
  // underlying generator used to escape into the outer generic
  // catch, bypassing the commit gate and emitting an `error` event
  // AFTER SSE headers had already been flushed. A client disconnect
  // (HTTP `close`/`error`) had no signal at all, so the consumer
  // would happily keep streaming deltas to a dead socket and the
  // post-loop block would run the success branch.
  //
  // `thrownError` sticks on a generator throw; `clientAborted`
  // sticks on an HTTP request `close`/`error`. Either one diverts
  // the post-loop block to the failure epilogue (single Anthropic
  // streaming `error` SSE event, no `message_stop`). The underlying
  // `chatStreamSessionStart` does not yet accept an AbortSignal, so
  // on a client disconnect we can only stop consuming deltas and
  // break out of the loop — the native decode runs to completion
  // under the per-model mutex but nothing it emits reaches the
  // client side.
  //
  // Iter-34: also listen on `res` and `res.socket`. Non-terminal
  // Anthropic SSE writes (`message_start`, `content_block_delta`,
  // etc.) are fire-and-forget through `writeSSEEvent` — on a
  // destroyed socket they can silently "succeed" while decode keeps
  // burning work under the per-model mutex. Attaching the listener
  // here lets the next loop iteration observe the disconnect and
  // break out; native decode still runs to completion (no
  // AbortSignal plumbed yet) but nothing it emits reaches a dead
  // socket and the post-loop block routes to the failure epilogue.
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
        // Final event

        // Iter-27 finding 3: if the terminal chunk reports an error,
        // short-circuit the content-flush/close sequence and hand off
        // to the post-loop block. Emitting tool_use blocks or closing
        // content blocks here would (a) race with the post-loop
        // `content_block_stop` close on the error path and (b)
        // advertise a clean tool-call fan-out to the client even
        // though the session rolled back everything.
        if (event.finishReason === 'error') {
          terminalErrorMessage = 'model reported finishReason=error';
          break;
        }

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

        // Capture terminal state and break out of the loop. We do NOT
        // emit `message_delta` / `message_stop` here — both are gated
        // on the session's commit signal, which only becomes
        // authoritative after the outer generator's finally has run.
        // The post-loop block below reads `wasCommitted()` and emits
        // the right terminal event (success: message_delta +
        // message_stop; failure: a single `error` SSE event).
        terminalStopReason = mapStopReason(event.finishReason, hasToolCalls);
        terminalNumTokens = event.numTokens;
        terminalPromptTokens = event.promptTokens;
        break;
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
  } catch (err: unknown) {
    // Iter-28 finding 2: a mid-decode throw from the underlying async
    // generator used to escape out into the outer catch in
    // `handleCreateMessage`, which emitted the error frame AFTER SSE
    // headers were already flushed but without the commit-gate
    // post-loop check. Capture the error into a sticky flag so the
    // post-loop block below routes through the failure epilogue
    // (single Anthropic streaming `error` SSE event, no
    // `message_stop`).
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

  // Post-loop terminal emission. The producer's finally has now run
  // (either via the `break` after a done event, via natural iterator
  // exhaustion, via a mid-decode throw surfaced through the
  // try/catch above, or via a client disconnect that flipped
  // `clientAborted`), so `wasCommitted()` reads an authoritative
  // `session.turns` baseline. Success requires ALL of: sawDone,
  // committed, no thrown error, no client abort. Every failure path
  // emits a single Anthropic streaming `error` SSE event and
  // withholds `message_stop` so the client can distinguish a real
  // completion from a mid-decode failure.
  //
  // The Anthropic `/v1/messages` endpoint is stateless and never
  // calls `sessionReg.adopt()`, so the registry cannot leak a
  // cached session on failure. This gate's sole job is to make the
  // client-visible event stream report failures accurately.
  const committed = wasCommitted();
  const successful = sawDone && committed && thrownError == null && !clientAborted;

  if (successful) {
    const stopReason = terminalStopReason ?? 'end_turn';
    writeSSEEvent(res, 'message_delta', buildMessageDelta(stopReason, terminalNumTokens, terminalPromptTokens));
    // `flushTerminalSSE` gates `terminalEmitted` on the write
    // callback firing without error — proving the kernel accepted
    // `message_stop`, not just that ServerResponse buffered it.
    await flushTerminalSSE(res, 'message_stop', buildMessageStop(), visibility);
  } else {
    // Uncommitted terminal. Close any dangling content block so the
    // error frame arrives at a well-defined stream state, then emit
    // the Anthropic streaming error event. We do NOT emit
    // `message_stop` — pairing `message_stop` with an error would
    // tell the client the turn completed cleanly.
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
    // The streaming `error` event IS the Anthropic terminal on the
    // failure path. Gate `terminalEmitted` on its kernel-ack so the
    // outer catch can correctly distinguish "client already got a
    // terminal, log-only" from "early write crash, emit fallback".
    await flushTerminalSSE(res, 'error', { type: 'error', error: { type: 'api_error', message } }, visibility);
  }
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

/**
 * Outcome of a streaming session dispatch. `wasCommitted()` is a
 * closure that reports the commit signal AFTER the stream has been
 * consumed by the SSE writer — it compares `session.turns` against
 * the baseline captured AFTER `primeHistory`, mirroring the
 * `/v1/responses` streaming commit gate. The SSE writer reads this
 * post-drain to decide whether to emit `message_stop` (committed) or
 * an `error` SSE event (uncommitted, e.g. finishReason=error).
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

// ---------------------------------------------------------------------------
// Public handler
// ---------------------------------------------------------------------------

export async function handleCreateMessage(
  res: ServerResponse,
  body: AnthropicMessagesRequest,
  registry: ModelRegistry,
  httpReq?: IncomingMessage,
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

  // Acquire a dispatch lease on `body.model`'s session-registry
  // binding. The lease keeps the binding (and its FIFO `execLock`
  // mutex chain) alive across every await in this handler — crucial
  // because a concurrent `unregister()` + `register(sameModel)`
  // sequence would otherwise tear the old `SessionRegistry` down and
  // allocate a fresh one, and the new request's `withExclusive`
  // would race against this in-flight dispatch on one shared native
  // model with two independent mutex chains. The lease MUST be
  // released in a `finally` below so the binding's teardown (if
  // deferred by a concurrent `unregister()`) completes once the last
  // dispatch lease finishes.
  const lease = registry.acquireDispatchLease(body.model);
  if (!lease) {
    sendAnthropicInternalError(res, 'session registry missing for registered model');
    return;
  }
  const leaseModel = lease.model;
  // Iter-35 finding 1: AbortController wired to the HTTP request's
  // disconnect events, declared at function scope so the outer
  // `finally` can always detach listeners even on an early `return`.
  // Listeners attach only after we pass pre-lock validation — the
  // holder variables default to the no-op state so the detach loop
  // safely skips on the early-return path.
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
    // Fetch the per-model session registry. The Anthropic API is
    // stateless — we allocate a fresh session for every request via
    // `getOrCreate(null)` and never adopt it into the cache.
    const sessionReg: SessionRegistry = lease.registry;
    // Capture the monotonic instance id alongside the session registry
    // so the in-mutex re-read can detect a hot-swap that lands after
    // this read but before we acquire the per-model execution mutex.
    // Unlike `/v1/responses`, the Anthropic handler has no stored
    // identity check later in the pipeline to catch a mismatch, so this
    // is the only defence against the race.
    const preLockInstanceId: number = lease.instanceId;

    // Map request
    let messages: ChatMessage[];
    let config: ChatConfig;
    try {
      ({ messages, config } = mapAnthropicRequest(body));
    } catch (err) {
      sendAnthropicBadRequest(res, err instanceof Error ? err.message : 'Invalid request');
      return;
    }

    // Walk the stateless history and canonicalize every assistant
    // fan-out's trailing tool block against its declared sibling order.
    //
    // The Anthropic `/v1/messages` endpoint is ALWAYS a stateless
    // cold-start — there is no continuation gate, no stored prior
    // chain, no `previous_response_id`. The caller ships a full
    // self-contained conversation in `req.messages` and
    // `mapAnthropicRequest` produces the `ChatMessage[]` verbatim.
    // That leaves caller-supplied tool_result ordering flowing
    // straight into `primeHistory()`, so a caller can reverse two
    // sibling tool outputs inside one fan-out's `tool_result` block
    // and silently bind each output to the wrong call because several
    // native session backends pair tool results to fan-out calls
    // POSITIONALLY (not by id). Run the same helper the `/v1/responses`
    // endpoint uses so a malformed block is rejected with a clear 400
    // and a reversed-but-valid block is rewritten to canonical
    // sibling order before dispatch.
    //
    // Pass `'anthropic'` so the helper's error strings reference
    // `tool_result` / `tool_use_id` — the vocabulary the Anthropic
    // caller actually posted — instead of OpenAI's
    // `function_call_output` / `call_id`. See iter-23 finding 4.
    const historyError = validateAndCanonicalizeHistoryToolOrder(messages, 'anthropic');
    if (historyError !== null) {
      sendAnthropicBadRequest(res, historyError);
      return;
    }

    // The Anthropic endpoint is stateless — every request allocates a
    // fresh `ChatSession` (via `getOrCreate(null, ...)`) and never
    // adopts it back into the cache. The `system` prompt is baked into
    // `messages` by `mapAnthropicRequest` and replayed via
    // `startFromHistory`, so there is no session-reuse path where a
    // stale system context could leak across requests. We still pass
    // the canonicalized system string to `getOrCreate` to keep the
    // registry API contract uniform across both OpenAI and Anthropic
    // endpoints — it is the caller's single "prefix/system state"
    // identity field.
    //
    // Anthropic's `system` field may be a string OR an array of content
    // blocks. We canonicalize to a deterministic JSON-stringified form
    // when it is structured so the equality check on a hypothetical
    // hit path would be stable across requests. Simple strings are
    // passed through unchanged to keep the common case readable.
    let requestedSystem: string | null;
    if (typeof body.system === 'string') {
      requestedSystem = body.system;
    } else if (body.system != null) {
      requestedSystem = JSON.stringify(body.system);
    } else {
      requestedSystem = null;
    }

    // Per-model execution mutex. Every dispatch through `/v1/messages`
    // serializes with every dispatch through `/v1/responses` for the
    // same model binding. The native `SessionCapableModel` is a single
    // mutable resource — one shared `cached_token_history` / one
    // `caches` vector per instance — so two concurrent `primeHistory`
    // + `startFromHistory` calls would clobber each other's KV state
    // even though each caller holds a distinct `ChatSession` wrapper.
    // Holding the registry's exclusive lock across the full dispatch
    // closes the race: at most one request at a time drives native
    // decode on this model, and the `finally` inside `withExclusive`
    // releases the lock regardless of whether the closure threw, so a
    // failed dispatch cannot leave the next waiter stuck.
    // Iter-35 finding 1: arm the AbortController now — we are past
    // every validation gate, so listeners have a matching detach in
    // the outer `finally` guarded by `abortListenersAttached`. The
    // streaming wrappers in `@mlx-node/lm` plumb this signal into
    // `_runChatStream`, which calls `handle.cancel()` on the native
    // ChatStreamHandle AND pushes a synthetic abort marker so the
    // pending `await waitForItem()` unblocks immediately on a dead
    // peer. Without the signal a client that dropped mid-eval would
    // pin this handler (and the per-model mutex) until the next
    // native chunk arrived.
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

    await sessionReg.withExclusive(async () => {
      // Hot-swap race guard inside the mutex.
      //
      // `withExclusive` can park this waiter behind a long-running
      // dispatch on the same model, and `ModelRegistry.register()` is
      // NOT coordinated with that lock — a concurrent
      // `registry.register(body.model, newModel)` can re-point the
      // friendly name while we are parked. Without this in-lock
      // re-read the closure would dispatch through the already-
      // captured `sessionReg`, running a session turn on a model
      // object that `body.model` no longer resolves to. Unlike
      // `/v1/responses` the Anthropic endpoint has no stored-identity
      // check later in the pipeline to catch the mismatch — two
      // requests for the same model name could silently be serviced
      // by different underlying models based purely on queue timing.
      //
      // Compare the live binding to the pre-lock snapshot captured
      // just before entering the mutex. Any drift — missing session
      // registry, missing instance id, session registry identity
      // change, or instance id change — is fatal and rejected with a
      // 400 so the caller can retry against the new binding.
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

      const session = sessionReg.getOrCreate(null, requestedSystem);

      // Visibility / wire-format tracker shared between the handler
      // body and the outer catch. The catch branches on
      // `responseMode` (JSON vs SSE) instead of `res.headersSent`,
      // which flips synchronously in `writeHead` before the body
      // lands and therefore lies about whether the client is
      // actually consuming JSON or SSE. Iter-33 finding 3: this is
      // the same wire-contract fix as `/v1/responses`; the
      // Anthropic handler is stateless so there is no session to
      // adopt, but the outer catch can still corrupt the response
      // by emitting an SSE frame into a JSON-declared body (or
      // hang the request by misclassifying an early SSE-write
      // crash).
      const visibility = createVisibility();

      try {
        if (body.stream === true) {
          const outcome = runSessionStreaming(session, messages, config, streamSignal);
          await handleStreamingNative(res, outcome.stream, body, outcome.wasCommitted, httpReq, visibility);
        } else {
          // Iter-35 finding 2 (part a): the non-streaming Anthropic
          // path has NO AbortSignal surface (native
          // `chatSessionStart` is a plain Promise, no cancel), so a
          // client that disconnects mid-generation still burns every
          // remaining token under this mutex. TODO(iter35): native
          // cancellation for `chatSession*` — until then the
          // disconnect-aware skip inside `handleNonStreaming`
          // short-circuits the JSON write on a dead peer so the
          // handler exits the mutex as soon as native decode returns.
          const result = await runSessionNonStreaming(session, messages, config);
          await handleNonStreaming(res, result, body, visibility);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown error during inference';
        if (visibility.responseMode === null) {
          sendAnthropicInternalError(res, message);
        } else if (visibility.responseMode === 'json') {
          // Already committed to JSON on the wire; emitting anything
          // else would corrupt the response. Destroy the socket so
          // the client sees a truncated JSON body instead of a
          // mismatched-MIME-type frame.
          try {
            res.destroy(err instanceof Error ? err : new Error(message));
          } catch {
            // Socket may already be gone.
          }
        } else {
          // `responseMode === 'sse'`: best-effort streaming `error`
          // event, but only if no terminal already landed. A double
          // terminal would confuse the client's event-stream state
          // machine.
          if (!visibility.terminalEmitted) {
            writeFallbackErrorSSE(res, 'error', { error: { type: 'api_error', message } });
          }
          try {
            endSSE(res);
          } catch {
            // Already closed / destroyed.
          }
        }
      }
    });
  } finally {
    // Iter-35 finding 1: drop the AbortController's socket/request
    // listeners so they do not keep the request object alive past
    // handler return. Only detach if we reached the installation
    // site — early-return validation gates exit before that point.
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
    // Release the dispatch lease on the ORIGINAL model object the
    // lease was acquired against (not a re-read of `body.model`,
    // which may have been hot-swapped while we held the mutex). A
    // pending teardown — `unregister()` called concurrently while
    // this dispatch held the lease — finalises here once the
    // in-flight counter drops to zero (the messages endpoint is
    // stateless, so it takes no `retainBinding` against the
    // post-commit persist counter).
    registry.releaseDispatchLease(leaseModel);
  }
}
