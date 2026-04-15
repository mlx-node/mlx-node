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

import type { ServerResponse } from 'node:http';

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
import type { AnthropicMessagesRequest } from '../types-anthropic.js';
import { validateAndCanonicalizeHistoryToolOrder } from './responses.js';

// ---------------------------------------------------------------------------
// Non-streaming path
// ---------------------------------------------------------------------------

async function handleNonStreaming(
  res: ServerResponse,
  result: ChatResult,
  body: AnthropicMessagesRequest,
): Promise<void> {
  const messageId = genId('msg_');
  const response = buildAnthropicResponse(result, body, messageId);
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(response));
}

// ---------------------------------------------------------------------------
// Streaming path
// ---------------------------------------------------------------------------

async function handleStreamingNative(
  res: ServerResponse,
  chatStream: AsyncGenerator<ChatStreamEvent>,
  body: AnthropicMessagesRequest,
  wasCommitted: () => boolean,
): Promise<void> {
  const messageId = genId('msg_');
  beginSSE(res);

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

  for await (const event of chatStream) {
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

  // Post-loop terminal emission. The producer's finally has now run
  // (either via the `break` after a done event or via natural
  // iterator exhaustion), so `wasCommitted()` reads an authoritative
  // `session.turns` baseline. Three cases:
  //
  //  1. sawDone && committed: happy path. Emit `message_delta` +
  //     `message_stop` so clients see a clean completion.
  //  2. sawDone && !committed: the final chunk carried
  //     `finishReason: 'error'` (the ChatSession gates `turnCount`
  //     on a non-error final chunk, so the session never committed).
  //     Emit a single streaming `error` SSE event in the Anthropic
  //     shape so the client can distinguish a real failure from a
  //     clean `message_stop`. Do NOT emit `message_stop` — that
  //     would report a failed generation as a successful one.
  //  3. !sawDone: the iterator exhausted before a terminal chunk
  //     arrived. The session also never committed in this path,
  //     so we emit an `error` SSE event the same way.
  //
  // The Anthropic `/v1/messages` endpoint is stateless and never
  // calls `sessionReg.adopt()`, so the registry cannot leak a
  // cached session on failure. This gate's sole job is to make the
  // client-visible event stream report failures accurately.
  const committed = wasCommitted();

  if (sawDone && committed) {
    const stopReason = terminalStopReason ?? 'end_turn';
    writeSSEEvent(res, 'message_delta', buildMessageDelta(stopReason, terminalNumTokens, terminalPromptTokens));
    writeSSEEvent(res, 'message_stop', buildMessageStop());
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
    const message =
      terminalErrorMessage ?? (sawDone ? 'model refused to commit the turn' : 'stream ended without a done event');
    writeSSEEvent(res, 'error', { type: 'error', error: { type: 'api_error', message } });
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
): MessagesStreamingOutcome {
  session.primeHistory(messages);
  const initialTurns = session.turns;
  return {
    stream: session.startFromHistoryStream(config),
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
  // deferred by `releaseBinding`) completes once the last dispatch
  // finishes.
  const lease = registry.acquireDispatchLease(body.model);
  if (!lease) {
    sendAnthropicInternalError(res, 'session registry missing for registered model');
    return;
  }
  const leaseModel = lease.model;
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

      try {
        if (body.stream === true) {
          const outcome = runSessionStreaming(session, messages, config);
          await handleStreamingNative(res, outcome.stream, body, outcome.wasCommitted);
        } else {
          const result = await runSessionNonStreaming(session, messages, config);
          await handleNonStreaming(res, result, body);
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
