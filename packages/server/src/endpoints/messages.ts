/**
 * POST /v1/messages — stateless Anthropic Messages API.
 *
 * Every request carries the full conversation in `req.messages`. The
 * Anthropic Messages API is stateless on the wire: there is no
 * `previous_response_id` to thread, and clients (e.g. Claude Code)
 * also do NOT propagate `prompt_cache_key` back to the server. The
 * cross-turn / cross-conversation prefix-reuse path is one of two
 * mutually-exclusive mechanisms, picked at request time based on
 * whether the underlying native model has the block-paged KV cache
 * adapter active (`SessionCapableModel.hasBlockPagedCache?.()`):
 *
 *   * **Paged-active path** (Qwen3 + LFM2 + Gemma4 are paged-active
 *     today; Qwen3.5 dense/MoE and Qianfan-OCR remain non-paged /
 *     default-off pending a perf decision and adapter wiring
 *     respectively). Each request allocates a fresh `ChatSession` via
 *     `SessionRegistry.createFreshSession()` and runs a full
 *     `session.reset()` + `primeHistory()` +
 *     `startFromHistory[Stream]()`. The JS-side warm slot is
 *     **not** consulted, **not** leased, and **not** adopted —
 *     cross-request prefix reuse is handled entirely by the native
 *     `BlockAllocator`'s content-addressed prefix-hash table, which
 *     refcounts SYS blocks shared across requests transparently
 *     (two parallel `/v1/messages` requests with the same system
 *     prompt run on distinct `ChatSession` objects but reference
 *     the same physical KV blocks). The non-streaming
 *     `X-Session-Cache` header is promoted from `fresh` to
 *     `prefix_hit` after dispatch when the engine reports
 *     `cachedTokens > 0`.
 *
 *   * **Non-paged path** (Qwen3.5 dense + MoE — default-off pending a
 *     perf decision; the Qianfan-OCR VLM — no adapter wired). Each
 *     request looks up the warm slot via
 *     `SessionRegistry.getOrCreateWarmAny(requestedSystem)`. On a
 *     HIT we keep the underlying native KV cache alive
 *     (`resetPreservingNativeCacheForWarmReuse` wipes only JS-side
 *     session state) so the native `verify_cache_prefix_direct` can
 *     recognize the cached prefix and re-prefill only the new
 *     suffix. On a MISS we run a full `session.reset()` to wipe
 *     both JS and native state — a fresh JS session does NOT imply
 *     a fresh native cache (the underlying `SessionCapableModel` is
 *     shared and its native `cached_token_history` persists across
 *     requests). After the dispatch settles we adopt the session
 *     back under the sentinel id `'__msg_warm__'` (or drop on
 *     uncommitted streams / thrown errors) so the next turn can
 *     lease it. The sentinel is never produced by either the OpenAI
 *     or the Anthropic wire format, so cross-endpoint capture via
 *     tier-1 is impossible by construction. The `/v1/responses` and
 *     `/v1/messages` endpoints still SHARE the single warm slot
 *     under the registry's single-warm invariant on this path — a
 *     turn on one side can evict the other's slot.
 *
 * The `prompt_cache_key` request field is still NOT exposed on this
 * endpoint. Cross-conversation block-level cache reuse on the
 * paged path is now driven by native content-addressing instead of
 * the JS warm slot, so adding the field is no longer a prerequisite
 * for that use case.
 */

import type { IncomingMessage, ServerResponse } from 'node:http';

import type { ChatConfig, ChatMessage, ChatResult, PerformanceMetrics } from '@mlx-node/core';
import type { ChatSession, ChatStreamEvent, SessionCapableModel } from '@mlx-node/lm';

import { resetPreservingNativeCacheForWarmReuse } from '../chat-session-warm-reuse.js';
import {
  sendAnthropicBadRequest,
  sendAnthropicInternalError,
  sendAnthropicNotFound,
  sendAnthropicRateLimit,
} from '../errors.js';
import type { IdleSweeper } from '../idle-sweeper.js';
import { canonicalizeSystemForCacheKey, mapAnthropicRequest } from '../mappers/anthropic-request.js';
import {
  buildAnthropicResponse,
  buildContentBlockDelta,
  buildContentBlockStart,
  buildContentBlockStop,
  buildMessageDelta,
  buildMessageStartEvent,
  buildMessageStop,
  containsToolCallMarkup,
  internalToolCallIdToAnthropic,
  recoverSuppressedToolCallText,
  mapStopReason,
} from '../mappers/anthropic-response.js';
import { genId } from '../mappers/response.js';
import type { ModelWorkCoordinator } from '../model-work-coordinator.js';
import type { ModelRegistry } from '../registry.js';
import { QueueFullError, type SessionRegistry } from '../session-registry.js';
import { StopSequenceBuffer } from '../stop-sequence-buffer.js';
import { beginSSE, endSSE, writeSSEEvent } from '../streaming.js';
import { longestSuffixPrefixOverlap } from '../text-recovery.js';
import { resolveServerTuningForUsage, type ServerTimingForUsage } from '../timing.js';
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
import { MAX_OUTPUT_TOKENS, validateAndCanonicalizeHistoryToolOrder } from './responses.js';

/**
 * Sentinel response id used to adopt and drop the per-model warm slot
 * for `/v1/messages` reuse. The Anthropic Messages API does not
 * produce a `previous_response_id` clients could echo back, and the
 * OpenAI `/v1/responses` side mints fresh `resp_*` ids — so this
 * literal can never collide with a tier-1 lookup from either
 * endpoint. Centralised here to keep the four call sites
 * (`adopt` on success, `drop` on failure, both for streaming and
 * non-streaming) in lockstep.
 */
const MESSAGES_WARM_SLOT_ID = '__msg_warm__';
const CLAUDE_CODE_TITLE_MAX_TOKENS = 128;

function withAdmissionControlledInference<T>(
  sessionReg: SessionRegistry,
  modelWorkCoordinator: ModelWorkCoordinator | undefined,
  fn: () => Promise<T>,
): Promise<T> {
  return sessionReg.withExclusive(() => (modelWorkCoordinator ? modelWorkCoordinator.withInference(fn) : fn()));
}

function requestAllowsToolUse(body: AnthropicMessagesRequest): boolean {
  return Array.isArray(body.tools) && body.tools.length > 0;
}

function hasSuppressedToolCalls(result: Pick<ChatResult, 'toolCalls'>, body: AnthropicMessagesRequest): boolean {
  return !requestAllowsToolUse(body) && result.toolCalls.some((t) => t.status === 'ok');
}

function applyOutputTokenLimit(config: ChatConfig, limit: number | undefined): ChatConfig {
  if (
    limit == null ||
    !Number.isFinite(limit) ||
    limit <= 0 ||
    config.maxNewTokens == null ||
    config.maxNewTokens <= limit
  ) {
    return config;
  }
  return { ...config, maxNewTokens: Math.floor(limit) };
}

function systemText(system: AnthropicMessagesRequest['system']): string {
  if (system == null) return '';
  if (typeof system === 'string') return system;
  return system
    .filter((block): block is Extract<(typeof system)[number], { type: 'text' }> => block.type === 'text')
    .map((block) => block.text)
    .join('\n');
}

function hasTitleJsonSchema(schema: unknown): boolean {
  if (schema == null || typeof schema !== 'object') return false;
  const obj = schema as {
    type?: unknown;
    properties?: unknown;
    required?: unknown;
  };
  if (obj.type !== 'object') return false;
  if (obj.properties == null || typeof obj.properties !== 'object') return false;
  const properties = obj.properties as Record<string, unknown>;
  const title = properties['title'];
  if (title == null || typeof title !== 'object') return false;
  if ((title as { type?: unknown }).type !== 'string') return false;
  return Array.isArray(obj.required) && obj.required.includes('title');
}

function isClaudeCodeTitleGenerationRequest(body: AnthropicMessagesRequest): boolean {
  if (requestAllowsToolUse(body)) return false;
  const format = body.output_config?.format;
  if (format?.type !== 'json_schema' || !hasTitleJsonSchema(format.schema)) return false;

  const prompt = systemText(body.system).toLowerCase();
  return (
    prompt.includes('generate a concise') &&
    prompt.includes('title') &&
    prompt.includes('return json') &&
    prompt.includes('"title"')
  );
}

function applyClaudeCodeTitleFastPath(config: ChatConfig, body: AnthropicMessagesRequest): ChatConfig {
  if (!isClaudeCodeTitleGenerationRequest(body)) return config;
  const cappedMax =
    config.maxNewTokens == null
      ? CLAUDE_CODE_TITLE_MAX_TOKENS
      : Math.min(config.maxNewTokens, CLAUDE_CODE_TITLE_MAX_TOKENS);
  return {
    ...config,
    maxNewTokens: cappedMax,
    reasoningEffort: 'none',
    thinkingTokenBudget: 0,
    includeReasoning: false,
  };
}

// Non-streaming path

async function handleNonStreaming(
  res: ServerResponse,
  result: ChatResult,
  body: AnthropicMessagesRequest,
  visibility: TransportVisibility,
  stopSequences: string[],
  serverTiming?: ServerTimingForUsage,
): Promise<void> {
  const messageId = genId('msg_');

  // Honor client-supplied `stop_sequences`: scan the SAME visible text the
  // response builder will emit for the earliest configured stop string. When
  // the request disallows tools but the parser still produced a tool call and
  // `result.text` is empty, `buildAnthropicContent` emits the recovered
  // suppressed-tool text — so the scan must mirror that recovery gate and run
  // over the recovered text, not the empty `result.text`. The scan does
  // push+flush so a complete stop that `push()` held back (a longer
  // overlapping stop was still viable) is resolved at end-of-text, matching
  // the streaming done-path. On a match we truncate the text the response is
  // built from at the match (dropping the stop string and everything after it)
  // and report `stop_reason: 'stop_sequence'` + `stop_sequence: '<matched>'`;
  // `buildAnthropicResponse` then suppresses tool calls and the recovery
  // branch and emits the truncated text verbatim. The native `ChatResult` is
  // left untouched. With no match `responseResult` stays `result` (full text
  // retained — `flush()` releases any held incomplete partial as normal text),
  // so behavior is byte-identical to a request without `stop_sequences`.
  const visibleText =
    !requestAllowsToolUse(body) &&
    result.text.length === 0 &&
    result.toolCalls.filter((t) => t.status === 'ok').length > 0 &&
    containsToolCallMarkup(result.rawText)
      ? recoverSuppressedToolCallText(result.rawText)
      : result.text;

  let matchedStopSequence: string | null = null;
  let responseResult = result;
  if (stopSequences.length > 0) {
    const stopBuffer = new StopSequenceBuffer(stopSequences);
    const pushed = stopBuffer.push(visibleText);
    const flushed = stopBuffer.flush();
    const matched = pushed.matched ?? flushed.matched;
    if (matched !== null) {
      matchedStopSequence = matched;
      responseResult = { ...result, text: pushed.safeText + flushed.safeText };
    }
  }

  // `result.performance` is only populated when `reportPerformance: true`
  // rides on the underlying `ChatConfig`; otherwise the field is
  // `undefined` and the mapper elides the wire-extension fields. The
  // launcher wires the flag on for verbose-log builds and leaves it off
  // by default, matching how `cachedTokens` is treated through
  // `buildAnthropicResponse`.
  const response = buildAnthropicResponse(
    responseResult,
    body,
    messageId,
    result.performance,
    requestAllowsToolUse(body),
    serverTiming,
    matchedStopSequence,
  );

  // Native `chatSession*` has no AbortSignal surface yet, so a client that
  // disconnects mid-decode still burns every remaining token under the
  // per-model mutex. Disconnect handling is delegated to `endJson`'s
  // pre-entry destroyed check, which rejects synchronously after `responseMode`
  // has been committed to 'json' — the outer catch then destroys the socket.
  await endJson(res, JSON.stringify(response), visibility);
}

// Streaming path

/**
 * Handler-side success signal for the streaming path. `ok === true` ONLY when
 * we reached the clean `message_stop` terminal — i.e. `successful` was true at
 * the post-loop gate. Every failure path that emits the streaming `error`
 * terminal (mid-decode throw, client abort, `finishReason=error`, iterator
 * exhaustion, missing-done) returns `ok: false`. The caller pairs this with
 * the producer-side `wasCommitted()` to decide adopt vs. drop on the warm
 * slot — both must be true to adopt. See the gate in `handleCreateMessage`.
 */
interface MessagesStreamingHandlerResult {
  ok: boolean;
  suppressedToolCalls: boolean;
}

async function handleStreamingNative(
  res: ServerResponse,
  chatStream: AsyncGenerator<ChatStreamEvent>,
  body: AnthropicMessagesRequest,
  wasCommitted: () => boolean,
  httpReq: IncomingMessage | undefined,
  visibility: TransportVisibility,
  emitReasoning: boolean,
  stopSequences: string[],
  serverTiming?: ServerTimingForUsage,
): Promise<MessagesStreamingHandlerResult> {
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
  // Whitespace-only text seen before any non-whitespace content is buffered
  // here so we don't open a text content block that the client would have
  // to render as a stray `"\n\n"` immediately before a tool_use block.
  // Flushed lazily when the first non-whitespace text delta arrives;
  // dropped silently when a non-text block (tool_use) is about to open or
  // when the stream ends without further text. Once `hasEmittedText` flips
  // true (a real text block exists) this buffer is no longer consulted —
  // subsequent whitespace-only deltas pass through to keep streamed text
  // byte-accurate.
  let pendingLeadingWhitespace = '';
  // Mirror of the actual streamed text body, used by the malformed-tool-call
  // recovery branches below. `emittedTextLength` counts bytes of streamed text
  // — but `event.text` on the terminal `done` chunk is the post-</think>-trim
  // cleaned text from the native `split_at_think_end`, so streamed and final
  // prefixes can diverge (e.g. streamed=`"\n\n<tool_call>..."`,
  // finalText=`"<tool_call>..."`). The recovery branches use
  // `longestSuffixPrefixOverlap(emittedText, finalText)` to find the unsent
  // suffix instead of a length-based slice that would chop characters.
  let emittedText = '';
  const tagBuffer = new ToolCallTagBuffer();
  // Client-supplied `stop_sequences` detector. Feeds on the visible text that
  // survives `tagBuffer` (structural-marker stripping) so it never sees tool
  // markup. An empty `stopSequences` constructs a pass-through buffer
  // (`push` returns its input verbatim, `flush` returns ''), so the wire is
  // byte-identical to a request without `stop_sequences`. When a stop string
  // matches, `matchedStopSequence` is recorded, all later visible text is
  // suppressed, and the done-path emits nothing past the stop. The done-path
  // also scans the terminal / recovered visible text on this SAME buffer (with
  // any held partial still in place), so a stop straddling the stream/terminal
  // boundary is caught with buffer continuity.
  const stopBuffer = new StopSequenceBuffer(stopSequences);
  let matchedStopSequence: string | null = null;

  // Terminal emission is deferred until after the loop drains so `wasCommitted()`
  // reads an authoritative `session.turns`. On a committed done chunk we emit
  // `message_delta` + `message_stop`; on an uncommitted terminal (finishReason=error,
  // mid-decode throw, client abort, iterator exhaustion) we emit a single streaming
  // `error` event and withhold `message_stop`.
  let sawDone = false;
  let terminalStopReason: string | null = null;
  let terminalNumTokens = 0;
  let terminalPromptTokens: number | undefined;
  // Captured from the terminal `done` chunk so the success-branch
  // `buildMessageDelta` can emit Anthropic-spec cache accounting
  // (`cache_read_input_tokens` + reduced `input_tokens`) on warm
  // hits. Stays `undefined` on streams whose terminal chunk omits
  // the field — mocks and any future in-process driver that hasn't
  // adopted the surface — so `buildMessageDelta` falls back to the
  // pre-Round-6 behaviour.
  let terminalCachedTokens: number | undefined;
  // Captured from the terminal `done` chunk so the success-branch
  // `buildMessageDelta` can attach the server-extension perf fields
  // (`time_to_first_token_ms`, `prefill_tokens_per_second`,
  // `decode_tokens_per_second`). Stays `undefined` when the underlying
  // dispatch did not opt into performance reporting (or when a mock
  // bridge omits the field) — the mapper elides the fields rather
  // than emitting zeros.
  let terminalPerformance: PerformanceMetrics | undefined;
  let terminalErrorMessage: string | null = null;
  const allowToolUse = requestAllowsToolUse(body);
  let suppressedToolCalls = false;

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

        // Flush the tag buffer's residual but keep the stop buffer intact: it
        // may still hold a partial from the streamed deltas, and a stop can
        // straddle the boundary between that held partial, the tag residue, and
        // the native terminal/recovered text. All of it runs through the SAME
        // buffer, in stream order, with a single flush, BEFORE the stop match,
        // the emitted text, and the tool decision are finalized.
        const tagResidual = tagBuffer.flush();
        const heldPartial = stopBuffer.pending;

        const parsedToolCalls = event.toolCalls.filter((t) => t.status === 'ok');
        if (!allowToolUse && parsedToolCalls.length > 0) {
          suppressedToolCalls = true;
        }
        const finalText =
          !allowToolUse &&
          event.text.length === 0 &&
          parsedToolCalls.length > 0 &&
          containsToolCallMarkup(event.rawText)
            ? recoverSuppressedToolCallText(event.rawText)
            : event.text;

        // The visible text the stream already RECEIVED, in stream order: what
        // reached the wire (`emittedText`), the parked leading whitespace the
        // detector already cleared but no block has shown yet
        // (`pendingLeadingWhitespace`), the still-held detector partial
        // (`heldPartial`), and the tag residue about to be scanned
        // (`tagResidual`). Every parked/held byte is counted exactly once so the
        // recovered terminal text is only the suffix of `finalText` the stream
        // has not already accounted for — omitting the parked whitespace would
        // make a full-text `finalText` look entirely unsent and replay the
        // already-received prefix.
        const streamedReceived = emittedText + pendingLeadingWhitespace + heldPartial + tagResidual;
        let recoveredTail = '';
        if (finalText) {
          if (!hasEmittedText && heldPartial.length === 0 && tagResidual.length === 0) {
            // Nothing was streamed or held: the whole `finalText` is terminal.
            recoveredTail = finalText;
          } else if (!streamedReceived.includes(finalText)) {
            // `finalText` extends past what the stream produced: recover the
            // suffix beyond the longest overlap. The `includes` guard skips the
            // duplicate-trim case where `finalText` is a substring of the
            // received text (native `.trim()` / post-`</think>` shrinkage).
            recoveredTail = finalText.slice(longestSuffixPrefixOverlap(streamedReceived, finalText));
          }
        }

        // One continuous scan — held partial (already buffered) + tag residue +
        // recovered tail, in stream order, with a single flush at the end. A
        // stop matched anywhere here is caught with buffer continuity, and
        // `matchedStopSequence` is finalized before tool emission and
        // `stop_reason`. With an empty `stopSequences` the buffer is a
        // pass-through, so `terminalVisible` equals the released text verbatim.
        let terminalVisible = '';
        for (const segment of [tagResidual, recoveredTail]) {
          const pushed = stopBuffer.push(segment);
          if (pushed.matched !== null) {
            matchedStopSequence = pushed.matched;
          }
          terminalVisible += pushed.safeText;
        }
        const flushed = stopBuffer.flush();
        if (flushed.matched !== null) {
          matchedStopSequence = flushed.matched;
        }
        terminalVisible += flushed.safeText;

        // Parked leading whitespace belongs in front of RELEASED held content
        // (a stop-buffer partial or tag residue). When the terminal text is
        // purely native `finalText` recovery, `finalText` already carries that
        // whitespace, so prepending it would double those bytes.
        const prependParked = heldPartial.length > 0 || tagResidual.length > 0;

        // Close a dangling reasoning block before any terminal text block opens.
        if (hasEmittedThinking && !hasEmittedText) {
          writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex - 1));
        }

        if (hasEmittedText) {
          // A text block is already open from the streamed deltas: append the
          // newly released terminal text (if any), then close the block.
          if (terminalVisible) {
            emittedText += terminalVisible;
            emittedTextLength += terminalVisible.length;
            writeSSEEvent(
              res,
              'content_block_delta',
              buildContentBlockDelta(contentBlockIndex, {
                type: 'text_delta',
                text: terminalVisible,
              }),
            );
          }
          writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex));
          contentBlockIndex++;
          pendingLeadingWhitespace = '';
        } else {
          // No text block open yet. Build the block body from the terminal text
          // (fronted by parked whitespace only when it precedes released held
          // content), or from the parked whitespace alone when a stop truncated
          // the turn right after it.
          const body = terminalVisible
            ? prependParked
              ? pendingLeadingWhitespace + terminalVisible
              : terminalVisible
            : matchedStopSequence !== null
              ? pendingLeadingWhitespace
              : '';
          pendingLeadingWhitespace = '';
          // Open a text block for non-whitespace content, for a stop-truncated
          // prefix (so the streamed body equals the non-streaming one), or for
          // pure native `finalText` recovery (mirrors emitting recovered text
          // verbatim). A stop-matched turn always opens a text block — empty if
          // the stop consumed all visible output — so the reconstructed content
          // matches the non-streaming `[{type:'text', text:''}]`. Whitespace-only
          // released held content opens no block.
          const openTextBlock =
            matchedStopSequence !== null || (body.length > 0 && (body.trim().length > 0 || !prependParked));
          if (openTextBlock) {
            hasEmittedText = true;
            writeSSEEvent(
              res,
              'content_block_start',
              buildContentBlockStart(contentBlockIndex, {
                type: 'text',
                text: '',
              }),
            );
            if (body.length > 0) {
              emittedText += body;
              emittedTextLength += body.length;
              writeSSEEvent(
                res,
                'content_block_delta',
                buildContentBlockDelta(contentBlockIndex, {
                  type: 'text_delta',
                  text: body,
                }),
              );
            }
            writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex));
            contentBlockIndex++;
          }
        }

        // Decide tool emission only AFTER the full terminal stop scan: when a
        // stop matched ANYWHERE (the pre-tool visible text OR the
        // terminal/recovered text) the tool calls are suppressed so a streamed
        // turn never carries both a tool_use block and
        // `stop_reason: 'stop_sequence'`. This keeps streaming in lockstep with
        // the non-streaming path (`buildAnthropicResponse`).
        const okToolCalls = allowToolUse && matchedStopSequence === null ? parsedToolCalls : [];
        const hasToolCalls = okToolCalls.length > 0;

        for (const tc of okToolCalls) {
          // Translate native `call_<uuid>` ids (minted by the Rust parser,
          // which keeps the OpenAI Responses convention) into the
          // Anthropic-spec `toolu_<uuid>` shape at the wire boundary.
          // The `genId('toolu_')` fallback covers the case where the
          // native side did not populate an id (an in-process driver or
          // a legacy bridge).
          const toolId = tc.id != null ? internalToolCallIdToAnthropic(tc.id) : genId('toolu_');
          const parsedInput =
            typeof tc.arguments === 'string'
              ? (JSON.parse(tc.arguments) as Record<string, unknown>)
              : (tc.arguments as Record<string, unknown>);

          writeSSEEvent(
            res,
            'content_block_start',
            buildContentBlockStart(contentBlockIndex, {
              type: 'tool_use',
              id: toolId,
              name: tc.name,
              input: {},
            }),
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
        terminalStopReason = mapStopReason(event.finishReason, hasToolCalls, matchedStopSequence);
        terminalNumTokens = event.numTokens;
        terminalPromptTokens = event.promptTokens;
        terminalCachedTokens = event.cachedTokens;
        terminalPerformance = event.performance;
        break;
      }

      // Delta event
      if (event.isReasoning) {
        if (!emitReasoning) continue;
        const deltaText = event.text.replace(/<\/think>/g, '');
        if (!deltaText) continue;

        if (!hasEmittedThinking) {
          hasEmittedThinking = true;
          writeSSEEvent(
            res,
            'content_block_start',
            buildContentBlockStart(contentBlockIndex, {
              type: 'thinking',
              thinking: '',
            }),
          );
          contentBlockIndex++;
        }
        writeSSEEvent(
          res,
          'content_block_delta',
          buildContentBlockDelta(contentBlockIndex - 1, {
            type: 'thinking_delta',
            thinking: deltaText,
          }),
        );
      } else {
        // Text delta with structural-marker buffering. Even when the
        // request did not advertise tools, model-side tool/channel/turn
        // markers are transport structure, not user-visible text.
        const { safeText, tagFound, cleanPrefix } = tagBuffer.push(event.text);
        if (tagFound) {
          // A structural tag (`<tool_call>` etc.) follows, so the visible
          // text before it terminates here. Only `cleanPrefix` is fresh
          // model text — route it through the stop-sequence detector so a
          // configured stop string landing in it (e.g. "...HALT " right
          // before a `<tool_call>`) is honored, not leaked. Do NOT flush the
          // detector here: a held partial (e.g. "HA" of "HALT") must stay
          // buffered, because the native cleaned done text can reconstitute
          // the bytes that followed the suppressed tag and complete the stop
          // across that boundary. The done-path scans the held partial
          // together with the terminal/recovered text and resolves it —
          // releasing it as visible text if it cannot complete, or suppressing
          // it if it does. `pendingLeadingWhitespace` is whitespace the
          // detector already cleared on an earlier delta (held back only
          // because no text block was open yet), so it is prepended OUTSIDE
          // the buffer: re-pushing it would double-scan it AND, because the
          // buffer queues it after any held partial, invert stream order
          // (e.g. held "H" + buffered " " -> "H ") or forge a false match. On
          // a match `matchedStopSequence` is recorded so the terminal reports
          // `stop_sequence`. With an empty `stopSequences` the detector is a
          // pass-through, so `visibleText === pendingLeadingWhitespace +
          // cleanPrefix` and the wire is byte-identical to today.
          const stopPushed = stopBuffer.push(cleanPrefix);
          if (stopPushed.matched !== null) {
            matchedStopSequence = stopPushed.matched;
          }
          const visibleText = pendingLeadingWhitespace + stopPushed.safeText;
          // Mirror the original `cleanPrefix.trim()` gate, now on the
          // detector's safe text: emit only when there is non-whitespace to
          // show, so a pure-whitespace prefix never ratifies a stray
          // whitespace-only text block before the tool_use frame. When the
          // safe text is whitespace-only (e.g. the detector is still holding a
          // partial), KEEP it parked so the done-path can join it with
          // whatever the held partial releases — clearing it here would drop
          // it before that text block opens.
          if (visibleText.trim().length > 0) {
            if (!hasEmittedText) {
              if (hasEmittedThinking) {
                writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex - 1));
              }
              hasEmittedText = true;
              writeSSEEvent(
                res,
                'content_block_start',
                buildContentBlockStart(contentBlockIndex, {
                  type: 'text',
                  text: '',
                }),
              );
            }
            pendingLeadingWhitespace = '';
            emittedText += visibleText;
            emittedTextLength += visibleText.length;
            writeSSEEvent(
              res,
              'content_block_delta',
              buildContentBlockDelta(contentBlockIndex, {
                type: 'text_delta',
                text: visibleText,
              }),
            );
          } else {
            pendingLeadingWhitespace = hasEmittedText ? '' : visibleText;
          }
        } else if (safeText) {
          // Run the tag-buffer's visible text through the stop-sequence
          // detector. `visibleText` is what survives suppression: the buffer
          // holds back a trailing suffix that could be the start of a stop
          // string (released on a later push or at flush), and once a full
          // stop string matches it returns empty `safeText` for the rest of
          // the stream. We record the match and keep consuming so the native
          // `done` chunk still fires the commit gate and history commit.
          const stopResult = stopBuffer.push(safeText);
          if (stopResult.matched !== null) {
            matchedStopSequence = stopResult.matched;
          }
          const visibleText = stopResult.safeText;
          if (visibleText) {
            if (!hasEmittedText) {
              // Hold back leading whitespace-only text so a `\n\n` emitted
              // right before a `<tool_call>` tag never gets ratified into a
              // standalone text content block. We can't open the block now
              // because we don't yet know whether the next event is a real
              // text delta (in which case the buffered prefix is flushed
              // together with it) or a structural tag (in which case the
              // buffer is dropped silently at tag-found / done time). When
              // any non-whitespace arrives we ratify the block exactly
              // once with `pendingLeadingWhitespace + visibleText`.
              const combined = pendingLeadingWhitespace + visibleText;
              if (combined.trim().length === 0) {
                pendingLeadingWhitespace = combined;
              } else {
                if (hasEmittedThinking) {
                  writeSSEEvent(res, 'content_block_stop', buildContentBlockStop(contentBlockIndex - 1));
                }
                hasEmittedText = true;
                writeSSEEvent(
                  res,
                  'content_block_start',
                  buildContentBlockStart(contentBlockIndex, {
                    type: 'text',
                    text: '',
                  }),
                );
                pendingLeadingWhitespace = '';
                emittedText += combined;
                emittedTextLength += combined.length;
                writeSSEEvent(
                  res,
                  'content_block_delta',
                  buildContentBlockDelta(contentBlockIndex, {
                    type: 'text_delta',
                    text: combined,
                  }),
                );
              }
            } else {
              emittedText += visibleText;
              emittedTextLength += visibleText.length;
              writeSSEEvent(
                res,
                'content_block_delta',
                buildContentBlockDelta(contentBlockIndex, {
                  type: 'text_delta',
                  text: visibleText,
                }),
              );
            }
          }
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

  // Success requires ALL of: sawDone, wasCommitted, no terminal error, no thrown
  // error, no client abort. `terminalErrorMessage` is set when a stream done event
  // arrives with `finishReason=error` (or other in-band model error paths) — those
  // turns must route to the failure epilogue so we emit a streaming `error` and
  // withhold `message_stop`. Every failure path emits a streaming `error` and
  // withholds `message_stop`.
  const committed = wasCommitted();
  const successful = sawDone && committed && terminalErrorMessage == null && thrownError == null && !clientAborted;

  if (successful) {
    const stopReason = terminalStopReason ?? 'end_turn';
    writeSSEEvent(
      res,
      'message_delta',
      buildMessageDelta(
        stopReason,
        terminalNumTokens,
        terminalPromptTokens,
        terminalCachedTokens,
        terminalPerformance,
        serverTiming,
        matchedStopSequence,
      ),
    );
    // HTTP/1.1 chunked-encoding trailer: report the engine's cache-hit
    // count once the SSE stream has settled. The header has to wait
    // for `terminalCachedTokens` because `beginSSE` flushes response
    // headers before the dispatch returns. Trailer-aware clients
    // (curl `--trailer-name`, custom HTTP libraries, the verbose
    // logger's response listener) get the authoritative value;
    // SSE-only clients get the same value via the `usage.cache_read_input_tokens`
    // field on `message_delta`. The `Trailer: X-Cached-Tokens` header
    // was announced before `beginSSE` flushed (see messages.ts call site).
    if (typeof terminalCachedTokens === 'number' && terminalCachedTokens > 0) {
      try {
        res.addTrailers({ 'X-Cached-Tokens': String(terminalCachedTokens) });
      } catch {
        // res.addTrailers throws if headers/trailers were not announced
        // up front — non-fatal; the SSE usage field still carries the value.
      }
    }
    await flushTerminalSSE(res, 'message_stop', buildMessageStop(), visibility);
    endSSE(res);
    return { ok: true, suppressedToolCalls };
  }
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
  endSSE(res);
  return { ok: false, suppressedToolCalls };
}

// Session routing

/**
 * Non-streaming dispatch outcome. Mirrors the `{ result, committed }`
 * shape from the sibling `/v1/responses` endpoint
 * (`responses.ts:1361-1362`) so the warm-slot adopt site can dual-gate
 * on commit success. `committed` is measured against `initialTurns`
 * captured AFTER `primeHistory` AND requires a non-error
 * `finishReason` — see the comment at the gate inside the helper.
 */
interface MessagesNonStreamingOutcome {
  result: ChatResult;
  committed: boolean;
}

/** Prime a session with the full history and run a single turn. */
async function runSessionNonStreaming(
  session: ChatSession<SessionCapableModel>,
  messages: ChatMessage[],
  config: ChatConfig,
  resetNativeCache: boolean,
): Promise<MessagesNonStreamingOutcome> {
  // Dual-branch reset gated by the caller's native-cache policy:
  //
  //   * Full native reset (`resetNativeCache === true`) — run a full
  //     `session.reset()`. A fresh JS session does NOT imply a fresh
  //     native cache — the underlying `SessionCapableModel` is shared
  //     across `ChatSession` lifetimes via `ModelRegistry`, and its
  //     native `cached_token_history` persists across requests. After
  //     the native refactor moved the unconditional wipe out of
  //     `chat_session_start_sync` into the miss branch of
  //     `verify_cache_prefix_direct`, skipping the wipe here would
  //     silently reuse whatever prefix happened to overlap with the
  //     previous (unrelated) request — the cross-request
  //     cache-affinity side channel documented at length in
  //     `responses.ts` (around the matching `runSessionNonStreaming`
  //     branches). Only registry HITS are authorized for cache reuse.
  //
  //   * Preserve native cache (`resetNativeCache === false`) — run the JS-only
  //     `resetPreservingNativeCacheForWarmReuse` so the registry-leased
  //     native KV cache stays alive for `verify_cache_prefix_direct` or
  //     the paged adapter's content-addressed prefix lookup to recover
  //     the reused prefix on this turn. `primeHistory` requires
  //     `turnCount === 0`, which the helper guarantees by wiping
  //     JS-side state only.
  if (resetNativeCache) {
    await session.reset();
  } else {
    await resetPreservingNativeCacheForWarmReuse(session);
  }
  session.primeHistory(messages);
  const initialTurns = session.turns;
  const result = await session.startFromHistory(config);
  // Mirror the streaming-side dual-gate (`streamResult.ok &&
  // outcome.wasCommitted()`) and the sibling `/v1/responses` adopt
  // gate. `ChatSession.startFromHistory` advances `turnCount`
  // unconditionally on a clean resolve, so `session.turns >
  // initialTurns` alone never trips today — every native error path
  // throws. The `finishReason !== 'error'` clause defends the
  // invariant LOCALLY so a future Rust change that resolves
  // `chat_session_start_sync` with `Ok(finish_reason="error")` cannot
  // silently poison the warm slot.
  const committed = session.turns > initialTurns && result.finishReason !== 'error';
  return { result, committed };
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
  resetNativeCache: boolean,
): Promise<MessagesStreamingOutcome> {
  // Same dual-branch reset as `runSessionNonStreaming`: native-reset
  // requests wipe both JS and native state to block the cross-request
  // cache-affinity leak described at length in `responses.ts`; native-
  // preserving requests wipe JS-only so the non-paged warm slot or the
  // paged content-addressed cache can recover a verified native prefix.
  // `initialTurns` MUST be captured AFTER the reset zeroes `turns` so
  // the committed check reads correctly.
  if (resetNativeCache) {
    await session.reset();
  } else {
    await resetPreservingNativeCacheForWarmReuse(session);
  }
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
  modelWorkCoordinator?: ModelWorkCoordinator,
): Promise<void> {
  const handlerStartedAt = Date.now();
  let serverModelResolveMs: number | undefined;
  // Split observability for the resolve path: a request that arrives
  // milliseconds after a peer's cold-load should not be billed the full
  // load latency as if it drove the load itself. `serverLoadWaitMs`
  // captures wall-clock spent blocked on the writer lock (whether
  // waiting on a peer or self-loading); `serverLoadOwner` is true only
  // when this request acquired the lock without contention.
  let serverLoadWaitMs: number | undefined;
  let serverLoadOwner: boolean | undefined;

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
  if (body.max_tokens > MAX_OUTPUT_TOKENS) {
    // The field is present and a positive integer but too large: the native
    // `ChatConfig.max_new_tokens` is `i32`, and NAPI truncates a JS integer
    // above `i32::MAX` to a NEGATIVE value (then clamped to 0 → a silent empty
    // completion), so an over-large budget must 400 with a clear message
    // rather than be reported as "missing" or silently no-op.
    sendAnthropicBadRequest(res, `Field "max_tokens" must be an integer between 1 and ${MAX_OUTPUT_TOKENS}`);
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
  // Client-supplied `stop_sequences`, normalized by the mapper (absent/empty
  // dropped). Threaded into the streaming + non-streaming handlers, which own
  // the detection/truncation. `ChatConfig` has no native stop field, so this
  // rides alongside `config` from the mapper.
  let mappedStopSequences: string[];
  try {
    ({
      messages: mappedMessages,
      config: mappedConfig,
      stopSequences: mappedStopSequences,
    } = mapAnthropicRequest(body));
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
      const resolveStartedAt = Date.now();
      const runResolve = () =>
        idleSweeper ? idleSweeper.withSuspendedDrains(() => resolveModel(body.model)) : resolveModel(body.model);
      if (modelWorkCoordinator) {
        // Use the instrumented variant so we can tell whether this
        // request actually drove the load (owner) or merely parked
        // behind a peer's in-flight load. Without the split, two
        // requests racing into a 60s cold-load both report
        // `resolve_ms=60000` and observers can't tell which one paid
        // the cost vs. inherited the wait.
        //
        // The coordinator internally partitions the call into a wait
        // phase (`acquireWrite()`) and an own-execution phase (`fn`)
        // so `waitMs + ownMs` covers the total elapsed time without
        // overlap. We plumb them straight through into the matching
        // observability fields:
        //  - owner driving a cold load → waitMs ≈ 0, ownMs ≈ load duration
        //  - follower parked behind peer → waitMs ≈ peer load, ownMs ≈ 0
        //  - already-loaded fast path  → waitMs ≈ 0, ownMs ≈ 0
        // This matches the documented contract in `timing.ts` where
        // `server_model_resolve_ms` excludes peer-wait time.
        const outcome = await modelWorkCoordinator.withModelLoadInstrumented(runResolve);
        serverLoadOwner = outcome.owner;
        serverLoadWaitMs = outcome.waitMs;
        serverModelResolveMs = outcome.ownMs;
      } else {
        await runResolve();
        serverModelResolveMs = Date.now() - resolveStartedAt;
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
    mappedConfig = applyOutputTokenLimit(mappedConfig, sessionReg.outputTokenLimit);
    mappedConfig = applyClaudeCodeTitleFastPath(mappedConfig, body);
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
    const stopSequences: string[] = mappedStopSequences;

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
    // `getOrCreate` to keep the registry API uniform with `/v1/responses`. The
    // helper is shared with `mapAnthropicRequest`'s system loop so the cache-key
    // view and the mapped messages can never drift — both drop the rotating
    // Anthropic billing-header block (cf. `canonicalizeSystemForCacheKey`).
    const requestedSystem = canonicalizeSystemForCacheKey(body.system);

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
      const mutexQueuedAt = Date.now();
      const runInference = () =>
        withAdmissionControlledInference(sessionReg, modelWorkCoordinator, async () => {
          const serverTiming: ServerTimingForUsage = {
            server_model_resolve_ms: serverModelResolveMs,
            server_load_wait_ms: serverLoadWaitMs,
            server_load_owner: serverLoadOwner,
            server_queue_ms: Date.now() - mutexQueuedAt,
            server_pre_inference_ms: Date.now() - handlerStartedAt,
            ...resolveServerTuningForUsage(),
          };
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

          // Per-model session selection for `/v1/messages` reuse.
          //
          // Two paths, gated on whether the underlying native model has
          // the block-paged KV cache adapter active
          // (`hasBlockPagedCache()` — captured at load time from
          // `<Inner>::paged_adapter.is_some()` and surfaced by the
          // `SessionCapableModel` structural interface):
          //
          //   * **Paged-active** (Qwen3 + LFM2 + Gemma4 today; Qwen3.5
          //     dense + Qwen3.5 MoE once their perf trade-off is
          //     decided). Allocate a fresh `ChatSession` per request via
          //     `createFreshSession()`, do NOT touch the warm slot.
          //     Cross-turn / cross-conversation prefix reuse is handled
          //     entirely by the native `BlockAllocator`'s prefix-hash
          //     table: SYS blocks shared across requests are refcounted
          //     transparently, so two parallel `/v1/messages` requests
          //     sharing a system prompt both run on distinct
          //     `ChatSession` objects but reference the SAME physical
          //     KV blocks. The JS-side warm slot would only serialize
          //     them and force one into cold replay.
          //
          //   * **Non-paged** (Qwen3.5 dense + MoE — default-OFF pending
          //     a perf decision against the compiled C++ flat path;
          //     the Qianfan-OCR VLM — no adapter wired). Fall through to
          //     `getOrCreateWarmAny`, which is the ONLY cross-conversation
          //     reuse mechanism these models have. The Anthropic Messages
          //     API is stateless on the wire (no `previous_response_id`,
          //     clients don't propagate `prompt_cache_key`), so without
          //     the warm slot every turn is a full cold start.
          //
          // The `hasBlockPagedCache?()` getter is optional on the
          // structural interface so the `QianfanOCRModel` VLM (which
          // has no paged-adapter wiring) still satisfies the type
          // contract — a missing getter falls into the non-paged branch
          // here.
          //
          // Adoption stays keyed by the literal sentinel
          // `MESSAGES_WARM_SLOT_ID = '__msg_warm__'`. The Anthropic
          // Messages API never produces a `previous_response_id` clients
          // could echo back (and the OpenAI side mints `resp_*` ids),
          // so cross-endpoint capture via tier-1 is impossible by
          // construction — no `/v1/responses` request can collide with
          // the sentinel through the tier-1 path.
          //
          // The two endpoints DO share the single warm slot under the
          // registry's single-warm invariant on the non-paged path: a
          // `/v1/messages` turn following a `/v1/responses` turn can
          // evict (and vice versa). On the paged path neither side
          // touches the warm slot, so cross-endpoint contention
          // disappears.
          //
          // The `prompt_cache_key` request field is still NOT exposed
          // on this endpoint. Cross-conversation block-level cache
          // reuse on paged-active models is now driven by native
          // content-addressing instead of the JS warm slot, so adding
          // the field is no longer a prerequisite for that use case.
          const pagedActive = leaseModel.hasBlockPagedCache?.() === true;
          const lookup = pagedActive ? sessionReg.createFreshSession() : sessionReg.getOrCreateWarmAny(requestedSystem);
          const session = lookup.session;
          // `X-Session-Cache` observability header.
          //
          // Non-paged path:
          //   * Non-streaming: set the optimistic `prefix_hit` value
          //     BEFORE dispatch on `lookup.hit` (so the header is on the
          //     wire even if the dispatch throws) and demote
          //     post-dispatch to `fresh` when the warm slot was leased
          //     but native prefix reuse did not actually happen
          //     (`result.cachedTokens === 0`). `res.end` has not fired
          //     yet, so the overwrite still lands on the wire.
          //   * Streaming: emits `streaming` to signal the authoritative
          //     post-dispatch value rides on the SSE stream
          //     (`message_delta.usage.cache_read_input_tokens` and the
          //     `X-Cached-Tokens` HTTP trailer, set below in
          //     `handleStreamingNative` once `terminalCachedTokens` is
          //     known). Reporting `fresh` here would be a lie — the
          //     paged engine routinely returns `cachedTokens > 0` on
          //     turn-2+ and the prior `'fresh'` default falsely advertised
          //     a cache miss. The previous comment documented this as
          //     intentional but it was a logging bug.
          //
          // Paged path:
          //   * Non-streaming: `lookup.hit` is always `false` (we
          //     `createFreshSession`); the post-dispatch promotion
          //     branch flips `prefix_hit` when the native engine
          //     reports `cachedTokens > 0`, which is the authoritative
          //     signal that the block allocator's content-addressed
          //     reuse picked up shared SYS blocks on this turn.
          //   * Streaming: same `streaming` value as non-paged; the SSE
          //     `usage.cache_read_input_tokens` field carries the
          //     authoritative value.
          //
          // Header values: `'fresh' | 'prefix_hit' | 'streaming'`. The
          // `'streaming'` value tells operators to read the SSE
          // `message_delta.usage.cache_read_input_tokens` for the
          // resolved cache-hit count (or `X-Cached-Tokens` trailer if
          // the client supports HTTP trailers).
          let sessionCacheStatus: 'fresh' | 'prefix_hit' | 'streaming' =
            body.stream === true ? 'streaming' : lookup.hit ? 'prefix_hit' : 'fresh';
          res.setHeader('X-Session-Cache', sessionCacheStatus);
          // HTTP/1.1 chunked-encoding trailer announcement for streaming.
          // The actual value is filled in by `handleStreamingNative`
          // once it has captured `terminalCachedTokens` from the final
          // SSE chunk.
          if (body.stream === true) {
            res.setHeader('Trailer', 'X-Cached-Tokens');
          }

          // Outer catch branches on `responseMode` (not `res.headersSent`, which
          // flips in `writeHead` before the body lands) so a crash after
          // `writeHead(application/json)` cannot leak SSE frames into a JSON body.
          const visibility = createVisibility();

          try {
            if (body.stream === true) {
              // On the paged path the underlying native cache is the
              // sole reuse mechanism, so preserve it even though the JS
              // `ChatSession` is freshly allocated. The native paged
              // adapter validates reuse by token/hash before any cached
              // prefix is trusted, and the MoE GDN checkpoint layer now
              // follows the same content-checked policy. Non-paged keeps
              // the original `!lookup.hit` semantics so only warm-slot
              // hits preserve native cache.
              const resetNativeCache = pagedActive ? false : !lookup.hit;
              const outcome = await runSessionStreaming(session, messages, config, streamSignal, resetNativeCache);
              const streamResult = await handleStreamingNative(
                res,
                outcome.stream,
                body,
                outcome.wasCommitted,
                httpReq,
                visibility,
                config.includeReasoning !== false,
                stopSequences,
                serverTiming,
              );
              // Warm-slot adopt/drop only applies to the non-paged
              // path. On the paged path the JS-side warm slot plays no
              // role (block reuse is content-addressed in native), so
              // we never touch it — the fresh `ChatSession` allocated
              // for this request is dropped on the floor and GC'd once
              // the handler scope exits.
              //
              // Non-paged dual-gate adopt: BOTH the producer-side commit
              // signal (`outcome.wasCommitted()`, which reads
              // `session.turns` bumped in `startFromHistoryStream`'s
              // `finally`) AND the handler-side success signal
              // (`streamResult.ok`, true only when we reached the clean
              // `message_stop` terminal) must be true to adopt. The
              // producer's `finally` runs on every break — including
              // client abort, mid-decode throw, and
              // `finishReason=error` — so `wasCommitted()` alone is NOT
              // sufficient: it can return `true` after the SSE side
              // emitted an `error` terminal (not re-thrown by
              // `handleStreamingNative`), leaving a session whose
              // observable wire state is failure but whose `turns`
              // counter advanced. Adopting in that window would seed the
              // warm slot with a session the next request can lease but
              // whose history does not match what the client received.
              //
              // Mirrors `responses.ts` (around line 3277) where the
              // analogous gate combines `committed`, `handlerError`, and
              // `streamFailureMode === null` — the producer-side commit
              // and a clean handler-side terminal must both hold before
              // the session is reachable from a subsequent request.
              if (!pagedActive) {
                if (streamResult.ok && outcome.wasCommitted() && !streamResult.suppressedToolCalls) {
                  sessionReg.adopt(MESSAGES_WARM_SLOT_ID, session, requestedSystem, null);
                } else {
                  sessionReg.drop(MESSAGES_WARM_SLOT_ID);
                }
              }
            } else {
              // See the streaming branch above for the rationale on
              // preserving native cache on the paged path.
              const resetNativeCache = pagedActive ? false : !lookup.hit;
              // Native `chatSessionStart` has no AbortSignal yet — disconnect handling
              // lives inside `handleNonStreaming` / `endJson`.
              const outcome = await runSessionNonStreaming(session, messages, config, resetNativeCache);
              const result = outcome.result;
              // Re-classify the `X-Session-Cache` header.
              //
              // Non-paged: a warm-slot hit that did NOT actually produce
              // native prefix reuse (`cachedTokens === 0` — e.g.
              // tokenizer change, system prompt drift squeaking past
              // the byte-equal compare via some upstream rewrite) gets
              // demoted from `prefix_hit` back to `fresh`.
              //
              // Paged: `lookup.hit` is always `false` so we entered
              // with `sessionCacheStatus = 'fresh'`. Promote to
              // `prefix_hit` when the native engine reports
              // `cachedTokens > 0` — that's the authoritative signal
              // that `BlockAllocator`'s content-addressed prefix lookup
              // recovered shared SYS blocks on this turn. `res.end` has
              // not fired yet (`handleNonStreaming` is what flushes via
              // `endJson`), so the overwrite still lands on the wire.
              if (lookup.hit && result.cachedTokens === 0) {
                sessionCacheStatus = 'fresh';
                res.setHeader('X-Session-Cache', sessionCacheStatus);
              } else if (pagedActive && result.cachedTokens > 0) {
                sessionCacheStatus = 'prefix_hit';
                res.setHeader('X-Session-Cache', sessionCacheStatus);
              }
              // Companion `X-Cached-Tokens` header: emitted only when
              // reuse genuinely happened, so operators can spot a stale
              // `prefix_hit` claim from telemetry alone.
              if (result.cachedTokens > 0) {
                res.setHeader('X-Cached-Tokens', String(result.cachedTokens));
              }
              await handleNonStreaming(res, result, body, visibility, stopSequences, serverTiming);
              // Non-paged success: adopt the warm slot only when the
              // dispatch actually committed. Mirrors the streaming-side
              // dual-gate at `streamResult.ok && outcome.wasCommitted()`
              // above and the sibling `/v1/responses` adopt gate, so the
              // local invariant — "never adopt an uncommitted session"
              // — is enforced by the same check on both wire formats and
              // both endpoints. Today every native failure throws (and
              // routes through the inner catch below), so the gate is
              // dead code on the current Rust paths; it defends the
              // invariant LOCALLY so a future native change that
              // resolves `chat_session_start_sync` with
              // `Ok(finish_reason="error")` cannot silently poison the
              // warm slot. Drop on the uncommitted branch matches the
              // streaming-side `else { drop(...) }` so the sentinel does
              // not accumulate stale entries from earlier turns.
              //
              // Paged success: never adopt — block-level reuse is
              // already in the native cache, and adopting would
              // re-introduce the cross-endpoint warm-slot eviction
              // that paged is supposed to eliminate.
              if (!pagedActive) {
                if (outcome.committed && !hasSuppressedToolCalls(result, body)) {
                  sessionReg.adopt(MESSAGES_WARM_SLOT_ID, session, requestedSystem, null);
                } else {
                  sessionReg.drop(MESSAGES_WARM_SLOT_ID);
                }
              }
            }
          } catch (err) {
            // A failed turn on the non-paged path must not leave a
            // poisoned warm slot for the next request to lease — drop
            // the sentinel before emitting the error response.
            // Streaming half-failures are already covered by the
            // `wasCommitted()` gate above; this catch handles
            // non-streaming throws and any pre-handler failures from
            // the streaming path. The paged path never adopts, so the
            // drop is a no-op there but kept unconditional for
            // simplicity (the registry treats `drop` of an absent key
            // as a no-op).
            sessionReg.drop(MESSAGES_WARM_SLOT_ID);
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
                writeFallbackErrorSSE(res, 'error', {
                  error: { type: 'api_error', message },
                });
              }
              try {
                endSSE(res);
              } catch {
                // Already closed.
              }
            }
          }
        });
      await runInference();
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
