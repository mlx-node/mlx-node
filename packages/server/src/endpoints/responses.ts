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

import { sendBadRequest, sendInternalError, sendNotFound, sendStorageTimeout } from '../errors.js';
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
 * Upper bound on how long the native-miss recovery path will wait for
 * an in-flight `store.store(...)` write to land before giving up.
 * Iter-38 finding 1: the iter-37 recovery path called `awaitPending(id)`
 * with no timeout, so a wedged SQLite write (or any never-settling
 * write promise) would pin the continuation request forever — no
 * cancellation, no observability, a silent request hang.
 *
 * Iter-39 finding 1: on timeout we no longer fall straight through to
 * 404. We first run one last `getChain` probe — a write landing at
 * (timeout + epsilon) would have succeeded for the client but the
 * iter-38 path spuriously reported 404 and permanently poisoned the
 * client's chain state. The probe catches that race. Only when the
 * probe STILL misses do we surface the condition to the client, and
 * even then as HTTP 503 `storage_timeout` (a retryable transient
 * error) rather than 404 (permanent / non-retryable).
 *
 * 2000ms default is short enough that a stuck native backend fails
 * fast from the client's perspective and long enough that a healthy
 * SQLite write — which ordinarily completes in single-digit
 * milliseconds on warm disks — has ample headroom to resolve before
 * the timer fires. Operators running on slower storage (e.g. encrypted
 * volumes, heavy WAL checkpoint contention) can override via
 * `MLX_CHAIN_WRITE_WAIT_TIMEOUT_MS`; non-positive / non-finite /
 * empty values fall back to the default.
 */
function getChainWriteWaitTimeoutMs(): number {
  const raw = process.env.MLX_CHAIN_WRITE_WAIT_TIMEOUT_MS;
  if (raw == null || raw === '') return 2000;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return 2000;
  return parsed;
}

/**
 * Upper bound on how long the outer handler's `finally` block will
 * await the off-lock `store.store(...)` write before detaching.
 *
 * Iter-35 moved persistence OFF the per-model mutex but still `await`ed
 * the write in the outer `finally`, so any wedged `store.store(...)`
 * pinned the request's socket/abort listeners and its dispatch lease
 * until the promise settled. A never-settling write would leak
 * listeners, keep the binding's `inFlight` counter elevated, and
 * block teardown from finalising after a hot-swap.
 *
 * Iter-39 finding 2: decouples post-commit persist from the request
 * lifetime. Abort listeners are removed and `releaseDispatchLease` is
 * called IMMEDIATELY after the terminal bytes go out; the persist
 * await is wrapped in a `Promise.race` against this timeout and, on
 * timeout, the handler returns control to the caller while the
 * promise continues running in the background (the pending-writes
 * tracker still holds its reference so chained continuations can
 * still observe it).
 *
 * Default 5000ms is intentionally larger than `CHAIN_WRITE_WAIT_TIMEOUT_MS`
 * because this bound is not client-facing — the client has already
 * received its terminal response by this point. Override via
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
 * Second-stage ("hard") timeout for the off-lock post-commit persist.
 *
 * Iter-43 intentionally left the iter-40 `retainBinding` pinned past
 * the soft `MLX_POST_COMMIT_PERSIST_TIMEOUT_MS` so that a SLOW-BUT-
 * EVENTUAL `store.store(...)` — the common case under SQLite back-
 * pressure — could still land its row against the live
 * `modelInstanceId` even if the handler returned first. That fix
 * closed iter-42's chain-break regression, but codex's iter-43
 * adversarial review flagged the corollary HIGH-severity leak: a
 * TRULY wedged write (promise that never settles) pins the retain
 * for the lifetime of the process, so `unregister()` can only park
 * the binding in `pendingTeardown` and never reaches final
 * teardown — pinning the model object, its `SessionRegistry`, and
 * the native KV/cache state until the server restarts.
 *
 * The fix is a second-stage breaker: start a hard-timeout timer at
 * the same moment we start waiting on the persist, armed OFF the
 * handler's await path so the response is never delayed by it. If
 * the persist settles naturally, the `.finally(...)` cancels the
 * timer via `clearTimeout`. If the persist is still wedged past
 * this much longer bound, the timer fires and force-releases the
 * iter-40 retain via the existing idempotent `persistRetainBox`.
 * The bounded leak duration is capped at this value instead of
 * process lifetime.
 *
 * Iter-45 (codex's iter-44 HIGH finding): dropping the retain on
 * elapsed time alone is not enough. A SLOW-BUT-EVENTUAL persist
 * whose wall-clock exceeds the hard bound could still settle
 * naturally, and if an `unregister + register(same_model)` fires
 * between the force-release and that late settlement, the fresh
 * `register()` would mint a NEW instance id while the pending
 * write still carries the OLD id — silently breaking
 * `previous_response_id` continuations. The iter-45 fix pairs the
 * force-release with
 * `registry.retireInstanceIdForForceRelease(leaseModel)` (called
 * FIRST, while the binding is still alive). The retired id is
 * tombstoned keyed on the model object, and a subsequent
 * `register()` of the SAME model object inherits it from the
 * tombstone — so the late-landing persist remains chainable. A
 * true hot-swap (re-register with a DIFFERENT model object) still
 * mints a fresh id, and its stale stored record is correctly
 * rejected with 400 instance-mismatch — the right semantic
 * outcome for a genuinely different model.
 *
 * The default is intentionally an order of magnitude larger than
 * any realistic SQLite commit window and well past the soft
 * timeout: the slow-but-eventual case iter-42 broke has to be
 * well-separated from the pathologically-wedged case that triggers
 * this breaker. Setting the value to `'0'` disables the hard
 * timeout entirely and reverts to strict iter-43 pin-forever
 * semantics — useful for tests that want to assert the iter-40
 * invariant without the second-stage safety net.
 *
 * Override via `MLX_POST_COMMIT_PERSIST_HARD_TIMEOUT_MS`. An EMPTY
 * string is treated as unset (falls back to the 60000ms default).
 * Explicit `'0'` disables. Any non-numeric value also falls back
 * to the default with no error — this is deliberate so a config-
 * templating typo (e.g. `${SOME_UNSET_VAR}` rendering to empty or
 * garbage) cannot silently disable the safety breaker and
 * reintroduce iter-41's unreclaimable leak.
 *
 * Iter-46 (codex's iter-45 MEDIUM finding): empty string AND any
 * whitespace-only input (`' '`, `'\n'`, `'\t'`, `'   '`) are
 * treated as unset and fall back to the 60000ms default —
 * padded/templated env values (e.g. `"${UNSET_VAR} "` rendering
 * to a single space, or Windows line-ending artefacts
 * introducing a trailing `\r`) cannot silently disable the
 * breaker the way pre-iter-46 `Number(' ')` -> `0` did. Any
 * leading/trailing whitespace around an otherwise-valid numeric
 * value is trimmed before parsing (`'  100  '` -> 100).
 *
 * Iter-46/48 also scopes the hard-timeout tombstone's lifetime
 * to the pending persists that installed it: the tombstone
 * installed by a breaker is released by the same persist's
 * `.finally(...)` via `registry.releaseTombstone(model)` —
 * inheritance is scoped to the narrow window where a late-
 * landing write is still unresolved; after every outstanding
 * persist settles the tombstone entry drains and
 * re-registrations mint fresh ids. Iter-48 stores one
 * refcounted `{ instanceId, outstandingCount }` entry per
 * model (each retire increments, each release decrements) so
 * overlapping hard-timeouts on the same live instance id share
 * one slot — keeping memory bounded at O(1) per model even
 * under a truly wedged store that never settles, and keeping
 * the tombstone alive as long as ANY outstanding persist
 * still needs it.
 *
 * Exported for direct unit testing in `__test__/server/handler.test.ts`
 * (iter-45 env-parsing coverage). Not part of the public API —
 * consumers should drive behavior via the env var, not this
 * function.
 */
export function getPostCommitPersistHardTimeoutMs(): number {
  const raw = process.env.MLX_POST_COMMIT_PERSIST_HARD_TIMEOUT_MS;
  const normalized = raw?.trim();
  if (normalized == null || normalized === '') return 60_000;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 60_000;
}

/**
 * TTL (in ms) for iter-50 hard-timed-out markers in the per-store
 * `PendingResponseWrites` tracker. See `pending-writes.ts` for the
 * full lifetime model.
 *
 * ## Why this exists (iter-51 codex HIGH finding 1)
 *
 * Iter-50 cleared hard-timed-out markers ONLY through the underlying
 * `store.store(...)` promise's `.finally(...)` handler installed in
 * `track()`. For a truly wedged SQLite writer or stuck native
 * backend, that promise never settles — `.finally(...)` never runs —
 * so the marker set accumulated one entry per hard-timed-out request
 * forever. Under sustained traffic against such a wedged backend the
 * iter-49 memory bound (pending tracker drained to zero) was
 * preserved, but the marker map grew linearly with traffic and every
 * one of those ids kept returning retryable 503 for a chain that the
 * server had long since given up on. Both an unbounded memory leak
 * AND an incorrect eventual classification.
 *
 * The fix is an independent TTL with lazy expiry on read. Marker
 * entries carry an `expiresAt` timestamp, `isHardTimedOut(id)`
 * treats any expired entry as absent and deletes it, and steady-state
 * marker memory is bounded at O(requestRate × TTL) regardless of
 * whether the underlying writes ever settle. A 5-minute default is
 * an order of magnitude larger than the hard-timeout breaker (60s
 * default) so a write that settles just after the breaker fires can
 * still deliver the correct retryable-503 signal to a concurrent
 * continuation; past 5 minutes the best-effort persist contract
 * (iter-35) has long since failed and a permanent 404 is the right
 * eventual outcome.
 *
 * ## Env override
 *
 * `MLX_HARD_TIMEOUT_MARKER_TTL_MS`. Semantics mirror
 * `MLX_POST_COMMIT_PERSIST_HARD_TIMEOUT_MS` exactly:
 *
 *   * Empty string or whitespace-only -> default (300_000ms).
 *   * Non-numeric garbage -> default.
 *   * Explicit `'0'` -> markers expire immediately on the next read
 *     (effectively disables the iter-50 retryable-503 classification
 *     and reverts to "hard-timed-out ids 404 immediately"). Useful
 *     for tests that need to assert the 404 branch without racing
 *     a long-lived marker.
 *   * Any finite >= 0 numeric value -> parsed.
 *
 * Exported for direct unit testing in
 * `__test__/server/handler.test.ts`. Not part of the public API —
 * consumers should drive behavior via the env var, not this
 * function.
 */
export function getHardTimedOutMarkerTtlMs(): number {
  const raw = process.env.MLX_HARD_TIMEOUT_MARKER_TTL_MS;
  const normalized = raw?.trim();
  if (normalized == null || normalized === '') return 300_000;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 300_000;
}

// ---------------------------------------------------------------------------
// Non-streaming path
// ---------------------------------------------------------------------------

/**
 * Outcome of the non-streaming handler. `response` is the committed
 * response object (non-null when `committed` — always, since the
 * non-streaming path only runs after `runSessionNonStreaming` resolves)
 * that the outer handler will persist AFTER releasing the per-model
 * mutex. Persistence is deliberately kept off the critical path so a
 * slow store does not pin the mutex on the next waiter.
 */
interface NonStreamingHandlerOutcome {
  /** Response object to persist once the outer `withExclusive` block exits. */
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

  // Iter-35 finding 2 (part a): the non-streaming native path has
  // no AbortSignal surface on `chatSession*`, so a client that
  // disconnects mid-decode still burns every token under the per-
  // model mutex — we only learn about the peer loss once native
  // decode resolves. TODO(iter35): add a native cancellation
  // surface for `chatSession*` so the dispatch can bail at the
  // next safepoint rather than running to completion.
  //
  // Disconnect-aware handling is delegated to `endJson`'s own
  // pre-entry `isSocketGone(res)` check (iter-34): on a dead peer
  // it synchronously rejects after `writeHead` has already flipped
  // `responseMode = 'json'`, which is exactly the shape the outer
  // catch expects — the JSON-mode branch destroys the socket, the
  // adopt gate refuses to cache the session under an unreachable
  // responseId (because the handler threw before
  // `responseBodyWritten` flipped), and the persist-initiator
  // path inside `withExclusive` never runs (`handlerError` set,
  // no `initiatePersist` call) so no store record gets written
  // for a turn the client never observed.
  //
  // `endJson` commits `responseMode = 'json'` synchronously so the
  // outer catch knows to emit a clean JSON error (or destroy the
  // socket) rather than an SSE frame if anything below throws. The
  // `responseBodyWritten` flag is flipped only from inside `res.end`'s
  // write callback — proving the kernel accepted the final chunk,
  // not just that ServerResponse buffered it. An async socket
  // failure surfaced through the callback rejects this promise so
  // the caller's catch can refuse to adopt the committed session
  // under an unreachable responseId.
  await endJson(res, JSON.stringify(response), visibility);
  return { response };
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
 * In every non-committed case we MUST skip `initiatePersist()` and
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
/**
 * Outcome of the streaming handler. `terminalToPersist` is non-null
 * only on the committed-success path — the outer handler writes it to
 * the `ResponseStore` AFTER releasing the per-model mutex so a slow
 * store write does not pin the lock on the next waiter. Every failure
 * path (mid-decode throw, finishReason=error, iterator exhaustion,
 * client disconnect) leaves `terminalToPersist` null — the turn never
 * committed in the session, so there is nothing authoritative to
 * persist and nothing for a later `previous_response_id` continuation
 * to cold-replay.
 */
interface StreamingHandlerOutcome {
  /**
   * Terminal response object captured on the committed-success
   * branch. The outer handler persists this once `withExclusive`
   * exits. `null` on every failure path — the commit gate already
   * decided not to advertise this turn to future continuations.
   */
  terminalToPersist: ResponseObject | null;
  /**
   * Which failure epilogue the handler took, or `null` on the
   * committed-success path. Iter-36 finding 2: the outer adopt gate
   * used to key purely on `committed && (handlerError == null ||
   * safeToSuppress)`, but `committed` only reports whether the
   * underlying session's turn counter advanced — it does NOT report
   * whether the response is safe to advertise as the new chain
   * head. A `res.close` that fires AFTER the final chunk has been
   * emitted (client dropped the last byte but the producer already
   * committed) takes the `client_abort` failure epilogue and
   * flushes `response.failed` successfully, which flips
   * `safeToSuppress = true` via `visibility.terminalEmitted` — and
   * the previous gate would then call `sessionReg.adopt(responseId,
   * …)` on a responseId the client will never chain off of,
   * evicting the last good hot session for the model in the
   * process. The outer gate now checks this signal explicitly and
   * refuses to adopt when the handler took the `client_abort`
   * branch even if the session committed.
   *
   * Legal values:
   *  - `null`                   success path (`response.completed`)
   *  - `'client_abort'`         HTTP peer dropped mid or post stream
   *  - `'error'`                underlying generator threw
   *  - `'finish_reason_error'`  terminal chunk carried `finishReason: 'error'`
   *  - `'stream_exhausted'`     iterator ended without a done event
   */
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
  // Commit to SSE wire format synchronously. The outer catch branches
  // on `responseMode` — not on `headersSent` — so an early throw from
  // `writeSSEEvent` below (e.g. socket died between `beginSSE` and
  // the first event) routes to the streaming `error` epilogue
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
  // the HTTP request OR the response socket emits `close`/`error`
  // while we're mid-iteration. Either one diverts the post-loop
  // block to the failure epilogue.
  //
  // Iter-34: also listen on `res` and `res.socket`. Non-terminal
  // SSE writes are fire-and-forget through `writeSSEEvent` — on a
  // destroyed socket they can silently "succeed" while decode keeps
  // burning work under the per-model mutex. Attaching the listener
  // here lets the next loop iteration observe the disconnect and
  // break out, so native decode still runs to completion (no
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
    res.off('close', onResClose);
    res.off('error', onResError);
    if (resSocketForAbort != null) {
      resSocketForAbort.off('close', onResClose);
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

    // Iter-35 finding 2: persistence moved OUT of the per-model
    // mutex. The terminal `response.completed` event still flushes
    // inside this critical section (the client expects it ordered
    // against the prior SSE deltas), but the `ResponseStore` write
    // that lets a future `previous_response_id` continuation cold-
    // replay this turn is deferred to the outer `handleCreateResponse`
    // body — AFTER `withExclusive` releases. Persistence is
    // best-effort (store failures are log-only) and does not touch
    // native model state, so holding the mutex across a slow SQLite
    // write only pins the next waiter for no reason.
    //
    // Gate `terminalEmitted` on the terminal SSE event's write
    // callback firing without error — `flushTerminalSSE` only flips
    // the flag once the kernel has accepted the frame. A synchronous
    // `res.write` return does not prove the client saw it (backpressure
    // can defer flushing, and a socket error surfaces via the callback
    // AFTER the write returned). An early throw or a callback-reported
    // error here rejects the promise so the outer catch refuses to
    // adopt under an unseen responseId.
    await flushTerminalSSE(res, 'response.completed', { response: terminal }, visibility);
    endSSE(res);
    return { terminalToPersist: terminal, failureMode: null };
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
  const reason: 'error' | 'client_abort' | 'finish_reason_error' | 'stream_exhausted' = thrownError
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
  // `flushTerminalSSE` only flips `terminalEmitted` after
  // `response.failed` is acknowledged by the kernel. If the write
  // callback reports a socket error the promise rejects here, the
  // flag stays false, and the outer catch refuses to adopt under an
  // unseen responseId.
  await flushTerminalSSE(res, 'response.failed', { response: failedTerminal }, visibility);
  endSSE(res);
  // Uncommitted terminal — the registry-level adopt gate already
  // skips caching this session, and the store must not be written
  // either. A later `previous_response_id` continuation that landed
  // on this record would cold-replay a turn the session rolled back,
  // silently resurrecting failed output as authoritative history.
  //
  // Iter-36 finding 2: `failureMode` carries the reason out to the
  // outer adopt gate, which uses `client_abort` specifically to
  // veto adoption even when the session's internal `turns`
  // counter advanced (a final-chunk commit followed by a post-
  // terminal `res.close`). Carrying every reason — not just the
  // abort case — keeps the signal complete for future gating that
  // might want to distinguish e.g. a stream-exhausted turn from a
  // finish_reason-error turn at the adopt site.
  return { terminalToPersist: null, failureMode: reason };
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

  if (newInputMessages.length === 1) {
    const last = newInputMessages[0]!;
    const initialTurns = session.turns;
    if (last.role === 'user') {
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
      return {
        stream: session.sendToolResultStream(last.toolCallId, last.content, { config, signal }),
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
    stream: session.startFromHistoryStream(config, signal),
    wasCommitted: () => session.turns > initialTurns,
  };
}

// ---------------------------------------------------------------------------
// Storage helper
// ---------------------------------------------------------------------------

/**
 * Build the `StoredResponseRecord` for a committed response.
 *
 * Pure function — constructs the record payload from the inputs and
 * does not touch the store. Split out from the initiate-persist site
 * so `handleCreateResponse` can (a) construct the record SYNCHRONOUSLY
 * inside `withExclusive`, (b) kick off the SQLite write through
 * `initiatePersist` on the in-lock side so the in-flight write is
 * registered in the per-store pending tracker BEFORE the mutex
 * releases, and (c) await the write off-lock just to surface errors
 * to the log. This closes the iter-35 race window where a
 * back-to-back continuation could fire `store.getChain(id)` before
 * the off-lock `await store.store(id)` had landed in SQLite —
 * returning 404 on a response id the client had just received in
 * `response.completed`. See `initiatePersist` and
 * `packages/server/src/pending-writes.ts` for the full rationale.
 *
 * Store only the NEW input messages from this request, not the full
 * expanded conversation. Chain reconstruction re-derives the full
 * history by following previous_response_id links.
 *
 * `modelInstanceId` is the monotonic id `ModelRegistry` assigned to
 * the model object that serviced this request. It is stashed inside
 * the `configJson` blob so the Rust-side schema stays untouched; on
 * a later `previous_response_id` continuation the responses endpoint
 * reads it back out of the trailing chain record and compares it
 * against the live id for `body.model`. See the endpoint's
 * `readStoredModelIdentity` helper and the guard block in
 * `handleCreateResponse` — records without this field are rejected
 * outright per iter-23 finding 1 (the iter-22 friendly-name
 * compat fallback silently reopened same-name hot-swap corruption
 * and has been removed).
 */
function buildResponseRecord(
  response: ResponseObject,
  newInputMessages: ChatMessage[],
  previousResponseId: string | undefined,
  modelInstanceId: number | undefined,
): StoredResponseRecord {
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
    }),
    expiresAt: Math.floor(Date.now() / 1000) + RESPONSE_TTL_SECONDS,
  };
}

/**
 * Initiate an off-lock `store.store(record)` write.
 *
 * SYNCHRONOUSLY kicks off the native `store.store(record)` promise,
 * registers it in the per-store pending-write tracker under the
 * record's id, and returns the promise to the caller. The caller
 * MUST await the returned promise off-lock — the caller is
 * responsible for catching errors and logging them.
 *
 * The crucial property for iter-36 finding 1 is that this function
 * is called SYNCHRONOUSLY from inside `withExclusive` so that the
 * tracker registration happens before the per-model mutex releases.
 * Any back-to-back continuation that slips in immediately after the
 * mutex release will observe the in-flight write through
 * `getPendingWritesFor(store).awaitPending(previous_response_id)`
 * and can await it before retrying `store.getChain(...)`, closing
 * the 404-before-store-landed race window.
 *
 * Even though `store.store(...)` returns an already-pending promise,
 * we never await it here — the whole point of this function is to
 * keep the SQLite flush off the critical path.
 */
function initiatePersist(store: ResponseStore, record: StoredResponseRecord): Promise<void> {
  const writePromise = store.store(record);
  getPendingWritesFor(store).track(record.id, writePromise);
  return writePromise;
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
  // deferred by a concurrent `unregister()`) completes once the last
  // dispatch lease AND the last persist retention releases.
  const lease = registry.acquireDispatchLease(body.model);
  if (!lease) {
    sendInternalError(res, 'session registry missing for registered model');
    return;
  }
  const leaseModel = lease.model;
  // Iter-35 finding 1: AbortController wired to the HTTP request's
  // disconnect events, declared at this scope so the outer `finally`
  // can always detach the listeners even on an early `return` from
  // inside the try. Listeners are attached only after we pass the
  // pre-lock validation gates — the `let` holders default to `null`
  // so the detach loop safely no-ops on the early-return path.
  const abortController = new AbortController();
  const abortSocket = res.socket;
  const onAbortClose = (): void => {
    abortController.abort();
  };
  const onAbortError = (_err: unknown): void => {
    abortController.abort();
  };
  let abortListenersAttached = false;
  // Iter-39 finding 2: the outer `try/finally` below runs
  // `runPostDispatchCleanup` eagerly after `withExclusive` returns
  // (so the post-commit persist wait does not pin abort listeners or
  // the dispatch lease) and ALSO idempotently from the finally on the
  // early-return / pre-dispatch-error path. These guards keep the
  // cleanup a no-op when it has already run on the eager path.
  let cleanupPerformed = false;
  let leaseReleased = false;
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
        // Iter-36 finding 1 / iter-37 finding 1: iter-35 moved the
        // `store.store(...)` write OUT of `withExclusive` so a slow
        // SQLite flush would not pin the per-model mutex on the next
        // waiter. That opened a 404 window: a client that received
        // `response.completed` with `responseId = A` and
        // immediately fires a follow-up request carrying
        // `previous_response_id: A` can reach this `getChain`
        // BEFORE the off-lock write for A has landed in SQLite —
        // the chain is missing and we would spuriously 404 on a
        // response id the client was just handed.
        //
        // The production `ResponseStore` is the native mlx-db
        // implementation; its `get_chain` THROWS `"Response not
        // found: <id>"` on a miss rather than returning `[]` (see
        // `crates/mlx-db/src/response_store/reader.rs`). The
        // in-memory mock used by tests returns `[]`. Handle BOTH
        // shapes: a thrown "not found" AND an empty array both
        // drop through the pending-writes retry path.
        //
        // `initiatePersist` registers every in-flight write in a
        // per-store pending-write tracker synchronously, BEFORE
        // the producing request's mutex releases. If the tracker
        // reports a pending write for the requested id, await it
        // and retry `getChain`. The retry is guaranteed to see
        // the row because the tracked promise only resolves after
        // the store's own serialization queue has accepted the
        // insert. Swallow any rejection from the tracked promise —
        // the producer's own awaiter already surfaces write
        // failures to the log, and a failed write correctly
        // leaves the store empty so the second `getChain` still
        // throws / returns `[]` and we fall through to the
        // original 404.
        let chain: StoredResponseRecord[];
        let firstAttemptError: unknown = null;
        try {
          chain = await store.getChain(body.previous_response_id);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          // Case-insensitive substring match is intentionally
          // lenient: we only want to route "the row is not
          // present" throws into the retry path. A genuine
          // infrastructure error (connection refused, SQL parse
          // error, etc.) should still bubble out to the outer
          // catch as an internal error.
          if (!/not found/i.test(msg)) {
            throw err;
          }
          firstAttemptError = err;
          chain = [];
        }

        if (chain.length === 0) {
          const pending = getPendingWritesFor(store).awaitPending(body.previous_response_id);
          if (pending !== undefined) {
            // Iter-38 finding 1: bound the wait. `awaitPending`
            // returns the raw `store.store(...)` promise, and if
            // the native backend is wedged (SQLite lock held by
            // a stuck writer, FFI hang, etc.) that promise may
            // never settle. Without a ceiling the continuation
            // request pins forever. Race it against a short
            // timer; on timeout, log a warning and fall through
            // to the 404 path — a clean bounded error is always
            // better than a silent hang.
            //
            // TIMED_OUT is a unique sentinel so callers can
            // distinguish "pending wait timed out" from "pending
            // wait rejected" without relying on exception text.
            // A rejection from the pending promise propagates
            // naturally through the `await` and the subsequent
            // catch below (the tracker's `.finally(...)` has
            // already cleared the entry, so the retry `getChain`
            // will see the store's true post-failure state and
            // 404 cleanly).
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
              // Write failure is the producer's problem; proceed
              // with the retry so the 404 epilogue matches the
              // true store state.
            } finally {
              if (timeoutHandle !== undefined) {
                clearTimeout(timeoutHandle);
              }
            }
            if (timedOut) {
              // Iter-39 finding 1: before declaring the write
              // stuck, run ONE last `getChain` probe. A write
              // landing at (CHAIN_WRITE_WAIT_TIMEOUT_MS + epsilon)
              // would have succeeded for the client but the
              // iter-38 code flipped to 404 the moment the timer
              // fired — permanently poisoning the client's chain
              // (404 is non-retryable, so the client discards
              // `previous_response_id`). The probe closes that
              // race: if the write slipped in during the thin
              // window between timer-fire and this check, we use
              // its result and continue normally. Only when the
              // probe STILL misses is the write genuinely wedged,
              // and we surface it as retryable 503 storage_timeout
              // so the client can try again with the same
              // `previous_response_id` instead of treating it as a
              // permanent miss.
              let probed: StoredResponseRecord[] | null = null;
              try {
                probed = await store.getChain(body.previous_response_id);
              } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                if (!/not found/i.test(msg)) {
                  throw err;
                }
                // Probe confirmed the row is still missing —
                // genuine storage timeout. `probed` stays null and
                // we emit 503 below.
                probed = null;
              }
              if (probed !== null && probed.length > 0) {
                // Iter-39 finding 1: the write slipped in between
                // timer-fire and the probe. Log this so operators
                // can still see the wedged-writer condition that
                // triggered the slow path, even when the client
                // got a coherent 200 via the probe. Without this
                // log the only observable signal for "the bounded
                // wait fired" is the `sendStorageTimeout` 503
                // branch below, which fires on a legitimate miss;
                // the successful-probe path would otherwise be
                // silent.
                console.warn(
                  `[responses] pending store write for previous_response_id "${body.previous_response_id}" did ` +
                    `not settle within ${chainWriteWaitTimeoutMs}ms, but a last-probe getChain found the record. ` +
                    `Continuing with the probed chain — likely a slow SQLite writer that landed just after the ` +
                    `timeout fired.`,
                );
                chain = probed;
              } else {
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
                // Retry also missed the row — this is a genuine
                // 404. Fall through to the empty-chain handler
                // below.
                chain = [];
              }
            }
          } else if (firstAttemptError !== null) {
            // Iter-50 / iter-51: before rethrowing the original
            // "not found" error (which the outer catch at the end
            // of this block turns into a permanent 404), check
            // whether the id has a hard-timed-out marker from the
            // post-commit persist breaker. If the hard-timeout
            // breaker fired against an in-flight
            // `store.store(...)` for this response id, the raw
            // write promise is still running in the background and
            // may yet land — classifying that as a permanent 404
            // would cause clients to discard `previous_response_id`
            // as invalid and silently break the conversation chain.
            //
            // Iter-51 (codex's iter-50 HIGH finding 2): before
            // emitting the retryable 503, run ONE last `getChain`
            // probe against the store. Mirrors the shape of the
            // iter-39 last-probe in the `awaitPending` timeout
            // branch above (lines ~2112-2142): a write landing at
            // the thin window between "marker set" and "classify
            // as 503" should be returned as the successful chain,
            // NOT misclassified as retryable 503 for a chain that
            // already exists. Only when the probe STILL misses is
            // the breaker-classified id genuinely unresolved and we
            // surface it as retryable 503 `storage_timeout`.
            if (getPendingWritesFor(store).isHardTimedOut(body.previous_response_id)) {
              let lastChance: StoredResponseRecord[] | null = null;
              try {
                lastChance = await store.getChain(body.previous_response_id);
              } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                if (!/not found/i.test(msg)) {
                  throw err;
                }
                // Probe confirmed the row is still missing —
                // genuine hard-timed-out persist. `lastChance`
                // stays null and we emit 503 below.
                lastChance = null;
              }
              if (lastChance !== null && lastChance.length > 0) {
                // Iter-51: the write slipped in between marker-set
                // and this probe. Log this so operators can still
                // see the wedged-writer condition that triggered
                // the slow path, even when the client got a
                // coherent 200 via the probe. Mirror the analogous
                // log in the `awaitPending` timeout branch above.
                console.warn(
                  `[responses] previous_response_id "${body.previous_response_id}" missing on first lookup and ` +
                    `its post-commit persist crossed the hard-timeout breaker, but a last-probe getChain found ` +
                    `the record. Continuing with the probed chain — likely a wedged SQLite writer that landed ` +
                    `just after the marker was set.`,
                );
                chain = lastChance;
                // Fall through into the happy-path continuation
                // code below (chain.length > 0 skips the empty-
                // chain branch and the outer hot-swap guard takes
                // over).
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
              // First call threw "not found", no pending write is
              // tracked for this id, and no hard-timed-out marker
              // is live. Rethrow the original error so the outer
              // catch turns it into the proper 404 response.
              throw firstAttemptError;
            }
          }
          if (chain.length === 0) {
            // Iter-50 / iter-51: mirror the rethrow branch above.
            // An empty chain coming back from a mock-compatible
            // store that never threw still needs to route through
            // the retryable-503 path when the id has a
            // hard-timed-out marker — otherwise slow-but-eventual
            // persists across the breaker misclassify as permanent
            // 404 here too.
            //
            // Iter-51: probe getChain one last time before emitting
            // 503. A write landing during the narrow window between
            // marker-set and this branch should resolve to the real
            // chain via the probe, NOT be misclassified as retryable
            // 503 for a chain that already exists.
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
                // Fall through: `chain.length > 0` so the empty-
                // chain branch below is skipped and the outer hot-
                // swap guard / normal continuation flow takes over.
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

    // Iter-35 finding 1: arm the AbortController wired at function
    // scope above. The streaming wrappers in `@mlx-node/lm` plumb
    // this signal into `_runChatStream`, which calls
    // `handle.cancel()` on the native ChatStreamHandle AND pushes a
    // synthetic abort marker into the queue so the
    // `await waitForItem()` blocking on the NEXT native chunk
    // unblocks immediately. Without the signal a client that dropped
    // mid-eval would still keep this handler (and the per-model
    // mutex) pinned until the next token arrived from native decode
    // — on a long eval that can be hundreds of milliseconds, during
    // which no other request on the same model makes any forward
    // progress. Listeners are attached HERE (not at function
    // entry) so early-return validation gates above do not need to
    // pair an install with a detach — the outer `finally` detaches
    // unconditionally, gated on `abortListenersAttached`.
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

    // Iter-36 finding 1: persistence is a two-step dance.
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
    // through to the 404 epilogue. That closes the iter-35 race
    // where a fresh response id on the wire could transiently
    // 404 under `getChain`.
    //
    // `pendingPersistOuter` is the in-flight promise captured
    // inside the lock; the out-of-lock awaiter just catches errors
    // and logs them. `persistMode` is populated alongside so the
    // log line keeps the streaming / non-streaming discrimination
    // the iter-35 code had.
    let pendingPersistOuter: Promise<void> | null = null;
    let persistMode: 'streaming' | 'non-streaming' | null = null;
    // Iter-43: structural scaffolding for the binding retain paired
    // with the in-flight persist. The persist's `.finally(...)`
    // still calls this closure on settlement to balance the
    // iter-40 `retainBinding` — the closure's idempotency flag
    // matters only to that one call site today.
    //
    // The box shape is retained from iter-42 deliberately even
    // though the post-commit timeout arm no longer invokes it:
    //   - Iter-42 introduced a force-release in the timeout arm
    //     to prevent a wedged `store.store(...)` from pinning
    //     `pendingPersists` forever. That force-release was
    //     reverted in iter-43 (see the timeout arm below) because
    //     it reopened iter-40: a slow-but-eventual persist could
    //     still land after timeout, and releasing the retain
    //     before the write actually settled let an intervening
    //     same-object `unregister()` + `register()` finalise the
    //     old binding and mint a fresh instance id, so the late
    //     write recorded a stale id and broke the next
    //     `previous_response_id` continuation.
    //   - The scaffolding stays so a future iteration can
    //     reintroduce a surgical "split teardown" (e.g. release
    //     heavy resources on timeout while keeping identity
    //     pinned until settlement) without rewiring the retain
    //     wrappers in both dispatch branches.
    //
    // Held in a box because TypeScript's control-flow analysis
    // otherwise narrows the in-closure assignment to `never`
    // across the intervening `await` / try-catch boundaries.
    const persistRetainBox: { release: (() => void) | null } = { release: null };
    // `failureMode` carries the streaming failure-epilogue reason
    // from `handleStreamingNative` out to the outer adopt gate.
    // Iter-36 finding 2: a final-chunk commit followed by a
    // post-terminal `res.close` takes the `client_abort` branch
    // and flushes `response.failed` successfully, which would
    // otherwise flip `safeToSuppress = true` and let the adopt
    // gate cache a session under a response id the client will
    // never chain off of. The gate now refuses to adopt when
    // `failureMode === 'client_abort'` regardless of how
    // `committed` / `safeToSuppress` landed.
    let streamFailureMode: StreamingHandlerOutcome['failureMode'] = null;

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
        // Iter-32 finding 1 & 2 / iter-33 adversarial review:
        //
        //   * The "safe to suppress" gate used to key on
        //     `res.headersSent`, which is a LIE for "the client
        //     received the response". Node's `writeHead` flips
        //     `headersSent = true` synchronously before any body
        //     bytes leave the buffer, and the sync return of
        //     `res.end()` / `writeSSEEvent` only proves the bytes
        //     were queued — an async socket failure after the queue
        //     could still leave the client with no terminal.
        //   * The outer catch used to pick "JSON vs SSE fallback"
        //     from `res.headersSent`, so a `writeHead(200,
        //     'application/json')` → `res.end()` crash would emit
        //     SSE frames into a JSON-declared response.
        //
        // The fix threads a `TransportVisibility` record that
        // tracks both the wire format the handler committed to
        // (`responseMode`) AND whether the client observed a
        // terminal artefact (`responseBodyWritten` /
        // `terminalEmitted`). Both flags are flipped only from the
        // kernel-ack callback of the underlying `res.end` /
        // `res.write` — synchronous return is NOT treated as proof
        // of visibility. The outer catch branches on
        // `responseMode` to choose the clean-up shape (JSON error,
        // SSE `error` frame, or socket destroy).
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
              // Iter-36 finding 1: initiate the write SYNCHRONOUSLY
              // inside the mutex so the pending-write tracker
              // observes it before the mutex releases. The promise
              // is awaited off-lock in the outer finally block.
              const record = buildResponseRecord(
                handlerOutcome.terminalToPersist,
                newInputMessages,
                previousResponseId,
                currentInstanceId,
              );
              // Iter-40 finding 1: pair a `retainBinding` against
              // the persist promise so the binding's
              // `modelInstanceId` survives a concurrent same-model
              // unregister + re-register that races the
              // post-commit write. `releaseBinding` runs in the
              // persist's `.finally(...)` regardless of outcome,
              // so the retention counter stays balanced whether
              // the write fulfils or rejects.
              //
              // Iter-43: the idempotent wrapper is kept from
              // iter-42 as structural scaffolding, but the
              // post-commit-persist SOFT timeout arm no longer
              // force-fires it — see the timeout handler below
              // for the rationale (short version: force-releasing
              // on the soft timeout reopens iter-40 for the slow-
              // but-eventual case, which is strictly more common
              // than the truly-wedged case iter-42 was trying to
              // bound).
              //
              // Iter-44 second-stage breaker: codex's iter-43
              // review called out that leaving the retain pinned
              // forever on a wedged write makes the binding
              // unreclaimable until process restart. We arm an
              // INDEPENDENT hard-timeout timer alongside the
              // persist (see `getPostCommitPersistHardTimeoutMs`
              // for the rationale on the default). If the persist
              // settles naturally the timer is cancelled via
              // `clearTimeout` inside the same `.finally(...)` —
              // so slow-but-eventual writes are unaffected. If
              // the persist is still wedged past the hard bound,
              // the timer fires and force-releases the iter-40
              // retain via the existing idempotent
              // `persistRetainBox`. The hard timer is ALSO armed
              // off the handler's await path, so the response is
              // never delayed by it.
              //
              // Iter-45: before the hard timeout force-releases
              // the retain (which unblocks binding teardown), we
              // also call
              // `registry.retireInstanceIdForForceRelease(leaseModel)`
              // to tombstone the binding's current instance id on
              // the model object. A subsequent `register()` of
              // the SAME model object inherits that retired id
              // rather than minting fresh — so the late-landing
              // persist's record (stamped with the retired id)
              // still matches the live binding and stays
              // chainable through `previous_response_id`. Only a
              // true hot-swap (re-register with a DIFFERENT model
              // object) mints a fresh id, and the 400 instance-
              // mismatch that results is the correct semantic
              // outcome because the new model is semantically
              // different from the one that produced the stored
              // record. Retirement MUST happen BEFORE release so
              // `instanceIds.get(model)` still returns the live
              // id the record carries.
              //
              // Iter-46/48 (codex's iter-45/46/47 findings):
              // the tombstone's lifetime is scoped to the
              // pending persists that installed it — the
              // `.finally(...)` calls
              // `registry.releaseTombstone(leaseModel)` so that
              // when the late write eventually settles
              // (fulfills or rejects), the shared refcount
              // drops and, once every outstanding persist has
              // released, any subsequent re-registration
              // correctly mints a fresh id. Without this
              // scoping, a past hard-timeout event would
              // permanently re-enable id inheritance across
              // unrelated later lifecycles — reopening stale-
              // chain replay across what should be logically
              // dead bindings. Iter-48's refcounted single-
              // entry layout handles OVERLAPPING hard-timeouts
              // on the same live instance id in bounded space:
              // every breaker targets the SAME retired id (the
              // register-inherit path keeps using it while the
              // tombstone is alive) so one shared refcount
              // safely collapses every in-flight retire, and
              // memory stays O(1) per model even under a truly
              // wedged store that never settles.
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
              const streamingHardTimeoutHandle: ReturnType<typeof setTimeout> | null =
                streamingHardTimeoutMs > 0
                  ? setTimeout(() => {
                      if (persistRetainReleased) return;
                      console.error(
                        `[responses] post-commit persist HARD timeout (${streamingHardTimeoutMs}ms, ` +
                          `${streamingPersistMode}): underlying store.store(...) has not settled; assuming ` +
                          `wedged backend, force-releasing the iter-40 retain so the binding can be torn ` +
                          `down. Retiring the current instance id via tombstone so a same-object ` +
                          `re-registration inherits it and a late-landing persist remains chainable; a ` +
                          `hot-swap to a DIFFERENT model object will mint a fresh id and the stale chain ` +
                          `will correctly fail with 400 instance-mismatch.`,
                      );
                      // Iter-49/50: move the pending-write tracker
                      // entry into the hard-timed-out marker state
                      // for this response id. The pending entry is
                      // dropped so a wedged store.store(...) does
                      // not pin one promise closure + tracker
                      // entry per hard-timed-out request (iter-49
                      // memory bound), AND the id is added to the
                      // `hardTimedOut` marker so a concurrent
                      // `previous_response_id` continuation can
                      // tell the difference between a permanent
                      // 404 and a slow-but-eventual persist that
                      // crossed the hard timeout. The continuation
                      // path consults `isHardTimedOut(id)` before
                      // falling through to `sendNotFound(...)` and
                      // returns retryable 503 `storage_timeout`
                      // instead, so clients keep retrying rather
                      // than discarding the chain (iter-50 fix).
                      // The marker has two cleanup paths (iter-51):
                      // (1) fast — the underlying store promise's
                      // `.finally(...)` inside `track()` fires when
                      // the wedged store unwedges; (2) slow — an
                      // independent TTL (`MLX_HARD_TIMEOUT_MARKER_TTL_MS`,
                      // default 300s) bounds memory at O(requestRate ×
                      // TTL) even against a truly wedged store that
                      // NEVER settles. Marker lifetime =
                      // min(settlement, TTL expiry).
                      getPendingWritesFor(store).markHardTimedOut(record.id, getHardTimedOutMarkerTtlMs());
                      // Iter-45: retire the id FIRST (binding is
                      // still alive here — retirement reads the
                      // live id) then drop the retain, which may
                      // trigger the deferred teardown. Iter-46:
                      // capture the retired id so the persist's
                      // `.finally(...)` can release the
                      // tombstone once the late write eventually
                      // settles. Iter-48: the registry stores
                      // one refcounted tombstone per model
                      // regardless of how many hard-timeouts
                      // overlap — each retire increments the
                      // shared counter and each release decrements
                      // it — so we just capture the returned
                      // `{ instanceId }` as a presence flag and
                      // call `releaseTombstone(leaseModel)` in
                      // the persist's `.finally(...)`.
                      retiredTombstone = registry.retireInstanceIdForForceRelease(leaseModel);
                      persistRetainBox.release?.();
                    }, streamingHardTimeoutMs)
                  : null;
              pendingPersistOuter = initiatePersist(store, record).finally(() => {
                if (streamingHardTimeoutHandle !== null) {
                  clearTimeout(streamingHardTimeoutHandle);
                }
                // Iter-46/48: if the hard-timeout breaker fired
                // and installed a tombstone on `leaseModel`,
                // decrement the shared refcount now that this
                // persist has settled. Iter-48's single-entry
                // refcount layout means overlapping breakers
                // share one slot — releasing one balances one
                // retire, and the entry survives until the last
                // outstanding persist releases.
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
          // Iter-35 finding 2 (part a): the non-streaming native
          // path has NO AbortSignal surface (plain
          // `chatSession*` returns a Promise, no cancel), so a
          // client that disconnects mid-generation still burns
          // the full decode budget under this mutex. TODO(iter35):
          // native cancellation for `chatSession*` — until then
          // the best we can do is the disconnect-aware skip inside
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
              // Iter-36 finding 1: same in-lock-initiate /
              // off-lock-await split as the streaming branch. The
              // non-streaming handler only returns when the JSON
              // body's `res.end()` callback has fired, so reaching
              // this point means the client observed the turn —
              // the pending-write tracker protects a back-to-back
              // continuation from a transient 404.
              const record = buildResponseRecord(
                handlerOutcome.response,
                newInputMessages,
                previousResponseId,
                currentInstanceId,
              );
              // Iter-40 finding 1: see the streaming branch for
              // the retain/release rationale — a same-model
              // unregister + re-register during the slow persist
              // must not mint a fresh `modelInstanceId` that
              // invalidates the row this write is about to land.
              //
              // Iter-43: see the streaming branch — the
              // idempotent-release scaffolding is retained from
              // iter-42 as a structural hook for a future split
              // teardown, but the post-commit SOFT timeout arm no
              // longer force-fires it.
              //
              // Iter-44 second-stage breaker: see the streaming
              // branch for the full rationale — a wedged persist
              // would otherwise leak the iter-40 retain for the
              // lifetime of the process. The hard-timeout timer
              // is armed here in the same shape, cancelled from
              // the persist's own `.finally(...)` when the write
              // settles naturally, and fires a force-release
              // through the idempotent `persistRetainBox`
              // otherwise. Default 60s, override via
              // `MLX_POST_COMMIT_PERSIST_HARD_TIMEOUT_MS`, `'0'`
              // disables. Empty string is treated as unset (falls
              // back to the 60000ms default) so a config-
              // templating typo cannot silently disable the
              // breaker.
              //
              // Iter-45: the force-release path also calls
              // `registry.retireInstanceIdForForceRelease(leaseModel)`
              // BEFORE releasing the retain so a same-object
              // re-registration AFTER teardown inherits the
              // retired instance id from the tombstone — a late-
              // landing persist against the retired id stays
              // chainable. A hot-swap to a DIFFERENT model object
              // mints fresh id and the 400 instance-mismatch is
              // correct. See the streaming branch for the full
              // rationale.
              //
              // Iter-46/48 (codex's iter-45/46/47 findings):
              // the tombstone's lifetime is scoped to the
              // pending persists that installed it — the
              // `.finally(...)` calls
              // `registry.releaseTombstone(leaseModel)` so that
              // when the late write eventually settles
              // (fulfills or rejects), the shared refcount
              // drops and, once every outstanding persist has
              // released, any subsequent re-registration
              // correctly mints a fresh id. Without this
              // scoping, a past hard-timeout event would
              // permanently re-enable id inheritance across
              // unrelated later lifecycles — reopening stale-
              // chain replay across what should be logically
              // dead bindings. Iter-48's refcounted single-
              // entry layout handles OVERLAPPING hard-timeouts
              // on the same live instance id in bounded space:
              // every breaker targets the SAME retired id (the
              // register-inherit path keeps using it while the
              // tombstone is alive) so one shared refcount
              // safely collapses every in-flight retire, and
              // memory stays O(1) per model even under a truly
              // wedged store that never settles.
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
              const nonStreamingHardTimeoutHandle: ReturnType<typeof setTimeout> | null =
                nonStreamingHardTimeoutMs > 0
                  ? setTimeout(() => {
                      if (persistRetainReleased) return;
                      console.error(
                        `[responses] post-commit persist HARD timeout (${nonStreamingHardTimeoutMs}ms, ` +
                          `${nonStreamingPersistMode}): underlying store.store(...) has not settled; ` +
                          `assuming wedged backend, force-releasing the iter-40 retain so the binding can ` +
                          `be torn down. Retiring the current instance id via tombstone so a same-object ` +
                          `re-registration inherits it and a late-landing persist remains chainable; a ` +
                          `hot-swap to a DIFFERENT model object will mint a fresh id and the stale chain ` +
                          `will correctly fail with 400 instance-mismatch.`,
                      );
                      // Iter-49/50: move the pending-write tracker
                      // entry into the hard-timed-out marker state
                      // for this response id. The pending entry is
                      // dropped so a wedged store.store(...) does
                      // not pin one promise closure + tracker
                      // entry per hard-timed-out request (iter-49
                      // memory bound), AND the id is added to the
                      // `hardTimedOut` marker so a concurrent
                      // `previous_response_id` continuation can
                      // tell the difference between a permanent
                      // 404 and a slow-but-eventual persist that
                      // crossed the hard timeout. The continuation
                      // path consults `isHardTimedOut(id)` before
                      // falling through to `sendNotFound(...)` and
                      // returns retryable 503 `storage_timeout`
                      // instead, so clients keep retrying rather
                      // than discarding the chain (iter-50 fix).
                      // The marker has two cleanup paths (iter-51):
                      // (1) fast — the underlying store promise's
                      // `.finally(...)` inside `track()` fires when
                      // the wedged store unwedges; (2) slow — an
                      // independent TTL (`MLX_HARD_TIMEOUT_MARKER_TTL_MS`,
                      // default 300s) bounds memory at O(requestRate ×
                      // TTL) even against a truly wedged store that
                      // NEVER settles. Marker lifetime =
                      // min(settlement, TTL expiry).
                      getPendingWritesFor(store).markHardTimedOut(record.id, getHardTimedOutMarkerTtlMs());
                      // Iter-45: retire the id FIRST (binding is
                      // still alive here — retirement reads the
                      // live id) then drop the retain, which may
                      // trigger the deferred teardown. Iter-46:
                      // capture the retired id so the persist's
                      // `.finally(...)` can release the
                      // tombstone once the late write eventually
                      // settles. Iter-48: the registry stores
                      // one refcounted tombstone per model
                      // regardless of how many hard-timeouts
                      // overlap — each retire increments the
                      // shared counter and each release decrements
                      // it — so we just capture the returned
                      // `{ instanceId }` as a presence flag and
                      // call `releaseTombstone(leaseModel)` in
                      // the persist's `.finally(...)`.
                      retiredTombstone = registry.retireInstanceIdForForceRelease(leaseModel);
                      persistRetainBox.release?.();
                    }, nonStreamingHardTimeoutMs)
                  : null;
              pendingPersistOuter = initiatePersist(store, record).finally(() => {
                if (nonStreamingHardTimeoutHandle !== null) {
                  clearTimeout(nonStreamingHardTimeoutHandle);
                }
                // Iter-46/48: if the hard-timeout breaker fired
                // and installed a tombstone on `leaseModel`,
                // decrement the shared refcount now that this
                // persist has settled. Iter-48's single-entry
                // refcount layout means overlapping breakers
                // share one slot — releasing one balances one
                // retire, and the entry survives until the last
                // outstanding persist releases.
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
        // Iter-36 finding 2 / iter-37 finding 2: refuse to adopt
        // whenever the streaming handler took ANY failure
        // epilogue, not just `client_abort`. The streaming
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
        // Iter-33 finding 2: branch on `responseMode` (the wire
        // format the handler committed to), NOT `res.headersSent`
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

    // Iter-39 finding 2: RELEASE the dispatch lease and DETACH the
    // abort listeners IMMEDIATELY now that `withExclusive` returned
    // and the terminal bytes have either been flushed or the outer
    // catch has emitted its error frame. The post-commit persist
    // wait that follows must NOT pin the request's lifecycle — a
    // wedged `store.store(...)` would otherwise leak socket/abort
    // listeners, keep the binding's `inFlight` counter elevated,
    // and block teardown after a hot-swap for the lifetime of the
    // wedged write.
    //
    // Iter-40 finding 1: the binding's `modelInstanceId` still needs
    // to survive until the post-commit write has actually landed —
    // otherwise a same-model unregister + re-register sequence
    // during a slow persist would mint a fresh id, and the row
    // (when it finally lands) would reference a dead id that the
    // very next `previous_response_id` continuation would reject.
    // That lifetime is covered by the ORTHOGONAL `retainBinding` /
    // `releaseBinding` retention counter paired around
    // `initiatePersist` below, so the eager dispatch-lease release
    // here stays lossless.
    //
    // The outer `finally` below re-runs both cleanups idempotently
    // so an early-return validation failure (before the
    // `withExclusive` site) still cleans up; `cleanupPerformed` is
    // the guard.
    cleanupPerformed = runPostDispatchCleanup();

    // Iter-35 finding 2 (part b) / iter-36 finding 1: the persist
    // write was INITIATED synchronously inside `withExclusive` via
    // `initiatePersist` — which registers the in-flight promise
    // in the per-store pending-write tracker BEFORE the mutex
    // releases. The SQLite flush is already on its way; a
    // back-to-back continuation observing the tracker will block
    // on the same promise instead of spuriously returning 404
    // under `getChain` (see the `getChain`-empty retry at the top
    // of this handler).
    //
    // Iter-39 finding 2: BOUND the wait on the persist promise
    // with `POST_COMMIT_PERSIST_TIMEOUT_MS`. A wedged native
    // backend can return a promise that never settles, and with
    // the iter-36 code an unconditional `await` would pin this
    // handler forever — leaking abort listeners and the dispatch
    // lease (now fixed above by running cleanup before this
    // wait). On timeout we leave the promise running in the
    // background: the pending-writes tracker still holds its
    // reference so chained continuations can still observe it,
    // and its `.finally(...)` handler will clear the tracker
    // entry whenever the write eventually settles (or stays
    // wedged until the process exits).
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
              `observe the in-flight write, and the iter-40 binding retain stays live until the write truly ` +
              `settles so the binding's modelInstanceId cannot be recycled under the late write. This ` +
              `condition usually signals a wedged SQLite writer or stuck native backend.`,
          );
          // Iter-43: do NOT force-release the iter-40
          // `retainBinding` here. Iter-42 added a
          // `persistRetainBox.release?.()` on this branch to
          // bound the worst case of a truly never-settling
          // `store.store(...)` so a later `unregister()` +
          // `register()` could reclaim the binding. That fix was
          // wrong: `Promise.race` treats any write that EXCEEDS
          // the timeout as "safe to unpin", but most timeouts in
          // practice are slow-but-eventual writes — the promise
          // still fulfils later, and the iter-40 invariant has
          // to hold for the entire interval until it does. If a
          // same-object unregister + re-register happens in the
          // window between timeout and actual settlement, force-
          // releasing the retain lets `pendingPersists` drop to
          // 0, the binding fully tears down, the re-register
          // mints a fresh `modelInstanceId`, and the late write
          // lands with the stale id that `buildResponseRecord`
          // stamped into `configJson` — exactly the iter-40
          // chain-break the retain was introduced to prevent.
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
          // drops and teardown proceeds normally.
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
    // Iter-35 finding 1: drop the AbortController's socket/request
    // listeners so they do not keep the request object alive past
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
    // retention has also released (see iter-40 below).
    //
    // Iter-39 finding 2: this now runs BEFORE the post-commit
    // persist wait, not after, so a wedged `store.store(...)` no
    // longer pins the lease. Teardown of a same-model unregister is
    // still deferred by the iter-40 `retainBinding` counter so the
    // binding's `modelInstanceId` survives until the pending write
    // has stamped its row durably — see `initiatePersist`.
    if (!leaseReleased) {
      leaseReleased = true;
      registry.releaseDispatchLease(leaseModel);
    }
    return true;
  }
}
