/**
 * Transport visibility tracking shared between the OpenAI Responses
 * and Anthropic Messages endpoints.
 *
 * Iter-32 introduced explicit visibility flags (`responseBodyWritten`
 * / `terminalEmitted`) to gate "safe to suppress" on whether the
 * client actually observed a terminal artefact for this turn. Iter-33
 * adversarial review flagged three distinct ways that gate still
 * lied:
 *
 *   1. Flags were flipped from the synchronous return of `res.end()`
 *      / `writeSSEEvent`, which only proves Node buffered the bytes —
 *      not that the client received them. An async socket failure
 *      queued after the sync write would still leave the client with
 *      no terminal while the gate thought it was safe to suppress.
 *
 *   2. The outer catch keyed "JSON vs SSE fallback" on
 *      `res.headersSent`. A JSON request whose `res.end()` threw
 *      AFTER `writeHead(200, 'application/json')` had already
 *      flipped `headersSent` would then hit the SSE fallback branch
 *      and emit SSE-formatted frames into a JSON-declared response.
 *
 *   3. `/v1/messages` used the old `headersSent`-based logic even
 *      though it was stateless — the wire-contract corruption was
 *      identical.
 *
 * Iter-34 then proved that awaiting `res.end(payload, cb)` /
 * `res.write(chunk, cb)` is still not enough under real peer
 * disconnect: when `socket.destroyed === true` but
 * `res.destroyed === false`, `ServerResponse._writeRaw()` returns
 * without firing the end/write callback and without emitting
 * `'error'`. The helper's Promise never settled, pinning the per-
 * model `withExclusive` mutex on a dead client. The rewrite below
 * rejects on ANY of (a) the write callback reporting err != null,
 * (b) `'error'`, (c) `'close'` on the response or its socket, or
 * (d) a pre-write check finding the response/socket already
 * destroyed. Visibility flags are flipped ONLY when the callback
 * fires with err == null.
 *
 * Non-terminal SSE writes stay synchronous — only the terminal event
 * is flushed through the async helper. To still stop decode when the
 * peer disconnects mid-stream (so the per-model mutex is not pinned
 * by background work the client cannot see), each streaming handler
 * attaches a `res.once('close', …)` listener that flips the existing
 * `clientAborted` sticky flag. The decode loop already checks that
 * flag at every iteration, so the break happens at the next event
 * boundary.
 */

import type { ServerResponse } from 'node:http';

import { writeSSEEvent } from './streaming.js';

/**
 * Which wire format the handler committed to. `null` means headers
 * have not been sent yet and the outer catch can still emit a clean
 * 500 JSON error. `'json'` means `writeHead(200, 'application/json')`
 * already fired — the outer catch must NOT emit SSE frames into a
 * JSON-declared response. `'sse'` means `beginSSE()` already fired —
 * the outer catch can emit a best-effort streaming `error` event.
 */
export type ResponseMode = 'json' | 'sse' | null;

export interface TransportVisibility {
  /** Wire format committed by the handler, or `null` pre-headers. */
  responseMode: ResponseMode;
  /**
   * Non-streaming: set ONLY after `res.end(body)`'s write callback
   * fires with err == null. Proves the JSON body was accepted by the
   * kernel, not just queued into the ServerResponse buffer.
   */
  responseBodyWritten: boolean;
  /**
   * Streaming: set ONLY after the terminal SSE event
   * (`response.completed` / `response.failed` / `message_stop` / a
   * streaming `error` event) has been flushed and the write callback
   * reported no error.
   */
  terminalEmitted: boolean;
}

/** Allocate a fresh visibility record with no wire format committed. */
export function createVisibility(): TransportVisibility {
  return {
    responseMode: null,
    responseBodyWritten: false,
    terminalEmitted: false,
  };
}

/**
 * Type helper. `ServerResponse` exposes `destroyed` and `socket` at
 * runtime but the Node type definitions mark `socket` as nullable
 * with a narrower shape. We read both defensively inside the
 * helpers below — if either is already destroyed the client cannot
 * see anything we write, so the promise must reject immediately
 * rather than parking on a callback that will never fire.
 */
function isSocketGone(res: ServerResponse): boolean {
  if (res.destroyed) return true;
  const sock = res.socket;
  if (sock != null && sock.destroyed) return true;
  return false;
}

/**
 * Write an HTTP 200 JSON response and await kernel acknowledgement
 * via Node's `res.end(body, callback)` contract. On the callback's
 * success path (err == null) the visibility flag is flipped; any
 * error in the callback, any synchronous throw from `writeHead` /
 * `end`, a `res.once('error', …)` during the write, a `'close'`
 * event on the response (or its socket), or a pre-entry destroyed
 * check all reject the returned promise so the caller's catch can
 * rethrow to the outer error epilogue.
 *
 * `responseMode` is flipped to `'json'` synchronously before
 * `writeHead` fires. That commits the wire format — the outer catch
 * now knows this request is JSON and MUST NOT emit SSE frames on
 * failure, even though `headersSent` is true.
 */
export async function endJson(res: ServerResponse, body: string, visibility: TransportVisibility): Promise<void> {
  // Commit `responseMode` AFTER `writeHead` returns. If `writeHead`
  // throws synchronously the outer catch still sees
  // `responseMode === null` (headers never actually landed) and
  // routes through the clean 500 JSON error path instead of
  // destroying the socket. Once `writeHead` returns the wire format
  // is locked in — any later failure must treat the response as
  // committed to JSON.
  res.writeHead(200, { 'Content-Type': 'application/json' });
  visibility.responseMode = 'json';
  await new Promise<void>((resolve, reject) => {
    let settled = false;
    // Keep references so we can `removeListener` from whichever
    // listeners did not fire once the promise settles. Leaking
    // listeners on an ending response is almost always fine, but
    // the hygiene matters when tests reuse the same mock `res`
    // across multiple assertions.
    const onError = (err: Error): void => {
      settle(err instanceof Error ? err : new Error(String(err)));
    };
    const onClose = (): void => {
      // `'close'` without a preceding successful `end` callback
      // means the peer disconnected (or the socket was destroyed)
      // before the kernel reported acceptance. The client cannot
      // see the JSON body, so the adopt gate must refuse.
      settle(new Error('response closed before write completion'));
    };
    const sock = res.socket;
    const settle = (err: Error | null): void => {
      if (settled) return;
      settled = true;
      res.removeListener('error', onError);
      res.removeListener('close', onClose);
      if (sock != null) sock.removeListener('close', onClose);
      if (err != null) {
        reject(err);
      } else {
        visibility.responseBodyWritten = true;
        resolve();
      }
    };

    // Pre-check: if the response or its socket is already destroyed
    // the write callback is not guaranteed to fire and `'close'`
    // may have already been emitted before our listeners attached.
    // Reject synchronously so the caller's catch runs instead of
    // parking on a callback that will never arrive.
    if (isSocketGone(res)) {
      settle(new Error('response socket already destroyed before write'));
      return;
    }

    res.once('error', onError);
    res.once('close', onClose);
    if (sock != null) sock.once('close', onClose);

    try {
      // Node's `ServerResponse.end` accepts `(data, callback)` as a
      // 2-arg overload. The callback fires once the final chunk is
      // accepted by the kernel (or a socket error surfaces). A
      // destroyed peer can leave this callback un-invoked — the
      // `'close'` listener above is the backstop for that path.
      res.end(body, (err?: Error | null) => {
        settle(err ?? null);
      });
    } catch (err) {
      settle(err instanceof Error ? err : new Error(String(err)));
    }
  });
}

/**
 * Emit the terminal SSE event for a streaming response and await
 * kernel acknowledgement via `res.write(chunk, cb)`. Used for
 * `response.completed` / `response.failed` (Responses API),
 * `message_stop` / streaming `error` (Anthropic API), and any other
 * event where "the client observed the terminal" must be
 * authoritative before the outer gate consults `terminalEmitted`.
 *
 * Rejection semantics mirror `endJson`: callback err, `'error'`,
 * `'close'` on the response or its socket, or a pre-entry destroyed
 * check all reject. The visibility flag is flipped ONLY on the
 * success path (callback err == null).
 *
 * Non-terminal events still go through the synchronous `writeSSEEvent`
 * — only the terminal write is gated on the async callback so per-
 * token overhead stays unchanged. The streaming handlers attach a
 * `res.once('close', …)` listener to the decode loop so a dead peer
 * still stops decode at the next iteration boundary.
 */
export async function flushTerminalSSE(
  res: ServerResponse,
  eventType: string,
  data: object,
  visibility: TransportVisibility,
): Promise<void> {
  const payload = { type: eventType, ...data };
  const chunk = `event: ${eventType}\ndata: ${JSON.stringify(payload)}\n\n`;
  await new Promise<void>((resolve, reject) => {
    let settled = false;
    const onError = (err: Error): void => {
      settle(err instanceof Error ? err : new Error(String(err)));
    };
    const onClose = (): void => {
      settle(new Error('response closed before terminal SSE write completion'));
    };
    const sock = res.socket;
    const settle = (err: Error | null): void => {
      if (settled) return;
      settled = true;
      res.removeListener('error', onError);
      res.removeListener('close', onClose);
      if (sock != null) sock.removeListener('close', onClose);
      if (err != null) {
        reject(err);
      } else {
        visibility.terminalEmitted = true;
        resolve();
      }
    };

    if (isSocketGone(res)) {
      settle(new Error('response socket already destroyed before terminal SSE write'));
      return;
    }

    res.once('error', onError);
    res.once('close', onClose);
    if (sock != null) sock.once('close', onClose);

    try {
      // `res.write` returning `false` is normally backpressure and
      // the callback will still fire once the buffer drains — but
      // on a destroyed socket the callback is NOT guaranteed to
      // fire. The `'close'` listener above plus the pre-entry
      // `isSocketGone` check cover that path. The returned boolean
      // is otherwise irrelevant for correctness; we only care about
      // the synchronous-throw case, which the try/catch handles.
      const ok = res.write(chunk, (err?: Error | null) => {
        settle(err ?? null);
      });
      void ok;
    } catch (err) {
      settle(err instanceof Error ? err : new Error(String(err)));
    }
  });
}

/**
 * Mark the response as committed to SSE format. Call this from the
 * streaming handler immediately after `beginSSE(res)` returns so the
 * outer catch knows it is dealing with an SSE stream if
 * `handleStreamingNative` later throws before any terminal event
 * lands.
 */
export function markSSEMode(visibility: TransportVisibility): void {
  visibility.responseMode = 'sse';
}

/**
 * Emit a last-ditch SSE `error` event from the outer catch when the
 * streaming handler threw before flushing a terminal. Uses the
 * synchronous `writeSSEEvent` writer since we cannot afford another
 * await in the catch path — this is best-effort output on an already-
 * broken stream.
 */
export function writeFallbackErrorSSE(res: ServerResponse, eventType: string, data: object): void {
  try {
    writeSSEEvent(res, eventType, data);
  } catch {
    // Writing the fallback frame itself failed — socket is gone. Let
    // the caller fall through to `res.end()` / destroy so the request
    // lifecycle completes.
  }
}
