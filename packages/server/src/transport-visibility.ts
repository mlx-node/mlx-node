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
 * This module centralises the fix: both endpoints track a
 * `responseMode` (which wire format the handler committed to) and
 * drive visibility flags from the ACTUAL write-completion signal.
 * The helpers below own the `res.end()` / terminal-SSE-flush calls so
 * the flag can only be flipped after the kernel has accepted the
 * bytes (and the optional write callback reports no error).
 *
 * Non-terminal SSE writes stay synchronous — only the terminal event
 * is flushed through the async helper. That keeps per-token overhead
 * unchanged while ensuring the visibility gate fires from an
 * authoritative signal.
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
   * fires with no error. Proves the JSON body was accepted by the
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
 * Write an HTTP 200 JSON response and await kernel acknowledgement
 * via Node's `res.end(body, callback)` contract. On the callback's
 * success path (err == null) the visibility flag is flipped; any
 * error in the callback, any synchronous throw from `writeHead` /
 * `end`, or a `res.once('error', …)` during the write rejects the
 * returned promise so the caller's catch can rethrow to the outer
 * error epilogue.
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
    const settle = (err: Error | null): void => {
      if (settled) return;
      settled = true;
      res.off('error', onError);
      if (err != null) {
        reject(err);
      } else {
        visibility.responseBodyWritten = true;
        resolve();
      }
    };
    const onError = (err: Error): void => {
      settle(err instanceof Error ? err : new Error(String(err)));
    };
    res.once('error', onError);
    try {
      // Node's `ServerResponse.end` accepts `(data, callback)` as a
      // 2-arg overload. The callback fires once the final chunk is
      // accepted by the kernel (or a socket error surfaces).
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
 * Non-terminal events still go through the synchronous `writeSSEEvent`
 * — only the terminal write is gated on the async callback so per-
 * token overhead stays unchanged.
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
    const settle = (err: Error | null): void => {
      if (settled) return;
      settled = true;
      res.off('error', onError);
      if (err != null) {
        reject(err);
      } else {
        visibility.terminalEmitted = true;
        resolve();
      }
    };
    const onError = (err: Error): void => {
      settle(err instanceof Error ? err : new Error(String(err)));
    };
    res.once('error', onError);
    try {
      const ok = res.write(chunk, (err?: Error | null) => {
        settle(err ?? null);
      });
      // `res.write` returning `false` is backpressure, NOT a failure
      // — the callback will still fire once the buffer drains, so we
      // let `settle` be the sole source of truth. We only care about
      // the synchronous-throw case, which the try/catch above
      // handles.
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
