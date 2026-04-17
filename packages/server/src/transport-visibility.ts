/**
 * Transport visibility tracking shared between the Responses and Messages
 * endpoints. Gates "safe to suppress" on whether the client actually observed
 * a terminal artefact for this turn.
 *
 * The helpers reject — and leave visibility flags unflipped — on ANY of:
 *   - the write callback reporting err != null;
 *   - a `'error'` event on `res`;
 *   - a `'close'` event on `res` or its socket;
 *   - a pre-write check finding the response/socket already destroyed.
 *
 * Flipping the flags from `res.end()` / `writeSSEEvent`'s synchronous return
 * is not sufficient: on a dead socket `_writeRaw` can return without ever
 * firing the callback or `'error'`, which would pin the per-model mutex on a
 * client that cannot see anything we write.
 *
 * Non-terminal SSE writes remain synchronous — only the terminal event is
 * flushed through the async helper. Streaming handlers independently attach a
 * `res.once('close', …)` listener to flip `clientAborted` so the decode loop
 * breaks at the next iteration boundary.
 */

import type { ServerResponse } from 'node:http';

import { writeSSEEvent } from './streaming.js';

/**
 * Wire format committed by the handler. `null` = pre-headers (outer catch
 * can still emit a clean 500 JSON). `'json'` = `writeHead(200, 'application/json')`
 * already fired — the outer catch MUST NOT emit SSE frames. `'sse'` = `beginSSE()`
 * fired — the outer catch may emit a best-effort streaming `error` event.
 */
export type ResponseMode = 'json' | 'sse' | null;

export interface TransportVisibility {
  responseMode: ResponseMode;
  /** Set only after `res.end(body)`'s callback fires with err == null — proves kernel acceptance, not buffer queue. */
  responseBodyWritten: boolean;
  /** Set only after the terminal SSE event's write callback reports no error. */
  terminalEmitted: boolean;
}

export function createVisibility(): TransportVisibility {
  return {
    responseMode: null,
    responseBodyWritten: false,
    terminalEmitted: false,
  };
}

/** True if `res` or its socket is already destroyed — writes will never be seen and callbacks may never fire. */
function isSocketGone(res: ServerResponse): boolean {
  if (res.destroyed) return true;
  const sock = res.socket;
  if (sock != null && sock.destroyed) return true;
  return false;
}

/**
 * Write an HTTP 200 JSON response and await kernel ack via `res.end(body, cb)`.
 * `responseMode` is committed AFTER `writeHead` returns so a synchronous throw
 * from `writeHead` leaves `responseMode === null` for the outer catch.
 * `responseBodyWritten` is flipped only on the callback's success path.
 */
export async function endJson(res: ServerResponse, body: string, visibility: TransportVisibility): Promise<void> {
  res.writeHead(200, { 'Content-Type': 'application/json' });
  visibility.responseMode = 'json';
  await new Promise<void>((resolve, reject) => {
    let settled = false;
    const onError = (err: Error): void => {
      settle(err instanceof Error ? err : new Error(String(err)));
    };
    const onClose = (): void => {
      // `'close'` without a successful end callback = client never saw the body.
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

    // Pre-check: `'close'` may have already fired and the write callback
    // may never arrive on a destroyed peer — reject synchronously instead.
    if (isSocketGone(res)) {
      settle(new Error('response socket already destroyed before write'));
      return;
    }

    res.once('error', onError);
    res.once('close', onClose);
    if (sock != null) sock.once('close', onClose);

    try {
      res.end(body, (err?: Error | null) => {
        settle(err ?? null);
      });
    } catch (err) {
      settle(err instanceof Error ? err : new Error(String(err)));
    }
  });
}

/**
 * Emit the terminal SSE event for a streaming response and await kernel ack via
 * `res.write(chunk, cb)`. `terminalEmitted` is flipped only on the success path.
 * Used for `response.completed` / `response.failed`, `message_stop`, and the
 * streaming `error` event. Non-terminal writes stay synchronous.
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
      // Ignore backpressure return value — the callback + `'close'` listener
      // + `isSocketGone` pre-check cover every settle path we need.
      const ok = res.write(chunk, (err?: Error | null) => {
        settle(err ?? null);
      });
      void ok;
    } catch (err) {
      settle(err instanceof Error ? err : new Error(String(err)));
    }
  });
}

/** Commit to SSE mode — call immediately after `beginSSE(res)` so the outer catch routes SSE-shaped failures correctly. */
export function markSSEMode(visibility: TransportVisibility): void {
  visibility.responseMode = 'sse';
}

/** Best-effort synchronous SSE `error` event for the outer catch when the handler threw before flushing a terminal. */
export function writeFallbackErrorSSE(res: ServerResponse, eventType: string, data: object): void {
  try {
    writeSSEEvent(res, eventType, data);
  } catch {
    // Socket is gone — let the caller complete the lifecycle via `end` / destroy.
  }
}
