/** SSE writer utilities. */

import type { IncomingMessage, ServerResponse } from 'node:http';

/**
 * Responses that have committed to SSE (`beginSSE`) but have not yet been
 * ended (`endSSE`) or had their connection torn down.
 *
 * Purely an accounting aid for {@link activeSSEStreamCount}, which graceful
 * shutdown reads to report how many streams a forced close cut short. Nothing
 * in the request path branches on membership, so a miscount can only skew a
 * diagnostic number — never the behaviour of a live stream.
 *
 * Module-scoped because `@mlx-node/server` is loaded once per process and
 * `beginSSE`/`endSSE` are free functions called from both endpoints.
 */
const activeSSEResponses = new Set<ServerResponse>();

/** Disconnect state whose listeners stay armed for one complete SSE handler. */
export interface SSEClientAbortTracker {
  readonly aborted: boolean;
  dispose(): void;
}

/**
 * Track request/response/socket disconnects until the caller's outermost
 * `finally`. Keeping this lifetime outside the decode loop matters: a final
 * native item can expand into backpressured residual protocol frames after the
 * iterator has already closed, and a disconnect during that drain must still
 * prevent a success terminal and session adoption.
 */
export function trackSSEClientAbort(res: ServerResponse, httpReq: IncomingMessage | undefined): SSEClientAbortTracker {
  let aborted = false;
  let disposed = false;
  const onClose = (): void => {
    aborted = true;
  };
  const onError = (_err: unknown): void => {
    aborted = true;
  };
  const socket = res.socket;

  if (httpReq != null) {
    httpReq.once('close', onClose);
    httpReq.once('error', onError);
  }
  res.once('close', onClose);
  res.once('error', onError);
  if (socket != null) socket.once('close', onClose);

  return {
    get aborted(): boolean {
      return aborted;
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      if (httpReq != null) {
        httpReq.removeListener('close', onClose);
        httpReq.removeListener('error', onError);
      }
      res.removeListener('close', onClose);
      res.removeListener('error', onError);
      if (socket != null) socket.removeListener('close', onClose);
    },
  };
}

export function beginSSE(res: ServerResponse): void {
  activeSSEResponses.add(res);
  // Belt-and-braces cleanup: a stream torn down by a client disconnect (or by
  // `server.closeAllConnections()`) may unwind through an error path that
  // never reaches `endSSE`. Without this the entry would leak for the life of
  // the process and inflate the count forever.
  res.once('close', () => {
    activeSSEResponses.delete(res);
  });
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });
}

/** Write one SSE event. Injects `type: eventType` into the payload (data's own `type` wins) for OpenAI SDK compatibility. */
export function writeSSEEvent(res: ServerResponse, eventType: string, data: object): boolean {
  const payload = { type: eventType, ...data };
  return res.write(`event: ${eventType}\ndata: ${JSON.stringify(payload)}\n\n`);
}

/**
 * Wait until a backpressured response can accept more data, or until its
 * transport closes. Close and error resolve rather than reject: the endpoint's
 * outer abort tracker owns the sticky state, and the next loop check exits
 * before another native item is written.
 *
 * Call this synchronously after `writeSSEEvent` returns false. In particular,
 * do not defer listener installation until the next iterator turn: `drain`
 * could fire while that turn is being fetched and leave the handler parked on
 * an event that already happened.
 */
export function awaitDrainOrClose(res: ServerResponse): Promise<void> {
  return new Promise<void>((resolve) => {
    let settled = false;
    const socket = res.socket;
    const settle = (): void => {
      if (settled) return;
      settled = true;
      res.removeListener('drain', onDrain);
      res.removeListener('close', onClose);
      res.removeListener('error', onError);
      if (socket != null) socket.removeListener('close', onClose);
      resolve();
    };
    const onDrain = (): void => {
      settle();
    };
    const onClose = (): void => {
      settle();
    };
    const onError = (_err: unknown): void => {
      settle();
    };

    // A destroyed peer cannot emit a future useful drain. Do not include
    // `writableEnded` here: write-after-end returns false and reports
    // ERR_STREAM_WRITE_AFTER_END asynchronously through the error listener.
    if (res.destroyed || (socket != null && socket.destroyed)) {
      settle();
      return;
    }

    res.once('drain', onDrain);
    res.once('close', onClose);
    res.once('error', onError);
    if (socket != null) socket.once('close', onClose);
  });
}

export function endSSE(res: ServerResponse): void {
  activeSSEResponses.delete(res);
  res.end();
}

/**
 * Number of SSE streams currently open process-wide. Diagnostics and
 * shutdown accounting only.
 */
export function activeSSEStreamCount(): number {
  return activeSSEResponses.size;
}

/**
 * Number of active SSE streams among a caller-owned collection of responses.
 *
 * `createServer()` uses this to intersect the process-wide SSE registry with
 * the responses accepted by one `node:http` Server. Keep the no-argument
 * {@link activeSSEStreamCount} above for process-wide diagnostics and
 * standalone `createHandler()` consumers.
 */
export function activeSSEStreamCountForResponses(responses: WeakSet<ServerResponse>): number {
  let count = 0;
  for (const response of activeSSEResponses) {
    if (responses.has(response)) count += 1;
  }
  return count;
}
