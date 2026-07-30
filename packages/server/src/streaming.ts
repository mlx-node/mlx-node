/** SSE writer utilities. */

import type { ServerResponse } from 'node:http';

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
export function writeSSEEvent(res: ServerResponse, eventType: string, data: object): void {
  const payload = { type: eventType, ...data };
  res.write(`event: ${eventType}\ndata: ${JSON.stringify(payload)}\n\n`);
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
