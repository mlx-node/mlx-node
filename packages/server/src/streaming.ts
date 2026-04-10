/**
 * SSE (Server-Sent Events) writer utilities.
 */

import type { ServerResponse } from 'node:http';

/**
 * Set headers for an SSE response.
 */
export function beginSSE(res: ServerResponse): void {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });
}

/**
 * Write a single SSE event.
 */
export function writeSSEEvent(res: ServerResponse, eventType: string, data: unknown): void {
  res.write(`event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`);
}

/**
 * End the SSE stream.
 */
export function endSSE(res: ServerResponse): void {
  res.end();
}
