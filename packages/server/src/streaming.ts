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
 *
 * Includes the `type` field in the data payload for OpenAI SDK compatibility.
 * The `type` is always the SSE event type, even if the data object contains
 * its own `type` field (which would be for a different purpose).
 */
export function writeSSEEvent(res: ServerResponse, eventType: string, data: Record<string, unknown>): void {
  // Spread data first, then override type to ensure the SSE event type wins
  const payload = { ...data, type: eventType };
  res.write(`event: ${eventType}\ndata: ${JSON.stringify(payload)}\n\n`);
}

/**
 * End the SSE stream.
 */
export function endSSE(res: ServerResponse): void {
  res.end();
}
