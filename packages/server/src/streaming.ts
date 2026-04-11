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
 * If the data object already contains a `type` field, it takes precedence.
 * Otherwise, `type` defaults to the SSE event type name.
 */
export function writeSSEEvent(res: ServerResponse, eventType: string, data: object): void {
  // Default type to eventType, but let data's own type field take precedence
  const payload = { type: eventType, ...data };
  res.write(`event: ${eventType}\ndata: ${JSON.stringify(payload)}\n\n`);
}

/**
 * End the SSE stream.
 */
export function endSSE(res: ServerResponse): void {
  res.end();
}
