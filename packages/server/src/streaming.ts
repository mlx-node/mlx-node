/** SSE writer utilities. */

import type { ServerResponse } from 'node:http';

export function beginSSE(res: ServerResponse): void {
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
  res.end();
}
