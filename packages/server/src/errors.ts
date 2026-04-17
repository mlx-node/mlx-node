/** OpenAI- and Anthropic-compatible JSON error responses. */

import type { ServerResponse } from 'node:http';

export interface APIError {
  type: string;
  message: string;
  code: string | null;
  param: string | null;
}

export function sendError(
  res: ServerResponse,
  status: number,
  type: string,
  message: string,
  param?: string | null,
): void {
  const body: { error: APIError } = {
    error: {
      type,
      message,
      code: null,
      param: param ?? null,
    },
  };
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
}

export function sendBadRequest(res: ServerResponse, message: string, param?: string): void {
  sendError(res, 400, 'invalid_request_error', message, param);
}

export function sendNotFound(res: ServerResponse, message: string): void {
  sendError(res, 404, 'not_found_error', message);
}

export function sendMethodNotAllowed(res: ServerResponse, allowed: string): void {
  res.writeHead(405, { Allow: allowed, 'Content-Type': 'application/json' });
  res.end(
    JSON.stringify({
      error: { type: 'invalid_request_error', message: 'Method not allowed', code: null, param: null },
    }),
  );
}

export function sendInternalError(res: ServerResponse, message: string): void {
  sendError(res, 500, 'server_error', message);
}

/**
 * 503 with `type: 'storage_timeout'`. Emitted by the responses endpoint when
 * an in-flight `store.store(...)` gating a `previous_response_id` continuation
 * fails to settle within `CHAIN_WRITE_WAIT_TIMEOUT_MS` and the final `getChain`
 * probe still misses. 503 (not 404) because the write may yet land, so the
 * same id can be retried — a 404 would wrongly mark it permanently invalid.
 */
export function sendStorageTimeout(res: ServerResponse, message: string): void {
  sendError(res, 503, 'storage_timeout', message);
}

/**
 * 429 with `type: 'rate_limit_error'` and `code: 'queue_full'`. Emitted by
 * `/v1/responses` when the per-model execution queue is already holding
 * `maxQueueDepth` waiters behind the current dispatch. Always sets
 * `Retry-After: 1` (string seconds) so clients back off briefly before
 * retrying — short enough to encourage a retry, long enough to avoid
 * busy-looping the server.
 */
export function sendRateLimit(res: ServerResponse, message: string): void {
  res.writeHead(429, { 'Retry-After': '1', 'Content-Type': 'application/json' });
  res.end(
    JSON.stringify({
      error: {
        type: 'rate_limit_error',
        message,
        code: 'queue_full',
        param: null,
      },
    }),
  );
}

export function sendAnthropicError(res: ServerResponse, status: number, type: string, message: string): void {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ type: 'error', error: { type, message } }));
}

export function sendAnthropicBadRequest(res: ServerResponse, message: string): void {
  sendAnthropicError(res, 400, 'invalid_request_error', message);
}

export function sendAnthropicNotFound(res: ServerResponse, message: string): void {
  sendAnthropicError(res, 404, 'not_found_error', message);
}

export function sendAnthropicInternalError(res: ServerResponse, message: string): void {
  sendAnthropicError(res, 500, 'api_error', message);
}

export function sendAnthropicMethodNotAllowed(res: ServerResponse, allowed: string): void {
  res.writeHead(405, { Allow: allowed, 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ type: 'error', error: { type: 'invalid_request_error', message: 'Method not allowed' } }));
}

/**
 * 429 Anthropic-shape rate-limit response. Mirror of {@link sendRateLimit}
 * for `/v1/messages`. Body uses the `{ type: 'error', error: { type, message } }`
 * envelope the rest of the Anthropic error helpers use; `Retry-After: 1`
 * is set verbatim so clients can wait one second before retrying.
 */
export function sendAnthropicRateLimit(res: ServerResponse, message: string): void {
  res.writeHead(429, { 'Retry-After': '1', 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ type: 'error', error: { type: 'rate_limit_error', message } }));
}
