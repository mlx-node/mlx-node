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
