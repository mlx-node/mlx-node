/**
 * OpenAI-compatible error responses
 */

import type { ServerResponse } from 'node:http';

export interface APIError {
  type: string;
  message: string;
  code: string | null;
  param: string | null;
}

/**
 * Send an OpenAI-compatible JSON error response.
 */
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

/**
 * Send a 400 Bad Request error.
 */
export function sendBadRequest(res: ServerResponse, message: string, param?: string): void {
  sendError(res, 400, 'invalid_request_error', message, param);
}

/**
 * Send a 404 Not Found error.
 */
export function sendNotFound(res: ServerResponse, message: string): void {
  sendError(res, 404, 'not_found_error', message);
}

/**
 * Send a 405 Method Not Allowed error.
 */
export function sendMethodNotAllowed(res: ServerResponse, allowed: string): void {
  res.writeHead(405, { Allow: allowed, 'Content-Type': 'application/json' });
  res.end(
    JSON.stringify({
      error: { type: 'invalid_request_error', message: 'Method not allowed', code: null, param: null },
    }),
  );
}

/**
 * Send a 500 Internal Server Error.
 */
export function sendInternalError(res: ServerResponse, message: string): void {
  sendError(res, 500, 'server_error', message);
}

/**
 * Send a 503 Service Unavailable with a `storage_timeout` error type.
 *
 * Used by the responses endpoint when an in-flight `store.store(...)`
 * write that gates a `previous_response_id` continuation fails to
 * settle within `CHAIN_WRITE_WAIT_TIMEOUT_MS` AND a final `getChain`
 * probe still cannot find the chain. 503 is deliberately chosen over
 * 404 here because:
 *
 *   * The condition is transient — the write may yet land, so a later
 *     retry with the same `previous_response_id` could succeed.
 *   * 404 is non-retryable from the client's perspective and would
 *     cause clients to discard the response id as permanently invalid,
 *     silently breaking the conversation chain on storage backpressure.
 *
 * `type: 'storage_timeout'` is a fresh error type (no existing
 * `type: "..."` in the server emits it) so clients can disambiguate
 * it from the true 404 shape and classify it as a retryable storage
 * condition.
 */
export function sendStorageTimeout(res: ServerResponse, message: string): void {
  sendError(res, 503, 'storage_timeout', message);
}

/**
 * Send an Anthropic-compatible JSON error response.
 */
export function sendAnthropicError(res: ServerResponse, status: number, type: string, message: string): void {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ type: 'error', error: { type, message } }));
}

/**
 * Send an Anthropic 400 Bad Request error.
 */
export function sendAnthropicBadRequest(res: ServerResponse, message: string): void {
  sendAnthropicError(res, 400, 'invalid_request_error', message);
}

/**
 * Send an Anthropic 404 Not Found error.
 */
export function sendAnthropicNotFound(res: ServerResponse, message: string): void {
  sendAnthropicError(res, 404, 'not_found_error', message);
}

/**
 * Send an Anthropic 500 Internal Server Error.
 */
export function sendAnthropicInternalError(res: ServerResponse, message: string): void {
  sendAnthropicError(res, 500, 'api_error', message);
}

/**
 * Send an Anthropic 405 Method Not Allowed error.
 */
export function sendAnthropicMethodNotAllowed(res: ServerResponse, allowed: string): void {
  res.writeHead(405, { Allow: allowed, 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ type: 'error', error: { type: 'invalid_request_error', message: 'Method not allowed' } }));
}
