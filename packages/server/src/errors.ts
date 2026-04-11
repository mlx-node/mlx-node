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
