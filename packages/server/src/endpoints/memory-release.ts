/**
 * `POST /v1/memory/release` (Splash `_post_memory_release`): free the chat
 * weights now. The request takes no fields; the answer is 200 with the
 * weights' snapshot once they are released, 409 while a request is in
 * service or a load runs, 503 while the server shuts down.
 */

import type { IncomingMessage, ServerResponse } from 'node:http';

import { sendError } from '../errors.js';
import { stringifyStrictJson } from '../strict-json.js';
import type { WeightsSnapshot } from '../weight-residency.js';

/** The server's answer to a release request. */
export type MemoryReleaseAnswer = { kind: 'closing' } | { kind: 'answered'; weights: WeightsSnapshot };

/** The largest body read before the request is refused (the router's limit). */
const MAX_BODY_BYTES = 10 * 1024 * 1024;

class BodyError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly code = 'invalid_request_error',
  ) {
    super(message);
  }
}

/** Every value of a header, as Splash `headers.get_all` reads them. */
function headerValues(req: IncomingMessage, name: string): string[] {
  const distinct = (req as { headersDistinct?: Record<string, string[] | undefined> }).headersDistinct?.[name];
  if (distinct !== undefined) return distinct;
  const raw = req.headers[name];
  if (raw === undefined) return [];
  return Array.isArray(raw) ? raw : [raw];
}

/** Splash `_content_length`: a decimal byte count, else `undefined`. */
function contentLength(value: string): number | undefined {
  if (value.length === 0) return undefined;
  for (let index = 0; index < value.length; index++) {
    const code = value.charCodeAt(index);
    if (code < 48 || code > 57) return undefined;
  }
  const length = Number(value);
  return Number.isSafeInteger(length) ? length : undefined;
}

/** Python `Message.get_content_type()`: `type/subtype` in lower case, `text/plain` when malformed. */
function contentType(value: string | undefined): string {
  if (value === undefined) return 'text/plain';
  const semicolon = value.indexOf(';');
  const type = (semicolon === -1 ? value : value.slice(0, semicolon)).trim().toLowerCase();
  const slash = type.indexOf('/');
  if (slash <= 0 || slash !== type.lastIndexOf('/') || slash === type.length - 1) return 'text/plain';
  return type;
}

/** Whether the request carries a body to read (Splash `_unread_body`). */
function hasBody(req: IncomingMessage): boolean {
  if (headerValues(req, 'transfer-encoding').length > 0) return true;
  const lengths = headerValues(req, 'content-length');
  if (lengths.length === 0) return false;
  if (lengths.length > 1) return true;
  const length = contentLength(lengths[0]!);
  return length === undefined || length > 0;
}

function readRaw(req: IncomingMessage, length: number): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    req.on('data', (chunk: Buffer) => {
      total += chunk.length;
      if (total > length) {
        reject(new BodyError(400, 'request body is longer than Content-Length'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      if (total < length) reject(new BodyError(400, 'request body ended before Content-Length'));
      else resolve(Buffer.concat(chunks));
    });
    req.on('error', reject);
  });
}

/** Splash `_read_json_body`: the checks in its order, with its words. */
async function readJsonObject(req: IncomingMessage): Promise<Record<string, unknown>> {
  if (headerValues(req, 'transfer-encoding').length > 0) {
    throw new BodyError(400, 'transfer encoding is not supported');
  }
  const encodings = headerValues(req, 'content-encoding');
  if (encodings.length > 1 || (encodings.length === 1 && encodings[0]!.trim().toLowerCase() !== 'identity')) {
    throw new BodyError(415, 'content encoding is not supported');
  }
  const types = headerValues(req, 'content-type');
  const type = contentType(types[0]);
  if (
    types.length !== 1 ||
    !(type === 'application/json' || (type.startsWith('application/') && type.endsWith('+json')))
  ) {
    throw new BodyError(415, 'Content-Type must be application/json');
  }
  const lengths = headerValues(req, 'content-length');
  if (lengths.length !== 1) throw new BodyError(400, 'exactly one Content-Length header is required');
  const length = contentLength(lengths[0]!);
  if (length === undefined) throw new BodyError(400, 'invalid Content-Length header');
  if (length <= 0) throw new BodyError(400, 'request body must not be empty');
  if (length > MAX_BODY_BYTES) {
    throw new BodyError(413, `request body is ${length} bytes; limit is ${MAX_BODY_BYTES} bytes`, 'request_too_large');
  }
  const raw = await readRaw(req, length);
  let body: unknown;
  try {
    body = JSON.parse(raw.toString('utf-8'));
  } catch {
    throw new BodyError(400, 'invalid JSON request body');
  }
  if (body === null || typeof body !== 'object' || Array.isArray(body)) {
    throw new BodyError(400, 'request body must be an object');
  }
  return body as Record<string, unknown>;
}

/** Splash `WeightsSnapshot` as its status JSON spells it. */
export function weightsJson(weights: WeightsSnapshot): Record<string, unknown> {
  return {
    idle_release_seconds: weights.idleReleaseSeconds,
    released: weights.released,
    restoring: weights.restoring,
    restores: weights.restores,
    restore_failures: weights.restoreFailures,
  };
}

export async function handleMemoryRelease(
  req: IncomingMessage,
  res: ServerResponse,
  releaseMemory: () => Promise<MemoryReleaseAnswer>,
): Promise<void> {
  if (hasBody(req)) {
    let body: Record<string, unknown>;
    try {
      body = await readJsonObject(req);
    } catch (error) {
      if (!(error instanceof BodyError)) throw error;
      sendError(res, error.status, 'invalid_request_error', error.message, null, error.code);
      return;
    }
    // A field set to null is one left out (Splash `_serve`).
    for (const value of Object.values(body)) {
      if (value !== null) {
        sendError(res, 400, 'invalid_request_error', 'this endpoint takes no fields', null, 'invalid_request_error');
        return;
      }
    }
  }
  const answer = await releaseMemory();
  if (answer.kind === 'closing') {
    res.setHeader('Retry-After', '1');
    sendError(res, 503, 'server_error', 'server is shutting down', null, 'server_shutdown');
    return;
  }
  // A release leaves the weights released until the next load is done.
  if (answer.weights.released && !answer.weights.restoring) {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(stringifyStrictJson({ released: true, weights: weightsJson(answer.weights) }));
    return;
  }
  sendError(
    res,
    409,
    'invalid_request_error',
    'the engine is serving requests or restoring its weights; it releases them only while idle',
    null,
    'engine_busy',
  );
}
