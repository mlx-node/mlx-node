/** Path-based router for /v1/* endpoints. */

import type { IncomingMessage, ServerResponse } from 'node:http';

import type { ResponseStore } from '@mlx-node/core';

import { handleCountMessageTokens } from './endpoints/messages-count-tokens.js';
import { handleCreateMessage } from './endpoints/messages.js';
import { handleListModels } from './endpoints/models.js';
import { handleCreateResponse } from './endpoints/responses.js';
import {
  sendAnthropicBadRequest,
  sendAnthropicMethodNotAllowed,
  sendBadRequest,
  sendMethodNotAllowed,
  sendNotFound,
} from './errors.js';
import type { PublicModelEntry } from './handler.js';
import { toMinimalHealth, type ServerHealth } from './health.js';
import type { IdleSweeper } from './idle-sweeper.js';
import type { ModelWorkCoordinator } from './model-work-coordinator.js';
import type { ModelRegistry } from './registry.js';
import type { AnthropicCountTokensRequest, AnthropicMessagesRequest } from './types-anthropic.js';
import type { ResponsesAPIRequest } from './types.js';

/** Max request body size (10 MB). */
const MAX_BODY_BYTES = 10 * 1024 * 1024;

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let totalBytes = 0;
    req.on('data', (chunk: Buffer) => {
      totalBytes += chunk.length;
      if (totalBytes > MAX_BODY_BYTES) {
        reject(new Error('Request body too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf-8')));
    req.on('error', reject);
  });
}

/**
 * Trailing options bag. Added as an object rather than two more positional
 * parameters — `routeRequest` already carries nine, and the two knobs here
 * are unrelated to each other.
 */
export interface RouteExtras {
  /** Builds the `/health` body. Omitted ⇒ the legacy constant `{ status: 'ok' }`. */
  health?: () => ServerHealth;
  /**
   * Whether the caller presented a valid token. Only consulted by `/health`,
   * which is the one route reachable without one. `true` when no token is
   * configured at all, so an unprotected server keeps serving the full body.
   */
  authenticated?: boolean;
}

export async function routeRequest(
  req: IncomingMessage,
  res: ServerResponse,
  registry: ModelRegistry,
  store: ResponseStore | null,
  responseRetentionSec?: number,
  idleSweeper?: IdleSweeper | null,
  resolveModel?: (name: string) => Promise<void>,
  listModels?: () => PublicModelEntry[],
  modelWorkCoordinator?: ModelWorkCoordinator,
  extras?: RouteExtras,
): Promise<void> {
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
  const path = url.pathname;

  if (path === '/v1/models') {
    if (req.method !== 'GET') {
      sendMethodNotAllowed(res, 'GET');
      return;
    }
    handleListModels(res, registry, listModels);
    return;
  }

  if (path === '/v1/responses') {
    if (req.method !== 'POST') {
      sendMethodNotAllowed(res, 'POST');
      return;
    }

    let body: ResponsesAPIRequest;
    try {
      const raw = await readBody(req);
      body = JSON.parse(raw) as ResponsesAPIRequest;
    } catch (err) {
      const msg =
        err instanceof Error && err.message === 'Request body too large' ? err.message : 'Invalid JSON in request body';
      sendBadRequest(res, msg);
      return;
    }

    await handleCreateResponse(
      res,
      body,
      registry,
      store,
      req,
      responseRetentionSec,
      idleSweeper,
      modelWorkCoordinator,
    );
    return;
  }

  if (path === '/v1/messages/count_tokens') {
    if (req.method !== 'POST') {
      sendAnthropicMethodNotAllowed(res, 'POST');
      return;
    }

    let body: AnthropicCountTokensRequest;
    try {
      const raw = await readBody(req);
      body = JSON.parse(raw) as AnthropicCountTokensRequest;
    } catch (err) {
      const msg =
        err instanceof Error && err.message === 'Request body too large' ? err.message : 'Invalid JSON in request body';
      sendAnthropicBadRequest(res, msg);
      return;
    }

    await handleCountMessageTokens(res, body, registry, idleSweeper, resolveModel, modelWorkCoordinator);
    return;
  }

  if (path === '/v1/messages') {
    if (req.method !== 'POST') {
      sendAnthropicMethodNotAllowed(res, 'POST');
      return;
    }

    let body: AnthropicMessagesRequest;
    try {
      const raw = await readBody(req);
      body = JSON.parse(raw) as AnthropicMessagesRequest;
    } catch (err) {
      const msg =
        err instanceof Error && err.message === 'Request body too large' ? err.message : 'Invalid JSON in request body';
      sendAnthropicBadRequest(res, msg);
      return;
    }

    await handleCreateMessage(res, body, registry, req, idleSweeper, resolveModel, modelWorkCoordinator);
    return;
  }

  if (path === '/health' || path === '/v1/health') {
    // Deliberately NOT bracketed by `idleSweeper.beginRequest/endRequest`
    // and free of any native call: a supervisor polling on an interval must
    // not keep pushing the drain timer out, nor touch the MLX allocator.
    // Every field is read from plain JavaScript state.
    const health = extras?.health?.();
    if (health === undefined) {
      // No reporter wired (a bare `createHandler` mounted by hand): keep the
      // historical constant so existing consumers are unaffected.
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok' }));
      return;
    }
    // Unauthenticated pollers get liveness only. `models.resident` leaks
    // project names and local paths, so it stays behind the token.
    const body = extras?.authenticated === false ? toMinimalHealth(health) : health;
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(body));
    return;
  }

  // Liveness probe at `/`. Claude Code issues `HEAD /` before its first
  // request; respond 200 so the probe doesn't leave a 404 in the logs.
  if (path === '/') {
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      sendMethodNotAllowed(res, 'GET, HEAD');
      return;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(req.method === 'HEAD' ? undefined : JSON.stringify({ service: 'mlx-node' }));
    return;
  }

  sendNotFound(res, `No route matches ${req.method} ${path}`);
}
