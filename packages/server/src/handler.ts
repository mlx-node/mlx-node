/** Composable `(req, res)` handler for node:http — usable standalone or mounted into an existing server. */

import type { IncomingMessage, ServerResponse } from 'node:http';

import type { ResponseStore } from '@mlx-node/core';

import { sendInternalError } from './errors.js';
import type { IdleSweeper } from './idle-sweeper.js';
import type { ModelRegistry } from './registry.js';
import { routeRequest } from './router.js';

export interface HandlerOptions {
  /** Enable CORS headers (default: true). */
  cors?: boolean;
  /** Response store for previous_response_id support. */
  store?: ResponseStore | null;
  /**
   * Retention (seconds) stamped as `expires_at` when committing a response row.
   * When omitted, the endpoint falls back to its own default (see `responses.ts`
   * and `ServerConfig.responseRetentionSec`).
   */
  responseRetentionSec?: number;
  /**
   * Optional idle sweeper. Forwarded to `routeRequest` so that the
   * inference endpoints (`/v1/responses` + `/v1/messages`) can bracket
   * their native-model dispatch with `beginRequest()` / `endRequest()`.
   *
   * Note: the begin/end hooks are DELIBERATELY scoped to the inference
   * endpoints — wrapping every HTTP call (OPTIONS preflights,
   * `/v1/models`, `/v1/health`, 404s) would let purely observational
   * traffic keep the allocator pinned forever.
   */
  idleSweeper?: IdleSweeper | null;
}

export function createHandler(
  registry: ModelRegistry,
  options?: HandlerOptions,
): (req: IncomingMessage, res: ServerResponse) => Promise<void> {
  const cors = options?.cors ?? true;
  const store = options?.store ?? null;
  const responseRetentionSec = options?.responseRetentionSec;
  const idleSweeper = options?.idleSweeper ?? null;

  return async (req: IncomingMessage, res: ServerResponse): Promise<void> => {
    if (cors) {
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, x-api-key, anthropic-version');

      if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
      }
    }

    // Returning the promise lets tests await the full lifecycle including
    // post-`res.end()` bookkeeping (e.g. `SessionRegistry.adopt`). `http.createServer`
    // ignores the return value, so this is transparent to production callers.
    try {
      await routeRequest(req, res, registry, store, responseRetentionSec, idleSweeper);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Internal server error';
      if (!res.headersSent) {
        sendInternalError(res, message);
      } else {
        res.end();
      }
    }
  };
}
