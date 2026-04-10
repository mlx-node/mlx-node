/**
 * createHandler() -- composable (req, res) handler for node:http.
 *
 * Can be used standalone with `http.createServer(handler)` or composed
 * into an existing server.
 */

import type { IncomingMessage, ServerResponse } from 'node:http';

import type { ResponseStore } from '@mlx-node/core';

import { sendInternalError } from './errors.js';
import type { ModelRegistry } from './registry.js';
import { routeRequest } from './router.js';

export interface HandlerOptions {
  /** Enable CORS headers (default: true). */
  cors?: boolean;
  /** Response store for previous_response_id support. */
  store?: ResponseStore | null;
}

/**
 * Create a composable request handler.
 *
 * @param registry - The model registry to serve.
 * @param options - Optional configuration.
 * @returns An async (req, res) handler suitable for `http.createServer()`.
 */
export function createHandler(
  registry: ModelRegistry,
  options?: HandlerOptions,
): (req: IncomingMessage, res: ServerResponse) => void {
  const cors = options?.cors ?? true;
  const store = options?.store ?? null;

  return (req: IncomingMessage, res: ServerResponse) => {
    // CORS preflight
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

    // Route the request (async)
    routeRequest(req, res, registry, store).catch((err: unknown) => {
      const message = err instanceof Error ? err.message : 'Internal server error';
      if (!res.headersSent) {
        sendInternalError(res, message);
      } else {
        res.end();
      }
    });
  };
}
