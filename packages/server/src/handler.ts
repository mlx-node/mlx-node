/** Composable `(req, res)` handler for node:http — usable standalone or mounted into an existing server. */

import type { IncomingMessage, ServerResponse } from 'node:http';

import type { ResponseStore } from '@mlx-node/core';

import { hasCredential, isAuthorized, sendUnauthorized } from './auth.js';
import { sendInternalError } from './errors.js';
import { createHealthReporter, type ServerHealth } from './health.js';
import type { IdleSweeper } from './idle-sweeper.js';
import { ModelWorkCoordinator } from './model-work-coordinator.js';
import type { ModelRegistry } from './registry.js';
import { requestPathname, routeRequest } from './router.js';

/**
 * Routes reachable WITHOUT a token when one is configured.
 *
 * A supervisor has to poll liveness before it can be handed a token (it may
 * be the thing that generates it). Everything served here is scrubbed to
 * `{ status, uptimeMs, pid }` by the router when the caller is
 * unauthenticated — see `toMinimalHealth`.
 *
 * `/` is included because it is a pure liveness stub: the router answers it
 * with a constant `{ service: 'mlx-node' }` and reads no state, so there is
 * nothing to leak. Claude Code issues `HEAD /` before its first request — it
 * does send `x-api-key`, so it would pass the check anyway, but gating a
 * contentless probe would 401 every other client's liveness check for no
 * security gain.
 */
const UNAUTHENTICATED_PATHS = new Set(['/', '/health', '/v1/health']);

/**
 * Entry in the list returned by `GET /v1/models`. Matches the shape
 * `ModelRegistry.list()` produces — exported so callers that supply a
 * custom `listModels` callback (e.g. `mlx launch claude` discovering
 * every model on disk) can build entries without importing internals.
 */
export interface PublicModelEntry {
  id: string;
  object: 'model';
  created: number;
  owned_by: string;
}

export interface HandlerOptions {
  /**
   * Enable CORS headers.
   *
   * Default: `true` when no `authToken` is set (historical behaviour), and
   * `false` once one is — `Access-Control-Allow-Origin: *` on a
   * token-protected server would invite any web page to spend a leaked token
   * from the user's browser. An explicit value always wins.
   */
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
  /**
   * Optional async callback invoked by `/v1/messages` before it looks
   * the model up in the registry. The callback should register the
   * model on demand; on return, the endpoint does `registry.get(name)`
   * and 404s if still unresolved.
   */
  resolveModel?: (name: string) => Promise<void>;
  /**
   * Coordinates process-wide MLX work so lazy model loads / warmups do not
   * overlap live inference on another model.
   */
  modelWorkCoordinator?: ModelWorkCoordinator;
  /**
   * Optional override for `GET /v1/models`. When provided, that endpoint
   * returns this list instead of `registry.list()`.
   */
  listModels?: () => PublicModelEntry[];
  /**
   * Shared secret required on every route except `/health` and `/v1/health`.
   *
   * `undefined` (the default) disables the gate entirely and is byte-for-byte
   * identical to the pre-auth behaviour: no header is inspected, no
   * `WWW-Authenticate` is emitted, and CORS keeps its `true` default.
   *
   * Accepted as `x-api-key: <token>` (checked first — Anthropic clients send
   * it) or `authorization: Bearer <token>` (scheme case-insensitive).
   */
  authToken?: string;
  /**
   * Builds the `/health` body. Supplied by `createServer` so the HTTP
   * endpoint and `ServerInstance.health()` share one uptime origin. When
   * omitted, `createHandler` builds its own reporter from `registry`,
   * `idleSweeper` and `modelWorkCoordinator`.
   */
  health?: () => ServerHealth;
}

export function createHandler(
  registry: ModelRegistry,
  options?: HandlerOptions,
): (req: IncomingMessage, res: ServerResponse) => Promise<void> {
  const authToken = options?.authToken;
  // CORS defaults follow the auth posture; an explicit value still wins.
  const cors = options?.cors ?? authToken === undefined;
  const store = options?.store ?? null;
  const responseRetentionSec = options?.responseRetentionSec;
  const idleSweeper = options?.idleSweeper ?? null;
  const resolveModel = options?.resolveModel;
  const modelWorkCoordinator = options?.modelWorkCoordinator ?? (resolveModel ? new ModelWorkCoordinator() : undefined);
  const listModels = options?.listModels;
  const health =
    options?.health ?? createHealthReporter({ registry, idleSweeper, modelWorkCoordinator: modelWorkCoordinator });

  return async (req: IncomingMessage, res: ServerResponse): Promise<void> => {
    // The guard spans the WHOLE listener, not just `routeRequest`. `http.createServer`
    // discards the returned promise, so anything that escapes here is an unhandled
    // rejection — and Node's default `--unhandled-rejections=throw` turns that into
    // process death. A crash-by-request is the one failure a request handler must
    // not have, so nothing before the routing call gets to be outside the try either.
    try {
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

      // Single auth choke point. `routeRequest` is called from exactly one
      // place (below), so no route can be added that bypasses this.
      //
      // `authToken === undefined` short-circuits before any header is touched:
      // an unprotected server behaves exactly as it did before auth existed.
      let authenticated = true;
      if (authToken !== undefined) {
        authenticated = isAuthorized(req, authToken);
        if (!authenticated) {
          // Host-independent: see `requestPathname`. Building the base from the
          // `Host` header threw on `Host: [`, from a branch only an
          // UNAUTHENTICATED request reaches — so enabling auth was what made the
          // server killable by anyone who could open the socket.
          const path = requestPathname(req);
          // `/health` degrades to a liveness-only body instead of 401 — but
          // ONLY when no credential was offered. A caller who presented a
          // WRONG token gets the 401 it needs to notice the typo.
          if (!UNAUTHENTICATED_PATHS.has(path) || hasCredential(req)) {
            sendUnauthorized(res);
            return;
          }
        }
      }

      // Returning the promise lets tests await the full lifecycle including
      // post-`res.end()` bookkeeping (e.g. `SessionRegistry.adopt`). `http.createServer`
      // ignores the return value, so this is transparent to production callers.
      await routeRequest(
        req,
        res,
        registry,
        store,
        responseRetentionSec,
        idleSweeper,
        resolveModel,
        listModels,
        modelWorkCoordinator,
        { health, authenticated },
      );
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
