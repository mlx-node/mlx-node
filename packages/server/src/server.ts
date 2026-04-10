/**
 * createServer() -- full HTTP server lifecycle.
 *
 * Creates an `http.Server`, wires up the handler, and manages a ResponseStore
 * with periodic cleanup of expired entries.
 */

import { mkdir } from 'node:fs/promises';
import { createServer as httpCreateServer } from 'node:http';
import type { Server } from 'node:http';
import { homedir } from 'node:os';
import { join } from 'node:path';

import { ResponseStore } from '@mlx-node/core';

import { createHandler } from './handler.js';
import { ModelRegistry } from './registry.js';

/** Cleanup interval for expired responses (ms). */
const CLEANUP_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

export interface ServerConfig {
  /** Port to listen on (default: 8080). */
  port?: number;
  /** Hostname to bind to (default: '127.0.0.1'). */
  host?: string;
  /** Path to the SQLite response store (default: ~/.mlx-node/responses.db). */
  storePath?: string;
  /** Disable response storage entirely (default: false). */
  disableStore?: boolean;
  /** Enable CORS headers (default: true). */
  cors?: boolean;
}

export interface ServerInstance {
  /** The underlying `http.Server`. */
  server: Server;
  /** The model registry -- register models before or after starting. */
  registry: ModelRegistry;
  /** The response store (null if disabled). */
  store: ResponseStore | null;
  /** Gracefully close the server and cleanup resources. */
  close(): Promise<void>;
}

/**
 * Create and start an MLX-Node HTTP server exposing the Responses API and Messages API.
 *
 * Endpoints:
 * - `POST /v1/responses` — OpenAI Responses API
 * - `POST /v1/messages` — Anthropic Messages API
 * - `GET /v1/models` — List registered models
 *
 * @example
 * ```typescript
 * import { createServer } from '@mlx-node/server';
 * import { Qwen35Model } from '@mlx-node/lm';
 *
 * const { registry, close } = await createServer({ port: 8080 });
 * const model = await Qwen35Model.load('./models/qwen3.5-3b');
 * registry.register('qwen3.5-3b', model);
 *
 * // Server is now accepting requests at POST /v1/responses and POST /v1/messages
 * // Ctrl-C or call close() to stop.
 * ```
 */
export async function createServer(config?: ServerConfig): Promise<ServerInstance> {
  const port = config?.port ?? 8080;
  const host = config?.host ?? '127.0.0.1';
  const cors = config?.cors ?? true;
  const disableStore = config?.disableStore ?? false;

  const registry = new ModelRegistry();

  // Initialize response store
  let store: ResponseStore | null = null;
  if (!disableStore) {
    const storePath = config?.storePath ?? join(homedir(), '.mlx-node', 'responses.db');
    const storeDir = join(storePath, '..');
    await mkdir(storeDir, { recursive: true });
    store = await ResponseStore.open(storePath);
  }

  // Periodic cleanup
  let cleanupTimer: ReturnType<typeof setInterval> | null = null;
  if (store) {
    cleanupTimer = setInterval(() => {
      store!.cleanupExpired().catch(() => {
        // Silently ignore cleanup errors
      });
    }, CLEANUP_INTERVAL_MS);
    // Allow the process to exit even if the timer is still active
    cleanupTimer.unref();
  }

  const handler = createHandler(registry, { cors, store });
  const server = httpCreateServer(handler);

  await new Promise<void>((resolve, reject) => {
    const onError = (err: Error) => {
      server.removeListener('error', onError);
      reject(err);
    };
    server.on('error', onError);
    server.listen(port, host, () => {
      server.removeListener('error', onError);
      resolve();
    });
  });

  return {
    server,
    registry,
    store,
    async close() {
      if (cleanupTimer) {
        clearInterval(cleanupTimer);
        cleanupTimer = null;
      }
      await new Promise<void>((resolve, reject) => {
        server.close((err) => {
          if (err) reject(err);
          else resolve();
        });
      });
    },
  };
}
