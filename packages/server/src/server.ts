/** Full HTTP server lifecycle: wires up the handler and periodically sweeps expired `ResponseStore` rows and sessions. */

import { mkdir } from 'node:fs/promises';
import { createServer as httpCreateServer } from 'node:http';
import type { Server } from 'node:http';
import { homedir } from 'node:os';
import { join } from 'node:path';

import { ResponseStore } from '@mlx-node/core';

import { createHandler } from './handler.js';
import { createIdleSweeper, DEFAULT_IDLE_CLEAR_CACHE_MS, parseIdleClearCacheEnv } from './idle-sweeper.js';
import { ModelRegistry } from './registry.js';

/** Cleanup interval for expired responses (ms). */
const CLEANUP_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

/**
 * Default retention for persisted response rows in SQLite, in seconds.
 *
 * Decoupled from the in-memory `SessionRegistry` TTL (30 min) so a client
 * sending `previous_response_id` after the warm KV cache has been evicted
 * can still cold-replay from disk via `reconstructMessagesFromChain` +
 * `ChatSession.startFromHistory`. 7 days trades disk for continuity at the
 * cost of a one-time prefill on recovery.
 */
const DEFAULT_RESPONSE_RETENTION_SECONDS = 7 * 24 * 60 * 60; // 7 days

/**
 * Parse a positive integer seconds value; returns undefined for unset/invalid so caller can apply its own default.
 *
 * Non-integer positive values (e.g. `"1.5"`) are rejected rather than
 * silently truncated — a typo like `"1.5"` meant as `"15"` would otherwise
 * be accepted as 1 second, expiring persisted response rows almost
 * immediately and breaking `previous_response_id` continuity. We prefer
 * falling through to the caller's default over crashing on startup so a
 * config-template typo in a Dockerfile / CI manifest does not take the
 * service down.
 *
 * Exported for unit tests.
 */
export function parseEnvSeconds(name: string): number | undefined {
  const raw = process.env[name];
  if (raw == null || raw === '') return undefined;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return undefined;
  if (!Number.isInteger(parsed)) return undefined;
  return parsed;
}

/**
 * Parse a positive integer count from env; shares the reject-unset-or-invalid
 * semantics used by {@link parseEnvSeconds} (including the non-integer
 * reject) so callers can fall back to their own default when the var is
 * missing or malformed.
 *
 * Exported for unit tests.
 */
export function parseEnvPositiveInt(name: string): number | undefined {
  const raw = process.env[name];
  if (raw == null || raw === '') return undefined;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return undefined;
  if (!Number.isInteger(parsed)) return undefined;
  return parsed;
}

/**
 * Validate a caller-supplied positive-integer config knob.
 *
 * Mirrors the reject-invalid semantics of {@link parseEnvPositiveInt} /
 * {@link parseEnvSeconds} but fails fast with a descriptive error when
 * the caller explicitly passes a bogus value. Silent coercion would hide
 * a config bug that can take the model offline (e.g. a
 * `maxQueueDepthPerModel: 0` makes `queuedCount >= limit` true for every
 * request, immediately returning HTTP 429; a `responseRetentionSec: 0`
 * stamps `expires_at = now` on every row and the next cleanup sweep
 * deletes it). `undefined` falls through so the env/default path still
 * applies.
 */
function normalizePositiveIntConfig(value: number | undefined, name: string): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer; received ${String(value)}`);
  }
  return value;
}

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
  /**
   * Retention for persisted response rows, in seconds. Stamped as `expires_at`
   * on each committed response; controls how long `previous_response_id`
   * cold-replay from SQLite remains possible after the warm session is evicted.
   *
   * Default: 7 days. Env override: `MLX_RESPONSE_RETENTION_SECONDS`. Ignored
   * when `disableStore` is true.
   */
  responseRetentionSec?: number;
  /**
   * Maximum number of concurrent requests that may be WAITING for the
   * per-model execution mutex (the one actively running does not count).
   * When the cap is reached, further requests return HTTP 429 with a
   * `Retry-After: 1` header so clients can back off instead of piling
   * into an unbounded queue.
   *
   * Default: `undefined` (unbounded — current behaviour). Env override:
   * `MLX_MAX_QUEUE_DEPTH_PER_MODEL` (positive integer).
   */
  maxQueueDepthPerModel?: number;
  /**
   * Milliseconds of HTTP inactivity (no request arrivals or completions)
   * before the server issues a single `clearCache()` to drain the MLX
   * Metal allocator's free pool. Replaces the old per-request
   * `ClearCacheOnDrop` guard in the Rust layer, which was unsafe on a
   * multi-model server because the pool is process-wide.
   *
   * Default: 30_000 ms (30 seconds). `0` disables the sweeper. Env
   * override: `MLX_IDLE_CLEAR_CACHE_MS` (non-negative integer; 0
   * disables; unset falls through to default).
   *
   * The decode-loop drain every 256 tokens inside each generative
   * model is untouched and covers in-flight memory churn.
   */
  idleClearCacheMs?: number;
}

export interface ServerInstance {
  server: Server;
  /** Register models before or after starting. */
  registry: ModelRegistry;
  /** Null when disabled. */
  store: ResponseStore | null;
  /** Graceful shutdown. */
  close(): Promise<void>;
  /**
   * Run `fn` with the idle-drain timer suspended for the duration of
   * an unbracketed, allocator-heavy operation — most commonly a hot
   * `Model::load()` invoked AFTER the server has already served at
   * least one request. In that scenario the post-request drain timer
   * armed by `endRequest()` (at t+idleClearCacheMs) can otherwise
   * fire MID-LOAD, racing the Metal allocator while weight
   * materialization is still in progress.
   *
   * Handles try/finally bracketing so a thrown load never leaks the
   * internal suspend counter. Accepts both sync and async functions;
   * returns `fn`'s own return value (or resolved promise). When the
   * bracket exits (normal or thrown) AND `inFlight === 0` with no
   * other suspend active, a fresh drain timer is armed.
   *
   * Safe to nest — each call allocates its own token-scoped release
   * so overlapping brackets unwind independently.
   *
   * The common `serve.ts` pattern (load all models BEFORE
   * `createServer()`) does not need this API — there is no armed
   * timer before the first request, so there is no race.
   *
   * Pass-through when the sweeper is disabled (`idleClearCacheMs: 0`
   * or missing `__internal__.clearCache`): `fn` is invoked directly.
   *
   * @example
   * ```ts
   * await instance.withSuspendedDrains(async () => {
   *   const model = await Qwen35Model.load(modelPath);
   *   instance.registry.register('new-model', model);
   * });
   * ```
   */
  withSuspendedDrains<T>(fn: () => Promise<T>): Promise<T>;
  withSuspendedDrains<T>(fn: () => T): T;
  /**
   * Low-level: suspend drains and return an idempotent, token-scoped
   * disposer. Prefer {@link withSuspendedDrains} unless you need
   * manual control over when the suspend is released. Safe to nest;
   * calling the disposer more than once is a no-op.
   *
   * No-op when the sweeper is disabled (`idleClearCacheMs: 0` or
   * missing `__internal__.clearCache`): returns a no-op disposer.
   */
  suspendDrains(): () => void;
}

/**
 * Start an MLX-Node HTTP server exposing `POST /v1/responses`,
 * `POST /v1/messages`, and `GET /v1/models`.
 *
 * @example
 * ```typescript
 * const { registry, close } = await createServer({ port: 8080 });
 * registry.register('qwen3.5-3b', await Qwen35Model.load('./models/qwen3.5-3b'));
 * ```
 */
export async function createServer(config?: ServerConfig): Promise<ServerInstance> {
  const port = config?.port ?? 8080;
  const host = config?.host ?? '127.0.0.1';
  const cors = config?.cors ?? true;
  const disableStore = config?.disableStore ?? false;
  // Validate caller-supplied numeric knobs BEFORE consulting env fallbacks
  // so a bogus explicit value surfaces as a descriptive error instead of
  // silently falling through to env / default. See
  // `normalizePositiveIntConfig` for the failure modes we're guarding.
  const configRetentionSec = normalizePositiveIntConfig(config?.responseRetentionSec, 'responseRetentionSec');
  const responseRetentionSec =
    configRetentionSec ?? parseEnvSeconds('MLX_RESPONSE_RETENTION_SECONDS') ?? DEFAULT_RESPONSE_RETENTION_SECONDS;
  // Opt-in queue-depth cap; resolved exactly once at server construction
  // so the registry (and its per-model `SessionRegistry` instances
  // allocated on `register()`) all share a single effective value.
  const configMaxQueueDepth = normalizePositiveIntConfig(config?.maxQueueDepthPerModel, 'maxQueueDepthPerModel');
  const maxQueueDepthPerModel = configMaxQueueDepth ?? parseEnvPositiveInt('MLX_MAX_QUEUE_DEPTH_PER_MODEL');

  // Idle sweeper wiring. `0` is a legal explicit "off" value so we
  // cannot reuse `normalizePositiveIntConfig` (which rejects 0).
  // Precedence: explicit config value wins over env wins over default.
  const idleClearCacheMs = resolveIdleClearCacheMs(config?.idleClearCacheMs);
  const idleSweeper = createIdleSweeper(idleClearCacheMs);
  // The sweeper is driven exclusively by `endRequest()` calls in the
  // inference endpoints (`/v1/responses`, `/v1/messages`). There is no
  // cold-start drain: a process that loads models but never receives a
  // request will not fire `clearCache()`. See the `idle-sweeper.ts`
  // module doc (section "Drain is post-request only") for rationale.
  const registry = new ModelRegistry({ maxQueueDepth: maxQueueDepthPerModel });

  let store: ResponseStore | null = null;
  if (!disableStore) {
    const storePath = config?.storePath ?? join(homedir(), '.mlx-node', 'responses.db');
    const storeDir = join(storePath, '..');
    await mkdir(storeDir, { recursive: true });
    store = await ResponseStore.open(storePath);
  }

  // Always schedule the sweep — sessions need TTL sweeps even without a store.
  const cleanupTimer: ReturnType<typeof setInterval> = setInterval(() => {
    if (store) {
      store.cleanupExpired().catch(() => {});
    }
    for (const sessReg of registry.listSessionRegistries()) {
      sessReg.sweep();
    }
  }, CLEANUP_INTERVAL_MS);
  cleanupTimer.unref();

  const handler = createHandler(registry, { cors, store, responseRetentionSec, idleSweeper });
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
      clearInterval(cleanupTimer);
      idleSweeper.close();
      await new Promise<void>((resolve, reject) => {
        server.close((err) => {
          if (err) reject(err);
          else resolve();
        });
      });
    },
    withSuspendedDrains<T>(fn: () => T | Promise<T>): T | Promise<T> {
      return idleSweeper.withSuspendedDrains(fn as () => Promise<T>);
    },
    suspendDrains(): () => void {
      return idleSweeper.suspendDrains();
    },
  };
}

/**
 * Resolve the effective idle-drain delay from (constructor value, env,
 * default). Unlike the other knobs, `0` is a legal explicit opt-out —
 * we must NOT fall through to env/default when the caller explicitly
 * passes `0`. Returns a non-negative integer milliseconds value.
 */
function resolveIdleClearCacheMs(configValue: number | undefined): number {
  if (configValue !== undefined) {
    if (
      typeof configValue !== 'number' ||
      !Number.isFinite(configValue) ||
      !Number.isInteger(configValue) ||
      configValue < 0
    ) {
      throw new Error(`idleClearCacheMs must be a non-negative integer; received ${String(configValue)}`);
    }
    return configValue;
  }
  const envValue = parseIdleClearCacheEnv();
  return envValue ?? DEFAULT_IDLE_CLEAR_CACHE_MS;
}
