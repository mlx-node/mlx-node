/** Full HTTP server lifecycle: wires up the handler and periodically sweeps expired `ResponseStore` rows and sessions. */

import { mkdir } from 'node:fs/promises';
import { createServer as httpCreateServer } from 'node:http';
import type { Server, ServerResponse } from 'node:http';
import { homedir } from 'node:os';
import { join } from 'node:path';

import { ResponseStore } from '@mlx-node/core';

import type { PublicModelEntry } from './handler.js';
import { createHandler } from './handler.js';
import { createHealthReporter, type ServerHealth } from './health.js';
import { createIdleSweeper, DEFAULT_IDLE_CLEAR_CACHE_MS, parseIdleClearCacheEnv } from './idle-sweeper.js';
import { runGuardedModelLoad, type LoadModelOptions } from './load-model.js';
import { ModelWorkCoordinator } from './model-work-coordinator.js';
import { ModelRegistry } from './registry.js';
import { activeSSEStreamCountForResponses } from './streaming.js';

/** Cleanup interval for expired responses (ms). */
const CLEANUP_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

/**
 * Default grace period before {@link ServerInstance.close} destroys
 * still-open connections.
 *
 * 5 s is deliberately a little longer than the ~5 s GPU watchdog window: a
 * decode loop that is about to yield its next token should get the chance to
 * unwind cleanly rather than being cut off a hair early.
 */
const DEFAULT_CLOSE_TIMEOUT_MS = 5000;

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

/**
 * Resolve the effective auth token from an explicit value plus the env
 * fallback.
 *
 * Exported because `createInferenceHost` has to answer "is this server going
 * to be protected?" BEFORE it binds, in order to refuse a non-loopback bind
 * that would serve anonymously. Two independent copies of this rule would
 * drift, and the direction it would drift is a host that refuses to start
 * while `MLX_SERVER_AUTH_TOKEN` is sitting right there in the environment.
 *
 * An empty env var means "not set". An accidental `MLX_SERVER_AUTH_TOKEN=` in
 * a launcher script must not enable auth with an empty secret that every
 * credential-less request would then fail against.
 *
 * An empty EXPLICIT token is a different case and is rejected outright, for the
 * same reason {@link normalizePositiveIntConfig} rejects a bogus explicit knob:
 * somebody asked for a token and supplied nothing, which is
 * `--auth-token "$TOKEN"` with `TOKEN` unset. Returning `''` made the bind
 * guard read "auth is configured" and allow `0.0.0.0`, while the comparator
 * accepted an empty `x-api-key` because both strings were empty — a wildcard
 * bind published under a credential anyone can guess. Quietly downgrading to
 * "no auth" instead would be fail-open in the other direction: on loopback it
 * hands back the unauthenticated, wildcard-CORS server the operator was
 * explicitly trying not to start. Throwing is the only answer that is wrong in
 * neither bind mode, and it happens before anything binds or loads.
 */
export function resolveAuthToken(explicit: string | undefined): string | undefined {
  if (explicit === '') {
    throw new Error(
      'authToken was set to an empty string; pass a real secret or omit it entirely ' +
        '(an unset shell variable in `--auth-token "$VAR"` is the usual cause).',
    );
  }
  if (explicit !== undefined) return explicit;
  const fromEnv = process.env.MLX_SERVER_AUTH_TOKEN;
  return fromEnv != null && fromEnv !== '' ? fromEnv : undefined;
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
  /**
   * Enable CORS headers.
   *
   * Default: `true` when no `authToken` is in effect (historical behaviour),
   * `false` once one is. An explicit value always wins. See
   * {@link ServerConfig.authToken}.
   */
  cors?: boolean;
  /**
   * Shared secret required on every route except `/health` and `/v1/health`.
   *
   * Default: `process.env.MLX_SERVER_AUTH_TOKEN`, or `undefined` (no auth)
   * when that is unset or empty. `undefined` is byte-for-byte identical to
   * the pre-auth behaviour. An explicit `''` is rejected rather than treated as
   * either — see {@link resolveAuthToken}.
   *
   * Accepted as `x-api-key: <token>` or `authorization: Bearer <token>`.
   * Setting it also flips the `cors` default to `false`.
   */
  authToken?: string;
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
  /**
   * Optional async callback invoked before the endpoint layer looks
   * the model up in the registry. Intended for lazy-load schemes:
   * the callback should register the model into `registry` if it can;
   * on return, the endpoint does `registry.get(body.model)` and 404s
   * if still unresolved. Callback errors bubble up as 500s.
   */
  resolveModel?: (name: string) => Promise<void>;
  /**
   * Optional override for `GET /v1/models` enumeration. When provided,
   * the endpoint returns this list instead of `registry.list()`.
   * Intended for dynamic discovery schemes (e.g. enumerate every model
   * on disk while only the currently-resident one is registered).
   */
  listModels?: () => PublicModelEntry[];
}

/** Options for {@link ServerInstance.close}. */
export interface CloseOptions {
  /**
   * Grace period, in milliseconds, before still-open connections are
   * destroyed. Default: {@link DEFAULT_CLOSE_TIMEOUT_MS} (5000).
   *
   * Only the FIRST `close()` call's value is honoured — later calls receive
   * the memoized promise of the first, so their timeout is ignored.
   */
  timeoutMs?: number;
}

/** Outcome of {@link ServerInstance.close}. */
export interface CloseResult {
  /** `true` when the grace period expired and connections were destroyed. */
  forced: boolean;
  /** SSE streams open at the moment of the forced destroy. `0` when not forced. */
  streamsAborted: number;
  /** Wall-clock duration of the shutdown. */
  durationMs: number;
}

export interface ServerInstance {
  server: Server;
  /** Register models before or after starting. */
  registry: ModelRegistry;
  /** Null when disabled. */
  store: ResponseStore | null;
  /**
   * Coordinates process-wide MLX work: model loads take the exclusive writer
   * slot, inference takes shared reader slots. Exposed so callers can compose
   * their own brackets (or read `writerActive` / `lastLoad` for diagnostics).
   * Prefer {@link loadModel} for the common load case — it also handles the
   * drain suspension, which is easy to get wrong.
   */
  readonly modelWork: ModelWorkCoordinator;
  /**
   * Current readiness snapshot — the same body an authenticated
   * `GET /health` returns. Pure JavaScript state; no native calls.
   */
  health(): ServerHealth;
  /**
   * Bounded, idempotent shutdown.
   *
   * Stops accepting new connections, drops idle (keep-alive) ones
   * immediately, then waits up to `timeoutMs` for the rest to finish. On
   * expiry every remaining connection is destroyed, which fires the same
   * `res.on('close')` path a client disconnect fires — so in-flight SSE
   * generations are cancelled through `@mlx-node/lm` down to the native
   * `ChatStreamHandle`.
   *
   * Idempotent: the promise is memoized, so repeated calls return the same
   * promise and the same result. In particular a second call does NOT
   * reject with `ERR_SERVER_NOT_RUNNING`.
   *
   * RESIDUAL: `ResponseStore` has no `close()` (it is a Rust-side handle),
   * so the SQLite connection is released by process exit, not here. A
   * long-lived process that creates and closes many servers will hold one
   * store handle per server.
   */
  close(opts?: CloseOptions): Promise<CloseResult>;
  /**
   * Load a model out-of-band and register it, with idle drains suspended and
   * inference excluded for the entire operation — including the wait for the
   * coordinator's writer lock.
   *
   * This is the safe way to swap the resident model on a server that is
   * already serving. Hand-rolling the two brackets in the wrong order races
   * the process-wide Metal allocator; see `load-model.ts` for the full
   * rationale.
   *
   * Rejects with the underlying error if `load()` throws; both brackets
   * unwind cleanly, and `health().lastLoad` records the failure under
   * `opts.name`.
   */
  loadModel(opts: LoadModelOptions): Promise<void>;
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
  const authToken = resolveAuthToken(config?.authToken);
  // Forwarded unresolved so `createHandler` applies the auth-aware default
  // (`true` without a token, `false` with one) in exactly one place.
  const cors = config?.cors;
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
  const modelWorkCoordinator = new ModelWorkCoordinator();

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

  // One reporter shared by the HTTP endpoint and `ServerInstance.health()`
  // so both report the same uptime origin.
  const health = createHealthReporter({ registry, idleSweeper, modelWorkCoordinator });

  const handler = createHandler(registry, {
    cors,
    store,
    responseRetentionSec,
    idleSweeper,
    resolveModel: config?.resolveModel,
    modelWorkCoordinator,
    listModels: config?.listModels,
    authToken,
    health,
  });
  /**
   * Responses owned by THIS HTTP server. The SSE registry is deliberately
   * process-wide because `beginSSE` is also used by standalone handlers;
   * shutdown accounting intersects it with this weak ownership set so one
   * server cannot claim streams that another server leaves live. A WeakSet
   * needs no second response cleanup lifecycle.
   */
  const ownedResponses = new WeakSet<ServerResponse>();
  const server = httpCreateServer((req, res) => {
    // Synchronous, before `handler` can reach either endpoint's `beginSSE`.
    ownedResponses.add(res);
    void handler(req, res);
  });

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

  // Memoized so a second `close()` returns the first call's promise instead
  // of re-entering `server.close()` (which invokes its callback with
  // ERR_SERVER_NOT_RUNNING once the server is already down).
  let closePromise: Promise<CloseResult> | null = null;

  const close = (opts?: CloseOptions): Promise<CloseResult> => {
    if (closePromise !== null) return closePromise;
    const startedAt = Date.now();
    const requested = opts?.timeoutMs;
    const timeoutMs =
      typeof requested === 'number' && Number.isFinite(requested) && requested >= 0
        ? requested
        : DEFAULT_CLOSE_TIMEOUT_MS;

    closePromise = (async (): Promise<CloseResult> => {
      clearInterval(cleanupTimer);
      idleSweeper.close();

      let forced = false;
      let streamsAborted = 0;

      // Arm the wait BEFORE dropping idle sockets so nothing can complete in
      // the gap and settle `server.close()` before we are listening.
      const serverClosed = new Promise<void>((resolve, reject) => {
        server.close((err) => {
          // Tolerated defensively: memoization should make this unreachable,
          // but a caller who also closed `instance.server` directly would
          // otherwise turn a successful shutdown into a rejection.
          if (err && (err as NodeJS.ErrnoException).code !== 'ERR_SERVER_NOT_RUNNING') reject(err);
          else resolve();
        });
      });

      // Keep-alive sockets parked in a client pool hold no request; without
      // this the shutdown would sit on them until the client's own timeout.
      server.closeIdleConnections();

      const forceTimer = setTimeout(() => {
        forced = true;
        // Snapshot BEFORE destroying: `closeAllConnections()` fires each
        // response's `'close'` event, which unregisters it from the global SSE
        // registry used by the intersection.
        streamsAborted = activeSSEStreamCountForResponses(ownedResponses);
        // Destroying the socket fires exactly the `res.on('close')` path a
        // client disconnect fires, so the endpoint's AbortController cancels
        // the native stream handle rather than leaking a running decode.
        server.closeAllConnections();
      }, timeoutMs);

      try {
        await serverClosed;
      } finally {
        clearTimeout(forceTimer);
      }

      return { forced, streamsAborted, durationMs: Date.now() - startedAt };
    })();

    return closePromise;
  };

  return {
    server,
    registry,
    store,
    modelWork: modelWorkCoordinator,
    health,
    close,
    loadModel(opts: LoadModelOptions): Promise<void> {
      return runGuardedModelLoad({ idleSweeper, modelWorkCoordinator, registry }, opts);
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
