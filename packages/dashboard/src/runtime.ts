/**
 * The dashboard runtime: everything the API needs to answer a call, with no
 * transport attached.
 *
 * It spans TWO threads. This one owns the download manager, the rescan timer and
 * whatever transport is attached (`server.ts`'s HTTP adapter, or the Electron
 * MessagePort bridge). A `node:worker_threads` worker owns the SQLite index, the
 * serialized incremental ingest, and every route whose handler does synchronous
 * work. `node:sqlite` and drizzle are synchronous, so with everything on one
 * thread a single heavy query froze download progress for its whole duration
 * (measured: a trivial RPC issued during a 1.5 s synchronous call waited
 * 1449 ms), delayed a cancel click from even being RECEIVED, and false-positived
 * the supervisor's per-RPC timeouts.
 *
 * Which side answers a route is declared on the route itself (`routes.ts`), not
 * decided here.
 */

import { existsSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { IngestSummary, MainApiContext } from './api/context.js';
import { dispatchMain, isApiPath, routeThreadFor } from './api/dispatch.js';
import { failure, toFailure, type ApiResponse } from './api/errors.js';
import { DownloadManager, type DownloadEvent } from './download.js';
import { defaultModelsDir } from './models.js';
import { agentSessionsRoot, dashboardDbPath, metricsTraceDir } from './paths.js';
import { startDbWorker, type DbWorkerLifecycle } from './worker/client.js';

/** Incremental rescan cadence while the runtime is alive. */
const INGEST_INTERVAL_MS = 30_000;
/** Default budget for one shutdown step before the worker is terminated. */
const DEFAULT_SHUTDOWN_TIMEOUT_MS = 5_000;

export type RuntimeLifecycleEvent = DbWorkerLifecycle;

export interface DashboardRuntimeOptions {
  dbPath?: string;
  modelsDir?: string;
  sessionsRoot?: string;
  tracesDir?: string;
  cacheRoot?: string;
  /**
   * Override the database worker's entry module. Packaging needs this: under
   * Electron the worker must load from an unpacked path outside the asar archive.
   */
  workerUrl?: URL | string;
  /**
   * How long each shutdown step waits for the worker before it is terminated
   * outright. A supervisor with its own quit deadline (Electron's `before-quit`)
   * should set this below that deadline.
   */
  shutdownTimeoutMs?: number;
  /** Worker lifecycle notifications, for a supervisor that must react to a dead worker. */
  onLifecycle?: (event: RuntimeLifecycleEvent) => void;
}

/** One API call, addressed the way `fetch` would address it. */
export interface ApiCall {
  method: string;
  /** Path, optionally with a query string (`/api/sessions?limit=1`). */
  path: string;
  /** Already-parsed payload; omit when the call carries none. */
  body?: unknown;
  /** Set instead of {@link body} when the transport could not parse the payload. */
  bodyError?: string;
}

export interface DashboardRuntime {
  readonly modelsDir: string;
  readonly sessionsRoot: string;
  readonly tracesDir: string;
  readonly cacheRoot: string | undefined;
  /**
   * What THIS thread owns, for a transport that dispatches its own main-thread
   * routes. The database is deliberately absent — it lives on the worker and is
   * reachable only through {@link call}.
   */
  readonly context: MainApiContext;
  /**
   * Answer one API call. Never rejects for an API failure — see `ApiResponse`.
   * An optional signal withdraws worker work that has not started yet; a handler
   * already running is allowed to finish rather than being interrupted halfway
   * through a mutation.
   */
  call(call: ApiCall, signal?: AbortSignal): Promise<ApiResponse>;
  /** Subscribe to a download job's progress events; returns the unsubscribe. */
  subscribe(jobId: string, listener: (event: DownloadEvent) => void): () => void;
  /** Run (and await) an incremental rescan, serialized with every other ingest. */
  ingestNow(): Promise<IngestSummary>;
  /**
   * Stop the periodic rescan, drain in-flight downloads and await any in-flight
   * ingest — everything that must finish while the database is still open.
   * Idempotent; `close()` calls it for you. Rejects if the database worker cannot
   * confirm that its ingest chain reached a fixpoint within the shutdown budget.
   */
  drain(): Promise<void>;
  /**
   * {@link drain}, then close the SQLite handle and end the worker. Idempotent.
   * A drain failure is reported only after the bounded worker teardown still runs.
   */
  close(): Promise<void>;
}

/**
 * The shipped worker entry, or the source-run bootstrap when there is no build
 * beside us (vitest, `oxnode scripts/dev-dashboard.ts`). Checking for the built
 * file rather than sniffing the environment keeps a real `dist` authoritative —
 * a stale build can never be silently papered over by the `.ts`.
 */
function defaultWorkerUrl(): URL {
  const built = new URL('./worker/db-worker.js', import.meta.url);
  return existsSync(fileURLToPath(built)) ? built : new URL('./worker/ts-bootstrap.ts', import.meta.url);
}

export function createDashboardRuntime(opts: DashboardRuntimeOptions = {}): DashboardRuntime {
  const dbPath = opts.dbPath ?? dashboardDbPath();
  const modelsDir = opts.modelsDir ?? defaultModelsDir();
  const sessionsRoot = opts.sessionsRoot ?? agentSessionsRoot();
  const tracesDir = opts.tracesDir ?? metricsTraceDir();
  const cacheRoot = opts.cacheRoot;

  // Kept on this thread: one `mkdir` at construction is not the blocking problem,
  // and an unwritable location stays a synchronous, actionable throw from the
  // factory instead of an asynchronous worker death.
  if (dbPath !== ':memory:') {
    mkdirSync(dirname(dbPath), { recursive: true });
  }

  // Spawning is asynchronous but `new Worker` is not, and messages queue on the
  // port until the thread is up — so the factory stays synchronous and no caller
  // has to await readiness.
  const worker = startDbWorker({
    workerUrl: opts.workerUrl ?? defaultWorkerUrl(),
    shutdownTimeoutMs: opts.shutdownTimeoutMs ?? DEFAULT_SHUTDOWN_TIMEOUT_MS,
    onLifecycle: opts.onLifecycle ?? ((): void => {}),
    dbPath,
    modelsDir,
    sessionsRoot,
    tracesDir,
    cacheRoot,
  });
  const downloads = new DownloadManager({ modelsDir });

  const context: MainApiContext = { modelsDir, sessionsRoot, tracesDir, cacheRoot, downloads };

  // The boot ingest runs inside the worker; only the periodic rescan is driven
  // from here.
  const ingestTimer = setInterval(() => {
    void worker.ingest();
  }, INGEST_INTERVAL_MS);
  ingestTimer.unref();

  let draining: Promise<void> | null = null;

  const drain = (): Promise<void> => {
    draining ??= (async () => {
      clearInterval(ingestTimer);
      // Abort and drain in-flight downloads BEFORE anything else so a shutdown
      // (SIGINT/SIGTERM → `process.exit`) can't kill the process mid-write and
      // orphan a partial, potentially multi-GB `.staging` tree, nor let a
      // background job publish a model after the runtime is considered closed.
      await downloads.shutdown();
      // `clearInterval` only stops FUTURE ticks: a rescan already in flight kept
      // running while the database closed underneath it, which only ever looked
      // harmless because the ingest catch swallowed the resulting "database is
      // not open". The worker awaits its chain to a fixpoint for us.
      await worker.drain();
    })();
    return draining;
  };

  return {
    modelsDir,
    sessionsRoot,
    tracesDir,
    cacheRoot,
    context,
    async call(c: ApiCall, signal?: AbortSignal): Promise<ApiResponse> {
      if (signal?.aborted) {
        return failure('E_UNAVAILABLE', 'Dashboard request was cancelled before it started');
      }
      let url: URL;
      try {
        // A constant, trusted base: only the path + query of `c.path` are used.
        url = new URL(c.path, 'http://localhost');
      } catch {
        return failure('E_BAD_REQUEST', `Malformed request path ${c.path}`);
      }
      if (!isApiPath(url.pathname)) {
        return failure('E_NOT_FOUND', `No route matches ${c.method} ${url.pathname}`);
      }
      try {
        // Ownership is data on the route, never a path list here. An unmatched
        // request has no owner and is answered locally — its 405/404 needs no
        // context, and a garbage path never costs a round trip.
        //
        // INSIDE the try: deciding the owner runs the same route matcher, which
        // decodes each `:param`. A malformed escape threw from here, above the
        // catch, so `call` rejected instead of answering — breaking the contract
        // the catch below exists to keep. `worker.call` returns envelopes rather
        // than throwing, so nesting it changes nothing else.
        if (routeThreadFor(c.method, url.pathname) === 'worker') {
          return await worker.call(
            {
              method: c.method,
              path: c.path,
              body: c.body,
              ...(c.bodyError !== undefined ? { bodyError: c.bodyError } : {}),
            },
            signal,
          );
        }
        if (signal?.aborted) {
          return failure('E_UNAVAILABLE', 'Dashboard request was cancelled before it started');
        }
        return await dispatchMain(context, {
          method: c.method,
          pathname: url.pathname,
          query: url.searchParams,
          body: c.body,
          ...(c.bodyError !== undefined ? { bodyError: c.bodyError } : {}),
        });
      } catch (err) {
        // `dispatchMain` converts handler throws itself; this catches a fault in
        // matching (e.g. `decodeURIComponent` on a malformed `%` escape) so the
        // documented "never rejects" contract holds for every caller.
        return toFailure(err);
      }
    },
    subscribe(jobId: string, listener: (event: DownloadEvent) => void): () => void {
      return downloads.subscribe(jobId, listener);
    },
    ingestNow(): Promise<IngestSummary> {
      return worker.ingest();
    },
    drain,
    async close(): Promise<void> {
      // Even a failed drain must not leave the database worker alive. The drain
      // rejection remains the caller-visible proof that shutdown was not clean,
      // while close keeps its separate bounded teardown guarantee. Preserve both
      // errors if teardown itself also fails; otherwise a terminate failure would
      // hide the earlier unconfirmed ingest barrier.
      let drainFailed = false;
      let drainError: unknown;
      try {
        await drain();
      } catch (error) {
        drainFailed = true;
        drainError = error;
      }
      try {
        await worker.close();
      } catch (closeError) {
        if (drainFailed) {
          throw new AggregateError([drainError, closeError], 'Dashboard runtime failed to drain and close cleanly');
        }
        throw closeError;
      }
      if (drainFailed) throw drainError;
    },
  };
}
