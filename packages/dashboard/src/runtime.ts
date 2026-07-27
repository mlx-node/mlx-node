/**
 * The dashboard runtime: everything the API needs to answer a call, with no
 * transport attached.
 *
 * It owns the SQLite index, the download manager and the serialized incremental
 * ingest (boot run + periodic rescan), and exposes them as `call` / `subscribe` /
 * `ingestNow` / `close`. `server.ts` is a thin HTTP adapter over this; the
 * Electron app will drive the same object over a MessagePort.
 */

import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

import type { ApiContext, IngestSummary } from './api/context.js';
import { dispatch, isApiPath } from './api/dispatch.js';
import { failure, toFailure, type ApiResponse } from './api/errors.js';
import { openDashboardDb, type DashboardDb } from './db/open.js';
import { DownloadManager, type DownloadEvent } from './download.js';
import { ingestSessions } from './ingest/sessions.js';
import { ingestTraces } from './ingest/traces.js';
import { defaultModelsDir } from './models.js';
import { agentSessionsRoot, dashboardDbPath, metricsTraceDir } from './paths.js';

/** Trace-file retention passed to the periodic + boot ingest. */
const RETENTION_DAYS = 30;
/** Incremental rescan cadence while the runtime is alive. */
const INGEST_INTERVAL_MS = 30_000;

export interface DashboardRuntimeOptions {
  dbPath?: string;
  modelsDir?: string;
  sessionsRoot?: string;
  tracesDir?: string;
  cacheRoot?: string;
}

/** One API call, addressed the way `fetch` would address it. */
export interface ApiCall {
  method: string;
  /** Path, optionally with a query string (`/api/sessions?limit=1`). */
  path: string;
  /** Already-parsed payload; omit when the call carries none. */
  body?: unknown;
}

export interface DashboardRuntime {
  readonly modelsDir: string;
  readonly sessionsRoot: string;
  readonly tracesDir: string;
  readonly cacheRoot: string | undefined;
  /** The handler context, for a transport that needs to dispatch itself. */
  readonly context: ApiContext;
  /** Answer one API call. Never rejects for an API failure — see `ApiResponse`. */
  call(call: ApiCall): Promise<ApiResponse>;
  /** Subscribe to a download job's progress events; returns the unsubscribe. */
  subscribe(jobId: string, listener: (event: DownloadEvent) => void): () => void;
  /** Run (and await) an incremental rescan, serialized with every other ingest. */
  ingestNow(): Promise<IngestSummary>;
  /**
   * Stop the periodic rescan, drain in-flight downloads and await any in-flight
   * ingest — everything that must finish while the database is still open.
   * Idempotent; `close()` calls it for you.
   */
  drain(): Promise<void>;
  /** {@link drain} then close the SQLite handle. Idempotent. */
  close(): Promise<void>;
}

export function createDashboardRuntime(opts: DashboardRuntimeOptions = {}): DashboardRuntime {
  const dbPath = opts.dbPath ?? dashboardDbPath();
  const modelsDir = opts.modelsDir ?? defaultModelsDir();
  const sessionsRoot = opts.sessionsRoot ?? agentSessionsRoot();
  const tracesDir = opts.tracesDir ?? metricsTraceDir();
  const cacheRoot = opts.cacheRoot;

  if (dbPath !== ':memory:') {
    mkdirSync(dirname(dbPath), { recursive: true });
  }
  const dash: DashboardDb = openDashboardDb(dbPath);
  const downloads = new DownloadManager({ modelsDir });

  // Serialize ingests so overlapping timer/boot/manual runs never interleave
  // SQLite transactions. `doIngest` never throws, so a failed run cannot poison
  // the chain for later callers (a rejected promise would skip every queued
  // `.then`).
  const doIngest = async (): Promise<IngestSummary> => {
    try {
      const sessionsResult = await ingestSessions(dash, sessionsRoot);
      const tracesResult = await ingestTraces(dash, tracesDir, { retentionDays: RETENTION_DAYS });
      return { sessions: sessionsResult, traces: tracesResult };
    } catch (err) {
      return {
        sessions: { scanned: 0, updated: 0, removed: 0, warnings: [String(err)] },
        traces: { files: 0, records: 0, pruned: 0, warnings: [] },
      };
    }
  };
  let ingestChain: Promise<IngestSummary> = Promise.resolve({
    sessions: { scanned: 0, updated: 0, removed: 0, warnings: [] },
    traces: { files: 0, records: 0, pruned: 0, warnings: [] },
  });
  const runIngest = (): Promise<IngestSummary> => {
    ingestChain = ingestChain.then(doIngest);
    return ingestChain;
  };

  const context: ApiContext = { dash, modelsDir, sessionsRoot, tracesDir, cacheRoot, downloads, runIngest };

  // Ingest on boot (non-blocking) plus a periodic rescan.
  runIngest().catch(() => {});
  const ingestTimer = setInterval(() => {
    runIngest().catch(() => {});
  }, INGEST_INTERVAL_MS);
  ingestTimer.unref();

  let draining: Promise<void> | null = null;
  let dbClosed = false;

  const drain = (): Promise<void> => {
    draining ??= (async () => {
      clearInterval(ingestTimer);
      // Abort and drain in-flight downloads BEFORE anything else so a shutdown
      // (SIGINT/SIGTERM → `process.exit`) can't kill the process mid-write and
      // orphan a partial, potentially multi-GB `.staging` tree, nor let a
      // background job publish a model after the runtime is considered closed.
      await downloads.shutdown();
      // `clearInterval` only stops FUTURE ticks: a rescan already in flight kept
      // running while `dash.close()` executed underneath it, which only ever
      // looked harmless because `doIngest`'s catch swallowed the resulting
      // "database is not open". Await the chain to its fixpoint instead — re-read
      // after each await, since a caller can queue another ingest while we wait.
      let awaited: Promise<IngestSummary> | null = null;
      while (awaited !== ingestChain) {
        awaited = ingestChain;
        await awaited;
      }
    })();
    return draining;
  };

  return {
    modelsDir,
    sessionsRoot,
    tracesDir,
    cacheRoot,
    context,
    async call(c: ApiCall): Promise<ApiResponse> {
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
        return await dispatch(context, {
          method: c.method,
          pathname: url.pathname,
          query: url.searchParams,
          body: c.body,
        });
      } catch (err) {
        // `dispatch` converts handler throws itself; this catches a fault in
        // matching (e.g. `decodeURIComponent` on a malformed `%` escape) so the
        // documented "never rejects" contract holds for every caller. The HTTP
        // adapter keeps its own 500 path for the same case.
        return toFailure(err);
      }
    },
    subscribe(jobId: string, listener: (event: DownloadEvent) => void): () => void {
      return downloads.subscribe(jobId, listener);
    },
    ingestNow(): Promise<IngestSummary> {
      return runIngest();
    },
    drain,
    async close(): Promise<void> {
      await drain();
      if (dbClosed) return;
      dbClosed = true;
      dash.close();
    },
  };
}
