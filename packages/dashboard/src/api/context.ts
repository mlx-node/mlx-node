/**
 * The transport-independent handler contract.
 *
 * A handler receives the context of the thread that owns its route plus an
 * {@link ApiRequest} (one already-parsed call) and returns its response body, or
 * throws an `ApiError`. Nothing here knows about `node:http`: the same handlers
 * serve the HTTP adapter in `server.ts` and the Electron MessagePort bridge.
 *
 * The resources are split by THREAD, because they live on different ones. The
 * SQLite handle (`node:sqlite` + drizzle are synchronous) and every synchronous
 * filesystem walk run on the database worker; the download manager stays on the
 * thread that owns the transport so progress events and a cancel click are never
 * queued behind a multi-second query. Which context a handler declares is what
 * pins it to a thread — `routes.ts` cannot put it on the other one.
 */

import type { DashboardDb } from '../db/open.js';
import type { DownloadManager } from '../download.js';
import { ApiError } from './errors.js';

export interface IngestSummary {
  sessions: { scanned: number; updated: number; removed: number; warnings: string[] };
  traces: { files: number; records: number; pruned: number; warnings: string[] };
}

/** Resolved locations, readable from either thread (both are handed the same values). */
export interface ApiPaths {
  modelsDir: string;
  sessionsRoot: string;
  tracesDir: string;
  /** Cold-tier root; `undefined` lets `cache.ts` resolve the running-tier default. */
  cacheRoot: string | undefined;
}

/** The database worker's context: it alone holds the SQLite handle and the ingest chain. */
export interface WorkerApiContext extends ApiPaths {
  dash: DashboardDb;
  /** Serialized incremental rescan (sessions + traces). */
  runIngest: () => Promise<IngestSummary>;
}

/** The transport thread's context: it alone holds the download manager. */
export interface MainApiContext extends ApiPaths {
  downloads: DownloadManager;
}

/**
 * Both halves at once. No thread owns this — it is the input to the omniscient
 * {@link dispatch}, used by callers that genuinely hold everything (a test
 * driving the route table against stubs).
 */
export interface ApiContext extends WorkerApiContext, MainApiContext {}

/** One already-parsed API call. */
export interface ApiRequest {
  method: string;
  /** Path only, no query string. */
  path: string;
  query: URLSearchParams;
  /** `:param` values captured from the matched route. */
  params: Record<string, string>;
  /** Parsed request payload (`null` when the caller sent none). */
  body: unknown;
  /**
   * Set instead of {@link body} when the transport could not parse the payload
   * (malformed JSON, over the size cap). Kept as data rather than raised at
   * parse time so each handler surfaces it exactly where it used to `await` the
   * body — preserving handlers whose identity/ownership checks run first.
   */
  bodyError?: string;
}

/**
 * A route handler: return the response body, or throw an `ApiError`. The return
 * type covers a promise too (`unknown` subsumes it) — `dispatch` awaits whatever
 * comes back, so a handler may be sync or async.
 *
 * `C` is the context the handler needs, and therefore the thread it can run on:
 * a handler declaring {@link ApiPaths} fits either side, one declaring
 * {@link WorkerApiContext} only fits a worker route.
 */
export type Handler<C = ApiContext> = (ctx: C, req: ApiRequest) => unknown;

/**
 * The request payload, or a 400 when the transport failed to parse it. Call this
 * at the point the old handler `await`ed `readJsonBody`, so earlier checks
 * (404 on an unknown id, …) keep winning over a body-parse failure.
 */
export function requireBody(req: ApiRequest): unknown {
  if (req.bodyError !== undefined) throw ApiError.badRequest(req.bodyError);
  return req.body;
}

/**
 * Apply the wire JSON rules a transport needs: an empty payload is `null` (not a
 * parse error), anything else must be valid JSON. Shared so every transport
 * agrees on the shape handed to {@link ApiRequest.body}.
 */
export function parseJsonBody(raw: string): { body: unknown } | { bodyError: string } {
  const trimmed = raw.trim();
  if (trimmed === '') return { body: null };
  try {
    return { body: JSON.parse(trimmed) };
  } catch {
    return { bodyError: 'Invalid JSON in request body' };
  }
}

export function toNum(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'bigint') return Number(value);
  return null;
}

export function toInt(value: unknown): number {
  const n = toNum(value);
  return n === null ? 0 : Math.trunc(n);
}

/** Positive-integer query param, or `null`. */
export function queryInt(query: URLSearchParams, name: string): number | null {
  const raw = query.get(name);
  if (raw === null || raw === '') return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}
