/**
 * The transport-independent handler contract.
 *
 * A handler receives an {@link ApiContext} (the long-lived resources the
 * dashboard runtime owns) plus an {@link ApiRequest} (one already-parsed call)
 * and returns its response body, or throws an `ApiError`. Nothing here knows
 * about `node:http`: the same handlers serve the HTTP adapter in `server.ts` and,
 * later, an Electron MessagePort bridge.
 */

import type { DashboardDb } from '../db/open.js';
import type { DownloadManager } from '../download.js';
import { ApiError } from './errors.js';

export interface IngestSummary {
  sessions: { scanned: number; updated: number; removed: number; warnings: string[] };
  traces: { files: number; records: number; pruned: number; warnings: string[] };
}

/** Long-lived resources the handlers read, supplied by the dashboard runtime. */
export interface ApiContext {
  dash: DashboardDb;
  modelsDir: string;
  sessionsRoot: string;
  tracesDir: string;
  /** Cold-tier root; `undefined` lets `cache.ts` resolve the running-tier default. */
  cacheRoot: string | undefined;
  downloads: DownloadManager;
  /** Serialized incremental rescan (sessions + traces). */
  runIngest: () => Promise<IngestSummary>;
}

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
 */
export type Handler = (ctx: ApiContext, req: ApiRequest) => unknown;

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
