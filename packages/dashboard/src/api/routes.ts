/**
 * The route table: `(method, pattern)` → handler, with `:param` segments.
 *
 * Pure data plus handler references — no transport, no matching logic (that is
 * `dispatch.ts`). Every transport walks this same table, so a route added here is
 * reachable over HTTP and over the Electron MessagePort bridge alike.
 */

import type { Handler } from './context.js';
import { handleCacheDelete, handleCacheGet } from './handlers/cache.js';
import {
  handleDownloadCancel,
  handleDownloadEventsUnsupported,
  handleDownloadStart,
  handleDownloadsList,
} from './handlers/downloads.js';
import { handleHealth } from './handlers/health.js';
import { handleIngest } from './handlers/ingest.js';
import { handleMetricsOverview, handleSessionMetrics } from './handlers/metrics.js';
import { handleCatalog, handleDeleteModel, handleModels } from './handlers/models.js';
import {
  handleSessionDelete,
  handleSessionDetail,
  handleSessionRename,
  handleSessionsList,
} from './handlers/sessions.js';

export interface Route {
  method: string;
  segments: string[];
  handler: Handler;
  /**
   * Status the HTTP adapter answers on success. Defaults to 200; a route that
   * only ACCEPTS work (the download start) declares 202 here instead of the
   * handler knowing anything about status codes.
   */
  successStatus?: number;
  /**
   * A long-lived stream the transport owns end to end (SSE framing, heartbeats,
   * backpressure). Kept in the table so path/method matching — and therefore the
   * 405-vs-404 distinction — stays identical for it, but intercepted by the
   * transport before `dispatch` ever runs its handler.
   */
  stream?: true;
}

function route(
  method: string,
  pattern: string,
  handler: Handler,
  extra?: Omit<Route, 'method' | 'segments' | 'handler'>,
): Route {
  return { method, segments: pattern.split('/').filter((s) => s.length > 0), handler, ...extra };
}

export const ROUTES: Route[] = [
  route('GET', '/health', handleHealth),
  // Alias under `/api` so the health check is reachable through the UI dev
  // server's `/api` proxy (which does not forward the bare `/health`).
  route('GET', '/api/health', handleHealth),
  route('GET', '/api/models', handleModels),
  route('DELETE', '/api/models/:name', handleDeleteModel),
  route('GET', '/api/catalog', handleCatalog),
  route('GET', '/api/downloads', handleDownloadsList),
  route('POST', '/api/downloads', handleDownloadStart, { successStatus: 202 }),
  route('DELETE', '/api/downloads/:id', handleDownloadCancel),
  route('GET', '/api/downloads/:id/events', handleDownloadEventsUnsupported, { stream: true }),
  route('GET', '/api/sessions', handleSessionsList),
  route('GET', '/api/sessions/:id', handleSessionDetail),
  route('PATCH', '/api/sessions/:id', handleSessionRename),
  route('DELETE', '/api/sessions/:id', handleSessionDelete),
  route('GET', '/api/sessions/:id/metrics', handleSessionMetrics),
  route('GET', '/api/metrics/overview', handleMetricsOverview),
  route('GET', '/api/cache', handleCacheGet),
  route('DELETE', '/api/cache', handleCacheDelete),
  route('POST', '/api/ingest', handleIngest),
];
