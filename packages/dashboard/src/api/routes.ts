/**
 * The route table: `(method, pattern)` → handler, with `:param` segments, plus
 * the THREAD that owns the handler.
 *
 * Pure data plus handler references — no transport, no matching logic (that is
 * `dispatch.ts`). Every transport walks this same table, so a route added here is
 * reachable over HTTP and over the Electron MessagePort bridge alike.
 *
 * Ownership is declared here and nowhere else: `runtime.ts` routes by this
 * marker rather than by a path list, so a route added later cannot silently land
 * on the wrong thread. The two variants also make it a TYPE error to move a route
 * across threads — a worker handler needs `dash`/`runIngest`, which the transport
 * thread does not have, and vice versa for the download manager.
 */

import type { Handler, MainApiContext, WorkerApiContext } from './context.js';
import { handleCacheDelete, handleCacheGet } from './handlers/cache.js';
import {
  handleChatAgent,
  handleChatCancel,
  handleChatConversations,
  handleChatCreate,
  handleChatDelete,
  handleChatEngineUnload,
  handleChatInteraction,
  handleChatPatch,
  handleChatPermissionMode,
  handleChatPrompt,
  handleChatState,
} from './handlers/chat.js';
import { handleCodingAgents, handleCodingAgentAction } from './handlers/coding-agents.js';
import {
  handleDownloadCancel,
  handleDownloadEventsUnsupported,
  handleDownloadStart,
  handleDownloadsList,
} from './handlers/downloads.js';
import { handleHealth } from './handlers/health.js';
import { handleIngest } from './handlers/ingest.js';
import {
  handleMetricsDistributions,
  handleMetricsModels,
  handleMetricsSpeculative,
  handleMetricsSummary,
  handleMetricsTimeseries,
  handleMetricsTurns,
} from './handlers/metrics.js';
import {
  handleCatalog,
  handleCatalogUpdates,
  handleDeleteModel,
  handleModels,
  handleCodingAgentModels,
} from './handlers/models.js';
import {
  handleProviderKeyDelete,
  handleProviderKeyPut,
  handleProviderSelection,
  handleProvidersList,
  handleProviderValidate,
} from './handlers/providers.js';
import { handleRuntimeLive } from './handlers/runtime-live.js';
import { handleRuntimeSeries } from './handlers/runtime.js';
import {
  handleSessionDelete,
  handleSessionDetail,
  handleSessionRename,
  handleSessionsList,
  handleSessionTimeline,
} from './handlers/sessions.js';

/** Which thread answers a route. */
export type RouteThread = 'main' | 'worker';

interface RouteShape {
  method: string;
  segments: string[];
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

/** Answered on the transport thread, which owns the download manager. */
export interface MainRoute extends RouteShape {
  thread: 'main';
  handler: Handler<MainApiContext>;
}

/** Answered on the database worker, which owns SQLite and the synchronous FS walks. */
export interface WorkerRoute extends RouteShape {
  thread: 'worker';
  handler: Handler<WorkerApiContext>;
}

export type Route = MainRoute | WorkerRoute;

type RouteExtra = Omit<RouteShape, 'method' | 'segments'>;

function segmentsOf(pattern: string): string[] {
  return pattern.split('/').filter((s) => s.length > 0);
}

function mainRoute(method: string, pattern: string, handler: Handler<MainApiContext>, extra?: RouteExtra): MainRoute {
  return { thread: 'main', method, segments: segmentsOf(pattern), handler, ...extra };
}

function workerRoute(
  method: string,
  pattern: string,
  handler: Handler<WorkerApiContext>,
  extra?: RouteExtra,
): WorkerRoute {
  return { thread: 'worker', method, segments: segmentsOf(pattern), handler, ...extra };
}

export const ROUTES: Route[] = [
  // Health answers from the worker deliberately: a supervisor pinging it wants to
  // know the thread that holds the database is still servicing messages, which a
  // reply from the transport thread would not tell it.
  workerRoute('GET', '/health', handleHealth),
  // Alias under `/api` so the health check is reachable through the UI dev
  // server's `/api` proxy (which does not forward the bare `/health`).
  workerRoute('GET', '/api/health', handleHealth),
  workerRoute('GET', '/api/models', handleModels),
  mainRoute('GET', '/api/coding-agents', handleCodingAgents),
  workerRoute('GET', '/api/coding-agents/models', handleCodingAgentModels),
  mainRoute('POST', '/api/coding-agents', handleCodingAgentAction),
  workerRoute('DELETE', '/api/models/:name', handleDeleteModel),
  workerRoute('GET', '/api/catalog', handleCatalog),
  mainRoute('GET', '/api/catalog/updates', handleCatalogUpdates),
  // The download routes stay on the transport thread: network I/O and progress
  // events must never queue behind a synchronous query, and a cancel click has to
  // be RECEIVED while one is running.
  mainRoute('GET', '/api/downloads', handleDownloadsList),
  mainRoute('POST', '/api/downloads', handleDownloadStart, { successStatus: 202 }),
  mainRoute('DELETE', '/api/downloads/:id', handleDownloadCancel),
  mainRoute('GET', '/api/downloads/:id/events', handleDownloadEventsUnsupported, { stream: true }),
  workerRoute('GET', '/api/metrics/models', handleMetricsModels),
  workerRoute('GET', '/api/metrics/summary', handleMetricsSummary),
  workerRoute('GET', '/api/metrics/timeseries', handleMetricsTimeseries),
  workerRoute('GET', '/api/metrics/distributions', handleMetricsDistributions),
  workerRoute('GET', '/api/metrics/speculative', handleMetricsSpeculative),
  workerRoute('GET', '/api/metrics/turns', handleMetricsTurns),
  workerRoute('GET', '/api/runtime/series', handleRuntimeSeries),
  // Small files read with async fs on the transport thread; no database involved.
  mainRoute('GET', '/api/runtime/live', handleRuntimeLive),
  workerRoute('GET', '/api/cache', handleCacheGet),
  workerRoute('DELETE', '/api/cache', handleCacheDelete),
  workerRoute('POST', '/api/ingest', handleIngest),
  workerRoute('GET', '/api/sessions', handleSessionsList),
  workerRoute('GET', '/api/sessions/:id', handleSessionDetail),
  workerRoute('GET', '/api/sessions/:id/timeline', handleSessionTimeline),
  workerRoute('PATCH', '/api/sessions/:id', handleSessionRename),
  workerRoute('DELETE', '/api/sessions/:id', handleSessionDelete),
  // Chat routes own agent processes and their streams, so they live with the transport. Only the list reads the
  // session folders synchronously, on the worker.
  mainRoute('GET', '/api/chat/state', handleChatState),
  workerRoute('GET', '/api/chat/conversations', handleChatConversations),
  mainRoute('POST', '/api/chat/conversations', handleChatCreate, { successStatus: 201 }),
  mainRoute('POST', '/api/chat/conversations/:id/prompt', handleChatPrompt, { successStatus: 202 }),
  mainRoute('POST', '/api/chat/conversations/:id/cancel', handleChatCancel),
  mainRoute('POST', '/api/chat/conversations/:id/interactions/:iid', handleChatInteraction),
  mainRoute('POST', '/api/chat/conversations/:id/permission-mode', handleChatPermissionMode),
  mainRoute('PATCH', '/api/chat/conversations/:id', handleChatPatch),
  mainRoute('DELETE', '/api/chat/conversations/:id', handleChatDelete),
  mainRoute('POST', '/api/chat/agent', handleChatAgent),
  mainRoute('POST', '/api/chat/engine/unload', handleChatEngineUnload),
  // Provider routes call the Keychain and the provider APIs and hold the provider lock, with the Chat service.
  mainRoute('GET', '/api/providers', handleProvidersList),
  mainRoute('PUT', '/api/providers/selection', handleProviderSelection),
  mainRoute('PUT', '/api/providers/:id/key', handleProviderKeyPut),
  mainRoute('DELETE', '/api/providers/:id/key', handleProviderKeyDelete),
  mainRoute('POST', '/api/providers/:id/validate', handleProviderValidate),
];
