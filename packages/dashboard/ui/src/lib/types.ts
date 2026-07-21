/**
 * Client-side mirrors of the dashboard API response shapes (server: C5
 * `packages/dashboard/src/api.ts`, data modules `models.ts` / `catalog.ts` /
 * `cache.ts` / `download.ts`). Kept in one place so the Overview and Models
 * pages consume identical types; only fields the UI reads are modelled.
 */

export interface LocalModel {
  name: string;
  path: string;
  modelType: string;
  quant: string | null;
  contextWindow: number | null;
  sizeBytes: number;
  fileCount: number;
}

export interface ModelsResponse {
  models: LocalModel[];
  warnings: string[];
}

export interface CatalogItem {
  label: string;
  hfRepo: string;
  sizeGb: number;
  description: string;
  isDefault?: boolean;
  hidden?: boolean;
  slug: string;
  installed: boolean;
}

export interface CatalogResponse {
  items: CatalogItem[];
}

export interface DownloadJob {
  id: string;
  repo: string;
  state: 'running' | 'done' | 'error';
  receivedBytes: number;
  totalBytes: number;
}

export interface DownloadsResponse {
  jobs: DownloadJob[];
}

export interface DownloadStartResponse {
  id: string;
  repo: string;
}

export interface DeleteModelResponse {
  deleted: boolean;
  name: string;
}

export interface SessionRow {
  id: string;
  path: string;
  cwd: string;
  name: string | null;
  created: number;
  modified: number;
  messageCount: number;
  firstMessage: string | null;
  models: string[];
  inputTokens: number;
  outputTokens: number;
}

export interface SessionsResponse {
  sessions: SessionRow[];
}

export interface MetricsOverviewResponse {
  range: { from: number | null; to: number | null };
  totals: {
    turns: number;
    traces: number;
    inputTokens: number;
    outputTokens: number;
    cachedTokens: number;
    reasoningTokens: number;
  };
}

export interface ColdCacheDiskInfo {
  root: string;
  exists: boolean;
  entryCount: number;
  totalBytes: number;
  quotaBytes: number;
  oldestMtime: number | null;
  newestMtime: number | null;
  ageHistogram: Array<{ label: string; count: number; bytes: number }>;
}

export interface CacheTrendRow {
  day: string;
  hits: number;
  misses: number;
  bytesWritten: number;
  bytesRestored: number;
}

export interface CacheResponse {
  disk: ColdCacheDiskInfo;
  trend: CacheTrendRow[];
}

export interface TranscriptToolCall {
  id: string;
  name: string;
  arguments: unknown;
}

/** One flattened transcript message from `GET /api/sessions/:id` (server: `TranscriptEntry`). */
export interface TranscriptEntry {
  role: string;
  text: string;
  toolCalls: TranscriptToolCall[];
  ts: number;
  /** Present on `toolResult` messages. */
  toolName?: string;
  isError?: boolean;
}

export interface SessionSummary {
  id: string;
  path: string;
  cwd: string;
  name: string | null;
  created: number;
  modified: number;
  messageCount: number;
  firstMessage: string | null;
}

export interface SessionDetailResponse {
  session: SessionSummary;
  transcript: TranscriptEntry[];
  /** Set when the transcript could not be read from the session file. */
  transcriptError?: string;
}

/**
 * A row from `GET /api/sessions/:id/metrics` `turns` — a turn LEFT JOINed to its
 * trace, so the trace-derived fields (`ttftMs`/`decodeTps`/…) are null when the
 * turn has no matching trace. Numeric fields arrive raw from SQLite.
 */
export interface SessionTurnMetric {
  entryId: string | null;
  traceId: string | null;
  ts: number;
  model: string | null;
  inputTokens: number | null;
  outputTokens: number | null;
  cachedTokens: number | null;
  reasoningTokens: number | null;
  ttftMs: number | null;
  prefillTps: number | null;
  decodeTps: number | null;
  mtpCycles: number | null;
  mtpMeanAccepted: number | null;
  durationMs: number | null;
  finishReason: string | null;
  coldHits: number | null;
  coldMisses: number | null;
  coldBytesWritten: number | null;
  coldBytesRestored: number | null;
}

/** A row from `GET /api/sessions/:id/metrics` `traces` (session traces, unjoined). */
export interface SessionTraceMetric {
  traceId: string;
  ts: number;
  model: string | null;
  ttftMs: number | null;
  prefillTps: number | null;
  decodeTps: number | null;
  mtpCycles: number | null;
  mtpMeanAccepted: number | null;
  durationMs: number | null;
  finishReason: string | null;
  coldHits: number | null;
  coldMisses: number | null;
  coldBytesWritten: number | null;
  coldBytesRestored: number | null;
}

export interface SessionMetricsResponse {
  sessionId: string;
  turns: SessionTurnMetric[];
  traces: SessionTraceMetric[];
}

export interface SessionRenameResponse {
  id: string;
  name: string;
}

export interface SessionDeleteResponse {
  deleted: boolean;
  id: string;
}
