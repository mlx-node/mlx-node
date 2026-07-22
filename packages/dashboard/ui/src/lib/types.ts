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
  state: 'running' | 'done' | 'error' | 'cancelled';
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

/** Result of `DELETE /api/downloads/:id` (cancel an in-flight/queued download). */
export interface CancelDownloadResponse {
  cancelled: boolean;
  id: string;
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

/** One day's token totals from `GET /api/metrics/overview` `tokensByDay` (UTC day). */
export interface TokensByDayRow {
  /** `YYYY-MM-DD` (UTC). */
  day: string;
  input: number;
  output: number;
  cached: number;
  reasoning: number;
}

/** Per-model throughput averages (not a time series) — averages are null when no sample carried the column. */
export interface ThroughputByModelRow {
  model: string;
  avgDecodeTps: number | null;
  avgPrefillTps: number | null;
  avgTtftMs: number | null;
  samples: number;
}

/**
 * One (model, UTC-day) throughput bucket from `GET /api/metrics/overview`
 * `throughputTrend` — the time-bucketed series backing the temporal charts.
 * Shared contract with the server overview query; the values are per-day
 * averages over `samples` traces for that model.
 */
export interface ThroughputTrendPoint {
  model: string;
  /** `YYYY-MM-DD` (UTC). */
  day: string;
  decodeTps: number;
  prefillTps: number;
  ttftMs: number;
  samples: number;
}

/** Per-model speculative-decoding (MTP) acceptance averages; only models with a recorded mean appear. */
export interface MtpByModelRow {
  model: string;
  meanAccepted: number | null;
  avgCycles: number | null;
  samples: number;
}

/** Per-model usage totals for the share chart. */
export interface ModelShareRow {
  model: string;
  turns: number;
  outputTokens: number;
}

export interface MetricsOverviewResponse {
  range: { from: number | null; to: number | null };
  tokensByDay: TokensByDayRow[];
  throughputByModel: ThroughputByModelRow[];
  throughputTrend: ThroughputTrendPoint[];
  mtpByModel: MtpByModelRow[];
  modelShare: ModelShareRow[];
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

/** Result of `DELETE /api/cache` (clear all or evict older-than). */
export interface CacheMutationResult {
  removed: number;
  freedBytes: number;
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
