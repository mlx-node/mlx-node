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
  /** Absolute path of the models directory these checkpoints were discovered in. */
  dir: string;
}

export interface CatalogItem {
  label: string;
  hfRepo: string;
  sizeGb: number;
  description: string;
  isDefault?: boolean;
  hidden?: boolean;
  slug: string;
  /** A dashboard-owned (completion-marker) install of this model is present. */
  installed: boolean;
  /**
   * A loadable checkpoint is present on disk regardless of the dashboard marker —
   * true for {@link installed}, and also for a model installed via the `mlx
   * download` CLI / wizard. Gate the Install action on this (an unowned present
   * model can't be overwritten).
   */
  present: boolean;
}

export interface CatalogResponse {
  items: CatalogItem[];
}

export interface DownloadJob {
  id: string;
  repo: string;
  state: 'running' | 'committing' | 'done' | 'error' | 'cancelled';
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
  /** Total sessions matching the filter, before `limit`/`offset` paging. */
  total: number;
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
  /** Averages are null when no trace in the bucket carried the column. */
  decodeTps: number | null;
  prefillTps: number | null;
  ttftMs: number | null;
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
  /** KV blocks only; state sidecars are counted by `sidecarCount`. */
  entryCount: number;
  sidecarCount: number;
  /** Blocks + sidecars, i.e. everything the cold-tier quota covers. */
  totalBytes: number;
  sidecarBytes: number;
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
  /** One-line digest of the arguments (path / command / …), shown on the header. */
  summary: string;
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
  /** The model id that produced an `assistant` message; drives its logo + name. */
  model?: string;
  /** Base64 image blocks, rendered inline as thumbnails. */
  images?: Array<{ mimeType: string; data: string }>;
  /** Chip labels (e.g. `HEIC · 32 KB`) for binary blobs shown in place of raw bytes. */
  binaryNotes?: string[];
  /** Coordinate-mapping image-read notes split out of `text`; hidden by default. */
  imageNotes?: string[];
  /** One-line digest of the originating call's args (path / command); on `toolResult` rows. */
  title?: string;
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
