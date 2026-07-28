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
  /**
   * The slug directory is occupied by something the downloader does not own and
   * holds no loadable checkpoint (an interrupted `mlx download`, a hand-made
   * folder, a symlink). Install is refused server-side in this state, so the card
   * names the blockage rather than offering a button that always errors.
   */
  blockedByForeignDir: boolean;
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
  /**
   * Input + output tokens over exactly the sessions {@link total} counts.
   *
   * Served beside the count so a caller showing both cannot pair a session count
   * scoped to the current `--session-dir` with a token total taken from the
   * machine-wide `/metrics/overview`.
   */
  tokens: number;
  /**
   * Every directory the index holds, sorted — the source for the directory
   * filter. Unfiltered by the request's own `cwd`/`q`/date params (the dropdown
   * has to offer the way out of a filter too) and independent of `limit`, so a
   * directory whose sessions all fall outside the returned page is still
   * offerable.
   */
  cwds: string[];
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

/** Lookups the trend's root filter left out, so the page can say so out loud. */
export interface CacheExcludedBucket {
  turns: number;
  hits: number;
  misses: number;
}

/** What `trend` is (and is not) counting. */
export interface CacheScope {
  /** Canonical cache root every `trend` row is filtered to. */
  root: string;
  /** Trace retention, i.e. the real width of the trend window. */
  trendWindowDays: number;
  /** Turns recorded before cache attribution existed — excluded, not lost. */
  legacy: CacheExcludedBucket;
  /** Turns attributed to a DIFFERENT cache directory. */
  otherRoots: CacheExcludedBucket;
  /**
   * Turns whose tier was ON but which recorded no root. The writer cannot
   * produce these any more; the bucket keeps the partition EXHAUSTIVE so such a
   * row's lookups are shown rather than silently dropped.
   */
  unattributed: CacheExcludedBucket;
  /** Turns that ran with the cold tier switched off (their deltas are all 0). */
  disabledTurns: number;
  /**
   * Sidecar captures recorded by turns carrying no root — the one counter the
   * buckets above cannot report, because a tier-less turn has no hits or
   * misses to put in one. `record_capture_reached()` is the first statement of
   * every family's capture, above its `adapter.cold_tier()` guard, so a hybrid
   * turn under `--no-persist-cache` or one whose tier failed to open still
   * counts the capture. Kept out of {@link ColdTierHealth} because these turns
   * name no cache root and `health` promises one.
   */
  unrootedSidecarCaptures: number;
}

/**
 * Cold-tier counters beyond hit/miss. The `*Total` fields are CUMULATIVE
 * process totals reduced with MAX (not summed deltas), so a fault during a
 * turn that aborted before it could record a delta still shows.
 */
export interface ColdTierHealth {
  enqueued: number;
  queueDrops: number;
  evictions: number;
  corruptions: number;
  corruptionsTotal: number;
  queueDropsTotal: number;
  /**
   * Writes the queue accepted that never reached disk. The native writer is
   * fail-open and reports its error to nobody, so this is the only number that
   * separates a cache storing nothing from a cache with nothing to store.
   */
  writeErrors: number;
  writeErrorsTotal: number;
  /**
   * Restores the walk refused. Neither a hit nor a miss — without this, a
   * refused restore renders as "no lookups recorded", which reads as "nothing
   * ran" rather than "reuse was refused". SUMMED, not MAX'd: a decline is
   * normal on any prompt's first turn, so a latch would carry no information.
   */
  restoreDeclines: number;
  /**
   * Restores a family threw away after the walk served them. Also summed, and
   * the eighth of the `ColdSidecarStats` family below — unprefixed because it
   * is read beside `restoreDeclines`, the other way reuse is refused.
   */
  restoreSuppressed: number;
  /**
   * Turns that entered a family's sidecar capture. Every other `sidecar*`
   * counter is a sub-count of this one. Only the hybrid families
   * (gemma4 / qwen3_5 / qwen3_5_moe) ever enter it, so `0` beside real block
   * traffic is the normal reading for a dense model, not a fault.
   */
  sidecarCaptureReached: number;
  /** Captures that found no whole persisted block to anchor state under. */
  sidecarChainEmpty: number;
  /** Captures whose chain covered blocks but had no retained checkpoint at or below its reach. */
  sidecarBoundarySkips: number;
  /**
   * Captures that selected a boundary already on disk — the steady state of a
   * repeated prompt. It is what separates a healthy run (`sidecarEnqueued = 0`
   * because the first turn already wrote the chain) from one whose ladder
   * never reaches a persistable rung.
   */
  sidecarAlreadyPersisted: number;
  /**
   * Sidecars handed to the writer queue. The sidecar SHARE of `enqueued`, not
   * a second population — one admission bumps both, so never add them.
   */
  sidecarEnqueued: number;
  /** Sidecars the queue refused. Likewise the sidecar share of `queueDrops`. */
  sidecarQueueDrops: number;
  /**
   * Restored sidecars a family INSTALLED as live state. The one counter here
   * that cannot be inferred: every `install_*_cold_sidecar` early-return falls
   * through to a full O(prefix) replay producing CORRECT state, so a
   * regression from "restored and used" to "restored and re-derived" leaves
   * the hit rate, the trend and the corruption count untouched.
   */
  sidecarInstalled: number;
}

export interface CacheResponse {
  disk: ColdCacheDiskInfo;
  trend: CacheTrendRow[];
  scope: CacheScope;
  health: ColdTierHealth;
  /** Families whose prefixes may be restored — served, never bundled. */
  restoreFamilies: string[];
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
