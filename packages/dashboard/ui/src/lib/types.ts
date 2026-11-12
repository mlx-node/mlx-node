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
  /**
   * Names this checkpoint is served and traced under (a GGUF folder serves one per variant). Match with
   * `servedIds.includes(key) || (servedIds.length === 0 && name === key)`.
   */
  servedIds: string[];
}

export interface ModelsResponse {
  models: LocalModel[];
  /** Draft-only disk inventory, excluded from model counts and the model table. */
  companions?: Array<Pick<LocalModel, 'name' | 'path' | 'sizeBytes' | 'fileCount'>>;
  warnings: string[];
  /** Absolute path of the models directory these checkpoints were discovered in. */
  dir: string;
  host: { totalMemBytes: number; diskFreeBytes: number | null };
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
  /** What occupies the slug; non-null exactly when `blockedByForeignDir` is true. */
  blocker: CatalogBlocker | null;
  /**
   * Commit sha the local bytes were pinned to, or `null` when no completion
   * marker exists to read (hand copy, pre-marker CLI install, renamed dir).
   * `null` means staleness is unknowable, never "up to date".
   */
  localRevision: string | null;
  /**
   * Base-model repo the install's tokenizer sidecars came from, and the commit
   * they were pinned to — or `null` when the install used none or predates the
   * field. Update discovery compares this pair too: a sidecar-only upstream
   * change moves nothing in the primary repo, so without it no badge appears
   * and the Installed button (disabled) keeps the repair job unreachable.
   */
  localAssetsRepo: string | null;
  localAssetsRevision: string | null;
  /**
   * Base-model repo the catalog entry CURRENTLY sources its tokenizer
   * sidecars from — compared against `localAssetsRepo` so a switched source
   * is itself an update (the marker's repo is what the probe map lacks).
   */
  assetsRepo?: string;
  /** Separate optional draft weights, never a standalone chat-model entry. */
  draft?: CatalogDraftItem;
  globs?: string[];
}

export type CatalogDraftItem = Pick<
  CatalogItem,
  'label' | 'hfRepo' | 'sizeGb' | 'slug' | 'installed' | 'present' | 'blockedByForeignDir' | 'blocker' | 'localRevision'
>;

/** An entry at a catalog slug that the app did not create, measured without following a link or opening a file. */
export interface CatalogBlocker {
  kind: 'directory' | 'symlink' | 'file';
  /** directory: files below it. file: 1. symlink: 0. */
  fileCount: number;
  sizeBytes: number;
  /** The walk stopped early: both numbers are lower bounds. */
  truncated: boolean;
  /** Relative paths, only for 3 files or fewer. */
  names: string[];
  /** Local models lists this entry. */
  listed: boolean;
  /** A `config.json` is at its root. */
  modelLike: boolean;
  /** `mlx serve` can list a model in it (a checkpoint or a target GGUF), so Reinstall refuses it. */
  holdsModel: boolean;
  linkTarget: string | null;
  dangling: boolean;
}

export interface CatalogResponse {
  items: CatalogItem[];
}

/** One row of `/catalog/updates` — the upstream half of the staleness check. */
export interface CatalogUpdateItem {
  hfRepo: string;
  /** Upstream commit sha, or `null` when Hugging Face could not be reached. */
  remoteRevision: string | null;
}

export interface CatalogUpdatesResponse {
  items: CatalogUpdateItem[];
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
  /** Local day start, epoch ms. */
  t: number;
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

/**
 * Sessions API (server: `src/agent-sessions/types.ts`), read from grok-build
 * (`mlx agent`) session folders. Timestamps in `SessionRow` are ISO strings as
 * stored in `summary.json`; transcript and turn `ts` are epoch milliseconds.
 */
export interface SessionTokens {
  input: number;
  output: number;
  cached: number;
}

/** Who started a session: a terminal `mlx agent`, or the app in Chat or Work mode. */
export type SessionOrigin = 'agent' | 'chat' | 'work';

export interface SessionRow {
  id: string;
  origin: SessionOrigin;
  /** The agent's title, else the title of the first typed prompt; '' only for a session with no prompt. */
  title: string;
  cwd: string;
  model: string | null;
  branch: string | null;
  createdAt: string;
  lastActiveAt: string;
  messageCount: number;
  live: boolean;
  kind: string | null;
  /** Fork source or subagent parent; may name a session that no longer exists. */
  parentId: string | null;
  subagentCount: number;
  /** `null` when no completed turn carried usage. */
  tokens: SessionTokens | null;
}

export interface SubagentRef {
  id: string;
  title: string;
  type: string;
  description: string;
  status: string;
  model: string | null;
  /** False when the child folder is gone or ambiguous; the UI must not link it. */
  exists: boolean;
}

export interface SessionDetail extends SessionRow {
  gitRoot: string | null;
  headCommit: string | null;
  remotes: string[];
  agentName: string | null;
  reasoningEffort: string | null;
  lastTurnSummary: string | null;
  /** `title: null` means the parent folder is missing. */
  parent: { id: string; title: string | null } | null;
  subagents: SubagentRef[];
}

/** What woke the agent for a turn the user did not type, read from the prompt id the agent recorded. */
export type WakeCause = 'workflow' | 'subagent' | 'task' | 'notifications' | 'goal' | 'unknown';

export type TranscriptEntry =
  | { kind: 'user'; ts: number; text: string; promptIndex: number | null; interjection?: boolean }
  /** A hidden prompt the agent wrote to itself when a background event woke it. Never a message of the user. */
  | {
      kind: 'wake';
      ts: number;
      cause: WakeCause;
      /** cause 'workflow': the run's name and status when it woke the agent. */
      workflow?: { name: string; status: string };
      /** cause 'subagent': the child whose finish woke the agent. */
      subagent?: { childId: string; description: string };
      /** Cut at 16 KB. */
      text: string;
      truncated?: true;
    }
  | { kind: 'assistant'; ts: number; text: string; endTs?: number }
  | { kind: 'thinking'; ts: number; text: string; endTs?: number }
  | {
      kind: 'tool';
      ts: number;
      id: string;
      name: string;
      title: string;
      status: string;
      input?: string;
      output?: string;
      truncated?: boolean;
      endTs?: number;
      /** A shell command's non-zero exit code. */
      exitCode?: number;
      /** A shell command stopped by the agent or the system (`timeout`, `oom`, …). */
      signal?: string;
      timedOut?: true;
    }
  | { kind: 'plan'; ts: number; entries: { content: string; status: string; priority: string }[] }
  | {
      kind: 'subagent';
      ts: number;
      childId: string;
      type: string;
      description: string;
      status?: string;
      durationMs?: number;
      tokens?: number;
      error?: string;
      /** Set by the detail route; only `true` may be linked. */
      exists?: boolean;
      /** Time of the `subagent_finished` line. */
      endTs?: number;
      /** Made by a finish line with no spawn line: `ts` is the finish time. */
      noSpawn?: true;
    }
  | { kind: 'model'; ts: number; modelId: string }
  | { kind: 'turn-end'; ts: number; stopReason: string; elapsedMs?: number; error?: string };

export interface TurnMetric {
  promptId: string;
  ts: number;
  inputTokens: number;
  outputTokens: number;
  cachedTokens: number;
  reasoningTokens: number;
  elapsedMs: number | null;
  /** Time to first token of the turn's first model call. */
  ttftMs: number | null;
  /** Summed measured prefill windows; set with `prefillTokens`, never alone. */
  prefillMs: number | null;
  prefillTokens: number | null;
  /** Model calls in the prefill sums. */
  prefillCalls: number | null;
  /** Model calls of the turn. */
  modelCalls: number | null;
  decodeTps: number | null;
  model: string | null;
}

export interface SessionListResponse {
  sessions: SessionRow[];
  total: number;
  cwds: string[];
}

export interface SessionDetailResponse {
  session: SessionDetail;
  transcript: TranscriptEntry[];
  turns: TurnMetric[];
  transcriptError?: string;
  /** Null for a Chat conversation: a terminal cannot open the Chat agent's home. */
  resumeCommand: string | null;
}

/** `PATCH /api/sessions/:id` body `{title}`; `title` is the sanitized title as stored. */
export interface SessionRenameResponse {
  id: string;
  title: string;
}

/** `DELETE /api/sessions/:id`: the session first, then its subagent children. */
export interface SessionDeleteResponse {
  deleted: string[];
}

export type TimelineLaneStatus = 'completed' | 'failed' | 'cancelled' | 'running' | 'unknown';
export type TimelineNoSteps =
  | 'missing'
  | 'empty'
  | 'too-large'
  | 'unreadable'
  /** The file has events, none inside this run. */
  | 'out-of-range'
  /** A finish line with no spawn line. */
  | 'no-spawn'
  | 'folded';

export interface TimelineLane {
  /** 'main', the child session id, or 'folded'. */
  id: string;
  role: 'main' | 'subagent' | 'folded';
  name: string;
  /** Subagent type; null on the main lane of a top-level session and on the folded lane. */
  type: string | null;
  model: string | null;
  /** 0..7 for a subagent lane; null = neutral. */
  slot: number | null;
  /** The lane's own session when its folder exists; only then the UI links it. */
  sessionId: string | null;
  status: TimelineLaneStatus;
  /** The lane's first run was spawned before this window. */
  carriedIn: boolean;
  /** The window is finished and the lane's last run returns after it. */
  carriedOut: boolean;
  /** Set when the lane has one `run` span instead of steps. */
  noSteps: TimelineNoSteps | null;
  /** Size of the file that was refused, for 'too-large'. */
  fileBytes: number | null;
  /** A step crossed a handoff time and was cut there. Events are never moved. */
  clamped: boolean;
  /** Subagents this lane spawned itself (not drawn). On the folded lane: how many subagents it stands for. */
  subagents: number;
  /** Usage reports inside the window, ascending. */
  tokenReports: Array<{ atMs: number; tokens: number }>;
}

interface TimelineSpanBase {
  /** `${lane}:${index}`, stable while the files do not change. */
  id: string;
  lane: string;
  /** ms from range.startMs. */
  startMs: number;
  /** null only with status 'running'. */
  endMs: number | null;
  status: 'ok' | 'failed' | 'running' | 'unfinished';
  inferredStart: boolean;
  inferredEnd: boolean;
  /** The turn's window cut the span on this side: it began before the window or went on after it. */
  cutStart: boolean;
  cutEnd: boolean;
  /** A model request retry at or right before this span. Only on `wait` / `model` and `generate` spans. */
  retry: { attempt: number; max: number } | null;
}

export type TimelineSpan =
  | (TimelineSpanBase & { kind: 'generate'; phase: 'reasoning' | 'answer' })
  | (TimelineSpanBase & { kind: 'tool'; tools: string[]; calls: number; failedCalls: number })
  /** until: what ended the wait. 'open' = still waiting (running) or cut by the lane's end. */
  | (TimelineSpanBase & { kind: 'wait'; reason: 'model'; until: 'text' | 'tool' | 'open' })
  /** calls: spawn tool calls merged into it; 0 = built from spawn lines alone. */
  | (TimelineSpanBase & { kind: 'wait'; reason: 'subagent'; tools: string[]; calls: number; failedCalls: number })
  | (TimelineSpanBase & { kind: 'run'; noSteps: TimelineNoSteps })
  /** The main lane before a woken turn: no turn was open. At most one, the main lane's first span. */
  | (TimelineSpanBase & { kind: 'idle' });

export interface TimelineHandoff {
  /** `${lane}:${kind}:${run index}`. */
  id: string;
  /** The child lane. The other end is always 'main'. */
  lane: string;
  kind: 'spawn' | 'return';
  atMs: number;
  failed: boolean;
}

/** `GET /api/sessions/:id/timeline?turn=<n>`: one turn as lanes of spans. Metadata only, never text. */
export interface SessionTimelineResponse {
  turn: number;
  /** Finished turns in the transcript; an open turn is `turns + 1`. */
  turns: number;
  open: boolean;
  /** The session's live verdict when this body was built (the rule of SessionRow.live). */
  live: boolean;
  stopReason: string | null;
  /** Worker clock when this body was built, epoch ms. */
  asOfMs: number;
  /**
   * Epoch ms. null = the turn has no usable time; lanes, spans and handoffs are then empty.
   * endMs >= turnStartMs >= startMs always. leadMs: how far the prompt lies before the start the agent measured (0 = not
   * at all).
   */
  range: {
    /** Axis origin. Earlier than turnStartMs only when a background event started the turn and subagents ran before it. */
    startMs: number;
    /** Where the turn itself starts: the prompt or the start the agent measured, whichever is earlier. */
    turnStartMs: number;
    endMs: number | null;
    lastEventMs: number;
    /** Refers to turnStartMs. */
    startExact: boolean;
    leadMs: number;
  } | null;
  lanes: TimelineLane[];
  /** Ordered by lane order, then start. */
  spans: TimelineSpan[];
  handoffs: TimelineHandoff[];
  notes: { untimedEvents: number; foldedSubagents: number; droppedSpans: number; unplacedRetries: number };
  /** Same text as SessionDetailResponse.transcriptError; range is then null. */
  transcriptError?: string;
}

export type MetricsBucket = '5m' | 'hour' | 'day';
export type TurnOutcome = 'ok' | 'error' | 'cancelled';

/** A median below this many samples is drawn as low-confidence. */
export const LOW_SAMPLE_N = 5;
/** A p05 / p95 below this many samples is drawn as low-confidence. */
export const LOW_SAMPLE_TAIL_N = 20;

export interface Quantiles {
  n: number;
  p05: number | null;
  p50: number | null;
  p95: number | null;
}

export interface MetricsRange {
  from: number | null;
  to: number | null;
}

// GET /api/metrics/models  — no params. Whole retention window.
// Order: firstTs ascending, then model. A model's index here is its color index on every page.
export interface MetricsModels {
  models: Array<{
    model: string;
    modelType: string | null;
    firstTs: number;
    lastTs: number;
    turns: number;
  }>;
}

export interface MetricsTotals {
  turns: number;
  ok: number;
  error: number;
  cancelled: number;
  /** Ok turns that carry a measured prefill window. */
  prefillMeasured: number;
  /** Net of cache. */
  inputTokens: number;
  cachedTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  /** Wall-time sums in ms, for the time-breakdown bar. */
  queueWallMs: number;
  loadWaitWallMs: number;
  resolveWallMs: number;
  /** Time to first token of turns without a prefill window: setup and prefill together. */
  firstTokenWallMs: number;
  /** Time to first token outside the prefill window, measured turns only. */
  setupWallMs: number;
  /** Sum of measured prefill windows, each capped at its time to first token. */
  prefillWallMs: number;
  decodeWallMs: number;
  /** total − queue − loadWait − resolve − firstToken − setup − prefill − decode, clamped at 0. */
  otherWallMs: number;
  totalWallMs: number;
}

export interface MetricsLatency {
  ttftMs: Quantiles;
  /** Time to first token outside the prefill window, measured turns only. */
  setupMs: Quantiles;
  decodeTps: Quantiles;
  prefillTps: Quantiles;
  /** preDispatch + duration. */
  totalMs: Quantiles;
  queueMs: Quantiles;
  /** Percent of the context window in use at turn end. */
  ctxFillPct: Quantiles;
}

// GET /api/metrics/summary?from&to&model
export interface MetricsSummary {
  range: MetricsRange;
  model: string | null;
  totals: MetricsTotals;
  latency: MetricsLatency;
  /** The window of equal length just before `from`; null unless both bounds were given. */
  previous: { totals: MetricsTotals; latency: MetricsLatency } | null;
  byModel: Array<{
    model: string;
    modelType: string | null;
    ctxWindow: number | null;
    totals: MetricsTotals;
    latency: MetricsLatency;
    mtpTurns: number;
    lastTs: number;
  }>;
  finishReasons: Array<{ reason: string; turns: number }>;
  failures: Array<{
    outcome: 'error' | 'cancelled';
    errorKind: 'finish_error' | 'exception' | null;
    failPhase: 'queued' | 'prefill' | 'decode' | null;
    turns: number;
  }>;
  endpoints: Array<{ endpoint: string; stream: boolean; turns: number }>;
  dispatches: Array<{ dispatch: string; turns: number; promptTokens: number; cachedTokens: number }>;
}

// GET /api/metrics/timeseries?from&to&bucket&model&groupBy
export interface MetricsTimeseries {
  range: MetricsRange;
  bucket: MetricsBucket;
  groupBy: 'model' | 'none';
  points: Array<{
    /** Bucket start, epoch ms. */
    t: number;
    model: string;
    totals: MetricsTotals;
    ttftMs: Quantiles;
    decodeTps: Quantiles;
    prefillTps: Quantiles;
    totalMs: Quantiles;
    /** Cycle-weighted committed tokens per cycle; null when no MTP turn. */
    mtpMeanAccepted: number | null;
    mtpTurns: number;
    coldHits: number;
    coldMisses: number;
    concurrencyPeak: number | null;
  }>;
}

export interface Histogram {
  /** Samples counted. */
  n: number;
  /** One count per bin; `counts.length === edges.length`. */
  counts: number[];
}

// GET /api/metrics/distributions?from&to&model&groupBy
export interface MetricsDistributions {
  range: MetricsRange;
  groupBy: 'model' | 'none';
  /** Lower bounds. Bin i is [edges[i], edges[i+1]); the last bin is open-ended. */
  edges: {
    ttftMs: number[];
    decodeTps: number[];
    prefillTps: number[];
    totalMs: number[];
    ctxFillPct: number[];
    promptTokens: number[];
    outputTokens: number[];
  };
  series: Array<{
    model: string;
    ttftMs: Histogram;
    decodeTps: Histogram;
    prefillTps: Histogram;
    totalMs: Histogram;
    ctxFillPct: Histogram;
    promptTokens: Histogram;
    outputTokens: Histogram;
  }>;
}

// GET /api/metrics/speculative?from&to&model&bucket
export interface MetricsSpeculative {
  range: MetricsRange;
  models: Array<{
    model: string;
    turns: number;
    mtpTurns: number;
    cycles: number;
    /** Cycle-weighted. */
    meanAccepted: number | null;
    meanDrafts: number | null;
    meanDepth: number | null;
    /** `pos` is 0-based: pos 0 = the first draft position. */
    acceptByPos: Array<{ pos: number; rate: number; cycles: number }>;
    decodeTpsWithMtp: Quantiles;
    decodeTpsWithoutMtp: Quantiles;
  }>;
  trend: Array<{ t: number; model: string; meanAccepted: number | null; mtpTurns: number }>;
}

export interface TurnRow {
  traceId: string;
  /** End of turn, epoch ms. */
  ts: number;
  model: string;
  /** Alias the client used, when different from `model`. */
  alias: string | null;
  modelType: string | null;
  outcome: TurnOutcome;
  errorKind: string | null;
  failPhase: string | null;
  finishReason: string | null;
  endpoint: string | null;
  stream: boolean | null;
  dispatch: string | null;
  promptTokens: number | null;
  cachedTokens: number | null;
  outputTokens: number | null;
  reasoningTokens: number | null;
  ctxWindow: number | null;
  ctxFillPct: number | null;
  /** Span segments in ms; `t0 = ts − durationMs − (preDispatchMs ?? 0)`. */
  preDispatchMs: number | null;
  queueMs: number | null;
  loadWaitMs: number | null;
  resolveMs: number | null;
  ttftMs: number | null;
  /** Measured prefill window; null when the turn has none. */
  prefillMs: number | null;
  prefillTokens: number | null;
  /** `ttftMs − prefillMs`, measured turns only. */
  setupMs: number | null;
  decodeMs: number | null;
  durationMs: number | null;
  /** Rated turns only: measured, at least `PREFILL_MIN_TOKENS`. */
  prefillTps: number | null;
  decodeTps: number | null;
  mtpCycles: number | null;
  mtpMeanAccepted: number | null;
  coldHits: number | null;
  coldMisses: number | null;
  toolCount: number | null;
  toolCallCount: number | null;
  imageCount: number | null;
  concurrency: number | null;
  concurrencyPeak: number | null;
  memActiveBytes: number | null;
  pid: number | null;
}

// GET /api/metrics/turns?from&to&model&outcome&limit&beforeTs&beforeTraceId
// Newest first. limit default 200, max 2000.
// Cursor: WHERE (ts, trace_id) < (beforeTs, beforeTraceId).
// traceId=<id>: exact match on trace_id; every other param is ignored; `rows` holds 0 or 1 row, `next` is null,
// `total` is rows.length. Used by the turn drawer to open a turn that is not on the loaded page.
export interface MetricsTurns {
  range: MetricsRange;
  rows: TurnRow[];
  next: { beforeTs: number; beforeTraceId: string } | null;
  /** Rows matching the filter, ignoring the cursor and limit. */
  total: number;
}

export type HostKind = 'serve' | 'launch' | 'desktop' | 'embedded';
export type EngineStatus =
  | 'running'
  | 'loading'
  | 'starting'
  | 'restarting'
  | 'failed'
  | 'untrusted'
  | 'unresponsive'
  | 'stopped';

export interface RuntimeProcessLive {
  status: 'running' | 'unresponsive' | 'dead';
  /** now − ts. */
  ageMs: number;
  pid: number;
  startedAt: number;
  ts: number;
  host: HostKind;
  version: string;
  port: number;
  health: 'ok' | 'loading' | 'degraded' | 'error';
  mem: { active: number; peak: number; cache: number; wiredLimit: number; rss: number };
  decodeTps: number | null;
  loading: { model: string; startedAt: number } | null;
  counters: { ok: number; error: number; cancelled: number; outputTokens: number; loads: number; rejected: number };
  models: Array<{
    name: string;
    modelType: string;
    ctxWindow: number;
    paged: boolean;
    maxConcurrent: number;
    loadedAt: number | null;
    loadMs: number | null;
    inFlight: number;
    queued: number;
    queueLimit: number | null;
    pool: {
      at: number;
      blockCapacity: number;
      allocatedBlocks: number;
      freeBlocks: number;
      reclaimableBlocks: number;
    } | null;
  }>;
  turns: Array<{
    traceId: string;
    /** Alias the client used. Display and color use `modelId ?? model`. */
    model: string;
    modelId: string | null;
    ctxWindow: number | null;
    startedAt: number;
    stream: boolean;
    dispatch: 'start' | 'continue' | 'continue_tool';
    phase: 'queued' | 'prefill' | 'decode' | null;
    promptTokens: number | null;
    cachedTokens: number | null;
    prefillTotal: number | null;
    prefillDone: number | null;
    generatedTokens: number | null;
  }>;
}

export interface SupervisorLive {
  state: 'stopped' | 'starting' | 'running' | 'restarting' | 'failed' | 'lying';
  pid: number | null;
  generation: number;
  consecutiveCrashes: number;
  lastExit: { verdict: 'clean' | 'crash'; code: number | null; signal: string | null; atMs: number } | null;
  healthStatus: 'ok' | 'loading' | 'degraded' | 'error' | null;
  nativeErrorCount: number;
  /** The count is at the supervisor's cap of 20; print "20+". */
  nativeErrorsCapped: boolean;
  lastNativeErrorContext: string | null;
  lastNativeErrorAt: number | null;
  writtenAt: number;
}

// GET /api/runtime/live  — no params. Also the payload of the `runtime` live event.
export interface RuntimeLive {
  now: number;
  status: EngineStatus;
  /** Desktop-managed sidecar; null when not run by the desktop app. */
  supervisor: SupervisorLive | null;
  /** Running first, then unresponsive, then dead; newest `startedAt` first within each. */
  processes: RuntimeProcessLive[];
}

export interface SchedSnapshot {
  ts: number;
  blockCapacity: number;
  allocatedBlocks: number;
  freeBlocks: number;
  reclaimableBlocks: number;
  watermarkBlocks: number;
  reservedBlocks: number;
  memoryCapacityBytes: number;
  /** Largest decode batch since the model was loaded. A high-water mark, not a counter. */
  maxBatchOccupancy: number;
  /** Turns sent to the model since its load, all lanes; null when the engine did not write the count. */
  turns: number | null;
}

// GET /api/runtime/series?from&to&bucketMs
// bucketMs ∈ {5000, 15000, 60000, 300000, 900000, 3600000}; (to − from) / bucketMs ≤ 2000. Both bounds required.
export interface RuntimeSeries {
  range: { from: number; to: number };
  bucketMs: number;
  points: Array<{
    t: number;
    samples: number;
    memActiveAvg: number | null;
    memActiveMax: number | null;
    memPeakMax: number | null;
    memCacheAvg: number | null;
    rssMax: number | null;
    inFlightAvg: number | null;
    inFlightMax: number | null;
    queuedMax: number | null;
    /** Sum of the samples' measured `busyMs`. */
    busyMs: number;
    okTurns: number;
    errorTurns: number;
    cancelledTurns: number;
    outTokens: number;
    rejected: number;
  }>;
  procs: Array<{
    procKey: string;
    pid: number;
    startedAt: number;
    host: string | null;
    version: string | null;
    port: number | null;
    chip: string | null;
    wiredLimitBytes: number | null;
    totalMemBytes: number | null;
    firstTs: number;
    lastTs: number;
    exitTs: number | null;
  }>;
  loads: Array<{
    ts: number;
    procKey: string;
    model: string;
    modelType: string | null;
    capability: string | null;
    ms: number | null;
    ok: boolean;
  }>;
  /** One entry per (process, model) with at least one snapshot in or before the range. */
  sched: Array<{
    procKey: string;
    model: string;
    latest: SchedSnapshot;
    /** Pool at rest over time, snapshots inside the range. */
    snapshots: SchedSnapshot[];
    /** Growth of monotonic counters inside the range. */
    delta: {
      globalSteps: number;
      admitted: number;
      completed: number;
      preemptions: number;
      preemptionsRecompute: number;
      preemptionsSsd: number;
      admissionDeferredBlocks: number;
      admissionDeferredState: number;
      rowsAllocEvicted: number;
      ssdRestoreBytes: number;
      ssdRestoreWaitMs: number;
      /** Turns sent to the model, all lanes. 0 when the engine did not write the count. */
      turns: number;
      /** Decode steps per batch size. */
      occupancy: Array<{ occupancy: number; steps: number }>;
    };
  }>;
}

// Push events
export type LiveEvent =
  | { type: 'runtime'; snapshot: RuntimeLive }
  | { type: 'data'; version: number; traces: boolean; runtime: boolean };

/** `/api/health`: the directories this Control Panel reads. */
export interface HealthResponse {
  status: string;
  modelsDir: string;
  tracesDir: string;
  runtimeDir: string;
  /** The user's home folder: paths under it are shown as `~/…`. */
  homeDir: string;
}
