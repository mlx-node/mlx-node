/**
 * JSON + SSE API route handlers for the dashboard, over plain `node:http`.
 *
 * A tiny `(method, pathname)` matcher with `:param` segments dispatches to the
 * handlers below; there is no web framework. All data comes from the C1–C4
 * modules (SQLite index, local model store, cold-tier scan, download runner)
 * and pi's `SessionManager` for transcripts/rename — never the native addon.
 */

import { readFileSync, realpathSync, rmSync, statSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { basename, dirname, join, sep } from 'node:path';

import {
  SessionManager,
  parseSessionEntries,
  type FileEntry,
  type SessionEntry,
} from '@earendil-works/pi-coding-agent';
import { canonicalCacheRoot, coldTierRestoreFamilyList } from '@mlx-node/agent/catalog';
import { eq } from 'drizzle-orm';

import { scanColdCache, clearColdCache, coldCacheRoot, evictOlderThan } from './cache.js';
import { catalogWithState } from './catalog.js';
import type { DashboardDb } from './db/open.js';
import { sessions, turns } from './db/schema.js';
import { DownloadsClosedError, type DownloadManager, type DownloadEvent } from './download.js';
import {
  activeBranchEntries,
  classifySessionFile,
  countJsonlLines,
  findSessionHeader,
  isValidSessionTopology,
  lastLineParses,
  readSessionEntries,
  verifySessionFileId,
} from './ingest/sessions.js';
import { TRACE_RETENTION_DAYS } from './ingest/traces.js';
import { discoverLocalModels, deleteLocalModel } from './models.js';

/** Runtime dependencies the handlers close over, supplied by `server.ts`. */
export interface ApiDeps {
  dash: DashboardDb;
  modelsDir: string;
  sessionsRoot: string;
  tracesDir: string;
  /** Cold-tier root; `undefined` lets `cache.ts` resolve the running-tier default. */
  cacheRoot: string | undefined;
  downloads: DownloadManager;
  /** Serialized incremental rescan (sessions + traces). */
  runIngest: () => Promise<IngestSummary>;
  /** Live SSE connections, tracked so `close()` can end them. */
  sseClients: Set<SseClient>;
}

export interface IngestSummary {
  sessions: { scanned: number; updated: number; removed: number; warnings: string[] };
  traces: { files: number; records: number; pruned: number; warnings: string[] };
}

export interface SseClient {
  res: ServerResponse;
  cleanup: () => void;
}

/** One transcript message projected from a pi session entry. */
export interface TranscriptEntry {
  role: string;
  text: string;
  /** `summary` is a one-line digest of the call's arguments (path / command / …). */
  toolCalls: Array<{ id: string; name: string; arguments: unknown; summary: string }>;
  ts: number;
  /** Present for `toolResult` messages. */
  toolName?: string;
  isError?: boolean;
  /** The model that produced an `assistant` message (verbatim id), for its logo/name. */
  model?: string;
  /**
   * One-line digest of the ORIGINATING tool call's arguments (its `path` /
   * `command` / …), joined by `toolCallId`. Lets a collapsed result row show what
   * it acted on without expanding. Present only for `toolResult` messages.
   */
  title?: string;
  /** Decoded image blocks (base64), rendered inline as thumbnails by the UI. */
  images?: Array<{ mimeType: string; data: string }>;
  /**
   * Short labels (e.g. `HEIC · 32 KB`) for binary blobs a tool returned verbatim
   * as text, or images too large to inline. The UI shows a chip instead of the
   * raw bytes.
   */
  binaryNotes?: string[];
  /**
   * Model-facing image-read notes (`[Image: original …, displayed at …. Multiply
   * coordinates by … …]`) split out of {@link text}. Pure coordinate-mapping
   * plumbing next to a rendered thumbnail, so the UI hides these by default.
   */
  imageNotes?: string[];
}

/**
 * Shape of `GET /api/metrics/overview`. Documented here because the C10 UI is
 * the sole consumer. All token/count fields are non-negative integers; average
 * fields are `number | null` (null when no sample carried that column).
 */
export interface MetricsOverview {
  range: { from: number | null; to: number | null };
  tokensByDay: Array<{ day: string; input: number; output: number; cached: number; reasoning: number }>;
  throughputByModel: Array<{
    model: string;
    avgDecodeTps: number | null;
    avgPrefillTps: number | null;
    avgTtftMs: number | null;
    samples: number;
  }>;
  /**
   * Day-bucketed throughput/TTFT trend per model — the time series the spec
   * promises alongside the range-wide `throughputByModel` averages. Buckets share
   * `tokensByDay`'s `date(ts/1000,'unixepoch')` expression so the UI can align
   * them. Numeric averages are coerced to a number (0 when the bucket carried no
   * sample for that column).
   */
  throughputTrend: Array<{
    model: string;
    day: string;
    decodeTps: number | null;
    prefillTps: number | null;
    ttftMs: number | null;
    samples: number;
  }>;
  mtpByModel: Array<{ model: string; meanAccepted: number | null; avgCycles: number | null; samples: number }>;
  modelShare: Array<{ model: string; turns: number; outputTokens: number }>;
  totals: {
    turns: number;
    traces: number;
    inputTokens: number;
    outputTokens: number;
    cachedTokens: number;
    reasoningTokens: number;
  };
}

const MAX_BODY_BYTES = 1024 * 1024;

/**
 * A session file modified within this window may be actively written by a live
 * agent turn (a SEPARATE process — pi has no cross-process lock). The rename writes
 * the durable name into the pi session JSONL: `SessionManager.open` snapshots the
 * current leaf and `appendSessionInfo` parents the new entry to it, so a turn
 * appended by a concurrent agent between our snapshot and append becomes a sibling
 * of the rename — on the next resume one of them is orphaned (the turn is lost, or
 * the rename silently vanishes). We refuse the rename while the file looks active
 * (mtime inside this window) and only proceed for an idle session.
 *
 * This is a best-effort PRODUCT rule, not a hard guarantee: a session idle past this
 * window that then goes live, or one that goes live within the stat→append window,
 * can still race. It removes the realistic reachability, not the theoretical race.
 * Storing the name index-only is not an alternative — it would break the disposable-
 * index invariant (the rename would be lost when `dashboard.db` is rebuilt from the
 * JSONL source of truth), so the durable JSONL write is the only correct home.
 */
const LIVE_SESSION_WINDOW_MS = 30_000;

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-cache' });
  res.end(payload);
}

function sendError(res: ServerResponse, status: number, message: string): void {
  sendJson(res, status, { error: message });
}

/** Read a JSON request body, capped at 1 MB. Empty body resolves to `null`. */
function readJsonBody(req: IncomingMessage): Promise<unknown> {
  return new Promise((resolveBody, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    req.on('data', (chunk: Buffer) => {
      total += chunk.length;
      if (total > MAX_BODY_BYTES) {
        reject(new Error('Request body too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf-8').trim();
      if (raw === '') {
        resolveBody(null);
        return;
      }
      try {
        resolveBody(JSON.parse(raw));
      } catch {
        reject(new Error('Invalid JSON in request body'));
      }
    });
    req.on('error', reject);
  });
}

function toNum(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'bigint') return Number(value);
  return null;
}

function toInt(value: unknown): number {
  const n = toNum(value);
  return n === null ? 0 : Math.trunc(n);
}

/** Positive-integer query param, or `null`. */
function queryInt(url: URL, name: string): number | null {
  const raw = url.searchParams.get(name);
  if (raw === null || raw === '') return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

// --- Route table ------------------------------------------------------------

type Handler = (ctx: RouteCtx) => void | Promise<void>;

interface RouteCtx {
  req: IncomingMessage;
  res: ServerResponse;
  url: URL;
  params: Record<string, string>;
  deps: ApiDeps;
}

interface Route {
  method: string;
  segments: string[];
  handler: Handler;
}

function route(method: string, pattern: string, handler: Handler): Route {
  return { method, segments: pattern.split('/').filter((s) => s.length > 0), handler };
}

/** Match a `:param` route against concrete path segments. */
function matchRoute(route: Route, method: string, segments: string[]): Record<string, string> | null {
  if (route.method !== method) return null;
  if (route.segments.length !== segments.length) return null;
  const params: Record<string, string> = {};
  for (let i = 0; i < route.segments.length; i++) {
    const pat = route.segments[i];
    if (pat.startsWith(':')) {
      params[pat.slice(1)] = decodeURIComponent(segments[i]);
    } else if (pat !== segments[i]) {
      return null;
    }
  }
  return params;
}

// --- Handlers ---------------------------------------------------------------

function handleHealth({ res, deps }: RouteCtx): void {
  sendJson(res, 200, {
    status: 'ok',
    modelsDir: deps.modelsDir,
    sessionsRoot: deps.sessionsRoot,
    tracesDir: deps.tracesDir,
  });
}

function handleModels({ res, deps }: RouteCtx): void {
  const { models, warnings } = discoverLocalModels(deps.modelsDir);
  // `dir` lets the UI show WHERE these checkpoints live — the directory is
  // configurable (`--models-dir`), so the count alone is ambiguous.
  sendJson(res, 200, { models, warnings, dir: deps.modelsDir });
}

function handleDeleteModel({ res, params, deps }: RouteCtx): void {
  try {
    deleteLocalModel(deps.modelsDir, params.name);
    sendJson(res, 200, { deleted: true, name: params.name });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    sendError(res, /not found/i.test(message) ? 404 : 400, message);
  }
}

function handleCatalog({ res, deps }: RouteCtx): void {
  sendJson(res, 200, { items: catalogWithState(deps.modelsDir) });
}

function handleDownloadsList({ res, deps }: RouteCtx): void {
  sendJson(res, 200, { jobs: deps.downloads.jobs() });
}

async function handleDownloadStart({ req, res, deps }: RouteCtx): Promise<void> {
  let body: unknown;
  try {
    body = await readJsonBody(req);
  } catch (err) {
    sendError(res, 400, err instanceof Error ? err.message : 'Invalid request body');
    return;
  }
  const repo = (body as { repo?: unknown } | null)?.repo;
  if (typeof repo !== 'string' || repo === '') {
    sendError(res, 400, 'Body must include a "repo" string');
    return;
  }
  try {
    const id = deps.downloads.start(repo);
    sendJson(res, 202, { id, repo });
  } catch (err) {
    // The manager is draining for shutdown: the request is well-formed, the
    // server is simply going away, so answer 503 rather than blaming the caller.
    if (err instanceof DownloadsClosedError) {
      sendError(res, 503, err.message);
      return;
    }
    sendError(res, 400, err instanceof Error ? err.message : 'Failed to start download');
  }
}

/**
 * Cancel/abort or dismiss a download. A `running` job is aborted (its
 * job-private staging dir cleans up); a terminal job (`done`/`error`/`cancelled`)
 * is dismissed from the registry. Both cases return 200. This deliberately never
 * touches the shared HF blob cache (the `mlx download` CLI resumes from it). A
 * 404 means no such id, or the job is in its brief non-cancellable `committing`
 * (publish) window.
 */
function handleDownloadCancel({ res, params, deps }: RouteCtx): void {
  if (deps.downloads.cancel(params.id)) {
    sendJson(res, 200, { cancelled: true, id: params.id });
  } else {
    sendError(res, 404, `No cancellable download "${params.id}"`);
  }
}

/** Minimal writable surface a backpressure-aware SSE sender depends on. */
interface SseWritable {
  write(chunk: string): boolean;
  on(event: 'drain', listener: () => void): void;
}

function sseFrame(event: DownloadEvent): string {
  return `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
}

/**
 * Backpressure-aware SSE sender for download progress. `res.write` returning
 * false means the socket buffer is full; further frames are queued in memory
 * until `'drain'`. A subscriber that never reads during a multi-GB download
 * would otherwise accumulate one buffered frame per progress tick without
 * bound, so a run of consecutive `progress` frames is coalesced to the latest.
 * Lifecycle frames (`start`/`done`/`error`) are kept in order so terminal
 * delivery is never dropped, bounding the queue to at most one frame between
 * lifecycle events.
 */
export function createDownloadSseSender(res: SseWritable): (event: DownloadEvent) => void {
  let blocked = false;
  const pending: DownloadEvent[] = [];

  // A `false` return from `write` means the chunk was BUFFERED, not refused — so
  // a frame is dequeued before its result is read, exactly as the unblocked path
  // below never re-queues one. Keeping the head instead would rewrite an
  // already-sent frame on every drain, and for a frame larger than the socket
  // high-water mark (an unbounded remote `error.message` from the hub) every
  // rewrite returns false again, replaying it without bound.
  const flush = (): void => {
    blocked = false;
    while (pending.length > 0) {
      const event = pending.shift()!;
      if (!res.write(sseFrame(event))) {
        blocked = true;
        return;
      }
    }
  };
  res.on('drain', flush);

  return (event: DownloadEvent): void => {
    if (blocked) {
      const last = pending[pending.length - 1];
      if (event.type === 'progress' && last !== undefined && last.type === 'progress') {
        pending[pending.length - 1] = event;
      } else {
        pending.push(event);
      }
      return;
    }
    if (!res.write(sseFrame(event))) blocked = true;
  };
}

function handleDownloadEvents({ req, res, params, deps }: RouteCtx): void {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });
  res.write(': connected\n\n');

  const send = createDownloadSseSender(res);
  const unsubscribe = deps.downloads.subscribe(params.id, send);

  const heartbeat = setInterval(() => res.write(': ping\n\n'), 15_000);
  heartbeat.unref();

  const client: SseClient = {
    res,
    cleanup: () => {
      clearInterval(heartbeat);
      unsubscribe();
    },
  };
  deps.sseClients.add(client);

  req.on('close', () => {
    client.cleanup();
    deps.sseClients.delete(client);
  });
}

function handleSessionsList({ res, url, deps }: RouteCtx): void {
  const q = url.searchParams.get('q');
  const cwd = url.searchParams.get('cwd');
  const model = url.searchParams.get('model');
  const from = queryInt(url, 'from');
  const to = queryInt(url, 'to');
  // Page the (possibly large) list. `limit` defaults to 500 and is clamped to
  // [1, 500] so an unbounded/absurd value can't be forced; `offset` clamps to >= 0.
  const limit = Math.min(Math.max(queryInt(url, 'limit') ?? 500, 1), 500);
  const offset = Math.max(queryInt(url, 'offset') ?? 0, 0);

  const where: string[] = [];
  const args: Array<string | number> = [];
  if (q !== null && q !== '') {
    where.push('(s.name LIKE ? OR s.first_message LIKE ?)');
    args.push(`%${q}%`, `%${q}%`);
  }
  if (cwd !== null && cwd !== '') {
    where.push('s.cwd = ?');
    args.push(cwd);
  }
  if (model !== null && model !== '') {
    where.push('EXISTS (SELECT 1 FROM turns t WHERE t.session_id = s.id AND t.model = ?)');
    args.push(model);
  }
  if (from !== null) {
    where.push('s.modified >= ?');
    args.push(from);
  }
  if (to !== null) {
    where.push('s.modified <= ?');
    args.push(to);
  }

  const whereSql = where.length > 0 ? `WHERE ${where.join(' AND ')}` : '';
  const matchedIds = `SELECT s.id FROM sessions s ${whereSql}`;
  const sql = `
    SELECT s.id, s.path, s.cwd, s.name, s.created, s.modified,
           s.message_count AS messageCount, s.first_message AS firstMessage,
           (SELECT group_concat(DISTINCT t.model) FROM turns t
              WHERE t.session_id = s.id AND t.model IS NOT NULL) AS models,
           (SELECT COALESCE(SUM(t.input_tokens), 0) FROM turns t WHERE t.session_id = s.id) AS inputTokens,
           (SELECT COALESCE(SUM(t.output_tokens), 0) FROM turns t WHERE t.session_id = s.id) AS outputTokens
    FROM sessions s
    ${whereSql}
    ORDER BY s.modified DESC
    LIMIT ? OFFSET ?`;

  // A separate COUNT over the SAME filter so the client (e.g. the Overview tile)
  // reports the true match total rather than the capped page length.
  const total = toInt(
    (deps.dash.sqlite.prepare(`SELECT COUNT(*) AS n FROM sessions s ${whereSql}`).get(...args) as { n: unknown }).n,
  );

  // Tokens over THOSE SAME sessions, from the SAME filter.
  //
  // This exists so a caller that shows both numbers cannot pair them with a
  // different scope. `sessions`/`turns` hold only what lives under the CURRENT
  // `--session-dir` (ingest deletes rows whose file left the root), while
  // `traces` is a machine-wide log from `~/.mlx-node/metrics/traces` and
  // `/metrics/overview` sums it whole. Reading a count from here and a token
  // total from there produces a tile whose halves describe different session
  // sets — the same defect the Cache page's root scoping fixed, where a
  // filesystem scan of one cache root sat beside a hit rate summed over every
  // root the machine ever used.
  //
  // This counts each DISTINCT inference in the matched set once — the same rule
  // `/metrics/overview` applies — and is deliberately NOT the sum of the
  // per-session `/api/sessions/:id/metrics` totals. The first arm collapses
  // `turns` on their canonical identity across the WHOLE matched set, so a
  // forked session (which copies inherited turns VERBATIM, same `trace_id` and
  // `entry_id`) contributes its inherited history once, whereas each per-session
  // call reports every copy to the session holding it. Adding those per-session
  // totals up therefore exceeds this number by the shared history, and so does
  // adding up the per-row `inputTokens`/`outputTokens` columns below, which stay
  // plain per-session sums with no dedup and no trace arm precisely because a
  // session row must show what THAT session spent. Both readings are wanted; the
  // headline total is the one that must not bill a fork twice.
  //
  // The second arm adds subagent work, which runs on an in-memory session manager
  // and therefore writes a `traces` row with no `turns` row at all. It is
  // admitted only via `root_session_id` pointing INTO the matched set, and only
  // when no `turns` row already carries that `trace_id`; the inner
  // `trace_id IS NOT NULL` keeps `NOT IN` from going NULL and dropping the whole
  // set. That dedup subquery is unscoped by session on purpose (the per-session
  // handler scopes its own to `session_id = ?`): a trace some session already
  // accounts for must never be added again to a set-wide total. Trace rows store
  // GROSS `prompt_tokens`, so the input is clamped to the producer's net
  // `MAX(prompt - cached, 0)` rather than double-counting cache reads against the
  // already-net turns side.
  const tokenTotals = deps.dash.sqlite
    .prepare(
      `SELECT COALESCE(SUM(inputTokens), 0) AS inputTokens, COALESCE(SUM(outputTokens), 0) AS outputTokens
       FROM (SELECT COALESCE(input_tokens, 0) AS inputTokens, COALESCE(output_tokens, 0) AS outputTokens
             FROM turns WHERE session_id IN (${matchedIds})
             GROUP BY COALESCE(trace_id, entry_id, CAST(id AS TEXT))
             UNION ALL
             SELECT MAX(COALESCE(prompt_tokens, 0) - COALESCE(cached_tokens, 0), 0),
                    COALESCE(output_tokens, 0)
             FROM traces
             WHERE root_session_id IN (${matchedIds}) AND session_id != root_session_id
               AND trace_id NOT IN (SELECT trace_id FROM turns WHERE trace_id IS NOT NULL))`,
    )
    .get(...args, ...args) as { inputTokens: unknown; outputTokens: unknown } | undefined;
  const tokens = toInt(tokenTotals?.inputTokens) + toInt(tokenTotals?.outputTokens);

  // Every directory the index holds — deliberately UNFILTERED, and deliberately
  // served rather than left to the client.
  //
  // The Sessions page builds its directory dropdown from this. Deriving it from
  // the rows instead cannot work: the page above is capped at `limit`, so a
  // directory whose sessions are all older than the cap has no row to be read
  // off, and it silently disappears from the filter — at exactly the moment the
  // page's own footnote ("narrow with the directory filter to reach older ones")
  // sends the user there, since that footnote only appears once `total` exceeds
  // the page. The same defect `tokens` above exists to prevent: a number the
  // client cannot compute from what it was handed must come from the server.
  //
  // Unfiltered because the dropdown is the way OUT of a filter as well as in.
  // Scoping it to the current `where` would leave a chosen directory as the only
  // option it lists, so the user could never switch to another one.
  const cwds = (
    deps.dash.sqlite.prepare('SELECT DISTINCT cwd FROM sessions ORDER BY cwd').all() as Array<{
      cwd: unknown;
    }>
  ).map((row) => String(row.cwd));

  const rows = deps.dash.sqlite.prepare(sql).all(...args, limit, offset);
  const list = rows.map((row) => ({
    id: String(row.id),
    path: String(row.path),
    cwd: String(row.cwd),
    name: row.name === null ? null : String(row.name),
    created: toInt(row.created),
    modified: toInt(row.modified),
    messageCount: toInt(row.messageCount),
    firstMessage: row.firstMessage === null ? null : String(row.firstMessage),
    models: typeof row.models === 'string' && row.models !== '' ? row.models.split(',') : [],
    inputTokens: toInt(row.inputTokens),
    outputTokens: toInt(row.outputTokens),
  }));
  sendJson(res, 200, { sessions: list, total, tokens, cwds });
}

function lookupSession(deps: ApiDeps, id: string): { path: string; row: typeof sessions.$inferSelect } | null {
  const rows = deps.dash.db.select().from(sessions).where(eq(sessions.id, id)).all();
  if (rows.length === 0) return null;
  return { path: rows[0].path, row: rows[0] };
}

/**
 * Guard: the CANONICAL session file must stay inside the canonical sessions
 * root. Both sides are resolved through `realpathSync` so a symlink at any
 * component can't escape a purely lexical containment check. When the target
 * itself does not exist (e.g. its file was already removed — the delete path
 * still cleans up its stale rows), a missing path has no symlink to follow, so
 * its existing parent is canonicalized and the final segment re-attached
 * lexically. A root that cannot be canonicalized, or a target whose parent is
 * also gone, is treated as outside (fail closed).
 */
function insideSessionsRoot(sessionsRoot: string, path: string): boolean {
  const contained = (real: string, root: string): boolean => real === root || real.startsWith(root + sep);
  let root: string;
  try {
    root = realpathSync(sessionsRoot);
  } catch {
    return false;
  }
  try {
    return contained(realpathSync(path), root);
  } catch {
    try {
      return contained(join(realpathSync(dirname(path)), basename(path)), root);
    } catch {
      return false;
    }
  }
}

/** A base64 image bigger than this (~1.5 MB decoded) is chipped, not inlined. */
const MAX_INLINE_IMAGE_B64 = 2_000_000;
/** Cap inlined images per message so one turn can't bloat the transcript payload. */
const MAX_INLINE_IMAGES = 8;

/**
 * Whether a string is raw binary read verbatim as "text" — a `read` of a `.heic`,
 * PDF, etc. returns the file bytes in a `text` block, and control bytes (NUL and
 * friends, excluding tab/newline/CR) never occur in real transcript prose. Such a
 * block must not be dumped into the rendered text; it becomes a chip instead.
 */
function isBinaryText(text: string): boolean {
  // Scan a bounded prefix (enough to classify; cheap for a multi-MB blob) for any
  // C0 control byte other than tab (9), newline (10), or carriage return (13).
  const limit = Math.min(text.length, 4096);
  for (let i = 0; i < limit; i++) {
    const code = text.charCodeAt(i);
    if (code <= 8 || code === 11 || code === 12 || (code >= 14 && code <= 31)) return true;
  }
  return false;
}

/** Human byte size for a chip label, from a raw byte count. */
function sizeLabel(bytes: number): string {
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}

/** A short `HEIC · 32 KB`-style label for a binary blob, sniffing common magics. */
function describeBinary(text: string): string {
  let format = 'binary';
  if (text.startsWith('%PDF')) format = 'PDF';
  else if (/^.{0,16}ftyp(heic|heix|hevc|mif1|msf1)/.test(text)) format = 'HEIC';
  else if (text.startsWith('\u0089PNG')) format = 'PNG';
  else if (text.startsWith('\u00ff\u00d8\u00ff')) format = 'JPEG';
  else if (text.startsWith('GIF8')) format = 'GIF';
  else if (text.startsWith('PK')) format = 'ZIP';
  return `${format} · ${sizeLabel(text.length)}`;
}

/**
 * Rendered prose for a message. Skips two non-prose payloads a tool can put in a
 * `text` block: raw binary read verbatim (see {@link isBinaryText}), which would
 * otherwise dump kilobytes of garbage. Those are surfaced via
 * {@link extractBinaryNotes} as chips instead.
 */
function extractText(content: unknown): string {
  if (typeof content === 'string') return isBinaryText(content) ? '' : content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (block && typeof block === 'object' && (block as { type?: unknown }).type === 'text') {
      const text = (block as { text?: unknown }).text;
      if (typeof text === 'string' && !isBinaryText(text)) parts.push(text);
    }
  }
  return parts.join('\n');
}

/**
 * A model-facing image-read note: a whole line reading
 * `[Image: original …, displayed at …. Multiply coordinates by … to map …]`.
 * Anchored to the bracket delimiters and one of the tool's fixed clauses so it
 * only ever matches the generated note, never human prose that mentions images.
 */
const IMAGE_DETAIL_RE = /^\s*\[Image:[^\]]*(?:displayed at|Multiply coordinates by)[^\]]*\]\s*$/;

/**
 * Split the coordinate-mapping image note(s) out of a message's prose. The note
 * is plumbing for the model sitting right next to a rendered thumbnail, so the
 * UI keeps it collapsed; the remaining prose is returned as {@link visible}.
 */
function partitionImageDetails(text: string): { visible: string; notes: string[] } {
  if (text === '' || !text.includes('[Image:')) return { visible: text, notes: [] };
  const notes: string[] = [];
  const kept: string[] = [];
  for (const line of text.split('\n')) {
    if (IMAGE_DETAIL_RE.test(line)) notes.push(line.trim());
    else kept.push(line);
  }
  if (notes.length === 0) return { visible: text, notes };
  return {
    visible: kept
      .join('\n')
      .replace(/\n{3,}/g, '\n\n')
      .trim(),
    notes,
  };
}

/** Inline-able image blocks (`{type:'image', data, mimeType}`), capped. */
function extractImages(content: unknown): Array<{ mimeType: string; data: string }> {
  if (!Array.isArray(content)) return [];
  const images: Array<{ mimeType: string; data: string }> = [];
  for (const block of content) {
    if (images.length >= MAX_INLINE_IMAGES) break;
    if (!block || typeof block !== 'object' || (block as { type?: unknown }).type !== 'image') continue;
    const { data, mimeType } = block as { data?: unknown; mimeType?: unknown };
    if (typeof data === 'string' && data.length <= MAX_INLINE_IMAGE_B64) {
      images.push({ mimeType: typeof mimeType === 'string' ? mimeType : 'image/png', data });
    }
  }
  return images;
}

/**
 * Chip labels for binary payloads NOT rendered as prose or thumbnails: raw binary
 * read verbatim into a `text` block, and images too large to inline.
 */
function extractBinaryNotes(content: unknown): string[] {
  const notes: string[] = [];
  const consider = (block: unknown): void => {
    if (typeof block === 'string') {
      if (isBinaryText(block)) notes.push(describeBinary(block));
      return;
    }
    if (!block || typeof block !== 'object') return;
    const b = block as { type?: unknown; text?: unknown; data?: unknown };
    if (b.type === 'text' && typeof b.text === 'string' && isBinaryText(b.text)) {
      notes.push(describeBinary(b.text));
    } else if (b.type === 'image' && typeof b.data === 'string' && b.data.length > MAX_INLINE_IMAGE_B64) {
      // base64 decodes to ~3/4 its length.
      notes.push(`image · ${sizeLabel(Math.floor((b.data.length * 3) / 4))} (too large to preview)`);
    }
  };
  if (typeof content === 'string') consider(content);
  else if (Array.isArray(content)) for (const block of content) consider(block);
  return notes;
}

/**
 * A one-line digest of a tool call's arguments — the `path` it read/edited, the
 * `command` bash ran, the `agent` a subagent spawned — so a collapsed row can show
 * what it did. Whitespace is flattened and the result capped to bound the payload;
 * the UI truncates the rest to the row width.
 */
function summarizeToolArgs(args: unknown): string {
  const pick = (): string => {
    if (typeof args === 'string') return args;
    if (args === null || typeof args !== 'object') return '';
    const o = args as Record<string, unknown>;
    // Salient keys first (path covers read/ls/edit/write; command bash; agent
    // subagent), then any first string field as a last resort.
    const keys = [
      'command',
      'path',
      'file_path',
      'filePath',
      'file',
      'pattern',
      'glob',
      'query',
      'url',
      'agent',
      'task',
      'description',
      'name',
    ];
    for (const k of keys) {
      const v = o[k];
      if (typeof v === 'string' && v.trim() !== '') return v;
    }
    for (const v of Object.values(o)) {
      if (typeof v === 'string' && v.trim() !== '') return v;
    }
    return '';
  };
  return pick().replace(/\s+/g, ' ').trim().slice(0, 300);
}

function extractToolCalls(content: unknown): TranscriptEntry['toolCalls'] {
  if (!Array.isArray(content)) return [];
  const calls: TranscriptEntry['toolCalls'] = [];
  for (const block of content) {
    if (block && typeof block === 'object' && (block as { type?: unknown }).type === 'toolCall') {
      const call = block as { id?: unknown; name?: unknown; arguments?: unknown };
      calls.push({
        id: typeof call.id === 'string' ? call.id : '',
        name: typeof call.name === 'string' ? call.name : '',
        arguments: call.arguments ?? null,
        summary: summarizeToolArgs(call.arguments),
      });
    }
  }
  return calls;
}

/** Map every tool call's `id` → its raw arguments, for joining results back. */
function collectCallArgs(entries: SessionEntry[]): Map<string, unknown> {
  const map = new Map<string, unknown>();
  for (const entry of entries) {
    if (entry.type !== 'message') continue;
    const content = (entry.message as { content?: unknown }).content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (block && typeof block === 'object' && (block as { type?: unknown }).type === 'toolCall') {
        const call = block as { id?: unknown; arguments?: unknown };
        if (typeof call.id === 'string') map.set(call.id, call.arguments);
      }
    }
  }
  return map;
}

function mapTranscriptEntry(entry: SessionEntry, callArgs: Map<string, unknown>): TranscriptEntry | null {
  if (entry.type !== 'message') return null;
  const msg = entry.message as {
    role?: unknown;
    content?: unknown;
    timestamp?: unknown;
    toolName?: unknown;
    isError?: unknown;
    toolCallId?: unknown;
    model?: unknown;
  };
  const role = typeof msg.role === 'string' ? msg.role : 'unknown';
  const ts =
    typeof entry.timestamp === 'string' && !Number.isNaN(Date.parse(entry.timestamp))
      ? Date.parse(entry.timestamp)
      : typeof msg.timestamp === 'number'
        ? msg.timestamp
        : 0;
  const { visible, notes: imageNotes } = partitionImageDetails(extractText(msg.content));
  const mapped: TranscriptEntry = {
    role,
    text: visible,
    toolCalls: extractToolCalls(msg.content),
    ts,
  };
  if (role === 'toolResult') {
    if (typeof msg.toolName === 'string') mapped.toolName = msg.toolName;
    if (typeof msg.isError === 'boolean') mapped.isError = msg.isError;
    // Recover what the call acted on (path/command) by joining `toolCallId` back
    // to its originating tool call's arguments.
    if (typeof msg.toolCallId === 'string' && callArgs.has(msg.toolCallId)) {
      const title = summarizeToolArgs(callArgs.get(msg.toolCallId));
      if (title !== '') mapped.title = title;
    }
  }
  if (role === 'assistant' && typeof msg.model === 'string' && msg.model !== '') {
    mapped.model = msg.model;
  }
  const images = extractImages(msg.content);
  if (images.length > 0) mapped.images = images;
  const binaryNotes = extractBinaryNotes(msg.content);
  if (binaryNotes.length > 0) mapped.binaryNotes = binaryNotes;
  if (imageNotes.length > 0) mapped.imageNotes = imageNotes;
  return mapped;
}

async function handleSessionDetail({ res, params, deps }: RouteCtx): Promise<void> {
  const found = lookupSession(deps, params.id);
  if (found === null) {
    sendError(res, 404, `Session "${params.id}" not found`);
    return;
  }
  // A GET must not disclose a transcript whose file resolves outside the managed
  // root (a stale/symlinked row). Ingestion no longer indexes symlinked files,
  // but a row predating that fix, or a path swapped to a symlink after indexing,
  // is caught here — the same guard PATCH/DELETE already apply.
  if (!insideSessionsRoot(deps.sessionsRoot, found.path)) {
    sendError(res, 403, 'Session file resolves outside the managed sessions root');
    return;
  }
  const { row } = found;
  let entries: FileEntry[] | null = null;
  let transcript: TranscriptEntry[] = [];
  let transcriptError: string | undefined;
  try {
    // Read-only, byte-for-byte the way ingest reads a session (parse + in-memory
    // v1→v3 migrate, never a rewrite). `SessionManager.open` opens the file for
    // write and migrates on construction, so a plain GET of a v1 or partially
    // corrupt session would persist the migration and permanently drop malformed
    // lines — a read must never mutate the source of truth.
    entries = readSessionEntries(row.path);
  } catch (err) {
    transcriptError = err instanceof Error ? err.message : String(err);
  }
  if (entries !== null) {
    // The indexed path may resolve (in-root) to a DIFFERENT session than this row
    // — its file swapped for another transcript, or for an in-root symlink to one.
    // Containment alone can't catch that (the target is still in-root), so require
    // the parsed header id to still be THIS row before serving its metadata with
    // that file's transcript. On mismatch, reconcile the stale row and refuse.
    //
    // Through `findSessionHeader`, never a raw `.type` filter: `readSessionEntries`
    // returns the array UNMIGRATED once any record is a non-object, and a bare
    // `null` line is a valid JSONL record that `parseSessionEntries` keeps
    // verbatim. Dereferencing one throws here, ahead of the topology gate below
    // whose whole contract is to run "BEFORE any `.type`/`.id` access" — turning a
    // malformed file into a 500 carrying a raw TypeError, where every other
    // non-object record already reports the transcript invalid and still serves
    // the row. The lookup stays FIRST, though: it is the identity guard, and
    // running the topology check ahead of it would serve this row's metadata for a
    // file that had been swapped for a different, also-corrupt session.
    const header = findSessionHeader(entries) as { id?: unknown } | undefined;
    if (header === undefined || header.id !== row.id) {
      await deps.runIngest();
      sendError(res, 409, `Session "${params.id}" no longer matches its indexed file`);
      return;
    }
    // A file mutated into a cycle/self-parent since it was indexed (ingest
    // warns+skips but leaves the stale row) would send pi's visited-set-free
    // branch walker into a non-terminating loop that no try/catch can intercept.
    // Gate the projection on the same topology guard ingest uses, surfacing the
    // failure through `transcriptError` like every other detail error here.
    if (!isValidSessionTopology(entries)) {
      transcriptError = 'Session tree is invalid (cycle, duplicate id, or non-object message); transcript unavailable';
    } else {
      const isMessage = (entry: TranscriptEntry | null): entry is TranscriptEntry => entry !== null;
      // Project the SAME active, message-bearing branch the index derives its turns
      // from, so the transcript never disagrees with the indexed turn set. When the
      // natural leaf is a detached `session_info` (e.g. after a rename), this
      // re-projects from the latest message-bearing leaf — never a flat union of
      // every abandoned branch, which would resurrect superseded turns.
      //
      // The branch walk returns a root-to-leaf parent chain, which IS the order to
      // render: a tool result follows its call because it is that call's child.
      // Re-sorting by wall clock would let a clock step backwards, or one message
      // with no parseable timestamp (`ts = 0`, which the topology guard accepts),
      // hoist a result above its call — and would then disagree with `firstMessage`,
      // which the index derives from this same chain without sorting. `ts` is for
      // display only.
      const branch = activeBranchEntries(entries);
      const callArgs = collectCallArgs(branch);
      transcript = branch.map((entry) => mapTranscriptEntry(entry, callArgs)).filter(isMessage);
    }
  }
  sendJson(res, 200, {
    session: {
      id: row.id,
      path: row.path,
      cwd: row.cwd,
      name: row.name,
      created: row.created,
      modified: row.modified,
      messageCount: row.messageCount,
      firstMessage: row.firstMessage,
    },
    transcript,
    ...(transcriptError !== undefined ? { transcriptError } : {}),
  });
}

async function handleSessionRename({ req, res, params, deps }: RouteCtx): Promise<void> {
  const found = lookupSession(deps, params.id);
  if (found === null) {
    sendError(res, 404, `Session "${params.id}" not found`);
    return;
  }
  let body: unknown;
  try {
    body = await readJsonBody(req);
  } catch (err) {
    sendError(res, 400, err instanceof Error ? err.message : 'Invalid request body');
    return;
  }
  const name = (body as { name?: unknown } | null)?.name;
  if (typeof name !== 'string' || name === '') {
    sendError(res, 400, 'Body must include a non-empty "name" string');
    return;
  }
  if (!insideSessionsRoot(deps.sessionsRoot, found.path)) {
    sendError(res, 403, 'Session file resolves outside the managed sessions root');
    return;
  }
  // The indexed path may have been reused by a newer session since it was
  // indexed. Verify the file header still identifies THIS session before writing
  // its name — otherwise `appendSessionInfo` would stamp the name into a foreign
  // session's file. On mismatch, reconcile the index and refuse rather than mutate.
  if (!verifySessionFileId(found.path, params.id)) {
    await deps.runIngest();
    sendError(res, 409, `Session "${params.id}" no longer matches its indexed file`);
    return;
  }
  // `SessionManager.open` migrates and rewrites the file on construction,
  // persisting only the successfully-parsed in-memory entries — so an unparseable
  // line or an incomplete trailing write would be permanently truncated from disk
  // (the GET detail handler avoids this exact call by reading read-only). There is
  // no non-destructive rename in the pi SDK, so refuse rather than lose records:
  // an unparseable line (parsed count < non-blank line count) or a malformed
  // trailing line means opening for write would drop data. A complete v1 file is
  // still safe — its migration preserves every record — so only true data loss
  // is blocked here.
  let wouldDropRecords: boolean;
  try {
    const raw = readFileSync(found.path, 'utf8');
    wouldDropRecords = countJsonlLines(raw) !== parseSessionEntries(raw).length || !lastLineParses(raw);
  } catch {
    // The file changed under us since the header check; refuse rather than open
    // for write, and reconcile the index to reflect reality.
    await deps.runIngest();
    sendError(res, 409, `Session "${params.id}" no longer matches its indexed file`);
    return;
  }
  if (wouldDropRecords) {
    sendError(res, 409, 'Session file has incomplete/malformed records; cannot rename without data loss');
    return;
  }
  // Liveness pre-check: a session whose file was modified within LIVE_SESSION_WINDOW_MS
  // may be actively written by a concurrent agent turn (see the constant's note).
  // Renaming it would race that turn with no cross-process lock to protect us, so
  // refuse while it looks active and only append for an idle session.
  let mtimeMs: number;
  try {
    mtimeMs = statSync(found.path).mtimeMs;
  } catch {
    // The file changed under us since the checks above; reconcile and refuse rather
    // than open a file that may have been swapped out.
    await deps.runIngest();
    sendError(res, 409, `Session "${params.id}" no longer matches its indexed file`);
    return;
  }
  if (Date.now() - mtimeMs < LIVE_SESSION_WINDOW_MS) {
    sendError(res, 409, 'Cannot rename a session that is currently active; try again once the agent is idle.');
    return;
  }
  try {
    const manager = SessionManager.open(found.path);
    manager.appendSessionInfo(name);
  } catch (err) {
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to rename session');
    return;
  }
  // Re-index so the stored name reflects the freshly appended session_info line.
  await deps.runIngest();
  sendJson(res, 200, { id: params.id, name });
}

async function handleSessionDelete({ res, params, deps }: RouteCtx): Promise<void> {
  const found = lookupSession(deps, params.id);
  if (found === null) {
    sendError(res, 404, `Session "${params.id}" not found`);
    return;
  }
  if (!insideSessionsRoot(deps.sessionsRoot, found.path)) {
    sendError(res, 403, 'Session file resolves outside the managed sessions root');
    return;
  }
  // Decide what to do with the on-disk transcript BEFORE touching any rows. A bare
  // "does it still belong to this id" boolean cannot tell a file that provably
  // belongs to a DIFFERENT session (its path reused by a newer one) apart from one we
  // simply cannot verify (unreadable / corrupt / headerless):
  //   - 'matches'      → unlink the file, then drop the rows.
  //   - 'different'    → a newer session's file; must NOT unlink it. Drop our stale
  //                      rows only and let re-ingest reconcile.
  //   - 'missing'      → nothing to unlink; drop the rows.
  //   - 'unverifiable' → deleting the rows while leaving the file on disk desyncs the
  //                      DB from disk: the transcript could re-index on a later
  //                      successful ingest (reappearing after we said `deleted`), or
  //                      linger as an orphan. Reconcile the index and refuse, mirroring
  //                      the rename handler, rather than half-delete.
  // Unlike the rename above, delete deliberately carries NO liveness pre-check.
  // Rename is a read-modify-write, so a concurrent turn silently orphans a record
  // INSIDE a file that still looks intact — corruption the user never asked for and
  // cannot see. Unlinking is neither: it is exactly the outcome the user asked for,
  // behind a confirm dialog that names this hazard verbatim ("If an agent is
  // currently using this session, deleting it may orphan in-progress turns"). A live
  // agent's next append — pi's `SessionManager._persist` writes by PATH, not through
  // a held fd — recreates the path holding that entry alone; ingest reports `no
  // valid session header` and indexes nothing, and pi's own discovery drops
  // headerless files from the resume list, so the leftover is inert. Borrowing
  // LIVE_SESSION_WINDOW_MS here would not even prevent it: an agent idle at its
  // prompt (where a session spends most of its life) has a stale mtime and still
  // appends on the next message. It would only make a just-finished or just-crashed
  // session undeletable until the window passes.
  const verdict = classifySessionFile(found.path, params.id);
  if (verdict === 'unverifiable') {
    await deps.runIngest();
    sendError(
      res,
      409,
      `Session "${params.id}" file exists but could not be verified (unreadable or corrupt); delete refused to avoid orphaning it`,
    );
    return;
  }
  if (verdict === 'matches') {
    rmSync(found.path, { force: true });
  }
  const { db, sqlite } = deps.dash;
  sqlite.exec('BEGIN');
  try {
    db.delete(turns).where(eq(turns.sessionId, params.id)).run();
    db.delete(sessions).where(eq(sessions.id, params.id)).run();
    sqlite.exec('COMMIT');
  } catch (err) {
    sqlite.exec('ROLLBACK');
    sendError(res, 500, err instanceof Error ? err.message : 'Failed to delete session rows');
    return;
  }
  sendJson(res, 200, { deleted: true, id: params.id });
}

function handleSessionMetrics({ res, params, deps }: RouteCtx): void {
  const found = lookupSession(deps, params.id);
  if (found === null) {
    sendError(res, 404, `Session "${params.id}" not found`);
    return;
  }
  // The per-turn set the SPA charts / counts / token totals are built from is this
  // session's persisted `turns` (LEFT JOINed to their trace) UNIONed with any
  // delegated subagent turns: a child runs on an in-memory session manager (no
  // session JSONL → no `turns` row), yet its shared-provider `traces` row carries
  // the token columns and points back here via `root_session_id`. Without the union
  // the child's tokens/count are dropped from the totals even though the SPA already
  // shows its model badge + folds its throughput in (from the `traces` array below),
  // and it disagrees with the global overview. The union admits ONLY GENUINE children:
  // a child stamps `session_id = <child in-memory id>` (≠ root) and
  // `root_session_id = <root id>` (stream-adapter.ts: `sessionId` from the per-request
  // options, `rootSessionId` from the submit-time cache-owner root), so keying on
  // `root_session_id = ? AND session_id != ?` selects them. Keying on `session_id = ?`
  // too would resurrect an ABANDONED root turn — when a root branches, ingestion drops
  // its assistant turn from `turns` but the trace lingers with `session_id = root`
  // (== root_session_id); it would then pass the trace-only dedup and be miscounted as
  // a delegated turn. The dedup admits ONLY traces with no correlated `turns` row for
  // THIS session (the inner `AND trace_id IS NOT NULL` keeps `NOT IN` from going NULL
  // and dropping the whole set), and — mirroring the global fix in
  // handleMetricsOverview — the trace side stores GROSS `prompt_tokens` so its
  // `inputTokens` is clamped to the producer's net value `MAX(prompt-cached,0)` to
  // avoid a gross-vs-net double-count against the turns side's already-net input.
  const turnRows = deps.dash.sqlite
    .prepare(
      `SELECT * FROM (
         SELECT t.entry_id AS entryId, t.trace_id AS traceId, t.ts AS ts, t.model AS model,
                t.input_tokens AS inputTokens, t.output_tokens AS outputTokens,
                t.cached_tokens AS cachedTokens, t.reasoning_tokens AS reasoningTokens,
                tr.ttft_ms AS ttftMs, tr.prefill_tps AS prefillTps, tr.decode_tps AS decodeTps,
                tr.mtp_cycles AS mtpCycles, tr.mtp_mean_accepted AS mtpMeanAccepted,
                tr.duration_ms AS durationMs, tr.finish_reason AS finishReason,
                tr.cold_hits AS coldHits, tr.cold_misses AS coldMisses,
                tr.cold_bytes_written AS coldBytesWritten, tr.cold_bytes_restored AS coldBytesRestored
         FROM turns t
         LEFT JOIN traces tr ON tr.trace_id = t.trace_id
         WHERE t.session_id = ?
         UNION ALL
         SELECT NULL AS entryId, tr.trace_id AS traceId, tr.ts AS ts, tr.model AS model,
                MAX(COALESCE(tr.prompt_tokens, 0) - COALESCE(tr.cached_tokens, 0), 0) AS inputTokens,
                tr.output_tokens AS outputTokens, tr.cached_tokens AS cachedTokens,
                tr.reasoning_tokens AS reasoningTokens,
                tr.ttft_ms AS ttftMs, tr.prefill_tps AS prefillTps, tr.decode_tps AS decodeTps,
                tr.mtp_cycles AS mtpCycles, tr.mtp_mean_accepted AS mtpMeanAccepted,
                tr.duration_ms AS durationMs, tr.finish_reason AS finishReason,
                tr.cold_hits AS coldHits, tr.cold_misses AS coldMisses,
                tr.cold_bytes_written AS coldBytesWritten, tr.cold_bytes_restored AS coldBytesRestored
         FROM traces tr
         WHERE tr.root_session_id = ? AND tr.session_id != ?
           AND tr.trace_id NOT IN (SELECT trace_id FROM turns WHERE session_id = ? AND trace_id IS NOT NULL)
       )
       ORDER BY ts`,
    )
    .all(params.id, params.id, params.id, params.id);
  // Include this session's own ACTIVE-branch turns AND any subagent turns delegated
  // under it — but NOT an abandoned root turn's lingering trace. The SPA derives its
  // model badges and avg TTFT/decode chips from this `traces` array while the per-turn
  // charts use `turns` (above), so keying on `session_id = ?` alone would readmit the
  // abandoned root trace `turnRows` deliberately excludes (a branched root drops its
  // assistant turn from `turns` but the trace lingers with `session_id = root`), making
  // the chips disagree with the transcript. Mirror `turnRows`: select a root trace only
  // when it correlates to an ACTIVE `turns` row (via `trace_id`), plus genuine delegated
  // children (`root_session_id = ? AND session_id != ?`) not already correlated here. A
  // child (subagent) trace carries no persisted session JSONL, but its `root_session_id`
  // points back here (Finding 11b).
  const traceRows = deps.dash.sqlite
    .prepare(
      `SELECT trace_id AS traceId, session_id AS sessionId, root_session_id AS rootSessionId,
              ts, model, ttft_ms AS ttftMs, prefill_tps AS prefillTps,
              decode_tps AS decodeTps, mtp_cycles AS mtpCycles, mtp_mean_accepted AS mtpMeanAccepted,
              duration_ms AS durationMs, finish_reason AS finishReason,
              cold_hits AS coldHits, cold_misses AS coldMisses,
              cold_bytes_written AS coldBytesWritten, cold_bytes_restored AS coldBytesRestored
       FROM traces tr
       WHERE tr.trace_id IN (SELECT trace_id FROM turns WHERE session_id = ? AND trace_id IS NOT NULL)
          OR (tr.root_session_id = ? AND tr.session_id != ?
              AND tr.trace_id NOT IN (SELECT trace_id FROM turns WHERE session_id = ? AND trace_id IS NOT NULL))
       ORDER BY ts`,
    )
    .all(params.id, params.id, params.id, params.id);
  sendJson(res, 200, { sessionId: params.id, turns: turnRows, traces: traceRows });
}

function rangeClause(from: number | null, to: number | null, column: string): { sql: string; args: number[] } {
  const parts: string[] = [];
  const args: number[] = [];
  if (from !== null) {
    parts.push(`${column} >= ?`);
    args.push(from);
  }
  if (to !== null) {
    parts.push(`${column} <= ?`);
    args.push(to);
  }
  return { sql: parts.length > 0 ? parts.join(' AND ') : '', args };
}

function handleMetricsOverview({ res, url, deps }: RouteCtx): void {
  const from = queryInt(url, 'from');
  const to = queryInt(url, 'to');
  const { sqlite } = deps.dash;

  const turnsRange = rangeClause(from, to, 'ts');
  const tracesRange = rangeClause(from, to, 'ts');
  const turnsWhere = (extra: string): string => {
    const parts = [extra, turnsRange.sql].filter((p) => p !== '');
    return parts.length > 0 ? `WHERE ${parts.join(' AND ')}` : '';
  };
  const tracesWhere = (extra: string): string => {
    const parts = [extra, tracesRange.sql].filter((p) => p !== '');
    return parts.length > 0 ? `WHERE ${parts.join(' AND ')}` : '';
  };

  // A forked session copies inherited turns VERBATIM (same trace_id/entry_id) into
  // a new session file, so the same inference lands as multiple `turns` rows. The
  // per-session views keep every copy (each transcript is correct), but these
  // GLOBAL token sums must count each inference once — collapse copies on their
  // canonical identity `COALESCE(trace_id, entry_id, CAST(id AS TEXT))` (the
  // autoincrement id keeps genuinely-distinct, both-null rows separate).
  const dedupKey = 'COALESCE(trace_id, entry_id, CAST(id AS TEXT))';

  // Subagent turns run on an in-memory session manager (no session JSONL → no
  // `turns` row), yet the shared provider still writes a `traces` row carrying the
  // token columns. UNION those TRACE-ONLY rows into the turns-derived token
  // aggregates below so delegated work is not silently underreported. This guard
  // admits ONLY traces with no correlated `turns` row, so a normal/forked turn's
  // tokens stay sourced from `turns` and are never counted twice. The inner
  // `WHERE trace_id IS NOT NULL` is load-bearing: a NULL in the subquery would make
  // `NOT IN` evaluate to NULL for every row and drop the whole trace-only set. Each
  // trace-only row is one delegated turn of real work, so it also adds 1 to the
  // per-model / overall turn COUNT.
  const traceOnly = 'trace_id NOT IN (SELECT trace_id FROM turns WHERE trace_id IS NOT NULL)';

  // The turns side's `input_tokens` is pi `usage.input`, already NET of cache
  // (`max(0, promptTokens - cacheRead)`; see packages/agent/src/provider/events.ts).
  // A trace row instead stores GROSS `prompt_tokens` (provider/index.ts). Projecting
  // gross into the same `input` column as the turns side double-counts cached tokens
  // for trace-only rows, so clamp the trace side to the producer's net value:
  // `MAX(prompt - cached, 0)` (SQLite two-arg `MAX(a,b)` is the scalar clamp). The
  // other projected trace columns are already apples-to-apples with turns.
  const traceNetInput = 'MAX(COALESCE(prompt_tokens, 0) - COALESCE(cached_tokens, 0), 0)';

  const tokensByDay = sqlite
    .prepare(
      `SELECT date(ts / 1000, 'unixepoch') AS day,
              COALESCE(SUM(input_tokens), 0) AS input,
              COALESCE(SUM(output_tokens), 0) AS output,
              COALESCE(SUM(cached_tokens), 0) AS cached,
              COALESCE(SUM(reasoning_tokens), 0) AS reasoning
       FROM (SELECT input_tokens, output_tokens, cached_tokens, reasoning_tokens, ts
             FROM turns ${turnsWhere('ts > 0')} GROUP BY ${dedupKey}
             UNION ALL
             SELECT ${traceNetInput}, output_tokens, cached_tokens, reasoning_tokens, ts
             FROM traces ${tracesWhere(`ts > 0 AND ${traceOnly}`)})
       GROUP BY day ORDER BY day`,
    )
    .all(...turnsRange.args, ...tracesRange.args)
    .map((row) => ({
      day: String(row.day),
      input: toInt(row.input),
      output: toInt(row.output),
      cached: toInt(row.cached),
      reasoning: toInt(row.reasoning),
    }));

  const throughputByModel = sqlite
    .prepare(
      `SELECT COALESCE(model, 'unknown') AS model,
              AVG(decode_tps) AS avgDecodeTps, AVG(prefill_tps) AS avgPrefillTps,
              AVG(ttft_ms) AS avgTtftMs, COUNT(*) AS samples
       FROM traces ${tracesWhere('')} GROUP BY model ORDER BY samples DESC`,
    )
    .all(...tracesRange.args)
    .map((row) => ({
      model: String(row.model),
      avgDecodeTps: toNum(row.avgDecodeTps),
      avgPrefillTps: toNum(row.avgPrefillTps),
      avgTtftMs: toNum(row.avgTtftMs),
      samples: toInt(row.samples),
    }));

  // Same per-model averages as above, but bucketed per day so the UI can chart a
  // trend. Uses the identical `date(ts/1000,'unixepoch')` bucket and `ts > 0`
  // guard as `tokensByDay` so the two series line up on the same day keys.
  const throughputTrend = sqlite
    .prepare(
      `SELECT COALESCE(model, 'unknown') AS model,
              date(ts / 1000, 'unixepoch') AS day,
              AVG(decode_tps) AS avgDecodeTps, AVG(prefill_tps) AS avgPrefillTps,
              AVG(ttft_ms) AS avgTtftMs, COUNT(*) AS samples
       FROM traces ${tracesWhere('ts > 0')} GROUP BY model, day ORDER BY day, model`,
    )
    .all(...tracesRange.args)
    .map((row) => ({
      model: String(row.model),
      day: String(row.day),
      decodeTps: toNum(row.avgDecodeTps),
      prefillTps: toNum(row.avgPrefillTps),
      ttftMs: toNum(row.avgTtftMs),
      samples: toInt(row.samples),
    }));

  const mtpByModel = sqlite
    .prepare(
      `SELECT COALESCE(model, 'unknown') AS model,
              AVG(mtp_mean_accepted) AS meanAccepted, AVG(mtp_cycles) AS avgCycles,
              COUNT(mtp_mean_accepted) AS samples
       FROM traces ${tracesWhere('mtp_mean_accepted IS NOT NULL')} GROUP BY model ORDER BY samples DESC`,
    )
    .all(...tracesRange.args)
    .map((row) => ({
      model: String(row.model),
      meanAccepted: toNum(row.meanAccepted),
      avgCycles: toNum(row.avgCycles),
      samples: toInt(row.samples),
    }));

  const modelShare = sqlite
    .prepare(
      `SELECT COALESCE(model, 'unknown') AS model, COUNT(*) AS turns,
              COALESCE(SUM(output_tokens), 0) AS outputTokens
       FROM (SELECT model, output_tokens
             FROM turns ${turnsWhere('')} GROUP BY ${dedupKey}
             UNION ALL
             SELECT model, output_tokens
             FROM traces ${tracesWhere(traceOnly)})
       GROUP BY model ORDER BY turns DESC`,
    )
    .all(...turnsRange.args, ...tracesRange.args)
    .map((row) => ({ model: String(row.model), turns: toInt(row.turns), outputTokens: toInt(row.outputTokens) }));

  const turnTotals = sqlite
    .prepare(
      `SELECT COUNT(*) AS turns, COALESCE(SUM(input_tokens), 0) AS inputTokens,
              COALESCE(SUM(output_tokens), 0) AS outputTokens,
              COALESCE(SUM(cached_tokens), 0) AS cachedTokens,
              COALESCE(SUM(reasoning_tokens), 0) AS reasoningTokens
       FROM (SELECT input_tokens, output_tokens, cached_tokens, reasoning_tokens
             FROM turns ${turnsWhere('')} GROUP BY ${dedupKey}
             UNION ALL
             SELECT ${traceNetInput}, output_tokens, cached_tokens, reasoning_tokens
             FROM traces ${tracesWhere(traceOnly)})`,
    )
    .get(...turnsRange.args, ...tracesRange.args);
  const traceTotals = sqlite
    .prepare(`SELECT COUNT(*) AS traces FROM traces ${tracesWhere('')}`)
    .get(...tracesRange.args);

  const overview: MetricsOverview = {
    range: { from, to },
    tokensByDay,
    throughputByModel,
    throughputTrend,
    mtpByModel,
    modelShare,
    totals: {
      turns: toInt(turnTotals?.turns),
      traces: toInt(traceTotals?.traces),
      inputTokens: toInt(turnTotals?.inputTokens),
      outputTokens: toInt(turnTotals?.outputTokens),
      cachedTokens: toInt(turnTotals?.cachedTokens),
      reasoningTokens: toInt(turnTotals?.reasoningTokens),
    },
  };
  sendJson(res, 200, overview);
}

/**
 * Cold-tier view for ONE cache root: the on-disk scan, the trace-derived trend
 * SCOPED TO THAT SAME ROOT, and an explicit account of everything the scope
 * excluded.
 *
 * The scoping is the whole point. Before it, `disk` was a filesystem scan of one
 * root while `trend` was an unfiltered `SUM()` over every trace row ever
 * ingested, so a dashboard pointed at an EMPTY cache dir reported
 * `exists: false, entryCount: 0` beside a hit rate summed from every cache
 * directory the machine had ever used. Two disjoint notions of "the cache" in
 * one payload.
 *
 * Rows the scope drops are REPORTED, never silently discarded. The buckets
 * below are disjoint AND EXHAUSTIVE over every trace row — that is the point of
 * them, and it is asserted directly (the sum of trend + legacy + otherRoots +
 * unattributed hits equals the table's own `SUM(cold_hits)`). An earlier
 * partition had a hole: `cold_enabled = 1` with a NULL `cold_root` matched no
 * arm at all, so a turn whose tier was open but whose root went unrecorded had
 * its hits vanish — the exact outcome the scope design exists to refuse.
 *  - `legacy` — written before the agent recorded a root (`cold_enabled` NULL).
 *    Counting these into the shown cache IS the original bug; dropping them
 *    without a word makes a real 16K-lookup history vanish next to a root path
 *    the user is staring at. The bucket is bounded and self-clearing: trace
 *    JSONL is pruned at {@link TRACE_RETENTION_DAYS} days and the rows go with
 *    it, so it empties on its own once the recording build has shipped that
 *    long.
 *  - `otherRoots` — genuinely a different cache directory. A different user
 *    story from `legacy`, so a different line.
 *  - `unattributed` — the tier was ON (`cold_enabled` truthy) but the turn
 *    carried no root. The writer no longer produces this (it gates the record
 *    on the CANONICAL root, the same emptiness test `canonicalCacheRoot`
 *    applies), so the bucket should stay empty; it exists so that if one ever
 *    appears the lookups are shown rather than dropped on the floor.
 *  - `disabledTurns` — the tier was off (`--no-persist-cache`, or a family off
 *    the restore allowlist). Their deltas are all zero, so they are counted but
 *    carry no lookups. Overlaps the buckets above by design: it is a TURN
 *    count, not a lookup bucket.
 *  - `unrootedSidecarCaptures` — the one delta a rootless turn can still carry,
 *    and the reason "their deltas are all zero" is true of the BLOCK counters
 *    only. `record_capture_reached()` is the first statement of every family's
 *    sidecar capture, above its `adapter.cold_tier()` guard
 *    (`crates/mlx-core/src/models/gemma4/model.rs:2839` vs `:2843`, and the
 *    same shape in qwen3_5 and qwen3_5_moe), so a hybrid turn with persistence
 *    off — or one whose tier failed to open — counts the capture and then
 *    returns with no tier to name a root from. Summed here rather than folded
 *    into `health`: `health` means "against the root above", and these rows
 *    name no root, so the root scoping cannot be widened to reach them without
 *    also letting `otherRoots` in. Dropping them is what let the page report
 *    "not reached — dense families" over a hybrid capture that ran every turn
 *    and failed before it acquired a tier.
 *
 * `health` carries the counters that are NOT lookups, and therefore cannot be
 * read off the hit rate at all:
 *  - `writeErrors` — writes the queue accepted that never reached disk. The
 *    native writer is fail-open and returns the error to nobody, so a root that
 *    is read-only, full or unmounted otherwise produces a spotless payload:
 *    zero drops, zero corruptions, and a hit rate computed over a cache that
 *    is storing nothing.
 *  - `restoreDeclines` / `restoreSuppressed` — restores that were refused. A
 *    refusal performs no block lookup, so it moves neither `hits` nor `misses`
 *    and lands in `trend` as an absent row, which the UI would otherwise render
 *    as "no lookups recorded" — "nothing ran", when the truth is the opposite.
 *  - `sidecar*` — the family state that lives OUTSIDE the paged pool.
 *    `sidecarInstalled` is the only read-side one and the only counter in this
 *    payload that nothing else can stand in for: every
 *    `install_*_cold_sidecar` early-return falls through to a full O(prefix)
 *    replay that produces CORRECT state, so "restored and INSTALLED" and
 *    "restored and silently re-derived" agree on hits, cached tokens,
 *    corruptions and the emitted text alike.
 *
 * The two families reduce differently and must not be swapped. `*Total` fields
 * are per-process cumulative counters reduced with MAX (see below); declines
 * have no total and are SUMMED, because unlike a corruption a decline has a
 * legitimate non-zero steady state — every prompt's first turn declines — so a
 * "did this ever happen" latch would be pinned on from the first minute.
 */
function handleCacheGet({ res, deps }: RouteCtx): void {
  const requestedRoot = coldCacheRoot(deps.cacheRoot);
  const disk = scanColdCache(requestedRoot);
  // Both sides of the join canonicalize through the SAME helper the agent used
  // when it wrote the row. A raw string compare would be a silent-zero trap:
  // the writer and the reader are different processes resolving the root from
  // their own environments, and on macOS `/tmp` → `/private/tmp` alone makes
  // two identical-looking spellings unequal.
  const scopeRoot = canonicalCacheRoot(requestedRoot);
  const trend = deps.dash.sqlite
    .prepare(
      `SELECT date(ts / 1000, 'unixepoch') AS day,
              COALESCE(SUM(cold_hits), 0) AS hits, COALESCE(SUM(cold_misses), 0) AS misses,
              COALESCE(SUM(cold_bytes_written), 0) AS bytesWritten,
              COALESCE(SUM(cold_bytes_restored), 0) AS bytesRestored
       FROM traces WHERE ts > 0 AND cold_root = ? GROUP BY day ORDER BY day`,
    )
    .all(scopeRoot)
    .map((row) => ({
      day: String(row.day),
      hits: toInt(row.hits),
      misses: toInt(row.misses),
      bytesWritten: toInt(row.bytesWritten),
      bytesRestored: toInt(row.bytesRestored),
    }));

  // `CASE WHEN` rather than `FILTER` — portable across whatever SQLite the
  // bundled `node:sqlite` links.
  const excluded = deps.dash.sqlite
    .prepare(
      `SELECT
         COUNT(CASE WHEN cold_root IS NULL AND cold_enabled IS NULL THEN 1 END) AS legacyTurns,
         COALESCE(SUM(CASE WHEN cold_root IS NULL AND cold_enabled IS NULL THEN cold_hits END), 0) AS legacyHits,
         COALESCE(SUM(CASE WHEN cold_root IS NULL AND cold_enabled IS NULL THEN cold_misses END), 0) AS legacyMisses,
         COUNT(CASE WHEN cold_root IS NOT NULL AND cold_root <> ? THEN 1 END) AS otherTurns,
         COALESCE(SUM(CASE WHEN cold_root IS NOT NULL AND cold_root <> ? THEN cold_hits END), 0) AS otherHits,
         COALESCE(SUM(CASE WHEN cold_root IS NOT NULL AND cold_root <> ? THEN cold_misses END), 0) AS otherMisses,
         COUNT(CASE WHEN cold_root IS NULL AND cold_enabled IS NOT NULL AND cold_enabled <> 0 THEN 1 END)
           AS unattributedTurns,
         COALESCE(SUM(CASE WHEN cold_root IS NULL AND cold_enabled IS NOT NULL AND cold_enabled <> 0
                           THEN cold_hits END), 0) AS unattributedHits,
         COALESCE(SUM(CASE WHEN cold_root IS NULL AND cold_enabled IS NOT NULL AND cold_enabled <> 0
                           THEN cold_misses END), 0) AS unattributedMisses,
         COUNT(CASE WHEN cold_enabled = 0 THEN 1 END) AS disabledTurns,
         COALESCE(SUM(CASE WHEN cold_root IS NULL THEN cold_sidecar_capture_reached END), 0)
           AS unrootedSidecarCaptures
       FROM traces WHERE ts > 0`,
    )
    .get(scopeRoot, scopeRoot, scopeRoot);

  // Cumulative counters use MAX, not SUM: each process's total restarts at 0
  // when its tier opens, so summing across processes double-counts. We only
  // need 0-vs-non-zero ("did this EVER happen against this cache"), and
  // MAX > 0 answers exactly that — including for a turn that aborted before it
  // could record a delta.
  const health = deps.dash.sqlite
    .prepare(
      `SELECT COALESCE(SUM(cold_enqueued), 0) AS enqueued,
              COALESCE(SUM(cold_queue_drops), 0) AS queueDrops,
              COALESCE(SUM(cold_evictions), 0) AS evictions,
              COALESCE(SUM(cold_corruptions), 0) AS corruptions,
              COALESCE(MAX(cold_corruptions_total), 0) AS corruptionsTotal,
              COALESCE(MAX(cold_queue_drops_total), 0) AS queueDropsTotal,
              COALESCE(SUM(cold_write_errors), 0) AS writeErrors,
              COALESCE(MAX(cold_write_errors_total), 0) AS writeErrorsTotal,
              COALESCE(SUM(cold_restore_declines), 0) AS restoreDeclines,
              COALESCE(SUM(cold_sidecar_restore_suppressed), 0) AS restoreSuppressed,
              COALESCE(SUM(cold_sidecar_capture_reached), 0) AS sidecarCaptureReached,
              COALESCE(SUM(cold_sidecar_chain_empty), 0) AS sidecarChainEmpty,
              COALESCE(SUM(cold_sidecar_boundary_skips), 0) AS sidecarBoundarySkips,
              COALESCE(SUM(cold_sidecar_already_persisted), 0) AS sidecarAlreadyPersisted,
              COALESCE(SUM(cold_sidecar_enqueued), 0) AS sidecarEnqueued,
              COALESCE(SUM(cold_sidecar_queue_drops), 0) AS sidecarQueueDrops,
              COALESCE(SUM(cold_sidecar_installed), 0) AS sidecarInstalled
       FROM traces WHERE ts > 0 AND cold_root = ?`,
    )
    .get(scopeRoot);

  sendJson(res, 200, {
    disk,
    trend,
    scope: {
      root: scopeRoot,
      trendWindowDays: TRACE_RETENTION_DAYS,
      legacy: {
        turns: toInt(excluded?.legacyTurns),
        hits: toInt(excluded?.legacyHits),
        misses: toInt(excluded?.legacyMisses),
      },
      otherRoots: {
        turns: toInt(excluded?.otherTurns),
        hits: toInt(excluded?.otherHits),
        misses: toInt(excluded?.otherMisses),
      },
      unattributed: {
        turns: toInt(excluded?.unattributedTurns),
        hits: toInt(excluded?.unattributedHits),
        misses: toInt(excluded?.unattributedMisses),
      },
      disabledTurns: toInt(excluded?.disabledTurns),
      // The one counter above that keeps moving after the root is gone, so the
      // one the buckets above cannot stand in for: they carry hits and misses,
      // and a tier-less turn has neither. Reported here rather than merged into
      // `health` because `health` means "this cache root", and these rows name
      // no root at all — merging them would put numbers from a run that never
      // touched the shown cache under a heading that promises it did.
      unrootedSidecarCaptures: toInt(excluded?.unrootedSidecarCaptures),
    },
    health: {
      enqueued: toInt(health?.enqueued),
      queueDrops: toInt(health?.queueDrops),
      evictions: toInt(health?.evictions),
      corruptions: toInt(health?.corruptions),
      corruptionsTotal: toInt(health?.corruptionsTotal),
      queueDropsTotal: toInt(health?.queueDropsTotal),
      writeErrors: toInt(health?.writeErrors),
      writeErrorsTotal: toInt(health?.writeErrorsTotal),
      // SUM, not MAX, and deliberately so: unlike corruptions and write
      // errors these two have a legitimate non-zero steady state (any prompt's
      // first turn declines), so a "did it ever happen" latch would be pinned
      // on from the first minute and carry no information.
      restoreDeclines: toInt(health?.restoreDeclines),
      restoreSuppressed: toInt(health?.restoreSuppressed),
      // The other seven `ColdSidecarStats` counters. Every one is a per-turn
      // DELTA, so every one SUMS — none has a cumulative `*_total` twin, and
      // MAX would not stand in for one: over deltas it reports the busiest
      // single turn rather than "did this ever happen".
      //
      // `sidecarEnqueued` / `sidecarQueueDrops` are the sidecar SHARE of the
      // object-scoped `enqueued` / `queueDrops` above (a sidecar admission
      // bumps both counters), so a consumer reads them as a subset and never
      // adds the pairs. `restoreSuppressed` is the eighth of this family and
      // keeps its unprefixed name because it is read beside `restoreDeclines`,
      // the other way reuse is refused.
      sidecarCaptureReached: toInt(health?.sidecarCaptureReached),
      sidecarChainEmpty: toInt(health?.sidecarChainEmpty),
      sidecarBoundarySkips: toInt(health?.sidecarBoundarySkips),
      sidecarAlreadyPersisted: toInt(health?.sidecarAlreadyPersisted),
      sidecarEnqueued: toInt(health?.sidecarEnqueued),
      sidecarQueueDrops: toInt(health?.sidecarQueueDrops),
      sidecarInstalled: toInt(health?.sidecarInstalled),
    },
    // Sent over the wire rather than bundled into the SPA: the browser build
    // cannot import `@mlx-node/agent` (it transitively loads the native addon),
    // and a hardcoded copy in the UI would sit outside the drift guard in
    // `packages/agent/__test__/cold-tier-families.test.ts`.
    restoreFamilies: coldTierRestoreFamilyList(),
  });
}

async function handleCacheDelete({ req, res, deps }: RouteCtx): Promise<void> {
  let body: unknown;
  try {
    body = await readJsonBody(req);
  } catch (err) {
    sendError(res, 400, err instanceof Error ? err.message : 'Invalid request body');
    return;
  }
  const parsed = body as { all?: unknown; olderThanDays?: unknown } | null;
  const all = parsed?.all;
  const olderThanDays = parsed?.olderThanDays;
  const root = deps.cacheRoot;
  let result: { removed: number; freedBytes: number };
  // Clear-all needs an explicit `{"all": true}` discriminator; selective
  // eviction needs a positive finite `olderThanDays`. Anything else (absent,
  // string, zero, negative, misspelled) is a 400 — never a silent whole-cache
  // wipe from a typing slip.
  if (all === true) {
    result = root !== undefined ? clearColdCache(root) : clearColdCache();
  } else if (typeof olderThanDays === 'number' && Number.isFinite(olderThanDays) && olderThanDays > 0) {
    result = root !== undefined ? evictOlderThan(olderThanDays, root) : evictOlderThan(olderThanDays);
  } else {
    sendError(res, 400, 'Body must be {"all":true} to clear all, or {"olderThanDays":<positive number>} to evict');
    return;
  }
  sendJson(res, 200, result);
}

async function handleIngest({ res, deps }: RouteCtx): Promise<void> {
  const summary = await deps.runIngest();
  sendJson(res, 200, summary);
}

const ROUTES: Route[] = [
  route('GET', '/health', handleHealth),
  // Alias under `/api` so the health check is reachable through the UI dev
  // server's `/api` proxy (which does not forward the bare `/health`).
  route('GET', '/api/health', handleHealth),
  route('GET', '/api/models', handleModels),
  route('DELETE', '/api/models/:name', handleDeleteModel),
  route('GET', '/api/catalog', handleCatalog),
  route('GET', '/api/downloads', handleDownloadsList),
  route('POST', '/api/downloads', handleDownloadStart),
  route('DELETE', '/api/downloads/:id', handleDownloadCancel),
  route('GET', '/api/downloads/:id/events', handleDownloadEvents),
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

/**
 * Dispatch an API/health request. Returns `true` when a route matched (response
 * written), `false` when the path is not an API route (caller serves static).
 * A path under `/api` that matches no route yields a 404 JSON here.
 */
export async function handleApiRequest(
  req: IncomingMessage,
  res: ServerResponse,
  url: URL,
  deps: ApiDeps,
): Promise<boolean> {
  const pathname = url.pathname;
  const isApi = pathname === '/health' || pathname === '/api' || pathname.startsWith('/api/');
  if (!isApi) return false;

  const method = req.method ?? 'GET';
  const segments = pathname.split('/').filter((s) => s.length > 0);
  let pathMatched = false;
  for (const r of ROUTES) {
    const params = matchRoute(r, method, segments);
    if (params === null) {
      if (segmentsMatch(r.segments, segments)) pathMatched = true;
      continue;
    }
    try {
      await r.handler({ req, res, url, params, deps });
    } catch (err) {
      if (!res.headersSent) {
        sendError(res, 500, err instanceof Error ? err.message : 'Internal server error');
      } else {
        res.end();
      }
    }
    return true;
  }

  // A known path with the wrong method → 405; otherwise 404.
  if (pathMatched) {
    sendError(res, 405, `Method ${method} not allowed for ${pathname}`);
  } else {
    sendError(res, 404, `No route matches ${method} ${pathname}`);
  }
  return true;
}

/** Whether a route's segment pattern shape matches concrete segments (ignoring method/values). */
function segmentsMatch(pattern: string[], segments: string[]): boolean {
  if (pattern.length !== segments.length) return false;
  for (let i = 0; i < pattern.length; i++) {
    if (!pattern[i].startsWith(':') && pattern[i] !== segments[i]) return false;
  }
  return true;
}
