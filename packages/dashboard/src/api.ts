/**
 * JSON + SSE API route handlers for the dashboard, over plain `node:http`.
 *
 * A tiny `(method, pathname)` matcher with `:param` segments dispatches to the
 * handlers below; there is no web framework. All data comes from the C1–C4
 * modules (SQLite index, local model store, cold-tier scan, download runner)
 * and pi's `SessionManager` for transcripts/rename — never the native addon.
 */

import { existsSync, readFileSync, realpathSync, rmSync, statSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { basename, dirname, join, sep } from 'node:path';

import {
  SessionManager,
  parseSessionEntries,
  type FileEntry,
  type SessionEntry,
} from '@earendil-works/pi-coding-agent';
import { eq } from 'drizzle-orm';

import { scanColdCache, clearColdCache, evictOlderThan } from './cache.js';
import { catalogWithState } from './catalog.js';
import type { DashboardDb } from './db/open.js';
import { sessions, turns } from './db/schema.js';
import type { DownloadManager, DownloadEvent } from './download.js';
import {
  activeBranchEntries,
  countJsonlLines,
  isValidSessionTopology,
  lastLineParses,
  readSessionEntries,
  verifySessionFileId,
} from './ingest/sessions.js';
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
  traces: { files: number; records: number; pruned: number };
}

export interface SseClient {
  res: ServerResponse;
  cleanup: () => void;
}

/** One transcript message projected from a pi session entry. */
export interface TranscriptEntry {
  role: string;
  text: string;
  toolCalls: Array<{ id: string; name: string; arguments: unknown }>;
  ts: number;
  /** Present for `toolResult` messages. */
  toolName?: string;
  isError?: boolean;
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
  sendJson(res, 200, { models, warnings });
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

  const flush = (): void => {
    blocked = false;
    while (pending.length > 0) {
      if (!res.write(sseFrame(pending[0]))) {
        blocked = true;
        return;
      }
      pending.shift();
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
    LIMIT 500`;

  const rows = deps.dash.sqlite.prepare(sql).all(...args);
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
  sendJson(res, 200, { sessions: list });
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

function extractText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (block && typeof block === 'object' && (block as { type?: unknown }).type === 'text') {
      const text = (block as { text?: unknown }).text;
      if (typeof text === 'string') parts.push(text);
    }
  }
  return parts.join('\n');
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
      });
    }
  }
  return calls;
}

function mapTranscriptEntry(entry: SessionEntry): TranscriptEntry | null {
  if (entry.type !== 'message') return null;
  const msg = entry.message as {
    role?: unknown;
    content?: unknown;
    timestamp?: unknown;
    toolName?: unknown;
    isError?: unknown;
  };
  const role = typeof msg.role === 'string' ? msg.role : 'unknown';
  const ts =
    typeof entry.timestamp === 'string' && !Number.isNaN(Date.parse(entry.timestamp))
      ? Date.parse(entry.timestamp)
      : typeof msg.timestamp === 'number'
        ? msg.timestamp
        : 0;
  const mapped: TranscriptEntry = {
    role,
    text: extractText(msg.content),
    toolCalls: extractToolCalls(msg.content),
    ts,
  };
  if (role === 'toolResult') {
    if (typeof msg.toolName === 'string') mapped.toolName = msg.toolName;
    if (typeof msg.isError === 'boolean') mapped.isError = msg.isError;
  }
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
    const header = entries.find((e) => e.type === 'session') as { id?: unknown } | undefined;
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
      transcriptError = 'Session tree is invalid (cycle or duplicate entry id); transcript unavailable';
    } else {
      const isMessage = (entry: TranscriptEntry | null): entry is TranscriptEntry => entry !== null;
      // Project the SAME active, message-bearing branch the index derives its turns
      // from, so the transcript never disagrees with the indexed turn set. When the
      // natural leaf is a detached `session_info` (e.g. after a rename), this
      // re-projects from the latest message-bearing leaf — never a flat union of
      // every abandoned branch, which would resurrect superseded turns.
      transcript = activeBranchEntries(entries).map(mapTranscriptEntry).filter(isMessage);
      transcript.sort((a, b) => a.ts - b.ts);
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

function handleSessionDelete({ res, params, deps }: RouteCtx): void {
  const found = lookupSession(deps, params.id);
  if (found === null) {
    sendError(res, 404, `Session "${params.id}" not found`);
    return;
  }
  if (!insideSessionsRoot(deps.sessionsRoot, found.path)) {
    sendError(res, 403, 'Session file resolves outside the managed sessions root');
    return;
  }
  // Last-defense guard: only unlink the file if it still belongs to this id. A
  // stale index row (path since reused by a newer session) must not delete the
  // newer session's file — drop our rows and let re-ingest reconcile instead.
  if (existsSync(found.path) && verifySessionFileId(found.path, params.id)) {
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
  // Include this session's own turns AND any subagent turns delegated under it:
  // a child (subagent) trace carries no persisted session JSONL, but its
  // root_session_id points back here (Finding 11b).
  const traceRows = deps.dash.sqlite
    .prepare(
      `SELECT trace_id AS traceId, session_id AS sessionId, root_session_id AS rootSessionId,
              ts, model, ttft_ms AS ttftMs, prefill_tps AS prefillTps,
              decode_tps AS decodeTps, mtp_cycles AS mtpCycles, mtp_mean_accepted AS mtpMeanAccepted,
              duration_ms AS durationMs, finish_reason AS finishReason,
              cold_hits AS coldHits, cold_misses AS coldMisses,
              cold_bytes_written AS coldBytesWritten, cold_bytes_restored AS coldBytesRestored
       FROM traces WHERE session_id = ? OR root_session_id = ? ORDER BY ts`,
    )
    .all(params.id, params.id);
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

function handleCacheGet({ res, deps }: RouteCtx): void {
  const disk = deps.cacheRoot !== undefined ? scanColdCache(deps.cacheRoot) : scanColdCache();
  const trend = deps.dash.sqlite
    .prepare(
      `SELECT date(ts / 1000, 'unixepoch') AS day,
              COALESCE(SUM(cold_hits), 0) AS hits, COALESCE(SUM(cold_misses), 0) AS misses,
              COALESCE(SUM(cold_bytes_written), 0) AS bytesWritten,
              COALESCE(SUM(cold_bytes_restored), 0) AS bytesRestored
       FROM traces WHERE ts > 0 GROUP BY day ORDER BY day`,
    )
    .all()
    .map((row) => ({
      day: String(row.day),
      hits: toInt(row.hits),
      misses: toInt(row.misses),
      bytesWritten: toInt(row.bytesWritten),
      bytesRestored: toInt(row.bytesRestored),
    }));
  sendJson(res, 200, { disk, trend });
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
