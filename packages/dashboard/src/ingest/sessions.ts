import { existsSync, lstatSync, readFileSync, readdirSync, statSync, type Dirent, type Stats } from 'node:fs';
import { join } from 'node:path';

import {
  buildContextEntries,
  migrateSessionEntries,
  parseSessionEntries,
  type FileEntry,
  type SessionEntry,
  type SessionHeader,
  type SessionInfoEntry,
  type SessionMessageEntry,
} from '@earendil-works/pi-coding-agent';
import { and, eq, ne } from 'drizzle-orm';

import type { DashboardDb } from '../db/open.js';
import { sessions, turns } from '../db/schema.js';
import { agentSessionsRoot } from '../paths.js';

export interface SessionIngestResult {
  scanned: number;
  updated: number;
  removed: number;
  /** Human-readable notes for files that were skipped (malformed / unreadable). */
  warnings: string[];
}

/**
 * Structural view of a persisted pi message. The custom `mlxTraceId` field is
 * stamped by our provider (B1) and is absent from pi's own `AgentMessage` type,
 * so we read the message defensively rather than through the union.
 */
interface ParsedMessage {
  role?: string;
  content?: unknown;
  model?: unknown;
  mlxTraceId?: unknown;
  usage?: {
    input?: unknown;
    output?: unknown;
    cacheRead?: unknown;
    reasoning?: unknown;
  };
}

function numOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function parseTs(value: unknown): number {
  if (typeof value !== 'string') return 0;
  const ms = Date.parse(value);
  return Number.isNaN(ms) ? 0 : ms;
}

/** First user message's text, string content or first text block, capped at 200 chars. */
function firstUserText(content: unknown): string | null {
  if (typeof content === 'string') return content.slice(0, 200);
  if (Array.isArray(content)) {
    for (const block of content) {
      if (
        block &&
        typeof block === 'object' &&
        (block as { type?: unknown }).type === 'text' &&
        typeof (block as { text?: unknown }).text === 'string'
      ) {
        return (block as { text: string }).text.slice(0, 200);
      }
    }
  }
  return null;
}

type TurnRow = typeof turns.$inferInsert;

interface DerivedSession {
  id: string;
  cwd: string;
  name: string | null;
  created: number;
  messageCount: number;
  firstMessage: string | null;
  turnRows: TurnRow[];
}

/**
 * The active, compaction-aware branch both the index and the detail API render —
 * pi's `buildContextEntries` follows the current leaf (the last appended entry)
 * back to the root, dropping abandoned tree branches. When that natural leaf is
 * metadata-only (e.g. a detached `session_info` leaf appended by a rename), it
 * carries no messages; rather than flattening EVERY branch (which resurrects
 * superseded turns), re-project from the latest message-bearing leaf so only its
 * ancestors are indexed. A file with no message entries at all projects to none.
 *
 * Caller must have validated topology (`isValidSessionTopology`) first: this
 * walks the parent chain, which would not terminate on a cyclic tree.
 */
export function activeBranchEntries(entries: FileEntry[]): SessionEntry[] {
  const sessionEntries = entries.filter((e): e is SessionEntry => e.type !== 'session');
  const active = buildContextEntries(sessionEntries);
  if (active.some((e) => e.type === 'message')) return active;
  let lastMessageId: string | null = null;
  for (const entry of sessionEntries) {
    if (entry.type === 'message') lastMessageId = entry.id;
  }
  if (lastMessageId === null) return [];
  return buildContextEntries(sessionEntries, lastMessageId);
}

/**
 * Whether every parsed record is a non-null object. A syntactically-valid but
 * non-object line (`null`) is kept verbatim by `parseSessionEntries`, and pi's
 * migrator dereferences records without guarding: `migrateToCurrentVersion`
 * locates the header with `entries.find((e) => e.type === 'session')`, and
 * `migrateV1ToV2`/`migrateV2ToV3` re-walk EVERY record. So a top-level `null`
 * throws a TypeError from `migrateSessionEntries` whenever it sits before the
 * header, or anywhere at all once the header is legacy (v1/v2) or missing — the
 * `find` short-circuit only protects a current-version file whose header is the
 * first record. `isValidSessionTopology` rejects such a record too, but it needs
 * the ids migration assigns and so can only run afterwards; screen with this
 * first so the migrator is never handed a record it will dereference and throw on.
 */
function hasOnlyObjectRecords(entries: FileEntry[]): boolean {
  for (const entry of entries) {
    if (entry === null || typeof entry !== 'object') return false;
  }
  return true;
}

/** The session header, tolerating the non-object records pi's parser keeps verbatim. */
export function findSessionHeader(entries: FileEntry[]): SessionHeader | undefined {
  for (const entry of entries) {
    if (entry === null || typeof entry !== 'object') continue;
    if (entry.type === 'session') return entry as SessionHeader;
  }
  return undefined;
}

/**
 * Whether the session entry tree is safe to project. pi's branch walker follows
 * `parentId` from the leaf to a root with no visited-set, so a self-parented
 * entry, a multi-entry cycle, or a duplicate id (which aliases an ancestor) grows
 * an unbounded path → hang/OOM. One damaged file is otherwise retried on every
 * scan, wedging ingest. Reject: non-string / duplicate ids, and any leaf whose
 * ancestor chain revisits an id or exceeds the entry count. A `parentId` that
 * references no in-file entry is a root terminator (matches the walker), not a
 * cycle.
 */
export function isValidSessionTopology(entries: FileEntry[]): boolean {
  // A syntactically-valid but non-object entry (`null`, a scalar, an array) has no
  // `type`/`id`/`parentId` to walk. `parseSessionEntries` keeps such lines verbatim
  // (a `header\nnull\n` file parses to `[header, null]`), so dereferencing one —
  // starting with the `.type` filter just below — throws a TypeError that escapes
  // every ingest quarantine branch, leaving the stale rows to feed the overview
  // forever. Reject the whole file's topology BEFORE any `.type`/`.id` access so the
  // caller routes it into the existing topology-quarantine branch (removing the
  // rows), and so the downstream `activeBranchEntries`/`deriveSession` projections
  // — which would throw the same way — are never reached. This guard runs only
  // after migration (it needs the ids that assigns), so callers must ALSO screen
  // with `hasOnlyObjectRecords` first — pi's migrator derefs the same records.
  if (!hasOnlyObjectRecords(entries)) return false;
  for (const entry of entries) {
    // A message entry whose `message` payload is itself a non-object (`null`, a
    // scalar) passes the object check above but throws on the downstream
    // `msg.role`/`msg.content` deref in `deriveSession`/`mapTranscriptEntry`. That
    // TypeError escapes ingest's per-file quarantine (so a file corrupted after
    // indexing keeps its stale session/turn rows) and 500s the detail handler.
    // Reject it here so the caller routes the file into the same topology
    // quarantine, and the projections that would throw are never reached.
    if ((entry as { type?: unknown }).type === 'message') {
      const message = (entry as { message?: unknown }).message;
      if (message === null || typeof message !== 'object') return false;
    }
  }
  const sessionEntries = entries.filter((e): e is SessionEntry => e.type !== 'session');
  const byId = new Map<string, SessionEntry>();
  for (const entry of sessionEntries) {
    if (typeof entry.id !== 'string') return false;
    if (byId.has(entry.id)) return false;
    byId.set(entry.id, entry);
  }
  // Amortized O(n): once an id is proven to reach a root terminator without a
  // cycle it is memoized in `safe`, so a shared ancestor spine is walked once
  // total rather than re-walked per descendant (the naive per-entry walk is
  // O(n²) and stalls the event loop on a ~20k-entry session).
  const safe = new Set<string>();
  for (const start of sessionEntries) {
    if (safe.has(start.id)) continue;
    const path = new Set<string>();
    const chain: string[] = [];
    let current: SessionEntry | undefined = start;
    while (current) {
      if (safe.has(current.id)) break;
      if (path.has(current.id)) return false;
      path.add(current.id);
      chain.push(current.id);
      const parentId = current.parentId;
      if (parentId === null || parentId === undefined) break;
      const parent = byId.get(parentId);
      if (parent === undefined) break;
      current = parent;
    }
    for (const id of chain) safe.add(id);
  }
  return true;
}

/** Fold parsed file entries into the row shapes the index stores. Returns null when unusable. */
function deriveSession(entries: FileEntry[]): DerivedSession | null {
  const header = entries.find((e) => e.type === 'session') as SessionHeader | undefined;
  if (!header || typeof header.id !== 'string') return null;

  // Session name is session-level metadata (the rename API appends a
  // `session_info` leaf); track the latest across ALL entries so a detached
  // info line still names the session.
  let name: string | null = null;
  for (const entry of entries) {
    if (entry.type === 'session_info') {
      const infoName = (entry as SessionInfoEntry).name;
      if (typeof infoName === 'string') name = infoName;
    }
  }

  // Token/turn metrics and counts must match the rendered transcript, which
  // follows the active branch — not abandoned branches in the append-only tree.
  let firstMessage: string | null = null;
  let messageCount = 0;
  const turnRows: TurnRow[] = [];
  for (const entry of activeBranchEntries(entries)) {
    if (entry.type !== 'message') continue;
    messageCount++;
    const msg = (entry as SessionMessageEntry).message as unknown as ParsedMessage;
    if (msg.role === 'user' && firstMessage === null) {
      firstMessage = firstUserText(msg.content);
    }
    if (msg.role === 'assistant' && msg.usage) {
      turnRows.push({
        sessionId: header.id,
        entryId: (entry as SessionMessageEntry).id,
        traceId: typeof msg.mlxTraceId === 'string' ? msg.mlxTraceId : null,
        ts: parseTs((entry as SessionMessageEntry).timestamp),
        model: typeof msg.model === 'string' ? msg.model : null,
        inputTokens: numOrNull(msg.usage.input),
        outputTokens: numOrNull(msg.usage.output),
        cachedTokens: numOrNull(msg.usage.cacheRead),
        reasoningTokens: numOrNull(msg.usage.reasoning),
      });
    }
  }

  return {
    id: header.id,
    cwd: typeof header.cwd === 'string' ? header.cwd : '',
    name,
    created: parseTs(header.timestamp),
    messageCount,
    firstMessage,
    turnRows,
  };
}

/** Count non-blank JSONL lines the way pi's `parseSessionEntries` iterates them. */
export function countJsonlLines(raw: string): number {
  const trimmed = raw.trim();
  if (trimmed === '') return 0;
  let count = 0;
  for (const line of trimmed.split('\n')) {
    if (line.trim() !== '') count++;
  }
  return count;
}

/** Whether the last non-blank JSONL line is itself valid JSON (an incomplete trailing write is not). */
export function lastLineParses(raw: string): boolean {
  const trimmed = raw.trim();
  if (trimmed === '') return true;
  const lines = trimmed.split('\n').filter((line) => line.trim() !== '');
  try {
    JSON.parse(lines[lines.length - 1]);
    return true;
  } catch {
    return false;
  }
}

/**
 * Read a pi session file into the in-memory entry list the index and the detail
 * API both project from — READ-ONLY. `parseSessionEntries` does not migrate, so
 * legacy v1 message entries carry no id/parentId the topology guard and branch
 * walker require; migrate in place (v1→v3) exactly as `ingestSessions` does. This
 * assigns id/parentId in memory only and never rewrites the file, so a plain read
 * of a v1 or partially-corrupt session cannot persist the migration or drop
 * malformed lines the way `SessionManager.open` (which opens for write) would.
 * Read/parse errors propagate to the caller.
 *
 * A file carrying a non-object record is returned UNMIGRATED rather than throwing
 * out of pi's migrator: the caller gates on `isValidSessionTopology`, which rejects
 * it and reports the same "invalid session tree" verdict ingest does, instead of
 * surfacing a raw TypeError as the transcript error.
 */
export function readSessionEntries(path: string): FileEntry[] {
  const raw = readFileSync(path, 'utf8');
  const entries = parseSessionEntries(raw);
  if (hasOnlyObjectRecords(entries)) migrateSessionEntries(entries);
  return entries;
}

/**
 * How the on-disk session file at `path` relates to `expectedId`. The delete guard
 * must tell three "not this session" cases apart, which a plain boolean cannot:
 *   - `'missing'`      — no file on disk; there is nothing to unlink, so dropping the
 *                        stale index row is safe.
 *   - `'different'`    — a readable file whose header names ANOTHER session: the path
 *                        was reused by a newer session B. B's file must be left
 *                        untouched, but the stale row for id A can be dropped.
 *   - `'unverifiable'` — the file exists but cannot be read, cannot be parsed, or
 *                        carries no header, so we cannot prove whose it is. Deleting
 *                        the row while leaving the file desyncs the DB from disk — the
 *                        transcript can re-index on a later successful ingest (so it
 *                        reappears after we reported it deleted) or linger as an
 *                        orphan — so the caller treats this as a conflict, not a
 *                        delete. Never delete on doubt.
 *   - `'matches'`      — the header still names THIS session; safe to unlink.
 */
export function classifySessionFile(
  path: string,
  expectedId: string,
): 'missing' | 'matches' | 'different' | 'unverifiable' {
  if (!existsSync(path)) return 'missing';
  let raw: string;
  try {
    raw = readFileSync(path, 'utf8');
  } catch {
    return 'unverifiable';
  }
  let entries: FileEntry[];
  try {
    entries = parseSessionEntries(raw);
  } catch {
    return 'unverifiable';
  }
  const header = findSessionHeader(entries);
  if (header === undefined) return 'unverifiable';
  return header.id === expectedId ? 'matches' : 'different';
}

/**
 * Whether the session file at `path` currently parses to `expectedId`. A thin
 * exact-match wrapper over `classifySessionFile` (one source of truth) for callers
 * that only proceed when the file provably belongs to this session — the rename
 * handler, which must refuse to mutate any file it cannot verify. A
 * missing/unreadable/headerless/foreign file returns `false`.
 */
export function verifySessionFileId(path: string, expectedId: string): boolean {
  return classifySessionFile(path, expectedId) === 'matches';
}

/**
 * Every session `.jsonl` file under `root`, in directory-listing order, across
 * BOTH pi layouts:
 *   - the project-scoped layout `<root>/--<cwd>--/*.jsonl` (pi's default), and
 *   - the flat explicit-session-dir layout `<root>/*.jsonl` (what pi writes with
 *     `--session-dir X`, which `mlx dashboard --session-dir` is documented to
 *     point at) — root-level files that would otherwise be ignored.
 * The two sets are disjoint (a subdir file is never a root-level file), so no file
 * is double-counted. Every candidate is `lstatSync`'d and kept only if it is a
 * regular file (`isFile()` is false for a symlink), and a root-level symlinked
 * directory is never descended — so a symlinked transcript is never indexed and an
 * external Pi session cannot be surfaced under the target's id, matching the native
 * cold tier's no-follow policy.
 *
 * Returns `null` when the ROOT itself could not be listed — a state the caller must
 * tell apart from an empty result, which reconciliation reads as "every session was
 * deleted".
 */
function listSessionFiles(root: string): string[] | null {
  if (!existsSync(root)) return [];
  const files: string[] = [];

  const pushIfRegularJsonl = (full: string): void => {
    if (!full.endsWith('.jsonl')) return;
    let stat: Stats;
    try {
      stat = lstatSync(full);
    } catch {
      return;
    }
    if (stat.isFile()) files.push(full);
  };

  // The root listing itself can fail where `existsSync` cannot see it: EACCES on a
  // chmod'd/ACL'd root, ENOTDIR when `--session-dir` names a transcript instead of
  // its directory, EMFILE under fd pressure, or ENOENT racing another process. Any
  // of those leaves the root's contents UNKNOWN, which is not the same as empty —
  // hence `null` rather than `[]`.
  let rootEntries: Dirent[];
  try {
    rootEntries = readdirSync(root, { withFileTypes: true });
  } catch {
    return null;
  }

  for (const entry of rootEntries) {
    // `--<cwd>--/` project subdir: scan the `.jsonl` files inside it.
    if (entry.isDirectory()) {
      if (!entry.name.startsWith('--') || !entry.name.endsWith('--')) continue;
      const dirPath = join(root, entry.name);
      // Isolate a single unreadable / vanished project subdir (chmod, or a TOCTOU
      // race after the root listing) like the per-item guards around it: skip it
      // rather than throw out of the whole scan, which would abort ingestion for
      // every sibling session AND skip the subsequent trace ingest / metrics pass.
      let names: string[];
      try {
        names = readdirSync(dirPath);
      } catch {
        continue;
      }
      for (const name of names) pushIfRegularJsonl(join(dirPath, name));
      continue;
    }
    // Flat explicit-session-dir layout: a `.jsonl` directly under the root. The
    // same no-follow guard applies — `pushIfRegularJsonl` lstat-skips a symlink.
    pushIfRegularJsonl(join(root, entry.name));
  }
  return files;
}

/**
 * Incrementally index every pi session JSONL under `root` into the SQLite
 * index. Files whose stored mtime+size are unchanged are skipped; changed files
 * have their session row upserted and their turn rows replaced atomically. DB
 * rows whose backing file has vanished are removed. Never throws for a bad file
 * — it is skipped and noted in `warnings`.
 */
export async function ingestSessions(dash: DashboardDb, root?: string): Promise<SessionIngestResult> {
  const { db, sqlite } = dash;
  const sessionRoot = root ?? agentSessionsRoot();
  const warnings: string[] = [];
  let scanned = 0;
  let updated = 0;
  let removed = 0;

  // Drop any previously-indexed rows for a path whose CURRENT content is
  // definitively invalid (broken topology or no header), so a transcript swapped
  // to garbage stops feeding the overview and session lists forever. Mirrors the
  // transactional delete reconciliation and stale-path handling use. Returns
  // whether any rows were actually removed.
  const quarantinePath = (path: string): boolean => {
    const rowsOnPath = db.select({ id: sessions.id }).from(sessions).where(eq(sessions.path, path)).all();
    if (rowsOnPath.length === 0) return false;
    sqlite.exec('BEGIN');
    try {
      for (const stale of rowsOnPath) {
        db.delete(turns).where(eq(turns.sessionId, stale.id)).run();
      }
      db.delete(sessions).where(eq(sessions.path, path)).run();
      sqlite.exec('COMMIT');
      return true;
    } catch {
      sqlite.exec('ROLLBACK');
      return false;
    }
  };

  // Snapshot the current root's discoverable files ONCE, up front. The scan
  // iterates it, and reconciliation (below) retains a row only if its path is in
  // this set — i.e. a regular file UNDER THE CURRENT root. The shared DB is not
  // keyed by session root, so after switching --session-dir A→B the old root's
  // files still exist on disk; a plain "is a regular file anywhere" check would
  // keep A's out-of-root rows (whose detail/rename/delete all 403). Unchanged
  // files skipped by the mtime/size watermark are still in this set, so valid
  // in-root rows are never wrongly dropped.
  const discoverable = listSessionFiles(sessionRoot);
  if (discoverable === null) {
    // The root could not be listed, so nothing about the indexed rows can be
    // proven stale — reconciling against an unknown set would wipe the whole
    // session index on one chmod or a transient EMFILE. Warn and skip. Returning
    // normally (rather than throwing) also keeps the caller on its happy path, so
    // the trace ingest and its 30-day prune still run this pass.
    warnings.push(`${sessionRoot}: session root could not be listed; scan skipped`);
    return { scanned, updated, removed, warnings };
  }
  const discoverableSet = new Set(discoverable);

  for (const filePath of discoverable) {
    scanned++;
    try {
      const stat = statSync(filePath);
      const mtime = Math.floor(stat.mtimeMs);
      const size = stat.size;

      const existing = db
        .select({ mtime: sessions.lastIngestedMtime, size: sessions.lastIngestedSize })
        .from(sessions)
        .where(eq(sessions.path, filePath))
        .all();
      if (existing.length > 0 && existing[0].mtime === mtime && existing[0].size === size) {
        continue;
      }

      let raw: string;
      try {
        raw = readFileSync(filePath, 'utf8');
      } catch (err) {
        warnings.push(`${filePath}: read failed (${String(err)})`);
        continue;
      }
      let entries: FileEntry[];
      try {
        entries = parseSessionEntries(raw);
      } catch (err) {
        warnings.push(`${filePath}: parse failed (${String(err)})`);
        continue;
      }
      // `parseSessionEntries` does not migrate: genuine legacy v1 message entries
      // carry no id/parentId, which the topology guard and branch walker require.
      // Migrate in place (v1→v3) before validating or deriving. This assigns
      // id/parentId in memory only (no disk write) and never changes the entry
      // count, so the `countJsonlLines` completeness check below still holds.
      // The migrator itself derefs records unguarded, and it runs before the
      // topology guard can reject a non-object one, so screen for that here —
      // otherwise the TypeError lands in the per-file catch below, which only
      // warns, and the file keeps its stale rows on every later scan.
      const migratable = hasOnlyObjectRecords(entries);
      if (migratable) migrateSessionEntries(entries);

      // Quarantine a topologically broken tree (cycle / duplicate ids) BEFORE any
      // branch projection: the walker would otherwise loop unbounded and OOM.
      if (!migratable || !isValidSessionTopology(entries)) {
        warnings.push(`${filePath}: invalid session tree (cycle, duplicate id, or non-object message); skipped`);
        if (quarantinePath(filePath)) removed++;
        continue;
      }

      const derived = deriveSession(entries);
      if (!derived) {
        warnings.push(`${filePath}: no valid session header`);
        if (quarantinePath(filePath)) removed++;
        continue;
      }

      // pi's parser silently drops malformed JSONL lines. If more non-blank
      // lines exist than entries parsed, the file is truncated/corrupt.
      const droppedLines = countJsonlLines(raw) - entries.length;
      const complete = droppedLines <= 0;
      // Only a single non-parsing FINAL line is a legitimate live-append
      // in-progress prefix (`droppedLines === 1` with the last line failing to
      // parse ⇒ the dropped line IS the last line). Anything else — an interior
      // drop, or more than one dropped line — means pi silently discarded a
      // record and kept going: an orphaned child whose parent was dropped is
      // accepted by the topology guard as a false root, so the derived
      // transcript is a corrupt subset. That must NOT be indexed as a real
      // session; quarantine any stale rows and skip, mirroring the
      // topology/no-header branches.
      const trailingOnly = droppedLines === 1 && !lastLineParses(raw);
      if (!complete && !trailingOnly) {
        warnings.push(`${filePath}: malformed session records; skipped`);
        if (quarantinePath(filePath)) removed++;
        continue;
      }
      if (!complete) {
        // trailingOnly: index what parsed but do NOT stamp the current
        // mtime/size as fully-ingested, so a later completed write re-ingests
        // instead of being skipped.
        warnings.push(
          `${filePath}: incomplete trailing line; indexed ${entries.length} entries, not marking fully-ingested`,
        );
      }

      sqlite.exec('BEGIN');
      try {
        // A path holds exactly one current session. If a prior row recorded
        // THIS path under a different id (the file was replaced by a new
        // session), drop it first so a later delete of the stale id cannot
        // rmSync the new session's file.
        const staleOnPath = db
          .select({ id: sessions.id })
          .from(sessions)
          .where(and(eq(sessions.path, filePath), ne(sessions.id, derived.id)))
          .all();
        for (const stale of staleOnPath) {
          db.delete(turns).where(eq(turns.sessionId, stale.id)).run();
          db.delete(sessions).where(eq(sessions.id, stale.id)).run();
        }
        db.delete(turns).where(eq(turns.sessionId, derived.id)).run();
        db.delete(sessions).where(eq(sessions.id, derived.id)).run();
        db.insert(sessions)
          .values({
            id: derived.id,
            path: filePath,
            cwd: derived.cwd,
            name: derived.name,
            created: derived.created,
            modified: mtime,
            messageCount: derived.messageCount,
            firstMessage: derived.firstMessage,
            lastIngestedMtime: complete ? mtime : 0,
            lastIngestedSize: complete ? size : 0,
          })
          .run();
        if (derived.turnRows.length > 0) {
          db.insert(turns).values(derived.turnRows).run();
        }
        sqlite.exec('COMMIT');
        updated++;
      } catch (err) {
        sqlite.exec('ROLLBACK');
        warnings.push(`${filePath}: write failed (${String(err)})`);
      }
    } catch (err) {
      warnings.push(`${filePath}: ${String(err)}`);
    }
  }

  // A row survives reconciliation only while its path is in the CURRENT root's
  // discoverable set (`listSessionFiles`, which no-follow-`lstat`s each candidate
  // and keeps only regular files). This is strictly stronger than an
  // "is a regular file anywhere" check: it also drops a row whose file was
  // swapped for an in-root symlink (`existsSync` would follow it and keep serving
  // a foreign transcript) AND a row left behind by switching --session-dir to a
  // different root (the old root's file still exists but is no longer discoverable).
  const known = db.select({ id: sessions.id, path: sessions.path }).from(sessions).all();
  for (const row of known) {
    if (discoverableSet.has(row.path)) continue;
    sqlite.exec('BEGIN');
    try {
      db.delete(turns).where(eq(turns.sessionId, row.id)).run();
      db.delete(sessions).where(eq(sessions.id, row.id)).run();
      sqlite.exec('COMMIT');
      removed++;
    } catch {
      sqlite.exec('ROLLBACK');
    }
  }

  return { scanned, updated, removed, warnings };
}
