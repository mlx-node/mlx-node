import { existsSync, lstatSync, readFileSync, readdirSync, statSync, type Stats } from 'node:fs';
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
 */
export function readSessionEntries(path: string): FileEntry[] {
  const raw = readFileSync(path, 'utf8');
  const entries = parseSessionEntries(raw);
  migrateSessionEntries(entries);
  return entries;
}

/**
 * Whether the session file at `path` currently parses to `expectedId`. Used as a
 * last-defense guard before deleting a session file: a stale index row for id A
 * must not `rmSync` a file whose path was reused by a newer session B. A
 * missing/unreadable/headerless file returns `false` — never delete on doubt.
 */
export function verifySessionFileId(path: string, expectedId: string): boolean {
  let raw: string;
  try {
    raw = readFileSync(path, 'utf8');
  } catch {
    return false;
  }
  let entries: FileEntry[];
  try {
    entries = parseSessionEntries(raw);
  } catch {
    return false;
  }
  const header = entries.find((e) => e.type === 'session') as SessionHeader | undefined;
  return header !== undefined && header.id === expectedId;
}

/**
 * All `<root>/--*--/*.jsonl` session files, in directory-listing order. The
 * outer `Dirent.isDirectory()` already skips a symlinked project dir; each
 * candidate `.jsonl` is then `lstatSync`'d and skipped unless it is a regular
 * file (`isFile()` is false for a symlink). A symlinked transcript is never
 * indexed, so an external Pi session cannot be surfaced under the target's id
 * — matching the native cold tier's no-follow policy.
 */
function listSessionFiles(root: string): string[] {
  if (!existsSync(root)) return [];
  const files: string[] = [];
  for (const dir of readdirSync(root, { withFileTypes: true })) {
    if (!dir.isDirectory()) continue;
    if (!dir.name.startsWith('--') || !dir.name.endsWith('--')) continue;
    const dirPath = join(root, dir.name);
    for (const name of readdirSync(dirPath)) {
      if (!name.endsWith('.jsonl')) continue;
      const full = join(dirPath, name);
      let stat: Stats;
      try {
        stat = lstatSync(full);
      } catch {
        continue;
      }
      if (!stat.isFile()) continue;
      files.push(full);
    }
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

  for (const filePath of listSessionFiles(sessionRoot)) {
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
      migrateSessionEntries(entries);

      // Quarantine a topologically broken tree (cycle / duplicate ids) BEFORE any
      // branch projection: the walker would otherwise loop unbounded and OOM.
      if (!isValidSessionTopology(entries)) {
        warnings.push(`${filePath}: invalid session tree (cycle or duplicate entry id); skipped`);
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
      // lines exist than entries parsed, the file is truncated/corrupt: index
      // what parsed but do NOT stamp the current mtime/size as fully-ingested,
      // so a later completed write re-ingests instead of being skipped.
      const droppedLines = countJsonlLines(raw) - entries.length;
      const complete = droppedLines <= 0;
      if (!complete) {
        const kind =
          droppedLines === 1 && !lastLineParses(raw) ? 'incomplete trailing line' : `${droppedLines} malformed line(s)`;
        warnings.push(`${filePath}: ${kind}; indexed ${entries.length} entries, not marking fully-ingested`);
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

  // A row survives reconciliation only while its path is still a REGULAR file —
  // the same lstat predicate the scanner uses. `existsSync` follows a symlink, so
  // a row whose file was swapped for an in-root symlink to another transcript
  // would otherwise persist and serve a foreign session's content on detail.
  const stillPresent = (p: string): boolean => {
    try {
      return lstatSync(p).isFile();
    } catch {
      return false;
    }
  };

  const known = db.select({ id: sessions.id, path: sessions.path }).from(sessions).all();
  for (const row of known) {
    if (stillPresent(row.path)) continue;
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
