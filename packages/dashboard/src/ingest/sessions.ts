import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

import {
  parseSessionEntries,
  type FileEntry,
  type SessionHeader,
  type SessionInfoEntry,
  type SessionMessageEntry,
} from '@earendil-works/pi-coding-agent';
import { eq } from 'drizzle-orm';

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

/** Fold parsed file entries into the row shapes the index stores. Returns null when unusable. */
function deriveSession(entries: FileEntry[]): DerivedSession | null {
  const header = entries.find((e) => e.type === 'session') as SessionHeader | undefined;
  if (!header || typeof header.id !== 'string') return null;

  let name: string | null = null;
  let firstMessage: string | null = null;
  let messageCount = 0;
  const turnRows: TurnRow[] = [];

  for (const entry of entries) {
    if (entry.type === 'session_info') {
      const infoName = (entry as SessionInfoEntry).name;
      if (typeof infoName === 'string') name = infoName;
      continue;
    }
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

/** All `<root>/--*--/*.jsonl` session files, in directory-listing order. */
function listSessionFiles(root: string): string[] {
  if (!existsSync(root)) return [];
  const files: string[] = [];
  for (const dir of readdirSync(root, { withFileTypes: true })) {
    if (!dir.isDirectory()) continue;
    if (!dir.name.startsWith('--') || !dir.name.endsWith('--')) continue;
    const dirPath = join(root, dir.name);
    for (const name of readdirSync(dirPath)) {
      if (name.endsWith('.jsonl')) files.push(join(dirPath, name));
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

      let entries: FileEntry[];
      try {
        entries = parseSessionEntries(readFileSync(filePath, 'utf8'));
      } catch (err) {
        warnings.push(`${filePath}: parse failed (${String(err)})`);
        continue;
      }

      const derived = deriveSession(entries);
      if (!derived) {
        warnings.push(`${filePath}: no valid session header`);
        continue;
      }

      sqlite.exec('BEGIN');
      try {
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
            lastIngestedMtime: mtime,
            lastIngestedSize: size,
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

  let removed = 0;
  const known = db.select({ id: sessions.id, path: sessions.path }).from(sessions).all();
  for (const row of known) {
    if (existsSync(row.path)) continue;
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
