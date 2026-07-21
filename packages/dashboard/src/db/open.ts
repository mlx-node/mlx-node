import { existsSync, renameSync, rmSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';

import { drizzle, type NodeSQLiteDatabase } from 'drizzle-orm/node-sqlite';

export interface DashboardDb {
  db: NodeSQLiteDatabase;
  sqlite: DatabaseSync;
  close: () => void;
}

/**
 * On-disk schema revision. Bump on every DDL change: an existing db stamped
 * with an older version (or never stamped, i.e. 0) is quarantined+rebuilt
 * rather than migrated in place. The index is disposable and repopulated from
 * JSONL on boot, so a rebuild never loses source-of-truth data.
 */
const SCHEMA_VERSION = 1;

const DDL = `
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  cwd TEXT NOT NULL,
  name TEXT,
  created INTEGER NOT NULL,
  modified INTEGER NOT NULL,
  message_count INTEGER NOT NULL DEFAULT 0,
  first_message TEXT,
  last_ingested_mtime INTEGER NOT NULL DEFAULT 0,
  last_ingested_size INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS turns (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  entry_id TEXT,
  trace_id TEXT,
  ts INTEGER NOT NULL,
  model TEXT,
  input_tokens INTEGER,
  output_tokens INTEGER,
  cached_tokens INTEGER,
  reasoning_tokens INTEGER
);

CREATE TABLE IF NOT EXISTS traces (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  trace_id TEXT NOT NULL UNIQUE,
  session_id TEXT,
  root_session_id TEXT,
  ts INTEGER NOT NULL,
  model TEXT,
  ttft_ms REAL,
  prefill_tps REAL,
  decode_tps REAL,
  mtp_cycles INTEGER,
  mtp_mean_accepted REAL,
  duration_ms REAL,
  finish_reason TEXT,
  prompt_tokens INTEGER,
  cached_tokens INTEGER,
  output_tokens INTEGER,
  reasoning_tokens INTEGER,
  cold_hits INTEGER,
  cold_misses INTEGER,
  cold_bytes_written INTEGER,
  cold_bytes_restored INTEGER,
  source_file TEXT
);

CREATE INDEX IF NOT EXISTS idx_turns_session_id ON turns (session_id);
CREATE INDEX IF NOT EXISTS idx_traces_session_id ON traces (session_id);
CREATE INDEX IF NOT EXISTS idx_traces_root_session_id ON traces (root_session_id);
CREATE INDEX IF NOT EXISTS idx_traces_trace_id ON traces (trace_id);
`;

/** SQLite errors that mean the file on disk is not a usable database. */
const CORRUPTION_RE = /not a database|malformed|disk image|corrupt/i;

/** SQLite DDL errors that mean the on-disk schema predates this build. */
const SCHEMA_MISMATCH_RE = /no such (table|column)|has no column named/i;

function isCorruptionError(err: unknown): boolean {
  return CORRUPTION_RE.test(err instanceof Error ? err.message : String(err));
}

function isSchemaMismatchError(err: unknown): boolean {
  return SCHEMA_MISMATCH_RE.test(err instanceof Error ? err.message : String(err));
}

/** An existing db whose pages are damaged or whose schema is older than this
 * build expects. Both are unrecoverable for a disposable index → rebuild. */
class RebuildRequiredError extends Error {}

/**
 * Reject an existing db this build cannot safely use:
 *  - `quick_check` catches page-level damage OUTSIDE the schema pages. The DDL
 *    only no-ops on an existing schema and never walks data leaf pages, so such
 *    damage would otherwise slip past open and throw later at query time.
 *  - `user_version` below SCHEMA_VERSION means an older/foreign schema (e.g. a
 *    `traces` table predating root_session_id) the current DDL cannot extend in
 *    place — a stale index rebuilt in place would wedge on CREATE INDEX.
 * A non-database file makes `quick_check` throw "file is not a database", which
 * the caller treats as corruption; both routes lead to quarantine+rebuild.
 */
function assertUsableSchema(sqlite: DatabaseSync): void {
  const check = sqlite.prepare('PRAGMA quick_check').get() as { quick_check?: unknown } | undefined;
  const status = typeof check?.quick_check === 'string' ? check.quick_check : '';
  if (status.toLowerCase() !== 'ok') {
    throw new RebuildRequiredError(`quick_check failed: ${status || 'no result'}`);
  }
  const uv = sqlite.prepare('PRAGMA user_version').get() as { user_version?: unknown } | undefined;
  const version = typeof uv?.user_version === 'number' ? uv.user_version : 0;
  if (version < SCHEMA_VERSION) {
    throw new RebuildRequiredError(`schema version ${version} < ${SCHEMA_VERSION}`);
  }
}

function openWithSchema(path: string): DatabaseSync {
  // Only validate a file that already existed; a fresh (or quarantined-then-
  // reopened) path starts empty and is bootstrapped below.
  const preExisting = path !== ':memory:' && existsSync(path);
  const sqlite = new DatabaseSync(path);
  try {
    if (preExisting) assertUsableSchema(sqlite);
    sqlite.exec(DDL);
    sqlite.exec(`PRAGMA user_version = ${SCHEMA_VERSION};`);
  } catch (err) {
    try {
      sqlite.close();
    } catch {
      // Already unusable; nothing to salvage.
    }
    throw err;
  }
  return sqlite;
}

/**
 * Rename the corrupt db and its `-wal`/`-shm` sidecars aside (never silently
 * lost) so a fresh empty database can take the path. A sidecar we cannot rename
 * is removed instead — leaving a stale WAL beside a new db would re-corrupt it.
 */
function quarantineDbFiles(path: string): void {
  const stamp = Date.now();
  for (const suffix of ['', '-wal', '-shm']) {
    const p = path + suffix;
    if (!existsSync(p)) continue;
    try {
      renameSync(p, `${p}.corrupt-${stamp}`);
    } catch {
      try {
        rmSync(p, { force: true });
      } catch {
        // Best effort: fall through and let the reopen surface any real error.
      }
    }
  }
}

/**
 * Open (and bootstrap) the disposable SQLite index used by the dashboard.
 *
 * The schema is created idempotently via CREATE TABLE IF NOT EXISTS, so deleting
 * the file loses nothing — it is rebuilt from JSONL on next open. Pass ':memory:'
 * for an ephemeral in-process index.
 *
 * The index is disposable, so a file that cannot back the current schema must
 * not block startup. Three cases quarantine the file aside and recreate an
 * empty schema for boot ingest to repopulate: non-SQLite bytes / a malformed db
 * (corruption), page-level damage a `quick_check` flags on open, and an
 * older/incompatible on-disk schema (`user_version` below SCHEMA_VERSION, or a
 * DDL that references a column an earlier build never created). Non-recoverable
 * errors (permission, unrelated I/O) are rethrown — not a reason to discard data.
 */
export function openDashboardDb(path: string): DashboardDb {
  let sqlite: DatabaseSync;
  try {
    sqlite = openWithSchema(path);
  } catch (err) {
    const rebuildable =
      err instanceof RebuildRequiredError || isCorruptionError(err) || isSchemaMismatchError(err);
    if (path === ':memory:' || !rebuildable) throw err;
    quarantineDbFiles(path);
    sqlite = openWithSchema(path);
  }
  const db = drizzle({ client: sqlite });
  return {
    db,
    sqlite,
    close: () => sqlite.close(),
  };
}
