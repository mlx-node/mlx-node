import { existsSync, renameSync, rmSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';

import { getTableColumns } from 'drizzle-orm';
import { drizzle, type NodeSQLiteDatabase } from 'drizzle-orm/node-sqlite';

import { sessions, traceFiles, traces, turns } from './schema.js';

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
const SCHEMA_VERSION = 5;

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
  queue_ms INTEGER,
  finish_reason TEXT,
  resident INTEGER,
  prompt_tokens INTEGER,
  cached_tokens INTEGER,
  output_tokens INTEGER,
  reasoning_tokens INTEGER,
  cold_hits INTEGER,
  cold_misses INTEGER,
  cold_bytes_written INTEGER,
  cold_bytes_restored INTEGER,
  cold_root TEXT,
  cold_enabled INTEGER,
  cold_enqueued INTEGER,
  cold_queue_drops INTEGER,
  cold_evictions INTEGER,
  cold_corruptions INTEGER,
  cold_corruptions_total INTEGER,
  cold_queue_drops_total INTEGER,
  cold_write_errors INTEGER,
  cold_write_errors_total INTEGER,
  cold_restore_declines INTEGER,
  cold_sidecar_capture_reached INTEGER,
  cold_sidecar_chain_empty INTEGER,
  cold_sidecar_boundary_skips INTEGER,
  cold_sidecar_already_persisted INTEGER,
  cold_sidecar_enqueued INTEGER,
  cold_sidecar_queue_drops INTEGER,
  cold_sidecar_installed INTEGER,
  cold_sidecar_restore_suppressed INTEGER,
  source_file TEXT
);

CREATE TABLE IF NOT EXISTS trace_files (
  name TEXT PRIMARY KEY,
  last_ingested_mtime INTEGER NOT NULL,
  last_ingested_size INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_session_id ON turns (session_id);
CREATE INDEX IF NOT EXISTS idx_traces_session_id ON traces (session_id);
CREATE INDEX IF NOT EXISTS idx_traces_root_session_id ON traces (root_session_id);
CREATE INDEX IF NOT EXISTS idx_traces_trace_id ON traces (trace_id);
CREATE INDEX IF NOT EXISTS idx_traces_cold_root ON traces (cold_root);
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
 *  - `user_version` other than SCHEMA_VERSION means a foreign schema — older
 *    (e.g. a `traces` table predating root_session_id) OR newer (a future schema
 *    this build predates). The index is disposable and rebuilt from JSONL, so an
 *    exact match is required; anything else is rebuilt rather than opened blind.
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
  if (version !== SCHEMA_VERSION) {
    throw new RebuildRequiredError(`schema version ${version} != ${SCHEMA_VERSION}`);
  }
}

/**
 * Complete expected column set per table, derived straight from the drizzle
 * schema so a column added to `schema.ts` (without a matching SCHEMA_VERSION
 * bump) is automatically part of this open-time contract. Keys are the SQLite
 * table names; values the full set of DB column names each must carry.
 */
const EXPECTED_COLUMNS: Record<string, string[]> = {
  sessions: Object.values(getTableColumns(sessions)).map((c) => c.name),
  turns: Object.values(getTableColumns(turns)).map((c) => c.name),
  traces: Object.values(getTableColumns(traces)).map((c) => c.name),
  trace_files: Object.values(getTableColumns(traceFiles)).map((c) => c.name),
};

/**
 * Verify every expected table carries every expected column on the UNTOUCHED
 * pre-existing db — this MUST run BEFORE the DDL. A matching `user_version` alone
 * does not prove the schema: `CREATE TABLE IF NOT EXISTS` recreates a dropped
 * table EMPTY and no-ops over a table missing a newer column, so a signature
 * check run after the DDL would inspect a state the DDL just fabricated (and the
 * watermark-gated ingest would never re-populate a silently recreated table).
 * `PRAGMA table_info` touches only the schema, never data; a missing table
 * returns zero rows (every column absent). Any gap → quarantine+rebuild.
 */
function assertSchemaSignature(sqlite: DatabaseSync): void {
  for (const [table, expected] of Object.entries(EXPECTED_COLUMNS)) {
    // `PRAGMA table_info` reports the columns of a VIEW and of an FTS5 VIRTUAL
    // table too, so either could clear the column check and then wedge startup
    // (`CREATE INDEX` throws "views may not be indexed" / "virtual tables may not
    // be indexed", both outside the rebuildable set). `sqlite_schema.type` cannot
    // tell them apart — it reports a virtual table as `'table'`. `PRAGMA
    // table_list` exposes a distinct `type` (ordinary → `'table'`, FTS5 →
    // `'virtual'`, its shadow tables → `'shadow'`, view → `'view'`), so requiring
    // `'table'` rejects virtual/shadow/view in one check; a missing object returns
    // undefined → rebuild (same path as before). Table names are fixed internal
    // constants, never user input — safe to inline.
    const meta = sqlite.prepare(`PRAGMA table_list(${table})`).get() as { type?: unknown } | undefined;
    if (meta?.type !== 'table') {
      const actualType = typeof meta?.type === 'string' ? meta.type : 'missing';
      throw new RebuildRequiredError(`schema signature mismatch: ${table} is ${actualType}, expected a table`);
    }
    // Table names are fixed internal constants, never user input — safe to inline.
    const rows = sqlite.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name?: unknown }>;
    const actual = new Set(rows.map((r) => (typeof r.name === 'string' ? r.name : '')));
    for (const col of expected) {
      if (!actual.has(col)) {
        throw new RebuildRequiredError(`schema signature mismatch: ${table}.${col} missing`);
      }
    }
  }
}

function openWithSchema(path: string): DatabaseSync {
  // Only validate a file that already existed; a fresh (or quarantined-then-
  // reopened) path starts empty and is bootstrapped below.
  const preExisting = path !== ':memory:' && existsSync(path);
  const sqlite = new DatabaseSync(path);
  try {
    // Validate the UNTOUCHED pre-existing db BEFORE any DDL: `CREATE TABLE IF NOT
    // EXISTS` would recreate a dropped table empty (masking data loss) or no-op
    // over a table missing a newer column, so a post-DDL probe inspects a
    // fabricated state. A fresh or just-rebuilt path (preExisting === false)
    // carries no drift and skips straight to DDL bootstrap.
    if (preExisting) {
      assertUsableSchema(sqlite);
      assertSchemaSignature(sqlite);
    }
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
    const rebuildable = err instanceof RebuildRequiredError || isCorruptionError(err) || isSchemaMismatchError(err);
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
