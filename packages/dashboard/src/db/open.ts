import { DatabaseSync } from 'node:sqlite';

import { drizzle, type NodeSQLiteDatabase } from 'drizzle-orm/node-sqlite';

export interface DashboardDb {
  db: NodeSQLiteDatabase;
  sqlite: DatabaseSync;
  close: () => void;
}

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
CREATE INDEX IF NOT EXISTS idx_traces_trace_id ON traces (trace_id);
`;

/**
 * Open (and bootstrap) the disposable SQLite index used by the dashboard.
 *
 * The schema is created idempotently via CREATE TABLE IF NOT EXISTS, so deleting
 * the file loses nothing — it is rebuilt from JSONL on next open. Pass ':memory:'
 * for an ephemeral in-process index.
 */
export function openDashboardDb(path: string): DashboardDb {
  const sqlite = new DatabaseSync(path);
  sqlite.exec(DDL);
  const db = drizzle({ client: sqlite });
  return {
    db,
    sqlite,
    close: () => sqlite.close(),
  };
}
