import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';

import { describe, expect, it } from 'vite-plus/test';

import { openDashboardDb } from '../src/db/open.js';
import { sessions, traces, turns } from '../src/db/schema.js';

describe('dashboard db', () => {
  it('bootstraps schema and round-trips a session row', () => {
    const { db, close } = openDashboardDb(':memory:');
    db.insert(sessions)
      .values({
        id: 's1',
        path: '/tmp/s1.jsonl',
        cwd: '/w',
        name: null,
        created: 1,
        modified: 2,
        messageCount: 3,
        firstMessage: 'hi',
        lastIngestedMtime: 0,
        lastIngestedSize: 0,
      })
      .run();
    const rows = db.select().from(sessions).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].firstMessage).toBe('hi');
    close();
  });
  it('bootstraps idempotently on an existing db file', () => {
    const file = join(tmpdir(), `dash-${process.pid}-${Date.now()}.db`);
    const first = openDashboardDb(file);
    first.db
      .insert(sessions)
      .values({
        id: 's1',
        path: '/tmp/s1.jsonl',
        cwd: '/w',
        name: null,
        created: 1,
        modified: 2,
        messageCount: 0,
        firstMessage: null,
        lastIngestedMtime: 0,
        lastIngestedSize: 0,
      })
      .run();
    first.close();
    const second = openDashboardDb(file);
    expect(second.db.select().from(sessions).all()).toHaveLength(1);
    second.close();
    rmSync(file, { force: true });
  });

  // Finding 10: a corrupt disposable index must not block startup.
  it('quarantines a corrupt db and boots a fresh empty schema', () => {
    const d = mkdtempSync(join(tmpdir(), 'dash-corrupt-'));
    const file = join(d, 'index.db');
    writeFileSync(file, 'this is not a sqlite database, just junk bytes '.repeat(20));

    const dash = openDashboardDb(file);
    // Fresh, empty, usable schema.
    expect(dash.db.select().from(sessions).all()).toHaveLength(0);
    dash.db
      .insert(sessions)
      .values({
        id: 's1',
        path: '/tmp/s1.jsonl',
        cwd: '/w',
        name: null,
        created: 1,
        modified: 2,
        messageCount: 0,
        firstMessage: null,
        lastIngestedMtime: 0,
        lastIngestedSize: 0,
      })
      .run();
    expect(dash.db.select().from(sessions).all()).toHaveLength(1);
    dash.close();

    // The junk is quarantined aside (not silently lost); the path is now a real db.
    const quarantined = readdirSync(d).filter((n) => n.startsWith('index.db.corrupt-'));
    expect(quarantined).toHaveLength(1);
    expect(readFileSync(file).subarray(0, 15).toString('utf-8')).toContain('SQLite format 3');

    rmSync(d, { recursive: true, force: true });
  });

  // Finding D: an existing db with an older/incompatible schema (created before
  // the traces.root_session_id column existed) must not wedge startup — the
  // current DDL's CREATE INDEX on the missing column would otherwise throw.
  it('quarantines an old-schema traces db (missing root_session_id) and rebuilds', () => {
    const d = mkdtempSync(join(tmpdir(), 'dash-oldschema-'));
    const file = join(d, 'index.db');
    // Build the OLD traces schema by hand: no root_session_id column, unstamped.
    const raw = new DatabaseSync(file);
    raw.exec(
      `CREATE TABLE traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trace_id TEXT NOT NULL UNIQUE,
        session_id TEXT,
        ts INTEGER NOT NULL
      );`,
    );
    raw.close();

    const dash = openDashboardDb(file);
    // Current schema is live: an insert using root_session_id round-trips.
    dash.db.insert(traces).values({ traceId: 't1', sessionId: 's1', rootSessionId: 'r1', ts: 1 }).run();
    const rows = dash.db.select().from(traces).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].rootSessionId).toBe('r1');
    dash.close();

    // The old db is quarantined aside, not silently lost.
    const quarantined = readdirSync(d).filter((n) => n.startsWith('index.db.corrupt-'));
    expect(quarantined).toHaveLength(1);
    rmSync(d, { recursive: true, force: true });
  });

  // Finding 8: a NEWER on-disk schema (a future build's index, stamped one
  // version ahead) is as unusable as an older one — an exact-match check must
  // reject it and rebuild, not open it blind.
  it('quarantines a newer-schema db (user_version above this build) and rebuilds', () => {
    const d = mkdtempSync(join(tmpdir(), 'dash-newer-'));
    const file = join(d, 'index.db');
    const seed = openDashboardDb(file);
    seed.close();
    const bump = new DatabaseSync(file);
    bump.exec('PRAGMA user_version = 2;'); // > SCHEMA_VERSION (1)
    bump.close();

    const dash = openDashboardDb(file);
    expect(dash.db.select().from(sessions).all()).toHaveLength(0);
    dash.db
      .insert(sessions)
      .values({
        id: 's1',
        path: '/tmp/s1.jsonl',
        cwd: '/w',
        name: null,
        created: 1,
        modified: 2,
        messageCount: 0,
        firstMessage: null,
        lastIngestedMtime: 0,
        lastIngestedSize: 0,
      })
      .run();
    expect(dash.db.select().from(sessions).all()).toHaveLength(1);
    dash.close();

    const quarantined = readdirSync(d).filter((n) => n.startsWith('index.db.corrupt-'));
    expect(quarantined).toHaveLength(1);
    rmSync(d, { recursive: true, force: true });
  });

  // Finding 8: a DDL change that drops/renames a column WITHOUT bumping
  // user_version passes the version check and no-ops under CREATE TABLE IF NOT
  // EXISTS. The signature probe must catch the missing column and rebuild.
  it('quarantines a column-drifted db (matching version) via the signature probe and rebuilds', () => {
    const d = mkdtempSync(join(tmpdir(), 'dash-drift-'));
    const file = join(d, 'index.db');
    const raw = new DatabaseSync(file);
    // Full sessions/turns, but traces is missing the newest `source_file` column.
    raw.exec(
      `CREATE TABLE sessions (
        id TEXT PRIMARY KEY, path TEXT NOT NULL, cwd TEXT NOT NULL, name TEXT,
        created INTEGER NOT NULL, modified INTEGER NOT NULL,
        message_count INTEGER NOT NULL DEFAULT 0, first_message TEXT,
        last_ingested_mtime INTEGER NOT NULL DEFAULT 0, last_ingested_size INTEGER NOT NULL DEFAULT 0
      );
      CREATE TABLE turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, entry_id TEXT,
        trace_id TEXT, ts INTEGER NOT NULL, model TEXT, input_tokens INTEGER,
        output_tokens INTEGER, cached_tokens INTEGER, reasoning_tokens INTEGER
      );
      CREATE TABLE traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT NOT NULL UNIQUE,
        session_id TEXT, root_session_id TEXT, ts INTEGER NOT NULL, model TEXT,
        cold_hits INTEGER, cold_bytes_restored INTEGER
      );`,
    );
    raw.exec('PRAGMA user_version = 1;'); // matches SCHEMA_VERSION → version check passes
    raw.close();

    const dash = openDashboardDb(file);
    // Rebuilt schema carries source_file; an insert using it round-trips.
    dash.db
      .insert(traces)
      .values({ traceId: 't1', sessionId: 's1', rootSessionId: 'r1', ts: 1, sourceFile: 'f.jsonl' })
      .run();
    const rows = dash.db.select().from(traces).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].sourceFile).toBe('f.jsonl');
    dash.close();

    const quarantined = readdirSync(d).filter((n) => n.startsWith('index.db.corrupt-'));
    expect(quarantined).toHaveLength(1);
    rmSync(d, { recursive: true, force: true });
  });

  // Finding 6: a pre-existing db missing a WHOLE table (matching version) must be
  // quarantined+rebuilt, not silently recreated empty. `CREATE TABLE IF NOT
  // EXISTS` would fabricate an empty `turns` that the watermark-gated ingest never
  // re-populates → per-turn metrics silently vanish. The signature check must run
  // BEFORE the DDL and catch the missing table.
  it('quarantines a db missing a whole table (turns) and rebuilds', () => {
    const d = mkdtempSync(join(tmpdir(), 'dash-notable-'));
    const file = join(d, 'index.db');
    const raw = new DatabaseSync(file);
    // Full sessions + traces, but NO turns table at all.
    raw.exec(
      `CREATE TABLE sessions (
        id TEXT PRIMARY KEY, path TEXT NOT NULL, cwd TEXT NOT NULL, name TEXT,
        created INTEGER NOT NULL, modified INTEGER NOT NULL,
        message_count INTEGER NOT NULL DEFAULT 0, first_message TEXT,
        last_ingested_mtime INTEGER NOT NULL DEFAULT 0, last_ingested_size INTEGER NOT NULL DEFAULT 0
      );
      CREATE TABLE traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT NOT NULL UNIQUE,
        session_id TEXT, root_session_id TEXT, ts INTEGER NOT NULL, model TEXT,
        ttft_ms REAL, prefill_tps REAL, decode_tps REAL, mtp_cycles INTEGER,
        mtp_mean_accepted REAL, duration_ms REAL, finish_reason TEXT,
        prompt_tokens INTEGER, cached_tokens INTEGER, output_tokens INTEGER,
        reasoning_tokens INTEGER, cold_hits INTEGER, cold_misses INTEGER,
        cold_bytes_written INTEGER, cold_bytes_restored INTEGER, source_file TEXT
      );`,
    );
    raw.exec('PRAGMA user_version = 1;'); // matches SCHEMA_VERSION → version check passes
    raw.close();

    const dash = openDashboardDb(file);
    // Rebuilt schema carries a real `turns` table; an insert round-trips.
    dash.db.insert(turns).values({ sessionId: 's1', ts: 1, model: 'qwen3_5', inputTokens: 5 }).run();
    const rows = dash.db.select().from(turns).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].model).toBe('qwen3_5');
    dash.close();

    const quarantined = readdirSync(d).filter((n) => n.startsWith('index.db.corrupt-'));
    expect(quarantined).toHaveLength(1);
    rmSync(d, { recursive: true, force: true });
  });

  // Finding 6: a pre-existing db missing a SINGLE column (matching version) must
  // rebuild — the earlier partial probe omitted `sessions.name`, so a name drop
  // slipped through. The full signature check must catch it.
  it('quarantines a db missing a single column (sessions.name) and rebuilds', () => {
    const d = mkdtempSync(join(tmpdir(), 'dash-nocol-'));
    const file = join(d, 'index.db');
    const raw = new DatabaseSync(file);
    // sessions is missing the `name` column; turns + traces are complete.
    raw.exec(
      `CREATE TABLE sessions (
        id TEXT PRIMARY KEY, path TEXT NOT NULL, cwd TEXT NOT NULL,
        created INTEGER NOT NULL, modified INTEGER NOT NULL,
        message_count INTEGER NOT NULL DEFAULT 0, first_message TEXT,
        last_ingested_mtime INTEGER NOT NULL DEFAULT 0, last_ingested_size INTEGER NOT NULL DEFAULT 0
      );
      CREATE TABLE turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, entry_id TEXT,
        trace_id TEXT, ts INTEGER NOT NULL, model TEXT, input_tokens INTEGER,
        output_tokens INTEGER, cached_tokens INTEGER, reasoning_tokens INTEGER
      );
      CREATE TABLE traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT NOT NULL UNIQUE,
        session_id TEXT, root_session_id TEXT, ts INTEGER NOT NULL, model TEXT,
        ttft_ms REAL, prefill_tps REAL, decode_tps REAL, mtp_cycles INTEGER,
        mtp_mean_accepted REAL, duration_ms REAL, finish_reason TEXT,
        prompt_tokens INTEGER, cached_tokens INTEGER, output_tokens INTEGER,
        reasoning_tokens INTEGER, cold_hits INTEGER, cold_misses INTEGER,
        cold_bytes_written INTEGER, cold_bytes_restored INTEGER, source_file TEXT
      );`,
    );
    raw.exec('PRAGMA user_version = 1;'); // matches SCHEMA_VERSION → version check passes
    raw.close();

    const dash = openDashboardDb(file);
    // Rebuilt sessions carries `name`; an insert using it round-trips.
    dash.db
      .insert(sessions)
      .values({
        id: 's1',
        path: '/tmp/s1.jsonl',
        cwd: '/w',
        name: 'named',
        created: 1,
        modified: 2,
        messageCount: 0,
        firstMessage: null,
        lastIngestedMtime: 0,
        lastIngestedSize: 0,
      })
      .run();
    const rows = dash.db.select().from(sessions).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].name).toBe('named');
    dash.close();

    const quarantined = readdirSync(d).filter((n) => n.startsWith('index.db.corrupt-'));
    expect(quarantined).toHaveLength(1);
    rmSync(d, { recursive: true, force: true });
  });

  // Finding C: a pre-existing db whose `turns` object is a VIEW (not a table)
  // exposing the expected columns must be quarantined+rebuilt. PRAGMA table_info
  // reports a view's columns, so the column check alone would pass — then
  // CREATE INDEX ON turns throws "views may not be indexed", an error outside the
  // rebuildable set that would wedge startup. The type guard must rebuild.
  it('quarantines a db whose turns object is a VIEW (not a table) and rebuilds', () => {
    const d = mkdtempSync(join(tmpdir(), 'dash-view-'));
    const file = join(d, 'index.db');
    const raw = new DatabaseSync(file);
    // sessions + traces are real tables; `turns` is a VIEW over a backing table
    // that projects exactly the expected `turns` columns.
    raw.exec(
      `CREATE TABLE sessions (
        id TEXT PRIMARY KEY, path TEXT NOT NULL, cwd TEXT NOT NULL, name TEXT,
        created INTEGER NOT NULL, modified INTEGER NOT NULL,
        message_count INTEGER NOT NULL DEFAULT 0, first_message TEXT,
        last_ingested_mtime INTEGER NOT NULL DEFAULT 0, last_ingested_size INTEGER NOT NULL DEFAULT 0
      );
      CREATE TABLE traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT NOT NULL UNIQUE,
        session_id TEXT, root_session_id TEXT, ts INTEGER NOT NULL, model TEXT,
        ttft_ms REAL, prefill_tps REAL, decode_tps REAL, mtp_cycles INTEGER,
        mtp_mean_accepted REAL, duration_ms REAL, finish_reason TEXT,
        prompt_tokens INTEGER, cached_tokens INTEGER, output_tokens INTEGER,
        reasoning_tokens INTEGER, cold_hits INTEGER, cold_misses INTEGER,
        cold_bytes_written INTEGER, cold_bytes_restored INTEGER, source_file TEXT
      );
      CREATE TABLE turns_backing (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, entry_id TEXT,
        trace_id TEXT, ts INTEGER NOT NULL, model TEXT, input_tokens INTEGER,
        output_tokens INTEGER, cached_tokens INTEGER, reasoning_tokens INTEGER
      );
      CREATE VIEW turns AS SELECT
        id, session_id, entry_id, trace_id, ts, model,
        input_tokens, output_tokens, cached_tokens, reasoning_tokens
      FROM turns_backing;`,
    );
    raw.exec('PRAGMA user_version = 1;'); // matches SCHEMA_VERSION → version check passes
    raw.close();

    const dash = openDashboardDb(file);
    // Rebuilt schema carries a real `turns` table; an insert round-trips.
    dash.db.insert(turns).values({ sessionId: 's1', ts: 1, model: 'qwen3_5', inputTokens: 5 }).run();
    const rows = dash.db.select().from(turns).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].model).toBe('qwen3_5');
    dash.close();

    const quarantined = readdirSync(d).filter((n) => n.startsWith('index.db.corrupt-'));
    expect(quarantined).toHaveLength(1);
    rmSync(d, { recursive: true, force: true });
  });

  // Finding 5: a pre-existing db whose `turns` object is an FTS5 VIRTUAL table
  // exposing the expected columns must be quarantined+rebuilt. `sqlite_schema`
  // reports a virtual table as type='table', so the earlier type probe passed it;
  // PRAGMA table_info reports its columns, so the column check passed too — then
  // CREATE INDEX ON turns throws "virtual tables may not be indexed", an error
  // outside the rebuildable set that would wedge startup. The table_list type
  // guard (type='virtual' for FTS5) must rebuild instead.
  it('quarantines a db whose turns object is an FTS5 VIRTUAL table and rebuilds', () => {
    const d = mkdtempSync(join(tmpdir(), 'dash-fts5-'));
    const file = join(d, 'index.db');
    const raw = new DatabaseSync(file);
    // sessions + traces are real tables; `turns` is an FTS5 virtual table whose
    // declared columns are exactly the expected `turns` column set.
    raw.exec(
      `CREATE TABLE sessions (
        id TEXT PRIMARY KEY, path TEXT NOT NULL, cwd TEXT NOT NULL, name TEXT,
        created INTEGER NOT NULL, modified INTEGER NOT NULL,
        message_count INTEGER NOT NULL DEFAULT 0, first_message TEXT,
        last_ingested_mtime INTEGER NOT NULL DEFAULT 0, last_ingested_size INTEGER NOT NULL DEFAULT 0
      );
      CREATE TABLE traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT NOT NULL UNIQUE,
        session_id TEXT, root_session_id TEXT, ts INTEGER NOT NULL, model TEXT,
        ttft_ms REAL, prefill_tps REAL, decode_tps REAL, mtp_cycles INTEGER,
        mtp_mean_accepted REAL, duration_ms REAL, finish_reason TEXT,
        prompt_tokens INTEGER, cached_tokens INTEGER, output_tokens INTEGER,
        reasoning_tokens INTEGER, cold_hits INTEGER, cold_misses INTEGER,
        cold_bytes_written INTEGER, cold_bytes_restored INTEGER, source_file TEXT
      );
      CREATE VIRTUAL TABLE turns USING fts5(
        id, session_id, entry_id, trace_id, ts, model,
        input_tokens, output_tokens, cached_tokens, reasoning_tokens
      );`,
    );
    raw.exec('PRAGMA user_version = 1;'); // matches SCHEMA_VERSION → version check passes
    raw.close();

    const dash = openDashboardDb(file);
    // Rebuilt schema carries a real `turns` table; an insert round-trips.
    dash.db.insert(turns).values({ sessionId: 's1', ts: 1, model: 'qwen3_5', inputTokens: 5 }).run();
    const rows = dash.db.select().from(turns).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].model).toBe('qwen3_5');
    dash.close();

    const quarantined = readdirSync(d).filter((n) => n.startsWith('index.db.corrupt-'));
    expect(quarantined).toHaveLength(1);
    rmSync(d, { recursive: true, force: true });
  });

  // Finding #7: a db predating the traces.queue_ms + traces.resident columns
  // (but matching version) must be caught by the signature probe and rebuilt.
  // The auto-derived EXPECTED_COLUMNS must include the new columns, and the
  // rebuilt DDL must carry them — a divergence would either wedge startup or
  // re-quarantine on every open. This asserts both: the rebuilt db round-trips
  // the new columns AND re-validates in place on a second open (no wedge).
  it('quarantines a traces db missing queue_ms/resident and rebuilds, then re-validates without wedging', () => {
    const d = mkdtempSync(join(tmpdir(), 'dash-metrics-'));
    const file = join(d, 'index.db');
    const raw = new DatabaseSync(file);
    // Full sessions + turns; traces omits ONLY the newest queue_ms + resident.
    raw.exec(
      `CREATE TABLE sessions (
        id TEXT PRIMARY KEY, path TEXT NOT NULL, cwd TEXT NOT NULL, name TEXT,
        created INTEGER NOT NULL, modified INTEGER NOT NULL,
        message_count INTEGER NOT NULL DEFAULT 0, first_message TEXT,
        last_ingested_mtime INTEGER NOT NULL DEFAULT 0, last_ingested_size INTEGER NOT NULL DEFAULT 0
      );
      CREATE TABLE turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, entry_id TEXT,
        trace_id TEXT, ts INTEGER NOT NULL, model TEXT, input_tokens INTEGER,
        output_tokens INTEGER, cached_tokens INTEGER, reasoning_tokens INTEGER
      );
      CREATE TABLE traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT NOT NULL UNIQUE,
        session_id TEXT, root_session_id TEXT, ts INTEGER NOT NULL, model TEXT,
        ttft_ms REAL, prefill_tps REAL, decode_tps REAL, mtp_cycles INTEGER,
        mtp_mean_accepted REAL, duration_ms REAL, finish_reason TEXT,
        prompt_tokens INTEGER, cached_tokens INTEGER, output_tokens INTEGER,
        reasoning_tokens INTEGER, cold_hits INTEGER, cold_misses INTEGER,
        cold_bytes_written INTEGER, cold_bytes_restored INTEGER, source_file TEXT
      );`,
    );
    raw.exec('PRAGMA user_version = 1;'); // matches SCHEMA_VERSION → version check passes
    raw.close();

    // First open: signature probe finds queue_ms/resident missing → quarantine+rebuild.
    const first = openDashboardDb(file);
    first.db.insert(traces).values({ traceId: 't1', ts: 1, queueMs: 42, resident: 1 }).run();
    const rows = first.db.select().from(traces).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].queueMs).toBe(42);
    expect(rows[0].resident).toBe(1);
    first.close();

    const quarantined = readdirSync(d).filter((n) => n.startsWith('index.db.corrupt-'));
    expect(quarantined).toHaveLength(1);

    // Second open of the freshly-rebuilt db: DDL and EXPECTED_COLUMNS agree, so it
    // validates in place — the seeded row survives and nothing new is quarantined.
    const second = openDashboardDb(file);
    expect(second.db.select().from(traces).all()).toHaveLength(1);
    second.close();
    expect(readdirSync(d).filter((n) => n.startsWith('index.db.corrupt-'))).toHaveLength(1);

    rmSync(d, { recursive: true, force: true });
  });

  // Finding 6: a complete current-schema db must open WITHOUT rebuilding — the
  // full signature check must not false-positive on a matching schema.
  it('opens a complete current-schema db without rebuilding', () => {
    const d = mkdtempSync(join(tmpdir(), 'dash-complete-'));
    const file = join(d, 'index.db');
    const seed = openDashboardDb(file);
    seed.db
      .insert(sessions)
      .values({
        id: 's1',
        path: '/tmp/s1.jsonl',
        cwd: '/w',
        name: 'keep',
        created: 1,
        modified: 2,
        messageCount: 0,
        firstMessage: null,
        lastIngestedMtime: 0,
        lastIngestedSize: 0,
      })
      .run();
    seed.close();

    const dash = openDashboardDb(file);
    // No rebuild: the seeded row survives (a rebuild would quarantine to an empty db).
    const rows = dash.db.select().from(sessions).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].name).toBe('keep');
    dash.close();

    // Nothing was quarantined.
    const quarantined = readdirSync(d).filter((n) => n.startsWith('index.db.corrupt-'));
    expect(quarantined).toHaveLength(0);
    rmSync(d, { recursive: true, force: true });
  });

  // Finding E: page-level damage OUTSIDE the schema pages passes round-1's
  // DDL-only open but fails later at query time. quick_check must catch it up
  // front and quarantine+rebuild.
  it('quarantines a db with a corrupt data page and rebuilds a working schema', () => {
    const d = mkdtempSync(join(tmpdir(), 'dash-pagecorrupt-'));
    const file = join(d, 'index.db');
    // Grow the file past the header page so real data leaf pages exist.
    const seed = openDashboardDb(file);
    for (let i = 0; i < 500; i++) {
      seed.db
        .insert(sessions)
        .values({
          id: `s${i}`,
          path: '/tmp/s.jsonl',
          cwd: '/w',
          name: null,
          created: 1,
          modified: 2,
          messageCount: 0,
          firstMessage: 'x'.repeat(40),
          lastIngestedMtime: 0,
          lastIngestedSize: 0,
        })
        .run();
    }
    seed.close();

    // Overwrite every page after page 1 with garbage: sqlite_master (page 1)
    // stays valid so CREATE TABLE IF NOT EXISTS still no-ops, but the b-tree
    // data pages are corrupt — exactly the case round-1 accepted.
    const buf = readFileSync(file);
    const pageSize = 4096;
    expect(buf.length).toBeGreaterThan(pageSize * 2);
    buf.fill(0xdd, pageSize);
    writeFileSync(file, buf);

    const dash = openDashboardDb(file);
    // Rebuilt empty schema that actually works (SELECT does not throw malformed).
    expect(dash.db.select().from(sessions).all()).toHaveLength(0);
    dash.db
      .insert(sessions)
      .values({
        id: 's1',
        path: '/tmp/s1.jsonl',
        cwd: '/w',
        name: null,
        created: 1,
        modified: 2,
        messageCount: 0,
        firstMessage: null,
        lastIngestedMtime: 0,
        lastIngestedSize: 0,
      })
      .run();
    expect(dash.db.select().from(sessions).all()).toHaveLength(1);
    dash.close();

    // Corrupt file preserved (not silently lost).
    const quarantined = readdirSync(d).filter((n) => n.startsWith('index.db.corrupt-'));
    expect(quarantined).toHaveLength(1);
    rmSync(d, { recursive: true, force: true });
  });

  it('rethrows a non-corruption open error instead of discarding data', () => {
    const d = mkdtempSync(join(tmpdir(), 'dash-noopen-'));
    const blocker = join(d, 'blocker');
    writeFileSync(blocker, 'x');
    // Parent path is a file → SQLite cannot open the db (ENOTDIR), a non-corruption error.
    expect(() => openDashboardDb(join(blocker, 'index.db'))).toThrow();
    rmSync(d, { recursive: true, force: true });
  });
});
