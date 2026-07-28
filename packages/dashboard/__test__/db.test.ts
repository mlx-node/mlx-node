import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';

import { getTableColumns } from 'drizzle-orm';
import type { SQLiteTable } from 'drizzle-orm/sqlite-core';
import { describe, expect, it } from 'vite-plus/test';

import { openDashboardDb } from '../src/db/open.js';
import { sessions, traceFiles, traces, turns } from '../src/db/schema.js';

/** The four tables `assertSchemaSignature` checks, in the order it checks them. */
const SCHEMA_TABLES: Array<[string, SQLiteTable]> = [
  ['sessions', sessions],
  ['turns', turns],
  ['traces', traces],
  ['trace_files', traceFiles],
];

/** DB column names the live drizzle schema declares for a table. */
function columnNames(table: SQLiteTable): string[] {
  return Object.values(getTableColumns(table)).map((c) => c.name);
}

/**
 * A `CREATE TABLE` for `name` carrying every column the live drizzle schema
 * declares, minus `omit`.
 *
 * Derived rather than hand-listed on purpose. `assertSchemaSignature` throws on
 * the FIRST gap it finds and walks the tables in a fixed order, so a fixture
 * whose hand-written DDL falls behind a newly added schema column starts
 * quarantining for THAT column instead of the gap its test names — the test
 * keeps passing while the thing it guards stops being guarded. Deriving the DDL
 * makes every fixture track `schema.ts` automatically, so the only gap is the
 * one the caller asks for. An `omit` naming a column that does not exist throws
 * rather than silently producing a complete fixture.
 */
function fixtureTable(name: string, table: SQLiteTable, omit: string[] = []): string {
  const columns = Object.values(getTableColumns(table));
  for (const col of omit) {
    if (!columns.some((c) => c.name === col)) throw new Error(`fixtureTable(${name}): no such column "${col}"`);
  }
  const dropped = new Set(omit);
  const defs = columns
    .filter((c) => !dropped.has(c.name))
    .map((c) => {
      const parts = [c.name, c.getSQLType().toUpperCase()];
      if (c.primary) parts.push('PRIMARY KEY');
      else if (c.isUnique) parts.push('UNIQUE');
      if (c.notNull && !c.primary) parts.push('NOT NULL');
      if (typeof c.default === 'number' || typeof c.default === 'string') {
        parts.push(`DEFAULT ${typeof c.default === 'number' ? c.default : `'${c.default}'`}`);
      }
      return parts.join(' ');
    });
  return `CREATE TABLE ${name} (${defs.join(', ')});`;
}

/**
 * Assert the fixture on disk departs from the live schema in exactly the listed
 * places (`"trace_files"` for a whole table, `"traces.source_file"` for a single
 * column), reported in probe order.
 *
 * This is what keeps each quarantine test honest. The probe stops at the first
 * gap, so any EXTRA drift short-circuits it and the test below still sees a
 * quarantine — passing for a reason it never claimed. Asserting the gap set up
 * front turns that silent meaning-change into a failure.
 */
function expectFixtureGaps(file: string, expected: string[]): void {
  const raw = new DatabaseSync(file);
  const gaps: string[] = [];
  try {
    for (const [name, table] of SCHEMA_TABLES) {
      const meta = raw.prepare(`PRAGMA table_list(${name})`).get() as { type?: unknown } | undefined;
      if (meta?.type !== 'table') {
        gaps.push(name);
        continue;
      }
      const rows = raw.prepare(`PRAGMA table_info(${name})`).all() as Array<{ name?: unknown }>;
      const present = new Set(rows.map((r) => (typeof r.name === 'string' ? r.name : '')));
      for (const col of columnNames(table)) if (!present.has(col)) gaps.push(`${name}.${col}`);
    }
  } finally {
    raw.close();
  }
  expect(gaps).toEqual(expected);
}

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
    bump.exec('PRAGMA user_version = 6;'); // > SCHEMA_VERSION (5)
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
    // Everything complete except the single dropped column under test.
    raw.exec(
      [
        fixtureTable('sessions', sessions),
        fixtureTable('turns', turns),
        fixtureTable('traces', traces, ['source_file']),
        fixtureTable('trace_files', traceFiles),
      ].join('\n'),
    );
    raw.exec('PRAGMA user_version = 5;'); // matches SCHEMA_VERSION → version check passes
    raw.close();
    expectFixtureGaps(file, ['traces.source_file']);

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
    // Every other table complete; NO turns table at all.
    raw.exec(
      [
        fixtureTable('sessions', sessions),
        fixtureTable('traces', traces),
        fixtureTable('trace_files', traceFiles),
      ].join('\n'),
    );
    raw.exec('PRAGMA user_version = 5;'); // matches SCHEMA_VERSION → version check passes
    raw.close();
    expectFixtureGaps(file, ['turns']);

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
    // sessions is missing the `name` column; every other table is complete.
    raw.exec(
      [
        fixtureTable('sessions', sessions, ['name']),
        fixtureTable('turns', turns),
        fixtureTable('traces', traces),
        fixtureTable('trace_files', traceFiles),
      ].join('\n'),
    );
    raw.exec('PRAGMA user_version = 5;'); // matches SCHEMA_VERSION → version check passes
    raw.close();
    expectFixtureGaps(file, ['sessions.name']);

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
    // Every other table is real and complete; `turns` is a VIEW over a backing
    // table that projects exactly the expected `turns` columns.
    raw.exec(
      [
        fixtureTable('sessions', sessions),
        fixtureTable('traces', traces),
        fixtureTable('trace_files', traceFiles),
        fixtureTable('turns_backing', turns),
        `CREATE VIEW turns AS SELECT ${columnNames(turns).join(', ')} FROM turns_backing;`,
      ].join('\n'),
    );
    raw.exec('PRAGMA user_version = 5;'); // matches SCHEMA_VERSION → version check passes
    raw.close();
    expectFixtureGaps(file, ['turns']);

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
    // Every other table is real and complete; `turns` is an FTS5 virtual table
    // whose declared columns are exactly the expected `turns` column set.
    raw.exec(
      [
        fixtureTable('sessions', sessions),
        fixtureTable('traces', traces),
        fixtureTable('trace_files', traceFiles),
        `CREATE VIRTUAL TABLE turns USING fts5(${columnNames(turns).join(', ')});`,
      ].join('\n'),
    );
    raw.exec('PRAGMA user_version = 5;'); // matches SCHEMA_VERSION → version check passes
    raw.close();
    expectFixtureGaps(file, ['turns']);

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
    // Every table complete; traces omits ONLY queue_ms + resident.
    raw.exec(
      [
        fixtureTable('sessions', sessions),
        fixtureTable('turns', turns),
        fixtureTable('traces', traces, ['queue_ms', 'resident']),
        fixtureTable('trace_files', traceFiles),
      ].join('\n'),
    );
    raw.exec('PRAGMA user_version = 5;'); // matches SCHEMA_VERSION → version check passes
    raw.close();
    expectFixtureGaps(file, ['traces.queue_ms', 'traces.resident']);

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

  // F2: a db predating the trace_files watermark table (matching version, every
  // other table complete) must be quarantined+rebuilt. The auto-derived
  // EXPECTED_COLUMNS now includes trace_files, so its absence is caught by the
  // signature probe (it iterates last, after sessions/turns/traces all pass).
  // The rebuilt db round-trips a trace_files row AND re-validates on a second
  // open — proving the SCHEMA_VERSION bump + DDL + validation all agree (no
  // startup wedge, no re-quarantine loop).
  //
  // trace_files being checked LAST is exactly why the fixture must be derived
  // from the live schema: while its traces DDL was hand-written it fell four
  // columns behind, the probe threw on `traces.cold_write_errors` and never
  // reached trace_files, and this test passed with the guard it exists for dead.
  it('quarantines a db missing the trace_files table and rebuilds, then re-validates without wedging', () => {
    const d = mkdtempSync(join(tmpdir(), 'dash-notracefiles-'));
    const file = join(d, 'index.db');
    const raw = new DatabaseSync(file);
    // Full sessions + turns + traces, but NO trace_files watermark table.
    raw.exec(
      [fixtureTable('sessions', sessions), fixtureTable('turns', turns), fixtureTable('traces', traces)].join('\n'),
    );
    raw.exec('PRAGMA user_version = 5;'); // matches SCHEMA_VERSION → version check passes
    raw.close();
    expectFixtureGaps(file, ['trace_files']);

    // First open: signature probe finds trace_files missing → quarantine+rebuild.
    const first = openDashboardDb(file);
    first.db.insert(traceFiles).values({ name: 'f.jsonl', lastIngestedMtime: 5, lastIngestedSize: 9 }).run();
    const rows = first.db.select().from(traceFiles).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].name).toBe('f.jsonl');
    expect(rows[0].lastIngestedMtime).toBe(5);
    first.close();

    const quarantined = readdirSync(d).filter((n) => n.startsWith('index.db.corrupt-'));
    expect(quarantined).toHaveLength(1);

    // Second open of the freshly-rebuilt db: DDL and EXPECTED_COLUMNS agree, so
    // it validates in place — the seeded row survives, nothing new is quarantined.
    const second = openDashboardDb(file);
    expect(second.db.select().from(traceFiles).all()).toHaveLength(1);
    second.close();
    expect(readdirSync(d).filter((n) => n.startsWith('index.db.corrupt-'))).toHaveLength(1);

    rmSync(d, { recursive: true, force: true });
  });

  // F1 migration: the cold-attribution columns (cold_root / cold_enabled) and the
  // four dropped cold-tier counters are a DDL change, so an existing v2 index —
  // complete and valid for the build that wrote it — must be quarantined ASIDE
  // (never deleted) and rebuilt carrying them, and the rebuilt file must then
  // re-validate in place instead of re-quarantining on every subsequent open.
  it('quarantines a complete v2 db lacking the cold-attribution columns, rebuilds with them, and re-validates', () => {
    const d = mkdtempSync(join(tmpdir(), 'dash-coldcols-'));
    const file = join(d, 'index.db');
    const raw = new DatabaseSync(file);
    // Byte-for-byte the schema SCHEMA_VERSION 2 shipped: every table complete,
    // every v2 column present. The ONLY gap is the new cold-attribution set.
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
        mtp_mean_accepted REAL, duration_ms REAL, queue_ms INTEGER, finish_reason TEXT,
        resident INTEGER, prompt_tokens INTEGER, cached_tokens INTEGER, output_tokens INTEGER,
        reasoning_tokens INTEGER, cold_hits INTEGER, cold_misses INTEGER,
        cold_bytes_written INTEGER, cold_bytes_restored INTEGER, source_file TEXT
      );
      CREATE TABLE trace_files (
        name TEXT PRIMARY KEY, last_ingested_mtime INTEGER NOT NULL, last_ingested_size INTEGER NOT NULL
      );`,
    );
    raw.exec("INSERT INTO traces (trace_id, ts, cold_hits) VALUES ('legacy-1', 1, 7);");
    raw.exec('PRAGMA user_version = 2;'); // three SCHEMA_VERSIONs back
    raw.close();

    // Opening does not throw and does not wedge: quarantine + rebuild.
    const first = openDashboardDb(file);
    const quarantined = readdirSync(d).filter((n) => n.startsWith('index.db.corrupt-'));
    expect(quarantined).toHaveLength(1);
    // The user's old index is preserved on disk, not discarded.
    expect(readFileSync(join(d, quarantined[0])).subarray(0, 15).toString('utf-8')).toContain('SQLite format 3');

    // Every new column is live and round-trips through drizzle.
    first.db
      .insert(traces)
      .values({
        traceId: 't1',
        ts: 1,
        coldRoot: '/canonical/cache/root',
        coldEnabled: 1,
        coldEnqueued: 4,
        coldQueueDrops: 1,
        coldEvictions: 2,
        coldCorruptions: 0,
        coldCorruptionsTotal: 0,
        coldQueueDropsTotal: 3,
        // The v4 set: the two silent failure modes. Distinct values so a
        // column wired to its neighbour in `schema.ts` cannot round-trip.
        coldWriteErrors: 5,
        coldWriteErrorsTotal: 6,
        coldRestoreDeclines: 7,
        coldSidecarRestoreSuppressed: 8,
        // The v5 set: the seven sidecar counters the ingest mapping dropped
        // while only `restoreSuppressed` reached a column. Distinct values
        // again, and distinct from the object-scoped `coldEnqueued` (4) and
        // `coldQueueDrops` (1) they are a subset of — so a column wired to its
        // block-level namesake cannot round-trip.
        coldSidecarCaptureReached: 9,
        coldSidecarChainEmpty: 10,
        coldSidecarBoundarySkips: 11,
        coldSidecarAlreadyPersisted: 12,
        coldSidecarEnqueued: 13,
        coldSidecarQueueDrops: 14,
        coldSidecarInstalled: 15,
      })
      .run();
    const rows = first.db.select().from(traces).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].coldRoot).toBe('/canonical/cache/root');
    expect(rows[0].coldEnabled).toBe(1);
    expect(rows[0].coldEnqueued).toBe(4);
    expect(rows[0].coldQueueDrops).toBe(1);
    expect(rows[0].coldEvictions).toBe(2);
    expect(rows[0].coldCorruptions).toBe(0);
    expect(rows[0].coldCorruptionsTotal).toBe(0);
    expect(rows[0].coldQueueDropsTotal).toBe(3);
    expect(rows[0].coldWriteErrors).toBe(5);
    expect(rows[0].coldWriteErrorsTotal).toBe(6);
    expect(rows[0].coldRestoreDeclines).toBe(7);
    expect(rows[0].coldSidecarRestoreSuppressed).toBe(8);
    expect(rows[0].coldSidecarCaptureReached).toBe(9);
    expect(rows[0].coldSidecarChainEmpty).toBe(10);
    expect(rows[0].coldSidecarBoundarySkips).toBe(11);
    expect(rows[0].coldSidecarAlreadyPersisted).toBe(12);
    expect(rows[0].coldSidecarEnqueued).toBe(13);
    expect(rows[0].coldSidecarQueueDrops).toBe(14);
    expect(rows[0].coldSidecarInstalled).toBe(15);

    // The DDL change came WITH a version bump. Without one, the next build to
    // touch the schema would open a drifted db on the version check alone.
    // If this number needs changing, every `PRAGMA user_version` fixture above
    // must be restamped in the same commit — otherwise the version check
    // short-circuits and their signature probes silently stop running. Their
    // COLUMNS need no such maintenance: those fixtures build their DDL from the
    // live schema via `fixtureTable`, so a new column lands in all of them at
    // once and each keeps exactly the one gap `expectFixtureGaps` pins.
    const uv = first.sqlite.prepare('PRAGMA user_version').get() as { user_version: number };
    expect(uv.user_version).toBe(5);
    first.close();

    // Reopening the rebuilt file validates in place: no second quarantine.
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
