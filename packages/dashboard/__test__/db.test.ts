import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';

import { describe, expect, it } from 'vite-plus/test';

import { openDashboardDb } from '../src/db/open.js';
import { sessions, traces } from '../src/db/schema.js';

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
