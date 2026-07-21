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
