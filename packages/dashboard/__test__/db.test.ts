import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { openDashboardDb } from '../src/db/open.js';
import { sessions } from '../src/db/schema.js';

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

  it('rethrows a non-corruption open error instead of discarding data', () => {
    const d = mkdtempSync(join(tmpdir(), 'dash-noopen-'));
    const blocker = join(d, 'blocker');
    writeFileSync(blocker, 'x');
    // Parent path is a file → SQLite cannot open the db (ENOTDIR), a non-corruption error.
    expect(() => openDashboardDb(join(blocker, 'index.db'))).toThrow();
    rmSync(d, { recursive: true, force: true });
  });
});
