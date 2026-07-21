import { rmSync } from 'node:fs';
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
});
