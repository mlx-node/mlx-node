import { appendFileSync, cpSync, mkdirSync, mkdtempSync, rmSync, unlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { eq } from 'drizzle-orm';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { openDashboardDb, type DashboardDb } from '../src/db/open.js';
import { sessions, turns } from '../src/db/schema.js';
import { ingestSessions } from '../src/ingest/sessions.js';

const FIXTURE_SESSIONS = fileURLToPath(new URL('./fixtures/sessions', import.meta.url));

let dash: DashboardDb;
let workRoot: string;

beforeEach(() => {
  dash = openDashboardDb(':memory:');
  const base = mkdtempSync(join(tmpdir(), 'dash-sessions-'));
  workRoot = join(base, 'sessions');
  cpSync(FIXTURE_SESSIONS, workRoot, { recursive: true });
});

afterEach(() => {
  dash.close();
  rmSync(join(workRoot, '..'), { recursive: true, force: true });
});

describe('ingestSessions', () => {
  it('indexes fixture sessions with derived fields', async () => {
    const res = await ingestSessions(dash, workRoot);
    expect(res.scanned).toBe(3);
    expect(res.updated).toBe(2);
    expect(res.removed).toBe(0);
    expect(res.warnings.length).toBeGreaterThanOrEqual(1);

    const rows = dash.db.select().from(sessions).all();
    expect(rows).toHaveLength(2);

    const one = dash.db.select().from(sessions).where(eq(sessions.id, 'fix-1')).all()[0];
    expect(one.name).toBe('First session');
    expect(one.messageCount).toBe(4);
    expect(one.firstMessage).toBe('Hello, world');
    expect(one.cwd).toBe('/w');
    expect(one.created).toBe(Date.parse('2026-07-01T10:00:00.000Z'));

    const allTurns = dash.db.select().from(turns).all();
    expect(allTurns).toHaveLength(3);

    const traced = dash.db.select().from(turns).where(eq(turns.traceId, 'trace-aaa')).all();
    expect(traced).toHaveLength(1);
    expect(traced[0].sessionId).toBe('fix-1');
    expect(traced[0].entryId).toBe('m2');
    expect(traced[0].inputTokens).toBe(100);
    expect(traced[0].outputTokens).toBe(50);
    expect(traced[0].cachedTokens).toBe(10);
    expect(traced[0].reasoningTokens).toBe(5);
  });

  it('skips unchanged files on re-ingest', async () => {
    await ingestSessions(dash, workRoot);
    const second = await ingestSessions(dash, workRoot);
    expect(second.scanned).toBe(3);
    expect(second.updated).toBe(0);
    expect(second.removed).toBe(0);
    expect(dash.db.select().from(sessions).all()).toHaveLength(2);
  });

  it('re-ingests only the file that changed', async () => {
    await ingestSessions(dash, workRoot);
    const file = join(workRoot, '--w--', '2026-07-01T10-00-00_fix-1.jsonl');
    appendFileSync(
      file,
      `${JSON.stringify({
        type: 'message',
        id: 'm5',
        parentId: 'm4',
        timestamp: '2026-07-01T10:00:06.000Z',
        message: { role: 'user', content: 'One more', timestamp: 1782036006000 },
      })}\n`,
    );

    const second = await ingestSessions(dash, workRoot);
    expect(second.updated).toBe(1);
    expect(second.removed).toBe(0);

    const one = dash.db.select().from(sessions).where(eq(sessions.id, 'fix-1')).all()[0];
    expect(one.messageCount).toBe(5);
  });

  it('removes rows when a session file is deleted', async () => {
    await ingestSessions(dash, workRoot);
    unlinkSync(join(workRoot, '--w--', '2026-07-02T10-00-00_fix-2.jsonl'));

    const second = await ingestSessions(dash, workRoot);
    expect(second.removed).toBe(1);

    expect(dash.db.select().from(sessions).all()).toHaveLength(1);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'fix-2')).all()).toHaveLength(0);
  });

  it('skips and reports a malformed file without throwing', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-bad-'));
    const soloRoot = join(soloBase, 'sessions');
    mkdirSync(join(soloRoot, '--x--'), { recursive: true });
    writeFileSync(join(soloRoot, '--x--', 'bad.jsonl'), '{"type":"session","id":"trunc","cw');

    const res = await ingestSessions(dash, soloRoot);
    expect(res.scanned).toBe(1);
    expect(res.updated).toBe(0);
    expect(res.warnings).toHaveLength(1);
    expect(dash.db.select().from(sessions).all()).toHaveLength(0);

    rmSync(soloBase, { recursive: true, force: true });
  });
});
