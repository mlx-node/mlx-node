import {
  appendFileSync,
  cpSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  unlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { eq } from 'drizzle-orm';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { openDashboardDb, type DashboardDb } from '../src/db/open.js';
import { sessions, turns } from '../src/db/schema.js';
import { ingestSessions, verifySessionFileId } from '../src/ingest/sessions.js';

/** Write JSONL lines (objects) to a fresh `--w--` session dir; returns the file path. */
function writeSessionFile(base: string, fileName: string, lines: object[]): string {
  const dir = join(base, 'sessions', '--w--');
  mkdirSync(dir, { recursive: true });
  const file = join(dir, fileName);
  writeFileSync(file, `${lines.map((l) => JSON.stringify(l)).join('\n')}\n`);
  return file;
}

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

  // Finding 12: turns must follow the active branch, not abandoned tree branches.
  it('indexes only the active-branch turn, not an abandoned one', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-branch-'));
    const root = join(soloBase, 'sessions');
    writeSessionFile(soloBase, 'branched.jsonl', [
      { type: 'session', version: 3, id: 'br-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
      { type: 'message', id: 'u1', parentId: null, timestamp: '2026-07-01T10:00:01.000Z', message: { role: 'user', content: 'q' } },
      // Abandoned assistant branch off u1 (huge tokens — must NOT be counted).
      {
        type: 'message',
        id: 'a1',
        parentId: 'u1',
        timestamp: '2026-07-01T10:00:02.000Z',
        message: { role: 'assistant', content: [{ type: 'text', text: 'abandoned' }], model: 'gemma4', usage: { input: 999, output: 999 } },
      },
      // Active replacement off u1, appended last → the leaf.
      {
        type: 'message',
        id: 'a2',
        parentId: 'u1',
        timestamp: '2026-07-01T10:00:03.000Z',
        message: { role: 'assistant', content: [{ type: 'text', text: 'active' }], model: 'qwen3_5', usage: { input: 1, output: 2 } },
      },
    ]);

    await ingestSessions(dash, root);

    const turnRows = dash.db.select().from(turns).where(eq(turns.sessionId, 'br-1')).all();
    expect(turnRows).toHaveLength(1);
    expect(turnRows[0].entryId).toBe('a2');
    expect(turnRows[0].model).toBe('qwen3_5');
    expect(turnRows[0].inputTokens).toBe(1);

    const row = dash.db.select().from(sessions).where(eq(sessions.id, 'br-1')).all()[0];
    // Active branch is [u1, a2]; abandoned a1 is excluded from the count.
    expect(row.messageCount).toBe(2);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // Finding 3: a reused path must not leave a stale row that can delete the new file.
  it('reconciles a stale row when a path is reused by a new session', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-reuse-'));
    const root = join(soloBase, 'sessions');
    const file = writeSessionFile(soloBase, 's.jsonl', [
      { type: 'session', version: 3, id: 'sess-A', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
      { type: 'message', id: 'a1', parentId: null, timestamp: '2026-07-01T10:00:01.000Z', message: { role: 'user', content: 'from A' } },
    ]);
    await ingestSessions(dash, root);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'sess-A')).all()).toHaveLength(1);

    // Replace the path with session B (different id) and bump mtime forward.
    writeFileSync(
      file,
      `${JSON.stringify({ type: 'session', version: 3, id: 'sess-B', timestamp: '2026-07-02T10:00:00.000Z', cwd: '/w' })}\n${JSON.stringify({ type: 'message', id: 'b1', parentId: null, timestamp: '2026-07-02T10:00:01.000Z', message: { role: 'user', content: 'from B is longer' } })}\n`,
    );
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);
    await ingestSessions(dash, root);

    // Only B remains; the stale A row (which still pointed at this path) is gone.
    const rows = dash.db.select().from(sessions).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].id).toBe('sess-B');
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'sess-A')).all()).toHaveLength(0);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // Finding 3: the delete-time guard used by the API before rmSync.
  it('verifySessionFileId matches only the file current header id', () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-verify-'));
    const file = writeSessionFile(soloBase, 's.jsonl', [
      { type: 'session', version: 3, id: 'sess-B', timestamp: '2026-07-02T10:00:00.000Z', cwd: '/w' },
    ]);
    expect(verifySessionFileId(file, 'sess-B')).toBe(true);
    expect(verifySessionFileId(file, 'sess-A')).toBe(false);
    expect(verifySessionFileId(join(soloBase, 'missing.jsonl'), 'sess-B')).toBe(false);
    rmSync(soloBase, { recursive: true, force: true });
  });

  // Finding 13: a truncated trailing line must not advance the ingest watermark.
  it('does not mark a truncated file fully-ingested and re-ingests once completed', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-trunc-'));
    const root = join(soloBase, 'sessions');
    const dir = join(root, '--w--');
    mkdirSync(dir, { recursive: true });
    const file = join(dir, 'partial.jsonl');

    const header = JSON.stringify({ type: 'session', version: 3, id: 'trunc-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' });
    const user = JSON.stringify({ type: 'message', id: 'm1', parentId: null, timestamp: '2026-07-01T10:00:01.000Z', message: { role: 'user', content: 'hi' } });
    const asst = JSON.stringify({
      type: 'message',
      id: 'm2',
      parentId: 'm1',
      timestamp: '2026-07-01T10:00:02.000Z',
      message: { role: 'assistant', content: [{ type: 'text', text: 'yo' }], model: 'qwen3_5', usage: { input: 5, output: 6 } },
    });
    const truncated = '{"type":"message","id":"m3","parentId":"m2","timestamp":"2026-07-01T10:00:03.000Z","message":{"role":"asst';

    writeFileSync(file, `${header}\n${user}\n${asst}\n${truncated}`);
    const res1 = await ingestSessions(dash, root);
    expect(res1.warnings.some((w) => /trailing|malformed/i.test(w))).toBe(true);

    const row1 = dash.db.select().from(sessions).where(eq(sessions.id, 'trunc-1')).all()[0];
    // Partial data indexed, but the watermark is NOT the current mtime/size.
    expect(row1.lastIngestedMtime).toBe(0);
    expect(row1.lastIngestedSize).toBe(0);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'trunc-1')).all()).toHaveLength(1);

    // Complete the trailing write and bump mtime.
    const asst3 = JSON.stringify({
      type: 'message',
      id: 'm3',
      parentId: 'm2',
      timestamp: '2026-07-01T10:00:03.000Z',
      message: { role: 'assistant', content: [{ type: 'text', text: 'done' }], model: 'qwen3_5', usage: { input: 7, output: 8 } },
    });
    writeFileSync(file, `${header}\n${user}\n${asst}\n${asst3}\n`);
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);

    const res2 = await ingestSessions(dash, root);
    // The stale (0,0) watermark forced a re-ingest that now sees the completed line.
    expect(res2.updated).toBe(1);
    expect(res2.warnings.some((w) => /trailing|malformed/i.test(w))).toBe(false);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'trunc-1')).all()).toHaveLength(2);
    const row2 = dash.db.select().from(sessions).where(eq(sessions.id, 'trunc-1')).all()[0];
    expect(row2.lastIngestedMtime).toBeGreaterThan(0);

    rmSync(soloBase, { recursive: true, force: true });
  });
});
