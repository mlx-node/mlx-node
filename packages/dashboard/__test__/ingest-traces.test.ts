import { cpSync, mkdtempSync, existsSync, rmSync, utimesSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { eq } from 'drizzle-orm';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { openDashboardDb, type DashboardDb } from '../src/db/open.js';
import { traces } from '../src/db/schema.js';
import { ingestTraces } from '../src/ingest/traces.js';

const FIXTURE_TRACES = fileURLToPath(new URL('./fixtures/traces', import.meta.url));
const DAY_MS = 86_400_000;

function traceLine(traceId: string): string {
  return `${JSON.stringify({
    v: 1,
    traceId,
    ts: 1782036002000,
    model: 'qwen3_5',
    durationMs: 100,
    finishReason: 'stop',
    promptTokens: 1,
    cachedTokens: 0,
    outputTokens: 1,
    reasoningTokens: 0,
  })}\n`;
}

let dash: DashboardDb;
let base: string;
let dir: string;

beforeEach(() => {
  dash = openDashboardDb(':memory:');
  base = mkdtempSync(join(tmpdir(), 'dash-traces-'));
  dir = join(base, 'traces');
});

afterEach(() => {
  dash.close();
  rmSync(base, { recursive: true, force: true });
});

describe('ingestTraces', () => {
  it('indexes trace records with numeric fields', async () => {
    cpSync(FIXTURE_TRACES, dir, { recursive: true });
    const res = await ingestTraces(dash, dir);
    expect(res.files).toBe(1);
    expect(res.records).toBe(3);
    expect(res.pruned).toBe(0);

    const rows = dash.db.select().from(traces).all();
    expect(rows).toHaveLength(3);

    const a = dash.db.select().from(traces).where(eq(traces.traceId, 'trace-aaa')).all()[0];
    expect(a.sessionId).toBe('fix-1');
    expect(a.model).toBe('qwen3_5');
    expect(typeof a.ttftMs).toBe('number');
    expect(a.ttftMs).toBe(120.5);
    expect(a.decodeTps).toBe(95.4);
    expect(a.mtpCycles).toBe(3);
    expect(a.promptTokens).toBe(100);
    expect(a.durationMs).toBe(1234.5);
    expect(a.coldBytesRestored).toBe(2048);
  });

  it('is idempotent on duplicate traceId', async () => {
    cpSync(FIXTURE_TRACES, dir, { recursive: true });
    await ingestTraces(dash, dir);
    const second = await ingestTraces(dash, dir);
    expect(second.records).toBe(3);
    expect(dash.db.select().from(traces).all()).toHaveLength(3);
  });

  it('prunes files older than retentionDays', async () => {
    cpSync(FIXTURE_TRACES, dir, { recursive: true });
    const oldFile = join(dir, '2026-05-01-11111.jsonl');
    writeFileSync(oldFile, traceLine('trace-old'));
    const oldSec = (Date.now() - 60 * DAY_MS) / 1000;
    utimesSync(oldFile, oldSec, oldSec);

    const res = await ingestTraces(dash, dir, { retentionDays: 30 });
    expect(res.pruned).toBe(1);
    expect(res.files).toBe(1);
    expect(existsSync(oldFile)).toBe(false);
    expect(dash.db.select().from(traces).where(eq(traces.traceId, 'trace-old')).all()).toHaveLength(0);
    expect(dash.db.select().from(traces).all()).toHaveLength(3);
  });
});
