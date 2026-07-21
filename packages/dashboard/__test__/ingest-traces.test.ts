import { cpSync, mkdtempSync, existsSync, rmSync, unlinkSync, utimesSync, writeFileSync } from 'node:fs';
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

  // Finding 11b: a subagent turn's root session id must survive ingest so the
  // root session's metrics view can include its delegated children.
  it('stores root_session_id from a subagent trace record', async () => {
    cpSync(FIXTURE_TRACES, dir, { recursive: true });
    writeFileSync(
      join(dir, '2026-07-02-child.jsonl'),
      `${JSON.stringify({
        v: 1,
        traceId: 'trace-child',
        ts: 1782036302000,
        sessionId: 'child-x',
        rootSessionId: 'root-r',
        model: 'qwen3_5',
        durationMs: 10,
        finishReason: 'stop',
        promptTokens: 1,
        cachedTokens: 0,
        outputTokens: 1,
        reasoningTokens: 0,
      })}\n`,
    );
    await ingestTraces(dash, dir);
    const row = dash.db.select().from(traces).where(eq(traces.traceId, 'trace-child')).all()[0];
    expect(row.sessionId).toBe('child-x');
    expect(row.rootSessionId).toBe('root-r');
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

  // Finding 8: pruning a file must delete its DB rows in the same operation.
  it('deletes a pruned file rows, not just the file', async () => {
    cpSync(FIXTURE_TRACES, dir, { recursive: true });
    await ingestTraces(dash, dir);
    expect(dash.db.select().from(traces).all()).toHaveLength(3);

    // Age the ingested fixture file past retention, then re-ingest.
    const fixtureFile = join(dir, '2026-07-01-99999.jsonl');
    const oldSec = (Date.now() - 60 * DAY_MS) / 1000;
    utimesSync(fixtureFile, oldSec, oldSec);

    const res = await ingestTraces(dash, dir, { retentionDays: 30 });
    expect(res.pruned).toBe(1);
    expect(existsSync(fixtureFile)).toBe(false);
    // The rows it produced are gone with it — no orphaned telemetry.
    expect(dash.db.select().from(traces).all()).toHaveLength(0);
  });

  // Finding 8: rows whose source file vanished by other means are reconciled.
  it('reconciles rows when a source file is manually deleted', async () => {
    cpSync(FIXTURE_TRACES, dir, { recursive: true });
    await ingestTraces(dash, dir);
    expect(dash.db.select().from(traces).all()).toHaveLength(3);

    // Delete the source file out from under the index, then re-ingest.
    unlinkSync(join(dir, '2026-07-01-99999.jsonl'));
    const res = await ingestTraces(dash, dir);
    expect(res.files).toBe(0);
    expect(dash.db.select().from(traces).all()).toHaveLength(0);
  });

  // Finding J: deleting the WHOLE trace dir must still reconcile tracked rows,
  // not short-circuit before reconciliation and leave them visible forever.
  it('reconciles all tracked rows when the entire trace dir is deleted', async () => {
    cpSync(FIXTURE_TRACES, dir, { recursive: true });
    await ingestTraces(dash, dir);
    expect(dash.db.select().from(traces).all()).toHaveLength(3);

    rmSync(dir, { recursive: true, force: true });
    const res = await ingestTraces(dash, dir);
    expect(res.files).toBe(0);
    expect(dash.db.select().from(traces).all()).toHaveLength(0);
  });
});
