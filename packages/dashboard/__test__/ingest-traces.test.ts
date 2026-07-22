import {
  appendFileSync,
  cpSync,
  mkdirSync,
  mkdtempSync,
  existsSync,
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
import { traceFiles, traces } from '../src/db/schema.js';
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

  // Finding #7: queue_ms + resident telemetry must land in their own columns;
  // the boolean `resident` is encoded 0/1.
  it('stores queueMs and resident (boolean → 0/1) from a trace record', async () => {
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, '2026-07-20-metrics.jsonl'),
      `${JSON.stringify({
        v: 1,
        traceId: 'trace-warm',
        ts: 1782036402000,
        model: 'qwen3_5',
        durationMs: 500,
        queueMs: 42,
        resident: true,
        finishReason: 'stop',
        promptTokens: 1,
        cachedTokens: 0,
        outputTokens: 1,
        reasoningTokens: 0,
      })}\n${JSON.stringify({
        v: 1,
        traceId: 'trace-cold',
        ts: 1782036403000,
        model: 'qwen3_5',
        durationMs: 900,
        queueMs: 3100,
        resident: false,
        finishReason: 'stop',
        promptTokens: 1,
        cachedTokens: 0,
        outputTokens: 1,
        reasoningTokens: 0,
      })}\n`,
    );
    const res = await ingestTraces(dash, dir);
    expect(res.records).toBe(2);

    const warm = dash.db.select().from(traces).where(eq(traces.traceId, 'trace-warm')).all()[0];
    expect(warm.queueMs).toBe(42);
    expect(warm.resident).toBe(1);

    const cold = dash.db.select().from(traces).where(eq(traces.traceId, 'trace-cold')).all()[0];
    expect(cold.queueMs).toBe(3100);
    expect(cold.resident).toBe(0);
  });

  // Finding #7: older JSONL without the new fields ingests them as NULL, not 0.
  it('leaves queueMs/resident NULL when a trace record omits them', async () => {
    cpSync(FIXTURE_TRACES, dir, { recursive: true });
    await ingestTraces(dash, dir);
    const row = dash.db.select().from(traces).where(eq(traces.traceId, 'trace-bbb')).all()[0];
    expect(row.queueMs).toBeNull();
    expect(row.resident).toBeNull();
  });

  it('is idempotent on duplicate traceId', async () => {
    cpSync(FIXTURE_TRACES, dir, { recursive: true });
    await ingestTraces(dash, dir);
    // F2: the unchanged file is now skipped by its watermark, so the second pass
    // reads/parses nothing — yet the index still holds exactly the 3 rows (no
    // duplicates). The onConflict guard still covers a genuine re-read (see the
    // appended-file test below).
    const second = await ingestTraces(dash, dir);
    expect(second.files).toBe(0);
    expect(second.records).toBe(0);
    expect(dash.db.select().from(traces).all()).toHaveLength(3);
  });

  // F2: a fully-ingested file records a watermark; an unchanged file is skipped
  // on the next rescan (not re-read/re-parsed) so a 30s poll is O(new files).
  it('skips an unchanged file on the second ingest (watermark)', async () => {
    cpSync(FIXTURE_TRACES, dir, { recursive: true });
    const first = await ingestTraces(dash, dir);
    expect(first.files).toBe(1);
    expect(first.records).toBe(3);

    // The watermark for the file was written.
    const wm = dash.db.select().from(traceFiles).all();
    expect(wm).toHaveLength(1);
    expect(wm[0].name).toBe('2026-07-01-99999.jsonl');
    expect(wm[0].lastIngestedSize).toBeGreaterThan(0);

    // Second pass: the file matches its watermark → not read (files === 0), not
    // re-parsed (records === 0), and the row set is unchanged.
    const second = await ingestTraces(dash, dir);
    expect(second.files).toBe(0);
    expect(second.records).toBe(0);
    expect(dash.db.select().from(traces).all()).toHaveLength(3);
  });

  // F2: a file that grew since its watermark (a new appended record) is re-read;
  // the new record lands while the existing rows dedupe via onConflict.
  it('re-ingests a file that was appended to after its watermark', async () => {
    cpSync(FIXTURE_TRACES, dir, { recursive: true });
    await ingestTraces(dash, dir);
    expect(dash.db.select().from(traces).all()).toHaveLength(3);

    const file = join(dir, '2026-07-01-99999.jsonl');
    appendFileSync(file, traceLine('trace-appended'));
    // Bump mtime forward so the (mtime, size) watermark definitively mismatches.
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);

    const res = await ingestTraces(dash, dir);
    // The changed file is read again (files === 1); the 3 originals re-process
    // idempotently and the appended record is added → 4 rows total.
    expect(res.files).toBe(1);
    expect(dash.db.select().from(traces).where(eq(traces.traceId, 'trace-appended')).all()).toHaveLength(1);
    expect(dash.db.select().from(traces).all()).toHaveLength(4);
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

  // Finding 6: a `null` JSONL line is valid JSON but not an object. Field access
  // on it throws; that throw must be contained per-line, not abort the whole pass
  // (which would skip every later valid record and the reconciliation forever).
  it('skips a null line and still ingests a following valid trace', async () => {
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, '2026-07-01-nullline.jsonl'), `null\n${traceLine('trace-after-null')}`);

    const res = await ingestTraces(dash, dir);
    expect(res.files).toBe(1);
    expect(res.records).toBe(1);

    const row = dash.db.select().from(traces).where(eq(traces.traceId, 'trace-after-null')).all()[0];
    expect(row).toBeDefined();
    expect(row.traceId).toBe('trace-after-null');
  });

  // Finding 6 (precision): other non-object scalars/arrays are also skipped, and a
  // valid record on the same line-loop still lands — one bad line never wins.
  it('skips scalar and array lines without aborting the file', async () => {
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, '2026-07-01-mixed.jsonl'),
      `42\n"a string"\n[1,2,3]\n{"not":"a trace"}\n${traceLine('trace-survivor')}`,
    );

    const res = await ingestTraces(dash, dir);
    expect(res.files).toBe(1);
    expect(res.records).toBe(1);
    expect(dash.db.select().from(traces).where(eq(traces.traceId, 'trace-survivor')).all()).toHaveLength(1);
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
