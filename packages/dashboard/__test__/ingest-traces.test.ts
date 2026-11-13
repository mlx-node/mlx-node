import {
  appendFileSync,
  chmodSync,
  cpSync,
  mkdirSync,
  mkdtempSync,
  existsSync,
  renameSync,
  rmSync,
  symlinkSync,
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
  it('an index from the previous version is rebuilt; old rows and rows with the native timers both load', async () => {
    mkdirSync(dir, { recursive: true });
    const row = (traceId: string, extra: Record<string, unknown>) =>
      JSON.stringify({
        v: 2,
        traceId,
        ts: Date.now() - 60_000,
        pid: 7,
        model: 'Ornith-35B',
        outcome: 'ok',
        endpoint: 'messages',
        stream: true,
        dispatch: 'start',
        finishReason: 'stop',
        durationMs: 77_700,
        toolCount: 0,
        imageCount: 0,
        concurrency: 1,
        concurrencyPeak: 1,
        ...extra,
      });
    writeFileSync(
      join(dir, '2026-11-13-55843.jsonl'),
      [
        row('old', { ttftMs: 2_600 }),
        row('new', { ttftMs: 2_600, waitMs: 75_070, slotMs: 0, prepareMs: 2_100, pipelinesCreated: 0 }),
        row('cut', {
          outcome: 'cancelled',
          finishReason: 'cancelled',
          failPhase: 'prepare',
          waitMs: 1,
          slotMs: 0,
          prepareMs: 800,
          pipelinesCreated: 4,
          pipelineCreateMs: 9_000.5,
        }),
        row('bad', { outcome: 'cancelled', finishReason: 'cancelled', failPhase: 'warming' }),
      ].join('\n') + '\n',
    );
    const file = join(base, 'index.db');
    const first = openDashboardDb(file);
    await ingestTraces(first, dir);
    first.sqlite.exec('PRAGMA user_version = 11;');
    first.close();

    const rebuilt = openDashboardDb(file);
    try {
      expect(rebuilt.db.select().from(traces).all()).toHaveLength(0);
      await ingestTraces(rebuilt, dir);
      const rows = new Map(
        rebuilt.db
          .select()
          .from(traces)
          .all()
          .map((r) => [r.traceId, r]),
      );
      expect(rows.get('old')).toMatchObject({ waitMs: null, slotMs: null, prepareMs: null, pipelinesCreated: null });
      expect(rows.get('new')).toMatchObject({ waitMs: 75_070, slotMs: 0, prepareMs: 2_100, pipelinesCreated: 0 });
      expect(rows.get('cut')).toMatchObject({
        failPhase: 'prepare',
        prepareMs: 800,
        pipelinesCreated: 4,
        pipelineCreateMs: 9_000.5,
      });
      expect(rows.get('bad')?.failPhase).toBeNull();
    } finally {
      rebuilt.close();
    }
  });

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

  it('maps every v2 field', async () => {
    mkdirSync(dir, { recursive: true });
    const v2 = {
      v: 2,
      traceId: 'trace-v2',
      ts: 1782036402000,
      pid: 4242,
      sessionId: 'sess-1',
      model: 'claude-haiku-4-5',
      modelId: 'Qwen3.6-35B-A3B-4bit',
      modelType: 'qwen3_5_moe',
      outcome: 'error',
      errorKind: 'finish_error',
      failPhase: 'decode',
      endpoint: 'messages',
      stream: true,
      dispatch: 'continue_tool',
      finishReason: 'error',
      durationMs: 900.5,
      queueMs: 12.25,
      loadWaitMs: 30.5,
      resolveMs: 4000.75,
      preDispatchMs: 4100.5,
      promptTokens: 2000,
      cachedTokens: 1500,
      outputTokens: 64,
      reasoningTokens: 8,
      prefillDoneTokens: 500,
      ctxWindow: 262144,
      ttftMs: 210.5,
      prefillMs: 160.25,
      prefillTokens: 500,
      decodeTps: 71.5,
      mtpCycles: 30,
      mtpMeanAccepted: 2.1,
      mtpMeanDrafts: 1.1,
      mtpMeanDepth: 3,
      mtpAcceptByPos: [0.9, 0.55, 0.2],
      toolCount: 12,
      toolCallCount: 2,
      imageCount: 1,
      concurrency: 2,
      concurrencyPeak: 3,
      memActiveBytes: 20_000_000_000,
      memPeakBytes: 24_000_000_000,
    };
    writeFileSync(join(dir, '2026-07-20-4242.jsonl'), `${JSON.stringify(v2)}\n`);
    await ingestTraces(dash, dir);
    const row = dash.db.select().from(traces).where(eq(traces.traceId, 'trace-v2')).all()[0];
    expect(row).toMatchObject({
      sessionId: 'sess-1',
      ts: 1782036402000,
      pid: 4242,
      model: 'claude-haiku-4-5',
      modelId: 'Qwen3.6-35B-A3B-4bit',
      modelType: 'qwen3_5_moe',
      modelKey: 'Qwen3.6-35B-A3B-4bit',
      outcome: 'error',
      errorKind: 'finish_error',
      failPhase: 'decode',
      endpoint: 'messages',
      stream: 1,
      dispatch: 'continue_tool',
      finishReason: 'error',
      durationMs: 900.5,
      queueMs: 12.25,
      loadWaitMs: 30.5,
      resolveMs: 4000.75,
      preDispatchMs: 4100.5,
      promptTokens: 2000,
      cachedTokens: 1500,
      outputTokens: 64,
      reasoningTokens: 8,
      prefillDoneTokens: 500,
      ctxWindow: 262144,
      ttftMs: 210.5,
      prefillMs: 160.25,
      prefillTokens: 500,
      decodeTps: 71.5,
      mtpCycles: 30,
      mtpMeanAccepted: 2.1,
      mtpMeanDrafts: 1.1,
      mtpMeanDepth: 3,
      mtpAcceptByPos: '[0.9,0.55,0.2]',
      toolCount: 12,
      toolCallCount: 2,
      imageCount: 1,
      concurrency: 2,
      concurrencyPeak: 3,
      memActiveBytes: 20_000_000_000,
      memPeakBytes: 24_000_000_000,
      sourceFile: '2026-07-20-4242.jsonl',
    });
  });

  it('fills the prefill window only from a valid measured pair', async () => {
    mkdirSync(dir, { recursive: true });
    const line = (traceId: string, extra: Record<string, unknown>) =>
      JSON.stringify({ v: 2, traceId, ts: 1782036402000, model: 'm', outcome: 'ok', ttftMs: 500, ...extra });
    writeFileSync(
      join(dir, '2026-07-20-2.jsonl'),
      [
        line('measured', { prefillMs: 250, prefillTokens: 400 }),
        line('legacy', { prefillTps: 900 }),
        line('no-tokens', { prefillMs: 250 }),
        line('no-ms', { prefillTokens: 400 }),
        line('zero-ms', { prefillMs: 0, prefillTokens: 400 }),
        line('zero-tokens', { prefillMs: 250, prefillTokens: 0 }),
      ].join('\n') + '\n',
    );
    await ingestTraces(dash, dir);
    const pairs = new Map(
      dash.db
        .select()
        .from(traces)
        .all()
        .map((r) => [r.traceId, [r.prefillMs, r.prefillTokens]]),
    );
    expect(pairs.get('measured')).toEqual([250, 400]);
    for (const id of ['legacy', 'no-tokens', 'no-ms', 'zero-ms', 'zero-tokens']) {
      expect(pairs.get(id)).toEqual([null, null]);
    }
  });

  it('drops unknown enum values and an oversized mtpAcceptByPos', async () => {
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, '2026-07-20-1.jsonl'),
      `${JSON.stringify({
        v: 2,
        traceId: 'trace-odd',
        ts: 1782036402000,
        model: 'm',
        outcome: 'ok',
        endpoint: 'chat',
        dispatch: 'resume',
        failPhase: 'later',
        mtpAcceptByPos: Array.from({ length: 17 }, () => 0.5),
      })}\n`,
    );
    await ingestTraces(dash, dir);
    const row = dash.db.select().from(traces).where(eq(traces.traceId, 'trace-odd')).all()[0];
    expect(row.endpoint).toBeNull();
    expect(row.dispatch).toBeNull();
    expect(row.failPhase).toBeNull();
    expect(row.mtpAcceptByPos).toBeNull();
  });

  it('treats a line without outcome as ok', async () => {
    cpSync(FIXTURE_TRACES, dir, { recursive: true });
    await ingestTraces(dash, dir);
    const outcomes = dash.db.select({ outcome: traces.outcome }).from(traces).all();
    expect(outcomes.map((r) => r.outcome)).toEqual(['ok', 'ok', 'ok']);
  });

  it('stores model_key from modelId, then model', async () => {
    mkdirSync(dir, { recursive: true });
    const base = { v: 2, ts: 1782036402000, outcome: 'ok', endpoint: 'other', dispatch: 'start' };
    writeFileSync(
      join(dir, '2026-07-20-2.jsonl'),
      [
        { ...base, traceId: 'with-id', model: 'alias', modelId: 'Checkpoint-A' },
        { ...base, traceId: 'empty-id', model: 'alias', modelId: '' },
        { ...base, traceId: 'model-only', model: 'plain' },
        { ...base, traceId: 'nothing' },
      ]
        .map((rec) => `${JSON.stringify(rec)}\n`)
        .join(''),
    );
    await ingestTraces(dash, dir);
    const keyOf = (id: string): string => dash.db.select().from(traces).where(eq(traces.traceId, id)).all()[0].modelKey;
    expect(keyOf('with-id')).toBe('Checkpoint-A');
    expect(keyOf('empty-id')).toBe('alias');
    expect(keyOf('model-only')).toBe('plain');
    expect(keyOf('nothing')).toBe('unknown');
  });

  /**
   * The ingest mapping is written one line per column, so a field the agent
   * emits and this list forgets is dropped in silence — which is what happened
   * to the `coldSidecar*` counters. `coldSidecarInstalled` is the one that cannot be re-derived from
   * anything else in the row: every `install_*_cold_sidecar` early-return falls
   * through to a full O(prefix) replay that produces CORRECT state, so a
   * regression from "restored and used" to "restored and silently re-derived"
   * leaves `cachedTokens`, `coldHits` and `coldCorruptions` untouched.
   *
   * Every value is DISTINCT, and the two object-scoped supersets
   * (`coldEnqueued` / `coldQueueDrops`) carry values far from their sidecar
   * subsets, so a column wired to its neighbour — or to the block counter that
   * shares its name — cannot round-trip.
   */
  it('stores every cold sidecar counter', async () => {
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, '2026-07-27-sidecar.jsonl'),
      `${JSON.stringify({
        v: 1,
        traceId: 'trace-sidecar',
        ts: 1782036502000,
        model: 'qwen3_5',
        durationMs: 100,
        finishReason: 'stop',
        promptTokens: 1,
        cachedTokens: 0,
        outputTokens: 1,
        reasoningTokens: 0,
        // Object-scoped: blocks AND sidecars share this queue, so these are the
        // SUPERSETS of the two sidecar counters below, never their peers.
        coldEnqueued: 90,
        coldQueueDrops: 91,
        coldSidecarCaptureReached: 11,
        coldSidecarChainEmpty: 12,
        coldSidecarBoundarySkips: 13,
        coldSidecarAlreadyPersisted: 14,
        coldSidecarEnqueued: 15,
        coldSidecarQueueDrops: 16,
        coldSidecarInstalled: 17,
      })}\n`,
    );
    await ingestTraces(dash, dir);
    const row = dash.db.select().from(traces).where(eq(traces.traceId, 'trace-sidecar')).all()[0];
    expect(row.coldEnqueued).toBe(90);
    expect(row.coldQueueDrops).toBe(91);
    expect(row.coldSidecarCaptureReached).toBe(11);
    expect(row.coldSidecarChainEmpty).toBe(12);
    expect(row.coldSidecarBoundarySkips).toBe(13);
    expect(row.coldSidecarAlreadyPersisted).toBe(14);
    expect(row.coldSidecarEnqueued).toBe(15);
    expect(row.coldSidecarQueueDrops).toBe(16);
    expect(row.coldSidecarInstalled).toBe(17);
  });

  // A record from a build that predates the sidecar counters must ingest them as
  // NULL, not 0: `MAX(...) = 0` and "never recorded" are different answers, and
  // only NULL keeps them apart.
  it('leaves the cold sidecar counters NULL when a trace record omits them', async () => {
    cpSync(FIXTURE_TRACES, dir, { recursive: true });
    await ingestTraces(dash, dir);
    const row = dash.db.select().from(traces).where(eq(traces.traceId, 'trace-bbb')).all()[0];
    expect(row.coldSidecarCaptureReached).toBeNull();
    expect(row.coldSidecarInstalled).toBeNull();
  });

  // Older JSONL without the newer fields ingests them as NULL, not 0.
  it('leaves queueMs and the v2 fields NULL when a trace record omits them', async () => {
    cpSync(FIXTURE_TRACES, dir, { recursive: true });
    await ingestTraces(dash, dir);
    const row = dash.db.select().from(traces).where(eq(traces.traceId, 'trace-bbb')).all()[0];
    expect(row.queueMs).toBeNull();
    expect(row.ctxWindow).toBeNull();
    expect(row.endpoint).toBeNull();
    expect(row.stream).toBeNull();
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

  it('appends only the new tail when a file grows', async () => {
    cpSync(FIXTURE_TRACES, dir, { recursive: true });
    await ingestTraces(dash, dir);
    expect(dash.db.select().from(traces).all()).toHaveLength(3);

    const file = join(dir, '2026-07-01-99999.jsonl');
    appendFileSync(file, traceLine('trace-appended'));
    // Bump mtime forward so the (mtime, size) watermark definitively mismatches.
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);

    const res = await ingestTraces(dash, dir);
    expect(res.files).toBe(1);
    expect(res.records).toBe(1);
    expect(dash.db.select().from(traces).where(eq(traces.traceId, 'trace-appended')).all()).toHaveLength(1);
    expect(dash.db.select().from(traces).all()).toHaveLength(4);
    const mark = dash.db.select().from(traceFiles).all()[0];
    expect(mark.consumed).toBe(mark.lastIngestedSize);
  });

  it('falls back to replace when the bytes before the offset changed', async () => {
    mkdirSync(dir, { recursive: true });
    const file = join(dir, '2026-07-05-rewritten.jsonl');
    writeFileSync(file, `${traceLine('trace-A')}${traceLine('trace-B')}`);
    await ingestTraces(dash, dir);

    // Grown, but the bytes just before the old offset differ (the id is the last
    // key): an append from the old offset would keep A and B.
    const idLast = (traceId: string): string =>
      `${JSON.stringify({ v: 1, ts: 1782036002000, model: 'qwen3_5', durationMs: 100, finishReason: 'stop', promptTokens: 100, outputTokens: 20, traceId })}\n`;
    const rewritten = `${idLast('trace-X')}${idLast('trace-Y')}${idLast('trace-Z')}`;
    expect(Buffer.byteLength(rewritten)).toBeGreaterThan(Buffer.byteLength(traceLine('trace-A')) * 2);
    writeFileSync(file, rewritten);
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);
    const res = await ingestTraces(dash, dir);
    expect(res.records).toBe(3);
    const ids = dash.db
      .select({ id: traces.traceId })
      .from(traces)
      .all()
      .map((r) => r.id)
      .sort();
    expect(ids).toEqual(['trace-X', 'trace-Y', 'trace-Z']);
  });

  it('leaves a partial last line for the next pass', async () => {
    mkdirSync(dir, { recursive: true });
    const file = join(dir, '2026-07-05-torn.jsonl');
    const whole = traceLine('trace-torn');
    writeFileSync(file, `${traceLine('trace-first')}${whole.slice(0, 20)}`);
    const first = await ingestTraces(dash, dir);
    expect(first.records).toBe(1);
    expect(dash.db.select().from(traceFiles).all()[0].consumed).toBe(Buffer.byteLength(traceLine('trace-first')));

    appendFileSync(file, whole.slice(20));
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);
    const second = await ingestTraces(dash, dir);
    expect(second.records).toBe(1);
    expect(dash.db.select().from(traces).where(eq(traces.traceId, 'trace-torn')).all()).toHaveLength(1);
    expect(dash.db.select().from(traces).all()).toHaveLength(2);
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

  // G3: a changed file is the authoritative snapshot of its own rows. When its
  // records are REPLACED ([A,B] → [C]), re-ingest must leave C only — insert-only
  // ingest would retain the stale A and B.
  it('replaces a changed file rows from the current snapshot ([A,B] → [C])', async () => {
    mkdirSync(dir, { recursive: true });
    const file = join(dir, '2026-07-05-replace.jsonl');
    writeFileSync(file, `${traceLine('trace-A')}${traceLine('trace-B')}`);
    await ingestTraces(dash, dir);
    expect(dash.db.select().from(traces).all()).toHaveLength(2);

    // Rewrite the file down to a single, different record; bump mtime forward.
    writeFileSync(file, traceLine('trace-C'));
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);

    await ingestTraces(dash, dir);
    const rows = dash.db.select().from(traces).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].traceId).toBe('trace-C');
    expect(dash.db.select().from(traces).where(eq(traces.traceId, 'trace-A')).all()).toHaveLength(0);
    expect(dash.db.select().from(traces).where(eq(traces.traceId, 'trace-B')).all()).toHaveLength(0);
  });

  // G3: rewriting a record in place (a corrected field, same trace_id) must be
  // reflected — the insert-only + onConflictDoNothing path keeps the OLD value.
  it('reflects a corrected field when a record is rewritten in place', async () => {
    mkdirSync(dir, { recursive: true });
    const file = join(dir, '2026-07-05-rewrite.jsonl');
    writeFileSync(
      file,
      `${JSON.stringify({ v: 1, traceId: 'trace-fix', ts: 1782036002000, model: 'qwen3_5', durationMs: 100 })}\n`,
    );
    await ingestTraces(dash, dir);
    expect(dash.db.select().from(traces).where(eq(traces.traceId, 'trace-fix')).all()[0].durationMs).toBe(100);

    // Rewrite the SAME trace_id with a corrected durationMs; bump mtime.
    writeFileSync(
      file,
      `${JSON.stringify({ v: 1, traceId: 'trace-fix', ts: 1782036002000, model: 'qwen3_5', durationMs: 777 })}\n`,
    );
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);

    await ingestTraces(dash, dir);
    expect(dash.db.select().from(traces).where(eq(traces.traceId, 'trace-fix')).all()[0].durationMs).toBe(777);
  });

  it('replaces a two-record file rewritten at the same size that keeps its last record', async () => {
    mkdirSync(dir, { recursive: true });
    const file = join(dir, '2026-07-05-samesize.jsonl');
    const first = (durationMs: number): string =>
      `${JSON.stringify({ v: 1, traceId: 'trace-first', ts: 1782036002000, model: 'qwen3_5', durationMs })}\n`;
    writeFileSync(file, `${first(100)}${traceLine('trace-last')}`);
    await ingestTraces(dash, dir);
    expect(dash.db.select().from(traces).where(eq(traces.traceId, 'trace-first')).all()[0].durationMs).toBe(100);

    writeFileSync(file, `${first(777)}${traceLine('trace-last')}`);
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);

    await ingestTraces(dash, dir);
    expect(dash.db.select().from(traces).where(eq(traces.traceId, 'trace-first')).all()[0].durationMs).toBe(777);
    expect(dash.db.select().from(traces).all()).toHaveLength(2);
  });

  it('replaces a file that shrank into its torn last line', async () => {
    mkdirSync(dir, { recursive: true });
    const file = join(dir, '2026-07-05-shrunk.jsonl');
    const torn = traceLine('trace-torn');
    writeFileSync(file, `${traceLine('trace-A')}${torn.slice(0, 40)}`);
    await ingestTraces(dash, dir);
    expect(dash.db.select().from(traces).all()).toHaveLength(1);

    writeFileSync(file, `${traceLine('trace-B')}${torn.slice(0, 20)}`);
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);

    await ingestTraces(dash, dir);
    const ids = dash.db
      .select({ id: traces.traceId })
      .from(traces)
      .all()
      .map((r) => r.id);
    expect(ids).toEqual(['trace-B']);
  });

  // G3: an APPEND ([A] → [A,B]) still ends with both records after the
  // delete-then-reinsert snapshot replace (no rows are lost).
  it('keeps both records when a file is appended ([A] → [A,B])', async () => {
    mkdirSync(dir, { recursive: true });
    const file = join(dir, '2026-07-05-append.jsonl');
    writeFileSync(file, traceLine('trace-A'));
    await ingestTraces(dash, dir);
    expect(dash.db.select().from(traces).all()).toHaveLength(1);

    appendFileSync(file, traceLine('trace-B'));
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);

    await ingestTraces(dash, dir);
    expect(dash.db.select().from(traces).all()).toHaveLength(2);
    expect(dash.db.select().from(traces).where(eq(traces.traceId, 'trace-A')).all()).toHaveLength(1);
    expect(dash.db.select().from(traces).where(eq(traces.traceId, 'trace-B')).all()).toHaveLength(1);
  });

  // F2: the 30-day prune must never follow a SYMLINKED trace root and delete
  // external files in the link target. `statSync`/`unlinkSync` follow symlinks, so
  // a relocated-metrics symlink pointing at an external dir would let the prune age
  // out and delete real external *.jsonl. The no-follow root guard skips the whole
  // pass (prunes nothing, ingests nothing) rather than reaching through the link.
  it('never prunes through a symlinked trace root (no external deletion)', async () => {
    // An external dir OUTSIDE any managed root, holding a victim old enough to prune.
    const external = join(base, 'external-target');
    mkdirSync(external, { recursive: true });
    const victim = join(external, '2026-01-01-victim.jsonl');
    writeFileSync(victim, traceLine('trace-victim'));
    const oldSec = (Date.now() - 60 * DAY_MS) / 1000;
    utimesSync(victim, oldSec, oldSec);

    // The trace ROOT is a symlink to that external dir (a user relocating metrics
    // onto another volume via `ln -s`).
    const linkRoot = join(base, 'traces-link');
    symlinkSync(external, linkRoot);

    const res = await ingestTraces(dash, linkRoot, { retentionDays: 30 });

    // The external victim survives untouched and nothing was pruned/ingested.
    expect(existsSync(victim)).toBe(true);
    expect(res.pruned).toBe(0);
    expect(res.files).toBe(0);
    expect(res.records).toBe(0);
  });

  // Finding #7: a trace_id is globally UNIQUE (independent of source_file). When a
  // trace file is renamed (a.jsonl → b.jsonl), ingesting b cannot insert the trace
  // while a.jsonl still owns the unique row, so the vanished-file reconciliation of
  // a.jsonl must run BEFORE the ingest loop — otherwise a's row is deleted after b
  // was skipped by onConflict, and the trace is lost from every row forever.
  it('re-indexes a trace whose file was renamed (moved), not dropping its row', async () => {
    mkdirSync(dir, { recursive: true });
    const aFile = join(dir, 'a.jsonl');
    writeFileSync(aFile, traceLine('trace-moved'));
    await ingestTraces(dash, dir);
    const before = dash.db.select().from(traces).where(eq(traces.traceId, 'trace-moved')).all();
    expect(before).toHaveLength(1);
    expect(before[0].sourceFile).toBe('a.jsonl');

    // Rename the file on disk; the trace_id inside is unchanged.
    renameSync(aFile, join(dir, 'b.jsonl'));

    await ingestTraces(dash, dir);
    // The moved trace is still indexed — now owned by b.jsonl, not dropped.
    const after = dash.db.select().from(traces).where(eq(traces.traceId, 'trace-moved')).all();
    expect(after).toHaveLength(1);
    expect(after[0].sourceFile).toBe('b.jsonl');
    // No orphaned row left behind under the old name.
    expect(dash.db.select().from(traces).where(eq(traces.sourceFile, 'a.jsonl')).all()).toHaveLength(0);
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
    expect(res.warnings).toEqual([]);
  });

  // chmod 000 is a no-op for root, so the unreadable dir would still list — skip.
  const canTestUnreadable = (process.getuid?.() ?? 0) !== 0;
  (canTestUnreadable ? it : it.skip)('skips the pass when the trace root is unreadable, keeping rows', async () => {
    // A root that exists but cannot be listed (chmod/ACL drift, an external volume
    // returning EACCES, fd exhaustion) is UNKNOWN, not empty. Throwing would abort
    // the whole ingest pass; listing it as empty would let the reconciliation above
    // delete every tracked source file. Warn, skip, touch nothing.
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'a.jsonl'), traceLine('trace-keep'));
    await ingestTraces(dash, dir);
    expect(dash.db.select().from(traces).all()).toHaveLength(1);
    expect(dash.db.select().from(traceFiles).all()).toHaveLength(1);

    chmodSync(dir, 0o000);
    try {
      const res = await ingestTraces(dash, dir, { retentionDays: 30 });
      expect(res.files).toBe(0);
      expect(res.records).toBe(0);
      expect(res.pruned).toBe(0);
      expect(res.warnings.some((w) => w.includes('scan skipped'))).toBe(true);
      expect(dash.db.select().from(traces).all()).toHaveLength(1);
      expect(dash.db.select().from(traceFiles).all()).toHaveLength(1);
    } finally {
      chmodSync(dir, 0o755); // restore so afterEach cleanup can remove it
    }
  });

  // The live-source set decides which tracked rows survive reconciliation, so it
  // must be decided from what an entry IS, not from what it is named. A DIRECTORY
  // named `*.jsonl` judged by suffix alone reads as live: its rows and watermark are
  // kept while the ingest loop fails EISDIR and continues, and because the metrics
  // queries are unwindowed by default that stale telemetry keeps counting in the
  // all-time aggregates for as long as the directory sits there.
  it('reconciles rows when a trace file is replaced by a directory of the same name', async () => {
    mkdirSync(dir, { recursive: true });
    const file = join(dir, 'a.jsonl');
    writeFileSync(file, traceLine('trace-A'));
    await ingestTraces(dash, dir, { retentionDays: 30 });
    expect(dash.db.select().from(traces).all()).toHaveLength(1);
    expect(dash.db.select().from(traceFiles).all()).toHaveLength(1);

    unlinkSync(file);
    mkdirSync(file); // a DIRECTORY now shadows the indexed name

    const res = await ingestTraces(dash, dir, { retentionDays: 30 });
    expect(res.files).toBe(0);
    expect(res.pruned).toBe(0);
    expect(dash.db.select().from(traces).all()).toHaveLength(0);
    expect(dash.db.select().from(traceFiles).all()).toHaveLength(0);
  });

  // The kind check FOLLOWS symlinks on purpose: a
  // symlinked trace entry ingests today and must keep ingesting: an `lstat().isFile()`
  // check is false for a symlink, so a no-follow fix would silently stop reading it
  // and reconcile its rows away.
  it('ingests a trace entry that is a symlink to a real jsonl file', async () => {
    const external = join(base, 'external');
    mkdirSync(external, { recursive: true });
    const target = join(external, 'real.jsonl');
    writeFileSync(target, traceLine('trace-sym'));

    mkdirSync(dir, { recursive: true });
    symlinkSync(target, join(dir, 'b.jsonl'));

    const res = await ingestTraces(dash, dir, { retentionDays: 30 });
    expect(res.files).toBe(1);
    expect(res.records).toBe(1);
    const rows = dash.db.select().from(traces).where(eq(traces.traceId, 'trace-sym')).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].sourceFile).toBe('b.jsonl');
    expect(dash.db.select().from(traceFiles).all()).toHaveLength(1);
  });

  // Over-correction guard: excluding non-regular entries must not cost the regular
  // ones sharing the root. Green before and after the kind check by design.
  it('keeps ingesting a regular trace file beside a *.jsonl directory', async () => {
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'good.jsonl'), traceLine('trace-good'));
    mkdirSync(join(dir, 'bogus.jsonl'));

    const res = await ingestTraces(dash, dir, { retentionDays: 30 });
    expect(res.files).toBe(1);
    expect(res.records).toBe(1);
    const rows = dash.db.select().from(traces).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].sourceFile).toBe('good.jsonl');
    expect(
      dash.db
        .select()
        .from(traceFiles)
        .all()
        .map((w) => w.name),
    ).toEqual(['good.jsonl']);
  });

  // A stat FAILURE is not a deletion. A trace root that is readable but not
  // searchable (mode 0600) lists its entries and then gives EACCES on every child;
  // reading an unstattable entry as "not a regular file" would delete its rows and
  // watermark on a fault that says nothing at all about the file — the same
  // wipe-on-fault trap the unreadable-root guard above closes.
  (canTestUnreadable ? it : it.skip)('keeps rows when a trace entry cannot be stat-ed', async () => {
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'a.jsonl'), traceLine('trace-keep'));
    await ingestTraces(dash, dir);
    expect(dash.db.select().from(traces).all()).toHaveLength(1);
    expect(dash.db.select().from(traceFiles).all()).toHaveLength(1);

    chmodSync(dir, 0o600); // readable, NOT searchable: readdir lists, child stat EACCES
    try {
      const res = await ingestTraces(dash, dir, { retentionDays: 30 });
      expect(res.files).toBe(0);
      expect(res.pruned).toBe(0);
      expect(dash.db.select().from(traces).all()).toHaveLength(1);
      expect(dash.db.select().from(traceFiles).all()).toHaveLength(1);
    } finally {
      chmodSync(dir, 0o755); // restore so afterEach cleanup can remove it
    }
  });
});
