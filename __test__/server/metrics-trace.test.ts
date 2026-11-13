import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { MetricsTrace, type MetricsTraceRecord } from '../../packages/server/src/metrics-trace.js';

/** A fixed instant so `<YYYY-MM-DD>` is deterministic (UTC). */
const FIXED_NOW = Date.UTC(2026, 6, 21, 12, 34, 56); // 2026-07-21
const FIXED_DATE = '2026-07-21';

type RecordInput = Omit<MetricsTraceRecord, 'v' | 'pid'>;

function baseRecord(overrides: Partial<RecordInput> = {}): RecordInput {
  return {
    traceId: 'trace-1',
    ts: FIXED_NOW,
    sessionId: 'sess-a',
    model: 'qwen-small',
    outcome: 'ok',
    endpoint: 'messages',
    stream: true,
    dispatch: 'start',
    durationMs: 42,
    finishReason: 'stop',
    promptTokens: 100,
    cachedTokens: 60,
    outputTokens: 7,
    reasoningTokens: 0,
    toolCount: 0,
    imageCount: 0,
    concurrency: 1,
    concurrencyPeak: 1,
    ...overrides,
  };
}

function readLines(file: string): unknown[] {
  return readFileSync(file, 'utf8')
    .split('\n')
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line));
}

describe('MetricsTrace', () => {
  let dir: string;
  const savedEnv = process.env.MLX_AGENT_METRICS;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'mlx-metrics-'));
    delete process.env.MLX_AGENT_METRICS;
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
    if (savedEnv === undefined) delete process.env.MLX_AGENT_METRICS;
    else process.env.MLX_AGENT_METRICS = savedEnv;
  });

  it('is enabled by default and names the file <YYYY-MM-DD>-<pid>.jsonl', () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    expect(trace.enabled).toBe(true);
    expect(basename(trace.currentFile())).toBe(`${FIXED_DATE}-${process.pid}.jsonl`);
    expect(trace.currentFile()).toBe(join(dir, `${FIXED_DATE}-${process.pid}.jsonl`));
  });

  it('appends exactly one JSON line per record to the dated file', () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    trace.record(baseRecord({ traceId: 'a' }));
    trace.record(baseRecord({ traceId: 'b' }));

    const lines = readLines(trace.currentFile()) as MetricsTraceRecord[];
    expect(lines).toHaveLength(2);
    expect(lines.map((l) => l.traceId)).toEqual(['a', 'b']);
    expect(lines[0]).toMatchObject({
      v: 2,
      traceId: 'a',
      model: 'qwen-small',
      finishReason: 'stop',
      promptTokens: 100,
      cachedTokens: 60,
      outputTokens: 7,
      reasoningTokens: 0,
      durationMs: 42,
      sessionId: 'sess-a',
    });
  });

  it('writes a v2 record with every allowlisted field', () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    trace.record(
      baseRecord({
        modelId: 'Qwen3.6-35B-A3B',
        modelType: 'qwen3_5_moe',
        outcome: 'error',
        errorKind: 'exception',
        failPhase: 'decode',
        endpoint: 'responses',
        stream: false,
        dispatch: 'continue_tool',
        finishReason: 'error',
        queueMs: 3,
        loadWaitMs: 4,
        resolveMs: 5,
        preDispatchMs: 12,
        prefillDoneTokens: 40,
        ctxWindow: 262144,
        ttftMs: 125,
        prefillMs: 90,
        prefillTokens: 40,
        decodeTps: 47.25,
        mtpCycles: 3,
        mtpMeanAccepted: 1.6,
        mtpMeanDrafts: 0.6,
        mtpMeanDepth: 2,
        mtpAcceptByPos: [0.5, 0.25],
        toolCount: 4,
        toolCallCount: 2,
        imageCount: 1,
        concurrency: 2,
        concurrencyPeak: 3,
        memActiveBytes: 1000,
        memPeakBytes: 2000,
      }),
    );
    const [line] = readLines(trace.currentFile()) as Record<string, unknown>[];
    expect(line).toEqual({
      v: 2,
      traceId: 'trace-1',
      ts: FIXED_NOW,
      pid: process.pid,
      sessionId: 'sess-a',
      model: 'qwen-small',
      modelId: 'Qwen3.6-35B-A3B',
      modelType: 'qwen3_5_moe',
      outcome: 'error',
      errorKind: 'exception',
      failPhase: 'decode',
      endpoint: 'responses',
      stream: false,
      dispatch: 'continue_tool',
      finishReason: 'error',
      durationMs: 42,
      queueMs: 3,
      loadWaitMs: 4,
      resolveMs: 5,
      preDispatchMs: 12,
      promptTokens: 100,
      cachedTokens: 60,
      outputTokens: 7,
      reasoningTokens: 0,
      prefillDoneTokens: 40,
      ctxWindow: 262144,
      ttftMs: 125,
      decodeTps: 47.25,
      mtpCycles: 3,
      mtpMeanAccepted: 1.6,
      mtpMeanDrafts: 0.6,
      mtpMeanDepth: 2,
      mtpAcceptByPos: [0.5, 0.25],
      toolCount: 4,
      toolCallCount: 2,
      imageCount: 1,
      concurrency: 2,
      concurrencyPeak: 3,
      memActiveBytes: 1000,
      memPeakBytes: 2000,
    });
  });

  it('drops unknown enum values and unsafe finish reasons', () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    trace.record(baseRecord({ traceId: 'bad-outcome', outcome: 'maybe' as never }));
    trace.record(baseRecord({ traceId: 'bad-endpoint', endpoint: 'chat' as never }));
    trace.record(baseRecord({ traceId: 'bad-dispatch', dispatch: 'resume' as never }));
    trace.record(baseRecord({ traceId: 'unsafe', finishReason: 'Stop: user said "secret"' }));
    trace.record(baseRecord({ traceId: 'long', finishReason: 'a'.repeat(33) }));
    trace.record(
      baseRecord({ traceId: 'ok-kind', errorKind: 'exception', failPhase: 'decode', finishReason: 'tool_calls' }),
    );
    trace.record(
      baseRecord({ traceId: 'bad-kind', outcome: 'error', errorKind: 'oom' as never, failPhase: 'x' as never }),
    );

    const lines = readLines(trace.currentFile()) as MetricsTraceRecord[];
    expect(lines.map((l) => l.traceId)).toEqual(['unsafe', 'long', 'ok-kind', 'bad-kind']);
    expect(lines[0]!.finishReason).toBe('other');
    expect(lines[1]!.finishReason).toBe('other');
    // errorKind and failPhase belong to failed turns only.
    expect(lines[2]!.finishReason).toBe('tool_calls');
    expect('errorKind' in lines[2]!).toBe(false);
    expect('failPhase' in lines[2]!).toBe(false);
    expect('errorKind' in lines[3]!).toBe(false);
    expect('failPhase' in lines[3]!).toBe(false);
  });

  it('caps mtpAcceptByPos at 16 rounded rates', () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    const rates = [0.123456, -0.5, 1.5, Number.NaN, ...Array.from({ length: 20 }, () => 0.5)];
    trace.record(baseRecord({ mtpAcceptByPos: rates }));
    const [line] = readLines(trace.currentFile()) as MetricsTraceRecord[];
    expect(line!.mtpAcceptByPos).toHaveLength(16);
    expect(line!.mtpAcceptByPos!.slice(0, 4)).toEqual([0.1235, 0, 1, 0.5]);
  });

  it('never writes rootSessionId, rootSessionFile or resident', () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    trace.record({
      ...baseRecord(),
      rootSessionId: 'root-7',
      rootSessionFile: '/sessions/root-7.jsonl',
      resident: true,
    } as unknown as RecordInput);
    const [line] = readLines(trace.currentFile()) as Record<string, unknown>[];
    expect('rootSessionId' in line).toBe(false);
    expect('rootSessionFile' in line).toBe(false);
    expect('resident' in line).toBe(false);
  });

  it('re-creates the directory and retries once when the append fails', () => {
    const traceDir = join(dir, 'traces');
    const trace = new MetricsTrace({ dir: traceDir, now: () => FIXED_NOW });
    trace.record(baseRecord({ traceId: 'first' }));
    rmSync(traceDir, { recursive: true, force: true });
    expect(existsSync(traceDir)).toBe(false);
    trace.record(baseRecord({ traceId: 'second' }));
    const lines = readLines(trace.currentFile()) as MetricsTraceRecord[];
    expect(lines.map((l) => l.traceId)).toEqual(['second']);
  });

  it('does not write a zero decodeTps', () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    trace.record(baseRecord({ decodeTps: 0, ttftMs: 0 }));
    const [line] = readLines(trace.currentFile()) as Record<string, unknown>[];
    expect('decodeTps' in line).toBe(false);
    expect(line.ttftMs).toBe(0);
  });

  it('records optional throughput / mtp / cold-tier fields when present', () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    trace.record(
      baseRecord({
        ttftMs: 125,
        prefillMs: 90.5,
        prefillTokens: 40,
        decodeTps: 47.25,
        mtpCycles: 3,
        mtpMeanAccepted: 1.6,
        coldHits: 2,
        coldMisses: 1,
        coldBytesWritten: 4096,
        coldBytesRestored: 8192,
      }),
    );
    const [line] = readLines(trace.currentFile()) as MetricsTraceRecord[];
    expect(line).toMatchObject({
      ttftMs: 125,
      prefillMs: 90.5,
      prefillTokens: 40,
      decodeTps: 47.25,
      mtpCycles: 3,
      mtpMeanAccepted: 1.6,
      coldHits: 2,
      coldMisses: 1,
      coldBytesWritten: 4096,
      coldBytesRestored: 8192,
    });
  });

  it('writes the three native timers and the pipeline pair, and drops invalid values', () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    trace.record(
      baseRecord({
        traceId: 'timed',
        waitMs: 75_070.25,
        slotMs: 0,
        prepareMs: 4_210.5,
        pipelinesCreated: 3,
        pipelineCreateMs: 812.5,
      }),
    );
    trace.record(
      baseRecord({
        traceId: 'cancelled-in-prepare',
        outcome: 'cancelled',
        failPhase: 'prepare',
        waitMs: 1,
        slotMs: 0,
        prepareMs: 800,
      }),
    );
    trace.record(baseRecord({ traceId: 'cancelled-in-slot', outcome: 'cancelled', failPhase: 'slot' }));
    trace.record(
      baseRecord({
        traceId: 'invalid',
        waitMs: -1,
        slotMs: Number.NaN,
        prepareMs: Number.POSITIVE_INFINITY,
        pipelinesCreated: 1.5,
        pipelineCreateMs: 10,
      }),
    );
    trace.record(baseRecord({ traceId: 'count-only', pipelinesCreated: 2 }));
    const [timed, prepare, slot, invalid, countOnly] = readLines(trace.currentFile()) as Record<string, unknown>[];
    expect(timed).toMatchObject({
      waitMs: 75_070.25,
      slotMs: 0,
      prepareMs: 4_210.5,
      pipelinesCreated: 3,
      pipelineCreateMs: 812.5,
    });
    expect(prepare).toMatchObject({ failPhase: 'prepare', waitMs: 1, slotMs: 0, prepareMs: 800 });
    expect(slot).toMatchObject({ failPhase: 'slot' });
    for (const key of ['waitMs', 'slotMs', 'prepareMs', 'pipelinesCreated', 'pipelineCreateMs']) {
      expect(key in invalid!).toBe(false);
    }
    expect('pipelinesCreated' in countOnly!).toBe(false);
  });

  it('writes the prefill window only as a valid pair on an ok turn', () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    const cases: Array<[string, Partial<RecordInput>]> = [
      ['valid', { prefillMs: 250, prefillTokens: 400 }],
      ['zero-ms', { prefillMs: 0, prefillTokens: 400 }],
      ['zero-tokens', { prefillMs: 250, prefillTokens: 0 }],
      ['negative', { prefillMs: -1, prefillTokens: 400 }],
      ['nan', { prefillMs: Number.NaN, prefillTokens: 400 }],
      ['infinite', { prefillMs: Number.POSITIVE_INFINITY, prefillTokens: 400 }],
      ['fractional', { prefillMs: 250, prefillTokens: 1.5 }],
      ['no-tokens', { prefillMs: 250 }],
      ['no-ms', { prefillTokens: 400 }],
      ['error', { outcome: 'error', errorKind: 'exception', prefillMs: 250, prefillTokens: 400 }],
      ['cancelled', { outcome: 'cancelled', prefillMs: 250, prefillTokens: 400 }],
    ];
    for (const [traceId, fields] of cases) trace.record(baseRecord({ traceId, ...fields }));
    trace.record({ ...baseRecord({ traceId: 'legacy' }), prefillTps: 900 } as unknown as RecordInput);
    const lines = readLines(trace.currentFile()) as Record<string, unknown>[];
    expect(lines.map((l) => l.traceId)).toEqual([...cases.map(([id]) => id), 'legacy']);
    expect(lines[0]).toMatchObject({ prefillMs: 250, prefillTokens: 400 });
    for (const line of lines.slice(1)) {
      expect('prefillMs' in line).toBe(false);
      expect('prefillTokens' in line).toBe(false);
    }
    for (const line of lines) expect('prefillTps' in line).toBe(false);
  });

  /**
   * `record()` rebuilds the output field by field from an allowlist, so a
   * counter added to the interface and to the delta computation but NOT to the
   * rebuild is silently discarded — a green typecheck and an empty column. Four
   * counters `coldCacheStats()` has always exposed were dropped this way before
   * ever reaching the allowlist.
   */
  it('writes the cold-tier counters and cache identity the allowlist used to drop', () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    trace.record(
      baseRecord({
        coldRoot: '/private/tmp/cache/mlx-paged-v1',
        coldEnabled: true,
        coldEnqueued: 12,
        coldQueueDrops: 3,
        coldEvictions: 5,
        coldCorruptions: 1,
        coldCorruptionsTotal: 9,
        coldQueueDropsTotal: 7,
        // The one cumulative total NOT carried by `COLD_DELTA_FIELDS`: like
        // the two above it is spelled out by hand in the allowlist, which is
        // precisely the shape that gets forgotten. Without it an operator
        // whose cache root is refusing writes has no latch that survives the
        // errored turn the failure most likely lands in.
        coldWriteErrorsTotal: 11,
      }),
    );
    const [line] = readLines(trace.currentFile()) as MetricsTraceRecord[];
    expect(line).toMatchObject({
      coldRoot: '/private/tmp/cache/mlx-paged-v1',
      coldEnabled: true,
      coldEnqueued: 12,
      coldQueueDrops: 3,
      coldEvictions: 5,
      coldCorruptions: 1,
      coldCorruptionsTotal: 9,
      coldQueueDropsTotal: 7,
      coldWriteErrorsTotal: 11,
    });
  });

  // A cache identity is only meaningful for a tier that was open. An empty root
  // (what the native struct reports while disabled) would create a bucket no
  // dashboard root can ever match, so it must be omitted — while `coldEnabled:
  // false` IS recorded, because "the tier was off" and "we do not know" are
  // different answers the Cache page reports differently.
  it('omits an empty coldRoot but still records coldEnabled: false', () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    trace.record(baseRecord({ coldRoot: '', coldEnabled: false }));
    const [line] = readLines(trace.currentFile()) as Record<string, unknown>[];
    expect('coldRoot' in line).toBe(false);
    expect(line.coldEnabled).toBe(false);
  });

  it('omits absent optional fields rather than writing null', () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    trace.record(baseRecord({ sessionId: undefined }));
    const [line] = readLines(trace.currentFile()) as Record<string, unknown>[];
    expect('ttftMs' in line).toBe(false);
    expect('coldHits' in line).toBe(false);
    expect('sessionId' in line).toBe(false);
  });

  it('only writes allowlisted fields — free text passed by a caller never reaches disk', () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    // A caller (or a future refactor) may hand extra props; they must be dropped.
    trace.record({
      ...baseRecord(),
      text: 'the model said something secret',
      content: [{ type: 'text', text: 'do not persist me' }],
      prompt: 'raw user prompt',
    } as unknown as RecordInput);

    const [line] = readLines(trace.currentFile()) as Record<string, unknown>[];
    expect('text' in line).toBe(false);
    expect('content' in line).toBe(false);
    expect('prompt' in line).toBe(false);
    // The legitimate telemetry still landed.
    expect(line.traceId).toBe('trace-1');
  });

  it('is disabled by MLX_AGENT_METRICS=0/false/off and writes nothing', () => {
    for (const value of ['0', 'false', 'off', 'OFF', 'False']) {
      process.env.MLX_AGENT_METRICS = value;
      const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
      expect(trace.enabled, `value=${value}`).toBe(false);
      trace.record(baseRecord());
      expect(() => readFileSync(trace.currentFile(), 'utf8'), `value=${value}`).toThrow();
    }
  });

  it('stays enabled for other MLX_AGENT_METRICS values (e.g. "1")', () => {
    process.env.MLX_AGENT_METRICS = '1';
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    expect(trace.enabled).toBe(true);
  });

  it('swallows fs errors instead of throwing (dir is actually a file)', () => {
    const filePath = join(dir, 'not-a-dir');
    writeFileSync(filePath, 'i am a file');
    const trace = new MetricsTrace({ dir: filePath, now: () => FIXED_NOW });
    expect(() => trace.record(baseRecord())).not.toThrow();
  });
});
