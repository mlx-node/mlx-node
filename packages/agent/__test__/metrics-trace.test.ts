import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { MetricsTrace, type MetricsTraceRecord } from '../src/provider/metrics-trace.js';

/** A fixed instant so `<YYYY-MM-DD>` is deterministic (UTC). */
const FIXED_NOW = Date.UTC(2026, 6, 21, 12, 34, 56); // 2026-07-21
const FIXED_DATE = '2026-07-21';

type RecordInput = Omit<MetricsTraceRecord, 'v'>;

function baseRecord(overrides: Partial<RecordInput> = {}): RecordInput {
  return {
    traceId: 'trace-1',
    ts: FIXED_NOW,
    sessionId: 'sess-a',
    model: 'qwen-small',
    durationMs: 42,
    finishReason: 'stop',
    promptTokens: 100,
    cachedTokens: 60,
    outputTokens: 7,
    reasoningTokens: 0,
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
      v: 1,
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

  it('stamps the root session id/file supplied WITH the record (per-turn snapshot)', () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    trace.record(baseRecord({ rootSessionId: 'root-7', rootSessionFile: '/sessions/root-7.jsonl' }));

    const [line] = readLines(trace.currentFile()) as MetricsTraceRecord[];
    expect(line.rootSessionId).toBe('root-7');
    expect(line.rootSessionFile).toBe('/sessions/root-7.jsonl');
  });

  it('records optional throughput / mtp / cold-tier fields when present', () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    trace.record(
      baseRecord({
        ttftMs: 125,
        prefillTps: 812.5,
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
      prefillTps: 812.5,
      decodeTps: 47.25,
      mtpCycles: 3,
      mtpMeanAccepted: 1.6,
      coldHits: 2,
      coldMisses: 1,
      coldBytesWritten: 4096,
      coldBytesRestored: 8192,
    });
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
    expect('rootSessionId' in line).toBe(false);
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
